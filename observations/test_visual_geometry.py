"""Small full-topology geometry fixtures; no SAM3D or licensed MANO inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from observations.object.shape import write_mesh
from observations.object.visual_geometry import build_visual_geometry, can_reuse_native
from data.geometry import load_visual_geometry, bind_visual_to_candidate
from utils.geometry import apply_native_pose
from utils.state import sha


def native_fixture(root, image, mask, *, name='synthetic'):
    root.mkdir(parents=True,exist_ok=True)
    side=25
    xy=np.linspace(-.3,.3,side)
    xx,yy=np.meshgrid(xy,xy)
    raw=np.column_stack([xx.ravel(),yy.ravel(),np.ones(side*side)])
    faces=[]
    for y in range(side-1):
        for x in range(side-1):
            a=y*side+x;faces.extend([[a,a+1,a+side],[a+1,a+side+1,a+side]])
    faces=np.asarray(faces,np.int64)
    p3d,cv,contract=apply_native_pose(raw,np.array([1.,0,0,0]),np.zeros(3),np.ones(3))
    for filename,v in [('raw.ply',raw),('opencv.ply',cv),('p3d.ply',p3d)]:write_mesh(root/filename,v,faces)
    (root/'mesh.glb').write_bytes(b'synthetic metadata-only GLB; never rendered')
    moge={'checkpoint_sha256':'da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f','network_resolution_allowed':False}
    dino={'checkpoint_sha256':'36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51','hubconf_sha256':'c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64','network_resolution_allowed':False}
    qc={'status':'ok','name':name,'image':str(image),'mask':str(mask),'seed':42,
        'mesh':str(root/'raw.ply'),'glb':str(root/'mesh.glb'),'gaussian':None,
        'native_pose_mesh_opencv_camera':str(root/'opencv.ply'),
        'native_pose_mesh_pytorch3d_camera':str(root/'p3d.ply'),
        'pose':{'rotation':[1.,0,0,0],'translation':[0.,0,0],'scale':[1.,1.,1.]},
        'native_pose_contract':contract,'moge_offline':moge,'dinov2_offline':dino,
        'huggingface_offline':{'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','network_resolution_allowed':False}}
    path=root/'qc.json';path.write_text(json.dumps(qc));return path,cv,faces


def visual_fixture(root):
    image=root/'rgb.png';mask=root/'mask.png'
    cv2.imwrite(str(image),np.full((64,64,3),128,np.uint8))
    m=np.zeros((64,64),np.uint8);m[17:47,17:47]=255;cv2.imwrite(str(mask),m)
    qc,cv,faces=native_fixture(root/'native',image,mask)
    T=np.eye(4);T[:3,3]=[.1,-.2,.3]
    observed=cv*2.;world=observed@T[:3,:3].T+T[:3,3]
    geom={'frame_idx':7,'source_width':64,'source_height':64,'centroid_world_m':world.mean(0).tolist(),
        'world_vertices_sample_m':world.tolist(),'camera_vertices_sample_m':observed.tolist(),
        'rigid_pose_observation_eligible':True,'first_surface_depth_ownership':{'enabled':True,'retained_fraction':1.,'fail_closed':False}}
    frame={'frame_idx':7,'time_s':.2,'source_width':64,'source_height':64,
        'camera':{'T_world_camera_metric':T.tolist(),'intrinsics_fx_fy_cx_cy':[40.,40.,31.5,31.5]},
        'objects':[{'object_id':'synthetic','rigid_pose_observation_eligible':True,'mask_path':str(mask),'visible_geometry_candidate':geom}]}
    ann=root/'annotations.json';ann.write_text(json.dumps({'frames':[frame]}))
    row={'frame_idx':7,'time_s':.2,'status':'available','anchor_eligible':True,'source_size_wh':[64,64],
        'raw_rgb_path':str(image),'raw_rgb_sha256':sha(image),'appearance_mask_path':str(mask),'appearance_mask_sha256':sha(mask)}
    app=root/'appearance.json';app.write_text(json.dumps({'object_id':'synthetic','source_annotations':str(ann),
        'source_annotations_sha256':sha(ann),'metric_owned_masks_consumed':False,'frames':[row]}))
    dec=root/'decision.json';dec.write_text(json.dumps({'object_id':'synthetic','selected_frame_idx':7}))
    args=dict(annotations=ann,appearance=app,decision=dec,native_qc=qc,
        output_dir=root/'visual',object_id='synthetic',run_root=root)
    return args,frame,cv,faces


class VisualGeometryTests(unittest.TestCase):
    def test_full_topology_metric_observation_and_explicit_candidate_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,frame,cv,faces=visual_fixture(root)
            before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
            path=build_visual_geometry(**args)
            obs=load_visual_geometry(path,frame_ids=[7],frames=[frame],object_id='synthetic',
                expected_inputs={k:args[k] for k in ['annotations','appearance','decision','native_qc']})
            self.assertEqual(len(obs.faces),len(faces));np.testing.assert_array_equal(obs.faces,faces)
            np.testing.assert_allclose(obs.vertices_metric_camera,cv*2,atol=1e-6)
            self.assertFalse(obs.report['image_scale_optimized'])
            R=Rotation.from_rotvec([.1,.3,-.2]).as_matrix();t=np.array([.05,.02,1.])
            bound=bind_visual_to_candidate(obs,R,t,candidate_revision='synthetic-k0')
            expected=obs.vertices_metric_camera@obs.T_world_camera[:3,:3].T+obs.T_world_camera[:3,3]
            np.testing.assert_allclose(bound['vertices_canonical']@R.T+t,expected,atol=1e-8)
            with self.assertRaises(ValueError):obs.vertices_metric_camera.setflags(write=True)
            with self.assertRaises(ValueError):bind_visual_to_candidate(obs,R,t,candidate_revision='')
            self.assertTrue(all(sha(p)==h for p,h in before.items()))
            with self.assertRaises(ValueError):build_visual_geometry(**args)

    def test_reuse_requires_same_frame_seed_and_exact_conditioning(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,_,_,_=visual_fixture(root);qc=json.loads(args['native_qc'].read_text())
            self.assertTrue(can_reuse_native(args['native_qc'],qc['image'],qc['mask'],selected_frame=7,native_frame=7))
            self.assertFalse(can_reuse_native(args['native_qc'],qc['image'],qc['mask'],selected_frame=8,native_frame=7))
            self.assertFalse(can_reuse_native(args['native_qc'],qc['image'],qc['mask'],selected_frame=7,native_frame=7,seed=3))
            other=root/'other.png';cv2.imwrite(str(other),np.zeros((64,64),np.uint8))
            self.assertFalse(can_reuse_native(args['native_qc'],qc['image'],other,selected_frame=7,native_frame=7))

    def test_rejects_topology_plane_hash_and_generated_authority_confusion(self):
        for defect in ['topology','mask_hash','metric_eligibility','authority','source_frame']:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as d:
                root=Path(d);args,frame,_,_=visual_fixture(root)
                if defect=='topology':
                    from observations.object.shape import mesh_arrays
                    qc=json.loads(args['native_qc'].read_text());path=Path(qc['native_pose_mesh_opencv_camera']);v,f=mesh_arrays(path);write_mesh(path,v,f[:-1])
                    with self.assertRaisesRegex(ValueError,'topology'):build_visual_geometry(**args)
                elif defect=='mask_hash':
                    app=json.loads(args['appearance'].read_text());app['frames'][0]['appearance_mask_sha256']='0'*64;args['appearance'].write_text(json.dumps(app))
                    with self.assertRaises(ValueError):build_visual_geometry(**args)
                elif defect=='metric_eligibility':
                    a=json.loads(args['annotations'].read_text());a['frames'][0]['objects'][0]['rigid_pose_observation_eligible']=False;args['annotations'].write_text(json.dumps(a))
                    app=json.loads(args['appearance'].read_text());app['source_annotations_sha256']=sha(args['annotations']);args['appearance'].write_text(json.dumps(app))
                    with self.assertRaisesRegex(ValueError,'metric support'):build_visual_geometry(**args)
                else:
                    path=build_visual_geometry(**args);report=json.loads(path.read_text())
                    if defect=='authority':report['physical_authority']=True
                    else:report['T_world_camera_at_anchor'][0][3]+=.1
                    path.write_text(json.dumps(report))
                    with self.assertRaises(ValueError):load_visual_geometry(path,frame_ids=[7],frames=[frame],object_id='synthetic',expected_inputs={})


if __name__=='__main__':unittest.main()
