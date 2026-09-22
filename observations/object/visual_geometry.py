"""Full appearance-native observation, before any pose/shape optimization.

The output stays in the selected frame's METRIC CAMERA coordinates. Canonical
binding needs a separately identified candidate pose and must not be smuggled
into a source bundle. No image-scale grid search or trajectory solver runs here.
"""
from pathlib import Path
import json
import numpy as np
import cv2
from observations.object.shape import (sam3d_candidate_record, mesh_arrays, write_mesh,
    scene_similarity, metric_object)
from envs.qualification.object_observation import retained_metric_alignment
from utils.geometry import apply_native_pose, resized_K, validate_transforms
from utils.state import sha
from utils.data import indexed

SCHEMA = 'ego.annotation.visual_geometry_observation.v1'
ROLE = 'generated_full_visual_mesh_not_collision_authority'


def record(path):
    path = Path(path).resolve()
    return {'path':str(path),'sha256':sha(path)}


def can_reuse_native(qc_path, image, mask, *, selected_frame, native_frame, seed=42):
    """Memoize only this run's same-frame, same-byte native call.

    Main's immutable StageRecovery prefix verifies the original model/config,
    code and successful worker receipt; this is not a foreign-cache importer.
    """
    if selected_frame != native_frame:
        return False
    qc = json.loads(Path(qc_path).read_text())
    if qc.get('status') != 'ok' or qc.get('seed') != seed:
        return False
    return (sha(qc['image']) == sha(image) and sha(qc['mask']) == sha(mask))


def build_visual_geometry(*, annotations, appearance, decision, native_qc, output_dir,
                          object_id, run_root):
    root=Path(run_root).resolve();output=Path(output_dir).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError('visual observation requires new output directory in this run')
    paths={name:Path(p).resolve() for name,p in dict(annotations=annotations,
        appearance=appearance,decision=decision,native_qc=native_qc).items()}
    if any(not p.is_relative_to(root) or not p.is_file() for p in paths.values()):
        raise ValueError('visual observation inputs must belong to this run')
    ann=json.loads(paths['annotations'].read_text());app=json.loads(paths['appearance'].read_text())
    choice=json.loads(paths['decision'].read_text());qc=json.loads(paths['native_qc'].read_text())
    if (app.get('object_id')!=object_id or choice.get('object_id')!=object_id
            or app.get('source_annotations_sha256')!=sha(paths['annotations'])
            or Path(app.get('source_annotations','')).resolve()!=paths['annotations']
            or app.get('metric_owned_masks_consumed') is not False):
        raise ValueError('appearance observation identity/role mismatch')
    frames=indexed(ann['frames']);masks=indexed(app['frames']);selected=choice['selected_frame_idx']
    if set(frames)!=set(masks) or selected not in frames:raise ValueError('appearance timeline mismatch')
    row=masks[selected];frame=frames[selected]
    if row.get('status')!='available' or row.get('anchor_eligible') is not True:
        raise ValueError('appearance anchor is not eligible')
    for name,key in [('image','raw_rgb'),('mask','appearance_mask')]:
        p=Path(row[key+'_path']).resolve()
        if not p.is_relative_to(root) or sha(p)!=row[key+'_sha256']:
            raise ValueError('appearance image/mask binding changed')
        paths[name]=p
    image=cv2.imread(str(paths['image']));mask=cv2.imread(str(paths['mask']),0)
    if image is None or mask is None or image.shape[:2]!=mask.shape or list(mask.shape[::-1])!=row['source_size_wh']:
        raise ValueError('appearance raster mismatch')
    obj=metric_object(frame,object_id);world,reason,_=retained_metric_alignment(obj,frame)
    if world is None:raise ValueError('appearance requires retained metric support: '+reason)
    T=np.asarray(frame['camera'].get('T_world_camera_metric',frame['camera'].get('T_world_camera')),float)
    validate_transforms(T[None]);observed=(world-T[:3,3])@T[:3,:3]
    K=resized_K(frame['camera']['intrinsics_fx_fy_cx_cy'],
                [frame['source_width'],frame['source_height']],mask.shape[::-1])
    candidate=sam3d_candidate_record(paths['native_qc'],qc.get('name'),paths['image'],paths['mask'],
                                    {'mode':'current_run_native_observation'})
    native=candidate['native_outputs'];paths['native_mesh']=Path(native['raw_mesh']['path'])
    vertices,faces=mesh_arrays(paths['native_mesh']);pose=native['native_pose']
    p3d,cv,contract=apply_native_pose(vertices,np.asarray(pose['rotation']),np.asarray(pose['translation']),np.asarray(pose['scale']))
    for name,expected in [('native_pose_mesh_pytorch3d_camera',p3d),('native_pose_mesh_opencv_camera',cv)]:
        path=Path(native[name]['path']);paths[name]=path
        if not path.resolve().is_relative_to(root):raise ValueError('native mesh escapes this run')
        v,f=mesh_arrays(path)
        if not np.array_equal(f,faces) or v.shape!=expected.shape or not np.allclose(v,expected,atol=1e-5,rtol=0):
            raise ValueError('complete native topology/pose replay mismatch')
    if not paths['native_mesh'].resolve().is_relative_to(root):raise ValueError('native source escapes run')
    scale,diagnostic=scene_similarity(cv,observed,K,mask>0)
    output.mkdir(parents=True)
    mesh=output/'visual_full_metric_camera.ply'
    topology=write_mesh(mesh,cv*scale,faces)
    report={'schema':SCHEMA,'object_id':object_id,'frame_ids':list(frames),
        'appearance_anchor_frame_idx':selected,'role':ROLE,'classes':['R2','R3'],
        'coordinate_frame':'opencv_camera_at_appearance_anchor','units':'meters',
        'T_world_camera_at_anchor':T.tolist(),'intrinsics_appearance':np.asarray(K).tolist(),
        'appearance_size_wh':list(mask.shape[::-1]),'source_K':frame['camera']['intrinsics_fx_fy_cx_cy'],
        'scene_similarity':diagnostic,'native_pose_contract':contract,'topology':topology,
        'source_native_mesh':record(paths['native_mesh']),'output_mesh':record(mesh),
        'inputs':{k:record(p) for k,p in paths.items()},
        'candidate_dependency':'none','canonical_binding':'deferred_requires_explicit_candidate_snapshot',
        'image_scale_optimized':False,'contains_optimized_state':False,'physical_authority':False,
        'generated_faces_signed_eligible':False,'generated_faces_collision_eligible':False}
    path=output/'visual_geometry_observation.json'
    with path.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    return path
