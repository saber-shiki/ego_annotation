"""Rigid canonical rebinding preserves every world vertex."""
import copy
import numpy as np
from observations.common import pose_from_values
from utils.data import RKEY, TKEY

def rebind_canonical(vertices, rows, Q, b):
    """c_old=Q*c_new+b; preserve all world vertices under a rigid gauge change."""
    Q,t=pose_from_values(Q,b)
    new_vertices=(vertices-t)@Q
    new_rows=copy.deepcopy(rows)
    for old,new in zip(rows,new_rows):
        r,p=pose_from_values(old[RKEY],old[TKEY])
        new[RKEY]=(r@Q).tolist();new[TKEY]=(p+r@t).tolist()
        if not np.allclose(new_vertices@(r@Q).T+p+r@t,vertices@r.T+p,atol=1e-8):
            raise ValueError('canonical gauge broke world geometry')
    return new_vertices,new_rows

from pathlib import Path
import json
import cv2
from utils.data import load, indexed
from utils.state import sha, in_run
from utils.geometry import resized_K
from gates.object.geometry import validate_posterior
from gates.object.observation import p09_geometry_validation
from observations.object.shape import AUTHORITY, record, mesh_arrays, write_mesh, canonical_gauge, scene_similarity, estimate_scale

def bind(posterior_path, annotations_path, appearance_path, decision_path, qc_path, output, object_id, root):
    if output.exists(): raise ValueError('create-only shape binding required')
    posterior=validate_posterior(load(posterior_path),object_id,root)
    if posterior['evidence']['annotations_sha256']!=sha(annotations_path): raise ValueError('posterior annotation identity changed')
    frames=indexed(load(annotations_path)['frames']);poses=indexed(posterior['pose_rows'])
    appearance=load(appearance_path)
    if appearance.get('object_id')!=object_id or appearance.get('source_annotations_sha256')!=sha(annotations_path) or appearance.get('metric_owned_masks_consumed') is not False:
        raise ValueError('appearance/metric separation mismatch')
    masks=indexed(appearance['frames']);decision=load(decision_path)
    idx=decision['selected_frame_idx']
    if decision.get('object_id')!=object_id or idx not in poses: raise ValueError('appearance anchor identity mismatch')
    def observation(i):
        a=masks[i];f=frames[i]
        mask_path=in_run(a['appearance_mask_path'],root);rgb=in_run(a['raw_rgb_path'],root)
        if sha(mask_path)!=a['appearance_mask_sha256'] or sha(rgb)!=a['raw_rgb_sha256']: raise ValueError('appearance bytes changed')
        mask=cv2.imread(str(mask_path),0)
        if mask is None or mask.shape[::-1]!=tuple(a['source_size_wh']): raise ValueError('appearance raster mismatch')
        cam=f['camera'];source_wh=load(root/'input/raw_frame_manifest/manifest.json')['frames'][0]
        size=[source_wh['source_width'],source_wh['source_height']]
        K=resized_K(cam['intrinsics_fx_fy_cx_cy'],size,a['source_size_wh'])
        T=np.asarray(cam.get('T_world_camera_metric',cam.get('T_world_camera')),float)
        r,t=pose_from_values(poses[i][RKEY],poses[i][TKEY])
        return {'mask':mask>0,'K':K,'T':T,'R':r,'t':t},rgb,mask_path
    anchor,rgb,mask_path=observation(idx)
    obj=next(o for o in frames[idx]['objects'] if o['object_id']==object_id)
    world,reason,_=p09_geometry_validation(obj,frames[idx])
    if world is None or obj.get('rigid_pose_observation_eligible') is not True or obj['visible_geometry_candidate'].get('rigid_pose_observation_eligible') is not True:
        raise ValueError('appearance anchor requires retained metric alignment, not mask depth')
    T=anchor['T'];observed=(world-T[:3,3])@T[:3,:3]
    qc=load(qc_path)
    if qc.get('status')!='ok' or Path(qc['image']).resolve()!=rgb or Path(qc['mask']).resolve()!=mask_path:
        raise ValueError('native reconstruction conditioning mismatch')
    raw,faces=mesh_arrays(in_run(qc['mesh'],root))
    from utils.geometry import apply_native_pose
    p=qc['pose'];p3d,cv,_=apply_native_pose(raw,np.asarray(p['rotation']),np.asarray(p['translation']),np.asarray(p['scale']))
    for key,expected in [('native_pose_mesh_opencv_camera',cv),('native_pose_mesh_pytorch3d_camera',p3d)]:
        v,f=mesh_arrays(in_run(qc[key],root))
        if not np.array_equal(f,faces) or v.shape!=expected.shape or not np.allclose(v,expected,atol=1e-5,rtol=0):
            raise ValueError('native full mesh pose replay mismatch')
    scale,depth_report=scene_similarity(cv,observed,anchor['K'],anchor['mask'])
    canonical,gauge=canonical_gauge(cv,scale,T,poses[idx])
    candidates=[i for i in poses if masks[i]['status']=='available']
    chosen=[candidates[j] for j in np.unique(np.linspace(0,len(candidates)-1,min(7,len(candidates)),dtype=int))]
    observations=[]
    for i in chosen:
        obs,_,_=observation(i);h,w=obs['mask'].shape;size=[160,max(1,round(160*h/w))]
        obs['K']=resized_K(obs['K'],[w,h],size)
        obs['mask']=cv2.resize(obs['mask'].astype('uint8'),tuple(size),interpolation=cv2.INTER_NEAREST_EXACT)>0
        observations.append(obs)
    image_scale=estimate_scale(canonical,faces,observations,samples=21)
    pivot=np.asarray(image_scale['pivot_canonical_m']);scaled=(canonical-pivot)*image_scale['uniform_scale']+pivot
    output.mkdir(parents=True)
    mesh=output/'object_canonical.ply';topology=write_mesh(mesh,scaled,faces)
    receipt={**AUTHORITY,'output_mesh_path':str(mesh),'output_mesh_sha256':sha(mesh),'topology':topology,
             'native_qc':record(qc_path),'native_gauge':gauge,'scene_similarity':depth_report,'static_image_scale':image_scale}
    binding={**copy.deepcopy(posterior),'schema':'rgb_scene_visual_refined_binding_v1',
             'refined_posterior':record(posterior_path),'appearance_mask_manifest':record(appearance_path),
             'appearance_decision':record(decision_path),'visual_mesh_receipt':receipt,
             'render_mesh_contract':{'required_mesh_path':str(mesh),'required_mesh_sha256':sha(mesh),'renderer_must_match':True}}
    # Exact pose rows, not rewritten source/readiness rows.
    if binding['pose_rows']!=posterior['pose_rows']: raise ValueError('posterior reset during shape binding')
    with (output/'pose_mesh.json').open('x') as f: json.dump(binding,f,indent=2,allow_nan=False)
    return binding
