"""Canonical left/right MANO world state from independent WiLoR + DROID only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from utils.state import sha
from utils.mano import patch_legacy_mano_loader, JOINT_ORDER, TIP_VERTICES
from utils.hand_contract import FK_REPLAY_TOLERANCE_M,validate_replay_errors

MODE='explicit_wilor_source_K_independent_droid_canonical_MANO'


def canonical_rotations(global_right, hand_right, side, camera_rotation):
    M=np.diag([-1.,1.,1.]) if side=='left' else np.eye(3)
    root=np.asarray(camera_rotation)@(M@np.asarray(global_right).reshape(-1,3,3)@M)
    pose=M@np.asarray(hand_right).reshape(-1,15,3,3)@M
    return root,pose


def layer_geometry(model,root,pose,beta):
    import torch
    tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cpu')
    with torch.no_grad():
        o=model(global_orient=tensor(root).reshape(-1,1,3,3),hand_pose=tensor(pose),betas=tensor(beta),return_verts=True)
    joints=torch.cat([o.joints[:,:16],o.vertices[:,TIP_VERTICES]],dim=1)[:,JOINT_ORDER]
    return o.vertices.numpy(),joints.numpy()


def run(a):
    patch_legacy_mano_loader()
    import torch,smplx
    torch.set_num_threads(1)
    with np.load(a.wilor,allow_pickle=False) as z:q={k:z[k] for k in z.files}
    if 'T_world_camera' in q:raise ValueError('camera-local WiLoR must not inherit a world trajectory')
    if sha(a.wilor)!=json.loads(a.wilor.with_suffix('.completed.json').read_text())['output_sha256']:raise ValueError('WiLoR input changed')
    with np.load(a.camera,allow_pickle=False) as z:c={k:z[k] for k in z.files}
    camera_meta=json.loads(a.camera.with_suffix('.json').read_text())
    if camera_meta.get('HaWoR_inputs') is not False or sha(a.camera)!=camera_meta['output_sha256']:raise ValueError('independent camera provenance missing or changed')
    if not np.array_equal(q['frame_idx'],c['frame_idx']) or not np.array_equal(q['intrinsics'],c['intrinsics']):raise ValueError('camera/hand K or timeline differs')
    C=c['T_world_camera'];n=len(C);out=dict(frame_idx=q['frame_idx'],R_c2w=C[:,:3,:3],t_c2w=C[:,:3,3],
        camera_intrinsics_fx_fy_cx_cy=q['intrinsics'],img_focal=np.array([q['intrinsics'][0,0]]),
        video_path=np.array([str(a.video.resolve())]),video_sha256=np.array([sha(a.video)]),
        camera_intrinsics_contract_mode=np.array([MODE]),hand_backend=np.array(['wilor']),
        hand_contract_path=np.array([str((a.output/'mano_contract.json').resolve())]))
    replay={};model_hashes={}
    for side in ['left','right']:
        path=a.mano_models/f'MANO_{side.upper()}.pkl';model_hashes[side]=sha(path)
        model=smplx.MANOLayer(str(path),is_rhand=side=='right',use_pca=False).cpu().eval()
        if side=='left':model.shapedirs[:,0,:]*=-1
        valid=q[side+'_valid'].astype(bool);good=np.flatnonzero(valid)
        R,P=canonical_rotations(q[side+'_mano_global_orient_right'][good],q[side+'_mano_hand_pose_right'][good],side,C[good,:3,:3])
        beta=q[side+'_betas'][good]
        V,J=layer_geometry(model,R,P,beta)
        target=np.einsum('nij,nvj->nvi',C[good,:3,:3],q[side+'_vertices_camera_m'][good])+C[good,None,:3,3]
        targetj=np.einsum('nij,nvj->nvi',C[good,:3,:3],q[side+'_joints_camera_m'][good])+C[good,None,:3,3]
        # Shaped wrist pivot is nonzero: composing rotations is NOT R*t+camera_t.
        t=np.mean(target-V,axis=1)
        err=float(np.linalg.norm(V+t[:,None]-target,axis=-1).max()) if len(good) else 0.
        jerr=float(np.linalg.norm(J+t[:,None]-targetj,axis=-1).max()) if len(good) else 0.
        validate_replay_errors(err,jerr)
        fields={'vertices_world_m':(778,3),'joints_world_m':(21,3),'trans_world_m':(3,),
                'root_orient_axis_angle':(3,),'hand_pose_axis_angle':(45,),'betas':(10,)}
        for key,shape in fields.items():out[side+'_'+key]=np.full((n,*shape),np.nan,np.float32)
        for key,value in [('vertices_world_m',V+t[:,None]),('joints_world_m',J+t[:,None]),('trans_world_m',t),
                          ('root_orient_axis_angle',Rotation.from_matrix(R).as_rotvec()),
                          ('hand_pose_axis_angle',Rotation.from_matrix(P.reshape(-1,3,3)).as_rotvec().reshape(-1,45)),('betas',beta)]:
            out[side+'_'+key][good]=value
        out[side+'_faces']=np.asarray(model.faces,dtype=np.int32)
        out[side+'_valid']=valid.astype(np.uint8);out[side+'_detected_same_frame']=valid.astype(np.uint8)
        out[side+'_det_box_xyxyscore']=np.c_[q[side+'_bbox_xyxy'],q[side+'_detector_score']]
        out[side+'_track_id']=np.array(['wilor_'+side if v else '' for v in valid])
        out[side+'_hybrid_source']=np.array(['wilor_only_independent_droid_canonical_mano']*n)
        replay[side]=dict(max_vertex_error_m=err,max_joint_error_m=jerr,valid_frames=int(valid.sum()))
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'wilor_world_hands.npz';np.savez_compressed(path,**out)
    contract=dict(schema='ego.annotation.independent_wilor_mano_contract.v1',backend='wilor',camera_backend='independent_droid_unidepth',
        HaWoR_inputs=False,mode=MODE,output_path=str(path.resolve()),output_sha256=sha(path),
        video_path=str(a.video.resolve()),video_sha256=sha(a.video),image_size_wh=q['image_size_wh'].tolist(),
        frame_ids=q['frame_idx'].tolist(),intrinsics=q['intrinsics'].tolist(),canonical_FK_replay=replay,
        numerical_FK_replay_tolerance_m=FK_REPLAY_TOLERANCE_M,
        model_hashes=model_hashes,source_wilor_path=str(a.wilor),source_wilor_sha256=sha(a.wilor),
        independent_camera_path=str(a.camera),independent_camera_sha256=sha(a.camera),
        limitations=['WiLoR single-frame metric depth remains uncertain','no temporal infilling or contact authority'])
    (a.output/'mano_contract.json').write_text(json.dumps(contract,indent=2)+'\n');print(json.dumps(replay,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['wilor','camera','video','mano-models','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
