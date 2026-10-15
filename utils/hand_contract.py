"""Backend-specific camera/MANO provenance; never impersonate HaWoR contracts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from utils.state import sha

MODE='explicit_wilor_source_K_independent_droid_canonical_MANO'
# Owner-authorized numerical replay allowance (2026-10-15): 5 micrometres.
# This bounds float32 FK/coordinate conversion only, never physical accuracy.
FK_REPLAY_TOLERANCE_M=5e-6


def validate_replay_errors(vertex_error, joint_error):
    errors=np.asarray([vertex_error,joint_error],float)
    if not np.isfinite(errors).all() or np.any(errors<0) or errors.max()>FK_REPLAY_TOLERANCE_M:
        raise ValueError(f'canonical MANO numerical replay mismatch: {errors.tolist()}m; tolerance={FK_REPLAY_TOLERANCE_M}m')


def validate_independent_archive(arrays, archive_path, video_path, source_size_wh, active_K):
    path=Path(str(np.asarray(arrays['hand_contract_path']).reshape(-1)[0]))
    contract=json.loads(path.read_text())
    if contract.get('schema')!='ego.annotation.independent_wilor_mano_contract.v1' or contract.get('HaWoR_inputs') is not False:
        raise ValueError('independent WiLoR producer contract missing')
    if contract.get('backend')!='wilor' or contract.get('camera_backend')!='independent_droid_unidepth':
        raise ValueError('independent backend identity differs')
    if Path(contract['output_path']).resolve()!=Path(archive_path).resolve() or contract['output_sha256']!=sha(archive_path):
        raise ValueError('canonical MANO archive differs from producer')
    if contract['video_sha256']!=sha(video_path) or contract['image_size_wh']!=list(source_size_wh):
        raise ValueError('independent MANO source image differs')
    if not np.array_equal(contract['frame_ids'],arrays['frame_idx']):raise ValueError('independent hand timeline differs')
    K=np.asarray(arrays['camera_intrinsics_fx_fy_cx_cy'],float)
    if K.shape!=(len(arrays['frame_idx']),4) or not np.allclose(K,active_K,atol=1e-6,rtol=0):raise ValueError('independent hand source K differs')
    if not np.allclose(K,np.asarray(contract['intrinsics']),atol=1e-9,rtol=0):raise ValueError('independent hand K differs from producer')
    for side in ['left','right']:
        replay=contract['canonical_FK_replay'][side]
        validate_replay_errors(replay['max_vertex_error_m'],replay['max_joint_error_m'])
    return dict(backend='wilor',mode=MODE,active_contract_reinference_required=False,
        source_camera_intrinsics_fx_fy_cx_cy=K[0].tolist(),source_image_size_wh=list(source_size_wh),
        source_K_bound_to_projection=True,canonical_MANO_FK_replayed=True,
        independent_camera_backend='independent_droid_unidepth',HaWoR_inputs=False,
        producer_contract_path=str(path.resolve()),producer_contract_sha256=sha(path),
        claim_scope='WiLoR own-2D camera-local perspective translation + canonical left/right MANO, composed with fresh independent DROID/UniDepth camera. No claim of absolute metric accuracy.')


def independent_alignment_valid(alignment, K):
    if alignment.get('backend')!='wilor':return False
    if (alignment.get('mode')!=MODE or alignment.get('HaWoR_inputs') is not False
        or alignment.get('active_contract_reinference_required') is not False
        or alignment.get('source_K_bound_to_projection') is not True
        or alignment.get('canonical_MANO_FK_replayed') is not True
        or alignment.get('independent_camera_backend')!='independent_droid_unidepth'):
        raise ValueError('unproven independent WiLoR camera/MANO contract')
    p=Path(alignment['producer_contract_path'])
    if sha(p)!=alignment['producer_contract_sha256']:raise ValueError('independent MANO producer contract changed')
    source=np.asarray(alignment.get('source_camera_intrinsics_fx_fy_cx_cy',[]),float)
    if source.shape!=(4,) or not np.allclose(source,K,atol=.01,rtol=0):raise ValueError('WiLoR projection K differs from camera')
    return True


def hand_row(*, side,frame_idx,arrays,archive_path,position,bridge_path,bridge_rows,T_world_camera,intrinsics,alignment):
    valid=bool(arrays[side+'_valid'][position]);detected=bool(arrays[side+'_detected_same_frame'][position])
    base=dict(hand_side=side,same_frame_detection=detected,confidence=.65 if detected else 0.,
              visibility_state='wilor_measured' if valid else 'not_measured',hand_geometry_source='wilor_only_independent_droid_canonical_mano',
              uncertainty=['single_frame_depth_and_shape_uncertain','not_a_contact_or_nonpenetration_claim'],camera_contract_alignment=alignment)
    if not valid:return base
    vertices=arrays[side+'_vertices_world_m'][position];joints=arrays[side+'_joints_world_m'][position]
    to_cam=lambda x:(x-T_world_camera[:3,3])@T_world_camera[:3,:3]
    row=bridge_rows[(frame_idx,side)];params={key:arrays[side+'_'+key][position].tolist() for key in ['root_orient_axis_angle','hand_pose_axis_angle','betas','trans_world_m']}
    params.update(parameterization='canonical_MANO_axis_angle_world',side=side,source_frame_index=int(position),source_mano_npz=archive_path,
                  source_hawor_npz=archive_path,legacy_pointer_alias_only=True,
                  arrays={key:side+'_'+key for key in ['root_orient_axis_angle','hand_pose_axis_angle','betas','trans_world_m']})
    reference=dict(bridge_npz=str(bridge_path),bridge_vertices_world_array='vertices_current_v18_world_from_hawor_projection_relift_m',
        bridge_vertices_camera_array='vertices_current_v18_camera_m',bridge_row_index=int(row),source_mano_npz=archive_path,
        source_hawor_npz=archive_path,source_frame_index=int(position),legacy_pointer_alias_only=True,
        source_vertices_world_array=side+'_vertices_world_m',source_joints_world_array=side+'_joints_world_m',shape_vertices=[778,3],shape_joints=[21,3])
    metric=dict(source='independent_wilor_canonical_MANO',coordinate_status='metric_world_from_independent_droid_unidepth',
        case_frame_idx=int(frame_idx),hand_side=side,mano_params=params,vertices_reference=reference,
        joints_current_v18_world_m=joints.tolist(),joints_world_m=joints.tolist(),joints_current_v18_camera_m=to_cam(joints).tolist(),
        current_v18_camera_intrinsics_fx_fy_cx_cy=list(intrinsics),current_v18_camera_intrinsics_source=MODE,
        active_camera_contract_intrinsics_fx_fy_cx_cy=list(intrinsics),camera_contract_alignment=alignment,support_state=base['hand_geometry_source'])
    base.update(metric_mano_state=metric,bbox_xyxy=arrays[side+'_det_box_xyxyscore'][position,:4].tolist())
    return base
