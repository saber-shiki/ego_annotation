from __future__ import annotations
"""Complete renderable mesh/timeline contract, independent of physical acceptance."""
import numpy as np
from observations.common import pose_from_values

def mesh(vertices,faces):
    v=np.asarray(vertices);f=np.asarray(faces)
    if v.ndim!=2 or v.shape[1:]!=(3,) or not len(v) or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or not np.issubdtype(f.dtype,np.integer) or f.min()<0 or f.max()>=len(v):
        raise ValueError('invalid complete triangle geometry')
    return v,f


def timeline(z):
    ids=np.asarray(z['frame_idx']);times=np.asarray(z['time_s'],float)
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(set(ids.tolist()))!=len(ids) or not len(ids) or times.shape!=ids.shape or not np.isfinite(times).all():
        raise ValueError('invalid full scene timeline')
    order=np.argsort(times)
    if np.any(np.diff(times[order])<=0):raise ValueError('duplicate/nonincreasing times')
    n=len(ids)
    T=np.asarray(z['T_world_camera']);K=np.asarray(z['intrinsics_render']);size=np.asarray(z['render_size_wh'])
    if T.shape!=(n,4,4) or K.shape!=(n,4) or not np.isfinite(K).all() or np.any(K[:,:2]<=0) or size.shape!=(2,) or not np.issubdtype(size.dtype,np.integer) or np.any(size<=0):
        raise ValueError('invalid explicit camera/image plane')
    for t in T:
        pose_from_values(t[:3,:3],t[:3,3])
        if not np.allclose(t[3],[0,0,0,1]):raise ValueError('invalid camera homogeneous row')
    if np.asarray(z['object_rotation']).shape!=(n,3,3) or np.asarray(z['object_translation_world']).shape!=(n,3):raise ValueError('object pose length mismatch')
    if 'object_valid' not in z:raise ValueError('explicit object_valid required from standalone fit')
    valid=np.asarray(z['object_valid'])
    if valid.shape!=(n,) or valid.dtype!=bool:raise ValueError('object validity must be boolean')
    for i in np.flatnonzero(valid):pose_from_values(z['object_rotation'][i],z['object_translation_world'][i])
    for side in ('left','right'):
        hv=np.asarray(z[side+'_valid'])
        if hv.shape!=(n,) or hv.dtype!=bool or np.shape(z[side+'_vertices_world'])!=(n,778,3):raise ValueError('full side-specific MANO surface/validity required')
        for i in np.flatnonzero(hv):mesh(z[side+'_vertices_world'][i],z[side+'_faces'])
    mesh(z['object_vertices_canonical'],z['object_faces'])
    return ids,times,order,T,K,size,valid

def reject_rejected_annotation_path(path: Path) -> None:
    from pathlib import Path
    raw = str(path)
    hits = [marker for marker in REJECTED_ANNOTATION_PATH_MARKERS if marker in raw]
    if hits:
        raise ValueError(f'rejected H-prime/final-v7 annotation source supplied to interval MANO solver: {path}. Use same-run source observations, not a previously corrected candidate.')
REJECTED_ANNOTATION_PATH_MARKERS = ('v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard', 'verified_hprime_final', 'hprime_final')
