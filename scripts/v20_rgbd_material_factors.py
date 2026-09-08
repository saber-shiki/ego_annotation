"""Fixed RGB-D material observations and their source-to-target SE(3) factors.

No generated mesh, candidate construction, GT, or model inference in this module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from v20_prediction_contracts import assert_prediction_only


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def valid_camera(T: np.ndarray) -> bool:
    return bool(T.shape==(4,4) and np.isfinite(T).all() and np.allclose(T[3],[0,0,0,1],atol=1e-7)
                and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
                and abs(np.linalg.det(T[:3,:3])-1)<1e-5)


@dataclass(frozen=True)
class MaterialFactor:
    source_frame_idx: int
    target_frame_idx: int
    source_pos: int
    target_pos: int
    source_world: np.ndarray
    target_world: np.ndarray
    target_uv: np.ndarray
    target_camera: np.ndarray
    target_intrinsics: np.ndarray
    metric_valid: np.ndarray
    weights: np.ndarray
    edge_weight: float


def transport_source(factor: MaterialFactor, poses: Mapping[int, tuple[np.ndarray,np.ndarray]]) -> np.ndarray:
    Rs,ts=poses[factor.source_frame_idx];Rt,tt=poses[factor.target_frame_idx]
    return ((factor.source_world-ts)@Rs)@Rt.T+tt


def errors(factor: MaterialFactor, poses: Mapping[int, tuple[np.ndarray,np.ndarray]]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Unweighted pixel/metric errors and forward camera depth; both poses vary."""
    pred_world=transport_source(factor,poses);C=factor.target_camera
    camera=(pred_world-C[:3,3])@C[:3,:3]
    fx,fy,cx,cy=factor.target_intrinsics
    safe_z=np.maximum(camera[:,2],1e-3)
    uv=camera[:,:2]/safe_z[:,None]*[fx,fy]+[cx,cy]
    xyz=(pred_world-factor.target_world)@C[:3,:3]
    # Missing target depth is not a fabricated point at world zero.
    xyz=np.where(factor.metric_valid[:,None],xyz,0.)
    return uv-factor.target_uv,xyz,camera[:,2]


def residual(factor: MaterialFactor, poses: Mapping[int, tuple[np.ndarray,np.ndarray]], *, image_weight: float, metric_weight: float, sigma_px: float, sigma_m: float) -> np.ndarray:
    uv,xyz,z=errors(factor,poses)
    weights=np.sqrt(factor.weights*factor.edge_weight)
    iw=np.sqrt(image_weight)/sigma_px/np.sqrt(max(1,len(uv)))
    dw=np.sqrt(metric_weight)/sigma_m/np.sqrt(max(1,int(factor.metric_valid.sum())))
    # A nonzero cheirality barrier prevents a clipped projection behind the
    # camera from becoming a zero-gradient "safe" observation.
    barrier=np.maximum(0.,.01-z)/sigma_m/np.sqrt(max(1,len(z)))
    return (np.column_stack([iw*uv,dw*xyz,barrier])*weights[:,None]).reshape(-1)


def metrics(factors: Sequence[MaterialFactor], poses: Mapping[int, tuple[np.ndarray,np.ndarray]]) -> dict[str,Any]:
    pixels=[];distances=[];rows=[];behind=0
    for f in factors:
        uv,xyz,z=errors(f,poses);p=np.linalg.norm(uv,axis=1);d=np.linalg.norm(xyz[f.metric_valid],axis=1)
        pixels.extend(p.tolist());distances.extend(d.tolist());behind+=int(np.count_nonzero(z<=.01))
        rows.append({'source_frame':f.source_frame_idx,'target_frame':f.target_frame_idx,'image_points':len(p),'metric_points':len(d),
                     'pixel_median_px':float(np.median(p)), 'metric_median_m':float(np.median(d)) if len(d) else None})
    def q(x,n):return float(np.percentile(x,n)) if len(x) else None
    return {'material_pixel_median_px':q(pixels,50),'material_pixel_p95_px':q(pixels,95),
            'material_metric_median_m':q(distances,50),'material_metric_p95_m':q(distances,95),
            'material_image_count':len(pixels),'material_metric_count':len(distances),'material_behind_camera_count':behind,'material_per_pair':rows}


def load_factors(path: Path, nodes: Sequence[Any], *, annotation_sha256: str, object_id: str, max_points: int=96) -> tuple[list[MaterialFactor],dict[str,Any]]:
    """Load a packed fixed-world RGB-D observation bundle, fail closed."""
    with np.load(path,allow_pickle=False) as data:
        required={'metadata','source_frame_idx','target_frame_idx','offsets','source_world','target_world','target_uv','target_camera','target_intrinsics','metric_valid','weights','edge_weight','source_metric_eligible','target_metric_eligible'}
        if required-set(data.files):raise ValueError(f'material bundle missing {sorted(required-set(data.files))}')
        raw=np.asarray(data['metadata']).reshape(-1)
        if len(raw)!=1:raise ValueError('material metadata must have one object')
        meta=json.loads(str(raw[0]));assert_prediction_only(meta,label='material RGB-D observations',object_id=object_id)
        if meta.get('schema')!='v20_fixed_world_material_rgbd_v1' or meta.get('gt_consumed') is not False:
            raise ValueError('material schema/provenance mismatch')
        if meta.get('object_id')!=object_id or meta.get('input_sha256',{}).get('annotations')!=annotation_sha256:
            raise ValueError('material annotation/object identity mismatch')
        if meta.get('source_frame')!='fixed_world_observation':raise ValueError('material world frame undeclared')
        arrays={k:np.asarray(data[k]).copy() for k in required-{'metadata'}}
    src=arrays['source_frame_idx'];dst=arrays['target_frame_idx'];off=arrays['offsets'];n=len(src);p=len(arrays['source_world'])
    if src.shape!=(n,) or dst.shape!=(n,) or off.shape!=(n+1,) or off.dtype.kind not in 'iu' or off[0]!=0 or off[-1]!=p or np.any(np.diff(off)<0):
        raise ValueError('invalid material edge offsets')
    if len(set(zip(src.tolist(),dst.tolist())))!=n or np.any(src==dst):raise ValueError('duplicate/self material edges')
    for key,shape in [('source_world',(p,3)),('target_world',(p,3)),('target_uv',(p,2)),('weights',(p,)),('metric_valid',(p,)),('target_camera',(n,4,4)),('target_intrinsics',(n,4)),('edge_weight',(n,)),('source_metric_eligible',(n,)),('target_metric_eligible',(n,))]:
        if arrays[key].shape!=shape:raise ValueError(f'material {key} shape mismatch')
        if not np.isfinite(arrays[key]).all():raise ValueError(f'material {key} is non-finite')
    if arrays['metric_valid'].dtype.kind!='b' or arrays['source_metric_eligible'].dtype.kind!='b' or arrays['target_metric_eligible'].dtype.kind!='b':
        raise ValueError('material eligibility masks must be boolean')
    if np.any(arrays['weights']<=0) or np.any(arrays['weights']>1) or np.any(arrays['edge_weight']<=0) or np.any(arrays['edge_weight']>1):
        raise ValueError('material weights outside (0,1]')
    positions={node.frame_idx:i for i,node in enumerate(nodes)};factors=[];rejected=[]
    for i,(s,t) in enumerate(zip(src,dst)):
        s,t=int(s),int(t);l,r=map(int,off[i:i+2])
        if s not in positions or t not in positions:
            rejected.append({'source':s,'target':t,'point_count':r-l,'reason':'frame_not_in_requested_timeline'})
            continue
        if r-l<4:rejected.append({'source':s,'target':t,'reason':'too_few_points'});continue
        if not arrays['source_metric_eligible'][i] or not nodes[positions[s]].metric_observation:
            raise ValueError(f'material source {s} lacks P09 metric authority')
        valid=arrays['metric_valid'][l:r]
        if valid.any() and (not arrays['target_metric_eligible'][i] or not nodes[positions[t]].metric_observation):
            raise ValueError(f'material target {t} fabricated metric observation')
        T=arrays['target_camera'][i];K=arrays['target_intrinsics'][i]
        if not valid_camera(T) or not np.all(K[:2]>0):raise ValueError('invalid material target camera/intrinsics')
        ix=np.arange(l,r)
        if max_points>0 and len(ix)>max_points:ix=ix[np.linspace(0,len(ix)-1,max_points,dtype=int)]
        factors.append(MaterialFactor(s,t,positions[s],positions[t],arrays['source_world'][ix],arrays['target_world'][ix],arrays['target_uv'][ix],T,K,arrays['metric_valid'][ix],arrays['weights'][ix],float(arrays['edge_weight'][i])))
    if not factors:raise ValueError('no material factors in requested timeline')
    # Connectedness is evidence, not an artificial guarantee of observability.
    adj={f: set() for f in positions}
    for f in factors:adj[f.source_frame_idx].add(f.target_frame_idx);adj[f.target_frame_idx].add(f.source_frame_idx)
    unseen=set(positions);components=[]
    while unseen:
        todo=[min(unseen)];part=[]
        while todo:
            cur=todo.pop()
            if cur not in unseen:continue
            unseen.remove(cur);part.append(cur);todo.extend(adj[cur])
        components.append(sorted(part))
    return factors,{'metadata':meta,'bundle_sha256':sha256_file(path),'pair_count':len(factors),
                    'image_points':sum(len(f.weights) for f in factors),'metric_points':sum(int(f.metric_valid.sum()) for f in factors),
                    'components':components,'rejected':rejected,'factor_kind':'fixed_material_world_observation_transport','generated_factors_included':False}
