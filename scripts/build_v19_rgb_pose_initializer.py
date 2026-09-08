#!/usr/bin/env python3
"""Fresh RGB world-relative SE(3) initializer from THIS run's P14 and edges.

A quality-weighted spanning graph seeds orientation/translation; a robust SO(3)
loop solve reconciles accepted edges. It does not use generated geometry or GT,
and never promotes inferred timeline rows to direct metric observations.
"""
from __future__ import annotations
import argparse
import copy
import heapq
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import fit_v20_late_window_prediction_only_se3 as se3
from v20_prediction_contracts import assert_prediction_only
from v19_run_lineage import RunLineage
from v19_runtime_contracts import in_run, validate_references


def initialize(base, annotations, object_id, edge_arrays, edge_rows, anchor):
    original=se3._timeline_rows(base);timeline=sorted(int(f['frame_idx']) for f in annotations['frames'])
    if len(set(timeline))!=len(timeline) or anchor not in timeline:raise ValueError('invalid timeline/anchor')
    initial={}
    for idx,row in original.items():
        pose=se3.pose_from_values(row.get('rotation_world_from_completed_canonical_matrix'),row.get('translation_world_m'))
        if pose is not None:initial[idx]=pose
    if anchor not in initial:raise ValueError('anchor has no source P14 pose')
    original_initial = dict(initial)
    for idx in timeline:
        if idx not in initial:initial[idx]=se3.initialize_missing_se3_pose(idx,original_initial)[:2]
    quality={(int(r['source_frame_idx']),int(r['target_frame_idx'])):r for r in edge_rows}
    src,dst=edge_arrays['source_frame_idx'],edge_arrays['target_frame_idx'];Rs=edge_arrays['rotation_source_to_target_world'];ts=edge_arrays['translation_source_to_target_world_m'];accepted=edge_arrays['accepted']
    if not(len(src)==len(dst)==len(Rs)==len(ts)==len(accepted)):raise ValueError('edge length mismatch')
    adjacency={i:[] for i in timeline};edges=[];seen=set()
    for a,b,R,t,ok in zip(src,dst,Rs,ts,accepted):
        a,b=int(a),int(b)
        if (a,b) in seen:raise ValueError('duplicate RGB edge')
        seen.add((a,b))
        if a not in adjacency or b not in adjacency:continue
        row=quality.get((a,b))
        if row is None or (row.get('status')=='accepted')!=bool(ok):raise ValueError('edge NPZ/JSON acceptance mismatch')
        if not ok:continue
        if not se3.valid_rotation(R) or np.asarray(t).shape!=(3,) or not np.isfinite(t).all():raise ValueError('invalid world-relative edge')
        condition=float(row.get('source_3d_conditioning',0));inliers=float(row.get('inlier_fraction',0));error=float(row.get('reprojection_median_px',np.inf))
        if not np.isfinite([condition,inliers,error]).all() or condition < 0 or not 0 <= inliers <= 1 or error < 0:
            raise ValueError('invalid RGB quality evidence')
        weight=float(np.clip(inliers*np.exp(-error/2)*np.clip(condition/.02,.05,1.),0,1))
        if weight<=.01:continue
        cost=-np.log(weight)+.01*abs(b-a);edges.append((a,b,R,t,weight))
        adjacency[a].append((b,R,t,cost));adjacency[b].append((a,R.T,-R.T@t,cost))
    distance={anchor:0.};seed={anchor:initial[anchor]};queue=[(0.,anchor)]
    while queue:
        cost,a=heapq.heappop(queue)
        if cost!=distance[a]:continue
        for b,R,t,w in adjacency[a]:
            candidate=cost+w
            if candidate<distance.get(b,float('inf')):
                Ra,ta=seed[a];seed[b]=(R@Ra,R@ta+t);distance[b]=candidate;heapq.heappush(queue,(candidate,b))
    connected=set(seed);poses={idx:seed.get(idx,initial[idx]) for idx in timeline}
    seed_R={idx:p[0] for idx,p in poses.items()};free=[idx for idx in timeline if idx in connected and idx!=anchor];fp={idx:n for n,idx in enumerate(free)}
    used=[e for e in edges if e[0] in connected and e[1] in connected]
    def rotations(x):
        rr=dict(seed_R)
        for idx,n in fp.items():rr[idx]=Rotation.from_rotvec(x.reshape(-1,3)[n]).as_matrix()@seed_R[idx]
        return rr
    def residual(x):
        rr=rotations(x);blocks=[x/1.0]
        for a,b,R,t,w in used:blocks.append(np.sqrt(w)*Rotation.from_matrix((rr[b]@rr[a].T)@R.T).as_rotvec()/.08)
        return np.concatenate(blocks)
    if free and used:
        result=least_squares(residual,np.zeros(len(free)*3),loss='soft_l1',max_nfev=80)
        final_R=rotations(result.x);solve={'success':bool(result.success),'nfev':int(result.nfev),'cost':float(result.cost)}
    else:final_R=seed_R;solve={'success':False,'reason':'no_connected_RGB_edges'}
    output=copy.deepcopy(base);new_rows=[]
    frame_map={int(f['frame_idx']):f for f in annotations['frames']}
    for idx in timeline:
        row=copy.deepcopy(original.get(idx,{}));Rbefore,tbefore=poses[idx];R=final_R[idx]
        # Preserve the current source support point under SO3 loop corrections,
        # rather than rotating the object's world-origin translation.
        obj=se3._object_for_frame(frame_map[idx],object_id);points,_,_=se3.p09_geometry_validation(obj,frame_map[idx])
        support=np.mean(points,axis=0) if points is not None else tbefore
        canonical=(support-tbefore)@Rbefore;t=support-R@canonical
        row.update(frame_idx=idx,rotation_world_from_completed_canonical_matrix=R.tolist(),translation_world_m=t.tolist())
        if idx not in original:row['status']='rgb_initialized_uncertain'
        stale = {key: row.pop(key) for key in ('observed_to_mesh_initial', 'observed_to_mesh_final', 'mesh_to_observed_final') if key in row}
        row['source_P14_distance_diagnostics'] = stale
        row['rgb_initializer']={'anchor_connected':idx in connected,'direct_metric_observation_added':False,'source_status':original.get(idx,{}).get('status'),'translation_policy':'RGB_SE3_chain_then_support_preserving_SO3_correction','uncertain':idx not in connected or points is None}
        new_rows.append(row)
    if 'final_observed_to_mesh_median_summary_m' in output:
        output['source_P14_final_distance_summary']=output.pop('final_observed_to_mesh_median_summary_m')
    output['pose_rows']=new_rows
    if 'frame_fit_rows' in output:
        output['source_P14_frame_fit_rows']=output.pop('frame_fit_rows')
        output['frame_fit_rows']=copy.deepcopy(new_rows)
    # P14's current producer names the mesh at top level. P14b consumes an
    # explicit inputs.completed_mesh; normalize that binding, not the geometry.
    mesh_path=base.get('pose_hypothesis_mesh_labeled') or base.get('inputs',{}).get('completed_mesh')
    if mesh_path:
        output['inputs']={**base.get('inputs',{}),'completed_mesh':mesh_path}
    output['source_P14_method']=base.get('method')
    output['method']='build_v19_rgb_pose_initializer'
    output['pose_estimation_method']='quality_weighted_RGB_SE3_spanning_graph_with_robust_SO3_loop_refinement'
    for name in ('observed_pose_evidence_diagnostics','pairwise_observed_surface_registration','rotation_unary_regularization','translation_regularization'):
        if name in output:output['source_P14_'+name]=output.pop(name)
    output['rgb_full_timeline_initialization']={'anchor':anchor,'accepted_edge_count':len(edges),'connected_frame_ids':sorted(connected),'disconnected_frame_ids':sorted(set(timeline)-connected),'loop_optimizer':solve,'generated_geometry_consumed':False,'gt_consumed':False}
    # Retain P14's schema/readiness/observation statuses and source binding for
    # P14b/P15; this refinement must not relabel an inferred pose as a sample.
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['annotations','pose-report','rgb-edge-npz','rgb-edge-json','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--object-id',required=True);p.add_argument('--anchor-frame',type=int,required=True);a=p.parse_args()
    root=a.run_root.resolve()
    lineage=RunLineage(root,Path(__file__).resolve().parents[1],asset_role='pose_initializer')
    for name in ['annotations','pose_report','rgb_edge_npz','rgb_edge_json']:
        lineage.require(getattr(a,name))
    in_run(a.output,root,must_exist=False)
    if a.output.exists():raise ValueError('refusing to overwrite pose initializer')
    base=se3.load_json(a.pose_report);ann=se3.load_json(a.annotations);ed=se3.load_json(a.rgb_edge_json)
    for payload in (base,ann,ed):validate_references(payload,root)
    for name,x in [('P14',base),('RGB',ed)]:assert_prediction_only(x,label=name,object_id=a.object_id)
    with np.load(a.rgb_edge_npz,allow_pickle=False) as z:
        meta=json.loads(str(np.asarray(z['metadata']).reshape(-1)[0]));arrays={k:np.asarray(z[k]) for k in ['source_frame_idx','target_frame_idx','rotation_source_to_target_world','translation_source_to_target_world_m','accepted']}
    for r in [ed,meta]:
        assert_prediction_only(r,label='RGB initializer evidence metadata',object_id=a.object_id)
        if r.get('input_sha256',{}).get('annotations')!=se3.sha256_file(a.annotations) or r.get('input_sha256',{}).get('initial_pose_report')!=se3.sha256_file(a.pose_report):raise ValueError('fresh RGB source binding mismatch')
    result=initialize(base,ann,a.object_id,arrays,ed['rows'],a.anchor_frame)
    result['rgb_full_timeline_initialization']['inputs']={k:{'path':str(getattr(a,k).resolve()),'sha256':se3.sha256_file(getattr(a,k))} for k in ['annotations','pose_report','rgb_edge_npz','rgb_edge_json']}
    if 'output' in result:
        result['output']={**result['output'],'pose_report':str(a.output.resolve())}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(se3.json_safe(result),indent=2,allow_nan=False)+'\n');print(str(a.output))


if __name__=='__main__':main()
