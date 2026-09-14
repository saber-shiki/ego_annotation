"""Grouped prediction entry: prepared metric/RGB trajectory refinement."""
from __future__ import annotations
import argparse, copy, json, math
from pathlib import Path
import numpy as np
from utils.data import RGBEdge, RGBReprojectionFactor, indexed, RKEY, TKEY, load
from utils.state import sha, in_run, assert_prediction_only
from observations.common import build_nodes, pose_from_values
from optimization.problems import StandaloneScene
from optimization.solve import solve


def load_edges(npz_path, report_path, source_path, annotations_path, nodes, root):
    report = load(report_path)
    with np.load(npz_path, allow_pickle=False) as z: data = dict(z)
    meta = json.loads(str(data['metadata'].reshape(-1)[0]))
    for key, path in [('initial_pose_report', source_path), ('annotations', annotations_path)]:
        if meta['input_sha256'][key] != sha(path) or Path(meta[key]).resolve() != Path(path).resolve():
            raise ValueError('current RGB source hash/identity mismatch')
    for key in ('depth_npz',):
        path = in_run(meta[key], root)
        if sha(path) != meta['input_sha256'][key]: raise ValueError('RGB metric source mutated')
    positions = {n.frame_idx: i for i,n in enumerate(nodes)}
    source_rows = indexed(load(source_path)['pose_rows'])
    edges, projections = [], []
    offsets = data['reprojection_evidence_offsets']
    if len(offsets) != len(data['source_frame_idx'])+1 or offsets[0] != 0 or offsets[-1] != len(data['reprojection_canonical_points']) or np.any(np.diff(offsets)<0):
        raise ValueError('invalid RGB evidence offsets')
    rows = {(r['source_frame_idx'],r['target_frame_idx']):r for r in report['rows']}
    for k, (a,b) in enumerate(zip(data['source_frame_idx'],data['target_frame_idx'])):
        a,b=int(a),int(b)
        if not data['accepted'][k]: continue
        row=rows[a,b]
        q=float(row['inlier_fraction'])*math.exp(-float(row['reprojection_median_px'])/2.)
        q *= max(.02,min(1.,float(row['source_3d_conditioning'])/.02))
        q *= .45 if row['topological_role']=='adjacent' else 1.
        e=RGBEdge(a,b,data['rotation_source_to_target_world'][k],data['translation_source_to_target_world_m'][k],max(.001,q),row)
        edges.append(e)
        lo,hi=map(int,offsets[k:k+2])
        if hi>lo:
            # Producer serialized canonical coordinates. Recover FIXED observed world
            # points once from its hash-bound source, never from the optimizer pose.
            r,t=pose_from_values(source_rows[a][RKEY],source_rows[a][TKEY])
            world=data['reprojection_canonical_points'][lo:hi]@r.T+t
            take=np.linspace(0,len(world)-1,min(96,len(world)),dtype=int)
            camera=data['reprojection_target_camera'][k]
            projections.append(RGBReprojectionFactor(a,b,positions[a],positions[b],world[take],
                data['reprojection_target_uv'][lo:hi][take],data['reprojection_target_intrinsics'][k],
                camera[:3,:3],camera[:3,3],data['reprojection_weights'][lo:hi][take],e.weight))
    return edges,projections


def refine(annotations_path, source_path, edge_path, edge_report, object_id, anchor, output, root):
    if output.exists(): raise ValueError('create-only posterior required')
    before=sha(source_path)
    source=load(source_path); annotations=load(annotations_path)
    assert_prediction_only(source, label='scene source prior', object_id=object_id)
    assert_prediction_only(annotations, label='scene observed geometry', object_id=object_id)
    nodes,times=build_nodes(annotations,source,object_id)
    if not any(n.frame_idx==anchor and n.metric_observation for n in nodes):
        raise ValueError('gauge anchor must have current retained metric support')
    edges,reprojections=load_edges(edge_path,edge_report,source_path,annotations_path,nodes,root)
    poses,diagnostics=solve(StandaloneScene(nodes,times,edges,anchor,reprojections=reprojections))
    seed_rows = {row['frame_idx']: row for row in diagnostics['local_RGB_rotation']['seed_rows']}
    rows=[]
    for n,stamp in zip(nodes,times):
        r,t=poses[n.frame_idx]
        rows.append({'frame_idx':n.frame_idx,'time_s':float(stamp),RKEY:r.tolist(),TKEY:t.tolist(),
                     'pose_source':'global_SO3_observed_RGB_SE3_visual_posterior',
                     'metric_observation':n.metric_observation,
                     'depth_feasibility_uncertain':n.frame_idx in diagnostics['depth_restoration']['affected_frame_ids'],
                     'uncertainty':max(seed_rows[n.frame_idx]['uncertainty'], .9 if n.frame_idx in diagnostics['depth_restoration']['affected_frame_ids'] else 0., n.uncertainty, .6 if n.frame_idx in diagnostics['relative_evidence_disconnected_ids'] else .2),
                     'uncertainty_kind':'heuristic_support_uncertainty_not_marginal_covariance',
                     'relative_evidence_connected':n.frame_idx not in diagnostics['relative_evidence_disconnected_ids'],
                     'local_RGB_seed':seed_rows[n.frame_idx],
                     'source_prior':n.source_row})
    report={'schema':'rgb_scene_refined_pose_v1','object_id':object_id,
            'prediction_side_only':True,'annotation_ready':False,'physical_contact_authority':False,
            'canonical_gauge':{'name':'initial_completed_canonical','anchor_frame_idx':anchor,'anchor_reason':'current_visual_selected_retained_metric_source'},
            'source_prior':{'path':str(source_path),'sha256':before},
            'evidence':{'annotations_sha256':sha(annotations_path),'rgb_npz_sha256':sha(edge_path),'rgb_report_sha256':sha(edge_report)},
            'source_pose_rows':copy.deepcopy(source['pose_rows']),'pose_rows':rows,'solver':diagnostics}
    if sha(source_path)!=before: raise ValueError('source prior mutated')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    return report



def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command', required=True)
    pose=commands.add_parser('refine-pose', help='refine current-run metric and RGB evidence')
    for name in ('annotations','source-pose','rgb-edges','rgb-report','output','run-root'):
        pose.add_argument('--'+name,type=Path,required=True)
    pose.add_argument('--object-id',required=True)
    pose.add_argument('--anchor',type=int,required=True)
    args=parser.parse_args(argv)
    return refine(args.annotations,args.source_pose,args.rgb_edges,args.rgb_report,
                  args.object_id,args.anchor,args.output,args.run_root)

if __name__=='__main__':
    main()
