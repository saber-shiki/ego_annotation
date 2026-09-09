"""Current-run pose IO and time-based evidence planning; no historical pose inputs."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from rgb_scene_pose_kernels import PoseNode, RGBEdge, RGBReprojectionFactor, p09_geometry_validation, valid_rotation
from rgb_scene_pose_graph import solve, GraphConfig
from v19_runtime_contracts import sha, load, in_run
from v20_prediction_contracts import assert_prediction_only

RKEY = 'rotation_world_from_completed_canonical_matrix'
TKEY = 'translation_world_m'


def pose_from_values(rotation, translation):
    if rotation is None and translation is None:
        return None
    r, t = np.asarray(rotation, float), np.asarray(translation, float)
    if not valid_rotation(r) or t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError('malformed declared pose')
    return r, t


def indexed(rows):
    result = {}
    for row in rows:
        idx = row['frame_idx']
        if type(idx) is not int or idx in result:
            raise ValueError('duplicate/noninteger frame ID')
        result[idx] = row
    return result


def time_keyframes(ids, times, spacing):
    if not ids or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('positive time spacing required')
    keys = [ids[0]]
    for idx in ids[1:]:
        if times[idx] - times[keys[-1]] >= spacing:
            keys.append(idx)
    if ids[-1] != keys[-1]:
        keys.append(ids[-1])
    return keys


def time_source_pairs(ids, keys, sources, span, times, max_gap):
    if span < 1 or max_gap <= 0:
        raise ValueError('positive pair planning span/gap required')
    planned = set(zip(ids[:-1], ids[1:]))
    for hop in range(1, span + 1):
        planned.update(zip(keys[:-hop], keys[hop:]))
    if len(keys) > 2:
        planned.update([(keys[0], keys[-1]), (keys[0], keys[len(keys)//2]), (keys[len(keys)//2], keys[-1])])
    pairs, rejected = set(), []
    for a, b in planned:
        if a in sources: pairs.add((a, b))
        elif b in sources: pairs.add((b, a))
        else: rejected.append({'source_frame_idx': a, 'target_frame_idx': b, 'reason': 'neither_endpoint_has_retained_depth'})
    for b in ids:
        if b in sources: continue
        before = [a for a in sources if 0 < times[b] - times[a] <= max_gap]
        after = [a for a in sources if 0 < times[a] - times[b] <= max_gap]
        if before: pairs.add((max(before, key=times.get), b))
        if after: pairs.add((min(after, key=times.get), b))
    return sorted(pairs, key=lambda p: (times[p[0]], times[p[1]])), rejected


def build_nodes(annotations, source, object_id):
    frames = sorted(indexed(annotations['frames']).values(), key=lambda f: f['time_s'])
    rows = indexed(source['pose_rows'])
    if set(rows) - {f['frame_idx'] for f in frames}:
        raise ValueError('source pose outside annotation timeline')
    known = {i: p for i, row in rows.items() if (p := pose_from_values(row.get(RKEY), row.get(TKEY))) is not None}
    if not known: raise ValueError('no source pose establishes a canonical gauge')
    times = {f['frame_idx']: float(f['time_s']) for f in frames}
    ordered = sorted(known, key=times.get)
    nodes = []
    for f in frames:
        idx = f['frame_idx']; objects = [o for o in f.get('objects', []) if o['object_id'] == object_id]
        if len(objects) > 1: raise ValueError('duplicate selected object')
        obj = objects[0] if objects else None
        points, reason, diagnostics = p09_geometry_validation(obj, f)
        # Unverified ownership never licenses a depth source; appearance is separate.
        if obj is None or obj.get('rigid_pose_observation_eligible') is not True or (obj.get('visible_geometry_candidate') or {}).get('rigid_pose_observation_eligible') is not True:
            points = None
        if idx in known:
            r, t = known[idx]
        else:
            before = [k for k in ordered if times[k] <= times[idx]]
            after = [k for k in ordered if times[k] >= times[idx]]
            a, b = (before[-1] if before else ordered[0]), (after[0] if after else ordered[-1])
            if a == b: r, t = known[a]
            else:
                alpha = (times[idx] - times[a]) / (times[b] - times[a])
                r = Slerp([times[a], times[b]], Rotation.from_matrix([known[a][0], known[b][0]]))([times[idx]]).as_matrix()[0]
                t = (1-alpha)*known[a][1] + alpha*known[b][1]
        nodes.append(PoseNode(idx, r.copy(), t.copy(), points, reason, points is not None,
                              .1 if idx in known and points is not None else .95,
                              copy.deepcopy(rows.get(idx, {'frame_idx': idx}))))
    return nodes, np.array([times[n.frame_idx] for n in nodes])


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
    poses,diagnostics=solve(nodes,times,edges,anchor,reprojections=reprojections)
    rows=[]
    for n,stamp in zip(nodes,times):
        r,t=poses[n.frame_idx]
        rows.append({'frame_idx':n.frame_idx,'time_s':float(stamp),RKEY:r.tolist(),TKEY:t.tolist(),
                     'pose_source':'global_SO3_observed_RGB_SE3_visual_posterior',
                     'metric_observation':n.metric_observation,
                     'uncertainty':max(n.uncertainty, .6 if n.frame_idx in diagnostics['relative_evidence_disconnected_ids'] else .2),
                     'uncertainty_kind':'heuristic_support_uncertainty_not_marginal_covariance',
                     'relative_evidence_connected':n.frame_idx not in diagnostics['relative_evidence_disconnected_ids'],
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('annotations','source-pose','rgb-edges','rgb-report','output','run-root'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--object-id',required=True);p.add_argument('--anchor',type=int,required=True)
    a=p.parse_args();refine(a.annotations,a.source_pose,a.rgb_edges,a.rgb_report,a.object_id,a.anchor,a.output,a.run_root)

if __name__=='__main__':main()
