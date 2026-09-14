"""Assemble qualified current-run metric observations on the complete timeline."""
from __future__ import annotations
import copy
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from utils.data import PoseNode, indexed, RKEY, TKEY
from utils.geometry import valid_rotation
from gates.object.observation import p09_geometry_validation

def pose_from_values(rotation, translation):
    if rotation is None and translation is None:
        return None
    r, t = np.asarray(rotation, float), np.asarray(translation, float)
    if not valid_rotation(r) or t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError('malformed declared pose')
    return r, t


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
