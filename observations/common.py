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

from utils.data import frame_ids, join_indices, validate_times, render_size
from utils.geometry import validate_transforms, resized_K
def annotation_timeline(ann, params, pose_rows, width=None):
    frames = ann["frames"]
    aid = frame_ids([f["frame_idx"] for f in frames], "annotation")
    order = np.argsort([f["time_s"] for f in frames])
    frames = [frames[i] for i in order]
    ids = aid[order]
    times = validate_times([f["time_s"] for f in frames], len(ids))
    zi = join_indices(params["frame_idx"], ids)
    if "time_s" in params:
        parameter_times = np.asarray(params["time_s"], dtype=float)
        if parameter_times.shape != (len(ids),) or not np.allclose(parameter_times[zi], times, atol=1e-6, rtol=0):
            raise ValueError("HaWoR/annotation timestamp mismatch")
    pi = join_indices([r["frame_idx"] for r in pose_rows], ids)
    poses = [pose_rows[i] for i in pi]
    for row, time_s in zip(poses, times):
        if "time_s" in row and not np.isclose(row["time_s"], time_s, atol=1e-6, rtol=0):
            raise ValueError("pose/annotation timestamp mismatch")
    source = np.array([[f["source_width"], f["source_height"]] for f in frames])
    if not np.all(source == source[0]):
        raise ValueError("source video dimensions change across timeline")
    size = render_size(source[0], width)
    K = np.array([resized_K(f["camera"]["intrinsics_fx_fy_cx_cy"], source[0], size) for f in frames])
    T = validate_transforms([f["camera"]["T_world_camera_metric"] for f in frames])
    aligned = {k: np.asarray(v) for k, v in params.items()}
    for side in SIDES:
        for suffix in ("root_orient_axis_angle", "hand_pose_axis_angle", "betas", "trans_world_m", "vertices_world_m", "joints_world_m", "valid"):
            k = side + "_" + suffix
            if k in aligned:
                if len(aligned[k]) != len(zi):
                    raise ValueError(f"{k}: timeline length mismatch")
                aligned[k] = aligned[k][zi]
    aligned["frame_idx"] = ids
    return frames, aligned, poses, times, T, K, size, source[0]

SIDES = ("left", "right")
