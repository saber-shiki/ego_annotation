"""Validated frame joins and image planes for the additive MANO/mesh demo."""
import numpy as np
from v20_demo_geometry import resized_K

SIDES = ("left", "right")


def frame_ids(values, label="frame_idx"):
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{label}: expected nonempty integer IDs")
    ids = raw.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{label}: duplicate frame IDs")
    return ids


def join_indices(source_ids, target_ids):
    source = frame_ids(source_ids)
    target = frame_ids(target_ids)
    lookup = {int(f): i for i, f in enumerate(source)}
    if set(source) != set(target):
        raise ValueError("frame ID coverage mismatch")
    return np.array([lookup[int(f)] for f in target], dtype=int)


def validate_times(values, count):
    times = np.asarray(values, dtype=float)
    if times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    return times


def validate_transforms(values):
    T = np.asarray(values, dtype=float)
    if T.ndim != 3 or T.shape[1:] != (4, 4) or not np.isfinite(T).all():
        raise ValueError("invalid camera transforms")
    R = T[:, :3, :3]
    if not (np.allclose(T[:, 3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-4)
            and np.allclose(np.linalg.det(R), 1, atol=1e-4)):
        raise ValueError("camera transforms must be rigid world-from-camera")
    return T


def render_size(source_wh, width=None):
    wh = np.asarray(source_wh)
    if wh.shape != (2,) or not np.isfinite(wh).all() or np.any(wh <= 0) or np.any(wh != wh.astype(int)):
        raise ValueError("invalid image size")
    if width is None:
        return wh.astype(int)
    if width < 2:
        raise ValueError("render width must be at least 2")
    return np.array([width, max(1, round(wh[1] * width / wh[0]))], dtype=int)


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


def hand_validity(z, side, count, fitted=False):
    suffixes = ("vertices_world",) if fitted else ("vertices_world_m", "joints_world_m", "root_orient_axis_angle", "hand_pose_axis_angle", "betas", "trans_world_m")
    keys = [side + "_" + s for s in suffixes]
    present = [k in z for k in keys]
    raw = np.asarray(z.get(side + "_valid", np.ones(count) if all(present) else np.zeros(count)))
    if raw.shape != (count,) or not np.isin(raw, [0, 1]).all():
        raise ValueError(f"{side}: invalid validity mask")
    valid = raw.astype(bool)
    if not all(present):
        if valid.any():
            raise ValueError(f"{side}: valid hand lacks MANO state")
        return valid
    for k in keys:
        value = np.asarray(z[k])
        if len(value) != count:
            raise ValueError(f"{k}: timeline length mismatch")
        valid &= np.isfinite(value.reshape(count, -1)).all(axis=1)
    return valid


def supported_segments(valid):
    indices = np.flatnonzero(valid)
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []


def fit_keyframes(count, valid, stride=5):
    keys = [0, count - 1, *range(0, count, stride)]
    for mask in valid.values():
        for segment in supported_segments(mask):
            keys.extend([segment[0], segment[-1]])
    return np.unique(keys)


def interpolate_supported(times, keys, values, valid):
    """Zero correction outside observed segments; never bridge a missing-hand gap."""
    full = np.zeros((len(times), *values.shape[1:]), dtype=values.dtype)
    for segment in supported_segments(valid):
        chosen = np.flatnonzero(np.isin(keys, segment))
        if not len(chosen):
            continue
        flat = values[chosen].reshape(len(chosen), -1)
        full[segment] = np.stack([np.interp(times[segment], times[keys[chosen]], flat[:, j]) for j in range(flat.shape[1])], axis=1).reshape(len(segment), *values.shape[1:])
    return full


def interaction_states(payload, ids, object_id):
    states = np.full((len(ids), 2), "legacy_visual_hypothesis", dtype="U32")
    if payload is None:
        return states
    if not isinstance(payload, dict):
        raise ValueError("interaction judgments payload is not an object")
    ids = frame_ids(ids)
    if payload.get("status") not in (None, "ok"):
        raise ValueError("interaction judgments status is not ok")
    rows = payload.get("interaction_judgments", payload.get("judgments", payload.get("segments")))
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing interaction_judgments")
    states[:] = "unresolved"
    occupied = np.zeros(states.shape, bool)
    matched_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("interaction judgment segment is not an object")
        target = row.get("target_entity_id", row.get("object_id", row.get("track_id")))
        if target is not None and not isinstance(target, str):
            raise ValueError("interaction judgment target must be a string")
        if target and target.split(":", 1)[-1] != object_id.split(":", 1)[-1]:
            continue
        side = row.get("hand_side", row.get("side"))
        if side not in SIDES:
            raise ValueError("interaction judgment side must be left or right")
        state = row.get("contact_state")
        if not isinstance(state, str) or state not in {"likely_contact", "possible_contact", "no_contact", "unresolved"}:
            raise ValueError("invalid interaction contact_state")
        bounds = [row.get("frame_start", row.get("start_frame", int(min(ids)))),
                  row.get("frame_end", row.get("end_frame", int(max(ids))))]
        try:
            start, end = (int(value) for value in bounds)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("interaction interval bounds must be integer frame IDs") from exc
        if any(isinstance(value, bool) or (isinstance(value, float) and value != int(value)) for value in bounds):
            raise ValueError("interaction interval bounds must be integer frame IDs")
        if end < start:
            raise ValueError("inverted interaction interval")
        mask = (ids >= start) & (ids <= end)
        si = SIDES.index(side)
        if occupied[mask, si].any():
            raise ValueError("overlapping interaction judgments")
        occupied[mask, si] = True
        states[mask, si] = state
        matched_rows += int(mask.any())
    if not matched_rows:
        raise ValueError(f"interaction judgments contained no segments for target {object_id!r} on this timeline")
    return states


def state_timeline(z, fps, video_count, width=None):
    ids = frame_ids(z["frame_idx"])
    if not np.isfinite(fps) or fps <= 0 or video_count != len(ids):
        raise ValueError("source video/state timeline mismatch")
    legacy = "time_s" not in z
    if legacy and not np.array_equal(np.sort(ids), np.arange(len(ids))):
        raise ValueError("legacy state needs zero-based source frame IDs")
    times = np.asarray(z["time_s"] if not legacy else ids / fps, float)
    if times.shape != (len(ids),) or not np.isfinite(times).all():
        raise ValueError("invalid state time_s")
    order = np.argsort(times)
    times_sorted = validate_times(times[order], len(ids))
    # The MP4 writer is constant-rate. Reject an ambiguous/VFR mapping rather than seek by IDs.
    if not np.allclose(times_sorted - times_sorted[0], np.arange(len(ids)) / fps, atol=0.5 / fps, rtol=0):
        raise ValueError("timestamps do not map one-to-one to source video frames")
    source_size = np.asarray(z.get("render_size_wh", [960, 960]))
    size = render_size(source_size, width)
    intr = np.asarray(z["intrinsics_render"] if "intrinsics_render" in z else z["intrinsics_960"])
    if intr.shape != (len(ids), 4):
        raise ValueError("intrinsics timeline mismatch")
    K = np.array([resized_K(k, source_size, size) for k in intr])
    T = validate_transforms(z["T_world_camera"])
    if len(T) != len(ids):
        raise ValueError("camera timeline mismatch")
    if "object_vertices_canonical" in z:
        vertices = np.asarray(z["object_vertices_canonical"])
        faces = np.asarray(z["object_faces"])
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
                or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
                or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("intact object triangle geometry required")
        for prefix in ("", "base_"):
            if prefix and prefix + "object_rotation" not in z:
                continue
            R = np.asarray(z[prefix + "object_rotation"])
            t = np.asarray(z[prefix + "object_translation_world"])
            if R.shape != (len(ids), 3, 3) or t.shape != (len(ids), 3):
                raise ValueError("full-timeline object poses required")
            object_T = np.tile(np.eye(4), (len(ids), 1, 1))
            object_T[:, :3, :3], object_T[:, :3, 3] = R, t
            validate_transforms(object_T)
    return ids, times, order, size, K
