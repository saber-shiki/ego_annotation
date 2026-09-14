"""Shared typed numeric records and JSON IO; no business dependencies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import numpy as np
import json

@dataclass
class PoseNode:
    frame_idx: int
    initial_rotation: np.ndarray
    initial_translation: np.ndarray
    observed_world: np.ndarray | None
    observation_source: str
    metric_observation: bool
    uncertainty: float
    source_row: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RGBEdge:
    source_frame_idx: int
    target_frame_idx: int
    rotation_source_to_target_world: np.ndarray
    translation_source_to_target_world_m: np.ndarray
    weight: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    translation_reference_rotation: np.ndarray | None = None
    translation_source_support_world: np.ndarray | None = None


@dataclass(frozen=True)
class RGBReprojectionFactor:
    source_frame_idx: int
    target_frame_idx: int
    source_pos: int
    target_pos: int
    source_world: np.ndarray
    target_uv: np.ndarray
    target_intrinsics: np.ndarray
    target_camera_rotation: np.ndarray
    target_camera_translation: np.ndarray
    weights: np.ndarray
    edge_weight: float
    rotation_eligible: bool = True


@dataclass(frozen=True)
class PointFactor:
    source_pos: int
    target_pos: int
    source_world: np.ndarray
    target_world: np.ndarray
    target_normals: np.ndarray
    weights: np.ndarray
    kind: str = "p09_observed_rebuilt"


@dataclass
class SolverConfig:
    """Only the residual/correspondence options consumed by the extracted kernels."""
    max_points: int = 350
    min_pairs: int = 12
    max_correspondence_m: float = 0.08
    observed_factor_mode: str = "legacy_nn"
    rgb_reprojection_weight_scale: float = 0.0
    sigma_rgb_reprojection_px: float = 2.0
    max_rgb_reprojection_residual_px: float = 40.0
    seed: int = 20260907


def load(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


RKEY = 'rotation_world_from_completed_canonical_matrix'


TKEY = 'translation_world_m'


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

from utils.geometry import resized_K, validate_transforms
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


def render_size(source_wh, width=None):
    wh = np.asarray(source_wh)
    if wh.shape != (2,) or not np.isfinite(wh).all() or np.any(wh <= 0) or np.any(wh != wh.astype(int)):
        raise ValueError("invalid image size")
    if width is None:
        return wh.astype(int)
    if width < 2:
        raise ValueError("render width must be at least 2")
    return np.array([width, max(1, round(wh[1] * width / wh[0]))], dtype=int)


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
