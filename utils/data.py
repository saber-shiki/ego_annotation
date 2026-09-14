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

"""CFR-only video timing contracts, using decoded presentation timestamps.

Frame counts and average FPS alone do not establish CFR. Integer PTS are checked
against the exact rational cadence. Fractional tick rounding is allowed only
where that cadence is not exactly representable in the source time base; there
is no half-frame allowance. Annotation timestamps use seconds and must match
the nominal grid within one microsecond of numerical/decimal precision.

This module is stdlib-only so launch preflight can use it before model imports.
"""
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
PACKAGE_TIME_TOLERANCE_S = 1e-06

@dataclass(frozen=True)
class CfrVideoTiming:
    frame_rate: Fraction
    time_base: Fraction
    presentation_ticks: tuple[int, ...]
    last_duration_ticks: int
    source_size_wh: tuple[int, int]
    rate_field: str
    max_grid_error_s: float

    @property
    def fps(self) -> float:
        return float(self.frame_rate)

    @property
    def presentation_times_s(self) -> tuple[float, ...]:
        return tuple((float(t * self.time_base) for t in self.presentation_ticks))

    def report(self) -> dict:
        return {'fps_ratio': str(self.frame_rate), 'fps_source_field': self.rate_field, 'time_base_s': str(self.time_base), 'first_pts_s': self.presentation_times_s[0], 'frame_count': len(self.presentation_ticks), 'duration_s': float((self.presentation_ticks[-1] - self.presentation_ticks[0] + self.last_duration_ticks) * self.time_base), 'max_cfr_grid_error_s': self.max_grid_error_s, 'validation': 'decoded integer PTS against rational CFR grid; only fractional tick quantization allowed'}

def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f'missing or invalid {label}')
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f'missing or invalid {label}') from exc

def _positive_ratio(value, label):
    try:
        ratio = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'invalid {label}') from exc
    if ratio <= 0:
        raise ValueError(f'invalid {label}')
    return ratio

def _frame_pts(frame):
    fields = [frame[k] for k in ('pts', 'pkt_pts') if frame.get(k) is not None]
    if not fields:
        raise ValueError('missing source presentation timestamp')
    values = [_integer(v, 'source presentation timestamp') for v in fields]
    if len(set(values)) != 1:
        raise ValueError('conflicting source presentation timestamp aliases')
    return values[0]

def _tick_matches(ticks, expected):
    lower = expected.numerator // expected.denominator
    upper = -(-expected.numerator // expected.denominator)
    return lower <= ticks <= upper

def read_cfr_video_timing(path, expected_frame_count=None):
    command = ['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0', '-show_frames', '-show_streams', '-show_entries', 'stream=avg_frame_rate,r_frame_rate,time_base,width,height,duration_ts:frame=pts,pkt_pts,duration,pkt_duration', '-of', 'json', str(Path(path))]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    payload = json.loads(result.stdout)
    streams = payload.get('streams', [])
    frames = payload.get('frames', [])
    if len(streams) != 1 or not frames:
        raise ValueError('source presentation timestamps require one nonempty video stream')
    if expected_frame_count is not None and len(frames) != expected_frame_count:
        raise ValueError('video/package decoded presentation frame count mismatch')
    stream = streams[0]
    rate_field = 'avg_frame_rate'
    if stream.get(rate_field) in (None, '0/0', '0/1'):
        rate_field = 'r_frame_rate'
    rate = _positive_ratio(stream.get(rate_field), 'source video frame rate')
    time_base = _positive_ratio(stream.get('time_base'), 'source video time base')
    pts = tuple((_frame_pts(f) for f in frames))
    if any((b <= a for (a, b) in zip(pts, pts[1:]))):
        raise ValueError('source presentation timestamps must be strictly increasing')
    step = 1 / (rate * time_base)
    errors = []
    for (i, value) in enumerate(pts):
        relative = value - pts[0]
        expected = i * step
        if not _tick_matches(relative, expected):
            raise ValueError(f'source presentation timestamps are not CFR at frame {i}; VFR is unsupported')
        errors.append(float(abs(relative - expected) * time_base))
    last_duration = frames[-1].get('duration', frames[-1].get('pkt_duration'))
    if last_duration is None:
        total = _integer(stream.get('duration_ts'), 'source video duration')
        last_duration = total - (pts[-1] - pts[0])
    last_duration = _integer(last_duration, 'source final frame duration')
    if last_duration <= 0 or not _tick_matches(last_duration, step):
        raise ValueError('source final frame duration does not match CFR cadence')
    width = _integer(stream.get('width'), 'source width')
    height = _integer(stream.get('height'), 'source height')
    if width <= 0 or height <= 0:
        raise ValueError('invalid source video dimensions')
    return CfrVideoTiming(rate, time_base, pts, last_duration, (width, height), rate_field, max(errors))

def validate_package_cfr_times(times, frame_rate):
    values = tuple((float(t) for t in times))
    if not values or not all((math.isfinite(t) for t in values)):
        raise ValueError('invalid package timestamps')
    if any((b <= a for (a, b) in zip(values, values[1:]))):
        raise ValueError('package timestamps must be strictly increasing')
    errors = [abs(value - values[0] - float(Fraction(i) / frame_rate)) for (i, value) in enumerate(values)]
    if max(errors) > PACKAGE_TIME_TOLERANCE_S:
        raise ValueError('package constant-rate timeline mismatch; VFR is unsupported')
    return max(errors)

def write_json(path: Path, data: Any) -> None:
    from typing import Any
    from pathlib import Path
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def load_json(path: Path) -> Any:
    from typing import Any
    from pathlib import Path
    import json
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    from typing import Any
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {'count': 0, 'median': None, 'p90': None, 'p95': None, 'max': None, 'mean': None}
    return {'count': int(arr.size), 'median': float(np.median(arr)), 'p90': float(np.percentile(arr, 90)), 'p95': float(np.percentile(arr, 95)), 'max': float(np.max(arr)), 'mean': float(np.mean(arr))}
from typing import Any
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class FrameHandRow:
    frame_idx: int
    side: str
    frame: dict[str, Any]
    current_vertices_world: np.ndarray
    current_joints_world: np.ndarray
    raw_vertices_world: np.ndarray
    raw_joints_world: np.ndarray
    root_orient_axis_angle: np.ndarray
    hand_pose_axis_angle: np.ndarray
    betas: np.ndarray
    trans_world_m: np.ndarray
    similarity_scale: float
    similarity_rotation_raw_to_current: np.ndarray
    similarity_translation_raw_to_current: np.ndarray
    source_hawor_npz: Path
    source_frame_index: int
    constraint_indices: np.ndarray
    constraint_normals_world: np.ndarray
    constraint_depths_m: np.ndarray
    observed_initial_measure: dict[str, Any]
    observed_constraint_count: int
    object_rotation_world_from_object: np.ndarray
    object_translation_world_m: np.ndarray
    face_strict_observed_raw: np.ndarray
    face_strict_observed: np.ndarray
    signed_face_authority_required: bool
    signed_face_authority_state: str
    signed_face_authority_face_count: int
    signed_face_authority_eligible_face_count: int
    hand_owned_quarantined_face_count: int
    surface_eligibility_npz_path: str | None
    surface_eligibility_mode: str | None
    observed_surface_support_uncertainty_m: float
    surface_eligible_face_count: int
    surface_input_face_count: int
    surface_applied_face_delta: int
    visible_ownership_non_object_mask_path: str | None
    visible_ownership_object_owned_mask_path: str | None
    visible_ownership_constraint_eligible_mask_path: str | None
    visible_ownership_non_object_owned_px: int
    visible_ownership_object_owned_px: int
    visible_ownership_constraint_eligible_px: int
    visible_ownership_quarantined_face_count: int
    visible_mask_A_mask_from_source: np.ndarray
    visible_mask_image_plane_contract: dict[str, Any]
    visible_object_mask_path: str | None
    visible_object_mask_face_count_raw: int
    visible_object_mask_face_count: int
    visible_surface_track_factor_state: str | None
    visible_surface_track_mask_path: str | None
    visible_surface_track_npz_path: str | None
    visible_surface_track_valid_depth_pixels: int
    visible_surface_track_quarantined_face_count: int
    visible_surface_depth_order_vertex_indices: np.ndarray
    visible_surface_depth_order_depth_m: np.ndarray
    visible_surface_depth_order_initial_delta_m: np.ndarray
    visible_surface_depth_order_initial_measure: dict[str, Any]
    visible_surface_translation_support_vertex_indices: np.ndarray
    visible_surface_translation_support_depth_m: np.ndarray
    visible_surface_translation_support_initial_delta_m: np.ndarray
    visible_surface_translation_support_measure: dict[str, Any]
    hand_observation_visibility_factor_state: str | None
    hand_observation_visibility_candidate_px: int
    hand_observation_visibility_weight_multiplier: float
    contact_patch_factor_state: str | None
    contact_patch_vertex_indices: np.ndarray
    contact_patch_target_world_m: np.ndarray
    contact_patch_normal_world: np.ndarray
    contact_patch_initial_distance_m: np.ndarray
    contact_patch_weight: float
    contact_patch_prior_probability: float
    contact_patch_band_m: float
    contact_patch_target_margin_m: float
    contact_patch_support_uncertainty_m: float
    contact_patch_support_uncertainty_source: str | None
    local_patch_support_state: str | None
    local_patch_support_consumed: bool
    local_patch_support_uncertainty_m: float | None
    global_object_support_uncertainty_m: float | None
    local_patch_sample_count: int
    local_patch_temporal_sample_count: int
    contact_anchor_state: str | None
    contact_anchor_residual_allowed: bool
    contact_anchor_blockers: list[str]
    contact_pose_anchor_key: str | None
    joint_visibility_weights: np.ndarray
    joint_depth_residual_m: np.ndarray
    hand_ray_shift_prior_world_m: np.ndarray
    hand_ray_shift_prior_source_m: float | None
    hand_ray_shift_prior_weight: float

def as_list(value: Any) -> list[Any]:
    from typing import Any
    return value if isinstance(value, list) else []

def pose_map(pose_report: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    from typing import Any
    import numpy as np
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in as_list(pose_report.get('pose_rows')):
        if not isinstance(row, dict):
            continue
        status = str(row.get('status') or '')
        if status.startswith('fit_to_visible_depth') and status not in ACCEPTED_VISIBLE_DEPTH_POSE_STATUSES:
            raise ValueError(f"unrecognized visible-depth pose status {status!r} for frame {row.get('frame_idx')}")
        if status not in ACCEPTED_VISIBLE_DEPTH_POSE_STATUSES:
            continue
        r = np.asarray(row.get('rotation_world_from_completed_canonical_matrix') or [], dtype=float)
        t = np.asarray(row.get('translation_world_m') or [], dtype=float)
        if r.shape != (3, 3) or t.shape != (3,):
            raise ValueError(f"pose row frame={row.get('frame_idx')} status={status!r} has invalid pose shapes rotation={r.shape} translation={t.shape}")
        out[int(row['frame_idx'])] = (r, t)
    return out
ACCEPTED_VISIBLE_DEPTH_POSE_STATUSES = {'fit_to_visible_depth_samples', 'fit_to_visible_depth_archive_vertices', 'corrected_temporal_rigid_pose_graph', 'completed_temporal_rigid_pose_uncertain'}

def load_mesh(path: Path) -> trimesh.Trimesh:
    import numpy as np
    from pathlib import Path
    import trimesh
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [mesh for mesh in geom.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f'No mesh geometry in {path}')
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f'Unsupported mesh geometry from {path}: {type(geom)}')
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)
