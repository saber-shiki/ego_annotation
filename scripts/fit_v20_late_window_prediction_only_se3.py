#!/usr/bin/env python3
"""Prediction-only full-timeline SE(3) repair for a configurable late window.

This is a diagnostic solver.  It optimizes one six degree-of-freedom correction
for every timeline row in the selected window (including rows without a metric
P09 observation), while copying rows outside that window byte-for-byte at the
JSON value level.  RGB/PnP edges and P09 visible points are prediction-side
measurements.  An optional generated mesh is loaded only by the validation
path; no generated factor is ever passed to :func:`pose_residual`.

The pose convention is ``p_world = p_canonical @ R.T + t``.  Corrections use
``R = Exp(delta_rotation) @ R_initial`` and ``t = t_initial +
delta_translation``.  The selected anchor is the sole gauge constraint.  In
particular, there is deliberately no small cumulative-rotation correction cap.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from v20_prediction_contracts import append_prediction_stage, assert_prediction_only, require_equal_object_ids
except ModuleNotFoundError:  # pragma: no cover - direct source import support
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v20_prediction_contracts import append_prediction_stage, assert_prediction_only, require_equal_object_ids


def json_safe(value: Any) -> Any:
    """Convert numpy/scipy scalar containers to strict JSON primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


import v20_rgbd_material_factors as material


# ---------------------------------------------------------------------------
# Small, dependency-light SE(3) and coordinate-contract primitives.


def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("array contains non-finite values")
    return result


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = _array(vector, (3,)).tolist()
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def valid_rotation(value: Any) -> bool:
    try:
        matrix = _array(value, (3, 3))
    except (TypeError, ValueError):
        return False
    return bool(np.linalg.det(matrix) > 0.0 and np.linalg.norm(matrix.T @ matrix - np.eye(3)) < 1.0e-3)


def pose_from_values(rotation: Any, translation: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        R = _array(rotation, (3, 3))
        t = _array(translation, (3,))
    except (TypeError, ValueError):
        return None
    if not valid_rotation(R):
        return None
    # Project tiny serialization noise back onto SO(3), but never repair a
    # substantially invalid pose silently.
    R = Rotation.from_matrix(R).as_matrix()
    return R, t


def apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = _array(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    return points @ _array(rotation, (3, 3)).T + _array(translation, (3,))[None, :]


def inverse_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = _array(points)
    return (points - _array(translation, (3,))[None, :]) @ _array(rotation, (3, 3))


def camera_to_world(camera_points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    """Convert camera-frame row vectors to world exactly once."""
    T = _array(T_world_camera, (4, 4))
    return apply_pose(camera_points, T[:3, :3], T[:3, 3])


def world_to_camera(world_points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    """Convert world-frame row vectors to camera exactly once."""
    T = _array(T_world_camera, (4, 4))
    return inverse_pose(world_points, T[:3, :3], T[:3, 3])


def _so3_left_jacobian(rotvec: np.ndarray) -> np.ndarray:
    w = _array(rotvec, (3,))
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1.0e-7:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    return np.eye(3) + ((1.0 - math.cos(theta)) / theta**2) * K + ((theta - math.sin(theta)) / theta**3) * (K @ K)


def _so3_left_jacobian_inverse(rotvec: np.ndarray) -> np.ndarray:
    w = _array(rotvec, (3,))
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1.0e-5:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    half = 0.5 * theta
    coefficient = (1.0 - half / math.tan(half)) / theta**2
    return np.eye(3) - 0.5 * K + coefficient * (K @ K)


def se3_log(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Lie-algebra twist of a row-vector pose transform."""
    w = Rotation.from_matrix(_array(R, (3, 3))).as_rotvec()
    return w, _so3_left_jacobian_inverse(w) @ _array(t, (3,))


def se3_exp(rotvec: np.ndarray, twist_translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = _array(rotvec, (3,))
    v = _array(twist_translation, (3,))
    R = Rotation.from_rotvec(w).as_matrix()
    return R, _so3_left_jacobian(w) @ v


def compose_pose(R0: np.ndarray, t0: np.ndarray, R1: np.ndarray, t1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R0 = _array(R0, (3, 3)); t0 = _array(t0, (3,))
    return R0 @ _array(R1, (3, 3)), R0 @ _array(t1, (3,)) + t0


def inverse_transform(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = _array(R, (3, 3)); t = _array(t, (3,))
    return R.T, -R.T @ t


def interpolate_se3_pose(
    R0: np.ndarray,
    t0: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate two poses along the constant-twist SE(3) geodesic."""
    a = float(np.clip(alpha, 0.0, 1.0))
    R0 = _array(R0, (3, 3)); t0 = _array(t0, (3,))
    R1 = _array(R1, (3, 3)); t1 = _array(t1, (3,))
    R0_inv, t0_inv = inverse_transform(R0, t0)
    R_rel, t_rel = compose_pose(R0_inv, t0_inv, R1, t1)
    w, v = se3_log(R_rel, t_rel)
    R_step, t_step = se3_exp(a * w, a * v)
    return compose_pose(R0, t0, R_step, t_step)


# Short aliases are intentionally public: they make the coordinate and
# interpolation contracts straightforward to exercise without running main.
se3_interpolate = interpolate_se3_pose


def initialize_missing_se3_pose(frame_idx: int, direct_poses: Mapping[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, str]:
    """Initialize a missing timeline row by neighboring SE(3) interpolation."""
    if frame_idx in direct_poses:
        R, t = direct_poses[frame_idx]
        return R.copy(), t.copy(), "direct_metric_pose"
    ids = sorted(int(i) for i in direct_poses)
    if not ids:
        raise ValueError("cannot interpolate without at least one valid pose")
    lower = [i for i in ids if i < int(frame_idx)]
    upper = [i for i in ids if i > int(frame_idx)]
    if lower and upper:
        lo, hi = lower[-1], upper[0]
        alpha = (float(frame_idx) - lo) / float(hi - lo)
        R, t = interpolate_se3_pose(*direct_poses[lo], *direct_poses[hi], alpha)
        return R, t, f"continuous_se3_interpolation_{lo}_{hi}"
    nearest = min(ids, key=lambda value: abs(value - int(frame_idx)))
    R, t = direct_poses[nearest]
    return R.copy(), t.copy(), f"se3_nearest_hold_{nearest}"


def apply_left_correction(
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    delta_rotation: np.ndarray,
    delta_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the V20 additive-translation/left-SO(3) correction contract."""
    return (
        Rotation.from_rotvec(_array(delta_rotation, (3,))).as_matrix()
        @ _array(initial_rotation, (3, 3)),
        _array(initial_translation, (3,)) + _array(delta_translation, (3,)),
    )


def pose_correction(
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    final_rotation: np.ndarray,
    final_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover additive correction from two poses under the left-SO(3) model."""
    delta_rotation = _array(final_rotation, (3, 3)) @ _array(initial_rotation, (3, 3)).T
    return Rotation.from_matrix(delta_rotation).as_rotvec(), _array(final_translation, (3,)) - _array(initial_translation, (3,))


# ---------------------------------------------------------------------------
# Graph data and input loading.


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
    sigma_pose_rotation_rad: float = 0.35
    sigma_pose_translation_m: float = 0.06
    sigma_rgb_rotation_rad: float = 0.08
    sigma_rgb_translation_m: float = 0.02
    sigma_point_plane_m: float = 0.008
    sigma_point_point_m: float = 0.020
    sigma_velocity_rotation_rad: float = 0.30
    sigma_velocity_translation_m: float = 0.05
    sigma_acceleration_rotation_rad: float = 0.25
    sigma_acceleration_translation_m: float = 0.04
    sigma_correction_velocity_rotation_rad: float = 0.40
    sigma_correction_velocity_translation_m: float = 0.08
    max_point_residual_m: float = 0.08
    max_points: int = 350
    min_pairs: int = 12
    max_correspondence_m: float = 0.08
    outer_iterations: int = 3
    inner_iterations: int = 40
    update_rotation_trust_rad: float = 0.45
    update_translation_trust_m: float = 0.05
    max_step_rotation_rad: float = 0.60
    max_step_translation_m: float = 0.10
    max_base_cycle_rotation_rad: float = math.radians(45.0)
    max_base_cycle_translation_m: float = 0.20
    min_rgb_edge_weight: float = 0.10
    min_rgb_translation_edge_weight: float = 0.02
    observed_factor_weight_scale: float = 1.0
    observed_factor_rotation_scale: float = 1.0
    observed_factor_mode: str = "legacy_nn"
    material_image_weight: float = 0.25
    material_metric_weight: float = 1.0
    sigma_material_px: float = 2.0
    sigma_material_m: float = 0.008
    max_material_pixel_degradation_px: float = 3.0
    rgb_rotation_weight_scale: float = 1.0
    rgb_translation_weight_scale: float = 1.0
    rgb_reprojection_weight_scale: float = 0.0
    sigma_rgb_reprojection_px: float = 2.0
    max_rgb_reprojection_residual_px: float = 40.0
    max_rgb_reprojection_points_per_edge: int = 96
    max_metric_degradation_m: float = 0.010
    max_continuity_rotation_rad: float = math.radians(30.0)
    max_continuity_translation_m: float = 0.15
    max_acceleration_rotation_rad: float = math.radians(45.0)
    max_acceleration_translation_m: float = 0.20
    source_conditioning_full_weight_scale: float = 0.02
    min_rgb_rotation_conditioning: float = 0.01
    sigma_anchor_rotation_rad: float = 1.0e-6
    sigma_anchor_translation_m: float = 1.0e-6
    seed: int = 20260907


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_camera_transform(frame: Mapping[str, Any]) -> np.ndarray | None:
    camera = frame.get("camera")
    if not isinstance(camera, Mapping):
        return None
    for key in ("T_world_camera_metric", "T_world_camera"):
        value = camera.get(key)
        try:
            T = _array(value, (4, 4))
        except (TypeError, ValueError):
            continue
        if (
            valid_rotation(T[:3, :3])
            and np.linalg.norm(T[3] - np.asarray([0.0, 0.0, 0.0, 1.0])) < 1.0e-5
        ):
            return T
    return None


def _finite_point_array(value: Any) -> np.ndarray | None:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        return None
    return points


def _p09_ownership_validation(obj: Mapping[str, Any], geometry: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Validate stored P09 ownership without inventing ownership from boxes.

    Older synthetic/unit-test fixtures may omit ownership summaries, so absence
    is recorded as unverified and remains an uncertain usable measurement.
    Explicit failed-closed summaries, failed first-surface states, and a
    negative retained fraction are rejected because they contradict the P09
    visible-surface contract.
    """
    checks: dict[str, Any] = {}
    ownership_values: list[tuple[str, Mapping[str, Any]]] = []
    for container_name, container in (("geometry", geometry), ("object", obj)):
        for key in ("object_surface_ownership_filter", "first_surface_depth_ownership", "ownership_contract"):
            value = container.get(key)
            if isinstance(value, Mapping):
                ownership_values.append((f"{container_name}.{key}", value))
    if not ownership_values:
        return True, {"status": "ownership_metadata_unavailable", "verified": False, "reasons": ["missing_explicit_P09_ownership_summary"]}
    reasons: list[str] = []
    for name, value in ownership_values:
        if value.get("fail_closed") is True:
            reasons.append(f"{name}.fail_closed")
        failure_reasons = value.get("failure_reasons")
        if isinstance(failure_reasons, Sequence) and not isinstance(failure_reasons, (str, bytes, bytearray)) and len(failure_reasons):
            reasons.append(f"{name}.failure_reasons")
        if "enabled" in value and value.get("enabled") is not True:
            reasons.append(f"{name}.not_enabled")
        if "retained_fraction" in value:
            try:
                retained = float(value.get("retained_fraction"))
            except (TypeError, ValueError):
                retained = float("nan")
            if not np.isfinite(retained) or retained <= 0.0:
                reasons.append(f"{name}.invalid_retained_fraction")
    return not reasons, {
        "status": "verified" if not reasons else "failed_closed",
        "verified": not reasons,
        "reasons": reasons,
        "summary_count": len(ownership_values),
    }


def p09_geometry_validation(
    obj: Mapping[str, Any] | None,
    frame: Mapping[str, Any],
    *,
    camera_world_tolerance_m: float = 1.0e-4,
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    """Validate P09 ownership and camera/world consistency before use.

    P09 may provide either world samples or camera samples.  When both are
    present they must describe the same points under the frame's metric
    ``T_world_camera``; silently preferring one would hide a frame-convention
    error.  The function returns an explicit diagnostic for every outcome.
    """
    if obj is None:
        return None, "missing_object", {"status": "missing_object", "verified": False}
    geometry = obj.get("visible_geometry_candidate")
    if not isinstance(geometry, Mapping):
        return None, "missing_p09_visible_geometry", {"status": "missing_geometry", "verified": False}
    eligibility_values = [
        ("object.rigid_pose_observation_eligible", obj.get("rigid_pose_observation_eligible")),
        ("geometry.rigid_pose_observation_eligible", geometry.get("rigid_pose_observation_eligible")),
    ]
    ineligible = [name for name, value in eligibility_values if value is False]
    if ineligible:
        return None, "P09_rigid_pose_observation_ineligible", {
            "status": "rigid_pose_ineligible", "verified": False,
            "ineligible_fields": ineligible,
        }
    ownership_ok, ownership_diag = _p09_ownership_validation(obj, geometry)
    if not ownership_ok:
        return None, "P09_ownership_failed_closed", {
            "status": "ownership_failed_closed", "verified": False,
            "ownership": ownership_diag,
        }
    world_present = "world_vertices_sample_m" in geometry and geometry.get("world_vertices_sample_m") is not None
    camera_present = "camera_vertices_sample_m" in geometry and geometry.get("camera_vertices_sample_m") is not None
    world = _finite_point_array(geometry.get("world_vertices_sample_m")) if world_present else None
    camera = _finite_point_array(geometry.get("camera_vertices_sample_m")) if camera_present else None
    if world_present and world is None:
        return None, "invalid_P09_world_surface", {"status": "invalid_world_points", "verified": False, "ownership": ownership_diag}
    if camera_present and camera is None:
        return None, "invalid_P09_camera_surface", {"status": "invalid_camera_points", "verified": False, "ownership": ownership_diag}
    if camera is not None and np.any(camera[:, 2] <= 0.0):
        return None, "invalid_P09_camera_depth", {"status": "nonpositive_camera_depth", "verified": False, "ownership": ownership_diag}
    T = _frame_camera_transform(frame)
    if camera is not None and T is None:
        return None, "missing_or_invalid_P09_camera_transform", {"status": "invalid_camera_transform", "verified": False, "ownership": ownership_diag}
    transform_error_m: float | None = None
    if world is not None and camera is not None:
        if len(world) != len(camera):
            return None, "P09_camera_world_count_mismatch", {"status": "camera_world_count_mismatch", "verified": False, "ownership": ownership_diag}
        assert T is not None
        lifted = camera_to_world(camera, T)
        transform_error_m = float(np.max(np.linalg.norm(lifted - world, axis=1)))
        if not np.isfinite(transform_error_m) or transform_error_m > float(camera_world_tolerance_m):
            return None, "P09_camera_world_transform_mismatch", {
                "status": "camera_world_mismatch", "verified": False,
                "max_camera_world_error_m": transform_error_m,
                "camera_world_tolerance_m": float(camera_world_tolerance_m),
                "ownership": ownership_diag,
            }
    points = world if world is not None else (camera_to_world(camera, T) if camera is not None and T is not None else None)
    if points is None:
        return None, "missing_or_invalid_P09_surface", {"status": "missing_surface", "verified": False, "ownership": ownership_diag}
    source = "P09.world_vertices_sample_m" if world is not None else "P09.camera_vertices_sample_m->world"
    return points, source, {
        "status": "accepted",
        "verified": bool(world is not None and camera is not None and transform_error_m is not None),
        "ownership": ownership_diag,
        "camera_world_consistency": {
            "checked": bool(world is not None and camera is not None),
            "max_error_m": transform_error_m,
            "tolerance_m": float(camera_world_tolerance_m),
        },
        "source": source,
        "point_count": int(len(points)),
    }


def _object_for_frame(frame: Mapping[str, Any], object_id: str) -> Mapping[str, Any] | None:
    objects = frame.get("objects")
    if not isinstance(objects, list):
        return None
    for obj in objects:
        if isinstance(obj, Mapping) and str(obj.get("object_id")) == str(object_id):
            return obj
    return None


def observed_points_from_object(obj: Mapping[str, Any], frame: Mapping[str, Any]) -> tuple[np.ndarray | None, str]:
    """Read validated P09 world points, converting camera points exactly once."""
    points, source, _diagnostics = p09_geometry_validation(obj, frame)
    return points, source


def _timeline_rows(report: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows = report.get("pose_rows", [])
    result: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        idx = int(row["frame_idx"])
        if idx in result:
            raise RuntimeError(f"duplicate timeline pose row {idx}")
        result[idx] = row
    return result


def build_pose_nodes(
    annotations: Mapping[str, Any],
    timeline_report: Mapping[str, Any],
    immutable_report: Mapping[str, Any],
    object_id: str,
    frame_start: int,
    frame_end: int,
) -> tuple[list[PoseNode], dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]], dict[int, dict[str, Any]]]:
    frames = {
        int(frame["frame_idx"]): frame
        for frame in annotations.get("frames", [])
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }
    timeline_rows = _timeline_rows(timeline_report)
    immutable_rows = _timeline_rows(immutable_report)
    all_ids = sorted(set(frames) | set(timeline_rows) | set(immutable_rows))
    if not all_ids:
        raise RuntimeError("no timeline rows or annotation frames available")
    immutable_direct: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, row in immutable_rows.items():
        pose = pose_from_values(row.get("rotation_world_from_completed_canonical_matrix"), row.get("translation_world_m"))
        if pose is not None:
            immutable_direct[idx] = pose
    timeline_direct: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, row in timeline_rows.items():
        pose = pose_from_values(row.get("rotation_world_from_completed_canonical_matrix"), row.get("translation_world_m"))
        if pose is not None:
            timeline_direct[idx] = pose
    # Determine metric availability before building the interpolation source.
    # An existing pose on a no-metric row is not a direct measurement: frame
    # 143 must be an explicit latent graph node initialized from neighboring
    # direct metric SE(3) poses rather than silently inheriting its row pose.
    observed_by_frame: dict[int, tuple[np.ndarray | None, str, dict[str, Any]]] = {}
    for idx in all_ids:
        frame = frames.get(idx, {})
        obj = _object_for_frame(frame, object_id)
        observed_by_frame[idx] = p09_geometry_validation(obj, frame)
    metric_timeline_direct = {idx: pose for idx, pose in timeline_direct.items() if observed_by_frame.get(idx, (None, "", {}))[0] is not None}
    metric_immutable_direct = {idx: pose for idx, pose in immutable_direct.items() if observed_by_frame.get(idx, (None, "", {}))[0] is not None}
    # Existing full-timeline states are the current base only when they also
    # have direct P09 metric evidence.  Latent rows are filled continuously
    # from neighboring direct metric poses.
    interpolation_source = dict(metric_timeline_direct)
    interpolation_source.update({idx: pose for idx, pose in metric_immutable_direct.items() if idx not in interpolation_source})
    if not interpolation_source:
        raise RuntimeError("timeline has no direct metric P09 pose to initialize SE(3) states")
    nodes: list[PoseNode] = []
    for idx in all_ids:
        observed, observed_source, p09_diagnostics = observed_by_frame[idx]
        metric = observed is not None
        if metric and idx in timeline_direct:
            R, t = timeline_direct[idx]
            source = "timeline_metric_pose"
        elif metric and idx in immutable_direct:
            R, t = immutable_direct[idx]
            source = "immutable_metric_pose"
        else:
            R, t, source = initialize_missing_se3_pose(idx, interpolation_source)
        # A row can carry a pose without direct metric depth.  It remains a
        # latent uncertain variable, not a promoted metric measurement.
        uncertainty = 0.0 if metric and source in {"timeline_metric_pose", "immutable_metric_pose"} else 0.45 if metric else 0.85
        if idx == 143 and not metric:
            source = f"{source};metric_missing_frame_143_latent_uncertain"
            uncertainty = 1.0
        source_row = dict(timeline_rows.get(idx, {"frame_idx": idx}))
        source_row["v20_p09_validation"] = p09_diagnostics
        nodes.append(PoseNode(idx, R.copy(), t.copy(), observed, observed_source if metric else source, metric, uncertainty, source_row))
    return nodes, immutable_direct, timeline_direct, frames


# ---------------------------------------------------------------------------
# Prediction-only RGB/PnP edge loading and diagnostics.


def _first_array(data: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def _edge_rotation(row: Mapping[str, Any]) -> Any:
    return _first_array(row, ("rotation_source_to_target_world", "rotation_source_to_target", "rotation_rgb", "rgb_rotation", "relative_rotation"))


def _edge_translation(row: Mapping[str, Any]) -> Any:
    return _first_array(row, ("translation_source_to_target_world_m", "translation_source_to_target_m", "translation_rgb_m", "rgb_translation_m", "relative_translation_m"))


def _edge_accepted(row: Mapping[str, Any]) -> bool:
    if "accepted" in row:
        return bool(row.get("accepted"))
    return str(row.get("status") or "").lower() == "accepted"


def _finite_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _conditioning_weight(
    row: Mapping[str, Any],
    full_weight_scale: float | None = None,
    *,
    config: SolverConfig | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute an RGB edge weight with explicitly normalized conditioning.

    ``source_3d_conditioning`` is a prediction-side lambda-min/lambda-max
    ratio.  Its absolute scale varies with the edge producer, so the ratio is
    divided by a configurable value that earns full conditioning weight.
    """
    if full_weight_scale is None:
        full_weight_scale = (config or SolverConfig()).source_conditioning_full_weight_scale
    scale = _finite_float(full_weight_scale, 0.0)
    if scale <= 0.0:
        raise ValueError("source conditioning full-weight scale must be positive")
    inlier_fraction = _finite_float(row.get("inlier_fraction", row.get("pnp_inlier_fraction", 1.0)), 0.0)
    reprojection = _finite_float(row.get("reprojection_median_px", row.get("reprojection_error_median_px", 0.0)), float("inf"))
    condition = row.get("source_3d_conditioning")
    condition_number = row.get("rotation_information_condition")
    eig = row.get("source_3d_covariance_eigenvalues", row.get("rotation_information_eigenvalues"))
    if condition is not None:
        condition_ratio_raw = max(0.0, _finite_float(condition, 0.0))
        condition_source = "source_3d_conditioning"
    elif condition_number is not None:
        raw_number = _finite_float(condition_number, float("inf"))
        condition_ratio_raw = 1.0 / max(raw_number, 1.0) if math.isfinite(raw_number) else 0.0
        condition_source = "rotation_information_condition_inverse"
    elif eig is not None:
        try:
            ev = np.maximum(_array(eig).reshape(-1), 0.0)
            condition_ratio_raw = float(ev.min() / max(float(ev.max()), 1.0e-12)) if len(ev) else 0.0
        except (TypeError, ValueError):
            condition_ratio_raw = 0.0
        condition_source = "covariance_eigenvalue_ratio"
    else:
        condition_ratio_raw = scale
        condition_source = "not_serialized_assumed_full_weight"
    condition_ratio_normalized = float(np.clip(condition_ratio_raw / scale, 0.0, 1.0))
    condition_weight = float(np.clip(condition_ratio_normalized, 0.05, 1.0))
    inlier_weight = float(np.clip(inlier_fraction, 0.05, 1.0))
    reprojection_weight = float(np.exp(-max(0.0, reprojection) / 2.0)) if math.isfinite(reprojection) else 0.0
    quality = max(0.0, _finite_float(row.get("quality_weight", row.get("weight", 1.0)), 0.0))
    weight = float(np.clip(quality * inlier_weight * reprojection_weight * condition_weight, 0.0, 1.0))
    return weight, {
        "quality_weight": quality,
        "inlier_fraction": inlier_fraction,
        "reprojection_median_px": reprojection if math.isfinite(reprojection) else None,
        "conditioning_source": condition_source,
        "conditioning_weight": condition_weight,
        "conditioning_ratio": condition_ratio_normalized,
        "conditioning_ratio_raw": condition_ratio_raw,
        "source_conditioning_full_weight_scale": scale,
        "inlier_weight": inlier_weight,
        "reprojection_weight": reprojection_weight,
        "effective_weight": weight,
    }


def _json_edge_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _transform_identity_diagnostics(
    first_rotation: Any,
    first_translation: Any,
    second_rotation: Any,
    second_translation: Any,
    *,
    rotation_tolerance_rad: float = 1.0e-5,
    translation_tolerance_m: float = 1.0e-5,
) -> dict[str, Any]:
    """Verify that two serialized source-to-target SE(3) transforms agree."""
    try:
        first_R = _array(first_rotation, (3, 3))
        first_t = _array(first_translation, (3,))
        second_R = _array(second_rotation, (3, 3))
        second_t = _array(second_translation, (3,))
        if not valid_rotation(first_R) or not valid_rotation(second_R):
            raise ValueError("invalid rotation")
        identity_R = first_R @ second_R.T
        identity_t = first_t - identity_R @ second_t
        rotation_error = float(Rotation.from_matrix(identity_R).magnitude())
        translation_error = float(np.linalg.norm(identity_t))
        verified = rotation_error <= float(rotation_tolerance_rad) and translation_error <= float(translation_tolerance_m)
    except (TypeError, ValueError):
        rotation_error = translation_error = None
        verified = False
    return {
        "npz_json_transform_verified": bool(verified),
        "npz_json_transform_identity_rotation_rad": rotation_error,
        "npz_json_transform_identity_translation_m": translation_error,
        "npz_json_transform_rotation_tolerance_rad": float(rotation_tolerance_rad),
        "npz_json_transform_translation_tolerance_m": float(translation_tolerance_m),
    }


def load_rgb_edges(
    npz_path: Path | None,
    *,
    json_path: Path | None = None,
    pose_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
    frame_ids: set[int] | None = None,
    config: SolverConfig | None = None,
    source_support_by_frame: Mapping[int, np.ndarray] | None = None,
) -> tuple[list[RGBEdge], list[dict[str, Any]]]:
    """Load accepted world-relative PnP edges from both V19 and V20 formats.

    V19 stores ``rotation_rgb``/``translation_rgb_m``; V20 global orientation
    edges store ``rotation_source_to_target_world``/
    ``translation_source_to_target_world_m``.  Both represent the transform
    mapping a source-world point to a target-world point.
    """
    if npz_path is not None and json_path is None:
        companion = npz_path.with_suffix(".json")
        json_path = companion if companion.exists() else None
    json_rows = _json_edge_rows(json_path)
    json_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in json_rows:
        try:
            key = (int(row.get("source_frame_idx", row.get("source_idx"))), int(row.get("target_frame_idx", row.get("target_idx"))))
        except (TypeError, ValueError):
            continue
        json_by_key[key] = dict(row)
    npz_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if npz_path is not None:
        with np.load(npz_path.expanduser().resolve(), allow_pickle=False) as data:
            keys = set(data.files)
            src_raw = _first_array(data, ("source_frame_idx", "source_idx"))
            dst_raw = _first_array(data, ("target_frame_idx", "target_idx"))
            rot_raw = _first_array(data, ("rotation_rgb", "rotation_source_to_target_world", "rotation_source_to_target"))
            trans_raw = _first_array(data, ("translation_rgb_m", "translation_source_to_target_world_m", "translation_source_to_target_m"))
            accepted_raw = _first_array(data, ("accepted", "status"), None)
            if src_raw is None or dst_raw is None or rot_raw is None or trans_raw is None:
                if not json_by_key:
                    raise RuntimeError(f"RGB edge NPZ lacks supported edge arrays: {sorted(keys)}")
            else:
                src = np.asarray(src_raw).reshape(-1); dst = np.asarray(dst_raw).reshape(-1)
                rot = np.asarray(rot_raw, dtype=np.float64); trans = np.asarray(trans_raw, dtype=np.float64)
                n = len(src)
                if len(dst) != n or len(rot) != n or len(trans) != n:
                    raise RuntimeError("RGB edge NPZ arrays have inconsistent lengths")
                if accepted_raw is None:
                    accepted = np.ones(n, dtype=bool)
                else:
                    raw_accepted = np.asarray(accepted_raw).reshape(-1)
                    if len(raw_accepted) != n:
                        raise RuntimeError("RGB edge accepted array length mismatch")
                    if raw_accepted.dtype.kind in {"U", "S", "O"}:
                        accepted = np.asarray([str(value).lower() in {"1", "true", "accepted", "yes"} for value in raw_accepted], dtype=bool)
                    else:
                        accepted = raw_accepted.astype(bool)
                quality_raw = _first_array(data, ("quality_weight", "weight"), None)
                quality = None if quality_raw is None else np.asarray(quality_raw, dtype=np.float64).reshape(-1)
                if quality is not None and len(quality) not in {1, n}:
                    raise RuntimeError("RGB edge quality array length mismatch")
                for i in range(n):
                    npz_by_key[(int(src[i]), int(dst[i]))] = {
                        "source_frame_idx": int(src[i]), "target_frame_idx": int(dst[i]),
                        "rotation_source_to_target_world": rot[i].tolist(),
                        "translation_source_to_target_world_m": trans[i].tolist(),
                        "accepted": bool(accepted[i]),
                        "quality_weight": float(quality[0] if len(quality) == 1 else quality[i]) if quality is not None else 1.0,
                    }
    # Companion JSON carries richer diagnostics; NPZ carries the numeric
    # transform/accepted arrays.  Merge one row per pair and verify both.
    rows: list[dict[str, Any]] = []
    for key in sorted(set(json_by_key) | set(npz_by_key)):
        npz_row = npz_by_key.get(key); json_row = json_by_key.get(key)
        if npz_row is None:
            rows.append(dict(json_row)); continue
        merged = dict(npz_row)
        if json_row is not None:
            merged.update(json_row)
            json_rotation = _edge_rotation(json_row)
            json_translation = _edge_translation(json_row)
            if json_rotation is None or json_translation is None:
                merged.update({
                    "npz_json_transform_verified": True,
                    "npz_json_transform_verification_mode": "NPZ_numeric_transform_JSON_diagnostics_only",
                })
            else:
                merged.update(_transform_identity_diagnostics(
                    npz_row["rotation_source_to_target_world"],
                    npz_row["translation_source_to_target_world_m"],
                    json_rotation,
                    json_translation,
                ))
            if _edge_accepted(json_row) != bool(npz_row["accepted"]):
                merged["npz_json_acceptance_verified"] = False
            else:
                merged["npz_json_acceptance_verified"] = True
            # Always use the NPZ transform after verification; JSON diagnostics
            # and quality/conditioning fields remain authoritative.
            merged["rotation_source_to_target_world"] = npz_row["rotation_source_to_target_world"]
            merged["translation_source_to_target_world_m"] = npz_row["translation_source_to_target_world_m"]
            merged["accepted"] = bool(npz_row["accepted"])
        rows.append(merged)
    config = config or SolverConfig()
    pose_by_frame = pose_by_frame or {}
    edges: list[RGBEdge] = []
    diagnostics: list[dict[str, Any]] = []
    for raw in rows:
        try:
            source = int(raw.get("source_frame_idx", raw.get("source_idx")))
            target = int(raw.get("target_frame_idx", raw.get("target_idx")))
        except (TypeError, ValueError):
            diagnostics.append({"status": "rejected", "reason": "missing_frame_indices", "raw": dict(raw)})
            continue
        diag: dict[str, Any] = {
            "source_frame_idx": source,
            "target_frame_idx": target,
            "npz_json_transform_verified": raw.get("npz_json_transform_verified"),
            "npz_json_transform_identity_rotation_rad": raw.get("npz_json_transform_identity_rotation_rad"),
            "npz_json_transform_identity_translation_m": raw.get("npz_json_transform_identity_translation_m"),
            "npz_json_acceptance_verified": raw.get("npz_json_acceptance_verified"),
        }
        if raw.get("npz_json_transform_verified") is False:
            diag["status"] = "rejected"; diag["reason"] = "npz_json_transform_mismatch"; diagnostics.append(diag); continue
        if raw.get("npz_json_acceptance_verified") is False:
            diag["status"] = "rejected"; diag["reason"] = "npz_json_acceptance_mismatch"; diagnostics.append(diag); continue
        if frame_ids is not None and (source not in frame_ids or target not in frame_ids):
            diag["status"] = "rejected"; diag["reason"] = "outside_timeline"; diagnostics.append(diag); continue
        if not _edge_accepted(raw):
            diag["status"] = "rejected"; diag["reason"] = "serialized_rejection"; diagnostics.append(diag); continue
        try:
            R = _array(_edge_rotation(raw), (3, 3)); t = _array(_edge_translation(raw), (3,))
            R = Rotation.from_matrix(R).as_matrix()
        except (TypeError, ValueError):
            diag["status"] = "rejected"; diag["reason"] = "invalid_world_relative_transform"; diagnostics.append(diag); continue
        try:
            weight, qdiag = _conditioning_weight(raw, config=config)
        except ValueError as exc:
            diag["status"] = "rejected"; diag["reason"] = "invalid_conditioning_configuration"; diag["detail"] = str(exc); diagnostics.append(diag); continue
        diag.update(qdiag)
        raw_conditioning = qdiag.get("conditioning_ratio_raw")
        rotation_conditioning_threshold = float(config.min_rgb_rotation_conditioning)
        rotation_eligible = (
            raw_conditioning is None
            or float(raw_conditioning) >= rotation_conditioning_threshold
        )
        diag["rotation_eligible"] = bool(rotation_eligible)
        diag["rotation_conditioning_threshold"] = rotation_conditioning_threshold
        diag["rotation_weight_scale"] = 1.0 if rotation_eligible else 0.0
        diag["rotation_exclusion_reason"] = (
            None if rotation_eligible else "low_3d_conditioning_planar_edge"
        )
        if weight < float(config.min_rgb_edge_weight):
            # A poorly conditioned PnP edge may still carry weak translation
            # evidence. Keep it only as translation-only evidence; never let
            # it contribute a spurious rotation residual.
            translation_floor = float(config.min_rgb_translation_edge_weight)
            if (not rotation_eligible) and weight >= translation_floor:
                diag["translation_only"] = True
            else:
                diag["status"] = "rejected"; diag["reason"] = "low_conditioning_planar_edge"; diagnostics.append(diag); continue
        else:
            diag["translation_only"] = not rotation_eligible
        if pose_by_frame and source in pose_by_frame and target in pose_by_frame:
            Rs, ts = pose_by_frame[source]; Rt, tt = pose_by_frame[target]
            base_R = Rt @ Rs.T
            base_t = tt - base_R @ ts
            cycle_R = R @ base_R.T
            cycle_t = t - base_t
            cycle_rotation = float(Rotation.from_matrix(cycle_R).magnitude())
            cycle_translation = float(np.linalg.norm(cycle_t))
            diag.update({"base_cycle_rotation_rad": cycle_rotation, "base_cycle_translation_m": cycle_translation})
            if cycle_rotation > float(config.max_base_cycle_rotation_rad) or cycle_translation > float(config.max_base_cycle_translation_m):
                diag["status"] = "rejected"; diag["reason"] = "base_trajectory_cycle_gate"; diagnostics.append(diag); continue
        if weight <= 0.0:
            diag["status"] = "rejected"; diag["reason"] = "zero_conditioned_weight"; diagnostics.append(diag); continue
        reference_R = support = None
        if diag.get("translation_only"):
            if source not in pose_by_frame or target not in pose_by_frame or source not in (source_support_by_frame or {}):
                diag.update(status="rejected", reason="translation_only_reference_or_support_missing")
                diagnostics.append(diag)
                continue
            reference_R = pose_by_frame[target][0] @ pose_by_frame[source][0].T
            support = _array(source_support_by_frame[source], (3,)).copy()
            diag["translation_residual_contract"] = "fixed_orientation_displacement_at_source_support"
        diag["status"] = "accepted"
        diagnostics.append(diag)
        edges.append(RGBEdge(source, target, R, t, weight, diag, reference_R, support))
    return edges, diagnostics


def fixed_rgb_source_world(data: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover fixed observations using the producer's hash-bound pose once.

    A v2 bundle serialized source points in the producer's canonical frame;
    they are not landmarks to be anchored to that pose during optimization.
    """
    if "metadata" not in data:
        raise RuntimeError("RGB point evidence has no provenance metadata")
    raw_metadata = np.asarray(data["metadata"]).reshape(-1)
    if len(raw_metadata) != 1:
        raise RuntimeError("RGB point evidence metadata must contain one JSON object")
    meta = json.loads(str(raw_metadata[0]))
    assert_prediction_only(meta, label="RGB point evidence metadata")
    path = Path(str(meta.get("initial_pose_report") or "")).expanduser().resolve()
    expected = (meta.get("input_sha256") or {}).get("initial_pose_report")
    if not expected or not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError("legacy RGB canonical evidence initial-pose hash mismatch")
    producer = load_json(path)
    assert_prediction_only(producer, label="RGB canonical evidence producer pose", object_id=meta.get("object_id"))
    if "reprojection_source_world" in data:
        points = _array(data["reprojection_source_world"])
        if points.ndim != 2 or points.shape[1] != 3 or meta.get("reprojection_evidence_contract", {}).get("source_frame") != "fixed_world_observation":
            raise RuntimeError("RGB source-world evidence lacks valid points/coordinate contract")
        return points, meta
    poses = _timeline_rows(producer)
    points = _array(data["reprojection_canonical_points"]).copy()
    src = np.asarray(data["source_frame_idx"], dtype=np.int64).reshape(-1)
    offsets = np.asarray(data["reprojection_evidence_offsets"], dtype=np.int64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(offsets) != len(src) + 1 or offsets[0] != 0 or offsets[-1] != len(points) or np.any(np.diff(offsets) < 0):
        raise RuntimeError("invalid legacy RGB canonical evidence shape/offsets")
    for i, idx in enumerate(src):
        left, right = offsets[i:i+2]
        if right == left:
            continue
        row = poses.get(int(idx), {})
        pose = pose_from_values(row.get("rotation_world_from_completed_canonical_matrix"), row.get("translation_world_m"))
        if pose is None:
            raise RuntimeError(f"RGB canonical evidence producer lacks valid source pose {idx}")
        points[left:right] = apply_pose(points[left:right], *pose)
    return points, meta


def load_rgb_reprojection_factors(
    npz_path: Path | None,
    *,
    edges: Sequence[RGBEdge],
    nodes: Sequence[PoseNode],
    config: SolverConfig,
) -> tuple[list[RGBReprojectionFactor], dict[str, Any]]:
    """Load optional prediction-side source-canonical to target-UV evidence.

    The edge builder stores variable-length evidence using flattened arrays and
    offsets.  This adapter is intentionally optional: old edge bundles remain
    valid, and the objective includes these rows only when the explicit weight
    scale is positive.  Low-conditioning edges retain translation evidence but
    freeze the rotations used by this pixel residual.
    """
    diagnostics: dict[str, Any] = {
        "available": False,
        "enabled": bool(float(config.rgb_reprojection_weight_scale) > 0.0),
        "edge_count": 0,
        "point_count": 0,
        "disabled_reason": None,
    }
    if not diagnostics["enabled"]:
        diagnostics["disabled_reason"] = "weight_disabled"
        return [], diagnostics
    if npz_path is None or not npz_path.exists():
        diagnostics["disabled_reason"] = "npz_missing"
        return [], diagnostics
    with np.load(npz_path.expanduser().resolve(), allow_pickle=False) as data:
        required = {
            "source_frame_idx", "target_frame_idx", "reprojection_evidence_offsets",
            "reprojection_target_uv",
            "reprojection_weights", "reprojection_target_intrinsics",
            "reprojection_target_camera",
        }
        missing = sorted(required - set(data.files))
        if missing:
            diagnostics["disabled_reason"] = "evidence_arrays_missing"
            diagnostics["missing_arrays"] = missing
            return [], diagnostics
        src = np.asarray(data["source_frame_idx"], dtype=np.int64).reshape(-1)
        dst = np.asarray(data["target_frame_idx"], dtype=np.int64).reshape(-1)
        offsets = np.asarray(data["reprojection_evidence_offsets"], dtype=np.int64).reshape(-1)
        if "reprojection_source_world" not in data and "reprojection_canonical_points" not in data:
            raise RuntimeError("RGB reprojection evidence has no source points")
        points, evidence_metadata = fixed_rgb_source_world(data)
        diagnostics["input_metadata"] = evidence_metadata
        target_uv = np.asarray(data["reprojection_target_uv"], dtype=np.float64)
        weights = np.asarray(data["reprojection_weights"], dtype=np.float64).reshape(-1)
        intrinsics = np.asarray(data["reprojection_target_intrinsics"], dtype=np.float64)
        cameras = np.asarray(data["reprojection_target_camera"], dtype=np.float64)
    n = len(src)
    if len(dst) != n or len(offsets) != n + 1 or intrinsics.shape != (n, 4) or cameras.shape != (n, 4, 4):
        raise RuntimeError("RGB reprojection evidence arrays have inconsistent edge dimensions")
    if points.ndim != 2 or points.shape[1] != 3 or target_uv.shape != (len(points), 2) or len(weights) != len(points):
        raise RuntimeError("RGB reprojection evidence point arrays have inconsistent dimensions")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0) or int(offsets[-1]) != len(points):
        raise RuntimeError("RGB reprojection evidence offsets are invalid")
    positions = {node.frame_idx: index for index, node in enumerate(nodes)}
    edge_by_key = {(edge.source_frame_idx, edge.target_frame_idx): edge for edge in edges}
    factors: list[RGBReprojectionFactor] = []
    for i in range(n):
        key = (int(src[i]), int(dst[i]))
        edge = edge_by_key.get(key)
        left, right = int(offsets[i]), int(offsets[i + 1])
        if edge is None or right <= left:
            continue
        if key[0] not in positions or key[1] not in positions:
            continue
        if not nodes[positions[key[0]]].metric_observation:
            continue
        R_camera = np.asarray(cameras[i, :3, :3], dtype=np.float64)
        t_camera = np.asarray(cameras[i, :3, 3], dtype=np.float64)
        if not valid_rotation(R_camera) or not np.isfinite(t_camera).all():
            continue
        p = np.asarray(points[left:right], dtype=np.float64)
        uv = np.asarray(target_uv[left:right], dtype=np.float64)
        w = np.asarray(weights[left:right], dtype=np.float64)
        valid = np.isfinite(p).all(axis=1) & np.isfinite(uv).all(axis=1) & np.isfinite(w) & (w > 0.0)
        if int(np.count_nonzero(valid)) < 4:
            continue
        p = p[valid]
        uv = uv[valid]
        w = np.clip(w[valid], 0.05, 1.0)
        max_points = int(config.max_rgb_reprojection_points_per_edge)
        if max_points > 0 and len(p) > max_points:
            keep = np.linspace(0, len(p) - 1, max_points, dtype=np.int64)
            p, uv, w = p[keep], uv[keep], w[keep]
        factors.append(RGBReprojectionFactor(
            source_frame_idx=key[0], target_frame_idx=key[1],
            source_pos=positions[key[0]], target_pos=positions[key[1]],
            source_world=p, target_uv=uv,
            target_intrinsics=np.asarray(intrinsics[i], dtype=np.float64),
            target_camera_rotation=R_camera, target_camera_translation=t_camera,
            weights=w, edge_weight=float(edge.weight),
            rotation_eligible=bool(edge.diagnostics.get("rotation_eligible", True)),
        ))
    diagnostics.update({
        "available": bool(factors),
        "edge_count": len(factors),
        "point_count": int(sum(len(factor.weights) for factor in factors)),
        "disabled_reason": None if factors else "no_valid_evidence_for_accepted_edges",
    })
    return factors, diagnostics


def rgb_rotation_observability(
    edges: Sequence[RGBEdge],
    nodes: Sequence[PoseNode],
) -> dict[str, Any]:
    """Report which timeline nodes have non-planar RGB rotation evidence."""
    per_frame: list[dict[str, Any]] = []
    for node in sorted(nodes, key=lambda item: item.frame_idx):
        incident = [
            edge for edge in edges
            if edge.source_frame_idx == node.frame_idx or edge.target_frame_idx == node.frame_idx
        ]
        rotation_edges = [
            edge for edge in incident
            if bool(edge.diagnostics.get("rotation_eligible", True))
            and float(edge.diagnostics.get("rotation_weight_scale", 1.0)) > 0.0
        ]
        conditions = [
            float(edge.diagnostics.get("conditioning_ratio_raw"))
            for edge in rotation_edges
            if edge.diagnostics.get("conditioning_ratio_raw") is not None
            and math.isfinite(float(edge.diagnostics.get("conditioning_ratio_raw")))
        ]
        per_frame.append({
            "frame_idx": int(node.frame_idx),
            "incident_edge_count": len(incident),
            "rotation_eligible_edge_count": len(rotation_edges),
            "translation_only_edge_count": sum(
                bool(edge.diagnostics.get("translation_only", False)) for edge in incident
            ),
            "min_rotation_conditioning": min(conditions) if conditions else None,
            "max_rotation_conditioning": max(conditions) if conditions else None,
            "rotation_observability_uncertain": not bool(rotation_edges),
        })
    uncertain = [row["frame_idx"] for row in per_frame if row["rotation_observability_uncertain"]]
    return {
        "per_frame": per_frame,
        "rotation_observability_uncertain_frames": uncertain,
        "rotation_observability_uncertain_frame_count": len(uncertain),
        "definition": "only accepted RGB edges with conditioning above min_rgb_rotation_conditioning provide rotation authority; low-conditioning edges remain translation-only",
    }


# ---------------------------------------------------------------------------
# Dynamic P09 observed factors and pose objective.


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    points = _array(points)
    if len(points) < 3:
        return np.tile(np.asarray([[0.0, 0.0, 1.0]]), (len(points), 1))
    kk = min(max(3, int(k)), len(points))
    _distances, indices = cKDTree(points).query(points, k=kk, workers=-1)
    centered = points[indices] - points[indices].mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(1, kk - 1)
    _values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1.0e-12)


def _mutual_pairs(source: np.ndarray, target: np.ndarray, max_distance: float, max_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source) == 0 or len(target) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0)
    d, j = cKDTree(target).query(source, k=1, workers=-1)
    reverse = cKDTree(source).query(target, k=1, workers=-1)[1]
    source_indices = np.arange(len(source), dtype=np.int64)
    keep = np.flatnonzero(reverse[j] == source_indices)
    if len(keep) < 3:
        keep = np.argsort(d)[: min(len(d), max(3, int(max_pairs)))]
    keep = keep[d[keep] <= float(max_distance)]
    if len(keep) > int(max_pairs):
        keep = keep[np.argsort(d[keep])[: int(max_pairs)]]
    return keep.astype(np.int64), j[keep].astype(np.int64), d[keep].astype(np.float64)


def build_observed_point_factors(
    nodes: Sequence[PoseNode],
    pose_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]],
    anchor_frame: int,
    config: SolverConfig | None = None,
    outer_iteration: int = 0,
) -> tuple[list[PointFactor], dict[str, Any]]:
    """Rebuild P09 correspondences in the candidate canonical frames.

    The returned factors retain observed world points, not generated mesh
    points.  Re-running this function for every outer pass is important: the
    nearest-neighbor association must follow the current SE(3) candidate.
    """
    cfg = config or SolverConfig()
    if cfg.observed_factor_mode == "material_tracks":
        return [], {"outer_iteration": int(outer_iteration), "pair_count": 0, "factor_count": 0,
                    "reason": "NN replaced by fixed RGB-D material correspondences", "generated_factors_included": False}
    positions = {node.frame_idx: i for i, node in enumerate(nodes)}
    direct = [node for node in nodes if node.metric_observation and node.observed_world is not None and node.frame_idx in pose_by_frame]
    if len(direct) < 2:
        return [], {"outer_iteration": int(outer_iteration), "pair_count": 0, "factor_count": 0, "reason": "fewer_than_two_P09_metric_rows"}
    canonical: dict[int, np.ndarray] = {}
    normals: dict[int, np.ndarray] = {}
    sampled_world: dict[int, np.ndarray] = {}
    for node in direct:
        R, t = pose_by_frame[node.frame_idx]
        points = np.asarray(node.observed_world, dtype=np.float64)
        if len(points) > int(cfg.max_points):
            rng = np.random.default_rng(int(cfg.seed) + int(node.frame_idx) + 31 * int(outer_iteration))
            points = points[rng.choice(len(points), int(cfg.max_points), replace=False)]
        sampled_world[node.frame_idx] = points
        canonical[node.frame_idx] = inverse_pose(points, R, t)
        normals[node.frame_idx] = estimate_normals(canonical[node.frame_idx])
    pairs: list[tuple[int, int, str]] = []
    direct_ids = sorted(canonical)
    pairs.extend((a, b, "temporal_observed") for a, b in zip(direct_ids[:-1], direct_ids[1:]))
    if int(anchor_frame) in canonical:
        pairs.extend((int(anchor_frame), idx, "anchor_observed_loop") for idx in direct_ids if idx != int(anchor_frame))
    factors: list[PointFactor] = []
    pair_diagnostics: list[dict[str, Any]] = []
    for source_id, target_id, kind in pairs:
        source_c = canonical[source_id]; target_c = canonical[target_id]
        si, ti, distance = _mutual_pairs(source_c, target_c, float(cfg.max_correspondence_m), int(cfg.max_points))
        if len(si) < int(cfg.min_pairs):
            pair_diagnostics.append({"source_frame_idx": source_id, "target_frame_idx": target_id, "kind": kind, "pair_count": int(len(si)), "accepted": False, "reason": "too_few_rebuilt_correspondences"})
            continue
        quality = np.exp(-distance / max(float(cfg.max_correspondence_m), 1.0e-6))
        factors.append(PointFactor(
            positions[source_id], positions[target_id],
            np.asarray(sampled_world[source_id], dtype=np.float64)[si],
            np.asarray(sampled_world[target_id], dtype=np.float64)[ti],
            normals[target_id][ti], np.clip(quality, 0.05, 1.0), kind,
        ))
        pair_diagnostics.append({"source_frame_idx": source_id, "target_frame_idx": target_id, "kind": kind, "pair_count": int(len(si)), "median_initial_canonical_distance_m": float(np.median(distance)), "accepted": True})
    return factors, {"outer_iteration": int(outer_iteration), "pair_count": len(pair_diagnostics), "factor_count": len(factors), "pairs": pair_diagnostics, "rebuilt_from": "P09 world_vertices_sample_m or camera_vertices_sample_m", "generated_factors_included": False}


def unpack_corrections(x: np.ndarray, count: int) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    if value.size != int(count) * 6:
        raise ValueError(f"expected {count * 6} correction values, got {value.size}")
    return value.reshape(int(count), 6)


def current_poses(nodes: Sequence[PoseNode], x: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    corrections = unpack_corrections(x, len(nodes))
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for node, correction in zip(nodes, corrections):
        # Additive world translation is intentional; do not rotate t by the
        # left correction (that is a different, unapproved convention).
        R = Rotation.from_rotvec(correction[:3]).as_matrix() @ node.initial_rotation
        t = node.initial_translation + correction[3:]
        result[node.frame_idx] = (R, t)
    return result


def rgb_edge_residual(edge: RGBEdge, pose_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    Rs, ts = pose_by_frame[edge.source_frame_idx]
    Rt, tt = pose_by_frame[edge.target_frame_idx]
    predicted_R = Rt @ Rs.T
    rotation_error = Rotation.from_matrix(predicted_R @ edge.rotation_source_to_target_world.T).as_rotvec()
    if edge.diagnostics.get("translation_only"):
        if edge.translation_reference_rotation is None or edge.translation_source_support_world is None:
            raise ValueError("translation-only edge requires a fixed orientation and source support")
        reference_R = edge.translation_reference_rotation
        c = edge.translation_source_support_world
        predicted_displacement = tt - ts + (reference_R - np.eye(3)) @ (c - ts)
        measured_displacement = (edge.rotation_source_to_target_world - np.eye(3)) @ c + edge.translation_source_to_target_world_m
        # Both orientation-dependent terms are immutable measurements; this
        # conditional translation residual has exactly zero rotation Jacobian.
        return np.zeros(3), predicted_displacement - measured_displacement
    predicted_t = tt - predicted_R @ ts
    return rotation_error, predicted_t - edge.translation_source_to_target_world_m


def _robust_clip(values: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), -float(limit), float(limit))


def rgb_reprojection_residual(
    factor: RGBReprojectionFactor,
    pose_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]],
    nodes: Sequence[PoseNode],
    config: SolverConfig,
) -> np.ndarray:
    """Return observed RGB pixel residuals for one optional evidence factor."""
    source_node = nodes[factor.source_pos]
    target_node = nodes[factor.target_pos]
    source_R, source_t = pose_by_frame[factor.source_frame_idx]
    target_R, target_t = pose_by_frame[factor.target_frame_idx]
    if not factor.rotation_eligible:
        source_R = source_node.initial_rotation
        target_R = target_node.initial_rotation
    source_canonical = inverse_pose(factor.source_world, source_R, source_t)
    world = apply_pose(source_canonical, target_R, target_t)
    camera = inverse_pose(
        world,
        factor.target_camera_rotation,
        factor.target_camera_translation,
    )
    fx, fy, cx, cy = factor.target_intrinsics.tolist()
    z = np.asarray(camera[:, 2], dtype=np.float64)
    safe_z = np.where(np.abs(z) > 1.0e-6, z, np.where(z >= 0.0, 1.0e-6, -1.0e-6))
    projected = np.column_stack((fx * camera[:, 0] / safe_z + cx, fy * camera[:, 1] / safe_z + cy))
    error = projected - factor.target_uv
    error[~np.isfinite(error)] = float(config.max_rgb_reprojection_residual_px)
    error = _robust_clip(error, config.max_rgb_reprojection_residual_px)
    scale = math.sqrt(max(float(config.rgb_reprojection_weight_scale), 0.0) * max(float(factor.edge_weight), 0.0))
    scale *= 1.0 / max(float(config.sigma_rgb_reprojection_px), 1.0e-8)
    scale /= math.sqrt(max(1, len(error)))
    return (scale * np.sqrt(np.clip(factor.weights, 0.0, 1.0))[:, None] * error).reshape(-1)


def pose_residual(
    x: np.ndarray,
    nodes: Sequence[PoseNode],
    rgb_edges: Sequence[RGBEdge],
    observed_factors: Sequence[PointFactor],
    anchor_frame: int,
    config: SolverConfig | None = None,
    reprojection_factors: Sequence[RGBReprojectionFactor] | None = None,
    material_factors: Sequence[material.MaterialFactor] | None = None,
) -> np.ndarray:
    """The complete pose objective; generated mesh factors cannot enter here."""
    cfg = config or SolverConfig()
    pose_by_frame = current_poses(nodes, x)
    corrections = unpack_corrections(x, len(nodes))
    blocks: list[np.ndarray] = []
    by_frame = {node.frame_idx: i for i, node in enumerate(nodes)}
    for correction in corrections:
        blocks.append(correction[:3] / max(cfg.sigma_pose_rotation_rad, 1.0e-8))
        blocks.append(correction[3:] / max(cfg.sigma_pose_translation_m, 1.0e-8))
    for edge in rgb_edges:
        rotation_error, translation_error = rgb_edge_residual(edge, pose_by_frame)
        weight = math.sqrt(max(float(edge.weight), 0.0))
        edge_rotation_scale = _finite_float(
            edge.diagnostics.get("rotation_weight_scale", 1.0), 1.0
        )
        edge_translation_scale = _finite_float(
            edge.diagnostics.get("translation_weight_scale", 1.0), 1.0
        )
        blocks.extend([
            weight * math.sqrt(max(cfg.rgb_rotation_weight_scale * edge_rotation_scale, 0.0)) * rotation_error / max(cfg.sigma_rgb_rotation_rad, 1.0e-8),
            weight * math.sqrt(max(cfg.rgb_translation_weight_scale * edge_translation_scale, 0.0)) * translation_error / max(cfg.sigma_rgb_translation_m, 1.0e-8),
        ])
    if float(cfg.rgb_reprojection_weight_scale) > 0.0:
        for reprojection_factor in reprojection_factors or ():
            blocks.append(rgb_reprojection_residual(reprojection_factor, pose_by_frame, nodes, cfg))
    for factor in material_factors or ():
        blocks.append(material.residual(factor, pose_by_frame,
            image_weight=cfg.material_image_weight, metric_weight=cfg.material_metric_weight,
            sigma_px=cfg.sigma_material_px, sigma_m=cfg.sigma_material_m))
    for factor in observed_factors:
        source_node = nodes[factor.source_pos]
        target_node = nodes[factor.target_pos]
        source_R, source_t = pose_by_frame[source_node.frame_idx]
        target_R, target_t = pose_by_frame[target_node.frame_idx]
        # P09 is allowed to constrain translation without becoming a rotation
        # authority.  In that mode the correspondence residual uses the
        # immutable per-frame orientation and the candidate translation.
        if cfg.observed_factor_rotation_scale <= 0.0:
            source_R = source_node.initial_rotation
            target_R = target_node.initial_rotation
        source_c = inverse_pose(factor.source_world, source_R, source_t)
        target_c = inverse_pose(factor.target_world, target_R, target_t)
        difference = source_c - target_c
        plane = np.einsum("ij,ij->i", factor.target_normals, difference)
        normalization = math.sqrt(max(1, len(plane)))
        weights = np.sqrt(np.clip(factor.weights, 0.0, 1.0))
        point_scale = math.sqrt(max(cfg.observed_factor_weight_scale, 0.0))
        blocks.append(point_scale * weights * _robust_clip(plane, cfg.max_point_residual_m) / max(cfg.sigma_point_plane_m, 1.0e-8) / normalization)
        blocks.append(point_scale * (weights[:, None] * _robust_clip(difference, cfg.max_point_residual_m)).reshape(-1) / max(cfg.sigma_point_point_m, 1.0e-8) / normalization)
    # Absolute-pose velocity and acceleration terms are centered on the
    # immutable initial trajectory, preserving real object motion rather than
    # pulling every frame toward zero velocity.
    ordered = sorted(nodes, key=lambda n: n.frame_idx)
    velocities_t: list[np.ndarray] = []; velocities_r: list[np.ndarray] = []
    initial_velocities_t: list[np.ndarray] = []; initial_velocities_r: list[np.ndarray] = []
    for previous, current in zip(ordered[:-1], ordered[1:]):
        dt = float(max(1, current.frame_idx - previous.frame_idx))
        R0, t0 = pose_by_frame[previous.frame_idx]; R1, t1 = pose_by_frame[current.frame_idx]
        velocities_t.append((t1 - t0) / dt)
        velocities_r.append(Rotation.from_matrix(R1 @ R0.T).as_rotvec() / dt)
        initial_velocities_t.append((current.initial_translation - previous.initial_translation) / dt)
        initial_velocities_r.append(Rotation.from_matrix(current.initial_rotation @ previous.initial_rotation.T).as_rotvec() / dt)
        blocks.extend([(velocities_t[-1] - initial_velocities_t[-1]) / max(cfg.sigma_velocity_translation_m, 1.0e-8), (velocities_r[-1] - initial_velocities_r[-1]) / max(cfg.sigma_velocity_rotation_rad, 1.0e-8)])
    for i in range(1, len(velocities_t)):
        blocks.extend([(velocities_t[i] - velocities_t[i - 1]) / max(cfg.sigma_acceleration_translation_m, 1.0e-8), (velocities_r[i] - velocities_r[i - 1]) / max(cfg.sigma_acceleration_rotation_rad, 1.0e-8)])
    for previous, current in zip(ordered[:-1], ordered[1:]):
        p = by_frame[previous.frame_idx]; c = by_frame[current.frame_idx]; dt = float(max(1, current.frame_idx - previous.frame_idx))
        blocks.extend([(corrections[c, 3:] - corrections[p, 3:]) / dt / max(cfg.sigma_correction_velocity_translation_m, 1.0e-8), (corrections[c, :3] - corrections[p, :3]) / dt / max(cfg.sigma_correction_velocity_rotation_rad, 1.0e-8)])
    # Sole gauge constraint.  Other rows intentionally have no gauge prior.
    if int(anchor_frame) not in by_frame:
        raise ValueError(f"anchor frame {anchor_frame} is not a timeline node")
    anchor = corrections[by_frame[int(anchor_frame)]]
    blocks.append(anchor / np.asarray([cfg.sigma_anchor_rotation_rad] * 3 + [cfg.sigma_anchor_translation_m] * 3))
    return np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in blocks]) if blocks else np.empty(0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Candidate gates and immutable-state handling.


def pose_metrics(nodes: Sequence[PoseNode], pose_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]], anchor_frame: int) -> dict[str, Any]:
    direct = [node for node in nodes if node.metric_observation and node.observed_world is not None and node.frame_idx in pose_by_frame]
    distances: list[float] = []
    anchor = next((node for node in direct if node.frame_idx == int(anchor_frame)), direct[0] if direct else None)
    if anchor is not None:
        aR, at = pose_by_frame[anchor.frame_idx]
        anchor_c = inverse_pose(anchor.observed_world, aR, at)
        tree = cKDTree(anchor_c)
        for node in direct:
            R, t = pose_by_frame[node.frame_idx]
            canonical = inverse_pose(node.observed_world, R, t)
            distances.extend(tree.query(canonical, k=1, workers=-1)[0].tolist())
    ordered = sorted(nodes, key=lambda n: n.frame_idx)
    step_rot: list[float] = []
    step_trans: list[float] = []
    accel_rot: list[float] = []
    accel_trans: list[float] = []
    velocities_r: list[np.ndarray] = []
    velocities_t: list[np.ndarray] = []
    for prev, cur in zip(ordered[:-1], ordered[1:]):
        Rp, tp = pose_by_frame[prev.frame_idx]; Rc, tc = pose_by_frame[cur.frame_idx]; dt = float(max(1, cur.frame_idx - prev.frame_idx))
        velocities_r.append(Rotation.from_matrix(Rc @ Rp.T).as_rotvec() / dt); velocities_t.append((tc - tp) / dt)
        step_rot.append(float(np.linalg.norm(velocities_r[-1]))); step_trans.append(float(np.linalg.norm(velocities_t[-1])))
    for i in range(1, len(velocities_r)):
        accel_rot.append(float(np.linalg.norm(velocities_r[i] - velocities_r[i - 1])))
        accel_trans.append(float(np.linalg.norm(velocities_t[i] - velocities_t[i - 1])))
    return {
        "observed_surface_abs_median_m": float(np.median(distances)) if distances else None,
        "observed_surface_abs_p95_m": float(np.percentile(distances, 95)) if distances else None,
        "continuity_max_step_rotation_rad": max(step_rot, default=0.0),
        "continuity_max_step_translation_m": max(step_trans, default=0.0),
        "continuity_max_acceleration_rotation_rad": max(accel_rot, default=0.0),
        "continuity_max_acceleration_translation_m": max(accel_trans, default=0.0),
        "metric_frame_count": len(direct),
        "timeline_frame_count": len(nodes),
    }


def candidate_gate(
    candidate_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    immutable_initial_metrics: Mapping[str, Any],
    config: SolverConfig | None = None,
    generated_validation: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Gate against both baselines without mutating the accepted state."""
    cfg = config or SolverConfig()
    reasons: list[str] = []
    candidate_surface = candidate_metrics.get("observed_surface_abs_median_m")
    for label, baseline in (("current", current_metrics.get("observed_surface_abs_median_m")), ("immutable_initial", immutable_initial_metrics.get("observed_surface_abs_median_m"))):
        if cfg.observed_factor_mode == "legacy_nn" and candidate_surface is not None and baseline is not None and float(candidate_surface) > float(baseline) + float(cfg.max_metric_degradation_m):
            reasons.append(f"observed_surface_degradation_vs_{label}")
    if cfg.observed_factor_mode == "material_tracks":
        if not candidate_metrics.get("material_metric_count"):
            reasons.append("material_metric_evidence_unavailable")
        if candidate_metrics.get("material_behind_camera_count", 0):
            reasons.append("material_cheirality_gate")
        for label, baseline in (("current", current_metrics), ("immutable_initial", immutable_initial_metrics)):
            for key, allowance in (("material_metric_median_m", cfg.max_metric_degradation_m),
                                   ("material_metric_p95_m", cfg.max_metric_degradation_m),
                                   ("material_pixel_median_px", cfg.max_material_pixel_degradation_px)):
                value = candidate_metrics.get(key); previous = baseline.get(key)
                if value is None or previous is None or not math.isfinite(float(value)):
                    reasons.append(f"material_validation_unavailable:{key}")
                elif float(value) > float(previous) + allowance:
                    reasons.append(f"{key}_degradation_vs_{label}")
    if float(candidate_metrics.get("continuity_max_step_rotation_rad", 0.0)) > float(cfg.max_continuity_rotation_rad):
        reasons.append("rotation_continuity_gate")
    if float(candidate_metrics.get("continuity_max_step_translation_m", 0.0)) > float(cfg.max_continuity_translation_m):
        reasons.append("translation_continuity_gate")
    if float(candidate_metrics.get("continuity_max_acceleration_rotation_rad", 0.0)) > float(cfg.max_acceleration_rotation_rad):
        reasons.append("rotation_acceleration_gate")
    if float(candidate_metrics.get("continuity_max_acceleration_translation_m", 0.0)) > float(cfg.max_acceleration_translation_m):
        reasons.append("translation_acceleration_gate")
    if generated_validation:
        if generated_validation.get("common_hit_gate_pass") is False:
            reasons.append("generated_common_hit_validation_gate")
        if generated_validation.get("coverage_gate_pass") is False:
            reasons.append("generated_coverage_validation_gate")
        if generated_validation.get("silhouette_gate_pass") is False:
            reasons.append("generated_silhouette_validation_gate")
        if generated_validation.get("signed_front_bias_gate_pass") is False:
            reasons.append("signed_front_bias_validation_gate")
        if generated_validation.get("signed_front_bias_gate_reasons"):
            reasons.extend(str(value) for value in generated_validation["signed_front_bias_gate_reasons"] if str(value) not in reasons)
    return not reasons, reasons


def select_candidate_state(current_x: np.ndarray, candidate_x: np.ndarray, accepted: bool) -> np.ndarray:
    """Commit only an accepted candidate; rejected arrays never leak."""
    return np.asarray(candidate_x if accepted else current_x, dtype=np.float64).copy()


def candidate_correction_diagnostics(
    nodes: Sequence[PoseNode],
    x: np.ndarray,
    *,
    rotation_clipped: bool = False,
    translation_clipped: bool = False,
) -> dict[str, Any]:
    corrections = unpack_corrections(x, len(nodes))
    rotation_norms = np.linalg.norm(corrections[:, :3], axis=1)
    translation_norms = np.linalg.norm(corrections[:, 3:], axis=1)
    return {
        "max_total_rotation_from_initial_rad": float(np.max(rotation_norms)) if len(rotation_norms) else 0.0,
        "max_total_translation_from_initial_m": float(np.max(translation_norms)) if len(translation_norms) else 0.0,
        "rotation_clipped": bool(rotation_clipped),
        "translation_clipped": bool(translation_clipped),
        "small_cumulative_rotation_cap_applied": False,
        "broad_rotation_allowance_rad": 2.0 * math.pi,
    }


def pose_residual_sparsity(
    nodes: Sequence[PoseNode],
    rgb_edges: Sequence[RGBEdge],
    observed_factors: Sequence[PointFactor],
    anchor_frame: int,
    reprojection_factors: Sequence[RGBReprojectionFactor] | None = None,
    material_factors: Sequence[material.MaterialFactor] | None = None,
) -> Any:
    """Return sparsity matching :func:`pose_residual` row order."""
    from scipy import sparse

    positions = {node.frame_idx: index for index, node in enumerate(nodes)}
    entries: list[tuple[int, int]] = []
    row = 0

    def add(count: int, positions_for_row: Sequence[int]) -> None:
        nonlocal row
        for current_row in range(row, row + int(count)):
            for position in positions_for_row:
                entries.extend(
                    (current_row, column)
                    for column in range(6 * position, 6 * position + 6)
                )
        row += int(count)

    for index in range(len(nodes)):
        add(6, [index])
    for edge in rgb_edges:
        add(3, [positions[edge.source_frame_idx], positions[edge.target_frame_idx]])
        add(3, [positions[edge.source_frame_idx], positions[edge.target_frame_idx]])
    if any(reprojection_factors or ()):
        for reprojection_factor in reprojection_factors or ():
            add(2 * len(reprojection_factor.weights), [reprojection_factor.source_pos, reprojection_factor.target_pos])
    for factor in material_factors or ():
        add(6 * len(factor.weights), [factor.source_pos, factor.target_pos])
    for factor in observed_factors:
        add(len(factor.weights), [factor.source_pos, factor.target_pos])
        add(3 * len(factor.weights), [factor.source_pos, factor.target_pos])
    for index in range(1, len(nodes)):
        add(3, [index - 1, index])
        add(3, [index - 1, index])
    for index in range(1, len(nodes) - 1):
        add(3, [index - 1, index, index + 1])
        add(3, [index - 1, index, index + 1])
    for index in range(1, len(nodes)):
        add(3, [index - 1, index])
        add(3, [index - 1, index])
    if int(anchor_frame) not in positions:
        raise ValueError(f"anchor frame {anchor_frame} is not a timeline node")
    add(6, [positions[int(anchor_frame)]])
    row_indices, column_indices = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix(
        (np.ones(len(row_indices), dtype=bool), (row_indices, column_indices)),
        shape=(row, 6 * len(nodes)),
    )


def solve_window(
    nodes: Sequence[PoseNode],
    rgb_edges: Sequence[RGBEdge],
    anchor_frame: int,
    config: SolverConfig | None = None,
    immutable_initial_metrics: Mapping[str, Any] | None = None,
    generated_validation_callback: Any | None = None,
    reprojection_factors: Sequence[RGBReprojectionFactor] | None = None,
    material_factors: Sequence[material.MaterialFactor] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = config or SolverConfig()
    if not nodes:
        raise ValueError("cannot solve empty window")
    frame_ids = {node.frame_idx for node in nodes}
    if int(anchor_frame) not in frame_ids:
        raise ValueError("anchor is outside solve window")
    current_x = np.zeros(len(nodes) * 6, dtype=np.float64)
    active_reprojection_factors = (
        list(reprojection_factors or ())
        if float(cfg.rgb_reprojection_weight_scale) > 0.0
        else []
    )
    if cfg.observed_factor_mode == "material_tracks" and not material_factors:
        raise ValueError("material mode requires fixed RGB-D observations")
    def all_metrics(pose):
        values = pose_metrics(nodes, pose, anchor_frame)
        if material_factors:
            values.update(material.metrics(material_factors, pose))
        return values
    initial_pose = current_poses(nodes, current_x)
    initial_factors, initial_factor_diag = build_observed_point_factors(
        nodes, initial_pose, anchor_frame, cfg, 0
    )
    initial_metrics = dict(
        immutable_initial_metrics
        or all_metrics(initial_pose)
    )
    current_metrics = all_metrics(initial_pose)
    outer_reports: list[dict[str, Any]] = []
    rotation_update_clipped = False
    translation_update_clipped = False
    step_scales = (1.0, 0.5, 0.25, 0.1, 0.05)

    for outer in range(int(cfg.outer_iterations)):
        current_pose = current_poses(nodes, current_x)
        factors, factor_diag = build_observed_point_factors(
            nodes, current_pose, anchor_frame, cfg, outer
        )
        residual = lambda values: pose_residual(
            values, nodes, rgb_edges, factors, anchor_frame, cfg, active_reprojection_factors, material_factors
        )
        before = residual(current_x)
        before_cost = float(before @ before)
        sparsity = pose_residual_sparsity(
            nodes, rgb_edges, factors, anchor_frame, active_reprojection_factors, material_factors
        )
        if sparsity.shape != (len(before), len(current_x)):
            raise RuntimeError(
                f"late-window sparsity mismatch {sparsity.shape} != {(len(before), len(current_x))}"
            )
        anchor_pos = next(
            index for index, node in enumerate(nodes)
            if node.frame_idx == int(anchor_frame)
        )
        lower = np.full(len(current_x), -np.inf, dtype=np.float64)
        upper = np.full(len(current_x), np.inf, dtype=np.float64)
        for position in range(len(nodes)):
            lower[6 * position : 6 * position + 3] = (
                current_x[6 * position : 6 * position + 3]
                - float(cfg.update_rotation_trust_rad)
            )
            upper[6 * position : 6 * position + 3] = (
                current_x[6 * position : 6 * position + 3]
                + float(cfg.update_rotation_trust_rad)
            )
            lower[6 * position + 3 : 6 * position + 6] = (
                current_x[6 * position + 3 : 6 * position + 6]
                - float(cfg.update_translation_trust_m)
            )
            upper[6 * position + 3 : 6 * position + 6] = (
                current_x[6 * position + 3 : 6 * position + 6]
                + float(cfg.update_translation_trust_m)
            )
        # scipy requires strict lower < upper, so use a tiny interval for the
        # sole gauge-fixed anchor and verify the resulting correction below.
        anchor_slice = 6 * anchor_pos
        anchor_epsilon = 1.0e-12
        lower[anchor_slice : anchor_slice + 6] = -anchor_epsilon
        upper[anchor_slice : anchor_slice + 6] = anchor_epsilon
        result = least_squares(
            residual,
            current_x,
            jac_sparsity=sparsity,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(cfg.inner_iterations),
            xtol=1.0e-7,
            ftol=1.0e-7,
            gtol=1.0e-7,
        )
        optimizer_delta = np.abs(np.asarray(result.x, dtype=np.float64) - current_x)
        rotation_update_clipped = rotation_update_clipped or bool(
            np.any(optimizer_delta.reshape(len(nodes), 6)[:, :3] >= float(cfg.update_rotation_trust_rad) - 1.0e-7)
        )
        translation_update_clipped = translation_update_clipped or bool(
            np.any(optimizer_delta.reshape(len(nodes), 6)[:, 3:] >= float(cfg.update_translation_trust_m) - 1.0e-7)
        )
        trial_reports: list[dict[str, Any]] = []
        selected: tuple[np.ndarray, dict[str, Any], dict[int, tuple[np.ndarray, np.ndarray]]] | None = None
        for scale in step_scales:
            candidate_x = current_x + float(scale) * (np.asarray(result.x) - current_x)
            candidate_pose = current_poses(nodes, candidate_x)
            candidate_metrics = all_metrics(candidate_pose)
            generated_validation = (
                generated_validation_callback(candidate_pose, outer)
                if generated_validation_callback is not None
                else {}
            )
            accepted, reasons = candidate_gate(
                candidate_metrics,
                current_metrics,
                initial_metrics,
                cfg,
                generated_validation,
            )
            if not bool(result.success):
                accepted = False
                reasons = list(reasons) + ["optimizer_not_success"]
            trial_residual = residual(candidate_x)
            trial_cost = float(trial_residual @ trial_residual)
            robust_before = float(np.sum(2.0 * (np.sqrt(1.0 + before * before) - 1.0)))
            robust_after = float(np.sum(2.0 * (np.sqrt(1.0 + trial_residual * trial_residual) - 1.0)))
            if robust_after > robust_before + 1e-9 * max(1.0, robust_before):
                accepted = False
                reasons = list(reasons) + ["robust_pose_objective_increased"]
            trial = {
                "step_scale": float(scale),
                "candidate_accepted": bool(accepted),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "optimizer_nfev": int(result.nfev),
                "cost_before": before_cost,
                "candidate_cost": trial_cost,
                "robust_cost_before": robust_before,
                "robust_candidate_cost": robust_after,
                "rejection_reasons": list(dict.fromkeys(reasons)),
                "candidate_metrics": candidate_metrics,
                "generated_validation": generated_validation,
                "candidate_correction": candidate_correction_diagnostics(nodes, candidate_x),
            }
            trial_reports.append(trial)
            if accepted:
                selected = (candidate_x, trial, candidate_pose)
                break
        if selected is None:
            unchanged_x = current_x.copy()
            outer_reports.append(
                {
                    "outer_iteration": int(outer),
                    "optimizer_success": bool(result.success),
                    "accepted_update": False,
                    "factor_rebuild": factor_diag,
                    "trials": trial_reports,
                    "state_leak_check": bool(np.array_equal(current_x, unchanged_x)),
                }
            )
            break
        selected_x, selected_trial, selected_pose = selected
        old_x = current_x.copy()
        current_x = select_candidate_state(current_x, selected_x, True)
        current_metrics = selected_trial["candidate_metrics"]
        if generated_validation_callback is not None and hasattr(
            generated_validation_callback, "commit"
        ):
            generated_validation_callback.commit(selected_pose)
        outer_reports.append(
            {
                "outer_iteration": int(outer),
                "optimizer_success": bool(result.success),
                "accepted_update": True,
                "factor_rebuild": factor_diag,
                "selected_step_scale": selected_trial["step_scale"],
                "selected_candidate_metrics": current_metrics,
                "trials": trial_reports,
                "state_leak_check": bool(np.array_equal(current_x, selected_x)),
            }
        )
    return current_x, {
        "outer_iterations": outer_reports,
        "initial_observed_factor_rebuild": initial_factor_diag,
        "accepted_x": current_x.copy(),
        "accepted_metrics": current_metrics,
        "immutable_initial_metrics": initial_metrics,
        "rotation_update_trust_clipped": bool(rotation_update_clipped),
        "translation_update_trust_clipped": bool(translation_update_clipped),
    }


# ---------------------------------------------------------------------------
# Optional generated validation and exact mesh contract.


def generated_mesh_contract(mesh_path: Path) -> dict[str, Any]:
    """Record the exact validation mesh without making it pose authority."""
    resolved = mesh_path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RuntimeError(f"validation mesh does not exist: {resolved}")
    try:
        import trimesh  # type: ignore
        mesh = trimesh.load(resolved, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([value for value in mesh.geometry.values() if isinstance(value, trimesh.Trimesh)])
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not np.isfinite(vertices).all():
            raise RuntimeError("validation mesh arrays are invalid")
        vertex_count, face_count = len(vertices), len(faces)
    except ImportError:
        vertex_count = face_count = None
    return {
        "path": str(resolved), "sha256": sha256_file(resolved),
        "vertex_count": vertex_count, "face_count": face_count,
        "contract": "exact supplied mesh path/hash; all faces retained; validation-only",
        "generated_geometry_pose_authority": False,
        "generated_factors_in_pose_objective": False,
    }


def rebuild_generated_first_hit_validation(
    validation_context: Any,
    pose_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]],
    args: Any,
) -> dict[str, Any]:
    """Rebuild generated factors for validation only, never pose objective."""
    factors, metrics = validation_context.module.rebuild_factors(
        validation_context.context, dict(pose_by_frame), args
    )
    return {
        "enabled": True,
        "validation_available": bool(factors) and bool(metrics.get("per_frame")),
        "validation_status": "available" if factors else "validation-unavailable",
        "factor_count": int(sum(len(value.depth_observed_z) for value in factors.values())),
        "metrics": metrics,
        "pose_objective_inclusion": False,
    }


def _factor_signed_values(factor: Any) -> np.ndarray:
    raw = getattr(factor, "evaluation_signed_depth", None)
    if raw is None:
        return np.empty(0, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def signed_front_bias_diagnostics(
    factors: Mapping[int, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    *,
    negative_threshold_m: float = 0.005,
) -> dict[str, Any]:
    """Summarize signed first-hit depth at aggregate, frame, and segment levels.

    Negative generated-minus-observed depth means the generated first hit is in
    front of the observed surface.  The aggregate statistic is intentionally
    retained for compatibility, but per-frame and contiguous-segment values
    prevent a late failure from being hidden by an all-frame median.  This is
    visible-pose/render evidence only; it is never a pose residual or physical
    geometry authority.
    """
    values: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    if factors:
        for frame_idx in sorted(factors):
            frame_values = _factor_signed_values(factors[frame_idx])
            if len(frame_values):
                values.append(frame_values)
                negative = frame_values[frame_values < 0.0]
                per_frame.append({
                    "frame_idx": int(frame_idx),
                    "sample_count": int(len(frame_values)),
                    "signed_median_m": float(np.median(frame_values)),
                    "signed_p95_m": float(np.percentile(frame_values, 95)),
                    "front_fraction": float(len(negative) / len(frame_values)),
                    "front_bias_m": float(max(0.0, -np.median(negative))) if len(negative) else 0.0,
                    "negative_front": bool(float(np.median(frame_values)) < -float(negative_threshold_m)),
                })
    if not per_frame and isinstance(metrics, Mapping):
        for row in metrics.get("per_frame", []) or []:
            if not isinstance(row, Mapping) or row.get("frame_idx") is None:
                continue
            summary = row.get("true_first_hit_signed_depth_m")
            if not isinstance(summary, Mapping) or int(summary.get("count") or 0) <= 0:
                continue
            median = _finite_float(summary.get("median"), float("nan"))
            if not math.isfinite(median):
                continue
            per_frame.append({
                "frame_idx": int(row["frame_idx"]),
                "sample_count": int(summary.get("count") or 0),
                "signed_median_m": median,
                "signed_p95_m": _finite_float(summary.get("p95"), float("nan")),
                "front_fraction": None,
                "front_bias_m": max(0.0, -median),
                "negative_front": bool(median < -float(negative_threshold_m)),
            })
    signed = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    if len(signed):
        front = signed[signed < 0.0]
        signed_median = float(np.median(signed))
        front_fraction = float(len(front) / len(signed))
        front_bias = float(max(0.0, -np.median(front))) if len(front) else 0.0
    else:
        summary = (metrics or {}).get("true_first_hit_signed_depth_m", {}) if isinstance(metrics, Mapping) else {}
        signed_median = _finite_float(summary.get("median"), float("nan")) if isinstance(summary, Mapping) else float("nan")
        sample_count = int(summary.get("count") or 0) if isinstance(summary, Mapping) else 0
        front_fraction = None
        front_bias = max(0.0, -signed_median) if math.isfinite(signed_median) else None
        if not per_frame:
            return {
                "sample_count": sample_count,
                "signed_median_m": signed_median if math.isfinite(signed_median) else None,
                "front_fraction": front_fraction,
                "front_bias_m": front_bias,
                "negative_threshold_m": float(negative_threshold_m),
                "per_frame": [],
                "negative_front_segments": [],
                "sustained_negative_front_segments": [],
                "max_negative_front_segment_length": 0,
                "signed_front_definition": "negative generated_depth_minus_observed_depth means generated is in front",
            }

    segments: list[dict[str, Any]] = []
    for row in sorted(per_frame, key=lambda item: int(item["frame_idx"])):
        is_negative = bool(row.get("negative_front"))
        if not is_negative:
            continue
        if not segments or int(row["frame_idx"]) != int(segments[-1]["right_frame"]) + 1:
            segments.append({
                "left_frame": int(row["frame_idx"]),
                "right_frame": int(row["frame_idx"]),
                "frame_count": 1,
                "frame_indices": [int(row["frame_idx"])],
                "median_signed_depth_m": [float(row["signed_median_m"])],
                "front_bias_m": [float(row["front_bias_m"])],
            })
        else:
            segment = segments[-1]
            segment["right_frame"] = int(row["frame_idx"])
            segment["frame_count"] += 1
            segment["frame_indices"].append(int(row["frame_idx"]))
            segment["median_signed_depth_m"].append(float(row["signed_median_m"]))
            segment["front_bias_m"].append(float(row["front_bias_m"]))
    for segment in segments:
        signed_values = segment.pop("median_signed_depth_m")
        bias_values = segment.pop("front_bias_m")
        segment["median_signed_depth_m"] = float(np.median(signed_values))
        segment["max_front_bias_m"] = float(np.max(bias_values))
        segment["sustained"] = bool(segment["frame_count"] >= 3)
    return {
        "sample_count": int(len(signed)) if len(signed) else int(sum(int(row.get("sample_count") or 0) for row in per_frame)),
        "signed_median_m": signed_median if math.isfinite(signed_median) else None,
        "front_fraction": front_fraction,
        "front_bias_m": front_bias,
        "negative_threshold_m": float(negative_threshold_m),
        "per_frame": per_frame,
        "negative_front_segments": segments,
        "sustained_negative_front_segments": [segment for segment in segments if segment["sustained"]],
        "max_negative_front_segment_length": max((int(segment["frame_count"]) for segment in segments), default=0),
        "signed_front_definition": "negative generated_depth_minus_observed_depth means generated is in front",
    }


def signed_front_bias_gate(
    candidate: Mapping[str, Any],
    current: Mapping[str, Any] | None = None,
    immutable_initial: Mapping[str, Any] | None = None,
    *,
    max_front_bias_m: float | None = 0.03,
    max_front_fraction: float | None = 0.95,
    max_increase_m: float = 0.005,
    max_per_frame_front_bias_m: float | None = None,
    max_negative_front_segment_length: int | None = None,
) -> tuple[bool, list[str]]:
    """Gate render evidence against current/fixed initial front bias.

    Aggregate checks are retained for backward compatibility.  Optional
    per-frame and sustained-segment checks are deliberately explicit because
    they are a visible-pose/render gate, not generated-geometry pose authority.
    """
    reasons: list[str] = []
    bias = candidate.get("front_bias_m")
    fraction = candidate.get("front_fraction")
    if "sample_count" in candidate and int(candidate.get("sample_count") or 0) <= 0:
        reasons.append("signed_front_validation_unavailable")
    if bias is None or not math.isfinite(float(bias)):
        reasons.append("signed_front_validation_unavailable")
    if (
        max_front_fraction is not None
        and (fraction is None or not math.isfinite(float(fraction)))
    ):
        reasons.append("signed_front_fraction_validation_unavailable")
    if (
        max_front_bias_m is not None
        and bias is not None
        and math.isfinite(float(bias))
        and float(bias) > float(max_front_bias_m)
    ):
        reasons.append("signed_front_bias_absolute_gate")
    if (
        max_front_fraction is not None
        and fraction is not None
        and math.isfinite(float(fraction))
        and float(fraction) > float(max_front_fraction)
    ):
        reasons.append("signed_front_fraction_gate")
    if max_per_frame_front_bias_m is not None:
        for row in candidate.get("per_frame", []) or []:
            row_bias = row.get("front_bias_m") if isinstance(row, Mapping) else None
            if row_bias is not None and math.isfinite(float(row_bias)) and float(row_bias) > float(max_per_frame_front_bias_m):
                reasons.append(f"signed_front_per_frame_absolute_gate:{int(row.get('frame_idx', -1))}")
    if max_negative_front_segment_length is not None:
        for segment in candidate.get("negative_front_segments", []) or []:
            if int(segment.get("frame_count") or 0) > int(max_negative_front_segment_length):
                reasons.append(
                    "signed_front_sustained_segment_gate:"
                    f"{int(segment.get('left_frame', -1))}-{int(segment.get('right_frame', -1))}"
                )
    for label, baseline in (("current", current), ("immutable_initial", immutable_initial)):
        if baseline is None or bias is None or baseline.get("front_bias_m") is None:
            continue
        if float(bias) > float(baseline["front_bias_m"]) + float(max_increase_m):
            reasons.append(f"signed_front_bias_degradation_vs_{label}")
    return not reasons, list(dict.fromkeys(reasons))


def _make_generated_args(args: argparse.Namespace) -> SimpleNamespace:
    """Populate the audited generated-factor adapter without touching pose code."""
    values = vars(args).copy()
    values.update({
        "generated_visible_mesh": args.validation_mesh,
        "generated_visible_factor_builder_script": args.generated_factor_builder_script,
        "generated_visible_hand_npz": args.generated_visible_hand_npz,
        "generated_visible_mano_faces_pkl": args.generated_visible_mano_faces_pkl,
        "generated_visible_factor_module": args.generated_factor_module,
        "generated_visible_raster_size": args.generated_visible_raster_size,
        "generated_visible_source_size": args.generated_visible_source_size,
        "generated_visible_min_frames": args.generated_visible_min_frames,
        "generated_visible_min_observed_points": args.generated_visible_min_observed_points,
        "generated_visible_min_ownership_fraction": args.generated_visible_min_ownership_fraction,
        "generated_visible_hand_unknown_dilation_px": args.generated_visible_hand_unknown_dilation_px,
        "generated_visible_max_removed_fraction_for_full_weight": args.generated_visible_max_removed_fraction_for_full_weight,
        "generated_visible_min_depth_factor_weight": args.generated_visible_min_depth_factor_weight,
        "generated_visible_boundary_downweight_radius_px": args.generated_visible_boundary_downweight_radius_px,
        "generated_visible_boundary_weight": args.generated_visible_boundary_weight,
        "generated_visible_max_observed_factors_per_frame": args.generated_visible_max_observed_factors_per_frame,
        "generated_visible_max_outside_factors_per_frame": args.generated_visible_max_outside_factors_per_frame,
        "generated_visible_max_missing_factors_per_frame": args.generated_visible_max_missing_factors_per_frame,
        "generated_visible_render_batch_size": args.generated_visible_render_batch_size,
        "generated_visible_device": args.generated_visible_device,
    })
    return SimpleNamespace(**values)


def load_generated_validation_context(args: argparse.Namespace) -> Any | None:
    """Load the audited first-hit builder only when all validation inputs exist."""
    if args.validation_mesh is None:
        return None
    if args.generated_factor_builder_script is None or args.generated_visible_hand_npz is None or args.generated_visible_mano_faces_pkl is None:
        return None
    module_path = args.generated_factor_module.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("v20_late_window_generated_validation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated validation module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    generated_args = _make_generated_args(args)
    context = module.create_context(generated_args)
    if context is None:
        return None
    return SimpleNamespace(module=module, context=context, args=generated_args)


def make_generated_validation_callback(
    validation_context: Any,
    nodes: Sequence[PoseNode],
    immutable_pose: Mapping[int, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    generated_args = validation_context.args
    base_pose = _pose_map(nodes)
    immutable_full = dict(base_pose); immutable_full.update(immutable_pose)
    current_factors, current_raw = validation_context.module.rebuild_factors(validation_context.context, base_pose, generated_args)
    immutable_factors, immutable_raw = validation_context.module.rebuild_factors(validation_context.context, immutable_full, generated_args)
    current_diag = {**signed_front_bias_diagnostics(current_factors, current_raw, negative_threshold_m=float(args.generated_negative_front_threshold_m)), "raw_metrics": current_raw}
    immutable_diag = {**signed_front_bias_diagnostics(immutable_factors, immutable_raw, negative_threshold_m=float(args.generated_negative_front_threshold_m)), "raw_metrics": immutable_raw}
    state: dict[str, Any] = {"current_factors": current_factors, "current_raw": current_raw}

    def callback(candidate_pose: Mapping[int, tuple[np.ndarray, np.ndarray]], _outer: int) -> dict[str, Any]:
        try:
            candidate_factors, candidate_raw = validation_context.module.rebuild_factors(validation_context.context, dict(candidate_pose), generated_args)
            comparison = generated_metric_comparison(
                candidate_factors,
                candidate_raw,
                state["current_factors"],
                state["current_raw"],
                immutable_factors,
                immutable_raw,
                args,
            )
            candidate_diag = signed_front_bias_diagnostics(candidate_factors, candidate_raw, negative_threshold_m=float(args.generated_negative_front_threshold_m))
            return {
                **candidate_diag,
                "validation_available": bool(comparison["validation_available"]),
                "validation_status": "available" if comparison["validation_available"] else "validation-unavailable",
                "validation_unavailable_reason": None if comparison["validation_available"] else "missing_common_hit_coverage_silhouette_or_signed_front_evidence",
                "raw_metrics": candidate_raw,
                "comparisons_to_current_and_immutable_initial": comparison["comparisons"],
                "common_hit_gate_pass": bool(comparison["common_hit_gate_pass"]),
                "coverage_gate_pass": bool(comparison["coverage_gate_pass"]),
                "silhouette_gate_pass": bool(comparison["silhouette_gate_pass"]),
                "signed_front_bias_gate_pass": bool(comparison["signed_front_bias_gate_pass"]),
                "signed_front_bias_gate_reasons": list(comparison["signed_front_bias_gate_reasons"]),
                "gate_reasons": list(comparison["gate_reasons"]),
                "pose_objective_inclusion": False,
            }
        except Exception as exc:
            return {
                "validation_available": False,
                "validation_status": "validation-unavailable",
                "validation_unavailable_reason": f"generated_validation_exception:{type(exc).__name__}:{exc}",
                "common_hit_gate_pass": False,
                "coverage_gate_pass": False,
                "silhouette_gate_pass": False,
                "signed_front_bias_gate_pass": False,
                "signed_front_bias_gate_reasons": ["generated_validation_unavailable"],
                "gate_reasons": ["generated_validation_unavailable"],
                "pose_objective_inclusion": False,
            }

    def commit(candidate_pose: Mapping[int, tuple[np.ndarray, np.ndarray]]) -> None:
        """Advance only the accepted current baseline; immutable stays fixed."""
        factors, raw = validation_context.module.rebuild_factors(validation_context.context, dict(candidate_pose), generated_args)
        state["current_factors"] = factors
        state["current_raw"] = raw

    def current_diagnostics() -> dict[str, Any]:
        return {
            **signed_front_bias_diagnostics(
                state["current_factors"],
                state["current_raw"],
                negative_threshold_m=float(args.generated_negative_front_threshold_m),
            ),
            "validation_available": True,
            "validation_status": "available",
            "raw_metrics": state["current_raw"],
        }

    callback.commit = commit
    callback.current_diagnostics = current_diagnostics
    return callback, current_diag, immutable_diag


def _generated_frame_rows(raw: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = raw.get("per_frame", []) if isinstance(raw, Mapping) else []
    return {int(row["frame_idx"]): row for row in rows if isinstance(row, Mapping) and row.get("frame_idx") is not None}


def _generated_common_depth(before: Any, after: Any) -> tuple[float, float, int]:
    before_hit = np.asarray(getattr(before, "evaluation_hit", []), dtype=bool).reshape(-1)
    after_hit = np.asarray(getattr(after, "evaluation_hit", []), dtype=bool).reshape(-1)
    before_signed = np.asarray(getattr(before, "evaluation_signed_depth", []), dtype=np.float64).reshape(-1)
    after_signed = np.asarray(getattr(after, "evaluation_signed_depth", []), dtype=np.float64).reshape(-1)
    if not (before_hit.shape == after_hit.shape == before_signed.shape == after_signed.shape):
        return float("inf"), float("inf"), 0
    common = before_hit & after_hit & np.isfinite(before_signed) & np.isfinite(after_signed)
    if not np.any(common):
        return float("inf"), float("inf"), 0
    before_abs = np.abs(before_signed[common]); after_abs = np.abs(after_signed[common])
    return float(np.median(after_abs) - np.median(before_abs)), float(np.percentile(after_abs, 95) - np.percentile(before_abs, 95)), int(np.count_nonzero(common))


def _summary_number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summary_metric_number(row: Mapping[str, Any], summary_key: str, *, require_count: bool = True) -> float | None:
    value = row.get(summary_key)
    if not isinstance(value, Mapping):
        return None
    if require_count and int(value.get("count") or 0) <= 0:
        return None
    return _summary_number(value, "median")


def _generated_pair_comparison(
    before_factors: Mapping[int, Any],
    before_raw: Mapping[str, Any],
    candidate_factors: Mapping[int, Any],
    candidate_raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two validation renders without treating missing evidence as pass."""
    before_rows = _generated_frame_rows(before_raw)
    candidate_rows = _generated_frame_rows(candidate_raw)
    shared = sorted(set(before_rows) & set(candidate_rows))
    frame_rows: list[dict[str, Any]] = []
    unavailable_frames: list[int] = []
    for idx in shared:
        before_factor = before_factors.get(idx)
        candidate_factor = candidate_factors.get(idx)
        common_available = False
        depth_delta = p95_delta = float("inf")
        common_count = 0
        if before_factor is not None and candidate_factor is not None:
            depth_delta, p95_delta, common_count = _generated_common_depth(before_factor, candidate_factor)
            common_available = common_count > 0 and np.isfinite(depth_delta) and np.isfinite(p95_delta)
        before_row = before_rows[idx]
        candidate_row = candidate_rows[idx]
        before_coverage = _summary_number(before_row, "true_first_hit_coverage_fraction")
        candidate_coverage = _summary_number(candidate_row, "true_first_hit_coverage_fraction")
        before_silhouette = _summary_number(before_row, "initial_silhouette_iou")
        candidate_silhouette = _summary_number(candidate_row, "initial_silhouette_iou")
        coverage_available = before_coverage is not None and candidate_coverage is not None
        silhouette_available = before_silhouette is not None and candidate_silhouette is not None
        if not (common_available and coverage_available and silhouette_available):
            unavailable_frames.append(idx)
        frame_rows.append({
            "frame_idx": idx,
            "common_hit_available": common_available,
            "common_hit_depth_median_delta_m": depth_delta,
            "common_hit_depth_p95_delta_m": p95_delta,
            "common_hit_count": common_count,
            "coverage_available": coverage_available,
            "coverage_delta": (candidate_coverage - before_coverage) if coverage_available else float("nan"),
            "silhouette_available": silhouette_available,
            "silhouette_iou_delta": (candidate_silhouette - before_silhouette) if silhouette_available else float("nan"),
        })

    def _finite_max(key: str) -> float:
        values = [float(row[key]) for row in frame_rows if np.isfinite(row[key])]
        return max(values, default=0.0)

    def _finite_min(key: str) -> float:
        values = [float(row[key]) for row in frame_rows if np.isfinite(row[key])]
        return min(values, default=0.0)

    # A segment is each contiguous run of shared frames.  Segment diagnostics
    # are retained even when unavailable so the report identifies the gap.
    segments: list[dict[str, Any]] = []
    for row in frame_rows:
        if not segments or int(row["frame_idx"]) != int(segments[-1]["right_frame"]) + 1:
            segments.append({"left_frame": int(row["frame_idx"]), "right_frame": int(row["frame_idx"]), "frame_rows": [row]})
        else:
            segments[-1]["right_frame"] = int(row["frame_idx"])
            segments[-1]["frame_rows"].append(row)
    for segment in segments:
        values = segment.pop("frame_rows")
        segment.update({
            "frame_count": len(values),
            "common_hit_available": all(bool(row["common_hit_available"]) for row in values),
            "common_hit_depth_median_delta_m": float(np.median([row["common_hit_depth_median_delta_m"] for row in values])),
            "common_hit_depth_p95_delta_m": float(np.median([row["common_hit_depth_p95_delta_m"] for row in values])),
            "coverage_available": all(bool(row["coverage_available"]) for row in values),
            "min_coverage_delta": _finite_min_from_rows(values, "coverage_delta"),
            "silhouette_available": all(bool(row["silhouette_available"]) for row in values),
            "min_silhouette_iou_delta": _finite_min_from_rows(values, "silhouette_iou_delta"),
        })
    candidate_bias = signed_front_bias_diagnostics(candidate_factors, candidate_raw)
    before_bias = signed_front_bias_diagnostics(before_factors, before_raw)
    return {
        "available": bool(shared) and not unavailable_frames,
        "shared_frame_count": len(shared),
        "unavailable_frame_ids": unavailable_frames,
        "global_common_hit_depth_median_delta_m": float(np.median([row["common_hit_depth_median_delta_m"] for row in frame_rows])) if frame_rows else float("inf"),
        "global_common_hit_depth_p95_delta_m": float(np.median([row["common_hit_depth_p95_delta_m"] for row in frame_rows])) if frame_rows else float("inf"),
        "max_frame_common_hit_depth_median_delta_m": _finite_max("common_hit_depth_median_delta_m"),
        "max_frame_common_hit_depth_p95_delta_m": _finite_max("common_hit_depth_p95_delta_m"),
        "min_frame_coverage_delta": _finite_min("coverage_delta"),
        "min_frame_silhouette_iou_delta": _finite_min("silhouette_iou_delta"),
        "candidate_signed_front_bias": candidate_bias,
        "before_signed_front_bias": before_bias,
        "per_frame": frame_rows,
        "segments": segments,
    }


def _finite_min_from_rows(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return min(values, default=float("inf"))




def generated_metric_comparison(
    candidate_factors: Mapping[int, Any], candidate_raw: Mapping[str, Any],
    current_factors: Mapping[int, Any], current_raw: Mapping[str, Any],
    immutable_factors: Mapping[int, Any], immutable_raw: Mapping[str, Any],
    args: Any,
) -> dict[str, Any]:
    """Compare candidate validation against current and immutable baselines."""
    pairs = {
        "vs_current": _generated_pair_comparison(current_factors, current_raw, candidate_factors, candidate_raw),
        "vs_immutable_initial": _generated_pair_comparison(immutable_factors, immutable_raw, candidate_factors, candidate_raw),
    }
    reasons: list[str] = []
    for label, comparison in pairs.items():
        if comparison["global_common_hit_depth_median_delta_m"] > float(args.generated_max_common_hit_depth_increase_m) or comparison["max_frame_common_hit_depth_median_delta_m"] > float(args.generated_max_common_hit_depth_increase_m):
            reasons.append(f"{label}_global_or_frame_common_hit_median_depth")
        if comparison["global_common_hit_depth_p95_delta_m"] > float(args.generated_max_common_hit_depth_increase_m) or comparison["max_frame_common_hit_depth_p95_delta_m"] > float(args.generated_max_common_hit_depth_increase_m):
            reasons.append(f"{label}_global_or_frame_common_hit_p95_depth")
        if comparison["min_frame_coverage_delta"] < -float(args.generated_max_coverage_decrease):
            reasons.append(f"{label}_per_frame_coverage")
        if comparison["min_frame_silhouette_iou_delta"] < -float(args.generated_max_silhouette_decrease):
            reasons.append(f"{label}_per_frame_silhouette")
        for segment in comparison["segments"]:
            if segment["common_hit_depth_median_delta_m"] > float(args.generated_max_common_hit_depth_increase_m) or segment["common_hit_depth_p95_delta_m"] > float(args.generated_max_common_hit_depth_increase_m):
                reasons.append(f"{label}_segment_common_hit_depth")
            if segment["min_coverage_delta"] < -float(args.generated_max_coverage_decrease):
                reasons.append(f"{label}_segment_coverage")
            if segment["min_silhouette_iou_delta"] < -float(args.generated_max_silhouette_decrease):
                reasons.append(f"{label}_segment_silhouette")
    front_pass, front_reasons = signed_front_bias_gate(
        pairs["vs_current"]["candidate_signed_front_bias"],
        pairs["vs_current"]["before_signed_front_bias"],
        pairs["vs_immutable_initial"]["before_signed_front_bias"],
        max_front_bias_m=(float(args.generated_max_front_bias_m) if args.generated_max_front_bias_m is not None else None),
        max_front_fraction=(float(args.generated_max_front_fraction) if args.generated_max_front_fraction is not None else None),
        max_increase_m=float(args.generated_max_front_bias_increase_m),
        max_per_frame_front_bias_m=(float(args.generated_max_per_frame_front_bias_m) if args.generated_max_per_frame_front_bias_m is not None else None),
        max_negative_front_segment_length=(int(args.generated_max_negative_front_segment_length) if args.generated_max_negative_front_segment_length is not None else None),
    )
    reasons.extend(front_reasons)
    validation_available = bool(
        pairs["vs_current"].get("available")
        and pairs["vs_immutable_initial"].get("available")
    )
    if not validation_available:
        reasons.append("generated_validation_unavailable")
    return {"validation_available": validation_available, "comparisons": pairs, "common_hit_gate_pass": not reasons, "coverage_gate_pass": not any("coverage" in reason for reason in reasons), "silhouette_gate_pass": not any("silhouette" in reason for reason in reasons), "signed_front_bias_gate_pass": front_pass, "signed_front_bias_gate_reasons": front_reasons, "gate_reasons": reasons}


# ---------------------------------------------------------------------------
# CLI orchestration and report writing.


def _build_config(args: argparse.Namespace) -> SolverConfig:
    cfg = SolverConfig()
    for field_name in ("outer_iterations", "inner_iterations", "max_points", "min_pairs", "seed"):
        setattr(cfg, field_name, int(getattr(args, field_name)))
    for field_name in ("sigma_pose_rotation_rad", "sigma_pose_translation_m", "sigma_rgb_rotation_rad", "sigma_rgb_translation_m", "sigma_point_plane_m", "sigma_point_point_m", "sigma_velocity_rotation_rad", "sigma_velocity_translation_m", "sigma_acceleration_rotation_rad", "sigma_acceleration_translation_m", "sigma_correction_velocity_rotation_rad", "sigma_correction_velocity_translation_m", "max_point_residual_m", "update_rotation_trust_rad", "update_translation_trust_m", "max_base_cycle_rotation_deg", "max_base_cycle_translation_m", "min_rgb_edge_weight", "max_metric_degradation_m", "max_continuity_rotation_deg", "max_continuity_translation_m", "max_acceleration_rotation_deg", "max_acceleration_translation_m"):
        if not hasattr(args, field_name):
            continue
        value = float(getattr(args, field_name))
        target = field_name
        if field_name.endswith("_deg"):
            target = field_name[:-4] + "_rad"
            value = math.radians(value)
        if field_name == "max_base_cycle_translation_m": target = "max_base_cycle_translation_m"
        setattr(cfg, target, value)
    cfg.observed_factor_mode = args.observed_factor_mode
    cfg.material_image_weight = float(args.material_image_weight)
    cfg.material_metric_weight = float(args.material_metric_weight)
    cfg.sigma_material_px = float(args.sigma_material_px)
    cfg.sigma_material_m = float(args.sigma_material_m)
    cfg.max_material_pixel_degradation_px = float(args.max_material_pixel_degradation_px)
    if min(cfg.material_image_weight, cfg.material_metric_weight) < 0 or min(cfg.sigma_material_px, cfg.sigma_material_m) <= 0:
        raise ValueError("material weights must be nonnegative and sigmas positive")
    cfg.source_conditioning_full_weight_scale = float(args.source_conditioning_full_weight_scale)
    cfg.min_rgb_translation_edge_weight = float(args.min_rgb_translation_edge_weight)
    cfg.min_rgb_rotation_conditioning = float(args.min_rgb_rotation_conditioning)
    cfg.observed_factor_weight_scale = float(args.observed_factor_weight_scale)
    cfg.observed_factor_rotation_scale = float(args.observed_factor_rotation_scale)
    cfg.rgb_rotation_weight_scale = float(args.rgb_rotation_weight_scale)
    cfg.rgb_translation_weight_scale = float(args.rgb_translation_weight_scale)
    cfg.rgb_reprojection_weight_scale = float(args.rgb_reprojection_weight_scale)
    cfg.sigma_rgb_reprojection_px = float(args.sigma_rgb_reprojection_px)
    cfg.max_rgb_reprojection_residual_px = float(args.max_rgb_reprojection_residual_px)
    cfg.max_rgb_reprojection_points_per_edge = int(args.max_rgb_reprojection_points_per_edge)
    return cfg


def _pose_map(nodes: Sequence[PoseNode]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {node.frame_idx: (node.initial_rotation.copy(), node.initial_translation.copy()) for node in nodes}


def _update_rows(
    timeline_report: Mapping[str, Any],
    nodes: Sequence[PoseNode],
    final_pose: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frame_start: int,
    frame_end: int,
    solver_report: Mapping[str, Any],
    mesh_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    by_frame = {node.frame_idx: node for node in nodes}
    output = dict(timeline_report)
    source_rows = timeline_report.get("pose_rows", [])
    rows: list[Any] = []
    seen: set[int] = set()
    for original in source_rows if isinstance(source_rows, list) else []:
        if not isinstance(original, dict) or original.get("frame_idx") is None:
            rows.append(original); continue
        idx = int(original["frame_idx"]); seen.add(idx)
        if not (int(frame_start) <= idx <= int(frame_end)) or idx not in final_pose:
            rows.append(original)
            continue
        row = dict(original)
        R, t = final_pose[idx]
        node = by_frame[idx]
        row["rotation_world_from_completed_canonical_matrix"] = R.tolist()
        row["translation_world_m"] = t.tolist()
        row["pose_source"] = "v20_late_window_prediction_only_full_timeline_se3"
        row["v20_late_window_uncertainty"] = {
            "metric_observation": bool(node.metric_observation),
            "observation_source": node.observation_source,
            "uncertainty": float(node.uncertainty),
            "latent_node": not bool(node.metric_observation),
            "frame_143_metric_missing": idx == 143 and not node.metric_observation,
        }
        row["generated_geometry_pose_evidence_consumed"] = False
        row["generated_geometry_validation_only"] = bool(mesh_contract is not None)
        rows.append(row)
    # A full annotation timeline can contain rows absent from a sparse base
    # report.  Add them as explicit latent rows, while never touching existing
    # rows outside the requested window.
    for node in sorted(nodes, key=lambda value: value.frame_idx):
        if node.frame_idx in seen or not (int(frame_start) <= node.frame_idx <= int(frame_end)):
            continue
        R, t = final_pose[node.frame_idx]
        rows.append({
            "frame_idx": node.frame_idx,
            "rotation_world_from_completed_canonical_matrix": R.tolist(),
            "translation_world_m": t.tolist(),
            "pose_source": "v20_late_window_prediction_only_full_timeline_se3",
            "v20_late_window_uncertainty": {"metric_observation": False, "latent_node": True, "uncertainty": float(node.uncertainty)},
            "generated_geometry_pose_evidence_consumed": False,
        })
    output["pose_rows"] = rows
    output["schema"] = "v20_late_window_prediction_only_full_timeline_se3_v1"
    output["annotation_ready"] = False; output["diagnostic_only"] = True; output["gt_consumed"] = False; output["formal_state_modified"] = False
    output["claim_scope"] = "Prediction-only late-window full-6DoF SE(3) graph; generated mesh is validation/render evidence only."
    output["v20_late_window"] = {
        "frame_start": int(frame_start), "frame_end": int(frame_end),
        "all_timeline_rows_used": True, "window_node_count": len(nodes),
        "anchor_only_gauge_fixed": True, "small_total_rotation_cap_applied": False,
        "frame_143_latent_uncertain_node": any(n.frame_idx == 143 and not n.metric_observation for n in nodes),
        "rows_outside_window_preserved": True,
    }
    output["solver"] = dict(solver_report)
    output["mesh_validation_contract"] = dict(mesh_contract) if mesh_contract is not None else None
    return output


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True, help="immutable prediction-only initial pose report")
    parser.add_argument("--full-timeline-pose-report", type=Path, default=None, help="current prediction-only full-timeline base; defaults to --pose-report")
    parser.add_argument("--current-pose-report", dest="full_timeline_pose_report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--rgb-edge-npz", type=Path, default=None)
    parser.add_argument("--rgb-edge-json", type=Path, default=None)
    parser.add_argument("--material-rgbd-npz", type=Path, default=None)
    parser.add_argument("--observed-factor-mode", choices=["legacy_nn", "material_tracks"], default="legacy_nn")
    parser.add_argument("--material-image-weight", type=float, default=0.25)
    parser.add_argument("--material-metric-weight", type=float, default=1.0)
    parser.add_argument("--sigma-material-px", type=float, default=2.0)
    parser.add_argument("--sigma-material-m", type=float, default=0.008)
    parser.add_argument("--max-material-pixel-degradation-px", type=float, default=3.0)
    parser.add_argument("--output-report", "--output-pose-report", dest="output_report", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=115); parser.add_argument("--frame-end", type=int, default=149); parser.add_argument("--anchor-frame", type=int, default=None)
    parser.add_argument("--validation-mesh", "--generated-visible-mesh", dest="validation_mesh", type=Path, default=None)
    parser.add_argument("--outer-iterations", type=int, default=3); parser.add_argument("--inner-iterations", type=int, default=40); parser.add_argument("--max-points", type=int, default=350); parser.add_argument("--min-pairs", type=int, default=12); parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--sigma-pose-rotation-rad", type=float, default=0.35); parser.add_argument("--sigma-pose-translation-m", type=float, default=0.06); parser.add_argument("--sigma-rgb-rotation-rad", type=float, default=0.08); parser.add_argument("--sigma-rgb-translation-m", type=float, default=0.02); parser.add_argument("--sigma-point-plane-m", type=float, default=0.008); parser.add_argument("--sigma-point-point-m", type=float, default=0.02)
    parser.add_argument("--sigma-velocity-rotation-rad", type=float, default=0.30); parser.add_argument("--sigma-velocity-translation-m", type=float, default=0.05); parser.add_argument("--sigma-acceleration-rotation-rad", type=float, default=0.25); parser.add_argument("--sigma-acceleration-translation-m", type=float, default=0.04); parser.add_argument("--sigma-correction-velocity-rotation-rad", type=float, default=0.40); parser.add_argument("--sigma-correction-velocity-translation-m", type=float, default=0.08)
    parser.add_argument("--max-point-residual-m", type=float, default=0.08); parser.add_argument("--update-rotation-trust-rad", type=float, default=0.45); parser.add_argument("--update-translation-trust-m", type=float, default=0.05); parser.add_argument("--max-base-cycle-rotation-deg", type=float, default=45.0); parser.add_argument("--max-base-cycle-translation-m", type=float, default=0.20); parser.add_argument("--min-rgb-edge-weight", type=float, default=0.02); parser.add_argument("--min-rgb-translation-edge-weight", type=float, default=0.02, help="Minimum effective weight for low-conditioning translation-only RGB edges."); parser.add_argument("--source-conditioning-full-weight-scale", type=float, default=0.02); parser.add_argument("--rgb-reprojection-weight-scale", type=float, default=0.0, help="Optional weight for serialized source-canonical to target-UV RGB factors."); parser.add_argument("--sigma-rgb-reprojection-px", type=float, default=2.0); parser.add_argument("--max-rgb-reprojection-residual-px", type=float, default=40.0); parser.add_argument("--max-rgb-reprojection-points-per-edge", type=int, default=96); parser.add_argument("--min-rgb-rotation-conditioning", type=float, default=0.01, help="Disable RGB rotation residuals below this raw 3D conditioning ratio; translation evidence remains available."); parser.add_argument("--observed-factor-weight-scale", type=float, default=1.0); parser.add_argument("--observed-factor-rotation-scale", type=float, default=1.0); parser.add_argument("--rgb-rotation-weight-scale", type=float, default=1.0); parser.add_argument("--rgb-translation-weight-scale", type=float, default=1.0); parser.add_argument("--max-metric-degradation-m", type=float, default=0.010); parser.add_argument("--max-continuity-rotation-deg", type=float, default=30.0); parser.add_argument("--max-continuity-translation-m", type=float, default=0.15); parser.add_argument("--max-acceleration-rotation-deg", type=float, default=45.0); parser.add_argument("--max-acceleration-translation-m", type=float, default=0.20)
    parser.add_argument("--generated-max-abs-depth-gate-m", type=float, default=0.05)
    parser.add_argument("--generated-max-front-bias-m", type=float, default=None, help="Optional absolute front-bias limit; baseline comparison remains active.")
    parser.add_argument("--generated-max-front-fraction", type=float, default=None, help="Optional absolute front-fraction limit; baseline comparison remains active.")
    parser.add_argument("--generated-max-front-bias-increase-m", type=float, default=0.005)
    parser.add_argument("--generated-negative-front-threshold-m", type=float, default=0.005, help="Per-frame signed median below this is classified as a front-bias segment.")
    parser.add_argument("--generated-max-per-frame-front-bias-m", type=float, default=None, help="Optional visible-pose/render gate for every frame's front bias.")
    parser.add_argument("--generated-max-negative-front-segment-length", type=int, default=None, help="Optional maximum contiguous segment length classified as front-biased.")
    parser.add_argument("--generated-max-common-hit-depth-increase-m", type=float, default=0.010)
    parser.add_argument("--generated-max-coverage-decrease", type=float, default=0.05)
    parser.add_argument("--generated-max-silhouette-decrease", type=float, default=0.05)
    parser.add_argument("--generated-factor-module", type=Path, default=Path(__file__).with_name("v20_generated_first_hit_factors.py"))
    parser.add_argument("--generated-factor-builder-script", type=Path, default=None, help="audited first-hit builder; enables generated validation only")
    parser.add_argument("--generated-visible-hand-npz", type=Path, default=None)
    parser.add_argument("--generated-visible-mano-faces-pkl", type=Path, default=None)
    parser.add_argument("--generated-visible-raster-size", type=int, default=256)
    parser.add_argument("--generated-visible-source-size", type=int, default=1408)
    parser.add_argument("--generated-visible-min-frames", type=int, default=1)
    parser.add_argument("--generated-visible-min-observed-points", type=int, default=10)
    parser.add_argument("--generated-visible-min-ownership-fraction", type=float, default=0.05)
    parser.add_argument("--generated-visible-hand-unknown-dilation-px", type=int, default=3)
    parser.add_argument("--generated-visible-max-removed-fraction-for-full-weight", type=float, default=0.10)
    parser.add_argument("--generated-visible-min-depth-factor-weight", type=float, default=0.05)
    parser.add_argument("--generated-visible-boundary-downweight-radius-px", type=float, default=2.0)
    parser.add_argument("--generated-visible-boundary-weight", type=float, default=0.25)
    parser.add_argument("--generated-visible-max-observed-factors-per-frame", type=int, default=1000)
    parser.add_argument("--generated-visible-max-outside-factors-per-frame", type=int, default=1000)
    parser.add_argument("--generated-visible-max-missing-factors-per-frame", type=int, default=1000)
    parser.add_argument("--generated-visible-render-batch-size", type=int, default=4)
    parser.add_argument("--generated-visible-device", default="cuda")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.frame_end) < int(args.frame_start):
        raise RuntimeError("frame-end must be >= frame-start")
    annotations = load_json(args.annotations)
    assert_prediction_only(annotations, label="late-window annotations", object_id=args.object_id)
    immutable = load_json(args.pose_report)
    timeline = load_json(args.full_timeline_pose_report) if args.full_timeline_pose_report is not None else immutable
    assert_prediction_only(immutable, label="late-window immutable pose report", object_id=args.object_id)
    assert_prediction_only(timeline, label="late-window timeline pose report", object_id=args.object_id)
    require_equal_object_ids(("immutable pose report", immutable), ("timeline pose report", timeline))
    cfg = _build_config(args)
    nodes, immutable_direct, _timeline_direct, _frames = build_pose_nodes(annotations, timeline, immutable, args.object_id, int(args.frame_start), int(args.frame_end))
    window_nodes = [node for node in nodes if int(args.frame_start) <= node.frame_idx <= int(args.frame_end)]
    if not window_nodes:
        raise RuntimeError("requested window contains no timeline rows")
    anchor = int(args.anchor_frame) if args.anchor_frame is not None else int(timeline.get("anchor_frame_idx", immutable.get("anchor_frame_idx", window_nodes[0].frame_idx)))
    if anchor not in {node.frame_idx for node in window_nodes}:
        raise RuntimeError(f"anchor frame {anchor} is not in requested window")
    base_pose = _pose_map(window_nodes)
    if args.rgb_edge_json is not None and args.rgb_edge_json.exists():
        edge_report = load_json(args.rgb_edge_json)
        assert_prediction_only(edge_report, label="late-window RGB edge report", object_id=args.object_id)
    source_support = {
        node.frame_idx: np.mean(node.observed_world, axis=0)
        for node in window_nodes
        if node.metric_observation and node.observed_world is not None
    }
    rgb_edges, edge_diagnostics = load_rgb_edges(args.rgb_edge_npz, json_path=args.rgb_edge_json, pose_by_frame=base_pose, frame_ids={node.frame_idx for node in window_nodes}, config=cfg, source_support_by_frame=source_support)
    reprojection_factors, reprojection_diagnostics = load_rgb_reprojection_factors(
        args.rgb_edge_npz,
        edges=rgb_edges,
        nodes=window_nodes,
        config=cfg,
    )
    if cfg.rgb_reprojection_weight_scale > 0 and not reprojection_factors:
        raise RuntimeError(f"requested RGB pixel objective unavailable: {reprojection_diagnostics}")
    if reprojection_factors:
        meta = reprojection_diagnostics["input_metadata"]
        if meta.get("object_id") != args.object_id or (meta.get("input_sha256") or {}).get("annotations") != sha256_file(args.annotations):
            raise RuntimeError("RGB pixel evidence annotation/object contract mismatch")
    material_factors = []
    material_diagnostics = None
    if args.material_rgbd_npz is not None:
        if cfg.rgb_reprojection_weight_scale > 0 or cfg.observed_factor_mode != "material_tracks":
            raise ValueError("material mode replaces NN and legacy pixel factors; do not double-count")
        material_factors, material_diagnostics = material.load_factors(
            args.material_rgbd_npz, window_nodes, annotation_sha256=sha256_file(args.annotations),
            object_id=args.object_id, max_points=cfg.max_rgb_reprojection_points_per_edge)
    elif cfg.observed_factor_mode == "material_tracks":
        raise ValueError("--material-rgbd-npz is required for material mode")
    raw_pairs = {(f.source_frame_idx, f.target_frame_idx) for f in [*reprojection_factors, *material_factors]}
    duplicate_pnp_pairs = [ [edge.source_frame_idx,edge.target_frame_idx] for edge in rgb_edges
                           if (edge.source_frame_idx,edge.target_frame_idx) in raw_pairs ]
    # These are the same RGB matches, not independent measurements. Use their
    # raw observation factor OR their compressed PnP edge, never both.
    rgb_edges = [edge for edge in rgb_edges if (edge.source_frame_idx,edge.target_frame_idx) not in raw_pairs]
    immutable_pose = {idx: pose for idx, pose in immutable_direct.items() if int(args.frame_start) <= idx <= int(args.frame_end)}
    initial_metrics = pose_metrics(window_nodes, base_pose, anchor)
    if immutable_pose:
        # Immutable metrics use immutable poses where available and base pose
        # only for latent rows, never HOT3D/GT.
        metric_pose = dict(base_pose); metric_pose.update(immutable_pose)
        initial_metrics = pose_metrics(window_nodes, metric_pose, anchor)
    if material_factors:
        metric_pose = dict(base_pose); metric_pose.update(immutable_pose)
        initial_metrics.update(material.metrics(material_factors, metric_pose))
    mesh_contract = generated_mesh_contract(args.validation_mesh) if args.validation_mesh is not None else None
    validation_context = load_generated_validation_context(args) if args.validation_mesh is not None else None
    generated_callback = None
    generated_current_baseline = None
    generated_immutable_baseline = None
    if validation_context is not None:
        generated_callback, generated_current_baseline, generated_immutable_baseline = make_generated_validation_callback(validation_context, window_nodes, immutable_pose, args)
    solver_x, solver_report = solve_window(
        window_nodes, rgb_edges, anchor, cfg, initial_metrics, generated_callback,
        reprojection_factors, material_factors,
    )
    final_pose = current_poses(window_nodes, solver_x)
    solver_report = dict(solver_report)
    solver_report["generated_validation_context_available"] = bool(validation_context is not None)
    solver_report["generated_validation_limitation"] = None if validation_context is not None or args.validation_mesh is None else "mesh contract recorded, but audited first-hit validation needs --generated-factor-builder-script, --generated-visible-hand-npz, and --generated-visible-mano-faces-pkl"
    if generated_current_baseline is not None:
        solver_report["generated_current_baseline"] = generated_current_baseline
        solver_report["generated_immutable_initial_baseline"] = generated_immutable_baseline
    solver_report["signed_front_bias_definition"] = "negative generated_depth_minus_observed_depth means generated is in front"
    solver_report["signed_front_gate_configured"] = bool(
        args.generated_max_per_frame_front_bias_m is not None
        or args.generated_max_negative_front_segment_length is not None
    )
    solver_report["observed_factor_mode"] = cfg.observed_factor_mode
    solver_report["material_rgbd_diagnostics"] = material_diagnostics
    solver_report["material_rgbd_parameters"] = {
        "image_weight": cfg.material_image_weight, "metric_weight": cfg.material_metric_weight,
        "sigma_px": cfg.sigma_material_px, "sigma_m": cfg.sigma_material_m}
    solver_report["compressed_pnp_pairs_replaced_by_raw_observations"] = duplicate_pnp_pairs
    solver_report["rgb_edge_diagnostics"] = edge_diagnostics
    solver_report["rgb_reprojection_diagnostics"] = reprojection_diagnostics
    solver_report["rgb_edge_count_accepted"] = len(rgb_edges)
    solver_report["conditioning_aware_weights"] = True
    solver_report["min_rgb_rotation_conditioning"] = float(cfg.min_rgb_rotation_conditioning)
    solver_report["min_rgb_translation_edge_weight"] = float(cfg.min_rgb_translation_edge_weight)
    solver_report["rgb_rotation_eligible_edge_count"] = sum(bool(edge.diagnostics.get("rotation_eligible", True)) for edge in rgb_edges)
    solver_report["rgb_translation_only_edge_count"] = sum(bool(edge.diagnostics.get("translation_only", False)) for edge in rgb_edges)
    solver_report["rgb_rotation_observability"] = rgb_rotation_observability(rgb_edges, window_nodes)
    solver_report["base_trajectory_cycle_gating"] = True
    solver_report["pose_objective_terms"] = ["accepted_rgb_world_relative_pnp_rotation", "accepted_rgb_world_relative_pnp_translation", "rebuilt_P09_point_to_plane", "rebuilt_P09_point_to_point", "absolute_pose_velocity", "absolute_pose_acceleration", "correction_priors", "anchor_only_gauge"]
    if cfg.observed_factor_mode == "material_tracks":
        solver_report["pose_objective_terms"] = [term for term in solver_report["pose_objective_terms"] if not term.startswith("rebuilt_P09")]
        solver_report["pose_objective_terms"].append("fixed_material_world_observations_XYZ_and_target_UV")
    if not rgb_edges:
        solver_report["pose_objective_terms"] = [term for term in solver_report["pose_objective_terms"] if not term.startswith("accepted_rgb_world")]
    if float(cfg.rgb_reprojection_weight_scale) > 0.0 and reprojection_factors:
        solver_report["pose_objective_terms"].append("fixed_source_world_via_source_inverse_and_target_pose_reprojection")
    solver_report["generated_factors_in_pose_objective"] = False
    if generated_callback is not None and hasattr(generated_callback, "current_diagnostics"):
        final_generated_validation = generated_callback.current_diagnostics()
        final_front_pass, final_front_reasons = signed_front_bias_gate(
            final_generated_validation,
            max_front_bias_m=(float(args.generated_max_front_bias_m) if args.generated_max_front_bias_m is not None else None),
            max_front_fraction=(float(args.generated_max_front_fraction) if args.generated_max_front_fraction is not None else None),
            max_increase_m=float(args.generated_max_front_bias_increase_m),
            max_per_frame_front_bias_m=(float(args.generated_max_per_frame_front_bias_m) if args.generated_max_per_frame_front_bias_m is not None else None),
            max_negative_front_segment_length=(int(args.generated_max_negative_front_segment_length) if args.generated_max_negative_front_segment_length is not None else None),
        )
        final_generated_validation["signed_front_bias_gate_pass"] = bool(final_front_pass)
        final_generated_validation["signed_front_bias_gate_reasons"] = final_front_reasons
        solver_report["final_generated_validation"] = final_generated_validation
        solver_report["final_signed_front_gate_pass"] = bool(final_front_pass)
        solver_report["final_signed_front_gate_reasons"] = final_front_reasons
    else:
        solver_report["final_generated_validation"] = None
        solver_report["final_signed_front_gate_pass"] = None
        solver_report["final_signed_front_gate_reasons"] = []
    solver_report["observed_factor_weight_scale"] = float(cfg.observed_factor_weight_scale)
    solver_report["observed_factor_rotation_scale"] = float(cfg.observed_factor_rotation_scale)
    solver_report["rgb_rotation_weight_scale"] = float(cfg.rgb_rotation_weight_scale)
    solver_report["rgb_translation_weight_scale"] = float(cfg.rgb_translation_weight_scale)
    solver_report["rgb_reprojection_weight_scale"] = float(cfg.rgb_reprojection_weight_scale)
    solver_report["sigma_rgb_reprojection_px"] = float(cfg.sigma_rgb_reprojection_px)
    solver_report["max_rgb_reprojection_points_per_edge"] = int(cfg.max_rgb_reprojection_points_per_edge)
    solver_report["clipping_and_continuity"] = candidate_correction_diagnostics(
        window_nodes,
        solver_x,
        rotation_clipped=bool(solver_report.get("rotation_update_trust_clipped", False)),
        translation_clipped=bool(solver_report.get("translation_update_trust_clipped", False)),
    )
    solver_report["uncertainty"] = {str(node.frame_idx): {"metric_observation": node.metric_observation, "source": node.observation_source, "uncertainty": node.uncertainty} for node in window_nodes}
    solver_report["input_sha256"] = {"annotations": sha256_file(args.annotations), "immutable_pose_report": sha256_file(args.pose_report), "timeline_pose_report": sha256_file(args.full_timeline_pose_report) if args.full_timeline_pose_report is not None else sha256_file(args.pose_report), "rgb_edge_npz": sha256_file(args.rgb_edge_npz) if args.rgb_edge_npz is not None else None, "rgb_edge_json": sha256_file(args.rgb_edge_json) if args.rgb_edge_json is not None and args.rgb_edge_json.exists() else None}
    if args.material_rgbd_npz is not None:
        solver_report["input_sha256"]["material_rgbd_npz"] = sha256_file(args.material_rgbd_npz)
    result = _update_rows(timeline, window_nodes, final_pose, int(args.frame_start), int(args.frame_end), solver_report, mesh_contract)
    result["object_id"] = args.object_id
    accepted_count = sum(
        bool(item.get("accepted_update"))
        for item in solver_report.get("outer_iterations", [])
    )
    strict_signed_front_configured = bool(
        args.generated_max_per_frame_front_bias_m is not None
        or args.generated_max_negative_front_segment_length is not None
    )
    if (
        validation_context is not None
        and strict_signed_front_configured
        and solver_report.get("final_signed_front_gate_pass") is False
    ):
        # This is a visible-pose/render evidence failure, not a physical
        # geometry claim. Keep the pose for diagnosis but do not label it
        # complete when a sustained per-frame front bias remains.
        result["status"] = "v20_late_window_se3_incomplete_signed_front_validation"
    elif accepted_count <= 0:
        result["status"] = "v20_late_window_se3_no_safe_update"
    elif validation_context is not None and solver_report.get("final_generated_validation", {}).get("validation_available") is False:
        result["status"] = "v20_late_window_se3_incomplete_validation_unavailable"
    elif validation_context is not None and not strict_signed_front_configured:
        result["status"] = "v20_late_window_se3_complete_signed_front_unchecked"
    else:
        result["status"] = "v20_late_window_se3_complete"
    result["prediction_provenance"] = {
        "gt_consumed": False,
        "generated_mesh_pose_authority": False,
        "generated_mesh_validation_only": bool(mesh_contract),
        "immutable_initial_report": str(args.pose_report.expanduser().resolve()),
        "current_base_report": (
            str(args.full_timeline_pose_report.expanduser().resolve())
            if args.full_timeline_pose_report is not None
            else str(args.pose_report.expanduser().resolve())
        ),
        "stage": "prediction_only_late_window_full_se3",
    }
    append_prediction_stage(
        result,
        stage="prediction_only_late_window_full_se3",
        input_hashes={
            "annotations": sha256_file(args.annotations),
            "immutable_pose_report": sha256_file(args.pose_report),
            "timeline_pose_report": (
                sha256_file(args.full_timeline_pose_report)
                if args.full_timeline_pose_report is not None
                else sha256_file(args.pose_report)
            ),
            "rgb_edge_npz": (
                sha256_file(args.rgb_edge_npz)
                if args.rgb_edge_npz is not None
                else None
            ),
            "rgb_edge_json": (
                sha256_file(args.rgb_edge_json)
                if args.rgb_edge_json is not None and args.rgb_edge_json.exists()
                else None
            ),
        },
        notes={
            "generated_factors_in_pose_objective": False,
            "frame_143_explicit_latent_node": bool(
                result.get("v20_late_window", {}).get(
                    "frame_143_latent_uncertain_node"
                )
            ),
            "anchor_only_gauge": True,
        },
    )
    args.output_report.expanduser().resolve().parent.mkdir(
        parents=True, exist_ok=True
    )
    args.output_report.expanduser().resolve().write_text(
        json.dumps(
            json_safe(result), indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_report": str(args.output_report.resolve()),
                "window_node_count": len(window_nodes),
                "accepted_rgb_edges": len(rgb_edges),
                "generated_factors_in_pose_objective": False,
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
