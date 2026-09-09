"""Audited numerical kernels extracted from fit_v20_late_window_prediction_only_se3.

Left SO(3), additive world translation; observed points never generated geometry.
The experiment CLI, frame-specific node builder and candidate gates are excluded.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("array contains non-finite values")
    return result

def valid_rotation(value: Any) -> bool:
    try:
        matrix = _array(value, (3, 3))
    except (TypeError, ValueError):
        return False
    return bool(np.linalg.det(matrix) > 0.0 and np.linalg.norm(matrix.T @ matrix - np.eye(3)) < 1.0e-3)

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

def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    points = _array(points)
    if len(points) < 3:
        return np.tile(np.asarray([[0.0, 0.0, 1.0]]), (len(points), 1))
    kk = min(max(3, int(k)), len(points))
    _distances, indices = cKDTree(points).query(points, k=kk, workers=1)
    centered = points[indices] - points[indices].mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(1, kk - 1)
    _values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1.0e-12)

def _mutual_pairs(source: np.ndarray, target: np.ndarray, max_distance: float, max_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source) == 0 or len(target) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0)
    d, j = cKDTree(target).query(source, k=1, workers=1)
    reverse = cKDTree(source).query(target, k=1, workers=1)[1]
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
            rng = np.random.default_rng(int(cfg.seed) + positions[node.frame_idx] + 31 * int(outer_iteration))
            points = points[rng.choice(len(points), int(cfg.max_points), replace=False)]
        sampled_world[node.frame_idx] = points
        canonical[node.frame_idx] = inverse_pose(points, R, t)
        normals[node.frame_idx] = estimate_normals(canonical[node.frame_idx])
    pairs: list[tuple[int, int, str]] = []
    direct_ids = [node.frame_idx for node in nodes if node.frame_idx in canonical]
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
