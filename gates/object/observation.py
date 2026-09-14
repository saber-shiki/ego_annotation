"""Owned metric surface eligibility; camera/world disagreement is explicit."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from utils.geometry import _array, valid_rotation, camera_to_world

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
