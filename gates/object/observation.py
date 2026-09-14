"""Owned metric surface eligibility; camera/world disagreement is explicit."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from utils.geometry import _array, valid_rotation, camera_to_world

def _frame_camera_transform(frame: Mapping[str, Any]) -> np.ndarray | None:
    camera = frame.get('camera')
    if not isinstance(camera, Mapping):
        return None
    for key in ('T_world_camera_metric', 'T_world_camera'):
        value = camera.get(key)
        try:
            T = _array(value, (4, 4))
        except (TypeError, ValueError):
            continue
        if valid_rotation(T[:3, :3]) and np.linalg.norm(T[3] - np.asarray([0.0, 0.0, 0.0, 1.0])) < 1e-05:
            return T
    return None

def _finite_point_array(value: Any) -> np.ndarray | None:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or (not np.isfinite(points).all()):
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
    for (container_name, container) in (('geometry', geometry), ('object', obj)):
        for key in ('object_surface_ownership_filter', 'first_surface_depth_ownership', 'ownership_contract'):
            value = container.get(key)
            if isinstance(value, Mapping):
                ownership_values.append((f'{container_name}.{key}', value))
    if not ownership_values:
        return (True, {'status': 'ownership_metadata_unavailable', 'verified': False, 'reasons': ['missing_explicit_P09_ownership_summary']})
    reasons: list[str] = []
    for (name, value) in ownership_values:
        if value.get('fail_closed') is True:
            reasons.append(f'{name}.fail_closed')
        failure_reasons = value.get('failure_reasons')
        if isinstance(failure_reasons, Sequence) and (not isinstance(failure_reasons, (str, bytes, bytearray))) and len(failure_reasons):
            reasons.append(f'{name}.failure_reasons')
        if 'enabled' in value and value.get('enabled') is not True:
            reasons.append(f'{name}.not_enabled')
        if 'retained_fraction' in value:
            try:
                retained = float(value.get('retained_fraction'))
            except (TypeError, ValueError):
                retained = float('nan')
            if not np.isfinite(retained) or retained <= 0.0:
                reasons.append(f'{name}.invalid_retained_fraction')
    return (not reasons, {'status': 'verified' if not reasons else 'failed_closed', 'verified': not reasons, 'reasons': reasons, 'summary_count': len(ownership_values)})

def p09_geometry_validation(obj: Mapping[str, Any] | None, frame: Mapping[str, Any], *, camera_world_tolerance_m: float=0.0001) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    """Validate P09 ownership and camera/world consistency before use.

    P09 may provide either world samples or camera samples.  When both are
    present they must describe the same points under the frame's metric
    ``T_world_camera``; silently preferring one would hide a frame-convention
    error.  The function returns an explicit diagnostic for every outcome.
    """
    if obj is None:
        return (None, 'missing_object', {'status': 'missing_object', 'verified': False})
    geometry = obj.get('visible_geometry_candidate')
    if not isinstance(geometry, Mapping):
        return (None, 'missing_p09_visible_geometry', {'status': 'missing_geometry', 'verified': False})
    eligibility_values = [('object.rigid_pose_observation_eligible', obj.get('rigid_pose_observation_eligible')), ('geometry.rigid_pose_observation_eligible', geometry.get('rigid_pose_observation_eligible'))]
    ineligible = [name for (name, value) in eligibility_values if value is False]
    if ineligible:
        return (None, 'P09_rigid_pose_observation_ineligible', {'status': 'rigid_pose_ineligible', 'verified': False, 'ineligible_fields': ineligible})
    (ownership_ok, ownership_diag) = _p09_ownership_validation(obj, geometry)
    if not ownership_ok:
        return (None, 'P09_ownership_failed_closed', {'status': 'ownership_failed_closed', 'verified': False, 'ownership': ownership_diag})
    world_present = 'world_vertices_sample_m' in geometry and geometry.get('world_vertices_sample_m') is not None
    camera_present = 'camera_vertices_sample_m' in geometry and geometry.get('camera_vertices_sample_m') is not None
    world = _finite_point_array(geometry.get('world_vertices_sample_m')) if world_present else None
    camera = _finite_point_array(geometry.get('camera_vertices_sample_m')) if camera_present else None
    if world_present and world is None:
        return (None, 'invalid_P09_world_surface', {'status': 'invalid_world_points', 'verified': False, 'ownership': ownership_diag})
    if camera_present and camera is None:
        return (None, 'invalid_P09_camera_surface', {'status': 'invalid_camera_points', 'verified': False, 'ownership': ownership_diag})
    if camera is not None and np.any(camera[:, 2] <= 0.0):
        return (None, 'invalid_P09_camera_depth', {'status': 'nonpositive_camera_depth', 'verified': False, 'ownership': ownership_diag})
    T = _frame_camera_transform(frame)
    if camera is not None and T is None:
        return (None, 'missing_or_invalid_P09_camera_transform', {'status': 'invalid_camera_transform', 'verified': False, 'ownership': ownership_diag})
    transform_error_m: float | None = None
    if world is not None and camera is not None:
        if len(world) != len(camera):
            return (None, 'P09_camera_world_count_mismatch', {'status': 'camera_world_count_mismatch', 'verified': False, 'ownership': ownership_diag})
        assert T is not None
        lifted = camera_to_world(camera, T)
        transform_error_m = float(np.max(np.linalg.norm(lifted - world, axis=1)))
        if not np.isfinite(transform_error_m) or transform_error_m > float(camera_world_tolerance_m):
            return (None, 'P09_camera_world_transform_mismatch', {'status': 'camera_world_mismatch', 'verified': False, 'max_camera_world_error_m': transform_error_m, 'camera_world_tolerance_m': float(camera_world_tolerance_m), 'ownership': ownership_diag})
    points = world if world is not None else camera_to_world(camera, T) if camera is not None and T is not None else None
    if points is None:
        return (None, 'missing_or_invalid_P09_surface', {'status': 'missing_surface', 'verified': False, 'ownership': ownership_diag})
    source = 'P09.world_vertices_sample_m' if world is not None else 'P09.camera_vertices_sample_m->world'
    return (points, source, {'status': 'accepted', 'verified': bool(world is not None and camera is not None and (transform_error_m is not None)), 'ownership': ownership_diag, 'camera_world_consistency': {'checked': bool(world is not None and camera is not None), 'max_error_m': transform_error_m, 'tolerance_m': float(camera_world_tolerance_m)}, 'source': source, 'point_count': int(len(points))})

def rigid_pose_observation_eligibility(obj: dict[str, Any], geom: dict[str, Any]) -> tuple[bool | None, list[str], list[str]]:
    """Resolve explicit P09 eligibility without rejecting legacy rows lacking the field."""
    from typing import Any
    values: list[bool] = []
    reasons: list[str] = []
    sources: list[str] = []
    for (source, payload) in (('object', obj), ('visible_geometry_candidate', geom)):
        value = payload.get('rigid_pose_observation_eligible')
        if isinstance(value, bool):
            values.append(value)
            sources.append(source)
        reason = payload.get('rigid_pose_observation_reason')
        if isinstance(reason, str) and reason and (reason not in reasons):
            reasons.append(reason)
    if False in values:
        return (False, reasons, sources)
    if True in values:
        return (True, reasons, sources)
    return (None, reasons, sources)

def conservative_mask_on_depth(mask: np.ndarray, A_depth_from_mask: np.ndarray, depth_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV-nearest mask samples and an explicit coordinate-valid guard.

    OpenCV can assign a border source sample to a destination center whose
    inverse-mapped continuous coordinate lies outside the declared source
    raster.  Such subpixel extrapolation remains appearance support but cannot
    become a metric candidate.
    """
    import cv2
    import numpy as np
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(A_depth_from_mask, dtype=np.float64)
    if mask.ndim != 2 or len(depth_shape) != 2 or min(depth_shape) <= 0 or (affine.shape != (3, 3)) or (not np.isfinite(affine).all()) or (abs(float(np.linalg.det(affine))) < 1e-12):
        raise ValueError('invalid mask/depth affine contract')
    sampled = cv2.resize(mask.astype(np.uint8), (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT) > 0 if mask.shape != depth_shape else mask.copy()
    inverse = np.linalg.inv(affine)
    (yy, xx) = np.mgrid[:depth_shape[0], :depth_shape[1]]
    points = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ inverse.T
    finite = np.isfinite(points).all(axis=2) & (np.abs(points[..., 2]) > 1e-12)
    mx = np.divide(points[..., 0], points[..., 2], out=np.zeros_like(points[..., 0], dtype=float), where=finite)
    my = np.divide(points[..., 1], points[..., 2], out=np.zeros_like(points[..., 1], dtype=float), where=finite)
    coordinate_valid = finite & (mx >= 0.0) & (mx <= mask.shape[1] - 1.0) & (my >= 0.0) & (my <= mask.shape[0] - 1.0)
    return (sampled, coordinate_valid)

def classify_native_depth(*, owned_mask: np.ndarray, hand_z: np.ndarray, depth_z: np.ndarray, A_depth_from_mask: np.ndarray, complete_hand_geometry: bool, validated_hand_absence: bool, tolerance_m: float, uncertainty_padding_mask_px: int, raw_mask: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Classify at the exact depth raster and return robust-filter candidates.

    Clear no-overlap object body remains a candidate; no-hit pixels within the
    conservative projected-hand band do not.  Missing declared hand geometry
    makes the frame ineligible while preserving appearance elsewhere.
    """
    import cv2
    from observations.object.surface import CLASS_CODES
    import numpy as np
    depth_z = np.asarray(depth_z, dtype=np.float64)
    hand_z = np.asarray(hand_z, dtype=np.float64)
    if depth_z.ndim != 2 or hand_z.shape != depth_z.shape or (not np.isfinite(tolerance_m)) or (tolerance_m < 0) or (type(uncertainty_padding_mask_px) is not int) or (uncertainty_padding_mask_px < 0):
        raise ValueError('invalid native depth-order inputs')
    (owned_depth, coordinate_valid) = conservative_mask_on_depth(owned_mask, A_depth_from_mask, depth_z.shape)
    (raw_depth, raw_coordinate_valid) = conservative_mask_on_depth(owned_mask if raw_mask is None else raw_mask, A_depth_from_mask, depth_z.shape)
    if not np.array_equal(coordinate_valid, raw_coordinate_valid):
        raise ValueError('raw/owned mask coordinate guards disagree')
    codes = np.full(depth_z.shape, CLASS_CODES['outside_owned_appearance'], dtype=np.uint8)
    support = np.isfinite(depth_z) & (depth_z > 0)
    active = raw_depth & coordinate_valid
    codes[raw_depth & ~coordinate_valid] = CLASS_CODES['invalid_mask_coordinate']
    codes[active & ~support] = CLASS_CODES['unsupported_depth']
    if not complete_hand_geometry and (not validated_hand_absence):
        codes[active & support] = CLASS_CODES['unresolved_hand_geometry']
        eligible = np.zeros(depth_z.shape, dtype=bool)
    else:
        hit = np.isfinite(hand_z) & (hand_z > 0)
        valid_hit = active & support & hit
        delta = np.zeros(depth_z.shape, dtype=np.float64)
        np.subtract(hand_z, depth_z, out=delta, where=valid_hit)
        front = valid_hit & (delta < -tolerance_m)
        behind = valid_hit & (delta > tolerance_m)
        near = valid_hit & ~front & ~behind
        codes[front] = CLASS_CODES['hand_clearly_front']
        codes[behind] = CLASS_CODES['hand_clearly_behind']
        codes[near] = CLASS_CODES['near_depth_ambiguous']
        no_hit = active & support & ~hit
        if validated_hand_absence:
            fringe = np.zeros(depth_z.shape, dtype=bool)
        else:
            linear = np.asarray(A_depth_from_mask, dtype=np.float64)[:2, :2]
            depth_pad = int(np.ceil(float(uncertainty_padding_mask_px) * max(np.linalg.norm(linear[:, 0]), np.linalg.norm(linear[:, 1]))))
            if depth_pad:
                kernel = np.ones((2 * depth_pad + 1, 2 * depth_pad + 1), np.uint8)
                fringe = cv2.dilate(hit.astype(np.uint8), kernel) > 0
            else:
                fringe = hit.copy()
        codes[no_hit & fringe] = CLASS_CODES['no_hit_uncertainty_fringe']
        codes[no_hit & ~fringe] = CLASS_CODES['clear_no_hand_overlap_candidate']
        eligible = (behind | no_hit & ~fringe) & owned_depth
    counts = {name: int(np.count_nonzero(codes == code)) for (name, code) in CLASS_CODES.items()}
    counts['metric_candidate_eligible'] = int(np.count_nonzero(eligible))
    counts['appearance_owned_at_depth'] = int(np.count_nonzero(owned_depth))
    counts['raw_appearance_at_depth'] = int(np.count_nonzero(raw_depth))
    return (codes, eligible, counts)
