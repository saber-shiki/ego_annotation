"""Extracted formal interval mechanisms; no historical script imports."""
from __future__ import annotations

def active_frame_intrinsics(frame: dict[str, Any], side: str, *, require_full_k: bool=False) -> list[float]:
    """Return source projection K, optionally requiring exact centered-plane provenance."""
    from typing import Any
    import numpy as np
    from utils.data import as_list
    matching = [hand for hand in as_list(frame.get('hands')) if isinstance(hand, dict) and str(hand.get('hand_side')) == str(side)]
    if len(matching) != 1:
        raise RuntimeError(f"frame {frame.get('frame_idx')} {side}: expected exactly one MANO row, got {len(matching)}")
    hand = matching[0]
    metric = hand.get('metric_mano_state') if isinstance(hand.get('metric_mano_state'), dict) else {}
    mano = finite_intrinsics(metric.get('current_v18_camera_intrinsics_fx_fy_cx_cy'), 'MANO inference intrinsics')
    if not require_full_k:
        return mano.astype(float).tolist()
    camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
    active = finite_intrinsics(camera.get('intrinsics_fx_fy_cx_cy'), 'active frame camera intrinsics')
    alignment = metric.get('camera_contract_alignment') if isinstance(metric.get('camera_contract_alignment'), dict) else hand.get('camera_contract_alignment') if isinstance(hand.get('camera_contract_alignment'), dict) else {}
    if require_full_k and alignment.get('backend') == 'wilor':
        from utils.hand_contract import independent_alignment_valid
        independent_alignment_valid(alignment, active)
        if not np.allclose(active, mano, atol=.01, rtol=0):
            raise RuntimeError('independent WiLoR hand K differs from active camera')
        return active.astype(float).tolist()
    if require_full_k:
        if alignment.get('active_contract_reinference_required') is not False:
            raise RuntimeError(f"frame {frame.get('frame_idx')} {side}: active camera/MANO reinference is still required; refusing P18 promotion")
        if alignment.get('source_hawor_state_intrinsics_match_active_contract') is not True:
            raise RuntimeError(f"frame {frame.get('frame_idx')} {side}: active source-K/centered-inference HaWoR binding is not explicitly proven")
        if alignment.get('source_hawor_full_K_bound_to_inference') is not True:
            raise RuntimeError(f"frame {frame.get('frame_idx')} {side}: HaWoR source-K to centered-inference-plane binding is not proven")
        plane = alignment.get('hawor_camera_image_plane_contract')
        if not isinstance(plane, dict) or plane.get('validated') is not True or plane.get('status') != 'validated_source_K_affine_centered_hawor_inference_plane':
            raise RuntimeError(f"frame {frame.get('frame_idx')} {side}: centered HaWoR image-plane binding is not validated")
    if require_full_k and (not np.allclose(active, mano, atol=0.01, rtol=0.0)):
        raise RuntimeError(f"frame {frame.get('frame_idx')} {side}: active camera K {active.tolist()} != MANO K {mano.tolist()}")
    return active.astype(float).tolist()

def finite_intrinsics(value: Any, label: str) -> np.ndarray:
    from typing import Any
    import numpy as np
    intrinsics = np.asarray(value, dtype=np.float64).reshape(-1)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0.0):
        raise RuntimeError(f'{label} must be finite [fx,fy,cx,cy], got {intrinsics}')
    return intrinsics

def translation_support_gate_applies(*, enabled: bool, support_count: int, minimum_supported_vertices: int) -> bool:
    """Whether a row lacks enough independent visible first-surface support.

    ``minimum_supported_vertices`` is the largest unsupported count. Thus the
    controlled threshold 0 gates exactly zero support, while one supported
    vertex is not silently treated as zero.
    """
    count = int(support_count)
    threshold = int(minimum_supported_vertices)
    if count < 0 or threshold < 0:
        raise ValueError('translation support counts/thresholds must be nonnegative')
    return bool(enabled and count <= threshold)


def hand_observation_visibility_for_row(row: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    from typing import Any
    import numpy as np
    import argparse
    if not isinstance(row, dict):
        return {'state': 'missing_hand_observation_visibility_row', 'candidate_px': 0, 'weight_multiplier': 1.0}
    state = str(row.get('state') or row.get('visibility_state') or 'active_hand_observation_visibility')
    candidate_px = int(row.get('candidate_px') or row.get('candidate_non_object_owned_px') or 0)
    raw_mult = row.get('joint_observation_weight_multiplier', row.get('weight_multiplier', args.occluded_joint_observation_weight))
    try:
        multiplier = float(raw_mult)
    except Exception:
        multiplier = float(args.occluded_joint_observation_weight)
    multiplier = float(np.clip(multiplier, 0.0, 1.0))
    return {'state': state, 'candidate_px': candidate_px, 'weight_multiplier': multiplier}

def joint_visibility_from_metric_depth(row_frame: dict[str, Any], side: str, joints_world: np.ndarray, depth_row: dict[str, Any] | None, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Return per-openpose-joint observation weights from first-surface depth.

    A joint close to the observed depth surface keeps full HaWoR anchoring.  A
    joint behind the observed first surface is treated as occluded by foreground
    geometry; a joint far in front of the observed depth is depth-inconsistent.
    Both cases reduce the zero-state observation force without removing temporal
    coherence or object nonpenetration.
    """
    from typing import Any
    import numpy as np
    import argparse
    from utils.geometry import world_to_camera
    from observations.interaction.evidence import project_world
    weights = np.ones((21,), dtype=float)
    residual = np.full((21,), np.nan, dtype=float)
    if depth_row is None:
        weights[:] = float(args.invalid_joint_observation_weight)
        return (weights, residual)
    uv = project_world(joints_world, row_frame, side)
    if uv is None:
        weights[:] = float(args.invalid_joint_observation_weight)
        return (weights, residual)
    depth = np.asarray(depth_row.get('depth'), dtype=np.float32)
    if depth.ndim != 2:
        weights[:] = float(args.invalid_joint_observation_weight)
        return (weights, residual)
    (height, width) = depth.shape
    cam = world_to_camera(joints_world, row_frame)
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    valid = (cam[:, 2] > 1e-05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_depth = np.zeros((21,), dtype=bool)
    valid_depth[valid] = np.isfinite(depth[v[valid], u[valid]]) & (depth[v[valid], u[valid]] > 1e-05)
    valid = valid & valid_depth
    weights[~valid] = float(args.invalid_joint_observation_weight)
    if np.any(valid):
        residual[valid] = cam[valid, 2] - depth[v[valid], u[valid]].astype(float)
        margin = float(args.visible_joint_depth_margin_m)
        behind = valid & (residual > margin)
        in_front = valid & (residual < -margin)
        weights[behind] = float(args.occluded_joint_observation_weight)
        weights[in_front] = float(args.front_inconsistent_joint_observation_weight)
    return (np.clip(weights, 0.0, 1.0), residual)

def validate_interval_camera_mano_contract(args: argparse.Namespace) -> dict[str, Any]:
    import argparse
    from typing import Any
    from utils.data import as_list
    from utils.data import load_json
    annotations = load_json(args.annotations)
    frames = {int(frame.get('frame_idx', pos)): frame for (pos, frame) in enumerate(as_list(annotations.get('frames'))) if isinstance(frame, dict)}
    checked = 0
    unique_intrinsics: set[tuple[float, float, float, float]] = set()
    missing: list[dict[str, Any]] = []
    for frame_idx in range(int(args.start_frame), int(args.end_frame) + 1):
        frame = frames.get(frame_idx)
        if frame is None:
            missing.append({'frame_idx': frame_idx, 'reason': 'missing_annotation_frame'})
            continue
        for side in args.sides:
            try:
                intr = active_frame_intrinsics(frame, side, require_full_k=bool(getattr(args, 'require_active_full_K_mano_contract', False)))
            except Exception as exc:
                raise RuntimeError(f'P18 camera/MANO preflight failed before model loading at frame={frame_idx} side={side}: {exc}') from exc
            unique_intrinsics.add(tuple((float(value) for value in intr)))
            checked += 1
    expected = (int(args.end_frame) - int(args.start_frame) + 1) * len(args.sides)
    if missing or checked != expected:
        raise RuntimeError(f'P18 camera/MANO preflight covers {checked}/{expected} rows; missing={missing[:8]}')
    return {'status': 'exact_source_K_affine_bound_to_centered_hawor_plane_for_all_interval_mano_rows', 'row_count': checked, 'expected_row_count': expected, 'unique_intrinsics_fx_fy_cx_cy': [list(row) for row in sorted(unique_intrinsics)], 'validated_before_mano_model_loading': True}
