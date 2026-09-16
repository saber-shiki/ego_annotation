"""Direct object-pose observations from metric surfel registration and RGB evidence.

This module produces source measurements only. It does not run the legacy P15/P18
trajectory optimizers and does not grant annotation acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
METHOD = 'observed_surface_temporal_chain_unary_rgb_bridge_v2'

def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def load_mesh(path: Path) -> trimesh.Trimesh:
    import trimesh
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f'no mesh geometry in {path}')
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f'not mesh: {path}')
    if len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f'empty mesh: {path}')
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)

def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    import trimesh
    rng = np.random.default_rng(1801)
    (points, _) = trimesh.sample.sample_surface(mesh, min(count, max(1, len(mesh.faces) * 2)), seed=rng)
    return np.asarray(points, dtype=float)

def apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation

def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())))

def numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {'count': 0, 'min': None, 'median': None, 'p90': None, 'p95': None, 'mean': None, 'max': None}
    return {'count': int(len(array)), 'min': float(np.min(array)), 'median': float(np.median(array)), 'p90': float(np.percentile(array, 90)), 'p95': float(np.percentile(array, 95)), 'mean': float(np.mean(array)), 'max': float(np.max(array))}

def nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    if len(query) == 0 or len(target) == 0:
        return {'count': int(len(query)), 'median_m': None, 'p90_m': None, 'p95_m': None, 'mean_m': None, 'max_m': None}
    (distances, _) = cKDTree(target).query(query, k=1, workers=-1)
    return {'count': int(len(query)), 'median_m': float(np.median(distances)), 'p90_m': float(np.percentile(distances, 90)), 'p95_m': float(np.percentile(distances, 95)), 'mean_m': float(np.mean(distances)), 'max_m': float(np.max(distances))}

def rotation_observability_summary(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
        return {'observable': False, 'score': 0.0, 'eigenvalues_ascending': [], 'reason': 'too_few_points'}
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(points) - 1)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    score = float(eigenvalues[1] / max(eigenvalues[2], 1e-12))
    return {'observable': bool(np.isfinite(score) and score >= 0.02), 'score': score, 'eigenvalues_ascending': eigenvalues.astype(float).tolist(), 'threshold': 0.02, 'reason': 'second_to_first_principal_variance_ratio'}

def frame_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for candidate in frame.get('objects', []) if isinstance(frame.get('objects'), list) else []:
        if isinstance(candidate, dict) and candidate.get('object_id') == object_id:
            return candidate
    return None

def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise RuntimeError(f'failed to read grayscale frame: {path}')
    return image

def read_mask_at_size(path: str | Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2:
        raise RuntimeError(f'failed to read object-owned mask: {path}')
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
    return mask > 0

def frame_camera_for_raster(frame: dict[str, Any], width: int, height: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
    transform = np.asarray(camera.get('T_world_camera_metric') or [], dtype=float)
    values = np.asarray(camera.get('intrinsics_fx_fy_cx_cy') or [], dtype=float)
    source_width = int(frame.get('source_width') or width)
    source_height = int(frame.get('source_height') or height)
    if transform.shape != (4, 4) or values.shape != (4,):
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks calibrated camera state for optical-flow PnP")
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"frame {frame.get('frame_idx')} has invalid source raster size")
    sx = float(width) / float(source_width)
    sy = float(height) / float(source_height)
    fx = float(values[0] * sx)
    fy = float(values[1] * sy)
    cx = float((values[2] + 0.5) * sx - 0.5)
    cy = float((values[3] + 0.5) * sy - 0.5)
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    return (transform, intrinsics, {'source_intrinsics_fx_fy_cx_cy': values.astype(float).tolist(), 'source_size_wh': [source_width, source_height], 'raster_size_wh': [int(width), int(height)], 'A_raster_from_source': [[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]], 'raster_intrinsics': intrinsics.astype(float).tolist(), 'pixel_center_convention': 'OpenCV half-pixel resize mapping'})

def bridge_positions_for_gap(left: int, right: int, maximum_gap: int) -> list[int]:
    gap = int(right - left)
    if gap <= maximum_gap:
        return []
    segments = int(math.ceil(gap / maximum_gap))
    positions = [int(round(left + gap * segment / segments)) for segment in range(1, segments)]
    return sorted(set((position for position in positions if left < position < right)))

def complete_translation_measurements(direct_translations: dict[int, np.ndarray], frame_indices: list[int], *, bridge_frames: set[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    direct_indices = sorted(direct_translations)
    measurements: list[np.ndarray] = []
    weights: list[float] = []
    sources: list[str] = []
    for frame_idx in frame_indices:
        if frame_idx in direct_translations:
            measurements.append(np.asarray(direct_translations[frame_idx], dtype=float))
            weights.append(0.75 if frame_idx in bridge_frames else 1.0)
            sources.append('direct_rgb_flow_pnp' if frame_idx in bridge_frames else 'direct_metric_surface')
            continue
        lower = max((idx for idx in direct_indices if idx < frame_idx), default=None)
        upper = min((idx for idx in direct_indices if idx > frame_idx), default=None)
        if lower is not None and upper is not None:
            alpha = float((frame_idx - lower) / (upper - lower))
            value = (1.0 - alpha) * direct_translations[lower] + alpha * direct_translations[upper]
            measurements.append(np.asarray(value, dtype=float))
            weights.append(0.15)
            sources.append('interpolated_regularization_support')
        else:
            nearest = min(direct_indices, key=lambda idx: abs(idx - frame_idx))
            measurements.append(np.asarray(direct_translations[nearest], dtype=float))
            weights.append(0.08)
            sources.append('edge_hold_regularization_support')
    return (np.asarray(measurements, dtype=float), np.asarray(weights, dtype=float), sources)

def fit_observed_pose(argv=None) -> None:
    from observations.object.rgb_tracking import estimate_optical_flow_pnp_bridge
    from observations.object.registration import refine_pose_from_observed_anchor
    from observations.object.registration import refine_translation_from_observed_anchor
    from observations.object.registration import register_adjacent_observed_surfaces
    from utils.numerics import regularize_translation_timeline
    from envs.qualification.object_observation import rigid_pose_observation_eligibility
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--completion-report', type=Path, required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=6000)
    parser.add_argument('--iterations', type=int, default=4, help='Compatibility base; observed pairwise ICP uses max(8, 3*iterations)')
    parser.add_argument('--pairwise-trim-fraction', type=float, default=0.65)
    parser.add_argument('--pairwise-max-correspondence-m', type=float, default=0.035)
    parser.add_argument('--pairwise-min-matches', type=int, default=30)
    parser.add_argument('--pairwise-max-median-residual-m', type=float, default=0.015)
    parser.add_argument('--anchor-translation-trim-fraction', type=float, default=0.8)
    parser.add_argument('--anchor-translation-iterations', type=int, default=6)
    parser.add_argument('--rotation-unary-trim-fraction', type=float, default=0.75)
    parser.add_argument('--rotation-unary-iterations', type=int, default=6)
    parser.add_argument('--rotation-unary-damping', type=float, default=0.5)
    parser.add_argument('--temporal-rotation-correction-velocity-weight', type=float, default=6.0)
    parser.add_argument('--temporal-rotation-correction-acceleration-weight', type=float, default=4.0)
    parser.add_argument('--temporal-rotation-chain-prior-weight', type=float, default=0.75)
    parser.add_argument('--maximum-rotation-unary-correction-deg', type=float, default=60.0)
    parser.add_argument('--temporal-translation-velocity-weight', type=float, default=0.1)
    parser.add_argument('--temporal-translation-acceleration-weight', type=float, default=2.0)
    parser.add_argument('--maximum-direct-pose-gap', type=int, default=10)
    parser.add_argument('--minimum-direct-pose-fraction', type=float, default=0.8)
    parser.add_argument('--flow-max-seed-points', type=int, default=1200)
    parser.add_argument('--flow-min-tracked-points', type=int, default=50)
    parser.add_argument('--flow-min-pnp-inliers', type=int, default=40)
    parser.add_argument('--flow-min-pnp-inlier-fraction', type=float, default=0.5)
    parser.add_argument('--flow-max-reprojection-median-px', type=float, default=2.5)
    parser.add_argument('--flow-max-reprojection-p95-px', type=float, default=4.0)
    parser.add_argument('--include-ineligible-rigid-pose-observations', action='store_true', help='Historical-reproduction override: consume rows explicitly rejected by P09 (forbidden in annotation runs)')
    args = parser.parse_args(argv)
    if args.include_ineligible_rigid_pose_observations:
        raise RuntimeError('--include-ineligible-rigid-pose-observations is historical-only and forbidden by the observed-only annotation contract')
    if not 0.0 < args.pairwise_trim_fraction <= 1.0:
        raise RuntimeError('--pairwise-trim-fraction must be in (0,1]')
    if not 0.0 < args.anchor_translation_trim_fraction <= 1.0:
        raise RuntimeError('--anchor-translation-trim-fraction must be in (0,1]')
    if not 0.0 < args.rotation_unary_trim_fraction <= 1.0:
        raise RuntimeError('--rotation-unary-trim-fraction must be in (0,1]')
    if not 0.0 < args.rotation_unary_damping <= 1.0:
        raise RuntimeError('--rotation-unary-damping must be in (0,1]')
    if not 0.0 < args.minimum_direct_pose_fraction <= 1.0:
        raise RuntimeError('--minimum-direct-pose-fraction must be in (0,1]')
    annotations = load_json(args.annotations)
    completion = load_json(args.completion_report)
    completion_outputs = completion.get('outputs') if isinstance(completion.get('outputs'), dict) else {}
    mesh_value = completion_outputs.get('pose_hypothesis_mesh_labeled') or completion_outputs.get('completed_mesh_labeled')
    if not mesh_value:
        raise RuntimeError('completion report lacks outputs.pose_hypothesis_mesh_labeled/completed_mesh_labeled')
    mesh_path = Path(str(mesh_value))
    generated_diagnostic_mesh = load_mesh(mesh_path)
    generated_diagnostic_samples = deterministic_sample_mesh(generated_diagnostic_mesh, args.sample_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_frames = [frame for frame in annotations.get('frames', []) if isinstance(frame, dict)]
    if not annotation_frames:
        raise RuntimeError('annotations contain no frames')
    frames = {int(frame['frame_idx']): frame for frame in annotation_frames}
    frame_indices = sorted(frames)
    objects: dict[int, dict[str, Any]] = {}
    observations: dict[int, np.ndarray] = {}
    eligibility_by_frame: dict[int, tuple[bool | None, list[str], list[str]]] = {}
    initial_rows: dict[int, dict[str, Any]] = {}
    explicit_eligible_input_count = 0
    explicit_ineligible_input_count = 0
    eligibility_unspecified_input_count = 0
    missing_observed = 0
    ineligible_reason_counts: dict[str, int] = {}
    for frame_idx in frame_indices:
        frame = frames[frame_idx]
        obj = frame_object(frame, args.object_id)
        if obj is None:
            initial_rows[frame_idx] = {'frame_idx': int(frame_idx), 'status': 'missing_object_row', 'object_id': args.object_id}
            continue
        objects[frame_idx] = obj
        geom = obj.get('visible_geometry_candidate') if isinstance(obj.get('visible_geometry_candidate'), dict) else {}
        eligibility = rigid_pose_observation_eligibility(obj, geom)
        eligibility_by_frame[frame_idx] = eligibility
        observed = np.asarray(geom.get('world_vertices_sample_m') or [], dtype=float)
        if eligibility[0] is True:
            explicit_eligible_input_count += 1
        elif eligibility[0] is False:
            explicit_ineligible_input_count += 1
            for reason in eligibility[1] or ['explicit_false_without_reason']:
                ineligible_reason_counts[reason] = ineligible_reason_counts.get(reason, 0) + 1
        else:
            eligibility_unspecified_input_count += 1
        if eligibility[0] is False:
            initial_rows[frame_idx] = {'frame_idx': int(frame_idx), 'status': 'rigid_pose_observation_ineligible', 'object_id': args.object_id, 'rigid_pose_observation_eligible': False, 'rigid_pose_observation_reasons': eligibility[1], 'rigid_pose_observation_eligibility_sources': eligibility[2], 'visible_sample_count': int(len(observed)) if observed.ndim == 2 else 0, 'policy': 'explicit P09 false is a hard metric-surface measurement rejection gate'}
            continue
        if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 6 or (not np.all(np.isfinite(observed))):
            missing_observed += 1
            initial_rows[frame_idx] = {'frame_idx': int(frame_idx), 'status': 'missing_accepted_metric_surface', 'object_id': args.object_id, 'appearance_mask_path': obj.get('mask_path')}
            continue
        observations[frame_idx] = observed
    adapter = annotations.get('v19_visible_geometry_adapter') if isinstance(annotations.get('v19_visible_geometry_adapter'), dict) else {}
    anchor_value = adapter.get('anchor_frame_idx')
    if anchor_value is None:
        raise RuntimeError('annotations lack v19_visible_geometry_adapter.anchor_frame_idx')
    anchor_frame_idx = int(anchor_value)
    completion_inputs = completion.get('inputs') if isinstance(completion.get('inputs'), dict) else {}
    completion_binding = completion_inputs.get('selected_anchor_atomic_binding')
    evidence_path = Path(str(completion_inputs.get('candidate_evidence_report') or ''))
    expected_evidence_sha256 = str(completion_inputs.get('candidate_evidence_report_sha256') or '')
    if not evidence_path.is_file() or not expected_evidence_sha256:
        raise RuntimeError('P14 completion lacks a byte-bound selected-anchor evidence report')
    actual_evidence_sha256 = content_sha256(evidence_path)
    if actual_evidence_sha256 != expected_evidence_sha256:
        raise RuntimeError('P14 selected-anchor evidence report hash changed after P14 reference creation')
    anchor_evidence = load_json(evidence_path)
    if anchor_evidence.get('selected_anchor_atomic_binding') != completion_binding:
        raise RuntimeError('P14 completion/evidence selected-anchor bindings differ')
    evidence_selected = anchor_evidence.get('selected') if isinstance(anchor_evidence.get('selected'), dict) else {}
    evidence_geom = evidence_selected.get('visible_geometry_candidate') if isinstance(evidence_selected.get('visible_geometry_candidate'), dict) else {}
    completion_binding_centroid = np.asarray(evidence_geom.get('centroid_world_m') or [], dtype=float).reshape(-1)
    if not isinstance(completion_binding, dict) or completion_binding.get('validated') is not True or int(completion_binding.get('selected_frame_idx', -1)) != anchor_frame_idx or (int(completion_binding.get('visible_geometry_frame_idx', -1)) != anchor_frame_idx) or (int(completion_binding.get('canonical_surface_frame_idx', -1)) != anchor_frame_idx) or (completion_binding_centroid.shape != (3,)) or (not np.isfinite(completion_binding_centroid).all()):
        raise RuntimeError(f'P14 annotations/completion do not share one atomic selected anchor: {completion_binding}')
    if anchor_frame_idx not in observations:
        raise RuntimeError(f'selected anchor frame {anchor_frame_idx} lacks accepted metric surfels')
    anchor_world = observations[anchor_frame_idx]
    anchor_centroid = anchor_world.mean(axis=0)
    anchor_binding_centroid_error_m = float(np.linalg.norm(anchor_centroid - completion_binding_centroid))
    if anchor_binding_centroid_error_m > 1e-06:
        raise RuntimeError(f'P14 selected anchor surfels disagree with the atomic completion centroid by {anchor_binding_centroid_error_m} m')
    anchor_canonical = anchor_world - anchor_centroid
    anchor_extent = np.ptp(anchor_world, axis=0)
    anchor_diag = float(np.linalg.norm(anchor_extent))
    if not np.isfinite(anchor_diag) or anchor_diag <= 0.0:
        raise RuntimeError('selected anchor observed surface has invalid metric extent')
    depth_indices = sorted(observations)
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {anchor_frame_idx: (np.eye(3, dtype=float), anchor_centroid.astype(float))}
    edge_reports: list[dict[str, Any]] = []
    pairwise_iterations = max(8, int(args.iterations) * 3)
    previous = anchor_frame_idx
    for current in [idx for idx in depth_indices if idx > anchor_frame_idx]:
        (delta_rotation, delta_translation, edge_report) = register_adjacent_observed_surfaces(observations[previous], observations[current], frame_gap=current - previous, iterations=pairwise_iterations, trim_fraction=float(args.pairwise_trim_fraction), max_correspondence_m=float(args.pairwise_max_correspondence_m), min_matches=int(args.pairwise_min_matches), max_median_residual_m=float(args.pairwise_max_median_residual_m))
        (previous_rotation, previous_translation) = poses[previous]
        poses[current] = (delta_rotation @ previous_rotation, delta_rotation @ previous_translation + delta_translation)
        edge_reports.append({'source_frame_idx': previous, 'target_frame_idx': current, **edge_report})
        previous = current
    previous = anchor_frame_idx
    for current in sorted([idx for idx in depth_indices if idx < anchor_frame_idx], reverse=True):
        (delta_rotation, delta_translation, edge_report) = register_adjacent_observed_surfaces(observations[previous], observations[current], frame_gap=previous - current, iterations=pairwise_iterations, trim_fraction=float(args.pairwise_trim_fraction), max_correspondence_m=float(args.pairwise_max_correspondence_m), min_matches=int(args.pairwise_min_matches), max_median_residual_m=float(args.pairwise_max_median_residual_m))
        (previous_rotation, previous_translation) = poses[previous]
        poses[current] = (delta_rotation @ previous_rotation, delta_rotation @ previous_translation + delta_translation)
        edge_reports.append({'source_frame_idx': previous, 'target_frame_idx': current, **edge_report})
        previous = current
    preliminary_translation_reports: dict[int, dict[str, Any]] = {}
    for frame_idx in depth_indices:
        (rotation, initial_translation) = poses[frame_idx]
        (stabilized_translation, stabilization_report) = refine_translation_from_observed_anchor(anchor_canonical, observations[frame_idx], rotation, initial_translation, iterations=int(args.anchor_translation_iterations), trim_fraction=float(args.anchor_translation_trim_fraction))
        poses[frame_idx] = (rotation, stabilized_translation)
        preliminary_translation_reports[frame_idx] = stabilization_report
    chain_poses = {idx: (pose[0].copy(), pose[1].copy()) for (idx, pose) in poses.items()}
    rotation_unary_reports: dict[int, dict[str, Any]] = {}
    rotation_corrections: dict[int, np.ndarray] = {}
    for frame_idx in depth_indices:
        (chain_rotation, chain_translation) = chain_poses[frame_idx]
        (unary_rotation, _, unary_report) = refine_pose_from_observed_anchor(anchor_canonical, observations[frame_idx], chain_rotation, chain_translation, iterations=int(args.rotation_unary_iterations), trim_fraction=float(args.rotation_unary_trim_fraction), damping=float(args.rotation_unary_damping))
        correction = Rotation.from_matrix(unary_rotation @ chain_rotation.T).as_rotvec()
        rotation_corrections[frame_idx] = correction.astype(float)
        rotation_unary_reports[frame_idx] = unary_report
    (correction_measurements, correction_weights, _) = complete_translation_measurements(rotation_corrections, frame_indices, bridge_frames=set())
    regularized_corrections = regularize_translation_timeline(correction_measurements, correction_weights, velocity_weight=float(args.temporal_rotation_correction_velocity_weight), acceleration_weight=float(args.temporal_rotation_correction_acceleration_weight), zero_prior_weight=float(args.temporal_rotation_chain_prior_weight))
    frame_to_row = {frame_idx: row for (row, frame_idx) in enumerate(frame_indices)}
    regularized_corrections -= regularized_corrections[frame_to_row[anchor_frame_idx]]
    direct_rotation_correction_deg = {idx: float(np.degrees(np.linalg.norm(regularized_corrections[frame_to_row[idx]]))) for idx in depth_indices}
    if max(direct_rotation_correction_deg.values(), default=0.0) > float(args.maximum_rotation_unary_correction_deg):
        raise RuntimeError(f'observed-anchor unary rotation correction left the pairwise temporal basin: summary={numeric_summary(list(direct_rotation_correction_deg.values()))} maximum_allowed_deg={args.maximum_rotation_unary_correction_deg}')
    for frame_idx in depth_indices:
        (chain_rotation, chain_translation) = chain_poses[frame_idx]
        correction_rotation = Rotation.from_rotvec(regularized_corrections[frame_to_row[frame_idx]]).as_matrix()
        poses[frame_idx] = (correction_rotation @ chain_rotation, chain_translation)
    translation_fit_reports: dict[int, dict[str, Any]] = {}
    for frame_idx in depth_indices:
        (rotation, initial_translation) = poses[frame_idx]
        (translation, translation_report) = refine_translation_from_observed_anchor(anchor_canonical, observations[frame_idx], rotation, initial_translation, iterations=int(args.anchor_translation_iterations), trim_fraction=float(args.anchor_translation_trim_fraction))
        poses[frame_idx] = (rotation, translation)
        translation_fit_reports[frame_idx] = translation_report
    bridge_reports: dict[int, dict[str, Any]] = {}
    bridge_observability: dict[int, dict[str, Any]] = {}
    initial_direct_indices = sorted(poses)
    requested_bridge_positions: list[tuple[int, int, int]] = []
    for (left, right) in zip(initial_direct_indices[:-1], initial_direct_indices[1:]):
        for target in bridge_positions_for_gap(left, right, int(args.maximum_direct_pose_gap)):
            requested_bridge_positions.append((left, target, right))
    required_direct_count = int(math.ceil(float(args.minimum_direct_pose_fraction) * len(frame_indices) - 1e-12))
    if len(poses) + len(requested_bridge_positions) < required_direct_count:
        missing_candidates = [idx for idx in frame_indices if idx not in poses]
        ranked = sorted(missing_candidates, key=lambda idx: min((abs(idx - direct) for direct in initial_direct_indices)))
        already = {target for (_, target, _) in requested_bridge_positions}
        for target in ranked:
            if len(poses) + len(already) >= required_direct_count:
                break
            if target in already:
                continue
            left = max((idx for idx in initial_direct_indices if idx < target), default=None)
            right = min((idx for idx in initial_direct_indices if idx > target), default=None)
            if left is not None and right is not None:
                requested_bridge_positions.append((left, target, right))
                already.add(target)
    for (left, target, right) in requested_bridge_positions:
        if target in poses:
            continue
        (rotation, translation, bridge_report, observability) = estimate_optical_flow_pnp_bridge(source_idx=left, target_idx=target, frames=frames, objects=objects, observed_world=observations[left], source_rotation=poses[left][0], source_translation=poses[left][1], max_seed_points=int(args.flow_max_seed_points), min_tracked_points=int(args.flow_min_tracked_points), min_pnp_inliers=int(args.flow_min_pnp_inliers), min_inlier_fraction=float(args.flow_min_pnp_inlier_fraction), max_reprojection_median_px=float(args.flow_max_reprojection_median_px), max_reprojection_p95_px=float(args.flow_max_reprojection_p95_px))
        left_gap = max(1, target - left)
        right_gap = max(1, right - target)
        left_rotation_rate = rotation_angle_deg(rotation @ poses[left][0].T) / left_gap
        left_translation_rate = float(np.linalg.norm(translation - poses[left][1]) / left_gap)
        right_rotation_rate = rotation_angle_deg(poses[right][0] @ rotation.T) / right_gap
        right_translation_rate = float(np.linalg.norm(poses[right][1] - translation) / right_gap)
        endpoint_consistency_passed = bool(max(left_rotation_rate, right_rotation_rate) <= 20.0 and max(left_translation_rate, right_translation_rate) <= 0.1)
        bridge_report['neighbor_depth_pose_consistency'] = {'left_depth_frame_idx': int(left), 'right_depth_frame_idx': int(right), 'left_implied_rotation_deg_per_frame': float(left_rotation_rate), 'right_implied_rotation_deg_per_frame': float(right_rotation_rate), 'left_implied_translation_m_per_frame': float(left_translation_rate), 'right_implied_translation_m_per_frame': float(right_translation_rate), 'maximum_rotation_deg_per_frame': 20.0, 'maximum_translation_m_per_frame': 0.1, 'passed': endpoint_consistency_passed}
        if not endpoint_consistency_passed:
            raise RuntimeError(f'optical-flow PnP bridge endpoint consistency failed closed: {bridge_report}')
        poses[target] = (rotation, translation)
        bridge_reports[target] = bridge_report
        bridge_observability[target] = observability
    direct_indices = sorted(poses)
    direct_gaps = [right - left for (left, right) in zip(direct_indices[:-1], direct_indices[1:])]
    maximum_direct_gap = max(direct_gaps, default=0)
    if len(direct_indices) < required_direct_count or maximum_direct_gap > int(args.maximum_direct_pose_gap):
        raise RuntimeError(f'observed pose support remains below the fixed temporal contract after fail-closed RGB bridges: direct={len(direct_indices)}/{len(frame_indices)} required={required_direct_count} max_gap={maximum_direct_gap} allowed={args.maximum_direct_pose_gap}')
    direct_translations = {idx: pose[1].copy() for (idx, pose) in poses.items()}
    bridge_frames = set(bridge_reports)
    (translation_measurements, translation_weights, translation_sources) = complete_translation_measurements(direct_translations, frame_indices, bridge_frames=bridge_frames)
    regularized_translations = regularize_translation_timeline(translation_measurements, translation_weights, velocity_weight=float(args.temporal_translation_velocity_weight), acceleration_weight=float(args.temporal_translation_acceleration_weight))
    frame_to_row = {frame_idx: row for (row, frame_idx) in enumerate(frame_indices)}
    anchor_offset = anchor_centroid - regularized_translations[frame_to_row[anchor_frame_idx]]
    regularized_translations += anchor_offset
    direct_adjustments = {idx: float(np.linalg.norm(regularized_translations[frame_to_row[idx]] - direct_translations[idx])) for idx in direct_indices}
    maximum_allowed_adjustment_m = max(0.075, 0.25 * anchor_diag)
    if float(np.median(list(direct_adjustments.values()))) > 0.03 or max(direct_adjustments.values(), default=0.0) > maximum_allowed_adjustment_m:
        raise RuntimeError(f'temporal translation regularization required implausibly large corrections: summary={numeric_summary(list(direct_adjustments.values()))} max_allowed={maximum_allowed_adjustment_m}')
    for idx in direct_indices:
        poses[idx] = (poses[idx][0], regularized_translations[frame_to_row[idx]].copy())
    rows: list[dict[str, Any]] = []
    generated_diagnostic_medians: list[float] = []
    observed_anchor_medians: list[float] = []
    for frame_idx in frame_indices:
        if frame_idx not in poses:
            rows.append(initial_rows.get(frame_idx, {'frame_idx': frame_idx, 'status': 'no_direct_pose_observation'}))
            continue
        (rotation, translation) = poses[frame_idx]
        is_bridge = frame_idx in bridge_reports
        if is_bridge:
            generated_observed = nearest_summary(np.empty((0, 3), dtype=float), generated_diagnostic_samples)
            generated_reverse = nearest_summary(np.empty((0, 3), dtype=float), generated_diagnostic_samples)
            anchor_observed = nearest_summary(np.empty((0, 3), dtype=float), anchor_canonical)
            observability = bridge_observability[frame_idx]
            visible_sample_count = 0
            direct_source = 'object_owned_rgb_optical_flow_calibrated_pnp'
        else:
            observed = observations[frame_idx]
            generated_world = apply_pose(generated_diagnostic_samples, rotation, translation)
            anchor_world_model = apply_pose(anchor_canonical, rotation, translation)
            generated_observed = nearest_summary(observed, generated_world)
            generated_reverse = nearest_summary(generated_world, observed)
            anchor_observed = nearest_summary(observed, anchor_world_model)
            generated_diagnostic_medians.append(float(generated_observed['median_m']))
            observed_anchor_medians.append(float(anchor_observed['median_m']))
            observability = rotation_observability_summary(observed)
            visible_sample_count = int(len(observed))
            direct_source = 'adjacent_observed_metric_surfel_registration'
        (eligibility, eligibility_reasons, eligibility_sources) = eligibility_by_frame.get(frame_idx, (None, [], []))
        row = {'frame_idx': int(frame_idx), 'status': 'fit_to_object_owned_rgb_calibrated_pnp' if is_bridge else 'fit_to_visible_depth_samples', 'object_id': args.object_id, 'rotation_world_from_completed_canonical_matrix': rotation.astype(float).tolist(), 'translation_world_m': translation.astype(float).tolist(), 'pose_source': 'direct_observed_evidence_with_fixed_temporal_regularization', 'direct_pose_observation_source': direct_source, 'generated_geometry_pose_evidence_consumed': False, 'visible_sample_count': visible_sample_count, 'rigid_pose_observation_eligible': eligibility, 'rigid_pose_observation_reasons': eligibility_reasons, 'rigid_pose_observation_eligibility_sources': eligibility_sources, 'observed_to_mesh_initial': generated_observed, 'observed_to_mesh_final': generated_observed, 'mesh_to_observed_final': generated_reverse, 'generated_mesh_metric_role': 'post_fit_diagnostic_only_not_pose_evidence', 'observed_anchor_to_current_final': anchor_observed, 'rotation_observability': observability, 'translation_regularization': {'measurement_translation_world_m': direct_translations[frame_idx].astype(float).tolist(), 'regularized_translation_world_m': translation.astype(float).tolist(), 'adjustment_m': direct_adjustments[frame_idx], 'measurement_weight': 0.75 if is_bridge else 1.0}, 'icp_trace': translation_fit_reports.get(frame_idx, {}).get('trace', [])}
        if is_bridge:
            evidence_canonical = np.asarray(bridge_reports[frame_idx].get('pose_evidence_canonical_points_m') or [], dtype=float)
            row['rgb_optical_flow_pnp_bridge'] = bridge_reports[frame_idx]
            row['direct_pose_evidence_world_points_m'] = apply_pose(evidence_canonical, rotation, translation).astype(float).tolist()
            row['direct_pose_evidence_kind'] = 'tracked_source_metric_surfels_at_target_rgb_frame'
        else:
            row['observed_anchor_translation_fit'] = translation_fit_reports[frame_idx]
            row['preliminary_chain_translation_stabilization'] = preliminary_translation_reports[frame_idx]
            row['observed_anchor_rotation_unary_fit'] = rotation_unary_reports[frame_idx]
            row['rotation_correction_from_pairwise_chain'] = {'regularized_correction_rotvec_rad': regularized_corrections[frame_to_row[frame_idx]].astype(float).tolist(), 'regularized_correction_deg': direct_rotation_correction_deg[frame_idx], 'generated_geometry_consumed': False}
        rows.append(row)
    direct_rotation_steps: list[float] = []
    direct_translation_steps: list[float] = []
    for (left, right) in zip(direct_indices[:-1], direct_indices[1:]):
        (left_rotation, left_translation) = poses[left]
        (right_rotation, right_translation) = poses[right]
        gap = max(1, right - left)
        direct_rotation_steps.append(rotation_angle_deg(right_rotation @ left_rotation.T) / gap)
        direct_translation_steps.append(float(np.linalg.norm(right_translation - left_translation) / gap))
    report = {'method': 'fit_v18_compact_rigid_object_pose', 'status': 'ok', 'pose_estimation_method': METHOD, 'object_id': args.object_id, 'annotations': str(args.annotations), 'completion_report': str(args.completion_report), 'pose_hypothesis_mesh_labeled': str(mesh_path), 'generated_geometry_pose_evidence_consumed': False, 'generated_mesh_role': 'post_fit_distance_diagnostic_and_downstream_render_binding_only', 'generated_faces_pose_eligible': False, 'generated_faces_collision_eligible': False, 'generated_faces_contact_eligible': False, 'claim_scope': 'Shared prediction-side rigid trajectory from accepted metric surfel-to-surfel temporal registration, anchor observed-surface translation, and sparse object-owned RGB calibrated-PnP gap bridges. Generated completion faces are never pose correspondences.', 'anchor_frame_idx': int(anchor_frame_idx), 'anchor_centroid_world_m': anchor_centroid.astype(float).tolist(), 'selected_anchor_atomic_binding': completion_binding, 'selected_anchor_evidence_report': str(evidence_path), 'selected_anchor_evidence_report_sha256': actual_evidence_sha256, 'selected_anchor_binding_centroid_error_m': anchor_binding_centroid_error_m, 'anchor_observed_extent_m': anchor_extent.astype(float).tolist(), 'anchor_observed_extent_diag_m': anchor_diag, 'anchor_canonical_source': 'selected_anchor_metric_surfels_minus_anchor_centroid_world_m', 'sample_count': int(len(generated_diagnostic_samples)), 'iterations': int(args.iterations), 'pairwise_observed_surface_registration': {'edge_count': int(len(edge_reports)), 'iterations_per_edge': int(pairwise_iterations), 'trim_fraction': float(args.pairwise_trim_fraction), 'max_correspondence_m_base': float(args.pairwise_max_correspondence_m), 'minimum_matches': int(args.pairwise_min_matches), 'maximum_median_residual_m': float(args.pairwise_max_median_residual_m), 'delta_rotation_deg': numeric_summary([float(row['delta_rotation_deg']) for row in edge_reports]), 'median_residual_m': numeric_summary([float(row['residual_m']['median']) for row in edge_reports]), 'edges': edge_reports}, 'rgb_optical_flow_pnp_gap_bridges': {'enabled': True, 'requested': [{'left_depth_frame_idx': left, 'target_frame_idx': target, 'right_depth_frame_idx': right} for (left, target, right) in requested_bridge_positions], 'accepted_frame_count': int(len(bridge_reports)), 'accepted_frames': sorted(bridge_reports), 'reports': {str(idx): bridge_reports[idx] for idx in sorted(bridge_reports)}, 'generated_geometry_consumed': False}, 'rotation_unary_regularization': {'method': 'observed_anchor_surface_unary_icp_correction_over_pairwise_observed_chain', 'unary_iterations': int(args.rotation_unary_iterations), 'unary_trim_fraction': float(args.rotation_unary_trim_fraction), 'unary_damping': float(args.rotation_unary_damping), 'correction_velocity_weight': float(args.temporal_rotation_correction_velocity_weight), 'correction_acceleration_weight': float(args.temporal_rotation_correction_acceleration_weight), 'pairwise_chain_zero_correction_prior_weight': float(args.temporal_rotation_chain_prior_weight), 'anchor_correction_fixed_to_zero': True, 'direct_regularized_correction_deg': numeric_summary(list(direct_rotation_correction_deg.values())), 'maximum_allowed_correction_deg': float(args.maximum_rotation_unary_correction_deg), 'generated_geometry_consumed': False}, 'translation_regularization': {'method': 'weighted_quadratic_velocity_and_acceleration_prior', 'velocity_weight': float(args.temporal_translation_velocity_weight), 'acceleration_weight': float(args.temporal_translation_acceleration_weight), 'direct_metric_surface_weight': 1.0, 'direct_rgb_flow_pnp_weight': 0.75, 'interpolated_support_weight': 0.15, 'edge_hold_support_weight': 0.08, 'anchor_recenter_offset_m': anchor_offset.astype(float).tolist(), 'direct_adjustment_m': numeric_summary(list(direct_adjustments.values())), 'maximum_allowed_direct_adjustment_m': maximum_allowed_adjustment_m, 'measurement_source_counts': {source: int(translation_sources.count(source)) for source in sorted(set(translation_sources))}}, 'frame_fit_rows': rows, 'pose_rows': rows, 'fitted_frame_count': int(len(direct_indices)), 'direct_pose_frames': direct_indices, 'direct_metric_surface_pose_frames': sorted(observations), 'direct_rgb_flow_pnp_pose_frames': sorted(bridge_reports), 'direct_support': {'frame_count': int(len(frame_indices)), 'required_direct_count': required_direct_count, 'direct_count': int(len(direct_indices)), 'direct_fraction': float(len(direct_indices) / len(frame_indices)), 'maximum_direct_gap': int(maximum_direct_gap), 'maximum_allowed_direct_gap': int(args.maximum_direct_pose_gap)}, 'missing_initial_pose_count': 0, 'missing_observed_count': int(missing_observed), 'rigid_pose_eligibility_gate': {'include_ineligible_override': False, 'explicit_eligible_input_count': int(explicit_eligible_input_count), 'explicit_ineligible_input_count': int(explicit_ineligible_input_count), 'eligibility_unspecified_input_count': int(eligibility_unspecified_input_count), 'ineligible_observation_count': int(explicit_ineligible_input_count), 'ineligible_observation_frames': sorted((idx for (idx, value) in eligibility_by_frame.items() if value[0] is False)), 'ineligible_reason_counts': ineligible_reason_counts, 'explicit_eligible_fit_count': int(sum((eligibility_by_frame.get(idx, (None, [], []))[0] is True for idx in observations))), 'eligibility_unspecified_fit_count': int(sum((eligibility_by_frame.get(idx, (None, [], []))[0] is None for idx in observations))), 'ineligible_override_fit_count': 0, 'excluded_from_fit_count': int(explicit_ineligible_input_count), 'policy': 'explicit P09 false remains a hard metric-surface rejection; RGB bridges use appearance masks but never rejected depth'}, 'observed_pose_evidence_diagnostics': {'observed_anchor_to_current_median_summary_m': numeric_summary(observed_anchor_medians), 'direct_rotation_deg_per_frame': numeric_summary(direct_rotation_steps), 'direct_translation_m_per_frame': numeric_summary(direct_translation_steps)}, 'final_observed_to_mesh_median_summary_m': numeric_summary(generated_diagnostic_medians), 'final_observed_to_mesh_notice': 'generated mesh distance is diagnostic only and was not optimized', 'output': {'pose_hypothesis_mesh_labeled': str(mesh_path), 'collision_eligible_mesh_labeled': completion_outputs.get('collision_eligible_mesh_labeled'), 'pose_report': str(args.output_dir / 'v18_compact_rigid_object_pose_fit_report.json')}}
    output_path = args.output_dir / 'v18_compact_rigid_object_pose_fit_report.json'
    output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
