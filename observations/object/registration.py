from __future__ import annotations

def register_adjacent_observed_surfaces(source: np.ndarray, target: np.ndarray, *, frame_gap: int, iterations: int, trim_fraction: float, max_correspondence_m: float, min_matches: int, max_median_residual_m: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate a world-to-world rigid delta using only accepted observed surfels."""
    from optimization.common.correspondences import correspondence_state
    from typing import Any
    import numpy as np
    from optimization.object.pose import rotation_angle_deg
    from optimization.object.pose import numeric_summary
    from optimization.common.solvers import rigid_umeyama
    rotation = np.eye(3, dtype=float)
    translation = target.mean(axis=0) - source.mean(axis=0)
    trace: list[dict[str, Any]] = []
    for iteration in range(iterations):
        state = correspondence_state(source, target, rotation, translation, frame_gap=frame_gap, trim_fraction=trim_fraction, max_correspondence_m=max_correspondence_m, min_matches=min_matches)
        keep = state['keep']
        (rotation, translation) = rigid_umeyama(source[keep], target[state['target_indices'][keep]])
        trace.append({'iteration': int(iteration + 1), 'match_count': int(len(keep)), 'mutual_count': int(state['mutual_count']), 'active_cap_m': float(state['active_cap_m']), 'median_residual_m': float(np.median(state['distances'][keep]))})
    final = correspondence_state(source, target, rotation, translation, frame_gap=frame_gap, trim_fraction=trim_fraction, max_correspondence_m=max_correspondence_m, min_matches=min_matches)
    keep = final['keep']
    residuals = final['distances'][keep]
    median_residual = float(np.median(residuals))
    quality_passed = bool(len(keep) >= min_matches and np.isfinite(median_residual) and (median_residual <= max_median_residual_m) and np.all(np.isfinite(rotation)) and np.all(np.isfinite(translation)) and (abs(np.linalg.det(rotation) - 1.0) <= 1e-05))
    report = {'method': 'trimmed_mutual_nearest_observed_surfel_icp', 'frame_gap': int(frame_gap), 'source_point_count': int(len(source)), 'target_point_count': int(len(target)), 'match_count': int(len(keep)), 'mutual_count': int(final['mutual_count']), 'trim_fraction': float(trim_fraction), 'active_cap_m': float(final['active_cap_m']), 'max_correspondence_m_at_gap': float(final['metric_cap_m']), 'residual_m': numeric_summary(residuals), 'delta_rotation_deg': rotation_angle_deg(rotation), 'delta_translation_m': float(np.linalg.norm(translation)), 'quality_thresholds': {'minimum_matches': int(min_matches), 'maximum_median_residual_m': float(max_median_residual_m)}, 'quality_passed': quality_passed, 'trace': trace}
    if not quality_passed:
        raise RuntimeError(f'observed-surface pairwise registration failed closed: {report}')
    return (rotation, translation, report)

def refine_translation_from_observed_anchor(anchor_canonical_points: np.ndarray, observed_world: np.ndarray, rotation: np.ndarray, initial_translation: np.ndarray, *, iterations: int, trim_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine translation at fixed rotation using no generated mesh faces."""
    from typing import Any
    import numpy as np
    from scipy.spatial import cKDTree
    from optimization.object.pose import nearest_summary
    from optimization.object.pose import apply_pose
    translation = np.asarray(initial_translation, dtype=float).copy()
    trace: list[dict[str, Any]] = []
    for iteration in range(iterations):
        model_world = apply_pose(anchor_canonical_points, rotation, translation)
        (distances, model_indices) = cKDTree(model_world).query(observed_world, k=1, workers=-1)
        keep_count = max(30, int(len(observed_world) * trim_fraction))
        keep = np.argsort(distances)[:min(len(distances), keep_count)]
        deltas = observed_world[keep] - model_world[model_indices[keep]]
        update = np.median(deltas, axis=0)
        translation += update
        trace.append({'iteration': int(iteration + 1), 'match_count': int(len(keep)), 'median_residual_m': float(np.median(distances[keep])), 'translation_update_m': update.astype(float).tolist()})
    model_world = apply_pose(anchor_canonical_points, rotation, translation)
    final = nearest_summary(observed_world, model_world)
    return (translation, {'method': 'fixed_rotation_trimmed_anchor_observed_surfel_translation', 'trim_fraction': float(trim_fraction), 'iterations': int(iterations), 'observed_to_anchor_surface_final': final, 'trace': trace})

def refine_pose_from_observed_anchor(anchor_canonical_points: np.ndarray, observed_world: np.ndarray, initial_rotation: np.ndarray, initial_translation: np.ndarray, *, iterations: int, trim_fraction: float, damping: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Obtain an observed-only unary pose candidate in the temporal chain's local basin."""
    from scipy.spatial.transform import Rotation
    from typing import Any
    import numpy as np
    from scipy.spatial import cKDTree
    from optimization.object.pose import nearest_summary
    from optimization.object.pose import apply_pose
    from optimization.common.solvers import rigid_umeyama
    rotation = np.asarray(initial_rotation, dtype=float).copy()
    translation = np.asarray(initial_translation, dtype=float).copy()
    trace: list[dict[str, Any]] = []
    for iteration in range(iterations):
        model_world = apply_pose(anchor_canonical_points, rotation, translation)
        (distances, model_indices) = cKDTree(model_world).query(observed_world, k=1, workers=-1)
        keep_count = max(30, int(len(observed_world) * trim_fraction))
        keep = np.argsort(distances)[:min(len(distances), keep_count)]
        (candidate_rotation, candidate_translation) = rigid_umeyama(anchor_canonical_points[model_indices[keep]], observed_world[keep])
        rotation_delta = Rotation.from_matrix(candidate_rotation @ rotation.T).as_rotvec()
        damped_delta = Rotation.from_rotvec(float(damping) * rotation_delta).as_matrix()
        rotation = damped_delta @ rotation
        translation = (1.0 - float(damping)) * translation + float(damping) * candidate_translation
        trace.append({'iteration': int(iteration + 1), 'match_count': int(len(keep)), 'median_residual_m': float(np.median(distances[keep])), 'undamped_rotation_update_deg': float(np.degrees(np.linalg.norm(rotation_delta)))})
    final_world = apply_pose(anchor_canonical_points, rotation, translation)
    return (rotation, translation, {'method': 'temporally_initialized_trimmed_observed_anchor_surface_icp', 'iterations': int(iterations), 'trim_fraction': float(trim_fraction), 'damping': float(damping), 'observed_to_anchor_surface_final': nearest_summary(observed_world, final_world), 'trace': trace})

def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(1, len(points) - 1)
    (vals, vecs) = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if np.linalg.det(vecs) < 0:
        vecs[:, -1] *= -1.0
    return (center, vecs, vals)

def rotation_candidates(obs_basis: np.ndarray, model_basis: np.ndarray) -> list[np.ndarray]:
    import itertools
    import numpy as np
    candidates: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        perm_mat = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            sign_mat = np.diag(signs)
            r = obs_basis @ perm_mat @ sign_mat @ model_basis.T
            if np.linalg.det(r) < 0:
                continue
            candidates.append(r)
    return candidates

def align_trellis_to_observed(trellis: trimesh.Trimesh, observed_points: np.ndarray, max_samples: int=20000) -> dict[str, Any]:
    from optimization.common.solvers import umeyama_similarity
    from observations.object.shape import nearest_stats
    import trimesh
    from typing import Any
    import numpy as np
    from scipy.spatial import cKDTree
    from observations.object.shape import completion_support_deterministic_sample_mesh
    import trimesh
    (obs_center, obs_basis, obs_vals) = pca_basis(observed_points)
    model_vertices = np.asarray(trellis.vertices, dtype=float)
    (model_center, model_basis, model_vals) = pca_basis(model_vertices)
    model_samples = completion_support_deterministic_sample_mesh(trellis, min(max_samples, max(2000, len(trellis.faces))))
    obs_radius = float(np.sqrt(np.mean(np.sum((observed_points - obs_center) ** 2, axis=1))))
    model_radius = float(np.sqrt(np.mean(np.sum((model_samples - model_center) ** 2, axis=1))))
    initial_scale = obs_radius / max(model_radius, 1e-12)
    best: dict[str, Any] | None = None
    for r in rotation_candidates(obs_basis, model_basis):
        transformed_samples = initial_scale * (model_samples @ r.T) + obs_center - initial_scale * (model_center @ r.T)
        stats = nearest_stats(observed_points, transformed_samples)
        score = stats['median_m'] + 0.25 * stats['p90_m']
        if best is None or score < best['score']:
            best = {'scale': initial_scale, 'rotation': r, 'translation': obs_center - initial_scale * (r @ model_center), 'score': float(score), 'observed_to_trellis_stats_initial': stats}
    assert best is not None
    r_total = np.asarray(best['rotation'], dtype=float)
    s_total = float(best['scale'])
    t_total = np.asarray(best['translation'], dtype=float)
    refinement_stats = []
    raw_tree_points = model_samples
    for _ in range(6):
        transformed_samples = s_total * (raw_tree_points @ r_total.T) + t_total
        tree = cKDTree(transformed_samples)
        (_, idx) = tree.query(observed_points, k=1, workers=-1)
        src_match = raw_tree_points[idx]
        (s_new, r_new, t_new) = umeyama_similarity(src_match, observed_points)
        if not np.isfinite(s_new) or s_new <= 0:
            break
        (s_total, r_total, t_total) = (s_new, r_new, t_new)
        refinement_stats.append(nearest_stats(observed_points, s_total * (raw_tree_points @ r_total.T) + t_total))
    final_samples = s_total * (model_samples @ r_total.T) + t_total
    final_stats_obs_to_model = nearest_stats(observed_points, final_samples)
    final_stats_model_to_obs = nearest_stats(final_samples, observed_points)
    return {'scale': float(s_total), 'rotation': r_total.astype(float).tolist(), 'translation': t_total.astype(float).tolist(), 'matrix_model_to_canonical': np.vstack([np.hstack([s_total * r_total, t_total.reshape(3, 1)]), np.array([[0.0, 0.0, 0.0, 1.0]])]).astype(float).tolist(), 'initial_scale_from_rms_radius': float(initial_scale), 'observed_pca_eigenvalues': obs_vals.astype(float).tolist(), 'trellis_pca_eigenvalues': model_vals.astype(float).tolist(), 'observed_to_trellis_stats_initial': best['observed_to_trellis_stats_initial'], 'observed_to_trellis_stats_final': final_stats_obs_to_model, 'trellis_to_observed_stats_final': final_stats_model_to_obs, 'icp_refinement_stats': refinement_stats}

def identity_metric_canonical_alignment(generated: trimesh.Trimesh, observed_points: np.ndarray, input_report: dict[str, Any], mesh_path: Path, max_samples: int=20000) -> dict[str, Any]:
    from utils.data import read_json as completion_support_load_json
    from observations.object.shape import nearest_stats
    import trimesh
    from pathlib import Path
    from typing import Any
    import numpy as np
    from observations.object.shape import completion_support_deterministic_sample_mesh
    from utils.state import content_sha256 as completion_support_sha256_file
    import trimesh
    if input_report.get('schema') != 'v19_metric_canonical_generated_prior_input_v1':
        raise RuntimeError('--input-mesh-already-metric-canonical requires the strict metric-canonical input schema')
    if input_report.get('status') != 'verified_sam3d_native_sensor_metric_canonical_render_prior':
        raise RuntimeError('metric-canonical generated input is not a verified SAM3D native bridge')
    contract = input_report.get('metric_canonical_input_contract')
    if not isinstance(contract, dict):
        raise RuntimeError('metric-canonical generated input lacks its contract')
    required_true = ('alignment_must_be_identity', 'generated_faces_render_only', 'native_orientation_preserved', 'camera_origin_scene_similarity_applied')
    if any((contract.get(key) is not True for key in required_true)):
        raise RuntimeError(f'metric-canonical generated input contract is incomplete: {contract}')
    if str(input_report.get('mesh_sha256') or '') != completion_support_sha256_file(mesh_path):
        raise RuntimeError('metric-canonical generated mesh hash disagrees with its input contract')
    bridge_path = Path(str(input_report.get('bridge_report') or '')).expanduser().resolve()
    if not bridge_path.is_file() or completion_support_sha256_file(bridge_path) != str(input_report.get('bridge_report_sha256') or ''):
        raise RuntimeError('SAM3D metric bridge report is missing or hash-mismatched')
    bridge = completion_support_load_json(bridge_path)
    if bridge.get('schema') != 'v19_sam3d_native_sensor_metric_canonical_bridge_v1' or not str(bridge.get('status') or '').startswith('ok_'):
        raise RuntimeError('SAM3D metric bridge report is not successful')
    output_mesh = Path(str((bridge.get('outputs') or {}).get('metric_canonical_render_prior') or '')).expanduser().resolve()
    if output_mesh != mesh_path.expanduser().resolve():
        raise RuntimeError('SAM3D metric bridge output path disagrees with P13 input mesh')
    samples = completion_support_deterministic_sample_mesh(generated, min(max_samples, max(2000, len(generated.faces))))
    observed_to_generated = nearest_stats(observed_points, samples)
    generated_to_observed = nearest_stats(samples, observed_points)
    return {'mode': 'verified_pre_aligned_metric_canonical_identity', 'scale': 1.0, 'rotation': np.eye(3, dtype=float).tolist(), 'translation': [0.0, 0.0, 0.0], 'matrix_model_to_canonical': np.eye(4, dtype=float).tolist(), 'initial_scale_from_rms_radius': None, 'observed_pca_eigenvalues': None, 'trellis_pca_eigenvalues': None, 'observed_to_trellis_stats_initial': observed_to_generated, 'observed_to_trellis_stats_final': observed_to_generated, 'trellis_to_observed_stats_final': generated_to_observed, 'icp_refinement_stats': [], 'native_orientation_preserved': True, 'pca_axis_permutation_used': False, 'icp_rotation_or_scale_used': False, 'bridge_report': str(bridge_path), 'bridge_report_sha256': completion_support_sha256_file(bridge_path)}

def transform_mesh(mesh: trimesh.Trimesh, align: dict[str, Any]) -> trimesh.Trimesh:
    import trimesh
    from typing import Any
    import numpy as np
    import trimesh
    m = np.asarray(align['matrix_model_to_canonical'], dtype=float)
    verts = np.asarray(mesh.vertices, dtype=float)
    hom = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    out = (hom @ m.T)[:, :3]
    return trimesh.Trimesh(vertices=out, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
