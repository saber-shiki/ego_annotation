"""Candidate-dependent observed correspondences, refreshed once per outer pass."""
from __future__ import annotations
from typing import Any, Sequence, Mapping
import numpy as np
from scipy.spatial import cKDTree
from utils.data import PoseNode, PointFactor, SolverConfig
from utils.geometry import _array, inverse_pose

def estimate_normals(points: np.ndarray, k: int=16) -> np.ndarray:
    points = _array(points)
    if len(points) < 3:
        return np.tile(np.asarray([[0.0, 0.0, 1.0]]), (len(points), 1))
    kk = min(max(3, int(k)), len(points))
    (_distances, indices) = cKDTree(points).query(points, k=kk, workers=1)
    centered = points[indices] - points[indices].mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered) / max(1, kk - 1)
    (_values, vectors) = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)

def _mutual_pairs(source: np.ndarray, target: np.ndarray, max_distance: float, max_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source) == 0 or len(target) == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0))
    (d, j) = cKDTree(target).query(source, k=1, workers=1)
    reverse = cKDTree(source).query(target, k=1, workers=1)[1]
    source_indices = np.arange(len(source), dtype=np.int64)
    keep = np.flatnonzero(reverse[j] == source_indices)
    if len(keep) < 3:
        keep = np.argsort(d)[:min(len(d), max(3, int(max_pairs)))]
    keep = keep[d[keep] <= float(max_distance)]
    if len(keep) > int(max_pairs):
        keep = keep[np.argsort(d[keep])[:int(max_pairs)]]
    return (keep.astype(np.int64), j[keep].astype(np.int64), d[keep].astype(np.float64))

def build_observed_point_factors(nodes: Sequence[PoseNode], pose_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]], anchor_frame: int, config: SolverConfig | None=None, outer_iteration: int=0) -> tuple[list[PointFactor], dict[str, Any]]:
    """Rebuild P09 correspondences in the candidate canonical frames.

    The returned factors retain observed world points, not generated mesh
    points.  Re-running this function for every outer pass is important: the
    nearest-neighbor association must follow the current SE(3) candidate.
    """
    cfg = config or SolverConfig()
    if cfg.observed_factor_mode == 'material_tracks':
        return ([], {'outer_iteration': int(outer_iteration), 'pair_count': 0, 'factor_count': 0, 'reason': 'NN replaced by fixed RGB-D material correspondences', 'generated_factors_included': False})
    positions = {node.frame_idx: i for (i, node) in enumerate(nodes)}
    direct = [node for node in nodes if node.metric_observation and node.observed_world is not None and (node.frame_idx in pose_by_frame)]
    if len(direct) < 2:
        return ([], {'outer_iteration': int(outer_iteration), 'pair_count': 0, 'factor_count': 0, 'reason': 'fewer_than_two_P09_metric_rows'})
    canonical: dict[int, np.ndarray] = {}
    normals: dict[int, np.ndarray] = {}
    sampled_world: dict[int, np.ndarray] = {}
    for node in direct:
        (R, t) = pose_by_frame[node.frame_idx]
        points = np.asarray(node.observed_world, dtype=np.float64)
        if len(points) > int(cfg.max_points):
            rng = np.random.default_rng(int(cfg.seed) + positions[node.frame_idx] + 31 * int(outer_iteration))
            points = points[rng.choice(len(points), int(cfg.max_points), replace=False)]
        sampled_world[node.frame_idx] = points
        canonical[node.frame_idx] = inverse_pose(points, R, t)
        normals[node.frame_idx] = estimate_normals(canonical[node.frame_idx])
    pairs: list[tuple[int, int, str]] = []
    direct_ids = [node.frame_idx for node in nodes if node.frame_idx in canonical]
    pairs.extend(((a, b, 'temporal_observed') for (a, b) in zip(direct_ids[:-1], direct_ids[1:])))
    if int(anchor_frame) in canonical:
        pairs.extend(((int(anchor_frame), idx, 'anchor_observed_loop') for idx in direct_ids if idx != int(anchor_frame)))
    factors: list[PointFactor] = []
    pair_diagnostics: list[dict[str, Any]] = []
    for (source_id, target_id, kind) in pairs:
        source_c = canonical[source_id]
        target_c = canonical[target_id]
        (si, ti, distance) = _mutual_pairs(source_c, target_c, float(cfg.max_correspondence_m), int(cfg.max_points))
        if len(si) < int(cfg.min_pairs):
            pair_diagnostics.append({'source_frame_idx': source_id, 'target_frame_idx': target_id, 'kind': kind, 'pair_count': int(len(si)), 'accepted': False, 'reason': 'too_few_rebuilt_correspondences'})
            continue
        quality = np.exp(-distance / max(float(cfg.max_correspondence_m), 1e-06))
        factors.append(PointFactor(positions[source_id], positions[target_id], np.asarray(sampled_world[source_id], dtype=np.float64)[si], np.asarray(sampled_world[target_id], dtype=np.float64)[ti], normals[target_id][ti], np.clip(quality, 0.05, 1.0), kind))
        pair_diagnostics.append({'source_frame_idx': source_id, 'target_frame_idx': target_id, 'kind': kind, 'pair_count': int(len(si)), 'median_initial_canonical_distance_m': float(np.median(distance)), 'accepted': True})
    return (factors, {'outer_iteration': int(outer_iteration), 'pair_count': len(pair_diagnostics), 'factor_count': len(factors), 'pairs': pair_diagnostics, 'rebuilt_from': 'P09 world_vertices_sample_m or camera_vertices_sample_m', 'generated_factors_included': False})

def make_scene(vertices, faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)), o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)))
    return scene

def closest(scene, points):
    import open3d as o3d
    shape = points.shape
    out = scene.compute_closest_points(o3d.core.Tensor(np.asarray(points, dtype=np.float32).reshape(-1, 3)), nthreads=4)
    q = out['points'].numpy().reshape(shape)
    n = out['primitive_normals'].numpy().reshape(shape)
    return (q, n)

def active_constraints_from_vertices(vertices_world: np.ndarray, row: FrameHandRow, scene: Any, max_constraints: int, eps: float, reference_vertices_world: np.ndarray | None=None, object_translation_delta_world: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from typing import Any
    import numpy as np
    import open3d as o3d
    from utils.geometry import inverse_object
    from utils.geometry import object_vec_to_world
    from utils.data import FrameHandRow
    obj_delta = np.zeros(3, dtype=float) if object_translation_delta_world is None else np.asarray(object_translation_delta_world, dtype=float)
    vertices_object = inverse_object(vertices_world, row.object_rotation_world_from_object, row.object_translation_world_m + obj_delta)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    hard_eps = float(eps) + max(0.0, float(row.observed_surface_support_uncertainty_m))
    penetrating = np.where(signed > hard_eps)[0]
    if penetrating.size == 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float))
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive_ids = closest['primitive_ids'].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(row.face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = row.face_strict_observed[primitive_ids[valid]]
    idx = penetrating[observed]
    if idx.size == 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float))
    closest_obj = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[idx], dtype=np.float32)))['points'].numpy().astype(float)
    disp = closest_obj - vertices_object[idx]
    norms = np.linalg.norm(disp, axis=1)
    good = norms > 1e-12
    idx = idx[good]
    if idx.size == 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float))
    normals_world = object_vec_to_world(disp[good] / norms[good, None], row.object_rotation_world_from_object)
    depths = signed[idx]
    reference = row.current_vertices_world if reference_vertices_world is None else reference_vertices_world
    reference_to_query = vertices_world[idx] - reference[idx] - obj_delta[None, :]
    required = np.sum(normals_world * reference_to_query, axis=1) + depths
    order = np.argsort(required)[::-1]
    if len(order) > int(max_constraints):
        order = order[:int(max_constraints)]
    return (idx[order].astype(np.int64), normals_world[order].astype(float), required[order].astype(float))

def merge_constraints(base_idx: np.ndarray, base_normals: np.ndarray, base_depths: np.ndarray, new_idx: np.ndarray, new_normals: np.ndarray, new_depths: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np
    best: dict[int, tuple[np.ndarray, float]] = {}
    for (idx, normal, depth) in zip(base_idx.astype(int), base_normals, base_depths.astype(float)):
        best[int(idx)] = (np.asarray(normal, dtype=float), float(depth))
    for (idx, normal, depth) in zip(new_idx.astype(int), new_normals, new_depths.astype(float)):
        old = best.get(int(idx))
        if old is None or float(depth) > old[1]:
            best[int(idx)] = (np.asarray(normal, dtype=float), float(depth))
    if not best:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float))
    items = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[:int(cap)]
    idx = np.asarray([k for (k, _) in items], dtype=np.int64)
    normals = np.stack([v[0] for (_, v) in items]).astype(float)
    depths = np.asarray([v[1] for (_, v) in items], dtype=float)
    return (idx, normals, depths)

def dense_observed_surface_constraints_from_vertices(vertices_world: np.ndarray, row: FrameHandRow, scene: Any, reference_vertices_world: np.ndarray | None=None, object_translation_delta_world: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearized observed-surface barrier for every MANO vertex near a trusted face.

    The active set only constrains vertices already known to penetrate.  This
    dense barrier adds tangent-plane inequalities for all vertices whose nearest
    object face is depth-observed.  A vertex currently outside the object gets a
    negative required displacement, so it contributes zero loss unless the
    optimizer moves it inward across that observed surface.  A vertex currently
    inside gets a positive required displacement along the outward normal.
    """
    from typing import Any
    import numpy as np
    import open3d as o3d
    from utils.geometry import inverse_object
    from utils.geometry import object_vec_to_world
    from utils.data import FrameHandRow
    obj_delta = np.zeros(3, dtype=float) if object_translation_delta_world is None else np.asarray(object_translation_delta_world, dtype=float)
    vertices_object = inverse_object(vertices_world, row.object_rotation_world_from_object, row.object_translation_world_m + obj_delta)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32)))
    primitive_ids = closest['primitive_ids'].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(row.face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = row.face_strict_observed[primitive_ids[valid]]
    idx = np.where(observed)[0].astype(np.int64)
    if idx.size == 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float))
    closest_obj = closest['points'].numpy().astype(float)[idx]
    disp = closest_obj - vertices_object[idx]
    norms = np.linalg.norm(disp, axis=1)
    good = norms > 1e-12
    idx = idx[good]
    if idx.size == 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float))
    signed_idx = signed[idx]
    normal_obj = disp[good] / norms[good, None]
    outside = signed_idx < 0.0
    normal_obj[outside] *= -1.0
    normals_world = object_vec_to_world(normal_obj, row.object_rotation_world_from_object)
    reference = row.current_vertices_world if reference_vertices_world is None else reference_vertices_world
    reference_to_query = vertices_world[idx] - reference[idx] - obj_delta[None, :]
    required = np.sum(normals_world * reference_to_query, axis=1) + signed_idx
    return (idx.astype(np.int64), normals_world.astype(float), required.astype(float))

def contact_patch_targets_from_vertices(vertices_world: np.ndarray, scene: Any, object_rotation_world_from_object: np.ndarray, object_translation_world_m: np.ndarray, face_strict_observed: np.ndarray, face_normals_object: np.ndarray, *, max_vertices: int, band_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from typing import Any
    import numpy as np
    import open3d as o3d
    from utils.geometry import inverse_object
    from utils.geometry import object_vec_to_world
    if int(max_vertices) <= 0 or float(band_m) <= 0.0 or (not np.any(face_strict_observed)):
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float))
    vertices_object = inverse_object(vertices_world, object_rotation_world_from_object, object_translation_world_m)
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32)))
    primitive_ids = closest['primitive_ids'].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = np.asarray(face_strict_observed, dtype=bool)[primitive_ids[valid]]
    closest_obj = closest['points'].numpy().astype(float)
    closest_world = closest_obj @ np.asarray(object_rotation_world_from_object, dtype=float).T + np.asarray(object_translation_world_m, dtype=float)[None, :]
    distances = np.linalg.norm(np.asarray(vertices_world, dtype=float) - closest_world, axis=1)
    candidate = observed & np.isfinite(distances) & (distances <= float(band_m))
    idx = np.where(candidate)[0].astype(np.int64)
    if idx.size == 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float))
    order = np.argsort(distances[idx])[:int(max_vertices)]
    idx = idx[order]
    primitive = primitive_ids[idx]
    normals_obj = np.asarray(face_normals_object, dtype=float)[primitive]
    normal_norms = np.linalg.norm(normals_obj, axis=1)
    good = normal_norms > 1e-09
    idx = idx[good]
    primitive = primitive[good]
    normals_obj = normals_obj[good]
    normal_norms = normal_norms[good]
    if idx.size == 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float))
    normals_obj = normals_obj / normal_norms[:, None]
    normals = object_vec_to_world(normals_obj, object_rotation_world_from_object)
    return (idx.astype(np.int64), closest_world[idx].astype(float), normals.astype(float), distances[idx].astype(float))

def correspondence_state(source: np.ndarray, target: np.ndarray, rotation: np.ndarray, translation: np.ndarray, *, frame_gap: int, trim_fraction: float, max_correspondence_m: float, min_matches: int) -> dict[str, Any]:
    from typing import Any
    import numpy as np
    from scipy.spatial import cKDTree
    import math
    from utils.geometry import apply_pose
    transformed = apply_pose(source, rotation, translation)
    target_tree = cKDTree(target)
    (distances, target_indices) = target_tree.query(transformed, k=1, workers=-1)
    source_tree = cKDTree(transformed)
    reverse_indices = source_tree.query(target, k=1, workers=-1)[1]
    source_indices = np.arange(len(source), dtype=np.int64)
    mutual = source_indices == reverse_indices[target_indices]
    pool = np.flatnonzero(mutual)
    if len(pool) < min_matches:
        pool = source_indices
    quantile_cap = float(np.percentile(distances[pool], trim_fraction * 100.0))
    metric_cap = float(max_correspondence_m * math.sqrt(max(1, frame_gap)))
    active_cap = min(metric_cap, quantile_cap)
    keep = pool[distances[pool] <= active_cap]
    if len(keep) < min_matches:
        keep = np.argsort(distances)[:min(len(distances), max(min_matches, int(len(source) * 0.35)))]
    return {'transformed': transformed, 'distances': distances, 'target_indices': target_indices, 'mutual_count': int(np.count_nonzero(mutual)), 'keep': np.asarray(keep, dtype=np.int64), 'active_cap_m': float(active_cap), 'metric_cap_m': metric_cap, 'quantile_cap_m': quantile_cap}

def first_hit_z(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, shape: tuple[int, int], near_m: float=1e-06) -> tuple[np.ndarray, dict[str, int]]:
    """Rasterize camera-z using perspective-correct screen barycentrics.

    Triangles crossing the near plane are clipped in camera coordinates.  A
    nonfinite face is diagnosed and skipped (therefore cannot erase appearance),
    while malformed array/topology/camera contracts are mechanical failures.
    """
    import numpy as np
    vertices = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces)
    K = np.asarray(K, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise ValueError('vertices must have shape [V,3]')
    if faces_array.ndim != 2 or faces_array.shape[1:] != (3,) or (not len(faces_array)) or (faces_array.dtype.kind not in 'iu') or (int(faces_array.min()) < 0) or (int(faces_array.max()) >= len(vertices)):
        raise ValueError('malformed triangle topology')
    if K.shape != (3, 3) or not np.isfinite(K).all() or (not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-12)) or (float(K[0, 0]) <= 0) or (float(K[1, 1]) <= 0) or (abs(float(K[1, 0])) > 1e-12):
        raise ValueError('invalid pinhole camera K')
    if not isinstance(shape, tuple) or len(shape) != 2 or any((isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)) or (not np.isfinite(near_m)) or (near_m <= 0):
        raise ValueError('invalid raster shape or near plane')
    faces = faces_array.astype(np.int64, copy=False)
    zbuf = np.full(shape, np.inf, dtype=np.float64)
    report = {'faces_total': int(len(faces)), 'nonfinite_faces': 0, 'behind_near_faces': 0, 'clipped_faces': 0, 'degenerate_faces': 0, 'offscreen_faces': 0}
    (height, width) = shape
    for face in faces:
        triangle = vertices[face]
        if not np.isfinite(triangle).all():
            report['nonfinite_faces'] += 1
            continue
        if np.all(triangle[:, 2] < near_m):
            report['behind_near_faces'] += 1
            continue
        polygon = list(triangle)
        if np.any(triangle[:, 2] < near_m):
            report['clipped_faces'] += 1
            polygon = []
            for (start, end) in zip(triangle, np.roll(triangle, -1, axis=0)):
                (start_in, end_in) = (start[2] >= near_m, end[2] >= near_m)
                if start_in:
                    polygon.append(start)
                if start_in != end_in:
                    alpha = (near_m - start[2]) / (end[2] - start[2])
                    polygon.append(start + alpha * (end - start))
        for fan_index in range(1, len(polygon) - 1):
            tri = np.asarray([polygon[0], polygon[fan_index], polygon[fan_index + 1]])
            homogeneous = tri @ K.T
            uv = homogeneous[:, :2] / homogeneous[:, 2:3]
            lower = np.maximum(np.floor(uv.min(axis=0)), [0, 0]).astype(int)
            upper = np.minimum(np.ceil(uv.max(axis=0)), [width - 1, height - 1]).astype(int)
            if np.any(lower > upper):
                report['offscreen_faces'] += 1
                continue
            (x0, y0) = lower
            (x1, y1) = upper
            (a, b, c) = uv
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(float(denominator)) < 1e-12:
                report['degenerate_faces'] += 1
                continue
            (yy, xx) = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
            wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
            wc = 1.0 - wa - wb
            inside = (wa >= -1e-10) & (wb >= -1e-10) & (wc >= -1e-10)
            inverse_z = wa / tri[0, 2] + wb / tri[1, 2] + wc / tri[2, 2]
            depth = np.divide(1.0, inverse_z, out=np.full_like(inverse_z, np.inf), where=inverse_z > 0)
            view = zbuf[y0:y1 + 1, x0:x1 + 1]
            np.minimum(view, np.where(inside, depth, np.inf), out=view)
    report['hit_pixels'] = int(np.isfinite(zbuf).sum())
    return (zbuf, report)

def nearest_hand_first_hits(hands: list[tuple[np.ndarray, np.ndarray]], K: np.ndarray, shape: tuple[int, int], near_m: float=1e-06) -> tuple[np.ndarray, list[dict[str, int]]]:
    import numpy as np
    nearest = np.full(shape, np.inf, dtype=np.float64)
    reports = []
    for (vertices, faces) in hands:
        (hand_z, report) = first_hit_z(vertices, faces, K, shape, near_m)
        np.minimum(nearest, hand_z, out=nearest)
        reports.append(report)
    return (nearest, reports)
