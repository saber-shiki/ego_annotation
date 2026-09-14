"""Candidate-dependent observed correspondences, refreshed once per outer pass."""
from __future__ import annotations
from typing import Any, Sequence, Mapping
import numpy as np
from scipy.spatial import cKDTree
from utils.data import PoseNode, PointFactor, SolverConfig
from utils.geometry import _array, inverse_pose

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

def make_scene(vertices, faces):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)),
    )
    return scene


def closest(scene, points):
    import open3d as o3d

    shape = points.shape
    out = scene.compute_closest_points(
        o3d.core.Tensor(np.asarray(points, dtype=np.float32).reshape(-1, 3)), nthreads=4
    )
    q = out["points"].numpy().reshape(shape)
    n = out["primitive_normals"].numpy().reshape(shape)
    return q, n

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
