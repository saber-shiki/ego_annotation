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
