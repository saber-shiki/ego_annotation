#!/usr/bin/env python3
"""V20 observed-only keyframe/global SE(3) graph with refreshed 3D factors.

This experimental solver starts from formal P14 poses but optimizes absolute
per-frame SE(3) corrections in several outer passes.  Each pass rebuilds
observed-anchor and keyframe correspondences, uses point-to-plane residuals,
relative SE(3) factors estimated from those edges, optional RGB/PnP relative
factors, and temporal motion priors.  It is intentionally diagnostic-only.
Generated SAM3D faces are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "fit_to_object_owned_rgb_calibrated_pnp",
}


@dataclass
class Node:
    frame_idx: int
    source_row: dict[str, Any]
    base_rotation: np.ndarray
    base_translation: np.ndarray
    observed_world: np.ndarray
    depth_quality: float = 1.0
    keyframe: bool = False


@dataclass
class PointFactor:
    source_pos: int
    target_pos: int
    # Observations stay fixed in the metric world frame.  During each residual
    # evaluation they are pulled back through the candidate object poses and
    # compared in the shared canonical frame.  Storing canonical points frozen
    # under the outer-loop pose and then pushing them back to two different
    # world poses would incorrectly penalize the object's real inter-frame
    # motion.
    source_world: np.ndarray
    target_world: np.ndarray
    target_normals_canonical: np.ndarray
    weights: np.ndarray
    kind: str


@dataclass
class RelativeFactor:
    source_pos: int
    target_pos: int
    source_frame_idx: int
    target_frame_idx: int
    rotation_source_to_target_canonical: np.ndarray
    translation_source_to_target_canonical: np.ndarray
    weight: float
    kind: str


@dataclass
class ImageFactor:
    frame_idx: int
    T_world_camera: np.ndarray
    K_raster: np.ndarray
    depth_points_canonical: np.ndarray
    depth_observed_z: np.ndarray
    depth_weight: np.ndarray
    silhouette_points_canonical: np.ndarray
    silhouette_target_uv: np.ndarray
    silhouette_kind: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T + np.asarray(translation, dtype=np.float64)[None, :]


def inverse_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - np.asarray(translation, dtype=np.float64)[None, :]) @ np.asarray(rotation, dtype=np.float64)


def numeric_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {"count": int(len(a)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)), "max": float(np.max(a)), "mean": float(np.mean(a))}


def deterministic_sample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= int(count):
        return points.copy()
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(count), replace=False)]


def estimate_normals(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    kk = min(max(3, int(k)), max(3, len(points)))
    _d, indices = cKDTree(points).query(points, k=kk, workers=-1)
    neighbors = points[indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(1, kk - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    normals = eigenvectors[:, :, 0]
    quality = eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1.0e-12)
    return normals, quality


def mutual_indices(source: np.ndarray, target: np.ndarray, trim_fraction: float, max_distance: float, max_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) == 0 or len(target) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    d, j = cKDTree(target).query(source, k=1, workers=-1)
    reverse = cKDTree(source).query(target, k=1, workers=-1)[1]
    si = np.arange(len(source), dtype=np.int64)
    pool = np.flatnonzero(reverse[j] == si)
    if len(pool) < 30:
        pool = np.argsort(d)[:min(len(d), max(30, int(max_pairs)))]
    if len(pool) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    cap = min(float(max_distance), float(np.percentile(d[pool], 100.0 * float(trim_fraction))))
    keep = pool[d[pool] <= cap]
    if len(keep) < 30:
        keep = np.argsort(d)[:min(len(d), max(30, int(max_pairs)))]
    if len(keep) > int(max_pairs):
        keep = keep[np.argsort(d[keep])[:int(max_pairs)]]
    return keep.astype(np.int64), j[keep].astype(np.int64), d[keep].astype(np.float64)


def load_nodes(args: argparse.Namespace, annotations: dict[str, Any], pose_report: dict[str, Any]) -> tuple[list[Node], dict[int, dict[str, Any]]]:
    frames = {int(f["frame_idx"]): f for f in annotations.get("frames", []) if isinstance(f, dict) and f.get("frame_idx") is not None}
    rows = {int(r["frame_idx"]): r for r in pose_report.get("pose_rows", []) if isinstance(r, dict) and str(r.get("status") or "") in POSE_STATUSES}
    nodes: list[Node] = []
    for idx in sorted(rows):
        if args.frame_start is not None and idx < int(args.frame_start):
            continue
        if args.frame_end is not None and idx > int(args.frame_end):
            continue
        frame = frames.get(idx)
        if frame is None:
            continue
        obj = next((o for o in frame.get("objects", []) if o.get("object_id") == args.object_id), None)
        geom = obj.get("visible_geometry_candidate") if isinstance(obj, dict) and isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        if obj is None or obj.get("rigid_pose_observation_eligible") is False or geom.get("rigid_pose_observation_eligible") is False:
            continue
        points = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < int(args.min_points) or not np.isfinite(points).all():
            continue
        R = np.asarray(rows[idx].get("rotation_world_from_completed_canonical_matrix"), dtype=np.float64)
        t = np.asarray(rows[idx].get("translation_world_m"), dtype=np.float64)
        if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all() or abs(np.linalg.det(R) - 1.0) > 1.0e-4:
            raise RuntimeError(f"invalid initial P14 pose at frame {idx}")
        ownership = geom.get("first_surface_depth_ownership") if isinstance(geom.get("first_surface_depth_ownership"), dict) else {}
        if ownership.get("enabled") is not True or ownership.get("fail_closed") is True:
            raise RuntimeError(f"frame {idx} lacks validated first-surface ownership")
        removed = float(ownership.get("removed_fraction") or 0.0)
        quality = float(np.clip(1.0 - removed / max(float(args.removed_fraction_full_weight), 1.0e-6), float(args.min_depth_quality), 1.0))
        nodes.append(Node(idx, rows[idx], R, t, deterministic_sample(points, int(args.max_points), int(args.seed) + idx), quality))
    if len(nodes) < int(args.min_frames):
        raise RuntimeError(f"only {len(nodes)} accepted direct nodes, need {args.min_frames}")
    return nodes, frames


def keyframe_positions(nodes: list[Node], anchor_frame: int, interval: int, max_keyframes: int) -> list[int]:
    direct = [n.frame_idx for n in nodes]
    keys = [i for i in direct if i == anchor_frame or (i - anchor_frame) % int(interval) == 0]
    if direct[-1] not in keys:
        keys.append(direct[-1])
    keys = sorted(set(keys))
    if len(keys) > int(max_keyframes):
        # Preserve the anchor and endpoint, uniformly subsample the interior.
        interior = keys[1:-1]
        take = max(0, int(max_keyframes) - 2)
        selected = [interior[int(round(i * (len(interior) - 1) / max(1, take - 1)))] for i in range(take)] if take and interior else []
        keys = sorted(set([keys[0], *selected, keys[-1]]))
    return keys


def build_dynamic_point_factors(nodes: list[Node], key_positions: list[int], anchor_position: int, args: argparse.Namespace, outer_index: int) -> tuple[list[PointFactor], dict[str, Any]]:
    canonical: list[np.ndarray] = [inverse_pose(n.observed_world, n.base_rotation, n.base_translation) for n in nodes]
    normals: list[np.ndarray] = []
    normal_quality: list[np.ndarray] = []
    for c in canonical:
        n, q = estimate_normals(c, int(args.normal_k))
        normals.append(n)
        normal_quality.append(q)
    anchor_c = canonical[anchor_position]
    anchor_normals = normals[anchor_position]
    factors: list[PointFactor] = []
    anchor_count = 0
    local_count = 0
    node_key_set = set(key_positions)
    for target_pos, target in enumerate(nodes):
        # All frames get a weak anchor loop; keyframes get stronger support.
        src_idx, dst_idx, distances = mutual_indices(anchor_c, canonical[target_pos], float(args.trim_fraction), float(args.max_correspondence_m), int(args.max_anchor_pairs))
        if len(src_idx) < int(args.min_pairs) or float(args.anchor_point_factor_scale) <= 0.0:
            continue
        source_c = anchor_c[src_idx]
        target_c = canonical[target_pos][dst_idx]
        q = normal_quality[target_pos][dst_idx]
        weights = np.clip(target.depth_quality * (float(args.keyframe_anchor_weight) if target.frame_idx in node_key_set else float(args.non_keyframe_anchor_weight)) * float(args.anchor_point_factor_scale) * np.clip(q / max(float(args.normal_quality_scale), 1.0e-6), 0.1, 1.0) * np.exp(-distances / max(float(args.point_weight_scale_m), 1.0e-6)), 0.05, 1.0)
        factors.append(
            PointFactor(
                anchor_position,
                target_pos,
                nodes[anchor_position].observed_world[src_idx],
                nodes[target_pos].observed_world[dst_idx],
                normals[target_pos][dst_idx],
                weights,
                "anchor_loop",
            )
        )
        anchor_count += len(source_c)
    # Consecutive keyframe edges are rebuilt at every outer pass.
    for source_pos, target_pos in zip(key_positions[:-1], key_positions[1:]):
        src_idx, dst_idx, distances = mutual_indices(canonical[source_pos], canonical[target_pos], float(args.trim_fraction), float(args.max_correspondence_m) * math.sqrt(max(1, nodes[target_pos].frame_idx - nodes[source_pos].frame_idx)), int(args.max_edge_pairs))
        if len(src_idx) < int(args.min_pairs) or float(args.local_point_factor_scale) <= 0.0:
            continue
        q = normal_quality[target_pos][dst_idx]
        edge_q = min(nodes[source_pos].depth_quality, nodes[target_pos].depth_quality)
        weights = np.clip(edge_q * float(args.local_edge_weight) * float(args.local_point_factor_scale) * np.clip(q / max(float(args.normal_quality_scale), 1.0e-6), 0.1, 1.0) * np.exp(-distances / max(float(args.point_weight_scale_m), 1.0e-6)), 0.05, 1.0)
        factors.append(
            PointFactor(
                source_pos,
                target_pos,
                nodes[source_pos].observed_world[src_idx],
                nodes[target_pos].observed_world[dst_idx],
                normals[target_pos][dst_idx],
                weights,
                "keyframe_local",
            )
        )
        local_count += len(src_idx)
    return factors, {"outer_iteration": int(outer_index), "factor_group_count": len(factors), "anchor_point_count": int(anchor_count), "local_point_count": int(local_count), "normal_quality_median": float(np.median(np.concatenate(normal_quality))) if normal_quality else None}


def load_relative_factors(args: argparse.Namespace, nodes: list[Node], keyframe_json: Path, keyframe_npz: Path, rgb_npz: Path | None) -> tuple[list[RelativeFactor], list[RelativeFactor]]:
    pos = {n.frame_idx: i for i, n in enumerate(nodes)}
    key_report = load_json(keyframe_json)
    with np.load(keyframe_npz, allow_pickle=False) as data:
        src = np.asarray(data["source_frame_idx"], dtype=np.int64)
        dst = np.asarray(data["target_frame_idx"], dtype=np.int64)
        rotations = np.asarray(data["rotation_correction"], dtype=np.float64)
        translations = np.asarray(data["translation_correction_m"], dtype=np.float64)
        accepted = np.asarray(data["accepted"], dtype=bool)
    report_rows = {(int(r["source_frame_idx"]), int(r["target_frame_idx"])): r for r in key_report.get("rows", [])}
    key_factors: list[RelativeFactor] = []
    for i in range(len(src)):
        if not accepted[i] or int(src[i]) not in pos or int(dst[i]) not in pos:
            continue
        row = report_rows.get((int(src[i]), int(dst[i])), {})
        eig = np.asarray(row.get("rotation_information_eigenvalues") or [0.0], dtype=np.float64)
        cond = float(row.get("rotation_information_condition") or 1.0e9)
        plane = float((row.get("point_to_plane_abs_m") or {}).get("median") or 0.01)
        scale = float(args.keyframe_relative_weight_scale)
        if scale <= 0.0:
            continue
        quality = float(np.clip(math.exp(-plane / max(float(args.edge_residual_scale_m), 1.0e-6)) * np.clip(np.min(eig) / max(float(args.rotation_information_scale), 1.0e-9), 0.1, 1.0) * np.clip(float(args.max_rotation_condition) / max(cond, 1.0), 0.1, 1.0) * scale, float(args.min_relative_weight), 1.0))
        key_factors.append(RelativeFactor(pos[int(src[i])], pos[int(dst[i])], int(src[i]), int(dst[i]), rotations[i], translations[i], quality, str(row.get("kind") or "keyframe")))
    rgb_factors: list[RelativeFactor] = []
    if rgb_npz is not None:
        with np.load(rgb_npz, allow_pickle=False) as data:
            required = {"source_frame_idx", "target_frame_idx", "rotation_rgb", "translation_rgb_m", "quality_weight", "accepted"}
            if not required.issubset(data.files):
                raise RuntimeError(f"RGB factor NPZ missing {sorted(required - set(data.files))}")
            rs = np.asarray(data["source_frame_idx"], dtype=np.int64); rt = np.asarray(data["target_frame_idx"], dtype=np.int64); rr = np.asarray(data["rotation_rgb"], dtype=np.float64); tt = np.asarray(data["translation_rgb_m"], dtype=np.float64); ww = np.asarray(data["quality_weight"], dtype=np.float64); ok = np.asarray(data["accepted"], dtype=bool)
        scale = float(args.rgb_relative_weight_scale)
        if scale > 0.0:
            for i in range(len(rs)):
                if not (ok[i] and int(rs[i]) in pos and int(rt[i]) in pos):
                    continue
                weight = float(np.clip(ww[i] * scale, float(args.min_rgb_weight), 1.0))
                rgb_factors.append(RelativeFactor(pos[int(rs[i])], pos[int(rt[i])], int(rs[i]), int(rt[i]), rr[i], tt[i], weight, "rgb_pnp"))

    return key_factors, rgb_factors



def load_image_factors(path: Path | None, frame_ids: set[int], args: argparse.Namespace | None = None) -> dict[int, ImageFactor]:
    if path is None:
        return {}
    with np.load(path.expanduser().resolve(), allow_pickle=False) as data:
        metadata_raw = data["metadata"][0] if "metadata" in data.files else "{}"
        try:
            metadata = json.loads(str(metadata_raw))
        except json.JSONDecodeError as exc:
            raise RuntimeError("image factor metadata is not valid JSON") from exc
        if args is not None:
            if int(metadata.get("contract_version") or 0) < 2:
                raise RuntimeError("strict v20 image factors require contract_version >= 2")
            expected_paths = {
                "annotations": args.annotations,
                "pose_report": args.pose_report,
                "factor_pose_report": args.pose_report,
                "completed_mesh": args.pose_mesh,
            }
            for key, expected_path in expected_paths.items():
                declared = str(metadata.get(key) or "")
                if declared != str(expected_path.expanduser().resolve()):
                    raise RuntimeError(
                        f"image factor contract mismatch for {key}: {declared} != {expected_path.expanduser().resolve()}"
                    )
                declared_hash = str((metadata.get("input_sha256") or {}).get(key) or "")
                if declared_hash != sha256_file(expected_path):
                    raise RuntimeError(f"image factor hash mismatch for {key}")
            if metadata.get("generated_faces_consumed") is not False or metadata.get("collision_surface_consumed") is not False:
                raise RuntimeError("v20 image factors cannot consume generated/collision geometry as pose authority")
            if str(metadata.get("object_id") or "") != str(args.object_id):
                raise RuntimeError("image factor object_id mismatch")
            for optional_key in ("hand_npz", "mano_faces_pkl"):
                optional_path = Path(str(metadata.get(optional_key) or "")).expanduser().resolve()
                optional_hash = str((metadata.get("input_sha256") or {}).get(optional_key) or "")
                if not optional_path.is_file() or not optional_hash or optional_hash != sha256_file(optional_path):
                    raise RuntimeError(f"image factor {optional_key} hash mismatch")
        required = {
            "frame_idx", "T_world_camera", "K_raster", "depth_offsets",
            "depth_points_canonical", "depth_observed_z", "silhouette_offsets",
            "silhouette_points_canonical", "silhouette_target_uv", "silhouette_kind",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"image factor NPZ missing keys: {missing}")
        ids = np.asarray(data["frame_idx"], dtype=np.int64)
        T = np.asarray(data["T_world_camera"], dtype=np.float64)
        K = np.asarray(data["K_raster"], dtype=np.float64)
        doff = np.asarray(data["depth_offsets"], dtype=np.int64)
        dpts = np.asarray(data["depth_points_canonical"], dtype=np.float64)
        dz = np.asarray(data["depth_observed_z"], dtype=np.float64)
        dw = np.asarray(data["depth_weight"], dtype=np.float64) if "depth_weight" in data.files else np.ones(len(dz), dtype=np.float64)
        soff = np.asarray(data["silhouette_offsets"], dtype=np.int64)
        spts = np.asarray(data["silhouette_points_canonical"], dtype=np.float64)
        suv = np.asarray(data["silhouette_target_uv"], dtype=np.float64)
        skind = np.asarray(data["silhouette_kind"], dtype=np.int8)
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise RuntimeError("image factor frame_idx must be unique and non-empty")
    if T.shape != (len(ids), 4, 4) or K.shape != (3, 3):
        raise RuntimeError("invalid image factor camera arrays")
    if doff.shape != (len(ids) + 1,) or soff.shape != (len(ids) + 1,):
        raise RuntimeError("invalid image factor offsets")
    if doff[-1] != len(dpts) or doff[-1] != len(dz) or doff[-1] != len(dw):
        raise RuntimeError("image depth arrays disagree with offsets")
    if soff[-1] != len(spts) or soff[-1] != len(suv) or soff[-1] != len(skind):
        raise RuntimeError("image silhouette arrays disagree with offsets")
    if not (np.isfinite(T).all() and np.isfinite(K).all() and np.isfinite(dpts).all() and np.isfinite(dz).all() and np.isfinite(dw).all() and np.isfinite(spts).all() and np.isfinite(suv).all()):
        raise RuntimeError("image factor arrays contain non-finite values")
    out: dict[int, ImageFactor] = {}
    for i, frame_idx in enumerate(ids.tolist()):
        frame_idx = int(frame_idx)
        if frame_idx not in frame_ids:
            continue
        d0, d1 = int(doff[i]), int(doff[i + 1])
        s0, s1 = int(soff[i]), int(soff[i + 1])
        out[frame_idx] = ImageFactor(
            frame_idx,
            T[i], K.copy(), dpts[d0:d1], dz[d0:d1], dw[d0:d1],
            spts[s0:s1], suv[s0:s1], skind[s0:s1],
        )
    return out


def image_factor_blocks(
    node: Node,
    factor: ImageFactor | None,
    rotation: np.ndarray,
    translation: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if factor is None:
        return None, None
    depth_block = None
    silhouette_block = None
    if len(factor.depth_observed_z) and float(args.image_first_hit_weight) > 0.0:
        world = apply_pose(factor.depth_points_canonical, rotation, translation)
        camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
        depth = np.clip(
            camera[:, 2] - factor.depth_observed_z,
            -float(args.max_image_first_hit_residual_m),
            float(args.max_image_first_hit_residual_m),
        ) / float(args.sigma_image_first_hit_m)
        depth_block = (
            math.sqrt(float(args.image_first_hit_weight))
            * np.sqrt(factor.depth_weight)
            * depth
            / math.sqrt(max(float(np.sum(factor.depth_weight)), 1.0))
        )
    if len(factor.silhouette_kind) and float(args.image_silhouette_weight) > 0.0:
        world = apply_pose(factor.silhouette_points_canonical, rotation, translation)
        camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
        z = np.maximum(camera[:, 2], 1.0e-9)
        uv = np.column_stack((
            factor.K_raster[0, 0] * camera[:, 0] / z + factor.K_raster[0, 2],
            factor.K_raster[1, 1] * camera[:, 1] / z + factor.K_raster[1, 2],
        ))
        diff = uv - factor.silhouette_target_uv
        norms = np.linalg.norm(diff, axis=1)
        clip_scale = np.minimum(
            1.0,
            float(args.max_image_silhouette_residual_px) / np.maximum(norms, 1.0e-9),
        )
        silhouette_block = (
            math.sqrt(float(args.image_silhouette_weight))
            * (diff * clip_scale[:, None]).reshape(-1)
            / float(args.sigma_image_silhouette_px)
            / math.sqrt(max(1, len(factor.silhouette_kind)))
        )
    return depth_block, silhouette_block
def unpack(x: np.ndarray, count: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(count, 6)


def current_poses(nodes: list[Node], x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    deltas = unpack(x, len(nodes)); Rs=[];ts=[]
    for node, d in zip(nodes, deltas):
        delta_rotation = Rotation.from_rotvec(d[:3]).as_matrix()
        # Proper left SE(3) perturbation in world coordinates.
        Rs.append(delta_rotation @ node.base_rotation)
        ts.append(delta_rotation @ node.base_translation + d[3:])
    return Rs, ts


def relative_residual(Rs: list[np.ndarray], ts: list[np.ndarray], factor: RelativeFactor) -> tuple[np.ndarray, np.ndarray]:
    if factor.kind == "rgb_absolute":
        rotation_error = Rotation.from_matrix(
            Rs[factor.target_pos] @ factor.rotation_source_to_target_canonical.T
        ).as_rotvec()
        return rotation_error, ts[factor.target_pos] - factor.translation_source_to_target_canonical
    if factor.kind == "rgb_pnp":
        predicted_rotation = Rs[factor.target_pos] @ Rs[factor.source_pos].T
        predicted_translation = ts[factor.target_pos] - predicted_rotation @ ts[factor.source_pos]
        rotation_error = Rotation.from_matrix(
            predicted_rotation @ factor.rotation_source_to_target_canonical.T
        ).as_rotvec()
        return rotation_error, predicted_translation - factor.translation_source_to_target_canonical
    # Keyframe surface registration C maps source canonical coordinates to target
    # canonical coordinates.  Compare T_source with T_target composed with C.
    target_equiv_rotation = Rs[factor.target_pos] @ factor.rotation_source_to_target_canonical
    target_equiv_translation = (
        Rs[factor.target_pos] @ factor.translation_source_to_target_canonical
        + ts[factor.target_pos]
    )
    rotation_error = Rotation.from_matrix(
        Rs[factor.source_pos] @ target_equiv_rotation.T
    ).as_rotvec()
    return rotation_error, ts[factor.source_pos] - target_equiv_translation


def residual_blocks(x: np.ndarray, nodes: list[Node], point_factors: list[PointFactor], key_factors: list[RelativeFactor], rgb_factors: list[RelativeFactor], args: argparse.Namespace, image_factors: dict[int, ImageFactor] | None = None) -> dict[str, list[np.ndarray]]:
    deltas = unpack(x, len(nodes)); Rs,ts=current_poses(nodes,x);blocks: dict[str,list[np.ndarray]]={"pose_prior":[],"image_first_hit":[],"image_silhouette":[],"point_to_plane":[],"point_to_point":[],"keyframe_relative_rotation":[],"keyframe_relative_translation":[],"rgb_relative_rotation":[],"rgb_relative_translation":[],"correction_velocity":[],"correction_acceleration":[],"motion_acceleration":[],"anchor_gauge":[]}
    for i,d in enumerate(deltas):
        blocks["pose_prior"].append(np.concatenate([d[3:] / float(args.sigma_pose_prior_translation_m), d[:3] / float(args.sigma_pose_prior_rotation_rad)]))
        if image_factors:
            depth_block, silhouette_block = image_factor_blocks(nodes[i], image_factors.get(nodes[i].frame_idx), Rs[i], ts[i], args)
            if depth_block is not None:
                blocks["image_first_hit"].append(depth_block)
            if silhouette_block is not None:
                blocks["image_silhouette"].append(silhouette_block)
    for factor in point_factors:
        source_canonical = inverse_pose(
            factor.source_world,
            Rs[factor.source_pos],
            ts[factor.source_pos],
        )
        target_canonical = inverse_pose(
            factor.target_world,
            Rs[factor.target_pos],
            ts[factor.target_pos],
        )
        diff = source_canonical - target_canonical
        plane = np.einsum("ij,ij->i", factor.target_normals_canonical, diff)
        scale = math.sqrt(max(1, len(plane)))
        blocks["point_to_plane"].append(
            math.sqrt(float(args.point_to_plane_weight))
            * np.sqrt(factor.weights)
            * np.clip(
                plane,
                -float(args.max_point_residual_m),
                float(args.max_point_residual_m),
            )
            / float(args.sigma_point_to_plane_m)
            / scale
        )
        blocks["point_to_point"].append(
            math.sqrt(float(args.point_to_point_weight))
            * np.sqrt(factor.weights[:, None])
            * np.clip(
                diff,
                -float(args.max_point_residual_m),
                float(args.max_point_residual_m),
            )
            / float(args.sigma_point_to_point_m)
            / scale
        )
    for factor in key_factors:
        r,t=relative_residual(Rs,ts,factor);blocks["keyframe_relative_rotation"].append(math.sqrt(factor.weight)*r/float(args.sigma_relative_rotation_rad));blocks["keyframe_relative_translation"].append(math.sqrt(factor.weight)*t/float(args.sigma_relative_translation_m))
    for factor in rgb_factors:
        r,t=relative_residual(Rs,ts,factor);blocks["rgb_relative_rotation"].append(math.sqrt(factor.weight)*r/float(args.sigma_rgb_rotation_rad));blocks["rgb_relative_translation"].append(math.sqrt(factor.weight)*t/float(args.sigma_rgb_translation_m))
    for i in range(1,len(nodes)):
        gap=max(1,nodes[i].frame_idx-nodes[i-1].frame_idx);blocks["correction_velocity"].append(np.concatenate([(deltas[i,3:]-deltas[i-1,3:])/(float(args.sigma_correction_translation_step_m)*math.sqrt(gap)),(deltas[i,:3]-deltas[i-1,:3])/(float(args.sigma_correction_rotation_step_rad)*math.sqrt(gap))]))
    for i in range(1,len(nodes)-1):
        g0=max(1,nodes[i].frame_idx-nodes[i-1].frame_idx);g1=max(1,nodes[i+1].frame_idx-nodes[i].frame_idx);vt=(deltas[i,3:]-deltas[i-1,3:])/g0-(deltas[i+1,3:]-deltas[i,3:])/g1;vr=(deltas[i,:3]-deltas[i-1,:3])/g0-(deltas[i+1,:3]-deltas[i,:3])/g1;blocks["correction_acceleration"].append(np.concatenate([vt/(float(args.sigma_correction_translation_accel_m)*math.sqrt(max(g0,g1))),vr/(float(args.sigma_correction_rotation_accel_rad)*math.sqrt(max(g0,g1)))]))
        # Soft acceleration prior on the actual absolute motion, not only the correction field.
        if i < len(nodes)-1:
            m0_t=(ts[i]-ts[i-1])/g0;m1_t=(ts[i+1]-ts[i])/g1;m0_r=Rotation.from_matrix(Rs[i]@Rs[i-1].T).as_rotvec()/g0;m1_r=Rotation.from_matrix(Rs[i+1]@Rs[i].T).as_rotvec()/g1;blocks["motion_acceleration"].append(np.concatenate([(m1_t-m0_t)/float(args.sigma_motion_translation_accel_m),(m1_r-m0_r)/float(args.sigma_motion_rotation_accel_rad)]))
    a=int(args.anchor_position);blocks["anchor_gauge"].append(np.concatenate([deltas[a,3:]/float(args.sigma_anchor_gauge_translation_m),deltas[a,:3]/float(args.sigma_anchor_gauge_rotation_rad)]))
    return blocks


def flatten_blocks(blocks: dict[str,list[np.ndarray]]) -> np.ndarray:
    parts=[v.reshape(-1) for values in blocks.values() for v in values]
    return np.concatenate(parts).astype(np.float64) if parts else np.empty(0,dtype=np.float64)


def residual_sparsity(
    nodes: list[Node],
    point_factors: list[PointFactor],
    key_factors: list[RelativeFactor],
    rgb_factors: list[RelativeFactor],
    args: argparse.Namespace,
    image_factors: dict[int, ImageFactor] | None = None,
) -> sparse.csr_matrix:
    """Match the row order emitted by ``flatten_blocks(residual_blocks(...))``."""
    entries: list[tuple[int, int]] = []
    row = 0

    def add(count: int, positions: list[int]) -> None:
        nonlocal row
        for rr in range(row, row + int(count)):
            for position in positions:
                entries.extend((rr, col) for col in range(6 * position, 6 * position + 6))
        row += int(count)

    # residual_blocks dictionary order: pose_prior, point-to-plane,
    # point-to-point, keyframe R/t, RGB R/t, correction velocity/acceleration,
    # actual motion acceleration, anchor gauge.
    add(6 * len(nodes), list(range(len(nodes))))
    if image_factors:
        for i, node in enumerate(nodes):
            factor = image_factors.get(node.frame_idx)
            if factor is not None and float(args.image_first_hit_weight) > 0.0:
                add(len(factor.depth_observed_z), [i])
        for i, node in enumerate(nodes):
            factor = image_factors.get(node.frame_idx)
            if factor is not None and float(args.image_silhouette_weight) > 0.0:
                add(2 * len(factor.silhouette_kind), [i])
    for factor in point_factors:
        add(len(factor.source_world), [factor.source_pos, factor.target_pos])
    for factor in point_factors:
        add(3 * len(factor.source_world), [factor.source_pos, factor.target_pos])
    for factor in key_factors:
        add(3, [factor.source_pos, factor.target_pos])
    for factor in key_factors:
        add(3, [factor.source_pos, factor.target_pos])
    for factor in rgb_factors:
        add(3, [factor.source_pos, factor.target_pos])
    for factor in rgb_factors:
        add(3, [factor.source_pos, factor.target_pos])
    for i in range(1, len(nodes)):
        add(6, [i - 1, i])
    for i in range(1, len(nodes) - 1):
        add(6, [i - 1, i, i + 1])
    for i in range(1, len(nodes) - 1):
        add(6, [i - 1, i, i + 1])
    add(6, [int(args.anchor_position)])
    if row == 0:
        raise RuntimeError("empty v20 pose graph sparsity")
    rr, cc = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(rr), dtype=bool), (rr, cc)), shape=(row, 6 * len(nodes)))

def mesh_surface_metrics_from_poses(
    nodes: list[Node],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    mesh_points: np.ndarray,
) -> dict[str, Any]:
    observed_to_mesh: list[float] = []
    mesh_to_observed: list[float] = []
    per_frame: dict[str, Any] = {}
    for node, rotation, translation in zip(nodes, rotations, translations):
        mesh_world = apply_pose(mesh_points, rotation, translation)
        observed_dist = cKDTree(mesh_world).query(node.observed_world, k=1, workers=-1)[0]
        mesh_dist = cKDTree(node.observed_world).query(mesh_world, k=1, workers=-1)[0]
        observed_summary = {
            "count": int(len(observed_dist)),
            "median_m": float(np.median(observed_dist)),
            "p90_m": float(np.percentile(observed_dist, 90)),
            "p95_m": float(np.percentile(observed_dist, 95)),
            "mean_m": float(np.mean(observed_dist)),
            "max_m": float(np.max(observed_dist)),
        }
        mesh_summary = {
            "count": int(len(mesh_dist)),
            "median_m": float(np.median(mesh_dist)),
            "p90_m": float(np.percentile(mesh_dist, 90)),
            "p95_m": float(np.percentile(mesh_dist, 95)),
            "mean_m": float(np.mean(mesh_dist)),
            "max_m": float(np.max(mesh_dist)),
        }
        observed_to_mesh.append(observed_summary["median_m"])
        mesh_to_observed.append(mesh_summary["median_m"])
        per_frame[str(node.frame_idx)] = {
            "observed_to_mesh": observed_summary,
            "mesh_to_observed": mesh_summary,
        }
    return {
        "observed_to_mesh_m": numeric_summary(observed_to_mesh),
        "mesh_to_observed_m": numeric_summary(mesh_to_observed),
        "per_frame": per_frame,
    }

def term_metrics(blocks: dict[str,list[np.ndarray]]) -> dict[str,Any]:
    out={}
    for key,values in blocks.items():
        flat=np.concatenate([v.reshape(-1) for v in values]) if values else np.empty(0)
        out[key]={"count":int(len(flat)),"rms":float(np.sqrt(np.mean(flat*flat))) if len(flat) else None,"median_abs":float(np.median(np.abs(flat))) if len(flat) else None}
    return out


def build_report(args: argparse.Namespace, initial_report: dict[str,Any], nodes: list[Node], keyframes: list[int], outer_reports: list[dict[str,Any]], final_x: np.ndarray, mesh_before: dict[str,Any], mesh_after: dict[str,Any], key_factors: list[RelativeFactor], rgb_factors: list[RelativeFactor], frames: dict[int,dict[str,Any]]) -> dict[str,Any]:
    Rs,ts=current_poses(nodes,final_x);pos={n.frame_idx:i for i,n in enumerate(nodes)};rows=[]
    for original in initial_report.get("pose_rows",[]):
        if not isinstance(original,dict):rows.append(original);continue
        idx=int(original.get("frame_idx",-1));row=dict(original)
        if idx in pos:
            i = pos[idx]
            n = nodes[i]
            row["rotation_world_from_completed_canonical_matrix"] = Rs[i].astype(float).tolist()
            row["translation_world_m"] = ts[i].astype(float).tolist()
            row["pose_source"] = "v20_keyframe_global_observed_only"
            row["direct_pose_observation_source"] = "v20_keyframe_point_to_plane_graph"
            row["generated_geometry_pose_evidence_consumed"] = False
            row["observed_to_mesh_initial"] = mesh_before.get("per_frame", {}).get(str(idx), {}).get("observed_to_mesh", {})
            row["observed_to_mesh_final"] = mesh_after.get("per_frame", {}).get(str(idx), {}).get("observed_to_mesh", {})
            row["mesh_to_observed_final"] = mesh_after.get("per_frame", {}).get(str(idx), {}).get("mesh_to_observed", {})
            row["v20_uncertainty"] = {"depth_quality": n.depth_quality, "keyframe": n.keyframe}
        rows.append(row)
    timeline=sorted(frames);direct=sorted(pos);steps_r=[];steps_t=[]
    for a,b in zip(direct[:-1],direct[1:]):
        i,j=pos[a],pos[b];gap=max(1,b-a);steps_r.append(float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rs[j]@Rs[i].T).as_rotvec()))/gap));steps_t.append(float(np.linalg.norm(ts[j]-ts[i])/gap))
    binding={k:initial_report.get(k) for k in ["selected_anchor_atomic_binding","selected_anchor_evidence_report","selected_anchor_evidence_report_sha256","anchor_frame_idx","anchor_centroid_world_m","anchor_observed_extent_m"]}
    return {"schema":"v20_keyframe_global_observed_pose_graph_v1","status":"v20_global_complete" if outer_reports and outer_reports[-1].get("optimizer_success") else "v20_global_incomplete","annotation_ready":False,"diagnostic_only":True,"claim_scope":"Observed-only keyframe/global SE(3) diagnostic. Absolute poses are optimized with refreshed point-to-plane anchor/local factors, relative edge factors, optional RGB/PnP and temporal priors. Generated SAM3D faces are not loaded or consumed.","object_id":args.object_id,"inputs":{"annotations":str(args.annotations.expanduser().resolve()),"initial_pose_report":str(args.pose_report.expanduser().resolve()),"pose_mesh":str(args.pose_mesh.expanduser().resolve()),"keyframe_edge_npz":str(args.keyframe_edge_npz.expanduser().resolve()),"keyframe_edge_report":str(args.keyframe_edge_json.expanduser().resolve()),"rgb_edge_npz":str(args.rgb_edge_npz.expanduser().resolve()) if args.rgb_edge_npz else None,"generated_geometry_consumed":False,"sha256":{"annotations":sha256_file(args.annotations),"initial_pose_report":sha256_file(args.pose_report),"pose_mesh":sha256_file(args.pose_mesh),"keyframe_edge_npz":sha256_file(args.keyframe_edge_npz),"keyframe_edge_report":sha256_file(args.keyframe_edge_json),"rgb_edge_npz":sha256_file(args.rgb_edge_npz) if args.rgb_edge_npz else None}},**binding,"keyframes":{"interval":int(args.keyframe_interval),"frame_ids":keyframes,"count":len(keyframes),"periodic_reanchor_outer_iterations":len(outer_reports),"recomputed_correspondences_each_outer_iteration":True},"factors":{"point_to_plane_refreshed":sum(int(r.get("point_factor_metrics",{}).get("anchor_point_count",0))+int(r.get("point_factor_metrics",{}).get("local_point_count",0)) for r in outer_reports),"relative_keyframe":len(key_factors),"relative_rgb_pnp":len(rgb_factors),"generated_geometry_consumed":False},"optimizer_outer_iterations":outer_reports,"final_optimizer":{"success":bool(outer_reports and outer_reports[-1].get("optimizer_success")),"outer_iteration_count":len(outer_reports)},"surface_before":mesh_before,"surface_after":mesh_after,"surface_median_degradation_m":(mesh_after["observed_to_mesh_m"]["median"]-mesh_before["observed_to_mesh_m"]["median"] if mesh_before["observed_to_mesh_m"]["median"] is not None and mesh_after["observed_to_mesh_m"]["median"] is not None else None),"temporal":{"direct_frame_count":len(direct),"timeline_frame_count":len(timeline),"direct_fraction":len(direct)/max(1,len(timeline)),"rotation_step_deg":numeric_summary(steps_r),"translation_step_m":numeric_summary(steps_t),"periodic_reanchor_is_batch_outer_loop":True},"pose_rows":rows,"outputs":{"pose_report":str(args.output_dir/"v20_keyframe_global_pose_report.json")}}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--pose-report',type=Path,required=True);p.add_argument('--pose-mesh',type=Path,required=True);p.add_argument('--keyframe-edge-npz',type=Path,required=True);p.add_argument('--keyframe-edge-json',type=Path,required=True);p.add_argument('--rgb-edge-npz',type=Path,default=None);p.add_argument('--object-id',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--frame-start',type=int,default=None);p.add_argument('--frame-end',type=int,default=None);p.add_argument('--anchor-frame',type=int,default=0);p.add_argument('--keyframe-interval',type=int,default=5);p.add_argument('--max-keyframes',type=int,default=40);p.add_argument('--outer-iterations',type=int,default=3);p.add_argument('--inner-max-nfev',type=int,default=20);p.add_argument('--max-points',type=int,default=600);p.add_argument('--max-mesh-points',type=int,default=2500);p.add_argument('--min-points',type=int,default=50);p.add_argument('--min-frames',type=int,default=8);p.add_argument('--normal-k',type=int,default=24);p.add_argument('--max-anchor-pairs',type=int,default=120);p.add_argument('--max-edge-pairs',type=int,default=120);p.add_argument('--min-pairs',type=int,default=30);p.add_argument('--trim-fraction',type=float,default=.70);p.add_argument('--max-correspondence-m',type=float,default=.045);p.add_argument('--max-anchor-correspondence-m',type=float,default=.055);p.add_argument('--max-edge-correspondence-m',type=float,default=.045);p.add_argument('--point-weight-scale-m',type=float,default=.012);p.add_argument('--normal-quality-scale',type=float,default=.25);p.add_argument('--keyframe-anchor-weight',type=float,default=1.0);p.add_argument('--non-keyframe-anchor-weight',type=float,default=.35);p.add_argument('--local-edge-weight',type=float,default=1.0);p.add_argument('--removed-fraction-full-weight',type=float,default=.10);p.add_argument('--min-depth-quality',type=float,default=.25);p.add_argument('--edge-residual-scale-m',type=float,default=.004);p.add_argument('--rotation-information-scale',type=float,default=.01);p.add_argument('--max-rotation-condition',type=float,default=1e8);p.add_argument('--min-relative-weight',type=float,default=.10);p.add_argument('--min-rgb-weight',type=float,default=.05);p.add_argument('--sigma-pose-prior-translation-m',type=float,default=.04);p.add_argument('--sigma-pose-prior-rotation-rad',type=float,default=.20);p.add_argument('--sigma-point-to-plane-m',type=float,default=.006);p.add_argument('--sigma-point-to-point-m',type=float,default=.015);p.add_argument('--point-to-point-weight',type=float,default=.20);p.add_argument('--max-point-residual-m',type=float,default=.05);p.add_argument('--sigma-relative-rotation-rad',type=float,default=.10);p.add_argument('--sigma-relative-translation-m',type=float,default=.018);p.add_argument('--sigma-rgb-rotation-rad',type=float,default=.08);p.add_argument('--sigma-rgb-translation-m',type=float,default=.015);p.add_argument('--sigma-correction-translation-step-m',type=float,default=.012);p.add_argument('--sigma-correction-rotation-step-rad',type=float,default=.10);p.add_argument('--sigma-correction-translation-accel-m',type=float,default=.008);p.add_argument('--sigma-correction-rotation-accel-rad',type=float,default=.06);p.add_argument('--sigma-motion-translation-accel-m',type=float,default=.04);p.add_argument('--sigma-motion-rotation-accel-rad',type=float,default=.20);p.add_argument('--sigma-anchor-gauge-translation-m',type=float,default=1e-6);p.add_argument('--sigma-anchor-gauge-rotation-rad',type=float,default=1e-6);p.add_argument('--max-correction-translation-m',type=float,default=.05);p.add_argument('--max-correction-rotation-rad',type=float,default=.25);p.add_argument('--min-outer-cost-improvement',type=float,default=1e-4);p.add_argument('--seed',type=int,default=20260904);args=p.parse_args()
    if args.keyframe_interval<1 or args.outer_iterations<1:raise RuntimeError('invalid keyframe/outer settings')
    annotations=load_json(args.annotations);initial=load_json(args.pose_report);nodes,frames=load_nodes(args,annotations,initial);anchor_frame=int(args.anchor_frame if args.anchor_frame is not None else initial.get('anchor_frame_idx',nodes[0].frame_idx));args.anchor_position=next((i for i,n in enumerate(nodes) if n.frame_idx==anchor_frame),None)
    if args.anchor_position is None:raise RuntimeError(f'anchor frame {anchor_frame} not in nodes')
    key_ids=keyframe_positions(nodes,anchor_frame,int(args.keyframe_interval),int(args.max_keyframes));node_pos={n.frame_idx:i for i,n in enumerate(nodes)};key_positions=[node_pos[i] for i in key_ids];
    for i in key_positions:nodes[i].keyframe=True
    key_factors,rgb_factors=load_relative_factors(args,nodes,args.keyframe_edge_json,args.keyframe_edge_npz,args.rgb_edge_npz)
    import trimesh
    mesh = trimesh.load(
        args.pose_mesh.expanduser().resolve(),
        force="mesh",
        process=False,
    )
    mesh_points, _ = trimesh.sample.sample_surface(
        mesh,
        min(int(args.max_mesh_points), max(1, len(mesh.faces) * 2)),
        seed=np.random.default_rng(int(args.seed) + 701),
    )
    mesh_points = np.asarray(mesh_points, dtype=np.float64)
    initial_rotations = [node.base_rotation.copy() for node in nodes]
    initial_translations = [node.base_translation.copy() for node in nodes]
    mesh_before = mesh_surface_metrics_from_poses(
        nodes, initial_rotations, initial_translations, mesh_points
    )

    x_current = np.zeros(len(nodes) * 6, dtype=np.float64)
    outer_reports: list[dict[str, Any]] = []
    for outer in range(int(args.outer_iterations)):
        point_factors, point_info = build_dynamic_point_factors(
            nodes, key_positions, int(args.anchor_position), args, outer
        )
        before_blocks = residual_blocks(
            x_current, nodes, point_factors, key_factors, rgb_factors, args
        )
        before = flatten_blocks(before_blocks)
        pattern = residual_sparsity(
            nodes, point_factors, key_factors, rgb_factors, args
        )
        if pattern.shape != (len(before), len(x_current)):
            raise RuntimeError(
                f"sparsity shape {pattern.shape} != {(len(before), len(x_current))}"
            )

        lower = np.full_like(x_current, -np.inf)
        upper = np.full_like(x_current, np.inf)
        lower[0::6] = -float(args.max_correction_rotation_rad)
        lower[1::6] = -float(args.max_correction_rotation_rad)
        lower[2::6] = -float(args.max_correction_rotation_rad)
        upper[0::6] = float(args.max_correction_rotation_rad)
        upper[1::6] = float(args.max_correction_rotation_rad)
        upper[2::6] = float(args.max_correction_rotation_rad)
        lower[3::6] = -float(args.max_correction_translation_m)
        lower[4::6] = -float(args.max_correction_translation_m)
        lower[5::6] = -float(args.max_correction_translation_m)
        upper[3::6] = float(args.max_correction_translation_m)
        upper[4::6] = float(args.max_correction_translation_m)
        upper[5::6] = float(args.max_correction_translation_m)
        anchor_slice = 6 * int(args.anchor_position)
        lower[anchor_slice:anchor_slice + 6] = -1.0e-10
        upper[anchor_slice:anchor_slice + 6] = 1.0e-10

        result = least_squares(
            lambda x: flatten_blocks(
                residual_blocks(x, nodes, point_factors, key_factors, rgb_factors, args)
            ),
            x_current,
            jac_sparsity=pattern,
            bounds=(lower, upper),
            max_nfev=int(args.inner_max_nfev),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
        )
        candidate_blocks = residual_blocks(
            result.x, nodes, point_factors, key_factors, rgb_factors, args
        )
        candidate = flatten_blocks(candidate_blocks)
        before_cost = float(np.sum(before * before))
        candidate_cost = float(np.sum(candidate * candidate))
        required_drop = float(args.min_outer_cost_improvement) * max(before_cost, 1.0)
        accepted_update = bool(result.success and candidate_cost <= before_cost - required_drop)
        if accepted_update:
            accepted_x = result.x.copy()
            accepted_blocks = candidate_blocks
            accepted = candidate
            Rs, ts = current_poses(nodes, accepted_x)
            for i, node in enumerate(nodes):
                node.base_rotation = Rs[i]
                node.base_translation = ts[i]
            accepted_outer_count = outer + 1
        else:
            accepted_x = np.zeros_like(x_current)
            accepted_blocks = before_blocks
            accepted = before
            accepted_outer_count = sum(
                bool(row.get("accepted_update")) for row in outer_reports
            )
        outer_reports.append(
            {
                "outer_iteration": int(outer),
                "optimizer_success": bool(result.success),
                "accepted_update": accepted_update,
                "nfev": int(result.nfev),
                "cost_before": before_cost,
                "cost_after": float(np.sum(accepted * accepted)),
                "residual_rms_before": float(np.sqrt(np.mean(before * before))) if len(before) else None,
                "residual_rms_after": float(np.sqrt(np.mean(accepted * accepted))) if len(accepted) else None,
                "term_before": term_metrics(before_blocks),
                "term_after": term_metrics(accepted_blocks),
                "point_factor_metrics": point_info,
                "correction_norm_translation_m": float(np.max(np.linalg.norm(accepted_x.reshape(-1, 6)[:, 3:], axis=1))),
                "correction_norm_rotation_deg": float(np.degrees(np.max(np.linalg.norm(accepted_x.reshape(-1, 6)[:, :3], axis=1)))),
            }
        )
        if not accepted_update:
            break
        x_current = np.zeros_like(x_current)

    final_rotations = [node.base_rotation.copy() for node in nodes]
    final_translations = [node.base_translation.copy() for node in nodes]
    mesh_after = mesh_surface_metrics_from_poses(
        nodes, final_rotations, final_translations, mesh_points
    )
    # The node bases now contain the accepted absolute pose.  A zero local
    # correction is therefore the correct representation for the final report.
    final_x = np.zeros(len(nodes) * 6, dtype=np.float64)
    report = build_report(
        args,
        initial,
        nodes,
        key_ids,
        outer_reports,
        final_x,
        mesh_before,
        mesh_after,
        key_factors,
        rgb_factors,
        frames,
    )
    report["surface_median_degradation_m"] = (
        mesh_after["observed_to_mesh_m"]["median"]
        - mesh_before["observed_to_mesh_m"]["median"]
        if mesh_after["observed_to_mesh_m"]["median"] is not None
        and mesh_before["observed_to_mesh_m"]["median"] is not None
        else None
    )
    report["outputs"] = {
        "pose_report": str(args.output_dir / "v20_keyframe_global_pose_report.json"),
        "v18_compatible_pose_report": str(
            args.output_dir / "v18_compact_rigid_object_pose_fit_report.json"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v20_keyframe_global_pose_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "v18_compact_rigid_object_pose_fit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "keyframes": key_ids,
                "outer_iterations": len(outer_reports),
                "accepted_outer_iterations": sum(
                    bool(row.get("accepted_update")) for row in outer_reports
                ),
                "surface_before": report["surface_before"]["observed_to_mesh_m"],
                "surface_after": report["surface_after"]["observed_to_mesh_m"],
                "surface_median_degradation_m": report["surface_median_degradation_m"],
            },
            indent=2,
        )
    )

if __name__=='__main__':main()
