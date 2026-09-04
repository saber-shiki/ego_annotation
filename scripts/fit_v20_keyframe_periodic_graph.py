#!/usr/bin/env python3
"""Fast V20 keyframe-only observed pose graph with periodic re-registration.

This is the first production-sized experiment for the V20 design: only
keyframe nodes are optimized, all-frame corrections are interpolated from the
keyframe updates, and correspondences are rebuilt at each outer pass.  It uses
point-to-plane-derived relative SE(3) factors, refreshed anchor/local point
factors, optional RGB/PnP keyframe edges, and correction temporal priors.
Generated SAM3D faces are never loaded.
"""
from __future__ import annotations

import argparse
import cv2
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import lsmr
from scipy.spatial.transform import Rotation


def import_v20_core(path: Path):
    spec = importlib.util.spec_from_file_location("v20_core", path.expanduser().resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import V20 core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["v20_core"] = module
    spec.loader.exec_module(module)
    return module


def solve_linearized_gn(fun, x0, pattern, lower, upper, args):
    """Small sparse Gauss--Newton solver used to avoid repeated finite-difference TRF passes."""
    x = np.asarray(x0, dtype=np.float64).copy()
    r = np.asarray(fun(x), dtype=np.float64)
    cost = float(np.dot(r, r))
    iterations = 0
    accepted_any = False
    for _ in range(int(args.gn_iterations)):
        iterations += 1
        jac = approx_derivative(
            fun,
            x,
            method="2-point",
            rel_step=float(args.gn_relative_step),
            sparsity=pattern,
            f0=r,
        )
        if not hasattr(jac, "tocsr"):
            jac = sparse.csr_matrix(jac)
        # Match the robust soft-L1 behavior used by the fallback solver.
        robust_w = 1.0 / np.sqrt(1.0 + r * r)
        weighted_jac = jac.multiply(robust_w[:, None])
        weighted_r = robust_w * r
        damping = float(args.gn_damping)
        step = lsmr(
            weighted_jac,
            -weighted_r,
            damp=math.sqrt(max(damping, 0.0)),
            atol=1.0e-6,
            btol=1.0e-6,
            maxiter=max(100, 4 * len(x)),
        )[0]
        step = np.asarray(step, dtype=np.float64)
        # Keep the gauge node fixed and honor the explicit per-node trust box.
        step = np.minimum(np.maximum(step, lower - x), upper - x)
        if not np.isfinite(step).all() or float(np.linalg.norm(step)) <= float(args.gn_step_tolerance):
            break
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.1):
            candidate_x = np.minimum(np.maximum(x + alpha * step, lower), upper)
            candidate_r = np.asarray(fun(candidate_x), dtype=np.float64)
            candidate_cost = float(np.dot(candidate_r, candidate_r))
            if candidate_cost < cost - float(args.gn_cost_tolerance) * max(1.0, cost):
                x, r, cost = candidate_x, candidate_r, candidate_cost
                accepted_any = True
                improved = True
                break
        if not improved:
            break
    return SimpleNamespace(
        x=x,
        success=bool(accepted_any),
        nfev=int(iterations),
        cost=0.5 * cost,
        message="linearized sparse Gauss-Newton",
    )

def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def pose_correction(initial_R: np.ndarray, initial_t: np.ndarray, final_R: np.ndarray, final_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_R = np.asarray(final_R, dtype=np.float64) @ np.asarray(initial_R, dtype=np.float64).T
    delta_t = np.asarray(final_t, dtype=np.float64) - delta_R @ np.asarray(initial_t, dtype=np.float64)
    return Rotation.from_matrix(delta_R).as_rotvec(), delta_t


def interpolate_pose_correction(
    frame_idx: int,
    all_frame_ids: list[int],
    key_ids: list[int],
    initial_by_frame: dict[int, tuple[np.ndarray, np.ndarray]],
    final_by_frame: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, str]:
    if frame_idx in final_by_frame:
        return final_by_frame[frame_idx][0].copy(), final_by_frame[frame_idx][1].copy(), "keyframe_direct"
    lower = [value for value in key_ids if value < frame_idx]
    upper = [value for value in key_ids if value > frame_idx]
    if lower and upper:
        lo, hi = lower[-1], upper[0]
        alpha = float(frame_idx - lo) / float(max(1, hi - lo))
        lo_dr, lo_dt = pose_correction(*initial_by_frame[lo], *final_by_frame[lo])
        hi_dr, hi_dt = pose_correction(*initial_by_frame[hi], *final_by_frame[hi])
        dr = (1.0 - alpha) * lo_dr + alpha * hi_dr
        dt = (1.0 - alpha) * lo_dt + alpha * hi_dt
        R0, t0 = initial_by_frame[frame_idx]
        delta_R = Rotation.from_rotvec(dr).as_matrix()
        return delta_R @ R0, delta_R @ t0 + dt, "interpolated_keyframe_correction"
    nearest = min(key_ids, key=lambda value: abs(value - frame_idx))
    dr, dt = pose_correction(*initial_by_frame[nearest], *final_by_frame[nearest])
    R0, t0 = initial_by_frame[frame_idx]
    delta_R = Rotation.from_rotvec(dr).as_matrix()
    return delta_R @ R0, delta_R @ t0 + dt, "nearest_keyframe_correction_hold"


def build_rgb_keyframe_absolute_factors(
    core,
    args: argparse.Namespace,
    key_nodes: list[Any],
    initial_report: dict[str, Any],
    rgb_npz: Path | None,
) -> list[Any]:
    """Build unary absolute target-pose factors from RGB/PnP edge deltas.

    The stored RGB edge translation/rotation is a left world delta relative to
    the *source P14 pose*.  Deltas from distinct source frames must not be
    chained as if they shared one baseline.  We therefore recover each edge's
    absolute target pose and keep the best incoming measurement per keyframe.
    """
    if rgb_npz is None:
        return []
    with np.load(rgb_npz, allow_pickle=False) as data:
        required = {
            "source_frame_idx", "target_frame_idx", "rotation_rgb",
            "translation_rgb_m", "quality_weight", "accepted",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"RGB edge NPZ missing keys: {missing}")
        sources = np.asarray(data["source_frame_idx"], dtype=np.int64)
        targets = np.asarray(data["target_frame_idx"], dtype=np.int64)
        deltas_R = np.asarray(data["rotation_rgb"], dtype=np.float64)
        deltas_t = np.asarray(data["translation_rgb_m"], dtype=np.float64)
        weights = np.asarray(data["quality_weight"], dtype=np.float64)
        accepted = np.asarray(data["accepted"], dtype=bool)
    edge_rows: dict[tuple[int, int], dict[str, Any]] = {}
    edge_json = rgb_npz.with_suffix(".json")
    if edge_json.exists():
        payload = json.loads(edge_json.read_text(encoding="utf-8"))
        edge_rows = {
            (int(row.get("source_frame_idx")), int(row.get("target_frame_idx"))): row
            for row in payload.get("rows", [])
            if isinstance(row, dict)
        }
    initial_rows = {
        int(row["frame_idx"]): row
        for row in initial_report.get("pose_rows", [])
        if isinstance(row, dict)
    }
    key_pos = {node.frame_idx: i for i, node in enumerate(key_nodes)}
    incoming: dict[int, list[tuple[np.ndarray, np.ndarray, float, int]]] = {}
    for i in range(len(sources)):
        source_idx = int(sources[i])
        target_idx = int(targets[i])
        if not accepted[i] or target_idx not in key_pos or source_idx not in initial_rows:
            continue
        edge_row = edge_rows.get((source_idx, target_idx), {})
        if (
            float(weights[i]) < float(args.rgb_absolute_min_quality)
            or float(edge_row.get("rotation_conflict_deg", 0.0)) > float(args.rgb_absolute_max_rotation_conflict_deg)
            or float(edge_row.get("translation_conflict_m", 0.0)) > float(args.rgb_absolute_max_translation_conflict_m)
            or float(edge_row.get("reprojection_median_px", 0.0)) > float(args.rgb_absolute_max_reprojection_median_px)
        ):
            continue
        source_row = initial_rows[source_idx]
        source_R = np.asarray(
            source_row["rotation_world_from_completed_canonical_matrix"],
            dtype=np.float64,
        )
        source_t = np.asarray(source_row["translation_world_m"], dtype=np.float64)
        absolute_R = deltas_R[i] @ source_R
        absolute_t = deltas_R[i] @ source_t + deltas_t[i]
        incoming.setdefault(target_idx, []).append(
            (absolute_R, absolute_t, float(weights[i]), source_idx)
        )
    factors = []
    scale = float(args.rgb_relative_weight_scale)
    if scale <= 0.0:
        return factors
    for target_idx, candidates in incoming.items():
        candidates.sort(key=lambda value: abs(target_idx - value[3]))
        absolute_R, absolute_t, quality, _source_idx = candidates[0]
        factors.append(
            core.RelativeFactor(
                key_pos[target_idx],
                key_pos[target_idx],
                target_idx,
                target_idx,
                absolute_R,
                absolute_t,
                float(np.clip(quality * scale, float(args.min_rgb_weight), 1.0)),
                "rgb_absolute",
            )
        )
    return factors


def resize_intrinsics_xy(K: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    sx = float(target_wh[0]) / float(source_wh[0])
    sy = float(target_wh[1]) / float(source_wh[1])
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = sx * (out[0, 2] + 0.5) - 0.5
    out[1, 2] = sy * (out[1, 2] + 0.5) - 0.5
    return out

def mask_metrics_from_mesh(
    core,
    nodes: list[Any],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    frames: dict[int, dict[str, Any]],
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    values = []
    centroids = []
    per_frame = {}
    raster_size = int(args.mask_gate_raster_size)
    for node, rotation, translation in zip(nodes, rotations, translations):
        frame = frames[node.frame_idx]
        obj = next((o for o in frame.get("objects", []) if o.get("object_id") == args.object_id), None)
        if obj is None:
            continue
        mask = cv2.imread(str(obj.get("mask_path") or ""), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        height, width = mask.shape
        geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        values_k = np.asarray(geom.get("intrinsics_fx_fy_cx_cy") or [], dtype=np.float64)
        if values_k.shape != (4,):
            continue
        k = np.asarray([[values_k[0], 0.0, values_k[2]], [0.0, values_k[1], values_k[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
        k = resize_intrinsics_xy(k, (int(frame.get("source_width") or 1408), int(frame.get("source_height") or 1408)), (width, height))
        k = resize_intrinsics_xy(k, (width, height), (raster_size, raster_size))
        target = cv2.resize((mask > 0).astype(np.uint8), (raster_size, raster_size), interpolation=cv2.INTER_NEAREST_EXACT) > 0
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        world = np.asarray(mesh_vertices, dtype=np.float64) @ rotation.T + translation[None, :]
        camera = (world - T[:3, 3]) @ T[:3, :3]
        valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1.0e-6)
        if np.count_nonzero(valid) < 3:
            continue
        uv_all = np.zeros((len(camera), 2), dtype=np.float64)
        uv_all[valid] = np.column_stack((k[0, 0] * camera[valid, 0] / camera[valid, 2] + k[0, 2], k[1, 1] * camera[valid, 1] / camera[valid, 2] + k[1, 2]))
        uv_all = np.rint(uv_all).astype(np.int32)
        face_indices = np.asarray(mesh_faces, dtype=np.int64)
        valid_faces = face_indices[np.all(valid[face_indices], axis=1)]
        polygons = [uv_all[tri] for tri in valid_faces if cv2.contourArea(uv_all[tri].astype(np.float32)) > 0.0]
        rendered = np.zeros((raster_size, raster_size), dtype=np.uint8)
        if polygons:
            cv2.fillPoly(rendered, polygons, 1)
        rendered = rendered > 0
        intersection = np.count_nonzero(rendered & target)
        union = np.count_nonzero(rendered | target)
        rendered_coords = np.argwhere(rendered)
        target_coords = np.argwhere(target)
        centroid = float(np.linalg.norm(rendered_coords.mean(axis=0) - target_coords.mean(axis=0))) if len(rendered_coords) and len(target_coords) else float("inf")
        iou = float(intersection / max(1, union))
        values.append(iou); centroids.append(centroid); per_frame[str(node.frame_idx)] = {"iou": iou, "centroid_delta_px": centroid}
    return {"iou": core.numeric_summary(values), "centroid_delta_px": core.numeric_summary(centroids), "per_frame": per_frame}


def mesh_metrics(core, nodes, rotations, translations, mesh_points):
    return core.mesh_surface_metrics_from_poses(nodes, rotations, translations, mesh_points)




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-script", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--pose-mesh", type=Path, required=True)
    parser.add_argument("--keyframe-edge-npz", type=Path, required=True)
    parser.add_argument("--keyframe-edge-json", type=Path, required=True)
    parser.add_argument("--rgb-edge-npz", type=Path, default=None)
    parser.add_argument("--image-factor-npz", type=Path, default=None)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--anchor-frame", type=int, default=0)
    parser.add_argument("--keyframe-interval", type=int, default=5)
    parser.add_argument("--max-keyframes", type=int, default=40)
    parser.add_argument("--outer-iterations", type=int, default=3)
    parser.add_argument("--inner-max-nfev", type=int, default=8)
    parser.add_argument("--optimizer-mode", choices=("least_squares", "linearized_gn"), default="linearized_gn")
    parser.add_argument("--gn-iterations", type=int, default=2)
    parser.add_argument("--gn-relative-step", type=float, default=1.0e-5)
    parser.add_argument("--gn-damping", type=float, default=1.0e-4)
    parser.add_argument("--gn-step-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--gn-cost-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--optimizer-ftol", type=float, default=1e-5)
    parser.add_argument("--optimizer-xtol", type=float, default=1e-5)
    parser.add_argument("--optimizer-gtol", type=float, default=1e-5)
    parser.add_argument("--max-points", type=int, default=300)
    parser.add_argument("--max-mesh-points", type=int, default=1500)
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--normal-k", type=int, default=24)
    parser.add_argument("--max-anchor-pairs", type=int, default=80)
    parser.add_argument("--max-edge-pairs", type=int, default=80)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--trim-fraction", type=float, default=0.70)
    parser.add_argument("--max-correspondence-m", type=float, default=0.045)
    parser.add_argument("--max-anchor-correspondence-m", type=float, default=0.055)
    parser.add_argument("--max-edge-correspondence-m", type=float, default=0.045)
    parser.add_argument("--edge-weight-scale-m", type=float, default=0.010)
    parser.add_argument("--point-weight-scale-m", type=float, default=0.012)
    parser.add_argument("--normal-quality-scale", type=float, default=0.25)
    parser.add_argument("--keyframe-anchor-weight", type=float, default=1.0)
    parser.add_argument("--non-keyframe-anchor-weight", type=float, default=0.35)
    parser.add_argument("--local-edge-weight", type=float, default=1.0)
    parser.add_argument("--anchor-point-factor-scale", type=float, default=1.0)
    parser.add_argument("--local-point-factor-scale", type=float, default=1.0)
    parser.add_argument("--keyframe-relative-weight-scale", type=float, default=1.0)
    parser.add_argument("--rgb-relative-weight-scale", type=float, default=1.0)
    parser.add_argument("--rgb-absolute-min-quality", type=float, default=0.50)
    parser.add_argument("--rgb-absolute-max-rotation-conflict-deg", type=float, default=4.0)
    parser.add_argument("--rgb-absolute-max-translation-conflict-m", type=float, default=0.020)
    parser.add_argument("--rgb-absolute-max-reprojection-median-px", type=float, default=0.80)
    parser.add_argument("--removed-fraction-full-weight", type=float, default=0.10)
    parser.add_argument("--min-depth-quality", type=float, default=0.25)
    parser.add_argument("--edge-residual-scale-m", type=float, default=0.004)
    parser.add_argument("--rotation-information-scale", type=float, default=0.01)
    parser.add_argument("--max-rotation-condition", type=float, default=1e8)
    parser.add_argument("--min-relative-weight", type=float, default=0.10)
    parser.add_argument("--min-rgb-weight", type=float, default=0.05)
    parser.add_argument("--sigma-pose-prior-translation-m", type=float, default=0.04)
    parser.add_argument("--sigma-pose-prior-rotation-rad", type=float, default=0.20)
    parser.add_argument("--sigma-point-to-plane-m", type=float, default=0.006)
    parser.add_argument("--point-to-plane-weight", type=float, default=1.0)
    parser.add_argument("--sigma-point-to-point-m", type=float, default=0.015)
    parser.add_argument("--point-to-point-weight", type=float, default=0.20)
    parser.add_argument("--max-point-residual-m", type=float, default=0.05)
    parser.add_argument("--sigma-relative-rotation-rad", type=float, default=0.10)
    parser.add_argument("--sigma-relative-translation-m", type=float, default=0.018)
    parser.add_argument("--sigma-rgb-rotation-rad", type=float, default=0.08)
    parser.add_argument("--sigma-rgb-translation-m", type=float, default=0.015)
    parser.add_argument("--sigma-correction-translation-step-m", type=float, default=0.012)
    parser.add_argument("--sigma-correction-rotation-step-rad", type=float, default=0.10)
    parser.add_argument("--sigma-correction-translation-accel-m", type=float, default=0.008)
    parser.add_argument("--sigma-correction-rotation-accel-rad", type=float, default=0.06)
    parser.add_argument("--sigma-motion-translation-accel-m", type=float, default=0.04)
    parser.add_argument("--sigma-motion-rotation-accel-rad", type=float, default=0.20)
    parser.add_argument("--sigma-anchor-gauge-translation-m", type=float, default=1e-6)
    parser.add_argument("--sigma-anchor-gauge-rotation-rad", type=float, default=1e-6)
    parser.add_argument("--max-correction-translation-m", type=float, default=0.008)
    parser.add_argument("--max-correction-rotation-rad", type=float, default=0.04)
    parser.add_argument("--max-cumulative-translation-m", type=float, default=0.03)
    parser.add_argument("--max-cumulative-rotation-rad", type=float, default=0.15)
    parser.add_argument("--max-surface-degradation-m", type=float, default=0.003)
    parser.add_argument("--mask-gate-raster-size", type=int, default=128)
    parser.add_argument("--max-mask-iou-degradation", type=float, default=0.005)
    parser.add_argument("--max-mask-centroid-degradation-px", type=float, default=3.0)
    parser.add_argument("--max-mask-iou-mean-degradation", type=float, default=0.001)
    parser.add_argument("--max-mask-centroid-mean-degradation-px", type=float, default=0.5)
    parser.add_argument("--image-first-hit-weight", type=float, default=1.0)
    parser.add_argument("--image-silhouette-weight", type=float, default=1.0)
    parser.add_argument("--sigma-image-first-hit-m", type=float, default=0.008)
    parser.add_argument("--sigma-image-silhouette-px", type=float, default=4.0)
    parser.add_argument("--max-image-first-hit-residual-m", type=float, default=0.03)
    parser.add_argument("--max-image-silhouette-residual-px", type=float, default=16.0)
    parser.add_argument("--min-outer-cost-improvement", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    run_start = time.perf_counter()

    core = import_v20_core(args.core_script)
    annotations = load_json(args.annotations)
    initial_report = load_json(args.pose_report)
    all_nodes, frames = core.load_nodes(args, annotations, initial_report)
    anchor_frame = int(args.anchor_frame if args.anchor_frame is not None else initial_report.get("anchor_frame_idx", all_nodes[0].frame_idx))
    all_pos = {node.frame_idx: i for i, node in enumerate(all_nodes)}
    if anchor_frame not in all_pos:
        raise RuntimeError(f"anchor frame {anchor_frame} is not an accepted direct node")
    key_ids = core.keyframe_positions(all_nodes, anchor_frame, int(args.keyframe_interval), int(args.max_keyframes))
    key_nodes = [all_nodes[all_pos[idx]] for idx in key_ids]
    key_pos = {node.frame_idx: i for i, node in enumerate(key_nodes)}
    for node in key_nodes:
        node.keyframe = True
    args.anchor_position = key_pos[anchor_frame]
    key_factors, _ = core.load_relative_factors(
        args, key_nodes, args.keyframe_edge_json, args.keyframe_edge_npz, None
    )
    rgb_factors = build_rgb_keyframe_absolute_factors(
        core, args, key_nodes, initial_report, args.rgb_edge_npz
    )
    image_factors = core.load_image_factors(
        args.image_factor_npz, {node.frame_idx for node in key_nodes}, args
    )

    import trimesh
    mesh = trimesh.load(args.pose_mesh.expanduser().resolve(), force="mesh", process=False)
    mesh_points, _ = trimesh.sample.sample_surface(mesh, min(int(args.max_mesh_points), max(1, len(mesh.faces) * 2)), seed=np.random.default_rng(int(args.seed) + 701))
    mesh_points = np.asarray(mesh_points, dtype=np.float64)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    initial_key_R = [node.base_rotation.copy() for node in key_nodes]
    initial_key_t = [node.base_translation.copy() for node in key_nodes]
    initial_by_frame = {
        node.frame_idx: (node.base_rotation.copy(), node.base_translation.copy())
        for node in all_nodes
    }
    all_frame_ids = [node.frame_idx for node in all_nodes]
    def expand_key_poses(key_rotations, key_translations):
        final_by_frame = {
            node.frame_idx: (key_rotations[i], key_translations[i])
            for i, node in enumerate(key_nodes)
        }
        expanded_R = []
        expanded_t = []
        for node in all_nodes:
            rotation, translation, _mode = interpolate_pose_correction(
                node.frame_idx, all_frame_ids, key_ids, initial_by_frame, final_by_frame
            )
            expanded_R.append(rotation)
            expanded_t.append(translation)
        return expanded_R, expanded_t
    initial_all_R = [initial_by_frame[node.frame_idx][0] for node in all_nodes]
    initial_all_t = [initial_by_frame[node.frame_idx][1] for node in all_nodes]
    initial_mesh = mesh_metrics(core, key_nodes, initial_key_R, initial_key_t, mesh_points)
    initial_mask = mask_metrics_from_mesh(core, all_nodes, initial_all_R, initial_all_t, frames, mesh_vertices, mesh_faces, args)
    accepted_mask = initial_mask

    outer_reports: list[dict[str, Any]] = []
    for outer in range(int(args.outer_iterations)):
        print(
            f"[v20-kf] outer {outer + 1}/{args.outer_iterations}: "
            f"rebuilding correspondences for {len(key_nodes)} keyframes",
            flush=True,
        )
        if float(args.anchor_point_factor_scale) > 0.0 or float(args.local_point_factor_scale) > 0.0:
            point_factors, point_info = core.build_dynamic_point_factors(
                key_nodes,
                list(range(len(key_nodes))),
                int(args.anchor_position),
                args,
                outer,
            )
        else:
            point_factors = []
            point_info = {
                "outer_iteration": int(outer),
                "factor_group_count": 0,
                "anchor_point_count": 0,
                "local_point_count": 0,
                "skipped": "both point-factor scales are zero",
            }
        x0 = np.zeros(len(key_nodes) * 6, dtype=np.float64)
        before_blocks = core.residual_blocks(
            x0, key_nodes, point_factors, key_factors, rgb_factors, args, image_factors
        )
        before = core.flatten_blocks(before_blocks)
        pattern = core.residual_sparsity(
            key_nodes, point_factors, key_factors, rgb_factors, args, image_factors
        )
        if pattern.shape != (len(before), len(x0)):
            raise RuntimeError(
                f"v20 sparsity mismatch {pattern.shape} vs {(len(before), len(x0))}"
            )

        lower = np.full_like(x0, -np.inf)
        upper = np.full_like(x0, np.inf)
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

        residual_fun = lambda x: core.flatten_blocks(
            core.residual_blocks(
                x, key_nodes, point_factors, key_factors, rgb_factors, args, image_factors
            )
        )
        if args.optimizer_mode == "linearized_gn":
            result = solve_linearized_gn(
                residual_fun, x0, pattern, lower, upper, args
            )
        else:
            result = core.least_squares(
                residual_fun,
                x0,
                jac_sparsity=pattern,
                bounds=(lower, upper),
                max_nfev=int(args.inner_max_nfev),
                loss="soft_l1",
                f_scale=1.0,
                x_scale="jac",
                ftol=float(args.optimizer_ftol),
                xtol=float(args.optimizer_xtol),
                gtol=float(args.optimizer_gtol),
            )
        candidate_blocks = core.residual_blocks(
            result.x, key_nodes, point_factors, key_factors, rgb_factors, args, image_factors
        )
        candidate = core.flatten_blocks(candidate_blocks)
        before_cost = float(np.sum(before * before))
        candidate_cost = float(np.sum(candidate * candidate))
        current_R = [node.base_rotation.copy() for node in key_nodes]
        current_t = [node.base_translation.copy() for node in key_nodes]
        candidate_R, candidate_t = core.current_poses(key_nodes, result.x)
        current_surface = mesh_metrics(
            core, key_nodes, current_R, current_t, mesh_points
        )
        candidate_surface = mesh_metrics(
            core, key_nodes, candidate_R, candidate_t, mesh_points
        )
        current_all_R, current_all_t = expand_key_poses(current_R, current_t)
        candidate_all_R, candidate_all_t = expand_key_poses(candidate_R, candidate_t)
        current_mask = accepted_mask
        candidate_mask = mask_metrics_from_mesh(
            core, all_nodes, candidate_all_R, candidate_all_t, frames, mesh_vertices, mesh_faces, args
        )
        current_iou = current_mask["iou"]["median"]
        candidate_iou = candidate_mask["iou"]["median"]
        current_centroid = current_mask["centroid_delta_px"]["median"]
        candidate_centroid = candidate_mask["centroid_delta_px"]["median"]
        mask_iou_delta = (
            float(candidate_iou - current_iou)
            if current_iou is not None and candidate_iou is not None
            else float("-inf")
        )
        mask_centroid_delta = (
            float(candidate_centroid - current_centroid)
            if current_centroid is not None and candidate_centroid is not None
            else float("inf")
        )
        mask_iou_mean_delta = (
            float(candidate_mask["iou"].get("mean") - current_mask["iou"].get("mean"))
            if current_mask["iou"].get("mean") is not None and candidate_mask["iou"].get("mean") is not None
            else float("-inf")
        )
        mask_centroid_mean_delta = (
            float(candidate_mask["centroid_delta_px"].get("mean") - current_mask["centroid_delta_px"].get("mean"))
            if current_mask["centroid_delta_px"].get("mean") is not None and candidate_mask["centroid_delta_px"].get("mean") is not None
            else float("inf")
        )
        current_med = current_surface["observed_to_mesh_m"]["median"]
        candidate_med = candidate_surface["observed_to_mesh_m"]["median"]
        surface_delta = (
            float(candidate_med - current_med)
            if current_med is not None and candidate_med is not None
            else float("inf")
        )
        cumulative_rot = max(
            float(
                np.degrees(
                    np.linalg.norm(
                        Rotation.from_matrix(candidate_R[i] @ initial_key_R[i].T).as_rotvec()
                    )
                )
            )
            for i in range(len(key_nodes))
        )
        cumulative_trans = max(
            float(np.linalg.norm(candidate_t[i] - initial_key_t[i]))
            for i in range(len(key_nodes))
        )
        required_drop = float(args.min_outer_cost_improvement) * max(before_cost, 1.0)
        accepted = bool(
            result.success
            and candidate_cost <= before_cost - required_drop
            and surface_delta <= float(args.max_surface_degradation_m)
            and np.radians(cumulative_rot) <= float(args.max_cumulative_rotation_rad)
            and cumulative_trans <= float(args.max_cumulative_translation_m)
            and mask_iou_delta >= -float(args.max_mask_iou_degradation)
            and mask_iou_mean_delta >= -float(args.max_mask_iou_mean_degradation)
            and mask_centroid_delta <= float(args.max_mask_centroid_degradation_px)
            and mask_centroid_mean_delta <= float(args.max_mask_centroid_mean_degradation_px)
        )
        outer_reports.append(
            {
                "outer_iteration": int(outer),
                "optimizer_success": bool(result.success),
                "accepted_update": accepted,
                "nfev": int(result.nfev),
                "cost_before": before_cost,
                "candidate_cost": candidate_cost,
                "cost_after": candidate_cost if accepted else before_cost,
                "residual_rms_before": float(np.sqrt(np.mean(before * before))) if len(before) else None,
                "candidate_residual_rms": float(np.sqrt(np.mean(candidate * candidate))) if len(candidate) else None,
                "residual_rms_after": float(np.sqrt(np.mean(candidate * candidate))) if accepted and len(candidate) else (float(np.sqrt(np.mean(before * before))) if len(before) else None),
                "term_before": core.term_metrics(before_blocks),
                "term_candidate": core.term_metrics(candidate_blocks),
                "term_after": core.term_metrics(candidate_blocks if accepted else before_blocks),
                "point_factor_metrics": point_info,
                "candidate_surface_median_degradation_m": surface_delta,
                "candidate_mask_iou_delta": mask_iou_delta,
                "candidate_mask_iou_mean_delta": mask_iou_mean_delta,
                "candidate_mask_centroid_delta_px": mask_centroid_delta,
                "candidate_mask_centroid_mean_delta_px": mask_centroid_mean_delta,
                "current_mask": current_mask,
                "candidate_mask": candidate_mask,
                "candidate_cumulative_rotation_deg": cumulative_rot,
                "candidate_cumulative_translation_m": cumulative_trans,
                "rejection_reasons": [] if accepted else [
                    reason for reason, failed in [
                        ("optimizer_not_success", not bool(result.success)),
                        ("insufficient_cost_drop", not (candidate_cost <= before_cost - required_drop)),
                        ("surface_degradation", not (surface_delta <= float(args.max_surface_degradation_m))),
                        ("cumulative_rotation_bound", not (np.radians(cumulative_rot) <= float(args.max_cumulative_rotation_rad))),
                        ("cumulative_translation_bound", not (cumulative_trans <= float(args.max_cumulative_translation_m))),
                        ("mask_iou_degradation", not (mask_iou_delta >= -float(args.max_mask_iou_degradation))),
                        ("mask_iou_mean_degradation", not (mask_iou_mean_delta >= -float(args.max_mask_iou_mean_degradation))),
                        ("mask_centroid_degradation", not (mask_centroid_delta <= float(args.max_mask_centroid_degradation_px))),
                        ("mask_centroid_mean_degradation", not (mask_centroid_mean_delta <= float(args.max_mask_centroid_mean_degradation_px))),
                    ] if failed
                ],
            }
        )
        print(
            f"[v20-kf] outer {outer + 1}: success={result.success} "
            f"accepted={accepted} nfev={result.nfev} "
            f"rms={outer_reports[-1]['residual_rms_before']:.5f}->"
            f"{outer_reports[-1]['residual_rms_after']:.5f} "
            f"surface_delta={surface_delta * 1000.0:.2f}mm",
            flush=True,
        )
        if accepted:
            accepted_mask = candidate_mask
            for i, node in enumerate(key_nodes):
                node.base_rotation = candidate_R[i]
                node.base_translation = candidate_t[i]
        else:
            break
    final_key_R={node.frame_idx:node.base_rotation.copy() for node in key_nodes};final_key_t={node.frame_idx:node.base_translation.copy() for node in key_nodes}
    initial_by_frame = {
        node.frame_idx: (
            np.asarray(node.source_row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64),
            np.asarray(node.source_row["translation_world_m"], dtype=np.float64),
        )
        for node in all_nodes
    }
    # Use the original P14 pose row as the interpolation baseline, not a mutable node reference.
    final_by_frame={node.frame_idx:(final_key_R[node.frame_idx],final_key_t[node.frame_idx]) for node in key_nodes}
    final_R=[];final_t=[];modes={}
    for node in all_nodes:
        R,t,mode=interpolate_pose_correction(node.frame_idx,[n.frame_idx for n in all_nodes],key_ids,initial_by_frame,final_by_frame);final_R.append(R);final_t.append(t);modes[mode]=modes.get(mode,0)+1
    final_mesh=mesh_metrics(core,all_nodes,final_R,final_t,mesh_points)
    initial_all_R=[initial_by_frame[node.frame_idx][0] for node in all_nodes];initial_all_t=[initial_by_frame[node.frame_idx][1] for node in all_nodes];initial_all_mesh=mesh_metrics(core,all_nodes,initial_all_R,initial_all_t,mesh_points)
    node_pos_all={node.frame_idx:i for i,node in enumerate(all_nodes)};rows=[]
    for original in initial_report.get('pose_rows',[]):
        if not isinstance(original,dict):rows.append(original);continue
        idx=int(original.get('frame_idx',-1));row=dict(original)
        if idx in node_pos_all:
            i=node_pos_all[idx];node=all_nodes[i];row['rotation_world_from_completed_canonical_matrix']=final_R[i].astype(float).tolist();row['translation_world_m']=final_t[i].astype(float).tolist();row['pose_source']='v20_keyframe_periodic_global_observed_only';row['direct_pose_observation_source']='v20_keyframe_point_to_plane_graph';row['generated_geometry_pose_evidence_consumed']=False;row['observed_to_mesh_initial']=initial_all_mesh['per_frame'].get(str(idx),{}).get('observed_to_mesh',{});row['observed_to_mesh_final']=final_mesh['per_frame'].get(str(idx),{}).get('observed_to_mesh',{});row['mesh_to_observed_final']=final_mesh['per_frame'].get(str(idx),{}).get('mesh_to_observed',{});row['v20_temporal_mode']='keyframe_direct' if idx in final_by_frame else 'interpolated_keyframe_correction';row['v20_uncertainty']={'keyframe':bool(node.keyframe),'depth_quality':float(node.depth_quality)}
        rows.append(row)
    direct=sorted(node_pos_all);rot_steps=[];trans_steps=[]
    for a,b in zip(direct[:-1],direct[1:]):
        i=node_pos_all[a];j=node_pos_all[b];gap=max(1,b-a);rot_steps.append(float(np.degrees(np.linalg.norm(Rotation.from_matrix(final_R[j]@final_R[i].T).as_rotvec()))/gap));trans_steps.append(float(np.linalg.norm(final_t[j]-final_t[i])/gap))
    final_mask = accepted_mask
    image_depth_count = int(sum(len(f.depth_observed_z) for f in image_factors.values()))
    image_silhouette_count = int(sum(len(f.silhouette_kind) for f in image_factors.values()))
    initial_mask_iou = initial_mask.get("iou", {}).get("median")
    final_mask_iou = final_mask.get("iou", {}).get("median")
    key_edge_report = load_json(args.keyframe_edge_json)
    key_edge_rows = key_edge_report.get("rows", []) if isinstance(key_edge_report.get("rows"), list) else []
    key_edge_evidence = {
        "row_count": int(key_edge_report.get("pair_count") or len(key_edge_rows)),
        "accepted_count": int(key_edge_report.get("accepted_count") or sum(bool(row.get("accepted")) for row in key_edge_rows if isinstance(row, dict))),
        "rejected_count": int(key_edge_report.get("rejected_count") or sum(not bool(row.get("accepted")) for row in key_edge_rows if isinstance(row, dict))),
        "accepted_fraction": key_edge_report.get("accepted_fraction"),
        "rotation_information_min_eigen": key_edge_report.get("rotation_information_min_eigen"),
        "rotation_condition": key_edge_report.get("rotation_condition"),
        "point_to_plane_abs_median": key_edge_report.get("point_to_plane_abs_median"),
    }
    report = {
        "schema": "v20_keyframe_periodic_global_observed_pose_graph_v2",
        "status": "v20_keyframe_global_complete" if any(bool(r.get("accepted_update")) for r in outer_reports) else "v20_keyframe_global_rejected_or_incomplete",
        "annotation_ready": False,
        "diagnostic_only": True,
        "formal_state_modified": False,
        "claim_scope": (
            "Observed-only keyframe SE(3) diagnostic. Only keyframe nodes are optimized; "
            "non-keyframes receive an explicit left-SE(3) interpolated correction. "
            "3D observed-surface correspondences are rebuilt per outer pass, while the "
            "strict P15 image factors are frozen/relinearized (not silently regenerated). "
            "Generated SAM3D faces are never used as pose, collision, contact, or SDF authority."
        ),
        "object_id": args.object_id,
        "inputs": {
            "annotations": str(args.annotations.resolve()),
            "initial_pose_report": str(args.pose_report.resolve()),
            "pose_mesh_observed_surface": str(args.pose_mesh.resolve()),
            "keyframe_edge_npz": str(args.keyframe_edge_npz.resolve()),
            "keyframe_edge_json": str(args.keyframe_edge_json.resolve()),
            "rgb_edge_npz": str(args.rgb_edge_npz.resolve()) if args.rgb_edge_npz else None,
            "image_factor_npz": str(args.image_factor_npz.resolve()) if args.image_factor_npz else None,
            "generated_geometry_consumed": False,
            "sha256": {
                "annotations": core.sha256_file(args.annotations),
                "initial_pose_report": core.sha256_file(args.pose_report),
                "pose_mesh_observed_surface": core.sha256_file(args.pose_mesh),
                "keyframe_edge_npz": core.sha256_file(args.keyframe_edge_npz),
                "keyframe_edge_json": core.sha256_file(args.keyframe_edge_json),
                "rgb_edge_npz": core.sha256_file(args.rgb_edge_npz) if args.rgb_edge_npz else None,
                "image_factor_npz": core.sha256_file(args.image_factor_npz) if args.image_factor_npz else None,
            },
        },
        "three_d_edge_evidence": key_edge_evidence,
        "keyframes": {
            "interval": int(args.keyframe_interval),
            "frame_ids": key_ids,
            "count": len(key_ids),
            "optimized_node_count": len(key_nodes),
            "all_direct_node_count": len(all_nodes),
            "outer_iterations_requested": int(args.outer_iterations),
            "outer_iterations_completed": len(outer_reports),
            "correspondences_rebuilt_each_outer_iteration": True,
            "image_correspondences_frozen": bool(args.image_factor_npz),
            "periodic_reanchor": True,
        },
        "factors": {
            "keyframe_point_factor_group_count": len(point_factors) if "point_factors" in locals() else 0,
            "keyframe_relative_edges": len(key_factors),
            "rgb_absolute_or_relative_edges": len(rgb_factors),
            "image_factor_keyframe_count": len(image_factors),
            "image_depth_factor_count": image_depth_count,
            "image_silhouette_factor_count": image_silhouette_count,
            "generated_geometry_consumed": False,
        },
        "parameters": {
            "inner_max_nfev": int(args.inner_max_nfev),
            "optimizer_mode": str(args.optimizer_mode),
            "optimizer_ftol": float(args.optimizer_ftol),
            "optimizer_xtol": float(args.optimizer_xtol),
            "optimizer_gtol": float(args.optimizer_gtol),
            "image_first_hit_weight": float(args.image_first_hit_weight),
            "image_silhouette_weight": float(args.image_silhouette_weight),
            "max_mask_iou_degradation": float(args.max_mask_iou_degradation),
            "max_mask_iou_mean_degradation": float(args.max_mask_iou_mean_degradation),
            "max_mask_centroid_degradation_px": float(args.max_mask_centroid_degradation_px),
            "max_mask_centroid_mean_degradation_px": float(args.max_mask_centroid_mean_degradation_px),
            "max_surface_degradation_m": float(args.max_surface_degradation_m),
        },
        "optimizer_outer_iterations": outer_reports,
        "temporal": {
            "direct_frame_count": len(all_nodes),
            "timeline_frame_count": len(frames),
            "direct_fraction": len(all_nodes) / max(1, len(frames)),
            "output_mode_counts": modes,
            "rotation_step_deg": core.numeric_summary(rot_steps),
            "translation_step_m": core.numeric_summary(trans_steps),
            "correction_is_applied_periodically_at_keyframes": True,
        },
        "mask_evidence_before": initial_mask,
        "mask_evidence_after": final_mask,
        "mask_median_iou_delta": (
            float(final_mask_iou - initial_mask_iou)
            if initial_mask_iou is not None and final_mask_iou is not None else None
        ),
        "surface_before": initial_all_mesh,
        "surface_after": final_mesh,
        "surface_median_degradation_m": (
            final_mesh["observed_to_mesh_m"]["median"] - initial_all_mesh["observed_to_mesh_m"]["median"]
            if final_mesh["observed_to_mesh_m"]["median"] is not None
            and initial_all_mesh["observed_to_mesh_m"]["median"] is not None else None
        ),
        "pose_rows": rows,
        "outputs": {
            "pose_report": str(args.output_dir / "v20_keyframe_global_pose_report.json"),
            "v18_compatible_pose_report": str(args.output_dir / "v18_compact_rigid_object_pose_fit_report.json"),
        },
        "elapsed_s": float(time.perf_counter() - run_start),
    }
    args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/'v20_keyframe_global_pose_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');(args.output_dir/'v18_compact_rigid_object_pose_fit_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':report['status'],'keyframes':key_ids,'outer_iterations':len(outer_reports),'output_mode_counts':modes,'surface_before':initial_all_mesh['observed_to_mesh_m'],'surface_after':final_mesh['observed_to_mesh_m'],'surface_median_degradation_m':report['surface_median_degradation_m']},indent=2))

if __name__=='__main__':main()
