#!/usr/bin/env python3
"""V20 keyframe pose graph with dynamic generated-surface first hits.

Only keyframe nodes are optimized, while every accepted visible frame can
constrain its neighboring keyframe corrections.  Observed-surface canonical
consistency, optional RGB/PnP evidence, and temporal priors are combined with
true first-hit depth and bidirectional silhouette factors rebuilt from the
same generated canonical mesh at every outer pass.

Generated faces remain diagnostic visible-pose/render evidence; this solver
does not promote them to collision, contact, SDF, or signed-volume authority.
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


def import_generated_factor_module(path: Path):
    resolved = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("v20_generated_factors", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import V20 generated-factor module: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["v20_generated_factors"] = module
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


def pose_map_from_report(report: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in report.get("pose_rows", []):
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        rotation = np.asarray(
            row.get("rotation_world_from_completed_canonical_matrix") or [],
            dtype=np.float64,
        )
        translation = np.asarray(row.get("translation_world_m") or [], dtype=np.float64)
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or abs(np.linalg.det(rotation) - 1.0) > 1.0e-4
        ):
            continue
        frame_idx = int(row["frame_idx"])
        if frame_idx in poses:
            raise RuntimeError(f"duplicate pose row at frame {frame_idx}")
        poses[frame_idx] = (rotation, translation)
    return poses


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


def key_support_positions(
    frame_idx: int,
    key_ids: list[int],
    key_pos: dict[int, int],
) -> tuple[int, ...]:
    """Return key variables that control one frame's interpolated correction."""
    if frame_idx in key_pos:
        return (int(key_pos[frame_idx]),)
    lower = [value for value in key_ids if value < frame_idx]
    upper = [value for value in key_ids if value > frame_idx]
    if lower and upper:
        return (int(key_pos[lower[-1]]), int(key_pos[upper[0]]))
    nearest = min(key_ids, key=lambda value: abs(value - frame_idx))
    return (int(key_pos[nearest]),)


def expand_candidate_poses(
    core: Any,
    key_nodes: list[Any],
    x: np.ndarray,
    all_nodes: list[Any],
    key_ids: list[int],
    initial_by_frame: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[np.ndarray], list[np.ndarray]]:
    key_rotations, key_translations = core.current_poses(key_nodes, x)
    final_by_frame = {
        node.frame_idx: (key_rotations[pos], key_translations[pos])
        for pos, node in enumerate(key_nodes)
    }
    all_frame_ids = [node.frame_idx for node in all_nodes]
    pose_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for node in all_nodes:
        rotation, translation, _mode = interpolate_pose_correction(
            node.frame_idx,
            all_frame_ids,
            key_ids,
            initial_by_frame,
            final_by_frame,
        )
        pose_by_frame[int(node.frame_idx)] = (rotation, translation)
        rotations.append(rotation)
        translations.append(translation)
    return pose_by_frame, rotations, translations


def generated_image_factor_blocks(
    core: Any,
    factor: Any,
    rotation: np.ndarray,
    translation: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    depth_block = None
    silhouette_block = None
    if (
        len(factor.depth_observed_z)
        and float(args.generated_visible_first_hit_weight) > 0.0
    ):
        world = core.apply_pose(
            factor.depth_points_canonical, rotation, translation
        )
        camera = (
            world - factor.T_world_camera[:3, 3]
        ) @ factor.T_world_camera[:3, :3]
        depth = camera[:, 2] - factor.depth_observed_z
        max_depth = float(args.generated_visible_max_depth_residual_m)
        if max_depth > 0.0:
            depth = np.clip(depth, -max_depth, max_depth)
        depth_block = (
            math.sqrt(float(args.generated_visible_first_hit_weight))
            * np.sqrt(factor.depth_weight)
            * depth
            / float(args.generated_visible_sigma_depth_m)
            / math.sqrt(max(float(np.sum(factor.depth_weight)), 1.0))
        )
    if (
        len(factor.silhouette_kind)
        and float(args.generated_visible_silhouette_weight) > 0.0
    ):
        world = core.apply_pose(
            factor.silhouette_points_canonical, rotation, translation
        )
        camera = (
            world - factor.T_world_camera[:3, 3]
        ) @ factor.T_world_camera[:3, :3]
        z = np.maximum(camera[:, 2], 1.0e-9)
        uv = np.column_stack(
            (
                factor.K_raster[0, 0] * camera[:, 0] / z
                + factor.K_raster[0, 2],
                factor.K_raster[1, 1] * camera[:, 1] / z
                + factor.K_raster[1, 2],
            )
        )
        diff = uv - factor.silhouette_target_uv
        norms = np.linalg.norm(diff, axis=1)
        max_pixels = float(args.generated_visible_max_silhouette_residual_px)
        if max_pixels > 0.0:
            diff = diff * np.minimum(
                1.0, max_pixels / np.maximum(norms, 1.0e-9)
            )[:, None]
        silhouette_block = (
            math.sqrt(float(args.generated_visible_silhouette_weight))
            * diff.reshape(-1)
            / float(args.generated_visible_sigma_silhouette_px)
            / math.sqrt(max(1, len(factor.silhouette_kind)))
        )
    return depth_block, silhouette_block


def joint_residual_blocks(
    core: Any,
    x: np.ndarray,
    key_nodes: list[Any],
    point_factors: list[Any],
    key_factors: list[Any],
    rgb_factors: list[Any],
    args: argparse.Namespace,
    observed_image_factors: dict[int, Any],
    generated_factors: dict[int, Any],
    all_nodes: list[Any],
    key_ids: list[int],
    initial_by_frame: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    blocks = core.residual_blocks(
        x,
        key_nodes,
        point_factors,
        key_factors,
        rgb_factors,
        args,
        observed_image_factors,
    )
    blocks["generated_first_hit"] = []
    blocks["generated_silhouette"] = []
    if not generated_factors:
        return blocks
    pose_by_frame, _rotations, _translations = expand_candidate_poses(
        core, key_nodes, x, all_nodes, key_ids, initial_by_frame
    )
    for frame_idx in sorted(generated_factors):
        if frame_idx not in pose_by_frame:
            raise RuntimeError(
                f"generated factor frame {frame_idx} has no interpolated pose"
            )
        factor = generated_factors[frame_idx]
        rotation, translation = pose_by_frame[frame_idx]
        depth, silhouette = generated_image_factor_blocks(
            core, factor, rotation, translation, args
        )
        if depth is not None:
            blocks["generated_first_hit"].append(depth)
        if silhouette is not None:
            blocks["generated_silhouette"].append(silhouette)
    return blocks


def joint_residual_sparsity(
    core: Any,
    key_nodes: list[Any],
    point_factors: list[Any],
    key_factors: list[Any],
    rgb_factors: list[Any],
    args: argparse.Namespace,
    observed_image_factors: dict[int, Any],
    generated_factors: dict[int, Any],
    key_ids: list[int],
) -> sparse.csr_matrix:
    base = core.residual_sparsity(
        key_nodes,
        point_factors,
        key_factors,
        rgb_factors,
        args,
        observed_image_factors,
    )
    if not generated_factors:
        return base
    key_pos = {int(node.frame_idx): pos for pos, node in enumerate(key_nodes)}
    entries: list[tuple[int, int]] = []
    row = 0

    def add(count: int, positions: tuple[int, ...]) -> None:
        nonlocal row
        for residual_row in range(row, row + int(count)):
            for position in positions:
                entries.extend(
                    (residual_row, column)
                    for column in range(6 * position, 6 * position + 6)
                )
        row += int(count)

    for frame_idx in sorted(generated_factors):
        factor = generated_factors[frame_idx]
        if float(args.generated_visible_first_hit_weight) > 0.0:
            add(
                len(factor.depth_observed_z),
                key_support_positions(frame_idx, key_ids, key_pos),
            )
    for frame_idx in sorted(generated_factors):
        factor = generated_factors[frame_idx]
        if float(args.generated_visible_silhouette_weight) > 0.0:
            add(
                2 * len(factor.silhouette_kind),
                key_support_positions(frame_idx, key_ids, key_pos),
            )
    if row == 0:
        return base
    rr, cc = np.asarray(entries, dtype=np.int64).T
    extra = sparse.csr_matrix(
        (np.ones(len(rr), dtype=bool), (rr, cc)),
        shape=(row, 6 * len(key_nodes)),
    )
    return sparse.vstack([base, extra], format="csr")


def parse_exact_gate_step_scales(value: str) -> list[float]:
    try:
        scales = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "exact-gate step scales must be comma-separated floats"
        ) from exc
    if not scales or any(not np.isfinite(scale) or scale <= 0.0 or scale > 1.0 for scale in scales):
        raise argparse.ArgumentTypeError(
            "exact-gate step scales must be nonempty values in (0,1]"
        )
    if any(scales[i] <= scales[i + 1] for i in range(len(scales) - 1)):
        raise argparse.ArgumentTypeError(
            "exact-gate step scales must be strictly descending"
        )
    return scales


def generated_metric_deltas(
    current: dict[str, Any],
    candidate: dict[str, Any],
    key_ids: list[int],
    current_factors: dict[int, Any] | None = None,
    candidate_factors: dict[int, Any] | None = None,
) -> dict[str, Any]:
    if current.get("mesh_sha256") != candidate.get("mesh_sha256"):
        raise RuntimeError("generated metric comparison used different mesh hashes")
    current_rows = {
        int(row["frame_idx"]): row for row in current.get("per_frame", [])
    }
    candidate_rows = {
        int(row["frame_idx"]): row for row in candidate.get("per_frame", [])
    }
    shared = sorted(set(current_rows).intersection(candidate_rows))
    if not shared:
        raise RuntimeError("generated metric comparison has no shared frames")

    def value(row: dict[str, Any], group: str, field: str) -> float:
        raw = (row.get(group) or {}).get(field)
        return float(raw) if raw is not None and np.isfinite(raw) else float("inf")

    if (current_factors is None) != (candidate_factors is None):
        raise RuntimeError(
            "generated common-hit comparison needs both current and candidate factors"
        )
    frame_rows = []
    global_common_before_abs: list[np.ndarray] = []
    global_common_after_abs: list[np.ndarray] = []
    for frame_idx in shared:
        before = current_rows[frame_idx]
        after = candidate_rows[frame_idx]
        support_median_before = value(
            before, "true_first_hit_abs_depth_m", "median"
        )
        support_median_after = value(
            after, "true_first_hit_abs_depth_m", "median"
        )
        support_p95_before = value(before, "true_first_hit_abs_depth_m", "p95")
        support_p95_after = value(after, "true_first_hit_abs_depth_m", "p95")
        if current_factors is not None and candidate_factors is not None:
            if frame_idx not in current_factors or frame_idx not in candidate_factors:
                raise RuntimeError(
                    f"generated common-hit comparison lacks frame {frame_idx}"
                )
            before_factor = current_factors[frame_idx]
            after_factor = candidate_factors[frame_idx]
            before_hit = np.asarray(before_factor.evaluation_hit, dtype=bool)
            after_hit = np.asarray(after_factor.evaluation_hit, dtype=bool)
            before_signed = np.asarray(
                before_factor.evaluation_signed_depth, dtype=np.float64
            )
            after_signed = np.asarray(
                after_factor.evaluation_signed_depth, dtype=np.float64
            )
            if not (
                before_hit.shape
                == after_hit.shape
                == before_signed.shape
                == after_signed.shape
            ):
                raise RuntimeError(
                    f"generated evaluation arrays disagree at frame {frame_idx}"
                )
            common_hit = (
                before_hit
                & after_hit
                & np.isfinite(before_signed)
                & np.isfinite(after_signed)
            )
            before_common_abs = np.abs(before_signed[common_hit])
            after_common_abs = np.abs(after_signed[common_hit])
            if len(before_common_abs):
                global_common_before_abs.append(before_common_abs)
                global_common_after_abs.append(after_common_abs)
            if len(before_common_abs) == 0:
                common_median_before = common_median_after = float("inf")
                common_p95_before = common_p95_after = float("inf")
            else:
                common_median_before = float(np.median(before_common_abs))
                common_median_after = float(np.median(after_common_abs))
                common_p95_before = float(np.percentile(before_common_abs, 95.0))
                common_p95_after = float(np.percentile(after_common_abs, 95.0))
            common_hit_count = int(np.count_nonzero(common_hit))
        else:
            common_median_before = support_median_before
            common_median_after = support_median_after
            common_p95_before = support_p95_before
            common_p95_after = support_p95_after
            common_hit_count = int(
                (before.get("true_first_hit_abs_depth_m") or {}).get("count") or 0
            )
        median_delta = (
            common_median_after - common_median_before
            if np.isfinite(common_median_before)
            and np.isfinite(common_median_after)
            else float("inf")
        )
        p95_delta = (
            common_p95_after - common_p95_before
            if np.isfinite(common_p95_before) and np.isfinite(common_p95_after)
            else float("inf")
        )
        before_coverage = before.get("true_first_hit_coverage_fraction")
        after_coverage = after.get("true_first_hit_coverage_fraction")
        if before_coverage is None or after_coverage is None:
            raise RuntimeError(
                "generated exact metrics lack true first-hit coverage at frame "
                f"{frame_idx}"
            )
        frame_rows.append(
            {
                "frame_idx": frame_idx,
                "support_abs_depth_median_before_m": support_median_before,
                "support_abs_depth_median_candidate_m": support_median_after,
                "support_abs_depth_p95_before_m": support_p95_before,
                "support_abs_depth_p95_candidate_m": support_p95_after,
                "common_hit_count": common_hit_count,
                "common_hit_abs_depth_median_before_m": common_median_before,
                "common_hit_abs_depth_median_candidate_m": common_median_after,
                "common_hit_abs_depth_median_delta_m": median_delta,
                "common_hit_abs_depth_p95_before_m": common_p95_before,
                "common_hit_abs_depth_p95_candidate_m": common_p95_after,
                "common_hit_abs_depth_p95_delta_m": p95_delta,
                "first_hit_coverage_before": float(before_coverage),
                "first_hit_coverage_candidate": float(after_coverage),
                "silhouette_iou_before": float(
                    before.get("initial_silhouette_iou") or 0.0
                ),
                "silhouette_iou_candidate": float(
                    after.get("initial_silhouette_iou") or 0.0
                ),
            }
        )
    segments = []
    for left, right in zip(key_ids[:-1], key_ids[1:]):
        rows = [
            row
            for row in frame_rows
            if int(left) <= int(row["frame_idx"]) <= int(right)
        ]
        if not rows:
            continue
        before_values = np.asarray(
            [row["common_hit_abs_depth_median_before_m"] for row in rows],
            dtype=np.float64,
        )
        after_values = np.asarray(
            [row["common_hit_abs_depth_median_candidate_m"] for row in rows],
            dtype=np.float64,
        )
        segments.append(
            {
                "left_keyframe": int(left),
                "right_keyframe": int(right),
                "frame_count": int(len(rows)),
                "median_common_hit_abs_depth_before_m": float(
                    np.median(before_values)
                ),
                "median_common_hit_abs_depth_candidate_m": float(
                    np.median(after_values)
                ),
                "median_common_hit_abs_depth_delta_m": float(
                    np.median(after_values) - np.median(before_values)
                ),
            }
        )
    current_abs = current.get("true_first_hit_abs_depth_m") or {}
    candidate_abs = candidate.get("true_first_hit_abs_depth_m") or {}

    def summary_delta(field: str) -> float:
        before = current_abs.get(field)
        after = candidate_abs.get(field)
        if before is None or after is None:
            return float("inf")
        before_value = float(before)
        after_value = float(after)
        if not np.isfinite(before_value) or not np.isfinite(after_value):
            return float("inf")
        return after_value - before_value

    support_global_median_delta = summary_delta("median")
    support_global_p95_delta = summary_delta("p95")
    if current_factors is not None and candidate_factors is not None:
        common_before = (
            np.concatenate(global_common_before_abs)
            if global_common_before_abs
            else np.empty(0, dtype=np.float64)
        )
        common_after = (
            np.concatenate(global_common_after_abs)
            if global_common_after_abs
            else np.empty(0, dtype=np.float64)
        )
        if len(common_before) == 0 or len(common_before) != len(common_after):
            global_common_median_delta = float("inf")
            global_common_p95_delta = float("inf")
            global_common_count = 0
        else:
            global_common_median_delta = float(
                np.median(common_after) - np.median(common_before)
            )
            global_common_p95_delta = float(
                np.percentile(common_after, 95.0)
                - np.percentile(common_before, 95.0)
            )
            global_common_count = int(len(common_before))
        global_comparison_support = "before_after_common_first_hits"
    else:
        global_common_median_delta = support_global_median_delta
        global_common_p95_delta = support_global_p95_delta
        global_common_count = int(current_abs.get("count") or 0)
        global_comparison_support = "support_specific_fallback_without_factor_arrays"
    return {
        "mesh_sha256": current.get("mesh_sha256"),
        "shared_frame_count": int(len(shared)),
        "global_common_hit_count": global_common_count,
        "global_depth_comparison_support": global_comparison_support,
        "global_abs_depth_median_delta_m": global_common_median_delta,
        "global_abs_depth_p95_delta_m": global_common_p95_delta,
        "global_support_abs_depth_median_delta_m": support_global_median_delta,
        "global_support_abs_depth_p95_delta_m": support_global_p95_delta,
        "max_frame_common_hit_abs_depth_median_delta_m": float(
            max(row["common_hit_abs_depth_median_delta_m"] for row in frame_rows)
        ),
        "max_frame_common_hit_abs_depth_p95_delta_m": float(
            max(row["common_hit_abs_depth_p95_delta_m"] for row in frame_rows)
        ),
        "min_frame_first_hit_coverage_delta": float(
            min(
                row["first_hit_coverage_candidate"]
                - row["first_hit_coverage_before"]
                for row in frame_rows
            )
        ),
        "min_frame_silhouette_iou_delta": float(
            min(
                row["silhouette_iou_candidate"]
                - row["silhouette_iou_before"]
                for row in frame_rows
            )
        ),
        "max_segment_common_hit_abs_depth_median_delta_m": float(
            max(
                (
                    row["median_common_hit_abs_depth_delta_m"]
                    for row in segments
                ),
                default=0.0,
            )
        ),
        "per_frame": frame_rows,
        "segments": segments,
    }


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
    parser.add_argument(
        "--full-timeline-pose-report",
        type=Path,
        default=None,
        help=(
            "Optional full-timeline pose baseline used only for frames without a "
            "direct metric node; such rows retain unresolved/completed provenance"
        ),
    )
    parser.add_argument("--pose-mesh", type=Path, required=True)
    parser.add_argument("--keyframe-edge-npz", type=Path, required=True)
    parser.add_argument("--keyframe-edge-json", type=Path, required=True)
    parser.add_argument("--rgb-edge-npz", type=Path, default=None)
    parser.add_argument("--image-factor-npz", type=Path, default=None)
    parser.add_argument(
        "--generated-visible-mesh",
        type=Path,
        default=None,
        help=(
            "Canonical generated mesh used by both dynamic true-first-hit pose "
            "factors and downstream render/QC"
        ),
    )
    parser.add_argument(
        "--require-generated-visible-factors",
        action="store_true",
        help="Fail unless the generated-mesh outer-loop factor path is active",
    )
    parser.add_argument(
        "--generated-visible-factor-module-script",
        type=Path,
        default=Path(__file__).resolve().parent / "v20_generated_first_hit_factors.py",
    )
    parser.add_argument(
        "--generated-visible-factor-builder-script",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "sam3d_native_ghost_lite_20260826"
            / "build_p15_first_hit_silhouette_factors.py"
        ),
    )
    parser.add_argument("--generated-visible-hand-npz", type=Path, default=None)
    parser.add_argument(
        "--generated-visible-mano-faces-pkl", type=Path, default=None
    )
    parser.add_argument("--generated-visible-device", default="cuda:0")
    parser.add_argument("--generated-visible-source-size", type=int, default=1408)
    parser.add_argument("--generated-visible-raster-size", type=int, default=256)
    parser.add_argument("--generated-visible-render-batch-size", type=int, default=16)
    parser.add_argument("--generated-visible-min-frames", type=int, default=8)
    parser.add_argument("--generated-visible-min-observed-points", type=int, default=20)
    parser.add_argument(
        "--generated-visible-min-ownership-fraction", type=float, default=0.80
    )
    parser.add_argument(
        "--generated-visible-hand-unknown-dilation-px", type=int, default=1
    )
    parser.add_argument(
        "--generated-visible-boundary-downweight-radius-px",
        type=float,
        default=1.5,
    )
    parser.add_argument("--generated-visible-boundary-weight", type=float, default=0.65)
    parser.add_argument(
        "--generated-visible-min-depth-factor-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--generated-visible-max-removed-fraction-for-full-weight",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--generated-visible-max-observed-factors-per-frame",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--generated-visible-max-outside-factors-per-frame",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--generated-visible-max-missing-factors-per-frame",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--generated-visible-first-hit-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--generated-visible-silhouette-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--generated-visible-sigma-depth-m", type=float, default=0.008
    )
    parser.add_argument(
        "--generated-visible-sigma-silhouette-px", type=float, default=4.0
    )
    parser.add_argument(
        "--generated-visible-max-depth-residual-m",
        type=float,
        default=0.10,
        help="Optimizer-only symmetric residual cap; exact acceptance remains uncapped",
    )
    parser.add_argument(
        "--generated-visible-max-silhouette-residual-px",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--max-generated-global-depth-median-degradation-m",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--max-generated-global-depth-p95-degradation-m",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--max-generated-segment-depth-median-degradation-m",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--max-generated-frame-depth-median-degradation-m",
        type=float,
        default=0.003,
    )
    parser.add_argument(
        "--max-generated-frame-depth-p95-degradation-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--max-generated-frame-first-hit-coverage-degradation",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--max-generated-frame-silhouette-iou-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--anchor-frame", type=int, default=0)
    parser.add_argument("--keyframe-interval", type=int, default=5)
    parser.add_argument("--max-keyframes", type=int, default=40)
    parser.add_argument("--outer-iterations", type=int, default=3)
    parser.add_argument(
        "--exact-gate-step-scales",
        type=parse_exact_gate_step_scales,
        default=parse_exact_gate_step_scales("1.0,0.5,0.25,0.1,0.05"),
        help=(
            "Descending fractions of the optimizer step to exact-rerender; "
            "the first candidate passing every generated/legacy gate is accepted"
        ),
    )
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
    parser.add_argument(
        "--keyframe-relative-weight-scale",
        type=float,
        default=0.0,
        help=(
            "Precomputed canonical-edge pose-composition factors remain disabled; "
            "dynamic inverse-pose canonical point factors replace them"
        ),
    )
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
    if args.require_generated_visible_factors and args.generated_visible_mesh is None:
        raise RuntimeError(
            "--require-generated-visible-factors needs --generated-visible-mesh"
        )
    if args.generated_visible_mesh is not None:
        if args.generated_visible_hand_npz is None:
            raise RuntimeError(
                "dynamic generated-visible factors need --generated-visible-hand-npz"
            )
        if args.generated_visible_mano_faces_pkl is None:
            raise RuntimeError(
                "dynamic generated-visible factors need --generated-visible-mano-faces-pkl"
            )
        if float(args.generated_visible_first_hit_weight) <= 0.0:
            raise RuntimeError(
                "generated-visible mode requires a positive first-hit weight"
            )
        if float(args.generated_visible_sigma_depth_m) <= 0.0:
            raise RuntimeError("generated-visible depth sigma must be positive")
        if float(args.generated_visible_sigma_silhouette_px) <= 0.0:
            raise RuntimeError("generated-visible silhouette sigma must be positive")
        if float(args.keyframe_relative_weight_scale) > 0.0:
            raise RuntimeError(
                "generated-visible mode forbids the old precomputed canonical-edge "
                "pose-composition residual; use dynamic inverse-pose point factors"
            )
        for name in (
            "generated_visible_boundary_weight",
            "generated_visible_min_depth_factor_weight",
        ):
            value = float(getattr(args, name))
            if not (0.0 < value <= 1.0):
                raise RuntimeError(f"--{name.replace('_', '-')} must be in (0,1]")

    core = import_v20_core(args.core_script)
    annotations = load_json(args.annotations)
    initial_report = load_json(args.pose_report)
    timeline_report = (
        load_json(args.full_timeline_pose_report)
        if args.full_timeline_pose_report is not None
        else initial_report
    )
    if (
        timeline_report.get("object_id") is not None
        and str(timeline_report.get("object_id")) != str(args.object_id)
    ):
        raise RuntimeError("full-timeline pose report object_id mismatch")
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
    generated_module = None
    generated_context = None
    if args.generated_visible_mesh is not None:
        generated_module = import_generated_factor_module(
            args.generated_visible_factor_module_script
        )
        generated_context = generated_module.create_context(args)
        if generated_context is None:
            raise RuntimeError("generated-visible factor context was not created")
    generated_metrics_before: dict[str, Any] | None = None
    generated_metrics_after: dict[str, Any] | None = None
    generated_factor_info_after: dict[str, Any] | None = None

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
        generated_factors: dict[int, Any] = {}
        generated_before_info: dict[str, Any] | None = None
        if generated_context is not None:
            current_pose_by_frame, _current_all_R, _current_all_t = (
                expand_candidate_poses(
                    core,
                    key_nodes,
                    x0,
                    all_nodes,
                    key_ids,
                    initial_by_frame,
                )
            )
            generated_factors, generated_before_info = (
                generated_module.rebuild_factors(
                    generated_context, current_pose_by_frame, args
                )
            )
            if generated_metrics_before is None:
                generated_metrics_before = generated_before_info
            print(
                f"[v20-kf] generated first-hit outer={outer + 1} "
                f"frames={generated_before_info['frame_count']} "
                f"depth={generated_before_info['depth_factor_count']} "
                f"silhouette={generated_before_info['silhouette_factor_count']}",
                flush=True,
            )
        before_blocks = joint_residual_blocks(
            core,
            x0,
            key_nodes,
            point_factors,
            key_factors,
            rgb_factors,
            args,
            image_factors,
            generated_factors,
            all_nodes,
            key_ids,
            initial_by_frame,
        )
        before = core.flatten_blocks(before_blocks)
        pattern = joint_residual_sparsity(
            core,
            key_nodes,
            point_factors,
            key_factors,
            rgb_factors,
            args,
            image_factors,
            generated_factors,
            key_ids,
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
            joint_residual_blocks(
                core,
                x,
                key_nodes,
                point_factors,
                key_factors,
                rgb_factors,
                args,
                image_factors,
                generated_factors,
                all_nodes,
                key_ids,
                initial_by_frame,
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
        before_cost = float(np.sum(before * before))
        current_R = [node.base_rotation.copy() for node in key_nodes]
        current_t = [node.base_translation.copy() for node in key_nodes]
        current_surface = mesh_metrics(
            core, key_nodes, current_R, current_t, mesh_points
        )
        current_all_R, current_all_t = expand_key_poses(current_R, current_t)
        current_mask = accepted_mask
        observed_proxy_mask_gate_active = generated_context is None
        required_drop = float(args.min_outer_cost_improvement) * max(before_cost, 1.0)

        def evaluate_trial(trial_x: np.ndarray, step_scale: float) -> dict[str, Any]:
            trial_blocks = joint_residual_blocks(
                core,
                trial_x,
                key_nodes,
                point_factors,
                key_factors,
                rgb_factors,
                args,
                image_factors,
                generated_factors,
                all_nodes,
                key_ids,
                initial_by_frame,
            )
            trial_residual = core.flatten_blocks(trial_blocks)
            trial_cost = float(np.sum(trial_residual * trial_residual))
            trial_R, trial_t = core.current_poses(key_nodes, trial_x)
            trial_surface = mesh_metrics(
                core, key_nodes, trial_R, trial_t, mesh_points
            )
            trial_all_R, trial_all_t = expand_key_poses(trial_R, trial_t)
            trial_generated_info: dict[str, Any] | None = None
            trial_generated_deltas: dict[str, Any] | None = None
            trial_generated_checks: dict[str, bool] = {}
            if generated_context is not None:
                trial_pose_by_frame = {
                    int(node.frame_idx): (trial_all_R[pos], trial_all_t[pos])
                    for pos, node in enumerate(all_nodes)
                }
                trial_generated_factors, trial_generated_info = (
                    generated_module.rebuild_factors(
                        generated_context, trial_pose_by_frame, args
                    )
                )
                assert generated_before_info is not None
                trial_generated_deltas = generated_metric_deltas(
                    generated_before_info,
                    trial_generated_info,
                    key_ids,
                    generated_factors,
                    trial_generated_factors,
                )
                trial_generated_checks = {
                    "global_depth_median": trial_generated_deltas[
                        "global_abs_depth_median_delta_m"
                    ]
                    <= float(args.max_generated_global_depth_median_degradation_m),
                    "global_depth_p95": trial_generated_deltas[
                        "global_abs_depth_p95_delta_m"
                    ]
                    <= float(args.max_generated_global_depth_p95_degradation_m),
                    "segment_depth_median": trial_generated_deltas[
                        "max_segment_common_hit_abs_depth_median_delta_m"
                    ]
                    <= float(args.max_generated_segment_depth_median_degradation_m),
                    "frame_depth_median": trial_generated_deltas[
                        "max_frame_common_hit_abs_depth_median_delta_m"
                    ]
                    <= float(args.max_generated_frame_depth_median_degradation_m),
                    "frame_depth_p95": trial_generated_deltas[
                        "max_frame_common_hit_abs_depth_p95_delta_m"
                    ]
                    <= float(args.max_generated_frame_depth_p95_degradation_m),
                    "frame_first_hit_coverage": trial_generated_deltas[
                        "min_frame_first_hit_coverage_delta"
                    ]
                    >= -float(
                        args.max_generated_frame_first_hit_coverage_degradation
                    ),
                    "frame_silhouette_iou": trial_generated_deltas[
                        "min_frame_silhouette_iou_delta"
                    ]
                    >= -float(
                        args.max_generated_frame_silhouette_iou_degradation
                    ),
                }
            trial_mask = mask_metrics_from_mesh(
                core,
                all_nodes,
                trial_all_R,
                trial_all_t,
                frames,
                mesh_vertices,
                mesh_faces,
                args,
            )
            current_iou = current_mask["iou"]["median"]
            trial_iou = trial_mask["iou"]["median"]
            current_centroid = current_mask["centroid_delta_px"]["median"]
            trial_centroid = trial_mask["centroid_delta_px"]["median"]
            mask_iou_delta = (
                float(trial_iou - current_iou)
                if current_iou is not None and trial_iou is not None
                else float("-inf")
            )
            mask_centroid_delta = (
                float(trial_centroid - current_centroid)
                if current_centroid is not None and trial_centroid is not None
                else float("inf")
            )
            mask_iou_mean_delta = (
                float(trial_mask["iou"].get("mean") - current_mask["iou"].get("mean"))
                if current_mask["iou"].get("mean") is not None
                and trial_mask["iou"].get("mean") is not None
                else float("-inf")
            )
            mask_centroid_mean_delta = (
                float(
                    trial_mask["centroid_delta_px"].get("mean")
                    - current_mask["centroid_delta_px"].get("mean")
                )
                if current_mask["centroid_delta_px"].get("mean") is not None
                and trial_mask["centroid_delta_px"].get("mean") is not None
                else float("inf")
            )
            current_med = current_surface["observed_to_mesh_m"]["median"]
            trial_med = trial_surface["observed_to_mesh_m"]["median"]
            surface_delta = (
                float(trial_med - current_med)
                if current_med is not None and trial_med is not None
                else float("inf")
            )
            cumulative_rot = max(
                float(
                    np.degrees(
                        np.linalg.norm(
                            Rotation.from_matrix(
                                trial_R[i] @ initial_key_R[i].T
                            ).as_rotvec()
                        )
                    )
                )
                for i in range(len(key_nodes))
            )
            cumulative_trans = max(
                float(np.linalg.norm(trial_t[i] - initial_key_t[i]))
                for i in range(len(key_nodes))
            )
            gate_failures = [
                ("optimizer_not_success", not bool(result.success)),
                (
                    "insufficient_cost_drop",
                    not (trial_cost <= before_cost - required_drop),
                ),
                (
                    "surface_degradation",
                    not (surface_delta <= float(args.max_surface_degradation_m)),
                ),
                (
                    "cumulative_rotation_bound",
                    not (
                        np.radians(cumulative_rot)
                        <= float(args.max_cumulative_rotation_rad)
                    ),
                ),
                (
                    "cumulative_translation_bound",
                    not (
                        cumulative_trans
                        <= float(args.max_cumulative_translation_m)
                    ),
                ),
                (
                    "mask_iou_degradation",
                    observed_proxy_mask_gate_active
                    and not (
                        mask_iou_delta >= -float(args.max_mask_iou_degradation)
                    ),
                ),
                (
                    "mask_iou_mean_degradation",
                    observed_proxy_mask_gate_active
                    and not (
                        mask_iou_mean_delta
                        >= -float(args.max_mask_iou_mean_degradation)
                    ),
                ),
                (
                    "mask_centroid_degradation",
                    observed_proxy_mask_gate_active
                    and not (
                        mask_centroid_delta
                        <= float(args.max_mask_centroid_degradation_px)
                    ),
                ),
                (
                    "mask_centroid_mean_degradation",
                    observed_proxy_mask_gate_active
                    and not (
                        mask_centroid_mean_delta
                        <= float(args.max_mask_centroid_mean_degradation_px)
                    ),
                ),
                *[
                    (f"generated_{name}_degradation", not passed)
                    for name, passed in trial_generated_checks.items()
                ],
            ]
            rejection_reasons = [
                reason for reason, failed in gate_failures if failed
            ]
            return {
                "step_scale": float(step_scale),
                "accepted": not rejection_reasons,
                "x": trial_x,
                "blocks": trial_blocks,
                "residual": trial_residual,
                "cost": trial_cost,
                "R": trial_R,
                "t": trial_t,
                "all_R": trial_all_R,
                "all_t": trial_all_t,
                "surface": trial_surface,
                "surface_delta": surface_delta,
                "mask": trial_mask,
                "mask_iou_delta": mask_iou_delta,
                "mask_iou_mean_delta": mask_iou_mean_delta,
                "mask_centroid_delta": mask_centroid_delta,
                "mask_centroid_mean_delta": mask_centroid_mean_delta,
                "generated_info": trial_generated_info,
                "generated_deltas": trial_generated_deltas,
                "generated_checks": trial_generated_checks,
                "cumulative_rotation_deg": cumulative_rot,
                "cumulative_translation_m": cumulative_trans,
                "rejection_reasons": rejection_reasons,
            }

        step_scales = (
            list(args.exact_gate_step_scales)
            if generated_context is not None
            else [1.0]
        )
        trial_states: list[dict[str, Any]] = []
        selected_trial: dict[str, Any] | None = None
        for step_scale in step_scales:
            trial = evaluate_trial(result.x * float(step_scale), float(step_scale))
            trial_states.append(trial)
            if trial["accepted"]:
                selected_trial = trial
                break
        chosen_trial = selected_trial or trial_states[0]
        accepted = selected_trial is not None
        selected_step_scale = (
            float(selected_trial["step_scale"])
            if selected_trial is not None
            else None
        )
        candidate_blocks = chosen_trial["blocks"]
        candidate = chosen_trial["residual"]
        candidate_cost = float(chosen_trial["cost"])
        candidate_R = chosen_trial["R"]
        candidate_t = chosen_trial["t"]
        candidate_all_R = chosen_trial["all_R"]
        candidate_all_t = chosen_trial["all_t"]
        candidate_surface = chosen_trial["surface"]
        surface_delta = float(chosen_trial["surface_delta"])
        candidate_mask = chosen_trial["mask"]
        mask_iou_delta = float(chosen_trial["mask_iou_delta"])
        mask_iou_mean_delta = float(chosen_trial["mask_iou_mean_delta"])
        mask_centroid_delta = float(chosen_trial["mask_centroid_delta"])
        mask_centroid_mean_delta = float(
            chosen_trial["mask_centroid_mean_delta"]
        )
        candidate_generated_info = chosen_trial["generated_info"]
        generated_deltas = chosen_trial["generated_deltas"]
        generated_gate_checks = chosen_trial["generated_checks"]
        cumulative_rot = float(chosen_trial["cumulative_rotation_deg"])
        cumulative_trans = float(chosen_trial["cumulative_translation_m"])
        rejection_reasons = (
            [] if accepted else list(chosen_trial["rejection_reasons"])
        )
        exact_gate_step_trials = [
            {
                "step_scale": float(trial["step_scale"]),
                "accepted": bool(trial["accepted"]),
                "cost": float(trial["cost"]),
                "residual_rms": float(
                    np.sqrt(np.mean(trial["residual"] * trial["residual"]))
                )
                if len(trial["residual"])
                else None,
                "surface_median_degradation_m": float(trial["surface_delta"]),
                "generated_gate_checks": trial["generated_checks"],
                "generated_metric_deltas": trial["generated_deltas"],
                "cumulative_rotation_deg": float(
                    trial["cumulative_rotation_deg"]
                ),
                "cumulative_translation_m": float(
                    trial["cumulative_translation_m"]
                ),
                "rejection_reasons": list(trial["rejection_reasons"]),
            }
            for trial in trial_states
        ]
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
                "generated_factor_info_before": generated_before_info,
                "generated_factor_info_candidate": candidate_generated_info,
                "generated_metric_deltas": generated_deltas,
                "generated_gate_checks": generated_gate_checks,
                "candidate_surface_median_degradation_m": surface_delta,
                "candidate_mask_iou_delta": mask_iou_delta,
                "candidate_mask_iou_mean_delta": mask_iou_mean_delta,
                "candidate_mask_centroid_delta_px": mask_centroid_delta,
                "candidate_mask_centroid_mean_delta_px": mask_centroid_mean_delta,
                "observed_proxy_mask_gate_active": observed_proxy_mask_gate_active,
                "current_mask": current_mask,
                "candidate_mask": candidate_mask,
                "candidate_cumulative_rotation_deg": cumulative_rot,
                "candidate_cumulative_translation_m": cumulative_trans,
                "selected_exact_gate_step_scale": selected_step_scale,
                "selected_local_correction_parameters_by_keyframe": (
                    [
                        {
                            "frame_idx": int(node.frame_idx),
                            "rotation_rotvec_rad": selected_trial["x"].reshape(-1, 6)[i, :3].astype(float).tolist(),
                            "left_translation_delta_m": selected_trial["x"].reshape(-1, 6)[i, 3:].astype(float).tolist(),
                        }
                        for i, node in enumerate(key_nodes)
                    ]
                    if selected_trial is not None
                    else None
                ),
                "exact_gate_step_trials": exact_gate_step_trials,
                "rejection_reasons": rejection_reasons,
            }
        )
        print(
            f"[v20-kf] outer {outer + 1}: success={result.success} "
            f"accepted={accepted} step_scale={selected_step_scale} nfev={result.nfev} "
            f"rms={outer_reports[-1]['residual_rms_before']:.5f}->"
            f"{outer_reports[-1]['residual_rms_after']:.5f} "
            f"surface_delta={surface_delta * 1000.0:.2f}mm",
            flush=True,
        )
        if accepted:
            accepted_mask = candidate_mask
            if candidate_generated_info is not None:
                generated_metrics_after = candidate_generated_info
                generated_factor_info_after = candidate_generated_info
            for i, node in enumerate(key_nodes):
                node.base_rotation = candidate_R[i]
                node.base_translation = candidate_t[i]
        else:
            if generated_metrics_after is None and generated_before_info is not None:
                generated_metrics_after = generated_before_info
                generated_factor_info_after = generated_before_info
            break
    final_key_R = {
        node.frame_idx: node.base_rotation.copy() for node in key_nodes
    }
    final_key_t = {
        node.frame_idx: node.base_translation.copy() for node in key_nodes
    }
    direct_initial_by_frame = {
        node.frame_idx: (
            np.asarray(
                node.source_row["rotation_world_from_completed_canonical_matrix"],
                dtype=np.float64,
            ),
            np.asarray(node.source_row["translation_world_m"], dtype=np.float64),
        )
        for node in all_nodes
    }
    timeline_initial_by_frame = pose_map_from_report(timeline_report)
    if args.full_timeline_pose_report is not None:
        missing_timeline_poses = sorted(set(frames).difference(timeline_initial_by_frame))
        if missing_timeline_poses:
            raise RuntimeError(
                "full-timeline pose report lacks annotation frames: "
                f"{missing_timeline_poses[:20]}"
            )
    # Direct P14 metric rows remain the correction reference wherever available;
    # the full-timeline report contributes only missing-observation baselines.
    timeline_initial_by_frame.update(direct_initial_by_frame)
    final_by_frame = {
        node.frame_idx: (
            final_key_R[node.frame_idx],
            final_key_t[node.frame_idx],
        )
        for node in key_nodes
    }
    timeline_frame_ids = sorted(timeline_initial_by_frame)
    timeline_pose_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    timeline_mode_by_frame: dict[int, str] = {}
    modes: dict[str, int] = {}
    for frame_idx in timeline_frame_ids:
        in_optimized_interval = bool(
            (args.frame_start is None or frame_idx >= int(args.frame_start))
            and (args.frame_end is None or frame_idx <= int(args.frame_end))
        )
        if in_optimized_interval:
            rotation, translation, mode = interpolate_pose_correction(
                frame_idx,
                timeline_frame_ids,
                key_ids,
                timeline_initial_by_frame,
                final_by_frame,
            )
            if frame_idx not in direct_initial_by_frame:
                mode = f"full_timeline_baseline_{mode}"
        else:
            rotation, translation = timeline_initial_by_frame[frame_idx]
            mode = "outside_optimized_interval_preserved"
        timeline_pose_by_frame[frame_idx] = (rotation, translation)
        timeline_mode_by_frame[frame_idx] = mode
        modes[mode] = modes.get(mode, 0) + 1

    final_R = [timeline_pose_by_frame[node.frame_idx][0] for node in all_nodes]
    final_t = [timeline_pose_by_frame[node.frame_idx][1] for node in all_nodes]
    final_mesh = mesh_metrics(core, all_nodes, final_R, final_t, mesh_points)
    initial_all_R = [direct_initial_by_frame[node.frame_idx][0] for node in all_nodes]
    initial_all_t = [direct_initial_by_frame[node.frame_idx][1] for node in all_nodes]
    initial_all_mesh = mesh_metrics(
        core, all_nodes, initial_all_R, initial_all_t, mesh_points
    )
    node_pos_all = {node.frame_idx: i for i, node in enumerate(all_nodes)}
    rows = []
    for original in timeline_report.get("pose_rows", []):
        if not isinstance(original, dict):
            rows.append(original)
            continue
        idx = int(original.get("frame_idx", -1))
        row = dict(original)
        if idx in timeline_pose_by_frame:
            rotation, translation = timeline_pose_by_frame[idx]
            direct_position = node_pos_all.get(idx)
            direct_node = (
                all_nodes[direct_position]
                if direct_position is not None
                else None
            )
            row["rotation_world_from_completed_canonical_matrix"] = rotation.astype(float).tolist()
            row["translation_world_m"] = translation.astype(float).tolist()
            correction_applied = (
                timeline_mode_by_frame[idx] != "outside_optimized_interval_preserved"
            )
            row["pose_source"] = (
                "full_timeline_baseline_preserved_outside_optimized_interval"
                if not correction_applied
                else "v20_keyframe_periodic_generated_first_hit"
                if direct_node is not None and generated_context is not None
                else "v20_keyframe_periodic_global_observed_only"
                if direct_node is not None
                else "v20_full_timeline_baseline_with_interpolated_keyframe_correction"
            )
            if direct_node is not None:
                row["direct_pose_observation_source"] = (
                    "generated_true_first_hit_plus_observed_canonical_graph"
                    if generated_context is not None
                    else "v20_keyframe_point_to_plane_graph"
                )
            row["generated_geometry_pose_evidence_consumed"] = bool(
                generated_context is not None and correction_applied
            )
            row["generated_geometry_pose_evidence_mode"] = (
                "direct_frame_factor"
                if direct_node is not None and generated_context is not None
                else "interpolated_neighbor_keyframe_correction"
                if generated_context is not None and correction_applied
                else "not_consumed"
            )
            row["observed_to_mesh_initial"] = initial_all_mesh["per_frame"].get(str(idx), {}).get("observed_to_mesh", {})
            row["observed_to_mesh_final"] = final_mesh["per_frame"].get(str(idx), {}).get("observed_to_mesh", {})
            row["mesh_to_observed_final"] = final_mesh["per_frame"].get(str(idx), {}).get("mesh_to_observed", {})
            row["v20_temporal_mode"] = timeline_mode_by_frame[idx]
            row["v20_uncertainty"] = {
                "keyframe": bool(direct_node.keyframe) if direct_node is not None else False,
                "depth_quality": float(direct_node.depth_quality) if direct_node is not None else None,
                "direct_metric_pose_observation": direct_node is not None,
                "generated_visible_factor_active": bool(
                    generated_context is not None and direct_node is not None
                ),
                "full_timeline_baseline_only": direct_node is None,
            }
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
        "schema": "v20_keyframe_periodic_generated_first_hit_pose_graph_v4",
        "status": "v20_keyframe_global_complete" if any(bool(r.get("accepted_update")) for r in outer_reports) else "v20_keyframe_global_rejected_or_incomplete",
        "annotation_ready": False,
        "diagnostic_only": True,
        "formal_state_modified": False,
        "claim_scope": (
            "Diagnostic keyframe SE(3) graph with all-frame interpolated generated-mesh "
            "true-first-hit and bidirectional silhouette factors rebuilt every outer "
            "pass when configured. Observed world surfels are compared only after "
            "candidate-pose inverse transport into the shared canonical frame. "
            "Generated geometry is visible pose/render evidence only and is never "
            "collision, contact, SDF, sign, or nonpenetration authority."
        ),
        "object_id": args.object_id,
        "inputs": {
            "annotations": str(args.annotations.resolve()),
            "initial_pose_report": str(args.pose_report.resolve()),
            "full_timeline_pose_report": str(args.full_timeline_pose_report.resolve()) if args.full_timeline_pose_report else None,
            "pose_mesh_observed_surface": str(args.pose_mesh.resolve()),
            "keyframe_edge_npz": str(args.keyframe_edge_npz.resolve()),
            "keyframe_edge_json": str(args.keyframe_edge_json.resolve()),
            "rgb_edge_npz": str(args.rgb_edge_npz.resolve()) if args.rgb_edge_npz else None,
            "image_factor_npz": str(args.image_factor_npz.resolve()) if args.image_factor_npz else None,
            "generated_visible_mesh": str(generated_context.mesh_path) if generated_context is not None else None,
            "generated_visible_factor_module": str(args.generated_visible_factor_module_script.resolve()) if generated_context is not None else None,
            "generated_visible_factor_builder": str(args.generated_visible_factor_builder_script.resolve()) if generated_context is not None else None,
            "generated_visible_hand_npz": str(args.generated_visible_hand_npz.resolve()) if generated_context is not None else None,
            "generated_visible_mano_faces_pkl": str(args.generated_visible_mano_faces_pkl.resolve()) if generated_context is not None else None,
            "generated_geometry_consumed": bool(generated_context is not None),
            "sha256": {
                "annotations": core.sha256_file(args.annotations),
                "initial_pose_report": core.sha256_file(args.pose_report),
                "full_timeline_pose_report": core.sha256_file(args.full_timeline_pose_report) if args.full_timeline_pose_report else None,
                "pose_mesh_observed_surface": core.sha256_file(args.pose_mesh),
                "keyframe_edge_npz": core.sha256_file(args.keyframe_edge_npz),
                "keyframe_edge_json": core.sha256_file(args.keyframe_edge_json),
                "rgb_edge_npz": core.sha256_file(args.rgb_edge_npz) if args.rgb_edge_npz else None,
                "image_factor_npz": core.sha256_file(args.image_factor_npz) if args.image_factor_npz else None,
                "generated_visible_mesh": generated_context.mesh_sha256 if generated_context is not None else None,
                "generated_visible_factor_module": core.sha256_file(args.generated_visible_factor_module_script) if generated_context is not None else None,
                "generated_visible_factor_builder": core.sha256_file(args.generated_visible_factor_builder_script) if generated_context is not None else None,
                "generated_visible_hand_npz": core.sha256_file(args.generated_visible_hand_npz) if generated_context is not None else None,
                "generated_visible_mano_faces_pkl": core.sha256_file(args.generated_visible_mano_faces_pkl) if generated_context is not None else None,
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
            "generated_correspondences_rebuilt_each_outer_iteration": bool(generated_context is not None),
            "generated_all_visible_frames_constrain_neighboring_keyframes": bool(generated_context is not None),
            "periodic_reanchor": True,
        },
        "factors": {
            "keyframe_point_factor_group_count": len(point_factors) if "point_factors" in locals() else 0,
            "keyframe_relative_edges": len(key_factors),
            "rgb_absolute_or_relative_edges": len(rgb_factors),
            "image_factor_keyframe_count": len(image_factors),
            "image_depth_factor_count": image_depth_count,
            "image_silhouette_factor_count": image_silhouette_count,
            "generated_visible_frame_count": int((generated_factor_info_after or {}).get("frame_count", 0)),
            "generated_visible_depth_factor_count": int((generated_factor_info_after or {}).get("depth_factor_count", 0)),
            "generated_visible_silhouette_factor_count": int((generated_factor_info_after or {}).get("silhouette_factor_count", 0)),
            "generated_geometry_consumed": bool(generated_context is not None),
            "generated_geometry_authority": "visible_pose_and_render_only",
        },
        "parameters": {
            "inner_max_nfev": int(args.inner_max_nfev),
            "optimizer_mode": str(args.optimizer_mode),
            "optimizer_ftol": float(args.optimizer_ftol),
            "optimizer_xtol": float(args.optimizer_xtol),
            "optimizer_gtol": float(args.optimizer_gtol),
            "exact_gate_step_scales": [float(value) for value in args.exact_gate_step_scales],
            "exact_gate_backtracking_active": bool(generated_context is not None),
            "image_first_hit_weight": float(args.image_first_hit_weight),
            "image_silhouette_weight": float(args.image_silhouette_weight),
            "max_mask_iou_degradation": float(args.max_mask_iou_degradation),
            "max_mask_iou_mean_degradation": float(args.max_mask_iou_mean_degradation),
            "max_mask_centroid_degradation_px": float(args.max_mask_centroid_degradation_px),
            "max_mask_centroid_mean_degradation_px": float(args.max_mask_centroid_mean_degradation_px),
            "observed_proxy_mask_gate_active": bool(generated_context is None),
            "observed_proxy_mask_metrics_diagnostic_only_when_generated_active": bool(generated_context is not None),
            "max_surface_degradation_m": float(args.max_surface_degradation_m),
            "generated_visible_first_hit_weight": float(args.generated_visible_first_hit_weight),
            "generated_visible_silhouette_weight": float(args.generated_visible_silhouette_weight),
            "generated_visible_sigma_depth_m": float(args.generated_visible_sigma_depth_m),
            "generated_visible_sigma_silhouette_px": float(args.generated_visible_sigma_silhouette_px),
            "generated_visible_max_depth_residual_m_optimizer_only": float(args.generated_visible_max_depth_residual_m),
            "generated_visible_exact_acceptance_uncapped": True,
            "generated_acceptance_limits": {
                "global_abs_depth_median_degradation_m": float(args.max_generated_global_depth_median_degradation_m),
                "global_abs_depth_p95_degradation_m": float(args.max_generated_global_depth_p95_degradation_m),
                "segment_abs_depth_median_degradation_m": float(args.max_generated_segment_depth_median_degradation_m),
                "frame_abs_depth_median_degradation_m": float(args.max_generated_frame_depth_median_degradation_m),
                "frame_abs_depth_p95_degradation_m": float(args.max_generated_frame_depth_p95_degradation_m),
                "frame_first_hit_coverage_degradation": float(args.max_generated_frame_first_hit_coverage_degradation),
                "frame_silhouette_iou_degradation": float(args.max_generated_frame_silhouette_iou_degradation),
            },
        },
        "optimizer_outer_iterations": outer_reports,
        "temporal": {
            "direct_frame_count": len(all_nodes),
            "timeline_frame_count": len(frames),
            "output_pose_frame_count": len(timeline_pose_by_frame),
            "full_timeline_baseline_frame_count": int(
                len(set(timeline_pose_by_frame).difference(node_pos_all))
            ),
            "full_timeline_pose_report_consumed": bool(args.full_timeline_pose_report),
            "direct_fraction": len(all_nodes) / max(1, len(frames)),
            "output_mode_counts": modes,
            "rotation_step_deg": core.numeric_summary(rot_steps),
            "translation_step_m": core.numeric_summary(trans_steps),
            "correction_is_applied_periodically_at_keyframes": True,
            "all_visible_generated_factors_depend_on_neighboring_keyframes": bool(generated_context is not None),
        },
        "generated_visible_surface_alignment": {
            "active": bool(generated_context is not None),
            "correspondences_rebuilt_each_outer_iteration": bool(generated_context is not None),
            "all_visible_frames_in_objective": bool(generated_context is not None),
            "mesh_path": str(generated_context.mesh_path) if generated_context is not None else None,
            "mesh_sha256": generated_context.mesh_sha256 if generated_context is not None else None,
            "metrics_before": generated_metrics_before,
            "metrics_after": generated_metrics_after,
            "final_factor_info": generated_factor_info_after,
            "authority": "visible_pose_and_render_only_not_collision_contact_sdf_or_sign",
        },
        "render_mesh_contract": {
            "required_mesh_path": str(generated_context.mesh_path) if generated_context is not None else None,
            "required_mesh_sha256": generated_context.mesh_sha256 if generated_context is not None else None,
            "renderer_must_match": bool(generated_context is not None),
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
