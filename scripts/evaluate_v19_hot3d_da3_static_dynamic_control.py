#!/usr/bin/env python3
"""Evaluator-only static/dynamic control for a frozen UniDepth/DA3 pair.

Released HOT3D poses and rendered foreground depth/labels are consumed only
after prediction archives have been frozen by a prediction-side acceptance.
This evaluator never writes into the prediction root and is intentionally not
part of the prediction runtime bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEMA = "v19_hot3d_da3_static_dynamic_control_evaluation_v3"
STATIC_TRANSLATION_MAX_M = 0.003
STATIC_ROTATION_MAX_DEG = 2.0
ACTIVE_TRANSLATION_MIN_M = 0.020
ACTIVE_ROTATION_MIN_DEG = 5.0
ACTIVE_OBJECT_CAMERA_RATIO_MIN = 1.5
INACTIVE_TRANSLATION_MAX_M = 0.002
INACTIVE_ROTATION_MAX_DEG = 1.0
DEFAULT_DEPTH_BIN_M = 0.050
DEFAULT_RADIUS_BIN_PX = 128.0
DEFAULT_MAX_SAMPLES_PER_CELL = 4096


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, role: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {role}: {path}")
    return path


def asset(path: Path, role: str) -> dict[str, Any]:
    path = require_file(path, role)
    return {
        "role": role,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def summarize(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def max_pairwise_baseline(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)))


def rotation_extent_deg(rotations: np.ndarray) -> float:
    rotations = np.asarray(rotations, dtype=np.float64)
    if len(rotations) < 2:
        return 0.0
    relative = np.einsum("ji,njk->nik", rotations[0], rotations)
    return float(np.max(np.degrees(Rotation.from_matrix(relative).magnitude())))


def classify_motion(translation_m: float, rotation_deg: float, camera_baseline_m: float) -> str:
    ratio = translation_m / max(camera_baseline_m, 1.0e-9)
    if translation_m <= INACTIVE_TRANSLATION_MAX_M and rotation_deg <= INACTIVE_ROTATION_MAX_DEG:
        return "motion_inactive"
    if (
        (translation_m >= ACTIVE_TRANSLATION_MIN_M or rotation_deg >= ACTIVE_ROTATION_MIN_DEG)
        and ratio >= ACTIVE_OBJECT_CAMERA_RATIO_MIN
    ):
        return "motion_active"
    return "motion_intermediate"


def exclusive_evaluation_ranges(windows: list[dict[str, Any]], frame_count: int) -> list[tuple[int, int]]:
    """Assign each frame to one DA3 context window without overlap duplication."""
    starts = [int(window["frame_start"]) for window in windows]
    if not starts or starts[0] != 0 or starts != sorted(set(starts)):
        raise RuntimeError("DA3 context windows must have unique ordered starts beginning at zero")
    ranges = [
        (start, starts[index + 1] - 1 if index + 1 < len(starts) else int(frame_count) - 1)
        for index, start in enumerate(starts)
    ]
    assigned = [frame for start, end in ranges for frame in range(start, end + 1)]
    if assigned != list(range(int(frame_count))):
        raise RuntimeError("exclusive DA3 evaluation ranges do not partition the timeline exactly")
    return ranges


def cluster_active_rows_by_window(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average active objects inside each unique window before pooled inference."""
    clusters: list[dict[str, Any]] = []
    for window_index in sorted({int(row["window_index"]) for row in rows}):
        selected = [
            row for row in rows
            if int(row["window_index"]) == window_index and row["motion_class"] == "motion_active"
        ]
        if not selected:
            continue
        clusters.append({
            "window_index": window_index,
            "active_object_count": len(selected),
            "active_objects": [str(row["dynamic_object"]) for row in selected],
            "mean_da3_dynamic_static_excess_absolute_log_ratio": float(np.mean([
                row["da3_dynamic_static_excess_absolute_log_ratio"] for row in selected
            ])),
            "mean_difference_in_differences_absolute_log_ratio": float(np.mean([
                row["difference_in_differences_absolute_log_ratio"] for row in selected
            ])),
        })
    return clusters


def deterministic_subsample(indices: np.ndarray, count: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    count = int(count)
    if count < 0 or count > len(indices):
        raise ValueError("invalid deterministic subsample count")
    if count == len(indices):
        return indices
    if count == 0:
        return indices[:0]
    positions = np.floor((np.arange(count, dtype=np.float64) + 0.5) * len(indices) / count).astype(np.int64)
    return indices[positions]


def metric_means(prediction: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(gt) & (prediction > 0.0) & (gt > 0.0)
    if not np.any(valid):
        raise RuntimeError("matched cell contains no valid metric-depth samples")
    prediction = prediction[valid]
    gt = gt[valid]
    ratio = prediction / gt
    return {
        "absolute_log_ratio": float(np.mean(np.abs(np.log(ratio)))),
        "signed_log_ratio": float(np.mean(np.log(ratio))),
        "absolute_relative_error": float(np.mean(np.abs(prediction - gt) / gt)),
        "absolute_error_m": float(np.mean(np.abs(prediction - gt))),
        "prediction_over_gt_ratio": float(np.mean(ratio)),
    }


def matched_cell_rows(
    *,
    frame_idx: int,
    dynamic_object: str,
    dynamic_label: int,
    static_labels: list[int],
    labels: np.ndarray,
    gt_depth_m: np.ndarray,
    predictions: dict[str, np.ndarray],
    principal_point_xy: tuple[float, float],
    depth_bin_m: float,
    radius_bin_px: float,
    max_samples_per_cell: int,
) -> list[dict[str, Any]]:
    """Match dynamic/static pixels using GT-only frame/depth/radius cells."""
    labels = np.asarray(labels)
    gt_depth_m = np.asarray(gt_depth_m, dtype=np.float64)
    height, width = labels.shape
    if gt_depth_m.shape != labels.shape:
        raise RuntimeError("GT depth and labels have different rasters")
    for provider, prediction in predictions.items():
        if np.asarray(prediction).shape != labels.shape:
            raise RuntimeError(f"{provider} prediction raster differs from GT")
    valid_prediction = np.ones(labels.shape, dtype=bool)
    for prediction in predictions.values():
        prediction = np.asarray(prediction)
        valid_prediction &= np.isfinite(prediction) & (prediction > 0.0)
    valid = np.isfinite(gt_depth_m) & (gt_depth_m > 0.0) & valid_prediction
    dynamic = valid & (labels == int(dynamic_label))
    static = valid & np.isin(labels, np.asarray(static_labels, dtype=labels.dtype))
    if not np.any(dynamic) or not np.any(static):
        return []

    ys, xs = np.indices((height, width), dtype=np.float64)
    cx, cy = [float(x) for x in principal_point_xy]
    radius = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    z_bin = np.floor(gt_depth_m / float(depth_bin_m)).astype(np.int32)
    r_bin = np.floor(radius / float(radius_bin_px)).astype(np.int32)
    flat_dynamic = np.flatnonzero(dynamic.ravel())
    dynamic_keys = np.stack((z_bin.ravel()[flat_dynamic], r_bin.ravel()[flat_dynamic]), axis=1)
    rows: list[dict[str, Any]] = []
    for z_value, r_value in np.unique(dynamic_keys, axis=0):
        cell = valid & (z_bin == int(z_value)) & (r_bin == int(r_value))
        dynamic_indices = np.flatnonzero((cell & (labels == int(dynamic_label))).ravel())
        static_indices = np.flatnonzero((cell & np.isin(labels, static_labels)).ravel())
        count = min(len(dynamic_indices), len(static_indices), int(max_samples_per_cell))
        if count <= 0:
            continue
        dynamic_indices = deterministic_subsample(dynamic_indices, count)
        static_indices = deterministic_subsample(static_indices, count)
        flat_gt = gt_depth_m.ravel()
        row: dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "dynamic_object": dynamic_object,
            "gt_depth_bin": int(z_value),
            "radial_bin": int(r_value),
            "sample_count_per_role": int(count),
            "available_dynamic_pixels": int(len(np.flatnonzero((cell & (labels == int(dynamic_label))).ravel()))),
            "available_static_pixels": int(len(np.flatnonzero((cell & np.isin(labels, static_labels)).ravel()))),
            "gt_dynamic_depth_median_m": float(np.median(flat_gt[dynamic_indices])),
            "gt_static_depth_median_m": float(np.median(flat_gt[static_indices])),
            "metrics": {},
        }
        for provider, prediction in predictions.items():
            flat_prediction = np.asarray(prediction, dtype=np.float64).ravel()
            row["metrics"][provider] = {
                "dynamic": metric_means(flat_prediction[dynamic_indices], flat_gt[dynamic_indices]),
                "static": metric_means(flat_prediction[static_indices], flat_gt[static_indices]),
            }
        rows.append(row)
    return rows


def weighted_metric(cells: list[dict[str, Any]], provider: str, role: str, metric: str) -> float:
    weights = np.asarray([int(row["sample_count_per_role"]) for row in cells], dtype=np.float64)
    values = np.asarray([float(row["metrics"][provider][role][metric]) for row in cells], dtype=np.float64)
    if not len(weights) or np.sum(weights) <= 0:
        raise RuntimeError("cannot aggregate empty matched cells")
    return float(np.sum(weights * values) / np.sum(weights))


def aggregate_window(cells: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "matched_cell_count": len(cells),
        "matched_sample_count_per_role": int(sum(int(row["sample_count_per_role"]) for row in cells)),
        "providers": {},
    }
    for provider in ("unidepth", "depth_anything_3"):
        dynamic = weighted_metric(cells, provider, "dynamic", "absolute_log_ratio")
        static = weighted_metric(cells, provider, "static", "absolute_log_ratio")
        result["providers"][provider] = {
            "dynamic_absolute_log_ratio": dynamic,
            "static_absolute_log_ratio": static,
            "dynamic_minus_static_absolute_log_ratio": dynamic - static,
            "dynamic_absolute_relative_error": weighted_metric(
                cells, provider, "dynamic", "absolute_relative_error"
            ),
            "static_absolute_relative_error": weighted_metric(
                cells, provider, "static", "absolute_relative_error"
            ),
            "dynamic_signed_log_ratio": weighted_metric(cells, provider, "dynamic", "signed_log_ratio"),
            "static_signed_log_ratio": weighted_metric(cells, provider, "static", "signed_log_ratio"),
        }
    result["difference_in_differences_absolute_log_ratio"] = (
        result["providers"]["depth_anything_3"]["dynamic_minus_static_absolute_log_ratio"]
        - result["providers"]["unidepth"]["dynamic_minus_static_absolute_log_ratio"]
    )
    return result


def exact_sign_flip_p_greater(values: Iterable[float]) -> float | None:
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0 or len(values) > 20:
        return None
    observed = float(np.mean(values))
    exceed = 0
    total = 1 << len(values)
    for bits in range(total):
        signs = np.asarray([1.0 if bits & (1 << i) else -1.0 for i in range(len(values))])
        exceed += int(float(np.mean(values * signs)) >= observed - 1.0e-15)
    return float(exceed / total)


def exact_partition_p_greater(active: Iterable[float], inactive: Iterable[float]) -> float | None:
    active = np.asarray(list(active), dtype=np.float64)
    inactive = np.asarray(list(inactive), dtype=np.float64)
    values = np.concatenate((active, inactive))
    if not len(active) or not len(inactive) or len(values) > 20:
        return None
    observed = float(np.mean(active) - np.mean(inactive))
    exceed = 0
    total = 0
    for active_indices in itertools.combinations(range(len(values)), len(active)):
        chosen = np.zeros(len(values), dtype=bool)
        chosen[list(active_indices)] = True
        statistic = float(np.mean(values[chosen]) - np.mean(values[~chosen]))
        exceed += int(statistic >= observed - 1.0e-15)
        total += 1
    return float(exceed / total)


def bootstrap_mean_ci(values: Iterable[float], seed: int, draws: int = 10000) -> list[float] | None:
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(draws), len(values)))
    means = np.mean(values[indices], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    x = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"count": int(len(x)), "pearson": None, "spearman": None}
    return {
        "count": int(len(x)),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def render_timeline_review(rows: list[dict[str, Any]], output_path: Path) -> None:
    names = sorted({str(row["dynamic_object"]) for row in rows})
    figure, axes = plt.subplots(len(names), 2, figsize=(14, 4.5 * len(names)), squeeze=False)
    colors = {"motion_active": "#d62728", "motion_intermediate": "#ffbf00", "motion_inactive": "#2ca02c"}
    for row_index, name in enumerate(names):
        selected = sorted(
            (row for row in rows if row["dynamic_object"] == name),
            key=lambda row: int(row["window_index"]),
        )
        x = np.asarray([int(row["window_index"]) for row in selected])
        point_colors = [colors[str(row["motion_class"])] for row in selected]
        axes[row_index, 0].plot(x, [row["object_over_camera_translation_ratio"] for row in selected], color="0.35")
        axes[row_index, 0].scatter(x, [row["object_over_camera_translation_ratio"] for row in selected], c=point_colors, s=55)
        axes[row_index, 0].axhline(ACTIVE_OBJECT_CAMERA_RATIO_MIN, color="black", ls="--", lw=1)
        axes[row_index, 0].set_ylabel("object / camera baseline")
        axes[row_index, 0].set_title(f"{name}: GT motion class (red active, green inactive)")
        axes[row_index, 1].axhline(0.0, color="black", lw=1)
        axes[row_index, 1].plot(
            x, [row["da3_dynamic_static_excess_absolute_log_ratio"] for row in selected],
            marker="o", label="DA3 dynamic - matched static",
        )
        axes[row_index, 1].plot(
            x, [row["difference_in_differences_absolute_log_ratio"] for row in selected],
            marker="s", label="DA3 excess - UniDepth excess",
        )
        axes[row_index, 1].set_ylabel("absolute log-ratio error excess")
        axes[row_index, 1].set_title(f"{name}: unblended single-window matched error")
        axes[row_index, 1].legend(loc="best")
        for column in range(2):
            axes[row_index, column].set_xlabel("DA3 context window index")
            axes[row_index, column].grid(alpha=0.25)
    figure.suptitle("HOT3D P0002 static/dynamic control — frozen DA3 Nested vs UniDepth", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def analyze_window_rows(
    window_rows: list[dict[str, Any]], dynamic_names: list[str], object_names: list[str], seed: int
) -> dict[str, Any]:
    object_results: dict[str, Any] = {}
    for name in dynamic_names:
        rows = [row for row in window_rows if row["dynamic_object"] == name]
        active = [row for row in rows if row["motion_class"] == "motion_active"]
        inactive = [row for row in rows if row["motion_class"] == "motion_inactive"]
        active_excess = [row["da3_dynamic_static_excess_absolute_log_ratio"] for row in active]
        inactive_excess = [row["da3_dynamic_static_excess_absolute_log_ratio"] for row in inactive]
        active_did = [row["difference_in_differences_absolute_log_ratio"] for row in active]
        inactive_did = [row["difference_in_differences_absolute_log_ratio"] for row in inactive]
        contrast = None
        if active and inactive:
            contrast = {
                "statistic": "mean(active) - mean(inactive) over disjoint frame-set window estimates",
                "da3_dynamic_static_excess_contrast_absolute_log_ratio": float(
                    np.mean(active_excess) - np.mean(inactive_excess)
                ),
                "da3_excess_exact_label_partition_one_sided_p_greater": exact_partition_p_greater(
                    active_excess, inactive_excess
                ),
                "provider_difference_in_differences_contrast_absolute_log_ratio": float(
                    np.mean(active_did) - np.mean(inactive_did)
                ),
                "difference_in_differences_exact_label_partition_one_sided_p_greater": exact_partition_p_greater(
                    active_did, inactive_did
                ),
            }
        object_results[name] = {
            "window_count": len(rows),
            "motion_active_window_count": len(active),
            "motion_inactive_window_count": len(inactive),
            "motion_intermediate_window_count": len(rows) - len(active) - len(inactive),
            "active_da3_dynamic_static_excess_absolute_log_ratio": summarize(active_excess),
            "inactive_da3_dynamic_static_excess_absolute_log_ratio": summarize(inactive_excess),
            "active_difference_in_differences_absolute_log_ratio": summarize(active_did),
            "active_da3_excess_exact_sign_flip_one_sided_p_greater_zero": exact_sign_flip_p_greater(active_excess),
            "active_da3_excess_window_bootstrap_mean_95pct_ci": bootstrap_mean_ci(
                active_excess, seed + object_names.index(name)
            ),
            "motion_active_vs_inactive_natural_control": contrast,
            "correlations_all_windows": {
                "da3_excess_vs_object_camera_ratio": correlation(
                    rows, "object_over_camera_translation_ratio", "da3_dynamic_static_excess_absolute_log_ratio"
                ),
                "difference_in_differences_vs_object_camera_ratio": correlation(
                    rows, "object_over_camera_translation_ratio", "difference_in_differences_absolute_log_ratio"
                ),
                "da3_excess_vs_object_translation": correlation(
                    rows, "gt_object_translation_baseline_m", "da3_dynamic_static_excess_absolute_log_ratio"
                ),
            },
        }

    active_rows = [row for row in window_rows if row["motion_class"] == "motion_active"]
    active_window_clusters = cluster_active_rows_by_window(window_rows)
    pooled_excess = [
        row["mean_da3_dynamic_static_excess_absolute_log_ratio"] for row in active_window_clusters
    ]
    pooled_did = [
        row["mean_difference_in_differences_absolute_log_ratio"] for row in active_window_clusters
    ]
    natural_controls = [
        result["motion_active_vs_inactive_natural_control"]
        for result in object_results.values()
        if result["motion_active_vs_inactive_natural_control"] is not None
    ]
    natural_supported = any(
        row["da3_dynamic_static_excess_contrast_absolute_log_ratio"] > 0
        and row["provider_difference_in_differences_contrast_absolute_log_ratio"] > 0
        and row["da3_excess_exact_label_partition_one_sided_p_greater"] is not None
        and row["da3_excess_exact_label_partition_one_sided_p_greater"] <= 0.05
        and row["difference_in_differences_exact_label_partition_one_sided_p_greater"] is not None
        and row["difference_in_differences_exact_label_partition_one_sided_p_greater"] <= 0.05
        for row in natural_controls
    )
    pooled_excess_p = exact_sign_flip_p_greater(pooled_excess)
    pooled_did_p = exact_sign_flip_p_greater(pooled_did)
    pooled_supported = (
        len(pooled_excess) >= 3
        and float(np.mean(pooled_excess)) > 0
        and float(np.mean(pooled_did)) > 0
        and pooled_excess_p is not None
        and pooled_excess_p <= 0.05
        and pooled_did_p is not None
        and pooled_did_p <= 0.05
    )
    if natural_supported and pooled_supported:
        verdict = "supported_dynamic_motion_is_a_context_dependent_da3_nested_error_contributor"
    elif natural_supported or pooled_supported:
        verdict = "partially_supported_dynamic_motion_contributes_but_is_not_sufficient"
    else:
        verdict = "not_supported_by_this_static_dynamic_control"
    pooled = {
        "active_object_window_row_count": len(active_rows),
        "unique_window_cluster_count": len(active_window_clusters),
        "cluster_definition": (
            "equal-weight mean across motion-active dynamic objects inside each unique DA3 context window"
        ),
        "window_clusters": active_window_clusters,
        "da3_dynamic_static_excess_absolute_log_ratio": summarize(pooled_excess),
        "difference_in_differences_absolute_log_ratio": summarize(pooled_did),
        "da3_excess_exact_window_sign_flip_one_sided_p_greater_zero": pooled_excess_p,
        "difference_in_differences_exact_window_sign_flip_one_sided_p_greater_zero": pooled_did_p,
        "da3_excess_window_cluster_bootstrap_mean_95pct_ci": bootstrap_mean_ci(pooled_excess, seed),
        "difference_in_differences_window_cluster_bootstrap_mean_95pct_ci": bootstrap_mean_ci(
            pooled_did, seed + 1
        ),
    }
    return {
        "verdict": verdict,
        "per_dynamic_object": object_results,
        "pooled_motion_active": pooled,
        "natural_controls": natural_controls,
        "decision_rule": {
            "natural_control_supported": natural_supported,
            "pooled_supported": pooled_supported,
        },
    }


def build_window_rows(
    *,
    all_cells: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    evaluation_frames_by_window: list[list[int]],
    dynamic_names: list[str],
    object_names: list[str],
    world_object: np.ndarray,
    camera_centers: np.ndarray,
    evaluation_frame_role: str,
) -> list[dict[str, Any]]:
    if len(windows) != len(evaluation_frames_by_window):
        raise RuntimeError("window/evaluation-frame plan length differs")
    rows: list[dict[str, Any]] = []
    for window, evaluation_frames in zip(windows, evaluation_frames_by_window, strict=True):
        start = int(window["frame_start"])
        end = int(window["frame_end"])
        evaluation_set = {int(frame) for frame in evaluation_frames}
        if not evaluation_set or min(evaluation_set) < start or max(evaluation_set) > end:
            raise RuntimeError("evaluation frames are empty or outside their DA3 context")
        camera_baseline = max_pairwise_baseline(camera_centers[start:end + 1])
        for name in dynamic_names:
            object_position = object_names.index(name)
            translation = max_pairwise_baseline(world_object[start:end + 1, object_position, :3, 3])
            rotation = rotation_extent_deg(world_object[start:end + 1, object_position, :3, :3])
            cells = [
                row for row in all_cells
                if row["dynamic_object"] == name and int(row["frame_idx"]) in evaluation_set
            ]
            if not cells:
                continue
            aggregate = aggregate_window(cells)
            da3_excess = aggregate["providers"]["depth_anything_3"]["dynamic_minus_static_absolute_log_ratio"]
            unidepth_excess = aggregate["providers"]["unidepth"]["dynamic_minus_static_absolute_log_ratio"]
            rows.append({
                "window_index": int(window["window_index"]),
                "context_frames": [start, end],
                "evaluation_frame_role": evaluation_frame_role,
                "evaluation_frames": sorted(evaluation_set),
                "dynamic_object": name,
                "motion_class": classify_motion(translation, rotation, camera_baseline),
                "gt_camera_baseline_m": camera_baseline,
                "gt_object_translation_baseline_m": translation,
                "gt_object_rotation_extent_deg": rotation,
                "object_over_camera_translation_ratio": translation / max(camera_baseline, 1.0e-9),
                **aggregate,
                "da3_dynamic_static_excess_absolute_log_ratio": da3_excess,
                "unidepth_dynamic_static_excess_absolute_log_ratio": unidepth_excess,
            })
    return rows


def validate_prediction_binding(
    acceptance_path: Path,
    unidepth_path: Path,
    da3_path: Path,
    da3_qc_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    acceptance_path = require_file(acceptance_path, "prediction acceptance")
    acceptance = load_json(acceptance_path)
    if acceptance.get("status") != "accepted_prediction_frozen_for_evaluator_only_static_dynamic_test":
        raise RuntimeError("prediction acceptance is not evaluator-ready")
    if (acceptance.get("runtime") or {}).get("sam3d_invoked") is not False:
        raise RuntimeError("prediction acceptance does not preserve SAM3D exclusion")
    expected = acceptance.get("assets") or {}
    checks = {
        "unidepth": (unidepth_path, "unidepth"),
        "da3_camera_bound": (da3_path, "da3_camera_bound"),
        "qc": (da3_qc_path, "qc"),
    }
    bindings: dict[str, Any] = {}
    for name, (path, key) in checks.items():
        path = require_file(path, name)
        row = expected.get(key)
        if not isinstance(row, dict):
            raise RuntimeError(f"prediction acceptance lacks {key}")
        digest = sha256_file(path)
        if path != Path(str(row.get("path"))).resolve() or digest != str(row.get("sha256")):
            raise RuntimeError(f"{name} differs from frozen prediction acceptance")
        bindings[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
    qc = load_json(da3_qc_path)
    overlap = ((qc.get("windowing") or {}).get("overlap_consistency") or {})
    if (
        qc.get("status") != "ok"
        or qc.get("frame_count") != 150
        or overlap.get("passed") is not True
        or (qc.get("pose_conditioning") or {}).get("metric_scale_mode") != "nested_metric_branch"
    ):
        raise RuntimeError("DA3 QC is not downstream-eligible Nested metric depth")
    return acceptance, bindings


def directory_digest(paths: list[Path]) -> dict[str, Any]:
    rows = []
    combined = hashlib.sha256()
    for path in sorted(paths):
        digest = sha256_file(path)
        combined.update(bytes.fromhex(digest))
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
    return {
        "file_count": len(rows),
        "ordered_digest_of_per_file_sha256": combined.hexdigest(),
        "digest_definition": "sha256(concat(raw 32-byte per-file SHA256 digests in lexical filename order))",
        "files": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-acceptance", type=Path, required=True)
    parser.add_argument("--unidepth-depth", type=Path, required=True)
    parser.add_argument("--da3-depth", type=Path, required=True)
    parser.add_argument("--da3-qc", type=Path, required=True)
    parser.add_argument("--gt-state-npz", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--foreground-depth-dir", type=Path, required=True)
    parser.add_argument("--foreground-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-bin-m", type=float, default=DEFAULT_DEPTH_BIN_M)
    parser.add_argument("--radius-bin-px", type=float, default=DEFAULT_RADIUS_BIN_PX)
    parser.add_argument("--max-samples-per-cell", type=int, default=DEFAULT_MAX_SAMPLES_PER_CELL)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()
    if args.depth_bin_m <= 0 or args.radius_bin_px <= 0 or args.max_samples_per_cell <= 0:
        parser.error("matching bins and max samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"fresh evaluator output directory required: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    acceptance, prediction_bindings = validate_prediction_binding(
        args.prediction_acceptance, args.unidepth_depth, args.da3_depth, args.da3_qc
    )
    gt_state_path = require_file(args.gt_state_npz, "HOT3D GT state")
    label_map_path = require_file(args.label_map, "foreground label map")
    label_map = load_json(label_map_path)
    depth_paths = sorted(args.foreground_depth_dir.resolve().glob("*.png"))
    label_paths = sorted(args.foreground_label_dir.resolve().glob("*.png"))
    if len(depth_paths) != 150 or len(label_paths) != 150:
        raise RuntimeError("rendered foreground GT must contain exactly 150 depth and label PNGs")

    with np.load(gt_state_path, allow_pickle=False) as archive:
        required = {
            "frame_idx", "K", "T_world_from_camera", "object_names", "object_uids",
            "T_world_from_object", "object_pose_available",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"GT state misses {missing}")
        frame_idx = np.asarray(archive["frame_idx"], dtype=np.int32)
        gt_K = np.asarray(archive["K"], dtype=np.float64)
        world_camera = np.asarray(archive["T_world_from_camera"], dtype=np.float64)
        object_names = [str(x) for x in archive["object_names"]]
        object_uids = [str(x) for x in archive["object_uids"]]
        world_object = np.asarray(archive["T_world_from_object"], dtype=np.float64)
        object_available = np.asarray(archive["object_pose_available"], dtype=bool)
    if not np.array_equal(frame_idx, np.arange(150, dtype=np.int32)):
        raise RuntimeError("GT state is not the complete ordered 150-frame timeline")
    if world_camera.shape != (150, 4, 4) or world_object.shape != (150, len(object_names), 4, 4):
        raise RuntimeError("GT camera/object transform shapes are invalid")
    if not np.all(object_available):
        raise RuntimeError("static/dynamic control requires complete released object poses")

    label_by_name = {str(row["object_name"]): int(row["label_id"]) for row in label_map.get("objects") or []}
    if set(label_by_name) != set(object_names):
        raise RuntimeError("foreground label map and GT object state names differ")
    with np.load(args.unidepth_depth, allow_pickle=False) as archive:
        unidepth_frame = np.asarray(archive["frame_idx"], dtype=np.int32)
        unidepth = np.asarray(archive["depth"])
        unidepth_K = np.asarray(archive["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    with np.load(args.da3_depth, allow_pickle=False) as archive:
        da3_frame = np.asarray(archive["frame_idx"], dtype=np.int32)
        da3 = np.asarray(archive["depth"])
        da3_K = np.asarray(archive["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        provider = str(np.asarray(archive["depth_provider"]).item())
        scale_mode = str(np.asarray(archive["metric_scale_mode"]).item())
        overlap_passed = bool(np.asarray(archive["overlap_consistency_passed"]).item())
        contribution_count = np.asarray(archive["window_contribution_count"], dtype=np.int32)
    if provider != "depth_anything_3" or scale_mode != "nested_metric_branch" or not overlap_passed:
        raise RuntimeError("DA3 archive provenance is not eligible for the static/dynamic control")
    if (
        unidepth.shape != (150, 1408, 1408)
        or da3.shape != (150, 1408, 1408)
        or not np.array_equal(unidepth_frame, frame_idx)
        or not np.array_equal(da3_frame, frame_idx)
        or contribution_count.shape != (150,)
        or not np.isin(contribution_count, np.asarray([1, 2], dtype=np.int32)).all()
    ):
        raise RuntimeError("prediction archives do not match the evaluator timeline/raster")
    expected_intrinsics = np.asarray([gt_K[0, 0], gt_K[1, 1], gt_K[0, 2], gt_K[1, 2]])
    if (
        unidepth_K.shape != (150, 4)
        or da3_K.shape != (150, 4)
        or np.max(np.abs(unidepth_K - expected_intrinsics)) > 1.0e-12
        or np.max(np.abs(da3_K - expected_intrinsics)) > 1.0e-12
    ):
        raise RuntimeError("official-K closure failed between predictions and evaluator GT raster")

    object_motion: dict[str, Any] = {}
    static_names: list[str] = []
    for position, name in enumerate(object_names):
        translation = max_pairwise_baseline(world_object[:, position, :3, 3])
        rotation = rotation_extent_deg(world_object[:, position, :3, :3])
        is_static = translation <= STATIC_TRANSLATION_MAX_M and rotation <= STATIC_ROTATION_MAX_DEG
        object_motion[name] = {
            "object_uid": object_uids[position],
            "full_timeline_world_translation_baseline_m": translation,
            "full_timeline_rotation_extent_deg": rotation,
            "static_control_eligible": is_static,
        }
        if is_static:
            static_names.append(name)
    dynamic_names = [name for name in object_names if name not in static_names]
    if len(static_names) < 2 or not dynamic_names:
        raise RuntimeError("GT motion rules did not produce both static controls and dynamic candidates")
    static_labels = [label_by_name[name] for name in static_names]

    all_cells: list[dict[str, Any]] = []
    principal_point = (float(gt_K[0, 2]), float(gt_K[1, 2]))
    for position in range(150):
        labels = cv2.imread(str(label_paths[position]), cv2.IMREAD_UNCHANGED)
        depth_mm = cv2.imread(str(depth_paths[position]), cv2.IMREAD_UNCHANGED)
        if labels is None or depth_mm is None or labels.shape != (1408, 1408) or depth_mm.shape != labels.shape:
            raise RuntimeError(f"invalid rendered GT raster at frame {position}")
        gt_depth_m = np.asarray(depth_mm, dtype=np.float64) / 1000.0
        predictions = {"unidepth": unidepth[position], "depth_anything_3": da3[position]}
        for name in dynamic_names:
            all_cells.extend(matched_cell_rows(
                frame_idx=position,
                dynamic_object=name,
                dynamic_label=label_by_name[name],
                static_labels=static_labels,
                labels=labels,
                gt_depth_m=gt_depth_m,
                predictions=predictions,
                principal_point_xy=principal_point,
                depth_bin_m=float(args.depth_bin_m),
                radius_bin_px=float(args.radius_bin_px),
                max_samples_per_cell=int(args.max_samples_per_cell),
            ))

    qc = load_json(args.da3_qc)
    windows = (qc.get("windowing") or {}).get("windows") or []
    if len(windows) != 13:
        raise RuntimeError("expected the fixed 13-window DA3 plan")
    camera_centers = world_camera[:, :3, 3]
    evaluation_ranges = exclusive_evaluation_ranges(windows, 150)
    coverage_frames_by_window = [
        list(range(start, end + 1)) for start, end in evaluation_ranges
    ]
    unblended_frames_by_window = [
        [frame for frame in range(int(window["frame_start"]), int(window["frame_end"]) + 1)
         if int(contribution_count[frame]) == 1]
        for window in windows
    ]
    unblended_flat = [frame for frames in unblended_frames_by_window for frame in frames]
    expected_unblended = np.flatnonzero(contribution_count == 1).tolist()
    if sorted(unblended_flat) != expected_unblended or len(unblended_flat) != len(set(unblended_flat)):
        raise RuntimeError("unblended frame assignment does not bind each contribution-count-one frame once")
    primary_window_rows = build_window_rows(
        all_cells=all_cells,
        windows=windows,
        evaluation_frames_by_window=unblended_frames_by_window,
        dynamic_names=dynamic_names,
        object_names=object_names,
        world_object=world_object,
        camera_centers=camera_centers,
        evaluation_frame_role="primary_unblended_window_contribution_count_equals_one",
    )
    coverage_window_rows = build_window_rows(
        all_cells=all_cells,
        windows=windows,
        evaluation_frames_by_window=coverage_frames_by_window,
        dynamic_names=dynamic_names,
        object_names=object_names,
        world_object=world_object,
        camera_centers=camera_centers,
        evaluation_frame_role="coverage_sensitivity_all_150_frames_exclusive_assignment_includes_overlap_averages",
    )
    primary_analysis = analyze_window_rows(
        primary_window_rows, dynamic_names, object_names, args.seed
    )
    coverage_analysis = analyze_window_rows(
        coverage_window_rows, dynamic_names, object_names, args.seed
    )
    verdict = primary_analysis["verdict"]

    report = {
        "schema": SCHEMA,
        "status": "ok_evaluator_only_frozen_predictions",
        "verdict": verdict,
        "study_design_status": (
            "exploratory_case_selected_then_v2_plan_executed_then_v3_primary_refined_to_exclude_overlap_"
            "averaged_frames_not_independent_preregistration"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Camera-conditioned DA3 Nested incurs target-relative metric-depth error when foreground object "
            "motion violates the static multi-view assumption; motion should increase error relative to "
            "same-frame static rigid controls after GT-only depth/radius matching."
        ),
        "discriminating_predictions_fixed_before_v2_execution_and_retained_for_stricter_v3": {
            "if_supported": [
                "Motion-active dynamic objects have positive DA3 dynamic-minus-static absolute log-ratio error.",
                "The DA3 excess is larger than the equivalent UniDepth excess (positive difference-in-differences).",
                "For an object with both active and inactive windows, excess decreases when the same object stops moving.",
            ],
            "if_not_sufficient": (
                "Some dynamic objects remain accurate, showing that motion is a contributor conditioned on object, "
                "appearance, occlusion, and window context rather than a sufficient failure condition."
            ),
            "if_rejected": (
                "Matched dynamic excess is non-positive and same-object error does not decrease in motion-inactive windows."
            ),
        },
        "prediction_binding": {
            "acceptance": asset(args.prediction_acceptance, "prediction-side acceptance created before GT evaluation"),
            "archives": prediction_bindings,
            "prediction_mutated": False,
            "sam3d_invoked": False,
        },
        "evaluator_assets": {
            "gt_state": asset(gt_state_path, "released HOT3D evaluator-only camera/object state"),
            "label_map": asset(label_map_path, "rendered foreground label contract"),
            "foreground_depth": directory_digest(depth_paths),
            "foreground_labels": directory_digest(label_paths),
        },
        "camera_contract_closure": {
            "official_K": expected_intrinsics.tolist(),
            "unidepth_max_abs_error": float(np.max(np.abs(unidepth_K - expected_intrinsics))),
            "da3_max_abs_error": float(np.max(np.abs(da3_K - expected_intrinsics))),
        },
        "motion_rules_fixed_before_prediction_error_comparison": {
            "static_control": {
                "full_timeline_translation_baseline_max_m": STATIC_TRANSLATION_MAX_M,
                "full_timeline_rotation_extent_max_deg": STATIC_ROTATION_MAX_DEG,
            },
            "motion_active_window": {
                "translation_baseline_min_m_or_rotation_extent_min_deg": [
                    ACTIVE_TRANSLATION_MIN_M, ACTIVE_ROTATION_MIN_DEG
                ],
                "object_over_camera_translation_ratio_min": ACTIVE_OBJECT_CAMERA_RATIO_MIN,
            },
            "motion_inactive_window": {
                "translation_baseline_max_m": INACTIVE_TRANSLATION_MAX_M,
                "rotation_extent_max_deg": INACTIVE_ROTATION_MAX_DEG,
            },
        },
        "matching": {
            "uses_prediction_depth_or_error_for_matching": False,
            "prediction_provenance_used_only_for_primary_frame_eligibility": "DA3 window_contribution_count == 1",
            "pixel_match_keys": ["exact frame_idx", "GT camera-Z depth bin", "GT radial-distance bin"],
            "depth_bin_m": float(args.depth_bin_m),
            "radius_bin_px": float(args.radius_bin_px),
            "max_samples_per_cell_per_role": int(args.max_samples_per_cell),
            "sampling": "equal dynamic/static counts; deterministic evenly spaced raster-order subsampling",
            "all_150_frame_matched_cell_count": len(all_cells),
            "all_150_frame_matched_sample_count_per_role": int(
                sum(row["sample_count_per_role"] for row in all_cells)
            ),
            "primary_unblended_matched_cell_count": int(sum(
                int(row["frame_idx"]) in set(expected_unblended) for row in all_cells
            )),
            "primary_unblended_matched_sample_count_per_role": int(sum(
                int(row["sample_count_per_role"])
                for row in all_cells if int(row["frame_idx"]) in set(expected_unblended)
            )),
        },
        "frame_eligibility": {
            "primary": {
                "rule": "frozen DA3 window_contribution_count == 1; excludes all overlap-averaged frames",
                "frame_count": len(expected_unblended),
                "frame_indices": expected_unblended,
                "evaluation_frames_by_context_window": unblended_frames_by_window,
                "each_eligible_frame_assigned_once": True,
            },
            "coverage_sensitivity": {
                "rule": "all 150 frames assigned once by [context_start,next_context_start-1]",
                "frame_count": 150,
                "exclusive_evaluation_ranges": [list(row) for row in evaluation_ranges],
            },
            "frozen_window_contribution_count": {
                str(value): int(np.sum(contribution_count == value))
                for value in sorted(np.unique(contribution_count).tolist())
            },
        },
        "object_motion": object_motion,
        "static_control_objects": static_names,
        "dynamic_candidate_objects": dynamic_names,
        "primary_metric": {
            "name": "absolute_log_ratio_error",
            "formula": "abs(log(predicted_camera_z_m / GT_camera_z_m))",
            "reason": "dimensionless symmetric scale error avoids favoring nearer surfaces in meter-error comparisons",
        },
        "primary_unblended_analysis": {
            "frame_rule": "DA3 window_contribution_count == 1",
            **primary_analysis,
            "window_rows": primary_window_rows,
        },
        "coverage_sensitivity_all_150_frames": {
            "frame_rule": "exclusive assignment includes overlap-averaged prediction frames",
            **coverage_analysis,
            "window_rows": coverage_window_rows,
        },
        "matched_cells": all_cells,
        "interpretation_boundaries": [
            "Released poses/depth/labels are evaluator-only and no GT-derived value is written into prediction state.",
            "The two prediction providers are compared on identical pixels; matching itself reads GT only.",
            "The primary analysis excludes every overlap-averaged DA3 frame and binds each remaining frame to its one contributing context window.",
            "The all-150-frame analysis is coverage sensitivity only; it assigns each frame once but includes overlap-averaged predictions.",
            "Objects share one clip; pooled inference first averages objects within each unique window cluster.",
            "The clip and matching design were selected after exploratory inspection, and unblended-only primary eligibility was introduced after auditing v2 overlap provenance. Exact randomization values are descriptive sensitivity analyses, not confirmatory preregistered p-values or dataset-wide inference.",
            "Motion is tested as a contributor, not claimed as a sufficient condition for every object or context.",
            "Static/dynamic objects differ in appearance and geometry; same-object active/inactive natural control is the stronger causal channel.",
            "Best scalar rescaling is not applied and no evaluator-derived calibration is returned to either provider.",
        ],
        "decision_rule": {
            "supported": (
                "At least one same-object active/inactive natural control has positive DA3-excess and provider-"
                "difference-in-differences contrasts, both with descriptive exact one-sided partition p<=0.05; "
                "unique-window pooled active clusters also have positive DA3 excess and positive provider "
                "difference-in-differences, both with descriptive sign-flip p<=0.05."
            ),
            "partially_supported": "Only the natural-control or pooled criterion is met.",
            "not_supported": "Neither criterion is met.",
            "natural_control_supported": primary_analysis["decision_rule"]["natural_control_supported"],
            "pooled_supported": primary_analysis["decision_rule"]["pooled_supported"],
            "verdict_uses_primary_unblended_analysis_only": True,
        },
    }
    review_path = args.output_dir / "static_dynamic_window_timeline_review.png"
    render_timeline_review(primary_window_rows, review_path)
    report["outputs"] = {
        "timeline_review": asset(review_path, "static/dynamic motion and matched-error timeline QC")
    }
    report_path = args.output_dir / "hot3d_da3_static_dynamic_control_evaluation.json"
    write_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "verdict": verdict,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "static_controls": static_names,
        "dynamic_candidates": dynamic_names,
        "all_matched_cells": len(all_cells),
        "primary_unblended_frame_count": len(expected_unblended),
        "primary_matched_samples_per_role": report["matching"]["primary_unblended_matched_sample_count_per_role"],
        "pooled": primary_analysis["pooled_motion_active"],
        "natural_controls": primary_analysis["natural_controls"],
    }, indent=2))


if __name__ == "__main__":
    main()
