#!/usr/bin/env python3
"""Summarize frozen-parameter cross-video SAM3D/P15 alignment runs.

This is a diagnostic aggregator, not an optimizer. It compares native-pose
contract evidence, sampled shared-Sim(3) before/after metrics, full-timeline
true-raster metrics, and P15 temporal motion statistics without category-
specific branches or thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
        "p90": float(np.percentile(array, 90.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def signed_median(row: dict[str, Any]) -> float:
    depth = row.get("first_hit_minus_observed_depth_m") or {}
    return float(depth.get("median_m", np.nan))


def temporal_metrics(pose_rows: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    rows = sorted(pose_rows, key=lambda row: int(row["frame_idx"]))
    rotations = np.asarray([row["rotation_world_from_completed_canonical_matrix"] for row in rows], dtype=np.float64)
    translations = np.asarray([row["translation_world_m"] for row in rows], dtype=np.float64)
    relative_rotation = np.einsum("nij,njk->nik", np.transpose(rotations[:-1], (0, 2, 1)), rotations[1:])
    rotation_step_deg = np.rad2deg(Rotation.from_matrix(relative_rotation).magnitude())
    translation_step_m = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    translation_accel_m = np.linalg.norm(np.diff(translations, n=2, axis=0), axis=1)
    relative_vectors = Rotation.from_matrix(relative_rotation).as_rotvec()
    rotation_accel_deg = np.rad2deg(np.linalg.norm(np.diff(relative_vectors, axis=0), axis=1))
    return {
        "fps": fps,
        "translation_step_m_per_frame": summary(translation_step_m),
        "translation_velocity_m_per_s": summary(translation_step_m * fps),
        "translation_second_difference_m": summary(translation_accel_m),
        "rotation_step_deg_per_frame": summary(rotation_step_deg),
        "rotation_velocity_deg_per_s": summary(rotation_step_deg * fps),
        "rotation_step_change_deg": summary(rotation_accel_deg),
    }


def metric_rows(rows: dict[str, Any], section: str, metric: str) -> np.ndarray:
    result = []
    for frame_idx in sorted((int(key) for key in rows)):
        value = rows[str(frame_idx)][section].get(metric)
        result.append(float(value) if value is not None else np.nan)
    return np.asarray(result, dtype=np.float64)


def segment_summary(rows: dict[str, Any], section: str) -> list[dict[str, Any]]:
    output = []
    indices = sorted(int(key) for key in rows)
    for start in range(0, max(indices) + 1, 30):
        active = [idx for idx in indices if start <= idx <= start + 29]
        if not active:
            continue
        output.append({
            "frame_range": [start, start + 29],
            "iou": summary([rows[str(idx)][section]["iou"] for idx in active]),
            "centroid_error_px": summary([rows[str(idx)][section]["centroid_error_px"] for idx in active]),
            "symmetric_boundary_median_px": summary([
                rows[str(idx)][section]["symmetric_boundary_median_px"] for idx in active
            ]),
        })
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    cases = []
    for case in config["cases"]:
        name = str(case["name"])
        case_dir = args.benchmark_root / name
        native = load_json(case_dir / "native_first_hit_audit" / "sam3d_native_pose_contract_audit.json")
        sim3 = load_json(case_dir / "shared_sim3_bounded_v4_frozen" / "qc_sam3d_p15_first_hit_alignment.json")
        video_before = load_json(case_dir / "before_shared_sim3" / "qc_optimized_object_full_video.json")
        video_after = load_json(case_dir / "after_shared_sim3" / "qc_optimized_object_full_video.json")
        pose = load_json(Path(case["run_root"]) / "experiments/sam3d_trellis_controlled/P15_observed_pose_graph/v19_rigid_object_pose_graph_report.json")
        manifest_report = load_json(Path(case["run_root"]) / "input/raw_frame_manifest/v19_raw_frame_manifest_report.json")
        before = sim3["frame_metrics_before"]
        after = sim3["frame_metrics_after"]
        before_iou = [float(row["silhouette_iou"]) for row in before.values()]
        after_iou = [float(row["silhouette_iou"]) for row in after.values()]
        before_depth = [abs(signed_median(row)) for row in before.values()]
        after_depth = [abs(signed_median(row)) for row in after.values()]
        per_frame_before = video_before["per_frame"]
        per_frame_after = video_after["per_frame"]
        section = "hand_occluded_render_vs_owned_mask"
        iou_before = metric_rows(per_frame_before, section, "iou")
        iou_after = metric_rows(per_frame_after, section, "iou")
        centroid_before = metric_rows(per_frame_before, section, "centroid_error_px")
        centroid_after = metric_rows(per_frame_after, section, "centroid_error_px")
        coverage_before = metric_rows(per_frame_before, section, "observed_coverage")
        coverage_after = metric_rows(per_frame_after, section, "observed_coverage")
        outside_before = metric_rows(per_frame_before, section, "rendered_outside_fraction")
        outside_after = metric_rows(per_frame_after, section, "rendered_outside_fraction")
        boundary_before = metric_rows(per_frame_before, section, "symmetric_boundary_median_px")
        boundary_after = metric_rows(per_frame_after, section, "symmetric_boundary_median_px")
        worst = np.argsort(np.where(np.isfinite(iou_after), iou_after, np.inf))[:10]
        cases.append({
            "name": name,
            "object_id": case["object_id"],
            "geometry_challenge": case["geometry_challenge"],
            "anchor_frame": int(case["anchor_frame"]),
            "native_contract": {
                "status": native["status"],
                "native_opencv_iou": native["verdict"]["native_opencv_iou"],
                "raw_local_iou": native["verdict"]["raw_local_iou"],
                "wrong_axis_iou": native["verdict"]["wrong_axis_iou"],
                "first_hit_metric_scale": native["metric_scene_similarity"]["camera_origin_uniform_scale"],
            },
            "shared_sim3": {
                "status": sim3["status"],
                "used_frames": sim3["used_frames"],
                "increment": sim3["sim3_increment"],
                "sampled_iou_before": summary(before_iou),
                "sampled_iou_after": summary(after_iou),
                "sampled_abs_first_hit_median_m_before": summary(before_depth),
                "sampled_abs_first_hit_median_m_after": summary(after_depth),
            },
            "full_timeline": {
                "frame_count": len(per_frame_after),
                "before_shared_sim3": {
                    "hand_occluded_iou": summary(iou_before),
                    "observed_coverage": summary(coverage_before),
                    "rendered_outside_fraction": summary(outside_before),
                    "hand_occluded_centroid_error_px": summary(centroid_before),
                    "symmetric_boundary_median_px": summary(boundary_before),
                    "segments": segment_summary(per_frame_before, section),
                },
                "after_shared_sim3": {
                    "hand_occluded_iou": summary(iou_after),
                    "observed_coverage": summary(coverage_after),
                    "rendered_outside_fraction": summary(outside_after),
                    "hand_occluded_centroid_error_px": summary(centroid_after),
                    "symmetric_boundary_median_px": summary(boundary_after),
                    "segments": segment_summary(per_frame_after, section),
                },
                "per_frame_iou_delta_after_minus_before": summary(iou_after - iou_before),
                "frames_improved_iou": int(np.count_nonzero(iou_after > iou_before)),
                "frames_degraded_iou": int(np.count_nonzero(iou_after < iou_before)),
                "worst_after_iou_frames": [
                    {
                        "frame_idx": int(idx),
                        "iou_before": float(iou_before[idx]),
                        "iou_after": float(iou_after[idx]),
                        "centroid_error_px_after": float(centroid_after[idx]),
                    }
                    for idx in worst if np.isfinite(iou_after[idx])
                ],
            },
            "p15_temporal": temporal_metrics(pose["pose_rows"], float(manifest_report["fps"])),
            "artifacts": {
                "before_overlay_video": video_before["outputs"]["camera_overlay_video"],
                "before_side_by_side_video": video_before["outputs"]["side_by_side_video"],
                "after_overlay_video": video_after["outputs"]["camera_overlay_video"],
                "after_side_by_side_video": video_after["outputs"]["side_by_side_video"],
                "before_video_qc": str(case_dir / "before_shared_sim3" / "qc_optimized_object_full_video.json"),
                "after_video_qc": str(case_dir / "after_shared_sim3" / "qc_optimized_object_full_video.json"),
            },
        })
    output = {
        "schema": "sam3d_native_ghost_lite_cross_video_benchmark_summary_v1",
        "diagnostic_only": True,
        "annotation_ready": False,
        "selection_policy": config["selection_policy"],
        "parameter_policy": config["parameter_policy"],
        "parameters": config["parameters"],
        "cases": cases,
        "claim_scope": (
            "Frozen-parameter cross-video evaluation of a shared canonical Sim(3) under existing P15 per-frame poses. "
            "It does not evaluate a per-frame pose-refinement solver and gives generated hidden faces no physical authority."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
