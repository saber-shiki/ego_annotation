#!/usr/bin/env python3
"""Compare full-video GHOST render QC under the corrected mask contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BASE = Path(
    "/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/"
    "backend_tests/20260819T122452Z_hot3d_milk_local_authority_local29_v1/runs/"
    "P0014_84ea2dcc_carton_milk_f2370_2519"
)
GHOST = BASE / "experiments/sam3d_native_ghost_lite_20260826"
CF = GHOST / "depth_order_counterfactual_masks_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-mesh-corrected-mask",
        type=Path,
        default=CF / "baseline_ghost_mesh_with_front_only_masks_full_video/qc_optimized_object_full_video.json",
    )
    parser.add_argument(
        "--counterfactual-mesh-corrected-mask",
        type=Path,
        default=CF / "ghost_front_only_p15_first_hit_depth_authority_v1/full_video_correct_intrinsics/qc_optimized_object_full_video.json",
    )
    parser.add_argument(
        "--baseline-mesh-original-owned-mask",
        type=Path,
        default=GHOST / "p15_first_hit_depth_authority_bounded_v4/full_video_correct_intrinsics_v2/qc_optimized_object_full_video.json",
    )
    parser.add_argument("--output", type=Path, default=CF / "front_only_full_video_ab_comparison.json")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing report: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
    }


def metric_map(report: dict[str, Any], section: str, metric: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for raw_idx, row in (report.get("per_frame") or {}).items():
        value = ((row.get(section) or {}).get(metric))
        if isinstance(value, (int, float)):
            out[int(raw_idx)] = float(value)
    return out


def compare_same_target(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    section = "hand_occluded_render_vs_owned_mask"
    before_iou = metric_map(before, section, "iou")
    after_iou = metric_map(after, section, "iou")
    before_centroid = metric_map(before, section, "centroid_error_px")
    after_centroid = metric_map(after, section, "centroid_error_px")
    common = sorted(set(before_iou) & set(after_iou) & set(before_centroid) & set(after_centroid))
    rows = [
        {
            "frame_idx": idx,
            "baseline_iou": before_iou[idx],
            "counterfactual_iou": after_iou[idx],
            "delta_iou": after_iou[idx] - before_iou[idx],
            "baseline_centroid_error_px": before_centroid[idx],
            "counterfactual_centroid_error_px": after_centroid[idx],
            "delta_centroid_error_px": after_centroid[idx] - before_centroid[idx],
        }
        for idx in common
    ]
    return {
        "frame_count": len(rows),
        "iou": {
            "baseline": summary([r["baseline_iou"] for r in rows]),
            "counterfactual": summary([r["counterfactual_iou"] for r in rows]),
            "delta": summary([r["delta_iou"] for r in rows]),
            "improved_frames": int(sum(r["delta_iou"] > 0 for r in rows)),
        },
        "centroid_error_px": {
            "baseline": summary([r["baseline_centroid_error_px"] for r in rows]),
            "counterfactual": summary([r["counterfactual_centroid_error_px"] for r in rows]),
            "delta": summary([r["delta_centroid_error_px"] for r in rows]),
            "improved_frames": int(sum(r["delta_centroid_error_px"] < 0 for r in rows)),
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    before = load_json(args.baseline_mesh_corrected_mask)
    after = load_json(args.counterfactual_mesh_corrected_mask)
    legacy = load_json(args.baseline_mesh_original_owned_mask)
    report = {
        "schema": "sam3d_depth_order_front_only_full_video_ab_v1",
        "status": "ok",
        "inputs": {
            "baseline_mesh_corrected_mask": str(args.baseline_mesh_corrected_mask),
            "counterfactual_mesh_corrected_mask": str(args.counterfactual_mesh_corrected_mask),
            "baseline_mesh_original_owned_mask": str(args.baseline_mesh_original_owned_mask),
        },
        "same_corrected_target_mesh_ab": compare_same_target(before, after),
        "summary_context": {
            "baseline_mesh_original_owned_mask": (legacy.get("silhouette_summary") or {}).get("hand_occluded_render_vs_owned_mask"),
            "baseline_mesh_corrected_mask": (before.get("silhouette_summary") or {}).get("hand_occluded_render_vs_owned_mask"),
            "counterfactual_mesh_corrected_mask": (after.get("silhouette_summary") or {}).get("hand_occluded_render_vs_owned_mask"),
        },
        "outputs": {
            "baseline_mesh_corrected_mask": (before.get("outputs") or {}),
            "counterfactual_mesh_corrected_mask": (after.get("outputs") or {}),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "same_target_ab": report["same_corrected_target_mesh_ab"]}, indent=2))


if __name__ == "__main__":
    main()
