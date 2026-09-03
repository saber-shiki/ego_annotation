#!/usr/bin/env python3
"""Compare baseline and depth-order-counterfactual GHOST-lite alignment reports."""
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
    parser.add_argument("--baseline", type=Path, default=GHOST / "p15_first_hit_depth_authority_bounded_v4/qc_sam3d_p15_first_hit_alignment.json")
    parser.add_argument(
        "--counterfactual",
        type=Path,
        default=CF / "ghost_front_only_p15_first_hit_depth_authority_v1/qc_sam3d_p15_first_hit_alignment.json",
    )
    parser.add_argument("--output", type=Path, default=CF / "ghost_front_only_alignment_comparison.json")
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
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def frame_rows(report: dict[str, Any], key: str) -> dict[int, dict[str, Any]]:
    return {int(idx): row for idx, row in (report.get(key) or {}).items()}


def compare(baseline: dict[str, Any], counter: dict[str, Any]) -> dict[str, Any]:
    ids = sorted(set(frame_rows(baseline, "frame_metrics_after")) & set(frame_rows(counter, "frame_metrics_after")))
    rows: list[dict[str, Any]] = []
    for idx in ids:
        b = frame_rows(baseline, "frame_metrics_after")[idx]
        c = frame_rows(counter, "frame_metrics_after")[idx]
        b_depth = abs(float(b["first_hit_minus_observed_depth_m"]["median_m"]))
        c_depth = abs(float(c["first_hit_minus_observed_depth_m"]["median_m"]))
        rows.append(
            {
                "frame_idx": idx,
                "baseline_iou": float(b["silhouette_iou"]),
                "counterfactual_iou": float(c["silhouette_iou"]),
                "delta_iou_counter_minus_baseline": float(c["silhouette_iou"] - b["silhouette_iou"]),
                "baseline_abs_depth_median_mm": b_depth * 1000.0,
                "counterfactual_abs_depth_median_mm": c_depth * 1000.0,
                "delta_abs_depth_mm_counter_minus_baseline": (c_depth - b_depth) * 1000.0,
                "baseline_coverage": float(b["surfel_first_hit_coverage_fraction"]),
                "counterfactual_coverage": float(c["surfel_first_hit_coverage_fraction"]),
                "delta_coverage_counter_minus_baseline": float(c["surfel_first_hit_coverage_fraction"] - b["surfel_first_hit_coverage_fraction"]),
            }
        )
    return {
        "frames": ids,
        "rows": rows,
        "iou": {
            "baseline": summary([r["baseline_iou"] for r in rows]),
            "counterfactual": summary([r["counterfactual_iou"] for r in rows]),
            "delta": summary([r["delta_iou_counter_minus_baseline"] for r in rows]),
            "improved_frames": int(sum(r["delta_iou_counter_minus_baseline"] > 0 for r in rows)),
        },
        "abs_depth_median_mm": {
            "baseline": summary([r["baseline_abs_depth_median_mm"] for r in rows]),
            "counterfactual": summary([r["counterfactual_abs_depth_median_mm"] for r in rows]),
            "delta": summary([r["delta_abs_depth_mm_counter_minus_baseline"] for r in rows]),
            "improved_frames": int(sum(r["delta_abs_depth_mm_counter_minus_baseline"] < 0 for r in rows)),
        },
        "coverage": {
            "baseline": summary([r["baseline_coverage"] for r in rows]),
            "counterfactual": summary([r["counterfactual_coverage"] for r in rows]),
            "delta": summary([r["delta_coverage_counter_minus_baseline"] for r in rows]),
            "improved_frames": int(sum(r["delta_coverage_counter_minus_baseline"] > 0 for r in rows)),
        },
        "sim3_increment": {
            "baseline": baseline.get("sim3_increment"),
            "counterfactual": counter.get("sim3_increment"),
        },
        "optimizer": {
            "baseline": baseline.get("optimizer"),
            "counterfactual": counter.get("optimizer"),
        },
        "residual_rms_after": {
            "baseline": baseline.get("residual_rms_after"),
            "counterfactual": counter.get("residual_rms_after"),
        },
    }


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    counter = load_json(args.counterfactual)
    report = {
        "schema": "sam3d_ghost_depth_order_counterfactual_alignment_comparison_v1",
        "status": "ok",
        "baseline_report": str(args.baseline),
        "counterfactual_report": str(args.counterfactual),
        "comparison": compare(baseline, counter),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "comparison": report["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
