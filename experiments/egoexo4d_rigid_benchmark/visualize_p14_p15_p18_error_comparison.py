#!/usr/bin/env python3
"""Visualize P14/P15 diagnostics and P18 independent hand-GT errors.

P14 has no released object-pose GT, so its values are internal visible-depth to
completed-mesh residuals. P15 likewise has no object SE(3) GT; its sparse-mask
plot is a combined mesh/pose/camera/visibility diagnostic. Only P18 hand-joint
plots use independent released 3-D hand GT.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def p14_rows(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["frame_idx"]): row for row in report["pose_rows"]}


def p14_fit_values(rows: dict[int, dict[str, Any]]) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for frame, row in rows.items():
        initial = row.get("observed_to_mesh_initial")
        final = row.get("observed_to_mesh_final")
        if isinstance(initial, dict) and isinstance(final, dict):
            result[frame] = (float(initial["median_m"]) * 1000.0, float(final["median_m"]) * 1000.0)
    return result


def p15_source(row: dict[str, Any]) -> str:
    graph = row.get("temporal_pose_graph") or {}
    source = str(graph.get("pose_source", ""))
    if source.startswith("direct_visible_pose_observation"):
        return "direct"
    if source == "interpolated_between_visible_pose_observations":
        return "interpolated"
    if source == "nearest_visible_pose_hold":
        return "nearest_hold"
    return "other"


def p15_counts(report: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in report["pose_rows"]:
        counts[p15_source(row)] += 1
    return dict(sorted(counts.items()))


def weighted_hand_timeline(report: dict[str, Any], metric: str) -> dict[int, float]:
    by_frame: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row in report["per_frame"]:
        value = row.get(metric)
        if row.get("status") != "scored" or value is None:
            continue
        by_frame[int(row["local_frame_idx"])].append((float(value), int(row["eligible_gt_joint_count"])))
    result: dict[int, float] = {}
    for frame, values in by_frame.items():
        weights = np.asarray([weight for _value, weight in values], dtype=np.float64)
        data = np.asarray([value for value, _weight in values], dtype=np.float64)
        result[frame] = float(np.sum(data * weights) / np.sum(weights))
    return result


def method_per_frame(report: dict[str, Any], method: str) -> dict[int, dict[str, Any]]:
    return {int(row["local_frame_idx"]): row for row in report["methods"][method]["per_frame"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-p14", type=Path, required=True)
    parser.add_argument("--fixed-p14", type=Path, required=True)
    parser.add_argument("--frozen-p15", type=Path, required=True)
    parser.add_argument("--fixed-p15", type=Path, required=True)
    parser.add_argument("--current-output-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace:
        raise FileExistsError(f"output directory is non-empty: {output_dir}; pass --replace")
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_p14 = load_json(args.frozen_p14)
    fixed_p14 = load_json(args.fixed_p14)
    frozen_p15 = load_json(args.frozen_p15)
    fixed_p15 = load_json(args.fixed_p15)
    evaluation = load_json(args.current_output_evaluation)

    frozen_p14_rows = p14_rows(frozen_p14)
    fixed_p14_rows = p14_rows(fixed_p14)
    frozen_fit = p14_fit_values(frozen_p14_rows)
    fixed_fit = p14_fit_values(fixed_p14_rows)
    eligible_frames = sorted(
        frame
        for frame, row in fixed_p14_rows.items()
        if row.get("rigid_pose_observation_eligible") is True and frame in fixed_fit
    )
    rejected_frames = sorted(
        frame for frame, row in fixed_p14_rows.items() if row.get("rigid_pose_observation_eligible") is False
    )
    if set(frozen_fit) != set(eligible_frames).union(rejected_frames):
        raise RuntimeError("frozen P14 fit rows do not partition into fixed eligible/rejected rows")
    eligible_initial = [frozen_fit[frame][0] for frame in eligible_frames]
    eligible_final = [frozen_fit[frame][1] for frame in eligible_frames]
    rejected_initial = [frozen_fit[frame][0] for frame in rejected_frames]
    rejected_final = [frozen_fit[frame][1] for frame in rejected_frames]
    fixed_final = [fixed_fit[frame][1] for frame in eligible_frames]

    frozen_sources = {int(row["frame_idx"]): p15_source(row) for row in frozen_p15["pose_rows"]}
    fixed_sources = {int(row["frame_idx"]): p15_source(row) for row in fixed_p15["pose_rows"]}
    source_color = {
        "direct": "#1f77b4",
        "interpolated": "#f2a900",
        "nearest_hold": "#d62728",
        "other": "#7f7f7f",
    }

    projection = evaluation["projected_mesh_visible_mask_diagnostic"]
    projected_frozen = method_per_frame(projection, "frozen_v1")
    projected_fixed = method_per_frame(projection, "pose_gate_fixed")
    segmentation_rows = {
        int(row["local_frame_idx"]): row for row in evaluation["object_visible_segmentation"]["per_frame"]
    }
    mask_frames = sorted(set(projected_frozen).intersection(projected_fixed).intersection(segmentation_rows))

    hands = evaluation["hand_pose"]
    hand_methods = {
        "HaWoR": "hawor_baseline",
        "P18 raw": "frozen_p18_raw",
        "P18b canonical": "frozen_p18b_canonical",
    }
    summary_hand_methods = {
        **hand_methods,
        "fixed P18 quarantine": "fixed_p18_quarantine",
        "fixed P18b quarantine": "fixed_p18b_quarantine",
    }
    absolute_timelines = {
        label: weighted_hand_timeline(hands[key], "absolute_mpjpe_mm") for label, key in hand_methods.items()
    }
    relative_timelines = {
        label: weighted_hand_timeline(hands[key], "root_relative_mpjpe_mm") for label, key in hand_methods.items()
    }
    common_hand_frames = sorted(
        set.intersection(
            *(set(values) for values in absolute_timelines.values()),
            *(set(values) for values in relative_timelines.values()),
        )
    )
    raw_absolute_delta = [
        absolute_timelines["P18 raw"][frame] - absolute_timelines["HaWoR"][frame]
        for frame in common_hand_frames
    ]
    raw_relative_delta = [
        relative_timelines["P18 raw"][frame] - relative_timelines["HaWoR"][frame]
        for frame in common_hand_frames
    ]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    ax = axes[0, 0]
    fit_frames = sorted(frozen_fit)
    ax.scatter(fit_frames, [frozen_fit[frame][0] for frame in fit_frames], s=18, c="#b0b0b0", label="P14 initial")
    ax.scatter(eligible_frames, eligible_final, s=42, c="#1f77b4", marker="o", label="final: eligible (6)")
    ax.scatter(rejected_frames, rejected_final, s=34, c="#d62728", marker="x", label="final: rejected (27)")
    ax.scatter(eligible_frames, fixed_final, s=70, facecolors="none", edgecolors="#17becf", linewidths=1.5, label="fixed P14 fits")
    ax.set_yscale("log")
    ax.set_xlabel("local frame")
    ax.set_ylabel("observed→mesh median (mm, log)")
    ax.set_title("P14 internal visible-depth residual by frame\n(not object-pose GT)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    box_values = [eligible_initial, eligible_final, rejected_initial, rejected_final]
    box = ax.boxplot(
        box_values,
        tick_labels=["eligible\ninitial", "eligible\nfinal", "rejected\ninitial", "rejected\nfinal"],
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(box["boxes"], ["#a6cee3", "#1f78b4", "#fdbf6f", "#e31a1c"], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_yscale("log")
    ax.set_ylabel("observed→mesh median (mm, log)")
    ax.set_title("P14 eligible vs leaked/rejected rows")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.02,
        0.98,
        f"eligible final median: {np.median(eligible_final):.3f} mm\n"
        f"rejected final median: {np.median(rejected_final):.3f} mm\n"
        f"all-row report median: {frozen_p14['final_observed_to_mesh_median_summary_m']['median']*1000:.3f} mm",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax = axes[0, 2]
    for method_y, (label, sources) in enumerate((("frozen P15", frozen_sources), ("fixed P15", fixed_sources))):
        for source in source_color:
            frames = [frame for frame, value in sources.items() if value == source]
            if frames:
                ax.scatter(frames, np.full(len(frames), method_y), marker="|", s=180, linewidths=2.0, c=source_color[source])
    ax.set_yticks([0, 1], ["frozen P15", "fixed P15"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(-2, 151)
    ax.set_xlabel("local frame")
    ax.set_title("P15 direct support vs completion\n(corrections are exactly zero)")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(
        handles=[Patch(color=color, label=label.replace("_", " ")) for label, color in source_color.items() if label != "other"],
        loc="lower left",
        fontsize=8,
    )
    ax.text(
        0.99,
        0.04,
        f"frozen: {p15_counts(frozen_p15)}\nfixed: {p15_counts(fixed_p15)}\n"
        f"both nfev=1, cost=0; fixed ready={fixed_p15.get('annotation_ready')}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    ax = axes[1, 0]
    ax.plot(mask_frames, [segmentation_rows[frame]["iou"] for frame in mask_frames], "o--", color="#2ca02c", label="SAM2 visible Mask")
    ax.plot(mask_frames, [projected_frozen[frame]["iou"] for frame in mask_frames], "o-", color="#9467bd", label="frozen Mesh+P15")
    ax.plot(mask_frames, [projected_fixed[frame]["iou"] for frame in mask_frames], "s-", color="#ff7f0e", label="fixed Mesh+P15")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(mask_frames)
    ax.set_xlabel("local frame with sparse Mask GT")
    ax.set_ylabel("visible-mask IoU")
    ax.set_title("P15 projected-Mesh diagnostic\n(combined geometry/pose/camera, not SE(3) GT)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    hand_colors = {"HaWoR": "#202020", "P18 raw": "#ff7f0e", "P18b canonical": "#1f77b4"}
    hand_styles = {"HaWoR": "-", "P18 raw": "-", "P18b canonical": "--"}
    for label, timeline in absolute_timelines.items():
        frames = sorted(timeline)
        ax.plot(frames, [timeline[frame] for frame in frames], hand_styles[label], color=hand_colors[label], linewidth=1.4, alpha=0.9, label=label)
    ax.set_xlabel("local annotated hand frame")
    ax.set_ylabel("absolute MPJPE (mm)")
    ax.set_title("P18 independent 3-D hand-GT error\n(joint-count-weighted across visible sides)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    aggregate_text = []
    for label, key in hand_methods.items():
        all_hands = hands[key]["all_hands"]
        aggregate_text.append(
            f"{label}: {all_hands['absolute_joint_error_mm']['mean']:.3f} / "
            f"{all_hands['root_relative_joint_error_mm']['mean']:.3f} mm"
        )
    ax.text(
        0.02,
        0.98,
        "absolute / root-relative\n" + "\n".join(aggregate_text),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax = axes[1, 2]
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(common_hand_frames, raw_absolute_delta, color="#d62728", linewidth=1.3, label="P18 raw − HaWoR absolute")
    ax.plot(common_hand_frames, raw_relative_delta, color="#1f77b4", linewidth=1.3, label="P18 raw − HaWoR root-relative")
    ax.set_xlabel("local annotated hand frame")
    ax.set_ylabel("error delta (mm; negative = improvement)")
    ax.set_title("P18 change relative to HaWoR\nP18b/fixed quarantine are exactly zero delta")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Tire-lever P14 / P15 / P18 error and support comparison", fontsize=16)
    fig.text(
        0.5,
        0.002,
        "Claim boundary: P14 residuals are internal; P15 silhouette is a joint diagnostic; only P18 uses independent released 3-D hand GT. No released object SE(3) GT exists.",
        ha="center",
        fontsize=9,
    )
    png_path = output_dir / "p14_p15_p18_error_comparison.png"
    jpg_path = output_dir / "p14_p15_p18_error_comparison.jpg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(jpg_path, dpi=150, pil_kwargs={"quality": 94})
    plt.close(fig)

    summary = {
        "status": "p14_p15_p18_error_visualization_complete",
        "claim_scope": {
            "p14": "internal observed visible-depth to completed-mesh nearest-surface residual; not object pose GT",
            "p15": "direct/completion support plus projected completed-mesh versus sparse visible Mask; no released object SE(3) GT",
            "p18": "independent released Ego-Exo4D 3-D named hand-joint GT",
        },
        "inputs": {
            "frozen_p14": str(args.frozen_p14.resolve()),
            "fixed_p14": str(args.fixed_p14.resolve()),
            "frozen_p15": str(args.frozen_p15.resolve()),
            "fixed_p15": str(args.fixed_p15.resolve()),
            "current_output_evaluation": str(args.current_output_evaluation.resolve()),
        },
        "p14": {
            "eligible_frames": eligible_frames,
            "rejected_frames": rejected_frames,
            "eligible_initial_median_mm": finite_summary(eligible_initial),
            "eligible_final_median_mm": finite_summary(eligible_final),
            "rejected_initial_median_mm": finite_summary(rejected_initial),
            "rejected_final_median_mm": finite_summary(rejected_final),
            "frozen_all_row_report_final_median_mm": float(frozen_p14["final_observed_to_mesh_median_summary_m"]["median"]) * 1000.0,
        },
        "p15": {
            "frozen_source_counts": p15_counts(frozen_p15),
            "fixed_source_counts": p15_counts(fixed_p15),
            "frozen_optimizer": frozen_p15["optimizer"],
            "fixed_optimizer": fixed_p15["optimizer"],
            "fixed_annotation_ready": fixed_p15.get("annotation_ready"),
            "projected_mesh_mean_iou": {
                "frozen": projection["methods"]["frozen_v1"]["aggregate"]["iou"]["mean"],
                "fixed": projection["methods"]["pose_gate_fixed"]["aggregate"]["iou"]["mean"],
            },
        },
        "p18": {
            "methods": {
                label: {
                    "absolute_mpjpe_mm": hands[key]["all_hands"]["absolute_joint_error_mm"]["mean"],
                    "root_relative_mpjpe_mm": hands[key]["all_hands"]["root_relative_joint_error_mm"]["mean"],
                }
                for label, key in summary_hand_methods.items()
            },
            "candidate_vs_hawor_exact_delta": evaluation["hand_state_exact_delta"],
        },
        "outputs": {"png": str(png_path), "jpg": str(jpg_path)},
    }
    write_json(output_dir / "p14_p15_p18_error_comparison.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
