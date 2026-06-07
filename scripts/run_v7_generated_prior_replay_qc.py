#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def metric(report: dict, key: str, stat: str) -> float:
    raw = report.get(key)
    if not isinstance(raw, dict) or stat not in raw:
        raise RuntimeError(f"report lacks metric {key}.{stat}")
    return float(raw[stat])


def run_command(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_archive = args.output_dir / "aligned_prior_meshes_world.npz"
    align_report = args.output_dir / "qc_aligned_mesh_prior_v7.json"
    zbuffer_dir = args.output_dir / "zbuffer_qc"
    run_command(
        [
            sys.executable,
            str(args.scripts_dir / "archive_aligned_mesh_prior_v7.py"),
            "--mesh-prior",
            str(args.mesh_prior),
            "--observed-mesh-archive",
            str(args.observed_mesh_archive),
            "--frame-start",
            str(args.frame_start),
            "--frame-end",
            str(args.frame_end),
            "--output-mesh-archive",
            str(aligned_archive),
            "--output-json",
            str(align_report),
            "--samples",
            str(args.samples),
            "--max-bidirectional-p95-m",
            str(args.max_alignment_p95_m),
        ]
    )
    run_command(
        [
            sys.executable,
            str(args.scripts_dir / "render_mesh_zbuffer_qc_v3.py"),
            "--mesh-archive",
            str(aligned_archive),
            "--manifest",
            str(args.manifest),
            "--annotations",
            str(args.annotations),
            "--metric-depth-npz",
            str(args.metric_depth_npz),
            "--intrinsics-source",
            args.intrinsics_source,
            "--frame-start",
            str(args.frame_start),
            "--frame-end",
            str(args.frame_end),
            "--max-faces",
            str(args.max_faces),
            "--vertex-splat-radius-px",
            str(args.vertex_splat_radius_px),
            "--output-dir",
            str(zbuffer_dir),
        ]
    )
    align = load_json(align_report)
    zbuffer = load_json(zbuffer_dir / "qc_mesh_zbuffer_projection_v3.json")
    alignment_p95 = metric(align, "alignment_bidirectional_p95_m", "p95")
    iou_median = metric(zbuffer, "silhouette_mask_iou", "median")
    visible_inside_median = metric(zbuffer, "visible_silhouette_inside_mask_fraction", "median")
    zbuffer_abs_p95_median = metric(zbuffer, "zbuffer_depth_abs_p95_m", "median")
    pass_rows = {
        "alignment_p95": bool(alignment_p95 <= float(args.max_alignment_p95_m)),
        "silhouette_iou_median": bool(iou_median >= float(args.min_iou_median)),
        "visible_inside_median": bool(visible_inside_median >= float(args.min_visible_inside_median)),
        "zbuffer_abs_p95_median": bool(zbuffer_abs_p95_median <= float(args.max_zbuffer_abs_p95_median_m)),
    }
    accepted = all(pass_rows.values())
    report = {
        "status": "accepted" if accepted else "rejected",
        "annotation_ready": bool(accepted),
        "method": "run_v7_generated_prior_replay_qc",
        "claim_tested": "a generated object mesh prior can be accepted as object geometry only after metric alignment and image-depth replay pass",
        "mesh_prior": str(args.mesh_prior),
        "observed_mesh_archive": str(args.observed_mesh_archive),
        "aligned_mesh_archive": str(aligned_archive),
        "alignment_report": str(align_report),
        "zbuffer_report": str(zbuffer_dir / "qc_mesh_zbuffer_projection_v3.json"),
        "zbuffer_video": str(zbuffer_dir / "mesh_zbuffer_projection_qc.mp4"),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "metrics": {
            "alignment_bidirectional_p95_m": alignment_p95,
            "silhouette_iou_median": iou_median,
            "visible_inside_median": visible_inside_median,
            "zbuffer_abs_p95_median_m": zbuffer_abs_p95_median,
        },
        "thresholds": {
            "max_alignment_p95_m": float(args.max_alignment_p95_m),
            "min_iou_median": float(args.min_iou_median),
            "min_visible_inside_median": float(args.min_visible_inside_median),
            "max_zbuffer_abs_p95_median_m": float(args.max_zbuffer_abs_p95_median_m),
        },
        "pass": pass_rows,
        "next_required_if_accepted": [
            "mesh-surface contact recomputation",
            "selected-contact SDF",
            "full-hand SDF",
            "stakeholder render inspection",
        ],
    }
    output_json = args.output_dir / "qc_v7_generated_prior_replay.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--observed-mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="annotation-vggt")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--vertex-splat-radius-px", type=int, default=0)
    parser.add_argument("--max-alignment-p95-m", type=float, default=0.010)
    parser.add_argument("--min-iou-median", type=float, default=0.900)
    parser.add_argument("--min-visible-inside-median", type=float, default=0.900)
    parser.add_argument("--max-zbuffer-abs-p95-median-m", type=float, default=0.010)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
