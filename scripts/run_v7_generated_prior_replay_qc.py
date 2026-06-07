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


def run_zbuffer(args: argparse.Namespace, mesh_archive: Path, output_dir: Path) -> dict:
    run_command(
        [
            sys.executable,
            str(args.scripts_dir / "render_mesh_zbuffer_qc_v3.py"),
            "--mesh-archive",
            str(mesh_archive),
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
            str(output_dir),
        ]
    )
    return load_json(output_dir / "qc_mesh_zbuffer_projection_v3.json")


def write_report(path: Path, report: dict) -> dict:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_archive = args.output_dir / "aligned_prior_meshes_world.npz"
    align_report = args.output_dir / "qc_aligned_mesh_prior_v7.json"
    observed_zbuffer_dir = args.output_dir / "observed_target_zbuffer_qc"
    zbuffer_dir = args.output_dir / "zbuffer_qc"
    observed_zbuffer = run_zbuffer(args, args.observed_mesh_archive, observed_zbuffer_dir)
    observed_visible_inside = metric(observed_zbuffer, "visible_silhouette_inside_mask_fraction", "median")
    observed_zbuffer_abs_p95 = metric(observed_zbuffer, "zbuffer_depth_abs_p95_m", "median")
    observed_target_pass = {
        "observed_visible_inside_median": bool(observed_visible_inside >= float(args.min_observed_visible_inside_median)),
        "observed_zbuffer_abs_p95_median": bool(observed_zbuffer_abs_p95 <= float(args.max_observed_zbuffer_abs_p95_median_m)),
    }
    if not all(observed_target_pass.values()):
        return write_report(
            args.output_dir / "qc_v7_generated_prior_replay.json",
            {
                "status": "invalid_observed_target",
                "annotation_ready": False,
                "method": "run_v7_generated_prior_replay_qc",
                "claim_tested": "a generated object mesh prior can be accepted as object geometry only after its observed target replay is internally consistent and the prior replay passes",
                "mesh_prior": str(args.mesh_prior),
                "observed_mesh_archive": str(args.observed_mesh_archive),
                "observed_target_zbuffer_report": str(observed_zbuffer_dir / "qc_mesh_zbuffer_projection_v3.json"),
                "observed_target_zbuffer_video": str(observed_zbuffer_dir / "mesh_zbuffer_projection_qc.mp4"),
                "frame_start": int(args.frame_start),
                "frame_end": int(args.frame_end),
                "observed_target_metrics": {
                    "visible_inside_median": observed_visible_inside,
                    "zbuffer_abs_p95_median_m": observed_zbuffer_abs_p95,
                },
                "thresholds": {
                    "min_observed_visible_inside_median": float(args.min_observed_visible_inside_median),
                    "max_observed_zbuffer_abs_p95_median_m": float(args.max_observed_zbuffer_abs_p95_median_m),
                },
                "pass": observed_target_pass,
                "reason": "observed target mesh does not replay against the supplied mask/depth/camera contract, so prior acceptance would be causally uninterpretable",
            },
        )
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
            "--max-visible-surface-p95-m",
            str(args.max_visible_surface_p95_m),
        ]
    )
    zbuffer = run_zbuffer(args, aligned_archive, zbuffer_dir)
    align = load_json(align_report)
    alignment_p95 = metric(align, "alignment_bidirectional_p95_m", "p95")
    visible_surface_p95 = metric(align, "visible_surface_coverage_p95_m", "p95")
    hidden_surface_p95 = metric(align, "hidden_surface_conflict_p95_m", "p95")
    iou_median = metric(zbuffer, "silhouette_mask_iou", "median")
    visible_inside_median = metric(zbuffer, "visible_silhouette_inside_mask_fraction", "median")
    zbuffer_abs_p95_median = metric(zbuffer, "zbuffer_depth_abs_p95_m", "median")
    pass_rows = {
        "strict_full_surface_alignment_p95": bool(alignment_p95 <= float(args.max_alignment_p95_m)),
        "visible_surface_coverage_p95": bool(visible_surface_p95 <= float(args.max_visible_surface_p95_m)),
        "silhouette_iou_median": bool(iou_median >= float(args.min_iou_median)),
        "visible_inside_median": bool(visible_inside_median >= float(args.min_visible_inside_median)),
        "zbuffer_abs_p95_median": bool(zbuffer_abs_p95_median <= float(args.max_zbuffer_abs_p95_median_m)),
    }
    delivery_pass_keys = [
        "visible_surface_coverage_p95",
        "silhouette_iou_median",
        "visible_inside_median",
        "zbuffer_abs_p95_median",
    ]
    accepted = all(pass_rows[key] for key in delivery_pass_keys)
    report = {
        "status": "accepted" if accepted else "rejected",
        "annotation_ready": bool(accepted),
        "method": "run_v7_generated_prior_replay_qc",
        "claim_tested": "a generated object mesh prior can be accepted as object geometry only after its visible surface covers measured geometry and image-depth replay passes",
        "mesh_prior": str(args.mesh_prior),
        "observed_mesh_archive": str(args.observed_mesh_archive),
        "observed_target_zbuffer_report": str(observed_zbuffer_dir / "qc_mesh_zbuffer_projection_v3.json"),
        "observed_target_zbuffer_video": str(observed_zbuffer_dir / "mesh_zbuffer_projection_qc.mp4"),
        "aligned_mesh_archive": str(aligned_archive),
        "alignment_report": str(align_report),
        "zbuffer_report": str(zbuffer_dir / "qc_mesh_zbuffer_projection_v3.json"),
        "zbuffer_video": str(zbuffer_dir / "mesh_zbuffer_projection_qc.mp4"),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "metrics": {
            "observed_target_visible_inside_median": observed_visible_inside,
            "observed_target_zbuffer_abs_p95_median_m": observed_zbuffer_abs_p95,
            "alignment_bidirectional_p95_m": alignment_p95,
            "visible_surface_coverage_p95_m": visible_surface_p95,
            "hidden_surface_conflict_p95_m": hidden_surface_p95,
            "silhouette_iou_median": iou_median,
            "visible_inside_median": visible_inside_median,
            "zbuffer_abs_p95_median_m": zbuffer_abs_p95_median,
        },
        "thresholds": {
            "min_observed_visible_inside_median": float(args.min_observed_visible_inside_median),
            "max_observed_zbuffer_abs_p95_median_m": float(args.max_observed_zbuffer_abs_p95_median_m),
            "max_alignment_p95_m": float(args.max_alignment_p95_m),
            "max_visible_surface_p95_m": float(args.max_visible_surface_p95_m),
            "min_iou_median": float(args.min_iou_median),
            "min_visible_inside_median": float(args.min_visible_inside_median),
            "max_zbuffer_abs_p95_median_m": float(args.max_zbuffer_abs_p95_median_m),
        },
        "pass": pass_rows,
        "observed_target_pass": observed_target_pass,
        "strict_full_surface_alignment_is_diagnostic": True,
        "delivery_pass_keys": delivery_pass_keys,
        "next_required_if_accepted": [
            "mesh-surface contact recomputation",
            "selected-contact SDF",
            "full-hand SDF",
            "stakeholder render inspection",
        ],
    }
    return write_report(args.output_dir / "qc_v7_generated_prior_replay.json", report)


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
    parser.add_argument("--min-observed-visible-inside-median", type=float, default=0.900)
    parser.add_argument("--max-observed-zbuffer-abs-p95-median-m", type=float, default=0.010)
    parser.add_argument("--max-alignment-p95-m", type=float, default=0.010)
    parser.add_argument("--max-visible-surface-p95-m", type=float, default=0.010)
    parser.add_argument("--min-iou-median", type=float, default=0.900)
    parser.add_argument("--min-visible-inside-median", type=float, default=0.900)
    parser.add_argument("--max-zbuffer-abs-p95-median-m", type=float, default=0.010)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
