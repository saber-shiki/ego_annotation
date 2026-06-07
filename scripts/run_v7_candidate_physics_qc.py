#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_path(raw: object, key: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"{key} must be a non-empty path string")
    path = Path(raw)
    if not path.exists():
        raise RuntimeError(f"{key} does not exist: {path}")
    return path


def run_command(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def summary_number(report: dict, path: tuple[str, ...]) -> float:
    cur: object = report
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise RuntimeError(f"report lacks summary path: {'.'.join(path)}")
        cur = cur[key]
    return float(cur)


def run(args: argparse.Namespace) -> dict:
    replay = load_json(args.replay_report)
    if replay.get("status") != "accepted":
        raise RuntimeError(f"physics QC requires accepted replay report, got {replay.get('status')}: {args.replay_report}")
    mesh_archive = require_path(replay.get("aligned_mesh_archive"), "replay.aligned_mesh_archive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_json = args.output_dir / "mesh_surface_contact_qc.json"
    selected_sdf_json = args.output_dir / "selected_contact_sdf_qc.json"
    full_hand_sdf_json = args.output_dir / "full_hand_sdf_qc.json"
    run_command(
        [
            sys.executable,
            str(args.scripts_dir / "diagnose_mesh_surface_contact_v3.py"),
            "--annotations",
            str(args.annotations),
            "--metric-depth-npz",
            str(args.metric_depth_npz),
            "--object-mesh-npz",
            str(mesh_archive),
            "--frame-start",
            str(replay["frame_start"]),
            "--frame-end",
            str(replay["frame_end"]),
            "--intrinsics-source",
            args.intrinsics_source,
            "--output-json",
            str(contact_json),
        ]
    )
    contact = load_json(contact_json)
    reliable_rows = int(contact.get("reliable_temporal_contact_rows", 0))
    geometry_rows = int(contact.get("geometry_backed_temporal_contact_rows", 0))
    if reliable_rows <= 0 and geometry_rows <= 0:
        report = {
            "status": "rejected_no_contact_evidence",
            "annotation_ready": False,
            "method": "run_v7_candidate_physics_qc",
            "replay_report": str(args.replay_report),
            "mesh_archive": str(mesh_archive),
            "contact_report": str(contact_json),
            "selected_contact_sdf_report": None,
            "full_hand_sdf_report": None,
            "frame_start": int(replay["frame_start"]),
            "frame_end": int(replay["frame_end"]),
            "metrics": {
                "reliable_temporal_contact_rows": reliable_rows,
                "geometry_backed_temporal_contact_rows": geometry_rows,
            },
            "thresholds": {
                "max_selected_contact_abs_sdf_p95_m": float(args.max_selected_contact_abs_sdf_p95_m),
                "min_selected_contact_near_surface_fraction": float(args.min_selected_contact_near_surface_fraction),
                "max_selected_contact_penetration_fraction": float(args.max_selected_contact_penetration_fraction),
                "max_full_hand_penetration_fraction": float(args.max_full_hand_penetration_fraction),
            },
            "reason": "accepted visible replay has no reliable or geometry-backed temporal contact row, so contact and nonpenetration cannot be claimed",
        }
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return report
    run_command(
        [
            sys.executable,
            str(args.scripts_dir / "diagnose_volume_sdf_contact_v3.py"),
            "--annotations",
            str(args.annotations),
            "--mesh-archive",
            str(mesh_archive),
            "--contact-report",
            str(contact_json),
            "--frame-start",
            str(replay["frame_start"]),
            "--frame-end",
            str(replay["frame_end"]),
            "--output-json",
            str(selected_sdf_json),
            "--pitch-m",
            str(args.sdf_pitch_m),
        ]
    )
    run_command(
        [
            sys.executable,
            str(args.scripts_dir / "diagnose_hand_object_sdf_penetration_v3.py"),
            "--annotations",
            str(args.annotations),
            "--mesh-archive",
            str(mesh_archive),
            "--contact-report",
            str(contact_json),
            "--frame-start",
            str(replay["frame_start"]),
            "--frame-end",
            str(replay["frame_end"]),
            "--output-json",
            str(full_hand_sdf_json),
            "--pitch-m",
            str(args.sdf_pitch_m),
        ]
    )
    selected = load_json(selected_sdf_json)
    full_hand = load_json(full_hand_sdf_json)
    selected_abs_p95 = summary_number(selected, ("summary", "abs_sdf_m", "p95"))
    selected_near = summary_number(selected, ("summary", "near_surface_fraction"))
    selected_pen = summary_number(selected, ("summary", "penetration_fraction"))
    full_pen = summary_number(full_hand, ("summary", "penetration_fraction"))
    pass_rows = {
        "selected_contact_abs_sdf_p95": bool(selected_abs_p95 <= float(args.max_selected_contact_abs_sdf_p95_m)),
        "selected_contact_near_surface_fraction": bool(selected_near >= float(args.min_selected_contact_near_surface_fraction)),
        "selected_contact_penetration_fraction": bool(selected_pen <= float(args.max_selected_contact_penetration_fraction)),
        "full_hand_penetration_fraction": bool(full_pen <= float(args.max_full_hand_penetration_fraction)),
    }
    accepted = all(pass_rows.values())
    report = {
        "status": "accepted" if accepted else "rejected",
        "annotation_ready": bool(accepted),
        "method": "run_v7_candidate_physics_qc",
        "claim_tested": "an image-replay-accepted generated object mesh also satisfies mesh-surface contact, selected-contact SDF, and full-hand nonpenetration checks",
        "replay_report": str(args.replay_report),
        "mesh_archive": str(mesh_archive),
        "contact_report": str(contact_json),
        "selected_contact_sdf_report": str(selected_sdf_json),
        "full_hand_sdf_report": str(full_hand_sdf_json),
        "frame_start": int(replay["frame_start"]),
        "frame_end": int(replay["frame_end"]),
        "metrics": {
            "reliable_temporal_contact_rows": reliable_rows,
            "geometry_backed_temporal_contact_rows": geometry_rows,
            "selected_contact_abs_sdf_p95_m": selected_abs_p95,
            "selected_contact_near_surface_fraction": selected_near,
            "selected_contact_penetration_fraction": selected_pen,
            "full_hand_penetration_fraction": full_pen,
        },
        "thresholds": {
            "max_selected_contact_abs_sdf_p95_m": float(args.max_selected_contact_abs_sdf_p95_m),
            "min_selected_contact_near_surface_fraction": float(args.min_selected_contact_near_surface_fraction),
            "max_selected_contact_penetration_fraction": float(args.max_selected_contact_penetration_fraction),
            "max_full_hand_penetration_fraction": float(args.max_full_hand_penetration_fraction),
        },
        "pass": pass_rows,
    }
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--intrinsics-source", choices=("annotation-vggt", "hand", "cli"), default="annotation-vggt")
    parser.add_argument("--sdf-pitch-m", type=float, default=0.003)
    parser.add_argument("--max-selected-contact-abs-sdf-p95-m", type=float, default=0.006)
    parser.add_argument("--min-selected-contact-near-surface-fraction", type=float, default=0.75)
    parser.add_argument("--max-selected-contact-penetration-fraction", type=float, default=0.10)
    parser.add_argument("--max-full-hand-penetration-fraction", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
