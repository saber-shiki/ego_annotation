#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
REQUIRED_TARGET_KEYS = (
    "observed_mesh_archive",
    "manifest",
    "annotations",
    "metric_depth_npz",
    "frame_start",
    "frame_end",
    "intrinsics_source",
    "physics_intrinsics_source",
    "baseline_zbuffer_json",
)


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


def parse_candidate(raw: str) -> tuple[str, str, Path, str]:
    parts = raw.split("|")
    if len(parts) != 4:
        raise RuntimeError("--candidate must have format target_id|candidate_name|mesh_path|note")
    target_id, name, mesh, note = [part.strip() for part in parts]
    if not target_id or not name or not mesh:
        raise RuntimeError("--candidate target_id, candidate_name, and mesh_path must be non-empty")
    return target_id, name, Path(mesh), note


def read_candidate_file(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"candidate file does not exist: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        try:
            parse_candidate(stripped)
        except RuntimeError as exc:
            raise RuntimeError(f"{path}:{line_number}: {exc}") from exc
        rows.append(stripped)
    return rows


def candidate_rows(args: argparse.Namespace) -> list[str]:
    rows = []
    for path in args.candidate_file:
        rows.extend(read_candidate_file(path))
    rows.extend(args.candidate)
    if not rows:
        raise RuntimeError("provide at least one --candidate or --candidate-file row")
    return rows


def shell_token(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_command(argv: list[str], dry_run: bool) -> None:
    print(" ".join(shell_token(arg) for arg in argv))
    if dry_run:
        return
    subprocess.run(argv, check=True)


def validate_target(target_id: str, raw: object) -> dict:
    if not isinstance(raw, dict):
        raise RuntimeError(f"target {target_id} must be a JSON object")
    missing = [key for key in REQUIRED_TARGET_KEYS if key not in raw]
    if missing:
        raise RuntimeError(f"target {target_id} lacks keys: {', '.join(missing)}")
    target = dict(raw)
    for key in ("observed_mesh_archive", "manifest", "annotations", "metric_depth_npz", "baseline_zbuffer_json"):
        target[key] = require_path(target[key], f"{target_id}.{key}")
    target["frame_start"] = int(target["frame_start"])
    target["frame_end"] = int(target["frame_end"])
    if target["frame_end"] < target["frame_start"]:
        raise RuntimeError(f"target {target_id} has inverted frame range")
    if target["intrinsics_source"] not in {"manifest", "annotation-vggt"}:
        raise RuntimeError(f"target {target_id} has invalid intrinsics_source: {target['intrinsics_source']}")
    if target["physics_intrinsics_source"] not in {"annotation-vggt", "hand", "cli"}:
        raise RuntimeError(f"target {target_id} has invalid physics_intrinsics_source: {target['physics_intrinsics_source']}")
    return target


def candidate_output_dir(root: Path, target_id: str, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    if not safe:
        raise RuntimeError("candidate name produced an empty output directory")
    return root / target_id / safe


def run_replay(args: argparse.Namespace, target_id: str, name: str, mesh: Path, note: str, target: dict) -> dict:
    if not mesh.exists():
        raise RuntimeError(f"candidate mesh does not exist: {mesh}")
    out_dir = candidate_output_dir(args.output_root, target_id, name)
    argv = [
        sys.executable,
        str(args.scripts_dir / "run_v7_generated_prior_replay_qc.py"),
        "--mesh-prior",
        str(mesh),
        "--observed-mesh-archive",
        str(target["observed_mesh_archive"]),
        "--manifest",
        str(target["manifest"]),
        "--annotations",
        str(target["annotations"]),
        "--metric-depth-npz",
        str(target["metric_depth_npz"]),
        "--frame-start",
        str(target["frame_start"]),
        "--frame-end",
        str(target["frame_end"]),
        "--intrinsics-source",
        str(target["intrinsics_source"]),
        "--output-dir",
        str(out_dir),
        "--samples",
        str(args.samples),
    ]
    if args.max_faces:
        argv.extend(["--max-faces", str(args.max_faces)])
    if args.vertex_splat_radius_px:
        argv.extend(["--vertex-splat-radius-px", str(args.vertex_splat_radius_px)])
    run_command(argv, bool(args.dry_run))
    report_path = out_dir / "qc_v7_generated_prior_replay.json"
    result = {
        "target_id": target_id,
        "candidate_name": name,
        "note": note,
        "mesh_prior": str(mesh),
        "output_dir": str(out_dir),
        "report": str(report_path),
        "baseline_zbuffer_json": str(target["baseline_zbuffer_json"]),
        "replay_controls": {
            "samples": int(args.samples),
            "max_faces": int(args.max_faces),
            "vertex_splat_radius_px": int(args.vertex_splat_radius_px),
            "full_fidelity_zbuffer": bool(args.max_faces == 0),
        },
    }
    if not args.dry_run:
        report = load_json(report_path)
        result["status"] = report.get("status")
        result["annotation_ready"] = bool(report.get("annotation_ready", False))
        result["metrics"] = report.get("metrics")
        result["pass"] = report.get("pass")
    return result


def run_physics(args: argparse.Namespace, replay_result: dict, target: dict) -> dict | None:
    if not args.run_physics:
        return None
    physics_dir = Path(replay_result["output_dir"]) / "physics_qc"
    physics_report = physics_dir / "qc_v7_candidate_physics.json"
    replay_status = replay_result.get("status")
    if args.dry_run and replay_status is None:
        return {
            "status": "dry_run_replay_not_executed",
            "reason": "dry-run mode prints the replay command without creating the replay report required by physics QC",
            "output_dir": str(physics_dir),
            "report": str(physics_report),
        }
    if replay_status != "accepted":
        return {
            "status": "skipped_replay_not_accepted",
            "reason": f"physics QC requires accepted replay, got {replay_status}",
            "output_dir": str(physics_dir),
            "report": str(physics_report),
        }
    argv = [
        sys.executable,
        str(args.scripts_dir / "run_v7_candidate_physics_qc.py"),
        "--replay-report",
        str(replay_result["report"]),
        "--annotations",
        str(target["annotations"]),
        "--metric-depth-npz",
        str(target["metric_depth_npz"]),
        "--intrinsics-source",
        str(target["physics_intrinsics_source"]),
        "--output-dir",
        str(physics_dir),
        "--output-json",
        str(physics_report),
        "--sdf-pitch-m",
        str(args.sdf_pitch_m),
        "--max-selected-contact-abs-sdf-p95-m",
        str(args.max_selected_contact_abs_sdf_p95_m),
        "--min-selected-contact-near-surface-fraction",
        str(args.min_selected_contact_near_surface_fraction),
        "--max-selected-contact-penetration-fraction",
        str(args.max_selected_contact_penetration_fraction),
        "--max-full-hand-penetration-fraction",
        str(args.max_full_hand_penetration_fraction),
    ]
    run_command(argv, bool(args.dry_run))
    if args.dry_run:
        return {
            "status": "dry_run",
            "output_dir": str(physics_dir),
            "report": str(physics_report),
        }
    physics = load_json(physics_report)
    return {
        "status": physics.get("status"),
        "annotation_ready": bool(physics.get("annotation_ready", False)),
        "output_dir": str(physics_dir),
        "report": str(physics_report),
        "metrics": physics.get("metrics"),
        "pass": physics.get("pass"),
    }


def run_deliverables(args: argparse.Namespace, replay_result: dict, physics: dict | None, target: dict) -> dict | None:
    if not args.render_deliverables:
        return None
    render_dir = Path(replay_result["output_dir"]) / "deliverables"
    render_report = render_dir / "v7_candidate_deliverables_manifest.json"
    if physics is None:
        return {
            "status": "skipped_physics_not_requested",
            "reason": "delivery rendering requires --run-physics and accepted physics QC",
            "output_dir": str(render_dir),
            "report": str(render_report),
        }
    if physics.get("status") != "accepted" or not bool(physics.get("annotation_ready", False)):
        return {
            "status": "skipped_physics_not_accepted",
            "reason": f"delivery rendering requires accepted physics QC, got {physics.get('status')}",
            "output_dir": str(render_dir),
            "report": str(render_report),
        }
    argv = [
        sys.executable,
        str(args.scripts_dir / "render_v7_candidate_deliverables.py"),
        "--replay-report",
        str(replay_result["report"]),
        "--physics-report",
        str(physics["report"]),
        "--manifest",
        str(target["manifest"]),
        "--annotations",
        str(target["annotations"]),
        "--output-dir",
        str(render_dir),
        "--output-fps",
        str(args.render_fps),
        "--caption-prefix",
        f"V7 accepted generated mesh: {replay_result['candidate_name']}",
    ]
    run_command(argv, bool(args.dry_run))
    if args.dry_run:
        return {
            "status": "dry_run",
            "output_dir": str(render_dir),
            "report": str(render_report),
        }
    rendered = load_json(render_report)
    return {
        "status": rendered.get("status"),
        "output_dir": str(render_dir),
        "report": str(render_report),
        "videos": rendered.get("videos"),
        "structural_qc": rendered.get("structural_qc"),
    }


def write_matrix(args: argparse.Namespace, results: list[dict]) -> None:
    if args.dry_run:
        return
    entries = []
    for result in results:
        entries.extend(
            [
                "--entry",
                "|".join(
                    [
                        f"{result['target_id']}_{result['candidate_name']}",
                        result["baseline_zbuffer_json"],
                        result["report"],
                        result["note"],
                    ]
                ),
            ]
        )
    if not entries:
        return
    argv = [
        sys.executable,
        str(args.scripts_dir / "summarize_v7_prior_replay_matrix.py"),
        *entries,
        "--output-json",
        str(args.output_root / "qc_v7_prior_candidate_batch_matrix.json"),
        "--output-md",
        str(args.output_root / "qc_v7_prior_candidate_batch_matrix.md"),
    ]
    run_command(argv, False)


def run(args: argparse.Namespace) -> dict:
    targets_payload = load_json(args.targets_json)
    targets = {target_id: validate_target(target_id, raw) for target_id, raw in targets_payload.items()}
    raw_candidates = candidate_rows(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for raw in raw_candidates:
        target_id, name, mesh, note = parse_candidate(raw)
        if target_id not in targets:
            raise RuntimeError(f"candidate {name} references unknown target_id: {target_id}")
        target = targets[target_id]
        result = run_replay(args, target_id, name, mesh, note, target)
        physics = run_physics(args, result, target)
        if physics is not None:
            result["physics_qc"] = physics
        deliverables = run_deliverables(args, result, physics, target)
        if deliverables is not None:
            result["deliverables"] = deliverables
        results.append(result)
    report = {
        "status": "dry_run" if args.dry_run else "ok",
        "method": "run_v7_prior_candidate_batch",
        "targets_json": str(args.targets_json),
        "candidate_files": [str(path) for path in args.candidate_file],
        "output_root": str(args.output_root),
        "replay_controls": {
            "samples": int(args.samples),
            "max_faces": int(args.max_faces),
            "vertex_splat_radius_px": int(args.vertex_splat_radius_px),
            "full_fidelity_zbuffer": bool(args.max_faces == 0),
        },
        "physics_enabled": bool(args.run_physics),
        "deliverable_rendering_enabled": bool(args.render_deliverables),
        "candidates": results,
    }
    report_path = args.output_root / "qc_v7_prior_candidate_batch.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_matrix(args, results)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-json", type=Path, default=REPO_DIR / "configs" / "v7_prior_replay_targets.json")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--candidate-file", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--vertex-splat-radius-px", type=int, default=0)
    parser.add_argument("--run-physics", action="store_true")
    parser.add_argument("--render-deliverables", action="store_true")
    parser.add_argument("--render-fps", type=float, default=6.0)
    parser.add_argument("--sdf-pitch-m", type=float, default=0.003)
    parser.add_argument("--max-selected-contact-abs-sdf-p95-m", type=float, default=0.006)
    parser.add_argument("--min-selected-contact-near-surface-fraction", type=float, default=0.75)
    parser.add_argument("--max-selected-contact-penetration-fraction", type=float, default=0.10)
    parser.add_argument("--max-full-hand-penetration-fraction", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
