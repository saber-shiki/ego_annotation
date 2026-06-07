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
    ]
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
    }
    if not args.dry_run:
        report = load_json(report_path)
        result["status"] = report.get("status")
        result["annotation_ready"] = bool(report.get("annotation_ready", False))
        result["metrics"] = report.get("metrics")
        result["pass"] = report.get("pass")
    return result


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
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for raw in args.candidate:
        target_id, name, mesh, note = parse_candidate(raw)
        if target_id not in targets:
            raise RuntimeError(f"candidate {name} references unknown target_id: {target_id}")
        results.append(run_replay(args, target_id, name, mesh, note, targets[target_id]))
    report = {
        "status": "dry_run" if args.dry_run else "ok",
        "method": "run_v7_prior_candidate_batch",
        "targets_json": str(args.targets_json),
        "output_root": str(args.output_root),
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
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
