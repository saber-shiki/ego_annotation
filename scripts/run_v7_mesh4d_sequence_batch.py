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


def run_command(argv: list[str], dry_run: bool) -> None:
    print(" ".join(shell_token(arg) for arg in argv))
    if dry_run:
        return
    subprocess.run(argv, check=True)


def shell_token(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def case_from_report(path: Path, report: dict, mapping: dict[str, str]) -> tuple[str, str]:
    sequence_name = str(report.get("model_name") or report.get("sequence_name") or path.parent.name)
    for prefix, target in mapping.items():
        if sequence_name.startswith(prefix) or path.as_posix().find(prefix) >= 0:
            return prefix, str(target)
    raise RuntimeError(f"could not map Mesh4D sequence to target: {path}")


def discover_mesh4d_reports(root: Path, mapping: dict[str, str]) -> list[dict]:
    if not root.exists():
        return []
    rows = []
    for report_path in sorted(root.glob("**/qc_mesh4d_sequence_v7.json")):
        report = load_json(report_path)
        if report.get("status") != "ok":
            continue
        case, target = case_from_report(report_path, report, mapping)
        rows.append(
            {
                "target_id": target,
                "case": case,
                "name": "mesh4d_" + safe_name(report_path.parent.relative_to(root).as_posix()),
                "mesh4d_json": str(report_path),
                "note": f"Mesh4D animated six-frame mesh hypothesis for {case}",
            }
        )
    return rows


def safe_name(raw: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)
    if not value:
        raise RuntimeError("empty safe name")
    return value


def archive_sequence(args: argparse.Namespace, row: dict, target: dict, out_dir: Path) -> dict:
    aligned_archive = out_dir / "aligned_mesh4d_meshes_world.npz"
    align_report = out_dir / "qc_aligned_mesh4d_sequence_v7.json"
    argv = [
        sys.executable,
        str(args.scripts_dir / "archive_mesh4d_sequence_prior_v7.py"),
        "--mesh4d-json",
        row["mesh4d_json"],
        "--observed-mesh-archive",
        str(target["observed_mesh_archive"]),
        "--output-mesh-archive",
        str(aligned_archive),
        "--output-json",
        str(align_report),
        "--samples",
        str(args.samples),
    ]
    run_command(argv, bool(args.dry_run))
    result = {
        "aligned_mesh_archive": str(aligned_archive),
        "alignment_report": str(align_report),
    }
    if not args.dry_run:
        align = load_json(align_report)
        result["alignment_status"] = align.get("status")
        result["alignment_metrics"] = {
            "alignment_bidirectional_p95_m": align.get("alignment_bidirectional_p95_m"),
            "visible_surface_coverage_p95_m": align.get("visible_surface_coverage_p95_m"),
            "hidden_surface_conflict_p95_m": align.get("hidden_surface_conflict_p95_m"),
        }
    return result


def replay_sequence(args: argparse.Namespace, row: dict, target: dict, archive: dict, out_dir: Path) -> dict:
    replay_dir = out_dir / "replay_qc"
    replay_report = replay_dir / "qc_v7_generated_prior_replay.json"
    argv = [
        sys.executable,
        str(args.scripts_dir / "run_v7_generated_prior_replay_qc.py"),
        "--prealigned-mesh-archive",
        archive["aligned_mesh_archive"],
        "--prealigned-report",
        archive["alignment_report"],
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
        str(replay_dir),
        "--samples",
        str(args.samples),
    ]
    if args.max_faces:
        argv.extend(["--max-faces", str(args.max_faces)])
    if args.vertex_splat_radius_px:
        argv.extend(["--vertex-splat-radius-px", str(args.vertex_splat_radius_px)])
    run_command(argv, bool(args.dry_run))
    result = {
        "output_dir": str(replay_dir),
        "report": str(replay_report),
        "baseline_zbuffer_json": str(target["baseline_zbuffer_json"]),
        "replay_controls": {
            "samples": int(args.samples),
            "max_faces": int(args.max_faces),
            "vertex_splat_radius_px": int(args.vertex_splat_radius_px),
            "full_fidelity_zbuffer": bool(args.max_faces == 0),
        },
    }
    if not args.dry_run:
        replay = load_json(replay_report)
        result["status"] = replay.get("status")
        result["annotation_ready"] = bool(replay.get("annotation_ready", False))
        result["metrics"] = replay.get("metrics")
        result["pass"] = replay.get("pass")
    return result


def run(args: argparse.Namespace) -> dict:
    targets_raw = load_json(args.targets_json)
    targets = {key: validate_target(key, value) for key, value in targets_raw.items()}
    mapping = {str(key): str(value) for key, value in load_json(args.case_targets_json).items()}
    rows = discover_mesh4d_reports(args.mesh4d_root, mapping)
    if args.require_sequences and not rows:
        raise RuntimeError(f"no Mesh4D reports found under {args.mesh4d_root}")
    results = []
    for row in rows:
        target = targets[row["target_id"]]
        out_dir = args.output_root / row["target_id"] / safe_name(row["name"])
        archive = archive_sequence(args, row, target, out_dir)
        replay = replay_sequence(args, row, target, archive, out_dir)
        results.append({**row, "output_dir": str(out_dir), "archive": archive, "replay": replay})
    status_counts = {}
    for row in results:
        status = str(row.get("replay", {}).get("status", "dry_run"))
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "status": "ok",
        "method": "run_v7_mesh4d_sequence_batch",
        "mesh4d_root": str(args.mesh4d_root),
        "sequence_count": int(len(rows)),
        "status_counts": status_counts,
        "sequences": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "qc_v7_mesh4d_sequence_batch.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh4d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets-json", type=Path, default=REPO_DIR / "configs" / "v7_prior_replay_targets.json")
    parser.add_argument("--case-targets-json", type=Path, default=REPO_DIR / "configs" / "v7_mesh4d_case_targets.json")
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--vertex-splat-radius-px", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-sequences", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
