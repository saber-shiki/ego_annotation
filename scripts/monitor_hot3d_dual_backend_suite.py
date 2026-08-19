#!/usr/bin/env python3
"""Continuously summarize progress for the five-case HOT3D backend suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def file_done(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def blocker_paths(run_root: Path) -> list[str]:
    root = run_root / "state/runtime_blockers"
    if not root.is_dir():
        return []
    return [str(path) for path in sorted(root.glob("*.json")) if file_done(path)]


def parse_agent_done(path: Path) -> dict[str, Any] | None:
    if not file_done(path):
        return None
    values: dict[str, Any] = {"path": str(path)}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    try:
        values["exit_code_int"] = int(values.get("exit_code", "-1"))
    except ValueError:
        values["exit_code_int"] = -1
    return values


def log_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "bytes": 0, "tail": ""}
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - 16384))
        text = handle.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(size),
        "mtime_unix_s": path.stat().st_mtime,
        "tail": lines[-1][-1000:] if lines else "",
    }


def stages_for(case: dict[str, Any]) -> list[tuple[str, Path]]:
    run_root = Path(case["run_root"])
    case_id = str(case["case_id"])
    object_id = str(case["object_id"])
    experiment = run_root / "experiments/sam3d_trellis_controlled"
    return [
        ("P00 startup", run_root / "input/runtime_input_contract.json"),
        ("P01 frames", run_root / "input/raw_frame_manifest/manifest.json"),
        ("P03 depth+camera", run_root / "state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz"),
        ("P04 MANO", run_root / "measurements/hand_candidates/hawor_world/hawor_world_hands.npz"),
        ("P06 detector", run_root / f"measurements/object_candidates/object_box_prompts_owlv2/{object_id}/v19_owlv2_object_box_prompt_report.json"),
        ("P07 SAM2", run_root / "measurements/object_tracks/sam2_owlv2_box_points/qc_sam2_multiobject_points.json"),
        ("P09 visible geometry", run_root / f"measurements/object_geometry/visible_geometry/{object_id}/annotations_v19_visible_geometry.json"),
        ("P11 shared evidence", run_root / f"measurements/geometry_completion/rigid_evidence/{case_id}/{object_id}/evidence_bundle/evidence_bundle_report.json"),
        ("P12 TRELLIS", experiment / "P12_trellis/qc_trellis_shape_v3.json"),
        ("P12 SAM3D", experiment / "P12_parallel/p12_parallel_geometry_priors_report.json"),
        ("P13 controlled adaptation", experiment / "P13_controlled/p13_controlled_geometry_prior_ab_report.json"),
        ("P13 SAM3D dual mesh", experiment / "P13_sam3d_dual/p13_dual_mesh_geometry_prior_report.json"),
        ("P15 shared pose", experiment / "P15_observed_pose_graph/v19_rigid_object_pose_graph_report.json"),
        ("D15b shared signed geometry", experiment / "P15b_shared_signed_geometry/shared_signed_geometry_completion_report.json"),
        ("D16 shared MANO/object", experiment / "P16_shared_mano_object/v18_mano_object_constraint_state.json"),
        ("shared P17/P18/P18b", experiment / "P16b_shared_p17_p18_tail/shared_p17_p18_tail_report.json"),
        ("D17 branch states", experiment / "P15_layered_states/p14_p15_layered_render_state_adapter_report.json"),
        ("render SAM3D", experiment / "renders/sam3d/p14_p15_layered_full_mano_render_manifest.json"),
        ("render TRELLIS", experiment / "renders/trellis/p14_p15_layered_full_mano_render_manifest.json"),
        ("final publish", run_root / "SUITE_DONE.json"),
    ]


def inspect_case(case: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(case["run_root"])
    rows = [{"name": name, "path": str(path), "complete": file_done(path)} for name, path in stages_for(case)]
    completed = sum(int(row["complete"]) for row in rows)
    done = parse_agent_done(Path(case["agent_done"]))
    blockers = blocker_paths(run_root)
    final_complete = bool(rows[-1]["complete"])
    if final_complete:
        status = "complete"
    elif blockers:
        status = "blocked"
    elif done is not None:
        status = "failed" if int(done.get("exit_code_int", -1)) != 0 else "incomplete_exit"
    elif run_root.exists():
        status = "running"
    else:
        status = "queued"
    next_stage = next((row["name"] for row in rows if not row["complete"]), None)
    return {
        **case,
        "status": status,
        "completed_stage_count": completed,
        "stage_count": len(rows),
        "progress_percent": round(100.0 * completed / len(rows), 1),
        "next_stage": next_stage,
        "stages": rows,
        "runtime_blockers": blockers,
        "agent_done": done,
        "log": log_summary(Path(case["log"])),
    }


def tmux_windows(session: str) -> list[dict[str, str]]:
    result = subprocess.run(
        ["tmux", "list-windows", "-t", session, "-F", "#{window_index}|#{window_name}|#{pane_dead}|#{pane_current_command}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            rows.append({"index": parts[0], "name": parts[1], "pane_dead": parts[2], "command": parts[3]})
    return rows


def build_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    cases = [inspect_case(case) for case in manifest.get("cases", [])]
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["status"]] = counts.get(case["status"], 0) + 1
    return {
        "schema": "hot3d_dual_backend_suite_live_progress_v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "updated_unix_s": time.time(),
        "suite_root": manifest.get("suite_root"),
        "tmux_session": manifest.get("tmux_session"),
        "status_counts": counts,
        "case_count": len(cases),
        "all_complete": bool(cases) and all(case["status"] == "complete" for case in cases),
        "all_terminal": bool(cases) and all(case["status"] in {"complete", "blocked", "failed", "incomplete_exit"} for case in cases),
        "tmux_windows": tmux_windows(str(manifest.get("tmux_session", ""))),
        "cases": cases,
    }


def render_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        "# HOT3D SAM3D / TRELLIS suite live progress",
        "",
        f"- Updated: `{snapshot['updated_utc']}`",
        f"- tmux: `{snapshot['tmux_session']}`",
        f"- Counts: `{json.dumps(snapshot['status_counts'], sort_keys=True)}`",
        "",
        "| Case | Object | GPU | Status | Progress | Next stage | Log |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for case in snapshot["cases"]:
        tail = str(case["log"].get("tail") or "").replace("|", "\\|").replace("\n", " ")[-160:]
        lines.append(
            f"| `{case['case_id']}` | `{case['object_id']}` | {case['gpu_id']} | **{case['status']}** | "
            f"{case['completed_stage_count']}/{case['stage_count']} ({case['progress_percent']}%) | "
            f"{case['next_stage'] or '-'} | {tail} |"
        )
    lines.extend(["", "## Per-case stages", ""])
    for case in snapshot["cases"]:
        lines.append(f"### {case['case_id']}")
        lines.append("")
        for stage in case["stages"]:
            lines.append(f"- [{'x' if stage['complete'] else ' '}] {stage['name']}: `{stage['path']}`")
        if case["runtime_blockers"]:
            lines.append(f"- Runtime blockers: `{case['runtime_blockers']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def compact_signature(snapshot: dict[str, Any]) -> str:
    payload = [
        {
            "case": case["case_id"],
            "status": case["status"],
            "completed": case["completed_stage_count"],
            "next": case["next_stage"],
            "log_bytes": case["log"].get("bytes", 0),
        }
        for case in snapshot["cases"]
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def append_event(path: Path, snapshot: dict[str, Any], reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_utc": snapshot["updated_utc"],
        "reason": reason,
        "status_counts": snapshot["status_counts"],
        "cases": [
            {
                "case_id": case["case_id"],
                "status": case["status"],
                "progress": f"{case['completed_stage_count']}/{case['stage_count']}",
                "next_stage": case["next_stage"],
                "log_tail": case["log"].get("tail", ""),
            }
            for case in snapshot["cases"]
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    manifest_path = args.suite_manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    progress_dir = args.output_dir.expanduser().resolve()
    progress_json = progress_dir / "progress.json"
    progress_md = progress_dir / "progress.md"
    live_log = progress_dir / "live_progress.log"
    last_signature = None
    last_event = 0.0
    while True:
        snapshot = build_snapshot(manifest)
        atomic_write(progress_json, json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
        markdown = render_markdown(snapshot)
        atomic_write(progress_md, markdown)
        signature = compact_signature(snapshot)
        now = time.time()
        reason = None
        if signature != last_signature:
            reason = "state_change"
        elif now - last_event >= float(args.heartbeat_seconds):
            reason = "heartbeat"
        if reason is not None:
            append_event(live_log, snapshot, reason)
            last_event = now
            last_signature = signature
        print("\n" + markdown.split("## Per-case stages", 1)[0], flush=True)
        if args.stop_when_terminal and snapshot["all_terminal"]:
            break
        time.sleep(float(args.interval_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=20.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0)
    parser.add_argument("--stop-when-terminal", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
