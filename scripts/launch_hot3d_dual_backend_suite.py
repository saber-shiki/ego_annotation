#!/usr/bin/env python3
"""Launch all HOT3D controlled backend cases in one dedicated tmux session."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["tmux", *args], check=check, text=True, capture_output=True)


def has_session(name: str) -> bool:
    return tmux("has-session", "-t", name, check=False).returncode == 0


def verify_bundle(bundle: Path) -> dict[str, Any]:
    manifest_path = require_file(bundle / "RUNTIME_BUNDLE_MANIFEST.json", "runtime bundle manifest")
    manifest = load_json(manifest_path)
    failures = []
    for row in manifest.get("files", []):
        path = bundle / str(row.get("path"))
        if not path.is_file():
            failures.append({"path": row.get("path"), "reason": "missing"})
        elif sha256_file(path) != row.get("sha256"):
            failures.append({"path": row.get("path"), "reason": "hash_mismatch"})
    if failures:
        raise RuntimeError(f"bundle integrity failed: {failures[:10]}")
    require_file(bundle / "runtime/hot3d_dual_backend_runtime_spec.md", "suite runtime spec")
    require_file(bundle / "configs/hot3d_dual_backend_agent_system_prompt.md", "suite system prompt")
    return manifest


def same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)


def validate_case(case: dict[str, Any], bundle: Path) -> dict[str, Any]:
    required = [
        "case_id", "object_id", "window", "gpu_id", "input_video", "sensor_calibration_metadata",
        "run_root", "preflight_report", "log", "agent_done", "target_hint", "target_exclusions",
    ]
    missing = [key for key in required if key not in case]
    if missing:
        raise RuntimeError(f"case lacks fields {missing}: {case}")
    input_video = require_file(Path(case["input_video"]), "input video")
    sensor = require_file(Path(case["sensor_calibration_metadata"]), "sensor calibration metadata")
    preflight_path = require_file(Path(case["preflight_report"]), "case preflight report")
    preflight = load_json(preflight_path)
    run_root = Path(case["run_root"]).expanduser().resolve()
    if run_root.exists():
        raise RuntimeError(f"run root is not fresh: {run_root}")
    if preflight.get("status") != "ready_for_runtime_agent_launch":
        raise RuntimeError(f"preflight is not ready for {case['case_id']}: {preflight.get('status')}")
    preflight_checks = preflight.get("checks") if isinstance(preflight.get("checks"), dict) else {}
    checks = {
        "bundle_integrity": preflight_checks.get("bundle_integrity", {}).get("status") == "ok",
        "fixed_asset_hashes": preflight_checks.get("fixed_asset_hashes", {}).get("status") == "ok",
        "dinov2_checkpoint": preflight_checks.get("trellis_dinov2_cache", {}).get("status") == "ok",
        "dinov2_source": preflight_checks.get("trellis_dinov2_offline_source", {}).get("status") == "ok",
        "script_cli_contracts": preflight_checks.get("script_cli_contracts", {}).get("status") == "ok",
        "fresh_run_root": preflight_checks.get("fresh_run_root", {}).get("status") == "ok",
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise RuntimeError(
            f"preflight lacks required immutable/offline/fresh launch checks for {case['case_id']}: {failed_checks}"
        )
    sam3d_contract = preflight_checks.get("sam3d_contract") if isinstance(preflight_checks.get("sam3d_contract"), dict) else {}
    interpreter_imports = preflight_checks.get("interpreter_imports") if isinstance(preflight_checks.get("interpreter_imports"), dict) else {}
    import_results = interpreter_imports.get("results") if isinstance(interpreter_imports.get("results"), dict) else {}
    sam3d_import = import_results.get("sam3d") if isinstance(import_results.get("sam3d"), dict) else {}
    if sam3d_contract.get("status") != "ok" or int(sam3d_import.get("returncode", -1)) != 0:
        raise RuntimeError(
            f"preflight lacks a passing frozen SAM3D repository/activation/import contract for {case['case_id']}"
        )
    if (
        sam3d_contract.get("checks", {}).get("moge_checkpoint") is not True
        or "SAM3D_IMPORT_NATIVE_POSE_AND_OFFLINE_ASSETS_OK" not in str(sam3d_import.get("stdout_tail") or "")
    ):
        raise RuntimeError(
            f"preflight lacks hash-bound offline MoGe/DINOv2 proof for {case['case_id']}"
        )
    for key, expected in (("bundle", bundle), ("input_video", input_video), ("run_root", run_root)):
        if not same_path(preflight.get(key, ""), expected):
            raise RuntimeError(f"preflight {key} mismatch for {case['case_id']}: {preflight.get(key)} != {expected}")
    if Path(case["log"]).exists() or Path(case["agent_done"]).exists():
        raise RuntimeError(f"case log/sentinel already exists for {case['case_id']}")
    return {
        **case,
        "input_video": str(input_video),
        "sensor_calibration_metadata": str(sensor),
        "preflight_report": str(preflight_path),
        "run_root": str(run_root),
        "gpu_id": int(case["gpu_id"]),
    }


def prompt_text(case: dict[str, Any]) -> str:
    return f"""Execute the controlled HOT3D dual-backend runtime through D19 now.

Authoritative instructions:
- runtime/hot3d_dual_backend_runtime_spec.md for the whole run
- runtime/v19_runtime_spec.md only for common P00-P11

Exact launch bindings:
INPUT_VIDEO={case['input_video']}
RUN_ROOT={case['run_root']}
CASE_ID={case['case_id']}
OBJECT_ID={case['object_id']}
TRACK_ID={case['object_id']}
GPU_ID={case['gpu_id']}
SENSOR_CALIBRATION_METADATA={case['sensor_calibration_metadata']}
SENSOR_SOURCE_VIDEO={case['input_video']}
SENSOR_CALIBRATION_AUTHORITY=prediction_side_sensor_metadata
SENSOR_FRAME_INTRINSICS_KEY=<empty>
TARGET_HINT={case['target_hint']}
TARGET_EXCLUSIONS={case['target_exclusions']}
ANCHOR_GUIDANCE={case.get('anchor_guidance', 'none; select only after inspecting this fresh run P09 review')}
LAUNCH_PREFLIGHT_REPORT={case['preflight_report']}

The target hint is semantic only. Confirm it from your own raw/review image inspection.
Anchor guidance, when present, names a frame to scrutinize in this fresh run; it is not an
artifact, mask, pose, or permission to skip the fresh P09 candidate review and decision.
Use the dedicated GPU above; another case owns every other declared suite GPU. Do not stop
after planning and do not execute the common document's P12-P21. Finish only when the D19
finalizer writes {case['run_root']}/SUITE_DONE.json, or after writing a named hard blocker.
"""


def send_command(session: str, window: str, command: str) -> None:
    target = f"{session}:{window}"
    subprocess.run(["tmux", "send-keys", "-t", target, "-l", command], check=True)
    subprocess.run(["tmux", "send-keys", "-t", target, "Enter"], check=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite_manifest_path = args.suite_manifest.expanduser().resolve()
    suite = load_json(require_file(suite_manifest_path, "suite manifest"))
    bundle = Path(suite["bundle"]).expanduser().resolve()
    bundle_manifest = verify_bundle(bundle)
    session = str(suite["tmux_session"])
    if has_session(session):
        raise RuntimeError(f"tmux session already exists: {session}")
    raw_cases = suite.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RuntimeError("suite manifest contains no cases")
    cases = [validate_case(case, bundle) for case in raw_cases if isinstance(case, dict)]
    if len(cases) != len(raw_cases):
        raise RuntimeError("suite manifest has non-object case rows")
    gpu_ids = [case["gpu_id"] for case in cases]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise RuntimeError(f"suite GPU assignments are not unique: {gpu_ids}")
    windows = [str(case["window"]) for case in cases]
    if len(set(windows)) != len(windows) or "monitor" in windows:
        raise RuntimeError(f"invalid/duplicate case window names: {windows}")

    suite_root = Path(suite["suite_root"]).expanduser().resolve()
    prompt_root = suite_root / "prompts"
    session_dir = suite_root / "pi_sessions"
    progress_dir = suite_root / "progress"
    logs_dir = suite_root / "logs"
    for directory in (prompt_root, session_dir, progress_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    provider = str(suite.get("pi_provider") or os.environ.get("PI_PROVIDER") or "dexgem-responses")
    model = str(suite.get("pi_model") or os.environ.get("PI_MODEL") or "gpt-5.6-sol")
    thinking = str(suite.get("pi_thinking") or os.environ.get("PI_REASONING_LEVEL") or "max")
    pi_executable = str(suite.get("pi_executable") or shutil_which_pi())
    system_prompt = bundle / "configs/hot3d_dual_backend_agent_system_prompt.md"

    tmux("new-session", "-d", "-s", session, "-n", "monitor", "-c", str(bundle))
    tmux("set-option", "-t", session, "remain-on-exit", "on")
    monitor_log = logs_dir / "monitor.log"
    monitor_command = (
        f"cd {shlex.quote(str(bundle))} && export PYTHONUNBUFFERED=1 && "
        f"{shlex.quote('/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python')} "
        f"scripts/monitor_hot3d_dual_backend_suite.py --suite-manifest {shlex.quote(str(suite_manifest_path))} "
        f"--output-dir {shlex.quote(str(progress_dir))} --interval-seconds 20 --heartbeat-seconds 300 "
        f"--stop-when-terminal 2>&1 | tee -a {shlex.quote(str(monitor_log))}"
    )
    send_command(session, "monitor", monitor_command)

    launch_rows = []
    for case in cases:
        window = str(case["window"])
        tmux("new-window", "-d", "-t", session, "-n", window, "-c", str(bundle))
        prompt_path = prompt_root / f"{case['case_id']}.txt"
        prompt_path.write_text(prompt_text(case), encoding="utf-8")
        log_path = Path(case["log"]).expanduser().resolve()
        done_path = Path(case["agent_done"]).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        done_path.parent.mkdir(parents=True, exist_ok=True)
        pi_session_id = f"hot3d-dual-{case['case_id']}-{suite.get('suite_id', 'v1')}"
        command = (
            "set -o pipefail; "
            f"cd {shlex.quote(str(bundle))} && "
            "export PYTHONUNBUFFERED=1 TORCH_HOME=/mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub && "
            f"{shlex.quote(pi_executable)} --print --approve --no-context-files --no-skills --no-extensions "
            f"--provider {shlex.quote(provider)} --model {shlex.quote(model)} --thinking {shlex.quote(thinking)} "
            "--mode text --tools read,bash,edit,write "
            f"--system-prompt \"$(cat {shlex.quote(str(system_prompt))})\" "
            f"--session-dir {shlex.quote(str(session_dir))} --session-id {shlex.quote(pi_session_id)} "
            f"\"$(cat {shlex.quote(str(prompt_path))})\" "
            f"2>&1 | tee -a {shlex.quote(str(log_path))}; "
            "rc=${PIPESTATUS[0]}; "
            f"tmp={shlex.quote(str(done_path) + '.tmp')}; "
            f"printf 'exit_code=%s\\ncase_id=%s\\nrun_root=%s\\nmodel=%s/%s\\nfinished_utc=%s\\n' "
            f"\"$rc\" {shlex.quote(str(case['case_id']))} {shlex.quote(str(case['run_root']))} "
            f"{shlex.quote(provider)} {shlex.quote(model)} \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \"$tmp\"; "
            f"mv \"$tmp\" {shlex.quote(str(done_path))}; exit $rc"
        )
        send_command(session, window, command)
        launch_rows.append(
            {
                "case_id": case["case_id"],
                "window": window,
                "gpu_id": case["gpu_id"],
                "prompt": str(prompt_path),
                "log": str(log_path),
                "agent_done": str(done_path),
                "pi_session_id": pi_session_id,
            }
        )

    launched = {
        "status": "launched",
        "launched_utc": datetime.now(timezone.utc).isoformat(),
        "suite_manifest": str(suite_manifest_path),
        "suite_manifest_sha256": sha256_file(suite_manifest_path),
        "bundle": str(bundle),
        "bundle_status": bundle_manifest.get("status"),
        "bundle_source_revision": bundle_manifest.get("source_revision"),
        "tmux_session": session,
        "monitor_window": "monitor",
        "monitor_log": str(monitor_log),
        "progress_json": str(progress_dir / "progress.json"),
        "progress_markdown": str(progress_dir / "progress.md"),
        "live_progress_log": str(progress_dir / "live_progress.log"),
        "pi_provider": provider,
        "pi_model": model,
        "pi_thinking": thinking,
        "cases": launch_rows,
    }
    launch_path = suite_root / "launch_state.json"
    launch_path.write_text(json.dumps(launched, indent=2) + "\n", encoding="utf-8")
    how_to = suite_root / "HOW_TO_MONITOR.txt"
    how_to.write_text(
        "\n".join(
            [
                f"tmux attach -t {session}",
                f"tmux select-window -t {session}:monitor",
                f"watch -n 5 cat {progress_dir / 'progress.md'}",
                f"tail -f {progress_dir / 'live_progress.log'}",
                f"tail -f {logs_dir / '<case>.log'}",
                f"tmux list-windows -t {session}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(launched, indent=2))
    return launched


def shutil_which_pi() -> str:
    import shutil

    value = shutil.which("pi")
    if not value:
        raise RuntimeError("pi executable not found")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
