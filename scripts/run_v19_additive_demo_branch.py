#!/usr/bin/env python3
"""Run the generic additive visual-v2 branch from a byte-bound contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def file(path, root):
    value = Path(path).resolve()
    if not value.is_relative_to(root) or not value.is_file():
        raise ValueError(f"missing/out-of-run demo input: {path}")
    return value


def verified_contract_inputs(contract, root):
    """Verify every byte-bound source before any probe or phase subprocess."""
    inputs = {name: file(item["path"], root) for name, item in contract["inputs"].items()}
    for name, path in inputs.items():
        if sha(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"demo input changed: {name}")
    for field, label in (("sam2_mask_inputs", "SAM2 appearance mask"),
                         ("raw_frame_inputs", "raw RGB")):
        for idx, item in contract.get(field, {}).items():
            path = file(item["path"], root)
            if sha(path) != item["sha256"]:
                raise ValueError(f"{label} changed: {idx}")
    return inputs


def python(path):
    value = Path(os.path.abspath(Path(path).expanduser()))
    if not value.is_file() or not os.access(value, os.X_OK):
        raise ValueError("demo interpreter unavailable")
    return value


def gpu_probe(index, minimum_free):
    query = ["nvidia-smi", f"--id={index}",
             "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"]
    row = subprocess.run(query, check=True, capture_output=True, text=True).stdout.strip()
    parts = [part.strip() for part in row.split(",")]
    if len(parts) != 7 or int(parts[0]) != index:
        raise RuntimeError("invalid demo GPU probe")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                           "--format=csv,noheader,nounits"], check=True,
                          capture_output=True, text=True).stdout.splitlines()
    record = {
        "gpu": parts,
        "existing_processes": [value for value in apps if parts[1] in value],
        "policy": "shared idle card explicitly user-authorized; require utilization<=5 and free memory floor",
    }
    if int(parts[6]) > 5 or int(parts[4]) < minimum_free:
        raise RuntimeError("authorized demo GPU unavailable: " + json.dumps(record))
    return record


def phase_event(name, argv, result, elapsed, log, input_paths, output_paths):
    return {
        "phase": name,
        "argv": list(map(str, argv)),
        "returncode": result.returncode,
        "elapsed_s": elapsed,
        "log": str(log),
        "input_sha256": {str(path): sha(path) for path in input_paths if Path(path).is_file()},
        "output_sha256": {str(path): sha(path) for path in output_paths if Path(path).is_file()},
        "authority": "demo_only_visual_hypothesis",
        "core_state_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = load(args.contract)
    root = Path(contract["run_root"]).resolve()
    object_id, demo_id = contract.get("object_id"), contract.get("demo_id")
    if not all(isinstance(value, str) and SAFE_ID.fullmatch(value) for value in (object_id, demo_id)):
        raise ValueError("unsafe demo object/id")
    receipts = root / "state/demo_visual" / object_id / demo_id / "receipts"
    if args.contract.resolve() != receipts / "launch_contract.json":
        raise ValueError("unexpected demo contract path")
    if sha(ROOT / "runtime/bundle_manifest.json") != contract["demo_bundle_manifest_sha256"]:
        raise ValueError("demo execution bundle changed")
    source_manifest = Path(contract["source_execution_bundle_manifest"]).resolve()
    if (not source_manifest.is_file()
            or sha(source_manifest) != contract["source_execution_bundle_manifest_sha256"]):
        raise ValueError("source execution bundle changed")
    if (contract.get("mode") != "additive_demo_visual_v2"
            or contract.get("authority") != "demo_only_visual_hypothesis"
            or contract.get("core_outputs_are_read_only") is not True):
        raise ValueError("invalid demo authority")
    if socket.gethostname() != contract["compute_host"]:
        raise ValueError("demo host mismatch")
    inputs = verified_contract_inputs(contract, root)

    measurement_root = root / "measurements/demo_visual" / object_id / demo_id
    state_root = root / "state/demo_visual" / object_id / demo_id
    render_root = root / "renders/demo_visual" / object_id / demo_id
    log_root = root / "logs/demo_visual" / object_id / demo_id
    appearance_root = measurement_root / "appearance_evidence"
    fit_root = measurement_root / "fit"
    pose_report = state_root / "pose_binding.json"
    visual_mesh = state_root / "visual_mesh.ply"
    result_path = receipts / "branch_result.json"
    failure_path = receipts / "failure.json"
    events_path = receipts / "events.jsonl"
    for path in (measurement_root, render_root, log_root, pose_report, visual_mesh,
                 result_path, failure_path, events_path):
        if path.exists():
            raise ValueError(f"demo output already exists: {path}")
    if any(path.resolve().is_relative_to(root / core) for path in
           (measurement_root, state_root, render_root, log_root)
           for core in ("measurements/object_geometry", "measurements/pose_fits",
                        "measurements/geometry_completion", "state/render_state")):
        raise ValueError("demo output resolves into core tree")

    cfg = load(contract["runtime_config"])
    py = python(cfg["main_python"])
    probe = gpu_probe(int(contract["physical_gpu"]), int(contract["minimum_free_mib"]))
    log_root.mkdir(parents=True)
    events = []
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(contract["physical_gpu"]),
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    appearance_manifest = appearance_root / "appearance_mask_manifest.json"
    anchor_report = appearance_root / "anchor_decision.json"
    scale_args = (["--visual-scale-record", inputs["visual_scale_record"]]
                  if "visual_scale_record" in inputs else [])
    commands = [
        ("D00_appearance_anchor", [py, ROOT / "scripts/build_v19_demo_visual_inputs.py",
          "--run-root", root, "--raw-frame-manifest", inputs["raw_frame_manifest"],
          "--annotations", inputs["annotations"], "--sam2-track", inputs["sam2_track"],
          "--object-id", object_id, "--output-dir", appearance_root],
         [inputs["raw_frame_manifest"], inputs["annotations"], inputs["sam2_track"],
          *[Path(item["path"]) for item in contract.get("sam2_mask_inputs", {}).values()],
          *[Path(item["path"]) for item in contract.get("raw_frame_inputs", {}).values()]],
         [appearance_manifest, anchor_report]),
        ("D01_visual_binding", [py, ROOT / "scripts/build_v19_demo_pose_binding.py",
          "--p15-report", inputs["p15"], "--p14-report", inputs["p14"],
          "--annotations", inputs["annotations"], "--completion-report", inputs["completion"],
          "--appearance-mask-manifest", appearance_manifest, "--object-id", object_id,
          "--output-mesh", visual_mesh, "--output", pose_report, *scale_args],
         [inputs["p15"], inputs["p14"], inputs["annotations"], inputs["completion"], appearance_manifest],
         [pose_report, visual_mesh]),
        ("D02_visual_fit", [py, ROOT / "scripts/fit_v20_visual_contact_demo.py",
          "--annotations", inputs["annotations"], "--appearance-mask-manifest", appearance_manifest,
          "--pose-report", pose_report, "--mano-bridge", inputs["mano_bridge"],
          "--hawor-params", inputs["hawor"], "--mano-models", cfg["mano_models"],
          "--object-id", object_id, "--device", "cuda:0",
          "--interaction-judgments", inputs["p17"], "--render-width",
          int(contract["render_width"]), "--output-dir", fit_root],
         [inputs["annotations"], appearance_manifest, pose_report, inputs["mano_bridge"],
          inputs["hawor"], inputs["judgment"], inputs["p17"]],
         [fit_root / "demo_state.npz", fit_root / "fit_report.json", fit_root / "contact_hypotheses.json"]),
        ("D03_visual_render", [py, ROOT / "scripts/render_v20_visual_contact_demo.py",
          "--state", fit_root / "demo_state.npz", "--source-video", inputs["source_video"],
          "--output-dir", render_root],
         [fit_root / "demo_state.npz", inputs["source_video"]],
         [render_root / "interaction_demo.mp4", render_root / "contact_detail.mp4",
          render_root / "interaction_demo.rrd", render_root / "render_report.json"]),
    ]
    try:
        for name, argv, phase_inputs, outputs in commands:
            started = time.monotonic()
            log = log_root / f"{name}.log"
            with log.open("x") as stream:
                completed = subprocess.run(list(map(str, argv)), cwd=ROOT, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
            event_outputs = list(outputs)
            if name == "D00_appearance_anchor" and appearance_root.exists():
                event_outputs.extend(path for path in appearance_root.rglob("*")
                                     if path.is_file() and path not in event_outputs)
            event = phase_event(name, argv, completed, time.monotonic() - started,
                                log, phase_inputs, event_outputs)
            events.append(event)
            with events_path.open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            if completed.returncode:
                raise RuntimeError(f"{name} failed: {log}")
            missing = [str(path) for path in outputs if not path.is_file()]
            if missing:
                raise RuntimeError(f"{name} outputs missing: {missing}")
    except Exception as exc:
        if not failure_path.exists():
            failure_path.write_text(json.dumps({
                "schema": "v19_additive_demo_visual_v2_failure",
                "error": str(exc), "events": events,
                "authority": "demo_only_visual_hypothesis", "core_state_modified": False,
            }, indent=2) + "\n")
        raise

    delivered = {}
    for base in (measurement_root, state_root, render_root, log_root):
        for path in base.rglob("*"):
            if path.is_file() and path != result_path:
                delivered[str(path.relative_to(root))] = sha(path)
    result = {
        "schema": "v19_additive_demo_branch_v2",
        "status": "completed_pending_visual_review",
        "authority": "demo_only_visual_hypothesis",
        "annotation_ready": False,
        "physical_authority": False,
        "core_state_modified": False,
        "input_contract": str(args.contract.resolve()),
        "input_contract_sha256": sha(args.contract),
        "demo_bundle_manifest_sha256": contract["demo_bundle_manifest_sha256"],
        "source_execution_bundle_manifest_sha256": contract["source_execution_bundle_manifest_sha256"],
        "gpu_preflight": probe,
        "events": events,
        "outputs": delivered,
        "sam3d_reconditioning": contract["sam3d_reconditioning"],
        "final_package_assembled": False,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(result_path)


if __name__ == "__main__":
    main()
