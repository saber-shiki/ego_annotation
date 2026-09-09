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

from v19_runtime_assets import verify_bundle

ROOT = Path(__file__).resolve().parents[1]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RUN_IDENTITY_KEYS = ("run_id", "run_root", "input_video_sha256", "bundle_manifest_sha256")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def expected_contract_paths(contract, root):
    expected = {}

    def bind(path, digest, label):
        resolved = file(path, root)
        old = expected.get(resolved)
        if old is not None and old != digest:
            raise ValueError(f"conflicting expected digest for {label}: {resolved}")
        expected[resolved] = digest

    for name, item in contract["inputs"].items():
        bind(item["path"], item["sha256"], name)
    for field, label in (("sam2_mask_inputs", "SAM2 appearance mask"),
                         ("raw_frame_inputs", "raw RGB")):
        for idx, item in contract.get(field, {}).items():
            bind(item["path"], item["sha256"], f"{label} {idx}")
    return expected


def verify_hashes(expected, label):
    for path, digest in expected.items():
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"{label} changed: {path}")


def verified_contract_inputs(contract, root):
    """Verify every byte-bound source; retained as a narrow testable seam."""
    expected = expected_contract_paths(contract, root)
    verify_hashes(expected, "contract input")
    return {name: Path(item["path"]).resolve() for name, item in contract["inputs"].items()}


def verify_live_run_identity(contract, root):
    live_path = root / "input/runtime_input_contract.json"
    live = load(file(live_path, root))
    live_identity = {key: live.get(key) for key in RUN_IDENTITY_KEYS}
    if live_identity != contract.get("run_identity"):
        raise ValueError("launch contract run identity is stale")
    if live.get("case") != contract.get("case_id"):
        raise ValueError("live runtime input case mismatch")
    if live.get("run_root") != str(root):
        raise ValueError("live runtime input run root mismatch")
    return live


def verify_execution_bundle(contract, execution_root=None):
    execution_root = ROOT if execution_root is None else execution_root
    expected_config = (execution_root / "runtime/runtime_config.json").resolve()
    if Path(contract.get("runtime_config", "")).resolve() != expected_config:
        raise ValueError("runtime config must be the execution bundle runtime_config.json")
    manifest = verify_bundle(execution_root, asset_role="demo")
    manifest_path = execution_root / "runtime/bundle_manifest.json"
    if sha(manifest_path) != contract.get("demo_bundle_manifest_sha256"):
        raise ValueError("demo execution bundle manifest identity mismatch")
    expected = {manifest_path.resolve(): contract["demo_bundle_manifest_sha256"]}
    for name, digest in manifest["files"].items():
        expected[(execution_root / name).resolve()] = digest
    return manifest, expected


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


def atomic_publish(staging, destination):
    staging, destination = Path(staging), Path(destination)
    if not staging.is_dir() or destination.exists():
        raise ValueError("atomic phase publication requires new staging and destination directories")
    staging.rename(destination)


def preserve_failed_staging(staging, phase):
    staging = Path(staging)
    if not staging.exists():
        return None
    failed = staging.with_name(f".{phase}.failed")
    if failed.exists():
        raise ValueError(f"failed debug namespace already exists: {failed}")
    staging.rename(failed)
    return failed


def write_failure(path, error, events, failed_debug=None):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "v19_additive_demo_visual_v2_failure",
        "error": str(error),
        "events": events,
        "failed_debug_namespace": None if failed_debug is None else str(failed_debug),
        "authority": "demo_only_visual_hypothesis",
        "core_state_modified": False,
    }, indent=2) + "\n")


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
    if (args.contract.resolve() != receipts / "launch_contract.json"
            or receipts.resolve() != receipts):
        raise ValueError("unexpected demo contract path")
    failure_path = receipts / "failure.json"
    result_path = receipts / "branch_result.json"
    events_path = receipts / "events.jsonl"
    events = []
    active_staging = None
    active_phase = None

    try:
        if (contract.get("mode") != "additive_demo_visual_v2"
                or contract.get("authority") != "demo_only_visual_hypothesis"
                or contract.get("core_outputs_are_read_only") is not True):
            raise ValueError("invalid demo authority")
        if not isinstance(contract.get("case_id"), str) or not SAFE_ID.fullmatch(contract["case_id"]):
            raise ValueError("unsafe demo case id")
        _, bundle_expected = verify_execution_bundle(contract)
        source_manifest = Path(contract["source_execution_bundle_manifest"]).resolve()
        if (not source_manifest.is_file()
                or sha(source_manifest) != contract["source_execution_bundle_manifest_sha256"]):
            raise ValueError("source execution bundle changed")
        verify_live_run_identity(contract, root)
        if contract.get("sam3d_reconditioning", {}).get("enabled") is not False:
            raise ValueError("demo SAM3D reconditioning is unimplemented; enabled hook rejected")
        if socket.gethostname() != contract["compute_host"]:
            raise ValueError("demo host mismatch")
        original_expected = expected_contract_paths(contract, root)
        original_expected[args.contract.resolve()] = sha(args.contract)
        original_expected[source_manifest] = contract["source_execution_bundle_manifest_sha256"]
        verify_hashes(original_expected, "contract input")
        inputs = {name: Path(item["path"]).resolve() for name, item in contract["inputs"].items()}
        cfg = load(contract["runtime_config"])
        py = python(cfg["main_python"])

        measurement_root = root / "measurements/demo_visual" / object_id / demo_id
        state_root = root / "state/demo_visual" / object_id / demo_id
        render_parent = root / "renders/demo_visual" / object_id
        render_root = render_parent / demo_id
        log_root = root / "logs/demo_visual" / object_id / demo_id
        appearance_root = measurement_root / "appearance_evidence"
        binding_root = state_root / "visual_binding"
        pose_report = binding_root / "pose_binding.json"
        visual_mesh = binding_root / "visual_mesh.ply"
        fit_root = measurement_root / "fit"
        for path in (measurement_root, render_root, log_root, binding_root,
                     result_path, failure_path, events_path):
            if path.exists():
                raise ValueError(f"demo output already exists: {path}")
        for path, namespace in ((measurement_root, "measurements"), (state_root, "state"),
                                (render_root, "renders"), (log_root, "logs")):
            expected_namespace = root / namespace / "demo_visual"
            if not path.resolve().is_relative_to(expected_namespace):
                raise ValueError("demo output escapes demo_visual namespace")
        if any(path.resolve().is_relative_to(root / core) for path in
               (measurement_root, state_root, render_root, log_root)
               for core in ("measurements/object_geometry", "measurements/pose_fits",
                            "measurements/geometry_completion", "state/render_state")):
            raise ValueError("demo output resolves into core tree")
        probe = gpu_probe(int(contract["physical_gpu"]), int(contract["minimum_free_mib"]))
        measurement_root.mkdir(parents=True)
        render_parent.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True)
        produced = {}
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(contract["physical_gpu"]),
               "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        scale_args = (["--visual-scale-record", inputs["visual_scale_record"]]
                      if "visual_scale_record" in inputs else [])

        d00_stage = measurement_root / ".D00_appearance_anchor.staging"
        d00_manifest = d00_stage / "appearance_mask_manifest.json"
        d00_anchor = d00_stage / "anchor_decision.json"
        d01_stage = state_root / ".D01_visual_binding.staging"
        d01_pose = d01_stage / "pose_binding.json"
        d01_mesh = d01_stage / "visual_mesh.ply"
        d02_stage = measurement_root / ".D02_visual_fit.staging"
        d03_stage = render_parent / f".{demo_id}.D03_visual_render.staging"
        phases = [
            {
                "name": "D00_appearance_anchor", "staging": d00_stage, "final": appearance_root,
                "argv": [py, ROOT / "scripts/build_v19_demo_visual_inputs.py",
                         "--run-root", root, "--raw-frame-manifest", inputs["raw_frame_manifest"],
                         "--annotations", inputs["annotations"], "--sam2-track", inputs["sam2_track"],
                         "--object-id", object_id, "--case-id", contract["case_id"],
                         "--track-id", contract["track_id"], "--output-dir", d00_stage,
                         "--published-output-dir", appearance_root],
                "required": [d00_manifest, d00_anchor],
            },
            {
                "name": "D01_visual_binding", "staging": d01_stage, "final": binding_root,
                "argv": [py, ROOT / "scripts/build_v19_demo_pose_binding.py",
                         "--p15-report", inputs["p15"], "--p14-report", inputs["p14"],
                         "--annotations", inputs["annotations"], "--completion-report", inputs["completion"],
                         "--appearance-mask-manifest", appearance_root / "appearance_mask_manifest.json",
                         "--object-id", object_id, "--output-mesh", d01_mesh,
                         "--output", d01_pose, "--published-output-mesh", visual_mesh, *scale_args],
                "required": [d01_pose, d01_mesh],
            },
            {
                "name": "D02_visual_fit", "staging": d02_stage, "final": fit_root,
                "argv": [py, ROOT / "scripts/fit_v20_visual_contact_demo.py",
                         "--annotations", inputs["annotations"],
                         "--appearance-mask-manifest", appearance_root / "appearance_mask_manifest.json",
                         "--pose-report", pose_report, "--mano-bridge", inputs["mano_bridge"],
                         "--hawor-params", inputs["hawor"], "--mano-models", cfg["mano_models"],
                         "--object-id", object_id, "--device", "cuda:0",
                         "--interaction-judgments", inputs["judgment"], "--render-width",
                         int(contract["render_width"]), "--output-dir", d02_stage],
                "required": [d02_stage / "demo_state.npz", d02_stage / "fit_report.json",
                             d02_stage / "contact_hypotheses.json"],
            },
            {
                "name": "D03_visual_render", "staging": d03_stage, "final": render_root,
                "argv": [py, ROOT / "scripts/render_v20_visual_contact_demo.py",
                         "--state", fit_root / "demo_state.npz", "--source-video", inputs["source_video"],
                         "--output-dir", d03_stage],
                "required": [d03_stage / "interaction_demo.mp4", d03_stage / "contact_detail.mp4",
                             d03_stage / "interaction_demo.rrd", d03_stage / "render_report.json"],
            },
        ]

        for phase in phases:
            active_phase = phase["name"]
            active_staging = phase["staging"]
            if active_staging.exists() or phase["final"].exists():
                raise ValueError(f"phase namespace already exists: {active_phase}")
            # Original, bundle and preceding generated bytes are immutable at every boundary.
            verify_live_run_identity(contract, root)
            verify_hashes(original_expected, "contract input before phase")
            verify_hashes(bundle_expected, "execution bundle before phase")
            verify_hashes(produced, "generated phase input before consumer")
            started = time.monotonic()
            log = log_root / f"{active_phase}.log"
            event = {
                "phase": active_phase,
                "argv": list(map(str, phase["argv"])),
                "returncode": None,
                "elapsed_s": None,
                "log": str(log),
                "input_sha256": {
                    **{str(path): digest for path, digest in original_expected.items()},
                    **{str(path): digest for path, digest in produced.items()},
                },
                "output_sha256": {},
                "authority": "demo_only_visual_hypothesis",
                "core_state_modified": False,
            }
            events.append(event)
            with log.open("x") as stream:
                completed = subprocess.run(list(map(str, phase["argv"])), cwd=ROOT, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
            event.update(returncode=completed.returncode, elapsed_s=time.monotonic() - started)
            verify_live_run_identity(contract, root)
            verify_hashes(original_expected, "contract input after phase")
            verify_hashes(bundle_expected, "execution bundle after phase")
            verify_hashes(produced, "generated phase input after consumer")
            missing = [str(path) for path in phase["required"] if not path.is_file()]
            if completed.returncode or missing:
                raise RuntimeError(f"{active_phase} failed or outputs missing: returncode={completed.returncode}, missing={missing}, log={log}")
            staged_hashes = {
                path.relative_to(active_staging): sha(path)
                for path in sorted(active_staging.rglob("*")) if path.is_file()
            }
            output_hashes = {(phase["final"] / relative).resolve(): digest
                             for relative, digest in staged_hashes.items()}
            atomic_publish(active_staging, phase["final"])
            # Publication must not become a chance to accept altered output bytes.
            try:
                verify_hashes(output_hashes, "published phase output")
            except Exception:
                preserve_failed_staging(phase["final"], active_phase)
                raise
            active_staging = None
            produced.update(output_hashes)
            event["output_sha256"] = {str(path): digest for path, digest in output_hashes.items()}
            with events_path.open("a") as stream:
                stream.write(json.dumps(event) + "\n")

        verify_hashes(original_expected, "contract input at branch closure")
        verify_hashes(produced, "generated output at branch closure")
        delivered = {str(path.relative_to(root)): digest for path, digest in produced.items()}
        for base in (state_root / "receipts", log_root):
            for path in base.rglob("*"):
                if path.is_file() and path != result_path:
                    delivered[str(path.relative_to(root))] = sha(path)
        verify_hashes(original_expected, "contract input after closure")
        verify_hashes(produced, "generated output after closure")
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
    except Exception as exc:
        failed_debug = None
        try:
            if active_staging is not None:
                failed_debug = preserve_failed_staging(active_staging, active_phase or "preflight")
        except Exception as preserve_error:
            exc = RuntimeError(f"{exc}; failed to preserve debug staging: {preserve_error}")
        write_failure(failure_path, exc, events, failed_debug)
        raise


if __name__ == "__main__":
    main()
