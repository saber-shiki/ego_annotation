#!/usr/bin/env python3
"""Run an additive P12 SAM3D candidate beside a frozen TRELLIS reference.

This orchestrator does not import either model and does not edit canonical P12
artifacts. It invokes the existing SAM3D runner through an explicitly supplied
Python interpreter, then writes one backend-neutral raw-prior report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

P11_SCHEMA = "v19_experimental_p11_dual_geometry_inputs_v1"
P12_SCHEMA = "v19_experimental_p12_parallel_geometry_priors_v2"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def prepare_new_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty experiment output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, description: str) -> dict[str, Any]:
    path = require_file(path, description)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def safe_id(value: Any) -> str:
    raw = str(value or "unknown")
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    return cleaned.strip("_") or "unknown"


def git_head(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def gpu_preflight(device_id: int, min_free_mib: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={int(device_id)}",
            "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"unexpected nvidia-smi result for GPU {device_id}: {rows}")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 5:
        raise RuntimeError(f"cannot parse nvidia-smi result: {rows[0]}")
    free_mib = int(parts[3])
    record = {
        "physical_gpu_id": int(parts[0]),
        "name": parts[1],
        "used_mib_before_launch": int(parts[2]),
        "free_mib_before_launch": free_mib,
        "utilization_percent_before_launch": int(parts[4]),
        "required_free_mib": int(min_free_mib),
        "passed": bool(free_mib >= int(min_free_mib)),
    }
    if not record["passed"]:
        raise RuntimeError(
            f"GPU {device_id} has {free_mib} MiB free, below SAM3D preflight "
            f"requirement {min_free_mib} MiB; no model process was launched"
        )
    return record


def resolve_case_report(payload: dict[str, Any], expected_name: str | None) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return payload
    valid = [row for row in cases if isinstance(row, dict)]
    if expected_name is not None:
        matches = [row for row in valid if str(row.get("name")) == expected_name]
        if len(matches) == 1:
            return matches[0]
    if len(valid) == 1:
        return valid[0]
    raise RuntimeError(
        f"cannot select SAM3D case {expected_name!r} from report with "
        f"case names {[row.get('name') for row in valid]}"
    )


def validate_input_binding(
    candidate_report: dict[str, Any], expected_image: Path, expected_mask: Path
) -> dict[str, Any]:
    actual_image = require_file(Path(str(candidate_report.get("image", ""))), "SAM3D report image")
    actual_mask = require_file(Path(str(candidate_report.get("mask", ""))), "SAM3D report mask")
    bindings = {
        "image": {
            "expected": str(expected_image),
            "actual": str(actual_image),
            "expected_sha256": sha256_file(expected_image),
            "actual_sha256": sha256_file(actual_image),
        },
        "mask": {
            "expected": str(expected_mask),
            "actual": str(actual_mask),
            "expected_sha256": sha256_file(expected_mask),
            "actual_sha256": sha256_file(actual_mask),
        },
    }
    for name, record in bindings.items():
        record["byte_identical"] = bool(record["expected_sha256"] == record["actual_sha256"])
        if not record["byte_identical"]:
            raise RuntimeError(f"SAM3D {name} report is not bound to the P11 native input")
    return bindings


def sam3d_candidate_record(
    report_path: Path,
    expected_name: str | None,
    expected_image: Path,
    expected_mask: Path,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    payload = load_json(report_path)
    case = resolve_case_report(payload, expected_name)
    if case.get("status") != "ok":
        raise RuntimeError(f"SAM3D candidate report is not ok: {report_path}")
    mesh = require_file(Path(str(case.get("mesh", ""))), "SAM3D raw mesh")
    glb = require_file(Path(str(case.get("glb", ""))), "SAM3D raw GLB")
    gaussian_text = str(case.get("gaussian") or "")
    gaussian = require_file(Path(gaussian_text), "SAM3D Gaussian") if gaussian_text else None
    native_p3d = require_file(
        Path(str(case.get("native_pose_mesh_pytorch3d_camera") or "")),
        "SAM3D native-pose PyTorch3D-camera mesh",
    )
    native_opencv = require_file(
        Path(str(case.get("native_pose_mesh_opencv_camera") or "")),
        "SAM3D native-pose OpenCV-camera mesh",
    )
    native_contract = case.get("native_pose_contract")
    if not isinstance(native_contract, dict) or native_contract.get("quaternion_order") != "wxyz_scalar_first_pytorch3d":
        raise RuntimeError("SAM3D runner lacks the verified native pose coordinate contract")
    if native_contract.get("metric_status") != "native_monocular_scene_units_not_sensor_meters":
        raise RuntimeError("SAM3D runner incorrectly claims native pose is sensor metric")
    return {
        "source_model": "sam3d_objects",
        "status": "generated_native_raw_prior",
        "runner_report": str(report_path),
        "conditioning": {
            "input_kind": "full_scene_rgb_plus_binary_object_owned_mask",
            "image": file_record(expected_image, "P11 SAM3D image"),
            "mask": file_record(expected_mask, "P11 SAM3D mask"),
            "external_pointmap": None,
            "pre_model_crop": None,
            "pre_model_rotation": None,
            "pre_model_rectification": None,
        },
        "input_binding": validate_input_binding(case, expected_image, expected_mask),
        "native_outputs": {
            "raw_mesh": file_record(mesh, "SAM3D raw mesh"),
            "glb": file_record(glb, "SAM3D GLB"),
            "gaussian": file_record(gaussian, "SAM3D Gaussian") if gaussian is not None else None,
            "native_pose_mesh_pytorch3d_camera": file_record(native_p3d, "SAM3D native P3D mesh"),
            "native_pose_mesh_opencv_camera": file_record(native_opencv, "SAM3D native OpenCV mesh"),
            "mesh_stats": case.get("mesh_stats"),
            "native_pose": case.get("pose"),
            "native_pose_contract": native_contract,
        },
        "invocation": invocation,
        "semantics": {
            "mesh_frame": "sam3d_native_local_generator_frame",
            "native_pose_frame": "verified_pytorch3d_camera_then_explicit_opencv_bridge",
            "native_pose_metric_status": "native_monocular_scene_units_not_sensor_meters",
            "metric_scale_ready": False,
            "v19_canonical_frame_ready": False,
            "collision_surface_ready": False,
            "allowed_use": "raw geometry prior evaluation and later explicit P13 adapter only",
        },
    }


def trellis_candidate_record(report_path: Path, expected_crop: Path) -> dict[str, Any]:
    report = load_json(report_path)
    if report.get("status") != "ok":
        raise RuntimeError(f"TRELLIS reference report is not ok: {report_path}")
    image = require_file(Path(str(report.get("image", ""))), "TRELLIS report input image")
    if sha256_file(image) != sha256_file(expected_crop):
        raise RuntimeError("TRELLIS report input is not byte-identical to the P11 TRELLIS crop")
    mesh = require_file(Path(str(report.get("mesh", ""))), "TRELLIS raw mesh")
    gaussian_text = str(report.get("gaussian") or "")
    glb_text = str(report.get("glb") or "")
    return {
        "source_model": "trellis",
        "status": "frozen_existing_raw_prior_reference",
        "runner_report": str(report_path),
        "conditioning": {
            "input_kind": "object_isolated_rgba_crop",
            "image": file_record(expected_crop, "P11 TRELLIS crop"),
            "input_binding": {
                "report_image": str(image),
                "byte_identical": True,
            },
        },
        "native_outputs": {
            "raw_mesh": file_record(mesh, "TRELLIS raw mesh"),
            "gaussian": file_record(Path(gaussian_text), "TRELLIS Gaussian") if gaussian_text else None,
            "glb": file_record(Path(glb_text), "TRELLIS GLB") if glb_text else None,
            "mesh_stats": {
                "vertices": report.get("vertices"),
                "faces": report.get("faces"),
                "extent_model_units": report.get("extent_model_units"),
                "center_model_units": report.get("center_model_units"),
            },
        },
        "semantics": {
            "mesh_frame": "trellis_native_local_object_frame",
            "metric_scale_ready": False,
            "v19_canonical_frame_ready": False,
            "collision_surface_ready": False,
            "allowed_use": "frozen baseline raw geometry prior reference",
        },
        "source_artifacts_mutated": False,
    }


def write_failure_report(
    output_dir: Path,
    p11_report: Path,
    command: list[str],
    returncode: int,
    elapsed_s: float,
    log_path: Path,
) -> None:
    payload = {
        "schema": P12_SCHEMA,
        "status": "sam3d_runner_failed",
        "claim_scope": "experimental P12 failure record; no canonical artifact mutated",
        "p11_report": str(p11_report),
        "command": command,
        "command_shell_escaped": shlex.join(command),
        "returncode": int(returncode),
        "elapsed_s": float(elapsed_s),
        "log": str(log_path),
        "source_artifacts_mutated": False,
    }
    (output_dir / "p12_parallel_geometry_priors_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    p11_path = require_file(args.p11_report, "experimental P11 dual-input report")
    p11 = load_json(p11_path)
    if p11.get("schema") != P11_SCHEMA or p11.get("status") != "ok":
        raise RuntimeError(f"unsupported or failed P11 report: {p11_path}")
    contracts = p11.get("conditioning_contracts")
    if not isinstance(contracts, dict):
        raise RuntimeError("P11 report has no conditioning_contracts")
    sam_contract = contracts.get("sam3d_objects_native")
    trellis_contract = contracts.get("trellis_native")
    if not isinstance(sam_contract, dict) or not isinstance(trellis_contract, dict):
        raise RuntimeError("P11 report is missing SAM3D or TRELLIS native contract")

    sam_image = require_file(Path(str(sam_contract.get("image", ""))), "P11 SAM3D full RGB")
    sam_mask = require_file(Path(str(sam_contract.get("mask", ""))), "P11 SAM3D object-owned mask")
    trellis_crop = require_file(Path(str(trellis_contract.get("image", ""))), "P11 TRELLIS crop")
    if sam_contract.get("external_pointmap") is not None:
        raise RuntimeError("native P12 refuses a P11 contract with an external SAM3D pointmap")

    gpu_record = None
    if args.reuse_sam3d_report is None:
        gpu_record = gpu_preflight(args.cuda_visible_device, args.min_free_mib)

    output_dir = prepare_new_output_dir(args.output_dir)
    case_name = safe_id(
        args.case_name
        or f"{p11.get('case')}_{p11.get('object_id')}_frame_{int(p11.get('selected_frame_idx')):06d}"
    )

    command: list[str] | None = None
    elapsed_s: float | None = None
    log_path: Path | None = None
    if args.reuse_sam3d_report is not None:
        sam_report_path = require_file(args.reuse_sam3d_report, "reused SAM3D report")
        invocation = {
            "mode": "reuse_existing_report_for_contract_assembly",
            "elapsed_s": None,
            "command": None,
        }
    else:
        sam_python = require_file(args.sam3d_python, "SAM3D Python interpreter")
        sam_runner = require_file(args.sam3d_runner, "SAM3D runner")
        sam_repo = args.sam3d_repo.expanduser().resolve()
        if not sam_repo.is_dir():
            raise RuntimeError(f"missing SAM3D repository: {sam_repo}")
        sam_config = require_file(args.sam3d_config, "SAM3D pipeline config")
        sam_output_root = output_dir / "sam3d_objects_native"
        command = [
            str(sam_python),
            str(sam_runner),
            "--repo",
            str(sam_repo),
            "--config",
            str(sam_config),
            "--case",
            f"{case_name}|{sam_image}|{sam_mask}|{int(args.seed)}",
            "--output-dir",
            str(sam_output_root),
        ]
        if args.compile:
            command.append("--compile")
        log_path = output_dir / "sam3d_runner.log"
        start = time.perf_counter()
        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = str(int(args.cuda_visible_device))
        completed = subprocess.run(command, capture_output=True, text=True, env=child_env)
        elapsed_s = time.perf_counter() - start
        log_path.write_text(
            completed.stdout + ("\n--- STDERR ---\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            write_failure_report(
                output_dir, p11_path, command, completed.returncode, elapsed_s, log_path
            )
            raise RuntimeError(
                f"SAM3D runner failed with code {completed.returncode}; see {log_path}"
            )
        sam_report_path = require_file(
            sam_output_root / case_name / "qc_sam3d_objects_mesh_v7.json",
            "SAM3D case report",
        )
        invocation = {
            "mode": "external_unmodified_sam3d_runner",
            "command": command,
            "command_shell_escaped": shlex.join(command),
            "elapsed_s": float(elapsed_s),
            "log": str(log_path),
            "python": str(sam_python),
            "runner": file_record(sam_runner, "SAM3D runner"),
            "repo": str(sam_repo),
            "repo_git_head": git_head(sam_repo),
            "config": file_record(sam_config, "SAM3D config"),
            "compile": bool(args.compile),
            "seed": int(args.seed),
            "cuda_visible_devices": str(int(args.cuda_visible_device)),
            "gpu_preflight": gpu_record,
        }

    candidates: dict[str, Any] = {}
    if args.trellis_report is not None:
        trellis_report = require_file(args.trellis_report, "frozen TRELLIS reference report")
        candidates["trellis"] = trellis_candidate_record(trellis_report, trellis_crop)

    candidates["sam3d_objects"] = sam3d_candidate_record(
        sam_report_path,
        case_name,
        sam_image,
        sam_mask,
        invocation,
    )

    report = {
        "schema": P12_SCHEMA,
        "status": "ok",
        "method": "run_experimental_p12_parallel_geometry_priors",
        "claim_scope": (
            "raw P12 geometry priors only; no metric alignment, canonical-frame conversion, "
            "observed-surface fusion, pose fitting, collision promotion, or canonical mutation"
        ),
        "p11_report": str(p11_path),
        "case": p11.get("case"),
        "object_id": p11.get("object_id"),
        "selected_frame_idx": int(p11.get("selected_frame_idx")),
        "seed": int(args.seed),
        "candidates": candidates,
        "backend_neutral_contract": {
            "candidate_kind": "single_image_generated_raw_geometry_prior",
            "metric_alignment_applied": False,
            "observed_surface_fusion_applied": False,
            "pose_fit_applied": False,
            "multiview_support_applied": False,
            "collision_ready": False,
            "next_stage": "backend-correct P13 adapters with shared observed evidence and render-only semantics",
        },
        "canonical_trellis_rerun": False,
        "source_artifacts_mutated": False,
    }
    report_path = output_dir / "p12_parallel_geometry_priors_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p11-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trellis-report", type=Path, default=None)
    parser.add_argument("--reuse-sam3d-report", type=Path, default=None)
    parser.add_argument("--sam3d-python", type=Path, default=None)
    parser.add_argument("--sam3d-runner", type=Path, default=None)
    parser.add_argument("--sam3d-repo", type=Path, default=None)
    parser.add_argument("--sam3d-config", type=Path, default=None)
    parser.add_argument("--cuda-visible-device", type=int, default=None)
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=30000,
        help="One-shot preflight requirement; the wrapper never waits or polls for a GPU.",
    )
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.reuse_sam3d_report is None:
        missing = [
            name
            for name in [
                "sam3d_python",
                "sam3d_runner",
                "sam3d_repo",
                "sam3d_config",
                "cuda_visible_device",
            ]
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"model run requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    return args


if __name__ == "__main__":
    run(parse_args())
