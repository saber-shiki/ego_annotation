#!/usr/bin/env python3
"""Launch preflight for the isolated current-user V19 runtime bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

EXPECTED_INPUT_SHA256 = "7a9baf0553e5dcfb4411b6cfabbe3a734f815ee4c5b014bdd965b9e547ec2310"
EXPECTED_DINOV2_SHA256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
EXPECTED_HASHES = {
    "sam2": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    "owlv2": "e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7",
    "unidepth": "ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6",
    "mano_left": "c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30",
    "mano_right": "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767",
}
FORBIDDEN_WORDS = ("yiwen", "Workbench", ".memory", "EPISTEMIC", "parent", "GT", "evaluation", "evaluator", "ablation")
FORBIDDEN_BUNDLE_PATHS = ("/mnt/user-home/yiwen", "/mnt/truenas-user-home/yiwen", "/home/yiwen")
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_result(command: list[str], cwd: Path, timeout: int = 120) -> dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    proc = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "elapsed_s": time.monotonic() - started,
    }


def require_hash(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "missing", "expected_sha256": expected}
    actual = sha256_file(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "status": "ok" if actual == expected else "hash_mismatch",
    }


def verify_bundle_manifest(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "RUNTIME_BUNDLE_MANIFEST.json"
    if not manifest_path.is_file():
        return {"status": "missing_manifest", "path": str(manifest_path)}
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for row in manifest.get("files", []):
        path = bundle / row["path"]
        if not path.is_file():
            failures.append({"path": row["path"], "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != row["sha256"]:
            failures.append({"path": row["path"], "reason": "hash_mismatch", "actual": actual, "expected": row["sha256"]})
    return {
        "status": "ok" if not failures else "failed",
        "path": str(manifest_path),
        "source_revision": manifest.get("source_revision"),
        "wilor_source_revision": manifest.get("wilor_source_revision"),
        "declared_file_count": manifest.get("file_count"),
        "failures": failures,
    }


def prompt_isolation(bundle: Path) -> dict[str, Any]:
    paths = [
        bundle / "runtime" / "v19_runtime_spec.md",
        bundle / "configs" / "v19_agent_system_prompt.md",
        bundle / ".pi" / "prompts" / "v19-run.md",
    ]
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for word in FORBIDDEN_WORDS:
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", re.IGNORECASE)
            if pattern.search(text):
                findings.append({"path": str(path), "word": word})
    return {"status": "ok" if not findings else "failed", "findings": findings, "paths": [str(p) for p in paths]}


def bundle_path_isolation(bundle: Path) -> dict[str, Any]:
    findings = []
    checked = 0
    for path in sorted(p for p in bundle.rglob("*") if p.is_file() and not p.is_symlink()):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        checked += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        for forbidden in FORBIDDEN_BUNDLE_PATHS:
            if forbidden.lower() in text.lower():
                findings.append({"path": str(path), "pattern": forbidden})
    return {"status": "ok" if not findings else "failed", "checked_text_files": checked, "findings": findings}


def load_manifest(path: Path, expected_status_prefix: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    payload = json.loads(path.read_text())
    status = str(payload.get("status", ""))
    return {
        "path": str(path),
        "status": "ok" if status.startswith(expected_status_prefix) else "unexpected_status",
        "declared_status": status,
        "source_commit": payload.get("source_commit"),
        "model": payload.get("model"),
    }


def video_info(path: Path, expected: dict[str, Any], expected_sha256: str | None) -> dict[str, Any]:
    import cv2

    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "status": "open_failed"}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    actual = {"fps": fps, "frame_count": frames, "width": width, "height": height}
    digest = sha256_file(path)
    metadata_ok = actual == expected
    hash_ok = expected_sha256 is None or digest == expected_sha256
    return {
        "path": str(path),
        "status": "ok" if metadata_ok and hash_ok else "metadata_or_hash_mismatch",
        "expected": expected,
        "actual": actual,
        "bytes": path.stat().st_size,
        "expected_sha256": expected_sha256,
        "sha256": digest,
        "hash_check_enabled": expected_sha256 is not None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = args.bundle.resolve()
    checks: dict[str, Any] = {}
    checks["bundle_integrity"] = verify_bundle_manifest(bundle)
    checks["prompt_isolation"] = prompt_isolation(bundle)
    checks["bundle_path_isolation"] = bundle_path_isolation(bundle)
    expected_video = {
        "fps": float(args.expected_fps),
        "frame_count": int(args.expected_frame_count),
        "width": int(args.expected_width),
        "height": int(args.expected_height),
    }
    expected_input_sha256 = None if str(args.expected_input_sha256).lower() in {"", "none", "skip"} else str(args.expected_input_sha256)
    checks["input_video"] = video_info(args.input_video, expected_video, expected_input_sha256)
    input_files = sorted(p.name for p in args.input_video.parent.iterdir() if p.is_file()) if args.input_video.parent.is_dir() else []
    allowed_input_files = {"input.mp4", "INPUT_PROVENANCE.json", *args.allow_input_sidecar}
    unexpected_input_files = sorted(set(input_files).difference(allowed_input_files))
    checks["input_isolation"] = {
        "status": "ok" if not unexpected_input_files else "unexpected_sidecars",
        "files": input_files,
        "allowed_files": sorted(allowed_input_files),
        "unexpected_files": unexpected_input_files,
    }
    checks["fresh_run_root"] = {
        "path": str(args.run_root),
        "status": "ok" if not args.run_root.exists() else "already_exists",
    }

    hashes = {
        "sam2": require_hash(args.sam2_checkpoint, EXPECTED_HASHES["sam2"]),
        "owlv2": require_hash(args.owlv2_model / "model.safetensors", EXPECTED_HASHES["owlv2"]),
        "unidepth": require_hash(args.unidepth_model / "model.safetensors", EXPECTED_HASHES["unidepth"]),
        "mano_left": require_hash(bundle / "third_party/WiLoR/mano_data/MANO_LEFT.pkl", EXPECTED_HASHES["mano_left"]),
        "mano_right": require_hash(bundle / "third_party/WiLoR/mano_data/MANO_RIGHT.pkl", EXPECTED_HASHES["mano_right"]),
    }
    checks["fixed_asset_hashes"] = {"status": "ok" if all(row["status"] == "ok" for row in hashes.values()) else "failed", "assets": hashes}
    checks["hawor_manifest"] = load_manifest(args.hawor_asset_root / "HAWOR_RUNTIME_MANIFEST.json", "installed_import_validated")
    checks["trellis_manifest"] = load_manifest(args.trellis_model / "TRELLIS_RUNTIME_MANIFEST.json", "installed_import_validated")
    dino = args.torch_home / "hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
    dino_hash = sha256_file(dino) if dino.is_file() else None
    checks["trellis_dinov2_cache"] = {
        "path": str(dino),
        "status": "ok" if dino_hash == EXPECTED_DINOV2_SHA256 else "missing_incomplete_or_hash_mismatch",
        "bytes": dino.stat().st_size if dino.is_file() else None,
        "expected_sha256": EXPECTED_DINOV2_SHA256,
        "sha256": dino_hash,
    }

    import_commands = {
        "main": [str(args.main_python), "-c", "import cv2,open3d,smplx,torch,transformers,trimesh; from PIL import Image; print(torch.__version__, cv2.__version__, open3d.__version__, transformers.__version__)"],
        "unidepth": [str(args.unidepth_python), "-c", f"import sys,torch,numpy,cv2; sys.path.insert(0,{str(args.unidepth_repo)!r}); import unidepth; print(torch.__version__,numpy.__version__,cv2.__version__)"],
        "hawor": [str(args.hawor_python), "-c", f"import sys; sys.path.insert(0,{str(args.hawor_repo)!r}); import torch; import cv2,droid_backends,lietorch,mmcv,pytorch3d,smplx; print(torch.__version__,cv2.__version__)"],
        "trellis": [str(args.trellis_python), "-c", "import kaolin,spconv,torch,transformers,trimesh,xformers; print(torch.__version__,transformers.__version__,kaolin.__version__,spconv.__version__)"],
    }
    import_results = {name: command_result(command, bundle) for name, command in import_commands.items()}
    checks["interpreter_imports"] = {
        "status": "ok" if all(row["returncode"] == 0 for row in import_results.values()) else "failed",
        "results": import_results,
    }

    bundle_manifest = json.loads((bundle / "RUNTIME_BUNDLE_MANIFEST.json").read_text())
    help_results = []
    for name in bundle_manifest.get("scripts", []):
        path = bundle / "scripts" / name
        if path.suffix == ".sh":
            result = command_result(["bash", "-n", str(path)], bundle)
        else:
            python = args.main_python
            if name.startswith("run_unidepth_"):
                python = args.unidepth_python
            elif name == "export_hawor_world.py":
                python = args.hawor_python
            elif name == "remote_run_trellis_shape_v3.py":
                python = args.trellis_python
            result = command_result([str(python), str(path), "--help"], bundle)
        result["script"] = name
        help_results.append(result)
    checks["script_cli_contracts"] = {
        "status": "ok" if all(row["returncode"] == 0 for row in help_results) else "failed",
        "failures": [row for row in help_results if row["returncode"] != 0],
        "checked": len(help_results),
    }

    failed = [name for name, row in checks.items() if row.get("status") != "ok"]
    report = {
        "status": "ready_for_runtime_agent_launch" if not failed else "blocked_launch_preflight",
        "claim_scope": "infrastructure and isolated-runtime launch preflight only; no prediction output",
        "failed_checks": failed,
        "bundle": str(bundle),
        "input_video": str(args.input_video),
        "run_root": str(args.run_root),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-python", type=Path, required=True)
    parser.add_argument("--unidepth-python", type=Path, required=True)
    parser.add_argument("--unidepth-repo", type=Path, required=True)
    parser.add_argument("--unidepth-model", type=Path, required=True)
    parser.add_argument("--hawor-python", type=Path, required=True)
    parser.add_argument("--hawor-repo", type=Path, required=True)
    parser.add_argument("--hawor-asset-root", type=Path, required=True)
    parser.add_argument("--trellis-python", type=Path, required=True)
    parser.add_argument("--trellis-model", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--owlv2-model", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", default=EXPECTED_INPUT_SHA256, help="Expected input hash, or 'none'/'skip' to disable the hash check")
    parser.add_argument("--expected-width", type=int, default=1408)
    parser.add_argument("--expected-height", type=int, default=1408)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--expected-frame-count", type=int, default=150)
    parser.add_argument("--allow-input-sidecar", action="append", default=[], help="Additional prediction-side sensor/provenance filename allowed next to input.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
