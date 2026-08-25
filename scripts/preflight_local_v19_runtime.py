#!/usr/bin/env python3
"""Launch preflight for the isolated current-user V19 runtime bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

EXPECTED_INPUT_SHA256 = "7a9baf0553e5dcfb4411b6cfabbe3a734f815ee4c5b014bdd965b9e547ec2310"
EXPECTED_DINOV2_SHA256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
EXPECTED_DINOV2_HUBCONF_SHA256 = "c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64"
EXPECTED_MOGE_SHA256 = "da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f"
EXPECTED_DA3_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
EXPECTED_DA3_MODEL_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
EXPECTED_HASHES = {
    "sam2": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    "owlv2": "e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7",
    "unidepth": "ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6",
    "mano_left": "c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30",
    "mano_right": "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767",
}
EXPECTED_SAM3D_CONFIG_SHA256 = "53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c"
EXPECTED_SAM3D_REPO_REVISION = "f91db411c50efee93d8db7aeb323885650f6f722"
FORBIDDEN_WORDS = ("yiwen", "Workbench", ".memory", "EPISTEMIC", "parent", "GT", "evaluation", "evaluator", "ablation")
_FORBIDDEN_USER = "".join(("yi", "wen"))
FORBIDDEN_BUNDLE_PATHS = tuple(
    prefix + _FORBIDDEN_USER
    for prefix in ("/mnt/user-home/", "/mnt/truenas-user-home/", "/home/")
)
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
    env["PYTHONDONTWRITEBYTECODE"] = "1"
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


def sam3d_contract_check(args: argparse.Namespace) -> dict[str, Any]:
    python = args.sam3d_python.expanduser().resolve()
    repo = args.sam3d_repo.expanduser().resolve()
    config = args.sam3d_config.expanduser().resolve()
    activation = args.sam3d_activation.expanduser().resolve()
    moge_checkpoint = args.sam3d_moge_checkpoint.expanduser().resolve()
    inference_module = repo / "notebook/inference.py"
    revision_result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else None
    config_sha256 = sha256_file(config) if config.is_file() else None
    checks = {
        "python_executable": bool(python.is_file() and os.access(python, os.X_OK)),
        "repo_directory": repo.is_dir(),
        "inference_module": inference_module.is_file(),
        "activation_script": activation.is_file(),
        "repo_revision": revision == EXPECTED_SAM3D_REPO_REVISION,
        "config_sha256": config_sha256 == EXPECTED_SAM3D_CONFIG_SHA256,
        "moge_checkpoint": bool(
            moge_checkpoint.is_file() and sha256_file(moge_checkpoint) == EXPECTED_MOGE_SHA256
        ),
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "python": str(python),
        "repo": str(repo),
        "inference_module": str(inference_module),
        "activation": str(activation),
        "config": str(config),
        "expected_repo_revision": EXPECTED_SAM3D_REPO_REVISION,
        "actual_repo_revision": revision,
        "expected_config_sha256": EXPECTED_SAM3D_CONFIG_SHA256,
        "actual_config_sha256": config_sha256,
        "moge_checkpoint": str(moge_checkpoint),
        "expected_moge_sha256": EXPECTED_MOGE_SHA256,
        "actual_moge_sha256": sha256_file(moge_checkpoint) if moge_checkpoint.is_file() else None,
        "checks": checks,
    }


def sam3d_import_command(args: argparse.Namespace) -> list[str]:
    python = args.sam3d_python.expanduser().resolve()
    repo = args.sam3d_repo.expanduser().resolve()
    activation = args.sam3d_activation.expanduser().resolve()
    runner = args.bundle.expanduser().resolve() / "scripts/remote_run_sam3d_objects_mesh_v7.py"
    moge_checkpoint = args.sam3d_moge_checkpoint.expanduser().resolve()
    dino_repo = args.torch_home.expanduser().resolve() / "hub/facebookresearch_dinov2_main"
    dino_checkpoint = args.torch_home.expanduser().resolve() / "hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
    code = f"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
repo = Path({str(repo)!r})
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / 'notebook'))
from inference import Inference
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
spec = importlib.util.spec_from_file_location('frozen_sam3d_runner', {str(runner)!r})
runner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner_module)
moge_state = runner_module.install_offline_moge_checkpoint(Path({str(moge_checkpoint)!r}))
dino_state = runner_module.install_offline_dinov2_hub(Path({str(dino_repo)!r}), Path({str(dino_checkpoint)!r}))
assert moge_state['network_resolution_allowed'] is False
assert moge_state['checkpoint_sha256'] == {EXPECTED_MOGE_SHA256!r}
assert dino_state['network_resolution_allowed'] is False
assert dino_state['checkpoint_sha256'] == {EXPECTED_DINOV2_SHA256!r}
assert dino_state['expected_hubconf_sha256'] == {EXPECTED_DINOV2_HUBCONF_SHA256!r}
try:
    runner_module.install_offline_dinov2_hub(Path({str(dino_repo)!r}), Path('/definitely/missing/dino.pth'))
except RuntimeError:
    pass
else:
    raise AssertionError('missing explicit DINO checkpoint did not fail closed')
vertices = np.asarray([[0.2, -0.3, 0.4], [-0.7, 0.1, 0.5]], dtype=np.float64)
q = np.asarray([0.75, -0.2, 0.3, 0.55], dtype=np.float64)
q /= np.linalg.norm(q)
t = np.asarray([0.13, -0.21, 0.94], dtype=np.float64)
s = np.asarray([0.17, 0.21, 0.14], dtype=np.float64)
actual, actual_cv, _ = runner_module.apply_native_pose(vertices, q, t, s)
transform = compose_transform(
    torch.tensor(s[None], dtype=torch.float64),
    quaternion_to_matrix(torch.tensor(q[None], dtype=torch.float64)),
    torch.tensor(t[None], dtype=torch.float64),
)
expected = transform.transform_points(torch.tensor(vertices[None], dtype=torch.float64))[0].numpy()
error = float(np.max(np.abs(actual - expected)))
assert error < 1.0e-12, error
assert np.allclose(actual_cv, actual @ np.diag([-1.0, -1.0, 1.0]))
print('SAM3D_IMPORT_NATIVE_POSE_AND_OFFLINE_ASSETS_OK', Inference.__module__, error, moge_state['checkpoint_sha256'], dino_state['hubconf_sha256'])
"""
    shell = "\n".join(
        [
            "set -euo pipefail",
            "set +u",
            f"source {shlex.quote(str(activation))}",
            "set -u",
            f"test \"$(readlink -f \"$CONDA_PREFIX/bin/python\")\" = {shlex.quote(str(python.resolve()))}",
            f"cd {shlex.quote(str(repo))}",
            f"CUDA_VISIBLE_DEVICES='' {shlex.quote(str(python))} -c {shlex.quote(code)}",
        ]
    )
    return ["bash", "-lc", shell]


def da3_contract_check(args: argparse.Namespace) -> dict[str, Any]:
    python = args.da3_python.expanduser().resolve()
    repo = args.da3_repo.expanduser().resolve()
    model = args.da3_model.expanduser().resolve()
    manifest = json.loads((args.bundle.expanduser().resolve() / "RUNTIME_BUNDLE_MANIFEST.json").read_text())
    offline = manifest.get("offline_model_assets") if isinstance(manifest.get("offline_model_assets"), dict) else {}
    declared = offline.get("da3_nested") if isinstance(offline.get("da3_nested"), dict) else {}
    declared_files = declared.get("model_files") if isinstance(declared.get("model_files"), list) else []
    failures = []
    for row in declared_files:
        path = Path(str(row.get("path") or ""))
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
        elif sha256_file(path) != row.get("sha256"):
            failures.append({"path": str(path), "reason": "sha256_mismatch"})
    revision_result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True)
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else None
    checks = {
        "python_executable": bool(python.is_file() and os.access(python, os.X_OK)),
        "repo_directory": repo.is_dir(),
        "api_module": (repo / "src/depth_anything_3/api.py").is_file(),
        "repo_revision": revision == declared.get("repository_revision") == EXPECTED_DA3_SOURCE_REVISION,
        "model_directory": model.is_dir() and str(model) == declared.get("model_path"),
        "model_file_hashes": bool(declared_files and not failures),
        "model_id": declared.get("model_id") == "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "model_revision": declared.get("model_revision") == EXPECTED_DA3_MODEL_REVISION,
        "runtime_manifest_hash_bound": any(
            row.get("relative_path") == "DA3_RUNTIME_MANIFEST.json" for row in declared_files
        ),
        "license": declared.get("license") == "CC BY-NC 4.0",
        "network_forbidden": declared.get("network_resolution_allowed") is False,
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "python": str(python),
        "repo": str(repo),
        "actual_repo_revision": revision,
        "model": str(model),
        "model_files": declared_files,
        "file_failures": failures,
        "checks": checks,
    }


def verify_bundle_manifest(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "RUNTIME_BUNDLE_MANIFEST.json"
    if not manifest_path.is_file():
        return {"status": "missing_manifest", "path": str(manifest_path)}
    manifest = json.loads(manifest_path.read_text())
    failures = []
    declared_paths = {
        str(row.get("path")) for row in manifest.get("files", []) if isinstance(row, dict)
    }
    actual_paths = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "RUNTIME_BUNDLE_MANIFEST.json"
    }
    for undeclared in sorted(actual_paths - declared_paths):
        failures.append({"path": undeclared, "reason": "undeclared_file"})
    for missing_declared in sorted(declared_paths - actual_paths):
        failures.append({"path": missing_declared, "reason": "declared_file_missing"})
    for row in manifest.get("files", []):
        path = bundle / row["path"]
        if not path.is_file():
            failures.append({"path": row["path"], "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != row["sha256"]:
            failures.append({"path": row["path"], "reason": "hash_mismatch", "actual": actual, "expected": row["sha256"]})
    if manifest.get("source_worktree_dirty") is not False:
        failures.append({"path": "source_worktree_dirty", "reason": "immutable_bundle_not_clean"})
    offline_assets = manifest.get("offline_model_assets") if isinstance(manifest.get("offline_model_assets"), dict) else {}
    expected_assets = {
        "dinov2_source_hubconf": EXPECTED_DINOV2_HUBCONF_SHA256,
        "dinov2_checkpoint": EXPECTED_DINOV2_SHA256,
        "sam3d_moge_checkpoint": EXPECTED_MOGE_SHA256,
    }
    for name, expected in expected_assets.items():
        row = offline_assets.get(name) if isinstance(offline_assets.get(name), dict) else {}
        path = Path(str(row.get("path") or ""))
        if row.get("sha256") != expected:
            failures.append({"path": f"offline_model_assets.{name}", "reason": "declared_sha256_mismatch"})
        elif not path.is_file():
            failures.append({"path": str(path), "reason": "offline_asset_missing"})
        elif sha256_file(path) != expected:
            failures.append({"path": str(path), "reason": "offline_asset_sha256_mismatch"})
    if offline_assets.get("network_resolution_allowed") is not False:
        failures.append({"path": "offline_model_assets", "reason": "network_resolution_not_forbidden"})
    return {
        "status": "ok" if not failures else "failed",
        "path": str(manifest_path),
        "source_revision": manifest.get("source_revision"),
        "source_worktree_dirty": manifest.get("source_worktree_dirty"),
        "offline_model_assets": offline_assets,
        "wilor_source_revision": manifest.get("wilor_source_revision"),
        "declared_file_count": manifest.get("file_count"),
        "failures": failures,
    }


def prompt_isolation(bundle: Path) -> dict[str, Any]:
    paths = [
        bundle / "runtime" / "v19_runtime_spec.md",
        bundle / "configs" / "v19_agent_system_prompt.md",
        bundle / ".pi" / "prompts" / "v19-run.md",
        bundle / "runtime" / "hot3d_dual_backend_runtime_spec.md",
        bundle / "configs" / "hot3d_dual_backend_agent_system_prompt.md",
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
    checks["da3_contract"] = da3_contract_check(args)
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
    dino_repo = args.torch_home / "hub/facebookresearch_dinov2_main"
    dino_hubconf = dino_repo / "hubconf.py"
    dino_hubconf_hash = sha256_file(dino_hubconf) if dino_hubconf.is_file() else None
    checks["trellis_dinov2_offline_source"] = {
        "path": str(dino_repo),
        "hubconf": str(dino_hubconf),
        "expected_hubconf_sha256": EXPECTED_DINOV2_HUBCONF_SHA256,
        "hubconf_sha256": dino_hubconf_hash,
        "status": "ok" if dino_hubconf_hash == EXPECTED_DINOV2_HUBCONF_SHA256 else "missing_incomplete_or_hash_mismatch",
        "network_resolution_allowed": False,
    }
    if args.sam3d_python is not None:
        checks["sam3d_contract"] = sam3d_contract_check(args)

    trellis_offline_code = (
        "import importlib.util; from pathlib import Path; "
        f"p=Path({str(bundle / 'scripts/remote_run_trellis_shape_v3.py')!r}); "
        "s=importlib.util.spec_from_file_location('trellis_runner_preflight',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"state=m.install_offline_dinov2_hub(Path({str(dino_repo)!r}), Path({str(dino)!r})); "
        "assert state['network_resolution_allowed'] is False; "
        "assert state['required_entry']=='dinov2_vitl14_reg'; "
        f"assert state['checkpoint_sha256']=={EXPECTED_DINOV2_SHA256!r}; "
        f"assert state['hubconf_sha256']=={EXPECTED_DINOV2_HUBCONF_SHA256!r}; "
        "print('TRELLIS_DINOV2_OFFLINE_SOURCE_OK',state['hubconf_sha256'],state['checkpoint_sha256'])"
    )
    import_commands = {
        "main": [str(args.main_python), "-c", "import cv2,open3d,smplx,torch,transformers,trimesh; from PIL import Image; print(torch.__version__, cv2.__version__, open3d.__version__, transformers.__version__)"],
        "unidepth": [str(args.unidepth_python), "-c", f"import sys,torch,numpy,cv2; sys.path.insert(0,{str(args.unidepth_repo)!r}); import unidepth; print(torch.__version__,numpy.__version__,cv2.__version__)"],
        "da3": [str(args.da3_python), "-c", f"import sys,torch,numpy,cv2; sys.path.insert(0,{str(args.da3_repo / 'src')!r}); from depth_anything_3.api import DepthAnything3; print(torch.__version__,numpy.__version__,cv2.__version__,DepthAnything3.__name__)"],
        "hawor": [str(args.hawor_python), "-c", f"import sys; sys.path.insert(0,{str(args.hawor_repo)!r}); import torch; import cv2,droid_backends,lietorch,mmcv,pytorch3d,smplx; print(torch.__version__,cv2.__version__)"],
        "trellis": [str(args.trellis_python), "-c", "import kaolin,spconv,torch,transformers,trimesh,xformers; print(torch.__version__,transformers.__version__,kaolin.__version__,spconv.__version__)"],
        "trellis_dinov2_offline_source": [str(args.trellis_python), "-c", trellis_offline_code],
    }
    if args.sam3d_python is not None:
        import_commands["sam3d"] = sam3d_import_command(args)
    import_results = {name: command_result(command, bundle) for name, command in import_commands.items()}
    checks["interpreter_imports"] = {
        "status": "ok" if all(row["returncode"] == 0 for row in import_results.values()) else "failed",
        "results": import_results,
    }

    bundle_manifest = json.loads((bundle / "RUNTIME_BUNDLE_MANIFEST.json").read_text())
    help_results = []
    cli_entries = [("scripts", name) for name in bundle_manifest.get("scripts", [])]
    cli_entries.extend(
        ("experiment", str(relative))
        for relative in bundle_manifest.get("suite_experiment_files", [])
    )
    for kind, name in cli_entries:
        path = bundle / "scripts" / name if kind == "scripts" else bundle / name
        if path.suffix == ".sh":
            result = command_result(["bash", "-n", str(path)], bundle)
        else:
            python = args.main_python
            if name.startswith("run_unidepth_"):
                python = args.unidepth_python
            elif name.startswith("run_da3_"):
                python = args.da3_python
            elif name == "export_hawor_world.py":
                python = args.hawor_python
            elif name == "remote_run_trellis_shape_v3.py":
                python = args.trellis_python
            result = command_result([str(python), str(path), "--help"], bundle)
        result["script"] = name
        result["kind"] = kind
        help_results.append(result)
    checks["script_cli_contracts"] = {
        "status": "ok" if all(row["returncode"] == 0 for row in help_results) else "failed",
        "failures": [row for row in help_results if row["returncode"] != 0],
        "checked": len(help_results),
    }

    self_test_results = []
    for relative in bundle_manifest.get("bundle_self_tests", []):
        path = bundle / str(relative)
        result = command_result([str(args.main_python), str(path)], bundle, timeout=600)
        result["self_test"] = str(relative)
        self_test_results.append(result)
    checks["bundle_self_tests"] = {
        "status": "ok" if self_test_results and all(row["returncode"] == 0 for row in self_test_results) else "failed",
        "results": self_test_results,
        "checked": len(self_test_results),
    }

    # Every preflight command must leave the immutable bundle byte-identical and
    # free of undeclared caches/artifacts.
    checks["bundle_integrity_after_checks"] = verify_bundle_manifest(bundle)

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
    parser.add_argument("--da3-python", type=Path, required=True)
    parser.add_argument("--da3-repo", type=Path, required=True)
    parser.add_argument("--da3-model", type=Path, required=True)
    parser.add_argument("--hawor-python", type=Path, required=True)
    parser.add_argument("--hawor-repo", type=Path, required=True)
    parser.add_argument("--hawor-asset-root", type=Path, required=True)
    parser.add_argument("--trellis-python", type=Path, required=True)
    parser.add_argument("--trellis-model", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--owlv2-model", type=Path, required=True)
    parser.add_argument("--sam3d-python", type=Path, default=None)
    parser.add_argument("--sam3d-repo", type=Path, default=None)
    parser.add_argument("--sam3d-config", type=Path, default=None)
    parser.add_argument("--sam3d-activation", type=Path, default=None)
    parser.add_argument("--sam3d-moge-checkpoint", type=Path, default=None)
    parser.add_argument("--expected-input-sha256", default=EXPECTED_INPUT_SHA256, help="Expected input hash, or 'none'/'skip' to disable the hash check")
    parser.add_argument("--expected-width", type=int, default=1408)
    parser.add_argument("--expected-height", type=int, default=1408)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--expected-frame-count", type=int, default=150)
    parser.add_argument("--allow-input-sidecar", action="append", default=[], help="Additional prediction-side sensor/provenance filename allowed next to input.mp4")
    args = parser.parse_args()
    sam3d_fields = (
        "sam3d_python",
        "sam3d_repo",
        "sam3d_config",
        "sam3d_activation",
        "sam3d_moge_checkpoint",
    )
    supplied = [getattr(args, name) is not None for name in sam3d_fields]
    if any(supplied) and not all(supplied):
        missing = ["--" + name.replace("_", "-") for name, present in zip(sam3d_fields, supplied) if not present]
        parser.error(f"partial SAM3D preflight contract; missing: {', '.join(missing)}")
    return args


if __name__ == "__main__":
    run(parse_args())
