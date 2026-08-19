#!/usr/bin/env python3
"""Derive the isolated HOT3D controlled dual-backend bundle from a V19 bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_FILES = [
    "build_p11_dual_geometry_inputs.py",
    "run_p12_parallel_geometry_priors.py",
    "run_p13_controlled_geometry_prior_ab.py",
    "build_p13_sam3d_native_metric_bridge.py",
    "render_p12_raw_sam_mask_ab.py",
    "build_p13_dual_mesh_geometry_prior.py",
    "build_p14_p15_layered_render_states.py",
    "render_p14_p15_layered_state.py",
]
SELF_TEST_FILES = [
    ("experiments/sam3d_p11_p12_branch/self_test.py", "experiments/sam3d_p11_p12_branch/self_test.py"),
    ("experiments/v19_metric_camera_contract/self_test.py", "experiments/v19_metric_camera_contract/self_test.py"),
    ("experiments/hot3d_shared_p18_reintegration/self_test.py", "experiments/hot3d_shared_p18_reintegration/self_test.py"),
    ("experiments/hot3d_shared_signed_geometry/self_test.py", "experiments/hot3d_shared_signed_geometry/self_test.py"),
]
SCRIPT_FILES = [
    "run_unidepth_metric_source_v3.py",
    "run_unidepth_full_frame_v3.py",
    "adapt_v19_depth_to_camera_contract.py",
    "build_v19_base_annotations.py",
    "build_v19_visible_geometry_from_sam2_depth.py",
    "build_v18_compact_rigid_evidence_bundle.py",
    "build_v18_compact_rigid_trellis_completion.py",
    "fit_v18_compact_rigid_object_pose.py",
    "solve_v19_rigid_object_pose_graph.py",
    "remote_run_trellis_shape_v3.py",
    "remote_run_sam3d_objects_mesh_v7.py",
    "build_v19_observed_only_completion_reference.py",
    "build_hot3d_shared_signed_geometry.py",
    "export_hawor_world.py",
    "remote_run_hawor_export.sh",
    "build_v19_visible_contact_ownership_factor.py",
    "solve_v18_joint_mano_interval_trajectory.py",
    "build_v19_mano_surface_hypothesis_state.py",
    "build_v19_rigid_render_state.py",
    "run_hot3d_shared_p17_p18_tail.py",
    "run_hot3d_dual_backend_d18_renders.py",
    "finalize_hot3d_dual_backend_case.py",
    "monitor_hot3d_dual_backend_suite.py",
    "preflight_local_v19_runtime.py",
]
EXTRA_FILES = [
    ("runtime/hot3d_dual_backend_runtime_spec.md", "runtime/hot3d_dual_backend_runtime_spec.md"),
    ("configs/hot3d_dual_backend_agent_system_prompt.md", "configs/hot3d_dual_backend_agent_system_prompt.md"),
    ("docs/hot3d_shared_signed_geometry_milk_design_zh.md", "docs/hot3d_shared_signed_geometry_milk_design_zh.md"),
]
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def verify_manifest(bundle: Path) -> dict[str, Any]:
    manifest_path = require_file(bundle / "RUNTIME_BUNDLE_MANIFEST.json", "base bundle manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for row in payload.get("files", []):
        path = bundle / str(row.get("path"))
        if not path.is_file():
            failures.append({"path": row.get("path"), "reason": "missing"})
        elif sha256_file(path) != row.get("sha256"):
            failures.append({"path": row.get("path"), "reason": "sha256_mismatch"})
    if failures:
        raise RuntimeError(f"base bundle integrity failed: {failures[:10]}")
    return payload


def copy_required(source: Path, destination: Path) -> None:
    source = require_file(source, "suite source file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def make_tree_owner_writable(root: Path) -> None:
    """Make only the copied destination writable, preserving an immutable base."""
    paths = [root, *sorted(root.rglob("*"))]
    for path in paths:
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
        elif path.is_file() and not path.is_symlink():
            path.chmod(mode | stat.S_IWUSR)


def rewrite_bundle_root(bundle: Path, old: str, new: str) -> int:
    changed = 0
    for path in sorted(item for item in bundle.rglob("*") if item.is_file() and not item.is_symlink()):
        if path.name == "RUNTIME_BUNDLE_MANIFEST.json" or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8")
            changed += 1
    return changed


def file_rows(bundle: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in bundle.rglob("*") if item.is_file() and not item.is_symlink()):
        if path.name == "RUNTIME_BUNDLE_MANIFEST.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(bundle)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base_bundle.expanduser().resolve()
    source = args.source_root.expanduser().resolve()
    output = args.bundle_root.expanduser().resolve()
    if output == base or output == source or base in output.parents or source in output.parents:
        raise RuntimeError("bundle root must be a distinct path outside the base bundle and source worktree")
    revision = git_value(source, "rev-parse", "HEAD")
    dirty = git_value(source, "status", "--short") or ""
    if not revision:
        raise RuntimeError(f"source root is not a readable Git revision: {source}")
    if dirty:
        raise RuntimeError(
            "immutable HOT3D runtime bundles require a committed clean source worktree; "
            f"dirty entries: {dirty.splitlines()}"
        )
    base_manifest = verify_manifest(base)
    if output.exists():
        if not args.replace:
            raise RuntimeError(f"bundle root exists: {output}")
        shutil.rmtree(output)
    shutil.copytree(base, output, symlinks=False)
    # Immutable released bundles may be used as a base. copytree preserves their
    # read-only modes, so grant owner-write only to the new destination before
    # overlays and root rewriting; the caller locks the completed bundle again.
    make_tree_owner_writable(output)
    for cache in output.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache)

    experiment_source = source / "experiments/sam3d_p11_p12_branch"
    experiment_destination = output / "experiments/sam3d_p11_p12_branch"
    for name in EXPERIMENT_FILES:
        copy_required(experiment_source / name, experiment_destination / name)
    for name in SCRIPT_FILES:
        copy_required(source / "scripts" / name, output / "scripts" / name)
    for source_relative, destination_relative in [*SELF_TEST_FILES, *EXTRA_FILES]:
        copy_required(source / source_relative, output / destination_relative)

    source_bound_paths = {
        *[f"experiments/sam3d_p11_p12_branch/{name}" for name in EXPERIMENT_FILES],
        *[f"scripts/{name}" for name in SCRIPT_FILES],
        *[destination_relative for _source_relative, destination_relative in SELF_TEST_FILES],
        *[destination_relative for _source_relative, destination_relative in EXTRA_FILES],
    }
    committed_failures = []
    for relative in sorted(source_bound_paths):
        source_relative = relative
        source_path = source / source_relative
        result = subprocess.run(
            ["git", "-C", str(source), "show", f"HEAD:{source_relative}"],
            capture_output=True,
        )
        if result.returncode != 0:
            committed_failures.append({"path": source_relative, "reason": "not_in_source_revision"})
        elif hashlib.sha256(result.stdout).hexdigest() != sha256_file(source_path):
            committed_failures.append({"path": source_relative, "reason": "working_tree_differs_from_source_revision"})
    if committed_failures:
        shutil.rmtree(output)
        raise RuntimeError(f"bundle source files are not byte-bound to revision {revision}: {committed_failures}")

    old_root = str(base)
    new_root = str(output)
    rewritten_files = rewrite_bundle_root(output, old_root, new_root)
    rows = file_rows(output)
    scripts = sorted(set(base_manifest.get("scripts", [])).union(SCRIPT_FILES))
    manifest = {
        **base_manifest,
        "status": "hot3d_dual_backend_runtime_bundle_built",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(output),
        "base_bundle": str(base),
        "base_bundle_source_revision": base_manifest.get("source_revision"),
        "source_repo": str(source),
        "source_revision": revision,
        "source_worktree_dirty": False,
        "source_worktree_status": [],
        "suite_runtime_spec": "runtime/hot3d_dual_backend_runtime_spec.md",
        "suite_system_prompt": "configs/hot3d_dual_backend_agent_system_prompt.md",
        "suite_experiment_files": [f"experiments/sam3d_p11_p12_branch/{name}" for name in EXPERIMENT_FILES],
        "bundle_self_tests": [destination for _source, destination in SELF_TEST_FILES],
        "offline_model_assets": {
            "dinov2_source_hubconf": {
                "path": "/mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub/hub/facebookresearch_dinov2_main/hubconf.py",
                "sha256": "c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64",
            },
            "dinov2_checkpoint": {
                "path": "/mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
                "sha256": "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
            },
            "sam3d_moge_checkpoint": {
                "path": "/mnt/user-home/kupingxin/sam3d-objects/hf-cache/hub/models--Ruicheng--moge-vitl/blobs/da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f",
                "sha256": "da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f",
            },
            "network_resolution_allowed": False,
        },
        "scripts": scripts,
        "bundle_root_rewritten_text_file_count": rewritten_files,
        "file_count": len(rows),
        "files": rows,
    }
    manifest_path = output / "RUNTIME_BUNDLE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    compact = {
        "status": manifest["status"],
        "bundle_root": str(output),
        "source_revision": revision,
        "source_worktree_dirty": False,
        "file_count": len(rows),
        "scripts": len(scripts),
        "experiment_files": len(EXPERIMENT_FILES),
        "self_tests": len(SELF_TEST_FILES),
        "bundle_root_rewritten_text_file_count": rewritten_files,
    }
    print(json.dumps(compact, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
