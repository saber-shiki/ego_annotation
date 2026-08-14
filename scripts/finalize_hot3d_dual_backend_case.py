#!/usr/bin/env python3
"""Validate and publish one HOT3D dual-backend annotation case.

This is a prediction-only finalizer.  It verifies complete-duration render videos,
materializes stable backend result names, records geometry/state provenance, and
writes ``SUITE_DONE.json`` only after both branches pass their output contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2

VIDEO_OUTPUTS = {
    "camera_overlay": "overlay_video",
    "world_view": "world_video",
    "side_world_view": "side_world_video",
    "side_by_side": "side_by_side_video",
}
BACKEND_BRANCHES = {
    "sam3d": "sam3d_owned_dual_mesh",
    "trellis": "trellis_frozen_legacy_cut",
}
CONTROLLED_CANDIDATES = {
    "sam3d": "sam3d_new_object_owned_mask",
    "trellis": "trellis_frozen",
}


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


def file_record(path: Path) -> dict[str, Any]:
    path = require_file(path, "published artifact")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def video_record(path: Path, expected_frames: int, expected_fps: float) -> dict[str, Any]:
    path = require_file(path, "render video")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open render video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count != int(expected_frames):
        raise RuntimeError(f"video frame count mismatch for {path}: {frame_count} != {expected_frames}")
    if abs(fps - float(expected_fps)) > 0.05:
        raise RuntimeError(f"video FPS mismatch for {path}: {fps} != {expected_fps}")
    return {
        **file_record(path),
        "frame_count": frame_count,
        "fps": fps,
        "duration_s": float(frame_count / fps),
        "width": width,
        "height": height,
    }


def link_or_copy(source: Path, destination: Path) -> str:
    source = require_file(source, "source artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
        mode = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        mode = "copy"
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"publish hash mismatch: {source} -> {destination}")
    return mode


def candidate_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [
        row
        for row in report.get("candidates", [])
        if isinstance(row, dict) and str(row.get("name")) == name
    ]
    if len(rows) != 1:
        raise RuntimeError(f"controlled candidate {name!r} appears {len(rows)} times")
    return rows[0]


def branch_by_id(report: dict[str, Any], branch_id: str) -> dict[str, Any]:
    rows = [
        row
        for row in report.get("branches", [])
        if isinstance(row, dict) and str(row.get("branch_id")) == branch_id
    ]
    if len(rows) != 1:
        raise RuntimeError(f"state branch {branch_id!r} appears {len(rows)} times")
    return rows[0]


def prepare_output(path: Path, replace: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not replace:
            raise RuntimeError(f"refusing to overwrite non-empty final result: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def publish_backend(
    backend: str,
    render_manifest_path: Path,
    controlled: dict[str, Any],
    dual_report: dict[str, Any],
    adapter: dict[str, Any],
    output_root: Path,
    expected_frames: int,
    expected_fps: float,
) -> dict[str, Any]:
    manifest = load_json(render_manifest_path)
    branch_id = BACKEND_BRANCHES[backend]
    if manifest.get("status") != "ok" or (manifest.get("branch") or {}).get("branch_id") != branch_id:
        raise RuntimeError(f"{backend} render manifest does not bind expected branch {branch_id}: {render_manifest_path}")
    if int(manifest.get("frame_count", -1)) != int(expected_frames):
        raise RuntimeError(f"{backend} manifest frame count mismatch")

    backend_root = output_root / backend
    videos_root = backend_root / "videos"
    geometry_root = backend_root / "geometry"
    state_root = backend_root / "state"
    reports_root = backend_root / "reports"
    for directory in (videos_root, geometry_root, state_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)

    output_values = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    published_videos: dict[str, Any] = {}
    for stable_name, manifest_key in VIDEO_OUTPUTS.items():
        source = require_file(Path(str(output_values.get(manifest_key, ""))), f"{backend} {manifest_key}")
        destination = videos_root / f"{stable_name}.mp4"
        mode = link_or_copy(source, destination)
        published_videos[stable_name] = {**video_record(destination, expected_frames, expected_fps), "publish_mode": mode, "source": str(source)}

    candidate = candidate_by_name(controlled, CONTROLLED_CANDIDATES[backend])
    raw_mesh = require_file(Path(str((candidate.get("raw_mesh") or {}).get("path", ""))), f"{backend} raw mesh")
    neutral_outputs = candidate.get("source_neutral_outputs") if isinstance(candidate.get("source_neutral_outputs"), dict) else {}
    if backend == "sam3d":
        aligned_mesh = require_file(Path(str((dual_report.get("outputs") or {}).get("aligned_raw_render_prior", ""))), "SAM3D aligned mesh")
        geometry_sources = {
            "raw_generated_prior.ply": raw_mesh,
            "aligned_generated_render_prior.ply": aligned_mesh,
            "observed_metric_surface.ply": require_file(Path(str((dual_report.get("outputs") or {}).get("observed_metric_surface", ""))), "observed metric surface"),
            "dual_mesh_scene.glb": require_file(Path(str((dual_report.get("outputs") or {}).get("dual_mesh_scene_glb", ""))), "SAM3D dual mesh scene"),
        }
    else:
        geometry_sources = {
            "raw_generated_prior.ply": raw_mesh,
            "completed_render_hypothesis.ply": require_file(Path(str(neutral_outputs.get("pose_hypothesis_mesh", ""))), "TRELLIS completed hypothesis"),
            "observed_metric_surface.ply": require_file(Path(str(neutral_outputs.get("collision_eligible_mesh", ""))), "TRELLIS observed metric surface"),
        }
    published_geometry: dict[str, Any] = {}
    for name, source in geometry_sources.items():
        destination = geometry_root / name
        mode = link_or_copy(source, destination)
        published_geometry[name] = {**file_record(destination), "publish_mode": mode, "source": str(source)}

    state_branch = branch_by_id(adapter, branch_id)
    render_state = require_file(Path(str(state_branch.get("state_path", ""))), f"{backend} render state")
    state_destination = state_root / "render_state.json"
    state_mode = link_or_copy(render_state, state_destination)
    manifest_destination = reports_root / "render_manifest.json"
    manifest_mode = link_or_copy(render_manifest_path, manifest_destination)

    backend_report = {
        "status": "ok",
        "backend": backend,
        "branch_id": branch_id,
        "source_model": (manifest.get("branch") or {}).get("source_model"),
        "integration": (manifest.get("branch") or {}).get("integration"),
        "claim_scope": (
            "Full-timeline visual annotation branch. Generated geometry is render-only; the shared observed metric "
            "surface supplies the common pose reference and unsigned physical surface."
        ),
        "videos": published_videos,
        "geometry": published_geometry,
        "state": {**file_record(state_destination), "publish_mode": state_mode, "source": str(render_state)},
        "render_manifest": {**file_record(manifest_destination), "publish_mode": manifest_mode, "source": str(render_manifest_path)},
    }
    report_path = backend_root / "backend_result.json"
    report_path.write_text(json.dumps(backend_report, indent=2) + "\n", encoding="utf-8")
    return {**backend_report, "backend_result": file_record(report_path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise RuntimeError(f"missing run root: {run_root}")
    sam_manifest = require_file(args.sam3d_render_manifest, "SAM3D render manifest")
    trellis_manifest = require_file(args.trellis_render_manifest, "TRELLIS render manifest")
    controlled_path = require_file(args.controlled_report, "controlled P13 report")
    dual_path = require_file(args.dual_report, "SAM3D dual-mesh report")
    adapter_path = require_file(args.state_adapter_report, "layered state adapter report")
    controlled = load_json(controlled_path)
    dual_report = load_json(dual_path)
    adapter = load_json(adapter_path)
    if controlled.get("status") != "ok" or dual_report.get("status") != "ok" or adapter.get("status") != "ok":
        raise RuntimeError("controlled/dual/state-adapter reports must all be ok")

    output_root = prepare_output(args.output_dir, bool(args.replace))
    backends = {
        "sam3d": publish_backend(
            "sam3d", sam_manifest, controlled, dual_report, adapter, output_root,
            int(args.expected_frame_count), float(args.expected_fps),
        ),
        "trellis": publish_backend(
            "trellis", trellis_manifest, controlled, dual_report, adapter, output_root,
            int(args.expected_frame_count), float(args.expected_fps),
        ),
    }

    shared = adapter.get("source_validation") if isinstance(adapter.get("source_validation"), dict) else {}
    report = {
        "schema": "hot3d_sam3d_trellis_dual_backend_case_result_v1",
        "status": "complete",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "run_root": str(run_root),
        "created_unix_s": time.time(),
        "elapsed_s": time.time() - started,
        "fairness_contract": {
            "shared_prediction_evidence": True,
            "shared_anchor": True,
            "shared_object_owned_mask": True,
            "shared_observed_metric_surface": True,
            "shared_observed_only_pose_trajectory": True,
            "shared_camera_and_metric_mano_state": True,
            "backend_variable": "single-image generated render geometry prior and its integration",
            "generated_faces_collision_eligible": False,
            "released_reference_labels_consumed_by_prediction": False,
        },
        "shared_state": shared,
        "source_reports": {
            "controlled_p13": file_record(controlled_path),
            "sam3d_dual_mesh": file_record(dual_path),
            "layered_state_adapter": file_record(adapter_path),
        },
        "backends": backends,
        "final_result_root": str(output_root),
    }
    report_path = output_root / "case_result_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    done = {
        "status": "complete",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "result_manifest": str(report_path),
        "result_manifest_sha256": sha256_file(report_path),
        "completed_unix_s": time.time(),
    }
    done_path = run_root / "SUITE_DONE.json"
    temp_path = done_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, done_path)
    print(json.dumps({**done, "backends": list(backends)}, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sam3d-render-manifest", type=Path, required=True)
    parser.add_argument("--trellis-render-manifest", type=Path, required=True)
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--dual-report", type=Path, required=True)
    parser.add_argument("--state-adapter-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frame-count", type=int, default=150)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
