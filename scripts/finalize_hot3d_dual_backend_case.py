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
    expected_p18b_payload_value_sha256: str,
) -> dict[str, Any]:
    manifest = load_json(render_manifest_path)
    branch_id = BACKEND_BRANCHES[backend]
    if manifest.get("status") != "ok" or (manifest.get("branch") or {}).get("branch_id") != branch_id:
        raise RuntimeError(f"{backend} render manifest does not bind expected branch {branch_id}: {render_manifest_path}")
    if int(manifest.get("frame_count", -1)) != int(expected_frames):
        raise RuntimeError(f"{backend} manifest frame count mismatch")
    temporal = (
        manifest.get("shared_p18b_temporal_surface")
        if isinstance(manifest.get("shared_p18b_temporal_surface"), dict)
        else {}
    )
    if temporal.get("value_sha256") != expected_p18b_payload_value_sha256:
        raise RuntimeError(
            f"{backend} render did not consume the published shared P18b payload: "
            f"{temporal.get('value_sha256')} != {expected_p18b_payload_value_sha256}"
        )
    if int(temporal.get("row_count", 0)) != int(expected_frames) * 2:
        raise RuntimeError(
            f"{backend} rendered P18b row count {temporal.get('row_count')} != {expected_frames * 2}"
        )
    if int(temporal.get("surface_point_count", 0)) > 0:
        if int(temporal.get("rendered_frame_count_with_input_points", 0)) <= 0:
            raise RuntimeError(f"{backend} P18b samples were not connected to rendered frames")
        for view_key in (
            "rendered_overlay_point_count",
            "rendered_world_point_count",
            "rendered_side_world_point_count",
        ):
            if int(temporal.get(view_key, 0)) <= 0:
                raise RuntimeError(
                    f"{backend} P18b samples were not visible in {view_key}"
                )

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
            "Full-timeline visual annotation branch. Backend-generated geometry is render-only; the shared observed metric "
            "surface supplies the common pose reference, while an independently reconstructed backend-neutral proxy may supply signed physics."
        ),
        "videos": published_videos,
        "geometry": published_geometry,
        "state": {**file_record(state_destination), "publish_mode": state_mode, "source": str(render_state)},
        "render_manifest": {**file_record(manifest_destination), "publish_mode": manifest_mode, "source": str(render_manifest_path)},
        "shared_p18b_temporal_surface": temporal,
        "shared_signed_geometry_ready": bool(temporal.get("signed_geometry_ready")),
        "shared_signed_full_mano_accepted": bool(temporal.get("accepted_signed_full_mano")),
    }
    report_path = backend_root / "backend_result.json"
    report_path.write_text(json.dumps(backend_report, indent=2) + "\n", encoding="utf-8")
    return {**backend_report, "backend_result": file_record(report_path)}


def publish_shared_tail(
    shared_tail_path: Path,
    shared_tail: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    if shared_tail.get("status") != "ok_shared_p17_p18_p18b_tail":
        raise RuntimeError(f"shared P17/P18 tail is not ok: {shared_tail_path}")
    outputs = shared_tail.get("outputs") if isinstance(shared_tail.get("outputs"), dict) else {}
    inputs = shared_tail.get("inputs") if isinstance(shared_tail.get("inputs"), dict) else {}
    p17 = require_file(Path(str(outputs.get("factor_report") or "")), "shared P17 factor report")
    p18 = require_file(Path(str(outputs.get("p18_state") or "")), "shared raw P18 state")
    p18b = require_file(Path(str(outputs.get("p18b_state") or "")), "shared P18b state")
    judgment = require_file(
        Path(str(inputs.get("interaction_judgment") or "")),
        "shared P17 interaction judgment",
    )
    expected_records = {
        "factor_report_sha256": (p17, outputs.get("factor_report_sha256")),
        "p18_state_sha256": (p18, outputs.get("p18_state_sha256")),
        "p18b_state_sha256": (p18b, outputs.get("p18b_state_sha256")),
        "interaction_judgment_sha256": (
            judgment,
            inputs.get("interaction_judgment_sha256"),
        ),
    }
    for label, (path, expected) in expected_records.items():
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise RuntimeError(
                f"shared-tail {label} binding mismatch for {path}: {actual} != {expected}"
            )
    p18_payload = load_json(p18)
    p18b_payload = load_json(p18b)
    p18_parameters = (
        p18_payload.get("parameters")
        if isinstance(p18_payload.get("parameters"), dict)
        else {}
    )
    if p18_parameters.get("optimize_object_translation") is not False:
        raise RuntimeError("published shared P18 did not disable object translation")
    max_object_delta = 0.0
    for row in p18_payload.get("per_frame_states") or []:
        if not isinstance(row, dict):
            continue
        delta = row.get("optimized_object_translation_world_m")
        if not isinstance(delta, list) or len(delta) != 3:
            raise RuntimeError("published shared P18 has malformed object delta")
        max_object_delta = max(
            max_object_delta,
            sum(float(value) ** 2 for value in delta) ** 0.5,
        )
    if max_object_delta > 1.0e-10:
        raise RuntimeError(
            f"published shared P18 privately moved the object by {max_object_delta} m"
        )
    p18b_value_hash = hashlib.sha256(
        json.dumps(
            p18b_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if p18b_value_hash != outputs.get("p18b_state_value_sha256"):
        raise RuntimeError(
            "shared P18b payload value hash differs from the shared-tail report"
        )

    shared_root = output_root / "shared"
    state_root = shared_root / "state"
    reports_root = shared_root / "reports"
    source_root = shared_root / "source"
    destinations = {
        "p18b_temporal_mano_state": (p18b, state_root / "p18b_temporal_mano_state.json"),
        "raw_p18_mano_interval_state": (p18, state_root / "raw_p18_mano_interval_state.json"),
        "p17_visible_contact_ownership_factor": (p17, reports_root / "p17_visible_contact_ownership_factor.json"),
        "p17_interaction_judgment": (judgment, source_root / "p17_interaction_judgment.json"),
        "shared_tail_report": (shared_tail_path, reports_root / "shared_p17_p18_tail_report.json"),
    }
    signed_completion_value = inputs.get("signed_completion_report")
    if signed_completion_value:
        signed_completion = require_file(
            Path(str(signed_completion_value)), "shared signed completion report"
        )
        expected_signed_hash = inputs.get("signed_completion_report_sha256")
        if not expected_signed_hash or sha256_file(signed_completion) != expected_signed_hash:
            raise RuntimeError("shared signed completion report hash mismatch")
        signed_payload = load_json(signed_completion)
        signed_outputs = signed_payload.get("outputs") if isinstance(signed_payload.get("outputs"), dict) else {}
        signed_candidate_value = signed_outputs.get("signed_geometry_candidate_mesh")
        signed_collision_value = signed_outputs.get("collision_eligible_mesh_labeled")
        destinations["shared_signed_completion_report"] = (
            signed_completion,
            reports_root / "shared_signed_geometry_completion_report.json",
        )
        if signed_collision_value:
            destinations["shared_collision_eligible_surface"] = (
                require_file(Path(str(signed_collision_value)), "shared signed/unsigned collision surface"),
                shared_root / "geometry/shared_collision_eligible_surface.ply",
            )
        if signed_candidate_value:
            destinations["shared_signed_geometry_candidate"] = (
                require_file(Path(str(signed_candidate_value)), "shared signed geometry candidate"),
                shared_root / "geometry/shared_signed_geometry_candidate.ply",
            )
    full_archive_value = p18_payload.get("full_mano_vertices_world_archive")
    if full_archive_value:
        full_archive = require_file(
            Path(str(full_archive_value)), "shared P18 full MANO archive"
        )
        expected_archive_hash = p18_payload.get("full_mano_vertices_world_archive_sha256")
        if not expected_archive_hash or sha256_file(full_archive) != expected_archive_hash:
            raise RuntimeError("shared P18 full MANO archive hash mismatch")
        destinations["p18_full_mano_vertices_world_archive"] = (
            full_archive,
            state_root / "p18_full_mano_vertices_world.npz",
        )
    published: dict[str, Any] = {}
    for name, (source, destination) in destinations.items():
        mode = link_or_copy(source, destination)
        published[name] = {
            **file_record(destination),
            "publish_mode": mode,
            "source": str(source),
        }
    signed_geometry_ready = shared_tail.get("signed_geometry_ready") is True
    full_acceptance = p18b_payload.get("full_mano_acceptance") if isinstance(p18b_payload.get("full_mano_acceptance"), dict) else {}
    signed_full_mano_accepted = full_acceptance.get("accepted") is True
    if signed_full_mano_accepted and not signed_geometry_ready:
        raise RuntimeError("shared P18b accepted signed full MANO without signed geometry readiness")
    manifest = {
        "status": "ok",
        "claim_scope": (
            "Published shared pre-branch P17/P18/P18b provenance. P18 uses the D15 "
            "observed-only pose authority with zero private object translation. A backend-neutral shared signed proxy may activate "
            "signed nonpenetration; full P18 MANO is published only when its independent acceptance checks pass."
        ),
        "p18b_payload_value_sha256": p18b_value_hash,
        "p18_max_private_object_translation_delta_m": max_object_delta,
        "generated_geometry_consumed_by_shared_tail": False,
        "signed_geometry_ready": signed_geometry_ready,
        "signed_full_mano_accepted": signed_full_mano_accepted,
        "signed_full_mano_acceptance": full_acceptance,
        "artifacts": published,
    }
    manifest_path = shared_root / "shared_state_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest": file_record(manifest_path)}


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
    shared_tail_path = require_file(args.shared_tail_report, "shared P17/P18/P18b tail report")
    controlled = load_json(controlled_path)
    dual_report = load_json(dual_path)
    adapter = load_json(adapter_path)
    shared_tail = load_json(shared_tail_path)
    if controlled.get("status") != "ok" or dual_report.get("status") != "ok" or adapter.get("status") != "ok":
        raise RuntimeError("controlled/dual/state-adapter reports must all be ok")

    output_root = prepare_output(args.output_dir, bool(args.replace))
    shared_result = publish_shared_tail(shared_tail_path, shared_tail, output_root)
    expected_p18b_hash = str(shared_result["p18b_payload_value_sha256"])
    backends = {
        "sam3d": publish_backend(
            "sam3d", sam_manifest, controlled, dual_report, adapter, output_root,
            int(args.expected_frame_count), float(args.expected_fps), expected_p18b_hash,
        ),
        "trellis": publish_backend(
            "trellis", trellis_manifest, controlled, dual_report, adapter, output_root,
            int(args.expected_frame_count), float(args.expected_fps), expected_p18b_hash,
        ),
    }

    shared = adapter.get("source_validation") if isinstance(adapter.get("source_validation"), dict) else {}
    temporal_readiness = (
        shared.get("temporal_readiness")
        if isinstance(shared.get("temporal_readiness"), dict)
        else {}
    )
    rotation_step_gate = (
        temporal_readiness.get("rotation_step_gate")
        if isinstance(temporal_readiness.get("rotation_step_gate"), dict)
        else {}
    )
    conditional_rotation_tail = {
        "applied": rotation_step_gate.get("conditional_tier_applied") is True,
        "acceptance_mode": rotation_step_gate.get("acceptance_mode"),
        "strict_max_rotation_step_deg": rotation_step_gate.get("strict_max_rotation_step_deg"),
        "conditional_max_rotation_step_deg": rotation_step_gate.get("conditional_max_rotation_step_deg"),
        "strict_exceedance_count": rotation_step_gate.get("strict_exceedance_count"),
        "transitions": rotation_step_gate.get("conditional_transitions") or [],
        "trajectory_values_modified_or_clipped": rotation_step_gate.get(
            "trajectory_values_modified_or_clipped"
        ),
        "generated_geometry_pose_evidence_consumed": rotation_step_gate.get(
            "generated_geometry_pose_evidence_consumed"
        ),
        "uncertainty": temporal_readiness.get("conditional_temporal_uncertainty"),
    }
    if conditional_rotation_tail["applied"]:
        if conditional_rotation_tail["acceptance_mode"] != "conditional_sparse_underobservable_rotation_tail":
            raise RuntimeError("malformed conditional rotation-tail acceptance mode")
        if conditional_rotation_tail["trajectory_values_modified_or_clipped"] is not False:
            raise RuntimeError("conditional rotation-tail trajectory was modified or clipped")
        if conditional_rotation_tail["generated_geometry_pose_evidence_consumed"] is not False:
            raise RuntimeError("conditional rotation-tail consumed generated pose evidence")
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
            "shared_prebranch_p17_p18_p18b_state": True,
            "shared_p18_object_translation_optimized": False,
            "shared_signed_geometry_ready": bool(shared_result.get("signed_geometry_ready")),
            "shared_signed_full_mano_accepted": bool(shared_result.get("signed_full_mano_accepted")),
            "backend_variable": "single-image generated render geometry prior and its integration",
            "generated_faces_collision_eligible": False,
            "released_reference_labels_consumed_by_prediction": False,
        },
        "shared_state": shared,
        "trajectory_uncertainty": {
            "annotation_readiness_mode": shared.get("annotation_readiness_mode"),
            "rotation_step_acceptance_mode": shared.get("rotation_step_acceptance_mode"),
            "conditional_rotation_tail": conditional_rotation_tail,
            "estimated_rotation_is_not_ground_truth_angular_velocity": True,
        },
        "shared_p17_p18_p18b": shared_result,
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
    parser.add_argument("--shared-tail-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frame-count", type=int, default=150)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
