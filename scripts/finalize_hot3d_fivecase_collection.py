#!/usr/bin/env python3
"""Validate and publish a read-only index over completed HOT3D dual-backend cases.

This is a post-prediction collection finalizer. It never writes inside a case run
root and never supplies evidence to prediction. It validates D19 manifests,
actually decodes every published video, byte-checks artifacts, verifies that the
two backends share pose/MANO/camera state, and then publishes validated collection
links plus atomically replaced JSON/Markdown reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

VIDEO_KEYS = ("camera_overlay", "world_view", "side_world_view", "side_by_side")
BACKENDS = ("sam3d", "trellis")
MAX_COMPLETED_POSE_FRACTION = 0.20
READINESS_NUMERIC_EPSILON = 1.0e-12
SHARED_STATE_KEYS = (
    "annotation_backbone",
    "object_pose_trajectory",
    "mano_constraint_state",
    "temporal_mano_state",
    "hidden_volume_validation",
    "projection_contract",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def load_json(path: Path, description: str = "JSON") -> dict[str, Any]:
    path = require_file(path, description)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object for {description}: {path}")
    return payload


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def file_record(path: Path) -> dict[str, Any]:
    path = require_file(path, "artifact")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def verify_declared_file(row: dict[str, Any], description: str) -> Path:
    path = require_file(Path(str(row.get("path") or "")), description)
    actual = sha256_file(path)
    if actual != row.get("sha256"):
        raise RuntimeError(f"{description} SHA256 mismatch: {actual} != {row.get('sha256')}: {path}")
    if row.get("bytes") is not None and int(row["bytes"]) != int(path.stat().st_size):
        raise RuntimeError(f"{description} byte count mismatch: {path}")
    return path


def decoded_video_record(path: Path, expected_frames: int, expected_fps: float) -> dict[str, Any]:
    path = require_file(path, "published video")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open published video: {path}")
    metadata_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    first_shape: list[int] | None = None
    last_shape: list[int] | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"malformed decoded frame {decoded}: {path}")
            shape = [int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])]
            if first_shape is None:
                first_shape = shape
            last_shape = shape
            decoded += 1
    finally:
        capture.release()
    if metadata_frames != expected_frames or decoded != expected_frames:
        raise RuntimeError(
            f"video frame count mismatch metadata={metadata_frames} decoded={decoded} "
            f"expected={expected_frames}: {path}"
        )
    if abs(fps - expected_fps) > 1.0e-3:
        raise RuntimeError(f"video FPS mismatch {fps} != {expected_fps}: {path}")
    return {
        **file_record(path),
        "metadata_frame_count": metadata_frames,
        "decoded_frame_count": decoded,
        "fps": fps,
        "duration_s": float(decoded / fps),
        "width": width,
        "height": height,
        "first_decoded_shape_hwc": first_shape,
        "last_decoded_shape_hwc": last_shape,
    }


def numeric_summary(values: list[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if value is not None)
    if not finite:
        return {"count": 0, "median": None, "p10": None, "p90": None, "min": None, "max": None}

    def percentile(q: float) -> float:
        if len(finite) == 1:
            return finite[0]
        index = q * (len(finite) - 1)
        lower = int(index)
        upper = min(lower + 1, len(finite) - 1)
        fraction = index - lower
        return float(finite[lower] * (1.0 - fraction) + finite[upper] * fraction)

    return {
        "count": len(finite),
        "median": float(statistics.median(finite)),
        "mean": float(statistics.fmean(finite)),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
        "min": finite[0],
        "max": finite[-1],
    }


def parse_case_binding(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case binding must be NAME=/absolute/run/root")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("case binding must have a nonempty name and path")
    return name, Path(path).expanduser().resolve()


def resolved_symlink_target(path: Path, *, strict: bool) -> Path:
    raw_target = Path(os.readlink(path))
    joined = raw_target if raw_target.is_absolute() else path.parent / raw_target
    return joined.resolve(strict=strict)


def validate_symlink(path: Path, target: Path) -> str:
    expected = target.expanduser().resolve(strict=True)
    if not path.is_symlink():
        raise RuntimeError(f"collection link is not a symlink: {path}")
    if not path.exists():
        raise RuntimeError(f"collection link is dangling: {path} -> {os.readlink(path)!r}")
    actual = resolved_symlink_target(path, strict=True)
    if actual != expected:
        raise RuntimeError(f"collection link resolves to {actual}, expected {expected}: {path}")
    return os.readlink(path)


def replace_symlink(
    path: Path,
    target: Path,
    *,
    allow_empty_link_stub_repair: bool = False,
) -> dict[str, Any]:
    """Publish and validate a portable relative symlink.

    Symlink rename is deliberately not used. CIFS ``nounix`` mounts can convert a
    symlink renamed with ``os.replace`` into a zero-byte regular file, and can
    strip the leading slash from absolute symlink targets. A direct relative
    symlink is capability-probed first, then published and resolved back to the
    exact target. Existing non-symlinks fail closed; repairing a zero-byte stub
    left by the affected older finalizer requires an explicit CLI opt-in.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target.expanduser().resolve(strict=True)
    if not target.is_dir():
        raise RuntimeError(f"collection link target is not a directory: {target}")
    relative_target = os.path.relpath(str(target), str(path.parent.resolve(strict=True)))

    probe = path.with_name(f".{path.name}.symlink-probe-{os.getpid()}")
    if os.path.lexists(probe):
        raise RuntimeError(f"refusing to replace existing symlink capability probe: {probe}")
    try:
        os.symlink(relative_target, probe, target_is_directory=True)
        validate_symlink(probe, target)
    finally:
        if os.path.lexists(probe):
            probe.unlink()

    old_link_text: str | None = None
    old_link_target: Path | None = None
    repaired_empty_stub = False
    if os.path.lexists(path):
        if path.is_symlink():
            old_link_text = os.readlink(path)
            old_link_target = resolved_symlink_target(path, strict=False)
        elif (
            allow_empty_link_stub_repair
            and path.is_file()
            and int(path.stat().st_size) == 0
        ):
            repaired_empty_stub = True
        else:
            raise RuntimeError(f"refusing to replace non-symlink collection path: {path}")
        path.unlink()

    try:
        os.symlink(relative_target, path, target_is_directory=True)
        published_text = validate_symlink(path, target)
    except Exception as error:
        if os.path.lexists(path):
            path.unlink()
        try:
            if old_link_target is not None:
                rollback_text = os.path.relpath(
                    str(old_link_target), str(path.parent.resolve(strict=True))
                )
                os.symlink(rollback_text, path, target_is_directory=True)
            elif repaired_empty_stub:
                path.touch(exist_ok=False)
        except Exception as rollback_error:
            raise RuntimeError(
                f"collection link publication and rollback both failed for {path}: "
                f"publication={error}; rollback={rollback_error}"
            ) from error
        raise

    return {
        "path": str(path),
        "target": str(target),
        "link_text": published_text,
        "target_is_relative": not Path(published_text).is_absolute(),
        "update_mode": "validated_direct_relative_symlink",
        "atomic_symlink_rename_used": False,
        "repaired_empty_link_stub": repaired_empty_stub,
        "validated": True,
        "previous_link_text": old_link_text,
    }


def candidate_geometry_rows(run_root: Path) -> tuple[int | None, dict[str, dict[str, Any]], Path]:
    report_path = require_file(
        run_root
        / "experiments/sam3d_trellis_controlled/P13_controlled/p13_controlled_geometry_prior_ab_report.json",
        "P13 controlled geometry comparison",
    )
    report = load_json(report_path)
    candidates = report.get("candidates") if isinstance(report.get("candidates"), list) else []
    rows: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        source_model = str(candidate.get("source_model") or "")
        backend = "sam3d" if source_model == "sam3d_objects" else "trellis" if source_model == "trellis" else None
        if backend is None:
            continue
        alignment = candidate.get("metric_alignment") if isinstance(candidate.get("metric_alignment"), dict) else {}
        observed_to_generated = (
            alignment.get("observed_to_generated_final")
            if isinstance(alignment.get("observed_to_generated_final"), dict)
            else {}
        )
        generated_to_observed = (
            alignment.get("generated_to_observed_final")
            if isinstance(alignment.get("generated_to_observed_final"), dict)
            else {}
        )
        semantics = candidate.get("face_semantics") if isinstance(candidate.get("face_semantics"), dict) else {}
        readiness = candidate.get("geometry_readiness") if isinstance(candidate.get("geometry_readiness"), dict) else {}
        if readiness.get("collision_eligible_hidden_face_count") != 0:
            raise RuntimeError(f"{backend} generated hidden faces became collision eligible: {report_path}")
        for metric in (observed_to_generated, generated_to_observed):
            if not isinstance(metric.get("median_m"), (int, float)) or not isinstance(metric.get("p95_m"), (int, float)):
                raise RuntimeError(f"{backend} lacks comparable P13 distance metrics: {report_path}")
        native = candidate.get("native_metric_bridge") if isinstance(candidate.get("native_metric_bridge"), dict) else {}
        projection = native.get("projection_validation") if isinstance(native.get("projection_validation"), dict) else {}
        front = (
            native.get("geometry_validation", {}).get("observed_front_surface_quality", {})
            if isinstance(native.get("geometry_validation"), dict)
            else {}
        )
        rows[backend] = {
            "candidate_name": candidate.get("name"),
            "source_model": source_model,
            "status": candidate.get("status"),
            "observed_to_generated_m": observed_to_generated,
            "generated_to_observed_m": generated_to_observed,
            "generated_hidden_faces": int(semantics.get("generated_hidden_faces_in_pose_hypothesis") or 0),
            "collision_eligible_hidden_faces": int(readiness.get("collision_eligible_hidden_face_count") or 0),
            "generated_faces_collision_eligible": readiness.get("generated_faces_collision_eligible", False),
            "generated_faces_contact_eligible": readiness.get("generated_faces_contact_eligible", False),
            "native_projection_iou": projection.get("convex_projection_iou"),
            "observed_front_acceptance_mode": front.get("acceptance_mode"),
        }
    if set(rows) != set(BACKENDS):
        raise RuntimeError(f"P13 report does not contain exactly SAM3D and TRELLIS: {report_path}")
    return report.get("selected_frame_idx"), rows, report_path


def shared_hand_state(run_root: Path, expected_frames: int) -> tuple[dict[str, Any], Path]:
    path = require_file(
        run_root
        / "experiments/sam3d_trellis_controlled/P16_unsigned_mano_object/v18_mano_object_constraint_state.json",
        "shared unsigned MANO/object state",
    )
    report = load_json(path)
    if report.get("object_pose_input_ready_for_constraint_measurement") is not True:
        raise RuntimeError(f"P16 does not consume a ready shared pose: {path}")
    if report.get("unsigned_collision_surface_measurement_available") is not True:
        raise RuntimeError(f"P16 lacks unsigned observed-surface measurements: {path}")
    if report.get("signed_geometry_query_eligible") is not False:
        raise RuntimeError(f"P16 unexpectedly declares signed geometry eligible: {path}")
    if report.get("signed_nonpenetration_factor_active") is not False:
        raise RuntimeError(f"P16 unexpectedly activates signed nonpenetration: {path}")
    readiness = (
        report.get("completion_geometry_readiness")
        if isinstance(report.get("completion_geometry_readiness"), dict)
        else {}
    )
    for key in ("generated_faces_collision_eligible", "generated_faces_contact_eligible", "signed_geometry_ready"):
        if readiness.get(key) is not False:
            raise RuntimeError(f"P16 lacks explicit {key}=false: {path}")
    rows = report.get("constraint_rows") if isinstance(report.get("constraint_rows"), list) else []
    by_side: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        side_rows = [row for row in rows if isinstance(row, dict) and row.get("hand_side") == side]
        if len(side_rows) != expected_frames:
            raise RuntimeError(f"P16 {side} row count {len(side_rows)} != {expected_frames}: {path}")
        if any(row.get("signed_geometry_query_eligible") is not False for row in side_rows):
            raise RuntimeError(f"P16 {side} row activates signed queries: {path}")
        minima = [
            float(row["nearest_surface_unsigned_m"]["min"])
            for row in side_rows
            if isinstance(row.get("nearest_surface_unsigned_m"), dict)
            and isinstance(row["nearest_surface_unsigned_m"].get("min"), (int, float))
        ]
        bands = [float(row["observed_band_m"]) for row in side_rows if isinstance(row.get("observed_band_m"), (int, float))]
        by_side[side] = {
            "measured_frames": len(side_rows),
            "frames_with_any_near_surface_sample": sum(int(row.get("near_surface_vertex_count") or 0) > 0 for row in side_rows),
            "near_surface_sample_observations": sum(int(row.get("near_surface_vertex_count") or 0) for row in side_rows),
            "sampled_hand_vertices_per_frame": sorted({int(row.get("hand_vertex_count") or 0) for row in side_rows}),
            "observed_near_surface_band_m": numeric_summary(bands),
            "per_frame_min_unsigned_distance_m": numeric_summary(minima),
        }
    return {
        "status": report.get("status"),
        "annotation_ready_for_signed_physical_correction": report.get("annotation_ready"),
        "unsigned_collision_surface_measurement_available": True,
        "completed_surface_mesh_watertight": report.get("completed_surface_mesh_watertight"),
        "sign_mesh_watertight": report.get("sign_mesh_watertight"),
        "signed_geometry_query_eligible": False,
        "signed_nonpenetration_factor_active": False,
        "signed_nonpenetration_inactive_reason": report.get("signed_nonpenetration_inactive_reason"),
        "generated_faces_collision_eligible": False,
        "generated_faces_contact_eligible": False,
        "sides": by_side,
        "interpretation": (
            "Unsigned proximity to the partial prediction-side observed surface only. Zero penetrating rows "
            "cannot be interpreted as no penetration because signed queries are inactive."
        ),
    }, path


def validate_case(name: str, run_root: Path, expected_frames: int, expected_fps: float) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    done_path = require_file(run_root / "SUITE_DONE.json", f"{name} D19 sentinel")
    done = load_json(done_path)
    if done.get("status") != "complete":
        raise RuntimeError(f"{name} D19 sentinel is not complete: {done_path}")
    manifest_path = require_file(run_root / "final_results/case_result_manifest.json", f"{name} D19 manifest")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete" or Path(str(manifest.get("run_root"))).resolve() != run_root:
        raise RuntimeError(f"{name} case result manifest is not a complete result for its run root")
    if Path(str(done.get("result_manifest"))).resolve() != manifest_path:
        raise RuntimeError(f"{name} SUITE_DONE does not bind the final manifest")
    if done.get("result_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError(f"{name} SUITE_DONE final manifest hash mismatch")

    fairness = manifest.get("fairness_contract") if isinstance(manifest.get("fairness_contract"), dict) else {}
    for key in (
        "shared_prediction_evidence",
        "shared_anchor",
        "shared_object_owned_mask",
        "shared_observed_metric_surface",
        "shared_observed_only_pose_trajectory",
        "shared_camera_and_metric_mano_state",
    ):
        if fairness.get(key) is not True:
            raise RuntimeError(f"{name} fairness contract lacks {key}=true")
    if fairness.get("generated_faces_collision_eligible") is not False:
        raise RuntimeError(f"{name} fairness contract promotes generated collision faces")

    shared = manifest.get("shared_state") if isinstance(manifest.get("shared_state"), dict) else {}
    temporal = shared.get("temporal_readiness") if isinstance(shared.get("temporal_readiness"), dict) else {}
    if temporal.get("ready") is not True or int(temporal.get("accepted_full_timeline_pose_count") or 0) != expected_frames:
        raise RuntimeError(f"{name} lacks a ready exact full-timeline shared pose")
    completed_pose_fraction = temporal.get("completed_pose_fraction")
    if not isinstance(completed_pose_fraction, (int, float)):
        raise RuntimeError(f"{name} lacks a numeric completed-pose fraction")
    completed_pose_fraction = float(completed_pose_fraction)
    completed_pose_fraction_passed = (
        completed_pose_fraction <= MAX_COMPLETED_POSE_FRACTION + READINESS_NUMERIC_EPSILON
    )
    if not completed_pose_fraction_passed:
        raise RuntimeError(
            f"{name} completed-pose fraction {completed_pose_fraction} exceeds the inclusive "
            f"maximum {MAX_COMPLETED_POSE_FRACTION}"
        )
    pose_path = require_file(Path(str(shared.get("pose_report") or "")), f"{name} shared pose report")
    if sha256_file(pose_path) != shared.get("pose_report_sha256"):
        raise RuntimeError(f"{name} shared pose report hash mismatch")
    pose_report = load_json(pose_path)
    if pose_report.get("annotation_ready") is not True:
        raise RuntimeError(f"{name} shared pose report is not annotation ready")
    if int(pose_report.get("nonpenetration_target_frame_count", -1)) != 0:
        raise RuntimeError(f"{name} pose trajectory consumed nonpenetration targets")
    eligibility = (
        pose_report.get("pose_observation_eligibility_policy")
        if isinstance(pose_report.get("pose_observation_eligibility_policy"), dict)
        else {}
    )
    if eligibility.get("include_ineligible_override") is not False:
        raise RuntimeError(f"{name} pose report does not fail closed on ineligible observations")
    geometry_contract = (
        pose_report.get("geometry_contract")
        if isinstance(pose_report.get("geometry_contract"), dict)
        else {}
    )
    pose_readiness = (
        geometry_contract.get("completion_geometry_readiness")
        if isinstance(geometry_contract.get("completion_geometry_readiness"), dict)
        else {}
    )
    if pose_readiness.get("pose_hypothesis_source") != "prediction_side_observed_metric_surface_only":
        raise RuntimeError(f"{name} pose body is not explicitly observed-metric-only")
    for key in (
        "generated_hidden_surface_included",
        "generated_faces_collision_eligible",
        "generated_faces_contact_eligible",
    ):
        if pose_readiness.get(key) is not False:
            raise RuntimeError(f"{name} pose geometry contract lacks explicit {key}=false")
    pose_rows = pose_report.get("pose_rows") if isinstance(pose_report.get("pose_rows"), list) else []
    if len(pose_rows) != expected_frames:
        raise RuntimeError(f"{name} pose report row count {len(pose_rows)} != {expected_frames}")
    generated_pose_rows = [
        int(row.get("frame_idx", -1))
        for row in pose_rows
        if isinstance(row, dict)
        and row.get("generated_geometry_pose_evidence_consumed") is True
    ]
    if generated_pose_rows:
        raise RuntimeError(f"{name} pose rows consumed generated geometry: {generated_pose_rows}")
    rotation_gate = (
        temporal.get("rotation_step_gate")
        if isinstance(temporal.get("rotation_step_gate"), dict)
        else {}
    )
    conditional_applied = rotation_gate.get("conditional_tier_applied") is True
    conditional_frames = [int(row["to_frame_idx"]) for row in rotation_gate.get("conditional_transitions") or []] if conditional_applied else []
    if conditional_applied:
        if rotation_gate.get("acceptance_mode") != "conditional_sparse_underobservable_rotation_tail":
            raise RuntimeError(f"{name} has malformed conditional rotation-tail mode")
        if rotation_gate.get("trajectory_values_modified_or_clipped") is not False:
            raise RuntimeError(f"{name} conditional trajectory was modified or clipped")
        if rotation_gate.get("generated_geometry_pose_evidence_consumed") is not False:
            raise RuntimeError(f"{name} conditional trajectory consumed generated geometry")

    backends_payload = manifest.get("backends") if isinstance(manifest.get("backends"), dict) else {}
    if set(backends_payload) != set(BACKENDS):
        raise RuntimeError(f"{name} does not publish exactly two required backends")
    state_payloads: dict[str, dict[str, Any]] = {}
    backend_rows: dict[str, Any] = {}
    videos: list[dict[str, Any]] = []
    observed_hashes = set()
    for backend in BACKENDS:
        row = backends_payload[backend]
        if row.get("status") != "ok":
            raise RuntimeError(f"{name}/{backend} backend is not ok")
        declared_videos = row.get("videos") if isinstance(row.get("videos"), dict) else {}
        if set(declared_videos) != set(VIDEO_KEYS):
            raise RuntimeError(f"{name}/{backend} does not publish exactly four videos")
        video_rows = {}
        for key in VIDEO_KEYS:
            declared_path = verify_declared_file(declared_videos[key], f"{name}/{backend}/{key}")
            actual = decoded_video_record(declared_path, expected_frames, expected_fps)
            video_rows[key] = actual
            videos.append({"case": name, "backend": backend, "kind": key, **actual})
        state_path = verify_declared_file(row["state"], f"{name}/{backend} render state")
        render_manifest_path = verify_declared_file(row["render_manifest"], f"{name}/{backend} render manifest")
        state_payloads[backend] = load_json(state_path)
        render_manifest = load_json(render_manifest_path)
        consumption = (
            render_manifest.get("shared_state_consumption")
            if isinstance(render_manifest.get("shared_state_consumption"), dict)
            else {}
        )
        for key in ("generated_faces_collision_eligible", "generated_faces_contact_eligible", "signed_geometry_ready"):
            if consumption.get(key) is not False:
                raise RuntimeError(f"{name}/{backend} render manifest lacks {key}=false")
        actual_warning_frames = [int(value) for value in render_manifest.get("conditional_rotation_tail_frames") or []]
        if actual_warning_frames != conditional_frames:
            raise RuntimeError(
                f"{name}/{backend} conditional warning frames {actual_warning_frames} != {conditional_frames}"
            )
        geometry = row.get("geometry") if isinstance(row.get("geometry"), dict) else {}
        observed = geometry.get("observed_metric_surface.ply")
        if not isinstance(observed, dict):
            raise RuntimeError(f"{name}/{backend} lacks published observed metric surface")
        verify_declared_file(observed, f"{name}/{backend} observed surface")
        observed_hashes.add(str(observed.get("sha256")))
        for geometry_name, geometry_row in geometry.items():
            verify_declared_file(geometry_row, f"{name}/{backend}/{geometry_name}")
        backend_rows[backend] = {
            "source_model": row.get("source_model"),
            "integration": row.get("integration"),
            "videos": video_rows,
            "state": file_record(state_path),
            "render_manifest": file_record(render_manifest_path),
            "observed_metric_surface_sha256": observed.get("sha256"),
        }
    if len(observed_hashes) != 1:
        raise RuntimeError(f"{name} backends do not share one observed metric surface")

    shared_state_hashes = {}
    for key in SHARED_STATE_KEYS:
        left = state_payloads["sam3d"].get(key)
        right = state_payloads["trellis"].get(key)
        left_hash = canonical_sha256(left)
        right_hash = canonical_sha256(right)
        if left_hash != right_hash:
            raise RuntimeError(f"{name} backend states differ in shared key {key}")
        shared_state_hashes[key] = left_hash

    selected_frame, geometry_rows, geometry_report_path = candidate_geometry_rows(run_root)
    hand_state, hand_report_path = shared_hand_state(run_root, expected_frames)
    geometry_winner = min(
        BACKENDS,
        key=lambda backend: float(geometry_rows[backend]["observed_to_generated_m"]["median_m"]),
    )
    trajectory = {
        "readiness_mode": pose_report.get("annotation_readiness_mode") or (
            "conditional_sparse_underobservable_rotation_tail" if conditional_applied else "strict_temporal_readiness_legacy_report"
        ),
        "rotation_step_acceptance_mode": rotation_gate.get("acceptance_mode") or "strict_legacy_report",
        "conditional_rotation_tail_applied": conditional_applied,
        "conditional_rotation_tail_frames": conditional_frames,
        "conditional_temporal_uncertainty": temporal.get("conditional_temporal_uncertainty"),
        "trajectory_values_modified_or_clipped": rotation_gate.get("trajectory_values_modified_or_clipped", False),
        "generated_geometry_pose_evidence_consumed": rotation_gate.get("generated_geometry_pose_evidence_consumed", False),
        "pose_hypothesis_source": pose_readiness.get("pose_hypothesis_source"),
        "generated_hidden_surface_included_in_pose": pose_readiness.get("generated_hidden_surface_included"),
        "ineligible_pose_observation_override": eligibility.get("include_ineligible_override"),
        "nonpenetration_target_frame_count": pose_report.get("nonpenetration_target_frame_count"),
        "timeline_frame_count": temporal.get("timeline_frame_count"),
        "direct_pose_count": temporal.get("direct_pose_count"),
        "direct_pose_fraction": temporal.get("direct_pose_fraction"),
        "max_direct_frame_gap": temporal.get("max_direct_frame_gap"),
        "completed_pose_count": temporal.get("completed_pose_count"),
        "completed_pose_fraction": completed_pose_fraction,
        "completed_pose_fraction_gate": {
            "comparison": "less_than_or_equal",
            "maximum": MAX_COMPLETED_POSE_FRACTION,
            "inclusive_boundary": True,
            "passed": completed_pose_fraction_passed,
        },
        "nearest_hold_count": temporal.get("nearest_hold_count"),
        "nearest_hold_fraction": temporal.get("nearest_hold_fraction"),
        "max_rotation_step_deg": temporal.get("max_rotation_step_deg"),
        "max_translation_step_m": temporal.get("max_translation_step_m"),
        "observable_fraction": (temporal.get("rotation_observability") or {}).get("observable_fraction"),
        "estimated_rotation_is_not_ground_truth_angular_velocity": True,
    }
    return {
        "name": name,
        "case_id": manifest.get("case"),
        "object_id": manifest.get("object_id"),
        "status": "complete_validated",
        "run_root": str(run_root),
        "suite_done": file_record(done_path),
        "case_result_manifest": file_record(manifest_path),
        "selected_anchor_frame": selected_frame,
        "fairness_contract": fairness,
        "shared_observed_metric_surface_sha256": next(iter(observed_hashes)),
        "shared_state_value_sha256": shared_state_hashes,
        "trajectory": trajectory,
        "hand_object_state": hand_state,
        "hand_object_state_report": file_record(hand_report_path),
        "geometry_comparison": geometry_rows,
        "geometry_comparison_report": file_record(geometry_report_path),
        "visible_fit_median_winner": geometry_winner,
        "backends": backend_rows,
        "videos": videos,
    }


def artifact_index(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for case in cases:
        run_root = Path(case["run_root"])
        paths = [
            run_root / "SUITE_DONE.json",
            run_root / "final_results/case_result_manifest.json",
            Path(case["geometry_comparison_report"]["path"]),
            Path(case["hand_object_state_report"]["path"]),
            Path(case["case_result_manifest"]["path"]),
        ]
        paths.extend(path for path in (run_root / "final_results").rglob("*") if path.is_file() and not path.is_symlink())
        for path in paths:
            path = require_file(path, "indexed artifact")
            key = str(path)
            if key not in rows:
                record = file_record(path)
                record["case"] = case["name"]
                record["relative_to_run_root"] = str(path.relative_to(run_root)) if run_root in path.parents else None
                rows[key] = record
    return [rows[key] for key in sorted(rows)]


def geometry_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_rows = []
    wins = {backend: 0 for backend in BACKENDS}
    medians = {backend: [] for backend in BACKENDS}
    p95s = {backend: [] for backend in BACKENDS}
    for case in cases:
        row = {
            "name": case["name"],
            "case_id": case["case_id"],
            "object_id": case["object_id"],
            "selected_anchor_frame": case["selected_anchor_frame"],
            "visible_fit_median_winner": case["visible_fit_median_winner"],
            "backends": case["geometry_comparison"],
        }
        wins[case["visible_fit_median_winner"]] += 1
        for backend in BACKENDS:
            medians[backend].append(float(row["backends"][backend]["observed_to_generated_m"]["median_m"]))
            p95s[backend].append(float(row["backends"][backend]["observed_to_generated_m"]["p95_m"]))
        case_rows.append(row)
    return {
        "schema": "hot3d_fivecase_backend_geometry_comparison_v1",
        "status": "complete",
        "comparison_scope": (
            "Controlled P13 visible observed-surface support for render-only generated priors. "
            "This is not ground-truth hidden-shape accuracy."
        ),
        "geometry_is_sole_branch_variable": True,
        "generated_faces_pose_eligible": False,
        "generated_faces_collision_eligible": False,
        "generated_faces_contact_eligible": False,
        "primary_backend": "sam3d",
        "baseline_backend": "trellis",
        "case_visible_fit_median_wins": wins,
        "aggregate_observed_to_generated_m": {
            backend: {"case_median_summary": numeric_summary(medians[backend]), "case_p95_summary": numeric_summary(p95s[backend])}
            for backend in BACKENDS
        },
        "cases": case_rows,
        "interpretation_limits": [
            "Lower observed-to-generated distance indicates better support for the prediction-side visible surface only.",
            "Generated-to-observed tails mix unsupported hidden extent with completeness and are not hidden-shape ground truth.",
            "Mesh density and generated hidden-face count are topology diagnostics, not quality scores.",
            "Native SAM3D projection IoU has no homologous TRELLIS score and is not used as a symmetric backend rank.",
        ],
    }


def hand_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "hot3d_fivecase_shared_hand_object_state_v1",
        "status": "complete_unsigned_partial_surface_state",
        "backend_specific_physical_state": False,
        "shared_between_sam3d_and_trellis": True,
        "generated_faces_contact_eligible": False,
        "generated_faces_collision_eligible": False,
        "signed_contact_available": False,
        "signed_nonpenetration_available": False,
        "claim_scope": (
            "Shared prediction-side HaWoR/MANO and observed-only object trajectory with unsigned proximity "
            "to partial non-watertight observed surfaces. No backend physical ranking is valid."
        ),
        "cases": [
            {
                "name": case["name"],
                "case_id": case["case_id"],
                "object_id": case["object_id"],
                "trajectory": case["trajectory"],
                "hand_object_state": case["hand_object_state"],
            }
            for case in cases
        ],
        "global_uncertainties": [
            "Signed distance, penetration, and nonpenetration remain unresolved because observed surfaces are partial and non-watertight.",
            "Zero signed penetrating rows mean signed evaluation was inactive, not that penetration was absent.",
            "HaWoR image-center principal point differs slightly from the exact sensor K; fine hand projection and handedness alignment remain uncertain.",
            "Conditional rotation-tail rows, when present, preserve observed estimates unchanged and are not ground-truth angular velocity.",
        ],
    }


def geometry_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# HOT3D 五例 SAM3D / TRELLIS 生成几何对比",
        "",
        "本报告只比较共享 prediction-side observed metric surface 对两个 render-only 生成 prior 的支持。",
        "隐藏面没有 GT，不能把完整度、面数或反向距离直接解释成真实隐藏形状精度。",
        "",
        "| Case | Anchor | SAM3D o→g median / P95 | TRELLIS o→g median / P95 | visible median winner |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["cases"]:
        sam = row["backends"]["sam3d"]["observed_to_generated_m"]
        tre = row["backends"]["trellis"]["observed_to_generated_m"]
        lines.append(
            f"| {row['name']} | {row['selected_anchor_frame']} | {1000*sam['median_m']:.3f} / {1000*sam['p95_m']:.3f} mm "
            f"| {1000*tre['median_m']:.3f} / {1000*tre['p95_m']:.3f} mm | {row['visible_fit_median_winner']} |"
        )
    lines.extend([
        "",
        f"按 observed→generated median 的逐例胜数：SAM3D {report['case_visible_fit_median_wins']['sam3d']}，"
        f"TRELLIS {report['case_visible_fit_median_wins']['trellis']}。",
        "",
        "两后端生成面始终 `pose/contact/collision eligible = false`；SAM3D 作为主要展示后端，TRELLIS 作为受控 baseline。",
        "",
    ])
    return "\n".join(lines)


def hand_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# HOT3D 五例共享手物状态报告",
        "",
        "SAM3D 与 TRELLIS 共享同一 camera、MANO、observed surface 和 observed-only object trajectory，",
        "因此手物状态不能按生成后端排名。当前仅报告 partial observed surface 上的 unsigned proximity。",
        "",
        "| Case | direct | completed | max rot | max trans | readiness | conditional frames |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for case in report["cases"]:
        trajectory = case["trajectory"]
        frames = trajectory["conditional_rotation_tail_frames"]
        lines.append(
            f"| {case['name']} | {trajectory['direct_pose_count']}/150 | {trajectory['completed_pose_count']}/150 "
            f"| {trajectory['max_rotation_step_deg']:.3f}° | {1000*trajectory['max_translation_step_m']:.3f} mm "
            f"| {trajectory['readiness_mode']} | {frames or 'none'} |"
        )
    lines.extend([
        "",
        "所有 case 的 signed geometry query 与 signed nonpenetration 均未激活。`penetrating=0` 不能解释为无穿透。",
        "generated hidden faces 不参与 contact、collision、signed distance 或 pose。",
        "",
    ])
    return "\n".join(lines)


def readme_markdown(cases: list[dict[str, Any]], geometry: dict[str, Any]) -> str:
    lines = [
        "# HOT3D five-case SAM3D/TRELLIS demo collection",
        "",
        "本目录是 5 个 150-frame / 30 FPS / 5 秒 HOT3D 针孔片段的最终双后端索引。",
        "每例都由 named D19 finalizer 发布；`runs/` 与 `final_results/` 只用于集中查看，不是 prediction 输入。",
        "",
        "## 完成状态",
        "",
        "| Case | Object | D19 manifest SHA256 | 轨迹模式 |",
        "|---|---|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case['name']} | `{case['object_id']}` | `{case['case_result_manifest']['sha256']}` "
            f"| `{case['trajectory']['readiness_mode']}` |"
        )
    lines.extend([
        "",
        "每例每后端包含 `camera_overlay.mp4`、`world_view.mp4`、`side_world_view.mp4`、`side_by_side.mp4`，",
        "本 collection finalizer 已逐帧实际解码全部 40 个视频。",
        f"Temporal audit 使用 inclusive gate `completed_pose_fraction <= {MAX_COMPLETED_POSE_FRACTION:.2f}`；",
        "Spatula 的 `30/150 = 0.20` 位于允许边界并判定通过。",
        "",
        "## 公平性与语义",
        "",
        "- SAM3D/TRELLIS 共享 P00–P11 evidence、anchor、camera、MANO、observed metric surface 和 object trajectory。",
        "- 唯一 branch variable 是单图 generated render geometry prior 及其 integration。",
        "- generated faces 永远不提供 pose/contact/collision/signed evidence。",
        "- SAM3D 为主要 demo 后端；TRELLIS 为公平 baseline。",
        "- signed contact/nonpenetration 对所有 case 仍 unresolved，不能从零 penetrating rows 推断无穿透。",
        "",
        "## 报告",
        "",
        "- `ARTIFACT_SHA256_INDEX.json`：统一 artifact 索引。",
        "- `SHA256SUMS.txt`：所有 frozen case artifacts 与 collection reports 的 SHA256。",
        "- `BACKEND_GEOMETRY_COMPARISON.json/.md`：后端可见面几何对比。",
        "- `SHARED_HAND_OBJECT_STATE_REPORT.json/.md`：共享轨迹和 unsigned 手物状态。",
        "- `collection_manifest.json`：五例总清单与验证摘要。",
        "",
        f"逐例 visible-fit median 胜数：SAM3D {geometry['case_visible_fit_median_wins']['sam3d']}，"
        f"TRELLIS {geometry['case_visible_fit_median_wins']['trellis']}。这不是隐藏形状 GT 排名。",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    collection_root = args.collection_root.expanduser().resolve()
    bindings = [parse_case_binding(value) for value in args.case]
    names = [name for name, _root in bindings]
    if len(bindings) != int(args.expected_case_count):
        raise RuntimeError(f"expected {args.expected_case_count} cases, received {len(bindings)}")
    if len(set(names)) != len(names):
        raise RuntimeError(f"duplicate collection case names: {names}")
    cases = [
        validate_case(name, root, int(args.expected_frame_count), float(args.expected_fps))
        for name, root in bindings
    ]
    geometry = geometry_report(cases)
    hands = hand_report(cases)
    indexed_artifacts = artifact_index(cases)
    videos = [row for case in cases for row in case["videos"]]
    if len(videos) != len(cases) * len(BACKENDS) * len(VIDEO_KEYS):
        raise RuntimeError("collection does not contain exactly four videos per backend per case")

    collection_root.mkdir(parents=True, exist_ok=True)
    collection_links: list[dict[str, Any]] = []
    if args.replace_links:
        for case in cases:
            collection_links.append(
                replace_symlink(
                    collection_root / "runs" / case["name"],
                    Path(case["run_root"]),
                    allow_empty_link_stub_repair=bool(args.repair_empty_link_stubs),
                )
            )
            collection_links.append(
                replace_symlink(
                    collection_root / "final_results" / case["name"],
                    Path(case["run_root"]) / "final_results",
                    allow_empty_link_stub_repair=bool(args.repair_empty_link_stubs),
                )
            )
        for stale_name in ("remaining4_suite", "live_progress"):
            stale = collection_root / stale_name
            if stale.is_symlink():
                stale.unlink()
        for row in collection_links:
            validate_symlink(Path(row["path"]), Path(row["target"]))

    geometry_json = collection_root / "BACKEND_GEOMETRY_COMPARISON.json"
    geometry_md = collection_root / "BACKEND_GEOMETRY_COMPARISON_ZH.md"
    hand_json = collection_root / "SHARED_HAND_OBJECT_STATE_REPORT.json"
    hand_md = collection_root / "SHARED_HAND_OBJECT_STATE_REPORT_ZH.md"
    index_json = collection_root / "ARTIFACT_SHA256_INDEX.json"
    readme = collection_root / "README_ZH.md"
    write_json(geometry_json, geometry)
    atomic_write(geometry_md, geometry_markdown(geometry))
    write_json(hand_json, hands)
    atomic_write(hand_md, hand_markdown(hands))
    index_payload = {
        "schema": "hot3d_fivecase_artifact_sha256_index_v1",
        "status": "complete",
        "case_count": len(cases),
        "backend_count": len(cases) * len(BACKENDS),
        "video_count": len(videos),
        "all_videos_actually_decoded": True,
        "artifacts": indexed_artifacts,
    }
    write_json(index_json, index_payload)
    atomic_write(readme, readme_markdown(cases, geometry))

    report_records = {
        "artifact_index": file_record(index_json),
        "backend_geometry_comparison_json": file_record(geometry_json),
        "backend_geometry_comparison_zh": file_record(geometry_md),
        "shared_hand_object_state_json": file_record(hand_json),
        "shared_hand_object_state_zh": file_record(hand_md),
        "readme_zh": file_record(readme),
    }
    manifest = {
        "schema": "hot3d_fivecase_demo_collection_v2",
        "status": (
            "complete_validated_five_cases"
            if len(cases) == 5
            else "complete_validated_collection_rehearsal"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "collection_root": str(collection_root),
        "case_count": len(cases),
        "backend_count": len(cases) * len(BACKENDS),
        "video_count": len(videos),
        "expected_frame_count_per_video": int(args.expected_frame_count),
        "expected_fps": float(args.expected_fps),
        "all_videos_actually_decoded": True,
        "readiness_policy": {
            "completed_pose_fraction": {
                "comparison": "less_than_or_equal",
                "maximum": MAX_COMPLETED_POSE_FRACTION,
                "inclusive_boundary": True,
                "all_cases_passed": all(
                    bool(case["trajectory"]["completed_pose_fraction_gate"]["passed"])
                    for case in cases
                ),
                "decision_provenance": (
                    "Operator accepted an exact completed-pose fraction of 0.20 as passing; "
                    "the collection audit therefore applies <= 0.20."
                ),
            }
        },
        "fairness_contract": {
            "shared_prediction_evidence": True,
            "shared_anchor_per_case": True,
            "shared_object_owned_mask_per_case": True,
            "shared_observed_metric_surface_per_case": True,
            "shared_observed_only_pose_trajectory_per_case": True,
            "shared_camera_and_metric_mano_state_per_case": True,
            "geometry_is_sole_branch_variable": True,
            "generated_faces_pose_eligible": False,
            "generated_faces_collision_eligible": False,
            "generated_faces_contact_eligible": False,
        },
        "primary_backend": "sam3d",
        "comparison_backend": "trellis",
        "finalizer_implementation": file_record(Path(__file__).resolve()),
        "collection_links": {
            "requested": bool(args.replace_links),
            "validated": bool(args.replace_links) and all(
                bool(row.get("validated")) for row in collection_links
            ),
            "link_count": len(collection_links),
            "update_mode": (
                "validated_direct_relative_symlink" if args.replace_links else "not_requested"
            ),
            "atomic_symlink_rename_used": False,
            "reason": (
                "Relative symlinks are published directly and resolved back to their exact "
                "targets; symlink rename is not assumed safe on CIFS nounix mounts."
            ),
            "empty_link_stub_repair_enabled": bool(args.repair_empty_link_stubs),
            "links": collection_links,
        },
        "cases": cases,
        "reports": report_records,
        "post_prediction_aggregation_only": True,
        "prediction_artifacts_mutated": False,
        "elapsed_s": time.time() - started,
    }
    manifest_path = collection_root / "collection_manifest.json"
    write_json(manifest_path, manifest)

    sum_paths = [Path(row["path"]) for row in indexed_artifacts]
    sum_paths.extend(Path(row["path"]) for row in report_records.values())
    sum_paths.append(manifest_path)
    unique_sum_paths = sorted(set(path.resolve() for path in sum_paths), key=str)
    sums_path = collection_root / "SHA256SUMS.txt"
    atomic_write(
        sums_path,
        "".join(f"{sha256_file(path)}  {path}\n" for path in unique_sum_paths),
    )
    done = {
        "schema": "hot3d_fivecase_collection_done_v1",
        "status": "complete",
        "collection_manifest": str(manifest_path),
        "collection_manifest_sha256": sha256_file(manifest_path),
        "artifact_index": str(index_json),
        "artifact_index_sha256": sha256_file(index_json),
        "sha256sums": str(sums_path),
        "sha256sums_sha256": sha256_file(sums_path),
        "case_count": len(cases),
        "video_count": len(videos),
        "collection_link_count": len(collection_links),
        "collection_links_validated": bool(manifest["collection_links"]["validated"]),
        "completed_pose_fraction_maximum": MAX_COMPLETED_POSE_FRACTION,
        "completed_pose_fraction_boundary_inclusive": True,
        "finalizer_sha256": manifest["finalizer_implementation"]["sha256"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    done_path = collection_root / "COLLECTION_DONE.json"
    write_json(done_path, done)
    print(json.dumps({**done, "collection_done": str(done_path)}, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="NAME=/absolute/completed/run/root")
    parser.add_argument("--expected-case-count", type=int, default=5)
    parser.add_argument("--expected-frame-count", type=int, default=150)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--replace-links", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--repair-empty-link-stubs",
        action="store_true",
        help=(
            "Explicitly allow replacement of zero-byte regular files at expected collection-link "
            "paths. This is only for repairing stubs produced by the older CIFS-unsafe symlink "
            "rename implementation; other non-symlinks still fail closed."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
