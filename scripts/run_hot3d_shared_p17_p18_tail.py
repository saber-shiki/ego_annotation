#!/usr/bin/env python3
"""Run the shared HOT3D P17 -> P18 -> P18b tail exactly once.

This orchestrator is deliberately upstream of the SAM3D/TRELLIS render split. It
accepts one D14 pose body and one D15 object trajectory, builds one P17 factor
report from an agent-authored interaction judgment, and runs one shared P18/P18b
MANO state with object translation disabled. An optional backend-neutral signed
completion report may replace only the physical surface consumed by P18; the
D14 observed mesh remains the pose canonical body. Generated backend geometry is
never an input to this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

SCHEMA = "v19_hot3d_shared_p17_p18_tail_v1"
ACCEPTED_POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "corrected_temporal_rigid_pose_graph",
    "completed_temporal_rigid_pose_uncertain",
}


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def load_json(path: Path, description: str = "JSON input") -> dict[str, Any]:
    path = require_file(path, description)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must be one JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mesh_geometry_sha256(path: Path) -> tuple[str, dict[str, Any]]:
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        meshes = [
            item
            for item in geometry.geometry.values()
            if isinstance(item, trimesh.Trimesh) and len(item.vertices) and len(item.faces)
        ]
        if not meshes:
            raise RuntimeError(f"mesh scene has no triangle geometry: {path}")
        geometry = trimesh.util.concatenate(meshes)
    if not isinstance(geometry, trimesh.Trimesh):
        raise RuntimeError(f"unsupported mesh type {type(geometry)}: {path}")
    vertices = np.ascontiguousarray(np.asarray(geometry.vertices, dtype="<f8"))
    faces = np.ascontiguousarray(np.asarray(geometry.faces, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(vertices.tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest(), {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "geometry_sha256_f64_i64": digest.hexdigest(),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "watertight": bool(geometry.is_watertight),
    }


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def active_frame_ids(annotations: dict[str, Any]) -> list[int]:
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotations have no frames")
    ids = [
        int(frame.get("frame_idx", pos))
        for pos, frame in enumerate(frames)
        if isinstance(frame, dict)
    ]
    if len(ids) != len(frames) or len(ids) != len(set(ids)):
        raise RuntimeError("annotation frame IDs are missing or duplicated")
    return ids


def validate_camera_mano_contract(
    annotations: dict[str, Any], frame_span: tuple[int, int], sides: set[str]
) -> dict[str, Any]:
    start, end = frame_span
    rows = 0
    intrinsics: list[list[float]] = []
    for pos, frame in enumerate(annotations.get("frames") or []):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", pos))
        if frame_idx < start or frame_idx > end:
            continue
        camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
        active = np.asarray(camera.get("intrinsics_fx_fy_cx_cy") or [], dtype=np.float64)
        if active.shape != (4,) or not np.isfinite(active).all() or np.any(active[:2] <= 0):
            raise RuntimeError(f"frame {frame_idx}: invalid active camera K")
        hands = {
            str(hand.get("hand_side")): hand
            for hand in frame.get("hands") or []
            if isinstance(hand, dict)
        }
        for side in sorted(sides):
            hand = hands.get(side)
            if not isinstance(hand, dict):
                raise RuntimeError(f"frame {frame_idx}: missing {side} MANO row")
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            alignment = (
                metric.get("camera_contract_alignment")
                if isinstance(metric.get("camera_contract_alignment"), dict)
                else hand.get("camera_contract_alignment")
                if isinstance(hand.get("camera_contract_alignment"), dict)
                else {}
            )
            if alignment.get("active_contract_reinference_required") is not False:
                raise RuntimeError(
                    f"frame {frame_idx} {side}: active source-K/centered-inference HaWoR reinference is required"
                )
            if alignment.get("source_hawor_state_intrinsics_match_active_contract") is not True:
                raise RuntimeError(
                    f"frame {frame_idx} {side}: source-K camera/MANO binding is not proven"
                )
            if alignment.get("source_hawor_full_K_bound_to_inference") is not True:
                raise RuntimeError(
                    f"frame {frame_idx} {side}: HaWoR source-K to centered-inference-plane binding is not proven"
                )
            plane = alignment.get("hawor_camera_image_plane_contract")
            if (
                not isinstance(plane, dict)
                or plane.get("validated") is not True
                or plane.get("status")
                != "validated_source_K_affine_centered_hawor_inference_plane"
                or not plane.get("contract_sha256")
                or not plane.get("inference_frames_aggregate_sha256")
            ):
                raise RuntimeError(
                    f"frame {frame_idx} {side}: source-K/centered-HaWoR image-plane contract is not validated"
                )
            mano = np.asarray(
                metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy") or [],
                dtype=np.float64,
            )
            if mano.shape != (4,) or not np.allclose(mano, active, atol=0.01, rtol=0.0):
                raise RuntimeError(
                    f"frame {frame_idx} {side}: MANO K {mano.tolist()} != active K {active.tolist()}"
                )
            rows += 1
            intrinsics.append(active.astype(float).tolist())
    expected = (end - start + 1) * len(sides)
    if rows != expected:
        raise RuntimeError(f"camera/MANO contract covers {rows}/{expected} interval rows")
    unique = sorted({tuple(row) for row in intrinsics})
    plane_hashes = {
        str(
            (
                (hand.get("metric_mano_state") or {}).get(
                    "camera_contract_alignment", hand.get("camera_contract_alignment", {})
                )
                or {}
            ).get("hawor_camera_image_plane_contract", {}).get("contract_sha256")
        )
        for frame in annotations.get("frames") or []
        if isinstance(frame, dict)
        and start <= int(frame.get("frame_idx", -1)) <= end
        for hand in frame.get("hands") or []
        if isinstance(hand, dict) and str(hand.get("hand_side")) in sides
    }
    if len(plane_hashes) != 1 or "" in plane_hashes or "None" in plane_hashes:
        raise RuntimeError(
            f"camera/MANO rows do not share one source-to-centered inference contract: {sorted(plane_hashes)}"
        )
    return {
        "status": "exact_source_K_affine_bound_to_centered_hawor_plane_for_all_mano_rows",
        "row_count": rows,
        "expected_row_count": expected,
        "unique_intrinsics_fx_fy_cx_cy": [list(row) for row in unique],
        "hawor_camera_image_plane_contract_sha256": next(iter(plane_hashes)),
    }


def validate_pose_graph(
    pose: dict[str, Any], expected_frame_ids: list[int]
) -> dict[str, Any]:
    graph = pose.get("graph_support") if isinstance(pose.get("graph_support"), dict) else {}
    temporal = (
        pose.get("temporal_readiness")
        if isinstance(pose.get("temporal_readiness"), dict)
        else {}
    )
    if pose.get("annotation_ready") is not True or graph.get("sufficient") is not True:
        raise RuntimeError("D15 pose graph is not annotation-ready with sufficient support")
    if temporal.get("ready") is not True:
        raise RuntimeError("D15 temporal readiness is not true")
    rows = [
        row
        for row in pose.get("pose_rows") or []
        if isinstance(row, dict) and str(row.get("status")) in ACCEPTED_POSE_STATUSES
    ]
    ids = sorted(int(row["frame_idx"]) for row in rows)
    if ids != sorted(expected_frame_ids):
        raise RuntimeError(
            f"D15 accepted pose timeline does not equal annotations: {len(ids)} != {len(expected_frame_ids)}"
        )
    if int(pose.get("nonpenetration_target_frame_count", -1)) != 0:
        raise RuntimeError("D15 is not observed-only: nonpenetration targets are present")
    return {
        "status": "ready_full_timeline_observed_only_pose_authority",
        "frame_count": len(ids),
        "frame_range": [min(ids), max(ids)],
        "annotation_ready": True,
        "graph_support_sufficient": True,
        "nonpenetration_target_frame_count": 0,
        "pose_rows_value_sha256": value_sha256(rows),
    }


def validate_observed_surface(
    completion: dict[str, Any], pose: dict[str, Any]
) -> tuple[Path, Path, dict[str, Any]]:
    outputs = completion.get("outputs") if isinstance(completion.get("outputs"), dict) else {}
    readiness = (
        completion.get("geometry_readiness")
        if isinstance(completion.get("geometry_readiness"), dict)
        else {}
    )
    pose_mesh = require_file(
        Path(str(outputs.get("pose_hypothesis_mesh_labeled") or "")),
        "D14 pose-hypothesis mesh",
    )
    physical = require_file(
        Path(str(outputs.get("collision_eligible_mesh_labeled") or "")),
        "D14 observed physical surface",
    )
    if not same_path(pose_mesh, physical):
        raise RuntimeError(
            "shared P18 tail requires the current D14 observed-only pose and physical surfaces to be the same exact file"
        )
    for key in ("generated_hidden_surface_included", "generated_faces_collision_eligible", "generated_faces_contact_eligible", "signed_geometry_ready"):
        if readiness.get(key) is not False:
            raise RuntimeError(f"D14 does not explicitly preserve {key}=false")
    pose_input = require_file(
        Path(str((pose.get("inputs") or {}).get("completed_mesh") or "")),
        "D15 canonical pose mesh",
    )
    physical_hash, physical_summary = mesh_geometry_sha256(physical)
    pose_hash, _pose_summary = mesh_geometry_sha256(pose_input)
    if pose_hash != physical_hash:
        raise RuntimeError("D15 canonical pose mesh is not the shared observed physical surface")
    if physical_summary["watertight"]:
        # A watertight observed-only surface is not forbidden, but the current
        # completion contract still controls signed readiness. Keep it unsigned.
        physical_summary["watertight_but_signed_readiness_still_false"] = True
    return pose_mesh, physical, {
        "status": "shared_observed_only_unsigned_physical_surface",
        **physical_summary,
        "pose_hypothesis_same_file": True,
        "d15_canonical_geometry_sha256_match": True,
        "generated_faces_pose_eligible": False,
        "generated_faces_contact_eligible": False,
        "generated_faces_collision_eligible": False,
        "signed_geometry_ready": False,
    }


def validate_signed_completion(
    signed_completion: dict[str, Any],
    pose_mesh: Path,
    observed_physical_surface: Path,
    pose_report: Path,
) -> tuple[Path, bool, dict[str, Any]]:
    outputs = signed_completion.get("outputs") if isinstance(signed_completion.get("outputs"), dict) else {}
    readiness = signed_completion.get("geometry_readiness") if isinstance(signed_completion.get("geometry_readiness"), dict) else {}
    signed_pose = require_file(Path(str(outputs.get("pose_hypothesis_mesh_labeled") or "")), "signed completion pose hypothesis")
    signed_surface = require_file(Path(str(outputs.get("collision_eligible_mesh_labeled") or "")), "signed completion collision surface")
    if not same_path(signed_pose, pose_mesh):
        raise RuntimeError("signed completion pose hypothesis differs from D14 canonical pose body")
    signed_ready = readiness.get("signed_geometry_ready") is True
    for key in (
        "generated_hidden_surface_included",
        "generated_faces_collision_eligible",
        "generated_faces_contact_eligible",
        "generated_faces_signed_distance_eligible",
    ):
        if readiness.get(key) is True:
            raise RuntimeError(f"signed completion illegally promotes generated geometry: {key}")
    mesh = trimesh.load(signed_surface, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [item for item in mesh.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("signed completion surface scene has no mesh")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("signed completion surface is not a triangle mesh")
    if signed_ready and not bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume):
        raise RuntimeError("signed-ready completion surface is not a watertight winding-consistent volume")
    if not signed_ready:
        signed_hash, _ = mesh_geometry_sha256(signed_surface)
        observed_hash, _ = mesh_geometry_sha256(observed_physical_surface)
        if signed_hash != observed_hash:
            raise RuntimeError("unready signed completion fallback differs from D14 observed physical surface")
    signed_pose_report = signed_completion.get("inputs", {}).get("pose_report") if isinstance(signed_completion.get("inputs"), dict) else None
    if not signed_pose_report or not same_path(Path(str(signed_pose_report)), pose_report):
        raise RuntimeError("signed completion was not built from the shared D15 pose report")
    return signed_surface, signed_ready, {
        "status": (
            "shared_backend_neutral_signed_physical_surface"
            if signed_ready
            else "shared_signed_geometry_attempt_unsigned_observed_fallback"
        ),
        "signed_geometry_ready": signed_ready,
        "signed_completion_report_status": signed_completion.get("status"),
        "signed_completion_report_sha256": None,
        "signed_surface_mesh": str(signed_surface),
        "signed_surface_watertight": bool(mesh.is_watertight),
        "signed_surface_winding_consistent": bool(mesh.is_winding_consistent),
        "signed_surface_is_volume": bool(mesh.is_volume),
        "candidate_failure": signed_completion.get("candidate_failure"),
        "candidate_validation": signed_completion.get("candidate_validation"),
        "generated_geometry_consumed": False,
        "pose_report": str(pose_report),
    }


def validate_interaction_judgment(
    judgment: dict[str, Any], *, case: str, target: str, start: int, end: int, sides: set[str]
) -> dict[str, Any]:
    if judgment.get("status") not in (None, "ok"):
        raise RuntimeError("agent interaction judgment status is not ok")
    if judgment.get("case") not in (None, case):
        raise RuntimeError("agent interaction judgment case does not match")
    rows = judgment.get("interaction_judgments")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("agent interaction judgment has no interaction_judgments")
    covered: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("agent interaction judgment contains a non-object row")
        raw_target = str(row.get("target_entity_id") or "")
        if raw_target.split(":", 1)[-1] != target.split(":", 1)[-1]:
            continue
        side = str(row.get("hand_side") or "")
        if side not in sides:
            continue
        row_start = max(start, int(row.get("frame_start", start)))
        row_end = min(end, int(row.get("frame_end", end)))
        for frame_idx in range(row_start, row_end + 1):
            key = (frame_idx, side)
            if key in covered:
                raise RuntimeError(f"overlapping interaction judgment at {key}")
            covered.add(key)
    expected = {(frame_idx, side) for frame_idx in range(start, end + 1) for side in sides}
    missing = sorted(expected - covered)
    if missing:
        raise RuntimeError(
            f"interaction judgment is not a complete interval prior: missing {len(missing)} rows, first={missing[:8]}"
        )
    return {
        "status": "complete_agent_visual_interaction_prior",
        "segment_count": len(rows),
        "frame_side_count": len(covered),
        "expected_frame_side_count": len(expected),
    }


def run_command(command: list[str], *, label: str, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"label": label, "command": command, "executed": False}
    print(json.dumps({"event": "launch", "label": label, "command": command}), flush=True)
    subprocess.run(command, check=True)
    return {"label": label, "command": command, "executed": True}


def expected_p17_keys(
    annotations: dict[str, Any], *, target_entity_id: str, start: int, end: int, sides: set[str]
) -> set[tuple[int, str]]:
    bare = target_entity_id.split(":", 1)[-1]
    expected: set[tuple[int, str]] = set()
    for pos, frame in enumerate(annotations.get("frames") or []):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", pos))
        if frame_idx < start or frame_idx > end:
            continue
        target = None
        for obj in frame.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            ids = {
                str(obj.get("object_id")),
                str(obj.get("track_id")),
                str(obj.get("object_id")).split(":", 1)[-1],
                str(obj.get("track_id")).split(":", 1)[-1],
            }
            if bare in ids or target_entity_id in ids:
                target = obj
                break
        if (
            not isinstance(target, dict)
            or not bool(target.get("visible", True))
            or target.get("rigid_pose_observation_eligible") is False
        ):
            continue
        mask = target.get("mask_path")
        if not isinstance(mask, str) or not Path(mask).is_file():
            continue
        for hand in frame.get("hands") or []:
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side") or hand.get("side") or "")
            if side in sides:
                expected.add((frame_idx, side))
    if not expected:
        raise RuntimeError("P09 annotations expose no eligible P17 frame/side rows")
    return expected


def validate_p17_factor(
    factor: dict[str, Any], expected_keys: set[tuple[int, str]]
) -> dict[str, Any]:
    ownership = [row for row in factor.get("ownership_rows") or [] if isinstance(row, dict)]
    actual_keys = {(int(row["frame_idx"]), str(row["hand_side"])) for row in ownership}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"P17 ownership keys mismatch: actual={len(actual_keys)} expected={len(expected_keys)} "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    factor_rows = [row for row in factor.get("factor_rows") or [] if isinstance(row, dict)]
    factor_keys = {(int(row["frame_idx"]), str(row["hand_side"])) for row in factor_rows}
    if factor_keys != expected_keys:
        raise RuntimeError(
            f"complete interaction judgment did not produce one P17 contact-prior row per eligible key: "
            f"actual={len(factor_keys)} expected={len(expected_keys)}"
        )
    affine_hashes: set[str] = set()
    for row in ownership:
        plane = row.get("image_plane_transform")
        if not isinstance(plane, dict) or plane.get("camera_contract_consistent") is not True:
            raise RuntimeError("P17 row lacks camera-consistent source/mask affine")
        A = np.asarray(plane.get("A_mask_from_source_coordinate_model"), dtype=np.float64)
        if A.shape != (3, 3) or not np.isfinite(A).all():
            raise RuntimeError("P17 row has invalid A_mask_from_source")
        mask_path = require_file(
            Path(str(row.get("non_object_owned_mask_path") or "")),
            "P17 ownership mask",
        )
        affine_hashes.add(value_sha256({"A": A.tolist(), "mask": str(mask_path)}))
    return {
        "status": "exact_source_to_mask_affines_bound",
        "ownership_row_count": len(ownership),
        "factor_row_count": len(factor_rows),
        "eligible_annotation_key_count": len(expected_keys),
        "affine_mask_binding_count": len(affine_hashes),
    }


def validate_p18(
    state: dict[str, Any], *, expected_rows: int, pose_report: Path, physical_surface: Path, signed_expected: bool = False
) -> dict[str, Any]:
    if state.get("optimization_skipped") is True:
        raise RuntimeError("P18 unexpectedly skipped optimization")
    parameters = state.get("parameters") if isinstance(state.get("parameters"), dict) else {}
    if parameters.get("optimize_object_translation") is not False:
        raise RuntimeError("P18 did not explicitly disable object translation optimization")
    inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
    input_pose = require_file(Path(str(inputs.get("pose_report") or "")), "P18 pose report")
    input_surface = require_file(
        Path(str(inputs.get("physical_surface_mesh") or "")), "P18 physical surface"
    )
    if not same_path(input_pose, pose_report):
        raise RuntimeError("P18 did not consume the one D15 pose authority")
    if not same_path(input_surface, physical_surface):
        raise RuntimeError("P18 did not consume the shared observed physical surface")
    rows = [row for row in state.get("per_frame_states") or [] if isinstance(row, dict)]
    if len(rows) != expected_rows:
        raise RuntimeError(f"P18 state rows {len(rows)} != expected {expected_rows}")
    object_delta_norms = []
    for row in rows:
        delta = np.asarray(
            row.get("optimized_object_translation_world_m") or [], dtype=np.float64
        )
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise RuntimeError("P18 row has invalid object translation delta")
        object_delta_norms.append(float(np.linalg.norm(delta)))
    max_delta = max(object_delta_norms, default=math.inf)
    if max_delta > 1.0e-10:
        raise RuntimeError(f"P18 privately moved the object by up to {max_delta} m")
    signed_states = [
        str(row.get("signed_object_surface_factor_state") or "") for row in rows
    ]
    signed_active = bool(signed_states) and all(
        state.startswith("active_explicit_signed_geometry_ready") for state in signed_states
    )
    if signed_expected and not signed_active:
        raise RuntimeError("P18 did not activate the explicitly ready signed physical surface")
    if not signed_expected and any(state.startswith("active") for state in signed_states):
        raise RuntimeError("P18 activated signed geometry on the unsigned observed surface")
    return {
        "status": (
            "shared_signed_mano_candidate_object_pose_frozen"
            if signed_expected
            else "shared_unsigned_mano_candidate_object_pose_frozen"
        ),
        "signed_geometry_ready": bool(signed_expected),
        "row_count": len(rows),
        "max_object_translation_delta_m": max_delta,
        "signed_object_surface_factor_active": signed_active,
        "pose_report_sha256": sha256_file(pose_report),
        "physical_surface_sha256": sha256_file(physical_surface),
    }


def validate_p18b(
    state: dict[str, Any], *, expected_rows: int, p18_state_path: Path, signed_expected: bool = False
) -> dict[str, Any]:
    rows = [row for row in state.get("per_frame_states") or [] if isinstance(row, dict)]
    if len(rows) != expected_rows:
        raise RuntimeError(f"P18b state rows {len(rows)} != expected {expected_rows}")
    sample_rows = 0
    sample_points = 0
    signed_full_rows = 0
    for row in rows:
        policy = str(row.get("joint_state_policy") or "")
        if "metric_mano_preserved" not in policy and "p18_signed_full_778_mano_accepted" not in policy:
            raise RuntimeError("P18b row does not carry an accepted metric MANO policy")
        if "p18_signed_full_778_mano_accepted" in policy:
            signed_full_rows += 1
        delta = np.asarray(
            row.get("optimized_object_translation_world_m") or [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        if delta.shape != (3,) or float(np.linalg.norm(delta)) > 1.0e-10:
            raise RuntimeError("P18b carries a nonzero private object translation")
        samples = np.asarray(
            row.get("contact_surface_vertices_world_sample_m") or [], dtype=np.float64
        )
        if samples.size:
            if samples.ndim != 2 or samples.shape[1] != 3 or not np.isfinite(samples).all():
                raise RuntimeError("P18b has malformed contact-surface samples")
            sample_rows += 1
            sample_points += int(len(samples))
    inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
    contact_input = require_file(
        Path(str(inputs.get("contact_state") or "")), "P18b contact-state input"
    )
    if not same_path(contact_input, p18_state_path):
        raise RuntimeError("P18b did not consume this run's shared P18 state")
    if signed_full_rows not in (0, expected_rows):
        raise RuntimeError(
            f"P18b partially accepted full MANO on {signed_full_rows}/{expected_rows} rows"
        )
    if not signed_expected and signed_full_rows:
        raise RuntimeError("unsigned P18b unexpectedly accepted signed full MANO")
    acceptance = state.get("full_mano_acceptance") if isinstance(state.get("full_mano_acceptance"), dict) else {}
    signed_full_accepted = signed_full_rows == expected_rows
    if bool(acceptance.get("accepted")) != signed_full_accepted:
        raise RuntimeError("P18b full-MANO acceptance flag differs from accepted row policy")
    return {
        "status": (
            "signed_full_778_mano_accepted"
            if signed_full_accepted
            else "signed_geometry_ready_source_mano_preserved_due_to_acceptance_blockers"
            if signed_expected
            else "metric_mano_preserved_with_shared_uncertain_surface_hypothesis"
        ),
        "row_count": len(rows),
        "rows_with_surface_samples": sample_rows,
        "surface_sample_point_count": sample_points,
        "signed_full_mano_accepted_row_count": signed_full_rows,
        "signed_geometry_ready": bool(signed_expected),
        "signed_full_mano_accepted": signed_full_accepted,
        "signed_full_mano_acceptance_blockers": acceptance.get("blockers") or [],
        "accepted_full_mano_vertices_world_archive": state.get("accepted_full_mano_vertices_world_archive"),
        "accepted_full_mano_vertices_world_archive_sha256": state.get("accepted_full_mano_vertices_world_archive_sha256"),
        "temporal_state_value_sha256": value_sha256(state),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations_path = require_file(args.annotations, "P09 annotations")
    pose_path = require_file(args.pose_report, "D15 pose graph")
    completion_path = require_file(args.completion_report, "D14 completion report")
    signed_completion_path = require_file(args.signed_completion_report, "shared signed completion report") if args.signed_completion_report is not None else None
    depth_path = require_file(args.depth_npz, "camera-bound depth NPZ")
    hawor_path = require_file(args.hawor_npz, "HaWoR world MANO NPZ")
    judgment_path = require_file(args.interaction_judgment, "P17 interaction judgment")
    wilor_root = args.wilor_root.expanduser().resolve()
    mano_left = require_file(args.wilor_mano_left, "left MANO model")
    mano_right = require_file(
        args.wilor_mano_right or wilor_root / "mano_data/MANO_RIGHT.pkl",
        "right MANO model",
    )
    scripts_root = Path(__file__).resolve().parent
    p17_script = require_file(
        scripts_root / "build_v19_visible_contact_ownership_factor.py", "P17 script"
    )
    p18_script = require_file(
        scripts_root / "solve_v18_joint_mano_interval_trajectory.py", "P18 script"
    )
    p18b_script = require_file(
        scripts_root / "build_v19_mano_surface_hypothesis_state.py", "P18b script"
    )

    annotations = load_json(annotations_path, "P09 annotations")
    pose = load_json(pose_path, "D15 pose graph")
    completion = load_json(completion_path, "D14 completion report")
    judgment = load_json(judgment_path, "P17 interaction judgment")
    frame_ids = active_frame_ids(annotations)
    start, end = int(args.start_frame), int(args.end_frame)
    if start > end or start < min(frame_ids) or end > max(frame_ids):
        raise RuntimeError(f"invalid shared P18 frame span [{start},{end}]")
    sides = set(args.sides)
    expected_rows = (end - start + 1) * len(sides)
    target_entity_id = f"object:{str(args.object_id).split(':', 1)[-1]}"
    p17_expected_keys = expected_p17_keys(
        annotations,
        target_entity_id=target_entity_id,
        start=start,
        end=end,
        sides=sides,
    )

    preflight = {
        "camera_mano": validate_camera_mano_contract(
            annotations, (start, end), sides
        ),
        "pose": validate_pose_graph(pose, frame_ids),
    }
    pose_mesh, observed_physical_surface, surface_report = validate_observed_surface(completion, pose)
    physical_surface = observed_physical_surface
    signed_geometry_ready = False
    if signed_completion_path is not None:
        signed_completion = load_json(signed_completion_path, "shared signed completion report")
        physical_surface, signed_geometry_ready, signed_report = validate_signed_completion(
            signed_completion,
            pose_mesh,
            observed_physical_surface,
            pose_path,
        )
        signed_report["signed_completion_report_path"] = str(signed_completion_path)
        signed_report["signed_completion_report_sha256"] = sha256_file(signed_completion_path)
        surface_report = {**surface_report, **signed_report}
    preflight["physical_surface"] = surface_report
    signed_surface_uncertainty_m = float(
        ((signed_completion.get("geometry_readiness") or {}).get("signed_geometry_support_uncertainty_m") or 0.0)
        if signed_completion_path is not None
        else 0.0
    )
    if signed_surface_uncertainty_m < 0.0 or not np.isfinite(signed_surface_uncertainty_m):
        raise RuntimeError("signed completion has invalid support uncertainty")
    preflight["interaction_judgment"] = validate_interaction_judgment(
        judgment,
        case=str(args.case),
        target=target_entity_id,
        start=start,
        end=end,
        sides=sides,
    )

    output_root = args.output_root.expanduser().resolve()
    p17_root = output_root / "P17_visible_contact_ownership"
    p18_root = output_root / "P18_shared_mano_interval"
    p18b_path = (
        output_root
        / "P18b_shared_surface_hypothesis_metric_mano"
        / str(args.case)
        / "v18_joint_mano_interval_trajectory_state.json"
    )
    factor_path = (
        p17_root / str(args.case) / "v19_visible_contact_ownership_factor_report.json"
    )
    p18_path = (
        p18_root / str(args.case) / "v18_joint_mano_interval_trajectory_state.json"
    )
    report_path = output_root / "shared_p17_p18_tail_report.json"
    for output in (factor_path, p18_path, p18b_path, report_path):
        if output.exists() and not args.replace:
            raise RuntimeError(f"refusing to overwrite shared tail output: {output}")
    if args.replace:
        import shutil
        for directory in (p17_root, p18_root, p18b_path.parents[1]):
            if directory.exists():
                shutil.rmtree(directory)

    commands = []
    commands.append(
        run_command(
            [
                sys.executable,
                str(p17_script),
                "--annotations",
                str(annotations_path),
                "--case",
                str(args.case),
                "--target-entity-id",
                target_entity_id,
                "--frame-span",
                str(start),
                str(end),
                "--sides",
                *args.sides,
                "--output-root",
                str(p17_root),
                "--agent-interaction-judgment",
                str(judgment_path),
                "--require-camera-mano-contract-aligned",
                "--review-frames",
                *[str(frame) for frame in args.review_frames],
            ],
            label="P17_shared_visible_contact_ownership",
            dry_run=bool(args.dry_run),
        )
    )
    if args.dry_run:
        report = {
            "schema": SCHEMA,
            "status": "dry_run_preflight_ok",
            "claim_scope": "Shared-tail command plan only; P17/P18/P18b were not executed.",
            "preflight": preflight,
            "commands": commands,
            "outputs": {
                "factor_report": str(factor_path),
                "p18_state": str(p18_path),
                "p18b_state": str(p18b_path),
            },
        }
        print(json.dumps(report, indent=2), flush=True)
        return report

    factor = load_json(factor_path, "P17 factor report")
    p17_validation = validate_p17_factor(factor, p17_expected_keys)

    commands.append(
        run_command(
            [
                sys.executable,
                str(p18_script),
                "--case",
                str(args.case),
                "--object-id",
                target_entity_id,
                "--annotations",
                str(annotations_path),
                "--pose-report",
                str(pose_path),
                "--completed-mesh",
                str(pose_mesh),
                "--physical-surface-mesh",
                str(physical_surface),
                "--completion-report",
                str(signed_completion_path or completion_path),
                "--depth-npz",
                str(depth_path),
                "--wilor-root",
                str(wilor_root),
                "--wilor-mano-left",
                str(mano_left),
                "--wilor-mano-right",
                str(mano_right),
                "--output-dir",
                str(p18_root),
                "--start-frame",
                str(start),
                "--end-frame",
                str(end),
                "--sides",
                *args.sides,
                "--factor-report",
                str(factor_path),
                "--optimize-contact-state",
                "--visible-surface-depth-order-term",
                "--gate-translation-with-visible-surface-support",
                "--translation-gate-min-visible-surface-depth-vertices",
                "0",
                "--observed-surface-support-uncertainty-m",
                str(signed_surface_uncertainty_m),
                "--no-optimize-object-translation",
                "--require-active-full-K-mano-contract",
                "--device",
                str(args.device),
            ],
            label="P18_shared_unsigned_mano_interval",
            dry_run=False,
        )
    )
    p18 = load_json(p18_path, "P18 shared state")
    p18_validation = validate_p18(
        p18,
        expected_rows=expected_rows,
        pose_report=pose_path,
        physical_surface=physical_surface,
        signed_expected=signed_geometry_ready,
    )

    commands.append(
        run_command(
            [
                sys.executable,
                str(p18b_script),
                "--contact-state",
                str(p18_path),
                "--joint-source",
                "hawor_npz",
                "--hawor-npz",
                str(hawor_path),
                "--case",
                str(args.case),
                "--object-id",
                str(args.object_id).split(":", 1)[-1],
                "--output",
                str(p18b_path),
                *(["--accept-signed-full-mano"] if signed_geometry_ready else ["--no-accept-signed-full-mano"]),
            ],
            label="P18b_shared_metric_mano_surface_hypothesis",
            dry_run=False,
        )
    )
    p18b = load_json(p18b_path, "P18b shared state")
    p18b_validation = validate_p18b(
        p18b,
        expected_rows=expected_rows,
        p18_state_path=p18_path,
        signed_expected=signed_geometry_ready,
    )

    report = {
        "schema": SCHEMA,
        "status": "ok_shared_p17_p18_p18b_tail",
        "claim_scope": (
            "One pre-branch P17/P18/P18b state using the D14 canonical pose body, the D15 object trajectory, "
            "active-K MANO/depth, an agent visual interaction prior, and an optional backend-neutral signed physical surface. "
            "Generated SAM3D/TRELLIS faces were not consumed."
        ),
        "case": str(args.case),
        "object_id": str(args.object_id).split(":", 1)[-1],
        "frame_span": [start, end],
        "sides": args.sides,
        "preflight": preflight,
        "validations": {
            "P17": p17_validation,
            "P18": p18_validation,
            "P18b": p18b_validation,
        },
        "single_object_pose_authority": {
            "source": "D15 observed-only pose graph",
            "path": str(pose_path),
            "file_sha256": sha256_file(pose_path),
            "pose_rows_value_sha256": preflight["pose"]["pose_rows_value_sha256"],
            "P18_object_translation_optimized": False,
            "max_private_object_translation_delta_m": p18_validation[
                "max_object_translation_delta_m"
            ],
        },
        "physical_surface": surface_report,
        "signed_surface_support_uncertainty_m": signed_surface_uncertainty_m,
        "generated_geometry_consumed": False,
        "signed_geometry_ready": signed_geometry_ready,
        "commands": commands,
        "inputs": {
            "annotations": str(annotations_path),
            "annotations_sha256": sha256_file(annotations_path),
            "pose_report": str(pose_path),
            "pose_report_sha256": sha256_file(pose_path),
            "completion_report": str(completion_path),
            "completion_report_sha256": sha256_file(completion_path),
            "signed_completion_report": str(signed_completion_path) if signed_completion_path is not None else None,
            "signed_completion_report_sha256": sha256_file(signed_completion_path) if signed_completion_path is not None else None,
            "depth_npz": str(depth_path),
            "depth_npz_sha256": sha256_file(depth_path),
            "hawor_npz": str(hawor_path),
            "hawor_npz_sha256": sha256_file(hawor_path),
            "interaction_judgment": str(judgment_path),
            "interaction_judgment_sha256": sha256_file(judgment_path),
        },
        "outputs": {
            "factor_report": str(factor_path),
            "factor_report_sha256": sha256_file(factor_path),
            "p18_state": str(p18_path),
            "p18_state_sha256": sha256_file(p18_path),
            "p18b_state": str(p18b_path),
            "p18b_state_sha256": sha256_file(p18b_path),
            "p18b_state_value_sha256": p18b_validation[
                "temporal_state_value_sha256"
            ],
            "report": str(report_path),
        },
    }
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "factor_report": str(factor_path),
                "p18_state": str(p18_path),
                "p18b_state": str(p18b_path),
                "p18b_surface_sample_points": p18b_validation[
                    "surface_sample_point_count"
                ],
                "signed_geometry_ready": signed_geometry_ready,
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--completion-report", type=Path, required=True)
    parser.add_argument("--signed-completion-report", type=Path, default=None, help="Optional backend-neutral D15b signed completion report")
    parser.add_argument("--depth-npz", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--interaction-judgment", type=Path, required=True)
    parser.add_argument("--wilor-root", type=Path, required=True)
    parser.add_argument("--wilor-mano-left", type=Path, required=True)
    parser.add_argument("--wilor-mano-right", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument(
        "--sides", nargs="+", choices=("left", "right"), default=["left", "right"]
    )
    parser.add_argument(
        "--review-frames", type=int, nargs="*", default=[0, 30, 60, 90, 120, 149]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
