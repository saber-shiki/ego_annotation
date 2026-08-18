#!/usr/bin/env python3
"""Run the shared HOT3D P17 -> P18 -> P18b tail exactly once.

This orchestrator is deliberately upstream of the SAM3D/TRELLIS render split. It
accepts one observed-only D14 surface and one D15 object trajectory, builds one
P17 factor report from an agent-authored interaction judgment, runs one unsigned
P18 MANO interval candidate with object translation disabled, and builds one P18b
metric-MANO-preserved surface hypothesis. Generated backend geometry is not an
input to this script.
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
) -> tuple[Path, dict[str, Any]]:
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
    return physical, {
        "status": "shared_observed_only_unsigned_physical_surface",
        **physical_summary,
        "pose_hypothesis_same_file": True,
        "d15_canonical_geometry_sha256_match": True,
        "generated_faces_pose_eligible": False,
        "generated_faces_contact_eligible": False,
        "generated_faces_collision_eligible": False,
        "signed_geometry_ready": False,
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
    state: dict[str, Any], *, expected_rows: int, pose_report: Path, physical_surface: Path
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
    signed_active = any(
        str(row.get("signed_object_surface_factor_state") or "").startswith("active")
        for row in rows
    )
    if signed_active:
        raise RuntimeError("P18 activated signed geometry on the unsigned observed surface")
    return {
        "status": "shared_unsigned_mano_candidate_object_pose_frozen",
        "row_count": len(rows),
        "max_object_translation_delta_m": max_delta,
        "signed_object_surface_factor_active": False,
        "pose_report_sha256": sha256_file(pose_report),
        "physical_surface_sha256": sha256_file(physical_surface),
    }


def validate_p18b(
    state: dict[str, Any], *, expected_rows: int, p18_state_path: Path
) -> dict[str, Any]:
    rows = [row for row in state.get("per_frame_states") or [] if isinstance(row, dict)]
    if len(rows) != expected_rows:
        raise RuntimeError(f"P18b state rows {len(rows)} != expected {expected_rows}")
    sample_rows = 0
    sample_points = 0
    for row in rows:
        policy = str(row.get("joint_state_policy") or "")
        if "metric_mano_preserved" not in policy:
            raise RuntimeError("P18b row does not preserve metric MANO")
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
    return {
        "status": "metric_mano_preserved_with_shared_uncertain_surface_hypothesis",
        "row_count": len(rows),
        "rows_with_surface_samples": sample_rows,
        "surface_sample_point_count": sample_points,
        "temporal_state_value_sha256": value_sha256(state),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations_path = require_file(args.annotations, "P09 annotations")
    pose_path = require_file(args.pose_report, "D15 pose graph")
    completion_path = require_file(args.completion_report, "D14 completion report")
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
    physical_surface, surface_report = validate_observed_surface(completion, pose)
    preflight["physical_surface"] = surface_report
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
                str(physical_surface),
                "--physical-surface-mesh",
                str(physical_surface),
                "--completion-report",
                str(completion_path),
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
            ],
            label="P18b_shared_metric_mano_surface_hypothesis",
            dry_run=False,
        )
    )
    p18b = load_json(p18b_path, "P18b shared state")
    p18b_validation = validate_p18b(
        p18b, expected_rows=expected_rows, p18_state_path=p18_path
    )

    report = {
        "schema": SCHEMA,
        "status": "ok_shared_p17_p18_p18b_tail",
        "claim_scope": (
            "One pre-branch P17/P18/P18b state using only the D14 observed physical surface, "
            "the D15 object trajectory, active-K MANO/depth, and an agent visual interaction prior. "
            "Generated SAM3D/TRELLIS faces were not consumed; signed contact/nonpenetration remain unavailable."
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
        "generated_geometry_consumed": False,
        "signed_geometry_ready": False,
        "commands": commands,
        "inputs": {
            "annotations": str(annotations_path),
            "annotations_sha256": sha256_file(annotations_path),
            "pose_report": str(pose_path),
            "pose_report_sha256": sha256_file(pose_path),
            "completion_report": str(completion_path),
            "completion_report_sha256": sha256_file(completion_path),
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
