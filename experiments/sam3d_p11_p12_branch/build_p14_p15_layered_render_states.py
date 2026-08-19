#!/usr/bin/env python3
"""Build additive P14/P15 render-state clones for a controlled geometry A/B.

The adapter keeps the observed-only object trajectory, annotation/camera backbone,
MANO constraint payload, and temporal MANO payload byte-for-byte equivalent at the
JSON-value level across branches.  Only the render geometry and branch provenance
differ.  Generated faces remain render-only. Every branch points at one shared observed
render surface and, when independently reconstructed, one backend-neutral shared
signed physical surface.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

SCHEMA = "v19_experimental_p14_p15_layered_render_state_adapter_v1"
STATE_SCHEMA = "v19_experimental_p14_p15_layered_rigid_render_state_v1"
SHARED_BLOCKS = (
    "annotation_backbone",
    "object_pose_trajectory",
    "mano_constraint_state",
    "temporal_mano_state",
    "hidden_volume_validation",
    "projection_contract",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def prepare_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        meshes = [
            item
            for item in geometry.geometry.values()
            if isinstance(item, trimesh.Trimesh) and len(item.vertices) and len(item.faces)
        ]
        if not meshes:
            raise RuntimeError(f"scene has no triangle mesh: {path}")
        geometry = trimesh.util.concatenate(meshes)
    if not isinstance(geometry, trimesh.Trimesh):
        raise RuntimeError(f"unsupported mesh type {type(geometry)}: {path}")
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    faces = np.asarray(geometry.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"invalid mesh vertices {vertices.shape}: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"invalid mesh faces {faces.shape}: {path}")
    return vertices, faces


def geometry_digest(path: Path) -> dict[str, Any]:
    vertices, faces = load_mesh_arrays(path)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices.astype("<f8")).tobytes())
    digest.update(np.ascontiguousarray(faces.astype("<i8")).tobytes())
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "geometry_sha256_f64_i64": digest.hexdigest(),
        "bounds": np.asarray([vertices.min(axis=0), vertices.max(axis=0)], dtype=float).tolist(),
    }


def candidate_by_name(controlled: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in controlled.get("candidates", []) if isinstance(row, dict) and row.get("name") == name]
    if len(rows) != 1:
        raise RuntimeError(f"controlled report candidate {name!r} appears {len(rows)} times")
    return rows[0]


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def validate_shared_temporal_mano(
    source: dict[str, Any],
    *,
    pose_report_path: Path,
    common_collision: Path,
    frame_ids: list[int],
) -> dict[str, Any]:
    block = (
        source.get("temporal_mano_state")
        if isinstance(source.get("temporal_mano_state"), dict)
        else {}
    )
    payload = block.get("payload") if isinstance(block.get("payload"), dict) else None
    if payload is None:
        raise RuntimeError("source state lacks the required shared P18b temporal MANO payload")
    temporal_path = require_file(
        Path(str(block.get("path") or "")), "shared P18b temporal MANO state"
    )
    if value_sha256(load_json(temporal_path)) != value_sha256(payload):
        raise RuntimeError("embedded P18b payload differs from its bound state file")
    rows = [
        row for row in payload.get("per_frame_states") or [] if isinstance(row, dict)
    ]
    expected_keys = {
        (int(frame_idx), side) for frame_idx in frame_ids for side in ("left", "right")
    }
    actual_keys = {(int(row["frame_idx"]), str(row["hand_side"])) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(actual_keys):
        raise RuntimeError(
            f"shared P18b timeline mismatch: rows={len(rows)} unique={len(actual_keys)} expected={len(expected_keys)}"
        )
    p18b_private_object_delta_max = 0.0
    sample_rows = 0
    sample_points = 0
    signed_full_rows = 0
    for row in rows:
        policy = str(row.get("joint_state_policy") or "")
        if "metric_mano_preserved" not in policy and "p18_signed_full_778_mano_accepted" not in policy:
            raise RuntimeError("shared P18b row lacks a recognized metric MANO policy")
        if "p18_signed_full_778_mano_accepted" in policy:
            signed_full_rows += 1
        delta = np.asarray(
            row.get("optimized_object_translation_world_m") or [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise RuntimeError("shared P18b row has invalid object translation delta")
        p18b_private_object_delta_max = max(
            p18b_private_object_delta_max, float(np.linalg.norm(delta))
        )
        samples = np.asarray(
            row.get("contact_surface_vertices_world_sample_m") or [], dtype=np.float64
        )
        if samples.size:
            if samples.ndim != 2 or samples.shape[1] != 3 or not np.isfinite(samples).all():
                raise RuntimeError("shared P18b row has invalid uncertain surface samples")
            sample_rows += 1
            sample_points += int(len(samples))
    if p18b_private_object_delta_max > 1.0e-10:
        raise RuntimeError(
            f"shared P18b contains private object motion up to {p18b_private_object_delta_max} m"
        )

    p18_path = require_file(
        Path(str((payload.get("inputs") or {}).get("contact_state") or "")),
        "raw shared P18 state",
    )
    p18 = load_json(p18_path)
    p18_inputs = p18.get("inputs") if isinstance(p18.get("inputs"), dict) else {}
    p18_parameters = (
        p18.get("parameters") if isinstance(p18.get("parameters"), dict) else {}
    )
    p18_pose = require_file(
        Path(str(p18_inputs.get("pose_report") or "")), "P18 D15 pose authority"
    )
    if not same_file(p18_pose, pose_report_path):
        raise RuntimeError("shared P18 did not consume the source state's D15 pose authority")
    p18_surface = require_file(
        Path(str(p18_inputs.get("physical_surface_mesh") or "")),
        "P18 shared physical surface",
    )
    source_physical = require_file(
        Path(
            str(
                (((source.get("object_geometry") or {}).get("physical_surface") or {}).get("mesh"))
                or common_collision
            )
        ),
        "source-state shared physical surface",
    )
    if geometry_digest(p18_surface)["geometry_sha256_f64_i64"] != geometry_digest(
        source_physical
    )["geometry_sha256_f64_i64"]:
        raise RuntimeError("shared P18 physical surface differs from the source-state physical surface")
    if p18_parameters.get("optimize_object_translation") is not False:
        raise RuntimeError("shared P18 did not explicitly disable object translation")
    raw_rows = [
        row for row in p18.get("per_frame_states") or [] if isinstance(row, dict)
    ]
    if len(raw_rows) != len(expected_keys):
        raise RuntimeError(
            f"raw shared P18 timeline has {len(raw_rows)} rows, expected {len(expected_keys)}"
        )
    p18_private_object_delta_max = 0.0
    for row in raw_rows:
        delta = np.asarray(
            row.get("optimized_object_translation_world_m") or [], dtype=np.float64
        )
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise RuntimeError("raw shared P18 row has invalid object translation delta")
        p18_private_object_delta_max = max(
            p18_private_object_delta_max, float(np.linalg.norm(delta))
        )
    if p18_private_object_delta_max > 1.0e-10:
        raise RuntimeError(
            f"raw shared P18 privately moved the object by {p18_private_object_delta_max} m"
        )
    signed_geometry_ready = bool(
        ((p18.get("physical_surface_contract") or {}).get("signed_geometry_declared_ready") is True)
    )
    if signed_full_rows not in (0, len(rows)):
        raise RuntimeError(
            f"shared P18b partially accepted full MANO on {signed_full_rows}/{len(rows)} rows"
        )
    if not signed_geometry_ready and signed_full_rows:
        raise RuntimeError("unsigned shared P18b contains signed-accepted full MANO rows")
    return {
        "status": "shared_prebranch_P18b_bound_to_D15_and_physical_surface",
        "temporal_state": str(temporal_path),
        "temporal_state_sha256": sha256_file(temporal_path),
        "temporal_state_value_sha256": value_sha256(payload),
        "raw_p18_state": str(p18_path),
        "raw_p18_state_sha256": sha256_file(p18_path),
        "row_count": len(rows),
        "surface_sample_row_count": sample_rows,
        "surface_sample_point_count": sample_points,
        "metric_mano_preserved": not signed_geometry_ready,
        "signed_geometry_ready": signed_geometry_ready,
        "signed_full_mano_accepted": signed_full_rows == len(rows),
        "signed_full_mano_accepted_row_count": signed_full_rows,
        "P18_object_translation_optimized": False,
        "P18_max_private_object_translation_delta_m": p18_private_object_delta_max,
        "P18b_max_private_object_translation_delta_m": p18b_private_object_delta_max,
        "pose_authority": str(pose_report_path),
        "pose_authority_sha256": sha256_file(pose_report_path),
        "physical_surface": str(source_physical),
        "physical_surface_sha256": sha256_file(source_physical),
        "generated_geometry_consumed": False,
    }


def validate_source_state(source: dict[str, Any], source_path: Path, common_collision: Path) -> dict[str, Any]:
    if not str(source.get("status") or "").startswith("ok"):
        raise RuntimeError(f"source render state is not ok: {source_path}")
    pose = source.get("object_pose_trajectory")
    if not isinstance(pose, dict):
        raise RuntimeError("source state lacks object_pose_trajectory")
    rows = pose.get("pose_rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("source state has no pose rows")
    frame_ids = [int(row["frame_idx"]) for row in rows if isinstance(row, dict) and "frame_idx" in row]
    if len(frame_ids) != len(rows) or len(set(frame_ids)) != len(frame_ids):
        raise RuntimeError("source state pose rows have invalid/duplicate frame ids")
    pose_report_path = require_file(Path(str(pose.get("pose_report_path", ""))), "observed-only pose report")
    pose_report = load_json(pose_report_path)
    if pose_report.get("annotation_ready") is not True:
        raise RuntimeError(f"pose report is not annotation-ready: {pose_report_path}")
    temporal_readiness = pose_report.get("temporal_readiness")
    if not isinstance(temporal_readiness, dict) or temporal_readiness.get("ready") is not True:
        raise RuntimeError("pose report lacks a passing temporal coverage/observability/SE(3)-jump gate")
    if int(temporal_readiness.get("accepted_full_timeline_pose_count") or 0) != int(temporal_readiness.get("timeline_frame_count") or -1):
        raise RuntimeError("pose report temporal gate does not cover the exact full timeline")
    rotation_step_gate = (
        temporal_readiness.get("rotation_step_gate")
        if isinstance(temporal_readiness.get("rotation_step_gate"), dict)
        else {}
    )
    if rotation_step_gate.get("gate_passed") is not True:
        raise RuntimeError("pose report rotation-step gate is not explicitly passed")
    if rotation_step_gate.get("conditional_tier_applied") is True:
        if rotation_step_gate.get("acceptance_mode") != "conditional_sparse_underobservable_rotation_tail":
            raise RuntimeError("pose report has malformed conditional rotation-tail mode")
        if rotation_step_gate.get("trajectory_values_modified_or_clipped") is not False:
            raise RuntimeError("conditional rotation-tail trajectory was modified or clipped")
        if rotation_step_gate.get("generated_geometry_pose_evidence_consumed") is not False:
            raise RuntimeError("conditional rotation-tail consumed generated pose evidence")
    if int(pose_report.get("nonpenetration_target_frame_count", -1)) != 0:
        raise RuntimeError("pose trajectory is not observed-only: nonpenetration targets are present")
    completed_mesh = require_file(
        Path(str((pose_report.get("inputs") or {}).get("completed_mesh", ""))),
        "pose-report canonical observed mesh",
    )
    pose_geometry = geometry_digest(completed_mesh)
    collision_geometry = geometry_digest(common_collision)
    if pose_geometry["geometry_sha256_f64_i64"] != collision_geometry["geometry_sha256_f64_i64"]:
        raise RuntimeError(
            "observed-only trajectory canonical mesh does not match the shared collision/observation mesh: "
            f"pose={pose_geometry} collision={collision_geometry}"
        )
    annotation_path = require_file(
        Path(str((source.get("annotation_backbone") or {}).get("path", ""))),
        "annotation backbone",
    )
    annotation = load_json(annotation_path)
    annotation_frames = annotation.get("frames")
    if not isinstance(annotation_frames, list) or len(annotation_frames) != len(frame_ids):
        raise RuntimeError(
            f"annotation/pose frame-count mismatch: annotations={len(annotation_frames or [])} poses={len(frame_ids)}"
        )
    annotation_frame_ids = [
        int(frame.get("frame_idx", pos))
        for pos, frame in enumerate(annotation_frames)
        if isinstance(frame, dict)
    ]
    if sorted(annotation_frame_ids) != sorted(frame_ids):
        raise RuntimeError("annotation and pose frame IDs differ")
    shared_temporal_mano = validate_shared_temporal_mano(
        source,
        pose_report_path=pose_report_path,
        common_collision=common_collision,
        frame_ids=annotation_frame_ids,
    )
    return {
        "pose_report": str(pose_report_path),
        "pose_report_sha256": sha256_file(pose_report_path),
        "pose_frame_count": len(frame_ids),
        "pose_frame_range": [min(frame_ids), max(frame_ids)],
        "pose_nonpenetration_target_frame_count": 0,
        "temporal_readiness": temporal_readiness,
        "annotation_readiness_mode": pose_report.get("annotation_readiness_mode"),
        "rotation_step_acceptance_mode": rotation_step_gate.get("acceptance_mode"),
        "conditional_temporal_uncertainty": pose_report.get("conditional_temporal_uncertainty"),
        "pose_canonical_observed_mesh": str(completed_mesh),
        "pose_canonical_observed_geometry": pose_geometry,
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "annotation_frame_count": len(annotation_frames),
        "shared_temporal_mano": shared_temporal_mano,
    }


def legacy_layers(candidate: dict[str, Any], common_observed: Path) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    outputs = candidate.get("source_neutral_outputs") if isinstance(candidate.get("source_neutral_outputs"), dict) else {}
    completed = require_file(Path(str(outputs.get("pose_hypothesis_mesh", ""))), "legacy pose-hypothesis mesh")
    builder_path = require_file(Path(str(candidate.get("legacy_builder_report", ""))), "legacy builder report")
    builder = load_json(builder_path)
    counts = builder.get("face_label_counts") if isinstance(builder.get("face_label_counts"), dict) else {}
    completed_counts = counts.get("completed_mesh") if isinstance(counts.get("completed_mesh"), dict) else {}
    observed_count = int(completed_counts.get("observed_depth_surface", 0))
    generated_count = int(completed_counts.get("trellis_inferred_hidden_surface", 0))
    vertices, faces = load_mesh_arrays(completed)
    if observed_count <= 0 or generated_count <= 0 or observed_count + generated_count != len(faces):
        raise RuntimeError(
            f"invalid legacy face partition for {candidate.get('name')}: "
            f"observed={observed_count} generated={generated_count} mesh_faces={len(faces)}"
        )
    builder_observed = require_file(
        Path(str((builder.get("outputs") or {}).get("collision_eligible_mesh_labeled", ""))),
        "legacy observed collision mesh",
    )
    if geometry_digest(builder_observed)["geometry_sha256_f64_i64"] != geometry_digest(common_observed)["geometry_sha256_f64_i64"]:
        raise RuntimeError(f"legacy candidate {candidate.get('name')} does not share the canonical observed surface")
    layers = [
        {
            "role": "generated_complete_prior_underlay",
            "mesh": str(completed),
            "face_selection": {"mode": "contiguous_range", "start": observed_count, "stop": len(faces)},
            "source_model": candidate.get("source_model"),
            "integration": "legacy_observed_band_face_deletion",
            "generated_topology_preserved": False,
            "collision_eligible": False,
        },
        {
            "role": "observed_metric_surface_overlay",
            "mesh": str(completed),
            "face_selection": {"mode": "contiguous_range", "start": 0, "stop": observed_count},
            "source": "prediction_side_depth_observed_surface",
            "depth_priority": "observed_object_ownership_override_preserving_mano_depth_winners",
            "collision_eligible": True,
        },
    ]
    details = {
        "completed_mesh": str(completed),
        "completed_mesh_sha256": sha256_file(completed),
        "completed_mesh_vertices": int(len(vertices)),
        "completed_mesh_faces": int(len(faces)),
        "observed_faces": observed_count,
        "generated_retained_faces": generated_count,
        "legacy_builder_report": str(builder_path),
        "legacy_builder_report_sha256": sha256_file(builder_path),
    }
    return layers, completed, details


def write_state(
    source: dict[str, Any],
    output_path: Path,
    *,
    branch_id: str,
    label: str,
    source_model: str,
    integration: str,
    layers: list[dict[str, Any]],
    fallback_mesh: Path,
    geometry_report_path: Path,
    common_collision: Path,
    source_state_path: Path,
) -> dict[str, Any]:
    state = copy.deepcopy(source)
    temporal_payload = (
        ((source.get("temporal_mano_state") or {}).get("payload") or {})
        if isinstance(source.get("temporal_mano_state"), dict)
        else {}
    )
    signed_acceptance = temporal_payload.get("full_mano_acceptance") if isinstance(temporal_payload, dict) else {}
    p18_path = Path(str((temporal_payload.get("inputs") or {}).get("contact_state") or "")) if isinstance(temporal_payload, dict) else Path("")
    p18_payload = load_json(require_file(p18_path, "shared raw P18 state for signed readiness")) if p18_path.is_file() else {}
    signed_geometry_ready = bool(
        ((p18_payload.get("physical_surface_contract") or {}).get("signed_geometry_declared_ready") is True)
    )
    signed_full_mano_accepted = bool(
        isinstance(signed_acceptance, dict) and signed_acceptance.get("accepted") is True
    )
    state["status"] = (
        "ok_experimental_layered_render_prior_shared_signed_mano"
        if signed_full_mano_accepted
        else "ok_experimental_layered_render_prior_physics_quarantined"
    )
    state["method"] = "build_experimental_p14_p15_layered_render_state_adapter"
    state["claim_scope"] = (
        "Full-timeline render-only geometry A/B using one frozen observed-only SE(3) trajectory, "
        "camera/annotation backbone, and one shared source or signed-accepted metric MANO state. "
        "Backend-generated faces are never collision/sign/contact surfaces."
    )
    state["object_label"] = label
    state["created_unix_s"] = time.time()
    inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
    inputs["source_render_state"] = str(source_state_path)
    inputs["completed_mesh"] = str(fallback_mesh)
    inputs["experimental_geometry_report"] = str(geometry_report_path)
    inputs["shared_observed_collision_surface"] = str(common_collision)
    state["inputs"] = inputs
    source_physical_payload = copy.deepcopy(
        ((source.get("object_geometry") or {}).get("physical_surface") or {})
    )
    if not source_physical_payload.get("mesh"):
        source_physical_payload = {
            "mesh": str(common_collision),
            "source": "prediction_side_observed_metric_surface_only",
            "generated_faces_collision_eligible": False,
            "generated_faces_contact_eligible": False,
            "generated_faces_signed_distance_eligible": False,
            "signed_geometry_ready": False,
        }
    if bool(source_physical_payload.get("signed_geometry_ready")) != signed_geometry_ready:
        raise RuntimeError(
            "source physical signed readiness differs from raw P18 signed surface contract"
        )
    if source_physical_payload.get("generated_faces_collision_eligible") is True:
        raise RuntimeError("source physical surface promotes backend-generated collision faces")
    state["object_geometry"] = {
        "state": "experimental_source_neutral_layered_render_prior",
        "completed_mesh_path": str(fallback_mesh),
        "completed_mesh_path_semantics": "single-mesh compatibility fallback_only_layered_renderer_uses_render_layers",
        "mesh_frame": "completed_canonical",
        "source_model": source_model,
        "integration": integration,
        "render_layers_back_to_front": layers,
        "observation_surface_path": str(common_collision),
        "physical_surface": source_physical_payload,
    }
    state["experimental_p14_p15_adapter"] = {
        "schema": STATE_SCHEMA,
        "branch_id": branch_id,
        "source_model": source_model,
        "integration": integration,
        "source_render_state": str(source_state_path),
        "geometry_report": str(geometry_report_path),
        "trajectory_policy": "frozen_observed_only_v19_pose_rows",
        "camera_policy": "frozen_annotation_backbone_camera_state",
        "mano_policy": (
            "shared signed P18 full-778 archive when globally accepted; otherwise source full-778 metric MANO; "
            "P18 surface samples remain visible diagnostics"
        ),
        "temporal_mano_policy": (
            "one prebranch P18b state; signed-accepted runs render its full-778 P18 archive, "
            "otherwise the metric full-MANO body remains the shared annotation source"
        ),
        "inherited_constraint_payload_policy": (
            "D16 constraint payload preserved identically for provenance but not rendered as signed physics; "
            "shared P18b temporal surface hypothesis is rendered without branch recomputation"
        ),
        "render_contact_claim_enabled": False,
        "render_collision_claim_enabled": False,
        "generated_faces_collision_eligible": False,
        "generated_faces_contact_eligible": False,
        "signed_geometry_ready": signed_geometry_ready,
        "signed_full_mano_accepted": signed_full_mano_accepted,
    }
    output_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_state_path = require_file(args.source_render_state, "source observed-only render state")
    dual_state_path = require_file(args.dual_mesh_state, "P13 dual-mesh state")
    dual_report_path = require_file(args.dual_mesh_report, "P13 dual-mesh report")
    controlled_path = require_file(args.controlled_report, "controlled P13 report")
    output_dir = prepare_output(args.output_dir)

    source = load_json(source_state_path)
    dual_state = load_json(dual_state_path)
    dual_report = load_json(dual_report_path)
    controlled = load_json(controlled_path)
    if dual_report.get("status") != "ok" or controlled.get("status") != "ok":
        raise RuntimeError("P13 dual/controlled reports must both be ok")

    render_order = dual_state.get("render_order_back_to_front")
    if not isinstance(render_order, list) or len(render_order) != 2:
        raise RuntimeError("dual mesh state must contain exactly generated + observed render layers")
    dual_generated = require_file(Path(str(render_order[0].get("mesh", ""))), "intact aligned generated prior")
    dual_observed = require_file(Path(str(render_order[1].get("mesh", ""))), "dual observed metric surface")
    physical = dual_state.get("physical_surface") if isinstance(dual_state.get("physical_surface"), dict) else {}
    common_collision = require_file(Path(str(physical.get("mesh", ""))), "shared observed-only collision mesh")
    observed_digest = geometry_digest(dual_observed)
    collision_digest = geometry_digest(common_collision)
    if observed_digest["geometry_sha256_f64_i64"] != collision_digest["geometry_sha256_f64_i64"]:
        raise RuntimeError("dual observation and collision meshes are not geometrically identical")
    if physical.get("generated_faces_collision_eligible") is not False:
        raise RuntimeError("dual mesh generated-face physical quarantine is not explicit")

    source_validation = validate_source_state(source, source_state_path, common_collision)
    sam_candidate = candidate_by_name(controlled, args.sam_candidate)
    trellis_candidate = candidate_by_name(controlled, args.trellis_candidate)

    dual_layers = [
        {
            "role": "generated_complete_prior_underlay",
            "mesh": str(dual_generated),
            "face_selection": {"mode": "all"},
            "source_model": sam_candidate.get("source_model"),
            "integration": "topology_preserving_global_sim3_only",
            "generated_topology_preserved": True,
            "collision_eligible": False,
        },
        {
            "role": "observed_metric_surface_overlay",
            "mesh": str(dual_observed),
            "face_selection": {"mode": "all"},
            "source": "prediction_side_depth_observed_surface",
            "depth_priority": "observed_object_ownership_override_preserving_mano_depth_winners",
            "collision_eligible": True,
        },
    ]
    sam_legacy_layers, sam_legacy_mesh, sam_legacy_details = legacy_layers(sam_candidate, common_collision)
    trellis_layers, trellis_mesh, trellis_details = legacy_layers(trellis_candidate, common_collision)

    branch_specs = [
        {
            "branch_id": "sam3d_owned_dual_mesh",
            "label": "SAM3D owned-mask | intact dual mesh",
            "source_model": str(sam_candidate.get("source_model")),
            "integration": "topology_preserving_dual_mesh",
            "layers": dual_layers,
            "fallback_mesh": dual_generated,
            "geometry_report": dual_report_path,
            "details": {
                "generated_mesh": str(dual_generated),
                "generated_mesh_sha256": sha256_file(dual_generated),
                "generated_geometry": geometry_digest(dual_generated),
                "observed_geometry": observed_digest,
                "raw_generated_face_deletion_applied": False,
            },
        },
        {
            "branch_id": "sam3d_owned_legacy_cut",
            "label": "SAM3D owned-mask | legacy cut",
            "source_model": str(sam_candidate.get("source_model")),
            "integration": "legacy_observed_band_face_deletion",
            "layers": sam_legacy_layers,
            "fallback_mesh": sam_legacy_mesh,
            "geometry_report": Path(str(sam_candidate["legacy_builder_report"])),
            "details": sam_legacy_details,
        },
        {
            "branch_id": "trellis_frozen_legacy_cut",
            "label": "TRELLIS frozen | legacy cut",
            "source_model": str(trellis_candidate.get("source_model")),
            "integration": "legacy_observed_band_face_deletion",
            "layers": trellis_layers,
            "fallback_mesh": trellis_mesh,
            "geometry_report": Path(str(trellis_candidate["legacy_builder_report"])),
            "details": trellis_details,
        },
    ]

    states: list[dict[str, Any]] = []
    loaded_states: list[dict[str, Any]] = []
    for spec in branch_specs:
        branch_dir = output_dir / str(spec["branch_id"])
        branch_dir.mkdir(parents=True, exist_ok=False)
        state_path = branch_dir / "experimental_layered_render_state.json"
        state = write_state(
            source,
            state_path,
            branch_id=str(spec["branch_id"]),
            label=str(spec["label"]),
            source_model=str(spec["source_model"]),
            integration=str(spec["integration"]),
            layers=list(spec["layers"]),
            fallback_mesh=Path(spec["fallback_mesh"]),
            geometry_report_path=Path(spec["geometry_report"]),
            common_collision=common_collision,
            source_state_path=source_state_path,
        )
        loaded_states.append(state)
        states.append(
            {
                "branch_id": spec["branch_id"],
                "source_model": spec["source_model"],
                "integration": spec["integration"],
                "state_path": str(state_path),
                "state_sha256": sha256_file(state_path),
                "render_layers": spec["layers"],
                "geometry_details": spec["details"],
            }
        )

    shared_hashes: dict[str, str] = {}
    for block in SHARED_BLOCKS:
        hashes = [value_sha256(state.get(block)) for state in loaded_states]
        if len(set(hashes)) != 1:
            raise RuntimeError(f"shared state block differs across branches: {block} -> {hashes}")
        source_hash = value_sha256(source.get(block))
        if hashes[0] != source_hash:
            raise RuntimeError(f"shared state block changed relative to source: {block}")
        shared_hashes[block] = hashes[0]

    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "build_experimental_p14_p15_layered_render_state_adapter",
        "claim_scope": (
            "Controlled full-timeline render-state clones. Geometry is the sole branch variable; "
            "observed-only pose, camera/annotations, MANO payloads, and projection contract are identical. "
            "Generated geometry is render-only and cannot support signed contact/collision claims."
        ),
        "inputs": {
            "source_render_state": str(source_state_path),
            "source_render_state_sha256": sha256_file(source_state_path),
            "dual_mesh_state": str(dual_state_path),
            "dual_mesh_state_sha256": sha256_file(dual_state_path),
            "dual_mesh_report": str(dual_report_path),
            "controlled_report": str(controlled_path),
        },
        "source_validation": source_validation,
        "shared_state_value_sha256": shared_hashes,
        "shared_physical_surface": {
            "path": str(
                (((source.get("object_geometry") or {}).get("physical_surface") or {}).get("mesh"))
                or common_collision
            ),
            "file_sha256": sha256_file(
                require_file(
                    Path(str((((source.get("object_geometry") or {}).get("physical_surface") or {}).get("mesh")) or common_collision)),
                    "shared source physical surface",
                )
            ),
            "geometry": geometry_digest(
                require_file(
                    Path(str((((source.get("object_geometry") or {}).get("physical_surface") or {}).get("mesh")) or common_collision)),
                    "shared source physical surface",
                )
            ),
            "source": (((source.get("object_geometry") or {}).get("physical_surface") or {}).get("source")) or "prediction_side_observed_metric_surface_only",
            "generated_faces_collision_eligible": False,
            "generated_faces_contact_eligible": False,
            "signed_geometry_ready": bool(source_validation["shared_temporal_mano"].get("signed_geometry_ready")),
        },
        "branch_variable_policy": {
            "allowed_to_differ": [
                "object_geometry.render_layers_back_to_front",
                "object_geometry.completed_mesh_path compatibility fallback",
                "object_label",
                "experimental branch provenance",
            ],
            "must_be_identical": list(SHARED_BLOCKS),
            "inherited_geometry_dependent_constraint_payload_rendered": False,
            "shared_prebranch_temporal_mano_surface_hypothesis_rendered": True,
            "contact_or_collision_recomputed": False,
        },
        "branches": states,
    }
    report_path = output_dir / "p14_p15_layered_render_state_adapter_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "branches": [{k: row[k] for k in ("branch_id", "state_path")} for row in states],
        "pose_frame_count": source_validation["pose_frame_count"],
        "shared_collision_geometry": collision_digest,
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-render-state", type=Path, required=True)
    parser.add_argument("--dual-mesh-state", type=Path, required=True)
    parser.add_argument("--dual-mesh-report", type=Path, required=True)
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-candidate", default="sam3d_new_object_owned_mask")
    parser.add_argument("--trellis-candidate", default="trellis_frozen")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
