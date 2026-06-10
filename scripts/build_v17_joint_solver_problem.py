#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATUS = "v17_joint_problem_spec_incomplete_not_solver"
CLAIM = (
    "This artifact materializes the V3-required full joint hand-object-camera-depth-contact state. "
    "It is a problem contract and gap audit, not an optimizer and not annotation closure."
)


@dataclass(frozen=True)
class CaseInputs:
    case: str
    manifest: Path
    object_roster: Path
    multi_object_timeline: Path
    visible_surface_report: Path
    geometry_state_report: Path
    object_track_dataset_summary: Path
    object_material_track_summary: Path
    object_material_motion_state_summary: Path
    object_material_pose_candidate_summary: Path
    object_material_surface_replay_summary: Path
    multi_object_contact_evidence_summary: Path
    geometry_source_audit_report: Path
    object_geometry_hypothesis_state_report: Path
    object_geometry_factor_problem_report: Path
    geometry_reconstruction_jobs_report: Path
    geometry_reconstruction_results_report: Path
    depth_contact_consistency_audit_report: Path
    sparse_report: Path
    contact_mode_report: Path
    mesh_metadata: Path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite number") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be a finite number")
    return out


def optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return require_int(value, label)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def case_inputs(
    case_row: dict[str, Any],
    measurement_store_root: Path,
    multi_object_timeline_root: Path,
    visible_surface_root: Path,
    geometry_state_root: Path,
    object_track_dataset_root: Path,
    object_material_track_root: Path,
    object_material_motion_state_root: Path,
    object_material_pose_candidate_root: Path,
    object_material_surface_replay_root: Path,
    multi_object_contact_evidence_root: Path,
    geometry_source_audit_root: Path,
    object_geometry_hypothesis_state_root: Path,
    object_geometry_factor_problem_root: Path,
    geometry_reconstruction_jobs_root: Path,
    geometry_reconstruction_results_root: Path,
    depth_contact_consistency_audit_root: Path,
    sparse_graph_root: Path,
    contact_mode_graph_root: Path,
) -> CaseInputs:
    case = require_str(case_row.get("case"), "measurement summary case")
    manifest = existing_path(
        measurement_store_root / case / "v17_measurement_manifest.json",
        f"{case} measurement manifest",
    )
    object_roster = existing_path(
        measurement_store_root / case / "object_roster_v17.json",
        f"{case} object roster",
    )
    multi_object_timeline = existing_path(
        multi_object_timeline_root / case / "v17_multi_object_timeline.json",
        f"{case} multi-object timeline",
    )
    visible_surface_report = existing_path(
        visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
        f"{case} multi-object visible-surface report",
    )
    geometry_state_report = existing_path(
        geometry_state_root / case / "v17_multi_object_geometry_state_report.json",
        f"{case} multi-object geometry-state report",
    )
    object_track_dataset_summary = existing_path(
        object_track_dataset_root / case / "v17_object_track_dataset_summary.json",
        f"{case} object-track dataset summary",
    )
    object_material_track_summary = existing_path(
        object_material_track_root / case / "v17_object_material_track_summary.json",
        f"{case} object material-track summary",
    )
    object_material_motion_state_summary = existing_path(
        object_material_motion_state_root / case / "v17_object_material_motion_state_report.json",
        f"{case} object material-motion state report",
    )
    object_material_pose_candidate_summary = existing_path(
        object_material_pose_candidate_root / case / "v17_object_material_pose_candidate_report.json",
        f"{case} object material-pose candidate report",
    )
    object_material_surface_replay_summary = existing_path(
        object_material_surface_replay_root / case / "v17_object_material_surface_replay_report.json",
        f"{case} object material-surface replay report",
    )
    multi_object_contact_evidence_summary = existing_path(
        multi_object_contact_evidence_root / case / "v17_multi_object_contact_evidence_report.json",
        f"{case} multi-object contact evidence report",
    )
    geometry_source_audit_report = existing_path(
        geometry_source_audit_root / case / "v17_geometry_source_audit_report.json",
        f"{case} geometry-source audit report",
    )
    object_geometry_hypothesis_state_report = existing_path(
        object_geometry_hypothesis_state_root / case / "v17_object_geometry_hypothesis_state_report.json",
        f"{case} object geometry hypothesis-state report",
    )
    object_geometry_factor_problem_report = existing_path(
        object_geometry_factor_problem_root / case / "v17_object_geometry_factor_problem.json",
        f"{case} object geometry factor-problem report",
    )
    geometry_reconstruction_jobs_report = existing_path(
        geometry_reconstruction_jobs_root / case / "v17_geometry_reconstruction_jobs_report.json",
        f"{case} geometry reconstruction jobs report",
    )
    geometry_reconstruction_results_report = existing_path(
        geometry_reconstruction_results_root / case / "v17_geometry_reconstruction_results_report.json",
        f"{case} geometry reconstruction results report",
    )
    depth_contact_consistency_audit_report = existing_path(
        depth_contact_consistency_audit_root / case / "v17_depth_contact_consistency_audit_report.json",
        f"{case} depth-contact consistency audit report",
    )
    sparse_report = existing_path(
        sparse_graph_root / case / "v17_full_timeline_factor_graph_report.json",
        f"{case} sparse graph report",
    )
    contact_mode_report = existing_path(
        contact_mode_graph_root / case / "v17_contact_mode_graph_report.json",
        f"{case} contact-mode report",
    )
    mesh_metadata = existing_path(
        sparse_graph_root / case / "object_meshes_v17_full_timeline_graph.npz.metadata.json",
        f"{case} corrected mesh metadata",
    )
    return CaseInputs(
        case=case,
        manifest=manifest,
        object_roster=object_roster,
        multi_object_timeline=multi_object_timeline,
        visible_surface_report=visible_surface_report,
        geometry_state_report=geometry_state_report,
        object_track_dataset_summary=object_track_dataset_summary,
        object_material_track_summary=object_material_track_summary,
        object_material_motion_state_summary=object_material_motion_state_summary,
        object_material_pose_candidate_summary=object_material_pose_candidate_summary,
        object_material_surface_replay_summary=object_material_surface_replay_summary,
        multi_object_contact_evidence_summary=multi_object_contact_evidence_summary,
        geometry_source_audit_report=geometry_source_audit_report,
        object_geometry_hypothesis_state_report=object_geometry_hypothesis_state_report,
        object_geometry_factor_problem_report=object_geometry_factor_problem_report,
        geometry_reconstruction_jobs_report=geometry_reconstruction_jobs_report,
        geometry_reconstruction_results_report=geometry_reconstruction_results_report,
        depth_contact_consistency_audit_report=depth_contact_consistency_audit_report,
        sparse_report=sparse_report,
        contact_mode_report=contact_mode_report,
        mesh_metadata=mesh_metadata,
    )


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": payload.get("status"),
        "method": payload.get("method"),
    }


def active_frame_count(row: dict[str, Any]) -> int:
    value = optional_int(row.get("active_frame_count"), f"{row.get('object_id')} active_frame_count")
    return int(value) if value is not None and value > 0 else 0


def active_vlm_roster_rows(roster: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(roster):
        row = require_dict(item, f"object roster row {i}")
        if row.get("source") == "vlm_object_plan" and active_frame_count(row) > 0:
            rows.append(row)
    return rows


def roster_audit(roster: list[Any]) -> dict[str, Any]:
    rows = [require_dict(item, f"object roster row {i}") for i, item in enumerate(roster)]
    vlm_rows = active_vlm_roster_rows(roster)
    ambiguous = [
        require_str(row.get("object_id"), "ambiguous object_id")
        for row in rows
        if row.get("expected_coverage_status") == "ambiguous"
    ]
    covered_aliases = [
        require_str(row.get("object_id"), "covered alias object_id")
        for row in rows
        if row.get("role_status") == "covered_by_vlm_plan"
    ]
    return {
        "roster_row_count": len(rows),
        "active_vlm_object_count": len(vlm_rows),
        "active_vlm_object_frame_rows": sum(active_frame_count(row) for row in vlm_rows),
        "active_vlm_object_ids": [
            require_str(row.get("object_id"), "active object_id") for row in vlm_rows
        ],
        "covered_alias_object_ids": covered_aliases,
        "ambiguous_expected_object_ids": ambiguous,
    }


def measurement_counts(manifest: dict[str, Any]) -> dict[str, int]:
    raw = require_dict(manifest.get("measurement_counts"), "measurement_counts")
    out: dict[str, int] = {}
    for key, value in raw.items():
        out[str(key)] = require_int(value, f"measurement_counts.{key}")
    return out


def graph_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_count": require_int(report.get("frame_count"), "sparse report frame_count"),
        "object_variable_frames": require_int(
            report.get("object_variable_frames"), "sparse report object_variable_frames"
        ),
        "hand_ray_shift_variables": require_int(
            report.get("hand_variable_count"), "sparse report hand_variable_count"
        ),
        "scalar_variable_count": require_int(report.get("variable_count"), "sparse report variable_count"),
        "contact_factor_count": require_int(
            report.get("contact_factor_count"), "sparse report contact_factor_count"
        ),
        "linearized_contact_correspondences": require_int(
            report.get("linearized_contact_correspondences"),
            "sparse report linearized_contact_correspondences",
        ),
        "contact_factor_complete": bool(report.get("contact_factor_complete") is True),
        "status": report.get("status"),
        "solver_completeness": report.get("solver_completeness"),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "multi_object_timeline_ready": bool(report.get("multi_object_timeline_ready") is True),
    }


def contact_mode_counts(report: dict[str, Any]) -> dict[str, Any]:
    rows = require_list(report.get("rows"), "contact-mode rows")
    return {
        "row_count": len(rows),
        "frame_count": require_int(report.get("frame_count"), "contact-mode frame_count"),
        "contact_mode_count": require_int(
            report.get("contact_mode_count"), "contact-mode contact_mode_count"
        ),
        "contact_factor_ready_count": require_int(
            report.get("contact_factor_ready_count"), "contact-mode contact_factor_ready_count"
        ),
        "active_observation_count": require_int(
            report.get("active_observation_count"), "contact-mode active_observation_count"
        ),
        "unobserved_row_count": require_int(
            report.get("unobserved_row_count"), "contact-mode unobserved_row_count"
        ),
        "status": report.get("status"),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
    }


def multi_object_timeline_counts(timeline: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": timeline.get("status"),
        "frame_count": require_int(timeline.get("frame_count"), "multi-object timeline frame_count"),
        "object_count": require_int(timeline.get("object_count"), "multi-object timeline object_count"),
        "object_frame_rows": require_int(
            timeline.get("object_frame_rows"), "multi-object timeline object_frame_rows"
        ),
        "visible_mask_frame_rows": require_int(
            timeline.get("visible_mask_frame_rows"), "multi-object timeline visible_mask_frame_rows"
        ),
        "active_without_visible_mask_frame_rows": require_int(
            timeline.get("active_without_visible_mask_frame_rows"),
            "multi-object timeline active_without_visible_mask_frame_rows",
        ),
        "multi_object_timeline_ready": bool(timeline.get("multi_object_timeline_ready") is True),
        "object_geometry_complete": bool(timeline.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(timeline.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(timeline.get("annotation_ready") is True),
    }


def visible_surface_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "visible-surface frame_count"),
        "visible_object_frame_rows": require_int(
            report.get("visible_object_frame_rows"), "visible-surface visible_object_frame_rows"
        ),
        "surface_frame_rows": require_int(
            report.get("surface_frame_rows"), "visible-surface surface_frame_rows"
        ),
        "rejected_visible_object_frame_rows": require_int(
            report.get("rejected_visible_object_frame_rows"),
            "visible-surface rejected_visible_object_frame_rows",
        ),
        "depth_frame_count": require_int(report.get("depth_frame_count"), "visible-surface depth_frame_count"),
        "rejection_reason_counts": require_dict(
            report.get("rejection_reason_counts"), "visible-surface rejection_reason_counts"
        ),
        "mesh_archive": require_str(report.get("mesh_archive"), "visible-surface mesh_archive"),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def geometry_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "geometry-state frame_count"),
        "object_count": require_int(report.get("object_count"), "geometry-state object_count"),
        "surface_frame_rows": require_int(report.get("surface_frame_rows"), "geometry-state surface_frame_rows"),
        "rejected_visible_object_frame_rows": require_int(
            report.get("rejected_visible_object_frame_rows"),
            "geometry-state rejected_visible_object_frame_rows",
        ),
        "visible_surface_envelope_candidate_count": require_int(
            report.get("visible_surface_envelope_candidate_count"),
            "geometry-state visible_surface_envelope_candidate_count",
        ),
        "persistent_visible_surface_candidate_count": require_int(
            report.get("persistent_visible_surface_candidate_count"),
            "geometry-state persistent_visible_surface_candidate_count",
        ),
        "rigid_pose_candidate_count": require_int(
            report.get("rigid_pose_candidate_count"), "geometry-state rigid_pose_candidate_count"
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def object_track_dataset_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "object_count": require_int(report.get("object_count"), "object-track dataset object_count"),
        "exported_object_count": require_int(
            report.get("exported_object_count"), "object-track dataset exported_object_count"
        ),
        "total_exported_frames": require_int(
            report.get("total_exported_frames"), "object-track dataset total_exported_frames"
        ),
        "total_rejected_frames": require_int(
            report.get("total_rejected_frames"), "object-track dataset total_rejected_frames"
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def object_material_track_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "dataset_exported_object_count": require_int(
            report.get("dataset_exported_object_count"),
            "object material-track dataset_exported_object_count",
        ),
        "dataset_exported_frames": require_int(
            report.get("dataset_exported_frames"), "object material-track dataset_exported_frames"
        ),
        "material_track_window_count": require_int(
            report.get("material_track_window_count"),
            "object material-track material_track_window_count",
        ),
        "material_tracked_object_count": require_int(
            report.get("material_tracked_object_count"),
            "object material-track material_tracked_object_count",
        ),
        "rigid_motion_ready_window_count": require_int(
            report.get("rigid_motion_ready_window_count"),
            "object material-track rigid_motion_ready_window_count",
        ),
        "rigid_factor_ready_pair_count": require_int(
            report.get("rigid_factor_ready_pair_count"),
            "object material-track rigid_factor_ready_pair_count",
        ),
        "exported_object_ids_without_material_tracks": require_list(
            report.get("exported_object_ids_without_material_tracks"),
            "object material-track exported_object_ids_without_material_tracks",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def object_material_motion_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "material_track_window_count": require_int(
            report.get("material_track_window_count"),
            "object material-motion material_track_window_count",
        ),
        "material_tracked_object_count": require_int(
            report.get("material_tracked_object_count"),
            "object material-motion material_tracked_object_count",
        ),
        "rigid_factor_ready_pair_count": require_int(
            report.get("rigid_factor_ready_pair_count"),
            "object material-motion rigid_factor_ready_pair_count",
        ),
        "window_rigid_motion_candidate_count": require_int(
            report.get("window_rigid_motion_candidate_count"),
            "object material-motion window_rigid_motion_candidate_count",
        ),
        "persistent_window_motion_candidate_count": require_int(
            report.get("persistent_window_motion_candidate_count"),
            "object material-motion persistent_window_motion_candidate_count",
        ),
        "local_adjacent_material_motion_window_count": require_int(
            report.get("local_adjacent_material_motion_window_count"),
            "object material-motion local_adjacent_material_motion_window_count",
        ),
        "noncandidate_local_adjacent_material_motion_window_count": require_int(
            report.get("noncandidate_local_adjacent_material_motion_window_count"),
            "object material-motion noncandidate_local_adjacent_material_motion_window_count",
        ),
        "no_ready_material_motion_window_count": require_int(
            report.get("no_ready_material_motion_window_count"),
            "object material-motion no_ready_material_motion_window_count",
        ),
        "candidate_window_ids": require_list(
            report.get("candidate_window_ids"),
            "object material-motion candidate_window_ids",
        ),
        "max_candidate_segment_pairs": require_int(
            report.get("max_candidate_segment_pairs"),
            "object material-motion max_candidate_segment_pairs",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "rigid_pose_requirement_met": bool(report.get("rigid_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def object_material_pose_candidate_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "material_track_window_count": require_int(
            report.get("material_track_window_count"),
            "object material-pose material_track_window_count",
        ),
        "persistent_window_motion_candidate_count": require_int(
            report.get("persistent_window_motion_candidate_count"),
            "object material-pose persistent_window_motion_candidate_count",
        ),
        "partial_material_pose_candidate_segment_count": require_int(
            report.get("partial_material_pose_candidate_segment_count"),
            "object material-pose partial_material_pose_candidate_segment_count",
        ),
        "partial_material_pose_candidate_ready_segment_count": require_int(
            report.get("partial_material_pose_candidate_ready_segment_count"),
            "object material-pose partial_material_pose_candidate_ready_segment_count",
        ),
        "candidate_window_ids": require_list(
            report.get("candidate_window_ids"),
            "object material-pose candidate_window_ids",
        ),
        "candidate_segment_ids": require_list(
            report.get("candidate_segment_ids"),
            "object material-pose candidate_segment_ids",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "rigid_pose_requirement_met": bool(report.get("rigid_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def object_material_surface_replay_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "partial_material_pose_candidate_segment_count": require_int(
            report.get("partial_material_pose_candidate_segment_count"),
            "object material-surface partial_material_pose_candidate_segment_count",
        ),
        "partial_material_pose_candidate_ready_segment_count": require_int(
            report.get("partial_material_pose_candidate_ready_segment_count"),
            "object material-surface partial_material_pose_candidate_ready_segment_count",
        ),
        "partial_visible_surface_replay_candidate_count": require_int(
            report.get("partial_visible_surface_replay_candidate_count"),
            "object material-surface partial_visible_surface_replay_candidate_count",
        ),
        "partial_visible_surface_replay_ready_count": require_int(
            report.get("partial_visible_surface_replay_ready_count"),
            "object material-surface partial_visible_surface_replay_ready_count",
        ),
        "ready_candidate_ids": require_list(
            report.get("ready_candidate_ids"),
            "object material-surface ready_candidate_ids",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "rigid_pose_requirement_met": bool(report.get("rigid_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def multi_object_contact_evidence_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "multi-object contact frame_count"),
        "object_frame_rows": require_int(
            report.get("object_frame_rows"), "multi-object contact object_frame_rows"
        ),
        "hand_object_rows": require_int(
            report.get("hand_object_rows"), "multi-object contact hand_object_rows"
        ),
        "measured_distance_rows": require_int(
            report.get("measured_distance_rows"), "multi-object contact measured_distance_rows"
        ),
        "unobserved_rows": require_int(
            report.get("unobserved_rows"), "multi-object contact unobserved_rows"
        ),
        "visible_surface_distance_candidate_rows": require_int(
            report.get("visible_surface_distance_candidate_rows"),
            "multi-object contact visible_surface_distance_candidate_rows",
        ),
        "contact_distance_candidate_rows": require_int(
            report.get("contact_distance_candidate_rows"),
            "multi-object contact contact_distance_candidate_rows",
        ),
        "contact_factor_ready_rows": require_int(
            report.get("contact_factor_ready_rows"),
            "multi-object contact contact_factor_ready_rows",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def geometry_source_audit_counts(report: dict[str, Any]) -> dict[str, Any]:
    geometry = require_dict(report.get("geometry_source_counts"), "geometry-source audit geometry_source_counts")
    contact = require_dict(report.get("contact_source_counts"), "geometry-source audit contact_source_counts")
    findings = require_dict(report.get("source_compatibility_findings"), "geometry-source audit findings")
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "geometry-source audit frame_count"),
        "multi_object_visible_surface_rows": require_int(
            geometry.get("multi_object_visible_surface_rows"),
            "geometry-source audit multi_object_visible_surface_rows",
        ),
        "multi_object_visible_surface_rejected_rows": require_int(
            geometry.get("multi_object_visible_surface_rejected_rows"),
            "geometry-source audit multi_object_visible_surface_rejected_rows",
        ),
        "legacy_single_stream_object_variable_frames": require_int(
            geometry.get("legacy_single_stream_object_variable_frames"),
            "geometry-source audit legacy_single_stream_object_variable_frames",
        ),
        "legacy_single_stream_mesh_frames": require_int(
            geometry.get("legacy_single_stream_mesh_frames"),
            "geometry-source audit legacy_single_stream_mesh_frames",
        ),
        "legacy_single_stream_missing_mesh_frame_count": require_int(
            geometry.get("legacy_single_stream_missing_mesh_frame_count"),
            "geometry-source audit legacy_single_stream_missing_mesh_frame_count",
        ),
        "local_contact_patch_state_rows": require_int(
            geometry.get("local_contact_patch_state_rows"),
            "geometry-source audit local_contact_patch_state_rows",
        ),
        "accepted_local_contact_patch_state_rows": require_int(
            geometry.get("accepted_local_contact_patch_state_rows"),
            "geometry-source audit accepted_local_contact_patch_state_rows",
        ),
        "partial_visible_surface_replay_candidate_count": require_int(
            geometry.get("partial_visible_surface_replay_candidate_count"),
            "geometry-source audit partial_visible_surface_replay_candidate_count",
        ),
        "partial_visible_surface_replay_ready_count": require_int(
            geometry.get("partial_visible_surface_replay_ready_count"),
            "geometry-source audit partial_visible_surface_replay_ready_count",
        ),
        "contact_mode_factor_ready_rows": require_int(
            contact.get("contact_mode_factor_ready_rows"),
            "geometry-source audit contact_mode_factor_ready_rows",
        ),
        "contact_mode_factor_ready_rows_with_selected_measurement": require_int(
            contact.get("contact_mode_factor_ready_rows_with_selected_measurement"),
            "geometry-source audit contact_mode_factor_ready_rows_with_selected_measurement",
        ),
        "contact_mode_factor_ready_rows_without_selected_measurement": require_int(
            contact.get("contact_mode_factor_ready_rows_without_selected_measurement"),
            "geometry-source audit contact_mode_factor_ready_rows_without_selected_measurement",
        ),
        "contact_mode_ready_rows_with_same_frame_side_multi_object_measurement": require_int(
            contact.get("contact_mode_ready_rows_with_same_frame_side_multi_object_measurement"),
            "geometry-source audit contact_mode_ready_rows_with_same_frame_side_multi_object_measurement",
        ),
        "contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate": require_int(
            contact.get("contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate"),
            "geometry-source audit contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate",
        ),
        "multi_object_hand_object_rows": require_int(
            contact.get("multi_object_hand_object_rows"),
            "geometry-source audit multi_object_hand_object_rows",
        ),
        "multi_object_measured_distance_rows": require_int(
            contact.get("multi_object_measured_distance_rows"),
            "geometry-source audit multi_object_measured_distance_rows",
        ),
        "multi_object_unobserved_rows": require_int(
            contact.get("multi_object_unobserved_rows"),
            "geometry-source audit multi_object_unobserved_rows",
        ),
        "multi_object_visible_surface_distance_candidate_rows": require_int(
            contact.get("multi_object_visible_surface_distance_candidate_rows"),
            "geometry-source audit multi_object_visible_surface_distance_candidate_rows",
        ),
        "multi_object_contact_factor_ready_rows": require_int(
            contact.get("multi_object_contact_factor_ready_rows"),
            "geometry-source audit multi_object_contact_factor_ready_rows",
        ),
        "selected_measurement_audit_counts": require_dict(
            contact.get("selected_measurement_audit_counts"),
            "geometry-source audit selected_measurement_audit_counts",
        ),
        "local_patch_visible_surface_conflict_count": len(
            [
                row
                for row in require_list(
                    report.get("local_patch_visible_surface_conflicts"),
                    "geometry-source audit local_patch_visible_surface_conflicts",
                )
                if require_dict(row, "local patch conflict").get("source_conflict") is True
            ]
        ),
        "source_incompatibility_count": require_int(
            report.get("source_incompatibility_count"),
            "geometry-source audit source_incompatibility_count",
        ),
        "legacy_contact_factors_supported_by_multi_object_visible_surface_contact_rows": bool(
            findings.get("legacy_contact_factors_supported_by_multi_object_visible_surface_contact_rows") is True
        ),
        "legacy_contact_factors_have_any_same_frame_side_visible_surface_candidate": bool(
            findings.get("legacy_contact_factors_have_any_same_frame_side_visible_surface_candidate") is True
        ),
        "accepted_local_patches_conflict_with_multi_object_visible_surface_distance": bool(
            findings.get("accepted_local_patches_conflict_with_multi_object_visible_surface_distance") is True
        ),
        "partial_material_pose_replay_is_complete_object_geometry": bool(
            findings.get("partial_material_pose_replay_is_complete_object_geometry") is True
        ),
        "unified_object_geometry_source_ready": bool(findings.get("unified_object_geometry_source_ready") is True),
        "contact_factor_source_compatible_with_multi_object_geometry": bool(
            findings.get("contact_factor_source_compatible_with_multi_object_geometry") is True
        ),
        "object_pose_source_compatible_with_contact_factors": bool(
            findings.get("object_pose_source_compatible_with_contact_factors") is True
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "rigid_pose_requirement_met": bool(report.get("rigid_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def object_geometry_hypothesis_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "object-geometry hypothesis frame_count"),
        "object_count": require_int(report.get("object_count"), "object-geometry hypothesis object_count"),
        "object_frame_rows": require_int(
            report.get("object_frame_rows"),
            "object-geometry hypothesis object_frame_rows",
        ),
        "visible_surface_frame_rows": require_int(
            report.get("visible_surface_frame_rows"),
            "object-geometry hypothesis visible_surface_frame_rows",
        ),
        "state_counts": require_dict(report.get("state_counts"), "object-geometry hypothesis state_counts"),
        "objects_with_persistent_visible_surface_shape": require_int(
            report.get("objects_with_persistent_visible_surface_shape"),
            "object-geometry hypothesis objects_with_persistent_visible_surface_shape",
        ),
        "objects_with_object_depth_repair_candidates": require_int(
            report.get("objects_with_object_depth_repair_candidates"),
            "object-geometry hypothesis objects_with_object_depth_repair_candidates",
        ),
        "objects_with_local_contact_patches": require_int(
            report.get("objects_with_local_contact_patches"),
            "object-geometry hypothesis objects_with_local_contact_patches",
        ),
        "objects_with_material_surface_replay_ready_segments": require_int(
            report.get("objects_with_material_surface_replay_ready_segments"),
            "object-geometry hypothesis objects_with_material_surface_replay_ready_segments",
        ),
        "objects_with_accepted_reconstruction_results": require_int(
            report.get("objects_with_accepted_reconstruction_results"),
            "object-geometry hypothesis objects_with_accepted_reconstruction_results",
        ),
        "accepted_reconstruction_result_count": require_int(
            report.get("accepted_reconstruction_result_count"),
            "object-geometry hypothesis accepted_reconstruction_result_count",
        ),
        "complete_object_geometry_hypothesis_count": require_int(
            report.get("complete_object_geometry_hypothesis_count"),
            "object-geometry hypothesis complete_object_geometry_hypothesis_count",
        ),
        "contact_compatible_object_geometry_hypothesis_count": require_int(
            report.get("contact_compatible_object_geometry_hypothesis_count"),
            "object-geometry hypothesis contact_compatible_object_geometry_hypothesis_count",
        ),
        "object_pose_factor_ready_hypothesis_count": require_int(
            report.get("object_pose_factor_ready_hypothesis_count"),
            "object-geometry hypothesis object_pose_factor_ready_hypothesis_count",
        ),
        "source_incompatibility_count": require_int(
            report.get("source_incompatibility_count"),
            "object-geometry hypothesis source_incompatibility_count",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "rigid_pose_requirement_met": bool(report.get("rigid_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def object_geometry_factor_problem_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "object-geometry factor frame_count"),
        "factor_problem_object_rows": require_int(
            report.get("factor_problem_object_rows"),
            "object-geometry factor problem object rows",
        ),
        "state_counts": require_dict(report.get("state_counts"), "object-geometry factor state_counts"),
        "solve_activation_ready_object_count": require_int(
            report.get("solve_activation_ready_object_count"),
            "object-geometry factor solve_activation_ready_object_count",
        ),
        "visible_surface_factor_rows": require_int(
            report.get("visible_surface_factor_rows"),
            "object-geometry factor visible_surface_factor_rows",
        ),
        "material_rigidity_pair_factor_count": require_int(
            report.get("material_rigidity_pair_factor_count"),
            "object-geometry factor material_rigidity_pair_factor_count",
        ),
        "partial_material_pose_ready_segment_count": require_int(
            report.get("partial_material_pose_ready_segment_count"),
            "object-geometry factor partial_material_pose_ready_segment_count",
        ),
        "partial_visible_surface_replay_ready_segment_count": require_int(
            report.get("partial_visible_surface_replay_ready_segment_count"),
            "object-geometry factor partial_visible_surface_replay_ready_segment_count",
        ),
        "observed_surface_geometry_seed_count": require_int(
            report.get("observed_surface_geometry_seed_count"),
            "object-geometry factor observed_surface_geometry_seed_count",
        ),
        "observed_surface_geometry_seed_vertices": require_int(
            report.get("observed_surface_geometry_seed_vertices"),
            "object-geometry factor observed_surface_geometry_seed_vertices",
        ),
        "observed_surface_geometry_seed_faces": require_int(
            report.get("observed_surface_geometry_seed_faces"),
            "object-geometry factor observed_surface_geometry_seed_faces",
        ),
        "geometry_reconstruction_job_count": require_int(
            report.get("geometry_reconstruction_job_count"),
            "object-geometry factor geometry_reconstruction_job_count",
        ),
        "geometry_reconstruction_solver_job_ready_count": require_int(
            report.get("geometry_reconstruction_solver_job_ready_count"),
            "object-geometry factor geometry_reconstruction_solver_job_ready_count",
        ),
        "geometry_reconstruction_hidden_topology_job_count": require_int(
            report.get("geometry_reconstruction_hidden_topology_job_count"),
            "object-geometry factor geometry_reconstruction_hidden_topology_job_count",
        ),
        "geometry_reconstruction_result_job_count": require_int(
            report.get("geometry_reconstruction_result_job_count"),
            "object-geometry factor geometry_reconstruction_result_job_count",
        ),
        "geometry_reconstruction_pending_solver_output_count": require_int(
            report.get("geometry_reconstruction_pending_solver_output_count"),
            "object-geometry factor geometry_reconstruction_pending_solver_output_count",
        ),
        "geometry_reconstruction_solver_output_detected_count": require_int(
            report.get("geometry_reconstruction_solver_output_detected_count"),
            "object-geometry factor geometry_reconstruction_solver_output_detected_count",
        ),
        "geometry_reconstruction_mesh_file_detected_count": require_int(
            report.get("geometry_reconstruction_mesh_file_detected_count"),
            "object-geometry factor geometry_reconstruction_mesh_file_detected_count",
        ),
        "geometry_reconstruction_pose_sequence_complete_count": require_int(
            report.get("geometry_reconstruction_pose_sequence_complete_count"),
            "object-geometry factor geometry_reconstruction_pose_sequence_complete_count",
        ),
        "geometry_reconstruction_mesh_scale_plausible_count": require_int(
            report.get("geometry_reconstruction_mesh_scale_plausible_count"),
            "object-geometry factor geometry_reconstruction_mesh_scale_plausible_count",
        ),
        "geometry_reconstruction_mesh_projection_qc_passed_count": require_int(
            report.get("geometry_reconstruction_mesh_projection_qc_passed_count"),
            "object-geometry factor geometry_reconstruction_mesh_projection_qc_passed_count",
        ),
        "geometry_reconstruction_result_hidden_topology_job_count": require_int(
            report.get("geometry_reconstruction_result_hidden_topology_job_count"),
            "object-geometry factor geometry_reconstruction_result_hidden_topology_job_count",
        ),
        "geometry_reconstruction_accepted_result_count": require_int(
            report.get("geometry_reconstruction_accepted_result_count"),
            "object-geometry factor geometry_reconstruction_accepted_result_count",
        ),
        "depth_contact_evaluated_frame_count": require_int(
            report.get("depth_contact_evaluated_frame_count"),
            "object-geometry factor depth_contact_evaluated_frame_count",
        ),
        "depth_contact_evaluated_hand_rows": require_int(
            report.get("depth_contact_evaluated_hand_rows"),
            "object-geometry factor depth_contact_evaluated_hand_rows",
        ),
        "depth_contact_near_reconstructed_mesh_hand_rows": require_int(
            report.get("depth_contact_near_reconstructed_mesh_hand_rows"),
            "object-geometry factor depth_contact_near_reconstructed_mesh_hand_rows",
        ),
        "depth_contact_reconstructed_mesh_contact_candidate_rows": require_int(
            report.get("depth_contact_reconstructed_mesh_contact_candidate_rows"),
            "object-geometry factor depth_contact_reconstructed_mesh_contact_candidate_rows",
        ),
        "depth_contact_shared_depth_state_ready_frame_count": require_int(
            report.get("depth_contact_shared_depth_state_ready_frame_count"),
            "object-geometry factor depth_contact_shared_depth_state_ready_frame_count",
        ),
        "depth_contact_owner_incompatibility_count": require_int(
            report.get("depth_contact_owner_incompatibility_count"),
            "object-geometry factor depth_contact_owner_incompatibility_count",
        ),
        "multi_object_contact_factor_ready_rows": require_int(
            report.get("multi_object_contact_factor_ready_rows"),
            "object-geometry factor multi_object_contact_factor_ready_rows",
        ),
        "geometry_source_conflict_count": require_int(
            report.get("geometry_source_conflict_count"),
            "object-geometry factor geometry_source_conflict_count",
        ),
        "complete_object_geometry_hypothesis_count": require_int(
            report.get("complete_object_geometry_hypothesis_count"),
            "object-geometry factor complete_object_geometry_hypothesis_count",
        ),
        "contact_compatible_object_geometry_hypothesis_count": require_int(
            report.get("contact_compatible_object_geometry_hypothesis_count"),
            "object-geometry factor contact_compatible_object_geometry_hypothesis_count",
        ),
        "object_pose_factor_ready_hypothesis_count": require_int(
            report.get("object_pose_factor_ready_hypothesis_count"),
            "object-geometry factor object_pose_factor_ready_hypothesis_count",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "rigid_pose_requirement_met": bool(report.get("rigid_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def geometry_reconstruction_jobs_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "job_count": require_int(report.get("job_count"), "geometry reconstruction job_count"),
        "solver_job_ready_count": require_int(
            report.get("solver_job_ready_count"),
            "geometry reconstruction solver_job_ready_count",
        ),
        "skipped_job_count": require_int(report.get("skipped_job_count"), "geometry reconstruction skipped_job_count"),
        "hidden_topology_reconstructed_job_count": require_int(
            report.get("hidden_topology_reconstructed_job_count"),
            "geometry reconstruction hidden_topology_reconstructed_job_count",
        ),
        "complete_geometry_seed_count": require_int(
            report.get("complete_geometry_seed_count"),
            "geometry reconstruction complete_geometry_seed_count",
        ),
        "contact_compatible_geometry_seed_count": require_int(
            report.get("contact_compatible_geometry_seed_count"),
            "geometry reconstruction contact_compatible_geometry_seed_count",
        ),
        "full_active_interval_geometry_seed_count": require_int(
            report.get("full_active_interval_geometry_seed_count"),
            "geometry reconstruction full_active_interval_geometry_seed_count",
        ),
        "rectification_nearest_3d_residual_p95_m": require_dict(
            report.get("rectification_nearest_3d_residual_p95_m"),
            "geometry reconstruction rectification residual",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "rigid_pose_requirement_met": bool(report.get("rigid_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def mesh_counts(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_count": require_int(metadata.get("frame_count"), "mesh metadata frame_count"),
        "mesh_frames": require_int(metadata.get("mesh_frames"), "mesh metadata mesh_frames"),
        "missing_mesh_frame_count": require_int(
            metadata.get("missing_mesh_frame_count"), "mesh metadata missing_mesh_frame_count"
        ),
        "object_geometry_complete": bool(metadata.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(metadata.get("object_pose_requirement_met") is True),
        "multi_object_timeline_ready": bool(metadata.get("multi_object_timeline_ready") is True),
    }


def geometry_reconstruction_results_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "job_count": require_int(report.get("job_count"), "geometry reconstruction results job_count"),
        "solver_job_ready_count": require_int(
            report.get("solver_job_ready_count"),
            "geometry reconstruction results solver_job_ready_count",
        ),
        "pending_solver_output_count": require_int(
            report.get("pending_solver_output_count"),
            "geometry reconstruction results pending_solver_output_count",
        ),
        "solver_output_detected_count": require_int(
            report.get("solver_output_detected_count"),
            "geometry reconstruction results solver_output_detected_count",
        ),
        "mesh_file_detected_count": require_int(
            report.get("mesh_file_detected_count"),
            "geometry reconstruction results mesh_file_detected_count",
        ),
        "pose_sequence_complete_count": require_int(
            report.get("pose_sequence_complete_count"),
            "geometry reconstruction results pose_sequence_complete_count",
        ),
        "mesh_scale_plausible_count": require_int(
            report.get("mesh_scale_plausible_count"),
            "geometry reconstruction results mesh_scale_plausible_count",
        ),
        "mesh_projection_qc_passed_count": require_int(
            report.get("mesh_projection_qc_passed_count"),
            "geometry reconstruction results mesh_projection_qc_passed_count",
        ),
        "hidden_topology_reconstructed_job_count": require_int(
            report.get("hidden_topology_reconstructed_job_count"),
            "geometry reconstruction results hidden_topology_reconstructed_job_count",
        ),
        "accepted_reconstruction_result_count": require_int(
            report.get("accepted_reconstruction_result_count"),
            "geometry reconstruction results accepted_reconstruction_result_count",
        ),
        "complete_geometry_seed_count": require_int(
            report.get("complete_geometry_seed_count"),
            "geometry reconstruction results complete_geometry_seed_count",
        ),
        "contact_compatible_geometry_seed_count": require_int(
            report.get("contact_compatible_geometry_seed_count"),
            "geometry reconstruction results contact_compatible_geometry_seed_count",
        ),
        "full_active_interval_geometry_seed_count": require_int(
            report.get("full_active_interval_geometry_seed_count"),
            "geometry reconstruction results full_active_interval_geometry_seed_count",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "rigid_pose_requirement_met": bool(report.get("rigid_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def depth_contact_consistency_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "accepted_reconstruction_job_count": require_int(
            report.get("accepted_reconstruction_job_count"),
            "depth-contact accepted_reconstruction_job_count",
        ),
        "evaluated_frame_count": require_int(
            report.get("evaluated_frame_count"),
            "depth-contact evaluated_frame_count",
        ),
        "evaluated_hand_rows": require_int(
            report.get("evaluated_hand_rows"),
            "depth-contact evaluated_hand_rows",
        ),
        "near_reconstructed_mesh_hand_rows": require_int(
            report.get("near_reconstructed_mesh_hand_rows"),
            "depth-contact near_reconstructed_mesh_hand_rows",
        ),
        "reconstructed_mesh_contact_candidate_rows": require_int(
            report.get("reconstructed_mesh_contact_candidate_rows"),
            "depth-contact reconstructed_mesh_contact_candidate_rows",
        ),
        "shared_depth_state_ready_frame_count": require_int(
            report.get("shared_depth_state_ready_frame_count"),
            "depth-contact shared_depth_state_ready_frame_count",
        ),
        "depth_owner_incompatibility_count": require_int(
            report.get("depth_owner_incompatibility_count"),
            "depth-contact depth_owner_incompatibility_count",
        ),
        "visible_unidepth_m": require_dict(
            report.get("visible_unidepth_m"),
            "depth-contact visible_unidepth_m",
        ),
        "reconstructed_mesh_camera_depth_m": require_dict(
            report.get("reconstructed_mesh_camera_depth_m"),
            "depth-contact reconstructed_mesh_camera_depth_m",
        ),
        "reconstructed_mesh_front_surface_depth_abs_p95_m": require_dict(
            report.get("reconstructed_mesh_front_surface_depth_abs_p95_m"),
            "depth-contact reconstructed_mesh_front_surface_depth_abs_p95_m",
        ),
        "legacy_object_center_depth_m": require_dict(
            report.get("legacy_object_center_depth_m"),
            "depth-contact legacy_object_center_depth_m",
        ),
        "hand_source_depth_m": require_dict(
            report.get("hand_source_depth_m"),
            "depth-contact hand_source_depth_m",
        ),
        "reconstructed_mesh_to_hand_min_m": require_dict(
            report.get("reconstructed_mesh_to_hand_min_m"),
            "depth-contact reconstructed_mesh_to_hand_min_m",
        ),
        "shared_depth_contact_state_ready": bool(report.get("shared_depth_contact_state_ready") is True),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "rigid_pose_requirement_met": bool(report.get("rigid_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def variable_family(
    name: str,
    required_scope: str,
    current_graph_role: str,
    current_evidence: dict[str, Any],
    missing_state: list[str],
) -> dict[str, Any]:
    return {
        "family": name,
        "required_scope": required_scope,
        "current_graph_role": current_graph_role,
        "estimated_by_current_sparse_graph": current_graph_role == "estimated_variable",
        "current_evidence": current_evidence,
        "missing_state": missing_state,
        "v3_requirement_met": False,
    }


def required_variable_families(
    roster: dict[str, Any],
    timeline: dict[str, Any],
    visible_surface: dict[str, Any],
    geometry_state: dict[str, Any],
    object_track_dataset: dict[str, Any],
    object_material_track: dict[str, Any],
    object_material_motion_state: dict[str, Any],
    object_material_pose_candidate: dict[str, Any],
    object_material_surface_replay: dict[str, Any],
    multi_object_contact_evidence: dict[str, Any],
    geometry_source_audit: dict[str, Any],
    object_geometry_hypothesis_state: dict[str, Any],
    object_geometry_factor_problem: dict[str, Any],
    geometry_reconstruction_jobs: dict[str, Any],
    geometry_reconstruction_results: dict[str, Any],
    depth_contact_consistency: dict[str, Any],
    counts: dict[str, int],
    sparse: dict[str, Any],
    contact: dict[str, Any],
    mesh: dict[str, Any],
) -> list[dict[str, Any]]:
    frame_count = require_int(sparse["frame_count"], "sparse counts frame_count")
    object_frame_rows = require_int(timeline["object_frame_rows"], "multi-object object_frame_rows")
    required_contact_rows = 2 * object_frame_rows
    return [
        variable_family(
            "camera_trajectory_se3_per_frame",
            "one SE(3) head-camera pose for every raw frame, jointly constrained by hand, object, depth, and contact evidence",
            "fixed_input",
            {
                "frame_count": frame_count,
                "current_source": "source annotation camera stream inside graph-corrected V17 annotations",
            },
            [
                "joint camera updates are absent",
                "camera residuals against object tracks, hand reprojection, and contact consistency are not optimized",
            ],
        ),
        variable_family(
            "mano_articulation_shape_per_hand_frame",
            "MANO wrist pose, articulation, shape, and visible/predicted hand state for each active hand frame",
            "fixed_input_plus_scalar_ray_shift",
            {
                "wilor_measurements": counts.get("wilor", 0),
                "hamer_measurements": counts.get("hamer", 0),
                "hawor_measurements": counts.get("hawor", 0),
                "v16_hand_state_measurements": counts.get("v16_hand_state", 0),
                "current_ray_shift_variables": sparse["hand_ray_shift_variables"],
            },
            [
                "MANO pose parameters are not graph variables",
                "MANO shape parameters are not graph variables",
                "world wrist pose is only corrected along camera rays",
                "occluded hands are not represented as prediction/update latent states",
            ],
        ),
        variable_family(
            "object_identity_multiobject_timeline",
            "simultaneous object roster with identity, role, active interval, and per-frame state for every manipulated or contact-relevant object",
            "mask_evidence_timeline_materialized",
            {
                "roster_row_count": roster["roster_row_count"],
                "active_vlm_object_count": timeline["object_count"],
                "object_frame_rows": object_frame_rows,
                "visible_mask_frame_rows": timeline["visible_mask_frame_rows"],
                "active_without_visible_mask_frame_rows": timeline["active_without_visible_mask_frame_rows"],
                "active_vlm_object_ids": roster["active_vlm_object_ids"],
                "ambiguous_expected_object_ids": roster["ambiguous_expected_object_ids"],
            },
            [
                "the new multi-object timeline carries mask evidence but no mesh or pose variables",
                "downstream graph-corrected annotations still expose one legacy object stream",
                "the sparse graph has variables for one object per frame, not simultaneous roster objects",
                "ambiguous expected objects are not resolved into graph states",
            ],
        ),
        variable_family(
            "object_geometry_topology_per_object",
            "mesh-backed reconstructed geometry for each manipulated object, with rigid/deformable topology state as data",
            "partial_fixed_input",
            {
                "legacy_or_partial_mesh_frames": mesh["mesh_frames"],
                "missing_mesh_frame_count": mesh["missing_mesh_frame_count"],
                "multi_object_visible_surface_rows": visible_surface["surface_frame_rows"],
                "multi_object_visible_surface_rejected_rows": visible_surface[
                    "rejected_visible_object_frame_rows"
                ],
                "multi_object_visible_surface_rejection_reasons": visible_surface[
                    "rejection_reason_counts"
                ],
                "center_normalized_visible_surface_envelope_candidates": geometry_state[
                    "visible_surface_envelope_candidate_count"
                ],
                "persistent_visible_surface_candidates": geometry_state[
                    "persistent_visible_surface_candidate_count"
                ],
                "rigid_pose_candidates": geometry_state["rigid_pose_candidate_count"],
                "object_track_dataset_exported_frames": object_track_dataset[
                    "total_exported_frames"
                ],
                "object_track_dataset_exported_objects": object_track_dataset[
                    "exported_object_count"
                ],
                "material_track_windows": object_material_track[
                    "material_track_window_count"
                ],
                "material_tracked_objects": object_material_track[
                    "material_tracked_object_count"
                ],
                "persistent_window_motion_candidates": object_material_motion_state[
                    "persistent_window_motion_candidate_count"
                ],
                "partial_material_pose_candidate_segments": object_material_pose_candidate[
                    "partial_material_pose_candidate_segment_count"
                ],
                "partial_material_pose_ready_segments": object_material_pose_candidate[
                    "partial_material_pose_candidate_ready_segment_count"
                ],
                "partial_visible_surface_replay_candidates": object_material_surface_replay[
                    "partial_visible_surface_replay_candidate_count"
                ],
                "partial_visible_surface_replay_ready_segments": object_material_surface_replay[
                    "partial_visible_surface_replay_ready_count"
                ],
                "geometry_source_audit_incompatibilities": geometry_source_audit[
                    "source_incompatibility_count"
                ],
                "object_geometry_hypothesis_state_counts": object_geometry_hypothesis_state[
                    "state_counts"
                ],
                "objects_with_accepted_reconstruction_results": object_geometry_hypothesis_state[
                    "objects_with_accepted_reconstruction_results"
                ],
                "hypothesis_accepted_reconstruction_result_count": object_geometry_hypothesis_state[
                    "accepted_reconstruction_result_count"
                ],
                "object_geometry_factor_problem_state_counts": object_geometry_factor_problem[
                    "state_counts"
                ],
                "object_geometry_factor_problem_rows": object_geometry_factor_problem[
                    "factor_problem_object_rows"
                ],
                "object_geometry_factor_visible_surface_rows": object_geometry_factor_problem[
                    "visible_surface_factor_rows"
                ],
                "object_geometry_factor_material_rigidity_pairs": object_geometry_factor_problem[
                    "material_rigidity_pair_factor_count"
                ],
                "object_geometry_factor_solve_activation_ready_objects": object_geometry_factor_problem[
                    "solve_activation_ready_object_count"
                ],
                "geometry_reconstruction_job_count": geometry_reconstruction_jobs["job_count"],
                "geometry_reconstruction_solver_job_ready_count": geometry_reconstruction_jobs[
                    "solver_job_ready_count"
                ],
                "geometry_reconstruction_hidden_topology_job_count": geometry_reconstruction_jobs[
                    "hidden_topology_reconstructed_job_count"
                ],
                "geometry_reconstruction_rectification_residual_p95_m": geometry_reconstruction_jobs[
                    "rectification_nearest_3d_residual_p95_m"
                ],
                "geometry_reconstruction_pending_solver_output_count": geometry_reconstruction_results[
                    "pending_solver_output_count"
                ],
                "geometry_reconstruction_solver_output_detected_count": geometry_reconstruction_results[
                    "solver_output_detected_count"
                ],
                "geometry_reconstruction_mesh_file_detected_count": geometry_reconstruction_results[
                    "mesh_file_detected_count"
                ],
                "geometry_reconstruction_pose_sequence_complete_count": geometry_reconstruction_results[
                    "pose_sequence_complete_count"
                ],
                "geometry_reconstruction_mesh_scale_plausible_count": geometry_reconstruction_results[
                    "mesh_scale_plausible_count"
                ],
                "geometry_reconstruction_mesh_projection_qc_passed_count": geometry_reconstruction_results[
                    "mesh_projection_qc_passed_count"
                ],
                "geometry_reconstruction_result_hidden_topology_job_count": geometry_reconstruction_results[
                    "hidden_topology_reconstructed_job_count"
                ],
                "geometry_reconstruction_accepted_result_count": geometry_reconstruction_results[
                    "accepted_reconstruction_result_count"
                ],
                "depth_contact_evaluated_frame_count": depth_contact_consistency[
                    "evaluated_frame_count"
                ],
                "depth_contact_evaluated_hand_rows": depth_contact_consistency[
                    "evaluated_hand_rows"
                ],
                "depth_contact_near_reconstructed_mesh_hand_rows": depth_contact_consistency[
                    "near_reconstructed_mesh_hand_rows"
                ],
                "depth_contact_owner_incompatibility_count": depth_contact_consistency[
                    "depth_owner_incompatibility_count"
                ],
                "depth_contact_shared_depth_state_ready_frame_count": depth_contact_consistency[
                    "shared_depth_state_ready_frame_count"
                ],
                "complete_object_geometry_hypothesis_count": object_geometry_hypothesis_state[
                    "complete_object_geometry_hypothesis_count"
                ],
                "contact_compatible_object_geometry_hypothesis_count": object_geometry_hypothesis_state[
                    "contact_compatible_object_geometry_hypothesis_count"
                ],
                "unified_object_geometry_source_ready": geometry_source_audit[
                    "unified_object_geometry_source_ready"
                ],
                "contact_factor_source_compatible_with_multi_object_geometry": geometry_source_audit[
                    "contact_factor_source_compatible_with_multi_object_geometry"
                ],
                "noncandidate_local_adjacent_material_motion_windows": object_material_motion_state[
                    "noncandidate_local_adjacent_material_motion_window_count"
                ],
                "persistent_object_shape_measurements": counts.get("persistent_object_shape", 0),
                "local_contact_patch_measurements": counts.get("local_contact_patch", 0),
                "object_geometry_complete": mesh["object_geometry_complete"],
            },
            [
                "complete object meshes are absent for several active objects",
                "local contact patches and visible surfaces are still QC evidence, not complete object geometry",
                "current contact factors and multi-object visible surfaces are not source-compatible",
                "per-object geometry hypotheses are now materialized but none is complete or contact-compatible",
                "object-centric geometry factor blocks are now materialized but no object is activatable for solving",
                "accepted hidden-topology reconstructions exist only for short observed-surface seed windows and are not full-interval or contact-compatible object geometry",
                "accepted reconstructions are in the visible-depth state, while the current hand/contact graph uses a different source-camera depth state",
                "topology/deformation variables are not optimized",
            ],
        ),
        variable_family(
            "object_pose_se3_or_deformation_per_object_frame",
            "per-object frame pose for rigid objects and explicit deformation state for deformable objects",
            "estimated_variable_for_single_legacy_stream_only",
            {
                "current_single_stream_object_variable_frames": sparse["object_variable_frames"],
                "required_multi_object_frame_rows": object_frame_rows,
                "object_pose_requirement_met": sparse["object_pose_requirement_met"],
                "visible_surface_envelope_candidates": geometry_state[
                    "visible_surface_envelope_candidate_count"
                ],
                "rigid_pose_candidates": geometry_state["rigid_pose_candidate_count"],
                "object_track_dataset_exported_frames": object_track_dataset[
                    "total_exported_frames"
                ],
                "material_track_windows": object_material_track[
                    "material_track_window_count"
                ],
                "rigid_motion_ready_windows": object_material_track[
                    "rigid_motion_ready_window_count"
                ],
                "rigid_factor_ready_pair_count": object_material_track[
                    "rigid_factor_ready_pair_count"
                ],
                "persistent_window_motion_candidates": object_material_motion_state[
                    "persistent_window_motion_candidate_count"
                ],
                "local_adjacent_material_motion_windows": object_material_motion_state[
                    "local_adjacent_material_motion_window_count"
                ],
                "noncandidate_local_adjacent_material_motion_windows": object_material_motion_state[
                    "noncandidate_local_adjacent_material_motion_window_count"
                ],
                "candidate_window_ids": object_material_motion_state["candidate_window_ids"],
                "partial_material_pose_candidate_segments": object_material_pose_candidate[
                    "partial_material_pose_candidate_segment_count"
                ],
                "partial_material_pose_ready_segments": object_material_pose_candidate[
                    "partial_material_pose_candidate_ready_segment_count"
                ],
                "partial_material_pose_candidate_segment_ids": object_material_pose_candidate[
                    "candidate_segment_ids"
                ],
                "partial_visible_surface_replay_ready_candidate_ids": object_material_surface_replay[
                    "ready_candidate_ids"
                ],
                "object_pose_source_compatible_with_contact_factors": geometry_source_audit[
                    "object_pose_source_compatible_with_contact_factors"
                ],
                "object_pose_factor_ready_hypothesis_count": object_geometry_hypothesis_state[
                    "object_pose_factor_ready_hypothesis_count"
                ],
                "object_geometry_factor_pose_ready_hypothesis_count": object_geometry_factor_problem[
                    "object_pose_factor_ready_hypothesis_count"
                ],
                "object_geometry_factor_partial_pose_ready_segments": object_geometry_factor_problem[
                    "partial_material_pose_ready_segment_count"
                ],
                "object_geometry_factor_partial_surface_replay_ready_segments": object_geometry_factor_problem[
                    "partial_visible_surface_replay_ready_segment_count"
                ],
                "observed_surface_geometry_seed_count": object_geometry_factor_problem[
                    "observed_surface_geometry_seed_count"
                ],
                "observed_surface_geometry_seed_vertices": object_geometry_factor_problem[
                    "observed_surface_geometry_seed_vertices"
                ],
                "observed_surface_geometry_seed_faces": object_geometry_factor_problem[
                    "observed_surface_geometry_seed_faces"
                ],
                "geometry_reconstruction_job_count": geometry_reconstruction_jobs["job_count"],
                "geometry_reconstruction_solver_job_ready_count": geometry_reconstruction_jobs[
                    "solver_job_ready_count"
                ],
                "geometry_reconstruction_hidden_topology_job_count": geometry_reconstruction_jobs[
                    "hidden_topology_reconstructed_job_count"
                ],
                "geometry_reconstruction_result_hidden_topology_job_count": geometry_reconstruction_results[
                    "hidden_topology_reconstructed_job_count"
                ],
                "geometry_reconstruction_accepted_result_count": geometry_reconstruction_results[
                    "accepted_reconstruction_result_count"
                ],
                "depth_contact_reconstructed_mesh_contact_candidate_rows": depth_contact_consistency[
                    "reconstructed_mesh_contact_candidate_rows"
                ],
                "depth_contact_shared_depth_state_ready_frame_count": depth_contact_consistency[
                    "shared_depth_state_ready_frame_count"
                ],
                "depth_contact_owner_incompatibility_count": depth_contact_consistency[
                    "depth_owner_incompatibility_count"
                ],
                "partial_material_pose_replay_is_complete_object_geometry": geometry_source_audit[
                    "partial_material_pose_replay_is_complete_object_geometry"
                ],
                "legacy_single_stream_object_variable_frames": geometry_source_audit[
                    "legacy_single_stream_object_variable_frames"
                ],
                "exported_object_ids_without_material_tracks": object_material_track[
                    "exported_object_ids_without_material_tracks"
                ],
            },
            [
                "simultaneous object poses are missing",
                "deformable-bag state is represented by local patches and legacy centers, not deformation variables",
                "object pose corrections are small per-frame updates around fixed input geometry",
                "center-normalized visible-surface envelopes do not provide material correspondence or SE(3) pose",
                "material tracks cover sampled object windows only and are not integrated as full-timeline object variables",
                "persistent material-motion candidates do not provide canonical object meshes or full-timeline pose/deformation variables",
                "partial material-point SE(3) candidates exist only for accepted short segments and are not connected to complete object geometry",
                "visible-surface replay tests only observed surfaces and does not reconstruct hidden topology",
                "observed-surface geometry seeds are short-segment canonical seeds, not full active-interval object pose timelines",
                "accepted RGBD reconstructions are short-window mesh and pose evidence, not full active-interval object pose timelines",
                "accepted RGBD reconstructions do not share a contact-depth state with current MANO geometry",
                "object-centric pose factors are listed but no object has a complete geometry state that can own them",
                "object-pose evidence is not source-compatible with the current contact factors",
                "no per-object geometry hypothesis is ready to own pose factors",
            ],
        ),
        variable_family(
            "contact_mode_per_hand_object_frame",
            "contact, no-contact, or unobserved state for each hand-object pair across the full timeline",
            "visible_surface_distance_evidence_materialized",
            {
                "current_hand_side_rows": contact["row_count"],
                "current_contact_mode_rows": contact["contact_mode_count"],
                "current_factor_ready_rows": contact["contact_factor_ready_count"],
                "minimum_required_hand_object_rows_from_roster": required_contact_rows,
                "multi_object_hand_object_rows": multi_object_contact_evidence["hand_object_rows"],
                "multi_object_measured_distance_rows": multi_object_contact_evidence["measured_distance_rows"],
                "multi_object_unobserved_rows": multi_object_contact_evidence["unobserved_rows"],
                "multi_object_visible_surface_distance_candidate_rows": multi_object_contact_evidence[
                    "visible_surface_distance_candidate_rows"
                ],
                "multi_object_contact_factor_ready_rows": multi_object_contact_evidence[
                    "contact_factor_ready_rows"
                ],
                "contact_mode_ready_rows_with_same_frame_side_multi_object_measurement": geometry_source_audit[
                    "contact_mode_ready_rows_with_same_frame_side_multi_object_measurement"
                ],
                "contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate": geometry_source_audit[
                    "contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate"
                ],
                "local_patch_visible_surface_conflict_count": geometry_source_audit[
                    "local_patch_visible_surface_conflict_count"
                ],
                "accepted_local_patches_conflict_with_multi_object_visible_surface_distance": geometry_source_audit[
                    "accepted_local_patches_conflict_with_multi_object_visible_surface_distance"
                ],
                "contact_factor_source_compatible_with_multi_object_geometry": geometry_source_audit[
                    "contact_factor_source_compatible_with_multi_object_geometry"
                ],
                "contact_compatible_object_geometry_hypothesis_count": object_geometry_hypothesis_state[
                    "contact_compatible_object_geometry_hypothesis_count"
                ],
                "object_geometry_factor_contact_ready_rows": object_geometry_factor_problem[
                    "multi_object_contact_factor_ready_rows"
                ],
                "object_geometry_factor_contact_compatible_hypothesis_count": object_geometry_factor_problem[
                    "contact_compatible_object_geometry_hypothesis_count"
                ],
                "depth_contact_near_reconstructed_mesh_hand_rows": depth_contact_consistency[
                    "near_reconstructed_mesh_hand_rows"
                ],
                "depth_contact_reconstructed_mesh_contact_candidate_rows": depth_contact_consistency[
                    "reconstructed_mesh_contact_candidate_rows"
                ],
                "depth_contact_shared_depth_state_ready_frame_count": depth_contact_consistency[
                    "shared_depth_state_ready_frame_count"
                ],
            },
            [
                "contact modes are estimated before the sparse geometry graph and then fixed",
                "the full hand-object table measures visible-surface distance but does not estimate contact modes",
                "accepted local contact-patch states are not unified with multi-object visible surfaces",
                "contact-mode ready rows have no same-frame multi-object visible-surface contact candidates",
                "object-centric contact factor blocks have zero factor-ready rows against multi-object geometry",
                "accepted reconstruction meshes have no near-contact hand rows under the current depth state",
                "unobserved rows do not carry uncertainty variables or prediction/update state",
            ],
        ),
        variable_family(
            "contact_patch_identity_and_hand_support",
            "latent object patch, hand support patch, and anatomical support identity for each accepted contact interval",
            "derived_correspondence",
            {
                "contact_factor_count": sparse["contact_factor_count"],
                "linearized_contact_correspondences": sparse["linearized_contact_correspondences"],
                "nearest_mano_vertices_per_factor": 16,
            },
            [
                "nearest MANO vertices are selected from fixed geometry rather than estimated as stable contact support",
                "object patch identity is not temporally optimized",
                "hand support stability and non-contact neighboring surface are diagnostics, not factors",
            ],
        ),
        variable_family(
            "dense_depth_and_visible_surface_state",
            "metric depth, visible object surface, occlusion, and scale state tied to image/depth evidence",
            "fixed_measurement_input",
            {
                "object_mesh_measurements": counts.get("object_mesh", 0),
                "sam2_object_mask_measurements": counts.get("sam2_object_mask", 0),
                "multi_object_visible_surface_rows": visible_surface["surface_frame_rows"],
                "visible_object_frame_rows": visible_surface["visible_object_frame_rows"],
                "depth_frame_count": visible_surface["depth_frame_count"],
                "rejected_visible_object_frame_rows": visible_surface[
                    "rejected_visible_object_frame_rows"
                ],
                "center_normalized_visible_surface_envelope_candidates": geometry_state[
                    "visible_surface_envelope_candidate_count"
                ],
                "depth_contact_visible_unidepth_m": depth_contact_consistency["visible_unidepth_m"],
                "depth_contact_reconstructed_mesh_camera_depth_m": depth_contact_consistency[
                    "reconstructed_mesh_camera_depth_m"
                ],
                "depth_contact_reconstructed_mesh_front_surface_depth_abs_p95_m": depth_contact_consistency[
                    "reconstructed_mesh_front_surface_depth_abs_p95_m"
                ],
                "depth_contact_legacy_object_center_depth_m": depth_contact_consistency[
                    "legacy_object_center_depth_m"
                ],
                "depth_contact_hand_source_depth_m": depth_contact_consistency[
                    "hand_source_depth_m"
                ],
                "depth_contact_owner_incompatibility_count": depth_contact_consistency[
                    "depth_owner_incompatibility_count"
                ],
            },
            [
                "visible surfaces are now materialized as fixed measurements where mask and metric depth overlap",
                "depth/object/camera contradictions are not jointly optimized",
                "accepted object reconstructions, legacy object centers, and MANO hands do not currently share one depth owner",
                "occlusion state is not a latent variable with uncertainty",
            ],
        ),
        variable_family(
            "physical_consistency_terms",
            "nonpenetration, support, contact persistence, sliding, object rigidity/deformation, and manipulation dynamics",
            "partial_residuals_only",
            {
                "local_contact_factors": sparse["contact_factor_count"],
                "material_track_rigid_ready_pairs": object_material_track[
                    "rigid_factor_ready_pair_count"
                ],
                "persistent_window_motion_candidates": object_material_motion_state[
                    "persistent_window_motion_candidate_count"
                ],
                "partial_material_pose_ready_segments": object_material_pose_candidate[
                    "partial_material_pose_candidate_ready_segment_count"
                ],
                "partial_visible_surface_replay_ready_segments": object_material_surface_replay[
                    "partial_visible_surface_replay_ready_count"
                ],
                "geometry_reconstruction_solver_job_ready_count": geometry_reconstruction_jobs[
                    "solver_job_ready_count"
                ],
                "geometry_reconstruction_hidden_topology_job_count": geometry_reconstruction_jobs[
                    "hidden_topology_reconstructed_job_count"
                ],
                "geometry_reconstruction_accepted_result_count": geometry_reconstruction_results[
                    "accepted_reconstruction_result_count"
                ],
                "depth_contact_shared_depth_state_ready_frame_count": depth_contact_consistency[
                    "shared_depth_state_ready_frame_count"
                ],
                "depth_contact_owner_incompatibility_count": depth_contact_consistency[
                    "depth_owner_incompatibility_count"
                ],
                "source_incompatibility_count": geometry_source_audit[
                    "source_incompatibility_count"
                ],
                "unified_object_geometry_source_ready": geometry_source_audit[
                    "unified_object_geometry_source_ready"
                ],
                "noncandidate_local_adjacent_material_motion_windows": object_material_motion_state[
                    "noncandidate_local_adjacent_material_motion_window_count"
                ],
                "sparse_graph_solver_completeness": sparse["solver_completeness"],
            },
            [
                "local equality and smoothness do not model nonpenetration or force/support feasibility",
                "contact dynamics are not coupled to object identity, deformation, and MANO articulation",
                "broad hand-object distances remain diagnostics rather than physical constraints",
                "physical terms cannot share one object state until geometry-source ownership is unified",
                "physical contact terms cannot attach to accepted reconstruction meshes until depth ownership is unified",
            ],
        ),
    ]


def case_problem(inputs: CaseInputs) -> dict[str, Any]:
    manifest = require_dict(load_json(inputs.manifest), f"{inputs.case} manifest")
    roster_payload = require_list(load_json(inputs.object_roster), f"{inputs.case} object roster")
    multi_object_timeline = require_dict(load_json(inputs.multi_object_timeline), f"{inputs.case} multi-object timeline")
    visible_surface_report = require_dict(load_json(inputs.visible_surface_report), f"{inputs.case} visible-surface report")
    geometry_state_report = require_dict(load_json(inputs.geometry_state_report), f"{inputs.case} geometry-state report")
    object_track_dataset_summary = require_dict(load_json(inputs.object_track_dataset_summary), f"{inputs.case} object-track dataset summary")
    object_material_track_summary = require_dict(load_json(inputs.object_material_track_summary), f"{inputs.case} object material-track summary")
    object_material_motion_state_summary = require_dict(
        load_json(inputs.object_material_motion_state_summary),
        f"{inputs.case} object material-motion state report",
    )
    object_material_pose_candidate_summary = require_dict(
        load_json(inputs.object_material_pose_candidate_summary),
        f"{inputs.case} object material-pose candidate report",
    )
    object_material_surface_replay_summary = require_dict(
        load_json(inputs.object_material_surface_replay_summary),
        f"{inputs.case} object material-surface replay report",
    )
    multi_object_contact_evidence_summary = require_dict(
        load_json(inputs.multi_object_contact_evidence_summary),
        f"{inputs.case} multi-object contact evidence report",
    )
    geometry_source_audit_report = require_dict(
        load_json(inputs.geometry_source_audit_report),
        f"{inputs.case} geometry-source audit report",
    )
    object_geometry_hypothesis_state_report = require_dict(
        load_json(inputs.object_geometry_hypothesis_state_report),
        f"{inputs.case} object geometry hypothesis-state report",
    )
    object_geometry_factor_problem_report = require_dict(
        load_json(inputs.object_geometry_factor_problem_report),
        f"{inputs.case} object geometry factor-problem report",
    )
    geometry_reconstruction_jobs_report = require_dict(
        load_json(inputs.geometry_reconstruction_jobs_report),
        f"{inputs.case} geometry reconstruction jobs report",
    )
    geometry_reconstruction_results_report = require_dict(
        load_json(inputs.geometry_reconstruction_results_report),
        f"{inputs.case} geometry reconstruction results report",
    )
    depth_contact_consistency_audit_report = require_dict(
        load_json(inputs.depth_contact_consistency_audit_report),
        f"{inputs.case} depth-contact consistency audit report",
    )
    sparse_report = require_dict(load_json(inputs.sparse_report), f"{inputs.case} sparse report")
    contact_report = require_dict(load_json(inputs.contact_mode_report), f"{inputs.case} contact-mode report")
    mesh_metadata = require_dict(load_json(inputs.mesh_metadata), f"{inputs.case} mesh metadata")

    counts = measurement_counts(manifest)
    sparse = graph_counts(sparse_report)
    contact = contact_mode_counts(contact_report)
    timeline = multi_object_timeline_counts(multi_object_timeline)
    visible_surface = visible_surface_counts(visible_surface_report)
    geometry_state = geometry_state_counts(geometry_state_report)
    object_track_dataset = object_track_dataset_counts(object_track_dataset_summary)
    object_material_track = object_material_track_counts(object_material_track_summary)
    object_material_motion_state = object_material_motion_state_counts(object_material_motion_state_summary)
    object_material_pose_candidate = object_material_pose_candidate_counts(object_material_pose_candidate_summary)
    object_material_surface_replay = object_material_surface_replay_counts(object_material_surface_replay_summary)
    multi_object_contact_evidence = multi_object_contact_evidence_counts(multi_object_contact_evidence_summary)
    geometry_source_audit = geometry_source_audit_counts(geometry_source_audit_report)
    object_geometry_hypothesis_state = object_geometry_hypothesis_state_counts(
        object_geometry_hypothesis_state_report
    )
    object_geometry_factor_problem = object_geometry_factor_problem_counts(object_geometry_factor_problem_report)
    geometry_reconstruction_jobs = geometry_reconstruction_jobs_counts(geometry_reconstruction_jobs_report)
    geometry_reconstruction_results = geometry_reconstruction_results_counts(geometry_reconstruction_results_report)
    depth_contact_consistency = depth_contact_consistency_counts(depth_contact_consistency_audit_report)
    mesh = mesh_counts(mesh_metadata)
    roster = roster_audit(roster_payload)

    frame_count = require_int(sparse["frame_count"], f"{inputs.case} sparse frame_count")
    if frame_count != require_int(contact["frame_count"], f"{inputs.case} contact frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse and contact-mode reports")
    if frame_count != require_int(timeline["frame_count"], f"{inputs.case} multi-object timeline frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and multi-object timeline")
    if frame_count != require_int(visible_surface["frame_count"], f"{inputs.case} visible-surface frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and visible-surface report")
    if frame_count != require_int(geometry_state["frame_count"], f"{inputs.case} geometry-state frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and geometry-state report")
    if frame_count != require_int(mesh["frame_count"], f"{inputs.case} mesh frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and mesh metadata")
    if frame_count != require_int(multi_object_contact_evidence["frame_count"], f"{inputs.case} multi-object contact frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and multi-object contact evidence")
    if frame_count != require_int(geometry_source_audit["frame_count"], f"{inputs.case} geometry-source audit frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and geometry-source audit")
    if frame_count != require_int(
        object_geometry_hypothesis_state["frame_count"],
        f"{inputs.case} object-geometry hypothesis frame_count",
    ):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and object-geometry hypothesis state")
    if frame_count != require_int(
        object_geometry_factor_problem["frame_count"],
        f"{inputs.case} object-geometry factor-problem frame_count",
    ):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and object-geometry factor problem")
    if require_int(timeline["visible_mask_frame_rows"], f"{inputs.case} timeline visible mask rows") != require_int(
        visible_surface["visible_object_frame_rows"], f"{inputs.case} visible-surface visible rows"
    ):
        raise RuntimeError(f"{inputs.case} visible mask rows disagree between timeline and visible-surface report")
    if require_int(timeline["object_frame_rows"], f"{inputs.case} timeline object rows") != require_int(
        multi_object_contact_evidence["object_frame_rows"], f"{inputs.case} multi-object contact object rows"
    ):
        raise RuntimeError(f"{inputs.case} object rows disagree between timeline and multi-object contact evidence")
    if require_int(timeline["object_count"], f"{inputs.case} timeline object count") != require_int(
        object_geometry_hypothesis_state["object_count"],
        f"{inputs.case} object-geometry hypothesis object count",
    ):
        raise RuntimeError(f"{inputs.case} object count disagrees with object-geometry hypothesis state")
    if require_int(timeline["object_count"], f"{inputs.case} timeline object count") != require_int(
        object_geometry_factor_problem["factor_problem_object_rows"],
        f"{inputs.case} object-geometry factor problem object rows",
    ):
        raise RuntimeError(f"{inputs.case} object count disagrees with object-geometry factor problem")
    if require_int(timeline["object_frame_rows"], f"{inputs.case} timeline object rows") != require_int(
        object_geometry_hypothesis_state["object_frame_rows"],
        f"{inputs.case} object-geometry hypothesis object rows",
    ):
        raise RuntimeError(f"{inputs.case} object rows disagree with object-geometry hypothesis state")
    if 2 * require_int(timeline["object_frame_rows"], f"{inputs.case} timeline object rows") != require_int(
        multi_object_contact_evidence["hand_object_rows"], f"{inputs.case} multi-object contact hand-object rows"
    ):
        raise RuntimeError(f"{inputs.case} multi-object contact rows must equal two hand sides times object rows")
    if require_int(visible_surface["surface_frame_rows"], f"{inputs.case} visible surface rows") != require_int(
        geometry_state["surface_frame_rows"], f"{inputs.case} geometry-state surface rows"
    ):
        raise RuntimeError(f"{inputs.case} visible-surface rows disagree with geometry-state report")
    if require_int(visible_surface["surface_frame_rows"], f"{inputs.case} visible surface rows") != require_int(
        geometry_source_audit["multi_object_visible_surface_rows"],
        f"{inputs.case} audit visible-surface rows",
    ):
        raise RuntimeError(f"{inputs.case} visible-surface rows disagree with geometry-source audit")
    if require_int(visible_surface["surface_frame_rows"], f"{inputs.case} visible surface rows") != require_int(
        object_geometry_hypothesis_state["visible_surface_frame_rows"],
        f"{inputs.case} object-geometry hypothesis visible-surface rows",
    ):
        raise RuntimeError(f"{inputs.case} visible-surface rows disagree with object-geometry hypothesis state")
    if require_int(visible_surface["surface_frame_rows"], f"{inputs.case} visible surface rows") != require_int(
        object_geometry_factor_problem["visible_surface_factor_rows"],
        f"{inputs.case} object-geometry factor visible-surface rows",
    ):
        raise RuntimeError(f"{inputs.case} visible-surface rows disagree with object-geometry factor problem")
    if require_int(
        visible_surface["rejected_visible_object_frame_rows"],
        f"{inputs.case} visible-surface rejected rows",
    ) != require_int(
        geometry_source_audit["multi_object_visible_surface_rejected_rows"],
        f"{inputs.case} audit visible-surface rejected rows",
    ):
        raise RuntimeError(f"{inputs.case} visible-surface rejections disagree with geometry-source audit")
    if require_int(sparse["object_variable_frames"], f"{inputs.case} sparse object variable frames") != require_int(
        geometry_source_audit["legacy_single_stream_object_variable_frames"],
        f"{inputs.case} audit legacy object variable frames",
    ):
        raise RuntimeError(f"{inputs.case} sparse object variables disagree with geometry-source audit")
    if require_int(mesh["mesh_frames"], f"{inputs.case} mesh frames") != require_int(
        geometry_source_audit["legacy_single_stream_mesh_frames"],
        f"{inputs.case} audit mesh frames",
    ):
        raise RuntimeError(f"{inputs.case} mesh frames disagree with geometry-source audit")
    if require_int(mesh["missing_mesh_frame_count"], f"{inputs.case} missing mesh frames") != require_int(
        geometry_source_audit["legacy_single_stream_missing_mesh_frame_count"],
        f"{inputs.case} audit missing mesh frames",
    ):
        raise RuntimeError(f"{inputs.case} missing mesh frames disagree with geometry-source audit")
    if require_int(contact["contact_factor_ready_count"], f"{inputs.case} contact ready rows") != require_int(
        geometry_source_audit["contact_mode_factor_ready_rows"],
        f"{inputs.case} audit contact ready rows",
    ):
        raise RuntimeError(f"{inputs.case} contact ready rows disagree with geometry-source audit")
    if require_int(
        multi_object_contact_evidence["hand_object_rows"],
        f"{inputs.case} multi-object contact hand-object rows",
    ) != require_int(
        geometry_source_audit["multi_object_hand_object_rows"],
        f"{inputs.case} audit multi-object hand-object rows",
    ):
        raise RuntimeError(f"{inputs.case} hand-object rows disagree with geometry-source audit")
    if require_int(
        multi_object_contact_evidence["measured_distance_rows"],
        f"{inputs.case} multi-object measured rows",
    ) != require_int(
        geometry_source_audit["multi_object_measured_distance_rows"],
        f"{inputs.case} audit measured rows",
    ):
        raise RuntimeError(f"{inputs.case} measured contact rows disagree with geometry-source audit")
    if require_int(
        multi_object_contact_evidence["unobserved_rows"],
        f"{inputs.case} multi-object unobserved rows",
    ) != require_int(
        geometry_source_audit["multi_object_unobserved_rows"],
        f"{inputs.case} audit unobserved rows",
    ):
        raise RuntimeError(f"{inputs.case} unobserved contact rows disagree with geometry-source audit")
    if require_int(
        multi_object_contact_evidence["visible_surface_distance_candidate_rows"],
        f"{inputs.case} multi-object visible distance candidates",
    ) != require_int(
        geometry_source_audit["multi_object_visible_surface_distance_candidate_rows"],
        f"{inputs.case} audit visible distance candidates",
    ):
        raise RuntimeError(f"{inputs.case} visible-distance candidates disagree with geometry-source audit")
    if require_int(
        multi_object_contact_evidence["contact_factor_ready_rows"],
        f"{inputs.case} multi-object contact factor rows",
    ) != require_int(
        geometry_source_audit["multi_object_contact_factor_ready_rows"],
        f"{inputs.case} audit multi-object contact factor rows",
    ):
        raise RuntimeError(f"{inputs.case} multi-object contact factors disagree with geometry-source audit")
    if require_int(
        multi_object_contact_evidence["contact_factor_ready_rows"],
        f"{inputs.case} multi-object contact factor rows",
    ) != require_int(
        object_geometry_factor_problem["multi_object_contact_factor_ready_rows"],
        f"{inputs.case} object-geometry factor contact factor rows",
    ):
        raise RuntimeError(f"{inputs.case} multi-object contact factors disagree with object-geometry factor problem")
    if require_int(
        object_track_dataset["total_exported_frames"],
        f"{inputs.case} object-track dataset exported frames",
    ) != require_int(
        object_material_track["dataset_exported_frames"],
        f"{inputs.case} material-track dataset exported frames",
    ):
        raise RuntimeError(f"{inputs.case} object-track dataset frame count disagrees with material-track summary")
    if require_int(
        object_material_track["material_track_window_count"],
        f"{inputs.case} material-track window count",
    ) != require_int(
        object_material_motion_state["material_track_window_count"],
        f"{inputs.case} material-motion window count",
    ):
        raise RuntimeError(f"{inputs.case} material-track window count disagrees with material-motion report")
    if require_int(
        object_material_track["rigid_factor_ready_pair_count"],
        f"{inputs.case} material-track ready pair count",
    ) != require_int(
        object_material_motion_state["rigid_factor_ready_pair_count"],
        f"{inputs.case} material-motion ready pair count",
    ):
        raise RuntimeError(f"{inputs.case} material-track ready pair count disagrees with material-motion report")
    if require_int(
        object_material_track["rigid_factor_ready_pair_count"],
        f"{inputs.case} material-track ready pair count",
    ) != require_int(
        object_geometry_factor_problem["material_rigidity_pair_factor_count"],
        f"{inputs.case} object-geometry factor material pair count",
    ):
        raise RuntimeError(f"{inputs.case} material-track ready pair count disagrees with object-geometry factor problem")
    if require_int(
        object_material_motion_state["material_track_window_count"],
        f"{inputs.case} material-motion window count",
    ) != require_int(
        object_material_pose_candidate["material_track_window_count"],
        f"{inputs.case} material-pose window count",
    ):
        raise RuntimeError(f"{inputs.case} material-motion window count disagrees with material-pose report")
    if require_int(
        object_material_motion_state["persistent_window_motion_candidate_count"],
        f"{inputs.case} material-motion persistent candidate count",
    ) != require_int(
        object_material_pose_candidate["persistent_window_motion_candidate_count"],
        f"{inputs.case} material-pose persistent candidate count",
    ):
        raise RuntimeError(f"{inputs.case} material-motion persistent candidate count disagrees with material-pose report")
    if require_int(
        object_material_pose_candidate["partial_material_pose_candidate_segment_count"],
        f"{inputs.case} material-pose candidate segment count",
    ) != require_int(
        object_material_surface_replay["partial_material_pose_candidate_segment_count"],
        f"{inputs.case} material-surface candidate segment count",
    ):
        raise RuntimeError(f"{inputs.case} material-pose candidate count disagrees with surface replay report")
    if require_int(
        object_material_pose_candidate["partial_material_pose_candidate_ready_segment_count"],
        f"{inputs.case} material-pose ready segment count",
    ) != require_int(
        object_material_surface_replay["partial_material_pose_candidate_ready_segment_count"],
        f"{inputs.case} material-surface pose ready segment count",
    ):
        raise RuntimeError(f"{inputs.case} material-pose ready count disagrees with surface replay report")
    if require_int(
        object_material_pose_candidate["partial_material_pose_candidate_ready_segment_count"],
        f"{inputs.case} material-pose ready segment count",
    ) != require_int(
        object_geometry_factor_problem["partial_material_pose_ready_segment_count"],
        f"{inputs.case} object-geometry factor material-pose ready segment count",
    ):
        raise RuntimeError(f"{inputs.case} material-pose ready count disagrees with object-geometry factor problem")
    if require_int(
        object_material_surface_replay["partial_visible_surface_replay_candidate_count"],
        f"{inputs.case} material-surface candidate count",
    ) != require_int(
        geometry_source_audit["partial_visible_surface_replay_candidate_count"],
        f"{inputs.case} audit material-surface candidate count",
    ):
        raise RuntimeError(f"{inputs.case} material-surface candidate count disagrees with geometry-source audit")
    if require_int(
        object_material_surface_replay["partial_visible_surface_replay_ready_count"],
        f"{inputs.case} material-surface ready count",
    ) != require_int(
        geometry_source_audit["partial_visible_surface_replay_ready_count"],
        f"{inputs.case} audit material-surface ready count",
    ):
        raise RuntimeError(f"{inputs.case} material-surface ready count disagrees with geometry-source audit")
    if require_int(
        object_material_surface_replay["partial_visible_surface_replay_ready_count"],
        f"{inputs.case} material-surface ready count",
    ) != require_int(
        object_geometry_factor_problem["partial_visible_surface_replay_ready_segment_count"],
        f"{inputs.case} object-geometry factor surface replay ready count",
    ):
        raise RuntimeError(f"{inputs.case} material-surface ready count disagrees with object-geometry factor problem")
    if require_int(
        geometry_source_audit["source_incompatibility_count"],
        f"{inputs.case} audit source incompatibility count",
    ) != require_int(
        object_geometry_hypothesis_state["source_incompatibility_count"],
        f"{inputs.case} object-geometry hypothesis source incompatibility count",
    ):
        raise RuntimeError(f"{inputs.case} source incompatibility count disagrees with object-geometry hypothesis state")
    if require_int(
        geometry_source_audit["local_patch_visible_surface_conflict_count"],
        f"{inputs.case} audit local patch conflict count",
    ) != require_int(
        object_geometry_factor_problem["geometry_source_conflict_count"],
        f"{inputs.case} object-geometry factor source conflict count",
    ):
        raise RuntimeError(f"{inputs.case} source conflict count disagrees with object-geometry factor problem")
    if require_int(
        object_geometry_hypothesis_state["complete_object_geometry_hypothesis_count"],
        f"{inputs.case} object-geometry hypothesis complete count",
    ) != require_int(
        object_geometry_factor_problem["complete_object_geometry_hypothesis_count"],
        f"{inputs.case} object-geometry factor complete count",
    ):
        raise RuntimeError(f"{inputs.case} complete object-geometry count disagrees with factor problem")
    if require_int(
        object_geometry_hypothesis_state["contact_compatible_object_geometry_hypothesis_count"],
        f"{inputs.case} object-geometry hypothesis contact-compatible count",
    ) != require_int(
        object_geometry_factor_problem["contact_compatible_object_geometry_hypothesis_count"],
        f"{inputs.case} object-geometry factor contact-compatible count",
    ):
        raise RuntimeError(f"{inputs.case} contact-compatible object-geometry count disagrees with factor problem")
    if require_int(
        object_geometry_hypothesis_state["object_pose_factor_ready_hypothesis_count"],
        f"{inputs.case} object-geometry hypothesis pose-ready count",
    ) != require_int(
        object_geometry_factor_problem["object_pose_factor_ready_hypothesis_count"],
        f"{inputs.case} object-geometry factor pose-ready count",
    ):
        raise RuntimeError(f"{inputs.case} pose-ready object-geometry count disagrees with factor problem")
    if require_int(
        geometry_reconstruction_jobs["job_count"],
        f"{inputs.case} geometry reconstruction job count",
    ) != require_int(
        object_geometry_factor_problem["geometry_reconstruction_job_count"],
        f"{inputs.case} object-geometry factor reconstruction job count",
    ):
        raise RuntimeError(f"{inputs.case} geometry reconstruction job count disagrees with factor problem")
    if require_int(
        geometry_reconstruction_jobs["solver_job_ready_count"],
        f"{inputs.case} geometry reconstruction solver-ready job count",
    ) != require_int(
        object_geometry_factor_problem["geometry_reconstruction_solver_job_ready_count"],
        f"{inputs.case} object-geometry factor solver-ready reconstruction job count",
    ):
        raise RuntimeError(f"{inputs.case} solver-ready reconstruction job count disagrees with factor problem")
    if require_int(
        geometry_reconstruction_jobs["hidden_topology_reconstructed_job_count"],
        f"{inputs.case} geometry reconstruction hidden topology job count",
    ) != require_int(
        object_geometry_factor_problem["geometry_reconstruction_hidden_topology_job_count"],
        f"{inputs.case} object-geometry factor hidden topology job count",
    ):
        raise RuntimeError(f"{inputs.case} hidden topology reconstruction count disagrees with factor problem")
    if require_int(
        geometry_reconstruction_jobs["job_count"],
        f"{inputs.case} geometry reconstruction job count",
    ) != require_int(
        geometry_reconstruction_results["job_count"],
        f"{inputs.case} geometry reconstruction result job count",
    ):
        raise RuntimeError(f"{inputs.case} reconstruction result job count disagrees with job inputs")
    if require_int(
        geometry_reconstruction_results["job_count"],
        f"{inputs.case} geometry reconstruction result job count",
    ) != require_int(
        object_geometry_factor_problem["geometry_reconstruction_result_job_count"],
        f"{inputs.case} object-geometry factor reconstruction result job count",
    ):
        raise RuntimeError(f"{inputs.case} reconstruction result job count disagrees with factor problem")
    if require_int(
        geometry_reconstruction_results["hidden_topology_reconstructed_job_count"],
        f"{inputs.case} geometry reconstruction result hidden topology count",
    ) != require_int(
        object_geometry_factor_problem["geometry_reconstruction_result_hidden_topology_job_count"],
        f"{inputs.case} object-geometry factor result hidden topology count",
    ):
        raise RuntimeError(f"{inputs.case} reconstruction result hidden topology count disagrees with factor problem")
    if require_int(
        geometry_reconstruction_results["accepted_reconstruction_result_count"],
        f"{inputs.case} accepted reconstruction result count",
    ) != require_int(
        object_geometry_factor_problem["geometry_reconstruction_accepted_result_count"],
        f"{inputs.case} object-geometry factor accepted reconstruction result count",
    ):
        raise RuntimeError(f"{inputs.case} accepted reconstruction result count disagrees with factor problem")
    if require_int(
        geometry_reconstruction_results["accepted_reconstruction_result_count"],
        f"{inputs.case} accepted reconstruction result count",
    ) != require_int(
        depth_contact_consistency["accepted_reconstruction_job_count"],
        f"{inputs.case} depth-contact accepted reconstruction job count",
    ):
        raise RuntimeError(f"{inputs.case} accepted reconstruction count disagrees with depth-contact audit")
    if require_int(
        depth_contact_consistency["evaluated_frame_count"],
        f"{inputs.case} depth-contact evaluated frame count",
    ) != require_int(
        object_geometry_factor_problem["depth_contact_evaluated_frame_count"],
        f"{inputs.case} object-geometry factor depth-contact frame count",
    ):
        raise RuntimeError(f"{inputs.case} depth-contact evaluated frame count disagrees with factor problem")
    if require_int(
        depth_contact_consistency["depth_owner_incompatibility_count"],
        f"{inputs.case} depth-contact incompatibility count",
    ) != require_int(
        object_geometry_factor_problem["depth_contact_owner_incompatibility_count"],
        f"{inputs.case} object-geometry factor depth-contact incompatibility count",
    ):
        raise RuntimeError(f"{inputs.case} depth-contact incompatibility count disagrees with factor problem")

    raw_video = require_dict(load_json(Path(require_str(manifest.get("manifest"), "v16 manifest path"))).get("raw_video"), "raw_video")
    raw_frame_count = require_int(raw_video.get("frame_count"), f"{inputs.case} raw_video.frame_count")
    if raw_frame_count != frame_count:
        raise RuntimeError(f"{inputs.case} raw frame count {raw_frame_count} differs from graph frame count {frame_count}")
    finite_number(raw_video.get("fps"), f"{inputs.case} raw fps")

    families = required_variable_families(
        roster,
        timeline,
        visible_surface,
        geometry_state,
        object_track_dataset,
        object_material_track,
        object_material_motion_state,
        object_material_pose_candidate,
        object_material_surface_replay,
        multi_object_contact_evidence,
        geometry_source_audit,
        object_geometry_hypothesis_state,
        object_geometry_factor_problem,
        geometry_reconstruction_jobs,
        geometry_reconstruction_results,
        depth_contact_consistency,
        counts,
        sparse,
        contact,
        mesh,
    )
    unmet = [family["family"] for family in families if not bool(family["v3_requirement_met"])]
    return {
        "case": inputs.case,
        "status": STATUS,
        "claim": CLAIM,
        "frame_count": frame_count,
        "raw_video": raw_video,
        "sources": {
            "measurement_manifest": source_summary(inputs.manifest, manifest),
            "object_roster": {"path": str(inputs.object_roster), "row_count": roster["roster_row_count"]},
            "multi_object_timeline": source_summary(inputs.multi_object_timeline, multi_object_timeline),
            "multi_object_visible_surface_report": source_summary(
                inputs.visible_surface_report, visible_surface_report
            ),
            "multi_object_geometry_state_report": source_summary(
                inputs.geometry_state_report, geometry_state_report
            ),
            "object_track_dataset_summary": source_summary(
                inputs.object_track_dataset_summary, object_track_dataset_summary
            ),
            "object_material_track_summary": source_summary(
                inputs.object_material_track_summary, object_material_track_summary
            ),
            "object_material_motion_state_report": source_summary(
                inputs.object_material_motion_state_summary, object_material_motion_state_summary
            ),
            "object_material_pose_candidate_report": source_summary(
                inputs.object_material_pose_candidate_summary, object_material_pose_candidate_summary
            ),
            "object_material_surface_replay_report": source_summary(
                inputs.object_material_surface_replay_summary, object_material_surface_replay_summary
            ),
            "multi_object_contact_evidence_report": source_summary(
                inputs.multi_object_contact_evidence_summary, multi_object_contact_evidence_summary
            ),
            "geometry_source_audit_report": source_summary(
                inputs.geometry_source_audit_report, geometry_source_audit_report
            ),
            "object_geometry_hypothesis_state_report": source_summary(
                inputs.object_geometry_hypothesis_state_report, object_geometry_hypothesis_state_report
            ),
            "object_geometry_factor_problem_report": source_summary(
                inputs.object_geometry_factor_problem_report, object_geometry_factor_problem_report
            ),
            "geometry_reconstruction_jobs_report": source_summary(
                inputs.geometry_reconstruction_jobs_report, geometry_reconstruction_jobs_report
            ),
            "geometry_reconstruction_results_report": source_summary(
                inputs.geometry_reconstruction_results_report, geometry_reconstruction_results_report
            ),
            "depth_contact_consistency_audit_report": source_summary(
                inputs.depth_contact_consistency_audit_report, depth_contact_consistency_audit_report
            ),
            "sparse_graph_report": source_summary(inputs.sparse_report, sparse_report),
            "contact_mode_report": source_summary(inputs.contact_mode_report, contact_report),
            "mesh_metadata": source_summary(inputs.mesh_metadata, mesh_metadata),
        },
        "current_sparse_graph": sparse,
        "current_contact_mode_graph": contact,
        "current_multi_object_timeline": timeline,
        "current_multi_object_visible_surfaces": visible_surface,
        "current_multi_object_geometry_state": geometry_state,
        "current_object_track_datasets": object_track_dataset,
        "current_object_material_tracks": object_material_track,
        "current_object_material_motion_state": object_material_motion_state,
        "current_object_material_pose_candidates": object_material_pose_candidate,
        "current_object_material_surface_replay": object_material_surface_replay,
        "current_multi_object_contact_evidence": multi_object_contact_evidence,
        "current_geometry_source_audit": geometry_source_audit,
        "current_object_geometry_hypothesis_state": object_geometry_hypothesis_state,
        "current_object_geometry_factor_problem": object_geometry_factor_problem,
        "current_geometry_reconstruction_jobs": geometry_reconstruction_jobs,
        "current_geometry_reconstruction_results": geometry_reconstruction_results,
        "current_depth_contact_consistency_audit": depth_contact_consistency,
        "current_mesh_archive": mesh,
        "current_measurement_counts": counts,
        "object_roster_audit": roster,
        "required_variable_families": families,
        "unmet_required_variable_families": unmet,
        "missing_or_incomplete_required_variable_families": unmet,
        "v3_solver_complete": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "next_solver_owner": (
            "A V17 optimizer must create variables for the missing families above or explicitly keep a family fixed "
            "with a source-backed scientific reason. A sparse graph over one legacy object stream cannot close the task."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.measurement_store_root / "v17_measurement_store_summary.json",
        "measurement store summary",
    )
    summary = require_dict(load_json(summary_path), "measurement store summary")
    cases = require_list(summary.get("cases"), "measurement store summary cases")
    case_outputs: list[dict[str, Any]] = []
    for i, case_row in enumerate(cases):
        inputs = case_inputs(
            require_dict(case_row, f"measurement store summary case {i}"),
            args.measurement_store_root,
            args.multi_object_timeline_root,
            args.visible_surface_root,
            args.geometry_state_root,
            args.object_track_dataset_root,
            args.object_material_track_root,
            args.object_material_motion_state_root,
            args.object_material_pose_candidate_root,
            args.object_material_surface_replay_root,
            args.multi_object_contact_evidence_root,
            args.geometry_source_audit_root,
            args.object_geometry_hypothesis_state_root,
            args.object_geometry_factor_problem_root,
            args.geometry_reconstruction_jobs_root,
            args.geometry_reconstruction_results_root,
            args.depth_contact_consistency_audit_root,
            args.sparse_graph_root,
            args.contact_mode_graph_root,
        )
        problem = case_problem(inputs)
        write_json(args.output_root / inputs.case / "v17_joint_solver_problem.json", problem)
        case_outputs.append(problem)

    union_unmet = sorted(
        {family for case in case_outputs for family in case["unmet_required_variable_families"]}
    )
    payload = {
        "method": "build_v17_joint_solver_problem",
        "status": STATUS,
        "claim": CLAIM,
        "measurement_store_summary": str(summary_path),
        "sparse_graph_root": str(args.sparse_graph_root),
        "contact_mode_graph_root": str(args.contact_mode_graph_root),
        "multi_object_visible_surface_root": str(args.visible_surface_root),
        "multi_object_geometry_state_root": str(args.geometry_state_root),
        "object_track_dataset_root": str(args.object_track_dataset_root),
        "object_material_track_root": str(args.object_material_track_root),
        "object_material_motion_state_root": str(args.object_material_motion_state_root),
        "object_material_pose_candidate_root": str(args.object_material_pose_candidate_root),
        "object_material_surface_replay_root": str(args.object_material_surface_replay_root),
        "multi_object_contact_evidence_root": str(args.multi_object_contact_evidence_root),
        "geometry_source_audit_root": str(args.geometry_source_audit_root),
        "object_geometry_hypothesis_state_root": str(args.object_geometry_hypothesis_state_root),
        "object_geometry_factor_problem_root": str(args.object_geometry_factor_problem_root),
        "geometry_reconstruction_jobs_root": str(args.geometry_reconstruction_jobs_root),
        "geometry_reconstruction_results_root": str(args.geometry_reconstruction_results_root),
        "depth_contact_consistency_audit_root": str(args.depth_contact_consistency_audit_root),
        "case_count": len(case_outputs),
        "cases": [
            {
                "case": case["case"],
                "problem_path": str(args.output_root / case["case"] / "v17_joint_solver_problem.json"),
                "frame_count": case["frame_count"],
                "active_vlm_object_count": case["object_roster_audit"]["active_vlm_object_count"],
                "multi_object_frame_rows": case["current_multi_object_timeline"]["object_frame_rows"],
                "visible_mask_frame_rows": case["current_multi_object_timeline"]["visible_mask_frame_rows"],
                "multi_object_visible_surface_rows": case[
                    "current_multi_object_visible_surfaces"
                ]["surface_frame_rows"],
                "multi_object_visible_surface_rejected_rows": case[
                    "current_multi_object_visible_surfaces"
                ]["rejected_visible_object_frame_rows"],
                "visible_surface_envelope_candidate_count": case[
                    "current_multi_object_geometry_state"
                ]["visible_surface_envelope_candidate_count"],
                "rigid_pose_candidate_count": case["current_multi_object_geometry_state"][
                    "rigid_pose_candidate_count"
                ],
                "object_track_dataset_exported_frames": case["current_object_track_datasets"][
                    "total_exported_frames"
                ],
                "object_track_dataset_exported_objects": case["current_object_track_datasets"][
                    "exported_object_count"
                ],
                "material_track_window_count": case["current_object_material_tracks"][
                    "material_track_window_count"
                ],
                "material_tracked_object_count": case["current_object_material_tracks"][
                    "material_tracked_object_count"
                ],
                "rigid_motion_ready_window_count": case["current_object_material_tracks"][
                    "rigid_motion_ready_window_count"
                ],
                "rigid_factor_ready_pair_count": case["current_object_material_tracks"][
                    "rigid_factor_ready_pair_count"
                ],
                "persistent_window_motion_candidate_count": case[
                    "current_object_material_motion_state"
                ]["persistent_window_motion_candidate_count"],
                "local_adjacent_material_motion_window_count": case[
                    "current_object_material_motion_state"
                ]["local_adjacent_material_motion_window_count"],
                "noncandidate_local_adjacent_material_motion_window_count": case[
                    "current_object_material_motion_state"
                ]["noncandidate_local_adjacent_material_motion_window_count"],
                "no_ready_material_motion_window_count": case[
                    "current_object_material_motion_state"
                ]["no_ready_material_motion_window_count"],
                "partial_material_pose_candidate_segment_count": case[
                    "current_object_material_pose_candidates"
                ]["partial_material_pose_candidate_segment_count"],
                "partial_material_pose_candidate_ready_segment_count": case[
                    "current_object_material_pose_candidates"
                ]["partial_material_pose_candidate_ready_segment_count"],
                "partial_visible_surface_replay_candidate_count": case[
                    "current_object_material_surface_replay"
                ]["partial_visible_surface_replay_candidate_count"],
                "partial_visible_surface_replay_ready_count": case[
                    "current_object_material_surface_replay"
                ]["partial_visible_surface_replay_ready_count"],
                "multi_object_hand_object_rows": case[
                    "current_multi_object_contact_evidence"
                ]["hand_object_rows"],
                "multi_object_measured_distance_rows": case[
                    "current_multi_object_contact_evidence"
                ]["measured_distance_rows"],
                "multi_object_unobserved_rows": case[
                    "current_multi_object_contact_evidence"
                ]["unobserved_rows"],
                "multi_object_contact_factor_ready_rows": case[
                    "current_multi_object_contact_evidence"
                ]["contact_factor_ready_rows"],
                "geometry_source_incompatibility_count": case[
                    "current_geometry_source_audit"
                ]["source_incompatibility_count"],
                "local_patch_visible_surface_conflict_count": case[
                    "current_geometry_source_audit"
                ]["local_patch_visible_surface_conflict_count"],
                "contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate": case[
                    "current_geometry_source_audit"
                ]["contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate"],
                "unified_object_geometry_source_ready": case[
                    "current_geometry_source_audit"
                ]["unified_object_geometry_source_ready"],
                "contact_factor_source_compatible_with_multi_object_geometry": case[
                    "current_geometry_source_audit"
                ]["contact_factor_source_compatible_with_multi_object_geometry"],
                "object_pose_source_compatible_with_contact_factors": case[
                    "current_geometry_source_audit"
                ]["object_pose_source_compatible_with_contact_factors"],
                "object_geometry_hypothesis_state_counts": case[
                    "current_object_geometry_hypothesis_state"
                ]["state_counts"],
                "objects_with_accepted_reconstruction_results": case[
                    "current_object_geometry_hypothesis_state"
                ]["objects_with_accepted_reconstruction_results"],
                "hypothesis_accepted_reconstruction_result_count": case[
                    "current_object_geometry_hypothesis_state"
                ]["accepted_reconstruction_result_count"],
                "complete_object_geometry_hypothesis_count": case[
                    "current_object_geometry_hypothesis_state"
                ]["complete_object_geometry_hypothesis_count"],
                "contact_compatible_object_geometry_hypothesis_count": case[
                    "current_object_geometry_hypothesis_state"
                ]["contact_compatible_object_geometry_hypothesis_count"],
                "object_pose_factor_ready_hypothesis_count": case[
                    "current_object_geometry_hypothesis_state"
                ]["object_pose_factor_ready_hypothesis_count"],
                "object_geometry_factor_problem_rows": case[
                    "current_object_geometry_factor_problem"
                ]["factor_problem_object_rows"],
                "object_geometry_factor_solve_activation_ready_object_count": case[
                    "current_object_geometry_factor_problem"
                ]["solve_activation_ready_object_count"],
                "object_geometry_factor_visible_surface_rows": case[
                    "current_object_geometry_factor_problem"
                ]["visible_surface_factor_rows"],
                "object_geometry_factor_material_rigidity_pair_count": case[
                    "current_object_geometry_factor_problem"
                ]["material_rigidity_pair_factor_count"],
                "object_geometry_factor_partial_pose_ready_segment_count": case[
                    "current_object_geometry_factor_problem"
                ]["partial_material_pose_ready_segment_count"],
                "object_geometry_factor_surface_replay_ready_segment_count": case[
                    "current_object_geometry_factor_problem"
                ]["partial_visible_surface_replay_ready_segment_count"],
                "object_geometry_factor_observed_surface_seed_count": case[
                    "current_object_geometry_factor_problem"
                ]["observed_surface_geometry_seed_count"],
                "object_geometry_factor_observed_surface_seed_vertices": case[
                    "current_object_geometry_factor_problem"
                ]["observed_surface_geometry_seed_vertices"],
                "object_geometry_factor_observed_surface_seed_faces": case[
                    "current_object_geometry_factor_problem"
                ]["observed_surface_geometry_seed_faces"],
                "geometry_reconstruction_job_count": case[
                    "current_geometry_reconstruction_jobs"
                ]["job_count"],
                "geometry_reconstruction_solver_job_ready_count": case[
                    "current_geometry_reconstruction_jobs"
                ]["solver_job_ready_count"],
                "geometry_reconstruction_hidden_topology_job_count": case[
                    "current_geometry_reconstruction_jobs"
                ]["hidden_topology_reconstructed_job_count"],
                "geometry_reconstruction_rectification_residual_p95_m": case[
                    "current_geometry_reconstruction_jobs"
                ]["rectification_nearest_3d_residual_p95_m"],
                "geometry_reconstruction_pending_solver_output_count": case[
                    "current_geometry_reconstruction_results"
                ]["pending_solver_output_count"],
                "geometry_reconstruction_solver_output_detected_count": case[
                    "current_geometry_reconstruction_results"
                ]["solver_output_detected_count"],
                "geometry_reconstruction_mesh_file_detected_count": case[
                    "current_geometry_reconstruction_results"
                ]["mesh_file_detected_count"],
                "geometry_reconstruction_pose_sequence_complete_count": case[
                    "current_geometry_reconstruction_results"
                ]["pose_sequence_complete_count"],
                "geometry_reconstruction_mesh_scale_plausible_count": case[
                    "current_geometry_reconstruction_results"
                ]["mesh_scale_plausible_count"],
                "geometry_reconstruction_mesh_projection_qc_passed_count": case[
                    "current_geometry_reconstruction_results"
                ]["mesh_projection_qc_passed_count"],
                "geometry_reconstruction_accepted_result_count": case[
                    "current_geometry_reconstruction_results"
                ]["accepted_reconstruction_result_count"],
                "depth_contact_evaluated_frame_count": case[
                    "current_depth_contact_consistency_audit"
                ]["evaluated_frame_count"],
                "depth_contact_evaluated_hand_rows": case[
                    "current_depth_contact_consistency_audit"
                ]["evaluated_hand_rows"],
                "depth_contact_near_reconstructed_mesh_hand_rows": case[
                    "current_depth_contact_consistency_audit"
                ]["near_reconstructed_mesh_hand_rows"],
                "depth_contact_reconstructed_mesh_contact_candidate_rows": case[
                    "current_depth_contact_consistency_audit"
                ]["reconstructed_mesh_contact_candidate_rows"],
                "depth_contact_shared_depth_state_ready_frame_count": case[
                    "current_depth_contact_consistency_audit"
                ]["shared_depth_state_ready_frame_count"],
                "depth_contact_owner_incompatibility_count": case[
                    "current_depth_contact_consistency_audit"
                ]["depth_owner_incompatibility_count"],
                "object_geometry_factor_contact_ready_rows": case[
                    "current_object_geometry_factor_problem"
                ]["multi_object_contact_factor_ready_rows"],
                "current_single_stream_object_variable_frames": case["current_sparse_graph"]["object_variable_frames"],
                "contact_factor_ready_count": case["current_contact_mode_graph"]["contact_factor_ready_count"],
                "unmet_required_variable_families": case["unmet_required_variable_families"],
                "missing_or_incomplete_required_variable_families": case[
                    "missing_or_incomplete_required_variable_families"
                ],
                "v3_solver_complete": False,
                "annotation_ready": False,
                "deliverable_ready": False,
                "accuracy_target_met": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "rigid_pose_requirement_met": False,
            }
            for case in case_outputs
        ],
        "unmet_required_variable_families_union": union_unmet,
        "missing_or_incomplete_required_variable_families_union": union_unmet,
        "geometry_source_incompatibility_count": sum(
            case["current_geometry_source_audit"]["source_incompatibility_count"] for case in case_outputs
        ),
        "contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate": sum(
            case["current_geometry_source_audit"][
                "contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate"
            ]
            for case in case_outputs
        ),
        "unified_object_geometry_source_ready": False,
        "contact_factor_source_compatible_with_multi_object_geometry": False,
        "object_pose_source_compatible_with_contact_factors": False,
        "complete_object_geometry_hypothesis_count": sum(
            case["current_object_geometry_hypothesis_state"]["complete_object_geometry_hypothesis_count"]
            for case in case_outputs
        ),
        "objects_with_accepted_reconstruction_results": sum(
            case["current_object_geometry_hypothesis_state"]["objects_with_accepted_reconstruction_results"]
            for case in case_outputs
        ),
        "hypothesis_accepted_reconstruction_result_count": sum(
            case["current_object_geometry_hypothesis_state"]["accepted_reconstruction_result_count"]
            for case in case_outputs
        ),
        "contact_compatible_object_geometry_hypothesis_count": sum(
            case["current_object_geometry_hypothesis_state"]["contact_compatible_object_geometry_hypothesis_count"]
            for case in case_outputs
        ),
        "object_pose_factor_ready_hypothesis_count": sum(
            case["current_object_geometry_hypothesis_state"]["object_pose_factor_ready_hypothesis_count"]
            for case in case_outputs
        ),
        "object_geometry_factor_problem_rows": sum(
            case["current_object_geometry_factor_problem"]["factor_problem_object_rows"]
            for case in case_outputs
        ),
        "object_geometry_factor_solve_activation_ready_object_count": sum(
            case["current_object_geometry_factor_problem"]["solve_activation_ready_object_count"]
            for case in case_outputs
        ),
        "object_geometry_factor_visible_surface_rows": sum(
            case["current_object_geometry_factor_problem"]["visible_surface_factor_rows"]
            for case in case_outputs
        ),
        "object_geometry_factor_material_rigidity_pair_count": sum(
            case["current_object_geometry_factor_problem"]["material_rigidity_pair_factor_count"]
            for case in case_outputs
        ),
        "object_geometry_factor_partial_pose_ready_segment_count": sum(
            case["current_object_geometry_factor_problem"]["partial_material_pose_ready_segment_count"]
            for case in case_outputs
        ),
        "object_geometry_factor_surface_replay_ready_segment_count": sum(
            case["current_object_geometry_factor_problem"]["partial_visible_surface_replay_ready_segment_count"]
            for case in case_outputs
        ),
        "object_geometry_factor_observed_surface_seed_count": sum(
            case["current_object_geometry_factor_problem"]["observed_surface_geometry_seed_count"]
            for case in case_outputs
        ),
        "object_geometry_factor_observed_surface_seed_vertices": sum(
            case["current_object_geometry_factor_problem"]["observed_surface_geometry_seed_vertices"]
            for case in case_outputs
        ),
        "object_geometry_factor_observed_surface_seed_faces": sum(
            case["current_object_geometry_factor_problem"]["observed_surface_geometry_seed_faces"]
            for case in case_outputs
        ),
        "geometry_reconstruction_job_count": sum(
            case["current_geometry_reconstruction_jobs"]["job_count"] for case in case_outputs
        ),
        "geometry_reconstruction_solver_job_ready_count": sum(
            case["current_geometry_reconstruction_jobs"]["solver_job_ready_count"] for case in case_outputs
        ),
        "geometry_reconstruction_hidden_topology_job_count": sum(
            case["current_geometry_reconstruction_jobs"]["hidden_topology_reconstructed_job_count"]
            for case in case_outputs
        ),
        "geometry_reconstruction_pending_solver_output_count": sum(
            case["current_geometry_reconstruction_results"]["pending_solver_output_count"]
            for case in case_outputs
        ),
        "geometry_reconstruction_solver_output_detected_count": sum(
            case["current_geometry_reconstruction_results"]["solver_output_detected_count"]
            for case in case_outputs
        ),
        "geometry_reconstruction_mesh_file_detected_count": sum(
            case["current_geometry_reconstruction_results"]["mesh_file_detected_count"]
            for case in case_outputs
        ),
        "geometry_reconstruction_pose_sequence_complete_count": sum(
            case["current_geometry_reconstruction_results"]["pose_sequence_complete_count"]
            for case in case_outputs
        ),
        "geometry_reconstruction_mesh_scale_plausible_count": sum(
            case["current_geometry_reconstruction_results"]["mesh_scale_plausible_count"]
            for case in case_outputs
        ),
        "geometry_reconstruction_mesh_projection_qc_passed_count": sum(
            case["current_geometry_reconstruction_results"]["mesh_projection_qc_passed_count"]
            for case in case_outputs
        ),
        "geometry_reconstruction_accepted_result_count": sum(
            case["current_geometry_reconstruction_results"]["accepted_reconstruction_result_count"]
            for case in case_outputs
        ),
        "depth_contact_evaluated_frame_count": sum(
            case["current_depth_contact_consistency_audit"]["evaluated_frame_count"] for case in case_outputs
        ),
        "depth_contact_evaluated_hand_rows": sum(
            case["current_depth_contact_consistency_audit"]["evaluated_hand_rows"] for case in case_outputs
        ),
        "depth_contact_near_reconstructed_mesh_hand_rows": sum(
            case["current_depth_contact_consistency_audit"]["near_reconstructed_mesh_hand_rows"]
            for case in case_outputs
        ),
        "depth_contact_reconstructed_mesh_contact_candidate_rows": sum(
            case["current_depth_contact_consistency_audit"]["reconstructed_mesh_contact_candidate_rows"]
            for case in case_outputs
        ),
        "depth_contact_shared_depth_state_ready_frame_count": sum(
            case["current_depth_contact_consistency_audit"]["shared_depth_state_ready_frame_count"]
            for case in case_outputs
        ),
        "depth_contact_owner_incompatibility_count": sum(
            case["current_depth_contact_consistency_audit"]["depth_owner_incompatibility_count"]
            for case in case_outputs
        ),
        "object_geometry_factor_contact_ready_rows": sum(
            case["current_object_geometry_factor_problem"]["multi_object_contact_factor_ready_rows"]
            for case in case_outputs
        ),
        "v3_solver_complete": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
    }
    write_json(args.output_root / "v17_joint_solver_problem_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--sparse-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--multi-object-timeline-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--geometry-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_geometry_state"),
    )
    parser.add_argument(
        "--object-track-dataset-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_track_datasets"),
    )
    parser.add_argument(
        "--object-material-track-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_tracks"),
    )
    parser.add_argument(
        "--object-material-motion-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_motion_state"),
    )
    parser.add_argument(
        "--object-material-pose-candidate-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_pose_candidates"),
    )
    parser.add_argument(
        "--object-material-surface-replay-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_surface_replay"),
    )
    parser.add_argument(
        "--multi-object-contact-evidence-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_contact_evidence"),
    )
    parser.add_argument(
        "--geometry-source-audit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_source_audit"),
    )
    parser.add_argument(
        "--object-geometry-hypothesis-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state"),
    )
    parser.add_argument(
        "--object-geometry-factor-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_factor_problem"),
    )
    parser.add_argument(
        "--geometry-reconstruction-jobs-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs"),
    )
    parser.add_argument(
        "--geometry-reconstruction-results-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_results"),
    )
    parser.add_argument(
        "--depth-contact-consistency-audit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit"),
    )
    parser.add_argument(
        "--contact-mode-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_graph"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_joint_solver_problem"),
    )
    return parser.parse_args()


def main() -> None:
    payload = build(parse_args())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
