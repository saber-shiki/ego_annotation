#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
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
    pairwise_contact_state_report: Path
    pairwise_contact_depth_gap_report: Path
    hand_metric_depth_state_report: Path
    hand_depth_factor_problem_report: Path
    hand_intrinsics_depth_counterfactual_report: Path
    hand_scale_depth_counterfactual_report: Path
    hand_depth_repair_graph_report: Path
    hand_depth_repair_residual_owner_state_report: Path
    hand_local_projection_repair_problem_report: Path
    mano_parameter_ownership_state_report: Path
    hand_surface_depth_tail_state_report: Path
    hand_tail_support_state_report: Path
    hand_tail_depth_observation_state_report: Path
    contact_ownership_problem_report: Path
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
    pairwise_contact_state_root: Path,
    pairwise_contact_depth_gap_root: Path,
    hand_metric_depth_state_root: Path,
    hand_depth_factor_problem_root: Path,
    hand_intrinsics_depth_counterfactual_root: Path,
    hand_scale_depth_counterfactual_root: Path,
    hand_depth_repair_graph_root: Path,
    hand_depth_repair_residual_owner_state_root: Path,
    hand_local_projection_repair_problem_root: Path,
    mano_parameter_ownership_state_root: Path,
    hand_surface_depth_tail_state_root: Path,
    hand_tail_support_state_root: Path,
    hand_tail_depth_observation_state_root: Path,
    contact_ownership_problem_root: Path,
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
    pairwise_contact_state_report = existing_path(
        pairwise_contact_state_root / case / "v17_pairwise_contact_state.json",
        f"{case} pairwise contact state report",
    )
    pairwise_contact_depth_gap_report = existing_path(
        pairwise_contact_depth_gap_root / case / "v17_pairwise_contact_depth_gap.json",
        f"{case} pairwise contact depth-gap report",
    )
    hand_metric_depth_state_report = existing_path(
        hand_metric_depth_state_root / case / "v17_hand_metric_depth_state.json",
        f"{case} hand metric-depth state report",
    )
    hand_depth_factor_problem_report = existing_path(
        hand_depth_factor_problem_root / case / "v17_hand_depth_factor_problem.json",
        f"{case} hand-depth factor problem report",
    )
    hand_intrinsics_depth_counterfactual_report = existing_path(
        hand_intrinsics_depth_counterfactual_root / case / "v17_hand_intrinsics_depth_counterfactual.json",
        f"{case} hand intrinsics-depth counterfactual report",
    )
    hand_scale_depth_counterfactual_report = existing_path(
        hand_scale_depth_counterfactual_root / case / "v17_hand_scale_depth_counterfactual.json",
        f"{case} hand scale-depth counterfactual report",
    )
    hand_depth_repair_graph_report = existing_path(
        hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
        f"{case} hand depth repair graph report",
    )
    hand_depth_repair_residual_owner_state_report = existing_path(
        hand_depth_repair_residual_owner_state_root
        / case
        / "v17_hand_depth_repair_residual_owner_state.json",
        f"{case} hand depth repair residual-owner state report",
    )
    hand_local_projection_repair_problem_report = existing_path(
        hand_local_projection_repair_problem_root
        / case
        / "v17_hand_local_projection_repair_problem.json",
        f"{case} hand local projection repair problem report",
    )
    mano_parameter_ownership_state_report = existing_path(
        mano_parameter_ownership_state_root / case / "v17_mano_parameter_ownership_state.json",
        f"{case} MANO parameter ownership state report",
    )
    hand_surface_depth_tail_state_report = existing_path(
        hand_surface_depth_tail_state_root / case / "v17_hand_surface_depth_tail_state.json",
        f"{case} hand surface-depth tail state report",
    )
    hand_tail_support_state_report = existing_path(
        hand_tail_support_state_root / case / "v17_hand_tail_support_state.json",
        f"{case} hand tail support state report",
    )
    hand_tail_depth_observation_state_report = existing_path(
        hand_tail_depth_observation_state_root / case / "v17_hand_tail_depth_observation_state.json",
        f"{case} hand tail depth-observation state report",
    )
    contact_ownership_problem_report = existing_path(
        contact_ownership_problem_root / case / "v17_contact_ownership_problem.json",
        f"{case} contact-ownership problem report",
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
        pairwise_contact_state_report=pairwise_contact_state_report,
        pairwise_contact_depth_gap_report=pairwise_contact_depth_gap_report,
        hand_metric_depth_state_report=hand_metric_depth_state_report,
        hand_depth_factor_problem_report=hand_depth_factor_problem_report,
        hand_intrinsics_depth_counterfactual_report=hand_intrinsics_depth_counterfactual_report,
        hand_scale_depth_counterfactual_report=hand_scale_depth_counterfactual_report,
        hand_depth_repair_graph_report=hand_depth_repair_graph_report,
        hand_depth_repair_residual_owner_state_report=hand_depth_repair_residual_owner_state_report,
        hand_local_projection_repair_problem_report=hand_local_projection_repair_problem_report,
        mano_parameter_ownership_state_report=mano_parameter_ownership_state_report,
        hand_surface_depth_tail_state_report=hand_surface_depth_tail_state_report,
        hand_tail_support_state_report=hand_tail_support_state_report,
        hand_tail_depth_observation_state_report=hand_tail_depth_observation_state_report,
        contact_ownership_problem_report=contact_ownership_problem_report,
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


def pairwise_contact_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "pairwise contact frame_count"),
        "pairwise_contact_variable_count": require_int(
            report.get("pairwise_contact_variable_count"),
            "pairwise contact variable count",
        ),
        "measured_image_pair_rows": require_int(
            report.get("measured_image_pair_rows"),
            "pairwise contact measured image rows",
        ),
        "unobserved_image_pair_rows": require_int(
            report.get("unobserved_image_pair_rows"),
            "pairwise contact unobserved image rows",
        ),
        "image_overlap_candidate_rows": require_int(
            report.get("image_overlap_candidate_rows"),
            "pairwise contact image overlap rows",
        ),
        "pair_contact_image_candidate_rows": require_int(
            report.get("pair_contact_image_candidate_rows"),
            "pairwise contact image candidate rows",
        ),
        "contact_owner_image_supported_candidate_rows": require_int(
            report.get("contact_owner_image_supported_candidate_rows"),
            "pairwise contact owner image-supported rows",
        ),
        "owner_image_variables_with_single_supported_candidate": require_int(
            report.get("owner_image_variables_with_single_supported_candidate"),
            "pairwise contact owner image single-supported rows",
        ),
        "owner_image_variables_with_ambiguous_supported_candidates": require_int(
            report.get("owner_image_variables_with_ambiguous_supported_candidates"),
            "pairwise contact owner image ambiguous-supported rows",
        ),
        "physical_contact_factor_ready_rows": require_int(
            report.get("physical_contact_factor_ready_rows"),
            "pairwise contact physical factor-ready rows",
        ),
        "pair_contact_state_counts": require_dict(
            report.get("pair_contact_state_counts"),
            "pairwise contact state counts",
        ),
        "owner_image_state_counts": require_dict(
            report.get("owner_image_state_counts"),
            "pairwise owner image state counts",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def pairwise_contact_depth_gap_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "pairwise depth-gap frame_count"),
        "pairwise_contact_variable_count": require_int(
            report.get("pairwise_contact_variable_count"),
            "pairwise depth-gap contact variable count",
        ),
        "pair_contact_image_candidate_rows": require_int(
            report.get("pair_contact_image_candidate_rows"),
            "pairwise depth-gap image candidate rows",
        ),
        "evaluated_pair_depth_rows": require_int(
            report.get("evaluated_pair_depth_rows"),
            "pairwise depth-gap evaluated rows",
        ),
        "measured_pair_depth_rows": require_int(
            report.get("measured_pair_depth_rows"),
            "pairwise depth-gap measured rows",
        ),
        "unobserved_pair_depth_rows": require_int(
            report.get("unobserved_pair_depth_rows"),
            "pairwise depth-gap unobserved rows",
        ),
        "metric_depth_compatible_candidate_rows": require_int(
            report.get("metric_depth_compatible_candidate_rows"),
            "pairwise depth-gap compatible rows",
        ),
        "physical_contact_factor_ready_rows": require_int(
            report.get("physical_contact_factor_ready_rows"),
            "pairwise depth-gap physical factor-ready rows",
        ),
        "depth_gap_state_counts": require_dict(
            report.get("depth_gap_state_counts"),
            "pairwise depth-gap state counts",
        ),
        "hand_minus_object_depth_m": require_dict(
            report.get("hand_minus_object_depth_m"),
            "pairwise depth-gap hand_minus_object_depth_m",
        ),
        "abs_hand_minus_object_depth_m": require_dict(
            report.get("abs_hand_minus_object_depth_m"),
            "pairwise depth-gap abs_hand_minus_object_depth_m",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_metric_depth_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "hand metric-depth frame_count"),
        "hand_metric_depth_variable_count": require_int(
            report.get("hand_metric_depth_variable_count"),
            "hand metric-depth variable count",
        ),
        "measured_hand_depth_rows": require_int(
            report.get("measured_hand_depth_rows"),
            "hand metric-depth measured rows",
        ),
        "unobserved_hand_depth_rows": require_int(
            report.get("unobserved_hand_depth_rows"),
            "hand metric-depth unobserved rows",
        ),
        "projection_residual_ok_hand_rows": require_int(
            report.get("projection_residual_ok_hand_rows"),
            "hand metric-depth projection residual ok rows",
        ),
        "hand_metric_depth_state_counts": require_dict(
            report.get("hand_metric_depth_state_counts"),
            "hand metric-depth state counts",
        ),
        "partition_summaries": require_dict(
            report.get("partition_summaries"),
            "hand metric-depth partition summaries",
        ),
        "pairwise_contact_depth_gap_comparison": require_dict(
            report.get("pairwise_contact_depth_gap_comparison"),
            "hand metric-depth pairwise comparison",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_depth_factor_problem_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "hand-depth factor frame_count"),
        "hand_depth_variable_count": require_int(
            report.get("hand_depth_variable_count"),
            "hand-depth factor variable count",
        ),
        "metric_depth_factor_rows": require_int(
            report.get("metric_depth_factor_rows"),
            "hand-depth metric depth factor rows",
        ),
        "projection_factor_ready_rows": require_int(
            report.get("projection_factor_ready_rows"),
            "hand-depth projection factor ready rows",
        ),
        "depth_repair_factor_candidate_rows": require_int(
            report.get("depth_repair_factor_candidate_rows"),
            "hand-depth repair factor candidate rows",
        ),
        "metric_hand_state_accepted_rows": require_int(
            report.get("metric_hand_state_accepted_rows"),
            "hand-depth accepted hand state rows",
        ),
        "missing_annotation_hand_rows": require_int(
            report.get("missing_annotation_hand_rows"),
            "hand-depth missing annotation rows",
        ),
        "factor_problem_state_counts": require_dict(
            report.get("factor_problem_state_counts"),
            "hand-depth factor problem state counts",
        ),
        "current_hand_metric_depth_state_counts": require_dict(
            report.get("current_hand_metric_depth_state_counts"),
            "hand-depth current metric state counts",
        ),
        "source_camera_solve_status_counts": require_dict(
            report.get("source_camera_solve_status_counts"),
            "hand-depth source-camera solve status counts",
        ),
        "source_solve_median_depth_m": require_dict(
            report.get("source_solve_median_depth_m"),
            "hand-depth source solve median depth summary",
        ),
        "source_cam_t_z_m": require_dict(
            report.get("source_cam_t_z_m"),
            "hand-depth source cam_t_z summary",
        ),
        "sparse_graph_hand_ray_shift_m": require_dict(
            report.get("sparse_graph_hand_ray_shift_m"),
            "hand-depth sparse graph hand ray shift summary",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_intrinsics_depth_counterfactual_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "hand intrinsics counterfactual frame_count"),
        "hand_intrinsics_counterfactual_variable_count": require_int(
            report.get("hand_intrinsics_counterfactual_variable_count"),
            "hand intrinsics counterfactual variable count",
        ),
        "counterfactual_metric_depth_measured_rows": require_int(
            report.get("counterfactual_metric_depth_measured_rows"),
            "hand intrinsics counterfactual measured rows",
        ),
        "counterfactual_projection_factor_ready_rows": require_int(
            report.get("counterfactual_projection_factor_ready_rows"),
            "hand intrinsics counterfactual projection rows",
        ),
        "counterfactual_depth_repair_factor_candidate_rows": require_int(
            report.get("counterfactual_depth_repair_factor_candidate_rows"),
            "hand intrinsics counterfactual repair rows",
        ),
        "counterfactual_median_gap_improved_rows": require_int(
            report.get("counterfactual_median_gap_improved_rows"),
            "hand intrinsics counterfactual improved rows",
        ),
        "counterfactual_metric_hand_state_accepted_rows": require_int(
            report.get("counterfactual_metric_hand_state_accepted_rows"),
            "hand intrinsics counterfactual accepted rows",
        ),
        "counterfactual_state_counts": require_dict(
            report.get("counterfactual_state_counts"),
            "hand intrinsics counterfactual state counts",
        ),
        "counterfactual_owner_depth_state_counts": require_dict(
            report.get("counterfactual_owner_depth_state_counts"),
            "hand intrinsics counterfactual owner depth state counts",
        ),
        "partition_summaries": require_dict(
            report.get("partition_summaries"),
            "hand intrinsics counterfactual partition summaries",
        ),
        "intrinsics_focal_ratio_fx": require_dict(
            report.get("intrinsics_focal_ratio_fx"),
            "hand intrinsics counterfactual focal ratio",
        ),
        "counterfactual_owner_median_gap_m": require_dict(
            report.get("counterfactual_owner_median_gap_m"),
            "hand intrinsics counterfactual owner median gap",
        ),
        "counterfactual_hand_depth_m": require_dict(
            report.get("counterfactual_hand_depth_m"),
            "hand intrinsics counterfactual hand depth",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_scale_depth_counterfactual_counts(report: dict[str, Any]) -> dict[str, Any]:
    modes = require_dict(report.get("mode_summaries"), "hand scale mode summaries")
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "hand scale counterfactual frame_count"),
        "hand_scale_counterfactual_variable_count": require_int(
            report.get("hand_scale_counterfactual_variable_count"),
            "hand scale counterfactual variable count",
        ),
        "base_available_rows": require_int(
            report.get("base_available_rows"),
            "hand scale base available rows",
        ),
        "scale_candidate_rows": require_int(
            report.get("scale_candidate_rows"),
            "hand scale candidate rows",
        ),
        "case_global_scale": report.get("case_global_scale"),
        "side_global_scales": require_dict(
            report.get("side_global_scales"),
            "hand scale side scales",
        ),
        "row_scale_candidate_summary": require_dict(
            report.get("row_scale_candidate_summary"),
            "hand scale row scale candidate summary",
        ),
        "current_wrist_to_middle_tip_m": require_dict(
            report.get("current_wrist_to_middle_tip_m"),
            "hand scale current wrist-to-middle summary",
        ),
        "case_global_scaled_wrist_to_middle_tip_m": require_dict(
            report.get("case_global_scaled_wrist_to_middle_tip_m"),
            "hand scale case-global wrist-to-middle summary",
        ),
        "side_global_scaled_wrist_to_middle_tip_m": require_dict(
            report.get("side_global_scaled_wrist_to_middle_tip_m"),
            "hand scale side-global wrist-to-middle summary",
        ),
        "per_row_scaled_wrist_to_middle_tip_m": require_dict(
            report.get("per_row_scaled_wrist_to_middle_tip_m"),
            "hand scale per-row wrist-to-middle summary",
        ),
        "mode_summaries": modes,
        "case_global_scale_mode": require_dict(modes.get("case_global_scale"), "case_global_scale mode"),
        "side_global_scale_mode": require_dict(modes.get("side_global_scale"), "side_global_scale mode"),
        "per_row_scale_oracle_mode": require_dict(modes.get("per_row_scale_oracle"), "per_row_scale_oracle mode"),
        "source_intrinsics_counterfactual_comparison": require_dict(
            report.get("source_intrinsics_counterfactual_comparison"),
            "hand scale source intrinsics comparison",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_depth_repair_graph_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "hand depth repair graph frame_count"),
        "hand_depth_repair_graph_variable_count": require_int(
            report.get("hand_depth_repair_graph_variable_count"),
            "hand depth repair graph variable count",
        ),
        "base_available_rows": require_int(
            report.get("base_available_rows"),
            "hand depth repair graph base available rows",
        ),
        "depth_data_candidate_rows": require_int(
            report.get("depth_data_candidate_rows"),
            "hand depth repair graph data rows",
        ),
        "case_global_scale": report.get("case_global_scale"),
        "case_global_scale_bounds": require_dict(
            report.get("case_global_scale_bounds"),
            "hand depth repair graph scale bounds",
        ),
        "case_global_scale_bound_hit": bool(report.get("case_global_scale_bound_hit") is True),
        "case_global_scaled_wrist_to_middle_tip_m": require_dict(
            report.get("case_global_scaled_wrist_to_middle_tip_m"),
            "hand depth repair graph scaled wrist-to-middle summary",
        ),
        "hand_ray_shift_abs_m": require_dict(
            report.get("hand_ray_shift_abs_m"),
            "hand depth repair graph ray-shift summary",
        ),
        "hand_ray_shift_bound_hit_rows": require_int(
            report.get("hand_ray_shift_bound_hit_rows"),
            "hand depth repair graph ray-shift bound hits",
        ),
        "system": require_dict(report.get("system"), "hand depth repair graph system"),
        "solver": require_dict(report.get("solver"), "hand depth repair graph solver"),
        "solver_state_counts": require_dict(
            report.get("solver_state_counts"),
            "hand depth repair graph state counts",
        ),
        "owner_depth_state_counts": require_dict(
            report.get("owner_depth_state_counts"),
            "hand depth repair graph owner depth state counts",
        ),
        "metric_hand_state_accepted_rows": require_int(
            report.get("metric_hand_state_accepted_rows"),
            "hand depth repair graph accepted rows",
        ),
        "depth_repair_factor_candidate_rows": require_int(
            report.get("depth_repair_factor_candidate_rows"),
            "hand depth repair graph repair rows",
        ),
        "projection_residual_to_measurement_px": require_dict(
            report.get("projection_residual_to_measurement_px"),
            "hand depth repair graph projection residual summary",
        ),
        "owner_median_gap_m": require_dict(
            report.get("owner_median_gap_m"),
            "hand depth repair graph owner median gap",
        ),
        "source_scale_counterfactual_comparison": require_dict(
            report.get("source_scale_counterfactual_comparison"),
            "hand depth repair graph scale comparison",
        ),
        "source_tail_depth_observation_comparison": require_dict(
            report.get("source_tail_depth_observation_comparison"),
            "hand depth repair graph tail depth comparison",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_depth_repair_residual_owner_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(
            report.get("frame_count"),
            "hand depth repair residual-owner frame_count",
        ),
        "hand_depth_repair_residual_owner_variable_count": require_int(
            report.get("hand_depth_repair_residual_owner_variable_count"),
            "hand depth repair residual-owner variable count",
        ),
        "repair_residual_factor_candidate_rows": require_int(
            report.get("repair_residual_factor_candidate_rows"),
            "hand depth repair residual-owner candidate rows",
        ),
        "independent_supported_repair_residual_rows": require_int(
            report.get("independent_supported_repair_residual_rows"),
            "hand depth repair residual-owner supported rows",
        ),
        "independent_unsupported_repair_residual_rows": require_int(
            report.get("independent_unsupported_repair_residual_rows"),
            "hand depth repair residual-owner unsupported rows",
        ),
        "residual_independent_support_state_counts": require_dict(
            report.get("residual_independent_support_state_counts"),
            "hand depth repair residual-owner support state counts",
        ),
        "residual_depth_observation_state_counts": require_dict(
            report.get("residual_depth_observation_state_counts"),
            "hand depth repair residual-owner depth observation state counts",
        ),
        "supported_residual_depth_observation_state_counts": require_dict(
            report.get("supported_residual_depth_observation_state_counts"),
            "hand depth repair residual-owner supported depth observation state counts",
        ),
        "residual_owner_state_counts": require_dict(
            report.get("residual_owner_state_counts"),
            "hand depth repair residual-owner state counts",
        ),
        "residual_sample_count": require_int(
            report.get("residual_sample_count"),
            "hand depth repair residual-owner sample count",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_local_projection_repair_problem_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(
            report.get("frame_count"),
            "hand local projection repair frame_count",
        ),
        "hand_local_projection_repair_variable_count": require_int(
            report.get("hand_local_projection_repair_variable_count"),
            "hand local projection repair variable count",
        ),
        "repair_residual_factor_candidate_rows": require_int(
            report.get("repair_residual_factor_candidate_rows"),
            "hand local projection repair residual rows",
        ),
        "local_projection_repair_factor_candidate_rows": require_int(
            report.get("local_projection_repair_factor_candidate_rows"),
            "hand local projection repair candidate rows",
        ),
        "partial_projection_depth_mixed_owner_rows": require_int(
            report.get("partial_projection_depth_mixed_owner_rows"),
            "hand local projection mixed owner rows",
        ),
        "depth_observation_or_occlusion_owner_rows": require_int(
            report.get("depth_observation_or_occlusion_owner_rows"),
            "hand local projection depth observation owner rows",
        ),
        "projection_support_unresolved_rows": require_int(
            report.get("projection_support_unresolved_rows"),
            "hand local projection support unresolved rows",
        ),
        "residual_local_projection_repair_state_counts": require_dict(
            report.get("residual_local_projection_repair_state_counts"),
            "hand local projection repair state counts",
        ),
        "local_projection_assignment": require_dict(
            report.get("local_projection_assignment"),
            "hand local projection assignment summary",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def mano_parameter_ownership_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "MANO parameter ownership frame_count"),
        "mano_parameter_ownership_variable_count": require_int(
            report.get("mano_parameter_ownership_variable_count"),
            "MANO parameter ownership variable count",
        ),
        "repair_residual_factor_candidate_rows": require_int(
            report.get("repair_residual_factor_candidate_rows"),
            "MANO parameter ownership residual rows",
        ),
        "residual_mano_parameter_owned_rows": require_int(
            report.get("residual_mano_parameter_owned_rows"),
            "MANO parameter owned residual rows",
        ),
        "local_projection_repair_factor_candidate_rows": require_int(
            report.get("local_projection_repair_factor_candidate_rows"),
            "MANO parameter ownership local projection rows",
        ),
        "local_projection_articulation_factor_candidate_rows": require_int(
            report.get("local_projection_articulation_factor_candidate_rows"),
            "MANO parameter ownership local articulation rows",
        ),
        "mixed_projection_articulation_observation_candidate_rows": require_int(
            report.get("mixed_projection_articulation_observation_candidate_rows"),
            "MANO parameter ownership mixed articulation rows",
        ),
        "residual_mano_parameter_ownership_state_counts": require_dict(
            report.get("residual_mano_parameter_ownership_state_counts"),
            "MANO parameter ownership state counts",
        ),
        "owned_alignment_error_summary": require_dict(
            report.get("owned_alignment_error_summary"),
            "MANO parameter owned alignment summary",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_surface_depth_tail_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "hand surface-depth tail frame_count"),
        "hand_surface_depth_tail_variable_count": require_int(
            report.get("hand_surface_depth_tail_variable_count"),
            "hand surface-depth tail variable count",
        ),
        "scalar_depth_compatible_rows": require_int(
            report.get("scalar_depth_compatible_rows"),
            "hand surface-depth scalar-compatible rows",
        ),
        "scalar_depth_tail_factor_candidate_rows": require_int(
            report.get("scalar_depth_tail_factor_candidate_rows"),
            "hand surface-depth tail factor candidate rows",
        ),
        "projection_untrusted_after_scalar_scale_rows": require_int(
            report.get("projection_untrusted_after_scalar_scale_rows"),
            "hand surface-depth projection-untrusted rows",
        ),
        "unobserved_after_scalar_scale_rows": require_int(
            report.get("unobserved_after_scalar_scale_rows"),
            "hand surface-depth unobserved rows",
        ),
        "tail_state_counts": require_dict(
            report.get("tail_state_counts"),
            "hand surface-depth tail state counts",
        ),
        "tail_owner_partition_counts": require_dict(
            report.get("tail_owner_partition_counts"),
            "hand surface-depth owner partition counts",
        ),
        "tail_pattern_counts": require_dict(
            report.get("tail_pattern_counts"),
            "hand surface-depth tail pattern counts",
        ),
        "tail_candidate_pattern_counts": require_dict(
            report.get("tail_candidate_pattern_counts"),
            "hand surface-depth tail candidate pattern counts",
        ),
        "tail_candidate_owner_partition_counts": require_dict(
            report.get("tail_candidate_owner_partition_counts"),
            "hand surface-depth tail candidate owner partition counts",
        ),
        "tail_candidate_abs_gap_p95_m": require_dict(
            report.get("tail_candidate_abs_gap_p95_m"),
            "hand surface-depth tail candidate abs gap p95",
        ),
        "tail_candidate_signed_gap_p05_m": require_dict(
            report.get("tail_candidate_signed_gap_p05_m"),
            "hand surface-depth tail candidate signed gap p05",
        ),
        "tail_candidate_signed_gap_p95_m": require_dict(
            report.get("tail_candidate_signed_gap_p95_m"),
            "hand surface-depth tail candidate signed gap p95",
        ),
        "tail_candidate_row_scale_ratio_spread_p95_minus_p05": require_dict(
            report.get("tail_candidate_row_scale_ratio_spread_p95_minus_p05"),
            "hand surface-depth tail candidate scale-ratio spread",
        ),
        "all_rows_row_scale_ratio_spread_p95_minus_p05": require_dict(
            report.get("all_rows_row_scale_ratio_spread_p95_minus_p05"),
            "hand surface-depth all-row scale-ratio spread",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_tail_support_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "hand tail support frame_count"),
        "hand_tail_support_variable_count": require_int(
            report.get("hand_tail_support_variable_count"),
            "hand tail support variable count",
        ),
        "tail_factor_candidate_rows": require_int(
            report.get("tail_factor_candidate_rows"),
            "hand tail support candidate rows",
        ),
        "tail_selected_support_state_counts": require_dict(
            report.get("tail_selected_support_state_counts"),
            "hand tail selected support state counts",
        ),
        "tail_independent_support_state_counts": require_dict(
            report.get("tail_independent_support_state_counts"),
            "hand tail independent support state counts",
        ),
        "tail_abs_sample_count": require_int(report.get("tail_abs_sample_count"), "hand tail abs sample count"),
        "tail_negative_sample_count": require_int(
            report.get("tail_negative_sample_count"),
            "hand tail negative sample count",
        ),
        "tail_positive_sample_count": require_int(
            report.get("tail_positive_sample_count"),
            "hand tail positive sample count",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def hand_tail_depth_observation_state_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "hand tail depth-observation frame_count"),
        "hand_tail_depth_observation_variable_count": require_int(
            report.get("hand_tail_depth_observation_variable_count"),
            "hand tail depth-observation variable count",
        ),
        "tail_factor_candidate_rows": require_int(
            report.get("tail_factor_candidate_rows"),
            "hand tail depth-observation candidate rows",
        ),
        "independent_supported_tail_candidate_rows": require_int(
            report.get("independent_supported_tail_candidate_rows"),
            "hand tail depth-observation supported rows",
        ),
        "independent_unsupported_tail_candidate_rows": require_int(
            report.get("independent_unsupported_tail_candidate_rows"),
            "hand tail depth-observation unsupported rows",
        ),
        "tail_depth_observation_state_counts": require_dict(
            report.get("tail_depth_observation_state_counts"),
            "hand tail depth-observation state counts",
        ),
        "supported_tail_depth_observation_state_counts": require_dict(
            report.get("supported_tail_depth_observation_state_counts"),
            "hand tail supported depth-observation state counts",
        ),
        "tail_abs_sample_count": require_int(
            report.get("tail_abs_sample_count"),
            "hand tail depth-observation abs sample count",
        ),
        "object_geometry_complete": bool(report.get("object_geometry_complete") is True),
        "object_pose_requirement_met": bool(report.get("object_pose_requirement_met") is True),
        "annotation_ready": bool(report.get("annotation_ready") is True),
        "deliverable_ready": bool(report.get("deliverable_ready") is True),
        "accuracy_target_met": bool(report.get("accuracy_target_met") is True),
        "v3_solver_complete": bool(report.get("v3_solver_complete") is True),
    }


def contact_ownership_problem_counts(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "frame_count": require_int(report.get("frame_count"), "contact ownership frame_count"),
        "contact_owner_variable_count": require_int(
            report.get("contact_owner_variable_count"),
            "contact ownership variable count",
        ),
        "contact_owner_candidate_rows": require_int(
            report.get("contact_owner_candidate_rows"),
            "contact ownership candidate rows",
        ),
        "contact_owner_variables_with_selected_measurement": require_int(
            report.get("contact_owner_variables_with_selected_measurement"),
            "contact ownership selected measurement rows",
        ),
        "contact_owner_variables_without_selected_measurement": require_int(
            report.get("contact_owner_variables_without_selected_measurement"),
            "contact ownership rows without selected measurement",
        ),
        "contact_owner_variables_with_supported_candidate": require_int(
            report.get("contact_owner_variables_with_supported_candidate"),
            "contact ownership supported variables",
        ),
        "contact_owner_variables_with_geometry_supported_candidate": require_int(
            report.get("contact_owner_variables_with_geometry_supported_candidate"),
            "contact ownership geometry-supported variables",
        ),
        "contact_owner_image_supported_candidate_rows": require_int(
            report.get("contact_owner_image_supported_candidate_rows"),
            "contact ownership image-supported candidate rows",
        ),
        "pairwise_metric_depth_evaluated_rows": require_int(
            report.get("pairwise_metric_depth_evaluated_rows"),
            "contact ownership pairwise metric-depth evaluated rows",
        ),
        "pairwise_metric_depth_compatible_candidate_rows": require_int(
            report.get("pairwise_metric_depth_compatible_candidate_rows"),
            "contact ownership pairwise metric-depth compatible rows",
        ),
        "contact_owner_metric_depth_supported_candidate_rows": require_int(
            report.get("contact_owner_metric_depth_supported_candidate_rows"),
            "contact ownership metric-depth supported rows",
        ),
        "owner_image_variables_with_single_supported_candidate": require_int(
            report.get("owner_image_variables_with_single_supported_candidate"),
            "contact ownership single image-supported variables",
        ),
        "owner_image_variables_with_ambiguous_supported_candidates": require_int(
            report.get("owner_image_variables_with_ambiguous_supported_candidates"),
            "contact ownership ambiguous image-supported variables",
        ),
        "contact_owner_variables_without_supported_candidate": require_int(
            report.get("contact_owner_variables_without_supported_candidate"),
            "contact ownership unsupported variables",
        ),
        "contact_owner_factor_ready_rows": require_int(
            report.get("contact_owner_factor_ready_rows"),
            "contact ownership factor-ready rows",
        ),
        "owner_variable_state_counts": require_dict(
            report.get("owner_variable_state_counts"),
            "contact ownership owner_variable_state_counts",
        ),
        "candidate_evidence_state_counts": require_dict(
            report.get("candidate_evidence_state_counts"),
            "contact ownership candidate_evidence_state_counts",
        ),
        "selected_measurement_candidate_state_counts": require_dict(
            report.get("selected_measurement_candidate_state_counts"),
            "contact ownership selected_measurement_candidate_state_counts",
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
        "depth_contact_legacy_contact_ready_hand_rows": require_int(
            report.get("depth_contact_legacy_contact_ready_hand_rows"),
            "object-geometry factor depth_contact_legacy_contact_ready_hand_rows",
        ),
        "depth_contact_multi_object_reconstructed_object_contact_candidate_rows": require_int(
            report.get("depth_contact_multi_object_reconstructed_object_contact_candidate_rows"),
            "object-geometry factor depth_contact_multi_object_reconstructed_object_contact_candidate_rows",
        ),
        "depth_contact_legacy_owner_mismatch_frame_count": require_int(
            report.get("depth_contact_legacy_owner_mismatch_frame_count"),
            "object-geometry factor depth_contact_legacy_owner_mismatch_frame_count",
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
        "contact_owner_variable_count": require_int(
            report.get("contact_owner_variable_count"),
            "object-geometry factor contact_owner_variable_count",
        ),
        "contact_owner_candidate_rows": require_int(
            report.get("contact_owner_candidate_rows"),
            "object-geometry factor contact_owner_candidate_rows",
        ),
        "contact_owner_supported_candidate_rows": require_int(
            report.get("contact_owner_supported_candidate_rows"),
            "object-geometry factor contact_owner_supported_candidate_rows",
        ),
        "contact_owner_geometrically_supported_candidate_rows": require_int(
            report.get("contact_owner_geometrically_supported_candidate_rows"),
            "object-geometry factor contact_owner_geometrically_supported_candidate_rows",
        ),
        "contact_owner_image_supported_candidate_rows": require_int(
            report.get("contact_owner_image_supported_candidate_rows"),
            "object-geometry factor contact_owner_image_supported_candidate_rows",
        ),
        "contact_owner_factor_ready_rows": require_int(
            report.get("contact_owner_factor_ready_rows"),
            "object-geometry factor contact_owner_factor_ready_rows",
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
        "legacy_contact_ready_hand_rows": require_int(
            report.get("legacy_contact_ready_hand_rows"),
            "depth-contact legacy_contact_ready_hand_rows",
        ),
        "multi_object_reconstructed_object_contact_candidate_rows": require_int(
            report.get("multi_object_reconstructed_object_contact_candidate_rows"),
            "depth-contact multi_object_reconstructed_object_contact_candidate_rows",
        ),
        "legacy_owner_mismatch_frame_count": require_int(
            report.get("legacy_owner_mismatch_frame_count"),
            "depth-contact legacy_owner_mismatch_frame_count",
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
    pairwise_contact_state: dict[str, Any],
    pairwise_contact_depth_gap: dict[str, Any],
    hand_metric_depth_state: dict[str, Any],
    hand_depth_factor_problem: dict[str, Any],
    hand_intrinsics_depth_counterfactual: dict[str, Any],
    hand_scale_depth_counterfactual: dict[str, Any],
    hand_depth_repair_graph: dict[str, Any],
    hand_depth_repair_residual_owner_state: dict[str, Any],
    hand_local_projection_repair_problem: dict[str, Any],
    mano_parameter_ownership_state: dict[str, Any],
    hand_surface_depth_tail_state: dict[str, Any],
    hand_tail_support_state: dict[str, Any],
    hand_tail_depth_observation_state: dict[str, Any],
    contact_ownership_problem: dict[str, Any],
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
                "hand_metric_depth_variable_count": hand_metric_depth_state[
                    "hand_metric_depth_variable_count"
                ],
                "hand_metric_depth_measured_rows": hand_metric_depth_state[
                    "measured_hand_depth_rows"
                ],
                "hand_metric_depth_projection_residual_ok_rows": hand_metric_depth_state[
                    "projection_residual_ok_hand_rows"
                ],
                "hand_metric_depth_state_counts": hand_metric_depth_state[
                    "hand_metric_depth_state_counts"
                ],
                "hand_depth_factor_problem_state_counts": hand_depth_factor_problem[
                    "factor_problem_state_counts"
                ],
                "hand_depth_factor_depth_repair_candidate_rows": hand_depth_factor_problem[
                    "depth_repair_factor_candidate_rows"
                ],
                "metric_hand_state_accepted_rows": hand_depth_factor_problem[
                    "metric_hand_state_accepted_rows"
                ],
                "counterfactual_intrinsics_depth_repair_candidate_rows": hand_intrinsics_depth_counterfactual[
                    "counterfactual_depth_repair_factor_candidate_rows"
                ],
                "counterfactual_metric_hand_state_accepted_rows": hand_intrinsics_depth_counterfactual[
                    "counterfactual_metric_hand_state_accepted_rows"
                ],
                "counterfactual_median_gap_improved_rows": hand_intrinsics_depth_counterfactual[
                    "counterfactual_median_gap_improved_rows"
                ],
                "counterfactual_state_counts": hand_intrinsics_depth_counterfactual[
                    "counterfactual_state_counts"
                ],
                "counterfactual_owner_depth_state_counts": hand_intrinsics_depth_counterfactual[
                    "counterfactual_owner_depth_state_counts"
                ],
                "counterfactual_intrinsics_focal_ratio_fx": hand_intrinsics_depth_counterfactual[
                    "intrinsics_focal_ratio_fx"
                ],
                "counterfactual_owner_median_gap_m": hand_intrinsics_depth_counterfactual[
                    "counterfactual_owner_median_gap_m"
                ],
                "scale_counterfactual_base_available_rows": hand_scale_depth_counterfactual[
                    "base_available_rows"
                ],
                "scale_counterfactual_candidate_rows": hand_scale_depth_counterfactual[
                    "scale_candidate_rows"
                ],
                "scale_counterfactual_case_global_scale": hand_scale_depth_counterfactual[
                    "case_global_scale"
                ],
                "scale_counterfactual_side_global_scales": hand_scale_depth_counterfactual[
                    "side_global_scales"
                ],
                "scale_counterfactual_case_global_mode": hand_scale_depth_counterfactual[
                    "case_global_scale_mode"
                ],
                "scale_counterfactual_side_global_mode": hand_scale_depth_counterfactual[
                    "side_global_scale_mode"
                ],
                "scale_counterfactual_per_row_oracle_mode": hand_scale_depth_counterfactual[
                    "per_row_scale_oracle_mode"
                ],
                "scale_counterfactual_current_wrist_to_middle_tip_m": hand_scale_depth_counterfactual[
                    "current_wrist_to_middle_tip_m"
                ],
                "scale_counterfactual_case_scaled_wrist_to_middle_tip_m": hand_scale_depth_counterfactual[
                    "case_global_scaled_wrist_to_middle_tip_m"
                ],
                "hand_depth_repair_graph_base_available_rows": hand_depth_repair_graph[
                    "base_available_rows"
                ],
                "hand_depth_repair_graph_depth_data_candidate_rows": hand_depth_repair_graph[
                    "depth_data_candidate_rows"
                ],
                "hand_depth_repair_graph_case_global_scale": hand_depth_repair_graph[
                    "case_global_scale"
                ],
                "hand_depth_repair_graph_case_global_scale_bounds": hand_depth_repair_graph[
                    "case_global_scale_bounds"
                ],
                "hand_depth_repair_graph_case_global_scale_bound_hit": hand_depth_repair_graph[
                    "case_global_scale_bound_hit"
                ],
                "hand_depth_repair_graph_state_counts": hand_depth_repair_graph[
                    "solver_state_counts"
                ],
                "hand_depth_repair_graph_owner_depth_state_counts": hand_depth_repair_graph[
                    "owner_depth_state_counts"
                ],
                "hand_depth_repair_graph_metric_hand_state_accepted_rows": hand_depth_repair_graph[
                    "metric_hand_state_accepted_rows"
                ],
                "hand_depth_repair_graph_depth_repair_factor_candidate_rows": hand_depth_repair_graph[
                    "depth_repair_factor_candidate_rows"
                ],
                "hand_depth_repair_graph_hand_ray_shift_abs_m": hand_depth_repair_graph[
                    "hand_ray_shift_abs_m"
                ],
                "hand_depth_repair_graph_hand_ray_shift_bound_hit_rows": hand_depth_repair_graph[
                    "hand_ray_shift_bound_hit_rows"
                ],
                "hand_depth_repair_graph_projection_residual_to_measurement_px": hand_depth_repair_graph[
                    "projection_residual_to_measurement_px"
                ],
                "hand_depth_repair_graph_owner_median_gap_m": hand_depth_repair_graph[
                    "owner_median_gap_m"
                ],
                "hand_depth_repair_residual_candidate_rows": hand_depth_repair_residual_owner_state[
                    "repair_residual_factor_candidate_rows"
                ],
                "hand_depth_repair_residual_supported_rows": hand_depth_repair_residual_owner_state[
                    "independent_supported_repair_residual_rows"
                ],
                "hand_depth_repair_residual_unsupported_rows": hand_depth_repair_residual_owner_state[
                    "independent_unsupported_repair_residual_rows"
                ],
                "hand_depth_repair_residual_support_state_counts": hand_depth_repair_residual_owner_state[
                    "residual_independent_support_state_counts"
                ],
                "hand_depth_repair_residual_depth_observation_state_counts": hand_depth_repair_residual_owner_state[
                    "residual_depth_observation_state_counts"
                ],
                "supported_hand_depth_repair_residual_depth_observation_state_counts": hand_depth_repair_residual_owner_state[
                    "supported_residual_depth_observation_state_counts"
                ],
                "hand_depth_repair_residual_owner_state_counts": hand_depth_repair_residual_owner_state[
                    "residual_owner_state_counts"
                ],
                "hand_depth_repair_residual_sample_count": hand_depth_repair_residual_owner_state[
                    "residual_sample_count"
                ],
                "hand_local_projection_repair_factor_candidate_rows": hand_local_projection_repair_problem[
                    "local_projection_repair_factor_candidate_rows"
                ],
                "hand_local_projection_mixed_owner_rows": hand_local_projection_repair_problem[
                    "partial_projection_depth_mixed_owner_rows"
                ],
                "hand_local_projection_depth_observation_owner_rows": hand_local_projection_repair_problem[
                    "depth_observation_or_occlusion_owner_rows"
                ],
                "hand_local_projection_support_unresolved_rows": hand_local_projection_repair_problem[
                    "projection_support_unresolved_rows"
                ],
                "hand_local_projection_repair_state_counts": hand_local_projection_repair_problem[
                    "residual_local_projection_repair_state_counts"
                ],
                "hand_local_projection_assignment": hand_local_projection_repair_problem[
                    "local_projection_assignment"
                ],
                "mano_parameter_owned_residual_rows": mano_parameter_ownership_state[
                    "residual_mano_parameter_owned_rows"
                ],
                "mano_parameter_ownership_state_counts": mano_parameter_ownership_state[
                    "residual_mano_parameter_ownership_state_counts"
                ],
                "mano_parameter_owned_alignment_error_summary": mano_parameter_ownership_state[
                    "owned_alignment_error_summary"
                ],
                "mano_parameter_local_projection_articulation_factor_candidate_rows": mano_parameter_ownership_state[
                    "local_projection_articulation_factor_candidate_rows"
                ],
                "mano_parameter_mixed_projection_articulation_observation_candidate_rows": mano_parameter_ownership_state[
                    "mixed_projection_articulation_observation_candidate_rows"
                ],
                "surface_depth_tail_variable_count": hand_surface_depth_tail_state[
                    "hand_surface_depth_tail_variable_count"
                ],
                "surface_depth_tail_scalar_compatible_rows": hand_surface_depth_tail_state[
                    "scalar_depth_compatible_rows"
                ],
                "surface_depth_tail_factor_candidate_rows": hand_surface_depth_tail_state[
                    "scalar_depth_tail_factor_candidate_rows"
                ],
                "surface_depth_tail_projection_untrusted_rows": hand_surface_depth_tail_state[
                    "projection_untrusted_after_scalar_scale_rows"
                ],
                "surface_depth_tail_candidate_pattern_counts": hand_surface_depth_tail_state[
                    "tail_candidate_pattern_counts"
                ],
                "surface_depth_tail_candidate_abs_gap_p95_m": hand_surface_depth_tail_state[
                    "tail_candidate_abs_gap_p95_m"
                ],
                "tail_support_variable_count": hand_tail_support_state[
                    "hand_tail_support_variable_count"
                ],
                "tail_support_candidate_rows": hand_tail_support_state[
                    "tail_factor_candidate_rows"
                ],
                "tail_selected_support_state_counts": hand_tail_support_state[
                    "tail_selected_support_state_counts"
                ],
                "tail_independent_support_state_counts": hand_tail_support_state[
                    "tail_independent_support_state_counts"
                ],
                "tail_abs_sample_count": hand_tail_support_state["tail_abs_sample_count"],
                "tail_negative_sample_count": hand_tail_support_state["tail_negative_sample_count"],
                "tail_positive_sample_count": hand_tail_support_state["tail_positive_sample_count"],
                "tail_depth_observation_candidate_rows": hand_tail_depth_observation_state[
                    "tail_factor_candidate_rows"
                ],
                "tail_depth_observation_state_counts": hand_tail_depth_observation_state[
                    "tail_depth_observation_state_counts"
                ],
                "supported_tail_depth_observation_state_counts": hand_tail_depth_observation_state[
                    "supported_tail_depth_observation_state_counts"
                ],
                "source_camera_solve_status_counts": hand_depth_factor_problem[
                    "source_camera_solve_status_counts"
                ],
                "source_solve_median_depth_m": hand_depth_factor_problem[
                    "source_solve_median_depth_m"
                ],
                "hand_metric_depth_all_pixels_summary": hand_metric_depth_state[
                    "partition_summaries"
                ]["all_projected_hand_pixels"],
                "hand_metric_depth_far_from_object_summary": hand_metric_depth_state[
                    "partition_summaries"
                ]["far_from_active_object_masks"],
                "hand_metric_depth_near_object_summary": hand_metric_depth_state[
                    "partition_summaries"
                ]["near_active_object_masks"],
            },
            [
                "MANO pose parameters are not graph variables",
                "MANO shape parameters are not graph variables",
                "world wrist pose is only corrected along camera rays",
                "source-camera hand translation is an inherited monocular measurement, not a UniDepth-constrained variable",
                "UniDepth-aligned source intrinsics improve the hand-depth gap but still leave almost all rows depth-incompatible",
                "global or per-side hand scale reduces median depth bias but leaves most rows outside the p95 depth threshold",
                "a full-timeline bounded scale plus ray-depth repair graph improves the hand-depth state but still leaves many projection-trusted depth-repair candidates",
                "the remaining hand-depth repair residuals split into local hand-surface/projection owners and depth-observation owners after per-sample owner partitioning",
                "local projection assignment materializes the local hand-surface factor candidates but does not update MANO articulation",
                "saved MANO parameters own every local projection factor candidate, but an articulation optimizer has not consumed those factors",
                "the scale required by depth shrinks the median wrist-to-middle-tip length below the current hand-size prior",
                "per-row scalar hand-depth repair still leaves large visible-surface depth tails in many rows",
                "most residual hand-surface depth tails are inside or near independent same-side hand boxes, so local hand-surface or depth-observation mismatch dominates detector support failure",
                "supported residual tail pixels split between nearby compatible depth and absent nearby compatible depth, requiring both local MANO surface/projection repair variables and depth-observation variables",
                "front-surface MANO depth is not metric-depth compatible in the current source-camera state",
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
                "depth_contact_legacy_owner_mismatch_frame_count": depth_contact_consistency[
                    "legacy_owner_mismatch_frame_count"
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
                "legacy contact rows in accepted reconstruction windows do not name the reconstructed multi-object id",
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
                "depth_contact_legacy_contact_ready_hand_rows": depth_contact_consistency[
                    "legacy_contact_ready_hand_rows"
                ],
                "depth_contact_multi_object_reconstructed_object_contact_candidate_rows": depth_contact_consistency[
                    "multi_object_reconstructed_object_contact_candidate_rows"
                ],
                "depth_contact_legacy_owner_mismatch_frame_count": depth_contact_consistency[
                    "legacy_owner_mismatch_frame_count"
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
                "legacy contact rows in accepted reconstruction windows are attached to the legacy single-object stream, not to the reconstructed object id",
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
                "pairwise_contact_variable_count": pairwise_contact_state[
                    "pairwise_contact_variable_count"
                ],
                "pairwise_measured_image_pair_rows": pairwise_contact_state[
                    "measured_image_pair_rows"
                ],
                "pairwise_image_overlap_candidate_rows": pairwise_contact_state[
                    "image_overlap_candidate_rows"
                ],
                "pair_contact_image_candidate_rows": pairwise_contact_state[
                    "pair_contact_image_candidate_rows"
                ],
                "pairwise_physical_contact_factor_ready_rows": pairwise_contact_state[
                    "physical_contact_factor_ready_rows"
                ],
                "pairwise_contact_state_counts": pairwise_contact_state[
                    "pair_contact_state_counts"
                ],
                "contact_owner_variable_count": contact_ownership_problem[
                    "contact_owner_variable_count"
                ],
                "contact_owner_candidate_rows": contact_ownership_problem[
                    "contact_owner_candidate_rows"
                ],
                "contact_owner_variables_with_selected_measurement": contact_ownership_problem[
                    "contact_owner_variables_with_selected_measurement"
                ],
                "contact_owner_variables_without_selected_measurement": contact_ownership_problem[
                    "contact_owner_variables_without_selected_measurement"
                ],
                "contact_owner_variables_with_supported_candidate": contact_ownership_problem[
                    "contact_owner_variables_with_supported_candidate"
                ],
                "contact_owner_variables_with_geometry_supported_candidate": contact_ownership_problem[
                    "contact_owner_variables_with_geometry_supported_candidate"
                ],
                "contact_owner_image_supported_candidate_rows": contact_ownership_problem[
                    "contact_owner_image_supported_candidate_rows"
                ],
                "pairwise_metric_depth_evaluated_rows": pairwise_contact_depth_gap[
                    "evaluated_pair_depth_rows"
                ],
                "pairwise_metric_depth_compatible_candidate_rows": pairwise_contact_depth_gap[
                    "metric_depth_compatible_candidate_rows"
                ],
                "pairwise_depth_gap_state_counts": pairwise_contact_depth_gap[
                    "depth_gap_state_counts"
                ],
                "contact_owner_metric_depth_supported_candidate_rows": contact_ownership_problem[
                    "contact_owner_metric_depth_supported_candidate_rows"
                ],
                "owner_image_variables_with_single_supported_candidate": contact_ownership_problem[
                    "owner_image_variables_with_single_supported_candidate"
                ],
                "owner_image_variables_with_ambiguous_supported_candidates": contact_ownership_problem[
                    "owner_image_variables_with_ambiguous_supported_candidates"
                ],
                "contact_owner_variables_without_supported_candidate": contact_ownership_problem[
                    "contact_owner_variables_without_supported_candidate"
                ],
                "contact_owner_factor_ready_rows": contact_ownership_problem[
                    "contact_owner_factor_ready_rows"
                ],
                "contact_owner_state_counts": contact_ownership_problem[
                    "owner_variable_state_counts"
                ],
                "contact_owner_candidate_evidence_state_counts": contact_ownership_problem[
                    "candidate_evidence_state_counts"
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
                "object_geometry_factor_contact_owner_variable_count": object_geometry_factor_problem[
                    "contact_owner_variable_count"
                ],
                "object_geometry_factor_contact_owner_candidate_rows": object_geometry_factor_problem[
                    "contact_owner_candidate_rows"
                ],
                "object_geometry_factor_contact_owner_factor_ready_rows": object_geometry_factor_problem[
                    "contact_owner_factor_ready_rows"
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
                "depth_contact_legacy_contact_ready_hand_rows": depth_contact_consistency[
                    "legacy_contact_ready_hand_rows"
                ],
                "depth_contact_multi_object_reconstructed_object_contact_candidate_rows": depth_contact_consistency[
                    "multi_object_reconstructed_object_contact_candidate_rows"
                ],
                "depth_contact_legacy_owner_mismatch_frame_count": depth_contact_consistency[
                    "legacy_owner_mismatch_frame_count"
                ],
                "depth_contact_shared_depth_state_ready_frame_count": depth_contact_consistency[
                    "shared_depth_state_ready_frame_count"
                ],
            },
            [
                "contact modes are estimated before the sparse geometry graph and then fixed",
                "the full hand-object table measures visible-surface distance but does not estimate contact modes",
                "pairwise image contact variables do not yet share metric object geometry or depth",
                "pairwise image contact candidates are measured against UniDepth, and every evaluated candidate is depth-incompatible in the current hand state",
                "accepted local contact-patch states are not unified with multi-object visible surfaces",
                "contact-mode ready rows have no same-frame multi-object visible-surface contact candidates",
                "object-centric contact factor blocks have zero factor-ready rows against multi-object geometry",
                "contact-owner variables are materialized but no owner factor is ready against object geometry",
                "pairwise image evidence supports contact ownership candidates without making metric contact factors ready",
                "most contact-mode ready rows have no selected measurement that names an object",
                "accepted reconstruction meshes have no near-contact hand rows under the current depth state",
                "legacy contact factors are not object-id-owned in accepted reconstruction windows",
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
                "pairwise_contact_depth_gap_hand_minus_object_depth_m": pairwise_contact_depth_gap[
                    "hand_minus_object_depth_m"
                ],
                "pairwise_contact_depth_gap_abs_m": pairwise_contact_depth_gap[
                    "abs_hand_minus_object_depth_m"
                ],
                "pairwise_contact_depth_gap_state_counts": pairwise_contact_depth_gap[
                    "depth_gap_state_counts"
                ],
                "hand_metric_depth_state_counts": hand_metric_depth_state[
                    "hand_metric_depth_state_counts"
                ],
                "hand_depth_factor_problem_state_counts": hand_depth_factor_problem[
                    "factor_problem_state_counts"
                ],
                "hand_depth_factor_depth_repair_candidate_rows": hand_depth_factor_problem[
                    "depth_repair_factor_candidate_rows"
                ],
                "metric_hand_state_accepted_rows": hand_depth_factor_problem[
                    "metric_hand_state_accepted_rows"
                ],
                "counterfactual_intrinsics_depth_repair_candidate_rows": hand_intrinsics_depth_counterfactual[
                    "counterfactual_depth_repair_factor_candidate_rows"
                ],
                "counterfactual_metric_hand_state_accepted_rows": hand_intrinsics_depth_counterfactual[
                    "counterfactual_metric_hand_state_accepted_rows"
                ],
                "counterfactual_median_gap_improved_rows": hand_intrinsics_depth_counterfactual[
                    "counterfactual_median_gap_improved_rows"
                ],
                "counterfactual_intrinsics_focal_ratio_fx": hand_intrinsics_depth_counterfactual[
                    "intrinsics_focal_ratio_fx"
                ],
                "counterfactual_owner_median_gap_m": hand_intrinsics_depth_counterfactual[
                    "counterfactual_owner_median_gap_m"
                ],
                "counterfactual_owner_depth_state_counts": hand_intrinsics_depth_counterfactual[
                    "counterfactual_owner_depth_state_counts"
                ],
                "scale_counterfactual_case_global_mode": hand_scale_depth_counterfactual[
                    "case_global_scale_mode"
                ],
                "scale_counterfactual_side_global_mode": hand_scale_depth_counterfactual[
                    "side_global_scale_mode"
                ],
                "scale_counterfactual_per_row_oracle_mode": hand_scale_depth_counterfactual[
                    "per_row_scale_oracle_mode"
                ],
                "scale_counterfactual_row_scale_candidate_summary": hand_scale_depth_counterfactual[
                    "row_scale_candidate_summary"
                ],
                "scale_counterfactual_case_scaled_wrist_to_middle_tip_m": hand_scale_depth_counterfactual[
                    "case_global_scaled_wrist_to_middle_tip_m"
                ],
                "hand_depth_repair_graph_metric_hand_state_accepted_rows": hand_depth_repair_graph[
                    "metric_hand_state_accepted_rows"
                ],
                "hand_depth_repair_graph_depth_repair_factor_candidate_rows": hand_depth_repair_graph[
                    "depth_repair_factor_candidate_rows"
                ],
                "hand_depth_repair_graph_state_counts": hand_depth_repair_graph[
                    "solver_state_counts"
                ],
                "hand_depth_repair_graph_owner_depth_state_counts": hand_depth_repair_graph[
                    "owner_depth_state_counts"
                ],
                "hand_depth_repair_graph_owner_median_gap_m": hand_depth_repair_graph[
                    "owner_median_gap_m"
                ],
                "hand_depth_repair_residual_candidate_rows": hand_depth_repair_residual_owner_state[
                    "repair_residual_factor_candidate_rows"
                ],
                "hand_depth_repair_residual_supported_rows": hand_depth_repair_residual_owner_state[
                    "independent_supported_repair_residual_rows"
                ],
                "hand_depth_repair_residual_unsupported_rows": hand_depth_repair_residual_owner_state[
                    "independent_unsupported_repair_residual_rows"
                ],
                "hand_depth_repair_residual_support_state_counts": hand_depth_repair_residual_owner_state[
                    "residual_independent_support_state_counts"
                ],
                "hand_depth_repair_residual_depth_observation_state_counts": hand_depth_repair_residual_owner_state[
                    "residual_depth_observation_state_counts"
                ],
                "supported_hand_depth_repair_residual_depth_observation_state_counts": hand_depth_repair_residual_owner_state[
                    "supported_residual_depth_observation_state_counts"
                ],
                "hand_depth_repair_residual_owner_state_counts": hand_depth_repair_residual_owner_state[
                    "residual_owner_state_counts"
                ],
                "hand_depth_repair_residual_sample_count": hand_depth_repair_residual_owner_state[
                    "residual_sample_count"
                ],
                "hand_local_projection_repair_factor_candidate_rows": hand_local_projection_repair_problem[
                    "local_projection_repair_factor_candidate_rows"
                ],
                "hand_local_projection_mixed_owner_rows": hand_local_projection_repair_problem[
                    "partial_projection_depth_mixed_owner_rows"
                ],
                "hand_local_projection_depth_observation_owner_rows": hand_local_projection_repair_problem[
                    "depth_observation_or_occlusion_owner_rows"
                ],
                "hand_local_projection_support_unresolved_rows": hand_local_projection_repair_problem[
                    "projection_support_unresolved_rows"
                ],
                "hand_local_projection_repair_state_counts": hand_local_projection_repair_problem[
                    "residual_local_projection_repair_state_counts"
                ],
                "hand_local_projection_assignment": hand_local_projection_repair_problem[
                    "local_projection_assignment"
                ],
                "mano_parameter_owned_residual_rows": mano_parameter_ownership_state[
                    "residual_mano_parameter_owned_rows"
                ],
                "mano_parameter_ownership_state_counts": mano_parameter_ownership_state[
                    "residual_mano_parameter_ownership_state_counts"
                ],
                "mano_parameter_owned_alignment_error_summary": mano_parameter_ownership_state[
                    "owned_alignment_error_summary"
                ],
                "mano_parameter_local_projection_articulation_factor_candidate_rows": mano_parameter_ownership_state[
                    "local_projection_articulation_factor_candidate_rows"
                ],
                "mano_parameter_mixed_projection_articulation_observation_candidate_rows": mano_parameter_ownership_state[
                    "mixed_projection_articulation_observation_candidate_rows"
                ],
                "surface_depth_tail_scalar_compatible_rows": hand_surface_depth_tail_state[
                    "scalar_depth_compatible_rows"
                ],
                "surface_depth_tail_factor_candidate_rows": hand_surface_depth_tail_state[
                    "scalar_depth_tail_factor_candidate_rows"
                ],
                "surface_depth_tail_candidate_pattern_counts": hand_surface_depth_tail_state[
                    "tail_candidate_pattern_counts"
                ],
                "surface_depth_tail_candidate_owner_partition_counts": hand_surface_depth_tail_state[
                    "tail_candidate_owner_partition_counts"
                ],
                "surface_depth_tail_candidate_abs_gap_p95_m": hand_surface_depth_tail_state[
                    "tail_candidate_abs_gap_p95_m"
                ],
                "surface_depth_tail_candidate_row_scale_ratio_spread_p95_minus_p05": hand_surface_depth_tail_state[
                    "tail_candidate_row_scale_ratio_spread_p95_minus_p05"
                ],
                "tail_support_candidate_rows": hand_tail_support_state[
                    "tail_factor_candidate_rows"
                ],
                "tail_selected_support_state_counts": hand_tail_support_state[
                    "tail_selected_support_state_counts"
                ],
                "tail_independent_support_state_counts": hand_tail_support_state[
                    "tail_independent_support_state_counts"
                ],
                "tail_abs_sample_count": hand_tail_support_state["tail_abs_sample_count"],
                "tail_negative_sample_count": hand_tail_support_state["tail_negative_sample_count"],
                "tail_positive_sample_count": hand_tail_support_state["tail_positive_sample_count"],
                "tail_depth_observation_candidate_rows": hand_tail_depth_observation_state[
                    "tail_factor_candidate_rows"
                ],
                "tail_depth_independent_supported_candidate_rows": hand_tail_depth_observation_state[
                    "independent_supported_tail_candidate_rows"
                ],
                "tail_depth_independent_unsupported_candidate_rows": hand_tail_depth_observation_state[
                    "independent_unsupported_tail_candidate_rows"
                ],
                "tail_depth_observation_state_counts": hand_tail_depth_observation_state[
                    "tail_depth_observation_state_counts"
                ],
                "supported_tail_depth_observation_state_counts": hand_tail_depth_observation_state[
                    "supported_tail_depth_observation_state_counts"
                ],
                "source_camera_solve_status_counts": hand_depth_factor_problem[
                    "source_camera_solve_status_counts"
                ],
                "sparse_graph_hand_ray_shift_m": hand_depth_factor_problem[
                    "sparse_graph_hand_ray_shift_m"
                ],
                "hand_metric_depth_all_pixels_summary": hand_metric_depth_state[
                    "partition_summaries"
                ]["all_projected_hand_pixels"],
                "hand_metric_depth_far_from_object_summary": hand_metric_depth_state[
                    "partition_summaries"
                ]["far_from_active_object_masks"],
                "hand_metric_depth_near_object_summary": hand_metric_depth_state[
                    "partition_summaries"
                ]["near_active_object_masks"],
                "depth_contact_owner_incompatibility_count": depth_contact_consistency[
                    "depth_owner_incompatibility_count"
                ],
            },
            [
                "visible surfaces are now materialized as fixed measurements where mask and metric depth overlap",
                "depth/object/camera contradictions are not jointly optimized",
                "accepted object reconstructions, legacy object centers, and MANO hands do not currently share one depth owner",
                "hand source-camera translation is not solved against UniDepth in the current accepted hand stream",
                "UniDepth-aligned intrinsics alone leave thousands of hand-depth repair candidates",
                "stable hand-scale counterfactuals leave thousands of depth-repair candidates and imply implausibly small hands",
                "bounded hand-depth repair leaves many residual depth-repair candidates after exact post-solve surface resampling",
                "per-sample residual ownership separates unsupported projection rows, local hand-surface/projection rows, and depth-observation rows",
                "local projection assignment exposes which residual rows can become local hand-surface factors and which rows remain depth-observation or support owners",
                "MANO parameter ownership now identifies which local projection factors can attach to saved MANO pose parameters before an articulation solve",
                "per-row scalar repair exposes visible hand-surface depth tails that require local hand/depth state",
                "independent model-produced hand boxes support most residual depth-tail pixels, with unsupported projection accounting for a small minority",
                "local UniDepth search finds a mixed observation state: some supported tails have nearby compatible depth, some have partial compatible depth, and hundreds lack nearby compatible depth",
                "current MANO depth fails against UniDepth even before object-contact ownership can create physical factors",
                "projected image-contact MANO vertices sit behind the object UniDepth surface in the current hand state",
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
                "depth_contact_legacy_owner_mismatch_frame_count": depth_contact_consistency[
                    "legacy_owner_mismatch_frame_count"
                ],
                "contact_owner_variables_with_geometry_supported_candidate": contact_ownership_problem[
                    "contact_owner_variables_with_geometry_supported_candidate"
                ],
                "contact_owner_image_supported_candidate_rows": contact_ownership_problem[
                    "contact_owner_image_supported_candidate_rows"
                ],
                "contact_owner_metric_depth_supported_candidate_rows": contact_ownership_problem[
                    "contact_owner_metric_depth_supported_candidate_rows"
                ],
                "contact_owner_factor_ready_rows": contact_ownership_problem[
                    "contact_owner_factor_ready_rows"
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
                "physical contact terms cannot attach to accepted reconstruction meshes until contact ownership names the same object id",
                "physical contact terms cannot attach to any active object until contact-owner variables have geometry-supported candidates",
                "physical contact terms cannot attach to image-supported owners until pairwise metric depth is compatible with the object depth state",
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
    pairwise_contact_state_report = require_dict(
        load_json(inputs.pairwise_contact_state_report),
        f"{inputs.case} pairwise contact state report",
    )
    pairwise_contact_depth_gap_report = require_dict(
        load_json(inputs.pairwise_contact_depth_gap_report),
        f"{inputs.case} pairwise contact depth-gap report",
    )
    hand_metric_depth_state_report = require_dict(
        load_json(inputs.hand_metric_depth_state_report),
        f"{inputs.case} hand metric-depth state report",
    )
    hand_depth_factor_problem_report = require_dict(
        load_json(inputs.hand_depth_factor_problem_report),
        f"{inputs.case} hand-depth factor problem report",
    )
    hand_intrinsics_depth_counterfactual_report = require_dict(
        load_json(inputs.hand_intrinsics_depth_counterfactual_report),
        f"{inputs.case} hand intrinsics-depth counterfactual report",
    )
    hand_scale_depth_counterfactual_report = require_dict(
        load_json(inputs.hand_scale_depth_counterfactual_report),
        f"{inputs.case} hand scale-depth counterfactual report",
    )
    hand_depth_repair_graph_report = require_dict(
        load_json(inputs.hand_depth_repair_graph_report),
        f"{inputs.case} hand depth repair graph report",
    )
    hand_depth_repair_residual_owner_state_report = require_dict(
        load_json(inputs.hand_depth_repair_residual_owner_state_report),
        f"{inputs.case} hand depth repair residual-owner state report",
    )
    hand_local_projection_repair_problem_report = require_dict(
        load_json(inputs.hand_local_projection_repair_problem_report),
        f"{inputs.case} hand local projection repair problem report",
    )
    mano_parameter_ownership_state_report = require_dict(
        load_json(inputs.mano_parameter_ownership_state_report),
        f"{inputs.case} MANO parameter ownership state report",
    )
    hand_surface_depth_tail_state_report = require_dict(
        load_json(inputs.hand_surface_depth_tail_state_report),
        f"{inputs.case} hand surface-depth tail state report",
    )
    hand_tail_support_state_report = require_dict(
        load_json(inputs.hand_tail_support_state_report),
        f"{inputs.case} hand tail support state report",
    )
    hand_tail_depth_observation_state_report = require_dict(
        load_json(inputs.hand_tail_depth_observation_state_report),
        f"{inputs.case} hand tail depth-observation state report",
    )
    contact_ownership_problem_report = require_dict(
        load_json(inputs.contact_ownership_problem_report),
        f"{inputs.case} contact-ownership problem report",
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
    pairwise_contact_state = pairwise_contact_state_counts(pairwise_contact_state_report)
    pairwise_contact_depth_gap = pairwise_contact_depth_gap_counts(pairwise_contact_depth_gap_report)
    hand_metric_depth_state = hand_metric_depth_state_counts(hand_metric_depth_state_report)
    hand_depth_factor_problem = hand_depth_factor_problem_counts(hand_depth_factor_problem_report)
    hand_intrinsics_depth_counterfactual = hand_intrinsics_depth_counterfactual_counts(
        hand_intrinsics_depth_counterfactual_report
    )
    hand_scale_depth_counterfactual = hand_scale_depth_counterfactual_counts(hand_scale_depth_counterfactual_report)
    hand_depth_repair_graph = hand_depth_repair_graph_counts(hand_depth_repair_graph_report)
    hand_depth_repair_residual_owner_state = hand_depth_repair_residual_owner_state_counts(
        hand_depth_repair_residual_owner_state_report
    )
    hand_local_projection_repair_problem = hand_local_projection_repair_problem_counts(
        hand_local_projection_repair_problem_report
    )
    mano_parameter_ownership_state = mano_parameter_ownership_state_counts(
        mano_parameter_ownership_state_report
    )
    hand_surface_depth_tail_state = hand_surface_depth_tail_state_counts(hand_surface_depth_tail_state_report)
    hand_tail_support_state = hand_tail_support_state_counts(hand_tail_support_state_report)
    hand_tail_depth_observation_state = hand_tail_depth_observation_state_counts(
        hand_tail_depth_observation_state_report
    )
    contact_ownership_problem = contact_ownership_problem_counts(contact_ownership_problem_report)
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
    if frame_count != require_int(pairwise_contact_state["frame_count"], f"{inputs.case} pairwise contact frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and pairwise contact state")
    if frame_count != require_int(pairwise_contact_depth_gap["frame_count"], f"{inputs.case} pairwise depth-gap frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and pairwise contact depth-gap")
    if frame_count != require_int(hand_metric_depth_state["frame_count"], f"{inputs.case} hand metric-depth frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and hand metric-depth state")
    if frame_count != require_int(hand_depth_factor_problem["frame_count"], f"{inputs.case} hand-depth factor frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and hand-depth factor problem")
    if frame_count != require_int(
        hand_intrinsics_depth_counterfactual["frame_count"],
        f"{inputs.case} hand intrinsics-depth counterfactual frame_count",
    ):
        raise RuntimeError(
            f"{inputs.case} frame_count mismatch between sparse report and hand intrinsics-depth counterfactual"
        )
    if frame_count != require_int(
        hand_scale_depth_counterfactual["frame_count"],
        f"{inputs.case} hand scale-depth counterfactual frame_count",
    ):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and hand scale-depth counterfactual")
    if frame_count != require_int(
        hand_depth_repair_graph["frame_count"],
        f"{inputs.case} hand depth repair graph frame_count",
    ):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and hand depth repair graph")
    if frame_count != require_int(
        hand_depth_repair_residual_owner_state["frame_count"],
        f"{inputs.case} hand depth repair residual-owner frame_count",
    ):
        raise RuntimeError(
            f"{inputs.case} frame_count mismatch between sparse report and hand depth repair residual-owner state"
        )
    if frame_count != require_int(
        hand_local_projection_repair_problem["frame_count"],
        f"{inputs.case} hand local projection repair frame_count",
    ):
        raise RuntimeError(
            f"{inputs.case} frame_count mismatch between sparse report and hand local projection repair problem"
        )
    if frame_count != require_int(
        mano_parameter_ownership_state["frame_count"],
        f"{inputs.case} MANO parameter ownership frame_count",
    ):
        raise RuntimeError(
            f"{inputs.case} frame_count mismatch between sparse report and MANO parameter ownership state"
        )
    if frame_count != require_int(
        hand_surface_depth_tail_state["frame_count"],
        f"{inputs.case} hand surface-depth tail frame_count",
    ):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and hand surface-depth tail state")
    if frame_count != require_int(
        hand_tail_support_state["frame_count"],
        f"{inputs.case} hand tail support frame_count",
    ):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and hand tail support state")
    if frame_count != require_int(
        hand_tail_depth_observation_state["frame_count"],
        f"{inputs.case} hand tail depth-observation frame_count",
    ):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and hand tail depth-observation state")
    if frame_count != require_int(contact_ownership_problem["frame_count"], f"{inputs.case} contact ownership frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and contact ownership problem")
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
    if require_int(pairwise_contact_state["pairwise_contact_variable_count"], f"{inputs.case} pairwise contact rows") != require_int(
        multi_object_contact_evidence["hand_object_rows"],
        f"{inputs.case} multi-object hand-object rows",
    ):
        raise RuntimeError(f"{inputs.case} pairwise contact rows disagree with multi-object hand-object rows")
    if require_int(
        pairwise_contact_state["physical_contact_factor_ready_rows"],
        f"{inputs.case} pairwise physical factor rows",
    ) != 0:
        raise RuntimeError(f"{inputs.case} pairwise image-contact layer must not emit physical contact factors")
    if require_int(
        pairwise_contact_depth_gap["pairwise_contact_variable_count"],
        f"{inputs.case} pairwise depth-gap contact variable rows",
    ) != require_int(
        pairwise_contact_state["pairwise_contact_variable_count"],
        f"{inputs.case} pairwise contact rows",
    ):
        raise RuntimeError(f"{inputs.case} pairwise depth-gap variable count disagrees with pairwise contact state")
    if require_int(
        pairwise_contact_depth_gap["evaluated_pair_depth_rows"],
        f"{inputs.case} pairwise depth-gap evaluated rows",
    ) != require_int(
        pairwise_contact_state["pair_contact_image_candidate_rows"],
        f"{inputs.case} pairwise image candidate rows",
    ):
        raise RuntimeError(f"{inputs.case} pairwise depth-gap evaluated rows disagree with image-contact candidates")
    if require_int(
        pairwise_contact_depth_gap["metric_depth_compatible_candidate_rows"],
        f"{inputs.case} pairwise depth-gap compatible rows",
    ) != require_int(
        contact_ownership_problem["pairwise_metric_depth_compatible_candidate_rows"],
        f"{inputs.case} contact ownership metric-depth compatible rows",
    ):
        raise RuntimeError(f"{inputs.case} pairwise depth-gap compatible rows disagree with contact ownership")
    if require_int(
        pairwise_contact_depth_gap["physical_contact_factor_ready_rows"],
        f"{inputs.case} pairwise depth-gap physical factor rows",
    ) != 0:
        raise RuntimeError(f"{inputs.case} pairwise depth-gap layer must not emit physical contact factors")
    pairwise_comparison = require_dict(
        hand_metric_depth_state["pairwise_contact_depth_gap_comparison"],
        f"{inputs.case} hand metric-depth pairwise comparison",
    )
    if require_int(
        pairwise_comparison.get("evaluated_pair_depth_rows"),
        f"{inputs.case} hand metric-depth pairwise evaluated rows",
    ) != require_int(
        pairwise_contact_depth_gap["evaluated_pair_depth_rows"],
        f"{inputs.case} pairwise depth-gap evaluated rows",
    ):
        raise RuntimeError(f"{inputs.case} hand metric-depth report disagrees with pairwise depth-gap evaluated rows")
    if require_int(
        pairwise_comparison.get("metric_depth_compatible_candidate_rows"),
        f"{inputs.case} hand metric-depth pairwise compatible rows",
    ) != require_int(
        pairwise_contact_depth_gap["metric_depth_compatible_candidate_rows"],
        f"{inputs.case} pairwise depth-gap compatible rows",
    ):
        raise RuntimeError(f"{inputs.case} hand metric-depth report disagrees with pairwise depth-gap compatible rows")
    if require_int(
        hand_depth_factor_problem["hand_depth_variable_count"],
        f"{inputs.case} hand-depth factor variable count",
    ) != require_int(
        hand_metric_depth_state["hand_metric_depth_variable_count"],
        f"{inputs.case} hand metric-depth variable count",
    ):
        raise RuntimeError(f"{inputs.case} hand-depth factor variable count disagrees with hand metric-depth state")
    if require_int(
        hand_depth_factor_problem["metric_depth_factor_rows"],
        f"{inputs.case} hand-depth metric factor rows",
    ) != require_int(
        hand_metric_depth_state["measured_hand_depth_rows"],
        f"{inputs.case} hand metric-depth measured rows",
    ):
        raise RuntimeError(f"{inputs.case} hand-depth metric factor rows disagree with hand metric-depth state")
    if require_int(
        hand_depth_factor_problem["projection_factor_ready_rows"],
        f"{inputs.case} hand-depth projection factor rows",
    ) != require_int(
        hand_metric_depth_state["projection_residual_ok_hand_rows"],
        f"{inputs.case} hand metric-depth projection residual ok rows",
    ):
        raise RuntimeError(f"{inputs.case} hand-depth projection factor rows disagree with hand metric-depth state")
    if require_int(
        hand_depth_factor_problem["metric_hand_state_accepted_rows"],
        f"{inputs.case} accepted hand metric rows",
    ) != require_int(
        hand_metric_depth_state["hand_metric_depth_state_counts"].get("metric_depth_compatible", 0),
        f"{inputs.case} hand metric-depth compatible rows",
    ):
        raise RuntimeError(f"{inputs.case} accepted hand metric rows disagree with hand metric-depth state")
    if require_int(
        hand_intrinsics_depth_counterfactual["hand_intrinsics_counterfactual_variable_count"],
        f"{inputs.case} hand intrinsics counterfactual variable count",
    ) != require_int(
        hand_metric_depth_state["hand_metric_depth_variable_count"],
        f"{inputs.case} hand metric-depth variable count",
    ):
        raise RuntimeError(f"{inputs.case} hand intrinsics counterfactual variable count disagrees with hand metric-depth state")
    if require_int(
        hand_intrinsics_depth_counterfactual["counterfactual_metric_hand_state_accepted_rows"],
        f"{inputs.case} hand intrinsics counterfactual accepted rows",
    ) >= require_int(
        hand_intrinsics_depth_counterfactual["counterfactual_projection_factor_ready_rows"],
        f"{inputs.case} hand intrinsics counterfactual projection rows",
    ):
        raise RuntimeError(f"{inputs.case} hand intrinsics counterfactual cannot be interpreted as a failed repair")
    if require_int(
        hand_scale_depth_counterfactual["hand_scale_counterfactual_variable_count"],
        f"{inputs.case} hand scale counterfactual variable count",
    ) != require_int(
        hand_metric_depth_state["hand_metric_depth_variable_count"],
        f"{inputs.case} hand metric-depth variable count",
    ):
        raise RuntimeError(f"{inputs.case} hand scale counterfactual variable count disagrees with hand metric-depth state")
    if require_int(
        hand_depth_repair_graph["hand_depth_repair_graph_variable_count"],
        f"{inputs.case} hand depth repair graph variable count",
    ) != require_int(
        hand_metric_depth_state["hand_metric_depth_variable_count"],
        f"{inputs.case} hand metric-depth variable count",
    ):
        raise RuntimeError(f"{inputs.case} hand depth repair graph variable count disagrees with hand metric-depth state")
    if require_int(
        hand_depth_repair_graph["base_available_rows"],
        f"{inputs.case} hand depth repair graph base rows",
    ) != require_int(
        hand_scale_depth_counterfactual["base_available_rows"],
        f"{inputs.case} hand scale base rows",
    ):
        raise RuntimeError(f"{inputs.case} hand depth repair graph base rows disagree with hand scale counterfactual")
    if require_int(
        hand_depth_repair_graph["depth_data_candidate_rows"],
        f"{inputs.case} hand depth repair graph data rows",
    ) > require_int(
        hand_metric_depth_state["projection_residual_ok_hand_rows"],
        f"{inputs.case} hand metric-depth projection residual ok rows",
    ):
        raise RuntimeError(f"{inputs.case} hand depth repair graph data rows exceed projection-ready hand rows")
    if require_int(
        hand_depth_repair_graph["depth_data_candidate_rows"],
        f"{inputs.case} hand depth repair graph data rows",
    ) > require_int(
        hand_depth_repair_graph["base_available_rows"],
        f"{inputs.case} hand depth repair graph base rows",
    ):
        raise RuntimeError(f"{inputs.case} hand depth repair graph data rows exceed base-available rows")
    if require_int(
        hand_depth_repair_residual_owner_state["hand_depth_repair_residual_owner_variable_count"],
        f"{inputs.case} hand depth repair residual-owner variable count",
    ) != require_int(
        hand_depth_repair_graph["hand_depth_repair_graph_variable_count"],
        f"{inputs.case} hand depth repair graph variable count",
    ):
        raise RuntimeError(
            f"{inputs.case} hand depth repair residual-owner variables disagree with repair graph variables"
        )
    if require_int(
        hand_depth_repair_residual_owner_state["repair_residual_factor_candidate_rows"],
        f"{inputs.case} hand depth repair residual-owner candidate rows",
    ) != require_int(
        hand_depth_repair_graph["depth_repair_factor_candidate_rows"],
        f"{inputs.case} hand depth repair graph repair rows",
    ):
        raise RuntimeError(
            f"{inputs.case} hand depth repair residual-owner candidates disagree with repair graph residuals"
        )
    if require_int(
        hand_depth_repair_residual_owner_state["independent_supported_repair_residual_rows"],
        f"{inputs.case} supported hand depth repair residual rows",
    ) + require_int(
        hand_depth_repair_residual_owner_state["independent_unsupported_repair_residual_rows"],
        f"{inputs.case} unsupported hand depth repair residual rows",
    ) != require_int(
        hand_depth_repair_residual_owner_state["repair_residual_factor_candidate_rows"],
        f"{inputs.case} hand depth repair residual-owner candidate rows",
    ):
        raise RuntimeError(f"{inputs.case} hand depth repair residual support split does not sum to candidates")
    if require_int(
        hand_local_projection_repair_problem["hand_local_projection_repair_variable_count"],
        f"{inputs.case} hand local projection variable count",
    ) != require_int(
        hand_depth_repair_residual_owner_state["hand_depth_repair_residual_owner_variable_count"],
        f"{inputs.case} hand depth repair residual-owner variable count",
    ):
        raise RuntimeError(
            f"{inputs.case} hand local projection variables disagree with residual-owner variables"
        )
    if require_int(
        hand_local_projection_repair_problem["repair_residual_factor_candidate_rows"],
        f"{inputs.case} hand local projection residual rows",
    ) != require_int(
        hand_depth_repair_residual_owner_state["repair_residual_factor_candidate_rows"],
        f"{inputs.case} hand depth repair residual-owner candidate rows",
    ):
        raise RuntimeError(
            f"{inputs.case} hand local projection residual rows disagree with residual-owner candidates"
        )
    if require_int(
        hand_local_projection_repair_problem["local_projection_repair_factor_candidate_rows"],
        f"{inputs.case} local projection repair rows",
    ) + require_int(
        hand_local_projection_repair_problem["partial_projection_depth_mixed_owner_rows"],
        f"{inputs.case} mixed projection-depth rows",
    ) + require_int(
        hand_local_projection_repair_problem["depth_observation_or_occlusion_owner_rows"],
        f"{inputs.case} depth observation owner rows",
    ) + require_int(
        hand_local_projection_repair_problem["projection_support_unresolved_rows"],
        f"{inputs.case} projection support unresolved rows",
    ) != require_int(
        hand_local_projection_repair_problem["repair_residual_factor_candidate_rows"],
        f"{inputs.case} local projection residual rows",
    ):
        raise RuntimeError(f"{inputs.case} local projection residual split does not sum to residual rows")
    if require_int(
        hand_local_projection_repair_problem["projection_support_unresolved_rows"],
        f"{inputs.case} projection support unresolved rows",
    ) != require_int(
        hand_depth_repair_residual_owner_state["independent_unsupported_repair_residual_rows"],
        f"{inputs.case} residual-owner unsupported rows",
    ):
        raise RuntimeError(f"{inputs.case} local projection support-unresolved rows disagree with residual owner")
    if require_int(
        mano_parameter_ownership_state["mano_parameter_ownership_variable_count"],
        f"{inputs.case} MANO parameter ownership variable count",
    ) != require_int(
        hand_local_projection_repair_problem["hand_local_projection_repair_variable_count"],
        f"{inputs.case} hand local projection variable count",
    ):
        raise RuntimeError(
            f"{inputs.case} MANO parameter ownership variables disagree with local projection variables"
        )
    if require_int(
        mano_parameter_ownership_state["repair_residual_factor_candidate_rows"],
        f"{inputs.case} MANO parameter ownership residual rows",
    ) != require_int(
        hand_local_projection_repair_problem["repair_residual_factor_candidate_rows"],
        f"{inputs.case} hand local projection residual rows",
    ):
        raise RuntimeError(
            f"{inputs.case} MANO parameter ownership residual rows disagree with local projection residuals"
        )
    if require_int(
        mano_parameter_ownership_state["local_projection_repair_factor_candidate_rows"],
        f"{inputs.case} MANO ownership local projection rows",
    ) != require_int(
        hand_local_projection_repair_problem["local_projection_repair_factor_candidate_rows"],
        f"{inputs.case} local projection repair rows",
    ):
        raise RuntimeError(f"{inputs.case} MANO ownership local projection rows disagree with local projection report")
    if require_int(
        mano_parameter_ownership_state["local_projection_articulation_factor_candidate_rows"],
        f"{inputs.case} MANO local articulation factor rows",
    ) > require_int(
        mano_parameter_ownership_state["local_projection_repair_factor_candidate_rows"],
        f"{inputs.case} MANO ownership local projection rows",
    ):
        raise RuntimeError(f"{inputs.case} MANO local articulation rows exceed local projection rows")
    if require_int(
        mano_parameter_ownership_state["mixed_projection_articulation_observation_candidate_rows"],
        f"{inputs.case} MANO mixed articulation rows",
    ) > require_int(
        hand_local_projection_repair_problem["partial_projection_depth_mixed_owner_rows"],
        f"{inputs.case} mixed projection-depth rows",
    ):
        raise RuntimeError(f"{inputs.case} MANO mixed articulation rows exceed mixed projection-depth rows")
    if require_int(
        hand_scale_depth_counterfactual["base_available_rows"],
        f"{inputs.case} hand scale base available rows",
    ) != require_int(
        hand_intrinsics_depth_counterfactual["counterfactual_metric_depth_measured_rows"],
        f"{inputs.case} hand intrinsics counterfactual measured rows",
    ):
        raise RuntimeError(f"{inputs.case} hand scale base rows disagree with hand intrinsics counterfactual measured rows")
    scale_intrinsics_comparison = require_dict(
        hand_scale_depth_counterfactual["source_intrinsics_counterfactual_comparison"],
        f"{inputs.case} hand scale source intrinsics comparison",
    )
    if require_int(
        scale_intrinsics_comparison.get("intrinsics_counterfactual_metric_hand_state_accepted_rows"),
        f"{inputs.case} hand scale comparison accepted rows",
    ) != require_int(
        hand_intrinsics_depth_counterfactual["counterfactual_metric_hand_state_accepted_rows"],
        f"{inputs.case} hand intrinsics counterfactual accepted rows",
    ):
        raise RuntimeError(f"{inputs.case} hand scale report disagrees with hand intrinsics counterfactual")
    if require_int(
        hand_surface_depth_tail_state["hand_surface_depth_tail_variable_count"],
        f"{inputs.case} hand surface-depth tail variable count",
    ) != require_int(
        hand_scale_depth_counterfactual["hand_scale_counterfactual_variable_count"],
        f"{inputs.case} hand scale counterfactual variable count",
    ):
        raise RuntimeError(f"{inputs.case} hand surface-depth tail variable count disagrees with hand scale counterfactual")
    scale_oracle = require_dict(
        hand_scale_depth_counterfactual["per_row_scale_oracle_mode"],
        f"{inputs.case} hand scale per-row oracle mode",
    )
    if require_int(
        hand_surface_depth_tail_state["scalar_depth_compatible_rows"],
        f"{inputs.case} scalar depth compatible rows",
    ) != require_int(
        scale_oracle.get("metric_hand_state_accepted_rows"),
        f"{inputs.case} scale oracle accepted rows",
    ):
        raise RuntimeError(f"{inputs.case} hand surface-depth compatible rows disagree with scale oracle")
    if require_int(
        hand_surface_depth_tail_state["scalar_depth_tail_factor_candidate_rows"],
        f"{inputs.case} scalar depth tail factor candidate rows",
    ) != require_int(
        scale_oracle.get("depth_repair_factor_candidate_rows"),
        f"{inputs.case} scale oracle repair rows",
    ):
        raise RuntimeError(f"{inputs.case} hand surface-depth tail candidates disagree with scale oracle")
    if require_int(
        hand_tail_support_state["hand_tail_support_variable_count"],
        f"{inputs.case} hand tail support variable count",
    ) != require_int(
        hand_surface_depth_tail_state["hand_surface_depth_tail_variable_count"],
        f"{inputs.case} hand surface-depth tail variable count",
    ):
        raise RuntimeError(f"{inputs.case} hand tail support variable count disagrees with hand surface-depth tails")
    if require_int(
        hand_tail_support_state["tail_factor_candidate_rows"],
        f"{inputs.case} hand tail support candidate rows",
    ) != require_int(
        hand_surface_depth_tail_state["scalar_depth_tail_factor_candidate_rows"],
        f"{inputs.case} hand surface-depth tail candidate rows",
    ):
        raise RuntimeError(f"{inputs.case} hand tail support candidates disagree with hand surface-depth tails")
    if require_int(
        hand_tail_depth_observation_state["hand_tail_depth_observation_variable_count"],
        f"{inputs.case} hand tail depth-observation variable count",
    ) != require_int(
        hand_tail_support_state["hand_tail_support_variable_count"],
        f"{inputs.case} hand tail support variable count",
    ):
        raise RuntimeError(f"{inputs.case} hand tail depth-observation variable count disagrees with hand tail support")
    if require_int(
        hand_tail_depth_observation_state["tail_factor_candidate_rows"],
        f"{inputs.case} hand tail depth-observation candidate rows",
    ) != require_int(
        hand_tail_support_state["tail_factor_candidate_rows"],
        f"{inputs.case} hand tail support candidate rows",
    ):
        raise RuntimeError(f"{inputs.case} hand tail depth-observation candidates disagree with hand tail support")
    unsupported_support = require_int(
        hand_tail_support_state["tail_independent_support_state_counts"].get(
            "tail_pixels_unsupported_by_independent_model_boxes",
            0,
        ),
        f"{inputs.case} hand tail unsupported independent support rows",
    )
    if require_int(
        hand_tail_depth_observation_state["independent_unsupported_tail_candidate_rows"],
        f"{inputs.case} hand tail depth-observation unsupported rows",
    ) != unsupported_support:
        raise RuntimeError(f"{inputs.case} hand tail depth-observation unsupported count disagrees with support state")
    if require_int(contact["contact_factor_ready_count"], f"{inputs.case} contact ready rows") != require_int(
        contact_ownership_problem["contact_owner_variable_count"],
        f"{inputs.case} contact ownership variable count",
    ):
        raise RuntimeError(f"{inputs.case} contact ownership variables disagree with contact-mode ready rows")
    if require_int(
        contact_ownership_problem["contact_owner_variable_count"],
        f"{inputs.case} contact ownership variable count",
    ) != require_int(
        object_geometry_factor_problem["contact_owner_variable_count"],
        f"{inputs.case} object-geometry contact owner variable count",
    ):
        raise RuntimeError(f"{inputs.case} contact ownership variable count disagrees with object-geometry factor problem")
    if require_int(
        contact_ownership_problem["contact_owner_candidate_rows"],
        f"{inputs.case} contact ownership candidate rows",
    ) != require_int(
        object_geometry_factor_problem["contact_owner_candidate_rows"],
        f"{inputs.case} object-geometry contact owner candidate rows",
    ):
        raise RuntimeError(f"{inputs.case} contact ownership candidate rows disagree with object-geometry factor problem")
    if require_int(
        contact_ownership_problem["contact_owner_factor_ready_rows"],
        f"{inputs.case} contact ownership factor ready rows",
    ) != require_int(
        object_geometry_factor_problem["contact_owner_factor_ready_rows"],
        f"{inputs.case} object-geometry contact owner factor ready rows",
    ):
        raise RuntimeError(f"{inputs.case} contact ownership factor-ready rows disagree with object-geometry factor problem")
    if require_int(
        contact_ownership_problem["contact_owner_image_supported_candidate_rows"],
        f"{inputs.case} contact ownership image-supported rows",
    ) != require_int(
        object_geometry_factor_problem["contact_owner_image_supported_candidate_rows"],
        f"{inputs.case} object-geometry contact owner image-supported rows",
    ):
        raise RuntimeError(f"{inputs.case} contact ownership image-supported rows disagree with object-geometry factor problem")
    if require_int(
        contact_ownership_problem["contact_owner_image_supported_candidate_rows"],
        f"{inputs.case} contact ownership image-supported rows",
    ) != require_int(
        pairwise_contact_state["contact_owner_image_supported_candidate_rows"],
        f"{inputs.case} pairwise image-supported rows",
    ):
        raise RuntimeError(f"{inputs.case} contact ownership image-supported rows disagree with pairwise contact state")
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
    if require_int(
        depth_contact_consistency["legacy_owner_mismatch_frame_count"],
        f"{inputs.case} depth-contact legacy owner mismatch count",
    ) != require_int(
        object_geometry_factor_problem["depth_contact_legacy_owner_mismatch_frame_count"],
        f"{inputs.case} object-geometry factor depth-contact legacy owner mismatch count",
    ):
        raise RuntimeError(f"{inputs.case} depth-contact legacy owner mismatch count disagrees with factor problem")

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
        pairwise_contact_state,
        pairwise_contact_depth_gap,
        hand_metric_depth_state,
        hand_depth_factor_problem,
        hand_intrinsics_depth_counterfactual,
        hand_scale_depth_counterfactual,
        hand_depth_repair_graph,
        hand_depth_repair_residual_owner_state,
        hand_local_projection_repair_problem,
        mano_parameter_ownership_state,
        hand_surface_depth_tail_state,
        hand_tail_support_state,
        hand_tail_depth_observation_state,
        contact_ownership_problem,
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
            "pairwise_contact_state_report": source_summary(
                inputs.pairwise_contact_state_report, pairwise_contact_state_report
            ),
            "pairwise_contact_depth_gap_report": source_summary(
                inputs.pairwise_contact_depth_gap_report, pairwise_contact_depth_gap_report
            ),
            "hand_metric_depth_state_report": source_summary(
                inputs.hand_metric_depth_state_report, hand_metric_depth_state_report
            ),
            "hand_depth_factor_problem_report": source_summary(
                inputs.hand_depth_factor_problem_report, hand_depth_factor_problem_report
            ),
            "hand_intrinsics_depth_counterfactual_report": source_summary(
                inputs.hand_intrinsics_depth_counterfactual_report, hand_intrinsics_depth_counterfactual_report
            ),
            "hand_scale_depth_counterfactual_report": source_summary(
                inputs.hand_scale_depth_counterfactual_report, hand_scale_depth_counterfactual_report
            ),
            "hand_depth_repair_graph_report": source_summary(
                inputs.hand_depth_repair_graph_report, hand_depth_repair_graph_report
            ),
            "hand_depth_repair_residual_owner_state_report": source_summary(
                inputs.hand_depth_repair_residual_owner_state_report,
                hand_depth_repair_residual_owner_state_report,
            ),
            "hand_local_projection_repair_problem_report": source_summary(
                inputs.hand_local_projection_repair_problem_report,
                hand_local_projection_repair_problem_report,
            ),
            "mano_parameter_ownership_state_report": source_summary(
                inputs.mano_parameter_ownership_state_report,
                mano_parameter_ownership_state_report,
            ),
            "hand_surface_depth_tail_state_report": source_summary(
                inputs.hand_surface_depth_tail_state_report, hand_surface_depth_tail_state_report
            ),
            "hand_tail_support_state_report": source_summary(
                inputs.hand_tail_support_state_report, hand_tail_support_state_report
            ),
            "hand_tail_depth_observation_state_report": source_summary(
                inputs.hand_tail_depth_observation_state_report,
                hand_tail_depth_observation_state_report,
            ),
            "contact_ownership_problem_report": source_summary(
                inputs.contact_ownership_problem_report, contact_ownership_problem_report
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
        "current_pairwise_contact_state": pairwise_contact_state,
        "current_pairwise_contact_depth_gap": pairwise_contact_depth_gap,
        "current_hand_metric_depth_state": hand_metric_depth_state,
        "current_hand_depth_factor_problem": hand_depth_factor_problem,
        "current_hand_intrinsics_depth_counterfactual": hand_intrinsics_depth_counterfactual,
        "current_hand_scale_depth_counterfactual": hand_scale_depth_counterfactual,
        "current_hand_depth_repair_graph": hand_depth_repair_graph,
        "current_hand_depth_repair_residual_owner_state": hand_depth_repair_residual_owner_state,
        "current_hand_local_projection_repair_problem": hand_local_projection_repair_problem,
        "current_mano_parameter_ownership_state": mano_parameter_ownership_state,
        "current_hand_surface_depth_tail_state": hand_surface_depth_tail_state,
        "current_hand_tail_support_state": hand_tail_support_state,
        "current_hand_tail_depth_observation_state": hand_tail_depth_observation_state,
        "current_contact_ownership_problem": contact_ownership_problem,
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
            args.pairwise_contact_state_root,
            args.pairwise_contact_depth_gap_root,
            args.hand_metric_depth_state_root,
            args.hand_depth_factor_problem_root,
            args.hand_intrinsics_depth_counterfactual_root,
            args.hand_scale_depth_counterfactual_root,
            args.hand_depth_repair_graph_root,
            args.hand_depth_repair_residual_owner_state_root,
            args.hand_local_projection_repair_problem_root,
            args.mano_parameter_ownership_state_root,
            args.hand_surface_depth_tail_state_root,
            args.hand_tail_support_state_root,
            args.hand_tail_depth_observation_state_root,
            args.contact_ownership_problem_root,
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
        "pairwise_contact_state_root": str(args.pairwise_contact_state_root),
        "pairwise_contact_depth_gap_root": str(args.pairwise_contact_depth_gap_root),
        "hand_metric_depth_state_root": str(args.hand_metric_depth_state_root),
        "hand_depth_factor_problem_root": str(args.hand_depth_factor_problem_root),
        "hand_intrinsics_depth_counterfactual_root": str(args.hand_intrinsics_depth_counterfactual_root),
        "hand_scale_depth_counterfactual_root": str(args.hand_scale_depth_counterfactual_root),
        "hand_depth_repair_graph_root": str(args.hand_depth_repair_graph_root),
        "hand_depth_repair_residual_owner_state_root": str(args.hand_depth_repair_residual_owner_state_root),
        "hand_local_projection_repair_problem_root": str(args.hand_local_projection_repair_problem_root),
        "mano_parameter_ownership_state_root": str(args.mano_parameter_ownership_state_root),
        "hand_surface_depth_tail_state_root": str(args.hand_surface_depth_tail_state_root),
        "hand_tail_support_state_root": str(args.hand_tail_support_state_root),
        "hand_tail_depth_observation_state_root": str(args.hand_tail_depth_observation_state_root),
        "contact_ownership_problem_root": str(args.contact_ownership_problem_root),
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
                "pairwise_contact_variable_count": case[
                    "current_pairwise_contact_state"
                ]["pairwise_contact_variable_count"],
                "pairwise_measured_image_pair_rows": case[
                    "current_pairwise_contact_state"
                ]["measured_image_pair_rows"],
                "pairwise_image_overlap_candidate_rows": case[
                    "current_pairwise_contact_state"
                ]["image_overlap_candidate_rows"],
                "pair_contact_image_candidate_rows": case[
                    "current_pairwise_contact_state"
                ]["pair_contact_image_candidate_rows"],
                "pairwise_physical_contact_factor_ready_rows": case[
                    "current_pairwise_contact_state"
                ]["physical_contact_factor_ready_rows"],
                "pairwise_metric_depth_evaluated_rows": case[
                    "current_pairwise_contact_depth_gap"
                ]["evaluated_pair_depth_rows"],
                "pairwise_metric_depth_compatible_candidate_rows": case[
                    "current_pairwise_contact_depth_gap"
                ]["metric_depth_compatible_candidate_rows"],
                "pairwise_metric_depth_state_counts": case[
                    "current_pairwise_contact_depth_gap"
                ]["depth_gap_state_counts"],
                "hand_metric_depth_variable_count": case[
                    "current_hand_metric_depth_state"
                ]["hand_metric_depth_variable_count"],
                "hand_metric_depth_measured_rows": case[
                    "current_hand_metric_depth_state"
                ]["measured_hand_depth_rows"],
                "hand_metric_depth_projection_residual_ok_rows": case[
                    "current_hand_metric_depth_state"
                ]["projection_residual_ok_hand_rows"],
                "hand_metric_depth_state_counts": case[
                    "current_hand_metric_depth_state"
                ]["hand_metric_depth_state_counts"],
                "hand_depth_factor_problem_state_counts": case[
                    "current_hand_depth_factor_problem"
                ]["factor_problem_state_counts"],
                "hand_depth_repair_factor_candidate_rows": case[
                    "current_hand_depth_factor_problem"
                ]["depth_repair_factor_candidate_rows"],
                "metric_hand_state_accepted_rows": case[
                    "current_hand_depth_factor_problem"
                ]["metric_hand_state_accepted_rows"],
                "hand_depth_source_camera_solve_status_counts": case[
                    "current_hand_depth_factor_problem"
                ]["source_camera_solve_status_counts"],
                "hand_intrinsics_counterfactual_variable_count": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["hand_intrinsics_counterfactual_variable_count"],
                "hand_intrinsics_counterfactual_metric_depth_measured_rows": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["counterfactual_metric_depth_measured_rows"],
                "hand_intrinsics_counterfactual_projection_factor_ready_rows": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["counterfactual_projection_factor_ready_rows"],
                "hand_intrinsics_counterfactual_depth_repair_factor_candidate_rows": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["counterfactual_depth_repair_factor_candidate_rows"],
                "hand_intrinsics_counterfactual_median_gap_improved_rows": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["counterfactual_median_gap_improved_rows"],
                "hand_intrinsics_counterfactual_metric_hand_state_accepted_rows": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["counterfactual_metric_hand_state_accepted_rows"],
                "hand_intrinsics_counterfactual_state_counts": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["counterfactual_state_counts"],
                "hand_intrinsics_counterfactual_owner_depth_state_counts": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["counterfactual_owner_depth_state_counts"],
                "hand_intrinsics_counterfactual_focal_ratio_fx": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["intrinsics_focal_ratio_fx"],
                "hand_intrinsics_counterfactual_owner_median_gap_m": case[
                    "current_hand_intrinsics_depth_counterfactual"
                ]["counterfactual_owner_median_gap_m"],
                "hand_scale_counterfactual_variable_count": case[
                    "current_hand_scale_depth_counterfactual"
                ]["hand_scale_counterfactual_variable_count"],
                "hand_scale_counterfactual_base_available_rows": case[
                    "current_hand_scale_depth_counterfactual"
                ]["base_available_rows"],
                "hand_scale_counterfactual_scale_candidate_rows": case[
                    "current_hand_scale_depth_counterfactual"
                ]["scale_candidate_rows"],
                "hand_scale_counterfactual_case_global_scale": case[
                    "current_hand_scale_depth_counterfactual"
                ]["case_global_scale"],
                "hand_scale_counterfactual_side_global_scales": case[
                    "current_hand_scale_depth_counterfactual"
                ]["side_global_scales"],
                "hand_scale_counterfactual_case_global_mode": case[
                    "current_hand_scale_depth_counterfactual"
                ]["case_global_scale_mode"],
                "hand_scale_counterfactual_side_global_mode": case[
                    "current_hand_scale_depth_counterfactual"
                ]["side_global_scale_mode"],
                "hand_scale_counterfactual_per_row_oracle_mode": case[
                    "current_hand_scale_depth_counterfactual"
                ]["per_row_scale_oracle_mode"],
                "hand_scale_counterfactual_case_scaled_wrist_to_middle_tip_m": case[
                    "current_hand_scale_depth_counterfactual"
                ]["case_global_scaled_wrist_to_middle_tip_m"],
                "hand_depth_repair_graph_variable_count": case[
                    "current_hand_depth_repair_graph"
                ]["hand_depth_repair_graph_variable_count"],
                "hand_depth_repair_graph_base_available_rows": case[
                    "current_hand_depth_repair_graph"
                ]["base_available_rows"],
                "hand_depth_repair_graph_depth_data_candidate_rows": case[
                    "current_hand_depth_repair_graph"
                ]["depth_data_candidate_rows"],
                "hand_depth_repair_graph_case_global_scale": case[
                    "current_hand_depth_repair_graph"
                ]["case_global_scale"],
                "hand_depth_repair_graph_state_counts": case[
                    "current_hand_depth_repair_graph"
                ]["solver_state_counts"],
                "hand_depth_repair_graph_owner_depth_state_counts": case[
                    "current_hand_depth_repair_graph"
                ]["owner_depth_state_counts"],
                "hand_depth_repair_graph_metric_hand_state_accepted_rows": case[
                    "current_hand_depth_repair_graph"
                ]["metric_hand_state_accepted_rows"],
                "hand_depth_repair_graph_depth_repair_factor_candidate_rows": case[
                    "current_hand_depth_repair_graph"
                ]["depth_repair_factor_candidate_rows"],
                "hand_depth_repair_graph_hand_ray_shift_abs_m": case[
                    "current_hand_depth_repair_graph"
                ]["hand_ray_shift_abs_m"],
                "hand_depth_repair_graph_bound_hit_rows": case[
                    "current_hand_depth_repair_graph"
                ]["hand_ray_shift_bound_hit_rows"],
                "hand_depth_repair_residual_owner_variable_count": case[
                    "current_hand_depth_repair_residual_owner_state"
                ]["hand_depth_repair_residual_owner_variable_count"],
                "hand_depth_repair_residual_factor_candidate_rows": case[
                    "current_hand_depth_repair_residual_owner_state"
                ]["repair_residual_factor_candidate_rows"],
                "hand_depth_repair_residual_independent_supported_rows": case[
                    "current_hand_depth_repair_residual_owner_state"
                ]["independent_supported_repair_residual_rows"],
                "hand_depth_repair_residual_independent_unsupported_rows": case[
                    "current_hand_depth_repair_residual_owner_state"
                ]["independent_unsupported_repair_residual_rows"],
                "hand_depth_repair_residual_independent_support_state_counts": case[
                    "current_hand_depth_repair_residual_owner_state"
                ]["residual_independent_support_state_counts"],
                "hand_depth_repair_residual_depth_observation_state_counts": case[
                    "current_hand_depth_repair_residual_owner_state"
                ]["residual_depth_observation_state_counts"],
                "supported_hand_depth_repair_residual_depth_observation_state_counts": case[
                    "current_hand_depth_repair_residual_owner_state"
                ]["supported_residual_depth_observation_state_counts"],
                "hand_depth_repair_residual_owner_state_counts": case[
                    "current_hand_depth_repair_residual_owner_state"
                ]["residual_owner_state_counts"],
                "hand_depth_repair_residual_sample_count": case[
                    "current_hand_depth_repair_residual_owner_state"
                ]["residual_sample_count"],
                "hand_local_projection_repair_variable_count": case[
                    "current_hand_local_projection_repair_problem"
                ]["hand_local_projection_repair_variable_count"],
                "hand_local_projection_repair_factor_candidate_rows": case[
                    "current_hand_local_projection_repair_problem"
                ]["local_projection_repair_factor_candidate_rows"],
                "hand_local_projection_mixed_owner_rows": case[
                    "current_hand_local_projection_repair_problem"
                ]["partial_projection_depth_mixed_owner_rows"],
                "hand_local_projection_depth_observation_owner_rows": case[
                    "current_hand_local_projection_repair_problem"
                ]["depth_observation_or_occlusion_owner_rows"],
                "hand_local_projection_support_unresolved_rows": case[
                    "current_hand_local_projection_repair_problem"
                ]["projection_support_unresolved_rows"],
                "hand_local_projection_repair_state_counts": case[
                    "current_hand_local_projection_repair_problem"
                ]["residual_local_projection_repair_state_counts"],
                "hand_local_projection_assignment": case[
                    "current_hand_local_projection_repair_problem"
                ]["local_projection_assignment"],
                "mano_parameter_ownership_variable_count": case[
                    "current_mano_parameter_ownership_state"
                ]["mano_parameter_ownership_variable_count"],
                "mano_parameter_owned_residual_rows": case[
                    "current_mano_parameter_ownership_state"
                ]["residual_mano_parameter_owned_rows"],
                "mano_parameter_ownership_state_counts": case[
                    "current_mano_parameter_ownership_state"
                ]["residual_mano_parameter_ownership_state_counts"],
                "mano_parameter_owned_alignment_error_summary": case[
                    "current_mano_parameter_ownership_state"
                ]["owned_alignment_error_summary"],
                "mano_parameter_local_projection_articulation_factor_candidate_rows": case[
                    "current_mano_parameter_ownership_state"
                ]["local_projection_articulation_factor_candidate_rows"],
                "mano_parameter_mixed_projection_articulation_observation_candidate_rows": case[
                    "current_mano_parameter_ownership_state"
                ]["mixed_projection_articulation_observation_candidate_rows"],
                "hand_surface_depth_tail_variable_count": case[
                    "current_hand_surface_depth_tail_state"
                ]["hand_surface_depth_tail_variable_count"],
                "hand_surface_depth_scalar_compatible_rows": case[
                    "current_hand_surface_depth_tail_state"
                ]["scalar_depth_compatible_rows"],
                "hand_surface_depth_tail_factor_candidate_rows": case[
                    "current_hand_surface_depth_tail_state"
                ]["scalar_depth_tail_factor_candidate_rows"],
                "hand_surface_depth_projection_untrusted_after_scalar_scale_rows": case[
                    "current_hand_surface_depth_tail_state"
                ]["projection_untrusted_after_scalar_scale_rows"],
                "hand_surface_depth_tail_candidate_pattern_counts": case[
                    "current_hand_surface_depth_tail_state"
                ]["tail_candidate_pattern_counts"],
                "hand_tail_support_variable_count": case[
                    "current_hand_tail_support_state"
                ]["hand_tail_support_variable_count"],
                "hand_tail_support_factor_candidate_rows": case[
                    "current_hand_tail_support_state"
                ]["tail_factor_candidate_rows"],
                "hand_tail_selected_support_state_counts": case[
                    "current_hand_tail_support_state"
                ]["tail_selected_support_state_counts"],
                "hand_tail_independent_support_state_counts": case[
                    "current_hand_tail_support_state"
                ]["tail_independent_support_state_counts"],
                "hand_tail_abs_sample_count": case[
                    "current_hand_tail_support_state"
                ]["tail_abs_sample_count"],
                "hand_tail_negative_sample_count": case[
                    "current_hand_tail_support_state"
                ]["tail_negative_sample_count"],
                "hand_tail_positive_sample_count": case[
                    "current_hand_tail_support_state"
                ]["tail_positive_sample_count"],
                "hand_tail_depth_observation_variable_count": case[
                    "current_hand_tail_depth_observation_state"
                ]["hand_tail_depth_observation_variable_count"],
                "hand_tail_depth_observation_factor_candidate_rows": case[
                    "current_hand_tail_depth_observation_state"
                ]["tail_factor_candidate_rows"],
                "hand_tail_depth_independent_supported_candidate_rows": case[
                    "current_hand_tail_depth_observation_state"
                ]["independent_supported_tail_candidate_rows"],
                "hand_tail_depth_independent_unsupported_candidate_rows": case[
                    "current_hand_tail_depth_observation_state"
                ]["independent_unsupported_tail_candidate_rows"],
                "hand_tail_depth_observation_state_counts": case[
                    "current_hand_tail_depth_observation_state"
                ]["tail_depth_observation_state_counts"],
                "supported_hand_tail_depth_observation_state_counts": case[
                    "current_hand_tail_depth_observation_state"
                ]["supported_tail_depth_observation_state_counts"],
                "hand_metric_depth_far_from_object_summary": case[
                    "current_hand_metric_depth_state"
                ]["partition_summaries"]["far_from_active_object_masks"],
                "hand_metric_depth_near_object_summary": case[
                    "current_hand_metric_depth_state"
                ]["partition_summaries"]["near_active_object_masks"],
                "contact_owner_variable_count": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_variable_count"],
                "contact_owner_candidate_rows": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_candidate_rows"],
                "contact_owner_variables_with_selected_measurement": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_variables_with_selected_measurement"],
                "contact_owner_variables_without_selected_measurement": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_variables_without_selected_measurement"],
                "contact_owner_variables_with_supported_candidate": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_variables_with_supported_candidate"],
                "contact_owner_variables_with_geometry_supported_candidate": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_variables_with_geometry_supported_candidate"],
                "contact_owner_image_supported_candidate_rows": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_image_supported_candidate_rows"],
                "contact_owner_metric_depth_supported_candidate_rows": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_metric_depth_supported_candidate_rows"],
                "owner_image_variables_with_single_supported_candidate": case[
                    "current_contact_ownership_problem"
                ]["owner_image_variables_with_single_supported_candidate"],
                "owner_image_variables_with_ambiguous_supported_candidates": case[
                    "current_contact_ownership_problem"
                ]["owner_image_variables_with_ambiguous_supported_candidates"],
                "contact_owner_variables_without_supported_candidate": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_variables_without_supported_candidate"],
                "contact_owner_factor_ready_rows": case[
                    "current_contact_ownership_problem"
                ]["contact_owner_factor_ready_rows"],
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
                "depth_contact_legacy_contact_ready_hand_rows": case[
                    "current_depth_contact_consistency_audit"
                ]["legacy_contact_ready_hand_rows"],
                "depth_contact_multi_object_reconstructed_object_contact_candidate_rows": case[
                    "current_depth_contact_consistency_audit"
                ]["multi_object_reconstructed_object_contact_candidate_rows"],
                "depth_contact_legacy_owner_mismatch_frame_count": case[
                    "current_depth_contact_consistency_audit"
                ]["legacy_owner_mismatch_frame_count"],
                "depth_contact_shared_depth_state_ready_frame_count": case[
                    "current_depth_contact_consistency_audit"
                ]["shared_depth_state_ready_frame_count"],
                "depth_contact_owner_incompatibility_count": case[
                    "current_depth_contact_consistency_audit"
                ]["depth_owner_incompatibility_count"],
                "object_geometry_factor_contact_ready_rows": case[
                    "current_object_geometry_factor_problem"
                ]["multi_object_contact_factor_ready_rows"],
                "object_geometry_factor_contact_owner_variable_count": case[
                    "current_object_geometry_factor_problem"
                ]["contact_owner_variable_count"],
                "object_geometry_factor_contact_owner_candidate_rows": case[
                    "current_object_geometry_factor_problem"
                ]["contact_owner_candidate_rows"],
                "object_geometry_factor_contact_owner_image_supported_rows": case[
                    "current_object_geometry_factor_problem"
                ]["contact_owner_image_supported_candidate_rows"],
                "object_geometry_factor_contact_owner_factor_ready_rows": case[
                    "current_object_geometry_factor_problem"
                ]["contact_owner_factor_ready_rows"],
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
        "pairwise_contact_variable_count": sum(
            case["current_pairwise_contact_state"]["pairwise_contact_variable_count"]
            for case in case_outputs
        ),
        "pairwise_measured_image_pair_rows": sum(
            case["current_pairwise_contact_state"]["measured_image_pair_rows"]
            for case in case_outputs
        ),
        "pairwise_image_overlap_candidate_rows": sum(
            case["current_pairwise_contact_state"]["image_overlap_candidate_rows"]
            for case in case_outputs
        ),
        "pair_contact_image_candidate_rows": sum(
            case["current_pairwise_contact_state"]["pair_contact_image_candidate_rows"]
            for case in case_outputs
        ),
        "pairwise_physical_contact_factor_ready_rows": sum(
            case["current_pairwise_contact_state"]["physical_contact_factor_ready_rows"]
            for case in case_outputs
        ),
        "pairwise_metric_depth_evaluated_rows": sum(
            case["current_pairwise_contact_depth_gap"]["evaluated_pair_depth_rows"]
            for case in case_outputs
        ),
        "pairwise_metric_depth_compatible_candidate_rows": sum(
            case["current_pairwise_contact_depth_gap"]["metric_depth_compatible_candidate_rows"]
            for case in case_outputs
        ),
        "hand_metric_depth_variable_count": sum(
            case["current_hand_metric_depth_state"]["hand_metric_depth_variable_count"]
            for case in case_outputs
        ),
        "hand_metric_depth_measured_rows": sum(
            case["current_hand_metric_depth_state"]["measured_hand_depth_rows"]
            for case in case_outputs
        ),
        "hand_metric_depth_projection_residual_ok_rows": sum(
            case["current_hand_metric_depth_state"]["projection_residual_ok_hand_rows"]
            for case in case_outputs
        ),
        "hand_depth_repair_factor_candidate_rows": sum(
            case["current_hand_depth_factor_problem"]["depth_repair_factor_candidate_rows"]
            for case in case_outputs
        ),
        "metric_hand_state_accepted_rows": sum(
            case["current_hand_depth_factor_problem"]["metric_hand_state_accepted_rows"]
            for case in case_outputs
        ),
        "hand_intrinsics_counterfactual_variable_count": sum(
            case["current_hand_intrinsics_depth_counterfactual"]["hand_intrinsics_counterfactual_variable_count"]
            for case in case_outputs
        ),
        "hand_intrinsics_counterfactual_metric_depth_measured_rows": sum(
            case["current_hand_intrinsics_depth_counterfactual"]["counterfactual_metric_depth_measured_rows"]
            for case in case_outputs
        ),
        "hand_intrinsics_counterfactual_projection_factor_ready_rows": sum(
            case["current_hand_intrinsics_depth_counterfactual"]["counterfactual_projection_factor_ready_rows"]
            for case in case_outputs
        ),
        "hand_intrinsics_counterfactual_depth_repair_factor_candidate_rows": sum(
            case["current_hand_intrinsics_depth_counterfactual"]["counterfactual_depth_repair_factor_candidate_rows"]
            for case in case_outputs
        ),
        "hand_intrinsics_counterfactual_median_gap_improved_rows": sum(
            case["current_hand_intrinsics_depth_counterfactual"]["counterfactual_median_gap_improved_rows"]
            for case in case_outputs
        ),
        "hand_intrinsics_counterfactual_metric_hand_state_accepted_rows": sum(
            case["current_hand_intrinsics_depth_counterfactual"]["counterfactual_metric_hand_state_accepted_rows"]
            for case in case_outputs
        ),
        "hand_scale_counterfactual_variable_count": sum(
            case["current_hand_scale_depth_counterfactual"]["hand_scale_counterfactual_variable_count"]
            for case in case_outputs
        ),
        "hand_scale_counterfactual_base_available_rows": sum(
            case["current_hand_scale_depth_counterfactual"]["base_available_rows"]
            for case in case_outputs
        ),
        "hand_scale_counterfactual_scale_candidate_rows": sum(
            case["current_hand_scale_depth_counterfactual"]["scale_candidate_rows"]
            for case in case_outputs
        ),
        "hand_scale_counterfactual_case_global_accepted_rows": sum(
            case["current_hand_scale_depth_counterfactual"]["case_global_scale_mode"][
                "metric_hand_state_accepted_rows"
            ]
            for case in case_outputs
        ),
        "hand_scale_counterfactual_case_global_depth_repair_candidate_rows": sum(
            case["current_hand_scale_depth_counterfactual"]["case_global_scale_mode"][
                "depth_repair_factor_candidate_rows"
            ]
            for case in case_outputs
        ),
        "hand_scale_counterfactual_side_global_accepted_rows": sum(
            case["current_hand_scale_depth_counterfactual"]["side_global_scale_mode"][
                "metric_hand_state_accepted_rows"
            ]
            for case in case_outputs
        ),
        "hand_scale_counterfactual_side_global_depth_repair_candidate_rows": sum(
            case["current_hand_scale_depth_counterfactual"]["side_global_scale_mode"][
                "depth_repair_factor_candidate_rows"
            ]
            for case in case_outputs
        ),
        "hand_scale_counterfactual_per_row_oracle_accepted_rows": sum(
            case["current_hand_scale_depth_counterfactual"]["per_row_scale_oracle_mode"][
                "metric_hand_state_accepted_rows"
            ]
            for case in case_outputs
        ),
        "hand_scale_counterfactual_per_row_oracle_depth_repair_candidate_rows": sum(
            case["current_hand_scale_depth_counterfactual"]["per_row_scale_oracle_mode"][
                "depth_repair_factor_candidate_rows"
            ]
            for case in case_outputs
        ),
        "hand_depth_repair_graph_variable_count": sum(
            case["current_hand_depth_repair_graph"]["hand_depth_repair_graph_variable_count"]
            for case in case_outputs
        ),
        "hand_depth_repair_graph_base_available_rows": sum(
            case["current_hand_depth_repair_graph"]["base_available_rows"] for case in case_outputs
        ),
        "hand_depth_repair_graph_depth_data_candidate_rows": sum(
            case["current_hand_depth_repair_graph"]["depth_data_candidate_rows"] for case in case_outputs
        ),
        "hand_depth_repair_graph_metric_hand_state_accepted_rows": sum(
            case["current_hand_depth_repair_graph"]["metric_hand_state_accepted_rows"]
            for case in case_outputs
        ),
        "hand_depth_repair_graph_depth_repair_factor_candidate_rows": sum(
            case["current_hand_depth_repair_graph"]["depth_repair_factor_candidate_rows"]
            for case in case_outputs
        ),
        "hand_depth_repair_graph_bound_hit_rows": sum(
            case["current_hand_depth_repair_graph"]["hand_ray_shift_bound_hit_rows"]
            for case in case_outputs
        ),
        "hand_depth_repair_graph_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(case["current_hand_depth_repair_graph"]["solver_state_counts"])
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "hand_depth_repair_graph_owner_depth_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(case["current_hand_depth_repair_graph"]["owner_depth_state_counts"])
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "hand_depth_repair_residual_owner_variable_count": sum(
            case["current_hand_depth_repair_residual_owner_state"][
                "hand_depth_repair_residual_owner_variable_count"
            ]
            for case in case_outputs
        ),
        "hand_depth_repair_residual_factor_candidate_rows": sum(
            case["current_hand_depth_repair_residual_owner_state"][
                "repair_residual_factor_candidate_rows"
            ]
            for case in case_outputs
        ),
        "hand_depth_repair_residual_independent_supported_rows": sum(
            case["current_hand_depth_repair_residual_owner_state"][
                "independent_supported_repair_residual_rows"
            ]
            for case in case_outputs
        ),
        "hand_depth_repair_residual_independent_unsupported_rows": sum(
            case["current_hand_depth_repair_residual_owner_state"][
                "independent_unsupported_repair_residual_rows"
            ]
            for case in case_outputs
        ),
        "hand_depth_repair_residual_independent_support_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            case["current_hand_depth_repair_residual_owner_state"][
                                "residual_independent_support_state_counts"
                            ]
                        )
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "hand_depth_repair_residual_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            case["current_hand_depth_repair_residual_owner_state"][
                                "residual_depth_observation_state_counts"
                            ]
                        )
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "supported_hand_depth_repair_residual_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            case["current_hand_depth_repair_residual_owner_state"][
                                "supported_residual_depth_observation_state_counts"
                            ]
                        )
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "hand_depth_repair_residual_owner_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            case["current_hand_depth_repair_residual_owner_state"][
                                "residual_owner_state_counts"
                            ]
                        )
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "hand_depth_repair_residual_sample_count": sum(
            case["current_hand_depth_repair_residual_owner_state"]["residual_sample_count"]
            for case in case_outputs
        ),
        "hand_local_projection_repair_variable_count": sum(
            case["current_hand_local_projection_repair_problem"]["hand_local_projection_repair_variable_count"]
            for case in case_outputs
        ),
        "hand_local_projection_repair_factor_candidate_rows": sum(
            case["current_hand_local_projection_repair_problem"]["local_projection_repair_factor_candidate_rows"]
            for case in case_outputs
        ),
        "hand_local_projection_mixed_owner_rows": sum(
            case["current_hand_local_projection_repair_problem"]["partial_projection_depth_mixed_owner_rows"]
            for case in case_outputs
        ),
        "hand_local_projection_depth_observation_owner_rows": sum(
            case["current_hand_local_projection_repair_problem"]["depth_observation_or_occlusion_owner_rows"]
            for case in case_outputs
        ),
        "hand_local_projection_support_unresolved_rows": sum(
            case["current_hand_local_projection_repair_problem"]["projection_support_unresolved_rows"]
            for case in case_outputs
        ),
        "hand_local_projection_repair_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            case["current_hand_local_projection_repair_problem"][
                                "residual_local_projection_repair_state_counts"
                            ]
                        )
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "hand_local_projection_assignment": {
            "residual_sample_count": sum(
                require_int(
                    case["current_hand_local_projection_repair_problem"]["local_projection_assignment"].get(
                        "residual_sample_count"
                    ),
                    "local projection residual samples",
                )
                for case in case_outputs
            ),
            "assigned_residual_sample_count": sum(
                require_int(
                    case["current_hand_local_projection_repair_problem"]["local_projection_assignment"].get(
                        "assigned_residual_sample_count"
                    ),
                    "local projection assigned samples",
                )
                for case in case_outputs
            ),
            "compatible_seed_sample_count": sum(
                require_int(
                    case["current_hand_local_projection_repair_problem"]["local_projection_assignment"].get(
                        "compatible_seed_sample_count"
                    ),
                    "local projection compatible seed samples",
                )
                for case in case_outputs
            ),
        },
        "mano_parameter_ownership_variable_count": sum(
            case["current_mano_parameter_ownership_state"]["mano_parameter_ownership_variable_count"]
            for case in case_outputs
        ),
        "mano_parameter_owned_residual_rows": sum(
            case["current_mano_parameter_ownership_state"]["residual_mano_parameter_owned_rows"]
            for case in case_outputs
        ),
        "mano_parameter_ownership_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            case["current_mano_parameter_ownership_state"][
                                "residual_mano_parameter_ownership_state_counts"
                            ]
                        )
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "mano_parameter_local_projection_articulation_factor_candidate_rows": sum(
            case["current_mano_parameter_ownership_state"][
                "local_projection_articulation_factor_candidate_rows"
            ]
            for case in case_outputs
        ),
        "mano_parameter_mixed_projection_articulation_observation_candidate_rows": sum(
            case["current_mano_parameter_ownership_state"][
                "mixed_projection_articulation_observation_candidate_rows"
            ]
            for case in case_outputs
        ),
        "mano_parameter_owned_alignment_error_summary": {
            "vertex_median_error_m": {
                "case_summaries": [
                    case["current_mano_parameter_ownership_state"]["owned_alignment_error_summary"][
                        "vertex_median_error_m"
                    ]
                    for case in case_outputs
                ]
            },
            "vertex_p95_error_m": {
                "case_summaries": [
                    case["current_mano_parameter_ownership_state"]["owned_alignment_error_summary"][
                        "vertex_p95_error_m"
                    ]
                    for case in case_outputs
                ]
            },
            "joint_median_error_m": {
                "case_summaries": [
                    case["current_mano_parameter_ownership_state"]["owned_alignment_error_summary"][
                        "joint_median_error_m"
                    ]
                    for case in case_outputs
                ]
            },
            "joint_p95_error_m": {
                "case_summaries": [
                    case["current_mano_parameter_ownership_state"]["owned_alignment_error_summary"][
                        "joint_p95_error_m"
                    ]
                    for case in case_outputs
                ]
            },
            "wilor_similarity_scale": {
                "case_summaries": [
                    case["current_mano_parameter_ownership_state"]["owned_alignment_error_summary"][
                        "wilor_similarity_scale"
                    ]
                    for case in case_outputs
                ]
            },
        },
        "hand_surface_depth_tail_variable_count": sum(
            case["current_hand_surface_depth_tail_state"]["hand_surface_depth_tail_variable_count"]
            for case in case_outputs
        ),
        "hand_surface_depth_scalar_compatible_rows": sum(
            case["current_hand_surface_depth_tail_state"]["scalar_depth_compatible_rows"]
            for case in case_outputs
        ),
        "hand_surface_depth_tail_factor_candidate_rows": sum(
            case["current_hand_surface_depth_tail_state"]["scalar_depth_tail_factor_candidate_rows"]
            for case in case_outputs
        ),
        "hand_surface_depth_projection_untrusted_after_scalar_scale_rows": sum(
            case["current_hand_surface_depth_tail_state"]["projection_untrusted_after_scalar_scale_rows"]
            for case in case_outputs
        ),
        "hand_surface_depth_unobserved_after_scalar_scale_rows": sum(
            case["current_hand_surface_depth_tail_state"]["unobserved_after_scalar_scale_rows"]
            for case in case_outputs
        ),
        "hand_tail_support_variable_count": sum(
            case["current_hand_tail_support_state"]["hand_tail_support_variable_count"]
            for case in case_outputs
        ),
        "hand_tail_support_factor_candidate_rows": sum(
            case["current_hand_tail_support_state"]["tail_factor_candidate_rows"]
            for case in case_outputs
        ),
        "hand_tail_selected_support_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(case["current_hand_tail_support_state"]["tail_selected_support_state_counts"])
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "hand_tail_independent_support_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(case["current_hand_tail_support_state"]["tail_independent_support_state_counts"])
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "hand_tail_abs_sample_count": sum(
            case["current_hand_tail_support_state"]["tail_abs_sample_count"] for case in case_outputs
        ),
        "hand_tail_negative_sample_count": sum(
            case["current_hand_tail_support_state"]["tail_negative_sample_count"] for case in case_outputs
        ),
        "hand_tail_positive_sample_count": sum(
            case["current_hand_tail_support_state"]["tail_positive_sample_count"] for case in case_outputs
        ),
        "hand_tail_depth_observation_variable_count": sum(
            case["current_hand_tail_depth_observation_state"]["hand_tail_depth_observation_variable_count"]
            for case in case_outputs
        ),
        "hand_tail_depth_observation_factor_candidate_rows": sum(
            case["current_hand_tail_depth_observation_state"]["tail_factor_candidate_rows"]
            for case in case_outputs
        ),
        "hand_tail_depth_independent_supported_candidate_rows": sum(
            case["current_hand_tail_depth_observation_state"]["independent_supported_tail_candidate_rows"]
            for case in case_outputs
        ),
        "hand_tail_depth_independent_unsupported_candidate_rows": sum(
            case["current_hand_tail_depth_observation_state"]["independent_unsupported_tail_candidate_rows"]
            for case in case_outputs
        ),
        "hand_tail_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            case["current_hand_tail_depth_observation_state"][
                                "tail_depth_observation_state_counts"
                            ]
                        )
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "supported_hand_tail_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            case["current_hand_tail_depth_observation_state"][
                                "supported_tail_depth_observation_state_counts"
                            ]
                        )
                        for case in case_outputs
                    ),
                    Counter(),
                ).items()
            )
        ),
        "contact_owner_variable_count": sum(
            case["current_contact_ownership_problem"]["contact_owner_variable_count"]
            for case in case_outputs
        ),
        "contact_owner_candidate_rows": sum(
            case["current_contact_ownership_problem"]["contact_owner_candidate_rows"]
            for case in case_outputs
        ),
        "contact_owner_variables_with_selected_measurement": sum(
            case["current_contact_ownership_problem"]["contact_owner_variables_with_selected_measurement"]
            for case in case_outputs
        ),
        "contact_owner_variables_without_selected_measurement": sum(
            case["current_contact_ownership_problem"]["contact_owner_variables_without_selected_measurement"]
            for case in case_outputs
        ),
        "contact_owner_variables_with_supported_candidate": sum(
            case["current_contact_ownership_problem"]["contact_owner_variables_with_supported_candidate"]
            for case in case_outputs
        ),
        "contact_owner_variables_with_geometry_supported_candidate": sum(
            case["current_contact_ownership_problem"]["contact_owner_variables_with_geometry_supported_candidate"]
            for case in case_outputs
        ),
        "contact_owner_image_supported_candidate_rows": sum(
            case["current_contact_ownership_problem"]["contact_owner_image_supported_candidate_rows"]
            for case in case_outputs
        ),
        "contact_owner_metric_depth_supported_candidate_rows": sum(
            case["current_contact_ownership_problem"]["contact_owner_metric_depth_supported_candidate_rows"]
            for case in case_outputs
        ),
        "owner_image_variables_with_single_supported_candidate": sum(
            case["current_contact_ownership_problem"]["owner_image_variables_with_single_supported_candidate"]
            for case in case_outputs
        ),
        "owner_image_variables_with_ambiguous_supported_candidates": sum(
            case["current_contact_ownership_problem"]["owner_image_variables_with_ambiguous_supported_candidates"]
            for case in case_outputs
        ),
        "contact_owner_variables_without_supported_candidate": sum(
            case["current_contact_ownership_problem"]["contact_owner_variables_without_supported_candidate"]
            for case in case_outputs
        ),
        "contact_owner_factor_ready_rows": sum(
            case["current_contact_ownership_problem"]["contact_owner_factor_ready_rows"]
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
        "depth_contact_legacy_contact_ready_hand_rows": sum(
            case["current_depth_contact_consistency_audit"]["legacy_contact_ready_hand_rows"]
            for case in case_outputs
        ),
        "depth_contact_multi_object_reconstructed_object_contact_candidate_rows": sum(
            case["current_depth_contact_consistency_audit"][
                "multi_object_reconstructed_object_contact_candidate_rows"
            ]
            for case in case_outputs
        ),
        "depth_contact_legacy_owner_mismatch_frame_count": sum(
            case["current_depth_contact_consistency_audit"]["legacy_owner_mismatch_frame_count"]
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
        "object_geometry_factor_contact_owner_variable_count": sum(
            case["current_object_geometry_factor_problem"]["contact_owner_variable_count"]
            for case in case_outputs
        ),
        "object_geometry_factor_contact_owner_candidate_rows": sum(
            case["current_object_geometry_factor_problem"]["contact_owner_candidate_rows"]
            for case in case_outputs
        ),
        "object_geometry_factor_contact_owner_image_supported_rows": sum(
            case["current_object_geometry_factor_problem"]["contact_owner_image_supported_candidate_rows"]
            for case in case_outputs
        ),
        "object_geometry_factor_contact_owner_factor_ready_rows": sum(
            case["current_object_geometry_factor_problem"]["contact_owner_factor_ready_rows"]
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
        "--pairwise-contact-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_state"),
    )
    parser.add_argument(
        "--pairwise-contact-depth-gap-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap"),
    )
    parser.add_argument(
        "--hand-metric-depth-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_metric_depth_state"),
    )
    parser.add_argument(
        "--hand-depth-factor-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_factor_problem"),
    )
    parser.add_argument(
        "--hand-intrinsics-depth-counterfactual-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_intrinsics_depth_counterfactual"),
    )
    parser.add_argument(
        "--hand-scale-depth-counterfactual-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_scale_depth_counterfactual"),
    )
    parser.add_argument(
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--hand-depth-repair-residual-owner-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_residual_owner_state"),
    )
    parser.add_argument(
        "--hand-local-projection-repair-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_local_projection_repair_problem"),
    )
    parser.add_argument(
        "--mano-parameter-ownership-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_mano_parameter_ownership_state"),
    )
    parser.add_argument(
        "--hand-surface-depth-tail-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_surface_depth_tail_state"),
    )
    parser.add_argument(
        "--hand-tail-support-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_tail_support_state"),
    )
    parser.add_argument(
        "--hand-tail-depth-observation-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_tail_depth_observation_state"),
    )
    parser.add_argument(
        "--contact-ownership-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_ownership_problem"),
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
