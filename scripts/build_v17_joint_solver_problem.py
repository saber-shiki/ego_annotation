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
                "persistent_object_shape_measurements": counts.get("persistent_object_shape", 0),
                "local_contact_patch_measurements": counts.get("local_contact_patch", 0),
                "object_geometry_complete": mesh["object_geometry_complete"],
            },
            [
                "complete object meshes are absent for several active objects",
                "local contact patches and visible surfaces are still QC evidence, not complete object geometry",
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
            },
            [
                "simultaneous object poses are missing",
                "deformable-bag state is represented by local patches and legacy centers, not deformation variables",
                "object pose corrections are small per-frame updates around fixed input geometry",
            ],
        ),
        variable_family(
            "contact_mode_per_hand_object_frame",
            "contact, no-contact, or unobserved state for each hand-object pair across the full timeline",
            "fixed_input_to_sparse_geometry_graph",
            {
                "current_hand_side_rows": contact["row_count"],
                "current_contact_mode_rows": contact["contact_mode_count"],
                "current_factor_ready_rows": contact["contact_factor_ready_count"],
                "minimum_required_hand_object_rows_from_roster": required_contact_rows,
            },
            [
                "contact modes are estimated before the sparse geometry graph and then fixed",
                "contact state is hand-side to legacy-object, not hand-object for every roster object",
                "unobserved rows do not carry uncertainty variables",
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
            },
            [
                "visible surfaces are now materialized as fixed measurements where mask and metric depth overlap",
                "depth/object/camera contradictions are not jointly optimized",
                "occlusion state is not a latent variable with uncertainty",
            ],
        ),
        variable_family(
            "physical_consistency_terms",
            "nonpenetration, support, contact persistence, sliding, object rigidity/deformation, and manipulation dynamics",
            "partial_residuals_only",
            {
                "local_contact_factors": sparse["contact_factor_count"],
                "sparse_graph_solver_completeness": sparse["solver_completeness"],
            },
            [
                "local equality and smoothness do not model nonpenetration or force/support feasibility",
                "contact dynamics are not coupled to object identity, deformation, and MANO articulation",
                "broad hand-object distances remain diagnostics rather than physical constraints",
            ],
        ),
    ]


def case_problem(inputs: CaseInputs) -> dict[str, Any]:
    manifest = require_dict(load_json(inputs.manifest), f"{inputs.case} manifest")
    roster_payload = require_list(load_json(inputs.object_roster), f"{inputs.case} object roster")
    multi_object_timeline = require_dict(load_json(inputs.multi_object_timeline), f"{inputs.case} multi-object timeline")
    visible_surface_report = require_dict(load_json(inputs.visible_surface_report), f"{inputs.case} visible-surface report")
    sparse_report = require_dict(load_json(inputs.sparse_report), f"{inputs.case} sparse report")
    contact_report = require_dict(load_json(inputs.contact_mode_report), f"{inputs.case} contact-mode report")
    mesh_metadata = require_dict(load_json(inputs.mesh_metadata), f"{inputs.case} mesh metadata")

    counts = measurement_counts(manifest)
    sparse = graph_counts(sparse_report)
    contact = contact_mode_counts(contact_report)
    timeline = multi_object_timeline_counts(multi_object_timeline)
    visible_surface = visible_surface_counts(visible_surface_report)
    mesh = mesh_counts(mesh_metadata)
    roster = roster_audit(roster_payload)

    frame_count = require_int(sparse["frame_count"], f"{inputs.case} sparse frame_count")
    if frame_count != require_int(contact["frame_count"], f"{inputs.case} contact frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse and contact-mode reports")
    if frame_count != require_int(timeline["frame_count"], f"{inputs.case} multi-object timeline frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and multi-object timeline")
    if frame_count != require_int(visible_surface["frame_count"], f"{inputs.case} visible-surface frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and visible-surface report")
    if frame_count != require_int(mesh["frame_count"], f"{inputs.case} mesh frame_count"):
        raise RuntimeError(f"{inputs.case} frame_count mismatch between sparse report and mesh metadata")
    if require_int(timeline["visible_mask_frame_rows"], f"{inputs.case} timeline visible mask rows") != require_int(
        visible_surface["visible_object_frame_rows"], f"{inputs.case} visible-surface visible rows"
    ):
        raise RuntimeError(f"{inputs.case} visible mask rows disagree between timeline and visible-surface report")

    raw_video = require_dict(load_json(Path(require_str(manifest.get("manifest"), "v16 manifest path"))).get("raw_video"), "raw_video")
    raw_frame_count = require_int(raw_video.get("frame_count"), f"{inputs.case} raw_video.frame_count")
    if raw_frame_count != frame_count:
        raise RuntimeError(f"{inputs.case} raw frame count {raw_frame_count} differs from graph frame count {frame_count}")
    finite_number(raw_video.get("fps"), f"{inputs.case} raw fps")

    families = required_variable_families(roster, timeline, visible_surface, counts, sparse, contact, mesh)
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
            "sparse_graph_report": source_summary(inputs.sparse_report, sparse_report),
            "contact_mode_report": source_summary(inputs.contact_mode_report, contact_report),
            "mesh_metadata": source_summary(inputs.mesh_metadata, mesh_metadata),
        },
        "current_sparse_graph": sparse,
        "current_contact_mode_graph": contact,
        "current_multi_object_timeline": timeline,
        "current_multi_object_visible_surfaces": visible_surface,
        "current_mesh_archive": mesh,
        "current_measurement_counts": counts,
        "object_roster_audit": roster,
        "required_variable_families": families,
        "unmet_required_variable_families": unmet,
        "missing_or_incomplete_required_variable_families": unmet,
        "v3_solver_complete": False,
        "annotation_ready": False,
        "deliverable_ready": False,
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
                "current_single_stream_object_variable_frames": case["current_sparse_graph"]["object_variable_frames"],
                "contact_factor_ready_count": case["current_contact_mode_graph"]["contact_factor_ready_count"],
                "unmet_required_variable_families": case["unmet_required_variable_families"],
                "missing_or_incomplete_required_variable_families": case[
                    "missing_or_incomplete_required_variable_families"
                ],
                "v3_solver_complete": False,
                "annotation_ready": False,
                "deliverable_ready": False,
            }
            for case in case_outputs
        ],
        "unmet_required_variable_families_union": union_unmet,
        "missing_or_incomplete_required_variable_families_union": union_unmet,
        "v3_solver_complete": False,
        "annotation_ready": False,
        "deliverable_ready": False,
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
