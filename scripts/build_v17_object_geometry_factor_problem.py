#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


STATUS = "v17_object_geometry_factor_problem_qc"
CLAIM = (
    "This artifact converts current V17 per-object evidence into an object-centric geometry factor problem. "
    "It defines the canonical-geometry, per-frame pose, material-correspondence, visible-surface, and contact "
    "factor blocks that a real object solver must own. It is a problem materialization, not an optimizer."
)

FALSE_READY = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}


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


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if abs(out) < float("inf") else None


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def rows_by_object(rows: list[Any], *, key: str, label: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for i, raw in enumerate(rows):
        row = require_dict(raw, f"{label}[{i}]")
        object_id = require_str(row.get(key), f"{label}[{i}].{key}")
        out.setdefault(object_id, []).append(row)
    return out


def summarize(values: list[float]) -> dict[str, Any]:
    vals = sorted(float(v) for v in values if finite_float(v) is not None)
    if not vals:
        return {"count": 0}
    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = q * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac
    return {
        "count": len(vals),
        "median": pct(0.5),
        "p05": pct(0.05),
        "p95": pct(0.95),
        "min": vals[0],
        "max": vals[-1],
    }


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": payload.get("status"),
        "method": payload.get("method"),
    }


def visible_surface_by_object(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("object_summaries"), "visible-surface object_summaries")):
        row = require_dict(raw, f"visible-surface object_summaries[{i}]")
        object_id = require_str(row.get("object_id"), f"visible-surface object_summaries[{i}].object_id")
        out[object_id] = row
    return out


def load_case_inputs(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "object_geometry_hypothesis_state": existing_path(
            args.object_geometry_hypothesis_state_root / case / "v17_object_geometry_hypothesis_state_report.json",
            f"{case} object-geometry hypothesis-state report",
        ),
        "visible_surface": existing_path(
            args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
            f"{case} visible-surface report",
        ),
        "material_track": existing_path(
            args.object_material_track_root / case / "v17_object_material_track_summary.json",
            f"{case} material-track summary",
        ),
        "material_motion": existing_path(
            args.object_material_motion_state_root / case / "v17_object_material_motion_state_report.json",
            f"{case} material-motion report",
        ),
        "material_pose": existing_path(
            args.object_material_pose_candidate_root / case / "v17_object_material_pose_candidate_report.json",
            f"{case} material-pose report",
        ),
        "material_surface_replay": existing_path(
            args.object_material_surface_replay_root / case / "v17_object_material_surface_replay_report.json",
            f"{case} material-surface replay report",
        ),
        "multi_object_contact_evidence": existing_path(
            args.multi_object_contact_evidence_root / case / "v17_multi_object_contact_evidence_report.json",
            f"{case} multi-object contact evidence report",
        ),
        "geometry_source_audit": existing_path(
            args.geometry_source_audit_root / case / "v17_geometry_source_audit_report.json",
            f"{case} geometry-source audit report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    return {"paths": paths, "payloads": payloads}


def material_track_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        frames = [require_int(frame, "material-track frame") for frame in require_list(row.get("frames"), "frames")]
        out.append(
            {
                "window_id": require_str(row.get("window_id"), "window_id"),
                "report_path": require_str(row.get("report_path"), "report_path"),
                "rigid_pair_report_path": row.get("rigid_pair_report_path"),
                "frame_count": require_int(row.get("frame_count"), "frame_count"),
                "first_frame": min(frames),
                "last_frame": max(frames),
                "query_points": require_int(row.get("query_points"), "query_points"),
                "all_frame_accepted_tracks": require_int(
                    row.get("all_frame_accepted_tracks"),
                    "all_frame_accepted_tracks",
                ),
                "rigid_factor_ready_pairs": require_int(
                    row.get("rigid_factor_ready_pairs"),
                    "rigid_factor_ready_pairs",
                ),
                "rigid_motion_evidence_ready": bool(row.get("rigid_motion_evidence_ready") is True),
                "ready_pair_inlier_residual_m": row.get("ready_pair_inlier_residual_m"),
            }
        )
    return out


def motion_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        segments = []
        for raw in require_list(row.get("segments"), "material-motion segments"):
            segment = require_dict(raw, "material-motion segment")
            segments.append(
                {
                    "start_frame": require_int(segment.get("start_frame"), "segment start_frame"),
                    "end_frame": require_int(segment.get("end_frame"), "segment end_frame"),
                    "pair_count": require_int(segment.get("pair_count"), "segment pair_count"),
                    "ready": bool(segment.get("persistent_window_motion_candidate") is True)
                    or bool(segment.get("window_rigid_motion_candidate") is True),
                    "readiness_checks": segment.get("readiness_checks"),
                }
            )
        out.append(
            {
                "window_id": require_str(row.get("window_id"), "window_id"),
                "object_id": require_str(row.get("object_id"), "object_id"),
                "rigid_factor_ready_pairs": require_int(
                    row.get("rigid_factor_ready_pairs"),
                    "rigid_factor_ready_pairs",
                ),
                "local_adjacent_material_motion": bool(row.get("local_adjacent_material_motion") is True),
                "persistent_window_motion_candidate": bool(row.get("persistent_window_motion_candidate") is True),
                "ready_segment_count": require_int(row.get("ready_segment_count"), "ready_segment_count"),
                "candidate_segment_count": require_int(row.get("candidate_segment_count"), "candidate_segment_count"),
                "max_ready_segment_pairs": require_int(row.get("max_ready_segment_pairs"), "max_ready_segment_pairs"),
                "segments": segments,
            }
        )
    return out


def pose_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "candidate_id": require_str(row.get("candidate_id"), "candidate_id"),
                "window_id": require_str(row.get("window_id"), "window_id"),
                "start_frame": require_int(row.get("start_frame"), "start_frame"),
                "end_frame": require_int(row.get("end_frame"), "end_frame"),
                "frame_count": require_int(row.get("frame_count"), "frame_count"),
                "track_count": require_int(row.get("track_count"), "track_count"),
                "pair_count": require_int(row.get("pair_count"), "pair_count"),
                "partial_material_pose_candidate": bool(row.get("partial_material_pose_candidate") is True),
                "residual_m": row.get("residual_m"),
                "readiness_checks": row.get("readiness_checks"),
            }
        )
    return out


def replay_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "candidate_id": require_str(row.get("candidate_id"), "candidate_id"),
                "window_id": require_str(row.get("window_id"), "window_id"),
                "start_frame": require_int(row.get("start_frame"), "start_frame"),
                "end_frame": require_int(row.get("end_frame"), "end_frame"),
                "frame_count": require_int(row.get("frame_count"), "frame_count"),
                "partial_visible_surface_replay_candidate": bool(
                    row.get("partial_visible_surface_replay_candidate") is True
                ),
                "surface_replay_m": row.get("surface_replay_m"),
                "readiness_checks": row.get("readiness_checks"),
            }
        )
    return out


def contact_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row.get("contact_mode_state") == "measured_distance_evidence"]
    min_distances = [
        float(row["min_symmetric_distance_m"])
        for row in measured
        if finite_float(row.get("min_symmetric_distance_m")) is not None
    ]
    return {
        "hand_object_rows": len(rows),
        "measured_distance_rows": len(measured),
        "unobserved_rows": sum(1 for row in rows if row.get("contact_mode_state") == "unobserved"),
        "visible_surface_distance_candidate_rows": sum(
            1 for row in rows if row.get("visible_surface_distance_candidate") is True
        ),
        "contact_distance_candidate_rows": sum(1 for row in rows if row.get("contact_distance_candidate") is True),
        "contact_factor_ready_rows": sum(1 for row in rows if row.get("contact_factor_ready") is True),
        "min_symmetric_distance_m": summarize(min_distances),
        "missing_geometry_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in rows
                    for reason in require_list(row.get("missing_geometry", []), "missing_geometry")
                ).items()
            )
        ),
    }


def factor_blocks(
    obj: dict[str, Any],
    *,
    track_windows: list[dict[str, Any]],
    motion: list[dict[str, Any]],
    poses: list[dict[str, Any]],
    replays: list[dict[str, Any]],
    contacts: dict[str, Any],
    conflicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    visible = require_dict(obj.get("visible_surface_measurement"), "visible_surface_measurement")
    persistent = require_dict(obj.get("persistent_visible_surface_shape"), "persistent_visible_surface_shape")
    depth_candidates = require_dict(obj.get("object_depth_repair_candidates"), "object_depth_repair_candidates")
    return [
        {
            "factor_block": "visible_surface_depth_and_mask",
            "source": "multi_object_visible_surfaces",
            "frame_rows": require_int(visible.get("surface_frame_count"), "surface_frame_count"),
            "rejected_frame_rows": require_int(visible.get("rejected_frame_count"), "rejected_frame_count"),
            "surface_vertices": require_int(visible.get("surface_vertices"), "surface_vertices"),
            "surface_faces": require_int(visible.get("surface_faces"), "surface_faces"),
            "solver_role": "observed surface and silhouette residuals for canonical geometry and per-frame pose",
        },
        {
            "factor_block": "persistent_visible_surface_shape",
            "source": "persistent_object_shape_measurements",
            "accepted_measurement_count": require_int(
                persistent.get("accepted_measurement_count"),
                "accepted_measurement_count",
            ),
            "canonical_mesh_npz": require_list(persistent.get("canonical_mesh_npz"), "canonical_mesh_npz"),
            "pose_models": require_list(persistent.get("pose_models"), "pose_models"),
            "solver_role": "canonical visible-surface geometry seed; hidden topology remains unresolved",
        },
        {
            "factor_block": "material_correspondence_rigidity",
            "source": "object_material_tracks",
            "window_count": len(track_windows),
            "rigid_factor_ready_pair_count": sum(
                require_int(row.get("rigid_factor_ready_pairs"), "rigid_factor_ready_pairs")
                for row in track_windows
            ),
            "windows": track_windows,
            "solver_role": "pairwise material-point SE(3) and rigidity residuals",
        },
        {
            "factor_block": "material_motion_segments",
            "source": "object_material_motion_state",
            "window_count": len(motion),
            "persistent_window_motion_candidate_count": sum(
                1 for row in motion if row.get("persistent_window_motion_candidate") is True
            ),
            "local_adjacent_material_motion_window_count": sum(
                1 for row in motion if row.get("local_adjacent_material_motion") is True
            ),
            "windows": motion,
            "solver_role": "temporal grouping for object pose variables",
        },
        {
            "factor_block": "partial_material_pose_segments",
            "source": "object_material_pose_candidates",
            "candidate_segment_count": len(poses),
            "ready_segment_count": sum(1 for row in poses if row.get("partial_material_pose_candidate") is True),
            "candidates": poses,
            "solver_role": "observed material-point pose priors for short segments",
        },
        {
            "factor_block": "visible_surface_replay_segments",
            "source": "object_material_surface_replay",
            "candidate_segment_count": len(replays),
            "ready_segment_count": sum(
                1 for row in replays if row.get("partial_visible_surface_replay_candidate") is True
            ),
            "candidates": replays,
            "solver_role": "surface replay constraints for short material-pose segments",
        },
        {
            "factor_block": "multi_object_hand_contact_distance",
            "source": "multi_object_contact_evidence",
            **contacts,
            "solver_role": "hand-object distance and contact-mode residuals against unified object geometry",
        },
        {
            "factor_block": "object_depth_repair_temporal_validation",
            "source": "object_depth_repair_candidate_measurements",
            "candidate_count": require_int(depth_candidates.get("candidate_count"), "candidate_count"),
            "temporal_validation_status_counts": require_dict(
                depth_candidates.get("temporal_validation_status_counts"),
                "temporal_validation_status_counts",
            ),
            "solver_role": "candidate depth repairs that need temporal support before entering geometry factors",
        },
        {
            "factor_block": "geometry_source_compatibility",
            "source": "geometry_source_audit",
            "source_conflict_count": len(conflicts),
            "conflicts": conflicts,
            "solver_role": "prevents local patches or legacy single-stream geometry from silently owning object pose",
        },
    ]


def variable_blocks(obj: dict[str, Any], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    object_id = require_str(obj.get("object_id"), "object_id")
    active_frames = require_int(obj.get("active_frame_count"), f"{object_id} active_frame_count")
    persistent = require_dict(obj.get("persistent_visible_surface_shape"), "persistent_visible_surface_shape")
    visible = require_dict(obj.get("visible_surface_measurement"), "visible_surface_measurement")
    canonical_meshes = require_list(persistent.get("canonical_mesh_npz"), "canonical_mesh_npz")
    material_pose_ready = sum(
        require_int(block.get("ready_segment_count"), "ready_segment_count")
        for block in blocks
        if block.get("factor_block") == "partial_material_pose_segments"
    )
    geometry_seed = "persistent_visible_surface_mesh" if canonical_meshes else "visible_surface_samples_only"
    orientation_observable = not any("orientation_unobservable" in str(model) for model in persistent.get("pose_models", []))
    return [
        {
            "variable_block": "canonical_object_geometry",
            "variable_type": "mesh_or_sdf_field",
            "object_id": object_id,
            "current_seed": geometry_seed if require_int(visible.get("surface_frame_count"), "surface_frame_count") else "none",
            "canonical_mesh_npz": canonical_meshes,
            "instantiated_by_current_v17_solver": False,
            "complete_hidden_topology_ready": False,
        },
        {
            "variable_block": "object_pose_per_active_frame",
            "variable_type": "SE3_per_frame",
            "object_id": object_id,
            "active_frame_count": active_frames,
            "parameter_count": active_frames * 6,
            "orientation_observable_from_current_seed": bool(orientation_observable and material_pose_ready > 0),
            "instantiated_by_current_v17_solver": False,
        },
        {
            "variable_block": "object_deformation_or_topology_delta",
            "variable_type": "geometry_delta_field",
            "object_id": object_id,
            "reason_required": "current evidence can contain partial visible surfaces, hand occlusion, and local patches without complete hidden topology",
            "instantiated_by_current_v17_solver": False,
        },
        {
            "variable_block": "contact_attachment_per_hand_object_frame",
            "variable_type": "contact_mode_and_patch_identity",
            "object_id": object_id,
            "instantiated_by_current_v17_solver": False,
            "requires_unified_geometry_source": True,
        },
    ]


def readiness_checks(obj: dict[str, Any], contacts: dict[str, Any], conflicts: list[dict[str, Any]]) -> dict[str, bool]:
    visible = require_dict(obj.get("visible_surface_measurement"), "visible_surface_measurement")
    persistent = require_dict(obj.get("persistent_visible_surface_shape"), "persistent_visible_surface_shape")
    material_motion = require_dict(obj.get("material_motion_state"), "material_motion_state")
    material_pose = require_dict(obj.get("material_pose_candidates"), "material_pose_candidates")
    material_replay = require_dict(obj.get("material_surface_replay"), "material_surface_replay")
    active = require_int(obj.get("active_frame_count"), "active_frame_count")
    visible_masks = require_int(obj.get("visible_mask_frame_count"), "visible_mask_frame_count")
    visible_surfaces = require_int(visible.get("surface_frame_count"), "surface_frame_count")
    pose_models = [str(model) for model in require_list(persistent.get("pose_models"), "pose_models")]
    return {
        "mask_evidence_available": visible_masks > 0,
        "visible_surface_evidence_available": visible_surfaces > 0,
        "full_active_visible_surface_coverage": active > 0 and visible_surfaces == active,
        "canonical_geometry_seed_available": require_int(
            persistent.get("accepted_measurement_count"),
            "accepted_measurement_count",
        )
        > 0,
        "hidden_topology_reconstructed": False,
        "orientation_observable": not any("orientation_unobservable" in model for model in pose_models),
        "persistent_material_motion_available": require_int(
            material_motion.get("persistent_window_motion_candidate_count"),
            "persistent_window_motion_candidate_count",
        )
        > 0,
        "partial_material_pose_segments_available": require_int(
            material_pose.get("ready_segment_count"),
            "ready_segment_count",
        )
        > 0,
        "partial_visible_surface_replay_available": require_int(
            material_replay.get("ready_segment_count"),
            "ready_segment_count",
        )
        > 0,
        "contact_distance_candidates_available": require_int(
            contacts.get("contact_distance_candidate_rows"),
            "contact_distance_candidate_rows",
        )
        > 0,
        "contact_factor_ready_against_multi_object_geometry": require_int(
            contacts.get("contact_factor_ready_rows"),
            "contact_factor_ready_rows",
        )
        > 0,
        "source_compatible_with_visible_surface_geometry": len(conflicts) == 0,
    }


def blocked_reasons(checks: dict[str, bool]) -> list[str]:
    messages = {
        "visible_surface_evidence_available": "no visible RGBD surface measurements",
        "full_active_visible_surface_coverage": "visible surfaces do not cover every active object frame",
        "canonical_geometry_seed_available": "no canonical object geometry seed",
        "hidden_topology_reconstructed": "hidden topology is not reconstructed",
        "orientation_observable": "orientation is not observable from current geometry seed",
        "contact_factor_ready_against_multi_object_geometry": "no contact factors are ready against multi-object geometry",
        "source_compatible_with_visible_surface_geometry": "local patch or legacy geometry conflicts with visible-surface geometry",
    }
    return [message for key, message in messages.items() if checks.get(key) is not True]


def build_object_row(
    obj: dict[str, Any],
    *,
    track_rows: list[dict[str, Any]],
    motion_rows: list[dict[str, Any]],
    pose_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    conflict_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tracks = material_track_windows(track_rows)
    motion = motion_windows(motion_rows)
    poses = pose_candidates(pose_rows)
    replays = replay_candidates(replay_rows)
    contacts = contact_summary(contact_rows)
    blocks = factor_blocks(
        obj,
        track_windows=tracks,
        motion=motion,
        poses=poses,
        replays=replays,
        contacts=contacts,
        conflicts=conflict_rows,
    )
    checks = readiness_checks(obj, contacts, conflict_rows)
    solve_activation_ready = all(
        checks[key]
        for key in (
            "visible_surface_evidence_available",
            "full_active_visible_surface_coverage",
            "canonical_geometry_seed_available",
            "hidden_topology_reconstructed",
            "orientation_observable",
            "contact_factor_ready_against_multi_object_geometry",
            "source_compatible_with_visible_surface_geometry",
        )
    )
    return {
        "object_id": require_str(obj.get("object_id"), "object_id"),
        "track_id": require_str(obj.get("track_id"), "track_id"),
        "name": require_str(obj.get("name"), "name"),
        "geometry_hypothesis_state": require_str(obj.get("geometry_hypothesis_state"), "geometry_hypothesis_state"),
        "active_frame_count": require_int(obj.get("active_frame_count"), "active_frame_count"),
        "visible_mask_frame_count": require_int(obj.get("visible_mask_frame_count"), "visible_mask_frame_count"),
        "variable_blocks": variable_blocks(obj, blocks),
        "factor_blocks": blocks,
        "readiness_checks": checks,
        "blocked_reasons": blocked_reasons(checks),
        "object_geometry_factor_problem_materialized": True,
        "solve_activation_ready": bool(solve_activation_ready),
        "can_own_contact_factors": False,
        "can_own_object_pose_factors": False,
        "complete_mesh_timeline_ready": False,
        **FALSE_READY,
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_case_inputs(case, args)
    paths: dict[str, Path] = loaded["paths"]
    payloads: dict[str, dict[str, Any]] = loaded["payloads"]

    hypothesis = payloads["object_geometry_hypothesis_state"]
    visible_surface = payloads["visible_surface"]
    material_track = payloads["material_track"]
    material_motion = payloads["material_motion"]
    material_pose = payloads["material_pose"]
    material_replay = payloads["material_surface_replay"]
    contact = payloads["multi_object_contact_evidence"]
    audit = payloads["geometry_source_audit"]

    objects = [require_dict(row, f"{case} objects[{i}]") for i, row in enumerate(require_list(hypothesis.get("objects"), "objects"))]
    track_by_object = rows_by_object(require_list(material_track.get("windows"), "material-track windows"), key="object_id", label="material-track windows")
    motion_by_object = rows_by_object(require_list(material_motion.get("windows"), "material-motion windows"), key="object_id", label="material-motion windows")
    pose_by_object = rows_by_object(require_list(material_pose.get("candidates"), "material-pose candidates"), key="object_id", label="material-pose candidates")
    replay_by_object = rows_by_object(require_list(material_replay.get("candidates"), "material-surface replay candidates"), key="object_id", label="material-surface replay candidates")
    contact_by_object = rows_by_object(require_list(contact.get("rows"), "multi-object contact rows"), key="object_id", label="multi-object contact rows")
    conflict_by_object = rows_by_object(require_list(audit.get("local_patch_visible_surface_conflicts"), "geometry-source conflicts"), key="object_id", label="geometry-source conflicts")

    visible_by_object = visible_surface_by_object(visible_surface)
    object_rows = [
        build_object_row(
            obj,
            track_rows=track_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            motion_rows=motion_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            pose_rows=pose_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            replay_rows=replay_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            contact_rows=contact_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            conflict_rows=conflict_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
        )
        for obj in objects
    ]
    missing_visible_objects = sorted(
        set(require_str(obj.get("object_id"), "object_id") for obj in objects).difference(visible_by_object)
    )
    if missing_visible_objects:
        for object_id in missing_visible_objects:
            visible_by_object[object_id] = {}

    summary_counts = {
        "object_count": len(object_rows),
        "solve_activation_ready_object_count": sum(1 for row in object_rows if row["solve_activation_ready"] is True),
        "visible_surface_factor_rows": sum(
            require_int(row["factor_blocks"][0].get("frame_rows"), "visible frame_rows") for row in object_rows
        ),
        "material_rigidity_pair_factor_count": sum(
            require_int(block.get("rigid_factor_ready_pair_count"), "rigid factor count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "material_correspondence_rigidity"
        ),
        "partial_material_pose_ready_segment_count": sum(
            require_int(block.get("ready_segment_count"), "ready segment count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "partial_material_pose_segments"
        ),
        "partial_visible_surface_replay_ready_segment_count": sum(
            require_int(block.get("ready_segment_count"), "ready segment count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "visible_surface_replay_segments"
        ),
        "multi_object_contact_factor_ready_rows": sum(
            require_int(block.get("contact_factor_ready_rows"), "contact factor rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "multi_object_hand_contact_distance"
        ),
        "geometry_source_conflict_count": sum(
            require_int(block.get("source_conflict_count"), "source conflict count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "geometry_source_compatibility"
        ),
    }
    if summary_counts["visible_surface_factor_rows"] != require_int(
        visible_surface.get("surface_frame_rows"),
        "visible_surface surface_frame_rows",
    ):
        raise RuntimeError(f"{case} object factor visible-surface rows disagree with visible-surface report")
    if summary_counts["multi_object_contact_factor_ready_rows"] != require_int(
        contact.get("contact_factor_ready_rows"),
        "contact factor ready rows",
    ):
        raise RuntimeError(f"{case} contact factor-ready rows disagree with multi-object contact evidence")
    if summary_counts["geometry_source_conflict_count"] != len(
        require_list(audit.get("local_patch_visible_surface_conflicts"), "local_patch_visible_surface_conflicts")
    ):
        raise RuntimeError(f"{case} source conflict rows disagree with geometry-source audit")

    report = {
        "method": "build_v17_object_geometry_factor_problem",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": require_int(hypothesis.get("frame_count"), "hypothesis frame_count"),
        "factor_problem_object_rows": len(object_rows),
        "state_counts": require_dict(hypothesis.get("state_counts"), "state_counts"),
        "missing_visible_surface_object_ids": missing_visible_objects,
        "object_rows": object_rows,
        **summary_counts,
        "complete_object_geometry_hypothesis_count": 0,
        "contact_compatible_object_geometry_hypothesis_count": 0,
        "object_pose_factor_ready_hypothesis_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_object_geometry_factor_problem.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.object_geometry_hypothesis_state_root / "v17_object_geometry_hypothesis_state_summary.json",
        "object-geometry hypothesis-state summary",
    )
    summary = require_dict(load_json(summary_path), "object-geometry hypothesis-state summary")
    reports = [
        build_case(
            require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    state_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(require_dict(report.get("state_counts"), "state_counts"))
    payload = {
        "method": "build_v17_object_geometry_factor_problem",
        "status": STATUS,
        "claim": CLAIM,
        "source_object_geometry_hypothesis_state_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "problem_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_object_geometry_factor_problem.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "factor_problem_object_rows": require_int(
                    report.get("factor_problem_object_rows"),
                    "factor_problem_object_rows",
                ),
                "state_counts": require_dict(report.get("state_counts"), "state_counts"),
                "solve_activation_ready_object_count": require_int(
                    report.get("solve_activation_ready_object_count"),
                    "solve_activation_ready_object_count",
                ),
                "visible_surface_factor_rows": require_int(
                    report.get("visible_surface_factor_rows"),
                    "visible_surface_factor_rows",
                ),
                "material_rigidity_pair_factor_count": require_int(
                    report.get("material_rigidity_pair_factor_count"),
                    "material_rigidity_pair_factor_count",
                ),
                "partial_material_pose_ready_segment_count": require_int(
                    report.get("partial_material_pose_ready_segment_count"),
                    "partial_material_pose_ready_segment_count",
                ),
                "partial_visible_surface_replay_ready_segment_count": require_int(
                    report.get("partial_visible_surface_replay_ready_segment_count"),
                    "partial_visible_surface_replay_ready_segment_count",
                ),
                "multi_object_contact_factor_ready_rows": require_int(
                    report.get("multi_object_contact_factor_ready_rows"),
                    "multi_object_contact_factor_ready_rows",
                ),
                "geometry_source_conflict_count": require_int(
                    report.get("geometry_source_conflict_count"),
                    "geometry_source_conflict_count",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "object_count": sum(require_int(report.get("factor_problem_object_rows"), "object rows") for report in reports),
        "state_counts": dict(sorted(state_counts.items())),
        "solve_activation_ready_object_count": sum(
            require_int(report.get("solve_activation_ready_object_count"), "solve-ready")
            for report in reports
        ),
        "visible_surface_factor_rows": sum(
            require_int(report.get("visible_surface_factor_rows"), "visible rows") for report in reports
        ),
        "material_rigidity_pair_factor_count": sum(
            require_int(report.get("material_rigidity_pair_factor_count"), "rigid pair count")
            for report in reports
        ),
        "partial_material_pose_ready_segment_count": sum(
            require_int(report.get("partial_material_pose_ready_segment_count"), "material pose ready count")
            for report in reports
        ),
        "partial_visible_surface_replay_ready_segment_count": sum(
            require_int(report.get("partial_visible_surface_replay_ready_segment_count"), "surface replay ready count")
            for report in reports
        ),
        "multi_object_contact_factor_ready_rows": sum(
            require_int(report.get("multi_object_contact_factor_ready_rows"), "contact factor rows")
            for report in reports
        ),
        "geometry_source_conflict_count": sum(
            require_int(report.get("geometry_source_conflict_count"), "source conflict count")
            for report in reports
        ),
        "complete_object_geometry_hypothesis_count": 0,
        "contact_compatible_object_geometry_hypothesis_count": 0,
        "object_pose_factor_ready_hypothesis_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_object_geometry_factor_problem_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-geometry-hypothesis-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
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
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_factor_problem"),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
