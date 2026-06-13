#!/usr/bin/env python3
"""Build a full-timeline V18 corrective annotation-state delta.

This converts the corrective V18 mechanisms from visual-only attempts into a
machine-readable annotation artifact. It does not duplicate the full V18 source
annotation and does not claim full V18 closure. It records only states actually
changed or attempted by the corrective work: factor-graph-driven hand state,
graph-shifted MANO projections, generic rigid SE(3) object priors, and HaWoR
prior/provisioning status.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def rounded(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, ndigits) if math.isfinite(value) else None
    if isinstance(value, np.floating):
        x = float(value)
        return round(x, ndigits) if math.isfinite(x) else None
    if isinstance(value, list):
        return [rounded(v, ndigits) for v in value]
    if isinstance(value, tuple):
        return [rounded(v, ndigits) for v in value]
    if isinstance(value, np.ndarray):
        return rounded(value.tolist(), ndigits)
    return value


def bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def bbox_center(value: Any) -> tuple[float, float] | None:
    box = bbox_tuple(value)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def shift_box_to_center(box: tuple[float, float, float, float], center: tuple[float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    cx, cy = center
    return cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h


def project_mano_joints_source_px(mano: dict[str, Any], source_w: float, source_h: float) -> list[tuple[float, float]]:
    joints = mano.get("joints3d_camera")
    cam_t = mano.get("cam_t")
    intr = mano.get("source_intrinsics") or [2304.0, 2304.0, source_w / 2.0, source_h / 2.0]
    if not (isinstance(joints, list) and isinstance(cam_t, list) and len(cam_t) == 3 and isinstance(intr, list) and len(intr) == 4):
        return []
    fx, fy, cx, cy = [finite_float(v) for v in intr]
    pts: list[tuple[float, float]] = []
    for raw in joints:
        if not (isinstance(raw, list) and len(raw) == 3):
            return []
        x = finite_float(raw[0]) + finite_float(cam_t[0])
        y = finite_float(raw[1]) + finite_float(cam_t[1])
        z = finite_float(raw[2]) + finite_float(cam_t[2])
        if z <= 1e-6:
            return []
        u = fx * x / z + cx
        v = fy * y / z + cy
        if not (math.isfinite(u) and math.isfinite(v)):
            return []
        pts.append((u, v))
    return pts


def graph_hand_estimates(frame: dict[str, Any], source_w: float, source_h: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = frame.get("factor_graph_solution", {}).get("variables", {}).get("hand_state", [])
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("variable_id", ""))
        if not vid.startswith("hand::"):
            continue
        est = row.get("estimate")
        if isinstance(est, list) and len(est) >= 2:
            side = vid.split("::", 1)[1]
            out[side] = {
                "variable_id": vid,
                "estimate_normalized_xy": [finite_float(est[0]), finite_float(est[1])],
                "center_source_px": [finite_float(est[0]) * source_w, finite_float(est[1]) * source_h],
                "source": row.get("source"),
                "observation_residual_norm": row.get("observation_residual_norm"),
            }
    return out


def graph_object_poses(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = frame.get("factor_graph_solution", {}).get("variables", {}).get("object_se3", [])
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("variable_id", ""))
        if not vid.startswith("object_se3::"):
            continue
        est = row.get("estimate")
        if isinstance(est, list) and len(est) >= 6:
            oid = vid.split("::", 1)[1]
            out[oid] = {
                "variable_id": vid,
                "pose6_world_from_object": [finite_float(v) for v in est[:6]],
                "source": row.get("source"),
                "observation_residual_norm": row.get("observation_residual_norm"),
            }
    return out


def load_hawor_measurement_index(path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], list[Path]]:
    if not path.exists():
        return {}, []
    rows = load_json(path)
    index: dict[tuple[int, str], dict[str, Any]] = {}
    sources: set[Path] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        entity = str(row.get("entity_id", ""))
        if not entity.startswith("hand:"):
            continue
        frame_idx = int(row.get("frame_idx", -1))
        side = entity.split(":", 1)[1]
        index[(frame_idx, side)] = row
        src = row.get("source_annotation")
        if isinstance(src, str) and Path(src).exists():
            sources.add(Path(src))
    return index, sorted(sources)


def load_hawor_source_hands(paths: list[Path]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        frames = payload.get("frames", [])
        for raw_frame in frames if isinstance(frames, list) else []:
            if not isinstance(raw_frame, dict):
                continue
            frame_idx = int(raw_frame.get("frame_idx", -1))
            for hand in raw_frame.get("hands", []):
                if not isinstance(hand, dict) or hand.get("backend") != "HaWoR":
                    continue
                side = str(hand.get("hand_side") or hand.get("side"))
                if side in {"left", "right"}:
                    out[(frame_idx, side)] = hand
    return out


def rigid_candidates_from_report(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    report = load_json(path)
    candidates = report.get("candidate_objects", {})
    return {str(k): v for k, v in candidates.items() if isinstance(v, dict)} if isinstance(candidates, dict) else {}


def visible_surface_row_index(report_path: Path, candidate_ids: set[str]) -> tuple[dict[tuple[int, str], dict[str, Any]], str | None]:
    if not report_path.exists():
        return {}, None
    report = load_json(report_path)
    rows = report.get("surface_archive_rows", [])
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        oid = str(row.get("object_id"))
        if oid in candidate_ids:
            out[(int(row.get("frame_idx", -1)), oid)] = row
    archive_npz = report.get("archive_npz")
    return out, str(archive_npz) if isinstance(archive_npz, str) else None


def selected_occlusion_owner_index(report_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for hand_graph in report.get("hand_graphs", []) if isinstance(report.get("hand_graphs"), list) else []:
        if not isinstance(hand_graph, dict):
            continue
        for row in hand_graph.get("assignments", []):
            if not isinstance(row, dict):
                continue
            owner = row.get("chosen_owner_object_id")
            if isinstance(owner, str) and owner.startswith("object:"):
                out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")))] = row
    return out, report


def selected_contact_index(report_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for hand_graph in report.get("hand_graphs", []) if isinstance(report.get("hand_graphs"), list) else []:
        if not isinstance(hand_graph, dict):
            continue
        for row in hand_graph.get("assignments", []):
            if not isinstance(row, dict):
                continue
            owner = row.get("chosen_owner_object_id")
            if isinstance(owner, str) and owner.startswith("object:"):
                out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")))] = row
    return out, report


def nonpenetration_row_index(report_path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    if not report_path.exists():
        return {}
    report = load_json(report_path)
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in report.get("rows", []) if isinstance(report.get("rows"), list) else []:
        if not isinstance(row, dict):
            continue
        out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")), str(row.get("object_id")))] = row
    return out


def rigid_residual_row_index(report_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get("residual_rows", []) if isinstance(report.get("residual_rows"), list) else []:
        if isinstance(row, dict):
            out[(int(row.get("frame_idx", -1)), str(row.get("object_id")))] = row
    return out, report


def stable_rigid_pose_index(frames: list[Any], candidate_ids: set[str], radius: int) -> dict[tuple[int, str], list[float]]:
    raw: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        poses = graph_object_poses(frame)
        for oid in candidate_ids:
            pose = poses.get(oid)
            if pose is not None:
                raw[oid].append((frame_idx, np.array(pose["pose6_world_from_object"], dtype=np.float64)))
    stable: dict[tuple[int, str], list[float]] = {}
    for oid, rows in raw.items():
        ordered = sorted(rows, key=lambda x: x[0])
        if not ordered:
            continue
        translations = [pose[:3] for _frame_idx, pose in ordered]
        median_rot = np.median(np.stack([pose[3:6] for _frame_idx, pose in ordered], axis=0), axis=0)
        for i, (frame_idx, pose6) in enumerate(ordered):
            lo = max(0, i - radius)
            hi = min(len(ordered), i + radius + 1)
            stable_pose = pose6.copy()
            stable_pose[:3] = np.mean(np.stack(translations[lo:hi], axis=0), axis=0)
            stable_pose[3:6] = median_rot
            stable[(frame_idx, oid)] = stable_pose.tolist()
    return stable


def hand_corrective_state(
    frame_idx: int,
    hand: dict[str, Any],
    graph_est: dict[str, Any] | None,
    hawor_row: dict[str, Any] | None,
    hawor_hand: dict[str, Any] | None,
    hawor_available_for_case: bool,
    occlusion_owner_row: dict[str, Any] | None,
    contact_row: dict[str, Any] | None,
    signed_nonpenetration_row: dict[str, Any] | None,
    triangle_nonpenetration_row: dict[str, Any] | None,
    source_w: float,
    source_h: float,
) -> dict[str, Any]:
    side = str(hand.get("hand_side") or hand.get("side"))
    bbox = bbox_tuple(hand.get("bbox_xyxy"))
    obs_center = bbox_center(hand.get("bbox_xyxy"))
    out: dict[str, Any] = {
        "hand_side": side,
        "visibility_state": hand.get("visibility_state"),
        "source_bbox_xyxy": rounded(list(bbox) if bbox else None, 3),
        "source_confidence": hand.get("confidence"),
        "best_current_state": "source_visible_hand_no_graph_estimate",
        "uncertainty": [],
    }
    if graph_est is not None:
        center = tuple(float(v) for v in graph_est["center_source_px"])
        graph: dict[str, Any] = {
            "variable_id": graph_est.get("variable_id"),
            "center_source_px": rounded(center, 3),
            "estimate_normalized_xy": rounded(graph_est.get("estimate_normalized_xy"), 6),
            "source": graph_est.get("source"),
            "observation_residual_norm": graph_est.get("observation_residual_norm"),
            "state_role": "render_driver_best_current_image_space_hand_state",
        }
        if bbox is not None:
            shifted_bbox = shift_box_to_center(bbox, center)
            graph["shifted_bbox_xyxy"] = rounded(shifted_bbox, 3)
        if obs_center is not None:
            dx = center[0] - obs_center[0]
            dy = center[1] - obs_center[1]
            graph["center_shift_from_observation_px"] = rounded([dx, dy], 3)
            graph["center_residual_px"] = round(math.hypot(dx, dy), 3)
            mano = hand.get("mano_candidate", {}) if isinstance(hand.get("mano_candidate"), dict) else {}
            pts = project_mano_joints_source_px(mano, source_w, source_h)
            if len(pts) >= 21:
                graph["shifted_mano_joints2d_source_px"] = rounded([(x + dx, y + dy) for x, y in pts[:21]], 2)
                graph["mano_state_scope"] = "source_mano_projection_shifted_by_graph_center_not_solved_articulation"
            else:
                out["uncertainty"].append("no_projectable_mano_candidate_for_graph_shifted_skeleton")
        out["graph_hand_state"] = graph
        out["best_current_state"] = "graph_shifted_mano_if_available_else_graph_shifted_bbox"
    else:
        out["uncertainty"].append("missing_factor_graph_hand_state")

    if hawor_row is not None:
        prior: dict[str, Any] = {
            "status": "available_uncertain_prior",
            "measurement_id": hawor_row.get("measurement_id"),
            "evidence_role": hawor_row.get("evidence_role"),
            "measurement_available": hawor_row.get("measurement_available"),
            "visibility_state": hawor_row.get("visibility_state"),
            "coordinate_frame": hawor_row.get("coordinate_frame"),
            "projection_residual_px_median": hawor_row.get("projection_residual_px_median"),
            "projection_residual_px_p95": hawor_row.get("projection_residual_px_p95"),
            "bbox_xyxy": rounded(hawor_row.get("bbox_xyxy"), 3),
            "source_annotation": hawor_row.get("source_annotation"),
            "state_role": "uncertain_temporal_motion_prior_not_accepted_occlusion_solution",
        }
        if hawor_hand is not None and isinstance(hawor_hand.get("joints2d"), list):
            prior["joints2d_source_px"] = rounded(hawor_hand.get("joints2d")[:21], 2)
        out["hawor_temporal_prior"] = prior
    elif not hawor_available_for_case:
        out["hawor_temporal_prior"] = {"status": "provisioning_failed_no_case_measurements", "state_role": "required_baseline_missing_execution_not_silent_absence"}
        out["uncertainty"].append("hawor_not_executed_or_not_provisioned_for_case")
    else:
        out["hawor_temporal_prior"] = {"status": "not_in_hawor_measurement_window"}
    if occlusion_owner_row is not None:
        source_row = occlusion_owner_row.get("source_row", {}) if isinstance(occlusion_owner_row.get("source_row"), dict) else {}
        out["occlusion_owner_best_effort"] = {
            "status": "temporal_graph_selected_not_strictly_accepted",
            "chosen_owner_object_id": occlusion_owner_row.get("chosen_owner_object_id"),
            "chosen_unary_energy": occlusion_owner_row.get("chosen_unary_energy"),
            "next_best_unary_energy": occlusion_owner_row.get("next_best_unary_energy"),
            "unary_energy_margin": occlusion_owner_row.get("unary_energy_margin"),
            "accepted_occlusion_owner": False,
            "acceptance_blockers": occlusion_owner_row.get("acceptance_blockers"),
            "depth_pair_evidence_state": occlusion_owner_row.get("depth_pair_evidence_state"),
            "mesh_temporal_support": source_row.get("mesh_contact_temporal_support"),
            "state_role": "best_current_tentative_owner_with_blockers_not_pose_fill_acceptance",
        }
    if contact_row is not None:
        oid = str(contact_row.get("chosen_owner_object_id"))
        signed_pen = bool(signed_nonpenetration_row and signed_nonpenetration_row.get("local_penetration_detected"))
        tri_pen = bool(triangle_nonpenetration_row and triangle_nonpenetration_row.get("local_triangle_penetration_detected"))
        if not contact_row.get("accepted_contact_owner"):
            status = "graph_selected_not_accepted"
        elif signed_pen or tri_pen:
            status = "graph_accepted_but_local_penetration_veto"
        else:
            status = "graph_accepted_no_local_penetration_flag"
        out["contact_nonpenetration_state"] = {
            "status": status,
            "chosen_contact_object_id": oid,
            "accepted_contact_owner_before_nonpenetration_veto": contact_row.get("accepted_contact_owner"),
            "min_hand_surface_to_object_mesh_m": contact_row.get("min_hand_surface_to_object_mesh_m"),
            "unary_energy_margin": contact_row.get("unary_energy_margin"),
            "source_row_blockers": contact_row.get("source_row_blockers"),
            "signed_nonpenetration": {
                "available": signed_nonpenetration_row is not None,
                "complete": False,
                "local_penetration_detected": signed_nonpenetration_row.get("local_penetration_detected") if signed_nonpenetration_row else None,
                "min_local_signed_distance_m": signed_nonpenetration_row.get("min_local_signed_distance_m") if signed_nonpenetration_row else None,
                "semantics": signed_nonpenetration_row.get("local_signed_distance_semantics") if signed_nonpenetration_row else None,
            },
            "triangle_nonpenetration": {
                "available": triangle_nonpenetration_row is not None,
                "complete": False,
                "mesh_watertight_by_edges": triangle_nonpenetration_row.get("mesh_watertight_by_edges") if triangle_nonpenetration_row else None,
                "local_triangle_penetration_detected": triangle_nonpenetration_row.get("local_triangle_penetration_detected") if triangle_nonpenetration_row else None,
                "min_local_triangle_signed_distance_m": triangle_nonpenetration_row.get("min_local_triangle_signed_distance_m") if triangle_nonpenetration_row else None,
                "semantics": triangle_nonpenetration_row.get("local_triangle_signed_distance_semantics") if triangle_nonpenetration_row else None,
            },
            "state_role": "contact_graph_selection_with_local_nonpenetration_evidence_not_complete_sdf_solution",
        }
    return out


def object_corrective_state(frame_idx: int, obj: dict[str, Any], graph_pose: dict[str, Any] | None, rigid_candidate: dict[str, Any] | None, stable_pose: list[float] | None, visible_surface_row: dict[str, Any] | None, residual_row: dict[str, Any] | None) -> dict[str, Any]:
    oid = str(obj.get("object_id"))
    out: dict[str, Any] = {
        "object_id": oid,
        "name": obj.get("name"),
        "physical_state_candidate": obj.get("physical_state_candidate"),
        "visibility_state": obj.get("visibility_state"),
        "source_bbox_xyxy": rounded(obj.get("bbox_xyxy"), 3),
        "best_current_state": "source_object_state_no_graph_pose",
        "uncertainty": [],
    }
    if graph_pose is not None:
        out["graph_object_se3"] = {
            "variable_id": graph_pose.get("variable_id"),
            "pose6_world_from_object": rounded(graph_pose.get("pose6_world_from_object"), 6),
            "source": graph_pose.get("source"),
            "observation_residual_norm": graph_pose.get("observation_residual_norm"),
            "state_role": "factor_graph_object_pose_observation",
        }
        out["best_current_state"] = "graph_object_se3_observation"
    else:
        out["uncertainty"].append("missing_factor_graph_object_se3_for_frame")
    if visible_surface_row is not None:
        out["frame_local_visible_surface_state"] = {
            "status": "available_best_visible_geometry_evidence",
            "vertex_count": visible_surface_row.get("vertex_count"),
            "face_count": visible_surface_row.get("face_count"),
            "center_world_m": rounded(visible_surface_row.get("center_world_m"), 6),
            "bbox_world_min_m": rounded(visible_surface_row.get("bbox_world_min_m"), 6),
            "bbox_world_max_m": rounded(visible_surface_row.get("bbox_world_max_m"), 6),
            "world_extent_m": rounded(visible_surface_row.get("world_extent_m"), 6),
            "mask_path": visible_surface_row.get("mask_path"),
            "state_role": "frame_local_rgbd_visible_surface_not_hidden_completion",
        }
        if out["best_current_state"] == "source_object_state_no_graph_pose":
            out["best_current_state"] = "frame_local_visible_surface_geometry"
    if rigid_candidate is not None:
        attempt: dict[str, Any] = {
            "status": "candidate_selected_by_generic_metadata",
            "selection_reasons": rigid_candidate.get("selection_reasons"),
            "model_physical_state_type": rigid_candidate.get("model_physical_state_type"),
            "fast_motion_state": rigid_candidate.get("fast_motion_state"),
            "fused_point_cloud_path": rigid_candidate.get("fused_point_cloud_path"),
            "hidden_geometry_status": rigid_candidate.get("hidden_geometry_status"),
            "object_geometry_complete": False,
            "stable_rigid_prior_method": "componentwise_median_rotation_vector_plus_local_mean_translation_no_object_name_branch",
            "state_role": "generic_rigid_se3_render_driver_attempt",
        }
        out["uncertainty"].append("rigid_candidate_geometry_not_complete")
        if str(rigid_candidate.get("model_physical_state_type")) != "rigid":
            out["uncertainty"].append("rigid_candidate_selected_by_motion_metadata_not_model_rigid_state")
        if stable_pose is not None:
            attempt["stable_pose6_world_from_object"] = rounded(stable_pose, 6)
            attempt["status"] = "stable_pose_available_uncertain_render_driver"
            out["best_current_state"] = "uncertain_rigid_prior_with_frame_local_visible_surface_when_available"
        else:
            attempt["status"] = "selected_but_no_graph_pose_this_frame"
            out["uncertainty"].append("rigid_candidate_without_pose_this_frame")
        if residual_row is not None:
            residual_status = str(residual_row.get("status"))
            attempt["residual_check"] = {
                "status": residual_status,
                "visible_to_fused_median_m": residual_row.get("visible_to_fused_median_m"),
                "visible_to_fused_p95_m": residual_row.get("visible_to_fused_p95_m"),
                "fused_to_visible_p95_m": residual_row.get("fused_to_visible_p95_m"),
                "thresholds_m": residual_row.get("thresholds_m"),
                "state_role": "bidirectional_residual_check_not_pose_acceptance",
            }
            if residual_status == "visible_supported_but_fused_overspread":
                out["best_current_state"] = "frame_local_visible_surface_preferred_fused_geometry_overspread"
                out["uncertainty"].append("fused_canonical_geometry_overspread_relative_to_visible_surface")
            elif residual_status == "visible_surface_not_explained_by_fused_pose":
                out["best_current_state"] = "frame_local_visible_surface_preferred_rigid_pose_residual_rejected"
                out["uncertainty"].append("rigid_pose_residual_rejected_by_visible_surface")
            elif residual_status == "bidirectional_residual_supported_uncertain":
                out["uncertainty"].append("rigid_residual_supported_but_pose_not_accepted")
        else:
            out["uncertainty"].append("rigid_residual_check_missing_for_frame")
        out["generic_rigid_se3_attempt"] = attempt
    return out


def source_dimensions(ann: dict[str, Any]) -> tuple[float, float]:
    raw = ann.get("raw_video", {}) if isinstance(ann.get("raw_video"), dict) else {}
    return finite_float(raw.get("width"), 1920.0), finite_float(raw.get("height"), 1080.0)


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    source_path = args.source_root / case / "annotations_v18_full.json"
    ann = load_json(source_path)
    frames = ann.get("frames", [])
    if not isinstance(frames, list):
        frames = []
    source_w, source_h = source_dimensions(ann)
    hawor_measurement_path = args.measurement_root / case / "measurements_v17" / "hawor_measurements.json"
    hawor_index, hawor_sources = load_hawor_measurement_index(hawor_measurement_path)
    hawor_hands = load_hawor_source_hands(hawor_sources)
    rigid_report_path = args.corrective_root / case / "rigid_se3_attempt" / "v18_rigid_se3_attempt_report.json"
    rigid_candidates = rigid_candidates_from_report(rigid_report_path)
    visible_rows, visible_archive_npz = visible_surface_row_index(args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json", set(rigid_candidates))
    occlusion_owner_rows, occlusion_owner_report = selected_occlusion_owner_index(args.occlusion_owner_graph_root / case / "v18_occlusion_owner_graph_report.json")
    contact_rows, contact_report = selected_contact_index(args.contact_graph_root / case / "v18_contact_ownership_graph_report.json")
    signed_rows = nonpenetration_row_index(args.signed_nonpenetration_root / case / "v18_signed_nonpenetration_evidence_report.json")
    triangle_rows = nonpenetration_row_index(args.triangle_nonpenetration_root / case / "v18_triangle_nonpenetration_evidence_report.json")
    residual_rows, residual_report = rigid_residual_row_index(args.corrective_root / case / "rigid_se3_residual_check" / "v18_rigid_se3_residual_check_report.json")
    stable_pose = stable_rigid_pose_index(frames, set(rigid_candidates), args.translation_smoothing_radius)
    counts: Counter[str] = Counter()
    out_frames: list[dict[str, Any]] = []
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", len(out_frames)))
        graph_hands = graph_hand_estimates(frame, source_w, source_h)
        graph_objects = graph_object_poses(frame)
        hand_states = []
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side") or hand.get("side"))
            state = hand_corrective_state(
                frame_idx,
                hand,
                graph_hands.get(side),
                hawor_index.get((frame_idx, side)),
                hawor_hands.get((frame_idx, side)),
                bool(hawor_index),
                occlusion_owner_rows.get((frame_idx, side)),
                contact_rows.get((frame_idx, side)),
                signed_rows.get((frame_idx, side, str(contact_rows.get((frame_idx, side), {}).get("chosen_owner_object_id")))) if contact_rows.get((frame_idx, side)) else None,
                triangle_rows.get((frame_idx, side, str(contact_rows.get((frame_idx, side), {}).get("chosen_owner_object_id")))) if contact_rows.get((frame_idx, side)) else None,
                source_w,
                source_h,
            )
            if "graph_hand_state" in state:
                counts["graph_hand_states"] += 1
                if state["graph_hand_state"].get("shifted_mano_joints2d_source_px") is not None:
                    counts["graph_shifted_mano_states"] += 1
            prior_status = state.get("hawor_temporal_prior", {}).get("status") if isinstance(state.get("hawor_temporal_prior"), dict) else None
            if prior_status == "available_uncertain_prior":
                counts["hawor_prior_states"] += 1
            if prior_status == "provisioning_failed_no_case_measurements":
                counts["hawor_provisioning_failed_hand_states"] += 1
            if "occlusion_owner_best_effort" in state:
                counts["occlusion_owner_best_effort_states"] += 1
            if "contact_nonpenetration_state" in state:
                counts["contact_nonpenetration_states"] += 1
                contact_status = state["contact_nonpenetration_state"].get("status")
                if isinstance(contact_status, str):
                    counts[f"contact_nonpenetration::{contact_status}"] += 1
            hand_states.append(state)
        object_states = []
        for obj in frame.get("objects", []):
            if not isinstance(obj, dict):
                continue
            oid = str(obj.get("object_id"))
            state = object_corrective_state(frame_idx, obj, graph_objects.get(oid), rigid_candidates.get(oid), stable_pose.get((frame_idx, oid)), visible_rows.get((frame_idx, oid)), residual_rows.get((frame_idx, oid)))
            if "graph_object_se3" in state:
                counts["graph_object_se3_states"] += 1
            if "frame_local_visible_surface_state" in state:
                counts["frame_local_visible_surface_states"] += 1
            rigid_attempt = state.get("generic_rigid_se3_attempt", {}) if isinstance(state.get("generic_rigid_se3_attempt"), dict) else {}
            if rigid_attempt.get("stable_pose6_world_from_object") is not None:
                counts["generic_rigid_stable_pose_states"] += 1
            residual_check = rigid_attempt.get("residual_check", {}) if isinstance(rigid_attempt.get("residual_check"), dict) else {}
            residual_status = residual_check.get("status")
            if isinstance(residual_status, str):
                counts["rigid_residual_checked_states"] += 1
                counts[f"rigid_residual::{residual_status}"] += 1
            object_states.append(state)
        out_frames.append({
            "frame_idx": frame_idx,
            "time_s": frame.get("time_s"),
            "raw_frame_path": frame.get("raw_frame_path"),
            "hands": hand_states,
            "objects": object_states,
            "state_scope": "corrective_delta_only_refs_source_full_annotation_for_other_fields",
        })
    output_path = args.output_root / case / "annotations_v18_corrective_state.json"
    payload = {
        "method": "build_v18_corrective_annotation_state",
        "status": "corrective_state_delta_not_full_v18_closure",
        "case": case,
        "source_annotation": str(source_path),
        "frame_count": len(out_frames),
        "fps": ann.get("fps"),
        "duration_s": ann.get("duration_s"),
        "source_dimensions": {"width": source_w, "height": source_h},
        "claim_scope": "best_current_annotation_state_for_corrective_graph_hand_rigid_object_and_hawor_prior_attempts; does_not_claim_solved_occlusion_contact_or_complete_geometry",
        "corrective_sources": {
            "graph_render_report": str(args.corrective_root / case / "v18_corrective_state_report.json"),
            "rigid_se3_report": str(rigid_report_path),
            "hawor_measurement_file": str(hawor_measurement_path),
            "hawor_source_annotations": [str(p) for p in hawor_sources],
            "hawor_execution_failure_logs": {
                "task5_export_attempt": str(args.corrective_root / "hawor_execution_attempt" / "task5_tomato_960" / "export_hawor_world_attempt.log"),
                "setup_preflight_attempt": str(args.corrective_root / "hawor_execution_attempt" / "setup_preflight" / "remote_setup_hawor_local_attempt.log"),
            },
            "visible_surface_archive_npz": visible_archive_npz,
            "visible_surface_state_report": str(args.corrective_root / case / "visible_surface_state" / "v18_visible_surface_state_report.json"),
            "occlusion_owner_graph_report": str(args.occlusion_owner_graph_root / case / "v18_occlusion_owner_graph_report.json"),
            "occlusion_owner_best_effort_report": str(args.corrective_root / case / "occlusion_owner_best_effort" / "v18_occlusion_owner_best_effort_report.json"),
            "contact_ownership_graph_report": str(args.contact_graph_root / case / "v18_contact_ownership_graph_report.json"),
            "signed_nonpenetration_report": str(args.signed_nonpenetration_root / case / "v18_signed_nonpenetration_evidence_report.json"),
            "triangle_nonpenetration_report": str(args.triangle_nonpenetration_root / case / "v18_triangle_nonpenetration_evidence_report.json"),
            "contact_nonpenetration_state_report": str(args.corrective_root / case / "contact_nonpenetration_state" / "v18_contact_nonpenetration_state_report.json"),
            "rigid_se3_residual_check_report": str(args.corrective_root / case / "rigid_se3_residual_check" / "v18_rigid_se3_residual_check_report.json"),
        },
        "occlusion_owner_selected_rows": len(occlusion_owner_rows),
        "occlusion_owner_strict_accepted_rows": 0,
        "occlusion_owner_acceptance_blocker_counts": occlusion_owner_report.get("acceptance_blocker_counts") if isinstance(occlusion_owner_report, dict) else None,
        "contact_graph_selected_rows": len(contact_rows),
        "contact_graph_accepted_rows_before_nonpenetration_veto": contact_report.get("contact_ownership_accepted_rows") if isinstance(contact_report, dict) else None,
        "rigid_residual_candidate_objects": residual_report.get("candidate_objects") if isinstance(residual_report, dict) else None,
        "counts": dict(sorted(counts.items())),
        "rigid_candidate_ids": sorted(rigid_candidates),
        "hawor_measurement_rows": len(hawor_index),
        "frames": out_frames,
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(output_path, payload)
    return {
        "case": case,
        "output_path": str(output_path),
        "frame_count": len(out_frames),
        "expected_frame_count": ann.get("frame_count"),
        "counts": dict(sorted(counts.items())),
        "rigid_candidate_ids": sorted(rigid_candidates),
        "hawor_measurement_rows": len(hawor_index),
        "elapsed_s": payload["elapsed_s"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_corrective_annotation_state",
        "status": "corrective_annotation_state_delta_not_full_v18_closure",
        "output_root": str(args.output_root),
        "source_root": str(args.source_root),
        "cases": reports,
        "all_frame_counts_match_source": all(r["frame_count"] == r["expected_frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_corrective_annotation_state_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--measurement-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--corrective-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--occlusion-owner-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_graph"))
    parser.add_argument("--contact-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--signed-nonpenetration-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_signed_nonpenetration_evidence"))
    parser.add_argument("--triangle-nonpenetration-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_triangle_nonpenetration_evidence"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--translation-smoothing-radius", type=int, default=3)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
