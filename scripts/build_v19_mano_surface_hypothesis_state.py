#!/usr/bin/env python3
"""Build a V19 temporal state that separates metric MANO joints from visible-surface hypotheses.

The input state may contain a point-to-plane similarity correction whose
surface samples look physically useful but whose optimized MANO joints are not a
valid metric hand state.  This adapter preserves a chosen source MANO joint state
for evaluation/rendered skeletons and carries the optimized surface samples as an
explicit uncertain visible-surface proximity hypothesis.

It does not accept contact, nonpenetration, or ownership.  It encodes a separate
state variable: a local MANO-to-visible-surface proximity residual constrained by
object geometry.  Contact priors must come from VLM/agent visual evidence, never
from MANO/object geometry distance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contact-state", type=Path, required=True, help="Temporal state containing optimized surface/contact rows")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--joint-source", choices=("hawor_npz", "annotations"), default="hawor_npz")
    p.add_argument("--hawor-npz", type=Path, help="HaWoR NPZ used when --joint-source=hawor_npz and stored for evaluator camera trajectory")
    p.add_argument("--annotations", type=Path, help="Annotations JSON used when --joint-source=annotations")
    p.add_argument("--case", default=None)
    p.add_argument("--object-id", default=None)
    p.add_argument(
        "--accept-signed-full-mano",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Accept the bound full-778 P18 archive only when explicit signed geometry and all global acceptance checks pass.",
    )
    p.add_argument("--max-accepted-visible-joint-shift-px", type=float, default=12.1)
    p.add_argument("--max-accepted-joint-depth-shift-m", type=float, default=0.035)
    p.add_argument("--max-accepted-penetration-residual-m", type=float, default=0.001)
    return p.parse_args()


def annotation_joint_map(path: Path) -> dict[tuple[int, str], np.ndarray]:
    data = load_json(path)
    rows: dict[tuple[int, str], np.ndarray] = {}
    for pos, frame in enumerate(data.get("frames") if isinstance(data.get("frames"), list) else []):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", pos))
        for hand in frame.get("hands") if isinstance(frame.get("hands"), list) else []:
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side", ""))
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints = np.asarray(metric.get("joints_current_v18_world_m") or metric.get("joints_world_m") or [], dtype=np.float64)
            if side in {"left", "right"} and joints.shape == (21, 3) and np.isfinite(joints).all():
                rows[(frame_idx, side)] = joints
    return rows


def hawor_joint_map(path: Path) -> dict[tuple[int, str], np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        frame_idx = np.asarray(z["frame_idx"], dtype=int) if "frame_idx" in z.files else None
        if frame_idx is None:
            # Infer from joint array length if frame_idx is missing.
            n = int(np.asarray(z["left_joints_world_m"]).shape[0])
            frame_idx = np.arange(n, dtype=int)
        rows: dict[tuple[int, str], np.ndarray] = {}
        for side in ("left", "right"):
            key = f"{side}_joints_world_m"
            valid_key = f"{side}_valid"
            if key not in z.files:
                continue
            joints_all = np.asarray(z[key], dtype=np.float64)
            valid = np.asarray(z[valid_key]).astype(bool) if valid_key in z.files else np.ones((len(frame_idx),), dtype=bool)
            for i, frame in enumerate(frame_idx.tolist()):
                if i >= len(joints_all) or i >= len(valid) or not bool(valid[i]):
                    continue
                joints = np.asarray(joints_all[i], dtype=np.float64)
                if joints.shape == (21, 3) and np.isfinite(joints).all():
                    rows[(int(frame), side)] = joints
        return rows


def zero_summary() -> dict[str, Any]:
    return {"count": 21, "min": 0.0, "p10": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}


def numeric_summary(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def row_contact_median(row: dict[str, Any], field: str) -> float | None:
    contact = row.get("contact_similarity_refit") if isinstance(row.get("contact_similarity_refit"), dict) else {}
    report = contact.get(field)
    if isinstance(report, dict) and report.get("median") is not None:
        try:
            return float(report["median"])
        except (TypeError, ValueError):
            return None
    return None


def signed_full_mano_acceptance(
    contact_state: dict[str, Any],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any], dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]]:
    surface = contact_state.get("physical_surface_contract") if isinstance(contact_state.get("physical_surface_contract"), dict) else {}
    signed_active = surface.get("signed_object_surface_factor_active") is True
    signed_declared = surface.get("signed_geometry_declared_ready") is True
    quarantined = contact_state.get("physical_state_quarantined") is True
    archive_value = contact_state.get("full_mano_vertices_world_archive")
    archive = Path(str(archive_value or ""))
    expected_sha = str(contact_state.get("full_mano_vertices_world_archive_sha256") or "")
    blockers: list[str] = []
    if not bool(args.accept_signed_full_mano):
        blockers.append("signed_full_mano_acceptance_not_requested")
    if not signed_active or not signed_declared:
        blockers.append("signed_object_surface_factor_not_explicitly_active")
    if quarantined:
        blockers.append("p18_physical_state_quarantined")
    if not archive.is_file() or archive.stat().st_size <= 0:
        blockers.append("missing_full_mano_vertices_archive")
    elif not expected_sha or sha256_file(archive) != expected_sha:
        blockers.append("full_mano_vertices_archive_hash_mismatch")

    keys = {(int(row["frame_idx"]), str(row["hand_side"])) for row in rows}
    object_delta_max = 0.0
    visible_shift_max = 0.0
    depth_shift_max = 0.0
    penetration_residual_max = 0.0
    full_penetration_after_uncertainty_max = 0.0
    unauthorized_penetration_max = 0.0
    unauthorized_penetrating_vertex_count = 0
    active_set_closed = True
    output_translation_gate_applied_count = 0
    for row in rows:
        object_delta = np.asarray(row.get("optimized_object_translation_world_m") or [], dtype=np.float64)
        if object_delta.shape != (3,) or not np.isfinite(object_delta).all():
            blockers.append("malformed_object_translation_delta")
            continue
        object_delta_max = max(object_delta_max, float(np.linalg.norm(object_delta)))
        visible = row.get("visible_joint_shift_px") if isinstance(row.get("visible_joint_shift_px"), dict) else {}
        depth = row.get("joint_camera_depth_shift_m") if isinstance(row.get("joint_camera_depth_shift_m"), dict) else {}
        residual = row.get("final_active_constraint_residual_after_solver_m") if isinstance(row.get("final_active_constraint_residual_after_solver_m"), dict) else {}
        full_penetration = row.get("full_observed_surface_penetration_after_solver_m") if isinstance(row.get("full_observed_surface_penetration_after_solver_m"), dict) else {}
        unauthorized_penetration = row.get("full_unauthorized_surface_penetration_after_solver_m") if isinstance(row.get("full_unauthorized_surface_penetration_after_solver_m"), dict) else {}
        support_uncertainty = max(0.0, float(row.get("observed_surface_support_uncertainty_m") or 0.0))
        visible_shift_max = max(visible_shift_max, float(visible.get("max") or 0.0))
        depth_shift_max = max(depth_shift_max, float(depth.get("max") or 0.0))
        penetration_residual_max = max(penetration_residual_max, float(residual.get("max") or 0.0))
        full_penetration_after_uncertainty_max = max(
            full_penetration_after_uncertainty_max,
            max(0.0, float(full_penetration.get("max") or 0.0) - support_uncertainty),
        )
        unauthorized_penetration_max = max(
            unauthorized_penetration_max,
            float(unauthorized_penetration.get("max") or 0.0),
        )
        unauthorized_penetrating_vertex_count += int(
            row.get("full_unauthorized_surface_penetrating_vertex_count_after_solver")
            or 0
        )
        if not str(row.get("signed_object_surface_factor_state") or "").startswith("active_explicit"):
            active_set_closed = False
        output_gate = (
            row.get("output_translation_gate")
            if isinstance(row.get("output_translation_gate"), dict)
            else {}
        )
        if output_gate.get("applied") is True:
            output_translation_gate_applied_count += 1
    for interval in contact_state.get("intervals") or []:
        if isinstance(interval, dict) and interval.get("active_set_closed") is not True:
            active_set_closed = False
    if object_delta_max > 1.0e-10:
        blockers.append("nonzero_private_object_translation")
    if visible_shift_max > float(args.max_accepted_visible_joint_shift_px):
        blockers.append("visible_joint_shift_exceeds_bound")
    if depth_shift_max > float(args.max_accepted_joint_depth_shift_m):
        blockers.append("joint_depth_shift_exceeds_bound")
    if penetration_residual_max > float(args.max_accepted_penetration_residual_m):
        blockers.append("signed_penetration_residual_exceeds_bound")
    if full_penetration_after_uncertainty_max > float(args.max_accepted_penetration_residual_m):
        blockers.append("full_signed_penetration_after_uncertainty_exceeds_bound")
    if unauthorized_penetrating_vertex_count > 0 or unauthorized_penetration_max > 0.0:
        blockers.append("unresolved_penetration_nearest_unauthorized_closure_face")
    if not active_set_closed:
        blockers.append("signed_active_set_not_closed")
    if output_translation_gate_applied_count:
        blockers.append("published_candidate_contains_translation_gated_rows")

    archive_rows: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    if not blockers and archive.is_file():
        with np.load(archive, allow_pickle=False) as z:
            frame_idx = np.asarray(z["frame_idx"], dtype=np.int64)
            hand_side = np.asarray(z["hand_side"]).astype(str)
            vertices = np.asarray(z["vertices_world_m"], dtype=np.float64)
            joints = np.asarray(z["joints_world_m"], dtype=np.float64)
        if vertices.shape != (len(frame_idx), 778, 3) or joints.shape != (len(frame_idx), 21, 3) or len(hand_side) != len(frame_idx):
            blockers.append("invalid_full_mano_archive_shape")
        elif not np.isfinite(vertices).all() or not np.isfinite(joints).all():
            blockers.append("nonfinite_full_mano_archive")
        else:
            for pos, frame in enumerate(frame_idx.tolist()):
                key = (int(frame), str(hand_side[pos]))
                if key in archive_rows:
                    blockers.append("duplicate_full_mano_archive_key")
                    break
                archive_rows[key] = (vertices[pos], joints[pos])
            if set(archive_rows) != keys:
                blockers.append("full_mano_archive_timeline_mismatch")
    accepted = not blockers
    report = {
        "requested": bool(args.accept_signed_full_mano),
        "accepted": accepted,
        "blockers": sorted(set(blockers)),
        "signed_object_surface_factor_active": signed_active,
        "signed_geometry_declared_ready": signed_declared,
        "physical_state_quarantined": quarantined,
        "full_mano_vertices_world_archive": str(archive) if archive_value else None,
        "full_mano_vertices_world_archive_sha256": expected_sha or None,
        "row_count": len(rows),
        "archive_row_count": len(archive_rows),
        "max_private_object_translation_delta_m": object_delta_max,
        "max_visible_joint_shift_px": visible_shift_max,
        "max_joint_camera_depth_shift_m": depth_shift_max,
        "max_signed_penetration_residual_m": penetration_residual_max,
        "max_full_signed_penetration_after_uncertainty_m": full_penetration_after_uncertainty_max,
        "max_unauthorized_closure_penetration_m": unauthorized_penetration_max,
        "unauthorized_closure_penetrating_vertex_count": int(
            unauthorized_penetrating_vertex_count
        ),
        "signed_active_set_closed": active_set_closed,
        "output_translation_gate_applied_count": int(
            output_translation_gate_applied_count
        ),
        "thresholds": {
            "max_accepted_visible_joint_shift_px": float(args.max_accepted_visible_joint_shift_px),
            "max_accepted_joint_depth_shift_m": float(args.max_accepted_joint_depth_shift_m),
            "max_accepted_penetration_residual_m": float(args.max_accepted_penetration_residual_m),
        },
    }
    return accepted, report, archive_rows


def main() -> None:
    args = parse_args()
    if args.joint_source == "hawor_npz" and args.hawor_npz is None:
        raise SystemExit("--hawor-npz is required with --joint-source=hawor_npz")
    if args.joint_source == "annotations" and args.annotations is None:
        raise SystemExit("--annotations is required with --joint-source=annotations")

    contact_state = load_json(args.contact_state)
    contact_rows = contact_state.get("per_frame_states")
    if not isinstance(contact_rows, list) or not contact_rows:
        raise SystemExit(f"{args.contact_state} lacks nonempty per_frame_states")
    contact_pose_readiness = contact_state.get("object_pose_readiness") if isinstance(contact_state.get("object_pose_readiness"), dict) else {}
    signed_full_accepted, signed_full_report, signed_full_rows = signed_full_mano_acceptance(
        contact_state,
        [row for row in contact_rows if isinstance(row, dict)],
        args,
    )
    input_pose_quarantined = bool(
        contact_state.get("optimization_skipped") is True
        or contact_state.get("physical_state_quarantined") is True
        or contact_state.get("status") == "completed_quarantined_unready_object_pose_trajectory"
        or contact_pose_readiness.get("explicitly_unready") is True
    )

    if args.joint_source == "hawor_npz":
        source_joints = hawor_joint_map(args.hawor_npz)  # type: ignore[arg-type]
        source_desc = str(args.hawor_npz)
    else:
        source_joints = annotation_joint_map(args.annotations)  # type: ignore[arg-type]
        source_desc = str(args.annotations)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    normal_vals: list[float] = []
    tangent_vals: list[float] = []
    distance_vals: list[float] = []
    for row in contact_rows:
        if not isinstance(row, dict):
            continue
        try:
            key = (int(row["frame_idx"]), str(row["hand_side"]))
        except Exception:
            continue
        joints = source_joints.get(key)
        if joints is None:
            skipped.append({"frame_idx": key[0], "hand_side": key[1], "reason": "missing_source_joints"})
            continue
        signed_full = signed_full_rows.get(key) if signed_full_accepted else None
        out = dict(row)
        original_joints = np.asarray(row.get("optimized_joints_world_m") or [], dtype=np.float64)
        if original_joints.shape == (21, 3) and np.isfinite(original_joints).all():
            out["surface_fit_joints_world_m"] = original_joints.astype(float).tolist()
            out["surface_fit_joint_delta_from_source_m"] = numeric_summary(np.linalg.norm(original_joints - joints, axis=1).astype(float).tolist())
        accepted_vertices = signed_full[0] if signed_full is not None else None
        accepted_joints = signed_full[1] if signed_full is not None else joints
        out["optimized_joints_world_m"] = accepted_joints.astype(float).tolist()
        out["joint_state_policy"] = (
            "p18_signed_full_778_mano_accepted_shared_prebranch"
            if signed_full is not None
            else f"{args.joint_source}_metric_mano_preserved_due_to_unready_object_pose"
            if input_pose_quarantined
            else f"{args.joint_source}_metric_mano_preserved"
        )
        out["temporal_mano_state"] = (
            "v19_shared_signed_p18_full_mano_accepted"
            if signed_full is not None
            else "v19_source_metric_mano_only_object_pose_quarantine"
            if input_pose_quarantined
            else "v19_source_metric_mano_plus_uncertain_contact_surface_hypothesis"
        )
        out["accepted_full_mano_archive_key"] = (
            {"frame_idx": int(key[0]), "hand_side": str(key[1])}
            if signed_full is not None
            else None
        )
        out["accepted_full_mano_vertices_world_archive"] = (
            signed_full_report.get("full_mano_vertices_world_archive")
            if signed_full is not None
            else None
        )
        out["accepted_full_mano_vertices_world_archive_sha256"] = (
            signed_full_report.get("full_mano_vertices_world_archive_sha256")
            if signed_full is not None
            else None
        )
        out["accepted_full_mano_state"] = (
            "accepted_signed_p18_full_778" if accepted_vertices is not None else "source_metric_mano_preserved"
        )
        out["source_metric_mano_state"] = {"kind": args.joint_source, "path": source_desc}
        if args.hawor_npz is not None:
            out["source_hawor_npz"] = str(args.hawor_npz)
        out["visible_surface_hypothesis_state"] = (
            "accepted_signed_nonpenetration_surface_fit"
            if signed_full is not None
            else "not_built_unready_object_pose_trajectory"
            if input_pose_quarantined
            else "uncertain_visible_surface_proximity_not_contact_ownership"
        )
        out["contact_surface_vertices_world_sample_m"] = (
            [] if input_pose_quarantined else (out.get("optimized_vertices_world_sample_m") or [])
        )
        if input_pose_quarantined:
            out["optimized_vertices_world_sample_m"] = []
            out["annotation_ready"] = False
            out["physical_constraint_quarantine"] = "p15_unready_object_pose_trajectory"
            out["contact_state"] = "unresolved_object_pose_trajectory"
        if signed_full is None:
            out["metric_joint_shift_px"] = zero_summary()
            out["visible_joint_shift_px"] = zero_summary()
            out["optimized_similarity_scale"] = 1.0
            out["optimized_rotation_norm_rad"] = 0.0
            out["optimized_rotation_vector_camera_rad"] = [0.0, 0.0, 0.0]
            out["optimized_rotation_vector_world_rad"] = [0.0, 0.0, 0.0]
            out["optimized_translation_camera_m"] = [0.0, 0.0, 0.0]
            out["optimized_translation_world_m"] = [0.0, 0.0, 0.0]
        else:
            out["metric_joint_shift_px"] = out.get("visible_joint_shift_px")
            out["accepted_full_mano_residuals_preserved_from_p18"] = True
        for field, vals in (
            ("contact_normal_abs_after_m", normal_vals),
            ("contact_tangent_after_m", tangent_vals),
            ("contact_distance_after_m", distance_vals),
        ):
            val = row_contact_median(row, field)
            if val is not None:
                vals.append(val)
        rows.append(out)

    if not rows:
        raise SystemExit(f"no rows could be built from {args.contact_state}; skipped={skipped[:10]}")

    payload = {
        "method": "v19_shared_signed_or_source_metric_mano_acceptance_state",
        "status": (
            "ok_signed_full_mano_accepted"
            if signed_full_accepted
            else "completed_source_metric_mano_object_pose_quarantine"
            if input_pose_quarantined
            else "ok"
        ),
        "annotation_ready": False,
        "physical_constraint_quarantined": input_pose_quarantined,
        "case": args.case or contact_state.get("case"),
        "object_id": args.object_id or contact_state.get("object_id"),
        "claim_scope": (
            "P15 object-pose support is insufficient, so metric MANO joints are preserved from the selected source and no object-relative visible-surface/contact hypothesis is exposed."
            if input_pose_quarantined
            else (
                "Explicitly ready shared signed geometry and bounded P18 acceptance promoted one full-778 MANO state before the backend branch. "
                "Object translation remains frozen and contact ownership still comes only from visual evidence."
                if signed_full_accepted
                else "Metric MANO joints are preserved from the selected source; optimized point-to-plane vertices are rendered only as "
                "uncertain visible-surface proximity hypotheses. This state does not accept contact ownership or nonpenetration. "
                "Contact priors must come from VLM/agent visual evidence, never from MANO/object geometry distance."
            )
        ),
        "inputs": {
            "contact_state": str(args.contact_state),
            "joint_source": args.joint_source,
            "source_path": source_desc,
            "source_hawor_npz": str(args.hawor_npz) if args.hawor_npz is not None else None,
            "annotations": str(args.annotations) if args.annotations is not None else None,
        },
        "full_mano_acceptance": signed_full_report,
        "accepted_full_mano_vertices_world_archive": (
            signed_full_report.get("full_mano_vertices_world_archive")
            if signed_full_accepted
            else None
        ),
        "accepted_full_mano_vertices_world_archive_sha256": (
            signed_full_report.get("full_mano_vertices_world_archive_sha256")
            if signed_full_accepted
            else None
        ),
        "summary": {
            "contact_rows_in": len(contact_rows),
            "rows_out": len(rows),
            "skipped_count": len(skipped),
            "input_pose_quarantined": input_pose_quarantined,
            "visible_surface_normal_abs_after_median": numeric_summary(normal_vals),
            "visible_surface_tangent_after_median": numeric_summary(tangent_vals),
            "visible_surface_distance_after_median": numeric_summary(distance_vals),
            "contact_normal_abs_after_median": numeric_summary(normal_vals),
            "contact_tangent_after_median": numeric_summary(tangent_vals),
            "contact_distance_after_median": numeric_summary(distance_vals),
            "metric_joint_shift_px": zero_summary(),
            "signed_full_mano_acceptance": signed_full_report,
        },
        "skipped_preview": skipped[:50],
        "per_frame_states": rows,
    }
    write_json(args.output, payload)
    report = {k: v for k, v in payload.items() if k != "per_frame_states"}
    write_json(args.output.with_name(args.output.stem + "_report.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
