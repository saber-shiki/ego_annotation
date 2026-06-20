#!/usr/bin/env python3
"""Build generic V18 contact_patch factor records from existing contact hypotheses.

This script does not infer contact from labels alone. It promotes only supported
active contact hypotheses into a solver input that can affect H_t: the solver
will select current MANO vertices near eligible observed object surface and add
a near-contact patch residual. When an independent object-pose support report is
provided, the emitted residual is bounded by that support uncertainty so contact
acts as a latent/sliding patch likelihood rather than a hard current-surface
anchor. The factor is object-agnostic; target differences are data fields, not
code branches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def numeric_summary(vals: list[float]) -> dict[str, Any]:
    finite = sorted(float(v) for v in vals if isinstance(v, (int, float)))
    if not finite:
        return {"count": 0}
    def q(frac: float) -> float:
        idx = min(len(finite) - 1, max(0, int(round(frac * (len(finite) - 1)))))
        return finite[idx]
    return {
        "count": len(finite),
        "min": finite[0],
        "median": q(0.5),
        "p90": q(0.9),
        "p95": q(0.95),
        "max": finite[-1],
    }


def nested_get(row: dict[str, Any], dotted: str) -> Any:
    cur: Any = row
    for key in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def load_object_support_uncertainty(path: Path | None, *, stat: str, default_m: float) -> dict[int, float]:
    if path is None:
        return {}
    payload = load_json(path)
    pose_rows = payload.get("pose_rows")
    if not isinstance(pose_rows, list):
        raise ValueError(f"object pose fit report has no pose_rows list: {path}")
    out: dict[int, float] = {}
    for row in pose_rows:
        if not isinstance(row, dict) or "frame_idx" not in row:
            continue
        raw = nested_get(row, stat)
        if raw is None:
            raw = default_m
        try:
            val = float(raw)
        except Exception:
            val = float(default_m)
        if val < 0.0:
            val = float(default_m)
        out[int(row["frame_idx"])] = val
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--target-entity-id", required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weight", type=float, default=5.0e4)
    p.add_argument("--contact-patch-band-m", type=float, default=0.020)
    p.add_argument("--contact-patch-target-margin-m", type=float, default=0.0025)
    p.add_argument("--max-vertices", type=int, default=96)
    p.add_argument("--object-pose-fit-report", type=Path, default=None, help="Optional independent object pose/support report. When supplied, per-frame support uncertainty is added to the contact deadband so the factor cannot force sub-support-scale MANO motion.")
    p.add_argument("--object-support-uncertainty-stat", default="observed_to_mesh_final.p95_m", help="Dotted field in pose_rows[] used as object_support_uncertainty_m. Default uses visible-depth-to-mesh p95 support.")
    p.add_argument("--default-object-support-uncertainty-m", type=float, default=0.0, help="Fallback support uncertainty when the pose report is absent or lacks the selected stat.")
    p.add_argument("--contact-evidence-report", type=Path, default=None, help="Optional independent contact evidence report. With --require-independent-contact-evidence, contact_patch rows are emitted only when the report has matching visual association plus metric-depth compatibility or an accepted contact owner.")
    p.add_argument("--require-independent-contact-evidence", action="store_true", help="Reject annotation-only/proximity-only contact hypotheses unless --contact-evidence-report has independent visual+metric support for the same target/frame/side.")
    p.add_argument("--include-unsupported-near", action="store_true", help="Include raw near-contact proposals without final support. Default is false because unsupported proposals should not constrain H_t.")
    return p.parse_args()


def load_contact_evidence(path: Path | None, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None:
        return {}
    payload = load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"contact evidence report has no rows list: {path}")
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("object_id")) != str(target_entity_id):
            continue
        side = str(row.get("hand_side") or "")
        if side not in {"left", "right"}:
            continue
        key = (int(row["frame_idx"]), side)
        if key in out:
            raise ValueError(f"duplicate contact evidence row for {key} and target {target_entity_id}: {path}")
        out[key] = row
    return out


def independent_contact_evidence_supported(row: dict[str, Any] | None) -> tuple[bool, str]:
    if not isinstance(row, dict):
        return False, "missing_independent_contact_evidence_row"
    ev = row.get("source_contact_evidence") if isinstance(row.get("source_contact_evidence"), dict) else {}
    graph = ev.get("contact_ownership_graph") if isinstance(ev.get("contact_ownership_graph"), dict) else {}
    accepted_owner = bool(graph.get("accepted_contact_owner")) or str(row.get("contact_owner_claim") or "").startswith("accepted")
    image_supported = bool(ev.get("image_overlap_candidate") or ev.get("pair_contact_image_candidate"))
    metric_supported = bool(ev.get("metric_depth_compatible_candidate"))
    depth_gap_state = str(ev.get("pair_depth_gap_state") or "")
    source_state = str(row.get("source_contact_state") or "")
    if source_state == "image_contact_rejected_by_metric_depth" or depth_gap_state in {"hand_behind_object_depth", "object_behind_hand_depth"}:
        return False, f"metric_depth_rejects_visual_contact:{source_state}:{depth_gap_state}"
    if accepted_owner:
        return True, "accepted_contact_owner"
    if image_supported and metric_supported:
        return True, "visual_association_and_metric_depth_compatible"
    if image_supported and not metric_supported:
        return False, "visual_association_without_metric_depth_support"
    if metric_supported and not image_supported:
        return False, "metric_depth_without_visual_association"
    return False, "no_independent_visual_metric_contact_support"


def contact_supported(row: dict[str, Any], *, include_unsupported_near: bool) -> bool:
    if row.get("physical_contact_claim_supported") is True:
        return True
    if not include_unsupported_near:
        return False
    state = str(row.get("state") or row.get("contact_physical_mode") or "")
    evidence = row.get("final_metric_contact_evidence") if isinstance(row.get("final_metric_contact_evidence"), dict) else {}
    return state == "raw_contact_proposal_without_final_validated_physical_support" and evidence.get("contact_switch_observation") == "near"


def main() -> None:
    args = parse_args()
    payload = load_json(args.annotations)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"annotations file has no frames list: {args.annotations}")
    support_by_frame = load_object_support_uncertainty(
        args.object_pose_fit_report,
        stat=str(args.object_support_uncertainty_stat),
        default_m=float(args.default_object_support_uncertainty_m),
    )
    contact_evidence = load_contact_evidence(args.contact_evidence_report, str(args.target_entity_id))
    if bool(args.require_independent_contact_evidence) and args.contact_evidence_report is None:
        raise ValueError("--require-independent-contact-evidence requires --contact-evidence-report")
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < int(args.start_frame) or frame_idx > int(args.end_frame):
            continue
        for hyp in frame.get("contact_hypotheses") or []:
            if not isinstance(hyp, dict):
                continue
            if str(hyp.get("object_id")) != str(args.target_entity_id):
                continue
            side = str(hyp.get("hand_side") or "")
            if side not in {"left", "right"}:
                skipped.append({"frame_idx": frame_idx, "reason": "missing_or_invalid_hand_side", "hypothesis": hyp})
                continue
            if not contact_supported(hyp, include_unsupported_near=bool(args.include_unsupported_near)):
                skipped.append({"frame_idx": frame_idx, "hand_side": side, "reason": "contact_not_supported", "state": hyp.get("state"), "physical_contact_claim_supported": hyp.get("physical_contact_claim_supported")})
                continue
            independent_evidence_row = contact_evidence.get((frame_idx, side)) if contact_evidence else None
            independent_supported, independent_reason = independent_contact_evidence_supported(independent_evidence_row)
            if bool(args.require_independent_contact_evidence) and not independent_supported:
                skipped.append({
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "reason": "independent_contact_evidence_rejected",
                    "independent_contact_evidence_reason": independent_reason,
                    "annotation_state": hyp.get("state"),
                    "annotation_contact_owner_hypothesis": hyp.get("contact_owner_hypothesis"),
                })
                continue
            evidence = hyp.get("final_metric_contact_evidence") if isinstance(hyp.get("final_metric_contact_evidence"), dict) else {}
            support_uncertainty_m = float(support_by_frame.get(frame_idx, float(args.default_object_support_uncertainty_m)))
            contact_deadband_m = float(args.contact_patch_target_margin_m) + max(0.0, support_uncertainty_m)
            rows.append(
                {
                    "factor_family": "contact_patch",
                    "target_entity_id": str(args.target_entity_id),
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "variable_affected": "H_t",
                    "observation_type": "supported_active_contact_to_uncertain_observed_visible_surface_patch",
                    "residual_or_quarantine_rule": "select current MANO vertices near eligible observed object surface and penalize surface-normal distance only beyond contact_patch_target_margin_m + object_support_uncertainty_m while allowing tangential sliding; existing nonpenetration handles crossing",
                    "rendered_uncertainty_channel": "bounded latent/sliding contact patch MANO hypothesis; no object pose or hidden geometry claim",
                    "state": "active_contact_patch",
                    "weight": float(args.weight),
                    "contact_patch_band_m": float(args.contact_patch_band_m),
                    "contact_patch_target_margin_m": float(args.contact_patch_target_margin_m),
                    "object_support_uncertainty_m": max(0.0, support_uncertainty_m),
                    "contact_patch_support_uncertainty_m": max(0.0, support_uncertainty_m),
                    "contact_patch_deadband_m": contact_deadband_m,
                    "max_vertices": int(args.max_vertices),
                    "source_contact_state": hyp.get("state"),
                    "source_contact_owner_hypothesis": hyp.get("contact_owner_hypothesis"),
                    "source_min_distance_m": evidence.get("min_distance_m"),
                    "source_near_contact_band_m": evidence.get("near_contact_band_m"),
                    "source_object_support_uncertainty_stat": str(args.object_support_uncertainty_stat),
                    "source_object_pose_fit_report": str(args.object_pose_fit_report) if args.object_pose_fit_report else None,
                    "source_contact_coupling_state": (hyp.get("active_contact_coupling_state") or {}).get("coupling_state") if isinstance(hyp.get("active_contact_coupling_state"), dict) else None,
                    "source_stable_contact_pose_anchor_factor_emitted": (hyp.get("active_contact_coupling_state") or {}).get("stable_contact_pose_anchor_factor_emitted") if isinstance(hyp.get("active_contact_coupling_state"), dict) else None,
                    "independent_contact_evidence_supported": bool(independent_supported),
                    "independent_contact_evidence_reason": independent_reason,
                    "source_contact_evidence_report": str(args.contact_evidence_report) if args.contact_evidence_report else None,
                    "provenance": {
                        "annotations": str(args.annotations),
                        "frame_contact_hypothesis_key": "frames[].contact_hypotheses[]",
                        "selection_rule": "target object, supported active physical contact, side in left/right, and independent contact evidence when required",
                        "final_metric_contact_evidence": evidence,
                        "independent_contact_evidence_row": independent_evidence_row,
                    },
                }
            )
    seen: set[tuple[int, str]] = set()
    deduped: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        key = (int(row["frame_idx"]), str(row["hand_side"]))
        if key in seen:
            duplicates.append(row)
            continue
        seen.add(key)
        deduped.append(row)
    if duplicates:
        raise ValueError(f"duplicate contact_patch rows for target/side/frame: {duplicates[:3]}")
    report = {
        "method": "v18_contact_patch_factor_from_supported_contact_hypotheses",
        "case": str(args.case),
        "target_entity_id": str(args.target_entity_id),
        "claim_scope": "Generic H_t contact patch residual input. It can constrain MANO hand state near observed eligible surface patches; it does not by itself prove object pose, hidden geometry, contact closure, or nonpenetration.",
        "inputs": {
            "annotations": str(args.annotations),
            "object_pose_fit_report": str(args.object_pose_fit_report) if args.object_pose_fit_report else None,
            "contact_evidence_report": str(args.contact_evidence_report) if args.contact_evidence_report else None,
        },
        "parameters": {
            "start_frame": int(args.start_frame),
            "end_frame": int(args.end_frame),
            "weight": float(args.weight),
            "contact_patch_band_m": float(args.contact_patch_band_m),
            "contact_patch_target_margin_m": float(args.contact_patch_target_margin_m),
            "max_vertices": int(args.max_vertices),
            "object_support_uncertainty_stat": str(args.object_support_uncertainty_stat),
            "default_object_support_uncertainty_m": float(args.default_object_support_uncertainty_m),
            "include_unsupported_near": bool(args.include_unsupported_near),
            "require_independent_contact_evidence": bool(args.require_independent_contact_evidence),
        },
        "summary": {
            "factor_row_count": len(deduped),
            "skipped_count": len(skipped),
            "frames": sorted({int(r["frame_idx"]) for r in deduped}),
            "sides": sorted({str(r["hand_side"]) for r in deduped}),
            "object_support_uncertainty_m": numeric_summary([float(r.get("object_support_uncertainty_m", 0.0)) for r in deduped]),
            "contact_patch_deadband_m": numeric_summary([float(r.get("contact_patch_deadband_m", 0.0)) for r in deduped]),
            "independent_contact_evidence_rejected_count": sum(1 for r in skipped if r.get("reason") == "independent_contact_evidence_rejected"),
        },
        "factor_rows": deduped,
        "skipped_rows_sample": skipped[:50],
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
