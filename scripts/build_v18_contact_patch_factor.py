#!/usr/bin/env python3
"""Build generic V18 contact_patch factor records from existing contact hypotheses.

This script does not infer contact from labels alone. It promotes only supported
active contact hypotheses into a solver input that can affect H_t: the solver
will select current MANO vertices near eligible observed object surface and add
a near-contact patch residual. The factor is object-agnostic; target differences
are data fields, not code branches.
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
    p.add_argument("--include-unsupported-near", action="store_true", help="Include raw near-contact proposals without final support. Default is false because unsupported proposals should not constrain H_t.")
    return p.parse_args()


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
            evidence = hyp.get("final_metric_contact_evidence") if isinstance(hyp.get("final_metric_contact_evidence"), dict) else {}
            rows.append(
                {
                    "factor_family": "contact_patch",
                    "target_entity_id": str(args.target_entity_id),
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "variable_affected": "H_t",
                    "observation_type": "supported_active_contact_to_observed_visible_surface_patch",
                    "residual_or_quarantine_rule": "select current MANO vertices near eligible observed object surface and penalize surface-normal distance beyond contact_patch_target_margin_m while allowing tangential sliding; existing nonpenetration handles crossing",
                    "rendered_uncertainty_channel": "normal-contact-patch-constrained MANO hypothesis; no object pose or hidden geometry claim",
                    "state": "active_contact_patch",
                    "weight": float(args.weight),
                    "contact_patch_band_m": float(args.contact_patch_band_m),
                    "contact_patch_target_margin_m": float(args.contact_patch_target_margin_m),
                    "max_vertices": int(args.max_vertices),
                    "source_contact_state": hyp.get("state"),
                    "source_contact_owner_hypothesis": hyp.get("contact_owner_hypothesis"),
                    "source_min_distance_m": evidence.get("min_distance_m"),
                    "source_near_contact_band_m": evidence.get("near_contact_band_m"),
                    "source_contact_coupling_state": (hyp.get("active_contact_coupling_state") or {}).get("coupling_state") if isinstance(hyp.get("active_contact_coupling_state"), dict) else None,
                    "source_stable_contact_pose_anchor_factor_emitted": (hyp.get("active_contact_coupling_state") or {}).get("stable_contact_pose_anchor_factor_emitted") if isinstance(hyp.get("active_contact_coupling_state"), dict) else None,
                    "provenance": {
                        "annotations": str(args.annotations),
                        "frame_contact_hypothesis_key": "frames[].contact_hypotheses[]",
                        "selection_rule": "target object, supported active physical contact, side in left/right",
                        "final_metric_contact_evidence": evidence,
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
        "inputs": {"annotations": str(args.annotations)},
        "parameters": {
            "start_frame": int(args.start_frame),
            "end_frame": int(args.end_frame),
            "weight": float(args.weight),
            "contact_patch_band_m": float(args.contact_patch_band_m),
            "contact_patch_target_margin_m": float(args.contact_patch_target_margin_m),
            "max_vertices": int(args.max_vertices),
            "include_unsupported_near": bool(args.include_unsupported_near),
        },
        "summary": {
            "factor_row_count": len(deduped),
            "skipped_count": len(skipped),
            "frames": sorted({int(r["frame_idx"]) for r in deduped}),
            "sides": sorted({str(r["hand_side"]) for r in deduped}),
        },
        "factor_rows": deduped,
        "skipped_rows_sample": skipped[:50],
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
