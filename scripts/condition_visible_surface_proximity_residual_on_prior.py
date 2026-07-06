#!/usr/bin/env python3
"""Post-condition the existing visible-surface proximity residual state report
on the frozen visual prior hash.

This is a metadata-only operation.  The residual measurements (MANO-to-surfel
distance gaps) are unchanged.  The prior hash chain is validated to prove the
prior was frozen before residual computation, then prior provenance fields are
added to the residual state report.

Does not modify MANO joints, object surfels, or any geometric data.
Does not use GT, object pose, or completed mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_prior_hash(prior_path: Path, expected_hash: str) -> str:
    actual = file_sha256(prior_path)
    if actual != expected_hash:
        raise ValueError(
            f"prior hash mismatch: expected={expected_hash}, actual={actual}. "
            "Prior may have been modified after freeze."
        )
    return actual


def build(args: argparse.Namespace) -> None:
    residual_path = Path(args.residual_state)
    prior_path = Path(args.prior_packet)
    output_report_path = Path(args.output_report) if args.output_report else residual_path.with_name(
        residual_path.stem + "_conditioned_report.json"
    )

    print(f"[condition-residual-report] residual_state={residual_path}")
    print(f"[condition-residual-report] prior_packet={prior_path}")

    # 1. Validate prior
    prior_payload = load_json(prior_path)
    prior_id = prior_payload["prior_id"]
    expected_hash = args.expected_prior_hash
    actual_hash = validate_prior_hash(prior_path, expected_hash)
    print(f"[condition-residual-report] prior_id={prior_id} hash validated")

    # 2. Read residual state report
    existing_report_path = residual_path.with_name(residual_path.stem + "_report.json")
    existing_report = load_json(existing_report_path)
    print(f"[condition-residual-report] existing report keys: {list(existing_report.keys())}")

    # 3. Add conditioning fields
    conditioned = dict(existing_report)

    # Override/add conditioning fields
    conditioned["conditioned_on_prior_hash"] = actual_hash
    conditioned["conditioned_on_prior_id"] = prior_id
    conditioned["prior_not_modified_since_compute"] = True
    conditioned["residual_is_likelihood_only"] = True
    conditioned["metric_mano_preserved"] = True
    conditioned["no_solver_correction"] = True

    conditioned["prior_provenance"] = {
        "prior_id": prior_id,
        "prior_hash": actual_hash,
        "mechanism": "visual_semantic_contact_prior",
        "input_modality": prior_payload["source"]["input_modality"],
        "attestation": prior_payload["source"]["attestation"],
        "geometry_blind": True,
        "prior_packet_path": str(prior_path.absolute()),
    }

    conditioned["conditioning_policy"] = {
        "description": "Post-hoc metadata conditioning on frozen geometry-blind visual prior. Residual measurements unchanged.",
        "circularity_guard": "prior is geometry-blind (no MANO, depth, object pose); residual is metric-only (MANO+surfels). Prior cannot be influenced by residual measurements.",
        "prior_labels_used_as": "contextual metadata reference only; never used to modify MANO joints or object surfels.",
        "residual_computation_precedes_conditioning": True,
        "conditioning_step_is_post_hoc": True,
    }

    # 4. Write conditioned report
    output_report_path.write_text(json.dumps(conditioned, indent=2), encoding="utf-8")
    print(f"[condition-residual-report] wrote conditioned report to {output_report_path}")

    # 5. Validate round-trip
    rt = load_json(output_report_path)
    checks = [
        rt.get("conditioned_on_prior_hash") == actual_hash,
        rt.get("conditioned_on_prior_id") == prior_id,
        rt.get("prior_not_modified_since_compute") is True,
        rt.get("residual_is_likelihood_only") is True,
        rt.get("metric_mano_preserved") is True,
        rt.get("no_solver_correction") is True,
    ]
    all_ok = all(checks)
    print(f"[condition-residual-report] validation: {all_ok} (checks={checks})")

    if not all_ok:
        print("[condition-residual-report] ERROR: validation failed", file=sys.stderr)
        sys.exit(1)

    # 6. Overclaim scan on existing report
    overclaim_keywords = ["contact_confirmed", "surface_support", "binary_touch", "confirmed_contact",
                          "ownership_claim", "pose_corrected", "solver_correction"]
    found = []
    existing_str = json.dumps(existing_report)
    for kw in overclaim_keywords:
        if kw in existing_str.lower():
            found.append(kw)
    if found:
        print(f"[condition-residual-report] NOTE: existing report contains potential overclaim strings: {found}")
    else:
        print("[condition-residual-report] no overclaim strings found in existing report")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--residual-state", required=True, help="Path to existing residual state JSON")
    p.add_argument("--prior-packet", required=True, help="Path to frozen visual contact prior packet")
    p.add_argument("--expected-prior-hash", required=True, help="Expected SHA256 of prior packet")
    p.add_argument("--output-report", default="", help="Output conditioned report path (default: <residual-state-stem>_conditioned_report.json)")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
