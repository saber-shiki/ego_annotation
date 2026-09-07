#!/usr/bin/env python3
"""Assess a V20 late-window candidate without changing or optimizing it.

The assessor is fail-closed for visible-pose/render evidence. It does not
consume HOT3D GT and does not make generated geometry physical authority.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from v20_prediction_contracts import assert_prediction_only
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v20_prediction_contracts import assert_prediction_only


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def check_render(render: Mapping[str, Any] | None) -> tuple[bool, list[str], dict[str, Any]]:
    if render is None:
        return False, ["render_report_missing"], {}
    reasons: list[str] = []
    if render.get("status") != "ok":
        reasons.append("render_status_not_ok")
    if render.get("objective_mesh_preserved") is not True:
        reasons.append("objective_mesh_not_preserved")
    contract = render.get("render_mesh_contract")
    if not isinstance(contract, Mapping) or contract.get("validated") is not True or contract.get("path_matches") is not True:
        reasons.append("render_mesh_contract_not_validated")
    stats = render.get("stats")
    stat_count = len(stats) if isinstance(stats, list) else 0
    if stat_count != 150:
        reasons.append(f"render_frame_count:{stat_count}")
    if isinstance(stats, list) and any(int(row.get("dropped_front_faces") or 0) != 0 for row in stats if isinstance(row, Mapping)):
        reasons.append("display_front_face_pruning_detected")
    return not reasons, reasons, {"rendered_frame_count": stat_count, "status": render.get("status")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--shape-report", type=Path, required=True)
    parser.add_argument("--signed-front-audit", type=Path, required=True)
    parser.add_argument("--render-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = load(args.candidate)
    pose = load(args.pose_report)
    shape = load(args.shape_report)
    audit = load(args.signed_front_audit)
    render = load(args.render_report)
    assert_prediction_only(candidate, label="candidate assessment")
    assert_prediction_only(pose, label="candidate assessment pose")
    assert_prediction_only(shape, label="candidate assessment shape")
    assert_prediction_only(audit, label="candidate assessment signed-front audit")

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    if render.get("pose_report") != str(args.candidate.expanduser().resolve()):
        reasons.append("render_pose_binding_mismatch")
    candidate_contract = candidate.get("render_mesh_contract") or {}
    render_contract = render.get("render_mesh_contract") or {}
    if (
        candidate_contract.get("required_mesh_sha256")
        and render_contract.get("actual_sha256")
        and candidate_contract.get("required_mesh_sha256") != render_contract.get("actual_sha256")
    ):
        reasons.append("render_mesh_hash_binding_mismatch")
    checks["candidate_not_annotation_ready"] = candidate.get("annotation_ready") is not True
    checks["gt_not_consumed"] = not bool(candidate.get("gt_consumed")) and not bool(pose.get("gt_consumed")) and not bool(shape.get("gt_consumed"))
    checks["pose_rows_present"] = isinstance(candidate.get("pose_rows"), list) and len(candidate["pose_rows"]) == 150
    checks["pose_status_complete"] = pose.get("status") == "v20_late_window_se3_complete"
    checks["pose_signed_front_checked"] = pose.get("status") == "v20_late_window_se3_complete" and bool((pose.get("solver") or {}).get("signed_front_gate_configured"))
    checks["shape_metric_gates"] = bool(isinstance(shape.get("final_gate_checks"), Mapping) and all(bool(v) for v in shape["final_gate_checks"].values()))
    checks["signed_front_render_gate"] = audit.get("status") == "pass"
    render_ok, render_reasons, render_info = check_render(render)
    checks["render_contract"] = render_ok
    reason_aliases = {
        "pose_status_complete": "pose_status_not_complete",
        "pose_signed_front_checked": "pose_signed_front_unchecked",
        "shape_metric_gates": "shape_metric_gate_failure",
        "signed_front_render_gate": "signed_front_render_gate_failure",
    }
    reasons.extend(reason_aliases.get(key, key) for key, value in checks.items() if not value)
    reasons.extend(render_reasons)
    result = {
        "schema": "v20_late_window_candidate_assessment_v1",
        "status": "ready_for_owner_review" if not reasons else "diagnostic_incomplete",
        "annotation_ready": False,
        "diagnostic_only": True,
        "gt_consumed": False,
        "candidate": str(args.candidate.expanduser().resolve()),
        "checks": checks,
        "reasons": list(dict.fromkeys(reasons)),
        "render": render_info,
        "signed_front_audit_status": audit.get("status"),
        "authority_scope": "generated mesh is visible-pose/render evidence only; not collision/contact/SDF/sign/signed-volume/nonpenetration authority",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "reasons": result["reasons"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
