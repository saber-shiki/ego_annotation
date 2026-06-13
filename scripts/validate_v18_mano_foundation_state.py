#!/usr/bin/env python3
"""Validate the V18 MANO foundation audit/state.

This validator is intentionally strict about scope: recovered MANO candidates are
evidence, but the current V18 physical pipeline remains invalid unless a full
metric MANO foundation is present and explicitly marked valid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED = {
    "trash_1050": {"expected_rows": 2100, "wilor_rows": 1617, "wilor_unique_frame_side_rows": 1601, "hawor_rows": 182},
    "task5_tomato_960": {"expected_rows": 1920, "wilor_rows": 1744, "wilor_unique_frame_side_rows": 1733, "hawor_rows": 0},
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


def finite(value: Any) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def validate_npz(case: str, report: dict[str, Any], failures: list[str]) -> None:
    wilor = report.get("recovered_wilor_world_mano_candidates", {}) if isinstance(report.get("recovered_wilor_world_mano_candidates"), dict) else {}
    rows = int(wilor.get("complete_world_rows", 0))
    npz_path_raw = wilor.get("npz_path")
    require(isinstance(npz_path_raw, str) and bool(npz_path_raw), f"{case}: missing WiLoR MANO NPZ path", failures)
    if not isinstance(npz_path_raw, str):
        return
    npz_path = Path(npz_path_raw)
    require(npz_path.exists(), f"{case}: WiLoR MANO NPZ does not exist: {npz_path}", failures)
    if not npz_path.exists():
        return
    arrays = np.load(npz_path)
    expected_shapes = {
        "frame_idx": (rows,),
        "hand_side_code": (rows,),
        "detector_score": (rows,),
        "bbox_xyxy": (rows, 4),
        "source_intrinsics": (rows, 4),
        "cam_t": (rows, 3),
        "joints_world_m": (rows, 21, 3),
        "vertices_world_m": (rows, 778, 3),
        "mano_global_orient": (rows, 9),
        "mano_hand_pose": (rows, 135),
        "mano_betas": (rows, 10),
    }
    for key, shape in expected_shapes.items():
        require(key in arrays.files, f"{case}: NPZ missing array {key}", failures)
        if key in arrays.files:
            require(tuple(arrays[key].shape) == shape, f"{case}: NPZ array {key} shape {arrays[key].shape} != {shape}", failures)
            require(bool(np.all(np.isfinite(arrays[key]))), f"{case}: NPZ array {key} contains non-finite values", failures)


def validate_case(root: Path, case: str, failures: list[str]) -> dict[str, Any] | None:
    report_path = root / case / "v18_mano_foundation_state_report.json"
    require(report_path.exists(), f"{case}: report missing", failures)
    if not report_path.exists():
        return None
    report = load_json(report_path)
    exp = EXPECTED[case]
    require(report.get("claim_scope") == "MANO_first_foundation_state_and_validity_gate_not_contact_occlusion_or_object_pose_closure", f"{case}: wrong claim scope", failures)
    require(report.get("foundational_mano_state_valid") is False, f"{case}: unexpectedly marked foundational MANO valid", failures)
    require(report.get("v18_physical_pipeline_valid_without_further_hand_work") is False, f"{case}: physical pipeline should be invalid without further hand work", failures)
    require(report.get("expected_two_hand_rows") == exp["expected_rows"], f"{case}: expected row count changed", failures)
    current = report.get("current_v18_full_mano_storage", {}).get("counts", {}) if isinstance(report.get("current_v18_full_mano_storage"), dict) else {}
    require(int(current.get("surface_candidates_stored_in_v18_full", 0)) == 0, f"{case}: V18 full unexpectedly stores MANO surfaces", failures)
    require(int(current.get("mano_params_stored_in_v18_full", 0)) == 0, f"{case}: V18 full unexpectedly stores MANO params", failures)
    wilor = report.get("recovered_wilor_world_mano_candidates", {}) if isinstance(report.get("recovered_wilor_world_mano_candidates"), dict) else {}
    hawor = report.get("hawor_world_mano_candidates", {}) if isinstance(report.get("hawor_world_mano_candidates"), dict) else {}
    require(int(wilor.get("complete_world_rows", -1)) == exp["wilor_rows"], f"{case}: recovered WiLoR raw world candidate rows mismatch", failures)
    require(int(wilor.get("unique_complete_world_frame_side_rows", -1)) == exp["wilor_unique_frame_side_rows"], f"{case}: recovered WiLoR unique frame-side coverage mismatch", failures)
    require(int(hawor.get("complete_world_surface_param_rows", -1)) == exp["hawor_rows"], f"{case}: HaWoR world rows mismatch", failures)
    require(finite(wilor.get("projection_residual_px_median")) and float(wilor["projection_residual_px_median"]) < 0.01, f"{case}: WiLoR projection residual does not validate camera interpretation", failures)
    blockers = report.get("blocking_reasons") if isinstance(report.get("blocking_reasons"), list) else []
    require("current_v18_full_annotations_drop_mano_vertices" in blockers, f"{case}: missing blocker for dropped MANO vertices", failures)
    require("current_v18_full_annotations_drop_mano_parameters" in blockers, f"{case}: missing blocker for dropped MANO params", failures)
    require("recovered_wilor_mano_not_full_two_hand_timeline" in blockers, f"{case}: missing blocker for incomplete WiLoR timeline", failures)
    if case == "task5_tomato_960":
        require("hawor_missing_for_case" in blockers, f"{case}: missing task5 HaWoR blocker", failures)
    validate_npz(case, report, failures)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600/mano_foundation_audit"))
    args = parser.parse_args()
    failures: list[str] = []
    summary_path = args.root / "v18_mano_foundation_audit_summary.json"
    require(summary_path.exists(), f"summary missing: {summary_path}", failures)
    summary = load_json(summary_path) if summary_path.exists() else {}
    require(summary.get("all_cases_foundational_mano_valid") is False, "summary should mark foundation invalid", failures)
    require(summary.get("v18_physical_pipeline_valid_without_further_hand_work") is False, "summary should mark physical pipeline invalid", failures)
    reports = []
    for case in EXPECTED:
        report = validate_case(args.root, case, failures)
        if report is not None:
            reports.append({"case": case, "foundational_mano_state_valid": report.get("foundational_mano_state_valid"), "blocking_reasons": report.get("blocking_reasons")})
    out = {"method": "validate_v18_mano_foundation_state", "status": "ok" if not failures else "failed", "root": str(args.root), "cases": reports, "failures": failures}
    print(json.dumps(out, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
