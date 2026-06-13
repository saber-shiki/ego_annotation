#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_trash(case: dict[str, Any], failures: list[str]) -> None:
    require(case.get("status") == "trash_hawor_bridge_candidate_built_not_accepted", f"trash status unexpected: {case.get('status')}", failures)
    require(case.get("accepted_v18_hawor_foundation") is False, "trash bridge must not be accepted foundation", failures)
    require(case.get("downstream_physical_modules_recomputed_from_bridge") is False, "trash downstream recomputation must remain false", failures)
    require(case.get("bridge_candidate_rows") == 2098, f"trash expected 2098 bridge rows, got {case.get('bridge_candidate_rows')}", failures)
    require(case.get("expected_frame_side_rows") == 2100, f"trash expected 2100 frame-side rows, got {case.get('expected_frame_side_rows')}", failures)
    require(case.get("valid_hawor_frame_side_rows") == 2098, f"trash expected 2098 valid rows, got {case.get('valid_hawor_frame_side_rows')}", failures)
    residual = case.get("reference_projection_residual_px_median_per_row") if isinstance(case.get("reference_projection_residual_px_median_per_row"), dict) else {}
    require(residual.get("count") == 1909, f"trash expected 1909 reference residual rows, got {residual.get('count')}", failures)
    med = float(residual.get("median", -1.0)) if residual.get("median") is not None else -1.0
    p95 = float(residual.get("p95", -1.0)) if residual.get("p95") is not None else -1.0
    require(25.0 <= med <= 45.0, f"trash median residual outside expected candidate range: {med}", failures)
    require(p95 > 500.0, f"trash residual tail should block acceptance, p95={p95}", failures)
    thresholds = case.get("reference_projection_residual_threshold_counts") if isinstance(case.get("reference_projection_residual_threshold_counts"), dict) else {}
    require(thresholds.get("median_px_gt_200", 0) > 0, "trash should preserve large-residual rows as blocker evidence", failures)
    tail = case.get("projection_residual_tail_localization") if isinstance(case.get("projection_residual_tail_localization"), dict) else {}
    tail_1000 = tail.get("median_px_gt_1000") if isinstance(tail.get("median_px_gt_1000"), dict) else {}
    require(tail_1000.get("count") == 30, f"trash expected 30 >1000px tail rows, got {tail_1000.get('count')}", failures)
    require(tail_1000.get("frame_min") == 509 and tail_1000.get("frame_max") == 528, f"trash >1000px tail should remain localized to frames 509-528, got {tail_1000.get('frame_min')}-{tail_1000.get('frame_max')}", failures)
    visibility = tail_1000.get("current_visibility_counts") if isinstance(tail_1000.get("current_visibility_counts"), dict) else {}
    require(int(visibility.get("unresolved", 0)) >= 29, f"trash >1000px tail should mostly be unresolved visibility, got {visibility}", failures)
    hawor_inside = tail_1000.get("hawor_projected_inside_image_fraction") if isinstance(tail_1000.get("hawor_projected_inside_image_fraction"), dict) else {}
    require(float(hawor_inside.get("median", 999.0)) == 0.0, f"trash >1000px tail should preserve HaWoR out-of-frame evidence, got {hawor_inside}", failures)
    cam = case.get("camera_trajectory_alignment") if isinstance(case.get("camera_trajectory_alignment"), dict) else {}
    global_err = cam.get("global_sim3", {}).get("error_m", {}) if isinstance(cam.get("global_sim3"), dict) else {}
    require(float(global_err.get("median", 999.0)) > 0.1, f"global Sim3 median unexpectedly small; check acceptance logic: {global_err}", failures)
    blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
    for blocker in [
        "bridge_candidate_not_consumed_by_contact_occlusion_nonpenetration",
        "not_all_cases_have_hawor_bridge_candidates",
        "projection_residual_tail_too_large_for_foundation_acceptance",
        "single_global_HaWoR_to_V18_world_sim3_alignment_too_loose_for_physical_contact",
    ]:
        require(blocker in blockers, f"trash missing blocker {blocker}", failures)
    npz_path = Path(str(case.get("bridge_candidate_npz")))
    require(npz_path.exists(), f"trash bridge NPZ missing: {npz_path}", failures)
    if npz_path.exists():
        z = np.load(npz_path)
        require(z["frame_idx"].shape == (2098,), f"trash frame_idx shape mismatch: {z['frame_idx'].shape}", failures)
        require(z["vertices_hawor_camera_m"].shape == (2098, 778, 3), f"trash camera vertices shape mismatch: {z['vertices_hawor_camera_m'].shape}", failures)
        require(z["joints_hawor_camera_m"].shape == (2098, 21, 3), f"trash camera joints shape mismatch: {z['joints_hawor_camera_m'].shape}", failures)
        require(z["vertices_current_v18_world_from_hawor_camera_local_m"].shape == (2098, 778, 3), f"trash V18-world vertices shape mismatch: {z['vertices_current_v18_world_from_hawor_camera_local_m'].shape}", failures)
        require(z["joints_current_v18_world_from_hawor_camera_local_m"].shape == (2098, 21, 3), f"trash V18-world joints shape mismatch: {z['joints_current_v18_world_from_hawor_camera_local_m'].shape}", failures)
        require(np.isfinite(z["joints_hawor_camera_m"]).all(), "trash camera joints contain non-finite values", failures)
        require(np.all(z["joints_hawor_camera_m"][:, :, 2] > 0.0), "trash camera joints contain non-positive depth", failures)


def validate_task5(case: dict[str, Any], failures: list[str]) -> None:
    require(case.get("accepted_v18_hawor_foundation") is False, "task5 bridge must not be accepted foundation", failures)
    require(case.get("hawor_npz") == "/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/task5_tomato_960/hawor_world_hands.npz", f"task5 bridge should point to contract NPZ path, got {case.get('hawor_npz')}", failures)
    blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
    if case.get("status") == "blocked_no_hawor_npz_for_case":
        require(case.get("bridge_candidate_rows") == 0, f"task5 expected 0 bridge rows, got {case.get('bridge_candidate_rows')}", failures)
        require(case.get("bridge_candidate_npz") is None, f"task5 should not have bridge NPZ while blocked: {case.get('bridge_candidate_npz')}", failures)
        for blocker in ["case_hawor_world_hands_npz_missing", "HaWoR_repo_weights_or_MANO_assets_missing_locally"]:
            require(blocker in blockers, f"task5 missing blocker {blocker}", failures)
    elif case.get("status") == "hawor_bridge_candidate_built_not_accepted":
        require(int(case.get("expected_frame_side_rows") or 0) == 1920, f"task5 expected frame-side rows 1920, got {case.get('expected_frame_side_rows')}", failures)
        require(int(case.get("bridge_candidate_rows") or 0) <= 1920, f"task5 bridge rows cannot exceed 1920, got {case.get('bridge_candidate_rows')}", failures)
        require(case.get("bridge_candidate_npz") is not None, "task5 present bridge should record bridge NPZ", failures)
        require("bridge_candidate_not_consumed_by_contact_occlusion_nonpenetration" in blockers, "task5 present bridge must keep downstream-not-consumed blocker", failures)
    else:
        require(False, f"task5 status unexpected: {case.get('status')}", failures)


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    path = args.root / "hawor_bridge_state" / "v18_hawor_bridge_state_summary.json"
    require(path.exists(), f"missing summary {path}", failures)
    summary = load_json(path) if path.exists() else {}
    require(summary.get("status") in {"trash_bridge_candidate_built_task5_blocked_not_v18_foundation", "hawor_bridge_candidates_built_not_v18_foundation"}, f"summary status unexpected: {summary.get('status')}", failures)
    require(summary.get("all_cases_hawor_bridge_accepted") is False, "all_cases_hawor_bridge_accepted must be false", failures)
    require(summary.get("v18_physical_hand_state_valid_from_bridge") is False, "physical hand state from bridge must be false", failures)
    require(summary.get("claim_scope") == "HaWoR_bridge_candidate_state_no_model_substitution_no_full_V18_closure", f"summary claim scope unexpected: {summary.get('claim_scope')}", failures)
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    by_case = {case.get("case"): case for case in cases if isinstance(case, dict)}
    require(set(by_case) == {"trash_1050", "task5_tomato_960"}, f"unexpected cases {sorted(by_case)}", failures)
    if "trash_1050" in by_case:
        validate_trash(by_case["trash_1050"], failures)
    if "task5_tomato_960" in by_case:
        validate_task5(by_case["task5_tomato_960"], failures)
    return {
        "method": "validate_v18_hawor_bridge_state",
        "status": "ok" if not failures else "failed",
        "root": str(args.root),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
