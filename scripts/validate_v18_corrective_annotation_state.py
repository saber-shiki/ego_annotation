#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_case(case: str, root: Path, expected_root: Path, failures: list[str]) -> dict[str, Any]:
    path = root / case / "annotations_v18_corrective_state.json"
    require(path.exists(), f"{case}: missing {path}", failures)
    if not path.exists():
        return {"case": case, "status": "missing"}
    ann = load_json(path)
    src_path = expected_root / case / "annotations_v18_full.json"
    require(src_path.exists(), f"{case}: missing source annotation {src_path}", failures)
    src = load_json(src_path) if src_path.exists() else {}
    frames = ann.get("frames", [])
    counts = ann.get("counts", {}) if isinstance(ann.get("counts"), dict) else {}
    expected_count = src.get("frame_count")
    require(isinstance(frames, list), f"{case}: frames is not a list", failures)
    require(len(frames) == expected_count, f"{case}: frame count {len(frames)} != source {expected_count}", failures)
    require(ann.get("status") == "corrective_state_delta_not_full_v18_closure", f"{case}: incorrect scoped status", failures)
    require(int(counts.get("graph_shifted_mano_states", 0)) > 0, f"{case}: no graph-shifted MANO states", failures)
    require(int(counts.get("graph_hand_states", 0)) == int(counts.get("graph_shifted_mano_states", -1)), f"{case}: graph hand count != shifted MANO count", failures)
    require(int(counts.get("graph_object_se3_states", 0)) > 0, f"{case}: no graph object SE3 states", failures)
    require(int(counts.get("frame_local_visible_surface_states", 0)) > 0, f"{case}: no frame-local visible surface states", failures)
    stable_without_uncertainty = 0
    stable_without_residual = 0
    for frame in frames if isinstance(frames, list) else []:
        if not isinstance(frame, dict):
            continue
        for obj in frame.get("objects", []):
            if not isinstance(obj, dict):
                continue
            attempt = obj.get("generic_rigid_se3_attempt", {}) if isinstance(obj.get("generic_rigid_se3_attempt"), dict) else {}
            if attempt.get("stable_pose6_world_from_object") is not None:
                if not obj.get("uncertainty"):
                    stable_without_uncertainty += 1
                if not isinstance(attempt.get("residual_check"), dict):
                    stable_without_residual += 1
    require(stable_without_uncertainty == 0, f"{case}: stable rigid rows without uncertainty: {stable_without_uncertainty}", failures)
    require(stable_without_residual == 0, f"{case}: stable rigid rows without residual check: {stable_without_residual}", failures)
    if case == "trash_1050":
        require(int(counts.get("hawor_prior_states", 0)) == 182, f"{case}: expected 182 HaWoR prior states", failures)
        require(int(counts.get("frame_local_visible_surface_states", 0)) == 232, f"{case}: expected 232 visible surface states for the rigid lid", failures)
        require(int(counts.get("occlusion_owner_best_effort_states", 0)) == 64, f"{case}: expected 64 tentative occlusion owner rows", failures)
        require(ann.get("occlusion_owner_selected_rows") == 64, f"{case}: selected owner row metadata should be 64", failures)
        require(ann.get("contact_graph_selected_rows") == 371, f"{case}: expected 371 selected contact rows", failures)
        require(int(counts.get("contact_nonpenetration_states", 0)) == 371, f"{case}: expected 371 contact/nonpenetration states", failures)
        require(int(counts.get("contact_nonpenetration::graph_accepted_but_local_penetration_veto", 0)) == 293, f"{case}: expected 293 local penetration contact veto states", failures)
        require(int(counts.get("contact_nonpenetration::graph_accepted_no_local_penetration_flag", 0)) == 2, f"{case}: expected 2 graph-accepted contact states without local penetration flag", failures)
        require(int(counts.get("rigid_residual_checked_states", 0)) == 232, f"{case}: expected 232 rigid residual checked states", failures)
        require(int(counts.get("rigid_residual::bidirectional_residual_supported_uncertain", 0)) == 150, f"{case}: expected 150 bidirectional residual supported states", failures)
        require(int(counts.get("rigid_residual::visible_supported_but_fused_overspread", 0)) == 82, f"{case}: expected 82 fused-overspread residual states", failures)
        require("object:pink_lid_trash_can_second" in ann.get("rigid_candidate_ids", []), f"{case}: missing pink lid rigid candidate", failures)
    if case == "task5_tomato_960":
        require(ann.get("hawor_measurement_rows") == 0, f"{case}: expected zero HaWoR measurement rows", failures)
        require(ann.get("occlusion_owner_selected_rows") == 0, f"{case}: expected zero selected tentative occlusion owner rows", failures)
        require(int(counts.get("hawor_provisioning_failed_hand_states", 0)) == 1920, f"{case}: expected 1920 HaWoR provisioning-failure hand states", failures)
        require(int(counts.get("frame_local_visible_surface_states", 0)) == 449, f"{case}: expected 449 visible surface states for rigid candidates", failures)
        require(ann.get("contact_graph_selected_rows") == 808, f"{case}: expected 808 selected contact rows", failures)
        require(int(counts.get("contact_nonpenetration_states", 0)) == 808, f"{case}: expected 808 contact/nonpenetration states", failures)
        require(int(counts.get("contact_nonpenetration::graph_accepted_but_local_penetration_veto", 0)) == 705, f"{case}: expected 705 local penetration contact veto states", failures)
        require(int(counts.get("contact_nonpenetration::graph_accepted_no_local_penetration_flag", 0)) == 16, f"{case}: expected 16 graph-accepted contact states without local penetration flag", failures)
        require(int(counts.get("rigid_residual_checked_states", 0)) == 449, f"{case}: expected 449 rigid residual checked states", failures)
        require(int(counts.get("rigid_residual::bidirectional_residual_supported_uncertain", 0)) == 22, f"{case}: expected 22 bidirectional residual supported states", failures)
        require(int(counts.get("rigid_residual::visible_supported_but_fused_overspread", 0)) == 425, f"{case}: expected 425 fused-overspread residual states", failures)
        require(int(counts.get("rigid_residual::visible_surface_not_explained_by_fused_pose", 0)) == 2, f"{case}: expected 2 residual rejected states", failures)
        require("object:obj_tomato" in ann.get("rigid_candidate_ids", []), f"{case}: missing tomato generic rigid candidate", failures)
        tomato_pose_rows = 0
        tomato_surface_rows = 0
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for obj in frame.get("objects", []):
                if isinstance(obj, dict) and obj.get("object_id") == "object:obj_tomato":
                    attempt = obj.get("generic_rigid_se3_attempt", {}) if isinstance(obj.get("generic_rigid_se3_attempt"), dict) else {}
                    if attempt.get("stable_pose6_world_from_object") is not None:
                        tomato_pose_rows += 1
                    if isinstance(obj.get("frame_local_visible_surface_state"), dict):
                        tomato_surface_rows += 1
        require(tomato_pose_rows > 0, f"{case}: no tomato stable rigid pose rows", failures)
        require(tomato_surface_rows == 447, f"{case}: tomato visible surface rows {tomato_surface_rows} != 447", failures)
    return {"case": case, "frame_count": len(frames), "counts": counts, "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    failures: list[str] = []
    cases = [validate_case(case, args.root, args.source_root, failures) for case in args.cases]
    report = {"status": "ok" if not failures else "failed", "cases": cases, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
