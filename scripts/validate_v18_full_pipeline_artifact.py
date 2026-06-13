#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def validate_case(case_report: dict[str, Any], require_contact_owner: bool) -> dict[str, Any]:
    case = str(case_report.get("case"))
    require(case_report.get("frame_count_match") is True, f"{case}: frame counts do not match")
    expected = int(case_report.get("expected_frame_count", -1))
    require(expected > 0, f"{case}: missing expected frame count")
    require(int(case_report.get("overlay_frame_count", -1)) == expected, f"{case}: overlay frame count mismatch")
    require(int(case_report.get("world_frame_count", -1)) == expected, f"{case}: world frame count mismatch")
    require(int(case_report.get("side_by_side_frame_count", -1)) == expected, f"{case}: side-by-side frame count mismatch")
    monotonicity_raw = case_report.get("monotonicity")
    monotonicity: dict[str, Any] = monotonicity_raw if isinstance(monotonicity_raw, dict) else {}
    require(monotonicity.get("preserves_v16_overlay_mano_object_render") is True, f"{case}: V16 overlay not preserved")
    require(monotonicity.get("preserves_v16_metric_world_render") is True, f"{case}: V16 world render not preserved")
    for key in ["annotations", "overlay_video", "world_video", "side_by_side_video", "base_v16_overlay", "base_v16_world"]:
        path = Path(str(case_report.get(key)))
        require(path.exists(), f"{case}: missing {key}: {path}")
    ann = load_json(Path(str(case_report.get("annotations"))))
    frames = ann.get("frames")
    require(isinstance(frames, list) and len(frames) == expected, f"{case}: annotation frame count mismatch")
    modules_raw = ann.get("modules")
    modules: dict[str, Any] = modules_raw if isinstance(modules_raw, dict) else {}
    require("depth_scale_correction" in str(modules.get("camera_depth_backbone")), f"{case}: camera/depth correction not listed in modules")
    require("contact_owner_graph" in str(modules.get("contact_ownership")) or "contact_owner" in str(modules.get("contact_ownership")), f"{case}: contact owner graph not listed in modules")
    require("signed_normal" in str(modules.get("contact_ownership")), f"{case}: signed nonpenetration evidence not listed in modules")
    require("triangle_nonpenetration" in str(modules.get("contact_ownership")), f"{case}: triangle nonpenetration evidence not listed in modules")
    require("hand_baseline_evidence" in str(modules.get("hand_branch")), f"{case}: hand baseline evidence not listed in modules")
    require("pose_fill_gate" in str(modules.get("hand_branch")), f"{case}: pose fill gate not listed in hand module")
    require("temporal_occlusion_owner_graph" in str(modules.get("occlusion_ownership")), f"{case}: temporal occlusion owner graph not listed in modules")
    accepted_contact = 0
    selected_contact = 0
    occlusion_mesh_rows = 0
    factor_contact_accept = 0
    hand_baseline_rows = 0
    camera_depth_observed_rows = 0
    signed_nonpenetration_rows = 0
    triangle_nonpenetration_rows = 0
    occlusion_temporal_graph_rows = 0
    pose_fill_gate_rows = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for hyp in frame.get("contact_hypotheses", []):
            if not isinstance(hyp, dict):
                continue
            evidence_raw = hyp.get("evidence")
            evidence: dict[str, Any] = evidence_raw if isinstance(evidence_raw, dict) else {}
            graph_raw = evidence.get("contact_ownership_graph")
            graph: dict[str, Any] | None = graph_raw if isinstance(graph_raw, dict) else None
            signed_raw = evidence.get("signed_nonpenetration_evidence")
            signed: dict[str, Any] | None = signed_raw if isinstance(signed_raw, dict) else None
            if signed is not None:
                signed_nonpenetration_rows += 1
                require(signed.get("signed_nonpenetration_complete") is False, f"{case}: signed evidence overclaims complete nonpenetration")
            triangle_raw = evidence.get("triangle_nonpenetration_evidence")
            triangle: dict[str, Any] | None = triangle_raw if isinstance(triangle_raw, dict) else None
            if triangle is not None:
                triangle_nonpenetration_rows += 1
                require(triangle.get("triangle_nonpenetration_complete") is False, f"{case}: triangle evidence overclaims complete nonpenetration")
                require(triangle.get("mesh_watertight_by_edges") is not True, f"{case}: unexpected watertight triangle evidence needs review")
            if graph:
                if graph.get("selected_by_contact_graph") is True:
                    selected_contact += 1
                if hyp.get("contact_owner_hypothesis") == "accepted_contact_owner_by_temporal_mesh_distance_graph":
                    require(signed is None or signed.get("local_penetration_detected") is not True, f"{case}: accepted contact owner contradicted by signed penetration")
                    require(triangle is None or triangle.get("local_triangle_penetration_detected") is not True, f"{case}: accepted contact owner contradicted by triangle penetration")
                    accepted_contact += 1
                elif graph.get("accepted_contact_owner") is True:
                    require(hyp.get("contact_owner_hypothesis") == "contact_owner_graph_conflicted_by_local_nonpenetration_evidence_not_accepted", f"{case}: graph accepted row must be accepted or explicitly nonpenetration-conflicted")
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            baseline_raw = hand.get("hand_baseline_branch")
            baseline: dict[str, Any] = baseline_raw if isinstance(baseline_raw, dict) else {}
            if baseline.get("hand_baseline_state"):
                hand_baseline_rows += 1
            require(baseline.get("temporal_occlusion_pose_accepted") is not True, f"{case}: unsupported accepted occlusion hand pose")
            pose_gate_raw = hand.get("occlusion_pose_fill_gate")
            pose_gate: dict[str, Any] = pose_gate_raw if isinstance(pose_gate_raw, dict) else {}
            if pose_gate:
                pose_fill_gate_rows += 1
                require(pose_gate.get("pose_fill_through_occlusion_accepted") is not True, f"{case}: unsupported accepted pose fill-through-occlusion")
                blockers_raw = pose_gate.get("blockers")
                require(isinstance(blockers_raw, list) and len(blockers_raw) > 0, f"{case}: blocked pose fill lacks blockers")
            occ_raw = hand.get("occlusion_owner_hypothesis")
            occ: dict[str, Any] = occ_raw if isinstance(occ_raw, dict) else {}
            occ_evidence = occ.get("mesh_owner_evidence")
            if isinstance(occ_evidence, list) and len(occ_evidence) > 0:
                occlusion_mesh_rows += 1
            temporal_occ = occ.get("temporal_owner_graph")
            if isinstance(temporal_occ, dict):
                occlusion_temporal_graph_rows += 1
                gate_raw = temporal_occ.get("acceptance_gate")
                gate: dict[str, Any] = gate_raw if isinstance(gate_raw, dict) else {}
                blockers_raw = temporal_occ.get("acceptance_blockers")
                blockers = blockers_raw if isinstance(blockers_raw, list) else gate.get("acceptance_blockers")
                if temporal_occ.get("accepted_occlusion_owner") is True:
                    require(gate.get("accepted_by_strict_depth_mesh_temporal_gate") is True, f"{case}: accepted temporal occlusion owner failed strict gate")
                    require(isinstance(blockers, list) and len(blockers) == 0, f"{case}: accepted temporal occlusion owner has blockers")
                elif gate:
                    require(isinstance(blockers, list) and len(blockers) > 0, f"{case}: nonaccepted temporal occlusion owner lacks blockers")
        fg_raw = frame.get("factor_graph_solution")
        fg: dict[str, Any] = fg_raw if isinstance(fg_raw, dict) else {}
        fg_variables_raw = fg.get("variables")
        fg_variables: dict[str, Any] = fg_variables_raw if isinstance(fg_variables_raw, dict) else {}
        camera_depth = fg_variables.get("camera_depth_correction") if isinstance(fg_variables.get("camera_depth_correction"), dict) else {}
        if isinstance(camera_depth, dict) and camera_depth.get("has_direct_observation") is True:
            camera_depth_observed_rows += 1
        contact_switch_raw = fg_variables.get("contact_switch")
        if isinstance(contact_switch_raw, list):
            for variable_raw in contact_switch_raw:
                variable: dict[str, Any] = variable_raw if isinstance(variable_raw, dict) else {}
                signed_var_conflict = variable.get("signed_nonpenetration_conflict") is True
                triangle_var_conflict = variable.get("triangle_nonpenetration_conflict") is True
                union_var_conflict = variable.get("nonpenetration_conflict") is True
                require(union_var_conflict == bool(signed_var_conflict or triangle_var_conflict), f"{case}: nonpenetration conflict union inconsistent")
                if variable.get("estimate") is True:
                    require(not union_var_conflict, f"{case}: factor graph active contact despite nonpenetration conflict")
                    factor_contact_accept += 1
    if require_contact_owner:
        require(accepted_contact > 0, f"{case}: no accepted contact owner rows in final annotations")
        require(selected_contact >= accepted_contact, f"{case}: selected contact count less than accepted count")
    require(occlusion_mesh_rows > 0, f"{case}: no occlusion mesh evidence integrated")
    require(factor_contact_accept > 0, f"{case}: factor graph contact switches absent")
    require(hand_baseline_rows > 0, f"{case}: no hand baseline rows integrated")
    require(camera_depth_observed_rows > 0, f"{case}: no observed camera/depth correction rows integrated")
    require(signed_nonpenetration_rows > 0, f"{case}: no signed nonpenetration evidence integrated")
    require(triangle_nonpenetration_rows > 0, f"{case}: no triangle nonpenetration evidence integrated")
    require(occlusion_temporal_graph_rows > 0, f"{case}: no temporal occlusion owner graph rows integrated")
    require(pose_fill_gate_rows == expected * 2, f"{case}: pose fill gate rows do not cover both hands/full timeline")
    return {"case": case, "expected_frame_count": expected, "accepted_contact_owner_rows": accepted_contact, "selected_contact_owner_rows": selected_contact, "occlusion_mesh_evidence_frames": occlusion_mesh_rows, "active_factor_contact_switch_sum": factor_contact_accept, "hand_baseline_rows": hand_baseline_rows, "camera_depth_observed_rows": camera_depth_observed_rows, "signed_nonpenetration_rows": signed_nonpenetration_rows, "triangle_nonpenetration_rows": triangle_nonpenetration_rows, "occlusion_temporal_graph_rows": occlusion_temporal_graph_rows, "pose_fill_gate_rows": pose_fill_gate_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline/v18_full_pipeline_report.json"))
    parser.add_argument("--require-contact-owner", action="store_true", default=True)
    args = parser.parse_args()
    report = load_json(args.report)
    require(report.get("all_frame_counts_match") is True, "global frame count mismatch")
    cases = report.get("cases")
    require(isinstance(cases, list) and len(cases) > 0, "report has no cases")
    rows = [validate_case(case_report, args.require_contact_owner) for case_report in cases if isinstance(case_report, dict)]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
