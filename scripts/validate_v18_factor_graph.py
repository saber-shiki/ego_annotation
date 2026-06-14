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


def validate_case(path: Path) -> dict[str, Any]:
    ann = load_json(path)
    case = ann.get("case")
    frames = ann.get("frames")
    require(isinstance(frames, list) and len(frames) > 0, f"{case}: frames missing")
    fg = ann.get("factor_graph_summary")
    require(isinstance(fg, dict), f"{case}: factor_graph_summary missing")
    objective = fg.get("objective")
    require(isinstance(objective, dict), f"{case}: objective missing")
    energy_initial = float(objective.get("energy_initial"))
    energy_after = float(objective.get("energy_after"))
    require(energy_after <= energy_initial + 1e-6, f"{case}: graph energy did not decrease")
    variable_counts = fg.get("variable_counts")
    factor_counts = fg.get("factor_counts")
    require(isinstance(variable_counts, dict) and isinstance(factor_counts, dict), f"{case}: counts missing")
    for key in ["camera_depth_correction", "hand_state", "object_se3", "part_se3", "contact_switch", "occlusion_owner"]:
        require(int(variable_counts.get(key, 0)) > 0, f"{case}: missing {key} variables")
    for key in ["camera_depth_correction_observation", "hand_state_observation", "object_se3_observation", "part_se3_observation", "contact_switch_discrete", "contact_switch_temporal", "contact_local_nonpenetration", "occlusion_owner_discrete"]:
        require(int(factor_counts.get(key, 0)) > 0, f"{case}: missing {key} factors")
    implemented_status = fg.get("implemented_variable_status")
    spec_gaps = fg.get("spec_factor_gaps_remaining")
    require(isinstance(implemented_status, dict), f"{case}: implemented variable status missing")
    require("camera_depth_correction" in implemented_status and "observed_depth_scale_correction" in str(implemented_status.get("camera_depth_correction")), f"{case}: camera/depth correction observation status not explicit")
    require("part_se3" in implemented_status and "pca_rotvec" in str(implemented_status.get("part_se3")), f"{case}: part SE3 PCA status not explicit")
    require(isinstance(spec_gaps, list) and len(spec_gaps) > 0, f"{case}: spec factor gaps missing")
    require(any("camera_depth_correction_is_scale_only" in str(gap) for gap in spec_gaps), f"{case}: camera/depth correction limitation not explicit")
    require(any("visible_surface_PCA" in str(gap) for gap in spec_gaps), f"{case}: visible-surface part SE3 limitation not explicit")
    implemented_families = fg.get("implemented_factor_families")
    require(isinstance(implemented_families, list) and any("contact_switch_temporal" in str(item) for item in implemented_families), f"{case}: contact temporal factor family missing")
    require(isinstance(implemented_families, list) and any("contact_local_nonpenetration" in str(item) for item in implemented_families), f"{case}: contact local nonpenetration factor family missing")
    require(isinstance(implemented_families, list) and any("contact_part_pose_anchor" in str(item) for item in implemented_families), f"{case}: contact-part pose anchor factor family missing")
    require(int(factor_counts.get("contact_part_pose_anchor", 0)) > 0, f"{case}: contact-part pose anchor factors missing")
    if case == "task5_tomato_960":
        require(int(factor_counts.get("contact_object_pose_anchor", 0)) > 0, f"{case}: contact-object pose anchor factors missing")
    inference = fg.get("inference")
    require(isinstance(inference, dict), f"{case}: inference missing")
    require("SciPy" in str(inference.get("continuous_method")), f"{case}: continuous solve is not SciPy-backed")
    require("viterbi" in str(inference.get("discrete_method")).lower(), f"{case}: contact switch inference is not temporal Viterbi")
    series = inference.get("series_summaries")
    require(isinstance(series, dict) and len(series) > 0, f"{case}: series summaries missing")
    object_se3_series = {k: v for k, v in series.items() if str(k).startswith("object_se3::") and isinstance(v, dict)}
    part_se3_series = {k: v for k, v in series.items() if str(k).startswith("part_se3::") and isinstance(v, dict)}
    require(len(object_se3_series) > 0, f"{case}: object SE3 series missing")
    require(len(part_se3_series) > 0, f"{case}: part SE3 series missing")
    object_6d_count = sum(1 for v in object_se3_series.values() if int(v.get("dimension", 0)) == 6)
    part_6d_count = sum(1 for v in part_se3_series.values() if int(v.get("dimension", 0)) == 6)
    require(object_6d_count > 0, f"{case}: no 6D object SE3 series")
    require(part_6d_count > 0, f"{case}: no 6D part SE3 series")
    frame_with_graph = 0
    temporal_contact_rows = 0
    temporal_contact_factor_rows = 0
    temporal_contact_active_conflicts = 0
    temporal_contact_bad_gaps = 0
    local_temporal_factor_count_sum = 0
    local_nonpenetration_factor_count_sum = 0
    contact_nonpenetration_factor_rows = 0
    contact_part_component_rows = 0
    occlusion_owner_rows = 0
    occlusion_owner_with_temporal_or_mesh = 0
    accepted_occlusion_owner_rows = 0
    local_occlusion_factor_count_sum = 0
    for frame in frames:
        g = frame.get("factor_graph_solution")
        if isinstance(g, dict) and isinstance(g.get("variables"), dict) and isinstance(g.get("objective"), dict):
            frame_with_graph += 1
            factors_raw = g.get("factors")
            factors: dict[str, Any] = factors_raw if isinstance(factors_raw, dict) else {}
            local_temporal_factor_count_sum += int(factors.get("contact_switch_temporal", 0))
            local_nonpenetration_factor_count_sum += int(factors.get("contact_local_nonpenetration", 0))
            local_occlusion_factor_count_sum += int(factors.get("occlusion_owner_discrete", 0))
            variables_raw = g.get("variables")
            variables: dict[str, Any] = variables_raw if isinstance(variables_raw, dict) else {}
            occlusion_raw = variables.get("occlusion_owner")
            if isinstance(occlusion_raw, list):
                for occ_raw in occlusion_raw:
                    occ: dict[str, Any] = occ_raw if isinstance(occ_raw, dict) else {}
                    occlusion_owner_rows += 1
                    require(str(occ.get("inference_method")) in {"box_mesh_depth_temporal_energy_with_unowned_competitor", "box_mesh_depth_temporal_energy_with_unowned_competitor_support_gated_by_hawor_observation"}, f"{case}: occlusion owner lacks integrated inference method")
                    if occ.get("accepted_owner") is True:
                        accepted_occlusion_owner_rows += 1
                    candidates_raw = occ.get("candidate_energies")
                    candidates: list[Any] = candidates_raw if isinstance(candidates_raw, list) else []
                    require(len(candidates) > 0, f"{case}: occlusion owner candidates missing")
                    unowned_count = 0
                    for cand_raw in candidates:
                        cand: dict[str, Any] = cand_raw if isinstance(cand_raw, dict) else {}
                        if cand.get("object_id") is None:
                            unowned_count += 1
                        else:
                            require("mesh_temporal_support" in cand and "temporal_graph_selected" in cand and "depth_evidence_state" in cand, f"{case}: occlusion candidate missing mesh/temporal/depth fields")
                            depth_state = str(cand.get("depth_evidence_state"))
                            expected_support = "foreground" in depth_state and "support" in depth_state and "no_support" not in depth_state and "contradict" not in depth_state
                            expected_contradiction = "foreground" in depth_state and "contradict" in depth_state
                            require(cand.get("foreground_depth_support") is expected_support, f"{case}: foreground support flag mismatch")
                            require(cand.get("foreground_depth_contradiction") is expected_contradiction, f"{case}: foreground contradiction flag mismatch")
                            if cand.get("temporal_graph_selected") is True or float(cand.get("mesh_temporal_support", 0.0)) > 0.0:
                                occlusion_owner_with_temporal_or_mesh += 1
                    require(unowned_count == 1, f"{case}: occlusion owner missing exactly one unowned competitor")
                    if occ.get("accepted_owner") is True:
                        chosen = occ.get("chosen_owner_object_id")
                        chosen_candidates = [cand for cand in candidates if isinstance(cand, dict) and cand.get("object_id") == chosen]
                        require(any(cand.get("accepted_by_depth_evidence") is True or cand.get("temporal_graph_accepted") is True for cand in chosen_candidates), f"{case}: accepted occlusion owner lacks source support")
            part_raw = variables.get("part_se3")
            if isinstance(part_raw, list):
                for part_var_raw in part_raw:
                    part_var: dict[str, Any] = part_var_raw if isinstance(part_var_raw, dict) else {}
                    components = part_var.get("contact_part_coupling_components")
                    if isinstance(components, list) and components:
                        contact_part_component_rows += len(components)
                        require(any(isinstance(comp, dict) and comp.get("factor_family") == "contact_part_pose_anchor" for comp in components), f"{case}: part contact component missing factor family")
            contact_raw = variables.get("contact_switch")
            if isinstance(contact_raw, list):
                for row_raw in contact_raw:
                    row: dict[str, Any] = row_raw if isinstance(row_raw, dict) else {}
                    require(row.get("temporal_inference_method") == "gap_aware_binary_viterbi_contact_switch", f"{case}: contact switch lacks temporal inference method")
                    temporal_contact_rows += 1
                    signed_conflict = row.get("signed_nonpenetration_conflict") is True
                    triangle_conflict = row.get("triangle_nonpenetration_conflict") is True
                    union_conflict = row.get("nonpenetration_conflict") is True
                    require(union_conflict == bool(signed_conflict or triangle_conflict), f"{case}: contact nonpenetration union inconsistent")
                    if row.get("local_nonpenetration_factor_present") is True:
                        contact_nonpenetration_factor_rows += 1
                        require(row.get("local_nonpenetration_factor_complete") is False, f"{case}: local nonpenetration factor overclaims completeness")
                        require("not_watertight_sdf" in str(row.get("local_nonpenetration_factor_scope")), f"{case}: local nonpenetration factor scope missing")
                        require(row.get("signed_local_nonpenetration_factor_present") is True or row.get("triangle_local_nonpenetration_factor_present") is True, f"{case}: local nonpenetration factor lacks evidence source")
                    if row.get("estimate") is True and union_conflict:
                        temporal_contact_active_conflicts += 1
                    if row.get("estimate") is True:
                        require(row.get("geometry_contact_evidence_available") is True, f"{case}: active contact lacks geometry evidence")
                        effective_distance = row.get("effective_metric_contact_distance_m")
                        mesh_support = float(row.get("mesh_contact_support_score", 0.0) or 0.0)
                        near_effective = isinstance(effective_distance, (int, float)) and float(effective_distance) <= 0.20
                        require(near_effective or mesh_support > 0.5, f"{case}: active contact lacks near metric distance or strong mesh support")
                    gap = row.get("temporal_contact_previous_frame_gap")
                    has_factor = row.get("temporal_contact_has_factor") is True
                    applied = row.get("temporal_contact_transition_applied") is True
                    require(has_factor == applied, f"{case}: contact temporal factor/applied mismatch")
                    if has_factor:
                        temporal_contact_factor_rows += 1
                        require(isinstance(gap, int) and gap <= int(row.get("temporal_contact_max_gap_frames", 30)), f"{case}: contact temporal factor across invalid gap")
                    elif isinstance(gap, int) and gap <= int(row.get("temporal_contact_max_gap_frames", 30)) and gap > 0:
                        temporal_contact_bad_gaps += 1
    require(frame_with_graph == len(frames), f"{case}: not every frame has graph solution")
    require(occlusion_owner_rows == int(variable_counts.get("occlusion_owner", -1)), f"{case}: occlusion owner variable count mismatch")
    require(local_occlusion_factor_count_sum == occlusion_owner_rows and occlusion_owner_rows == int(factor_counts.get("occlusion_owner_discrete", -1)), f"{case}: occlusion owner factor count mismatch")
    require(occlusion_owner_with_temporal_or_mesh > 0, f"{case}: occlusion owner variables missing temporal/mesh evidence")
    require(temporal_contact_rows == int(variable_counts.get("contact_switch", -1)), f"{case}: contact switch variable count mismatch")
    require(temporal_contact_factor_rows == int(factor_counts.get("contact_switch_temporal", -1)), f"{case}: temporal contact factor count mismatch")
    require(contact_nonpenetration_factor_rows == int(factor_counts.get("contact_local_nonpenetration", -1)), f"{case}: contact local nonpenetration factor count mismatch")
    require(contact_part_component_rows == int(factor_counts.get("contact_part_pose_anchor", -1)), f"{case}: contact-part component count mismatch")
    require(local_nonpenetration_factor_count_sum == contact_nonpenetration_factor_rows, f"{case}: local nonpenetration factor sum mismatch")
    require(local_temporal_factor_count_sum == temporal_contact_factor_rows, f"{case}: local temporal contact factor sum mismatch")
    require(temporal_contact_active_conflicts == 0, f"{case}: active temporal contact has nonpenetration conflict")
    require(temporal_contact_bad_gaps == 0, f"{case}: missing temporal contact factor for valid adjacent gap")
    return {
        "case": case,
        "frame_count": len(frames),
        "energy_initial": energy_initial,
        "energy_after": energy_after,
        "energy_delta": energy_initial - energy_after,
        "variable_counts": variable_counts,
        "factor_counts": factor_counts,
        "object_6d_series_count": object_6d_count,
        "part_6d_series_count": part_6d_count,
        "contact_switch_temporal_factors": int(factor_counts.get("contact_switch_temporal", 0)),
        "contact_switch_temporal_rows": temporal_contact_rows,
        "contact_local_nonpenetration_factors": int(factor_counts.get("contact_local_nonpenetration", 0)),
        "occlusion_owner_rows": occlusion_owner_rows,
        "frame_with_graph_count": frame_with_graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        rows.append(validate_case(args.root / case / "annotations_v18_full.json"))
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
