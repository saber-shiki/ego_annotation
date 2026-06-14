#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


FORBIDDEN_FINAL_STRINGS = [
    "not_accepted",
    "not accepted",
    "not_complete",
    "not complete",
    "unaccepted",
    "verification status",
    "available_partial_score_2d_terms_only",
    "partial_score",
    "candidate-only",
    "candidate_only",
    "object_pose_candidate",
    "acceptance",
    "accepted",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def ffprobe_frame_count(path: Path) -> tuple[int, str, float]:
    data = json.loads(subprocess.check_output([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]))
    stream = data["streams"][0]
    return int(stream["nb_read_frames"]), str(stream.get("r_frame_rate")), float(stream.get("duration", 0.0))


def serialized_contains_forbidden(report_text: str, ann_text: str) -> list[str]:
    combined = f"{report_text}\n{ann_text}".lower()
    return [term for term in FORBIDDEN_FINAL_STRINGS if term.lower() in combined]


def validate_case(case_report: dict[str, Any], report_text: str) -> dict[str, Any]:
    case = str(case_report.get("case"))
    expected = int(case_report.get("expected_frame_count", -1))
    fps = float(case_report.get("fps", 0.0))
    require(expected > 0, f"{case}: missing expected frame count")
    require(fps > 0, f"{case}: missing fps")
    require(case_report.get("frame_count_match") is True, f"{case}: report says frame counts do not match")
    require(int(case_report.get("overlay_frame_count", -1)) == expected, f"{case}: overlay report count mismatch")
    require(int(case_report.get("world_frame_count", -1)) == expected, f"{case}: world report count mismatch")
    require(int(case_report.get("side_by_side_frame_count", -1)) == expected, f"{case}: side-by-side report count mismatch")

    monotonicity_raw = case_report.get("monotonicity")
    monotonicity: dict[str, Any] = monotonicity_raw if isinstance(monotonicity_raw, dict) else {}
    require(monotonicity.get("preserves_v16_overlay_mano_object_render") is True, f"{case}: V16 overlay not preserved")
    require(monotonicity.get("preserves_v16_metric_world_render") is True, f"{case}: V16 world render not preserved")
    require(monotonicity.get("v18_additions_are_overlay_layers") is True, f"{case}: V18 additions not marked as additive")

    for key in ["annotations", "overlay_video", "world_video", "side_by_side_video", "base_v16_overlay", "base_v16_world"]:
        path = Path(str(case_report.get(key)))
        require(path.exists(), f"{case}: missing {key}: {path}")

    for key in ["overlay_video", "world_video", "side_by_side_video"]:
        count, rate, duration = ffprobe_frame_count(Path(str(case_report[key])))
        require(count == expected, f"{case}: ffprobe {key} frame count {count} != {expected}")
        require(abs(duration - expected / fps) < 0.05, f"{case}: ffprobe {key} duration {duration} inconsistent with expected")
    overlay_draw = case_report.get("overlay_draw_counts") if isinstance(case_report.get("overlay_draw_counts"), dict) else {}
    world_draw = case_report.get("world_draw_counts") if isinstance(case_report.get("world_draw_counts"), dict) else {}
    overlay_occ = int(overlay_draw.get("occlusion_owner_edges", 0)) + int(overlay_draw.get("occlusion_unowned_or_unresolved_labels", 0))
    world_occ = int(world_draw.get("world_occlusion_owner_edges", 0)) + int(world_draw.get("world_occlusion_unowned_or_unresolved_labels", 0))
    require(overlay_occ > 0, f"{case}: overlay rendered no occlusion-owner evidence")
    require(world_occ > 0, f"{case}: world render drew no occlusion-owner evidence")
    require(int(overlay_draw.get("pose_fill_gate_markers", 0)) > 0, f"{case}: overlay rendered no pose-fill gate markers")
    require(int(world_draw.get("world_pose_fill_gate_markers", 0)) > 0, f"{case}: world render drew no pose-fill gate markers")

    ann_path = Path(str(case_report.get("annotations")))
    ann_text = ann_path.read_text(encoding="utf-8")
    forbidden = serialized_contains_forbidden(report_text, ann_text)
    require(not forbidden, f"{case}: forbidden final-artifact wording present: {forbidden}")
    ann = json.loads(ann_text)
    frames = ann.get("frames")
    require(isinstance(frames, list) and len(frames) == expected, f"{case}: annotation frame count mismatch")

    modules_raw = ann.get("modules")
    modules: dict[str, Any] = modules_raw if isinstance(modules_raw, dict) else {}
    module_text = json.dumps(modules)
    for needle, label in [
        ("depth_scale_correction", "camera/depth correction"),
        ("HaWoR_metric_MANO", "HaWoR metric MANO"),
        ("WiLoR", "WiLoR hand evidence"),
        ("RTMLib", "RTMLib hand evidence"),
        ("hand_baseline_evidence", "hand baseline evidence"),
        ("pose_fill_gate", "pose fill gate"),
        ("VLM_OWLv2_SAM2", "VLM/OWLv2/SAM2 perception"),
        ("depth_visible_surface", "depth visible geometry"),
        ("object_part_SE3", "object/part SE3"),
        ("contact_owner_graph", "contact-owner graph"),
        ("signed_normal_nonpenetration", "signed normal nonpenetration"),
        ("triangle_nonpenetration", "triangle nonpenetration"),
        ("temporal_occlusion_owner_graph", "temporal occlusion owner graph"),
        ("factor_graph", "factor graph"),
    ]:
        require(needle in module_text, f"{case}: {label} not listed in modules")

    counts = {
        "hand_total": 0,
        "hawor_metric_mano": 0,
        "wilor_key_rows": 0,
        "rtmlib_key_rows": 0,
        "hand_graph_metric": 0,
        "pose_fill_gate_rows": 0,
        "object_states": 0,
        "object_physical_state_rows": 0,
        "object_se3_rows": 0,
        "object_visible_geometry_rows": 0,
        "object_hidden_or_unresolved_geometry_rows": 0,
        "object_vertex_sample_rows": 0,
        "part_rows": 0,
        "contacts": 0,
        "contacts_with_final_metric_distance": 0,
        "signed_nonpenetration_rows": 0,
        "triangle_nonpenetration_rows": 0,
        "contact_switch_vars": 0,
        "active_contact_switch_vars": 0,
        "occlusion_owner_vars": 0,
        "camera_depth_observed_rows": 0,
        "factor_frames": 0,
    }

    for frame in frames:
        require(isinstance(frame, dict), f"{case}: non-dict frame row")
        fg_raw = frame.get("factor_graph_solution")
        fg: dict[str, Any] = fg_raw if isinstance(fg_raw, dict) else {}
        vars_raw = fg.get("variables")
        vars: dict[str, Any] = vars_raw if isinstance(vars_raw, dict) else {}
        if vars:
            counts["factor_frames"] += 1
        camera_depth = vars.get("camera_depth_correction") if isinstance(vars.get("camera_depth_correction"), dict) else {}
        if camera_depth.get("has_direct_observation") is True:
            counts["camera_depth_observed_rows"] += 1
        hand_vars = vars.get("hand_state") if isinstance(vars.get("hand_state"), list) else []
        counts["hand_graph_metric"] += sum(1 for row in hand_vars if isinstance(row, dict) and row.get("source") == "HaWoR_metric_MANO_wrist_current_V18_world_m")
        contact_vars = vars.get("contact_switch") if isinstance(vars.get("contact_switch"), list) else []
        counts["contact_switch_vars"] += len(contact_vars)
        counts["active_contact_switch_vars"] += sum(1 for row in contact_vars if isinstance(row, dict) and row.get("estimate") is True)
        counts["occlusion_owner_vars"] += len(vars.get("occlusion_owner") if isinstance(vars.get("occlusion_owner"), list) else [])

        hands = frame.get("hands") if isinstance(frame.get("hands"), list) else []
        require(len(hands) == 2, f"{case}: frame {frame.get('frame_idx')} does not have two hand rows")
        for hand in hands:
            require(isinstance(hand, dict), f"{case}: non-dict hand row")
            counts["hand_total"] += 1
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            if hand.get("hand_geometry_source") == "HaWoR_metric_MANO_current_V18_world" or str(metric.get("source", "")).startswith("HaWoR_metric_MANO"):
                counts["hawor_metric_mano"] += 1
            if "wilor_or_v16_candidate_present" in hand:
                counts["wilor_key_rows"] += 1
            if "rtmlib_anchor_available" in hand:
                counts["rtmlib_key_rows"] += 1
            pose_gate = hand.get("occlusion_pose_fill_gate") if isinstance(hand.get("occlusion_pose_fill_gate"), dict) else {}
            if pose_gate:
                counts["pose_fill_gate_rows"] += 1
            require(isinstance(hand.get("occlusion_owner_hypothesis"), dict), f"{case}: missing hand occlusion owner hypothesis")

        objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []
        require(objects, f"{case}: frame {frame.get('frame_idx')} has no object rows")
        for obj in objects:
            require(isinstance(obj, dict), f"{case}: non-dict object row")
            counts["object_states"] += 1
            if isinstance(obj.get("physical_state_decision"), dict) and obj.get("physical_state_decision", {}).get("decision"):
                counts["object_physical_state_rows"] += 1
            if isinstance(obj.get("object_se3_observation"), dict):
                counts["object_se3_rows"] += 1
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            if geom:
                counts["object_visible_geometry_rows"] += 1
                if isinstance(geom.get("world_vertices_sample_m"), list) and geom.get("world_vertices_sample_m"):
                    counts["object_vertex_sample_rows"] += 1
            hidden = obj.get("hidden_geometry_candidate")
            if hidden is not None:
                counts["object_hidden_or_unresolved_geometry_rows"] += 1
            counts["part_rows"] += len(obj.get("parts") if isinstance(obj.get("parts"), list) else [])

        for hyp in frame.get("contact_hypotheses", []) if isinstance(frame.get("contact_hypotheses"), list) else []:
            if not isinstance(hyp, dict):
                continue
            counts["contacts"] += 1
            if isinstance(hyp.get("final_metric_contact_evidence"), dict):
                counts["contacts_with_final_metric_distance"] += 1
            evidence = hyp.get("evidence") if isinstance(hyp.get("evidence"), dict) else {}
            if isinstance(evidence.get("signed_nonpenetration_evidence"), dict):
                counts["signed_nonpenetration_rows"] += 1
            if isinstance(evidence.get("triangle_nonpenetration_evidence"), dict):
                counts["triangle_nonpenetration_rows"] += 1

    expected_hand_rows = expected * 2
    require(counts["hand_total"] == expected_hand_rows, f"{case}: hand rows do not cover full timeline")
    require(counts["hawor_metric_mano"] == expected_hand_rows, f"{case}: HaWoR metric MANO does not cover all hand rows")
    require(counts["hand_graph_metric"] == expected_hand_rows, f"{case}: graph hand variables do not all consume HaWoR metric MANO")
    require(counts["wilor_key_rows"] == expected_hand_rows, f"{case}: WiLoR/V16 hand evidence keys missing")
    require(counts["rtmlib_key_rows"] == expected_hand_rows, f"{case}: RTMLib hand evidence keys missing")
    require(counts["pose_fill_gate_rows"] == expected_hand_rows, f"{case}: pose fill gate rows do not cover both hands/full timeline")
    require(counts["object_states"] > 0, f"{case}: no object states")
    require(counts["object_physical_state_rows"] == counts["object_states"], f"{case}: physical-state decisions missing on object rows")
    require(counts["object_se3_rows"] == counts["object_states"], f"{case}: object SE3 observations missing on object rows")
    require(counts["object_visible_geometry_rows"] > 0, f"{case}: no depth-visible geometry rows")
    require(counts["object_vertex_sample_rows"] > 0, f"{case}: no visible geometry vertex samples")
    require(counts["object_hidden_or_unresolved_geometry_rows"] > 0, f"{case}: no hidden/unresolved geometry state rows")
    require(counts["part_rows"] > 0, f"{case}: no part rows")
    require(counts["factor_frames"] == expected, f"{case}: factor graph not present for every frame")
    require(counts["camera_depth_observed_rows"] > 0, f"{case}: no observed camera/depth correction rows")
    require(counts["contacts"] > 0, f"{case}: no contact hypotheses")
    require(counts["contact_switch_vars"] == counts["contacts"], f"{case}: contact switch variables do not cover contact hypotheses")
    require(counts["active_contact_switch_vars"] > 0, f"{case}: no active contact switches in factor graph")
    require(counts["contacts_with_final_metric_distance"] > 0, f"{case}: no final metric MANO-to-object-surface distances")
    require(counts["signed_nonpenetration_rows"] > 0, f"{case}: no signed nonpenetration evidence rows")
    require(counts["triangle_nonpenetration_rows"] > 0, f"{case}: no triangle nonpenetration evidence rows")
    require(counts["occlusion_owner_vars"] > 0, f"{case}: no occlusion owner graph variables")
    return {"case": case, "expected_frame_count": expected, **counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline/v18_full_pipeline_report.json"))
    args = parser.parse_args()
    report_text = args.report.read_text(encoding="utf-8")
    self_inspection_path = args.report.parent / "v18_completion_self_inspection.json"
    if self_inspection_path.exists():
        report_text = report_text + "\n" + self_inspection_path.read_text(encoding="utf-8")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    require(report.get("all_frame_counts_match") is True, "global frame count mismatch")
    cases = report.get("cases")
    require(isinstance(cases, list) and len(cases) == 2, "report must contain the two representative cases")
    rows = [validate_case(case_report, report_text) for case_report in cases if isinstance(case_report, dict)]
    print(json.dumps({"validation": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
