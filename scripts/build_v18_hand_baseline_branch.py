#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_hand_baseline_branch"
CLAIM = (
    "This artifact restores the V18 hand-branch baseline as explicit evidence: HaWoR temporal hand measurements, "
    "WiLoR visible-frame measurements, RTMLib 2D keypoint anchors, and interior hand/depth state are joined on the "
    "full timeline. It does not accept occluded hand pose, contact, or ownership; missing coverage and missing score "
    "components remain explicit blockers."
)
HAND_SIDES = ("left", "right")
ACCEPT_MEDIAN_2D_PX = 35.0
ACCEPT_P95_2D_PX = 90.0
ACCEPT_RTMLIB_MEDIAN_DELTA_PX = 35.0


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "count": len(clean),
        "median": percentile(clean, 50.0),
        "p05": percentile(clean, 5.0),
        "p95": percentile(clean, 95.0),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def source_from_measurement_manifest(manifest: dict[str, Any], source_key: str) -> tuple[Path | None, str | None]:
    rows = manifest.get(source_key)
    if not isinstance(rows, list):
        return None, None
    for raw in rows:
        row = require_dict(raw, source_key)
        status = str(row.get("status"))
        if status in {"ok", "loaded"} and row.get("path"):
            return Path(require_str(row.get("path"), f"{source_key}.path")), status
    return None, None


def entity_side(entity_id: Any, fallback: Any = None) -> str | None:
    if isinstance(entity_id, str) and entity_id.startswith("hand:"):
        side = entity_id.split(":", 1)[1]
        if side in HAND_SIDES:
            return side
    if fallback in HAND_SIDES:
        return str(fallback)
    return None


def best_measurements_by_frame_side(rows: list[Any], label: str) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in rows:
        row = require_dict(raw, f"{label} row")
        frame_idx = row.get("frame_idx")
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int):
            continue
        side = entity_side(row.get("entity_id"), row.get("side"))
        if side is None:
            continue
        confidence = finite_float_or_none(row.get("confidence"))
        available = row.get("measurement_available") is not False
        score = (1.0 if available else 0.0, confidence if confidence is not None else -1.0)
        key = (frame_idx, side)
        current = out.get(key)
        if current is None:
            out[key] = row
            continue
        current_conf = finite_float_or_none(current.get("confidence"))
        current_score = (1.0 if current.get("measurement_available") is not False else 0.0, current_conf if current_conf is not None else -1.0)
        if score > current_score:
            out[key] = row
    return out


def rtmlib_index(path: Path | None, frame_count: int) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    if path is None or not path.exists():
        return {}, {}
    payload = require_dict(load_json(path), f"RTMLib {path}")
    by_frame: dict[int, dict[str, Any]] = {}
    by_frame_side: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in require_list(payload.get("frames"), "RTMLib frames"):
        frame = require_dict(raw_frame, "RTMLib frame")
        frame_idx = require_int(frame.get("frame_idx"), "RTMLib frame_idx")
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        hands = [require_dict(raw, "RTMLib hand") for raw in require_list(frame.get("hands", []), "RTMLib hands")]
        by_frame[frame_idx] = {
            "frame_idx": frame_idx,
            "hand_detection_count": len(hands),
            "max_mean_score": max([finite_float_or_none(row.get("mean_score")) or 0.0 for row in hands], default=None),
            "valid_keypoint_count_max": max([int(row.get("valid_keypoints") or 0) for row in hands], default=0),
        }
        for raw_cmp in require_list(frame.get("wilor_comparisons", []), "RTMLib WiLoR comparisons"):
            cmp_row = require_dict(raw_cmp, "RTMLib WiLoR comparison")
            side = cmp_row.get("wilor_side")
            if side not in HAND_SIDES:
                continue
            key = (frame_idx, str(side))
            current = by_frame_side.get(key)
            matched = int(cmp_row.get("matched_keypoints") or 0)
            median_delta = finite_float_or_none(cmp_row.get("median_keypoint_delta_px"))
            score = (matched, -(median_delta if median_delta is not None else 1e9))
            if current is None:
                by_frame_side[key] = cmp_row
                continue
            current_matched = int(current.get("matched_keypoints") or 0)
            current_delta = finite_float_or_none(current.get("median_keypoint_delta_px"))
            current_score = (current_matched, -(current_delta if current_delta is not None else 1e9))
            if score > current_score:
                by_frame_side[key] = cmp_row
    return by_frame, by_frame_side


def interior_hand_lookup(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    payload = require_dict(load_json(path), f"interior hand graph {path}")
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in require_list(payload.get("rows"), "interior rows"):
        row = require_dict(raw, "interior row")
        frame_idx = require_int(row.get("frame_idx"), "interior frame_idx")
        side = require_str(row.get("hand_side"), "interior hand_side")
        if side not in HAND_SIDES:
            continue
        key = (frame_idx, side)
        compatible = row.get("interior_metric_depth_compatible") is True
        score = 1 if compatible else 0
        current = best.get(key)
        current_score = 1 if current and current.get("interior_metric_depth_compatible") is True else 0
        if current is None or score > current_score:
            best[key] = row
    return best


def measurement_paths(case: str, args: argparse.Namespace) -> dict[str, Path]:
    root = args.measurement_store_root / case
    return {
        "wilor": root / "measurements_v17" / "wilor_measurements.json",
        "hawor": root / "measurements_v17" / "hawor_measurements.json",
        "manifest": root / "v17_measurement_manifest.json",
    }


def component_state(
    hawor: dict[str, Any] | None,
    wilor: dict[str, Any] | None,
    rtmlib_cmp: dict[str, Any] | None,
    interior: dict[str, Any] | None,
) -> tuple[str, list[str], dict[str, Any]]:
    blockers: list[str] = []
    med_proj = finite_float_or_none(hawor.get("projection_residual_px_median")) if hawor is not None else None
    p95_proj = finite_float_or_none(hawor.get("projection_residual_px_p95")) if hawor is not None else None
    rtmlib_delta = finite_float_or_none(rtmlib_cmp.get("median_keypoint_delta_px")) if rtmlib_cmp is not None else None
    hawor_available = hawor is not None and hawor.get("measurement_available") is True
    hawor_infill = hawor is not None and hawor.get("measurement_available") is False and hawor.get("evidence_role") == "hawor_motion_infill_candidate"
    projection_ok = med_proj is not None and med_proj <= ACCEPT_MEDIAN_2D_PX and (p95_proj is None or p95_proj <= ACCEPT_P95_2D_PX)
    rtmlib_ok = rtmlib_delta is not None and rtmlib_delta <= ACCEPT_RTMLIB_MEDIAN_DELTA_PX
    depth_compatible = interior is not None and interior.get("interior_metric_depth_compatible") is True
    if hawor is None:
        blockers.append("hawor_missing_for_frame_side")
    elif not hawor_available:
        blockers.append("hawor_temporal_infill_candidate_not_measurement")
    if med_proj is None:
        blockers.append("hawor_projection_residual_missing")
    elif not projection_ok:
        blockers.append("hawor_projection_residual_above_threshold")
    if rtmlib_cmp is None:
        blockers.append("rtmlib_wilor_comparison_missing")
    elif not rtmlib_ok:
        blockers.append("rtmlib_wilor_2d_delta_above_threshold")
    if interior is None:
        blockers.append("interior_hand_depth_state_missing")
    elif not depth_compatible:
        blockers.append("interior_hand_depth_not_metric_compatible")
    blockers.extend(["median_metric_depth_abs_residual_component_missing", "temporal_acceleration_component_missing", "hand_bone_scale_component_missing"])
    if hawor_available and projection_ok:
        state = "hawor_visible_measurement_2d_supported_depth_components_missing"
    elif hawor_infill:
        state = "hawor_motion_infill_candidate_not_accepted"
    elif wilor is not None:
        state = "wilor_visible_candidate_no_accepted_hawor"
    else:
        state = "no_hand_baseline_candidate"
    score = None
    if med_proj is not None or rtmlib_delta is not None:
        score = (med_proj or 0.0) / 25.0 + (rtmlib_delta or 0.0) / 25.0
    components = {
        "hawor_projection_residual_px_median": med_proj,
        "hawor_projection_residual_px_p95": p95_proj,
        "rtmlib_wilor_median_keypoint_delta_px": rtmlib_delta,
        "interior_metric_depth_compatible": depth_compatible,
        "available_partial_score_2d_terms_only": score,
        "score_contract_missing_components": [
            "median_metric_depth_abs_m/0.05",
            "temporal_acceleration_m/0.05",
            "hand_bone_scale_error_m/0.025",
        ],
    }
    return state, sorted(set(blockers)), components


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    v16_manifest_path = args.v16_root / case / "v16_full_pipeline_manifest.json"
    v16 = require_dict(load_json(v16_manifest_path), f"{case} V16 manifest")
    raw = require_dict(v16.get("raw_video"), f"{case} raw video")
    frame_count = require_int(raw.get("frame_count"), f"{case} frame count")
    paths = measurement_paths(case, args)
    measurement_manifest = require_dict(load_json(paths["manifest"]), f"{case} measurement manifest")
    rtmlib_path, rtmlib_manifest_status = source_from_measurement_manifest(measurement_manifest, "rtmlib_hand2d_sources")
    if not paths["wilor"].exists():
        raise RuntimeError(f"{case} WiLoR measurement file missing: {paths['wilor']}")
    if not paths["hawor"].exists():
        raise RuntimeError(f"{case} HaWoR measurement file missing: {paths['hawor']}")
    wilor_rows = require_list(load_json(paths["wilor"]), f"{case} WiLoR measurements")
    hawor_rows = require_list(load_json(paths["hawor"]), f"{case} HaWoR measurements")
    wilor_by_side = best_measurements_by_frame_side(wilor_rows, "WiLoR")
    hawor_by_side = best_measurements_by_frame_side(hawor_rows, "HaWoR")
    rtmlib_by_frame, rtmlib_by_side = rtmlib_index(rtmlib_path, frame_count)
    interior_path = args.interior_hand_graph_root / case / "v17_interior_owned_full_residual_hand_graph.json"
    interior_by_side = interior_hand_lookup(interior_path)
    hand_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    for frame_idx in range(frame_count):
        for side in HAND_SIDES:
            key = (frame_idx, side)
            wilor = wilor_by_side.get(key)
            hawor = hawor_by_side.get(key)
            rtmlib_cmp = rtmlib_by_side.get(key)
            interior = interior_by_side.get(key)
            state, blockers, components = component_state(hawor, wilor, rtmlib_cmp, interior)
            state_counts[state] += 1
            blocker_counts.update(blockers)
            hand_rows.append(
                {
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "hand_baseline_state": state,
                    "acceptance_blockers": blockers,
                    "baseline_score_components": components,
                    "wilor_measurement_available": wilor is not None,
                    "wilor_confidence": finite_float_or_none(wilor.get("confidence")) if wilor is not None else None,
                    "wilor_bbox_xyxy": wilor.get("bbox_xyxy") if wilor is not None else None,
                    "hawor_candidate_present": hawor is not None,
                    "hawor_measurement_available": hawor.get("measurement_available") is True if hawor is not None else False,
                    "hawor_evidence_role": hawor.get("evidence_role") if hawor is not None else None,
                    "hawor_confidence": finite_float_or_none(hawor.get("confidence")) if hawor is not None else None,
                    "hawor_projection_residual_px_median": components["hawor_projection_residual_px_median"],
                    "hawor_projection_residual_px_p95": components["hawor_projection_residual_px_p95"],
                    "rtmlib_frame_detection_count": int(rtmlib_by_frame.get(frame_idx, {}).get("hand_detection_count") or 0),
                    "rtmlib_wilor_comparison_available": rtmlib_cmp is not None,
                    "rtmlib_wilor_median_keypoint_delta_px": components["rtmlib_wilor_median_keypoint_delta_px"],
                    "interior_metric_depth_state": interior.get("interior_state") if interior is not None else None,
                    "interior_metric_depth_compatible": components["interior_metric_depth_compatible"],
                    "temporal_occlusion_pose_accepted": False,
                    "pose_claim": "no_occluded_pose_accepted_from_current_hand_baseline",
                }
            )
    frames: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in hand_rows:
        by_frame[require_int(row.get("frame_idx"), "hand row frame")].append(row)
    for frame_idx in range(frame_count):
        frames.append({"frame_idx": frame_idx, "hands": by_frame.get(frame_idx, [])})
    hawor_available_rows = [row for row in hawor_rows if require_dict(row, "HaWoR row").get("measurement_available") is True]
    hawor_infill_rows = [row for row in hawor_rows if require_dict(row, "HaWoR row").get("measurement_available") is False and require_dict(row, "HaWoR row").get("evidence_role") == "hawor_motion_infill_candidate"]
    hawor_frames = sorted({require_int(require_dict(row, "HaWoR row").get("frame_idx"), "HaWoR frame_idx") for row in hawor_rows})
    required_frame_side_keys = {(frame_idx, side) for frame_idx in range(frame_count) for side in HAND_SIDES}
    hawor_available_frame_side_keys = {
        (require_int(require_dict(row, "HaWoR row").get("frame_idx"), "HaWoR frame_idx"), require_str(require_dict(row, "HaWoR row").get("entity_id"), "HaWoR entity_id").split(":", 1)[1])
        for row in hawor_rows
        if require_dict(row, "HaWoR row").get("measurement_available") is True
        and isinstance(require_dict(row, "HaWoR row").get("entity_id"), str)
        and require_str(require_dict(row, "HaWoR row").get("entity_id"), "HaWoR entity_id").startswith("hand:")
        and require_str(require_dict(row, "HaWoR row").get("entity_id"), "HaWoR entity_id").split(":", 1)[1] in HAND_SIDES
    }
    hawor_projection_residuals = [
        value
        for row in hawor_rows
        for value in [finite_float_or_none(require_dict(row, "HaWoR row").get("projection_residual_px_median"))]
        if value is not None
    ]
    hawor_confidences = [
        value
        for row in hawor_rows
        for value in [finite_float_or_none(require_dict(row, "HaWoR row").get("confidence"))]
        if value is not None
    ]
    rtmlib_frames_with_hands = sum(1 for row in rtmlib_by_frame.values() if int(row.get("hand_detection_count") or 0) > 0)
    full_video_hawor_ready = required_frame_side_keys.issubset(hawor_available_frame_side_keys)
    output_dir = args.output_root / case
    state_path = output_dir / "v18_hand_baseline_branch.json"
    report = {
        "method": "build_v18_hand_baseline_branch",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "raw_video": raw,
        "frame_count": frame_count,
        "state_path": str(state_path),
        "sources": {
            "v16_manifest": str(v16_manifest_path),
            "v17_measurement_manifest": str(paths["manifest"]),
            "wilor_measurements": str(paths["wilor"]),
            "hawor_measurements": str(paths["hawor"]),
            "rtmlib_hand2d": str(rtmlib_path) if rtmlib_path is not None else None,
            "rtmlib_manifest_status": rtmlib_manifest_status,
            "interior_hand_graph": str(interior_path),
        },
        "hand_state_row_count": len(hand_rows),
        "hand_baseline_state_counts": dict(sorted(state_counts.items())),
        "acceptance_blocker_counts": dict(sorted(blocker_counts.items())),
        "wilor_measurement_row_count": len(wilor_rows),
        "wilor_frame_side_count": len(wilor_by_side),
        "hawor_measurement_row_count": len(hawor_rows),
        "hawor_available_measurement_count": len(hawor_available_rows),
        "hawor_motion_infill_candidate_count": len(hawor_infill_rows),
        "hawor_frame_min": min(hawor_frames) if hawor_frames else None,
        "hawor_frame_max": max(hawor_frames) if hawor_frames else None,
        "hawor_unique_frame_count": len(hawor_frames),
        "hawor_required_frame_side_count": len(required_frame_side_keys),
        "hawor_available_frame_side_count": len(hawor_available_frame_side_keys),
        "hawor_missing_available_frame_side_count": len(required_frame_side_keys - hawor_available_frame_side_keys),
        "hawor_projection_residual_px_median_stats": stats(hawor_projection_residuals),
        "hawor_confidence_stats": stats(hawor_confidences),
        "hawor_full_video_baseline_ready": full_video_hawor_ready,
        "hawor_full_video_blockers": [] if full_video_hawor_ready else ["hawor_measurements_do_not_cover_full_video_all_frame_sides"],
        "rtmlib_manifest_status": rtmlib_manifest_status,
        "rtmlib_source_status_normalized": rtmlib_path is not None,
        "rtmlib_frame_count": len(rtmlib_by_frame),
        "rtmlib_frames_with_hands": rtmlib_frames_with_hands,
        "rtmlib_wilor_comparison_count": len(rtmlib_by_side),
        "temporal_occlusion_pose_accepted_count": 0,
        "pose_filled_through_occlusion_rows": 0,
        "acceptance_policy": "HaWoR/WiLoR/RTMLib rows are measurement evidence only until all score components and boundary/occlusion checks pass; no current row fills occluded pose.",
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    state = {**report, "frames": frames}
    write_json(state_path, state)
    write_json(output_dir / "v18_hand_baseline_branch_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "build_v18_hand_baseline_branch",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "hand_state_row_count": sum(require_int(report.get("hand_state_row_count"), "hand rows") for report in reports),
        "wilor_measurement_row_count": sum(require_int(report.get("wilor_measurement_row_count"), "WiLoR rows") for report in reports),
        "hawor_measurement_row_count": sum(require_int(report.get("hawor_measurement_row_count"), "HaWoR rows") for report in reports),
        "hawor_available_measurement_count": sum(require_int(report.get("hawor_available_measurement_count"), "HaWoR available") for report in reports),
        "hawor_motion_infill_candidate_count": sum(require_int(report.get("hawor_motion_infill_candidate_count"), "HaWoR infill") for report in reports),
        "hawor_full_video_ready_case_count": sum(1 for report in reports if report.get("hawor_full_video_baseline_ready") is True),
        "hawor_full_video_baseline_ready_all_cases": all(report.get("hawor_full_video_baseline_ready") is True for report in reports),
        "rtmlib_loaded_case_count": sum(1 for report in reports if report.get("rtmlib_source_status_normalized") is True),
        "rtmlib_frames_with_hands": sum(require_int(report.get("rtmlib_frames_with_hands"), "RTMLib frames with hands") for report in reports),
        "rtmlib_wilor_comparison_count": sum(require_int(report.get("rtmlib_wilor_comparison_count"), "RTMLib WiLoR comparisons") for report in reports),
        "temporal_occlusion_pose_accepted_count": 0,
        "pose_filled_through_occlusion_rows": 0,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_hand_baseline_branch_report.json"),
                "state_path": str(args.output_root / str(report["case"]) / "v18_hand_baseline_branch.json"),
                "hawor_measurement_row_count": report.get("hawor_measurement_row_count"),
                "hawor_available_measurement_count": report.get("hawor_available_measurement_count"),
                "hawor_motion_infill_candidate_count": report.get("hawor_motion_infill_candidate_count"),
                "hawor_full_video_baseline_ready": report.get("hawor_full_video_baseline_ready"),
                "rtmlib_source_status_normalized": report.get("rtmlib_source_status_normalized"),
                "rtmlib_frames_with_hands": report.get("rtmlib_frames_with_hands"),
                **FALSE_READY,
            }
            for report in reports
        ],
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_hand_baseline_branch_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--measurement-store-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--interior-hand-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_hand_baseline_branch"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
