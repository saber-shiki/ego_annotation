#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseSpec:
    name: str
    root: Path
    anchor_frames: tuple[int, ...]
    expected_visible_hands: dict[int, int]
    expected_contact: dict[int, str]
    expected_objects: tuple[str, ...]
    hawor_annotation_paths: tuple[Path, ...] = ()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def bbox_area(bbox: Any) -> float | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    vals = [as_float(x) for x in bbox]
    if any(v is None for v in vals):
        return None
    x0, y0, x1, y1 = vals  # type: ignore[misc]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def compact_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    vals = [as_float(x) for x in bbox]
    if any(v is None for v in vals):
        return None
    return [float(v) for v in vals if v is not None]


def source_path_from_manifest(manifest: dict[str, Any], key: str) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"manifest missing path field {key}")
    path = Path(value)
    if not path.exists():
        raise RuntimeError(f"manifest path for {key} does not exist: {path}")
    return path


def wilor_raw_path(manifest: dict[str, Any]) -> Path:
    qc_path = source_path_from_manifest(manifest, "hand_qc")
    qc = load_json(qc_path)
    raw = qc.get("raw_path")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"WiLoR QC missing raw_path: {qc_path}")
    path = Path(raw)
    if not path.exists():
        raise RuntimeError(f"WiLoR raw_path does not exist: {path}")
    return path


def measurements_from_wilor(raw_path: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    payload = load_json(raw_path)
    measurements: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in payload.get("frames", []):
        idx = int(frame["frame_idx"])
        entries = frame.get("raw_hands") or []
        frame_rows: list[dict[str, Any]] = []
        for det_i, hand in enumerate(entries):
            if not isinstance(hand, dict):
                continue
            bbox = compact_bbox(hand.get("bbox_xyxy"))
            score = as_float(hand.get("detector_score"))
            row = {
                "measurement_id": f"wilor:{idx}:{det_i}",
                "frame_idx": idx,
                "entity_type": "hand",
                "entity_id": f"hand:{hand.get('side', 'unknown')}",
                "measurement_type": "mano_per_frame",
                "source_model": hand.get("backend", "WiLoR"),
                "coordinate_frame": "source_camera",
                "confidence": score,
                "bbox_xyxy": bbox,
                "bbox_area_px2": bbox_area(bbox),
                "has_joints2d": hand.get("joints2d") is not None,
                "has_joints3d_camera": hand.get("joints3d_camera") is not None,
                "has_vertices_camera": hand.get("vertices_camera") is not None or hand.get("vertices_camera_sample") is not None,
                "has_mano_params": hand.get("mano_params") is not None,
                "failure_reason": None,
            }
            measurements.append(row)
            frame_rows.append(row)
        by_frame[idx] = frame_rows
    return measurements, by_frame


def measurements_from_v16_hands(frames: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    measurements: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in frames.values():
        idx = int(frame["frame_idx"])
        frame_rows: list[dict[str, Any]] = []
        for hand_i, hand in enumerate(frame.get("hands") or []):
            if not isinstance(hand, dict):
                continue
            bbox = compact_bbox(hand.get("bbox_xyxy"))
            row = {
                "measurement_id": f"v16_hand:{idx}:{hand_i}",
                "frame_idx": idx,
                "entity_type": "hand",
                "entity_id": f"hand:{hand.get('side', 'unknown')}",
                "measurement_type": "delivered_v16_hand_state",
                "source_model": hand.get("backend") or hand.get("source") or "unknown_v16_hand_source",
                "coordinate_frame": "v16_annotation_world_and_source_camera",
                "confidence": as_float(hand.get("score")),
                "bbox_xyxy": bbox,
                "bbox_area_px2": bbox_area(bbox),
                "has_joints2d": hand.get("joints2d") is not None,
                "has_joints3d_camera": hand.get("joints3d_camera") is not None,
                "has_vertices_camera": hand.get("vertices_camera") is not None or hand.get("vertices_camera_sample") is not None,
                "has_mano_params": hand.get("mano_params") is not None,
                "failure_reason": "missing_source_confidence" if hand.get("score") is None and hand.get("source") is None else None,
            }
            measurements.append(row)
            frame_rows.append(row)
        by_frame[idx] = frame_rows
    return measurements, by_frame


def residual_summary_px(hand: dict[str, Any]) -> dict[str, float | None]:
    residual = hand.get("projection_residual_to_measurement_px")
    if not isinstance(residual, dict):
        return {"median": None, "p95": None}
    return {
        "median": as_float(residual.get("median")),
        "p95": as_float(residual.get("p95")),
    }


def measurements_from_hawor(annotation_paths: tuple[Path, ...]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for source_i, path in enumerate(annotation_paths):
        if not path.exists():
            sources.append({"path": str(path), "status": "missing"})
            continue
        payload = load_json(path)
        source_rows = 0
        source_frames: list[int] = []
        for frame in payload.get("frames", []):
            idx = int(frame["frame_idx"])
            frame_rows: list[dict[str, Any]] = []
            for hand_i, hand in enumerate(frame.get("hands") or []):
                if not isinstance(hand, dict) or hand.get("backend") != "HaWoR":
                    continue
                residual = residual_summary_px(hand)
                bbox = compact_bbox(hand.get("bbox_xyxy"))
                measurement_available = bool(hand.get("measurement_available", False))
                row = {
                    "measurement_id": f"hawor:{source_i}:{idx}:{hand_i}",
                    "frame_idx": idx,
                    "entity_type": "hand",
                    "entity_id": f"hand:{hand.get('side', 'unknown')}",
                    "measurement_type": "mano_temporal_motion_prior",
                    "source_model": "HaWoR",
                    "coordinate_frame": hand.get("world_coordinate_status")
                    or "hawor_camera_local_existing_camera_pose_bridge",
                    "confidence": as_float(hand.get("detector_score")) if measurement_available else None,
                    "bbox_xyxy": bbox,
                    "bbox_area_px2": bbox_area(bbox),
                    "has_joints2d": hand.get("joints2d") is not None,
                    "has_joints3d_camera": hand.get("joints3d_source_camera_m") is not None
                    or hand.get("joints3d_camera") is not None,
                    "has_vertices_camera": hand.get("vertices_source_camera_m") is not None
                    or hand.get("vertices_camera") is not None,
                    "has_mano_params": hand.get("mano_params") is not None,
                    "measurement_available": measurement_available,
                    "filter_status": hand.get("filter_status"),
                    "projection_residual_px_median": residual["median"],
                    "projection_residual_px_p95": residual["p95"],
                    "mano_vertex_count": hand.get("mano_vertex_count"),
                    "source_annotation": str(path),
                    "failure_reason": None if measurement_available else "hawor_geometry_without_2d_observation_support",
                }
                measurements.append(row)
                frame_rows.append(row)
                source_rows += 1
            if frame_rows:
                by_frame.setdefault(idx, []).extend(frame_rows)
                source_frames.append(idx)
        sources.append(
            {
                "path": str(path),
                "status": "loaded",
                "measurement_count": source_rows,
                "active_frame_min": min(source_frames) if source_frames else None,
                "active_frame_max": max(source_frames) if source_frames else None,
                "active_frame_count": len(set(source_frames)),
            }
        )
    return measurements, by_frame, sources


def measurements_from_object_mesh_qc(qc_path: Path, frames: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    qc = load_json(qc_path)
    measurements: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    rows = list(qc.get("rows") or []) + list(qc.get("prediction_rows") or [])
    for row_i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        idx = int(row["frame_idx"])
        ann_obj = frames.get(idx, {}).get("object", {})
        object_status = ann_obj.get("status")
        state = row.get("delivered_state") or ann_obj.get("mesh_state")
        if object_status == "outside_semantic_interval":
            state = "inactive_diagnostic_mesh_row"
        entry = {
            "measurement_id": f"object_mesh:{idx}:{row_i}",
            "frame_idx": idx,
            "entity_type": "object",
            "entity_id": f"object:{ann_obj.get('label') or 'unknown'}",
            "measurement_type": "object_inactive_diagnostic_mesh"
            if state == "inactive_diagnostic_mesh_row"
            else ("object_visible_surface_mesh" if state != "predicted" else "object_mesh_prediction"),
            "source_model": row.get("surface_depth_model") or row.get("status") or "v16_mesh_stream",
            "coordinate_frame": "v16_world",
            "confidence": None,
            "status": row.get("status"),
            "object_status": object_status,
            "mesh_state": state,
            "vertices": row.get("vertices"),
            "faces": row.get("faces"),
            "bbox_xyxy": compact_bbox(ann_obj.get("bbox_xyxy")),
            "depth_median_m": as_float(row.get("depth_median_m")),
            "world_extent_m": row.get("world_extent_m"),
            "failure_reason": None,
        }
        measurements.append(entry)
        if state != "inactive_diagnostic_mesh_row":
            by_frame.setdefault(idx, []).append(entry)
    return measurements, by_frame


def frame_state(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(frame["frame_idx"]): frame for frame in payload.get("frames", [])}


def object_roster_from_v16(frames: dict[int, dict[str, Any]], expected_objects: tuple[str, ...]) -> list[dict[str, Any]]:
    labels = set(expected_objects)
    for frame in frames.values():
        label = frame.get("object", {}).get("label")
        if isinstance(label, str) and label:
            labels.add(label)
    roster = []
    for label in sorted(labels):
        active = [
            idx
            for idx, frame in frames.items()
            if frame.get("object", {}).get("label") == label
            and frame.get("object", {}).get("status") != "outside_semantic_interval"
        ]
        roster.append(
            {
                "object_id": f"object:{label}",
                "name": label,
                "source": "v16_label_or_v17_expected_context",
                "active_frame_min": min(active) if active else None,
                "active_frame_max": max(active) if active else None,
                "active_frame_count": len(active),
                "role_status": "measured_or_inherited" if active else "expected_missing_from_v16",
            }
        )
    return roster


def anchor_qc(
    spec: CaseSpec,
    frames: dict[int, dict[str, Any]],
    wilor_by_frame: dict[int, list[dict[str, Any]]],
    hawor_by_frame: dict[int, list[dict[str, Any]]],
    v16_hand_by_frame: dict[int, list[dict[str, Any]]],
    object_by_frame: dict[int, list[dict[str, Any]]],
    roster: list[dict[str, Any]],
) -> dict[str, Any]:
    anchors = []
    for idx in spec.anchor_frames:
        frame = frames.get(idx)
        if frame is None:
            anchors.append({"frame_idx": idx, "status": "missing_annotation_frame"})
            continue
        v16_hands = v16_hand_by_frame.get(idx, [])
        wilor_hands = wilor_by_frame.get(idx, [])
        hawor_hands = hawor_by_frame.get(idx, [])
        object_measurements = object_by_frame.get(idx, [])
        expected_visible = spec.expected_visible_hands.get(idx)
        obj = frame.get("object", {})
        object_status = obj.get("status")
        failures = []
        if expected_visible is not None and len(v16_hands) < expected_visible:
            failures.append("visible_hands_missing_from_v16_state")
        if v16_hands and any(row.get("failure_reason") == "missing_source_confidence" for row in v16_hands):
            failures.append("v16_hand_state_lacks_source_confidence")
        if expected_visible is not None and len(wilor_hands) < expected_visible:
            failures.append("wilor_measurements_missing_for_visible_hands")
        if expected_visible is not None and hawor_hands and len(hawor_hands) < expected_visible:
            failures.append("hawor_measurements_incomplete_for_visible_hands")
        if hawor_hands and any(row.get("failure_reason") for row in hawor_hands):
            failures.append("hawor_geometry_without_2d_observation_support")
        if object_status == "outside_semantic_interval" and spec.expected_contact.get(idx):
            failures.append("object_inactive_despite_expected_interaction_context")
        if spec.expected_contact.get(idx) == "contact" and not object_measurements:
            failures.append("contact_anchor_without_object_mesh_measurement")
        if spec.expected_contact.get(idx):
            failures.append("missing_contact_state_measurement")
        if idx == 856 and object_measurements:
            failures.append("known_bad_state_can_still_emit_small_distance_contact_label")
        anchors.append(
            {
                "frame_idx": idx,
                "caption": frame.get("caption"),
                "expected_visible_hands": expected_visible,
                "expected_contact": spec.expected_contact.get(idx),
                "v16_hand_count": len(v16_hands),
                "wilor_raw_hand_count": len(wilor_hands),
                "hawor_hand_count": len(hawor_hands),
                "hawor_observed_hand_count": sum(1 for row in hawor_hands if row.get("measurement_available")),
                "object_status": object_status,
                "object_label": obj.get("label"),
                "object_mesh_measurement_count": len(object_measurements),
                "failures": failures,
                "status": "pass" if not failures else "fail",
            }
        )
    missing_expected_objects = [row["name"] for row in roster if row["role_status"] == "expected_missing_from_v16"]
    return {
        "case": spec.name,
        "status": "pass" if all(row.get("status") == "pass" for row in anchors) and not missing_expected_objects else "fail",
        "anchors": anchors,
        "missing_expected_objects": missing_expected_objects,
    }


def build_case(spec: CaseSpec, output_root: Path) -> dict[str, Any]:
    manifest_path = spec.root / "v16_full_pipeline_manifest.json"
    manifest = load_json(manifest_path)
    annotations_path = source_path_from_manifest(manifest, "annotations")
    object_qc_path = source_path_from_manifest(manifest, "object_mesh_qc")
    raw_wilor_path = wilor_raw_path(manifest)

    annotations = load_json(annotations_path)
    frames = frame_state(annotations)
    wilor_measurements, wilor_by_frame = measurements_from_wilor(raw_wilor_path)
    hawor_measurements, hawor_by_frame, hawor_sources = measurements_from_hawor(spec.hawor_annotation_paths)
    v16_hand_measurements, v16_hand_by_frame = measurements_from_v16_hands(frames)
    object_measurements, object_by_frame = measurements_from_object_mesh_qc(object_qc_path, frames)
    roster = object_roster_from_v16(frames, spec.expected_objects)

    case_dir = output_root / spec.name
    measurements_dir = case_dir / "measurements_v17"
    write_json(measurements_dir / "wilor_measurements.json", wilor_measurements)
    write_json(measurements_dir / "hawor_measurements.json", hawor_measurements)
    write_json(measurements_dir / "v16_hand_state_measurements.json", v16_hand_measurements)
    write_json(measurements_dir / "object_mesh_measurements.json", object_measurements)
    write_json(case_dir / "object_roster_v17.json", roster)
    anchor = anchor_qc(spec, frames, wilor_by_frame, hawor_by_frame, v16_hand_by_frame, object_by_frame, roster)
    write_json(case_dir / "v17_anchor_qc.json", anchor)

    report = {
        "case": spec.name,
        "status": anchor["status"],
        "v16_root": str(spec.root),
        "manifest": str(manifest_path),
        "annotations": str(annotations_path),
        "wilor_raw": str(raw_wilor_path),
        "hawor_sources": hawor_sources,
        "object_mesh_qc": str(object_qc_path),
        "measurement_counts": {
            "wilor": len(wilor_measurements),
            "hawor": len(hawor_measurements),
            "v16_hand_state": len(v16_hand_measurements),
            "object_mesh": len(object_measurements),
        },
        "object_roster": str(case_dir / "object_roster_v17.json"),
        "anchor_qc": str(case_dir / "v17_anchor_qc.json"),
    }
    write_json(case_dir / "v17_measurement_manifest.json", report)
    return report


def default_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            name="trash_1050",
            root=Path("/data2/ego_annotation_outputs/v16_full_pipeline/trash_1050"),
            anchor_frames=(182, 260, 764, 856, 949, 970),
            expected_visible_hands={182: 2, 260: 2, 764: 2, 856: 2, 949: 2, 970: 2},
            expected_contact={764: "contact", 856: "contact_or_near_contact"},
            expected_objects=("trash_bag", "trash_can", "trash_can_lid"),
            hawor_annotation_paths=(
                Path(
                    "/data2/ego_annotation_outputs/representative_trash/"
                    "v3_hawor_camera_local_840_930/annotations_hawor_camera_local.json"
                ),
            ),
        ),
        CaseSpec(
            name="task5_tomato_960",
            root=Path("/data2/ego_annotation_outputs/v16_full_pipeline/task5_tomato_960"),
            anchor_frames=(480, 720, 760),
            expected_visible_hands={480: 2, 720: 2, 760: 2},
            expected_contact={480: "contact", 720: "contact", 760: "contact"},
            expected_objects=("tomato", "bowl", "plate", "tray"),
        ),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cases = default_cases()
    reports = [build_case(spec, output_root) for spec in cases]
    summary = {
        "status": "pass" if all(row["status"] == "pass" for row in reports) else "fail",
        "method": "build_v17_measurement_store",
        "claim": "V17 measurement store preserves model outputs and exposes V16 anchor failures before graph optimization",
        "cases": reports,
    }
    write_json(output_root / "v17_measurement_store_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
