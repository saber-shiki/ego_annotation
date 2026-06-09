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
    expected_persistent_object_labels: tuple[str, ...] = ()
    expected_hand_repair_frames: tuple[int, ...] = ()
    hawor_annotation_paths: tuple[Path, ...] = ()
    object_plan_paths: tuple[Path, ...] = ()
    expected_object_coverage_paths: tuple[Path, ...] = ()
    sam2_multiobject_roots: tuple[Path, ...] = ()
    contact_measurement_paths: tuple[Path, ...] = ()
    rtmlib_hand2d_paths: tuple[Path, ...] = ()
    hamer_measurement_paths: tuple[Path, ...] = ()


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


def measurements_from_rtmlib(
    paths: tuple[Path, ...],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for source_i, path in enumerate(paths):
        if not path.exists():
            sources.append({"path": str(path), "status": "missing"})
            continue
        payload = load_json(path)
        frames = payload.get("frames")
        if not isinstance(frames, list):
            sources.append({"path": str(path), "status": "invalid_payload"})
            continue
        source_rows = 0
        frames_with_hands = 0
        for frame in frames:
            idx = int(frame["frame_idx"])
            frame_rows: list[dict[str, Any]] = []
            hands = frame.get("hands") or []
            if hands:
                frames_with_hands += 1
            for det_i, hand in enumerate(hands):
                if not isinstance(hand, dict):
                    continue
                bbox = compact_bbox(hand.get("bbox_xyxy"))
                keypoints = hand.get("keypoints")
                scores = hand.get("scores")
                row = {
                    "measurement_id": f"rtmlib:{source_i}:{idx}:{det_i}",
                    "frame_idx": idx,
                    "entity_type": "hand",
                    "entity_id": f"hand:rtmlib_{hand.get('hand_idx', det_i)}",
                    "measurement_type": "hand_2d_keypoints",
                    "source_model": "RTMLib",
                    "coordinate_frame": "source_image_pixels",
                    "confidence": as_float(hand.get("mean_score")),
                    "bbox_xyxy": bbox,
                    "bbox_area_px2": bbox_area(bbox),
                    "valid_keypoints": hand.get("valid_keypoints"),
                    "median_score": as_float(hand.get("median_score")),
                    "keypoints": keypoints if isinstance(keypoints, list) else None,
                    "scores": scores if isinstance(scores, list) else None,
                    "source_file": str(path),
                    "failure_reason": None,
                }
                measurements.append(row)
                frame_rows.append(row)
                source_rows += 1
            if frame_rows:
                by_frame.setdefault(idx, []).extend(frame_rows)
        sources.append(
            {
                "path": str(path),
                "status": "loaded",
                "measurement_count": source_rows,
                "frame_start": payload.get("frame_start"),
                "frame_end": payload.get("frame_end"),
                "frames": len(frames),
                "frames_with_hands": frames_with_hands,
            }
        )
    return measurements, by_frame, sources


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
            confidence = as_float(hand.get("score"))
            if confidence is None:
                confidence = as_float(hand.get("detector_score"))
            residual = residual_summary_px(hand)
            measurement_available = hand.get("measurement_available")
            failure_reason = None
            if confidence is None and hand.get("source") is None and hand.get("backend") is None:
                failure_reason = "missing_source_confidence"
            elif measurement_available is False:
                failure_reason = "hand_measurement_unavailable"
            row = {
                "measurement_id": f"v16_hand:{idx}:{hand_i}",
                "frame_idx": idx,
                "entity_type": "hand",
                "entity_id": f"hand:{hand.get('side', 'unknown')}",
                "measurement_type": "delivered_v16_hand_state",
                "source_model": hand.get("backend") or hand.get("source") or "unknown_v16_hand_source",
                "coordinate_frame": "v16_annotation_world_and_source_camera",
                "confidence": confidence,
                "bbox_xyxy": bbox,
                "bbox_area_px2": bbox_area(bbox),
                "has_joints2d": hand.get("joints2d") is not None,
                "has_joints3d_camera": hand.get("joints3d_camera") is not None,
                "has_vertices_camera": hand.get("vertices_camera") is not None or hand.get("vertices_camera_sample") is not None,
                "has_mano_params": hand.get("mano_params") is not None,
                "measurement_available": measurement_available,
                "projection_residual_px_median": residual["median"],
                "projection_residual_px_p95": residual["p95"],
                "failure_reason": failure_reason,
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


def measurements_from_hamer_summaries(
    summary_paths: tuple[Path, ...],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for source_i, path in enumerate(summary_paths):
        if not path.exists():
            sources.append({"path": str(path), "status": "missing"})
            continue
        payload = load_json(path)
        rows = payload.get("measurements")
        if not isinstance(rows, list):
            raise RuntimeError(f"{path} has no measurements list")
        source_rows = 0
        measured_rows = 0
        source_frames: list[int] = []
        for row_i, raw in enumerate(rows):
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            idx = int(row["frame_idx"])
            row["measurement_id"] = f"hamer_summary:{source_i}:{row_i}"
            row["source_summary"] = str(path)
            measurements.append(row)
            by_frame.setdefault(idx, []).append(row)
            source_rows += 1
            if bool(row.get("measurement_available", False)):
                measured_rows += 1
            source_frames.append(idx)
        sources.append(
            {
                "path": str(path),
                "status": "loaded",
                "source_annotations": payload.get("source_annotations"),
                "measurement_count": source_rows,
                "measured_hand_rows": measured_rows,
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


def interval_frames(intervals: Any) -> tuple[int | None, int | None, int]:
    starts: list[int] = []
    ends: list[int] = []
    for interval in intervals if isinstance(intervals, list) else []:
        if not isinstance(interval, dict):
            continue
        start = as_float(interval.get("start_frame"))
        end = as_float(interval.get("end_frame"))
        if start is None or end is None:
            continue
        starts.append(int(start))
        ends.append(int(end))
    if not starts or not ends:
        return None, None, 0
    frame_count = sum(max(0, end - start + 1) for start, end in zip(starts, ends, strict=False))
    return min(starts), max(ends), frame_count


def measurements_from_object_plans(plan_paths: tuple[Path, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    roster_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for source_i, path in enumerate(plan_paths):
        if not path.exists():
            sources.append({"path": str(path), "status": "missing"})
            continue
        payload = load_json(path)
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
        objects = plan.get("objects") if isinstance(plan, dict) else None
        if not isinstance(objects, list):
            sources.append({"path": str(path), "status": "invalid_no_objects"})
            continue
        sources.append(
            {
                "path": str(path),
                "status": "loaded",
                "backend": payload.get("backend"),
                "model": payload.get("model"),
                "object_count": len(objects),
            }
        )
        for object_i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            track_id = str(obj.get("track_id") or f"object_plan_{source_i}_{object_i}")
            start, end, count = interval_frames(obj.get("active_intervals"))
            row = {
                "measurement_id": f"object_plan:{source_i}:{object_i}",
                "frame_idx": None,
                "entity_type": "object",
                "entity_id": f"object:{track_id}",
                "measurement_type": "vlm_object_plan",
                "source_model": payload.get("model") or payload.get("backend") or "vlm_object_plan",
                "coordinate_frame": "video_timeline",
                "confidence": as_float(obj.get("confidence")),
                "description": obj.get("description"),
                "open_vocabulary_prompts": obj.get("open_vocabulary_prompts") or [],
                "active_intervals": obj.get("active_intervals") or [],
                "physical_notes": obj.get("physical_notes"),
                "failure_reason": None,
            }
            measurements.append(row)
            roster_rows.append(
                {
                    "object_id": f"object:{track_id}",
                    "name": track_id,
                    "source": "vlm_object_plan",
                    "active_frame_min": start,
                    "active_frame_max": end,
                    "active_frame_count": count,
                    "role_status": "planned_from_vlm",
                    "description": obj.get("description"),
                    "confidence": as_float(obj.get("confidence")),
                    "physical_notes": obj.get("physical_notes"),
                }
            )
    return measurements, roster_rows, sources


def local_mask_path(mask_path: Any, track_json: Path) -> str | None:
    if not isinstance(mask_path, str) or not mask_path:
        return None
    path = Path(mask_path)
    if path.exists():
        return str(path)
    candidate = track_json.parent / "sam2_masks" / path.name
    return str(candidate) if candidate.exists() else str(path)


def measurements_from_sam2_multiobject_roots(roots: tuple[Path, ...]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for root_i, root in enumerate(roots):
        summary_path = root / "qc_sam2_multiobject_points.json"
        if not summary_path.exists():
            sources.append({"path": str(root), "status": "missing_summary"})
            continue
        summary = load_json(summary_path)
        track_ids = summary.get("track_ids") or []
        if not isinstance(track_ids, list):
            sources.append({"path": str(root), "status": "invalid_track_ids"})
            continue
        sources.append(
            {
                "path": str(root),
                "status": summary.get("status", "loaded"),
                "backend": summary.get("backend"),
                "track_count": len(track_ids),
                "frame_start": summary.get("frame_start"),
                "frame_end": summary.get("frame_end"),
                "frames": summary.get("frames"),
            }
        )
        for track_id in track_ids:
            track_json = root / str(track_id) / "sam2" / "sam2_track.json"
            qc_json = root / str(track_id) / "sam2" / "qc_sam2_vlm_points_track.json"
            if not track_json.exists():
                continue
            track_payload = load_json(track_json)
            track_qc = load_json(qc_json) if qc_json.exists() else {}
            for frame_key, row in track_payload.items():
                if not isinstance(row, dict):
                    continue
                idx = int(frame_key)
                visible = bool(row.get("visible", False))
                bbox = compact_bbox(row.get("bbox_xyxy"))
                entry = {
                    "measurement_id": f"sam2:{root_i}:{track_id}:{idx}",
                    "frame_idx": idx,
                    "entity_type": "object",
                    "entity_id": f"object:{track_id}",
                    "measurement_type": "sam2_video_mask_track",
                    "source_model": summary.get("backend") or "SAM2",
                    "coordinate_frame": "source_image_pixels",
                    "confidence": None,
                    "visible": visible,
                    "bbox_xyxy": bbox,
                    "bbox_area_px2": bbox_area(bbox),
                    "center_xy": row.get("center_xy") if visible else None,
                    "mask_area_px": as_float(row.get("area_px")),
                    "mask_path": local_mask_path(row.get("mask_path"), track_json) if visible else None,
                    "prompt_frames": track_qc.get("prompt_frames"),
                    "prompt_contract_reports": len(track_qc.get("prompt_contract_reports") or []),
                    "failure_reason": None if visible else "sam2_track_not_visible",
                }
                measurements.append(entry)
                if visible:
                    by_frame.setdefault(idx, []).append(entry)
    return measurements, by_frame, sources


def measurements_from_expected_object_coverage(coverage_paths: tuple[Path, ...]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    by_expected: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for source_i, path in enumerate(coverage_paths):
        if not path.exists():
            sources.append({"path": str(path), "status": "missing"})
            continue
        payload = load_json(path)
        rows = payload.get("coverage")
        if not isinstance(rows, list):
            sources.append({"path": str(path), "status": "invalid_no_coverage"})
            continue
        sources.append(
            {
                "path": str(path),
                "status": payload.get("status", "loaded"),
                "backend": payload.get("backend"),
                "model": payload.get("model"),
                "coverage_count": len(rows),
            }
        )
        for row_i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            expected_label = str(row.get("expected_label") or "")
            if not expected_label:
                continue
            status = str(row.get("coverage_status") or "missing")
            covered_by = [str(x) for x in row.get("covered_by_object_ids") or []]
            entry = {
                "measurement_id": f"expected_object_coverage:{source_i}:{row_i}",
                "frame_idx": None,
                "entity_type": "object",
                "entity_id": f"expected_object:{expected_label}",
                "measurement_type": "vlm_expected_object_coverage",
                "source_model": payload.get("model") or payload.get("backend") or "vlm_expected_object_coverage",
                "coordinate_frame": "object_roster_semantics",
                "confidence": None,
                "expected_label": expected_label,
                "coverage_status": status,
                "covered_by_object_ids": covered_by,
                "reason": row.get("reason"),
                "failure_reason": None if status in {"covered", "ambiguous"} else "expected_object_missing_from_vlm_plan",
            }
            measurements.append(entry)
            by_expected[expected_label] = entry
    return measurements, by_expected, sources


def measurements_from_contact_measurements(paths: tuple[Path, ...]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            sources.append({"path": str(path), "status": "missing"})
            continue
        payload = load_json(path)
        if not isinstance(payload, list):
            sources.append({"path": str(path), "status": "invalid_payload"})
            continue
        state_counts: dict[str, int] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            entry = dict(row)
            idx = int(entry["frame_idx"])
            entry["entity_type"] = "contact"
            entry["measurement_type"] = entry.get("measurement_type") or "hand_object_contact_evidence"
            entry["source_file"] = str(path)
            measurements.append(entry)
            by_frame.setdefault(idx, []).append(entry)
            state = str(entry.get("contact_state_measurement"))
            state_counts[state] = state_counts.get(state, 0) + 1
        sources.append({"path": str(path), "status": "loaded", "measurement_count": len(payload), "contact_state_counts": state_counts})
    return measurements, by_frame, sources


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


def merge_object_rosters(v16_roster: list[dict[str, Any]], plan_roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in v16_roster + plan_roster:
        object_id = str(row["object_id"])
        if object_id not in merged:
            merged[object_id] = dict(row)
            continue
        current = merged[object_id]
        current["source"] = "+".join(sorted(set(str(x) for x in [current.get("source"), row.get("source")] if x)))
        for key in ("active_frame_min", "active_frame_max"):
            a = current.get(key)
            b = row.get(key)
            if a is None:
                current[key] = b
            elif b is not None:
                current[key] = min(a, b) if key.endswith("_min") else max(a, b)
        current["active_frame_count"] = max(int(current.get("active_frame_count") or 0), int(row.get("active_frame_count") or 0))
        if current.get("role_status") == "expected_missing_from_v16":
            current["role_status"] = row.get("role_status", current["role_status"])
    return sorted(merged.values(), key=lambda row: str(row["object_id"]))


def apply_expected_object_coverage(roster: list[dict[str, Any]], coverage: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_object_id = {str(row["object_id"]): dict(row) for row in roster}
    for expected_label, row in coverage.items():
        expected_id = f"object:{expected_label}"
        status = row.get("coverage_status")
        covered_by = row.get("covered_by_object_ids") or []
        if expected_id not in by_object_id:
            by_object_id[expected_id] = {
                "object_id": expected_id,
                "name": expected_label,
                "source": "expected_object_coverage",
                "active_frame_min": None,
                "active_frame_max": None,
                "active_frame_count": 0,
                "role_status": "expected_missing_from_v16",
            }
        target = by_object_id[expected_id]
        target["expected_coverage_status"] = status
        target["expected_covered_by_object_ids"] = [f"object:{obj_id}" for obj_id in covered_by]
        target["expected_coverage_reason"] = row.get("reason")
        if status == "covered":
            target["role_status"] = "covered_by_vlm_plan"
        elif status == "ambiguous" and target.get("role_status") == "expected_missing_from_v16":
            target["role_status"] = "ambiguous_vlm_plan_coverage"
    return sorted(by_object_id.values(), key=lambda row: str(row["object_id"]))


def anchor_qc(
    spec: CaseSpec,
    frames: dict[int, dict[str, Any]],
    wilor_by_frame: dict[int, list[dict[str, Any]]],
    hawor_by_frame: dict[int, list[dict[str, Any]]],
    hamer_by_frame: dict[int, list[dict[str, Any]]],
    v16_hand_by_frame: dict[int, list[dict[str, Any]]],
    rtmlib_by_frame: dict[int, list[dict[str, Any]]],
    object_by_frame: dict[int, list[dict[str, Any]]],
    contact_by_frame: dict[int, list[dict[str, Any]]],
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
        observed_hawor = [row for row in hawor_hands if row.get("measurement_available")]
        hamer_hands = hamer_by_frame.get(idx, [])
        measured_hamer = [row for row in hamer_hands if row.get("measurement_available")]
        hamer_measured_crop_ids = {
            row.get("detector_hand_idx") if row.get("detector_hand_idx") is not None else row.get("measurement_id")
            for row in measured_hamer
        }
        rtmlib_hands = rtmlib_by_frame.get(idx, [])
        object_measurements = object_by_frame.get(idx, [])
        contact_measurements = contact_by_frame.get(idx, [])
        contact_states = sorted({str(row.get("contact_state_measurement")) for row in contact_measurements})
        expected_visible = spec.expected_visible_hands.get(idx)
        obj = frame.get("object", {})
        object_status = obj.get("status")
        failures = []
        if expected_visible is not None and len(v16_hands) < expected_visible:
            failures.append("visible_hands_missing_from_v16_state")
        if v16_hands and any(row.get("failure_reason") == "missing_source_confidence" for row in v16_hands):
            failures.append("v16_hand_state_lacks_source_confidence")
        if v16_hands and any(row.get("failure_reason") == "hand_measurement_unavailable" for row in v16_hands):
            failures.append("v16_hand_state_contains_unavailable_measurement")
        mano_source_count = max(len(wilor_hands), len(observed_hawor), len(hamer_measured_crop_ids))
        if expected_visible is not None and mano_source_count < expected_visible:
            failures.append("source_mano_measurements_missing_for_visible_hands")
        if expected_visible is not None and not wilor_hands and not rtmlib_hands and not observed_hawor and not measured_hamer:
            failures.append("visible_hand_detector_measurement_missing")
        if expected_visible is not None and hawor_hands and len(hawor_hands) < expected_visible:
            failures.append("hawor_measurements_incomplete_for_visible_hands")
        if hawor_hands and any(row.get("failure_reason") for row in hawor_hands):
            failures.append("hawor_geometry_without_2d_observation_support")
        if object_status == "outside_semantic_interval" and spec.expected_contact.get(idx):
            failures.append("object_inactive_despite_expected_interaction_context")
        if obj.get("label") in spec.expected_persistent_object_labels and object_measurements:
            has_persistent_state = any(row.get("measurement_type") == "object_persistent_canonical_mesh" for row in object_measurements)
            if not has_persistent_state:
                failures.append("persistent_object_shape_state_missing")
        if spec.expected_contact.get(idx) == "contact" and not object_measurements:
            failures.append("contact_anchor_without_object_mesh_measurement")
        if spec.expected_contact.get(idx) and not contact_measurements:
            failures.append("missing_contact_state_measurement")
        if (
            spec.expected_contact.get(idx) == "contact"
            and contact_measurements
            and "candidate_contact_image_and_metric" not in contact_states
        ):
            failures.append("contact_anchor_lacks_joint_image_metric_support")
        if "contact_evidence_requires_hand_repair" in contact_states:
            failures.append("contact_evidence_requires_hand_repair")
        hand_repair_failures = {
            "visible_hands_missing_from_v16_state",
            "v16_hand_state_contains_unavailable_measurement",
            "source_mano_measurements_missing_for_visible_hands",
            "hawor_geometry_without_2d_observation_support",
            "contact_evidence_requires_hand_repair",
        }
        if idx in spec.expected_hand_repair_frames and not any(failure in hand_repair_failures for failure in failures):
            failures.append("known_v16_hand_failure_needs_repair_state")
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
                "hamer_hand_count": len(hamer_hands),
                "hamer_measured_hand_count": len(measured_hamer),
                "hamer_measured_crop_count": len(hamer_measured_crop_ids),
                "rtmlib_hand2d_count": len(rtmlib_hands),
                "hawor_hand_count": len(hawor_hands),
                "hawor_observed_hand_count": len(observed_hawor),
                "object_status": object_status,
                "object_label": obj.get("label"),
                "object_mesh_measurement_count": len(object_measurements),
                "contact_measurement_count": len(contact_measurements),
                "contact_state_measurements": contact_states,
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
    hamer_measurements, hamer_by_frame, hamer_sources = measurements_from_hamer_summaries(spec.hamer_measurement_paths)
    v16_hand_measurements, v16_hand_by_frame = measurements_from_v16_hands(frames)
    rtmlib_measurements, rtmlib_by_frame, rtmlib_sources = measurements_from_rtmlib(spec.rtmlib_hand2d_paths)
    object_measurements, object_by_frame = measurements_from_object_mesh_qc(object_qc_path, frames)
    object_plan_measurements, plan_roster, object_plan_sources = measurements_from_object_plans(spec.object_plan_paths)
    sam2_measurements, sam2_by_frame, sam2_sources = measurements_from_sam2_multiobject_roots(spec.sam2_multiobject_roots)
    expected_coverage_measurements, expected_coverage, expected_coverage_sources = measurements_from_expected_object_coverage(
        spec.expected_object_coverage_paths
    )
    contact_measurements, contact_by_frame, contact_sources = measurements_from_contact_measurements(spec.contact_measurement_paths)
    object_by_frame_combined = {idx: list(rows) for idx, rows in object_by_frame.items()}
    for idx, rows in sam2_by_frame.items():
        object_by_frame_combined.setdefault(idx, []).extend(rows)
    roster = apply_expected_object_coverage(
        merge_object_rosters(object_roster_from_v16(frames, spec.expected_objects), plan_roster),
        expected_coverage,
    )

    case_dir = output_root / spec.name
    measurements_dir = case_dir / "measurements_v17"
    write_json(measurements_dir / "wilor_measurements.json", wilor_measurements)
    write_json(measurements_dir / "hawor_measurements.json", hawor_measurements)
    write_json(measurements_dir / "hamer_measurements.json", hamer_measurements)
    write_json(measurements_dir / "v16_hand_state_measurements.json", v16_hand_measurements)
    write_json(measurements_dir / "rtmlib_hand2d_measurements.json", rtmlib_measurements)
    write_json(measurements_dir / "object_mesh_measurements.json", object_measurements)
    write_json(measurements_dir / "object_plan_measurements.json", object_plan_measurements)
    write_json(measurements_dir / "expected_object_coverage_measurements.json", expected_coverage_measurements)
    write_json(measurements_dir / "sam2_object_mask_measurements.json", sam2_measurements)
    write_json(measurements_dir / "contact_measurements.json", contact_measurements)
    write_json(case_dir / "object_roster_v17.json", roster)
    anchor = anchor_qc(
        spec,
        frames,
        wilor_by_frame,
        hawor_by_frame,
        hamer_by_frame,
        v16_hand_by_frame,
        rtmlib_by_frame,
        object_by_frame_combined,
        contact_by_frame,
        roster,
    )
    write_json(case_dir / "v17_anchor_qc.json", anchor)

    report = {
        "case": spec.name,
        "status": anchor["status"],
        "v16_root": str(spec.root),
        "manifest": str(manifest_path),
        "annotations": str(annotations_path),
        "wilor_raw": str(raw_wilor_path),
        "hawor_sources": hawor_sources,
        "hamer_sources": hamer_sources,
        "rtmlib_hand2d_sources": rtmlib_sources,
        "object_plan_sources": object_plan_sources,
        "expected_object_coverage_sources": expected_coverage_sources,
        "sam2_multiobject_sources": sam2_sources,
        "contact_measurement_sources": contact_sources,
        "object_mesh_qc": str(object_qc_path),
        "measurement_counts": {
            "wilor": len(wilor_measurements),
            "hawor": len(hawor_measurements),
            "hamer": len(hamer_measurements),
            "v16_hand_state": len(v16_hand_measurements),
            "rtmlib_hand2d": len(rtmlib_measurements),
            "object_mesh": len(object_measurements),
            "object_plan": len(object_plan_measurements),
            "expected_object_coverage": len(expected_coverage_measurements),
            "sam2_object_mask": len(sam2_measurements),
            "contact": len(contact_measurements),
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
            expected_hand_repair_frames=(182, 260, 856, 949, 970),
            hawor_annotation_paths=(
                Path(
                    "/data2/ego_annotation_outputs/representative_trash/"
                    "v3_hawor_camera_local_840_930/annotations_hawor_camera_local.json"
                ),
            ),
            object_plan_paths=(
                Path("/data2/ego_annotation_outputs/representative_trash/v2_object_plan/object_plan_vlm.json"),
            ),
            expected_object_coverage_paths=(
                Path("/data2/ego_annotation_outputs/v17_object_plan/trash_1050/expected_object_coverage_vlm.json"),
            ),
            sam2_multiobject_roots=(
                Path("/data2/ego_annotation_outputs/representative_trash/v3_contact_surface_sam2_multi_840_930"),
                Path("/data2/ego_annotation_outputs/v17_object_plan/trash_1050/sam2_multiobject_full"),
            ),
            contact_measurement_paths=(
                Path("/data2/ego_annotation_outputs/v17_contact_measurements/trash_1050/contact_measurements_anchor.json"),
            ),
            rtmlib_hand2d_paths=(
                Path("/data2/ego_annotation_outputs/v17_hand_evidence/trash_1050/rtmlib_full_r2/rtmlib_hand2d.json"),
            ),
            hamer_measurement_paths=(
                Path("/data2/ego_annotation_outputs/v17_hand_evidence/trash_1050/hamer_full_r1/hamer_measurements_summary.json"),
            ),
        ),
        CaseSpec(
            name="task5_tomato_960",
            root=Path("/data2/ego_annotation_outputs/v16_full_pipeline/task5_tomato_960"),
            anchor_frames=(480, 720, 760),
            expected_visible_hands={480: 2, 720: 2, 760: 2},
            expected_contact={480: "contact", 720: "contact", 760: "contact"},
            expected_objects=("tomato", "bowl", "plate", "tray"),
            expected_persistent_object_labels=("tomato",),
            object_plan_paths=(
                Path("/data2/ego_annotation_outputs/v17_object_plan/task5_tomato_960/object_plan_vlm.json"),
            ),
            expected_object_coverage_paths=(
                Path("/data2/ego_annotation_outputs/v17_object_plan/task5_tomato_960/expected_object_coverage_vlm.json"),
            ),
            sam2_multiobject_roots=(
                Path("/data2/ego_annotation_outputs/v17_object_plan/task5_tomato_960/sam2_multiobject_interval"),
            ),
            contact_measurement_paths=(
                Path("/data2/ego_annotation_outputs/v17_contact_measurements/task5_tomato_960/contact_measurements_anchor.json"),
            ),
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
