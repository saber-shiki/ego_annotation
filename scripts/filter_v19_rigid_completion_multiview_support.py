#!/usr/bin/env python3
"""Classify P13 hidden-completion faces using trusted P14 multi-view evidence.

The P13 completed mesh remains a pose/render hypothesis. Generated hidden faces
enter the collision-eligible surface only when they receive pose-conditioned
visible-depth support in multiple trusted frames *and* trusted poses cover the
timeline. Sparse, clustered, interpolated, or nearest-held poses never create
hidden-geometry support.

This stage also requires model first-hit visibility before calling a face sample
visible; a self-occluded generated face cannot gain support merely because its
projection shares a pixel/depth with the front surface.

This stage consumes prediction measurements only. Released benchmark masks or
other evaluation-only ground truth are not inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import trimesh

OBSERVED_LABEL = "observed_depth_surface"
HIDDEN_LABEL = "trellis_inferred_hidden_surface"
SUPPORTED_HIDDEN_LABEL = "generated_hidden_surface_multiview_visible_depth_supported"
STATE_OBSERVED = 0
STATE_SUPPORTED_CANDIDATE = 1
STATE_FREE_SPACE_CONTRADICTED = 2
STATE_UNSUPPORTED = 3
STATE_REPEATED_SAME_VIEW_SUPPORT = 4
STATE_CONFLICTING_SUPPORT_AND_FREE_SPACE = 5


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_file_path(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def load_mesh(path: Path) -> trimesh.Trimesh:
    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        meshes = [item for item in geometry.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh geometry in {path}")
        geometry = trimesh.util.concatenate(meshes)
    if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return trimesh.Trimesh(
        vertices=np.asarray(geometry.vertices, dtype=np.float64),
        faces=np.asarray(geometry.faces, dtype=np.int64),
        process=False,
    )


def load_face_labels(path: Path, expected_count: int) -> list[str]:
    payload = load_json(path)
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, list) or len(labels) != expected_count:
        raise RuntimeError(f"face-label count mismatch: {path}: {0 if not isinstance(labels, list) else len(labels)} != {expected_count}")
    return [str(value) for value in labels]


def object_row(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for candidate in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(candidate, dict) and str(candidate.get("object_id")) == object_id:
            return candidate
    return None


def trusted_pose_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in report.get("pose_rows", []) if isinstance(report.get("pose_rows"), list) else []:
        if not isinstance(row, dict):
            continue
        if str(row.get("status")) != "fit_to_visible_depth_samples":
            continue
        if row.get("rigid_pose_observation_eligible") is not True:
            continue
        rotation = np.asarray(row.get("rotation_world_from_completed_canonical_matrix") or [], dtype=np.float64)
        translation = np.asarray(row.get("translation_world_m") or [], dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            continue
        rows.append(row)
    return sorted(rows, key=lambda item: int(item["frame_idx"]))


def load_depth_sources(paths: list[Path]) -> dict[int, tuple[np.ndarray, np.ndarray, str]]:
    rows: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        if "frame_idx" not in archive or "depth" not in archive:
            raise RuntimeError(f"depth archive lacks frame_idx/depth: {path}")
        frame_ids = np.asarray(archive["frame_idx"], dtype=np.int64)
        depths = np.asarray(archive["depth"])
        intrinsics = np.asarray(archive["intrinsics_fx_fy_cx_cy"]) if "intrinsics_fx_fy_cx_cy" in archive else None
        for position, frame_idx in enumerate(frame_ids.tolist()):
            if frame_idx in rows:
                continue
            intr = (
                np.asarray(intrinsics[position], dtype=np.float64)
                if intrinsics is not None and intrinsics.ndim == 2
                else np.asarray([], dtype=np.float64)
            )
            rows[int(frame_idx)] = (np.asarray(depths[position], dtype=np.float32), intr, str(path))
    return rows


def camera_contract(
    frame: dict[str, Any],
    obj: dict[str, Any],
    depth_intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    transform = np.asarray(camera.get("T_world_camera_metric") or camera.get("T_world_camera") or [], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks valid metric camera pose")
    geometry = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
    camera_intrinsics = np.asarray(
        camera.get("intrinsics_fx_fy_cx_cy") or geometry.get("intrinsics_fx_fy_cx_cy") or [],
        dtype=np.float64,
    )
    depth_intrinsics = np.asarray(depth_intrinsics, dtype=np.float64)
    depth_valid = depth_intrinsics.shape == (4,) and np.isfinite(depth_intrinsics).all() and np.all(depth_intrinsics[:2] > 0.0)
    camera_valid = camera_intrinsics.shape == (4,) and np.isfinite(camera_intrinsics).all() and np.all(camera_intrinsics[:2] > 0.0)
    if depth_valid:
        intrinsics = depth_intrinsics
        source = "depth_npz_intrinsics_for_depth_grid_projection"
    elif camera_valid:
        intrinsics = camera_intrinsics
        source = "annotation_camera_intrinsics_fallback_depth_npz_missing"
    else:
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks valid depth-grid/camera intrinsics")
    metadata = {
        "selected_intrinsics_fx_fy_cx_cy": intrinsics.astype(float).tolist(),
        "selected_intrinsics_source": source,
        "depth_npz_intrinsics_fx_fy_cx_cy": depth_intrinsics.astype(float).tolist() if depth_valid else None,
        "annotation_camera_intrinsics_fx_fy_cx_cy": camera_intrinsics.astype(float).tolist() if camera_valid else None,
        "depth_vs_annotation_intrinsics_delta": (
            (depth_intrinsics - camera_intrinsics).astype(float).tolist()
            if depth_valid and camera_valid
            else None
        ),
        "claim_scope": "Projection into a depth NPZ raster uses that NPZ row's intrinsics. Annotation-camera intrinsics are fallback only and are never silently substituted when depth-grid intrinsics exist.",
    }
    return transform, intrinsics, metadata


def face_sample_points(mesh: trimesh.Trimesh) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    centers = triangles.mean(axis=1, keepdims=True)
    return np.concatenate([triangles, centers], axis=1)


def temporal_coverage(frame_ids: list[int], timeline_start: int, timeline_count: int, bin_count: int) -> dict[str, Any]:
    if not frame_ids:
        return {
            "frame_count": 0,
            "timeline_frame_span": [int(timeline_start), int(timeline_start + max(0, timeline_count - 1))],
            "frame_span": None,
            "frame_span_fraction": 0.0,
            "occupied_temporal_bins": [],
            "occupied_temporal_bin_count": 0,
            "unobserved_gap_frames": [],
            "maximum_unobserved_gap_frames": timeline_count,
            "maximum_unobserved_gap_fraction": 1.0,
        }
    sorted_ids = sorted(set(frame_ids))
    occupied = sorted(
        {
            min(bin_count - 1, max(0, (frame - timeline_start) * bin_count // max(1, timeline_count)))
            for frame in sorted_ids
        }
    )
    span = max(sorted_ids) - min(sorted_ids)
    gaps = [max(0, sorted_ids[0] - timeline_start)]
    gaps.extend(max(0, right - left - 1) for left, right in zip(sorted_ids[:-1], sorted_ids[1:]))
    gaps.append(max(0, timeline_start + timeline_count - 1 - sorted_ids[-1]))
    max_gap = max(gaps) if gaps else timeline_count
    return {
        "frame_count": len(sorted_ids),
        "timeline_frame_span": [int(timeline_start), int(timeline_start + timeline_count - 1)],
        "frame_span": [min(sorted_ids), max(sorted_ids)],
        "frame_span_fraction": float(span / max(1, timeline_count - 1)),
        "occupied_temporal_bins": occupied,
        "occupied_temporal_bin_count": len(occupied),
        "unobserved_gap_frames": gaps,
        "maximum_unobserved_gap_frames": int(max_gap),
        "maximum_unobserved_gap_fraction": float(max_gap / max(1, timeline_count)),
    }


def viewpoint_diversity(
    pose_rows: list[dict[str, Any]],
    frames: dict[int, dict[str, Any]],
    object_center_canonical: np.ndarray,
    object_diameter: float,
    bin_separation_deg: float,
) -> tuple[dict[str, Any], list[int], list[tuple[int, int]]]:
    directions: list[np.ndarray] = []
    camera_centers_canonical: list[np.ndarray] = []
    valid_frame_ids: list[int] = []
    for row in pose_rows:
        frame_idx = int(row["frame_idx"])
        frame = frames.get(frame_idx, {})
        camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
        transform = np.asarray(camera.get("T_world_camera_metric") or camera.get("T_world_camera") or [], dtype=np.float64)
        rotation = np.asarray(row.get("rotation_world_from_completed_canonical_matrix") or [], dtype=np.float64)
        translation = np.asarray(row.get("translation_world_m") or [], dtype=np.float64)
        if transform.shape != (4, 4) or rotation.shape != (3, 3) or translation.shape != (3,):
            raise RuntimeError(f"trusted pose frame {frame_idx} lacks camera/object transform for viewpoint diversity")
        camera_canonical = (transform[:3, 3] - translation) @ rotation
        view_vector = camera_canonical - object_center_canonical
        norm = float(np.linalg.norm(view_vector))
        if not np.isfinite(norm) or norm <= 1.0e-9:
            raise RuntimeError(f"trusted pose frame {frame_idx} has degenerate canonical camera viewpoint")
        directions.append(view_vector / norm)
        camera_centers_canonical.append(camera_canonical)
        valid_frame_ids.append(frame_idx)
    pairwise_angles: list[float] = []
    pairwise_baselines: list[float] = []
    separated_pose_row_pairs: list[tuple[int, int]] = []
    for left in range(len(directions)):
        for right in range(left + 1, len(directions)):
            cosine = float(np.clip(np.dot(directions[left], directions[right]), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            pairwise_angles.append(angle)
            if angle >= float(bin_separation_deg):
                separated_pose_row_pairs.append((left, right))
            pairwise_baselines.append(float(np.linalg.norm(camera_centers_canonical[left] - camera_centers_canonical[right])))
    bin_representatives: list[np.ndarray] = []
    bin_ids: list[int] = []
    for direction in directions:
        assigned = None
        for bin_idx, representative in enumerate(bin_representatives):
            angle = float(np.degrees(np.arccos(np.clip(np.dot(direction, representative), -1.0, 1.0))))
            if angle < float(bin_separation_deg):
                assigned = bin_idx
                break
        if assigned is None:
            assigned = len(bin_representatives)
            bin_representatives.append(direction)
        bin_ids.append(int(assigned))
    maximum_baseline = max(pairwise_baselines, default=0.0)
    return {
        "valid_frame_ids": valid_frame_ids,
        "viewpoint_bin_separation_deg": float(bin_separation_deg),
        "viewpoint_bin_ids_by_pose_row": bin_ids,
        "distinct_viewpoint_bin_count": len(bin_representatives),
        "pose_row_pair_count_meeting_viewpoint_separation": len(separated_pose_row_pairs),
        "pairwise_view_angle_deg": {
            "count": len(pairwise_angles),
            "median": float(np.median(pairwise_angles)) if pairwise_angles else None,
            "max": float(max(pairwise_angles)) if pairwise_angles else None,
        },
        "pairwise_camera_baseline_m": {
            "count": len(pairwise_baselines),
            "median": float(np.median(pairwise_baselines)) if pairwise_baselines else None,
            "max": float(maximum_baseline),
        },
        "maximum_camera_baseline_over_object_diameter": float(maximum_baseline / max(object_diameter, 1.0e-9)),
        "claim_scope": "View directions and baselines are measured in the P14 object-canonical frame. Repeated adjacent frames in one angular bin count as one viewpoint, not independent hidden-surface support.",
    }, bin_ids, separated_pose_row_pairs


def faces_with_separated_view_support(
    support_masks_by_pose_row: list[np.ndarray],
    separated_pose_row_pairs: list[tuple[int, int]],
    face_count: int,
) -> np.ndarray:
    if not support_masks_by_pose_row:
        return np.zeros(face_count, dtype=bool)
    packed_support = np.packbits(np.stack(support_masks_by_pose_row, axis=0), axis=1, bitorder="little")
    packed_diverse = np.zeros(packed_support.shape[1], dtype=np.uint8)
    for left, right in separated_pose_row_pairs:
        packed_diverse |= packed_support[left] & packed_support[right]
    return np.unpackbits(packed_diverse, count=face_count, bitorder="little").astype(bool)


def colors_for_states(states: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(states), 4), dtype=np.uint8)
    colors[states == STATE_OBSERVED] = [30, 180, 255, 255]
    colors[states == STATE_SUPPORTED_CANDIDATE] = [40, 220, 80, 255]
    colors[states == STATE_FREE_SPACE_CONTRADICTED] = [255, 35, 35, 210]
    colors[states == STATE_UNSUPPORTED] = [255, 150, 30, 175]
    colors[states == STATE_REPEATED_SAME_VIEW_SUPPORT] = [180, 55, 220, 220]
    colors[states == STATE_CONFLICTING_SUPPORT_AND_FREE_SPACE] = [255, 40, 180, 230]
    return colors


def render_support_qc(
    output_path: Path,
    mesh: trimesh.Trimesh,
    states: np.ndarray,
    pose_rows: list[dict[str, Any]],
    frames: dict[int, dict[str, Any]],
    object_id: str,
    depth_rows: dict[int, tuple[np.ndarray, np.ndarray, str]],
) -> None:
    face_bgr = np.zeros((len(states), 3), dtype=np.uint8)
    face_bgr[states == STATE_OBSERVED] = [255, 210, 30]
    face_bgr[states == STATE_SUPPORTED_CANDIDATE] = [60, 220, 60]
    face_bgr[states == STATE_FREE_SPACE_CONTRADICTED] = [40, 40, 240]
    face_bgr[states == STATE_UNSUPPORTED] = [0, 165, 255]
    face_bgr[states == STATE_REPEATED_SAME_VIEW_SUPPORT] = [220, 55, 180]
    face_bgr[states == STATE_CONFLICTING_SUPPORT_AND_FREE_SPACE] = [180, 40, 255]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    panels: list[np.ndarray] = []
    for pose_row in pose_rows:
        frame_idx = int(pose_row["frame_idx"])
        frame = frames[frame_idx]
        raw_path = Path(str(frame.get("raw_frame_path") or ""))
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read raw QC frame {frame_idx}: {raw_path}")
        obj = object_row(frame, object_id)
        if obj is None:
            raise RuntimeError(f"QC frame {frame_idx} lacks object {object_id}")
        _depth, depth_intrinsics, _source = depth_rows[frame_idx]
        transform, intrinsics, _intrinsics_metadata = camera_contract(frame, obj, depth_intrinsics)
        rotation = np.asarray(pose_row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        translation = np.asarray(pose_row["translation_world_m"], dtype=np.float64)
        world = vertices @ rotation.T + translation[None, :]
        camera = (world - transform[:3, 3][None, :]) @ transform[:3, :3]
        z = camera[:, 2]
        fx, fy, cx, cy = intrinsics.tolist()
        uv = np.full((len(vertices), 2), np.nan, dtype=np.float64)
        positive = np.isfinite(camera).all(axis=1) & (z > 0.01)
        uv[positive, 0] = fx * camera[positive, 0] / z[positive] + cx
        uv[positive, 1] = fy * camera[positive, 1] / z[positive] + cy
        triangles_uv = uv[faces]
        triangles_z = z[faces]
        valid = positive[faces].all(axis=1) & np.isfinite(triangles_uv).all(axis=(1, 2))
        height, width = image.shape[:2]
        intersects = (
            (triangles_uv[:, :, 0].max(axis=1) >= 0)
            & (triangles_uv[:, :, 1].max(axis=1) >= 0)
            & (triangles_uv[:, :, 0].min(axis=1) < width)
            & (triangles_uv[:, :, 1].min(axis=1) < height)
        )
        face_ids = np.where(valid & intersects)[0]
        order = face_ids[np.argsort(np.mean(triangles_z[face_ids], axis=1))[::-1]]
        overlay = image.copy()
        rendered = np.zeros((height, width), dtype=np.uint8)
        for face_idx in order.tolist():
            polygon = np.rint(triangles_uv[face_idx]).astype(np.int32)
            polygon[:, 0] = np.clip(polygon[:, 0], -2 * width, 3 * width)
            polygon[:, 1] = np.clip(polygon[:, 1], -2 * height, 3 * height)
            color = tuple(int(value) for value in face_bgr[face_idx])
            cv2.fillConvexPoly(overlay, polygon, color, lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(rendered, polygon, 255, lineType=cv2.LINE_8)
        alpha = 0.62
        composed = image.copy()
        mask = rendered > 0
        composed[mask] = cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0.0)[mask]
        cv2.rectangle(composed, (0, 0), (width, 72), (0, 0, 0), -1)
        cv2.putText(composed, f"trusted direct pose frame {frame_idx}", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(composed, "cyan=observed green=multi-view purple=same-view red=contradicted pink=conflict orange=unsupported", (8, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(cv2.resize(composed, (480, 480), interpolation=cv2.INTER_AREA))
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    sheet = np.full((rows * 480, columns * 480, 3), 238, dtype=np.uint8)
    for index, panel in enumerate(panels):
        y = (index // columns) * 480
        x = (index % columns) * 480
        sheet[y : y + 480, x : x + 480] = panel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])


def label_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--completion-report", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, action="append", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-dilate-px", type=int, default=4)
    parser.add_argument("--depth-front-tolerance-m", type=float, default=0.008)
    parser.add_argument("--depth-surface-tolerance-m", type=float, default=0.010)
    parser.add_argument("--model-frontmost-tolerance-m", type=float, default=0.003)
    parser.add_argument("--min-face-samples-per-frame", type=int, default=2)
    parser.add_argument("--min-visible-support-frames", type=int, default=2)
    parser.add_argument("--min-free-space-contradiction-frames", type=int, default=2)
    parser.add_argument("--min-trusted-pose-frames-for-promotion", type=int, default=8)
    parser.add_argument("--temporal-bin-count", type=int, default=5)
    parser.add_argument("--min-occupied-temporal-bins-for-promotion", type=int, default=3)
    parser.add_argument("--min-pose-frame-span-fraction-for-promotion", type=float, default=0.5)
    parser.add_argument("--max-trusted-pose-gap-fraction-for-promotion", type=float, default=0.35)
    parser.add_argument("--viewpoint-bin-separation-deg", type=float, default=15.0)
    parser.add_argument("--render-qc", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= int(args.min_face_samples_per_frame) <= 4:
        raise ValueError("--min-face-samples-per-frame must be in [1,4]")
    if int(args.min_visible_support_frames) < 1 or int(args.min_free_space_contradiction_frames) < 1:
        raise ValueError("support/contradiction frame thresholds must be positive")
    if int(args.temporal_bin_count) < 1:
        raise ValueError("--temporal-bin-count must be positive")
    if not 0.0 < float(args.viewpoint_bin_separation_deg) <= 180.0:
        raise ValueError("--viewpoint-bin-separation-deg must be in (0,180]")
    if float(args.depth_front_tolerance_m) < 0.0 or float(args.depth_surface_tolerance_m) < 0.0 or float(args.model_frontmost_tolerance_m) < 0.0:
        raise ValueError("depth/model-frontmost tolerances must be non-negative")
    annotations = load_json(args.annotations)
    completion = load_json(args.completion_report)
    pose_report = load_json(args.pose_report)
    frames = {
        int(frame["frame_idx"]): frame
        for frame in annotations.get("frames", [])
        if isinstance(frame, dict) and "frame_idx" in frame
    }
    timeline_start = min(frames) if frames else 0
    timeline_count = max(frames) - timeline_start + 1 if frames else 0
    pose_rows = trusted_pose_rows(pose_report)
    if not pose_rows:
        raise RuntimeError("no explicitly eligible direct P14 pose rows for multi-view support")

    outputs = completion.get("outputs") if isinstance(completion.get("outputs"), dict) else {}
    pose_mesh_path = Path(outputs.get("pose_hypothesis_mesh_labeled") or outputs.get("completed_mesh_labeled") or "")
    labels_path = Path(outputs.get("completed_face_labels") or "")
    if not pose_mesh_path.is_file() or not labels_path.is_file():
        raise RuntimeError("completion report lacks pose hypothesis mesh or completed face labels")
    mesh = load_mesh(pose_mesh_path)
    pose_inputs = pose_report.get("inputs") if isinstance(pose_report.get("inputs"), dict) else {}
    pose_input_mesh_value = pose_inputs.get("completed_mesh") or pose_inputs.get("pose_hypothesis_mesh")
    if not pose_input_mesh_value:
        raise RuntimeError("P14 pose report does not bind its fits to a pose-hypothesis mesh")
    pose_input_mesh = Path(str(pose_input_mesh_value))
    if not same_file_path(pose_input_mesh, pose_mesh_path):
        raise RuntimeError(
            "P14 pose/completion canonical-frame mismatch: pose report was fit against "
            f"{pose_input_mesh}, but completion pose hypothesis is {pose_mesh_path}"
        )
    pose_object_id = pose_report.get("object_id")
    if pose_object_id is not None and str(pose_object_id) != str(args.object_id):
        raise RuntimeError(f"P14 pose object mismatch: {pose_object_id} != {args.object_id}")
    pose_annotations_value = pose_inputs.get("annotations")
    if pose_annotations_value and not same_file_path(Path(str(pose_annotations_value)), args.annotations):
        raise RuntimeError(
            f"P14 pose/annotation mismatch: pose report used {pose_annotations_value}, current annotations are {args.annotations}"
        )
    labels = load_face_labels(labels_path, len(mesh.faces))
    observed_faces = np.asarray([label == OBSERVED_LABEL for label in labels], dtype=bool)
    hidden_faces = np.asarray([label == HIDDEN_LABEL for label in labels], dtype=bool)
    unknown_labels = sorted({label for label in labels if label not in {OBSERVED_LABEL, HIDDEN_LABEL}})
    if unknown_labels:
        raise RuntimeError(f"unsupported pose-hypothesis labels: {unknown_labels}")
    if not np.any(observed_faces) or not np.any(hidden_faces):
        raise RuntimeError("multi-view support requires both observed and generated hidden faces")

    depth_rows = load_depth_sources([path.resolve() for path in args.depth_npz])
    samples = face_sample_points(mesh)
    face_count, samples_per_face = samples.shape[:2]
    flat_samples = samples.reshape(-1, 3)
    visibility_scene = o3d.t.geometry.RaycastingScene()
    visibility_scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    observed_visibility_mesh = mesh.submesh([np.where(observed_faces)[0]], append=True, repair=False)
    observed_visibility_scene = o3d.t.geometry.RaycastingScene()
    observed_visibility_scene.add_triangles(
        o3d.core.Tensor(np.asarray(observed_visibility_mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(observed_visibility_mesh.faces, dtype=np.uint32)),
    )
    sample_from_observed_face = np.repeat(observed_faces, samples_per_face)
    object_center_canonical = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    object_diameter = float(np.linalg.norm(np.asarray(mesh.extents, dtype=np.float64)))
    viewpoint_info, viewpoint_bin_ids, separated_pose_row_pairs = viewpoint_diversity(
        pose_rows,
        frames,
        object_center_canonical,
        object_diameter,
        float(args.viewpoint_bin_separation_deg),
    )
    valid_frame_count = np.zeros(face_count, dtype=np.uint16)
    visible_support_frame_count = np.zeros(face_count, dtype=np.uint16)
    free_space_contradiction_frame_count = np.zeros(face_count, dtype=np.uint16)
    inside_mask_frame_count = np.zeros(face_count, dtype=np.uint16)
    frame_reports: list[dict[str, Any]] = []
    support_masks_by_pose_row: list[np.ndarray] = []

    for pose_position, pose_row in enumerate(pose_rows):
        frame_idx = int(pose_row["frame_idx"])
        frame = frames.get(frame_idx)
        if frame is None or frame_idx not in depth_rows:
            raise RuntimeError(f"trusted pose frame {frame_idx} lacks annotation/depth")
        obj = object_row(frame, args.object_id)
        if obj is None:
            raise RuntimeError(f"trusted pose frame {frame_idx} lacks object {args.object_id}")
        mask_path = Path(str(obj.get("mask_path") or ""))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot read object-owned mask at frame {frame_idx}: {mask_path}")
        mask_bool = mask > 0
        if int(args.mask_dilate_px) > 0:
            size = 2 * int(args.mask_dilate_px) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask_bool = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0
        depth, depth_intrinsics, depth_source = depth_rows[frame_idx]
        if depth.shape != mask_bool.shape:
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        transform, intrinsics, intrinsics_metadata = camera_contract(frame, obj, depth_intrinsics)
        rotation = np.asarray(pose_row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        translation = np.asarray(pose_row["translation_world_m"], dtype=np.float64)
        points_world = flat_samples @ rotation.T + translation[None, :]
        points_camera = (points_world - transform[:3, 3][None, :]) @ transform[:3, :3]
        z = points_camera[:, 2]
        fx, fy, cx, cy = intrinsics.tolist()
        u = fx * points_camera[:, 0] / np.maximum(z, 1.0e-12) + cx
        v = fy * points_camera[:, 1] / np.maximum(z, 1.0e-12) + cy
        x = np.rint(u).astype(np.int64)
        y = np.rint(v).astype(np.int64)
        in_image = (
            np.isfinite(points_camera).all(axis=1)
            & (z > 0.01)
            & (x >= 0)
            & (x < depth.shape[1])
            & (y >= 0)
            & (y < depth.shape[0])
        )
        sampled_depth = np.full(len(flat_samples), np.nan, dtype=np.float64)
        sampled_mask = np.zeros(len(flat_samples), dtype=bool)
        sampled_depth[in_image] = depth[y[in_image], x[in_image]].astype(np.float64)
        sampled_mask[in_image] = mask_bool[y[in_image], x[in_image]]
        depth_valid = in_image & np.isfinite(sampled_depth) & (sampled_depth > 0.05)
        camera_origin_canonical = (transform[:3, 3] - translation) @ rotation
        sample_ray = flat_samples - camera_origin_canonical[None, :]
        sample_ray_distance = np.linalg.norm(sample_ray, axis=1)
        valid_ray = np.isfinite(sample_ray).all(axis=1) & (sample_ray_distance > 1.0e-9)
        ray_directions = np.zeros_like(sample_ray)
        ray_directions[valid_ray] = sample_ray[valid_ray] / sample_ray_distance[valid_ray, None]
        rays = np.concatenate(
            [
                np.broadcast_to(camera_origin_canonical[None, :], flat_samples.shape),
                ray_directions,
            ],
            axis=1,
        ).astype(np.float32)
        full_pose_first_hit_distance = visibility_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
        observed_surface_first_hit_distance = observed_visibility_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
        first_hit_distance = np.where(
            sample_from_observed_face,
            observed_surface_first_hit_distance,
            full_pose_first_hit_distance,
        )
        model_frontmost = (
            in_image
            & valid_ray
            & np.isfinite(first_hit_distance)
            & (sample_ray_distance <= first_hit_distance + float(args.model_frontmost_tolerance_m))
        )
        near_surface = (
            depth_valid
            & model_frontmost
            & sampled_mask
            & (np.abs(z - sampled_depth) <= float(args.depth_surface_tolerance_m))
        )
        free_space = depth_valid & model_frontmost & (z < sampled_depth - float(args.depth_front_tolerance_m))

        depth_valid_face = depth_valid.reshape(face_count, samples_per_face)
        model_frontmost_face = model_frontmost.reshape(face_count, samples_per_face)
        near_face = near_surface.reshape(face_count, samples_per_face)
        free_face = free_space.reshape(face_count, samples_per_face)
        inside_face = (depth_valid & sampled_mask).reshape(face_count, samples_per_face)
        minimum = int(args.min_face_samples_per_frame)
        valid_this = np.count_nonzero(depth_valid_face, axis=1) >= minimum
        model_frontmost_this = np.count_nonzero(model_frontmost_face, axis=1) >= minimum
        support_this = np.count_nonzero(near_face, axis=1) >= minimum
        contradiction_this = np.count_nonzero(free_face, axis=1) >= minimum
        inside_this = np.count_nonzero(inside_face, axis=1) >= minimum
        valid_frame_count += valid_this.astype(np.uint16)
        visible_support_frame_count += support_this.astype(np.uint16)
        free_space_contradiction_frame_count += contradiction_this.astype(np.uint16)
        inside_mask_frame_count += inside_this.astype(np.uint16)
        support_masks_by_pose_row.append(support_this)
        frame_reports.append(
            {
                "frame_idx": frame_idx,
                "viewpoint_bin_id": int(viewpoint_bin_ids[pose_position]),
                "mask_path": str(mask_path),
                "mask_sha256": sha256_file(mask_path),
                "depth_source": depth_source,
                "projection_intrinsics_contract": intrinsics_metadata,
                "valid_projection_face_count": int(np.count_nonzero(valid_this)),
                "model_frontmost_face_count": int(np.count_nonzero(model_frontmost_this)),
                "inside_mask_face_count": int(np.count_nonzero(inside_this)),
                "visible_depth_supported_face_count": int(np.count_nonzero(support_this)),
                "free_space_contradicted_face_count": int(np.count_nonzero(contradiction_this)),
            }
        )

    distinct_support_viewpoint_bin_count = np.zeros(face_count, dtype=np.uint16)
    for bin_id in sorted(set(viewpoint_bin_ids)):
        positions = [position for position, value in enumerate(viewpoint_bin_ids) if value == bin_id]
        supported_in_bin = np.logical_or.reduce([support_masks_by_pose_row[position] for position in positions])
        distinct_support_viewpoint_bin_count += supported_in_bin.astype(np.uint16)
    viewpoint_separated_support = faces_with_separated_view_support(
        support_masks_by_pose_row,
        separated_pose_row_pairs,
        face_count,
    )
    hidden_repeated_frame_support = (
        hidden_faces
        & (visible_support_frame_count >= int(args.min_visible_support_frames))
        & (free_space_contradiction_frame_count == 0)
    )
    hidden_supported_candidate = hidden_repeated_frame_support & viewpoint_separated_support
    hidden_same_view_support_only = hidden_repeated_frame_support & ~hidden_supported_candidate
    hidden_conflicting = hidden_faces & (visible_support_frame_count > 0) & (free_space_contradiction_frame_count > 0)
    observed_free_space_contradicted = observed_faces & (
        free_space_contradiction_frame_count >= int(args.min_free_space_contradiction_frames)
    )
    observed_support_and_free_space_conflicting = observed_faces & (
        (visible_support_frame_count > 0) & (free_space_contradiction_frame_count > 0)
    )
    hidden_contradicted = (
        hidden_faces
        & (free_space_contradiction_frame_count >= int(args.min_free_space_contradiction_frames))
        & (visible_support_frame_count == 0)
    )
    coverage = temporal_coverage(
        [int(row["frame_idx"]) for row in pose_rows],
        timeline_start,
        timeline_count,
        int(args.temporal_bin_count),
    )
    pose_coverage_sufficient = bool(
        coverage["frame_count"] >= int(args.min_trusted_pose_frames_for_promotion)
        and coverage["occupied_temporal_bin_count"] >= int(args.min_occupied_temporal_bins_for_promotion)
        and coverage["frame_span_fraction"] >= float(args.min_pose_frame_span_fraction_for_promotion)
        and coverage["maximum_unobserved_gap_fraction"] <= float(args.max_trusted_pose_gap_fraction_for_promotion)
        and viewpoint_info["pose_row_pair_count_meeting_viewpoint_separation"] >= 1
    )
    promoted_hidden = hidden_supported_candidate & pose_coverage_sufficient
    collision_face_mask = observed_faces | promoted_hidden

    states = np.full(face_count, STATE_UNSUPPORTED, dtype=np.uint8)
    states[observed_faces] = STATE_OBSERVED
    states[hidden_contradicted] = STATE_FREE_SPACE_CONTRADICTED
    states[hidden_conflicting] = STATE_CONFLICTING_SUPPORT_AND_FREE_SPACE
    states[hidden_same_view_support_only] = STATE_REPEATED_SAME_VIEW_SUPPORT
    states[hidden_supported_candidate] = STATE_SUPPORTED_CANDIDATE
    diagnostic_mesh = mesh.copy()
    diagnostic_mesh.visual.face_colors = colors_for_states(states)
    collision_mesh = mesh.submesh([np.where(collision_face_mask)[0]], append=True, repair=False)
    collision_labels = [
        OBSERVED_LABEL if observed_faces[index] else SUPPORTED_HIDDEN_LABEL
        for index in np.where(collision_face_mask)[0].tolist()
    ]
    collision_colors = np.asarray(
        [[30, 180, 255, 255] if label == OBSERVED_LABEL else [40, 220, 80, 255] for label in collision_labels],
        dtype=np.uint8,
    )
    collision_mesh.visual.face_colors = collision_colors
    collision_surface_watertight = bool(collision_mesh.is_watertight)
    collision_surface_winding_consistent = bool(collision_mesh.is_winding_consistent)
    collision_surface_is_volume = bool(collision_mesh.is_volume)
    signed_geometry_ready = bool(
        collision_surface_watertight
        and collision_surface_winding_consistent
        and collision_surface_is_volume
        and pose_coverage_sufficient
        and not np.any(observed_free_space_contradicted)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = args.output_dir / f"{args.object_id}_multiview_face_support_diagnostic_labeled.ply"
    collision_path = args.output_dir / f"{args.object_id}_multiview_collision_eligible_surface_labeled.ply"
    support_npz_path = args.output_dir / "multiview_face_support_state.npz"
    collision_labels_path = args.output_dir / "collision_eligible_surface_face_labels.json"
    qc_path = args.output_dir / "multiview_face_support_qc.jpg"
    diagnostic_mesh.export(str(diagnostic_path))
    collision_mesh.export(str(collision_path))
    np.savez_compressed(
        support_npz_path,
        state_code=states,
        valid_frame_count=valid_frame_count,
        inside_mask_frame_count=inside_mask_frame_count,
        visible_support_frame_count=visible_support_frame_count,
        visible_support_distinct_viewpoint_bin_count=distinct_support_viewpoint_bin_count,
        visible_support_has_separated_viewpoint_pair=viewpoint_separated_support,
        free_space_contradiction_frame_count=free_space_contradiction_frame_count,
        observed_face_mask=observed_faces,
        observed_free_space_contradicted_mask=observed_free_space_contradicted,
        observed_support_and_free_space_conflicting_mask=observed_support_and_free_space_conflicting,
        hidden_face_mask=hidden_faces,
        hidden_supported_candidate_mask=hidden_supported_candidate,
        hidden_repeated_same_view_support_only_mask=hidden_same_view_support_only,
        hidden_conflicting_support_and_free_space_mask=hidden_conflicting,
        hidden_promoted_collision_mask=promoted_hidden,
        hidden_free_space_contradicted_mask=hidden_contradicted,
    )
    write_json(
        collision_labels_path,
        {
            "source_mesh": str(collision_path),
            "face_count": len(collision_labels),
            "label_counts": label_counts(collision_labels),
            "labels": collision_labels,
        },
    )
    if args.render_qc:
        render_support_qc(qc_path, mesh, states, pose_rows, frames, args.object_id, depth_rows)

    output_report = copy.deepcopy(completion)
    output_report["method"] = "filter_v19_rigid_completion_multiview_support"
    output_report["status"] = (
        "ok_multiview_hidden_support_promoted"
        if np.any(promoted_hidden)
        else "ok_hidden_completion_quarantined_insufficient_trusted_pose_coverage"
        if not pose_coverage_sufficient
        else "ok_hidden_completion_quarantined_no_supported_hidden_faces"
    )
    output_report["annotation_ready"] = False
    output_report["claim_scope"] = (
        "P13 generated hidden faces remain a pose/render hypothesis. P14 direct eligible poses condition multi-view mask/depth remeasurement. "
        "Only faces with repeated visible-depth support in distinct object-canonical viewpoint bins and sufficiently distributed trusted poses can enter the unsigned collision surface; "
        "repeated adjacent views, interpolation, nearest hold, and clustered poses do not support hidden geometry."
    )
    output_report["inputs"] = {
        **(output_report.get("inputs") if isinstance(output_report.get("inputs"), dict) else {}),
        "base_completion_report": str(args.completion_report),
        "pose_report": str(args.pose_report),
        "annotations": str(args.annotations),
        "depth_npz": [str(path) for path in args.depth_npz],
    }
    output_outputs = output_report.get("outputs") if isinstance(output_report.get("outputs"), dict) else {}
    output_outputs.update(
        {
            "pose_hypothesis_mesh_labeled": str(pose_mesh_path),
            "completed_mesh_labeled": str(pose_mesh_path),
            "collision_eligible_mesh_labeled": str(collision_path),
            "collision_eligible_face_labels": str(collision_labels_path),
            "multiview_face_support_diagnostic_mesh_labeled": str(diagnostic_path),
            "multiview_face_support_state_npz": str(support_npz_path),
            "multiview_face_support_qc": str(qc_path) if args.render_qc else None,
        }
    )
    output_report["outputs"] = output_outputs
    output_report["provenance_sha256"] = {
        "inputs": {
            "annotations": sha256_file(args.annotations),
            "base_completion_report": sha256_file(args.completion_report),
            "pose_report": sha256_file(args.pose_report),
            "pose_hypothesis_mesh": sha256_file(pose_mesh_path),
            "completed_face_labels": sha256_file(labels_path),
            "depth_npz": {str(path): sha256_file(path) for path in args.depth_npz},
        },
        "outputs": {
            "collision_eligible_mesh": sha256_file(collision_path),
            "collision_eligible_face_labels": sha256_file(collision_labels_path),
            "multiview_face_support_diagnostic_mesh": sha256_file(diagnostic_path),
            "multiview_face_support_state_npz": sha256_file(support_npz_path),
            "multiview_face_support_qc": sha256_file(qc_path) if args.render_qc else None,
        },
    }
    output_report["multiview_support"] = {
        "pose_conditioning": "explicitly eligible direct P14 fits only; no P15 interpolation or nearest hold",
        "trusted_pose_coverage": coverage,
        "viewpoint_diversity": viewpoint_info,
        "pose_coverage_sufficient_for_hidden_collision_promotion": pose_coverage_sufficient,
        "criteria": {
            "mask_dilate_px": int(args.mask_dilate_px),
            "depth_front_tolerance_m": float(args.depth_front_tolerance_m),
            "depth_surface_tolerance_m": float(args.depth_surface_tolerance_m),
            "model_frontmost_tolerance_m": float(args.model_frontmost_tolerance_m),
            "model_frontmost_test": "Open3D canonical-mesh first-hit ray distance from each P14 camera to each face support sample; observed faces are tested against the measured observed surface alone, while generated hidden faces are tested against the full pose hypothesis so hidden prior cannot occlude measured evidence or count self-occluded generated faces as visible",
            "face_samples": ["three triangle vertices", "triangle center"],
            "min_face_samples_per_frame": int(args.min_face_samples_per_frame),
            "min_visible_support_frames": int(args.min_visible_support_frames),
            "min_free_space_contradiction_frames": int(args.min_free_space_contradiction_frames),
            "min_trusted_pose_frames_for_promotion": int(args.min_trusted_pose_frames_for_promotion),
            "temporal_bin_count": int(args.temporal_bin_count),
            "min_occupied_temporal_bins_for_promotion": int(args.min_occupied_temporal_bins_for_promotion),
            "min_pose_frame_span_fraction_for_promotion": float(args.min_pose_frame_span_fraction_for_promotion),
            "max_trusted_pose_gap_fraction_for_promotion": float(args.max_trusted_pose_gap_fraction_for_promotion),
            "viewpoint_bin_separation_deg": float(args.viewpoint_bin_separation_deg),
            "minimum_supporting_viewpoints_per_face": 2,
            "per_face_viewpoint_rule": "support first requires a model-frontmost sample on the depth raster, then at least one pair of supporting direct pose rows with object-canonical view-angle separation >= viewpoint_bin_separation_deg; packed exact pair test prevents greedy-bin boundary promotion",
            "minimum_global_separated_viewpoint_pairs_for_promotion": 1,
        },
        "face_counts": {
            "total": face_count,
            "observed": int(np.count_nonzero(observed_faces)),
            "observed_with_any_visible_depth_support": int(np.count_nonzero(observed_faces & (visible_support_frame_count > 0))),
            "observed_with_repeated_visible_depth_support": int(np.count_nonzero(observed_faces & (visible_support_frame_count >= int(args.min_visible_support_frames)))),
            "observed_free_space_contradicted": int(np.count_nonzero(observed_free_space_contradicted)),
            "observed_support_and_free_space_conflicting": int(np.count_nonzero(observed_support_and_free_space_conflicting)),
            "generated_hidden": int(np.count_nonzero(hidden_faces)),
            "generated_hidden_repeated_frame_support_any_view": int(np.count_nonzero(hidden_repeated_frame_support)),
            "generated_hidden_repeated_same_view_support_only": int(np.count_nonzero(hidden_same_view_support_only)),
            "generated_hidden_distinct_viewpoint_visible_depth_supported_candidate": int(np.count_nonzero(hidden_supported_candidate)),
            "generated_hidden_free_space_contradicted": int(np.count_nonzero(hidden_contradicted)),
            "generated_hidden_conflicting_visible_support_and_free_space": int(np.count_nonzero(hidden_conflicting)),
            "generated_hidden_unsupported": int(
                np.count_nonzero(
                    hidden_faces
                    & ~hidden_supported_candidate
                    & ~hidden_same_view_support_only
                    & ~hidden_contradicted
                    & ~hidden_conflicting
                )
            ),
            "generated_hidden_promoted_to_collision": int(np.count_nonzero(promoted_hidden)),
            "collision_eligible_total": int(np.count_nonzero(collision_face_mask)),
        },
        "frame_reports": frame_reports,
        "caveat": "The support test reuses prediction masks/depth and P14 poses; it is a physical-consistency mechanism, not independent benchmark GT or object SE(3) certification.",
    }
    output_report["face_label_counts"] = {
        **(output_report.get("face_label_counts") if isinstance(output_report.get("face_label_counts"), dict) else {}),
        "collision_eligible_mesh": label_counts(collision_labels),
    }
    output_report["geometry_readiness"] = {
        "pose_hypothesis_available": True,
        "pose_hypothesis_mesh": str(pose_mesh_path),
        "completed_mesh_legacy_field_semantics": "pose_and_render_hypothesis_not_collision_or_signed_geometry",
        "single_view_hidden_prior_independently_supported": False,
        "multiview_hidden_support_evaluated": True,
        "pose_coverage_sufficient_for_hidden_collision_promotion": pose_coverage_sufficient,
        "candidate_multiview_supported_hidden_face_count": int(np.count_nonzero(hidden_supported_candidate)),
        "repeated_same_view_support_hidden_face_count": int(np.count_nonzero(hidden_same_view_support_only)),
        "collision_eligible_hidden_face_count": int(np.count_nonzero(promoted_hidden)),
        "local_observed_collision_surface_available": True,
        "collision_eligible_mesh": str(collision_path),
        "collision_surface_watertight": collision_surface_watertight,
        "collision_surface_winding_consistent": collision_surface_winding_consistent,
        "collision_surface_is_volume": collision_surface_is_volume,
        "observed_surface_repeated_free_space_contradiction_count": int(np.count_nonzero(observed_free_space_contradicted)),
        "signed_geometry_ready": signed_geometry_ready,
        "signed_geometry_readiness_reason": (
            "collision surface is a winding-consistent volume supported by distributed direct poses"
            if signed_geometry_ready
            else "signed geometry requires a watertight winding-consistent volume, distributed direct-pose support, and no repeated free-space contradiction on measured observed faces"
        ),
        "annotation_ready": False,
        "required_next_evidence": (
            "P15 ready trajectory plus signed-geometry validation"
            if pose_coverage_sufficient and np.any(promoted_hidden)
            else "more temporally distributed trusted direct object poses; do not lower support thresholds or count completion rows"
        ),
    }
    output_report["accepted_body_semantics"] = {
        "observed_depth_surface_faces_collision_eligible": int(np.count_nonzero(observed_faces)),
        "generated_hidden_faces_repeated_same_view_support": int(np.count_nonzero(hidden_same_view_support_only)),
        "generated_hidden_faces_candidate_supported_distinct_viewpoints": int(np.count_nonzero(hidden_supported_candidate)),
        "generated_hidden_faces_collision_eligible": int(np.count_nonzero(promoted_hidden)),
        "generated_hidden_faces_quarantined": int(np.count_nonzero(hidden_faces & ~promoted_hidden)),
        "pose_hypothesis_face_count": face_count,
        "collision_eligible_face_count": int(np.count_nonzero(collision_face_mask)),
        "claim_scope": "Pose/render hypothesis is preserved. Physical consumers must use collision_eligible_mesh_labeled; generated hidden faces require repeated visible-depth support in distinct object-canonical viewpoint bins plus distributed trusted direct poses.",
    }
    output_report["mesh_counts"] = {
        **(output_report.get("mesh_counts") if isinstance(output_report.get("mesh_counts"), dict) else {}),
        "multiview_collision_eligible_vertices": int(len(collision_mesh.vertices)),
        "multiview_collision_eligible_faces": int(len(collision_mesh.faces)),
    }
    report_path = args.output_dir / "v19_multiview_supported_completion_report.json"
    write_json(report_path, output_report)
    print(
        json.dumps(
            {
                "status": output_report["status"],
                "report": str(report_path),
                "trusted_pose_coverage": coverage,
                "pose_coverage_sufficient": pose_coverage_sufficient,
                "face_counts": output_report["multiview_support"]["face_counts"],
                "geometry_readiness": output_report["geometry_readiness"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
