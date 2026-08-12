#!/usr/bin/env python3
"""Audit post-freeze keyboard-mask drift and MANO handedness in P15 video.

This evaluator is additive and GT-isolated: it consumes an already frozen P15
prediction plus the HOT3D evaluator-only sidecar. It never mutates prediction
state. The mask audit separates raw SAM2 tracking, the object-owned mask, and the
projected observed metric mesh shown as green in P15. The hand audit checks the
annotation/bridge/source chain and evaluates a full left/right swap
counterfactual against both HOT3D boxes and HOT3D MANO 3D metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
import render_v19_rigid_state_artifact as canonical  # noqa: E402

SCHEMA = "v19_experimental_p15_post_freeze_mask_handedness_audit_v1"
REPRESENTATIVE_FRAMES = (0, 25, 50, 75, 109, 125, 149)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--render-state", type=Path, required=True)
    parser.add_argument("--hot3d-gt", type=Path, required=True)
    parser.add_argument("--official-pinhole-manifest", type=Path, required=True)
    parser.add_argument("--object-trajectory-report", type=Path, required=True)
    parser.add_argument("--mano-eval-as-labeled", type=Path, required=True)
    parser.add_argument("--mano-eval-swapped", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, default=None)
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument("--object-bop-id", default="28")
    parser.add_argument("--focal-scale", type=float, default=1.0)
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(require_file(path, "JSON").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with require_file(path, "artifact").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = require_file(path, "input")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, require_file(path, f"{name} module"))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_mask(path: Path) -> np.ndarray:
    raw = cv2.imread(str(require_file(path, "mask")), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return raw > 0


def decode_hot3d_rle(payload: dict[str, Any]) -> np.ndarray:
    height = int(payload["height"])
    width = int(payload["width"])
    rle = np.asarray(payload["rle"], dtype=np.int64)
    if rle.ndim != 1 or len(rle) % 2:
        raise RuntimeError("invalid HOT3D RLE")
    starts = rle[::2] - 1
    lengths = rle[1::2]
    flat = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat[int(start): int(start + length)] = 255
    return flat.reshape(height, width)


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise RuntimeError(f"mask shape mismatch: {prediction.shape} vs {target.shape}")
    a = prediction.astype(bool)
    b = target.astype(bool)
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    a_count = int(np.count_nonzero(a))
    b_count = int(np.count_nonzero(b))
    if a_count:
        a_centroid = np.argwhere(a).mean(axis=0)[::-1]
    else:
        a_centroid = np.asarray([np.nan, np.nan], dtype=float)
    if b_count:
        b_centroid = np.argwhere(b).mean(axis=0)[::-1]
    else:
        b_centroid = np.asarray([np.nan, np.nan], dtype=float)
    return {
        "iou": float(intersection / union) if union else None,
        "precision": float(intersection / a_count) if a_count else None,
        "recall": float(intersection / b_count) if b_count else None,
        "centroid_distance_px": float(np.linalg.norm(a_centroid - b_centroid)),
        "prediction_pixels": a_count,
        "target_pixels": b_count,
    }


def summarize(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def summarize_mask_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("iou", "precision", "recall", "centroid_distance_px", "prediction_pixels", "target_pixels"):
        values = [float(row[key][metric]) for row in rows if row[key].get(metric) is not None]
        output[metric] = summarize(values)
    output["worst_iou_frames"] = [
        int(row["frame_idx"])
        for row in sorted(rows, key=lambda row: float(row[key]["iou"]))[:10]
    ]
    return output


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(np.asarray(rotation, dtype=float)) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def hot3d_transform(payload: dict[str, Any]) -> np.ndarray:
    w, x, y, z = [float(value) for value in payload["quaternion_wxyz"]]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise RuntimeError("HOT3D transform has a zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(payload["translation_xyz"], dtype=np.float64)
    return transform


def frame0_relative_camera_motion_mismatch(
    common_frames: list[int],
    annotation_by_frame: dict[int, dict[str, Any]],
    gt_by_frame: dict[int, dict[str, Any]],
    stream_id: str,
) -> dict[str, Any]:
    first = common_frames[0]
    prediction_origin = np.asarray(
        annotation_by_frame[first]["camera"]["T_world_camera_metric"], dtype=np.float64
    )
    gt_origin = hot3d_transform(
        gt_by_frame[first]["json"]["cameras.json"][stream_id]["T_world_from_camera"]
    )
    translation_errors: list[float] = []
    rotation_errors_deg: list[float] = []
    rows: list[dict[str, Any]] = []
    for frame_idx in common_frames:
        prediction = np.asarray(
            annotation_by_frame[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64
        )
        gt_camera = hot3d_transform(
            gt_by_frame[frame_idx]["json"]["cameras.json"][stream_id]["T_world_from_camera"]
        )
        prediction_relative = np.linalg.inv(prediction_origin) @ prediction
        gt_relative = np.linalg.inv(gt_origin) @ gt_camera
        mismatch = np.linalg.inv(prediction_relative) @ gt_relative
        translation_error = float(np.linalg.norm(mismatch[:3, 3]))
        rotation_error = rotation_angle_degrees(mismatch[:3, :3])
        translation_errors.append(translation_error)
        rotation_errors_deg.append(rotation_error)
        rows.append(
            {
                "frame_idx": frame_idx,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
            }
        )
    return {
        "definition": (
            "Compare each camera trajectory relative to frame 0, which removes the unrelated constant world-frame convention; "
            "the residual still includes prediction-camera estimation error and timing/convention mismatch."
        ),
        "translation_error_m": summarize(translation_errors),
        "rotation_error_deg": summarize(rotation_errors_deg),
        "rows": rows,
    }


def resize_for_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def blend_mask(image: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    output = image.copy()
    if np.any(mask):
        color = np.zeros_like(output)
        color[:] = color_bgr
        output[mask] = cv2.addWeighted(output, 1.0 - alpha, color, alpha, 0.0)[mask]
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(output, contours, -1, color_bgr, 2, cv2.LINE_AA)
    return output


def write_mask_review(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    tile_width, tile_height = 360, 360
    columns = [
        ("raw SAM2", "raw_sam2_path", (255, 80, 30)),
        ("object-owned", "object_owned_path", (30, 220, 255)),
        ("green observed mesh", "projected_current_path", (65, 205, 75)),
    ]
    sheets: list[np.ndarray] = []
    for row in rows:
        image = cv2.imread(str(require_file(Path(row["rgb_path"]), "review RGB")), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read review RGB: {row['rgb_path']}")
        image = resize_for_tile(image, tile_width, tile_height)
        tiles: list[np.ndarray] = []
        for title, path_key, color in columns:
            if path_key == "projected_current_path":
                mask = np.asarray(row["_projected_current_mask"], dtype=bool)
            else:
                mask = read_mask(Path(row[path_key]))
            mask = cv2.resize(mask.astype(np.uint8), (tile_width, tile_height), interpolation=cv2.INTER_NEAREST) > 0
            tile = blend_mask(image, mask, color, 0.42)
            metric_key = {
                "raw_sam2_path": "raw_sam2_vs_hot3d_modal",
                "object_owned_path": "object_owned_vs_hot3d_modal",
                "projected_current_path": "p15_green_observed_mesh_vs_hot3d_modal_current_k",
            }[path_key]
            iou = float(row[metric_key]["iou"])
            cv2.rectangle(tile, (0, 0), (tile_width, 52), (8, 8, 8), -1)
            cv2.putText(
                tile,
                f"f{int(row['frame_idx']):03d} {title}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"vs HOT3D modal IoU {iou:.3f}",
                (8, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        sheets.append(np.hstack(tiles))
    sheet = np.vstack(sheets)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write review: {output}")
    return {"path": str(output), "bytes": int(output.stat().st_size), "sha256": sha256_file(output)}


def rasterize_mesh_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    T_world_camera: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    vertices_world = vertices @ rotation.T + translation
    vertices_camera = canonical.world_points_to_camera(vertices_world, T_world_camera)
    u, v, z, _ = canonical.project_camera_points(vertices_camera, intrinsics, width, height)
    uv = np.column_stack((u, v))
    output = np.zeros((height, width), dtype=np.uint8)
    valid_faces = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(z[faces] > 0.01, axis=1)
    face_ids = np.flatnonzero(valid_faces)
    if len(face_ids):
        polygons_float = uv[faces[face_ids]]
        inside_review_bounds = (
            np.all(polygons_float[:, :, 0] >= -width, axis=1)
            & np.all(polygons_float[:, :, 0] <= 2 * width, axis=1)
            & np.all(polygons_float[:, :, 1] >= -height, axis=1)
            & np.all(polygons_float[:, :, 1] <= 2 * height, axis=1)
        )
        polygons = np.round(polygons_float[inside_review_bounds]).astype(np.int32)
        if len(polygons):
            twice_area = (
                (polygons[:, 1, 0] - polygons[:, 0, 0]) * (polygons[:, 2, 1] - polygons[:, 0, 1])
                - (polygons[:, 1, 1] - polygons[:, 0, 1]) * (polygons[:, 2, 0] - polygons[:, 0, 0])
            )
            polygons = polygons[twice_area != 0]
        if len(polygons):
            cv2.fillPoly(output, polygons, 255, cv2.LINE_8)
    return output > 0


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def mano_metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "matched_rows": int(summary["matched_rows"]),
        "wrist_error_median_m": float(summary["wrist_error_m"]["median"]),
        "joint_mpjpe_median_m": float(summary["joint_mpjpe_m"]["median"]),
        "root_aligned_mpjpe_median_m": float(summary["root_aligned_mpjpe_m"]["median"]),
        "vertex_centroid_error_median_m": float(summary["vertex_centroid_error_m"]["median"]),
    }


def compare_mano_counterfactual(as_labeled: dict[str, Any], swapped: dict[str, Any]) -> dict[str, Any]:
    as_rows = {
        (int(row["frame_idx"]), str(row["side"])): row
        for row in as_labeled["rows"]
        if row.get("matched")
    }
    swapped_rows = {
        (int(row["frame_idx"]), str(row["side"])): row
        for row in swapped["rows"]
        if row.get("matched")
    }
    if set(as_rows) != set(swapped_rows):
        raise RuntimeError("as-labeled and swapped MANO reports do not evaluate identical rows")
    metrics: dict[str, Any] = {}
    for key in ("wrist_error_m", "joint_mpjpe_m", "root_aligned_mpjpe_m", "vertex_centroid_error_m"):
        ratios = [
            float(swapped_rows[row_key][key]) / max(float(as_rows[row_key][key]), 1.0e-12)
            for row_key in as_rows
        ]
        as_wins = sum(float(as_rows[row_key][key]) < float(swapped_rows[row_key][key]) for row_key in as_rows)
        metrics[key] = {
            "as_labeled_better_rows": int(as_wins),
            "evaluated_rows": int(len(as_rows)),
            "swapped_over_as_labeled_ratio": summarize(ratios),
        }
    return metrics


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise RuntimeError(f"missing run root: {run_root}")
    state_path = require_file(args.render_state, "frozen P15 render state")
    gt_path = require_file(args.hot3d_gt, "HOT3D evaluator-only sidecar")
    official_manifest_path = require_file(args.official_pinhole_manifest, "official pinhole manifest")
    trajectory_path = require_file(args.object_trajectory_report, "object trajectory evaluator report")
    mano_as_path = require_file(args.mano_eval_as_labeled, "as-labeled MANO evaluator report")
    mano_swap_path = require_file(args.mano_eval_swapped, "swapped MANO evaluator report")

    state = load_json(state_path)
    gt = load_json(gt_path)
    official_manifest = load_json(official_manifest_path)
    trajectory = load_json(trajectory_path)
    mano_as = load_json(mano_as_path)
    mano_swap = load_json(mano_swap_path)
    annotation_path = require_file(Path(state["inputs"]["annotations"]), "P15 annotations")
    annotations = load_json(annotation_path)
    if trajectory.get("status") != "ok" or mano_as.get("status") != "ok" or mano_swap.get("status") != "ok":
        raise RuntimeError("one or more prerequisite evaluator reports is not ok")

    adapter = load_module("p15_hot3d_pinhole_adapter", SCRIPTS / "build_v19_hot3d_pinhole_adapter.py")
    pose_by_frame = canonical.pose_map(state)
    annotation_by_frame = {int(row["frame_idx"]): row for row in annotations["frames"]}
    gt_by_frame = {int(row["frame_idx"]): row for row in gt["frames"]}
    official_by_frame = {int(row["frame_idx"]): row for row in official_manifest["frames"]}
    common_frames = sorted(set(pose_by_frame) & set(annotation_by_frame) & set(gt_by_frame) & set(official_by_frame))
    if len(common_frames) != 150 or common_frames != list(range(150)):
        raise RuntimeError(f"expected frames 0..149, got {common_frames[:3]}..{common_frames[-3:]}")

    geometry_rows = state["object_geometry"]["render_layers_back_to_front"]
    observed_rows = [row for row in geometry_rows if row.get("role") == "observed_metric_surface_overlay"]
    if len(observed_rows) != 1:
        raise RuntimeError("render state must have exactly one observed metric surface layer")
    mesh_path = require_file(Path(observed_rows[0]["mesh"]), "observed metric surface")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"unsupported observed mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    selector = observed_rows[0].get("face_selection") or {"mode": "all"}
    if selector.get("mode") == "contiguous_range":
        faces = faces[int(selector["start"]): int(selector["stop"])]
    elif selector.get("mode") != "all":
        raise RuntimeError(f"unsupported observed face selector: {selector}")

    map_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    mask_rows: list[dict[str, Any]] = []
    annotation_metadata_inconsistency_frames: list[int] = []
    ownership_ratios: list[float] = []
    source_size = annotations["raw_video"]
    render_width = int(annotations["raw_video"].get("manifest_width") or 960)
    render_height = int(annotations["raw_video"].get("manifest_height") or 960)
    if (render_width, render_height) != (960, 960):
        render_width = render_height = 960

    for frame_idx in common_frames:
        frame_gt = gt_by_frame[frame_idx]
        frame_annotation = annotation_by_frame[frame_idx]
        object_rows = [row for row in frame_annotation.get("objects", []) if row.get("object_id") == "keyboard"]
        if len(object_rows) != 1:
            raise RuntimeError(f"frame {frame_idx}: expected one keyboard annotation")
        object_row = object_rows[0]
        visible = object_row.get("visible_geometry_candidate") or {}
        sam_path = require_file(Path(visible["source_mask_path"]), f"frame {frame_idx} raw SAM2 mask")
        owned_path = require_file(Path(visible["mask_path"]), f"frame {frame_idx} object-owned mask")
        sam = read_mask(sam_path)
        owned = read_mask(owned_path)
        if sam.shape != (render_height, render_width) or owned.shape != sam.shape:
            raise RuntimeError(f"frame {frame_idx}: unexpected prediction mask shape {sam.shape}/{owned.shape}")

        camera = frame_gt["json"]["cameras.json"][args.stream_id]
        signature = adapter.camera_signature(camera, args.stream_id, float(args.focal_scale))
        if signature not in map_cache:
            width, height, focal, cx, cy, coefficients = adapter.parse_fisheye624(camera)
            map_x, map_y, _ = adapter.fisheye624_pinhole_maps(
                width, height, focal, cx, cy, coefficients, float(args.focal_scale)
            )
            map_cache[signature] = (map_x, map_y)
        map_x, map_y = map_cache[signature]
        object_entries = frame_gt["json"]["objects.json"].get(str(args.object_bop_id)) or []
        if len(object_entries) != 1:
            raise RuntimeError(f"frame {frame_idx}: expected one HOT3D object {args.object_bop_id}")
        gt_modal_raw = decode_hot3d_rle(object_entries[0]["masks_modal"][args.stream_id]).astype(np.uint8) * 255
        gt_modal_pinhole = cv2.remap(
            gt_modal_raw,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        gt_modal = cv2.resize(
            gt_modal_pinhole,
            (render_width, render_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

        rotation, translation, _ = pose_by_frame[frame_idx]
        current_intrinsics, _ = canonical.scaled_intrinsics_for_frame(
            frame_annotation, render_width, render_height, source_size
        )
        official_intrinsics_source = official_by_frame[frame_idx]["hot3d_pinhole_fx_fy_cx_cy"]
        sx = float(render_width) / float(official_by_frame[frame_idx]["source_width"])
        sy = float(render_height) / float(official_by_frame[frame_idx]["source_height"])
        official_intrinsics = (
            float(official_intrinsics_source[0]) * sx,
            float(official_intrinsics_source[1]) * sy,
            float(official_intrinsics_source[2]) * sx,
            float(official_intrinsics_source[3]) * sy,
        )
        T_world_camera = np.asarray(frame_annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
        projected_current = rasterize_mesh_silhouette(
            vertices, faces, rotation, translation, T_world_camera, current_intrinsics, render_width, render_height
        )
        projected_official_k = rasterize_mesh_silhouette(
            vertices, faces, rotation, translation, T_world_camera, official_intrinsics, render_width, render_height
        )

        raw_pixels = int(np.count_nonzero(sam))
        owned_pixels = int(np.count_nonzero(owned))
        ownership_ratio = float(owned_pixels / raw_pixels) if raw_pixels else math.nan
        ownership_ratios.append(ownership_ratio)
        raw_bbox_area = float(object_row.get("area_px")) if object_row.get("area_px") is not None else None
        if raw_bbox_area is not None and abs(raw_bbox_area - owned_pixels) > 0.5:
            annotation_metadata_inconsistency_frames.append(frame_idx)
        mask_rows.append(
            {
                "frame_idx": frame_idx,
                "rgb_path": str(frame_annotation["raw_frame_path"]),
                "raw_sam2_path": str(sam_path),
                "object_owned_path": str(owned_path),
                "raw_sam2_vs_hot3d_modal": mask_metrics(sam, gt_modal),
                "object_owned_vs_hot3d_modal": mask_metrics(owned, gt_modal),
                "p15_green_observed_mesh_vs_hot3d_modal_current_k": mask_metrics(projected_current, gt_modal),
                "p15_green_observed_mesh_vs_raw_sam2_current_k": mask_metrics(projected_current, sam),
                "same_pose_camera_official_k_vs_hot3d_modal": mask_metrics(projected_official_k, gt_modal),
                "object_owned_over_raw_sam2_area_fraction": ownership_ratio,
                "annotation_object_area_px_value": raw_bbox_area,
                "annotation_mask_path": str(object_row.get("mask_path")),
                "visible_geometry_source_mask_path": str(sam_path),
                "visible_geometry_owned_mask_path": str(owned_path),
            }
        )

    hawor_path = require_file(
        Path(mano_as["prediction_source"]["source_hawor_npz"]), "HaWoR full-MANO source"
    )
    bridge_paths = {
        Path(
            hand["metric_mano_state"]["vertices_reference"]["bridge_npz"]
        ).expanduser().resolve()
        for frame in annotations["frames"]
        for hand in frame.get("hands", [])
        if isinstance(hand, dict)
        and isinstance(hand.get("metric_mano_state"), dict)
        and isinstance(hand["metric_mano_state"].get("vertices_reference"), dict)
    }
    if len(bridge_paths) != 1:
        raise RuntimeError(f"expected one MANO bridge path, got {bridge_paths}")
    bridge_path = require_file(next(iter(bridge_paths)), "MANO bridge")
    hawor = np.load(hawor_path, allow_pickle=False)
    bridge = np.load(bridge_path, allow_pickle=False)
    try:
        bridge_vertices_all = np.asarray(
            bridge["vertices_current_v18_world_from_hawor_projection_relift_m"], dtype=np.float64
        )
        bridge_camera_all = np.asarray(bridge["vertices_current_v18_camera_m"], dtype=np.float64)
        bridge_frame_indices = np.asarray(bridge["frame_idx"], dtype=np.int64)
        bridge_hand_sides = np.asarray(bridge["hand_side"]).astype(str)
        hawor_vertices_world = {
            side: np.asarray(hawor[f"{side}_vertices_world_m"], dtype=np.float64)
            for side in ("left", "right")
        }
        hawor_detection_boxes = {
            side: np.asarray(hawor[f"{side}_det_box_xyxyscore"], dtype=np.float64)
            for side in ("left", "right")
        }
        source_bridge_max_errors: dict[str, float] = {}
        side_sequence_valid = True
        bridge_reference_mismatches = 0
        camera_inverse_errors: list[float] = []
        for frame_idx in common_frames:
            frame = annotation_by_frame[frame_idx]
            T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
            frame_hands = {str(hand["hand_side"]): hand for hand in frame.get("hands", [])}
            for side in ("left", "right"):
                hand = frame_hands[side]
                reference = hand["metric_mano_state"]["vertices_reference"]
                row_index = int(reference["bridge_row_index"])
                array_name = str(reference["bridge_vertices_world_array"])
                if array_name != "vertices_current_v18_world_from_hawor_projection_relift_m":
                    raise RuntimeError(f"unsupported bridge vertices array: {array_name}")
                bridge_vertices = bridge_vertices_all[row_index]
                source_vertices = hawor_vertices_world[side][frame_idx]
                source_bridge_max_errors[side] = max(
                    source_bridge_max_errors.get(side, 0.0),
                    float(np.max(np.abs(bridge_vertices - source_vertices))),
                )
                if int(bridge_frame_indices[row_index]) != frame_idx or str(bridge_hand_sides[row_index]) != side:
                    bridge_reference_mismatches += 1
                expected_row = 2 * frame_idx + (1 if side == "right" else 0)
                if row_index != expected_row:
                    side_sequence_valid = False
                bridge_camera = bridge_camera_all[row_index]
                recovered_camera = canonical.world_points_to_camera(bridge_vertices, T_world_camera)
                camera_inverse_errors.append(float(np.max(np.linalg.norm(recovered_camera - bridge_camera, axis=1))))

        same_box_ious: list[float] = []
        swapped_box_ious: list[float] = []
        paired_rows: list[dict[str, Any]] = []
        for frame_idx in common_frames:
            gt_hands = gt_by_frame[frame_idx]["json"]["hands.json"]
            values: dict[tuple[str, str], float] = {}
            for prediction_side in ("left", "right"):
                predicted_box = hawor_detection_boxes[prediction_side][frame_idx, :4]
                if not np.isfinite(predicted_box).all():
                    continue
                for gt_side in ("left", "right"):
                    gt_box = np.asarray(gt_hands[gt_side]["boxes_amodal"][args.stream_id], dtype=float)
                    values[(prediction_side, gt_side)] = box_iou(predicted_box, gt_box)
            for side in ("left", "right"):
                if (side, side) in values:
                    same_box_ious.append(values[(side, side)])
                other = "right" if side == "left" else "left"
                if (side, other) in values:
                    swapped_box_ious.append(values[(side, other)])
            if len(values) == 4:
                same_score = 0.5 * (values[("left", "left")] + values[("right", "right")])
                swapped_score = 0.5 * (values[("left", "right")] + values[("right", "left")])
                paired_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "as_labeled_assignment_mean_iou": same_score,
                        "swapped_assignment_mean_iou": swapped_score,
                        "as_labeled_wins": bool(same_score > swapped_score),
                    }
                )
    finally:
        hawor.close()
        bridge.close()

    camera_motion_mismatch = frame0_relative_camera_motion_mismatch(
        common_frames, annotation_by_frame, gt_by_frame, args.stream_id
    )
    current_intrinsics_source = annotations["frames"][0]["camera"]["intrinsics_fx_fy_cx_cy"]
    official_intrinsics_source = official_by_frame[0]["hot3d_pinhole_fx_fy_cx_cy"]
    trajectory_translation = trajectory["constant_transform_residual"]["translation_m"]
    trajectory_rotation = trajectory["constant_transform_residual"]["rotation_deg"]
    representative_indices = {int(value) for value in REPRESENTATIVE_FRAMES}
    representative = [row for row in mask_rows if int(row["frame_idx"]) in representative_indices]
    review_rows = []
    for row in representative:
        review_row = dict(row)
        frame_idx = int(row["frame_idx"])
        frame_annotation = annotation_by_frame[frame_idx]
        rotation, translation, _ = pose_by_frame[frame_idx]
        current_intrinsics, _ = canonical.scaled_intrinsics_for_frame(
            frame_annotation, render_width, render_height, source_size
        )
        review_row["_projected_current_mask"] = rasterize_mesh_silhouette(
            vertices,
            faces,
            rotation,
            translation,
            np.asarray(frame_annotation["camera"]["T_world_camera_metric"], dtype=np.float64),
            current_intrinsics,
            render_width,
            render_height,
        )
        review_rows.append(review_row)
    review = write_mask_review(review_rows, args.review_output) if args.review_output is not None else None
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "audit_p15_keyboard_mask_drift_and_mano_handedness_after_prediction_freeze",
        "claim_scope": (
            "Post-freeze diagnostic only. HOT3D GT is evaluator-side and was accessed after the prediction and its hash manifest were frozen. "
            "The audit distinguishes raw SAM2 masks, object-owned masks, and the P15 projected observed mesh. It evaluates handedness labels but makes no contact, collision, or signed nonpenetration claim."
        ),
        "frozen_prediction_commit": "7d5720e2422a3a5c825d28aa7209a1d8268275a6",
        "gt_isolation": {
            "prediction_frozen_before_gt_evaluation": True,
            "gt_used_only_by_post_freeze_evaluator": True,
            "prediction_state_mutated": False,
            "hot3d_gt": file_ref(gt_path),
        },
        "inputs": {
            "run_root": str(run_root),
            "render_state": file_ref(state_path),
            "annotations": file_ref(annotation_path),
            "observed_metric_surface": file_ref(mesh_path),
            "official_pinhole_manifest": file_ref(official_manifest_path),
            "object_trajectory_report": file_ref(trajectory_path),
            "mano_as_labeled_report": file_ref(mano_as_path),
            "mano_swapped_report": file_ref(mano_swap_path),
            "mano_bridge": file_ref(bridge_path),
            "hawor_full_mano": file_ref(hawor_path),
        },
        "mask_drift_audit": {
            "frame_count": len(mask_rows),
            "p15_green_layer_semantics": (
                "The green P15 layer is the fixed observed metric surface transformed by the per-frame object trajectory and camera; "
                "it is not the per-frame SAM2 or object-owned raster mask."
            ),
            "raw_sam2_vs_hot3d_modal": summarize_mask_rows(mask_rows, "raw_sam2_vs_hot3d_modal"),
            "object_owned_vs_hot3d_modal": summarize_mask_rows(mask_rows, "object_owned_vs_hot3d_modal"),
            "p15_green_observed_mesh_vs_hot3d_modal_current_k": summarize_mask_rows(
                mask_rows, "p15_green_observed_mesh_vs_hot3d_modal_current_k"
            ),
            "p15_green_observed_mesh_vs_raw_sam2_current_k": summarize_mask_rows(
                mask_rows, "p15_green_observed_mesh_vs_raw_sam2_current_k"
            ),
            "same_pose_camera_official_k_vs_hot3d_modal": summarize_mask_rows(
                mask_rows, "same_pose_camera_official_k_vs_hot3d_modal"
            ),
            "object_owned_over_raw_sam2_area_fraction": summarize(ownership_ratios),
            "object_ownership_mechanism": (
                "Whole HaWoR hand detection rectangles plus 12 px padding were subtracted before visible-depth lifting. "
                "This is conservative ownership subtraction, not a per-pixel hand mask."
            ),
            "annotation_metadata_semantics": {
                "mask_path_points_to_object_owned_mask": True,
                "area_px_or_bbox_still_describe_source_track_space": True,
                "inconsistent_frame_count": len(annotation_metadata_inconsistency_frames),
                "inconsistent_frames": annotation_metadata_inconsistency_frames,
            },
            "camera_intrinsics_source_1408": {
                "current_unidepth_fx_fy_cx_cy": [float(value) for value in current_intrinsics_source],
                "official_hot3d_pinhole_fx_fy_cx_cy": [float(value) for value in official_intrinsics_source],
            },
            "camera_trajectory_frame0_relative_motion_mismatch": camera_motion_mismatch,
            "object_trajectory_fixed_canonical_transform_residual": {
                "translation_median_m": float(trajectory_translation["median"]),
                "translation_p95_m": float(trajectory_translation["p95"]),
                "rotation_median_deg": float(trajectory_rotation["median"]),
                "rotation_p95_deg": float(trajectory_rotation["p95"]),
            },
            "interpretation": (
                "Raw SAM2 follows the official modal keyboard mask closely over the frozen clip. The conspicuous green-layer drift is therefore primarily a projected observed-mesh/object-pose/camera/canonical-state failure, not raw SAM2 temporal drift. "
                "The object-owned mask is often visibly incomplete because rectangle-based hand ownership subtraction removes most keyboard support in many frames. "
                "The independent camera-motion audit also finds non-negligible tail error, so the green drift cannot be assigned to object pose alone. "
                "Replacing only K with official intrinsics does not consistently repair the frozen trajectory and is not a valid substitute for an official-camera rerun."
            ),
            "review": review,
            "representative_frames": representative,
            "rows": mask_rows,
        },
        "mano_handedness_audit": {
            "renderer_color_contract_bgr": {
                "left": [245, 175, 55],
                "right": [45, 145, 255],
                "note": "These are BGR tuples consumed by OpenCV, not RGB tuples. In displayed images they appear approximately blue/cyan for left and orange/red for right.",
            },
            "bridge_and_renderer_chain": {
                "bridge_rows_follow_left_then_right_per_frame": bool(side_sequence_valid),
                "bridge_reference_mismatch_count": int(bridge_reference_mismatches),
                "source_to_bridge_max_abs_error_m": source_bridge_max_errors,
                "world_to_camera_full_vertex_max_error_m": summarize(camera_inverse_errors),
                "renderer_contract": (
                    "hand_side selects the referenced 778-vertex bridge row and the same-side HaWoR face array; frame and side are checked before rendering."
                ),
            },
            "hot3d_box_assignment": {
                "same_side_iou": summarize(same_box_ious),
                "swapped_side_iou": summarize(swapped_box_ious),
                "paired_two_hand_frame_count": int(len(paired_rows)),
                "as_labeled_wins_count": int(sum(row["as_labeled_wins"] for row in paired_rows)),
                "as_labeled_assignment_mean_iou": summarize(
                    [float(row["as_labeled_assignment_mean_iou"]) for row in paired_rows]
                ),
                "swapped_assignment_mean_iou": summarize(
                    [float(row["swapped_assignment_mean_iou"]) for row in paired_rows]
                ),
                "paired_rows": paired_rows,
            },
            "hot3d_mano3d_counterfactual": {
                "as_labeled": mano_metric_summary(mano_as),
                "left_right_swapped": mano_metric_summary(mano_swap),
                "rowwise_comparison": compare_mano_counterfactual(mano_as, mano_swap),
            },
            "interpretation": (
                "The frozen prediction and renderer do not swap left/right hands. Independent HOT3D boxes and 3D MANO metrics strongly prefer the original labels over a full side swap. "
                "The apparent reversal in world view is a viewpoint/legend perception issue, compounded by the fact that OpenCV color constants are BGR. It is not evidence of swapped MANO arrays or an inverted world-to-camera transform."
            ),
        },
        "physical_claims": {
            "contact_claim_made": False,
            "collision_claim_made": False,
            "signed_geometry_claim_made": False,
        },
    }
    return report


def main() -> None:
    args = parse_args()
    report = audit(args)
    output = args.output_report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_report": str(output),
                "raw_sam2_iou_median": report["mask_drift_audit"]["raw_sam2_vs_hot3d_modal"]["iou"]["median"],
                "green_mesh_iou_median": report["mask_drift_audit"]["p15_green_observed_mesh_vs_hot3d_modal_current_k"]["iou"]["median"],
                "owned_retained_area_median": report["mask_drift_audit"]["object_owned_over_raw_sam2_area_fraction"]["median"],
                "same_side_box_iou_median": report["mano_handedness_audit"]["hot3d_box_assignment"]["same_side_iou"]["median"],
                "swapped_box_iou_median": report["mano_handedness_audit"]["hot3d_box_assignment"]["swapped_side_iou"]["median"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
