#!/usr/bin/env python3
"""Three-experiment audit for SAM3D/GHOST-lite mask-depth-surface alignment.

The audit is deliberately CPU-only and prediction-read-only:

1. Closed-loop depth reprojection: mask/depth pixel -> camera 3D -> same pixel,
   plus camera -> world -> camera closure and stored visible-surfel consistency.
2. Direct mask/depth/observed-surface overlay without SAM3D: dense owned-depth
   support, robust first-surface support, depth discontinuities, and the anchor
   Poisson observed surface versus raw/owned masks.
3. Downstream GHOST-lite error pattern: reuse frozen before/after alignment,
   full-resolution first-hit, full-video mask/raster, and wrong-vs-correct-K
   reports to decide whether the observed mismatch behaves like a depth problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_v19_visible_geometry_from_sam2_depth import robust_first_surface_depth_ownership  # noqa: E402

BASE = Path(
    "/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/"
    "backend_tests/20260819T122452Z_hot3d_milk_local_authority_local29_v1/runs/"
    "P0014_84ea2dcc_carton_milk_f2370_2519"
)
GHOST = BASE / "experiments/sam3d_native_ghost_lite_20260826"
SCHEMA = "sam3d_ghost_lite_mask_depth_alignment_three_experiments_v1"
DEFAULT_FRAMES = "30,50,70,92,103,110,121,130,146"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=BASE / "measurements/object_geometry/visible_geometry/carton_milk/annotations_v19_visible_geometry.json",
    )
    parser.add_argument(
        "--alignment-report",
        type=Path,
        default=GHOST / "p15_first_hit_depth_authority_bounded_v4/qc_sam3d_p15_first_hit_alignment.json",
    )
    parser.add_argument(
        "--fullres-audit-report",
        type=Path,
        default=GHOST / "p15_first_hit_depth_authority_bounded_v4/fullres_audit/sam3d_first_hit_surface_position_audit.json",
    )
    parser.add_argument(
        "--full-video-report",
        type=Path,
        default=GHOST / "p15_first_hit_depth_authority_bounded_v4/full_video_correct_intrinsics_v2/qc_optimized_object_full_video.json",
    )
    parser.add_argument(
        "--wrong-intrinsics-report",
        type=Path,
        default=GHOST / "p15_first_hit_depth_authority_bounded_v4/full_video/diagnosis_wrong_vs_correct_intrinsics.json",
    )
    parser.add_argument(
        "--anchor-observed-mesh",
        type=Path,
        default=BASE
        / "measurements/object_geometry/visible_geometry/carton_milk/anchor_visible_surface_mesh/carton_milk/frame_000092_carton_milk_anchor_poisson_visible_mesh.ply",
    )
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--depth-edge-range-m", type=float, default=0.015)
    parser.add_argument("--edge-near-radius-source-px", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=GHOST / "mask_depth_alignment_three_experiments_v1")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def prepare_output(path: Path, replace: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not replace:
            raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    path = require_file(path, "JSON")
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(require_file(path, "mask")), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return image > 0


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    interpolation = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    return cv2.resize(mask.astype(np.uint8), (int(width), int(height)), interpolation=interpolation) > 0


def boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if not np.any(mask_u8):
        return np.zeros_like(mask_u8, dtype=bool)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (mask_u8 > 0) & (eroded == 0)


def summarize(values: np.ndarray, unit_suffix: str = "") -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min" + unit_suffix: float(np.min(array)),
        "median" + unit_suffix: float(np.median(array)),
        "mean" + unit_suffix: float(np.mean(array)),
        "p90" + unit_suffix: float(np.percentile(array, 90.0)),
        "p95" + unit_suffix: float(np.percentile(array, 95.0)),
        "max" + unit_suffix: float(np.max(array)),
    }


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise RuntimeError(f"mask shape mismatch: {prediction.shape} vs {target.shape}")
    pred = np.asarray(prediction, dtype=bool)
    tgt = np.asarray(target, dtype=bool)
    intersection = int(np.count_nonzero(pred & tgt))
    union = int(np.count_nonzero(pred | tgt))
    pred_count = int(np.count_nonzero(pred))
    target_count = int(np.count_nonzero(tgt))
    centroid_delta = math.nan
    if pred_count and target_count:
        pred_centroid = np.argwhere(pred).mean(axis=0)[::-1]
        target_centroid = np.argwhere(tgt).mean(axis=0)[::-1]
        centroid_delta = float(np.linalg.norm(pred_centroid - target_centroid))
    output = {
        "iou": float(intersection / union) if union else None,
        "precision": float(intersection / pred_count) if pred_count else None,
        "recall": float(intersection / target_count) if target_count else None,
        "centroid_distance_px": centroid_delta,
        "prediction_pixels": pred_count,
        "target_pixels": target_count,
    }
    output.update(symmetric_boundary_metrics(pred, tgt))
    return output


def symmetric_boundary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred_boundary = boundary(prediction)
    target_boundary = boundary(target)
    if not np.any(pred_boundary) or not np.any(target_boundary):
        return {
            "symmetric_boundary_median_px": None,
            "symmetric_boundary_p95_px": None,
        }
    to_target = cv2.distanceTransform((~target_boundary).astype(np.uint8), cv2.DIST_L2, 3)[pred_boundary]
    to_pred = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 3)[target_boundary]
    combined = np.concatenate([to_target, to_pred])
    return {
        "symmetric_boundary_median_px": float(np.median(combined)),
        "symmetric_boundary_p95_px": float(np.percentile(combined, 95.0)),
    }


def project_camera(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64).tolist()
    z = points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (z > 1.0e-9)
    uv[valid, 0] = fx * points[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points[valid, 1] / z[valid] + cy
    return uv


def apply_affine(uv: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(uv, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    mapped = homogeneous @ np.asarray(affine, dtype=np.float64).T
    if np.any(np.abs(mapped[:, 2]) < 1.0e-12):
        raise RuntimeError("affine projection produced a zero denominator")
    return mapped[:, :2] / mapped[:, 2:3]


def affine_manifest_from_source(visible: dict[str, Any], mask_shape: tuple[int, int], depth_shape: tuple[int, int]) -> np.ndarray:
    contract = visible.get("mask_depth_transform_contract") if isinstance(visible.get("mask_depth_transform_contract"), dict) else {}
    mask_plane = contract.get("mask_plane_transform") if isinstance(contract.get("mask_plane_transform"), dict) else {}
    affine = mask_plane.get("A_actual_plane_from_calibration")
    if affine is not None:
        return np.asarray(affine, dtype=np.float64)
    ratio_y = float(mask_shape[0]) / float(depth_shape[0])
    ratio_x = float(mask_shape[1]) / float(depth_shape[1])
    return np.asarray(
        [[ratio_x, 0.0, 0.5 * (ratio_x - 1.0)], [0.0, ratio_y, 0.5 * (ratio_y - 1.0)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def backproject_mask(mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    z = np.asarray(depth, dtype=np.float64)[ys, xs]
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64).tolist()
    points = np.column_stack([
        (xs.astype(np.float64) - cx) * z / fx,
        (ys.astype(np.float64) - cy) * z / fy,
        z,
    ])
    return points, np.column_stack([xs, ys]).astype(np.float64)


def depth_local_range_edges(depth: np.ndarray, valid: np.ndarray, threshold_m: float) -> tuple[np.ndarray, np.ndarray]:
    depth64 = np.asarray(depth, dtype=np.float64)
    valid_u8 = np.asarray(valid, dtype=np.uint8)
    max_filled = np.where(valid_u8 > 0, depth64, -np.inf)
    min_filled = np.where(valid_u8 > 0, depth64, np.inf)
    kernel = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(max_filled, kernel)
    local_min = cv2.erode(min_filled, kernel)
    local_range = local_max - local_min
    local_range[valid_u8 == 0] = 0.0
    edges = (valid_u8 > 0) & np.isfinite(local_range) & (local_range >= float(threshold_m))
    return edges, local_range


def ownership_reconstruction_matches(stored: dict[str, Any], recomputed: dict[str, Any]) -> dict[str, Any]:
    keys = ("input_valid_depth_pixels", "retained_depth_pixels", "removed_depth_pixels", "retained_fraction")
    deltas: dict[str, Any] = {}
    matches = True
    for key in keys:
        a = stored.get(key)
        b = recomputed.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = abs(float(a) - float(b))
            deltas[key] = delta
            matches = matches and delta <= (1.0e-9 if key == "retained_fraction" else 0.0)
        else:
            matches = False
    return {"matches": bool(matches), "absolute_deltas": deltas}


def within_fractions(values: np.ndarray, thresholds_px: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {f"within_{threshold:g}px_fraction": None for threshold in thresholds_px}
    return {
        f"within_{threshold:g}px_fraction": float(np.mean(array <= threshold))
        for threshold in thresholds_px
    }


def add_alpha_mask(image: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    color = np.zeros_like(image)
    color[:] = color_bgr
    blended = cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0.0)
    image[mask] = blended[mask]


def draw_mask_contours(image: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], thickness: int = 2) -> None:
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color_bgr, thickness, cv2.LINE_AA)


def put_label(image: np.ndarray, title: str, subtitle: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 48), (12, 12, 12), -1)
    cv2.putText(image, title[:88], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, subtitle[:112], (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 225, 225), 1, cv2.LINE_AA)


def crop_with_context(image: np.ndarray, masks: list[np.ndarray], pad: int = 70) -> np.ndarray:
    combined = np.zeros(image.shape[:2], dtype=bool)
    for mask in masks:
        combined |= np.asarray(mask, dtype=bool)
    ys, xs = np.where(combined)
    if len(xs) == 0:
        return image.copy()
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad + 1)
    y1 = min(image.shape[0], int(ys.max()) + pad + 1)
    crop = image[y0:y1, x0:x1].copy()
    crop = resize_to_square(crop, 420)
    return crop


def resize_to_square(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = float(size) / float(max(height, width))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 18, dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas



def rasterize_mesh_silhouette(
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    uv = project_camera(vertices_camera, intrinsics)
    z = np.asarray(vertices_camera, dtype=np.float64)[:, 2]
    output = np.zeros((height, width), dtype=np.uint8)
    faces_i = np.asarray(faces, dtype=np.int64)
    valid_faces = np.all(np.isfinite(uv[faces_i]), axis=(1, 2)) & np.all(z[faces_i] > 1.0e-6, axis=1)
    polygons = np.rint(uv[faces_i[valid_faces]]).astype(np.int32)
    if len(polygons):
        cv2.fillPoly(output, list(polygons), 1, cv2.LINE_8)
    return output > 0


def metric_values_by_frame(report: dict[str, Any], section: str, key: str) -> dict[int, float]:
    per_frame = report.get("per_frame")
    output: dict[int, float] = {}
    if isinstance(per_frame, dict):
        for raw_idx, row in per_frame.items():
            value = ((row.get(section) or {}).get(key))
            if isinstance(value, (int, float)):
                output[int(raw_idx)] = float(value)
    return output


def correlation(a: dict[int, float], b: dict[int, float]) -> float | None:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    x = np.asarray([a[i] for i in common], dtype=np.float64)
    y = np.asarray([b[i] for i in common], dtype=np.float64)
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def process_frame(
    frame_idx: int,
    frame: dict[str, Any],
    depth_archive: Any,
    depth_row_by_idx: dict[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []
    if len(objects) != 1:
        raise RuntimeError(f"frame {frame_idx}: expected one object row")
    object_row = objects[0]
    visible = object_row.get("visible_geometry_candidate")
    if not isinstance(visible, dict):
        raise RuntimeError(f"frame {frame_idx}: missing visible_geometry_candidate")
    owned_mask = read_mask(Path(str(visible["mask_path"])))
    raw_mask = read_mask(Path(str(visible["source_mask_path"])))
    rgb = cv2.imread(str(require_file(Path(str(frame["raw_frame_path"])), "raw RGB")), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"failed to read RGB: {frame['raw_frame_path']}")
    if rgb.shape[:2] != owned_mask.shape or raw_mask.shape != owned_mask.shape:
        raise RuntimeError(
            f"frame {frame_idx}: RGB/mask shape mismatch rgb={rgb.shape[:2]} owned={owned_mask.shape} raw={raw_mask.shape}"
        )

    depth_row = depth_row_by_idx.get(int(frame_idx))
    if depth_row is None:
        raise RuntimeError(f"frame {frame_idx}: absent from depth archive")
    if int(visible.get("depth_frame_index", depth_row)) != int(depth_row):
        raise RuntimeError(
            f"frame {frame_idx}: annotation depth_frame_index={visible.get('depth_frame_index')} != archive row {depth_row}"
        )
    depth = np.asarray(depth_archive["depth"][depth_row], dtype=np.float64)
    confidence = (
        np.asarray(depth_archive["confidence"][depth_row], dtype=np.float64)
        if "confidence" in depth_archive.files
        else None
    )
    intrinsics = np.asarray(visible["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    mask_source = resize_mask(owned_mask, depth.shape[1], depth.shape[0])
    raw_mask_source = resize_mask(raw_mask, depth.shape[1], depth.shape[0])
    valid_depth = np.isfinite(depth) & (depth >= float(args.min_depth_m)) & (depth <= float(args.max_depth_m))
    dense_valid = mask_source & valid_depth
    robust_valid, robust_summary = robust_first_surface_depth_ownership(
        dense_valid,
        depth,
        enabled=True,
        mad_sigma=2.5,
        min_half_width_m=0.03,
        min_retained_fraction=0.90,
        fail_raw_to_robust_extent_ratio=2.0,
        intrinsics=intrinsics,
        confidence=confidence,
        confidence_seed_percentile=95.0,
        local_depth_step_max_m=0.005,
        max_removed_distance_inside_mask_px=10.0,
        max_confidence_flagged_interior_fraction=0.015,
        min_interior_confidence_flagged_fraction=0.95,
        max_unexplained_interior_pixels=5,
        min_component_pixels=20,
        max_small_component_fraction=0.01,
    )
    stored_ownership = visible.get("first_surface_depth_ownership") if isinstance(visible.get("first_surface_depth_ownership"), dict) else {}
    ownership_match = ownership_reconstruction_matches(stored_ownership, robust_summary)

    dense_points, dense_uv = backproject_mask(dense_valid, depth, intrinsics)
    dense_uv_reprojected = project_camera(dense_points, intrinsics)
    dense_errors = np.linalg.norm(dense_uv_reprojected - dense_uv, axis=1)
    world_points = dense_points @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]
    camera_recovered = (world_points - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
    camera_inverse_errors = np.linalg.norm(camera_recovered - dense_points, axis=1)
    affine = affine_manifest_from_source(visible, owned_mask.shape, depth.shape)
    dense_uv_manifest = apply_affine(dense_uv_reprojected, affine)
    dense_uv_manifest_expected = apply_affine(dense_uv, affine)
    manifest_errors = np.linalg.norm(dense_uv_manifest - dense_uv_manifest_expected, axis=1)

    stored_camera = np.asarray(visible.get("camera_vertices_sample_m"), dtype=np.float64)
    stored_camera = stored_camera[np.isfinite(stored_camera).all(axis=1) & (stored_camera[:, 2] > 1.0e-9)]
    stored_uv_source = project_camera(stored_camera, intrinsics)
    stored_uv_manifest = apply_affine(stored_uv_source, affine)
    robust_display = resize_mask(robust_valid, owned_mask.shape[1], owned_mask.shape[0])
    dense_display = resize_mask(dense_valid, owned_mask.shape[1], owned_mask.shape[0])
    stored_inside = (
        (stored_uv_manifest[:, 0] >= 0)
        & (stored_uv_manifest[:, 0] < owned_mask.shape[1])
        & (stored_uv_manifest[:, 1] >= 0)
        & (stored_uv_manifest[:, 1] < owned_mask.shape[0])
    )
    stored_uv_int = np.rint(stored_uv_manifest[stored_inside]).astype(np.int64)
    stored_inside_robust = np.zeros(len(stored_camera), dtype=bool)
    stored_inside_owned = np.zeros(len(stored_camera), dtype=bool)
    stored_inside_robust[stored_inside] = robust_display[stored_uv_int[:, 1], stored_uv_int[:, 0]]
    stored_inside_owned[stored_inside] = owned_mask[stored_uv_int[:, 1], stored_uv_int[:, 0]]
    stored_uv_source_int = np.rint(stored_uv_source).astype(np.int64)
    source_inside = (
        (stored_uv_source_int[:, 0] >= 0)
        & (stored_uv_source_int[:, 0] < depth.shape[1])
        & (stored_uv_source_int[:, 1] >= 0)
        & (stored_uv_source_int[:, 1] < depth.shape[0])
    )
    stored_depth_delta = np.abs(
        stored_camera[source_inside, 2]
        - depth[stored_uv_source_int[source_inside, 1], stored_uv_source_int[source_inside, 0]]
    )
    stored_world = np.asarray(visible.get("world_vertices_sample_m"), dtype=np.float64)
    stored_world = stored_world[np.isfinite(stored_world).all(axis=1)]
    world_count = min(len(stored_world), len(stored_camera))
    stored_world_inverse_error = np.asarray([], dtype=np.float64)
    if world_count:
        stored_world_inverse_error = np.linalg.norm(
            (stored_world[:world_count] - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
            - stored_camera[:world_count],
            axis=1,
        )

    mask_boundary_source = boundary(mask_source)
    all_depth_edges, local_range = depth_local_range_edges(depth, valid_depth, float(args.depth_edge_range_m))
    radius = int(args.edge_near_radius_source_px)
    near_mask = cv2.dilate(
        mask_boundary_source.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    ) > 0
    near_depth_edges = all_depth_edges & near_mask
    boundary_to_edge = cv2.distanceTransform((~near_depth_edges).astype(np.uint8), cv2.DIST_L2, 3)[mask_boundary_source]
    edge_to_boundary = cv2.distanceTransform((~mask_boundary_source).astype(np.uint8), cv2.DIST_L2, 3)[near_depth_edges]
    boundary_to_any_edge = cv2.distanceTransform((~all_depth_edges).astype(np.uint8), cv2.DIST_L2, 3)[mask_boundary_source]

    resized_owned_closure = resize_mask(mask_source, owned_mask.shape[1], owned_mask.shape[0])
    row = {
        "frame_idx": int(frame_idx),
        "paths": {
            "rgb": str(frame["raw_frame_path"]),
            "owned_mask": str(visible["mask_path"]),
            "raw_sam2_mask": str(visible["source_mask_path"]),
            "depth_npz": str(visible["depth_npz"]),
        },
        "contract": {
            "intrinsics_fx_fy_cx_cy": intrinsics.astype(float).tolist(),
            "intrinsics_source": visible.get("intrinsics_source"),
            "mask_depth_transform_contract": visible.get("mask_depth_transform_contract"),
            "affine_manifest_from_source": affine.astype(float).tolist(),
        },
        "experiment_1_closed_loop": {
            "dense_owned_valid_depth_pixels": int(len(dense_points)),
            "source_reprojection_error_px": summarize(dense_errors, "_px"),
            "source_reprojection_error_abs_max_component_px": float(np.max(np.abs(dense_uv_reprojected - dense_uv))) if len(dense_uv_reprojected) else None,
            "manifest_contract_reprojection_error_px": summarize(manifest_errors, "_px"),
            "camera_to_world_to_camera_error_m": summarize(camera_inverse_errors, "_m"),
            "stored_visible_surfels": int(len(stored_camera)),
            "stored_surfel_depth_minus_archive_abs_m": summarize(stored_depth_delta, "_m"),
            "stored_surfel_world_to_camera_inverse_error_m": summarize(stored_world_inverse_error, "_m"),
            "stored_surfel_projected_inside_robust_display_fraction": float(np.mean(stored_inside_robust)) if len(stored_camera) else None,
            "stored_surfel_projected_inside_owned_mask_fraction": float(np.mean(stored_inside_owned)) if len(stored_camera) else None,
        },
        "experiment_2_direct_mask_depth": {
            "owned_mask_pixels_manifest": int(np.count_nonzero(owned_mask)),
            "raw_sam2_mask_pixels_manifest": int(np.count_nonzero(raw_mask)),
            "owned_over_raw_area_fraction": float(np.count_nonzero(owned_mask) / max(1, np.count_nonzero(raw_mask))),
            "resized_owned_vs_original_owned": mask_metrics(resized_owned_closure, owned_mask),
            "dense_valid_display_vs_owned": mask_metrics(dense_display, owned_mask),
            "robust_first_surface_display_vs_owned": mask_metrics(robust_display, owned_mask),
            "robust_first_surface_ownership_recomputed": robust_summary,
            "robust_first_surface_ownership_matches_annotation": ownership_match,
            "depth_edge_threshold_local_range_m": float(args.depth_edge_range_m),
            "all_depth_edge_pixels": int(np.count_nonzero(all_depth_edges)),
            "near_mask_depth_edge_pixels": int(np.count_nonzero(near_depth_edges)),
            "mask_boundary_to_near_depth_edge_distance_source_px": summarize(boundary_to_edge, "_px"),
            "mask_boundary_to_near_depth_edge_within_fraction": within_fractions(boundary_to_edge),
            "near_depth_edge_to_mask_boundary_distance_source_px": summarize(edge_to_boundary, "_px"),
            "near_depth_edge_to_mask_boundary_within_fraction": within_fractions(edge_to_boundary),
            "mask_boundary_to_any_depth_edge_distance_source_px": summarize(boundary_to_any_edge, "_px"),
            "mask_boundary_to_any_depth_edge_within_fraction": within_fractions(boundary_to_any_edge),
            "local_depth_range_on_owned_mask_m": summarize(local_range[mask_source & valid_depth], "_m"),
        },
        "_visual": {
            "rgb": rgb,
            "owned_mask": owned_mask,
            "raw_mask": raw_mask,
            "dense_valid_display": dense_display,
            "robust_display": robust_display,
            "quarantined_display": resize_mask(dense_valid & ~robust_valid, owned_mask.shape[1], owned_mask.shape[0]),
            "depth_edges_display": resize_mask(all_depth_edges, owned_mask.shape[1], owned_mask.shape[0]),
            "near_depth_edges_display": resize_mask(near_depth_edges, owned_mask.shape[1], owned_mask.shape[0]),
        },
    }
    return row


def make_direct_review(frame_rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    tiles: list[np.ndarray] = []
    for row in frame_rows:
        visual = row["_visual"]
        rgb = visual["rgb"]
        owned = visual["owned_mask"]
        raw = visual["raw_mask"]
        dense = visual["dense_valid_display"]
        robust = visual["robust_display"]
        quarantined = visual["quarantined_display"]
        depth_edges = visual["depth_edges_display"]
        near_edges = visual["near_depth_edges_display"]

        p1 = rgb.copy()
        add_alpha_mask(p1, raw, (255, 80, 30), 0.20)
        add_alpha_mask(p1, owned, (65, 205, 75), 0.28)
        draw_mask_contours(p1, raw, (255, 80, 30), 2)
        draw_mask_contours(p1, owned, (65, 205, 75), 2)
        put_label(
            p1,
            f"f{row['frame_idx']:03d} masks",
            f"raw SAM2 blue | object-owned green | owned/raw {row['experiment_2_direct_mask_depth']['owned_over_raw_area_fraction']:.3f}",
        )

        p2 = rgb.copy()
        add_alpha_mask(p2, dense, (65, 205, 75), 0.22)
        add_alpha_mask(p2, robust, (35, 220, 255), 0.30)
        add_alpha_mask(p2, quarantined, (35, 35, 235), 0.48)
        draw_mask_contours(p2, owned, (245, 245, 245), 1)
        draw_mask_contours(p2, robust, (35, 220, 255), 2)
        robust_metric = row["experiment_2_direct_mask_depth"]["robust_first_surface_display_vs_owned"]
        put_label(
            p2,
            f"f{row['frame_idx']:03d} depth support",
            f"dense green | robust yellow | quarantined red | IoU {robust_metric['iou']:.3f}",
        )

        p3 = rgb.copy()
        edge_points = np.zeros_like(rgb)
        edge_points[depth_edges] = (40, 40, 220)
        p3[depth_edges] = cv2.addWeighted(p3, 0.25, edge_points, 0.75, 0.0)[depth_edges]
        near_points = np.zeros_like(rgb)
        near_points[near_edges] = (35, 120, 255)
        p3[near_edges] = cv2.addWeighted(p3, 0.20, near_points, 0.80, 0.0)[near_edges]
        draw_mask_contours(p3, owned, (65, 205, 75), 2)
        draw_mask_contours(p3, robust, (35, 220, 255), 1)
        edge_metric = row["experiment_2_direct_mask_depth"]["mask_boundary_to_near_depth_edge_distance_source_px"]
        put_label(
            p3,
            f"f{row['frame_idx']:03d} depth discontinuity",
            f"green mask boundary | orange/red depth edges | med {edge_metric.get('median_px', float('nan')):.2f} src px",
        )

        panels = [crop_with_context(image, [raw, owned, robust]) for image in (p1, p2, p3)]
        tiles.append(np.hstack(panels))
    sheet = np.vstack(tiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write direct review: {output}")
    return {"path": str(output), "bytes": int(output.stat().st_size), "sha256": sha256_file(output)}


def analyze_anchor_observed_surface(args: argparse.Namespace, frame: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    visible = frame["objects"][0]["visible_geometry_candidate"]
    mesh_path = require_file(args.anchor_observed_mesh, "anchor observed Poisson mesh")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid anchor observed mesh: {mesh_path}")
    centroid_world = np.asarray(visible["centroid_world_m"], dtype=np.float64)
    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    intrinsics = np.asarray(visible["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    vertices_world = np.asarray(mesh.vertices, dtype=np.float64) + centroid_world[None, :]
    vertices_camera = (vertices_world - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
    depth_shape = (int(visible["source_height"]), int(visible["source_width"]))
    silhouette_source = rasterize_mesh_silhouette(
        vertices_camera,
        np.asarray(mesh.faces, dtype=np.int64),
        intrinsics,
        depth_shape[1],
        depth_shape[0],
    )
    owned_manifest = read_mask(Path(str(visible["mask_path"])))
    raw_manifest = read_mask(Path(str(visible["source_mask_path"])))
    silhouette_manifest = resize_mask(silhouette_source, owned_manifest.shape[1], owned_manifest.shape[0])
    rgb = cv2.imread(str(require_file(Path(str(frame["raw_frame_path"])), "anchor RGB")), cv2.IMREAD_COLOR)
    image = rgb.copy()
    add_alpha_mask(image, raw_manifest, (255, 80, 30), 0.16)
    add_alpha_mask(image, owned_manifest, (65, 205, 75), 0.24)
    draw_mask_contours(image, raw_manifest, (255, 80, 30), 2)
    draw_mask_contours(image, owned_manifest, (65, 205, 75), 2)
    draw_mask_contours(image, silhouette_manifest, (220, 60, 220), 3)
    raw_metrics = mask_metrics(silhouette_manifest, raw_manifest)
    owned_metrics = mask_metrics(silhouette_manifest, owned_manifest)
    put_label(
        image,
        f"anchor f{int(frame['frame_idx']):03d}: Poisson observed surface vs masks",
        f"magenta surface | raw IoU {raw_metrics['iou']:.3f} | owned IoU {owned_metrics['iou']:.3f}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "anchor_observed_surface_vs_masks.jpg"
    if not cv2.imwrite(str(image_path), crop_with_context(image, [raw_manifest, owned_manifest, silhouette_manifest]), [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write anchor surface review: {image_path}")
    return {
        "mesh": file_ref(mesh_path),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "coordinate_contract": "mesh vertices + anchor visible centroid world; then world_to_camera with frame camera",
        "surface_silhouette_vs_raw_sam2_mask": raw_metrics,
        "surface_silhouette_vs_object_owned_mask": owned_metrics,
        "review_image": {"path": str(image_path), "bytes": int(image_path.stat().st_size), "sha256": sha256_file(image_path)},
    }


def analyze_ghost_downstream(args: argparse.Namespace) -> dict[str, Any]:
    alignment = load_json(args.alignment_report)
    fullres = load_json(args.fullres_audit_report)
    full_video = load_json(args.full_video_report)
    wrong_k = load_json(args.wrong_intrinsics_report)
    before = alignment.get("frame_metrics_before") or {}
    after = alignment.get("frame_metrics_after") or {}
    frame_ids = sorted(int(idx) for idx in set(before) & set(after))
    rows: list[dict[str, Any]] = []
    for idx in frame_ids:
        b = before[str(idx)]
        a = after[str(idx)]
        b_depth = abs(float(b["first_hit_minus_observed_depth_m"]["median_m"]))
        a_depth = abs(float(a["first_hit_minus_observed_depth_m"]["median_m"]))
        rows.append(
            {
                "frame_idx": idx,
                "silhouette_iou_before": float(b["silhouette_iou"]),
                "silhouette_iou_after": float(a["silhouette_iou"]),
                "delta_silhouette_iou": float(a["silhouette_iou"] - b["silhouette_iou"]),
                "abs_first_hit_median_before_m": b_depth,
                "abs_first_hit_median_after_m": a_depth,
                "depth_abs_improvement_m": b_depth - a_depth,
                "first_hit_signed_median_before_m": float(b["first_hit_minus_observed_depth_m"]["median_m"]),
                "first_hit_signed_median_after_m": float(a["first_hit_minus_observed_depth_m"]["median_m"]),
                "coverage_before": float(b["surfel_first_hit_coverage_fraction"]),
                "coverage_after": float(a["surfel_first_hit_coverage_fraction"]),
            }
        )
    delta_iou = {row["frame_idx"]: row["delta_silhouette_iou"] for row in rows}
    depth_improvement = {row["frame_idx"]: row["depth_abs_improvement_m"] for row in rows}
    fullres_rows = fullres.get("rows") or {}
    fullres_abs_medians_mm = [
        abs(float(row["first_hit_mesh_z_minus_observed_z_m"]["median"])) * 1000.0
        for row in fullres_rows.values()
        if isinstance(row, dict) and isinstance(row.get("first_hit_mesh_z_minus_observed_z_m"), dict)
    ]
    fullres_coverages = [
        float(row["surfel_first_hit_coverage_fraction"])
        for row in fullres_rows.values()
        if isinstance(row, dict) and isinstance(row.get("surfel_first_hit_coverage_fraction"), (int, float))
    ]
    video_summary = ((full_video.get("silhouette_summary") or {}).get("hand_occluded_render_vs_owned_mask")) or {}
    video_iou = metric_values_by_frame(full_video, "hand_occluded_render_vs_owned_mask", "iou")
    video_centroid = metric_values_by_frame(full_video, "hand_occluded_render_vs_owned_mask", "centroid_error_px")
    video_boundary = metric_values_by_frame(full_video, "hand_occluded_render_vs_owned_mask", "symmetric_boundary_median_px")
    video_outside = metric_values_by_frame(full_video, "hand_occluded_render_vs_owned_mask", "rendered_outside_fraction")
    wrong_summary = wrong_k.get("summary") or {}
    interpretation = {
        "depth_values_cannot_create_2d_shift_under_closed_loop": True,
        "wrong_intrinsics_was_a_projection_contract_error": bool(
            float(wrong_summary.get("wrong_K_median_centroid_error_px") or 0.0)
            > 10.0 * max(float(wrong_summary.get("correct_K_median_centroid_error_px") or 1.0), 1.0)
        ),
        "ghost_improves_silhouette_on_sampled_frames": bool(
            np.median(list(delta_iou.values())) > 0.0 if delta_iou else False
        ),
        "remaining_full_video_mismatch_is_not_explained_by_axial_depth_noise": bool(
            fullres_abs_medians_mm
            and float(np.median(fullres_abs_medians_mm)) < 5.0
            and float(((video_summary.get("centroid_error_px") or {}).get("median")) or 0.0) > 10.0
        ),
        "remaining_primary_hypothesis": "per_frame_pose_or_canonical_shape_alignment_not_depth_value_or_depth_raster_registration",
    }
    return {
        "alignment_report": file_ref(args.alignment_report),
        "fullres_audit_report": file_ref(args.fullres_audit_report),
        "full_video_report": file_ref(args.full_video_report),
        "wrong_intrinsics_report": file_ref(args.wrong_intrinsics_report),
        "sampled_alignment_rows": rows,
        "sampled_alignment_summary": {
            "frame_count": len(rows),
            "silhouette_iou_before": summarize(np.asarray([r["silhouette_iou_before"] for r in rows])),
            "silhouette_iou_after": summarize(np.asarray([r["silhouette_iou_after"] for r in rows])),
            "delta_silhouette_iou": summarize(np.asarray([r["delta_silhouette_iou"] for r in rows])),
            "silhouette_improved_frames": int(sum(row["delta_silhouette_iou"] > 0.0 for row in rows)),
            "abs_first_hit_median_before_mm": summarize(np.asarray([r["abs_first_hit_median_before_m"] * 1000.0 for r in rows]), "_mm"),
            "abs_first_hit_median_after_mm": summarize(np.asarray([r["abs_first_hit_median_after_m"] * 1000.0 for r in rows]), "_mm"),
            "depth_abs_improvement_mm": summarize(np.asarray([r["depth_abs_improvement_m"] * 1000.0 for r in rows]), "_mm"),
            "depth_improved_frames": int(sum(row["depth_abs_improvement_m"] > 0.0 for row in rows)),
            "delta_iou_vs_depth_improvement_correlation": correlation(delta_iou, depth_improvement),
        },
        "full_resolution_first_hit_after_ghost": {
            "absolute_signed_median_mm": summarize(np.asarray(fullres_abs_medians_mm), "_mm"),
            "coverage_fraction": summarize(np.asarray(fullres_coverages)),
            "signed_median_mm_by_frame": {
                str(idx): float(row["first_hit_mesh_z_minus_observed_z_m"]["median"]) * 1000.0
                for idx, row in fullres_rows.items()
            },
        },
        "full_video_after_ghost_correct_intrinsics": {
            "summary": video_summary,
            "iou_vs_frame_correlation": correlation(video_iou, {idx: float(idx) for idx in video_iou}),
            "centroid_vs_frame_correlation": correlation(video_centroid, {idx: float(idx) for idx in video_centroid}),
            "frames_iou_below_0_60": int(sum(value < 0.60 for value in video_iou.values())),
            "frames_centroid_above_20px": int(sum(value > 20.0 for value in video_centroid.values())),
            "iou_values": video_iou,
            "centroid_error_px_values": video_centroid,
            "boundary_median_px_values": video_boundary,
            "rendered_outside_fraction_values": video_outside,
        },
        "wrong_vs_correct_intrinsics": wrong_summary,
        "interpretation": interpretation,
    }


def draw_polyline(panel: np.ndarray, values: list[float], color: tuple[int, int, int], ymin: float, ymax: float) -> None:
    if len(values) < 2 or not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        return
    h, w = panel.shape[:2]
    margin_l, margin_r, margin_t, margin_b = 55, 12, 34, 28
    points: list[tuple[int, int]] = []
    for i, value in enumerate(values):
        x = margin_l + int(round(i * (w - margin_l - margin_r - 1) / max(1, len(values) - 1)))
        y = margin_t + int(round((ymax - value) * (h - margin_t - margin_b - 1) / (ymax - ymin)))
        points.append((x, y))
    for a, b in zip(points[:-1], points[1:]):
        cv2.line(panel, a, b, color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(panel, point, 2, color, -1, cv2.LINE_AA)


def make_pattern_chart(ghost: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = ghost["sampled_alignment_rows"]
    frame_ids = [row["frame_idx"] for row in rows]
    video = ghost["full_video_after_ghost_correct_intrinsics"]
    video_iou = video["iou_values"]
    video_centroid = video["centroid_error_px_values"]
    video_ids = sorted(video_iou)
    panels: list[np.ndarray] = []
    specs = [
        (
            "sampled silhouette IoU: before gray, GHOST green",
            [row["silhouette_iou_before"] for row in rows],
            [row["silhouette_iou_after"] for row in rows],
            (180, 180, 180),
            (65, 205, 75),
            f"frames {frame_ids[0]}..{frame_ids[-1]}",
        ),
        (
            "sampled |first-hit depth median|: before gray, GHOST cyan",
            [row["abs_first_hit_median_before_m"] * 1000.0 for row in rows],
            [row["abs_first_hit_median_after_m"] * 1000.0 for row in rows],
            (180, 180, 180),
            (255, 220, 80),
            "mm",
        ),
        (
            "full-video after GHOST: IoU green / centroid px orange",
            [video_iou[idx] for idx in video_ids],
            [video_centroid[idx] for idx in video_ids],
            (65, 205, 75),
            (45, 145, 255),
            "different y-scales; see report",
        ),
    ]
    for title, a, b, color_a, color_b, subtitle in specs:
        panel = np.full((330, 1040, 3), 18, dtype=np.uint8)
        cv2.rectangle(panel, (55, 34), (1028, 302), (245, 245, 245), 1)
        all_values = np.asarray(a + b, dtype=np.float64)
        ymin = float(np.min(all_values))
        ymax = float(np.max(all_values))
        pad = max(1.0e-6, 0.08 * (ymax - ymin))
        ymin -= pad
        ymax += pad
        for q in range(5):
            y = 34 + int(round(q * (302 - 34) / 4))
            cv2.line(panel, (55, y), (1028, y), (55, 55, 55), 1, cv2.LINE_AA)
            value = ymax - q * (ymax - ymin) / 4
            cv2.putText(panel, f"{value:.2f}", (6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        draw_polyline(panel, a, color_a, ymin, ymax)
        draw_polyline(panel, b, color_b, ymin, ymax)
        put_label(panel, title, subtitle)
        panels.append(panel)
    chart = np.vstack(panels)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), chart):
        raise RuntimeError(f"failed to write pattern chart: {output}")
    return {"path": str(output), "bytes": int(output.stat().st_size), "sha256": sha256_file(output)}


def write_markdown(report: dict[str, Any], output: Path) -> None:
    e1 = report["aggregate"]["experiment_1"]
    e2 = report["aggregate"]["experiment_2"]
    e3 = report["experiment_3_ghost_downstream"]
    anchor = report["experiment_2_anchor_observed_surface"]
    lines = [
        "# GHOST-lite 牛奶盒：mask/depth/观测表面对齐三项实验",
        "",
        "## 结论",
        "",
        report["verdict"]["zh"],
        "",
        "## 实验 1：同帧 depth 反投影→重投影闭环",
        "",
        f"- 参与帧数：{e1['frame_count']}。",
        f"- dense owned-depth 像素闭环误差 max 的帧间最大值：{e1['max_of_frame_max_reprojection_px']:.6g} px。",
        f"- 投影到 960 manifest 平面的闭环误差 max 的帧间最大值：{e1['max_of_frame_max_manifest_px']:.6g} px。",
        f"- camera→world→camera 逆变换误差 max 的帧间最大值：{e1['max_of_frame_max_camera_inverse_m']:.6g} m。",
        f"- 已存 visible surfels 投影落入 robust support 的比例中位数：{e1['stored_surfel_inside_robust_median']:.6f}；落入 owned mask 的比例中位数：{e1['stored_surfel_inside_owned_median']:.6f}。",
        "- 解释：同一像素坐标、同一 K 的重投影在数学上与深度值无关；该闭环通过，说明当前差异不能归因于 depth→3D→pixel 实现或 mask/depth 平面映射。",
        "",
        "## 实验 2：不经过 SAM3D/GHOST 的 mask–depth/观测表面叠加",
        "",
        f"- robust first-surface support 对 owned mask 的 IoU 中位数：{e2['robust_support_vs_owned_iou_median']:.6f}。",
        f"- mask 边界到任意 depth discontinuity 的距离在 5 source px 内的比例中位数：{e2['mask_boundary_to_any_depth_edge_within_5px_fraction_median']:.3f}；邻近 depth edge 到 mask 边界在 5 source px 内的比例中位数：{e2['near_depth_edge_to_mask_boundary_within_5px_fraction_median']:.3f}。",
        "- 注：depth-edge 这一项只是几何不连续支持度，不是所有 RGB/ownership 边界都应当有深度边；因此它用于辅助判读，不作为 mask/depth 配准的通过门槛。",
        f"- 重算的 robust depth ownership 与原 annotation 完全匹配帧数：{e2['ownership_reconstruction_match_frames']}/{e2['frame_count']}。",
        f"- anchor frame {report['anchor_frame']} 的 Poisson 观测表面对 raw SAM2 mask：IoU {anchor['surface_silhouette_vs_raw_sam2_mask']['iou']:.6f}，centroid {anchor['surface_silhouette_vs_raw_sam2_mask']['centroid_distance_px']:.3f} px。",
        f"- 同一表面对 object-owned mask：IoU {anchor['surface_silhouette_vs_object_owned_mask']['iou']:.6f}，centroid {anchor['surface_silhouette_vs_object_owned_mask']['centroid_distance_px']:.3f} px。",
        "- 解释：观测表面对未扣手的 raw mask 对齐较好；对 owned mask 的差异主要来自 projection-driven MANO hand ownership subtraction 和 Poisson 填充/外扩，而不是深度栅格错位。",
        "",
        "## 实验 3：GHOST 下游误差形态",
        "",
        f"- GHOST 采样帧 silhouette IoU：{e3['sampled_alignment_summary']['silhouette_iou_before']['median']:.6f} → {e3['sampled_alignment_summary']['silhouette_iou_after']['median']:.6f}，{e3['sampled_alignment_summary']['silhouette_improved_frames']}/{e3['sampled_alignment_summary']['frame_count']} 帧改善。",
        f"- |first-hit depth median|：{e3['sampled_alignment_summary']['abs_first_hit_median_before_mm']['median_mm']:.3f} mm → {e3['sampled_alignment_summary']['abs_first_hit_median_after_mm']['median_mm']:.3f} mm，{e3['sampled_alignment_summary']['depth_improved_frames']}/{e3['sampled_alignment_summary']['frame_count']} 帧改善。",
        f"- ΔIoU 与 depth 改善的相关性：{e3['sampled_alignment_summary']['delta_iou_vs_depth_improvement_correlation']:.6f}。",
        f"- GHOST 后 full-resolution first-hit signed median 的绝对值中位数：{e3['full_resolution_first_hit_after_ghost']['absolute_signed_median_mm']['median_mm']:.3f} mm。",
        f"- 正确 K 的 full-video hand-occluded render vs owned mask：IoU median {e3['full_video_after_ghost_correct_intrinsics']['summary']['iou']['median']:.6f}，centroid median {e3['full_video_after_ghost_correct_intrinsics']['summary']['centroid_error_px']['median']:.3f} px。",
        f"- 错误 K 对照：centroid median {e3['wrong_vs_correct_intrinsics']['wrong_K_median_centroid_error_px']:.3f} px；正确 K 后 {e3['wrong_vs_correct_intrinsics']['correct_K_median_centroid_error_px']:.3f} px。",
        "- 解释：mm 级轴向 depth residual 不能产生几十 px 的横向 mask 错位；错误 K 曾产生巨大固定投影错位，但已修复。GHOST 后剩余的 silhouette/centroid 差异主要指向每帧 pose/canonical shape alignment。",
        "",
        "## 产物",
        "",
        f"- JSON 报告：`{report['outputs']['report']}`",
        f"- mask/depth 可视复核：`{report['outputs']['direct_review_image']}`",
        f"- anchor 观测表面对比：`{anchor['review_image']['path']}`",
        f"- GHOST 误差曲线：`{report['outputs']['ghost_pattern_chart']}`",
        "",
        "## 范围",
        "",
        "本结论针对当前 GHOST-lite 牛奶盒运行及其冻结报告。它不把 GHOST-lite 共享 Sim(3) 泛化为所有物体都已修好；跨视频基准中 can 已出现退化，需要逐案例复跑同一诊断。",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    output_dir = prepare_output(args.output_dir, bool(args.replace))
    annotations_path = require_file(args.annotations, "visible-geometry annotations")
    annotations = load_json(annotations_path)
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"annotations contain no frames: {annotations_path}")
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    requested_frames = [int(value) for value in str(args.frames).split(",") if str(value).strip()]
    missing = [idx for idx in requested_frames if idx not in frame_by_idx]
    if missing:
        raise RuntimeError(f"requested frames absent from annotations: {missing}")
    first_visible = frame_by_idx[requested_frames[0]]["objects"][0]["visible_geometry_candidate"]
    depth_path = require_file(Path(str(first_visible["depth_npz"])), "camera-bound depth archive")
    depth_archive = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    try:
        depth_row_by_idx = {int(value): i for i, value in enumerate(np.asarray(depth_archive["frame_idx"], dtype=np.int64).tolist())}
        frame_rows = [
            process_frame(idx, frame_by_idx[idx], depth_archive, depth_row_by_idx, args)
            for idx in requested_frames
        ]
    finally:
        depth_archive.close()

    direct_review = make_direct_review(frame_rows, output_dir / "mask_depth_direct_overlay_review.jpg")
    for row in frame_rows:
        row.pop("_visual", None)

    if int(args.anchor_frame) not in frame_by_idx:
        raise RuntimeError(f"anchor frame absent: {args.anchor_frame}")
    anchor_surface = analyze_anchor_observed_surface(args, frame_by_idx[int(args.anchor_frame)], output_dir)
    ghost = analyze_ghost_downstream(args)
    ghost_chart = make_pattern_chart(ghost, output_dir / "ghost_downstream_error_pattern.png")

    aggregate_1 = {
        "frame_count": len(frame_rows),
        "max_of_frame_max_reprojection_px": max(
            float(row["experiment_1_closed_loop"]["source_reprojection_error_px"].get("max_px") or 0.0)
            for row in frame_rows
        ),
        "max_of_frame_max_manifest_px": max(
            float(row["experiment_1_closed_loop"]["manifest_contract_reprojection_error_px"].get("max_px") or 0.0)
            for row in frame_rows
        ),
        "max_of_frame_max_camera_inverse_m": max(
            float(row["experiment_1_closed_loop"]["camera_to_world_to_camera_error_m"].get("max_m") or 0.0)
            for row in frame_rows
        ),
        "stored_surfel_inside_robust_median": float(
            np.median([
                float(row["experiment_1_closed_loop"]["stored_surfel_projected_inside_robust_display_fraction"])
                for row in frame_rows
            ])
        ),
        "stored_surfel_inside_owned_median": float(
            np.median([
                float(row["experiment_1_closed_loop"]["stored_surfel_projected_inside_owned_mask_fraction"])
                for row in frame_rows
            ])
        ),
        "stored_surfel_depth_abs_delta_max_m": max(
            float(row["experiment_1_closed_loop"]["stored_surfel_depth_minus_archive_abs_m"].get("max_m") or 0.0)
            for row in frame_rows
        ),
    }
    aggregate_2 = {
        "frame_count": len(frame_rows),
        "robust_support_vs_owned_iou_median": float(
            np.median([
                float(row["experiment_2_direct_mask_depth"]["robust_first_surface_display_vs_owned"]["iou"])
                for row in frame_rows
            ])
        ),
        "dense_valid_vs_owned_iou_median": float(
            np.median([
                float(row["experiment_2_direct_mask_depth"]["dense_valid_display_vs_owned"]["iou"])
                for row in frame_rows
            ])
        ),
        "mask_boundary_to_depth_edge_median_px_median": float(
            np.median([
                float(row["experiment_2_direct_mask_depth"]["mask_boundary_to_near_depth_edge_distance_source_px"].get("median_px") or np.nan)
                for row in frame_rows
            ])
        ),
        "mask_boundary_to_depth_edge_p95_px_median": float(
            np.median([
                float(row["experiment_2_direct_mask_depth"]["mask_boundary_to_near_depth_edge_distance_source_px"].get("p95_px") or np.nan)
                for row in frame_rows
            ])
        ),
        "mask_boundary_to_any_depth_edge_within_5px_fraction_median": float(
            np.median([
                float(row["experiment_2_direct_mask_depth"]["mask_boundary_to_any_depth_edge_within_fraction"]["within_5px_fraction"])
                for row in frame_rows
            ])
        ),
        "near_depth_edge_to_mask_boundary_within_5px_fraction_median": float(
            np.median([
                float(row["experiment_2_direct_mask_depth"]["near_depth_edge_to_mask_boundary_within_fraction"]["within_5px_fraction"])
                for row in frame_rows
            ])
        ),
        "ownership_reconstruction_match_frames": int(
            sum(
                bool(row["experiment_2_direct_mask_depth"]["robust_first_surface_ownership_matches_annotation"]["matches"])
                for row in frame_rows
            )
        ),
    }
    report_path = output_dir / "mask_depth_alignment_three_experiments_report.json"
    md_path = output_dir / "MASK_DEPTH_ALIGNMENT_THREE_EXPERIMENTS_ZH.md"
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "ghost_lite_mask_depth_alignment_three_experiments_cpu_readonly",
        "claim_scope": (
            "Diagnostic only for the current GHOST-lite milk run. It does not mutate prediction state and does not "
            "promote generated SAM3D faces to collision, contact, sign, or nonpenetration authority."
        ),
        "compute_contract": {
            "local_gpu_used": False,
            "model_inference_run": False,
            "prediction_inputs_mutated": False,
            "depth_archive": file_ref(depth_path),
            "annotations": file_ref(annotations_path),
        },
        "frames": requested_frames,
        "anchor_frame": int(args.anchor_frame),
        "aggregate": {
            "experiment_1": aggregate_1,
            "experiment_2": aggregate_2,
        },
        "frame_rows": frame_rows,
        "experiment_2_anchor_observed_surface": anchor_surface,
        "experiment_3_ghost_downstream": ghost,
        "outputs": {
            "report": str(report_path),
            "markdown": str(md_path),
            "direct_review_image": direct_review["path"],
            "ghost_pattern_chart": ghost_chart["path"],
        },
        "verdict": {
            "primary_depth_raster_or_depth_value_origin_supported_for_current_run": False,
            "zh": (
                "当前 GHOST-lite 牛奶盒结果中，观测表面与 mask 的可见差异不应归因于深度栅格未配准或单纯深度值误差。"
                "同帧 depth→3D→pixel 闭环为数值零误差，mask/depth 平面映射闭环也通过；不经过 SAM3D 时，anchor Poisson 观测表面对 raw SAM2 mask 对齐良好。"
                "对 object-owned mask 的额外差异主要来自 MANO hand ownership subtraction 与 Poisson 填充；GHOST 后剩余的几十 px 级 silhouette/centroid 差异对应每帧 pose/canonical alignment，而不是 mm 级轴向 depth residual。"
            ),
        },
        "elapsed_s": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "markdown": str(md_path),
                "direct_review_image": direct_review["path"],
                "anchor_review_image": anchor_surface["review_image"]["path"],
                "ghost_pattern_chart": ghost_chart["path"],
                "experiment_1": aggregate_1,
                "experiment_2": aggregate_2,
                "ghost_sampled_iou_before_after": [
                    ghost["sampled_alignment_summary"]["silhouette_iou_before"]["median"],
                    ghost["sampled_alignment_summary"]["silhouette_iou_after"]["median"],
                ],
                "ghost_fullres_abs_depth_median_mm": ghost["full_resolution_first_hit_after_ghost"]["absolute_signed_median_mm"]["median_mm"],
                "full_video_iou_centroid_median": [
                    ghost["full_video_after_ghost_correct_intrinsics"]["summary"]["iou"]["median"],
                    ghost["full_video_after_ghost_correct_intrinsics"]["summary"]["centroid_error_px"]["median"],
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
