#!/usr/bin/env python3
"""Audit whether hand ownership subtraction removes behind-object hands.

Current V19 ownership removes every projected full-MANO triangle silhouette from
an object mask. This diagnostic reconstructs that operation exactly, then
rasterizes each hand's first-hit depth and compares it with same-pixel metric
object/visible depth:

    hand_z - object_z > tolerance  => hand is behind the measured object surface
    hand_z - object_z < -tolerance => hand is in front of the measured surface

A behind-object hand projection should not erase object support. Padded pixels
without a hand first hit are reported separately, because they are caused by
morphological padding rather than a physical depth-order decision.

CPU-only and prediction-read-only.
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

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_v19_visible_geometry_from_sam2_depth import projected_mano_hand_silhouette  # noqa: E402

BASE = Path(
    "/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/"
    "backend_tests/20260819T122452Z_hot3d_milk_local_authority_local29_v1/runs/"
    "P0014_84ea2dcc_carton_milk_f2370_2519"
)
GHOST = BASE / "experiments/sam3d_native_ghost_lite_20260826"
SCHEMA = "v19_hand_depth_order_ownership_audit_v1"
REPRESENTATIVE_FRAMES = (0, 30, 50, 70, 92, 103, 110, 121, 130, 146, 149)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=BASE / "measurements/object_geometry/visible_geometry/carton_milk/annotations_v19_visible_geometry.json",
    )
    parser.add_argument("--output-dir", type=Path, default=GHOST / "hand_depth_order_ownership_audit_v1")
    parser.add_argument("--frames", default="all", help="Comma-separated frame ids or 'all'.")
    parser.add_argument("--review-frames", default=",".join(str(v) for v in REPRESENTATIVE_FRAMES))
    parser.add_argument("--depth-order-tolerance-m", type=float, default=0.005)
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


def summarize(values: np.ndarray, suffix: str = "") -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min" + suffix: float(np.min(array)),
        "median" + suffix: float(np.median(array)),
        "mean" + suffix: float(np.mean(array)),
        "p90" + suffix: float(np.percentile(array, 90.0)),
        "p95" + suffix: float(np.percentile(array, 95.0)),
        "max" + suffix: float(np.max(array)),
    }


def summarize_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return summarize(np.asarray([float(row[key]) for row in rows], dtype=np.float64))


def resize_mask_nearest_exact(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    interpolation = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    return cv2.resize(mask.astype(np.uint8), (int(width), int(height)), interpolation=interpolation) > 0


def affine_mask_from_source(visible: dict[str, Any], mask_shape: tuple[int, int], depth_shape: tuple[int, int]) -> np.ndarray:
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


def project_camera(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    z = points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (z > 1.0e-9)
    uv[valid, 0] = K[0, 0] * points[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * points[valid, 1] / z[valid] + K[1, 2]
    return uv, valid


def rasterize_first_hit_z(
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Small CPU first-hit rasterizer for a MANO-sized triangle mesh."""
    vertices = np.asarray(vertices_camera, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    uv, valid_vertices = project_camera(vertices, K)
    z = vertices[:, 2]
    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    for tri in faces:
        if not np.all(valid_vertices[tri]):
            continue
        pts = uv[tri]
        zs = z[tri]
        if not np.all(np.isfinite(pts)) or not np.all(zs > 1.0e-9):
            continue
        x0 = max(0, int(math.floor(float(np.min(pts[:, 0])))))
        x1 = min(width - 1, int(math.ceil(float(np.max(pts[:, 0])))))
        y0 = max(0, int(math.floor(float(np.min(pts[:, 1])))))
        y1 = min(height - 1, int(math.ceil(float(np.max(pts[:, 1])))))
        if x1 < x0 or y1 < y0:
            continue
        ax, ay = pts[0]
        bx, by = pts[1]
        cx, cy = pts[2]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(float(denom)) < 1.0e-12:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        w0 = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / denom
        w1 = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not np.any(inside):
            continue
        z_interp = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        view = zbuf[y0 : y1 + 1, x0 : x1 + 1]
        update = inside & (z_interp < view)
        view[update] = z_interp[update]
    return np.where(np.isfinite(zbuf), zbuf, np.nan)


def hand_geometry_for_row(hand: dict[str, Any], bridge_cache: dict[Path, Any], hawor_cache: dict[Path, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    state = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
    reference = state.get("vertices_reference") if isinstance(state.get("vertices_reference"), dict) else {}
    bridge_path = require_file(Path(str(reference.get("bridge_npz") or "")), "MANO bridge")
    hawor_path = require_file(Path(str(reference.get("source_hawor_npz") or "")), "HaWoR source")
    row_index = int(reference.get("bridge_row_index", -1))
    camera_key = str(reference.get("bridge_vertices_camera_array") or "vertices_current_v18_camera_m")
    side = str(hand.get("hand_side") or "").lower()
    if side not in ("left", "right"):
        raise RuntimeError(f"invalid hand side {side!r}")
    if bridge_path not in bridge_cache:
        bridge_cache[bridge_path] = np.load(bridge_path, mmap_mode="r", allow_pickle=False)
    if hawor_path not in hawor_cache:
        hawor_cache[hawor_path] = np.load(hawor_path, mmap_mode="r", allow_pickle=False)
    bridge = bridge_cache[bridge_path]
    hawor = hawor_cache[hawor_path]
    if camera_key not in bridge.files or row_index < 0 or row_index >= len(bridge[camera_key]):
        raise RuntimeError(f"invalid bridge reference {camera_key}:{row_index}")
    if "frame_idx" in bridge.files and int(bridge["frame_idx"][row_index]) != int(state.get("case_frame_idx")):
        raise RuntimeError("bridge frame mismatch")
    if "hand_side" in bridge.files and str(bridge["hand_side"][row_index]) != side:
        raise RuntimeError("bridge hand side mismatch")
    faces_key = f"{side}_faces"
    if faces_key not in hawor.files:
        raise RuntimeError(f"HaWoR archive lacks {faces_key}")
    vertices = np.asarray(bridge[camera_key][row_index], dtype=np.float64)
    faces = np.asarray(hawor[faces_key], dtype=np.int64)
    return vertices, faces, {"bridge": bridge_path, "hawor": hawor_path, "row_index": row_index, "side": side}


def classify_delta(delta: np.ndarray, tolerance_m: float) -> np.ndarray:
    classes = np.full(delta.shape, "no_depth", dtype=object)
    classes[~np.isfinite(delta)] = "no_depth"
    classes[np.isfinite(delta) & (delta < -float(tolerance_m))] = "hand_in_front"
    classes[np.isfinite(delta) & (delta > float(tolerance_m))] = "hand_behind"
    classes[np.isfinite(delta) & (np.abs(delta) <= float(tolerance_m))] = "ambiguous_near_contact"
    return classes


def draw_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    color_image = np.zeros_like(image)
    color_image[:] = color
    image[mask] = cv2.addWeighted(image, 1.0 - alpha, color_image, alpha, 0.0)[mask]


def put_label(image: np.ndarray, title: str, subtitle: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 48), (12, 12, 12), -1)
    cv2.putText(image, title[:92], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, subtitle[:116], (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 225, 225), 1, cv2.LINE_AA)


def crop_to_content(image: np.ndarray, masks: list[np.ndarray], pad: int = 75, square: int = 420) -> np.ndarray:
    combined = np.zeros(image.shape[:2], dtype=bool)
    for mask in masks:
        combined |= mask
    ys, xs = np.where(combined)
    if not len(xs):
        return cv2.resize(image, (square, square), interpolation=cv2.INTER_AREA)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad + 1)
    y1 = min(image.shape[0], int(ys.max()) + pad + 1)
    crop = image[y0:y1, x0:x1]
    scale = float(square) / float(max(crop.shape[:2]))
    resized = cv2.resize(crop, (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    canvas = np.full((square, square, 3), 18, dtype=np.uint8)
    y = (square - resized.shape[0]) // 2
    x = (square - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def process_frame(
    frame: dict[str, Any],
    depth_archive: Any,
    depth_row_by_idx: dict[int, int],
    bridge_cache: dict[Path, Any],
    hawor_cache: dict[Path, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_idx = int(frame["frame_idx"])
    object_rows = frame.get("objects") if isinstance(frame.get("objects"), list) else []
    if len(object_rows) != 1:
        raise RuntimeError(f"frame {frame_idx}: expected one object")
    visible = object_rows[0].get("visible_geometry_candidate")
    if not isinstance(visible, dict):
        raise RuntimeError(f"frame {frame_idx}: lacks visible geometry")
    raw_mask = read_mask(Path(str(visible["source_mask_path"])))
    owned_mask = read_mask(Path(str(visible["mask_path"])))
    depth_row = depth_row_by_idx.get(frame_idx)
    if depth_row is None:
        raise RuntimeError(f"frame {frame_idx}: missing depth row")
    depth_source = np.asarray(depth_archive["depth"][depth_row], dtype=np.float64)
    depth = cv2.resize(
        depth_source,
        (raw_mask.shape[1], raw_mask.shape[0]),
        interpolation=getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST),
    )
    A_mask = affine_mask_from_source(visible, raw_mask.shape, depth_source.shape)
    K_source_4 = np.asarray(visible["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    K_source = np.asarray(
        [[K_source_4[0], 0.0, K_source_4[2]], [0.0, K_source_4[1], K_source_4[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    K_mask = A_mask @ K_source

    ownership = visible.get("object_surface_ownership_filter") if isinstance(visible.get("object_surface_ownership_filter"), dict) else {}
    pad_px = int(ownership.get("padding_px_in_mask_coordinates") or 0)
    union_silhouette = np.zeros_like(raw_mask)
    union_unpadded_silhouette = np.zeros_like(raw_mask)
    union_first_hit = np.full(raw_mask.shape, np.nan, dtype=np.float64)
    per_hand: list[dict[str, Any]] = []
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        side = str(hand.get("hand_side") or "").lower()
        if side not in ("left", "right"):
            continue
        silhouette, silhouette_record = projected_mano_hand_silhouette(
            hand,
            mask_shape=raw_mask.shape,
            source_width=int(visible["source_width"]),
            source_height=int(visible["source_height"]),
            pad_px=pad_px,
            A_mask_from_source=A_mask,
        )
        silhouette_unpadded, _unpadded_record = projected_mano_hand_silhouette(
            hand,
            mask_shape=raw_mask.shape,
            source_width=int(visible["source_width"]),
            source_height=int(visible["source_height"]),
            pad_px=0,
            A_mask_from_source=A_mask,
        )
        vertices, faces, geometry_ref = hand_geometry_for_row(hand, bridge_cache, hawor_cache)
        z_hit = rasterize_first_hit_z(vertices, faces, K_mask, raw_mask.shape[1], raw_mask.shape[0])
        hand_vertex_uv, hand_vertex_valid = project_camera(vertices, K_mask)
        hand_vertex_uv_i = np.rint(hand_vertex_uv[hand_vertex_valid]).astype(np.int64)
        vertex_inside = (
            (hand_vertex_uv_i[:, 0] >= 0)
            & (hand_vertex_uv_i[:, 0] < raw_mask.shape[1])
            & (hand_vertex_uv_i[:, 1] >= 0)
            & (hand_vertex_uv_i[:, 1] < raw_mask.shape[0])
        )
        hand_vertex_z = vertices[hand_vertex_valid, 2]
        depth_at_hand_vertices = np.asarray([], dtype=np.float64)
        depth_minus_hand_vertex_z = np.asarray([], dtype=np.float64)
        if np.any(vertex_inside):
            uv_used = hand_vertex_uv_i[vertex_inside]
            z_used = hand_vertex_z[vertex_inside]
            depth_at_hand_vertices = depth[uv_used[:, 1], uv_used[:, 0]]
            finite_vertex_depth = np.isfinite(depth_at_hand_vertices) & (depth_at_hand_vertices > 0.0)
            depth_at_hand_vertices = depth_at_hand_vertices[finite_vertex_depth]
            depth_minus_hand_vertex_z = depth_at_hand_vertices - z_used[finite_vertex_depth]
        union_silhouette |= silhouette
        union_unpadded_silhouette |= silhouette_unpadded
        union_first_hit = np.where(np.isfinite(z_hit), np.fmin(union_first_hit, z_hit), union_first_hit)
        overlap = raw_mask & silhouette
        valid = overlap & np.isfinite(depth) & (depth > 0.0)
        hand_valid = valid & np.isfinite(z_hit)
        unpadded_overlap = raw_mask & silhouette_unpadded
        unpadded_valid = unpadded_overlap & np.isfinite(depth) & (depth > 0.0)
        delta = np.full(raw_mask.shape, np.nan, dtype=np.float64)
        delta[hand_valid] = z_hit[hand_valid] - depth[hand_valid]
        classes = np.full(raw_mask.shape, "not_removed", dtype=object)
        classes[valid & ~np.isfinite(z_hit)] = "no_hand_first_hit_padding_or_raster_gap"
        class_values = classify_delta(delta[hand_valid], float(args.depth_order_tolerance_m))
        classes[hand_valid] = class_values
        per_hand.append(
            {
                "side": side,
                "geometry": {k: str(v) if isinstance(v, Path) else v for k, v in geometry_ref.items()},
                "silhouette_record": silhouette_record,
                "raw_mask_overlap_pixels": int(np.count_nonzero(overlap)),
                "overlap_with_valid_object_depth_pixels": int(np.count_nonzero(valid)),
                "overlap_without_hand_first_hit_pixels": int(np.count_nonzero(valid & ~np.isfinite(z_hit))),
                "padding_only_overlap_without_hand_first_hit_pixels": int(np.count_nonzero(valid & ~silhouette_unpadded & ~np.isfinite(z_hit))),
                "unpadded_silhouette_raster_gap_pixels": int(np.count_nonzero(unpadded_valid & ~np.isfinite(z_hit))),
                "hand_in_front_pixels": int(np.count_nonzero(classes == "hand_in_front")),
                "hand_behind_pixels": int(np.count_nonzero(classes == "hand_behind")),
                "ambiguous_near_contact_pixels": int(np.count_nonzero(classes == "ambiguous_near_contact")),
                "hand_minus_object_depth_m": summarize(delta[hand_valid], "_m"),
                "mano_vertex_depth_probe": {
                    "projected_vertices": int(np.count_nonzero(hand_vertex_valid)),
                    "projected_vertices_inside_image": int(np.count_nonzero(vertex_inside)),
                    "finite_depth_vertex_samples": int(len(depth_minus_hand_vertex_z)),
                    "hand_vertex_z_m": summarize(hand_vertex_z[vertex_inside] if np.any(vertex_inside) else np.asarray([]), "_m"),
                    "depth_at_projected_vertices_m": summarize(depth_at_hand_vertices, "_m"),
                    "depth_minus_hand_vertex_z_m": summarize(depth_minus_hand_vertex_z, "_m"),
                    "fraction_depth_closer_than_hand_by_tolerance": float(
                        np.mean(depth_minus_hand_vertex_z < -float(args.depth_order_tolerance_m))
                    ) if len(depth_minus_hand_vertex_z) else None,
                    "semantics": "negative depth_minus_hand_vertex_z means same-ray metric depth is closer than the projected MANO vertex",
                },
            }
        )

    reproduced_owned = raw_mask & ~union_silhouette
    removed = raw_mask & ~reproduced_owned
    reproduction_xor = np.count_nonzero(reproduced_owned != owned_mask)
    valid_removed = removed & np.isfinite(depth) & (depth > 0.0)
    has_hand_hit = valid_removed & np.isfinite(union_first_hit)
    delta = np.full(raw_mask.shape, np.nan, dtype=np.float64)
    delta[has_hand_hit] = union_first_hit[has_hand_hit] - depth[has_hand_hit]
    classes = classify_delta(delta, float(args.depth_order_tolerance_m))
    classes[valid_removed & ~np.isfinite(union_first_hit)] = "no_hand_first_hit_padding_or_raster_gap"
    front_mask = valid_removed & (classes == "hand_in_front")
    behind_mask = valid_removed & (classes == "hand_behind")
    ambiguous_mask = valid_removed & (classes == "ambiguous_near_contact")
    no_hit_mask = valid_removed & (classes == "no_hand_first_hit_padding_or_raster_gap")
    padding_only_mask = valid_removed & ~union_unpadded_silhouette
    padding_only_no_hit_mask = padding_only_mask & ~np.isfinite(union_first_hit)
    raster_gap_mask = valid_removed & union_unpadded_silhouette & ~np.isfinite(union_first_hit)
    counterfactual_front_only = raw_mask & ~front_mask
    counterfactual_front_or_ambiguous = raw_mask & ~(front_mask | ambiguous_mask)
    row = {
        "frame_idx": frame_idx,
        "paths": {
            "rgb": str(frame["raw_frame_path"]),
            "raw_mask": str(visible["source_mask_path"]),
            "owned_mask": str(visible["mask_path"]),
        },
        "mask_counts": {
            "raw_mask_pixels": int(np.count_nonzero(raw_mask)),
            "owned_mask_pixels": int(np.count_nonzero(owned_mask)),
            "removed_pixels": int(np.count_nonzero(removed)),
            "removed_fraction_of_raw": float(np.count_nonzero(removed) / max(1, np.count_nonzero(raw_mask))),
            "reproduced_owned_mask_pixels": int(np.count_nonzero(reproduced_owned)),
            "reproduction_xor_pixels_vs_stored_owned": int(reproduction_xor),
            "ownership_reproduced_exactly": bool(reproduction_xor == 0),
        },
        "depth_order": {
            "tolerance_m": float(args.depth_order_tolerance_m),
            "valid_removed_depth_pixels": int(np.count_nonzero(valid_removed)),
            "hand_in_front_pixels": int(np.count_nonzero(front_mask)),
            "hand_behind_pixels": int(np.count_nonzero(behind_mask)),
            "ambiguous_near_contact_pixels": int(np.count_nonzero(ambiguous_mask)),
            "no_hand_first_hit_padding_or_raster_gap_pixels": int(np.count_nonzero(no_hit_mask)),
            "padding_only_removed_pixels": int(np.count_nonzero(padding_only_mask)),
            "padding_only_no_hand_hit_pixels": int(np.count_nonzero(padding_only_no_hit_mask)),
            "unpadded_silhouette_raster_gap_pixels": int(np.count_nonzero(raster_gap_mask)),
            "hand_in_front_fraction_of_valid_removed": float(np.count_nonzero(front_mask) / max(1, np.count_nonzero(valid_removed))),
            "hand_behind_fraction_of_valid_removed": float(np.count_nonzero(behind_mask) / max(1, np.count_nonzero(valid_removed))),
            "ambiguous_fraction_of_valid_removed": float(np.count_nonzero(ambiguous_mask) / max(1, np.count_nonzero(valid_removed))),
            "no_hit_fraction_of_valid_removed": float(np.count_nonzero(no_hit_mask) / max(1, np.count_nonzero(valid_removed))),
            "behind_fraction_of_raw_mask": float(np.count_nonzero(behind_mask) / max(1, np.count_nonzero(raw_mask))),
            "hand_minus_object_depth_on_removed_m": summarize(delta[has_hand_hit], "_m"),
        },
        "counterfactual_masks": {
            "preserve_behind_hands_pixels_recovered": int(np.count_nonzero(counterfactual_front_only) - np.count_nonzero(owned_mask)),
            "preserve_behind_hands_mask_pixels": int(np.count_nonzero(counterfactual_front_only)),
            "preserve_behind_hands_retained_raw_fraction": float(np.count_nonzero(counterfactual_front_only) / max(1, np.count_nonzero(raw_mask))),
            "preserve_behind_and_ambiguous_pixels_recovered": int(np.count_nonzero(counterfactual_front_or_ambiguous) - np.count_nonzero(owned_mask)),
            "preserve_behind_and_ambiguous_mask_pixels": int(np.count_nonzero(counterfactual_front_or_ambiguous)),
            "preserve_behind_and_ambiguous_retained_raw_fraction": float(np.count_nonzero(counterfactual_front_or_ambiguous) / max(1, np.count_nonzero(raw_mask))),
        },
        "per_hand": per_hand,
        "_visual": {
            "raw_mask": raw_mask,
            "owned_mask": owned_mask,
            "removed": removed,
            "front": front_mask,
            "behind": behind_mask,
            "ambiguous": ambiguous_mask,
            "no_hit": no_hit_mask,
        },
    }
    return row


def make_review(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    tiles: list[np.ndarray] = []
    for row in rows:
        frame_idx = int(row["frame_idx"])
        rgb = cv2.imread(str(require_file(Path(row["paths"]["rgb"]), "review RGB")), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"failed to read RGB: {row['paths']['rgb']}")
        visual = row["_visual"]
        raw = visual["raw_mask"]
        owned = visual["owned_mask"]
        removed = visual["removed"]
        front = visual["front"]
        behind = visual["behind"]
        ambiguous = visual["ambiguous"]
        no_hit = visual["no_hit"]

        p1 = rgb.copy()
        overlay_mask(p1, raw, (255, 80, 30), 0.18)
        overlay_mask(p1, owned, (65, 205, 75), 0.30)
        draw_contours(p1, raw, (255, 80, 30), 2)
        draw_contours(p1, owned, (65, 205, 75), 2)
        put_label(
            p1,
            f"f{frame_idx:03d} raw vs current owned",
            f"raw {np.count_nonzero(raw)} px | owned {np.count_nonzero(owned)} px | removed {np.count_nonzero(removed)} px",
        )

        p2 = rgb.copy()
        overlay_mask(p2, removed, (40, 40, 235), 0.52)
        draw_contours(p2, raw, (255, 80, 30), 1)
        draw_contours(p2, owned, (65, 205, 75), 2)
        put_label(
            p2,
            f"f{frame_idx:03d} pixels erased by MANO silhouettes",
            "red = raw object support removed from owned mask",
        )

        p3 = rgb.copy()
        overlay_mask(p3, front, (40, 40, 235), 0.52)
        overlay_mask(p3, ambiguous, (35, 220, 255), 0.52)
        overlay_mask(p3, behind, (65, 205, 75), 0.55)
        overlay_mask(p3, no_hit, (150, 150, 150), 0.35)
        draw_contours(p3, raw, (255, 80, 30), 1)
        draw_contours(p3, owned, (245, 245, 245), 1)
        d = row["depth_order"]
        put_label(
            p3,
            f"f{frame_idx:03d} depth order on erased pixels",
            f"green behind {d['hand_behind_pixels']} | red front {d['hand_in_front_pixels']} | yellow near {d['ambiguous_near_contact_pixels']} | gray no-hit {d['no_hand_first_hit_padding_or_raster_gap_pixels']}",
        )
        tiles.append(np.hstack([crop_to_content(p, [raw, removed]) for p in (p1, p2, p3)]))
    sheet = np.vstack(tiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write review: {output}")
    return {"path": str(output), "bytes": int(output.stat().st_size), "sha256": sha256_file(output)}


def write_markdown(report: dict[str, Any], output: Path) -> None:
    a = report["aggregate"]
    lines = [
        "# Hand ownership subtraction 前后深度关系诊断",
        "",
        "## 结论",
        "",
        report["verdict"]["zh"],
        "",
        "## 总量",
        "",
        f"- 诊断帧数：{a['frame_count']}；所有权复现完全一致的帧数：{a['ownership_exact_frames']}/{a['frame_count']}。",
        f"- raw object mask 中被 MANO silhouette 删除的像素总数：{a['removed_pixels_total']}，占 raw mask 的比例中位数：{a['removed_fraction_of_raw_median']:.4f}。",
        f"- 在有有效 object depth 的被删除像素中，手在物体后方：`{a['hand_behind_pixels_total']}`，比例 `{a['hand_behind_fraction_of_valid_removed']:.4f}`。",
        f"- 手在物体前方：`{a['hand_in_front_pixels_total']}`，比例 `{a['hand_in_front_fraction_of_valid_removed']:.4f}`。",
        f"- 近接触/不确定：`{a['ambiguous_pixels_total']}`，比例 `{a['ambiguous_fraction_of_valid_removed']:.4f}`。",
        f"- padding-only 被删像素：`{a['padding_only_removed_pixels_total']}`，比例 `{a['padding_only_removed_fraction_of_valid_removed']:.4f}`；无 MANO first-hit：`{a['no_hit_pixels_total']}`，比例 `{a['no_hit_fraction_of_valid_removed']:.4f}`，其中 padding-only no-hit `{a['padding_only_no_hand_hit_pixels_total']}` px，unpadded raster gap `{a['unpadded_silhouette_raster_gap_pixels_total']}` px。",
        f"- 独立 MANO vertex probe（不经过 CPU triangle rasterizer）：depth 比 MANO vertex 更近的手行中位比例 `{a['mano_vertex_depth_probe']['fraction_depth_closer_than_hand_median']:.4f}`；`depth_z - hand_z` 的行中位数为 `{a['mano_vertex_depth_probe']['depth_minus_hand_vertex_z_median_mm_across_hand_rows']:.2f} mm`，负值表示观测深度更近。",
        "",
        "## 反事实 mask",
        "",
        f"- 若只删除 hand-in-front，平均可保留的 raw mask 比例：{a['preserve_behind_retained_raw_fraction_median']:.4f}；相比当前 owned mask 可回收 `{a['preserve_behind_recovered_pixels_total']}` px。",
        f"- 若同时保留 behind 与 ambiguous，平均可保留 raw mask 比例：{a['preserve_behind_and_ambiguous_retained_raw_fraction_median']:.4f}；可回收 `{a['preserve_behind_and_ambiguous_recovered_pixels_total']}` px。",
        "",
        "## 判别",
        "",
        "`hand_z - object_z > 5 mm` 被视为手在物体后方。对这类像素执行 silhouette subtraction 会把本应属于物体的观测支持删掉。",
        "`hand_z < -5 mm` 才支持前方手扣除；`|Δ| <= 5 mm` 保留为接触/遮挡不确定。",
        "",
        "## 产物",
        "",
        f"- JSON：`{report['outputs']['report']}`",
        f"- 可视复核：`{report['outputs']['review_image']}`",
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
        raise RuntimeError("annotations have no frames")
    if str(args.frames).strip().lower() == "all":
        selected_frames = [frame for frame in frames if isinstance(frame, dict) and ((frame.get("objects") or [{}])[0].get("visible_geometry_candidate"))]
    else:
        wanted = {int(value) for value in str(args.frames).split(",") if value.strip()}
        selected_frames = [frame for frame in frames if int(frame.get("frame_idx", -1)) in wanted]
        found = {int(frame["frame_idx"]) for frame in selected_frames}
        missing = sorted(wanted - found)
        if missing:
            raise RuntimeError(f"requested frames absent: {missing}")
    if not selected_frames:
        raise RuntimeError("no frames selected")
    first_visible = selected_frames[0]["objects"][0]["visible_geometry_candidate"]
    depth_path = require_file(Path(str(first_visible["depth_npz"])), "camera-bound depth archive")
    depth_archive = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    bridge_cache: dict[Path, Any] = {}
    hawor_cache: dict[Path, Any] = {}
    try:
        depth_row_by_idx = {int(value): i for i, value in enumerate(np.asarray(depth_archive["frame_idx"], dtype=np.int64).tolist())}
        rows = [
            process_frame(frame, depth_archive, depth_row_by_idx, bridge_cache, hawor_cache, args)
            for frame in selected_frames
        ]
    finally:
        depth_archive.close()
        for cache in (bridge_cache, hawor_cache):
            for archive in cache.values():
                archive.close()

    review_wanted = {int(value) for value in str(args.review_frames).split(",") if value.strip()}
    review_rows = [row for row in rows if int(row["frame_idx"]) in review_wanted]
    review = make_review(review_rows, output_dir / "hand_depth_order_ownership_review.jpg")
    for row in rows:
        row.pop("_visual", None)
        for hand in row.get("per_hand", []):
            hand.pop("_silhouette", None)
            hand.pop("_z_hit", None)

    valid_removed_total = sum(int(row["depth_order"]["valid_removed_depth_pixels"]) for row in rows)
    raw_total = sum(int(row["mask_counts"]["raw_mask_pixels"]) for row in rows)
    behind_total = sum(int(row["depth_order"]["hand_behind_pixels"]) for row in rows)
    front_total = sum(int(row["depth_order"]["hand_in_front_pixels"]) for row in rows)
    ambiguous_total = sum(int(row["depth_order"]["ambiguous_near_contact_pixels"]) for row in rows)
    no_hit_total = sum(int(row["depth_order"]["no_hand_first_hit_padding_or_raster_gap_pixels"]) for row in rows)
    padding_only_removed_total = sum(int(row["depth_order"]["padding_only_removed_pixels"]) for row in rows)
    padding_only_no_hit_total = sum(int(row["depth_order"]["padding_only_no_hand_hit_pixels"]) for row in rows)
    unpadded_raster_gap_total = sum(int(row["depth_order"]["unpadded_silhouette_raster_gap_pixels"]) for row in rows)
    vertex_probe_deltas_mm = [
        float(hand["mano_vertex_depth_probe"]["depth_minus_hand_vertex_z_m"].get("median_m")) * 1000.0
        for row in rows
        for hand in row.get("per_hand", [])
        if hand.get("mano_vertex_depth_probe", {}).get("depth_minus_hand_vertex_z_m", {}).get("count")
    ]
    vertex_probe_closer_fractions = [
        float(hand["mano_vertex_depth_probe"]["fraction_depth_closer_than_hand_by_tolerance"])
        for row in rows
        for hand in row.get("per_hand", [])
        if hand.get("mano_vertex_depth_probe", {}).get("fraction_depth_closer_than_hand_by_tolerance") is not None
    ]
    aggregate = {
        "frame_count": len(rows),
        "ownership_exact_frames": int(sum(bool(row["mask_counts"]["ownership_reproduced_exactly"]) for row in rows)),
        "raw_mask_pixels_total": int(raw_total),
        "removed_pixels_total": int(sum(int(row["mask_counts"]["removed_pixels"]) for row in rows)),
        "removed_fraction_of_raw_median": float(np.median([row["mask_counts"]["removed_fraction_of_raw"] for row in rows])),
        "valid_removed_pixels_total": int(valid_removed_total),
        "hand_behind_pixels_total": int(behind_total),
        "hand_in_front_pixels_total": int(front_total),
        "ambiguous_pixels_total": int(ambiguous_total),
        "no_hit_pixels_total": int(no_hit_total),
        "padding_only_removed_pixels_total": int(padding_only_removed_total),
        "padding_only_no_hand_hit_pixels_total": int(padding_only_no_hit_total),
        "unpadded_silhouette_raster_gap_pixels_total": int(unpadded_raster_gap_total),
        "hand_behind_fraction_of_valid_removed": float(behind_total / max(1, valid_removed_total)),
        "hand_in_front_fraction_of_valid_removed": float(front_total / max(1, valid_removed_total)),
        "ambiguous_fraction_of_valid_removed": float(ambiguous_total / max(1, valid_removed_total)),
        "no_hit_fraction_of_valid_removed": float(no_hit_total / max(1, valid_removed_total)),
        "padding_only_removed_fraction_of_valid_removed": float(padding_only_removed_total / max(1, valid_removed_total)),
        "padding_only_no_hand_hit_fraction_of_valid_removed": float(padding_only_no_hit_total / max(1, valid_removed_total)),
        "unpadded_silhouette_raster_gap_fraction_of_valid_removed": float(unpadded_raster_gap_total / max(1, valid_removed_total)),
        "behind_fraction_by_frame": summarize(np.asarray([row["depth_order"]["behind_fraction_of_raw_mask"] for row in rows])),
        "removed_fraction_by_frame": summarize(np.asarray([row["mask_counts"]["removed_fraction_of_raw"] for row in rows])),
        "preserve_behind_recovered_pixels_total": int(sum(int(row["counterfactual_masks"]["preserve_behind_hands_pixels_recovered"]) for row in rows)),
        "preserve_behind_retained_raw_fraction_median": float(np.median([row["counterfactual_masks"]["preserve_behind_hands_retained_raw_fraction"] for row in rows])),
        "preserve_behind_and_ambiguous_recovered_pixels_total": int(sum(int(row["counterfactual_masks"]["preserve_behind_and_ambiguous_pixels_recovered"]) for row in rows)),
        "preserve_behind_and_ambiguous_retained_raw_fraction_median": float(np.median([row["counterfactual_masks"]["preserve_behind_and_ambiguous_retained_raw_fraction"] for row in rows])),
        "worst_behind_fraction_frames": [
            int(row["frame_idx"])
            for row in sorted(rows, key=lambda row: float(row["depth_order"]["hand_behind_fraction_of_valid_removed"]), reverse=True)[:15]
        ],
        "mano_vertex_depth_probe": {
            "depth_minus_hand_vertex_z_median_mm_across_hand_rows": float(np.median(vertex_probe_deltas_mm)) if vertex_probe_deltas_mm else None,
            "depth_minus_hand_vertex_z_p95_mm_across_hand_rows": float(np.percentile(vertex_probe_deltas_mm, 95.0)) if vertex_probe_deltas_mm else None,
            "fraction_depth_closer_than_hand_median": float(np.median(vertex_probe_closer_fractions)) if vertex_probe_closer_fractions else None,
            "hand_row_count": len(vertex_probe_deltas_mm),
        },
    }
    report_path = output_dir / "hand_depth_order_ownership_audit.json"
    md_path = output_dir / "HAND_DEPTH_ORDER_OWNERSHIP_AUDIT_ZH.md"
    behind_supported = behind_total > 0 and float(behind_total / max(1, valid_removed_total)) > 0.01
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "reconstruct_mano_silhouette_subtraction_and_classify_removed_pixels_by_first_hit_depth_order",
        "claim_scope": (
            "Diagnostic only. MANO is a prediction-side estimate; depth order uses camera-bound UniDepth and a CPU first-hit "
            "rasterizer. It identifies an ownership-subtraction mechanism defect but does not rewrite masks or promote physical contact claims."
        ),
        "compute_contract": {
            "local_gpu_used": False,
            "model_inference_run": False,
            "prediction_inputs_mutated": False,
            "annotations": file_ref(annotations_path),
            "depth_archive": file_ref(depth_path),
        },
        "depth_order_tolerance_m": float(args.depth_order_tolerance_m),
        "aggregate": aggregate,
        "rows": rows,
        "review_image": review,
        "outputs": {"report": str(report_path), "markdown": str(md_path), "review_image": review["path"]},
        "verdict": {
            "behind_object_hand_subtraction_supported": bool(behind_supported),
            "zh": (
                "当前实现确实没有 hand/object 前后关系判断；诊断也在被删除像素中发现了手位于物体后方的支持。"
                if behind_supported else
                "当前实现没有 hand/object 前后关系判断，但在该案例中没有发现足够规模的明确后方手误删。"
            ),
        },
        "elapsed_s": time.time() - started,
    }
    if behind_supported:
        report["verdict"]["zh"] += (
            f" 在全部有效被删像素中，约 {100.0 * aggregate['hand_behind_fraction_of_valid_removed']:.2f}% 的 MANO first-hit 位于 object depth 后方；"
            "这些像素不应被当作前方遮挡手从物体 mask 中删除。"
        )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "markdown": str(md_path),
        "review_image": review["path"],
        "aggregate": aggregate,
        "verdict": report["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
