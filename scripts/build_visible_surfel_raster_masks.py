#!/usr/bin/env python3
"""Rasterize prediction-side visible surfel samples into mask images for tracking.

This builds a CoTracker manifest whose masks are derived from
``visible_geometry_candidate.camera_vertices_sample_m`` / ``world_vertices_sample_m`` in
prediction annotations. The masks are not segmentation ground truth and not object
geometry; they are a query/acceptance support raster for the same visible metric surfel
samples consumed by sparse-edge diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return data


def frame_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("manifest missing frames list")
    return {int(row["frame_idx"]): row for row in frames if isinstance(row, dict) and "frame_idx" in row}


def parse_object_selector(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def select_object(frame: dict[str, Any], selector: str | int | None) -> dict[str, Any] | None:
    objects = frame.get("objects")
    if not isinstance(objects, list) or not objects:
        return None
    if selector is None:
        return objects[0] if isinstance(objects[0], dict) else None
    if isinstance(selector, int):
        return objects[selector] if 0 <= selector < len(objects) and isinstance(objects[selector], dict) else None
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        candidates = [obj.get("object_id"), obj.get("track_id"), obj.get("id"), obj.get("label")]
        if any(str(candidate) == selector for candidate in candidates if candidate is not None):
            return obj
    return None


def camera_points(frame: dict[str, Any], obj: dict[str, Any]) -> tuple[np.ndarray, list[float]] | None:
    geom = obj.get("visible_geometry_candidate")
    if not isinstance(geom, dict):
        return None
    intr = geom.get("intrinsics_fx_fy_cx_cy") or frame.get("camera", {}).get("intrinsics_fx_fy_cx_cy")
    if not isinstance(intr, list) or len(intr) != 4:
        return None
    cam_raw = geom.get("camera_vertices_sample_m")
    if isinstance(cam_raw, list) and cam_raw:
        cam = np.asarray(cam_raw, dtype=np.float64)
    else:
        world_raw = geom.get("world_vertices_sample_m")
        T = frame.get("camera", {}).get("T_world_camera_metric")
        if not isinstance(world_raw, list) or not world_raw or not isinstance(T, list):
            return None
        world = np.asarray(world_raw, dtype=np.float64)
        Tcw = np.linalg.inv(np.asarray(T, dtype=np.float64))
        hom = np.c_[world, np.ones(len(world), dtype=np.float64)]
        cam = (hom @ Tcw.T)[:, :3]
    if cam.ndim != 2 or cam.shape[1] != 3:
        return None
    finite = np.isfinite(cam).all(axis=1) & (cam[:, 2] > 0)
    cam = cam[finite]
    if len(cam) == 0:
        return None
    return cam, [float(v) for v in intr]


def rasterize(
    cam: np.ndarray,
    intr: list[float],
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    radius_px: int,
) -> tuple[np.ndarray, int, int]:
    fx, fy, cx, cy = intr
    xs_source = fx * cam[:, 0] / cam[:, 2] + cx
    ys_source = fy * cam[:, 1] / cam[:, 2] + cy
    finite = np.isfinite(xs_source) & np.isfinite(ys_source)
    scale_x = float(output_width) / max(1.0, float(source_width))
    scale_y = float(output_height) / max(1.0, float(source_height))
    xs = np.rint(xs_source[finite] * scale_x).astype(np.int64)
    ys = np.rint(ys_source[finite] * scale_y).astype(np.int64)
    keep = (0 <= xs) & (xs < output_width) & (0 <= ys) & (ys < output_height)
    xs = xs[keep]
    ys = ys[keep]
    mask = np.zeros((output_height, output_width), dtype=np.uint8)
    output_radius = max(0, int(round(float(radius_px) * min(scale_x, scale_y))))
    for x, y in zip(xs.tolist(), ys.tolist()):
        if output_radius == 0:
            mask[y, x] = 255
        else:
            cv2.circle(mask, (x, y), output_radius, 255, thickness=-1, lineType=cv2.LINE_8)
    return mask, int(len(xs)), int(output_radius)


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    base_manifest = load_json(args.base_manifest)
    base_by_frame = frame_map(base_manifest)
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("annotations missing frames list")
    selector = parse_object_selector(args.object_id)
    masks_dir = args.output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_frames: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict) or "frame_idx" not in frame:
            continue
        frame_idx = int(frame["frame_idx"])
        base = base_by_frame.get(frame_idx)
        if base is None:
            continue
        obj = select_object(frame, selector)
        if obj is None:
            continue
        cp = camera_points(frame, obj)
        if cp is None:
            continue
        source_width = int(frame.get("source_width") or args.width)
        source_height = int(frame.get("source_height") or args.height)
        base_mask_path = base.get("mask")
        base_mask = cv2.imread(str(base_mask_path), cv2.IMREAD_GRAYSCALE) if base_mask_path else None
        if base_mask is not None:
            output_height, output_width = base_mask.shape[:2]
        else:
            output_width = int(frame.get("manifest_width") or args.width)
            output_height = int(frame.get("manifest_height") or args.height)
        cam, intr = cp
        mask, projected, output_radius = rasterize(cam, intr, source_width, source_height, output_width, output_height, int(args.radius_px))
        mask_path = masks_dir / f"{frame_idx:06d}.png"
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"failed to write {mask_path}")
        area = int(np.count_nonzero(mask))
        out_frames.append({"frame_idx": frame_idx, "rgb": base.get("rgb"), "mask": str(mask_path), "mask_area_px": area})
        rows.append({"frame_idx": frame_idx, "projected_surfel_points": projected, "mask_area_px": area, "output_radius_px": output_radius, "source_size_wh": [source_width, source_height], "output_size_wh": [output_width, output_height]})
    if not out_frames:
        raise RuntimeError("no visible surfel masks written")
    manifest = {
        "schema": "cotracker_visible_surfel_raster_manifest_v1",
        "source": "prediction_annotations_visible_geometry_candidate",
        "claim_scope": "query/acceptance raster for prediction-side visible metric surfel samples; not segmentation GT and not object geometry",
        "annotations": str(args.annotations),
        "base_manifest": str(args.base_manifest),
        "object_id": args.object_id,
        "source_radius_px": int(args.radius_px),
        "radius_basis": "radius is specified in annotation source pixels and scaled to the manifest mask resolution; source radius 4 matches the default 4 px surfel sampling stride used by build_v19_visible_geometry_from_sam2_depth",
        "frames": out_frames,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    areas = np.asarray([row["mask_area_px"] for row in rows], dtype=np.float64)
    report = {
        "status": "ok",
        "method": "build_visible_surfel_raster_masks",
        "diagnostic_only": True,
        "claim_scope": manifest["claim_scope"],
        "frame_count": int(len(rows)),
        "first_frame": int(out_frames[0]["frame_idx"]),
        "last_frame": int(out_frames[-1]["frame_idx"]),
        "source_radius_px": int(args.radius_px),
        "output_radius_px": int(rows[0]["output_radius_px"]),
        "mask_area_px": {
            "median": float(np.median(areas)),
            "min": int(np.min(areas)),
            "max": int(np.max(areas)),
        },
        "rows": rows,
        "output_manifest": str(args.output_manifest),
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--radius-px", type=int, default=4)
    parser.add_argument("--width", type=int, default=1408)
    parser.add_argument("--height", type=int, default=1408)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
