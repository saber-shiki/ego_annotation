#!/usr/bin/env python3
"""Evaluate depth-order counterfactual visible geometry against the frozen baseline.

This compares the same depth/camera/anchor pipeline under two mask ownership
contracts:

* baseline: raw SAM2 mask minus every padded MANO silhouette;
* counterfactual: remove only pixels where MANO first hit is clearly in front of
  measured object depth.

The evaluation is CPU-only and prediction-read-only. It reports mask support,
visible-surfel support, anchor Poisson surface projections, and the existing
GHOST-aligned SAM3D mesh projection at the anchor frame. It does not rerun SAM3D
or GHOST optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

BASE = Path(
    "/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/"
    "backend_tests/20260819T122452Z_hot3d_milk_local_authority_local29_v1/runs/"
    "P0014_84ea2dcc_carton_milk_f2370_2519"
)
GHOST = BASE / "experiments/sam3d_native_ghost_lite_20260826"
CF = GHOST / "depth_order_counterfactual_masks_v1"
SCHEMA = "v19_depth_order_counterfactual_visible_geometry_eval_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-annotations",
        type=Path,
        default=BASE / "measurements/object_geometry/visible_geometry/carton_milk/annotations_v19_visible_geometry.json",
    )
    parser.add_argument(
        "--counterfactual-annotations",
        type=Path,
        default=CF / "visible_geometry_front_only/annotations_v19_visible_geometry.json",
    )
    parser.add_argument(
        "--alignment-report",
        type=Path,
        default=GHOST / "p15_first_hit_depth_authority_bounded_v4/qc_sam3d_p15_first_hit_alignment.json",
    )
    parser.add_argument(
        "--counterfactual-alignment-report",
        type=Path,
        default=CF / "ghost_front_only_p15_first_hit_depth_authority_v1/qc_sam3d_p15_first_hit_alignment.json",
    )
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--output-dir", type=Path, default=CF / "counterfactual_visible_geometry_evaluation_v1")
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


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(require_file(path, "mask")), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return image > 0


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    interpolation = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    return cv2.resize(mask.astype(np.uint8), (int(width), int(height)), interpolation=interpolation) > 0


def summarize(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise RuntimeError(f"mask shape mismatch: {prediction.shape} vs {target.shape}")
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    intersection = int(np.count_nonzero(pred & tgt))
    union = int(np.count_nonzero(pred | tgt))
    pred_count = int(np.count_nonzero(pred))
    target_count = int(np.count_nonzero(tgt))
    centroid_distance = None
    if pred_count and target_count:
        centroid_distance = float(np.linalg.norm(np.argwhere(pred).mean(axis=0)[::-1] - np.argwhere(tgt).mean(axis=0)[::-1]))
    return {
        "iou": float(intersection / union) if union else None,
        "precision": float(intersection / pred_count) if pred_count else None,
        "recall": float(intersection / target_count) if target_count else None,
        "centroid_distance_px": centroid_distance,
        "prediction_pixels": pred_count,
        "target_pixels": target_count,
    }


def affine_mask_from_source(visible: dict[str, Any], mask_shape: tuple[int, int]) -> np.ndarray:
    contract = visible.get("mask_depth_transform_contract") if isinstance(visible.get("mask_depth_transform_contract"), dict) else {}
    mask_plane = contract.get("mask_plane_transform") if isinstance(contract.get("mask_plane_transform"), dict) else {}
    affine = mask_plane.get("A_actual_plane_from_calibration")
    if affine is not None:
        return np.asarray(affine, dtype=np.float64)
    source_w = int(visible.get("source_width"))
    source_h = int(visible.get("source_height"))
    ratio_x = mask_shape[1] / source_w
    ratio_y = mask_shape[0] / source_h
    return np.asarray([[ratio_x, 0, 0.5 * (ratio_x - 1)], [0, ratio_y, 0.5 * (ratio_y - 1)], [0, 0, 1]], dtype=np.float64)


def project_camera(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    z = pts[:, 2]
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(pts).all(axis=1) & (z > 1.0e-9)
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64).tolist()
    uv[valid, 0] = fx * pts[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * pts[valid, 1] / z[valid] + cy
    return uv


def apply_affine(uv: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(uv, dtype=np.float64)
    mapped = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(affine, dtype=np.float64).T
    return mapped[:, :2] / mapped[:, 2:3]


def rasterize_mesh_silhouette(vertices_camera: np.ndarray, faces: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    uv = project_camera(vertices_camera, intrinsics)
    z = np.asarray(vertices_camera, dtype=np.float64)[:, 2]
    faces = np.asarray(faces, dtype=np.int64)
    valid = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(z[faces] > 1.0e-9, axis=1)
    polygons = np.rint(uv[faces[valid]]).astype(np.int32)
    out = np.zeros((height, width), dtype=np.uint8)
    if len(polygons):
        cv2.fillPoly(out, list(polygons), 1, cv2.LINE_8)
    return out > 0


def anchor_frame(annotations: dict[str, Any], frame_idx: int) -> dict[str, Any]:
    for frame in annotations.get("frames", []):
        if int(frame.get("frame_idx")) == int(frame_idx):
            return frame
    raise RuntimeError(f"frame {frame_idx} absent")


def visible_from_frame(frame: dict[str, Any]) -> dict[str, Any]:
    visible = (frame.get("objects") or [{}])[0].get("visible_geometry_candidate")
    if not isinstance(visible, dict):
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks visible geometry")
    return visible


def anchor_mesh_reconstruction(annotations_path: Path, annotations: dict[str, Any]) -> dict[str, Any]:
    value = annotations.get("anchor_visible_surface_mesh_reconstruction")
    if isinstance(value, dict):
        return value
    report_path = require_file(annotations_path.parent / "v19_visible_geometry_adapter_report.json", "visible-geometry adapter report")
    report = load_json(report_path)
    value = report.get("anchor_visible_surface_mesh_reconstruction")
    if not isinstance(value, dict):
        raise RuntimeError(f"adapter report lacks anchor_visible_surface_mesh_reconstruction: {report_path}")
    return value


def poisson_silhouette(frame: dict[str, Any], mesh_path: Path) -> np.ndarray:
    visible = visible_from_frame(frame)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"invalid Poisson mesh: {mesh_path}")
    vertices_world = np.asarray(mesh.vertices, dtype=np.float64) + np.asarray(visible["centroid_world_m"], dtype=np.float64)[None, :]
    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    vertices_camera = (vertices_world - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
    return rasterize_mesh_silhouette(
        vertices_camera,
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(visible["intrinsics_fx_fy_cx_cy"], dtype=np.float64),
        int(visible["source_width"]),
        int(visible["source_height"]),
    )


def ghost_silhouette(frame: dict[str, Any], mesh_path: Path) -> np.ndarray:
    visible = visible_from_frame(frame)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"invalid GHOST mesh: {mesh_path}")
    return rasterize_mesh_silhouette(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(visible["intrinsics_fx_fy_cx_cy"], dtype=np.float64),
        int(visible["source_width"]),
        int(visible["source_height"]),
    )


def surfel_inside_fraction(frame: dict[str, Any], mask: np.ndarray) -> float:
    visible = visible_from_frame(frame)
    camera = np.asarray(visible.get("camera_vertices_sample_m"), dtype=np.float64)
    camera = camera[np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1.0e-9)]
    if not len(camera):
        return float("nan")
    uv = project_camera(camera, np.asarray(visible["intrinsics_fx_fy_cx_cy"], dtype=np.float64))
    uv_mask = apply_affine(uv, affine_mask_from_source(visible, mask.shape))
    uv_i = np.rint(uv_mask).astype(np.int64)
    inside = (uv_i[:, 0] >= 0) & (uv_i[:, 0] < mask.shape[1]) & (uv_i[:, 1] >= 0) & (uv_i[:, 1] < mask.shape[0])
    if not np.any(inside):
        return 0.0
    uv_i = uv_i[inside]
    return float(np.count_nonzero(mask[uv_i[:, 1], uv_i[:, 0]]) / len(camera))


def draw_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    color_img = np.zeros_like(image)
    color_img[:] = color
    image[mask] = cv2.addWeighted(image, 1.0 - alpha, color_img, alpha, 0.0)[mask]


def crop(image: np.ndarray, masks: list[np.ndarray], pad: int = 75, size: int = 420) -> np.ndarray:
    combined = np.zeros(image.shape[:2], dtype=bool)
    for mask in masks:
        combined |= mask
    ys, xs = np.where(combined)
    if not len(xs):
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad + 1)
    y1 = min(image.shape[0], int(ys.max()) + pad + 1)
    c = image[y0:y1, x0:x1]
    scale = size / max(c.shape[:2])
    r = cv2.resize(c, (max(1, round(c.shape[1] * scale)), max(1, round(c.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 18, dtype=np.uint8)
    y = (size - r.shape[0]) // 2
    x = (size - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas


def make_anchor_review(
    frame: dict[str, Any],
    raw_mask: np.ndarray,
    baseline_owned: np.ndarray,
    counterfactual_mask: np.ndarray,
    baseline_poisson: np.ndarray,
    counterfactual_poisson: np.ndarray,
    ghost_mesh: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    rgb = cv2.imread(str(require_file(Path(str(frame["raw_frame_path"])), "anchor RGB")), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"failed to read anchor RGB: {frame['raw_frame_path']}")
    panels: list[np.ndarray] = []
    specs = [
        ("baseline owned + old Poisson", baseline_owned, baseline_poisson),
        ("counterfactual + new Poisson", counterfactual_mask, counterfactual_poisson),
        ("raw SAM2 + existing GHOST mesh", raw_mask, ghost_mesh),
    ]
    for title, mask, surface in specs:
        panel = rgb.copy()
        overlay(panel, mask, (65, 205, 75), 0.22)
        overlay(panel, surface, (220, 60, 220), 0.24)
        draw_contours(panel, mask, (65, 205, 75), 2)
        draw_contours(panel, surface, (220, 60, 220), 2)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (12, 12, 12), -1)
        cv2.putText(panel, title[:100], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(crop(panel, [raw_mask, baseline_owned, counterfactual_mask, surface]))
    sheet = np.hstack(panels)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write anchor review: {output}")
    return {"path": str(output), "bytes": int(output.stat().st_size), "sha256": sha256_file(output)}


def main() -> None:
    started = time.time()
    args = parse_args()
    output_dir = prepare_output(args.output_dir, bool(args.replace))
    baseline_annotations_path = require_file(args.baseline_annotations, "baseline annotations")
    counterfactual_annotations_path = require_file(args.counterfactual_annotations, "counterfactual annotations")
    alignment_report_path = require_file(args.alignment_report, "GHOST alignment report")
    baseline_annotations = load_json(baseline_annotations_path)
    counterfactual_annotations = load_json(counterfactual_annotations_path)
    alignment = load_json(alignment_report_path)
    counterfactual_alignment_path = require_file(args.counterfactual_alignment_report, "counterfactual GHOST alignment report")
    counterfactual_alignment = load_json(counterfactual_alignment_path)
    baseline_frame = anchor_frame(baseline_annotations, args.anchor_frame)
    counterfactual_frame = anchor_frame(counterfactual_annotations, args.anchor_frame)
    baseline_visible = visible_from_frame(baseline_frame)
    counterfactual_visible = visible_from_frame(counterfactual_frame)
    raw_mask = read_mask(Path(str(baseline_visible["source_mask_path"])))
    baseline_owned = read_mask(Path(str(baseline_visible["mask_path"])))
    counterfactual_mask = read_mask(Path(str(counterfactual_visible["mask_path"])))
    baseline_mesh_reconstruction = anchor_mesh_reconstruction(baseline_annotations_path, baseline_annotations)
    counterfactual_mesh_reconstruction = anchor_mesh_reconstruction(counterfactual_annotations_path, counterfactual_annotations)
    baseline_poisson_path = require_file(
        Path(str(baseline_mesh_reconstruction["poisson_mesh_path"])),
        "baseline anchor Poisson",
    )
    counterfactual_poisson_path = require_file(
        Path(str(counterfactual_mesh_reconstruction["poisson_mesh_path"])),
        "counterfactual anchor Poisson",
    )
    ghost_mesh_path = require_file(Path(str(alignment["mesh_output_anchor_camera"])), "baseline GHOST aligned mesh")
    counterfactual_ghost_mesh_path = require_file(
        Path(str(counterfactual_alignment["mesh_output_anchor_camera"])),
        "counterfactual GHOST aligned mesh",
    )
    baseline_poisson_source = poisson_silhouette(baseline_frame, baseline_poisson_path)
    counterfactual_poisson_source = poisson_silhouette(counterfactual_frame, counterfactual_poisson_path)
    ghost_source = ghost_silhouette(baseline_frame, ghost_mesh_path)
    counterfactual_ghost_source = ghost_silhouette(baseline_frame, counterfactual_ghost_mesh_path)
    baseline_poisson = resize_mask(baseline_poisson_source, raw_mask.shape[1], raw_mask.shape[0])
    counterfactual_poisson = resize_mask(counterfactual_poisson_source, raw_mask.shape[1], raw_mask.shape[0])
    ghost_mesh = resize_mask(ghost_source, raw_mask.shape[1], raw_mask.shape[0])
    counterfactual_ghost_mesh = resize_mask(counterfactual_ghost_source, raw_mask.shape[1], raw_mask.shape[0])

    projection_metrics = {
        "baseline_poisson_vs_baseline_owned": mask_metrics(baseline_poisson, baseline_owned),
        "baseline_poisson_vs_raw_sam2": mask_metrics(baseline_poisson, raw_mask),
        "baseline_poisson_vs_counterfactual": mask_metrics(baseline_poisson, counterfactual_mask),
        "counterfactual_poisson_vs_counterfactual": mask_metrics(counterfactual_poisson, counterfactual_mask),
        "counterfactual_poisson_vs_raw_sam2": mask_metrics(counterfactual_poisson, raw_mask),
        "counterfactual_poisson_vs_baseline_owned": mask_metrics(counterfactual_poisson, baseline_owned),
        "ghost_existing_mesh_vs_baseline_owned": mask_metrics(ghost_mesh, baseline_owned),
        "ghost_existing_mesh_vs_raw_sam2": mask_metrics(ghost_mesh, raw_mask),
        "ghost_existing_mesh_vs_counterfactual": mask_metrics(ghost_mesh, counterfactual_mask),
        "ghost_counterfactual_mesh_vs_baseline_owned": mask_metrics(counterfactual_ghost_mesh, baseline_owned),
        "ghost_counterfactual_mesh_vs_raw_sam2": mask_metrics(counterfactual_ghost_mesh, raw_mask),
        "ghost_counterfactual_mesh_vs_counterfactual": mask_metrics(counterfactual_ghost_mesh, counterfactual_mask),
    }
    baseline_centroid = np.asarray(baseline_visible["centroid_world_m"], dtype=np.float64)
    counterfactual_centroid = np.asarray(counterfactual_visible["centroid_world_m"], dtype=np.float64)
    baseline_extent = np.asarray(baseline_visible["world_extent_m"], dtype=np.float64)
    counterfactual_extent = np.asarray(counterfactual_visible["world_extent_m"], dtype=np.float64)
    row_metrics = []
    baseline_frames = {int(frame["frame_idx"]): frame for frame in baseline_annotations.get("frames", [])}
    counterfactual_frames = {int(frame["frame_idx"]): frame for frame in counterfactual_annotations.get("frames", [])}
    for frame_idx in sorted(set(baseline_frames) & set(counterfactual_frames)):
        bvis = (baseline_frames[frame_idx].get("objects") or [{}])[0].get("visible_geometry_candidate")
        cvis = (counterfactual_frames[frame_idx].get("objects") or [{}])[0].get("visible_geometry_candidate")
        if not isinstance(bvis, dict) or not isinstance(cvis, dict):
            continue
        bmask = read_mask(Path(str(bvis["mask_path"])))
        cmask = read_mask(Path(str(cvis["mask_path"])))
        row_metrics.append(
            {
                "frame_idx": frame_idx,
                "baseline_mask_pixels": int(np.count_nonzero(bmask)),
                "counterfactual_mask_pixels": int(np.count_nonzero(cmask)),
                "counterfactual_over_baseline_area": float(np.count_nonzero(cmask) / max(1, np.count_nonzero(bmask))),
                "baseline_surfels_inside_counterfactual_fraction": surfel_inside_fraction(baseline_frames[frame_idx], cmask),
                "counterfactual_surfels_inside_baseline_fraction": surfel_inside_fraction(counterfactual_frames[frame_idx], bmask),
                "counterfactual_surfels_inside_counterfactual_fraction": surfel_inside_fraction(counterfactual_frames[frame_idx], cmask),
                "baseline_depth_retained_fraction": float((bvis.get("first_surface_depth_ownership") or {}).get("retained_fraction") or np.nan),
                "counterfactual_depth_retained_fraction": float((cvis.get("first_surface_depth_ownership") or {}).get("retained_fraction") or np.nan),
                "baseline_rigid_eligible": bool(bvis.get("rigid_pose_observation_eligible")),
                "counterfactual_rigid_eligible": bool(cvis.get("rigid_pose_observation_eligible")),
            }
        )
    anchor_review = make_anchor_review(
        baseline_frame,
        raw_mask,
        baseline_owned,
        counterfactual_mask,
        baseline_poisson,
        counterfactual_poisson,
        ghost_mesh,
        output_dir / "anchor_counterfactual_surface_alignment_review.jpg",
    )
    report_path = output_dir / "counterfactual_visible_geometry_evaluation.json"
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "compare_baseline_padded_mano_subtraction_vs_depth_order_front_only_counterfactual",
        "claim_scope": (
            "CPU-only counterfactual evaluation. The existing GHOST mesh is not re-optimized; its anchor projection is only "
            "measured against alternative mask contracts. Generated SAM3D faces remain render-only."
        ),
        "compute_contract": {"local_gpu_used": False, "model_inference_run": False, "prediction_inputs_mutated": False},
        "inputs": {
            "baseline_annotations": file_ref(baseline_annotations_path),
            "counterfactual_annotations": file_ref(counterfactual_annotations_path),
            "alignment_report": file_ref(alignment_report_path),
            "counterfactual_alignment_report": file_ref(counterfactual_alignment_path),
            "baseline_poisson": file_ref(baseline_poisson_path),
            "counterfactual_poisson": file_ref(counterfactual_poisson_path),
            "ghost_mesh": file_ref(ghost_mesh_path),
            "counterfactual_ghost_mesh": file_ref(counterfactual_ghost_mesh_path),
        },
        "anchor_frame": int(args.anchor_frame),
        "mask_counts_anchor": {
            "raw_sam2_pixels": int(np.count_nonzero(raw_mask)),
            "baseline_owned_pixels": int(np.count_nonzero(baseline_owned)),
            "counterfactual_pixels": int(np.count_nonzero(counterfactual_mask)),
            "baseline_retained_raw_fraction": float(np.count_nonzero(baseline_owned) / max(1, np.count_nonzero(raw_mask))),
            "counterfactual_retained_raw_fraction": float(np.count_nonzero(counterfactual_mask) / max(1, np.count_nonzero(raw_mask))),
        },
        "anchor_projection_metrics": projection_metrics,
        "anchor_geometry_shift": {
            "baseline_centroid_world_m": baseline_centroid.astype(float).tolist(),
            "counterfactual_centroid_world_m": counterfactual_centroid.astype(float).tolist(),
            "centroid_shift_m": float(np.linalg.norm(counterfactual_centroid - baseline_centroid)),
            "baseline_world_extent_m": baseline_extent.astype(float).tolist(),
            "counterfactual_world_extent_m": counterfactual_extent.astype(float).tolist(),
        },
        "timeline_summary": {
            "frame_count": len(row_metrics),
            "counterfactual_over_baseline_area": summarize([row["counterfactual_over_baseline_area"] for row in row_metrics]),
            "baseline_surfels_inside_counterfactual_fraction": summarize([row["baseline_surfels_inside_counterfactual_fraction"] for row in row_metrics]),
            "counterfactual_surfels_inside_baseline_fraction": summarize([row["counterfactual_surfels_inside_baseline_fraction"] for row in row_metrics]),
            "counterfactual_surfels_inside_counterfactual_fraction": summarize([row["counterfactual_surfels_inside_counterfactual_fraction"] for row in row_metrics]),
            "baseline_depth_retained_fraction": summarize([row["baseline_depth_retained_fraction"] for row in row_metrics]),
            "counterfactual_depth_retained_fraction": summarize([row["counterfactual_depth_retained_fraction"] for row in row_metrics]),
            "baseline_rigid_eligible_frames": int(sum(row["baseline_rigid_eligible"] for row in row_metrics)),
            "counterfactual_rigid_eligible_frames": int(sum(row["counterfactual_rigid_eligible"] for row in row_metrics)),
        },
        "rows": row_metrics,
        "anchor_review": anchor_review,
        "interpretation": {
            "counterfactual_restores_baseline_poisson_alignment": bool(
                projection_metrics["baseline_poisson_vs_counterfactual"]["iou"]
                > projection_metrics["baseline_poisson_vs_baseline_owned"]["iou"] + 0.02
            ),
            "counterfactual_surface_keeps_alignment": bool(
                projection_metrics["counterfactual_poisson_vs_counterfactual"]["iou"] >= 0.80
            ),
            "ghost_mesh_not_reoptimized": True,
        },
        "elapsed_s": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "report": str(report_path),
        "anchor_review": anchor_review["path"],
        "anchor_projection_metrics": projection_metrics,
        "timeline_summary": report["timeline_summary"],
        "interpretation": report["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
