#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from run_unidepth_metric_source_v3 import infer_unidepth, load_model, localize_path, read_manifest, resize_depth, summarize
from v19_camera_contract import K_from_intrinsics, load_contract, plane_intrinsics, sha256_file


def camera_ray_angular_error_deg(rays: np.ndarray, camera_K: np.ndarray) -> dict:
    rays = np.asarray(rays, dtype=np.float64)
    if rays.ndim != 3 or rays.shape[0] != 3 or not np.isfinite(rays).all():
        raise RuntimeError(f"invalid UniDepth output rays: {rays.shape}")
    _, height, width = rays.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    inverse_K = np.linalg.inv(np.asarray(camera_K, dtype=np.float64))
    homogeneous = np.stack((xs, ys, np.ones_like(xs)), axis=0).reshape(3, -1)
    expected = (inverse_K @ homogeneous).reshape(3, height, width)
    expected /= np.linalg.norm(expected, axis=0, keepdims=True).clip(min=1.0e-12)
    actual = rays / np.linalg.norm(rays, axis=0, keepdims=True).clip(min=1.0e-12)
    dots = np.clip(np.sum(actual * expected, axis=0), -1.0, 1.0)
    angle = np.degrees(np.arccos(dots))
    return {
        "mean": float(np.mean(angle)),
        "median": float(np.median(angle)),
        "p95": float(np.percentile(angle, 95.0)),
        "max": float(np.max(angle)),
    }


def metric_radius_to_camera_z(radius: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float64)
    if radius.ndim != 2 or not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise RuntimeError(f"invalid UniDepth metric radius raster: {radius.shape}")
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64).reshape(4)
    height, width = radius.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    ray_norm = np.sqrt(((xs - cx) / fx) ** 2 + ((ys - cy) / fy) ** 2 + 1.0)
    return (radius / ray_norm).astype(np.float32)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    if args.unidepth_repo is not None:
        sys.path.insert(0, str(args.unidepth_repo))
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("UniDepth full-frame export requires CUDA unless --cpu is explicit")
    device = torch.device("cpu" if args.cpu else "cuda")
    model = load_model(args.model_id, device)
    rows_in = read_manifest(args.manifest, int(args.frame_start), int(args.frame_end))
    if not rows_in:
        raise RuntimeError("manifest contains no selected frames")

    camera_conditioning = None
    input_camera_K = None
    output_camera_intrinsics = None
    if args.camera_contract is not None:
        contract_path = args.camera_contract.expanduser().resolve()
        if not contract_path.is_file():
            raise RuntimeError(f"missing camera contract: {contract_path}")
        contract_payload, contract_normalized = load_contract(contract_path)
        selected_frame_ids = [int(row["frame_idx"]) for row in rows_in]
        contract_frame_ids = set(int(value) for value in contract_normalized["frame_ids"])
        missing_frame_ids = sorted(set(selected_frame_ids) - contract_frame_ids)
        if missing_frame_ids:
            raise RuntimeError(f"camera contract misses selected frame ids: {missing_frame_ids[:10]}")
        first_rgb_path = localize_path(str(rows_in[0]["rgb"]), args.remote_root, args.local_root)
        with Image.open(first_rgb_path) as first_image:
            input_size_wh = tuple(int(value) for value in first_image.size)
        input_intrinsics, input_plane = plane_intrinsics(
            contract_payload,
            contract_normalized,
            plane_name=args.camera_input_plane,
            actual_size_wh=input_size_wh,
            allow_implicit_resize=False,
        )
        output_intrinsics, output_plane = plane_intrinsics(
            contract_payload,
            contract_normalized,
            plane_name=args.camera_output_plane,
            actual_size_wh=(int(args.source_width), int(args.source_height)),
            allow_implicit_resize=False,
        )
        input_camera_K = K_from_intrinsics(input_intrinsics)
        output_camera_intrinsics = np.asarray(output_intrinsics, dtype=np.float64)
        camera_conditioning = {
            "mode": "provided_pinhole_intrinsics",
            "camera_contract_path": str(contract_path),
            "camera_contract_sha256": sha256_file(contract_path),
            "camera_input_plane": str(args.camera_input_plane),
            "camera_output_plane": str(args.camera_output_plane),
            "input_size_wh": list(input_size_wh),
            "output_size_wh": [int(args.source_width), int(args.source_height)],
            "input_intrinsics_fx_fy_cx_cy": np.asarray(input_intrinsics, dtype=np.float64).tolist(),
            "output_intrinsics_fx_fy_cx_cy": output_camera_intrinsics.tolist(),
            "input_plane_resolution": input_plane,
            "output_plane_resolution": output_plane,
            "model_camera_head_intrinsics_max_abs_error_px": 0.0,
            "active_depth_rays_mean_angular_error_deg": 0.0,
            "active_depth_rays_median_angular_error_deg": 0.0,
            "active_depth_rays_p95_angular_error_deg": 0.0,
            "active_depth_rays_max_angular_error_deg": 0.0,
            "output_depth_conversion": "resize_metric_radius_then_project_to_exact_output_contract_rays_as_camera_z",
            "depth_ray_geometry_reprojected": True,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    depth_png_dir = args.output_dir / "depth"
    depth_png_dir.mkdir(exist_ok=True)

    depth_stack = []
    confidence_stack = []
    frame_indices = []
    focal_px = []
    intrinsics_stack = []
    model_camera_head_intrinsics_stack = []
    rows = []
    for out_i, entry in enumerate(rows_in):
        frame_idx = int(entry["frame_idx"])
        rgb_path = localize_path(str(entry["rgb"]), args.remote_root, args.local_root)
        image = Image.open(rgb_path).convert("RGB")
        if camera_conditioning is not None and list(image.size) != camera_conditioning["input_size_wh"]:
            raise RuntimeError(
                f"frame {frame_idx} RGB size {list(image.size)} differs from camera input plane {camera_conditioning['input_size_wh']}"
            )
        depth_raw, intrinsics, output_rays, radius_raw, confidence_raw = infer_unidepth(model, image, device, input_camera_K)
        if intrinsics is None:
            raise RuntimeError(f"UniDepth returned no intrinsics for frame {frame_idx}")
        if input_camera_K is not None:
            returned = np.asarray([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]], dtype=np.float64)
            expected = np.asarray(camera_conditioning["input_intrinsics_fx_fy_cx_cy"], dtype=np.float64)
            delta_px = float(np.max(np.abs(returned - expected)))
            camera_conditioning["model_camera_head_intrinsics_max_abs_error_px"] = max(
                float(camera_conditioning["model_camera_head_intrinsics_max_abs_error_px"]), delta_px
            )
            if output_rays is None:
                raise RuntimeError(f"UniDepth frame {frame_idx} returned no rays for camera-contract validation")
            ray_error = camera_ray_angular_error_deg(output_rays, input_camera_K)
            camera_conditioning["active_depth_rays_max_angular_error_deg"] = max(
                float(camera_conditioning["active_depth_rays_max_angular_error_deg"]),
                float(ray_error["max"]),
            )
            camera_conditioning["active_depth_rays_mean_angular_error_deg"] = max(
                float(camera_conditioning["active_depth_rays_mean_angular_error_deg"]),
                float(ray_error["mean"]),
            )
            camera_conditioning["active_depth_rays_median_angular_error_deg"] = max(
                float(camera_conditioning["active_depth_rays_median_angular_error_deg"]),
                float(ray_error["median"]),
            )
            camera_conditioning["active_depth_rays_p95_angular_error_deg"] = max(
                float(camera_conditioning["active_depth_rays_p95_angular_error_deg"]),
                float(ray_error["p95"]),
            )
            if float(ray_error["max"]) > float(args.camera_rays_validation_max_angle_deg):
                raise RuntimeError(
                    f"UniDepth frame {frame_idx} output rays disagree with supplied K by {ray_error['max']:.6f} degrees"
                )
            model_camera_head_intrinsics_stack.append(returned.tolist())
            if radius_raw is None:
                raise RuntimeError(f"UniDepth frame {frame_idx} returned no metric radius for exact sensor-ray reprojection")
            radius_output = resize_depth(radius_raw, (int(args.source_height), int(args.source_width)))
            depth = metric_radius_to_camera_z(radius_output, output_camera_intrinsics)
            fx, fy, cx, cy = output_camera_intrinsics.tolist()
        else:
            depth = resize_depth(depth_raw, (int(args.source_height), int(args.source_width)))
            fx = float(intrinsics[0, 0]) * (int(args.source_width) / float(depth_raw.shape[1]))
            fy = float(intrinsics[1, 1]) * (int(args.source_height) / float(depth_raw.shape[0]))
            cx = float(intrinsics[0, 2]) * (int(args.source_width) / float(depth_raw.shape[1]))
            cy = float(intrinsics[1, 2]) * (int(args.source_height) / float(depth_raw.shape[0]))
            model_camera_head_intrinsics_stack.append([fx, fy, cx, cy])
        focal = float(np.sqrt(max(1e-9, fx * fy)))
        if confidence_raw is None:
            raise RuntimeError(f"UniDepth frame {frame_idx} returned no confidence raster")
        confidence = resize_depth(confidence_raw, (int(args.source_height), int(args.source_width)))
        if not np.isfinite(confidence).all() or np.any(confidence <= 0.0):
            raise RuntimeError(f"frame {frame_idx} has invalid UniDepth confidence")
        valid = np.isfinite(depth) & (depth > 0.0)
        if int(np.count_nonzero(valid)) < int(args.min_valid_pixels):
            raise RuntimeError(f"frame {frame_idx} has too few valid UniDepth pixels")
        depth_png_path = depth_png_dir / f"{out_i:06d}.png"
        depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
        if not cv2.imwrite(str(depth_png_path), depth_mm):
            raise RuntimeError(f"failed to write {depth_png_path}")
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"failed to read RGB {rgb_path}")
        norm = depth.copy()
        lo, hi = np.percentile(depth[valid], [5.0, 95.0])
        norm = np.clip((norm - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        color = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        # The manifest RGB is intentionally rendered at --render-width (960 px),
        # while the metric depth archive is resized to the source-coordinate
        # contract (1408 px here).  Keep the archive/intrinsics in source
        # coordinates and resize only the visualization overlay to the decoded
        # RGB frame; cv2.addWeighted requires identical H/W/C arrays.
        if color.shape[:2] != rgb.shape[:2]:
            color_review = cv2.resize(
                color,
                (int(rgb.shape[1]), int(rgb.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            color_review = color
        if color_review.ndim != rgb.ndim or color_review.shape[2] != rgb.shape[2]:
            raise RuntimeError(
                f"review channel mismatch for frame {frame_idx}: "
                f"rgb={rgb.shape} color={color_review.shape}"
            )
        review = cv2.addWeighted(rgb, 0.55, color_review, 0.45, 0.0)
        cv2.imwrite(str(still_dir / f"frame_{frame_idx:06d}.png"), review)

        frame_indices.append(frame_idx)
        focal_px.append(focal)
        intrinsics_stack.append([fx, fy, cx, cy])
        depth_stack.append(depth.astype(np.float16))
        confidence_stack.append(confidence.astype(np.float16))
        values = depth[valid].astype(np.float64)
        rows.append(
            {
                "frame_idx": frame_idx,
                "rgb": str(rgb_path),
                "depth_png": str(depth_png_path),
                "unidepth_focal_px": focal,
                "unidepth_fx_px": fx,
                "unidepth_fy_px": fy,
                "unidepth_cx_px": cx,
                "unidepth_cy_px": cy,
                "depth_median_m": float(np.median(values)),
                "depth_p05_m": float(np.percentile(values, 5.0)),
                "depth_p95_m": float(np.percentile(values, 95.0)),
            }
        )

    depth_archive = args.output_dir / "unidepth_full_frame_depth_v3.npz"
    archive_payload = {
        "frame_idx": np.asarray(frame_indices, dtype=np.int32),
        "depth": np.stack(depth_stack, axis=0),
        "confidence": np.stack(confidence_stack, axis=0),
        "source_size": np.asarray([int(args.source_width), int(args.source_height)], dtype=np.int32),
        "focal_px": np.asarray(focal_px, dtype=np.float64),
        "intrinsics_fx_fy_cx_cy": np.asarray(intrinsics_stack, dtype=np.float64),
        "model_camera_head_intrinsics_fx_fy_cx_cy": np.asarray(model_camera_head_intrinsics_stack, dtype=np.float64),
        "camera_conditioning_mode": np.asarray(
            "provided_pinhole_intrinsics" if camera_conditioning is not None else "model_inferred_intrinsics"
        ),
        "depth_ray_geometry_reprojected": np.asarray(camera_conditioning is not None),
        "depth_output_quantity": np.asarray(
            "camera_z_m_from_metric_radius_on_exact_contract_rays"
            if camera_conditioning is not None else "camera_z_m_from_model_inferred_rays"
        ),
    }
    if camera_conditioning is not None:
        archive_payload.update({
            "inference_camera_contract_path": np.asarray(camera_conditioning["camera_contract_path"]),
            "inference_camera_contract_sha256": np.asarray(camera_conditioning["camera_contract_sha256"]),
            "inference_camera_input_plane": np.asarray(camera_conditioning["camera_input_plane"]),
            "inference_camera_output_plane": np.asarray(camera_conditioning["camera_output_plane"]),
            "inference_camera_intrinsics_fx_fy_cx_cy": np.asarray(intrinsics_stack, dtype=np.float64),
        })
    np.savez_compressed(depth_archive, **archive_payload)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "run_unidepth_full_frame_v3",
        "model_id": str(args.model_id),
        "manifest": str(args.manifest),
        "frames": int(len(rows)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "depth_archive": str(depth_archive),
        "stills_dir": str(still_dir),
        "review_resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
        "depth_archive_source_size": [int(args.source_width), int(args.source_height)],
        "camera_conditioning": camera_conditioning,
        "camera_ray_contract": (
            "UniDepth depth head consumed the supplied camera rays. The model's metric radius is resized and explicitly projected onto exact output-plane contract rays to produce camera-z meters; model camera-head K remains diagnostic only."
            if camera_conditioning is not None else
            "UniDepth inferred rays/K. Do not relabel this byte-identical z raster with another K; rerun using --camera-contract."
        ),
        "unidepth_focal_px": summarize([row["unidepth_focal_px"] for row in rows]),
        "depth_median_m": summarize([row["depth_median_m"] for row in rows]),
        "rows": rows,
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "qc_unidepth_full_frame_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path)
    parser.add_argument("--camera-input-plane", default="manifest_rgb")
    parser.add_argument("--camera-output-plane", default="source_rgb")
    parser.add_argument("--camera-rays-validation-max-angle-deg", type=float, default=1.0)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--unidepth-repo", type=Path)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--min-valid-pixels", type=int, default=100000)
    parser.add_argument("--model-id", default="lpiccinelli/unidepth-v2-vitl14")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
