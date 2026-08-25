#!/usr/bin/env python3
"""Run DA3 Nested on fixed-K, fixed-trajectory overlapping video windows.

This is the controlled HOT3D depth intervention.  Official prediction-side
pinhole intrinsics and the prediction-side HaWoR metric camera trajectory are
provided to Depth Anything 3.  DA3-predicted camera state is never promoted to
camera authority.  The resulting camera-z metric depth is ray-remapped from
DA3's processed raster onto the exact V19 output camera plane.

SAM3D is not imported or modified by this component.  In particular, this file
must not be used to inject a DA3 point map into SAM3D Objects.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from v19_camera_contract import (
    K_from_intrinsics,
    load_contract,
    plane_intrinsics,
    require_ordered_frame_subset,
    sha256_file,
)


DEPTH_PROVIDER = "depth_anything_3"
POSE_CONDITIONING_MODE = "fixed_prediction_side_hawor_metric_w2c"
CONFIDENCE_SEMANTICS = "monotonic_inverse_DA3_confidence_error_proxy_higher_is_worse_not_metric_error"
METRIC_SCALE_MODES = {
    "nested_metric_branch": False,
    "input_trajectory_umeyama": True,
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summarize(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def localize_path(path: str | Path, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        for source, destination in ((remote_root, local_root), (local_root, remote_root)):
            try:
                relative = direct.relative_to(source)
            except ValueError:
                continue
            candidate = destination / relative
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def read_manifest(path: Path, frame_start: int, frame_end: int) -> list[dict[str, Any]]:
    rows = load_json(path).get("frames")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    selected = [
        dict(row)
        for row in rows
        if isinstance(row, dict) and frame_start <= int(row["frame_idx"]) <= frame_end
    ]
    selected.sort(key=lambda row: int(row["frame_idx"]))
    if not selected:
        raise RuntimeError(f"no manifest frames in {frame_start}:{frame_end}")
    frame_ids = [int(row["frame_idx"]) for row in selected]
    if len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError("manifest selection contains duplicate frame_idx values")
    return selected


def load_hawor_w2c(path: Path, expected_frame_ids: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"frame_idx", "R_c2w", "t_c2w"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"HaWoR camera archive misses {missing}: {path}")
        frame_idx = np.asarray(archive["frame_idx"], dtype=np.int32)
        rotations = np.asarray(archive["R_c2w"], dtype=np.float64)
        translations = np.asarray(archive["t_c2w"], dtype=np.float64)
    if rotations.shape != (len(frame_idx), 3, 3) or translations.shape != (len(frame_idx), 3):
        raise RuntimeError(
            f"invalid HaWoR camera arrays: frames={frame_idx.shape} R={rotations.shape} t={translations.shape}"
        )
    if len(set(frame_idx.tolist())) != len(frame_idx):
        raise RuntimeError("HaWoR camera archive contains duplicate frame_idx values")
    by_frame = {int(frame): position for position, frame in enumerate(frame_idx.tolist())}
    missing_frames = [frame for frame in expected_frame_ids if frame not in by_frame]
    if missing_frames:
        raise RuntimeError(f"HaWoR camera archive misses selected frames: {missing_frames[:10]}")

    c2w_rows: list[np.ndarray] = []
    orthogonality_errors: list[float] = []
    determinants: list[float] = []
    for frame in expected_frame_ids:
        position = by_frame[frame]
        rotation = rotations[position]
        translation = translations[position]
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise RuntimeError(f"HaWoR camera row {frame} is non-finite")
        error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
        if error > 2.0e-3 or abs(determinant - 1.0) > 2.0e-3:
            raise RuntimeError(
                f"HaWoR R_c2w is not a proper rotation at frame {frame}: orth={error} det={determinant}"
            )
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = translation
        c2w_rows.append(c2w)
        orthogonality_errors.append(error)
        determinants.append(determinant)
    c2w_stack = np.stack(c2w_rows, axis=0)
    w2c_stack = np.linalg.inv(c2w_stack)
    centers = c2w_stack[:, :3, 3]
    adjacent_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.empty(0)
    return w2c_stack, {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_arrays": ["R_c2w", "t_c2w"],
        "output_convention": "opencv_world_to_camera_4x4",
        "frame_count": len(expected_frame_ids),
        "rotation_orthogonality_max_abs_error": max(orthogonality_errors, default=0.0),
        "rotation_determinant": summarize(determinants),
        "camera_center_adjacent_translation_m": summarize(adjacent_steps.tolist()),
        "camera_center_extent_m": (centers.max(axis=0) - centers.min(axis=0)).tolist(),
        "claim_scope": (
            "prediction-side HaWoR metric SLAM trajectory used only as fixed DA3 pose conditioning; "
            "no HOT3D released camera pose or evaluator state is consumed"
        ),
    }


def make_windows(frame_count: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if window_size <= 1:
        raise ValueError("window_size must exceed one for pose-conditioned DA3")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")
    if frame_count <= window_size:
        return [(0, frame_count)]
    maximum_stride = window_size - overlap
    final_start = frame_count - window_size
    interval_count = int(math.ceil(final_start / maximum_stride))
    # Distribute the starts across the complete span instead of appending one
    # near-duplicate tail window, which can create accidental triple overlap.
    starts = np.rint(np.linspace(0, final_start, interval_count + 1)).astype(int).tolist()
    windows = [(start, start + window_size) for start in sorted(set(starts))]
    coverage = np.zeros(frame_count, dtype=np.int32)
    for start, end in windows:
        coverage[start:end] += 1
    if np.any(coverage == 0):
        raise RuntimeError(f"window plan leaves uncovered frames: {np.where(coverage == 0)[0].tolist()}")
    return windows


def window_taper(length: int, minimum: float = 0.25) -> np.ndarray:
    if length <= 0:
        raise ValueError("window length must be positive")
    if length == 1:
        return np.ones(1, dtype=np.float32)
    phase = np.linspace(0.0, math.pi, length, dtype=np.float64)
    values = minimum + (1.0 - minimum) * np.sin(phase)
    return values.astype(np.float32)


def ray_remap_coordinates(
    source_intrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
    target_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_intrinsics, dtype=np.float64).reshape(4)
    target = np.asarray(target_intrinsics, dtype=np.float64).reshape(4)
    if not np.isfinite(source).all() or not np.isfinite(target).all() or np.any(source[:2] <= 0) or np.any(target[:2] <= 0):
        raise RuntimeError("ray remap requires finite positive pinhole intrinsics")
    height, width = [int(value) for value in target_shape_hw]
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    normalized_x = (xs - target[2]) / target[0]
    normalized_y = (ys - target[3]) / target[1]
    map_x = source[0] * normalized_x + source[2]
    map_y = source[1] * normalized_y + source[3]
    return map_x.astype(np.float32), map_y.astype(np.float32)


def remap_camera_z_to_exact_rays(
    values: np.ndarray,
    source_intrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
    target_shape_hw: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_values = np.asarray(values, dtype=np.float32)
    if source_values.ndim != 2:
        raise RuntimeError(f"ray-remap source must be HxW, got {source_values.shape}")
    map_x, map_y = ray_remap_coordinates(source_intrinsics, target_intrinsics, target_shape_hw)
    tolerance_px = 1.0e-4
    valid_map = (
        (map_x >= -tolerance_px)
        & (map_x <= float(source_values.shape[1] - 1) + tolerance_px)
        & (map_y >= -tolerance_px)
        & (map_y <= float(source_values.shape[0] - 1) + tolerance_px)
    )
    remapped = cv2.remap(
        source_values,
        np.clip(map_x, 0.0, float(source_values.shape[1] - 1)),
        np.clip(map_y, 0.0, float(source_values.shape[0] - 1)),
        interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )
    remapped[~valid_map] = np.nan
    valid = valid_map & np.isfinite(remapped)
    source = np.asarray(source_intrinsics, dtype=np.float64).reshape(4)
    target = np.asarray(target_intrinsics, dtype=np.float64).reshape(4)
    source_x = (map_x.astype(np.float64) - source[2]) / source[0]
    source_y = (map_y.astype(np.float64) - source[3]) / source[1]
    height, width = target_shape_hw
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    target_x = (xs.astype(np.float64) - target[2]) / target[0]
    target_y = (ys.astype(np.float64) - target[3]) / target[1]
    source_rays = np.stack((source_x, source_y, np.ones_like(source_x)), axis=-1)
    target_rays = np.stack((target_x, target_y, np.ones_like(target_x)), axis=-1)
    source_rays /= np.linalg.norm(source_rays, axis=-1, keepdims=True)
    target_rays /= np.linalg.norm(target_rays, axis=-1, keepdims=True)
    sample_stride_y = max(1, height // 64)
    sample_stride_x = max(1, width // 64)
    sample = valid[::sample_stride_y, ::sample_stride_x]
    dots = np.sum(source_rays * target_rays, axis=-1)
    sampled_dots = dots[::sample_stride_y, ::sample_stride_x]
    angles = np.degrees(np.arccos(np.clip(sampled_dots[sample], -1.0, 1.0))) if np.any(sample) else np.asarray([])
    return remapped.astype(np.float32), {
        "method": "inverse_pinhole_ray_map_then_dense_resample_camera_z",
        "source_shape_hw": list(source_values.shape),
        "target_shape_hw": [height, width],
        "valid_fraction": float(np.mean(valid)),
        "sampled_ray_angle_error_deg": summarize(angles.tolist()),
    }


def matrix_intrinsics(K: np.ndarray) -> np.ndarray:
    matrix = np.asarray(K, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid DA3 K shape {matrix.shape}")
    values = np.asarray([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]], dtype=np.float64)
    if np.any(values[:2] <= 0.0):
        raise RuntimeError(f"invalid DA3 focal values: {values}")
    return values


def homogeneous_extrinsics(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-2:] == (4, 4):
        return array
    if array.shape[-2:] == (3, 4):
        output = np.repeat(np.eye(4, dtype=np.float64)[None], len(array), axis=0)
        output[:, :3, :] = array
        return output
    raise RuntimeError(f"invalid extrinsics shape {array.shape}")


def save_depth_review(path: Path, rgb_path: Path, depth: np.ndarray) -> None:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"cannot read RGB review image {rgb_path}")
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("cannot render empty DA3 depth")
    lo, hi = np.percentile(depth[valid], [5.0, 95.0])
    normalized = np.clip((depth - lo) / max(1.0e-6, hi - lo), 0.0, 1.0)
    color = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    if color.shape[:2] != rgb.shape[:2]:
        color = cv2.resize(color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    review = cv2.addWeighted(rgb, 0.55, color, 0.45, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), review):
        raise RuntimeError(f"cannot write DA3 depth review {path}")


def load_model(args: argparse.Namespace) -> Any:
    repo = args.da3_repo.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"missing DA3 repository: {repo}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"missing local DA3 model directory: {model_path}")
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from depth_anything_3.api import DepthAnything3  # noqa: PLC0415

    model = DepthAnything3.from_pretrained(str(model_path))
    return model.to(device=args.device).eval()


def evaluate_overlap_consistency(
    overlap_rows: list[dict[str, Any]],
    window_count: int,
    *,
    max_median_relative_difference: float,
    max_frame_relative_difference: float,
    max_scale_aligned_median_relative_difference: float,
) -> dict[str, Any]:
    relative_medians = [
        float(row["relative_depth_difference"].get("median", float("nan"))) for row in overlap_rows
    ]
    scale_aligned_relative_medians = [
        float(row["scale_aligned_relative_depth_difference"].get("median", float("nan")))
        for row in overlap_rows
    ]
    required = int(window_count) > 1
    passed = bool(
        not required
        or (
            overlap_rows
            and np.isfinite(relative_medians).all()
            and np.isfinite(scale_aligned_relative_medians).all()
            and float(np.median(relative_medians)) <= float(max_median_relative_difference)
            and float(np.max(relative_medians)) <= float(max_frame_relative_difference)
            and float(np.median(scale_aligned_relative_medians))
            <= float(max_scale_aligned_median_relative_difference)
        )
    )
    return {
        "required": required,
        "passed": passed,
        "thresholds": {
            "median_relative_difference": float(max_median_relative_difference),
            "max_frame_relative_difference": float(max_frame_relative_difference),
            "scale_aligned_median_relative_difference": float(
                max_scale_aligned_median_relative_difference
            ),
        },
        "observed": {
            "median_relative_difference": (
                float(np.median(relative_medians)) if relative_medians else None
            ),
            "max_frame_relative_difference": (
                float(np.max(relative_medians)) if relative_medians else None
            ),
            "scale_aligned_median_relative_difference": (
                float(np.median(scale_aligned_relative_medians))
                if scale_aligned_relative_medians
                else None
            ),
        },
    }


def close_memmap(array: Any) -> None:
    if array is None:
        return
    try:
        array.flush()
    except Exception:
        pass
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        try:
            mmap.close()
        except Exception:
            pass


def remove_temporary_tree(path: Path, attempts: int = 5) -> None:
    """Remove local or CIFS scratch without masking the inference exception."""
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.25 * (attempt + 1))
    if last_error is not None:
        raise last_error


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    rows = read_manifest(args.manifest, int(args.frame_start), int(args.frame_end))
    frame_ids = [int(row["frame_idx"]) for row in rows]
    rgb_paths = [localize_path(row["rgb"], args.remote_root, args.local_root) for row in rows]
    with Image.open(rgb_paths[0]) as first_image:
        input_size_wh = tuple(int(value) for value in first_image.size)
    for frame, path in zip(frame_ids, rgb_paths, strict=True):
        with Image.open(path) as image:
            if tuple(image.size) != input_size_wh:
                raise RuntimeError(f"frame {frame} RGB size {image.size} differs from {input_size_wh}")

    contract_path = args.camera_contract.expanduser().resolve()
    # Validate the complete immutable contract first, then bind this invocation's
    # frame range as an ordered subset. Requiring equality here would make the
    # advertised --frame-start/--frame-end interface unusable for smoke tests or
    # resumable chunks while adding no protection for a fixed-K contract.
    contract, normalized_contract = load_contract(contract_path)
    require_ordered_frame_subset(normalized_contract["frame_ids"], frame_ids)
    input_intrinsics, input_plane = plane_intrinsics(
        contract,
        normalized_contract,
        plane_name=args.camera_input_plane,
        actual_size_wh=input_size_wh,
        allow_implicit_resize=False,
    )
    output_size_wh = (int(args.source_width), int(args.source_height))
    output_intrinsics, output_plane = plane_intrinsics(
        contract,
        normalized_contract,
        plane_name=args.camera_output_plane,
        actual_size_wh=output_size_wh,
        allow_implicit_resize=False,
    )
    output_K = K_from_intrinsics(output_intrinsics)
    w2c, camera_report = load_hawor_w2c(args.hawor_npz.expanduser().resolve(), frame_ids)
    input_K_rows = np.repeat(K_from_intrinsics(input_intrinsics)[None], len(rows), axis=0)
    windows = make_windows(len(rows), int(args.window_size), int(args.window_overlap))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / args.output_name
    success_report_path = args.output_dir / "qc_da3_pose_conditioned_full_frame_v1.json"
    failure_report_path = args.output_dir / "qc_da3_pose_conditioned_overlap_failure_v1.json"
    stale_outputs = [path for path in (archive_path, success_report_path, failure_report_path) if path.exists()]
    if stale_outputs and not args.replace:
        raise RuntimeError(f"DA3 outputs exist; use --replace explicitly: {[str(path) for path in stale_outputs]}")
    for path in stale_outputs:
        path.unlink()
    temp_dir = args.output_dir / ".da3_accumulators"
    if temp_dir.exists():
        if not args.replace:
            raise RuntimeError(f"temporary accumulator directory exists: {temp_dir}")
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    shape = (len(rows), int(args.source_height), int(args.source_width))
    depth_numerator = np.lib.format.open_memmap(temp_dir / "depth_numerator.npy", mode="w+", dtype=np.float32, shape=shape)
    weight_sum = np.lib.format.open_memmap(temp_dir / "weight_sum.npy", mode="w+", dtype=np.float32, shape=shape)
    confidence_numerator = np.lib.format.open_memmap(temp_dir / "confidence_numerator.npy", mode="w+", dtype=np.float32, shape=shape)
    taper_sum = np.lib.format.open_memmap(temp_dir / "taper_sum.npy", mode="w+", dtype=np.float32, shape=shape)
    contribution_count = np.zeros(len(rows), dtype=np.int32)

    if args.metric_scale_mode not in METRIC_SCALE_MODES:
        raise RuntimeError(f"unsupported DA3 metric scale mode {args.metric_scale_mode!r}")
    align_to_input_ext_scale = METRIC_SCALE_MODES[args.metric_scale_mode]
    model = None
    depth_final = None
    confidence_error_final = None
    model = load_model(args)
    overlap_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    processed_intrinsics_by_frame: list[list[np.ndarray]] = [[] for _ in rows]
    minimum_valid_fraction = 1.0
    maximum_ray_error_deg = 0.0
    try:
        for window_index, (start, end) in enumerate(windows):
            images = [Image.open(path).convert("RGB") for path in rgb_paths[start:end]]
            prediction = model.inference(
                image=images,
                extrinsics=w2c[start:end].copy(),
                intrinsics=input_K_rows[start:end].copy(),
                align_to_input_ext_scale=align_to_input_ext_scale,
                use_ray_pose=False,
                ref_view_strategy=args.ref_view_strategy,
                process_res=int(args.process_res),
                process_res_method=args.process_res_method,
            )
            for image in images:
                image.close()
            depth_native = np.asarray(prediction.depth, dtype=np.float32)
            confidence_native = np.asarray(prediction.conf, dtype=np.float32) if prediction.conf is not None else None
            intrinsics_native = np.asarray(prediction.intrinsics, dtype=np.float64) if prediction.intrinsics is not None else None
            returned_extrinsics = homogeneous_extrinsics(prediction.extrinsics) if prediction.extrinsics is not None else None
            count = end - start
            if depth_native.ndim != 3 or depth_native.shape[0] != count:
                raise RuntimeError(f"DA3 window {window_index} returned depth shape {depth_native.shape}")
            if confidence_native is None or confidence_native.shape != depth_native.shape:
                raise RuntimeError(f"DA3 window {window_index} lacks aligned confidence: {None if confidence_native is None else confidence_native.shape}")
            if intrinsics_native is None or intrinsics_native.shape != (count, 3, 3):
                raise RuntimeError(f"DA3 window {window_index} returned K shape {None if intrinsics_native is None else intrinsics_native.shape}")
            if returned_extrinsics is None or returned_extrinsics.shape != (count, 4, 4):
                raise RuntimeError(f"DA3 window {window_index} returned extrinsics shape {None if returned_extrinsics is None else returned_extrinsics.shape}")
            if int(getattr(prediction, "is_metric", 0)) != 1:
                raise RuntimeError(f"DA3 window {window_index} output is not marked metric")
            if not np.isfinite(depth_native).all() or np.any(depth_native <= 0.0):
                raise RuntimeError(f"DA3 window {window_index} returned invalid metric depth")
            if not np.isfinite(confidence_native).all() or np.any(confidence_native <= 0.0):
                raise RuntimeError(f"DA3 window {window_index} returned invalid confidence")
            extrinsics_error = float(np.max(np.abs(returned_extrinsics - w2c[start:end])))
            if align_to_input_ext_scale and extrinsics_error > float(args.returned_camera_tolerance):
                raise RuntimeError(
                    f"DA3 window {window_index} did not preserve supplied W2C camera authority: max_error={extrinsics_error}"
                )
            taper = window_taper(count)
            window_c2w = np.linalg.inv(w2c[start:end])
            window_centers = window_c2w[:, :3, 3]
            window_extent = window_centers.max(axis=0) - window_centers.min(axis=0)
            window_baseline_m = float(
                np.max(np.linalg.norm(window_centers[:, None, :] - window_centers[None, :, :], axis=-1))
            )
            window_rows.append({
                "window_index": window_index,
                "start_position": start,
                "end_position_exclusive": end,
                "frame_start": frame_ids[start],
                "frame_end": frame_ids[end - 1],
                "frame_count": count,
                "returned_camera_max_abs_error_diagnostic": extrinsics_error,
                "returned_camera_authority": False,
                "prediction_scale_factor_diagnostic": (
                    float(prediction.scale_factor) if prediction.scale_factor is not None else None
                ),
                "metric_scale_mode": args.metric_scale_mode,
                "processed_depth_shape_hw": list(depth_native.shape[1:]),
                "conditioned_camera_center_extent_m": window_extent.tolist(),
                "conditioned_camera_center_max_baseline_m": window_baseline_m,
            })
            for local_position in range(count):
                global_position = start + local_position
                processed_intrinsics = matrix_intrinsics(intrinsics_native[local_position])
                processed_intrinsics_by_frame[global_position].append(processed_intrinsics)
                depth_output, ray_report = remap_camera_z_to_exact_rays(
                    depth_native[local_position],
                    processed_intrinsics,
                    output_intrinsics,
                    (int(args.source_height), int(args.source_width)),
                )
                raw_confidence_output, confidence_ray_report = remap_camera_z_to_exact_rays(
                    confidence_native[local_position],
                    processed_intrinsics,
                    output_intrinsics,
                    (int(args.source_height), int(args.source_width)),
                )
                valid = (
                    np.isfinite(depth_output)
                    & (depth_output > 0.0)
                    & np.isfinite(raw_confidence_output)
                    & (raw_confidence_output > 0.0)
                )
                valid_fraction = float(np.mean(valid))
                minimum_valid_fraction = min(minimum_valid_fraction, valid_fraction)
                ray_max = float(ray_report["sampled_ray_angle_error_deg"].get("max", 0.0))
                maximum_ray_error_deg = max(maximum_ray_error_deg, ray_max)
                if valid_fraction < float(args.min_reprojected_valid_fraction):
                    raise RuntimeError(
                        f"DA3 frame {frame_ids[global_position]} exact-ray remap valid fraction {valid_fraction:.6f} "
                        f"is below {args.min_reprojected_valid_fraction}"
                    )
                if ray_max > float(args.ray_validation_max_angle_deg):
                    raise RuntimeError(
                        f"DA3 frame {frame_ids[global_position]} exact-ray remap error {ray_max:.9f} degrees"
                    )
                if contribution_count[global_position] > 0:
                    existing = np.divide(
                        depth_numerator[global_position],
                        weight_sum[global_position],
                        out=np.full(depth_output.shape, np.nan, dtype=np.float32),
                        where=weight_sum[global_position] > 0.0,
                    )
                    comparison = valid & np.isfinite(existing)
                    sampled = comparison[:: int(args.overlap_diagnostic_stride), :: int(args.overlap_diagnostic_stride)]
                    existing_sampled = existing[:: int(args.overlap_diagnostic_stride), :: int(args.overlap_diagnostic_stride)][sampled]
                    current_sampled = depth_output[:: int(args.overlap_diagnostic_stride), :: int(args.overlap_diagnostic_stride)][sampled]
                    differences = np.abs(existing_sampled - current_sampled)
                    relative = differences / np.maximum(np.abs(existing_sampled), 1.0e-6)
                    # A global scalar cannot explain context-dependent geometry,
                    # but reporting its best median fit separates scale drift
                    # from residual shape/depth disagreement.
                    scale_ratio = float(np.median(existing_sampled / np.maximum(current_sampled, 1.0e-6)))
                    scale_aligned_differences = np.abs(existing_sampled - current_sampled * scale_ratio)
                    scale_aligned_relative = scale_aligned_differences / np.maximum(np.abs(existing_sampled), 1.0e-6)
                    overlap_rows.append({
                        "frame_idx": frame_ids[global_position],
                        "window_index": window_index,
                        "previous_contribution_count": int(contribution_count[global_position]),
                        "sample_count": int(differences.size),
                        "absolute_depth_difference_m": summarize(differences.tolist()),
                        "relative_depth_difference": summarize(relative.tolist()),
                        "median_scale_to_previous": scale_ratio,
                        "scale_aligned_absolute_depth_difference_m": summarize(scale_aligned_differences.tolist()),
                        "scale_aligned_relative_depth_difference": summarize(scale_aligned_relative.tolist()),
                    })
                blend_weight = raw_confidence_output * float(taper[local_position])
                blend_weight[~valid] = 0.0
                depth_numerator[global_position] += np.nan_to_num(depth_output, nan=0.0) * blend_weight
                confidence_numerator[global_position] += np.nan_to_num(raw_confidence_output, nan=0.0) * float(taper[local_position])
                taper_sum[global_position] += valid.astype(np.float32) * float(taper[local_position])
                weight_sum[global_position] += blend_weight
                contribution_count[global_position] += 1
                _ = confidence_ray_report
            depth_numerator.flush()
            confidence_numerator.flush()
            taper_sum.flush()
            weight_sum.flush()

        if np.any(contribution_count == 0):
            raise RuntimeError(f"DA3 windows produced no contribution for frames {np.where(contribution_count == 0)[0].tolist()}")
        overlap_relative_medians = [
            float(row["relative_depth_difference"].get("median", float("nan"))) for row in overlap_rows
        ]
        overlap_scale_aligned_relative_medians = [
            float(row["scale_aligned_relative_depth_difference"].get("median", float("nan")))
            for row in overlap_rows
        ]
        overlap_consistency = evaluate_overlap_consistency(
            overlap_rows,
            len(windows),
            max_median_relative_difference=float(args.max_overlap_median_relative_difference),
            max_frame_relative_difference=float(args.max_overlap_frame_relative_difference),
            max_scale_aligned_median_relative_difference=float(
                args.max_overlap_scale_aligned_median_relative_difference
            ),
        )
        if not overlap_consistency["passed"]:
            write_json(
                failure_report_path,
                {
                    "status": "failed_overlap_consistency",
                    "annotation_ready": False,
                    "frame_range": [frame_ids[0], frame_ids[-1]],
                    "window_size": int(args.window_size),
                    "window_overlap": int(args.window_overlap),
                    "overlap_consistency": overlap_consistency,
                    "overlap_rows": overlap_rows,
                    "window_rows": window_rows,
                    "depth_archive_written": False,
                },
            )
            raise RuntimeError(
                "DA3 overlapping windows violate metric-depth consistency; "
                f"diagnostics={failure_report_path} observed={overlap_consistency['observed']}"
            )
        depth_final = np.lib.format.open_memmap(temp_dir / "depth_final.npy", mode="w+", dtype=np.float16, shape=shape)
        confidence_error_final = np.lib.format.open_memmap(temp_dir / "confidence_error_final.npy", mode="w+", dtype=np.float16, shape=shape)
        depth_medians: list[float] = []
        valid_fractions: list[float] = []
        review_positions = set(np.linspace(0, len(rows) - 1, min(int(args.review_frame_count), len(rows)), dtype=int).tolist())
        still_dir = args.output_dir / "stills"
        for position in range(len(rows)):
            weights = np.asarray(weight_sum[position], dtype=np.float32)
            valid = weights > 0.0
            depth_row = np.divide(
                depth_numerator[position],
                weights,
                out=np.full(weights.shape, np.nan, dtype=np.float32),
                where=valid,
            )
            taper_weights = np.asarray(taper_sum[position], dtype=np.float32)
            # Recover a stable higher-is-better DA3 confidence by averaging each
            # overlapping window's raw confidence with the position taper only.
            # Depth itself uses raw-confidence*taper weights above.
            raw_confidence = np.divide(
                confidence_numerator[position],
                taper_weights,
                out=np.zeros(taper_weights.shape, dtype=np.float32),
                where=taper_weights > 0.0,
            )
            error_proxy = np.divide(
                1.0,
                np.maximum(raw_confidence, float(args.confidence_epsilon)),
                out=np.full(raw_confidence.shape, np.inf, dtype=np.float32),
                where=valid,
            )
            depth_final[position] = depth_row.astype(np.float16)
            confidence_error_final[position] = error_proxy.astype(np.float16)
            finite = valid & np.isfinite(depth_row) & (depth_row > 0.0)
            valid_fractions.append(float(np.mean(finite)))
            depth_medians.append(float(np.median(depth_row[finite])))
            if position in review_positions:
                save_depth_review(
                    still_dir / f"frame_{frame_ids[position]:06d}.png",
                    rgb_paths[position],
                    depth_row,
                )
        depth_final.flush()
        confidence_error_final.flush()

        processed_summary_rows = []
        for position, values in enumerate(processed_intrinsics_by_frame):
            array = np.stack(values, axis=0)
            processed_summary_rows.append({
                "frame_idx": frame_ids[position],
                "contribution_count": int(len(values)),
                "processed_intrinsics_median_fx_fy_cx_cy": np.median(array, axis=0).tolist(),
                "processed_intrinsics_max_abs_deviation": np.max(np.abs(array - np.median(array, axis=0)), axis=0).tolist(),
            })
        model_returned_processed_intrinsics = np.asarray(
            [row["processed_intrinsics_median_fx_fy_cx_cy"] for row in processed_summary_rows],
            dtype=np.float64,
        )
        output_intrinsics_rows = np.repeat(np.asarray(output_intrinsics, dtype=np.float64)[None], len(rows), axis=0)
        np.savez_compressed(
            archive_path,
            frame_idx=np.asarray(frame_ids, dtype=np.int32),
            depth=np.asarray(depth_final),
            confidence=np.asarray(confidence_error_final),
            confidence_semantics=np.asarray(CONFIDENCE_SEMANTICS),
            confidence_role=np.asarray("predicted_error_proxy_higher_is_worse"),
            source_size=np.asarray(output_size_wh, dtype=np.int32),
            focal_px=np.repeat(np.sqrt(output_intrinsics[0] * output_intrinsics[1]), len(rows)).astype(np.float64),
            intrinsics_fx_fy_cx_cy=output_intrinsics_rows,
            source_estimated_intrinsics_fx_fy_cx_cy=output_intrinsics_rows,
            model_returned_processed_intrinsics_fx_fy_cx_cy=model_returned_processed_intrinsics,
            model_input_intrinsics_fx_fy_cx_cy=np.repeat(np.asarray(input_intrinsics)[None], len(rows), axis=0),
            inference_camera_extrinsics_w2c=w2c.astype(np.float32),
            depth_provider=np.asarray(DEPTH_PROVIDER),
            model_id=np.asarray(str(args.model_id)),
            model_path=np.asarray(str(args.model_path.expanduser().resolve())),
            camera_conditioning_mode=np.asarray("provided_pinhole_intrinsics_and_metric_extrinsics"),
            pose_conditioning_mode=np.asarray(POSE_CONDITIONING_MODE),
            metric_scale_mode=np.asarray(args.metric_scale_mode),
            metric_scale_source=np.asarray(
                "DA3_Nested_metric_branch_no_input_trajectory_Umeyama_depth_rescale"
                if args.metric_scale_mode == "nested_metric_branch"
                else "DA3_API_input_trajectory_Umeyama_depth_rescale"
            ),
            camera_trajectory_source=np.asarray(str(args.hawor_npz.expanduser().resolve())),
            depth_ray_geometry_reprojected=np.asarray(True),
            depth_output_quantity=np.asarray("camera_z_m_ray_remapped_from_DA3_processed_K_to_exact_output_contract_K"),
            inference_camera_contract_path=np.asarray(str(contract_path)),
            inference_camera_contract_sha256=np.asarray(sha256_file(contract_path)),
            inference_camera_input_plane=np.asarray(str(args.camera_input_plane)),
            inference_camera_output_plane=np.asarray(str(args.camera_output_plane)),
            inference_camera_intrinsics_fx_fy_cx_cy=output_intrinsics_rows,
            window_size=np.asarray(int(args.window_size)),
            window_overlap=np.asarray(int(args.window_overlap)),
            window_contribution_count=contribution_count,
            overlap_consistency_passed=np.asarray(bool(overlap_consistency["passed"])),
            overlap_consistency_required=np.asarray(bool(overlap_consistency["required"])),
            overlap_median_relative_difference=np.asarray(
                overlap_consistency["observed"]["median_relative_difference"]
                if overlap_consistency["observed"]["median_relative_difference"] is not None
                else np.nan,
                dtype=np.float64,
            ),
            overlap_scale_aligned_median_relative_difference=np.asarray(
                overlap_consistency["observed"]["scale_aligned_median_relative_difference"]
                if overlap_consistency["observed"]["scale_aligned_median_relative_difference"] is not None
                else np.nan,
                dtype=np.float64,
            ),
        )
        report = {
            "status": "ok",
            "annotation_ready": False,
            "method": "run_da3_pose_conditioned_full_frame_v1",
            "claim_scope": (
                "controlled prediction-side DA3 metric-depth measurement with official fixed K and fixed HaWoR metric "
                "camera trajectory; DA3 camera outputs are not camera authority; SAM3D remains pointmap=None"
            ),
            "model": {
                "id": str(args.model_id),
                "path": str(args.model_path.expanduser().resolve()),
                "family": "DA3 Nested any-view plus monocular metric branch",
                "license": "CC BY-NC 4.0",
                "usage": "non-commercial research/evaluation only",
                "repository": str(args.da3_repo.expanduser().resolve()),
                "repository_revision": git_revision(args.da3_repo.expanduser().resolve()),
            },
            "inputs": {
                "manifest": str(args.manifest),
                "camera_contract": str(contract_path),
                "camera_contract_sha256": sha256_file(contract_path),
                "hawor_camera_npz": camera_report,
            },
            "outputs": {
                "depth_archive": str(archive_path),
                "stills_dir": str(still_dir),
            },
            "frame_count": len(rows),
            "frame_range": [frame_ids[0], frame_ids[-1]],
            "camera_contract": {
                "input_plane": args.camera_input_plane,
                "input_size_wh": list(input_size_wh),
                "input_intrinsics_fx_fy_cx_cy": np.asarray(input_intrinsics).tolist(),
                "input_plane_resolution": input_plane,
                "output_plane": args.camera_output_plane,
                "output_size_wh": list(output_size_wh),
                "output_intrinsics_fx_fy_cx_cy": np.asarray(output_intrinsics).tolist(),
                "output_plane_resolution": output_plane,
                "depth_ray_geometry_reprojected": True,
                "maximum_sampled_ray_angle_error_deg": maximum_ray_error_deg,
            },
            "pose_conditioning": {
                "mode": POSE_CONDITIONING_MODE,
                "input_extrinsics_convention": "OpenCV world_to_camera 4x4",
                "DA3_predicted_camera_promoted": False,
                "align_to_input_ext_scale": align_to_input_ext_scale,
                "metric_scale_mode": args.metric_scale_mode,
                "metric_scale_source": (
                    "DA3 Nested metric branch; supplied K/W2C condition any-view geometry but returned camera is diagnostic"
                    if args.metric_scale_mode == "nested_metric_branch"
                    else "DA3 API Umeyama alignment to supplied prediction-side trajectory; ablation only"
                ),
                "use_ray_pose": False,
                "camera_report": camera_report,
            },
            "windowing": {
                "window_size": int(args.window_size),
                "overlap": int(args.window_overlap),
                "windows": window_rows,
                "contribution_count": summarize(contribution_count.tolist()),
                "overlap_rows": overlap_rows,
                "overlap_absolute_depth_median_m": summarize([
                    row["absolute_depth_difference_m"].get("median", float("nan")) for row in overlap_rows
                ]),
                "overlap_relative_depth_median": summarize(overlap_relative_medians),
                "overlap_scale_aligned_relative_depth_median": summarize(
                    overlap_scale_aligned_relative_medians
                ),
                "overlap_consistency": overlap_consistency,
            },
            "confidence": {
                "archive_field": "confidence",
                "semantics": CONFIDENCE_SEMANTICS,
                "source_field": "DA3 prediction.conf (not archived separately)",
                "conversion": "error_proxy = 1 / max(raw_DA3_confidence, epsilon)",
                "metric_error_calibrated": False,
            },
            "minimum_reprojected_valid_fraction": minimum_valid_fraction,
            "final_valid_fraction": summarize(valid_fractions),
            "depth_median_m": summarize(depth_medians),
            "processed_intrinsics_rows": processed_summary_rows,
            "causal_boundaries": [
                "Only the external dense-depth measurement changes; SAM3D internal MoGe and pointmap=None remain unchanged.",
                "HaWoR camera trajectory is prediction-side model output, not HOT3D released camera ground truth.",
                "DA3 multi-view consistency may fail on moving hands or manipulated objects; downstream A/B must stratify dynamic foreground and boundaries.",
                "DA3 confidence is converted only to a monotonic error proxy and is not treated as calibrated metric uncertainty.",
            ],
            "elapsed_s": float(time.time() - started),
        }
        write_json(success_report_path, report)
        print(json.dumps({key: value for key, value in report.items() if key not in {"processed_intrinsics_rows"}}, indent=2))
        return report
    finally:
        # Explicitly close every mmap before CIFS deletion. Cleanup must never
        # replace the inference exception with a transient ENOTEMPTY from NAS.
        for array in (
            depth_final,
            confidence_error_final,
            depth_numerator,
            weight_sum,
            confidence_numerator,
            taper_sum,
        ):
            close_memmap(array)
        depth_final = confidence_error_final = None
        depth_numerator = weight_sum = confidence_numerator = taper_sum = None
        model = None
        gc.collect()
        if temp_dir.exists() and not args.keep_temporary_accumulators:
            try:
                remove_temporary_tree(temp_dir)
            except OSError as error:
                print(f"warning: could not remove DA3 temporary directory {temp_dir}: {error}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--da3-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="da3_pose_conditioned_full_frame_depth_v1.npz")
    parser.add_argument("--camera-input-plane", default="manifest_rgb")
    parser.add_argument("--camera-output-plane", default="source_rgb")
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--source-width", type=int, required=True)
    parser.add_argument("--source-height", type=int, required=True)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-overlap", type=int, default=4)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--ref-view-strategy", default="middle")
    parser.add_argument("--metric-scale-mode", choices=sorted(METRIC_SCALE_MODES), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--returned-camera-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--min-reprojected-valid-fraction", type=float, default=0.99)
    parser.add_argument("--ray-validation-max-angle-deg", type=float, default=1.0e-5)
    parser.add_argument("--confidence-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--overlap-diagnostic-stride", type=int, default=16)
    parser.add_argument("--max-overlap-median-relative-difference", type=float, default=0.10)
    parser.add_argument("--max-overlap-frame-relative-difference", type=float, default=0.25)
    parser.add_argument("--max-overlap-scale-aligned-median-relative-difference", type=float, default=0.10)
    parser.add_argument("--review-frame-count", type=int, default=12)
    parser.add_argument("--keep-temporary-accumulators", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if int(args.overlap_diagnostic_stride) <= 0:
        parser.error("--overlap-diagnostic-stride must be positive")
    if not 0.0 < float(args.min_reprojected_valid_fraction) <= 1.0:
        parser.error("--min-reprojected-valid-fraction must be in (0,1]")
    for name in (
        "max_overlap_median_relative_difference",
        "max_overlap_frame_relative_difference",
        "max_overlap_scale_aligned_median_relative_difference",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
