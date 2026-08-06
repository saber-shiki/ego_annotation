#!/usr/bin/env python3
"""Evaluate a blind V19 run against the available Ego-Exo4D reference data.

Supported quantitative claims are deliberately limited to sparse visible-object
mask quality and named 2-D/3-D hand keypoints. Released Ego-Exo4D annotations in
this NAS shard do not contain object CAD/6-DoF/contact/nonpenetration GT, so this
evaluator emits explicit `not_evaluated` records for those state families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_summary(values: list[float], scale: float = 1.0) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)] * scale
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def signed_error_summary(values: list[float], scale: float = 1.0) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)] * scale
    if not len(array):
        return {"count": 0, "mean_signed": None, "median_signed": None, "mean_absolute": None, "median_absolute": None, "p95_absolute": None}
    absolute = np.abs(array)
    return {
        "count": int(len(array)),
        "mean_signed": float(np.mean(array)),
        "median_signed": float(np.median(array)),
        "mean_absolute": float(np.mean(absolute)),
        "median_absolute": float(np.median(absolute)),
        "p95_absolute": float(np.percentile(absolute, 95)),
    }


def bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def bbox_iou(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    xy0 = np.maximum(left[:2], right[:2])
    xy1 = np.minimum(left[2:], right[2:])
    intersection = float(np.prod(np.maximum(0.0, xy1 - xy0)))
    left_area = float(np.prod(np.maximum(0.0, left[2:] - left[:2])))
    right_area = float(np.prod(np.maximum(0.0, right[2:] - right[:2])))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    return mask & ~eroded


def boundary_f1(prediction: np.ndarray, ground_truth: np.ndarray, tolerance_px: int) -> tuple[float, float, float]:
    pred_boundary = mask_boundary(prediction)
    gt_boundary = mask_boundary(ground_truth)
    kernel_size = 2 * tolerance_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel) > 0
    gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), kernel) > 0
    precision = float(np.count_nonzero(pred_boundary & gt_dilated) / max(1, np.count_nonzero(pred_boundary)))
    recall = float(np.count_nonzero(gt_boundary & pred_dilated) / max(1, np.count_nonzero(gt_boundary)))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def resolve_object_id(run_root: Path, requested: str | None) -> str:
    if requested:
        return requested
    root = run_root / "measurements" / "object_tracks" / "sam2_owlv2_box_points"
    candidates = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else []
    if len(candidates) != 1:
        raise RuntimeError(f"cannot infer one runtime object id: {candidates}")
    return candidates[0]


def evaluate_masks(
    run_root: Path,
    input_video: Path,
    gt_dir: Path,
    object_id: str,
    output_dir: Path,
    boundary_tolerance_px: int,
) -> dict[str, Any]:
    gt_report = load_json(gt_dir / "object_visible_mask_gt.json")
    gt_mask_dir = gt_dir / f"object_visible_masks_{gt_report['frames'][0]['runtime_grid'][0]}"
    prediction_dir = (
        run_root
        / "measurements"
        / "object_tracks"
        / "sam2_owlv2_box_points"
        / object_id
        / "sam2"
        / "sam2_masks"
    )
    capture = cv2.VideoCapture(str(input_video))
    rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    for gt_row in gt_report["frames"]:
        frame_idx = int(gt_row["local_frame_idx"])
        gt_path = gt_mask_dir / gt_row["runtime_mask_file"]
        pred_path = prediction_dir / f"{frame_idx:06d}.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(gt_path)
        gt_mask = gt > 0
        pred_image = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE) if pred_path.exists() else None
        if pred_image is None:
            pred_mask = np.zeros_like(gt_mask)
            status = "missing_prediction_mask"
        else:
            if pred_image.shape != gt.shape:
                pred_image = cv2.resize(pred_image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_mask = pred_image > 0
            status = "scored"
        intersection = int(np.count_nonzero(pred_mask & gt_mask))
        union = int(np.count_nonzero(pred_mask | gt_mask))
        pred_area = int(np.count_nonzero(pred_mask))
        gt_area = int(np.count_nonzero(gt_mask))
        iou = intersection / union if union else 1.0
        precision = intersection / pred_area if pred_area else 0.0
        recall = intersection / gt_area if gt_area else 0.0
        bprecision, brecall, bf1 = boundary_f1(pred_mask, gt_mask, boundary_tolerance_px)
        pred_center = np.asarray(np.where(pred_mask)[::-1], dtype=np.float64).mean(axis=1) if pred_area else None
        gt_center = np.asarray(np.where(gt_mask)[::-1], dtype=np.float64).mean(axis=1) if gt_area else None
        center_error = float(np.linalg.norm(pred_center - gt_center)) if pred_center is not None and gt_center is not None else None
        rows.append(
            {
                "local_frame_idx": frame_idx,
                "source_frame_idx": int(gt_row["source_frame_idx"]),
                "status": status,
                "gt_mask": str(gt_path),
                "prediction_mask": str(pred_path),
                "gt_area_px": gt_area,
                "prediction_area_px": pred_area,
                "intersection_px": intersection,
                "union_px": union,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "bbox_iou": bbox_iou(bbox_xyxy(pred_mask), bbox_xyxy(gt_mask)),
                "centroid_error_px": center_error,
                "boundary_precision": bprecision,
                "boundary_recall": brecall,
                "boundary_f1": bf1,
            }
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, image = capture.read()
        if ok:
            image = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            visualization = image.copy()
            visualization[gt_mask] = (40, 210, 40)
            visualization[pred_mask] = (220, 40, 220)
            visualization[gt_mask & pred_mask] = (245, 245, 245)
            visualization = cv2.addWeighted(image, 0.40, visualization, 0.60, 0)
            cv2.rectangle(visualization, (0, 0), (visualization.shape[1], 55), (0, 0, 0), -1)
            cv2.putText(
                visualization,
                f"f{frame_idx} IoU={iou:.3f} green=GT magenta=pred white=intersection",
                (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(cv2.resize(visualization, (480, 480), interpolation=cv2.INTER_AREA))
    capture.release()
    if panels:
        sheet = np.concatenate(panels, axis=1)
        cv2.imwrite(str(output_dir / "object_mask_comparison.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return {
        "status": "scored" if rows else "no_gt_mask_rows",
        "claim_scope": "sparse visible-mask quality at released Ego-Exo4D annotated frames; not amodal geometry or 6-DoF pose",
        "object_id": object_id,
        "prediction_mask_dir": str(prediction_dir),
        "gt_annotated_frame_count": len(rows),
        "predicted_mask_present_count": sum(row["status"] == "scored" for row in rows),
        "missing_predictions_are_scored_as_empty": True,
        "boundary_tolerance_px": boundary_tolerance_px,
        "aggregate": {
            "iou": finite_summary([row["iou"] for row in rows]),
            "precision": finite_summary([row["precision"] for row in rows]),
            "recall": finite_summary([row["recall"] for row in rows]),
            "bbox_iou": finite_summary([row["bbox_iou"] for row in rows]),
            "centroid_error_px": finite_summary([row["centroid_error_px"] for row in rows if row["centroid_error_px"] is not None]),
            "boundary_f1": finite_summary([row["boundary_f1"] for row in rows]),
        },
        "per_frame": rows,
        "visualization": str(output_dir / "object_mask_comparison.jpg") if panels else None,
    }


def world_to_camera_hawor(points_world: np.ndarray, rotation_camera_to_world: np.ndarray, translation_camera_to_world: np.ndarray) -> np.ndarray:
    return (points_world - translation_camera_to_world[None, :]) @ rotation_camera_to_world


def world_to_camera_gt(points_world: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1)
    return homogeneous @ world_to_camera.T


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = points_camera @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def load_hawor_joint_map(hawor_path: Path) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, Any]]:
    blob = np.load(hawor_path, allow_pickle=True)
    frame_indices = np.asarray(blob["frame_idx"], dtype=int)
    result: dict[tuple[int, str], np.ndarray] = {}
    for side in ("left", "right"):
        joints = np.asarray(blob[f"{side}_joints_world_m"], dtype=np.float64)
        valid = np.asarray(blob[f"{side}_valid"], dtype=bool)
        for row_index, frame_idx in enumerate(frame_indices):
            if valid[row_index] and np.isfinite(joints[row_index]).all():
                result[(int(frame_idx), side)] = joints[row_index]
    camera = {
        "frame_idx": frame_indices,
        "R_c2w": np.asarray(blob["R_c2w"], dtype=np.float64),
        "t_c2w": np.asarray(blob["t_c2w"], dtype=np.float64),
    }
    return result, camera


def load_interval_joint_map(path: Path) -> dict[tuple[int, str], np.ndarray]:
    payload = load_json(path)
    result: dict[tuple[int, str], np.ndarray] = {}
    for row in payload.get("per_frame_states", []):
        joints = np.asarray(row.get("optimized_joints_world_m", []), dtype=np.float64)
        if joints.shape == (21, 3) and np.isfinite(joints).all():
            result[(int(row["frame_idx"]), str(row["hand_side"]))] = joints
    return result


def camera_map(camera: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        int(frame_idx): (camera["R_c2w"][index], camera["t_c2w"][index])
        for index, frame_idx in enumerate(camera["frame_idx"])
    }


def fit_world_alignment(source_xyz: np.ndarray, target_xyz: np.ndarray, estimate_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target ~= scale * rotation @ source + translation (Umeyama)."""
    source_mean = source_xyz.mean(axis=0)
    target_mean = target_xyz.mean(axis=0)
    source_centered = source_xyz - source_mean
    target_centered = target_xyz - target_mean
    covariance = target_centered.T @ source_centered / len(source_xyz)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_transpose
    if estimate_scale:
        source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
        scale = float(np.sum(singular_values * sign) / source_variance) if source_variance > 0 else 1.0
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def rotation_angle_rad(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def fit_orientation_gauge(source_rotations: np.ndarray, target_rotations: np.ndarray) -> np.ndarray:
    cross_covariance = np.zeros((3, 3), dtype=np.float64)
    for source, target in zip(source_rotations, target_rotations):
        cross_covariance += target @ source.T
    left, _, right_transpose = np.linalg.svd(cross_covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def evaluate_camera_trajectory(hawor_camera: dict[str, Any], gt_camera: dict[str, Any]) -> dict[str, Any]:
    annotation_to_extracted = np.asarray(
        gt_camera["coordinate_contract"]["annotation_camera_to_prediction_extracted_camera_3x3"],
        dtype=np.float64,
    )
    gt_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in gt_camera["frames"]:
        world_to_camera = np.asarray(row["world_to_camera_3x4"], dtype=np.float64)
        rotation_world_to_camera = world_to_camera[:, :3]
        translation_world_to_camera = world_to_camera[:, 3]
        rotation_camera_to_world = rotation_world_to_camera.T
        center_world = -rotation_camera_to_world @ translation_world_to_camera
        gt_by_frame[int(row["local_frame_idx"])] = (rotation_camera_to_world, center_world)
    predicted_by_frame = camera_map(hawor_camera)
    frame_indices = sorted(set(gt_by_frame).intersection(predicted_by_frame))
    if len(frame_indices) < 3:
        return {"status": "not_evaluated_insufficient_common_camera_frames", "common_frame_count": len(frame_indices)}

    gt_rotations = np.asarray([gt_by_frame[index][0] for index in frame_indices], dtype=np.float64)
    gt_centers = np.asarray([gt_by_frame[index][1] for index in frame_indices], dtype=np.float64)
    pred_rotations_extracted = np.asarray([predicted_by_frame[index][0] for index in frame_indices], dtype=np.float64)
    pred_rotations = np.einsum("nij,jk->nik", pred_rotations_extracted, annotation_to_extracted)
    pred_centers = np.asarray([predicted_by_frame[index][1] for index in frame_indices], dtype=np.float64)

    alignments: dict[str, Any] = {}
    aligned_rotations_se3: np.ndarray | None = None
    for label, estimate_scale in (("se3_metric", False), ("sim3_diagnostic", True)):
        scale, rotation, translation = fit_world_alignment(pred_centers, gt_centers, estimate_scale)
        aligned_centers = scale * (pred_centers @ rotation.T) + translation
        center_errors = np.linalg.norm(aligned_centers - gt_centers, axis=1)
        aligned_rotations = np.einsum("ij,njk->nik", rotation, pred_rotations)
        orientation_errors = [
            rotation_angle_rad(gt_rotation.T @ pred_rotation)
            for gt_rotation, pred_rotation in zip(gt_rotations, aligned_rotations)
        ]
        if label == "se3_metric":
            aligned_rotations_se3 = aligned_rotations
        alignments[label] = {
            "alignment_scope": "all common frames; world-gauge alignment only" if not estimate_scale else "all common frames; diagnostic scale is fitted and therefore not a metric-scale claim",
            "scale": scale,
            "rotation_pred_world_to_gt_world_3x3": rotation.tolist(),
            "translation_pred_world_to_gt_world_m": translation.tolist(),
            "ate_center_error_mm": finite_summary(center_errors.tolist(), 1000.0),
            "ate_center_rmse_mm": float(np.sqrt(np.mean(center_errors**2)) * 1000.0),
            "orientation_error_under_position_fitted_world_gauge_deg": finite_summary(orientation_errors, 180.0 / math.pi),
            "orientation_metric_caveat": "position-only alignment can leave rotation about a weakly observed trajectory axis poorly determined; use orientation_gauge_alignment and gauge-invariant RPE for orientation conclusions",
        }

    orientation_rotation = fit_orientation_gauge(pred_rotations, gt_rotations)
    orientation_translation = gt_centers.mean(axis=0) - orientation_rotation @ pred_centers.mean(axis=0)
    orientation_aligned_centers = pred_centers @ orientation_rotation.T + orientation_translation
    orientation_aligned_rotations = np.einsum("ij,njk->nik", orientation_rotation, pred_rotations)
    orientation_errors = [
        rotation_angle_rad(gt_rotation.T @ pred_rotation)
        for gt_rotation, pred_rotation in zip(gt_rotations, orientation_aligned_rotations)
    ]
    orientation_center_errors = np.linalg.norm(orientation_aligned_centers - gt_centers, axis=1)
    orientation_gauge_report = {
        "alignment_scope": "all common camera orientations; SO(3) world gauge fitted from rotations, with center-mean translation and fixed unit scale",
        "scale": 1.0,
        "rotation_pred_world_to_gt_world_3x3": orientation_rotation.tolist(),
        "translation_pred_world_to_gt_world_m": orientation_translation.tolist(),
        "orientation_error_deg": finite_summary(orientation_errors, 180.0 / math.pi),
        "corresponding_center_error_mm": finite_summary(orientation_center_errors.tolist(), 1000.0),
        "corresponding_center_rmse_mm": float(np.sqrt(np.mean(orientation_center_errors**2)) * 1000.0),
        "interpretation": "primary absolute-orientation diagnostic; corresponding position error is shown to expose disagreement between position-fitted and orientation-fitted gauges",
    }

    assert aligned_rotations_se3 is not None
    relative_reports: dict[str, Any] = {}
    se3_scale, se3_rotation, se3_translation = fit_world_alignment(pred_centers, gt_centers, False)
    aligned_centers_se3 = se3_scale * (pred_centers @ se3_rotation.T) + se3_translation
    for delta in (1, 30):
        aligned_world_center_errors: list[float] = []
        local_camera_translation_errors: list[float] = []
        rotation_errors: list[float] = []
        for first in range(len(frame_indices) - delta):
            second = first + delta
            if frame_indices[second] - frame_indices[first] != delta:
                continue
            gt_displacement = gt_centers[second] - gt_centers[first]
            pred_displacement = aligned_centers_se3[second] - aligned_centers_se3[first]
            aligned_world_center_errors.append(float(np.linalg.norm(pred_displacement - gt_displacement)))
            gt_translation_in_first_camera = gt_rotations[first].T @ gt_displacement
            pred_translation_in_first_camera = pred_rotations[first].T @ (pred_centers[second] - pred_centers[first])
            local_camera_translation_errors.append(float(np.linalg.norm(pred_translation_in_first_camera - gt_translation_in_first_camera)))
            gt_relative_rotation = gt_rotations[first].T @ gt_rotations[second]
            pred_relative_rotation = aligned_rotations_se3[first].T @ aligned_rotations_se3[second]
            rotation_errors.append(rotation_angle_rad(gt_relative_rotation.T @ pred_relative_rotation))
        relative_reports[f"delta_{delta}_frames"] = {
            "pair_count": len(aligned_world_center_errors),
            "local_camera_translation_error_mm": finite_summary(local_camera_translation_errors, 1000.0),
            "position_aligned_world_center_displacement_error_mm": finite_summary(aligned_world_center_errors, 1000.0),
            "relative_rotation_error_deg": finite_summary(rotation_errors, 180.0 / math.pi),
            "primary_rpe_contract": "local-camera translation and relative rotation are world-gauge invariant; aligned-world center displacement is retained as a secondary position-fit diagnostic",
        }

    gt_steps = np.linalg.norm(np.diff(gt_centers, axis=0), axis=1)
    pred_steps = np.linalg.norm(np.diff(pred_centers, axis=0), axis=1)
    position_singular_values = np.linalg.svd(pred_centers - pred_centers.mean(axis=0), compute_uv=False)
    return {
        "status": "scored",
        "claim_scope": "HaWoR camera-to-world trajectory versus released Ego-Exo4D camera trajectory after the official extracted-view/original-camera axis adapter; SE(3) preserves predicted metric scale, while Sim(3) is diagnostic only",
        "camera_basis_adapter": {
            "annotation_camera_to_prediction_extracted_camera_3x3": annotation_to_extracted.tolist(),
            "source": "facebookresearch/Ego4d aria_original_to_extracted",
            "fitted_from_evaluation_data": False,
        },
        "common_frame_count": len(frame_indices),
        "frame_start": frame_indices[0],
        "frame_end": frame_indices[-1],
        "trajectory_length_m": {
            "ground_truth": float(np.sum(gt_steps)),
            "prediction": float(np.sum(pred_steps)),
            "prediction_over_ground_truth": float(np.sum(pred_steps) / np.sum(gt_steps)) if np.sum(gt_steps) > 0 else None,
        },
        "position_alignment_observability": {
            "predicted_center_singular_values_m": position_singular_values.tolist(),
            "smallest_over_largest": float(position_singular_values[-1] / position_singular_values[0]) if position_singular_values[0] > 0 else None,
        },
        "alignments": alignments,
        "orientation_gauge_alignment": orientation_gauge_report,
        "relative_pose": relative_reports,
    }


def audit_calibration_provenance(run_root: Path, input_video: Path, gt_camera: dict[str, Any]) -> dict[str, Any]:
    prediction_path = input_video.parent / "v19_camera_calibration_contract.json"
    runtime_path = run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
    if not runtime_path.exists():
        return {
            "status": "not_compared_missing_runtime_contract",
            "prediction_contract": str(prediction_path),
            "runtime_contract": str(runtime_path),
        }
    runtime = load_json(runtime_path)
    if not prediction_path.exists():
        return {
            "status": "runtime_estimated_pinhole_raw_aria_calibration_unavailable",
            "prediction_contract": None,
            "runtime_contract": str(runtime_path),
            "runtime_intrinsics_source": runtime.get("intrinsics_source"),
            "runtime_fx_fy_cx_cy": runtime.get("intrinsics_fx_fy_cx_cy"),
            "released_rectified_intrinsics_512": gt_camera.get("intrinsics_rectified_512_3x3"),
            "rectified_intrinsics_are_not_raw_rgb_intrinsics": True,
            "raw_distortion_calibration_available_in_local_shard": False,
            "evaluation_consequence": "the runtime pinhole model is a monocular estimate over raw distorted RGB; the released 512x512 rectified K is evaluation-only and must not be treated as the input-view K",
        }
    prediction = load_json(prediction_path)
    prediction_values = np.asarray(prediction.get("intrinsics_fx_fy_cx_cy", []), dtype=np.float64)
    runtime_values = np.asarray(runtime.get("intrinsics_fx_fy_cx_cy", []), dtype=np.float64)
    comparable = prediction_values.shape == (4,) and runtime_values.shape == (4,)
    delta = runtime_values - prediction_values if comparable else None
    matched = bool(comparable and np.allclose(runtime_values, prediction_values, rtol=0.0, atol=1e-6))
    return {
        "status": "invalid_legacy_prediction_contract_detected",
        "prediction_contract": str(prediction_path),
        "runtime_contract": str(runtime_path),
        "prediction_intrinsics_source": prediction.get("intrinsics_source"),
        "runtime_intrinsics_source": runtime.get("intrinsics_source"),
        "prediction_fx_fy_cx_cy": prediction_values.tolist() if prediction_values.shape == (4,) else None,
        "runtime_fx_fy_cx_cy": runtime_values.tolist() if runtime_values.shape == (4,) else None,
        "runtime_minus_prediction_px": delta.tolist() if delta is not None else None,
        "exact_within_1e_6_px": matched,
        "legacy_contract_invalid_reason": "the released ego-pose K belongs to a 512x512 undistorted linear view, not the raw 448/960 Aria RGB MP4",
        "runtime_consumed_legacy_contract": matched,
        "evaluation_consequence": "do not classify this as a released-intrinsics run; the runtime used its own estimated raw-view pinhole approximation",
    }


def evaluate_hand_joint_map(
    label: str,
    joint_map: dict[tuple[int, str], np.ndarray],
    hawor_camera: dict[int, tuple[np.ndarray, np.ndarray]],
    gt_hand: dict[str, Any],
    gt_camera: dict[str, Any],
    min_num_views: int,
    max_gt_reprojection_px_rectified_512: float,
) -> dict[str, Any]:
    gt_extrinsics = {int(row["local_frame_idx"]): np.asarray(row["world_to_camera_3x4"], dtype=np.float64) for row in gt_camera["frames"]}
    intrinsics = np.asarray(gt_camera["intrinsics_rectified_512_3x3"], dtype=np.float64)
    annotation_to_extracted = np.asarray(
        gt_camera["coordinate_contract"]["annotation_camera_to_prediction_extracted_camera_3x3"],
        dtype=np.float64,
    )
    absolute_errors: dict[str, list[float]] = {"left": [], "right": []}
    root_relative_errors: dict[str, list[float]] = {"left": [], "right": []}
    reprojection_errors: dict[str, list[float]] = {"left": [], "right": []}
    wrist_errors: dict[str, list[float]] = {"left": [], "right": []}
    axis_errors: dict[str, dict[str, list[float]]] = {
        side: {axis: [] for axis in ("x", "y", "z")} for side in ("left", "right")
    }
    wrist_axis_errors: dict[str, dict[str, list[float]]] = {
        side: {axis: [] for axis in ("x", "y", "z")} for side in ("left", "right")
    }
    per_frame: list[dict[str, Any]] = []
    gt_joint_count = 0
    matched_joint_count = 0
    for gt_row in gt_hand["frames"]:
        frame_idx = int(gt_row["local_frame_idx"])
        if frame_idx not in gt_extrinsics or frame_idx not in hawor_camera:
            continue
        world_to_camera = gt_extrinsics[frame_idx]
        rotation_camera_to_world, translation_camera_to_world = hawor_camera[frame_idx]
        for side in ("left", "right"):
            hand = gt_row.get("hands", {}).get(side)
            if not hand:
                continue
            prediction = joint_map.get((frame_idx, side))
            frame_absolute: list[float] = []
            frame_relative: list[float] = []
            frame_reprojection: list[float] = []
            gt_points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for joint in hand["joints"]:
                if int(joint.get("num_views_for_3d", 0)) < min_num_views:
                    continue
                gt_reprojection = joint.get("released_3d_to_2d_reprojection_error_px_rectified_512")
                if gt_reprojection is not None and float(gt_reprojection) > max_gt_reprojection_px_rectified_512:
                    continue
                uv = joint.get("uv_rectified_512_px")
                if uv is None:
                    continue
                gt_points[int(joint["hawor_openpose_index"])] = (
                    np.asarray(joint["xyz_world_m"], dtype=np.float64),
                    np.asarray(uv, dtype=np.float64),
                )
            gt_joint_count += len(gt_points)
            if prediction is None or prediction.shape != (21, 3):
                per_frame.append(
                    {
                        "local_frame_idx": frame_idx,
                        "side": side,
                        "status": "missing_prediction",
                        "eligible_gt_joint_count": len(gt_points),
                    }
                )
                continue
            gt_indices = sorted(gt_points)
            if not gt_indices:
                continue
            gt_world = np.asarray([gt_points[index][0] for index in gt_indices], dtype=np.float64)
            gt_uv = np.asarray([gt_points[index][1] for index in gt_indices], dtype=np.float64)
            pred_world = prediction[gt_indices]
            gt_cam = world_to_camera_gt(gt_world, world_to_camera)
            pred_cam_extracted = world_to_camera_hawor(pred_world, rotation_camera_to_world, translation_camera_to_world)
            pred_cam = pred_cam_extracted @ annotation_to_extracted
            errors = np.linalg.norm(pred_cam - gt_cam, axis=1)
            signed_deltas = pred_cam - gt_cam
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                axis_errors[side][axis_name].extend(signed_deltas[:, axis_index].tolist())
            pred_uv = project(pred_cam, intrinsics)
            uv_errors = np.linalg.norm(pred_uv - gt_uv, axis=1)
            frame_absolute.extend(errors.tolist())
            frame_reprojection.extend(uv_errors.tolist())
            absolute_errors[side].extend(errors.tolist())
            reprojection_errors[side].extend(uv_errors.tolist())
            matched_joint_count += len(errors)
            if 0 in gt_points:
                gt_wrist_cam = world_to_camera_gt(gt_points[0][0][None, :], world_to_camera)[0]
                pred_wrist_cam_extracted = world_to_camera_hawor(prediction[0][None, :], rotation_camera_to_world, translation_camera_to_world)[0]
                pred_wrist_cam = pred_wrist_cam_extracted @ annotation_to_extracted
                wrist_error = float(np.linalg.norm(pred_wrist_cam - gt_wrist_cam))
                wrist_errors[side].append(wrist_error)
                wrist_delta = pred_wrist_cam - gt_wrist_cam
                for axis_index, axis_name in enumerate(("x", "y", "z")):
                    wrist_axis_errors[side][axis_name].append(float(wrist_delta[axis_index]))
                relative = np.linalg.norm((pred_cam - pred_wrist_cam) - (gt_cam - gt_wrist_cam), axis=1)
                frame_relative.extend(relative.tolist())
                root_relative_errors[side].extend(relative.tolist())
            per_frame.append(
                {
                    "local_frame_idx": frame_idx,
                    "side": side,
                    "status": "scored",
                    "eligible_gt_joint_count": len(gt_indices),
                    "absolute_mpjpe_mm": float(np.mean(frame_absolute) * 1000.0),
                    "root_relative_mpjpe_mm": float(np.mean(frame_relative) * 1000.0) if frame_relative else None,
                    "reprojection_mean_px": float(np.mean(frame_reprojection)),
                }
            )
    all_absolute = absolute_errors["left"] + absolute_errors["right"]
    all_relative = root_relative_errors["left"] + root_relative_errors["right"]
    all_reprojection = reprojection_errors["left"] + reprojection_errors["right"]
    return {
        "status": "scored" if matched_joint_count else "no_matched_joints",
        "label": label,
        "claim_scope": "named Ego-Exo4D hand joints in the official annotation camera after a fixed, non-fitted extracted-view axis adapter; root-relative removes wrist translation only; reprojection is on the rectified 512x512 view, not raw RGB; not MANO vertex/contact scoring",
        "camera_basis_adapter": {
            "annotation_camera_to_prediction_extracted_camera_3x3": annotation_to_extracted.tolist(),
            "source": "facebookresearch/Ego4d aria_original_to_extracted",
            "fitted_from_evaluation_data": False,
        },
        "quality_filter": {
            "min_num_views_for_3d": min_num_views,
            "max_released_gt_reprojection_error_px_rectified_512": max_gt_reprojection_px_rectified_512,
        },
        "coverage": {
            "eligible_gt_joint_count": gt_joint_count,
            "matched_joint_count": matched_joint_count,
            "matched_fraction": matched_joint_count / gt_joint_count if gt_joint_count else 0.0,
        },
        "all_hands": {
            "absolute_joint_error_mm": finite_summary(all_absolute, 1000.0),
            "root_relative_joint_error_mm": finite_summary(all_relative, 1000.0),
            "wrist_error_mm": finite_summary(wrist_errors["left"] + wrist_errors["right"], 1000.0),
            "signed_camera_axis_error_mm": {
                axis: signed_error_summary(axis_errors["left"][axis] + axis_errors["right"][axis], 1000.0)
                for axis in ("x", "y", "z")
            },
            "signed_wrist_camera_axis_error_mm": {
                axis: signed_error_summary(wrist_axis_errors["left"][axis] + wrist_axis_errors["right"][axis], 1000.0)
                for axis in ("x", "y", "z")
            },
            "reprojection_error_px_rectified_512": finite_summary(all_reprojection),
            "pck_3d_20mm": float(np.mean(np.asarray(all_absolute) <= 0.020)) if all_absolute else None,
            "pck_3d_50mm": float(np.mean(np.asarray(all_absolute) <= 0.050)) if all_absolute else None,
            "pck_2d_20px": float(np.mean(np.asarray(all_reprojection) <= 20.0)) if all_reprojection else None,
            "pck_2d_40px": float(np.mean(np.asarray(all_reprojection) <= 40.0)) if all_reprojection else None,
        },
        "by_side": {
            side: {
                "absolute_joint_error_mm": finite_summary(absolute_errors[side], 1000.0),
                "root_relative_joint_error_mm": finite_summary(root_relative_errors[side], 1000.0),
                "wrist_error_mm": finite_summary(wrist_errors[side], 1000.0),
                "signed_camera_axis_error_mm": {
                    axis: signed_error_summary(axis_errors[side][axis], 1000.0) for axis in ("x", "y", "z")
                },
                "signed_wrist_camera_axis_error_mm": {
                    axis: signed_error_summary(wrist_axis_errors[side][axis], 1000.0) for axis in ("x", "y", "z")
                },
                "reprojection_error_px_rectified_512": finite_summary(reprojection_errors[side]),
            }
            for side in ("left", "right")
        },
        "per_frame": per_frame,
    }


def discover_interval_states(run_root: Path, case_id: str) -> list[tuple[str, Path]]:
    root = run_root / "measurements" / "mano_interval_correction"
    if not root.exists():
        return []
    found: list[tuple[str, Path]] = []
    for path in sorted(root.glob(f"*/{case_id}/v18_joint_mano_interval_trajectory_state.json")):
        label = "p18b_canonical" if "surface_hypothesis_metric_mano" in path.as_posix() else "p18_raw"
        found.append((label, path))
    return found


def compare_hand_reports(hand_reports: dict[str, Any]) -> dict[str, Any]:
    baseline = hand_reports.get("hawor_baseline", {}).get("all_hands", {})
    metric_paths = {
        "absolute_mpjpe_mm": ("absolute_joint_error_mm", "mean"),
        "root_relative_mpjpe_mm": ("root_relative_joint_error_mm", "mean"),
        "wrist_error_mm": ("wrist_error_mm", "mean"),
        "rectified_reprojection_error_px": ("reprojection_error_px_rectified_512", "mean"),
        "pck_3d_20mm": ("pck_3d_20mm",),
        "pck_3d_50mm": ("pck_3d_50mm",),
        "pck_2d_20px_rectified": ("pck_2d_20px",),
        "pck_2d_40px_rectified": ("pck_2d_40px",),
    }

    def lookup(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
        value: Any = payload
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    result: dict[str, Any] = {}
    for label, report in hand_reports.items():
        if label == "hawor_baseline" or report.get("status") != "scored":
            continue
        current = report.get("all_hands", {})
        metrics: dict[str, Any] = {}
        for metric, path in metric_paths.items():
            baseline_value = lookup(baseline, path)
            current_value = lookup(current, path)
            metrics[metric] = {
                "hawor_baseline": baseline_value,
                "method": current_value,
                "method_minus_hawor": current_value - baseline_value if current_value is not None and baseline_value is not None else None,
                "lower_is_better": not metric.startswith("pck_"),
            }
        result[label] = {
            "coverage_equal_to_hawor": report.get("coverage") == hand_reports.get("hawor_baseline", {}).get("coverage"),
            "metrics": metrics,
        }
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    gt_dir = args.ground_truth_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(gt_dir / "BENCHMARK_MANIFEST.json")
    case_id = str(manifest["case_id"])
    object_id = resolve_object_id(run_root, args.object_id)
    input_video = Path(manifest["prediction_video"]["path"])
    runtime_input_contract_path = run_root / "input" / "runtime_input_contract.json"
    runtime_input_contract = load_json(runtime_input_contract_path) if runtime_input_contract_path.exists() else {}
    runtime_input_video = Path(runtime_input_contract["input_video"]) if runtime_input_contract.get("input_video") else None
    benchmark_video_hash = str(manifest["prediction_video"]["sha256"])
    runtime_video_hash = sha256_file(runtime_input_video) if runtime_input_video is not None and runtime_input_video.exists() else None
    input_identity = {
        "status": "content_identical" if runtime_video_hash == benchmark_video_hash else "content_mismatch_or_missing",
        "benchmark_adapter_video": str(input_video),
        "benchmark_adapter_video_sha256": benchmark_video_hash,
        "runtime_input_video": str(runtime_input_video) if runtime_input_video is not None else None,
        "runtime_input_video_sha256": runtime_video_hash,
        "paths_may_differ_after_coordinate_contract_correction": True,
        "byte_content_identical": runtime_video_hash == benchmark_video_hash,
    }
    mask_report = evaluate_masks(run_root, input_video, gt_dir, object_id, output_dir, args.boundary_tolerance_px)

    hawor_path = run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz"
    hand_reports: dict[str, Any] = {}
    if hawor_path.exists():
        hawor_joints, camera_payload = load_hawor_joint_map(hawor_path)
        cameras = camera_map(camera_payload)
        gt_hand = load_json(gt_dir / "hand_pose_gt.json")
        gt_camera = load_json(gt_dir / "camera_pose_gt.json")
        hand_reports["hawor_baseline"] = evaluate_hand_joint_map(
            "hawor_baseline",
            hawor_joints,
            cameras,
            gt_hand,
            gt_camera,
            args.min_num_views,
            args.max_gt_reprojection_px_rectified_512,
        )
        for label, path in discover_interval_states(run_root, case_id):
            hand_reports[label] = evaluate_hand_joint_map(
                label,
                load_interval_joint_map(path),
                cameras,
                gt_hand,
                gt_camera,
                args.min_num_views,
                args.max_gt_reprojection_px_rectified_512,
            )
            hand_reports[label]["source_state"] = str(path)
        camera_report = evaluate_camera_trajectory(camera_payload, gt_camera)
    else:
        hand_reports["hawor_baseline"] = {"status": "not_evaluated_missing_hawor_npz", "path": str(hawor_path)}
        camera_report = {"status": "not_evaluated_missing_hawor_npz", "path": str(hawor_path)}

    availability = load_json(gt_dir / "GROUND_TRUTH_AVAILABILITY.json")
    calibration_report = audit_calibration_provenance(
        run_root,
        input_video,
        gt_camera if hawor_path.exists() else load_json(gt_dir / "camera_pose_gt.json"),
    )
    report = {
        "status": "partial_ground_truth_evaluation_complete",
        "method": "evaluate_egoexo4d_v19_partial_gt",
        "case_id": case_id,
        "run_root": str(run_root),
        "ground_truth_dir": str(gt_dir),
        "ground_truth_status": availability["status"],
        "input_identity": input_identity,
        "calibration_provenance": calibration_report,
        "object_visible_mask": mask_report,
        "hand_pose": hand_reports,
        "hand_pose_method_comparison": compare_hand_reports(hand_reports),
        "camera_trajectory": camera_report,
        "unsupported_metric_families": {
            "object_canonical_geometry": "not_evaluated: released GT has no CAD/metric mesh",
            "object_6dof_pose": "not_evaluated: released GT has no per-frame object SE(3)",
            "contact_patch": "not_evaluated: action narration is not metric contact GT",
            "nonpenetration": "not_evaluated: no signed object geometry or GT MANO surface",
            "dense_depth": "not_evaluated: no released dense metric depth in this NAS shard",
        },
        "interpretation_policy": [
            "Object-mask IoU evaluates P06/P07 visible segmentation only; it is not object-pose or completed-mesh accuracy.",
            "Hand MPJPE evaluates named joints after the documented fixed camera-basis adapter; rectified-view reprojection is not raw-RGB pixel accuracy or MANO surface/contact accuracy.",
            "Camera SE(3) alignment preserves the predicted metric scale; Sim(3) is reported only as a scale/gauge diagnostic.",
            "The released 512x512 rectified intrinsics are never relabeled as raw 448/960 RGB intrinsics.",
            "No scalar overall physical score is emitted because several required physical GT families are absent.",
        ],
    }
    report_path = output_dir / "egoexo4d_partial_gt_evaluation.json"
    write_json(report_path, report)
    claims_path = output_dir / "SUPPORTED_AND_UNSUPPORTED_CLAIMS.json"
    claims_report = {
        "status": "partial_gt_claim_scope_audited",
        "evaluation_report": str(report_path),
        "supported_quantitative_evaluations": {
            "raw_view_sparse_visible_object_mask": {
                "evaluated": mask_report.get("status") == "scored",
                "gt_frame_count": mask_report.get("gt_annotated_frame_count"),
                "scope": "P06/P07 visible segmentation only",
            },
            "named_hand_joints": {
                "evaluated_methods": [label for label, value in hand_reports.items() if value.get("status") == "scored"],
                "matched_joint_count": hand_reports.get("hawor_baseline", {}).get("coverage", {}).get("matched_joint_count"),
                "scope": "3D annotation-camera error and rectified-512 reprojection after the fixed official camera-basis adapter",
            },
            "camera_trajectory": {
                "evaluated": camera_report.get("status") == "scored",
                "common_frame_count": camera_report.get("common_frame_count"),
                "scope": "SE3/Sim3-aligned ATE diagnostics and gauge-invariant RPE",
            },
        },
        "unsupported_quantitative_evaluations": report["unsupported_metric_families"],
        "forbidden_interpretations": [
            "Do not interpret visible-mask IoU as object 6DoF or completed-geometry accuracy.",
            "Do not interpret rectified-512 hand reprojection as raw-RGB pixel accuracy.",
            "Do not interpret action narration, box/mask overlap, or agent likely_contact as metric contact GT.",
            "Do not interpret zero correction or zero count on a non-watertight mesh as nonpenetration proof.",
            "Do not emit an overall physical-accuracy scalar while object pose/geometry/contact/nonpenetration GT is absent.",
        ],
    }
    write_json(claims_path, claims_report)
    summary = {
        "status": report["status"],
        "report": str(report_path),
        "claims_report": str(claims_path),
        "object_mask_mean_iou": mask_report["aggregate"]["iou"]["mean"],
        "hand_absolute_mpjpe_mm": hand_reports.get("hawor_baseline", {}).get("all_hands", {}).get("absolute_joint_error_mm", {}).get("mean"),
        "hand_root_relative_mpjpe_mm": hand_reports.get("hawor_baseline", {}).get("all_hands", {}).get("root_relative_joint_error_mm", {}).get("mean"),
        "camera_se3_ate_rmse_mm": camera_report.get("alignments", {}).get("se3_metric", {}).get("ate_center_rmse_mm"),
        "camera_sim3_ate_rmse_mm": camera_report.get("alignments", {}).get("sim3_diagnostic", {}).get("ate_center_rmse_mm"),
        "camera_sim3_scale": camera_report.get("alignments", {}).get("sim3_diagnostic", {}).get("scale"),
        "camera_orientation_gauge_mean_deg": camera_report.get("orientation_gauge_alignment", {}).get("orientation_error_deg", {}).get("mean"),
        "camera_rpe_translation_delta_1_mean_mm": camera_report.get("relative_pose", {}).get("delta_1_frames", {}).get("local_camera_translation_error_mm", {}).get("mean"),
        "camera_rpe_rotation_delta_1_mean_deg": camera_report.get("relative_pose", {}).get("delta_1_frames", {}).get("relative_rotation_error_deg", {}).get("mean"),
        "calibration_status": calibration_report["status"],
        "full_physical_gt_available": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--boundary-tolerance-px", type=int, default=5)
    parser.add_argument("--min-num-views", type=int, default=2)
    parser.add_argument("--max-gt-reprojection-px-rectified-512", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
