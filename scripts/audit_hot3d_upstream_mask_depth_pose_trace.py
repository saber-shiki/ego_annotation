#!/usr/bin/env python3
"""Trace HOT3D pose anomalies upstream from correct 2D masks.

This is a read-only post-processing audit.  It compares:

    owned RGB mask -> P03 UniDepth camera-z -> P09 accepted components
    -> P14 pairwise observed-surface ICP -> pre-temporal measurement
    -> final P14 translation regularization

An independent comparison reuses the production MANO-subtracted LK + calibrated
PnP implementation.  PnP is reported as supporting evidence only when the
original production thresholds pass.  No GT, CAD, generated SAM3D/TRELLIS
geometry, or finalized artifact is modified.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import build_v19_visible_geometry_from_sam2_depth as visible_geometry
import fit_v18_compact_rigid_object_pose as pose_fit


CASE_ORDER = ("milk", "soup", "mug", "bbq", "spatula")
FOCUS_FRAMES = {
    "milk": {92: "legacy-anchor control"},
    "soup": {87: "suspected upstream depth transition", 146: "stable positive control"},
    "mug": {
        33: "suspected upstream depth conflict",
        118: "suspected upstream depth conflict",
        143: "low-RGB-consensus uncertainty control",
    },
    "bbq": {58: "downstream-stage discrepancy control"},
    "spatula": {
        10: "stable positive control",
        37: "partial-surface mixed-depth control",
        117: "suspected severe upstream depth conflict",
    },
}
STRICT_PNP_THRESHOLDS = {
    "minimum_tracked_points": 50,
    "minimum_pnp_inliers": 40,
    "minimum_pnp_inlier_fraction": 0.50,
    "maximum_reprojection_median_px": 2.5,
    "maximum_reprojection_p95_px": 4.0,
    "minimum_positive_camera_depth_fraction": 0.98,
}
SCHEMA = "hot3d_upstream_mask_depth_pose_trace_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--stage-ab-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diagnostic-code-commit", required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path | str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"missing {description}: {resolved}")
    return resolved


def numeric_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    median = float(np.median(array))
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": median,
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "mad": float(np.median(np.abs(array - median))),
    }


def image_at_size(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read RGB: {path}")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def mask_at_size(path: Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {path}")
    if mask.shape != (height, width):
        mask = cv2.resize(
            (mask > 0).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
    return mask > 0


def frame_object(frame: dict[str, Any]) -> dict[str, Any]:
    objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []
    if len(objects) != 1 or not isinstance(objects[0], dict):
        raise RuntimeError(f"frame {frame.get('frame_idx')} does not have exactly one object")
    return objects[0]


def camera_motion(
    first: dict[str, Any], second: dict[str, Any]
) -> tuple[float, float]:
    first_transform = np.asarray(first["camera"]["T_world_camera_metric"], dtype=np.float64)
    second_transform = np.asarray(second["camera"]["T_world_camera_metric"], dtype=np.float64)
    translation_m = float(np.linalg.norm(second_transform[:3, 3] - first_transform[:3, 3]))
    rotation_deg = float(
        np.degrees(
            Rotation.from_matrix(
                second_transform[:3, :3] @ first_transform[:3, :3].T
            ).magnitude()
        )
    )
    return translation_m, rotation_deg


def p09_parameters(report: dict[str, Any]) -> dict[str, Any]:
    parameters = report.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("visible-geometry report lacks parameters")
    return parameters


def reconstruct_accepted_depth(
    obj: dict[str, Any],
    depth: np.ndarray,
    confidence: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    geom = obj.get("visible_geometry_candidate")
    if not isinstance(geom, dict):
        raise RuntimeError("object has no accepted visible_geometry_candidate")
    height, width = depth.shape
    owned = mask_at_size(require_file(geom["mask_path"], "owned mask"), width, height)
    valid = (
        owned
        & np.isfinite(depth)
        & (depth >= float(parameters["min_depth_m"]))
        & (depth <= float(parameters["max_depth_m"]))
    )
    accepted, summary = visible_geometry.robust_first_surface_depth_ownership(
        valid,
        depth,
        enabled=bool(parameters["robust_first_surface_depth_ownership"]),
        mad_sigma=float(parameters["first_surface_mad_sigma"]),
        min_half_width_m=float(parameters["first_surface_min_half_width_m"]),
        min_retained_fraction=float(parameters["first_surface_min_retained_fraction"]),
        fail_raw_to_robust_extent_ratio=float(
            parameters["first_surface_fail_raw_to_robust_extent_ratio"]
        ),
        intrinsics=np.asarray(geom["intrinsics_fx_fy_cx_cy"], dtype=np.float64),
        confidence=confidence,
        confidence_seed_percentile=float(
            parameters["first_surface_confidence_seed_percentile"]
        ),
        local_depth_step_max_m=float(parameters["first_surface_local_depth_step_max_m"]),
        max_removed_distance_inside_mask_px=float(
            parameters["first_surface_max_removed_distance_inside_mask_px"]
        ),
        max_confidence_flagged_interior_fraction=float(
            parameters["first_surface_max_confidence_flagged_interior_fraction"]
        ),
        min_interior_confidence_flagged_fraction=float(
            parameters["first_surface_min_interior_confidence_flagged_fraction"]
        ),
        max_unexplained_interior_pixels=int(
            parameters["first_surface_max_unexplained_interior_pixels"]
        ),
        min_component_pixels=int(parameters["first_surface_min_component_pixels"]),
        max_small_component_fraction=float(
            parameters["first_surface_max_small_component_fraction"]
        ),
    )
    stored = geom.get("first_surface_depth_ownership")
    if not isinstance(stored, dict):
        raise RuntimeError("visible geometry lacks stored depth-ownership summary")
    checks = {
        "input_count": int(valid.sum()) == int(stored["input_valid_depth_pixels"]),
        "accepted_count": int(accepted.sum()) == int(stored["retained_depth_pixels"]),
        "state": summary.get("state") == stored.get("state"),
        "extent": np.array_equal(
            np.asarray(summary["robust_backprojected_extent_m"]),
            np.asarray(stored["robust_backprojected_extent_m"]),
        ),
        "median": float(np.median(depth[accepted])) == float(geom["depth_median_m"]),
        "p05": float(np.percentile(depth[accepted], 5)) == float(geom["depth_p05_m"]),
        "p95": float(np.percentile(depth[accepted], 95)) == float(geom["depth_p95_m"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"P09 accepted-raster reconstruction mismatch: {checks}")
    summary["exact_reconstruction_checks"] = checks
    return owned, accepted, summary


def compact_components(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in summary.get("component_rows", []):
        if not isinstance(component, dict):
            continue
        rows.append(
            {
                key: component.get(key)
                for key in (
                    "label",
                    "state",
                    "pixels",
                    "bbox_xywh",
                    "median_depth_m",
                    "median_absolute_deviation_m",
                    "retained_pixels",
                    "retained_fraction",
                    "accepted_depth_summary_m",
                )
            }
        )
    return rows


def pose_from_row(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(
        row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64
    )
    translation = np.asarray(row["translation_world_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise RuntimeError("malformed P14 pose")
    return rotation, translation


def relative_pose(
    source: tuple[np.ndarray, np.ndarray], target: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    source_rotation, source_translation = source
    target_rotation, target_translation = target
    rotation = target_rotation @ source_rotation.T
    translation = target_translation - rotation @ source_translation
    return rotation, translation


def compare_pose_to_pnp(
    stage_pose: tuple[np.ndarray, np.ndarray],
    pnp_pose: tuple[np.ndarray, np.ndarray],
    source_centroid: np.ndarray,
) -> dict[str, Any]:
    stage_rotation, stage_translation = stage_pose
    pnp_rotation, pnp_translation = pnp_pose
    stage_mapped = pose_fit.apply_pose(
        source_centroid[None, :], stage_rotation, stage_translation
    )[0]
    pnp_mapped = pose_fit.apply_pose(
        source_centroid[None, :], pnp_rotation, pnp_translation
    )[0]
    return {
        "rotation_difference_deg": float(
            np.degrees(
                Rotation.from_matrix(stage_rotation @ pnp_rotation.T).magnitude()
            )
        ),
        "mapped_source_centroid_difference_m": float(
            np.linalg.norm(stage_mapped - pnp_mapped)
        ),
    }


def production_pnp_pass(report: dict[str, Any]) -> bool:
    reprojection = report["reprojection_error_px"]
    return bool(
        int(report["target_track_count"])
        >= STRICT_PNP_THRESHOLDS["minimum_tracked_points"]
        and int(report["pnp_inlier_count"])
        >= STRICT_PNP_THRESHOLDS["minimum_pnp_inliers"]
        and float(report["pnp_inlier_fraction"])
        >= STRICT_PNP_THRESHOLDS["minimum_pnp_inlier_fraction"]
        and float(reprojection["median"])
        <= STRICT_PNP_THRESHOLDS["maximum_reprojection_median_px"]
        and float(reprojection["p95"])
        <= STRICT_PNP_THRESHOLDS["maximum_reprojection_p95_px"]
        and float(report["positive_camera_depth_fraction"])
        >= STRICT_PNP_THRESHOLDS["minimum_positive_camera_depth_fraction"]
    )


def draw_pnp_overlay(
    output_path: Path,
    target_frame: dict[str, Any],
    target_obj: dict[str, Any],
    projected_uv_source_plane: np.ndarray,
    depth_error_m: np.ndarray,
    probe_summary: dict[str, Any],
) -> None:
    raw_path = require_file(target_frame["raw_frame_path"], "target RGB")
    rgb = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"failed to read {raw_path}")
    height, width = rgb.shape[:2]
    source_width = int(target_frame["source_width"])
    source_height = int(target_frame["source_height"])
    sx = width / source_width
    sy = height / source_height
    uv = projected_uv_source_plane.copy()
    uv[:, 0] = (uv[:, 0] + 0.5) * sx - 0.5
    uv[:, 1] = (uv[:, 1] + 0.5) * sy - 0.5
    mask = mask_at_size(require_file(target_obj["mask_path"], "target mask"), width, height)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
    maximum = max(0.02, float(np.percentile(np.abs(depth_error_m), 95)))
    for point, error in zip(uv, depth_error_m):
        x, y = np.rint(point).astype(int)
        if not (0 <= x < width and 0 <= y < height):
            continue
        normalized = float(np.clip(error / maximum, -1.0, 1.0))
        if normalized >= 0:
            color = (int(255 * (1.0 - normalized)), int(255 * (1.0 - normalized)), 255)
        else:
            value = abs(normalized)
            color = (255, int(255 * (1.0 - value)), int(255 * (1.0 - value)))
        cv2.circle(rgb, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
    lines = [
        f"RGB LK+PnP {probe_summary['source_frame_idx']} -> {probe_summary['target_frame_idx']}",
        f"strict pass | inliers={probe_summary['pnp_inlier_count']} ({probe_summary['pnp_inlier_fraction']:.3f})",
        f"reproj med={probe_summary['pnp_reprojection_px']['median']:.2f}px p95={probe_summary['pnp_reprojection_px']['p95']:.2f}px",
        f"target P09 z - PnP z median={probe_summary['target_p09_accepted_depth_minus_rgb_pnp_predicted_z_m']['median']*1000:+.1f}mm",
        "red: target depth deeper; blue: target depth shallower",
    ]
    overlay = rgb.copy()
    cv2.rectangle(overlay, (8, 8), (width - 8, 132), (0, 0, 0), -1)
    rgb = cv2.addWeighted(overlay, 0.68, rgb, 0.32, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(
            rgb,
            line,
            (18, 31 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rgb):
        raise RuntimeError(f"failed to write {output_path}")


def run_pnp_probe(
    *,
    case_name: str,
    source_idx: int,
    target_idx: int,
    frames: dict[int, dict[str, Any]],
    objects: dict[int, dict[str, Any]],
    observations: dict[int, np.ndarray],
    depth_stack: np.ndarray,
    confidence_stack: np.ndarray,
    depth_index: dict[int, int],
    parameters: dict[str, Any],
    p14_rows: dict[int, dict[str, Any]],
    ab_rows: dict[int, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    source_points = observations[source_idx]
    target_points = observations[target_idx]
    base = {
        "case": case_name,
        "source_frame_idx": source_idx,
        "target_frame_idx": target_idx,
        "production_thresholds": STRICT_PNP_THRESHOLDS,
    }
    try:
        pnp_rotation, pnp_translation, report, observability = (
            pose_fit.estimate_optical_flow_pnp_bridge(
                source_idx=source_idx,
                target_idx=target_idx,
                frames=frames,
                objects=objects,
                observed_world=source_points,
                source_rotation=np.eye(3, dtype=np.float64),
                source_translation=np.zeros(3, dtype=np.float64),
                max_seed_points=1200,
                min_tracked_points=10,
                min_pnp_inliers=6,
                min_inlier_fraction=0.0,
                max_reprojection_median_px=1.0e9,
                max_reprojection_p95_px=1.0e9,
            )
        )
    except Exception as error:
        return {
            **base,
            "status": "pnp_solver_or_relaxed_support_failed",
            "production_strict_pass": False,
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
        }

    strict_pass = production_pnp_pass(report)
    icp_rotation, icp_translation, icp_report = (
        pose_fit.register_adjacent_observed_surfaces(
            source_points,
            target_points,
            frame_gap=abs(target_idx - source_idx),
            iterations=12,
            trim_fraction=0.65,
            max_correspondence_m=0.035,
            min_matches=30,
            max_median_residual_m=0.015,
        )
    )
    source_centroid = source_points.mean(axis=0)
    pnp_mapped_centroid = pose_fit.apply_pose(
        source_centroid[None, :], pnp_rotation, pnp_translation
    )[0]
    icp_mapped_centroid = pose_fit.apply_pose(
        source_centroid[None, :], icp_rotation, icp_translation
    )[0]

    target_geom = objects[target_idx]["visible_geometry_candidate"]
    target_depth = np.asarray(depth_stack[depth_index[target_idx]], dtype=np.float64)
    target_confidence = np.asarray(
        confidence_stack[depth_index[target_idx]], dtype=np.float64
    )
    _, target_accepted, target_ownership = reconstruct_accepted_depth(
        objects[target_idx], target_depth, target_confidence, parameters
    )
    evidence_points = np.asarray(
        report["pose_evidence_canonical_points_m"], dtype=np.float64
    )
    predicted_world = pose_fit.apply_pose(
        evidence_points, pnp_rotation, pnp_translation
    )
    target_world_camera = np.asarray(
        frames[target_idx]["camera"]["T_world_camera_metric"], dtype=np.float64
    )
    target_camera_world = np.linalg.inv(target_world_camera)
    predicted_camera = pose_fit.apply_pose(
        predicted_world,
        target_camera_world[:3, :3],
        target_camera_world[:3, 3],
    )
    intrinsics = np.asarray(target_geom["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    predicted_z = predicted_camera[:, 2]
    projected_uv = np.column_stack(
        [
            intrinsics[0] * predicted_camera[:, 0] / predicted_z + intrinsics[2],
            intrinsics[1] * predicted_camera[:, 1] / predicted_z + intrinsics[3],
        ]
    )
    height, width = target_depth.shape
    ui = np.rint(projected_uv[:, 0]).astype(int)
    vi = np.rint(projected_uv[:, 1]).astype(int)
    inside = (
        (predicted_z > 0.0)
        & (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
    )
    ui_inside = ui[inside]
    vi_inside = vi[inside]
    predicted_z_inside = predicted_z[inside]
    projected_inside = projected_uv[inside]
    raw_depth = target_depth[vi_inside, ui_inside]
    on_accepted = target_accepted[vi_inside, ui_inside]
    raw_error = raw_depth - predicted_z_inside
    accepted_error = raw_error[on_accepted]

    pnp_source_surface = pose_fit.apply_pose(
        source_points, pnp_rotation, pnp_translation
    )
    icp_source_surface = pose_fit.apply_pose(
        source_points, icp_rotation, icp_translation
    )
    pnp_to_target = cKDTree(target_points).query(pnp_source_surface, k=1)[0]
    icp_to_target = cKDTree(target_points).query(icp_source_surface, k=1)[0]

    stage_comparison: dict[str, Any] = {}
    if source_idx in p14_rows and target_idx in p14_rows:
        final_relative = relative_pose(
            pose_from_row(p14_rows[source_idx]), pose_from_row(p14_rows[target_idx])
        )
        stage_comparison["p14_final_regularized"] = compare_pose_to_pnp(
            final_relative,
            (pnp_rotation, pnp_translation),
            source_centroid,
        )
        source_ab = ab_rows.get(source_idx, {})
        target_ab = ab_rows.get(target_idx, {})
        if (
            source_ab.get("pairwise_chain_rotation_world_from_canonical_matrix")
            is not None
            and target_ab.get("pairwise_chain_rotation_world_from_canonical_matrix")
            is not None
        ):
            pairwise_relative = relative_pose(
                (
                    np.asarray(
                        source_ab["pairwise_chain_rotation_world_from_canonical_matrix"],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        source_ab["pairwise_chain_translation_world_m"],
                        dtype=np.float64,
                    ),
                ),
                (
                    np.asarray(
                        target_ab["pairwise_chain_rotation_world_from_canonical_matrix"],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        target_ab["pairwise_chain_translation_world_m"],
                        dtype=np.float64,
                    ),
                ),
            )
            stage_comparison["pairwise_chain"] = compare_pose_to_pnp(
                pairwise_relative,
                (pnp_rotation, pnp_translation),
                source_centroid,
            )
        if (
            source_ab.get("p14_measurement_translation_world_m") is not None
            and target_ab.get("p14_measurement_translation_world_m") is not None
        ):
            source_final_rotation = pose_from_row(p14_rows[source_idx])[0]
            target_final_rotation = pose_from_row(p14_rows[target_idx])[0]
            measurement_relative = relative_pose(
                (
                    source_final_rotation,
                    np.asarray(
                        source_ab["p14_measurement_translation_world_m"],
                        dtype=np.float64,
                    ),
                ),
                (
                    target_final_rotation,
                    np.asarray(
                        target_ab["p14_measurement_translation_world_m"],
                        dtype=np.float64,
                    ),
                ),
            )
            stage_comparison["pre_temporal_measurement"] = compare_pose_to_pnp(
                measurement_relative,
                (pnp_rotation, pnp_translation),
                source_centroid,
            )

    compact_report = {
        key: report.get(key)
        for key in (
            "method",
            "source_frame_idx",
            "target_frame_idx",
            "direction",
            "seed",
            "flow_trace",
            "target_camera_raster_contract",
            "target_track_count",
            "pnp_inlier_count",
            "pnp_inlier_fraction",
            "reprojection_error_px",
            "positive_camera_depth_fraction",
            "rotation_observability",
        )
    }
    result = {
        **base,
        "status": "pnp_solution_available",
        "production_strict_pass": strict_pass,
        "pnp": compact_report,
        "pnp_rotation_observability": observability,
        "rgb_pnp_rotation_deg": pose_fit.rotation_angle_deg(pnp_rotation),
        "rgb_pnp_source_centroid_displacement_m": float(
            np.linalg.norm(pnp_mapped_centroid - source_centroid)
        ),
        "depth_icp": icp_report,
        "depth_icp_source_centroid_displacement_m": float(
            np.linalg.norm(icp_mapped_centroid - source_centroid)
        ),
        "pnp_vs_depth_icp_rotation_deg": float(
            np.degrees(
                Rotation.from_matrix(pnp_rotation @ icp_rotation.T).magnitude()
            )
        ),
        "pnp_vs_depth_icp_mapped_source_centroid_m": float(
            np.linalg.norm(pnp_mapped_centroid - icp_mapped_centroid)
        ),
        "pnp_to_target_depth_nearest_m": numeric_summary(pnp_to_target),
        "icp_to_target_depth_nearest_m": numeric_summary(icp_to_target),
        "pnp_projected_point_count": int(len(raw_error)),
        "pnp_projected_on_p09_accepted_count": int(on_accepted.sum()),
        "target_raw_depth_minus_rgb_pnp_predicted_z_m": numeric_summary(raw_error),
        "target_p09_accepted_depth_minus_rgb_pnp_predicted_z_m": numeric_summary(
            accepted_error
        ),
        "target_p09_depth_median_m": float(target_geom["depth_median_m"]),
        "target_p09_components": compact_components(target_ownership),
        "saved_stage_vs_rgb_pnp": stage_comparison,
    }
    if strict_pass:
        overlay_path = (
            output_dir
            / f"{case_name}_f{source_idx:03d}_to_f{target_idx:03d}_strict_rgb_pnp_depth_error.jpg"
        )
        draw_pnp_overlay(
            overlay_path,
            frames[target_idx],
            objects[target_idx],
            projected_inside[on_accepted],
            accepted_error,
            {
                "source_frame_idx": source_idx,
                "target_frame_idx": target_idx,
                "pnp_inlier_count": int(report["pnp_inlier_count"]),
                "pnp_inlier_fraction": float(report["pnp_inlier_fraction"]),
                "pnp_reprojection_px": report["reprojection_error_px"],
                "target_p09_accepted_depth_minus_rgb_pnp_predicted_z_m": numeric_summary(
                    accepted_error
                ),
            },
        )
        result["strict_pnp_depth_error_overlay"] = str(overlay_path)
        result["strict_pnp_depth_error_overlay_sha256"] = sha256_file(overlay_path)
    else:
        result["strict_pnp_depth_error_overlay"] = None
    return result


def static_background_probe(
    *,
    source_idx: int,
    target_idx: int,
    frames: dict[int, dict[str, Any]],
    objects: dict[int, dict[str, Any]],
    depth_stack: np.ndarray,
    depth_index: dict[int, int],
    intrinsics_stack: np.ndarray,
    stride: int = 8,
) -> dict[str, Any]:
    source_depth = np.asarray(depth_stack[depth_index[source_idx]], dtype=np.float64)
    target_depth = np.asarray(depth_stack[depth_index[target_idx]], dtype=np.float64)
    height, width = source_depth.shape
    source_rgb = image_at_size(
        require_file(frames[source_idx]["raw_frame_path"], "source RGB"), width, height
    )
    target_rgb = image_at_size(
        require_file(frames[target_idx]["raw_frame_path"], "target RGB"), width, height
    )
    source_mask = mask_at_size(
        require_file(objects[source_idx]["mask_path"], "source owned mask"), width, height
    )
    target_mask = mask_at_size(
        require_file(objects[target_idx]["mask_path"], "target owned mask"), width, height
    )
    kernel = np.ones((121, 121), dtype=np.uint8)
    source_exclusion = cv2.dilate(source_mask.astype(np.uint8), kernel) > 0
    target_exclusion = cv2.dilate(target_mask.astype(np.uint8), kernel) > 0

    yy, xx = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    u = xx.reshape(-1)
    v = yy.reshape(-1)
    z = source_depth[v, u]
    valid = (
        np.isfinite(z)
        & (z > 0.15)
        & (z < 3.0)
        & ~source_exclusion[v, u]
    )
    u, v, z = u[valid], v[valid], z[valid]
    source_intrinsics = np.asarray(
        intrinsics_stack[depth_index[source_idx]], dtype=np.float64
    )
    target_intrinsics = np.asarray(
        intrinsics_stack[depth_index[target_idx]], dtype=np.float64
    )
    x = (u - source_intrinsics[2]) / source_intrinsics[0] * z
    y = (v - source_intrinsics[3]) / source_intrinsics[1] * z
    source_camera = np.column_stack([x, y, z, np.ones_like(z)])
    source_world_camera = np.asarray(
        frames[source_idx]["camera"]["T_world_camera_metric"], dtype=np.float64
    )
    target_world_camera = np.asarray(
        frames[target_idx]["camera"]["T_world_camera_metric"], dtype=np.float64
    )
    world = source_camera @ source_world_camera.T
    target_camera = world @ np.linalg.inv(target_world_camera).T
    predicted_z = target_camera[:, 2]
    projected_u = (
        target_intrinsics[0] * target_camera[:, 0] / predicted_z
        + target_intrinsics[2]
    )
    projected_v = (
        target_intrinsics[1] * target_camera[:, 1] / predicted_z
        + target_intrinsics[3]
    )
    ui = np.rint(projected_u).astype(int)
    vi = np.rint(projected_v).astype(int)
    inside = (
        (predicted_z > 0.15)
        & (ui >= 2)
        & (ui < width - 2)
        & (vi >= 2)
        & (vi < height - 2)
    )
    u, v = u[inside], v[inside]
    ui, vi, predicted_z = ui[inside], vi[inside], predicted_z[inside]
    keep = ~target_exclusion[vi, ui]
    u, v = u[keep], v[keep]
    ui, vi, predicted_z = ui[keep], vi[keep], predicted_z[keep]
    current_z = target_depth[vi, ui]
    color_difference = np.mean(
        np.abs(
            source_rgb[v, u].astype(np.float32)
            - target_rgb[vi, ui].astype(np.float32)
        ),
        axis=1,
    )
    valid = (
        np.isfinite(current_z)
        & (current_z > 0.15)
        & (current_z < 3.0)
        & (color_difference < 18.0)
    )
    current_z = current_z[valid]
    predicted_z = predicted_z[valid]
    color_difference = color_difference[valid]
    if len(current_z) < 100:
        return {
            "source_frame_idx": source_idx,
            "target_frame_idx": target_idx,
            "status": "too_few_static_background_points",
            "point_count": int(len(current_z)),
        }
    error = current_z - predicted_z
    median = float(np.median(error))
    mad = float(np.median(np.abs(error - median)))
    inlier = np.abs(error - median) <= max(0.015, 4 * 1.4826 * mad)
    return {
        "source_frame_idx": source_idx,
        "target_frame_idx": target_idx,
        "status": "diagnostic_only",
        "point_count_before_robust": int(len(error)),
        "inlier_fraction": float(np.mean(inlier)),
        "target_depth_minus_reprojected_source_depth_m": numeric_summary(error[inlier]),
        "target_to_reprojected_source_depth_ratio": numeric_summary(
            (current_z / predicted_z)[inlier]
        ),
        "rgb_absolute_difference": numeric_summary(color_difference[inlier]),
        "claim_scope": (
            "Static-background diagnostic after camera-motion reprojection and RGB-consistency filtering; "
            "not object-pose evidence."
        ),
    }


def recorded_extent_ratio(geom: dict[str, Any]) -> dict[str, Any]:
    """Read the saved P09 extent schema without upgrading legacy provenance."""
    if geom.get("extent_ratio_to_population_diag") is not None:
        axis = np.asarray(
            geom.get("extent_ratio_to_population_sorted_axis"), dtype=np.float64
        ).reshape(-1)
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise RuntimeError("malformed modern population-relative extent ratio")
        return {
            "extent_reference_mode": "orientation_invariant_visible_population_median",
            "recorded_extent_ratio_diag": float(
                geom["extent_ratio_to_population_diag"]
            ),
            "recorded_extent_ratio_axis_max": float(np.max(axis)),
            "modern_population_relative_schema": True,
        }
    if geom.get("extent_ratio_to_anchor_diag") is not None:
        axis = np.asarray(
            geom.get("extent_ratio_to_anchor_axis"), dtype=np.float64
        ).reshape(-1)
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise RuntimeError("malformed legacy anchor-relative extent ratio")
        return {
            "extent_reference_mode": "selected_anchor_legacy_schema",
            "recorded_extent_ratio_diag": float(geom["extent_ratio_to_anchor_diag"]),
            "recorded_extent_ratio_axis_max": float(np.max(axis)),
            "modern_population_relative_schema": False,
        }
    raise RuntimeError("visible geometry has no recognized saved extent-ratio schema")


def temporal_frame_row(
    frame_idx: int,
    frames: dict[int, dict[str, Any]],
    depth_stack: np.ndarray,
    confidence_stack: np.ndarray,
    depth_index: dict[int, int],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    frame = frames[frame_idx]
    obj = frame_object(frame)
    geom = obj.get("visible_geometry_candidate")
    bbox = obj.get("bbox_xyxy")
    row: dict[str, Any] = {
        "frame_idx": frame_idx,
        "object_status": obj.get("status"),
        "owned_mask_path": obj.get("mask_path"),
        "owned_mask_area_source_px": obj.get("area_px"),
        "owned_mask_bbox_source_xyxy": bbox,
        "owned_mask_center_source_px": (
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
            if isinstance(bbox, list) and len(bbox) >= 4
            else None
        ),
    }
    if frame_idx > min(frames):
        translation_m, rotation_deg = camera_motion(frames[frame_idx - 1], frame)
        row["camera_step_from_previous"] = {
            "translation_m": translation_m,
            "rotation_deg": rotation_deg,
        }
    if not isinstance(geom, dict):
        row["metric_surface_available"] = False
        return row
    depth = np.asarray(depth_stack[depth_index[frame_idx]], dtype=np.float64)
    confidence = np.asarray(
        confidence_stack[depth_index[frame_idx]], dtype=np.float64
    )
    _, accepted, ownership = reconstruct_accepted_depth(
        obj, depth, confidence, parameters
    )
    camera_points = np.asarray(geom["camera_vertices_sample_m"], dtype=np.float64)
    extent = recorded_extent_ratio(geom)
    row.update(
        {
            "metric_surface_available": True,
            "p03_source_camera_z_m": {
                "median": float(geom["depth_median_m"]),
                "p05": float(geom["depth_p05_m"]),
                "p95": float(geom["depth_p95_m"]),
            },
            "p09_accepted_depth_pixel_count": int(accepted.sum()),
            "p09_sampled_surfel_count": int(geom["vertex_count"]),
            "camera_surface_centroid_m": camera_points.mean(axis=0).tolist(),
            "world_surface_centroid_m": geom["centroid_world_m"],
            "world_surface_extent_m": geom["world_extent_m"],
            **extent,
            "rigid_pose_observation_eligible": geom[
                "rigid_pose_observation_eligible"
            ],
            "rigid_pose_observation_reason": geom[
                "rigid_pose_observation_reason"
            ],
            "p09_component_count": ownership["component_count"],
            "p09_components": compact_components(ownership),
            "p09_exact_accepted_raster_reconstructed": True,
        }
    )
    return row


def build_target_depth_sheet(
    *,
    case_name: str,
    target_idx: int,
    frames: dict[int, dict[str, Any]],
    depth_stack: np.ndarray,
    confidence_stack: np.ndarray,
    depth_index: dict[int, int],
    parameters: dict[str, Any],
    output_path: Path,
) -> None:
    frame_ids = list(range(max(min(frames), target_idx - 2), min(max(frames), target_idx + 2) + 1))
    review_width = int(frames[target_idx]["manifest_width"])
    review_height = int(frames[target_idx]["manifest_height"])
    row_data: list[dict[str, Any]] = []
    union_mask = np.zeros((review_height, review_width), dtype=bool)
    accepted_values: list[np.ndarray] = []
    for frame_idx in frame_ids:
        frame = frames[frame_idx]
        obj = frame_object(frame)
        rgb = image_at_size(
            require_file(frame["raw_frame_path"], "review RGB"),
            review_width,
            review_height,
        )
        mask_path = Path(str(obj.get("mask_path") or ""))
        mask = (
            mask_at_size(mask_path, review_width, review_height)
            if mask_path.is_file()
            else np.zeros((review_height, review_width), dtype=bool)
        )
        union_mask |= mask
        geom = obj.get("visible_geometry_candidate")
        if isinstance(geom, dict):
            depth = np.asarray(depth_stack[depth_index[frame_idx]], dtype=np.float64)
            confidence = np.asarray(
                confidence_stack[depth_index[frame_idx]], dtype=np.float64
            )
            owned_depth, accepted_depth, ownership = reconstruct_accepted_depth(
                obj, depth, confidence, parameters
            )
            depth_review = cv2.resize(
                depth.astype(np.float32),
                (review_width, review_height),
                interpolation=cv2.INTER_LINEAR,
            )
            owned_review = cv2.resize(
                owned_depth.astype(np.uint8),
                (review_width, review_height),
                interpolation=cv2.INTER_NEAREST_EXACT,
            ) > 0
            accepted_review = cv2.resize(
                accepted_depth.astype(np.uint8),
                (review_width, review_height),
                interpolation=cv2.INTER_NEAREST_EXACT,
            ) > 0
            accepted_values.append(depth_review[accepted_review])
        else:
            depth_review = None
            owned_review = mask
            accepted_review = np.zeros_like(mask)
            ownership = None
        row_data.append(
            {
                "frame_idx": frame_idx,
                "rgb": cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                "mask": mask,
                "depth": depth_review,
                "owned_depth": owned_review,
                "accepted": accepted_review,
                "ownership": ownership,
                "geom": geom,
            }
        )
    if union_mask.any():
        ys, xs = np.where(union_mask)
        margin = 50
        x0, x1 = max(0, int(xs.min()) - margin), min(review_width, int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(review_height, int(ys.max()) + margin + 1)
    else:
        x0, y0, x1, y1 = 0, 0, review_width, review_height
    if accepted_values:
        combined = np.concatenate([value for value in accepted_values if len(value)])
        depth_min, depth_max = np.percentile(combined, [1, 99])
        if depth_max <= depth_min:
            depth_max = depth_min + 0.01
    else:
        depth_min, depth_max = 0.0, 1.0
    norm = Normalize(vmin=float(depth_min), vmax=float(depth_max), clip=True)
    cmap = plt.get_cmap("turbo")
    figure, axes = plt.subplots(
        len(row_data), 3, figsize=(12, 3.15 * len(row_data)), squeeze=False
    )
    for row_index, data in enumerate(row_data):
        frame_idx = data["frame_idx"]
        rgb = data["rgb"].copy()
        contours, _ = cv2.findContours(
            data["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
        axes[row_index, 0].imshow(rgb[y0:y1, x0:x1])
        axes[row_index, 0].set_title(
            f"f{frame_idx:03d} RGB + owned mask" + (" [TARGET]" if frame_idx == target_idx else "")
        )
        depth = data["depth"]
        if depth is None:
            for column in (1, 2):
                axes[row_index, column].imshow(np.zeros((y1 - y0, x1 - x0, 3)))
                axes[row_index, column].text(
                    0.5,
                    0.5,
                    "no accepted metric surface",
                    ha="center",
                    va="center",
                    transform=axes[row_index, column].transAxes,
                    color="white",
                )
        else:
            raw_rgba = cmap(norm(depth))
            raw_rgba[~data["owned_depth"]] = (0.0, 0.0, 0.0, 1.0)
            axes[row_index, 1].imshow(raw_rgba[y0:y1, x0:x1])
            axes[row_index, 1].set_title(
                f"P03 z in owned mask | med={data['geom']['depth_median_m']:.3f}m"
            )
            accepted_rgba = cmap(norm(depth))
            rejected = data["owned_depth"] & ~data["accepted"]
            accepted_rgba[~data["owned_depth"]] = (0.0, 0.0, 0.0, 1.0)
            accepted_rgba[rejected] = (1.0, 0.0, 0.0, 1.0)
            axes[row_index, 2].imshow(accepted_rgba[y0:y1, x0:x1])
            ownership = data["ownership"]
            medians = [
                component["accepted_depth_summary_m"]["median"]
                for component in ownership.get("component_rows", [])
                if isinstance(component.get("accepted_depth_summary_m"), dict)
            ]
            axes[row_index, 2].set_title(
                "P09 accepted; rejected=red | comps="
                + ",".join(f"{value:.3f}m" for value in medians)
            )
        for column in range(3):
            axes[row_index, column].axis("off")
    figure.suptitle(
        f"{case_name} f{target_idx:03d}: correct-mask upstream depth/component trace\n"
        f"common depth color range {depth_min:.3f}–{depth_max:.3f} m; review-only crop",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_case_timeline(
    *,
    case_name: str,
    frames: dict[int, dict[str, Any]],
    ab_rows: dict[int, dict[str, Any]],
    focus: dict[int, str],
    output_path: Path,
) -> None:
    frame_ids = np.asarray(sorted(frames), dtype=int)
    mask_area = np.full(len(frame_ids), np.nan)
    mask_center_x = np.full(len(frame_ids), np.nan)
    depth_median = np.full(len(frame_ids), np.nan)
    component_count = np.full(len(frame_ids), np.nan)
    axis_extent_ratio = np.full(len(frame_ids), np.nan)
    temporal_adjustment = np.full(len(frame_ids), np.nan)
    temporal_residual_delta = np.full(len(frame_ids), np.nan)
    for row_index, frame_idx in enumerate(frame_ids):
        obj = frame_object(frames[int(frame_idx)])
        bbox = obj.get("bbox_xyxy")
        if obj.get("area_px") is not None:
            mask_area[row_index] = float(obj["area_px"])
        if isinstance(bbox, list) and len(bbox) >= 4:
            mask_center_x[row_index] = (bbox[0] + bbox[2]) / 2.0
        geom = obj.get("visible_geometry_candidate")
        if isinstance(geom, dict):
            depth_median[row_index] = float(geom["depth_median_m"])
            ownership = geom.get("first_surface_depth_ownership") or {}
            component_count[row_index] = float(ownership.get("component_count", np.nan))
            axis_extent_ratio[row_index] = recorded_extent_ratio(geom)[
                "recorded_extent_ratio_axis_max"
            ]
        ab_row = ab_rows.get(int(frame_idx), {})
        if ab_row.get("measurement_to_p14_temporal_translation_regularization_m") is not None:
            temporal_adjustment[row_index] = float(
                ab_row["measurement_to_p14_temporal_translation_regularization_m"]
            )
        if ab_row.get("temporal_translation_regularization_residual_delta_m") is not None:
            temporal_residual_delta[row_index] = float(
                ab_row["temporal_translation_regularization_residual_delta_m"]
            )
    depth_curvature = np.full(len(frame_ids), np.nan)
    for index in range(1, len(frame_ids) - 1):
        if np.isfinite(depth_median[index - 1 : index + 2]).all():
            depth_curvature[index] = abs(
                depth_median[index]
                - (depth_median[index - 1] + depth_median[index + 1]) / 2.0
            )
    median_area = float(np.nanmedian(mask_area))
    figure, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(frame_ids, mask_area / median_area, label="owned mask area / median")
    axes[0].plot(
        frame_ids,
        mask_center_x / np.nanmedian(mask_center_x),
        label="mask center-x / median",
        alpha=0.7,
    )
    axes[0].set_ylabel("2D normalized")
    axes[0].legend(loc="best")
    axes[1].plot(frame_ids, depth_median, label="P03/P09 accepted depth median")
    axes[1].set_ylabel("camera-z (m)")
    twin = axes[1].twinx()
    twin.plot(frame_ids, depth_curvature * 1000.0, color="tab:red", label="depth curvature")
    twin.set_ylabel("curvature (mm)", color="tab:red")
    axes[2].plot(frame_ids, axis_extent_ratio, label="P09 saved max axis extent ratio")
    axes[2].plot(frame_ids, component_count, label="P09 component count", alpha=0.75)
    axes[2].axhline(3.25, color="black", linestyle="--", alpha=0.5, label="extent gate")
    axes[2].set_ylabel("P09 diagnostics")
    axes[2].legend(loc="best")
    axes[3].plot(
        frame_ids,
        temporal_adjustment * 1000.0,
        label="P14 temporal translation adjustment",
    )
    axes[3].plot(
        frame_ids,
        temporal_residual_delta * 1000.0,
        label="residual delta to current P09 surface",
    )
    axes[3].axhline(0.0, color="black", linewidth=0.7)
    axes[3].set_ylabel("mm")
    axes[3].set_xlabel("frame")
    axes[3].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=0.2)
        for frame_idx in focus:
            axis.axvline(frame_idx, color="magenta", alpha=0.45)
    figure.suptitle(
        f"{case_name}: mask → P03 depth → P09 components → P14 timeline\n"
        "P14 residual is measured against the current P09 surface, not ground truth",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def audit_p11_binding(
    run_root: Path, frames: dict[int, dict[str, Any]], case_name: str
) -> tuple[dict[str, Any], list[Path]]:
    """Verify that P11 copied one P09 row without changing mask/camera/surfels."""
    report_path = require_file(
        run_root
        / "experiments/sam3d_trellis_controlled/P11_dual_inputs/p11_dual_geometry_inputs_report.json",
        f"{case_name} P11 report",
    )
    report = load_json(report_path)
    evidence_path = require_file(
        report["source_evidence_report"], f"{case_name} P11 source evidence report"
    )
    evidence = load_json(evidence_path)
    selected_frame_idx = int(report["selected_frame_idx"])
    if selected_frame_idx not in frames:
        raise RuntimeError(f"{case_name}: P11 selected frame is absent from P09 annotations")
    annotation_frame = frames[selected_frame_idx]
    annotation_object = frame_object(annotation_frame)
    annotation_geometry = annotation_object.get("visible_geometry_candidate")
    if not isinstance(annotation_geometry, dict):
        raise RuntimeError(f"{case_name}: P11 selected frame has no P09 metric surface")
    selected = evidence.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError(f"{case_name}: malformed P11 source evidence selected row")
    selected_geometry = selected.get("visible_geometry_candidate")
    if not isinstance(selected_geometry, dict):
        raise RuntimeError(f"{case_name}: selected P11 evidence has no metric surface")
    conditioning = report.get("conditioning_contracts", {}).get(
        "sam3d_objects_native", {}
    )
    mask_copy = conditioning.get("mask_source_copy")
    rgb_copy = conditioning.get("image_source_copy")
    if not isinstance(mask_copy, dict) or not isinstance(rgb_copy, dict):
        raise RuntimeError(f"{case_name}: P11 lacks source-copy provenance")
    source_mask_path = require_file(mask_copy["source"], f"{case_name} P11 source mask")
    copied_mask_path = require_file(mask_copy["copy"], f"{case_name} P11 copied mask")
    source_rgb_path = require_file(rgb_copy["source"], f"{case_name} P11 source RGB")
    copied_rgb_path = require_file(rgb_copy["copy"], f"{case_name} P11 copied RGB")
    selected_mask_path = require_file(selected["mask_path"], f"{case_name} evidence mask")
    annotation_mask_path = require_file(
        annotation_geometry["mask_path"], f"{case_name} annotation mask"
    )

    mask_hash = sha256_file(source_mask_path)
    rgb_hash = sha256_file(source_rgb_path)
    camera_annotation = annotation_frame["camera"]
    camera_selected = selected["camera"]
    checks = {
        "report_selected_frame_equals_evidence": selected_frame_idx
        == int(evidence["selected_frame_idx"]),
        "selected_row_frame_equals_P09": selected_frame_idx
        == int(selected["frame_idx"])
        == int(annotation_frame["frame_idx"]),
        "selected_surface_frame_equals_P09": selected_frame_idx
        == int(selected_geometry["frame_idx"])
        == int(annotation_geometry["frame_idx"]),
        "selected_mask_path_equals_P09": selected_mask_path == annotation_mask_path,
        "P11_mask_copy_source_path_equals_P09": source_mask_path
        == annotation_mask_path,
        "P11_mask_copy_byte_identical": source_mask_path.read_bytes()
        == copied_mask_path.read_bytes(),
        "P11_mask_hash_matches_report": mask_hash
        == str(mask_copy["sha256"])
        == str(report["mask_provenance"]["selected_sha256"])
        == str(report["mask_provenance"]["visible_geometry_sha256"]),
        "P11_mask_reported_byte_identical": bool(mask_copy["byte_identical"])
        and bool(report["mask_provenance"]["byte_identical"]),
        "selected_RGB_path_equals_P09": require_file(
            selected["raw_frame_path"], f"{case_name} selected RGB"
        )
        == require_file(annotation_frame["raw_frame_path"], f"{case_name} P09 RGB"),
        "P11_RGB_copy_byte_identical": source_rgb_path.read_bytes()
        == copied_rgb_path.read_bytes(),
        "P11_RGB_hash_matches_report": rgb_hash == str(rgb_copy["sha256"]),
        "camera_T_world_camera_metric_exact": np.array_equal(
            np.asarray(camera_annotation["T_world_camera_metric"], dtype=np.float64),
            np.asarray(camera_selected["T_world_camera_metric"], dtype=np.float64),
        ),
        "camera_intrinsics_exact": np.array_equal(
            np.asarray(camera_annotation["intrinsics_fx_fy_cx_cy"], dtype=np.float64),
            np.asarray(camera_selected["intrinsics_fx_fy_cx_cy"], dtype=np.float64),
        ),
        "camera_surfels_exact": np.array_equal(
            np.asarray(annotation_geometry["camera_vertices_sample_m"], dtype=np.float64),
            np.asarray(selected_geometry["camera_vertices_sample_m"], dtype=np.float64),
        ),
        "world_surfels_exact": np.array_equal(
            np.asarray(annotation_geometry["world_vertices_sample_m"], dtype=np.float64),
            np.asarray(selected_geometry["world_vertices_sample_m"], dtype=np.float64),
        ),
        "centroid_exact": np.array_equal(
            np.asarray(annotation_geometry["centroid_world_m"], dtype=np.float64),
            np.asarray(selected_geometry["centroid_world_m"], dtype=np.float64),
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{case_name}: P11/P09 binding mismatch: {checks}")

    mesh_reconstruction = evidence.get("depth_fused_object_row", {}).get(
        "mesh_reconstruction", {}
    )
    if not isinstance(mesh_reconstruction, dict):
        raise RuntimeError(f"{case_name}: malformed P11 mesh reconstruction provenance")
    mesh_anchor_frame_idx = int(mesh_reconstruction["anchor_frame_idx"])
    mesh_anchor_centroid = np.asarray(
        mesh_reconstruction["anchor_centroid_world_m"], dtype=np.float64
    )
    anchor_checks = {
        "mesh_anchor_frame_equals_selected": mesh_anchor_frame_idx
        == selected_frame_idx,
        "mesh_anchor_centroid_equals_selected": np.array_equal(
            mesh_anchor_centroid,
            np.asarray(annotation_geometry["centroid_world_m"], dtype=np.float64),
        ),
    }
    if not all(anchor_checks.values()):
        raise RuntimeError(f"{case_name}: P11 canonical anchor mismatch: {anchor_checks}")
    atomic_binding = evidence.get("selected_anchor_atomic_binding")
    if isinstance(atomic_binding, dict):
        atomic_mode = "modern_saved_selected_anchor_atomic_binding"
        atomic_checks = {
            "validated": bool(atomic_binding.get("validated")),
            "required_same_frame": bool(atomic_binding.get("required_same_frame")),
            "selected_frame_exact": int(atomic_binding["selected_frame_idx"])
            == selected_frame_idx,
            "visible_geometry_frame_exact": int(
                atomic_binding["visible_geometry_frame_idx"]
            )
            == selected_frame_idx,
            "canonical_surface_frame_exact": int(
                atomic_binding["canonical_surface_frame_idx"]
            )
            == selected_frame_idx,
            "centroid_error_exact_zero": float(
                atomic_binding["selected_vs_canonical_centroid_error_m"]
            )
            == 0.0,
        }
        if not all(atomic_checks.values()):
            raise RuntimeError(
                f"{case_name}: saved selected-anchor atomic binding failed: {atomic_checks}"
            )
    else:
        atomic_mode = "legacy_schema_without_saved_selected_anchor_atomic_binding"
        atomic_checks = {
            "saved_modern_atomic_binding_available": False,
            "legacy_binding_replayed_or_upgraded": False,
            "selected_frame_camera_mask_surfels_centroid_and_mesh_anchor_exact": True,
        }

    paths = [
        report_path,
        evidence_path,
        source_mask_path,
        copied_mask_path,
        source_rgb_path,
        copied_rgb_path,
    ]
    canonical_surface_path = mesh_reconstruction.get("fused_point_cloud_path")
    if canonical_surface_path:
        paths.append(
            require_file(canonical_surface_path, f"{case_name} canonical observed surface")
        )
    return (
        {
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "source_evidence_report": str(evidence_path),
            "source_evidence_report_sha256": sha256_file(evidence_path),
            "selected_frame_idx": selected_frame_idx,
            "source_owned_mask": str(source_mask_path),
            "copied_owned_mask": str(copied_mask_path),
            "owned_mask_sha256": mask_hash,
            "source_RGB": str(source_rgb_path),
            "copied_RGB": str(copied_rgb_path),
            "source_RGB_sha256": rgb_hash,
            "exact_transfer_checks": checks,
            "canonical_anchor_checks": anchor_checks,
            "atomic_binding_mode": atomic_mode,
            "saved_atomic_binding_checks": atomic_checks,
            "P11_created_new_depth_or_pose_evidence": False,
        },
        paths,
    )


def selected_source_paths(
    run_root: Path,
    stage_ab_root: Path,
    case_name: str,
    annotation_path: Path,
    adapter_report_path: Path,
    source_depth_path: Path,
    active_depth_path: Path,
) -> list[Path]:
    return [
        annotation_path,
        adapter_report_path,
        source_depth_path,
        active_depth_path,
        require_file(
            run_root
            / "experiments/sam3d_trellis_controlled/P14_observed_pose_fit/v18_compact_rigid_object_pose_fit_report.json",
            f"{case_name} P14 report",
        ),
        require_file(
            stage_ab_root
            / "cases"
            / case_name
            / "P14_PAIRWISE_REGULARIZED_P15_AB_REPORT.json",
            f"{case_name} stage A/B report",
        ),
    ]


def main() -> None:
    args = parse_args()
    started = time.time()
    collection_root = args.collection_root.expanduser().resolve(strict=True)
    stage_ab_root = args.stage_ab_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    repair_validation_root = stage_ab_root.parent.resolve(strict=True)
    if not output_root.is_relative_to(repair_validation_root):
        raise RuntimeError("output root must remain under repair_validation")
    if output_root == stage_ab_root or stage_ab_root.is_relative_to(output_root):
        raise RuntimeError("output root must not contain or overwrite the finalized stage A/B")
    if output_root.exists():
        if not args.replace:
            raise RuntimeError(f"output exists; use --replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    script_path = Path(__file__).resolve()

    global_source_hashes: dict[str, dict[str, Any]] = {}
    case_reports: list[dict[str, Any]] = []
    for case_name in CASE_ORDER:
        print(f"[{case_name}] loading read-only upstream artifacts", flush=True)
        collection_link = collection_root / "runs" / case_name
        if not collection_link.is_symlink():
            raise RuntimeError(f"collection case is not a symlink: {collection_link}")
        run_root = collection_link.resolve(strict=True)
        annotation_path = require_file(
            next(run_root.rglob("annotations_v19_visible_geometry.json")),
            f"{case_name} visible annotations",
        )
        adapter_report_path = require_file(
            next(run_root.rglob("v19_visible_geometry_adapter_report.json")),
            f"{case_name} P09 adapter report",
        )
        adapter_report = load_json(adapter_report_path)
        parameters = p09_parameters(adapter_report)
        p03c_report_path = require_file(
            run_root
            / "state/calibration/depth_camera_contract/v19_depth_camera_contract_adapter_report.json",
            f"{case_name} P03c report",
        )
        p03c_report = load_json(p03c_report_path)
        source_depth_path = require_file(
            p03c_report["inputs"]["source_depth_npz"], f"{case_name} P03 source depth"
        )
        active_depth_path = require_file(
            p03c_report["outputs"]["depth_npz"], f"{case_name} active depth"
        )
        p14_path = require_file(
            run_root
            / "experiments/sam3d_trellis_controlled/P14_observed_pose_fit/v18_compact_rigid_object_pose_fit_report.json",
            f"{case_name} P14 report",
        )
        ab_report_path = require_file(
            stage_ab_root
            / "cases"
            / case_name
            / "P14_PAIRWISE_REGULARIZED_P15_AB_REPORT.json",
            f"{case_name} A/B report",
        )
        for source_path in selected_source_paths(
            run_root,
            stage_ab_root,
            case_name,
            annotation_path,
            adapter_report_path,
            source_depth_path,
            active_depth_path,
        ) + [p03c_report_path]:
            key = str(source_path)
            if key not in global_source_hashes:
                global_source_hashes[key] = {
                    "path": key,
                    "sha256": sha256_file(source_path),
                    "size_bytes": source_path.stat().st_size,
                }

        annotations = load_json(annotation_path)
        frames = {
            int(frame["frame_idx"]): frame
            for frame in annotations.get("frames", [])
            if isinstance(frame, dict)
        }
        if sorted(frames) != list(range(150)):
            raise RuntimeError(f"{case_name}: annotations do not contain 150 frames")
        objects = {frame_idx: frame_object(frame) for frame_idx, frame in frames.items()}
        p11_binding, p11_source_paths = audit_p11_binding(
            run_root, frames, case_name
        )
        for source_path in p11_source_paths:
            key = str(source_path)
            if key not in global_source_hashes:
                global_source_hashes[key] = {
                    "path": key,
                    "sha256": sha256_file(source_path),
                    "size_bytes": source_path.stat().st_size,
                }
        observations = {
            frame_idx: np.asarray(
                obj.get("visible_geometry_candidate", {}).get("world_vertices_sample_m")
                or [],
                dtype=np.float64,
            )
            for frame_idx, obj in objects.items()
            if isinstance(obj.get("visible_geometry_candidate"), dict)
        }
        p14 = load_json(p14_path)
        p14_rows = {
            int(row["frame_idx"]): row
            for row in p14.get("pose_rows", [])
            if isinstance(row, dict) and row.get("frame_idx") is not None
        }
        ab_report = load_json(ab_report_path)
        ab_rows = {
            int(row["frame_idx"]): row
            for row in ab_report.get("frame_rows", [])
            if isinstance(row, dict) and row.get("frame_idx") is not None
        }

        with np.load(source_depth_path, allow_pickle=False) as source_archive:
            source_frame_idx = np.asarray(source_archive["frame_idx"], dtype=np.int32)
            source_depth = np.asarray(source_archive["depth"])
            source_confidence = np.asarray(source_archive["confidence"])
            source_intrinsics = np.asarray(
                source_archive["intrinsics_fx_fy_cx_cy"], dtype=np.float64
            )
            source_size = np.asarray(source_archive["source_size"])
        with np.load(active_depth_path, allow_pickle=False) as active_archive:
            p03c_array_identity = {
                "frame_idx_equal": bool(
                    np.array_equal(source_frame_idx, active_archive["frame_idx"])
                ),
                "depth_equal": bool(
                    np.array_equal(source_depth, active_archive["depth"], equal_nan=True)
                ),
                "confidence_equal": bool(
                    np.array_equal(
                        source_confidence, active_archive["confidence"], equal_nan=True
                    )
                ),
                "source_size_equal": bool(
                    np.array_equal(source_size, active_archive["source_size"])
                ),
                "active_intrinsics_equal_source": bool(
                    np.array_equal(
                        source_intrinsics,
                        active_archive["intrinsics_fx_fy_cx_cy"],
                    )
                ),
            }
        if not all(p03c_array_identity.values()):
            raise RuntimeError(
                f"{case_name}: P03c was not byte-identical in active arrays: {p03c_array_identity}"
            )
        depth_index = {
            int(frame_idx): row for row, frame_idx in enumerate(source_frame_idx)
        }
        case_output = output_root / "cases" / case_name
        focus_reports: list[dict[str, Any]] = []
        for target_idx, focus_reason in FOCUS_FRAMES[case_name].items():
            if target_idx not in observations:
                raise RuntimeError(f"{case_name} target {target_idx} lacks P09 surface")
            before = max(
                (idx for idx in observations if idx < target_idx), default=None
            )
            after = min((idx for idx in observations if idx > target_idx), default=None)
            source_neighbors = [idx for idx in (before, after) if idx is not None]
            neighborhood = [
                temporal_frame_row(
                    frame_idx,
                    frames,
                    source_depth,
                    source_confidence,
                    depth_index,
                    parameters,
                )
                for frame_idx in range(max(0, target_idx - 2), min(149, target_idx + 2) + 1)
            ]
            metric_by_frame = {
                row["frame_idx"]: row
                for row in neighborhood
                if row.get("metric_surface_available") is True
            }
            target_depth_curvature_m = None
            if target_idx - 1 in metric_by_frame and target_idx + 1 in metric_by_frame:
                target_depth = metric_by_frame[target_idx]["p03_source_camera_z_m"]["median"]
                neighbor_mean = (
                    metric_by_frame[target_idx - 1]["p03_source_camera_z_m"]["median"]
                    + metric_by_frame[target_idx + 1]["p03_source_camera_z_m"]["median"]
                ) / 2.0
                target_depth_curvature_m = abs(target_depth - neighbor_mean)
            target_mask_area_curvature = None
            if target_idx - 1 in frames and target_idx + 1 in frames:
                areas = [
                    frame_object(frames[idx]).get("area_px")
                    for idx in (target_idx - 1, target_idx, target_idx + 1)
                ]
                if all(value is not None and float(value) > 0 for value in areas):
                    target_mask_area_curvature = abs(
                        math.log(float(areas[1]))
                        - (math.log(float(areas[0])) + math.log(float(areas[2]))) / 2.0
                    )
            probes = [
                run_pnp_probe(
                    case_name=case_name,
                    source_idx=source_idx,
                    target_idx=target_idx,
                    frames=frames,
                    objects=objects,
                    observations=observations,
                    depth_stack=source_depth,
                    confidence_stack=source_confidence,
                    depth_index=depth_index,
                    parameters=parameters,
                    p14_rows=p14_rows,
                    ab_rows=ab_rows,
                    output_dir=case_output / "rgb_pnp",
                )
                for source_idx in source_neighbors
            ]
            background = [
                static_background_probe(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    frames=frames,
                    objects=objects,
                    depth_stack=source_depth,
                    depth_index=depth_index,
                    intrinsics_stack=source_intrinsics,
                )
                for source_idx in source_neighbors
            ]
            sheet_path = (
                case_output
                / "depth_component_sheets"
                / f"{case_name}_f{target_idx:03d}_mask_p03_p09_neighborhood.jpg"
            )
            build_target_depth_sheet(
                case_name=case_name,
                target_idx=target_idx,
                frames=frames,
                depth_stack=source_depth,
                confidence_stack=source_confidence,
                depth_index=depth_index,
                parameters=parameters,
                output_path=sheet_path,
            )
            strict_probes = [probe for probe in probes if probe["production_strict_pass"]]
            focus_reports.append(
                {
                    "frame_idx": target_idx,
                    "focus_reason": focus_reason,
                    "nearest_accepted_surface_neighbors": source_neighbors,
                    "target_depth_temporal_curvature_m": target_depth_curvature_m,
                    "target_log_mask_area_temporal_curvature": target_mask_area_curvature,
                    "temporal_neighborhood": neighborhood,
                    "rgb_pnp_probes": probes,
                    "strict_rgb_pnp_pass_count": len(strict_probes),
                    "strict_rgb_pnp_target_depth_minus_predicted_z_m": [
                        probe[
                            "target_p09_accepted_depth_minus_rgb_pnp_predicted_z_m"
                        ]["median"]
                        for probe in strict_probes
                    ],
                    "static_background_probes": background,
                    "mask_p03_p09_neighborhood_sheet": str(sheet_path),
                    "mask_p03_p09_neighborhood_sheet_sha256": sha256_file(sheet_path),
                    "mechanical_interpretation": (
                        "Current-frame P09 surface is not ground truth. Compare strict RGB-PnP in both directions; "
                        "a failed probe remains information insufficiency."
                    ),
                }
            )
        timeline_path = case_output / f"{case_name}_upstream_trace_timeline.jpg"
        build_case_timeline(
            case_name=case_name,
            frames=frames,
            ab_rows=ab_rows,
            focus=FOCUS_FRAMES[case_name],
            output_path=timeline_path,
        )
        modern_extent_reference = isinstance(
            adapter_report.get("extent_consistency_reference"), dict
        )
        case_reports.append(
            {
                "case": case_name,
                "run_root": str(run_root),
                "inputs": {
                    "annotations": str(annotation_path),
                    "annotations_sha256": sha256_file(annotation_path),
                    "p03_source_depth": str(source_depth_path),
                    "p03_source_depth_sha256": sha256_file(source_depth_path),
                    "p03c_active_depth": str(active_depth_path),
                    "p03c_active_depth_sha256": sha256_file(active_depth_path),
                    "p03c_adapter_report": str(p03c_report_path),
                    "p03c_adapter_report_sha256": sha256_file(p03c_report_path),
                    "p09_adapter_report": str(adapter_report_path),
                    "p09_adapter_report_sha256": sha256_file(adapter_report_path),
                    "p14_report": str(p14_path),
                    "p14_report_sha256": sha256_file(p14_path),
                    "stage_ab_report": str(ab_report_path),
                    "stage_ab_report_sha256": sha256_file(ab_report_path),
                },
                "p03c_source_to_active_array_identity": p03c_array_identity,
                "p03c_depth_adapter_created_or_changed_depth_values": False,
                "p11_binding": p11_binding,
                "p09_contract": {
                    "depth_ownership_method": "per_component_mad_seed_confidence_geodesic_growth",
                    "extent_reference_mode": (
                        "orientation_invariant_visible_population_median"
                        if modern_extent_reference
                        else "selected_anchor_legacy_schema"
                    ),
                    "modern_population_relative_schema": modern_extent_reference,
                    "legacy_schema_replayed_or_upgraded": False,
                    "rigid_pose_eligibility": (
                        "orientation-invariant population extent ratios only; no temporal depth, RGB-PnP, "
                        "component-rigidity, or SE3-step consistency gate"
                        if modern_extent_reference
                        else
                        "legacy selected-anchor-relative extent ratios only; no temporal depth, RGB-PnP, "
                        "component-rigidity, or SE3-step consistency gate"
                    ),
                    "rigid_extent_ratio_max": parameters["rigid_extent_ratio_max"],
                    "rigid_extent_axis_ratio_max": parameters[
                        "rigid_extent_axis_ratio_max"
                    ],
                },
                "p14_pairwise_contract": {
                    "method": "trimmed_mutual_nearest_observed_surfel_icp",
                    "quality_gate": "minimum matches plus local trimmed median residual",
                    "maximum_median_residual_m": 0.015,
                    "minimum_matches": 30,
                    "maximum_translation_or_rotation_step_gate_inside_pairwise_registration": False,
                    "RGB_consistency_gate_inside_pairwise_registration": False,
                    "component_correspondence_gate_inside_pairwise_registration": False,
                },
                "focus_frames": focus_reports,
                "timeline": str(timeline_path),
                "timeline_sha256": sha256_file(timeline_path),
            }
        )
        del source_depth
        del source_confidence

    source_hash_rows = list(global_source_hashes.values())
    for row in source_hash_rows:
        source_path = Path(row["path"])
        row["post_audit_sha256"] = sha256_file(source_path)
        row["post_audit_size_bytes"] = source_path.stat().st_size
        row["unchanged"] = bool(
            row["post_audit_sha256"] == row["sha256"]
            and row["post_audit_size_bytes"] == row["size_bytes"]
        )
    if not all(row["unchanged"] for row in source_hash_rows):
        raise RuntimeError("source artifact changed during read-only audit")

    report = {
        "schema": SCHEMA,
        "status": "mechanical_trace_complete_visual_review_pending",
        "claim_scope": (
            "Read-only upstream trace from owned mask through saved P03/P09/P14 stages. "
            "No GT or generated hidden geometry is consumed, and current-frame P09 residual is not treated as truth."
        ),
        "collection_root": str(collection_root),
        "stage_ab_root": str(stage_ab_root),
        "output_root": str(output_root),
        "diagnostic_script": str(script_path),
        "diagnostic_script_sha256": sha256_file(script_path),
        "diagnostic_code_commit": args.diagnostic_code_commit,
        "case_count": len(case_reports),
        "focus_frame_count": sum(len(row["focus_frames"]) for row in case_reports),
        "strict_pnp_thresholds": STRICT_PNP_THRESHOLDS,
        "global_mechanical_findings": {
            "P11_and_P09_owned_masks_are_byte_identical_for_all_selected_anchors": all(
                all(case["p11_binding"]["exact_transfer_checks"].values())
                for case in case_reports
            ),
            "P11_selected_camera_surfels_centroid_and_mesh_anchor_exact_for_all_cases": all(
                all(case["p11_binding"]["canonical_anchor_checks"].values())
                for case in case_reports
            ),
            "Milk_has_modern_saved_selected_anchor_atomic_binding_schema": False,
            "Milk_legacy_P11_binding_replayed_or_upgraded": False,
            "P03c_depth_values_changed": False,
            "P03_source_archive_is_earliest_persisted_camera_z_raster": True,
            "P09_acceptance_has_temporal_depth_consistency_gate": False,
            "P09_acceptance_has_calibrated_RGB_PnP_consistency_gate": False,
            "P09_per_component_depth_ownership_enforces_cross_component_rigidity": False,
            "P14_pairwise_ICP_gate_uses_local_residual_and_match_count_only": True,
            "P14_pairwise_ICP_has_RGB_or_max_SE3_step_gate": False,
            "P14_residual_to_current_P09_surface_is_ground_truth_error": False,
        },
        "cases": case_reports,
        "source_immutability": {
            "entry_count": len(source_hash_rows),
            "mismatch_count": sum(not row["unchanged"] for row in source_hash_rows),
            "finalized_case_roots_modified": False,
            "collection_modified": False,
            "rows": source_hash_rows,
        },
        "limitations": [
            "RGB-PnP inherits source-frame metric depth; bidirectional agreement is stronger than a single direction.",
            "PnP failure is information insufficiency, not evidence that target depth is correct.",
            "A temporally smoother final pose is not automatically true; this audit only compares independent saved evidence.",
            "Partial and disconnected surfaces can yield low local ICP residual for an incorrect global rigid transform.",
            "No GT pose, GT depth, GT MANO, CAD, or generated SAM3D/TRELLIS geometry was used.",
        ],
        "elapsed_s": float(time.time() - started),
    }
    report_path = output_root / "UPSTREAM_MASK_DEPTH_POSE_TRACE_REPORT.json"
    write_json(report_path, report)
    done = {
        "schema": SCHEMA,
        "status": report["status"],
        "visual_review_pending": True,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "focus_frame_count": report["focus_frame_count"],
        "source_hash_mismatch_count": 0,
        "finalized_case_roots_modified": False,
        "collection_modified": False,
    }
    write_json(output_root / "UPSTREAM_TRACE_MECHANICAL_DONE.json", done)
    print(json.dumps(done, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
