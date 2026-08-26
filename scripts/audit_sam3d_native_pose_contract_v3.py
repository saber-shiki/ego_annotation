#!/usr/bin/env python3
"""Audit raw SAM3D local geometry against the declared native local-to-camera pose.

The audit is deliberately separate from P13.  It replays the native transform,
checks it against the exported camera-frame meshes, and evaluates raw/local,
PyTorch3D-native, OpenCV-native, and sensor-metric-native hypotheses on the
same mask plane.  It never promotes generated geometry to annotation or
collision authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from remote_run_sam3d_objects_mesh_v7 import apply_native_pose  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {label}: {path}")
    return path


def finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RuntimeError(f"{label} must be finite length {size}, got {result}")
    return result


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"not a triangle mesh: {path}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"invalid vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"invalid faces: {path}")
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"non-finite vertices: {path}")
    return loaded


def sam3d_candidate(p12: dict[str, Any]) -> dict[str, Any]:
    candidates = p12.get("candidates")
    if not isinstance(candidates, dict) or not isinstance(candidates.get("sam3d_objects"), dict):
        raise RuntimeError("P12 report has no sam3d_objects candidate")
    candidate = candidates["sam3d_objects"]
    native = candidate.get("native_outputs")
    if not isinstance(native, dict):
        raise RuntimeError("P12 SAM3D candidate lacks native_outputs")
    raw = native.get("raw_mesh")
    raw_path = Path(str(raw.get("path") if isinstance(raw, dict) else raw or ""))
    p3d = native.get("native_pose_mesh_pytorch3d_camera")
    cv = native.get("native_pose_mesh_opencv_camera")
    p3d_path = Path(str(p3d.get("path") if isinstance(p3d, dict) else p3d or ""))
    cv_path = Path(str(cv.get("path") if isinstance(cv, dict) else cv or ""))
    pose = native.get("native_pose")
    contract = native.get("native_pose_contract")
    if not isinstance(pose, dict) or not isinstance(contract, dict):
        raise RuntimeError("P12 SAM3D candidate lacks native pose/contract")
    return {
        "candidate": candidate,
        "native": native,
        "raw_mesh": require_file(raw_path, "raw SAM3D local mesh"),
        "exported_pytorch3d": require_file(p3d_path, "exported PyTorch3D native mesh"),
        "exported_opencv": require_file(cv_path, "exported OpenCV native mesh"),
        "pose": pose,
        "contract": contract,
    }


def selected_visible_evidence(evidence: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = evidence.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("evidence report has no selected anchor")
    visible = selected.get("visible_geometry_candidate")
    if not isinstance(visible, dict):
        raise RuntimeError("selected anchor lacks visible geometry")
    ownership = visible.get("first_surface_depth_ownership")
    if not isinstance(ownership, dict) or ownership.get("enabled") is not True:
        raise RuntimeError("selected anchor lacks explicit first-surface ownership")
    if ownership.get("fail_closed") is True or ownership.get("failure_reasons"):
        raise RuntimeError("selected first-surface ownership failed closed")
    return selected, visible


def mask_plane_intrinsics(visible: dict[str, Any], mask_shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    fx, fy, cx, cy = finite_vector(visible.get("intrinsics_fx_fy_cx_cy"), 4, "depth intrinsics")
    K_depth = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform = visible.get("mask_depth_transform_contract")
    if not isinstance(transform, dict) or transform.get("camera_contract_consistent") is not True:
        raise RuntimeError("visible evidence lacks a camera-consistent mask/depth transform")
    height, width = mask_shape
    declared = [int(x) for x in transform.get("mask_size_wh") or []]
    if declared != [width, height]:
        raise RuntimeError(f"mask shape {(height, width)} disagrees with contract {declared}")
    A_depth_from_mask = np.asarray(transform.get("A_depth_from_mask_coordinate_model"), dtype=np.float64)
    if A_depth_from_mask.shape != (3, 3) or not np.isfinite(A_depth_from_mask).all():
        raise RuntimeError("invalid A_depth_from_mask_coordinate_model")
    K_mask = np.linalg.inv(A_depth_from_mask) @ K_depth
    intrinsics = np.asarray([K_mask[0, 0], K_mask[1, 1], K_mask[0, 2], K_mask[1, 2]], dtype=np.float64)
    if not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0.0):
        raise RuntimeError(f"invalid mask-plane intrinsics: {intrinsics}")
    return intrinsics, {
        "mask_size_wh": [width, height],
        "depth_intrinsics_fx_fy_cx_cy": [fx, fy, cx, cy],
        "A_depth_from_mask_coordinate_model": A_depth_from_mask.tolist(),
        "mask_intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        "pixel_center_convention": transform.get("pixel_center_convention"),
    }


def rasterize_convex_projection(vertices: np.ndarray, intrinsics: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    height, width = shape
    fx, fy, cx, cy = intrinsics.tolist()
    z = vertices[:, 2]
    positive = np.isfinite(vertices).all(axis=1) & (z > 1.0e-6)
    uv = np.empty((0, 2), dtype=np.float64)
    if np.any(positive):
        uv = np.column_stack(
            (fx * vertices[positive, 0] / z[positive] + cx, fy * vertices[positive, 1] / z[positive] + cy)
        )
        uv = uv[np.isfinite(uv).all(axis=1)]
    inside = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= height - 1)
    ) if len(uv) else np.zeros(0, dtype=bool)
    uv_inside = uv[inside]
    projected = np.zeros(shape, dtype=bool)
    if len(uv_inside) >= 3:
        hull = cv2.convexHull(np.rint(uv_inside).astype(np.int32))
        cv2.fillConvexPoly(projected.view(np.uint8), hull, 1)
    return projected, {
        "positive_depth_vertex_fraction": float(np.count_nonzero(positive) / max(1, len(vertices))),
        "finite_projected_vertices": int(len(uv)),
        "projected_vertices_inside_image": int(len(uv_inside)),
        "convex_projection_pixels": int(np.count_nonzero(projected)),
    }


def mask_metrics(projected: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    projected = np.asarray(projected, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.count_nonzero(projected & target))
    union = int(np.count_nonzero(projected | target))
    target_pixels = int(np.count_nonzero(target))
    projected_pixels = int(np.count_nonzero(projected))
    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(intersection / max(1, union)),
        "target_recall": float(intersection / max(1, target_pixels)),
        "projected_precision": float(intersection / max(1, projected_pixels)),
        "target_pixels": target_pixels,
        "projected_pixels": projected_pixels,
    }


def summarize_points(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    lo = np.percentile(points, 0.5, axis=0)
    hi = np.percentile(points, 99.5, axis=0)
    return {
        "count": int(len(points)),
        "mean": np.mean(points, axis=0).astype(float).tolist(),
        "median": np.median(points, axis=0).astype(float).tolist(),
        "extent_raw": (np.max(points, axis=0) - np.min(points, axis=0)).astype(float).tolist(),
        "extent_p005_p995": (hi - lo).astype(float).tolist(),
        "depth_median": float(np.median(points[:, 2])),
    }


def evaluate_hypothesis(name: str, vertices: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    projected, projection_stats = rasterize_convex_projection(vertices, intrinsics, mask.shape)
    return {
        "name": name,
        "point_stats": summarize_points(vertices),
        "projection": projection_stats,
        "mask_overlap": mask_metrics(projected, mask),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    p12_path = require_file(args.p12_report, "P12 report")
    evidence_path = require_file(args.evidence_report, "evidence report")
    p12 = load_json(p12_path)
    evidence = load_json(evidence_path)
    candidate = sam3d_candidate(p12)
    selected, visible = selected_visible_evidence(evidence)

    raw_mesh = load_mesh(candidate["raw_mesh"])
    raw_vertices = np.asarray(raw_mesh.vertices, dtype=np.float64)
    faces = np.asarray(raw_mesh.faces, dtype=np.int64)
    pose = candidate["pose"]
    rotation = finite_vector(pose.get("rotation"), 4, "SAM3D native quaternion")
    translation = finite_vector(pose.get("translation"), 3, "SAM3D native translation")
    scale = np.asarray(pose.get("scale"), dtype=np.float64).reshape(-1)
    p3d_vertices, opencv_vertices, replay_contract = apply_native_pose(
        raw_vertices, rotation, translation, scale
    )

    p3d_exported = np.asarray(load_mesh(candidate["exported_pytorch3d"]).vertices, dtype=np.float64)
    cv_exported = np.asarray(load_mesh(candidate["exported_opencv"]).vertices, dtype=np.float64)
    if p3d_exported.shape != p3d_vertices.shape or cv_exported.shape != opencv_vertices.shape:
        raise RuntimeError("exported native mesh topology changed")
    p3d_error = float(np.max(np.abs(p3d_exported - p3d_vertices)))
    cv_error = float(np.max(np.abs(cv_exported - opencv_vertices)))
    if max(p3d_error, cv_error) > float(args.max_replay_vertex_error):
        raise RuntimeError(f"native pose replay mismatch: p3d={p3d_error}, opencv={cv_error}")

    mask_path = require_file(
        Path(str(selected.get("mask_path") or visible.get("mask_path") or "")),
        "selected object-owned mask",
    )
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None or np.count_nonzero(mask_image) == 0:
        raise RuntimeError(f"cannot decode nonempty mask: {mask_path}")
    mask = mask_image > 0
    intrinsics, raster_contract = mask_plane_intrinsics(visible, mask.shape)

    observed_camera = np.asarray(visible.get("camera_vertices_sample_m"), dtype=np.float64)
    observed_camera = observed_camera[
        np.isfinite(observed_camera).all(axis=1) & (observed_camera[:, 2] > 0.0)
    ]
    if observed_camera.ndim != 2 or observed_camera.shape[1] != 3 or len(observed_camera) < int(args.min_observed_points):
        raise RuntimeError(f"insufficient observed camera points: {observed_camera.shape}")
    sensor_depth_median = float(np.median(observed_camera[:, 2]))

    native_projection, _ = rasterize_convex_projection(opencv_vertices, intrinsics, mask.shape)
    fx, fy, cx, cy = intrinsics.tolist()
    z = opencv_vertices[:, 2]
    positive = np.isfinite(opencv_vertices).all(axis=1) & (z > 1.0e-6)
    uv = np.column_stack(
        (fx * opencv_vertices[positive, 0] / z[positive] + cx, fy * opencv_vertices[positive, 1] / z[positive] + cy)
    )
    z_positive = z[positive]
    rounded = np.rint(uv).astype(np.int64)
    inside = (
        np.isfinite(uv).all(axis=1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < mask.shape[0])
    )
    rounded_inside = rounded[inside]
    z_inside = z_positive[inside]
    owned = mask[rounded_inside[:, 1], rounded_inside[:, 0]] if len(rounded_inside) else np.zeros(0, dtype=bool)
    if np.count_nonzero(owned) < int(args.min_owned_projected_vertices):
        raise RuntimeError("too few native vertices project into the owned mask")
    native_front_depth = float(np.median(z_inside[owned]))
    metric_scale = sensor_depth_median / native_front_depth
    if not np.isfinite(metric_scale) or metric_scale <= 0.0:
        raise RuntimeError(f"invalid camera-origin metric scale: {metric_scale}")
    metric_native = opencv_vertices * metric_scale

    raw_as_opencv = raw_vertices.copy()
    p3d_as_opencv = p3d_vertices @ np.diag([-1.0, -1.0, 1.0])
    hypotheses = {
        "raw_local_naively_as_opencv_camera": evaluate_hypothesis(
            "raw_local_naively_as_opencv_camera", raw_as_opencv, intrinsics, mask
        ),
        "native_pytorch3d_reinterpreted_as_opencv_without_axis_flip": evaluate_hypothesis(
            "native_pytorch3d_reinterpreted_as_opencv_without_axis_flip", p3d_vertices, intrinsics, mask
        ),
        "native_pytorch3d_then_declared_axis_flip": evaluate_hypothesis(
            "native_pytorch3d_then_declared_axis_flip", p3d_as_opencv, intrinsics, mask
        ),
        "native_opencv": evaluate_hypothesis("native_opencv", opencv_vertices, intrinsics, mask),
        "native_opencv_camera_origin_metric_scaled": evaluate_hypothesis(
            "native_opencv_camera_origin_metric_scaled", metric_native, intrinsics, mask
        ),
    }

    native_iou = hypotheses["native_opencv"]["mask_overlap"]["iou"]
    raw_iou = hypotheses["raw_local_naively_as_opencv_camera"]["mask_overlap"]["iou"]
    wrong_axis_iou = hypotheses[
        "native_pytorch3d_reinterpreted_as_opencv_without_axis_flip"
    ]["mask_overlap"]["iou"]
    metric_iou = hypotheses[
        "native_opencv_camera_origin_metric_scaled"
    ]["mask_overlap"]["iou"]
    verdict_reasons = []
    if native_iou < float(args.min_native_projection_iou):
        verdict_reasons.append(f"native_opencv_iou_below_{args.min_native_projection_iou}")
    if native_iou < raw_iou + float(args.min_native_iou_gain_over_raw):
        verdict_reasons.append("native_pose_does_not_outperform_raw_local")
    if native_iou < wrong_axis_iou + float(args.min_native_iou_gain_over_wrong_axis):
        verdict_reasons.append("declared_opencv_axis_flip_not_discriminative")
    if abs(metric_iou - native_iou) > float(args.max_metric_scale_iou_delta):
        verdict_reasons.append("camera_origin_uniform_scale_changed_projection")
    verdict = "native_pose_contract_supported" if not verdict_reasons else "native_pose_contract_unresolved"

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_mesh_path = output_dir / "sam3d_native_opencv_camera_origin_metric_scaled.ply"
    trimesh.Trimesh(vertices=metric_native, faces=faces, process=False).export(str(metric_mesh_path))

    report = {
        "schema": "sam3d_native_pose_contract_audit_v1",
        "status": verdict,
        "annotation_ready": False,
        "diagnostic_only": True,
        "claim_scope": (
            "Coordinate/transform and projection audit only. Generated SAM3D faces remain render-only "
            "and provide no collision, sign, contact, or nonpenetration authority."
        ),
        "inputs": {
            "p12_report": str(p12_path),
            "evidence_report": str(evidence_path),
            "raw_mesh": str(candidate["raw_mesh"]),
            "raw_mesh_sha256": sha256_file(candidate["raw_mesh"]),
            "object_owned_mask": str(mask_path),
            "selected_frame_idx": int(evidence.get("selected_frame_idx", p12.get("selected_frame_idx", -1))),
        },
        "declared_contract": candidate["contract"],
        "replayed_contract": replay_contract,
        "pose": {
            "rotation_wxyz": rotation.tolist(),
            "translation_native_scene": translation.tolist(),
            "scale_native_scene_per_model_unit": scale.astype(float).tolist(),
        },
        "replay_validation": {
            "exported_pytorch3d_max_abs_vertex_error": p3d_error,
            "exported_opencv_max_abs_vertex_error": cv_error,
            "maximum_allowed_error": float(args.max_replay_vertex_error),
            "topology_preserved": True,
        },
        "raster_contract": raster_contract,
        "metric_scene_similarity": {
            "observed_owned_surface_median_depth_m": sensor_depth_median,
            "native_owned_projected_vertex_median_depth": native_front_depth,
            "camera_origin_uniform_scale": metric_scale,
            "translation_and_object_scale_scaled_together": True,
            "projection_invariant_expected": True,
        },
        "hypotheses": hypotheses,
        "verdict": {
            "status": verdict,
            "reasons": verdict_reasons,
            "native_opencv_iou": native_iou,
            "raw_local_iou": raw_iou,
            "wrong_axis_iou": wrong_axis_iou,
            "metric_native_iou": metric_iou,
            "native_iou_gain_over_raw": float(native_iou - raw_iou),
            "native_iou_gain_over_wrong_axis": float(native_iou - wrong_axis_iou),
            "metric_scale_iou_delta": float(metric_iou - native_iou),
        },
        "outputs": {
            "native_opencv_camera_origin_metric_mesh": str(metric_mesh_path),
        },
    }
    report_path = output_dir / "sam3d_native_pose_contract_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": verdict,
        "raw_local_iou": raw_iou,
        "wrong_axis_iou": wrong_axis_iou,
        "native_opencv_iou": native_iou,
        "metric_scale": metric_scale,
        "report": str(report_path),
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p12-report", type=Path, required=True)
    parser.add_argument("--evidence-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-replay-vertex-error", type=float, default=1.0e-5)
    parser.add_argument("--min-observed-points", type=int, default=100)
    parser.add_argument("--min-owned-projected-vertices", type=int, default=100)
    parser.add_argument("--min-native-projection-iou", type=float, default=0.25)
    parser.add_argument("--min-native-iou-gain-over-raw", type=float, default=0.10)
    parser.add_argument("--min-native-iou-gain-over-wrong-axis", type=float, default=0.10)
    parser.add_argument("--max-metric-scale-iou-delta", type=float, default=1.0e-12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
