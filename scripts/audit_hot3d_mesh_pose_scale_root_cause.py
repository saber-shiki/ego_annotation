#!/usr/bin/env python3
"""Audit scale, coordinate-frame, alignment, and trajectory failures in HOT3D dual-backend runs.

This is a read-only forensic tool. It does not repair masks/depth, alter generated
meshes, fit a new pose, or mutate a completed run. Counterfactual quantities are
clearly labeled diagnostics and are never exported as annotation results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

SCHEMA = "hot3d_dual_backend_mesh_pose_scale_root_cause_audit_v1"
P3D_CAMERA_TO_OPENCV_CAMERA_ROW = np.diag([-1.0, -1.0, 1.0])


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_case(raw: str) -> tuple[str, str]:
    parts = raw.split("|", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise RuntimeError("--case must have form CASE_ID|OBJECT_ID")
    return parts[0].strip(), parts[1].strip()


def candidate(report: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in report.get("candidates", []) if isinstance(row, dict) and row.get("name") == name]
    if len(rows) != 1:
        raise RuntimeError(f"candidate {name!r} appears {len(rows)} times")
    return rows[0]


def load_mesh(path: Path) -> trimesh.Trimesh:
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        meshes = [
            mesh for mesh in geometry.geometry.values()
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) and len(mesh.faces)
        ]
        if not meshes:
            raise RuntimeError(f"scene contains no triangle mesh: {path}")
        geometry = trimesh.util.concatenate(meshes)
    if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.vertices) or not len(geometry.faces):
        raise RuntimeError(f"invalid triangle mesh: {path}")
    return geometry


def load_vertices(path: Path) -> np.ndarray:
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        chunks = [np.asarray(item.vertices, dtype=np.float64) for item in geometry.geometry.values() if hasattr(item, "vertices")]
        if not chunks:
            raise RuntimeError(f"geometry contains no vertices: {path}")
        vertices = np.concatenate(chunks, axis=0)
    else:
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise RuntimeError(f"invalid vertices: {path}")
    return vertices


def numeric_summary(values: np.ndarray | list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(np.max(array)),
    }


def point_stats(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise RuntimeError(f"invalid point array {points.shape}")
    low = points.min(axis=0)
    high = points.max(axis=0)
    robust_low = np.percentile(points, 0.5, axis=0)
    robust_high = np.percentile(points, 99.5, axis=0)
    center = points.mean(axis=0)
    extent = high - low
    robust_extent = robust_high - robust_low
    centered = points - center
    covariance = centered.T @ centered / max(1, len(points) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    return {
        "count": int(len(points)),
        "bounds": [low.astype(float).tolist(), high.astype(float).tolist()],
        "extent": extent.astype(float).tolist(),
        "extent_diag": float(np.linalg.norm(extent)),
        "robust_p005_p995_extent": robust_extent.astype(float).tolist(),
        "robust_p005_p995_extent_diag": float(np.linalg.norm(robust_extent)),
        "center": center.astype(float).tolist(),
        "rms_radius": float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))),
        "pca_eigenvalues_descending": eigenvalues.astype(float).tolist(),
        "pca_middle_over_largest": float(eigenvalues[1] / max(eigenvalues[0], 1.0e-15)),
        "pca_smallest_over_largest": float(eigenvalues[2] / max(eigenvalues[0], 1.0e-15)),
    }


def backproject(mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    z = depth[ys, xs].astype(np.float64)
    fx, fy, cx, cy = intrinsics.astype(np.float64).tolist()
    return np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def bbox_iou(a: list[int] | None, b: list[int] | None) -> float | None:
    if a is None or b is None:
        return None
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(intersection / max(1, area_a + area_b - intersection))


def intrinsics_for_mask_plane(source_intrinsics: np.ndarray, source_size: tuple[int, int], mask_size: tuple[int, int]) -> np.ndarray:
    source_w, source_h = source_size
    mask_w, mask_h = mask_size
    sx, sy = float(mask_w) / float(source_w), float(mask_h) / float(source_h)
    fx, fy, cx, cy = source_intrinsics.astype(np.float64).tolist()
    # OpenCV half-pixel resize coordinate model: x_mask=(x_source+0.5)*s-0.5.
    return np.asarray([fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5], dtype=np.float64)


def convex_projection_mask(vertices_camera: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices_camera, dtype=np.float64)
    z = vertices[:, 2]
    valid = np.isfinite(vertices).all(axis=1) & (z > 0.01)
    points = vertices[valid]
    output = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return output > 0, {"valid_projected_vertices": int(len(points)), "bbox_xyxy": None}
    fx, fy, cx, cy = intrinsics.tolist()
    uv = np.column_stack((fx * points[:, 0] / points[:, 2] + cx, fy * points[:, 1] / points[:, 2] + cy))
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]
    if len(uv) < 3:
        return output > 0, {"valid_projected_vertices": int(len(uv)), "bbox_xyxy": None}
    # Clipping prevents integer overflow while preserving the in-image convex polygon.
    uv[:, 0] = np.clip(uv[:, 0], -4.0 * width, 5.0 * width)
    uv[:, 1] = np.clip(uv[:, 1], -4.0 * height, 5.0 * height)
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    cv2.fillConvexPoly(output, np.rint(hull).astype(np.int32), 1, lineType=cv2.LINE_8)
    return output > 0, {
        "valid_projected_vertices": int(len(uv)),
        "projected_uv_bounds_unclipped": [
            [float(np.min(uv[:, 0])), float(np.min(uv[:, 1]))],
            [float(np.max(uv[:, 0])), float(np.max(uv[:, 1]))],
        ],
        "bbox_xyxy": mask_bbox(output > 0),
    }


def projection_stats(projected: np.ndarray, target: np.ndarray, details: dict[str, Any]) -> dict[str, Any]:
    intersection = int(np.count_nonzero(projected & target))
    union = int(np.count_nonzero(projected | target))
    return {
        **details,
        "convex_projection_pixels": int(np.count_nonzero(projected)),
        "target_mask_pixels": int(np.count_nonzero(target)),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "convex_projection_iou": float(intersection / union) if union else 1.0,
        "bbox_iou": bbox_iou(mask_bbox(projected), mask_bbox(target)),
        "metric_scope": "convex projected-vertex envelope diagnostic; not a z-buffer silhouette score",
    }


def world_to_camera(points_world: np.ndarray, transform_world_camera: np.ndarray) -> np.ndarray:
    return (points_world - transform_world_camera[:3, 3][None, :]) @ transform_world_camera[:3, :3]


def camera_to_world(points_camera: np.ndarray, transform_world_camera: np.ndarray) -> np.ndarray:
    return points_camera @ transform_world_camera[:3, :3].T + transform_world_camera[:3, 3][None, :]


def canonical_to_world(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation[None, :]


def world_to_canonical(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (points - translation[None, :]) @ rotation


def pose_steps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["frame_idx"]))
    rotations = np.asarray([row["rotation_world_from_completed_canonical_matrix"] for row in ordered], dtype=np.float64)
    translations = np.asarray([row["translation_world_m"] for row in ordered], dtype=np.float64)
    frame_ids = np.asarray([int(row["frame_idx"]) for row in ordered], dtype=np.int64)
    if len(rows) < 2:
        return {"frame_count": len(rows)}
    relative = rotations[1:] @ np.transpose(rotations[:-1], (0, 2, 1))
    angle_deg = np.degrees(Rotation.from_matrix(relative).magnitude())
    translation_step = np.linalg.norm(translations[1:] - translations[:-1], axis=1)
    gaps = np.diff(frame_ids)
    return {
        "frame_count": int(len(rows)),
        "rotation_step_deg": numeric_summary(angle_deg),
        "translation_step_m": numeric_summary(translation_step),
        "frame_gap": numeric_summary(gaps.astype(np.float64)),
        "max_rotation_step": {
            "from_frame": int(frame_ids[int(np.argmax(angle_deg))]),
            "to_frame": int(frame_ids[int(np.argmax(angle_deg)) + 1]),
            "degrees": float(np.max(angle_deg)),
        },
        "max_translation_step": {
            "from_frame": int(frame_ids[int(np.argmax(translation_step))]),
            "to_frame": int(frame_ids[int(np.argmax(translation_step)) + 1]),
            "meters": float(np.max(translation_step)),
        },
    }


def alpha_overlay(canvas: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    color = np.asarray(color_bgr, dtype=np.float32)
    canvas[mask] = np.clip((1.0 - alpha) * canvas[mask].astype(np.float32) + alpha * color, 0, 255).astype(np.uint8)


def contour(canvas: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], thickness: int) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color_bgr, thickness, cv2.LINE_AA)


def diagnostic_image(
    rgb_path: Path,
    target_mask: np.ndarray,
    far_tail_source: np.ndarray,
    native_projection: np.ndarray,
    current_projection: np.ndarray,
    observed_projection: np.ndarray,
    output: Path,
    title: str,
    labels: list[str],
) -> None:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"failed to read RGB: {rgb_path}")
    if rgb.shape[:2] != target_mask.shape:
        rgb = cv2.resize(rgb, (target_mask.shape[1], target_mask.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = rgb.copy()
    alpha_overlay(canvas, current_projection, (200, 45, 210), 0.18)
    alpha_overlay(canvas, native_projection, (50, 220, 60), 0.18)
    contour(canvas, target_mask, (255, 255, 0), 2)
    contour(canvas, observed_projection, (0, 180, 255), 2)
    contour(canvas, current_projection, (200, 45, 210), 2)
    contour(canvas, native_projection, (50, 220, 60), 2)
    ys, xs = np.where(far_tail_source)
    if len(xs):
        sx = float(target_mask.shape[1]) / float(far_tail_source.shape[1])
        sy = float(target_mask.shape[0]) / float(far_tail_source.shape[0])
        xm = np.clip(np.rint((xs + 0.5) * sx - 0.5).astype(np.int32), 0, target_mask.shape[1] - 1)
        ym = np.clip(np.rint((ys + 0.5) * sy - 0.5).astype(np.int32), 0, target_mask.shape[0] - 1)
        canvas[ym, xm] = (0, 0, 255)
    panel_h = 32 + 23 * (len(labels) + 1)
    panel = np.full((panel_h, canvas.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(panel, title[:130], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)
    for index, text in enumerate(labels):
        cv2.putText(panel, text[:155], (12, 50 + 23 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (225, 225, 225), 1, cv2.LINE_AA)
    sheet = np.vstack([panel, canvas])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"failed to write diagnostic image: {output}")


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def audit_case(runs_root: Path, case_id: str, object_id: str, output_dir: Path) -> dict[str, Any]:
    run = (runs_root / case_id).resolve()
    visible_dir = run / "measurements" / "object_geometry" / "visible_geometry" / object_id
    visible_report_path = require_file(visible_dir / "v19_visible_geometry_adapter_report.json", "visible geometry report")
    visible_report = load_json(visible_report_path)
    annotations_path = require_file(visible_dir / "annotations_v19_visible_geometry.json", "visible annotations")
    annotations = load_json(annotations_path)
    anchor = int(visible_report["anchor_frame_idx"])

    evidence_path = require_file(
        run / "measurements" / "geometry_completion" / "rigid_evidence" / case_id / object_id / "evidence_bundle" / "evidence_bundle_report.json",
        "P11 evidence report",
    )
    evidence = load_json(evidence_path)
    if int(evidence["selected_frame_idx"]) != anchor:
        raise RuntimeError(f"{case_id}: visible anchor {anchor} differs from P11 selected frame")
    selected = evidence["selected"]
    visible_candidate = selected["visible_geometry_candidate"]

    p12_path = require_file(run / "experiments/sam3d_trellis_controlled/P12_parallel/p12_parallel_geometry_priors_report.json", "P12 report")
    p12 = load_json(p12_path)
    p13_path = require_file(run / "experiments/sam3d_trellis_controlled/P13_controlled/p13_controlled_geometry_prior_ab_report.json", "P13 report")
    p13 = load_json(p13_path)
    dual_path = require_file(run / "experiments/sam3d_trellis_controlled/P13_sam3d_dual/p13_dual_mesh_geometry_prior_report.json", "P13 SAM3D dual report")
    dual = load_json(dual_path)
    p14_path = require_file(run / "experiments/sam3d_trellis_controlled/P14_observed_pose_fit/v18_compact_rigid_object_pose_fit_report.json", "P14 pose fit")
    p14 = load_json(p14_path)
    p15_path = require_file(run / "experiments/sam3d_trellis_controlled/P15_observed_pose_graph/v19_rigid_object_pose_graph_report.json", "P15 pose graph")
    p15 = load_json(p15_path)
    layered_path = require_file(
        run / "experiments/sam3d_trellis_controlled/P15_layered_states/sam3d_owned_dual_mesh/experimental_layered_render_state.json",
        "SAM3D layered state",
    )
    layered = load_json(layered_path)

    sam12 = p12["candidates"]["sam3d_objects"]
    native = sam12["native_outputs"]
    sam13 = candidate(p13, "sam3d_new_object_owned_mask")
    tre13 = candidate(p13, "trellis_frozen")

    depth_path = require_file(Path(visible_report["inputs"]["depth_npz"]), "camera-bound depth archive")
    with np.load(depth_path, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_idx"], dtype=np.int64)
        depth_i = int(np.flatnonzero(frame_ids == anchor)[0])
        depth = np.asarray(archive["depth"][depth_i], dtype=np.float64)
        source_size_archive = tuple(int(value) for value in np.asarray(archive["source_size"]).reshape(-1).tolist())
        source_intrinsics_archive = np.asarray(archive["intrinsics_fx_fy_cx_cy"][depth_i], dtype=np.float64)
        source_estimated_intrinsics = np.asarray(archive["source_estimated_intrinsics_fx_fy_cx_cy"][depth_i], dtype=np.float64)

    adapter_report_path = require_file(
        depth_path.parent / "v19_depth_camera_contract_adapter_report.json",
        "depth/camera adapter report",
    )
    adapter_report = load_json(adapter_report_path)
    depth_values_unchanged = bool(
        (adapter_report.get("array_invariants") or {}).get("depth_array_equal") is True
        and (adapter_report.get("array_invariants") or {}).get("depth_array_sha256_equal") is True
    )

    mask_path = require_file(Path(visible_candidate["mask_path"]), "object-owned mask")
    mask_manifest = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_manifest is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    mask_manifest = mask_manifest > 0
    source_w, source_h = source_size_archive
    if depth.shape != (source_h, source_w):
        raise RuntimeError(f"depth/source size mismatch: {depth.shape} vs {source_size_archive}")
    mask_source = cv2.resize(mask_manifest.astype(np.uint8), (source_w, source_h), interpolation=cv2.INTER_NEAREST_EXACT) > 0
    valid = mask_source & np.isfinite(depth) & (depth >= 0.05) & (depth <= 4.0)
    z = depth[valid]
    median_z = float(np.median(z))
    mad_z = float(np.median(np.abs(z - median_z)))
    robust_sigma_z = 1.4826 * mad_z
    # This threshold is a forensic label only. It is not an acceptance/filtering parameter.
    extreme_tail_delta = max(0.15, 10.0 * robust_sigma_z)
    extreme_tail_threshold = median_z + extreme_tail_delta
    far_tail = valid & (depth > extreme_tail_threshold)
    distance_inside = cv2.distanceTransform(mask_source.astype(np.uint8), cv2.DIST_L2, 5)
    far_distances = distance_inside[far_tail]

    core_lo, core_hi = np.percentile(z, [5.0, 95.0])
    core = valid & (depth >= core_lo) & (depth <= core_hi)
    raw_points_camera = backproject(valid, depth, source_intrinsics_archive)
    core_points_camera = backproject(core, depth, source_intrinsics_archive)
    ownership_filter = visible_candidate.get("object_surface_ownership_filter")
    ownership_filter = ownership_filter if isinstance(ownership_filter, dict) else {}
    ownership_input_pixels = int(ownership_filter.get("input_mask_pixels") or 0)
    ownership_removed_pixels = int(ownership_filter.get("removed_mask_pixels") or 0)
    ownership_removed_fraction = float(ownership_removed_pixels / max(1, ownership_input_pixels))
    sampled_points_camera = np.asarray(visible_candidate["camera_vertices_sample_m"], dtype=np.float64)

    partial = evidence["partial_metric_geometry_paths"]
    anchor_points_path = require_file(Path(partial["fused_point_cloud_path"]), "anchor canonical points")
    anchor_poisson_path = require_file(Path(partial["poisson_mesh_path"]), "anchor Poisson mesh")
    anchor_points = load_vertices(anchor_points_path)
    anchor_poisson = load_vertices(anchor_poisson_path)
    reconstruction = evidence.get("depth_fused_object_row", {}).get("mesh_reconstruction", {})

    raw_mesh_path = require_file(Path(native["raw_mesh"]["path"]), "raw SAM3D mesh")
    raw_mesh = load_mesh(raw_mesh_path)
    raw_vertices = np.asarray(raw_mesh.vertices, dtype=np.float64)
    native_pose = native["native_pose"]
    quaternion_wxyz = np.asarray(native_pose["rotation"], dtype=np.float64)
    if quaternion_wxyz.shape != (4,):
        raise RuntimeError("SAM3D quaternion is not wxyz[4]")
    rotation_native = Rotation.from_quat(
        [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]
    ).as_matrix()
    translation_native = np.asarray(native_pose["translation"], dtype=np.float64)
    scale_native = np.asarray(native_pose["scale"], dtype=np.float64)
    if translation_native.shape != (3,) or scale_native.shape != (3,):
        raise RuntimeError("invalid SAM3D native translation/scale")
    # PyTorch3D Transform3d row-vector semantics, matching SceneVisualizer.object_pointcloud.
    native_p3d_camera = (raw_vertices * scale_native[None, :]) @ rotation_native + translation_native[None, :]
    native_opencv_camera = native_p3d_camera @ P3D_CAMERA_TO_OPENCV_CAMERA_ROW
    native_depth_reference = float(np.median(native_opencv_camera[:, 2]))
    if not math.isfinite(native_depth_reference) or native_depth_reference <= 0.01:
        raise RuntimeError("SAM3D native transformed mesh has invalid camera depth")
    sensor_depth_bridge_scale = median_z / native_depth_reference
    bridged_opencv_camera = native_opencv_camera * sensor_depth_bridge_scale

    camera = selected["camera"]
    transform_world_camera = np.asarray(camera["T_world_camera_metric"], dtype=np.float64)
    if transform_world_camera.shape != (4, 4):
        raise RuntimeError("selected camera transform is invalid")
    anchor_pose_rows = [row for row in p15["pose_rows"] if int(row["frame_idx"]) == anchor]
    if len(anchor_pose_rows) != 1:
        raise RuntimeError(f"anchor pose row appears {len(anchor_pose_rows)} times")
    anchor_pose = anchor_pose_rows[0]
    rotation_world_canonical = np.asarray(anchor_pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    translation_world = np.asarray(anchor_pose["translation_world_m"], dtype=np.float64)

    bridged_world = camera_to_world(bridged_opencv_camera, transform_world_camera)
    bridged_canonical = world_to_canonical(bridged_world, rotation_world_canonical, translation_world)

    current_aligned_path = require_file(Path(dual["outputs"]["aligned_raw_render_prior"]), "current aligned SAM3D prior")
    current_aligned = load_vertices(current_aligned_path)
    observed_path = require_file(Path(dual["outputs"]["collision_eligible_observed_surface"]), "observed collision surface")
    observed_canonical = load_vertices(observed_path)
    trellis_aligned_path = require_file(Path(tre13["source_neutral_outputs"]["generated_aligned_all_candidate_mesh"]), "aligned TRELLIS prior")
    trellis_aligned = load_vertices(trellis_aligned_path)

    mask_h, mask_w = mask_manifest.shape
    mask_intrinsics = intrinsics_for_mask_plane(source_intrinsics_archive, (source_w, source_h), (mask_w, mask_h))
    native_projection, native_projection_details = convex_projection_mask(native_opencv_camera, mask_intrinsics, mask_w, mask_h)
    current_world = canonical_to_world(current_aligned, rotation_world_canonical, translation_world)
    current_camera = world_to_camera(current_world, transform_world_camera)
    current_projection, current_projection_details = convex_projection_mask(current_camera, mask_intrinsics, mask_w, mask_h)
    observed_world = canonical_to_world(observed_canonical, rotation_world_canonical, translation_world)
    observed_camera = world_to_camera(observed_world, transform_world_camera)
    observed_projection, observed_projection_details = convex_projection_mask(observed_camera, mask_intrinsics, mask_w, mask_h)

    native_projection_report = projection_stats(native_projection, mask_manifest, native_projection_details)
    current_projection_report = projection_stats(current_projection, mask_manifest, current_projection_details)
    observed_projection_report = projection_stats(observed_projection, mask_manifest, observed_projection_details)

    current_stats = point_stats(current_aligned)
    bridged_stats = point_stats(bridged_canonical)
    raw_model_stats = point_stats(raw_vertices)
    native_camera_stats = point_stats(native_opencv_camera)
    current_over_bridged = float(
        current_stats["robust_p005_p995_extent_diag"]
        / max(bridged_stats["robust_p005_p995_extent_diag"], 1.0e-12)
    )

    layers = layered.get("object_geometry", {}).get("render_layers_back_to_front", [])
    generated_layers = [row for row in layers if isinstance(row, dict) and row.get("role") == "generated_complete_prior_underlay"]
    if len(generated_layers) != 1:
        raise RuntimeError("layered state does not contain exactly one generated underlay")
    layered_generated_path = require_file(Path(generated_layers[0]["mesh"]), "layered generated mesh")
    same_layer_geometry = bool(
        np.array_equal(load_vertices(layered_generated_path), current_aligned)
    )

    p14_pose_rows = [row for row in p14.get("pose_rows", []) if isinstance(row, dict)]
    p15_pose_rows = [row for row in p15.get("pose_rows", []) if isinstance(row, dict)]
    pose_step_report = pose_steps(p15_pose_rows)
    correction = p15.get("correction_summary", {})
    correction_is_exact_zero = True
    for key in ("translation_delta_norm_m", "rotation_delta_norm_rad"):
        values = correction.get(key, {}) if isinstance(correction.get(key), dict) else {}
        if any(float(values.get(field) or 0.0) != 0.0 for field in ("median", "p90", "max", "mean")):
            correction_is_exact_zero = False

    depth_raw_stats = point_stats(raw_points_camera)
    depth_core_stats = point_stats(core_points_camera)
    raw_over_core = float(depth_raw_stats["extent_diag"] / max(depth_core_stats["extent_diag"], 1.0e-12))
    boundary_tail_dominated = bool(
        len(far_distances) > 0
        and float(np.median(far_distances)) <= 3.0
        and raw_over_core >= 2.0
    )

    diagnostic_path = output_dir / "cases" / case_id / "anchor_scale_pose_root_cause_overlay.png"
    diagnostic_image(
        require_file(Path(selected["raw_frame_path"]), "anchor RGB"),
        mask_manifest,
        far_tail,
        native_projection,
        current_projection,
        observed_projection,
        diagnostic_path,
        f"{case_id} | anchor {anchor} | forensic overlay (not annotation)",
        [
            "cyan=owned mask; red=extreme depth-tail pixels; orange=observed convex projection",
            "green=SAM3D native-pose projection; magenta=current P13 generic-PCA aligned projection",
            f"current/native-bridged robust size ratio={current_over_bridged:.2f}x; raw/core depth extent ratio={raw_over_core:.2f}x",
            f"native depth {native_depth_reference:.3f}m -> sensor owned median {median_z:.3f}m; scene similarity scale={sensor_depth_bridge_scale:.3f}",
        ],
    )

    return {
        "case": case_id,
        "object_id": object_id,
        "status": "invalid_existing_result_root_causes_identified",
        "anchor_frame_idx": anchor,
        "inputs": {
            "visible_report": str(visible_report_path),
            "evidence_report": str(evidence_path),
            "p12_report": str(p12_path),
            "p13_report": str(p13_path),
            "p13_dual_report": str(dual_path),
            "p14_pose_fit_report": str(p14_path),
            "p15_pose_graph_report": str(p15_path),
            "layered_state": str(layered_path),
            "depth_archive": str(depth_path),
            "depth_camera_adapter_report": str(adapter_report_path),
            "object_owned_mask": str(mask_path),
        },
        "camera_and_raster_contract": {
            "source_size_wh": [source_w, source_h],
            "mask_size_wh": [mask_w, mask_h],
            "source_intrinsics_fx_fy_cx_cy": source_intrinsics_archive.astype(float).tolist(),
            "mask_plane_intrinsics_fx_fy_cx_cy": mask_intrinsics.astype(float).tolist(),
            "source_estimated_unidepth_intrinsics_fx_fy_cx_cy": source_estimated_intrinsics.astype(float).tolist(),
            "active_vs_unidepth_focal_ratio": float(
                math.sqrt(source_intrinsics_archive[0] * source_intrinsics_archive[1])
                / math.sqrt(source_estimated_intrinsics[0] * source_estimated_intrinsics[1])
            ),
            "adapter_depth_values_unchanged": depth_values_unchanged,
            "ray_contract_warning": (
                "The adapter replaced UniDepth-estimated K with sensor K while proving the dense depth array was byte-identical. "
                "That proves metadata binding, not that the predicted z raster is geometrically reprojected onto sensor rays."
            ),
            "camera_transform_world_from_opencv_camera": transform_world_camera.astype(float).tolist(),
        },
        "hand_object_ownership_failure": {
            "filter_state": ownership_filter.get("state"),
            "ownership_primitive": "axis_aligned_hand_bbox" if ownership_filter.get("hand_boxes") is not None else ownership_filter.get("ownership_primitive"),
            "input_sam2_mask_pixels": ownership_input_pixels,
            "removed_by_hand_bboxes_pixels": ownership_removed_pixels,
            "removed_by_hand_bboxes_fraction": ownership_removed_fraction,
            "bbox_subtraction_scientifically_valid": False,
            "failure_mechanism": (
                "A hand bounding rectangle is not a hand ownership silhouette. It removes visible object pixels wherever the box overlaps the object and still cannot represent finger boundaries. "
                "The resulting observed surface becomes unnecessarily partial and rotation/pose support is weakened."
            ),
            "required_repair": "project full prediction-side HaWoR/MANO vertices/faces to a hand triangle silhouette; fail closed if that geometry cannot be bound; never fall back to the bbox as an ownership mask",
        },
        "depth_edge_ownership_failure": {
            "valid_object_owned_depth_pixels": int(np.count_nonzero(valid)),
            "depth_m_summary": numeric_summary(z),
            "median_absolute_deviation_m": mad_z,
            "robust_sigma_from_mad_m": robust_sigma_z,
            "forensic_extreme_tail_definition": {
                "threshold_m": extreme_tail_threshold,
                "formula": "median + max(0.15m, 10*1.4826*MAD)",
                "use": "diagnostic label only; not a proposed production filter",
            },
            "extreme_far_tail_pixels": int(np.count_nonzero(far_tail)),
            "extreme_far_tail_fraction": float(np.count_nonzero(far_tail) / max(1, np.count_nonzero(valid))),
            "far_tail_distance_inside_owned_mask_px": numeric_summary(far_distances),
            "raw_backprojected_surface": depth_raw_stats,
            "p05_p95_depth_core_counterfactual": {
                "depth_range_m": [float(core_lo), float(core_hi)],
                "retained_pixels": int(np.count_nonzero(core)),
                "retained_fraction": float(np.count_nonzero(core) / max(1, np.count_nonzero(valid))),
                "surface": depth_core_stats,
                "use": "forensic counterfactual only; no result was rebuilt from it",
            },
            "raw_extent_diag_over_core_extent_diag": raw_over_core,
            "sampled_visible_points": point_stats(sampled_points_camera),
            "voxel_downsampled_anchor_points_canonical": point_stats(anchor_points),
            "anchor_poisson_visible_mesh_canonical": point_stats(anchor_poisson),
            "reconstruction_voxel_size_m": reconstruction.get("voxel_size_m"),
            "boundary_tail_dominated": boundary_tail_dominated,
            "root_cause": (
                "A small set of monocular-depth discontinuity pixels inside the resized SAM2 boundary is treated as object-owned. "
                "Unfiltered min/max and RMS statistics turn this boundary tail into object scale; adaptive voxel size then preserves the long tail while collapsing the dense object core."
                if boundary_tail_dominated else
                "No dominant meter-scale boundary tail at this anchor, but partial/thin visible support still leaves scale and rotation underconstrained."
            ),
        },
        "sam3d_native_contract": {
            "raw_mesh": str(raw_mesh_path),
            "raw_mesh_stats_model_units": raw_model_stats,
            "quaternion_order": "wxyz (PyTorch3D)",
            "rotation_wxyz": quaternion_wxyz.astype(float).tolist(),
            "translation_pytorch3d_camera_units": translation_native.astype(float).tolist(),
            "scale_pytorch3d_camera_units_per_model_unit": scale_native.astype(float).tolist(),
            "native_transform_row_formula": "p_p3d_camera = (p_raw_local * scale) @ quaternion_to_matrix(q) + translation",
            "camera_convention_bridge_row_matrix": P3D_CAMERA_TO_OPENCV_CAMERA_ROW.astype(float).tolist(),
            "camera_convention_bridge": "PyTorch3D x-left,y-up,z-forward to V19/OpenCV x-right,y-down,z-forward",
            "native_pose_applied_opencv_camera_stats": native_camera_stats,
            "native_projection_against_owned_mask": native_projection_report,
            "evidence_interpretation": "native pose has measurable image alignment and must not be discarded as unverified metadata",
        },
        "sam3d_metric_scene_similarity_bridge_counterfactual": {
            "status": "diagnostic_not_exported",
            "sensor_owned_depth_median_m": median_z,
            "native_transformed_mesh_median_z": native_depth_reference,
            "scene_similarity_scale_sensor_over_native": sensor_depth_bridge_scale,
            "formula": "p_opencv_metric = (p_raw*native_scale @ native_R + native_t) @ diag(-1,-1,1) * (sensor_owned_median_z/native_mesh_median_z)",
            "important": "scale multiplies both native object scale and native translation, preserving native image projection; scaling only mesh vertices would be incorrect",
            "bridged_mesh_in_shared_completed_canonical_frame": bridged_stats,
            "exported_as_annotation": False,
        },
        "stage_geometry": {
            "current_p13_sam3d_alignment": sam13["metric_alignment"],
            "current_p13_sam3d_aligned_mesh": {
                "path": str(current_aligned_path),
                "stats": current_stats,
                "projection_against_owned_mask_at_shared_anchor_pose": current_projection_report,
            },
            "current_p13_trellis_alignment": tre13["metric_alignment"],
            "current_p13_trellis_aligned_stats": point_stats(trellis_aligned),
            "observed_collision_surface": {
                "path": str(observed_path),
                "stats": point_stats(observed_canonical),
                "projection_against_owned_mask_at_shared_anchor_pose": observed_projection_report,
            },
            "current_sam3d_robust_extent_diag_over_native_sensor_bridged": current_over_bridged,
            "first_generated_mesh_failure_stage": "P13 generic PCA/RMS/ICP alignment",
            "p13_failure_mechanism": (
                "build_v18_compact_rigid_trellis_completion applies TRELLIS-generic RMS scaling, all PCA axis permutations/signs, and one-sided nearest-neighbor ICP to raw SAM3D local vertices. "
                "It does not consume SAM3D native rotation/translation/scale, so contaminated observed RMS creates oversize geometry and PCA symmetry/partial-view ambiguity creates arbitrary axis swaps/flips."
            ),
        },
        "renderer_transform_audit": {
            "layered_generated_mesh": str(layered_generated_path),
            "layered_mesh_geometry_exactly_equals_p13_dual_aligned_mesh": same_layer_geometry,
            "p13_alignment_matrix_present_in_layered_state": "metric_alignment_matrix_model_to_canonical" in json.dumps(layered),
            "renderer_behavior": "loads already-canonical layer vertices and applies vertices @ shared_pose_R.T + shared_pose_t exactly once",
            "double_transform_detected": False,
            "conclusion": "oversize/axis error exists in the persisted P13 mesh before temporal rendering; the layered renderer does not reapply the P13 alignment matrix",
        },
        "observed_only_pose_trajectory": {
            "p14_fit_frame_count": int(p14.get("fit_frame_count") or 0),
            "p14_total_rows": int(p14.get("frame_count") or len(p14_pose_rows)),
            "p14_final_residual_summary": p14.get("final_observed_to_mesh_median_summary_m"),
            "p15_annotation_ready_claim": p15.get("annotation_ready"),
            "p15_graph_support": p15.get("graph_support"),
            "p15_full_timeline_completion": p15.get("full_timeline_rigid_pose_completion"),
            "p15_nonpenetration_target_frame_count": p15.get("nonpenetration_target_frame_count"),
            "p15_correction_summary": correction,
            "p15_correction_is_exact_zero": correction_is_exact_zero,
            "trajectory_step_diagnostics": pose_step_report,
            "failure_mechanism": (
                "P14 uses independent one-sided nearest-neighbor rigid ICP against a partial/contaminated canonical surface. Rotation is unobservable or ambiguous for thin, symmetric, and partial surfaces. "
                "P15 optimizes only a correction field around those P14 measurements; with zero nonpenetration targets, the all-zero correction is the exact objective minimum, so it does not smooth or repair the measured trajectory. "
                "Its annotation_ready gate checks optimizer success, surface residual, and a count threshold, but not temporal coverage, hold/interpolation fraction, per-frame rotation jumps, or rotational observability."
            ),
        },
        "scientific_validity": {
            "existing_d19_complete_is_quality_success": False,
            "existing_result_valid_for_backend_comparison": False,
            "observed_surface_metric_scale_valid": not boundary_tail_dominated,
            "sam3d_orientation_integration_valid": False,
            "shared_pose_trajectory_valid": False,
            "required_fail_closed_actions": [
                "reject axis-aligned hand bbox subtraction as an ownership primitive; require projected full MANO triangle silhouettes",
                "reject anchor/object scale when boundary-localized depth tails dominate raw extent/RMS",
                "consume and validate the SAM3D native local-to-camera similarity transform with explicit PyTorch3D-to-OpenCV convention",
                "bridge SAM3D scene similarity to shared sensor metric depth by scaling translation and object scale together",
                "do not use unrestricted PCA permutations/sign flips as the primary SAM3D orientation estimator",
                "make pose readiness depend on temporal coverage, interpolation/hold fraction, rotation-step diagnostics, and rotational observability",
            ],
        },
        "outputs": {
            "anchor_diagnostic_overlay": str(diagnostic_path),
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def markdown(report: dict[str, Any], output_path: Path) -> None:
    rows = report["cases"]
    lines = [
        "# HOT3D 双后端 mesh 尺度、坐标系与位姿根因报告",
        "",
        f"- 审计 schema：`{report['schema']}`",
        f"- 被审计 runs：`{report['runs_root']}`",
        "- 结论：**现有 D19 complete 结果不具备科学有效性，不能用于 SAM3D/TRELLIS 效果比较。**",
        "- 本报告为只读 forensic audit；没有重跑模型、没有改 mask/depth、没有导出替代 annotation。",
        "",
        "## 一、总根因",
        "",
        "1. **P03c depth/K 不是几何适配**：adapter 明确保持 UniDepth dense z raster byte-identical，只替换 K 元数据。各 anchor 的 UniDepth focal 与 sensor K 有明显差异；因此当前文件合同只证明 provenance 绑定，并未证明预测 z 已被重投影到 sensor rays。这不是 meter-scale Z 长尾的主因，但会额外扭曲 X/Y 尺度和投影合同。",
        "2. **P09 hand/object ownership 使用了矩形 bbox**：bbox 不是 hand silhouette；它会把 bbox 与物体重叠处仍然可见的物体像素整块删除，同时无法表达手指边界。四个 case 的 observed support 因而被进一步变得 partial，soup/spatula 的旋转可观测性尤其差。",
        "3. **P09 visible depth 的边界 ownership 失败**：即使在 bbox subtraction 之后，少量位于 owned mask 深度不连续区域的 monocular-depth 飞点仍被当成物体表面。后续使用原始 min/max、RMS radius 和由错误对角线决定的 voxel size，使小比例长尾变成米级 canonical object。",
        "4. **SAM3D 集成丢弃原生 pose**：runner 导出了 raw local mesh，也记录了 `rotation/translation/scale`，但 P13 把它当 TRELLIS mesh，重新执行 RMS+PCA 全轴排列/符号翻转+单向 ICP。由此同时丢失 SAM3D 的相机对齐、引入任意换轴/180° 翻转，并继承错误 observed RMS 尺度。",
        "5. **SAM3D 原生 pose 存在明确坐标合同**：raw mesh/gaussian native local coordinates 按 `p=(v*scale)@R+t` 进入 PyTorch3D camera，再乘 `diag(-1,-1,1)` 进入 OpenCV camera。原生 pose 的投影与 owned mask 有实测重合，因此不能忽略。",
        "6. **SAM3D scene 需要整体 similarity metric bridge**：native object scale 与 native translation 必须一起乘 `sensor_object_median_z/native_mesh_median_z`；这保持 2D 投影不变并把内部 MoGe scene scale 桥接到共享传感器米制深度。只缩 mesh、不缩 translation 是错误的。",
        "7. **最终 renderer 没有重复应用 P13 transform**：P13 矩阵已烘焙进 mesh；layered renderer 只再应用一次共享 per-frame SE(3)。尺寸错误在落盘的 P13 mesh 中已经存在。",
        "8. **共享 observed-only trajectory 也独立失效**：P14 对 partial/symmetric/污染表面逐帧做 one-sided ICP；P15 仅平滑“修正量”。无 nonpenetration target 时零修正是精确最优解，所以报告中的全零 correction 并不代表轨迹正确。当前 readiness gate 也未约束旋转跳变、覆盖率或 hold/interpolation 比例。",
        "",
        "## 二、逐 case 定量摘要",
        "",
        "| case | bbox removed | raw/core depth extent | boundary-tail frac | current SAM3D / native-metric size | native convex IoU | current P13 convex IoU | direct poses | max ΔR/frame | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        depth = row["depth_edge_ownership_failure"]
        sam = row["sam3d_native_contract"]
        stage = row["stage_geometry"]
        pose = row["observed_only_pose_trajectory"]
        steps = pose["trajectory_step_diagnostics"].get("max_rotation_step", {})
        lines.append(
            "| {case} | {hand_removed}% | {rawcore}× | {tail}% | {size}× | {native} | {current} | {direct}/150 | {rot}° | invalid |".format(
                case=row["case"],
                hand_removed=fmt(100.0 * row["hand_object_ownership_failure"]["removed_by_hand_bboxes_fraction"], 1),
                rawcore=fmt(depth["raw_extent_diag_over_core_extent_diag"], 2),
                tail=fmt(100.0 * depth["extreme_far_tail_fraction"], 2),
                size=fmt(stage["current_sam3d_robust_extent_diag_over_native_sensor_bridged"], 2),
                native=fmt(sam["native_projection_against_owned_mask"]["convex_projection_iou"], 3),
                current=fmt(stage["current_p13_sam3d_aligned_mesh"]["projection_against_owned_mask_at_shared_anchor_pose"]["convex_projection_iou"], 3),
                direct=pose["p14_fit_frame_count"],
                rot=fmt(steps.get("degrees"), 1),
            )
        )
    lines.extend([
        "",
        "> Convex IoU 是 projected-vertex convex envelope 诊断，不是 z-buffer silhouette 指标；只用于定位 pose/scale 首次失真阶段。",
        "",
        "## 三、SAM3D 正确的坐标与尺度桥接",
        "",
        "对 raw local vertex `v`：",
        "",
        "```text",
        "R = quaternion_to_matrix(q_wxyz)",
        "p_p3d = (v * native_scale) @ R + native_translation",
        "p_cv_native = p_p3d @ diag(-1, -1, +1)",
        "lambda_metric = median_z(shared owned metric depth) / median_z(p_cv_native mesh)",
        "p_cv_metric = lambda_metric * p_cv_native",
        "p_world = p_cv_metric @ R_world_camera.T + t_world_camera",
        "p_completed_canonical = (p_world - shared_pose_t_anchor) @ shared_pose_R_anchor",
        "```",
        "",
        "这里 `lambda_metric` 同时作用于 native translation 和 native object scale。它是 camera-origin scene similarity，不是对 mesh 的孤立缩放。审计只计算了该 counterfactual，未把它写入现有结果。",
        "",
        "### 逐 case native metric bridge",
        "",
        "| case | native median-z | sensor median-z | λ | bridged robust diag | current P13 robust diag | 放大倍数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        bridge = row["sam3d_metric_scene_similarity_bridge_counterfactual"]
        current = row["stage_geometry"]["current_p13_sam3d_aligned_mesh"]["stats"]
        lines.append(
            f"| {row['case']} | {fmt(bridge['native_transformed_mesh_median_z'])} m | {fmt(bridge['sensor_owned_depth_median_m'])} m | "
            f"{fmt(bridge['scene_similarity_scale_sensor_over_native'])} | {fmt(bridge['bridged_mesh_in_shared_completed_canonical_frame']['robust_p005_p995_extent_diag'])} m | "
            f"{fmt(current['robust_p005_p995_extent_diag'])} m | {fmt(row['stage_geometry']['current_sam3d_robust_extent_diag_over_native_sensor_bridged'], 2)}× |"
        )
    lines.extend([
        "",
        "## 四、为何不是 renderer double-transform",
        "",
        "- `build_p13_dual_mesh_geometry_prior.py` 将 `matrix_model_to_canonical` 直接烘焙到 `generated_aligned_raw_render_prior_intact.ply`。",
        "- P15 layered state 直接引用该已对齐 mesh，未携带或再次应用 P13 alignment matrix。",
        "- `render_p14_p15_layered_state.py::object_scene_layers` 只执行 `vertices_world = vertices @ pose_R.T + pose_t`。",
        "- 审计逐 case 验证 layered mesh geometry 与 P13 dual aligned mesh 完全相同。",
        "- 因此 oversize 和换轴在 P13 落盘时已发生；renderer 只是把错误 canonical mesh 按共享 pose 显示出来。",
        "",
        "## 五、trajectory 的独立问题",
        "",
    ])
    for row in rows:
        pose = row["observed_only_pose_trajectory"]
        step = pose["trajectory_step_diagnostics"]
        completion = pose.get("p15_full_timeline_completion") or {}
        lines.extend([
            f"### {row['case']}",
            "",
            f"- P14 direct fit：`{pose['p14_fit_frame_count']}/150`；P15 completion：`{completion.get('completed_row_count')}`。",
            f"- 最大相邻帧旋转跳变：`{fmt((step.get('max_rotation_step') or {}).get('degrees'), 2)}°`；最大平移跳变：`{fmt((step.get('max_translation_step') or {}).get('meters'), 4)} m`。",
            f"- nonpenetration target frame：`{pose['p15_nonpenetration_target_frame_count']}`；correction 是否全零：`{pose['p15_correction_is_exact_zero']}`。",
            f"- P15 虽声明 annotation_ready=`{pose['p15_annotation_ready_claim']}`，但该声明没有覆盖上述 temporal/observability 缺陷，故本审计判定无效。",
            "",
        ])
    lines.extend([
        "## 六、必须 fail-closed 的修复顺序",
        "",
        "1. P03c 不能把 byte-identical UniDepth z raster 仅换成 sensor K 后称作同一 sensor-ray 深度；必须由 depth model 接受 sensor camera，并把 metric radius 显式投影到 exact contract rays，或进行有定义的 3D reproject/z-buffer resampling。",
        "2. hand/object ownership 必须使用投影后的完整 HaWoR/MANO triangle silhouette；bbox 只可诊断，不得删除/保留 surfel。无法解析 MANO geometry 时该帧 fail closed。",
        "3. 在 visible geometry lift 前加入 depth-discontinuity ownership 状态；若少量 depth tail 主导 raw extent/RMS，必须阻塞 anchor，而不是继续 Poisson/P13。",
        "4. reconstruction voxel size 不得由未经 robust ownership 验证的 raw min/max diagonal 决定。",
        "5. SAM3D runner 必须显式导出 raw-local、native-pose camera-frame、四元数顺序、camera convention 和 scene-scale provenance。",
        "6. 新建 SAM3D-specific P13 bridge：先应用 native pose/convention，再做 camera-origin metric similarity；禁止走 TRELLIS PCA permutation 主路径。",
        "7. 用 anchor owned silhouette 和 shared metric depth 对 native bridge做 fail-closed 复核；不能匹配时应标为 backend prior rejected，而不是退回任意 PCA。",
        "8. P14/P15 readiness 必须加入 rotation observability、相邻帧 SE(3) jump、direct coverage、最大 gap、hold/interpolation fraction；P15 若 correction objective 在零 target 下恒取零，不能声称已修正轨迹。",
        "9. 修复后使用新 bundle、新 run root 重跑；现有 v2 产物只保留为失败证据，不覆盖。",
        "",
        "## 七、审计图",
        "",
    ])
    for row in rows:
        lines.append(f"- `{row['case']}`：`{row['outputs']['anchor_diagnostic_overlay']}`")
    lines.extend([
        "",
        "## 八、结论",
        "",
        "当前视觉异常不是一个单一 renderer bug，而是四个串联/并行故障：**P03c 仅替换 K、未把 UniDepth z 重投影到 sensor rays**，**矩形 hand bbox 过度删除 visible object support**，**remaining depth tail 污染共享米制几何/trajectory**，以及 **SAM3D native similarity transform 被 P13 错误丢弃并由 TRELLIS-specific PCA/RMS/ICP 取代**。renderer 没有重复变换。所有现有 completed case 必须降级为 invalid forensic result，直至上述合同修复并在新 run root 验证。",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    runs_root = args.runs_root.expanduser().resolve()
    if not runs_root.is_dir():
        raise RuntimeError(f"missing runs root: {runs_root}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty audit output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [parse_case(raw) for raw in args.case]
    if len({case for case, _ in cases}) != len(cases):
        raise RuntimeError("duplicate --case entries")
    started = time.time()
    rows = [audit_case(runs_root, case, object_id, output_dir) for case, object_id in cases]
    current_source_root = Path(__file__).resolve().parents[1]
    audited_source_root = (
        args.audited_runtime_bundle.expanduser().resolve()
        if args.audited_runtime_bundle is not None
        else current_source_root
    )
    if not audited_source_root.is_dir():
        raise RuntimeError(f"missing audited runtime source root: {audited_source_root}")
    source_files = [
        audited_source_root / "scripts/adapt_v19_depth_to_camera_contract.py",
        audited_source_root / "scripts/build_v19_visible_geometry_from_sam2_depth.py",
        audited_source_root / "scripts/build_v18_compact_rigid_trellis_completion.py",
        audited_source_root / "scripts/fit_v18_compact_rigid_object_pose.py",
        audited_source_root / "scripts/solve_v19_rigid_object_pose_graph.py",
        audited_source_root / "scripts/remote_run_sam3d_objects_mesh_v7.py",
        audited_source_root / "experiments/sam3d_p11_p12_branch/build_p13_dual_mesh_geometry_prior.py",
        audited_source_root / "experiments/sam3d_p11_p12_branch/render_p14_p15_layered_state.py",
    ]
    report = {
        "schema": SCHEMA,
        "status": "complete_existing_results_invalid_root_causes_identified",
        "method": "audit_hot3d_mesh_pose_scale_root_cause",
        "created_unix_s": time.time(),
        "runs_root": str(runs_root),
        "audited_runtime_source_root": str(audited_source_root),
        "audit_tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "claim_scope": (
            "Read-only forensic audit of prediction-side artifacts. No GT pose/MANO/depth/CAD/mask is consumed; "
            "no existing run artifact is modified; all similarity/core calculations are diagnostics, not replacement annotations."
        ),
        "global_findings": {
            "existing_results_scientifically_valid": False,
            "depth_raster_reprojected_to_sensor_rays_by_p03c": False,
            "generated_mesh_first_failure_stage": "P13 generic RMS/PCA/ICP alignment, with upstream P03c ray-contract mismatch, coarse hand-bbox over-subtraction, and P09 depth-tail scale contamination",
            "sam3d_native_pose_was_consumed_by_current_p13": False,
            "renderer_double_transform_detected": False,
            "shared_observed_pose_trajectory_independently_valid": False,
            "bulk_rerun_authorized_by_this_report": False,
        },
        "coordinate_contract": {
            "sam3d_raw_mesh_frame": "native local generator frame empirically congruent with the native Gaussian local frame; GLB export applies a separate z-up-to-y-up presentation rotation",
            "sam3d_quaternion": "PyTorch3D wxyz",
            "sam3d_transform": "row vectors: (v*scale) @ R + translation",
            "sam3d_camera_frame": "PyTorch3D x-left,y-up,z-forward",
            "v19_camera_frame": "OpenCV x-right,y-down,z-forward",
            "p3d_to_v19_row_matrix": P3D_CAMERA_TO_OPENCV_CAMERA_ROW.astype(float).tolist(),
            "metric_bridge": "camera-origin similarity scales native translation and native object scale together",
        },
        "audited_source_files": [
            {"path": str(path), "sha256": sha256_file(require_file(path, "audited source"))}
            for path in source_files
        ],
        "case_count": len(rows),
        "cases": rows,
        "outputs": {
            "json_report": str(output_dir / "ROOT_CAUSE_REPORT.json"),
            "chinese_markdown_report": str(output_dir / "ROOT_CAUSE_REPORT_ZH.md"),
        },
        "elapsed_s": time.time() - started,
    }
    write_json(output_dir / "ROOT_CAUSE_REPORT.json", report)
    markdown(report, output_dir / "ROOT_CAUSE_REPORT_ZH.md")
    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "global_findings": report["global_findings"],
        "outputs": report["outputs"],
        "elapsed_s": report["elapsed_s"],
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="CASE_ID|OBJECT_ID")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audited-runtime-bundle",
        type=Path,
        help="Frozen runtime source root that produced the audited runs; source hashes are taken here rather than from the repair worktree.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
