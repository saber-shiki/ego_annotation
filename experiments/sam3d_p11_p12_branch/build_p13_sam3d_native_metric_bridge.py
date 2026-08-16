#!/usr/bin/env python3
"""Bridge a SAM3D native local mesh into the shared metric canonical frame.

This adapter preserves SAM3D's decoded native orientation and image-plane pose.
It does not use the TRELLIS RMS/PCA/permutation/ICP path. The native monocular
scene similarity is converted to sensor metric scale from prediction-side,
robust first-surface depth while scaling object size and camera translation
together. Generated faces remain render-only.
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
from scipy.spatial import cKDTree
import trimesh

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from remote_run_sam3d_objects_mesh_v7 import apply_native_pose  # noqa: E402

SCHEMA = "v19_sam3d_native_sensor_metric_canonical_bridge_v1"
COMPATIBILITY_SCHEMA = "v19_metric_canonical_generated_prior_input_v1"


def load_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise RuntimeError(f"{label} must be finite length {size}, got {array}")
    return array


def robust_point_stats(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise RuntimeError("point stats require finite Nx3 points")
    lo = np.percentile(points, 0.5, axis=0)
    hi = np.percentile(points, 99.5, axis=0)
    extent = hi - lo
    return {
        "count": int(len(points)),
        "center_mean": points.mean(axis=0).astype(float).tolist(),
        "center_median": np.median(points, axis=0).astype(float).tolist(),
        "raw_extent": (points.max(axis=0) - points.min(axis=0)).astype(float).tolist(),
        "raw_extent_diag": float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))),
        "robust_p005_p995_extent": extent.astype(float).tolist(),
        "robust_p005_p995_extent_diag": float(np.linalg.norm(extent)),
    }


def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    rng = np.random.default_rng(1313)
    points, _ = trimesh.sample.sample_surface(
        mesh,
        min(int(count), max(1, int(len(mesh.faces)) * 2)),
        seed=rng,
    )
    return np.asarray(points, dtype=np.float64)


def nearest_surface_summary(query: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    distances, _ = cKDTree(np.asarray(target, dtype=np.float64)).query(
        np.asarray(query, dtype=np.float64), k=1, workers=-1
    )
    return {
        "count": int(len(distances)),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "mean_m": float(np.mean(distances)),
        "max_m": float(np.max(distances)),
    }


def observed_front_quality_decision(
    *,
    median_fraction: float,
    p95_fraction: float,
    native_projection_iou: float,
    maximum_median_fraction: float,
    strict_maximum_p95_fraction: float,
    conditional_maximum_p95_fraction: float,
    conditional_minimum_projection_iou: float,
) -> dict[str, Any]:
    """Apply a strict tier plus a bounded, explicitly uncertain P95-tail tier.

    The conditional tier is intentionally narrow: the robust median must still
    satisfy the unchanged strict limit, only the P95 tail may exceed the strict
    limit, and the native image-plane pose must have substantial mask overlap.
    This keeps a small partial-surface tail from suppressing a usable render
    prior without admitting a gross native-pose failure.
    """
    values = np.asarray(
        [
            median_fraction,
            p95_fraction,
            native_projection_iou,
            maximum_median_fraction,
            strict_maximum_p95_fraction,
            conditional_maximum_p95_fraction,
            conditional_minimum_projection_iou,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite observed-front quality inputs: {values}")
    if conditional_maximum_p95_fraction < strict_maximum_p95_fraction:
        raise RuntimeError("conditional P95 limit must not be below the strict P95 limit")

    median_passed = bool(median_fraction <= maximum_median_fraction)
    strict_p95_passed = bool(p95_fraction <= strict_maximum_p95_fraction)
    conditional_p95_passed = bool(p95_fraction <= conditional_maximum_p95_fraction)
    conditional_projection_passed = bool(
        native_projection_iou >= conditional_minimum_projection_iou
    )
    strict_passed = bool(median_passed and strict_p95_passed)
    conditional_tail_passed = bool(
        not strict_passed
        and median_passed
        and p95_fraction > strict_maximum_p95_fraction
        and conditional_p95_passed
        and conditional_projection_passed
    )
    if strict_passed:
        mode = "strict_observed_front_quality"
    elif conditional_tail_passed:
        mode = "conditional_p95_tail_uncertain_native_projection_supported"
    else:
        mode = "failed_observed_front_quality"
    return {
        "quality_passed": bool(strict_passed or conditional_tail_passed),
        "acceptance_mode": mode,
        "strict_quality_passed": strict_passed,
        "conditional_tail_quality_passed": conditional_tail_passed,
        "conditional_tail_uncertainty": conditional_tail_passed,
        "median_passed": median_passed,
        "strict_p95_passed": strict_p95_passed,
        "conditional_p95_passed": conditional_p95_passed,
        "conditional_projection_passed": conditional_projection_passed,
        "strict_maximum_p95_fraction_of_extent_diag": float(
            strict_maximum_p95_fraction
        ),
        "conditional_maximum_p95_fraction_of_extent_diag": float(
            conditional_maximum_p95_fraction
        ),
        "conditional_minimum_native_projection_iou": float(
            conditional_minimum_projection_iou
        ),
    }


def mesh_from_path(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"not a triangular mesh: {path}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
        raise RuntimeError(f"invalid mesh vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"invalid mesh faces: {path}")
    return loaded


def resolve_sam3d_candidate(p12: dict[str, Any]) -> dict[str, Any]:
    candidates = p12.get("candidates")
    if isinstance(candidates, dict) and isinstance(candidates.get("sam3d_objects"), dict):
        candidate = candidates["sam3d_objects"]
    elif str(p12.get("source_model")) == "sam3d_objects":
        candidate = p12
    else:
        raise RuntimeError("P12 report has no sam3d_objects candidate")
    native = candidate.get("native_outputs")
    if not isinstance(native, dict):
        raise RuntimeError("SAM3D P12 candidate lacks native_outputs")
    raw = native.get("raw_mesh")
    raw_path = Path(str(raw.get("path"))) if isinstance(raw, dict) else Path(str(raw or ""))
    raw_path = require_file(raw_path, "SAM3D raw local mesh")
    pose = native.get("native_pose")
    if not isinstance(pose, dict):
        raise RuntimeError("SAM3D P12 candidate lacks native_pose")
    native_contract = native.get("native_pose_contract")
    if not isinstance(native_contract, dict):
        raise RuntimeError("SAM3D P12 candidate lacks native_pose_contract")
    if native_contract.get("quaternion_order") != "wxyz_scalar_first_pytorch3d":
        raise RuntimeError("SAM3D P12 candidate has an unsupported quaternion convention")
    native_opencv_record = native.get("native_pose_mesh_opencv_camera")
    native_opencv_path = Path(str(native_opencv_record.get("path") or "")) if isinstance(native_opencv_record, dict) else Path("")
    native_opencv_path = require_file(native_opencv_path, "SAM3D exported native-pose OpenCV mesh")
    conditioning = candidate.get("conditioning") if isinstance(candidate.get("conditioning"), dict) else {}
    return {
        "candidate": candidate,
        "native": native,
        "raw_mesh": raw_path,
        "pose": pose,
        "native_pose_contract": native_contract,
        "native_opencv_mesh": native_opencv_path,
        "conditioning": conditioning,
    }


def resolve_anchor_centroid_world(evidence: dict[str, Any], visible: dict[str, Any]) -> np.ndarray:
    binding = evidence.get("selected_anchor_atomic_binding")
    if not isinstance(binding, dict) or binding.get("validated") is not True:
        raise RuntimeError("evidence lacks a validated atomic selected-anchor binding")
    selected_frame_idx = int(evidence.get("selected_frame_idx", -1))
    if (
        int(binding.get("selected_frame_idx", -2)) != selected_frame_idx
        or int(binding.get("visible_geometry_frame_idx", -2)) != selected_frame_idx
        or int(binding.get("canonical_surface_frame_idx", -2)) != selected_frame_idx
    ):
        raise RuntimeError(f"selected-anchor frame binding is inconsistent: {binding}")
    row = evidence.get("depth_fused_object_row") if isinstance(evidence.get("depth_fused_object_row"), dict) else {}
    reconstruction = row.get("mesh_reconstruction") if isinstance(row.get("mesh_reconstruction"), dict) else {}
    if reconstruction.get("atomic_anchor_binding") is not True:
        raise RuntimeError("selected mesh reconstruction is not atomically bound to the P11 row")
    mesh_frame_idx = int(reconstruction.get("anchor_frame_idx", -1))
    visible_frame_idx = int(visible.get("frame_idx", -1))
    if mesh_frame_idx != selected_frame_idx or visible_frame_idx != selected_frame_idx:
        raise RuntimeError(
            f"SAM3D bridge received mixed anchor frames: selected={selected_frame_idx}, "
            f"visible={visible_frame_idx}, canonical_surface={mesh_frame_idx}"
        )
    mesh_centroid = finite_vector(
        reconstruction.get("anchor_centroid_world_m"), 3, "selected canonical-surface centroid"
    )
    visible_centroid = finite_vector(visible.get("centroid_world_m"), 3, "selected visible centroid")
    error_m = float(np.linalg.norm(mesh_centroid - visible_centroid))
    if error_m > 1.0e-6:
        raise RuntimeError(f"selected visible/canonical centroids disagree by {error_m} m")
    return visible_centroid


def resolve_mask_intrinsics(visible: dict[str, Any], mask_size_wh: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    intrinsics = finite_vector(visible.get("intrinsics_fx_fy_cx_cy"), 4, "depth-plane intrinsics")
    K_depth = np.asarray(
        [[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = visible.get("mask_depth_transform_contract")
    if not isinstance(transform, dict) or transform.get("camera_contract_consistent") is not True:
        raise RuntimeError("visible evidence lacks a camera-consistent mask/depth transform")
    declared_mask_size = [int(value) for value in transform.get("mask_size_wh") or []]
    if declared_mask_size != [int(mask_size_wh[0]), int(mask_size_wh[1])]:
        raise RuntimeError(f"mask size {mask_size_wh} disagrees with transform contract {declared_mask_size}")
    A_depth_from_mask = np.asarray(transform.get("A_depth_from_mask_coordinate_model"), dtype=np.float64)
    if A_depth_from_mask.shape != (3, 3) or not np.isfinite(A_depth_from_mask).all():
        raise RuntimeError("invalid A_depth_from_mask_coordinate_model")
    K_mask = np.linalg.inv(A_depth_from_mask) @ K_depth
    if not np.allclose(K_mask[2], [0.0, 0.0, 1.0], atol=1.0e-9):
        raise RuntimeError("mask-plane K is not pinhole")
    intr_mask = np.asarray([K_mask[0, 0], K_mask[1, 1], K_mask[0, 2], K_mask[1, 2]], dtype=np.float64)
    if not np.isfinite(intr_mask).all() or np.any(intr_mask[:2] <= 0.0):
        raise RuntimeError(f"invalid mask-plane intrinsics: {intr_mask}")
    return intr_mask, {
        "depth_intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        "A_depth_from_mask": A_depth_from_mask.tolist(),
        "mask_intrinsics_fx_fy_cx_cy": intr_mask.tolist(),
    }


def convex_projection_and_front_depth(
    vertices_camera: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    vertices = np.asarray(vertices_camera, dtype=np.float64)
    fx, fy, cx, cy = finite_vector(intrinsics, 4, "projection intrinsics")
    height, width = mask.shape
    z = vertices[:, 2]
    valid = np.isfinite(vertices).all(axis=1) & (z > 1.0e-5)
    if np.count_nonzero(valid) < 3:
        raise RuntimeError("SAM3D native pose has fewer than three positive-depth vertices")
    u = fx * vertices[valid, 0] / z[valid] + cx
    v = fy * vertices[valid, 1] / z[valid] + cy
    uv = np.column_stack((u, v))
    finite_uv = np.isfinite(uv).all(axis=1)
    uv = uv[finite_uv]
    z_projected = z[valid][finite_uv]
    inside = (uv[:, 0] >= 0.0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= height - 1)
    uv_inside = uv[inside]
    z_inside = z_projected[inside]
    if len(uv_inside) < 3:
        raise RuntimeError("SAM3D native mesh projection does not enter the owned-mask image")

    hull = cv2.convexHull(np.rint(uv_inside).astype(np.int32))
    projected = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(projected, hull, 1)
    target = np.asarray(mask, dtype=bool)
    projected_bool = projected > 0
    intersection = int(np.count_nonzero(projected_bool & target))
    union = int(np.count_nonzero(projected_bool | target))
    iou = float(intersection / max(1, union))

    xi = np.rint(uv_inside[:, 0]).astype(np.int64)
    yi = np.rint(uv_inside[:, 1]).astype(np.int64)
    pixel_inside_target = target[yi, xi]
    xi = xi[pixel_inside_target]
    yi = yi[pixel_inside_target]
    z_owned = z_inside[pixel_inside_target]
    if len(z_owned) == 0:
        raise RuntimeError("SAM3D native projected vertices have no object-owned depth samples")
    flat_index = yi * width + xi
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, flat_index, z_owned)
    finite_zbuffer = zbuffer[np.isfinite(zbuffer)]
    return {
        "positive_depth_vertex_fraction": float(np.count_nonzero(valid) / len(vertices)),
        "projected_vertices_inside_image": int(len(uv_inside)),
        "projected_vertices_inside_owned_mask": int(len(z_owned)),
        "owned_zbuffer_pixel_count": int(len(finite_zbuffer)),
        "owned_zbuffer_coverage_fraction": float(len(finite_zbuffer) / max(1, np.count_nonzero(target))),
        "owned_zbuffer_median_native_depth": float(np.median(finite_zbuffer)),
        "convex_projection_iou": iou,
        "convex_projection_pixels": int(np.count_nonzero(projected_bool)),
        "owned_mask_pixels": int(np.count_nonzero(target)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence_path = require_file(args.evidence_report, "P11 evidence report")
    p12_path = require_file(args.p12_report, "P12 parallel report")
    evidence = load_json(evidence_path)
    p12 = load_json(p12_path)
    selected = evidence.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("evidence has no selected anchor")
    visible = selected.get("visible_geometry_candidate")
    if not isinstance(visible, dict):
        raise RuntimeError("selected anchor lacks visible_geometry_candidate")
    ownership = visible.get("first_surface_depth_ownership")
    if not isinstance(ownership, dict) or ownership.get("enabled") is not True:
        raise RuntimeError("SAM3D metric bridge requires explicit robust first-surface depth ownership")
    if ownership.get("fail_closed") is True or ownership.get("failure_reasons"):
        raise RuntimeError(f"selected first-surface depth ownership is failed closed: {ownership.get('failure_reasons')}")
    if float(ownership.get("retained_fraction") or 0.0) < float(args.min_robust_depth_support_fraction):
        raise RuntimeError("selected robust first-surface support is below the bridge threshold")

    candidate = resolve_sam3d_candidate(p12)
    raw_mesh_path = candidate["raw_mesh"]
    raw_mesh = mesh_from_path(raw_mesh_path)
    raw_vertices = np.asarray(raw_mesh.vertices, dtype=np.float64)
    pose = candidate["pose"]
    rotation = finite_vector(pose.get("rotation"), 4, "SAM3D native quaternion")
    translation = finite_vector(pose.get("translation"), 3, "SAM3D native translation")
    scale = np.asarray(pose.get("scale"), dtype=np.float64).reshape(-1)
    native_p3d, native_opencv, native_contract = apply_native_pose(
        raw_vertices, rotation, translation, scale
    )
    exported_native_opencv = mesh_from_path(candidate["native_opencv_mesh"])
    exported_native_vertices = np.asarray(exported_native_opencv.vertices, dtype=np.float64)
    if exported_native_vertices.shape != native_opencv.shape:
        raise RuntimeError("SAM3D exported native-pose mesh changed vertex topology")
    native_pose_mesh_max_abs_error = float(np.max(np.abs(exported_native_vertices - native_opencv)))
    if native_pose_mesh_max_abs_error > 1.0e-5:
        raise RuntimeError(
            f"SAM3D exported native-pose mesh disagrees with the declared transform: {native_pose_mesh_max_abs_error}"
        )

    mask_path = require_file(Path(str(selected.get("mask_path") or visible.get("mask_path") or "")), "selected object-owned mask")
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None or np.count_nonzero(mask_image) == 0:
        raise RuntimeError(f"cannot decode nonempty owned mask: {mask_path}")
    mask = mask_image > 0
    mask_h, mask_w = mask.shape
    mask_intrinsics, raster_contract = resolve_mask_intrinsics(visible, (mask_w, mask_h))
    projection = convex_projection_and_front_depth(native_opencv, mask_intrinsics, mask)
    if projection["positive_depth_vertex_fraction"] < float(args.min_positive_depth_fraction):
        raise RuntimeError(f"SAM3D native positive-depth fraction failed: {projection['positive_depth_vertex_fraction']}")
    if projection["convex_projection_iou"] < float(args.min_native_convex_projection_iou):
        raise RuntimeError(f"SAM3D native projection IoU failed: {projection['convex_projection_iou']}")
    if projection["owned_zbuffer_pixel_count"] < int(args.min_native_owned_zbuffer_pixels):
        raise RuntimeError(f"SAM3D native owned z-buffer support failed: {projection['owned_zbuffer_pixel_count']}")

    observed_camera = np.asarray(visible.get("camera_vertices_sample_m"), dtype=np.float64)
    if observed_camera.ndim != 2 or observed_camera.shape[1] != 3 or len(observed_camera) < int(args.min_observed_points):
        raise RuntimeError(f"insufficient robust observed camera points: {observed_camera.shape}")
    observed_camera = observed_camera[np.isfinite(observed_camera).all(axis=1) & (observed_camera[:, 2] > 0.0)]
    if len(observed_camera) < int(args.min_observed_points):
        raise RuntimeError("insufficient finite positive robust observed camera points")
    sensor_depth_median = float(np.median(observed_camera[:, 2]))
    native_front_depth = float(projection["owned_zbuffer_median_native_depth"])
    metric_scale = sensor_depth_median / native_front_depth
    if not np.isfinite(metric_scale) or not (float(args.min_scene_similarity_scale) <= metric_scale <= float(args.max_scene_similarity_scale)):
        raise RuntimeError(f"SAM3D scene similarity scale outside contract: {metric_scale}")

    metric_camera = native_opencv * metric_scale
    camera = selected.get("camera")
    if not isinstance(camera, dict):
        raise RuntimeError("selected anchor lacks camera transform")
    T_world_camera = np.asarray(camera.get("T_world_camera_metric") or camera.get("T_world_camera"), dtype=np.float64)
    if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all() or not np.allclose(T_world_camera[3], [0, 0, 0, 1]):
        raise RuntimeError("invalid selected T_world_camera_metric")
    anchor_centroid_world = resolve_anchor_centroid_world(evidence, visible)
    metric_world = metric_camera @ T_world_camera[:3, :3].T + T_world_camera[:3, 3][None, :]
    canonical = metric_world - anchor_centroid_world[None, :]
    observed_metric_canonical = (
        observed_camera @ T_world_camera[:3, :3].T
        + T_world_camera[:3, 3][None, :]
        - anchor_centroid_world[None, :]
    )

    generated_stats = robust_point_stats(canonical)
    observed_stats = robust_point_stats(observed_camera)
    extent_ratio = float(
        generated_stats["robust_p005_p995_extent_diag"]
        / max(observed_stats["robust_p005_p995_extent_diag"], 1.0e-12)
    )
    if not (float(args.min_complete_to_observed_extent_ratio) <= extent_ratio <= float(args.max_complete_to_observed_extent_ratio)):
        raise RuntimeError(f"SAM3D complete/observed robust extent ratio outside contract: {extent_ratio}")

    metric_mesh = trimesh.Trimesh(
        vertices=canonical,
        faces=np.asarray(raw_mesh.faces, dtype=np.int64),
        process=False,
    )
    generated_surface_samples = deterministic_sample_mesh(
        metric_mesh, int(args.geometry_quality_surface_samples)
    )
    observed_to_generated = nearest_surface_summary(
        observed_metric_canonical, generated_surface_samples
    )
    observed_extent_diag = float(observed_stats["robust_p005_p995_extent_diag"])
    median_fraction = float(observed_to_generated["median_m"] / max(observed_extent_diag, 1.0e-12))
    p95_fraction = float(observed_to_generated["p95_m"] / max(observed_extent_diag, 1.0e-12))
    quality_decision = observed_front_quality_decision(
        median_fraction=median_fraction,
        p95_fraction=p95_fraction,
        native_projection_iou=float(projection["convex_projection_iou"]),
        maximum_median_fraction=float(
            args.max_observed_to_generated_median_extent_fraction
        ),
        strict_maximum_p95_fraction=float(
            args.strict_observed_to_generated_p95_extent_fraction
        ),
        conditional_maximum_p95_fraction=float(
            args.max_observed_to_generated_p95_extent_fraction
        ),
        conditional_minimum_projection_iou=float(
            args.min_conditional_tail_native_projection_iou
        ),
    )
    geometry_quality = {
        "method": "selected_anchor_observed_metric_surfels_to_sampled_generated_surface",
        "generated_surface_sample_count": int(len(generated_surface_samples)),
        "observed_to_generated": observed_to_generated,
        "observed_robust_extent_diag_m": observed_extent_diag,
        "observed_to_generated_median_fraction_of_extent_diag": median_fraction,
        "observed_to_generated_p95_fraction_of_extent_diag": p95_fraction,
        "maximum_median_fraction_of_extent_diag": float(
            args.max_observed_to_generated_median_extent_fraction
        ),
        "maximum_p95_fraction_of_extent_diag": float(
            args.max_observed_to_generated_p95_extent_fraction
        ),
        **quality_decision,
        "generated_faces_pose_evidence_consumed": False,
        "interpretation": (
            "Render-prior coverage of the selected frame's measured front surface only. "
            "The strict tier remains P95/extent <= 0.15. A bounded conditional tail tier "
            "may carry explicit uncertainty only when the strict median still passes and "
            "native mask projection is substantial. This diagnostic cannot promote "
            "generated hidden faces to pose/contact/collision evidence."
        ),
    }
    if not geometry_quality["quality_passed"]:
        raise RuntimeError(f"SAM3D observed-front geometry quality failed: {geometry_quality}")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mesh_path = output_dir / "sam3d_native_sensor_metric_canonical_render_prior.ply"
    output_mesh = metric_mesh.copy()
    output_mesh.export(str(output_mesh_path))

    report_path = output_dir / "sam3d_native_sensor_metric_canonical_bridge_report.json"
    compatibility_path = output_dir / "p13_metric_canonical_input.json"
    report = {
        "schema": SCHEMA,
        "status": "ok_render_prior_metric_canonical",
        "method": "bridge_sam3d_native_pose_to_sensor_metric_canonical",
        "annotation_ready": False,
        "claim_scope": (
            "Prediction-side SAM3D generated render prior only. Native orientation/image pose are preserved; "
            "sensor metric scale comes from robust owned first-surface depth. Generated faces are not pose, contact, collision, sign, or nonpenetration evidence."
        ),
        "inputs": {
            "evidence_report": str(evidence_path),
            "p12_report": str(p12_path),
            "raw_mesh": str(raw_mesh_path),
            "raw_mesh_sha256": sha256_file(raw_mesh_path),
            "object_owned_mask": str(mask_path),
        },
        "native_pose_contract": {
            **native_contract,
            "p12_exported_contract": candidate["native_pose_contract"],
            "exported_native_pose_opencv_mesh": str(candidate["native_opencv_mesh"]),
            "exported_vs_recomputed_max_abs_error": native_pose_mesh_max_abs_error,
        },
        "native_pose": {
            "rotation_wxyz": rotation.tolist(),
            "translation_native_scene": translation.tolist(),
            "scale_native_scene_per_model_unit": scale.astype(float).tolist(),
            "native_pytorch3d_camera_stats": robust_point_stats(native_p3d),
            "native_opencv_camera_stats": robust_point_stats(native_opencv),
        },
        "projection_validation": {
            **projection,
            "minimum_convex_projection_iou": float(args.min_native_convex_projection_iou),
            "minimum_owned_zbuffer_pixels": int(args.min_native_owned_zbuffer_pixels),
            "raster_contract": raster_contract,
        },
        "sensor_metric_scene_similarity": {
            "sensor_robust_first_surface_median_depth_m": sensor_depth_median,
            "native_owned_zbuffer_median_depth": native_front_depth,
            "camera_origin_scene_similarity_scale": metric_scale,
            "formula": "p_cv_metric = lambda * (((p_raw * native_scale) @ R_wxyz + native_translation) @ diag(-1,-1,1))",
            "translation_and_object_scale_scaled_together": True,
            "robust_first_surface_depth_ownership": ownership,
        },
        "shared_canonical_frame": {
            "T_world_camera_metric": T_world_camera.tolist(),
            "anchor_centroid_world_m": anchor_centroid_world.tolist(),
            "formula": "p_canonical = p_cv_metric @ R_world_camera.T + t_world_camera - anchor_centroid_world",
        },
        "geometry_validation": {
            "observed_robust_camera_surface": observed_stats,
            "generated_metric_canonical": generated_stats,
            "complete_to_observed_robust_extent_diag_ratio": extent_ratio,
            "observed_front_surface_quality": geometry_quality,
            "allowed_extent_ratio": [
                float(args.min_complete_to_observed_extent_ratio),
                float(args.max_complete_to_observed_extent_ratio),
            ],
        },
        "semantics": {
            "generated_faces_render_underlay": True,
            "generated_faces_pose_eligible": False,
            "generated_faces_collision_eligible": False,
            "generated_faces_contact_eligible": False,
            "generated_faces_signed_distance_eligible": False,
            "observed_front_quality_acceptance_mode": geometry_quality["acceptance_mode"],
            "conditional_observed_tail_uncertainty": geometry_quality[
                "conditional_tail_uncertainty"
            ],
        },
        "outputs": {
            "metric_canonical_render_prior": str(output_mesh_path),
            "p13_metric_canonical_input": str(compatibility_path),
        },
    }
    write_json(report_path, report)
    compatibility = {
        "schema": COMPATIBILITY_SCHEMA,
        "status": "verified_sam3d_native_sensor_metric_canonical_render_prior",
        "source_model": "sam3d_objects",
        "mesh": str(output_mesh_path),
        "mesh_sha256": sha256_file(output_mesh_path),
        "bridge_report": str(report_path),
        "bridge_report_sha256": sha256_file(report_path),
        "metric_canonical_input_contract": {
            "frame": "shared_completed_canonical_meters",
            "alignment_must_be_identity": True,
            "generated_faces_render_only": True,
            "native_orientation_preserved": True,
            "camera_origin_scene_similarity_applied": True,
        },
    }
    write_json(compatibility_path, compatibility)
    print(json.dumps({
        "status": report["status"],
        "output_mesh": str(output_mesh_path),
        "metric_scale": metric_scale,
        "projection_iou": projection["convex_projection_iou"],
        "extent_ratio": extent_ratio,
        "report": str(report_path),
        "p13_input": str(compatibility_path),
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-report", type=Path, required=True)
    parser.add_argument("--p12-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-robust-depth-support-fraction", type=float, default=0.90)
    parser.add_argument("--min-positive-depth-fraction", type=float, default=0.99)
    parser.add_argument("--min-native-convex-projection-iou", type=float, default=0.05)
    parser.add_argument("--min-native-owned-zbuffer-pixels", type=int, default=100)
    parser.add_argument("--min-observed-points", type=int, default=100)
    parser.add_argument("--min-scene-similarity-scale", type=float, default=0.05)
    parser.add_argument("--max-scene-similarity-scale", type=float, default=5.0)
    parser.add_argument("--min-complete-to-observed-extent-ratio", type=float, default=0.25)
    parser.add_argument("--max-complete-to-observed-extent-ratio", type=float, default=5.0)
    parser.add_argument("--geometry-quality-surface-samples", type=int, default=40000)
    parser.add_argument(
        "--max-observed-to-generated-median-extent-fraction", type=float, default=0.05
    )
    parser.add_argument(
        "--strict-observed-to-generated-p95-extent-fraction", type=float, default=0.15,
        help="Unchanged strict observed-front P95/extent tier.",
    )
    parser.add_argument(
        "--max-observed-to-generated-p95-extent-fraction", type=float, default=0.18,
        help="Maximum bounded conditional P95/extent tier; reported as uncertain when above strict.",
    )
    parser.add_argument(
        "--min-conditional-tail-native-projection-iou", type=float, default=0.25,
        help="Minimum native convex projection IoU required by the conditional P95-tail tier.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
