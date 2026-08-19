#!/usr/bin/env python3
"""Build one backend-neutral HOT3D signed-geometry candidate from direct observations.

The physical body is reconstructed in the D15 canonical object frame from
prediction-side object-owned masks, active-K metric depth, explicit direct P14
observations, the unchanged D15 pose values, and projected MANO occlusion masks.
SAM3D/TRELLIS meshes are not inputs.  Failed geometric evidence falls back to
the observed-only collision surface while preserving the candidate and report.
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
import open3d as o3d
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    label as connected_components,
)
from scipy.spatial import cKDTree
from skimage import measure
import trimesh

from build_v19_visible_geometry_from_sam2_depth import projected_mano_hand_silhouette

SCHEMA = "v19_hot3d_shared_observation_signed_geometry_v1"
DIRECT_POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
}
D15_POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "corrected_temporal_rigid_pose_graph",
    "completed_temporal_rigid_pose_uncertain",
}


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def load_json(path: Path, description: str = "JSON") -> dict[str, Any]:
    path = require_file(path, description)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} is not one JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def load_mesh(path: Path) -> trimesh.Trimesh:
    path = require_file(path, "mesh")
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        meshes = [
            item for item in geometry.geometry.values()
            if isinstance(item, trimesh.Trimesh) and len(item.vertices) and len(item.faces)
        ]
        if not meshes:
            raise RuntimeError(f"mesh scene has no triangles: {path}")
        geometry = trimesh.util.concatenate(meshes)
    if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return trimesh.Trimesh(
        vertices=np.asarray(geometry.vertices, dtype=np.float64),
        faces=np.asarray(geometry.faces, dtype=np.int64),
        process=False,
    )


def mesh_topology(mesh: trimesh.Trimesh) -> dict[str, Any]:
    edges = np.asarray(mesh.edges_sorted, dtype=np.int64)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    components = mesh.split(only_watertight=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "components": int(len(components)),
        "euler_number": int(mesh.euler_number),
        "area_m2": float(mesh.area),
        "volume_m3": float(abs(mesh.volume)) if mesh.is_watertight else None,
        "bounds_m": np.asarray(mesh.bounds, dtype=np.float64).astype(float).tolist(),
        "extent_m": np.asarray(mesh.extents, dtype=np.float64).astype(float).tolist(),
    }


def object_row(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    bare = object_id.split(":", 1)[-1]
    for row in frame.get("objects") or []:
        if not isinstance(row, dict):
            continue
        ids = {
            str(row.get("object_id")),
            str(row.get("track_id")),
            str(row.get("object_id")).split(":", 1)[-1],
            str(row.get("track_id")).split(":", 1)[-1],
        }
        if object_id in ids or bare in ids:
            return row
    return None


def finite_pose(row: dict[str, Any], label: str) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(row.get("rotation_world_from_completed_canonical_matrix") or [], dtype=np.float64)
    translation = np.asarray(row.get("translation_world_m") or [], dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-3, rtol=0.0)
        or float(np.linalg.det(rotation)) < 0.99
    ):
        raise RuntimeError(f"invalid {label} object pose")
    return rotation, translation


def direct_frame_ids(p14: dict[str, Any]) -> list[int]:
    ids = []
    for row in p14.get("pose_rows") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("status")) not in DIRECT_POSE_STATUSES:
            continue
        if row.get("rigid_pose_observation_eligible") is not True:
            continue
        if row.get("generated_geometry_pose_evidence_consumed") not in (None, False):
            raise RuntimeError("P14 direct pose row consumed generated geometry")
        finite_pose(row, f"P14 frame {row.get('frame_idx')}")
        ids.append(int(row["frame_idx"]))
    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError("no explicitly eligible direct P14 metric pose rows")
    return ids


def d15_pose_rows(pose: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in pose.get("pose_rows") or []:
        if not isinstance(row, dict) or str(row.get("status")) not in D15_POSE_STATUSES:
            continue
        finite_pose(row, f"D15 frame {row.get('frame_idx')}")
        frame_idx = int(row["frame_idx"])
        if frame_idx in out:
            raise RuntimeError(f"duplicate D15 pose row {frame_idx}")
        out[frame_idx] = row
    return out


def camera_pose(frame: dict[str, Any]) -> np.ndarray:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    transform = np.asarray(camera.get("T_world_camera_metric") or [], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks metric camera pose")
    return transform


def load_depth_archive(path: Path) -> dict[str, Any]:
    path = require_file(path, "active-K depth NPZ")
    with np.load(path, allow_pickle=False) as archive:
        needed = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy"}
        missing = sorted(needed - set(archive.files))
        if missing:
            raise RuntimeError(f"depth NPZ lacks {missing}: {path}")
        frame_idx = np.asarray(archive["frame_idx"], dtype=np.int64)
        depth = np.asarray(archive["depth"])
        intrinsics = np.asarray(archive["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        confidence = np.asarray(archive["confidence"]) if "confidence" in archive.files else None
        metadata = {
            key: np.asarray(archive[key]).tolist()
            for key in (
                "camera_contract_sha256",
                "camera_contract_plane",
                "depth_output_quantity",
                "camera_conditioning_mode",
                "metadata_only_ray_relabel_override",
            )
            if key in archive.files
        }
    if depth.ndim != 3 or len(depth) != len(frame_idx) or intrinsics.shape != (len(frame_idx), 4):
        raise RuntimeError(
            f"invalid depth arrays frame_idx={frame_idx.shape} depth={depth.shape} K={intrinsics.shape}"
        )
    if confidence is not None and confidence.shape != depth.shape:
        raise RuntimeError(f"confidence/depth shape mismatch: {confidence.shape} != {depth.shape}")
    if len(set(frame_idx.tolist())) != len(frame_idx):
        raise RuntimeError("depth frame_idx contains duplicates")
    return {
        "path": path,
        "frame_idx": frame_idx,
        "position": {int(value): pos for pos, value in enumerate(frame_idx.tolist())},
        "depth": depth,
        "intrinsics": intrinsics,
        "confidence": confidence,
        "metadata": metadata,
    }


def mask_affine(candidate: dict[str, Any], mask_shape: tuple[int, int], depth_shape: tuple[int, int]) -> np.ndarray:
    contract = candidate.get("mask_depth_transform_contract")
    if not isinstance(contract, dict) or contract.get("camera_contract_consistent") is not True:
        raise RuntimeError("visible geometry row lacks a camera-consistent mask/depth affine")
    A_depth_from_mask = np.asarray(contract.get("A_depth_from_mask_coordinate_model") or [], dtype=np.float64)
    if A_depth_from_mask.shape != (3, 3) or not np.isfinite(A_depth_from_mask).all():
        raise RuntimeError("invalid A_depth_from_mask_coordinate_model")
    declared_mask = tuple(int(v) for v in contract.get("mask_size_wh") or [])
    declared_depth = tuple(int(v) for v in contract.get("depth_size_wh") or [])
    actual_mask = (int(mask_shape[1]), int(mask_shape[0]))
    actual_depth = (int(depth_shape[1]), int(depth_shape[0]))
    if declared_mask != actual_mask or declared_depth != actual_depth:
        raise RuntimeError(
            f"mask/depth raster contract mismatch declared={declared_mask}/{declared_depth} actual={actual_mask}/{actual_depth}"
        )
    A_mask_from_source = np.linalg.inv(A_depth_from_mask)
    if not np.allclose(A_mask_from_source @ A_depth_from_mask, np.eye(3), atol=1.0e-10, rtol=0.0):
        raise RuntimeError("mask/depth affines are not numerically invertible")
    return A_mask_from_source


def transform_uv(uv: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[np.asarray(uv, dtype=np.float64), np.ones((len(uv),), dtype=np.float64)]
    transformed = homogeneous @ np.asarray(affine, dtype=np.float64).T
    return transformed[:, :2] / transformed[:, 2:3]


def hand_unknown_mask(
    frame: dict[str, Any],
    mask_shape: tuple[int, int],
    source_size_wh: tuple[int, int],
    A_mask_from_source: np.ndarray,
    pad_px: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    unknown = np.zeros(mask_shape, dtype=bool)
    rows = []
    for hand in frame.get("hands") or []:
        if not isinstance(hand, dict):
            continue
        silhouette, report = projected_mano_hand_silhouette(
            hand,
            mask_shape=mask_shape,
            source_width=int(source_size_wh[0]),
            source_height=int(source_size_wh[1]),
            pad_px=int(pad_px),
            A_mask_from_source=A_mask_from_source,
        )
        unknown |= silhouette
        rows.append(report)
    return unknown, rows


def canonical_observed_points(
    frames: dict[int, dict[str, Any]],
    frame_ids: list[int],
    d15: dict[int, dict[str, Any]],
    object_id: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    chunks = []
    reports = []
    for frame_idx in frame_ids:
        frame = frames.get(frame_idx)
        row = d15.get(frame_idx)
        if frame is None or row is None:
            raise RuntimeError(f"direct frame {frame_idx} missing annotation or D15 pose")
        obj = object_row(frame, object_id)
        candidate = obj.get("visible_geometry_candidate") if isinstance(obj, dict) else None
        if not isinstance(candidate, dict):
            raise RuntimeError(f"direct frame {frame_idx} lacks visible geometry candidate")
        points_world = np.asarray(candidate.get("world_vertices_sample_m") or [], dtype=np.float64)
        if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) < 32 or not np.isfinite(points_world).all():
            raise RuntimeError(f"direct frame {frame_idx} has invalid observed surfels {points_world.shape}")
        rotation, translation = finite_pose(row, f"D15 direct frame {frame_idx}")
        points_canonical = (points_world - translation[None, :]) @ rotation
        chunks.append(points_canonical)
        reports.append({"frame_idx": frame_idx, "observed_surfel_count": int(len(points_canonical))})
    return np.vstack(chunks), reports


def viewpoint_coverage(
    frames: dict[int, dict[str, Any]],
    frame_ids: list[int],
    d15: dict[int, dict[str, Any]],
    timeline_ids: list[int],
    minimum_angle_deg: float,
    temporal_bin_count: int,
) -> tuple[dict[str, Any], np.ndarray]:
    directions = []
    for frame_idx in frame_ids:
        transform = camera_pose(frames[frame_idx])
        rotation, translation = finite_pose(d15[frame_idx], f"D15 frame {frame_idx}")
        camera_canonical = (transform[:3, 3] - translation) @ rotation
        norm = float(np.linalg.norm(camera_canonical))
        if not np.isfinite(norm) or norm <= 1.0e-8:
            raise RuntimeError(f"frame {frame_idx}: degenerate canonical camera direction")
        directions.append(camera_canonical / norm)
    direction_array = np.asarray(directions, dtype=np.float64)
    cosine = np.clip(direction_array @ direction_array.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    pair_values = angles[np.triu_indices(len(frame_ids), 1)] if len(frame_ids) > 1 else np.asarray([], dtype=float)
    timeline_start, timeline_end = min(timeline_ids), max(timeline_ids)
    timeline_count = timeline_end - timeline_start + 1
    bins = np.clip(
        ((np.asarray(frame_ids) - timeline_start) * int(temporal_bin_count) // max(timeline_count, 1)).astype(int),
        0,
        int(temporal_bin_count) - 1,
    )
    gaps = np.diff(np.asarray(frame_ids, dtype=int))
    leading = frame_ids[0] - timeline_start
    trailing = timeline_end - frame_ids[-1]
    max_gap = max([int(leading), int(trailing), *[int(max(0, value - 1)) for value in gaps.tolist()]])
    report = {
        "direct_frame_count": int(len(frame_ids)),
        "direct_frame_span": [int(frame_ids[0]), int(frame_ids[-1])],
        "direct_frame_span_fraction": float((frame_ids[-1] - frame_ids[0] + 1) / max(timeline_count, 1)),
        "occupied_temporal_bin_count": int(len(set(bins.tolist()))),
        "temporal_bin_count": int(temporal_bin_count),
        "maximum_unobserved_gap_frames": int(max_gap),
        "maximum_unobserved_gap_fraction": float(max_gap / max(timeline_count, 1)),
        "viewpoint_pair_count": int(len(pair_values)),
        "viewpoint_pair_count_at_or_above_threshold": int(np.count_nonzero(pair_values >= float(minimum_angle_deg))),
        "viewpoint_angle_deg": {
            "median": float(np.median(pair_values)) if len(pair_values) else None,
            "p95": float(np.percentile(pair_values, 95.0)) if len(pair_values) else None,
            "max": float(np.max(pair_values)) if len(pair_values) else 0.0,
        },
        "minimum_separated_viewpoint_angle_deg": float(minimum_angle_deg),
        "camera_directions_object_canonical": direction_array.astype(float).tolist(),
    }
    return report, angles


def select_evidence_frames(frame_ids: list[int], angles: np.ndarray, maximum: int) -> list[int]:
    if len(frame_ids) <= int(maximum):
        return list(frame_ids)
    selected_positions: set[int] = {0, len(frame_ids) - 1}
    temporal_targets = np.linspace(0, len(frame_ids) - 1, min(8, int(maximum))).round().astype(int)
    selected_positions.update(int(value) for value in temporal_targets.tolist())
    while len(selected_positions) < int(maximum):
        selected = sorted(selected_positions)
        distance = np.min(angles[:, selected], axis=1)
        for position in selected:
            distance[position] = -1.0
        position = int(np.argmax(distance))
        if distance[position] < 0.0:
            break
        selected_positions.add(position)
    return [frame_ids[position] for position in sorted(selected_positions)]


def build_grid(points: np.ndarray, observed_mesh: trimesh.Trimesh, pitch: float, pad: float, maximum_voxels: int) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, dict[str, Any]]:
    robust_low = np.percentile(points, 0.5, axis=0)
    robust_high = np.percentile(points, 99.5, axis=0)
    low = np.minimum(robust_low, np.asarray(observed_mesh.bounds[0], dtype=np.float64)) - float(pad)
    high = np.maximum(robust_high, np.asarray(observed_mesh.bounds[1], dtype=np.float64)) + float(pad)
    extent = high - low
    if np.any(~np.isfinite(extent)) or np.any(extent <= 2.0 * float(pitch)):
        raise RuntimeError(f"invalid canonical grid bounds low={low} high={high}")
    shape = tuple(int(math.ceil(float(value) / float(pitch))) + 1 for value in extent)
    count = int(np.prod(shape))
    if count > int(maximum_voxels):
        raise RuntimeError(f"canonical voxel grid {shape} has {count} cells, above {maximum_voxels}")
    axes = [low[axis] + np.arange(shape[axis], dtype=np.float64) * float(pitch) for axis in range(3)]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid = np.c_[gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)]
    return grid, shape, low, {
        "robust_observed_bounds_m": [robust_low.astype(float).tolist(), robust_high.astype(float).tolist()],
        "grid_bounds_m": [low.astype(float).tolist(), high.astype(float).tolist()],
        "grid_extent_m": extent.astype(float).tolist(),
        "grid_shape": [int(value) for value in shape],
        "grid_voxel_count": count,
        "pitch_m": float(pitch),
        "padding_m": float(pad),
    }


def project_canonical(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    T_world_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_world = points @ rotation.T + translation[None, :]
    points_camera = (points_world - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]
    z = points_camera[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    positive = np.isfinite(points_camera).all(axis=1) & (z > 0.01)
    fx, fy, cx, cy = intrinsics.tolist()
    uv[positive, 0] = fx * points_camera[positive, 0] / z[positive] + cx
    uv[positive, 1] = fy * points_camera[positive, 1] / z[positive] + cy
    return uv, z, positive


def accepted_first_surface_depth_in_mask_plane(
    candidate: dict[str, Any],
    intrinsics: np.ndarray,
    A_mask_from_source: np.ndarray,
    mask_shape: tuple[int, int],
    radius_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    camera_points = np.asarray(candidate.get("camera_vertices_sample_m") or [], dtype=np.float64)
    if (
        camera_points.ndim != 2
        or camera_points.shape[1] != 3
        or len(camera_points) < 32
        or not np.isfinite(camera_points).all()
    ):
        raise RuntimeError(
            f"visible geometry candidate has invalid accepted first-surface camera points {camera_points.shape}"
        )
    declared_intrinsics = np.asarray(
        candidate.get("intrinsics_fx_fy_cx_cy") or [], dtype=np.float64
    )
    if (
        declared_intrinsics.shape != (4,)
        or not np.allclose(declared_intrinsics, intrinsics, atol=0.01, rtol=0.0)
    ):
        raise RuntimeError(
            "P09 accepted first-surface points are not bound to the active depth K"
        )
    z = camera_points[:, 2]
    positive = z > 0.01
    fx, fy, cx, cy = intrinsics.tolist()
    uv_source = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    uv_source[positive, 0] = fx * camera_points[positive, 0] / z[positive] + cx
    uv_source[positive, 1] = fy * camera_points[positive, 1] / z[positive] + cy
    uv_mask = transform_uv(uv_source, A_mask_from_source)
    x = np.rint(uv_mask[:, 0]).astype(np.int64)
    y = np.rint(uv_mask[:, 1]).astype(np.int64)
    valid = (
        positive
        & np.isfinite(uv_mask).all(axis=1)
        & (x >= 0)
        & (x < int(mask_shape[1]))
        & (y >= 0)
        & (y < int(mask_shape[0]))
    )
    if int(np.count_nonzero(valid)) < 32:
        raise RuntimeError("too few accepted first-surface points project into the mask plane")
    sparse = np.full(mask_shape, np.inf, dtype=np.float32)
    flat_index = y[valid] * int(mask_shape[1]) + x[valid]
    np.minimum.at(sparse.reshape(-1), flat_index, z[valid].astype(np.float32))
    seeds = np.isfinite(sparse)
    distance, indices = distance_transform_edt(
        ~seeds,
        return_distances=True,
        return_indices=True,
    )
    accepted = np.full(mask_shape, np.nan, dtype=np.float32)
    local = distance <= float(radius_px)
    nearest_depth = sparse[tuple(indices)]
    accepted[local] = nearest_depth[local]
    return accepted, {
        "accepted_first_surface_camera_point_count": int(len(camera_points)),
        "accepted_first_surface_projected_point_count": int(np.count_nonzero(valid)),
        "accepted_first_surface_unique_seed_pixel_count": int(np.count_nonzero(seeds)),
        "accepted_first_surface_local_support_pixel_count": int(np.count_nonzero(local)),
        "accepted_first_surface_local_support_radius_mask_px": float(radius_px),
        "source": "P09 visible_geometry_candidate.camera_vertices_sample_m after first_surface_depth_ownership",
    }


def frame_evidence(
    frame_idx: int,
    frame: dict[str, Any],
    object_id: str,
    pose_row: dict[str, Any],
    depth_archive: dict[str, Any],
    hand_pad_px: int,
    accepted_depth_radius_px: float,
) -> dict[str, Any]:
    position = depth_archive["position"].get(frame_idx)
    if position is None:
        raise RuntimeError(f"depth NPZ lacks direct frame {frame_idx}")
    depth = np.asarray(depth_archive["depth"][position], dtype=np.float32)
    intrinsics = np.asarray(depth_archive["intrinsics"][position], dtype=np.float64)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0.0):
        raise RuntimeError(f"frame {frame_idx}: invalid active depth K")
    obj = object_row(frame, object_id)
    candidate = obj.get("visible_geometry_candidate") if isinstance(obj, dict) else None
    if not isinstance(candidate, dict):
        raise RuntimeError(f"frame {frame_idx}: missing visible geometry candidate")
    mask_path = require_file(Path(str(obj.get("mask_path") or "")), f"frame {frame_idx} object-owned mask")
    mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_u8 is None or int(np.count_nonzero(mask_u8)) == 0:
        raise RuntimeError(f"frame {frame_idx}: unreadable/empty object-owned mask {mask_path}")
    mask = mask_u8 > 0
    A_mask_from_source = mask_affine(candidate, mask.shape, depth.shape)
    source_size = tuple(int(value) for value in candidate.get("depth_size_wh") or [depth.shape[1], depth.shape[0]])
    if source_size != (depth.shape[1], depth.shape[0]):
        source_size = (depth.shape[1], depth.shape[0])
    unknown, hand_rows = hand_unknown_mask(
        frame,
        mask.shape,
        source_size,
        A_mask_from_source,
        int(hand_pad_px),
    )
    hands = [hand for hand in frame.get("hands") or [] if isinstance(hand, dict)]
    if len(hand_rows) != len(hands) or not hand_rows:
        raise RuntimeError(f"frame {frame_idx}: projected MANO unknown masks are incomplete")
    for hand, hand_report in zip(hands, hand_rows):
        metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
        alignment = metric.get("camera_contract_alignment") if isinstance(metric.get("camera_contract_alignment"), dict) else hand.get("camera_contract_alignment") if isinstance(hand.get("camera_contract_alignment"), dict) else {}
        plane = alignment.get("hawor_camera_image_plane_contract") if isinstance(alignment.get("hawor_camera_image_plane_contract"), dict) else {}
        hand_K = np.asarray(hand_report.get("source_camera_intrinsics_fx_fy_cx_cy") or [], dtype=np.float64)
        if (
            alignment.get("active_contract_reinference_required") is not False
            or alignment.get("source_hawor_full_K_bound_to_inference") is not True
            or alignment.get("source_hawor_state_intrinsics_match_active_contract") is not True
            or plane.get("validated") is not True
            or hand_K.shape != (4,)
            or not np.allclose(hand_K, intrinsics, atol=0.01, rtol=0.0)
        ):
            raise RuntimeError(
                f"frame {frame_idx} {hand.get('hand_side')}: D15b requires active-K centered-plane MANO binding"
            )
    accepted_depth_mask, accepted_depth_report = accepted_first_surface_depth_in_mask_plane(
        candidate,
        intrinsics,
        A_mask_from_source,
        mask.shape,
        float(accepted_depth_radius_px),
    )
    ownership = candidate.get("object_surface_ownership_filter") if isinstance(candidate.get("object_surface_ownership_filter"), dict) else {}
    if ownership.get("fail_closed") is True:
        raise RuntimeError(f"frame {frame_idx}: P09 object ownership failed closed")
    rotation, translation = finite_pose(pose_row, f"D15 direct frame {frame_idx}")
    return {
        "frame_idx": frame_idx,
        "frame": frame,
        "rotation": rotation,
        "translation": translation,
        "T_world_camera": camera_pose(frame),
        "depth": depth,
        "intrinsics": intrinsics,
        "accepted_first_surface_depth_mask": accepted_depth_mask,
        "accepted_first_surface_depth_report": accepted_depth_report,
        "mask": mask,
        "mask_path": mask_path,
        "mask_sha256": sha256_file(mask_path),
        "A_mask_from_source": A_mask_from_source,
        "hand_unknown": unknown,
        "hand_rows": hand_rows,
        "object_owned_pixels": int(np.count_nonzero(mask)),
        "hand_unknown_pixels": int(np.count_nonzero(unknown)),
        "depth_position": int(position),
    }


def accumulate_volume_evidence(
    grid: np.ndarray,
    shape: tuple[int, int, int],
    evidence_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    n = len(grid)
    projection_count = np.zeros(n, dtype=np.uint16)
    silhouette_inside_count = np.zeros(n, dtype=np.uint16)
    silhouette_outside_count = np.zeros(n, dtype=np.uint16)
    behind_surface_count = np.zeros(n, dtype=np.uint16)
    free_space_count = np.zeros(n, dtype=np.uint16)
    unknown_count = np.zeros(n, dtype=np.uint16)
    frame_reports = []
    for evidence in evidence_rows:
        uv, z, positive = project_canonical(
            grid,
            evidence["rotation"],
            evidence["translation"],
            evidence["T_world_camera"],
            evidence["intrinsics"],
        )
        source_x = np.rint(uv[:, 0]).astype(np.int64)
        source_y = np.rint(uv[:, 1]).astype(np.int64)
        depth = evidence["depth"]
        source_valid = (
            positive
            & (source_x >= 0)
            & (source_x < depth.shape[1])
            & (source_y >= 0)
            & (source_y < depth.shape[0])
        )
        uv_mask = transform_uv(uv, evidence["A_mask_from_source"])
        mask_x = np.rint(uv_mask[:, 0]).astype(np.int64)
        mask_y = np.rint(uv_mask[:, 1]).astype(np.int64)
        mask = evidence["mask"]
        mask_valid = (
            source_valid
            & np.isfinite(uv_mask).all(axis=1)
            & (mask_x >= 0)
            & (mask_x < mask.shape[1])
            & (mask_y >= 0)
            & (mask_y < mask.shape[0])
        )
        ids = np.flatnonzero(mask_valid)
        sampled_depth = np.full(n, np.nan, dtype=np.float32)
        sampled_inside = np.zeros(n, dtype=bool)
        sampled_unknown = np.zeros(n, dtype=bool)
        sampled_depth[ids] = evidence["accepted_first_surface_depth_mask"][mask_y[ids], mask_x[ids]]
        sampled_inside[ids] = mask[mask_y[ids], mask_x[ids]]
        sampled_unknown[ids] = evidence["hand_unknown"][mask_y[ids], mask_x[ids]]
        finite_depth = mask_valid & sampled_inside & np.isfinite(sampled_depth) & (sampled_depth > float(args.min_depth_m))
        decisive = mask_valid & ~sampled_unknown
        inside = decisive & sampled_inside
        outside = decisive & ~sampled_inside
        behind = inside & finite_depth & (z >= sampled_depth.astype(np.float64) - float(args.depth_front_tolerance_m))
        free = decisive & finite_depth & (z < sampled_depth.astype(np.float64) - float(args.depth_front_tolerance_m))
        projection_count += decisive.astype(np.uint16)
        silhouette_inside_count += inside.astype(np.uint16)
        silhouette_outside_count += outside.astype(np.uint16)
        behind_surface_count += behind.astype(np.uint16)
        free_space_count += free.astype(np.uint16)
        unknown_count += (mask_valid & sampled_unknown).astype(np.uint16)
        frame_reports.append(
            {
                "frame_idx": int(evidence["frame_idx"]),
                "projected_decisive_voxels": int(np.count_nonzero(decisive)),
                "silhouette_inside_voxels": int(np.count_nonzero(inside)),
                "behind_first_surface_voxels": int(np.count_nonzero(behind)),
                "free_space_voxels": int(np.count_nonzero(free)),
                "hand_unknown_voxels": int(np.count_nonzero(mask_valid & sampled_unknown)),
                "object_owned_mask": str(evidence["mask_path"]),
                "object_owned_mask_sha256": evidence["mask_sha256"],
                "object_owned_pixels": evidence["object_owned_pixels"],
                "hand_unknown_pixels": evidence["hand_unknown_pixels"],
                "A_mask_from_source_coordinate_model": evidence["A_mask_from_source"].astype(float).tolist(),
                "intrinsics_fx_fy_cx_cy": evidence["intrinsics"].astype(float).tolist(),
                "accepted_first_surface_depth": evidence["accepted_first_surface_depth_report"],
            }
        )
    decisive_count = silhouette_inside_count.astype(np.int32) + silhouette_outside_count.astype(np.int32)
    inside_ratio = silhouette_inside_count.astype(np.float32) / np.maximum(decisive_count, 1)
    decisive_count = np.maximum(decisive_count, 1)
    behind_ratio = behind_surface_count.astype(np.float32) / decisive_count
    free_ratio = free_space_count.astype(np.float32) / decisive_count
    occupancy = (
        (decisive_count >= int(args.min_decisive_support_views))
        & (behind_surface_count >= int(args.min_volume_support_views))
        & (inside_ratio >= float(args.min_silhouette_inside_fraction))
        & (behind_ratio >= float(args.min_behind_surface_fraction))
        & (free_space_count <= int(args.max_free_space_contradiction_views))
        & (free_ratio <= float(args.max_free_space_fraction))
    )
    return occupancy.reshape(shape), {
        "projection_count": projection_count.reshape(shape),
        "silhouette_inside_count": silhouette_inside_count.reshape(shape),
        "silhouette_outside_count": silhouette_outside_count.reshape(shape),
        "behind_surface_count": behind_surface_count.reshape(shape),
        "free_space_count": free_space_count.reshape(shape),
        "unknown_count": unknown_count.reshape(shape),
        "silhouette_inside_fraction": inside_ratio.reshape(shape),
        "behind_surface_fraction": behind_ratio.reshape(shape),
        "free_space_fraction": free_ratio.reshape(shape),
        "decisive_view_count": decisive_count.reshape(shape),
    }, frame_reports


def retain_supported_component(occupancy: np.ndarray, grid: np.ndarray, observed_points: np.ndarray, pitch: float, origin: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    raw_count = int(np.count_nonzero(occupancy))
    if raw_count == 0:
        raise RuntimeError("multi-view mask/depth evidence produced zero occupied voxels")
    work = np.asarray(occupancy, dtype=bool).copy()
    if int(args.close_iterations) > 0:
        work = binary_closing(work, iterations=int(args.close_iterations))
    if int(args.open_iterations) > 0:
        work = binary_opening(work, iterations=int(args.open_iterations))
    labels, count = connected_components(work)
    if count == 0:
        raise RuntimeError("occupancy has no connected component after morphology")
    observed_indices = np.rint((observed_points - origin[None, :]) / float(pitch)).astype(np.int64)
    valid = np.all((observed_indices >= 0) & (observed_indices < np.asarray(work.shape)[None, :]), axis=1)
    observed_indices = observed_indices[valid]
    observed_labels = labels[tuple(observed_indices.T)] if len(observed_indices) else np.asarray([], dtype=np.int32)
    component_rows = []
    best_label = None
    best_score = None
    for component_id in range(1, count + 1):
        voxel_count = int(np.count_nonzero(labels == component_id))
        observed_hits = int(np.count_nonzero(observed_labels == component_id))
        score = (observed_hits, voxel_count)
        component_rows.append(
            {"component_id": component_id, "voxel_count": voxel_count, "observed_surfel_voxel_hits": observed_hits}
        )
        if best_score is None or score > best_score:
            best_score = score
            best_label = component_id
    assert best_label is not None
    retained = labels == int(best_label)
    retained = binary_fill_holes(retained)
    pre_erosion_count = int(np.count_nonzero(retained))
    if int(args.conservative_erosion_iterations) > 0:
        from scipy.ndimage import binary_erosion
        retained = binary_erosion(
            retained,
            iterations=int(args.conservative_erosion_iterations),
        )
    retained_count = int(np.count_nonzero(retained))
    if retained_count < int(args.min_occupied_voxels):
        raise RuntimeError(f"retained occupancy has only {retained_count} voxels")
    return retained, {
        "raw_occupied_voxels": raw_count,
        "component_count": int(count),
        "components": component_rows,
        "selected_component_id": int(best_label),
        "selected_component_score": list(best_score or (0, 0)),
        "retained_filled_voxels_before_conservative_erosion": pre_erosion_count,
        "retained_filled_voxels": retained_count,
        "conservative_erosion_iterations": int(args.conservative_erosion_iterations),
        "conservative_inward_offset_m": float(args.conservative_erosion_iterations) * float(pitch),
        "close_iterations": int(args.close_iterations),
        "open_iterations": int(args.open_iterations),
        "fill_holes": True,
    }


def occupancy_mesh(occupancy: np.ndarray, origin: np.ndarray, pitch: float, args: argparse.Namespace) -> trimesh.Trimesh:
    pad = int(args.sdf_pad_voxels)
    if pad < 2:
        raise RuntimeError("SDF marching cubes requires at least two pad voxels")
    padded = np.pad(occupancy, pad_width=pad, mode="constant", constant_values=False)
    outside = distance_transform_edt(~padded, sampling=[float(pitch)] * 3)
    inside = distance_transform_edt(padded, sampling=[float(pitch)] * 3)
    sdf = outside - inside
    if float(args.sdf_smooth_sigma_voxels) > 0.0:
        sdf = gaussian_filter(sdf, sigma=float(args.sdf_smooth_sigma_voxels), mode="nearest")
    vertices, faces, normals, _values = measure.marching_cubes(
        sdf.astype(np.float32),
        level=0.0,
        spacing=(float(pitch), float(pitch), float(pitch)),
        allow_degenerate=False,
    )
    vertices += origin[None, :] - float(pitch) * pad
    mesh = trimesh.Trimesh(
        vertices=vertices.astype(np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_normals=np.asarray(normals, dtype=np.float64),
        process=True,
    )
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    components = mesh.split(only_watertight=False)
    if not components:
        raise RuntimeError("marching-cubes proxy has no components")
    mesh = max(components, key=lambda item: float(abs(item.volume)) if item.is_watertight else float(item.area))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=True,
    )
    trimesh.repair.fix_normals(mesh, multibody=True)
    if mesh.volume < 0.0:
        mesh.invert()
    return mesh


def face_sample_points(mesh: trimesh.Trimesh, maximum_faces: int) -> tuple[np.ndarray, np.ndarray]:
    face_ids = np.arange(len(mesh.faces), dtype=np.int64)
    if len(face_ids) > int(maximum_faces):
        face_ids = np.linspace(0, len(face_ids) - 1, int(maximum_faces), dtype=np.int64)
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)[face_ids]]
    centers = triangles.mean(axis=1, keepdims=True)
    return np.concatenate([triangles, centers], axis=1), face_ids


def validate_proxy(
    mesh: trimesh.Trimesh,
    observed_points: np.ndarray,
    evidence_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    topology = mesh_topology(mesh)
    sampled_observed = observed_points[
        np.linspace(0, len(observed_points) - 1, min(len(observed_points), int(args.max_observed_validation_points)), dtype=np.int64)
    ]
    query = cKDTree(np.asarray(mesh.vertices, dtype=np.float64))
    observed_distance = query.query(sampled_observed, k=1, workers=-1)[0]
    distance_summary = {
        "count": int(len(observed_distance)),
        "median_m": float(np.median(observed_distance)),
        "p95_m": float(np.percentile(observed_distance, 95.0)),
        "max_m": float(np.max(observed_distance)),
    }
    samples, sampled_face_ids = face_sample_points(mesh, int(args.max_validation_faces))
    face_count, samples_per_face = samples.shape[:2]
    flat = samples.reshape(-1, 3)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    support_counts = np.zeros(face_count, dtype=np.uint16)
    free_counts = np.zeros(face_count, dtype=np.uint16)
    mask_counts = np.zeros(face_count, dtype=np.uint16)
    frame_rows = []
    for evidence in evidence_rows:
        uv, z, positive = project_canonical(
            flat,
            evidence["rotation"],
            evidence["translation"],
            evidence["T_world_camera"],
            evidence["intrinsics"],
        )
        source_x = np.rint(uv[:, 0]).astype(np.int64)
        source_y = np.rint(uv[:, 1]).astype(np.int64)
        depth = evidence["depth"]
        source_valid = (
            positive
            & (source_x >= 0)
            & (source_x < depth.shape[1])
            & (source_y >= 0)
            & (source_y < depth.shape[0])
        )
        uv_mask = transform_uv(uv, evidence["A_mask_from_source"])
        mask_x = np.rint(uv_mask[:, 0]).astype(np.int64)
        mask_y = np.rint(uv_mask[:, 1]).astype(np.int64)
        mask = evidence["mask"]
        mask_valid = (
            source_valid
            & np.isfinite(uv_mask).all(axis=1)
            & (mask_x >= 0)
            & (mask_x < mask.shape[1])
            & (mask_y >= 0)
            & (mask_y < mask.shape[0])
        )
        ids = np.flatnonzero(mask_valid)
        sampled_depth = np.full(len(flat), np.nan, dtype=np.float64)
        sampled_mask = np.zeros(len(flat), dtype=bool)
        sampled_unknown = np.zeros(len(flat), dtype=bool)
        sampled_depth[ids] = evidence["accepted_first_surface_depth_mask"][mask_y[ids], mask_x[ids]].astype(np.float64)
        sampled_mask[ids] = mask[mask_y[ids], mask_x[ids]]
        sampled_unknown[ids] = evidence["hand_unknown"][mask_y[ids], mask_x[ids]]
        rotation, translation = evidence["rotation"], evidence["translation"]
        camera_origin = (evidence["T_world_camera"][:3, 3] - translation) @ rotation
        rays_to_sample = flat - camera_origin[None, :]
        ray_distance = np.linalg.norm(rays_to_sample, axis=1)
        valid_ray = np.isfinite(rays_to_sample).all(axis=1) & (ray_distance > 1.0e-9)
        directions = np.zeros_like(rays_to_sample)
        directions[valid_ray] = rays_to_sample[valid_ray] / ray_distance[valid_ray, None]
        rays = np.c_[np.broadcast_to(camera_origin[None, :], flat.shape), directions].astype(np.float32)
        first_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
        frontmost = valid_ray & np.isfinite(first_hit) & (
            ray_distance <= first_hit + float(args.model_frontmost_tolerance_m)
        )
        depth_valid = mask_valid & sampled_mask & np.isfinite(sampled_depth) & (sampled_depth > float(args.min_depth_m))
        decisive = frontmost & mask_valid & ~sampled_unknown
        mask_support = decisive & sampled_mask
        depth_support = (
            mask_support
            & depth_valid
            & (np.abs(z - sampled_depth) <= float(args.depth_surface_tolerance_m))
        )
        free = (
            decisive
            & depth_valid
            & (z < sampled_depth - float(args.depth_front_tolerance_m))
        )
        support_face = np.count_nonzero(depth_support.reshape(face_count, samples_per_face), axis=1) >= int(args.min_face_samples_per_validation_view)
        free_face = np.count_nonzero(free.reshape(face_count, samples_per_face), axis=1) >= int(args.min_face_samples_per_validation_view)
        mask_face = np.count_nonzero(mask_support.reshape(face_count, samples_per_face), axis=1) >= int(args.min_face_samples_per_validation_view)
        support_counts += support_face.astype(np.uint16)
        free_counts += free_face.astype(np.uint16)
        mask_counts += mask_face.astype(np.uint16)
        decisive_count = int(np.count_nonzero(decisive))
        frame_rows.append(
            {
                "frame_idx": int(evidence["frame_idx"]),
                "model_frontmost_decisive_sample_count": decisive_count,
                "mask_supported_sample_count": int(np.count_nonzero(mask_support)),
                "depth_surface_supported_sample_count": int(np.count_nonzero(depth_support)),
                "free_space_contradicted_sample_count": int(np.count_nonzero(free)),
                "mask_supported_fraction": float(np.count_nonzero(mask_support) / max(decisive_count, 1)),
                "depth_surface_supported_fraction": float(np.count_nonzero(depth_support) / max(decisive_count, 1)),
                "free_space_contradicted_fraction": float(np.count_nonzero(free) / max(decisive_count, 1)),
            }
        )
    repeated_free = free_counts >= int(args.repeated_free_space_validation_views)
    supported_faces = support_counts > 0
    mask_supported_faces = mask_counts > 0
    repeated_free_fraction = float(np.count_nonzero(repeated_free) / max(face_count, 1))
    depth_supported_fraction = float(np.count_nonzero(supported_faces) / max(face_count, 1))
    mask_supported_fraction = float(np.count_nonzero(mask_supported_faces) / max(face_count, 1))
    topology_ok = bool(
        topology["watertight"]
        and topology["winding_consistent"]
        and topology["is_volume"]
        and topology["boundary_edges"] == 0
        and topology["nonmanifold_edges"] == 0
        and topology["components"] == 1
        and topology.get("volume_m3") is not None
        and float(topology["volume_m3"] or 0.0) >= float(args.min_proxy_volume_m3)
    )
    coverage_ok = bool(
        int(coverage["direct_frame_count"]) >= int(args.min_direct_pose_frames)
        and float(coverage["direct_frame_span_fraction"]) >= float(args.min_direct_frame_span_fraction)
        and int(coverage["occupied_temporal_bin_count"]) >= int(args.min_occupied_temporal_bins)
        and float(coverage["maximum_unobserved_gap_fraction"]) <= float(args.max_direct_gap_fraction)
        and int(coverage["viewpoint_pair_count_at_or_above_threshold"]) >= 1
    )
    distance_ok = bool(
        distance_summary["median_m"] <= float(args.max_observed_vertex_median_distance_m)
        and distance_summary["p95_m"] <= float(args.max_observed_vertex_p95_distance_m)
    )
    projection_ok = bool(
        repeated_free_fraction <= float(args.max_repeated_free_face_fraction)
        and depth_supported_fraction >= float(args.min_depth_supported_face_fraction)
        and mask_supported_fraction >= float(args.min_mask_supported_face_fraction)
    )
    signed_ready = bool(topology_ok and coverage_ok and distance_ok and projection_ok)
    return {
        "signed_geometry_ready": signed_ready,
        "topology": topology,
        "topology_gate_passed": topology_ok,
        "coverage_gate_passed": coverage_ok,
        "observed_surface_distance": distance_summary,
        "observed_surface_distance_gate_passed": distance_ok,
        "projection_gate_passed": projection_ok,
        "sampled_proxy_face_count": int(face_count),
        "sampled_proxy_face_ids_sha256": hashlib.sha256(np.asarray(sampled_face_ids, dtype="<i8").tobytes()).hexdigest(),
        "faces_with_any_depth_surface_support": int(np.count_nonzero(supported_faces)),
        "depth_surface_supported_face_fraction": depth_supported_fraction,
        "faces_with_any_mask_support": int(np.count_nonzero(mask_supported_faces)),
        "mask_supported_face_fraction": mask_supported_fraction,
        "repeated_free_space_contradicted_faces": int(np.count_nonzero(repeated_free)),
        "repeated_free_space_contradicted_face_fraction": repeated_free_fraction,
        "validation_frame_rows": frame_rows,
        "thresholds": {
            "max_observed_vertex_median_distance_m": float(args.max_observed_vertex_median_distance_m),
            "max_observed_vertex_p95_distance_m": float(args.max_observed_vertex_p95_distance_m),
            "max_repeated_free_face_fraction": float(args.max_repeated_free_face_fraction),
            "min_depth_supported_face_fraction": float(args.min_depth_supported_face_fraction),
            "min_mask_supported_face_fraction": float(args.min_mask_supported_face_fraction),
            "repeated_free_space_validation_views": int(args.repeated_free_space_validation_views),
        },
    }


def render_qc(path: Path, occupancy: np.ndarray, evidence: dict[str, np.ndarray], title: str) -> None:
    projections = []
    fields = [
        (occupancy.astype(np.float32), "occupied"),
        (evidence["behind_surface_count"].astype(np.float32), "behind support"),
        (evidence["free_space_count"].astype(np.float32), "free contradiction"),
    ]
    for volume, label in fields:
        views = []
        for axis in range(3):
            image = np.max(volume, axis=axis)
            if float(np.max(image)) > 0.0:
                image = image / float(np.max(image))
            image = cv2.resize((image * 255).astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)
            image = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
            cv2.putText(image, f"{label} axis={axis}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            views.append(image)
        projections.append(np.hstack(views))
    canvas = np.vstack(projections)
    cv2.putText(canvas, title[:140], (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write signed geometry QC {path}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotations_path = require_file(args.annotations, "P09 visible annotations")
    p14_path = require_file(args.direct_pose_report, "P14 direct pose report")
    d15_path = require_file(args.pose_report, "D15 pose authority")
    observed_completion_path = require_file(args.observed_completion_report, "D14 observed completion")
    depth_path = require_file(args.depth_npz, "active-K depth NPZ")
    annotations = load_json(annotations_path, "P09 annotations")
    p14 = load_json(p14_path, "P14 pose report")
    d15 = load_json(d15_path, "D15 pose report")
    observed_completion = load_json(observed_completion_path, "D14 completion report")
    if d15.get("annotation_ready") is not True or (d15.get("graph_support") or {}).get("sufficient") is not True:
        raise RuntimeError("D15 pose authority is not ready/sufficient")
    if (d15.get("temporal_readiness") or {}).get("ready") is not True:
        raise RuntimeError("D15 temporal readiness is not true")
    if int(d15.get("nonpenetration_target_frame_count", -1)) != 0:
        raise RuntimeError("D15 is not the unchanged observed-only pose authority")
    p14_annotation = Path(str(p14.get("annotations") or (p14.get("inputs") or {}).get("annotations") or ""))
    if p14_annotation and str(p14_annotation) and not same_path(p14_annotation, annotations_path):
        raise RuntimeError("P14 direct poses are not bound to the supplied annotations")
    outputs = observed_completion.get("outputs") if isinstance(observed_completion.get("outputs"), dict) else {}
    observed_pose_mesh_path = require_file(
        Path(str(outputs.get("pose_hypothesis_mesh_labeled") or outputs.get("completed_mesh_labeled") or "")),
        "D14 observed canonical pose mesh",
    )
    observed_collision_path = require_file(
        Path(str(outputs.get("collision_eligible_mesh_labeled") or "")),
        "D14 observed collision surface",
    )
    if not same_path(observed_pose_mesh_path, observed_collision_path):
        raise RuntimeError("D14 observed-only pose/collision files differ before signed reconstruction")
    d15_inputs = d15.get("inputs") if isinstance(d15.get("inputs"), dict) else {}
    d15_mesh = require_file(Path(str(d15_inputs.get("completed_mesh") or "")), "D15 canonical pose mesh")
    if not same_path(d15_mesh, observed_pose_mesh_path):
        raise RuntimeError("D15 canonical pose body differs from D14 observed mesh")
    d15_p14 = Path(str(d15_inputs.get("pose_report") or ""))
    if d15_p14 and str(d15_p14) and not same_path(d15_p14, p14_path):
        raise RuntimeError("D15 was not built from the supplied P14 direct pose report")

    frames_list = [row for row in annotations.get("frames") or [] if isinstance(row, dict)]
    frames = {int(row["frame_idx"]): row for row in frames_list}
    timeline_ids = sorted(frames)
    if not timeline_ids or len(timeline_ids) != len(frames_list):
        raise RuntimeError("annotation timeline is empty or has duplicate frame IDs")
    direct_ids = direct_frame_ids(p14)
    d15_rows = d15_pose_rows(d15)
    missing = sorted(set(direct_ids) - set(d15_rows))
    if missing:
        raise RuntimeError(f"D15 lacks direct P14 frame IDs: {missing[:20]}")
    observed_mesh = load_mesh(observed_pose_mesh_path)
    observed_points, observed_rows = canonical_observed_points(frames, direct_ids, d15_rows, args.object_id)
    coverage, angle_matrix = viewpoint_coverage(
        frames,
        direct_ids,
        d15_rows,
        timeline_ids,
        float(args.min_viewpoint_separation_deg),
        int(args.temporal_bin_count),
    )
    selected_ids = select_evidence_frames(direct_ids, angle_matrix, int(args.max_evidence_frames))
    depth_archive = load_depth_archive(depth_path)
    grid, grid_shape, grid_origin, grid_report = build_grid(
        observed_points,
        observed_mesh,
        float(args.pitch_m),
        float(args.grid_padding_m),
        int(args.max_grid_voxels),
    )
    evidence_rows = [
        frame_evidence(
            frame_idx,
            frames[frame_idx],
            args.object_id,
            d15_rows[frame_idx],
            depth_archive,
            int(args.hand_unknown_padding_mask_px),
            float(args.accepted_depth_support_radius_mask_px),
        )
        for frame_idx in selected_ids
    ]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "shared_multiview_signed_geometry_candidate.ply"
    selected_collision_path = output_dir / "shared_collision_eligible_surface.ply"
    labels_path = output_dir / "shared_collision_eligible_face_labels.json"
    evidence_npz_path = output_dir / "shared_signed_voxel_evidence.npz"
    qc_path = output_dir / "shared_signed_geometry_voxel_qc.jpg"
    report_path = output_dir / "shared_signed_geometry_completion_report.json"
    for path in (candidate_path, selected_collision_path, labels_path, evidence_npz_path, qc_path, report_path):
        if path.exists() and not args.replace:
            raise RuntimeError(f"refusing to overwrite signed geometry output: {path}")

    failure: dict[str, Any] | None = None
    proxy: trimesh.Trimesh | None = None
    proxy_validation: dict[str, Any] = {
        "signed_geometry_ready": False,
        "failure": "candidate_not_built",
    }
    morphology: dict[str, Any] = {}
    frame_reports: list[dict[str, Any]] = []
    evidence_volumes: dict[str, np.ndarray] = {}
    occupancy = np.zeros(grid_shape, dtype=bool)
    try:
        raw_occupancy, evidence_volumes, frame_reports = accumulate_volume_evidence(
            grid, grid_shape, evidence_rows, args
        )
        occupancy, morphology = retain_supported_component(
            raw_occupancy,
            grid,
            observed_points,
            float(args.pitch_m),
            grid_origin,
            args,
        )
        proxy = occupancy_mesh(occupancy, grid_origin, float(args.pitch_m), args)
        proxy.export(candidate_path)
        proxy_validation = validate_proxy(proxy, observed_points, evidence_rows, coverage, args)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        proxy_validation = {
            "signed_geometry_ready": False,
            "failure": failure,
        }

    signed_ready = bool(proxy is not None and proxy_validation.get("signed_geometry_ready") is True)
    selected_source = candidate_path if signed_ready else observed_collision_path
    selected_mesh = proxy if signed_ready else observed_mesh
    assert selected_mesh is not None
    selected_mesh.export(selected_collision_path)
    selected_label = (
        "shared_multiview_mask_depth_signed_proxy_surface"
        if signed_ready
        else "observed_depth_surface_unsigned_fallback"
    )
    write_json(
        labels_path,
        {
            "schema": "v19_collision_eligible_face_labels_v1",
            "source_mesh": str(selected_collision_path),
            "face_count": int(len(selected_mesh.faces)),
            "label_counts": {selected_label: int(len(selected_mesh.faces))},
            "labels": [selected_label] * int(len(selected_mesh.faces)),
        },
    )
    if evidence_volumes:
        np.savez_compressed(
            evidence_npz_path,
            occupancy=occupancy.astype(np.uint8),
            grid_origin_m=np.asarray(grid_origin, dtype=np.float64),
            pitch_m=np.asarray(float(args.pitch_m), dtype=np.float64),
            selected_frame_idx=np.asarray(selected_ids, dtype=np.int32),
            **evidence_volumes,
        )
        render_qc(
            qc_path,
            occupancy,
            evidence_volumes,
            f"{args.case} {args.object_id} | signed_ready={signed_ready}",
        )
    else:
        np.savez_compressed(
            evidence_npz_path,
            occupancy=occupancy.astype(np.uint8),
            grid_origin_m=np.asarray(grid_origin, dtype=np.float64),
            pitch_m=np.asarray(float(args.pitch_m), dtype=np.float64),
            selected_frame_idx=np.asarray(selected_ids, dtype=np.int32),
        )
        blank = np.zeros((320, 960, 3), dtype=np.uint8)
        cv2.putText(blank, f"signed candidate failed: {failure}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(qc_path), blank):
            raise RuntimeError(f"failed to write fallback QC {qc_path}")

    selected_topology = mesh_topology(selected_mesh)
    observed_topology = mesh_topology(observed_mesh)
    geometry_readiness = {
        "pose_hypothesis_available": True,
        "pose_hypothesis_mesh": str(observed_pose_mesh_path),
        "completed_mesh_legacy_field_semantics": "observed_pose_body_not_replaced_by_sign_proxy",
        "local_observed_collision_surface_available": True,
        "collision_eligible_mesh": str(selected_collision_path),
        "signed_geometry_mesh": str(selected_collision_path) if signed_ready else None,
        "collision_surface_watertight": bool(selected_topology["watertight"]),
        "collision_surface_winding_consistent": bool(selected_topology["winding_consistent"]),
        "collision_surface_is_volume": bool(selected_topology["is_volume"]),
        "signed_geometry_ready": signed_ready,
        "signed_geometry_source": "shared_prediction_mask_depth_direct_pose_voxel_reconstruction" if signed_ready else "unavailable_unsigned_observed_fallback",
        "signed_geometry_consumer_policy": "all_faces_of_validated_shared_proxy_signed_eligible" if signed_ready else "signed_queries_inactive",
        "signed_geometry_support_uncertainty_m": float(args.pitch_m) * 2.0,
        "backend_generated_geometry_consumed": False,
        "sam3d_geometry_consumed": False,
        "trellis_geometry_consumed": False,
        "generated_hidden_surface_included": False,
        "generated_faces_pose_eligible": False,
        "generated_faces_collision_eligible": False,
        "generated_faces_contact_eligible": False,
        "generated_faces_signed_distance_eligible": False,
        "annotation_ready": False,
        "signed_geometry_readiness_reason": (
            "backend-neutral mask/depth volume passed direct-pose coverage, topology, first-hit projection, observed-surface, and repeated free-space checks"
            if signed_ready
            else "shared sign proxy did not pass every topology/coverage/projection/observed-surface gate; physical consumers must remain unsigned"
        ),
    }
    report = {
        "schema": SCHEMA,
        "method": "build_hot3d_shared_signed_geometry",
        "status": "ok_shared_signed_geometry_ready" if signed_ready else "ok_unsigned_fallback_shared_signed_geometry_not_ready",
        "annotation_ready": False,
        "case": str(args.case),
        "object_id": str(args.object_id).split(":", 1)[-1],
        "claim_scope": (
            "Backend-neutral prediction-side signed collision hypothesis reconstructed from direct observed poses, object-owned masks, active-K depth, and MANO occlusion unknown regions. "
            "It does not consume SAM3D/TRELLIS generated geometry and does not alter the D15 object trajectory."
        ),
        "inputs": {
            "annotations": str(annotations_path),
            "direct_pose_report": str(p14_path),
            "pose_report": str(d15_path),
            "observed_completion_report": str(observed_completion_path),
            "observed_pose_mesh": str(observed_pose_mesh_path),
            "observed_collision_surface": str(observed_collision_path),
            "depth_npz": str(depth_path),
        },
        "provenance_sha256": {
            "annotations": sha256_file(annotations_path),
            "direct_pose_report": sha256_file(p14_path),
            "pose_report": sha256_file(d15_path),
            "observed_completion_report": sha256_file(observed_completion_path),
            "observed_pose_mesh": sha256_file(observed_pose_mesh_path),
            "depth_npz": sha256_file(depth_path),
            "selected_object_owned_masks": {
                str(row["frame_idx"]): row["mask_sha256"] for row in evidence_rows
            },
        },
        "pose_authority": {
            "source": "D15 observed-only pose graph",
            "path": str(d15_path),
            "nonpenetration_target_frame_count": 0,
            "object_pose_values_modified": False,
            "direct_frame_ids_from": "P14 explicitly eligible metric visible-depth rows",
            "transform_values_from": "same-frame D15 rows",
        },
        "direct_observations": {
            "all_direct_frame_ids": direct_ids,
            "selected_evidence_frame_ids": selected_ids,
            "selected_evidence_frame_count": int(len(selected_ids)),
            "canonical_observed_surfel_count": int(len(observed_points)),
            "per_frame_observed_surfels": observed_rows,
            "viewpoint_coverage": coverage,
        },
        "grid": grid_report,
        "volume_evidence": {
            "parameters": {
                "min_volume_support_views": int(args.min_volume_support_views),
                "min_decisive_support_views": int(args.min_decisive_support_views),
                "min_silhouette_inside_fraction": float(args.min_silhouette_inside_fraction),
                "min_behind_surface_fraction": float(args.min_behind_surface_fraction),
                "max_free_space_contradiction_views": int(args.max_free_space_contradiction_views),
                "max_free_space_fraction": float(args.max_free_space_fraction),
                "depth_front_tolerance_m": float(args.depth_front_tolerance_m),
                "hand_unknown_padding_mask_px": int(args.hand_unknown_padding_mask_px),
                "accepted_depth_support_radius_mask_px": float(args.accepted_depth_support_radius_mask_px),
                "depth_evidence_policy": "P09 accepted first-surface camera samples only; rejected/raw owned-mask depth is not signed evidence",
            },
            "morphology": morphology,
            "frame_reports": frame_reports,
        },
        "candidate_failure": failure,
        "candidate_validation": proxy_validation,
        "observed_surface_topology": observed_topology,
        "selected_collision_topology": selected_topology,
        "geometry_readiness": geometry_readiness,
        "outputs": {
            "pose_hypothesis_mesh_labeled": str(observed_pose_mesh_path),
            "completed_mesh_labeled": str(observed_pose_mesh_path),
            "collision_eligible_mesh_labeled": str(selected_collision_path),
            "collision_eligible_face_labels": str(labels_path),
            "signed_geometry_mesh": str(selected_collision_path) if signed_ready else None,
            "signed_geometry_candidate_mesh": str(candidate_path) if candidate_path.is_file() else None,
            "signed_voxel_evidence_npz": str(evidence_npz_path),
            "signed_geometry_qc": str(qc_path),
        },
        "output_sha256": {
            "collision_eligible_mesh": sha256_file(selected_collision_path),
            "collision_eligible_face_labels": sha256_file(labels_path),
            "signed_geometry_candidate_mesh": sha256_file(candidate_path) if candidate_path.is_file() else None,
            "signed_voxel_evidence_npz": sha256_file(evidence_npz_path),
            "signed_geometry_qc": sha256_file(qc_path),
        },
        "face_label_counts": {
            "collision_eligible_mesh": {selected_label: int(len(selected_mesh.faces))}
        },
        "observed_band_m": float(args.pitch_m) * 2.0,
        "accepted_body_semantics": {
            "pose_body": "D14 observed-only canonical mesh",
            "physical_collision_body": "shared signed proxy" if signed_ready else "D14 observed-only unsigned surface",
            "backend_generated_faces_consumed": False,
            "all_proxy_faces_signed_eligible": signed_ready,
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "signed_geometry_ready": signed_ready,
                "selected_collision_mesh": str(selected_collision_path),
                "candidate_validation": proxy_validation,
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--direct-pose-report", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--observed-completion-report", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pitch-m", type=float, default=0.0025)
    parser.add_argument("--grid-padding-m", type=float, default=0.015)
    parser.add_argument("--max-grid-voxels", type=int, default=3_000_000)
    parser.add_argument("--max-evidence-frames", type=int, default=36)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--depth-front-tolerance-m", type=float, default=0.008)
    parser.add_argument("--depth-surface-tolerance-m", type=float, default=0.012)
    parser.add_argument("--model-frontmost-tolerance-m", type=float, default=0.004)
    parser.add_argument("--min-volume-support-views", type=int, default=2)
    parser.add_argument("--min-decisive-support-views", type=int, default=8)
    parser.add_argument("--min-silhouette-inside-fraction", type=float, default=0.80)
    parser.add_argument("--min-behind-surface-fraction", type=float, default=0.35)
    parser.add_argument("--max-free-space-contradiction-views", type=int, default=0)
    parser.add_argument("--max-free-space-fraction", type=float, default=0.0)
    parser.add_argument("--hand-unknown-padding-mask-px", type=int, default=4)
    parser.add_argument("--accepted-depth-support-radius-mask-px", type=float, default=3.0)
    parser.add_argument("--close-iterations", type=int, default=0)
    parser.add_argument("--open-iterations", type=int, default=0)
    parser.add_argument("--conservative-erosion-iterations", type=int, default=1)
    parser.add_argument("--min-occupied-voxels", type=int, default=200)
    parser.add_argument("--sdf-pad-voxels", type=int, default=5)
    parser.add_argument("--sdf-smooth-sigma-voxels", type=float, default=0.0)
    parser.add_argument("--min-proxy-volume-m3", type=float, default=1.0e-5)
    parser.add_argument("--min-direct-pose-frames", type=int, default=8)
    parser.add_argument("--temporal-bin-count", type=int, default=5)
    parser.add_argument("--min-occupied-temporal-bins", type=int, default=3)
    parser.add_argument("--min-direct-frame-span-fraction", type=float, default=0.5)
    parser.add_argument("--max-direct-gap-fraction", type=float, default=0.35)
    parser.add_argument("--min-viewpoint-separation-deg", type=float, default=15.0)
    parser.add_argument("--max-observed-validation-points", type=int, default=20_000)
    parser.add_argument("--max-validation-faces", type=int, default=12_000)
    parser.add_argument("--min-face-samples-per-validation-view", type=int, default=2)
    parser.add_argument("--repeated-free-space-validation-views", type=int, default=2)
    parser.add_argument("--max-repeated-free-face-fraction", type=float, default=0.005)
    parser.add_argument("--min-depth-supported-face-fraction", type=float, default=0.10)
    parser.add_argument("--min-mask-supported-face-fraction", type=float, default=0.60)
    parser.add_argument("--max-observed-vertex-median-distance-m", type=float, default=0.012)
    parser.add_argument("--max-observed-vertex-p95-distance-m", type=float, default=0.025)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
