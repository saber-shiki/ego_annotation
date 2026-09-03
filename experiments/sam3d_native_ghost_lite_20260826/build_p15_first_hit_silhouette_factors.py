#!/usr/bin/env python3
"""Build frozen first-hit/silhouette image factors for the V19 P15 SE(3) graph.

The factors are image evidence, not a second canonical shape.  For every P14
visible-pose observation we render the observed-only pose-hypothesis mesh once
at a low raster resolution, then freeze three local correspondence sets:

* observed surfel pixel -> rasterized first-hit canonical point (depth),
* rendered pixel in known background -> nearest target/unknown pixel,
* observed surfel pixel without mesh coverage -> nearest rendered canonical
  point (silhouette coverage).

P15 re-linearizes those frozen canonical points under each per-frame SE(3)
correction.  Hand projections are explicit unknown support: rendered pixels in
hand regions are not treated as known background.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


POSE_MEASUREMENT_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "fit_to_object_owned_rgb_calibrated_pnp",
}


@dataclass
class FactorBuildFrame:
    frame_idx: int
    T_world_camera: np.ndarray
    rotation_world_object: np.ndarray
    translation_world_object: np.ndarray
    observed_camera: np.ndarray
    observed_uv: np.ndarray
    target_mask: np.ndarray
    target_or_unknown: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path.expanduser().resolve(), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid triangle mesh: {path}")
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise RuntimeError(f"non-finite mesh vertices: {path}")
    return mesh


def world_to_camera(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]


def camera_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]


def canonical_to_camera(
    canonical: np.ndarray,
    rotation_world_object: np.ndarray,
    translation_world_object: np.ndarray,
    T_world_camera: np.ndarray,
) -> np.ndarray:
    world = np.asarray(canonical) @ rotation_world_object.T + translation_world_object[None, :]
    return world_to_camera(world, T_world_camera)


def resize_intrinsics(K_source: np.ndarray, source_size: int, raster_size: int) -> np.ndarray:
    ratio = float(raster_size) / float(source_size)
    K = np.asarray(K_source, dtype=np.float64).copy()
    K[0, 0] *= ratio
    K[1, 1] *= ratio
    K[0, 2] = ratio * (K[0, 2] + 0.5) - 0.5
    K[1, 2] = ratio * (K[1, 2] + 0.5) - 0.5
    return K


def resize_mask(mask: np.ndarray, raster_size: int) -> np.ndarray:
    interpolation = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    return cv2.resize(mask.astype(np.uint8), (raster_size, raster_size), interpolation=interpolation) > 0


def project_camera(points_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    z = np.maximum(points[:, 2], 1.0e-9)
    return np.column_stack((
        K[0, 0] * points[:, 0] / z + K[0, 2],
        K[1, 1] * points[:, 1] / z + K[1, 2],
    ))


def deterministic_indices(count: int, max_count: int) -> np.ndarray:
    if count <= max_count:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, count - 1, num=max_count)).astype(np.int64))


class BatchFirstHitRasterizer:
    def __init__(self, faces: np.ndarray, raster_size: int, K_raster: np.ndarray, device: str):
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

        self.device = device
        self.faces = torch.tensor(faces, dtype=torch.int64, device=device)
        self.size = int(raster_size)
        cameras = cameras_from_opencv_projection(
            torch.eye(3, dtype=torch.float32, device=device)[None],
            torch.zeros(1, 3, dtype=torch.float32, device=device),
            torch.tensor(K_raster, dtype=torch.float32, device=device)[None],
            torch.tensor([[self.size, self.size]], dtype=torch.float32, device=device),
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=self.size,
                blur_radius=0.0,
                faces_per_pixel=1,
                cull_backfaces=False,
                bin_size=0,
            ),
        )

    def render(self, vertices_camera: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        from pytorch3d.structures import Meshes

        verts = [torch.tensor(v, dtype=torch.float32, device=self.device) for v in vertices_camera]
        with torch.no_grad():
            fragments = self.rasterizer(Meshes(verts=verts, faces=[self.faces] * len(verts)))
        zbuf = fragments.zbuf[..., 0].detach().cpu().numpy().astype(np.float64)
        pix = fragments.pix_to_face[..., 0].detach().cpu().numpy()
        valid = (pix >= 0) & np.isfinite(zbuf) & (zbuf > 1.0e-6)
        return np.where(valid, zbuf, np.nan), valid


def annotation_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if obj.get("object_id") == object_id:
            return obj
    legacy = frame.get("object") if isinstance(frame.get("object"), dict) else None
    if legacy and legacy.get("object_id") in (None, object_id):
        return legacy
    return None


def build_hand_unknown_mask(
    hand_npz: Any,
    hand_rows_by_frame: dict[int, list[int]],
    frame_idx: int,
    T_world_camera: np.ndarray,
    K_raster: np.ndarray,
    raster_size: int,
    dilation_px: int,
) -> np.ndarray:
    unknown = np.zeros((raster_size, raster_size), dtype=np.uint8)
    for row_pos in hand_rows_by_frame.get(int(frame_idx), []):
        vertices_world = np.asarray(
            hand_npz["vertices_current_v18_world_from_hawor_projection_relift_m"][row_pos],
            dtype=np.float64,
        )
        vertices_camera = world_to_camera(vertices_world, T_world_camera)
        valid = np.isfinite(vertices_camera).all(axis=1) & (vertices_camera[:, 2] > 1.0e-6)
        if np.count_nonzero(valid) < 3:
            continue
        uv = project_camera(vertices_camera[valid], K_raster)
        uv = uv[np.isfinite(uv).all(axis=1)]
        inside = (
            (uv[:, 0] >= 0.0) & (uv[:, 0] < raster_size)
            & (uv[:, 1] >= 0.0) & (uv[:, 1] < raster_size)
        )
        uv = uv[inside]
        if len(uv) < 3:
            continue
        hull = cv2.convexHull(np.rint(uv).astype(np.int32))
        cv2.fillConvexPoly(unknown, hull, 1)
    if dilation_px > 0:
        radius = int(dilation_px)
        unknown = cv2.dilate(
            unknown,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
        )
    return unknown > 0


def load_frames(args: argparse.Namespace) -> tuple[list[FactorBuildFrame], np.ndarray]:
    annotations = load_json(args.annotations)
    measurement_report = load_json(args.pose_report)
    factor_pose_report = load_json(args.factor_pose_report) if args.factor_pose_report else measurement_report
    measurement_rows = {
        int(row["frame_idx"]): row
        for row in measurement_report.get("pose_rows", [])
        if isinstance(row, dict) and str(row.get("status") or "") in POSE_MEASUREMENT_STATUSES
    }
    factor_pose_rows = {
        int(row["frame_idx"]): row
        for row in factor_pose_report.get("pose_rows", [])
        if isinstance(row, dict) and row.get("frame_idx") is not None
    }
    hand_npz = np.load(args.hand_npz)
    hand_rows_by_frame: dict[int, list[int]] = {}
    for pos, frame_idx in enumerate(np.asarray(hand_npz["frame_idx"]).astype(int).tolist()):
        hand_rows_by_frame.setdefault(int(frame_idx), []).append(int(pos))

    frames: list[FactorBuildFrame] = []
    K_raster: np.ndarray | None = None
    for frame in annotations.get("frames", []):
        if not isinstance(frame, dict) or frame.get("frame_idx") is None:
            continue
        idx = int(frame["frame_idx"])
        if args.frame_start is not None and idx < int(args.frame_start):
            continue
        if args.frame_end is not None and idx > int(args.frame_end):
            continue
        if idx not in measurement_rows or idx not in factor_pose_rows:
            continue
        obj = annotation_object(frame, args.object_id)
        if obj is None:
            continue
        visible = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else None
        if not isinstance(visible, dict):
            continue
        ownership = visible.get("first_surface_depth_ownership")
        if not isinstance(ownership, dict) or ownership.get("enabled") is not True or ownership.get("fail_closed") is True:
            continue
        if float(ownership.get("retained_fraction") or 0.0) < float(args.min_ownership_fraction):
            continue
        observed = np.asarray(visible.get("camera_vertices_sample_m") or [], dtype=np.float64)
        observed = observed[np.isfinite(observed).all(axis=1) & (observed[:, 2] > 1.0e-6)]
        if len(observed) < int(args.min_observed_points):
            continue
        fx, fy, cx, cy = np.asarray(visible["intrinsics_fx_fy_cx_cy"], dtype=np.float64).tolist()
        K_source = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        K_current = resize_intrinsics(K_source, int(frame.get("source_width") or args.source_size), int(args.raster_size))
        if K_raster is None:
            K_raster = K_current
        elif not np.allclose(K_current, K_raster, atol=1.0e-6):
            raise RuntimeError("per-frame intrinsics differ; factor batch contract is invalid")
        uv = project_camera(observed, K_current)
        inside = (
            (uv[:, 0] >= 0.0) & (uv[:, 0] < args.raster_size)
            & (uv[:, 1] >= 0.0) & (uv[:, 1] < args.raster_size)
        )
        observed = observed[inside]
        uv = uv[inside]
        if len(observed) < int(args.min_observed_points):
            continue
        mask_path = Path(str(visible.get("mask_path") or obj.get("mask_path") or ""))
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise RuntimeError(f"failed to read object-owned mask: {mask_path}")
        target_mask = resize_mask(mask_raw > 0, int(args.raster_size))
        T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        hand_unknown = build_hand_unknown_mask(
            hand_npz,
            hand_rows_by_frame,
            idx,
            T_world_camera,
            K_current,
            int(args.raster_size),
            int(args.hand_unknown_dilation_px),
        )
        pose_row = factor_pose_rows[idx]
        rotation = np.asarray(pose_row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        translation = np.asarray(pose_row["translation_world_m"], dtype=np.float64)
        frames.append(FactorBuildFrame(
            frame_idx=idx,
            T_world_camera=T_world_camera,
            rotation_world_object=rotation,
            translation_world_object=translation,
            observed_camera=observed,
            observed_uv=uv,
            target_mask=target_mask,
            target_or_unknown=target_mask | hand_unknown,
        ))
    if K_raster is None or len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} eligible factor frames, need {args.min_frames}")
    return frames, K_raster


def canonical_point_map(
    zbuf: np.ndarray,
    rendered: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    rotation_world_object: np.ndarray,
    translation_world_object: np.ndarray,
) -> np.ndarray:
    vv, uu = np.nonzero(rendered)
    z = zbuf[vv, uu]
    points_camera = np.column_stack((
        (uu - K[0, 2]) * z / K[0, 0],
        (vv - K[1, 2]) * z / K[1, 1],
        z,
    ))
    points_world = camera_to_world(points_camera, T_world_camera)
    points_canonical = (points_world - translation_world_object[None, :]) @ rotation_world_object
    out = np.full(zbuf.shape + (3,), np.nan, dtype=np.float64)
    out[vv, uu] = points_canonical
    return out


def build_frame_factors(
    frame: FactorBuildFrame,
    zbuf: np.ndarray,
    rendered: np.ndarray,
    K: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    point_map = canonical_point_map(
        zbuf,
        rendered,
        K,
        frame.T_world_camera,
        frame.rotation_world_object,
        frame.translation_world_object,
    )
    observed_idx = deterministic_indices(len(frame.observed_camera), int(args.max_observed_factors_per_frame))
    observed = frame.observed_camera[observed_idx]
    observed_uv = frame.observed_uv[observed_idx]
    observed_px = np.rint(observed_uv).astype(np.int64)
    observed_px[:, 0] = np.clip(observed_px[:, 0], 0, args.raster_size - 1)
    observed_px[:, 1] = np.clip(observed_px[:, 1], 0, args.raster_size - 1)
    hit = rendered[observed_px[:, 1], observed_px[:, 0]]

    depth_points = point_map[observed_px[hit, 1], observed_px[hit, 0]]
    depth_z = observed[hit, 2]

    sil_points: list[np.ndarray] = []
    sil_targets: list[np.ndarray] = []
    sil_kinds: list[int] = []

    # Rendered -> known background: nearest target-or-unknown pixel supplies the
    # local 2D silhouette direction; hand-unknown pixels are not background.
    outside = rendered & ~frame.target_or_unknown
    outside_coords = np.column_stack(np.nonzero(outside)[::-1])  # x, y
    if len(outside_coords) and np.any(frame.target_or_unknown):
        target_coords = np.column_stack(np.nonzero(frame.target_or_unknown)[::-1])
        selected = deterministic_indices(len(outside_coords), int(args.max_outside_silhouette_factors_per_frame))
        selected_coords = outside_coords[selected]
        _, nearest = cKDTree(target_coords).query(selected_coords, k=1, workers=-1)
        nearest_target = target_coords[nearest]
        points = point_map[selected_coords[:, 1], selected_coords[:, 0]]
        valid = np.isfinite(points).all(axis=1)
        sil_points.append(points[valid])
        sil_targets.append(nearest_target[valid].astype(np.float64))
        sil_kinds.append(np.zeros(int(np.count_nonzero(valid)), dtype=np.int8))

    # Observed surfel -> missing rendered coverage: use the nearest rendered
    # first-hit canonical point as a boundary correspondence.
    missing_uv = observed_uv[~hit]
    if len(missing_uv) and np.any(rendered):
        rendered_coords = np.column_stack(np.nonzero(rendered)[::-1])
        selected = deterministic_indices(len(missing_uv), int(args.max_missing_silhouette_factors_per_frame))
        selected_uv = missing_uv[selected]
        _, nearest = cKDTree(rendered_coords).query(selected_uv, k=1, workers=-1)
        nearest_rendered = rendered_coords[nearest]
        points = point_map[nearest_rendered[:, 1], nearest_rendered[:, 0]]
        valid = np.isfinite(points).all(axis=1)
        sil_points.append(points[valid])
        sil_targets.append(selected_uv[valid].astype(np.float64))
        sil_kinds.append(np.ones(int(np.count_nonzero(valid)), dtype=np.int8))

    if sil_points:
        sil_point_arr = np.vstack(sil_points)
        sil_target_arr = np.vstack(sil_targets)
        sil_kind_arr = np.concatenate(sil_kinds)
    else:
        sil_point_arr = np.empty((0, 3), dtype=np.float64)
        sil_target_arr = np.empty((0, 2), dtype=np.float64)
        sil_kind_arr = np.empty((0,), dtype=np.int8)

    intersection = int(np.count_nonzero(rendered & frame.target_mask))
    union = int(np.count_nonzero(rendered | frame.target_mask))
    metrics = {
        "frame_idx": int(frame.frame_idx),
        "observed_factor_count": int(len(depth_z)),
        "missing_silhouette_factor_count": int(np.count_nonzero(sil_kind_arr == 1)),
        "outside_silhouette_factor_count": int(np.count_nonzero(sil_kind_arr == 0)),
        "rendered_pixels": int(np.count_nonzero(rendered)),
        "target_pixels": int(np.count_nonzero(frame.target_mask)),
        "hand_unknown_pixels": int(np.count_nonzero(frame.target_or_unknown & ~frame.target_mask)),
        "initial_silhouette_iou": float(intersection / max(1, union)),
        "initial_outside_pixels": int(np.count_nonzero(outside)),
        "initial_observed_hit_fraction": float(np.mean(hit)) if len(hit) else 0.0,
        "known_pixels_contract": "hand-projected pixels are excluded from known-background factors",
    }
    return {
        "depth_points": depth_points.astype(np.float64),
        "depth_z": depth_z.astype(np.float64),
        "silhouette_points": sil_point_arr.astype(np.float64),
        "silhouette_target_uv": sil_target_arr.astype(np.float64),
        "silhouette_kind": sil_kind_arr.astype(np.int8),
    }, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-report", type=Path, required=True, help="Original P14 measurement report defining factor frames")
    p.add_argument("--factor-pose-report", type=Path, default=None, help="Optional current P15 pose report used only to freeze correspondences")
    p.add_argument("--completed-mesh", type=Path, required=True)
    p.add_argument("--hand-npz", type=Path, required=True)
    p.add_argument("--object-id", required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--raster-size", type=int, default=256)
    p.add_argument("--source-size", type=int, default=1408)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--min-observed-points", type=int, default=20)
    p.add_argument("--min-ownership-fraction", type=float, default=0.80)
    p.add_argument("--hand-unknown-dilation-px", type=int, default=1)
    p.add_argument("--max-observed-factors-per-frame", type=int, default=192)
    p.add_argument("--max-outside-silhouette-factors-per-frame", type=int, default=128)
    p.add_argument("--max-missing-silhouette-factors-per-frame", type=int, default=128)
    p.add_argument("--render-batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mesh = load_mesh(args.completed_mesh)
    frames, K_raster = load_frames(args)
    rasterizer = BatchFirstHitRasterizer(
        np.asarray(mesh.faces, dtype=np.int64),
        int(args.raster_size),
        K_raster,
        str(args.device),
    )

    frame_idx: list[int] = []
    T_world_camera_all: list[np.ndarray] = []
    depth_offsets = [0]
    silhouette_offsets = [0]
    depth_points_all: list[np.ndarray] = []
    depth_z_all: list[np.ndarray] = []
    silhouette_points_all: list[np.ndarray] = []
    silhouette_target_all: list[np.ndarray] = []
    silhouette_kind_all: list[np.ndarray] = []
    per_frame_metrics: list[dict[str, Any]] = []

    for start in range(0, len(frames), int(args.render_batch_size)):
        chunk = frames[start:start + int(args.render_batch_size)]
        vertices_camera = [
            canonical_to_camera(
                np.asarray(mesh.vertices, dtype=np.float64),
                frame.rotation_world_object,
                frame.translation_world_object,
                frame.T_world_camera,
            )
            for frame in chunk
        ]
        zbuf, rendered = rasterizer.render(vertices_camera)
        for local_i, frame in enumerate(chunk):
            factors, metrics = build_frame_factors(frame, zbuf[local_i], rendered[local_i], K_raster, args)
            frame_idx.append(int(frame.frame_idx))
            T_world_camera_all.append(frame.T_world_camera)
            depth_points_all.append(factors["depth_points"])
            depth_z_all.append(factors["depth_z"])
            silhouette_points_all.append(factors["silhouette_points"])
            silhouette_target_all.append(factors["silhouette_target_uv"])
            silhouette_kind_all.append(factors["silhouette_kind"])
            depth_offsets.append(depth_offsets[-1] + len(factors["depth_points"]))
            silhouette_offsets.append(silhouette_offsets[-1] + len(factors["silhouette_points"]))
            per_frame_metrics.append(metrics)
        print(f"[factors] rendered {min(start + len(chunk), len(frames))}/{len(frames)} frames", flush=True)

    metadata = {
        "method": "build_depth_order_p15_first_hit_silhouette_factors",
        "factor_semantics": "frozen local first-hit and silhouette correspondences for P15 per-frame SE(3) corrections",
        "annotations": str(args.annotations.expanduser().resolve()),
        "pose_report": str(args.pose_report.expanduser().resolve()),
        "factor_pose_report": str((args.factor_pose_report or args.pose_report).expanduser().resolve()),
        "completed_mesh": str(args.completed_mesh.expanduser().resolve()),
        "hand_npz": str(args.hand_npz.expanduser().resolve()),
        "object_id": str(args.object_id),
        "raster_size": int(args.raster_size),
        "hand_unknown_dilation_px": int(args.hand_unknown_dilation_px),
        "max_observed_factors_per_frame": int(args.max_observed_factors_per_frame),
        "max_outside_silhouette_factors_per_frame": int(args.max_outside_silhouette_factors_per_frame),
        "max_missing_silhouette_factors_per_frame": int(args.max_missing_silhouette_factors_per_frame),
        "generated_faces_consumed": False,
        "collision_surface_consumed": False,
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        metadata=np.asarray([json.dumps(metadata)]),
        frame_idx=np.asarray(frame_idx, dtype=np.int64),
        T_world_camera=np.asarray(T_world_camera_all, dtype=np.float64),
        K_raster=np.asarray(K_raster, dtype=np.float64),
        depth_offsets=np.asarray(depth_offsets, dtype=np.int64),
        depth_points_canonical=np.vstack(depth_points_all) if depth_points_all else np.empty((0, 3)),
        depth_observed_z=np.concatenate(depth_z_all) if depth_z_all else np.empty((0,)),
        silhouette_offsets=np.asarray(silhouette_offsets, dtype=np.int64),
        silhouette_points_canonical=np.vstack(silhouette_points_all) if silhouette_points_all else np.empty((0, 3)),
        silhouette_target_uv=np.vstack(silhouette_target_all) if silhouette_target_all else np.empty((0, 2)),
        silhouette_kind=np.concatenate(silhouette_kind_all) if silhouette_kind_all else np.empty((0,)),
    )
    summary = {
        **metadata,
        "frame_count": int(len(frame_idx)),
        "depth_factor_count": int(depth_offsets[-1]),
        "silhouette_factor_count": int(silhouette_offsets[-1]),
        "outside_silhouette_factor_count": int(np.sum([m["outside_silhouette_factor_count"] for m in per_frame_metrics])),
        "missing_silhouette_factor_count": int(np.sum([m["missing_silhouette_factor_count"] for m in per_frame_metrics])),
        "initial_silhouette_iou_median": float(np.median([m["initial_silhouette_iou"] for m in per_frame_metrics])),
        "initial_observed_hit_fraction_median": float(np.median([m["initial_observed_hit_fraction"] for m in per_frame_metrics])),
        "frames": per_frame_metrics,
        "outputs": {"factor_npz": str(args.output_npz), "factor_report": str(args.output_json)},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in [
        "frame_count",
        "depth_factor_count",
        "silhouette_factor_count",
        "initial_silhouette_iou_median",
        "initial_observed_hit_fraction_median",
        "outputs",
    ]}, indent=2))


if __name__ == "__main__":
    main()
