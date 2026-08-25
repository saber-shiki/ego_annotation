#!/usr/bin/env python3
"""Build a generated geometry prior with triangle-level visibility ordering.

Reliable observed metric surface remains authoritative. Generated candidate
triangles are tested over multiple metric-depth views by full pixel rasterization
when resolved and by vertices/edge-midpoints/centroid for subpixel support.
The adapter is model-agnostic and communicates through files so generators may
remain in isolated environments.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

SUPPORT_BARY = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ],
    dtype=np.float64,
)

COLORS = {
    "observed_depth_surface": [30, 180, 255, 255],
    "generated_hidden_surface": [70, 220, 110, 255],
    "observed_region_overwritten_candidate": [255, 200, 30, 255],
    "triangle_front_conflict_rejected": [255, 70, 180, 255],
    "triangle_visible_free_space_rejected": [235, 55, 55, 255],
    "base_visibility_rejected": [215, 90, 55, 220],
    "unsupported_uncertain": [155, 120, 210, 180],
}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        parts = [item for item in geom.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"no mesh in scene: {path}")
        geom = trimesh.util.concatenate(parts)
    if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"invalid or empty mesh: {path}")
    return trimesh.Trimesh(
        vertices=np.asarray(geom.vertices, dtype=np.float64),
        faces=np.asarray(geom.faces, dtype=np.int64),
        process=False,
    )


def load_vertices(path: Path) -> np.ndarray:
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        parts = [np.asarray(item.vertices, dtype=np.float64) for item in geom.geometry.values() if hasattr(item, "vertices")]
        if not parts:
            raise RuntimeError(f"no vertex geometry in scene: {path}")
        vertices = np.concatenate(parts, axis=0)
    elif hasattr(geom, "vertices"):
        vertices = np.asarray(geom.vertices, dtype=np.float64)
    else:
        raise RuntimeError(f"no vertices in geometry: {path}")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
        raise RuntimeError(f"invalid vertices: {path}")
    return vertices


def sample_bilinear(image: np.ndarray, uv: np.ndarray, cval: float = 0.0) -> np.ndarray:
    if len(uv) == 0:
        return np.zeros(0, dtype=np.float64)
    return map_coordinates(
        image,
        [uv[:, 1], uv[:, 0]],
        order=1,
        mode="constant",
        cval=float(cval),
        prefilter=False,
    )


def project_camera(points: np.ndarray, intrinsics: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    z = points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (z > 0.0)
    uv[valid, 0] = fx * points[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points[valid, 1] / z[valid] + cy
    return uv, z


def support_diagnostics(
    triangles_camera: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    depth: np.ndarray,
    object_interior: np.ndarray,
    object_dilated: np.ndarray,
    depth_tolerance_m: float,
) -> dict[str, np.ndarray]:
    count = len(triangles_camera)
    points = np.einsum("sj,njk->nsk", SUPPORT_BARY, triangles_camera).reshape(-1, 3)
    uv, z = project_camera(points, intrinsics)
    h, w = depth.shape
    inside = (
        (z > 0.05)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= w - 1)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= h - 1)
    )
    sampled_depth = sample_bilinear(depth.astype(np.float32, copy=False), uv)
    sampled_interior = sample_bilinear(object_interior.astype(np.float32, copy=False), uv) > 0.5
    sampled_dilated = sample_bilinear(object_dilated.astype(np.float32, copy=False), uv) > 0.5
    depth_valid = np.isfinite(sampled_depth) & (sampled_depth > 0.05)
    observed_valid = inside & sampled_interior & depth_valid
    front_margin = sampled_depth - z
    front = observed_valid & (front_margin > float(depth_tolerance_m))
    outside_object = inside & (~sampled_dilated) & depth_valid
    visible_free = outside_object & (z <= sampled_depth + float(depth_tolerance_m))
    shape = (count, len(SUPPORT_BARY))
    observed_valid_r = observed_valid.reshape(shape)
    front_r = front.reshape(shape)
    visible_free_r = visible_free.reshape(shape)
    margin_r = np.where(front_r, front_margin.reshape(shape), 0.0)
    return {
        "observed": observed_valid_r.sum(axis=1).astype(np.int16),
        "front": front_r.sum(axis=1).astype(np.int16),
        "visible_free": visible_free_r.sum(axis=1).astype(np.int16),
        "max_front_margin_m": margin_r.max(axis=1).astype(np.float32),
    }


def raster_triangle_diagnostics(
    triangle_uv: np.ndarray,
    triangle_z: np.ndarray,
    depth: np.ndarray,
    object_interior: np.ndarray,
    object_dilated: np.ndarray,
    depth_tolerance_m: float,
    min_projected_area_px2: float,
) -> tuple[int, int, int, float, int]:
    h, w = depth.shape
    if not np.isfinite(triangle_uv).all() or not np.isfinite(triangle_z).all() or np.any(triangle_z <= 0.05):
        return 0, 0, 0, 0.0, 0
    tri = np.asarray(triangle_uv, dtype=np.float64)
    denom = (
        (tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0])
        + (tri[2, 0] - tri[1, 0]) * (tri[0, 1] - tri[2, 1])
    )
    projected_area = 0.5 * abs(float(denom))
    if not np.isfinite(projected_area) or projected_area < float(min_projected_area_px2):
        return 0, 0, 0, 0.0, 0
    x0 = max(0, int(math.floor(float(np.min(tri[:, 0])))))
    y0 = max(0, int(math.floor(float(np.min(tri[:, 1])))))
    x1 = min(w, int(math.ceil(float(np.max(tri[:, 0])))) + 1)
    y1 = min(h, int(math.ceil(float(np.max(tri[:, 1])))) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0, 0.0, 0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px = xx.astype(np.float64) + 0.5
    py = yy.astype(np.float64) + 0.5
    w0 = ((tri[1, 1] - tri[2, 1]) * (px - tri[2, 0]) + (tri[2, 0] - tri[1, 0]) * (py - tri[2, 1])) / denom
    w1 = ((tri[2, 1] - tri[0, 1]) * (px - tri[2, 0]) + (tri[0, 0] - tri[2, 0]) * (py - tri[2, 1])) / denom
    w2 = 1.0 - w0 - w1
    covered = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(covered):
        return 0, 0, 0, 0.0, 0
    inv_z = w0 / triangle_z[0] + w1 / triangle_z[1] + w2 / triangle_z[2]
    valid_inv = np.isfinite(inv_z) & (inv_z > 0.0)
    z_pixel = np.full_like(inv_z, np.inf, dtype=np.float64)
    z_pixel[valid_inv] = 1.0 / inv_z[valid_inv]
    dep = depth[y0:y1, x0:x1]
    dep_valid = np.isfinite(dep) & (dep > 0.05)
    observed = covered & valid_inv & object_interior[y0:y1, x0:x1] & dep_valid
    margin = dep - z_pixel
    front = observed & (margin > float(depth_tolerance_m))
    outside = covered & valid_inv & (~object_dilated[y0:y1, x0:x1]) & dep_valid
    visible_free = outside & (z_pixel <= dep + float(depth_tolerance_m))
    max_margin = float(np.max(margin[front])) if np.any(front) else 0.0
    return (
        int(np.count_nonzero(observed)),
        int(np.count_nonzero(front)),
        int(np.count_nonzero(visible_free)),
        max_margin,
        int(np.count_nonzero(covered & valid_inv)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-mesh", type=Path, required=True)
    parser.add_argument("--observed-mesh", type=Path, required=True)
    parser.add_argument("--observed-points", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--base-completion-report", type=Path, required=True)
    parser.add_argument("--base-candidate-face-labels", type=Path)
    parser.add_argument(
        "--free-space-policy",
        choices=("triangle-multiview", "base-labels"),
        default="triangle-multiview",
        help="Use new multi-view scene free-space culling, or preserve eligibility from a prior shared face-center adapter while upgrading only front-depth ordering.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--conditioning-id", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 30, 60, 90, 109, 120, 135, 149])
    parser.add_argument("--anchor-frame", type=int, default=109)
    parser.add_argument("--observed-band-m", type=float, default=0.012119223035953178)
    parser.add_argument("--planar-slab-half-width-m", type=float, default=0.03376026587769261)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.005)
    parser.add_argument("--mask-interior-px", type=float, default=3.0)
    parser.add_argument("--free-space-mask-dilate-px", type=int, default=16)
    parser.add_argument("--min-conflicting-views", type=int, default=2)
    parser.add_argument("--min-raster-conflict-pixels", type=int, default=2)
    parser.add_argument("--min-support-conflict-samples", type=int, default=2)
    parser.add_argument("--strong-single-support-extra-margin-m", type=float, default=0.003)
    parser.add_argument("--min-projected-area-px2", type=float, default=0.25)
    parser.add_argument("--no-anchor-authoritative", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = load_mesh(args.candidate_mesh)
    observed = load_mesh(args.observed_mesh)
    observed_points = load_vertices(args.observed_points)
    vertices = np.asarray(candidate.vertices, dtype=np.float64)
    faces = np.asarray(candidate.faces, dtype=np.int64)
    centers = np.asarray(candidate.triangles_center, dtype=np.float64)
    nearest_distance = cKDTree(observed_points).query(centers, k=1, workers=-1)[0]
    near_observed = nearest_distance <= float(args.observed_band_m)
    plane_center = observed_points.mean(axis=0)
    _, _, vh = np.linalg.svd(observed_points - plane_center, full_matrices=False)
    plane_normal = vh[-1]
    plane_distance = np.abs((centers - plane_center) @ plane_normal)
    planar_supported = plane_distance <= float(args.planar_slab_half_width_m)
    test_face_ids = np.flatnonzero((~near_observed) & planar_supported)

    pose_data = load_json(args.pose_report)
    annotation_data = load_json(args.annotations)
    manifest_data = load_json(args.manifest)
    poses = {int(row["frame_idx"]): row for row in pose_data.get("pose_rows", [])}
    annotations = {int(row["frame_idx"]): row for row in annotation_data.get("frames", [])}
    entries = {int(row["frame_idx"]): row for row in manifest_data.get("frames", [])}
    missing = [frame for frame in args.frames if frame not in poses or frame not in annotations or frame not in entries]
    if missing:
        raise RuntimeError(f"frames missing from pose/annotations/manifest: {missing}")
    depth_blob = np.load(args.metric_depth_npz)
    depth_indices = np.asarray(depth_blob["frame_idx"], dtype=np.int64)
    depth_positions = []
    for frame in args.frames:
        hit = np.flatnonzero(depth_indices == int(frame))
        if len(hit) != 1:
            raise RuntimeError(f"depth archive frame lookup failed for {frame}: {len(hit)} matches")
        depth_positions.append(int(hit[0]))
    depths = np.asarray(depth_blob["depth"][depth_positions], dtype=np.float32)
    depth_blob.close()

    face_count = len(faces)
    front_view_count = np.zeros(face_count, dtype=np.uint8)
    free_view_count = np.zeros(face_count, dtype=np.uint8)
    anchor_front = np.zeros(face_count, dtype=bool)
    anchor_free = np.zeros(face_count, dtype=bool)
    support_observed_total = np.zeros(face_count, dtype=np.int32)
    support_front_total = np.zeros(face_count, dtype=np.int32)
    support_free_total = np.zeros(face_count, dtype=np.int32)
    raster_observed_total = np.zeros(face_count, dtype=np.int32)
    raster_front_total = np.zeros(face_count, dtype=np.int32)
    raster_free_total = np.zeros(face_count, dtype=np.int32)
    raster_covered_total = np.zeros(face_count, dtype=np.int32)
    max_front_margin_m = np.zeros(face_count, dtype=np.float32)
    per_frame: list[dict[str, Any]] = []

    for depth_position, frame in enumerate(args.frames):
        frame_started = time.time()
        pose = poses[int(frame)]
        rotation = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        translation = np.asarray(pose["translation_world_m"], dtype=np.float64)
        T_world_camera = np.asarray(annotations[int(frame)]["camera"]["T_world_camera_metric"], dtype=np.float64)
        T_camera_world = np.linalg.inv(T_world_camera)
        world_vertices = vertices @ rotation.T + translation
        camera_vertices = (T_camera_world @ np.c_[world_vertices, np.ones(len(world_vertices))].T).T[:, :3]
        intrinsics = tuple(float(value) for value in entries[int(frame)]["intrinsics_fx_fy_cx_cy"])
        vertex_uv, vertex_z = project_camera(camera_vertices, intrinsics)
        depth = depths[depth_position]
        mask = cv2.imread(str(entries[int(frame)]["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask for frame {frame}: {entries[int(frame)]['mask']}")
        if depth.shape != mask.shape:
            raise RuntimeError(f"frame {frame}: depth {depth.shape} != mask {mask.shape}")
        object_mask = mask > 0
        distance = cv2.distanceTransform(object_mask.astype(np.uint8), cv2.DIST_L2, 3)
        object_interior = distance > float(args.mask_interior_px)
        radius = max(0, int(args.free_space_mask_dilate_px))
        if radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            object_dilated = cv2.dilate(object_mask.astype(np.uint8), kernel) > 0
        else:
            object_dilated = object_mask

        triangles_camera = camera_vertices[faces[test_face_ids]]
        support = support_diagnostics(
            triangles_camera,
            intrinsics,
            depth,
            object_interior,
            object_dilated,
            float(args.depth_tolerance_m),
        )
        support_observed_total[test_face_ids] += support["observed"].astype(np.int32)
        support_front_total[test_face_ids] += support["front"].astype(np.int32)
        support_free_total[test_face_ids] += support["visible_free"].astype(np.int32)
        max_front_margin_m[test_face_ids] = np.maximum(max_front_margin_m[test_face_ids], support["max_front_margin_m"])

        local_count = len(test_face_ids)
        raster_observed = np.zeros(local_count, dtype=np.int32)
        raster_front = np.zeros(local_count, dtype=np.int32)
        raster_free = np.zeros(local_count, dtype=np.int32)
        raster_covered = np.zeros(local_count, dtype=np.int32)
        raster_margin = np.zeros(local_count, dtype=np.float32)
        for local_index, face_id in enumerate(test_face_ids):
            observed_px, front_px, free_px, margin_m, covered_px = raster_triangle_diagnostics(
                vertex_uv[faces[int(face_id)]],
                vertex_z[faces[int(face_id)]],
                depth,
                object_interior,
                object_dilated,
                float(args.depth_tolerance_m),
                float(args.min_projected_area_px2),
            )
            raster_observed[local_index] = observed_px
            raster_front[local_index] = front_px
            raster_free[local_index] = free_px
            raster_covered[local_index] = covered_px
            raster_margin[local_index] = margin_m
        raster_observed_total[test_face_ids] += raster_observed
        raster_front_total[test_face_ids] += raster_front
        raster_free_total[test_face_ids] += raster_free
        raster_covered_total[test_face_ids] += raster_covered
        max_front_margin_m[test_face_ids] = np.maximum(max_front_margin_m[test_face_ids], raster_margin)

        support_front = support["front"].astype(np.int32)
        support_free = support["visible_free"].astype(np.int32)
        strong_single = (support_front >= 1) & (
            support["max_front_margin_m"]
            >= float(args.depth_tolerance_m) + float(args.strong_single_support_extra_margin_m)
        )
        front_conflict_local = (
            (raster_front >= int(args.min_raster_conflict_pixels))
            | (support_front >= int(args.min_support_conflict_samples))
            | strong_single
        )
        free_conflict_local = (
            (raster_free >= int(args.min_raster_conflict_pixels))
            | (support_free >= int(args.min_support_conflict_samples))
        )
        front_view_count[test_face_ids] += front_conflict_local.astype(np.uint8)
        free_view_count[test_face_ids] += free_conflict_local.astype(np.uint8)
        if int(frame) == int(args.anchor_frame):
            anchor_front[test_face_ids] = front_conflict_local
            anchor_free[test_face_ids] = free_conflict_local
        per_frame.append(
            {
                "frame_idx": int(frame),
                "tested_faces": int(local_count),
                "front_conflicting_faces": int(np.count_nonzero(front_conflict_local)),
                "visible_free_space_conflicting_faces": int(np.count_nonzero(free_conflict_local)),
                "support_observed_samples": int(np.sum(support["observed"])),
                "support_front_samples": int(np.sum(support["front"])),
                "raster_covered_pixels": int(np.sum(raster_covered)),
                "raster_observed_pixels": int(np.sum(raster_observed)),
                "raster_front_pixels": int(np.sum(raster_front)),
                "elapsed_s": float(time.time() - frame_started),
            }
        )
        print(json.dumps(per_frame[-1]), flush=True)

    consensus_front = front_view_count >= int(args.min_conflicting_views)
    consensus_free = free_view_count >= int(args.min_conflicting_views)
    if args.no_anchor_authoritative:
        reject_front = consensus_front
        triangle_reject_free = consensus_free
    else:
        reject_front = anchor_front | consensus_front
        triangle_reject_free = anchor_free | consensus_free
    base_eligible = None
    if args.free_space_policy == "base-labels":
        if args.base_candidate_face_labels is None:
            raise RuntimeError("--free-space-policy base-labels requires --base-candidate-face-labels")
        base_sidecar = load_json(args.base_candidate_face_labels)
        raw_labels = base_sidecar.get("labels")
        if not isinstance(raw_labels, list) or len(raw_labels) != face_count:
            raise RuntimeError(
                f"base candidate label count mismatch: expected {face_count}, got "
                f"{len(raw_labels) if isinstance(raw_labels, list) else 'non-list'}"
            )
        eligible_names = {
            "generated_inferred_hidden_surface",
            "sam3d_inferred_hidden_surface",
            "trellis_inferred_hidden_surface",
            "depth_front_conflict_rejected",
        }
        base_eligible = np.asarray([str(label) in eligible_names for label in raw_labels], dtype=bool)
        reject_free = (~base_eligible) & (~near_observed) & planar_supported
        accepted_hidden = base_eligible & (~near_observed) & planar_supported & (~reject_front)
    else:
        reject_free = triangle_reject_free
        accepted_hidden = (~near_observed) & planar_supported & (~reject_front) & (~reject_free)

    labels: list[str] = []
    for face_index, (near, planar, front, free) in enumerate(zip(near_observed, planar_supported, reject_front, reject_free)):
        if near:
            labels.append("observed_region_overwritten_candidate")
        elif not planar:
            labels.append("unsupported_uncertain")
        elif front and (base_eligible is None or bool(base_eligible[face_index])):
            labels.append("triangle_front_conflict_rejected")
        elif free:
            labels.append("base_visibility_rejected" if args.free_space_policy == "base-labels" else "triangle_visible_free_space_rejected")
        else:
            labels.append("generated_hidden_surface")
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    candidate.visual.face_colors = np.asarray([COLORS[label] for label in labels], dtype=np.uint8)
    candidate_path = args.output_dir / "aligned_candidate_triangle_gate_labeled.ply"
    candidate.export(str(candidate_path))
    hidden_mesh = candidate.submesh([np.flatnonzero(accepted_hidden)], append=True, repair=False)
    completed = trimesh.util.concatenate([observed, hidden_mesh])
    completed_labels = ["observed_depth_surface"] * len(observed.faces) + ["generated_hidden_surface"] * len(hidden_mesh.faces)
    completed.visual.face_colors = np.asarray([COLORS[label] for label in completed_labels], dtype=np.uint8)
    completed_path = args.output_dir / "completed_mesh_triangle_gated_labeled.ply"
    completed.export(str(completed_path))

    write_json(
        args.output_dir / "candidate_face_labels.json",
        {
            "source_mesh": str(args.candidate_mesh),
            "face_count": int(face_count),
            "label_counts": counts,
            "labels": labels,
        },
    )
    write_json(
        args.output_dir / "completed_face_labels.json",
        {
            "source_mesh": str(completed_path),
            "face_count": int(len(completed_labels)),
            "label_counts": {
                "observed_depth_surface": int(len(observed.faces)),
                "generated_hidden_surface": int(len(hidden_mesh.faces)),
            },
            "labels": completed_labels,
        },
    )
    np.savez_compressed(
        args.output_dir / "triangle_gate_face_diagnostics.npz",
        nearest_observed_m=nearest_distance.astype(np.float32),
        plane_abs_distance_m=plane_distance.astype(np.float32),
        front_view_count=front_view_count,
        free_view_count=free_view_count,
        anchor_front=anchor_front,
        anchor_free=anchor_free,
        support_observed_total=support_observed_total,
        support_front_total=support_front_total,
        support_free_total=support_free_total,
        raster_observed_total=raster_observed_total,
        raster_front_total=raster_front_total,
        raster_free_total=raster_free_total,
        raster_covered_total=raster_covered_total,
        max_front_margin_m=max_front_margin_m,
        accepted_hidden=accepted_hidden,
    )

    base = load_json(args.base_completion_report)
    outputs = base.setdefault("outputs", {})
    outputs["completed_mesh_labeled"] = str(completed_path)
    outputs["completed_face_labels"] = str(args.output_dir / "completed_face_labels.json")
    outputs["aligned_candidate_triangle_gate_labeled"] = str(candidate_path)
    base["method"] = "build_v19_triangle_visibility_gated_geometry_prior"
    base["source_model"] = str(args.source_model)
    base["conditioning_id"] = str(args.conditioning_id)
    base["claim_scope"] = (
        "Observed metric surface is fixed and authoritative. Generated hidden triangles are accepted only after "
        "multi-view perspective-correct triangle raster/support depth ordering and scene-visibility filtering."
    )
    base["triangle_visibility_gate"] = {
        "frames": [int(frame) for frame in args.frames],
        "anchor_frame": int(args.anchor_frame),
        "anchor_authoritative": not bool(args.no_anchor_authoritative),
        "depth_tolerance_m": float(args.depth_tolerance_m),
        "mask_interior_px": float(args.mask_interior_px),
        "free_space_mask_dilate_px": int(args.free_space_mask_dilate_px),
        "min_conflicting_views": int(args.min_conflicting_views),
        "min_raster_conflict_pixels": int(args.min_raster_conflict_pixels),
        "min_support_conflict_samples": int(args.min_support_conflict_samples),
        "strong_single_support_extra_margin_m": float(args.strong_single_support_extra_margin_m),
        "min_projected_area_px2": float(args.min_projected_area_px2),
        "surface_support": "full covered pixels with perspective-correct inverse-z interpolation plus 3 vertices, 3 edge midpoints and centroid",
        "scene_visibility": "outside dilated object mask is rejected only where generated depth would be visible against valid closer/full-frame scene depth" if args.free_space_policy == "triangle-multiview" else "eligibility is preserved from the supplied shared face-center label sidecar; this run upgrades front-depth ordering only",
        "free_space_policy": str(args.free_space_policy),
        "base_candidate_face_labels": str(args.base_candidate_face_labels) if args.base_candidate_face_labels is not None else None,
        "limitations": [
            "whole-face culling rather than geometric triangle clipping/subdivision",
            "camera calibration and UniDepth are hypotheses rather than GT RGB-D",
            "hand/object depth boundaries remain uncertain",
        ],
    }
    base["accepted_body_semantics"] = {
        "observed_depth_surface_faces_accepted": int(len(observed.faces)),
        "generated_hidden_surface_faces_accepted": int(len(hidden_mesh.faces)),
        "triangle_front_conflict_faces_rejected": int(np.count_nonzero((~near_observed) & planar_supported & reject_front)),
        "triangle_visible_free_space_faces_rejected": int(np.count_nonzero((~near_observed) & planar_supported & (~reject_front) & reject_free)),
        "unsupported_uncertain_is_not_object_body": True,
    }
    base.setdefault("face_label_counts", {})["triangle_gated_candidate"] = counts
    base["face_label_counts"]["completed_mesh"] = {
        "observed_depth_surface": int(len(observed.faces)),
        "generated_hidden_surface": int(len(hidden_mesh.faces)),
    }
    completion_report_path = args.output_dir / "completion_report.json"
    write_json(completion_report_path, base)

    build_report = {
        "status": "ok",
        "method": "build_v19_triangle_visibility_gated_geometry_prior",
        "source_model": str(args.source_model),
        "conditioning_id": str(args.conditioning_id),
        "counts": {
            "input_candidate_faces": int(face_count),
            **counts,
            "accepted_observed_faces": int(len(observed.faces)),
            "accepted_hidden_faces": int(len(hidden_mesh.faces)),
            "completed_faces": int(len(completed.faces)),
            "completed_vertices": int(len(completed.vertices)),
        },
        "policy_sensitivity": {
            "strict_any_view_front_rejected": int(np.count_nonzero((~near_observed) & planar_supported & (front_view_count >= 1))),
            "consensus_front_rejected": int(np.count_nonzero((~near_observed) & planar_supported & consensus_front)),
            "anchor_or_consensus_front_rejected": int(np.count_nonzero((~near_observed) & planar_supported & (anchor_front | consensus_front))),
            "strict_any_view_visible_free_rejected": int(np.count_nonzero((~near_observed) & planar_supported & (free_view_count >= 1))),
            "consensus_visible_free_rejected": int(np.count_nonzero((~near_observed) & planar_supported & consensus_free)),
            "anchor_or_consensus_visible_free_rejected": int(np.count_nonzero((~near_observed) & planar_supported & (anchor_free | consensus_free))),
        },
        "per_frame": per_frame,
        "gate": base["triangle_visibility_gate"],
        "outputs": {
            "candidate_mesh": str(candidate_path),
            "completed_mesh": str(completed_path),
            "completion_report": str(completion_report_path),
            "face_diagnostics": str(args.output_dir / "triangle_gate_face_diagnostics.npz"),
        },
        "elapsed_s": float(time.time() - started),
    }
    write_json(args.output_dir / "build_report.json", build_report)
    print(json.dumps({key: value for key, value in build_report.items() if key != "per_frame"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
