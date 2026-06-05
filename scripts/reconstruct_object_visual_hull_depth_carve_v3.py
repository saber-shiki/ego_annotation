#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter
from scipy.spatial.transform import Rotation
from skimage import measure
import trimesh

from close_mesh_archive_with_voxel_fill_v3 import save_archive, topology, transform_points
from fuse_observed_surface_with_complete_prior_v3 import compute_frame_pose, intrinsics_for, read_mask
from diagnose_contact_kinematics_v3 import selected_vertex_ids
from optimize_contact_patch_object_pose_graph_v3 import contact_rows, hand_vertices_camera
from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame, load_depth_archive, manifest_by_frame
from optimize_mesh_prior_pose_graph_v3 import load_mesh
from render_bundlesdf_mesh_qc_v3 import load_mesh_archive


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def project(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    valid = z > 0.0
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    fx, fy, cx, cy = intrinsics.astype(np.float64).tolist()
    uv[valid, 0] = fx * points[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points[valid, 1] / z[valid] + cy
    return uv, valid


def carve_volume(
    grid_points: np.ndarray,
    grid_shape: tuple[int, int, int],
    frame_rows: list[dict],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    keep_votes = np.zeros(len(grid_points), dtype=np.int16)
    depth_hits = np.zeros(len(grid_points), dtype=np.int16)
    rows = []
    for row in frame_rows:
        r = row["rotation_prior_to_camera"]
        t = row["translation_prior_to_camera"]
        points_camera = grid_points @ r.T + t[None, :]
        uv, positive = project(points_camera, row["intrinsics"])
        xy = np.rint(uv).astype(np.int64)
        in_bounds = (
            positive
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < row["mask"].shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < row["mask"].shape[0])
        )
        frame_keep = np.zeros(len(grid_points), dtype=bool)
        if np.any(in_bounds):
            x = xy[in_bounds, 0]
            y = xy[in_bounds, 1]
            in_mask = row["mask"][y, x]
            depth = row["depth_m"][y, x].astype(np.float64)
            z = points_camera[in_bounds, 2]
            depth_valid = np.isfinite(depth) & (depth > float(args.min_depth_m))
            near_or_behind = z >= depth - float(args.depth_front_tolerance_m)
            not_too_far = z <= depth + float(args.depth_back_tolerance_m)
            accepted = in_mask & depth_valid & near_or_behind & not_too_far
            frame_keep[in_bounds] = accepted
            depth_hit = np.zeros(len(grid_points), dtype=bool)
            depth_hit[in_bounds] = in_mask & depth_valid
            depth_hits += depth_hit.astype(np.int16)
        keep_votes += frame_keep.astype(np.int16)
        rows.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "mask_depth_voxels": int(np.count_nonzero(frame_keep)),
                "visible_depth_voxels": int(np.count_nonzero(in_bounds)),
            }
        )
    min_votes = int(args.min_carve_votes)
    min_depth_hits = int(args.min_depth_hits)
    occupied = (keep_votes >= min_votes) & (depth_hits >= min_depth_hits)
    return occupied.reshape(grid_shape), rows


def contact_constraints_prior(
    annotations: dict[int, dict],
    contact_report: Path | None,
    frame_rows: list[dict],
) -> np.ndarray:
    if contact_report is None:
        return np.zeros((0, 3), dtype=np.float64)
    pose_by_frame = {int(row["frame_idx"]): row for row in frame_rows}
    points = []
    for row in contact_rows(contact_report):
        frame_idx = int(row["frame_idx"])
        if frame_idx not in pose_by_frame or frame_idx not in annotations:
            continue
        hand_idx = int(row["hand_idx"])
        hand = annotations[frame_idx]["hands"][hand_idx]
        vertices = hand_vertices_camera(hand)
        patch_ids = selected_vertex_ids(row)
        if int(patch_ids.max()) >= len(vertices):
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} patch id exceeds MANO vertex count")
        patch_camera = vertices[patch_ids]
        pose = pose_by_frame[frame_idx]
        r = pose["rotation_prior_to_camera"]
        t = pose["translation_prior_to_camera"]
        points.append((patch_camera - t[None, :]) @ r)
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(points).astype(np.float64)


def apply_contact_constraints(
    occupied: np.ndarray,
    grid_points: np.ndarray,
    contact_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    if len(contact_points) == 0:
        return occupied, {"contact_points": 0}
    diff = grid_points[:, None, :] - contact_points[None, :, :]
    nearest = np.linalg.norm(diff, axis=2).min(axis=1)
    protect = nearest <= float(args.contact_protect_radius_m)
    flat = occupied.reshape(-1).copy()
    flat[protect] = True
    return flat.reshape(occupied.shape), {
        "contact_points": int(len(contact_points)),
        "protected_voxels": int(np.count_nonzero(protect)),
        "contact_protect_radius_m": float(args.contact_protect_radius_m),
    }


def mesh_from_occupancy(
    occupied: np.ndarray,
    origin: np.ndarray,
    pitch_m: float,
    args: argparse.Namespace,
) -> trimesh.Trimesh:
    if int(np.count_nonzero(occupied)) == 0:
        raise RuntimeError("depth-carved occupancy is empty")
    occ = binary_closing(occupied, iterations=int(args.close_iterations))
    pad = int(args.sdf_pad_voxels)
    if pad < 2:
        raise RuntimeError("SDF extraction requires at least two pad voxels")
    occ_pad = np.pad(occ, pad_width=pad, mode="constant", constant_values=False)
    outside = distance_transform_edt(~occ_pad, sampling=[float(pitch_m)] * 3)
    inside = distance_transform_edt(occ_pad, sampling=[float(pitch_m)] * 3)
    sdf = outside - inside
    if float(args.sdf_smooth_sigma_voxels) > 0.0:
        sdf = gaussian_filter(sdf, sigma=float(args.sdf_smooth_sigma_voxels), mode="nearest")
    vertices, faces, normals, _values = measure.marching_cubes(
        sdf.astype(np.float32),
        level=0.0,
        spacing=(float(pitch_m), float(pitch_m), float(pitch_m)),
        allow_degenerate=False,
    )
    vertices = vertices + origin[None, :] - float(pitch_m) * pad
    mesh = trimesh.Trimesh(
        vertices=vertices.astype(np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        vertex_normals=np.asarray(normals, dtype=np.float64),
        process=True,
    )
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    components = mesh.split(only_watertight=False)
    if not components:
        raise RuntimeError("depth-carved mesh has no connected components")
    mesh = max(components, key=lambda component: float(component.area))
    trimesh.repair.fix_normals(mesh)
    topo = topology(mesh)
    if not topo["watertight"] or topo["boundary_edges"] != 0 or topo["nonmanifold_edges"] != 0:
        raise RuntimeError(f"depth-carved mesh is not topologically closed: {topo}")
    return mesh


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    depths = load_depth_archive(args.metric_depth_npz)
    graph_meshes = load_mesh_archive(args.graph_mesh_archive)
    prior_mesh = load_mesh(args.mesh_prior_camera)
    prior_vertices = np.asarray(prior_mesh.vertices, dtype=np.float64)
    bounds = prior_mesh.bounds.astype(np.float64)
    pad = float(args.grid_pad_m)
    lo = bounds[0] - pad
    hi = bounds[1] + pad
    pitch = float(args.pitch_m)
    axes = [np.arange(lo[i], hi[i] + 0.5 * pitch, pitch, dtype=np.float64) for i in range(3)]
    grid_shape = tuple(int(len(axis)) for axis in axes)
    if int(np.prod(grid_shape)) > int(args.max_grid_voxels):
        raise RuntimeError(f"grid has {int(np.prod(grid_shape))} voxels, above max {args.max_grid_voxels}")
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid_points = np.c_[gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)]
    frame_rows = []
    pose_internal_rows = []
    pose_rows = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in annotations or frame_idx not in manifest or frame_idx not in depths or frame_idx not in graph_meshes:
            continue
        depth_m, depth_intrinsics = depths[frame_idx]
        annotation = annotations[frame_idx]
        intrinsics = intrinsics_for(annotation, depth_intrinsics, str(args.intrinsics_source))
        mask = read_mask(Path(manifest[frame_idx]["mask"]), depth_m.shape)
        r, t, pose_row = compute_frame_pose(
            prior_vertices,
            graph_meshes[frame_idx][0],
            annotation,
            int(args.max_pose_correspondences),
            int(args.seed) + frame_idx,
        )
        frame_rows.append(
            {
                "frame_idx": int(frame_idx),
                "rotation_prior_to_camera": r,
                "translation_prior_to_camera": t,
                "intrinsics": intrinsics,
                "mask": mask,
                "depth_m": depth_m,
            }
        )
        pose_rows.append(
            {
                "frame_idx": int(frame_idx),
                "object_translation_camera_m": t.astype(float).tolist(),
                "object_rotation_delta_rad": Rotation.from_matrix(r).as_rotvec().astype(float).tolist(),
                **pose_row,
            }
        )
        pose_internal_rows.append(
            {
                "frame_idx": int(frame_idx),
                "rotation_prior_to_camera": r,
                "translation_prior_to_camera": t,
            }
        )
    if len(frame_rows) < int(args.min_frames):
        raise RuntimeError(f"only {len(frame_rows)} carving frames available")
    occupied, carve_rows = carve_volume(grid_points, grid_shape, frame_rows, args)
    contact_points = contact_constraints_prior(annotations, args.contact_report, pose_internal_rows)
    occupied, contact_constraint_row = apply_contact_constraints(occupied, grid_points, contact_points, args)
    mesh_prior = mesh_from_occupancy(occupied, lo, pitch, args)
    meshes_world = []
    frame_ids = []
    for row in frame_rows:
        vertices_camera = np.asarray(mesh_prior.vertices, dtype=np.float64) @ row["rotation_prior_to_camera"].T + row["translation_prior_to_camera"][None, :]
        T_world_camera = np.asarray(annotations[row["frame_idx"]]["camera"]["T_world_camera_metric"], dtype=np.float64)
        vertices_world = transform_points(vertices_camera, T_world_camera)
        meshes_world.append(trimesh.Trimesh(vertices=vertices_world.astype(np.float32), faces=np.asarray(mesh_prior.faces, dtype=np.int32), process=False))
        frame_ids.append(int(row["frame_idx"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = args.output_dir / "depth_carved_mesh_prior_frame.obj"
    mesh_prior.export(prior_path)
    archive_path = args.output_dir / "depth_carved_meshes_world.npz"
    save_archive(archive_path, frame_ids, meshes_world)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "object_visual_hull_depth_carve_v3",
        "claim_tested": "model-produced masks and metric depth can carve a closed object-centric mesh under solved object poses",
        "mesh_prior_camera": str(args.mesh_prior_camera),
        "graph_mesh_archive": str(args.graph_mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "prior_frame_mesh": str(prior_path),
        "mesh_archive_world": str(archive_path),
        "frames": frame_ids,
        "grid_shape": [int(v) for v in grid_shape],
        "grid_voxels": int(np.prod(grid_shape)),
        "occupied_voxels": int(np.count_nonzero(occupied)),
        "mesh_topology": topology(mesh_prior),
        "carve_rows": carve_rows,
        "contact_constraints": contact_constraint_row,
        "pose_rows": pose_rows,
        "parameters": {
            "pitch_m": float(args.pitch_m),
            "grid_pad_m": float(args.grid_pad_m),
            "depth_front_tolerance_m": float(args.depth_front_tolerance_m),
            "depth_back_tolerance_m": float(args.depth_back_tolerance_m),
            "min_carve_votes": int(args.min_carve_votes),
            "min_depth_hits": int(args.min_depth_hits),
            "close_iterations": int(args.close_iterations),
            "sdf_smooth_sigma_voxels": float(args.sdf_smooth_sigma_voxels),
        },
    }
    save_json(args.output_dir / "qc_depth_carved_mesh_v3.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"carve_rows", "pose_rows"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior-camera", type=Path, required=True)
    parser.add_argument("--graph-mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "metric-depth"], default="annotation-vggt")
    parser.add_argument("--pitch-m", type=float, default=0.003)
    parser.add_argument("--grid-pad-m", type=float, default=0.012)
    parser.add_argument("--max-grid-voxels", type=int, default=5000000)
    parser.add_argument("--depth-front-tolerance-m", type=float, default=0.006)
    parser.add_argument("--depth-back-tolerance-m", type=float, default=0.030)
    parser.add_argument("--min-carve-votes", type=int, default=2)
    parser.add_argument("--min-depth-hits", type=int, default=2)
    parser.add_argument("--close-iterations", type=int, default=1)
    parser.add_argument("--contact-protect-radius-m", type=float, default=0.0)
    parser.add_argument("--sdf-pad-voxels", type=int, default=8)
    parser.add_argument("--sdf-smooth-sigma-voxels", type=float, default=0.5)
    parser.add_argument("--max-pose-correspondences", type=int, default=12000)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=181)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
