#!/usr/bin/env python3
"""Full-resolution audit of observed surface position inside an aligned mesh.

The mesh is supplied in anchor-camera coordinates and propagated with P15
per-frame object SE(3).  At trusted surfel pixels this audit reports:

* first-hit depth residual: mesh_z - observed_z (negative = mesh in front),
* first-hit coverage,
* on the anchor frame, observed position between captured front/last hits.

The last-hit statistic is diagnostic for showing depth-centering; generated
hidden faces remain render-only and are not signed/collision authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25.0)),
        "q75": float(np.percentile(values, 75.0)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def camera_to_world(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def world_to_camera(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (points - T[:3, 3]) @ T[:3, :3]


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    pose = load_json(args.pose_graph)
    frame_by_idx = {int(f["frame_idx"]): f for f in annotations["frames"]}
    pose_by_idx = {int(r["frame_idx"]): r for r in pose["pose_rows"]}
    mesh = trimesh.load(args.mesh_anchor_camera, force="mesh", process=False)
    vertices_anchor = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    anchor_frame = frame_by_idx[args.anchor_frame]
    anchor_pose = pose_by_idx[args.anchor_frame]
    T_anchor = np.asarray(anchor_frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    R_anchor = np.asarray(anchor_pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    t_anchor = np.asarray(anchor_pose["translation_world_m"], dtype=np.float64)
    world_anchor = camera_to_world(vertices_anchor, T_anchor)
    canonical = (world_anchor - t_anchor[None, :]) @ R_anchor

    K = np.asarray([[974.3447265625, 0.0, 702.6607055664062], [0.0, 974.3447265625, 706.1102294921875], [0.0, 0.0, 1.0]])
    size = args.image_size
    cameras = cameras_from_opencv_projection(
        torch.eye(3, dtype=torch.float32, device=args.device)[None],
        torch.zeros(1, 3, dtype=torch.float32, device=args.device),
        torch.tensor(K, dtype=torch.float32, device=args.device)[None],
        torch.tensor([[size, size]], dtype=torch.float32, device=args.device),
    )
    faces_t = torch.tensor(faces, dtype=torch.int64, device=args.device)
    rows: dict[str, Any] = {}
    for idx in [int(x) for x in args.frames.split(",")]:
        frame = frame_by_idx[idx]
        row = pose_by_idx[idx]
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        R = np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        t = np.asarray(row["translation_world_m"], dtype=np.float64)
        world = canonical @ R.T + t[None, :]
        camera_vertices = world_to_camera(world, T)
        faces_per_pixel = args.anchor_faces_per_pixel if idx == args.anchor_frame else 1
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=size,
                blur_radius=0.0,
                faces_per_pixel=faces_per_pixel,
                cull_backfaces=False,
                bin_size=0,
            ),
        )
        with torch.no_grad():
            zbuf = rasterizer(Meshes(
                verts=[torch.tensor(camera_vertices, dtype=torch.float32, device=args.device)],
                faces=[faces_t],
            )).zbuf[0].detach().cpu().numpy().astype(np.float64)
        visible = frame["objects"][0]["visible_geometry_candidate"]
        observed = np.asarray(visible["camera_vertices_sample_m"], dtype=np.float64)
        observed = observed[np.isfinite(observed).all(axis=1) & (observed[:, 2] > 1.0e-6)]
        fx, fy, cx, cy = visible["intrinsics_fx_fy_cx_cy"]
        u = np.rint(fx * observed[:, 0] / observed[:, 2] + cx).astype(np.int64)
        v = np.rint(fy * observed[:, 1] / observed[:, 2] + cy).astype(np.int64)
        inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        observed = observed[inside]; u = u[inside]; v = v[inside]
        hits = zbuf[v, u]
        front = hits[:, 0]
        valid = np.isfinite(front) & (front > 1.0e-6)
        signed = front[valid] - observed[valid, 2]
        output: dict[str, Any] = {
            "first_hit_mesh_z_minus_observed_z_m": summarize(signed),
            "surfel_first_hit_coverage_fraction": float(np.mean(valid)),
        }
        if faces_per_pixel > 1:
            last = np.asarray([
                values[values > 1.0e-6][-1] if np.any(values > 1.0e-6) else np.nan
                for values in hits
            ])
            thickness_valid = valid & np.isfinite(last) & (last > front + 1.0e-4)
            thickness = last[thickness_valid] - front[thickness_valid]
            position = (observed[thickness_valid, 2] - front[thickness_valid]) / thickness
            output["captured_front_to_last_thickness_m"] = summarize(thickness)
            output["observed_fraction_from_front_to_last"] = summarize(position)
        rows[str(idx)] = output
        print(idx, "first-hit residual mm", {k: round(v * 1000.0, 2) if k != "count" else v for k, v in output["first_hit_mesh_z_minus_observed_z_m"].items()}, "coverage", round(output["surfel_first_hit_coverage_fraction"], 3))
        if "observed_fraction_from_front_to_last" in output:
            print(" anchor observed position fraction", output["observed_fraction_from_front_to_last"])

    report = {
        "schema": "sam3d_first_hit_surface_position_audit_v1",
        "diagnostic_only": True,
        "annotation_ready": False,
        "signed_depth_convention": "mesh_z_minus_observed_z; negative=mesh_front_surface_in_front_of_observation; positive=mesh_behind_observation",
        "mesh_anchor_camera": str(args.mesh_anchor_camera),
        "anchor_frame": args.anchor_frame,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "sam3d_first_hit_surface_position_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-anchor-camera", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--frames", default="30,50,70,92,103,110,121,130,146")
    parser.add_argument("--image-size", type=int, default=1408)
    parser.add_argument("--anchor-faces-per-pixel", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
