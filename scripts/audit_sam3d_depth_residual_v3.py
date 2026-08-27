#!/usr/bin/env python3
"""Analyze the SAM3D mesh vs observed metric surface depth residual.

Evaluates, per frame, three complementary views of how the SAM3D complete
mesh adheres to the observed first-surface (UniDepth-derived surfels):

  * same-pixel depth residual: rasterize the aligned mesh, sample the mesh
    depth at each object-owned pixel where the surfel lives, compare to the
    measured depth.  This is the *honest* depth error.  The signed convention
    is mesh_z - observed_z: negative means the generated front surface is in
    front of the measured surface; positive means it is behind.
  * nearest-neighbor 3D distance: observed surfel -> aligned mesh (the usual
    Chamfer view; includes oblique/edge matches that hide depth bias).
  * per-frame bbox extent ratio mesh/observed (x,y,z).

Diagnostic only. Generated SAM3D faces stay render-only; nothing here
promotes the mesh to collision/sign/contact authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from scipy.spatial import cKDTree

BASE = "/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/backend_tests/20260819T122452Z_hot3d_milk_local_authority_local29_v1/runs/P0014_84ea2dcc_carton_milk_f2370_2519"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median_m": float(np.median(arr)),
        "q25_m": float(np.percentile(arr, 25.0)),
        "q75_m": float(np.percentile(arr, 75.0)),
        "p05_m": float(np.percentile(arr, 5.0)),
        "p95_m": float(np.percentile(arr, 95.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-anchor-camera", type=Path, default=Path(BASE) / "experiments/sam3d_native_ghost_lite_20260826/p15_first_hit_depth_authority_bounded_v4/SAM3D_FIRST_HIT_ALIGNED_RECOMMENDED.binary.ply")
    parser.add_argument("--annotations", type=Path, default=Path(BASE) / "measurements/object_geometry/visible_geometry/carton_milk/annotations_v19_visible_geometry.json")
    parser.add_argument("--pose-graph", type=Path, default=Path(BASE) / "experiments/sam3d_trellis_controlled/P15_observed_pose_graph/v19_rigid_object_pose_graph_report.json")
    parser.add_argument("--depth-npz", type=Path, default=Path(BASE) / "measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz")
    parser.add_argument("--mask-dir", type=Path, default=Path(BASE) / "measurements/object_geometry/visible_geometry/carton_milk/object_owned_masks")
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--frames", default="30,50,70,92,103,110,121,130,146")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    pose = load_json(args.pose_graph)
    ann_by_idx = {int(f["frame_idx"]): f for f in annotations["frames"]}
    pose_by_idx = {int(r["frame_idx"]): r for r in pose["pose_rows"]}
    depth = np.load(args.depth_npz)
    dmap = depth["depth"].astype(np.float64)
    didx = depth["frame_idx"].astype(int).tolist()
    mesh = trimesh.load(args.mesh_anchor_camera, force="mesh", process=False)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    T92 = np.asarray(ann_by_idx[args.anchor_frame]["camera"]["T_world_camera_metric"], dtype=np.float64)
    anchor_pose = pose_by_idx[args.anchor_frame]
    cent92 = np.asarray(anchor_pose["translation_world_m"], dtype=np.float64)
    R92 = np.asarray(anchor_pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    canonical = ((v @ T92[:3, :3].T + T92[:3, 3]) - cent92[None, :]) @ R92

    device = args.device
    size = 1408
    K = np.asarray([[974.3447, 0, 702.6607], [0, 974.3447, 706.1102], [0, 0, 1.0]], dtype=np.float64)
    cam_pt3d = cameras_from_opencv_projection(
        torch.eye(3, device=device)[None],
        torch.zeros(1, 3, device=device),
        torch.tensor(K, dtype=torch.float32, device=device)[None],
        torch.tensor([[size, size]], device=device),
    )
    rasterizer = MeshRasterizer(
        cameras=cam_pt3d,
        raster_settings=RasterizationSettings(image_size=size, blur_radius=0, faces_per_pixel=1, cull_backfaces=False, bin_size=0),
    )

    rows: dict[str, dict] = {}
    for raw in args.frames.split(","):
        idx = int(raw)
        frame = ann_by_idx[idx]
        vis = frame["objects"][0].get("visible_geometry_candidate")
        if not isinstance(vis, dict):
            continue
        r = pose_by_idx[idx]
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        Rw = np.asarray(r["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        tw = np.asarray(r["translation_world_m"], dtype=np.float64)
        vw = canonical @ Rw.T + tw
        vc = (vw - T[:3, 3][None, :]) @ T[:3, :3]
        vt = torch.tensor(vc[None], dtype=torch.float32, device=device)
        ft = torch.tensor(faces[None], dtype=torch.int64, device=device)
        with torch.no_grad():
            zb = rasterizer(Meshes(verts=vt, faces=ft)).zbuf[0, ..., 0].detach().cpu().numpy()
        mesh_z = np.where(zb > 0.05, np.where(np.isfinite(zb), zb, np.nan), np.nan)
        d = dmap[didx.index(idx)]
        mask = cv2.imread(str(Path(args.mask_dir) / f"{idx:06d}_carton_milk_object_owned_mask.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0
        valid = np.isfinite(d) & (d > 0.05) & np.isfinite(mesh_z) & mask
        same_pixel = mesh_z[valid] - d[valid]  # positive=mesh behind; negative=mesh in front
        # surfel same-pixel residual
        obs = np.asarray(vis["camera_vertices_sample_m"], dtype=np.float64)
        obs = obs[np.isfinite(obs).all(axis=1) & (obs[:, 2] > 0.0)]
        fx, fy, cx, cy = vis["intrinsics_fx_fy_cx_cy"]
        u = fx * obs[:, 0] / obs[:, 2] + cx
        w = fy * obs[:, 1] / obs[:, 2] + cy
        uu = np.clip(np.rint(u).astype(np.int64), 0, size - 1)
        ww = np.clip(np.rint(w).astype(np.int64), 0, size - 1)
        mz_at = mesh_z[ww, uu]
        ok = np.isfinite(mz_at)
        surfel_same_pixel = mz_at[ok] - obs[:, 2][ok]
        # nearest-neighbor view
        tree = cKDTree(vc)
        nn, _ = tree.query(obs, k=1)
        # bbox extents
        ext_m = np.ptp(vc, axis=0)
        ext_o = np.ptp(obs, axis=0)
        rows[str(idx)] = {
            "same_pixel_mesh_minus_measured_m": stats(same_pixel),
            "surfel_same_pixel_mesh_minus_surfel_m": stats(surfel_same_pixel),
            "nearest_neighbor_obs_to_mesh_m": stats(nn),
            "bbox_extent_ratio_mesh_over_obs": (ext_m / ext_o).tolist(),
            "bbox_mesh_m": ext_m.tolist(),
            "bbox_obs_m": ext_o.tolist(),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "sam3d_depth_residual_audit.json"
    out_path.write_text(json.dumps({
        "schema": "sam3d_depth_residual_audit_v1",
        "diagnostic_only": True,
        "annotation_ready": False,
        "signed_depth_convention": "mesh_z_minus_observed_z; negative=mesh_front_surface_in_front_of_observation; positive=mesh_behind_observation",
        "claim_scope": "Same-pixel depth residual vs UniDepth and vs surfels, plus nearest-neighbor and bbox ratios. Generated faces remain render-only.",
        "frames": rows,
    }, indent=2) + "\n", encoding="utf-8")
    for idx, row in rows.items():
        s = row["same_pixel_mesh_minus_measured_m"]
        sp = row["surfel_same_pixel_mesh_minus_surfel_m"]
        nn_ = row["nearest_neighbor_obs_to_mesh_m"]
        print(f"frame {idx}: same_pixel med {s['median_m']*1000:.1f}mm q25/q75 "
              f"{s['q25_m']*1000:.1f}/{s['q75_m']*1000:.1f} | surfel_same_pixel med {sp['median_m']*1000:.1f}mm "
              f"| NN med {nn_['median_m']*1000:.1f}mm | bbox ratio {np.round(row['bbox_extent_ratio_mesh_over_obs'],3).tolist()}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
