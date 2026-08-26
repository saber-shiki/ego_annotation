#!/usr/bin/env python3
"""Independent diagnostic: aligned SAM3D mesh vs metric depth and hand depth order.

Evaluates the GHOST-lite aligned mesh on source-plane pixels:
  * signed depth error at object-owned pixels (mesh_z - UniDepth measured),
  * silhouette overlap (convex projection vs object-owned mask),
  * hand-vs-mesh depth order on hand-support pixels (does the hand appear in
    front of the mesh as the video suggests?),
  * free-space contradiction (mesh projecting outside the owned mask).
No generated face is promoted to collision/sign authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

sys_path = str(Path(__file__).resolve().parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RuntimeError(f"{label} invalid: {result}")
    return result


def summarize(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def project(pts: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-6)
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = fx * pts[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * pts[valid, 1] / z[valid] + cy
    return uv, valid


def zbuffer_batch(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, size: int, device: str, batch: int = 8) -> np.ndarray:
    import torch
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

    v = torch.tensor(vertices, dtype=torch.float32, device=device)
    f = torch.tensor(faces, dtype=torch.int64, device=device)
    Kt = torch.tensor(K, dtype=torch.float32, device=device)[None]
    n = v.shape[0]
    all_z = []
    with torch.no_grad():
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            m = stop - start
            R = torch.eye(3, device=device)[None].expand(m, -1, -1)
            T = torch.zeros(m, 3, device=device)
            Kk = Kt.expand(m, -1, -1)
            size_t = torch.tensor([[size, size]], device=device).expand(m, -1)
            cameras = cameras_from_opencv_projection(R, T, Kk, size_t)
            rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=size, blur_radius=0, faces_per_pixel=1, cull_backfaces=False
                ),
            )
            zb = rasterizer(Meshes(verts=[v[i] for i in range(start, stop)], faces=[f] * m)).zbuf
            all_z.append(zb[..., 0].detach().cpu().numpy())
    return np.concatenate(all_z, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-qc", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, required=True)
    parser.add_argument("--hand-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default="30,50,70,92,94,110,130,146")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    qc = load_json(args.alignment_qc)
    mesh = trimesh.load(qc["mesh_out"], force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)  # anchor camera frame
    faces = np.asarray(mesh.faces, dtype=np.int64)
    annotations = load_json(args.annotations)
    frame_by_idx = {int(f["frame_idx"]): f for f in annotations["frames"]}
    depths = np.load(args.depth_npz)
    depth_map = depths["depth"].astype(np.float64)
    depth_idx = depths["frame_idx"].astype(int).tolist()
    hands = np.load(args.hand_npz)
    hand_idx = hands["frame_idx"].astype(int).tolist()
    anchor_T = np.asarray(frame_by_idx[int(qc["anchor_frame"])]["camera"]["T_world_camera_metric"], dtype=np.float64)

    rows = {}
    for raw in args.frames.split(","):
        idx = int(raw)
        frame = frame_by_idx[idx]
        obj = frame["objects"][0]
        visible = obj["visible_geometry_candidate"]
        intr = finite_vector(visible["intrinsics_fx_fy_cx_cy"], 4, "intrinsics")
        fx, fy, cx, cy = intr.tolist()
        K = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
        depth_row = depth_idx.index(idx)
        depth = depth_map[depth_row]
        H, W = depth.shape
        mask = cv2.imread(str(visible["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0

        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        cam = (vertices - T[:3, 3][None, :]) @ T[:3, :3]
        zb = zbuffer_batch(cam[None, :, :], faces, K, H, args.device, batch=1)[0]  # (H, W)
        mesh_z = zb
        valid_depth = np.isfinite(depth) & (depth > 0.05)
        owned = mask & valid_depth
        signed_err = mesh_z[owned] - depth[owned]
        # free-space contradiction: mesh pixels outside owned mask (with depth)
        free = (~mask) & np.isfinite(mesh_z) & (mesh_z > 0.05)
        # hand depth order: for each hand, mean hand z vs mesh z at hand pixels
        hand_info = {}
        for side in ("left", "right"):
            pos = hand_idx.index(idx) if idx in hand_idx else -1
            if pos < 0 or int(np.asarray(hands[f"{side}_valid"])[pos]) != 1:
                continue
            vw = np.asarray(hands[f"{side}_vertices_world_m"][pos], dtype=np.float64)
            vc = (vw - T[:3, 3][None, :]) @ T[:3, :3]
            uv, valid = project(vc, fx, fy, cx, cy)
            inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if not np.any(inside):
                continue
            x = np.clip(np.rint(uv[inside, 0]).astype(np.int64), 0, W - 1)
            y = np.clip(np.rint(uv[inside, 1]).astype(np.int64), 0, H - 1)
            hand_z = vc[inside, 2]
            mesh_hit = np.isfinite(mesh_z[y, x]) & (mesh_z[y, x] > 0.05)
            d_order = hand_z[mesh_hit] - mesh_z[y, x][mesh_hit]
            hand_info[side] = {
                "hand_pixels": int(np.count_nonzero(inside)),
                "mesh_hit_pixels": int(np.count_nonzero(mesh_hit)),
                "hand_minus_mesh_depth_median_m": float(np.median(d_order)) if np.any(mesh_hit) else None,
                "hand_in_front_fraction": float(np.mean(d_order < 0.0)) if np.any(mesh_hit) else None,
                "hand_z_median_m": float(np.median(hand_z)),
                "mesh_z_median_at_hand_m": float(np.median(mesh_z[y, x][mesh_hit])) if np.any(mesh_hit) else None,
            }
        # convex silhouette IoU
        proj = cv2.convexHull(np.rint(np.column_stack([(fx * cam[cam[:, 2] > 1e-6, 0] / cam[cam[:, 2] > 1e-6, 2]) + cx,
                                                       (fy * cam[cam[:, 2] > 1e-6, 1] / cam[cam[:, 2] > 1e-6, 2]) + cy])).astype(np.int32))
        sil = np.zeros((H, W), dtype=np.uint8)
        if len(proj) >= 3:
            cv2.fillConvexPoly(sil, proj, 1)
        sil = sil > 0
        inter = np.count_nonzero(sil & mask)
        union = np.count_nonzero(sil | mask)
        rows[str(idx)] = {
            "signed_mesh_minus_measured_depth_m": summarize(signed_err),
            "mesh_behind_measured_fraction": float(np.mean(signed_err > 0.006)),
            "mesh_poking_in_front_fraction": float(np.mean(signed_err < -0.006)),
            "free_space_contradiction_pixels": int(np.count_nonzero(free)),
            "free_space_contradiction_fraction_of_image": float(np.count_nonzero(free) / (H * W)),
            "convex_silhouette_iou": float(inter / max(1, union)),
            "convex_silhouette_recall": float(inter / max(1, np.count_nonzero(mask))),
            "hand_depth_order": hand_info,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "sim3_aligned_depth_hand_diagnostic.json"
    report = {
        "schema": "sam3d_ghost_lite_sim3_depth_hand_diagnostic_v1",
        "diagnostic_only": True,
        "annotation_ready": False,
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for idx, row in rows.items():
        print("\nFRAME", idx)
        print("  signed depth (mesh - measured):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row["signed_mesh_minus_measured_depth_m"].items()})
        print("  behind frac", round(row["mesh_behind_measured_fraction"], 4), "poke frac", round(row["mesh_poking_in_front_fraction"], 4),
              "sil iou", round(row["convex_silhouette_iou"], 4), "free px", row["free_space_contradiction_pixels"])
        for side, h in row["hand_depth_order"].items():
            print("  hand", side, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()})


if __name__ == "__main__":
    main()
