#!/usr/bin/env python3
"""Render the SAM3D metric mesh as an RGB+alpha overlay for review.

Two modes:
  * --mode sim3 : static GHOST-lite aligned mesh in anchor camera frame
    (mesh_out from the multi-frame Sim(3) alignment QC).
  * --mode p15  : per-frame P15 rigid pose applied to the canonical mesh
    (canonical = anchor-world minus anchor centroid; rows give per-frame
    rotation_world_from_completed_canonical + translation_world_m).

Overlay shows the mesh convex-hull silhouette in cyan plus the object-owned
mask in translucent red for side-by-side alignment review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

K = [974.3447265625, 974.3447265625, 702.6607055664062, 706.1102294921875]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def project(points_camera: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    valid = z > 1e-6
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = fx * points_camera[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points_camera[valid, 1] / z[valid] + cy
    return uv, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sim3", choices=["sim3", "p15"])
    parser.add_argument("--alignment-qc", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-graph", type=Path)
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default="30,50,70,92,94,103,110,121,130,146")
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    frame_by_idx = {int(f["frame_idx"]): f for f in annotations["frames"]}
    fx, fy, cx, cy = K

    if args.mode == "sim3":
        assert args.alignment_qc is not None
        qc = load_json(args.alignment_qc)
        mesh = trimesh.load(qc["mesh_out"], force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)  # anchor camera frame
    else:
        assert args.pose_graph is not None
        pose = load_json(args.pose_graph)
        pose_by_idx = {int(r["frame_idx"]): r for r in pose["pose_rows"]}
        mesh = trimesh.load(args.alignment_qc, force="mesh", process=False) if args.alignment_qc else None
        assert mesh is not None, "p15 mode needs the anchor-camera metric mesh via --alignment-qc"
        v = np.asarray(mesh.vertices, dtype=np.float64)
        T92 = np.asarray(frame_by_idx[args.anchor_frame]["camera"]["T_world_camera_metric"], dtype=np.float64)
        cent92 = np.asarray(pose_by_idx[args.anchor_frame]["translation_world_m"], dtype=np.float64)
        canonical = (v @ T92[:3, :3].T + T92[:3, 3]) - cent92[None, :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for raw in args.frames.split(","):
        idx = int(raw)
        frame = frame_by_idx[idx]
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        if args.mode == "sim3":
            cam = (vertices - T[:3, 3][None, :]) @ T[:3, :3]
        else:
            r = pose_by_idx[idx]
            Rw = np.asarray(r["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
            tw = np.asarray(r["translation_world_m"], dtype=np.float64)
            vw = canonical @ Rw.T + tw
            cam = (vw - T[:3, 3][None, :]) @ T[:3, :3]
        uv, valid = project(cam, fx, fy, cx, cy)
        height = int(frame["source_height"])
        width = int(frame["source_width"])
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        inside = valid & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        pts = np.rint(uv[inside]).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillConvexPoly(canvas, cv2.convexHull(pts), (0, 200, 255))
        rgb_path = Path(args.rgb_dir) / f"{idx:06d}.jpg"
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            rgb_path = Path(args.rgb_dir) / f"{idx:06d}.png"
            rgb = cv2.imread(str(rgb_path))
        if rgb is not None and rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height))
        overlay = cv2.addWeighted(rgb, 0.6, canvas, 0.6, 0.0) if rgb is not None else canvas
        mask = cv2.imread(str(Path(args.mask_dir) / f"{idx:06d}_carton_milk_object_owned_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height))
            overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.asarray([80, 80, 255], dtype=np.float64) * 0.5).astype(np.uint8)
        out = args.output_dir / f"overlay_{idx:06d}.png"
        cv2.imwrite(str(out), overlay)
        print(out)


if __name__ == "__main__":
    main()
