#!/usr/bin/env python3
"""Visualize observed visible-surface surfels vs the SAM3D reconstructed mesh.

For each requested frame (P15 per-frame pose applied to the canonical SAM3D
metric mesh):
  * cyan  hull  : SAM3D mesh projection (convex hull silhouette)
  * red   mask  : object-owned mask (semi-transparent)
  * yellow dots : observed first-surface surfels (camera_vertices_sample_m)
  * green dots  : HaWoR hand vertices (per side)
This is a projection/silhouette visualization only. Yellow points inside the
cyan projection do not prove depth alignment; use the full-resolution
first-hit audit for metric surface placement.
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


def scatter(img: np.ndarray, uv: np.ndarray, valid: np.ndarray, width: int, height: int, color: tuple[int, int, int], radius: int = 3) -> None:
    inside = valid & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    pts = np.rint(uv[inside]).astype(np.int32)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-anchor-camera", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-graph", type=Path, required=True)
    parser.add_argument("--hand-npz", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--frames", default="92,103,121")
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    frame_by_idx = {int(f["frame_idx"]): f for f in annotations["frames"]}
    pose = load_json(args.pose_graph)
    pose_by_idx = {int(r["frame_idx"]): r for r in pose["pose_rows"]}
    hands = np.load(args.hand_npz)
    hand_idx = hands["frame_idx"].astype(int).tolist()
    mesh = trimesh.load(args.mesh_anchor_camera, force="mesh", process=False)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    T92 = np.asarray(frame_by_idx[args.anchor_frame]["camera"]["T_world_camera_metric"], dtype=np.float64)
    anchor_pose = pose_by_idx[args.anchor_frame]
    cent92 = np.asarray(anchor_pose["translation_world_m"], dtype=np.float64)
    R92 = np.asarray(anchor_pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    canonical = ((v @ T92[:3, :3].T + T92[:3, 3]) - cent92[None, :]) @ R92
    fx, fy, cx, cy = K
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for raw in args.frames.split(","):
        idx = int(raw)
        frame = frame_by_idx[idx]
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        r = pose_by_idx[idx]
        Rw = np.asarray(r["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        tw = np.asarray(r["translation_world_m"], dtype=np.float64)
        vw = canonical @ Rw.T + tw
        vc = (vw - T[:3, 3][None, :]) @ T[:3, :3]
        height = int(frame["source_height"])
        width = int(frame["source_width"])
        rgb = cv2.imread(str(Path(args.rgb_dir) / f"{idx:06d}.jpg"))
        if rgb is None:
            rgb = cv2.imread(str(Path(args.rgb_dir) / f"{idx:06d}.png"))
        if rgb is None or rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height))

        # cyan hull: SAM3D mesh
        uv, valid = project(vc, fx, fy, cx, cy)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        inside = valid & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        pts = np.rint(uv[inside]).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillConvexPoly(canvas, cv2.convexHull(pts), (0, 200, 255))
        overlay = cv2.addWeighted(rgb, 0.7, canvas, 0.45, 0.0)

        # red mask
        mask = cv2.imread(str(Path(args.mask_dir) / f"{idx:06d}_carton_milk_object_owned_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height))
            overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.asarray([80, 80, 255], dtype=np.float64) * 0.5).astype(np.uint8)

        # yellow dots: observed surfels
        obj = frame["objects"][0]
        visible = obj["visible_geometry_candidate"]
        obs = np.asarray(visible["camera_vertices_sample_m"], dtype=np.float64)
        obs = obs[np.isfinite(obs).all(axis=1) & (obs[:, 2] > 0.0)]
        uvo, valido = project(obs, fx, fy, cx, cy)
        scatter(overlay, uvo, valido, width, height, (0, 255, 255), radius=2)

        # green dots: hands
        if idx in hand_idx:
            pos = hand_idx.index(idx)
            for side in ("left", "right"):
                if int(np.asarray(hands[f"{side}_valid"])[pos]) != 1:
                    continue
                hw = np.asarray(hands[f"{side}_vertices_world_m"][pos], dtype=np.float64)
                hc = (hw - T[:3, 3][None, :]) @ T[:3, :3]
                uvh, validh = project(hc, fx, fy, cx, cy)
                scatter(overlay, uvh, validh, width, height, (0, 255, 0), radius=1)

        out = args.output_dir / f"fit_{idx:06d}.png"
        cv2.imwrite(str(out), overlay)
        print(out)


if __name__ == "__main__":
    main()
