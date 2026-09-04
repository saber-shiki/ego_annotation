#!/usr/bin/env python3
"""Render a compact visual A/B sheet for observed-only pose repair."""
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


def load(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text())


def resize_k(k: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    sx = float(target_wh[0]) / float(source_wh[0])
    sy = float(target_wh[1]) / float(source_wh[1])
    out = np.asarray(k, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = sx * (out[0, 2] + 0.5) - 0.5
    out[1, 2] = sy * (out[1, 2] + 0.5) - 0.5
    return out


def contour(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8)) > 0
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[edge] = np.asarray(color, dtype=np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--formal-pose", required=True)
    parser.add_argument("--repaired-pose", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", default="0,1,4,84,108,125,145,149")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=480)
    args = parser.parse_args()

    annotations = load(args.annotations)
    frames = {int(x["frame_idx"]): x for x in annotations["frames"]}
    formal_rows = {
        int(x["frame_idx"]): x
        for x in load(args.formal_pose)["pose_rows"]
        if x.get("rotation_world_from_completed_canonical_matrix") is not None
    }
    repaired_rows = {
        int(x["frame_idx"]): x
        for x in load(args.repaired_pose)["pose_rows"]
        if x.get("rotation_world_from_completed_canonical_matrix") is not None
    }
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = torch.tensor(np.asarray(mesh.faces, dtype=np.int64), device=args.device)
    selected = [int(x) for x in args.frames.split(",")]

    first = next(
        x for x in annotations["frames"]
        if isinstance(x.get("objects"), list)
        and x["objects"]
        and isinstance(x["objects"][0].get("visible_geometry_candidate"), dict)
    )
    values = first["objects"][0]["visible_geometry_candidate"]["intrinsics_fx_fy_cx_cy"]
    k_cal = np.asarray([[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    panels: list[np.ndarray] = []

    for frame_idx in selected:
        frame = frames[frame_idx]
        obj = next(o for o in frame["objects"] if o.get("object_id") == "carton_milk")
        rgb = cv2.imread(str(frame["raw_frame_path"]), cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(str(obj["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask_raw is None:
            raise RuntimeError(f"missing RGB/mask for frame {frame_idx}")
        height, width = rgb.shape[:2]
        target = cv2.resize(
            (mask_raw > 0).astype(np.uint8),
            (args.size, args.size),
            interpolation=cv2.INTER_NEAREST_EXACT,
        ) > 0
        k_image = resize_k(
            k_cal,
            (int(frame.get("source_width") or 1408), int(frame.get("source_height") or 1408)),
            (width, height),
        )
        k_raster = resize_k(k_image, (width, height), (args.size, args.size))
        t_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)

        def render(row: dict) -> np.ndarray:
            rotation = np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
            translation = np.asarray(row["translation_world_m"], dtype=np.float64)
            world = vertices @ rotation.T + translation
            camera_vertices = (world - t_world_camera[:3, 3]) @ t_world_camera[:3, :3]
            camera = cameras_from_opencv_projection(
                torch.eye(3, device=args.device, dtype=torch.float32)[None],
                torch.zeros((1, 3), device=args.device, dtype=torch.float32),
                torch.tensor(k_raster, device=args.device, dtype=torch.float32)[None],
                torch.tensor([[args.size, args.size]], device=args.device, dtype=torch.float32),
            )
            rasterizer = MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=args.size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    cull_backfaces=False,
                    bin_size=0,
                ),
            )
            with torch.no_grad():
                fragments = rasterizer(
                    Meshes(
                        verts=[torch.tensor(camera_vertices, device=args.device, dtype=torch.float32)],
                        faces=[faces],
                    )
                )
            return fragments.pix_to_face[0, :, :, 0].cpu().numpy() >= 0

        formal_mask = render(formal_rows[frame_idx])
        repaired_mask = render(repaired_rows[frame_idx])
        panel = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
        target_edge = contour(target, (40, 210, 40))
        formal_edge = contour(formal_mask, (40, 40, 230))
        repaired_edge = contour(repaired_mask, (230, 180, 30))
        panel[target_edge.any(axis=2)] = (40, 210, 40)
        panel[formal_edge.any(axis=2)] = (40, 40, 230)
        panel[repaired_edge.any(axis=2)] = (230, 180, 30)
        formal_iou = np.count_nonzero(formal_mask & target) / max(1, np.count_nonzero(formal_mask | target))
        repaired_iou = np.count_nonzero(repaired_mask & target) / max(1, np.count_nonzero(repaired_mask | target))
        cv2.putText(panel, f"frame {frame_idx}: green=mask red=formal blue=repaired", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"IoU formal={formal_iou:.3f} repaired={repaired_iou:.3f}", (10, args.size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)

    columns = 2
    rows = (len(panels) + columns - 1) // columns
    sheet = np.full((rows * args.size, columns * args.size, 3), 30, dtype=np.uint8)
    for index, panel in enumerate(panels):
        y = (index // columns) * args.size
        x = (index % columns) * args.size
        sheet[y:y + args.size, x:x + args.size] = panel
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(output)


if __name__ == "__main__":
    main()
