#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(transform, dtype=np.float64) @ homog.T).T[:, :3]


def build_faces(index_grid: np.ndarray, vertices: np.ndarray, max_edge_m: float) -> np.ndarray:
    faces = []
    rows, cols = index_grid.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            corners = (
                int(index_grid[r, c]),
                int(index_grid[r, c + 1]),
                int(index_grid[r + 1, c]),
                int(index_grid[r + 1, c + 1]),
            )
            for tri in ((corners[0], corners[1], corners[2]), (corners[1], corners[3], corners[2])):
                if min(tri) < 0:
                    continue
                pts = vertices[np.asarray(tri, dtype=np.int64)]
                edge = max(
                    np.linalg.norm(pts[0] - pts[1]),
                    np.linalg.norm(pts[1] - pts[2]),
                    np.linalg.norm(pts[2] - pts[0]),
                )
                if edge <= max_edge_m:
                    faces.append(tri)
    return np.asarray(faces, dtype=np.int32)


def remove_unreferenced(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used].astype(np.float32), remap[faces].astype(np.int32)


def load_intrinsics(dataset: Path) -> np.ndarray:
    K = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise RuntimeError(f"invalid intrinsics matrix: {dataset / 'cam_K.txt'}")
    return K


def annotation_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames list")
    out = {}
    for frame in frames:
        idx = int(frame["frame_idx"])
        out[idx] = frame
    return out


def mesh_from_entry(entry: dict, dataset: Path, K: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    idx = int(entry["index"])
    frame_idx = int(entry["frame_idx"])
    depth_path = dataset / "depth" / f"{idx:06d}.png"
    mask_path = dataset / "masks" / f"{idx:06d}.png"
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None:
        raise RuntimeError(f"failed to read depth/mask for frame {frame_idx}")
    if depth.shape != mask.shape:
        raise RuntimeError(f"depth/mask shape mismatch for frame {frame_idx}: {depth.shape} vs {mask.shape}")
    if args.mask_erode_px > 0:
        kernel = np.ones((2 * args.mask_erode_px + 1, 2 * args.mask_erode_px + 1), dtype=np.uint8)
        mask_bool = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=1) > 0
    else:
        mask_bool = mask > 0
    depth_m = depth.astype(np.float64) / 1000.0
    values = depth_m[mask_bool & np.isfinite(depth_m) & (depth_m > args.min_depth_m) & (depth_m < args.max_depth_m)]
    if values.size < int(args.min_depth_pixels):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), {
            "frame_idx": frame_idx,
            "status": "skipped_too_few_depth_pixels",
            "depth_pixels": int(values.size),
        }
    lo = float(np.quantile(values, args.depth_low_quantile))
    hi = float(np.quantile(values, args.depth_high_quantile))
    keep = mask_bool & np.isfinite(depth_m) & (depth_m >= lo) & (depth_m <= hi)

    ys = np.arange(0, depth.shape[0], int(args.mask_stride), dtype=np.int32)
    xs = np.arange(0, depth.shape[1], int(args.mask_stride), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled_keep = keep[np.ix_(ys, xs)]
    depth_sample = depth_m[np.ix_(ys, xs)]
    flat_x = grid_x[sampled_keep].astype(np.float64)
    flat_y = grid_y[sampled_keep].astype(np.float64)
    flat_z = depth_sample[sampled_keep].astype(np.float64)
    if len(flat_z) < int(args.min_vertices):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), {
            "frame_idx": frame_idx,
            "status": "skipped_too_few_vertices",
            "vertices": int(len(flat_z)),
        }
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    camera_vertices = np.column_stack(((flat_x - cx) * flat_z / fx, (flat_y - cy) * flat_z / fy, flat_z))
    index_grid = np.full(sampled_keep.shape, -1, dtype=np.int32)
    index_grid[sampled_keep] = np.arange(len(camera_vertices), dtype=np.int32)
    faces = build_faces(index_grid, camera_vertices, float(args.max_triangle_edge_m))
    vertices, faces = remove_unreferenced(camera_vertices, faces)
    if len(vertices) < int(args.min_vertices) or len(faces) < int(args.min_faces):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), {
            "frame_idx": frame_idx,
            "status": "skipped_underconstrained_mesh",
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
        }
    return vertices, faces, {
        "frame_idx": frame_idx,
        "status": "ok",
        "mask_pixels": int(np.count_nonzero(mask_bool)),
        "depth_pixels": int(values.size),
        "depth_low_m": lo,
        "depth_high_m": hi,
        "kept_pixels": int(np.count_nonzero(keep)),
        "vertices_camera": int(len(vertices)),
        "faces": int(len(faces)),
        "camera_extent_m": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
    }


def save_archive(path: Path, frame_indices: list[int], vertices: list[np.ndarray], faces: list[np.ndarray]) -> None:
    vertex_offsets = [0]
    face_offsets = [0]
    for v, f in zip(vertices, faces, strict=True):
        vertex_offsets.append(vertex_offsets[-1] + len(v))
        face_offsets.append(face_offsets[-1] + len(f))
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices).astype(np.float32),
        faces=np.vstack(faces).astype(np.int32),
    )


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest)
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{args.manifest} must contain nonempty frames list")
    K = load_intrinsics(args.dataset)
    annotations = annotation_by_frame(args.annotations) if args.annotations is not None else {}
    if args.coordinate == "world" and args.annotations is None:
        raise RuntimeError("--annotations is required when --coordinate world")
    frame_indices = []
    vertices_world = []
    faces_all = []
    rows = []
    for entry in entries:
        frame_idx = int(entry["frame_idx"])
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        if args.coordinate == "world" and frame_idx not in annotations:
            raise RuntimeError(f"missing annotation frame {frame_idx}")
        vertices_camera, faces, row = mesh_from_entry(entry, args.dataset, K, args)
        if row["status"] != "ok":
            rows.append(row)
            continue
        if args.coordinate == "world":
            T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
            if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all():
                raise RuntimeError(f"invalid T_world_camera_metric for frame {frame_idx}")
            vertices_out = transform_points(vertices_camera, T_world_camera)
        elif args.coordinate == "camera":
            vertices_out = vertices_camera
        else:
            raise RuntimeError(f"unsupported coordinate mode: {args.coordinate}")
        row["output_extent_m"] = (vertices_out.max(axis=0) - vertices_out.min(axis=0)).astype(float).tolist()
        rows.append(row)
        frame_indices.append(frame_idx)
        vertices_world.append(vertices_out.astype(np.float32))
        faces_all.append(faces.astype(np.int32))
    if len(frame_indices) < int(args.min_frames):
        raise RuntimeError(f"only {len(frame_indices)} observed-surface frames exported")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "observed_mask_depth_meshes_world.npz"
    save_archive(archive, frame_indices, vertices_world, faces_all)
    ext = np.asarray([row["output_extent_m"] for row in rows if row["status"] == "ok"], dtype=np.float64)
    report = {
        "status": "ok",
        "method": "mask_depth_observed_surface_mesh_archive_v3",
        "coordinate": args.coordinate,
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations) if args.annotations is not None else None,
        "archive": str(archive),
        "frames": int(len(frame_indices)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "intrinsics_fx_fy_cx_cy": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        "output_extent_median_m": np.median(ext, axis=0).astype(float).tolist(),
        "rows": rows,
    }
    (args.output_dir / "qc_mask_depth_observed_mesh_archive_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--coordinate", choices=["world", "camera"], default="world")
    parser.add_argument("--mask-stride", type=int, default=5)
    parser.add_argument("--mask-erode-px", type=int, default=1)
    parser.add_argument("--depth-low-quantile", type=float, default=0.02)
    parser.add_argument("--depth-high-quantile", type=float, default=0.98)
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=3.20)
    parser.add_argument("--min-depth-pixels", type=int, default=2500)
    parser.add_argument("--min-vertices", type=int, default=250)
    parser.add_argument("--min-faces", type=int, default=300)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.040)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
