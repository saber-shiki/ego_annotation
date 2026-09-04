#!/usr/bin/env python3
"""Diagnose prediction-side SAM3D/P09 camera-ray depth ordering.

The optional HOT3D depth file is read only for post-hoc comparison.  It is
never used to alter a pose or produce a candidate.  The main diagnostic uses
only the annotation P09 camera points, camera transform, generated mesh, and
selected V20 pose report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path.expanduser().resolve()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"invalid mesh: {path}")
    return vertices, faces


def cast(mesh_vertices_camera: np.ndarray, faces: np.ndarray, rays: np.ndarray) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(mesh_vertices_camera.astype(np.float32)),
            o3d.core.Tensor(faces.astype(np.uint32)),
        )
    )
    origins = np.zeros((len(rays), 3), dtype=np.float32)
    values = np.hstack([origins, rays.astype(np.float32)])
    return scene.cast_rays(o3d.core.Tensor(values))["t_hit"].numpy()


def summarize(hit: np.ndarray, observed_z: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(hit) & (hit < 1.0e10) & np.isfinite(observed_z)
    if not np.any(valid):
        return {"ray_count": int(len(hit)), "hit_count": 0}
    delta_mm = (hit[valid] - observed_z[valid]) * 1000.0
    return {
        "ray_count": int(len(hit)),
        "hit_count": int(np.count_nonzero(valid)),
        "front_fraction": float(np.mean(hit[valid] < observed_z[valid])),
        "delta_z_mm": {
            "p10": float(np.percentile(delta_mm, 10)),
            "median": float(np.percentile(delta_mm, 50)),
            "p90": float(np.percentile(delta_mm, 90)),
            "mean": float(np.mean(delta_mm)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--generated-mesh", type=Path, required=True)
    parser.add_argument("--observed-mesh", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--gt-depth-npz", type=Path, default=None)
    parser.add_argument("--frames", default="0,5,20,60,84,92,100,120,130,140,149")
    parser.add_argument("--max-rays-per-frame", type=int, default=1000)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    frames = {int(f["frame_idx"]): f for f in annotations.get("frames", []) if isinstance(f, dict) and f.get("frame_idx") is not None}
    report = load_json(args.pose_report)
    poses = {int(r["frame_idx"]): r for r in report.get("pose_rows", []) if isinstance(r, dict) and r.get("rotation_world_from_completed_canonical_matrix") is not None}
    generated_vertices, generated_faces = load_mesh(args.generated_mesh)
    observed_vertices, observed_faces = load_mesh(args.observed_mesh)
    requested = [int(x) for x in args.frames.split(",") if x.strip()]
    selected = [idx for idx in requested if idx in frames and idx in poses]
    if not selected:
        raise RuntimeError("no selected frames have both annotation and pose")

    gt_depth = None
    gt_valid = None
    gt_K = None
    if args.gt_depth_npz is not None:
        with np.load(args.gt_depth_npz.expanduser().resolve(), allow_pickle=False) as data:
            gt_depth = np.asarray(data["depth_m"], dtype=np.float64)
            gt_valid = np.asarray(data["valid_mask"], dtype=bool)
            gt_K = np.asarray(data["K"], dtype=np.float64)

    rows = []
    for frame_idx in selected:
        frame = frames[frame_idx]
        obj = next((o for o in frame.get("objects", []) if o.get("object_id") == "carton_milk"), None)
        geom = obj.get("visible_geometry_candidate") if isinstance(obj, dict) and isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        p09_camera = np.asarray(geom.get("camera_vertices_sample_m") or [], dtype=np.float64)
        if p09_camera.ndim != 2 or p09_camera.shape[1] != 3 or len(p09_camera) < 3:
            continue
        if len(p09_camera) > int(args.max_rays_per_frame):
            indices = np.linspace(0, len(p09_camera) - 1, int(args.max_rays_per_frame), dtype=np.int64)
            p09_camera = p09_camera[indices]
        K_values = np.asarray(frame["camera"].get("intrinsics_fx_fy_cx_cy") or [], dtype=np.float64)
        K = np.asarray([[K_values[0], 0.0, K_values[2]], [0.0, K_values[1], K_values[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
        rays = np.column_stack((p09_camera[:, 0] / p09_camera[:, 2], p09_camera[:, 1] / p09_camera[:, 2], np.ones(len(p09_camera))))
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        pose = poses[frame_idx]
        R = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        t = np.asarray(pose["translation_world_m"], dtype=np.float64)
        generated_world = generated_vertices @ R.T + t[None, :]
        observed_world = observed_vertices @ R.T + t[None, :]
        generated_camera = (generated_world - T[:3, 3][None, :]) @ T[:3, :3]
        observed_camera = (observed_world - T[:3, 3][None, :]) @ T[:3, :3]
        generated_hit = cast(generated_camera, generated_faces, rays)
        observed_hit = cast(observed_camera, observed_faces, rays)
        row = {
            "frame_idx": frame_idx,
            "p09_camera_z_m": {
                "p10": float(np.percentile(p09_camera[:, 2], 10)),
                "median": float(np.percentile(p09_camera[:, 2], 50)),
                "p90": float(np.percentile(p09_camera[:, 2], 90)),
            },
            "generated_mesh": summarize(generated_hit, p09_camera[:, 2]),
            "observed_mesh": summarize(observed_hit, p09_camera[:, 2]),
        }
        if gt_depth is not None and gt_valid is not None and gt_K is not None:
            # GT is only sampled for a post-hoc diagnostic at the P09 mask
            # pixels.  It never enters the above camera-ray calculation.
            mask_path = str(geom.get("mask_path") or "")
            import cv2
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                ys, xs = np.where(mask > 0)
                xi = np.clip(np.rint((xs + 0.5) * 1408.0 / mask.shape[1] - 0.5).astype(np.int64), 0, 1407)
                yi = np.clip(np.rint((ys + 0.5) * 1408.0 / mask.shape[0] - 0.5).astype(np.int64), 0, 1407)
                valid = gt_valid[frame_idx, yi, xi]
                z = gt_depth[frame_idx, yi, xi][valid]
                if len(z):
                    row["hot3d_gt_posthoc"] = {
                        "sample_count": int(len(z)),
                        "median_z_m": float(np.median(z)),
                        "p09_minus_gt_median_mm": float((np.median(geom.get("depth_median_m") or p09_camera[:, 2].mean()) - np.median(z)) * 1000.0),
                    }
        rows.append(row)
        print(frame_idx, row["generated_mesh"], row["observed_mesh"], flush=True)

    output = {
        "schema": "v20_prediction_depth_order_diagnostic_v1",
        "status": "ok",
        "gt_used_only_for_posthoc_evaluation": args.gt_depth_npz is not None,
        "gt_used_as_solver_input": False,
        "generated_mesh_role": "render_only_completion_hypothesis",
        "inputs": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "pose_report": str(args.pose_report.expanduser().resolve()),
            "generated_mesh": str(args.generated_mesh.expanduser().resolve()),
            "observed_mesh": str(args.observed_mesh.expanduser().resolve()),
            "gt_depth_npz_posthoc": str(args.gt_depth_npz.expanduser().resolve()) if args.gt_depth_npz else None,
        },
        "frames": rows,
        "interpretation": {
            "front_fraction_definition": "generated/observed mesh first ray hit has camera-z smaller than P09 observed camera-z",
            "positive_delta_definition": "mesh is farther from camera than P09 point on the sampled ray",
            "warning": "P09 uses prediction-side UniDepth; a front/back violation can be caused by depth scale bias, pose error, canonical completion mismatch, or any combination.",
            "policy": "This diagnostic does not modify pose and never promotes generated faces to physical authority.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "frame_count": len(rows), "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
