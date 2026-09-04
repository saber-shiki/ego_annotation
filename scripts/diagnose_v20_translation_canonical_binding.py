#!/usr/bin/env python3
"""Prediction-only diagnosis of V20 translation drift and SAM3D canonical binding.

This script never modifies a formal pose or annotation.  It compares the current
V20 pose applied to the generated completion and to the observed-surface mesh,
then evaluates two isolated counterfactuals:

1. translation-only camera-space correction estimated from P09 visible points
   against the observed surface mesh, with rotation fixed;
2. one static point-to-point rigid canonical binding from generated completion
   to the observed surface mesh, with per-frame V20 pose fixed.

The counterfactuals are diagnostic evidence only.  P09/UniDepth remains
prediction-side and generated SAM3D faces remain render-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def row_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points_world, dtype=np.float64) - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]


def row_from_camera_delta(delta_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    # p_camera = (p_world - t_wc) @ R_wc; therefore d_world @ R_wc=d_camera.
    return np.asarray(delta_camera, dtype=np.float64) @ T_world_camera[:3, :3].T


def transform_canonical(points: np.ndarray, R_world_from_canonical: np.ndarray, t_world: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ R_world_from_canonical.T + np.asarray(t_world, dtype=np.float64)[None, :]


def rays_from_points(points_camera: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    return np.column_stack((
        points[:, 0] / np.maximum(points[:, 2], 1.0e-9),
        points[:, 1] / np.maximum(points[:, 2], 1.0e-9),
        np.ones(len(points), dtype=np.float64),
    ))


def cast(vertices_camera: np.ndarray, faces: np.ndarray, rays: np.ndarray) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(vertices_camera, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)),
    ))
    origins = np.zeros((len(rays), 3), dtype=np.float32)
    query = np.hstack((origins, np.asarray(rays, dtype=np.float32)))
    return scene.cast_rays(o3d.core.Tensor(query))["t_hit"].numpy()


def depth_summary(hit: np.ndarray, observed_z: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(hit) & (hit < 1.0e10) & np.isfinite(observed_z)
    if not np.any(valid):
        return {"ray_count": int(len(hit)), "hit_count": 0}
    delta_mm = (hit[valid] - observed_z[valid]) * 1000.0
    return {
        "ray_count": int(len(hit)),
        "hit_count": int(np.count_nonzero(valid)),
        "front_fraction": float(np.mean(hit[valid] < observed_z[valid])),
        "delta_z_mm": {
            "p10": float(np.percentile(delta_mm, 10.0)),
            "median": float(np.percentile(delta_mm, 50.0)),
            "p90": float(np.percentile(delta_mm, 90.0)),
            "mean": float(np.mean(delta_mm)),
        },
    }


def sample_points(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= int(count):
        return points
    ids = np.linspace(0, len(points) - 1, int(count), dtype=np.int64)
    return points[ids]


def robust_translation_to_target(source: np.ndarray, target: np.ndarray, iterations: int = 6) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate translation d where source+d overlaps target.

    The source is the observed canonical mesh in the current camera frame and
    target is the P09 visible point set.  This is deliberately translation-only
    and trimmed; it is not a pose solver and does not use generated geometry.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 3 or len(target) < 3:
        return np.zeros(3), {"status": "insufficient_points"}
    delta = np.zeros(3, dtype=np.float64)
    last_count = 0
    last_med = None
    for _ in range(int(iterations)):
        tree = cKDTree(source + delta[None, :])
        distance, nearest = tree.query(target, k=1, workers=-1)
        residual = target - (source[nearest] + delta[None, :])
        finite = np.isfinite(distance) & np.isfinite(residual).all(axis=1)
        if np.count_nonzero(finite) < 3:
            break
        finite_residual = residual[finite]
        finite_distance = distance[finite]
        cutoff = float(np.percentile(finite_distance, 75.0))
        keep = finite_distance <= max(cutoff, 0.002)
        if np.count_nonzero(keep) < 3:
            keep = np.ones(len(finite_distance), dtype=bool)
        step = np.median(finite_residual[keep], axis=0)
        delta += step
        last_count = int(np.count_nonzero(keep))
        last_med = float(np.median(finite_distance[keep]))
        if float(np.linalg.norm(step)) < 1.0e-5:
            break
    return delta, {
        "status": "ok",
        "iterations": int(_ + 1),
        "support_count": last_count,
        "median_nearest_distance_m": last_med,
        "correction_camera_m": delta.tolist(),
        "correction_norm_mm": float(np.linalg.norm(delta) * 1000.0),
    }


def kabsch_row(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t so source @ R.T + t ~= target."""
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    x = source - source_mean
    y = target - target_mean
    h = x.T @ y
    u, _, vt = np.linalg.svd(h)
    r_col = vt.T @ u.T
    if np.linalg.det(r_col) < 0.0:
        vt[-1, :] *= -1.0
        r_col = vt.T @ u.T
    t = target_mean - source_mean @ r_col.T
    return r_col, t


def canonical_binding_icp(
    generated: np.ndarray,
    observed: np.ndarray,
    *,
    sample_count: int,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a static rigid generated->observed binding with trimmed ICP."""
    source = sample_points(generated, sample_count)
    target = sample_points(observed, max(1000, min(len(observed), sample_count)))
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    tree = cKDTree(target)
    used = 0
    med = None
    for iteration in range(int(iterations)):
        transformed = source @ R.T + t[None, :]
        distance, ids = tree.query(transformed, k=1, workers=-1)
        finite = np.isfinite(distance)
        if np.count_nonzero(finite) < 3:
            break
        cutoff = float(np.percentile(distance[finite], 70.0))
        keep = finite & (distance <= max(cutoff, 0.004))
        if np.count_nonzero(keep) < 3:
            keep = finite
        dR, dt = kabsch_row(transformed[keep], target[ids[keep]])
        R = dR @ R
        t = t @ dR.T + dt
        used = int(np.count_nonzero(keep))
        med = float(np.median(distance[keep]))
        step_angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec())))
        step_t = float(np.linalg.norm(dt))
        if step_angle < 1.0e-3 and step_t < 1.0e-6:
            break
    transformed = source @ R.T + t[None, :]
    distance, _ = tree.query(transformed, k=1, workers=-1)
    finite = np.isfinite(distance)
    return R, t, {
        "status": "ok",
        "iterations": int(iteration + 1),
        "support_count": used,
        "median_nearest_distance_m": float(np.median(distance[finite])) if np.any(finite) else None,
        "p90_nearest_distance_m": float(np.percentile(distance[finite], 90.0)) if np.any(finite) else None,
        "rotation_angle_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))),
        "translation_m": t.tolist(),
        "translation_norm_mm": float(np.linalg.norm(t) * 1000.0),
        "rotation_matrix": R.tolist(),
        "note": "static diagnostic generated-to-observed binding; not promoted to P13 or formal state",
    }


def project_center(point_camera: np.ndarray, K_values: np.ndarray) -> list[float] | None:
    if not np.isfinite(point_camera).all() or point_camera[2] <= 0.01:
        return None
    return [
        float(K_values[0] * point_camera[0] / point_camera[2] + K_values[2]),
        float(K_values[1] * point_camera[1] / point_camera[2] + K_values[3]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--generated-mesh", type=Path, required=True)
    parser.add_argument("--observed-mesh", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frames", default="0,5,20,60,84,92,100,120,130,140,149")
    parser.add_argument("--max-rays-per-frame", type=int, default=1000)
    parser.add_argument("--icp-sample-count", type=int, default=20000)
    parser.add_argument("--binding-iterations", type=int, default=20)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    frames = {int(f["frame_idx"]): f for f in annotations.get("frames", []) if isinstance(f, dict) and f.get("frame_idx") is not None}
    report = load_json(args.pose_report)
    poses = {int(r["frame_idx"]): r for r in report.get("pose_rows", []) if isinstance(r, dict) and r.get("rotation_world_from_completed_canonical_matrix") is not None}
    generated, generated_faces = load_mesh(args.generated_mesh)
    observed, observed_faces = load_mesh(args.observed_mesh)

    binding_R, binding_t, binding_report = canonical_binding_icp(
        generated,
        observed,
        sample_count=int(args.icp_sample_count),
        iterations=int(args.binding_iterations),
    )
    selected = [int(x) for x in args.frames.split(",") if x.strip() and int(x) in frames and int(x) in poses]
    rows: list[dict[str, Any]] = []
    for frame_idx in selected:
        frame = frames[frame_idx]
        obj = next((o for o in frame.get("objects", []) if o.get("object_id") == "carton_milk"), None)
        geom = obj.get("visible_geometry_candidate") if isinstance(obj, dict) and isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        p09 = np.asarray(geom.get("camera_vertices_sample_m") or [], dtype=np.float64)
        if p09.ndim != 2 or p09.shape[1] != 3 or len(p09) < 3:
            rows.append({"frame_idx": frame_idx, "status": "no_p09_metric_geometry"})
            continue
        if len(p09) > int(args.max_rays_per_frame):
            ids = np.linspace(0, len(p09) - 1, int(args.max_rays_per_frame), dtype=np.int64)
            p09 = p09[ids]
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        K_values = np.asarray(frame["camera"]["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        pose = poses[frame_idx]
        R = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        t_world = np.asarray(pose["translation_world_m"], dtype=np.float64)
        rays = rays_from_points(p09)

        generated_world = transform_canonical(generated, R, t_world)
        observed_world = transform_canonical(observed, R, t_world)
        generated_cam = row_to_camera(generated_world, T)
        observed_cam = row_to_camera(observed_world, T)
        generated_center_cam = np.mean(generated_cam, axis=0)
        observed_center_cam = np.mean(observed_cam, axis=0)
        pose_origin_cam = row_to_camera(t_world[None, :], T)[0]

        raw_generated = depth_summary(cast(generated_cam, generated_faces, rays), p09[:, 2])
        raw_observed = depth_summary(cast(observed_cam, observed_faces, rays), p09[:, 2])

        translation_camera, translation_report = robust_translation_to_target(
            sample_points(observed_cam, min(12000, len(observed_cam))),
            p09,
        )
        generated_translation_cam = generated_cam + translation_camera[None, :]
        observed_translation_cam = observed_cam + translation_camera[None, :]
        corrected_generated = depth_summary(cast(generated_translation_cam, generated_faces, rays), p09[:, 2])
        corrected_observed = depth_summary(cast(observed_translation_cam, observed_faces, rays), p09[:, 2])

        generated_bound_canonical = generated @ binding_R.T + binding_t[None, :]
        bound_world = transform_canonical(generated_bound_canonical, R, t_world)
        bound_cam = row_to_camera(bound_world, T)
        bound_summary = depth_summary(cast(bound_cam, generated_faces, rays), p09[:, 2])
        bound_translation_cam = bound_cam + translation_camera[None, :]
        bound_translation_summary = depth_summary(cast(bound_translation_cam, generated_faces, rays), p09[:, 2])

        delta_world = row_from_camera_delta(translation_camera, T)
        row = {
            "frame_idx": frame_idx,
            "status": "ok",
            "p09_camera_median_m": np.median(p09, axis=0).tolist(),
            "p09_camera_mean_m": np.mean(p09, axis=0).tolist(),
            "pose_translation_world_m": t_world.tolist(),
            "pose_origin_camera_m": pose_origin_cam.tolist(),
            "generated_center_camera_m": generated_center_cam.tolist(),
            "observed_surface_center_camera_m": observed_center_cam.tolist(),
            "projected_centers_px": {
                "p09_median": project_center(np.median(p09, axis=0), K_values),
                "generated_center": project_center(generated_center_cam, K_values),
                "observed_surface_center": project_center(observed_center_cam, K_values),
            },
            "raw": {"generated_mesh": raw_generated, "observed_surface_mesh": raw_observed},
            "translation_only_counterfactual": {
                "rotation_fixed": True,
                "source": "P09 visible points matched to observed-surface mesh only",
                "correction_camera_m": translation_camera.tolist(),
                "correction_world_m": delta_world.tolist(),
                "corrected_generated_mesh": corrected_generated,
                "corrected_observed_surface_mesh": corrected_observed,
                "fit": translation_report,
            },
            "static_canonical_binding_counterfactual": {
                "rotation_fixed_per_frame": True,
                "binding": binding_report,
                "corrected_generated_mesh": bound_summary,
                "binding_plus_translation_only": bound_translation_summary,
            },
            "interpretation": {
                "generated_mesh_is_physical_authority": False,
                "p09_is_prediction_side": True,
                "formal_state_modified": False,
                "translation_only_does_not_change_rotation": True,
            },
        }
        rows.append(row)
        print(json.dumps({
            "frame_idx": frame_idx,
            "raw_generated_median_mm": raw_generated.get("delta_z_mm", {}).get("median"),
            "raw_observed_median_mm": raw_observed.get("delta_z_mm", {}).get("median"),
            "translation_camera_mm": (translation_camera * 1000.0).tolist(),
            "translation_only_generated_median_mm": corrected_generated.get("delta_z_mm", {}).get("median"),
            "binding_generated_median_mm": bound_summary.get("delta_z_mm", {}).get("median"),
            "binding_plus_translation_median_mm": bound_translation_summary.get("delta_z_mm", {}).get("median"),
        }, ensure_ascii=False), flush=True)

    output = {
        "schema": "v20_translation_canonical_binding_diagnostic_v1",
        "status": "ok",
        "diagnostic_only": True,
        "formal_state_modified": False,
        "generated_mesh_role": "render_only_completion_hypothesis",
        "p09_role": "prediction_side_visible_first_hit_surface",
        "inputs": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "pose_report": str(args.pose_report.expanduser().resolve()),
            "generated_mesh": str(args.generated_mesh.expanduser().resolve()),
            "observed_mesh": str(args.observed_mesh.expanduser().resolve()),
            "sha256": {
                "annotations": sha256_file(args.annotations),
                "pose_report": sha256_file(args.pose_report),
                "generated_mesh": sha256_file(args.generated_mesh),
                "observed_mesh": sha256_file(args.observed_mesh),
            },
        },
        "parameters": {
            "max_rays_per_frame": int(args.max_rays_per_frame),
            "icp_sample_count": int(args.icp_sample_count),
            "binding_iterations": int(args.binding_iterations),
            "translation_estimator": "trimmed nearest-neighbor median translation only",
            "canonical_binding_estimator": "trimmed point-to-point rigid ICP generated-to-observed",
        },
        "static_canonical_binding": binding_report,
        "frames": rows,
        "policy": {
            "gt_used": False,
            "generated_mesh_used_as_solver_authority": False,
            "translation_correction_promoted": False,
            "canonical_binding_promoted": False,
            "note": "Both corrections are counterfactual evidence. No pose, formal annotation, collision, contact, P15, D18, D19, or RRD authority is modified.",
        },
    }
    args.output_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.expanduser().resolve().write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "frame_count": len(rows), "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
