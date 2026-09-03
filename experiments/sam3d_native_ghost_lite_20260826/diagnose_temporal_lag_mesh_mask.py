#!/usr/bin/env python3
"""Diagnose whether mesh/mask mismatch behaves like temporal lag.

A static pose error changes sign/magnitude with viewpoint; a temporal lag should
show up as residual ~= image velocity * tau and as a positive optimal shift when
current observed masks are compared with past rendered masks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path.expanduser().resolve(), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return mesh


def resize_intrinsics(K: np.ndarray, source_size: tuple[int, int], render_size: tuple[int, int]) -> np.ndarray:
    source_w, source_h = source_size
    render_w, render_h = render_size
    sx = render_w / source_w
    sy = render_h / source_h
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = sx * (out[0, 2] + 0.5) - 0.5
    out[1, 2] = sy * (out[1, 2] + 0.5) - 0.5
    return out


def world_to_camera(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]


def camera_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]


def anchor_camera_to_canonical(vertices: np.ndarray, T: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    world = camera_to_world(vertices, T)
    return (world - t[None, :]) @ R


def canonical_to_camera(canonical: np.ndarray, R: np.ndarray, t: np.ndarray, T: np.ndarray) -> np.ndarray:
    world = canonical @ R.T + t[None, :]
    return world_to_camera(world, T)


def make_rasterizer(K: np.ndarray, width: int, height: int, device: str):
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

    cameras = cameras_from_opencv_projection(
        torch.eye(3, dtype=torch.float32, device=device)[None],
        torch.zeros(1, 3, dtype=torch.float32, device=device),
        torch.tensor(K, dtype=torch.float32, device=device)[None],
        torch.tensor([[height, width]], dtype=torch.float32, device=device),
    )
    return MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            bin_size=0,
        ),
    )


def render_depth(rasterizer: Any, vertices: np.ndarray, faces: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    from pytorch3d.structures import Meshes

    with torch.no_grad():
        fragments = rasterizer(Meshes(
            verts=[torch.tensor(vertices, dtype=torch.float32, device=device)],
            faces=[torch.tensor(faces, dtype=torch.int64, device=device)],
        ))
    pix = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    zbuf = fragments.zbuf[0, ..., 0].detach().cpu().numpy().astype(np.float64)
    valid = (pix >= 0) & np.isfinite(zbuf) & (zbuf > 1.0e-6)
    return valid, np.where(valid, zbuf, np.nan)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    interpolation = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=interpolation) > 0


def mask_features(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "pixels": 0,
            "centroid_px": [None, None],
            "bbox_wh": [None, None],
            "orientation_rad": None,
            "aspect": None,
        }
    coords = np.column_stack((xs, ys)).astype(np.float64)
    centroid = coords.mean(axis=0)
    centered = coords - centroid[None, :]
    covariance = centered.T @ centered / max(1, len(coords) - 1)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    orientation = float(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    bbox_w = float(xs.max() - xs.min() + 1)
    bbox_h = float(ys.max() - ys.min() + 1)
    return {
        "pixels": int(len(xs)),
        "centroid_px": centroid.astype(float).tolist(),
        "bbox_wh": [bbox_w, bbox_h],
        "orientation_rad": orientation,
        "aspect": float(bbox_w / max(1.0, bbox_h)),
        "eigenvalue_ratio": float(np.sqrt(max(eigvals[1], 0.0) / max(eigvals[0], 1.0e-12))),
    }


def centroid_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        c = row[key]["centroid_px"]
        vals.append([np.nan, np.nan] if c[0] is None else c)
    return np.asarray(vals, dtype=np.float64)


def valid_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return a[keep], b[keep]


def best_integer_lag(observed: np.ndarray, rendered: np.ndarray, max_lag: int) -> dict[str, Any]:
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        # Positive lag compares observed[t] with rendered[t-lag]: positive means rendered is behind.
        obs_indices = []
        ren_indices = []
        for t in range(len(observed)):
            r = t - lag
            if 0 <= r < len(rendered):
                obs_indices.append(t)
                ren_indices.append(r)
        obs = observed[np.asarray(obs_indices, dtype=np.int64)]
        ren = rendered[np.asarray(ren_indices, dtype=np.int64)]
        obs, ren = valid_pairs(obs, ren)
        if len(obs) < 5:
            continue
        err = np.linalg.norm(obs - ren, axis=1)
        rows.append({
            "lag_frames": int(lag),
            "count": int(len(err)),
            "centroid_rms_px": float(np.sqrt(np.mean(err * err))),
            "centroid_median_px": float(np.median(err)),
        })
    best = min(rows, key=lambda row: row["centroid_rms_px"])
    return {"best": best, "rows": rows, "positive_lag_semantics": "rendered[t-lag] matches observed[t]; positive means rendered sequence lags observed"}


def continuous_lag_from_velocity(observed: np.ndarray, rendered: np.ndarray, fps: float) -> dict[str, Any]:
    obs, ren = valid_pairs(observed, rendered)
    # Keep original valid indices for central differences.
    valid = np.isfinite(observed).all(axis=1) & np.isfinite(rendered).all(axis=1)
    idx = np.nonzero(valid)[0]
    residual = observed[valid] - rendered[valid]
    velocity = np.full_like(observed, np.nan)
    velocity[1:-1] = (observed[2:] - observed[:-2]) * fps / 2.0
    velocity = velocity[valid]
    speed = np.linalg.norm(velocity, axis=1)
    finite = np.isfinite(velocity).all(axis=1) & (speed > 1.0e-6)
    idx = idx[finite]
    residual = residual[finite]
    velocity = velocity[finite]
    speed = speed[finite]
    if len(speed) < 5:
        return {"count": 0}
    tau_ls = float(np.sum(residual * velocity) / np.sum(velocity * velocity))
    per_component = []
    for axis in range(2):
        v = velocity[:, axis]
        e = residual[:, axis]
        keep = np.abs(v) > np.percentile(np.abs(v), 50.0)
        per_component.append(float(np.sum(e[keep] * v[keep]) / np.sum(v[keep] * v[keep])) if np.count_nonzero(keep) else None)
    high = speed >= np.percentile(speed, 75.0)
    tau_high = float(np.sum(residual[high] * velocity[high]) / np.sum(velocity[high] * velocity[high])) if np.count_nonzero(high) else None
    return {
        "count": int(len(speed)),
        "tau_seconds_least_squares": tau_ls,
        "tau_frames_least_squares": float(tau_ls * fps),
        "tau_seconds_high_speed": tau_high,
        "tau_frames_high_speed": float(tau_high * fps) if tau_high is not None else None,
        "tau_frames_per_component": per_component,
        "speed_median_px_s": float(np.median(speed)),
        "speed_p75_px_s": float(np.percentile(speed, 75.0)),
        "positive_semantics": "positive tau means rendered/mesh centroid lags observed mask centroid",
    }


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(inter / max(1, union))


def shifted_iou_lag(observed_masks: list[np.ndarray], rendered_masks: list[np.ndarray], max_lag: int) -> dict[str, Any]:
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        vals = []
        for t in range(len(observed_masks)):
            r = t - lag
            if 0 <= r < len(rendered_masks):
                vals.append(iou(observed_masks[t], rendered_masks[r]))
        rows.append({
            "lag_frames": int(lag),
            "count": int(len(vals)),
            "iou_mean": float(np.mean(vals)),
            "iou_median": float(np.median(vals)),
        })
    best = max(rows, key=lambda row: row["iou_mean"])
    return {"best": best, "rows": rows, "positive_lag_semantics": "rendered[t-lag] compared with observed[t]; positive means rendered lags"}


def window_shift_summary(
    observed_masks: list[np.ndarray],
    rendered_masks: list[np.ndarray],
    frame_indices: list[int],
    windows: list[tuple[int, int]],
    max_lag: int,
) -> dict[str, Any]:
    out = {}
    selected_positions_by_window = {
        (lo, hi): [pos for pos, frame_idx in enumerate(frame_indices) if lo <= frame_idx <= hi]
        for lo, hi in windows
    }
    for (lo, hi), positions in selected_positions_by_window.items():
        best = None
        for lag in range(-max_lag, max_lag + 1):
            vals = []
            for pos in positions:
                r = pos - lag
                if 0 <= r < len(rendered_masks):
                    vals.append(iou(observed_masks[pos], rendered_masks[r]))
            if not vals:
                continue
            score = float(np.mean(vals))
            if best is None or score > best["iou_mean"]:
                best = {"lag_frames": int(lag), "iou_mean": score, "count": int(len(vals))}
        out[f"{lo}-{hi}"] = best
    return out


def translation_regularization_lag(p14_report: dict[str, Any], fps: float) -> dict[str, Any]:
    rows = []
    for row in p14_report.get("pose_rows", []):
        reg = row.get("translation_regularization") if isinstance(row, dict) else None
        if not isinstance(reg, dict):
            continue
        meas = np.asarray(reg.get("measurement_translation_world_m") or [], dtype=np.float64)
        regularized = np.asarray(reg.get("regularized_translation_world_m") or [], dtype=np.float64)
        if meas.shape == (3,) and regularized.shape == (3,):
            rows.append((int(row["frame_idx"]), meas, regularized))
    if len(rows) < 5:
        return {"count": 0}
    rows.sort(key=lambda row: row[0])
    frame_idx = np.asarray([row[0] for row in rows], dtype=np.int64)
    measurement = np.asarray([row[1] for row in rows], dtype=np.float64)
    regularized = np.asarray([row[2] for row in rows], dtype=np.float64)
    velocity = np.zeros_like(measurement)
    velocity[1:-1] = (measurement[2:] - measurement[:-2]) * fps / 2.0
    residual = measurement - regularized
    valid = (frame_idx[1:] - frame_idx[:-1]) == 1
    inner = np.zeros(len(rows), dtype=bool)
    inner[1:-1] = valid[:-1] & valid[1:]
    residual = residual[inner]
    velocity = velocity[inner]
    speed = np.linalg.norm(velocity, axis=1)
    keep = speed > np.percentile(speed, 50.0)
    tau = float(np.sum(residual[keep] * velocity[keep]) / np.sum(velocity[keep] * velocity[keep]))
    return {
        "count": int(len(rows)),
        "used_count": int(np.count_nonzero(keep)),
        "tau_seconds": tau,
        "tau_frames": float(tau * fps),
        "residual_median_m": float(np.median(np.linalg.norm(residual, axis=1))),
        "residual_p95_m": float(np.percentile(np.linalg.norm(residual, axis=1), 95.0)),
        "positive_semantics": "positive tau means temporally regularized P14 translation lags its per-frame measurement",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-graph", type=Path, required=True)
    p.add_argument("--p14-report", type=Path, required=True)
    p.add_argument("--mesh-anchor-camera", type=Path, required=True)
    p.add_argument("--hand-npz", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--anchor-frame", type=int, default=92)
    p.add_argument("--render-size", type=int, default=480)
    p.add_argument("--max-lag-frames", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_json(args.annotations)
    pose_graph = load_json(args.pose_graph)
    p14_report = load_json(args.p14_report)
    mesh = load_mesh(args.mesh_anchor_camera)
    pose_by_idx = {int(row["frame_idx"]): row for row in pose_graph.get("pose_rows", [])}
    anchor = pose_by_idx[int(args.anchor_frame)]
    first_frame = annotations["frames"][0]
    K_source = np.asarray(first_frame["camera"]["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    K_source = np.asarray([[K_source[0], 0.0, K_source[2]], [0.0, K_source[1], K_source[3]], [0.0, 0.0, 1.0]])
    source_size = (int(first_frame["source_width"]), int(first_frame["source_height"]))
    render_size = (int(args.render_size), int(args.render_size))
    K_render = resize_intrinsics(K_source, source_size, render_size)
    T_anchor = np.asarray(next(frame for frame in annotations["frames"] if int(frame["frame_idx"]) == args.anchor_frame)["camera"]["T_world_camera_metric"], dtype=np.float64)
    canonical = anchor_camera_to_canonical(
        np.asarray(mesh.vertices, dtype=np.float64),
        T_anchor,
        np.asarray(anchor["rotation_world_from_completed_canonical_matrix"], dtype=np.float64),
        np.asarray(anchor["translation_world_m"], dtype=np.float64),
    )
    rasterizer = make_rasterizer(K_render, render_size[0], render_size[1], str(args.device))
    hands = np.load(args.hand_npz)
    hand_position = {int(idx): pos for pos, idx in enumerate(hands["frame_idx"].astype(int).tolist())}

    rows: list[dict[str, Any]] = []
    observed_masks: list[np.ndarray] = []
    rendered_masks: list[np.ndarray] = []
    frame_indices: list[int] = []
    for frame in annotations.get("frames", []):
        idx = int(frame["frame_idx"])
        if idx not in pose_by_idx:
            continue
        pose = pose_by_idx[idx]
        if "rotation_world_from_completed_canonical_matrix" not in pose or "translation_world_m" not in pose:
            continue
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        R = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        t = np.asarray(pose["translation_world_m"], dtype=np.float64)
        object_camera = canonical_to_camera(canonical, R, t, T)
        object_rendered, object_depth = render_depth(rasterizer, object_camera, np.asarray(mesh.faces), str(args.device))
        hand_depths = []
        if idx in hand_position:
            pos = hand_position[idx]
            offset = 0
            hand_vertices = []
            hand_faces = []
            for side in ("left", "right"):
                if int(np.asarray(hands[f"{side}_valid"])[pos]) != 1:
                    continue
                vertices_world = np.asarray(hands[f"{side}_vertices_world_m"][pos], dtype=np.float64)
                vertices_camera = world_to_camera(vertices_world, T)
                faces = np.asarray(hands[f"{side}_faces"], dtype=np.int64) + offset
                hand_vertices.append(vertices_camera)
                hand_faces.append(faces)
                offset += len(vertices_camera)
            if hand_vertices:
                _, hand_depth = render_depth(rasterizer, np.concatenate(hand_vertices, axis=0), np.concatenate(hand_faces, axis=0), str(args.device))
                hand_depths.append(hand_depth)
        visible_rendered = object_rendered.copy()
        if hand_depths:
            hand_depth = hand_depths[0]
            hand_hit = np.isfinite(hand_depth)
            visible_rendered &= ~(hand_hit & (hand_depth < object_depth - 0.0015))

        obj = frame["objects"][0]
        mask_raw = cv2.imread(str(obj.get("mask_path") or obj.get("visible_geometry_candidate", {}).get("mask_path")), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise RuntimeError(f"failed to read mask for frame {idx}")
        observed_mask = resize_mask(mask_raw > 0, render_size[0], render_size[1])
        observed_features = mask_features(observed_mask)
        rendered_features = mask_features(visible_rendered)
        visible = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        surfels = np.asarray(visible.get("camera_vertices_sample_m") or [], dtype=np.float64)
        surfel_centroid = [None, None]
        if surfels.ndim == 2 and len(surfels):
            z = np.maximum(surfels[:, 2], 1.0e-9)
            uv = np.column_stack((K_render[0, 0] * surfels[:, 0] / z + K_render[0, 2], K_render[1, 1] * surfels[:, 1] / z + K_render[1, 2]))
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < render_size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < render_size[1])
            if np.count_nonzero(inside):
                surfel_centroid = np.mean(uv[inside], axis=0).astype(float).tolist()
        rows.append({
            "frame_idx": idx,
            "observed_mask": observed_features,
            "rendered_mesh": rendered_features,
            "surfel_centroid_px": surfel_centroid,
            "iou": iou(observed_mask, visible_rendered),
        })
        observed_masks.append(observed_mask)
        rendered_masks.append(visible_rendered)
        frame_indices.append(idx)

    fps = 30.0
    if annotations.get("frames") and len(annotations["frames"]) > 1:
        times = np.asarray([float(frame.get("time_s") or 0.0) for frame in annotations["frames"]], dtype=np.float64)
        finite = times[np.isfinite(times)]
        if len(finite) > 1 and np.max(finite) > np.min(finite):
            fps = float((len(finite) - 1) / (np.max(finite) - np.min(finite)))
    observed_centroid = centroid_array(rows, "observed_mask")
    rendered_centroid = centroid_array(rows, "rendered_mesh")
    surfel_centroid = np.asarray([[np.nan, np.nan] if row["surfel_centroid_px"][0] is None else row["surfel_centroid_px"] for row in rows], dtype=np.float64)
    report = {
        "method": "diagnose_temporal_lag_mesh_mask",
        "inputs": {
            "annotations": str(args.annotations),
            "pose_graph": str(args.pose_graph),
            "p14_report": str(args.p14_report),
            "mesh_anchor_camera": str(args.mesh_anchor_camera),
            "hand_npz": str(args.hand_npz),
        },
        "render_size": render_size,
        "fps": fps,
        "frame_count": int(len(rows)),
        "centroid_lag": {
            "mesh_vs_mask_integer": best_integer_lag(observed_centroid, rendered_centroid, int(args.max_lag_frames)),
            "mesh_vs_mask_continuous_velocity": continuous_lag_from_velocity(observed_centroid, rendered_centroid, fps),
            "surfel_vs_mask_integer": best_integer_lag(observed_centroid, surfel_centroid, int(args.max_lag_frames)),
            "surfel_vs_mask_continuous_velocity": continuous_lag_from_velocity(observed_centroid, surfel_centroid, fps),
        },
        "mask_iou_lag": shifted_iou_lag(observed_masks, rendered_masks, int(args.max_lag_frames)),
        "window_mask_iou_lag": window_shift_summary(observed_masks, rendered_masks, frame_indices, [(0, 83), (84, 127), (128, 149), (132, 149)], int(args.max_lag_frames)),
        "p14_translation_regularization_lag": translation_regularization_lag(p14_report, fps),
        "per_frame": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "mesh_vs_mask_integer_best": report["centroid_lag"]["mesh_vs_mask_integer"]["best"],
        "mesh_vs_mask_continuous": report["centroid_lag"]["mesh_vs_mask_continuous_velocity"],
        "mask_iou_lag_best": report["mask_iou_lag"]["best"],
        "window_mask_iou_lag": report["window_mask_iou_lag"],
        "p14_translation_regularization_lag": report["p14_translation_regularization_lag"],
        "output": str(args.output_json),
    }, indent=2))


if __name__ == "__main__":
    main()
