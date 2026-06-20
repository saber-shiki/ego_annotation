#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Render a continuous joint MANO interval correction sequence.

This renderer is intentionally narrow: it shows the visual consequence of the
joint MANO trajectory solver.  It draws original current MANO and the optimized
continuous MANO trajectory over the same raw frames and in a local metric world
view.  It is for consuming the correction as a hand annotation, not for reporting
pipeline containers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

DEFAULT_ANNOTATIONS = Path(
    "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/"
    "task5_tomato_960/annotations_v18_full.json"
)
DEFAULT_POSE_REPORT = Path(
    "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/"
    "pose_fit_frame929prior_frame806scale_v1_from_tracked/v18_compact_rigid_object_pose_fit_report.json"
)
DEFAULT_MESH = Path(
    "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/"
    "completed_mesh_frame929prior_frame806scale_v1/object_obj_tomato_scale_sane_completed_mesh_labeled.ply"
)
DEFAULT_STATE = Path(
    "/data2/ego_annotation_outputs/v18_task5_joint_mano_interval_solver_v1/task5_tomato_960/"
    "v18_joint_mano_interval_trajectory_state.json"
)
DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_task5_joint_mano_interval_solver_render_v1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="task5_tomato_960")
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--pose-report", type=Path, default=DEFAULT_POSE_REPORT)
    p.add_argument("--completed-mesh", type=Path, default=DEFAULT_MESH)
    p.add_argument("--joint-mano-state", type=Path, default=DEFAULT_STATE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--mesh-stride", type=int, default=18)
    p.add_argument("--vertex-stride", type=int, default=2)
    p.add_argument("--padding-m", type=float, default=0.08)
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mesh_vertices(path: Path) -> np.ndarray:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [m for m in geom.geometry.values() if isinstance(m, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f"unsupported mesh type {type(geom)}")
    return np.asarray(geom.vertices, dtype=float)


def pose_map(report: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in report.get("pose_rows", []) if isinstance(report, dict) else []:
        if isinstance(row, dict) and row.get("status") == "fit_to_visible_depth_samples":
            out[int(row["frame_idx"])] = (
                np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=float),
                np.asarray(row["translation_world_m"], dtype=float),
            )
    return out


def state_map(state: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in state.get("per_frame_states", []) if isinstance(state, dict) else []:
        if isinstance(row, dict):
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return out


def project_camera(points_camera: np.ndarray, intr: tuple[float, float, float, float], width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intr
    z = points_camera[:, 2]
    valid = z > 1.0e-4
    u = (fx * points_camera[:, 0] / np.maximum(z, 1.0e-6) + cx).astype(np.int32)
    v = (fy * points_camera[:, 1] / np.maximum(z, 1.0e-6) + cy).astype(np.int32)
    valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def world_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (points_world - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]


def colors_for_side(side: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if side == "left":
        return (255, 80, 0), (255, 255, 0)  # original blue, corrected cyan in BGR
    return (0, 120, 255), (0, 255, 255)  # original orange, corrected yellow in BGR


def draw_skeleton(image: np.ndarray, joints_camera: np.ndarray, intr: tuple[float, float, float, float], color: tuple[int, int, int], width_px: int) -> None:
    h, w = image.shape[:2]
    u, v, valid = project_camera(joints_camera, intr, w, h)
    for a, b in HAND_EDGES:
        if valid[a] and valid[b]:
            cv2.line(image, (int(u[a]), int(v[a])), (int(u[b]), int(v[b])), color, width_px)
    for i in range(min(21, len(u))):
        if valid[i]:
            cv2.circle(image, (int(u[i]), int(v[i])), max(2, width_px), color, -1)


def world_bounds(points: list[np.ndarray], padding: float) -> tuple[np.ndarray, np.ndarray]:
    valid = [p.reshape(-1, 3) for p in points if isinstance(p, np.ndarray) and p.size and p.reshape(-1, 3).shape[0] > 0]
    if not valid:
        return np.array([-1, -1, -1], dtype=float), np.array([1, 1, 1], dtype=float)
    allp = np.vstack(valid)
    return allp.min(axis=0) - padding, allp.max(axis=0) + padding


def world_point(p: np.ndarray, mn: np.ndarray, mx: np.ndarray, w: int, h: int) -> tuple[int, int] | None:
    extent = np.maximum(mx - mn, 1.0e-6)
    x = int(round((p[0] - mn[0]) / extent[0] * w))
    y = int(round(h - (p[2] - mn[2]) / extent[2] * h))
    if 0 <= x < w and 0 <= y < h:
        return x, y
    return None


def draw_world_skeleton(image: np.ndarray, joints: np.ndarray, mn: np.ndarray, mx: np.ndarray, color: tuple[int, int, int], width_px: int) -> None:
    h, w = image.shape[:2]
    for a, b in HAND_EDGES:
        pa = world_point(joints[a], mn, mx, w, h)
        pb = world_point(joints[b], mn, mx, w, h)
        if pa is not None and pb is not None:
            cv2.line(image, pa, pb, color, width_px)
    for j in joints:
        p = world_point(j, mn, mx, w, h)
        if p is not None:
            cv2.circle(image, p, max(2, width_px), color, -1)


def encode(frame_dir: Path, out: Path, fps: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frame_dir / "%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)
    ], check=True)


def render(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    state = load_json(args.joint_mano_state)
    poses = pose_map(load_json(args.pose_report))
    mesh = load_mesh_vertices(args.completed_mesh)
    states = state_map(state)
    frames = [f for f in annotations.get("frames", []) if isinstance(f, dict)]
    frames_by_idx = {int(f["frame_idx"]): f for f in frames}
    frame_ids = sorted({k[0] for k in states})
    if not frame_ids:
        raise RuntimeError("joint state has no per-frame states")
    case_dir = args.output_root / str(args.case)
    overlay_dir = case_dir / "overlay_frames"
    world_dir = case_dir / "world_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    world_dir.mkdir(parents=True, exist_ok=True)
    fps = float((annotations.get("raw_video") or {}).get("fps", 30.0))
    rendered = 0
    for out_i, frame_idx in enumerate(frame_ids):
        frame = frames_by_idx[frame_idx]
        raw = cv2.imread(str(frame.get("raw_frame_path", "")))
        if raw is None:
            raw = np.zeros((1080, 1920, 3), dtype=np.uint8)
        overlay = raw.copy()
        height, width = overlay.shape[:2]
        T = np.asarray((frame.get("camera") or {}).get("T_world_camera_metric", np.eye(4)), dtype=float)
        object_points: np.ndarray | None = None
        if frame_idx in poses:
            R, t = poses[frame_idx]
            object_points = mesh[:: max(1, int(args.mesh_stride))] @ R.T + t[None, :]
            cam = world_to_camera(object_points, T)
            # Use first hand intrinsics for object dots.
            intr_any = None
            for hand in frame.get("hands", []):
                intr = ((hand.get("metric_mano_state") or {}).get("current_v18_camera_intrinsics_fx_fy_cx_cy"))
                if isinstance(intr, list) and len(intr) == 4:
                    intr_any = tuple(float(x) for x in intr)
                    break
            if intr_any is not None:
                u, v, valid = project_camera(cam, intr_any, width, height)
                for x, y in zip(u[valid], v[valid]):
                    cv2.circle(overlay, (int(x), int(y)), 1, (40, 210, 60), -1)
        world_chunks: list[np.ndarray] = []
        if object_points is not None:
            world_chunks.append(object_points)
        for hand in frame.get("hands", []):
            side = str(hand.get("hand_side"))
            st = states.get((frame_idx, side))
            if st is None:
                continue
            metric = hand.get("metric_mano_state") or {}
            joints_cam = np.asarray(metric.get("joints_current_v18_camera_m") or [], dtype=float)
            joints_world = np.asarray(metric.get("joints_current_v18_world_m") or [], dtype=float)
            opt_world = np.asarray(st.get("optimized_joints_world_m") or [], dtype=float)
            opt_verts = np.asarray(st.get("optimized_vertices_world_sample_m") or [], dtype=float)
            intr = metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
            if not (isinstance(intr, list) and len(intr) == 4 and joints_cam.shape == (21, 3) and joints_world.shape == (21, 3) and opt_world.shape == (21, 3)):
                continue
            intr_tuple = tuple(float(x) for x in intr)
            original_color, corrected_color = colors_for_side(side)
            draw_skeleton(overlay, joints_cam, intr_tuple, original_color, 4)  # original current MANO
            opt_cam = world_to_camera(opt_world, T)
            draw_skeleton(overlay, opt_cam, intr_tuple, corrected_color, 3)  # optimized trajectory
            if opt_verts.ndim == 2 and opt_verts.shape[1] == 3:
                vc = world_to_camera(opt_verts[:: max(1, int(args.vertex_stride))], T)
                u, v, valid = project_camera(vc, intr_tuple, width, height)
                for x, y in zip(u[valid], v[valid]):
                    cv2.circle(overlay, (int(x), int(y)), 1, corrected_color, -1)
            world_chunks.append(joints_world)
            world_chunks.append(opt_world)
            if opt_verts.ndim == 2 and opt_verts.shape[1] == 3:
                world_chunks.append(opt_verts)
        cv2.putText(overlay, f"frame {frame_idx}: original left/right = blue/orange; corrected left/right = cyan/yellow", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5)
        cv2.putText(overlay, f"frame {frame_idx}: original left/right = blue/orange; corrected left/right = cyan/yellow", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imwrite(str(overlay_dir / f"{out_i:06d}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        world = np.zeros((720, 1280, 3), dtype=np.uint8)
        mn, mx = world_bounds(world_chunks, float(args.padding_m))
        if object_points is not None:
            for p in object_points:
                q = world_point(p, mn, mx, 1280, 720)
                if q is not None:
                    cv2.circle(world, q, 1, (40, 210, 60), -1)
        for hand in frame.get("hands", []):
            side = str(hand.get("hand_side"))
            st = states.get((frame_idx, side))
            if st is None:
                continue
            metric = hand.get("metric_mano_state") or {}
            joints_world = np.asarray(metric.get("joints_current_v18_world_m") or [], dtype=float)
            opt_world = np.asarray(st.get("optimized_joints_world_m") or [], dtype=float)
            opt_verts = np.asarray(st.get("optimized_vertices_world_sample_m") or [], dtype=float)
            original_color, corrected_color = colors_for_side(side)
            if joints_world.shape == (21, 3):
                draw_world_skeleton(world, joints_world, mn, mx, original_color, 3)
            if opt_world.shape == (21, 3):
                draw_world_skeleton(world, opt_world, mn, mx, corrected_color, 2)
            if opt_verts.ndim == 2 and opt_verts.shape[1] == 3:
                for p in opt_verts[:: max(1, int(args.vertex_stride))]:
                    q = world_point(p, mn, mx, 1280, 720)
                    if q is not None:
                        cv2.circle(world, q, 1, corrected_color, -1)
        cv2.putText(world, f"local metric world frame {frame_idx}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imwrite(str(world_dir / f"{out_i:06d}.jpg"), world, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rendered += 1
    overlay_video = case_dir / "v18_overlay_joint_mano_interval_correction.mp4"
    world_video = case_dir / "v18_world_joint_mano_interval_correction.mp4"
    side_video = case_dir / "v18_side_by_side_joint_mano_interval_correction.mp4"
    encode(overlay_dir, overlay_video, fps)
    encode(world_dir, world_video, fps)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(overlay_video), "-i", str(world_video),
        "-filter_complex", "[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[l];[1:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[r];[l][r]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(side_video)
    ], check=True)
    manifest = {"case": args.case, "frame_count": rendered, "frame_ids": frame_ids, "overlay_video": str(overlay_video), "world_video": str(world_video), "side_by_side_video": str(side_video)}
    (case_dir / "v18_joint_mano_interval_correction_render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
