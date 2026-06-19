#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Render the task5 compact-rigid tomato hypothesis with interval-level MANO uncertainty.

This script is a renderer, not a verifier. It consumes the completed compact-rigid
obj_tomato mesh, per-frame rigid pose report, and full-bridge MANO/object
constraint rows to produce full-video overlay/world/side-by-side artifacts. It
does not accept sparse H-prime rows as a delivered MANO trajectory; conflict
states are rendered as interval-level uncertainty evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

CASE = "task5_tomato_960"
DEFAULT_ANNOTATIONS_PATH = Path(
    "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/"
    "task5_tomato_960/annotations_v18_full.json"
)
DEFAULT_POSE_REPORT_PATH = Path(
    "/data2/ego_annotation_outputs/v18_compact_rigid_completion_frame806/task5_tomato_960/"
    "object_obj_tomato/pose_fit_seed42_v2/v18_compact_rigid_object_pose_fit_report.json"
)
DEFAULT_COMPLETED_MESH_PLY = Path(
    "/data2/ego_annotation_outputs/v18_compact_rigid_completion_frame806/task5_tomato_960/"
    "object_obj_tomato/completed_mesh_seed42_v2/"
    "object_obj_tomato_compact_rigid_completed_mesh_labeled.ply"
)
DEFAULT_CONSTRAINT_REPORT_PATH = Path(
    "/data2/ego_annotation_outputs/v18_full_bridge_all_signed_rebuild_v1/task5_tomato_960/"
    "object_obj_tomato/surface806_sign929_full_bridge_all_signed/initial_measure/"
    "v18_mano_object_constraint_state_full_bridge.json"
)
DEFAULT_OUTPUT_ROOT = Path("/data2/ego_annotation_outputs/v18_temporal_rigid_tomato_artifact_v2")

HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--pose-report", type=Path, default=DEFAULT_POSE_REPORT_PATH)
    parser.add_argument("--completed-mesh", type=Path, default=DEFAULT_COMPLETED_MESH_PLY)
    parser.add_argument("--constraint-report", type=Path, default=DEFAULT_CONSTRAINT_REPORT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mesh-projection-stride", type=int, default=15)
    parser.add_argument("--world-mesh-stride", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def load_mesh_vertices(path: Path) -> np.ndarray:
    mesh_geom = trimesh.load(path, process=False)
    if isinstance(mesh_geom, trimesh.Scene):
        sub_meshes = [m for m in mesh_geom.geometry.values() if isinstance(m, trimesh.Trimesh)]
        if not sub_meshes:
            raise RuntimeError(f"No trimesh geometry found in scene: {path}")
        mesh_geom = trimesh.util.concatenate(sub_meshes)
    if not isinstance(mesh_geom, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh geometry type from {path}: {type(mesh_geom)}")
    return np.asarray(mesh_geom.vertices, dtype=np.float64)


def pose_map(pose_data: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in pose_data.get("pose_rows", []):
        if row.get("status") != "fit_to_visible_depth_samples":
            continue
        out[int(row["frame_idx"])] = (
            np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64),
            np.asarray(row["translation_world_m"], dtype=np.float64),
        )
    return out


def constraint_map(constraint_data: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in constraint_data.get("constraint_rows", []):
        out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return out


def collect_world_bounds(
    frames: list[dict[str, Any]],
    posed_object_vertices: np.ndarray,
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    for frame_idx, frame in enumerate(frames):
        for hand in frame.get("hands", []):
            metric = hand.get("metric_mano_state") or {}
            joints = np.asarray(metric.get("joints_current_v18_world_m") or [], dtype=float)
            if joints.shape == (21, 3):
                chunks.append(joints)
        if frame_idx in poses:
            rot, trans = poses[frame_idx]
            chunks.append(posed_object_vertices[::50] @ rot.T + trans[None, :])
    if not chunks:
        raise RuntimeError("Cannot render world view: no hand joints or posed object vertices found")
    all_points = np.vstack(chunks)
    min_xyz = all_points.min(axis=0) - 0.10
    max_xyz = all_points.max(axis=0) + 0.10
    return min_xyz, max_xyz


def world_to_screen(
    point_world: np.ndarray,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    canvas_w: int,
    canvas_h: int,
) -> tuple[int, int] | None:
    extent = np.maximum(max_xyz - min_xyz, 1.0e-6)
    x = int(round((point_world[0] - min_xyz[0]) / extent[0] * canvas_w))
    y = int(round(canvas_h - (point_world[2] - min_xyz[2]) / extent[2] * canvas_h))
    if 0 <= x < canvas_w and 0 <= y < canvas_h:
        return (x, y)
    return None


def first_intrinsics(frame: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    for hand in frame.get("hands", []):
        metric = hand.get("metric_mano_state") or {}
        intr = metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
        if isinstance(intr, list) and len(intr) == 4:
            return tuple(float(x) for x in intr)  # type: ignore[return-value]
    return (1000.0, 1000.0, float(width) / 2.0, float(height) / 2.0)


def project_camera_points(
    points_camera: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    z = points_camera[:, 2]
    valid = z > 0.01
    u = np.zeros(points_camera.shape[0], dtype=np.int32)
    v = np.zeros(points_camera.shape[0], dtype=np.int32)
    if valid.any():
        u_float = fx * points_camera[:, 0] / np.maximum(z, 1.0e-6) + cx
        v_float = fy * points_camera[:, 1] / np.maximum(z, 1.0e-6) + cy
        scale_x = width / max(1.0, 2.0 * cx)
        scale_y = height / max(1.0, 2.0 * cy)
        u = (u_float * scale_x).astype(np.int32)
        v = (v_float * scale_y).astype(np.int32)
        valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def constraint_style(state: str) -> tuple[tuple[int, int, int], int, str]:
    if "not_applied" in state or "candidate" in state:
        return (0, 255, 255), 4, "CONSTRAINT CONFLICT"
    if "uncertainty" in state:
        return (0, 200, 255), 3, "UNCERTAIN"
    if state == "no_penetration_no_coordinate_change_needed":
        return (180, 180, 180), 2, "no conflict"
    return (150, 150, 150), 2, "not measured"


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(output_path),
        ],
        check=True,
    )


def constraint_intervals(constraints: dict[tuple[int, str], dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    frames_by_side: dict[str, list[int]] = {}
    for (frame_idx, side), row in constraints.items():
        if row.get("candidate_application_state") == "no_penetration_no_coordinate_change_needed":
            continue
        frames_by_side.setdefault(side, []).append(frame_idx)

    intervals: dict[str, list[tuple[int, int]]] = {}
    for side, frames in frames_by_side.items():
        sorted_frames = sorted(set(frames))
        if not sorted_frames:
            intervals[side] = []
            continue
        start = end = sorted_frames[0]
        side_intervals: list[tuple[int, int]] = []
        for frame_idx in sorted_frames[1:]:
            if frame_idx <= end + 1:
                end = frame_idx
                continue
            side_intervals.append((start, end))
            start = end = frame_idx
        side_intervals.append((start, end))
        intervals[side] = side_intervals
    return intervals


def render(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    annotations = load_json(args.annotations)
    pose_data = load_json(args.pose_report)
    constraint_data = load_json(args.constraint_report)
    object_vertices = load_mesh_vertices(args.completed_mesh)
    poses = pose_map(pose_data)
    constraints = constraint_map(constraint_data)
    frames = annotations.get("frames", [])
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"No frames found in {args.annotations}")

    output_case_dir = args.output_root / CASE
    overlay_dir = output_case_dir / "overlay_frames"
    world_dir = output_case_dir / "world_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    world_dir.mkdir(parents=True, exist_ok=True)

    min_xyz, max_xyz = collect_world_bounds(frames, object_vertices, poses)
    fps = float((annotations.get("raw_video") or {}).get("fps", 30.0))
    canvas_w, canvas_h = 1280, 720

    for frame_idx, frame in enumerate(frames):
        raw_path = Path(frame.get("raw_frame_path", ""))
        overlay = cv2.imread(str(raw_path)) if raw_path.exists() else None
        if overlay is None:
            overlay = np.zeros((1080, 1920, 3), dtype=np.uint8)
        height, width = overlay.shape[:2]
        camera = frame.get("camera") or {}
        T_world_camera = np.asarray(camera.get("T_world_camera_metric", np.eye(4)), dtype=np.float64)

        tomato_present = frame_idx in poses
        if tomato_present:
            rot, trans = poses[frame_idx]
            vertices_world = object_vertices @ rot.T + trans[None, :]
            sampled_world = vertices_world[:: max(1, int(args.mesh_projection_stride))]
            vertices_camera = (sampled_world - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]
            u, v, valid = project_camera_points(vertices_camera, first_intrinsics(frame, width, height), width, height)
            for x, y in zip(u[valid], v[valid]):
                cv2.circle(overlay, (int(x), int(y)), 1, (40, 255, 80), -1)
            tomato_label = f"rigid tomato  {object_vertices.shape[0]} verts"
            tomato_label_color = (40, 255, 80)
        else:
            tomato_label = f"frame {frame_idx}: rigid tomato pose missing"
            tomato_label_color = (0, 165, 255)
        cv2.putText(overlay, tomato_label, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.60, tomato_label_color, 2)

        for hand_idx, hand in enumerate(frame.get("hands", [])):
            metric = hand.get("metric_mano_state") or {}
            side = str(hand.get("hand_side"))
            row = constraints.get((frame_idx, side))
            state = str((row or {}).get("candidate_application_state", "not_measured"))
            color, line_width, state_label = constraint_style(state)
            penetrating = row.get("penetrating_vertex_count", "?") if row else "?"
            label_y = 80 + hand_idx * 120
            cv2.putText(
                overlay,
                f"{side} {state_label} | {state[:50]}",
                (12, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            cv2.putText(
                overlay,
                f"penetrating verts={penetrating}",
                (12, label_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
            )
            joints_camera = np.asarray(metric.get("joints_current_v18_camera_m") or [], dtype=float)
            intr = metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
            if joints_camera.shape == (21, 3) and isinstance(intr, list) and len(intr) == 4:
                u, v, valid = project_camera_points(joints_camera, tuple(float(x) for x in intr), width, height)  # type: ignore[arg-type]
                for a, b in HAND_EDGES:
                    if valid[a] and valid[b]:
                        cv2.line(overlay, (int(u[a]), int(v[a])), (int(u[b]), int(v[b])), color, line_width)

        cv2.imwrite(str(overlay_dir / f"{frame_idx:06d}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])

        world = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        if tomato_present:
            rot, trans = poses[frame_idx]
            vertices_world = object_vertices[:: max(1, int(args.world_mesh_stride))] @ rot.T + trans[None, :]
            for vertex in vertices_world:
                point = world_to_screen(vertex, min_xyz, max_xyz, canvas_w, canvas_h)
                if point is not None:
                    cv2.circle(world, point, 1, (40, 255, 80), -1)
        for hand in frame.get("hands", []):
            metric = hand.get("metric_mano_state") or {}
            side = str(hand.get("hand_side"))
            state = str((constraints.get((frame_idx, side)) or {}).get("candidate_application_state", "not_measured"))
            color, line_width, _ = constraint_style(state)
            joints_world = np.asarray(metric.get("joints_current_v18_world_m") or [], dtype=float)
            if joints_world.shape == (21, 3):
                for a, b in HAND_EDGES:
                    pa = world_to_screen(joints_world[a], min_xyz, max_xyz, canvas_w, canvas_h)
                    pb = world_to_screen(joints_world[b], min_xyz, max_xyz, canvas_w, canvas_h)
                    if pa is not None and pb is not None:
                        cv2.line(world, pa, pb, color, max(2, line_width - 1))
        cv2.putText(
            world,
            f"metric world  frame {frame_idx:04d}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.imwrite(str(world_dir / f"{frame_idx:06d}.jpg"), world, [cv2.IMWRITE_JPEG_QUALITY, 88])

        if frame_idx % 120 == 0:
            print(f"rendered frame {frame_idx}/{len(frames)}")

    overlay_video = output_case_dir / "v18_overlay_rigid_tomato.mp4"
    world_video = output_case_dir / "v18_world_rigid_tomato.mp4"
    side_by_side_video = output_case_dir / "v18_side_by_side_rigid_tomato.mp4"
    encode_video(overlay_dir, overlay_video, fps)
    encode_video(world_dir, world_video, fps)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(overlay_video),
            "-i",
            str(world_video),
            "-filter_complex",
            "[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
            "pad=960:540:(ow-iw)/2:(oh-ih)/2:black[l];"
            "[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
            "pad=960:540:(ow-iw)/2:(oh-ih)/2:black[r];[l][r]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(side_by_side_video),
        ],
        check=True,
    )

    intervals = constraint_intervals(constraints)
    manifest: dict[str, Any] = {
        "method": "render_v18_compact_rigid_tomato_temporal_mano_attempt",
        "status": "ok",
        "case": CASE,
        "output_root": str(args.output_root),
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completed_mesh": str(args.completed_mesh),
            "constraint_report": str(args.constraint_report),
        },
        "outputs": {
            "overlay": str(overlay_video),
            "world": str(world_video),
            "side_by_side": str(side_by_side_video),
            "manifest": str(output_case_dir / "v18_temporal_rigid_tomato_manifest.json"),
        },
        "rendered_state": {
            "tomato_object": "compact-rigid completed mesh with per-frame visible-depth pose fit",
            "hand_model": "current V18 metric MANO skeletons with full-bridge compact-rigid tomato constraint state",
            "legacy_deformable_tomato_state_consumed": False,
            "coordinate_level_mano_correction_accepted": False,
        },
        "evidence": {
            "total_frames": len(frames),
            "frames_with_tomato_pose": len([idx for idx in range(len(frames)) if idx in poses]),
            "constraint_rows": len(constraints),
            "constraint_conflict_intervals": {side: ranges for side, ranges in intervals.items()},
            "conflict_interval_count": {side: len(ranges) for side, ranges in intervals.items()},
        },
        "physical_conclusion": (
            "The compact-rigid tomato mesh and per-frame pose are rendered directly as the active object state. "
            "The MANO/object constraint state is continuous over interaction intervals but does not yield an "
            "accepted coordinate-level MANO trajectory here; unresolved rows are exposed as interval-level hand "
            "state uncertainty rather than sparse accepted-row progress."
        ),
        "remaining_gap": (
            "A temporal MANO trajectory or bounded uncertainty sequence over the conflict intervals is still required. "
            "This renderer preserves the object/hand dataflow needed for that next mechanism but does not solve it."
        ),
        "visual_inspection_required": True,
        "claim_scope": "diagnostic integrated rigid-tomato/temporal-MANO-uncertainty artifact, not final V18 delivery",
        "total_elapsed_s": time.time() - started,
    }
    manifest_path = output_case_dir / "v18_temporal_rigid_tomato_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = render(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
