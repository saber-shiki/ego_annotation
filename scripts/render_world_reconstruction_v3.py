#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compare_hand_streams_scale055_v3 import load_frame_window
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from render_mesh_alignment_v3 import project, view_basis
from render_mesh_surface_contact_review_v3 import (
    HAND_EDGES,
    draw_contact_patch,
    draw_hand,
    draw_mesh_projection,
    draw_object_mask,
    read_frame,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reliable_contact_rows(contact: dict) -> dict[int, dict]:
    rows = [row for row in contact.get("rows_detail", []) if bool(row.get("reliable_for_contact", False))]
    return {int(row["frame_idx"]): row for row in rows}


def world_joints(hand: dict) -> np.ndarray:
    arr = np.asarray(hand.get("joints3d_world_m", []), dtype=float)
    if arr.shape != (21, 3) or not np.isfinite(arr).all():
        raise RuntimeError("hand lacks finite 21x3 joints3d_world_m")
    return arr


def world_vertices(hand: dict) -> np.ndarray:
    arr = np.asarray(hand.get("vertices_world_m", []), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise RuntimeError("hand lacks finite vertices_world_m")
    return arr


def camera_frustum_points(t_world_camera: np.ndarray, scale: float) -> np.ndarray:
    width = 0.65 * scale
    height = 0.42 * scale
    points_camera = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-width, -height, scale],
            [width, -height, scale],
            [width, height, scale],
            [-width, height, scale],
        ],
        dtype=float,
    )
    homog = np.c_[points_camera, np.ones(len(points_camera), dtype=float)]
    return (t_world_camera @ homog.T).T[:, :3]


def draw_polyline_3d(
    image: np.ndarray,
    points: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    color: tuple[int, int, int],
    thickness: int,
    closed: bool = False,
) -> None:
    xy, _ = project(points, center, basis, radius, (image.shape[1], image.shape[0]))
    pts = xy.astype(np.int32)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(image, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
    if closed and len(pts) > 2:
        cv2.line(image, tuple(pts[-1]), tuple(pts[0]), color, thickness, cv2.LINE_AA)


def draw_mesh_world(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    max_faces: int,
) -> None:
    face_ids = np.arange(len(faces), dtype=int)
    if len(face_ids) > max_faces:
        face_ids = face_ids[np.linspace(0, len(face_ids) - 1, max_faces, dtype=int)]
    xy, depth = project(vertices, center, basis, radius, (image.shape[1], image.shape[0]))
    hull = cv2.convexHull(xy.astype(np.float32)).astype(np.int32)
    hull_overlay = image.copy()
    cv2.fillConvexPoly(hull_overlay, hull, (86, 96, 224), cv2.LINE_AA)
    cv2.addWeighted(hull_overlay, 0.26, image, 0.74, 0.0, image)
    cv2.polylines(image, [hull], True, (48, 54, 170), 2, cv2.LINE_AA)
    face_depth = depth[faces[face_ids]].mean(axis=1)
    order = face_ids[np.argsort(face_depth)]
    overlay = image.copy()
    for face_id in order:
        poly = xy[faces[int(face_id)]]
        if np.any(poly[:, 0] < -image.shape[1]) or np.any(poly[:, 0] > 2 * image.shape[1]):
            continue
        if np.any(poly[:, 1] < -image.shape[0]) or np.any(poly[:, 1] > 2 * image.shape[0]):
            continue
        cv2.fillConvexPoly(overlay, poly.astype(np.int32), (72, 82, 214), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.20, image, 0.80, 0.0, image)
    edge_ids = order[np.linspace(0, len(order) - 1, min(len(order), 380), dtype=int)]
    for face_id in edge_ids:
        poly = xy[faces[int(face_id)]].astype(np.int32)
        cv2.polylines(image, [poly], True, (58, 62, 150), 1, cv2.LINE_AA)


def draw_camera_path(
    image: np.ndarray,
    annotations: dict[int, dict],
    frame_idx: int,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
) -> None:
    frames = sorted(annotations)
    path = np.asarray([annotations[f]["camera"]["position_world_m"] for f in frames], dtype=float)
    xy, _ = project(path, center, basis, radius, (image.shape[1], image.shape[0]))
    cv2.polylines(image, [xy.astype(np.int32)], False, (90, 90, 90), 2, cv2.LINE_AA)
    cur_i = frames.index(int(frame_idx))
    if cur_i > 0:
        cv2.polylines(image, [xy[: cur_i + 1].astype(np.int32)], False, (20, 20, 20), 3, cv2.LINE_AA)


def draw_camera(
    image: np.ndarray,
    t_world_camera: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    scale: float,
) -> None:
    frustum = camera_frustum_points(t_world_camera, scale)
    origin = frustum[0]
    corners = frustum[1:]
    for corner in corners:
        draw_polyline_3d(image, np.vstack([origin, corner]), center, basis, radius, (20, 20, 20), 2)
    draw_polyline_3d(image, np.vstack([corners, corners[0]]), center, basis, radius, (20, 20, 20), 2, closed=False)
    xy, _ = project(frustum[:1], center, basis, radius, (image.shape[1], image.shape[0]))
    cv2.circle(image, tuple(xy[0].astype(int)), 5, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.putText(image, "head camera", tuple((xy[0] + np.asarray([8, -8])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)


def draw_hand_world(
    image: np.ndarray,
    hand: dict,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    contact_ids: list[int] | None,
) -> None:
    joints = world_joints(hand)
    measured = bool(hand.get("measurement_available", False))
    if not measured:
        return
    side = str(hand.get("side", "unknown"))
    color = (40, 180, 70) if side == "right" else (215, 130, 45)
    xy, _ = project(joints, center, basis, radius, (image.shape[1], image.shape[0]))
    for a, b in HAND_EDGES:
        cv2.line(image, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), color, 3, cv2.LINE_AA)
    for point in xy:
        cv2.circle(image, tuple(point.astype(int)), 3, color, -1, cv2.LINE_AA)
    if contact_ids:
        vertices = world_vertices(hand)[np.asarray(contact_ids, dtype=int)]
        uv, _ = project(vertices, center, basis, radius, (image.shape[1], image.shape[0]))
        for point in uv:
            cv2.circle(image, tuple(point.astype(int)), 7, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(image, tuple(point.astype(int)), 9, (10, 10, 10), 1, cv2.LINE_AA)


def draw_world_panel(
    annotations: dict[int, dict],
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    contact_by_frame: dict[int, dict],
    frame_idx: int,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    args: argparse.Namespace,
) -> np.ndarray:
    image = np.full((args.panel_height, args.panel_width, 3), (247, 248, 244), dtype=np.uint8)
    ann = annotations[int(frame_idx)]
    vertices, faces = meshes[int(frame_idx)]
    draw_mesh_world(image, vertices, faces, center, basis, radius, int(args.max_mesh_faces))
    draw_camera_path(image, annotations, int(frame_idx), center, basis, radius)
    draw_camera(image, np.asarray(ann["camera"]["T_world_camera_metric"], dtype=float), center, basis, radius, float(args.frustum_scale_m))
    row = contact_by_frame.get(int(frame_idx))
    for i, hand in enumerate(ann.get("hands", [])):
        ids = row.get("best_patch_vertex_ids", []) if row is not None and int(row["hand_idx"]) == i else None
        draw_hand_world(image, hand, center, basis, radius, ids)
    cv2.putText(image, f"metric world reconstruction  frame {frame_idx}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
    scale_px = int(round(0.10 * 0.42 * min(image.shape[1], image.shape[0]) / radius))
    scale_px = max(20, min(scale_px, 180))
    sx, sy = 28, args.panel_height - 64
    cv2.line(image, (sx, sy), (sx + scale_px, sy), (25, 25, 25), 4, cv2.LINE_AA)
    cv2.putText(image, "0.10 m", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(image, "red object mesh   green/orange measured MANO   black current head camera", (20, args.panel_height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (45, 45, 45), 1, cv2.LINE_AA)
    return image


def frame_view(points: list[np.ndarray], padding: float) -> tuple[np.ndarray, np.ndarray, float]:
    cloud = np.vstack(points)
    center, basis, radius = view_basis(cloud)
    return center, basis, max(radius * padding, 1e-4)


def current_focus_view(ann: dict, mesh: tuple[np.ndarray, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    points = []
    vertices = mesh[0]
    points.append(vertices[np.linspace(0, len(vertices) - 1, min(len(vertices), 300), dtype=int)])
    points.append(camera_frustum_points(np.asarray(ann["camera"]["T_world_camera_metric"], dtype=float), float(args.frustum_scale_m)))
    for hand in ann.get("hands", []):
        if bool(hand.get("measurement_available", False)):
            points.append(world_joints(hand))
    if len(points) == 1:
        for hand in ann.get("hands", []):
            points.append(world_joints(hand))
    return frame_view(points, float(args.focus_radius_scale))


def render_overlay_frame(
    cap: cv2.VideoCapture,
    ann: dict,
    mesh: tuple[np.ndarray, np.ndarray],
    row: dict | None,
    frame_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    image = read_frame(cap, int(frame_idx))
    draw_object_mask(image, ann, args)
    draw_mesh_projection(image, ann, mesh, int(args.max_overlay_mesh_edges))
    for hand in ann.get("hands", []):
        draw_hand(image, hand)
    if row is not None:
        draw_contact_patch(image, ann["hands"][int(row["hand_idx"])], row)
    label = f"frame {frame_idx}"
    if row is None:
        label += "  no reliable mesh-surface contact"
    else:
        label += (
            f"  {row['side']} hand mesh contact  reproj {row['median_joint_reprojection_px']:.1f}px"
            f"  surface p95 {row['best_patch_distance_p95_m'] * 1000.0:.1f}mm"
        )
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(image, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def combine_panels(overlay: np.ndarray, world: np.ndarray, caption: str, args: argparse.Namespace) -> np.ndarray:
    half = args.output_width // 2
    panel_h = args.panel_height
    left = cv2.resize(overlay, (half, panel_h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(world, (args.output_width - half, panel_h), interpolation=cv2.INTER_AREA)
    joined = np.hstack([left, right])
    bar = np.zeros((args.caption_height, args.output_width, 3), dtype=np.uint8)
    cv2.putText(bar, caption, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([joined, bar])


def run(args: argparse.Namespace) -> dict:
    annotations = load_frame_window(args.annotations, args.frame_start, args.frame_end)
    meshes = load_mesh_archive(args.object_mesh_npz)
    missing_mesh = sorted(set(annotations).difference(meshes))
    if missing_mesh:
        raise RuntimeError(f"mesh archive missing frames: {missing_mesh[:8]}")
    contact_by_frame = reliable_contact_rows(load_json(args.contact_report))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = float(args.output_fps) if args.output_fps is not None else float(cap.get(cv2.CAP_PROP_FPS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    video_path = args.output_dir / "world_reconstruction_side_by_side.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(args.output_width), int(args.panel_height + args.caption_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer {video_path}")
    written_stills = []
    frames = list(range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)))
    try:
        for frame_idx in frames:
            ann = annotations[int(frame_idx)]
            row = contact_by_frame.get(int(frame_idx))
            overlay = render_overlay_frame(cap, ann, meshes[int(frame_idx)], row, int(frame_idx), args)
            center, basis, radius = current_focus_view(ann, meshes[int(frame_idx)], args)
            world = draw_world_panel(annotations, meshes, contact_by_frame, int(frame_idx), center, basis, radius, args)
            caption = str(ann.get("caption", "")).strip()
            if not caption:
                raise RuntimeError(f"frame {frame_idx} has no semantic caption")
            frame = combine_panels(overlay, world, caption, args)
            writer.write(frame)
            if row is not None or int(frame_idx) in set(args.still_frames):
                path = still_dir / f"frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError(f"failed to write {path}")
                written_stills.append(str(path))
    finally:
        writer.release()
        cap.release()
    report = {
        "status": "ok",
        "method": "render_world_reconstruction_v3",
        "video": str(video_path),
        "stills_dir": str(still_dir),
        "written_stills": written_stills,
        "frames": frames,
        "fps": fps,
        "contact_frames": sorted(contact_by_frame),
        "world_view": "per-frame object-and-measured-hand focus in the stored metric world frame",
        "interpretation": "The right panel is an orthographic third-person rendering of the stored metric world frame. It does not assert gravity alignment.",
        "annotations": str(args.annotations),
        "object_mesh_npz": str(args.object_mesh_npz),
        "contact_report": str(args.contact_report),
    }
    (args.output_dir / "render_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--output-width", type=int, default=1920)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--panel-height", type=int, default=720)
    parser.add_argument("--caption-height", type=int, default=58)
    parser.add_argument("--focus-radius-scale", type=float, default=1.28)
    parser.add_argument("--frustum-scale-m", type=float, default=0.045)
    parser.add_argument("--max-mesh-faces", type=int, default=1200)
    parser.add_argument("--max-overlay-mesh-edges", type=int, default=260)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[858, 866, 867, 868, 879, 880])
    parser.add_argument("--remote-output-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/data"))
    parser.add_argument("--local-output-root", type=Path, default=Path("/data2/ego_annotation_outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
