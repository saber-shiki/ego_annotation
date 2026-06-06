#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from compare_hand_streams_scale055_v3 import load_frame_window
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from render_mesh_alignment_v3 import project, view_basis
from render_mesh_surface_contact_review_v3 import (
    FrameSource,
    HAND_EDGES,
    draw_contact_patch,
    draw_hand,
    draw_mesh_projection,
    draw_object_mask,
)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise RuntimeError("cannot normalize degenerate vector")
    return np.asarray(vector, dtype=float) / norm


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reliable_contact_rows(contact: dict) -> dict[int, dict]:
    rows = [
        row
        for row in contact.get("rows_detail", [])
        if bool(row.get("reliable_for_contact", False))
        or bool(row.get("geometry_backed_temporal_contact", False))
    ]
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


def simplify_mesh_for_display(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    if max_faces <= 0 or len(faces) <= max_faces:
        return vertices, faces
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(max_faces))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    out_vertices = np.asarray(mesh.vertices, dtype=float)
    out_faces = np.asarray(mesh.triangles, dtype=np.int32)
    if out_vertices.ndim != 2 or out_vertices.shape[1] != 3 or out_faces.ndim != 2 or out_faces.shape[1] != 3:
        raise RuntimeError("display mesh simplification produced invalid geometry")
    if len(out_faces) == 0:
        raise RuntimeError("display mesh simplification produced no faces")
    return out_vertices, out_faces


def draw_mesh_world(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    max_faces: int,
) -> None:
    vertices, faces = simplify_mesh_for_display(vertices, faces, max_faces)
    face_ids = np.arange(len(faces), dtype=int)
    xy, depth = project(vertices, center, basis, radius, (image.shape[1], image.shape[0]))
    hull = cv2.convexHull(xy.astype(np.float32)).astype(np.int32)
    shadow = hull + np.asarray([12, 14], dtype=np.int32)[None, None, :]
    shadow_overlay = image.copy()
    cv2.fillConvexPoly(shadow_overlay, shadow, (214, 216, 210), cv2.LINE_AA)
    cv2.addWeighted(shadow_overlay, 0.46, image, 0.54, 0.0, image)
    face_depth = depth[faces[face_ids]].mean(axis=1)
    order = face_ids[np.argsort(face_depth)]
    overlay = image.copy()
    face_vertices = vertices[faces[order]]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(normal_norm[:, None], 1e-12)
    light = unit(-0.70 * basis[2] - 0.45 * basis[1] + 0.22 * basis[0])
    shade = 0.47 + 0.45 * np.clip(np.abs(normals @ light), 0.0, 1.0)
    base = np.asarray([86.0, 102.0, 224.0], dtype=float)
    for rank, face_id in enumerate(order):
        poly = xy[faces[int(face_id)]]
        if np.any(poly[:, 0] < -image.shape[1]) or np.any(poly[:, 0] > 2 * image.shape[1]):
            continue
        if np.any(poly[:, 1] < -image.shape[0]) or np.any(poly[:, 1] > 2 * image.shape[0]):
            continue
        color = tuple(np.clip(base * shade[rank], 0, 255).astype(np.uint8).tolist())
        cv2.fillConvexPoly(overlay, poly.astype(np.int32), color, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0.0, image)
    cv2.polylines(image, [hull], True, (44, 50, 165), 2, cv2.LINE_AA)
    edge_ids = order[np.linspace(0, len(order) - 1, min(len(order), 320), dtype=int)]
    for face_id in edge_ids:
        poly = xy[faces[int(face_id)]].astype(np.int32)
        cv2.polylines(image, [poly], True, (50, 56, 142), 1, cv2.LINE_AA)


def draw_metric_axes(image: np.ndarray, center: np.ndarray, basis: np.ndarray, radius: float) -> None:
    origin = center - 0.66 * radius * basis[0] - 0.62 * radius * basis[1]
    scale = max(0.045, 0.20 * radius)
    axes = [
        ("X", np.asarray([1.0, 0.0, 0.0]), (40, 40, 210)),
        ("Y", np.asarray([0.0, 1.0, 0.0]), (40, 150, 60)),
        ("Z", np.asarray([0.0, 0.0, 1.0]), (210, 95, 35)),
    ]
    xy0, _ = project(origin[None, :], center, basis, radius, (image.shape[1], image.shape[0]))
    p0 = tuple(xy0[0].astype(int))
    for label, direction, color in axes:
        xy1, _ = project((origin + scale * direction)[None, :], center, basis, radius, (image.shape[1], image.shape[0]))
        p1 = tuple(xy1[0].astype(int))
        cv2.arrowedLine(image, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(image, label, (p1[0] + 4, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


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
    label: bool = True,
) -> None:
    frustum = camera_frustum_points(t_world_camera, scale)
    origin = frustum[0]
    corners = frustum[1:]
    for corner in corners:
        draw_polyline_3d(image, np.vstack([origin, corner]), center, basis, radius, (20, 20, 20), 2)
    draw_polyline_3d(image, np.vstack([corners, corners[0]]), center, basis, radius, (20, 20, 20), 2, closed=False)
    xy, _ = project(frustum[:1], center, basis, radius, (image.shape[1], image.shape[0]))
    cv2.circle(image, tuple(xy[0].astype(int)), 5, (20, 20, 20), -1, cv2.LINE_AA)
    if label:
        cv2.putText(image, "head camera", tuple((xy[0] + np.asarray([8, -8])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)


def draw_camera_inset(
    image: np.ndarray,
    annotations: dict[int, dict],
    frame_idx: int,
    args: argparse.Namespace,
) -> None:
    frames = sorted(annotations)
    path = np.asarray([annotations[f]["camera"]["position_world_m"] for f in frames], dtype=float)
    current = np.asarray(annotations[int(frame_idx)]["camera"]["T_world_camera_metric"], dtype=float)
    frustum = camera_frustum_points(current, float(args.frustum_scale_m) * 2.8)
    center, basis, radius = frame_view([path, frustum], 1.8)
    h, w = 180, 250
    x0 = image.shape[1] - w - 22
    y0 = image.shape[0] - h - 58
    inset = image[y0 : y0 + h, x0 : x0 + w].copy()
    panel = np.full_like(inset, (238, 240, 236))
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (80, 80, 80), 1, cv2.LINE_AA)
    draw_camera_path(panel, annotations, int(frame_idx), center, basis, radius)
    draw_camera(panel, current, center, basis, radius, float(args.frustum_scale_m) * 2.8, label=False)
    cv2.putText(panel, "head path", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (25, 25, 25), 1, cv2.LINE_AA)
    image[y0 : y0 + h, x0 : x0 + w] = panel


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
            p = tuple(point.astype(int))
            cv2.circle(image, p, 13, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, p, 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, p, 7, (255, 0, 255), -1, cv2.LINE_AA)


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
    draw_camera_inset(image, annotations, int(frame_idx), args)
    draw_metric_axes(image, center, basis, radius)
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
    cv2.putText(image, "red shaded object mesh   green/orange MANO   black head camera/path", (20, args.panel_height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (45, 45, 45), 1, cv2.LINE_AA)
    return image


def frame_view(points: list[np.ndarray], padding: float) -> tuple[np.ndarray, np.ndarray, float]:
    cloud = np.vstack(points)
    center, basis, radius = view_basis(cloud)
    return center, basis, max(radius * padding, 1e-4)


def oblique_frame_view(points: list[np.ndarray], padding: float) -> tuple[np.ndarray, np.ndarray, float]:
    cloud = np.vstack(points)
    center = np.median(cloud, axis=0)
    centered = cloud - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if np.linalg.det(vh) < 0:
        vh[-1] *= -1.0
    screen_x = unit(vh[0])
    plane_normal = unit(vh[2])
    in_plane = unit(vh[1])
    view_dir = unit(0.58 * plane_normal + 0.82 * in_plane)
    screen_y = unit(np.cross(view_dir, screen_x))
    basis = np.vstack([screen_x, screen_y, view_dir])
    if np.linalg.det(basis) < 0:
        basis[1] *= -1.0
    q = centered @ basis.T
    radius = float(np.max(np.linalg.norm(q[:, :2], axis=1)))
    return center, basis, max(radius * padding, 1e-4)


def current_focus_view(ann: dict, mesh: tuple[np.ndarray, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    points = []
    vertices = mesh[0]
    points.append(vertices[np.linspace(0, len(vertices) - 1, min(len(vertices), 900), dtype=int)])
    points.append(camera_frustum_points(np.asarray(ann["camera"]["T_world_camera_metric"], dtype=float), float(args.frustum_scale_m)))
    for hand in ann.get("hands", []):
        if bool(hand.get("measurement_available", False)):
            points.append(world_joints(hand))
    if len(points) == 1:
        for hand in ann.get("hands", []):
            points.append(world_joints(hand))
    return oblique_frame_view(points, float(args.focus_radius_scale))


def render_overlay_frame(
    frame_source: FrameSource,
    ann: dict,
    mesh: tuple[np.ndarray, np.ndarray],
    row: dict | None,
    frame_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    image = frame_source.read(int(frame_idx))
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
    prefix = str(getattr(args, "caption_prefix", "") or "").strip()
    text = f"{prefix}: {caption}" if prefix else caption
    cv2.putText(bar, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([joined, bar])


def run(args: argparse.Namespace) -> dict:
    annotations = load_frame_window(args.annotations, args.frame_start, args.frame_end)
    meshes = load_mesh_archive(args.object_mesh_npz)
    missing_mesh = sorted(set(annotations).difference(meshes))
    if missing_mesh:
        raise RuntimeError(f"mesh archive missing frames: {missing_mesh[:8]}")
    contact_by_frame = reliable_contact_rows(load_json(args.contact_report))
    frame_source = FrameSource(args.video, args.manifest)
    fps = float(args.output_fps) if args.output_fps is not None else frame_source.fps()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    world_still_dir = args.output_dir / "stills_world_3d"
    still_dir.mkdir(exist_ok=True)
    world_still_dir.mkdir(exist_ok=True)
    video_path = args.output_dir / "world_reconstruction_side_by_side.mp4"
    world_video_path = args.output_dir / "world_reconstruction_3d.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(args.output_width), int(args.panel_height + args.caption_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer {video_path}")
    world_writer = cv2.VideoWriter(
        str(world_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(args.panel_width), int(args.panel_height)),
    )
    if not world_writer.isOpened():
        writer.release()
        raise RuntimeError(f"failed to open writer {world_video_path}")
    written_stills = []
    written_world_stills = []
    frames = list(range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)))
    try:
        for frame_idx in frames:
            ann = annotations[int(frame_idx)]
            row = contact_by_frame.get(int(frame_idx))
            overlay = render_overlay_frame(frame_source, ann, meshes[int(frame_idx)], row, int(frame_idx), args)
            center, basis, radius = current_focus_view(ann, meshes[int(frame_idx)], args)
            world = draw_world_panel(annotations, meshes, contact_by_frame, int(frame_idx), center, basis, radius, args)
            world_writer.write(world)
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
                world_path = world_still_dir / f"frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(world_path), world, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError(f"failed to write {world_path}")
                written_world_stills.append(str(world_path))
    finally:
        writer.release()
        world_writer.release()
        frame_source.close()
    report = {
        "status": "ok",
        "method": "render_world_reconstruction_v3",
        "video": str(video_path),
        "world_video": str(world_video_path),
        "stills_dir": str(still_dir),
        "world_stills_dir": str(world_still_dir),
        "written_stills": written_stills,
        "written_world_stills": written_world_stills,
        "frames": frames,
        "fps": fps,
        "contact_frames": sorted(contact_by_frame),
        "world_view": "per-frame oblique object-and-measured-hand focus in the stored metric world frame",
        "interpretation": "The right panel is an orthographic third-person rendering over the metric SLAM frame; screen vertical follows the selected virtual view and the axis triad shows the stored metric axes.",
        "annotations": str(args.annotations),
        "object_mesh_npz": str(args.object_mesh_npz),
        "contact_report": str(args.contact_report),
        "video_source": str(args.video) if args.video is not None else None,
        "manifest_source": str(args.manifest) if args.manifest is not None else None,
    }
    (args.output_dir / "render_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path)
    parser.add_argument("--manifest", type=Path)
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
    parser.add_argument("--caption-prefix", default="")
    parser.add_argument("--still-frames", type=int, nargs="*", default=[858, 866, 867, 868, 879, 880])
    parser.add_argument("--remote-output-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/data"))
    parser.add_argument("--local-output-root", type=Path, default=Path("/data2/ego_annotation_outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
