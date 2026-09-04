#!/usr/bin/env python3
"""Build a focused V20 world-coordinate SAM3D/MANO/P09/camera visualization.

The recording intentionally contains only four world layers:
  * the SAM3D generated mesh (render-only completion hypothesis),
  * prediction-side MANO hand meshes,
  * prediction-side P09 visible-surface points, and
  * the calibrated camera and its original RGB image.

The selected V20 pose report is used only as a visualization transform.  No
HOT3D GT, observed object mesh, formal pose, or generated geometry is used as an
optimization authority.  A side-by-side H.264 video is emitted as an easy
comparison against the original source video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh


LEFT_COLOR_BGR = (255, 150, 40)
RIGHT_COLOR_BGR = (80, 150, 255)
GENERATED_COLOR_BGR = (215, 45, 190)
VISIBLE_POINT_COLOR_BGR = (0, 220, 255)
MASK_CONTOUR_BGR = (60, 255, 80)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_mesh(path: Path) -> trimesh.Trimesh:
    value = trimesh.load(path.expanduser().resolve(), force="mesh", process=False)
    if isinstance(value, trimesh.Scene):
        parts = [g for g in value.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"no mesh in {path}")
        value = trimesh.util.concatenate(parts)
    if not isinstance(value, trimesh.Trimesh) or len(value.vertices) == 0 or len(value.faces) == 0:
        raise RuntimeError(f"invalid mesh in {path}")
    vertices = np.asarray(value.vertices, dtype=np.float32)
    faces = np.asarray(value.faces, dtype=np.int32)
    if not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"non-finite or malformed mesh in {path}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def pose_rows(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in load_json(path).get("pose_rows", []):
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        if row.get("rotation_world_from_completed_canonical_matrix") is None:
            continue
        if row.get("translation_world_m") is None:
            continue
        result[int(row["frame_idx"])] = row
    return result


def resize_intrinsics_half_pixel(K: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    sx = float(target_wh[0]) / float(source_wh[0])
    sy = float(target_wh[1]) / float(source_wh[1])
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = sx * (out[0, 2] + 0.5) - 0.5
    out[1, 2] = sy * (out[1, 2] + 0.5) - 0.5
    return out


def camera_points_from_world(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points_world, dtype=np.float64) - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]


def project_points(points_camera: np.ndarray, K: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & np.isfinite(z) & (z > 0.01)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        uv[valid, 0] = K[0, 0] * points[valid, 0] / z[valid] + K[0, 2]
        uv[valid, 1] = K[1, 1] * points[valid, 1] / z[valid] + K[1, 2]
        valid &= np.isfinite(uv).all(axis=1)
        valid &= uv[:, 0] >= -width
        valid &= uv[:, 0] < 2.0 * width
        valid &= uv[:, 1] >= -height
        valid &= uv[:, 1] < 2.0 * height
    return uv, valid


def overlay_mesh(
    image: np.ndarray,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    T_world_camera: np.ndarray,
    K: np.ndarray,
    face_ids: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    height, width = image.shape[:2]
    camera = camera_points_from_world(vertices_world, T_world_camera)
    uv, valid = project_points(camera, K, width, height)
    candidate_faces = faces[np.all(valid[faces], axis=1)]
    if len(candidate_faces) == 0:
        return
    # Map the deterministic face sample to valid faces without changing the
    # underlying RRD mesh.  This is only for the video projection overlay.
    if len(candidate_faces) > len(face_ids):
        selected = np.linspace(0, len(candidate_faces) - 1, len(face_ids), dtype=np.int64)
        candidate_faces = candidate_faces[selected]
    order = np.argsort(np.mean(camera[candidate_faces, 2], axis=1))[::-1]
    polygons: list[np.ndarray] = []
    for tri in candidate_faces[order]:
        poly = np.rint(uv[tri]).astype(np.int32)
        if len(np.unique(poly, axis=0)) < 3:
            continue
        if np.any(poly[:, 0] < -width) or np.any(poly[:, 0] > 2 * width):
            continue
        if np.any(poly[:, 1] < -height) or np.any(poly[:, 1] > 2 * height):
            continue
        if abs(float(cv2.contourArea(poly.astype(np.float32)))) < 0.25:
            continue
        polygons.append(poly)
    if not polygons:
        return
    layer = image.copy()
    cv2.fillPoly(layer, polygons, color_bgr)
    object_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(object_mask, polygons, 255)
    mask = object_mask > 0
    image[mask] = cv2.addWeighted(layer, float(alpha), image, 1.0 - float(alpha), 0)[mask]
    # A sparse wireframe makes the generated hypothesis recognizable without
    # spending the full 476k-face mesh on every video frame.
    for poly in polygons[:: max(1, len(polygons) // 1000)]:
        cv2.polylines(image, [poly], True, tuple(int(v * 0.65) for v in color_bgr), 1, cv2.LINE_AA)
    valid_uv = uv[valid]
    if len(valid_uv) >= 3:
        hull = cv2.convexHull(np.rint(valid_uv).astype(np.int32))
        cv2.polylines(image, [hull], True, color_bgr, 2, cv2.LINE_AA)


def overlay_hand(
    image: np.ndarray,
    vertices_world: np.ndarray,
    joints_world: np.ndarray | None,
    T_world_camera: np.ndarray,
    K: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    camera = camera_points_from_world(vertices_world, T_world_camera)
    uv, valid = project_points(camera, K, width, height)
    valid_uv = uv[valid]
    if len(valid_uv) >= 3:
        hull = cv2.convexHull(np.rint(valid_uv).astype(np.int32))
        layer = image.copy()
        cv2.fillConvexPoly(layer, hull, color_bgr)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        keep = mask > 0
        image[keep] = cv2.addWeighted(layer, 0.16, image, 0.84, 0)[keep]
        cv2.polylines(image, [hull], True, color_bgr, 2, cv2.LINE_AA)
    if joints_world is None:
        return
    joints_camera = camera_points_from_world(joints_world, T_world_camera)
    joints_uv, joints_valid = project_points(joints_camera, K, width, height)
    for a, b in HAND_EDGES:
        if joints_valid[a] and joints_valid[b]:
            p0 = tuple(np.rint(joints_uv[a]).astype(int))
            p1 = tuple(np.rint(joints_uv[b]).astype(int))
            cv2.line(image, p0, p1, color_bgr, 2, cv2.LINE_AA)
    for i, point in enumerate(joints_uv):
        if joints_valid[i]:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 4 if i == 0 else 2, color_bgr, -1, cv2.LINE_AA)


def overlay_visible_points(
    image: np.ndarray,
    points_world: np.ndarray,
    T_world_camera: np.ndarray,
    K: np.ndarray,
    stride: int,
) -> None:
    if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) == 0:
        return
    camera = camera_points_from_world(points_world[::stride], T_world_camera)
    uv, valid = project_points(camera, K, image.shape[1], image.shape[0])
    for point in np.rint(uv[valid]).astype(np.int32):
        cv2.circle(image, tuple(point), 1, VISIBLE_POINT_COLOR_BGR, -1, cv2.LINE_AA)


def add_banner(image: np.ndarray, lines: list[str]) -> None:
    height = 66
    layer = image.copy()
    cv2.rectangle(layer, (0, 0), (image.shape[1], height), (8, 8, 8), -1)
    image[:height] = cv2.addWeighted(layer[:height], 0.82, image[:height], 0.18, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (12, 24 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 if index == 0 else 0.40,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def draw_mask_contour(image: np.ndarray, mask_path: str | None) -> None:
    if not mask_path:
        return
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, MASK_CONTOUR_BGR, 2, cv2.LINE_AA)


def build_overlay(
    original_bgr: np.ndarray,
    frame_idx: int,
    frame: dict[str, Any],
    rotation: np.ndarray,
    translation: np.ndarray,
    generated_vertices: np.ndarray,
    generated_faces: np.ndarray,
    generated_face_ids: np.ndarray,
    hands: dict[str, tuple[np.ndarray, np.ndarray | None]],
    visible_points: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    overlay_point_stride: int,
) -> np.ndarray:
    image = original_bgr.copy()
    generated_world = generated_vertices.astype(np.float64) @ rotation.T + translation[None, :]
    overlay_mesh(image, generated_world, generated_faces, T_world_camera, K, generated_face_ids, GENERATED_COLOR_BGR, 0.25)
    for side, color in (("left", LEFT_COLOR_BGR), ("right", RIGHT_COLOR_BGR)):
        if side in hands:
            vertices, joints = hands[side]
            overlay_hand(image, vertices, joints, T_world_camera, K, color)
    overlay_visible_points(image, visible_points, T_world_camera, K, overlay_point_stride)
    obj = next((candidate for candidate in frame.get("objects", []) if candidate.get("object_id") == "carton_milk"), None)
    draw_mask_contour(image, obj.get("mask_path") if isinstance(obj, dict) else None)
    add_banner(
        image,
        [
            "V20 K5 global orientation + local refine | SAM3D=magenta render-only | MANO=cyan/orange | P09=yellow",
            f"frame={frame_idx:03d} | green contour=object-owned observed mask | camera/world contract preserved",
        ],
    )
    return image


def encode_video(frame_dir: Path, output: Path, fps: float, pattern: str = "%06d.jpg", start_number: int = 0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-threads", "1", "-framerate", str(fps), "-start_number", str(int(start_number)), "-i", str(frame_dir / pattern),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--generated-mesh", type=Path, required=True)
    parser.add_argument("--mano-bridge", type=Path, required=True)
    parser.add_argument("--mano-topology", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--object-id", default="carton_milk")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=149)
    parser.add_argument("--output-rrd", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--overlay-face-budget", type=int, default=30000)
    parser.add_argument("--overlay-point-stride", type=int, default=8)
    parser.add_argument("--rrd-point-stride", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=84)
    parser.add_argument("--label", default="v20_k5_sam3d_mano_surface_camera")
    args = parser.parse_args()
    if args.frame_start < 0 or args.frame_end < args.frame_start:
        raise RuntimeError("invalid frame range")
    if args.overlay_face_budget < 1 or args.overlay_point_stride < 1 or args.rrd_point_stride < 1:
        raise RuntimeError("strides and face budget must be positive")
    for path in (args.annotations, args.pose_report, args.generated_mesh, args.mano_bridge, args.mano_topology, args.source_video):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)

    annotations = load_json(args.annotations)
    frames = {
        int(frame["frame_idx"]): frame
        for frame in annotations.get("frames", [])
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }
    selected = list(range(int(args.frame_start), int(args.frame_end) + 1))
    missing_frames = [idx for idx in selected if idx not in frames]
    if missing_frames:
        raise RuntimeError(f"annotation frames missing: {missing_frames[:10]}")
    poses = pose_rows(args.pose_report)
    missing_poses = [idx for idx in selected if idx not in poses]
    if missing_poses:
        raise RuntimeError(f"pose report lacks selected frames: {missing_poses[:10]}")

    generated = load_mesh(args.generated_mesh)
    generated_vertices = np.asarray(generated.vertices, dtype=np.float32)
    generated_faces = np.asarray(generated.faces, dtype=np.int32)
    generated_face_ids = np.unique(np.linspace(0, len(generated_faces) - 1, min(int(args.overlay_face_budget), len(generated_faces)), dtype=np.int64))

    with np.load(args.mano_bridge.expanduser().resolve(), allow_pickle=False) as bridge:
        required = {"frame_idx", "hand_side", "vertices_current_v18_world_from_hawor_projection_relift_m"}
        missing = sorted(required.difference(bridge.files))
        if missing:
            raise RuntimeError(f"MANO bridge missing keys: {missing}")
        bridge_frame = np.asarray(bridge["frame_idx"], dtype=np.int64)
        bridge_side = np.asarray(bridge["hand_side"]).astype(str)
        bridge_vertices = np.asarray(bridge["vertices_current_v18_world_from_hawor_projection_relift_m"], dtype=np.float32)
        bridge_joints = np.asarray(bridge["joints_current_v18_world_from_hawor_projection_relift_m"], dtype=np.float32) if "joints_current_v18_world_from_hawor_projection_relift_m" in bridge.files else None
    with np.load(args.mano_topology.expanduser().resolve(), allow_pickle=False) as topology:
        left_faces = np.asarray(topology["left_faces"], dtype=np.int32)
        right_faces = np.asarray(topology["right_faces"], dtype=np.int32)
    if left_faces.shape != (1538, 3) or right_faces.shape != (1538, 3):
        raise RuntimeError("unexpected MANO topology shape")
    hands_by_frame: dict[int, dict[str, tuple[np.ndarray, np.ndarray | None]]] = {}
    for i, (idx, side, vertices) in enumerate(zip(bridge_frame.tolist(), bridge_side.tolist(), bridge_vertices)):
        joints = bridge_joints[i] if bridge_joints is not None else None
        hands_by_frame.setdefault(int(idx), {})[str(side)] = (vertices, joints)

    video = cv2.VideoCapture(str(args.source_video.expanduser().resolve()))
    if not video.isOpened():
        raise RuntimeError(f"failed to open source video: {args.source_video}")
    video_count = int(round(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    video_fps = float(video.get(cv2.CAP_PROP_FPS) or 30.0)
    video_width = int(round(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    video_height = int(round(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if video_count and video_count < max(selected) + 1:
        raise RuntimeError(f"source video has only {video_count} frames")
    if (video_width, video_height) != (1408, 1408):
        raise RuntimeError(f"unexpected source video size {(video_width, video_height)}; expected (1408,1408)")

    args.output_rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"milk_{args.label}", spawn=False)
    rr.save(str(args.output_rrd.expanduser().resolve()))
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "/world/sam3d/mesh",
        rr.Mesh3D(
            vertex_positions=generated_vertices,
            triangle_indices=generated_faces,
            albedo_factor=[215, 45, 190, 150],
        ),
        static=True,
    )
    rr.log(
        "/metadata/description",
        rr.TextDocument(
            "# V20 focused world visualization\n\n"
            "Layers: SAM3D generated completion hypothesis, prediction-side MANO, "
            "P09 visible surface points, and calibrated camera/original RGB.\n\n"
            "The SAM3D mesh is render-only; it is not pose/collision/contact/SDF authority.\n"
            "This recording is diagnostic-only and does not modify formal state."
        ),
        static=True,
    )

    camera_positions: list[np.ndarray] = []
    metric_frames: list[int] = []
    no_metric_frames: list[int] = []
    overlay_frame_dir = Path(tempfile.mkdtemp(prefix="v20_interaction_overlay_", dir=str(args.output_rrd.parent)))
    try:
        if int(args.frame_start) > 0:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame_start))
        for frame_idx in selected:
            ok, frame_bgr = video.read()
            frame = frames[frame_idx]
            pose = poses[frame_idx]
            rotation = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
            translation = np.asarray(pose["translation_world_m"], dtype=np.float64)
            T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
            if rotation.shape != (3, 3) or translation.shape != (3,) or T_world_camera.shape != (4, 4):
                raise RuntimeError(f"invalid transform at frame {frame_idx}")
            K_values = np.asarray(frame["camera"].get("intrinsics_fx_fy_cx_cy") or [], dtype=np.float64)
            if K_values.shape != (4,):
                raise RuntimeError(f"missing camera intrinsics at frame {frame_idx}")
            K = np.asarray([[K_values[0], 0.0, K_values[2]], [0.0, K_values[1], K_values[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
            rgb960_bgr = cv2.resize(frame_bgr, (960, 960), interpolation=cv2.INTER_AREA)
            K960 = resize_intrinsics_half_pixel(K, (1408, 1408), (960, 960))

            rr.set_time("frame", sequence=frame_idx)
            camera_position = T_world_camera[:3, 3].copy()
            camera_positions.append(camera_position)
            rr.log("/world/camera", rr.Transform3D(translation=camera_position.tolist(), mat3x3=T_world_camera[:3, :3].tolist()))
            rr.log("/world/camera/image", rr.Pinhole(image_from_camera=K960.tolist(), resolution=[960, 960]))
            rr.log("/world/camera/image", rr.Image(cv2.cvtColor(rgb960_bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=int(args.jpeg_quality)))
            rr.log("/world/camera/trajectory", rr.LineStrips3D([np.asarray(camera_positions, dtype=np.float32)], radii=0.0015, colors=[[180, 180, 180, 220]]))
            rr.log("/world/sam3d", rr.Transform3D(translation=translation.tolist(), mat3x3=rotation.tolist()))

            hands = hands_by_frame.get(frame_idx, {})
            for side, faces, color in (("left", left_faces, [255, 150, 40, 255]), ("right", right_faces, [80, 150, 255, 255])):
                hand = hands.get(side)
                if hand is None:
                    rr.log(f"/world/hands/{side}", rr.Clear(recursive=False))
                    continue
                vertices, joints = hand
                rr.log(f"/world/hands/{side}", rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, albedo_factor=color))
                if joints is not None:
                    rr.log(f"/world/hands/{side}/joints", rr.Points3D(joints, radii=0.0035, colors=color))

            obj = next((candidate for candidate in frame.get("objects", []) if candidate.get("object_id") == args.object_id), None)
            geom = obj.get("visible_geometry_candidate") if isinstance(obj, dict) and isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            visible_points = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float32)
            if visible_points.ndim == 2 and visible_points.shape[1] == 3 and len(visible_points) and np.isfinite(visible_points).all():
                metric_frames.append(frame_idx)
                rr.log("/world/visible_surface", rr.Points3D(visible_points[:: int(args.rrd_point_stride)], radii=0.0018, colors=[255, 220, 0, 255], point_shading=rr.components.PointShading.Flat))
            else:
                no_metric_frames.append(frame_idx)
                rr.log("/world/visible_surface", rr.Clear(recursive=False))
                rr.log("/world/visible_surface/status", rr.TextLog("no accepted P09 metric surface"))

            overlay_bgr = build_overlay(
                rgb960_bgr,
                frame_idx,
                frame,
                rotation,
                translation,
                generated_vertices,
                generated_faces,
                generated_face_ids,
                hands,
                visible_points,
                K960,
                T_world_camera,
                int(args.overlay_point_stride),
            )
            side_by_side = np.hstack([rgb960_bgr, overlay_bgr])
            cv2.imwrite(str(overlay_frame_dir / f"{frame_idx:06d}.jpg"), side_by_side, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])
            rr.log("/comparison/original_video", rr.Image(cv2.cvtColor(rgb960_bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=int(args.jpeg_quality)))
            rr.log("/comparison/projected_overlay", rr.Image(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=int(args.jpeg_quality)))
            rr.log("/comparison/side_by_side", rr.Image(cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=int(args.jpeg_quality)))
            if frame_idx == selected[0] or frame_idx % 25 == 0 or frame_idx == selected[-1]:
                print(f"[v20-vis] frame {frame_idx + 1}/{selected[-1] + 1}", flush=True)
        video.release()
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="/world",
                        name="World: SAM3D + MANO + P09 + camera",
                        contents=[
                            "+ /world/sam3d/**",
                            "+ /world/hands/**",
                            "+ /world/visible_surface",
                            "+ /world/camera/**",
                        ],
                    ),
                    rrb.Vertical(
                        rrb.Spatial2DView(origin="/comparison/original_video", name="Original video frame", contents="+ $origin"),
                        rrb.Spatial2DView(origin="/comparison/projected_overlay", name="Projected comparison", contents="+ $origin"),
                        rrb.Spatial2DView(origin="/comparison/side_by_side", name="Original | projected overlay", contents="+ $origin"),
                    ),
                    column_shares=[3, 2],
                ),
                collapse_panels=False,
            )
        )
        rr.disconnect()
        encode_video(overlay_frame_dir, args.output_video, video_fps, start_number=selected[0])
    finally:
        shutil.rmtree(overlay_frame_dir, ignore_errors=True)

    report = {
        "schema": "v20_focused_sam3d_mano_visible_surface_camera_rrd_v1",
        "status": "ok",
        "diagnostic_only": True,
        "formal_state_modified": False,
        "pose_report_role": "prediction-side V20 K5 global-orientation + local-refine visualization transform",
        "generated_mesh_role": "render_only_completion_hypothesis",
        "generated_geometry_pose_authority": False,
        "generated_geometry_collision_authority": False,
        "generated_geometry_contact_authority": False,
        "rrd": str(args.output_rrd.expanduser().resolve()),
        "comparison_video": str(args.output_video.expanduser().resolve()),
        "inputs": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "pose_report": str(args.pose_report.expanduser().resolve()),
            "generated_mesh_render_only": str(args.generated_mesh.expanduser().resolve()),
            "mano_bridge": str(args.mano_bridge.expanduser().resolve()),
            "mano_topology": str(args.mano_topology.expanduser().resolve()),
            "source_video": str(args.source_video.expanduser().resolve()),
            "object_id": args.object_id,
            "sha256": {
                "annotations": sha256_file(args.annotations),
                "pose_report": sha256_file(args.pose_report),
                "generated_mesh_render_only": sha256_file(args.generated_mesh),
                "mano_bridge": sha256_file(args.mano_bridge),
                "mano_topology": sha256_file(args.mano_topology),
                "source_video": sha256_file(args.source_video),
            },
        },
        "timeline": {
            "frame_start": selected[0],
            "frame_end": selected[-1],
            "frame_count": len(selected),
            "source_video_fps": video_fps,
            "source_video_size_wh": [video_width, video_height],
            "display_image_size_wh": [960, 960],
            "metric_surface_frame_count": len(metric_frames),
            "metric_surface_frames": metric_frames,
            "no_metric_surface_frames": no_metric_frames,
        },
        "layers": {
            "sam3d": {"entity": "/world/sam3d/mesh", "parent_transform": "/world/sam3d", "vertices": int(len(generated_vertices)), "faces": int(len(generated_faces)), "color": "translucent magenta", "role": "render-only completion hypothesis"},
            "mano": {"entity": "/world/hands/{left,right}", "vertices_per_hand": 778, "faces_per_hand": 1538, "source": "prediction-side HaWoR/MANO bridge", "role": "visual reference only"},
            "p09_visible_surface": {"entity": "/world/visible_surface", "points_per_metric_frame": 2500, "rrd_stride": int(args.rrd_point_stride), "color": "yellow", "source": "prediction-side corrected P09 first-hit world points"},
            "camera": {"entity": "/world/camera", "image_entity": "/world/camera/image", "source": "official prediction-side camera contract + original input video"},
            "comparison": {"original": "/comparison/original_video", "projected_overlay": "/comparison/projected_overlay", "side_by_side": "/comparison/side_by_side", "video": str(args.output_video.expanduser().resolve())},
        },
        "coordinate_contract": {
            "object": "V_world = V_canonical @ R_world_from_completed_canonical.T + t_world",
            "camera": "p_world = p_camera @ T_world_camera[:3,:3].T + T_world_camera[:3,3]",
            "image_resize": "OpenCV half-pixel affine from 1408x1408 to 960x960",
            "generated_geometry_consumed_as_authority": False,
        },
        "claim_scope": "Visualization only. The generated SAM3D mesh is transformed by the prediction-side V20 K5 pose for display and video comparison; it is not used as pose/collision/contact/SDF authority. MANO and P09 points are prediction-side observations, not GT.",
    }
    report_path = args.output_rrd.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rrd": str(args.output_rrd), "comparison_video": str(args.output_video), "report": str(report_path), "frame_count": len(selected), "metric_surface_frames": len(metric_frames), "no_metric_surface_frames": no_metric_frames}, indent=2))


if __name__ == "__main__":
    main()
