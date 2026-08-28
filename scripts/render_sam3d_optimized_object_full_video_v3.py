#!/usr/bin/env python3
"""Render a full-duration camera video of the corrected SAM3D object mesh.

Every source frame is preserved. The bounded first-hit-aligned anchor-camera
mesh is propagated with P15 per-frame object SE(3), transformed by
``T_world_camera_metric``, and rendered with a real triangle z-buffer. HaWoR
MANO meshes provide hand-depth occlusion so the object overlay does not paint
over fingers that are in front of the object.

Outputs:
  * camera overlay video (same frame count/fps as source manifest),
  * raw-vs-overlay side-by-side video,
  * optional rendered PNG sequence and QC JSON.

Generated hidden object faces remain a visual completion prior; this renderer
does not promote them to collision/sign/contact authority.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def camera_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]


def world_to_camera(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]


def anchor_camera_to_canonical(
    vertices: np.ndarray,
    anchor_camera: np.ndarray,
    anchor_rotation: np.ndarray,
    anchor_translation: np.ndarray,
) -> np.ndarray:
    world = camera_to_world(vertices, anchor_camera)
    return (world - anchor_translation[None, :]) @ anchor_rotation


def canonical_to_camera(
    canonical: np.ndarray,
    rotation_world_object: np.ndarray,
    translation_world_object: np.ndarray,
    T_world_camera: np.ndarray,
) -> np.ndarray:
    world = canonical @ rotation_world_object.T + translation_world_object[None, :]
    return world_to_camera(world, T_world_camera)


def resize_intrinsics(K: np.ndarray, source_size: tuple[int, int], render_size: tuple[int, int]) -> np.ndarray:
    source_w, source_h = source_size
    render_w, render_h = render_size
    sx = render_w / source_w
    sy = render_h / source_h
    result = np.asarray(K, dtype=np.float64).copy()
    result[0, 0] *= sx
    result[1, 1] *= sy
    result[0, 2] = sx * (result[0, 2] + 0.5) - 0.5
    result[1, 2] = sy * (result[1, 2] + 0.5) - 0.5
    return result


def make_rasterizer(K: np.ndarray, width: int, height: int, device: str) -> MeshRasterizer:
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


def render_fragments(
    rasterizer: MeshRasterizer,
    vertices: np.ndarray,
    faces_t: torch.Tensor,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        fragments = rasterizer(Meshes(
            verts=[torch.tensor(vertices, dtype=torch.float32, device=device)],
            faces=[faces_t],
        ))
    pix = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    zbuf = fragments.zbuf[0, ..., 0].detach().cpu().numpy().astype(np.float64)
    bary = fragments.bary_coords[0, ..., 0, :].detach().cpu().numpy().astype(np.float64)
    valid = (pix >= 0) & np.isfinite(zbuf) & (zbuf > 1.0e-6)
    return pix, np.where(valid, zbuf, np.nan), bary


def face_shading(vertices: np.ndarray, faces: np.ndarray, base_bgr: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1.0e-12)
    # Two-sided completion mesh: use |normal·view| plus a top-left fill light.
    centres = triangles.mean(axis=1)
    view = -centres
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1.0e-12)
    frontal = np.abs(np.sum(normals * view, axis=1))
    light = np.asarray([-0.35, -0.45, -0.82], dtype=np.float64)
    light /= np.linalg.norm(light)
    diffuse = np.abs(normals @ light)
    intensity = np.clip(0.34 + 0.38 * frontal + 0.28 * diffuse, 0.30, 1.0)
    return np.clip(base_bgr[None, :] * intensity[:, None], 0.0, 255.0).astype(np.uint8)


def mask_metrics(rendered: np.ndarray, observed: np.ndarray) -> dict[str, float | int | None]:
    rendered = np.asarray(rendered, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    intersection = int(np.count_nonzero(rendered & observed))
    union = int(np.count_nonzero(rendered | observed))
    rendered_count = int(np.count_nonzero(rendered))
    observed_count = int(np.count_nonzero(observed))
    yr, xr = np.nonzero(rendered)
    yo, xo = np.nonzero(observed)
    centroid_error = None
    if len(xr) and len(xo):
        centroid_error = float(np.hypot(xr.mean() - xo.mean(), yr.mean() - yo.mean()))

    kernel = np.ones((3, 3), np.uint8)
    rendered_boundary = cv2.morphologyEx(rendered.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    observed_boundary = cv2.morphologyEx(observed.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    distances: list[np.ndarray] = []
    for source_boundary, target_boundary in (
        (rendered_boundary, observed_boundary),
        (observed_boundary, rendered_boundary),
    ):
        if np.any(source_boundary) and np.any(target_boundary):
            distance_to_target = cv2.distanceTransform((~target_boundary).astype(np.uint8), cv2.DIST_L2, 5)
            distances.append(distance_to_target[source_boundary])
    boundary = np.concatenate(distances) if distances else np.empty(0, dtype=np.float32)
    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(intersection / union) if union else None,
        "observed_coverage": float(intersection / observed_count) if observed_count else None,
        "rendered_outside_fraction": float((rendered_count - intersection) / rendered_count) if rendered_count else None,
        "centroid_error_px": centroid_error,
        "symmetric_boundary_median_px": float(np.median(boundary)) if len(boundary) else None,
        "symmetric_boundary_p95_px": float(np.percentile(boundary, 95.0)) if len(boundary) else None,
    }


def draw_mesh_overlay(
    rgb_bgr: np.ndarray,
    pix: np.ndarray,
    mesh_depth: np.ndarray,
    face_colors: np.ndarray,
    hand_depth: np.ndarray | None,
    alpha: float,
    contour_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rendered = pix >= 0
    visible = rendered.copy()
    if hand_depth is not None:
        hand_hit = np.isfinite(hand_depth)
        visible &= ~(hand_hit & (hand_depth < mesh_depth - 0.0015))
    color = np.zeros_like(rgb_bgr)
    color[rendered] = face_colors[pix[rendered]]
    output = rgb_bgr.copy()
    output[visible] = np.clip(
        (1.0 - alpha) * output[visible].astype(np.float64) + alpha * color[visible].astype(np.float64),
        0.0,
        255.0,
    ).astype(np.uint8)
    contour = cv2.morphologyEx(visible.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    output[contour] = np.asarray(contour_color, dtype=np.uint8)
    return output, visible


def draw_observed_surface(
    image: np.ndarray,
    observed_camera: np.ndarray,
    K: np.ndarray,
    mesh_depth: np.ndarray,
    radius: int,
) -> None:
    points = np.asarray(observed_camera, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1.0e-6)
    points = points[valid]
    u = np.rint(K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2]).astype(np.int64)
    v = np.rint(K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
    u, v, points = u[inside], v[inside], points[inside]
    # Draw only points close to the rendered first hit; edge/tail outliers are not
    # allowed to dominate the visual surface evidence.
    hit = mesh_depth[v, u]
    close = np.isfinite(hit) & (np.abs(hit - points[:, 2]) <= 0.020)
    for x, y in zip(u[close], v[close]):
        cv2.circle(image, (int(x), int(y)), radius, (0, 235, 255), -1, lineType=cv2.LINE_AA)


def encode_pngs(frame_dir: Path, output: Path, fps: float, pattern: str = "%06d.png") -> None:
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:.8g}", "-i", str(frame_dir / pattern),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True)


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    report = load_json(args.manifest_report)
    annotations = load_json(args.annotations)
    pose_graph = load_json(args.pose_graph)
    frame_by_idx = {int(f["frame_idx"]): f for f in annotations["frames"]}
    pose_by_idx = {int(r["frame_idx"]): r for r in pose_graph["pose_rows"]}
    frame_count = int(manifest["frame_count"])
    fps = float(report["fps"])
    expected = list(range(frame_count))
    if sorted(frame_by_idx) != expected:
        raise RuntimeError("annotations do not cover the complete source timeline")
    if sorted(pose_by_idx) != expected:
        raise RuntimeError("P15 pose graph does not cover the complete source timeline")

    first = cv2.imread(str(args.rgb_dir / "000000.jpg"))
    if first is None:
        raise RuntimeError("cannot decode source frame 0")
    manifest_h, manifest_w = first.shape[:2]
    geometry_source_w = int(frame_by_idx[0]["source_width"])
    geometry_source_h = int(frame_by_idx[0]["source_height"])
    if geometry_source_w <= 0 or geometry_source_h <= 0:
        raise RuntimeError("invalid source image plane in annotations")
    for frame in frame_by_idx.values():
        if (int(frame["source_width"]), int(frame["source_height"])) != (geometry_source_w, geometry_source_h):
            raise RuntimeError("per-frame source image plane is not constant")
    render_w = args.render_width
    render_h = int(round(manifest_h * render_w / manifest_w))
    if render_w % 2 or render_h % 2:
        raise RuntimeError("H.264 yuv420p output dimensions must be even")

    object_mesh = trimesh.load(args.object_mesh_anchor_camera, force="mesh", process=False)
    if not isinstance(object_mesh, trimesh.Trimesh):
        raise RuntimeError(f"invalid object mesh: {args.object_mesh_anchor_camera}")
    object_anchor = np.asarray(object_mesh.vertices, dtype=np.float64)
    object_faces = np.asarray(object_mesh.faces, dtype=np.int64)
    object_faces_t = torch.tensor(object_faces, dtype=torch.int64, device=args.device)

    anchor_frame = frame_by_idx[args.anchor_frame]
    anchor_pose = pose_by_idx[args.anchor_frame]
    T_anchor = np.asarray(anchor_frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    R_anchor = np.asarray(anchor_pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    t_anchor = np.asarray(anchor_pose["translation_world_m"], dtype=np.float64)
    object_canonical = anchor_camera_to_canonical(object_anchor, T_anchor, R_anchor, t_anchor)

    hands = np.load(args.hand_npz)
    hand_position = {int(idx): pos for pos, idx in enumerate(hands["frame_idx"].astype(int).tolist())}
    intrinsics = frame_by_idx[0]["camera"]["intrinsics_fx_fy_cx_cy"]
    K_source = np.asarray([
        [intrinsics[0], 0.0, intrinsics[2]],
        [0.0, intrinsics[1], intrinsics[3]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    for frame in frame_by_idx.values():
        if not np.allclose(frame["camera"]["intrinsics_fx_fy_cx_cy"], intrinsics, atol=1.0e-6):
            raise RuntimeError("per-frame camera intrinsics are not constant")
    # K_source is defined on the 1408x1408 source pinhole plane, while the
    # manifest RGB JPEGs are resized review images (960x960 for this clip).
    # The old renderer incorrectly used the JPEG size as K_source's plane.
    K_render = resize_intrinsics(
        K_source,
        (geometry_source_w, geometry_source_h),
        (render_w, render_h),
    )
    rasterizer = make_rasterizer(K_render, render_w, render_h, args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlay_frames"
    side_dir = args.output_dir / "side_by_side_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)
    per_frame: dict[str, Any] = {}
    visible_rows = 0

    for idx in expected:
        raw = cv2.imread(str(args.rgb_dir / f"{idx:06d}.jpg"))
        if raw is None:
            raw = cv2.imread(str(args.rgb_dir / f"{idx:06d}.png"))
        if raw is None:
            raise RuntimeError(f"cannot decode source frame {idx}")
        raw = cv2.resize(raw, (render_w, render_h), interpolation=cv2.INTER_AREA)
        frame = frame_by_idx[idx]
        pose = pose_by_idx[idx]
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        R = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        t = np.asarray(pose["translation_world_m"], dtype=np.float64)
        object_camera = canonical_to_camera(object_canonical, R, t, T)
        pix, mesh_depth, _ = render_fragments(rasterizer, object_camera, object_faces_t, args.device)
        colors = face_shading(object_camera, object_faces, np.asarray([235.0, 185.0, 45.0]))

        hand_depth: np.ndarray | None = None
        hand_valid_sides: list[str] = []
        if idx in hand_position:
            pos = hand_position[idx]
            hand_vertices: list[np.ndarray] = []
            hand_faces: list[np.ndarray] = []
            offset = 0
            for side in ("left", "right"):
                if int(np.asarray(hands[f"{side}_valid"])[pos]) != 1:
                    continue
                vertices_world = np.asarray(hands[f"{side}_vertices_world_m"][pos], dtype=np.float64)
                vertices_camera = world_to_camera(vertices_world, T)
                faces = np.asarray(hands[f"{side}_faces"], dtype=np.int64)
                hand_vertices.append(vertices_camera)
                hand_faces.append(faces + offset)
                offset += len(vertices_camera)
                hand_valid_sides.append(side)
            if hand_vertices:
                hv = np.concatenate(hand_vertices, axis=0)
                hf = np.concatenate(hand_faces, axis=0)
                hf_t = torch.tensor(hf, dtype=torch.int64, device=args.device)
                _, hand_depth, _ = render_fragments(rasterizer, hv, hf_t, args.device)

        overlay, visible = draw_mesh_overlay(
            raw, pix, mesh_depth, colors, hand_depth, args.alpha, (255, 240, 40)
        )
        object = frame["objects"][0]
        visible_geometry = object.get("visible_geometry_candidate")
        surfel_count = 0
        mask_path = Path(str(object.get("mask_path") or ""))
        mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            raise RuntimeError(f"missing object-owned mask for frame {idx}: {mask_path}")
        mask = cv2.resize(
            mask_u8,
            (render_w, render_h),
            interpolation=getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST),
        ) > 0
        owned_mask_pixels = int(np.count_nonzero(mask))
        mask_contour = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        overlay[mask_contour] = np.asarray([60, 60, 255], dtype=np.uint8)
        if args.draw_visible_surface and isinstance(visible_geometry, dict):
            observed = np.asarray(visible_geometry.get("camera_vertices_sample_m"), dtype=np.float64)
            if observed.ndim == 2 and observed.shape[1] == 3:
                draw_observed_surface(overlay, observed, K_render, mesh_depth, args.surfel_radius_px)
                surfel_count = int(len(observed))
                visible_rows += 1

        caption = f"{args.caption} | frame {idx:03d}/{frame_count - 1:03d} | P15 per-frame SE(3)"
        cv2.rectangle(overlay, (0, 0), (render_w, 34), (0, 0, 0), -1)
        cv2.putText(overlay, caption, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        if args.draw_visible_surface:
            cv2.circle(overlay, (render_w - 210, 17), 4, (0, 235, 255), -1)
            cv2.putText(overlay, "trusted visible surfels", (render_w - 198, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 235, 255), 1, cv2.LINE_AA)
        side = np.hstack((raw, overlay))
        cv2.imwrite(str(overlay_dir / f"{idx:06d}.png"), overlay)
        cv2.imwrite(str(side_dir / f"{idx:06d}.png"), side)
        per_frame[str(idx)] = {
            "rendered_mesh_pixels": int(np.count_nonzero(pix >= 0)),
            "visible_after_hand_occlusion_pixels": int(np.count_nonzero(visible)),
            "hand_occluder_sides": hand_valid_sides,
            "visible_surfel_count": surfel_count,
            "object_owned_mask_pixels": owned_mask_pixels,
            "full_render_vs_owned_mask": mask_metrics(pix >= 0, mask),
            "hand_occluded_render_vs_owned_mask": mask_metrics(visible, mask),
        }
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == frame_count - 1):
            print(f"rendered {idx + 1}/{frame_count}", flush=True)

    def summarize_metric(section: str, metric: str) -> dict[str, float | int]:
        values = np.asarray([
            row[section][metric]
            for row in per_frame.values()
            if row[section][metric] is not None
        ], dtype=np.float64)
        return {
            "count": int(len(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10.0)),
            "p90": float(np.percentile(values, 90.0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    silhouette_summary = {
        section: {
            metric: summarize_metric(section, metric)
            for metric in (
                "iou",
                "observed_coverage",
                "rendered_outside_fraction",
                "centroid_error_px",
                "symmetric_boundary_median_px",
                "symmetric_boundary_p95_px",
            )
        }
        for section in ("full_render_vs_owned_mask", "hand_occluded_render_vs_owned_mask")
    }

    overlay_video = args.output_dir / "optimized_object_camera_overlay.mp4"
    side_video = args.output_dir / "optimized_object_side_by_side.mp4"
    encode_pngs(overlay_dir, overlay_video, fps)
    encode_pngs(side_dir, side_video, fps)
    overlay_probe = ffprobe(overlay_video)
    side_probe = ffprobe(side_video)
    qc = {
        "schema": "sam3d_optimized_object_full_video_v2",
        "status": "rendered_with_source_to_review_intrinsics_contract",
        "annotation_ready": False,
        "diagnostic_only": True,
        "frame_count": frame_count,
        "fps": fps,
        "duration_s": frame_count / fps,
        "geometry_source_size_wh": [geometry_source_w, geometry_source_h],
        "manifest_rgb_size_wh": [manifest_w, manifest_h],
        "render_size_wh": [render_w, render_h],
        "K_geometry_source": K_source.tolist(),
        "K_render": K_render.tolist(),
        "intrinsics_mapping": "K_geometry_source on annotation source plane resized to render plane with pixel-centre convention",
        "object_mesh_anchor_camera": str(args.object_mesh_anchor_camera),
        "pose_graph": str(args.pose_graph),
        "camera_contract": "T_world_camera_metric; OpenCV camera axes",
        "renderer": "PyTorch3D true triangle first-hit raster; normal shading; MANO z-buffer hand occlusion",
        "visible_surface_overlay": bool(args.draw_visible_surface),
        "visible_surface_frames": visible_rows,
        "outputs": {
            "camera_overlay_video": str(overlay_video),
            "side_by_side_video": str(side_video),
            "overlay_frames": str(overlay_dir),
            "side_by_side_frames": str(side_dir),
        },
        "ffprobe": {"camera_overlay": overlay_probe, "side_by_side": side_probe},
        "silhouette_summary": silhouette_summary,
        "per_frame": per_frame,
        "claim_scope": (
            "Full-duration visual QC of the corrected first-hit SAM3D render prior under P15 per-frame object SE(3). "
            "Generated hidden faces remain render-only; video does not establish collision/sign/contact authority."
        ),
    }
    qc_path = args.output_dir / "qc_optimized_object_full_video.json"
    qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "overlay": str(overlay_video), "side_by_side": str(side_video), "qc": str(qc_path)}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-mesh-anchor-camera", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-graph", type=Path, required=True)
    parser.add_argument("--hand-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--caption", default="SAM3D first-hit mesh")
    parser.add_argument("--draw-visible-surface", action="store_true")
    parser.add_argument("--surfel-radius-px", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
