#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compare_hand_streams_scale055_v3 import load_frame_window
from diagnose_hand_reprojection_depth_v3 import project_points
from optimize_object_factor_graph_v3 import localize_path, resize_bool_mask


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mesh_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(float)
    faces = blob["faces"].astype(np.int32)
    out = {}
    for i, idx in enumerate(frame_idx.tolist()):
        out[int(idx)] = (
            vertices[int(vertex_offsets[i]) : int(vertex_offsets[i + 1])],
            faces[int(face_offsets[i]) : int(face_offsets[i + 1])],
        )
    return out


def camera_points(world_points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[world_points, np.ones(len(world_points), dtype=float)]
    return (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx}")
    return frame


def draw_object_mask(frame: np.ndarray, ann: dict, args: argparse.Namespace) -> None:
    obj = ann.get("object", {})
    if not obj.get("mask_path"):
        return
    mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
    mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    tint = np.zeros_like(frame)
    tint[:, :] = (60, 80, 210)
    frame[mask] = cv2.addWeighted(frame, 0.68, tint, 0.32, 0.0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (40, 40, 180), 2, cv2.LINE_AA)


def draw_mesh_projection(frame: np.ndarray, ann: dict, mesh: tuple[np.ndarray, np.ndarray], max_edges: int) -> None:
    vertices_world, faces = mesh
    T = np.asarray(ann["camera"]["T_world_camera_metric"], dtype=float)
    intr = np.asarray(ann["camera"]["vggt_source_intrinsics_fx_fy_cx_cy"], dtype=float)
    vertices = camera_points(vertices_world, T)
    positive = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    uv = np.full((len(vertices), 2), np.nan, dtype=float)
    uv[positive] = project_points(vertices[positive], intr)
    valid_faces = np.flatnonzero(np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(vertices[faces, 2] > 0.0, axis=1))
    if len(valid_faces) > max_edges:
        valid_faces = valid_faces[np.linspace(0, len(valid_faces) - 1, max_edges, dtype=int)]
    for face_id in valid_faces:
        poly = np.round(uv[faces[int(face_id)]]).astype(np.int32)
        if np.any(poly[:, 0] < -frame.shape[1]) or np.any(poly[:, 0] > 2 * frame.shape[1]):
            continue
        if np.any(poly[:, 1] < -frame.shape[0]) or np.any(poly[:, 1] > 2 * frame.shape[0]):
            continue
        cv2.polylines(frame, [poly], True, (60, 210, 210), 1, cv2.LINE_AA)


def draw_hand(frame: np.ndarray, hand: dict) -> None:
    color = (80, 235, 80) if str(hand.get("side")) == "right" else (245, 170, 55)
    if not bool(hand.get("measurement_available", False)):
        color = (120, 120, 120)
    raw = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    joints = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
    intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    if raw.shape == (21, 2):
        for point in raw:
            p = tuple(np.rint(point).astype(int))
            cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 5, (20, 20, 20), 1, cv2.LINE_AA)
    if joints.shape == (21, 3) and intr.shape == (4,):
        uv = project_points(joints, intr)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int)), color, 3, cv2.LINE_AA)
        for point in uv:
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 4, color, -1, cv2.LINE_AA)


def hand_vertices(hand: dict) -> np.ndarray:
    for key in ("vertices_source_camera_m", "vertices_source_camera_m_sample"):
        if key in hand:
            arr = np.asarray(hand[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
    raise RuntimeError("hand has no source-camera vertices")


def draw_contact_patch(frame: np.ndarray, hand: dict, row: dict) -> None:
    vertices = hand_vertices(hand)
    intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    if intr.shape != (4,):
        raise RuntimeError("contact hand has no source intrinsics")
    ids = [int(i) for i in row.get("best_patch_vertex_ids", [])]
    if not ids:
        return
    uv = project_points(vertices[np.asarray(ids, dtype=int)], intr)
    for point in uv:
        p = tuple(np.rint(point).astype(int))
        cv2.circle(frame, p, 9, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 11, (0, 0, 0), 2, cv2.LINE_AA)


def put_label(frame: np.ndarray, frame_idx: int, row: dict | None) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
    if row is None:
        text = f"frame {frame_idx}  no reliable mesh-surface contact"
    else:
        text = (
            f"frame {frame_idx}  {row['side']} hand mesh contact  "
            f"reproj {row['median_joint_reprojection_px']:.1f}px  "
            f"surface p95 {row['best_patch_distance_p95_m']*1000:.1f}mm  "
            f"signed p95 {row['best_patch_signed_gap_p95_abs_m']*1000:.1f}mm"
        )
    cv2.putText(frame, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def run(args: argparse.Namespace) -> dict:
    annotations = load_frame_window(args.annotations, args.frame_start, args.frame_end)
    contact = load_json(args.contact_report)
    rows = [row for row in contact.get("rows_detail", []) if bool(row.get("reliable_for_contact", False))]
    contact_by_frame = {int(row["frame_idx"]): row for row in rows}
    frames = sorted(set(range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride))) | set(contact_by_frame))
    meshes = load_mesh_archive(args.object_mesh_npz)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = float(args.output_fps) if args.output_fps is not None else float(cap.get(cv2.CAP_PROP_FPS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    writer = None
    written = []
    try:
        for frame_idx in frames:
            ann = annotations.get(int(frame_idx))
            if ann is None:
                continue
            image = read_frame(cap, int(frame_idx))
            draw_object_mask(image, ann, args)
            if int(frame_idx) in meshes:
                draw_mesh_projection(image, ann, meshes[int(frame_idx)], int(args.max_mesh_edges))
            row = contact_by_frame.get(int(frame_idx))
            for hand in ann.get("hands", []):
                draw_hand(image, hand)
            if row is not None:
                hand = ann["hands"][int(row["hand_idx"])]
                draw_contact_patch(image, hand, row)
            put_label(image, int(frame_idx), row)
            if args.render_width and image.shape[1] != int(args.render_width):
                height = int(round(int(args.render_width) * image.shape[0] / image.shape[1]))
                image = cv2.resize(image, (int(args.render_width), height), interpolation=cv2.INTER_AREA)
            if writer is None:
                video_path = args.output_dir / "mesh_surface_contact_review.mp4"
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (image.shape[1], image.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open writer {video_path}")
            writer.write(image)
            if row is not None or int(frame_idx) in set(args.still_frames):
                still = still_dir / f"frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(still), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError(f"failed to write {still}")
                written.append(str(still))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    report = {
        "status": "ok",
        "video": str(args.output_dir / "mesh_surface_contact_review.mp4"),
        "stills_dir": str(still_dir),
        "written_stills": written,
        "contact_frames": sorted(int(frame) for frame in contact_by_frame),
        "annotations": str(args.annotations),
        "contact_report": str(args.contact_report),
        "object_mesh_npz": str(args.object_mesh_npz),
    }
    (args.output_dir / "review_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--max-mesh-edges", type=int, default=260)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[858, 866, 867, 868, 869, 879, 880])
    parser.add_argument("--remote-output-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/data"))
    parser.add_argument("--local-output-root", type=Path, default=Path("/data2/ego_annotation_outputs"))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
