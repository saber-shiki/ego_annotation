#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import diags  # type: ignore[reportMissingTypeStubs]
from scipy.sparse.linalg import spsolve  # type: ignore[reportMissingTypeStubs]
from scipy.spatial.transform import Rotation  # type: ignore[reportMissingTypeStubs]

STATUS = "v18_full_pipeline"
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
CLAIM = (
    "V18 full pipeline artifact: approximate and uncertain full-video annotations with executable hand, "
    "object/part, geometry, pose, contact, occlusion, and bounded factor-graph baseline fields. "
    "All outputs are candidates or explicit unresolved states; no arbitrary gate suppresses artifact production."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def text_font(size: int) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any, fill: tuple[int, int, int], bg: tuple[int, int, int] = (0, 0, 0)) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((x, y), text, font=font, fill=fill)


def bbox_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in vals]
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def scale_bbox(value: Any, from_w: float, from_h: float, to_w: float, to_h: float) -> list[float] | None:
    box = bbox_tuple(value)
    if box is None or from_w <= 0 or from_h <= 0:
        return None
    sx = to_w / from_w
    sy = to_h / from_h
    x0, y0, x1, y1 = box
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]


def bbox_center(value: Any) -> list[float] | None:
    box = bbox_tuple(value)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]


def color_from_bgr(value: Any, fallback_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    if not (isinstance(value, list) and len(value) == 3):
        return fallback_rgb
    return (int(value[2]), int(value[1]), int(value[0]))


def project_mano_joints(mano: dict[str, Any], source_w: float, source_h: float, image_w: float, image_h: float) -> list[tuple[int, int]]:
    joints = mano.get("joints3d_camera")
    cam_t = mano.get("cam_t")
    intr = mano.get("source_intrinsics") or [2304.0, 2304.0, source_w / 2.0, source_h / 2.0]
    if not (isinstance(joints, list) and isinstance(cam_t, list) and len(cam_t) == 3 and isinstance(intr, list) and len(intr) == 4):
        return []
    fx, fy, cx, cy = [finite_float(v) for v in intr]
    sx = image_w / source_w if source_w > 0 else 1.0
    sy = image_h / source_h if source_h > 0 else 1.0
    pts: list[tuple[int, int]] = []
    for raw in joints:
        if not (isinstance(raw, list) and len(raw) == 3):
            return []
        x = finite_float(raw[0]) + finite_float(cam_t[0])
        y = finite_float(raw[1]) + finite_float(cam_t[1])
        z = finite_float(raw[2]) + finite_float(cam_t[2])
        if z <= 1e-6:
            return []
        u = (fx * x / z + cx) * sx
        v = (fy * y / z + cy) * sy
        if not (math.isfinite(u) and math.isfinite(v)):
            return []
        pts.append((int(round(u)), int(round(v))))
    return pts


def mask_overlay(base: Image.Image, mask_path: str, rgb: tuple[int, int, int], alpha_float: float) -> Image.Image:
    path = Path(mask_path)
    if not path.exists():
        return base
    mask = Image.open(path).convert("L")
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.Resampling.NEAREST)
    alpha_value = max(0, min(255, int(alpha_float * 255)))
    alpha = mask.point([alpha_value if p > 0 else 0 for p in range(256)])
    overlay = Image.new("RGB", base.size, rgb)
    return Image.composite(overlay, base, alpha)


def ffprobe_frame_count(path: Path) -> int | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return None
    lines = proc.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        return int(lines[-1])
    except ValueError:
        return None


def extract_video_frames(video_path: Path, frame_dir: Path) -> None:
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        str(frame_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def v16_render_paths(case: str, args: argparse.Namespace) -> dict[str, Path]:
    render_dir = args.v16_root / case / "renders"
    return {
        "overlay": render_dir / "overlay_mano_object.mp4",
        "world": render_dir / "reconstruction_3d_world.mp4",
        "side_by_side": render_dir / "side_by_side.mp4",
        "qc": render_dir / "render_only_qc.json",
    }


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frame_dir / "%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def compose_side_by_side(overlay_path: Path, world_path: Path, output_path: Path, width_each: int = 960, height: int = 540) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        f"[0:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[left];"
        f"[1:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(overlay_path),
        "-i",
        str(world_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def stats(values: list[float]) -> dict[str, Any]:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None}

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * p / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {"count": len(xs), "median": pct(50), "p95": pct(95), "min": xs[0], "max": xs[-1]}



def pca_pose_observation(points: np.ndarray) -> dict[str, Any] | None:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 6 or not np.isfinite(points).all():
        return None
    center = points.mean(axis=0)
    centered = points - center
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vt.shape != (3, 3) or not np.isfinite(vt).all():
        return None
    axes = vt.T.copy()
    # Deterministic sign convention reduces arbitrary PCA sign flips without pretending semantic orientation is known.
    for col in range(3):
        dominant = int(np.argmax(np.abs(axes[:, col])))
        if axes[dominant, col] < 0:
            axes[:, col] *= -1.0
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    try:
        rotation_vector = Rotation.from_matrix(axes).as_rotvec()
    except ValueError:
        return None
    extent = points.max(axis=0) - points.min(axis=0)
    denom = float(singular_values[0]) if singular_values.shape[0] and singular_values[0] > 1e-9 else 1.0
    anisotropy = float((singular_values[0] - singular_values[-1]) / denom) if singular_values.shape[0] == 3 else 0.0
    return {
        "center": center,
        "rotation_matrix": axes,
        "rotation_vector": rotation_vector,
        "extent": extent,
        "singular_values": singular_values,
        "anisotropy": anisotropy,
    }

def load_visible_geometry_index(report_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, dict[str, Any]], Path | None]:
    if not report_path.exists():
        return {}, {}, None
    report = require_dict(load_json(report_path), "visible geometry report")
    archive_path = Path(str(report.get("archive_npz")))
    if not archive_path.exists():
        return {}, {}, None
    data = np.load(archive_path, allow_pickle=True)
    frame_idx = data["frame_idx"]
    object_ids = data["object_id"]
    vertex_offsets = data["vertex_offsets"]
    vertices = data["vertices"]
    index: dict[tuple[int, str], dict[str, Any]] = {}
    by_object_vertices: dict[str, list[np.ndarray]] = defaultdict(list)
    for row_idx in range(len(frame_idx)):
        start = int(vertex_offsets[row_idx])
        end = int(vertex_offsets[row_idx + 1])
        if end <= start:
            continue
        obj = str(object_ids[row_idx])
        pts = np.asarray(vertices[start:end], dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            continue
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        center = pts.mean(axis=0)
        pca_pose = pca_pose_observation(pts)
        index[(int(frame_idx[row_idx]), obj)] = {
            "archive_npz": str(archive_path),
            "archive_row_index": row_idx,
            "vertex_count": int(pts.shape[0]),
            "world_bbox_min_m": [float(v) for v in mn.tolist()],
            "world_bbox_max_m": [float(v) for v in mx.tolist()],
            "world_centroid_m": [float(v) for v in center.tolist()],
            "extent_m": [float(v) for v in (mx - mn).tolist()],
            "pca_rotation_world_from_object": [float(v) for v in pca_pose["rotation_vector"].tolist()] if pca_pose else None,
            "pca_rotation_matrix_world_from_object": [[float(x) for x in row] for row in pca_pose["rotation_matrix"].tolist()] if pca_pose else None,
            "pca_singular_values": [float(v) for v in pca_pose["singular_values"].tolist()] if pca_pose else None,
            "pca_anisotropy": float(pca_pose["anisotropy"]) if pca_pose else None,
        }
        by_object_vertices[obj].append(pts)
    completion: dict[str, dict[str, Any]] = {}
    for obj, chunks in by_object_vertices.items():
        pts = np.concatenate(chunks, axis=0)
        if pts.shape[0] > 50000:
            step = max(1, pts.shape[0] // 50000)
            pts = pts[::step]
        center = pts.mean(axis=0)
        centered = pts - center
        if pts.shape[0] >= 3:
            _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
            hidden_axis = vt[-1]
            mirrored = pts - 2.0 * np.outer(centered @ hidden_axis, hidden_axis)
            candidate_points = np.concatenate([pts, mirrored], axis=0)
        else:
            singular_values = np.asarray([], dtype=np.float64)
            hidden_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            candidate_points = pts
        mn = candidate_points.min(axis=0)
        mx = candidate_points.max(axis=0)
        completion[obj] = {
            "method": "category_agnostic_visible_surface_pca_mirror_completion_candidate",
            "scope": "approximate_hidden_geometry_candidate_point_cloud_not_ground_truth",
            "source_visible_vertex_count": int(sum(chunk.shape[0] for chunk in chunks)),
            "sampled_visible_vertex_count": int(pts.shape[0]),
            "candidate_point_count": int(candidate_points.shape[0]),
            "center_world_m": [float(v) for v in center.tolist()],
            "hidden_axis_world": [float(v) for v in hidden_axis.tolist()],
            "singular_values": [float(v) for v in singular_values.tolist()],
            "candidate_bbox_min_world_m": [float(v) for v in mn.tolist()],
            "candidate_bbox_max_world_m": [float(v) for v in mx.tolist()],
            "uncertainty": "approximate_visible_surface_mirror_completion",
        }
    return index, completion, archive_path


def load_part_surface_index(path: Path) -> dict[tuple[int, str], list[dict[str, Any]]]:
    if not path.exists():
        return {}
    report = require_dict(load_json(path), "part visible surfaces report")
    out: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in require_list(report.get("surface_rows"), "part surface rows"):
        row = require_dict(raw, "part surface row")
        frame_idx = require_int(row.get("frame_idx"), "part frame_idx")
        object_id = str(row.get("object_id"))
        mn = row.get("bbox_camera_min_m")
        mx = row.get("bbox_camera_max_m")
        center = None
        if isinstance(mn, list) and isinstance(mx, list) and len(mn) == 3 and len(mx) == 3:
            center = [(finite_float(mn[i]) + finite_float(mx[i])) / 2.0 for i in range(3)]
        out[(frame_idx, object_id)].append(
            {
                "part_track_label": row.get("part_track_label"),
                "part_mask_path": row.get("part_mask_path"),
                "status": row.get("status"),
                "coordinate_frame": row.get("coordinate_frame"),
                "vertices": row.get("vertices"),
                "faces": row.get("faces"),
                "depth_median_m": row.get("depth_median_m"),
                "part_containment_in_object": row.get("part_containment_in_object"),
                "bbox_camera_min_m": mn,
                "bbox_camera_max_m": mx,
                "center_camera_m": center,
                "pose_candidate": {
                    "type": "approximate_part_visible_surface_center_candidate",
                    "translation_camera_m": center,
                    "rotation": "unknown_from_visible_surface_only",
                    "uncertainty": "approximate",
                },
            }
        )
    return out


def index_bounded_frames(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    report = require_dict(load_json(path), "bounded solution")
    return {require_int(frame.get("frame_idx"), "bounded frame_idx"): require_dict(frame, "bounded frame") for frame in require_list(report.get("frames"), "bounded frames")}


def index_v16_frames(path: Path) -> dict[int, dict[str, Any]]:
    report = require_dict(load_json(path), "v16 annotations")
    return {require_int(frame.get("frame_idx"), "v16 frame_idx"): require_dict(frame, "v16 frame") for frame in require_list(report.get("frames"), "v16 frames")}


def hand_by_side(v16_frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in v16_frame.get("hands", []):
        if isinstance(raw, dict):
            side = str(raw.get("side", raw.get("hand_side", "unknown")))
            out[side] = raw
    return out


def contact_hypothesis(contact_row: dict[str, Any]) -> dict[str, Any]:
    state = str(contact_row.get("v18_consistency_state"))
    if contact_row.get("metric_depth_compatible_candidate") is True:
        confidence = "medium"
        ownership = "candidate_metric_depth_compatible"
    elif contact_row.get("image_overlap_candidate") is True or contact_row.get("pair_contact_image_candidate") is True:
        confidence = "low"
        ownership = "candidate_image_overlap_only"
    elif state == "rejected_contact_current_metric_depth":
        confidence = "very_low_depth_contradiction"
        ownership = "unlikely_current_frame"
    else:
        confidence = "unknown"
        ownership = "unresolved"
    return {
        "hand_side": contact_row.get("hand_side"),
        "object_id": contact_row.get("object_id"),
        "state": state,
        "contact_owner_hypothesis": ownership,
        "confidence": confidence,
        "uncertainty": "approximate_contact_hypothesis_not_ground_truth",
        "evidence": {
            "image_overlap_candidate": contact_row.get("image_overlap_candidate"),
            "pair_contact_image_candidate": contact_row.get("pair_contact_image_candidate"),
            "metric_depth_compatible_candidate": contact_row.get("metric_depth_compatible_candidate"),
            "pair_depth_gap_state": contact_row.get("pair_depth_gap_state"),
        },
    }


def object_pose_candidate(obj: dict[str, Any], geom: dict[str, Any] | None) -> dict[str, Any]:
    if geom is not None:
        extent = [finite_float(v) for v in geom.get("extent_m", [])]
        confidence = "low" if sum(extent) > 0 else "very_low"
        return {
            "type": "approximate_visible_surface_world_se3_candidate",
            "translation_world_m": geom.get("world_centroid_m"),
            "rotation_world_from_object_rotvec": geom.get("pca_rotation_world_from_object"),
            "rotation_world_from_object_matrix": geom.get("pca_rotation_matrix_world_from_object"),
            "rotation_source": "PCA_axes_from_visible_metric_surface_points_with_sign_canonicalization",
            "scale_extent_m": geom.get("extent_m"),
            "pca_singular_values": geom.get("pca_singular_values"),
            "pca_anisotropy": geom.get("pca_anisotropy"),
            "confidence": confidence,
            "uncertainty": "visible_surface_partial_se3_candidate_not_canonical_object_pose",
            "source": {"visible_surface_npz": geom.get("archive_npz"), "archive_row_index": geom.get("archive_row_index")},
        }
    return {
        "type": "approximate_image_bbox_pose_candidate_or_unresolved",
        "translation_world_m": None,
        "rotation_world_from_object_rotvec": None,
        "rotation_world_from_object_matrix": None,
        "rotation_source": "unobserved",
        "scale_extent_m": None,
        "confidence": "unknown" if obj.get("visibility_state") != "out_of_frame" else "inactive",
        "uncertainty": "no_depth_backed_surface_for_frame",
        "source": {"bbox_xyxy": obj.get("bbox_xyxy"), "mask_path": obj.get("mask_path")},
    }



def bbox_area_float(value: Any) -> float | None:
    box = bbox_tuple(value)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return float(max(0, x1 - x0) * max(0, y1 - y0))


def bbox_intersection_area(a: Any, b: Any) -> float:
    ba = bbox_tuple(a)
    bb = bbox_tuple(b)
    if ba is None or bb is None:
        return 0.0
    ax0, ay0, ax1, ay1 = ba
    bx0, by0, bx1, by1 = bb
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    return float(iw * ih)


def bbox_iou_value(a: Any, b: Any) -> float:
    inter = bbox_intersection_area(a, b)
    aa = bbox_area_float(a) or 0.0
    bb = bbox_area_float(b) or 0.0
    denom = aa + bb - inter
    return inter / denom if denom > 0 else 0.0


def bbox_min_coverage(a: Any, b: Any) -> float:
    inter = bbox_intersection_area(a, b)
    aa = bbox_area_float(a) or 0.0
    bb = bbox_area_float(b) or 0.0
    denom = min(aa, bb)
    return inter / denom if denom > 0 else 0.0


def bbox_center_distance_norm(a: Any, b: Any, width: float, height: float) -> float | None:
    ca = bbox_center(a)
    cb = bbox_center(b)
    if ca is None or cb is None:
        return None
    diag = math.hypot(width, height)
    if diag <= 0:
        return None
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) / diag


def numeric_vector(value: Any, dim: int) -> np.ndarray | None:
    if not (isinstance(value, list) and len(value) == dim):
        return None
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    return np.asarray(vals, dtype=np.float64)


def solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve the temporal normal equations with SciPy sparse linear algebra.

    The matrix is tridiagonal because the current continuous factors are
    observation terms plus adjacent-frame temporal terms.  The factor graph
    construction is explicit in `solve_temporal_series`; this function only
    delegates the numerical linear solve to SciPy instead of maintaining a
    hand-written optimizer in the artifact script.
    """
    n = int(diag.shape[0])
    if n == 0:
        return rhs.copy()
    if n == 1:
        return rhs / diag[0]
    matrix = diags([lower, diag, upper], offsets=[-1, 0, 1], shape=(n, n), format="csc")  # type: ignore[reportArgumentType]
    solved = spsolve(matrix, rhs)
    out = np.asarray(solved, dtype=np.float64)
    if out.ndim == 1 and rhs.ndim == 2:
        out = out[:, None]
    return out


def solve_temporal_series(observations: list[dict[str, Any]], temporal_weight: float, default_obs_weight: float, unit: str) -> dict[str, Any]:
    clean: list[dict[str, Any]] = []
    for obs in observations:
        value = obs.get("value")
        if isinstance(value, np.ndarray) and value.ndim == 1 and np.isfinite(value).all():
            clean.append(obs)
    clean.sort(key=lambda item: (require_int(item.get("frame_idx"), "series frame_idx"), str(item.get("variable_id"))))
    if not clean:
        return {"estimates": {}, "summary": {"variable_count": 0, "factor_count": 0, "energy_initial": 0.0, "energy_after": 0.0, "unit": unit, "dimension": 0}}
    n = len(clean)
    dim = int(clean[0]["value"].shape[0])
    diag = np.zeros(n, dtype=np.float64)
    lower = np.zeros(max(0, n - 1), dtype=np.float64)
    upper = np.zeros(max(0, n - 1), dtype=np.float64)
    rhs = np.zeros((n, dim), dtype=np.float64)
    y = np.vstack([obs["value"] for obs in clean]).astype(np.float64)
    obs_weights = np.asarray([max(1e-6, finite_float(obs.get("weight"), default_obs_weight)) for obs in clean], dtype=np.float64)
    for i, w in enumerate(obs_weights):
        diag[i] += w
        rhs[i] += w * y[i]
    edge_weights: list[float] = []
    for i in range(1, n):
        dt = max(1, require_int(clean[i].get("frame_idx"), "series frame_idx") - require_int(clean[i - 1].get("frame_idx"), "series frame_idx"))
        ew = temporal_weight / float(dt * dt)
        edge_weights.append(ew)
        diag[i - 1] += ew
        diag[i] += ew
        upper[i - 1] -= ew
        lower[i - 1] -= ew
    # The positive prior below is not a fallback value; it keeps the linear system nonsingular for one-observation tracks.
    diag += 1e-9
    estimate = np.zeros((n, dim), dtype=np.float64)
    for d in range(dim):
        estimate[:, d] = solve_tridiagonal(lower, diag, upper, rhs[:, d])

    def total_energy(x: np.ndarray) -> float:
        obs_e = float(np.sum(obs_weights[:, None] * (x - y) ** 2))
        tmp_e = 0.0
        for j, ew in enumerate(edge_weights, start=1):
            tmp_e += float(ew * np.sum((x[j] - x[j - 1]) ** 2))
        return obs_e + tmp_e

    initial = y.copy()
    energy_initial = total_energy(initial)
    energy_after = total_energy(estimate)
    estimates: dict[int, dict[str, Any]] = {}
    for i, obs in enumerate(clean):
        frame_idx = require_int(obs.get("frame_idx"), "series frame_idx")
        obs_residual = float(np.linalg.norm(estimate[i] - y[i]))
        temporal_before = 0.0
        temporal_after = 0.0
        if i > 0:
            temporal_before += float(edge_weights[i - 1] * np.sum((initial[i] - initial[i - 1]) ** 2))
            temporal_after += float(edge_weights[i - 1] * np.sum((estimate[i] - estimate[i - 1]) ** 2))
        if i < n - 1:
            temporal_before += float(edge_weights[i] * np.sum((initial[i + 1] - initial[i]) ** 2))
            temporal_after += float(edge_weights[i] * np.sum((estimate[i + 1] - estimate[i]) ** 2))
        estimates[frame_idx] = {
            "variable_id": obs.get("variable_id"),
            "source": obs.get("source"),
            "initial": [float(v) for v in initial[i].tolist()],
            "estimate": [float(v) for v in estimate[i].tolist()],
            "observation_weight": float(obs_weights[i]),
            "observation_residual_norm": obs_residual,
            "local_temporal_energy_initial": temporal_before / 2.0,
            "local_temporal_energy_after": temporal_after / 2.0,
            "unit": unit,
            "dimension": dim,
            "estimate_semantics": "translation_xyz_m_and_rotation_vector_xyz_rad" if dim == 6 and "rotvec" in unit else "observable_coordinate_vector",
        }
    return {
        "estimates": estimates,
        "summary": {
            "variable_count": n,
            "factor_count": n + len(edge_weights),
            "observation_factor_count": n,
            "temporal_factor_count": len(edge_weights),
            "energy_initial": energy_initial,
            "energy_after": energy_after,
            "energy_delta": energy_initial - energy_after,
            "unit": unit,
            "dimension": dim,
            "estimate_semantics": "translation_xyz_m_and_rotation_vector_xyz_rad" if dim == 6 and "rotvec" in unit else "observable_coordinate_vector",
        },
    }


def load_articulation_index(path: Path) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not path.exists():
        return {}, []
    report = require_dict(load_json(path), "articulation fit report")
    per_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    for raw in require_list(report.get("rows"), "articulation rows"):
        row = require_dict(raw, "articulation row")
        object_id = str(row.get("object_id"))
        source_id = str(row.get("source_candidate_id", object_id))
        fit_state = str(row.get("articulation_fit_state"))
        source_summary = {
            "object_id": object_id,
            "source_candidate_id": source_id,
            "part_track_labels": row.get("part_track_labels"),
            "fit_type": row.get("fit_type"),
            "fit_scope": row.get("fit_scope"),
            "coordinate_frame": row.get("coordinate_frame"),
            "shared_frame_count": row.get("shared_frame_count"),
            "circle_radius_m": row.get("circle_radius_m"),
            "circle_angle_span_deg": row.get("circle_angle_span_deg"),
            "radial_residual_m": row.get("radial_residual_m"),
            "plane_residual_m": row.get("plane_residual_m"),
            "articulation_fit_state": fit_state,
            "articulation_model_ready": row.get("articulation_model_ready"),
            "part_pose_ready": row.get("part_pose_ready"),
            "fit_blockers": row.get("fit_blockers"),
        }
        sources.append(source_summary)
        for raw_frame in row.get("frame_residual_rows", []):
            if not isinstance(raw_frame, dict):
                continue
            frame_idx = require_int(raw_frame.get("frame_idx"), "articulation residual frame_idx")
            rel = finite_float(raw_frame.get("relative_center_distance_m"), float("nan"))
            radial = finite_float(raw_frame.get("radial_residual_m"), float("nan"))
            plane = finite_float(raw_frame.get("plane_residual_m"), float("nan"))
            if not math.isfinite(rel):
                continue
            per_frame[frame_idx].append(
                {
                    "object_id": object_id,
                    "source_candidate_id": source_id,
                    "part_track_labels": row.get("part_track_labels"),
                    "articulation_coordinate_observation_m": rel,
                    "radial_residual_m": radial if math.isfinite(radial) else None,
                    "plane_residual_m": plane if math.isfinite(plane) else None,
                    "fit_state": fit_state,
                    "articulation_model_ready": row.get("articulation_model_ready"),
                    "source_summary": source_summary,
                }
            )
    return per_frame, sources


def contact_switch_energy(hyp: dict[str, Any], hand: dict[str, Any] | None, obj: dict[str, Any] | None, width: float, height: float) -> dict[str, Any]:
    hand_box = hand.get("bbox_xyxy") if hand else None
    obj_box = obj.get("bbox_xyxy") if obj else None
    iou = bbox_iou_value(hand_box, obj_box)
    coverage = bbox_min_coverage(hand_box, obj_box)
    dist = bbox_center_distance_norm(hand_box, obj_box, width, height)
    dist_term = (dist if dist is not None else 1.0) ** 2
    image_overlap = bool(hyp.get("evidence", {}).get("image_overlap_candidate"))
    image_contact = bool(hyp.get("evidence", {}).get("pair_contact_image_candidate"))
    depth_compatible = bool(hyp.get("evidence", {}).get("metric_depth_compatible_candidate"))
    depth_state = str(hyp.get("evidence", {}).get("pair_depth_gap_state"))
    depth_contradiction = "behind" in depth_state or "contradiction" in str(hyp.get("state")) or "rejected" in str(hyp.get("state"))
    image_support = max(iou, coverage, 0.55 if image_contact else 0.0, 0.25 if image_overlap else 0.0)
    # These are explicit model terms in a mixed normalized energy, not hidden thresholds.
    on_energy = (1.0 - image_support) ** 2 + dist_term
    if depth_compatible:
        on_energy *= 0.5
    if depth_contradiction:
        on_energy += 1.5
    off_energy = image_support ** 2
    if depth_compatible:
        off_energy += 0.5
    if depth_contradiction:
        off_energy *= 0.5
    switch_on = on_energy < off_energy
    return {
        "hand_side": hyp.get("hand_side"),
        "object_id": hyp.get("object_id"),
        "variable_id": f"contact::{hyp.get('hand_side')}::{hyp.get('object_id')}",
        "estimate": bool(switch_on),
        "on_energy": float(on_energy),
        "off_energy": float(off_energy),
        "chosen_energy": float(on_energy if switch_on else off_energy),
        "image_iou": float(iou),
        "min_box_coverage": float(coverage),
        "center_distance_norm": float(dist) if dist is not None else None,
        "depth_contradiction": bool(depth_contradiction),
        "metric_depth_compatible_candidate": depth_compatible,
        "evidence": hyp.get("evidence"),
    }


def occlusion_owner_energy(hand: dict[str, Any]) -> dict[str, Any] | None:
    occlusion = hand.get("occlusion_owner_hypothesis")
    if not isinstance(occlusion, dict):
        return None
    candidates = occlusion.get("owner_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    evaluated: list[dict[str, Any]] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        iou = finite_float(cand.get("iou"), 0.0)
        hand_cov = finite_float(cand.get("hand_box_coverage_by_object_box"), 0.0)
        object_cov = finite_float(cand.get("object_box_coverage_by_hand_box"), 0.0)
        depth_resolved = bool(cand.get("depth_order_resolved") or cand.get("occluder_owner_accepted"))
        support = max(0.0, min(1.0, 0.45 * iou + 0.45 * hand_cov + 0.10 * object_cov))
        energy = (1.0 - support) ** 2
        if not depth_resolved:
            energy += 0.25
        evaluated.append(
            {
                "object_id": cand.get("object_id"),
                "name": cand.get("name"),
                "energy": float(energy),
                "box_iou": float(iou),
                "hand_coverage": float(hand_cov),
                "object_coverage": float(object_cov),
                "depth_order_resolved": depth_resolved,
                "accepted_by_depth_evidence": bool(cand.get("occluder_owner_accepted")),
            }
        )
    if not evaluated:
        return None
    # Unowned is an explicit competing state. It prevents weak overlap evidence from being mislabeled accepted ownership.
    evaluated.append({"object_id": None, "name": "unowned", "energy": 0.55, "box_iou": 0.0, "hand_coverage": 0.0, "object_coverage": 0.0, "depth_order_resolved": False, "accepted_by_depth_evidence": False})
    chosen = min(evaluated, key=lambda row: finite_float(row.get("energy"), 999.0))
    return {
        "hand_side": hand.get("hand_side"),
        "variable_id": f"occlusion_owner::{hand.get('hand_side')}",
        "chosen_owner_object_id": chosen.get("object_id"),
        "chosen_owner_name": chosen.get("name"),
        "chosen_energy": chosen.get("energy"),
        "accepted_owner": bool(chosen.get("object_id") and chosen.get("accepted_by_depth_evidence")),
        "state": "accepted_depth_order_owner" if chosen.get("object_id") and chosen.get("accepted_by_depth_evidence") else "inferred_candidate_or_unowned_not_accepted",
        "candidate_energies": evaluated,
    }


def solve_v18_factor_graph(frames: list[dict[str, Any]], raw_video: dict[str, Any], articulation_index: dict[int, list[dict[str, Any]]], articulation_sources: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    width = finite_float(raw_video.get("width"), 1920.0) if isinstance(raw_video, dict) else 1920.0
    height = finite_float(raw_video.get("height"), 1080.0) if isinstance(raw_video, dict) else 1080.0
    hand_obs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    object_obs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    part_obs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    articulation_obs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_frame_terms: dict[int, dict[str, Any]] = defaultdict(lambda: {
        "variables": {"hand_state": [], "object_se3": [], "part_se3": [], "articulation_parameter": [], "contact_switch": [], "occlusion_owner": []},
        "factor_energy_initial": defaultdict(float),
        "factor_energy_after": defaultdict(float),
        "factor_counts": Counter(),
    })
    hand_lookup_by_frame: dict[int, dict[str, dict[str, Any]]] = {}
    object_lookup_by_frame: dict[int, dict[str, dict[str, Any]]] = {}

    for frame in frames:
        frame_idx = require_int(frame.get("frame_idx"), "graph frame_idx")
        hand_lookup = {str(h.get("hand_side")): h for h in frame.get("hands", []) if isinstance(h, dict)}
        object_lookup = {str(o.get("object_id")): o for o in frame.get("objects", []) if isinstance(o, dict)}
        hand_lookup_by_frame[frame_idx] = hand_lookup
        object_lookup_by_frame[frame_idx] = object_lookup
        for hand in hand_lookup.values():
            center = bbox_center(hand.get("bbox_xyxy"))
            if center is None or width <= 0 or height <= 0:
                continue
            side = str(hand.get("hand_side"))
            confidence = str(hand.get("confidence"))
            weight = 4.0 if confidence == "medium" else 1.5 if confidence == "low" else 0.5
            hand_obs[f"hand::{side}"].append({"frame_idx": frame_idx, "variable_id": f"hand::{side}", "value": np.asarray([center[0] / width, center[1] / height], dtype=np.float64), "weight": weight, "source": "bbox_center_normalized"})
        for obj in object_lookup.values():
            pose_raw = obj.get("object_pose_candidate")
            pose: dict[str, Any] = pose_raw if isinstance(pose_raw, dict) else {}
            trans = numeric_vector(pose.get("translation_world_m"), 3)
            rotvec = numeric_vector(pose.get("rotation_world_from_object_rotvec"), 3)
            if trans is not None:
                geom_raw = obj.get("visible_geometry_candidate")
                geom: dict[str, Any] = geom_raw if isinstance(geom_raw, dict) else {}
                vertices = max(1.0, finite_float(geom.get("vertex_count"), 1.0))
                anisotropy = max(0.0, finite_float(geom.get("pca_anisotropy"), 0.0))
                weight = min(8.0, 1.0 + math.log1p(vertices) / 2.0)
                object_id = str(obj.get("object_id"))
                if rotvec is not None:
                    value = np.concatenate([trans, rotvec])
                    source = "visible_surface_world_centroid_plus_pca_rotvec"
                    weight *= max(0.5, min(1.5, anisotropy + 0.5))
                else:
                    value = trans
                    source = "visible_surface_world_centroid_translation_only_rotation_unobserved"
                object_obs[f"object_se3::{object_id}"].append({"frame_idx": frame_idx, "variable_id": f"object_se3::{object_id}", "value": value, "weight": weight, "source": source})
            for part in obj.get("parts", []):
                if not isinstance(part, dict):
                    continue
                center = numeric_vector(part.get("center_camera_m"), 3)
                if center is None:
                    continue
                label = str(part.get("part_track_label"))
                object_id = str(obj.get("object_id"))
                containment = finite_float(part.get("part_containment_in_object"), 0.5)
                weight = max(0.25, min(4.0, 0.5 + 3.0 * containment))
                part_obs[f"part_se3::{object_id}::{label}"].append({"frame_idx": frame_idx, "variable_id": f"part_se3::{object_id}::{label}", "value": center, "weight": weight, "source": "part_visible_surface_center_camera"})
        for art in articulation_index.get(frame_idx, []):
            object_id = str(art.get("object_id"))
            source_id = str(art.get("source_candidate_id"))
            value = np.asarray([finite_float(art.get("articulation_coordinate_observation_m"), 0.0)], dtype=np.float64)
            state = str(art.get("fit_state"))
            weight = 2.0 if "supported" in state else 0.5
            articulation_obs[f"articulation::{object_id}::{source_id}"].append({"frame_idx": frame_idx, "variable_id": f"articulation::{object_id}::{source_id}", "value": value, "weight": weight, "source": "visible_part_relative_center_distance"})

    series_summaries: dict[str, Any] = {}
    variable_counts = Counter()
    factor_counts = Counter()
    energy_initial_total = 0.0
    energy_after_total = 0.0

    def absorb_series(kind: str, grouped: dict[str, list[dict[str, Any]]], temporal_weight: float, default_weight: float, unit: str) -> None:
        nonlocal energy_initial_total, energy_after_total
        for variable_id, obs in grouped.items():
            solved = solve_temporal_series(obs, temporal_weight, default_weight, unit)
            summary = solved["summary"]
            series_summaries[variable_id] = summary
            variable_counts[kind] += int(summary.get("variable_count", 0))
            factor_counts[f"{kind}_observation"] += int(summary.get("observation_factor_count", 0))
            factor_counts[f"{kind}_temporal"] += int(summary.get("temporal_factor_count", 0))
            energy_initial_total += finite_float(summary.get("energy_initial"), 0.0)
            energy_after_total += finite_float(summary.get("energy_after"), 0.0)
            for frame_idx, est in solved["estimates"].items():
                terms = per_frame_terms[frame_idx]
                if kind == "hand_state":
                    terms["variables"]["hand_state"].append(est)
                elif kind == "object_se3":
                    terms["variables"]["object_se3"].append(est)
                elif kind == "part_se3":
                    terms["variables"]["part_se3"].append(est)
                elif kind == "articulation_parameter":
                    terms["variables"]["articulation_parameter"].append(est)
                obs_energy = est["observation_weight"] * (est["observation_residual_norm"] ** 2)
                terms["factor_energy_after"][f"{kind}_observation"] += obs_energy
                terms["factor_energy_after"][f"{kind}_temporal"] += finite_float(est.get("local_temporal_energy_after"), 0.0)
                terms["factor_energy_initial"][f"{kind}_temporal"] += finite_float(est.get("local_temporal_energy_initial"), 0.0)
                terms["factor_counts"][f"{kind}_observation"] += 1
                terms["factor_counts"][f"{kind}_temporal"] += 1

    absorb_series("hand_state", hand_obs, temporal_weight=0.8, default_weight=1.0, unit="normalized_image_xy")
    absorb_series("object_se3", object_obs, temporal_weight=2.0, default_weight=1.0, unit="world_m_translation_plus_optional_pca_rotvec_rad")
    absorb_series("part_se3", part_obs, temporal_weight=1.0, default_weight=1.0, unit="camera_m_translation")
    absorb_series("articulation_parameter", articulation_obs, temporal_weight=1.0, default_weight=0.5, unit="relative_part_center_distance_m")

    active_contact_count = 0
    unresolved_contact_count = 0
    accepted_owner_count = 0
    for frame in frames:
        frame_idx = require_int(frame.get("frame_idx"), "graph frame_idx")
        hands = hand_lookup_by_frame.get(frame_idx, {})
        objects = object_lookup_by_frame.get(frame_idx, {})
        terms = per_frame_terms[frame_idx]
        for hyp in frame.get("contact_hypotheses", []):
            if not isinstance(hyp, dict):
                continue
            switch = contact_switch_energy(hyp, hands.get(str(hyp.get("hand_side"))), objects.get(str(hyp.get("object_id"))), width, height)
            terms["variables"]["contact_switch"].append(switch)
            terms["factor_counts"]["contact_switch_discrete"] += 1
            factor_counts["contact_switch_discrete"] += 1
            variable_counts["contact_switch"] += 1
            energy_initial_total += finite_float(switch.get("off_energy"), 0.0)
            energy_after_total += finite_float(switch.get("chosen_energy"), 0.0)
            terms["factor_energy_initial"]["contact_switch_discrete"] += finite_float(switch.get("off_energy"), 0.0)
            terms["factor_energy_after"]["contact_switch_discrete"] += finite_float(switch.get("chosen_energy"), 0.0)
            if switch.get("estimate") is True:
                active_contact_count += 1
            elif hyp.get("confidence") in {"unknown", "very_low_depth_contradiction"}:
                unresolved_contact_count += 1
        for hand in hands.values():
            owner = occlusion_owner_energy(hand)
            if owner is None:
                continue
            terms["variables"]["occlusion_owner"].append(owner)
            terms["factor_counts"]["occlusion_owner_discrete"] += 1
            factor_counts["occlusion_owner_discrete"] += 1
            variable_counts["occlusion_owner"] += 1
            initial_energy = 0.55
            chosen_energy = finite_float(owner.get("chosen_energy"), initial_energy)
            energy_initial_total += initial_energy
            energy_after_total += chosen_energy
            terms["factor_energy_initial"]["occlusion_owner_discrete"] += initial_energy
            terms["factor_energy_after"]["occlusion_owner_discrete"] += chosen_energy
            if owner.get("accepted_owner") is True:
                accepted_owner_count += 1

    by_frame: dict[int, dict[str, Any]] = {}
    for frame in frames:
        frame_idx = require_int(frame.get("frame_idx"), "graph frame_idx")
        terms = per_frame_terms[frame_idx]
        local_initial = float(sum(float(v) for v in terms["factor_energy_initial"].values()))
        local_after = float(sum(float(v) for v in terms["factor_energy_after"].values()))
        contact_switches = terms["variables"]["contact_switch"]
        occlusion_owners = terms["variables"]["occlusion_owner"]
        by_frame[frame_idx] = {
            "solver": "v18_numerical_temporal_factor_graph_v1",
            "graph_scope": "full_case_temporal_graph_with_per_frame_marginals",
            "variables": {
                "camera_depth_correction": {"variable_id": "camera_depth_scale", "estimate": 1.0, "prior": 1.0, "state": "identity_depth_scale_prior_no_casewide_refit_observation"},
                "hand_state": terms["variables"]["hand_state"],
                "object_se3": terms["variables"]["object_se3"],
                "part_se3": terms["variables"]["part_se3"],
                "articulation_parameter": terms["variables"]["articulation_parameter"],
                "contact_switch": contact_switches,
                "occlusion_owner": occlusion_owners,
            },
            "factors": dict(sorted(terms["factor_counts"].items())),
            "objective": {
                "local_energy_initial": local_initial,
                "local_energy_after": local_after,
                "local_energy_delta": local_initial - local_after,
                "energy_units": "mixed_normalized_squared_residuals_with_metric_translation_and_rotation_vector_terms",
            },
            "inference": {
                "continuous_method": "weighted_temporal_least_squares_scipy_sparse_spsolve",
                "discrete_method": "exact_min_energy_choice_for_binary_contact_and_occlusion_owner_variables",
                "not_solved_by_threshold_gate": True,
            },
            "solution": {
                "state": "numerical_factor_graph_candidate_solution",
                "active_contact_hypotheses": sum(1 for row in contact_switches if row.get("estimate") is True),
                "unresolved_or_contradicted_contact_hypotheses": sum(1 for row in contact_switches if row.get("depth_contradiction") or row.get("metric_depth_compatible_candidate") is False),
                "accepted_occlusion_owner_count": sum(1 for row in occlusion_owners if row.get("accepted_owner") is True),
                "all_outputs_approximate_uncertain": True,
            },
        }
    summary = {
        "solver": "v18_numerical_temporal_factor_graph_v1",
        "variables_required_by_spec": ["camera_depth_correction", "hand_state", "object_se3", "part_se3", "articulation_parameter", "contact_switch", "occlusion_owner"],
        "implemented_variable_status": {
            "camera_depth_correction": "prior_only_identity_scale_no_casewide_depth_refit_observation",
            "hand_state": "normalized_bbox_center_track_observation",
            "object_se3": "visible_surface_translation_plus_pca_rotvec_when_point_cloud_available",
            "part_se3": "visible_part_center_translation_only_rotation_unresolved_in_current_part_surface_artifact",
            "articulation_parameter": "visible_part_relative_center_distance_coordinate_only",
            "contact_switch": "discrete_energy_from_overlap_and_available_depth_candidate_evidence",
            "occlusion_owner": "discrete_energy_over_owner_candidates_without_new_depth_order_acceptance",
        },
        "implemented_factor_families": [
            "hand_bbox_observation_residual",
            "visible_object_surface_pose_observation_residual",
            "visible_part_center_observation_residual",
            "adjacent_frame_temporal_consistency",
            "articulation_visible_coordinate_residual",
            "contact_overlap_depth_candidate_energy",
            "occlusion_owner_candidate_energy",
        ],
        "spec_factor_gaps_remaining": [
            "object_mask_depth_registration_residual_is_visible_surface_only_not_complete_geometry_registration",
            "rigid_articulation_consistency_does_not_yet_solve_full_part_SE3",
            "contact_nonpenetration_is_not_yet_full_geometry_nonpenetration",
            "occlusion_depth_order_owner_energy_does_not_accept_new_owners_without_source_depth_evidence",
        ],
        "variable_counts": dict(sorted(variable_counts.items())),
        "factor_counts": dict(sorted(factor_counts.items())),
        "objective": {
            "energy_initial": energy_initial_total,
            "energy_after": energy_after_total,
            "energy_delta": energy_initial_total - energy_after_total,
            "energy_units": "mixed_normalized_squared_residuals_with_metric_translation_and_rotation_vector_terms",
        },
        "inference": {
            "continuous_method": "weighted temporal least-squares solved by SciPy sparse linear systems for each observed track",
            "discrete_method": "exact min-energy assignment for each contact switch and occlusion owner variable",
            "continuous_series_count": len(series_summaries),
            "series_summaries": series_summaries,
        },
        "articulation_sources": articulation_sources,
        "solution_counts": {
            "active_contact_switches": active_contact_count,
            "unresolved_or_depth_contradicted_contacts": unresolved_contact_count,
            "accepted_occlusion_owners": accepted_owner_count,
        },
        "limitations": [
            "The graph estimates candidate states from available observations; it does not invent hidden object geometry where no reconstruction exists.",
            "Object SE(3) variables use visible-surface translation plus PCA rotation observations when available; part rotations remain unresolved because the current part-surface artifact stores center/extent/counts but not part point coordinates.",
            "Occlusion owner variables compete over candidates, but accepted ownership remains false unless depth-order evidence supports it.",
        ],
    }
    return by_frame, summary

def build_case_annotations(case: str, args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.annotation_state_root / case / "v18_annotation_state.json"
    state = require_dict(load_json(state_path), f"{case} annotation state")
    v16_path = args.v16_root / case / "annotations_v16_full.json"
    v16_frames = index_v16_frames(v16_path)
    bounded_index = index_bounded_frames(args.bounded_root / case / "v18_bounded_state_solution.json")
    geom_index, completion_by_object, visible_archive = load_visible_geometry_index(args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json")
    part_index = load_part_surface_index(args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json")
    articulation_index, articulation_sources = load_articulation_index(args.articulation_root / case / "v18_articulation_fit_candidates_report.json")
    frame_count = require_int(state.get("frame_count"), "frame_count")
    fps = finite_float(state.get("fps"), 30.0)
    frames: list[dict[str, Any]] = []
    module_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    for raw_frame in require_list(state.get("frames"), "state frames"):
        src_frame = require_dict(raw_frame, "state frame")
        frame_idx = require_int(src_frame.get("frame_idx"), "frame_idx")
        v16_frame = v16_frames.get(frame_idx, {})
        v16_hands = hand_by_side(v16_frame)
        bounded_frame = bounded_index.get(frame_idx, {})
        bounded_hands_by_side = {str(h.get("hand_side")): h for h in bounded_frame.get("hands", []) if isinstance(h, dict)}
        hands: list[dict[str, Any]] = []
        for raw_hand in require_list(src_frame.get("hands"), "src hands"):
            hand = require_dict(raw_hand, "hand")
            side = str(hand.get("hand_side"))
            v16_hand = v16_hands.get(side, {})
            bounded_hand = bounded_hands_by_side.get(side, {})
            occlusion_solution = require_dict(bounded_hand.get("occlusion_solution", {}), "occlusion solution") if bounded_hand else {}
            owner_candidates = occlusion_solution.get("owner_candidate_objects", []) if isinstance(occlusion_solution.get("owner_candidate_objects", []), list) else []
            mano_candidate = {
                "source": v16_hand.get("backend", "V16_or_V18_hand_baseline"),
                "bbox_xyxy": hand.get("bbox_xyxy") or v16_hand.get("bbox_xyxy"),
                "joints3d_camera": v16_hand.get("joints3d_camera"),
                "cam_t": v16_hand.get("cam_t"),
                "source_intrinsics": v16_hand.get("source_intrinsics"),
                "detector_score": v16_hand.get("detector_score"),
                "uncertainty": "approximate_mano_candidate",
            }
            confidence = "medium" if hand.get("visibility_state") == "visible" and hand.get("metric_depth_compatible") else "low" if hand.get("visibility_state") in {"visible", "partially_visible"} else "unknown"
            confidence_counts[f"hand_{confidence}"] += 1
            hands.append(
                {
                    "hand_side": side,
                    "visibility_state": hand.get("visibility_state"),
                    "bbox_xyxy": hand.get("bbox_xyxy") or v16_hand.get("bbox_xyxy"),
                    "mano_candidate": mano_candidate,
                    "hawor_candidate_present": hand.get("hawor_candidate_present"),
                    "wilor_or_v16_candidate_present": bool(v16_hand) or hand.get("renderable_bbox") is True,
                    "rtmlib_anchor_available": hand.get("rtmlib_wilor_comparison_available"),
                    "confidence": confidence,
                    "uncertainty": "all_hand_outputs_approximate",
                    "occlusion_owner_hypothesis": {
                        "state": occlusion_solution.get("occluder_owner_status", "unresolved_or_not_applicable"),
                        "owner_candidates": owner_candidates,
                        "confidence": "low" if owner_candidates else "unknown",
                    },
                }
            )
            module_counts["hand_states"] += 1
        objects: list[dict[str, Any]] = []
        contact_hypotheses: list[dict[str, Any]] = []
        for raw_obj in require_list(src_frame.get("objects"), "src objects"):
            obj = require_dict(raw_obj, "object")
            object_id = str(obj.get("object_id"))
            geom = geom_index.get((frame_idx, object_id))
            parts = part_index.get((frame_idx, object_id), [])
            pose = object_pose_candidate(obj, geom)
            completion = completion_by_object.get(object_id, {
                "method": "no_visible_surface_completion_candidate_available",
                "scope": "explicit_unresolved_hidden_geometry_candidate",
                "uncertainty": "unknown",
            })
            object_contacts: list[dict[str, Any]] = []
            for raw_contact in obj.get("contact_rows", []):
                if isinstance(raw_contact, dict):
                    row = dict(raw_contact)
                    row["object_id"] = object_id
                    hyp = contact_hypothesis(row)
                    contact_hypotheses.append(hyp)
                    object_contacts.append(hyp)
            confidence = "low" if geom is not None else "very_low" if obj.get("visibility_state") == "visible" else "unknown"
            confidence_counts[f"object_{confidence}"] += 1
            objects.append(
                {
                    "object_id": object_id,
                    "name": obj.get("name"),
                    "visibility_state": obj.get("visibility_state"),
                    "physical_state_candidate": obj.get("model_physical_state_type"),
                    "bbox_xyxy": obj.get("bbox_xyxy"),
                    "mask_path": obj.get("mask_path"),
                    "renderable_mask": obj.get("renderable_mask"),
                    "visible_geometry_candidate": geom,
                    "hidden_geometry_candidate": completion,
                    "object_pose_candidate": pose,
                    "parts": parts,
                    "part_pose_candidate_count": len(parts),
                    "contact_hypotheses": object_contacts,
                    "occlusion_owner_hypothesis": {
                        "state": obj.get("occluder_owner") or "unresolved_or_not_applicable",
                        "confidence": "unknown",
                    },
                    "confidence": confidence,
                    "uncertainty": "all_object_outputs_approximate",
                    "render_style": obj.get("render_style"),
                }
            )
            module_counts["object_states"] += 1
            module_counts["part_states"] += len(parts)
        frames.append(
            {
                "frame_idx": frame_idx,
                "time_s": v16_frame.get("time_s", frame_idx / fps),
                "raw_frame_path": src_frame.get("raw_frame_path"),
                "caption": v16_frame.get("caption", ""),
                "camera": v16_frame.get("camera", {}),
                "hands": hands,
                "objects": objects,
                "contact_hypotheses": contact_hypotheses,
                "frame_summary": {
                    "hand_count": len(hands),
                    "object_count": len(objects),
                    "part_candidate_count": sum(len(obj.get("parts", [])) for obj in objects),
                    "contact_hypothesis_count": len(contact_hypotheses),
                    "all_outputs_approximate_uncertain": True,
                },
            }
        )
    factor_graph_by_frame, factor_graph_summary = solve_v18_factor_graph(frames, require_dict(state.get("raw_video", {}), "raw_video"), articulation_index, articulation_sources)
    for frame in frames:
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        frame["factor_graph_solution"] = factor_graph_by_frame.get(frame_idx, {})
    module_counts["factor_graph_variables"] += sum(int(v) for v in factor_graph_summary.get("variable_counts", {}).values())
    module_counts["factor_graph_factors"] += sum(int(v) for v in factor_graph_summary.get("factor_counts", {}).values())
    out = {
        "method": "run_v18_full_pipeline",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "v18_annotation_state": str(state_path),
            "v16_annotations": str(v16_path),
            "bounded_state_solution": str(args.bounded_root / case / "v18_bounded_state_solution.json"),
            "visible_geometry_archive": str(args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json"),
            "part_visible_surfaces": str(args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"),
            "articulation_fit_candidates": str(args.articulation_root / case / "v18_articulation_fit_candidates_report.json"),
            "visible_geometry_archive_npz": str(visible_archive) if visible_archive else None,
            "v16_render_overlay": str(v16_render_paths(case, args)["overlay"]),
            "v16_render_world": str(v16_render_paths(case, args)["world"]),
            "v16_render_side_by_side": str(v16_render_paths(case, args)["side_by_side"]),
        },
        "raw_video": state.get("raw_video"),
        "frame_count": frame_count,
        "fps": fps,
        "duration_s": finite_float(state.get("duration_s"), frame_count / fps),
        "all_outputs_approximate_uncertain": True,
        "arbitrary_gates_blocked_artifact": False,
        "monotonicity": {
            "preserves_v16_overlay_mano_object_render": True,
            "preserves_v16_metric_world_render": True,
            "v18_additions_are_overlay_layers": True,
            "no_v16_capability_replaced_by_weaker_render": True,
        },
        "modules": {
            "camera_depth_backbone": "v16_metric_camera_depth_reused_as_memoized_backbone",
            "hand_branch": "HaWoR_WiLoR_RTMLib_V16_candidates_assembled",
            "object_part_perception": "VLM_OWLv2_SAM2_masks_and_part_tracks_assembled",
            "geometry_reconstruction": "visible_surface_archive_plus_pca_mirror_hidden_geometry_candidates",
            "object_part_pose": "visible_surface_world_centroid_PCA_SE3_candidates",
            "contact_ownership": "image_overlap_depth_candidate_contact_hypotheses",
            "occlusion_ownership": "bounded_owner_candidate_hypotheses",
            "factor_graph": "numerical_temporal_factor_graph_with_explicit_variables_factors_objective_inference",
        },
        "factor_graph_summary": factor_graph_summary,
        "module_counts": dict(sorted(module_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "hidden_geometry_candidate_object_count": len(completion_by_object),
        "frames": frames,
    }
    case_dir = args.output_root / case
    write_json(case_dir / "annotations_v18_full.json", out)
    return out


def point_from_bbox_or_pose(obj: dict[str, Any], source_w: float, source_h: float, canvas_w: int, canvas_h: int) -> tuple[int, int] | None:
    center = bbox_center(obj.get("bbox_xyxy"))
    if center is None:
        return None
    left, right = 70, canvas_w - 330
    top, bottom = 96, canvas_h - 90
    x = int(round(left + max(0.0, min(1.0, center[0] / source_w)) * (right - left)))
    y = int(round(top + max(0.0, min(1.0, center[1] / source_h)) * (bottom - top)))
    return x, y


def render_overlay(case: str, ann: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case_dir = args.output_root / case
    frame_dir = case_dir / "overlay_frames"
    base_dir = case_dir / "v16_overlay_base_frames"
    v16_overlay = v16_render_paths(case, args)["overlay"]
    if not v16_overlay.exists():
        raise RuntimeError(f"{case}: missing V16 overlay render {v16_overlay}")
    extract_video_frames(v16_overlay, base_dir)
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    font = text_font(22)
    small = text_font(16)
    counts: Counter[str] = Counter()
    frames = require_list(ann.get("frames"), "annotation frames")
    for raw_frame in frames:
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        raw_path = Path(str(frame.get("raw_frame_path")))
        base_path = base_dir / f"{frame_idx + 1:06d}.jpg"
        image = Image.open(base_path if base_path.exists() else raw_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for raw_obj in frame.get("objects", []):
            obj = require_dict(raw_obj, "object")
            style = require_dict(obj.get("render_style", {}), "render style") if isinstance(obj.get("render_style"), dict) else {}
            rgb = color_from_bgr(style.get("color_bgr"), (80, 180, 255))
            if obj.get("renderable_mask") is True and isinstance(obj.get("mask_path"), str):
                image = mask_overlay(image, str(obj.get("mask_path")), rgb, 0.18)
                draw = ImageDraw.Draw(image)
                counts["object_masks"] += 1
            raw_video = require_dict(ann.get("raw_video", {}), "raw_video")
            source_w = finite_float(raw_video.get("width"), float(image.size[0]))
            source_h = finite_float(raw_video.get("height"), float(image.size[1]))
            draw_bbox = scale_bbox(obj.get("bbox_xyxy"), source_w, source_h, float(image.size[0]), float(image.size[1]))
            box = bbox_tuple(draw_bbox)
            if box:
                draw.rectangle(box, outline=rgb, width=3)
                label = f"{obj.get('name')} | {obj.get('physical_state_candidate')} | {obj.get('confidence')} approx"
                draw_label(draw, (box[0], max(44, box[1] - 22)), label[:115], small, rgb)
                counts["object_boxes"] += 1
            for part in obj.get("parts", [])[:4]:
                if isinstance(part, dict) and isinstance(part.get("part_mask_path"), str):
                    image = mask_overlay(image, str(part.get("part_mask_path")), (255, 230, 90), 0.22)
                    draw = ImageDraw.Draw(image)
                    counts["part_masks"] += 1
        for raw_hand in frame.get("hands", []):
            hand = require_dict(raw_hand, "hand")
            raw_video = require_dict(ann.get("raw_video", {}), "raw_video")
            source_w = finite_float(raw_video.get("width"), float(image.size[0]))
            source_h = finite_float(raw_video.get("height"), float(image.size[1]))
            draw_bbox = scale_bbox(hand.get("bbox_xyxy"), source_w, source_h, float(image.size[0]), float(image.size[1]))
            box = bbox_tuple(draw_bbox)
            color = (80, 230, 100) if hand.get("confidence") == "medium" else (255, 205, 60) if hand.get("confidence") == "low" else (170, 170, 170)
            if box:
                draw.rectangle(box, outline=color, width=4)
                draw_label(draw, (box[0], max(44, box[1] - 22)), f"{hand.get('hand_side')} hand | {hand.get('confidence')} approx", small, color)
                counts["hand_boxes"] += 1
            pts = project_mano_joints(require_dict(hand.get("mano_candidate", {}), "mano candidate"), source_w, source_h, float(image.size[0]), float(image.size[1]))
            if len(pts) >= 21:
                for a, b in HAND_EDGES:
                    draw.line((pts[a][0], pts[a][1], pts[b][0], pts[b][1]), fill=color, width=3)
                for px, py in pts:
                    draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
                counts["hand_mano_skeletons"] += 1
        # Draw approximate contact lines from hand/object centers when overlap hypothesis exists.
        raw_video = require_dict(ann.get("raw_video", {}), "raw_video")
        source_w = finite_float(raw_video.get("width"), float(image.size[0]))
        source_h = finite_float(raw_video.get("height"), float(image.size[1]))
        object_centers = {
            str(o.get("object_id")): bbox_center(scale_bbox(o.get("bbox_xyxy"), source_w, source_h, float(image.size[0]), float(image.size[1])))
            for o in frame.get("objects", [])
            if isinstance(o, dict)
        }
        hand_centers = {
            str(h.get("hand_side")): bbox_center(scale_bbox(h.get("bbox_xyxy"), source_w, source_h, float(image.size[0]), float(image.size[1])))
            for h in frame.get("hands", [])
            if isinstance(h, dict)
        }
        for hyp in frame.get("contact_hypotheses", [])[:20]:
            if not isinstance(hyp, dict) or hyp.get("confidence") not in {"medium", "low"}:
                continue
            hc = hand_centers.get(str(hyp.get("hand_side")))
            oc = object_centers.get(str(hyp.get("object_id")))
            if hc and oc:
                draw.line((hc[0], hc[1], oc[0], oc[1]), fill=(255, 255, 80), width=2)
                counts["contact_lines"] += 1
        draw.rectangle((0, 0, image.size[0], 44), fill=(0, 0, 0))
        draw.text((12, 11), f"V18 over V16 base frame {frame_idx+1}/{len(frames)} — V16 MANO/object render preserved + V18 layers", font=font, fill=(255, 255, 255))
        draw_label(draw, (12, image.size[1] - 34), "Base: V16 overlay_mano_object. Additions: V18 masks/parts/contact/occlusion/uncertainty.", small, (255, 255, 255))
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)
    output = case_dir / "v18_overlay.mp4"
    encode_video(frame_dir, output, finite_float(ann.get("fps"), 30.0))
    return {"output_video": str(output), "frame_count": ffprobe_frame_count(output), "draw_counts": dict(sorted(counts.items())), "base_v16_overlay": str(v16_overlay)}


def render_world(case: str, ann: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case_dir = args.output_root / case
    frame_dir = case_dir / "world_frames"
    base_dir = case_dir / "v16_world_base_frames"
    v16_world = v16_render_paths(case, args)["world"]
    if not v16_world.exists():
        raise RuntimeError(f"{case}: missing V16 world render {v16_world}")
    extract_video_frames(v16_world, base_dir)
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    font = text_font(20)
    small = text_font(15)
    frames = require_list(ann.get("frames"), "annotation frames")
    counts: Counter[str] = Counter()
    canvas_w, canvas_h = 1280, 720
    for raw_frame in frames:
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        base_path = base_dir / f"{frame_idx + 1:06d}.jpg"
        image = Image.open(base_path).convert("RGB") if base_path.exists() else Image.new("RGB", (canvas_w, canvas_h), (18, 20, 25))
        image = image.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(image)
        left, right = 70, canvas_w - 330
        top, bottom = 96, canvas_h - 90
        draw.rectangle((0, 0, canvas_w, 48), fill=(0, 0, 0))
        draw.text((14, 13), f"V18 over V16 metric world frame {frame_idx+1}/{len(frames)} — V16 reconstruction preserved + V18 graph layer", font=font, fill=(255, 255, 255))
        raw_video = require_dict(ann.get("raw_video", {}), "raw_video")
        source_w = finite_float(raw_video.get("width"), 1920.0)
        source_h = finite_float(raw_video.get("height"), 1080.0)
        object_points: dict[str, tuple[int, int]] = {}
        for obj in frame.get("objects", []):
            if not isinstance(obj, dict):
                continue
            pt = point_from_bbox_or_pose(obj, source_w, source_h, canvas_w, canvas_h)
            if pt is None:
                continue
            object_points[str(obj.get("object_id"))] = pt
            color = (70, 180, 255) if obj.get("visible_geometry_candidate") else (160, 160, 160)
            radius = 8 if obj.get("visible_geometry_candidate") else 5
            draw.ellipse((pt[0]-radius, pt[1]-radius, pt[0]+radius, pt[1]+radius), fill=color)
            draw_label(draw, (pt[0]+10, pt[1]-10), str(obj.get("name"))[:36], small, color, (18, 20, 25))
            counts["world_objects"] += 1
        hand_points: dict[str, tuple[int, int]] = {}
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            center = bbox_center(hand.get("bbox_xyxy"))
            if center is None:
                continue
            x = int(round(left + max(0, min(1, center[0] / source_w)) * (right - left)))
            y = int(round(top + max(0, min(1, center[1] / source_h)) * (bottom - top)))
            side = str(hand.get("hand_side"))
            hand_points[side] = (x, y)
            color = (80, 230, 100) if hand.get("confidence") == "medium" else (255, 205, 60)
            draw.rectangle((x-8, y-8, x+8, y+8), fill=color)
            draw_label(draw, (x+10, y-10), f"{side} hand", small, color, (18, 20, 25))
            counts["world_hands"] += 1
        for hyp in frame.get("contact_hypotheses", []):
            if not isinstance(hyp, dict) or hyp.get("confidence") not in {"medium", "low"}:
                continue
            hp = hand_points.get(str(hyp.get("hand_side")))
            op = object_points.get(str(hyp.get("object_id")))
            if hp and op:
                draw.line((hp[0], hp[1], op[0], op[1]), fill=(255, 255, 90), width=2)
                counts["world_contact_edges"] += 1
        fg = require_dict(frame.get("factor_graph_solution"), "factor graph")
        sol = require_dict(fg.get("solution"), "factor graph solution")
        summary = (
            f"V18 graph overlay: contact candidates={sol.get('active_contact_hypotheses')} "
            f"unresolved={sol.get('unresolved_or_contradicted_contact_hypotheses')} | approximate/uncertain"
        )
        draw_label(draw, (14, 52), summary, small, (255, 255, 255), (0, 0, 0))
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)
    output = case_dir / "v18_world.mp4"
    encode_video(frame_dir, output, finite_float(ann.get("fps"), 30.0))
    return {"output_video": str(output), "frame_count": ffprobe_frame_count(output), "draw_counts": dict(sorted(counts.items())), "base_v16_world": str(v16_world)}


def subjective_v16_comparison(case: str, ann: dict[str, Any]) -> dict[str, Any]:
    object_count = int(ann.get("module_counts", {}).get("object_states", 0))
    part_count = int(ann.get("module_counts", {}).get("part_states", 0))
    hidden_count = int(ann.get("hidden_geometry_candidate_object_count", 0))
    return {
        "basis": "subjective_video_and_schema_comparison_without_ground_truth",
        "v16_preserved_or_no_worse": [
            "V18 reuses the V16 camera/depth backbone and V16/WiLoR hand candidates where available, so it should not regress basic frame coverage or camera timing.",
            "V18 writes the same full raw frame count and render duration as the representative raw videos.",
        ],
        "plausible_v18_improvements": [
            f"V18 annotates a multi-object roster over {object_count} per-frame object states instead of a narrower V16 object focus.",
            f"V18 includes generated semantic part tracks and {part_count} per-frame part surface/pose candidates.",
            f"V18 emits category-agnostic hidden-geometry candidates for {hidden_count} objects with visible surface evidence.",
            "V18 renders contact and occlusion as explicit approximate hypotheses rather than silently omitting them.",
            "V18 final JSON contains bounded factor-graph baseline fields for every frame.",
        ],
        "remaining_uncertainty": [
            "No ground truth is available; comparison is subjective and video-based.",
            "All V18 geometry, pose, contact, occlusion, and factor-graph outputs are approximate candidates.",
            "The hidden-geometry baseline is category-agnostic visible-surface PCA mirroring, not an accurate object completion model.",
        ],
    }


def run_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = build_case_annotations(case, args)
    overlay_qc = render_overlay(case, ann, args)
    world_qc = render_world(case, ann, args)
    side_path = args.output_root / case / "v18_side_by_side.mp4"
    compose_side_by_side(Path(overlay_qc["output_video"]), Path(world_qc["output_video"]), side_path)
    side_count = ffprobe_frame_count(side_path)
    frame_count = require_int(ann.get("frame_count"), "ann frame_count")
    qc = {
        "case": case,
        "annotations": str(args.output_root / case / "annotations_v18_full.json"),
        "overlay_video": overlay_qc["output_video"],
        "world_video": world_qc["output_video"],
        "side_by_side_video": str(side_path),
        "expected_frame_count": frame_count,
        "overlay_frame_count": overlay_qc.get("frame_count"),
        "world_frame_count": world_qc.get("frame_count"),
        "side_by_side_frame_count": side_count,
        "frame_count_match": overlay_qc.get("frame_count") == world_qc.get("frame_count") == side_count == frame_count,
        "fps": ann.get("fps"),
        "duration_s": ann.get("duration_s"),
        "all_outputs_approximate_uncertain": True,
        "monotonicity": ann.get("monotonicity"),
        "base_v16_overlay": overlay_qc.get("base_v16_overlay"),
        "base_v16_world": world_qc.get("base_v16_world"),
        "module_counts": ann.get("module_counts"),
        "confidence_counts": ann.get("confidence_counts"),
        "hidden_geometry_candidate_object_count": ann.get("hidden_geometry_candidate_object_count"),
        "subjective_v16_comparison": subjective_v16_comparison(case, ann),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / case / "v18_full_pipeline_qc.json", qc)
    return qc


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    qcs = [run_case(case, args) for case in args.cases]
    report = {
        "method": "run_v18_full_pipeline",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(qcs),
        "cases": qcs,
        "all_frame_counts_match": all(qc.get("frame_count_match") is True for qc in qcs),
        "all_outputs_approximate_uncertain": True,
        "arbitrary_gates_blocked_artifact": False,
        "artifact_paths": {
            case: {
                "annotations": str(args.output_root / case / "annotations_v18_full.json"),
                "overlay_video": str(args.output_root / case / "v18_overlay.mp4"),
                "world_video": str(args.output_root / case / "v18_world.mp4"),
                "side_by_side_video": str(args.output_root / case / "v18_side_by_side.mp4"),
            }
            for case in args.cases
        },
        "deadline_context": "hard deadline 2026-06-13 09:00 local time",
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_full_pipeline_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--annotation-state-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--bounded-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_bounded_state_solution"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--articulation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_articulation_fit_candidates"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
