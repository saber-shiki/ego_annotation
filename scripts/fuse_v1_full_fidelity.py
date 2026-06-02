#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from scipy import sparse
from scipy.optimize import minimize
from tqdm import tqdm

from run_v1_wilor_colmap import HAND_EDGES, caption_for_frame, load_actions, open_video

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)

LEFT_COLOR = (0, 210, 0)
RIGHT_COLOR = (0, 135, 255)
OBJECT_COLOR = (35, 45, 235)
TIP_IDS = [4, 8, 12, 16, 20]
HAND_SPAN_TARGET_M = 0.175
DEFAULT_MANO_RIGHT = Path("third_party/WiLoR/mano_data/MANO_RIGHT.pkl")
MANO_EDGE_STRIDE = 3


def hand_vertices_field(hand: dict, suffix: str = "") -> str:
    full = f"vertices{suffix}"
    sample = f"vertices{suffix}_sample"
    if full in hand:
        return full
    if sample in hand:
        return sample
    raise RuntimeError(f"hand record has neither {full} nor {sample}")


def hand_vertices(hand: dict, suffix: str = "") -> np.ndarray:
    return np.asarray(hand[hand_vertices_field(hand, suffix)], dtype=float)


def mano_faces(path: Path) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(f"MANO topology file does not exist: {path}")
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {"bool": bool, "int": int, "float": float, "complex": complex, "object": object, "str": str, "unicode": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    with path.open("rb") as f, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Please import `csc_matrix` from the `scipy.sparse` namespace.*",
        )
        data = pickle.load(f, encoding="latin1")
    return np.asarray(data["f"], dtype=int)


def mano_edges_from_faces(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    if faces.size == 0 or vertex_count < 3:
        return np.empty((0, 2), dtype=int)
    if int(faces.max()) >= vertex_count:
        return np.empty((0, 2), dtype=int)
    pairs = set()
    for tri in faces:
        a, b, c = map(int, tri)
        if a < vertex_count and b < vertex_count and c < vertex_count:
            pairs.add(tuple(sorted((a, b))))
            pairs.add(tuple(sorted((b, c))))
            pairs.add(tuple(sorted((c, a))))
    edges = np.asarray(sorted(pairs), dtype=int)
    return edges[::MANO_EDGE_STRIDE] if len(edges) > MANO_EDGE_STRIDE else edges


@dataclass(frozen=True)
class RenderSpec:
    width: int
    height: int
    fps: float


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read video frame {frame_idx}")
    return frame


def manipulated_object_interval(actions: list[dict], label: str) -> tuple[int, int]:
    spans = [
        (int(action["start_frame"]), int(action["end_frame"]))
        for action in actions
        if label in (action.get("description", "") + " " + action.get("action", "")).lower()
    ]
    if not spans:
        raise RuntimeError(f"no semantic interval mentions object label: {label}")
    return min(s for s, _ in spans), max(e for _, e in spans)


def choose_hand_by_side(hands: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for hand in hands:
        side = hand["side"]
        cur = out.get(side)
        if cur is None or float(hand.get("detector_score", 0.0)) > float(cur.get("detector_score", 0.0)):
            out[side] = hand
    return out


def hand_vector(hand: dict) -> np.ndarray:
    vertices_key = hand_vertices_field(hand, "_camera")
    fields = [
        np.asarray(hand["bbox_xyxy"], dtype=float).reshape(-1),
        np.asarray(hand["cam_t"], dtype=float).reshape(-1),
        np.asarray(hand["joints3d_camera"], dtype=float).reshape(-1),
        np.asarray(hand[vertices_key], dtype=float).reshape(-1),
    ]
    return np.concatenate(fields)


def project_points(points_camera_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    z = np.clip(points_camera_m[:, 2], 1e-6, None)
    return np.c_[fx * points_camera_m[:, 0] / z + cx, fy * points_camera_m[:, 1] / z + cy]


def solve_source_camera_translation(local_points_m: np.ndarray, points2d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    qx = (points2d[:, 0] - cx) / fx
    qy = (points2d[:, 1] - cy) / fy
    rows = []
    rhs = []
    for (x, y, z), u, v in zip(local_points_m, qx, qy):
        rows.append([1.0, 0.0, -float(u)])
        rhs.append(float(u * z - x))
        rows.append([0.0, 1.0, -float(v)])
        rhs.append(float(v * z - y))
    trans, *_ = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float), rcond=None)
    return trans.astype(float)


def hand_metric_scale_from_raw(raw_frames: list[dict]) -> dict:
    spans = []
    for frame in raw_frames:
        for hand in frame["raw_hands"]:
            joints = np.asarray(hand["joints3d_camera"], dtype=float)
            dist = float(np.linalg.norm(joints[12] - joints[0]))
            if 0.04 < dist < 0.20:
                spans.append(dist)
    if not spans:
        raise RuntimeError("no plausible WiLoR MANO spans for source-camera metric solve")
    spans_arr = np.asarray(spans, dtype=float)
    median_span = float(np.median(spans_arr))
    scale = HAND_SPAN_TARGET_M / median_span
    residual = spans_arr * scale - HAND_SPAN_TARGET_M
    return {
        "status": "wilor_local_hand_geometry_scaled_by_wrist_to_middle_tip",
        "target_wrist_to_middle_tip_m": HAND_SPAN_TARGET_M,
        "median_wilor_wrist_to_middle_tip": median_span,
        "wilor_local_to_meters": scale,
        "sample_count": len(spans),
        "residual_iqr_m": [
            float(np.percentile(residual, 25)),
            float(np.percentile(residual, 75)),
        ],
    }


def normalize_hand_to_source_camera(hand: dict, intrinsics: np.ndarray, wilor_to_meters: float) -> dict | None:
    local_joints_m = np.asarray(hand["joints3d_camera"], dtype=float) * wilor_to_meters
    raw_vertex_key = hand_vertices_field(hand, "_camera")
    output_vertex_key = "vertices_camera" if raw_vertex_key == "vertices_camera" else "vertices_camera_sample"
    output_source_key = "vertices_source_camera_m" if output_vertex_key == "vertices_camera" else "vertices_source_camera_m_sample"
    local_verts_m = np.asarray(hand[raw_vertex_key], dtype=float) * wilor_to_meters
    raw_joints2d = np.asarray(hand["joints2d_raw"], dtype=float)
    trans_m = solve_source_camera_translation(local_joints_m, raw_joints2d, intrinsics)
    joints_camera_m = local_joints_m + trans_m
    projected = project_points(joints_camera_m, intrinsics)
    reproj = np.linalg.norm(projected - raw_joints2d, axis=1)
    depth = float(np.median(joints_camera_m[:, 2]))
    median_err = float(np.median(reproj))
    p95_err = float(np.percentile(reproj, 95))
    if not (0.2 <= depth <= 3.5 and median_err <= 45.0 and np.isfinite(joints_camera_m).all()):
        return None
    return {
        "backend": hand.get("backend", "WiLoR"),
        "side": hand["side"],
        "detector_score": float(hand.get("detector_score", 0.0)),
        "bbox_xyxy": np.asarray(hand["bbox_xyxy"], dtype=float).tolist(),
        "cam_t": trans_m.astype(float).tolist(),
        "source_intrinsics": intrinsics.astype(float).tolist(),
        "joints3d_camera": local_joints_m.astype(float).tolist(),
        "joints3d_source_camera_m": joints_camera_m.astype(float).tolist(),
        "joints2d_raw": raw_joints2d.astype(float).tolist(),
        "joints2d": projected.astype(float).tolist(),
        "mano_params": hand.get("mano_params", {}),
        output_vertex_key: local_verts_m.astype(float).tolist(),
        output_source_key: (local_verts_m + trans_m).astype(float).tolist(),
        "filter_status": "measured_source_camera_solve",
        "source_camera_solve": {
            "status": "least_squares_translation_from_mano_local_geometry_and_2d_keypoints",
            "wilor_virtual_focal_length": float(hand.get("focal_length", 0.0)),
            "wilor_virtual_cam_t": np.asarray(hand.get("cam_t", [0.0, 0.0, 0.0]), dtype=float).tolist(),
            "median_reprojection_error_px": median_err,
            "p95_reprojection_error_px": p95_err,
            "median_depth_m": depth,
        },
    }


def vector_to_hand(template: dict, vec: np.ndarray, status: str, measurement_score: float, intrinsics: np.ndarray) -> dict:
    cursor = 0

    def take(shape):
        nonlocal cursor
        n = int(np.prod(shape))
        arr = vec[cursor : cursor + n].reshape(shape)
        cursor += n
        return arr

    template_vertex_key = hand_vertices_field(template, "_camera")
    template_vertex_shape = np.asarray(template[template_vertex_key], dtype=float).shape
    output_source_key = "vertices_source_camera_m" if template_vertex_key == "vertices_camera" else "vertices_source_camera_m_sample"
    output_world_key = "vertices_world_m" if template_vertex_key == "vertices_camera" else "vertices_world_m_sample"
    hand = {
        "backend": template.get("backend", "WiLoR"),
        "side": template["side"],
        "detector_score": float(measurement_score),
        "bbox_xyxy": take((4,)).astype(float).tolist(),
        "cam_t": take((3,)).astype(float).tolist(),
        "source_intrinsics": intrinsics.astype(float).tolist(),
        "joints3d_camera": take((21, 3)).astype(float).tolist(),
        template_vertex_key: take(template_vertex_shape).astype(float).tolist(),
        "mano_params": template.get("mano_params", {}),
        "filter_status": status,
        "source_camera_solve": template.get("source_camera_solve", {}),
        "mano_vertex_count": int(template_vertex_shape[0]),
    }
    cam_t = np.asarray(hand["cam_t"], dtype=float)
    joints_camera_m = np.asarray(hand["joints3d_camera"], dtype=float) + cam_t
    verts_camera_m = np.asarray(hand[template_vertex_key], dtype=float) + cam_t
    hand["joints3d_source_camera_m"] = joints_camera_m.astype(float).tolist()
    hand[output_source_key] = verts_camera_m.astype(float).tolist()
    hand["mano_surface_status"] = "full_mano_vertices" if output_world_key == "vertices_world_m" else "sampled_mano_vertices"
    projected = project_points(joints_camera_m, intrinsics)
    hand["joints2d"] = projected.astype(float).tolist()
    if template.get("joints2d_raw") is not None:
        raw = np.asarray(template["joints2d_raw"], dtype=float)
        hand["joints2d_raw"] = raw.astype(float).tolist()
        err = np.linalg.norm(projected - raw, axis=1)
        hand["projection_residual_to_measurement_px"] = {
            "median": float(np.median(err)),
            "p95": float(np.percentile(err, 95)),
        }
    return hand


def kalman_rts(
    measurements: list[np.ndarray | None],
    confidences: list[float],
    fps: float,
    measurement_sigma: np.ndarray | float,
    process_position_sigma: np.ndarray | float,
    process_velocity_sigma: np.ndarray | float,
) -> tuple[list[np.ndarray], list[str]]:
    measured = [i for i, x in enumerate(measurements) if x is not None]
    if not measured:
        raise RuntimeError("no measurements for Kalman smoothing")
    dim = int(measurements[measured[0]].shape[0])  # type: ignore[index,union-attr]
    n = len(measurements)
    dt = 1.0 / fps
    meas_sigma = np.broadcast_to(np.asarray(measurement_sigma, dtype=float), (dim,))
    proc_pos = np.broadcast_to(np.asarray(process_position_sigma, dtype=float), (dim,))
    proc_vel = np.broadcast_to(np.asarray(process_velocity_sigma, dtype=float), (dim,))

    pos = np.asarray(measurements[measured[0]], dtype=float).copy()  # type: ignore[index]
    vel = np.zeros(dim, dtype=float)
    P00 = np.full(dim, 10.0, dtype=float)
    P01 = np.zeros(dim, dtype=float)
    P11 = np.full(dim, 10.0, dtype=float)

    pos_f = np.zeros((n, dim), dtype=float)
    vel_f = np.zeros((n, dim), dtype=float)
    P00_f = np.zeros((n, dim), dtype=float)
    P01_f = np.zeros((n, dim), dtype=float)
    P11_f = np.zeros((n, dim), dtype=float)
    pos_p = np.zeros((n, dim), dtype=float)
    vel_p = np.zeros((n, dim), dtype=float)
    P00_p = np.zeros((n, dim), dtype=float)
    P01_p = np.zeros((n, dim), dtype=float)
    P11_p = np.zeros((n, dim), dtype=float)

    for i in range(n):
        pred_pos = pos + dt * vel
        pred_vel = vel.copy()
        pred_P00 = P00 + 2.0 * dt * P01 + dt * dt * P11 + proc_pos * proc_pos
        pred_P01 = P01 + dt * P11
        pred_P11 = P11 + proc_vel * proc_vel
        pos_p[i], vel_p[i] = pred_pos, pred_vel
        P00_p[i], P01_p[i], P11_p[i] = pred_P00, pred_P01, pred_P11
        z = measurements[i]
        if z is None:
            pos, vel = pred_pos, pred_vel
            P00, P01, P11 = pred_P00, pred_P01, pred_P11
        else:
            conf = max(0.05, float(confidences[i]))
            R = (meas_sigma / conf) ** 2
            innovation = np.asarray(z, dtype=float) - pred_pos
            S = pred_P00 + R
            K0 = pred_P00 / S
            K1 = pred_P01 / S
            pos = pred_pos + K0 * innovation
            vel = pred_vel + K1 * innovation
            P00 = (1.0 - K0) * pred_P00
            P01 = (1.0 - K0) * pred_P01
            P11 = pred_P11 - K1 * pred_P01
        pos_f[i], vel_f[i] = pos, vel
        P00_f[i], P01_f[i], P11_f[i] = P00, P01, P11

    pos_s = pos_f.copy()
    vel_s = vel_f.copy()
    for i in range(n - 2, -1, -1):
        m00 = P00_f[i] + dt * P01_f[i]
        m01 = P01_f[i]
        m10 = P01_f[i] + dt * P11_f[i]
        m11 = P11_f[i]
        a = P00_p[i + 1]
        b = P01_p[i + 1]
        c = P11_p[i + 1]
        det = np.maximum(a * c - b * b, 1e-12)
        C00 = (m00 * c - m01 * b) / det
        C01 = (-m00 * b + m01 * a) / det
        C10 = (m10 * c - m11 * b) / det
        C11 = (-m10 * b + m11 * a) / det
        dpos = pos_s[i + 1] - pos_p[i + 1]
        dvel = vel_s[i + 1] - vel_p[i + 1]
        pos_s[i] = pos_f[i] + C00 * dpos + C01 * dvel
        vel_s[i] = vel_f[i] + C10 * dpos + C11 * dvel

    statuses = ["measured_kalman_rts" if measurements[i] is not None else "predicted_kalman_rts" for i in range(n)]
    return [pos_s[i] for i in range(n)], statuses


def hand_kalman_sigmas(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed_dim = 4 + 3 + 21 * 3
    vertex_dim = dim - fixed_dim
    if vertex_dim <= 0 or vertex_dim % 3 != 0:
        raise RuntimeError(f"unexpected hand state dimension {dim}")
    measurement = np.concatenate(
        [
            np.full(4, 18.0),
            np.full(3, 0.035),
            np.full(21 * 3, 0.018),
            np.full(vertex_dim, 0.018),
        ]
    )
    process_position = np.concatenate(
        [
            np.full(4, 4.0),
            np.full(3, 0.006),
            np.full(21 * 3, 0.004),
            np.full(vertex_dim, 0.004),
        ]
    )
    process_velocity = np.concatenate(
        [
            np.full(4, 60.0),
            np.full(3, 0.22),
            np.full(21 * 3, 0.12),
            np.full(vertex_dim, 0.12),
        ]
    )
    return measurement, process_position, process_velocity


def smooth_hands(raw_frames: list[dict], fps: float, intrinsics: np.ndarray, wilor_to_meters: float) -> tuple[list[dict], dict]:
    rejected = {"left": 0, "right": 0, "unknown": 0}
    normalized_frames = []
    for frame in raw_frames:
        normalized = []
        for raw_hand in frame["raw_hands"]:
            hand = normalize_hand_to_source_camera(raw_hand, intrinsics, wilor_to_meters)
            if hand is None:
                rejected[raw_hand.get("side", "unknown")] = rejected.get(raw_hand.get("side", "unknown"), 0) + 1
            else:
                normalized.append(hand)
        normalized_frames.append({"raw_hands": normalized})
    chosen = [choose_hand_by_side(frame["raw_hands"]) for frame in normalized_frames]
    out_frames = [{"frame_idx": f["frame_idx"], "time_s": f["time_s"], "caption": f["caption"], "hands": []} for f in raw_frames]
    stats: dict[str, dict] = {"rejected_source_camera_solves": rejected}
    for side in ["left", "right"]:
        templates = [c[side] for c in chosen if side in c]
        if not templates:
            continue
        template = templates[0]
        measurements = []
        confs = []
        best_by_frame = []
        for c in chosen:
            hand = c.get(side)
            best_by_frame.append(hand)
            measurements.append(hand_vector(hand) if hand is not None else None)
            confs.append(float(hand.get("detector_score", 0.0)) if hand is not None else 0.0)
        measured_indices = [i for i, m in enumerate(measurements) if m is not None]
        first_measured = measured_indices[0]
        last_measured = measured_indices[-1]
        dim = int(measurements[first_measured].shape[0])  # type: ignore[union-attr]
        meas_sigma, proc_pos, proc_vel = hand_kalman_sigmas(dim)
        smoothed, statuses = kalman_rts(measurements, confs, fps, meas_sigma, proc_pos, proc_vel)
        measured_count = 0
        predicted_count = 0
        outside_visibility = 0
        projection_medians = []
        for i, vec in enumerate(smoothed):
            if i < first_measured or i > last_measured:
                outside_visibility += 1
                continue
            source = best_by_frame[i] or template
            score = confs[i]
            hand = vector_to_hand(source, vec, statuses[i], score, intrinsics)
            hand["measurement_available"] = best_by_frame[i] is not None
            if hand["measurement_available"]:
                measured_count += 1
                projection_medians.append(float(hand.get("projection_residual_to_measurement_px", {}).get("median", math.nan)))
            else:
                predicted_count += 1
            out_frames[i]["hands"].append(hand)
        stats[side] = {
            "measured_frames": measured_count,
            "predicted_frames": predicted_count,
            "outside_visibility_frames": outside_visibility,
            "coverage": measured_count / max(1, len(raw_frames)),
            "median_projection_residual_px": float(np.nanmedian(projection_medians)) if projection_medians else None,
        }
    return out_frames, stats


def load_droid_reconstruction(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu")
    required = {"tstamps", "disps", "intrinsics"}
    missing = sorted(required.difference(blob))
    if missing:
        raise RuntimeError(f"DROID reconstruction missing keys: {missing}")
    return {
        "tstamps": blob["tstamps"].detach().cpu().numpy().astype(int),
        "disps": blob["disps"].detach().cpu().numpy().astype(float),
        "intrinsics": blob["intrinsics"].detach().cpu().numpy().astype(float),
        "depth_level": blob.get("depth_level", "unknown"),
    }


def sample_droid_depth_relative(recon: dict, frame_idx: int, point_xy: np.ndarray, image_size: tuple[int, int], max_keyframe_gap: int) -> tuple[float, int] | None:
    tstamps = recon["tstamps"]
    nearest = int(np.argmin(np.abs(tstamps - frame_idx)))
    source_idx = int(tstamps[nearest])
    if abs(source_idx - frame_idx) > max_keyframe_gap:
        return None
    disps = recon["disps"][nearest]
    width, height = image_size
    x = float(point_xy[0]) / width * disps.shape[1]
    y = float(point_xy[1]) / height * disps.shape[0]
    xi = int(np.clip(round(x), 0, disps.shape[1] - 1))
    yi = int(np.clip(round(y), 0, disps.shape[0] - 1))
    patch = disps[max(0, yi - 1) : min(disps.shape[0], yi + 2), max(0, xi - 1) : min(disps.shape[1], xi + 2)]
    valid = patch[np.isfinite(patch) & (patch > 1e-4)]
    if valid.size == 0:
        return None
    return 1.0 / float(np.median(valid)), source_idx


def estimate_droid_metric_scale(frames: list[dict], recon: dict, image_size: tuple[int, int], max_keyframe_gap: int) -> dict:
    ratios = []
    samples = []
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        for hand in frame["hands"]:
            if not hand.get("measurement_available", False):
                continue
            points2d = np.asarray(hand["joints2d"], dtype=float)
            center = np.median(points2d, axis=0)
            sampled = sample_droid_depth_relative(recon, frame_idx, center, image_size, max_keyframe_gap)
            if sampled is None:
                continue
            depth_rel, keyframe_idx = sampled
            hand_depth_m = float(np.median(np.asarray(hand["joints3d_source_camera_m"], dtype=float)[:, 2]))
            if 0.2 <= hand_depth_m <= 3.5 and 0.05 <= depth_rel <= 10.0:
                ratio = hand_depth_m / depth_rel
                ratios.append(ratio)
                samples.append(
                    {
                        "frame_idx": frame_idx,
                        "keyframe_idx": keyframe_idx,
                        "side": hand["side"],
                        "hand_depth_m": hand_depth_m,
                        "droid_depth_relative": depth_rel,
                        "droid_to_meters": ratio,
                    }
                )
    if len(ratios) < 12:
        raise RuntimeError(f"insufficient DROID depth scale samples: {len(ratios)}")
    arr = np.asarray(ratios, dtype=float)
    median = float(np.median(arr))
    residual = arr / median - 1.0
    return {
        "status": "droid_relative_depth_scaled_to_source_camera_mano_depth",
        "droid_to_meters": median,
        "sample_count": len(ratios),
        "ratio_iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "relative_residual_iqr": [
            float(np.percentile(residual, 25)),
            float(np.percentile(residual, 75)),
        ],
        "samples_preview": samples[:20],
    }


def estimate_droid_metric_scale_from_raw(
    raw_frames: list[dict],
    intrinsics: np.ndarray,
    wilor_to_meters: float,
    recon: dict,
    image_size: tuple[int, int],
    max_keyframe_gap: int,
) -> dict:
    keyframes = recon["tstamps"]
    ratios = []
    samples = []
    for frame in raw_frames:
        frame_idx = int(frame["frame_idx"])
        if int(np.min(np.abs(keyframes - frame_idx))) > max_keyframe_gap:
            continue
        normalized = []
        for raw_hand in frame["raw_hands"]:
            hand = normalize_hand_to_source_camera(raw_hand, intrinsics, wilor_to_meters)
            if hand is not None:
                normalized.append(hand)
        for hand in choose_hand_by_side(normalized).values():
            points2d = np.asarray(hand["joints2d"], dtype=float)
            center = np.median(points2d, axis=0)
            sampled = sample_droid_depth_relative(recon, frame_idx, center, image_size, max_keyframe_gap)
            if sampled is None:
                continue
            depth_rel, keyframe_idx = sampled
            hand_depth_m = float(np.median(np.asarray(hand["joints3d_source_camera_m"], dtype=float)[:, 2]))
            if 0.2 <= hand_depth_m <= 3.5 and 0.05 <= depth_rel <= 10.0:
                ratio = hand_depth_m / depth_rel
                ratios.append(ratio)
                samples.append(
                    {
                        "frame_idx": frame_idx,
                        "keyframe_idx": keyframe_idx,
                        "side": hand["side"],
                        "hand_depth_m": hand_depth_m,
                        "droid_depth_relative": depth_rel,
                        "droid_to_meters": ratio,
                    }
                )
    if len(ratios) < 12:
        raise RuntimeError(f"insufficient full-clip DROID depth scale samples: {len(ratios)}")
    arr = np.asarray(ratios, dtype=float)
    median = float(np.median(arr))
    residual = arr / median - 1.0
    return {
        "status": "full_clip_droid_relative_depth_scaled_to_source_camera_mano_depth",
        "droid_to_meters": median,
        "sample_count": len(ratios),
        "ratio_iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "relative_residual_iqr": [
            float(np.percentile(residual, 25)),
            float(np.percentile(residual, 75)),
        ],
        "samples_preview": samples[:20],
    }


def transform_hands_to_world(frames: list[dict], droid_npz: Path, droid_to_meters: float) -> np.ndarray:
    droid = np.load(droid_npz)
    dense = droid["T_world_camera"].astype(float)
    if len(dense) < max(int(frame["frame_idx"]) for frame in frames) + 1:
        raise RuntimeError(f"DROID frames {len(dense)} do not cover requested source frames")
    T = dense[np.asarray([int(frame["frame_idx"]) for frame in frames], dtype=int)]
    T_metric = T.copy()
    T_metric[:, :3, 3] *= droid_to_meters
    for i, frame in enumerate(frames):
        frame["camera"] = {
            "T_world_camera_metric": T_metric[i].tolist(),
            "position_world_m": T_metric[i, :3, 3].astype(float).tolist(),
        }
        for hand in frame["hands"]:
            joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
            verts = hand_vertices(hand, "_source_camera_m")
            joints_h = np.c_[joints, np.ones(len(joints))]
            verts_h = np.c_[verts, np.ones(len(verts))]
            hand["joints3d_world_m"] = (T_metric[i] @ joints_h.T).T[:, :3].astype(float).tolist()
            world_key = "vertices_world_m" if len(verts) > 100 else "vertices_world_m_sample"
            hand[world_key] = (T_metric[i] @ verts_h.T).T[:, :3].astype(float).tolist()
            hand["world_coordinate_status"] = "source_camera_mano_metric_transformed_by_droid_metric_camera_pose"
    return T_metric


def active_object_hands(frame_ann: dict) -> list[dict]:
    caption = frame_ann.get("caption", "").lower()
    wanted: set[str] = set()
    if "left hand places" in caption or "left hand holds" in caption:
        wanted.add("left")
    if "right hand" in caption and ("tomato" in caption or "knife" in caption):
        wanted.add("right")
    if "left hand" in caption and ("tomato" in caption or "bowl" in caption):
        wanted.add("left")
    if not wanted and "tomato" in caption:
        wanted = {"left", "right"}
    selected = [hand for hand in frame_ann["hands"] if hand["side"] in wanted]
    return selected if selected else frame_ann["hands"]


def hand_association_geometry(frame_ann: dict) -> dict:
    hands = active_object_hands(frame_ann)
    points = []
    boxes = []
    tip_points = []
    for hand in hands:
        joints = np.asarray(hand["joints2d"], dtype=float)
        points.extend(joints)
        tip_points.extend(joints[TIP_IDS])
        boxes.append(np.asarray(hand["bbox_xyxy"], dtype=float))
    return {
        "points": np.asarray(points, dtype=float) if points else np.zeros((0, 2), dtype=float),
        "tips": np.asarray(tip_points, dtype=float) if tip_points else np.zeros((0, 2), dtype=float),
        "boxes": boxes,
    }


def caption_phase(frame_ann: dict) -> str:
    caption = frame_ann.get("caption", "").lower()
    if "scrape" in caption:
        return "scrape"
    if "chop" in caption:
        return "chop"
    if "place" in caption:
        return "place"
    return "other"


def point_box_distance(point: np.ndarray, box: np.ndarray, pad: float) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - pad - point[0], 0.0, point[0] - x2 - pad)
    dy = max(y1 - pad - point[1], 0.0, point[1] - y2 - pad)
    return float(math.hypot(dx, dy))


def association_distance(center: np.ndarray, geom: dict) -> float:
    dists = []
    points = geom["points"]
    if points.size:
        dists.append(float(np.linalg.norm(points - center[None, :], axis=1).min()))
    for box in geom["boxes"]:
        dists.append(point_box_distance(center, box, pad=90.0))
    return min(dists) if dists else math.inf


def tomato_red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_or(
        cv2.inRange(hsv, (0, 120, 70), (12, 255, 255)),
        cv2.inRange(hsv, (168, 120, 70), (180, 255, 255)),
    ).astype(bool)


def red_mask_boxes(frame: np.ndarray, geom: dict, prev_box: list[float] | None) -> list[dict]:
    mask = tomato_red_mask(frame).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[dict] = []
    prev_center = None
    if prev_box is not None:
        p = np.asarray(prev_box, dtype=float)
        prev_center = 0.5 * (p[:2] + p[2:])
    for c in contours:
        area = cv2.contourArea(c)
        if area < 60 or area > 18000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 4.5:
            continue
        center = np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=float)
        assoc_dist = association_distance(center, geom)
        prev_dist = math.inf if prev_center is None else float(np.linalg.norm(prev_center - center))
        if min(assoc_dist, prev_dist) > 230.0:
            continue
        peri = cv2.arcLength(c, True)
        circularity = float(4.0 * math.pi * area / (peri * peri + 1e-6))
        score = 0.05 + 0.18 * circularity + 0.28 * math.exp(-min(assoc_dist, 260.0) / 95.0)
        if prev_center is not None:
            score += 0.12 * math.exp(-min(prev_dist, 260.0) / 90.0)
        pad = 18.0
        box = [
            float(max(0.0, x - pad)),
            float(max(0.0, y - pad)),
            float(min(frame.shape[1] - 1.0, x + w + pad)),
            float(min(frame.shape[0] - 1.0, y + h + pad)),
        ]
        boxes.append(
            {
                "box": box,
                "score": score,
                "label": "compact_red_contact_component",
                "source": "red_contact_component",
                "association_dist_px": assoc_dist,
                "prev_dist_px": prev_dist,
                "red_component_area_px": float(area),
                "red_component_circularity": circularity,
            }
        )
    boxes.sort(key=lambda item: item["score"], reverse=True)
    return boxes[:8]


def refine_deformable_tomato_mask(frame: np.ndarray, mask: np.ndarray, frame_ann: dict, geom: dict) -> tuple[np.ndarray, str]:
    phase = caption_phase(frame_ann)
    if phase not in {"chop", "scrape"}:
        return mask, "sam_single_mask"
    red = tomato_red_mask(frame).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base_ys, base_xs = np.where(mask)
    if len(base_xs) == 0:
        return mask, "sam_single_mask"
    base_center = np.asarray([base_xs.mean(), base_ys.mean()], dtype=float)
    selected = np.zeros_like(red, dtype=np.uint8)
    selected[mask] = 1
    radius = 260.0 if phase == "scrape" else 340.0
    max_area_ratio = 6.0 if phase == "scrape" else 4.5
    max_width = 420 if phase == "scrape" else 460
    max_height = 360 if phase == "scrape" else 620
    max_component_area = 15000 if phase == "scrape" else 24000
    for c in contours:
        area = cv2.contourArea(c)
        if area < 45 or area > max_component_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        center = np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=float)
        assoc = association_distance(center, geom)
        base_dist = float(np.linalg.norm(center - base_center))
        if assoc <= 135.0 and base_dist <= radius:
            cv2.drawContours(selected, [c], -1, 1, thickness=-1)
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    if int(selected.sum()) < int(mask.sum()):
        return mask, "sam_single_mask"
    if int(selected.sum()) > max_area_ratio * max(1, int(mask.sum())):
        return mask, "sam_single_mask"
    ys, xs = np.where(selected)
    if len(xs) == 0:
        return mask, "sam_single_mask"
    if (xs.max() - xs.min() + 1) > max_width or (ys.max() - ys.min() + 1) > max_height:
        return mask, "sam_single_mask"
    return selected, f"{phase}_red_component_union"


def warp_mask_forward(
    flow_estimator,
    prev_frame: np.ndarray,
    frame: np.ndarray,
    prev_mask: np.ndarray,
) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = flow_estimator.calc(prev_gray, gray, None)
    ys, xs = np.where(prev_mask)
    warped = np.zeros(prev_mask.shape, dtype=np.uint8)
    if len(xs) == 0:
        return warped.astype(bool)
    moved_x = np.rint(xs.astype(np.float32) + flow[ys, xs, 0]).astype(np.int32)
    moved_y = np.rint(ys.astype(np.float32) + flow[ys, xs, 1]).astype(np.int32)
    valid = (0 <= moved_x) & (moved_x < prev_mask.shape[1]) & (0 <= moved_y) & (moved_y < prev_mask.shape[0])
    warped[moved_y[valid], moved_x[valid]] = 1
    warped = cv2.dilate(warped, np.ones((3, 3), np.uint8), iterations=2)
    warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return warped.astype(bool)


def fuse_temporal_object_mask(
    flow_estimator,
    prev_frame: np.ndarray | None,
    frame: np.ndarray,
    prev_mask: np.ndarray | None,
    mask: np.ndarray,
    frame_ann: dict,
    geom: dict,
) -> tuple[np.ndarray, str]:
    phase = caption_phase(frame_ann)
    if phase != "scrape" or prev_frame is None or prev_mask is None:
        return mask, "sam_single_mask"
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask, "sam_single_mask"
    current_center = np.asarray([xs.mean(), ys.mean()], dtype=float)
    warped = warp_mask_forward(flow_estimator, prev_frame, frame, prev_mask)
    if int(warped.sum()) < 80:
        return mask, "sam_single_mask"
    selected = mask.astype(np.uint8)
    current_dilated = cv2.dilate(mask.astype(np.uint8), np.ones((71, 71), np.uint8), iterations=1).astype(bool)
    contours, _ = cv2.findContours(warped.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_center_dist = 280.0 if phase == "scrape" else 340.0
    max_width = 520 if phase == "scrape" else 560
    max_height = 520 if phase == "scrape" else 660
    for c in contours:
        area = cv2.contourArea(c)
        if area < 45 or area > 90000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        center = np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=float)
        component = np.zeros_like(selected, dtype=np.uint8)
        cv2.drawContours(component, [c], -1, 1, thickness=-1)
        center_dist = float(np.linalg.norm(center - current_center))
        has_current_overlap = bool(np.logical_and(component.astype(bool), current_dilated).any())
        if center_dist > max_center_dist and not has_current_overlap:
            continue
        if association_distance(center, geom) > 180.0 and center_dist > 140.0:
            continue
        selected[component.astype(bool)] = 1
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    if int(selected.sum()) <= int(mask.sum()) * 1.05:
        return mask, "sam_single_mask"
    ys, xs = np.where(selected)
    if len(xs) == 0:
        return mask, "sam_single_mask"
    if (xs.max() - xs.min() + 1) > max_width or (ys.max() - ys.min() + 1) > max_height:
        return mask, "sam_single_mask"
    if int(selected.sum()) > 120000:
        return mask, "sam_single_mask"
    return selected, f"{phase}_optical_flow_temporal_union"


def hand_contact_points(frame_ann: dict) -> np.ndarray:
    pts = []
    for hand in frame_ann["hands"]:
        joints = np.asarray(hand["joints2d"], dtype=float)
        pts.extend(joints[TIP_IDS])
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2), dtype=float)


def mask_contact_score(mask: np.ndarray, points: np.ndarray) -> tuple[float, float]:
    if points.size == 0:
        return 0.0, math.inf
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, math.inf
    center_dist = []
    inside = 0
    dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    for p in points:
        x = int(np.clip(round(p[0]), 0, mask.shape[1] - 1))
        y = int(np.clip(round(p[1]), 0, mask.shape[0] - 1))
        if mask[y, x]:
            inside += 1
            center_dist.append(0.0)
        else:
            center_dist.append(float(dt[y, x]))
    return inside / max(1, len(points)), min(center_dist)


def load_owl_detector(device: str):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_id = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model


def owl_boxes(processor, model, frame: np.ndarray, threshold: float) -> list[dict]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    texts = [["tomato", "red tomato", "cut tomato", "tomato pieces"]]
    inputs = processor(text=texts, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        out,
        target_sizes=torch.tensor([image.size[::-1]], device=model.device),
        threshold=threshold,
        text_labels=texts,
    )[0]
    boxes = []
    labels = result.get("text_labels", [])
    for i, score in enumerate(result["scores"].detach().cpu().tolist()):
        label = labels[i] if isinstance(labels, list) and i < len(labels) else str(int(result["labels"][i]))
        if "tomato" not in label:
            continue
        box = result["boxes"][i].detach().cpu().numpy().astype(float)
        boxes.append({"box": box.tolist(), "score": float(score), "label": label, "source": "owlv2"})
    return boxes


def load_sam(checkpoint: Path, device: str):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device).eval()
    return SamPredictor(sam)


def sam_mask_from_boxes(
    predictor,
    frame: np.ndarray,
    boxes: list[dict],
    contact_points: np.ndarray,
    geom: dict,
    prev_box: list[float] | None,
) -> dict | None:
    if not boxes:
        return None
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 60, 35), (14, 255, 255)),
        cv2.inRange(hsv, (165, 60, 35), (180, 255, 255)),
    ).astype(bool)
    scored = []
    prev_center = None
    if prev_box is not None:
        p = np.asarray(prev_box, dtype=float)
        prev_center = 0.5 * (p[:2] + p[2:])
    for box_info in boxes[:8]:
        box = np.asarray(box_info["box"], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        for mask, sam_score in zip(masks, scores):
            area = int(mask.sum())
            if area < 80:
                continue
            ys, xs = np.where(mask)
            bbox = np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)
            center = np.asarray([xs.mean(), ys.mean()], dtype=float)
            contact_ratio, min_tip_dist = mask_contact_score(mask, contact_points)
            red_fraction = float(red[mask].mean()) if area else 0.0
            assoc_dist = association_distance(center, geom)
            prev_dist = math.inf if prev_center is None else float(np.linalg.norm(prev_center - center))
            if min(assoc_dist, prev_dist) > 230.0:
                continue
            if red_fraction < 0.04 and min(assoc_dist, prev_dist) > 90.0:
                continue
            score = (
                float(box_info["score"])
                + 0.35 * float(sam_score)
                + 0.45 * contact_ratio
                + 0.18 * red_fraction
                - 0.0016 * min(assoc_dist, 180.0)
                - 0.001 * min(prev_dist, 160.0)
            )
            scored.append((score, mask, bbox, box_info, float(sam_score), contact_ratio, min_tip_dist, red_fraction, prev_dist, assoc_dist))
    if not scored:
        return None
    score, mask, bbox, box_info, sam_score, contact_ratio, min_tip_dist, red_fraction, prev_dist, assoc_dist = max(scored, key=lambda item: item[0])
    m = mask.astype(np.uint8)
    moments = cv2.moments(m)
    if moments["m00"] <= 0:
        return None
    center = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
    edge_touch = bool(bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= frame.shape[1] - 2 or bbox[3] >= frame.shape[0] - 2)
    box_w = float(bbox[2] - bbox[0] + 1.0)
    box_h = float(bbox[3] - bbox[1] + 1.0)
    if edge_touch and (box_w < 28.0 or box_h < 28.0 or min_tip_dist > 30.0):
        return None
    return {
        "mask": mask,
        "bbox_xyxy": bbox.astype(float).tolist(),
        "center_xy": center,
        "area_px": int(mask.sum()),
        "score": float(score),
        "owl_score": float(box_info["score"]),
        "owl_label": box_info["label"],
        "sam_score": sam_score,
        "contact_ratio": float(contact_ratio),
        "min_tip_dist_px": float(min_tip_dist),
        "association_dist_px": float(assoc_dist),
        "prev_center_dist_px": float(prev_dist),
        "red_fraction": red_fraction,
        "proposal_source": box_info.get("source", "unknown"),
        "edge_touch": edge_touch,
    }


def run_object_masks(args: argparse.Namespace, frames: list[dict], actions: list[dict], render: RenderSpec) -> tuple[list[dict], dict]:
    start, end = manipulated_object_interval(actions, args.object_label)
    available = {int(frame["frame_idx"]): i for i, frame in enumerate(frames)}
    frame_start = min(available)
    frame_end = max(available)
    start = max(start, frame_start)
    end = min(end, frame_end)
    if args.frame_start is not None:
        start = max(start, int(args.frame_start))
    if args.frame_end is not None:
        end = min(end, int(args.frame_end))
    if start > end:
        raise RuntimeError(f"empty object interval after frame limits: start={start}, end={end}")
    step = max(1, int(args.object_stride))
    cap, info = open_video(args.clip)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    owl_processor, owl_model = load_owl_detector(device)
    sam = load_sam(args.sam_checkpoint, device)

    object_meas: list[dict | None] = [None] * len(frames)
    prev_box: list[float] | None = None
    prev_mask: np.ndarray | None = None
    prev_frame: np.ndarray | None = None
    prev_source_idx: int | None = None
    flow_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    processed = 0
    detected = 0
    try:
        source_indices = [idx for idx in range(start, end + 1, step) if idx in available]
        for source_idx in tqdm(source_indices, desc="object_sam"):
            local_idx = available[source_idx]
            frame = read_video_frame(cap, source_idx)
            geom = hand_association_geometry(frames[local_idx])
            contact = geom["tips"]
            boxes = red_mask_boxes(frame, geom, prev_box)
            boxes.extend(owl_boxes(owl_processor, owl_model, frame, args.owl_threshold))
            if prev_box is not None:
                boxes.append({"box": prev_box, "score": 0.12, "label": "tomato_temporal_prior", "source": "temporal_prior"})
            boxes.sort(key=lambda item: float(item["score"]), reverse=True)
            mask_info = sam_mask_from_boxes(sam, frame, boxes, contact, geom, prev_box)
            processed += 1
            if mask_info is not None:
                refined_mask, refined_status = refine_deformable_tomato_mask(frame, mask_info["mask"], frames[local_idx], geom)
                if prev_source_idx is not None and source_idx - prev_source_idx == step:
                    temporal_mask, temporal_status = fuse_temporal_object_mask(
                        flow_estimator,
                        prev_frame,
                        frame,
                        prev_mask,
                        refined_mask,
                        frames[local_idx],
                        geom,
                    )
                    if temporal_mask is not refined_mask:
                        refined_mask = temporal_mask
                        refined_status = temporal_status
                if refined_mask is not mask_info["mask"]:
                    m = refined_mask.astype(np.uint8)
                    moments = cv2.moments(m)
                    if moments["m00"] > 0:
                        ys, xs = np.where(refined_mask)
                        mask_info["mask"] = refined_mask
                        mask_info["bbox_xyxy"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                        mask_info["center_xy"] = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
                        mask_info["area_px"] = int(refined_mask.sum())
                        mask_info["mask_refinement"] = refined_status
                prev_box = mask_info["bbox_xyxy"]
                prev_mask = mask_info["mask"].copy()
                prev_frame = frame.copy()
                prev_source_idx = source_idx
                object_meas[local_idx] = {k: v for k, v in mask_info.items() if k != "mask"}
                mask_small = cv2.resize(mask_info["mask"].astype(np.uint8) * 255, (render.width, render.height), interpolation=cv2.INTER_NEAREST)
                mask_path = args.output_dir / "object_masks" / f"{source_idx:06d}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_path), mask_small)
                object_meas[local_idx]["mask_path"] = str(mask_path)
                detected += 1
    finally:
        cap.release()
        del owl_model, sam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return object_meas, {
        "label": args.object_label,
        "semantic_interval": [start, end],
        "stride": step,
        "processed_frames": processed,
        "detected_frames": detected,
        "detection_rate_on_processed": detected / max(1, processed),
        "backend": "OWLv2 proposals + SAM ViT-B masks + contact/temporal proposal scoring + deformable tomato refinement + DIS optical-flow mask propagation",
    }


def fill_object_track(
    frames: list[dict],
    object_meas: list[dict | None],
    fps: float,
    image_size: tuple[int, int],
    object_label: str,
    semantic_interval: tuple[int, int],
) -> dict:
    meas = []
    conf = []
    template = None
    prev_valid_center: np.ndarray | None = None
    invalid_measurements = 0
    for m in object_meas:
        if m is None:
            meas.append(None)
            conf.append(0.0)
        else:
            bbox = np.asarray(m["bbox_xyxy"], dtype=float)
            center = np.asarray(m["center_xy"], dtype=float)
            edge = bool(m.get("edge_touch", False))
            width = bbox[2] - bbox[0] + 1.0
            height = bbox[3] - bbox[1] + 1.0
            prev_jump = math.inf if prev_valid_center is None else float(np.linalg.norm(center - prev_valid_center))
            valid = not (edge and (width < 36.0 or height < 36.0 or float(m.get("red_fraction", 0.0)) < 0.20))
            if prev_valid_center is not None and prev_jump > 520.0 and float(m.get("min_tip_dist_px", math.inf)) > 60.0:
                valid = False
            if valid:
                if template is None:
                    template = m
                meas.append(np.asarray(m["center_xy"] + m["bbox_xyxy"] + [math.sqrt(max(1.0, float(m["area_px"])))], dtype=float))
                conf.append(float(max(0.05, m["score"])))
                prev_valid_center = center
            else:
                meas.append(None)
                conf.append(0.0)
                invalid_measurements += 1
    if template is None:
        raise RuntimeError("object module produced no valid SAM masks")
    meas_sigma = np.asarray([16.0, 16.0, 28.0, 28.0, 28.0, 28.0, 18.0], dtype=float)
    proc_pos = np.asarray([3.5, 3.5, 6.0, 6.0, 6.0, 6.0, 3.5], dtype=float)
    proc_vel = np.asarray([80.0, 80.0, 120.0, 120.0, 120.0, 120.0, 80.0], dtype=float)
    smoothed, statuses = kalman_rts(meas, conf, fps, meas_sigma, proc_pos, proc_vel)
    measured_indices = [i for i, m in enumerate(meas) if m is not None]
    first_measured = measured_indices[0]
    last_measured = measured_indices[-1]
    max_edge_prediction = max(3, int(round(0.50 * fps)))
    observed = 0
    predicted = 0
    outside_visibility = 0
    edge_predicted = 0
    contact_frames = 0
    start, end = semantic_interval
    width, height = image_size
    for i, frame in enumerate(frames):
        source_idx = int(frame["frame_idx"])
        active = start <= source_idx <= end
        if not active:
            frame["object"] = {"label": object_label, "status": "outside_semantic_interval"}
            continue
        before = i < first_measured
        after = i > last_measured
        edge_gap = first_measured - i if before else i - last_measured if after else 0
        if (before or after) and edge_gap > max_edge_prediction:
            outside_visibility += 1
            frame["object"] = {"label": object_label, "status": "unobserved_before_or_after_object_track"}
            continue
        vec = smoothed[i]
        center = vec[:2]
        bbox = vec[2:6]
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        center[0] = float(np.clip(center[0], 0, width - 1))
        center[1] = float(np.clip(center[1], 0, height - 1))
        box_w = float(bbox[2] - bbox[0])
        box_h = float(bbox[3] - bbox[1])
        m = object_meas[i]
        if (box_w < 12.0 or box_h < 12.0) or (center[0] <= 1.0 and box_w < 60.0):
            frame["object"] = {"label": object_label, "status": "unobserved_degenerate_track_state"}
            if m is not None:
                invalid_measurements += 1
            continue
        status = "measured_sam_kalman" if m is not None else "predicted_kalman"
        if m is not None:
            observed += 1
            contact_ratio = float(m.get("contact_ratio", 0.0))
            min_tip = float(m.get("min_tip_dist_px", math.inf))
            red_fraction = float(m.get("red_fraction", 0.0))
        else:
            predicted += 1
            if before or after:
                edge_predicted += 1
            contact = hand_contact_points(frame)
            if contact.size:
                d = np.linalg.norm(contact - center[None, :], axis=1)
                min_tip = float(d.min())
                contact_ratio = float(min_tip < 45.0)
            else:
                min_tip = math.inf
                contact_ratio = 0.0
            red_fraction = 0.0
        if contact_ratio > 0 or min_tip < 45.0:
            contact_frames += 1
        frame["object"] = {
            "label": object_label,
            "status": status,
            "bbox_xyxy": bbox.astype(float).tolist(),
            "center_xy": center.astype(float).tolist(),
            "area_px": float(max(1.0, vec[6] * vec[6])),
            "measurement_available": m is not None,
            "mask_path": m.get("mask_path") if m else None,
            "contact_ratio": contact_ratio,
            "min_tip_dist_px": min_tip,
            "red_fraction": red_fraction,
            "mask_refinement": m.get("mask_refinement") if m else None,
            "proposal_source": m.get("proposal_source") if m else None,
            "sam_score": float(m["sam_score"]) if m and "sam_score" in m else None,
            "pose_status": "pending_world_ray_depth_optimization",
        }
    return {
        "measured_frames": observed,
        "predicted_frames": predicted,
        "outside_visibility_frames": outside_visibility,
        "edge_predicted_frames": edge_predicted,
        "contact_frames": contact_frames,
        "first_measured_source_frame": int(frames[first_measured]["frame_idx"]),
        "last_measured_source_frame": int(frames[last_measured]["frame_idx"]),
        "invalid_measurements_rejected": invalid_measurements,
    }


def source_camera_ray(center_xy: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    x, y = np.asarray(center_xy, dtype=float)
    return np.asarray([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=float)


def contact_anchor_world(frame: dict, center_xy: np.ndarray, radius_px: float) -> tuple[np.ndarray | None, float]:
    candidates = []
    for hand in frame["hands"]:
        points2d = np.asarray(hand["joints2d"], dtype=float)[TIP_IDS]
        points3d = np.asarray(hand["joints3d_world_m"], dtype=float)[TIP_IDS]
        dists = np.linalg.norm(points2d - center_xy[None, :], axis=1)
        for dist, point in zip(dists, points3d):
            if dist <= radius_px:
                candidates.append((float(dist), point))
    if not candidates:
        return None, math.inf
    candidates.sort(key=lambda item: item[0])
    selected = np.asarray([point for _, point in candidates[:4]], dtype=float)
    return selected.mean(axis=0), candidates[0][0]


def add_sparse_row(rows: list[int], cols: list[int], vals: list[float], rhs: list[float], row: int, terms: list[tuple[int, float]], target: float, sigma: float) -> int:
    weight = 1.0 / sigma
    for col, val in terms:
        rows.append(row)
        cols.append(col)
        vals.append(float(val) * weight)
    rhs.append(float(target) * weight)
    return row + 1


def attach_object_world(
    frames: list[dict],
    T_metric: np.ndarray,
    intrinsics: np.ndarray,
    recon: dict,
    droid_to_meters: float,
    image_size: tuple[int, int],
    max_keyframe_gap: int,
) -> dict:
    active = []
    droid_depth_samples = {}
    contact_samples = {}
    for i, frame in enumerate(frames):
        obj = frame.get("object", {})
        if obj.get("center_xy") is None or obj.get("status") == "outside_semantic_interval":
            continue
        active.append(i)
        center = np.asarray(obj["center_xy"], dtype=float)
        source_idx = int(frame["frame_idx"])
        sampled = sample_droid_depth_relative(recon, source_idx, center, image_size, max_keyframe_gap)
        if sampled is not None:
            droid_depth_samples[i] = (sampled[0] * droid_to_meters, sampled[1])
        anchor, min_dist = contact_anchor_world(frame, center, max(65.0, min(150.0, float(obj.get("min_tip_dist_px", 150.0)) + 35.0)))
        if anchor is not None:
            contact_samples[i] = (anchor, min_dist)
    if not active:
        raise RuntimeError("no active object frames for world pose optimization")

    index = {frame_idx: j for j, frame_idx in enumerate(active)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    row = 0
    absolute_rows = 0
    initial_depths: dict[int, float] = {}
    for frame_idx in active:
        col = index[frame_idx]
        if frame_idx in droid_depth_samples:
            depth_m, _ = droid_depth_samples[frame_idx]
            row = add_sparse_row(rows, cols, vals, rhs, row, [(col, 1.0)], depth_m, sigma=0.18)
            absolute_rows += 1
            initial_depths[frame_idx] = float(depth_m)
        if frame_idx in contact_samples:
            anchor, min_dist = contact_samples[frame_idx]
            center = np.asarray(frames[frame_idx]["object"]["center_xy"], dtype=float)
            ray = source_camera_ray(center, intrinsics)
            origin = T_metric[frame_idx, :3, 3]
            direction = T_metric[frame_idx, :3, :3] @ ray
            sigma = 0.055 + 0.00035 * min(min_dist, 150.0)
            depth_from_anchor = float(np.dot(anchor - origin, direction) / max(1e-9, np.dot(direction, direction)))
            if 0.20 <= depth_from_anchor <= 3.20:
                initial_depths.setdefault(frame_idx, depth_from_anchor)
            for axis in range(3):
                row = add_sparse_row(
                    rows,
                    cols,
                    vals,
                    rhs,
                    row,
                    [(col, float(direction[axis]))],
                    float(anchor[axis] - origin[axis]),
                    sigma=sigma,
                )
                absolute_rows += 1
    for a, b, c in zip(active[:-2], active[1:-1], active[2:]):
        if b - a != 1 or c - b != 1:
            continue
        for axis in range(3):
            terms = []
            for frame_idx, coeff in [(a, 1.0), (b, -2.0), (c, 1.0)]:
                center = np.asarray(frames[frame_idx]["object"]["center_xy"], dtype=float)
                ray = source_camera_ray(center, intrinsics)
                direction = T_metric[frame_idx, :3, :3] @ ray
                terms.append((index[frame_idx], coeff * float(direction[axis])))
            origin_term = (
                T_metric[c, axis, 3]
                - 2.0 * T_metric[b, axis, 3]
                + T_metric[a, axis, 3]
            )
            row = add_sparse_row(rows, cols, vals, rhs, row, terms, -float(origin_term), sigma=0.020)
    if row == 0:
        raise RuntimeError("object world optimizer has no depth/contact/smoothness rows")
    if absolute_rows == 0:
        raise RuntimeError("object world optimizer has no absolute DROID-depth or contact-anchor rows")
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(row, len(active))).tocsr()
    b = np.asarray(rhs, dtype=float)
    if initial_depths:
        median_initial = float(np.median(list(initial_depths.values())))
    else:
        median_initial = 1.40
    x0 = np.asarray([initial_depths.get(frame_idx, median_initial) for frame_idx in active], dtype=float)
    x0 = np.clip(x0, 0.20, 3.20)

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = A @ x - b
        value = 0.5 * float(np.dot(residual, residual))
        grad = A.T @ residual
        return value, np.asarray(grad, dtype=float)

    result = minimize(
        fun=lambda x: objective(x)[0],
        x0=x0,
        jac=lambda x: objective(x)[1],
        bounds=[(0.20, 3.20)] * len(active),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"object world depth optimization failed: {result.message}")
    depths = np.asarray(result.x, dtype=float)
    fx, fy, _, _ = intrinsics
    focal = 0.5 * (float(fx) + float(fy))
    for frame_idx, depth_m in zip(active, depths):
        obj = frames[frame_idx]["object"]
        center = np.asarray(obj["center_xy"], dtype=float)
        point_cam = source_camera_ray(center, intrinsics) * depth_m
        point_world = (T_metric[frame_idx] @ np.r_[point_cam, 1.0])[:3]
        radius_px = math.sqrt(max(1.0, float(obj["area_px"])) / math.pi)
        obj["center_source_camera_m"] = point_cam.astype(float).tolist()
        obj["center_world_m"] = point_world.astype(float).tolist()
        obj["depth_m"] = float(depth_m)
        obj["radius_m"] = float(radius_px * depth_m / focal)
        obj["pose_type"] = "deformable_object_centroid_with_spherical_extent"
        obj["pose_status"] = "world_ray_depth_optimized_from_droid_depth_contact_and_temporal_smoothness"
        obj["depth_evidence"] = {
            "droid_depth": frame_idx in droid_depth_samples,
            "contact_anchor": frame_idx in contact_samples,
            "temporal_smoothness": True,
        }
        if frame_idx in droid_depth_samples:
            obj["droid_depth_keyframe"] = int(droid_depth_samples[frame_idx][1])
        if frame_idx in contact_samples:
            obj["contact_anchor_min_tip_dist_px"] = float(contact_samples[frame_idx][1])
    return {
        "active_frames": len(active),
        "variables": len(active),
        "linear_rows": row,
        "absolute_rows": absolute_rows,
        "droid_depth_frames": len(droid_depth_samples),
        "contact_anchor_frames": len(contact_samples),
        "cost": float(result.fun),
        "optimizer_iterations": int(result.nit),
        "projected_gradient_inf_norm": float(np.max(np.abs(result.jac))) if result.jac is not None else None,
        "depth_m_iqr": [float(np.percentile(depths, 25)), float(np.percentile(depths, 75))],
        "depth_m_minmax": [float(np.min(depths)), float(np.max(depths))],
    }


def draw_hand_overlay(frame: np.ndarray, frame_ann: dict, sx: float, sy: float, mano_edges: dict[int, np.ndarray]) -> None:
    for hand in frame_ann["hands"]:
        color = LEFT_COLOR if hand["side"] == "left" else RIGHT_COLOR
        pts = np.asarray(hand["joints2d"], dtype=float) * np.asarray([sx, sy])
        verts_camera = hand_vertices(hand, "_source_camera_m")
        verts2d = project_points(verts_camera, np.asarray(hand["source_intrinsics"], dtype=float)) * np.asarray([sx, sy])
        box = np.asarray(hand["bbox_xyxy"], dtype=float) * np.asarray([sx, sy, sx, sy])
        x1, y1, x2, y2 = box.astype(int)
        measured = bool(hand.get("measurement_available", False))
        thickness = 2 if measured else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        edges = mano_edges.get(len(verts_camera), np.empty((0, 2), dtype=int))
        for a, b in edges:
            pa = tuple(np.clip(verts2d[a], [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
            pb = tuple(np.clip(verts2d[b], [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
            cv2.line(frame, pa, pb, color, max(1, thickness), cv2.LINE_AA)
        for p in verts2d[:: max(1, len(verts2d) // 120)]:
            q = tuple(np.clip(p, [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
            cv2.circle(frame, q, 1, color, -1, cv2.LINE_AA)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, thickness + 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p.astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(p.astype(int)), 4, color, 1, cv2.LINE_AA)


def draw_object_overlay(frame: np.ndarray, frame_ann: dict, sx: float, sy: float) -> None:
    obj = frame_ann.get("object", {})
    if obj.get("bbox_xyxy") is None or obj.get("status") == "not_visible":
        return
    if obj.get("mask_path"):
        mask = cv2.imread(obj["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            color_layer = np.zeros_like(frame)
            color_layer[:, :] = OBJECT_COLOR
            frame[:] = np.where(mask[..., None] > 0, (0.55 * frame + 0.45 * color_layer).astype(np.uint8), frame)
    box = np.asarray(obj["bbox_xyxy"], dtype=float) * np.asarray([sx, sy, sx, sy])
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), OBJECT_COLOR, 2)
    center = (np.asarray(obj["center_xy"], dtype=float) * np.asarray([sx, sy])).astype(int)
    cv2.drawMarker(frame, tuple(center), OBJECT_COLOR, cv2.MARKER_CROSS, 14, 2)


def put_caption(frame: np.ndarray, caption: str, frame_idx: int) -> None:
    words = f"{frame_idx:04d}  {caption}".split()
    lines: list[str] = []
    cur = ""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 2
    max_width = frame.shape[1] - 24
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if width <= max_width or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    lines = lines[:2]
    band_h = 20 + 24 * len(lines)
    cv2.rectangle(frame, (0, frame.shape[0] - band_h), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    y = frame.shape[0] - band_h + 26
    for line in lines:
        cv2.putText(frame, line, (12, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 24


def render_3d_frame(frames: list[dict], index: int, camera_positions: np.ndarray, size: tuple[int, int], mano_edges: dict[int, np.ndarray]) -> np.ndarray:
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], color="black", linewidth=1.2)
    cur = camera_positions[index]
    ax.scatter([cur[0]], [cur[1]], [cur[2]], color="black", s=28)
    frame = frames[index]
    for hand in frame["hands"]:
        joints = np.asarray(hand["joints3d_world_m"], dtype=float)
        verts = hand_vertices(hand, "_world_m")
        color = "tab:green" if hand["side"] == "left" else "tab:orange"
        segs = [[joints[a], joints[b]] for a, b in HAND_EDGES]
        ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=2.0))
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=5)
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color=color, s=1, alpha=0.35)
        edges = mano_edges.get(len(verts), np.empty((0, 2), dtype=int))
        if len(edges):
            mesh_segs = [[verts[a], verts[b]] for a, b in edges[::2]]
            ax.add_collection3d(Line3DCollection(mesh_segs, colors=color, linewidths=0.45, alpha=0.28))
    obj = frame.get("object", {})
    if obj.get("center_world_m") is not None:
        p = np.asarray(obj["center_world_m"], dtype=float)
        ax.scatter([p[0]], [p[1]], [p[2]], color="red", s=38)
    all_pts = [camera_positions[max(0, index - 90) : min(len(frames), index + 90)]]
    for hand in frame["hands"]:
        all_pts.append(np.asarray(hand["joints3d_world_m"], dtype=float))
        all_pts.append(hand_vertices(hand, "_world_m"))
    if obj.get("center_world_m") is not None:
        all_pts.append(np.asarray(obj["center_world_m"], dtype=float)[None])
    pts = np.concatenate(all_pts, axis=0)
    center = pts.mean(axis=0)
    radius = max(0.15, float(np.percentile(np.linalg.norm(pts - center, axis=1), 95)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x m")
    ax.set_ylabel("y m")
    ax.set_zlabel("z m")
    ax.view_init(elev=22, azim=-63)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render_outputs(args: argparse.Namespace, frames: list[dict], render: RenderSpec) -> None:
    cap, info = open_video(args.clip)
    faces = mano_faces(args.mano_right)
    vertex_counts = sorted(
        {
            len(hand_vertices(hand, "_source_camera_m"))
            for frame in frames
            for hand in frame.get("hands", [])
            if "source_intrinsics" in hand
        }
    )
    mano_edges = {count: mano_edges_from_faces(faces, count) for count in vertex_counts}
    overlay_path = args.output_dir / "overlay_mano_object.mp4"
    recon_path = args.output_dir / "reconstruction_3d_world.mp4"
    side_path = args.output_dir / "side_by_side.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay = cv2.VideoWriter(str(overlay_path), fourcc, render.fps, (render.width, render.height))
    recon = cv2.VideoWriter(str(recon_path), fourcc, render.fps, (render.width, render.height))
    side = cv2.VideoWriter(str(side_path), fourcc, render.fps, (render.width * 2, render.height))
    if not overlay.isOpened() or not recon.isOpened() or not side.isOpened():
        raise RuntimeError("failed to open video writers")
    camera_positions = np.asarray([frame["camera"]["position_world_m"] for frame in frames], dtype=float)
    sx, sy = render.width / info.width, render.height / info.height
    try:
        for i, frame_ann in enumerate(tqdm(frames, desc="render")):
            frame = read_video_frame(cap, int(frame_ann["frame_idx"]))
            frame = cv2.resize(frame, (render.width, render.height), interpolation=cv2.INTER_AREA)
            draw_object_overlay(frame, frame_ann, sx, sy)
            draw_hand_overlay(frame, frame_ann, sx, sy, mano_edges)
            put_caption(frame, frame_ann["caption"], frame_ann["frame_idx"])
            panel = render_3d_frame(frames, i, camera_positions, (render.width, render.height), mano_edges)
            overlay.write(frame)
            recon.write(panel)
            side.write(np.concatenate([frame, panel], axis=1))
    finally:
        cap.release()
        overlay.release()
        recon.release()
        side.release()


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.render_only_annotations is not None:
        frames = load_json(args.render_only_annotations)["frames"]
        cap, info = open_video(args.clip)
        cap.release()
        if not frames or any("camera" not in frame for frame in frames):
            raise RuntimeError("render-only annotations must contain camera/world fields")
        render = RenderSpec(args.render_width, int(round(args.render_width * info.height / info.width)), info.fps)
        render_outputs(args, frames, render)
        qc = {
            "status": "ok",
            "mode": "render_only",
            "clip": str(args.clip),
            "processed_frames": len(frames),
            "render": render.__dict__,
            "elapsed_s": time.time() - started,
            "outputs": {
                "overlay": str(args.output_dir / "overlay_mano_object.mp4"),
                "reconstruction_3d": str(args.output_dir / "reconstruction_3d_world.mp4"),
                "side_by_side": str(args.output_dir / "side_by_side.mp4"),
            },
        }
        (args.output_dir / "render_only_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        return qc
    raw_all = load_json(args.wilor_raw)["frames"]
    raw = raw_all
    full_raw_count = len(raw_all)
    if args.frame_start is not None or args.frame_end is not None:
        start = 0 if args.frame_start is None else int(args.frame_start)
        end = len(raw) - 1 if args.frame_end is None else int(args.frame_end)
        if start < 0 or end < start or end >= len(raw):
            raise RuntimeError(f"invalid frame slice {start}:{end} for {len(raw)} raw frames")
        raw = raw[start : end + 1]
        for expected, frame in enumerate(raw, start=start):
            if int(frame["frame_idx"]) != expected:
                raise RuntimeError("raw WiLoR frames are not source-contiguous")
    droid = np.load(args.droid_npz)
    actions = load_actions(args.clip.with_suffix(".json"))
    cap, info = open_video(args.clip)
    cap.release()
    intrinsics = droid["intrinsics_source"].astype(float)
    recon = load_droid_reconstruction(args.droid_reconstruction)
    render = RenderSpec(args.render_width, int(round(args.render_width * info.height / info.width)), info.fps)
    hand_scale_qc = hand_metric_scale_from_raw(raw_all)
    frames, hand_stats = smooth_hands(raw, info.fps, intrinsics, float(hand_scale_qc["wilor_local_to_meters"]))
    droid_scale_qc = estimate_droid_metric_scale_from_raw(
        raw_all,
        intrinsics,
        float(hand_scale_qc["wilor_local_to_meters"]),
        recon,
        (info.width, info.height),
        int(args.max_keyframe_gap),
    )
    T = transform_hands_to_world(frames, args.droid_npz, float(droid_scale_qc["droid_to_meters"]))

    object_meas, object_measure_qc = run_object_masks(args, frames, actions, render)
    semantic_interval = tuple(object_measure_qc["semantic_interval"])
    object_track_qc = fill_object_track(
        frames,
        object_meas,
        info.fps,
        (info.width, info.height),
        args.object_label,
        (int(semantic_interval[0]), int(semantic_interval[1])),
    )
    object_world_qc = attach_object_world(
        frames,
        T,
        intrinsics,
        recon,
        float(droid_scale_qc["droid_to_meters"]),
        (info.width, info.height),
        int(args.max_keyframe_gap),
    )

    annotations_path = args.output_dir / "annotations_v1_full.json"
    annotations_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    render_outputs(args, frames, render)

    qc = {
        "status": "ok",
        "clip": str(args.clip),
        "video": info.__dict__,
        "processed_frames": len(frames),
        "source_frame_range": [int(frames[0]["frame_idx"]), int(frames[-1]["frame_idx"])],
        "camera": {
            "backend": "DROID-SLAM",
            "dense_frames": int(len(T)),
            "source_dense_frames_available": int(len(droid["T_world_camera"])),
            "full_source_timeline": bool(len(T) == len(frames) == info.frame_count and full_raw_count == info.frame_count),
            "trajectory_npz": str(args.droid_npz),
        },
        "hands": hand_stats,
        "scale": {
            "hand": hand_scale_qc,
            "droid": droid_scale_qc,
            "intrinsics_source": intrinsics.astype(float).tolist(),
            "droid_reconstruction": str(args.droid_reconstruction),
            "droid_depth_level": recon["depth_level"],
        },
        "object_measurement": object_measure_qc,
        "object_track": object_track_qc,
        "object_world": object_world_qc,
        "render": render.__dict__,
        "elapsed_s": time.time() - started,
        "outputs": {
            "annotations": str(annotations_path),
            "overlay": str(args.output_dir / "overlay_mano_object.mp4"),
            "reconstruction_3d": str(args.output_dir / "reconstruction_3d_world.mp4"),
            "side_by_side": str(args.output_dir / "side_by_side.mp4"),
        },
    }
    (args.output_dir / "qc_v1_full.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--wilor-raw", type=Path, default=Path("outputs/examples/tomato_v1_full/wilor/wilor_raw.json"))
    parser.add_argument("--droid-npz", type=Path, default=Path("outputs/examples/tomato_v1_full/droid/droid_dense_trajectory.npz"))
    parser.add_argument("--droid-reconstruction", type=Path, default=Path("outputs/examples/tomato_v1_full/droid/droid_keyframe_reconstruction.pth"))
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("checkpoints/sam_vit_b_01ec64.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/examples/tomato_v1_full/fused"))
    parser.add_argument("--mano-right", type=Path, default=DEFAULT_MANO_RIGHT)
    parser.add_argument("--object-label", default="tomato")
    parser.add_argument("--object-stride", type=int, default=1)
    parser.add_argument("--owl-threshold", type=float, default=0.03)
    parser.add_argument("--max-keyframe-gap", type=int, default=15)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-only-annotations", type=Path)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
