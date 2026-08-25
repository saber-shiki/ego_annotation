#!/usr/bin/env python3
"""Build a fixed HOT3D Aria-VRS slice with an official pinhole RGB contract.

This adapter is deliberately split from prediction/runtime code.  It decodes the
native Aria RGB stream, uses the official FISHEYE624 calibration (preferentially
from MPS online calibration), warps to one fixed pinhole image plane, and writes
an RGB-only prediction input.  A separate evaluation sidecar contains the HOT3D
camera/object/hand records needed by the later oracle renderer and evaluator.

The default output is an upright image obtained by a clockwise 90 degree pixel
rotation after fisheye-to-pinhole warping.  The corresponding camera-coordinate
rotation is written into the GT sidecar and the pinhole calibration contract;
there is no silent image-only rotation.

Ground-truth object/hand records are never copied into ``--prediction-input-dir``.
The adapter itself does not run perception or write prediction state.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


STREAM_ID_DEFAULT = "214-1"
ROTATE_CW_R = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_array(values: Iterable[float], shape: tuple[int, ...], label: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise RuntimeError(f"{label} must be finite with shape {shape}, got {arr.shape}")
    return arr


def quat_wxyz_to_R(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0 or not math.isfinite(n):
        raise RuntimeError(f"invalid quaternion: {list(q)}")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def R_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """Stable enough conversion for sidecar serialization."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise RuntimeError(f"invalid rotation matrix {R.shape}")
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(max(1e-15, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(max(1e-15, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1e-15, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return [float(v) for v in q.tolist()]


def make_T(R: np.ndarray, t: Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(list(t), dtype=np.float64)
    return T


def T_to_dict(T: np.ndarray) -> dict[str, Any]:
    T = np.asarray(T, dtype=np.float64)
    return {
        "quaternion_wxyz": R_to_quat_wxyz(T[:3, :3]),
        "translation_xyz": [float(v) for v in T[:3, 3].tolist()],
    }


def csv_pose(row: dict[str, str], prefix: str) -> np.ndarray:
    return make_T(
        quat_wxyz_to_R(
            [
                float(row[f"q_{prefix}_w"]),
                float(row[f"q_{prefix}_x"]),
                float(row[f"q_{prefix}_y"]),
                float(row[f"q_{prefix}_z"]),
            ]
        ),
        [float(row[f"t_{prefix}_{axis}[m]"]) for axis in "xyz"],
    )


def parse_fisheye_params(params: Iterable[float], label: str) -> tuple[float, float, float, np.ndarray]:
    values = [float(x) for x in params]
    if len(values) != 15:
        raise RuntimeError(f"{label}: FISHEYE624 requires 15 parameters, got {len(values)}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label}: non-finite FISHEYE624 parameters")
    return values[0], values[1], values[2], np.asarray(values[3:], dtype=np.float64)


def ovr624_distort(p: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    if coeffs.shape != (12,):
        raise RuntimeError(f"FISHEYE624 requires 12 distortion coefficients, got {coeffs.shape}")
    k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4 = coeffs.tolist()
    r2 = np.sum(p * p, axis=-1, keepdims=True)
    r2 = np.clip(r2, -(np.pi**2), np.pi**2)
    r4 = r2 * r2
    r6 = r2 * r4
    r8 = r4 * r4
    r10 = r4 * r6
    r12 = r6 * r6
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
    uv = p * radial
    x = uv[..., 0].copy()
    y = uv[..., 1].copy()
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2_flat = x2 + y2
    x += 2.0 * p2 * xy + p1 * (r2_flat + 2.0 * x2)
    y += 2.0 * p1 * xy + p2 * (r2_flat + 2.0 * y2)
    r4_flat = r2_flat * r2_flat
    x += s1 * r2_flat + s2 * r4_flat
    y += s3 * r2_flat + s4 * r4_flat
    return np.stack((x, y), axis=-1)


def build_maps(
    width: int,
    height: int,
    src_params: Iterable[float],
    dst_fx: float,
    dst_fy: float,
    dst_cx: float,
    dst_cy: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build native pinhole destination -> online FISHEYE624 source maps."""
    src_f, src_cx, src_cy, coeffs = parse_fisheye_params(src_params, "online calibration")
    px, py = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    x = (px - float(dst_cx)) / float(dst_fx)
    y = (py - float(dst_cy)) / float(dst_fy)
    norm = np.sqrt(x * x + y * y + 1.0)
    vx, vy, vz = x / norm, y / norm, 1.0 / norm
    radius = np.sqrt(vx * vx + vy * vy)
    scale = np.arctan2(radius, vz) / np.maximum(radius, np.finfo(np.float64).tiny)
    projected = np.stack((vx * scale, vy * scale), axis=-1)
    distorted = ovr624_distort(projected, coeffs)
    map_x = (distorted[..., 0] * src_f + src_cx).astype(np.float32)
    map_y = (distorted[..., 1] * src_f + src_cy).astype(np.float32)
    valid = (
        (map_x >= -0.5)
        & (map_x <= width - 0.5)
        & (map_y >= -0.5)
        & (map_y <= height - 0.5)
    )
    return map_x, map_y, float(np.mean(valid))


def load_static_camera(sequence_root: Path, stream_id: str) -> dict[str, Any]:
    payload = json.loads((sequence_root / "camera_models.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("camera_models.json must be a list")
    for row in payload:
        if str(row.get("stream_id")) != stream_id:
            continue
        model = str(row.get("projectionModelType"))
        if model != "CameraModelType.FISHEYE624":
            raise RuntimeError(f"{stream_id}: expected FISHEYE624, got {model}")
        return row
    raise RuntimeError(f"stream {stream_id} not found in camera_models.json")


def load_online_camera_calibrations(
    path: Path, label: str, selected_device_ns: list[int]
) -> tuple[list[dict[str, Any]], list[float]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid online calibration JSON at line {line_no}") from exc
            ts_us = row.get("tracking_timestamp_us")
            if ts_us is None:
                continue
            for cam in row.get("CameraCalibrations") or []:
                if str(cam.get("Label")) != label:
                    continue
                proj = cam.get("Projection") or {}
                params = proj.get("Params")
                if str(proj.get("Name")) not in {"FisheyeRadTanThinPrism", "FISHEYE624"}:
                    continue
                if params is None:
                    continue
                Tdc = cam.get("T_Device_Camera") or {}
                q = Tdc.get("UnitQuaternion")
                t = Tdc.get("Translation")
                if not isinstance(q, list) or len(q) != 2 or not isinstance(t, list):
                    continue
                records.append(
                    {
                        "tracking_timestamp_us": int(ts_us),
                        "projection_params": [float(v) for v in params],
                        "T_device_camera": make_T(quat_wxyz_to_R([q[0], *q[1]]), t),
                        "line_no": line_no,
                    }
                )
                break
    if not records:
        return [], []
    records.sort(key=lambda x: x["tracking_timestamp_us"])
    # Keep one record per timestamp.  The last duplicate is the complete line.
    dedup: dict[int, dict[str, Any]] = {}
    for r in records:
        dedup[int(r["tracking_timestamp_us"])] = r
    records = [dedup[k] for k in sorted(dedup)]
    times = [int(r["tracking_timestamp_us"]) for r in records]
    selected: list[dict[str, Any]] = []
    deltas: list[float] = []
    for ns in selected_device_ns:
        target_us = float(ns) / 1000.0
        pos = bisect.bisect_left(times, int(round(target_us)))
        candidates = []
        if pos < len(records):
            candidates.append(records[pos])
        if pos > 0:
            candidates.append(records[pos - 1])
        best = min(candidates, key=lambda r: abs(float(r["tracking_timestamp_us"]) - target_us))
        delta_us = float(best["tracking_timestamp_us"]) - target_us
        selected.append(best)
        deltas.append(delta_us)
    return selected, deltas


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def index_by_timestamp(rows: Iterable[dict[str, str]], timestamp_key: str = "timestamp[ns]") -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if timestamp_key in row and row[timestamp_key] not in (None, ""):
            out[int(row[timestamp_key])].append(row)
    return out


def nearest_rows(index: dict[int, list[dict[str, str]]], timestamp_ns: int, tolerance_ns: int, label: str) -> tuple[list[dict[str, str]], int]:
    if timestamp_ns in index:
        return index[timestamp_ns], 0
    if not index:
        return [], 0
    keys = sorted(index)
    pos = bisect.bisect_left(keys, timestamp_ns)
    candidates = []
    if pos < len(keys):
        candidates.append(keys[pos])
    if pos > 0:
        candidates.append(keys[pos - 1])
    key = min(candidates, key=lambda x: abs(x - timestamp_ns))
    delta = abs(key - timestamp_ns)
    if delta > tolerance_ns:
        raise RuntimeError(f"{label}: no timestamp within {tolerance_ns} ns of {timestamp_ns}; nearest delta={delta}")
    return index[key], key - timestamp_ns


def rotmat_to_axis_angle(R: np.ndarray) -> list[float]:
    # Rodrigues logarithm, avoiding a scipy dependency in the adapter.
    R = np.asarray(R, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(c)
    if theta < 1e-9:
        return [0.0, 0.0, 0.0]
    s = 2.0 * math.sin(theta)
    if abs(s) < 1e-8:
        # This branch is not expected for the selected data; use a diagonal-safe form.
        axis = np.sqrt(np.maximum(0.0, (np.diag(R) + 1.0) / 2.0))
        axis /= max(np.linalg.norm(axis), 1e-12)
    else:
        axis = np.asarray([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    return [float(v * theta) for v in axis.tolist()]


def load_mano_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["timestamp_ns"])] = row.get("hand_poses") or {}
    return out


def load_instance_metadata(assets_root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((assets_root / "instance.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("assets/instance.json must be an object")
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}


def make_camera_calibration(
    static: dict[str, Any],
    online: dict[str, Any],
    output_K: np.ndarray,
    output_T_device_camera: np.ndarray,
    native_T_device_camera: np.ndarray,
    width: int,
    height: int,
    orientation: str,
) -> dict[str, Any]:
    raw_params = [float(v) for v in online["projection_params"]]
    return {
        "T_device_from_camera": T_to_dict(output_T_device_camera),
        "T_device_from_camera_native": T_to_dict(native_T_device_camera),
        "image_height": int(height),
        "image_width": int(width),
        "label": "camera-rgb",
        "max_solid_angle": float(static.get("maxSolidAngle", 1.0)),
        "projection_model_type": "CameraModelType.PINHOLE",
        "projection_params": [
            float(output_K[0, 0]),
            float(output_K[1, 1]),
            float(output_K[0, 2]),
            float(output_K[1, 2]),
        ],
        "source_projection_model_type": "CameraModelType.FISHEYE624",
        "source_projection_params_online": raw_params,
        "source_projection_params_static": [float(v) for v in static["projectionParams"]],
        "serial_number": str(static.get("serialNumber", "")),
        "stream_id": STREAM_ID_DEFAULT,
        "image_orientation": orientation,
    }


def build_gt_frame(
    frame_idx: int,
    source_frame_idx: int,
    timestamp_ns: int,
    device_timestamp_ns: int,
    camera: dict[str, Any],
    object_rows: list[dict[str, str]],
    hand_box_rows: list[dict[str, str]],
    mano_poses: dict[str, Any],
    instance_metadata: dict[str, dict[str, Any]],
    nearest_deltas: dict[str, int],
) -> dict[str, Any]:
    objects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in object_rows:
        uid = str(row["object_uid"])
        meta = instance_metadata.get(uid, {})
        bop_id = str(meta.get("instance_bop_id", uid))
        pose = csv_pose(row, "wo")
        box_values: list[float] | None = None
        visibility: float | None = None
        try:
            candidate_box = [
                float(row["x_min[pixel]"]),
                float(row["y_min[pixel]"]),
                float(row["x_max[pixel]"]),
                float(row["y_max[pixel]"]),
            ]
            if np.isfinite(candidate_box).all():
                box_values = candidate_box
            candidate_visibility = float(row["visibility_ratio[%]"])
            if math.isfinite(candidate_visibility):
                visibility = candidate_visibility / 100.0 if candidate_visibility > 1.0 else candidate_visibility
        except (KeyError, TypeError, ValueError):
            pass
        entry: dict[str, Any] = {
            "T_world_from_object": T_to_dict(pose),
            "object_bop_id": bop_id,
            "object_name": meta.get("instance_name"),
            "object_uid": uid,
            "boxes_amodal_source_fisheye": {STREAM_ID_DEFAULT: box_values} if box_values is not None else {},
            "boxes_coordinate_frame": "source_fisheye624_pixels; not a pinhole box",
            "visibilities_modeled": {STREAM_ID_DEFAULT: visibility} if visibility is not None else {},
            "masks_amodal": {},
            "masks_modal": {},
            "mask_source": "not_exported_by_full_VRS_slice_adapter",
        }
        objects[bop_id].append(entry)

    hand_by_side: dict[str, dict[str, Any]] = {}
    boxes_by_side: dict[str, dict[str, str]] = {}
    for row in hand_box_rows:
        boxes_by_side[str(row["hand_index"])] = row
    for index, side in (("0", "left"), ("1", "right")):
        pose_row = mano_poses.get(index)
        box_row = boxes_by_side.get(index)
        if pose_row is None and box_row is None:
            continue
        hand_box: dict[str, list[float]] = {}
        hand_visibility: dict[str, float] = {}
        if box_row is not None:
            try:
                candidate_box = [
                    float(box_row["x_min[pixel]"]),
                    float(box_row["y_min[pixel]"]),
                    float(box_row["x_max[pixel]"]),
                    float(box_row["y_max[pixel]"]),
                ]
                if np.isfinite(candidate_box).all():
                    hand_box[STREAM_ID_DEFAULT] = candidate_box
            except (KeyError, TypeError, ValueError):
                pass
            try:
                candidate_visibility = float(box_row["visibility_ratio[%]"])
                if math.isfinite(candidate_visibility):
                    hand_visibility[STREAM_ID_DEFAULT] = candidate_visibility / 100.0 if candidate_visibility > 1.0 else candidate_visibility
            except (KeyError, TypeError, ValueError):
                pass
        item: dict[str, Any] = {
            "boxes_amodal_source_fisheye": hand_box,
            "boxes_coordinate_frame": "source_fisheye624_pixels; not a pinhole box",
            "visibilities_modeled": hand_visibility,
        }
        if pose_row is not None:
            wrist = pose_row.get("wrist_xform") or {}
            q = wrist.get("q_wxyz")
            t = wrist.get("t_xyz")
            if isinstance(q, list) and len(q) == 4 and isinstance(t, list) and len(t) == 3:
                Tw = make_T(quat_wxyz_to_R(q), t)
                item["mano_pose"] = {
                    "thetas": [float(v) for v in pose_row.get("pose", [])],
                    "wrist_xform": rotmat_to_axis_angle(Tw[:3, :3]) + [float(v) for v in Tw[:3, 3].tolist()],
                    "T_world_from_wrist": T_to_dict(Tw),
                    "betas": [float(v) for v in pose_row.get("betas", [])],
                }
        hand_by_side[side] = item

    payload = {
        "cameras.json": {STREAM_ID_DEFAULT: camera},
        "objects.json": dict(objects),
        "hands.json": hand_by_side,
        "info.json": {
            "ref_timestamp_ns": int(timestamp_ns),
            "device_timestamp_ns": int(device_timestamp_ns),
            "source_frame_idx": int(source_frame_idx),
            "frame_idx": int(frame_idx),
            "gt_timestamp_match_deltas_ns": {k: int(v) for k, v in nearest_deltas.items()},
        },
    }
    return {"frame_idx": int(frame_idx), "source_frame_idx": int(source_frame_idx), "timestamp_ns": int(timestamp_ns), "json": payload}


def make_review(images: list[np.ndarray], labels: list[str], output: Path) -> None:
    if not images:
        return
    tiles: list[np.ndarray] = []
    for image, label in zip(images, labels):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (4, 4), (bgr.shape[1] - 5, bgr.shape[0] - 5), (0, 230, 255), 5)
        cv2.putText(bgr, label, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 3, cv2.LINE_AA)
        target_w = 480
        bgr = cv2.resize(bgr, (target_w, int(round(bgr.shape[0] * target_w / bgr.shape[1]))), interpolation=cv2.INTER_AREA)
        tiles.append(bgr)
    cols = min(3, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())
    lines = [np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
    sheet = np.vstack(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def import_aria():
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.sensor_data import TimeDomain
        from projectaria_tools.core.stream_id import StreamId
    except Exception as exc:  # pragma: no cover - dependency is supplied by uv at execution
        raise RuntimeError(
            "projectaria-tools is required for VRS extraction; run with "
            "uv --isolated --with projectaria-tools==1.5.1"
        ) from exc
    return data_provider, TimeDomain, StreamId


def build(args: argparse.Namespace) -> dict[str, Any]:
    sequence_root = args.sequence_root.resolve()
    stream_id = args.stream_id
    if stream_id != STREAM_ID_DEFAULT:
        raise RuntimeError("this experiment contract is fixed to Aria RGB stream 214-1")
    if args.end_frame < args.start_frame:
        raise RuntimeError("end frame must be >= start frame")
    frame_indices = list(range(int(args.start_frame), int(args.end_frame) + 1))
    if args.frame_indices:
        frame_indices = [int(v) for v in args.frame_indices]
    if len(set(frame_indices)) != len(frame_indices) or min(frame_indices) < 0:
        raise RuntimeError("frame indices must be unique and non-negative")

    adapter_root = args.output_root.resolve()
    prediction_dir = args.prediction_input_dir.resolve() if args.prediction_input_dir else None
    if adapter_root.exists():
        if not args.replace:
            raise FileExistsError(f"output root exists; use --replace explicitly: {adapter_root}")
        shutil.rmtree(adapter_root)
    if prediction_dir is not None and prediction_dir.exists():
        if not args.replace:
            raise FileExistsError(f"prediction input exists; use --replace explicitly: {prediction_dir}")
        shutil.rmtree(prediction_dir)
    (adapter_root / "input" / "raw_frame_manifest" / "rgb").mkdir(parents=True, exist_ok=True)
    (adapter_root / "evaluation" / "hot3d_gt").mkdir(parents=True, exist_ok=True)
    (adapter_root / "state" / "calibration").mkdir(parents=True, exist_ok=True)

    data_provider, TimeDomain, StreamId = import_aria()
    vrs_path = sequence_root / "recording.vrs"
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    stream = StreamId(stream_id)
    all_timecode = np.asarray(provider.get_timestamps_ns(stream, TimeDomain.TIME_CODE), dtype=np.int64)
    all_device = np.asarray(provider.get_timestamps_ns(stream, TimeDomain.DEVICE_TIME), dtype=np.int64)
    if all_timecode.shape != all_device.shape:
        raise RuntimeError("VRS timecode/device timestamp arrays have different lengths")
    if not frame_indices or max(frame_indices) >= len(all_timecode):
        raise RuntimeError(f"requested frame index outside VRS stream: max={max(frame_indices)} count={len(all_timecode)}")
    # Preserve requested order for explicit sparse probes, but the normal slice is sorted.
    if frame_indices != sorted(frame_indices):
        raise RuntimeError("frame indices must be ascending to preserve video chronology")
    timestamps = [int(all_timecode[i]) for i in frame_indices]
    device_timestamps = [int(all_device[i]) for i in frame_indices]

    static_camera = load_static_camera(sequence_root, stream_id)
    static_params = [float(v) for v in static_camera["projectionParams"]]
    static_f, static_cx, static_cy, static_coeffs = parse_fisheye_params(static_params, "camera_models.json")
    if int(static_camera["imageWidth"]) != int(static_camera["imageHeight"]):
        raise RuntimeError("the fixed Aria RGB contract currently requires a square source image")
    width = int(static_camera["imageWidth"])
    height = int(static_camera["imageHeight"])

    online_path = sequence_root / "mps" / "slam" / "online_calibration.jsonl"
    online_records, online_deltas_us = load_online_camera_calibrations(
        online_path, "camera-rgb", device_timestamps
    ) if online_path.exists() else ([], [])
    if not online_records:
        # This is an explicit fallback to the official static camera_models.json, not UniDepth.
        online_records = [
            {
                "tracking_timestamp_us": int(round(ns / 1000.0)),
                "projection_params": static_params,
                "T_device_camera": make_T(
                    quat_wxyz_to_R(
                        [
                            float(static_camera["T_Device_Camera"]["quaternion_wxyz"][0]),
                            *[float(v) for v in static_camera["T_Device_Camera"]["quaternion_wxyz"][1:]],
                        ]
                    ),
                    static_camera["T_Device_Camera"]["translation_xyz"],
                ),
                "line_no": None,
            }
            for ns in device_timestamps
        ]
        online_deltas_us = [0.0] * len(online_records)
        calibration_source = "HOT3D camera_models.json static official calibration (online file unavailable)"
    else:
        calibration_source = "HOT3D MPS online_calibration.jsonl official calibration"
        if max(abs(v) for v in online_deltas_us) > float(args.max_online_delta_us):
            raise RuntimeError(
                f"online calibration timestamp mismatch exceeds limit: max={max(abs(v) for v in online_deltas_us):.3f} us"
            )

    source_params = np.asarray([r["projection_params"] for r in online_records], dtype=np.float64)
    # A single destination plane is essential: both branches must receive one identical K.
    median_f = float(np.median(source_params[:, 0])) * float(args.focal_scale)
    median_cx = float(np.median(source_params[:, 1]))
    median_cy = float(np.median(source_params[:, 2]))
    native_K = np.asarray([[median_f, 0.0, median_cx], [0.0, median_f, median_cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if args.orientation == "upright-cw":
        output_K = np.asarray(
            [[median_f, 0.0, float(width - 1) - median_cy], [0.0, median_f, median_cx], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        camera_rotation_output_from_native = ROTATE_CW_R
    else:
        output_K = native_K.copy()
        camera_rotation_output_from_native = np.eye(3, dtype=np.float64)

    # Static source transform is retained as a sanity reference; online transforms are used per frame.
    static_Tdc = make_T(
        quat_wxyz_to_R(static_camera["T_Device_Camera"]["quaternion_wxyz"]),
        static_camera["T_Device_Camera"]["translation_xyz"],
    )
    map_cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray, float]] = {}
    review_images: list[np.ndarray] = []
    review_labels: list[str] = []
    rows: list[dict[str, Any]] = []
    camera_records: list[dict[str, Any]] = []
    video_path = adapter_root / "input" / "input.mp4"
    writer: cv2.VideoWriter | None = None
    frame_set_for_review = set(args.review_frames or [])
    try:
        for local_idx, source_idx in enumerate(frame_indices):
            image_data = provider.get_image_data_by_index(stream, int(source_idx))
            if image_data is None or not image_data[0].is_valid():
                raise RuntimeError(f"VRS image missing/invalid at source frame {source_idx}")
            rgb = np.asarray(image_data[0].to_numpy_array())
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[..., None], 3, axis=2)
            if rgb.shape[:2] != (height, width) or rgb.shape[2] != 3:
                raise RuntimeError(f"unexpected RGB shape at frame {source_idx}: {rgb.shape}")
            online = online_records[local_idx]
            key = tuple(round(float(v), 9) for v in online["projection_params"])
            if key not in map_cache:
                map_cache[key] = build_maps(
                    width,
                    height,
                    online["projection_params"],
                    native_K[0, 0],
                    native_K[1, 1],
                    native_K[0, 2],
                    native_K[1, 2],
                )
            map_x, map_y, valid_fraction = map_cache[key]
            native_pinhole_rgb = cv2.remap(
                rgb,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            if args.orientation == "upright-cw":
                output_rgb = np.rot90(native_pinhole_rgb, k=3).copy()
            else:
                output_rgb = native_pinhole_rgb
            if writer is None:
                oh, ow = output_rgb.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (int(ow), int(oh))
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer {video_path}")
            bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            frame_path = adapter_root / "input" / "raw_frame_manifest" / "rgb" / f"{local_idx:06d}.jpg"
            if not cv2.imwrite(str(frame_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
                raise RuntimeError(f"failed to write {frame_path}")
            if local_idx in frame_set_for_review:
                review_images.append(output_rgb.copy())
                review_labels.append(f"local={local_idx} source={source_idx}")
            native_Tdc = np.asarray(online["T_device_camera"], dtype=np.float64)
            output_Tdc = native_Tdc @ make_T(camera_rotation_output_from_native.T, [0.0, 0.0, 0.0])
            rows.append(
                {
                    "frame_idx": int(local_idx),
                    "source_frame_idx": int(source_idx),
                    "source_time_domain": "TIME_CODE",
                    "timestamp_ns": int(timestamps[local_idx]),
                    "device_timestamp_ns": int(device_timestamps[local_idx]),
                    "time_s": float(local_idx / float(args.fps)),
                    "rgb": str(frame_path.relative_to(adapter_root)),
                    "raw_frame_path": str(frame_path),
                    "source_width": int(width),
                    "source_height": int(height),
                    "manifest_width": int(output_rgb.shape[1]),
                    "manifest_height": int(output_rgb.shape[0]),
                    "intrinsics_fx_fy_cx_cy": [float(output_K[0, 0]), float(output_K[1, 1]), float(output_K[0, 2]), float(output_K[1, 2])],
                    "intrinsics_source": "HOT3D official FISHEYE624 online calibration -> fixed pinhole contract",
                    "official_online_calibration_timestamp_delta_us": float(online_deltas_us[local_idx]),
                    "source_fisheye_projection_params": [float(v) for v in online["projection_params"]],
                    "pinhole_warp_valid_fraction": float(valid_fraction),
                    "image_orientation": args.orientation,
                    "camera_rotation_output_from_native": camera_rotation_output_from_native.tolist(),
                }
            )
            camera_records.append(
                {
                    "frame_idx": int(local_idx),
                    "source_frame_idx": int(source_idx),
                    "timestamp_ns": int(timestamps[local_idx]),
                    "device_timestamp_ns": int(device_timestamps[local_idx]),
                    "T_device_camera_native": T_to_dict(native_Tdc),
                    "T_device_camera_output": T_to_dict(output_Tdc),
                    "online_calibration_tracking_timestamp_us": int(online["tracking_timestamp_us"]),
                    "online_calibration_line_no": online.get("line_no"),
                }
            )
    finally:
        if writer is not None:
            writer.release()

    if len(rows) != len(frame_indices):
        raise RuntimeError(f"wrote {len(rows)} rows for {len(frame_indices)} requested frames")
    make_review(review_images, review_labels, adapter_root / "evaluation" / "hot3d_vrs_slice_adapter_review.jpg")

    manifest = {
        "schema": "v19_hot3d_full_vrs_pinhole_raw_frame_manifest_v1",
        "source_dataset": "HOT3D full Aria recording.vrs",
        "source_sequence_id": sequence_root.name,
        "source_stream_id": stream_id,
        "source_vrs": str(vrs_path),
        "frame_count": len(rows),
        "fps": float(args.fps),
        "width": int(rows[0]["manifest_width"]),
        "height": int(rows[0]["manifest_height"]),
        "video": str(video_path),
        "frames": rows,
        "causal_boundary": "Prediction input uses RGB and official camera calibration only. HOT3D object/hand/pose records are evaluation-only in a separate sidecar.",
    }
    write_json(adapter_root / "input" / "raw_frame_manifest" / "manifest.json", manifest)

    # Build evaluation-only GT records from the exact RGB timecodes.
    instance_metadata = load_instance_metadata(args.assets_root.resolve())
    dynamic_index = index_by_timestamp(load_csv_rows(sequence_root / "dynamic_objects.csv"))
    headset_index = index_by_timestamp(load_csv_rows(sequence_root / "headset_trajectory.csv"))
    object_box_index = index_by_timestamp(
        [r for r in load_csv_rows(sequence_root / "box2d_objects.csv") if str(r.get("stream_id")) == stream_id]
    )
    hand_box_index = index_by_timestamp(
        [r for r in load_csv_rows(sequence_root / "box2d_hands.csv") if str(r.get("stream_id")) == stream_id]
    )
    mano_index = load_mano_jsonl(sequence_root / "mano_hand_pose_trajectory.jsonl")
    gt_frames: list[dict[str, Any]] = []
    gt_match_deltas: defaultdict[str, list[int]] = defaultdict(list)
    for local_idx, row in enumerate(rows):
        ts = int(row["timestamp_ns"])
        headset_rows, headset_delta = nearest_rows(headset_index, ts, int(args.gt_tolerance_ns), "headset trajectory")
        if not headset_rows:
            raise RuntimeError(f"missing headset pose at {ts}")
        device_pose = csv_pose(headset_rows[0], "wo")
        online = online_records[local_idx]
        native_Tdc = np.asarray(online["T_device_camera"], dtype=np.float64)
        output_Tdc = native_Tdc @ make_T(camera_rotation_output_from_native.T, [0.0, 0.0, 0.0])
        T_world_camera = device_pose @ output_Tdc
        camera_payload = make_camera_calibration(
            static_camera,
            online,
            output_K,
            output_Tdc,
            native_Tdc,
            int(rows[0]["manifest_width"]),
            int(rows[0]["manifest_height"]),
            args.orientation,
        )
        camera_payload["T_world_from_camera"] = T_to_dict(T_world_camera)
        camera_payload["T_world_from_camera_native"] = T_to_dict(device_pose @ native_Tdc)
        camera_payload["T_world_from_device"] = T_to_dict(device_pose)
        object_rows, object_delta = nearest_rows(dynamic_index, ts, int(args.gt_tolerance_ns), "dynamic object trajectory")
        box_rows, box_delta = nearest_rows(object_box_index, ts, int(args.gt_tolerance_ns), "object boxes")
        hand_box_rows, hand_box_delta = nearest_rows(hand_box_index, ts, int(args.gt_tolerance_ns), "hand boxes")
        # The CSV box rows contain one row per object/hand; pass the rows that match this timestamp.
        if object_delta != 0:
            object_rows = [r for r in object_rows]
        if box_delta != 0:
            box_rows = [r for r in box_rows]
        by_uid_box = {str(r["object_uid"]): r for r in box_rows if "object_uid" in r}
        object_rows_with_boxes: list[dict[str, str]] = []
        for obj in object_rows:
            obj_copy = dict(obj)
            if str(obj["object_uid"]) in by_uid_box:
                obj_copy.update(by_uid_box[str(obj["object_uid"])])
            else:
                # Keep pose rows even when a box annotation is absent; evaluator uses pose.
                obj_copy.update({
                    "x_min[pixel]": "nan",
                    "x_max[pixel]": "nan",
                    "y_min[pixel]": "nan",
                    "y_max[pixel]": "nan",
                    "visibility_ratio[%]": "nan",
                })
            object_rows_with_boxes.append(obj_copy)
        mano_poses = mano_index.get(ts)
        if mano_poses is None:
            # Exact source timestamps are expected; nearest MANO is not silently substituted.
            nearest_mano, mano_delta = nearest_rows({k: [ {"_payload": v} ] for k, v in mano_index.items()}, ts, int(args.gt_tolerance_ns), "MANO trajectory")
            mano_poses = nearest_mano[0]["_payload"] if nearest_mano else {}
            gt_match_deltas["mano"].append(int(mano_delta))
        nearest_deltas = {
            "headset_ns": int(headset_delta),
            "dynamic_objects_ns": int(object_delta),
            "object_boxes_ns": int(box_delta),
            "hand_boxes_ns": int(hand_box_delta),
        }
        gt_match_deltas["headset_ns"].append(int(headset_delta))
        gt_match_deltas["dynamic_objects_ns"].append(int(object_delta))
        gt_match_deltas["object_boxes_ns"].append(int(box_delta))
        gt_match_deltas["hand_boxes_ns"].append(int(hand_box_delta))
        gt_frames.append(
            build_gt_frame(
                local_idx,
                int(row["source_frame_idx"]),
                ts,
                int(row["device_timestamp_ns"]),
                camera_payload,
                object_rows_with_boxes,
                hand_box_rows,
                mano_poses or {},
                instance_metadata,
                nearest_deltas,
            )
        )

    gt = {
        "schema": "v19_hot3d_full_vrs_ground_truth_sidecar_v1",
        "source_dataset": "HOT3D full Aria recording.vrs and released annotation CSV/JSONL",
        "source_sequence": str(sequence_root),
        "source_stream_id": stream_id,
        "frame_count": len(gt_frames),
        "frame_indices_source_vrs": frame_indices,
        "object_pose_formula": "T_world_object from dynamic_objects.csv; renderer derives T_camera_object = inv(T_world_camera) @ T_world_object",
        "camera_pose_formula": "T_world_camera = T_world_device @ T_device_camera_output; output camera includes explicit upright pixel rotation",
        "causal_boundary": "Evaluation/oracle-only. This sidecar must never be copied into prediction input, measurements, or state.",
        "frames": gt_frames,
        "timestamp_match_delta_summary_ns": {
            key: {
                "max_abs": int(max(abs(v) for v in values)) if values else 0,
                "unique": sorted(set(int(v) for v in values)),
            }
            for key, values in gt_match_deltas.items()
        },
    }
    gt_path = adapter_root / "evaluation" / "hot3d_gt" / "hot3d_vrs_gt_sidecar.json"
    write_json(gt_path, gt)

    contract = {
        "schema": "v19_hot3d_vrs_pinhole_camera_calibration_contract_v2",
        "method": "build_v19_hot3d_vrs_slice_adapter",
        "claim_scope": "prediction-side sensor camera adaptation only; no HOT3D GT object/hand state is provided to runtime",
        "source": calibration_source,
        "source_sequence_id": sequence_root.name,
        "stream_id": stream_id,
        "source_projection_model_type": "CameraModelType.FISHEYE624",
        "source_static_projection_params": static_params,
        "focal_scale": float(args.focal_scale),
        "image_width": int(rows[0]["manifest_width"]),
        "image_height": int(rows[0]["manifest_height"]),
        "K": output_K.tolist(),
        "intrinsics_fx_fy_cx_cy": [float(output_K[0, 0]), float(output_K[1, 1]), float(output_K[0, 2]), float(output_K[1, 2])],
        "focal_px": float(output_K[0, 0]),
        "focal_geom_px": float(output_K[0, 0]),
        "intrinsics_source": "HOT3D official FISHEYE624 online calibration median over selected RGB frames, then fixed pinhole output plane",
        "intrinsics_vary_per_frame": False,
        "source_online_calibration_frame_count": len(online_records),
        "source_online_calibration_delta_us": {
            "min": float(min(online_deltas_us)),
            "median": float(np.median(online_deltas_us)),
            "max": float(max(online_deltas_us)),
            "max_abs": float(max(abs(v) for v in online_deltas_us)),
        },
        "image_orientation": args.orientation,
        "orientation_contract": {
            "native_image_shape_hw": [height, width],
            "output_image_shape_hw": [int(rows[0]["manifest_height"]), int(rows[0]["manifest_width"])],
            "pixel_mapping": "u_out = H-1-v_native_pinhole; v_out = u_native_pinhole" if args.orientation == "upright-cw" else "identity",
            "R_output_camera_from_native_camera": camera_rotation_output_from_native.tolist(),
            "T_world_camera_output": "T_world_camera_native @ inv(R_output_camera_from_native_camera) in homogeneous form",
        },
    }
    contract_path = adapter_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
    write_json(contract_path, contract)

    provenance = {
        "schema": "v19_hot3d_vrs_prediction_input_provenance_v1",
        "case_id": args.case_id,
        "source_dataset": "HOT3D full Aria VRS",
        "source_sequence_id": sequence_root.name,
        "source_stream_id": stream_id,
        "source_frame_idx_start": int(frame_indices[0]),
        "source_frame_idx_end": int(frame_indices[-1]),
        "frame_count": len(rows),
        "fps": float(args.fps),
        "width": int(rows[0]["manifest_width"]),
        "height": int(rows[0]["manifest_height"]),
        "rgb_sha256": sha256_file(video_path),
        "camera_adapter": "official FISHEYE624 -> fixed pinhole; explicit clockwise 90-degree image/camera transform",
        "calibration_contract_filename": "v19_camera_calibration_contract.json",
        "ground_truth_boundary": "No HOT3D object/hand/object-pose annotations are present in this prediction input directory. The evaluator-only sidecar lives outside it.",
    }
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(video_path, prediction_dir / "input.mp4")
        write_json(prediction_dir / "INPUT_PROVENANCE.json", provenance)
        shutil.copyfile(contract_path, prediction_dir / "v19_camera_calibration_contract.json")
        if sha256_file(prediction_dir / "input.mp4") != provenance["rgb_sha256"]:
            raise RuntimeError("prediction input copy hash mismatch")

    report = {
        "status": "ok",
        "method": "build_v19_hot3d_vrs_slice_adapter",
        "claim_scope": contract["claim_scope"],
        "case_id": args.case_id,
        "sequence_root": str(sequence_root),
        "output_root": str(adapter_root),
        "prediction_input_dir": str(prediction_dir) if prediction_dir else None,
        "input_video": str(video_path),
        "ground_truth_sidecar": str(gt_path),
        "calibration_contract": str(contract_path),
        "frame_count": len(rows),
        "source_frame_idx_range": [int(frame_indices[0]), int(frame_indices[-1])],
        "timestamp_ns_range": [int(timestamps[0]), int(timestamps[-1])],
        "device_timestamp_ns_range": [int(device_timestamps[0]), int(device_timestamps[-1])],
        "width": int(rows[0]["manifest_width"]),
        "height": int(rows[0]["manifest_height"]),
        "fps": float(args.fps),
        "orientation": args.orientation,
        "fixed_intrinsics_fx_fy_cx_cy": contract["intrinsics_fx_fy_cx_cy"],
        "source_online_calibration_count": len(online_records),
        "map_cache_count": len(map_cache),
        "pinhole_warp_valid_fraction": {
            "min": float(min(r["pinhole_warp_valid_fraction"] for r in rows)),
            "median": float(np.median([r["pinhole_warp_valid_fraction"] for r in rows])),
            "max": float(max(r["pinhole_warp_valid_fraction"] for r in rows)),
        },
        "rgb_sha256": provenance["rgb_sha256"],
        "prediction_input_files": sorted(p.name for p in prediction_dir.iterdir()) if prediction_dir else [],
        "causal_boundary": provenance["ground_truth_boundary"],
    }
    write_json(adapter_root / "evaluation" / "hot3d_vrs_slice_adapter_report.json", report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-root", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--prediction-input-dir", type=Path, default=None)
    p.add_argument("--case-id", required=True)
    p.add_argument("--stream-id", default=STREAM_ID_DEFAULT)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=149)
    p.add_argument("--frame-indices", type=int, nargs="*", default=None)
    p.add_argument("--orientation", choices=["upright-cw", "native"], default="upright-cw")
    p.add_argument("--focal-scale", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--review-frames", type=int, nargs="*", default=[0, 37, 75, 112, 149])
    p.add_argument("--max-online-delta-us", type=float, default=1000.0)
    p.add_argument("--gt-tolerance-ns", type=int, default=2_000_000)
    p.add_argument("--replace", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    try:
        build(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
