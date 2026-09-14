"""Build a V19-owned raw-frame manifest directly from an input video.

This is the first executable component of the V19 pipeline. It extracts the
source timeline into one row per source frame under the current run root. It has
no dependency on V16/V17/V18 cached artifacts.
"""
from __future__ import annotations
from utils.data import write_json_payload
from utils.data import read_json_mapping
from utils.data import write_json_utf8
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
from utils.state import sha as sha256_file

@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return float(self.frame_count / self.fps) if self.fps > 0 else 0.0

def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'failed to open input video: {path}')
    info = VideoInfo(fps=float(cap.get(cv2.CAP_PROP_FPS)), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if info.fps <= 0.0 or info.width <= 0 or info.height <= 0 or (info.frame_count <= 0):
        cap.release()
        raise RuntimeError(f'invalid video metadata for {path}: {info}')
    return (cap, info)

def even_height_for_width(width: int, source_width: int, source_height: int) -> int:
    height = int(round(float(width) * float(source_height) / float(source_width)))
    if height % 2:
        height += 1
    return max(2, height)

def build(args: argparse.Namespace) -> dict:
    started = time.time()
    video = args.video.resolve()
    if not video.exists():
        raise FileNotFoundError(f'missing input video: {video}')
    (cap, info) = open_video(video)
    if args.render_width is None or int(args.render_width) <= 0:
        manifest_width = int(info.width)
        manifest_height = int(info.height)
    else:
        manifest_width = int(args.render_width)
        manifest_height = even_height_for_width(manifest_width, info.width, info.height)
    frame_start = 0 if args.frame_start is None else int(args.frame_start)
    frame_end = int(info.frame_count) - 1 if args.frame_end is None else int(args.frame_end)
    if frame_start < 0 or frame_end < frame_start or frame_end >= int(info.frame_count):
        cap.release()
        raise RuntimeError(f'invalid frame range {frame_start}:{frame_end} for {info.frame_count}-frame video')
    output_dir = args.output_dir.resolve()
    rgb_dir = output_dir / 'rgb'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    extracted = 0
    try:
        for frame_idx in range(int(info.frame_count)):
            (ok, frame_bgr) = cap.read()
            if not ok:
                raise RuntimeError(f'video ended at frame {frame_idx}, expected {info.frame_count}')
            if frame_idx < frame_start or frame_idx > frame_end:
                continue
            if manifest_width != int(info.width) or manifest_height != int(info.height):
                frame_bgr = cv2.resize(frame_bgr, (manifest_width, manifest_height), interpolation=cv2.INTER_AREA)
            path = rgb_dir / f'{frame_idx:06d}.jpg'
            if not cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
                raise RuntimeError(f'failed to write frame image: {path}')
            frames.append({'index': int(frame_idx), 'frame_idx': int(frame_idx), 'time_s': float(frame_idx / info.fps), 'rgb': str(path), 'raw_frame_path': str(path), 'raw_frame_sha256': sha256_file(path), 'source_width': int(info.width), 'source_height': int(info.height), 'manifest_width': int(manifest_width), 'manifest_height': int(manifest_height), 'source_video': str(video)})
            extracted += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError('no frames extracted')
    manifest = {'status': 'ok', 'method': 'build_v19_raw_frame_manifest', 'claim_scope': 'fresh V19 source-timeline extraction; frame rows are measurements of video timing and image paths only', 'input_video': str(video), 'input_video_sha256': sha256_file(video), 'video': {**asdict(info), 'duration_s': info.duration_s}, 'frame_start': int(frame_start), 'frame_end': int(frame_end), 'frame_count': int(extracted), 'full_source_timeline': bool(frame_start == 0 and frame_end == int(info.frame_count) - 1), 'manifest_width': int(manifest_width), 'manifest_height': int(manifest_height), 'rgb_dir': str(rgb_dir), 'frames': frames}
    manifest_path = output_dir / 'manifest.json'
    write_json_payload(manifest_path, manifest)
    report = {'status': 'ok', 'method': 'build_v19_raw_frame_manifest', 'input_video': str(video), 'manifest': str(manifest_path), 'rgb_dir': str(rgb_dir), 'source_frame_count': int(info.frame_count), 'extracted_frame_count': int(extracted), 'frame_start': int(frame_start), 'frame_end': int(frame_end), 'full_source_timeline': bool(manifest['full_source_timeline']), 'source_width': int(info.width), 'source_height': int(info.height), 'manifest_width': int(manifest_width), 'manifest_height': int(manifest_height), 'fps': float(info.fps), 'elapsed_s': float(time.time() - started)}
    write_json_payload(output_dir / 'v19_raw_frame_manifest_report.json', report)
    return report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True, help='Input video path')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output raw_frame_manifest directory')
    parser.add_argument('--render-width', type=int, default=960, help='Extracted frame width; <=0 preserves source resolution')
    parser.add_argument('--frame-start', type=int, default=None)
    parser.add_argument('--frame-end', type=int, default=None)
    parser.add_argument('--jpeg-quality', type=int, default=94)
    return parser.parse_args()

def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
import argparse
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

def raw_frames(path: Path) -> tuple[list[int], tuple[int, int], dict[str, Any]]:
    payload = read_json_mapping(path)
    frames = payload.get('frames')
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f'raw frame manifest has no frames: {path}')
    ids: list[int] = []
    widths: list[int] = []
    heights: list[int] = []
    for row in frames:
        if not isinstance(row, dict):
            continue
        idx = int(row.get('frame_idx', row.get('index', -1)))
        if idx < 0:
            raise RuntimeError(f'raw manifest row lacks frame_idx/index: {row}')
        ids.append(idx)
        w = int(row.get('source_width') or row.get('width') or row.get('manifest_width') or 0)
        h = int(row.get('source_height') or row.get('height') or row.get('manifest_height') or 0)
        if w > 0:
            widths.append(w)
        if h > 0:
            heights.append(h)
    if not ids:
        raise RuntimeError(f'raw frame manifest has no valid frame ids: {path}')
    if not widths or not heights:
        video = payload.get('video') if isinstance(payload.get('video'), dict) else {}
        w = int(video.get('width') or 0)
        h = int(video.get('height') or 0)
    else:
        w = int(np.median(widths))
        h = int(np.median(heights))
    if w <= 0 or h <= 0:
        raise RuntimeError(f'cannot determine source image size from {path}')
    return (sorted(ids), (w, h), payload)

def load_unidepth_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(path)
    blob = np.load(path, allow_pickle=True)
    required = {'frame_idx', 'intrinsics_fx_fy_cx_cy'}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f'{path} missing UniDepth intrinsics keys: {missing}')
    frame_idx = np.asarray(blob['frame_idx'], dtype=int)
    intr = np.asarray(blob['intrinsics_fx_fy_cx_cy'], dtype=np.float64)
    if intr.shape != (len(frame_idx), 4):
        raise RuntimeError(f'{path} intrinsics shape mismatch: {intr.shape}')
    source_size = np.asarray(blob['source_size'], dtype=int) if 'source_size' in blob.files else None
    if not np.isfinite(intr).all():
        raise RuntimeError(f'{path} contains non-finite intrinsics')
    return (frame_idx, intr, source_size)

def summarize(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'count': 0}
    q = np.percentile(arr, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    med = float(q[5])
    mad = float(np.median(np.abs(arr - med)))
    return {'count': int(arr.size), 'min': float(q[0]), 'p01': float(q[1]), 'p05': float(q[2]), 'p10': float(q[3]), 'p25': float(q[4]), 'median': med, 'p75': float(q[6]), 'p90': float(q[7]), 'p95': float(q[8]), 'p99': float(q[9]), 'max': float(q[10]), 'mean': float(np.mean(arr)), 'std': float(np.std(arr)), 'mad': mad, 'relative_mad_fraction': float(mad / med) if med else 0.0, 'relative_p05_p95_fraction': float((q[8] - q[2]) / med) if med else 0.0}

def select_rows(frame_idx: np.ndarray, intr: np.ndarray, *, frame_start: int | None, frame_end: int | None) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(frame_idx), dtype=bool)
    if frame_start is not None:
        mask &= frame_idx >= int(frame_start)
    if frame_end is not None:
        mask &= frame_idx <= int(frame_end)
    if not np.any(mask):
        raise RuntimeError(f'no UniDepth intrinsics rows in requested range {frame_start}:{frame_end}')
    return (frame_idx[mask], intr[mask])

def aggregate_intrinsics(intr: np.ndarray, method: str, trim_low: float, trim_high: float) -> tuple[np.ndarray, dict[str, Any]]:
    (fx, fy, cx, cy) = intr.T
    focal = np.sqrt(np.maximum(1e-09, fx * fy))
    if method == 'median':
        k = np.asarray([np.median(fx), np.median(fy), np.median(cx), np.median(cy)], dtype=np.float64)
        used = np.ones(len(intr), dtype=bool)
    elif method == 'trimmed_mean':
        if not 0.0 <= trim_low < trim_high <= 1.0:
            raise RuntimeError(f'invalid trim quantiles: {trim_low}, {trim_high}')
        (lo, hi) = np.quantile(focal, [trim_low, trim_high])
        used = (focal >= lo) & (focal <= hi)
        if not np.any(used):
            raise RuntimeError('trimmed aggregation removed every row')
        k = np.asarray([np.mean(fx[used]), np.mean(fy[used]), np.mean(cx[used]), np.mean(cy[used])], dtype=np.float64)
    else:
        raise RuntimeError(f'unsupported aggregation method: {method}')
    return (k, {'method': method, 'trim_low_quantile': float(trim_low) if method == 'trimmed_mean' else None, 'trim_high_quantile': float(trim_high) if method == 'trimmed_mean' else None, 'used_frame_count': int(np.count_nonzero(used)), 'input_frame_count': int(len(intr))})
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
from gates.camera.calibration import SCHEMA
from gates.camera.calibration import K_from_intrinsics
from gates.camera.calibration import finite_affine
from gates.camera.calibration import finite_intrinsics
from gates.camera.calibration import fov_degrees
from utils.data import read_json_mapping as camera_load_json
from gates.camera.calibration import normalize_authority
from gates.camera.calibration import plane_record
from gates.camera.calibration import resize_affine
from gates.camera.calibration import sha256_file
from gates.camera.calibration import size_wh
from gates.camera.calibration import transform_intrinsics
from gates.camera.calibration import validate_contract

def parse_affine(values: list[float] | None, label: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.size != 9:
        raise RuntimeError(f'{label} requires 9 values in row-major order')
    return finite_affine(array.reshape(3, 3), label)

def manifest_sizes(raw_manifest_path: Path) -> tuple[list[int], tuple[int, int], tuple[int, int], dict[str, Any]]:
    (frame_ids, source_size, payload) = raw_frames(raw_manifest_path)
    rows = [row for row in payload.get('frames', []) if isinstance(row, dict)]
    manifest_widths = [int(row.get('manifest_width') or 0) for row in rows]
    manifest_heights = [int(row.get('manifest_height') or 0) for row in rows]
    manifest_width = next((value for value in manifest_widths if value > 0), source_size[0])
    manifest_height = next((value for value in manifest_heights if value > 0), source_size[1])
    if any((value > 0 and value != manifest_width for value in manifest_widths)):
        raise RuntimeError('raw frame manifest has varying manifest widths')
    if any((value > 0 and value != manifest_height for value in manifest_heights)):
        raise RuntimeError('raw frame manifest has varying manifest heights')
    return (frame_ids, source_size, (manifest_width, manifest_height), payload)

def validate_sensor_timeline_and_source(*, raw_payload: dict[str, Any], source_payload: dict[str, Any], expected_frame_ids: list[int], prediction_source_video: Path | None, sensor_source_video: Path | None, time_tolerance_s: float) -> dict[str, Any]:
    source_frames = source_payload.get('frames')
    source_by_idx: dict[int, dict[str, Any]] = {}
    if isinstance(source_frames, list) and source_frames:
        for row in source_frames:
            if not isinstance(row, dict):
                continue
            idx = int(row.get('frame_idx', row.get('index', -1)))
            if idx < 0 or idx in source_by_idx:
                raise RuntimeError(f'sensor metadata has invalid/duplicate frame index {idx}')
            source_by_idx[idx] = row
        expected = [int(v) for v in expected_frame_ids]
        missing = [idx for idx in expected if idx not in source_by_idx]
        extras = sorted(set(source_by_idx).difference(expected))
        if missing or extras:
            raise RuntimeError(f'sensor metadata timeline mismatch: missing={missing[:20]} extras={extras[:20]}')
    else:
        expected = [int(v) for v in expected_frame_ids]
    raw_rows = {int(row.get('frame_idx', row.get('index', -1))): row for row in raw_payload.get('frames', []) if isinstance(row, dict)}
    tolerance = float(time_tolerance_s)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise RuntimeError('sensor timeline time tolerance must be finite and nonnegative')
    time_deltas: list[float] = []
    time_rows_checked = 0
    for idx in expected:
        raw_time = raw_rows.get(idx, {}).get('time_s')
        sensor_time = source_by_idx.get(idx, {}).get('time_s')
        if sensor_time is None:
            continue
        if raw_time is None:
            raise RuntimeError(f'raw metadata frame {idx} lacks time_s while sensor metadata provides it')
        delta = abs(float(raw_time) - float(sensor_time))
        if not np.isfinite(delta) or delta > tolerance:
            raise RuntimeError(f'raw/sensor time_s mismatch at frame {idx}: raw={raw_time} sensor={sensor_time} tolerance={tolerance}')
        time_deltas.append(delta)
        time_rows_checked += 1
    if source_by_idx and 0 < time_rows_checked < len(expected):
        raise RuntimeError(f'sensor metadata has partial time_s coverage: {time_rows_checked}/{len(expected)} rows')
    raw_input_video_text = raw_payload.get('input_video')
    raw_input_video = Path(str(raw_input_video_text)).expanduser().resolve() if raw_input_video_text else None
    prediction_video = prediction_source_video.expanduser().resolve() if prediction_source_video is not None else raw_input_video
    if prediction_video is None:
        raise RuntimeError('sensor-first contract requires --prediction-source-video or raw manifest input_video')
    if sensor_source_video is None:
        raise RuntimeError('sensor-first contract requires --sensor-source-video for source identity validation')
    sensor_video = sensor_source_video.expanduser().resolve()
    if not prediction_video.is_file() or not sensor_video.is_file():
        raise FileNotFoundError(f'source video validation files missing: prediction={prediction_video} sensor={sensor_video}')
    prediction_hash = sha256_file(prediction_video)
    sensor_hash = sha256_file(sensor_video)
    if prediction_hash != sensor_hash:
        raise RuntimeError(f'prediction input video and sensor-metadata source video are not byte-identical: prediction={prediction_hash} sensor={sensor_hash}')
    if source_by_idx and time_rows_checked == len(expected):
        status = 'exact_source_video_hash_and_frame_time_timeline_match'
    elif source_by_idx:
        status = 'exact_source_video_hash_and_frame_id_timeline_match_time_unavailable'
    else:
        status = 'exact_source_video_hash_match_sensor_frame_timeline_unavailable'
    return {'status': status, 'prediction_source_video': {'path': str(prediction_video), 'sha256': prediction_hash}, 'sensor_metadata_source_video': {'path': str(sensor_video), 'sha256': sensor_hash}, 'frame_count': len(expected), 'sensor_metadata_frame_rows_present': bool(source_by_idx), 'frame_ids_exact': True if source_by_idx else None, 'time_s_tolerance': tolerance, 'time_s_rows_checked': int(time_rows_checked), 'time_s_max_abs_delta': float(max(time_deltas, default=0.0)) if time_deltas else None, 'time_s_all_available_rows_match': True if time_rows_checked else None}

def source_intrinsics(source_payload: dict[str, Any], *, frame_intrinsics_key: str | None, expected_frame_ids: list[int], fixed_intrinsics_tolerance_px: float) -> tuple[np.ndarray, dict[str, Any]]:
    value = source_payload.get('intrinsics_fx_fy_cx_cy')
    if value is not None:
        return (finite_intrinsics(value, 'source calibration intrinsics'), {'mode': 'top_level_intrinsics_fx_fy_cx_cy', 'frame_intrinsics_key': None})
    if source_payload.get('K') is not None:
        K = np.asarray(source_payload['K'], dtype=np.float64)
        if K.shape != (3, 3):
            raise RuntimeError('source calibration K must be 3x3')
        return (finite_intrinsics([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], 'source calibration K'), {'mode': 'top_level_K', 'frame_intrinsics_key': None})
    frames = source_payload.get('frames')
    if isinstance(frames, list) and frames:
        key = str(frame_intrinsics_key or 'intrinsics_fx_fy_cx_cy')
        rows_by_idx: dict[int, np.ndarray] = {}
        missing_rows: list[int] = []
        for row in frames:
            if not isinstance(row, dict):
                continue
            idx = int(row.get('frame_idx', row.get('index', -1)))
            if idx < 0:
                raise RuntimeError('source calibration frame row lacks frame_idx/index')
            if idx in rows_by_idx:
                raise RuntimeError(f'source calibration has duplicate frame {idx}')
            if row.get(key) is None:
                missing_rows.append(idx)
                continue
            rows_by_idx[idx] = finite_intrinsics(row.get(key), f'source frame {idx} intrinsics key {key!r}')
        expected = [int(v) for v in expected_frame_ids]
        missing = [idx for idx in expected if idx not in rows_by_idx]
        extras = sorted(set(rows_by_idx).difference(expected))
        if missing or extras:
            raise RuntimeError(f'source calibration frame intrinsics key {key!r} does not match prediction timeline; missing={missing[:20]} extras={extras[:20]} rows_missing_key={missing_rows[:20]}')
        array = np.asarray([rows_by_idx[idx] for idx in expected], dtype=np.float64)
        reference = np.median(array, axis=0)
        max_abs = np.max(np.abs(array - reference[None, :]), axis=0)
        tolerance = float(fixed_intrinsics_tolerance_px)
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise RuntimeError('fixed intrinsics tolerance must be finite and nonnegative')
        if np.any(max_abs > tolerance):
            raise RuntimeError(f'source calibration varies beyond fixed pinhole tolerance {tolerance} px: max_abs={max_abs.tolist()}')
        return (finite_intrinsics(reference, 'source frame intrinsics median'), {'mode': 'per_frame_metadata_fixed_pinhole_median', 'frame_intrinsics_key': key, 'frame_count': int(len(array)), 'max_abs_delta_from_median_px': max_abs.tolist(), 'fixed_intrinsics_tolerance_px': tolerance})
    raise RuntimeError('source calibration lacks intrinsics_fx_fy_cx_cy/K or frame intrinsics')

def source_calibration_size(source_payload: dict[str, Any]) -> tuple[int, int]:
    if isinstance(source_payload.get('calibration_plane'), dict):
        return size_wh(source_payload['calibration_plane'], 'source calibration plane')
    width = int(source_payload.get('image_width') or source_payload.get('width') or 0)
    height = int(source_payload.get('image_height') or source_payload.get('height') or 0)
    if width > 0 and height > 0:
        return (width, height)
    source_size = source_payload.get('source_size')
    if source_size is not None:
        return size_wh(source_size, 'source calibration size')
    frames = source_payload.get('frames')
    if isinstance(frames, list) and frames and isinstance(frames[0], dict):
        first = frames[0]
        width = int(first.get('source_width') or first.get('manifest_width') or 0)
        height = int(first.get('source_height') or first.get('manifest_height') or 0)
        if width > 0 and height > 0:
            return (width, height)
    raise RuntimeError('source calibration lacks image/calibration size')

def infer_source_affine(calibration_size: tuple[int, int], source_size: tuple[int, int], explicit: np.ndarray | None) -> tuple[np.ndarray, str]:
    if explicit is not None:
        return (explicit, 'explicit_A_source_from_calibration')
    if calibration_size == source_size:
        return (np.eye(3, dtype=np.float64), 'identity_same_calibration_and_source_plane')
    raise RuntimeError(f'calibration plane {calibration_size} differs from source RGB plane {source_size}; provide --source-from-calibration-affine to record crop/pad/resize explicitly')

def build_planes(*, calibration_size: tuple[int, int], source_size: tuple[int, int], manifest_size: tuple[int, int], A_source_from_calibration: np.ndarray, A_render_from_source: np.ndarray, pixel_center_convention: str) -> dict[str, Any]:
    A_manifest_from_source = resize_affine(source_size, manifest_size, pixel_center_convention=pixel_center_convention)
    A_manifest_from_calibration = A_manifest_from_source @ A_source_from_calibration
    A_render_from_calibration = A_render_from_source @ A_source_from_calibration
    return {'calibration': plane_record(name='calibration', width=calibration_size[0], height=calibration_size[1], A_from_calibration=np.eye(3), transform_kind='identity', interpolation=None, pixel_center_convention=pixel_center_convention), 'source_rgb': plane_record(name='source_rgb', width=source_size[0], height=source_size[1], A_from_calibration=A_source_from_calibration, transform_kind='explicit_source_image_transform', interpolation='dataset_or_video_adapter_declared', pixel_center_convention=pixel_center_convention), 'manifest_rgb': plane_record(name='manifest_rgb', width=manifest_size[0], height=manifest_size[1], A_from_calibration=A_manifest_from_calibration, transform_kind='source_to_manifest_resize', interpolation='cv2.INTER_AREA_for_RGB', pixel_center_convention=pixel_center_convention), 'sam2_mask': plane_record(name='sam2_mask', width=manifest_size[0], height=manifest_size[1], A_from_calibration=A_manifest_from_calibration, transform_kind='source_to_manifest_resize', interpolation='nearest_for_masks', pixel_center_convention=pixel_center_convention), 'render': plane_record(name='render', width=manifest_size[0], height=manifest_size[1], A_from_calibration=A_render_from_calibration, transform_kind='explicit_render_from_source_transform', interpolation='renderer_raster', pixel_center_convention=pixel_center_convention)}

def save_intrinsics_npz(*, path: Path, frame_ids: list[int], calibration_intrinsics: np.ndarray, calibration_size: tuple[int, int], source_intrinsics: np.ndarray, source_size: tuple[int, int], contract_path: Path, intrinsics_source: str, authority: str) -> None:
    rows = np.repeat(source_intrinsics[None, :], len(frame_ids), axis=0).astype(np.float32)
    np.savez_compressed(path, frame_idx=np.asarray(frame_ids, dtype=np.int32), intrinsics_fx_fy_cx_cy=rows, constant_intrinsics_fx_fy_cx_cy=source_intrinsics.astype(np.float32), calibration_intrinsics_fx_fy_cx_cy=np.asarray(calibration_intrinsics, dtype=np.float32), calibration_size=np.asarray(calibration_size, dtype=np.int32), source_intrinsics_fx_fy_cx_cy=np.asarray(source_intrinsics, dtype=np.float32), source_size=np.asarray(source_size, dtype=np.int32), calibration_contract_json=np.asarray([str(contract_path)]), intrinsics_source=np.asarray([intrinsics_source]), calibration_authority=np.asarray([authority]))

def sensor_contract(args: argparse.Namespace) -> dict[str, Any]:
    (frame_ids, source_size, manifest_size, raw_payload) = manifest_sizes(args.raw_frame_manifest)
    source_path = args.sensor_calibration_contract.resolve()
    source_payload = camera_load_json(source_path)
    source_timeline_validation = validate_sensor_timeline_and_source(raw_payload=raw_payload, source_payload=source_payload, expected_frame_ids=frame_ids, prediction_source_video=args.prediction_source_video, sensor_source_video=args.sensor_source_video, time_tolerance_s=float(args.sensor_timeline_time_tolerance_s))
    (intrinsics_calibration, intrinsics_extraction) = source_intrinsics(source_payload, frame_intrinsics_key=args.sensor_frame_intrinsics_key, expected_frame_ids=frame_ids, fixed_intrinsics_tolerance_px=float(args.fixed_intrinsics_tolerance_px))
    calibration_size = source_calibration_size(source_payload)
    (A_source, source_affine_reason) = infer_source_affine(calibration_size, source_size, parse_affine(args.source_from_calibration_affine, 'source-from-calibration affine'))
    A_render_from_source = parse_affine(args.render_from_source_affine, 'render-from-source affine')
    if A_render_from_source is None:
        A_render_from_source = resize_affine(source_size, manifest_size, pixel_center_convention=args.pixel_center_convention)
    authority = normalize_authority(args.sensor_calibration_authority)
    source_plane_intrinsics = transform_intrinsics(intrinsics_calibration, A_source)
    intrinsics_source = str(source_payload.get('intrinsics_source') or source_payload.get('source') or source_payload.get('method') or 'prediction_side_sensor_calibration')
    contract_path = args.output_dir.resolve() / 'v19_camera_calibration_contract.json'
    intrinsics_npz_path = args.output_dir.resolve() / 'v19_camera_calibration_intrinsics.npz'
    contract = {'schema': SCHEMA, 'status': 'ok', 'method': 'resolve_v19_camera_contract', 'case': args.case, 'claim_scope': 'prediction-side camera calibration and explicit image-plane transforms only; no object/hand/CAD/pose/contact/evaluator labels are consumed', 'calibration_authority': authority, 'source_contract': {'path': str(source_path), 'sha256': sha256_file(source_path), 'method': source_payload.get('method'), 'schema': source_payload.get('schema')}, 'fallback': {'used': False, 'reason': None}, 'inputs': {'raw_frame_manifest': str(args.raw_frame_manifest.resolve()), 'sensor_calibration_contract': str(source_path), 'unidepth_npz': None}, 'outputs': {'contract_json': str(contract_path), 'intrinsics_npz': str(intrinsics_npz_path)}, 'raw_video': raw_payload.get('video') if isinstance(raw_payload.get('video'), dict) else None, 'frame_ids': frame_ids, 'intrinsics_model': 'constant_pinhole_fx_fy_cx_cy', 'intrinsics_coordinate_plane': 'calibration', 'intrinsics_vary_per_frame': False, 'calibration_plane': {'width': calibration_size[0], 'height': calibration_size[1], 'pixel_center_convention': args.pixel_center_convention, 'camera_coordinate_convention': args.camera_coordinate_convention, 'depth_convention': args.depth_convention}, 'intrinsics_fx_fy_cx_cy': intrinsics_calibration.tolist(), 'K': K_from_intrinsics(intrinsics_calibration).tolist(), 'focal_px': float(intrinsics_calibration[0]), 'focal_geom_px': float(np.sqrt(intrinsics_calibration[0] * intrinsics_calibration[1])), 'fov_degrees': fov_degrees(calibration_size[0], calibration_size[1], intrinsics_calibration), 'intrinsics_source': intrinsics_source, 'image_planes': build_planes(calibration_size=calibration_size, source_size=source_size, manifest_size=manifest_size, A_source_from_calibration=A_source, A_render_from_source=A_render_from_source, pixel_center_convention=args.pixel_center_convention), 'image_transform_contract': {'formula': 'K_plane = A_plane_from_calibration @ K_calibration', 'source_affine_reason': source_affine_reason, 'no_implicit_crop_pad_rotation': True}, 'source_plane_intrinsics_fx_fy_cx_cy': source_plane_intrinsics.tolist(), 'provenance': {'dataset_name': args.dataset_name, 'sensor_stream': args.sensor_stream, 'calibration_is_prediction_side_metadata': True, 'evaluator_gt_consumed': False, 'intrinsics_extraction': intrinsics_extraction, 'source_timeline_validation': source_timeline_validation}}
    validate_contract(contract, expected_frame_ids=frame_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_utf8(contract_path, contract)
    save_intrinsics_npz(path=intrinsics_npz_path, frame_ids=frame_ids, calibration_intrinsics=intrinsics_calibration, calibration_size=calibration_size, source_intrinsics=source_plane_intrinsics, source_size=source_size, contract_path=contract_path, intrinsics_source=intrinsics_source, authority=authority)
    return contract

def fallback_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.unidepth_npz is None:
        raise RuntimeError('sensor calibration is unavailable and --unidepth-npz fallback was not supplied')
    (frame_ids, source_size, manifest_size, _) = manifest_sizes(args.raw_frame_manifest)
    (selected_frame_idx, selected_intrinsics) = select_rows(*load_unidepth_intrinsics(args.unidepth_npz)[:2], frame_start=args.frame_start, frame_end=args.frame_end)
    (intrinsics, aggregation) = aggregate_intrinsics(selected_intrinsics, method=args.aggregation, trim_low=float(args.trim_low), trim_high=float(args.trim_high))
    if args.square_focal:
        focal = float(np.sqrt(intrinsics[0] * intrinsics[1]))
        intrinsics[0] = focal
        intrinsics[1] = focal
    if args.center_principal_point:
        intrinsics[2] = float(source_size[0]) / 2.0
        intrinsics[3] = float(source_size[1]) / 2.0
    A_render_from_source = parse_affine(args.render_from_source_affine, 'render-from-source affine')
    if A_render_from_source is None:
        A_render_from_source = resize_affine(source_size, manifest_size, pixel_center_convention=args.pixel_center_convention)
    contract_path = args.output_dir.resolve() / 'v19_camera_calibration_contract.json'
    intrinsics_npz_path = args.output_dir.resolve() / 'v19_camera_calibration_intrinsics.npz'
    contract = {'schema': SCHEMA, 'status': 'ok', 'method': 'resolve_v19_camera_contract', 'case': args.case, 'claim_scope': 'estimated camera fallback from robust UniDepth intrinsics; same downstream contract as sensor calibration', 'calibration_authority': 'estimated_from_rgb_depth_model', 'source_contract': None, 'fallback': {'used': True, 'reason': args.fallback_reason, 'method': 'robust_unidepth_video_constant'}, 'inputs': {'raw_frame_manifest': str(args.raw_frame_manifest.resolve()), 'sensor_calibration_contract': None, 'unidepth_npz': str(args.unidepth_npz.resolve())}, 'outputs': {'contract_json': str(contract_path), 'intrinsics_npz': str(intrinsics_npz_path)}, 'frame_ids': frame_ids, 'intrinsics_model': 'constant_pinhole_fx_fy_cx_cy', 'intrinsics_coordinate_plane': 'calibration', 'intrinsics_vary_per_frame': False, 'calibration_plane': {'width': source_size[0], 'height': source_size[1], 'pixel_center_convention': args.pixel_center_convention, 'camera_coordinate_convention': args.camera_coordinate_convention, 'depth_convention': args.depth_convention}, 'intrinsics_fx_fy_cx_cy': intrinsics.tolist(), 'K': K_from_intrinsics(intrinsics).tolist(), 'focal_px': float(intrinsics[0]), 'focal_geom_px': float(np.sqrt(intrinsics[0] * intrinsics[1])), 'fov_degrees': fov_degrees(source_size[0], source_size[1], intrinsics), 'intrinsics_source': 'v19_calibration_contract_unidepth_robust_video_constant', 'aggregation': aggregation, 'image_planes': build_planes(calibration_size=source_size, source_size=source_size, manifest_size=manifest_size, A_source_from_calibration=np.eye(3), A_render_from_source=A_render_from_source, pixel_center_convention=args.pixel_center_convention), 'image_transform_contract': {'formula': 'K_plane = A_plane_from_calibration @ K_calibration', 'source_affine_reason': 'identity_estimator_and_source_plane', 'no_implicit_crop_pad_rotation': True}, 'diagnostics': {'selected_frame_count': int(len(selected_frame_idx)), 'selected_stats': {'fx': summarize(selected_intrinsics[:, 0]), 'fy': summarize(selected_intrinsics[:, 1]), 'cx': summarize(selected_intrinsics[:, 2]), 'cy': summarize(selected_intrinsics[:, 3])}}, 'provenance': {'dataset_name': args.dataset_name, 'sensor_stream': args.sensor_stream, 'calibration_is_prediction_side_metadata': False, 'evaluator_gt_consumed': False}}
    validate_contract(contract, expected_frame_ids=frame_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_utf8(contract_path, contract)
    save_intrinsics_npz(path=intrinsics_npz_path, frame_ids=frame_ids, calibration_intrinsics=intrinsics, calibration_size=source_size, source_intrinsics=intrinsics, source_size=source_size, contract_path=contract_path, intrinsics_source=contract['intrinsics_source'], authority=contract['calibration_authority'])
    return contract

def resolve_camera(args: argparse.Namespace) -> dict[str, Any]:
    sensor_supplied = args.sensor_calibration_contract is not None
    if sensor_supplied:
        if not args.sensor_calibration_contract.exists():
            if not args.allow_missing_sensor_calibration_fallback:
                raise FileNotFoundError(args.sensor_calibration_contract)
            if args.unidepth_npz is None:
                raise RuntimeError('missing sensor calibration fallback requires --unidepth-npz')
            return fallback_contract(args)
        return sensor_contract(args)
    return fallback_contract(args)

def camera_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--raw-frame-manifest', type=Path, required=True)
    parser.add_argument('--sensor-calibration-contract', type=Path, default=None)
    parser.add_argument('--sensor-calibration-authority', choices=['dataset_sensor_calibration', 'prediction_side_sensor_metadata', 'explicit_user_calibration'], default='dataset_sensor_calibration')
    parser.add_argument('--sensor-frame-intrinsics-key', default=None, help='Optional per-frame metadata key containing [fx,fy,cx,cy] when the sensor metadata has no top-level K.')
    parser.add_argument('--prediction-source-video', type=Path, default=None, help='Prediction input video. Sensor-first mode hashes it against --sensor-source-video; defaults to raw manifest input_video when available.')
    parser.add_argument('--sensor-source-video', type=Path, default=None, help='Local copy of the exact image-stream video associated with the supplied sensor metadata.')
    parser.add_argument('--sensor-timeline-time-tolerance-s', type=float, default=1e-09)
    parser.add_argument('--fixed-intrinsics-tolerance-px', type=float, default=0.01, help='Maximum per-component deviation from the per-frame median allowed for a fixed pinhole camera contract.')
    parser.add_argument('--unidepth-npz', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-name', default=None)
    parser.add_argument('--sensor-stream', default=None)
    parser.add_argument('--source-from-calibration-affine', type=float, nargs=9, default=None, metavar=('A00', 'A01', 'A02', 'A10', 'A11', 'A12', 'A20', 'A21', 'A22'))
    parser.add_argument('--render-from-source-affine', type=float, nargs=9, default=None, metavar=('A00', 'A01', 'A02', 'A10', 'A11', 'A12', 'A20', 'A21', 'A22'))
    parser.add_argument('--pixel-center-convention', default='integer_pixel_centers_opencv')
    parser.add_argument('--camera-coordinate-convention', default='x_right_y_down_z_forward')
    parser.add_argument('--depth-convention', default='camera_z_meters')
    parser.add_argument('--allow-missing-sensor-calibration-fallback', action='store_true')
    parser.add_argument('--fallback-reason', default='prediction-side sensor calibration unavailable')
    parser.add_argument('--frame-start', type=int, default=None)
    parser.add_argument('--frame-end', type=int, default=None)
    parser.add_argument('--aggregation', choices=['median', 'trimmed_mean'], default='median')
    parser.add_argument('--trim-low', type=float, default=0.1)
    parser.add_argument('--trim-high', type=float, default=0.9)
    parser.add_argument('--square-focal', action='store_true')
    parser.add_argument('--center-principal-point', action='store_true')
    return parser.parse_args(argv)
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from gates.camera.calibration import load_contract
from gates.camera.calibration import plane_intrinsics
from gates.camera.calibration import sha256_file as depth_sha256_file
from gates.camera.calibration import summarize_contract

def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode('utf-8'))
    digest.update(np.asarray(contiguous.shape, dtype='<i8').tobytes())
    digest.update(contiguous.view(np.uint8).tobytes())
    return digest.hexdigest()

def scalar_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f'{label} must be scalar')
    return str(array.reshape(-1)[0])

def bind_depth_contract(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_depth_npz.expanduser().resolve()
    contract_path = args.camera_contract.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    report_path = output_dir / 'v19_depth_camera_contract_adapter_report.json'
    if output_path.exists() and (not args.replace):
        raise FileExistsError(f'output exists; use --replace explicitly: {output_path}')
    with np.load(source, allow_pickle=False) as archive:
        required = {'frame_idx', 'depth', 'source_size', 'intrinsics_fx_fy_cx_cy'}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f'source depth archive missing keys: {missing}')
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    frame_idx = np.asarray(payload['frame_idx'], dtype=np.int32)
    depth = np.asarray(payload['depth'])
    source_size = np.asarray(payload['source_size'], dtype=np.int32).reshape(-1)
    source_intrinsics = np.asarray(payload['intrinsics_fx_fy_cx_cy'], dtype=np.float64)
    source_confidence = np.asarray(payload['confidence']) if 'confidence' in payload else None
    if depth.ndim != 3 or frame_idx.shape != (depth.shape[0],):
        raise RuntimeError(f'inconsistent depth/frame arrays: {depth.shape}, {frame_idx.shape}')
    if source_size.shape != (2,):
        raise RuntimeError(f'source_size must be [width,height], got {source_size.shape}')
    if source_intrinsics.shape != (depth.shape[0], 4):
        raise RuntimeError(f'source intrinsics shape mismatch: {source_intrinsics.shape}')
    depth_size = (int(depth.shape[2]), int(depth.shape[1]))
    if depth_size != tuple((int(v) for v in source_size.tolist())):
        raise RuntimeError(f'depth raster {depth_size} disagrees with source_size {source_size.tolist()}')
    (contract, normalized) = load_contract(contract_path, expected_frame_ids=frame_idx.tolist())
    (intrinsics, transform) = plane_intrinsics(contract, normalized, plane_name=args.depth_plane, actual_size_wh=depth_size, allow_implicit_resize=bool(args.allow_implicit_depth_resize))
    resolved_rows = np.repeat(intrinsics[None, :], depth.shape[0], axis=0).astype(np.float64)
    contract_hash = depth_sha256_file(contract_path)
    source_conditioning_mode = scalar_text(payload['camera_conditioning_mode'], 'camera_conditioning_mode') if 'camera_conditioning_mode' in payload else 'legacy_unspecified'
    source_conditioning: dict[str, Any] = {'mode': source_conditioning_mode}
    source_depth_rays_reprojected = bool(np.asarray(payload.get('depth_ray_geometry_reprojected', False)).reshape(-1)[0])
    if source_conditioning_mode == 'provided_pinhole_intrinsics':
        required_conditioning = {'inference_camera_contract_sha256', 'inference_camera_output_plane', 'inference_camera_intrinsics_fx_fy_cx_cy', 'depth_output_quantity', 'depth_ray_geometry_reprojected', 'confidence'}
        missing_conditioning = sorted(required_conditioning - set(payload))
        if missing_conditioning:
            raise RuntimeError(f'camera-conditioned depth archive misses {missing_conditioning}')
        source_contract_hash = scalar_text(payload['inference_camera_contract_sha256'], 'inference_camera_contract_sha256')
        source_output_plane = scalar_text(payload['inference_camera_output_plane'], 'inference_camera_output_plane')
        if source_contract_hash != contract_hash:
            raise RuntimeError('depth inference camera-contract hash disagrees with the requested contract')
        if source_output_plane != args.depth_plane:
            raise RuntimeError(f'depth inference output plane {source_output_plane!r} disagrees with requested {args.depth_plane!r}')
        conditioned_rows = np.asarray(payload['inference_camera_intrinsics_fx_fy_cx_cy'], dtype=np.float64)
        if conditioned_rows.shape != source_intrinsics.shape or not np.allclose(conditioned_rows, source_intrinsics, atol=1e-06, rtol=0.0):
            raise RuntimeError('camera-conditioned inference K rows disagree with source depth K rows')
        if not source_depth_rays_reprojected:
            raise RuntimeError('camera-conditioned UniDepth archive did not convert metric radius onto exact output-contract rays')
        source_conditioning.update({'camera_contract_sha256': source_contract_hash, 'output_plane': source_output_plane, 'intrinsics_match_source_rows': True, 'depth_ray_geometry_reprojected': True, 'depth_output_quantity': scalar_text(payload.get('depth_output_quantity'), 'depth_output_quantity')})
    elif source_conditioning_mode not in {'model_inferred_intrinsics', 'legacy_unspecified'}:
        raise RuntimeError(f'unsupported source depth camera conditioning mode {source_conditioning_mode!r}')
    same_rays = bool(np.allclose(source_intrinsics, resolved_rows, atol=1e-06, rtol=0.0))
    if not same_rays and (not bool(args.allow_metadata_only_ray_relabel)):
        raise RuntimeError('source depth K differs from the requested camera-contract plane, but this adapter preserves the depth raster byte-for-byte and cannot reproject it onto new sensor rays. Rerun depth with the supplied sensor camera or add an explicit 3D reproject/z-buffer adapter. Use --allow-metadata-only-ray-relabel only for forensic reproduction of legacy invalid runs.')
    depth_hash_before = array_sha256(depth)
    frame_hash_before = array_sha256(frame_idx)
    payload['source_depth_active_intrinsics_fx_fy_cx_cy'] = np.asarray(payload['intrinsics_fx_fy_cx_cy'])
    payload['source_estimated_intrinsics_fx_fy_cx_cy'] = np.asarray(payload.get('model_camera_head_intrinsics_fx_fy_cx_cy', payload['intrinsics_fx_fy_cx_cy']))
    payload['intrinsics_fx_fy_cx_cy'] = resolved_rows
    payload['focal_px'] = np.repeat(np.float64(np.sqrt(intrinsics[0] * intrinsics[1])), depth.shape[0])
    payload['intrinsics_source'] = np.asarray(str(contract.get('intrinsics_source') or contract.get('method')))
    payload['calibration_authority'] = np.asarray(str(normalized['calibration_authority']))
    payload['camera_contract_path'] = np.asarray(str(contract_path))
    payload['camera_contract_sha256'] = np.asarray(contract_hash)
    payload['camera_contract_plane'] = np.asarray(args.depth_plane)
    payload['A_depth_from_calibration'] = np.asarray(transform['A_actual_plane_from_calibration'], dtype=np.float64)
    payload['intrinsics_override_applied'] = np.asarray(not same_rays)
    payload['depth_ray_geometry_reprojected'] = np.asarray(source_depth_rays_reprojected)
    payload['metadata_only_ray_relabel_override'] = np.asarray(bool(not same_rays and args.allow_metadata_only_ray_relabel))
    payload['source_depth_archive_sha256'] = np.asarray(depth_sha256_file(source))
    np.savez_compressed(output_path, **payload)
    with np.load(output_path, allow_pickle=False) as check:
        output_depth = np.asarray(check['depth'])
        output_frame_idx = np.asarray(check['frame_idx'])
        output_source_size = np.asarray(check['source_size'])
        output_intrinsics = np.asarray(check['intrinsics_fx_fy_cx_cy'])
        archived_source_intrinsics = np.asarray(check['source_depth_active_intrinsics_fx_fy_cx_cy'])
        archived_model_estimated_intrinsics = np.asarray(check['source_estimated_intrinsics_fx_fy_cx_cy'])
        output_confidence = np.asarray(check['confidence']) if 'confidence' in check.files else None
    model_estimated_intrinsics = np.asarray(payload.get('model_camera_head_intrinsics_fx_fy_cx_cy', source_intrinsics), dtype=np.float64)
    invariants = {'depth_array_equal': bool(np.array_equal(depth, output_depth, equal_nan=True)), 'depth_array_sha256_equal': bool(depth_hash_before == array_sha256(output_depth)), 'frame_idx_equal': bool(np.array_equal(frame_idx, output_frame_idx)), 'frame_idx_sha256_equal': bool(frame_hash_before == array_sha256(output_frame_idx)), 'source_size_equal': bool(np.array_equal(source_size, output_source_size)), 'source_active_intrinsics_preserved': bool(np.array_equal(source_intrinsics, archived_source_intrinsics)), 'source_model_estimated_intrinsics_preserved': bool(np.array_equal(model_estimated_intrinsics, archived_model_estimated_intrinsics)), 'confidence_array_equal': bool(source_confidence is None and output_confidence is None or (source_confidence is not None and output_confidence is not None and np.array_equal(source_confidence, output_confidence, equal_nan=True))), 'output_intrinsics_float64': bool(output_intrinsics.dtype == np.dtype(np.float64)), 'output_intrinsics_equal_contract_plane': bool(np.array_equal(output_intrinsics, resolved_rows))}
    if not all(invariants.values()):
        raise RuntimeError(f'depth camera-contract adapter invariant failed: {invariants}')
    delta = model_estimated_intrinsics - intrinsics[None, :]
    report = {'status': 'ok', 'method': 'adapt_v19_depth_to_camera_contract', 'claim_scope': 'fail-closed ray-contract binding; metric depth values, frame rows, raster size, and dtype are unchanged; no evaluator/object/hand labels are consumed', 'inputs': {'source_depth_npz': str(source), 'camera_contract': str(contract_path)}, 'outputs': {'depth_npz': str(output_path), 'report': str(report_path)}, 'frame_count': int(depth.shape[0]), 'depth_shape': list(depth.shape), 'depth_dtype': str(depth.dtype), 'active_intrinsics_dtype': str(resolved_rows.dtype), 'depth_plane': args.depth_plane, 'source_depth_camera_conditioning': source_conditioning, 'depth_ray_geometry_reprojected': source_depth_rays_reprojected, 'source_and_resolved_rays_match': same_rays, 'metadata_only_ray_relabel_override': bool(not same_rays and args.allow_metadata_only_ray_relabel), 'resolved_intrinsics_fx_fy_cx_cy': intrinsics.tolist(), 'camera_contract': summarize_contract(contract_path, contract, normalized), 'image_transform': transform, 'source_model_estimated_intrinsics_delta_from_resolved': {'median_fx_fy_cx_cy': np.median(delta, axis=0).tolist(), 'max_abs_fx_fy_cx_cy': np.max(np.abs(delta), axis=0).tolist()}, 'array_invariants': invariants, 'source_depth_archive_sha256': depth_sha256_file(source), 'output_depth_archive_sha256': depth_sha256_file(output_path), 'downstream_contract': 'Visible geometry may consume this byte-identical archive only when source_and_resolved_rays_match=true. A metadata-only override is forensic legacy reproduction and must fail downstream metric backprojection.'}
    write_json_utf8(report_path, report)
    return report

def depth_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-depth-npz', type=Path, required=True)
    parser.add_argument('--camera-contract', type=Path, required=True)
    parser.add_argument('--depth-plane', default='source_rgb')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--output-name', default='unidepth_full_frame_depth_camera_contract_v2.npz')
    parser.add_argument('--allow-implicit-depth-resize', action='store_true')
    parser.add_argument('--allow-metadata-only-ray-relabel', action='store_true', help='Forensic legacy reproduction only: preserve z bytes while replacing a disagreeing K. Downstream metric backprojection must reject this state.')
    parser.add_argument('--replace', action='store_true')
    return parser.parse_args(argv)
