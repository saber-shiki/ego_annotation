from __future__ import annotations
from utils.data import read_json_mapping
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from gates.camera.calibration import K_from_intrinsics, load_contract, plane_intrinsics, sha256_file

def camera_ray_angular_error_deg(rays: np.ndarray, camera_K: np.ndarray) -> dict:
    rays = np.asarray(rays, dtype=np.float64)
    if rays.ndim != 3 or rays.shape[0] != 3 or (not np.isfinite(rays).all()):
        raise RuntimeError(f'invalid UniDepth output rays: {rays.shape}')
    (_, height, width) = rays.shape
    (ys, xs) = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing='ij')
    inverse_K = np.linalg.inv(np.asarray(camera_K, dtype=np.float64))
    homogeneous = np.stack((xs, ys, np.ones_like(xs)), axis=0).reshape(3, -1)
    expected = (inverse_K @ homogeneous).reshape(3, height, width)
    expected /= np.linalg.norm(expected, axis=0, keepdims=True).clip(min=1e-12)
    actual = rays / np.linalg.norm(rays, axis=0, keepdims=True).clip(min=1e-12)
    dots = np.clip(np.sum(actual * expected, axis=0), -1.0, 1.0)
    angle = np.degrees(np.arccos(dots))
    return {'mean': float(np.mean(angle)), 'median': float(np.median(angle)), 'p95': float(np.percentile(angle, 95.0)), 'max': float(np.max(angle))}

def metric_radius_to_camera_z(radius: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float64)
    if radius.ndim != 2 or not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise RuntimeError(f'invalid UniDepth metric radius raster: {radius.shape}')
    (fx, fy, cx, cy) = np.asarray(intrinsics, dtype=np.float64).reshape(4)
    (height, width) = radius.shape
    (ys, xs) = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing='ij')
    ray_norm = np.sqrt(((xs - cx) / fx) ** 2 + ((ys - cy) / fy) ** 2 + 1.0)
    return (radius / ray_norm).astype(np.float32)

def run(args: argparse.Namespace) -> dict:
    import torch
    started = time.time()
    if args.unidepth_repo is not None:
        sys.path.insert(0, str(args.unidepth_repo))
    if not args.cpu and (not torch.cuda.is_available()):
        raise RuntimeError('UniDepth full-frame export requires CUDA unless --cpu is explicit')
    device = torch.device('cpu' if args.cpu else 'cuda')
    model = load_model(args.model_id, device)
    rows_in = read_manifest(args.manifest, int(args.frame_start), int(args.frame_end))
    if not rows_in:
        raise RuntimeError('manifest contains no selected frames')
    camera_conditioning = None
    input_camera_K = None
    output_camera_intrinsics = None
    if args.camera_contract is not None:
        contract_path = args.camera_contract.expanduser().resolve()
        if not contract_path.is_file():
            raise RuntimeError(f'missing camera contract: {contract_path}')
        (contract_payload, contract_normalized) = load_contract(contract_path)
        selected_frame_ids = [int(row['frame_idx']) for row in rows_in]
        contract_frame_ids = set((int(value) for value in contract_normalized['frame_ids']))
        missing_frame_ids = sorted(set(selected_frame_ids) - contract_frame_ids)
        if missing_frame_ids:
            raise RuntimeError(f'camera contract misses selected frame ids: {missing_frame_ids[:10]}')
        first_rgb_path = localize_path(str(rows_in[0]['rgb']), args.remote_root, args.local_root)
        with Image.open(first_rgb_path) as first_image:
            input_size_wh = tuple((int(value) for value in first_image.size))
        (input_intrinsics, input_plane) = plane_intrinsics(contract_payload, contract_normalized, plane_name=args.camera_input_plane, actual_size_wh=input_size_wh, allow_implicit_resize=False)
        (output_intrinsics, output_plane) = plane_intrinsics(contract_payload, contract_normalized, plane_name=args.camera_output_plane, actual_size_wh=(int(args.source_width), int(args.source_height)), allow_implicit_resize=False)
        input_camera_K = K_from_intrinsics(input_intrinsics)
        output_camera_intrinsics = np.asarray(output_intrinsics, dtype=np.float64)
        camera_conditioning = {'mode': 'provided_pinhole_intrinsics', 'camera_contract_path': str(contract_path), 'camera_contract_sha256': sha256_file(contract_path), 'camera_input_plane': str(args.camera_input_plane), 'camera_output_plane': str(args.camera_output_plane), 'input_size_wh': list(input_size_wh), 'output_size_wh': [int(args.source_width), int(args.source_height)], 'input_intrinsics_fx_fy_cx_cy': np.asarray(input_intrinsics, dtype=np.float64).tolist(), 'output_intrinsics_fx_fy_cx_cy': output_camera_intrinsics.tolist(), 'input_plane_resolution': input_plane, 'output_plane_resolution': output_plane, 'model_camera_head_intrinsics_max_abs_error_px': 0.0, 'active_depth_rays_mean_angular_error_deg': 0.0, 'active_depth_rays_median_angular_error_deg': 0.0, 'active_depth_rays_p95_angular_error_deg': 0.0, 'active_depth_rays_max_angular_error_deg': 0.0, 'output_depth_conversion': 'resize_metric_radius_then_project_to_exact_output_contract_rays_as_camera_z', 'depth_ray_geometry_reprojected': True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / 'stills'
    still_dir.mkdir(exist_ok=True)
    depth_png_dir = args.output_dir / 'depth'
    depth_png_dir.mkdir(exist_ok=True)
    depth_stack = []
    confidence_stack = []
    frame_indices = []
    focal_px = []
    intrinsics_stack = []
    model_camera_head_intrinsics_stack = []
    rows = []
    for (out_i, entry) in enumerate(rows_in):
        frame_idx = int(entry['frame_idx'])
        rgb_path = localize_path(str(entry['rgb']), args.remote_root, args.local_root)
        image = Image.open(rgb_path).convert('RGB')
        if camera_conditioning is not None and list(image.size) != camera_conditioning['input_size_wh']:
            raise RuntimeError(f"frame {frame_idx} RGB size {list(image.size)} differs from camera input plane {camera_conditioning['input_size_wh']}")
        (depth_raw, intrinsics, output_rays, radius_raw, confidence_raw) = infer_unidepth(model, image, device, input_camera_K)
        if intrinsics is None:
            raise RuntimeError(f'UniDepth returned no intrinsics for frame {frame_idx}')
        if input_camera_K is not None:
            returned = np.asarray([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]], dtype=np.float64)
            expected = np.asarray(camera_conditioning['input_intrinsics_fx_fy_cx_cy'], dtype=np.float64)
            delta_px = float(np.max(np.abs(returned - expected)))
            camera_conditioning['model_camera_head_intrinsics_max_abs_error_px'] = max(float(camera_conditioning['model_camera_head_intrinsics_max_abs_error_px']), delta_px)
            if output_rays is None:
                raise RuntimeError(f'UniDepth frame {frame_idx} returned no rays for camera-contract validation')
            ray_error = camera_ray_angular_error_deg(output_rays, input_camera_K)
            camera_conditioning['active_depth_rays_max_angular_error_deg'] = max(float(camera_conditioning['active_depth_rays_max_angular_error_deg']), float(ray_error['max']))
            camera_conditioning['active_depth_rays_mean_angular_error_deg'] = max(float(camera_conditioning['active_depth_rays_mean_angular_error_deg']), float(ray_error['mean']))
            camera_conditioning['active_depth_rays_median_angular_error_deg'] = max(float(camera_conditioning['active_depth_rays_median_angular_error_deg']), float(ray_error['median']))
            camera_conditioning['active_depth_rays_p95_angular_error_deg'] = max(float(camera_conditioning['active_depth_rays_p95_angular_error_deg']), float(ray_error['p95']))
            if float(ray_error['max']) > float(args.camera_rays_validation_max_angle_deg):
                raise RuntimeError(f"UniDepth frame {frame_idx} output rays disagree with supplied K by {ray_error['max']:.6f} degrees")
            model_camera_head_intrinsics_stack.append(returned.tolist())
            if radius_raw is None:
                raise RuntimeError(f'UniDepth frame {frame_idx} returned no metric radius for exact sensor-ray reprojection')
            radius_output = resize_depth(radius_raw, (int(args.source_height), int(args.source_width)))
            depth = metric_radius_to_camera_z(radius_output, output_camera_intrinsics)
            (fx, fy, cx, cy) = output_camera_intrinsics.tolist()
        else:
            scale_x = int(args.source_width) / float(depth_raw.shape[1])
            scale_y = int(args.source_height) / float(depth_raw.shape[0])
            fx = float(intrinsics[0, 0]) * scale_x
            fy = float(intrinsics[1, 1]) * scale_y
            cx = (float(intrinsics[0, 2]) + 0.5) * scale_x - 0.5
            cy = (float(intrinsics[1, 2]) + 0.5) * scale_y - 0.5
            if radius_raw is None:
                raise RuntimeError(f'UniDepth frame {frame_idx} returned no metric radius for inferred-ray reprojection')
            radius_output = resize_depth(radius_raw, (int(args.source_height), int(args.source_width)))
            depth = metric_radius_to_camera_z(radius_output, np.asarray([fx, fy, cx, cy]))
            model_camera_head_intrinsics_stack.append([fx, fy, cx, cy])
        focal = float(np.sqrt(max(1e-09, fx * fy)))
        if confidence_raw is None:
            raise RuntimeError(f'UniDepth frame {frame_idx} returned no confidence raster')
        confidence = resize_depth(confidence_raw, (int(args.source_height), int(args.source_width)))
        if not np.isfinite(confidence).all() or np.any(confidence <= 0.0):
            raise RuntimeError(f'frame {frame_idx} has invalid UniDepth confidence')
        valid = np.isfinite(depth) & (depth > 0.0)
        if int(np.count_nonzero(valid)) < int(args.min_valid_pixels):
            raise RuntimeError(f'frame {frame_idx} has too few valid UniDepth pixels')
        depth_png_path = depth_png_dir / f'{out_i:06d}.png'
        depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
        if not cv2.imwrite(str(depth_png_path), depth_mm):
            raise RuntimeError(f'failed to write {depth_png_path}')
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f'failed to read RGB {rgb_path}')
        norm = depth.copy()
        (lo, hi) = np.percentile(depth[valid], [5.0, 95.0])
        norm = np.clip((norm - lo) / max(1e-06, hi - lo), 0.0, 1.0)
        color = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        if color.shape[:2] != rgb.shape[:2]:
            color_review = cv2.resize(color, (int(rgb.shape[1]), int(rgb.shape[0])), interpolation=cv2.INTER_LINEAR)
        else:
            color_review = color
        if color_review.ndim != rgb.ndim or color_review.shape[2] != rgb.shape[2]:
            raise RuntimeError(f'review channel mismatch for frame {frame_idx}: rgb={rgb.shape} color={color_review.shape}')
        review = cv2.addWeighted(rgb, 0.55, color_review, 0.45, 0.0)
        cv2.imwrite(str(still_dir / f'frame_{frame_idx:06d}.png'), review)
        frame_indices.append(frame_idx)
        focal_px.append(focal)
        intrinsics_stack.append([fx, fy, cx, cy])
        depth_stack.append(depth.astype(np.float16))
        confidence_stack.append(confidence.astype(np.float16))
        values = depth[valid].astype(np.float64)
        rows.append({'frame_idx': frame_idx, 'rgb': str(rgb_path), 'depth_png': str(depth_png_path), 'unidepth_focal_px': focal, 'unidepth_fx_px': fx, 'unidepth_fy_px': fy, 'unidepth_cx_px': cx, 'unidepth_cy_px': cy, 'depth_median_m': float(np.median(values)), 'depth_p05_m': float(np.percentile(values, 5.0)), 'depth_p95_m': float(np.percentile(values, 95.0))})
    depth_archive = args.output_dir / 'unidepth_full_frame_depth_v3.npz'
    archive_payload = {'frame_idx': np.asarray(frame_indices, dtype=np.int32), 'depth': np.stack(depth_stack, axis=0), 'confidence': np.stack(confidence_stack, axis=0), 'source_size': np.asarray([int(args.source_width), int(args.source_height)], dtype=np.int32), 'focal_px': np.asarray(focal_px, dtype=np.float64), 'intrinsics_fx_fy_cx_cy': np.asarray(intrinsics_stack, dtype=np.float64), 'model_camera_head_intrinsics_fx_fy_cx_cy': np.asarray(model_camera_head_intrinsics_stack, dtype=np.float64), 'camera_conditioning_mode': np.asarray('provided_pinhole_intrinsics' if camera_conditioning is not None else 'model_inferred_intrinsics'), 'depth_ray_geometry_reprojected': np.asarray(camera_conditioning is not None), 'depth_output_quantity': np.asarray('camera_z_m_from_metric_radius_on_exact_contract_rays' if camera_conditioning is not None else 'camera_z_m_from_model_inferred_rays')}
    if camera_conditioning is not None:
        archive_payload.update({'inference_camera_contract_path': np.asarray(camera_conditioning['camera_contract_path']), 'inference_camera_contract_sha256': np.asarray(camera_conditioning['camera_contract_sha256']), 'inference_camera_input_plane': np.asarray(camera_conditioning['camera_input_plane']), 'inference_camera_output_plane': np.asarray(camera_conditioning['camera_output_plane']), 'inference_camera_intrinsics_fx_fy_cx_cy': np.asarray(intrinsics_stack, dtype=np.float64)})
    np.savez_compressed(depth_archive, **archive_payload)
    report = {'status': 'ok', 'annotation_ready': False, 'method': 'run_unidepth_full_frame_v3', 'model_id': str(args.model_id), 'manifest': str(args.manifest), 'frames': int(len(rows)), 'first_frame': int(frame_indices[0]), 'last_frame': int(frame_indices[-1]), 'depth_archive': str(depth_archive), 'stills_dir': str(still_dir), 'review_resolution': [int(rgb.shape[1]), int(rgb.shape[0])], 'depth_archive_source_size': [int(args.source_width), int(args.source_height)], 'depth_ray_geometry_reprojected': bool(camera_conditioning is not None), 'depth_output_quantity': 'camera_z_m_from_metric_radius_on_exact_contract_rays' if camera_conditioning is not None else 'camera_z_m_from_model_inferred_rays', 'camera_output_plane': str(camera_conditioning['camera_output_plane']) if camera_conditioning is not None else None, 'camera_conditioning': camera_conditioning, 'camera_ray_contract': "UniDepth depth head consumed the supplied camera rays. The model's metric radius is resized and explicitly projected onto exact output-plane contract rays to produce camera-z meters; model camera-head K remains diagnostic only." if camera_conditioning is not None else 'UniDepth inferred rays/K. Do not relabel this byte-identical z raster with another K; rerun using --camera-contract.', 'unidepth_focal_px': summarize([row['unidepth_focal_px'] for row in rows]), 'depth_median_m': summarize([row['depth_median_m'] for row in rows]), 'rows': rows, 'elapsed_s': float(time.time() - started)}
    (args.output_dir / 'qc_unidepth_full_frame_v3.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for (k, v) in report.items() if k != 'rows'}, indent=2))
    return report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera-contract', type=Path)
    parser.add_argument('--camera-input-plane', default='manifest_rgb')
    parser.add_argument('--camera-output-plane', default='source_rgb')
    parser.add_argument('--camera-rays-validation-max-angle-deg', type=float, default=1.0)
    parser.add_argument('--frame-start', type=int, required=True)
    parser.add_argument('--frame-end', type=int, required=True)
    parser.add_argument('--unidepth-repo', type=Path)
    parser.add_argument('--remote-root', type=Path)
    parser.add_argument('--local-root', type=Path)
    parser.add_argument('--source-width', type=int, default=1920)
    parser.add_argument('--source-height', type=int, default=1080)
    parser.add_argument('--min-valid-pixels', type=int, default=100000)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()

def main() -> None:
    run(parse_args())

def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {'count': 0}
    return {'count': int(len(arr)), 'median': float(np.median(arr)), 'p05': float(np.percentile(arr, 5.0)), 'p95': float(np.percentile(arr, 95.0)), 'max': float(np.max(arr))}

def read_manifest(path: Path, frame_start: int, frame_end: int) -> list[dict]:
    frames = read_json_mapping(path).get('frames')
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f'{path} must contain a nonempty frames list')
    rows = [row for row in frames if int(frame_start) <= int(row['frame_idx']) <= int(frame_end)]
    rows.sort(key=lambda row: int(row['frame_idx']))
    if not rows:
        raise RuntimeError(f'no frames in requested range {frame_start}:{frame_end}')
    return rows

def localize_path(path: str, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        for (src, dst) in ((local_root, remote_root), (remote_root, local_root)):
            try:
                rel = direct.relative_to(src)
            except ValueError:
                continue
            candidate = dst / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)

def to_numpy(value: object, name: str) -> np.ndarray:
    import torch
    if isinstance(value, torch.Tensor):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.squeeze(arr)
    if not np.isfinite(arr).all():
        raise RuntimeError(f'UniDepth {name} contains non-finite values')
    return arr.astype(np.float32)

def _unidepth_resize_geometry(model: object, shape: tuple[int, int]) -> tuple[float, float, float, int, int]:
    """Mirror the audited deployed infer shape policy, not its erroneous K scale."""
    import inspect
    from math import ceil

    try:
        inspect.signature(model.infer).bind(None, camera=None)
        inspect.signature(model.encode_decode).bind(inputs={}, image_metas=[])
        constraints = model.shape_constraints
        low_ratio, high_ratio = constraints['ratio_bounds']
        low_pixels, high_pixels = constraints['pixels_min'], constraints['pixels_max']
        values = np.asarray([low_ratio, high_ratio, low_pixels, high_pixels], dtype=np.float64)
        if (not np.isfinite(values).all() or np.any(values <= 0)
                or low_ratio > high_ratio or low_pixels > high_pixels
                or constraints.get('shape_mult', 14) != 14
                or model.interpolation_mode != 'bilinear'):
            raise ValueError('unsupported shape constraints or interpolation')
        if hasattr(model, 'resolution_level'):
            level = model.resolution_level
            if not np.isfinite(level) or not 0 <= level < 10:
                raise ValueError('resolution_level must be in [0, 10)')
            interval = (high_pixels - low_pixels) / 10
            low_pixels, high_pixels = low_pixels + level * interval, low_pixels + (level + 1) * interval
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'unsupported UniDepthV2 infer/encode_decode interface: {exc}') from exc

    height, width = shape
    if height < 2 or width < 2:
        raise ValueError(f'UniDepth image dimensions must be at least 2: {shape}')
    ratio = min(high_ratio, max(low_ratio, width / height))
    if width / height > ratio:
        padded_h, padded_w = int(width / ratio), width
    else:
        padded_h, padded_w = height, int(height * ratio)
    left, top = (padded_w - width) // 2, (padded_h - height) // 2
    pixels = padded_h * padded_w
    nominal = (min(high_pixels, max(low_pixels, pixels)) / pixels) ** 0.5
    resized_h = ceil(int(padded_h * nominal) / 14) * 14
    resized_w = ceil(int(padded_w * nominal) / 14) * 14
    if min(resized_h, resized_w) <= 0:
        raise RuntimeError('unsupported UniDepthV2 zero-sized resized image')
    return nominal, resized_w / padded_w, resized_h / padded_h, left, top


def _validated_pinhole_K(camera_K: np.ndarray) -> np.ndarray:
    K = np.array(camera_K, dtype=np.float64, copy=True)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.array_equal(K[2], [0, 0, 1])
            or K[0, 1] != 0 or K[1, 0] != 0):
        raise ValueError('UniDepth camera K must be a finite zero-skew pinhole 3x3 matrix with positive focal lengths')
    return K


def infer_unidepth(model: object, image: Image.Image, device: torch.device, camera_K: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    import torch
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    nominal, scale_x, scale_y, left, top = _unidepth_resize_geometry(model, rgb.shape[:2])
    image_t = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
    camera_t = None
    if camera_K is not None:
        K = _validated_pinhole_K(camera_K)
        # infer adds padding then multiplies by nominal. Precompensate so the
        # depth head gets the actual rounded image scale and half-center K.
        K[0, 0] *= scale_x / nominal
        K[1, 1] *= scale_y / nominal
        K[0, 2] = (K[0, 2] + left + 0.5) * scale_x / nominal - left
        K[1, 2] = (K[1, 2] + top + 0.5) * scale_y / nominal - top
        camera_t = torch.tensor(K, dtype=torch.float32, device=device)
        if not torch.isfinite(camera_t).all() or torch.any(camera_t.diag()[:2] <= 0):
            raise ValueError('UniDepth camera K cannot be represented in float32')
    with torch.no_grad():
        prediction = model.infer(image_t, camera=camera_t)
    if not isinstance(prediction, dict):
        raise RuntimeError(f'UniDepth infer returned {type(prediction)}')
    depth_key = 'depth' if 'depth' in prediction else 'depths' if 'depths' in prediction else None
    if depth_key is None:
        raise RuntimeError(f'UniDepth prediction lacks depth key: {sorted(prediction)}')
    depth = to_numpy(prediction[depth_key], 'depth')
    if depth.ndim != 2:
        raise RuntimeError(f'UniDepth depth has invalid shape {depth.shape}')
    intrinsics = None
    for key in ('intrinsics', 'K', 'camera'):
        if key in prediction:
            candidate = to_numpy(prediction[key], key)
            if candidate.shape == (3, 3):
                intrinsics = candidate.astype(np.float64)
                break
            if candidate.size == 9:
                intrinsics = candidate.reshape(3, 3).astype(np.float64)
                break
    if intrinsics is None:
        raise RuntimeError('unsupported UniDepthV2 output: missing camera-head intrinsics')
    # This is the independently predicted camera-head K, NOT the supplied K.
    # Undo upstream's nominal inverse, then invert the true image transform.
    intrinsics[0, 0] *= nominal / scale_x
    intrinsics[1, 1] *= nominal / scale_y
    intrinsics[0, 2] = (intrinsics[0, 2] + left) * nominal / scale_x - left - 0.5
    intrinsics[1, 2] = (intrinsics[1, 2] + top) * nominal / scale_y - top - 0.5
    intrinsics = _validated_pinhole_K(intrinsics)
    rays = None
    if 'rays' in prediction:
        candidate = to_numpy(prediction['rays'], 'rays')
        if candidate.ndim == 3 and candidate.shape[0] == 3:
            rays = candidate.astype(np.float64)
        elif candidate.ndim == 3 and candidate.shape[-1] == 3:
            rays = np.moveaxis(candidate, -1, 0).astype(np.float64)
        else:
            raise RuntimeError(f'UniDepth rays have invalid shape {candidate.shape}')
    radius = None
    if 'radius' in prediction:
        candidate = to_numpy(prediction['radius'], 'radius')
        if candidate.ndim != 2:
            raise RuntimeError(f'UniDepth radius has invalid shape {candidate.shape}')
        radius = candidate.astype(np.float64)
    confidence = None
    if 'confidence' in prediction:
        candidate = to_numpy(prediction['confidence'], 'confidence')
        if candidate.ndim != 2:
            raise RuntimeError(f'UniDepth confidence has invalid shape {candidate.shape}')
        confidence = candidate.astype(np.float64)
    return (depth, intrinsics, rays, radius, confidence)

def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float32)
    return cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

def load_model(model_id: str, device: torch.device) -> object:
    import torch
    if not Path(model_id).is_dir():
        raise ValueError('UniDepth requires an explicitly provisioned local model directory')
    from unidepth.models import UniDepthV2
    if hasattr(UniDepthV2, 'from_pretrained'):
        model = UniDepthV2.from_pretrained(model_id)
    else:
        raise RuntimeError('UniDepthV2.from_pretrained is missing in this checkout')
    return model.to(device).eval()
if __name__ == '__main__':
    main()
