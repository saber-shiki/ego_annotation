"""Full-topology native-gauge and image-scale kernels extracted from visual fitting."""
from __future__ import annotations
from utils.data import read_json
from utils.state import content_sha256
from utils.data import write_json_utf8
from pathlib import Path
import hashlib, json
import cv2, numpy as np, trimesh
from utils.data import load
from utils.state import sha
from utils.geometry import validate_transforms
from observations.common import pose_from_values
from utils.data import RKEY, TKEY

def _pose(row):
    return pose_from_values(row[RKEY], row[TKEY])
AUTHORITY = {'demo_only': True, 'annotation_ready': False, 'physical_authority': False, 'demo_visual_proximity_fit_eligible': True, 'physical_contact_authority_eligible': False, **{f'generated_faces_{role}_eligible': False for role in ('contact', 'collision', 'sign', 'sdf', 'nonpenetration')}}

def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def record(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}

def mesh_arrays(path):
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('one complete triangular mesh required')
    (v, f) = (np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
    if v.ndim != 2 or v.shape[1:] != (3,) or (not len(v)) or (not np.isfinite(v).all()) or (f.ndim != 2) or (f.shape[1:] != (3,)) or (not len(f)) or (not np.issubdtype(f.dtype, np.integer)) or (f.min() < 0) or (f.max() >= len(v)):
        raise ValueError('invalid complete vertices/topology')
    return (v, f.astype(np.int64))

def write_mesh(path, vertices, faces):
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)
    (v, f) = mesh_arrays(path)
    if not np.array_equal(f, faces) or v.shape != vertices.shape or (not np.allclose(v, vertices, atol=1e-06, rtol=0)):
        raise ValueError('serialized mesh changed topology/vertex order or precision')
    return {'vertex_count': len(v), 'face_count': len(f), 'face_index_sha256': hashlib.sha256(np.ascontiguousarray(f, dtype=np.int64).tobytes()).hexdigest(), 'vertices_float64_sha256': hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(), 'faces_exactly_preserved': True, 'serialized_vertex_tolerance_m': 1e-06}

def scene_similarity(native_cv, observed_cam, K, mask):
    projection = convex_projection_and_front_depth(native_cv, K, mask)
    if projection['positive_depth_vertex_fraction'] < 0.99 or projection['owned_zbuffer_pixel_count'] < 100 or projection['convex_projection_iou'] < 0.05:
        raise ValueError('native positive/projected front-depth support insufficient')
    depth = float(np.median(observed_cam[:, 2]))
    scale = depth / projection['owned_zbuffer_median_native_depth']
    if not np.isfinite(scale) or not 0.05 <= scale <= 5.0:
        raise ValueError('scene similarity outside [0.05,5.0]')
    return (scale, {'scene_similarity_scale': scale, 'bounds': [0.05, 5.0], 'metric_depth_median_m': depth, 'metric_depth_p05_p95_m': np.percentile(observed_cam[:, 2], [5, 95]).tolist(), 'metric_surfel_count': len(observed_cam), 'native_front_projection': projection, 'uncertainty': 'single_view_native_front_depth_and_retained_monocular_metric_surfel_uncertainty'})
from typing import Any

def finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise RuntimeError(f'{label} must be finite length {size}, got {array}')
    return array

def convex_projection_and_front_depth(vertices_camera: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices_camera, dtype=np.float64)
    (fx, fy, cx, cy) = finite_vector(intrinsics, 4, 'projection intrinsics')
    (height, width) = mask.shape
    z = vertices[:, 2]
    valid = np.isfinite(vertices).all(axis=1) & (z > 1e-05)
    if np.count_nonzero(valid) < 3:
        raise RuntimeError('SAM3D native pose has fewer than three positive-depth vertices')
    u = fx * vertices[valid, 0] / z[valid] + cx
    v = fy * vertices[valid, 1] / z[valid] + cy
    uv = np.column_stack((u, v))
    finite_uv = np.isfinite(uv).all(axis=1)
    uv = uv[finite_uv]
    z_projected = z[valid][finite_uv]
    inside = (uv[:, 0] >= 0.0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= height - 1)
    uv_inside = uv[inside]
    z_inside = z_projected[inside]
    if len(uv_inside) < 3:
        raise RuntimeError('SAM3D native mesh projection does not enter the owned-mask image')
    hull = cv2.convexHull(np.rint(uv_inside).astype(np.int32))
    projected = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(projected, hull, 1)
    target = np.asarray(mask, dtype=bool)
    projected_bool = projected > 0
    intersection = int(np.count_nonzero(projected_bool & target))
    union = int(np.count_nonzero(projected_bool | target))
    iou = float(intersection / max(1, union))
    xi = np.rint(uv_inside[:, 0]).astype(np.int64)
    yi = np.rint(uv_inside[:, 1]).astype(np.int64)
    pixel_inside_target = target[yi, xi]
    xi = xi[pixel_inside_target]
    yi = yi[pixel_inside_target]
    z_owned = z_inside[pixel_inside_target]
    if len(z_owned) == 0:
        raise RuntimeError('SAM3D native projected vertices have no object-owned depth samples')
    flat_index = yi * width + xi
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, flat_index, z_owned)
    finite_zbuffer = zbuffer[np.isfinite(zbuffer)]
    return {'positive_depth_vertex_fraction': float(np.count_nonzero(valid) / len(vertices)), 'projected_vertices_inside_image': int(len(uv_inside)), 'projected_vertices_inside_owned_mask': int(len(z_owned)), 'owned_zbuffer_pixel_count': int(len(finite_zbuffer)), 'owned_zbuffer_coverage_fraction': float(len(finite_zbuffer) / max(1, np.count_nonzero(target))), 'owned_zbuffer_median_native_depth': float(np.median(finite_zbuffer)), 'convex_projection_iou': iou, 'convex_projection_pixels': int(np.count_nonzero(projected_bool)), 'owned_mask_pixels': int(np.count_nonzero(target))}
import argparse
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
import cv2
import numpy as np
SAFE_ID = re.compile('^[A-Za-z0-9][A-Za-z0-9_.-]*$')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def appearance_load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'JSON object required: {path}')
    return value

def in_run_file(path: str | Path, run_root: Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_relative_to(run_root) or not value.is_file():
        raise ValueError(f'{label} missing or outside run root: {path}')
    return value

def indexed_rows(rows, label: str) -> dict[int, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{label} must contain nonempty rows')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or isinstance(row.get('frame_idx'), bool):
            raise ValueError(f'invalid {label} row')
        frame_idx = int(row['frame_idx'])
        if frame_idx in result:
            raise ValueError(f'duplicate {label} frame {frame_idx}')
        result[frame_idx] = row
    return result

def track_rows(track: dict) -> dict[int, dict]:
    result = {}
    for (key, row) in track.items():
        try:
            frame_idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'SAM2 track key is not an integer: {key!r}') from exc
        if str(frame_idx) != str(key) or frame_idx in result or (not isinstance(row, dict)):
            raise ValueError(f'invalid/duplicate SAM2 track frame {key!r}')
        result[frame_idx] = row
    if not result:
        raise ValueError('empty SAM2 track')
    return result

def rgb_path(row: dict, root: Path) -> Path:
    raw = row.get('raw_frame_path') or row.get('rgb') or row.get('image_path')
    if not raw:
        raise ValueError(f"raw frame {row.get('frame_idx')} has no RGB path")
    return in_run_file(raw, root, 'raw RGB')

def metric_object(row: dict, object_id: str) -> dict | None:
    objects = row.get('objects')
    if not isinstance(objects, list):
        raise ValueError(f"annotation frame {row.get('frame_idx')} has invalid objects")
    matches = [obj for obj in objects if isinstance(obj, dict) and obj.get('object_id') == object_id]
    if len(matches) > 1:
        raise ValueError(f"duplicate object {object_id!r} at frame {row.get('frame_idx')}")
    return matches[0] if matches else None

def metric_alignment(obj: dict | None, frame: dict) -> dict | None:
    """Strict retained P09 alignment for appearance anchoring."""
    from gates.object.observation import retained_metric_alignment
    (points, reason, _) = retained_metric_alignment(obj, frame)
    if points is None:
        return None
    geometry = obj['visible_geometry_candidate']
    return {'camera': frame['camera'], 'centroid_world_m': np.asarray(geometry['centroid_world_m'], dtype=float).tolist(), 'world_surfels_m': np.asarray(geometry['world_vertices_sample_m'], dtype=float).tolist(), 'camera_surfels_m': np.asarray(geometry['camera_vertices_sample_m'], dtype=float).tolist(), 'owned_metric_mask_path': obj.get('mask_path'), 'role': 'same_frame_retained_P09_metric_alignment'}


def score_mask(image: np.ndarray, mask: np.ndarray, *, min_area_fraction: float, min_border_fraction: float, min_sharpness: float) -> dict:
    support = mask > 0
    (height, width) = support.shape
    area = int(support.sum())
    (ys, xs) = np.where(support)
    if not area:
        return {'eligible': False, 'rejection': 'empty_mask', 'area_px': 0}
    (x0, x1, y0, y1) = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    area_fraction = area / float(width * height)
    border_fraction = min(x0, y0, width - 1 - x1, height - 1 - y1) / float(min(width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(support.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian[dilated])) if np.any(dilated) else 0.0
    reasons = []
    if area_fraction < min_area_fraction:
        reasons.append('mask_too_small')
    if border_fraction < min_border_fraction:
        reasons.append('mask_near_image_border')
    if sharpness < min_sharpness:
        reasons.append('appearance_crop_too_blurred')
    sharpness_term = min(1.0, np.log1p(max(0.0, sharpness)) / np.log(1001.0))
    safe_clearance_fraction = 0.05
    clearance_term = min(1.0, border_fraction / safe_clearance_fraction)
    area_term = min(1.0, area_fraction / 0.25)
    score = 0.45 * area_term + 0.35 * sharpness_term + 0.2 * clearance_term
    return {'eligible': not reasons, 'rejection': None if not reasons else '+'.join(reasons), 'area_px': area, 'area_fraction': area_fraction, 'bbox_xyxy': [x0, y0, x1 + 1, y1 + 1], 'border_clearance_fraction': border_fraction, 'laplacian_variance_dilated_mask': sharpness, 'score_terms': {'area': area_term, 'sharpness': sharpness_term, 'border_clearance': clearance_term, 'safe_clearance_fraction': safe_clearance_fraction, 'weights': {'area': 0.45, 'sharpness': 0.35, 'border_clearance': 0.2}}, 'score': float(score)}
import argparse
import json
import shutil
from pathlib import Path
import cv2
SCHEMA = 'rgb_scene_depth_order_appearance_manifest_v1'
OWNERSHIP_SCHEMA = 'rgb_scene_depth_order_owned_sam2_v1'

def require_ref(value: object, expected: Path, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f'{label} record missing')
    path = Path(value.get('path', '')).expanduser().resolve()
    if path != expected.resolve() or not path.is_file() or value.get('sha256') != sha256(path):
        raise ValueError(f'{label} path/hash mismatch')
    return value

def validate_binding(annotation_payload: dict, ownership: dict, *, raw_track: Path, annotations: Path, case_id: str, object_id: str, track_id: str) -> dict:
    adapter = annotation_payload.get('v19_visible_geometry_adapter')
    if ownership.get('schema') != OWNERSHIP_SCHEMA or ownership.get('status') != 'ok' or (not isinstance(adapter, dict)):
        raise ValueError('corrected ownership/P09 contracts missing')
    expected_ids = {'case': case_id, 'object_id': object_id, 'track_id': track_id}
    if any((ownership.get(key) != value or adapter.get(key) != value for (key, value) in expected_ids.items())):
        raise ValueError('raw/owned/P09 IDs differ')
    raw_ref = require_ref((ownership.get('inputs') or {}).get('raw_sam2_track'), raw_track, 'raw SAM2 track')
    owned_value = (ownership.get('outputs') or {}).get('owned_sam2_track')
    if not isinstance(owned_value, dict):
        raise ValueError('owned SAM2 track record missing')
    owned_path = Path(owned_value.get('path', '')).expanduser().resolve()
    owned_ref = require_ref(owned_value, owned_path, 'owned SAM2 track')
    if Path(adapter.get('sam2_track', '')).expanduser().resolve() != owned_path:
        raise ValueError('corrected P09 did not consume owned SAM2 track')
    for key in ('base_annotations', 'depth_npz', 'calibration_contract'):
        producer = (ownership.get('inputs') or {}).get(key)
        path = Path(adapter.get(key, '')).expanduser().resolve()
        require_ref(producer, path, key)
    return {'preprocessor': {'path': str(ownership['_path']), 'sha256': sha256(ownership['_path'])}, 'raw_sam2_track': raw_ref, 'owned_sam2_track': owned_ref, 'corrected_p09_annotations': {'path': str(annotations), 'sha256': sha256(annotations)}}

def build_appearance(args: argparse.Namespace) -> tuple[dict, dict]:
    root = args.run_root.resolve()
    if not SAFE_ID.fullmatch(args.object_id):
        raise ValueError('unsafe object ID')
    output = args.output_dir.resolve()
    candidate_root = getattr(args, 'candidate_root', None)
    allowed = candidate_root.resolve() if candidate_root is not None else root / 'measurements' / 'demo_visual' / args.object_id
    if candidate_root is None and output.parent != allowed:
        raise ValueError('new output under measurements/demo_visual/<object> required')
    if candidate_root is not None:
        allowed = allowed / 'measurements' / 'demo_visual' / args.object_id
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError('new output under measurements/demo_visual/<object> required')
    raw_manifest_path = in_run_file(args.raw_frame_manifest, root, 'raw frame manifest')
    annotations_path = in_run_file(args.annotations, root, 'corrected P09 annotations')
    raw_track_path = in_run_file(args.raw_sam2_track, root, 'raw SAM2 track')
    ownership_path = in_run_file(args.depth_order_report, root, 'depth-order report')
    annotation_payload = appearance_load(annotations_path)
    ownership = appearance_load(ownership_path)
    ownership['_path'] = ownership_path
    binding = validate_binding(annotation_payload, ownership, raw_track=raw_track_path, annotations=annotations_path, case_id=args.case_id, object_id=args.object_id, track_id=args.track_id)
    raw = indexed_rows(appearance_load(raw_manifest_path).get('frames'), 'raw manifest')
    annotations = indexed_rows(annotation_payload.get('frames'), 'annotations')
    track = track_rows(appearance_load(raw_track_path))
    if set(raw) != set(annotations) or set(raw) != set(track):
        raise ValueError('raw/annotation/raw-SAM2 frame coverage mismatch')
    mask_dir = output / 'appearance_masks'
    mask_dir.mkdir(parents=True)
    rows = []
    ranked = []
    for idx in sorted(raw, key=lambda value: float(annotations[value]['time_s'])):
        (ann, track_row) = (annotations[idx], track[idx])
        image_path = rgb_path(raw[idx], root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'unreadable RGB at frame {idx}')
        source_value = track_row.get('mask_path')
        visible = track_row.get('visible') is True
        row = {'frame_idx': idx, 'time_s': float(ann['time_s']), 'object_id': args.object_id, 'visible': visible, 'semantic_role': 'raw_appearance_silhouette_only_not_metric_surface', 'raw_rgb_path': str(image_path), 'raw_rgb_sha256': sha256(image_path)}
        if source_value is None or not visible:
            if visible:
                raise ValueError(f'visible raw SAM2 frame {idx} lacks mask')
            row.update(status='unavailable_not_substituted', source_mask_path=None, appearance_mask_path=None, source_mask_sha256=None, appearance_mask_sha256=None, anchor_eligible=False, rejection='track_marks_frame_not_visible')
            rows.append(row)
            continue
        source = in_run_file(source_value, root, 'raw appearance mask')
        mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2] or (not mask.any()):
            raise ValueError(f'raw appearance mask invalid at frame {idx}')
        destination = mask_dir / f'frame_{idx:010d}{source.suffix.lower()}'
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError('raw appearance copy changed bytes')
        metrics = score_mask(image, mask, min_area_fraction=args.min_area_fraction, min_border_fraction=args.min_border_fraction, min_sharpness=args.min_sharpness)
        alignment = metric_alignment(metric_object(ann, args.object_id), ann)
        eligible = bool(metrics['eligible'] and alignment is not None)
        rejection = metrics['rejection'] if not metrics['eligible'] else None if alignment is not None else 'same_frame_corrected_metric_alignment_unavailable'
        row.update(status='available', source_mask_path=str(source), appearance_mask_path=str(destination), source_mask_sha256=sha256(source), appearance_mask_sha256=sha256(destination), source_size_wh=[mask.shape[1], mask.shape[0]], **metrics)
        row['anchor_eligible'] = eligible
        row['rejection'] = rejection
        rows.append(row)
        if eligible:
            ranked.append((float(metrics['score']), idx, row, alignment))
    manifest = {'schema': SCHEMA, 'authority': 'demo_only_visual_hypothesis', 'case': args.case_id, 'object_id': args.object_id, 'track_id': args.track_id, 'frame_count': len(rows), 'source_track': str(raw_track_path), 'source_track_sha256': sha256(raw_track_path), 'raw_frame_manifest': str(raw_manifest_path), 'raw_frame_manifest_sha256': sha256(raw_manifest_path), 'source_annotations': str(annotations_path), 'source_annotations_sha256': sha256(annotations_path), 'depth_order_binding': binding, 'raw_appearance_and_metric_eligibility_are_distinct': True, 'metric_owned_masks_consumed': False, 'frames': rows}
    (output / 'appearance_mask_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not ranked:
        (output / 'anchor_decision.json').write_text(json.dumps({'status':'no_eligible_anchor','candidate_scores':[{key:row.get(key) for key in ('frame_idx','anchor_eligible','score','rejection')} for row in rows]},indent=2)+'\n')
        raise ValueError('no raw appearance anchor has corrected metric alignment')
    (_, selected_idx, selected, alignment) = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
    anchor = {'schema': 'rgb_scene_depth_order_appearance_anchor_v1', 'authority': 'demo_only_visual_hypothesis', 'object_id': args.object_id, 'selected_frame_idx': selected_idx, 'selection': selected, 'appearance_conditioning': {'raw_rgb': selected['raw_rgb_path'], 'raw_rgb_sha256': selected['raw_rgb_sha256'], 'raw_sam2_mask': selected['appearance_mask_path'], 'raw_sam2_mask_sha256': selected['appearance_mask_sha256']}, 'metric_alignment': alignment, 'planes_are_distinct': True, 'physical_authority': False, 'candidate_scores': [{key: row.get(key) for key in ('frame_idx', 'anchor_eligible', 'rejection', 'score', 'score_terms')} for row in rows]}
    (output / 'anchor_decision.json').write_text(json.dumps(anchor, indent=2) + '\n')
    return (manifest, anchor)

def appearance_order_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'raw-frame-manifest', 'annotations', 'raw-sam2-track', 'depth-order-report', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('object-id', 'case-id', 'track-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--min-area-fraction', type=float, default=0.002)
    parser.add_argument('--min-border-fraction', type=float, default=0.005)
    parser.add_argument('--min-sharpness', type=float, default=2.0)
    return parser.parse_args(argv)
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
conditioning_SCHEMA = 'v19_experimental_p11_dual_geometry_inputs_v1'

def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f'expected JSON object: {path}')
    return payload

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda : f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f'missing {description}: {path}')
    return path

def prepare_new_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f'refusing to overwrite non-empty experiment output: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('L'), dtype=np.uint8) > 0

def mask_bbox(mask: np.ndarray) -> list[int]:
    (ys, xs) = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError('selected object mask is empty')
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]

def copy_byte_identical(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f'copy hash mismatch: {source} -> {destination}')
    return {'source': str(source), 'copy': str(destination), 'sha256': source_hash, 'bytes': int(source.stat().st_size), 'byte_identical': True}

def compare_masks(reference_path: Path, comparison_path: Path) -> dict[str, Any]:
    reference = load_mask(reference_path)
    comparison = load_mask(comparison_path)
    if reference.shape != comparison.shape:
        return {'reference': str(reference_path), 'comparison': str(comparison_path), 'same_shape': False, 'reference_shape_h_w': [int(v) for v in reference.shape], 'comparison_shape_h_w': [int(v) for v in comparison.shape]}
    intersection = int(np.logical_and(reference, comparison).sum())
    union = int(np.logical_or(reference, comparison).sum())
    xor = int(np.logical_xor(reference, comparison).sum())
    return {'reference': str(reference_path), 'comparison': str(comparison_path), 'same_shape': True, 'shape_h_w': [int(v) for v in reference.shape], 'reference_pixels': int(reference.sum()), 'comparison_pixels': int(comparison.sum()), 'xor_pixels': xor, 'intersection_pixels': intersection, 'union_pixels': union, 'iou': float(intersection / union) if union else 1.0, 'byte_identical': bool(sha256_file(reference_path) == sha256_file(comparison_path))}

def make_mask_review(rgb_path: Path, mask_path: Path, output_path: Path) -> None:
    rgb = Image.open(rgb_path).convert('RGB')
    mask_image = Image.open(mask_path).convert('L')
    if rgb.size != mask_image.size:
        raise RuntimeError(f'RGB/mask size mismatch: {rgb.size} vs {mask_image.size}')
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask_image, dtype=np.uint8) > 0
    overlay = rgb_arr.astype(np.float32)
    tint = np.asarray([20.0, 210.0, 60.0], dtype=np.float32)
    overlay[mask] = 0.62 * overlay[mask] + 0.38 * tint
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    eroded = padded[:-2, :-2] & padded[:-2, 1:-1] & padded[:-2, 2:] & padded[1:-1, :-2] & padded[1:-1, 1:-1] & padded[1:-1, 2:] & padded[2:, :-2] & padded[2:, 1:-1] & padded[2:, 2:]
    boundary = mask & ~eroded
    overlay[boundary] = np.asarray([255.0, 230.0, 20.0], dtype=np.float32)
    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode='RGB')
    max_panel = 640
    scale = min(1.0, max_panel / max(rgb.size))
    panel_size = (max(1, int(round(rgb.width * scale))), max(1, int(round(rgb.height * scale))))
    panels = [rgb.resize(panel_size), overlay_image.resize(panel_size)]
    canvas = Image.new('RGB', (panel_size[0] * 2, panel_size[1] + 36), (245, 245, 242))
    canvas.paste(panels[0], (0, 36))
    canvas.paste(panels[1], (panel_size[0], 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), 'full RGB (unchanged)', fill=(20, 20, 20))
    draw.text((panel_size[0] + 8, 10), 'object-owned mask overlay', fill=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

def build_completion_inputs(args: argparse.Namespace) -> dict[str, Any]:
    evidence_report_path = require_file(args.evidence_report, 'canonical P11 evidence report')
    evidence = load_json(evidence_report_path)
    selected = evidence.get('selected')
    if not isinstance(selected, dict):
        raise RuntimeError(f'P11 report has no selected object: {evidence_report_path}')
    frame_idx = int(evidence.get('selected_frame_idx', selected.get('frame_idx', -1)))
    if frame_idx < 0 or int(selected.get('frame_idx', frame_idx)) != frame_idx:
        raise RuntimeError('P11 selected frame fields are missing or inconsistent')
    raw_rgb = require_file(Path(str(selected.get('raw_frame_path', ''))), 'selected full RGB')
    selected_mask = require_file(Path(str(selected.get('mask_path', ''))), 'selected object-owned mask')
    trellis_info = selected.get('trellis_conditioning_crop')
    if not isinstance(trellis_info, dict):
        raise RuntimeError('P11 selected row has no trellis_conditioning_crop')
    trellis_crop = require_file(Path(str(trellis_info.get('crop_rgba', ''))), 'TRELLIS conditioning crop')
    visible_geometry = selected.get('visible_geometry_candidate')
    if not isinstance(visible_geometry, dict):
        visible_geometry = {}
    geometry_mask_text = str(visible_geometry.get('mask_path') or '')
    mask_provenance: dict[str, Any]
    if geometry_mask_text:
        geometry_mask = require_file(Path(geometry_mask_text), 'visible-geometry object-owned mask')
        selected_hash = sha256_file(selected_mask)
        geometry_hash = sha256_file(geometry_mask)
        mask_provenance = {'selected_mask': str(selected_mask), 'visible_geometry_mask': str(geometry_mask), 'selected_sha256': selected_hash, 'visible_geometry_sha256': geometry_hash, 'byte_identical': bool(selected_hash == geometry_hash)}
        if selected_hash != geometry_hash and (not args.allow_unverified_mask_provenance):
            raise RuntimeError('selected.mask_path is not byte-identical to selected.visible_geometry_candidate.mask_path; refusing SAM3D input')
    else:
        mask_provenance = {'selected_mask': str(selected_mask), 'visible_geometry_mask': None, 'byte_identical': None, 'verification': 'missing_visible_geometry_mask_path'}
        if not args.allow_unverified_mask_provenance:
            raise RuntimeError('selected.visible_geometry_candidate.mask_path is missing; use --allow-unverified-mask-provenance only for an explicit legacy experiment')
    rgb_image = Image.open(raw_rgb).convert('RGB')
    mask = load_mask(selected_mask)
    if (rgb_image.height, rgb_image.width) != mask.shape:
        raise RuntimeError(f'full RGB and object-owned mask are not pixel-aligned: RGB={(rgb_image.height, rgb_image.width)} mask={mask.shape}')
    bbox = mask_bbox(mask)
    output_dir = prepare_new_output_dir(args.output_dir)
    inputs_dir = output_dir / 'inputs'
    rgb_copy = inputs_dir / f"full_rgb{raw_rgb.suffix.lower() or '.img'}"
    mask_copy = inputs_dir / f"object_owned_mask{selected_mask.suffix.lower() or '.img'}"
    trellis_copy = inputs_dir / f"trellis_conditioning_rgba{trellis_crop.suffix.lower() or '.img'}"
    rgb_record = copy_byte_identical(raw_rgb, rgb_copy)
    mask_record = copy_byte_identical(selected_mask, mask_copy)
    trellis_record = copy_byte_identical(trellis_crop, trellis_copy)
    review_path = output_dir / 'review' / 'sam3d_native_input_mask_review.png'
    make_mask_review(rgb_copy, mask_copy, review_path)
    comparison = None
    if args.comparison_mask is not None:
        comparison_path = require_file(args.comparison_mask, 'comparison mask')
        comparison = compare_masks(selected_mask, comparison_path)
    report = {'schema': conditioning_SCHEMA, 'status': 'ok', 'method': 'build_experimental_p11_dual_geometry_inputs', 'claim_scope': 'additive P11 conditioning split only; canonical evidence and model outputs are read-only', 'source_evidence_report': str(evidence_report_path), 'case': evidence.get('case'), 'object_id': evidence.get('object_id'), 'selected_frame_idx': frame_idx, 'selection_rule': evidence.get('selection_rule'), 'selection_note': evidence.get('selection_note'), 'mask_provenance': mask_provenance, 'selected_input_summary': {'image_shape_h_w': [int(rgb_image.height), int(rgb_image.width)], 'mask_area_px': int(mask.sum()), 'mask_bbox_xyxy': bbox}, 'conditioning_contracts': {'trellis_native': {'input_kind': 'object_isolated_rgba_crop', 'image': str(trellis_copy), 'source_copy': trellis_record, 'external_depth_or_pointmap': None, 'pre_model_transform': 'canonical_P11_crop_only'}, 'sam3d_objects_native': {'input_kind': 'full_scene_rgb_plus_binary_object_owned_mask', 'image': str(rgb_copy), 'mask': str(mask_copy), 'image_source_copy': rgb_record, 'mask_source_copy': mask_record, 'external_pointmap': None, 'geometry_profile': 'legacy_internal_moge', 'pre_model_crop': None, 'pre_model_rotation': None, 'pre_model_rectification': None, 'model_internal_preprocessing': 'unmodified_SAM3D_internal_MoGe_then_joint_mask_crop'}}, 'comparison_mask_diagnostic': comparison, 'review': str(review_path), 'source_artifacts_mutated': False}
    report_path = output_dir / 'p11_dual_geometry_inputs_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report

def conditioning_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--comparison-mask', type=Path, default=None, help='Optional raw/pre-ownership mask recorded only as a diagnostic; never used for conditioning.')
    parser.add_argument('--allow-unverified-mask-provenance', action='store_true', help='Explicit legacy-only override when visible_geometry_candidate.mask_path is absent or differs.')
    return parser.parse_args(argv)
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any
P11_SCHEMA = 'v19_experimental_p11_dual_geometry_inputs_v1'
P12_SCHEMA = 'v19_experimental_p12_parallel_geometry_priors_v2'

def completion_load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f'expected JSON object: {path}')
    return payload

def completion_require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f'missing {description}: {path}')
    return path

def completion_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda : f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def file_record(path: Path, description: str) -> dict[str, Any]:
    path = completion_require_file(path, description)
    return {'path': str(path), 'sha256': completion_sha256_file(path), 'bytes': int(path.stat().st_size)}

def safe_id(value: Any) -> str:
    raw = str(value or 'unknown')
    cleaned = ''.join((ch if ch.isalnum() or ch in '._-' else '_' for ch in raw))
    return cleaned.strip('_') or 'unknown'

def resolve_case_report(payload: dict[str, Any], expected_name: str | None) -> dict[str, Any]:
    cases = payload.get('cases')
    if not isinstance(cases, list):
        return payload
    valid = [row for row in cases if isinstance(row, dict)]
    if expected_name is not None:
        matches = [row for row in valid if str(row.get('name')) == expected_name]
        if len(matches) == 1:
            return matches[0]
    if len(valid) == 1:
        return valid[0]
    raise RuntimeError(f"cannot select SAM3D case {expected_name!r} from report with case names {[row.get('name') for row in valid]}")

def validate_input_binding(candidate_report: dict[str, Any], expected_image: Path, expected_mask: Path) -> dict[str, Any]:
    actual_image = completion_require_file(Path(str(candidate_report.get('image', ''))), 'SAM3D report image')
    actual_mask = completion_require_file(Path(str(candidate_report.get('mask', ''))), 'SAM3D report mask')
    bindings = {'image': {'expected': str(expected_image), 'actual': str(actual_image), 'expected_sha256': completion_sha256_file(expected_image), 'actual_sha256': completion_sha256_file(actual_image)}, 'mask': {'expected': str(expected_mask), 'actual': str(actual_mask), 'expected_sha256': completion_sha256_file(expected_mask), 'actual_sha256': completion_sha256_file(actual_mask)}}
    for (name, record) in bindings.items():
        record['byte_identical'] = bool(record['expected_sha256'] == record['actual_sha256'])
        if not record['byte_identical']:
            raise RuntimeError(f'SAM3D {name} report is not bound to the P11 native input')
    return bindings

def validate_pointmap_conditioning(case, expected_image, expected_mask, *, expected_frame_idx=None,
                                    expected_pointmap_receipt=None, expected_geometry_profile=None):
    """Validate the native worker receipt for either anchor, without granting authority."""
    from observations.object.pointmap import load_pointmap, file_record, array_sha
    profile = case.get('geometry_profile', 'legacy_internal_moge')
    if expected_geometry_profile is not None and profile != expected_geometry_profile:
        raise ValueError('SAM3D conditioning profile mismatch')
    conditioning = case.get('external_pointmap')
    if profile == 'legacy_internal_moge':
        if conditioning is not None or expected_pointmap_receipt is not None:
            raise ValueError('SAM3D legacy profile cannot claim external pointmap conditioning')
        return None
    if profile != 'shared_metric_pointmap' or not isinstance(conditioning, dict):
        raise ValueError('SAM3D pointmap conditioning receipt missing')
    frame = conditioning.get('frame_idx')
    if expected_frame_idx is not None and frame != expected_frame_idx:
        raise ValueError('SAM3D pointmap conditioning frame mismatch')
    ref = conditioning.get('receipt', {})
    path = Path(ref.get('path', ''))
    if file_record(path) != ref or (expected_pointmap_receipt is not None and path.resolve() != Path(expected_pointmap_receipt).resolve()):
        raise ValueError('SAM3D pointmap receipt source/hash mismatch')
    points, receipt = load_pointmap(path, frame_idx=frame, image_path=expected_image, mask_path=expected_mask)
    if conditioning.get('schema') != 'sam3d_pointmap_consumption_v1' or conditioning.get('profile') != profile:
        raise ValueError('SAM3D pointmap consumption schema/profile mismatch')
    if conditioning.get('pointmap') != receipt['pointmap'] or conditioning.get('pointmap_array_sha256') != array_sha(points):
        raise ValueError('SAM3D consumed pointmap source/hash mismatch')
    for key, expected in {'pointmap_preserved_exactly': True, 'with_layout_postprocess': False,
                          'independent_exact_K_conditioning': False, 'inferred_K_read_by_generation': False}.items():
        if conditioning.get(key) is not expected:
            raise ValueError('SAM3D pointmap consumption invariant failed: ' + key)
    if conditioning.get('depth_model_calls') != 0 or conditioning.get('calls') != dict(run=1, compute=1, preprocess=1, sample=1, pose=1):
        raise ValueError('SAM3D pointmap consumption incomplete')
    if conditioning.get('inferred_K_role') != 'diagnostic_only_never_active_camera':
        raise ValueError('SAM3D inferred K cannot be active')
    normalized = conditioning.get('normalized_conditioning', {})
    if set(normalized) != {'pointmap', 'rgb_pointmap', 'pointmap_scale', 'pointmap_shift', 'rgb_pointmap_scale', 'rgb_pointmap_shift'}:
        raise ValueError('SAM3D pointmap normalization receipt incomplete')
    for key, source in (('scene_scale', 'pointmap_scale'), ('scene_shift', 'pointmap_shift')):
        value = np.asarray(conditioning.get('pose_decoder', {}).get(key), dtype=normalized[source]['dtype'])
        if list(value.shape) != normalized[source]['shape'] or array_sha(value) != normalized[source]['sha256']:
            raise ValueError('SAM3D pointmap normalization/pose receipt mismatch')
    return conditioning


def sam3d_candidate_record(report_path: Path, expected_name: str | None, expected_image: Path, expected_mask: Path, invocation: dict[str, Any], *, expected_frame_idx=None, expected_pointmap_receipt=None, expected_geometry_profile=None) -> dict[str, Any]:
    payload = completion_load_json(report_path)
    if payload.get('status') != 'ok':
        raise RuntimeError(f'SAM3D runner report is not ok: {report_path}')
    moge_offline = payload.get('moge_offline') if isinstance(payload.get('moge_offline'), dict) else {}
    dinov2_offline = payload.get('dinov2_offline') if isinstance(payload.get('dinov2_offline'), dict) else {}
    hf_offline = payload.get('huggingface_offline') if isinstance(payload.get('huggingface_offline'), dict) else {}
    expected_moge_sha256 = 'da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f'
    expected_dinov2_sha256 = '36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51'
    expected_hubconf_sha256 = 'c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64'
    if moge_offline.get('checkpoint_sha256') != expected_moge_sha256 or moge_offline.get('network_resolution_allowed') is not False or dinov2_offline.get('checkpoint_sha256') != expected_dinov2_sha256 or (dinov2_offline.get('hubconf_sha256') != expected_hubconf_sha256) or (dinov2_offline.get('network_resolution_allowed') is not False) or (hf_offline.get('HF_HUB_OFFLINE') != '1') or (hf_offline.get('TRANSFORMERS_OFFLINE') != '1') or (hf_offline.get('network_resolution_allowed') is not False):
        raise RuntimeError('SAM3D runner lacks byte-bound offline MoGe/DINOv2 provenance')
    case = resolve_case_report(payload, expected_name)
    case_moge = case.get('moge_offline') if isinstance(case.get('moge_offline'), dict) else {}
    case_dinov2 = case.get('dinov2_offline') if isinstance(case.get('dinov2_offline'), dict) else {}
    if case_moge != moge_offline or case_dinov2 != dinov2_offline:
        raise RuntimeError('SAM3D case report offline asset binding differs from its aggregate runner report')
    if case.get('status') != 'ok':
        raise RuntimeError(f'SAM3D candidate report is not ok: {report_path}')
    mesh = completion_require_file(Path(str(case.get('mesh', ''))), 'SAM3D raw mesh')
    glb = completion_require_file(Path(str(case.get('glb', ''))), 'SAM3D raw GLB')
    gaussian_text = str(case.get('gaussian') or '')
    gaussian = completion_require_file(Path(gaussian_text), 'SAM3D Gaussian') if gaussian_text else None
    native_p3d = completion_require_file(Path(str(case.get('native_pose_mesh_pytorch3d_camera') or '')), 'SAM3D native-pose PyTorch3D-camera mesh')
    native_opencv = completion_require_file(Path(str(case.get('native_pose_mesh_opencv_camera') or '')), 'SAM3D native-pose OpenCV-camera mesh')
    native_contract = case.get('native_pose_contract')
    if not isinstance(native_contract, dict) or native_contract.get('quaternion_order') != 'wxyz_scalar_first_pytorch3d':
        raise RuntimeError('SAM3D runner lacks the verified native pose coordinate contract')
    if native_contract.get('metric_status') != 'native_monocular_scene_units_not_sensor_meters':
        raise RuntimeError('SAM3D runner incorrectly claims native pose is sensor metric')
    pointmap_conditioning = validate_pointmap_conditioning(case, expected_image, expected_mask,
        expected_frame_idx=expected_frame_idx, expected_pointmap_receipt=expected_pointmap_receipt,
        expected_geometry_profile=expected_geometry_profile)
    return {'source_model': 'sam3d_objects', 'status': 'generated_native_raw_prior', 'runner_report': str(report_path), 'conditioning': {'input_kind': 'full_scene_rgb_plus_binary_object_owned_mask', 'image': file_record(expected_image, 'P11 SAM3D image'), 'mask': file_record(expected_mask, 'P11 SAM3D mask'), 'external_pointmap': pointmap_conditioning, 'geometry_profile': case.get('geometry_profile', 'legacy_internal_moge'), 'pre_model_crop': None, 'pre_model_rotation': None, 'pre_model_rectification': None}, 'input_binding': validate_input_binding(case, expected_image, expected_mask), 'offline_asset_binding': {'moge': moge_offline, 'dinov2': dinov2_offline, 'huggingface_offline': hf_offline}, 'native_outputs': {'raw_mesh': file_record(mesh, 'SAM3D raw mesh'), 'glb': file_record(glb, 'SAM3D GLB'), 'gaussian': file_record(gaussian, 'SAM3D Gaussian') if gaussian is not None else None, 'native_pose_mesh_pytorch3d_camera': file_record(native_p3d, 'SAM3D native P3D mesh'), 'native_pose_mesh_opencv_camera': file_record(native_opencv, 'SAM3D native OpenCV mesh'), 'mesh_stats': case.get('mesh_stats'), 'native_pose': case.get('pose'), 'native_pose_contract': native_contract}, 'invocation': invocation, 'semantics': {'mesh_frame': 'sam3d_native_local_generator_frame', 'native_pose_frame': 'verified_pytorch3d_camera_then_explicit_opencv_bridge', 'native_pose_metric_status': 'native_monocular_scene_units_not_sensor_meters', 'metric_scale_ready': False, 'v19_canonical_frame_ready': False, 'collision_surface_ready': False, 'allowed_use': 'raw geometry prior evaluation and later explicit P13 adapter only'}}

def trellis_candidate_record(report_path: Path, expected_crop: Path) -> dict[str, Any]:
    report = completion_load_json(report_path)
    if report.get('status') != 'ok':
        raise RuntimeError(f'TRELLIS reference report is not ok: {report_path}')
    image = completion_require_file(Path(str(report.get('image', ''))), 'TRELLIS report input image')
    if completion_sha256_file(image) != completion_sha256_file(expected_crop):
        raise RuntimeError('TRELLIS report input is not byte-identical to the P11 TRELLIS crop')
    mesh = completion_require_file(Path(str(report.get('mesh', ''))), 'TRELLIS raw mesh')
    gaussian_text = str(report.get('gaussian') or '')
    glb_text = str(report.get('glb') or '')
    return {'source_model': 'trellis', 'status': 'frozen_existing_raw_prior_reference', 'runner_report': str(report_path), 'conditioning': {'input_kind': 'object_isolated_rgba_crop', 'image': file_record(expected_crop, 'P11 TRELLIS crop'), 'input_binding': {'report_image': str(image), 'byte_identical': True}}, 'native_outputs': {'raw_mesh': file_record(mesh, 'TRELLIS raw mesh'), 'gaussian': file_record(Path(gaussian_text), 'TRELLIS Gaussian') if gaussian_text else None, 'glb': file_record(Path(glb_text), 'TRELLIS GLB') if glb_text else None, 'mesh_stats': {'vertices': report.get('vertices'), 'faces': report.get('faces'), 'extent_model_units': report.get('extent_model_units'), 'center_model_units': report.get('center_model_units')}}, 'semantics': {'mesh_frame': 'trellis_native_local_object_frame', 'metric_scale_ready': False, 'v19_canonical_frame_ready': False, 'collision_surface_ready': False, 'allowed_use': 'frozen baseline raw geometry prior reference'}, 'source_artifacts_mutated': False}
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy.spatial import cKDTree
from utils.geometry import apply_native_pose
native_bridge_SCHEMA = 'v19_sam3d_native_sensor_metric_canonical_bridge_v1'
COMPATIBILITY_SCHEMA = 'v19_metric_canonical_generated_prior_input_v1'

def native_bridge_load_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise RuntimeError(f'expected JSON object: {path}')
    return payload

def native_bridge_require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f'missing or empty {description}: {path}')
    return path

def native_bridge_finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise RuntimeError(f'{label} must be finite length {size}, got {array}')
    return array

def robust_point_stats(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or (not np.isfinite(points).all()):
        raise RuntimeError('point stats require finite Nx3 points')
    lo = np.percentile(points, 0.5, axis=0)
    hi = np.percentile(points, 99.5, axis=0)
    extent = hi - lo
    return {'count': int(len(points)), 'center_mean': points.mean(axis=0).astype(float).tolist(), 'center_median': np.median(points, axis=0).astype(float).tolist(), 'raw_extent': (points.max(axis=0) - points.min(axis=0)).astype(float).tolist(), 'raw_extent_diag': float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 'robust_p005_p995_extent': extent.astype(float).tolist(), 'robust_p005_p995_extent_diag': float(np.linalg.norm(extent))}

def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    import trimesh
    rng = np.random.default_rng(1313)
    (points, _) = trimesh.sample.sample_surface(mesh, min(int(count), max(1, int(len(mesh.faces)) * 2)), seed=rng)
    return np.asarray(points, dtype=np.float64)

def nearest_surface_summary(query: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    (distances, _) = cKDTree(np.asarray(target, dtype=np.float64)).query(np.asarray(query, dtype=np.float64), k=1, workers=-1)
    return {'count': int(len(distances)), 'median_m': float(np.median(distances)), 'p90_m': float(np.percentile(distances, 90)), 'p95_m': float(np.percentile(distances, 95)), 'mean_m': float(np.mean(distances)), 'max_m': float(np.max(distances))}

def mesh_from_path(path: Path) -> trimesh.Trimesh:
    import trimesh
    loaded = trimesh.load(str(path), process=False, force='mesh')
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f'not a triangular mesh: {path}')
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or (not np.isfinite(vertices).all()):
        raise RuntimeError(f'invalid mesh vertices: {path}')
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f'invalid mesh faces: {path}')
    return loaded

def resolve_sam3d_candidate(p12: dict[str, Any]) -> dict[str, Any]:
    candidates = p12.get('candidates')
    if isinstance(candidates, dict) and isinstance(candidates.get('sam3d_objects'), dict):
        candidate = candidates['sam3d_objects']
    elif str(p12.get('source_model')) == 'sam3d_objects':
        candidate = p12
    else:
        raise RuntimeError('P12 report has no sam3d_objects candidate')
    native = candidate.get('native_outputs')
    if not isinstance(native, dict):
        raise RuntimeError('SAM3D P12 candidate lacks native_outputs')
    raw = native.get('raw_mesh')
    raw_path = Path(str(raw.get('path'))) if isinstance(raw, dict) else Path(str(raw or ''))
    raw_path = native_bridge_require_file(raw_path, 'SAM3D raw local mesh')
    pose = native.get('native_pose')
    if not isinstance(pose, dict):
        raise RuntimeError('SAM3D P12 candidate lacks native_pose')
    native_contract = native.get('native_pose_contract')
    if not isinstance(native_contract, dict):
        raise RuntimeError('SAM3D P12 candidate lacks native_pose_contract')
    if native_contract.get('quaternion_order') != 'wxyz_scalar_first_pytorch3d':
        raise RuntimeError('SAM3D P12 candidate has an unsupported quaternion convention')
    native_opencv_record = native.get('native_pose_mesh_opencv_camera')
    native_opencv_path = Path(str(native_opencv_record.get('path') or '')) if isinstance(native_opencv_record, dict) else Path('')
    native_opencv_path = native_bridge_require_file(native_opencv_path, 'SAM3D exported native-pose OpenCV mesh')
    conditioning = candidate.get('conditioning') if isinstance(candidate.get('conditioning'), dict) else {}
    return {'candidate': candidate, 'native': native, 'raw_mesh': raw_path, 'pose': pose, 'native_pose_contract': native_contract, 'native_opencv_mesh': native_opencv_path, 'conditioning': conditioning}

def resolve_anchor_centroid_world(evidence: dict[str, Any], visible: dict[str, Any]) -> np.ndarray:
    binding = evidence.get('selected_anchor_atomic_binding')
    if not isinstance(binding, dict) or binding.get('validated') is not True:
        raise RuntimeError('evidence lacks a validated atomic selected-anchor binding')
    selected_frame_idx = int(evidence.get('selected_frame_idx', -1))
    if int(binding.get('selected_frame_idx', -2)) != selected_frame_idx or int(binding.get('visible_geometry_frame_idx', -2)) != selected_frame_idx or int(binding.get('canonical_surface_frame_idx', -2)) != selected_frame_idx:
        raise RuntimeError(f'selected-anchor frame binding is inconsistent: {binding}')
    row = evidence.get('depth_fused_object_row') if isinstance(evidence.get('depth_fused_object_row'), dict) else {}
    reconstruction = row.get('mesh_reconstruction') if isinstance(row.get('mesh_reconstruction'), dict) else {}
    if reconstruction.get('atomic_anchor_binding') is not True:
        raise RuntimeError('selected mesh reconstruction is not atomically bound to the P11 row')
    mesh_frame_idx = int(reconstruction.get('anchor_frame_idx', -1))
    visible_frame_idx = int(visible.get('frame_idx', -1))
    if mesh_frame_idx != selected_frame_idx or visible_frame_idx != selected_frame_idx:
        raise RuntimeError(f'SAM3D bridge received mixed anchor frames: selected={selected_frame_idx}, visible={visible_frame_idx}, canonical_surface={mesh_frame_idx}')
    mesh_centroid = native_bridge_finite_vector(reconstruction.get('anchor_centroid_world_m'), 3, 'selected canonical-surface centroid')
    visible_centroid = native_bridge_finite_vector(visible.get('centroid_world_m'), 3, 'selected visible centroid')
    error_m = float(np.linalg.norm(mesh_centroid - visible_centroid))
    if error_m > 1e-06:
        raise RuntimeError(f'selected visible/canonical centroids disagree by {error_m} m')
    return visible_centroid

def resolve_mask_intrinsics(visible: dict[str, Any], mask_size_wh: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    intrinsics = native_bridge_finite_vector(visible.get('intrinsics_fx_fy_cx_cy'), 4, 'depth-plane intrinsics')
    K_depth = np.asarray([[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform = visible.get('mask_depth_transform_contract')
    if not isinstance(transform, dict) or transform.get('camera_contract_consistent') is not True:
        raise RuntimeError('visible evidence lacks a camera-consistent mask/depth transform')
    declared_mask_size = [int(value) for value in transform.get('mask_size_wh') or []]
    if declared_mask_size != [int(mask_size_wh[0]), int(mask_size_wh[1])]:
        raise RuntimeError(f'mask size {mask_size_wh} disagrees with transform contract {declared_mask_size}')
    A_depth_from_mask = np.asarray(transform.get('A_depth_from_mask_coordinate_model'), dtype=np.float64)
    if A_depth_from_mask.shape != (3, 3) or not np.isfinite(A_depth_from_mask).all():
        raise RuntimeError('invalid A_depth_from_mask_coordinate_model')
    K_mask = np.linalg.inv(A_depth_from_mask) @ K_depth
    if not np.allclose(K_mask[2], [0.0, 0.0, 1.0], atol=1e-09):
        raise RuntimeError('mask-plane K is not pinhole')
    intr_mask = np.asarray([K_mask[0, 0], K_mask[1, 1], K_mask[0, 2], K_mask[1, 2]], dtype=np.float64)
    if not np.isfinite(intr_mask).all() or np.any(intr_mask[:2] <= 0.0):
        raise RuntimeError(f'invalid mask-plane intrinsics: {intr_mask}')
    return (intr_mask, {'depth_intrinsics_fx_fy_cx_cy': intrinsics.tolist(), 'A_depth_from_mask': A_depth_from_mask.tolist(), 'mask_intrinsics_fx_fy_cx_cy': intr_mask.tolist()})

def native_bridge_convex_projection_and_front_depth(vertices_camera: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices_camera, dtype=np.float64)
    (fx, fy, cx, cy) = native_bridge_finite_vector(intrinsics, 4, 'projection intrinsics')
    (height, width) = mask.shape
    z = vertices[:, 2]
    valid = np.isfinite(vertices).all(axis=1) & (z > 1e-05)
    if np.count_nonzero(valid) < 3:
        raise RuntimeError('SAM3D native pose has fewer than three positive-depth vertices')
    u = fx * vertices[valid, 0] / z[valid] + cx
    v = fy * vertices[valid, 1] / z[valid] + cy
    uv = np.column_stack((u, v))
    finite_uv = np.isfinite(uv).all(axis=1)
    uv = uv[finite_uv]
    z_projected = z[valid][finite_uv]
    inside = (uv[:, 0] >= 0.0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= height - 1)
    uv_inside = uv[inside]
    z_inside = z_projected[inside]
    if len(uv_inside) < 3:
        raise RuntimeError('SAM3D native mesh projection does not enter the owned-mask image')
    hull = cv2.convexHull(np.rint(uv_inside).astype(np.int32))
    projected = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(projected, hull, 1)
    target = np.asarray(mask, dtype=bool)
    projected_bool = projected > 0
    intersection = int(np.count_nonzero(projected_bool & target))
    union = int(np.count_nonzero(projected_bool | target))
    iou = float(intersection / max(1, union))
    xi = np.rint(uv_inside[:, 0]).astype(np.int64)
    yi = np.rint(uv_inside[:, 1]).astype(np.int64)
    pixel_inside_target = target[yi, xi]
    xi = xi[pixel_inside_target]
    yi = yi[pixel_inside_target]
    z_owned = z_inside[pixel_inside_target]
    if len(z_owned) == 0:
        raise RuntimeError('SAM3D native projected vertices have no object-owned depth samples')
    flat_index = yi * width + xi
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, flat_index, z_owned)
    finite_zbuffer = zbuffer[np.isfinite(zbuffer)]
    return {'positive_depth_vertex_fraction': float(np.count_nonzero(valid) / len(vertices)), 'projected_vertices_inside_image': int(len(uv_inside)), 'projected_vertices_inside_owned_mask': int(len(z_owned)), 'owned_zbuffer_pixel_count': int(len(finite_zbuffer)), 'owned_zbuffer_coverage_fraction': float(len(finite_zbuffer) / max(1, np.count_nonzero(target))), 'owned_zbuffer_median_native_depth': float(np.median(finite_zbuffer)), 'convex_projection_iou': iou, 'convex_projection_pixels': int(np.count_nonzero(projected_bool)), 'owned_mask_pixels': int(np.count_nonzero(target))}

def bind_sam3d_metric(args: argparse.Namespace) -> dict[str, Any]:
    from gates.object.geometry import observed_front_quality_decision
    import trimesh
    evidence_path = native_bridge_require_file(args.evidence_report, 'P11 evidence report')
    p12_path = native_bridge_require_file(args.p12_report, 'P12 parallel report')
    evidence = native_bridge_load_json(evidence_path)
    p12 = native_bridge_load_json(p12_path)
    selected = evidence.get('selected')
    if not isinstance(selected, dict):
        raise RuntimeError('evidence has no selected anchor')
    visible = selected.get('visible_geometry_candidate')
    if not isinstance(visible, dict):
        raise RuntimeError('selected anchor lacks visible_geometry_candidate')
    ownership = visible.get('first_surface_depth_ownership')
    if not isinstance(ownership, dict) or ownership.get('enabled') is not True:
        raise RuntimeError('SAM3D metric bridge requires explicit robust first-surface depth ownership')
    if ownership.get('fail_closed') is True or ownership.get('failure_reasons'):
        raise RuntimeError(f"selected first-surface depth ownership is failed closed: {ownership.get('failure_reasons')}")
    if float(ownership.get('retained_fraction') or 0.0) < float(args.min_robust_depth_support_fraction):
        raise RuntimeError('selected robust first-surface support is below the bridge threshold')
    candidate = resolve_sam3d_candidate(p12)
    raw_mesh_path = candidate['raw_mesh']
    raw_mesh = mesh_from_path(raw_mesh_path)
    raw_vertices = np.asarray(raw_mesh.vertices, dtype=np.float64)
    pose = candidate['pose']
    rotation = native_bridge_finite_vector(pose.get('rotation'), 4, 'SAM3D native quaternion')
    translation = native_bridge_finite_vector(pose.get('translation'), 3, 'SAM3D native translation')
    scale = np.asarray(pose.get('scale'), dtype=np.float64).reshape(-1)
    (native_p3d, native_opencv, native_contract) = apply_native_pose(raw_vertices, rotation, translation, scale)
    exported_native_opencv = mesh_from_path(candidate['native_opencv_mesh'])
    exported_native_vertices = np.asarray(exported_native_opencv.vertices, dtype=np.float64)
    if exported_native_vertices.shape != native_opencv.shape:
        raise RuntimeError('SAM3D exported native-pose mesh changed vertex topology')
    native_pose_mesh_max_abs_error = float(np.max(np.abs(exported_native_vertices - native_opencv)))
    if native_pose_mesh_max_abs_error > 1e-05:
        raise RuntimeError(f'SAM3D exported native-pose mesh disagrees with the declared transform: {native_pose_mesh_max_abs_error}')
    mask_path = native_bridge_require_file(Path(str(selected.get('mask_path') or visible.get('mask_path') or '')), 'selected object-owned mask')
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None or np.count_nonzero(mask_image) == 0:
        raise RuntimeError(f'cannot decode nonempty owned mask: {mask_path}')
    mask = mask_image > 0
    (mask_h, mask_w) = mask.shape
    (mask_intrinsics, raster_contract) = resolve_mask_intrinsics(visible, (mask_w, mask_h))
    projection = native_bridge_convex_projection_and_front_depth(native_opencv, mask_intrinsics, mask)
    if projection['positive_depth_vertex_fraction'] < float(args.min_positive_depth_fraction):
        raise RuntimeError(f"SAM3D native positive-depth fraction failed: {projection['positive_depth_vertex_fraction']}")
    if projection['convex_projection_iou'] < float(args.min_native_convex_projection_iou):
        raise RuntimeError(f"SAM3D native projection IoU failed: {projection['convex_projection_iou']}")
    if projection['owned_zbuffer_pixel_count'] < int(args.min_native_owned_zbuffer_pixels):
        raise RuntimeError(f"SAM3D native owned z-buffer support failed: {projection['owned_zbuffer_pixel_count']}")
    observed_camera = np.asarray(visible.get('camera_vertices_sample_m'), dtype=np.float64)
    if observed_camera.ndim != 2 or observed_camera.shape[1] != 3 or len(observed_camera) < int(args.min_observed_points):
        raise RuntimeError(f'insufficient robust observed camera points: {observed_camera.shape}')
    observed_camera = observed_camera[np.isfinite(observed_camera).all(axis=1) & (observed_camera[:, 2] > 0.0)]
    if len(observed_camera) < int(args.min_observed_points):
        raise RuntimeError('insufficient finite positive robust observed camera points')
    sensor_depth_median = float(np.median(observed_camera[:, 2]))
    native_front_depth = float(projection['owned_zbuffer_median_native_depth'])
    metric_scale = sensor_depth_median / native_front_depth
    if not np.isfinite(metric_scale) or not float(args.min_scene_similarity_scale) <= metric_scale <= float(args.max_scene_similarity_scale):
        raise RuntimeError(f'SAM3D scene similarity scale outside contract: {metric_scale}')
    metric_camera = native_opencv * metric_scale
    camera = selected.get('camera')
    if not isinstance(camera, dict):
        raise RuntimeError('selected anchor lacks camera transform')
    T_world_camera = np.asarray(camera.get('T_world_camera_metric') or camera.get('T_world_camera'), dtype=np.float64)
    if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all() or (not np.allclose(T_world_camera[3], [0, 0, 0, 1])):
        raise RuntimeError('invalid selected T_world_camera_metric')
    anchor_centroid_world = resolve_anchor_centroid_world(evidence, visible)
    metric_world = metric_camera @ T_world_camera[:3, :3].T + T_world_camera[:3, 3][None, :]
    canonical = metric_world - anchor_centroid_world[None, :]
    observed_metric_canonical = observed_camera @ T_world_camera[:3, :3].T + T_world_camera[:3, 3][None, :] - anchor_centroid_world[None, :]
    generated_stats = robust_point_stats(canonical)
    observed_stats = robust_point_stats(observed_camera)
    extent_ratio = float(generated_stats['robust_p005_p995_extent_diag'] / max(observed_stats['robust_p005_p995_extent_diag'], 1e-12))
    if not float(args.min_complete_to_observed_extent_ratio) <= extent_ratio <= float(args.max_complete_to_observed_extent_ratio):
        raise RuntimeError(f'SAM3D complete/observed robust extent ratio outside contract: {extent_ratio}')
    metric_mesh = trimesh.Trimesh(vertices=canonical, faces=np.asarray(raw_mesh.faces, dtype=np.int64), process=False)
    generated_surface_samples = deterministic_sample_mesh(metric_mesh, int(args.geometry_quality_surface_samples))
    observed_to_generated = nearest_surface_summary(observed_metric_canonical, generated_surface_samples)
    observed_extent_diag = float(observed_stats['robust_p005_p995_extent_diag'])
    median_fraction = float(observed_to_generated['median_m'] / max(observed_extent_diag, 1e-12))
    p95_fraction = float(observed_to_generated['p95_m'] / max(observed_extent_diag, 1e-12))
    quality_decision = observed_front_quality_decision(median_fraction=median_fraction, p95_fraction=p95_fraction, native_projection_iou=float(projection['convex_projection_iou']), maximum_median_fraction=float(args.max_observed_to_generated_median_extent_fraction), strict_maximum_p95_fraction=float(args.strict_observed_to_generated_p95_extent_fraction), conditional_maximum_p95_fraction=float(args.max_observed_to_generated_p95_extent_fraction), conditional_minimum_projection_iou=float(args.min_conditional_tail_native_projection_iou))
    geometry_quality = {'method': 'selected_anchor_observed_metric_surfels_to_sampled_generated_surface', 'generated_surface_sample_count': int(len(generated_surface_samples)), 'observed_to_generated': observed_to_generated, 'observed_robust_extent_diag_m': observed_extent_diag, 'observed_to_generated_median_fraction_of_extent_diag': median_fraction, 'observed_to_generated_p95_fraction_of_extent_diag': p95_fraction, 'maximum_median_fraction_of_extent_diag': float(args.max_observed_to_generated_median_extent_fraction), 'maximum_p95_fraction_of_extent_diag': float(args.max_observed_to_generated_p95_extent_fraction), **quality_decision, 'generated_faces_pose_evidence_consumed': False, 'interpretation': "Render-prior coverage of the selected frame's measured front surface only. The strict tier remains P95/extent <= 0.15. A bounded conditional tail tier may carry explicit uncertainty only when the strict median still passes and native mask projection is substantial. This diagnostic cannot promote generated hidden faces to pose/contact/collision evidence."}
    if not geometry_quality['quality_passed']:
        raise RuntimeError(f'SAM3D observed-front geometry quality failed: {geometry_quality}')
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f'refusing to overwrite nonempty output: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mesh_path = output_dir / 'sam3d_native_sensor_metric_canonical_render_prior.ply'
    output_mesh = metric_mesh.copy()
    output_mesh.export(str(output_mesh_path))
    report_path = output_dir / 'sam3d_native_sensor_metric_canonical_bridge_report.json'
    compatibility_path = output_dir / 'p13_metric_canonical_input.json'
    report = {'schema': native_bridge_SCHEMA, 'status': 'ok_render_prior_metric_canonical', 'method': 'bridge_sam3d_native_pose_to_sensor_metric_canonical', 'annotation_ready': False, 'claim_scope': 'Prediction-side SAM3D generated render prior only. Native orientation/image pose are preserved; sensor metric scale comes from robust owned first-surface depth. Generated faces are not pose, contact, collision, sign, or nonpenetration evidence.', 'inputs': {'evidence_report': str(evidence_path), 'p12_report': str(p12_path), 'raw_mesh': str(raw_mesh_path), 'raw_mesh_sha256': content_sha256(raw_mesh_path), 'object_owned_mask': str(mask_path)}, 'native_pose_contract': {**native_contract, 'p12_exported_contract': candidate['native_pose_contract'], 'exported_native_pose_opencv_mesh': str(candidate['native_opencv_mesh']), 'exported_vs_recomputed_max_abs_error': native_pose_mesh_max_abs_error}, 'native_pose': {'rotation_wxyz': rotation.tolist(), 'translation_native_scene': translation.tolist(), 'scale_native_scene_per_model_unit': scale.astype(float).tolist(), 'native_pytorch3d_camera_stats': robust_point_stats(native_p3d), 'native_opencv_camera_stats': robust_point_stats(native_opencv)}, 'projection_validation': {**projection, 'minimum_convex_projection_iou': float(args.min_native_convex_projection_iou), 'minimum_owned_zbuffer_pixels': int(args.min_native_owned_zbuffer_pixels), 'raster_contract': raster_contract}, 'sensor_metric_scene_similarity': {'sensor_robust_first_surface_median_depth_m': sensor_depth_median, 'native_owned_zbuffer_median_depth': native_front_depth, 'camera_origin_scene_similarity_scale': metric_scale, 'formula': 'p_cv_metric = lambda * (((p_raw * native_scale) @ R_wxyz + native_translation) @ diag(-1,-1,1))', 'translation_and_object_scale_scaled_together': True, 'robust_first_surface_depth_ownership': ownership}, 'shared_canonical_frame': {'T_world_camera_metric': T_world_camera.tolist(), 'anchor_centroid_world_m': anchor_centroid_world.tolist(), 'formula': 'p_canonical = p_cv_metric @ R_world_camera.T + t_world_camera - anchor_centroid_world'}, 'geometry_validation': {'observed_robust_camera_surface': observed_stats, 'generated_metric_canonical': generated_stats, 'complete_to_observed_robust_extent_diag_ratio': extent_ratio, 'observed_front_surface_quality': geometry_quality, 'allowed_extent_ratio': [float(args.min_complete_to_observed_extent_ratio), float(args.max_complete_to_observed_extent_ratio)]}, 'semantics': {'generated_faces_render_underlay': True, 'generated_faces_pose_eligible': False, 'generated_faces_collision_eligible': False, 'generated_faces_contact_eligible': False, 'generated_faces_signed_distance_eligible': False, 'observed_front_quality_acceptance_mode': geometry_quality['acceptance_mode'], 'conditional_observed_tail_uncertainty': geometry_quality['conditional_tail_uncertainty']}, 'outputs': {'metric_canonical_render_prior': str(output_mesh_path), 'p13_metric_canonical_input': str(compatibility_path)}}
    write_json_utf8(report_path, report)
    compatibility = {'schema': COMPATIBILITY_SCHEMA, 'status': 'verified_sam3d_native_sensor_metric_canonical_render_prior', 'source_model': 'sam3d_objects', 'mesh': str(output_mesh_path), 'mesh_sha256': content_sha256(output_mesh_path), 'bridge_report': str(report_path), 'bridge_report_sha256': content_sha256(report_path), 'metric_canonical_input_contract': {'frame': 'shared_completed_canonical_meters', 'alignment_must_be_identity': True, 'generated_faces_render_only': True, 'native_orientation_preserved': True, 'camera_origin_scene_similarity_applied': True}}
    write_json_utf8(compatibility_path, compatibility)
    print(json.dumps({'status': report['status'], 'output_mesh': str(output_mesh_path), 'metric_scale': metric_scale, 'projection_iou': projection['convex_projection_iou'], 'extent_ratio': extent_ratio, 'report': str(report_path), 'p13_input': str(compatibility_path)}, indent=2))
    return report

def native_bridge_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-report', type=Path, required=True)
    parser.add_argument('--p12-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--min-robust-depth-support-fraction', type=float, default=0.9)
    parser.add_argument('--min-positive-depth-fraction', type=float, default=0.99)
    parser.add_argument('--min-native-convex-projection-iou', type=float, default=0.05)
    parser.add_argument('--min-native-owned-zbuffer-pixels', type=int, default=100)
    parser.add_argument('--min-observed-points', type=int, default=100)
    parser.add_argument('--min-scene-similarity-scale', type=float, default=0.05)
    parser.add_argument('--max-scene-similarity-scale', type=float, default=5.0)
    parser.add_argument('--min-complete-to-observed-extent-ratio', type=float, default=0.25)
    parser.add_argument('--max-complete-to-observed-extent-ratio', type=float, default=5.0)
    parser.add_argument('--geometry-quality-surface-samples', type=int, default=40000)
    parser.add_argument('--max-observed-to-generated-median-extent-fraction', type=float, default=0.05)
    parser.add_argument('--strict-observed-to-generated-p95-extent-fraction', type=float, default=0.15, help='Unchanged strict observed-front P95/extent tier.')
    parser.add_argument('--max-observed-to-generated-p95-extent-fraction', type=float, default=0.18, help='Maximum bounded conditional P95/extent tier; reported as uncertain when above strict.')
    parser.add_argument('--min-conditional-tail-native-projection-iou', type=float, default=0.25, help='Minimum native convex projection IoU required by the conditional P95-tail tier.')
    return parser.parse_args(argv)
import argparse
import hashlib
import itertools
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy.spatial import cKDTree

def completion_support_safe_id(value: str) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '_', value).strip('_')

def load_vertices(path: Path) -> np.ndarray:
    import trimesh
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        parts = [g for g in geom.geometry.values() if hasattr(g, 'vertices')]
        if not parts:
            raise RuntimeError(f'no vertex geometry in {path}')
        verts = np.concatenate([np.asarray(g.vertices, dtype=float) for g in parts], axis=0)
    else:
        verts = np.asarray(geom.vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
        raise RuntimeError(f'invalid vertices in {path}')
    if not np.isfinite(verts).all():
        raise RuntimeError(f'non-finite vertices in {path}')
    return verts

def load_mesh(path: Path) -> trimesh.Trimesh:
    import trimesh
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f'no mesh geometry in {path}')
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f'not a mesh: {path}')
    if len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f'empty mesh: {path}')
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)

def nearest_stats(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    (d, _) = cKDTree(target).query(query, k=1, workers=-1)
    return {'mean_m': float(np.mean(d)), 'median_m': float(np.median(d)), 'p90_m': float(np.percentile(d, 90)), 'p95_m': float(np.percentile(d, 95)), 'max_m': float(np.max(d))}

def completion_support_deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    import trimesh
    count = min(max(1, count), max(1, len(mesh.faces) * 2))
    rng = np.random.default_rng(18)
    (pts, _) = trimesh.sample.sample_surface(mesh, count, seed=rng)
    return np.asarray(pts, dtype=float)

def face_center_labels(mesh: trimesh.Trimesh, observed_points: np.ndarray, observed_band_m: float) -> tuple[np.ndarray, np.ndarray]:
    import trimesh
    centers = np.asarray(mesh.triangles_center, dtype=float)
    (d, _) = cKDTree(observed_points).query(centers, k=1, workers=-1)
    near = d <= observed_band_m
    return (d, near)

def export_label_sidecar(path: Path, labels: list[str], distances: np.ndarray, source_mesh: str, offset: int=0) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    sidecar = {'source_mesh': source_mesh, 'face_index_offset_in_combined_mesh': int(offset), 'face_count': len(labels), 'label_counts': counts, 'labels': labels, 'nearest_observed_surfels_m': distances.astype(float).tolist()}
    path.write_text(json.dumps(sidecar, indent=2), encoding='utf-8')
    return sidecar

def color_for_labels(labels: list[str]) -> np.ndarray:
    colors = {'observed_depth_surface': [30, 180, 255, 255], 'trellis_inferred_hidden_surface': [255, 150, 30, 255], 'free_space_rejected': [255, 30, 30, 150], 'unsupported_uncertain': [150, 150, 150, 160]}
    return np.asarray([colors.get(l, [255, 255, 255, 255]) for l in labels], dtype=np.uint8)

def completion_support_resolve_anchor_centroid_world(evidence: dict[str, Any]) -> np.ndarray:
    binding = evidence.get('selected_anchor_atomic_binding')
    if not isinstance(binding, dict) or binding.get('validated') is not True:
        raise RuntimeError('evidence lacks a validated atomic selected-anchor binding')
    selected = evidence.get('selected') if isinstance(evidence.get('selected'), dict) else {}
    vg = selected.get('visible_geometry_candidate') if isinstance(selected.get('visible_geometry_candidate'), dict) else {}
    row = evidence.get('depth_fused_object_row') if isinstance(evidence.get('depth_fused_object_row'), dict) else {}
    mesh_recon = row.get('mesh_reconstruction') if isinstance(row.get('mesh_reconstruction'), dict) else {}
    selected_frame_idx = int(evidence.get('selected_frame_idx', -1))
    if int(selected.get('frame_idx', -2)) != selected_frame_idx or int(vg.get('frame_idx', -2)) != selected_frame_idx or int(mesh_recon.get('anchor_frame_idx', -2)) != selected_frame_idx or (mesh_recon.get('atomic_anchor_binding') is not True):
        raise RuntimeError('selected mask/camera/surfels/canonical mesh are not bound to one frame')
    selected_centroid = np.asarray(vg.get('centroid_world_m') or [], dtype=float).reshape(-1)
    mesh_centroid = np.asarray(mesh_recon.get('anchor_centroid_world_m') or [], dtype=float).reshape(-1)
    if selected_centroid.shape != (3,) or mesh_centroid.shape != (3,) or (not np.isfinite(selected_centroid).all()) or (not np.isfinite(mesh_centroid).all()):
        raise RuntimeError('atomic selected-anchor evidence lacks finite centroids')
    error_m = float(np.linalg.norm(selected_centroid - mesh_centroid))
    if error_m > 1e-06:
        raise RuntimeError(f'selected visible/canonical centroids disagree by {error_m} m')
    return selected_centroid

def world_points_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (points_world - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]

def estimate_projection_source_size(intr: np.ndarray, image_w: int, image_h: int, selected: dict[str, Any]) -> tuple[int, int, str]:
    vg = selected.get('visible_geometry_candidate') if isinstance(selected.get('visible_geometry_candidate'), dict) else {}
    source_w = int(vg.get('source_width') or selected.get('source_width') or 0)
    source_h = int(vg.get('source_height') or selected.get('source_height') or 0)
    if source_w > 0 and source_h > 0:
        return (source_w, source_h, 'visible_geometry_candidate_source_size')
    (fx, fy, cx, cy) = [float(x) for x in intr]
    source_w = max(int(image_w), int(round(2.0 * cx)))
    source_h = max(int(image_h), int(round(2.0 * cy)))
    return (source_w, source_h, 'estimated_from_principal_point_legacy_evidence')

def trellis_planar_slab_keep_mask(observed_points: np.ndarray, mesh_canonical: trimesh.Trimesh, *, enabled: bool, observed_band_m: float, eigenvalue_ratio_max: float, min_band_m: float, max_band_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    from observations.object.registration import pca_basis
    import trimesh
    face_count = int(len(mesh_canonical.faces))
    if not enabled:
        return (np.ones(face_count, dtype=bool), {'state': 'disabled', 'kept_faces': face_count, 'rejected_faces': 0})
    (center, basis, vals) = pca_basis(observed_points)
    ratio = float(vals[-1] / max(vals[0], 1e-12))
    if not np.isfinite(ratio) or ratio > float(eigenvalue_ratio_max):
        return (np.ones(face_count, dtype=bool), {'state': 'not_applied_observed_surface_not_planar_enough', 'observed_pca_eigenvalues': vals.astype(float).tolist(), 'planarity_ratio_smallest_over_largest': ratio, 'ratio_threshold': float(eigenvalue_ratio_max), 'kept_faces': face_count, 'rejected_faces': 0})
    normal = basis[:, -1]
    obs_dist = np.abs((observed_points - center[None, :]) @ normal)
    data_band = float(np.percentile(obs_dist, 95.0)) if len(obs_dist) else 0.0
    band = max(float(min_band_m), data_band + 2.0 * float(observed_band_m))
    band = min(float(max_band_m), band)
    centers = np.asarray(mesh_canonical.triangles_center, dtype=float)
    dist = np.abs((centers - center[None, :]) @ normal)
    keep = np.isfinite(dist) & (dist <= band)
    return (keep, {'state': 'planar_support_slab_filter_applied', 'observed_pca_eigenvalues': vals.astype(float).tolist(), 'planarity_ratio_smallest_over_largest': ratio, 'ratio_threshold': float(eigenvalue_ratio_max), 'observed_normal_abs_distance_p95_m': data_band, 'observed_band_m': float(observed_band_m), 'slab_half_width_m': float(band), 'min_band_m': float(min_band_m), 'max_band_m': float(max_band_m), 'input_trellis_faces': face_count, 'kept_by_planar_slab_faces': int(np.count_nonzero(keep)), 'unsupported_outside_planar_slab_faces': int(np.count_nonzero(~keep)), 'claim_scope': 'When the observed object surface is planar, hidden prior faces far off that observed support plane are not accepted as physical object body.'})

def trellis_silhouette_keep_mask(evidence: dict[str, Any], mesh_canonical: trimesh.Trimesh, *, enabled: bool, dilate_px: int) -> tuple[np.ndarray, dict[str, Any]]:
    import trimesh
    face_count = int(len(mesh_canonical.faces))
    if not enabled:
        return (np.ones(face_count, dtype=bool), {'state': 'disabled', 'kept_faces': face_count, 'rejected_faces': 0})
    selected = evidence.get('selected') if isinstance(evidence.get('selected'), dict) else {}
    mask_path = selected.get('mask_path')
    if not mask_path and isinstance(selected.get('trellis_conditioning_crop'), dict):
        mask_path = selected['trellis_conditioning_crop'].get('mask')
    if not mask_path:
        raise RuntimeError('evidence selected row lacks mask_path for silhouette free-space filtering')
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f'failed to read silhouette mask for free-space filtering: {mask_path}')
    mask_bool = mask > 0
    if int(dilate_px) > 0:
        k = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_bool = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0
    (h, w) = mask_bool.shape[:2]
    camera = selected.get('camera') if isinstance(selected.get('camera'), dict) else {}
    T = np.asarray(camera.get('T_world_camera_metric') or camera.get('T_world_camera') or [], dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise RuntimeError('evidence selected camera lacks a valid T_world_camera for silhouette free-space filtering')
    intr = np.asarray(camera.get('intrinsics_fx_fy_cx_cy') or [], dtype=float).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all():
        vg = selected.get('visible_geometry_candidate') if isinstance(selected.get('visible_geometry_candidate'), dict) else {}
        intr = np.asarray(vg.get('intrinsics_fx_fy_cx_cy') or [], dtype=float).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all() or intr[0] <= 0.0 or (intr[1] <= 0.0):
        raise RuntimeError('evidence selected row lacks valid intrinsics for silhouette free-space filtering')
    (source_w, source_h, source_note) = estimate_projection_source_size(intr, w, h, selected)
    sx = float(w) / float(source_w)
    sy = float(h) / float(source_h)
    (fx, fy, cx, cy) = [float(intr[0] * sx), float(intr[1] * sy), float(intr[2] * sx), float(intr[3] * sy)]
    centroid_world = completion_support_resolve_anchor_centroid_world(evidence)
    centers_canonical = np.asarray(mesh_canonical.triangles_center, dtype=float)
    centers_world = centers_canonical + centroid_world[None, :]
    centers_camera = world_points_to_camera(centers_world, T)
    z = centers_camera[:, 2]
    keep = np.zeros(face_count, dtype=bool)
    valid_z = np.isfinite(z) & (z > 0.01)
    uv = np.full((face_count, 2), np.nan, dtype=float)
    uv[valid_z, 0] = fx * centers_camera[valid_z, 0] / z[valid_z] + cx
    uv[valid_z, 1] = fy * centers_camera[valid_z, 1] / z[valid_z] + cy
    xi = np.rint(uv[:, 0]).astype(np.int64, copy=False)
    yi = np.rint(uv[:, 1]).astype(np.int64, copy=False)
    inside_image = valid_z & np.isfinite(uv).all(axis=1) & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    keep[inside_image] = mask_bool[yi[inside_image], xi[inside_image]]
    rejected = ~keep
    return (keep, {'state': 'silhouette_free_space_filter_applied', 'mask_path': str(mask_path), 'mask_size': [int(w), int(h)], 'source_size': [int(source_w), int(source_h)], 'source_size_note': source_note, 'scaled_intrinsics_fx_fy_cx_cy': [float(fx), float(fy), float(cx), float(cy)], 'dilate_px': int(max(0, dilate_px)), 'input_trellis_faces': int(face_count), 'projected_inside_image_faces': int(np.count_nonzero(inside_image)), 'kept_by_silhouette_faces': int(np.count_nonzero(keep)), 'free_space_rejected_faces': int(np.count_nonzero(rejected)), 'claim_scope': 'TRELLIS hidden faces whose evidence-frame projection falls outside the object-owned silhouette are not accepted as object body.'})

def build_completion_face_support(argv=None) -> None:
    from observations.object.registration import align_trellis_to_observed
    from observations.object.registration import identity_metric_canonical_alignment
    from observations.object.registration import transform_mesh
    import trimesh
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-report', type=Path, required=True)
    parser.add_argument('--trellis-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-mesh-already-metric-canonical', action='store_true', help='Require a verified SAM3D native metric-canonical input and preserve it with identity alignment; no RMS/PCA/ICP.')
    parser.add_argument('--observed-band-scale', type=float, default=math.sqrt(3.0), help='multiplier on depth-fusion voxel size; sqrt(3) covers one voxel diagonal')
    parser.add_argument('--silhouette-free-space-filter', action=argparse.BooleanOptionalAction, default=True, help='Reject TRELLIS hidden faces whose evidence-frame projection falls outside the object-owned mask silhouette.')
    parser.add_argument('--silhouette-dilate-px', type=int, default=16, help='Dilation radius in evidence-mask pixels before silhouette/free-space rejection.')
    parser.add_argument('--planar-slab-support-filter', action=argparse.BooleanOptionalAction, default=True, help='When observed surfels are planar, reject hidden-prior faces far off that observed support slab.')
    parser.add_argument('--planar-slab-eigenvalue-ratio-max', type=float, default=0.04, help='Apply planar slab filter only when smallest/largest observed PCA eigenvalue is at most this ratio.')
    parser.add_argument('--planar-slab-min-band-m', type=float, default=0.018, help='Minimum half-width for planar support slab.')
    parser.add_argument('--planar-slab-max-band-m', type=float, default=0.055, help='Maximum half-width for planar support slab.')
    parser.add_argument('--promote-single-view-hidden-prior-to-collision', action='store_true', help='Historical/diagnostic override. By default single-view TRELLIS hidden faces remain a pose/render hypothesis and are excluded from collision/sign geometry until independent multi-view support exists.')
    args = parser.parse_args(argv)
    if args.input_mesh_already_metric_canonical and args.promote_single_view_hidden_prior_to_collision:
        raise RuntimeError('verified generated render priors cannot be promoted to collision geometry')
    # Bind the exact P11 bytes consumed by this producer, not a later caller's
    # reconstruction of provenance when normalize_completion accepts the output.
    evidence_bytes = args.evidence_report.read_bytes()
    evidence = json.loads(evidence_bytes)
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    binding = evidence.get('selected_anchor_atomic_binding')
    if not isinstance(binding, dict) or binding.get('validated') is not True:
        raise RuntimeError('P13 requires a validated atomic P11 selected-anchor binding')
    trellis_report = read_json(args.trellis_report)
    partial_paths = evidence.get('partial_metric_geometry_paths', {})
    fused_points_path = Path(partial_paths.get('fused_point_cloud_path') or '')
    poisson_path = Path(partial_paths.get('poisson_mesh_path') or '')
    trellis_mesh_path = Path(trellis_report.get('mesh') or '')
    if not fused_points_path.is_file():
        raise RuntimeError(f'missing fused point cloud: {fused_points_path}')
    if not poisson_path.is_file():
        raise RuntimeError(f'missing Poisson visible mesh: {poisson_path}')
    if not trellis_mesh_path.is_file():
        raise RuntimeError(f'missing TRELLIS mesh: {trellis_mesh_path}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed_points = load_vertices(fused_points_path)
    trellis = load_mesh(trellis_mesh_path)
    observed_mesh = load_mesh(poisson_path)
    mesh_recon = evidence.get('depth_fused_object_row', {}).get('mesh_reconstruction', {})
    voxel_size_m = float(mesh_recon.get('voxel_size_m') or 0.006)
    observed_band_m = voxel_size_m * float(args.observed_band_scale)
    if args.input_mesh_already_metric_canonical:
        align = identity_metric_canonical_alignment(trellis, observed_points, trellis_report, trellis_mesh_path)
        trellis_canonical = trellis.copy()
    else:
        align = align_trellis_to_observed(trellis, observed_points)
        trellis_canonical = transform_mesh(trellis, align)
    (obs_d, obs_near) = face_center_labels(observed_mesh, observed_points, observed_band_m)
    observed_labels = ['observed_depth_surface' if x else 'unsupported_uncertain' for x in obs_near]
    (trellis_d, trellis_near) = face_center_labels(trellis_canonical, observed_points, observed_band_m)
    (silhouette_keep, silhouette_state) = trellis_silhouette_keep_mask(evidence, trellis_canonical, enabled=bool(args.silhouette_free_space_filter), dilate_px=int(args.silhouette_dilate_px))
    (planar_keep, planar_state) = trellis_planar_slab_keep_mask(observed_points, trellis_canonical, enabled=bool(args.planar_slab_support_filter), observed_band_m=float(observed_band_m), eigenvalue_ratio_max=float(args.planar_slab_eigenvalue_ratio_max), min_band_m=float(args.planar_slab_min_band_m), max_band_m=float(args.planar_slab_max_band_m))
    trellis_labels_all = []
    for (near_observed, keep_by_silhouette, keep_by_planar_slab) in zip(trellis_near, silhouette_keep, planar_keep):
        if near_observed:
            trellis_labels_all.append('observed_region_overwritten_candidate')
        elif not keep_by_silhouette:
            trellis_labels_all.append('free_space_rejected')
        elif not keep_by_planar_slab:
            trellis_labels_all.append('unsupported_uncertain')
        else:
            trellis_labels_all.append('trellis_inferred_hidden_surface')
    observed_mesh.visual.face_colors = color_for_labels(observed_labels)
    trellis_canonical.visual.face_colors = color_for_labels(trellis_labels_all)
    accepted_observed_faces = np.where(obs_near)[0]
    if len(accepted_observed_faces) == 0:
        raise RuntimeError('observed Poisson mesh has no faces close to fused visible surfels; refusing to construct accepted object body from unsupported fill')
    accepted_observed = observed_mesh.submesh([accepted_observed_faces], append=True, repair=False)
    accepted_observed_labels = ['observed_depth_surface'] * len(accepted_observed.faces)
    kept_trellis_faces = np.where(~trellis_near & silhouette_keep & planar_keep)[0]
    kept_trellis = trellis_canonical.submesh([kept_trellis_faces], append=True, repair=False)
    kept_trellis_labels = ['trellis_inferred_hidden_surface'] * len(kept_trellis.faces)
    completed = trimesh.util.concatenate([accepted_observed, kept_trellis])
    completed_labels = accepted_observed_labels + kept_trellis_labels
    completed.visual.face_colors = color_for_labels(completed_labels)
    if args.promote_single_view_hidden_prior_to_collision:
        collision_surface = completed.copy()
        collision_labels = list(completed_labels)
        collision_distances = np.concatenate([obs_d[accepted_observed_faces], trellis_d[kept_trellis_faces]])
        hidden_collision_faces = int(len(kept_trellis_faces))
        hidden_collision_state = 'historical_override_single_view_hidden_prior_promoted'
    else:
        collision_surface = accepted_observed.copy()
        collision_labels = list(accepted_observed_labels)
        collision_distances = obs_d[accepted_observed_faces]
        hidden_collision_faces = 0
        hidden_collision_state = 'single_view_hidden_prior_quarantined_pending_independent_multiview_support'
    collision_surface.visual.face_colors = color_for_labels(collision_labels)
    object_safe = completion_support_safe_id(str(evidence.get('object_id', 'object')).replace('object:', 'object_'))
    observed_mesh_path = args.output_dir / f'{object_safe}_observed_depth_surface_labeled.ply'
    trellis_all_path = args.output_dir / f'{object_safe}_trellis_aligned_all_candidate_labeled.ply'
    completed_path = args.output_dir / f'{object_safe}_compact_rigid_completed_mesh_labeled.ply'
    collision_surface_path = args.output_dir / f'{object_safe}_collision_eligible_surface_labeled.ply'
    observed_mesh.export(str(observed_mesh_path))
    trellis_canonical.export(str(trellis_all_path))
    completed.export(str(completed_path))
    collision_surface.export(str(collision_surface_path))
    observed_sidecar = export_label_sidecar(args.output_dir / 'observed_depth_surface_face_labels.json', observed_labels, obs_d, str(observed_mesh_path), 0)
    trellis_sidecar = export_label_sidecar(args.output_dir / 'trellis_candidate_face_labels.json', trellis_labels_all, trellis_d, str(trellis_all_path), 0)
    completed_sidecar = export_label_sidecar(args.output_dir / 'completed_mesh_face_labels.json', completed_labels, np.concatenate([obs_d[accepted_observed_faces], trellis_d[kept_trellis_faces]]), str(completed_path), 0)
    collision_sidecar = export_label_sidecar(args.output_dir / 'collision_eligible_surface_face_labels.json', collision_labels, collision_distances, str(collision_surface_path), 0)
    report = {'method': 'build_v18_compact_rigid_trellis_completion', 'status': 'ok_historical_override_single_view_hidden_prior_promoted_for_diagnostic' if args.promote_single_view_hidden_prior_to_collision else 'ok_with_single_view_hidden_completion_quarantined_from_physical_geometry', 'annotation_ready': False, 'case': evidence.get('case'), 'object_id': evidence.get('object_id'), 'claim_scope': 'A verified SAM3D native-pose bridge supplied a sensor-metric canonical generated render prior. This stage preserves that native orientation/scale with identity alignment and only applies shared observed-region/free-space labeling. Generated hidden faces remain excluded from collision/sign geometry.' if args.input_mesh_already_metric_canonical else 'Historical diagnostic override explicitly promoted the single-view TRELLIS hidden prior into collision geometry. This reproduces legacy behavior only; it does not create independent hidden-surface evidence or annotation readiness.' if args.promote_single_view_hidden_prior_to_collision else 'TRELLIS is metric-aligned as an RGB hidden-surface pose/render hypothesis. Observed depth-fused surfels remain the source of truth for physical surface regions. Single-view silhouette consistency is not independent hidden-geometry support, so generated hidden faces are excluded from collision/sign geometry by default.', 'inputs': {'evidence_report': str(args.evidence_report), 'trellis_report': str(args.trellis_report), 'trellis_mesh': str(trellis_mesh_path), 'input_mesh_already_metric_canonical': bool(args.input_mesh_already_metric_canonical), 'observed_fused_points': str(fused_points_path), 'observed_poisson_mesh': str(poisson_path)}, 'metric_alignment': align, 'observed_band_m': float(observed_band_m), 'observed_band_derivation': {'depth_fusion_voxel_size_m': float(voxel_size_m), 'scale': float(args.observed_band_scale), 'reason': 'one voxel diagonal around fused surfels distinguishes raw observed support from completion/extrapolation faces'}, 'outputs': {'observed_depth_surface_labeled_mesh': str(observed_mesh_path), 'trellis_aligned_all_candidate_labeled_mesh': str(trellis_all_path), 'completed_mesh_labeled': str(completed_path), 'pose_hypothesis_mesh_labeled': str(completed_path), 'collision_eligible_mesh_labeled': str(collision_surface_path), 'observed_face_labels': str(args.output_dir / 'observed_depth_surface_face_labels.json'), 'trellis_candidate_face_labels': str(args.output_dir / 'trellis_candidate_face_labels.json'), 'completed_face_labels': str(args.output_dir / 'completed_mesh_face_labels.json'), 'collision_eligible_face_labels': str(args.output_dir / 'collision_eligible_surface_face_labels.json')}, 'face_label_counts': {'observed_mesh': observed_sidecar['label_counts'], 'trellis_all_candidate': trellis_sidecar['label_counts'], 'completed_mesh': completed_sidecar['label_counts'], 'collision_eligible_mesh': collision_sidecar['label_counts'], 'free_space_rejected': int(trellis_sidecar['label_counts'].get('free_space_rejected', 0))}, 'geometry_readiness': {'pose_hypothesis_available': True, 'pose_hypothesis_mesh': str(completed_path), 'completed_mesh_legacy_field_semantics': 'pose_and_render_hypothesis_not_collision_or_signed_geometry', 'single_view_hidden_prior_independently_supported': False, 'hidden_prior_collision_promotion_override': bool(args.promote_single_view_hidden_prior_to_collision), 'hidden_prior_collision_state': hidden_collision_state, 'local_observed_collision_surface_available': True, 'collision_eligible_mesh': str(collision_surface_path), 'collision_eligible_hidden_face_count': hidden_collision_faces, 'collision_surface_watertight': bool(collision_surface.is_watertight), 'signed_geometry_ready': False, 'signed_geometry_readiness_reason': 'historical collision-promotion override does not create independent multi-view sign support' if args.promote_single_view_hidden_prior_to_collision else 'single-view hidden completion is quarantined and the observed partial surface is not a certified signed body', 'annotation_ready': False, 'required_next_evidence': 'independent trusted multi-view silhouette/depth support in solved object poses before generated hidden faces can enter collision/sign geometry'}, 'accepted_body_semantics': {'observed_depth_surface_faces_accepted': int(len(accepted_observed_faces)), 'observed_unsupported_uncertain_faces_excluded': int(len(observed_labels) - len(accepted_observed_faces)), 'trellis_hidden_surface_faces_in_pose_hypothesis': int(len(kept_trellis_faces)), 'trellis_hidden_surface_faces_collision_eligible': hidden_collision_faces, 'completed_mesh_is_pose_hypothesis_not_physical_body': True, 'unsupported_uncertain_is_not_object_body': True, 'claim_scope': 'completed_mesh_labeled is retained for backward-compatible P14 pose proposal and uncertainty rendering. Physical proximity/collision consumers must use collision_eligible_mesh_labeled and honor geometry_readiness; generated hidden faces require independent multi-view support.'}, 'free_space_rejection_state': 'silhouette_free_space_filter_applied' if bool(args.silhouette_free_space_filter) else 'silhouette_free_space_filter_disabled', 'silhouette_free_space_filter': silhouette_state, 'planar_slab_support_filter': planar_state, 'mesh_counts': {'observed_mesh_vertices': int(len(observed_mesh.vertices)), 'observed_mesh_faces': int(len(observed_mesh.faces)), 'trellis_vertices': int(len(trellis.vertices)), 'trellis_faces': int(len(trellis.faces)), 'completed_vertices': int(len(completed.vertices)), 'completed_faces': int(len(completed.faces)), 'collision_eligible_vertices': int(len(collision_surface.vertices)), 'collision_eligible_faces': int(len(collision_surface.faces))}}
    report['input_hashes'] = {'evidence_report': evidence_sha256}
    (args.output_dir / 'v18_compact_rigid_trellis_completion_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
