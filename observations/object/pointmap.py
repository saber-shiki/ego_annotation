"""Current-run dense metric pointmaps for the official SAM3D pointmap-only API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gates.camera.calibration import load_contract, plane_intrinsics, resize_affine, transform_intrinsics
from utils.state import sha

PROFILE = 'shared_metric_pointmap'
SCHEMA = 'shared_metric_pointmap_v1'
FRAME = 'pytorch3d_camera_x_left_y_up_z_forward'


def array_sha(value):
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(str(array.dtype).encode() + json.dumps(list(array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'pointmap source missing: {path}')
    return {'path': str(path), 'sha256': sha(path)}


def _in_run(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'pointmap source/output outside current run: {path}')
    return path


def _scalar(archive, key):
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f'pointmap depth field {key} must be scalar')
    return value.item()


def dense_pointmap(depth, depth_K, target_size_wh):
    """Resample camera-z (not XYZ), then backproject the explicit half-pixel plane.

    This is arm B's INTER_LINEAR observation mapping. Invalid depth is not filled
    or replaced with internal MoGe; this profile requires a dense input raster.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError('dense pointmap requires finite positive camera-z meters; no missing-depth imputation')
    affine = resize_affine(depth.shape[::-1], target_size_wh)
    K = transform_intrinsics(depth_K, affine)
    z = cv2.resize(depth, tuple(target_size_wh), interpolation=cv2.INTER_LINEAR)
    y, x = np.indices(z.shape, dtype=np.float64)
    points = np.stack((-(x-K[2])*z/K[0], -(y-K[3])*z/K[1], z), axis=-1).astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError('pointmap overflow')
    return points, K, affine


def _source_geometry(*, run_root, frame_idx, image_path, mask_path, depth_npz, camera_contract, raw_frame_manifest, object_id, mask_source):
    root = Path(run_root).resolve()
    paths = {key: _in_run(value, root) for key, value in dict(image=image_path, mask=mask_path,
        depth=depth_npz, camera=camera_contract, raw_manifest=raw_frame_manifest, mask_source=mask_source).items()}
    refs = {key: file_record(path) for key, path in paths.items()}
    manifest = json.loads(paths['raw_manifest'].read_text())
    rows = manifest['frames']
    ids = [row['frame_idx'] for row in rows]
    if len(set(ids)) != len(ids) or frame_idx not in ids:
        raise ValueError('pointmap raw manifest frame missing/duplicate')
    row = rows[ids.index(frame_idx)]
    raw = _in_run(row.get('raw_frame_path') or row['rgb'], root)
    raw_ref = file_record(raw)
    if refs['image']['sha256'] != raw_ref['sha256'] or row.get('raw_frame_sha256') != raw_ref['sha256']:
        raise ValueError('pointmap RGB source/frame/hash mismatch')
    mask_payload = json.loads(paths['mask_source'].read_text())
    mask_rows = [item for item in mask_payload['frames'] if item['frame_idx'] == frame_idx]
    if len(mask_rows) != 1:
        raise ValueError('pointmap mask source frame missing/duplicate')
    mask_row = mask_rows[0]
    if 'objects' in mask_row:
        matches = [obj for obj in mask_row['objects'] if obj.get('object_id') == object_id]
        if len(matches) != 1:
            raise ValueError('pointmap mask source object missing/duplicate')
        source_mask = matches[0].get('mask_path')
    else:
        if mask_row.get('object_id') != object_id:
            raise ValueError('pointmap appearance mask source object mismatch')
        source_mask = mask_row.get('appearance_mask_path')
    if not source_mask or file_record(_in_run(source_mask, root))['sha256'] != refs['mask']['sha256']:
        raise ValueError('pointmap mask source/frame/hash mismatch')
    refs['selected_source_mask'] = file_record(_in_run(source_mask, root))
    with Image.open(paths['image']) as image:
        size = image.size
    with Image.open(paths['mask']) as mask:
        if mask.size != size or not np.asarray(mask.convert('L')).any():
            raise ValueError('pointmap RGB/mask shape mismatch or empty mask')
    contract, normalized = load_contract(paths['camera'], expected_frame_ids=ids)
    source_size = (int(row['source_width']), int(row['source_height']))
    source_K, _ = plane_intrinsics(contract, normalized, plane_name='source_rgb', actual_size_wh=source_size)
    if size != (int(row['manifest_width']), int(row['manifest_height'])):
        raise ValueError('pointmap image shape differs from raw manifest plane')
    rgb_K = transform_intrinsics(source_K, resize_affine(source_size, size))
    with np.load(paths['depth'], allow_pickle=False) as archive:
        depth_ids = np.asarray(archive['frame_idx'])
        if depth_ids.tolist() != ids:
            raise ValueError('pointmap depth frame timeline mismatch')
        if str(_scalar(archive, 'camera_contract_sha256')) != refs['camera']['sha256'] or Path(str(_scalar(archive, 'camera_contract_path'))).resolve() != paths['camera']:
            raise ValueError('pointmap depth camera source/hash mismatch')
        if bool(_scalar(archive, 'metadata_only_ray_relabel_override')):
            raise ValueError('pointmap refuses metadata-only ray relabel')
        quantity = str(_scalar(archive, 'depth_output_quantity'))
        if quantity not in ('camera_z_m_from_metric_radius_on_exact_contract_rays', 'camera_z_m_from_model_inferred_rays'):
            raise ValueError('pointmap requires declared camera-z meters')
        index = ids.index(frame_idx)
        depths = archive['depth']
        ks = np.asarray(archive['intrinsics_fx_fy_cx_cy'])
        if depths.ndim != 3 or depths.shape[0] != len(ids) or ks.shape != (len(ids), 4):
            raise ValueError('pointmap depth/K shape mismatch')
        depth = depths[index]
        if np.asarray(archive['source_size']).tolist() != list(depth.shape[::-1]):
            raise ValueError('pointmap depth raster/source_size mismatch')
        plane = str(_scalar(archive, 'camera_contract_plane'))
        K, transform = plane_intrinsics(contract, normalized, plane_name=plane, actual_size_wh=depth.shape[::-1])
        if not np.allclose(ks[index], K, atol=1e-6, rtol=0) or not np.allclose(archive['source_depth_active_intrinsics_fx_fy_cx_cy'][index], K, atol=1e-6, rtol=0):
            raise ValueError('pointmap depth rays differ from camera contract')
        if not np.allclose(archive['A_depth_from_calibration'], transform['A_actual_plane_from_calibration'], atol=1e-9, rtol=0):
            raise ValueError('pointmap depth camera affine mismatch')
        points, point_K, affine = dense_pointmap(depth, K, size)
        if not np.allclose(point_K, rgb_K, atol=1e-6, rtol=0):
            raise ValueError('pointmap depth-to-RGB mapping is not the declared half-pixel resize')
        mapping = {'depth_size_wh': list(depth.shape[::-1]), 'image_size_wh': list(size),
            'A_image_from_depth': affine.tolist(), 'K_image_fx_fy_cx_cy': point_K.tolist(),
            'depth_plane': plane, 'depth_frame_position': index, 'depth_frame_array_sha256': array_sha(depth),
            'depth_quantity': quantity, 'interpolation': 'camera_z_cv2_INTER_LINEAR',
            'pixel_center_convention': 'integer_pixel_centers_opencv',
            'axis_conversion': 'OpenCV_to_PyTorch3D_diag(-1,-1,1)_exactly_once'}
    return points, refs, mapping


def build_pointmap(*, run_root, frame_idx, image_path, mask_path, depth_npz, camera_contract, raw_frame_manifest, object_id, mask_source, output_dir):
    """Build one immutable receipt; call separately for metric and appearance anchors.

    Returns the receipt Path consumed by --pointmap-case NAME|FRAME|RECEIPT.
    mask_source is P09 annotations or an appearance manifest with per-frame masks.
    All inputs and the fresh output directory must be inside run_root.
    """
    if isinstance(frame_idx, bool) or not isinstance(frame_idx, (int, np.integer)):
        raise ValueError('pointmap frame_idx must be an integer')
    root = Path(run_root).resolve()
    output = _in_run(output_dir, root)
    if output.exists():
        raise FileExistsError(output)
    points, refs, mapping = _source_geometry(run_root=root, frame_idx=int(frame_idx), image_path=image_path,
        mask_path=mask_path, depth_npz=depth_npz, camera_contract=camera_contract, raw_frame_manifest=raw_frame_manifest,
        object_id=object_id, mask_source=mask_source)
    output.mkdir(parents=True)
    np.save(output/'pointmap.npy', points, allow_pickle=False)
    receipt = {'schema': SCHEMA, 'profile': PROFILE, 'run_root': str(root), 'frame_idx': int(frame_idx), 'object_id': object_id,
        'sources': refs, 'mapping': mapping, 'pointmap': file_record(output/'pointmap.npy'),
        'pointmap_array_sha256': array_sha(points), 'shape_hwc': list(points.shape), 'units': 'meters',
        'coordinate_frame': FRAME, 'missing_depth_policy': 'reject_no_imputation',
        'depth_correlation': 'same_shared_depth_as_downstream_surface_not_independent_evidence',
        'annotation_ready': False, 'physical_contact_authority': False, 'collision_authority': False,
        'signed_authority': False}
    path = output/'pointmap.json'
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return path


def load_pointmap(receipt_path, *, frame_idx, image_path, mask_path):
    """Verify source bytes and reconstruct the selected frame to reject substitutions."""
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != SCHEMA or receipt.get('profile') != PROFILE or receipt.get('frame_idx') != frame_idx:
        raise ValueError('pointmap receipt schema/profile/frame mismatch')
    if receipt.get('units') != 'meters' or receipt.get('coordinate_frame') != FRAME:
        raise ValueError('pointmap units/frame convention mismatch')
    root = Path(receipt['run_root']).resolve()
    _in_run(receipt_path, root)
    sources = receipt['sources']
    for label, ref in sources.items():
        if file_record(_in_run(ref['path'], root)) != ref:
            raise ValueError(f'pointmap {label} source/hash mismatch')
    for key, path in (('image', image_path), ('mask', mask_path)):
        if file_record(path) != sources[key]:
            raise ValueError(f'pointmap {key} input source/hash mismatch')
    points, refs, mapping = _source_geometry(run_root=root, frame_idx=frame_idx, image_path=image_path,
        mask_path=mask_path, depth_npz=sources['depth']['path'], camera_contract=sources['camera']['path'],
        raw_frame_manifest=sources['raw_manifest']['path'], object_id=receipt['object_id'],
        mask_source=sources['mask_source']['path'])
    ref = receipt['pointmap']
    if file_record(_in_run(ref['path'], root)) != ref:
        raise ValueError('pointmap payload hash mismatch')
    actual = np.load(ref['path'], allow_pickle=False)
    if actual.dtype != np.float32 or list(actual.shape) != receipt['shape_hwc'] or not np.array_equal(actual, points):
        raise ValueError('pointmap shape/geometry differs from selected source frame')
    if array_sha(actual) != receipt['pointmap_array_sha256'] or mapping != receipt['mapping'] or refs != sources:
        raise ValueError('pointmap source mapping/array hash mismatch')
    return actual, receipt
