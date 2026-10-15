"""Typed immutable RGB source evidence. Legacy canonical edges are not upgraded."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np

from data.interaction import freeze_evidence
from observations.object.rgb_evidence import SCHEMA, MOTION_CONVENTION
from utils.geometry import validate_transforms, valid_rotation
from utils.state import sha


def _array(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class RGBSourceEdge:
    object_id: str
    source_frame_idx: int
    target_frame_idx: int
    source_time_s: float
    target_time_s: float
    source_world: np.ndarray
    source_uv: np.ndarray
    target_uv: np.ndarray
    source_camera: np.ndarray
    target_camera: np.ndarray
    source_intrinsics: np.ndarray
    target_intrinsics: np.ndarray
    source_size_wh: np.ndarray
    target_size_wh: np.ndarray
    rotation_source_to_target_world: np.ndarray
    translation_source_to_target_world_m: np.ndarray
    weights: np.ndarray
    matching_scores: np.ndarray
    inlier_indices: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RGBSourceEvidence:
    schema: str
    object_id: str
    edges: tuple[RGBSourceEdge, ...]
    report: Mapping[str, Any]


def load_rgb_source_evidence(npz_path: Path, report_path: Path, *, frame_ids, times,
                             frames, object_ids, expected_inputs) -> RGBSourceEvidence:
    """Verify the pair and exact inputs even if the outer bundle skipped verification."""
    report = json.loads(Path(report_path).read_text())
    if report.get('schema') != SCHEMA:
        raise ValueError('unsupported RGB source schema; legacy canonical evidence needs its reference consumer')
    if report.get('npz_sha256') != sha(npz_path):
        raise ValueError('RGB NPZ hash mismatch')
    with np.load(npz_path, allow_pickle=False) as archive:
        data = dict(archive)
    embedded = json.loads(str(data.pop('report_metadata').item()))
    if embedded != {k: v for k, v in report.items() if k != 'npz_sha256'}:
        raise ValueError('RGB NPZ/JSON pairing mismatch')
    if report.get('motion_convention') != MOTION_CONVENTION or report.get('point_frame') != 'fixed_source_world_m' or report.get('pixel_frame') != 'original_owned_mask_half_pixel_resize':
        raise ValueError('RGB coordinate convention mismatch')
    if (report.get('prediction_side_only') is not True or report.get('generated_geometry_consumed') is not False
            or report.get('legacy_optimizer_required') is not False or report.get('contains_optimized_state') is not False):
        raise ValueError('RGB evidence is not observation-only')
    parameters = report.get('parameters')
    if not isinstance(parameters, Mapping):
        raise ValueError('RGB missing measurement parameters')
    ransac_threshold = parameters.get('reprojection_threshold_px')
    if (isinstance(ransac_threshold, bool) or not isinstance(ransac_threshold, (int, float))
            or not np.isfinite(float(ransac_threshold)) or float(ransac_threshold) <= 0):
        raise ValueError('RGB invalid RANSAC reprojection threshold')
    if report.get('object_id') not in object_ids:
        raise ValueError('RGB foreign object identity')
    ids = [int(i) for i in frame_ids]
    if report.get('frame_ids') != ids or not np.array_equal(report.get('time_s'), times):
        raise ValueError('RGB foreign frame/timestamp timeline')
    inputs = report.get('inputs', {})
    for key in ('annotations', 'depth_npz', 'raw_frame_manifest', 'p09_adapter_report', 'assets_manifest'):
        if key not in inputs:
            raise ValueError(f'RGB missing input identity: {key}')
    for key, receipt in inputs.items():
        path = Path(receipt['path']).resolve()
        if not path.is_file() or sha(path) != receipt.get('sha256'):
            raise ValueError(f'RGB input identity mismatch: {key}')
    for key, path in expected_inputs.items():
        record = inputs.get(key)
        if record is None or Path(record['path']).resolve() != Path(path).resolve() or record['sha256'] != sha(path):
            raise ValueError(f'RGB bundle input mismatch: {key}')
    rows = report.get('rows', [])
    seen = set()
    for row in rows + report.get('pair_planning_rejections', []):
        a, b = row['source_frame_idx'], row['target_frame_idx']
        if type(a) is not int or type(b) is not int or a not in ids or b not in ids or a == b:
            raise ValueError('RGB foreign frame in diagnostic edge')
    for row in rows:
        pair = (row['source_frame_idx'], row['target_frame_idx'])
        if pair in seen or row.get('status') not in ('accepted', 'rejected'):
            raise ValueError('RGB duplicate/invalid edge row')
        seen.add(pair)
        if row['status'] == 'rejected' and not row.get('reason'):
            raise ValueError('RGB rejected edge lacks reason')
    for row in report.get('source_rejections', []):
        if row['frame_idx'] not in ids or not row.get('reason'):
            raise ValueError('RGB foreign frame/invalid source rejection')
    accepted = [r for r in rows if r['status'] == 'accepted']
    n = len(accepted)
    offsets = data['offsets']
    if (offsets.dtype.kind not in 'iu' or offsets.shape != (n + 1,) or offsets[0] != 0
            or np.any(np.diff(offsets) < 4)):
        raise ValueError('RGB invalid support offsets')
    count = int(offsets[-1])
    point_shapes = {'source_world': (count, 3), 'source_uv': (count, 2), 'target_uv': (count, 2),
                    'weights': (count,), 'matching_scores': (count,), 'inlier_indices': (count,)}
    edge_shapes = {'rotation_source_to_target_world': (n, 3, 3), 'translation_source_to_target_world_m': (n, 3)}
    for end in ('source', 'target'):
        edge_shapes.update({end + '_frame_idx': (n,), end + '_time_s': (n,),
            end + '_camera': (n, 4, 4), end + '_intrinsics': (n, 4), end + '_size_wh': (n, 2)})
    for key, shape in {**point_shapes, **edge_shapes}.items():
        value = data[key]
        if value.shape != shape or value.dtype.kind not in 'fiub' or not np.isfinite(value).all():
            raise ValueError(f'RGB invalid numeric array: {key}')
        if (key.endswith('frame_idx') or key.endswith('size_wh') or key == 'inlier_indices') and value.dtype.kind not in 'iu':
            raise ValueError(f'RGB integer array required: {key}')
    if np.any(data['weights'] <= 0) or np.any(data['weights'] > 1) or np.any(data['matching_scores'] < 0) or np.any(data['matching_scores'] > 1):
        raise ValueError('RGB invalid point quality')
    pairs = [[int(a), int(b)] for a, b in zip(data['source_frame_idx'], data['target_frame_idx'])]
    if pairs != report.get('accepted_pairs') or pairs != [[r['source_frame_idx'], r['target_frame_idx']] for r in accepted]:
        raise ValueError('RGB accepted NPZ/JSON edge mismatch')
    positions = {idx: pos for pos, idx in enumerate(ids)}
    result = []
    for k, row in enumerate(accepted):
        lo, hi = map(int, offsets[k:k+2])
        if row.get('inliers') != hi-lo or row.get('reprojection_evidence_count') != hi-lo:
            raise ValueError('RGB support count mismatch')
        eigen = np.asarray(row.get('source_3d_covariance_eigenvalues'), float)
        quality = np.asarray([row.get('inlier_fraction'), row.get('reprojection_median_px'),
                              row.get('reprojection_p95_px'), row.get('source_3d_conditioning')], float)
        if (eigen.shape != (3,) or not np.isfinite(eigen).all() or np.any(eigen < 0)
                or not np.isfinite(quality).all() or np.any(quality < 0) or quality[0] > 1 or quality[3] > 1):
            raise ValueError('RGB invalid quality/conditioning')
        indices = data['inlier_indices'][lo:hi]
        if len(set(indices.tolist())) != len(indices) or np.any(indices < 0) or np.any(indices >= row['matches_object']):
            raise ValueError('RGB invalid inlier indices')
        if not valid_rotation(data['rotation_source_to_target_world'][k]):
            raise ValueError('RGB invalid world motion rotation')
        target_reprojection = None
        for end in ('source', 'target'):
            idx = row[end + '_frame_idx']
            frame = frames[positions[idx]]
            if data[end + '_time_s'][k] != times[positions[idx]] or row[end + '_time_s'] != times[positions[idx]]:
                raise ValueError('RGB endpoint timestamp mismatch')
            camera = data[end + '_camera'][k]
            validate_transforms(camera[None])
            expected_camera = frame['camera'].get('T_world_camera_metric', frame['camera'].get('T_world_camera'))
            if not np.allclose(camera, expected_camera, atol=1e-10, rtol=0):
                raise ValueError('RGB endpoint camera mismatch')
            size = data[end + '_size_wh'][k]
            plane = report['image_planes'].get(str(idx), {})
            if np.any(size <= 0) or not np.array_equal(size, plane.get('owned_mask_size_wh')):
                raise ValueError('RGB endpoint image plane mismatch')
            source_size = [frame['source_width'], frame['source_height']]
            if plane.get('source_size_wh') != source_size:
                raise ValueError('RGB source image plane mismatch')
            scale = size / np.asarray(source_size)
            K = np.asarray(frame['camera']['intrinsics_fx_fy_cx_cy'], float)
            expected_K = np.r_[K[:2] * scale, (K[2:] + .5) * scale - .5]
            if not np.allclose(data[end + '_intrinsics'][k], expected_K, atol=1e-10, rtol=0) or np.any(expected_K[:2] <= 0):
                raise ValueError('RGB endpoint K mismatch')
            for prefix in ('rgb', 'mask'):
                if f'{prefix}:{idx}' not in inputs:
                    raise ValueError('RGB missing endpoint pixel identity')
            objects = [o for o in frame.get('objects', []) if o.get('object_id') == report['object_id']]
            if len(objects) != 1 or Path(objects[0]['mask_path']).resolve() != Path(inputs[f'mask:{idx}']['path']).resolve():
                raise ValueError('RGB endpoint owned mask identity mismatch')
            if Path(plane['rgb_path']).resolve() != Path(inputs[f'rgb:{idx}']['path']).resolve():
                raise ValueError('RGB endpoint image identity mismatch')
            if end == 'source':
                from envs.qualification.object_observation import p09_geometry_validation
                obj = objects[0]
                if (obj.get('rigid_pose_observation_eligible') is not True
                        or obj.get('visible_geometry_candidate', {}).get('rigid_pose_observation_eligible') is not True
                        or p09_geometry_validation(obj, frame)[0] is None):
                    raise ValueError('RGB source lacks eligible observed P09 support')
            world = data['source_world'][lo:hi]
            if end == 'target':
                world = world @ data['rotation_source_to_target_world'][k].T + data['translation_source_to_target_world_m'][k]
            cam_points = (world - camera[:3, 3]) @ camera[:3, :3]
            if np.any(cam_points[:, 2] <= 0):
                raise ValueError('RGB support violates camera cheirality')
            if end == 'source':
                projected = cam_points[:, :2] / cam_points[:, 2:] * expected_K[:2] + expected_K[2:]
                if np.any(np.abs(projected - data['source_uv'][lo:hi]) > .5001 * scale):
                    raise ValueError('RGB fixed source world/source pixel mismatch')
            else:
                target_uv = data['target_uv'][lo:hi]
                if np.any(target_uv < 0) or np.any(target_uv[:, 0] >= size[0]) or np.any(target_uv[:, 1] >= size[1]):
                    raise ValueError('RGB target pixels lie outside the validated image plane')
                projected = cam_points[:, :2] / cam_points[:, 2:] * expected_K[:2] + expected_K[2:]
                if not np.isfinite(projected).all():
                    raise ValueError('RGB target reprojection is not finite')
                target_reprojection = np.linalg.norm(projected - target_uv, axis=1)
                threshold = float(ransac_threshold)
                if np.any(target_reprojection > threshold + max(1e-8, threshold * 1e-5)):
                    raise ValueError('RGB target reprojection violates the RANSAC inlier threshold')
        if target_reprojection is None:
            raise ValueError('RGB target reprojection was not validated')
        reported_median = row.get('reprojection_median_px')
        reported_p95 = row.get('reprojection_p95_px')
        if (isinstance(reported_median, bool) or isinstance(reported_p95, bool)
                or not isinstance(reported_median, (int, float)) or not isinstance(reported_p95, (int, float))
                or not np.isfinite([reported_median, reported_p95]).all()
                or not np.isclose(float(np.median(target_reprojection)), float(reported_median), atol=1e-4, rtol=1e-5)
                or not np.isclose(float(np.percentile(target_reprojection, 95)), float(reported_p95), atol=1e-4, rtol=1e-5)):
            raise ValueError('RGB target reprojection diagnostics mismatch')
        arrays = {key: _array(data[key][lo:hi]) for key in point_shapes}
        arrays.update({key: _array(data[key][k]) for key in edge_shapes if not key.endswith(('frame_idx', 'time_s'))})
        result.append(RGBSourceEdge(object_id=report['object_id'],
            source_frame_idx=row['source_frame_idx'], target_frame_idx=row['target_frame_idx'],
            source_time_s=row['source_time_s'], target_time_s=row['target_time_s'],
            diagnostics=freeze_evidence(row), **arrays))
    return RGBSourceEvidence(SCHEMA, report['object_id'], tuple(result), freeze_evidence(report))
