"""Current local RGB rotation evidence and independent cycle decisions."""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np
from scipy.spatial.transform import Rotation

def local_rotation_evidence(nodes, times, edges, anchor, reprojections=(), max_gap_s=2.0):
    positions = {n.frame_idx: i for (i, n) in enumerate(nodes)}
    ids = list(positions)
    sources = [i for (i, n) in enumerate(nodes) if n.metric_observation]
    local_pairs = {frozenset((a, b)) for (a, b) in zip(ids[:-1], ids[1:]) if times[positions[b]] - times[positions[a]] <= max_gap_s}
    for (a, b) in zip(sources[:-1], sources[1:]):
        if times[b] - times[a] <= max_gap_s:
            local_pairs.add(frozenset((ids[a], ids[b])))
    for (i, n) in enumerate(nodes):
        if n.metric_observation:
            continue
        before = [j for j in sources if j < i]
        after = [j for j in sources if j > i]
        for j in ([before[-1]] if before else []) + ([after[0]] if after else []):
            if abs(times[j] - times[i]) <= max_gap_s:
                local_pairs.add(frozenset((ids[i], ids[j])))

    def quality(e):
        d = e.diagnostics
        return float(np.clip(float(d.get('inlier_fraction', 1.0)) * math.exp(-float(d.get('reprojection_median_px', 0.0)) / 2.0), 0, 1))
    candidates = []
    for (k, e) in enumerate(edges):
        (a, b) = (positions[e.source_frame_idx], positions[e.target_frame_idx])
        if e.weight <= 0 or e.diagnostics.get('translation_only') or frozenset((ids[a], ids[b])) not in local_pairs:
            continue
        rotation = e.rotation_source_to_target_world if a < b else e.rotation_source_to_target_world.T
        candidates.append((abs(times[b] - times[a]), -quality(e), -e.weight, min(a, b), max(a, b), tuple(rotation.ravel()), k))
    parent = list(range(len(nodes)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    adjacency = {i: [] for i in ids}
    tree = set()
    for (*_, k) in sorted(candidates):
        e = edges[k]
        (a, b) = (e.source_frame_idx, e.target_frame_idx)
        (ra, rb) = (root(positions[a]), root(positions[b]))
        if ra == rb:
            continue
        parent[rb] = ra
        tree.add(k)
        adjacency[a].append((b, e.rotation_source_to_target_world, k))
        adjacency[b].append((a, e.rotation_source_to_target_world.T, k))
    (relative, paths, components) = ({}, {}, {})
    for start in [anchor] + [i for i in ids if i != anchor]:
        if start in relative:
            continue
        relative[start] = np.eye(3)
        paths[start] = []
        components[start] = start
        queue = [start]
        for a in queue:
            for (b, rotation, k) in sorted(adjacency[a], key=lambda x: positions[x[0]]):
                if b in relative:
                    continue
                relative[b] = rotation @ relative[a]
                paths[b] = paths[a] + [k]
                components[b] = start
                queue.append(b)
    anchor_rotation = nodes[positions[anchor]].initial_rotation
    seeded = []
    seed_rows = []
    for (n, stamp) in zip(nodes, times):
        supported = components[n.frame_idx] == anchor
        path = paths[n.frame_idx]
        strength = min((quality(edges[k]) for k in path), default=1.0) if supported else 0.0
        r = relative[n.frame_idx] @ anchor_rotation if supported else n.initial_rotation.copy()
        seeded.append(replace(n, initial_rotation=r.copy(), uncertainty=max(n.uncertainty, 1.0 - strength)))
        seed_rows.append({'frame_idx': n.frame_idx, 'time_s': float(stamp), 'rotation_world_from_completed_canonical_matrix': r.tolist(), 'anchor_connected': supported, 'local_path_edge_indices': path, 'support_quality': strength, 'uncertainty': max(n.uncertainty, 1.0 - strength), 'kind': 'current_local_RGB_seed' if supported and path else 'selected_anchor_gauge' if n.frame_idx == anchor else 'source_or_latent_disconnected'})
    (filtered, decisions) = ([], [])
    pair_scales = {}
    for (k, e) in enumerate(edges):
        (a, b) = (e.source_frame_idx, e.target_frame_idx)
        independent = sorted(set(paths[a]) ^ set(paths[b])) if components[a] == components[b] else []
        disagreement = None
        fraction = float(np.clip(float(e.diagnostics.get('source_3d_conditioning', 0.02)) / 0.02, 0, 1))
        allowed = 6.0 + 14.0 * fraction
        (scale, reason) = (1.0, 'local_tree_measurement_not_cycle_validated')
        strength = min((quality(edges[j]) for j in independent), default=0.0)
        if k not in tree and (not e.diagnostics.get('translation_only')) and (e.weight > 0):
            if independent and strength >= 0.2:
                predicted = relative[b] @ relative[a].T
                disagreement = float(np.degrees(Rotation.from_matrix(predicted @ e.rotation_source_to_target_world.T).magnitude()))
                scale = 0.0 if disagreement > allowed else 1.0
                reason = 'rejected_independent_local_cycle' if scale == 0 else 'consistent_independent_local_cycle'
            else:
                (scale, reason) = (0.25, 'unverified_weak_or_disconnected_local_path')
        filtered.append(replace(e, weight=e.weight * scale))
        pair_scales[a, b] = min(pair_scales.get((a, b), 1.0), scale)
        decisions.append({'edge_index': k, 'source': a, 'target': b, 'reason': reason, 'weight_scale': scale, 'independent_path_edge_indices': independent if k not in tree else [], 'independent_support_quality': strength if k not in tree else None, 'disagreement_deg': disagreement, 'allowed_deg': allowed, 'input_weight': e.weight, 'output_weight': e.weight * scale})
    projections = [replace(f, edge_weight=f.edge_weight * pair_scales.get((f.source_frame_idx, f.target_frame_idx), 1.0)) for f in reprojections]
    return (seeded, filtered, projections, {'policy': 'temporal_local_forest_then_independent_conditioned_cycles', 'max_local_gap_s': max_gap_s, 'minimum_independent_path_quality': 0.2, 'seed_rows': seed_rows, 'edges': decisions, 'local_disconnected_ids': [i for i in ids if components[i] != anchor], 'source_prior_mutated': False})

def observed_rotation_authority(nodes, chain):
    """Partial near-planar NN surfaces constrain translation, not RGB rotation.

    Covariance ratio is a heuristic observability estimate, not a claim that all
    planar rotations are unobservable. Only locally RGB-supported nodes use it.
    """
    support = {r['frame_idx']: r for r in chain['seed_rows']}
    scales = {}
    for n in nodes:
        scale = 1.0
        row = support[n.frame_idx]
        if row['anchor_connected'] and row['support_quality'] >= 0.2 and (n.observed_world is not None):
            points = np.asarray(n.observed_world)
            eig = np.maximum(np.linalg.eigvalsh(np.cov(points.T)), 0.0)
            scale = float(np.clip(eig[0] / max(eig[-1], 1e-12) / 0.02, 0, 1))
        scales[n.frame_idx] = scale
    return scales
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from utils.state import append_prediction_stage
from utils.state import assert_prediction_only

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.expanduser().resolve().open('rb') as f:
        for chunk in iter(lambda : f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise RuntimeError(f'expected JSON object: {path}')
    return value

def resize_intrinsics(K: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    sx = float(target_wh[0]) / float(source_wh[0])
    sy = float(target_wh[1]) / float(source_wh[1])
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = sx * (out[0, 2] + 0.5) - 0.5
    out[1, 2] = sy * (out[1, 2] + 0.5) - 0.5
    return out

def validate_depth_contract(depth_ids: np.ndarray, depths: np.ndarray, confidences: np.ndarray, source_size: np.ndarray, intrinsics: np.ndarray) -> tuple[int, int]:
    source_size_array = np.asarray(source_size, dtype=np.int64).reshape(-1)
    if source_size_array.shape != (2,) or np.any(source_size_array <= 0):
        raise RuntimeError('depth source_size must be two positive dimensions')
    source_wh = (int(source_size_array[0]), int(source_size_array[1]))
    if depths.ndim != 3 or depths.shape[2] != source_wh[0] or depths.shape[1] != source_wh[1]:
        raise RuntimeError(f'depth raster {depths.shape} disagrees with source_size {source_wh}')
    if confidences.shape != depths.shape:
        raise RuntimeError('depth/confidence raster shapes disagree')
    if intrinsics.shape != (len(depth_ids), 4):
        raise RuntimeError('per-frame depth intrinsics shape is invalid')
    if len(np.unique(depth_ids)) != len(depth_ids):
        raise RuntimeError('depth frame_idx values are not unique')
    return source_wh

def find_case_root(annotation_path: Path) -> Path:
    return annotation_path.expanduser().resolve().parents[4]

def resolve_raw_frame_manifest(annotation_path: Path, explicit: Path | None=None) -> Path:
    """Separate the shared raw-video input from a candidate's annotation root.

    Legacy runs keep their existing adjacent input lookup. Candidate callers
    explicitly pass the source P01 manifest, whose own frame/raster checks and
    recorded input hash still apply at consumption.
    """
    if explicit is not None:
        return explicit.expanduser().resolve()
    return find_case_root(annotation_path) / 'input/raw_frame_manifest/manifest.json'

def object_for_frame(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get('objects', []) if isinstance(frame.get('objects'), list) else []:
        if isinstance(obj, dict) and obj.get('object_id') == object_id:
            return obj
    return None

def lift_source_world(frame_idx: int, uv: np.ndarray, mask: np.ndarray, depth: np.ndarray, confidence: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray, min_confidence: float) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv, dtype=np.float64)
    (h, w) = depth.shape[:2]
    x = np.clip(np.rint((uv[:, 0] + 0.5) * w / mask.shape[1] - 0.5).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint((uv[:, 1] + 0.5) * h / mask.shape[0] - 0.5).astype(np.int64), 0, h - 1)
    xi = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, mask.shape[1] - 1)
    yi = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, mask.shape[0] - 1)
    z = np.asarray(depth[y, x], dtype=np.float64)
    q = np.asarray(confidence[y, x], dtype=np.float64)
    valid = (mask[yi, xi] > 0) & np.isfinite(z) & (z > 0.0001) & np.isfinite(q) & (q >= float(min_confidence))
    (fx, fy, cx, cy) = np.asarray(K, dtype=np.float64).reshape(-1).tolist()
    camera = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
    homogeneous = np.column_stack((camera, np.ones(len(camera), dtype=np.float64)))
    world = (homogeneous @ np.asarray(T_world_camera, dtype=np.float64).T)[:, :3]
    valid &= np.isfinite(world).all(axis=1)
    return (world, valid)

def valid_initial_poses(report: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    from observations.common import pose_from_values as pose_from_values
    poses = {}
    for row in report.get('pose_rows', []):
        pose = pose_from_values(row.get('rotation_world_from_completed_canonical_matrix'), row.get('translation_world_m'))
        if pose is not None:
            poses[int(row['frame_idx'])] = pose
    return poses

def rgb_source_eligibility(frame: dict[str, Any], object_id: str, poses: dict) -> str | None:
    """A target need not have depth/pose; a source must have accepted P09 and P14."""
    from envs.qualification.object_observation import p09_geometry_validation as p09_geometry_validation
    obj = object_for_frame(frame, object_id)
    geom = obj.get('visible_geometry_candidate') if obj else None
    if not isinstance(geom, dict):
        return 'missing_p09_visible_geometry'
    if obj.get('rigid_pose_observation_eligible') is not True or geom.get('rigid_pose_observation_eligible') is not True:
        return 'source_p09_not_explicitly_eligible'
    (points, reason, _) = p09_geometry_validation(obj, frame)
    if points is None:
        return reason
    if int(frame['frame_idx']) not in poses:
        return 'missing_or_invalid_p14_source_pose'
    return None

def load_p09_metric_eligibility(adapter: dict[str, Any], depth_path: Path, depth_ids: np.ndarray, depths: np.ndarray):
    """Use the same recorded per-pixel eligibility consumer as P09."""
    inputs = adapter.get('inputs', {})
    recorded = inputs.get('metric_eligibility_manifest')
    enabled = adapter.get('parameters', {}).get('native_depth_metric_eligibility_enabled', False)
    if not enabled:
        if recorded:
            raise ValueError('P09 eligibility input conflicts with disabled parameter')
        return None
    if not recorded or sha256_file(Path(recorded)) != inputs.get('metric_eligibility_manifest_sha256'):
        raise ValueError('P09 metric eligibility manifest missing or changed')
    if Path(inputs['depth_npz']).resolve() != depth_path.resolve():
        raise ValueError('RGB depth differs from P09 eligibility depth')
    from observations.object.surface import MetricEligibility as MetricEligibility
    return MetricEligibility(Path(recorded), case=adapter['case'], object_id=adapter['object_id'], track_id=adapter['track_id'], sam2_track=Path(inputs['sam2_track']), depth_npz=depth_path, base_annotations=Path(inputs['base_annotations']), calibration_contract=Path(inputs['calibration_contract']), frame_to_depth_i={int(frame): i for (i, frame) in enumerate(depth_ids)}, depth=depths)

def reproduce_source_depth_support(obj: dict[str, Any], mask: np.ndarray, depth: np.ndarray, confidence: np.ndarray, intrinsics: np.ndarray, parameters: dict[str, Any], metric_eligibility=None) -> np.ndarray:
    """Replay P09's exact retained raster, not a depth interval or sampled proxy."""
    from observations.object.surface import robust_first_surface_depth_ownership as robust_first_surface_depth_ownership
    geometry = obj['visible_geometry_candidate']
    own = geometry['first_surface_depth_ownership']
    contract = geometry['mask_depth_transform_contract']
    if list(mask.shape[::-1]) != contract['mask_size_wh'] or list(depth.shape[::-1]) != contract['depth_size_wh']:
        raise RuntimeError('P09 mask/depth raster size mismatch')
    (sx, sy) = (depth.shape[1] / mask.shape[1], depth.shape[0] / mask.shape[0])
    affine = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    if not np.allclose(affine, contract['A_depth_from_mask_coordinate_model'], atol=1e-08, rtol=0):
        raise RuntimeError('P09 mask/depth affine is not the declared half-pixel resize')
    owned = cv2.resize(mask, depth.shape[::-1], interpolation=cv2.INTER_NEAREST_EXACT) > 0
    valid = owned & np.isfinite(depth) & (depth >= parameters['min_depth_m']) & (depth <= parameters['max_depth_m'])
    if parameters.get('native_depth_metric_eligibility_enabled') or geometry.get('native_depth_metric_eligibility'):
        if metric_eligibility is None:
            raise ValueError('P09 metric eligibility is required for RGB support replay')
        binding = geometry['native_depth_metric_eligibility']
        if Path(binding['manifest']).resolve() != metric_eligibility.path:
            raise ValueError('RGB/P09 metric eligibility identity differs')
        source_mask = Path(geometry['source_mask_path']).resolve()
        source_pixels = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
        if source_pixels is None or not np.array_equal(source_pixels > 0, mask > 0):
            raise ValueError('P09 emitted mask differs from its eligibility input')
        (fx, fy, cx, cy) = np.asarray(intrinsics, dtype=float).reshape(4)
        (eligible, checked) = metric_eligibility.frame(int(geometry['frame_idx']), owned_mask_path=source_mask, depth_shape=depth.shape, A_depth_from_mask=affine, K_depth=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=float))
        if Path(checked['raster']).resolve() != Path(binding['raster']).resolve():
            raise ValueError('RGB/P09 metric eligibility raster differs')
        valid &= eligible
    names = ('mad_sigma', 'min_half_width_m', 'min_retained_fraction', 'fail_raw_to_robust_extent_ratio', 'confidence_seed_percentile', 'local_depth_step_max_m', 'max_removed_distance_inside_mask_px', 'max_confidence_flagged_interior_fraction', 'min_interior_confidence_flagged_fraction', 'max_unexplained_interior_pixels', 'min_component_pixels', 'max_small_component_fraction')
    (retained, summary) = robust_first_surface_depth_ownership(valid, depth, enabled=parameters['robust_first_surface_depth_ownership'], intrinsics=intrinsics, confidence=confidence, **{name: parameters['first_surface_' + name] for name in names})
    if summary.get('fail_closed') or any((summary.get(k) != own.get(k) for k in ('input_valid_depth_pixels', 'retained_depth_pixels', 'removed_depth_pixels', 'state'))):
        raise RuntimeError(f"P09 retained depth reproduction mismatch at frame {geometry['frame_idx']}")
    return retained

def plan_source_edge_pairs(selected: list[int], key_ids: list[int], sources: set[int], span: int, max_gap: int) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    """Preserve valid pairs; reverse one-sided pairs; add at most two brackets."""
    pairs = set()
    rejected = []
    for (a, b) in build_edge_pairs(selected, key_ids, span):
        if a in sources:
            pairs.add((a, b))
        elif b in sources:
            pairs.add((b, a))
        else:
            rejected.append({'source_frame_idx': a, 'target_frame_idx': b, 'status': 'rejected', 'reason': 'neither_endpoint_is_valid_source'})
    for target in selected:
        if target in sources:
            continue
        before = [s for s in sources if 0 < target - s <= max_gap]
        after = [s for s in sources if 0 < s - target <= max_gap]
        for source in ([max(before)] if before else []) + ([min(after)] if after else []):
            pairs.add((source, target))
    return (sorted(pairs), rejected)

def preflight_manifest_images(manifest_path: Path, frames: dict[int, dict[str, Any]], selected: list[int], object_id: str, source_wh: tuple[int, int]):
    """Validate the P01 raster before constructing feature models.

    P09 source dimensions describe the video/depth plane, not the P01 JPEG.
    Return both untouched owned masks (P09/material evidence) and appearance
    masks resized to the actual manifest image for feature matching.
    """
    manifest = load_json(manifest_path)
    by_id = {}
    for row in manifest.get('frames', []):
        idx = row.get('frame_idx')
        if type(idx) is not int or idx in by_id:
            raise RuntimeError('invalid or duplicate manifest frame ID')
        by_id[idx] = row
    (images, native_masks, matching_masks, planes) = ({}, {}, {}, {})
    for idx in selected:
        if idx not in by_id:
            raise RuntimeError(f'manifest RGB frame missing: {idx}')
        (row, frame) = (by_id[idx], frames[idx])
        video_wh = (row.get('source_width'), row.get('source_height'))
        manifest_wh = (row.get('manifest_width'), row.get('manifest_height'))
        if video_wh != tuple(source_wh) or any((type(v) is not int or v <= 0 for v in manifest_wh)) or (not np.isclose(float(row['time_s']), float(frame['time_s']), atol=1e-06, rtol=0)):
            raise RuntimeError(f'manifest/source/depth/time contract mismatch at frame {idx}')
        path = Path(row.get('rgb') or row.get('raw_frame_path') or '')
        if not path.is_absolute() or not path.resolve().is_relative_to(manifest_path.resolve().parent):
            raise RuntimeError(f'manifest RGB path escapes declared input at frame {idx}')
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[1::-1] != manifest_wh:
            raise RuntimeError(f'RGB/manifest image plane mismatch at frame {idx}')
        obj = object_for_frame(frame, object_id)
        if obj is None:
            continue
        mask = cv2.imread(str(obj.get('mask_path') or ''), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        geometry = obj.get('visible_geometry_candidate') or {}
        if geometry:
            if (geometry.get('source_width'), geometry.get('source_height')) != video_wh:
                raise RuntimeError(f'P09/source image plane mismatch at frame {idx}')
            declared_mask = (geometry.get('mask_depth_transform_contract') or {}).get('mask_size_wh')
            if declared_mask is not None and list(mask.shape[::-1]) != declared_mask:
                raise RuntimeError(f'P09 owned-mask size mismatch at frame {idx}')
        images[idx] = image
        native_masks[idx] = mask
        matching_masks[idx] = cv2.resize(mask, manifest_wh, interpolation=cv2.INTER_NEAREST_EXACT)
        planes[idx] = {'source_size_wh': list(video_wh), 'manifest_size_wh': list(manifest_wh), 'owned_mask_size_wh': list(mask.shape[::-1]), 'rgb_path': str(path), 'matching_plane': 'manifest_rgb', 'serialized_evidence_plane': 'original_owned_mask', 'pixel_affine': 'u_to=(u_from+0.5)*size_to/size_from-0.5'}
    return (images, native_masks, matching_masks, planes)

def feature_pair(source_idx: int, target_idx: int, features: dict[int, dict[str, torch.Tensor]], masks: dict[int, np.ndarray], frames: dict[int, dict[str, Any]], depths: np.ndarray, confidences: np.ndarray, depth_pos: dict[int, int], depth_intrinsics: np.ndarray, source_wh: tuple[int, int], matcher: torch.nn.Module, device: torch.device, args: argparse.Namespace, initial_pose_by_frame: dict[int, tuple[np.ndarray, np.ndarray]], source_support_by_frame: dict[int, np.ndarray], target_evidence_size_wh: tuple[int, int] | None=None) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray] | None, dict[str, np.ndarray] | None]:
    import torch
    reason = rgb_source_eligibility(frames[source_idx], args.object_id, initial_pose_by_frame)
    if reason is not None or source_idx not in source_support_by_frame:
        return ({'source_frame_idx': source_idx, 'target_frame_idx': target_idx, 'status': 'rejected', 'reason': reason or 'missing_retained_p09_depth_support'}, None, None)
    from lightglue.utils import rbd
    with torch.inference_mode():
        matched = rbd(matcher({'image0': features[source_idx], 'image1': features[target_idx]}))
    matched = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value for (key, value) in matched.items()}
    raw_pairs = matched.get('matches')
    pairs = np.asarray(raw_pairs if raw_pairs is not None else [], dtype=np.int64)
    raw_scores = matched.get('scores')
    scores = np.asarray(raw_scores if raw_scores is not None else [], dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        pairs = np.empty((0, 2), dtype=np.int64)
    kp0 = features[source_idx]['keypoints'].detach().cpu().numpy()
    kp1 = features[target_idx]['keypoints'].detach().cpu().numpy()
    if kp0.ndim == 3:
        kp0 = kp0[0]
    if kp1.ndim == 3:
        kp1 = kp1[0]
    uv0 = kp0[pairs[:, 0]] if len(pairs) else np.empty((0, 2))
    uv1 = kp1[pairs[:, 1]] if len(pairs) else np.empty((0, 2))
    mask0 = masks[source_idx]
    mask1 = masks[target_idx]
    if len(pairs):
        x0 = np.clip(np.rint(uv0[:, 0]).astype(np.int64), 0, mask0.shape[1] - 1)
        y0 = np.clip(np.rint(uv0[:, 1]).astype(np.int64), 0, mask0.shape[0] - 1)
        x1 = np.clip(np.rint(uv1[:, 0]).astype(np.int64), 0, mask1.shape[1] - 1)
        y1 = np.clip(np.rint(uv1[:, 1]).astype(np.int64), 0, mask1.shape[0] - 1)
        erode0 = cv2.erode((mask0 > 0).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
        erode1 = cv2.erode((mask1 > 0).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
        keep = (erode0[y0, x0] > 0) & (erode1[y1, x1] > 0)
        uv0 = uv0[keep]
        uv1 = uv1[keep]
        scores = scores[keep] if len(scores) == len(keep) else np.ones(len(uv0))
    source_frame = frames[source_idx]
    target_frame = frames[target_idx]
    source_camera = np.asarray(source_frame['camera']['T_world_camera_metric'], dtype=np.float64)
    target_camera = np.asarray(target_frame['camera']['T_world_camera_metric'], dtype=np.float64)
    source_intr = depth_intrinsics[depth_pos[source_idx]]
    target_intr = np.asarray(target_frame['camera']['intrinsics_fx_fy_cx_cy'], dtype=float)
    support = source_support_by_frame[source_idx]
    dx = np.clip(np.rint((uv0[:, 0] + 0.5) * support.shape[1] / mask0.shape[1] - 0.5).astype(int), 0, support.shape[1] - 1)
    dy = np.clip(np.rint((uv0[:, 1] + 0.5) * support.shape[0] / mask0.shape[0] - 0.5).astype(int), 0, support.shape[0] - 1)
    retained = support[dy, dx]
    (uv0, uv1) = (uv0[retained], uv1[retained])
    scores = scores[retained] if len(scores) == len(retained) else np.ones(len(uv0))
    (source_world, valid) = lift_source_world(source_idx, uv0, mask0, depths[depth_pos[source_idx]], confidences[depth_pos[source_idx]], source_intr, source_camera, float(args.min_confidence))
    source_world = source_world[valid]
    target_uv = uv1[valid]
    if len(source_world) < int(args.min_matches):
        return ({'source_frame_idx': source_idx, 'target_frame_idx': target_idx, 'status': 'rejected', 'matches_total': int(len(pairs)), 'matches_object': int(len(source_world)), 'reason': 'too_few_object_matches'}, None, None)
    K_target = np.asarray([[target_intr[0], 0.0, target_intr[2]], [0.0, target_intr[1], target_intr[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_target = resize_intrinsics(K_target, source_wh, (mask1.shape[1], mask1.shape[0]))
    (success, rvec, tvec, inliers) = cv2.solvePnPRansac(source_world.astype(np.float64), target_uv.astype(np.float64), K_target, np.zeros(5, dtype=np.float64), iterationsCount=int(args.pnp_iterations), reprojectionError=float(args.reprojection_threshold_px), confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not success or inliers is None or len(inliers) < 3:
        return ({'source_frame_idx': source_idx, 'target_frame_idx': target_idx, 'status': 'rejected', 'matches_total': int(len(pairs)), 'matches_object': int(len(source_world)), 'reason': 'pnp_failed'}, None, None)
    inliers = inliers[:, 0]
    try:
        (rvec, tvec) = cv2.solvePnPRefineLM(source_world[inliers], target_uv[inliers], K_target, np.zeros(5), rvec, tvec)
    except cv2.error:
        pass
    (R_camera_source, _) = cv2.Rodrigues(rvec)
    T_wc_target = target_camera
    R_delta = T_wc_target[:3, :3] @ R_camera_source
    t_delta = T_wc_target[:3, :3] @ tvec[:, 0] + T_wc_target[:3, 3]
    (projected, _) = cv2.projectPoints(source_world[inliers], rvec, tvec, K_target, np.zeros(5))
    reprojection = np.linalg.norm(projected.reshape(-1, 2) - target_uv[inliers], axis=1)
    covariance = np.cov(source_world[inliers].T) if len(inliers) >= 4 else np.zeros((3, 3))
    eig = np.linalg.eigvalsh(covariance) if np.isfinite(covariance).all() else np.zeros(3)
    eig = np.maximum(eig, 0.0)
    conditioning = float(eig[0] / max(eig[-1], 1e-12))
    inlier_fraction = float(len(inliers) / max(1, len(source_world)))
    rotation_deg = float(np.degrees(Rotation.from_matrix(R_delta).magnitude()))
    transformed = source_world @ R_camera_source.T + tvec[:, 0]
    status = 'accepted' if len(inliers) >= 4 and np.isfinite(reprojection).all() and np.all(transformed[inliers, 2] > 0) else 'rejected'
    reason = None if status == 'accepted' else 'invalid_pnp_cheirality_or_finite_support'
    row = {'source_frame_idx': source_idx, 'target_frame_idx': target_idx, 'frame_gap': target_idx - source_idx, 'status': status, 'reason': reason, 'source_samples_quarantined': int((~retained).sum()), 'target_initial_pose_present': target_idx in initial_pose_by_frame, 'uncertainty': 'RGB_PnP_measurement_not_direct_target_metric_surface', 'matches_total': int(len(pairs)), 'matches_object': int(len(source_world)), 'inliers': int(len(inliers)), 'inlier_fraction': inlier_fraction, 'matching_score_median': float(np.median(scores)) if len(scores) else None, 'rotation_deg': rotation_deg, 'translation_norm_m': float(np.linalg.norm(t_delta)), 'reprojection_median_px': float(np.median(reprojection)), 'reprojection_p95_px': float(np.percentile(reprojection, 95)), 'source_3d_covariance_eigenvalues': eig.tolist(), 'source_3d_conditioning': conditioning}
    evidence = None
    if status == 'accepted':
        if source_idx not in initial_pose_by_frame:
            raise RuntimeError(f'initial pose report lacks RGB source frame {source_idx}')
        (source_R0, source_t0) = initial_pose_by_frame[source_idx]
        canonical_points = (source_world[inliers] - source_t0[None, :]) @ source_R0
        evidence_size = target_evidence_size_wh or (mask1.shape[1], mask1.shape[0])
        evidence_K = resize_intrinsics(K_target, (mask1.shape[1], mask1.shape[0]), evidence_size)
        evidence_uv = (target_uv[inliers] + 0.5) * (np.asarray(evidence_size) / np.asarray(mask1.shape[::-1])) - 0.5
        target_intrinsics = np.asarray([evidence_K[0, 0], evidence_K[1, 1], evidence_K[0, 2], evidence_K[1, 2]], dtype=np.float64)
        point_weights = np.clip(np.exp(-reprojection / 2.0), 0.25, 1.0)
        evidence = {'canonical_points': np.asarray(canonical_points, dtype=np.float64), 'target_uv': np.asarray(evidence_uv, dtype=np.float64), 'target_intrinsics': target_intrinsics, 'target_camera': np.asarray(target_camera, dtype=np.float64), 'weights': point_weights}
        row['reprojection_evidence_count'] = int(len(canonical_points))
        row['reprojection_evidence_contract'] = 'source_initial_canonical_points_to_target_uv'
    return (row, (R_delta, t_delta) if status == 'accepted' else None, evidence)

def build_edge_pairs(selected: list[int], key_ids: list[int], keyframe_neighbor_span: int) -> list[tuple[int, int]]:
    if int(keyframe_neighbor_span) < 1:
        raise RuntimeError('keyframe_neighbor_span must be positive')
    pairs = set(zip(selected[:-1], selected[1:]))
    for offset in range(1, int(keyframe_neighbor_span) + 1):
        for (a, b) in zip(key_ids[:-offset], key_ids[offset:]):
            pairs.add((a, b))
    if len(key_ids) >= 4:
        for (a, b) in ((key_ids[0], key_ids[len(key_ids) // 2]), (key_ids[len(key_ids) // 2], key_ids[-1]), (key_ids[0], key_ids[-1])):
            pairs.add((a, b))
    return sorted(((a, b) for (a, b) in pairs if a < b))

def diagnostic_motion_chain(anchor: int, poses: dict, motions: dict) -> dict:
    """Diagnostic traversal accepts reversed edges; never creates source poses."""
    adjacency = {}
    for ((a, b), (R, t)) in motions.items():
        adjacency.setdefault(a, []).append((b, R, t))
        adjacency.setdefault(b, []).append((a, R.T, -R.T @ t))
    chain = {anchor: poses[anchor]}
    queue = [anchor]
    for a in queue:
        (Ra, ta) = chain[a]
        for (b, R, t) in adjacency.get(a, []):
            if b not in chain:
                chain[b] = (R @ Ra, R @ ta + t)
                queue.append(b)
    return chain

def build_rgb_edges(argv=None) -> None:
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--raw-frame-manifest', type=Path, help='Source P01 manifest; explicit for candidate-local annotations.')
    parser.add_argument('--depth-npz', type=Path, required=True)
    parser.add_argument('--initial-pose-report', type=Path, required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--frame-start', type=int, default=0)
    parser.add_argument('--frame-end', type=int, required=True)
    parser.add_argument('--keyframe-seconds', type=float, default=0.5)
    parser.add_argument('--source-gap-seconds', type=float, default=2.0)
    parser.add_argument('--keyframe-interval', type=int, default=5)
    parser.add_argument('--keyframe-neighbor-span', type=int, default=1, help='Connect each keyframe to this many subsequent keyframes.')
    parser.add_argument('--max-keyframes', type=int, default=40)
    parser.add_argument('--max-keypoints', type=int, default=2048)
    parser.add_argument('--ratio-test', type=float, default=0.78)
    parser.add_argument('--min-confidence', type=float, default=0.05)
    parser.add_argument('--min-matches', type=int, default=8)
    parser.add_argument('--min-inliers', type=int, default=30)
    parser.add_argument('--min-inlier-fraction', type=float, default=0.2)
    parser.add_argument('--pnp-iterations', type=int, default=1000)
    parser.add_argument('--reprojection-threshold-px', type=float, default=2.5)
    parser.add_argument('--max-reprojection-median-px', type=float, default=1.5)
    parser.add_argument('--max-edge-rotation-deg', type=float, default=180.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lightglue-root', type=Path, required=True)
    parser.add_argument('--output-npz', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-pose-report', type=Path, default=None)
    parser.add_argument('--max-bracketing-source-gap', type=int, default=None, help='Default: keyframe interval times neighbor span; at most one source per side.')
    parser.add_argument('--assets-manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    from utils.runtime import bind_offline_weights as bind_offline_weights
    config = load_json(args.assets_manifest)
    bind_offline_weights(config['lightglue_weights'], torch)
    device = torch.device(args.device)
    annotations = load_json(args.annotations)
    assert_prediction_only(annotations, label='RGB edge annotations', object_id=args.object_id)
    initial_report_path = args.initial_pose_report.expanduser().resolve()
    initial_report = load_json(initial_report_path)
    assert_prediction_only(initial_report, label='RGB edge initial pose report', object_id=args.object_id)
    frames = {int(f['frame_idx']): f for f in annotations.get('frames', []) if isinstance(f, dict) and f.get('frame_idx') is not None}
    annotation_root = args.annotations.expanduser().resolve()
    case_root = find_case_root(annotation_root)
    with np.load(args.depth_npz.expanduser().resolve(), allow_pickle=False) as data:
        required = {'frame_idx', 'depth', 'confidence', 'source_size', 'intrinsics_fx_fy_cx_cy'}
        missing_keys = sorted(required - set(data.files))
        if missing_keys:
            raise RuntimeError(f'depth archive lacks camera contract keys: {missing_keys}')
        depth_ids = np.asarray(data['frame_idx'], dtype=np.int64)
        depths = np.asarray(data['depth'], dtype=np.float32)
        confidences = np.asarray(data['confidence'], dtype=np.float32)
        source_size_array = np.asarray(data['source_size'], dtype=np.int64).reshape(-1)
        depth_intrinsics = np.asarray(data['intrinsics_fx_fy_cx_cy'], dtype=np.float64)
        source_wh = validate_depth_contract(depth_ids, depths, confidences, source_size_array, depth_intrinsics)
    depth_pos = {int(idx): i for (i, idx) in enumerate(depth_ids.tolist())}
    selected = sorted((i for i in frames if args.frame_start <= i <= args.frame_end), key=lambda i: frames[i]['time_s'])
    times = {i: float(frames[i]['time_s']) for i in selected}
    if any((times[b] <= times[a] for (a, b) in zip(selected, selected[1:]))):
        raise ValueError('strictly increasing timestamps required')
    for idx in selected:
        if idx not in depth_pos:
            depth_pos[idx] = len(depth_intrinsics)
            depth_intrinsics = np.vstack([depth_intrinsics, frames[idx]['camera']['intrinsics_fx_fy_cx_cy']])
    if len(selected) < 2:
        raise RuntimeError('not enough selected frames')
    initial_pose_by_frame = valid_initial_poses(initial_report)
    manifest_path = resolve_raw_frame_manifest(args.annotations, args.raw_frame_manifest)
    (images, native_masks, masks, image_planes) = preflight_manifest_images(manifest_path, frames, selected, args.object_id, source_wh)
    source_rejections = {idx: rgb_source_eligibility(frames[idx], args.object_id, initial_pose_by_frame) for idx in selected}
    eligible_sources = {idx for (idx, reason) in source_rejections.items() if reason is None and depth_pos[idx] < len(depths)}
    if not eligible_sources:
        raise RuntimeError('no eligible P09/P14 RGB source/diagnostic anchor before feature extraction')
    adapter_path = args.annotations.parent / 'v19_visible_geometry_adapter_report.json'
    adapter = load_json(adapter_path)
    assert_prediction_only(adapter, label='RGB P09 retained depth parameters', object_id=args.object_id)
    parameters = adapter['parameters']
    metric_eligibility = load_p09_metric_eligibility(adapter, args.depth_npz.expanduser().resolve(), depth_ids, depths)
    max_gap = args.max_bracketing_source_gap
    if max_gap is None:
        max_gap = int(args.keyframe_interval) * int(args.keyframe_neighbor_span)
    if max_gap < 1:
        raise RuntimeError('max bracketing source gap must be positive')
    source_support_by_frame = {}
    for idx in sorted(eligible_sources):
        obj = object_for_frame(frames[idx], args.object_id)
        mask = native_masks.get(idx)
        if mask is None:
            source_rejections[idx] = 'missing_source_mask'
            continue
        pos = depth_pos[idx]
        source_support_by_frame[idx] = reproduce_source_depth_support(obj, mask, depths[pos], confidences[pos], depth_intrinsics[pos], parameters, metric_eligibility=metric_eligibility)
    if not source_support_by_frame:
        raise RuntimeError('no reproducible P09 RGB source/diagnostic anchor before feature extraction')
    sys.path.insert(0, str(args.lightglue_root.expanduser().resolve()))
    from lightglue import LightGlue, SuperPoint
    extractor = SuperPoint(max_num_keypoints=int(args.max_keypoints)).eval().to(device)
    matcher = LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95, filter_threshold=0.2).eval().to(device)
    features: dict[int, dict[str, torch.Tensor]] = {}
    for (n, idx) in enumerate(selected, 1):
        if idx not in images or idx not in masks:
            continue
        image = images.pop(idx)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).to(device)
        with torch.inference_mode():
            features[idx] = {key: value.detach() for (key, value) in extractor.extract(tensor, resize=None).items()}
        if n == 1 or n % 25 == 0 or n == len(selected):
            print(f'[global-rgb] extracted {n}/{len(selected)}', flush=True)
    from utils.data import time_keyframes as time_keyframes
    from utils.data import time_source_pairs as time_source_pairs
    key_ids = time_keyframes([i for i in selected if i in features], times, args.keyframe_seconds)
    eligible_sources = set(source_support_by_frame) & set(features)
    if not eligible_sources:
        raise RuntimeError('no eligible source has RGB features')
    (edge_pairs, rejected_pairs) = time_source_pairs(selected, key_ids, eligible_sources, int(args.keyframe_neighbor_span), times, args.source_gap_seconds)
    unavailable_pairs = [{'source_frame_idx': a, 'target_frame_idx': b, 'status': 'rejected', 'reason': 'missing_RGB_features'} for (a, b) in edge_pairs if a not in features or b not in features]
    edge_pairs = [(a, b) for (a, b) in edge_pairs if a in features and b in features]
    edges: list[dict[str, Any]] = []
    edge_motions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    edge_reprojection_evidence: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for (n, (a, b)) in enumerate(edge_pairs, 1):
        (row, motion, evidence) = feature_pair(a, b, features, masks, frames, depths, confidences, depth_pos, depth_intrinsics, source_wh, matcher, device, args, initial_pose_by_frame, source_support_by_frame, target_evidence_size_wh=tuple(native_masks[b].shape[::-1]))
        row['topological_role'] = 'adjacent' if abs(selected.index(a) - selected.index(b)) == 1 else 'keyframe_loop'
        (row['source_time_s'], row['target_time_s']) = (times[a], times[b])
        edges.append(row)
        if motion is not None:
            edge_motions[a, b] = motion
        if evidence is not None:
            edge_reprojection_evidence[a, b] = evidence
        print(f"[global-rgb] edge {n}/{len(edge_pairs)} {a}->{b} {row['status']} matches={row.get('matches_object')} inliers={row.get('inliers')}", flush=True)
    diagnostic_anchor = min(eligible_sources)
    chain = diagnostic_motion_chain(diagnostic_anchor, initial_pose_by_frame, edge_motions)
    chain_rows = []
    for original in initial_report.get('pose_rows', []):
        if not isinstance(original, dict):
            chain_rows.append(original)
            continue
        row = dict(original)
        idx = int(row.get('frame_idx', -1))
        if idx in chain:
            row['rotation_world_from_completed_canonical_matrix'] = chain[idx][0].tolist()
            row['translation_world_m'] = chain[idx][1].tolist()
            row['pose_source'] = 'v20_prediction_lightglue_rgb_depth_chain_diagnostic'
            row['diagnostic_only'] = True
            row['generated_geometry_pose_evidence_consumed'] = False
        chain_rows.append(row)
    metadata = {'schema': 'v20_prediction_lightglue_rgb_depth_global_orientation_edges_v2', 'annotations': str(args.annotations.expanduser().resolve()), 'depth_npz': str(args.depth_npz.expanduser().resolve()), 'initial_pose_report': str(args.initial_pose_report.expanduser().resolve()), 'raw_frame_manifest': str(manifest_path), 'object_id': args.object_id, 'prediction_side_only': True, 'generated_geometry_consumed': False, 'gt_used_as_solver_input': False, 'reprojection_evidence_contract': {'present': True, 'storage': 'NPZ_flattened_points_with_edge_offsets', 'canonical_frame': 'initial_prediction_pose_source_object_frame', 'target_frame': 'camera_pixels_at_mask_resolution', 'used_for_pose_objective_only_when_explicitly_enabled': True}, 'camera_contract': {'source_size_wh': list(source_wh), 'depth_raster_shape': list(depths.shape[1:]), 'intrinsics_plane': 'depth_source_plane', 'target_intrinsics_resize_from_source_wh': list(source_wh), 'matching_uv_plane': 'manifest_rgb', 'reprojection_evidence_uv_plane': 'original_owned_mask', 'per_frame_planes': {str(k): v for (k, v) in image_planes.items()}}, 'input_sha256': {'annotations': sha256_file(args.annotations), 'depth_npz': sha256_file(args.depth_npz), 'p09_adapter_report': sha256_file(adapter_path), 'raw_frame_manifest': sha256_file(manifest_path), 'initial_pose_report': sha256_file(args.initial_pose_report)}}
    if metric_eligibility is not None:
        metadata['metric_eligibility_manifest'] = str(metric_eligibility.path)
        metadata['input_sha256']['metric_eligibility_manifest'] = sha256_file(metric_eligibility.path)
        metadata['source_metric_support_policy'] = 'exact_P09_eligibility_intersection_then_robust_depth_replay'
    report = {**metadata, 'frame_start': int(args.frame_start), 'frame_end': int(args.frame_end), 'selected_frame_count': len(selected), 'keyframes': key_ids, 'edge_count': len(edges), 'accepted_count': sum((row.get('status') == 'accepted' for row in edges)), 'rejected_count': sum((row.get('status') != 'accepted' for row in edges)), 'reprojection_evidence_edge_count': len(edge_reprojection_evidence), 'reprojection_evidence_point_count': int(sum((len(value['weights']) for value in edge_reprojection_evidence.values()))), 'diagnostic_anchor_frame_idx': diagnostic_anchor, 'source_rejections': [{'frame_idx': idx, 'reason': reason} for (idx, reason) in source_rejections.items() if reason], 'pair_planning_rejections': rejected_pairs + unavailable_pairs, 'eligible_source_frame_ids': sorted(eligible_sources), 'missing_pose_target_frame_ids': sorted(set(selected) - set(initial_pose_by_frame)), 'targets_without_planned_source': sorted(set(selected) - eligible_sources - {b for (_, b) in edge_pairs}), 'targets_without_accepted_source': sorted(set(selected) - eligible_sources - {b for (_, b) in edge_motions}), 'uncertainty': 'Missing metric targets remain RGB-only uncertain observations; no direct P14 poses added.', 'chain_covered_frame_count': len(chain), 'rows': edges, 'parameters': {'max_bracketing_source_gap': max_gap, 'keyframe_interval': int(args.keyframe_interval), 'keyframe_neighbor_span': int(args.keyframe_neighbor_span), 'max_keyframes': int(args.max_keyframes), 'max_edge_rotation_deg': float(args.max_edge_rotation_deg), 'min_matches': int(args.min_matches), 'min_inliers': int(args.min_inliers), 'min_inlier_fraction': float(args.min_inlier_fraction), 'max_reprojection_median_px': float(args.max_reprojection_median_px)}}
    append_prediction_stage(report, stage='prediction_rgb_depth_edge_builder', input_hashes=metadata['input_sha256'], notes={'source_size_wh': list(source_wh), 'gt_consumed': False})
    evidence_offsets = [0]
    evidence_points: list[np.ndarray] = []
    evidence_uv: list[np.ndarray] = []
    evidence_weights: list[np.ndarray] = []
    evidence_intrinsics: list[np.ndarray] = []
    evidence_camera: list[np.ndarray] = []
    for row in edges:
        key = (int(row['source_frame_idx']), int(row['target_frame_idx']))
        evidence = edge_reprojection_evidence.get(key)
        if evidence is None:
            evidence_points.append(np.empty((0, 3), dtype=np.float64))
            evidence_uv.append(np.empty((0, 2), dtype=np.float64))
            evidence_weights.append(np.empty((0,), dtype=np.float64))
            evidence_intrinsics.append(np.zeros((4,), dtype=np.float64))
            evidence_camera.append(np.zeros((4, 4), dtype=np.float64))
            evidence_offsets.append(evidence_offsets[-1])
        else:
            evidence_points.append(evidence['canonical_points'])
            evidence_uv.append(evidence['target_uv'])
            evidence_weights.append(evidence['weights'])
            evidence_intrinsics.append(evidence['target_intrinsics'])
            evidence_camera.append(evidence['target_camera'])
            evidence_offsets.append(evidence_offsets[-1] + len(evidence['weights']))
    flat_points = np.concatenate(evidence_points, axis=0) if evidence_points else np.empty((0, 3), dtype=np.float64)
    flat_uv = np.concatenate(evidence_uv, axis=0) if evidence_uv else np.empty((0, 2), dtype=np.float64)
    flat_weights = np.concatenate(evidence_weights, axis=0) if evidence_weights else np.empty((0,), dtype=np.float64)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, metadata=np.asarray([json.dumps(metadata)]), source_frame_idx=np.asarray([row['source_frame_idx'] for row in edges], dtype=np.int64), target_frame_idx=np.asarray([row['target_frame_idx'] for row in edges], dtype=np.int64), rotation_source_to_target_world=np.asarray([edge_motions.get((row['source_frame_idx'], row['target_frame_idx']), (np.eye(3), np.zeros(3)))[0] for row in edges]), translation_source_to_target_world_m=np.asarray([edge_motions.get((row['source_frame_idx'], row['target_frame_idx']), (np.eye(3), np.zeros(3)))[1] for row in edges]), accepted=np.asarray([row.get('status') == 'accepted' for row in edges], dtype=bool), source_size=np.asarray(source_wh, dtype=np.int64), reprojection_evidence_offsets=np.asarray(evidence_offsets, dtype=np.int64), reprojection_canonical_points=flat_points, reprojection_target_uv=flat_uv, reprojection_weights=flat_weights, reprojection_target_intrinsics=np.asarray(evidence_intrinsics, dtype=np.float64), reprojection_target_camera=np.asarray(evidence_camera, dtype=np.float64))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    if args.output_pose_report is not None:
        chain_report = dict(initial_report)
        chain_report.update({'status': 'v20_prediction_lightglue_rgb_depth_chain_diagnostic', 'annotation_ready': False, 'diagnostic_only': True, 'generated_geometry_pose_evidence_consumed': False, 'gt_used_as_solver_input': False, 'pose_rows': chain_rows, 'feature_edge_report': str(args.output_json.expanduser().resolve())})
        append_prediction_stage(chain_report, stage='prediction_rgb_depth_chain_builder', input_hashes=metadata['input_sha256'], notes={'source_size_wh': list(source_wh), 'gt_consumed': False})
        args.output_pose_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_pose_report.write_text(json.dumps(chain_report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'ok', 'edge_count': len(edges), 'accepted_count': report['accepted_count'], 'rejected_count': report['rejected_count'], 'keyframes': key_ids, 'chain_covered_frame_count': len(chain), 'output_json': str(args.output_json), 'output_pose_report': str(args.output_pose_report) if args.output_pose_report else None}, indent=2))
import copy
import heapq
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from typing import Mapping
from utils.geometry import valid_rotation
from observations.common import pose_from_values
from envs.qualification.object_observation import p09_geometry_validation

def initialize(base, annotations, object_id, edge_arrays, edge_rows, anchor):
    original = _timeline_rows(base)
    timeline = sorted((int(f['frame_idx']) for f in annotations['frames']))
    if len(set(timeline)) != len(timeline) or anchor not in timeline:
        raise ValueError('invalid timeline/anchor')
    initial = {}
    for (idx, row) in original.items():
        pose = pose_from_values(row.get('rotation_world_from_completed_canonical_matrix'), row.get('translation_world_m'))
        if pose is not None:
            initial[idx] = pose
    if anchor not in initial:
        raise ValueError('anchor has no source P14 pose')
    original_initial = dict(initial)
    for idx in timeline:
        if idx not in initial:
            initial[idx] = initialize_missing_se3_pose(idx, original_initial)[:2]
    quality = {(int(r['source_frame_idx']), int(r['target_frame_idx'])): r for r in edge_rows}
    (src, dst) = (edge_arrays['source_frame_idx'], edge_arrays['target_frame_idx'])
    Rs = edge_arrays['rotation_source_to_target_world']
    ts = edge_arrays['translation_source_to_target_world_m']
    accepted = edge_arrays['accepted']
    if not len(src) == len(dst) == len(Rs) == len(ts) == len(accepted):
        raise ValueError('edge length mismatch')
    adjacency = {i: [] for i in timeline}
    edges = []
    seen = set()
    for (a, b, R, t, ok) in zip(src, dst, Rs, ts, accepted):
        (a, b) = (int(a), int(b))
        if (a, b) in seen:
            raise ValueError('duplicate RGB edge')
        seen.add((a, b))
        if a not in adjacency or b not in adjacency:
            continue
        row = quality.get((a, b))
        if row is None or (row.get('status') == 'accepted') != bool(ok):
            raise ValueError('edge NPZ/JSON acceptance mismatch')
        if not ok:
            continue
        if not valid_rotation(R) or np.asarray(t).shape != (3,) or (not np.isfinite(t).all()):
            raise ValueError('invalid world-relative edge')
        condition = float(row.get('source_3d_conditioning', 0))
        inliers = float(row.get('inlier_fraction', 0))
        error = float(row.get('reprojection_median_px', np.inf))
        if not np.isfinite([condition, inliers, error]).all() or condition < 0 or (not 0 <= inliers <= 1) or (error < 0):
            raise ValueError('invalid RGB quality evidence')
        weight = float(np.clip(inliers * np.exp(-error / 2) * np.clip(condition / 0.02, 0.05, 1.0), 0, 1))
        if weight <= 0.01:
            continue
        cost = -np.log(weight) + 0.01 * abs(b - a)
        edges.append((a, b, R, t, weight))
        adjacency[a].append((b, R, t, cost))
        adjacency[b].append((a, R.T, -R.T @ t, cost))
    distance = {anchor: 0.0}
    seed = {anchor: initial[anchor]}
    queue = [(0.0, anchor)]
    while queue:
        (cost, a) = heapq.heappop(queue)
        if cost != distance[a]:
            continue
        for (b, R, t, w) in adjacency[a]:
            candidate = cost + w
            if candidate < distance.get(b, float('inf')):
                (Ra, ta) = seed[a]
                seed[b] = (R @ Ra, R @ ta + t)
                distance[b] = candidate
                heapq.heappush(queue, (candidate, b))
    connected = set(seed)
    poses = {idx: seed.get(idx, initial[idx]) for idx in timeline}
    seed_R = {idx: p[0] for (idx, p) in poses.items()}
    free = [idx for idx in timeline if idx in connected and idx != anchor]
    fp = {idx: n for (n, idx) in enumerate(free)}
    used = [e for e in edges if e[0] in connected and e[1] in connected]

    def rotations(x):
        rr = dict(seed_R)
        for (idx, n) in fp.items():
            rr[idx] = Rotation.from_rotvec(x.reshape(-1, 3)[n]).as_matrix() @ seed_R[idx]
        return rr

    def residual(x):
        rr = rotations(x)
        blocks = [x / 1.0]
        for (a, b, R, t, w) in used:
            blocks.append(np.sqrt(w) * Rotation.from_matrix(rr[b] @ rr[a].T @ R.T).as_rotvec() / 0.08)
        return np.concatenate(blocks)
    if free and used:
        result = least_squares(residual, np.zeros(len(free) * 3), loss='soft_l1', max_nfev=80)
        final_R = rotations(result.x)
        solve = {'success': bool(result.success), 'nfev': int(result.nfev), 'cost': float(result.cost)}
    else:
        final_R = seed_R
        solve = {'success': False, 'reason': 'no_connected_RGB_edges'}
    output = copy.deepcopy(base)
    new_rows = []
    frame_map = {int(f['frame_idx']): f for f in annotations['frames']}
    for idx in timeline:
        row = copy.deepcopy(original.get(idx, {}))
        (Rbefore, tbefore) = poses[idx]
        R = final_R[idx]
        obj = _object_for_frame(frame_map[idx], object_id)
        (points, _, _) = p09_geometry_validation(obj, frame_map[idx])
        support = np.mean(points, axis=0) if points is not None else tbefore
        canonical = (support - tbefore) @ Rbefore
        t = support - R @ canonical
        row.update(frame_idx=idx, rotation_world_from_completed_canonical_matrix=R.tolist(), translation_world_m=t.tolist())
        if idx not in original:
            row['status'] = 'rgb_initialized_uncertain'
        stale = {key: row.pop(key) for key in ('observed_to_mesh_initial', 'observed_to_mesh_final', 'mesh_to_observed_final') if key in row}
        row['source_P14_distance_diagnostics'] = stale
        row['rgb_initializer'] = {'anchor_connected': idx in connected, 'direct_metric_observation_added': False, 'source_status': original.get(idx, {}).get('status'), 'translation_policy': 'RGB_SE3_chain_then_support_preserving_SO3_correction', 'uncertain': idx not in connected or points is None}
        new_rows.append(row)
    if 'final_observed_to_mesh_median_summary_m' in output:
        output['source_P14_final_distance_summary'] = output.pop('final_observed_to_mesh_median_summary_m')
    output['pose_rows'] = new_rows
    if 'frame_fit_rows' in output:
        output['source_P14_frame_fit_rows'] = output.pop('frame_fit_rows')
        output['frame_fit_rows'] = copy.deepcopy(new_rows)
    mesh_path = base.get('pose_hypothesis_mesh_labeled') or base.get('inputs', {}).get('completed_mesh')
    if mesh_path:
        output['inputs'] = {**base.get('inputs', {}), 'completed_mesh': mesh_path}
    output['source_P14_method'] = base.get('method')
    output['method'] = 'build_v19_rgb_pose_initializer'
    output['pose_estimation_method'] = 'quality_weighted_RGB_SE3_spanning_graph_with_robust_SO3_loop_refinement'
    for name in ('observed_pose_evidence_diagnostics', 'pairwise_observed_surface_registration', 'rotation_unary_regularization', 'translation_regularization'):
        if name in output:
            output['source_P14_' + name] = output.pop(name)
    output['rgb_full_timeline_initialization'] = {'anchor': anchor, 'accepted_edge_count': len(edges), 'connected_frame_ids': sorted(connected), 'disconnected_frame_ids': sorted(set(timeline) - connected), 'loop_optimizer': solve, 'generated_geometry_consumed': False, 'gt_consumed': False}
    return output

def initialize_missing_se3_pose(frame_idx: int, direct_poses: Mapping[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, str]:
    """Initialize a missing timeline row by neighboring SE(3) interpolation."""
    if frame_idx in direct_poses:
        (R, t) = direct_poses[frame_idx]
        return (R.copy(), t.copy(), 'direct_metric_pose')
    ids = sorted((int(i) for i in direct_poses))
    if not ids:
        raise ValueError('cannot interpolate without at least one valid pose')
    lower = [i for i in ids if i < int(frame_idx)]
    upper = [i for i in ids if i > int(frame_idx)]
    if lower and upper:
        (lo, hi) = (lower[-1], upper[0])
        alpha = (float(frame_idx) - lo) / float(hi - lo)
        (R, t) = interpolate_se3_pose(*direct_poses[lo], *direct_poses[hi], alpha)
        return (R, t, f'continuous_se3_interpolation_{lo}_{hi}')
    nearest = min(ids, key=lambda value: abs(value - int(frame_idx)))
    (R, t) = direct_poses[nearest]
    return (R.copy(), t.copy(), f'se3_nearest_hold_{nearest}')

def _object_for_frame(frame: Mapping[str, Any], object_id: str) -> Mapping[str, Any] | None:
    objects = frame.get('objects')
    if not isinstance(objects, list):
        return None
    for obj in objects:
        if isinstance(obj, Mapping) and str(obj.get('object_id')) == str(object_id):
            return obj
    return None

def _timeline_rows(report: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows = report.get('pose_rows', [])
    result: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict) or row.get('frame_idx') is None:
            continue
        idx = int(row['frame_idx'])
        if idx in result:
            raise RuntimeError(f'duplicate timeline pose row {idx}')
        result[idx] = row
    return result
from utils.geometry import interpolate_se3_pose

def select_flow_seed_points(frame: dict[str, Any], obj: dict[str, Any], observed_world: np.ndarray, source_rotation: np.ndarray, source_translation: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import cv2
    from observations.object.pose import read_gray
    from pathlib import Path
    from typing import Any
    import numpy as np
    from observations.object.pose import frame_camera_for_raster
    from observations.object.pose import numeric_summary
    from observations.object.pose import apply_pose
    from observations.object.pose import read_mask_at_size
    raw_path = Path(str(frame.get('raw_frame_path') or ''))
    mask_path = Path(str(obj.get('mask_path') or ''))
    if not raw_path.is_file() or not mask_path.is_file():
        raise RuntimeError(f"optical-flow source frame lacks raw image/object-owned mask: {frame.get('frame_idx')}")
    gray = read_gray(raw_path)
    (height, width) = gray.shape
    mask = read_mask_at_size(mask_path, width, height)
    (world_camera, intrinsics, raster_contract) = frame_camera_for_raster(frame, width, height)
    camera_world = np.linalg.inv(world_camera)
    observed_camera = apply_pose(observed_world, camera_world[:3, :3], camera_world[:3, 3])
    positive = observed_camera[:, 2] > 1e-05
    uv = np.column_stack([intrinsics[0, 0] * observed_camera[:, 0] / np.maximum(observed_camera[:, 2], 1e-09) + intrinsics[0, 2], intrinsics[1, 1] * observed_camera[:, 1] / np.maximum(observed_camera[:, 2], 1e-09) + intrinsics[1, 2]])
    inside = positive & (uv[:, 0] >= 2.0) & (uv[:, 0] < width - 2.0) & (uv[:, 1] >= 2.0) & (uv[:, 1] < height - 2.0)
    canonical = (observed_world - source_translation) @ source_rotation
    uv = uv[inside]
    canonical = canonical[inside]
    rounded = np.rint(uv).astype(np.int64)
    owned = mask[rounded[:, 1], rounded[:, 0]]
    uv = uv[owned]
    canonical = canonical[owned]
    if len(uv) < 50:
        raise RuntimeError(f'too few projected object-owned flow seeds: {len(uv)}')
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gradient = np.hypot(gradient_x, gradient_y)
    rounded = np.rint(uv).astype(np.int64)
    scores = gradient[rounded[:, 1], rounded[:, 0]]
    order = np.argsort(scores)[::-1]
    selected = order[:min(max_points, len(order))]
    return (gray, uv[selected].astype(np.float32), canonical[selected].astype(float), {'raw_frame': str(raw_path), 'object_owned_mask': str(mask_path), 'projected_owned_seed_count': int(len(uv)), 'selected_seed_count': int(len(selected)), 'selected_gradient_summary': numeric_summary(scores[selected]), 'camera_raster_contract': raster_contract})

def estimate_optical_flow_pnp_bridge(*, source_idx: int, target_idx: int, frames: dict[int, dict[str, Any]], objects: dict[int, dict[str, Any]], observed_world: np.ndarray, source_rotation: np.ndarray, source_translation: np.ndarray, max_seed_points: int, min_tracked_points: int, min_pnp_inliers: int, min_inlier_fraction: float, max_reprojection_median_px: float, max_reprojection_p95_px: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    from observations.object.pose import numeric_summary
    import cv2
    from observations.object.pose import read_gray
    from observations.object.pose import rotation_observability_summary
    import numpy as np
    from observations.object.pose import apply_pose
    from pathlib import Path
    from typing import Any
    from observations.object.pose import frame_camera_for_raster
    from observations.object.pose import read_mask_at_size
    if source_idx == target_idx:
        raise RuntimeError('optical-flow bridge source and target must differ')
    direction = 1 if target_idx > source_idx else -1
    source_frame = frames[source_idx]
    source_obj = objects[source_idx]
    (previous_gray, source_uv, canonical_points, seed_report) = select_flow_seed_points(source_frame, source_obj, observed_world, source_rotation, source_translation, max_points=max_seed_points)
    previous_uv = source_uv.reshape(-1, 1, 2)
    active_canonical = canonical_points
    flow_trace: list[dict[str, Any]] = []
    current_uv = source_uv
    for frame_idx in range(source_idx + direction, target_idx + direction, direction):
        frame = frames.get(frame_idx)
        obj = objects.get(frame_idx)
        if frame is None or obj is None:
            raise RuntimeError(f'optical-flow bridge lacks frame/object row {frame_idx}')
        appearance = obj.get('appearance_observation')
        if isinstance(appearance, dict):
            if appearance.get('rgb_tracking_pose_evidence_eligible') is not True or appearance.get('metric_depth_pose_eligible') is not False or (not isinstance(appearance.get('ownership_contract'), dict)) or (appearance['ownership_contract'].get('mano_subtraction_completed') is not True) or (appearance['ownership_contract'].get('bbox_subtraction_used') is not False) or (str(obj.get('mask_semantics') or '') != 'projected_mano_subtracted_object_owned_appearance_mask'):
                raise RuntimeError(f'optical-flow bridge frame {frame_idx} has malformed appearance-only ownership contract')
        else:
            geom = obj.get('visible_geometry_candidate') if isinstance(obj.get('visible_geometry_candidate'), dict) else {}
            if geom.get('rigid_pose_observation_eligible') is not True or not isinstance(geom.get('object_surface_ownership_filter'), dict):
                raise RuntimeError(f'optical-flow bridge frame {frame_idx} lacks accepted metric or explicit appearance ownership evidence')
        raw_path = Path(str(frame.get('raw_frame_path') or ''))
        mask_path = Path(str(obj.get('mask_path') or ''))
        if not raw_path.is_file() or not mask_path.is_file():
            raise RuntimeError(f'optical-flow bridge frame {frame_idx} lacks raw image/object-owned mask')
        current_gray = read_gray(raw_path)
        if current_gray.shape != previous_gray.shape:
            raise RuntimeError('optical-flow bridge raw raster size changed within one clip')
        (height, width) = current_gray.shape
        (next_uv, forward_status, _) = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_uv, None, winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01), minEigThreshold=1e-05)
        if next_uv is None or forward_status is None:
            raise RuntimeError(f'forward optical flow failed at frame {frame_idx}')
        (reverse_uv, reverse_status, _) = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, next_uv, None, winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01), minEigThreshold=1e-05)
        if reverse_uv is None or reverse_status is None:
            raise RuntimeError(f'reverse optical flow failed at frame {frame_idx}')
        next_points = next_uv[:, 0]
        fb_error = np.linalg.norm(reverse_uv[:, 0] - previous_uv[:, 0], axis=1)
        good = (forward_status[:, 0] > 0) & (reverse_status[:, 0] > 0) & (fb_error <= 1.5) & (next_points[:, 0] >= 0.0) & (next_points[:, 0] < width) & (next_points[:, 1] >= 0.0) & (next_points[:, 1] < height)
        owned_mask = read_mask_at_size(mask_path, width, height)
        rounded = np.rint(next_points).astype(np.int64)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        good &= owned_mask[rounded[:, 1], rounded[:, 0]]
        current_uv = next_points[good]
        active_canonical = active_canonical[good]
        flow_trace.append({'frame_idx': int(frame_idx), 'input_track_count': int(len(next_points)), 'accepted_track_count': int(len(current_uv)), 'forward_backward_error_px': numeric_summary(fb_error[good]), 'object_owned_mask': str(mask_path)})
        if len(current_uv) < min_tracked_points:
            raise RuntimeError(f'optical-flow bridge retained {len(current_uv)} < {min_tracked_points} tracks at frame {frame_idx}')
        previous_uv = current_uv.astype(np.float32).reshape(-1, 1, 2)
        previous_gray = current_gray
    target_frame = frames[target_idx]
    (target_height, target_width) = previous_gray.shape
    (world_camera, intrinsics, raster_contract) = frame_camera_for_raster(target_frame, target_width, target_height)
    (success, rotation_vector, translation_vector, inlier_indices) = cv2.solvePnPRansac(active_canonical.astype(np.float64), current_uv.astype(np.float64), intrinsics, None, flags=cv2.SOLVEPNP_EPNP, iterationsCount=300, reprojectionError=4.0, confidence=0.999)
    if not success or inlier_indices is None:
        raise RuntimeError(f'calibrated optical-flow PnP failed at bridge frame {target_idx}')
    inliers = np.asarray(inlier_indices[:, 0], dtype=np.int64)
    if len(inliers) >= 6:
        (rotation_vector, translation_vector) = cv2.solvePnPRefineLM(active_canonical[inliers].astype(np.float64), current_uv[inliers].astype(np.float64), intrinsics, None, rotation_vector, translation_vector)
    for _ in range(2):
        (all_projected, _) = cv2.projectPoints(active_canonical.astype(np.float64), rotation_vector, translation_vector, intrinsics, None)
        all_reprojection = np.linalg.norm(all_projected[:, 0] - current_uv, axis=1)
        expanded_inliers = np.flatnonzero(all_reprojection <= 4.0)
        if len(expanded_inliers) < max(6, len(inliers)):
            break
        inliers = expanded_inliers.astype(np.int64)
        (rotation_vector, translation_vector) = cv2.solvePnPRefineLM(active_canonical[inliers].astype(np.float64), current_uv[inliers].astype(np.float64), intrinsics, None, rotation_vector, translation_vector)
    rotation_camera_from_object = cv2.Rodrigues(rotation_vector)[0]
    translation_camera = np.asarray(translation_vector, dtype=float).reshape(3)
    rotation_world_from_object = world_camera[:3, :3] @ rotation_camera_from_object
    translation_world = world_camera[:3, :3] @ translation_camera + world_camera[:3, 3]
    (projected, _) = cv2.projectPoints(active_canonical[inliers].astype(np.float64), rotation_vector, translation_vector, intrinsics, None)
    reprojection = np.linalg.norm(projected[:, 0] - current_uv[inliers], axis=1)
    camera_inlier_points = apply_pose(active_canonical[inliers], rotation_camera_from_object, translation_camera)
    positive_depth_fraction = float(np.mean(camera_inlier_points[:, 2] > 0.0))
    inlier_fraction = float(len(inliers) / max(1, len(current_uv)))
    reprojection_median = float(np.median(reprojection))
    reprojection_p95 = float(np.percentile(reprojection, 95))
    observability = rotation_observability_summary(active_canonical[inliers])
    quality_passed = bool(len(current_uv) >= min_tracked_points and len(inliers) >= min_pnp_inliers and (inlier_fraction >= min_inlier_fraction) and (reprojection_median <= max_reprojection_median_px) and (reprojection_p95 <= max_reprojection_p95_px) and (positive_depth_fraction >= 0.98) and np.all(np.isfinite(rotation_world_from_object)) and np.all(np.isfinite(translation_world)) and (abs(np.linalg.det(rotation_world_from_object) - 1.0) <= 1e-05))
    report = {'method': 'object_owned_rgb_forward_backward_lk_plus_calibrated_pnp', 'source_frame_idx': int(source_idx), 'target_frame_idx': int(target_idx), 'direction': 'forward' if direction > 0 else 'backward', 'seed': seed_report, 'flow_trace': flow_trace, 'target_camera_raster_contract': raster_contract, 'target_track_count': int(len(current_uv)), 'pnp_inlier_count': int(len(inliers)), 'pnp_inlier_fraction': inlier_fraction, 'reprojection_error_px': numeric_summary(reprojection), 'positive_camera_depth_fraction': positive_depth_fraction, 'rotation_observability': observability, 'pose_evidence_canonical_points_m': active_canonical[inliers].astype(float).tolist(), 'pose_evidence_point_provenance': "Accepted metric surfels from the source depth-owned frame, converted to the shared anchor canonical frame, then tracked to the target object's MANO-subtracted RGB support; no generated point is included.", 'quality_thresholds': {'minimum_tracked_points': int(min_tracked_points), 'minimum_pnp_inliers': int(min_pnp_inliers), 'minimum_pnp_inlier_fraction': float(min_inlier_fraction), 'maximum_reprojection_median_px': float(max_reprojection_median_px), 'maximum_reprojection_p95_px': float(max_reprojection_p95_px), 'minimum_positive_camera_depth_fraction': 0.98, 'rotation_observability_diagnostic_score_threshold': 0.02, 'per_bridge_observability_hard_gate': False, 'timeline_minimum_observable_fraction_remains_fixed': 0.8}, 'rotation_observability_policy': 'A sparse bridge may remain explicitly rotation-underobservable; the downstream fixed score>=0.02 and observable-fraction>=0.80 timeline gate remains authoritative and cannot be lowered.', 'quality_passed': quality_passed, 'claim_scope': 'Direct prediction-side RGB pose observation: accepted anchor-frame metric surfels are tracked only through MANO-subtracted object-owned image support and solved with the exact calibrated pinhole camera.'}
    if not quality_passed:
        raise RuntimeError(f'optical-flow PnP bridge failed closed: {report}')
    return (rotation_world_from_object, translation_world, report, observability)
