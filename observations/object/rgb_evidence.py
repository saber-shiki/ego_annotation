"""Source-world RGB measurement serialization, independent of pose candidates."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from utils.state import sha
from utils.geometry import validate_transforms

SCHEMA = 'ego.annotation.rgb_source_measurements.v1'
EXTENSION = 'rgb_source_measurements'
MOTION_CONVENTION = 'X_target_world = D @ X_source_world + d; D = R_target @ R_source.T; d = t_target - D @ t_source'


def validate_source_frames(frames, selected, depth_pos, depths, depth_intrinsics, source_wh):
    validate_transforms([frames[i]['camera']['T_world_camera_metric'] for i in selected])
    for i in selected:
        frame = frames[i]
        if type(frame['frame_idx']) is not int or (frame['source_width'], frame['source_height']) != source_wh:
            raise ValueError('RGB source frame identity/image plane mismatch')
        K = np.asarray(frame['camera']['intrinsics_fx_fy_cx_cy'], float)
        if K.shape != (4,) or not np.isfinite(K).all() or np.any(K[:2] <= 0):
            raise ValueError('invalid RGB camera intrinsics')
        if depth_pos[i] < len(depths) and not np.allclose(K, depth_intrinsics[depth_pos[i]], atol=1e-8, rtol=0):
            raise ValueError('RGB annotation/depth intrinsics mismatch')
    times = np.asarray([frames[i]['time_s'] for i in selected], float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('invalid RGB timeline timestamps')


def write_rgb_source_evidence(*, args, frames, selected, rows, motions, evidence,
                              image_planes, manifest_path, adapter_path,
                              source_rejections, planning_rejections, metric_eligibility):
    """Only supported edges have numeric rows; failed attempts are JSON diagnostics.

    The NPZ embeds the complete canonical report (before its NPZ hash is added).
    The manifest then binds both files, preventing cross-producer pairing.
    """
    def receipt(path):
        path = Path(path).resolve()
        return {'path': str(path), 'sha256': sha(path)}

    inputs = {key: receipt(path) for key, path in {
        'annotations': args.annotations, 'depth_npz': args.depth_npz,
        'raw_frame_manifest': manifest_path, 'p09_adapter_report': adapter_path,
        'assets_manifest': args.assets_manifest,
    }.items()}
    for idx, plane in image_planes.items():
        inputs[f'rgb:{idx}'] = receipt(plane['rgb_path'])
        obj = next((o for o in frames[idx]['objects'] if o['object_id'] == args.object_id), None)
        if obj is None or plane['owned_mask_size_wh'] is None:
            continue
        inputs[f'mask:{idx}'] = receipt(obj['mask_path'])
        eligibility = obj.get('visible_geometry_candidate', {}).get('native_depth_metric_eligibility')
        if eligibility:
            inputs[f'eligibility_raster:{idx}'] = receipt(eligibility['raster'])
    if metric_eligibility is not None:
        inputs['metric_eligibility_manifest'] = receipt(metric_eligibility.path)
    accepted = [row for row in rows if row['status'] == 'accepted']
    report = {
        'schema': SCHEMA, 'object_id': args.object_id,
        'frame_ids': selected, 'time_s': [float(frames[i]['time_s']) for i in selected],
        'inputs': inputs, 'motion_convention': MOTION_CONVENTION,
        'point_frame': 'fixed_source_world_m', 'pixel_frame': 'original_owned_mask_half_pixel_resize',
        'prediction_side_only': True, 'generated_geometry_consumed': False,
        'legacy_optimizer_required': False, 'contains_optimized_state': False,
        'uncertainty': 'PnP is an uncertain RGB/depth measurement, not a target metric surface or solved trajectory',
        'image_planes': {str(k): v for k, v in image_planes.items()},
        'rows': rows, 'accepted_pairs': [[r['source_frame_idx'], r['target_frame_idx']] for r in accepted],
        'source_rejections': [{'frame_idx': i, 'reason': reason} for i, reason in source_rejections.items() if reason],
        'pair_planning_rejections': planning_rejections,
        'parameters': {k: getattr(args, k) for k in ('keyframe_seconds', 'source_gap_seconds',
            'keyframe_neighbor_span', 'max_keypoints', 'min_confidence', 'min_matches',
            'pnp_iterations', 'reprojection_threshold_px', 'min_inliers', 'min_inlier_fraction',
            'max_reprojection_median_px', 'max_edge_rotation_deg', 'pnp_refinement')},
        'measurement_refinement': 'local_PnP_LM_not_trajectory_solver' if args.pnp_refinement == 'lm' else 'none_after_RANSAC_EPNP',
        'mechanical_admissibility': {'minimum_input_matches': int(args.min_matches),
            'minimum_pnp_inliers': 4, 'finite_world_transform_required': True,
            'positive_inlier_camera_depth_required': True},
        'quality_policy': 'retain_noisy_mechanically_supported_measurements; diagnostic_reference_thresholds_are_not_gates; success_is_not_confidence',
        'point_weights_role': 'uncalibrated_reprojection_heuristic_not_probability',
        'reprojection_diagnostic_plane': 'manifest_rgb',
    }
    arrays = {}
    points = {'source_world': (3,), 'source_uv': (2,), 'target_uv': (2,),
              'weights': (), 'matching_scores': (), 'inlier_indices': ()}
    per_edge = {'source_camera': (4, 4), 'target_camera': (4, 4),
                'source_intrinsics': (4,), 'target_intrinsics': (4,),
                'source_size_wh': (2,), 'target_size_wh': (2,)}
    values = [evidence[(r['source_frame_idx'], r['target_frame_idx'])] for r in accepted]
    arrays['offsets'] = np.asarray([0, *np.cumsum([len(v['source_world']) for v in values])], dtype=np.int64)
    for key, shape in points.items():
        dtype = np.int64 if key == 'inlier_indices' else np.float64
        arrays[key] = np.concatenate([v[key] for v in values]).astype(dtype) if values else np.empty((0, *shape), dtype=dtype)
    for key, shape in per_edge.items():
        dtype = np.int64 if key.endswith('size_wh') else np.float64
        arrays[key] = np.asarray([v[key] for v in values], dtype=dtype).reshape((-1, *shape))
    for end in ('source', 'target'):
        arrays[end + '_frame_idx'] = np.asarray([r[end + '_frame_idx'] for r in accepted], dtype=np.int64)
        arrays[end + '_time_s'] = np.asarray([frames[r[end + '_frame_idx']]['time_s'] for r in accepted], dtype=np.float64)
    pairs = [(r['source_frame_idx'], r['target_frame_idx']) for r in accepted]
    arrays['rotation_source_to_target_world'] = np.asarray([motions[p][0] for p in pairs], float).reshape(-1, 3, 3)
    arrays['translation_source_to_target_world_m'] = np.asarray([motions[p][1] for p in pairs], float).reshape(-1, 3)
    arrays['report_metadata'] = np.asarray(json.dumps(report, sort_keys=True, allow_nan=False))
    for path in (args.output_npz, args.output_json):
        if path.exists():
            raise FileExistsError(f'RGB source evidence is create-only: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_npz.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    report['npz_sha256'] = sha(args.output_npz)
    with args.output_json.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report
