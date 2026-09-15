#!/usr/bin/env python3
"""Evaluate recorded HOT3D annotations; no inference, no GPU, no pipeline imports."""
import argparse
import json
import os
from pathlib import Path
import sys

# Applied before torch can be imported; this executable never requests a GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hot3d_eval.alignment import first_frame_registration, transform_points, validate_pose
from hot3d_eval.gt_loader import GroundTruth, read_json, resolve_source
from hot3d_eval.mano_fk import ManoFK, find_models, sha256
from hot3d_eval.metrics_hand import BetaDriftError, evaluate_hand, constant_betas, mean_metric, metric
from hot3d_eval.metrics_camera import evaluate_camera
from hot3d_eval.report import render_report


class ProjectionGateError(ValueError):
    def __init__(self, annotation, gate):
        super().__init__('GT projection gate failed: ' + '; '.join(gate['failures']))
        self.diagnostic = {'annotation': str(annotation), 'gate': gate}


def legacy_state_joints(path, annotation):
    """Read-only recovery for annotations entirely predating joints_world export.

    A present key (even null) marks the new schema and disables this fallback.
    Archive frame IDs and camera poses must agree with the JSON exactly enough
    to identify the same fitted state; no rendering or FK of predictions occurs.
    """
    frames = annotation['frames']
    if any('joints_world' in f.get('hands', {}).get(side, {}) for f in frames for side in ['left', 'right']):
        return None, {'description': 'JSON joints_world'}
    object_id = annotation['object_id']
    if not isinstance(object_id, str) or Path(object_id).name != object_id:
        raise ValueError('Invalid object_id for state archive lookup')
    archive_path = Path(path).parent.parent / 'measurements/scene_fit' / object_id / 'demo_state.npz'
    if not archive_path.is_file():
        return None, {'description': 'JSON (legacy joints_world absent; state archive unavailable)'}
    with np.load(archive_path, allow_pickle=False) as archive:
        ids = np.asarray(archive['frame_idx'])
        if len(np.unique(ids)) != len(ids):
            raise ValueError('State archive frame_idx contains duplicates')
        indices = {int(idx): i for i, idx in enumerate(ids)}
        chosen = [indices[int(f['frame_idx'])] for f in frames]
        cameras = np.asarray(archive['T_world_camera'])[chosen]
        expected = np.asarray([f['camera']['T_world_camera'] for f in frames])
        if not np.allclose(cameras, expected, atol=1e-6, rtol=0):
            raise ValueError('State archive camera poses disagree with annotation; refusing stale joints')
        joints = {side: np.asarray(archive[side + '_joints_world'], float)[chosen].copy() for side in ['left', 'right']}
        for side in joints:
            if joints[side].shape != (len(frames), 21, 3):
                raise ValueError(f'State archive {side} joints must be Nx21x3')
    return joints, {'description': 'state archive (annotation predates joints_world field)',
                    'archive': str(archive_path), 'sha256': sha256(archive_path)}


def registered_predictions(frames, A, legacy_joints=None):
    cameras, hands, objects = [], {'left': [], 'right': []}, []
    for i, f in enumerate(frames):
        cameras.append(A @ validate_pose(f['camera']['T_world_camera']))
        for side in hands:
            points = legacy_joints[side][i] if legacy_joints is not None else f.get('hands', {}).get(side, {}).get('joints_world')
            if points is None:
                hands[side].append(np.full((21, 3), np.nan))
            else:
                points = np.asarray(points, float)
                if points.shape != (21, 3) or not np.isfinite(points).all():
                    raise ValueError(f'{side} frame {f["frame_idx"]}: joints_world must be finite 21x3 or null')
                hands[side].append(transform_points(A, points))
        visual = f.get('object', {}).get('visual_fitted_pose')
        if visual is None:
            objects.append(None)
        else:
            T = np.eye(4)
            T[:3, :3] = visual['rotation_world_from_completed_canonical_matrix']
            T[:3, 3] = visual['translation_world_m']
            objects.append(A @ validate_pose(T))
    return np.asarray(cameras), {side: np.asarray(points) for side, points in hands.items()}, objects


def evaluate_annotation(path, args, fk):
    annotation = read_json(path)
    frames = annotation['frames']
    source = resolve_source(path, annotation, args.source_video, args.sensor_contract)
    gt = GroundTruth(args.dataset_root, source, frames, fk)
    gate = gt.projection_gate()
    for sample in gate['samples']:
        print(f'GATE {source["sequence"]} {sample["side"]} frame={sample["frame_idx"]} '
              f'inside={sample["inside"]}/{sample["total"]} offset_px={sample["mean_offset_from_box_center_px"]}', flush=True)
    if not gate['passed']:
        raise ProjectionGateError(path, gate)
    A = first_frame_registration(gt.cameras[0], frames[0]['camera']['T_world_camera'])
    legacy_joints, joints_source = legacy_state_joints(path, annotation)
    cameras, hands, objects = registered_predictions(frames, A, legacy_joints)
    print(f'joints source: {joints_source["description"]}', flush=True)
    result = {'annotation': str(Path(path).resolve()), 'annotation_sha256': sha256(path),
              'sequence': source['sequence'], 'source': source, 'source_indices': gt.source_indices,
              'timestamps': gt.timestamps, 'camera_sampling': gt.camera_sampling, 'gate': gate, 'A': A.tolist(),
              'registered_object_count': sum(o is not None for o in objects),
              'mano': fk.provenance, 'joints_source': joints_source,
              'hawor_provenance': 'hawor' in json.dumps(annotation.get('hand_model_provenance', {})).lower(),
              'left_shapedirs_fixed': fk.left_shapedirs_fixed, 'hands': {}, 'coverage': {}}
    for side in ['left', 'right']:
        pred_valid = np.isfinite(hands[side]).all(axis=(1, 2))
        valid = pred_valid & gt.valid[side]
        n = len(frames)
        missing = int((~pred_valid).sum())
        if missing:
            print(f'{source["sequence"]} {side}: 预测缺 joints_world {missing}/{n}; excluded explicitly', file=sys.stderr)
        result['coverage'][side] = {'valid': int(valid.sum()), 'total': n, 'gt_available': int(gt.valid[side].sum()), 'prediction_missing': missing}
        metrics = evaluate_hand(hands[side], gt.hands[side], valid, gt.source_indices)
        diagnostic_beta = None
        try:
            beta, reason = constant_betas(frames, side, valid)
        except BetaDriftError as e:
            beta, reason = None, str(e)
            print(f'ERROR MPVPE(official) {reason}; other metrics continue', file=sys.stderr)
            diagnostic_beta = np.mean([frames[i]['hands'][side]['parameters']['source_betas']
                                       for i in np.flatnonzero(valid)], axis=0)
        if beta is None:
            metrics['MPVPE_mm'] = metric(None, 0, n * 778, reason)
        else:
            vp = fk.shape_vertices(side, beta)
            vg = fk.shape_vertices(side, [gt.records[side][i]['betas'] for i in np.flatnonzero(valid)])
            metrics['MPVPE_mm'] = mean_metric(np.linalg.norm(vg - vp, axis=-1), n * 778, 1000)
        if diagnostic_beta is not None:
            vp = fk.shape_vertices(side, diagnostic_beta)
            vg = fk.shape_vertices(side, [gt.records[side][i]['betas'] for i in np.flatnonzero(valid)])
            metrics['MPVPE_mean_beta_diagnostic_mm'] = mean_metric(
                np.linalg.norm(vg - vp, axis=-1), n * 778, 1000,
                f'non-protocol: prediction lacks constant shape; mean-β over {valid.sum()} frames; {reason}')
        if not valid.any():
            for m in metrics.values():
                m['reason'] = '; '.join(filter(None, [m['reason'], '预测缺 joints_world or GT unavailable; zero valid samples']))
        result['hands'][side] = metrics
    result['camera'] = evaluate_camera(cameras, gt.cameras, gt.source_indices)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--annotations', nargs='+', type=Path)
    inputs.add_argument('--manifest', type=Path, help='UTF-8, one annotation path per line, # comments')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=Path('/mnt/nas-106/hot3d/dataset'))
    parser.add_argument('--source-video', type=Path, help='Single-input source record override')
    parser.add_argument('--sensor-contract', type=Path, help='Single-input sensor record override')
    parser.add_argument('--model-config', type=Path, help='Runtime model_envs.json (mano_models / hawor_root)')
    parser.add_argument('--mano-left', type=Path, help='Explicit licensed MANO_LEFT.pkl')
    parser.add_argument('--mano-right', type=Path, help='Explicit licensed MANO_RIGHT.pkl')
    args = parser.parse_args(argv)
    if args.manifest:
        args.annotations = []
        for line in args.manifest.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                p = Path(line)
                args.annotations.append(p if p.is_absolute() else args.manifest.parent / p)
    if not args.annotations:
        parser.error('Input list is empty')
    if len({p.resolve() for p in args.annotations}) != len(args.annotations):
        parser.error('Duplicate annotation paths would double-count samples')
    if len(args.annotations) != 1 and (args.source_video or args.sensor_contract):
        parser.error('Source overrides require exactly one annotation')
    if args.output.resolve() in {p.resolve() for p in args.annotations}:
        parser.error('Output must not overwrite an input annotation')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        fk = ManoFK(find_models(args.model_config, args.mano_left, args.mano_right))
        results = [evaluate_annotation(path, args, fk) for path in args.annotations]
        report = render_report(results)
    except ProjectionGateError as e:
        diagnostic = args.output.with_suffix(args.output.suffix + '.gate-failure.json')
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.write_text(json.dumps(e.diagnostic, indent=2) + '\n')
        print(json.dumps(e.diagnostic, indent=2), file=sys.stderr)
        print(f'ABORT: {e}. No report written. Diagnostics: {diagnostic}', file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError, StopIteration) as e:
        print(f'ERROR: {type(e).__name__}: {e}. No report written.', file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f'Report: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
