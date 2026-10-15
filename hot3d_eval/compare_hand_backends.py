"""Read-only hand backend comparison; never a prediction dependency.

GT joints use exact-timestamp official MANO FK, validated against cached vertices.
All arrays are metres; joint order is wrist then thumb/index/middle/ring/pinky.
"""
from __future__ import annotations
import argparse
import csv
import inspect
import json
from pathlib import Path
import numpy as np
from .alignment import fit_points, transform_points
from .mano_fk import ManoFK, OURS21_TO_OFFICIAL19, sha256


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def apply(T, points):
    return np.einsum('...ij,...kj->...ki', T[..., :3, :3], points) + T[..., None, :3, 3]


def stats(values):
    x = np.asarray(values, float).reshape(-1)
    x = x[np.isfinite(x)]
    return dict(count=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.percentile(x, 95))) if x.size else dict(count=0, mean=None, median=None, p95=None)


def project(points, K):
    return points[..., :2] / points[..., 2:] * K[..., None, :2] + K[..., None, 2:]


def score(pred, truth, valid, ids, K, wh):
    idx = np.asarray(OURS21_TO_OFFICIAL19)
    p, g = pred['camera_joints'], truth['camera_joints']
    pw, gw = pred['world_joints'], truth['world_joints']
    e = np.linalg.norm(p[:, idx] - g[:, idx], axis=-1) * 1000
    root = np.linalg.norm((p - p[:, :1])[:, idx] - (g - g[:, :1])[:, idx], axis=-1) * 1000
    positive = (p[:, idx, 2] > 0) & (g[:, idx, 2] > 0)
    uvp, uvg = project(p, K), project(g, K)
    epix = np.linalg.norm(uvp[:, idx] - uvg[:, idx], axis=-1)
    onscreen = ((uvg[:, idx] >= 0) & (uvg[:, idx] < wh)).all(-1)
    metrics = {
        'MPJPE19_camera_mm': stats(e[valid]),
        'MPJPE19_world_mm': stats((np.linalg.norm(pw[:, idx] - gw[:, idx], axis=-1) * 1000)[valid]),
        'wrist_relative_MPJPE19_camera_mm': stats(root[valid]),
        'projection19_px': stats(epix[valid[:, None] & positive]),
        'projection19_GT_in_image_px': stats(epix[valid[:, None] & positive & onscreen]),
        'projection21_px': stats(np.linalg.norm(uvp - uvg, axis=-1)[valid[:, None] & (p[..., 2] > 0) & (g[..., 2] > 0)]),
        'wrist_camera_mm': stats((np.linalg.norm(p[:, 0] - g[:, 0], axis=-1) * 1000)[valid]),
        'wrist_camera_depth_bias_mm': stats(((p[:, 0, 2] - g[:, 0, 2]) * 1000)[valid]),
        'posed_MPVPE_camera_mm': stats((np.linalg.norm(pred['camera_vertices'] - truth['camera_vertices'], axis=-1) * 1000)[valid]),
        'valid_hand_frames': int(valid.sum()), 'total_frames': len(valid),
    }
    if 'native_projection_K' in pred:
        # Native WiLoR image prediction uses its own assumed focal. This is a
        # different diagnostic from projecting its metric state with source K.
        native_uv = project(pred['native_camera_joints'], pred['native_projection_K'])
        native_error = np.linalg.norm(native_uv[:, idx] - uvg[:, idx], axis=-1)
        native_positive = pred['native_camera_joints'][:, idx, 2] > 0
        metrics['native_network_projection19_px'] = stats(native_error[valid[:, None] & native_positive & (g[:, idx, 2] > 0)])
    pa = []
    for a, b in zip(p[valid], g[valid]):
        try:
            pa.extend(np.linalg.norm(transform_points(fit_points(a, b, True), a) - b, axis=-1) * 1000)
        except ValueError:
            pass
    metrics['PA_MPJPE21_mm'] = stats(pa)
    consecutive = np.diff(ids) == 1
    triplets = valid[:-2] & valid[1:-1] & valid[2:] & consecutive[:-1] & consecutive[1:]
    ap, ag = np.diff(pw, n=2, axis=0) * 900, np.diff(gw, n=2, axis=0) * 900
    metrics['acceleration_error_world_m_s2'] = stats(np.linalg.norm(ap[triplets] - ag[triplets], axis=-1))
    metrics['pred_acceleration_world_m_s2'] = stats(np.linalg.norm(ap[triplets], axis=-1))
    metrics['valid_consecutive_triplets'] = int(triplets.sum())
    rows = []
    for i in range(len(ids)):
        row = dict(frame_idx=int(ids[i]), scored=bool(valid[i]))
        if valid[i]:
            row.update(MPJPE19_camera_mm=float(e[i].mean()), wrist_relative_MPJPE19_camera_mm=float(root[i].mean()),
                       projection19_px=float(epix[i, positive[i]].mean()) if positive[i].any() else None,
                       wrist_depth_bias_mm=float((p[i, 0, 2] - g[i, 0, 2]) * 1000))
        rows.append(row)
    return metrics, rows


def main(args):
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = read(args.source_bundle)
    root = args.source_bundle.parent
    source_path = lambda name: root / bundle['sources'][name]['path']
    hp = source_path('mano_archive')
    if sha256(hp) != bundle['sources']['mano_archive']['sha256']:
        raise ValueError('source HaWoR archive hash changed')
    h, gt = load(hp), load(args.gt_cache)
    annotations = read(source_path('annotations'))['frames']
    frames = sorted(annotations, key=lambda f: f['frame_idx'])
    ids = np.array([f['frame_idx'] for f in frames])
    C = np.array([f['camera'].get('T_world_camera_metric', f['camera'].get('T_world_camera')) for f in frames])
    K = np.array([f['camera']['intrinsics_fx_fy_cx_cy'] for f in frames])
    if not np.array_equal(ids, gt['frame_idx']) or not np.array_equal(ids, h['frame_idx']):
        raise ValueError('frame IDs differ')
    if not np.allclose(K, gt['intrinsics_fx_fy_cx_cy'], atol=1e-9, rtol=0):
        raise ValueError('GT and prediction image planes differ')
    if not np.array_equal(gt['source_frame_idx'], np.arange(args.source_start, args.source_start + len(ids))):
        raise ValueError('source frame range mismatch')
    D = gt['T_world_from_camera']
    A = D[0] @ np.linalg.inv(C[0])  # fixed first-camera SE3, never scale fit
    paths = {'left': args.mano_left, 'right': args.mano_right}
    fk = ManoFK(paths)
    timestamps = set(map(int, gt['timestamp_ns']))
    records = {}
    with (args.gt_sequence / 'mano_hand_pose_trajectory.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if int(r['timestamp_ns']) in timestamps:
                records[int(r['timestamp_ns'])] = r['hand_poses']
    quality = np.ones(len(ids), bool)
    masks = {}
    for name in ['mask_hand_pose_available', 'mask_headset_pose_available', 'mask_qa_pass']:
        with (args.gt_sequence / 'masks' / (name + '.csv')).open() as f:
            mask = {int(r['timestamp[ns]']): r['mask'].lower() == 'true' for r in csv.DictReader(f) if r['stream_id'] == '214-1'}
        masks[name] = np.array([mask.get(int(t), False) for t in gt['timestamp_ns']])
        quality &= masks[name]
    candidates = {}
    for name, path in args.candidate:
        path = Path(path)
        completion_path = path.with_suffix('.completed.json')
        if not completion_path.is_file():
            raise ValueError(f'candidate not frozen: {path}')
        if sha256(path) != read(completion_path)['output_sha256']:
            raise ValueError(f'candidate archive differs from completion record: {path}')
        q = load(path)
        if not np.array_equal(q['frame_idx'], ids) or not np.allclose(q['T_world_camera'], C, atol=1e-9, rtol=0):
            raise ValueError('candidate camera/timeline changed')
        if not np.allclose(q['intrinsics'], K, atol=1e-9, rtol=0):
            raise ValueError('candidate image plane differs')
        candidates[name] = q
    result = dict(frames=len(ids), GT_quality_frames=int(quality.sum()), A=A.tolist(),
                  input_hashes={str(p): sha256(p) for p in [args.source_bundle, hp, args.gt_cache, *paths.values(), *[Path(p) for _, p in args.candidate]]},
                  definitions={'2D': 'Official19 projected MANO GT, not manually annotated independent 2D landmarks; positive depth; source pinhole pixels.',
                               'world': 'Single first-camera SE3, scale=1; same source camera for all backends.',
                               'camera': 'Each hand in its own matched camera frame, no GT scale/translation fitting.',
                               'coverage': 'availability + headset + QA masks only; no prediction-dependent GT selection; paired intersection also reported.',
                               'acceleration': 'World second differences x30^2 on consecutive valid triplets only; m/s^2.',
                               'MPVPE': 'Posed full778 corresponding vertices, NOT official rest-shape MPVPE.'},
                  FK_provenance=fk.provenance, hands={})
    cache = dict(frame_idx=ids, gt_camera=D, pred_camera=A @ C, K=K)
    all_rows = []
    for side, hid in [('left', '0'), ('right', '1')]:
        si = int(np.flatnonzero(gt['mano_sides'] == side)[0])
        valid_gt = np.array([hid in records.get(int(t), {}) for t in gt['timestamp_ns']]) & gt['mano_available'][:, si] & quality
        good = np.flatnonzero(valid_gt)
        rr = [records[int(gt['timestamp_ns'][i])][hid] for i in good]
        gj = np.full((len(ids), 21, 3), np.nan)
        gj[good] = fk.forward(side, rr)
        torch = fk.torch
        from scipy.spatial.transform import Rotation
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        with torch.no_grad():
            v = fk.models[side](betas=tensor([r['betas'] for r in rr]), hand_pose=tensor([r['pose'] for r in rr]),
                global_orient=tensor([Rotation.from_quat(np.asarray(r['wrist_xform']['q_wxyz'])[[1,2,3,0]]).as_rotvec() for r in rr]),
                transl=tensor([r['wrist_xform']['t_xyz'] for r in rr]), return_verts=True).vertices.numpy()
        replay_error = float(np.max(np.linalg.norm(v - gt['mano_world_vertices_m'][good, si], axis=-1)))
        if replay_error > 1e-5:
            raise ValueError(f'{side} GT FK/cache mismatch {replay_error}m')
        truth = dict(world_joints=gj, camera_joints=apply(np.linalg.inv(D), gj),
                     camera_vertices=apply(np.linalg.inv(D), gt['mano_world_vertices_m'][:, si]))
        models = {'HaWoR': dict(world_joints=apply(A, h[side + '_joints_world_m']),
                               camera_joints=apply(np.linalg.inv(C), h[side + '_joints_world_m']),
                               camera_vertices=apply(np.linalg.inv(C), h[side + '_vertices_world_m']), valid=h[side + '_valid'].astype(bool))}
        for name, q in candidates.items():
            models[name] = dict(camera_joints=q[side + '_joints_camera_m'], camera_vertices=q[side + '_vertices_camera_m'],
                                world_joints=apply(A @ C, q[side + '_joints_camera_m']), valid=q[side + '_valid'].astype(bool))
            if side + '_default_joints_camera_m' in q:
                native_t = q[side + '_default_camera_translation_m']
                calibrated_t = q[side + '_camera_translation_m']
                if 'native_projection_intrinsics' in q:
                    native_K = q['native_projection_intrinsics']
                else:
                    native_f = native_t[:, 2] / calibrated_t[:, 2] * K[:, 0]
                    native_K = np.column_stack([native_f, native_f, np.broadcast_to(q['image_size_wh'] / 2, (len(ids), 2))])
                models[name]['native_projection_K'] = native_K
                models[name]['native_camera_joints'] = q[side + '_default_joints_camera_m']
                models[name + '_default_focal'] = dict(camera_joints=q[side + '_default_joints_camera_m'],
                    camera_vertices=q[side + '_default_vertices_camera_m'],
                    world_joints=apply(A @ C, q[side + '_default_joints_camera_m']), valid=q[side + '_valid'].astype(bool))
        common = valid_gt.copy()
        for p in models.values():
            p['valid'] &= np.isfinite(p['camera_joints']).all(axis=(1,2))
            common &= p['valid']
        section = dict(GT_valid_hand_frames=int(valid_gt.sum()), common_hand_frames=int(common.sum()), GT_FK_max_vertex_error_m=replay_error, backends={})
        for name, p in models.items():
            own = valid_gt & p['valid']
            full, rows = score(p, truth, own, ids, K, gt['image_size_wh'])
            paired, _ = score(p, truth, common, ids, K, gt['image_size_wh'])
            full['GT_hand_frame_detection_recall'] = float(own.sum() / valid_gt.sum()) if valid_gt.any() else None
            if name == 'HaWoR':
                full['same_frame_detected'] = int((own & h[side + '_detected_same_frame'].astype(bool)).sum())
            section['backends'][name] = dict(own_coverage=full, paired=paired)
            for row in rows:
                row.update(side=side, backend=name)
            all_rows.extend(rows)
            cache[name + '_' + side + '_camera_joints'] = p['camera_joints']
            cache[name + '_' + side + '_world_joints'] = p['world_joints']
            cache[name + '_' + side + '_valid'] = own
        cache['GT_' + side + '_world_joints'] = gj
        cache['GT_' + side + '_camera_joints'] = truth['camera_joints']
        result['hands'][side] = section
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output / 'frame_metrics.json').write_text(json.dumps(all_rows, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'comparison.npz', **cache)
    for p, digest in result['input_hashes'].items():
        if sha256(p) != digest:
            raise ValueError('evaluation mutated input: ' + p)
    print(json.dumps(result['hands'], indent=2))


def parser():
    p = argparse.ArgumentParser(__doc__)
    for name in ['source-bundle','gt-cache','gt-sequence','mano-left','mano-right','output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-start', type=int, required=True)
    p.add_argument('--candidate', nargs=2, action='append', default=[], metavar=('NAME','ARCHIVE'))
    return p


if __name__ == '__main__':
    main(parser().parse_args())
