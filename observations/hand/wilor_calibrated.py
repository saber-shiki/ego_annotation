"""Calibrated WiLoR sequence measurement, separate from the default RL producer.

Reuses native predictions, not GT or a hand-size heuristic. Left hands are a
reflected right MANO with explicit parameters; this is not the RL left FK schema.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def calibrated_translation(default_t, default_focal, size_wh, intrinsics):
    """Exact change of cam_crop_to_full's full-frame pinhole convention.

    Anatomy and network camera scale/offset are unchanged. A single weak scale
    supports square pixels only; unequal fx/fy is not silently approximated.
    """
    t = np.asarray(default_t, float).copy()
    k = np.asarray(intrinsics, float)
    if t.shape != (3,) or k.shape != (4,) or not np.isfinite(np.r_[t, k, default_focal]).all():
        raise ValueError('invalid camera conversion values')
    if default_focal <= 0 or k[0] <= 0 or not np.isclose(k[0], k[1], atol=1e-7, rtol=0):
        raise ValueError('WiLoR weak camera requires positive square focal')
    t[:2] += (np.asarray(size_wh) / 2 - k[2:]) * t[2] / default_focal
    t[2] *= k[0] / default_focal
    return t


def project(points, K):
    return points[..., :2] / points[..., 2:] * np.asarray(K)[:2] + np.asarray(K)[2:]


def run(args):
    import cv2
    import torch
    from .wilor import ensure_wilor_assets, load_wilor_backend, open_video, run_wilor_on_frame
    if os.uname().nodename != args.compute_host:
        raise ValueError('compute host differs from explicit target')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available():
        raise ValueError('an explicitly selected server CUDA device is required')
    args.output.mkdir(parents=True, exist_ok=False)
    cap, info = open_video(args.clip)
    if getattr(args, 'camera_contract', None) is not None:
        # Camera-local inference: no fabricated/old world trajectory is supplied.
        from envs.qualification.camera import load_contract, plane_intrinsics
        payload, normalized = load_contract(args.camera_contract)
        k, _ = plane_intrinsics(payload, normalized, plane_name='source_rgb', actual_size_wh=(info.width, info.height))
        ids = np.asarray(normalized['frame_ids'], int)
        C, K = None, np.tile(k, (len(ids), 1))
        camera_input = args.camera_contract
    else:
        with np.load(args.camera_track, allow_pickle=False) as track:
            ids, C, K = track['frame_idx'], track['T_world_camera'], track['intrinsics']
        camera_input = args.camera_track
    ensure_wilor_assets(args.wilor_root, args.wilor_root / 'mano_data/MANO_RIGHT.pkl')
    files = [args.clip, camera_input, args.wilor_root/'pretrained_models/wilor_final.ckpt',
             args.wilor_root/'pretrained_models/detector.pt', args.wilor_root/'pretrained_models/model_config.yaml',
             args.wilor_root/'mano_data/MANO_RIGHT.pkl']
    hashes = {str(p): sha(p) for p in files}
    started = time.monotonic()
    model, cfg, detector, device = load_wilor_backend(args.wilor_root)
    torch.set_num_threads(1)
    if len(ids) != info.frame_count or not np.array_equal(ids, np.arange(info.frame_count)):
        raise ValueError('camera track must cover the complete video')
    if (C is not None and C.shape != (len(ids), 4, 4)) or K.shape != (len(ids), 4):
        raise ValueError('invalid camera track')
    if not np.isclose(info.fps, 30):
        raise ValueError('this measurement protocol requires 30Hz')
    result = dict(frame_idx=ids, intrinsics=K,
                  image_size_wh=np.array([info.width, info.height]), time_s=ids / info.fps)
    if C is not None:
        result['T_world_camera'] = C
    for side in ['left', 'right']:
        for field, shape in [('joints_camera_m',(21,3)), ('vertices_camera_m',(778,3)),
                             ('default_joints_camera_m',(21,3)), ('default_vertices_camera_m',(778,3)),
                             ('mano_global_orient_right',(1,3,3)), ('mano_hand_pose_right',(15,3,3)),
                             ('betas',(10,)), ('camera_translation_m',(3,)), ('default_camera_translation_m',(3,)),
                             ('bbox_xyxy',(4,))]:
            result[side+'_'+field] = np.full((len(ids), *shape), np.nan, np.float32)
        result[side+'_valid'] = np.zeros(len(ids), bool)
        result[side+'_detection_count'] = np.zeros(len(ids), np.int32)
        result[side+'_detector_score'] = np.zeros(len(ids), np.float32)
        result[side+'_reflection'] = np.diag([-1.,1.,1.]) if side == 'left' else np.eye(3)
        result[side+'_faces'] = np.asarray(model.mano.faces)[:, [0,2,1]] if side == 'left' else np.asarray(model.mano.faces)
    max_replay = 0.
    first_frame_s = None
    try:
        for i in ids:
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'premature video end at {i}')
            hands = run_wilor_on_frame(model, cfg, detector, device, frame, args.rescale_factor, args.batch_size)
            for side in ['left', 'right']:
                detections = [h for h in hands if h['side'] == side]
                result[side+'_detection_count'][i] = len(detections)
                if not detections:
                    continue
                h = max(detections, key=lambda d: d['detector_score'])
                local_j, local_v = np.array(h['joints3d_camera']), np.array(h['vertices_camera'])
                native_t = np.array(h['cam_t'])
                t = calibrated_translation(native_t, h['focal_length'], [info.width, info.height], K[i])
                params = h['mano_params']
                # Replay actual full MANO; the left reflection is NOT an SO(3).
                tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
                pp = dict(global_orient=tensor(params['global_orient']).reshape(1,1,3,3),
                          hand_pose=tensor(params['hand_pose']).reshape(1,15,3,3),
                          betas=tensor(params['betas']).reshape(1,10))
                with torch.no_grad():
                    replay = model.mano(**pp, pose2rot=False).vertices[0].cpu().numpy()
                if side == 'left':
                    replay[:,0] *= -1
                error = float(np.linalg.norm(replay - local_v, axis=-1).max())
                max_replay = max(max_replay, error)
                if error > 1e-5:
                    raise ValueError(f'MANO parameter/surface replay failed: {side} {i} {error}')
                for name, a in [('joints_camera_m',local_j+t), ('vertices_camera_m',local_v+t),
                                ('default_joints_camera_m',local_j+native_t), ('default_vertices_camera_m',local_v+native_t),
                                ('mano_global_orient_right',np.asarray(params['global_orient']).reshape(1,3,3)),
                                ('mano_hand_pose_right',np.asarray(params['hand_pose']).reshape(15,3,3)),
                                ('betas',np.asarray(params['betas']).reshape(10)), ('camera_translation_m',t),
                                ('default_camera_translation_m',native_t), ('bbox_xyxy',h['bbox_xyxy'])]:
                    result[side+'_'+name][i] = a
                result[side+'_detector_score'][i] = h['detector_score']
                result[side+'_valid'][i] = True
            if first_frame_s is None:
                first_frame_s = time.monotonic() - started
            if i % 10 == 0:
                print(f'frame={i}/{len(ids)} elapsed={time.monotonic()-started:.2f}s', flush=True)
        if cap.read()[0]:
            raise ValueError('video has unexpected additional frames')
    finally:
        cap.release()
    archive = args.output/'wilor_hands.npz'
    np.savez_compressed(archive, **result)
    elapsed = time.monotonic() - started
    report = dict(schema='ego.annotation.wilor_hand_measurements.v1', compute_host=args.compute_host,
                  CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], frames=len(ids), elapsed_s=elapsed,
                  first_frame_including_load_s=first_frame_s, after_first_frame_s=elapsed-first_frame_s,
                  fps_after_first_frame=(len(ids)-1)/(elapsed-first_frame_s),
                  source_hashes=hashes, output_sha256=sha(archive),
                  full_source_timeline=True, max_MANO_vertex_replay_error_m=max_replay,
                  semantics={'joints_camera_m':'absolute source calibrated camera coordinates, including camera_translation_m',
                             'mano':'right MANO rotation-matrix parameters; apply side reflection to full FK then add camera_translation_m',
                             'missing':'not inferred, not filled; NaN geometry with valid=false',
                             'selection':'highest detector score per side, never GT',
                             'camera':'camera-local inference, no world trajectory' if C is None else 'fixed supplied prediction-side trajectory, not WiLoR-estimated SLAM',
                             'RL_drop_in_ready':False},
                  coverage={s:int(result[s+'_valid'].sum()) for s in ['left','right']},
                  duplicate_side_frames={s:int((result[s+'_detection_count']>1).sum()) for s in ['left','right']})
    # Freeze completion only after outputs are complete. Evaluation runs separately.
    archive.with_suffix('.completed.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(__doc__)
    for name in ['clip','wilor-root','output']:
        p.add_argument('--'+name, type=Path, required=True)
    camera = p.add_mutually_exclusive_group(required=True)
    camera.add_argument('--camera-track', type=Path)
    camera.add_argument('--camera-contract', type=Path)
    p.add_argument('--compute-host', required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--rescale-factor', type=float, default=2.)
    return p


if __name__ == '__main__':
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    run(parser().parse_args())
