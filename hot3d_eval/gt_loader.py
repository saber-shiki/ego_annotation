"""Recorded source resolution, TIME_CODE GT loading and projection validation."""
import csv
import json
import re
from pathlib import Path

import numpy as np
from .alignment import pose, interpolate_pose, transform_points, validate_pose
from .mano_fk import sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with open(path, newline='') as f:
        yield from csv.DictReader(f)


def recorded_path(value, base):
    p = Path(value)
    return p if p.is_absolute() else Path(base) / p


def resolve_source(annotation_path, annotation, source_video=None, sensor_contract=None):
    p = Path(annotation_path).resolve()
    identity = annotation.get('input_identity') or {}
    derivation_path = p.parent / '../../input_source/DERIVATION.json'
    derivation = read_json(derivation_path) if derivation_path.is_file() else None
    video = source_video or identity.get('source_video_path')
    sensor = sensor_contract or identity.get('source_sensor_calibration_path')
    route = 'CLI/JSON input_identity'
    if not video:
        if derivation is None:
            raise ValueError('Missing source_video_path and ../../input_source/DERIVATION.json; use --source-video')
        video = derivation['source_video']
        route = 'DERIVATION.json fallback'
    video = recorded_path(video, p.parent)
    if not sensor and derivation is not None:
        sensor = derivation_path.parent / 'sensor_calibration_derived.json'
    if not sensor:
        raise ValueError('Missing source_sensor_calibration_path; use --sensor-contract')
    sensor = recorded_path(sensor, p.parent)
    contract = read_json(sensor)
    provenance_path = video.parent / 'INPUT_PROVENANCE.json'
    provenance = read_json(provenance_path)
    sequence = provenance['source_sequence_id']
    if contract.get('source_sequence_id') != sequence or contract.get('stream_id') != '214-1':
        raise ValueError('Source sequence/stream disagrees with sensor contract')
    # A derived contract carries an explicit offset in the parent input clip.
    derived = contract.get('derived_short_input')
    if derived is None and route == 'DERIVATION.json fallback':
        derived = derivation
    offset = int(derived['source_frame_start']) if derived else 0
    recorded = provenance.get('test_provenance', '')
    paths = re.findall(r'(/[^\s\"\']+\.mp4)', recorded) if isinstance(recorded, str) else []
    if len(paths) > 1:
        raise ValueError('Ambiguous test_provenance video paths')
    manifest_path = Path(paths[0]).parent / 'input/raw_frame_manifest/manifest.json' if paths else None
    manifest = None
    if manifest_path is not None and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest['source_sequence_id'] != sequence or manifest['source_stream_id'] != '214-1':
            raise ValueError('Manifest sequence/stream does not match provenance')
    return {'sequence': sequence, 'video': str(video), 'sensor': str(sensor), 'contract': contract,
            'provenance': provenance, 'offset': offset, 'manifest': manifest,
            'route': route, 'source_sha256': sha256(provenance_path), 'sensor_sha256': sha256(sensor),
            'manifest_sha256': sha256(manifest_path) if manifest is not None else None,
            'timestamp_route': 'manifest TIME_CODE' if manifest is not None else 'box2d 214-1 sorted-unique index fallback'}


def frame_mapping(frames, source, box_timestamps):
    ids = [int(f['frame_idx']) for f in frames]
    if not ids or any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError('Annotation frame_idx must be strictly increasing and nonempty')
    manifest = source['manifest']
    rows = {int(f['frame_idx']): f for f in manifest['frames']} if manifest else None
    timestamps, indices = [], []
    for i in ids:
        clip_index = i + source['offset']
        expected = int(source['provenance']['source_frame_idx_start']) + clip_index
        if expected > int(source['provenance']['source_frame_idx_end']) or clip_index < 0:
            raise ValueError(f'Frame {i} outside recorded source range')
        if rows is not None:
            row = rows[clip_index]
            if row.get('source_time_domain') != 'TIME_CODE':
                raise ValueError('Manifest timestamp must be TIME_CODE, never DEVICE_TIME')
            idx, ts = int(row['source_frame_idx']), int(row['timestamp_ns'])
            if idx != expected:
                raise ValueError('Manifest source_frame_idx disagrees with provenance')
        else:
            idx = expected
            ts = int(box_timestamps[idx])
        indices.append(idx)
        timestamps.append(ts)
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Source timestamps must be strictly increasing')
    return ids, indices, timestamps


def camera_at(ts, times, transforms):
    pos = int(np.searchsorted(times, ts))
    if pos < len(times) and times[pos] == ts:
        return transforms[pos], {'method': 'exact', 'timestamp_ns': int(ts), 'delta_ns': 0}
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(times)]
    nearest = min(candidates, key=lambda i: abs(int(times[i]) - ts))
    delta = int(times[nearest]) - ts
    if abs(delta) <= 1_000_000:
        return transforms[nearest], {'method': 'nearest<=1ms', 'timestamp_ns': int(ts), 'delta_ns': delta}
    if pos == 0 or pos == len(times):
        raise ValueError(f'Headset trajectory cannot bracket timestamp {ts}')
    t0, t1 = int(times[pos - 1]), int(times[pos])
    T = interpolate_pose(transforms[pos - 1], transforms[pos], (ts - t0) / (t1 - t0))
    return T, {'method': 'SE3-interpolation', 'timestamp_ns': int(ts), 'bracket_ns': [t0, t1]}


def output_rotation(contract):
    oc = contract['orientation_contract']
    expected = 'T_world_camera_native @ inv(R_output_camera_from_native_camera) in homogeneous form'
    if oc['T_world_camera_output'] != expected:
        raise ValueError('Unknown orientation_contract T_world_camera_output semantics; explicit adapter required')
    R = np.asarray(oc['R_output_camera_from_native_camera'], float)
    Q = np.eye(4)
    Q[:3, :3] = R
    validate_pose(Q)
    return Q


def fisheye624(points, params):
    """Native FISHEYE624: equidistant + radial6 + tangential2 + thin-prism4."""
    xyz, k = np.asarray(points, float), np.asarray(params, float)
    if k.shape != (15,):
        raise ValueError('FISHEYE624 requires 15 projection parameters')
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    theta = np.arctan2(radius, xyz[:, 2])
    radial = theta * (1 + sum(k[3 + i] * theta ** (2 * (i + 1)) for i in range(6)))
    ab = xyz[:, :2] * np.divide(radial, radius, out=np.zeros_like(radius), where=radius > 1e-12)[:, None]
    r2 = np.sum(ab * ab, axis=1)
    tangential = 2 * (ab @ k[9:11])[:, None] * ab + r2[:, None] * k[9:11]
    prism = np.stack([k[11] * r2 + k[12] * r2 ** 2, k[13] * r2 + k[14] * r2 ** 2], axis=1)
    return k[0] * (ab + tangential + prism) + k[1:3]


class GroundTruth:
    def __init__(self, dataset_root, source, frames, fk):
        self.source, self.fk = source, fk
        root = Path(dataset_root) / source['sequence']
        self.boxes = {}
        for r in csv_rows(root / 'box2d_hands.csv'):
            if r['stream_id'] == '214-1':
                self.boxes[(int(r['timestamp[ns]']), int(r['hand_index']))] = r
        timeline = sorted({ts for ts, _ in self.boxes})
        self.frame_ids, self.source_indices, self.timestamps = frame_mapping(frames, source, timeline)
        wanted = set(self.timestamps)
        records = {}
        with (root / 'mano_hand_pose_trajectory.jsonl').open() as f:
            for line in f:
                r = json.loads(line)
                if r['timestamp_ns'] in wanted:
                    records[r['timestamp_ns']] = r['hand_poses']
        mask_path = root / 'masks/mask_hand_pose_available.csv'
        if not mask_path.is_file():
            raise FileNotFoundError('Required GT availability mask missing: masks/mask_hand_pose_available.csv')
        masks = {int(r['timestamp[ns]']): r['mask'].lower() == 'true'
                 for r in csv_rows(mask_path) if r['stream_id'] == '214-1'}
        self.records, self.hands, self.valid = {}, {}, {}
        for side, hand_id in [('left', '0'), ('right', '1')]:
            rr = [records.get(ts, {}).get(hand_id) if masks.get(ts, False) else None for ts in self.timestamps]
            self.records[side] = rr
            valid = np.asarray([r is not None for r in rr])
            self.valid[side] = valid
            joints = np.full((len(frames), 21, 3), np.nan)
            # Bounded CPU batches keep full-sequence evaluation practical.
            good = np.flatnonzero(valid)
            for start in range(0, len(good), 256):
                idx = good[start:start + 256]
                joints[idx] = fk.forward(side, [rr[i] for i in idx])
            self.hands[side] = joints
        models = read_json(root / 'camera_models.json')
        camera = next(c for c in models if c['stream_id'] == '214-1')
        if camera['projectionModelType'] != 'CameraModelType.FISHEYE624':
            raise ValueError('Expected native RGB FISHEYE624')
        self.params = np.asarray(source['contract']['source_static_projection_params'], float)
        if not np.allclose(self.params, camera['projectionParams'], atol=1e-5, rtol=0):
            raise ValueError('Contract native FISHEYE624 differs from GT static camera')
        extrinsic = camera['T_Device_Camera']
        Tdc = pose(extrinsic['translation_xyz'], extrinsic['quaternion_wxyz'])
        trajectory = {}
        for r in csv_rows(root / 'headset_trajectory.csv'):
            ts = int(r['timestamp[ns]'])
            trajectory[ts] = pose([float(r['t_wo_' + a + '[m]']) for a in 'xyz'],
                                  [float(r['q_wo_' + a]) for a in 'wxyz'])
        times = np.asarray(sorted(trajectory), dtype=np.int64)
        transforms = [trajectory[int(t)] for t in times]
        self.Q = output_rotation(source['contract'])
        self.cameras, self.camera_sampling = [], []
        for ts in self.timestamps:
            T, sampling = camera_at(ts, times, transforms)
            self.cameras.append(T @ Tdc @ np.linalg.inv(self.Q))
            self.camera_sampling.append(sampling)
        self.cameras = np.asarray(self.cameras)

    def projection_gate(self):
        """At least three visible GT samples per hand; no prediction-dependent selection."""
        diagnostics, failures = [], []
        for side, hand_id in [('left', 0), ('right', 1)]:
            candidates = [i for i, ts in enumerate(self.timestamps)
                          if self.valid[side][i] and (ts, hand_id) in self.boxes
                          and float(self.boxes[(ts, hand_id)]['visibility_ratio[%]']) > 0]
            if len(candidates) < 3:
                failures.append(f'{side}: only {len(candidates)} visible GT frames; gate requires >=3')
                continue
            sample = [candidates[int(i)] for i in np.linspace(0, len(candidates) - 1, 3)]
            for i in sample:
                ts = self.timestamps[i]
                box = self.boxes[(ts, hand_id)]
                lo = np.array([float(box['x_min[pixel]']), float(box['y_min[pixel]'])])
                hi = np.array([float(box['x_max[pixel]']), float(box['y_max[pixel]'])])
                center, half = (lo + hi) / 2, (hi - lo) * 0.6
                native = self.cameras[i] @ self.Q
                points = transform_points(np.linalg.inv(native), self.hands[side][i])
                uv = fisheye624(points, self.params)
                inside = ((uv >= center - half) & (uv <= center + half)).all(axis=1) & (points[:, 2] > 0)
                fraction = float(inside.mean())
                d = {'side': side, 'frame_idx': self.frame_ids[i], 'source_frame_idx': self.source_indices[i],
                     'timestamp_ns': ts, 'inside': int(inside.sum()), 'total': 21, 'fraction': fraction,
                     'box_xyxy': [*lo.tolist(), *hi.tolist()], 'expanded_xyxy': [*(center-half).tolist(), *(center+half).tolist()],
                     'projected_uv': uv.tolist(), 'mean_offset_from_box_center_px': (uv.mean(0) - center).tolist(),
                     'passed': fraction >= 0.8}
                diagnostics.append(d)
                if not d['passed']:
                    failures.append(f'{side} frame {self.frame_ids[i]}: {inside.sum()}/21 inside expanded box')
        return {'passed': not failures, 'projection': 'native FISHEYE624, contract parameters; output camera transformed back by R_output_from_native',
                'samples': diagnostics, 'failures': failures}
