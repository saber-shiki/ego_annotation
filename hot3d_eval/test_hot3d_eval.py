"""Synthetic closed-form tests: no downloads, licensed assets, GPU or real dataset."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from .alignment import first_frame_registration, fit_points, pose, transform_points
from .gt_loader import camera_at, fisheye624, frame_mapping, output_rotation, resolve_source
from .mano_fk import (ManoFK, OURS21_TO_OFFICIAL19, MANO21_TO_OURS21,
                      MANO21_TO_CANONICAL20, find_models)
from .metrics_hand import constant_betas, evaluate_hand, pck_auc
from .metrics_camera import evaluate_camera
from .report import summary, render_report
from .run_eval import (ProjectionGateError, evaluate_annotation, legacy_state_joints,
                       main, parse_args, registered_predictions)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(39)
        self.gt = self.rng.normal(size=(8, 21, 3)) * .03
        self.gt += np.arange(8)[:, None, None] * .01
        self.valid = np.ones(8, bool)
        self.ids = list(range(8))

    def test_mapping_identity(self):
        mapping = OURS21_TO_OFFICIAL19
        self.assertEqual(len(mapping), 19)
        self.assertEqual(len(set(mapping)), 19)
        self.assertTrue({0, 1}.isdisjoint(mapping))
        self.assertEqual([MANO21_TO_OURS21[i] for i in mapping],
                         MANO21_TO_CANONICAL20[:5] + MANO21_TO_CANONICAL20[6:])

    def test_zero_error(self):
        m = evaluate_hand(self.gt, self.gt, self.valid, self.ids)
        for key in ['MPJPE_mm', 'PA_MPJPE_mm', 'W_MPJPE_mm', 'WA_MPJPE_mm', 'RTE_pct', 'Accel_m_s2']:
            self.assertAlmostEqual(m[key]['value'], 0, places=8, msg=key)
        self.assertEqual(m['PCK_AUC_pct']['value'], 100)
        self.assertEqual(m['FINGERTIP_PCK_AUC_pct']['value'], 100)

    def test_exact_translation(self):
        m = evaluate_hand(self.gt + [0.003, .004, 0], self.gt, self.valid, self.ids)
        self.assertAlmostEqual(m['MPJPE_mm']['value'], 5, places=10)
        self.assertEqual(m['MPJPE_mm']['valid'], 8 * 19)

    def test_auc_closed_form(self):
        # Inclusive endpoint: error exactly at 25 has a half-bin contribution.
        self.assertEqual(pck_auc([0]), 100)
        self.assertEqual(pck_auc([25]), 50.5)
        self.assertEqual(pck_auc([25.25]), 49.5)
        self.assertEqual(pck_auc([50]), .5)
        self.assertEqual(pck_auc([51]), 0)
        self.assertAlmostEqual(pck_auc([0, 25, 51]), (100 + 50.5) / 3, places=12)

    def test_sim3_recovery(self):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec([.4, -.2, .7]).as_matrix() * 1.7
        T[:3, 3] = [2, -1, 3]
        pred = transform_points(T, self.gt)
        m = evaluate_hand(pred, self.gt, self.valid, self.ids)
        for key in ['PA_MPJPE_mm', 'W_MPJPE_mm', 'WA_MPJPE_mm']:
            self.assertAlmostEqual(m[key]['value'], 0, places=8)
        self.assertGreater(m['MPJPE_mm']['value'], 100)

    def test_first_frame_registration_all_world_quantities(self):
        pred_T = pose([.2, -.5, 1], [1, 0, 0, 0])
        gt_T = pose([2, 3, 4], [.70710678, 0, 0, .70710678])
        A = first_frame_registration(gt_T, pred_T)
        frames = [{'frame_idx': 0, 'camera': {'T_world_camera': pred_T.tolist()},
                   'hands': {'left': {'joints_world': self.gt[0].tolist()}},
                   'object': {'visual_fitted_pose': {'rotation_world_from_completed_canonical_matrix': np.eye(3).tolist(),
                                                    'translation_world_m': [0, 0, 0]}}}]
        cameras, hands, objects = registered_predictions(frames, A)
        np.testing.assert_allclose(cameras[0], gt_T, atol=1e-8)
        np.testing.assert_allclose(hands['left'][0], transform_points(A, self.gt[0]))
        np.testing.assert_allclose(objects[0], A)
        self.assertTrue(np.isnan(hands['right']).all())

    def test_rpe_translation_closed_form(self):
        gt = np.tile(np.eye(4), (4, 1, 1))
        gt[:, 0, 3] = np.arange(4)
        pred = gt.copy()
        pred[:, 0, 3] = 1.1 * np.arange(4)
        m = evaluate_camera(pred, gt, range(4))
        self.assertAlmostEqual(m['RPE_T_mm']['value'], 100)
        self.assertEqual(m['RPE_R_deg']['value'], 0)
        self.assertAlmostEqual(m['ATE_first_mm']['value'], np.sqrt(.035) * 1000)
        self.assertAlmostEqual(m['ATE_Sim3_mm']['value'], 0, places=8)

    def test_rpe_rotation_closed_form(self):
        gt = np.tile(np.eye(4), (4, 1, 1))
        pred = gt.copy()
        # Explicit (N,1) batch shape is accepted by both old and SciPy 1.17+.
        pred[:, :3, :3] = Rotation.from_euler('z', np.array([0, 10, 20, 30])[:, None], degrees=True).as_matrix()
        m = evaluate_camera(pred, gt, range(4))
        self.assertAlmostEqual(m['RPE_R_deg']['value'], 10)
        self.assertEqual(m['RPE_T_mm']['value'], 0)
        self.assertIsNone(m['ATE_Sim3_mm']['value'])

    def test_temporal_closed_forms(self):
        t = np.arange(8) / 30
        static = np.broadcast_to(self.gt[0], (8, 21, 3)).copy()
        quadratic = static.copy()
        quadratic[:, :, 0] += (2 * t ** 2)[:, None]
        m = evaluate_hand(quadratic, static, self.valid, self.ids)
        self.assertAlmostEqual(m['Accel_m_s2']['value'], 4)
        self.assertAlmostEqual(m['Jerk_3rd_diff_mm']['value'], 0)
        cubic = static.copy()
        cubic[:, :, 0] += t[:, None] ** 3
        m = evaluate_hand(cubic, static, self.valid, self.ids)
        self.assertAlmostEqual(m['Jerk_3rd_diff_mm']['value'], 6 / 30 ** 3 * 1000)

    def test_missing_and_gap_windows(self):
        valid = self.valid.copy()
        valid[3] = False
        ids = [0, 1, 2, 3, 4, 5, 7, 8]
        m = evaluate_hand(self.gt, self.gt, valid, ids)
        self.assertEqual(m['Accel_m_s2']['valid'], 21)  # only frames 0,1,2
        self.assertEqual(m['Jerk_3rd_diff_mm']['valid'], 0)
        self.assertEqual(m['W_MPJPE_mm']['valid'], 7 * 21)
        self.assertIn('lengths=[3, 2, 2]', m['W_MPJPE_mm']['reason'])

    def test_segment_tail_retained_singleton_excluded(self):
        gt = self.rng.normal(size=(201, 21, 3))
        m = evaluate_hand(gt, gt, np.ones(201, bool), range(201))
        self.assertEqual(m['W_MPJPE_mm']['valid'], 200 * 21)
        self.assertIn('lengths=[100, 100]', m['W_MPJPE_mm']['reason'])
        self.assertIn('1 singleton', m['W_MPJPE_mm']['reason'])

    def test_constant_beta_drift_rejected_even_invalid_frame(self):
        frames = [{'hands': {'left': {'parameters': {'source_betas': [0.] * 10}}}} for _ in range(3)]
        beta, _ = constant_betas(frames, 'left', [True, False, True])
        np.testing.assert_array_equal(beta, np.zeros(10))
        frames[1]['hands']['left']['parameters']['source_betas'][2] = 1
        with self.assertRaisesRegex(ValueError, 'not constant'):
            constant_betas(frames, 'left', [True, False, True])

    def test_zero_prediction_coverage(self):
        m = evaluate_hand(np.full_like(self.gt, np.nan), self.gt, np.zeros(8, bool), self.ids)
        self.assertTrue(all(v['value'] is None and v['valid'] == 0 for v in m.values()))

    def test_shape_only_beta_expansion(self):
        class FakeTensor:
            def __init__(self, v):
                self.v = v
            def detach(self):
                return self
            def numpy(self):
                return self.v
        dirs = np.zeros((778, 3, 10)); dirs[:, 0, 0] = .002
        fk = ManoFK.__new__(ManoFK)
        fk.models = {'left': SimpleNamespace(v_template=FakeTensor(np.zeros((778, 3))), shapedirs=FakeTensor(dirs))}
        vertices = fk.shape_vertices('left', [3] + [0] * 9)
        self.assertAlmostEqual(np.linalg.norm(vertices, axis=-1).mean() * 1000, 6)


class SourceTest(unittest.TestCase):
    def test_headset_exact_nearest_interpolated(self):
        a, b = np.eye(4), pose([10, 0, 0], [0, 0, 0, 1])
        times = np.array([0, 10_000_000])
        T, info = camera_at(0, times, [a, b])
        self.assertEqual(info['method'], 'exact')
        T, info = camera_at(900_000, times, [a, b])
        self.assertEqual(info['method'], 'nearest<=1ms')
        T, info = camera_at(5_000_000, times, [a, b])
        np.testing.assert_allclose(T[:3, 3], [5, 0, 0])
        np.testing.assert_allclose(T[:3, :3], Rotation.from_euler('z', 90, degrees=True).as_matrix(), atol=1e-10)
        self.assertEqual(info['method'], 'SE3-interpolation')
        with self.assertRaises(ValueError):
            camera_at(20_000_000, times, [a, b])

    def test_fisheye_zero_distortion(self):
        uv = fisheye624([[0, 0, 1], [1, 0, 1], [0, 1, 1]], [100, 50, 60] + [0] * 12)
        np.testing.assert_allclose(uv, [[50, 60], [50 + np.pi * 25, 60], [50, 60 + np.pi * 25]])

    def test_orientation_semantics(self):
        contract = {'orientation_contract': {'R_output_camera_from_native_camera': [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                      'T_world_camera_output': 'T_world_camera_native @ inv(R_output_camera_from_native_camera) in homogeneous form'}}
        Q = output_rotation(contract)
        np.testing.assert_allclose(np.linalg.inv(Q)[:3, :3], Rotation.from_euler('z', -90, degrees=True).as_matrix(), atol=1e-9)
        contract['orientation_contract']['T_world_camera_output'] = 'unknown'
        with self.assertRaises(ValueError):
            output_rotation(contract)

    def test_manifest_time_code_and_fallback(self):
        source = {'offset': 1, 'provenance': {'source_frame_idx_start': 4, 'source_frame_idx_end': 9}, 'manifest': None}
        ids, indices, ts = frame_mapping([{'frame_idx': 0}, {'frame_idx': 1}], source, np.arange(20) * 33)
        self.assertEqual(indices, [5, 6]); self.assertEqual(ts, [165, 198])
        source['manifest'] = {'frames': [{'frame_idx': 1, 'source_frame_idx': 5, 'timestamp_ns': 123,
                                        'device_timestamp_ns': 999, 'source_time_domain': 'TIME_CODE'}]}
        self.assertEqual(frame_mapping([{'frame_idx': 0}], source, [])[2], [123])
        source['manifest']['frames'][0]['source_time_domain'] = 'DEVICE_TIME'
        with self.assertRaisesRegex(ValueError, 'TIME_CODE'):
            frame_mapping([{'frame_idx': 0}], source, [])

    def test_asset_env_and_explicit_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['MANO_LEFT.pkl', 'MANO_RIGHT.pkl', 'override.pkl']:
                (root / name).write_bytes(b'not loaded')
            with patch.dict(os.environ, {'EGO_MANO_MODELS': tmp}):
                paths = find_models(mano_left=root / 'override.pkl')
                self.assertEqual(paths['left'].name, 'override.pkl')
                self.assertEqual(paths['right'].name, 'MANO_RIGHT.pkl')


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'source'; source.mkdir()
        self.path = self.root / 'run/final/annotation.json'; self.path.parent.mkdir(parents=True)
        self.frames = []
        self.joints = np.random.default_rng(31).normal(size=(21, 3)) * .015 + [0, 0, 1]
        for i in range(4):
            T = np.eye(4); T[0, 3] = i * .01
            self.frames.append({'frame_idx': i, 'camera': {'T_world_camera': T.tolist()},
                                'hands': {s: {'joints_world': self.joints.tolist(), 'parameters': {'source_betas': [0] * 10}}
                                          for s in ['left', 'right']}})
        self.annotation = {'object_id': 'synthetic_object', 'frames': self.frames,
                           'input_identity': {'source_video_path': str(source / 'input.mp4'),
                                              'source_sensor_calibration_path': str(source / 'sensor.json')}}
        self.path.write_text(json.dumps(self.annotation))
        prov = {'source_sequence_id': 'synthetic', 'source_frame_idx_start': 0, 'source_frame_idx_end': 3}
        (source / 'INPUT_PROVENANCE.json').write_text(json.dumps(prov))
        self.params = [100, 500, 500] + [0] * 12
        contract = {'source_sequence_id': 'synthetic', 'stream_id': '214-1', 'source_static_projection_params': self.params,
                    'orientation_contract': {'R_output_camera_from_native_camera': np.eye(3).tolist(),
                     'T_world_camera_output': 'T_world_camera_native @ inv(R_output_camera_from_native_camera) in homogeneous form'}}
        (source / 'sensor.json').write_text(json.dumps(contract))
        ds = self.root / 'dataset/synthetic'; ds.mkdir(parents=True); (ds / 'masks').mkdir()
        self.ds = ds
        camera = {'stream_id': '214-1', 'projectionModelType': 'CameraModelType.FISHEYE624', 'projectionParams': self.params,
                  'T_Device_Camera': {'quaternion_wxyz': [1, 0, 0, 0], 'translation_xyz': [0, 0, 0]}}
        (ds / 'camera_models.json').write_text(json.dumps([camera]))
        times = [i * 33333333 for i in range(4)]
        boxes = 'stream_id,hand_index,timestamp[ns],x_min[pixel],x_max[pixel],y_min[pixel],y_max[pixel],visibility_ratio[%]\n'
        boxes += ''.join(f'214-1,{s},{ts},450,550,450,550,1\n' for ts in times for s in range(2))
        (ds / 'box2d_hands.csv').write_text(boxes)
        (ds / 'masks/mask_hand_pose_available.csv').write_text('timestamp[ns],stream_id,mask\n' + ''.join(f'{ts},214-1,True\n' for ts in times))
        trajectory = 'object_uid,timestamp[ns],t_wo_x[m],t_wo_y[m],t_wo_z[m],q_wo_w,q_wo_x,q_wo_y,q_wo_z\n'
        trajectory += ''.join(f'1,{ts},{i*.01},0,0,1,0,0,0\n' for i, ts in enumerate(times))
        (ds / 'headset_trajectory.csv').write_text(trajectory)
        record = {'pose': [0] * 15, 'betas': [0] * 10, 'wrist_xform': {'q_wxyz': [1, 0, 0, 0], 't_xyz': [0, 0, 0]}}
        (ds / 'mano_hand_pose_trajectory.jsonl').write_text('\n'.join(json.dumps({'timestamp_ns': ts, 'hand_poses': {'0': record, '1': record}}) for ts in times))
        self.fk = SimpleNamespace(provenance={s: {'filename': 'synthetic', 'sha256': 'synthetic'} for s in ['left', 'right']},
                                  left_shapedirs_fixed=False,
                                  forward=lambda side, rows: np.tile(self.joints, (len(rows), 1, 1)),
                                  shape_vertices=lambda side, b: np.zeros(np.asarray(b).shape[:-1] + (778, 3)))
        self.args = SimpleNamespace(source_video=None, sensor_contract=None, dataset_root=self.root / 'dataset')

    def test_full_evaluation_report(self):
        r = evaluate_annotation(self.path, self.args, self.fk)
        self.assertTrue(r['gate']['passed'])
        self.assertEqual(len(r['gate']['samples']), 6)
        self.assertEqual(r['coverage']['left']['valid'], 4)
        self.assertEqual(r['hands']['left']['MPVPE_mm']['value'], 0)
        self.assertEqual(r['hands']['left']['MPJPE_mm']['value'], 0)
        self.assertEqual(r['camera']['ATE_first_mm']['value'], 0)
        report = render_report([r])
        self.assertIn('Sequence-equal mean', report)
        self.assertIn('Projection gate: **PASS**', report)
        self.assertIn('joints source: JSON joints_world', report)

    def test_gate_failure_aborts_no_report(self):
        self.fk.forward = lambda side, rows: np.tile(self.joints + [10, 0, 0], (len(rows), 1, 1))
        with self.assertRaises(ProjectionGateError) as caught:
            evaluate_annotation(self.path, self.args, self.fk)
        self.assertEqual(len(caught.exception.diagnostic['gate']['samples']), 6)
        output = self.root / 'report.md'
        with patch('hot3d_eval.run_eval.find_models', return_value={}), patch('hot3d_eval.run_eval.ManoFK', return_value=self.fk):
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                code = main(['--annotations', str(self.path), '--dataset-root', str(self.root / 'dataset'), '--output', str(output)])
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())
        self.assertTrue(output.with_suffix('.md.gate-failure.json').exists())

    def test_legacy_derivation_source_resolution(self):
        legacy = copy.deepcopy(self.annotation); del legacy['input_identity']
        source_dir = self.root / 'input_source'; source_dir.mkdir()
        (source_dir / 'DERIVATION.json').write_text(json.dumps({'source_video': str(self.root / 'source/input.mp4'), 'source_frame_start': 0}))
        (source_dir / 'sensor_calibration_derived.json').write_text((self.root / 'source/sensor.json').read_text())
        source = resolve_source(self.path, legacy)
        self.assertEqual(source['route'], 'DERIVATION.json fallback')
        self.assertEqual(source['sequence'], 'synthetic')

    def make_legacy_archive(self):
        legacy = copy.deepcopy(self.annotation)
        for f in legacy['frames']:
            for side in ['left', 'right']:
                del f['hands'][side]['joints_world']
        archive = self.root / 'run/measurements/scene_fit/synthetic_object/demo_state.npz'
        archive.parent.mkdir(parents=True)
        np.savez(archive, frame_idx=np.arange(4), T_world_camera=np.asarray([f['camera']['T_world_camera'] for f in self.frames]),
                 left_joints_world=np.tile(self.joints, (4, 1, 1)), right_joints_world=np.tile(self.joints, (4, 1, 1)))
        return legacy, archive

    def test_legacy_archive_read_only_and_new_null_authority(self):
        legacy, archive = self.make_legacy_archive()
        before = archive.read_bytes()
        joints, provenance = legacy_state_joints(self.path, legacy)
        self.assertEqual(joints['left'].shape, (4, 21, 3))
        self.assertIn('state archive', provenance['description'])
        self.assertEqual(before, archive.read_bytes())
        legacy['frames'][0]['hands']['left']['joints_world'] = None
        self.assertIsNone(legacy_state_joints(self.path, legacy)[0])

    def test_legacy_full_metrics_and_shape_drift_diagnostic(self):
        legacy, _ = self.make_legacy_archive()
        for i, f in enumerate(legacy['frames']):
            for side in ['left', 'right']:
                f['hands'][side]['parameters']['source_betas'][0] = i + 1
        self.path.write_text(json.dumps(legacy))
        def shape(side, beta):
            b = np.asarray(beta)
            vertices = np.zeros(b.shape[:-1] + (778, 3))
            vertices[..., 0] = b[..., 0, None] * .002
            return vertices
        self.fk.shape_vertices = shape
        errors = io.StringIO()
        with redirect_stderr(errors):
            r = evaluate_annotation(self.path, self.args, self.fk)
        self.assertIn('ERROR MPVPE(official)', errors.getvalue())
        self.assertEqual(r['coverage']['left']['valid'], 4)
        self.assertIsNone(r['hands']['left']['MPVPE_mm']['value'])
        self.assertIn('max abs drift', r['hands']['left']['MPVPE_mm']['reason'])
        self.assertAlmostEqual(r['hands']['left']['MPVPE_mean_beta_diagnostic_mm']['value'], 5)
        self.assertIn('MPVPE(mean-β, diagnostic)', render_report([r]))
        self.assertEqual(r['hands']['left']['MPJPE_mm']['value'], 0)

    def test_legacy_archive_stale_camera_rejected(self):
        legacy, archive = self.make_legacy_archive()
        legacy['frames'][0]['camera']['T_world_camera'][0][3] += .01
        with self.assertRaisesRegex(ValueError, 'camera poses disagree'):
            legacy_state_joints(self.path, legacy)

    def test_manifest_relative_paths(self):
        manifest = self.root / 'inputs.txt'; manifest.write_text('# comment\nrun/final/annotation.json\n')
        args = parse_args(['--manifest', str(manifest), '--output', str(self.root / 'report.md')])
        self.assertEqual(args.annotations, [self.path])

    def test_sequence_equal_summary(self):
        results = [{'sequence': 'a', 'camera': {'x': {'value': 1}}}, {'sequence': 'a', 'camera': {'x': {'value': 3}}},
                   {'sequence': 'b', 'camera': {'x': {'value': 10}}}]
        self.assertEqual(summary(results, 'camera', 'x'), 6)


if __name__ == '__main__':
    unittest.main()
