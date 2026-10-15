"""Synthetic source RGB -> producer -> immutable dataset contracts; no model loading."""
from __future__ import annotations
import argparse
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from data.bundle import EpisodeBundle, write_manifest
from data.rgb import load_rgb_source_evidence
from observations.object import rgb_tracking as rgb
from observations.object.rgb_evidence import SCHEMA, EXTENSION
from observations.object.surface import robust_first_surface_depth_ownership
from utils.state import sha


def source_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    ids = [3, 11]
    size = (80, 60)
    K = np.array([90., 90., 39.5, 29.5])
    yy, xx = np.indices(size[::-1])
    depth = .8 + .001 * xx + .0001 * (yy-30)**2
    mask = np.full(size[::-1], 255, np.uint8)
    settings = dict(mad_sigma=4., min_half_width_m=.2, min_retained_fraction=.9,
        fail_raw_to_robust_extent_ratio=4., confidence_seed_percentile=95.,
        local_depth_step_max_m=.05, max_removed_distance_inside_mask_px=10.,
        max_confidence_flagged_interior_fraction=.015, min_interior_confidence_flagged_fraction=.95,
        max_unexplained_interior_pixels=5, min_component_pixels=20, max_small_component_fraction=.01)
    retained, own = robust_first_surface_depth_ownership(mask > 0, depth, enabled=True,
        intrinsics=K, confidence=np.ones_like(depth), **settings)
    assert retained.all()
    parameters = {'min_depth_m': .05, 'max_depth_m': 4., 'robust_first_surface_depth_ownership': True,
        'native_depth_metric_eligibility_enabled': False,
        **{'first_surface_' + k: v for k, v in settings.items()}}
    uv = np.array([[x, y] for y in (16, 23, 30, 37, 44) for x in (20, 28, 36, 44, 52, 60)], float)
    Cs = np.eye(4); Cs[:3, :3] = Rotation.from_rotvec([.03, .05, .02]).as_matrix(); Cs[:3, 3] = [.03, -.02, .1]
    Ct = np.eye(4); Ct[:3, :3] = Rotation.from_rotvec([0, 0, -.04]).as_matrix(); Ct[:3, 3] = [.02, .01, .05]
    world, valid = rgb.lift_source_world(3, uv, mask, depth, np.ones_like(depth), K, Cs, .05)
    assert valid.all()
    Rs = Rotation.from_rotvec([.2, -.1, .15]).as_matrix(); ts = np.array([.1, -.03, .9])
    D = Rotation.from_rotvec([0, .08, .01]).as_matrix()
    Rt = D @ Rs; tt = D @ ts + np.array([-.06, .015, .02])
    d = tt - Rt @ Rs.T @ ts
    target_camera = ((world @ D.T + d) - Ct[:3, 3]) @ Ct[:3, :3]
    target_uv = target_camera[:, :2] / target_camera[:, 2:] * K[:2] + K[2:]
    input_root = root/'input'; input_root.mkdir()
    rows = []; raw_rows = []
    for idx, camera in zip(ids, (Cs, Ct)):
        image_path = input_root/f'{idx}.png'; mask_path = root/f'mask{idx}.png'
        cv2.imwrite(str(image_path), np.full((120, 160, 3), 128, np.uint8)); cv2.imwrite(str(mask_path), mask)
        obj = {'object_id': 'object:synthetic', 'mask_path': str(mask_path), 'rigid_pose_observation_eligible': idx == 3}
        if idx == 3:
            obj['visible_geometry_candidate'] = {'frame_idx': idx, 'source_width': 80, 'source_height': 60,
                'rigid_pose_observation_eligible': True, 'world_vertices_sample_m': world.tolist(),
                'camera_vertices_sample_m': ((world-Cs[:3, 3]) @ Cs[:3, :3]).tolist(),
                'first_surface_depth_ownership': own,
                'mask_depth_transform_contract': {'mask_size_wh': [80, 60], 'depth_size_wh': [80, 60],
                    'A_depth_from_mask_coordinate_model': np.eye(3).tolist()}}
        rows.append({'frame_idx': idx, 'time_s': .2 + (idx-3)*.05, 'source_width': 80, 'source_height': 60,
            'camera': {'T_world_camera_metric': camera.tolist(), 'intrinsics_fx_fy_cx_cy': K.tolist()},
            'objects': [obj], 'hands': []})
        raw_rows.append({'frame_idx': idx, 'time_s': rows[-1]['time_s'], 'raw_frame_path': str(image_path),
            'source_width': 80, 'source_height': 60, 'manifest_width': 160, 'manifest_height': 120})
    annotations = root/'annotations.json'; annotations.write_text(json.dumps({'frames': rows}))
    manifest = input_root/'manifest.json'; manifest.write_text(json.dumps({'frames': raw_rows}))
    depth_path = root/'depth.npz'; np.savez(depth_path, frame_idx=[3], depth=depth[None],
        confidence=np.ones_like(depth)[None], source_size=size, intrinsics_fx_fy_cx_cy=K[None])
    adapter = root/'adapter.json'; adapter.write_text(json.dumps({'parameters': parameters}))
    assets = root/'assets.json'; assets.write_text('{}')
    from main import rgb_source_worker_arguments
    argv = rgb_source_worker_arguments(annotations=annotations, manifest=manifest, depth=depth_path,
        object_id='object:synthetic', frame_ids=ids, adapter=adapter, assets_manifest=assets,
        lightglue_root=root/'not-provisioned', output_npz=root/'rgb.npz', output_json=root/'rgb.json')
    argv[argv.index('--device')+1] = 'cpu'
    features = {idx: {'keypoints': torch.tensor((pixels+.5)*2-.5, dtype=torch.float32)[None]}
                for idx, pixels in zip(ids, (uv, target_uv))}
    return {'root': root, 'argv': argv, 'features': features, 'world': world, 'D': D, 'd': d,
        'frames': rows, 'ids': ids, 'times': [r['time_s'] for r in rows],
        'inputs': {'annotations': annotations, 'depth_npz': depth_path,
            'raw_frame_manifest': manifest, 'p09_adapter_report': adapter}}


def run_fixture(fixture, *, matches=True, match_count=None):
    count = len(fixture['world']) if matches else 0
    if match_count is not None:
        count = match_count
    module = ModuleType('lightglue.utils'); module.rbd = lambda value: value
    matched = {'matches': torch.tensor(np.column_stack([np.arange(count)]*2), dtype=torch.int64),
               'scores': torch.full((count,), .8)}
    with patch.dict(sys.modules, {'lightglue.utils': module, 'legacy_optimize': None, 'optimize': None}), \
         patch.object(rgb, 'extract_rgb_features', return_value=fixture['features']), \
         patch.object(rgb, 'build_rgb_matcher', return_value=lambda inputs: matched), \
         patch.object(rgb, 'initialize', side_effect=AssertionError('observation cannot initialize')), \
         patch.object(cv2, 'solvePnPRefineLM', side_effect=AssertionError('no hidden refinement')):
        cv2.setRNGSeed(42)
        return rgb.build_rgb_edges(fixture['argv'])


def load_fixture(fixture):
    return load_rgb_source_evidence(fixture['root']/'rgb.npz', fixture['root']/'rgb.json',
        frame_ids=fixture['ids'], times=fixture['times'], frames=fixture['frames'],
        object_ids=['object:synthetic'], expected_inputs=fixture['inputs'])


def repack(root, *, change_data=None, change_report=None):
    report = json.loads((root/'rgb.json').read_text())
    with np.load(root/'rgb.npz', allow_pickle=False) as archive:
        data = dict(archive)
    if change_report:
        change_report(report)
    if change_data:
        change_data(data)
    data['report_metadata'] = np.asarray(json.dumps({k: v for k, v in report.items() if k != 'npz_sha256'}, sort_keys=True))
    np.savez(root/'rgb.npz', **data)
    report['npz_sha256'] = sha(root/'rgb.npz')
    (root/'rgb.json').write_text(json.dumps(report))


class RGBSourceTests(unittest.TestCase):
    def test_source_only_real_support_pnp_world_motion_and_target_without_pose_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            f = source_fixture(Path(directory))
            report = run_fixture(f)
            self.assertEqual(report['schema'], SCHEMA)
            self.assertEqual(report['accepted_pairs'], [[3, 11]])
            self.assertNotIn('initial_pose_report', report['inputs'])
            edge = load_fixture(f).edges[0]
            np.testing.assert_allclose(edge.source_world, f['world'][edge.inlier_indices], atol=1e-7)
            np.testing.assert_allclose(edge.rotation_source_to_target_world, f['D'], atol=1e-5)
            np.testing.assert_allclose(edge.translation_source_to_target_world_m, f['d'], atol=1e-5)
            self.assertFalse(edge.diagnostics['target_initial_pose_present'])
            self.assertEqual(edge.source_frame_idx, 3); self.assertEqual(edge.target_frame_idx, 11)
            with self.assertRaises(ValueError): edge.source_world.setflags(write=True)
            with self.assertRaises(TypeError): edge.diagnostics['inliers'] = 0
            self.assertGreater(edge.diagnostics['source_3d_conditioning'], 0)

    def test_empty_matches_and_missing_source_are_uncertain_not_zero_motion(self):
        for eligible in (True, False):
            with self.subTest(eligible=eligible), tempfile.TemporaryDirectory() as directory:
                f = source_fixture(Path(directory))
                if not eligible:
                    payload = json.loads(f['inputs']['annotations'].read_text())
                    payload['frames'][0]['objects'][0]['rigid_pose_observation_eligible'] = False
                    f['inputs']['annotations'].write_text(json.dumps(payload))
                with patch.object(rgb, 'extract_rgb_features', side_effect=AssertionError('no source needs no model')) if not eligible else ExitStack():
                    if eligible: run_fixture(f, matches=False)
                    else: rgb.build_rgb_edges(f['argv'])
                loaded = load_fixture(f)
                self.assertEqual(loaded.edges, ())
                self.assertTrue(loaded.report['source_rejections'])
                with np.load(Path(directory)/'rgb.npz', allow_pickle=False) as z:
                    self.assertEqual(z['rotation_source_to_target_world'].shape, (0, 3, 3))
                    np.testing.assert_array_equal(z['offsets'], [0])

    def test_four_to_seven_inliers_retain_uncertain_quality_and_roundtrip(self):
        for count in range(4, 8):
            with self.subTest(inliers=count), tempfile.TemporaryDirectory() as directory:
                f = source_fixture(Path(directory))
                camera = np.asarray(f['frames'][1]['camera']['T_world_camera_metric'])
                rvec = cv2.Rodrigues(camera[:3, :3].T @ f['D'])[0]
                tvec = (camera[:3, :3].T @ (f['d'] - camera[:3, 3])).reshape(3, 1)
                indices = np.array([0, 5, 24, 29, 10, 13, 21], dtype=np.int32)[:count, None]
                with patch.object(cv2, 'solvePnPRansac', return_value=(True, rvec, tvec, indices)):
                    run_fixture(f)
                loaded = load_fixture(f)
                self.assertEqual(len(loaded.edges), 1)
                edge = loaded.edges[0]
                self.assertEqual(len(edge.source_world), count)
                self.assertGreaterEqual(edge.diagnostics['matches_object'], 8)
                self.assertEqual(edge.diagnostics['status'], 'accepted')
                self.assertEqual(edge.diagnostics['quality_state'], 'uncertain_measurement')
                self.assertIn('few_inliers', edge.diagnostics['weak_reasons'])
                if count == 4:
                    self.assertIn('low_inlier_fraction', edge.diagnostics['weak_reasons'])
                    self.assertIn('underconditioned_source_surface', edge.diagnostics['weak_reasons'])
                self.assertEqual(edge.diagnostics['diagnostic_reference_thresholds_not_gates']['inliers'], 30)
                self.assertEqual(edge.diagnostics['point_weights_role'], 'uncalibrated_reprojection_heuristic_not_probability')
                np.testing.assert_allclose(edge.rotation_source_to_target_world, f['D'], atol=1e-10)
                self.assertEqual(loaded.report['mechanical_admissibility']['minimum_pnp_inliers'], 4)

    def test_unavailable_pnp_has_exact_reason_and_no_numeric_motion(self):
        cases = {'input': 'too_few_object_matches', 'three': 'insufficient_pnp_inliers',
            'failed': 'pnp_failed', 'nan': 'invalid_pnp_transform',
            'duplicate': 'invalid_pnp_inliers', 'foreign': 'invalid_pnp_inliers',
            'negative': 'invalid_pnp_cheirality_or_finite_support'}
        for case, reason in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                f = source_fixture(Path(directory))
                rvec = np.zeros((3, 1)); tvec = np.zeros((3, 1))
                indices = np.arange(4, dtype=np.int32)[:, None]
                success = True
                if case == 'three': indices = indices[:3]
                if case == 'failed': success = False; indices = None
                if case == 'nan': rvec[0, 0] = np.nan
                if case == 'duplicate': indices[:] = 0
                if case == 'foreign': indices[-1] = 1000
                if case == 'negative': tvec[2, 0] = -10
                with patch.object(cv2, 'solvePnPRansac', return_value=(success, rvec, tvec, indices)) as solver:
                    run_fixture(f, match_count=7 if case == 'input' else None)
                    if case == 'input': solver.assert_not_called()
                evidence = load_fixture(f)
                self.assertEqual(evidence.edges, ())
                self.assertEqual(evidence.report['rows'][0]['reason'], reason)
                self.assertEqual(evidence.report['rows'][0]['status'], 'rejected')
                with np.load(f['root']/'rgb.npz', allow_pickle=False) as arrays:
                    self.assertEqual(arrays['translation_source_to_target_world_m'].shape, (0, 3))
                    self.assertEqual(arrays['source_world'].shape, (0, 3))

    def test_tamper_pairing_and_foreign_frame_fail(self):
        for mode in ('tamper', 'pair', 'foreign', 'camera', 'offset', 'input', 'object', 'nan', 'rotation', 'negative',
                     'target_uv', 'reprojection_median', 'reprojection_p95', 'ransac_threshold'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); f = source_fixture(root); run_fixture(f)
                if mode == 'tamper':
                    with (root/'rgb.npz').open('ab') as handle: handle.write(b'tampered')
                elif mode == 'pair':
                    report = json.loads((root/'rgb.json').read_text()); report['parameters']['min_matches'] = 99
                    (root/'rgb.json').write_text(json.dumps(report))
                elif mode == 'foreign':
                    repack(root, change_report=lambda r: r['rows'][0].update(target_frame_idx=7))
                elif mode == 'camera':
                    repack(root, change_data=lambda d: d['target_camera'].__setitem__((0, 0, 3), .123))
                elif mode == 'offset':
                    repack(root, change_data=lambda d: d['offsets'].__setitem__(-1, 1))
                elif mode == 'nan':
                    repack(root, change_data=lambda d: d['source_world'].__setitem__((0, 0), np.nan))
                elif mode == 'rotation':
                    repack(root, change_data=lambda d: d['rotation_source_to_target_world'].__setitem__((0, 0, 0), 5.))
                elif mode == 'negative':
                    repack(root, change_data=lambda d: d['translation_source_to_target_world_m'].__setitem__((0, 2), -10.))
                elif mode == 'target_uv':
                    repack(root, change_data=lambda d: d['target_uv'].__setitem__((slice(None), 0), d['target_uv'][:, 0] + .25))
                elif mode == 'reprojection_median':
                    repack(root, change_report=lambda r: r['rows'][0].update(reprojection_median_px=1.0))
                elif mode == 'reprojection_p95':
                    repack(root, change_report=lambda r: r['rows'][0].update(reprojection_p95_px=1.0))
                elif mode == 'ransac_threshold':
                    repack(root, change_report=lambda r: r['parameters'].update(reprojection_threshold_px=1e-12))
                elif mode == 'object':
                    repack(root, change_report=lambda r: r.update(object_id='foreign'))
                else:
                    f['inputs']['annotations'].write_text('{}')
                with self.assertRaises(ValueError): load_fixture(f)

    def test_legacy_cli_requires_report_source_mode_forbids_report(self):
        with tempfile.TemporaryDirectory() as directory:
            f = source_fixture(Path(directory)); argv = f['argv'].copy()
            argv[argv.index('--observation-mode')+1] = 'legacy-canonical'
            with self.assertRaises(SystemExit): rgb.build_rgb_edges(argv)
            with self.assertRaises(SystemExit): rgb.build_rgb_edges(f['argv']+['--initial-pose-report', 'forbidden.json'])

    def test_observation_entry_and_worker_do_not_call_downstream_solver(self):
        import main
        tree = ast.parse(Path(main.__file__).read_text())
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(s.startswith(('legacy_optimize', 'optimize', 'render')) for s in imports))
        worker = ast.parse(Path(rgb.__file__).read_text())
        build = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'build_rgb_edges')
        calls = [n.func.id for n in ast.walk(build) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertTrue({'initialize', 'solve_scene_pose', 'least_squares'}.isdisjoint(calls))
        self.assertTrue(any(isinstance(n, ast.If) and '__name__' in ast.unparse(n.test) for n in worker.body))


if __name__ == '__main__':
    unittest.main()
