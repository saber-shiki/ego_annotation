"""Synthetic CPU checks of local RGB seeding, loop filtering and surface authority."""
from __future__ import annotations
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from utils.data import PoseNode, RGBEdge, RGBReprojectionFactor, PointFactor
from observations.object.rgb_tracking import local_rotation_evidence, observed_rotation_authority
from optimization.object.pose import solve_scene_pose as solve, GraphConfig, residual_blocks

def fixture(count=9, ids=None):
    ids = ids or [17 + 31 * i for i in range(count)]
    times = np.linspace(0, 2, count)
    truth = [Rot.from_rotvec(np.array([0.4, -0.2, 1.0]) * t).as_matrix() for t in times]
    anchor = count // 2
    nodes = [PoseNode(i, truth[anchor].copy() if k == anchor else np.eye(3), np.array([0.0, 0.0, 1.0]), None, 'synthetic', True, 0.1, {'frame_idx': i, 'source': 'immutable'}) for (k, i) in enumerate(ids)]
    edges = []
    for k in range(count - 1):
        (a, b) = (k + 1, k)
        d = truth[b] @ truth[a].T
        edges.append(RGBEdge(ids[a], ids[b], d, nodes[b].initial_translation - d @ nodes[a].initial_translation, 0.8, {'source_3d_conditioning': 0.0001, 'inlier_fraction': 0.9, 'reprojection_median_px': 0.1}))
    for (a, b) in [(0, count - 1), (1, count - 2), (0, count - 2), (1, count - 1)]:
        edges.append(RGBEdge(ids[a], ids[b], Rot.from_euler('x', 0.05).as_matrix(), np.zeros(3), 2.0, {'source_3d_conditioning': 0.0001, 'inlier_fraction': 1.0, 'reprojection_median_px': 0.05}))
    return (nodes, times, edges, truth, ids[anchor])

class RotationChainTests(unittest.TestCase):

    def test_large_rotation_multiple_planar_loops_and_full_solve(self):
        (nodes, times, edges, truth, anchor) = fixture()
        original = copy.deepcopy(nodes)
        cfg = GraphConfig(outer_iterations=3, inner_evaluations=35, observed_weight=0, correction_velocity_rad_s=10, acceleration_rad_s2=100)
        (result, report) = solve(nodes, times, edges, anchor, cfg)
        for (k, n) in enumerate(nodes):
            self.assertLess(Rot.from_matrix(result[n.frame_idx][0] @ truth[k].T).magnitude(), 0.02)
            np.testing.assert_array_equal(n.initial_rotation, original[k].initial_rotation)
            self.assertEqual(n.source_row, original[k].source_row)
        self.assertGreater(np.degrees(Rot.from_matrix(result[nodes[-1].frame_idx][0] @ result[nodes[0].frame_idx][0].T).magnitude()), 120)
        self.assertTrue(all((e['weight_scale'] == 0 for e in report['local_RGB_rotation']['edges'][-4:])))
        self.assertEqual(set(report['stage_snapshots']), {'SO3', 'SE3'})
        np.testing.assert_array_equal(result[anchor][0], original[4].initial_rotation)

    def test_ids_edge_order_direction_and_rate_do_not_change_seed(self):
        results = []
        for count in (9, 17):
            ids = [10000 - k * 117 for k in range(count)]
            (nodes, times, edges, truth, anchor) = fixture(count, ids)
            edges = [replace(e, source_frame_idx=e.target_frame_idx, target_frame_idx=e.source_frame_idx, rotation_source_to_target_world=e.rotation_source_to_target_world.T, translation_source_to_target_world_m=-e.rotation_source_to_target_world.T @ e.translation_source_to_target_world_m) for e in reversed(edges)]
            (seed, _, _, _) = local_rotation_evidence(nodes, times + 37.5, edges, anchor)
            for (n, r) in zip(seed, truth):
                np.testing.assert_allclose(n.initial_rotation, r, atol=1e-12)
            results.append(seed[-1].initial_rotation)
        np.testing.assert_allclose(*results, atol=1e-12)

    def test_gap_bridge_disconnection_and_weak_support_do_not_trust_icp(self):
        (nodes, times, edges, _, anchor) = fixture()
        nodes[3] = replace(nodes[3], metric_observation=False, uncertainty=0.95)
        kept = [e for e in edges[:8] if nodes[3].frame_idx not in (e.source_frame_idx, e.target_frame_idx)]
        kept.append(edges[3])
        d = edges[3].rotation_source_to_target_world @ edges[2].rotation_source_to_target_world
        d = edges[2].rotation_source_to_target_world @ edges[3].rotation_source_to_target_world
        kept.append(RGBEdge(nodes[4].frame_idx, nodes[2].frame_idx, d, np.zeros(3), 0.8))
        (seed, _, _, report) = local_rotation_evidence(nodes, times, kept, anchor)
        self.assertFalse(report['local_disconnected_ids'])
        self.assertGreaterEqual(report['seed_rows'][3]['uncertainty'], 0.95)
        self.assertFalse(seed[3].metric_observation)
        (seed, filtered, _, report) = local_rotation_evidence(nodes, times, [edges[-1]], anchor)
        self.assertGreater(len(report['local_disconnected_ids']), 0)
        self.assertEqual(report['edges'][0]['reason'], 'unverified_weak_or_disconnected_local_path')
        self.assertGreater(filtered[0].weight, 0)
        np.testing.assert_array_equal(seed[0].initial_rotation, nodes[0].initial_rotation)
        weak = [replace(e, diagnostics={**e.diagnostics, 'inlier_fraction': 0.01}) for e in edges[:8]] + [edges[-1]]
        (_, _, _, report) = local_rotation_evidence(nodes, times, weak, anchor)
        self.assertEqual(report['edges'][-1]['reason'], 'unverified_weak_or_disconnected_local_path')

    def test_rejected_edges_and_pixels_share_decision_without_mutation(self):
        (nodes, times, edges, _, anchor) = fixture()
        bad = edges[-1]
        (a, b) = (bad.source_frame_idx, bad.target_frame_idx)
        ids = [n.frame_idx for n in nodes]
        f = RGBReprojectionFactor(a, b, ids.index(a), ids.index(b), np.array([[0.0, 0.0, 1.0]]), np.zeros((1, 2)), np.array([100.0, 100.0, 0.0, 0.0]), np.eye(3), np.zeros(3), np.ones(1), 2.0)
        (seed, filtered, projections, report) = local_rotation_evidence(nodes, times, edges, anchor, [f])
        self.assertEqual(projections[0].edge_weight, 0)
        self.assertEqual(filtered[-1].weight, 0)
        self.assertEqual(f.edge_weight, 2.0)
        self.assertEqual(edges[-1].weight, 2.0)
        self.assertNotIn(len(edges) - 1, report['edges'][-1]['independent_path_edge_indices'])
        poses = {n.frame_idx: (n.initial_rotation, n.initial_translation) for n in seed}

        def residual(fs):
            return np.concatenate([v for (_, v) in residual_blocks(poses, seed, times, filtered, [], fs, GraphConfig())])
        np.testing.assert_array_equal(residual(projections), residual([]))

    def test_planar_surface_retains_translation_but_not_rotation_authority(self):
        (nodes, times, edges, _, anchor) = fixture()
        xy = np.array([[x, y, 0.0] for x in (-0.03, 0, 0.03) for y in (-0.02, 0, 0.02)])
        nodes = [replace(n, observed_world=xy + n.initial_translation) for n in nodes]
        (seed, _, _, chain) = local_rotation_evidence(nodes, times, edges, anchor)
        scales = observed_rotation_authority(seed, chain)
        self.assertTrue(all((v == 0 for v in scales.values())))
        poses = {n.frame_idx: (n.initial_rotation, n.initial_translation) for n in seed}
        f = PointFactor(0, 1, nodes[0].observed_world, nodes[1].observed_world, np.tile([0.0, 0.0, 1.0], (9, 1)), np.ones(9))

        def observed(p):
            blocks = residual_blocks(p, seed, times, [], [f], [], GraphConfig(), False, poses, scales)
            return np.concatenate([v for (_, v) in blocks[2 * len(nodes):2 * len(nodes) + 2]])
        changed = dict(poses)
        (r, t) = changed[nodes[1].frame_idx]
        changed[nodes[1].frame_idx] = (Rot.from_euler('x', 0.2).as_matrix() @ r, t)
        np.testing.assert_allclose(observed(changed), observed(poses), atol=1e-12)
        changed[nodes[1].frame_idx] = (r, t + [0, 0, 0.01])
        self.assertGreater(np.linalg.norm(observed(changed) - observed(poses)), 0.1)
'Timing parser edge cases; real CFR/VFR files are covered by the video tests.'
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from utils.data import _frame_pts, read_cfr_video_timing, validate_package_cfr_times

class VideoTimingTests(unittest.TestCase):

    def probe(self, pts, time_base='1/1000', last_duration=33):
        payload = {'streams': [{'avg_frame_rate': '30/1', 'time_base': time_base, 'width': 640, 'height': 480}], 'frames': [{'pts': p, 'pkt_duration': last_duration} for p in pts]}
        result = subprocess.CompletedProcess([], 0, json.dumps(payload), '')
        with patch('utils.data.subprocess.run', return_value=result):
            return read_cfr_video_timing('fixture.mp4', expected_frame_count=len(pts))

    def test_actual_clock_quantization_not_frame_fraction(self):
        timing = self.probe([1000, 1033, 1067])
        self.assertEqual(timing.frame_rate, Fraction(30))
        self.assertEqual(timing.presentation_times_s[0], 1.0)
        self.assertAlmostEqual(timing.max_grid_error_s, 1 / 3000)
        self.assertAlmostEqual(timing.report()['duration_s'], 0.1)
        with self.assertRaisesRegex(ValueError, 'not CFR'):
            self.probe([1000, 1045, 1067])

    def test_coarse_integral_clock_does_not_allow_a_one_tick_frame_shift(self):
        timing = self.probe([0, 1, 2], '1/30', 1)
        self.assertEqual(timing.max_grid_error_s, 0)
        with self.assertRaisesRegex(ValueError, 'not CFR'):
            self.probe([0, 2, 3], '1/30', 1)

    def test_no_inferred_or_repeated_presentation_timestamps(self):
        with self.assertRaisesRegex(ValueError, 'presentation timestamp'):
            self.probe([0, None, 67])
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            self.probe([0, 0, 67])

    def test_final_frame_duration_cannot_extend_or_shorten_output_silently(self):
        with self.assertRaisesRegex(ValueError, 'final frame duration'):
            self.probe([0, 512, 1024], '1/15360', 700)

    def test_explicit_ffprobe_pts_aliases_not_best_effort(self):
        self.assertEqual(_frame_pts({'pkt_pts': 512}), 512)
        self.assertEqual(_frame_pts({'pts': 512, 'pkt_pts': 512}), 512)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            _frame_pts({'pts': 512, 'pkt_pts': 513})
        with self.assertRaisesRegex(ValueError, 'missing'):
            _frame_pts({'best_effort_timestamp': 512})

    def test_package_tolerance_is_numeric_not_source_clock_resolution(self):
        self.assertLess(validate_package_cfr_times([10.0, 10.033333, 10.066667], Fraction(30)), 1e-06)
        with self.assertRaisesRegex(ValueError, 'constant-rate'):
            validate_package_cfr_times([0.0, 0.045, 2 / 30], Fraction(30))

class ProviderBoundaryTests(unittest.TestCase):

    def test_hawor_never_exposes_shared_raw_to_mutable_cache(self):
        from observations.hand.hawor import private_raw_input
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / 'source.mp4'
            raw.write_bytes(b'private-copy-fixture')
            from observations.hand.hawor import run
            from types import SimpleNamespace
            with patch('observations.hand.hawor.private_raw_input', side_effect=RuntimeError('private raw copy first')):
                with self.assertRaisesRegex(RuntimeError, 'private raw copy first'):
                    run(SimpleNamespace(video_path=raw, run_root=base / 'run'))
            private = private_raw_input(raw, base / 'run')
            self.assertEqual(private.read_bytes(), raw.read_bytes())
            self.assertEqual(private.parent.name, 'hawor')
            private.write_bytes(b'upstream-cache-test')
            self.assertEqual(raw.read_bytes(), b'private-copy-fixture')
            with self.assertRaises(FileExistsError):
                private_raw_input(raw, base / 'run')

    def test_unidepth_metric_radius_uses_exact_offcenter_rays(self):
        from observations.camera.unidepth import metric_radius_to_camera_z
        K = np.array([400.0, 500.0, 0.3, 1.2])
        (height, width) = (3, 4)
        (y, x) = np.mgrid[:height, :width]
        radius = 2 * np.sqrt(((x - K[2]) / K[0]) ** 2 + ((y - K[3]) / K[1]) ** 2 + 1.0)
        result = metric_radius_to_camera_z(radius, K)
        np.testing.assert_allclose(result, 2.0, atol=1e-07)
import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import observations.object.surface as ownership
import observations.object.surface as visible_geometry
from observations.object.surface import CLASS_CODES as CLASS_CODES
from observations.object.surface import MetricEligibility as MetricEligibility
from gates.object.observation import classify_native_depth as classify_native_depth
from observations.object.shape import build_appearance as build_appearance

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def triangle_at_depth(z):
    uv = np.array([[1.0, 1.0], [6.0, 1.0], [1.0, 6.0]])
    return np.c_[uv * z, np.full(3, z)]

class DepthOrderNumericsTest(unittest.TestCase):

    def setUp(self):
        self.mask = np.ones((4, 5), bool)

    def test_front_behind_near_nohit_and_unsupported_preservation(self):
        hand = np.full(self.mask.shape, np.inf)
        depth = np.ones(self.mask.shape)
        hand[0, 0] = 0.8
        hand[0, 1] = 1.2
        hand[0, 2] = 1.003
        hand[0, 3] = 0.8
        supported = np.ones(self.mask.shape, bool)
        supported[0, 3] = False
        (owned, counts, classes) = ownership.front_only(self.mask, hand, depth, tolerance_m=0.005, supported=supported)
        self.assertFalse(owned[0, 0])
        self.assertTrue(owned[0, 1])
        self.assertTrue(owned[0, 2])
        self.assertTrue(owned[0, 3])
        self.assertTrue(owned[1, 1])
        self.assertEqual(counts['removed_front_pixels'], 1)
        self.assertTrue(classes['behind'][0, 1])
        self.assertTrue(classes['near'][0, 2])
        self.assertTrue(classes['unsupported_depth'][0, 3])
        self.assertTrue(classes['no_hit'][1, 1])

    def test_slanted_triangle_uses_perspective_correct_depth(self):
        uv = np.array([[1.0, 1.0], [5.0, 1.0], [1.0, 5.0]])
        z = np.array([1.0, 2.0, 4.0])
        vertices = np.c_[uv * z[:, None], z]
        (raster, _) = first_hit_z(vertices, np.array([[0, 1, 2]]), np.eye(3), (7, 7))
        expected = 1.0 / (0.5 / 1.0 + 0.25 / 2.0 + 0.25 / 4.0)
        self.assertAlmostEqual(raster[2, 2], expected, places=10)
        self.assertNotAlmostEqual(raster[2, 2], 0.5 * 1 + 0.25 * 2 + 0.25 * 4, places=3)

    def test_multiple_hands_take_nearest_first_hit(self):
        K = np.eye(3)
        (nearest, reports) = nearest_hand_first_hits([(triangle_at_depth(2.0), np.array([[0, 1, 2]], np.int32)), (triangle_at_depth(0.7), np.array([[0, 1, 2]], np.int32))], K, (8, 8))
        self.assertAlmostEqual(nearest[2, 2], 0.7)
        self.assertEqual(len(reports), 2)

    def test_near_plane_nan_and_malformed_contracts(self):
        vertices = np.array([[0.0, 0.0, 0.001], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        (_, report) = first_hit_z(vertices, np.array([[0, 1, 2]], np.int32), np.eye(3), (5, 5), near_m=0.01)
        self.assertEqual(report['clipped_faces'], 1)
        vertices[0, 0] = np.nan
        (raster, report) = first_hit_z(vertices, np.array([[0, 1, 2]], np.int32), np.eye(3), (5, 5))
        self.assertEqual(report['nonfinite_faces'], 1)
        self.assertFalse(np.isfinite(raster).any())
        for faces in (np.array([[0, 1, 9]], np.int32), np.array([[0.0, 1.0, 2.0]])):
            with self.assertRaises(ValueError):
                first_hit_z(np.ones((3, 3)), faces, np.eye(3), (5, 5))
        with self.assertRaises(ValueError):
            first_hit_z(np.ones((3, 3)), np.array([[0, 1, 2]], np.int32), np.zeros((3, 3)), (5, 5))

    def test_half_pixel_non_square_depth_sampling(self):
        depth = np.arange(8 * 6).reshape(6, 8)
        A = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
        (sampled, inside) = ownership.sample_depth_at_mask_pixels(depth, A, (3, 4))
        self.assertTrue(inside.all())
        np.testing.assert_array_equal(sampled, depth[np.ix_([0, 2, 4], [0, 2, 4, 6])])

class DepthOrderAdapterTest(unittest.TestCase):

    def test_native_metric_classification_rejects_ambiguity_and_keeps_clear_body_candidates(self):
        mask = np.ones((3, 4), bool)
        depth = np.ones((6, 8), float)
        hand = np.full(depth.shape, np.inf)
        hand[3, 3] = 1.003
        A = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
        (codes, eligible, counts) = classify_native_depth(owned_mask=mask, hand_z=hand, depth_z=depth, A_depth_from_mask=A, complete_hand_geometry=True, validated_hand_absence=False, tolerance_m=0.005, uncertainty_padding_mask_px=0)
        self.assertFalse(eligible[3, 3])
        self.assertEqual(codes[3, 3], CLASS_CODES['near_depth_ambiguous'])
        self.assertGreater(counts['clear_no_hand_overlap_candidate'], 0)
        self.assertTrue(eligible[-2:, -2:].any())
        self.assertEqual(codes[0, 0], CLASS_CODES['invalid_mask_coordinate'])
        self.assertFalse(eligible[0, 0])
        hand[:] = np.inf
        hand[3, 3] = 1.02
        (codes, eligible, _) = classify_native_depth(owned_mask=mask, hand_z=hand, depth_z=depth, A_depth_from_mask=A, complete_hand_geometry=True, validated_hand_absence=False, tolerance_m=0.005, uncertainty_padding_mask_px=1)
        self.assertEqual(codes[3, 4], CLASS_CODES['no_hit_uncertainty_fringe'])
        self.assertFalse(eligible[3, 4])
        self.assertTrue(eligible[4, 6])
        hand[:] = 1.02
        (codes, eligible, _) = classify_native_depth(owned_mask=mask, hand_z=hand, depth_z=depth, A_depth_from_mask=A, complete_hand_geometry=True, validated_hand_absence=False, tolerance_m=0.005, uncertainty_padding_mask_px=1)
        self.assertTrue(eligible[1:-1, 1:-1].all())
        self.assertTrue((codes[1:-1, 1:-1] == CLASS_CODES['hand_clearly_behind']).all())
        depth[3, 3] = np.nan
        (codes, eligible, _) = classify_native_depth(owned_mask=mask, hand_z=hand, depth_z=depth, A_depth_from_mask=A, complete_hand_geometry=True, validated_hand_absence=False, tolerance_m=0.005, uncertainty_padding_mask_px=1)
        self.assertEqual(codes[3, 3], CLASS_CODES['unsupported_depth'])
        self.assertFalse(eligible[3, 3])

    def test_missing_hand_surface_is_never_metric_but_appearance_survives(self):
        mask = np.ones((5, 7), bool)
        (codes, eligible, _) = classify_native_depth(owned_mask=mask, hand_z=np.full(mask.shape, np.inf), depth_z=np.ones(mask.shape), A_depth_from_mask=np.eye(3), complete_hand_geometry=False, validated_hand_absence=False, tolerance_m=0.005, uncertainty_padding_mask_px=4)
        self.assertFalse(eligible.any())
        self.assertTrue((codes == CLASS_CODES['unresolved_hand_geometry']).all())
        self.assertTrue(mask.all())

    def make_contract(self, root, frame_ids):
        contract = {'schema': 'v19_camera_image_transform_contract_v2', 'calibration_authority': 'estimated_from_rgb_depth_model', 'intrinsics_model': 'constant_pinhole_fx_fy_cx_cy', 'intrinsics_coordinate_plane': 'calibration', 'intrinsics_fx_fy_cx_cy': [4.0, 4.0, 3.0, 3.0], 'K': [[4.0, 0.0, 3.0], [0.0, 4.0, 3.0], [0.0, 0.0, 1.0]], 'frame_ids': frame_ids, 'calibration_plane': {'width': 8, 'height': 8}, 'image_planes': {}}
        for name in ('source_rgb', 'sam2_mask'):
            contract['image_planes'][name] = {'plane_name': name, 'width': 8, 'height': 8, 'A_plane_from_calibration': np.eye(3).tolist(), 'transform_kind': 'identity', 'interpolation': 'nearest_for_masks', 'pixel_center_convention': 'integer_pixel_centers_opencv'}
        path = root / 'camera.json'
        path.write_text(json.dumps(contract))
        return path

    def test_preprocessor_drives_owned_track_with_arbitrary_ids(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            frame_ids = [17, 42, 77, 88]
            contract = self.make_contract(root, frame_ids)
            K4 = np.tile(np.array([4.0, 4.0, 3.0, 3.0]), (4, 1))
            depth_path = root / 'depth.npz'
            np.savez(depth_path, frame_idx=frame_ids, depth=np.ones((4, 8, 8), np.float32), confidence=np.ones((4, 8, 8), np.float32), intrinsics_fx_fy_cx_cy=K4, camera_contract_path=str(contract), camera_contract_sha256=digest(contract), camera_contract_plane='source_rgb', A_depth_from_calibration=np.eye(3), intrinsics_source='fixture', calibration_authority='estimated_from_rgb_depth_model', depth_ray_geometry_reprojected=True, metadata_only_ray_relabel_override=False, source_estimated_intrinsics_fx_fy_cx_cy=K4)
            cover_uv = np.array([[-100.0, -100.0], [200.0, -100.0], [-100.0, 200.0]])
            vertices = np.stack([np.c_[cover_uv * z, np.full(3, z)] for z in (0.5, 2.0, 1.003)])
            bridge_path = root / 'bridge.npz'
            np.savez(bridge_path, frame_idx=frame_ids[:3], hand_side=np.array(['left', 'left', 'left']), vertices_current_v18_camera_m=vertices)
            hawor_path = root / 'hawor.npz'
            np.savez(hawor_path, left_faces=np.array([[0, 1, 2]], np.int32), right_faces=np.array([[0, 1, 2]], np.int32))
            track = {}
            frames = []
            for (pos, idx) in enumerate(frame_ids):
                mask_path = root / f'raw_{idx}.png'
                cv2.imwrite(str(mask_path), np.ones((8, 8), np.uint8) * 255)
                track[str(idx)] = {'visible': True, 'mask_path': str(mask_path), 'area_px': 64.0}
                ref = {'bridge_npz': str(bridge_path), 'source_hawor_npz': str(hawor_path), 'bridge_row_index': pos, 'bridge_vertices_camera_array': 'vertices_current_v18_camera_m'}
                rgb_path = root / f'rgb_{idx}.png'
                cv2.imwrite(str(rgb_path), np.full((8, 8, 3), 80 + pos * 20, np.uint8))
                frames.append({'frame_idx': idx, 'raw_frame_path': str(rgb_path), 'camera': {'T_world_camera_metric': np.eye(4).tolist()}, 'hands': [{'hand_side': 'left', 'metric_mano_state': {'vertices_reference': ref}}] if pos < 3 else []})
            base = root / 'base.json'
            base.write_text(json.dumps({'case': 'Case.9', 'frames': frames}))
            raw = root / 'raw_track.json'
            raw.write_text(json.dumps(track))
            out = root / 'owned'
            args = argparse.Namespace(base_annotations=base, raw_sam2_track=raw, depth_npz=depth_path, calibration_contract=contract, mano_bridge=bridge_path, hawor_npz=hawor_path, output_dir=out, case='Case.9', object_id='object-A.2', track_id='track_Z-7', depth_image_plane='source_rgb', mask_image_plane='sam2_mask', depth_order_tolerance_m=0.005, near_m=1e-06, uncertainty_padding_mask_px=4)
            report = ownership.build_depth_order(args)
            owned_track = json.loads((out / 'sam2_track.json').read_text())
            front = cv2.imread(owned_track['17']['mask_path'], 0) > 0
            behind = cv2.imread(owned_track['42']['mask_path'], 0) > 0
            near = cv2.imread(owned_track['77']['mask_path'], 0) > 0
            self.assertLess(front.sum(), 64)
            self.assertEqual(behind.sum(), 64)
            self.assertEqual(near.sum(), 64)
            self.assertGreater(report['totals']['removed_front_pixels'], 0)
            self.assertGreater(report['totals']['preserved_behind_pixels'], 0)
            self.assertEqual(report['object_id'], 'object-A.2')
            eligibility_path = out / 'native_depth_metric_eligibility.json'
            self.assertEqual(report['outputs']['native_depth_metric_eligibility']['sha256'], digest(eligibility_path))
            raw_manifest = root / 'raw_manifest.json'
            raw_manifest.write_text(json.dumps({'frames': [{'frame_idx': idx, 'raw_frame_path': str(root / f'rgb_{idx}.png'), 'source_width': 8, 'source_height': 8} for idx in frame_ids]}))
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'objects': [{'object_id': 'object-A.2', 'track_id': 'track_Z-7', 'description': 'fixture'}]}))

            def p09(frame_idx, destination, *, eligibility=True):
                argv = ['p09', '--case', 'Case.9', '--track-id', 'track_Z-7', '--object-id', 'object-A.2', '--raw-frame-manifest', str(raw_manifest), '--sam2-track-json', str(out / 'sam2_track.json'), '--depth-npz', str(depth_path), '--output-dir', str(destination), '--base-annotations', str(base), '--calibration-contract', str(contract), '--depth-image-plane', 'source_rgb', '--mask-image-plane', 'sam2_mask', '--object-plan', str(plan), '--frame-start', str(frame_idx), '--frame-end', str(frame_idx), '--propose-anchor-candidates-only', '--pixel-stride', '1', '--min-valid-points', '1', '--no-robust-first-surface-depth-ownership', '--no-exclude-hand-bboxes']
                if eligibility:
                    argv.extend(['--metric-eligibility-manifest', str(eligibility_path)])
                with patch.object(sys, 'argv', argv):
                    return visible_geometry.build_visible_geometry(visible_geometry.surface_parse_args())
            proposal = p09(42, root / 'p09_behind')
            self.assertEqual(proposal['status'], 'anchor_candidates_proposed_only')
            proposed = json.loads((root / 'p09_behind/anchor_candidate_proposals.json').read_text())
            eligibility_summary = proposed['ranked_candidates'][0]['native_depth_metric_eligibility']
            self.assertEqual(eligibility_summary['counts']['hand_clearly_behind'], 64)
            with self.assertRaisesRegex(RuntimeError, 'no visible metric geometry'):
                p09(77, root / 'p09_near')
            legacy = p09(77, root / 'p09_near_legacy_default', eligibility=False)
            self.assertEqual(legacy['status'], 'anchor_candidates_proposed_only')
            legacy_proposal = json.loads((root / 'p09_near_legacy_default/anchor_candidate_proposals.json').read_text())
            self.assertGreater(legacy_proposal['ranked_candidates'][0]['valid_depth_mask_pixels'], 0)
            clear = p09(88, root / 'p09_clear_nonoverlap')
            self.assertEqual(clear['status'], 'anchor_candidates_proposed_only')
            clear_proposal = json.loads((root / 'p09_clear_nonoverlap/anchor_candidate_proposals.json').read_text())
            self.assertEqual(clear_proposal['ranked_candidates'][0]['native_depth_metric_eligibility']['counts']['clear_no_hand_overlap_candidate'], 64)
            manifest_bytes = eligibility_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            self.assertGreater(manifest['frames'][0]['counts']['hand_clearly_front'], 0)
            mutations = []
            bad_depth = json.loads(manifest_bytes)
            bad_depth['inputs']['depth_npz']['sha256'] = '0' * 64
            mutations.append(('depth input hash', eligibility_path, json.dumps(bad_depth).encode(), 'depth_npz'))
            bad_affine = json.loads(manifest_bytes)
            bad_affine['frames'][1]['A_depth_from_mask'][0][2] += 0.25
            mutations.append(('depth affine', eligibility_path, json.dumps(bad_affine).encode(), 'affine'))
            raster = Path(manifest['frames'][1]['raster']['path'])
            mutations.append(('eligibility raster', raster, raster.read_bytes() + b'changed', 'raster'))
            owned_mask_path = Path(owned_track['42']['mask_path'])
            changed_owned = cv2.imread(str(owned_mask_path), cv2.IMREAD_GRAYSCALE)
            changed_owned[0, 0] = 0
            changed_owned_path = root / 'changed_owned.png'
            cv2.imwrite(str(changed_owned_path), changed_owned)
            mutations.append(('owned mask', owned_mask_path, changed_owned_path.read_bytes(), 'owned mask'))
            changed_base = json.loads(base.read_text())
            changed_base['mutation'] = True
            mutations.append(('MANO/base lineage', base, json.dumps(changed_base).encode(), 'recorded input'))
            for (number, (label, path, changed, message)) in enumerate(mutations):
                with self.subTest(consumption_mutation=label):
                    original = path.read_bytes()
                    path.write_bytes(changed)
                    try:
                        with self.assertRaisesRegex((ValueError, RuntimeError), message):
                            p09(42, root / f'p09_mutated_{number}')
                    finally:
                        path.write_bytes(original)
            self.assertEqual(eligibility_path.read_bytes(), manifest_bytes)
            self.assertEqual(report['outputs']['owned_sam2_track']['sha256'], digest(out / 'sam2_track.json'))

    def test_corrected_p09_uses_owned_but_appearance_uses_raw(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / 'input').mkdir()
            (root / 'state').mkdir()
            raw_rgb = root / 'input' / 'rgb.png'
            image = np.indices((16, 16)).sum(axis=0).astype(np.uint8) * 8
            cv2.imwrite(str(raw_rgb), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            raw_mask = root / 'input' / 'raw.png'
            owned_mask = root / 'input' / 'owned.png'
            cv2.imwrite(str(raw_mask), np.ones((16, 16), np.uint8) * 255)
            owned_array = np.ones((16, 16), np.uint8) * 255
            owned_array[:, :8] = 0
            cv2.imwrite(str(owned_mask), owned_array)
            raw_track = root / 'input' / 'raw_track.json'
            owned_track = root / 'state' / 'owned_track.json'
            raw_track.write_text(json.dumps({'0': {'visible': True, 'mask_path': str(raw_mask)}}))
            owned_track.write_text(json.dumps({'0': {'visible': True, 'mask_path': str(owned_mask)}}))
            (base, depth, camera) = (root / 'state' / name for name in ('base.json', 'depth.bin', 'camera.json'))
            for path in (base, depth, camera):
                path.write_text(path.name)
            raw_manifest = root / 'input' / 'manifest.json'
            raw_manifest.write_text(json.dumps({'frames': [{'frame_idx': 0, 'raw_frame_path': str(raw_rgb)}]}))
            adapter = {'case': 'caseX', 'object_id': 'objX', 'track_id': 'trackX', 'sam2_track': str(owned_track), 'base_annotations': str(base), 'depth_npz': str(depth), 'calibration_contract': str(camera)}
            annotations = root / 'state' / 'p09.json'
            points = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]]
            annotations.write_text(json.dumps({'v19_visible_geometry_adapter': adapter, 'frames': [{'frame_idx': 0, 'time_s': 0.0, 'camera': {'T_world_camera_metric': np.eye(4).tolist()}, 'objects': [{'object_id': 'objX', 'mask_path': str(owned_mask), 'visible_geometry_candidate': {'world_vertices_sample_m': points, 'camera_vertices_sample_m': points, 'centroid_world_m': [0.03, 0.03, 1.0]}}]}]}))

            def rec(path):
                return {'path': str(path), 'sha256': digest(path)}
            ownership_report = root / 'state' / 'ownership.json'
            ownership_report.write_text(json.dumps({'schema': ownership.depth_order_SCHEMA, 'status': 'ok', 'case': 'caseX', 'object_id': 'objX', 'track_id': 'trackX', 'inputs': {'raw_sam2_track': rec(raw_track), 'base_annotations': rec(base), 'depth_npz': rec(depth), 'calibration_contract': rec(camera)}, 'outputs': {'owned_sam2_track': rec(owned_track)}}))
            out = root / 'measurements' / 'demo_visual' / 'objX' / 'appearance_evidence'
            appearance_args = argparse.Namespace(run_root=root, object_id='objX', case_id='caseX', track_id='trackX', raw_frame_manifest=raw_manifest, annotations=annotations, raw_sam2_track=raw_track, depth_order_report=ownership_report, output_dir=out, min_area_fraction=0, min_border_fraction=0, min_sharpness=0)
            (manifest, _) = build_appearance(appearance_args)
            copied = Path(manifest['frames'][0]['appearance_mask_path'])
            self.assertEqual(digest(copied), digest(raw_mask))
            self.assertNotEqual(digest(copied), digest(owned_mask))
            self.assertEqual(manifest['depth_order_binding']['owned_sam2_track']['path'], str(owned_track))
            self.assertTrue(manifest['raw_appearance_and_metric_eligibility_are_distinct'])
            self.assertIs(manifest['metric_owned_masks_consumed'], False)
            malformed = json.loads(ownership_report.read_text())
            malformed['inputs']['raw_sam2_track']['sha256'] = '0' * 64
            ownership_report.write_text(json.dumps(malformed))
            appearance_args.output_dir = root / 'measurements' / 'demo_visual' / 'objX' / 'bad'
            with self.assertRaisesRegex(ValueError, 'raw SAM2 track path/hash mismatch'):
                build_appearance(appearance_args)

    def test_standalone_wiring_has_no_second_blind_subtraction(self):
        from main import owned_surface_arguments
        args = owned_surface_arguments(Path('run'), 'case', 'object', 'track', Path('result'))
        self.assertIn('--no-exclude-hand-bboxes', args)
        self.assertIn('--metric-eligibility-manifest', args)
        self.assertTrue(args[args.index('--sam2-track-json') + 1].endswith('depth_order_owned/track/sam2_track.json'))
        self.assertNotIn('--exclude-hand-regions', args)

class WorkerImportIsolationTests(unittest.TestCase):

    def test_hawor_import_does_not_mutate_weight_loader_policy(self):
        import importlib, os
        import observations.hand.hawor as hawor
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(hawor)
            self.assertNotIn('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', os.environ)
import sys, unittest, json, tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from observations.object.rgb_tracking import initialize

class RGBInitializer(unittest.TestCase):

    def fixture(self):
        R = Rotation.from_euler('y', 0.7).as_matrix()
        t0 = np.array([0.1, 0.2, 0.8])
        t1 = np.array([0.2, 0.15, 0.6])
        E = t1 - R @ t0
        poses = {'object_id': 'thing', 'pose_rows': [{'frame_idx': i, 'status': 'fit_to_visible_depth_samples', 'rigid_pose_observation_eligible': True, 'rotation_world_from_completed_canonical_matrix': np.eye(3).tolist(), 'translation_world_m': t.tolist(), 'observed_to_mesh_final': {'median_m': 0.001}} for (i, t) in [(0, t0), (3, t0), (7, t0)]]}
        ann = {'frames': [{'frame_idx': i, 'objects': []} for i in [0, 3, 7]]}
        arrays = {'source_frame_idx': np.array([0]), 'target_frame_idx': np.array([3]), 'rotation_source_to_target_world': np.array([R]), 'translation_source_to_target_world_m': np.array([E]), 'accepted': np.array([True])}
        quality = [{'source_frame_idx': 0, 'target_frame_idx': 3, 'status': 'accepted', 'source_3d_conditioning': 0.1, 'inlier_fraction': 1.0, 'reprojection_median_px': 0.1}]
        return (poses, ann, arrays, quality, R, t1)

    def test_relative_motion_not_flattened_and_original_untouched(self):
        (base, ann, arrays, q, R, t) = self.fixture()
        before = json.dumps(base, sort_keys=True)
        out = initialize(base, ann, 'thing', arrays, q, 0)
        rows = {r['frame_idx']: r for r in out['pose_rows']}
        np.testing.assert_allclose(rows[3]['rotation_world_from_completed_canonical_matrix'], R, atol=1e-09)
        np.testing.assert_allclose(rows[3]['translation_world_m'], t, atol=1e-09)
        self.assertEqual(before, json.dumps(base, sort_keys=True))
        self.assertTrue(rows[7]['rgb_initializer']['uncertain'])
        self.assertNotIn('observed_to_mesh_final', rows[3])
        self.assertIn('source_P14_distance_diagnostics', rows[3])

    def test_rejected_edge_not_consumed(self):
        (base, ann, arrays, q, R, t) = self.fixture()
        arrays['accepted'][:] = False
        q[0]['status'] = 'rejected'
        out = initialize(base, ann, 'thing', arrays, q, 0)
        self.assertEqual(out['rgb_full_timeline_initialization']['accepted_edge_count'], 0)
        np.testing.assert_allclose(out['pose_rows'][1]['rotation_world_from_completed_canonical_matrix'], np.eye(3))

    def test_conflicting_acceptance_fails(self):
        (base, ann, arrays, q, _, _) = self.fixture()
        q[0]['status'] = 'rejected'
        with self.assertRaisesRegex(ValueError, 'acceptance mismatch'):
            initialize(base, ann, 'thing', arrays, q, 0)

    def test_invalid_rotation_fails(self):
        (base, ann, arrays, q, _, _) = self.fixture()
        arrays['rotation_source_to_target_world'][:] = 0
        with self.assertRaisesRegex(ValueError, 'invalid world-relative'):
            initialize(base, ann, 'thing', arrays, q, 0)

    def test_quality_nan_rejected(self):
        (base, ann, arrays, q, _, _) = self.fixture()
        q[0]['inlier_fraction'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'quality evidence'):
            initialize(base, ann, 'thing', arrays, q, 0)

    def test_direct_and_latent_rows_survive_P14b_P15_without_promotion(self):
        from types import SimpleNamespace
        from observations.object.signed_geometry import trusted_pose_rows
        from optimization.object.pose import build_pose_rows
        (base, ann, arrays, q, _, _) = self.fixture()
        base['pose_rows'][-1].update(status='interpolated', rigid_pose_observation_eligible=False)
        ann['frames'].append({'frame_idx': 8, 'objects': []})
        out = initialize(base, ann, 'thing', arrays, q, 0)
        self.assertEqual([r['frame_idx'] for r in trusted_pose_rows(out)], [0, 3])
        args = SimpleNamespace(complete_full_timeline_rigid_pose=True, max_rigid_pose_interpolation_gap_frames=10, max_rigid_pose_extrapolation_gap_frames=10)
        (rows, summary) = build_pose_rows(out['pose_rows'], [], np.zeros(0), args, False)
        self.assertEqual([r['frame_idx'] for r in rows], [0, 3, 7, 8])
        self.assertFalse(rows[2]['rigid_pose_observation_eligible'])
        self.assertEqual(rows[3]['status'], 'rgb_initialized_uncertain')
        self.assertEqual(rows, out['pose_rows'])

    def test_actual_p14_schema_reaches_p14b_and_preserves_measurements(self):
        from observations.object.signed_geometry import trusted_pose_rows
        (base, ann, arrays, q, _, _) = self.fixture()
        base.update(method='fit_v18_compact_rigid_object_pose', pose_hypothesis_mesh_labeled='/fresh/pose.ply', frame_fit_rows=base['pose_rows'])
        out = initialize(base, ann, 'thing', arrays, q, 0)
        self.assertEqual(out['inputs']['completed_mesh'], '/fresh/pose.ply')
        self.assertEqual(out['frame_fit_rows'], out['pose_rows'])
        self.assertEqual(out['source_P14_frame_fit_rows'], base['frame_fit_rows'])
        self.assertEqual(len(trusted_pose_rows(out)), 3)
        self.assertFalse(out['pose_rows'][1]['rgb_initializer']['direct_metric_observation_added'])
'RGB source replay must consume the same pixel eligibility as P09.'
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
from observations.object.rgb_tracking import reproduce_source_depth_support
from observations.object.surface import robust_first_surface_depth_ownership

class RGBMetricReplayTests(unittest.TestCase):

    def fixture(self, root):
        mask = np.full((40, 60), 255, np.uint8)
        depth = np.ones(mask.shape)
        allowed = np.ones(mask.shape, bool)
        allowed[10:20, 20:30] = False
        path = root / 'owned.png'
        cv2.imwrite(str(path), mask)
        intr = np.array([100.0, 100.0, 29.5, 19.5])
        settings = dict(mad_sigma=2.5, min_half_width_m=0.03, min_retained_fraction=0.9, fail_raw_to_robust_extent_ratio=2.0, confidence_seed_percentile=95.0, local_depth_step_max_m=0.005, max_removed_distance_inside_mask_px=10.0, max_confidence_flagged_interior_fraction=0.015, min_interior_confidence_flagged_fraction=0.95, max_unexplained_interior_pixels=5, min_component_pixels=20, max_small_component_fraction=0.01)
        (expected, own) = robust_first_surface_depth_ownership(allowed, depth, enabled=True, intrinsics=intr, confidence=np.ones_like(depth), **settings)
        parameters = {'min_depth_m': 0.05, 'max_depth_m': 4.0, 'robust_first_surface_depth_ownership': True, 'native_depth_metric_eligibility_enabled': True, **{'first_surface_' + k: v for (k, v) in settings.items()}}
        manifest = root / 'eligibility.json'
        raster = root / 'eligibility.npz'
        obj = {'visible_geometry_candidate': {'frame_idx': 7, 'source_mask_path': str(path), 'first_surface_depth_ownership': own, 'mask_depth_transform_contract': {'mask_size_wh': [60, 40], 'depth_size_wh': [60, 40], 'A_depth_from_mask_coordinate_model': np.eye(3).tolist()}, 'native_depth_metric_eligibility': {'manifest': str(manifest), 'raster': str(raster)}}}
        gate = SimpleNamespace(path=manifest, frame=Mock(return_value=(allowed, {'raster': str(raster)})))
        return (obj, mask, depth, intr, parameters, gate, expected)

    def test_replay_intersects_before_robust_filtering(self):
        with tempfile.TemporaryDirectory() as td:
            (obj, mask, depth, intr, parameters, gate, expected) = self.fixture(Path(td).resolve())
            result = reproduce_source_depth_support(obj, mask, depth, np.ones_like(depth), intr, parameters, metric_eligibility=gate)
            np.testing.assert_array_equal(result, expected)
            self.assertFalse(result[10:20, 20:30].any())
            gate.frame.assert_called_once()

    def test_declared_eligibility_cannot_be_silently_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            (obj, mask, depth, intr, parameters, _, _) = self.fixture(Path(td).resolve())
            with self.assertRaisesRegex(ValueError, 'eligibility is required'):
                reproduce_source_depth_support(obj, mask, depth, np.ones_like(depth), intr, parameters)
if __name__ == '__main__':
    unittest.main()

from optimization.common.correspondences import first_hit_z, nearest_hand_first_hits
