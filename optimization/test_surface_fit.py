"""Full-triangle ray, differentiable fitting, timeline and provenance tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from observations.object.envelope import build_envelopes
from observations.object.shape import record, write_mesh, mesh_arrays
from optimization.object.surface_fit import (TriangleFirstHit, SurfaceProblem,
    PreparedSurface, fit_surface, temporal_losses, PROFILES)
from gates.object.envelope import validate_envelopes
from gates.object.surface_candidate import validate_surface_candidate
from utils.data import RKEY, TKEY
from utils.geometry import resized_K, torch_rodrigues


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def fixture(root, missing=True):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=.10)
    mesh.vertices += [.08, .02, 0]
    mesh_path = root/'canonical.ply'
    topology = write_mesh(mesh_path, np.asarray(mesh.vertices), np.asarray(mesh.faces))
    v, f = mesh_arrays(mesh_path)
    K = np.array([100., 100., 31.5, 31.5])
    T, R = np.eye(4), np.eye(3)
    actual = np.array([-.08, -.02, .9])
    yy, xx = np.mgrid[:64, :64]; uv = np.c_[xx.ravel(), yy.ravel()]
    ray = TriangleFirstHit(v, f)
    depth, _, valid = ray.cast(uv, K, T, R, actual, 1., v.mean(axis=0))
    points = np.c_[(uv[valid]-K[2:])/K[:2], np.ones(valid.sum())] * depth[valid, None]
    mask = (valid.reshape(64, 64)*255).astype('uint8')
    mask_path = root/'mask.png'; cv2.imwrite(str(mask_path), mask)
    rgb_path = root/'rgb.png'; cv2.imwrite(str(rgb_path), np.zeros((64, 64, 3), 'uint8'))
    ids, times = [37, 104, 9], [.2, .245, .31]
    frames, poses, app = [], [], []
    for i, (idx, time) in enumerate(zip(ids, times)):
        obj = dict(object_id='thing', rigid_pose_observation_eligible=not (missing and i == 2))
        if obj['rigid_pose_observation_eligible']:
            obj['visible_geometry_candidate'] = dict(rigid_pose_observation_eligible=True,
                camera_vertices_sample_m=points.tolist(), world_vertices_sample_m=points.tolist(),
                centroid_world_m=points.mean(axis=0).tolist())
        frames.append(dict(frame_idx=idx, time_s=time, source_width=64, source_height=64,
            camera=dict(T_world_camera_metric=T.tolist(), intrinsics_fx_fy_cx_cy=K.tolist()), objects=[obj]))
        poses.append(dict(frame_idx=idx, time_s=time, **{RKEY:R.tolist(), TKEY:(actual+[0, 0, .03]).tolist()},
            visibility='visible', uncertainty='synthetic_source'))
        app.append(dict(frame_idx=idx, time_s=time, object_id='thing', status='available',
            appearance_mask_path=str(mask_path), appearance_mask_sha256=record(mask_path)['sha256'],
            raw_rgb_path=str(rgb_path), raw_rgb_sha256=record(rgb_path)['sha256'], source_size_wh=[64, 64]))
    annotations = write_json(root/'annotations.json', dict(frames=frames))
    prior = write_json(root/'prior.json', dict(pose_rows=poses))
    posterior = write_json(root/'refined.json', dict(schema='rgb_scene_refined_pose_v1', object_id='thing',
        source_prior=record(prior), source_pose_rows=poses, pose_rows=poses,
        generated_geometry_used_in_pose_estimation=False,
        evidence=dict(annotations_sha256=record(annotations)['sha256'])))
    appearance = write_json(root/'appearance.json', dict(schema='rgb_scene_depth_order_appearance_manifest_v1',
        object_id='thing', source_annotations_sha256=record(annotations)['sha256'],
        metric_owned_masks_consumed=False, frames=app))
    binding = json.loads(posterior.read_text())
    binding.update(schema='rgb_scene_visual_refined_binding_v1', refined_posterior=record(posterior),
        appearance_mask_manifest=record(appearance),
        visual_mesh_receipt=dict(output_mesh_path=str(mesh_path), output_mesh_sha256=record(mesh_path)['sha256'], topology=topology),
        render_mesh_contract=dict(required_mesh_path=str(mesh_path), required_mesh_sha256=record(mesh_path)['sha256'], renderer_must_match=True))
    binding_path = write_json(root/'binding.json', binding)
    envelope = root/'envelope.json'
    build_envelopes(annotations=annotations, mesh_binding=binding_path, refined_posterior=posterior,
        run_root=root, object_id='thing', output=envelope)
    return SurfaceProblem(annotations=annotations, refined_posterior=posterior, mesh_binding=binding_path,
        envelope_report=envelope, appearance=appearance, run_root=root, object_id='thing',
        output_dir=root/'candidate', outer_iterations=4, inner_iterations=15)


class SurfaceFitTests(unittest.TestCase):
    def test_full_triangle_first_near_far_miss_and_similarity(self):
        v = np.array([[-1, -1, 1], [1, -1, 1], [0, 1, 1],
                      [-1, -1, 2], [1, -1, 2], [0, 1, 2.]])
        f = np.array([[3, 4, 5], [0, 1, 2]])  # far triangle comes first in topology
        ray = TriangleFirstHit(v, f)
        K = np.array([1., 1., 0., 0.])
        d, q, valid = ray.cast([[0, 0], [9, 9]], K, np.eye(4), np.eye(3), np.zeros(3), 1., v.mean(0))
        np.testing.assert_allclose(d[0], 1., atol=1e-6)
        np.testing.assert_array_equal(valid, [True, False])
        self.assertTrue(np.isinf(d[1]))
        np.testing.assert_array_equal(q[1], [0, 0, 0])
        d, _, _ = TriangleFirstHit(v[3:], [[0, 1, 2]]).cast([[0, 0]], K, np.eye(4), np.eye(3), np.zeros(3), 1., v[3:].mean(0))
        self.assertAlmostEqual(d[0], 2., places=6)
        pivot = np.array([0., 0., 1.5])
        # Scale about pivot sends the near face to .9, not 1.2.
        d, _, _ = ray.cast([[0, 0]], K, np.eye(4), np.eye(3), np.array([0, 0, .3]), 1.2, pivot)
        self.assertAlmostEqual(d[0], 1.2, places=6)
        T = np.eye(4); T[:3, :3] = Rotation.from_rotvec([.1, .2, -.3]).as_matrix(); T[:3, 3] = [.2, -.4, .1]
        d, _, _ = ray.cast([[0, 0]], K, T, T[:3, :3], T[:3, 3], 1., pivot)
        self.assertAlmostEqual(d[0], 1., places=6)

    def test_half_pixel_and_zero_rotation_gradient(self):
        K = np.array([50, 60, 31.5, 15.5])
        k = resized_K(K, [64, 32], [96, 48])
        np.testing.assert_allclose(k, [75, 90, 47.5, 23.5])
        r = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        R = torch_rodrigues(r)
        R[0, 1, 2].backward()
        self.assertTrue(torch.isfinite(r.grad).all())
        self.assertNotEqual(float(r.grad[0, 0]), 0)

    def test_timestamp_temporal_linear_motion_and_rate(self):
        times = torch.tensor([.2, .245, .31], dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1)
        t = torch.stack([times*.3, times*0, times*0], dim=1)
        z = torch.zeros_like(t)
        vel, acc = temporal_losses(z, t, R, t, z[0], times)
        self.assertAlmostEqual(float(acc), 0, places=20)
        self.assertAlmostEqual(float(vel), 1/3, places=12)
        slower, _ = temporal_losses(z, t, R, t, z[0], times*2)
        self.assertAlmostEqual(float(slower), float(vel)/4, places=12)

    def test_fit_moves_actual_mesh_preserves_topology_sources_and_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = fixture(Path(tmp))
            original = p.refined_posterior.read_bytes()
            original_binding = p.mesh_binding.read_bytes()
            envelope = p.envelope_report.read_bytes()
            prepared = PreparedSurface(p)
            self.assertEqual(prepared.ids, [37, 104, 9])
            self.assertEqual(prepared.direct.tolist(), [1, 1, 0])
            self.assertFalse(prepared.S.flags.writeable)
            for train, heldout in zip(prepared.train_ids, prepared.eval_ids):
                self.assertFalse(set(train) & set(heldout))
            result = fit_surface(p)
            report = json.loads((p.output_dir/'surface_fit_report.json').read_text())
            before = report['before'][0]['first_hit_depth_mae_m']
            after = report['after'][0]['first_hit_depth_mae_m']
            self.assertLess(after, before*.75, (before, after))
            self.assertEqual(report['correspondence_refreshes'], 4)
            self.assertEqual([row['frame_idx'] for row in result['pose_rows']], [37, 104, 9])
            self.assertNotIn('generated_geometry_used_in_pose_estimation', result)
            self.assertTrue(result['generated_mesh_used_as_residual_model'])
            self.assertEqual(p.refined_posterior.read_bytes(), original)
            self.assertEqual(p.mesh_binding.read_bytes(), original_binding)
            self.assertEqual(p.envelope_report.read_bytes(), envelope)
            scaled, faces = mesh_arrays(result['render_mesh_contract']['required_mesh_path'])
            np.testing.assert_array_equal(faces, prepared.F)
            self.assertEqual(scaled.shape, prepared.V.shape)
            self.assertGreater(np.linalg.norm(np.asarray(result['pose_rows'][0][TKEY])-prepared.t0[0]), .005)
            validate_surface_candidate(result, object_id='thing', run_root=Path(tmp))
            bad = copy.deepcopy(result); bad['pose_rows'][0][TKEY][2] += .01
            with self.assertRaisesRegex(ValueError, 'pose/report'):
                validate_surface_candidate(bad, object_id='thing', run_root=Path(tmp))
            bad = copy.deepcopy(result); bad['source_refined_pose_rows'][0][TKEY][2] += .01
            with self.assertRaisesRegex(ValueError, 'source provenance'):
                validate_surface_candidate(bad, object_id='thing', run_root=Path(tmp))
            p.refined_posterior.write_text(original.decode()+' ')
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_surface_candidate(result, object_id='thing', run_root=Path(tmp))

    def test_direct_envelope_independent_of_source_candidate_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = fixture(Path(tmp))
            first = json.loads(p.envelope_report.read_text())
            prior_path = Path(tmp)/'prior.json'
            prior = json.loads(prior_path.read_text())
            prior['pose_rows'][0][TKEY][0] += .2
            prior['pose_rows'][2][TKEY][0] -= .3
            write_json(prior_path, prior)
            posterior = json.loads(p.refined_posterior.read_text())
            posterior.update(source_prior=record(prior_path), source_pose_rows=prior['pose_rows'], pose_rows=prior['pose_rows'])
            write_json(p.refined_posterior, posterior)
            binding = json.loads(p.mesh_binding.read_text())
            binding.update(pose_rows=prior['pose_rows'], source_prior=record(prior_path),
                source_pose_rows=prior['pose_rows'], refined_posterior=record(p.refined_posterior))
            write_json(p.mesh_binding, binding)
            second = build_envelopes(annotations=p.annotations, mesh_binding=p.mesh_binding,
                refined_posterior=p.refined_posterior, run_root=p.run_root,
                object_id=p.object_id, output=Path(tmp)/'second_envelope.json')
            for a, b in zip(first['frames'][:2], second['frames'][:2]):
                for key in ('T_world_surface', 'lower_surface_m', 'upper_surface_m'):
                    self.assertEqual(a[key], b[key])
            self.assertNotEqual(first['frames'][2]['T_world_surface'], second['frames'][2]['T_world_surface'])
            self.assertEqual(second['frames'][2]['optimization_weight'], 0.)

    def test_empty_observations_and_unavailable_masks_do_not_invent_factors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = fixture(Path(tmp))
            ann = json.loads(p.annotations.read_text())
            for frame in ann['frames']:
                frame['objects'][0] = dict(object_id='thing', rigid_pose_observation_eligible=False)
            write_json(p.annotations, ann)
            posterior = json.loads(p.refined_posterior.read_text())
            posterior['evidence']['annotations_sha256'] = record(p.annotations)['sha256']
            write_json(p.refined_posterior, posterior)
            app = json.loads(p.appearance.read_text())
            app['source_annotations_sha256'] = record(p.annotations)['sha256']
            for row in app['frames']:
                row['status'] = 'unavailable_not_substituted'
            write_json(p.appearance, app)
            binding = json.loads(p.mesh_binding.read_text())
            binding.update(refined_posterior=record(p.refined_posterior), appearance_mask_manifest=record(p.appearance))
            write_json(p.mesh_binding, binding)
            from dataclasses import replace
            p = replace(p, envelope_report=Path(tmp)/'empty_envelope.json', outer_iterations=1, inner_iterations=1)
            env = build_envelopes(annotations=p.annotations, mesh_binding=p.mesh_binding,
                refined_posterior=p.refined_posterior, run_root=p.run_root, object_id=p.object_id, output=p.envelope_report)
            self.assertTrue(all(row['source_kind'] == 'unresolved_no_observation' for row in env['frames']))
            result = fit_surface(p)
            self.assertEqual(result['shared_uniform_scale'], 1.)
            np.testing.assert_allclose([r[TKEY] for r in result['pose_rows']], [r[TKEY] for r in posterior['pose_rows']])
            report = json.loads((p.output_dir/'surface_fit_report.json').read_text())
            self.assertEqual(report['trace'][0]['hit_count'], 0)
            self.assertEqual(report['trace'][0]['box'], 0)
            self.assertEqual(report['trace'][0]['silhouette'], 0)

    def test_missing_weight_zero_and_bounds_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = fixture(Path(tmp))
            prepared = PreparedSurface(p)
            self.assertEqual(PROFILES, {'surface_no_box':0., 'surface_weak_box':.15})
            bad = copy.deepcopy(prepared.envelopes)
            bad['frames'][2]['optimization_weight'] = 1.
            with self.assertRaisesRegex(ValueError, 'weight'):
                validate_envelopes(bad, prepared.source)
            bad = copy.deepcopy(prepared.envelopes)
            bad['frames'][0]['lower_surface_m'][0] -= .01
            with self.assertRaisesRegex(ValueError, 'bounds changed'):
                validate_envelopes(bad, prepared.source)
            data = json.loads(p.mesh_binding.read_text()); data['pose_rows'][0][TKEY][2] += .01
            write_json(p.mesh_binding, data)
            with self.assertRaisesRegex(ValueError, 'pose rows differ'):
                PreparedSurface(p)


if __name__ == '__main__':
    unittest.main()
