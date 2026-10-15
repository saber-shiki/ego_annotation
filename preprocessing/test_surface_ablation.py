"""Mechanism tests: observation-driven motion, open envelope, full-mesh coupling."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from preprocessing.geometry import project
from preprocessing.material_motion import (Tracks, neighborhood_lift, rigid_fit, robust_rigid,
    MaterialMotionProblem, propagated_rotation, fixed_track_score)
from preprocessing.surface_constraints import observed_envelope, SurfaceFrame
from preprocessing.surface_ablation import JointSurfaceProblem, validate_rotations
from preprocessing.test_visible_surface import fixture


def material_fixture():
    rng = np.random.default_rng(47); n = 4; points = rng.uniform(-.07, .07, (16, 3))
    points[:, 2] *= .2
    R = Rotation.from_euler('xyz', [[0, 0, 0], [12, 18, 40], [28, 30, 85], [30, 50, 115]], degrees=True).as_matrix()
    t = np.c_[np.linspace(0, .04, n), np.linspace(0, -.01, n), np.full(n, .7)]
    C = np.repeat(np.eye(4)[None], n, axis=0)
    C[1:, :3, :3] = Rotation.from_euler('y', np.array([[4], [8], [12]]), degrees=True).as_matrix()
    C[:, :3, 3] = np.c_[np.linspace(0, .012, n), np.zeros(n), np.zeros(n)]
    k = np.tile([350., 350., 127.5, 127.5], (n, 1))
    tr = np.repeat(np.arange(len(points)), n); fr = np.tile(np.arange(n), len(points))
    world = np.einsum('nij,nj->ni', R[fr], points[tr]) + t[fr]
    q = np.einsum('nji,nj->ni', C[fr, :3, :3], world-C[fr, :3, 3])
    uv = project(q, k[fr]); normal = np.zeros_like(q); normal[:, 2] = -1
    heldout = np.arange(len(points)) % 5 == 0
    tracks = Tracks(tr, fr, uv, q, normal, np.zeros(len(fr)), heldout, len(points), {})
    inputs = SimpleNamespace(cameras=C, K=k, anchor=0, ids=np.arange(n), times=np.arange(n)/30.)
    return inputs, tracks, R, t, points


class MaterialMotionTests(unittest.TestCase):
    def test_inherited_float_precision_not_fabricated_rotation_failure(self):
        R = Rotation.from_euler('xyz', [[13, 27, 46], [34, 52, 79]], degrees=True).as_matrix().astype('float32').astype(float)
        self.assertLess(validate_rotations(R), 1e-6)
        bad = R.copy(); bad[0, 0, 0] += .001
        with self.assertRaises(ValueError):
            validate_rotations(bad)

    def test_rigid_fit_recovers_large_proper_rotation_with_outliers(self):
        rng = np.random.default_rng(48); p = rng.normal(0, .06, (40, 3))
        r = Rotation.from_euler('xyz', [25, 45, 110], degrees=True).as_matrix(); t = np.array([.1, -.04, .02])
        q = p@r.T+t; q[:8] += .2
        recovered, translation, good, _ = robust_rigid(p, q)
        np.testing.assert_allclose(recovered, r, atol=1e-8); np.testing.assert_allclose(translation, t, atol=1e-8)
        self.assertEqual(int(good.sum()), 32); self.assertGreater(np.linalg.det(recovered), .9999)

    def test_collinear_material_geometry_not_certified(self):
        p = np.c_[np.arange(8), np.zeros((8, 2))]*.01
        with self.assertRaises(ValueError):
            rigid_fit(p, p)
        self.assertIsNone(robust_rigid(p, p))

    def test_plane_lift_uses_source_ray_not_mesh(self):
        yy, xx = np.mgrid[-.04:.041:.004, -.04:.041:.004]
        q = np.c_[xx.ravel(), yy.ravel(), .7+xx.ravel()*.2]
        k = np.array([350., 350., 127.5, 127.5]); uv = np.array([[127.5, 127.5], [134., 123.]])
        observed, normal, quality = neighborhood_lift(q, uv, k, radius_px=7.)
        np.testing.assert_allclose(project(observed, k), uv, atol=1e-8)
        self.assertGreater(quality.min(), .8)
        self.assertTrue(np.all(np.sum(normal*observed, axis=1)<0))
        missing, _, score = neighborhood_lift(np.empty((0, 3)), uv, k)
        self.assertTrue(np.isnan(missing).all()); self.assertEqual(score.sum(), 0)

    def test_observation_bundle_recovers_motion_and_shared_material_points(self):
        inp, tracks, truth_R, truth_t, _ = material_fixture()
        edges = [{'a': 0, 'b': i, 'R': truth_R[i], 'weight': 1.} for i in range(1, 4)]
        initial_R, reached = propagated_rotation(np.tile(np.eye(3), (4, 1, 1)), 0, edges)
        self.assertEqual(len(reached), 4)
        initial_t = truth_t.copy(); initial_t[1:, :2] += .025
        problem = MaterialMotionProblem(inp, tracks.selected(False), initial_R, initial_t, edges)
        R, t, report = problem.solve(max_nfev=80)
        errors = Rotation.from_matrix(R@truth_R.transpose(0, 2, 1)).magnitude()
        self.assertLess(np.degrees(errors).max(), 3.)
        np.testing.assert_allclose(t, truth_t, atol=.003)
        self.assertLess(report['cost_after'], report['cost_before']*.05)
        np.testing.assert_array_equal(R[0], initial_R[0])

    def test_fixed_population_prefers_correct_motion(self):
        inp, tracks, R, t, _ = material_fixture()
        correct = fixed_track_score(inp, tracks, R, t)
        wrong = fixed_track_score(inp, tracks, np.tile(np.eye(3), (4, 1, 1)), t)
        self.assertEqual(correct['observations'], wrong['observations'])
        self.assertLess(correct['cost'], wrong['cost']*.01)
        self.assertTrue(correct['population_fixed'])

    def test_missing_depth_remains_2D_not_filled(self):
        inp, tracks, R, t, _ = material_fixture(); tracks.camera_points[tracks.frame==2] = np.nan
        p = MaterialMotionProblem(inp, tracks, R, t)
        self.assertEqual(len(p.metric_ix), 48)
        self.assertEqual(len(p.observations(R, t, p.points0)[0]), 128)
        self.assertTrue(np.isnan(tracks.camera_points[tracks.frame==2]).all())

    def test_sparse_dependencies_cover_finite_differences(self):
        inp, tracks, R, t, _ = material_fixture(); p = MaterialMotionProblem(inp, tracks, R, t)
        x = p.initial(); base = p.residual(x); J = p.sparsity().toarray()
        for col in [0, 3, 8, p.point_start, p.size-1]:
            q = x.copy(); q[col] += 1e-6
            changed = np.abs(p.residual(q)-base)>1e-10
            self.assertTrue(np.all(J[changed, col]>0))


class SurfaceEnvelopeTests(unittest.TestCase):
    def test_envelope_does_not_depend_on_existing_pose(self):
        inp, R, t, hd, truth = fixture(scale=1.)
        e = observed_envelope(inp.clouds[0], inp.cameras[0], inp.vertices, inp.faces)
        self.assertTrue(e.observed)
        self.assertGreater(np.linalg.det(e.basis_world), .999)
        self.assertLess(np.linalg.norm(e.residual(inp.clouds[0]), axis=1).mean(), .0001)
        self.assertGreater(e.residual(inp.clouds[0]+np.array([1., 0, 0])).max(), .5)
        empty = observed_envelope([], inp.cameras[0], inp.vertices, inp.faces)
        self.assertFalse(empty.observed); self.assertEqual(empty.confidence, 0.)

    def test_full_triangle_surface_and_box_terms_respond_to_pose(self):
        from preprocessing.geometry import TriangleMesh
        inp, R, t, hd, truth = fixture(scale=1.)
        mesh = TriangleMesh(inp.vertices, inp.faces); f = SurfaceFrame(inp, mesh, hd, 0)
        cor = f.correspondences(R[0], truth[0])
        before = f.residual(R[0], truth[0], cor, 0.)
        after = f.residual(R[0], truth[0]+np.array([0., 0., .05]), cor, 0.)
        self.assertGreater(np.linalg.norm(after), np.linalg.norm(before)+.5)
        box = f.residual(R[0], truth[0]+np.array([.3, 0., 0.]), cor, .15)
        self.assertEqual(len(box)-len(before), 256*3)
        self.assertGreater(np.linalg.norm(box[-256*3:]), .5)

    def test_cached_queries_preserve_exact_surface_scores(self):
        from preprocessing.geometry import TriangleMesh
        inp, R, t, hd, truth = fixture(scale=1.)
        mesh = TriangleMesh(inp.vertices, inp.faces); f = SurfaceFrame(inp, mesh, hd, 0)
        direct = f.measure(R[0], truth[0])
        view = f.view(R[0], truth[0]); cor = f.correspondences(R[0], truth[0], view)
        cached = f.measure(R[0], truth[0], cor, view)
        self.assertEqual(direct, cached)
        dims_before = mesh._envelope_dimensions.copy()
        SurfaceFrame(inp, mesh, hd, 0)
        np.testing.assert_array_equal(dims_before, mesh._envelope_dimensions)

    def test_joint_mesh_and_latent_sparsity(self):
        from preprocessing.geometry import TriangleMesh
        inp, R, t, hd, truth = fixture(scale=1., n=3); inp.anchor=0
        frames = np.tile(np.arange(3), 6); ids = np.repeat(np.arange(6), 3)
        xyz = np.array([[x, y, -.04] for x in [-.04, 0, .04] for y in [-.02, .02]])
        q = xyz[ids]+truth[frames]
        tracks = Tracks(ids, frames, project(q, inp.K[frames]), q, np.zeros_like(q), np.zeros(len(q)), np.zeros(6, bool), 6, {})
        mesh = TriangleMesh(inp.vertices, inp.faces); surfaces = [SurfaceFrame(inp, mesh, hd, i) for i in range(3)]
        cor = [f.correspondences(R[i], truth[i]) for i, f in enumerate(surfaces)]
        p = JointSurfaceProblem(inp, tracks, R, truth, [], surfaces, cor, .15)
        x = p.pack_pose(R, truth); res = p.residual(x); J = p.sparsity()
        self.assertEqual(J.shape, (len(res), p.size))
        shifted = x.copy(); shifted[3] += .01
        self.assertGreater(np.linalg.norm(p.residual(shifted)-res), .1)


if __name__ == '__main__':
    unittest.main()
