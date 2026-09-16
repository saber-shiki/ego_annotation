"""Mechanism tests for frozen observation frames and containment mathematics."""
import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from observations.object.envelope import local_normals, extract_planes, descriptive_frame, direct_bounds
from factors.object.envelope import (local_coordinates, outside_residual_m,
    conditional_translation_interval, scaled_vertices, robust_box_loss)


class EnvelopeMathTests(unittest.TestCase):
    def test_free_angle_plane_camera_sign_and_bounds(self):
        x, y = np.meshgrid(np.linspace(-.04, .04, 15), np.linspace(-.10, .10, 22))
        R = Rotation.from_rotvec([.24, -.31, .17]).as_matrix()
        p = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)] @ R.T + [0, 0, .8]
        normals, _ = local_normals(p)
        planes, labels = extract_planes(p, normals, np.random.default_rng(714))
        self.assertEqual(len(planes), 1)
        S, descriptor = descriptive_frame(p, planes, labels)
        self.assertIn('observed_plane', descriptor['method'])
        self.assertLess(S[:3, 2] @ S[:3, 3], 0)
        np.testing.assert_allclose(S[:3, :3].T @ S[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(S[:3, :3]), 1, atol=1e-12)
        self.assertLess(np.max(abs(local_coordinates(p, S)[:, 2])), 1e-12)
        lo, hi, _ = direct_bounds(p, S, [.24, .12, .05])
        self.assertEqual(robust_box_loss(p, S, lo, hi), 0.)
        self.assertGreater(robust_box_loss(p + .3*S[:3, 0], S, lo, hi), 0)
        np.testing.assert_array_equal([lo, hi], direct_bounds(p, S, [.24, .12, .05])[:2])

    def test_pca_fallback_explicit(self):
        p = np.array([[0, 0, .7], [.1, 0, .7], [0, .2, .7]])
        S, descriptor = descriptive_frame(p, [], np.full(3, -1))
        self.assertIn('fallback_not_planar_evidence', descriptor['method'])
        self.assertTrue(np.isfinite(S).all())

    def test_inverse_not_transpose_and_residual_sign(self):
        S = np.eye(4); S[:3, 3] = [.1, -.2, .3]
        S[0, 0] += 1e-6
        q = np.array([[-2, 0, 0], [0, 0, 0], [0, 3, 0.]])
        p = q @ S[:3, :3].T + S[:3, 3]
        np.testing.assert_allclose(local_coordinates(p, S), q, atol=1e-14)
        np.testing.assert_allclose(outside_residual_m(p, S, -np.ones(3), np.ones(3)),
                                   [[-1, 0, 0], [0, 0, 0], [0, 2, 0]], atol=1e-14)

    def test_full_vertex_interval_pivot_and_empty_interval(self):
        v = np.array([[1, 2, 3], [1.2, 2.1, 3.3], [.8, 2.2, 3.1]])
        pivot = v.mean(axis=0)
        vs = scaled_vertices(v, pivot, 1.3)
        np.testing.assert_allclose(vs.mean(axis=0), pivot)
        R = Rotation.from_rotvec([.2, -.1, .4]).as_matrix()
        S = np.eye(4); S[:3, :3] = Rotation.from_rotvec([-.3, .1, 0]).as_matrix()
        S[:3, 3] = [.2, -.3, .5]
        lo, hi = -np.ones(3), np.ones(3)
        low, high = conditional_translation_interval(vs, R, S, lo, hi)
        t = S[:3, 3] + S[:3, :3] @ ((low+high)/2)
        np.testing.assert_allclose(outside_residual_m(vs @ R.T + t, S, lo, hi), 0, atol=1e-14)
        low, high = conditional_translation_interval(vs, R, S, -np.ones(3)*.001, np.ones(3)*.001)
        self.assertTrue(np.any(low > high))


if __name__ == '__main__':
    unittest.main()
