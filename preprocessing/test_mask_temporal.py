"""Temporal factors constrain correction discontinuity, never absolute rotation."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from preprocessing.geometry import TriangleMesh
from preprocessing.mask_contract import FixedMaskDomain
from preprocessing.mask_repair import FrameRepair
from preprocessing.test_visible_surface import fixture


class MaskTemporalTests(unittest.TestCase):
    def fixture(self):
        inp, R, t, hd, truth = fixture(scale=1.)
        mesh = TriangleMesh(inp.vertices, inp.faces)
        fit = FrameRepair(inp, mesh, FixedMaskDomain.from_inputs(inp, hd, 0), 0, R[0], t[0])
        fit.neighbor_corrections = [(np.eye(3), np.zeros(3), 1.), (np.eye(3), np.zeros(3), 1.)]
        return fit

    def test_isolated_symmetry_flip_has_temporal_cost(self):
        fit = self.fixture()
        flip = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        self.assertEqual(fit.temporal_cost(fit.R0, fit.t0), 0.)
        self.assertGreater(fit.temporal_cost(flip @ fit.R0, fit.t0), 1.)

    def test_coherent_large_rotation_correction_is_not_clamped(self):
        fit = self.fixture()
        flip = Rotation.from_euler('z', 120, degrees=True).as_matrix()
        fit.neighbor_corrections = [(flip, np.zeros(3), 1.), (flip, np.zeros(3), 1.)]
        self.assertLess(fit.temporal_cost(flip @ fit.R0, fit.t0), 1e-25)

    def test_absolute_large_baseline_rotation_has_zero_correction_cost(self):
        fit = self.fixture()
        fit.world_R0 = fit.R0 = Rotation.from_euler('z', 140, degrees=True).as_matrix()
        self.assertLess(fit.temporal_cost(fit.R0, fit.t0), 1e-25)

    def test_temporal_fit_changes_mesh_without_losing_mask_reference(self):
        fit = self.fixture()
        _, _, report = fit.solve(outer_iterations=5, max_nfev=30)
        self.assertGreater(report['after']['mask']['visible']['iou'], report['before']['mask']['visible']['iou'])
        for row in report['iterations']:
            for trial in row['trials']:
                if trial['accepted']:
                    self.assertGreater(trial['joint_gain'], 0.)
        self.assertGreater(report['accepted_updates'], 0)

    def test_zero_update_second_sweep_keeps_prior_valid_pose(self):
        fit = self.fixture()
        R, t, first = fit.solve()
        R2, t2, second = fit.solve(outer_iterations=0, initial_pose=(R, t))
        np.testing.assert_array_equal(R2, R)
        np.testing.assert_array_equal(t2, t)
        self.assertEqual(first['after']['mask'], second['after']['mask'])


if __name__ == '__main__':
    unittest.main()
