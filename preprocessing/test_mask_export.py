"""Export checks use the actual saved world pose, including camera round trips."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from preprocessing.geometry import TriangleMesh
from preprocessing.mask_contract import FixedMaskDomain
from preprocessing.mask_repair import FrameRepair
from preprocessing.test_visible_surface import fixture


class MaskExportTests(unittest.TestCase):
    def make_repair(self):
        inp, R, t, hd, truth = fixture(scale=1.)
        C = inp.cameras[0]
        C[:3, :3] = Rotation.from_euler('xyz', [13, 7, 2], degrees=True).as_matrix()
        C[:3, 3] = [.03, .04, -.1]
        wr = C[:3, :3] @ R[0]
        wt = C[:3, :3] @ t[0] + C[:3, 3]
        mesh = TriangleMesh(inp.vertices, inp.faces)
        domain = FixedMaskDomain.from_inputs(inp, hd, 0)
        return FrameRepair(inp, mesh, domain, 0, wr, wt)

    def test_unchanged_pose_preserves_exact_world_seed(self):
        fit = self.make_repair()
        with patch('preprocessing.mask_repair.world_pose') as convert:
            wr, wt, report, reason = fit.export_checked(fit.R0, fit.t0, 0)
        convert.assert_not_called()
        np.testing.assert_array_equal(wr, fit.world_R0)
        np.testing.assert_array_equal(wt, fit.world_t0)
        self.assertEqual(reason, 'seed_unchanged')

    def test_export_side_corruption_cannot_bypass_mask_guard(self):
        fit = self.make_repair()
        with patch('preprocessing.mask_repair.world_pose', return_value=(fit.world_R0, fit.world_t0 + [1., 0., 0.])):
            wr, wt, report, reason = fit.export_checked(fit.R0, fit.t0, 1)
        np.testing.assert_array_equal(wr, fit.world_R0)
        np.testing.assert_array_equal(wt, fit.world_t0)
        self.assertTrue(reason.startswith('export_rejected:'))
        self.assertEqual(report, fit.reference)

    def test_exported_report_matches_independent_world_pose_replay(self):
        fit = self.make_repair()
        wr, wt, report = fit.solve()
        replay = fit.domain.render_score(fit.mesh, wr, wt, fit.inputs, fit.i)
        self.assertEqual(report['after']['mask'], replay)
        self.assertIn(report['export_verification'], ('seed_unchanged', 'exported_pose_verified'))


if __name__ == '__main__':
    unittest.main()
