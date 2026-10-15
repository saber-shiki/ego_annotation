"""Actual-triangle and fixed-population regressions for the opt-in repair."""
import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from preprocessing.geometry import TriangleMesh
from preprocessing.mask_contract import FixedMaskDomain, nonregression
from preprocessing.mask_repair import FrameRepair
from preprocessing.test_visible_surface import fixture
from preprocessing.visible_surface import cloud_digest


class MaskContractTests(unittest.TestCase):
    def setUp(self):
        self.target = np.zeros((64, 64), bool)
        self.target[16:48, 16:48] = True
        self.hand = np.full((64, 64), np.inf)
        self.domain = FixedMaskDomain.create(self.target, self.target, self.hand)

    def test_fixed_domain_rejects_background_spill(self):
        before = self.domain.score(self.target)
        mask = self.target.copy(); mask[10:54, 10:54] = True
        after = self.domain.score(mask)
        self.assertFalse(nonregression(before, after)[0])
        self.assertGreater(after['visible']['spill_pixels'], 0)

    def test_hand_occlusion_does_not_require_amodal_containment(self):
        visible = self.target.copy(); visible[:, 32:] = False
        self.hand[:, 32:] = .3
        domain = FixedMaskDomain.create(visible, visible, self.hand)
        full = domain.score(self.target)
        self.assertLess(full['raw']['iou'], .6)
        self.assertEqual(full['visible']['iou'], 1.)
        self.assertTrue(nonregression(domain.score(visible), full)[0])

    def test_owned_evidence_overrides_hand_and_is_immutable(self):
        self.hand[:] = .3
        domain = FixedMaskDomain.create(self.target, self.target, self.hand)
        self.assertEqual(domain.score(self.target)['visible']['iou'], 1.)
        self.assertTrue(domain.domain[self.target].all())
        self.assertFalse(domain.target.flags.writeable)
        self.target[:] = False
        self.assertTrue(domain.target.any())

    def test_population_change_is_rejected(self):
        other = FixedMaskDomain.create(self.target, self.target, np.full((64, 64), .3))
        self.assertFalse(nonregression(self.domain.score(self.target), other.score(self.target))[0])

    def test_tolerance_cannot_accumulate(self):
        reference = self.domain.score(self.target)
        first, second = copy.deepcopy(reference), copy.deepcopy(reference)
        first['visible']['iou'] -= .01
        second['visible']['iou'] -= .02
        self.assertTrue(nonregression(reference, first)[0])
        self.assertTrue(nonregression(first, second)[0])
        self.assertFalse(nonregression(reference, second)[0])

    def test_nonfinite_and_empty_candidates_fail_closed(self):
        before = self.domain.score(self.target)
        after = copy.deepcopy(before); after['visible']['recall'] = float('nan')
        self.assertFalse(nonregression(before, after)[0])
        self.assertFalse(nonregression(before, self.domain.score(np.zeros_like(self.target)))[0])
        empty = FixedMaskDomain.create(np.zeros_like(self.target), np.zeros_like(self.target), self.hand)
        self.assertFalse(nonregression(empty.score(self.target), empty.score(self.target))[0])

    def test_raw_positive_support_under_hand_is_not_erased(self):
        self.hand[:, 32:] = .3
        owned = self.target.copy(); owned[:, 32:] = False
        domain = FixedMaskDomain.create(self.target, owned, self.hand)
        before, after = domain.score(self.target), domain.score(owned)
        self.assertEqual(before['visible']['iou'], after['visible']['iou'])
        self.assertEqual(nonregression(before, after)[1], 'raw_recall_regressed')

    def test_missing_boundary_is_unavailable_not_perfect(self):
        full = np.ones((64, 64), bool)
        domain = FixedMaskDomain.create(full, full, self.hand)
        self.assertIsNone(domain.score(full)['boundary']['recall'])


class FrameRepairTests(unittest.TestCase):
    def test_translation_repair_changes_real_triangle_projection(self):
        inp, R, t, hd, truth = fixture(scale=1.)
        t = truth.copy(); t[0, 0] += .015
        mesh = TriangleMesh(inp.vertices, inp.faces)
        domain = FixedMaskDomain.from_inputs(inp, hd, 0)
        digest = cloud_digest(inp.clouds)
        repair = FrameRepair(inp, mesh, domain, 0, R[0], t[0])
        got_R, got_t, report = repair.solve(outer_iterations=6, max_nfev=30)
        self.assertGreater(report['after']['mask']['visible']['iou'], report['before']['mask']['visible']['iou'] + .05)
        self.assertLess(abs(got_t[0] - truth[0, 0]), .01)
        self.assertTrue(nonregression(report['before']['mask'], report['after']['mask'])[0])
        self.assertEqual(cloud_digest(inp.clouds), digest)
        np.testing.assert_allclose(got_R @ got_R.T, np.eye(3), atol=1e-10)

    def test_large_supported_rotation_is_not_clamped(self):
        inp, R, t, hd, truth = fixture(scale=1.)
        real = Rotation.from_euler('z', 100, degrees=True).as_matrix()
        mesh = TriangleMesh(inp.vertices, inp.faces)
        X, z, mask = mesh.raster(inp.K[0], real, truth[0], inp.wh)
        inp.masks[0] = mask; inp.owned_masks[0] = mask
        inp.clouds[0] = X[mask] @ real.T + truth[0]
        repair = FrameRepair(inp, mesh, FixedMaskDomain.from_inputs(inp, hd, 0), 0, real, truth[0])
        got_R, _, report = repair.solve()
        self.assertGreater(np.degrees(Rotation.from_matrix(got_R).magnitude()), 95)
        self.assertEqual(report['after']['mask']['visible']['iou'], 1.)

    def test_absent_depth_cannot_be_replaced_by_silhouette_depth(self):
        inp, R, t, hd, truth = fixture(scale=1.)
        inp.clouds[0] = np.empty((0, 3)); inp.cloud_valid[0] = False
        mesh = TriangleMesh(inp.vertices, inp.faces)
        repair = FrameRepair(inp, mesh, FixedMaskDomain.from_inputs(inp, hd, 0), 0, R[0], t[0])
        _, got_t, report = repair.solve()
        self.assertEqual(got_t[2], t[0, 2])
        self.assertTrue(report['missing_depth_frozen'])

    def test_aggregate_mask_gain_cannot_override_surface_regression(self):
        inp, R, t, hd, truth = fixture(scale=1.)
        mesh = TriangleMesh(inp.vertices, inp.faces)
        repair = FrameRepair(inp, mesh, FixedMaskDomain.from_inputs(inp, hd, 0), 0, R[0], t[0])
        candidate = copy.deepcopy(repair.reference)
        candidate['surface']['mean_m'] += .1
        self.assertEqual(repair.admissible(repair.reference, candidate)[1], 'source_surface_regressed')


if __name__ == '__main__':
    unittest.main()
