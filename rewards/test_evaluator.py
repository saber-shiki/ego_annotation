from __future__ import annotations

import unittest
import numpy as np


class SharedFactorReductionTest(unittest.TestCase):
    def test_interval_and_positive_count_normalizers_match_legacy_helpers(self):
        import torch
        from factors.hand.priors import formal_translation, formal_root_orientation, formal_articulation
        from factors.interaction.physics import signed_active_penetration, signed_dense_penetration, visible_surface_depth_order
        translation = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        visibility = torch.tensor([1., .25])
        expected_translation = 2.0 * (translation.square() * visibility[:, None]).sum() / (3 * visibility.sum())
        self.assertTrue(torch.allclose(formal_translation(translation, visibility, 2.0), expected_translation))
        root = translation.reshape(2, 1, 3)
        expected_root = 3.0 * (root.square() * visibility[:, None, None]).sum() / (9 * visibility.sum())
        self.assertTrue(torch.allclose(formal_root_orientation(root, visibility, 3.0), expected_root))
        pose = torch.arange(18., dtype=torch.float32).reshape(2, 3, 3)
        pose_visibility = torch.tensor([[1., 0., .5], [.2, .3, 0.]])
        expected_pose = 4.0 * (pose.square() * pose_visibility[..., None]).sum() / (3 * pose_visibility.sum())
        self.assertTrue(torch.allclose(formal_articulation(pose, pose_visibility, 4.0), expected_pose))
        moved = torch.tensor([[0., 0., 0.], [0., 0., .8], [0., 0., 2.]])
        normals = torch.tensor([[0., 0., 1.]]).expand(3, -1)
        depth = torch.ones(3)
        residual = torch.relu(depth - .1 - moved[:, 2])
        self.assertTrue(torch.allclose(signed_active_penetration(moved, normals, depth, .1, 5.),
                                       5. * residual.square().mean()))
        self.assertTrue(torch.allclose(signed_dense_penetration(moved, normals, depth, .1, 5.),
                                       5. * residual.square().sum() / (residual > 0).sum()))
        camera_depth = torch.tensor([.5, 1., 1.5])
        surface_depth = torch.ones(3)
        order = torch.relu(surface_depth - .1 - camera_depth)
        self.assertTrue(torch.allclose(visible_surface_depth_order(camera_depth, surface_depth, .1, 7.),
                                       7. * order.square().sum() / (order > 0).sum()))


if __name__ == "__main__":
    unittest.main()
