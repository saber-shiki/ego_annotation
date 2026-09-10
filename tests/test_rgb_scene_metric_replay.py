"""RGB source replay must consume the same pixel eligibility as P09."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_rgb_scene_edges import reproduce_source_depth_support
from build_v19_visible_geometry_from_sam2_depth import robust_first_surface_depth_ownership


class RGBMetricReplayTests(unittest.TestCase):
    def fixture(self, root):
        mask = np.full((40, 60), 255, np.uint8)
        depth = np.ones(mask.shape)
        allowed = np.ones(mask.shape, bool)
        allowed[10:20, 20:30] = False
        path = root / 'owned.png'; cv2.imwrite(str(path), mask)
        intr = np.array([100., 100., 29.5, 19.5])
        settings = dict(mad_sigma=2.5, min_half_width_m=.03, min_retained_fraction=.9,
                        fail_raw_to_robust_extent_ratio=2., confidence_seed_percentile=95.,
                        local_depth_step_max_m=.005, max_removed_distance_inside_mask_px=10.,
                        max_confidence_flagged_interior_fraction=.015,
                        min_interior_confidence_flagged_fraction=.95, max_unexplained_interior_pixels=5,
                        min_component_pixels=20, max_small_component_fraction=.01)
        expected, own = robust_first_surface_depth_ownership(
            allowed, depth, enabled=True, intrinsics=intr, confidence=np.ones_like(depth), **settings)
        parameters = {'min_depth_m':.05, 'max_depth_m':4., 'robust_first_surface_depth_ownership':True,
                      'native_depth_metric_eligibility_enabled':True,
                      **{'first_surface_' + k:v for k,v in settings.items()}}
        manifest = root / 'eligibility.json'; raster = root / 'eligibility.npz'
        obj = {'visible_geometry_candidate':{'frame_idx':7, 'source_mask_path':str(path),
               'first_surface_depth_ownership':own,
               'mask_depth_transform_contract':{'mask_size_wh':[60,40], 'depth_size_wh':[60,40],
                    'A_depth_from_mask_coordinate_model':np.eye(3).tolist()},
               'native_depth_metric_eligibility':{'manifest':str(manifest),'raster':str(raster)}}}
        gate = SimpleNamespace(path=manifest, frame=Mock(return_value=(allowed, {'raster':str(raster)})))
        return obj, mask, depth, intr, parameters, gate, expected

    def test_replay_intersects_before_robust_filtering(self):
        with tempfile.TemporaryDirectory() as td:
            obj, mask, depth, intr, parameters, gate, expected = self.fixture(Path(td))
            result = reproduce_source_depth_support(obj, mask, depth, np.ones_like(depth), intr,
                                                    parameters, metric_eligibility=gate)
            np.testing.assert_array_equal(result, expected)
            self.assertFalse(result[10:20,20:30].any())
            gate.frame.assert_called_once()

    def test_declared_eligibility_cannot_be_silently_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            obj, mask, depth, intr, parameters, _, _ = self.fixture(Path(td))
            with self.assertRaisesRegex(ValueError, 'eligibility is required'):
                reproduce_source_depth_support(obj, mask, depth, np.ones_like(depth), intr, parameters)


if __name__ == '__main__':
    unittest.main()
