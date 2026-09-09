"""CPU tests for full-geometry demo detail framing, not geometry substitution."""
import copy
from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from render_v20_visual_contact_demo import (
    detail_view_title, detail_viewports, scene_geometry_for_frame,
)
from v20_demo_geometry import project


def fixture():
    vertices = np.array(list(product([-.22, .22], [-.9, .9], [-.15, .15])))
    faces = np.array([[0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7],
                      [0, 4, 1], [1, 4, 5], [2, 3, 6], [3, 7, 6],
                      [0, 2, 4], [2, 6, 4], [1, 5, 3], [3, 5, 7]])
    rotations = Rotation.from_euler("z", [[0], [17], [-8]], degrees=True).as_matrix()
    translations = np.array([[0, 0, 2.], [.6, .7, 2.], [-.2, -.2, 2.]])
    hand = np.array([[-.8, -.8, 1.8], [-.7, -.9, 1.8], [-.7, -.8, 1.8]])
    state = {
        "frame_idx": np.array([40, 501, 3]),
        "object_vertices_canonical": vertices,
        "object_faces": faces,
        "object_rotation": rotations,
        "object_translation_world": translations,
        "base_object_rotation": rotations.copy(),
        "base_object_translation_world": translations + [0, -.4, 0],
        "T_world_camera": np.broadcast_to(np.eye(4), (3, 4, 4)).copy(),
        "left_valid": np.ones(3, bool),
        "right_valid": np.zeros(3, bool),
        "left_vertices_world": np.broadcast_to(hand, (3, 3, 3)).copy(),
        "left_base_vertices_world": np.broadcast_to(hand + [0, .3, 0], (3, 3, 3)).copy(),
        "left_faces": np.array([[0, 1, 2]]),
    }
    return state, np.array([2, 0, 1])


class DetailFraming(unittest.TestCase):
    def assert_contained(self, state, order, size, yaw, base=False):
        W, H = size
        K = np.tile([1.8 * W, 2.1 * H, W / 2 - .5, H / 2 - .5], (3, 1))
        before = copy.deepcopy(state)
        valid = {side: state[side + "_valid"] for side in ("left", "right")}
        zoom, affines, report = detail_viewports(state, order, K, size, yaw, base, valid)
        lower, upper = np.asarray(report["drawable_bounds_uv"])
        for i in range(3):
            for angle in (0, yaw):
                geometry, faces, labels = scene_geometry_for_frame(state, i, base, angle, valid)
                self.assertEqual(labels, [0, 1])
                np.testing.assert_array_equal(faces[0], state["object_faces"])
                for vertices in geometry:
                    uv = project(vertices, K[i])
                    remapped = uv @ affines[i, :, :2].T + affines[i, :, 2]
                    np.testing.assert_allclose(project(vertices, zoom[i]), remapped, atol=1e-9, rtol=0)
                    self.assertTrue(np.all(remapped >= lower - 1e-7), (i, remapped.min(0), lower))
                    self.assertTrue(np.all(remapped <= upper + 1e-7), (i, remapped.max(0), upper))
        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)
        self.assertFalse(report["geometry_or_world_state_modified"])
        return zoom, affines, report

    def test_tall_full_geometry_fits_original_and_yawed_views_after_smoothing(self):
        state, order = fixture()
        for size in ((960, 960), (640, 320), (320, 640), (40, 24)):
            for yaw in (-35.0, 0.0, 20.0, 45.0):
                with self.subTest(size=size, yaw=yaw):
                    self.assert_contained(state, order, size, yaw)

    def test_no_legacy_zoom_cap_or_percentile_vertex_exclusion(self):
        state, order = fixture()
        # One real mesh corner extends dramatically in the middle time row.
        # It must remain in the viewport rather than be clipped by a size cap.
        state["object_vertices_canonical"][7, 1] = 4.0
        _, affine, report = self.assert_contained(state, order, (960, 960), 20.)
        self.assertTrue(np.any(960 / affine[:, 0, 0] > 960 * .834))
        bounds = np.asarray(report["projected_bounds_uv"])
        self.assertTrue(np.isfinite(bounds).all())

    def test_base_mode_uses_actual_base_meshes_and_caption_is_explicit(self):
        state, order = fixture()
        self.assert_contained(state, order, (640, 480), 20., base=True)
        for base in (False, True):
            label = detail_view_title(20., base)
            self.assertIn("demo-only", label)
            self.assertIn("+20 deg", label)
            self.assertNotIn("fitted contact", label)

    def test_chronological_order_not_frame_id_or_array_position(self):
        state, order = fixture()
        K = np.tile([400., 420., 159.5, 119.5], (3, 1))
        a, b, _ = detail_viewports(state, order, K, (320, 240), 20.)
        reordered = copy.deepcopy(state)
        for key in ("frame_idx", "object_rotation", "object_translation_world", "T_world_camera",
                    "left_valid", "right_valid", "left_vertices_world"):
            reordered[key] = reordered[key][order]
        aa, bb, _ = detail_viewports(reordered, np.arange(3), K[order], (320, 240), 20.)
        np.testing.assert_allclose(aa, a[order], atol=1e-9, rtol=0)
        np.testing.assert_allclose(bb, b[order], atol=1e-9, rtol=0)

    def test_camera_plane_geometry_is_reported_not_removed_from_mesh(self):
        state, order = fixture()
        state["left_vertices_world"][:, :, 2] = -1
        before = state["left_vertices_world"].copy()
        K = np.tile([400., 420., 159.5, 119.5], (3, 1))
        _, _, report = detail_viewports(state, order, K, (320, 240), 0.)
        self.assertEqual(report["camera_plane_or_nonfinite_vertex_counts"], [3, 3, 3])
        np.testing.assert_array_equal(before, state["left_vertices_world"])
        geometry, _, _ = scene_geometry_for_frame(state, 0)
        np.testing.assert_array_equal(geometry[1], before[0])

    def test_malformed_viewport_inputs_fail(self):
        state, order = fixture()
        K = np.tile([400., 420., 159.5, 119.5], (3, 1))
        for bad_order in ([0, 0, 2], [0, 1], [0., 1., 2.]):
            with self.assertRaisesRegex(ValueError, "order"):
                detail_viewports(state, bad_order, K, (320, 240))
        with self.assertRaisesRegex(ValueError, "yaw"):
            detail_viewports(state, order, K, (320, 240), float("nan"))
        K[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "intrinsics"):
            detail_viewports(state, order, K, (320, 240))


if __name__ == "__main__":
    unittest.main()
