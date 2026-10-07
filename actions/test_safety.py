from __future__ import annotations

import unittest

import numpy as np

from actions.safety import HardActionBounds, project_action
from actions.scaling import ActionScales
from actions.space import ActionLayout


class HardActionProjectionTests(unittest.TestCase):
    def setUp(self):
        self.layout = ActionLayout(object_slots=("object:0",), contact_variables_per_hand=5)
        self.scales = ActionScales(
            hand_root_rad=0.05,
            hand_articulation_rad=0.05,
            hand_translation_m=0.01,
            object_rotation_rad=0.05,
            object_translation_m=0.01,
            contact_logit=1.0,
        )

    def test_projection_enforces_flow_chart_g8_bounds(self):
        raw = np.full(self.layout.size, 10.0, dtype=np.float32)
        result = project_action(
            raw,
            self.layout,
            self.scales,
            np.ones(self.layout.size, dtype=bool),
            T_world_camera=np.eye(4),
        )

        for block in self.layout.blocks:
            values = result.executed_delta[block.slice]
            if block.kind == "hand_root":
                self.assertLessEqual(float(np.abs(values).max()), 0.15 + 1e-6)
            elif block.kind == "hand_articulation":
                self.assertLessEqual(float(np.abs(values).max()), 0.35 + 1e-6)
            elif block.kind == "hand_translation":
                self.assertLessEqual(float(np.abs(values).max()), 0.025 + 1e-6)
            elif block.kind == "object_rotation":
                self.assertLessEqual(float(np.abs(values).max()), 0.08 + 1e-6)
            elif block.kind == "object_translation":
                np.testing.assert_allclose(np.abs(values), [0.015, 0.015, 0.035])
            elif block.kind == "contact":
                self.assertLessEqual(float(np.abs(values).max()), 4.0 + 1e-6)
        self.assertTrue(result.projected.any())
        self.assertTrue(all(value != "none" for value in result.constraint_ids))
        self.assertTrue(all(value.startswith("hard_bound:") for value in result.reasons))

    def test_object_translation_bound_is_camera_frame(self):
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        raw = np.zeros(self.layout.size, dtype=np.float32)
        block = self.layout.block("object.object:0.translation")
        raw[block.slice] = [3.0, 0.0, 0.0]
        result = project_action(
            raw,
            self.layout,
            self.scales,
            np.ones(self.layout.size, dtype=bool),
            T_world_camera=transform,
        )
        # world [3cm, 0, 0] maps to camera [0, -3cm, 0], so y is clipped
        # and the safe world vector rotates back to [1.5cm, 0, 0].
        np.testing.assert_allclose(result.executed_delta[block.slice], [0.015, 0.0, 0.0], atol=1e-7)
        self.assertEqual(result.constraint_ids[block.start + 1], "G8_object_translation_camera_xy")

    def test_camera_decoder_frame_is_not_double_transformed(self):
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        raw = np.zeros(self.layout.size, dtype=np.float32)
        block = self.layout.block("object.object:0.translation")
        raw[block.slice] = [3.0, 0.0, 0.0]
        result = project_action(
            raw, self.layout, self.scales, np.ones(self.layout.size, dtype=bool),
            T_world_camera=transform, object_translation_frame="camera",
        )
        np.testing.assert_allclose(result.executed_delta[block.slice], [0.015, 0.0, 0.0], atol=1e-7)

        raw = np.full(self.layout.size, 0.5, dtype=np.float32)
        allowed = np.ones(self.layout.size, dtype=bool)
        block = self.layout.block("contact.left")
        allowed[block.slice] = False
        reasons = ["allowed"] * self.layout.size
        reasons[block.start] = "G7_no_contact"
        result = project_action(
            raw,
            self.layout,
            self.scales,
            allowed,
            T_world_camera=np.eye(4),
            reason_by_dimension=reasons,
        )
        np.testing.assert_array_equal(result.executed_delta[block.slice], 0.0)
        self.assertEqual(result.reasons[block.start], "G7_no_contact")
        self.assertEqual(result.constraint_ids[block.start], "semantic_gate")
        self.assertFalse(result.projected[block.start])

    def test_optional_scale_blocks_match_scene_v7_action_contract(self):
        layout = ActionLayout(
            object_slots=("object:0",), contact_variables_per_hand=5,
            include_hand_scale=True, include_object_shared_scale=True,
        )
        self.assertEqual(layout.size, 121)
        self.assertEqual(layout.block("hand.left.scale").size, 1)
        self.assertEqual(layout.block("hand.right.scale").size, 1)
        self.assertEqual(layout.block("object.shared_scale").size, 1)
        raw = np.zeros(layout.size, dtype=np.float32)
        raw[layout.block("hand.left.scale").start] = 10.0
        raw[layout.block("object.shared_scale").start] = -10.0
        result = project_action(raw, layout, self.scales, np.ones(layout.size, dtype=bool), T_world_camera=np.eye(4))
        self.assertAlmostEqual(float(result.executed_delta[layout.block("hand.left.scale").start]), np.log(1.18), places=6)
        self.assertAlmostEqual(float(result.executed_delta[layout.block("object.shared_scale").start]), np.log(0.85), places=6)

    def test_disabled_bounds_preserves_allowed_scaled_action(self):
        raw = np.full(self.layout.size, 10.0, dtype=np.float32)
        result = project_action(
            raw,
            self.layout,
            self.scales,
            np.ones(self.layout.size, dtype=bool),
            hard_bounds=HardActionBounds(enabled=False),
            T_world_camera=np.eye(4),
        )
        np.testing.assert_allclose(result.executed_delta, self.scales.vector(self.layout) * 10.0)
        self.assertFalse(result.projected.any())


if __name__ == "__main__":
    unittest.main()
