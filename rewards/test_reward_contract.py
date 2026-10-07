from __future__ import annotations

import unittest

from rewards.evaluator import RewardEvaluator
from rewards.profiles import resolve_profile


class SceneV7RewardContractTests(unittest.TestCase):
    def test_profile_is_named_and_fail_closed(self):
        profile = resolve_profile({"reward_profile": "scene_v7_rl"})
        self.assertEqual(profile.required_families, ("pose", "prior", "time", "depth", "shape", "hand", "pad"))
        evaluator = RewardEvaluator({"reward_profile": "scene_v7_rl"}, "cpu")
        activation = {}
        with self.assertRaisesRegex(RuntimeError, "shape"):
            evaluator._validate_reward_contract({"object/observed_point_to_plane": 1.0}, {}, activation)
        self.assertIn("reward_contract", activation)
        self.assertIn("pad", activation["reward_contract"]["missing_families"])

    def test_contract_accepts_all_required_families(self):
        evaluator = RewardEvaluator({"reward_profile": "scene_v7_rl"}, "cpu")
        terms = {
            "object/observed_point_to_plane": 1.0,
            "hand/left/translation_prior": 1.0,
            "hand/left/translation_velocity": 1.0,
            "hand/left/joint_depth_deadband/frame=0": 1.0,
            "scene_v7/shape/factor=0": 1.0,
            "scene_v7/pad/factor=0": 1.0,
        }
        activation = {
            "scene_v7/shape": "active_test",
            "scene_v7/pad": "active_test",
        }
        evaluator._validate_reward_contract(terms, {}, activation)
        self.assertEqual(activation["reward_contract"]["missing_families"], [])


if __name__ == "__main__":
    unittest.main()
