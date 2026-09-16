from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from actions.scaling import ActionScales
from data.bundle import EpisodeBundle, file_sha256
from data.test_rl_data import make_synthetic_bundle
from envs.annotation_env import AnnotationEnv
from envs.dynamics import transformed_object_geometry


class AnnotationEnvironmentTest(unittest.TestCase):
    def config(self, models, **updates):
        cfg = {
            "max_objects": 1, "contact_variables_per_hand": 5, "mano_models": models,
            "verify_mano_replay": True, "mano_replay_tolerance_m": 1e-6,
            "object_observed_max_points": 20, "object_observed_min_pairs": 3,
            "max_contact_patch_vertices": 12, "seed": 3,
            "action_scales": ActionScales(hand_root_rad=.1, hand_articulation_rad=.2,
                hand_translation_m=.01, object_rotation_rad=.1, object_translation_m=.02,
                contact_logit=1.0),
        }
        cfg.update(updates)
        return cfg

    def test_true_geometry_full_se3_gates_temporal_return_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path, models = make_synthetic_bundle(Path(directory))
            env = AnnotationEnv([EpisodeBundle.open(path)], self.config(models), "cpu")
            obs, _ = env.reset(seed=11)
            self.assertEqual(obs.shape, env.observation_space.shape)
            layout = env.state.episode.layout
            left_before = env.state.hand_vertices["left"][0].copy()
            object_before, _ = transformed_object_geometry(env.state, 0, "object:0")
            action = np.zeros(layout.size, np.float32)
            action[layout.block("hand.left.root").start + 2] = .25
            action[layout.block("hand.left.articulation").start] = .5
            action[layout.block("hand.left.translation").start] = .5
            action[layout.block("object.object:0.rotation").start + 1] = .5
            action[layout.block("object.object:0.translation").start] = .5
            action[layout.block("contact.left").start] = .7
            _, reward0, terminated, truncated, info0 = env.step(action)
            self.assertFalse(terminated or truncated)
            np.testing.assert_array_equal(info0["raw_action"], action)
            self.assertGreater(float(np.max(np.abs(env.state.hand_vertices["left"][0] - left_before))), 1e-6)
            object_after, _ = transformed_object_geometry(env.state, 0, "object:0")
            self.assertGreater(float(np.max(np.abs(object_after - object_before))), 1e-4)
            self.assertGreater(float(np.linalg.norm(env.state.object_translation["object:0"][0] -
                                                    env.state.episode.objects["object:0"]["translation"][0])), 0.0)
            # Frame 1 has an absent right hand: only its own hand/contact blocks are filtered.
            action1 = np.ones(layout.size, np.float32) * .1
            _, reward1, _, _, info1 = env.step(action1)
            right_root = layout.block("hand.right.root")
            object_rot = layout.block("object.object:0.rotation")
            self.assertTrue(np.all(info1["executed_action"][right_root.slice] == 0))
            self.assertTrue(np.all(info1["executed_action"][object_rot.slice] != 0))
            self.assertTrue(np.all(info1["raw_action"][right_root.slice] == .1))
            _, reward2, terminated, truncated, info2 = env.step(np.zeros(layout.size, np.float32))
            self.assertTrue(terminated); self.assertFalse(truncated)
            self.assertIn("final_observation", info2)
            self.assertAlmostEqual(reward0 + reward1 + reward2, -info2["objective"], places=5)
            self.assertGreater(info2["factor_ledger"]["terms"]["hand/left/translation_velocity"], 0.0)
            export = env.export_annotation(Path(directory) / "rollout.json")
            with np.load(export["npz"], allow_pickle=False) as archive:
                self.assertEqual(archive["left_vertices_world"].shape, (3, 778, 3))
                self.assertEqual(archive["object_rotation"].shape, (3, 3, 3))
                self.assertTrue(np.array_equal(archive["left_display_valid"], archive["left_valid"]))
                self.assertTrue(np.array_equal(archive["right_display_valid"], archive["right_valid"]))
                self.assertFalse(np.allclose(archive["left_vertices_world"][0], left_before))
                self.assertFalse(np.allclose(archive["object_rotation"][0], np.eye(3)))
            self.assertEqual(export["hashes"]["json_sha256"], file_sha256(export["json"]))
            self.assertEqual(export["hashes"]["npz_sha256"], file_sha256(export["npz"]))
            self.assertEqual(export["hashes"]["integrity_sha256"], file_sha256(export["integrity"]))
            receipt = export["integrity_document"]
            self.assertEqual(receipt["files"]["policy_rollout_json"]["sha256"], export["hashes"]["json_sha256"])
            self.assertEqual(receipt["files"]["policy_rollout_npz"]["sha256"], export["hashes"]["npz_sha256"])
            self.assertFalse(export["document"]["acceptance_substitution_applied"])
            self.assertEqual(export["document"]["object_pose_trajectory"]["pose_rows"][0]["status"],
                             "policy_rollout_candidate")
            reset_obs, _ = env.reset(seed=11)
            np.testing.assert_allclose(reset_obs, obs)
            np.testing.assert_allclose(env.state.hand_vertices["left"][0], left_before)

    def test_explicit_index_variable_episode_rollout_exports_before_destructive_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path0, models = make_synthetic_bundle(root / "short", "short", "video-short", frame_count=2)
            path1, _ = make_synthetic_bundle(root / "long", "long", "video-long", frame_count=4)
            env = AnnotationEnv([EpisodeBundle.open(path0), EpisodeBundle.open(path1)], self.config(models), "cpu")
            _, info = env.reset(seed=99, options={"episode_index": 1})
            self.assertEqual(info["bundle_id"], "long")
            layout = env.state.episode.layout
            long_source = env.state.hand_vertices["left"][0].copy()
            while True:
                action = np.zeros(layout.size, np.float32)
                action[layout.block("hand.left.articulation").start + 1] = .3
                action[layout.block("hand.left.root").start + 2] = .2
                action[layout.block("object.object:0.rotation").start] = .2
                _, _, terminated, truncated, _ = env.step(action)
                self.assertFalse(truncated)
                if terminated:
                    break
            export = env.export_annotation(root / "long_rollout.json")
            with np.load(export["npz"], allow_pickle=False) as archive:
                self.assertEqual(len(archive["frame_idx"]), 4)
                self.assertFalse(np.allclose(archive["left_vertices_world"][0], long_source))
                preserved = archive["left_vertices_world"].copy()
            _, reset_info = env.reset(options={"episode_index": 0})
            self.assertEqual(reset_info["bundle_id"], "short")
            self.assertEqual(env.state.episode.frame_count, 2)
            np.testing.assert_allclose(env.state.hand_vertices["left"][0],
                                       env.state.episode.hands["left"]["vertices"][0])
            with np.load(export["npz"], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["left_vertices_world"], preserved)

    def test_translation_support_gate_and_truncation_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path, models = make_synthetic_bundle(Path(directory))
            cfg = self.config(models, freeze_translation_without_visible_surface_support=True,
                              translation_gate_min_visible_surface_depth_vertices=0,
                              max_episode_steps=1)
            env = AnnotationEnv([EpisodeBundle.open(path)], cfg, "cpu")
            env.reset(seed=0)
            layout = env.state.episode.layout
            action = np.zeros(layout.size, np.float32)
            hand_t = layout.block("hand.left.translation")
            hand_r = layout.block("hand.left.root")
            action[hand_t.slice] = 1.0; action[hand_r.slice] = 1.0
            obs, _, terminated, truncated, info = env.step(action)
            self.assertFalse(terminated); self.assertTrue(truncated)
            self.assertTrue(np.all(info["executed_action"][hand_t.slice] == 0.0))
            self.assertTrue(np.all(info["executed_action"][hand_r.slice] == 1.0))
            self.assertEqual(obs[-layout.size + hand_t.start], 0.0)
            with self.assertRaises(RuntimeError):
                env.step(action)


if __name__ == "__main__":
    unittest.main()
