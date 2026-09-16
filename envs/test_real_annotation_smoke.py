from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import numpy as np

from data.bundle import EpisodeBundle, file_sha256, prepared_manifest, write_manifest
from envs.annotation_env import AnnotationEnv
from envs.dynamics import transformed_object_geometry

RUN = Path("/mnt/truenas-user-home/zjh/ego_annotation_outputs/v19_runs/20260911_hot3d_P0012_bottle_bbq_zjh_v1")
MANO_ASSETS = Path("/mnt/user-home/zjh/ego_annotation_runtime/v19_bundle_zjh_58372bd_base/third_party/WiLoR/mano_data")
WILOR_ROOT = MANO_ASSETS.parent
SOURCE_VIDEO = Path("/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/pipeline_test_20260907/hot3d_P0012_ca1f6626_bottle_bbq_f1515_1664/input.mp4")


@unittest.skipUnless(RUN.is_dir() and MANO_ASSETS.is_dir() and SOURCE_VIDEO.is_file(),
                     "licensed prepared bottle fixture is unavailable")
class RealPreparedBottleSmokeTest(unittest.TestCase):
    def test_real_mano_object_evidence_reward_and_direct_export(self):
        annotations = RUN / "measurements/object_geometry/visible_geometry/bottle_bbq/annotations_v19_visible_geometry.json"
        mano_archive = RUN / "measurements/hand_candidates/hawor_world/hawor_world_hands.npz"
        completion = RUN / "measurements/geometry_completion/compact_bottle_bbq_seed42/v18_compact_rigid_trellis_completion_report.json"
        mesh = RUN / "measurements/geometry_completion/compact_bottle_bbq_seed42/bottle_bbq_compact_rigid_completed_mesh_labeled.ply"
        factors = RUN / "measurements/contact_visibility_factors/bottle_bbq_0_149/hot3d_P0012_ca1f6626_bottle_bbq_f1515_1664/v19_visible_contact_ownership_factor_report.json"
        bridge = RUN / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz"
        code = Path(__file__).parents[1] / "AGENTS.md"
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepared_manifest(bundle_id="real-bottle-0-2", clip_id="hot3d-P0012-bottle-0-2",
                source_video_id="hot3d-P0012", source_video=SOURCE_VIDEO, annotations=annotations,
                mano_archive=mano_archive, mano_bridge=bridge, object_mesh=mesh,
                completion_report=completion, contact_factors=factors, object_id="object:bottle_bbq",
                frame_range=(0, 2), code_identity=(str(code), file_sha256(code)),
                config_identity=(str(completion), file_sha256(completion)), mano_assets=MANO_ASSETS)
            manifest_path = write_manifest(Path(directory) / "manifest.json", manifest)
            cfg = {"max_objects": 1, "contact_variables_per_hand": 5, "wilor_root": str(WILOR_ROOT),
                   "verify_mano_replay": True, "mano_replay_tolerance_m": 1e-5,
                   "object_observed_max_points": 40, "object_observed_min_pairs": 3,
                   "max_contact_patch_vertices": 16, "seed": 9}
            env = AnnotationEnv([EpisodeBundle.open(manifest_path)], cfg, "cpu")
            env.reset(seed=0)
            layout = env.state.episode.layout
            base_vertices = env.state.hand_vertices["left"][0].copy()
            base_joints = env.state.hand_joints["left"][0].copy()
            base_object, _ = transformed_object_geometry(env.state, 0, "object:0")
            action = np.zeros(layout.size, np.float32)
            action[layout.block("hand.left.root").start] = .15
            action[layout.block("hand.left.articulation").start + 3] = .25
            action[layout.block("hand.left.translation").start + 2] = .1
            action[layout.block("object.object:0.rotation").start + 1] = .2
            action[layout.block("object.object:0.translation").start] = .3
            action[layout.block("contact.left").start] = .4
            _, reward0, _, _, info0 = env.step(action)
            self.assertTrue(np.isfinite(reward0))
            self.assertGreater(np.max(np.abs(env.state.hand_vertices["left"][0] - base_vertices)), 1e-7)
            self.assertGreater(np.max(np.abs(env.state.hand_joints["left"][0] - base_joints)), 1e-7)
            moved_object, _ = transformed_object_geometry(env.state, 0, "object:0")
            self.assertGreater(np.max(np.abs(moved_object - base_object)), 1e-5)
            self.assertIn("hand/left/visible_surface_depth_order", info0["factor_ledger"]["terms"])
            self.assertGreater(info0["factor_ledger"]["terms"]["contact/left/state_prior"], 0.0)
            self.assertEqual(info0["factor_ledger"]["activation"]["signed/object:0/0"],
                             "inactive_nonwatertight_or_missing_local_face_authority")
            contact = env.state.term_cache[0]["contact"]["left"]
            R = env.state.object_rotation["object:0"][0]; t = env.state.object_translation["object:0"][0]
            np.testing.assert_allclose(contact["targets_world"], contact["targets_object"] @ R.T + t, atol=1e-7)
            np.testing.assert_allclose(contact["normals_world"], contact["normals_object"] @ R.T, atol=1e-7)
            rewards = [reward0]
            for frame in (1, 2):
                current = np.zeros(layout.size, np.float32)
                current[layout.block("object.object:0.rotation").start + 2] = .1 * frame
                current[layout.block("object.object:0.translation").start + 1] = -.1 * frame
                _, reward, terminated, truncated, info = env.step(current)
                rewards.append(reward)
            self.assertTrue(terminated and not truncated)
            self.assertAlmostEqual(sum(rewards), -info["objective"], places=5)
            self.assertIn("object/object:0/observed_point_to_point", info["factor_ledger"]["terms"])
            exported = env.export_annotation(Path(directory) / "policy_rollout.json")
            with np.load(exported["npz"], allow_pickle=False) as archive:
                from envs.qualification.output import timeline
                timeline(dict(archive))
                self.assertEqual(archive["left_vertices_world"].shape, (3, 778, 3))
                self.assertEqual(archive["left_faces"].shape, (1538, 3))
                self.assertTrue(np.array_equal(archive["left_display_valid"], archive["left_valid"]))
                self.assertTrue(np.array_equal(archive["right_display_valid"], archive["right_valid"]))
                self.assertFalse(np.allclose(archive["object_rotation"][0], np.eye(3)))
            self.assertEqual(exported["hashes"]["json_sha256"], file_sha256(exported["json"]))
            self.assertEqual(exported["hashes"]["npz_sha256"], file_sha256(exported["npz"]))
            self.assertEqual(exported["hashes"]["integrity_sha256"], file_sha256(exported["integrity"]))
            self.assertEqual(exported["integrity_document"]["files"]["policy_rollout_json"]["sha256"],
                             exported["hashes"]["json_sha256"])
            self.assertEqual(exported["integrity_document"]["files"]["policy_rollout_npz"]["sha256"],
                             exported["hashes"]["npz_sha256"])
            self.assertEqual(exported["document"]["inputs"]["annotations"], str(annotations))
            self.assertFalse(exported["document"]["acceptance_substitution_applied"])
            # Existing evaluator adapters consume the JSON action-driven joint/SE3 rows
            # and the annotation camera trajectory, without requiring formal acceptance.
            import evaluate
            evaluator_prediction = evaluate.load_interval_state_prediction(
                SimpleNamespace(interval_state=Path(exported["json"]), remote_root=None, local_root=None),
                json.loads(annotations.read_text())["frames"],
            )
            self.assertEqual(evaluator_prediction["camera_trajectory_source"], f"annotations:{annotations}")
            self.assertGreater(np.max(np.abs(evaluator_prediction["sides"]["left"]["joints_world_m"][0] - base_joints)), 1e-7)
            object_rows = evaluate.load_v19_rows(json.loads(Path(exported["json"]).read_text()))
            self.assertIn(0, object_rows)
            self.assertFalse(np.allclose(object_rows[0]["rotation_world_from_completed_canonical_matrix"], np.eye(3)))
            # Existing direct triangle renderer consumes the policy display-valid arrays.
            baseline_env = AnnotationEnv([EpisodeBundle.open(manifest_path)], cfg, "cpu")
            baseline_env.reset(seed=0)
            baseline_export = baseline_env.export_annotation(Path(directory) / "baseline_rollout.json")
            import cv2
            import render
            raw_manifest = json.loads((RUN / "input/raw_frame_manifest/manifest.json").read_text())
            slice_manifest = Path(directory) / "raw_slice.json"
            slice_manifest.write_text(json.dumps({"frames": raw_manifest["frames"][:3]}))
            raw_image = cv2.imread(raw_manifest["frames"][0]["raw_frame_path"])
            with np.load(exported["npz"], allow_pickle=False) as archive:
                action_pixels = render.mesh_overlay(dict(archive), 0, raw_image)
            with np.load(baseline_export["npz"], allow_pickle=False) as archive:
                baseline_pixels = render.mesh_overlay(dict(archive), 0, raw_image)
            self.assertGreater(np.max(np.abs(action_pixels.astype(np.int16) - baseline_pixels.astype(np.int16))), 0)
            reviews = render.render_video(exported["npz"], slice_manifest, Path(directory) / "policy.mp4")
            self.assertEqual(len(reviews), 3)
            self.assertTrue((Path(directory) / "policy.mp4").is_file())
            self.assertTrue(all(Path(row["path"]).is_file() for row in reviews))
            baseline_env = AnnotationEnv([EpisodeBundle.open(manifest_path)], cfg, "cpu")
            baseline_env.reset(seed=0)
            _, _, _, _, baseline_info = baseline_env.step(np.zeros(layout.size, np.float32))
            self.assertNotEqual(info0["factor_ledger"]["terms"]["hand/left/visible_surface_depth_order"],
                                baseline_info["factor_ledger"]["terms"]["hand/left/visible_surface_depth_order"])
            self.assertNotEqual(info0["objective"], baseline_info["objective"])


if __name__ == "__main__":
    unittest.main()
