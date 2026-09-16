from __future__ import annotations

"""Bounded real-evidence reward smoke; runs only after the fixed DATA interface lands."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from data.bundle import EpisodeBundle, file_sha256, prepared_manifest, write_manifest
from envs.annotation_env import AnnotationEnv
from envs.state import EpisodeStatic

RUN = Path("/mnt/truenas-user-home/zjh/ego_annotation_outputs/v19_runs/20260911_hot3d_P0012_bottle_bbq_zjh_v1")
MANO_ASSETS = Path("/mnt/user-home/zjh/ego_annotation_runtime/v19_bundle_zjh_58372bd_base/third_party/WiLoR/mano_data")
WILOR_ROOT = MANO_ASSETS.parent
SOURCE_VIDEO = Path("/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/pipeline_test_20260907/hot3d_P0012_ca1f6626_bottle_bbq_f1515_1664/input.mp4")


@unittest.skipUnless("contacts" in EpisodeStatic.__dataclass_fields__, "fixed DATA/STATE reward interface not landed")
@unittest.skipUnless(RUN.is_dir() and MANO_ASSETS.is_dir() and SOURCE_VIDEO.is_file(),
                     "licensed prepared bottle fixture is unavailable")
class RealRewardCompositionSmokeTest(unittest.TestCase):
    def test_default_profile_is_finite_measured_and_fully_traced(self):
        annotations = RUN / "measurements/object_geometry/visible_geometry/bottle_bbq/annotations_v19_visible_geometry.json"
        mano_archive = RUN / "measurements/hand_candidates/hawor_world/hawor_world_hands.npz"
        completion = RUN / "measurements/geometry_completion/compact_bottle_bbq_seed42/v18_compact_rigid_trellis_completion_report.json"
        mesh = RUN / "measurements/geometry_completion/compact_bottle_bbq_seed42/bottle_bbq_compact_rigid_completed_mesh_labeled.ply"
        factors = RUN / "measurements/contact_visibility_factors/bottle_bbq_0_149/hot3d_P0012_ca1f6626_bottle_bbq_f1515_1664/v19_visible_contact_ownership_factor_report.json"
        bridge = RUN / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz"
        code = Path(__file__).parents[1] / "AGENTS.md"
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepared_manifest(bundle_id="reward-real-bottle-0-2", clip_id="hot3d-P0012-bottle-0-2",
                source_video_id="hot3d-P0012", source_video=SOURCE_VIDEO, annotations=annotations,
                mano_archive=mano_archive, mano_bridge=bridge, object_mesh=mesh,
                completion_report=completion, contact_factors=factors, object_id="object:bottle_bbq",
                frame_range=(0, 2), code_identity=(str(code), file_sha256(code)),
                config_identity=(str(completion), file_sha256(completion)), mano_assets=MANO_ASSETS)
            path = write_manifest(Path(directory) / "manifest.json", manifest)
            cfg = {"max_objects": 1, "contact_variables_per_hand": 5, "wilor_root": str(WILOR_ROOT),
                   "verify_mano_replay": True, "mano_replay_tolerance_m": 1e-5,
                   "object_observed_max_points": 40, "object_observed_min_pairs": 3,
                   "max_contact_patch_vertices": 16, "seed": 9}
            env = AnnotationEnv([EpisodeBundle.open(path)], cfg, "cpu")
            env.reset(seed=0)
            left = env.state.episode.hands["left"]
            self.assertEqual(np.asarray(left["joint_visibility"]).shape, (3, 21))
            self.assertEqual(np.asarray(left["pose_visibility"]).shape, (3, 15))
            self.assertTrue(np.isfinite(left["joint_visibility"]).all())
            self.assertTrue(np.isfinite(left["pose_visibility"]).all())

            layout = env.state.episode.layout
            rewards = []
            ledgers = []
            for frame in range(3):
                action = np.zeros(layout.size, np.float32)
                action[layout.block("hand.left.translation").start + 2] = .1 if frame == 0 else .02 * frame
                action[layout.block("hand.left.root").start] = .1 if frame == 0 else 0.
                action[layout.block("object.object:0.rotation").start + (frame % 3)] = .1
                action[layout.block("object.object:0.translation").start + ((frame + 1) % 3)] = .1
                _, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward); ledgers.append(info["factor_ledger"])
                self.assertTrue(np.isfinite(reward))
                self.assertTrue(all(np.isfinite(list(info["factor_ledger"]["terms"].values()))))
                self.assertEqual(info["factor_ledger"]["profile"], "formal_annotation")
                self.assertEqual(info["factor_ledger"]["reward_scale"], 1.0)
                self.assertTrue(info["factor_ledger"]["activation"])
            self.assertTrue(terminated and not truncated)
            final = ledgers[-1]
            self.assertAlmostEqual(sum(rewards), -final["objective"], places=5)
            self.assertIn("hand/left/translation_prior", final["terms"])
            self.assertIn("hand/left/visible_surface_depth_order", final["terms"])
            self.assertIn("contact/left/normal_deadband", final["terms"])
            self.assertTrue(any(key.startswith("contact/left/object:bottle_bbq/frame=")
                                for key in final["activation"]))
            self.assertIn("object/object:0/observed_point_to_point", final["terms"])
            self.assertEqual(final["activation"]["object/object:0/source_sigma_priors"],
                             "inactive_missing_measured_or_explicit_sigma")
            self.assertTrue(all(value == "inactive_nonwatertight_or_missing_local_face_authority"
                for key, value in final["activation"].items() if key.startswith("signed/object:0/frame=")))


if __name__ == "__main__":
    unittest.main()
