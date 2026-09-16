from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from data.bundle import EpisodeBundle, file_sha256, prepared_manifest, write_manifest
from data.test_rl_data import make_synthetic_bundle
from envs.annotation_env import AnnotationEnv
from envs.dynamics import _rotvec_matrix, build_episode, refresh_candidate_terms


class DomainStaticEvidenceTest(unittest.TestCase):
    """Synthetic source evidence tests; production MANO remains exercised separately."""

    def _multi_manifest(self, root: Path):
        base_path, models = make_synthetic_bundle(root / "base")
        base = json.loads(base_path.read_text())
        annotations = Path(base["sources"]["annotations"]["path"])
        payload = json.loads(annotations.read_text())
        for frame in payload["frames"]:
            i = int(frame["frame_idx"])
            frame["objects"].append({
                "object_id": "object:second",
                "rigid_pose_observation_eligible": False if i == 0 else True,
                "reconstructed_geometry_pose": {
                    "rotation_world_from_canonical_matrix": np.eye(3).tolist(),
                    "translation_world_m": [0.5, 0.0, 1.04],
                },
                "visible_geometry_candidate": {
                    "world_vertices_sample_m": (np.asarray([[.46,-.04,1.0],[.54,-.04,1.0],[.5,.04,1.0],[.5,0,1.08]])).tolist()
                },
            })
        annotations.write_text(json.dumps(payload))
        factor = Path(base["sources"]["contact_factors"]["path"])
        report = json.loads(factor.read_text())
        for row in report["factor_rows"]:
            row["target_entity_id"] = "object:synthetic"
        factor.write_text(json.dumps(report))
        extra = root / "generic_factor_rows.json"
        extra.write_text(json.dumps({"factor_rows": [
            {"factor_family": "hand_observation_visibility", "frame_idx": 0, "hand_side": "left",
             "state": "active_hand_observation_visibility", "target_entity_id": "object:synthetic",
             "joint_observation_weight_multiplier": .35,
             "joint_visibility_weights": [.5] * 21, "pose_visibility_weights": [.25] * 15,
             "joint_depth_residual_m": [.01] * 21},
            {"factor_family": "hand_depth_shift_prior", "frame_idx": 0, "hand_side": "left",
             "state": "active_hand_depth_shift_prior", "target_entity_id": "object:synthetic",
             "hand_ray_shift_prior_world_m": [.002, .003, .004],
             "hand_ray_shift_prior_source_m": .005, "hand_ray_shift_prior_weight": 7.0},
        ]}))
        source = base["sources"]
        manifest = prepared_manifest(
            bundle_id="multi", clip_id="multi", source_video_id="video-multi",
            source_video=source["object_mesh"]["path"].replace("_mesh.ply", ".mp4"),
            annotations=annotations, mano_archive=source["mano_archive"]["path"],
            object_mesh={"object:synthetic": source["object_mesh"]["path"], "object:second": source["object_mesh"]["path"]},
            object_id=["object:synthetic", "object:second"], frame_range=(0, 2),
            code_identity=(base["identity"]["code"]["path"], base["identity"]["code"]["sha256"]),
            config_identity=(base["identity"]["config"]["path"], base["identity"]["config"]["sha256"]),
            contact_factors=factor, factor_reports=[extra],
        )
        return write_manifest(root / "multi.json", manifest), models

    def test_multiobject_eligibility_visibility_and_frozen_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path, models = self._multi_manifest(Path(directory))
            bundle = EpisodeBundle.open(path); arrays = bundle.load()
            self.assertEqual(set(arrays["objects"]), {"object:synthetic", "object:second"})
            second = arrays["objects"]["object:second"]
            self.assertTrue(second["present"][0]); self.assertFalse(second["metric_observation_eligible"][0])
            self.assertEqual(len(second["raw_observed_points_world"][0]), 4)
            self.assertEqual(len(second["observed_points_world"][0]), 0)
            left = arrays["hands"]["left"]
            self.assertAlmostEqual(float(left["row_visibility"][0]), .35)
            self.assertAlmostEqual(float(left["joint_visibility"][0, 0]), .5)
            self.assertAlmostEqual(float(left["pose_visibility"][0, 0]), .25)
            np.testing.assert_allclose(left["hand_ray_shift_prior_world_m"][0], [.002, .003, .004])
            self.assertIn((0, "left", "object:synthetic"), arrays["contact"])
            env = AnnotationEnv([bundle], {"max_objects": 2, "contact_variables_per_hand": 1,
                                           "mano_models": models, "verify_mano_replay": True,
                                           "observation_point_samples": 4}, "cpu")
            observation, _ = env.reset(options={"episode_index": 0})
            self.assertEqual(observation.shape, env.observation_space.shape)
            before = refresh_candidate_terms(env.state, {}, 0)["contact_by_key"][("left", "object:synthetic")]
            ids = before["vertex_ids"].copy(); targets = before["targets_world"].copy()
            env.state.hand_vertices["left"][0] += .4
            moved_hand = refresh_candidate_terms(env.state, {}, 0)["contact_by_key"][("left", "object:synthetic")]
            np.testing.assert_array_equal(moved_hand["vertex_ids"], ids)
            np.testing.assert_allclose(moved_hand["targets_world"], targets)
            env.state.object_rotation["object:0"][0] = _rotvec_matrix([0., .3, 0.]) @ env.state.object_rotation["object:0"][0]
            moved_object = refresh_candidate_terms(env.state, {}, 0)["contact_by_key"][("left", "object:synthetic")]
            np.testing.assert_array_equal(moved_object["vertex_ids"], ids)
            self.assertGreater(float(np.max(np.abs(moved_object["targets_world"] - targets))), 1e-6)
            # First frame of second object is present but explicitly not a metric P09 row.
            self.assertNotIn("object:1", refresh_candidate_terms(env.state, {}, 0)["object_observed"])
            env.state.position = 1
            self.assertIn("object:1", refresh_candidate_terms(env.state, {}, 1)["object_observed"])
            exported = env.export_annotation(Path(directory) / "multi_rollout.json")
            with np.load(exported["npz"], allow_pickle=False) as archive:
                self.assertEqual(archive["object_entity_ids"].tolist(), ["object:synthetic", "object:second"])
                self.assertIn("object_1_rotation", archive.files)
                self.assertFalse(archive["object_1_metric_observation_eligible"][0])
            self.assertIn("object:second", exported["document"]["object_pose_trajectories"])

    def test_identity_hashes_and_source_geometry_change_are_observable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base_path, models = make_synthetic_bundle(root / "base")
            base = json.loads(base_path.read_text())
            # Identity verification rejects every mutated identity input, including assets/wrapper.
            asset_root = root / "assets"; asset_root.mkdir(); left = asset_root / "MANO_LEFT.pkl"; right = asset_root / "MANO_RIGHT.pkl"; wrapper = root / "wilor" / "models" / "mano_wrapper.py"; wrapper.parent.mkdir(parents=True)
            for path, data in ((left,b"left"),(right,b"right"),(wrapper,b"wrapper")): path.write_bytes(data)
            base["sources"]["mano_assets"] = {"path": str(asset_root), "left_sha256": file_sha256(left),
                "right_sha256": file_sha256(right), "wrapper_path": str(wrapper), "wrapper_sha256": file_sha256(wrapper)}
            for name, path in (("source_video", Path(base["identity"]["source_video"]["path"])),
                               ("code", Path(base["identity"]["code"]["path"])),
                               ("config", Path(base["identity"]["config"]["path"])),
                               ("factor_evidence", Path(base["sources"]["contact_factors"]["path"])),
                               ("left", left), ("wrapper", wrapper)):
                manifest_path = root / f"{name}.json"; write_manifest(manifest_path, base)
                original = path.read_bytes(); path.write_bytes(original + b"mutated")
                with self.assertRaises(ValueError): EpisodeBundle.open(manifest_path)
                path.write_bytes(original)
            bound = EpisodeBundle.open(root / "wrapper.json")
            wrapper.write_bytes(wrapper.read_bytes() + b"late-mutation")
            with self.assertRaises(ValueError):
                build_episode(bound, {"verify_mano_replay": False}, "cpu")
            wrapper.write_bytes(b"wrapper")
            # A changed source-object pose/surfel geometry is visible to the causal actor input.
            annotations = Path(base["sources"]["annotations"]["path"]); payload = json.loads(annotations.read_text())
            for frame in payload["frames"]:
                obj = frame["objects"][0]; obj["reconstructed_geometry_pose"]["translation_world_m"][0] += .5
                obj["visible_geometry_candidate"]["world_vertices_sample_m"] = (np.asarray(obj["visible_geometry_candidate"]["world_vertices_sample_m"]) + [0.5,0,0]).tolist()
            shifted = root / "shifted_annotations.json"; shifted.write_text(json.dumps(payload))
            source = base["sources"]
            shifted_manifest = prepared_manifest(bundle_id="shifted", clip_id="shifted", source_video_id="shifted",
                source_video=base["identity"]["source_video"]["path"], annotations=shifted,
                mano_archive=source["mano_archive"]["path"], object_mesh=source["object_mesh"]["path"],
                object_id="object:synthetic", frame_range=(0,2),
                code_identity=(base["identity"]["code"]["path"], base["identity"]["code"]["sha256"]),
                config_identity=(base["identity"]["config"]["path"], base["identity"]["config"]["sha256"]),
                contact_factors=source["contact_factors"]["path"])
            shifted_path = write_manifest(root / "shifted.json", shifted_manifest)
            cfg = {"max_objects":1,"contact_variables_per_hand":1,"mano_models":models,"verify_mano_replay":True}
            first = AnnotationEnv([EpisodeBundle.open(base_path)], cfg, "cpu"); second = AnnotationEnv([EpisodeBundle.open(shifted_path)], cfg, "cpu")
            observation0, _ = first.reset(options={"episode_index":0}); observation1, _ = second.reset(options={"episode_index":0})
            self.assertGreater(float(np.max(np.abs(observation0 - observation1))), 1e-4)


if __name__ == "__main__":
    unittest.main()
