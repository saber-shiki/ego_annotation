"""Synthetic source-only EpisodeBundle interaction contracts; no real MANO assets."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data.build_index import build_index
from data.bundle import EpisodeBundle, file_sha256, validate_manifest, write_manifest
from data.interaction import load_interaction_evidence, thaw_evidence
from data.test_rl_data import make_synthetic_bundle
from observations.test_interaction_adapter import adapted_fixture


class InteractionEvidenceTests(unittest.TestCase):
    def test_episode_loads_explicit_ids_pads_visibility_and_retains_factors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _ = make_synthetic_bundle(root)
            manifest = json.loads(manifest_path.read_text())
            ids = [2, 7, 11]
            for name in ("annotations", "contact_factors"):
                path = Path(manifest["sources"][name]["path"])
                payload = json.loads(path.read_text())
                for row in payload.get("frames", payload.get("factor_rows", [])):
                    row["frame_idx"] = ids[row["frame_idx"]]
                if name == "contact_factors":
                    payload["factor_rows"][0]["provenance"] = {"nested": [{"source": "synthetic"}]}
                path.write_text(json.dumps(payload))
                manifest["sources"][name]["sha256"] = file_sha256(path)
            path = Path(manifest["sources"]["mano_archive"]["path"])
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["frame_idx"] = np.asarray(ids)
            np.savez(path, **arrays)
            manifest["sources"]["mano_archive"]["sha256"] = file_sha256(path)
            manifest["frame_range"] = [2, 11]; manifest["frame_ids"] = ids
            path = root / "interaction.json"
            payload = adapted_fixture(ids, "object:synthetic")
            path.write_text(json.dumps(payload))
            manifest["sources"]["interaction_judgments"] = {"path": str(path), "sha256": file_sha256(path)}
            write_manifest(manifest_path, manifest)
            bundle = EpisodeBundle.open(manifest_path)
            evidence = bundle.load()
            np.testing.assert_array_equal(evidence["frame_idx"], ids)
            self.assertEqual(build_index([manifest_path])["episodes"][0]["frame_count"], len(evidence["frame_idx"]))
            interaction = evidence["interaction_evidence"]
            self.assertEqual(len(interaction.declarations), 6)
            self.assertEqual(len(interaction.visibility), 9)
            row = interaction.declarations[2, "left", "object:synthetic"]
            self.assertEqual([p.digit for p in row.pads if p.declared], ["thumb", "index"])
            self.assertEqual(row.support_uncertainty_m, .06)
            self.assertFalse(row.support_uncertainty_source["is_metric_measurement"])
            self.assertEqual({v.state for v in interaction.visibility.values()},
                {"visible", "partially_visible", "occluded", "out_of_frame", "unresolved"})
            self.assertEqual(interaction.visibility[2, "left"].occluder, "object")
            self.assertEqual(len(evidence["factor_rows_by_family"]["contact_patch"]), 6)
            self.assertEqual(evidence["hands"]["left"]["visibility_state"][0], "source_validity_only_no_joint_factor_row")
            # Raw declaration is retained even when no_contact/occluded: later
            # qualification may deny reachability, but cannot rewrite the evidence.
            self.assertEqual(interaction.declarations[7, "left", "object:synthetic"].contact_state, "no_contact")
            self.assertTrue(interaction.declarations[7, "left", "object:synthetic"].pads[0].declared)
            with self.assertRaises(FrozenInstanceError):
                row.contact_state = "no_contact"
            with self.assertRaises(TypeError):
                row.support_uncertainty_source["is_metric_measurement"] = True
            with self.assertRaises(TypeError):
                interaction.source["source_judgments"][0]["result"]["interaction_judgments"][0]["uncertainty"] = "changed"
            with self.assertRaises(TypeError):
                bundle.manifest["sources"]["annotations"]["sha256"] = "changed"
            with self.assertRaises(TypeError):
                evidence["factor_rows_by_family"]["contact_patch"][0]["provenance"]["nested"][0]["source"] = "changed"
            with self.assertRaises(ValueError):
                evidence["frame_idx"].setflags(write=True)
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertEqual(EpisodeBundle.open(manifest_path).load()["interaction_evidence"], interaction)

    def test_compiled_timeline_distinguishes_missing_and_visibility(self):
        from data.interaction_features import compile_interaction_timeline
        payload = adapted_fixture()
        evidence = load_interaction_evidence(payload, [2,7,11], ['synthetic'])
        value = compile_interaction_timeline(evidence, [2,7,11], 'synthetic')
        self.assertEqual(value.primary_known.shape, (3,2,5))
        self.assertTrue(value.primary_known.all())
        self.assertFalse(value.primary_visual_eligible[1,0].any())  # no_contact
        self.assertFalse(value.display_observation_eligible[1,0])  # object out_of_frame
        self.assertTrue(evidence.declarations[7,'left','synthetic'].pads[0].declared)
        with self.assertRaises(ValueError): value.visibility.setflags(write=True)
        missing = compile_interaction_timeline(load_interaction_evidence(None,[2,7,11],['synthetic']),[2,7,11],'synthetic')
        self.assertFalse(missing.primary_known.any())
        self.assertFalse(missing.prior_available.any())

    def test_v1_without_interaction_source_is_explicitly_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = make_synthetic_bundle(Path(directory))
            bundle = EpisodeBundle.open(path)
            self.assertNotIn("frame_ids", bundle.manifest)
            self.assertEqual(build_index([path])["episodes"][0]["frame_count"], len(bundle.load()["frame_idx"]))
            result = bundle.load()["interaction_evidence"]
            self.assertIsNone(result.schema)
            self.assertTrue(all(row.state == "unresolved" for row in result.visibility.values()))
            self.assertTrue(all(p.declared is None for row in result.declarations.values() for p in row.pads))

    def test_frozen_manifest_and_factors_serialize_without_mutation_or_hash_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = make_synthetic_bundle(root)
            payload = json.loads(path.read_text())
            factors = payload["sources"]["contact_factors"]
            extra = root / "extra_factors.json"
            extra.write_text(Path(factors["path"]).read_text())
            payload["sources"]["factor_reports"] = [{"path": str(extra), "sha256": file_sha256(extra)}]
            write_manifest(path, payload)
            bundle = EpisodeBundle.open(path)
            value = bundle.load()
            plain = thaw_evidence({"identity": bundle.manifest["identity"],
                "factors": value["factor_rows_by_family"]["contact_patch"],
                "scalars": (None, True, 4, .06, "uncertainty")})
            self.assertEqual(json.loads(json.dumps(plain)), plain)
            self.assertIs(type(plain["scalars"][1]), bool)
            self.assertIs(type(plain["scalars"][2]), int)
            plain["identity"]["source_video"]["sha256"] = "changed"
            plain["factors"][0]["state"] = "changed"
            self.assertNotEqual(bundle.manifest["identity"]["source_video"]["sha256"], "changed")
            self.assertNotEqual(value["factor_rows_by_family"]["contact_patch"][0]["state"], "changed")
            copied = root / "copy.manifest.json"
            write_manifest(copied, bundle.manifest)
            self.assertEqual(json.loads(copied.read_text()), payload)
            self.assertEqual(EpisodeBundle.open(copied).manifest, bundle.manifest)
            validate_manifest(bundle.manifest, path)
            # A frozen sequence must not bypass recursive hash verification.
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, r"sources.factor_reports\[0\] content identity mismatch"):
                validate_manifest(bundle.manifest, path)

    def test_unversioned_v1_declarations_are_not_silently_upgraded(self):
        payload = {"object_id": "synthetic", "interaction_judgments": [{
            "frame_start": 2, "frame_end": 11, "hand_side": "left", "target_entity_id": "object:synthetic",
            "contact_state": "possible_contact", "primary_contact_pads": {"digits": ["ring"], "rationale": "legacy"}}]}
        evidence = load_interaction_evidence(payload, [2, 7, 11], ["synthetic"])
        self.assertIsNone(evidence.schema)
        self.assertEqual(evidence.declarations[7, "left", "synthetic"].contact_state, "possible_contact")
        self.assertIsNone(evidence.declarations[7, "left", "synthetic"].support_uncertainty_m)
        self.assertEqual([p.digit for p in evidence.declarations[7, "left", "synthetic"].pads if p.declared], ["ring"])
        self.assertEqual(evidence.visibility[7, "left"].state, "unresolved")

    def test_typed_source_rejects_overlap_foreign_and_bad_visibility(self):
        payload = adapted_fixture()
        before = deepcopy(payload)
        result = load_interaction_evidence(payload, [2, 7, 11], ["synthetic"])
        payload["source_judgments"][0]["result"]["interaction_judgments"][0]["uncertainty"] = "changed"
        self.assertEqual(result.source["source_judgments"][0]["result"]["interaction_judgments"][0]["uncertainty"],
                         before["source_judgments"][0]["result"]["interaction_judgments"][0]["uncertainty"])
        for defect in ("duplicate", "foreign", "visibility", "timeline", "occlusion", "occlusion_legacy", "depth",
                       "prior_kind", "prior_parameter", "prior_value", "prior_authority"):
            bad = deepcopy(before)
            row = bad["interaction_judgments"][0]
            if defect == "duplicate": bad["interaction_judgments"].append(row)
            elif defect == "foreign": row["target_entity_id"] = "hand:synthetic"
            elif defect == "visibility": bad["frames"][0]["visibility"]["left"]["state"] = "certain"
            elif defect == "timeline": bad["frame_ids"] = [2, 3, 11]
            elif defect == "occlusion": row["occlusion_relation"] = "unknown"
            elif defect == "occlusion_legacy": row["occlusion_relation"] = "object_partially_occluded_by_hand"
            elif defect == "depth": row["depth_reliability"] = "unknown"
            elif defect == "prior_kind": row["contact_support_uncertainty_source"]["kind"] = "visual_measurement"
            elif defect == "prior_parameter": row["contact_support_uncertainty_source"]["parameter"] = "other"
            elif defect == "prior_value": row["contact_support_uncertainty_source"]["value_m"] = .01
            else: row["contact_support_uncertainty_source"]["is_metric_measurement"] = True
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                load_interaction_evidence(bad, [2, 7, 11], ["synthetic"])


if __name__ == "__main__":
    unittest.main()
