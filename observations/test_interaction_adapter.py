"""Current strict judge schemas -> deterministic assembly -> real P17 consumer."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jsonschema

from observations.interaction.loader import load_agent_interaction_judgments
from observations.interaction.adapter import assemble_interaction_evidence, OCCLUSION_MAP


def schema_results(ids=(2, 7, 11), object_id="synthetic"):
    intervals = [{"frame_start": frame, "frame_end": frame, "hand_side": side,
                  "rationale": "synthetic image judgment", "uncertainty": "ambiguous image boundary"}
                 for frame in ids for side in ("left", "right")]
    states = ("likely_contact", "possible_contact", "no_contact", "unresolved")
    contact = {"object_id": object_id, "status": "ok", "interaction_judgments": [
        {**row, "target_entity_id": "object:" + object_id.removeprefix("object:"),
         "contact_state": states[i % 4], "contact_prior_probability": (.8, .4, .0, .5)[i % 4]}
        for i, row in enumerate(intervals)]}
    pads = {"object_id": object_id, "intervals": [{**row, "digits": ["thumb", "index"]} for row in intervals]}
    occlusion = {"object_id": object_id, "intervals": [
        {**row, "relation": list(OCCLUSION_MAP)[i % 4]} for i, row in enumerate(intervals)]}
    depth = {"object_id": object_id, "intervals": [
        {"frame_start": frame, "frame_end": frame, "entity": entity, "reliability": "supported",
         "rationale": "synthetic depth judgment", "uncertainty": "qualitative only"}
        for frame in ids for entity in ("object", "left", "right")]}
    visibility_states = ("visible", "partially_visible", "occluded", "out_of_frame", "unresolved")
    visibility = {"object_id": object_id, "frames": [
        {"frame_idx": frame, "visibility": {entity: {"state": visibility_states[(3*i+j) % 5],
            "occluder": "object" if entity != "object" else "left", "uncertainty": "image uncertainty"}
            for j, entity in enumerate(("object", "left", "right"))}} for i, frame in enumerate(ids)]}
    return {"contact": contact, "contact_pads": pads, "occlusion": occlusion,
            "depth_reliability": depth, "visibility": visibility}


def adapted_fixture(ids=(2, 7, 11), object_id="synthetic"):
    return assemble_interaction_evidence(object_id=object_id, frame_ids=list(ids),
        batches=[{"frame_ids": list(ids), "judgments": schema_results(ids, object_id)}])


class InteractionAdapterTests(unittest.TestCase):
    def assemble(self, results, ids=(2, 7, 11)):
        return assemble_interaction_evidence(object_id="synthetic", frame_ids=list(ids),
            batches=[{"frame_ids": list(ids), "judgments": results}])

    def p17(self, payload, root):
        path = root / "interaction.json"
        path.write_text(json.dumps(payload))
        return load_agent_interaction_judgments(path, case="fixture", target_entity_id="object:synthetic",
            frame_span=(2, 11), sides={"left", "right"}, timeline=[2, 7, 11])

    def test_real_judge_results_reach_real_p17_loader(self):
        results = schema_results()
        for name, raw in results.items():
            module = importlib.import_module("observations.judgments.judge_" + name)
            jsonschema.validate(raw, module.SCHEMA)
            # Only the external transport is stubbed. No fake legacy schema fields.
            with patch.object(module, "astra_request", return_value=(deepcopy(raw), {"request_sha256": "a" * 64})):
                results[name] = module.judge({"frame_ids": [2, 7, 11], "object_id": "synthetic"}, [])
        before = deepcopy(results)
        payload = self.assemble(results)
        self.assertEqual(results, before)
        with tempfile.TemporaryDirectory() as directory:
            index, summary = self.p17(payload, Path(directory))
        self.assertEqual(set(index), {(f, s) for f in (2, 7, 11) for s in ("left", "right")})
        self.assertEqual(summary["frame_side_count"], 6)
        self.assertEqual(index[7, "left"]["contact_state"], "no_contact")
        self.assertEqual(index[7, "right"]["contact_state"], "unresolved")
        self.assertEqual(index[2, "left"]["contact_support_uncertainty_m"], .06)
        self.assertFalse(index[2, "left"]["contact_support_uncertainty_source"]["is_metric_measurement"])
        for raw, row in zip(results["occlusion"]["intervals"], payload["interaction_judgments"]):
            self.assertEqual(row["occlusion_relation"], OCCLUSION_MAP[raw["relation"]])
        self.assertEqual(payload["source_judgments"][2]["role"], "diagnostic_only_not_a_visual_fit_factor")
        self.assertEqual(payload["source_judgments"][0]["result"], results["contact"])
        results["contact"]["interaction_judgments"][0]["uncertainty"] = "mutated"
        self.assertNotEqual(payload["source_judgments"][0]["result"], results["contact"])

    def test_receipts_and_prior_are_preserved_in_factor_provenance_and_recovery(self):
        from observations.interaction.evidence import agent_judgment_provenance
        from observations.interaction.adapter import SCHEMA
        from utils.state import RunState, StageRecovery, planned_stage_contract, sha
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            raw = root / "input/raw.mp4"; raw.write_bytes(b"synthetic-not-video")
            contract = {"run_id": "fixture", "run_root": str(root), "input_video": str(raw), "input_video_sha256": sha(raw)}
            (root / "input/runtime_input_contract.json").write_text(json.dumps(contract))
            ledger = RunState(root)
            identity = {"run_id": "fixture", "agent_model": "gpt-6-astra", "code": "synthetic"}
            results = schema_results()
            sources = {}
            for name, value in results.items():
                path = root / (name + ".json"); path.write_text(json.dumps(value))
                sources[name] = {"path": str(path), "sha256": sha(path)}
            batches = [{"frame_ids": [2, 7, 11], "judgments": results, "sources": sources}]
            output = root / "assembled.json"
            def planned(prior):
                return planned_stage_contract("main.annotate.interaction_assembly", "inprocess",
                    inputs={"object_id": "synthetic", "frame_ids": [2, 7, 11], "values": {"batches": batches}},
                    parameters={"output": output, "schema": SCHEMA, "contact_support_uncertainty_prior_m": prior},
                    execution={"fixture": "synthetic CPU"})
            def produce():
                payload = assemble_interaction_evidence(object_id="synthetic", frame_ids=[2, 7, 11], batches=batches)
                output.write_text(json.dumps(payload))
                ledger.record("interaction_assembly", [(output, "interaction_judgment")], object_id="synthetic", frame_ids=[2, 7, 11])
                return output
            recovery = StageRecovery(ledger, identity)
            try:
                recovery.run("interaction_assembly", produce, contract=planned(.06))
            finally:
                recovery.close()
            payload = json.loads(output.read_text())
            self.assertEqual(payload["source_judgments"][0]["receipt"], sources["contact"])
            self.assertEqual(payload["timeline_evidence"]["contact"][0]["judgment_sha256"], sources["contact"]["sha256"])
            provenance = agent_judgment_provenance(payload["interaction_judgments"][0])["agent_interaction_judgment"]
            self.assertFalse(provenance["contact_support_uncertainty_source"]["is_metric_measurement"])
            self.assertIn("source_refs", provenance)
            recovery = StageRecovery(ledger, identity, resume=True)
            try:
                with self.assertRaisesRegex(ValueError, "planned stage contract changed"):
                    recovery.run("interaction_assembly", produce, contract=planned(.05))
            finally:
                recovery.close()

    def test_missing_evidence_is_not_contact_visibility_or_pad_support(self):
        results = schema_results()
        results["contact"]["interaction_judgments"] = []
        results["contact_pads"]["intervals"] = []
        results.pop("visibility"); results.pop("occlusion"); results.pop("depth_reliability")
        payload = self.assemble(results)
        with tempfile.TemporaryDirectory() as directory:
            index, _ = self.p17(payload, Path(directory))
        for row in index.values():
            self.assertEqual(row["contact_state"], "unresolved")
            self.assertEqual(row["occlusion_relation"], "unresolved")
            self.assertEqual(row["depth_reliability"], "not_evaluated")
            self.assertTrue(all(p["declared"] is None for p in row["primary_pad_declarations"]))
        self.assertEqual(payload["frames"][0]["visibility"]["left"]["state"], "unresolved")

    def test_rejects_foreign_identity_overlap_and_foreign_timeline(self):
        for name in schema_results():
            with self.subTest(name=name, defect="identity"):
                results = schema_results(); results[name]["object_id"] = "foreign"
                with self.assertRaisesRegex(ValueError, "foreign object"):
                    self.assemble(results)
            with self.subTest(name=name, defect="overlap"):
                results = schema_results(); key = "frames" if name == "visibility" else "interaction_judgments" if name == "contact" else "intervals"
                results[name][key].append(deepcopy(results[name][key][0]))
                with self.assertRaisesRegex(ValueError, "duplicate/overlapping"):
                    self.assemble(results)
        results = schema_results(); results["contact"]["interaction_judgments"][0]["target_entity_id"] = "hand:synthetic"
        with self.assertRaisesRegex(ValueError, "foreign target"):
            self.assemble(results)
        results = schema_results(); results["contact_pads"]["intervals"][0]["frame_end"] = 3
        with self.assertRaisesRegex(ValueError, "outside declared timeline"):
            self.assemble(results)
        batch = {"frame_ids": [2, 7, 11], "judgments": schema_results()}
        with self.assertRaisesRegex(ValueError, "overlapping batch"):
            assemble_interaction_evidence(object_id="synthetic", frame_ids=[2, 7, 11], batches=[batch, batch])

    def test_interval_joins_only_declared_ids_and_conservative_depth_mapping(self):
        results = schema_results()
        for name in ("contact", "contact_pads", "occlusion"):
            key = "interaction_judgments" if name == "contact" else "intervals"
            results[name][key] = [{**row, "frame_end": 11} for row in results[name][key][:2]]
        results["depth_reliability"]["intervals"][1]["reliability"] = "mixed_depth"
        results["depth_reliability"]["intervals"][4]["reliability"] = "unresolved"
        payload = self.assemble(results)
        with tempfile.TemporaryDirectory() as directory:
            index, _ = self.p17(payload, Path(directory))
            self.assertEqual(index[2, "left"]["depth_reliability"], "hand_depth_unreliable")
            self.assertEqual(index[7, "left"]["depth_reliability"], "mixed_or_unresolved")
            path = Path(directory) / "interaction.json"
            with self.assertRaisesRegex(ValueError, "timeline differs"):
                load_agent_interaction_judgments(path, case="fixture", target_entity_id="object:synthetic",
                    frame_span=(2, 11), sides={"left", "right"}, timeline=[2, 3, 11])
        self.assertEqual({key[0] for key in index}, {2, 7, 11})

    def test_real_p17_rejects_adapter_rows_with_foreign_identity_or_duplicate(self):
        for defect in ("target", "hand", "frame", "duplicate", "occlusion", "occlusion_legacy", "depth", "prior_kind",
                       "prior_parameter", "prior_value", "prior_authority"):
            payload = adapted_fixture()
            row = payload["interaction_judgments"][0]
            if defect == "target": row["target_entity_id"] = "hand:synthetic"
            elif defect == "hand": row["hand_side"] = "foreign"
            elif defect == "frame": row["frame_end"] = 3
            elif defect == "duplicate": payload["interaction_judgments"].append(deepcopy(row))
            elif defect == "occlusion": row["occlusion_relation"] = "unknown"
            elif defect == "occlusion_legacy": row["occlusion_relation"] = "object_partially_occluded_by_hand"
            elif defect == "depth": row["depth_reliability"] = "unknown"
            elif defect == "prior_kind": row["contact_support_uncertainty_source"]["kind"] = "visual_measurement"
            elif defect == "prior_parameter": row["contact_support_uncertainty_source"]["parameter"] = "other"
            elif defect == "prior_value": row["contact_support_uncertainty_source"]["value_m"] = .01
            else: row["contact_support_uncertainty_source"]["is_metric_measurement"] = True
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                self.p17(payload, Path(directory))

    def test_interaction_build_consumes_adapted_payload_without_synthetic_mano_loading(self):
        from observations.interaction.evidence import interaction_build, interaction_parse_args
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            judgments = root / "judgments.json"; judgments.write_text(json.dumps(adapted_fixture()))
            annotations = root / "annotations.json"
            # Explicit no-object observation: exercises assembly->build->real loader,
            # not geometry generation or a claim that contact factors were activated.
            annotations.write_text(json.dumps({"frames": [{"frame_idx": f, "objects": []} for f in (2, 7, 11)]}))
            args = interaction_parse_args(["--annotations", str(annotations), "--case", "fixture",
                "--target-entity-id", "object:synthetic", "--frame-span", "2", "11",
                "--output-root", str(root), "--agent-interaction-judgment", str(judgments)])
            report = interaction_build(args)
            self.assertEqual(report["agent_interaction_judgment_summary"]["frame_side_count"], 6)
            self.assertEqual(report["factor_rows"], [])


if __name__ == "__main__":
    unittest.main()
