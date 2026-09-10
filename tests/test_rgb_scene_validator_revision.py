import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rgb_scene_recovery as recovery
from rgb_scene_lineage_base import RunLineage
from test_rgb_scene_recovery import RecoveryFixture, write
from v19_runtime_contracts import load, sha


def refresh_original_events(fixture):
    prefix = b"".join((json.dumps(event) + "\n").encode() for event in fixture.prefix_events)
    events = prefix + b"".join((json.dumps(event) + "\n").encode() for event in fixture.suffix_events)
    write(fixture.events_path, events)
    stop = load(fixture.stop)
    stop.update(original_events_sha256=hashlib.sha256(events).hexdigest(),
                original_events_bytes=len(events), retained_event_prefix_bytes=len(prefix),
                retained_event_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    write(fixture.stop, stop)


def terminal_evidence(fixture, path):
    manifest = load(fixture.new / "runtime/bundle_manifest.json")
    return write(path, {
        "workflow": str(uuid.uuid4()), "actor": str(uuid.uuid4()),
        "state": "failed", "process_terminal": "observed",
        "pipeline_error": "P09-propose pre-producer ancestry validation failure",
        "completed_phases": ["P09_DEPTH_ORDER"],
        "alias_sha256": fixture.contract["input_video_sha256"],
        "birth_video_sha256": fixture.contract["input_video_sha256"],
        "source_state": {"cwd": str(fixture.new),
                         "head": manifest.get("source_commit"), "git_status": ""},
    })


class RetainedInputAliasTests(unittest.TestCase):
    def fixture(self):
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        return RecoveryFixture(td.name)

    def recorded_alias(self, fixture, data=None):
        raw = Path(fixture.contract["input_video"])
        alias = write(fixture.root / "input/hawor_sequence/rgb_scene_fixture.mp4",
                      raw.read_bytes() if data is None else data)
        p04 = fixture.prefix_events[list(recovery.RETAINED_PHASES).index("P04_HaWoR")]
        p04["input_sha256"] = {str(alias): sha(alias)}
        refresh_original_events(fixture)
        return alias

    def test_recorded_birth_equivalent_alias_is_a_valid_recovery_ancestor(self):
        f = self.fixture(); alias = self.recorded_alias(f)
        f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        output = write(f.root / "measurements/object_tracks/depth_order_owned/arbitrary_track/new.json", {"new": True})
        f.append_recovery_event("P09_DEPTH_ORDER", [output], inputs={alias: sha(alias)})
        context = recovery.verify_applied_recovery(f.root, f.new)
        self.assertEqual(context["events_path"], f.root / recovery.RECOVERY_EVENTS)

    def test_recorded_alias_mutation_and_non_birth_bytes_are_rejected(self):
        f = self.fixture(); alias = self.recorded_alias(f)
        f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        alias.write_bytes(b"mutated")
        with self.assertRaisesRegex(ValueError, "retained producer input changed"):
            recovery.verify_applied_recovery(f.root, f.new)

        f = self.fixture(); self.recorded_alias(f, b"not the birth video")
        f.prepare()
        with self.assertRaisesRegex(ValueError, "birth alias|birth"):
            recovery.apply_recovery(f.package, process_rows=[])

    def test_unrecorded_same_byte_alias_and_foreign_path_are_rejected(self):
        f = self.fixture()
        alias = write(f.root / "input/hawor_sequence/unrecorded.mp4",
                      Path(f.contract["input_video"]).read_bytes())
        f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        output = write(f.root / "measurements/object_tracks/depth_order_owned/arbitrary_track/new.json", {"new": True})
        f.append_recovery_event("P09_DEPTH_ORDER", [output], inputs={alias: sha(alias)})
        with self.assertRaisesRegex(ValueError, "lacks completed bound ancestor"):
            recovery.verify_applied_recovery(f.root, f.new)
        retained = f.prefix_events[list(recovery.RETAINED_PHASES).index("P04_HaWoR")]
        retained["input_sha256"] = {str(f.base / "foreign.mp4"): "0" * 64}
        with self.assertRaisesRegex(ValueError, "escapes run root"):
            recovery._retained_input_seeds(f.root, [retained], f.contract)


class ValidatorRevisionTests(unittest.TestCase):
    def fixture(self):
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        f = RecoveryFixture(td.name)
        f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        output = write(f.root / "measurements/object_tracks/depth_order_owned/arbitrary_track/new.json", {"new": True})
        event = f.append_recovery_event("P09_DEPTH_ORDER", [output])
        recovery.verify_applied_recovery(f.root, f.new)
        return f, output, event

    def revised_bundle(self, f, *, extra_change=None, asset_change=False):
        revised = f.base / "revised_bundle"
        shutil.copytree(f.new, revised)
        (revised / "runtime/bundle_manifest.json").unlink()
        write(revised / "scripts/rgb_scene_recovery.py", {"validator": "fixed"})
        write(revised / "scripts/rgb_scene_lineage_base.py", {"stamp": "validator revision"})
        config = load(revised / "runtime/runtime_config.json")
        config["copied_root"] = str(revised / "asset")
        write(revised / "runtime/runtime_config.json", config)
        if extra_change:
            write(revised / extra_change, {"unexpected": True})
        f._manifest(revised)
        if asset_change:
            manifest = load(revised / "runtime/bundle_manifest.json")
            manifest["external_assets"] = {"files": {"model": {"sha256": "changed"}}, "trees": {}}
            write(revised / "runtime/bundle_manifest.json", manifest)
        return revised

    def prepare_revision(self, f, revised):
        evidence = terminal_evidence(f, f.base / "terminal_actor.json")
        return recovery.prepare_validator_revision(
            f.root, f.new, revised, evidence, process_rows=[])

    def test_exact_prior_receipt_is_preserved_and_new_receipt_is_revision_stamped(self):
        f, depth_output, old_event = self.fixture(); revised = self.revised_bundle(f)
        revision_path = self.prepare_revision(f, revised)
        context = recovery.verify_validator_revision(f.root, revised)
        self.assertEqual(context["validator_revision"]["resume_phase"], "P09-propose")
        self.assertTrue(recovery.event_code_is_valid(
            old_event, recovery.code_identity(load(revised / "runtime/bundle_manifest.json")), context))
        lineage = RunLineage(f.root, revised, revised / "runtime/runtime_config.json")
        proposal = write(f.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_proposals.json",
                         {"object_id": "arbitrary_object", "review_candidates": []})
        event = {"phase": "P09-propose", "returncode": 0, "outputs": [str(proposal)],
                 "input_contract_sha256": sha(f.contract_path)}
        lineage.success(event, {str(depth_output): sha(depth_output)}, [proposal], [])
        self.assertEqual(event["execution_bundle_manifest_sha256"],
                         sha(revised / "runtime/bundle_manifest.json"))
        self.assertEqual(event["recovery_contract_sha256"], sha(f.root / recovery.RECOVERY_FILE))
        self.assertEqual(event["validator_revision_sha256"], sha(revision_path))
        unbound = dict(event); unbound.pop("validator_revision_sha256")
        self.assertFalse(recovery.event_code_is_valid(
            unbound, recovery.code_identity(load(revised / "runtime/bundle_manifest.json")), context))
        with (f.root / recovery.RECOVERY_EVENTS).open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        self.assertEqual(RunLineage(f.root, revised).require(proposal), proposal)

    def test_changed_prefix_or_accepted_output_is_rejected(self):
        f, output, _ = self.fixture(); revised = self.revised_bundle(f); self.prepare_revision(f, revised)
        event_path = f.root / recovery.RECOVERY_EVENTS
        original = json.loads(event_path.read_text())
        event_path.write_text(json.dumps(original, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "receipt differs|identity mismatch"):
            recovery.verify_validator_revision(f.root, revised)

        f, output, _ = self.fixture(); revised = self.revised_bundle(f); self.prepare_revision(f, revised)
        output.write_bytes(b"changed accepted output")
        with self.assertRaisesRegex(ValueError, "output changed"):
            recovery.verify_validator_revision(f.root, revised)

    def test_producer_source_model_and_malformed_contract_are_rejected(self):
        f, _, _ = self.fixture(); revised = self.revised_bundle(
            f, extra_change="scripts/rgb_scene_depth_order.py")
        with self.assertRaisesRegex(ValueError, "producer code/assets"):
            self.prepare_revision(f, revised)

        f, _, _ = self.fixture(); revised = self.revised_bundle(f, asset_change=True)
        with self.assertRaisesRegex(ValueError, "external assets|asset"):
            self.prepare_revision(f, revised)

        f, _, _ = self.fixture(); revised = self.revised_bundle(f)
        revision_path = self.prepare_revision(f, revised)
        malformed = load(revision_path); malformed["resume_phase"] = "P09_DEPTH_ORDER"
        write(revision_path, malformed)
        with self.assertRaisesRegex(ValueError, "resume_phase"):
            recovery.verify_validator_revision(f.root, revised)


if __name__ == "__main__":
    unittest.main()
