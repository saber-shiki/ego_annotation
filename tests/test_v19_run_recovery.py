"""Same-run recovery bindings retain original producer identities and bytes."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v19_run_recovery as recovery


class RecoveryBindings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "run"
        self.old = self.base / "old_bundle"
        self.new = self.base / "new_bundle"
        for directory in (self.old, self.new):
            (directory / "scripts").mkdir(parents=True)
            (directory / "runtime").mkdir()
            (directory / "scripts/actual_producer.py").write_text(
                'print("unchanged actual producer")\n'
            )
            (directory / "scripts/v19_run_lineage.py").write_text(
                "old verification" if directory == self.old else "fixed verification"
            )
            self.dump(
                directory / "runtime/runtime_config.json",
                {
                    "main_python": "/shared/venv/bin/python",
                    "copied_model": str(directory / "models"),
                    "gpu": 5,
                },
            )
        self.old_manifest = self.manifest(self.old)
        self.new_manifest = self.manifest(self.new)
        self.directory = self.root / recovery.RECOVERY_DIR
        self.directory.mkdir(parents=True)
        self.contract_path = self.root / "input/runtime_input_contract.json"
        self.contract = {
            "run_id": str(uuid.uuid4()),
            "run_root": str(self.root),
            "input_video_sha256": "raw_hash",
            "bundle_manifest_sha256": recovery.sha(
                self.old / "runtime/bundle_manifest.json"
            ),
        }
        self.dump(self.contract_path, self.contract)
        self.plan = self.root / "measurements/object_candidates/object_plan_agent.json"
        self.anchor = self.root / "state/anchor_decisions/thing.json"
        self.blocker = self.root / "state/runtime_blockers/P09.json"
        self.dump(self.plan, {"objects": [{"object_id": "thing"}]})
        self.dump(self.anchor, {"object_id": "thing", "selected_anchor_frame_idx": 92})
        self.dump(self.blocker, {"phase_id": "P09", "status": "blocked"})
        self.old_event = {
            "phase": "P09-propose",
            "returncode": 0,
            "run_identity": recovery.run_identity(self.contract),
            "code_sha256": recovery.code_identity(self.old_manifest),
            "input_sha256": {},
            "output_sha256": {"/run/evidence": "original_bytes"},
        }
        self.prefix = (json.dumps(self.old_event) + "\n").encode()
        self.events_path = self.root / "logs/integration/events.jsonl"
        self.events_path.parent.mkdir(parents=True)
        self.events_path.write_bytes(self.prefix)
        (self.directory / "source_events.jsonl").write_bytes(self.prefix)
        (self.directory / "source_bundle_manifest.json").write_bytes(
            (self.old / "runtime/bundle_manifest.json").read_bytes()
        )
        (self.directory / "source_runtime_config.json").write_bytes(
            (self.old / "runtime/runtime_config.json").read_bytes()
        )
        changed = recovery.validate_transition(
            self.old_manifest,
            self.new_manifest,
            recovery.load(self.old / "runtime/runtime_config.json"),
            recovery.load(self.new / "runtime/runtime_config.json"),
            self.old,
            self.new,
        )
        self.metadata = {
            "schema": recovery.SCHEMA,
            "mode": "same_run_recovery",
            "resume_phase": "P09",
            "object_id": "thing",
            "run_identity": recovery.run_identity(self.contract),
            "target_bundle": str(self.new),
            "source_bundle": str(self.old),
            "target_bundle_manifest_sha256": recovery.sha(
                self.new / "runtime/bundle_manifest.json"
            ),
            "input_contract_sha256": recovery.sha(self.contract_path),
            "changed_bundle_files": changed,
            "source_events_bytes": len(self.prefix),
            "source_events_sha256": hashlib.sha256(self.prefix).hexdigest(),
            "protected_inputs": {
                str(p): recovery.sha(p) for p in (self.plan, self.anchor, self.blocker)
            },
        }
        self.dump(self.root / recovery.RECOVERY_FILE, self.metadata)

    def dump(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def manifest(self, directory):
        value = {
            "files": {
                str(p.relative_to(directory)): recovery.sha(p)
                for p in directory.rglob("*")
                if p.is_file()
            },
            "external_assets": {"files": {}, "trees": {}},
        }
        self.dump(directory / "runtime/bundle_manifest.json", value)
        return value

    def context(self):
        return recovery.recovery_context(
            self.root, self.new, self.contract, self.new_manifest
        )

    def test_original_prefix_and_new_records_remain_distinct(self):
        before = {
            p: p.read_bytes()
            for p in (self.contract_path, self.events_path, self.anchor)
        }
        context = self.context()
        current = recovery.code_identity(self.new_manifest)
        self.assertTrue(recovery.event_code_is_valid(self.old_event, current, context))
        old_restamped = {**self.old_event, "phase": "P09"}
        self.assertFalse(recovery.event_code_is_valid(old_restamped, current, context))
        new_event = {
            **old_restamped,
            "code_sha256": current,
            "execution_bundle_manifest_sha256": context["execution_bundle_sha256"],
            "recovery_contract_sha256": context["contract_sha256"],
        }
        self.events_path.write_bytes(
            self.prefix + (json.dumps(new_event) + "\n").encode()
        )
        context = self.context()
        self.assertTrue(recovery.event_code_is_valid(new_event, current, context))
        unbound = dict(new_event)
        del unbound["recovery_contract_sha256"]
        self.assertFalse(recovery.event_code_is_valid(unbound, current, context))
        self.assertEqual(self.contract_path.read_bytes(), before[self.contract_path])
        self.assertEqual(self.anchor.read_bytes(), before[self.anchor])
        self.assertTrue(
            self.events_path.read_bytes().startswith(before[self.events_path])
        )

    def test_ordinary_fresh_run_remains_strict(self):
        self.assertIsNone(
            recovery.recovery_context(
                self.root, self.old, self.contract, self.old_manifest
            )
        )
        self.assertFalse(
            recovery.event_code_is_valid(
                self.old_event, recovery.code_identity(self.new_manifest), None
            )
        )
        (self.root / recovery.RECOVERY_FILE).unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.context()

    def test_unrelated_run_phase_or_bundle_cannot_reuse_receipt(self):
        for key, value in [
            ("resume_phase", "P01"),
            ("target_bundle", "/unrelated/bundle"),
            (
                "run_identity",
                {**self.metadata["run_identity"], "run_id": str(uuid.uuid4())},
            ),
        ]:
            with self.subTest(key=key):
                self.dump(
                    self.root / recovery.RECOVERY_FILE, {**self.metadata, key: value}
                )
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.context()
        self.dump(self.root / recovery.RECOVERY_FILE, self.metadata)

    def test_original_event_or_decision_mutation_is_rejected(self):
        self.events_path.write_bytes(
            self.prefix.replace(b"P09-propose", b"P08-propose")
        )
        with self.assertRaisesRegex(ValueError, "event prefix"):
            self.context()
        self.events_path.write_bytes(self.prefix)
        self.anchor.write_text('{"selected_anchor_frame_idx":0}')
        with self.assertRaisesRegex(ValueError, "protected input changed"):
            self.context()

    def test_snapshot_mutation_cannot_change_birth_code(self):
        p = self.directory / "source_bundle_manifest.json"
        p.write_text(p.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "birth contract"):
            self.context()

    def test_preparer_is_specific_to_stopped_preinvocation_P09(self):
        import prepare_v19_run_recovery as prep

        # Remove only the fixture recovery snapshot; the preparation command
        # must create it exclusively, never replace a prior transition.
        import shutil

        shutil.rmtree(self.directory)
        blocker = {
            "phase_id": "P09",
            "status": "blocked",
            "evidence": {"error": "input changed during byte verification: anchor"},
        }
        self.dump(self.blocker, blocker)
        events = [
            {**self.old_event, "phase": phase}
            for phase in sorted(prep.REQUIRED_COMPLETED)
        ]
        self.events_path.write_text("".join(json.dumps(e) + "\n" for e in events))
        before = {
            p: p.read_bytes()
            for p in [
                self.contract_path,
                self.plan,
                self.anchor,
                self.blocker,
                self.events_path,
            ]
        }
        fake = SimpleNamespace(require=lambda p: Path(p))
        with (
            patch.object(prep, "RunLineage", return_value=fake),
            patch.object(
                prep,
                "verify_bundle",
                side_effect=[
                    {**self.old_manifest, "preflight_status": "imports_checked"},
                    {**self.new_manifest, "preflight_status": "imports_checked"},
                ],
            ),
        ):
            path = prep.prepare_recovery(self.root, self.old, self.new)
        self.assertEqual(path, self.root / recovery.RECOVERY_FILE)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual(recovery.load(path)["resume_phase"], "P09")
        with self.assertRaisesRegex(ValueError, "new explicit"):
            prep.prepare_recovery(self.root, self.old, self.new)

    def test_native_recovery_request_keeps_original_handoff(self):
        import prepare_v19_native_launch as launch

        original = self.root / "state/native_prediction_handoff.md"
        original.write_text("stopped original launch")
        request_dir = self.base / "recovery_request"
        with patch.object(
            launch, "verify_workspace", return_value={"bundle": str(self.new)}
        ):
            request = launch.write_request(
                self.new,
                self.root,
                "recovery",
                self.root / recovery.RECOVERY_FILE,
                request_dir,
            )
        script = Path(request["workflowScriptPath"]).read_text()
        self.assertIn("native-recovery", script)
        self.assertIn(
            str(self.root / recovery.RECOVERY_DIR / "native_recovery_handoff.md"),
            script,
        )
        self.assertEqual(request["context"], "fresh")
        self.assertEqual(original.read_text(), "stopped original launch")

    def test_producer_assets_and_model_settings_cannot_change(self):
        old_config = recovery.load(self.old / "runtime/runtime_config.json")
        new_config = recovery.load(self.new / "runtime/runtime_config.json")
        bad = copy.deepcopy(self.new_manifest)
        bad["files"]["scripts/actual_producer.py"] = "new algorithm"
        with self.assertRaisesRegex(ValueError, "prediction/model code"):
            recovery.validate_transition(
                self.old_manifest, bad, old_config, new_config, self.old, self.new
            )
        bad = copy.deepcopy(self.new_manifest)
        bad["external_assets"]["files"]["new_model.pt"] = {"sha256": "other"}
        with self.assertRaisesRegex(ValueError, "external model/tool"):
            recovery.validate_transition(
                self.old_manifest, bad, old_config, new_config, self.old, self.new
            )
        with self.assertRaisesRegex(ValueError, "execution config"):
            recovery.validate_transition(
                self.old_manifest,
                self.new_manifest,
                old_config,
                {**new_config, "gpu": 4},
                self.old,
                self.new,
            )


if __name__ == "__main__":
    unittest.main()
