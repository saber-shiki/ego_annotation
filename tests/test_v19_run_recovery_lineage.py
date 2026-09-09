"""Recovery-aware lineage: old producer receipts remain original and immutable."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v19_runtime_contracts import load, sha, run_command
from v19_run_lineage import RunLineage
from v19_run_recovery import (
    SCHEMA,
    RECOVERY_FILE,
    RECOVERY_DIR,
    run_identity,
    validate_transition,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


class RecoveryLineageIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.root = self.home / "run"
        self.old = self.home / "old"
        self.new = self.home / "new"
        for bundle, text in [(self.old, "old verifier"), (self.new, "new verifier")]:
            (bundle / "scripts").mkdir(parents=True)
            (bundle / "runtime").mkdir()
            (bundle / "scripts/producer.py").write_text("same prediction producer")
            (bundle / "scripts/v19_run_lineage.py").write_text(text)
            dump(
                bundle / "runtime/runtime_config.json",
                {"model": "same", "root": str(bundle / "copied_model")},
            )
            self.make_manifest(bundle)
        oldm = load(self.old / "runtime/bundle_manifest.json")
        newm = load(self.new / "runtime/bundle_manifest.json")
        raw = dump(self.root / "input/source.mp4", {"raw": "fixture"})
        self.contract = {
            "run_id": str(uuid.uuid4()),
            "run_root": str(self.root),
            "input_video": str(raw),
            "input_video_sha256": sha(raw),
            "bundle_manifest_sha256": sha(self.old / "runtime/bundle_manifest.json"),
        }
        self.contract_path = dump(
            self.root / "input/runtime_input_contract.json", self.contract
        )
        oldcode = {k: v for k, v in oldm["files"].items() if k.endswith((".py", ".sh"))}
        original = dump(
            self.root / "input/raw_frame_manifest/manifest.json", {"frames": []}
        )
        event = {
            "phase": "P01",
            "returncode": 0,
            "run_identity": run_identity(self.contract),
            "code_sha256": oldcode,
            "input_sha256": {str(raw): sha(raw)},
            "output_sha256": {str(original): sha(original)},
            "outputs": [str(original)],
        }
        events = self.root / "logs/integration/events.jsonl"
        events.parent.mkdir(parents=True)
        prefix = (json.dumps(event) + "\n").encode()
        events.write_bytes(prefix)
        plan = dump(
            self.root / "measurements/object_candidates/object_plan_agent.json",
            {"objects": [{"object_id": "thing"}]},
        )
        anchor = dump(
            self.root / "state/anchor_decisions/thing.json",
            {"selected_anchor_frame_idx": 92},
        )
        blocker = dump(
            self.root / "state/runtime_blockers/P09.json",
            {"phase_id": "P09", "status": "blocked"},
        )
        recdir = self.root / RECOVERY_DIR
        recdir.mkdir(parents=True)
        (recdir / "source_bundle_manifest.json").write_bytes(
            (self.old / "runtime/bundle_manifest.json").read_bytes()
        )
        (recdir / "source_runtime_config.json").write_bytes(
            (self.old / "runtime/runtime_config.json").read_bytes()
        )
        (recdir / "source_events.jsonl").write_bytes(prefix)
        import hashlib

        changed = validate_transition(
            oldm,
            newm,
            load(self.old / "runtime/runtime_config.json"),
            load(self.new / "runtime/runtime_config.json"),
            self.old,
            self.new,
        )
        metadata = {
            "schema": SCHEMA,
            "mode": "same_run_recovery",
            "resume_phase": "P09",
            "run_identity": run_identity(self.contract),
            "input_contract_sha256": sha(self.contract_path),
            "object_id": "thing",
            "source_bundle": str(self.old),
            "target_bundle": str(self.new),
            "target_bundle_manifest_sha256": sha(
                self.new / "runtime/bundle_manifest.json"
            ),
            "changed_bundle_files": changed,
            "source_events_bytes": len(prefix),
            "source_events_sha256": hashlib.sha256(prefix).hexdigest(),
            "protected_inputs": {str(p): sha(p) for p in (plan, anchor, blocker)},
        }
        dump(self.root / RECOVERY_FILE, metadata)
        self.original = original
        self.events = events
        self.old_event = event

    def make_manifest(self, bundle):
        files = {
            str(p.relative_to(bundle)): sha(p) for p in bundle.rglob("*") if p.is_file()
        }
        dump(
            bundle / "runtime/bundle_manifest.json",
            {"files": files, "external_assets": {"files": {}, "trees": {}}},
        )

    def test_old_and_recovered_producers_are_both_verified_not_restamped(self):
        before = self.events.read_bytes()
        lineage = RunLineage(
            self.root, self.new, self.new / "runtime/runtime_config.json"
        )
        self.assertEqual(lineage.require(self.original), self.original)
        self.assertEqual(
            lineage.records[str(self.original)]["code_sha256"],
            self.old_event["code_sha256"],
        )
        output = self.root / "measurements/recovered/output.json"
        output.parent.mkdir(parents=True)
        output.write_text("{}")
        event = {
            "phase": "P09",
            "returncode": 0,
            "input_sha256": {},
            "outputs": [str(output)],
        }
        lineage.success(event, {}, [output], [])
        self.assertEqual(
            event["execution_bundle_manifest_sha256"],
            sha(self.new / "runtime/bundle_manifest.json"),
        )
        self.assertEqual(
            event["recovery_contract_sha256"], sha(self.root / RECOVERY_FILE)
        )
        with self.events.open("ab") as f:
            f.write((json.dumps(event) + "\n").encode())
        new = RunLineage(self.root, self.new, self.new / "runtime/runtime_config.json")
        self.assertEqual(new.require(output), output)
        self.assertTrue(self.events.read_bytes().startswith(before))

    def test_failed_P09_command_preserves_original_blocker_and_prefix(self):
        blocker = self.root / 'state/runtime_blockers/P09.json'
        before_blocker = blocker.read_bytes()
        before_prefix = self.events.read_bytes()
        lineage = RunLineage(self.root, self.new, self.new / 'runtime/runtime_config.json')
        output = self.root / 'measurements/new_P09/not_created.json'
        downstream = self.root / 'downstream_not_allowed'
        with self.assertRaisesRegex(RuntimeError, 'P09 failed'):
            run_command('P09', [sys.executable, '-c', 'raise SystemExit(7)'], [output],
                        root=self.root, cwd=self.new, env={}, lineage=lineage)
            downstream.touch()
        failure_path = self.root / RECOVERY_DIR / 'failure.json'
        failure = load(failure_path)
        self.assertEqual(failure['returncode'], 7)
        self.assertIn('P09 failed', failure['error'])
        self.assertEqual(blocker.read_bytes(), before_blocker)
        self.assertTrue(self.events.read_bytes().startswith(before_prefix))
        self.assertFalse(output.exists())
        self.assertFalse(downstream.exists())
        event = json.loads(self.events.read_text().splitlines()[-1])
        self.assertEqual(event['returncode'], 7)
        self.assertNotIn('output_sha256', event)
        # The failure did not invalidate the transition's protected evidence.
        RunLineage(self.root, self.new, self.new / 'runtime/runtime_config.json')
        saved = failure_path.read_bytes()
        with self.assertRaises(FileExistsError):
            run_command('P09', [sys.executable, '-c', 'raise SystemExit(8)'], [output],
                        root=self.root, cwd=self.new, env={}, lineage=lineage)
        self.assertEqual(failure_path.read_bytes(), saved)
        self.assertEqual(blocker.read_bytes(), before_blocker)

    def test_actor_preinvocation_failure_uses_same_distinct_path(self):
        repository = Path(__file__).resolve().parents[1]
        profile = (repository / '.pi/agents/ego-annotation.rgb-sam3d-runtime.md').read_text()
        spec = (repository / 'runtime/rgb_sam3d_runtime_spec.md').read_text()
        for text in (profile, spec):
            self.assertIn('state/runtime_recovery/P09/failure.json', text)
            self.assertIn('pre-invocation', text)
        self.assertIn('scripts/v19_runtime_contracts.py', __import__('v19_run_recovery').ALLOWED_CHANGED_FILES)
        original = (self.root / 'state/runtime_blockers/P09.json').read_bytes()
        failure = self.root / RECOVERY_DIR / 'failure.json'
        with failure.open('x') as handle:
            handle.write(json.dumps({'phase':'P09','status':'blocked','source':'pre-invocation'}))
        self.assertEqual((self.root / 'state/runtime_blockers/P09.json').read_bytes(), original)
        RunLineage(self.root, self.new, self.new / 'runtime/runtime_config.json')

    def test_unbound_new_code_event_cannot_enter_recovery(self):
        d = json.loads(self.events.read_text().splitlines()[0])
        d["code_sha256"] = {
            k: v
            for k, v in load(self.new / "runtime/bundle_manifest.json")["files"].items()
            if k.endswith(".py")
        }
        self.events.write_text(json.dumps(d) + "\n")
        with self.assertRaisesRegex(ValueError, "event prefix"):
            RunLineage(self.root, self.new, self.new / "runtime/runtime_config.json")


if __name__ == "__main__":
    unittest.main()
