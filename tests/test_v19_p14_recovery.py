"""P09 then P14 continuation preserves original and recovered event identities."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_v19_run_recovery_lineage as fixture
from v19_run_lineage import RunLineage
from v19_runtime_contracts import load, sha, run_command
from v19_run_recovery import (
    RECOVERY_FILE,
    RECOVERY_DIR,
    P14_RECOVERY_FILE,
    P14_RECOVERY_DIR,
    P14_SCHEMA,
    code_identity,
    event_code_is_valid,
    recovery_context,
    run_identity,
    validate_transition,
)


class P14RecoveryChainTests(unittest.TestCase):
    def setUp(self):
        f = fixture.RecoveryLineageIntegration()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.root = f.root
        self.source = f.new
        self.target = f.home / "third_bundle"
        shutil.copytree(self.source, self.target)
        (self.target / "scripts/v19_run_recovery.py").write_text(
            "second explicitly reviewed transition fixture"
        )
        fixture.dump(
            self.target / "runtime/runtime_config.json",
            {"model": "same", "root": str(self.target / "copied_model")},
        )
        (self.target / "runtime/bundle_manifest.json").unlink()
        f.make_manifest(self.target)
        self.old = load(self.source / "runtime/bundle_manifest.json")
        self.new = load(self.target / "runtime/bundle_manifest.json")
        predecessor = load(self.root / RECOVERY_FILE)
        predecessor["selected_anchor_frame_idx"] = 92
        fixture.dump(self.root / RECOVERY_FILE, predecessor)
        original_lineage = RunLineage(
            self.root, self.source, self.source / "runtime/runtime_config.json"
        )
        self.p09 = fixture.dump(
            self.root / "measurements/p09_result.json",
            {"original_current_geometry": True},
        )
        event = {"phase": "P09", "returncode": 0, "outputs": [str(self.p09)]}
        original_lineage.success(
            event, {str(f.original): sha(f.original)}, [self.p09], []
        )
        with f.events.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        self.p09_event = event
        self.completion = fixture.dump(
            self.root
            / "measurements/geometry_completion/sam3d_thing/completion_report.json",
            {"source": "unchanged SAM3D"},
        )
        self.annotations = fixture.dump(
            self.root
            / "measurements/object_geometry/visible_geometry/thing/annotations_v19_visible_geometry.json",
            {"source": "unchanged P09"},
        )
        self.blocker = fixture.dump(
            self.root / "state/runtime_blockers/P14_visible_pose.json",
            {"phase": "P14_visible_pose", "returncode": 1},
        )
        directory = self.root / P14_RECOVERY_DIR
        directory.mkdir(parents=True)
        for name in ("bundle_manifest.json", "runtime_config.json"):
            shutil.copyfile(
                self.source / "runtime" / name, directory / ("source_" + name)
            )
        prefix = f.events.read_bytes()
        (directory / "source_events.jsonl").write_bytes(prefix)
        changed = validate_transition(
            self.old,
            self.new,
            load(self.source / "runtime/runtime_config.json"),
            load(self.target / "runtime/runtime_config.json"),
            self.source,
            self.target,
            phase="P14",
        )
        protected = [
            self.root / "state/anchor_decisions/thing.json",
            self.root / "measurements/object_candidates/object_plan_agent.json",
            self.blocker,
            self.completion,
            self.annotations,
        ]
        self.metadata = {
            "schema": P14_SCHEMA,
            "mode": "same_run_recovery",
            "resume_phase": "P14",
            "object_id": "thing",
            "run_identity": run_identity(f.contract),
            "input_contract_sha256": sha(f.contract_path),
            "source_bundle": str(self.source),
            "target_bundle": str(self.target),
            "previous_recovery_sha256": sha(self.root / RECOVERY_FILE),
            "target_bundle_manifest_sha256": sha(
                self.target / "runtime/bundle_manifest.json"
            ),
            "changed_bundle_files": changed,
            "source_events_bytes": len(prefix),
            "source_events_sha256": hashlib.sha256(prefix).hexdigest(),
            "protected_inputs": {str(p): sha(p) for p in protected},
        }
        fixture.dump(self.root / P14_RECOVERY_FILE, self.metadata)

    def context(self):
        return recovery_context(self.root, self.target, self.f.contract, self.new)

    def test_preserves_both_previous_code_identities_and_binds_new_events(self):
        context = self.context()
        current = code_identity(self.new)
        self.assertTrue(event_code_is_valid(self.f.old_event, current, context))
        self.assertTrue(event_code_is_valid(self.p09_event, current, context))
        lineage = RunLineage(
            self.root, self.target, self.target / "runtime/runtime_config.json"
        )
        self.assertEqual(lineage.require(self.p09), self.p09)
        event = {"phase": "P14_anchor_bound_visible_pose", "returncode": 0}
        path = fixture.dump(
            self.root / "measurements/new_pose.json", {"fixture_pose": True}
        )
        lineage.success(event, {str(self.p09): sha(self.p09)}, [path], [])
        self.assertTrue(event_code_is_valid(event, current, context))
        self.assertEqual(event["run_identity"], self.p09_event["run_identity"])
        self.assertNotEqual(event["code_sha256"], self.p09_event["code_sha256"])
        bad = dict(event)
        bad.pop("recovery_contract_sha256")
        self.assertFalse(event_code_is_valid(bad, current, context))

    def test_retargeted_predecessor_or_changed_old_P09_event_is_rejected(self):
        bad = {**self.metadata, "previous_recovery_sha256": "0" * 64}
        fixture.dump(self.root / P14_RECOVERY_FILE, bad)
        with self.assertRaisesRegex(ValueError, "predecessor"):
            self.context()
        fixture.dump(self.root / P14_RECOVERY_FILE, self.metadata)
        self.f.events.write_bytes(
            self.f.events.read_bytes().replace(b'"phase": "P09"', b'"phase": "P00"')
        )
        with self.assertRaisesRegex(ValueError, "event prefix"):
            self.context()

    def test_arbitrary_third_transition_or_solver_change_is_rejected(self):
        bad = copy.deepcopy(self.new)
        bad["files"][
            "scripts/fit_v18_compact_rigid_object_pose.py"
        ] = "different_solver"
        with self.assertRaisesRegex(ValueError, "prediction/model"):
            validate_transition(
                self.old, bad, {}, {}, self.source, self.target, phase="P14"
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_transition(
                self.old, self.new, {}, {}, self.source, self.target, phase="P18"
            )
        bad = {**self.metadata, "resume_phase": "P18"}
        fixture.dump(self.root / P14_RECOVERY_FILE, bad)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.context()

    def test_P14_failure_does_not_overwrite_either_original_blocker(self):
        p09block = self.root / "state/runtime_blockers/P09.json"
        before = {p: p.read_bytes() for p in [p09block, self.blocker, self.f.events]}
        lineage = RunLineage(
            self.root, self.target, self.target / "runtime/runtime_config.json"
        )
        with self.assertRaisesRegex(
            RuntimeError, "P14_anchor_bound_visible_pose failed"
        ):
            run_command(
                "P14_anchor_bound_visible_pose",
                [sys.executable, "-c", "raise SystemExit(9)"],
                [],
                root=self.root,
                cwd=self.target,
                env={},
                lineage=lineage,
            )
        failure = load(self.root / P14_RECOVERY_DIR / "failure.json")
        self.assertEqual(failure["returncode"], 9)
        self.assertEqual(p09block.read_bytes(), before[p09block])
        self.assertEqual(self.blocker.read_bytes(), before[self.blocker])
        self.assertTrue(self.f.events.read_bytes().startswith(before[self.f.events]))
        self.context()

    def test_preparer_requires_exact_failure_and_never_mutates_successful_outputs(self):
        import prepare_v19_p14_recovery as prep
        from types import SimpleNamespace

        shutil.rmtree(self.root / P14_RECOVERY_DIR)
        old = RunLineage(
            self.root, self.source, self.source / "runtime/runtime_config.json"
        )
        for phase in ("P11", "P12_P13_SAM3D"):
            event = {"phase": phase, "returncode": 0, "outputs": []}
            old.success(event, {}, [], [])
            with self.f.events.open("a") as handle:
                handle.write(json.dumps(event) + "\n")
        log = self.root / "logs/integration/P14_visible_pose.log"
        log.write_text(
            "RuntimeError: P14 completion lacks a byte-bound selected-anchor evidence report"
        )
        (self.root / "measurements/pose_fits/thing/visible").mkdir(parents=True)
        fixture.dump(
            self.root / "state/object_selection.json",
            {"object_id": "thing", "anchor_frame": 92},
        )
        before = {
            p: p.read_bytes()
            for p in [
                self.f.contract_path,
                self.f.events,
                self.root / RECOVERY_FILE,
                self.completion,
                self.annotations,
                self.blocker,
                log,
            ]
        }
        fake = SimpleNamespace(
            require=lambda p: Path(p), verification_batch=old.verification_batch
        )
        with (
            patch.object(
                prep,
                "verify_bundle",
                side_effect=[
                    {**self.old, "preflight_status": "imports_checked"},
                    {**self.new, "preflight_status": "imports_checked"},
                ],
            ),
            patch.object(prep, "RunLineage", return_value=fake),
        ):
            result = prep.prepare_p14_recovery(self.root, self.source, self.target)
        self.assertEqual(result, self.root / P14_RECOVERY_FILE)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual(
            load(result)["previous_recovery_sha256"], sha(self.root / RECOVERY_FILE)
        )
        self.context()
        with self.assertRaisesRegex(ValueError, "new explicit"):
            prep.prepare_p14_recovery(self.root, self.source, self.target)

    def test_native_request_uses_P14_contract_and_separate_handoff(self):
        import prepare_v19_native_launch as launch

        with patch.object(launch, "verify_workspace", return_value={"fixture": True}):
            request = launch.write_request(
                self.target,
                self.root,
                "recovery",
                self.root / P14_RECOVERY_FILE,
                self.f.home / "requests",
            )
        script = Path(request["workflowScriptPath"]).read_text()
        self.assertIn(str(self.root / P14_RECOVERY_FILE), script)
        self.assertIn(
            str(self.root / P14_RECOVERY_DIR / "native_recovery_handoff.md"), script
        )
        profile = (
            Path(__file__).resolve().parents[1]
            / ".pi/agents/ego-annotation.rgb-sam3d-runtime.md"
        ).read_text()
        self.assertIn("P14", profile)
        self.assertIn("Do not rerun geometry/SAM3D", profile)


if __name__ == "__main__":
    unittest.main()
