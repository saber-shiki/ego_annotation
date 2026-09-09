"""A fixed RGB continuation retains actual prior-stage identities, not fake poses."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_v19_p14_recovery as fixture
from v19_run_lineage import RunLineage
from v19_runtime_contracts import load, sha, run_command
from v19_run_recovery import (
    RECOVERY_FILE,
    P14_RECOVERY_FILE,
    P14_RECOVERY_DIR,
    RGB_RECOVERY_FILE,
    RGB_RECOVERY_DIR,
    RGB_SCHEMA,
    code_identity,
    event_code_is_valid,
    recovery_context,
    run_identity,
    validate_transition,
)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")
    return path


class RGBRecoveryTests(unittest.TestCase):
    def setUp(self):
        f = fixture.P14RecoveryChainTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.root = f.root
        self.source = f.target
        self.target = f.f.home / "rgb_bundle"
        old_record = load(self.root / P14_RECOVERY_FILE)
        old_record["selected_anchor_frame_idx"] = 92
        dump(self.root / P14_RECOVERY_FILE, old_record)
        shutil.copytree(self.source, self.target)
        (self.target / "scripts/build_v20_global_rgb_orientation_edges.py").write_text(
            "reviewed RGB source-target eligibility fixture"
        )
        dump(
            self.target / "runtime/runtime_config.json",
            {"model": "same", "root": str(self.target / "copied_model")},
        )
        (self.target / "runtime/bundle_manifest.json").unlink()
        f.f.make_manifest(self.target)
        self.old = load(self.source / "runtime/bundle_manifest.json")
        self.new = load(self.target / "runtime/bundle_manifest.json")
        lineage = RunLineage(
            self.root, self.source, self.source / "runtime/runtime_config.json"
        )
        self.bound = dump(
            self.root
            / "measurements/geometry_completion/sam3d_thing/anchor_bound_completion_report.json",
            {"same_mesh_binding": True},
        )
        self.pose = dump(
            self.root
            / "measurements/pose_fits/thing/visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json",
            {"status": "ok", "direct": 136, "missing": 14},
        )
        self.p14_events = []
        for phase, output in [
            ("P13_anchor_binding", self.bound),
            ("P14_anchor_bound_visible_pose", self.pose),
        ]:
            event = {"phase": phase, "returncode": 0, "outputs": [str(output)]}
            lineage.success(event, {str(f.p09): sha(f.p09)}, [output], [])
            with f.f.events.open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            self.p14_events.append(event)
        self.blocker = dump(
            self.root / P14_RECOVERY_DIR / "failure.json",
            {"phase": "P14_RGB_observations", "returncode": 1},
        )
        self.log = self.root / "logs/integration/P14_RGB_observations.log"
        self.log.write_text(
            "RuntimeError: initial pose report lacks selected frames: [30,31]"
        )
        directory = self.root / RGB_RECOVERY_DIR
        directory.mkdir(parents=True)
        for name in ("bundle_manifest.json", "runtime_config.json"):
            shutil.copyfile(
                self.source / "runtime" / name, directory / ("source_" + name)
            )
        prefix = f.f.events.read_bytes()
        (directory / "source_events.jsonl").write_bytes(prefix)
        changed = validate_transition(
            self.old,
            self.new,
            load(self.source / "runtime/runtime_config.json"),
            load(self.target / "runtime/runtime_config.json"),
            self.source,
            self.target,
            phase="P14_RGB",
        )
        protected = [
            self.root / "state/anchor_decisions/thing.json",
            self.root / "measurements/object_candidates/object_plan_agent.json",
            self.blocker,
            f.completion,
            f.annotations,
            self.pose,
            self.bound,
        ]
        self.metadata = {
            "schema": RGB_SCHEMA,
            "mode": "same_run_recovery",
            "resume_phase": "P14_RGB",
            "object_id": "thing",
            "run_identity": run_identity(f.f.contract),
            "input_contract_sha256": sha(f.f.contract_path),
            "source_bundle": str(self.source),
            "target_bundle": str(self.target),
            "previous_recovery_sha256": sha(self.root / P14_RECOVERY_FILE),
            "target_bundle_manifest_sha256": sha(
                self.target / "runtime/bundle_manifest.json"
            ),
            "changed_bundle_files": changed,
            "source_events_bytes": len(prefix),
            "source_events_sha256": hashlib.sha256(prefix).hexdigest(),
            "protected_inputs": {str(p): sha(p) for p in protected},
        }
        dump(self.root / RGB_RECOVERY_FILE, self.metadata)

    def context(self):
        return recovery_context(self.root, self.target, self.f.f.contract, self.new)

    def test_three_prior_code_generations_are_verified_without_pose_fill(self):
        before = self.pose.read_bytes()
        context = self.context()
        current = code_identity(self.new)
        for event in [self.f.f.old_event, self.f.p09_event, *self.p14_events]:
            self.assertTrue(event_code_is_valid(event, current, context))
        lineage = RunLineage(
            self.root, self.target, self.target / "runtime/runtime_config.json"
        )
        lineage.require(self.pose)
        self.assertEqual(self.pose.read_bytes(), before)
        self.assertEqual(load(self.pose)["missing"], 14)
        new = {"phase": "P14_RGB_full_timeline_observations", "returncode": 0}
        lineage.success(new, {}, [], [])
        self.assertTrue(event_code_is_valid(new, current, context))
        new["recovery_contract_sha256"] = "0" * 64
        self.assertFalse(event_code_is_valid(new, current, context))

    def test_wrong_predecessor_and_modified_pose_are_rejected(self):
        dump(
            self.root / RGB_RECOVERY_FILE,
            {
                **self.metadata,
                "previous_recovery_sha256": sha(self.root / RECOVERY_FILE),
            },
        )
        with self.assertRaisesRegex(ValueError, "predecessor"):
            self.context()
        dump(self.root / RGB_RECOVERY_FILE, self.metadata)
        self.pose.write_text("changed pose")
        with self.assertRaisesRegex(ValueError, "protected input"):
            self.context()

    def test_RGB_failure_preserves_prior_failures_and_source_records(self):
        prefix = self.f.f.events.read_bytes()
        oldfailure = self.blocker.read_bytes()
        lineage = RunLineage(
            self.root, self.target, self.target / "runtime/runtime_config.json"
        )
        with self.assertRaisesRegex(
            RuntimeError, "P14_RGB_full_timeline_observations failed"
        ):
            run_command(
                "P14_RGB_full_timeline_observations",
                [sys.executable, "-c", "raise SystemExit(6)"],
                [],
                root=self.root,
                cwd=self.target,
                env={},
                lineage=lineage,
            )
        self.assertEqual(self.blocker.read_bytes(), oldfailure)
        self.assertTrue(self.f.f.events.read_bytes().startswith(prefix))
        self.assertEqual(
            load(self.root / RGB_RECOVERY_DIR / "failure.json")["returncode"], 6
        )
        self.context()

    def test_preparer_accepts_only_completed_P14_and_specific_failed_RGB(self):
        import prepare_v19_rgb_recovery as prep

        shutil.rmtree(self.root / RGB_RECOVERY_DIR)
        dump(
            self.root / "state/object_selection.json",
            {"object_id": "thing", "anchor_frame": 92},
        )
        before = {
            p: p.read_bytes()
            for p in [
                self.pose,
                self.bound,
                self.blocker,
                self.f.f.events,
                self.root / P14_RECOVERY_FILE,
            ]
        }
        source_lineage = RunLineage(
            self.root, self.source, self.source / "runtime/runtime_config.json"
        )
        fake = SimpleNamespace(
            require=lambda p: Path(p),
            verification_batch=source_lineage.verification_batch,
        )
        with (
            patch.object(prep, "RunLineage", return_value=fake),
            patch.object(
                prep,
                "verify_bundle",
                side_effect=[
                    {**self.old, "preflight_status": "imports_checked"},
                    {**self.new, "preflight_status": "imports_checked"},
                ],
            ),
        ):
            path = prep.prepare_rgb_recovery(self.root, self.source, self.target)
        self.assertEqual(path, self.root / RGB_RECOVERY_FILE)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        with self.assertRaisesRegex(ValueError, "new explicit"):
            prep.prepare_rgb_recovery(self.root, self.source, self.target)

    def test_native_RGB_handoff_is_separate_and_profile_skips_P14(self):
        import prepare_v19_native_launch as launch

        with patch.object(launch, "verify_workspace", return_value={"fixture": True}):
            request = launch.write_request(
                self.target,
                self.root,
                "recovery",
                self.root / RGB_RECOVERY_FILE,
                self.f.f.home / "request",
            )
        script = Path(request["workflowScriptPath"]).read_text()
        self.assertIn(
            str(self.root / RGB_RECOVERY_DIR / "native_recovery_handoff.md"), script
        )
        profile = (
            Path(__file__).resolve().parents[1]
            / ".pi/agents/ego-annotation.rgb-sam3d-runtime.md"
        ).read_text()
        self.assertIn("--stage pose-rgb", profile)
        self.assertIn("Do not rerun P14 fitting", profile)


if __name__ == "__main__":
    unittest.main()
