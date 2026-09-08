"""Native request preparation is not native execution or model inference."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_v19_native_launch as launch


class NativeLaunchPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "bundle"
        (self.bundle / "runtime").mkdir(parents=True)
        provider = self.root / "provider/extensions/shellgate.ts"
        provider.parent.mkdir(parents=True)
        for path in [
            provider,
            provider.with_name("shellgate-broker.py"),
            provider.parent.parent / "package.json",
            provider.parent.parent / "package-lock.json",
        ]:
            path.write_text("fixture tool source only, never executed")
        original = Path(__file__).resolve().parents[1] / launch.PROFILE
        profile = self.bundle / launch.PROFILE
        profile.parent.mkdir(parents=True)
        text = original.read_text()
        provider_line = next(
            line
            for line in text.splitlines()
            if line.startswith("subagentOnlyExtensions:")
        )
        text = text.replace(
            provider_line, "subagentOnlyExtensions: " + str(provider), 1
        )
        profile.write_text(text)
        (self.bundle / "runtime/runtime_config.json").write_text("{}")
        files = {
            str(p.relative_to(self.bundle)): launch.sha(p)
            for p in self.bundle.rglob("*")
            if p.is_file()
        }
        self.manifest = self.bundle / "runtime/bundle_manifest.json"
        providers = launch.profile_contract(profile)["provider_files_sha256"]
        external = {
            p: {"sha256": digest, "resolved_path": str(Path(p).resolve())}
            for p, digest in providers.items()
        }
        self.manifest.write_text(
            json.dumps(
                {
                    "preflight_status": "imports_checked",
                    "files": files,
                    "external_assets": {"files": external, "trees": {}},
                }
            )
        )

    def test_probe_request_is_native_fresh_and_does_not_launch(self):
        before = {str(p): launch.sha(p) for p in self.bundle.rglob("*") if p.is_file()}
        root, request_dir = self.root / "probe", self.root / "requests"
        request = launch.prepare_probe(self.bundle, root, request_dir)
        self.assertEqual(request["context"], "fresh")
        self.assertEqual(request["agentScope"], "project")
        self.assertEqual(request["cwd"], str(self.bundle))
        self.assertTrue(request["async"])
        self.assertFalse(request["worktree"])
        code = Path(request["workflowScriptPath"]).read_text()
        self.assertIn("runs.run(", code)
        self.assertIn(launch.AGENT, code)
        self.assertIn("tool_probe", code)
        self.assertNotIn("runs.host", code)
        contract = launch.load(root / "probe_contract.json")
        self.assertEqual(
            launch.sha(contract["input_marker_file"]), contract["input_marker_sha256"]
        )
        image = Path(contract["input_image"])
        self.assertTrue(image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(image.stat().st_size, 100000)
        self.assertFalse((root / "tool_probe_result.json").exists())
        record = launch.load(request_dir / "registration.json")
        self.assertEqual(record["native_harness_execution"], "pending")
        self.assertFalse(record["prediction_executed"])
        self.assertEqual(
            before,
            {str(p): launch.sha(p) for p in self.bundle.rglob("*") if p.is_file()},
        )

    def test_no_inherited_context_or_missing_provider_is_accepted(self):
        profile = self.bundle / launch.PROFILE
        original = profile.read_text()
        for key in ["inheritProjectContext", "inheritGlobalContext", "inheritSkills"]:
            profile.write_text(original.replace(key + ": false", key + ": true"))
            with self.assertRaisesRegex(ValueError, "context mismatch"):
                launch.profile_contract(profile)
        profile.write_text(
            original.replace(
                "thinking: high", "thinking: high\ndefaultReads: /outside/AGENTS.md"
            )
        )
        with self.assertRaisesRegex(ValueError, "undeclared native profile"):
            launch.profile_contract(profile)
        profile.write_text(original)
        (self.root / "provider/extensions/shellgate-broker.py").unlink()
        with self.assertRaisesRegex(ValueError, "provider dependency"):
            launch.profile_contract(profile)

    def test_changed_or_unregistered_bundle_context_rejected(self):
        profile = self.bundle / launch.PROFILE
        profile.write_text(profile.read_text() + "\nchanged")
        with self.assertRaisesRegex(ValueError, "changed/missing"):
            launch.verify_workspace(self.bundle)
        manifest = launch.load(self.manifest)
        manifest["files"][str(launch.PROFILE)] = launch.sha(profile)
        self.manifest.write_text(json.dumps(manifest))
        (self.bundle / "AGENTS.md").write_text("foreign instructions")
        with self.assertRaisesRegex(ValueError, "unregistered"):
            launch.verify_workspace(self.bundle)

    def test_probe_requires_fresh_separate_paths(self):
        root = self.root / "existing"
        root.mkdir()
        with self.assertRaisesRegex(ValueError, "new isolated"):
            launch.prepare_probe(self.bundle, root, self.root / "requests")
        with self.assertRaisesRegex(ValueError, "separate"):
            launch.prepare_probe(self.bundle, self.root / "same", self.root / "same")
        self.assertFalse((self.root / "same").exists())

    def test_provider_mutation_cannot_be_redeclared_at_request_time(self):
        broker = self.root / "provider/extensions/shellgate-broker.py"
        broker.write_text("changed after candidate preparation")
        with self.assertRaisesRegex(ValueError, "provider differs"):
            launch.prepare_probe(
                self.bundle, self.root / "probe", self.root / "request"
            )
        self.assertFalse((self.root / "probe").exists())

    def test_prediction_request_binds_prepared_input_and_rejects_cached_outputs(self):
        root = self.root / "run"
        (root / "input").mkdir(parents=True)
        video = root / "input/source.mp4"
        video.write_bytes(b"timing probe mocked in this planning-only test")
        contract = {
            "run_id": "new-run",
            "bundle_manifest_sha256": launch.sha(self.manifest),
            "input_video": str(video),
            "input_video_sha256": launch.sha(video),
        }
        (root / "input/runtime_input_contract.json").write_text(json.dumps(contract))
        with (
            patch.object(launch, "read_cfr_video_timing") as timing,
            patch.object(launch, "verify_bundle") as full,
        ):
            result = launch.prepare_prediction_request(
                self.bundle, root, self.root / "prediction_request"
            )
            timing.assert_called_once_with(video)
            full.assert_called_once_with(self.bundle)
        self.assertEqual(result["context"], "fresh")
        self.assertIn("prediction", Path(result["workflowScriptPath"]).read_text())
        (root / "measurements").mkdir()
        (root / "measurements/old_pose.json").write_text("{}")
        with patch.object(launch, "read_cfr_video_timing"):
            with self.assertRaisesRegex(ValueError, "pre-existing predictions"):
                launch.prepare_prediction_request(
                    self.bundle, root, self.root / "second_request"
                )


if __name__ == "__main__":
    unittest.main()
