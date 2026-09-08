"""Native request preparation is not native execution or model inference."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
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
        self.assertIn(
            "output: " + json.dumps(str(root / "tool_probe_result.json")), code
        )
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
        self.assertEqual(
            record["declared_agent_output"], str(root / "tool_probe_result.json")
        )
        self.assertFalse(record["prediction_executed"])
        self.assertEqual(
            before,
            {str(p): launch.sha(p) for p in self.bundle.rglob("*") if p.is_file()},
        )

    def test_large_probe_marker_requires_large_native_write_without_creating_result(
        self,
    ):
        root, request_dir = self.root / "large_probe", self.root / "large_request"
        request = launch.prepare_probe(self.bundle, root, request_dir, 8192)
        marker = root / "input_marker.txt"
        self.assertEqual(marker.stat().st_size, 8192)
        contract = launch.load(root / "probe_contract.json")
        self.assertEqual(contract["input_marker_bytes"], 8192)
        self.assertEqual(contract["input_marker_sha256"], launch.sha(marker))
        self.assertIn("complete marker text verbatim", contract["marker_copy_policy"])
        self.assertFalse((root / "tool_probe_result.json").exists())
        self.assertIn(
            json.dumps(str(root / "tool_probe_result.json")),
            Path(request["workflowScriptPath"]).read_text(),
        )
        with self.assertRaisesRegex(ValueError, "marker bytes"):
            launch.prepare_probe(
                self.bundle, self.root / "bad_probe", self.root / "bad_request", 16385
            )
        self.assertFalse((self.root / "bad_probe").exists())

    def test_authored_large_write_keeps_identity_marker_short(self):
        root, request_dir = self.root / "authored_probe", self.root / "authored_request"
        launch.prepare_probe(self.bundle, root, request_dir, min_write_bytes=8192)
        contract = launch.load(root / "probe_contract.json")
        self.assertEqual((root / "input_marker.txt").stat().st_size, 51)
        self.assertEqual(contract["minimum_native_write_bytes"], 8192)
        self.assertIn("transport_padding", contract["required_output_fields"])
        self.assertFalse((root / "tool_probe_result.json").exists())
        with self.assertRaisesRegex(ValueError, "minimum write bytes"):
            launch.prepare_probe(
                self.bundle,
                self.root / "bad_min",
                self.root / "bad_min_request",
                min_write_bytes=65536,
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
        with self.assertRaisesRegex(ValueError, "separate"):
            launch.prepare_probe(
                self.bundle, self.root / "nested", self.root / "nested/requests"
            )
        self.assertFalse((self.root / "nested").exists())
        self.assertFalse((self.root / "same").exists())

    def test_provider_mutation_cannot_be_redeclared_at_request_time(self):
        broker = self.root / "provider/extensions/shellgate-broker.py"
        broker.write_text("changed after candidate preparation")
        with self.assertRaisesRegex(ValueError, "provider differs"):
            launch.prepare_probe(
                self.bundle, self.root / "probe", self.root / "request"
            )
        self.assertFalse((self.root / "probe").exists())

    def prepared_run(self, name="run"):
        root = self.root / name
        for directory in ("input", "logs", "state", "measurements", "renders"):
            (root / directory).mkdir(parents=True)
        video = root / "input/source.mp4"
        video.write_bytes(b"timing probe mocked in this planning-only test")
        contract = {
            "run_id": str(uuid.uuid4()),
            "run_root": str(root),
            "bundle_manifest_sha256": launch.sha(self.manifest),
            "input_video": str(video),
            "input_video_sha256": launch.sha(video),
        }
        (root / "input/runtime_input_contract.json").write_text(json.dumps(contract))
        return root, video, contract

    def test_prediction_request_binds_prepared_input_and_rejects_cached_outputs(self):
        root, video, _ = self.prepared_run()
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
        self.assertIn(
            "output: " + json.dumps(str(root / "state/native_prediction_handoff.md")),
            Path(result["workflowScriptPath"]).read_text(),
        )
        (root / "measurements/old_pose.json").write_text("{}")
        with patch.object(launch, "read_cfr_video_timing"):
            with self.assertRaisesRegex(ValueError, "untouched prepared run"):
                launch.prepare_prediction_request(
                    self.bundle, root, self.root / "second_request"
                )

    def test_invalid_uuid_or_prepared_root_fails_before_expensive_checks(self):
        for index, (key, value, pattern) in enumerate(
            [
                ("run_id", "new-run", "UUID"),
                ("run_id", None, "UUID"),
                ("run_root", None, "root binding"),
                ("run_root", "/other/run", "root binding"),
            ]
        ):
            with self.subTest(key=key, value=value):
                root, _, contract = self.prepared_run("invalid_" + str(index))
                contract[key] = value
                (root / "input/runtime_input_contract.json").write_text(
                    json.dumps(contract)
                )
                with patch.object(launch, "verify_bundle") as full:
                    with self.assertRaisesRegex(ValueError, pattern):
                        launch.prepare_prediction_request(
                            self.bundle, root, self.root / ("request_" + str(index))
                        )
                    full.assert_not_called()

    def test_early_interruption_is_not_an_untouched_prepared_run(self):
        for index, relative in enumerate(
            [
                "state/startup.json",
                "logs/harness_events.jsonl",
                "input/raw_frame_manifest/manifest.json",
                "state/receipts/P00.json",
            ]
        ):
            with self.subTest(relative=relative):
                root, _, _ = self.prepared_run("interrupted_" + str(index))
                extra = root / relative
                extra.parent.mkdir(parents=True, exist_ok=True)
                extra.write_text("{}")
                self.assertFalse(any((root / "measurements").iterdir()))
                self.assertFalse(any((root / "renders").iterdir()))
                with patch.object(launch, "verify_bundle") as full:
                    with self.assertRaisesRegex(ValueError, "untouched prepared run"):
                        launch.prepare_prediction_request(
                            self.bundle,
                            root,
                            self.root / ("blocked_request_" + str(index)),
                        )
                    full.assert_not_called()
        root, _, _ = self.prepared_run("empty_generated_dir")
        (root / "input/raw_frame_manifest").mkdir()
        with self.assertRaisesRegex(ValueError, "untouched prepared run"):
            launch.prepare_prediction_request(
                self.bundle, root, self.root / "empty_dir_request"
            )

    def test_prediction_request_must_be_outside_run_tree(self):
        root, _, _ = self.prepared_run()
        with self.assertRaisesRegex(ValueError, "separate"):
            launch.prepare_prediction_request(self.bundle, root, root / "requests")
        self.assertFalse((root / "requests").exists())


if __name__ == "__main__":
    unittest.main()
