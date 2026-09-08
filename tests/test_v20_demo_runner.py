"""Lightweight launcher tests: no model, inference, GPU or ffmpeg execution."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_v20_visual_contact_demo.py"
SPEC = importlib.util.spec_from_file_location("demo_runner_tests", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class DemoRunner(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "input video.mp4"
        self.video.write_bytes(b"test-only video placeholder")
        self.state = self.root / "demo_state.npz"
        self.state.write_bytes(b"test-only state placeholder")
        self.output = self.root / "new output"

    def args(self, *extra):
        return runner.parser().parse_args(
            [
                "--source-video",
                str(self.video),
                "--output-dir",
                str(self.output),
                *extra,
            ]
        )

    def fit_args(self):
        values = ["--mode", "fit-render"]
        for name in runner.FIT_INPUTS[:-1]:
            path = self.root / f"{name}.json"
            path.write_text("{}")
            values.extend(["--" + name.replace("_", "-"), str(path)])
        models = self.root / "mano_models"
        models.mkdir()
        for side in ("LEFT", "RIGHT"):
            (models / f"MANO_{side}.pkl").write_bytes(b"model placeholder")
        values.extend(["--mano-models", str(models)])
        return self.args(*values)

    def test_replay_has_no_fitting_step(self):
        args = self.args("--mode", "replay", "--state", str(self.state))
        plan = runner.build_plan(args)
        self.assertEqual([name for name, _ in plan], ["render"])
        self.assertIn(str(self.state), plan[0][1])
        self.assertIn(str(self.video), plan[0][1])
        self.assertFalse(self.output.exists())

    def test_fit_render_uses_new_state(self):
        args = self.fit_args()
        plan = runner.build_plan(args)
        self.assertEqual([name for name, _ in plan], ["fit", "render"])
        self.assertIn(str(self.output / "fit/demo_state.npz"), plan[1][1])
        self.assertNotIn(str(self.state), plan[1][1])
        self.assertFalse(self.output.exists())

    def test_existing_output_is_rejected(self):
        self.output.mkdir()
        args = self.args("--mode", "replay", "--state", str(self.state))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            runner.build_plan(args)

    def test_fit_input_and_model_paths_required(self):
        args = self.args("--mode", "fit-render")
        with self.assertRaisesRegex(ValueError, "requires inputs"):
            runner.build_plan(args)
        args = self.fit_args()
        (args.mano_models / "MANO_RIGHT.pkl").unlink()
        with self.assertRaisesRegex(ValueError, "MANO asset not found"):
            runner.build_plan(args)

    def test_mode_inputs_cannot_be_mixed(self):
        args = self.fit_args()
        args.state = self.state
        with self.assertRaisesRegex(ValueError, "replay mode"):
            runner.build_plan(args)
        args = self.args(
            "--mode",
            "replay",
            "--state",
            str(self.state),
            "--annotations",
            str(self.video),
        )
        with self.assertRaisesRegex(ValueError, "does not consume fit inputs"):
            runner.build_plan(args)

    def test_dry_run_does_not_launch_or_write(self):
        argv = [
            str(SCRIPT),
            "--mode",
            "replay",
            "--state",
            str(self.state),
            "--source-video",
            str(self.video),
            "--output-dir",
            str(self.output),
            "--dry-run",
        ]
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", argv),
            patch.object(runner.subprocess, "run") as launch,
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(runner.main(), 0)
        launch.assert_not_called()
        self.assertIn("path validation only", stdout.getvalue())
        self.assertIn("render:", stdout.getvalue())
        self.assertFalse(self.output.exists())

    def test_failed_fit_stops_before_render_and_records_error(self):
        args = self.fit_args()
        plan = runner.build_plan(args)
        with (
            patch.object(
                runner.subprocess, "run", return_value=SimpleNamespace(returncode=7)
            ) as launch,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(runner.execute(args, plan), 7)
        self.assertEqual(launch.call_count, 1)
        report = json.loads((self.output / "run_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["steps"][0]["returncode"], 7)
        self.assertFalse((self.output / "render.log").exists())
        self.assertIs(report["is_v19_fresh_run"], False)

    def test_launch_oserror_is_recorded(self):
        args = self.args("--mode", "replay", "--state", str(self.state))
        with (
            patch.object(
                runner.subprocess,
                "run",
                side_effect=OSError("synthetic launch failure"),
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(runner.execute(args, runner.build_plan(args)), 1)
        report = json.loads((self.output / "run_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("synthetic launch failure", report["steps"][0]["launch_error"])

    def test_completed_execution_does_not_claim_visual_acceptance(self):
        args = self.args("--mode", "replay", "--state", str(self.state))
        with (
            patch.object(
                runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(runner.execute(args, runner.build_plan(args)), 0)
        report = json.loads((self.output / "run_report.json").read_text())
        self.assertEqual(report["status"], "completed_pending_visual_review")
        self.assertFalse(report["annotation_ready"])
        self.assertFalse(report["is_v19_fresh_run"])
        self.assertEqual(report["inputs"]["state"]["sha256"], runner.sha256(self.state))


if __name__ == "__main__":
    unittest.main()
