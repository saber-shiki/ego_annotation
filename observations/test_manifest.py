from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from observations.manifest import (
    SCHEMA,
    build_source_observation_manifest,
    load_source_observation_manifest,
    write_source_observation_manifest,
)


class SourceObservationManifestTests(unittest.TestCase):
    def _artifacts(self, root: Path):
        names = {
            "runtime_input_contract": "runtime_input_contract",
            "code_manifest": "code_manifest",
            "model_asset_manifest": "model_asset_manifest",
            "raw_video": "raw_video",
            "raw_frames": "raw_frame_manifest",
            "camera_calibration": "camera_calibration",
            "conditioned_depth": "conditioned_camera_z_depth",
            "hawor_world": "raw_hawor_world_mano",
            "mano_bridge": "raw_mano_bridge",
            "visible_annotations": "visible_metric_annotations",
            "completion_report": "reconstructed_shape_contract",
            "reconstructed_mesh": "reconstructed_object_mesh",
            "raw_object_masks": "raw_object_masks",
            "owned_object_masks": "depth_order_owned_object_masks",
            "metric_eligibility": "metric_surface_eligibility",
            "interaction_judgments": "contact_visibility_occlusion_evidence",
            "interaction_factors": "qualified_contact_visibility_factors",
        }
        records = {}
        for name, role in names.items():
            path = root / "source" / f"{name}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "runtime_input_contract":
                path.write_text(json.dumps({"run_id": "fixture-run"}))
            else:
                path.write_bytes(name.encode())
            records[name] = (path, role)
        return records

    def test_roundtrip_is_source_only_hash_bound_and_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifacts = self._artifacts(root)
            output = root / "source_observations.json"
            manifest = write_source_observation_manifest(
                output,
                run_root=root,
                case="case",
                object_id="object",
                track_id="track",
                frame_ids=[0, 1],
                time_s=[0.0, 1 / 30],
                source_timing={"fps_ratio": "30", "frame_count": 2},
                artifacts=artifacts,
            )
            self.assertEqual(manifest["schema"], SCHEMA)
            self.assertFalse(manifest["contains_optimized_state"])
            self.assertFalse(manifest["contains_policy_results"])
            self.assertEqual(load_source_observation_manifest(output), manifest)
            self.assertTrue(all(not Path(record["path"]).is_absolute() for record in manifest["artifacts"].values()))
            artifacts["hawor_world"][0].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                load_source_observation_manifest(output)

    def test_rejects_downstream_roles_and_incomplete_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifacts = self._artifacts(root)
            artifacts["legacy_result"] = (next(iter(artifacts.values()))[0], "optimized_pose")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                build_source_observation_manifest(
                    run_root=root,
                    case="case",
                    object_id="object",
                    track_id="track",
                    frame_ids=[0],
                    time_s=[0.0],
                    source_timing={},
                    artifacts=artifacts,
                )
            artifacts.pop("legacy_result")
            with self.assertRaisesRegex(ValueError, "timestamps"):
                build_source_observation_manifest(
                    run_root=root,
                    case="case",
                    object_id="object",
                    track_id="track",
                    frame_ids=[0, 1],
                    time_s=[0.0],
                    source_timing={},
                    artifacts=artifacts,
                )

    def test_main_observe_dispatch_does_not_load_optimizer_or_renderer(self):
        import main
        argv = [
            "observe",
            "--video", "/tmp/source.mp4",
            "--run-root", "/tmp/run",
            "--case", "case",
            "--assets-manifest", "/tmp/assets.json",
            "--gpu", "0",
        ]
        with patch.dict(sys.modules, {"legacy_optimize": None, "optimize": None, "render": None}), \
             patch.object(main, "annotate", return_value={"schema": SCHEMA}) as producer:
            result = main.main(argv)
        self.assertEqual(result["schema"], SCHEMA)
        producer.assert_called_once()

    def test_main_source_has_no_numerical_or_render_dispatch(self):
        import ast
        source = (Path(__file__).parents[1] / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_roots.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        self.assertTrue({"legacy_optimize", "optimize", "render"}.isdisjoint(imported_roots))
        for forbidden_command in ("fit-visual", "fit-formal", "formal-tail", "render-dual", "refine-pose"):
            self.assertNotIn(forbidden_command, source)


if __name__ == "__main__":
    unittest.main()
