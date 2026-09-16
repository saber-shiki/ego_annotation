from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import optimize
from observations.manifest import write_source_observation_manifest
from utils.state import sha


class _Ledger:
    records = []

    def __init__(self, root):
        self.root = Path(root)
        self.records = []

    def record(self, stage, outputs, **kwargs):
        self.records.append((stage, outputs, kwargs))


class LegacyReferenceEntryTests(unittest.TestCase):
    def test_verified_source_bundle_drives_retained_reference_composition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            assets = root / "assets.json"
            assets.write_text(json.dumps({
                "rgb_python": "/fixture/rgb-python", "lightglue_root": "/fixture/lightglue",
                "sam3d_python": "/fixture/sam3d-python", "sam3d_repo": "/fixture/sam3d",
                "sam3d_config": "/fixture/config", "sam3d_moge_checkpoint": "/fixture/moge",
                "sam3d_dinov2_repo": "/fixture/dino", "sam3d_dinov2_checkpoint": "/fixture/dino.pth",
                "mano_models": "/fixture/mano",
            }))
            paths = {}
            names = {
                "code_manifest": "code_manifest", "model_asset_manifest": "model_asset_manifest",
                "raw_video": "raw_video", "raw_frames": "raw_frame_manifest",
                "camera_calibration": "camera_calibration", "conditioned_depth": "conditioned_depth",
                "hawor_world": "raw_hawor_world_mano", "mano_bridge": "raw_mano_bridge",
                "visible_annotations": "visible_metric_annotations", "completion_report": "reconstructed_shape_contract",
                "reconstructed_mesh": "reconstructed_object_mesh", "appearance_masks": "appearance_mask_observations",
                "appearance_anchor_decision": "appearance_anchor_decision",
                "interaction_judgments": "contact_visibility_occlusion_evidence",
                "interaction_factors": "qualified_contact_visibility_factors", "raw_object_masks": "raw_object_masks",
                "owned_object_masks": "depth_order_owned_object_masks", "metric_eligibility": "metric_surface_eligibility",
            }
            for name, role in names.items():
                path = root / "source" / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
                paths[name] = (path, role)
            contract = root / "source" / "runtime_input_contract.json"
            contract.write_text(json.dumps({"run_id": "reference-test", "assets_manifest_sha256": sha(assets)}))
            paths["runtime_input_contract"] = (contract, "runtime_input_contract")
            paths["appearance_masks"][0].write_text(json.dumps({"frames": [{"frame_idx": 0, "raw_rgb_path": str(root / "raw.jpg"), "appearance_mask_path": str(root / "mask.png")}]}))
            paths["appearance_anchor_decision"][0].write_text(json.dumps({"selected_frame_idx": 0}))
            bundle = root / "source_observations.json"
            write_source_observation_manifest(
                bundle, run_root=root, case="case", object_id="object", track_id="track",
                frame_ids=[0], time_s=[0.0], source_timing={"frame_count": 1}, artifacts=paths,
            )
            calls = []

            def observed(argv):
                out = Path(argv[argv.index("--output-dir") + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / "v18_compact_rigid_object_pose_fit_report.json").write_text("{}")
                calls.append("p14")

            def worker(command, *, output, **kwargs):
                if any("rgb_tracking" in str(value) for value in command):
                    Path(command[command.index("--output-npz") + 1]).parent.mkdir(parents=True, exist_ok=True)
                    Path(command[command.index("--output-npz") + 1]).write_bytes(b"npz")
                    Path(command[command.index("--output-json") + 1]).write_text("{}")
                    calls.append("rgb")
                else:
                    name = command[command.index("--case") + 1].split("|")[0]
                    qc = Path(command[command.index("--output-dir") + 1]) / name / "qc_sam3d_objects_mesh_v7.json"
                    qc.parent.mkdir(parents=True, exist_ok=True)
                    qc.write_text("{}")
                    calls.append("native")
                return {"uuid": "fixture-gpu"}

            def refined(*args):
                output = Path(args[6]); output.write_text(json.dumps({"inputs": {}})); calls.append("p15")

            def solved(problem):
                output = Path(problem.output); output.mkdir(parents=True, exist_ok=True)
                (output / "pose_mesh.json").write_text("{}")
                calls.append("shape")
                return {"schema": "rgb_scene_visual_refined_binding_v1"}

            def visual(problem, root):
                problem.output_dir.mkdir(parents=True, exist_ok=True)
                (problem.output_dir / "demo_state.npz").write_bytes(b"state")
                (problem.output_dir / "fit_report.json").write_text("{}")
                calls.append("visual")
                return {"status": "fixture"}

            with patch("observations.object.pose.fit_observed_pose", side_effect=observed), \
                 patch.object(optimize, "_reference_worker", side_effect=worker), \
                 patch.object(optimize, "refine", side_effect=refined), \
                 patch.object(optimize, "solve", side_effect=solved), \
                 patch.object(optimize, "fit_prepared", side_effect=visual), \
                 patch.object(optimize, "RunState", _Ledger):
                result = optimize.reference_from_source_observations(
                    bundle, assets, root / "legacy_reference", gpu=0, device="cpu", iterations=1
                )
            self.assertEqual(calls, ["p14", "rgb", "p15", "native", "shape", "visual"])
            self.assertTrue(Path(result["stages"]["visual_state"]).is_file())
            self.assertTrue((root / "legacy_reference" / "legacy_reference.json").is_file())


if __name__ == "__main__":
    unittest.main()
