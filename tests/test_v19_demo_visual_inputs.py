import json
import re
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from build_v19_demo_visual_inputs import build, sha256
from fit_v20_visual_contact_demo import appearance_mask_rows

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "additive_demo_v2_runner", SCRIPTS / "run_v19_additive_demo_branch.py"
)
additive_runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(additive_runner)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class DemoVisualInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.root.mkdir()
        self.object_id = "generic-object.7"
        self.ids = [40, 75, 130]
        self.times = [1.0, 1.4, 3.2]
        raw_rows, ann_rows, track = [], [], {}
        self.source_masks = {}
        for position, (idx, time_s) in enumerate(zip(self.ids, self.times)):
            image = np.full((80, 120, 3), 128, np.uint8)
            if position != 2:
                yy, xx = np.indices(image.shape[:2])
                image[:] = ((xx + yy) % 2 * 255)[..., None]
            rgb = self.root / f"input/rgb/{idx}.png"
            rgb.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(rgb), image)
            mask = np.zeros((80, 120), np.uint8)
            if position == 0:  # sharp, interior, moderate area: desired anchor
                mask[20:60, 35:85] = 255
            elif position == 1:  # larger but touches border: rejected
                mask[0:70, 0:110] = 255
            else:  # interior but uniformly blurred RGB: rejected by sharpness
                mask[15:65, 25:95] = 255
            mask_path = self.root / f"measurements/tracks/mask_{idx}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)
            self.source_masks[idx] = mask_path
            raw_rows.append({"frame_idx": idx, "raw_frame_path": str(rgb)})
            geometry = {
                "world_vertices_sample_m": [[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]],
                "camera_vertices_sample_m": [[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]],
                "centroid_world_m": [0.03, 0.03, 1],
            }
            ann_rows.append({
                "frame_idx": idx, "time_s": time_s, "source_width": 120, "source_height": 80,
                "camera": {"T_world_camera_metric": np.eye(4).tolist()},
                "objects": [{
                    "object_id": self.object_id,
                    "mask_path": str(self.root / f"metric_owned_{idx}.png"),
                    "visible_geometry_candidate": geometry,
                }],
            })
            track[str(idx)] = {"visible": True, "mask_path": str(mask_path), "area_px": int((mask > 0).sum())}
        self.raw = dump(self.root / "input/raw_frame_manifest/manifest.json", {"frames": raw_rows})
        self.annotations = dump(self.root / "measurements/annotations.json", {"frames": ann_rows})
        self.track = dump(self.root / "measurements/tracks/sam2_track.json", track)

    def test_materializes_all_frames_and_selects_generic_sharp_interior_anchor(self):
        source_bytes = {idx: path.read_bytes() for idx, path in self.source_masks.items()}
        output = self.root / f"measurements/demo_visual/{self.object_id}/v2/appearance_evidence"
        manifest, anchor = build(self.root, self.object_id, self.raw, self.annotations, self.track,
                                 output, min_sharpness=2.0)
        self.assertEqual(anchor["selected_frame_idx"], 40)
        self.assertEqual(len(manifest["frames"]), 3)
        self.assertFalse(manifest["metric_owned_masks_consumed"])
        self.assertTrue(anchor["planes_are_distinct"])
        self.assertNotEqual(anchor["appearance_conditioning"]["raw_sam2_mask"],
                            anchor["metric_alignment"]["owned_metric_mask_path"])
        for row in manifest["frames"]:
            idx = row["frame_idx"]
            self.assertEqual(row["source_mask_sha256"], row["appearance_mask_sha256"])
            self.assertEqual(Path(row["appearance_mask_path"]).read_bytes(), source_bytes[idx])
            self.assertEqual(self.source_masks[idx].read_bytes(), source_bytes[idx])

    def test_fitter_manifest_validation_rejects_mutation_and_never_reads_owned_mask(self):
        output = self.root / f"measurements/demo_visual/{self.object_id}/v3/appearance_evidence"
        manifest, _ = build(self.root, self.object_id, self.raw, self.annotations, self.track,
                            output, min_sharpness=0)
        paths = appearance_mask_rows(manifest, self.annotations, self.object_id,
                                     np.asarray(self.ids), np.asarray([120, 80]))
        self.assertEqual(set(paths), set(self.ids))
        self.assertTrue(all("appearance_masks" in value for value in paths.values()))
        Path(paths[40]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            appearance_mask_rows(manifest, self.annotations, self.object_id,
                                 np.asarray(self.ids), np.asarray([120, 80]))

    def test_missing_track_frame_and_out_of_run_mask_fail(self):
        payload = json.loads(self.track.read_text())
        payload.pop("75")
        bad_track = dump(self.root / "measurements/tracks/missing.json", payload)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            build(self.root, self.object_id, self.raw, self.annotations, bad_track,
                  self.root / f"measurements/demo_visual/{self.object_id}/missing")
        payload = json.loads(self.track.read_text())
        outside = Path(self.temp.name) / "outside.png"
        outside.write_bytes(self.source_masks[40].read_bytes())
        payload["40"]["mask_path"] = str(outside)
        outside_track = dump(self.root / "measurements/tracks/outside.json", payload)
        with self.assertRaisesRegex(ValueError, "outside run"):
            build(self.root, self.object_id, self.raw, self.annotations, outside_track,
                  self.root / f"measurements/demo_visual/{self.object_id}/outside")

    def test_contract_tamper_aborts_during_preflight_verification(self):
        bound = self.root / "input/bound.json"
        bound.write_text("bound")
        contract = {
            "inputs": {"p15": {"path": str(bound), "sha256": sha256(bound)}},
            "sam2_mask_inputs": {
                "40": {"path": str(self.source_masks[40]), "sha256": sha256(self.source_masks[40])}
            },
            "raw_frame_inputs": {},
        }
        inputs = additive_runner.verified_contract_inputs(contract, self.root)
        self.assertEqual(inputs["p15"], bound.resolve())
        self.source_masks[40].write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "contract input changed"):
            additive_runner.verified_contract_inputs(contract, self.root)

    def test_invisible_stale_mask_is_unavailable_to_anchor_and_fitter(self):
        track = json.loads(self.track.read_text())
        track["75"]["visible"] = False
        self.track.write_text(json.dumps(track))
        out = self.root / f"measurements/demo_visual/{self.object_id}/invisible"
        manifest, anchor = build(self.root, self.object_id, self.raw, self.annotations, self.track, out)
        row = next(r for r in manifest["frames"] if r["frame_idx"] == 75)
        self.assertEqual(row["status"], "unavailable_not_substituted")
        self.assertIsNone(row["appearance_mask_path"])
        self.assertFalse(row["anchor_eligible"])
        paths = appearance_mask_rows(manifest, self.annotations, self.object_id, self.ids, [120, 80])
        self.assertIsNone(paths[75])
        row.update(status="available", appearance_mask_path=str(self.source_masks[75]))
        with self.assertRaisesRegex(ValueError, "invisible"):
            appearance_mask_rows(manifest, self.annotations, self.object_id, self.ids, [120, 80])

    def test_late_mask_failure_keeps_final_namespace_free_and_preserves_debug(self):
        last = self.source_masks[130]
        original = last.read_bytes()
        last.write_bytes(b"invalid late image")
        out = self.root / f"measurements/demo_visual/{self.object_id}/atomic"
        with self.assertRaisesRegex(ValueError, "dimensions"):
            build(self.root, self.object_id, self.raw, self.annotations, self.track, out)
        self.assertFalse(out.exists())
        self.assertTrue(list(out.parent.glob(".atomic.failed-*/appearance_masks/*")))
        last.write_bytes(original)
        manifest, _ = build(self.root, self.object_id, self.raw, self.annotations, self.track, out)
        self.assertTrue(out.is_dir())
        self.assertTrue(all(Path(r["appearance_mask_path"]).is_file() for r in manifest["frames"]))

    def test_runner_and_prepare_have_no_case_specific_literal(self):
        text = "\n".join((SCRIPTS / name).read_text() for name in (
            "prepare_v19_additive_demo.py", "run_v19_additive_demo_branch.py",
            "build_v19_demo_visual_inputs.py", "build_v19_demo_pose_binding.py",
            "fit_v20_visual_contact_demo.py", "v19_demo_timeline.py",
            "run_v20_visual_contact_demo.py", "render_v20_visual_contact_demo.py",
            "prepare_v19_sam3d_runtime.py", "run_v19_sam3d_integration_stage.py"))
        self.assertNotIn("milk_carton", text)
        self.assertNotIn("range(150)", text)
        for pattern in (r"(?i)milk[_ -]?carton|carton[_ -]?milk", r"frame_(?:idx|id)\s*==\s*(?:88|89|92)\b",
                        r"range\(\s*150\s*\)", r"len\([^\n]*?\)\s*==\s*150\b",
                        r"(?:frame_count|num_frames|N)\s*=\s*150\b"):
            self.assertIsNone(re.search(pattern, text), pattern)
        for phase in ("D00_appearance_anchor", "D01_visual_binding", "D02_visual_fit", "D03_visual_render"):
            self.assertIn(phase, text)
        self.assertIn("renders/demo_visual", text)
        self.assertNotIn("logs/integration/events.jsonl", text)


if __name__ == "__main__":
    unittest.main()
