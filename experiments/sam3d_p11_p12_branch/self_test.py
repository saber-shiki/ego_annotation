#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_p11_dual_geometry_inputs as p11  # noqa: E402
import render_p14_p15_layered_state as p15_render  # noqa: E402
import run_p12_parallel_geometry_priors as p12  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_tiny_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class P11P12BranchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="sam3d_p11_p12_test_")
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()

        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(32, dtype=np.uint8)[None, :] * 5
        rgb[..., 1] = np.arange(32, dtype=np.uint8)[:, None] * 4
        rgb[..., 2] = 80
        self.rgb = source / "frame.jpg"
        Image.fromarray(rgb, mode="RGB").save(self.rgb)

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[9:24, 6:27] = 255
        self.owned_mask = source / "object_owned_mask.png"
        Image.fromarray(mask, mode="L").save(self.owned_mask)

        raw_mask = mask.copy()
        raw_mask[5:9, 12:20] = 255
        self.raw_mask = source / "raw_sam2_mask.png"
        Image.fromarray(raw_mask, mode="L").save(self.raw_mask)

        rgba = np.dstack([rgb[6:27, 6:27], mask[6:27, 6:27]])
        self.trellis_crop = source / "trellis_crop.png"
        Image.fromarray(rgba, mode="RGBA").save(self.trellis_crop)

        self.evidence_report = source / "evidence_bundle_report.json"
        write_json(
            self.evidence_report,
            {
                "method": "synthetic_test_evidence",
                "status": "ok",
                "case": "synthetic_case",
                "object_id": "synthetic_object",
                "selected_frame_idx": 7,
                "selection_rule": "fixed_test_anchor",
                "selected": {
                    "frame_idx": 7,
                    "raw_frame_path": str(self.rgb),
                    "mask_path": str(self.owned_mask),
                    "visible_geometry_candidate": {"mask_path": str(self.owned_mask)},
                    "trellis_conditioning_crop": {"crop_rgba": str(self.trellis_crop)},
                },
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def build_p11(self) -> tuple[Path, dict]:
        output = self.root / "p11"
        report = p11.run(
            argparse.Namespace(
                evidence_report=self.evidence_report,
                output_dir=output,
                comparison_mask=self.raw_mask,
                allow_unverified_mask_provenance=False,
            )
        )
        return output / "p11_dual_geometry_inputs_report.json", report

    def test_p11_dual_contract_uses_owned_mask_and_preserves_bytes(self) -> None:
        _, report = self.build_p11()
        sam = report["conditioning_contracts"]["sam3d_objects_native"]
        trellis = report["conditioning_contracts"]["trellis_native"]
        self.assertTrue(report["mask_provenance"]["byte_identical"])
        self.assertEqual(sam["external_pointmap"], None)
        self.assertEqual(sam["pre_model_crop"], None)
        self.assertTrue(sam["image_source_copy"]["byte_identical"])
        self.assertTrue(sam["mask_source_copy"]["byte_identical"])
        self.assertTrue(trellis["source_copy"]["byte_identical"])
        self.assertGreater(report["comparison_mask_diagnostic"]["xor_pixels"], 0)
        self.assertEqual(report["selected_input_summary"]["mask_area_px"], 15 * 21)
        self.assertTrue(Path(report["review"]).is_file())

    def test_p11_rejects_mask_provenance_mismatch(self) -> None:
        payload = json.loads(self.evidence_report.read_text(encoding="utf-8"))
        payload["selected"]["visible_geometry_candidate"]["mask_path"] = str(self.raw_mask)
        mismatch_report = self.root / "source" / "mismatch_evidence.json"
        write_json(mismatch_report, payload)
        with self.assertRaisesRegex(RuntimeError, "not byte-identical"):
            p11.run(
                argparse.Namespace(
                    evidence_report=mismatch_report,
                    output_dir=self.root / "p11_mismatch",
                    comparison_mask=None,
                    allow_unverified_mask_provenance=False,
                )
            )

    def test_p12_assembles_native_sam_and_frozen_trellis_without_model_run(self) -> None:
        p11_report_path, p11_report = self.build_p11()
        sam_contract = p11_report["conditioning_contracts"]["sam3d_objects_native"]
        trellis_contract = p11_report["conditioning_contracts"]["trellis_native"]
        case_name = "synthetic_case_synthetic_object_frame_000007"

        generated = self.root / "generated"
        sam_mesh = generated / "sam_mesh.ply"
        sam_glb = generated / "sam_mesh.glb"
        sam_gs = generated / "sam_gs.ply"
        write_tiny_ply(sam_mesh)
        sam_glb.write_bytes(b"synthetic-glb")
        write_tiny_ply(sam_gs)
        sam_report = generated / "sam_report.json"
        write_json(
            sam_report,
            {
                "status": "ok",
                "name": case_name,
                "image": sam_contract["image"],
                "mask": sam_contract["mask"],
                "mesh": str(sam_mesh),
                "glb": str(sam_glb),
                "gaussian": str(sam_gs),
                "mesh_stats": {
                    "vertices": 3,
                    "faces": 1,
                    "watertight": False,
                    "winding_consistent": True,
                },
                "pose": {
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
            },
        )

        trellis_mesh = generated / "trellis_mesh.ply"
        trellis_gs = generated / "trellis_gs.ply"
        write_tiny_ply(trellis_mesh)
        write_tiny_ply(trellis_gs)
        trellis_report = generated / "trellis_report.json"
        write_json(
            trellis_report,
            {
                "status": "ok",
                "image": trellis_contract["image"],
                "mesh": str(trellis_mesh),
                "gaussian": str(trellis_gs),
                "glb": None,
                "vertices": 3,
                "faces": 1,
                "extent_model_units": [1.0, 1.0, 0.0],
                "center_model_units": [1.0 / 3.0, 1.0 / 3.0, 0.0],
            },
        )

        report = p12.run(
            argparse.Namespace(
                p11_report=p11_report_path,
                output_dir=self.root / "p12",
                case_name=None,
                seed=42,
                trellis_report=trellis_report,
                reuse_sam3d_report=sam_report,
                sam3d_python=None,
                sam3d_runner=None,
                sam3d_repo=None,
                sam3d_config=None,
                cuda_visible_device=None,
                min_free_mib=30000,
                compile=False,
            )
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(set(report["candidates"]), {"trellis", "sam3d_objects"})
        self.assertEqual(
            report["candidates"]["sam3d_objects"]["conditioning"]["external_pointmap"],
            None,
        )
        self.assertTrue(
            report["candidates"]["sam3d_objects"]["input_binding"]["mask"]["byte_identical"]
        )
        self.assertEqual(
            report["candidates"]["trellis"]["status"],
            "frozen_existing_raw_prior_reference",
        )
        self.assertFalse(report["backend_neutral_contract"]["collision_ready"])
        self.assertFalse(report["canonical_trellis_rerun"])

    def test_p15_observed_object_ownership_preserves_mano_depth_winner(self) -> None:
        background = np.zeros((80, 100, 3), dtype=np.uint8)
        vertices = np.zeros((3, 3), dtype=np.float64)
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)
        object_uv = np.asarray([[10, 10], [90, 10], [50, 70]], dtype=np.float64)
        hand_uv = np.asarray([[38, 28], [62, 28], [50, 52]], dtype=np.float64)
        layers = [
            p15_render.SceneLayer(
                "generated",
                "generated_complete_prior_underlay",
                vertices,
                faces,
                (205, 75, 190),
                0.46,
                0,
                0.0,
            ),
            p15_render.SceneLayer(
                "observed",
                "observed_metric_surface_overlay",
                vertices,
                faces,
                (65, 205, 75),
                0.72,
                0,
                0.0,
            ),
            p15_render.SceneLayer(
                "mano",
                "mano_right_full_surface",
                vertices,
                faces,
                (45, 145, 255),
                0.62,
                0,
                0.0,
            ),
        ]
        _image, stats, labels = p15_render.rasterize_scene(
            background,
            layers,
            [
                (object_uv, np.full(3, 0.50)),
                (object_uv, np.full(3, 0.80)),
                (hand_uv, np.full(3, 0.30)),
            ],
        )
        # Observed support owns object pixels even when the generated triangle is
        # closer, while a MANO triangle that wins depth ordering stays visible.
        self.assertEqual(int(labels[20, 50]), 2)
        self.assertEqual(int(labels[38, 50]), 3)
        self.assertEqual(stats["layers"][0]["visible_pixels"], 0)
        self.assertGreater(stats["layers"][1]["visible_pixels"], 0)
        self.assertGreater(stats["layers"][2]["visible_pixels"], 0)
        self.assertIn("mano_depth_winners_preserved", stats["depth_order_method"])

    def test_p15_contiguous_face_selection_is_exact(self) -> None:
        faces = np.arange(30, dtype=np.int32).reshape(10, 3)
        selected = p15_render.select_faces(
            faces,
            {"mode": "contiguous_range", "start": 2, "stop": 7},
            self.root / "synthetic.ply",
        )
        np.testing.assert_array_equal(selected, faces[2:7])
        with self.assertRaisesRegex(RuntimeError, "invalid face range"):
            p15_render.select_faces(
                faces,
                {"mode": "contiguous_range", "start": 7, "stop": 12},
                self.root / "synthetic.ply",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
