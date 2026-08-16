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
import trimesh

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))

import build_p11_dual_geometry_inputs as p11  # noqa: E402
import build_p13_sam3d_native_metric_bridge as sam3d_bridge  # noqa: E402
import build_v18_compact_rigid_evidence_bundle as evidence_bundle  # noqa: E402
import remote_run_sam3d_objects_mesh_v7 as sam3d_runner  # noqa: E402
import remote_run_trellis_shape_v3 as trellis_runner  # noqa: E402
import build_v18_compact_rigid_trellis_completion as completion  # noqa: E402
import build_v19_visible_geometry_from_sam2_depth as visible_geometry  # noqa: E402
import fit_v18_compact_rigid_object_pose as observed_pose  # noqa: E402
import solve_v19_rigid_object_pose_graph as pose_graph  # noqa: E402
import render_p14_p15_layered_state as p15_render  # noqa: E402
import run_p12_parallel_geometry_priors as p12  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False).export(str(path))


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
        sam_native_p3d = generated / "sam_native_p3d.ply"
        sam_native_cv = generated / "sam_native_cv.ply"
        write_tiny_ply(sam_mesh)
        write_mesh(
            sam_native_p3d,
            np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )
        write_mesh(
            sam_native_cv,
            np.asarray([[0, 0, 1], [-1, 0, 1], [0, -1, 1]], dtype=float),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )
        sam_glb.write_bytes(b"synthetic-glb")
        write_tiny_ply(sam_gs)
        sam_report = generated / "sam_report.json"
        write_json(
            sam_report,
            {
                "status": "ok",
                "moge_offline": {
                    "checkpoint_sha256": "da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f",
                    "network_resolution_allowed": False,
                },
                "dinov2_offline": {
                    "checkpoint_sha256": "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
                    "hubconf_sha256": "c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64",
                    "network_resolution_allowed": False,
                },
                "huggingface_offline": {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "network_resolution_allowed": False,
                },
                "name": case_name,
                "image": sam_contract["image"],
                "mask": sam_contract["mask"],
                "mesh": str(sam_mesh),
                "glb": str(sam_glb),
                "gaussian": str(sam_gs),
                "native_pose_mesh_pytorch3d_camera": str(sam_native_p3d),
                "native_pose_mesh_opencv_camera": str(sam_native_cv),
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
                "native_pose_contract": {
                    "quaternion_order": "wxyz_scalar_first_pytorch3d",
                    "row_vector_formula": "p_p3d_camera = (p_raw_local * scale_xyz) @ quaternion_to_matrix(q_wxyz) + translation_xyz",
                    "metric_status": "native_monocular_scene_units_not_sensor_meters",
                },
                "moge_offline": {
                    "checkpoint_sha256": "da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f",
                    "network_resolution_allowed": False,
                },
                "dinov2_offline": {
                    "checkpoint_sha256": "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
                    "hubconf_sha256": "c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64",
                    "network_resolution_allowed": False,
                },
                "huggingface_offline": {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "network_resolution_allowed": False,
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
                sam3d_moge_checkpoint=None,
                sam3d_dinov2_repo=None,
                sam3d_dinov2_checkpoint=None,
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

    def test_sam3d_offline_moge_binding_rejects_wrong_hash(self) -> None:
        checkpoint = self.root / "wrong_moge_model.pt"
        checkpoint.write_bytes(b"not-the-frozen-moge-checkpoint")
        with self.assertRaisesRegex(RuntimeError, "offline MoGe checkpoint hash mismatch"):
            sam3d_runner.install_offline_moge_checkpoint(checkpoint)

    def test_sam3d_offline_dinov2_binding_requires_explicit_checkpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing offline DINOv2 checkpoint"):
            sam3d_runner.install_offline_dinov2_hub(
                Path("/mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub/hub/facebookresearch_dinov2_main"),
                self.root / "missing_dinov2_checkpoint.pth",
            )

    def test_trellis_offline_dinov2_binding_requires_explicit_checkpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing offline DINOv2 checkpoint"):
            trellis_runner.install_offline_dinov2_hub(
                Path("/mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub/hub/facebookresearch_dinov2_main"),
                self.root / "missing_trellis_dinov2_checkpoint.pth",
            )

    def test_atomic_anchor_binding_rejects_mixed_frame_surface(self) -> None:
        selected = {
            "frame_idx": 55,
            "visible_geometry_candidate": {
                "frame_idx": 55,
                "centroid_world_m": [0.1, 0.2, 0.3],
            },
        }
        valid_mesh = {
            "anchor_frame_idx": 55,
            "anchor_centroid_world_m": [0.1, 0.2, 0.3],
        }
        binding = evidence_bundle.validate_selected_anchor_binding(
            selected=selected, mesh_reconstruction=valid_mesh
        )
        self.assertTrue(binding["validated"])
        with self.assertRaisesRegex(RuntimeError, "non-atomic selected P11 anchor binding"):
            evidence_bundle.validate_selected_anchor_binding(
                selected=selected,
                mesh_reconstruction={
                    "anchor_frame_idx": 83,
                    "anchor_centroid_world_m": [0.1, 0.2, 0.3],
                },
            )
        with self.assertRaisesRegex(RuntimeError, "non-atomic selected P11 anchor binding"):
            evidence_bundle.validate_selected_anchor_binding(
                selected=selected,
                mesh_reconstruction={
                    "anchor_frame_idx": 55,
                    "anchor_centroid_world_m": [0.2, 0.2, 0.3],
                },
            )

    def test_sam3d_observed_front_quality_has_strict_conditional_and_fail_modes(self) -> None:
        strict = sam3d_bridge.observed_front_quality_decision(
            median_fraction=0.03,
            p95_fraction=0.12,
            native_projection_iou=0.10,
            maximum_median_fraction=0.05,
            strict_maximum_p95_fraction=0.15,
            conditional_maximum_p95_fraction=0.18,
            conditional_minimum_projection_iou=0.25,
        )
        self.assertTrue(strict["quality_passed"])
        self.assertTrue(strict["strict_quality_passed"])
        self.assertFalse(strict["conditional_tail_uncertainty"])

        conditional = sam3d_bridge.observed_front_quality_decision(
            median_fraction=0.0469,
            p95_fraction=0.1692,
            native_projection_iou=0.3304,
            maximum_median_fraction=0.05,
            strict_maximum_p95_fraction=0.15,
            conditional_maximum_p95_fraction=0.18,
            conditional_minimum_projection_iou=0.25,
        )
        self.assertTrue(conditional["quality_passed"])
        self.assertFalse(conditional["strict_quality_passed"])
        self.assertTrue(conditional["conditional_tail_uncertainty"])
        self.assertEqual(
            conditional["acceptance_mode"],
            "conditional_p95_tail_uncertain_native_projection_supported",
        )

        wrong_pose = sam3d_bridge.observed_front_quality_decision(
            median_fraction=0.04,
            p95_fraction=0.17,
            native_projection_iou=0.013,
            maximum_median_fraction=0.05,
            strict_maximum_p95_fraction=0.15,
            conditional_maximum_p95_fraction=0.18,
            conditional_minimum_projection_iou=0.25,
        )
        self.assertFalse(wrong_pose["quality_passed"])
        self.assertFalse(wrong_pose["conditional_projection_passed"])

        wrong_surface = sam3d_bridge.observed_front_quality_decision(
            median_fraction=0.165,
            p95_fraction=0.17,
            native_projection_iou=0.64,
            maximum_median_fraction=0.05,
            strict_maximum_p95_fraction=0.15,
            conditional_maximum_p95_fraction=0.18,
            conditional_minimum_projection_iou=0.25,
        )
        self.assertFalse(wrong_surface["quality_passed"])
        self.assertFalse(wrong_surface["median_passed"])

    def test_anchor_conditioning_coherence_prefers_supported_single_component(self) -> None:
        preferred = visible_geometry.anchor_conditioning_coherence(
            {
                "component_count": 1,
                "mask_area_px": 8789,
                "visible_depth_vertex_count": 1134,
                "touches_or_near_image_border": False,
            },
            maximum_mask_area_px=30000,
            maximum_visible_depth_points=2500,
        )
        self.assertTrue(preferred["preferred"])
        self.assertEqual(preferred["minimum_mask_area_px"], 6000)
        self.assertEqual(preferred["minimum_visible_depth_points"], 500)

        fragmented = visible_geometry.anchor_conditioning_coherence(
            {
                "component_count": 2,
                "mask_area_px": 29500,
                "visible_depth_vertex_count": 2500,
                "touches_or_near_image_border": False,
            },
            maximum_mask_area_px=30000,
            maximum_visible_depth_points=2500,
        )
        self.assertFalse(fragmented["preferred"])
        self.assertTrue(fragmented["support_sufficient"])
        self.assertFalse(fragmented["one_owned_component"])

        too_small = visible_geometry.anchor_conditioning_coherence(
            {
                "component_count": 1,
                "mask_area_px": 900,
                "visible_depth_vertex_count": 90,
                "touches_or_near_image_border": False,
            },
            maximum_mask_area_px=30000,
            maximum_visible_depth_points=2500,
        )
        self.assertFalse(too_small["preferred"])
        self.assertFalse(too_small["support_sufficient"])

    def test_sam3d_native_metric_bridge_preserves_pose_and_uses_identity_p13_alignment(self) -> None:
        generated = self.root / "native_bridge_source"
        generated.mkdir()
        raw_mesh = trimesh.creation.box(extents=[1.0, 0.6, 0.4])
        raw_path = generated / "sam3d_raw.ply"
        raw_mesh.export(str(raw_path))
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        translation = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        native_scale = np.asarray([0.2, 0.2, 0.2], dtype=np.float64)
        native_p3d, native_cv, pose_contract = sam3d_bridge.apply_native_pose(
            np.asarray(raw_mesh.vertices), q, translation, native_scale
        )
        native_p3d_path = generated / "native_p3d.ply"
        native_cv_path = generated / "native_cv.ply"
        write_mesh(native_p3d_path, native_p3d, np.asarray(raw_mesh.faces))
        write_mesh(native_cv_path, native_cv, np.asarray(raw_mesh.faces))

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[42:59, 38:63] = 255
        mask_path = generated / "owned_mask.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        observed = []
        for v in range(44, 57):
            for u in range(40, 61):
                z = 0.45
                observed.append([(u - 50.0) * z / 100.0, (v - 50.0) * z / 100.0, z])
        observed = np.asarray(observed, dtype=np.float64)
        centroid = observed.mean(axis=0)
        evidence_path = generated / "evidence.json"
        write_json(
            evidence_path,
            {
                "status": "ok",
                "case": "synthetic_bridge",
                "object_id": "box",
                "selected_frame_idx": 0,
                "selected_anchor_atomic_binding": {
                    "selected_frame_idx": 0,
                    "visible_geometry_frame_idx": 0,
                    "canonical_surface_frame_idx": 0,
                    "selected_vs_canonical_centroid_error_m": 0.0,
                    "validated": True,
                },
                "selected": {
                    "frame_idx": 0,
                    "mask_path": str(mask_path),
                    "camera": {"T_world_camera_metric": np.eye(4).tolist()},
                    "visible_geometry_candidate": {
                        "frame_idx": 0,
                        "mask_path": str(mask_path),
                        "intrinsics_fx_fy_cx_cy": [100.0, 100.0, 50.0, 50.0],
                        "camera_vertices_sample_m": observed.tolist(),
                        "centroid_world_m": centroid.tolist(),
                        "first_surface_depth_ownership": {
                            "enabled": True,
                            "fail_closed": False,
                            "failure_reasons": [],
                            "retained_fraction": 1.0,
                        },
                        "mask_depth_transform_contract": {
                            "camera_contract_consistent": True,
                            "mask_size_wh": [100, 100],
                            "depth_size_wh": [100, 100],
                            "A_depth_from_mask_coordinate_model": np.eye(3).tolist(),
                        },
                    },
                },
                "depth_fused_object_row": {
                    "mesh_reconstruction": {
                        "anchor_frame_idx": 0,
                        "anchor_centroid_world_m": centroid.tolist(),
                        "atomic_anchor_binding": True,
                    }
                },
            },
        )
        p12_path = generated / "p12.json"
        write_json(
            p12_path,
            {
                "status": "ok",
                "candidates": {
                    "sam3d_objects": {
                        "source_model": "sam3d_objects",
                        "conditioning": {"mask": {"path": str(mask_path)}},
                        "native_outputs": {
                            "raw_mesh": {"path": str(raw_path)},
                            "native_pose": {
                                "rotation": q.tolist(),
                                "translation": translation.tolist(),
                                "scale": native_scale.tolist(),
                            },
                            "native_pose_contract": {
                                **pose_contract,
                                "quaternion_order": "wxyz_scalar_first_pytorch3d",
                            },
                            "native_pose_mesh_opencv_camera": {"path": str(native_cv_path)},
                        },
                    }
                },
            },
        )
        output = generated / "bridge_output"
        report = sam3d_bridge.run(
            argparse.Namespace(
                evidence_report=evidence_path,
                p12_report=p12_path,
                output_dir=output,
                min_robust_depth_support_fraction=0.70,
                min_positive_depth_fraction=0.99,
                min_native_convex_projection_iou=0.05,
                min_native_owned_zbuffer_pixels=4,
                min_observed_points=100,
                min_scene_similarity_scale=0.05,
                max_scene_similarity_scale=5.0,
                min_complete_to_observed_extent_ratio=0.25,
                max_complete_to_observed_extent_ratio=5.0,
                geometry_quality_surface_samples=40000,
                max_observed_to_generated_median_extent_fraction=0.15,
                strict_observed_to_generated_p95_extent_fraction=0.15,
                max_observed_to_generated_p95_extent_fraction=0.25,
                min_conditional_tail_native_projection_iou=0.25,
            )
        )
        metric_scale = report["sensor_metric_scene_similarity"]["camera_origin_scene_similarity_scale"]
        self.assertAlmostEqual(metric_scale, 0.45 / 0.96, places=5)
        self.assertTrue(report["sensor_metric_scene_similarity"]["translation_and_object_scale_scaled_together"])
        self.assertTrue(
            report["geometry_validation"]["observed_front_surface_quality"]["quality_passed"]
        )
        self.assertFalse(
            report["geometry_validation"]["observed_front_surface_quality"]
            ["generated_faces_pose_evidence_consumed"]
        )
        self.assertGreater(report["projection_validation"]["convex_projection_iou"], 0.5)
        output_mesh = trimesh.load(report["outputs"]["metric_canonical_render_prior"], force="mesh", process=False)
        expected_vertices = native_cv * metric_scale - centroid[None, :]
        np.testing.assert_allclose(np.asarray(output_mesh.vertices), expected_vertices, atol=1.0e-6)

        compatibility = json.loads(
            (output / "p13_metric_canonical_input.json").read_text(encoding="utf-8")
        )
        identity = completion.identity_metric_canonical_alignment(
            output_mesh,
            observed,
            compatibility,
            Path(report["outputs"]["metric_canonical_render_prior"]),
            max_samples=1000,
        )
        np.testing.assert_array_equal(identity["matrix_model_to_canonical"], np.eye(4))
        self.assertFalse(identity["pca_axis_permutation_used"])
        self.assertFalse(identity["icp_rotation_or_scale_used"])

    def test_extent_consistency_is_anchor_rotation_invariant(self) -> None:
        physical_extent = np.asarray([0.40, 0.25, 0.12], dtype=float)
        rotated_anchor_extent = np.asarray([0.12, 0.40, 0.25], dtype=float)
        population_reference = np.sort(physical_extent)[::-1]
        sorted_anchor = np.sort(rotated_anchor_extent)[::-1]
        ratio = np.maximum(
            sorted_anchor / population_reference,
            population_reference / sorted_anchor,
        )
        np.testing.assert_allclose(ratio, np.ones(3), atol=1.0e-12)
        # Camera-axis division would falsely flag this same rigid extent.
        self.assertGreater(float(np.max(physical_extent / rotated_anchor_extent)), 3.0)

    def test_hand_ownership_uses_projected_mano_silhouette_not_bbox(self) -> None:
        geometry = self.root / "hand_geometry"
        geometry.mkdir()
        vertices = np.zeros((1, 778, 3), dtype=np.float32)
        vertices[..., 2] = 1.0
        vertices[0, 0, :2] = [-0.2, -0.2]
        vertices[0, 1, :2] = [0.2, -0.2]
        vertices[0, 2, :2] = [0.0, 0.2]
        bridge = geometry / "bridge.npz"
        source = geometry / "hawor.npz"
        np.savez_compressed(bridge, vertices_camera=vertices)
        np.savez_compressed(
            source,
            left_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            right_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        hand = {
            "hand_side": "left",
            "same_frame_detection": True,
            "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
            "metric_mano_state": {
                "vertices_reference": {
                    "bridge_npz": str(bridge),
                    "bridge_vertices_camera_array": "vertices_camera",
                    "bridge_row_index": 0,
                    "source_hawor_npz": str(source),
                },
                "current_v18_camera_intrinsics_fx_fy_cx_cy": [100.0, 100.0, 50.0, 50.0],
            },
        }
        mask = np.ones((100, 100), dtype=bool)
        owned, diagnostic = visible_geometry.subtract_hand_owned_bbox_regions(
            mask,
            {"hands": [hand]},
            source_width=100,
            source_height=100,
            pad_px=0,
            enabled=True,
        )
        removed = int(np.count_nonzero(mask & ~owned))
        self.assertFalse(diagnostic["fail_closed"])
        self.assertFalse(diagnostic["bbox_subtraction_used"])
        self.assertGreater(removed, 500)
        self.assertLess(removed, 2000)
        self.assertGreater(int(np.count_nonzero(owned)), 8000)

    def test_observed_temporal_pose_pairwise_registration_recovers_small_rigid_motion(self) -> None:
        rng = np.random.default_rng(1801)
        source = rng.normal(size=(1200, 3)) * np.asarray([0.08, 0.04, 0.018])
        expected_rotation = observed_pose.Rotation.from_euler(
            "zyx", [4.0, -2.0, 1.0], degrees=True
        ).as_matrix()
        expected_translation = np.asarray([0.006, -0.004, 0.003])
        target = observed_pose.apply_pose(source, expected_rotation, expected_translation)
        target += rng.normal(scale=0.0002, size=target.shape)
        rotation, translation, report = observed_pose.register_adjacent_observed_surfaces(
            source,
            target,
            frame_gap=1,
            iterations=12,
            trim_fraction=0.65,
            max_correspondence_m=0.035,
            min_matches=30,
            max_median_residual_m=0.015,
        )
        rotation_error = observed_pose.rotation_angle_deg(rotation @ expected_rotation.T)
        self.assertTrue(report["quality_passed"])
        self.assertLess(rotation_error, 0.25)
        self.assertLess(float(np.linalg.norm(translation - expected_translation)), 0.001)
        self.assertLess(report["residual_m"]["median"], 0.001)

    def test_observed_temporal_rotation_correction_has_chain_prior_not_gate_clipping(self) -> None:
        measurements = np.zeros((30, 3), dtype=float)
        measurements[15, 2] = 1.0
        regularized = observed_pose.regularize_translation_timeline(
            measurements,
            np.ones(30, dtype=float),
            velocity_weight=6.0,
            acceleration_weight=4.0,
            zero_prior_weight=0.75,
        )
        self.assertGreater(float(np.max(np.abs(regularized[:, 2]))), 0.01)
        self.assertLess(float(np.max(np.abs(regularized[:, 2]))), 0.25)
        self.assertGreater(int(np.count_nonzero(np.abs(regularized[:, 2]) > 1.0e-5)), 1)
        self.assertEqual(observed_pose.bridge_positions_for_gap(56, 72, 10), [64])

    def test_pose_graph_readiness_rejects_rotation_jump_even_when_optimizer_could_succeed(self) -> None:
        rng = np.random.default_rng(9)
        shape = rng.normal(size=(300, 3)) * np.asarray([0.08, 0.04, 0.015])
        observations = []
        pose_rows = []
        for frame_idx in range(20):
            rotation = np.eye(3)
            if frame_idx >= 10:
                rotation = pose_graph.Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
            translation = np.asarray([0.001 * frame_idx, 0.0, 0.45])
            observations.append(
                pose_graph.PoseObservation(
                    frame_idx=frame_idx,
                    source_row={},
                    rotation_world_from_canonical=rotation,
                    translation_world_m=translation,
                    translation_sigma_m=0.01,
                    rotation_sigma_rad=0.1,
                    visible_sample_count=len(shape),
                    observed_points_world=shape + translation,
                    nonpenetration_target_world_m=None,
                    nonpenetration_weight=0.0,
                    nonpenetration_source_rows=0,
                )
            )
            pose_rows.append({
                "frame_idx": frame_idx,
                "status": pose_graph.CORRECTED_POSE_STATUS,
                "rotation_world_from_completed_canonical_matrix": rotation.tolist(),
                "translation_world_m": translation.tolist(),
            })
        diagnostics = pose_graph.temporal_readiness_diagnostics(
            pose_rows=pose_rows,
            annotations={"frames": [{"frame_idx": idx} for idx in range(20)]},
            observations=observations,
            full_timeline_completion={"completed_row_count": 0, "mode_counts": {}},
            args=argparse.Namespace(
                frame_start=None,
                frame_end=None,
                min_rotation_observability_score=0.02,
                min_direct_pose_fraction=0.80,
                max_direct_pose_gap_frames=10,
                max_rigid_pose_extrapolation_gap_frames=10,
                max_completed_pose_fraction=0.20,
                max_nearest_hold_fraction=0.05,
                max_rotation_step_deg=15.0,
                max_translation_step_m=0.05,
                min_rotation_observable_fraction=0.80,
            ),
        )
        self.assertFalse(diagnostics["ready"])
        self.assertIn("rotation_step_jump", diagnostics["failure_reasons"])
        self.assertAlmostEqual(diagnostics["max_rotation_step_deg"], 30.0, places=6)
        self.assertGreater(diagnostics["rotation_observability"]["observable_fraction"], 0.8)

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
