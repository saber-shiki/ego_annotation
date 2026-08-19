#!/usr/bin/env python3
"""CPU contract tests for the shared HOT3D P17/P18/P18b reintegration."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "scripts"
EXPERIMENT = REPO / "experiments/sam3d_p11_p12_branch"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EXPERIMENT))

import build_v19_base_annotations as base_annotations  # noqa: E402
import build_v19_visible_contact_ownership_factor as p17  # noqa: E402
import export_hawor_world as hawor_export  # noqa: E402
import render_p14_p15_layered_state as layered_renderer  # noqa: E402
import run_hot3d_shared_p17_p18_tail as shared_tail  # noqa: E402
import solve_v18_joint_mano_interval_trajectory as p18  # noqa: E402
from v19_camera_contract import resize_affine  # noqa: E402


def aligned_hand(side: str, K: list[float], *, full_k: bool = True) -> dict:
    alignment = {
        "active_contract_reinference_required": False,
        "source_hawor_state_intrinsics_match_active_contract": True,
        "source_hawor_full_K_bound_to_inference": full_k,
        "hawor_camera_image_plane_contract": (
            {
                "status": "validated_source_K_affine_centered_hawor_inference_plane",
                "validated": True,
                "contract_sha256": "synthetic-plane-contract-sha256",
                "inference_frames_aggregate_sha256": "synthetic-frame-aggregate-sha256",
            }
            if full_k
            else {
                "status": "legacy_or_unproven_hawor_camera_image_plane",
                "validated": False,
            }
        ),
    }
    return {
        "hand_side": side,
        "camera_contract_alignment": alignment,
        "metric_mano_state": {
            "current_v18_camera_intrinsics_fx_fy_cx_cy": K,
            "camera_contract_alignment": alignment,
        },
    }


def annotation_frame(K: list[float], *, full_k: bool = True) -> dict:
    return {
        "frame_idx": 0,
        "camera": {"intrinsics_fx_fy_cx_cy": K},
        "hands": [
            aligned_hand("left", K, full_k=full_k),
            aligned_hand("right", K, full_k=full_k),
        ],
    }


class SharedP18ReintegrationTest(unittest.TestCase):
    def test_source_to_mask_affine_is_exact_and_shared_by_p17_p18(self) -> None:
        A_mask_from_source = resize_affine(
            (1408, 1408),
            (960, 960),
            pixel_center_convention="integer_pixel_centers_opencv",
        )
        A_source_from_mask = np.linalg.inv(A_mask_from_source)
        mask_xy = np.asarray([200.0, 100.0, 1.0])
        source_xy = A_source_from_mask @ mask_xy
        projected = p17.transform_source_pixels(
            source_xy[None, :2], A_mask_from_source
        )
        np.testing.assert_allclose(projected[0], mask_xy[:2], atol=1.0e-9)

        mask = np.zeros((960, 960), dtype=bool)
        mask[100, 200] = True
        inside = p18.mask_membership(
            mask,
            source_xy[None, :2],
            A_mask_from_source=A_mask_from_source,
        )
        self.assertEqual(inside.tolist(), [True])
        # The historical bug interpreted source-grid UV as mask-grid UV.
        legacy_inside = p18.mask_membership(mask, source_xy[None, :2])
        self.assertEqual(legacy_inside.tolist(), [False])

    def test_shared_tail_rejects_numeric_K_without_full_K_inference_proof(self) -> None:
        K = [973.0, 973.0, 704.0, 704.0]
        annotations = {"frames": [annotation_frame(K, full_k=False)]}
        with self.assertRaisesRegex(RuntimeError, "source-K to centered-inference-plane binding"):
            shared_tail.validate_camera_mano_contract(
                annotations, (0, 0), {"left", "right"}
            )

        report = shared_tail.validate_camera_mano_contract(
            {"frames": [annotation_frame(K, full_k=True)]},
            (0, 0),
            {"left", "right"},
        )
        self.assertEqual(report["row_count"], 2)
        self.assertEqual(
            report["status"], "exact_source_K_affine_bound_to_centered_hawor_plane_for_all_mano_rows"
        )

    def test_p18_direct_preflight_rejects_old_center_K_before_models(self) -> None:
        K = [973.0, 973.0, 704.0, 704.0]
        with tempfile.TemporaryDirectory(prefix="p18_camera_preflight_") as temp:
            root = Path(temp)
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps({"frames": [annotation_frame(K, full_k=False)]}),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                annotations=annotations,
                start_frame=0,
                end_frame=0,
                sides=["left", "right"],
                require_active_full_K_mano_contract=True,
            )
            with self.assertRaisesRegex(RuntimeError, "before model loading"):
                p18.validate_interval_camera_mano_contract(args)

    def test_p18_validation_rejects_private_object_translation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p18_object_delta_") as temp:
            root = Path(temp)
            pose = root / "pose.json"
            surface = root / "surface.ply"
            pose.write_text("{}", encoding="utf-8")
            surface.write_text("ply\n", encoding="utf-8")
            base = {
                "parameters": {
                    "optimize_object_translation": False,
                    "freeze_translation_without_visible_surface_support": True,
                    "gate_translation_with_visible_surface_support": True,
                    "translation_gate_min_visible_surface_depth_vertices": 0,
                },
                "inputs": {
                    "pose_report": str(pose),
                    "physical_surface_mesh": str(surface),
                },
                "per_frame_states": [
                    {
                        "frame_idx": 0,
                        "hand_side": "left",
                        "optimized_object_translation_world_m": [0.0, 0.0, 0.0],
                        "visible_surface_translation_support_vertex_count": 1,
                        "optimizer_translation_support_gate": {"frozen": False},
                        "output_translation_gate": {"applied": False},
                        "signed_object_surface_factor_state": "inactive_signed_geometry_not_ready_or_nonwatertight",
                    }
                ],
            }
            accepted = shared_tail.validate_p18(
                base,
                expected_rows=1,
                pose_report=pose,
                physical_surface=surface,
            )
            self.assertEqual(accepted["max_object_translation_delta_m"], 0.0)
            contaminated = json.loads(json.dumps(base))
            contaminated["per_frame_states"][0][
                "optimized_object_translation_world_m"
            ] = [0.001, 0.0, 0.0]
            with self.assertRaisesRegex(RuntimeError, "privately moved"):
                shared_tail.validate_p18(
                    contaminated,
                    expected_rows=1,
                    pose_report=pose,
                    physical_surface=surface,
                )

    def test_independent_first_hit_support_survives_hand_ownership_cut_without_becoming_residual(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p17_independent_depth_support_") as temporary:
            root = Path(temporary)
            constraint = np.zeros((16, 16), dtype=np.uint8)
            # Projected MANO will query (8,8), deliberately absent from the
            # ownership constraint mask.
            constraint[2:6, 2:6] = 1
            support = np.zeros((16, 16), dtype=np.uint8)
            support[8, 8] = 1
            depth = np.full((16, 16), np.nan, dtype=np.float32)
            depth[8, 8] = 1.0
            non_object = root / "non_object.png"
            constraint_path = root / "constraint.png"
            support_path = root / "support.png"
            from PIL import Image
            Image.fromarray(constraint * 255).save(non_object)
            Image.fromarray(constraint * 255).save(constraint_path)
            Image.fromarray(support * 255).save(support_path)
            npz_path = root / "first_surface.npz"
            np.savez_compressed(
                npz_path,
                depth_mask_plane_m=depth,
                support_mask=support,
                A_mask_from_source_coordinate_model=np.eye(3),
            )
            row = {
                "non_object_owned_mask_path": str(non_object),
                "constraint_eligible_entity_mask_path": str(constraint_path),
                "depth_order_query_support_mask_path": str(support_path),
                "depth_order_query_support_mask_sha256": p18.sha256_file(support_path),
                "depth_order_first_surface_npz_path": str(npz_path),
                "depth_order_first_surface_npz_sha256": p18.sha256_file(npz_path),
                "image_plane_transform": {
                    "camera_contract_consistent": True,
                    "mask_size_wh": [16, 16],
                    "A_mask_from_source_coordinate_model": np.eye(3).tolist(),
                },
                "counts": {},
            }
            _non_object, constraint_mask, diag = p18.visible_ownership_masks_for_row(
                row, {}
            )
            support_mask, support_depth, support_diag = (
                p18.load_depth_order_first_surface_for_row(
                    diag, {}, constraint_mask.shape
                )
            )
            self.assertFalse(bool(constraint_mask[8, 8]))
            self.assertTrue(bool(support_mask[8, 8]))
            self.assertEqual(
                support_diag["raw_depth_under_hand_consumed"], False
            )
            frame = {
                "frame_idx": 0,
                "camera": {"T_world_camera_metric": np.eye(4).tolist()},
                "hands": [
                    {
                        "hand_side": "left",
                        "metric_mano_state": {
                            "current_v18_camera_intrinsics_fx_fy_cx_cy": [
                                8.0,
                                8.0,
                                8.0,
                                8.0,
                            ]
                        },
                    }
                ],
            }
            args = SimpleNamespace(
                visible_surface_depth_order_term=True,
                visible_object_mask_dilation_px=0,
                visible_surface_depth_order_margin_m=0.01,
                max_visible_surface_depth_vertices=16,
            )
            vertices = np.asarray([[0.0, 0.0, 1.05]], dtype=float)
            hard_ids, *_ = p18.visible_surface_depth_order_constraints(
                frame=frame,
                side="left",
                vertices_world=vertices,
                mask=constraint_mask,
                depth_row={"depth": support_depth},
                A_mask_from_source=np.eye(3),
                depth_is_mask_plane=True,
                mask_dilation_px=0,
                args=args,
                enabled=True,
            )
            support_ids, *_ = p18.visible_surface_depth_order_constraints(
                frame=frame,
                side="left",
                vertices_world=vertices,
                mask=support_mask,
                depth_row={"depth": support_depth},
                A_mask_from_source=np.eye(3),
                depth_is_mask_plane=True,
                mask_dilation_px=0,
                args=args,
                enabled=True,
            )
            self.assertEqual(len(hard_ids), 0)
            self.assertEqual(support_ids.tolist(), [0])

    def test_translation_gate_threshold_zero_distinguishes_zero_and_one_support(self) -> None:
        self.assertTrue(
            p18.translation_support_gate_applies(
                enabled=True, support_count=0, minimum_supported_vertices=0
            )
        )
        self.assertFalse(
            p18.translation_support_gate_applies(
                enabled=True, support_count=1, minimum_supported_vertices=0
            )
        )
        self.assertFalse(
            p18.translation_support_gate_applies(
                enabled=False, support_count=0, minimum_supported_vertices=0
            )
        )
        with self.assertRaises(ValueError):
            p18.translation_support_gate_applies(
                enabled=True, support_count=-1, minimum_supported_vertices=0
            )

    def test_temporal_surface_samples_change_camera_world_and_side_pixels(self) -> None:
        points = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
        camera = np.zeros((100, 100, 3), dtype=np.uint8)
        world = np.zeros((100, 100, 3), dtype=np.uint8)
        side = np.zeros((100, 100, 3), dtype=np.uint8)
        camera_count = layered_renderer.draw_camera_temporal_surface(
            camera,
            points,
            np.eye(4),
            (80.0, 80.0, 50.0, 50.0),
        )
        low = np.asarray([-1.0, -1.0, 0.0])
        high = np.asarray([1.0, 1.0, 2.0])
        world_count = layered_renderer.draw_world_temporal_surface(
            world, points, low, high, (0, 2)
        )
        side_count = layered_renderer.draw_world_temporal_surface(
            side, points, low, high, (1, 2)
        )
        self.assertEqual(camera_count, 1)
        self.assertEqual(world_count, 1)
        self.assertEqual(side_count, 1)
        self.assertGreater(int(np.count_nonzero(camera)), 0)
        self.assertGreater(int(np.count_nonzero(world)), 0)
        self.assertGreater(int(np.count_nonzero(side)), 0)

    def test_temporal_surface_map_requires_metric_preservation_and_zero_object_delta(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p18b_renderer_") as temp:
            root = Path(temp)
            row = {
                "frame_idx": 0,
                "hand_side": "left",
                "joint_state_policy": "hawor_npz_metric_mano_preserved",
                "optimized_object_translation_world_m": [0.0, 0.0, 0.0],
                "contact_surface_vertices_world_sample_m": [[0.0, 0.0, 0.5]],
            }
            payload = {"status": "ok", "per_frame_states": [row]}
            path = root / "p18b.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            state = {"temporal_mano_state": {"path": str(path), "payload": payload}}
            rows, report = layered_renderer.temporal_surface_map(state, [])
            self.assertEqual(report["surface_point_count"], 1)
            np.testing.assert_allclose(
                layered_renderer.temporal_surface_points(rows[(0, "left")]),
                [[0.0, 0.0, 0.5]],
            )

            contaminated = json.loads(json.dumps(payload))
            contaminated["per_frame_states"][0][
                "optimized_object_translation_world_m"
            ] = [0.0, 0.002, 0.0]
            bad_path = root / "bad.json"
            bad_path.write_text(json.dumps(contaminated), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "private object motion"):
                layered_renderer.temporal_surface_map(
                    {
                        "temporal_mano_state": {
                            "path": str(bad_path),
                            "payload": contaminated,
                        }
                    },
                    [],
                )

    def test_centered_hawor_plane_exactly_binds_source_K_and_boxes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hawor_centered_plane_") as temp:
            video = Path(temp) / "source.mp4"
            video.write_bytes(b"synthetic-source-video-contract")
            source_K = np.asarray([973.0, 973.0, 702.5, 705.5])
            contract = hawor_export.centered_hawor_plane_contract(
                source_K,
                source_size_wh=(1408, 1408),
                source_video=video,
            )
            inference_K = np.asarray(
                contract["hawor_inference_intrinsics_fx_fy_cx_cy"]
            )
            A = np.asarray(contract["A_hawor_inference_from_source"])
            np.testing.assert_allclose(inference_K, [973.0, 973.0, 704.0, 704.0])
            np.testing.assert_allclose(
                A @ hawor_export.K_from_intrinsics(source_K),
                hawor_export.K_from_intrinsics(inference_K),
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                np.asarray(contract["A_source_from_hawor_inference"]) @ A,
                np.eye(3),
                atol=1.0e-12,
            )
            source_box = np.asarray([100.0, 200.0, 300.0, 400.0, 0.9])
            corners = np.asarray(
                [[source_box[0], source_box[1], 1.0], [source_box[2], source_box[3], 1.0]]
            )
            inference_corners = corners @ A.T
            inference_box = np.asarray(
                [
                    inference_corners[0, 0],
                    inference_corners[0, 1],
                    inference_corners[1, 0],
                    inference_corners[1, 1],
                    0.9,
                ]
            )
            recovered = hawor_export.transform_boxes_to_source_plane(
                inference_box,
                np.asarray(contract["A_source_from_hawor_inference"]),
                (1408, 1408),
            )
            np.testing.assert_allclose(recovered, source_box, atol=1.0e-5)

    def test_p08_revalidates_rectified_frames_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p08_rectified_contract_") as temp:
            root = Path(temp)
            source_video = root / "source.mp4"
            source_video.write_bytes(b"source-video-for-p08-binding")
            frame_dir = root / "rectified" / "extracted_images"
            frame_dir.mkdir(parents=True)
            frames = []
            for index in range(2):
                path = frame_dir / f"{index:04d}.jpg"
                path.write_bytes(f"rectified-frame-{index}".encode())
                frames.append(path)
            source_K = np.asarray([973.0, 973.0, 702.5, 705.5])
            contract = hawor_export.centered_hawor_plane_contract(
                source_K,
                source_size_wh=(1408, 1408),
                source_video=source_video,
            )
            contract.update(
                {
                    "inference_extracted_frames": str(frame_dir),
                    "inference_frame_count": 2,
                    "inference_frames_aggregate_sha256": hawor_export.aggregate_file_sha256(
                        frames
                    ),
                    "inference_first_frame_sha256": hawor_export.sha256(frames[0]),
                    "inference_last_frame_sha256": hawor_export.sha256(frames[-1]),
                }
            )
            contract_path = root / "plane.json"
            contract["path"] = str(contract_path)
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            inference_K = np.asarray(
                contract["hawor_inference_intrinsics_fx_fy_cx_cy"]
            )
            arrays = {
                "frame_idx": np.arange(2, dtype=np.int32),
                "camera_intrinsics_contract_mode": np.asarray(
                    [hawor_export.FULL_K_RECTIFIED_MODE]
                ),
                "camera_intrinsics_fx_fy_cx_cy": np.repeat(
                    source_K[None, :], 2, axis=0
                ),
                "hawor_inference_intrinsics_fx_fy_cx_cy": np.repeat(
                    inference_K[None, :], 2, axis=0
                ),
                "A_hawor_inference_from_source": np.asarray(
                    contract["A_hawor_inference_from_source"]
                ),
                "A_source_from_hawor_inference": np.asarray(
                    contract["A_source_from_hawor_inference"]
                ),
                "camera_image_plane_contract_path": np.asarray([str(contract_path)]),
                "camera_image_plane_contract_sha256": np.asarray(
                    [hawor_export.sha256(contract_path)]
                ),
                "video_sha256": np.asarray([hawor_export.sha256(source_video)]),
            }
            report = base_annotations.validate_hawor_camera_image_plane_contract(
                arrays,
                source_intrinsics=source_K,
                source_size_wh=(1408, 1408),
                source_video=source_video,
            )
            self.assertTrue(report["validated"])
            self.assertEqual(
                report["status"],
                "validated_source_K_affine_centered_hawor_inference_plane",
            )
            frames[1].write_bytes(b"tampered-rectified-frame")
            with self.assertRaisesRegex(RuntimeError, "frame hash mismatch"):
                base_annotations.validate_hawor_camera_image_plane_contract(
                    arrays,
                    source_intrinsics=source_K,
                    source_size_wh=(1408, 1408),
                    source_video=source_video,
                )

    def test_hawor_full_K_is_injected_into_model_mask_renderer_and_slam_then_restored(self) -> None:
        calls: dict[str, object] = {}

        class FakeHAWOR:
            def inference(self, *_args, **kwargs):
                calls["model_focal"] = kwargs.get("img_focal")
                calls["model_center"] = kwargs.get("img_center")
                return "model-result"

        class FakeRenderer:
            def __init__(self) -> None:
                self.K = torch.tensor(
                    [[[1.0, 0.0, 50.0], [0.0, 1.0, 40.0], [0.0, 0.0, 1.0]]]
                )

            def create_camera_from_cv(self, _R, _T, K=None, image_size=None):
                calls["renderer_K"] = K.detach().cpu().numpy().copy()
                calls["renderer_image_size"] = image_size
                return "camera", "lights"

        module = SimpleNamespace(HAWOR=FakeHAWOR, Renderer=FakeRenderer)
        original_inference = FakeHAWOR.inference
        original_camera = FakeRenderer.create_camera_from_cv
        K = np.asarray([973.0, 973.0, 702.5, 705.5], dtype=np.float64)

        def run_fake_motion(*_args):
            model_result = module.HAWOR().inference(
                "images",
                "boxes",
                img_focal=1.0,
                img_center=[50.0, 40.0],
            )
            renderer = module.Renderer()
            camera_result = renderer.create_camera_from_cv(
                torch.eye(3)[None], torch.zeros((1, 3))
            )
            return model_result, camera_result

        result = hawor_export.call_with_hawor_inference_intrinsics(
            run_fake_motion,
            intrinsics=K,
            hawor_video_module=module,
            args=(),
        )
        self.assertEqual(result[0], "model-result")
        self.assertEqual(calls["model_focal"], 973.0)
        self.assertEqual(calls["model_center"], [702.5, 705.5])
        np.testing.assert_allclose(
            np.asarray(calls["renderer_K"])[0],
            [[973.0, 0.0, 702.5], [0.0, 973.0, 705.5], [0.0, 0.0, 1.0]],
        )
        self.assertIs(module.HAWOR.inference, original_inference)
        self.assertIs(module.Renderer.create_camera_from_cv, original_camera)

        def original_est_calib(_images):
            return [1.0, 1.0, 50.0, 40.0]

        slam_module = SimpleNamespace(est_calib=original_est_calib)

        def run_fake_slam(*_args):
            calls["slam_K"] = slam_module.est_calib("images")
            return "slam-result"

        self.assertEqual(
            hawor_export.call_slam_with_hawor_inference_intrinsics(
                run_fake_slam,
                intrinsics=K,
                hawor_slam_module=slam_module,
                args=(),
            ),
            "slam-result",
        )
        np.testing.assert_allclose(calls["slam_K"], K)
        self.assertIs(slam_module.est_calib, original_est_calib)

    def test_hawor_full_K_cache_binding_and_square_focal_contract(self) -> None:
        args = SimpleNamespace(
            camera_intrinsics=[973.0, 973.0, 702.5, 705.5],
            img_focal=973.0,
        )
        intrinsics = hawor_export.requested_camera_intrinsics(args)
        np.testing.assert_allclose(intrinsics, args.camera_intrinsics)
        with self.assertRaisesRegex(RuntimeError, "requires fx==fy"):
            hawor_export.requested_camera_intrinsics(
                SimpleNamespace(
                    camera_intrinsics=[973.0, 974.0, 702.5, 705.5],
                    img_focal=None,
                )
            )

        with tempfile.TemporaryDirectory(prefix="hawor_full_K_cache_") as temp:
            root = Path(temp)
            source_video = root / "source.mp4"
            source_video.write_bytes(b"source-video-for-cache-contract")
            plane = hawor_export.centered_hawor_plane_contract(
                intrinsics,
                source_size_wh=(1408, 1408),
                source_video=source_video,
            )
            plane_path = root / "plane.json"
            plane["path"] = str(plane_path)
            plane_path.write_text(json.dumps(plane), encoding="utf-8")
            tracks = root / "tracks_0_1"
            tracks.mkdir()
            stale = tracks / "model_masks.npy"
            stale.write_bytes(b"stale center-K mask")
            (root / "v19_hawor_focal_cache_contract.json").write_text(
                json.dumps({"img_focal": 973.0}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "full camera contract"):
                hawor_export.prepare_focal_cache_contract(
                    root,
                    0,
                    1,
                    973.0,
                    camera_intrinsics=intrinsics,
                    camera_image_plane_contract=plane,
                    force_refresh=False,
                )
            report = hawor_export.prepare_focal_cache_contract(
                root,
                0,
                1,
                973.0,
                camera_intrinsics=intrinsics,
                camera_image_plane_contract=plane,
                force_refresh=True,
            )
            self.assertFalse(stale.exists())
            self.assertEqual(report["camera_contract_mode"], hawor_export.FULL_K_RECTIFIED_MODE)
            np.testing.assert_allclose(
                report["camera_intrinsics_fx_fy_cx_cy"], intrinsics
            )

    def test_signed_completion_ready_and_unsigned_fallback_are_exactly_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="signed_completion_contract_") as temporary:
            root = Path(temporary)
            observed_path = root / "observed.ply"
            signed_path = root / "signed.ply"
            trimesh.creation.box(extents=[0.1, 0.08, 0.06]).export(signed_path)
            observed = trimesh.creation.box(extents=[0.1, 0.08, 0.06]).submesh([[0, 1]], append=True, repair=False)
            observed.export(observed_path)
            pose_report = root / "pose.json"
            pose_report.write_text("{}", encoding="utf-8")
            fallback_payload = {
                "inputs": {"pose_report": str(pose_report)},
                "outputs": {
                    "pose_hypothesis_mesh_labeled": str(observed_path),
                    "collision_eligible_mesh_labeled": str(observed_path),
                },
                "geometry_readiness": {
                    "signed_geometry_ready": False,
                    "generated_hidden_surface_included": False,
                    "generated_faces_collision_eligible": False,
                    "generated_faces_contact_eligible": False,
                    "generated_faces_signed_distance_eligible": False,
                },
            }
            surface, ready, report = shared_tail.validate_signed_completion(
                fallback_payload, observed_path, observed_path, pose_report
            )
            self.assertFalse(ready)
            self.assertEqual(surface, observed_path)
            self.assertEqual(report["status"], "shared_signed_geometry_attempt_unsigned_observed_fallback")

            signed_payload = json.loads(json.dumps(fallback_payload))
            signed_payload["outputs"]["collision_eligible_mesh_labeled"] = str(signed_path)
            signed_mesh = trimesh.load(signed_path, process=False)
            signed_mesh_sha256 = shared_tail.sha256_file(signed_path)
            authority_path = root / "signed_face_authority.npz"
            np.savez_compressed(
                authority_path,
                face_id=np.arange(len(signed_mesh.faces), dtype=np.int32),
                signed_distance_eligible=np.ones((len(signed_mesh.faces),), dtype=bool),
                provenance_code=np.full((len(signed_mesh.faces),), 3, dtype=np.uint8),
                source_mesh_sha256=np.asarray(signed_mesh_sha256),
                source_mesh_face_count=np.asarray(len(signed_mesh.faces), dtype=np.int64),
                authority_schema=np.asarray("v19_mesh_bound_signed_face_authority_v1"),
                authority_role=np.asarray("selected_collision_surface"),
                readiness_passed=np.asarray(True),
            )
            authority_sha256 = shared_tail.sha256_file(authority_path)
            signed_payload["outputs"]["signed_face_authority_npz"] = str(authority_path)
            signed_payload["geometry_readiness"].update(
                {
                    "signed_geometry_ready": True,
                    "signed_face_authority_required": True,
                    "signed_geometry_consumer_policy": "local_observation_authority_faces_only",
                    "signed_face_authority_npz": str(authority_path),
                    "signed_face_authority_npz_sha256": authority_sha256,
                    "signed_face_authority_mesh_sha256": signed_mesh_sha256,
                }
            )
            surface, ready, report = shared_tail.validate_signed_completion(
                signed_payload, observed_path, observed_path, pose_report
            )
            self.assertTrue(ready)
            self.assertEqual(surface, signed_path)
            self.assertTrue(report["signed_surface_is_volume"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
