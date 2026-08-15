#!/usr/bin/env python3
"""CPU self-tests for the source-neutral V19 camera/depth contract."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import adapt_v19_depth_to_camera_contract as depth_adapter  # noqa: E402
import build_v19_base_annotations as base_annotations  # noqa: E402
import build_v19_visible_geometry_from_sam2_depth as visible_geometry  # noqa: E402
import resolve_v19_camera_contract as resolver  # noqa: E402
import run_unidepth_full_frame_v3 as full_depth  # noqa: E402
from v19_camera_contract import plane_intrinsics, resize_affine, validate_contract  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_manifest(root: Path, *, count: int = 2, source_size: tuple[int, int] = (100, 80), manifest_size: tuple[int, int] = (50, 40)) -> Path:
    frames = []
    for frame_idx in range(count):
        image = np.zeros((manifest_size[1], manifest_size[0], 3), dtype=np.uint8)
        image_path = root / "rgb" / f"{frame_idx:06d}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), image)
        frames.append(
            {
                "frame_idx": frame_idx,
                "index": frame_idx,
                "rgb": str(image_path),
                "raw_frame_path": str(image_path),
                "source_width": source_size[0],
                "source_height": source_size[1],
                "manifest_width": manifest_size[0],
                "manifest_height": manifest_size[1],
            }
        )
    path = root / "manifest.json"
    source_video = root / "source_video.mp4"
    source_video.write_bytes(b"synthetic prediction and sensor source video")
    write_json(
        path,
        {
            "input_video": str(source_video),
            "video": {"width": source_size[0], "height": source_size[1], "frame_count": count, "fps": 30.0},
            "frames": frames,
        },
    )
    return path


def manifest_source_video(manifest: Path) -> Path:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return Path(payload["input_video"])


def make_depth(
    root: Path,
    *,
    count: int = 2,
    size: tuple[int, int] = (100, 80),
    intrinsics: tuple[float, float, float, float] = (70.0, 72.0, 49.0, 39.0),
) -> Path:
    width, height = size
    depth = np.arange(count * height * width, dtype=np.float16).reshape(count, height, width) / np.float16(1000.0)
    intrinsics_row = np.asarray(intrinsics, dtype=np.float64)
    path = root / "depth.npz"
    np.savez_compressed(
        path,
        frame_idx=np.arange(count, dtype=np.int32),
        depth=depth,
        source_size=np.asarray([width, height], dtype=np.int32),
        focal_px=np.asarray([math.sqrt(intrinsics_row[0] * intrinsics_row[1])] * count, dtype=np.float64),
        intrinsics_fx_fy_cx_cy=np.repeat(intrinsics_row[None, :], count, axis=0),
    )
    return path


def resolve_args(**overrides):
    values = dict(
        case="synthetic",
        raw_frame_manifest=None,
        sensor_calibration_contract=None,
        sensor_calibration_authority="dataset_sensor_calibration",
        sensor_frame_intrinsics_key=None,
        prediction_source_video=None,
        sensor_source_video=None,
        sensor_timeline_time_tolerance_s=1.0e-9,
        fixed_intrinsics_tolerance_px=0.01,
        unidepth_npz=None,
        output_dir=None,
        dataset_name="synthetic_dataset",
        sensor_stream="rgb",
        source_from_calibration_affine=None,
        render_from_source_affine=None,
        pixel_center_convention="integer_pixel_centers_opencv",
        camera_coordinate_convention="x_right_y_down_z_forward",
        depth_convention="camera_z_meters",
        allow_missing_sensor_calibration_fallback=False,
        fallback_reason="sensor calibration unavailable in synthetic data",
        frame_start=None,
        frame_end=None,
        aggregation="median",
        trim_low=0.1,
        trim_high=0.9,
        square_focal=True,
        center_principal_point=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class CameraContractTest(unittest.TestCase):
    def test_resize_affine_matches_opencv_pixel_center_mapping(self) -> None:
        affine = resize_affine((4, 4), (2, 2), pixel_center_convention="integer_pixel_centers_opencv")
        np.testing.assert_allclose(affine, [[0.5, 0.0, -0.25], [0.0, 0.5, -0.25], [0.0, 0.0, 1.0]])
        source_coordinate = np.asarray([0.5, 0.5, 1.0])
        np.testing.assert_allclose(affine @ source_coordinate, [0.0, 0.0, 1.0])
        inverse = np.linalg.inv(affine)
        np.testing.assert_allclose(inverse @ [0.0, 0.0, 1.0], source_coordinate)

    def test_sensor_contract_propagates_explicit_resize(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_camera_contract_sensor_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            sensor = root / "sensor.json"
            write_json(
                sensor,
                {
                    "schema": "synthetic_sensor_camera_v1",
                    "method": "synthetic_sensor_metadata",
                    "image_width": 100,
                    "image_height": 80,
                    "intrinsics_fx_fy_cx_cy": [80.0, 82.0, 50.0, 40.0],
                    "intrinsics_source": "synthetic official sensor K",
                },
            )
            output = root / "contract"
            contract = resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    sensor_calibration_contract=sensor,
                    sensor_source_video=manifest_source_video(manifest),
                    output_dir=output,
                )
            )
            normalized = validate_contract(contract, expected_frame_ids=[0, 1])
            self.assertEqual(
                contract["provenance"]["source_timeline_validation"]["status"],
                "exact_source_video_hash_match_sensor_frame_timeline_unavailable",
            )
            self.assertIsNone(
                contract["provenance"]["source_timeline_validation"]["frame_ids_exact"]
            )
            self.assertEqual(contract["calibration_authority"], "dataset_sensor_calibration")
            self.assertFalse(contract["fallback"]["used"])
            render_intrinsics, transform = plane_intrinsics(
                contract, normalized, plane_name="render", actual_size_wh=(50, 40)
            )
            np.testing.assert_allclose(render_intrinsics, [40.0, 41.0, 24.75, 19.75], atol=1.0e-7)
            np.testing.assert_allclose(
                transform["A_actual_plane_from_calibration"],
                [[0.5, 0.0, -0.25], [0.0, 0.5, -0.25], [0.0, 0.0, 1.0]],
            )
            self.assertTrue((output / "v19_camera_calibration_contract.json").is_file())
            self.assertTrue((output / "v19_camera_calibration_intrinsics.npz").is_file())

    def test_sensor_contract_rejects_different_source_video(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_camera_contract_source_mismatch_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            different_video = root / "different.mp4"
            different_video.write_bytes(b"different source bytes")
            sensor = root / "sensor.json"
            write_json(
                sensor,
                {
                    "image_width": 100,
                    "image_height": 80,
                    "intrinsics_fx_fy_cx_cy": [80.0, 82.0, 50.0, 40.0],
                },
            )
            with self.assertRaisesRegex(RuntimeError, "not byte-identical"):
                resolver.resolve(
                    resolve_args(
                        raw_frame_manifest=manifest,
                        sensor_calibration_contract=sensor,
                        sensor_source_video=different_video,
                        output_dir=root / "out",
                    )
                )

    def test_nearest_exact_sampler_matches_declared_half_pixel_affine(self) -> None:
        source_width, target_width = 960, 1408
        source = np.arange(source_width, dtype=np.float32)[None, :]
        resized = cv2.resize(source, (target_width, 1), interpolation=cv2.INTER_NEAREST_EXACT).reshape(-1)
        A_target_from_source = resize_affine(
            (source_width, 1),
            (target_width, 1),
            pixel_center_convention="integer_pixel_centers_opencv",
        )
        source_from_target = np.linalg.inv(A_target_from_source)
        target_x = np.arange(target_width, dtype=np.float64)
        homogeneous = np.column_stack([target_x, np.zeros(target_width), np.ones(target_width)])
        source_x = (homogeneous @ source_from_target.T)[:, 0]
        expected = np.clip(np.rint(source_x), 0, source_width - 1)
        np.testing.assert_array_equal(resized, expected)
        legacy = cv2.resize(source, (target_width, 1), interpolation=cv2.INTER_NEAREST).reshape(-1)
        self.assertGreater(int(np.count_nonzero(legacy != expected)), 0)

    def test_per_frame_sensor_metadata_key_is_fixed_and_timeline_checked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_camera_contract_frame_metadata_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            sensor = root / "sensor_manifest.json"
            write_json(
                sensor,
                {
                    "schema": "synthetic_sensor_manifest_v1",
                    "width": 100,
                    "height": 80,
                    "frames": [
                        {"frame_idx": 0, "synthetic_sensor_k": [80.0, 82.0, 50.0, 40.0]},
                        {"frame_idx": 1, "synthetic_sensor_k": [80.002, 82.002, 50.001, 40.001]},
                    ],
                },
            )
            contract = resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    sensor_calibration_contract=sensor,
                    sensor_frame_intrinsics_key="synthetic_sensor_k",
                    sensor_source_video=manifest_source_video(manifest),
                    output_dir=root / "out",
                )
            )
            self.assertEqual(
                contract["provenance"]["intrinsics_extraction"]["mode"],
                "per_frame_metadata_fixed_pinhole_median",
            )
            np.testing.assert_allclose(contract["intrinsics_fx_fy_cx_cy"], [80.001, 82.001, 50.0005, 40.0005])
            broken = json.loads(sensor.read_text(encoding="utf-8"))
            broken["frames"] = broken["frames"][:1]
            write_json(sensor, broken)
            with self.assertRaises(RuntimeError):
                resolver.resolve(
                    resolve_args(
                        raw_frame_manifest=manifest,
                        sensor_calibration_contract=sensor,
                        sensor_frame_intrinsics_key="synthetic_sensor_k",
                        sensor_source_video=manifest_source_video(manifest),
                        output_dir=root / "broken_out",
                    )
                )

    def test_hawor_focal_mismatch_is_explicit_not_silently_relabelled(self) -> None:
        mismatch = base_annotations.hawor_active_camera_contract_alignment(
            537.0213,
            [[609.8501, 609.8501, 707.4875, 702.3218]] * 2,
            source_hawor_state_present=True,
            source_hawor_image_size_wh=(1408, 1408),
        )
        self.assertTrue(mismatch["active_contract_reinference_required"])
        self.assertFalse(mismatch["source_hawor_state_intrinsics_match_active_contract"])
        self.assertFalse(mismatch["source_hawor_focal_matches_active_contract"])
        self.assertFalse(mismatch["source_hawor_principal_point_matches_active_contract"])
        self.assertFalse(mismatch["builder_reestimated_hawor_camera_or_mano"])
        np.testing.assert_allclose(
            mismatch["source_hawor_intrinsics_fx_fy_cx_cy"],
            [537.0213, 537.0213, 704.0, 704.0],
        )
        np.testing.assert_allclose(
            mismatch["active_minus_hawor_intrinsics_fx_fy_cx_cy_px"],
            [72.8288, 72.8288, 3.4875, -1.6782],
            atol=1.0e-4,
        )
        explicit_source = base_annotations.hawor_active_camera_contract_alignment(
            None,
            [[80.0, 82.0, 50.0, 40.0]] * 2,
            source_hawor_state_present=True,
            source_hawor_image_size_wh=(100, 80),
            source_hawor_intrinsics_fx_fy_cx_cy=[80.0, 82.0, 50.0, 40.0],
        )
        self.assertTrue(explicit_source["source_hawor_state_intrinsics_match_active_contract"])
        match = base_annotations.hawor_active_camera_contract_alignment(
            609.8501,
            [[609.8501, 609.8501, 704.0, 704.0]] * 2,
            source_hawor_state_present=True,
            source_hawor_image_size_wh=(1408, 1408),
        )
        self.assertFalse(match["active_contract_reinference_required"])
        self.assertTrue(match["source_hawor_state_intrinsics_match_active_contract"])
        absent = base_annotations.hawor_active_camera_contract_alignment(
            None,
            [],
            source_hawor_state_present=False,
        )
        self.assertEqual(absent["status"], "no_source_hawor_state")
        self.assertFalse(absent["active_contract_reinference_required"])

    def test_malformed_v2_contract_does_not_silently_downgrade_to_legacy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_camera_contract_no_downgrade_") as temp:
            path = Path(temp) / "malformed_v2.json"
            write_json(
                path,
                {
                    "schema": "v19_camera_image_transform_contract_v2",
                    "intrinsics_fx_fy_cx_cy": [80.0, 82.0, 50.0, 40.0],
                    "calibration_plane": {"width": 100, "height": 80},
                },
            )
            with self.assertRaises(RuntimeError):
                base_annotations.load_calibration_contract(path, [0, 1])
            with self.assertRaises(RuntimeError):
                visible_geometry.load_calibration_contract(path, frame_ids=[0, 1])

    def test_missing_sensor_calibration_fails_without_explicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_camera_contract_missing_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            depth = make_depth(root)
            with self.assertRaises(FileNotFoundError):
                resolver.resolve(
                    resolve_args(
                        raw_frame_manifest=manifest,
                        sensor_calibration_contract=root / "missing.json",
                        unidepth_npz=depth,
                        output_dir=root / "out",
                    )
                )

    def test_estimated_fallback_retains_common_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_camera_contract_fallback_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            depth = make_depth(root)
            contract = resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    unidepth_npz=depth,
                    output_dir=root / "out",
                )
            )
            normalized = validate_contract(contract, expected_frame_ids=[0, 1])
            self.assertEqual(contract["calibration_authority"], "estimated_from_rgb_depth_model")
            self.assertTrue(contract["fallback"]["used"])
            self.assertIn("source_rgb", normalized["image_planes"])
            self.assertIn("render", normalized["image_planes"])

    def test_base_annotations_consumes_source_rgb_plane_not_native_calibration_plane(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_base_camera_plane_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input", source_size=(100, 80), manifest_size=(50, 40))
            sensor = root / "sensor.json"
            write_json(
                sensor,
                {
                    "image_width": 200,
                    "image_height": 160,
                    "intrinsics_fx_fy_cx_cy": [160.0, 164.0, 100.0, 80.0],
                },
            )
            resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    sensor_calibration_contract=sensor,
                    sensor_source_video=manifest_source_video(manifest),
                    source_from_calibration_affine=[0.5, 0.0, -0.25, 0.0, 0.5, -0.25, 0.0, 0.0, 1.0],
                    output_dir=root / "contract",
                )
            )
            rows, summary = base_annotations.load_calibration_contract(
                root / "contract" / "v19_camera_calibration_contract.json", [0, 1]
            )
            np.testing.assert_allclose(rows[0][0], [80.0, 82.0, 49.75, 39.75])
            self.assertEqual(summary["consumed_image_plane"], "source_rgb")
            self.assertEqual(summary["calibration_size_wh"], [200, 160])

    def test_visible_geometry_v2_plane_backprojection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_visible_geometry_contract_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            sensor = root / "sensor.json"
            write_json(
                sensor,
                {
                    "image_width": 100,
                    "image_height": 80,
                    "intrinsics_fx_fy_cx_cy": [80.0, 82.0, 50.0, 40.0],
                    "intrinsics_source": "synthetic official sensor K",
                },
            )
            resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    sensor_calibration_contract=sensor,
                    sensor_source_video=manifest_source_video(manifest),
                    output_dir=root / "contract",
                )
            )
            contract_path = root / "contract" / "v19_camera_calibration_contract.json"
            _, source, summary, v2 = visible_geometry.load_calibration_contract(
                contract_path, frame_ids=[0, 1]
            )
            self.assertIsNotNone(v2)
            self.assertEqual(summary["calibration_authority"], "dataset_sensor_calibration")
            intrinsics, transform = visible_geometry.camera_contract_plane_intrinsics(
                v2["payload"],
                v2["normalized"],
                plane_name="source_rgb",
                actual_size_wh=(100, 80),
                allow_implicit_resize=False,
            )
            np.testing.assert_allclose(intrinsics, [80.0, 82.0, 50.0, 40.0])
            camera, world, sampling = visible_geometry.choose_visible_points(
                np.ones((1, 1), dtype=bool),
                np.asarray([[2.0]], dtype=np.float32),
                np.asarray([2.0, 2.0, 0.0, 0.0]),
                np.eye(4, dtype=np.float64),
                pixel_stride=1,
                max_points=10,
                rng=np.random.default_rng(0),
            )
            np.testing.assert_allclose(camera, [[0.0, 0.0, 2.0]])
            np.testing.assert_allclose(world, camera)
            self.assertEqual(sampling["sampled_points"], 1)

    def test_first_surface_depth_ownership_rejects_sparse_boundary_tail(self) -> None:
        mask = np.zeros((40, 50), dtype=bool)
        mask[5:35, 5:45] = True
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = 0.40
        # A gradual in-mask boundary tail defeats a single largest-gap split but
        # must not own metric surfels or reconstruction scale.
        depth[5, 5:45] = np.linspace(0.55, 1.40, 40, dtype=np.float32)
        confidence = np.ones(mask.shape, dtype=np.float32)
        confidence[5, 5:45] = 10.0
        robust, diagnostic = visible_geometry.robust_first_surface_depth_ownership(
            mask,
            depth,
            enabled=True,
            mad_sigma=2.5,
            min_half_width_m=0.03,
            min_retained_fraction=0.90,
            fail_raw_to_robust_extent_ratio=2.0,
            intrinsics=np.asarray([40.0, 41.0, 24.75, 19.75]),
            confidence=confidence,
        )
        self.assertFalse(diagnostic["fail_closed"])
        self.assertTrue(diagnostic["raw_tail_dominates_extent_but_is_quarantined"])
        self.assertGreater(diagnostic["removed_depth_pixels"], 0)
        self.assertTrue(np.all(~robust[5, 5:45]))
        self.assertGreater(diagnostic["raw_to_robust_extent_diag_ratio"], 2.0)

    def test_first_surface_depth_ownership_preserves_supported_object_thickness(self) -> None:
        mask = np.zeros((40, 50), dtype=bool)
        mask[5:35, 5:45] = True
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = np.tile(np.linspace(0.38, 0.46, 40, dtype=np.float32), 30)
        robust, diagnostic = visible_geometry.robust_first_surface_depth_ownership(
            mask,
            depth,
            enabled=True,
            mad_sigma=2.5,
            min_half_width_m=0.03,
            min_retained_fraction=0.90,
            fail_raw_to_robust_extent_ratio=2.0,
            intrinsics=np.asarray([40.0, 41.0, 24.75, 19.75]),
            confidence=np.ones(mask.shape, dtype=np.float32),
        )
        self.assertFalse(diagnostic["fail_closed"])
        self.assertEqual(int(np.count_nonzero(robust)), int(np.count_nonzero(mask)))
        self.assertAlmostEqual(diagnostic["retained_fraction"], 1.0)

    def test_first_surface_depth_ownership_fails_when_rejection_is_interior(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[5:55, 5:55] = True
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = 0.40
        depth[25:35, 25:35] = 0.90
        confidence = np.ones(mask.shape, dtype=np.float32)
        confidence[25:35, 25:35] = 10.0
        _robust, diagnostic = visible_geometry.robust_first_surface_depth_ownership(
            mask,
            depth,
            enabled=True,
            mad_sigma=2.5,
            min_half_width_m=0.03,
            min_retained_fraction=0.90,
            fail_raw_to_robust_extent_ratio=2.0,
            intrinsics=np.asarray([50.0, 50.0, 29.5, 29.5]),
            confidence=confidence,
            max_removed_distance_inside_mask_px=10.0,
        )
        self.assertTrue(diagnostic["fail_closed"])
        self.assertIn("rejected_interior_depth_not_sparse_and_confidence_flagged", diagnostic["failure_reasons"])

    def test_first_surface_depth_ownership_quarantines_sparse_confidence_flagged_interior_hole(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[5:55, 5:55] = True
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = 0.40
        depth[28:32, 28:32] = 0.90
        confidence = np.ones(mask.shape, dtype=np.float32)
        confidence[28:32, 28:32] = 10.0
        robust, diagnostic = visible_geometry.robust_first_surface_depth_ownership(
            mask,
            depth,
            enabled=True,
            mad_sigma=2.5,
            min_half_width_m=0.03,
            min_retained_fraction=0.90,
            fail_raw_to_robust_extent_ratio=2.0,
            intrinsics=np.asarray([50.0, 50.0, 29.5, 29.5]),
            confidence=confidence,
            max_removed_distance_inside_mask_px=10.0,
        )
        quarantine = diagnostic["interior_rejection_quarantine"]
        self.assertFalse(diagnostic["fail_closed"])
        self.assertEqual(diagnostic["state"], "validated_sparse_interior_and_boundary_depth_quarantined")
        self.assertEqual(quarantine["interior_removed_pixels"], 16)
        self.assertTrue(quarantine["validated"])
        self.assertEqual(quarantine["mode"], "sparse_unidepth_predicted_error_flagged_interior_holes")
        self.assertTrue(np.all(~robust[28:32, 28:32]))

    def test_first_surface_depth_ownership_rejects_unexplained_sparse_interior_surface(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[5:55, 5:55] = True
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = 0.40
        depth[28:32, 28:32] = 0.90
        confidence = np.ones(mask.shape, dtype=np.float32)
        _robust, diagnostic = visible_geometry.robust_first_surface_depth_ownership(
            mask,
            depth,
            enabled=True,
            mad_sigma=2.5,
            min_half_width_m=0.03,
            min_retained_fraction=0.90,
            fail_raw_to_robust_extent_ratio=2.0,
            intrinsics=np.asarray([50.0, 50.0, 29.5, 29.5]),
            confidence=confidence,
            max_removed_distance_inside_mask_px=10.0,
        )
        self.assertTrue(diagnostic["fail_closed"])
        self.assertFalse(diagnostic["interior_rejection_quarantine"]["validated"])
        self.assertIn("rejected_interior_depth_not_sparse_and_confidence_flagged", diagnostic["failure_reasons"])

    def test_visible_geometry_rejects_depth_bound_to_another_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_visible_depth_binding_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            depth_path = make_depth(root, intrinsics=(80.0, 82.0, 50.0, 40.0))
            sensor = root / "sensor.json"
            write_json(
                sensor,
                {
                    "image_width": 100,
                    "image_height": 80,
                    "intrinsics_fx_fy_cx_cy": [80.0, 82.0, 50.0, 40.0],
                    "intrinsics_source": "synthetic official sensor K",
                },
            )
            resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    sensor_calibration_contract=sensor,
                    sensor_source_video=manifest_source_video(manifest),
                    output_dir=root / "contract_a",
                )
            )
            sensor_payload = json.loads(sensor.read_text(encoding="utf-8"))
            sensor_payload["intrinsics_fx_fy_cx_cy"] = [81.0, 83.0, 50.0, 40.0]
            write_json(sensor, sensor_payload)
            resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    sensor_calibration_contract=sensor,
                    sensor_source_video=manifest_source_video(manifest),
                    output_dir=root / "contract_b",
                )
            )
            depth_adapter.adapt(
                SimpleNamespace(
                    source_depth_npz=depth_path,
                    camera_contract=root / "contract_a" / "v19_camera_calibration_contract.json",
                    depth_plane="source_rgb",
                    output_dir=root / "adapted",
                    output_name="adapted.npz",
                    allow_implicit_depth_resize=False,
                    allow_metadata_only_ray_relabel=False,
                    replace=False,
                )
            )
            loaded_depth = visible_geometry.load_depth_npz(root / "adapted" / "adapted.npz")
            _, _, _, contract_a = visible_geometry.load_calibration_contract(
                root / "contract_a" / "v19_camera_calibration_contract.json",
                frame_ids=[0, 1],
            )
            ok = visible_geometry.validate_depth_camera_contract_binding(
                depth=loaded_depth,
                camera_contract_v2=contract_a,
                calibration_contract_path=root / "contract_a" / "v19_camera_calibration_contract.json",
                depth_image_plane="source_rgb",
                allow_implicit_depth_resize=False,
            )
            self.assertEqual(ok["status"], "exact_camera_contract_hash_plane_intrinsics_and_affine_match")
            _, _, _, contract_b = visible_geometry.load_calibration_contract(
                root / "contract_b" / "v19_camera_calibration_contract.json",
                frame_ids=[0, 1],
            )
            with self.assertRaisesRegex(RuntimeError, "hash disagrees"):
                visible_geometry.validate_depth_camera_contract_binding(
                    depth=loaded_depth,
                    camera_contract_v2=contract_b,
                    calibration_contract_path=root / "contract_b" / "v19_camera_calibration_contract.json",
                    depth_image_plane="source_rgb",
                    allow_implicit_depth_resize=False,
                )

    def test_unidepth_full_frame_consumes_camera_contract_before_depth_decode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_unidepth_conditioned_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            sensor = root / "sensor.json"
            write_json(
                sensor,
                {
                    "image_width": 100,
                    "image_height": 80,
                    "intrinsics_fx_fy_cx_cy": [80.0, 82.0, 50.0, 40.0],
                    "intrinsics_source": "synthetic official sensor K",
                },
            )
            resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    sensor_calibration_contract=sensor,
                    sensor_source_video=manifest_source_video(manifest),
                    output_dir=root / "contract",
                )
            )
            contract_path = root / "contract" / "v19_camera_calibration_contract.json"

            class FakeUniDepth:
                def __init__(self) -> None:
                    self.cameras: list[np.ndarray] = []

                def infer(self, image, camera=None):
                    self.assert_camera(camera)
                    self.cameras.append(camera.detach().cpu().numpy().copy())
                    height, width = int(image.shape[-2]), int(image.shape[-1])
                    ys, xs = full_depth.torch.meshgrid(
                        full_depth.torch.arange(height, dtype=full_depth.torch.float32, device=image.device),
                        full_depth.torch.arange(width, dtype=full_depth.torch.float32, device=image.device),
                        indexing="ij",
                    )
                    homogeneous = full_depth.torch.stack((xs, ys, full_depth.torch.ones_like(xs)), dim=0).reshape(3, -1)
                    rays = (full_depth.torch.linalg.inv(camera) @ homogeneous).reshape(3, height, width)
                    rays = rays / full_depth.torch.linalg.norm(rays, dim=0, keepdim=True)
                    return {
                        "depth": full_depth.torch.ones((1, 1, height, width), device=image.device),
                        "confidence": full_depth.torch.ones((1, 1, height, width), device=image.device),
                        "radius": (1.0 / rays[2].clamp(min=1.0e-6))[None, None, ...],
                        "intrinsics": camera[None, ...],
                        "rays": rays[None, ...],
                    }

                @staticmethod
                def assert_camera(camera) -> None:
                    if camera is None:
                        raise AssertionError("camera K was not supplied to UniDepth")

            fake = FakeUniDepth()
            original_load_model = full_depth.load_model
            full_depth.load_model = lambda model_id, device: fake
            try:
                report = full_depth.run(
                    SimpleNamespace(
                        manifest=manifest,
                        output_dir=root / "depth",
                        camera_contract=contract_path,
                        camera_input_plane="manifest_rgb",
                        camera_output_plane="source_rgb",
                        camera_rays_validation_max_angle_deg=0.05,
                        frame_start=0,
                        frame_end=1,
                        unidepth_repo=None,
                        remote_root=None,
                        local_root=None,
                        source_width=100,
                        source_height=80,
                        min_valid_pixels=1,
                        model_id="synthetic",
                        cpu=True,
                    )
                )
            finally:
                full_depth.load_model = original_load_model

            self.assertEqual(report["camera_conditioning"]["mode"], "provided_pinhole_intrinsics")
            self.assertTrue(report["depth_ray_geometry_reprojected"])
            self.assertEqual(report["depth_output_quantity"], "camera_z_m_from_metric_radius_on_exact_contract_rays")
            self.assertEqual(report["camera_output_plane"], "source_rgb")
            self.assertEqual(len(fake.cameras), 2)
            np.testing.assert_allclose(fake.cameras[0][0, 0], 40.0)
            archive_path = Path(report["depth_archive"])
            with np.load(archive_path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["intrinsics_fx_fy_cx_cy"],
                    np.asarray([[80.0, 82.0, 50.0, 40.0]] * 2, dtype=np.float64),
                )
                self.assertEqual(str(archive["camera_conditioning_mode"]), "provided_pinhole_intrinsics")

            adapted = depth_adapter.adapt(
                SimpleNamespace(
                    source_depth_npz=archive_path,
                    camera_contract=contract_path,
                    depth_plane="source_rgb",
                    output_dir=root / "adapted",
                    output_name="adapted.npz",
                    allow_implicit_depth_resize=False,
                    allow_metadata_only_ray_relabel=False,
                    replace=False,
                )
            )
            self.assertTrue(adapted["source_and_resolved_rays_match"])
            self.assertEqual(
                adapted["source_depth_camera_conditioning"]["mode"],
                "provided_pinhole_intrinsics",
            )

    def test_depth_adapter_preserves_non_float32_camera_contract_precision(self) -> None:
        """Regression: official K values need not be exactly representable as float32."""
        with tempfile.TemporaryDirectory(prefix="v19_depth_camera_precision_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            official_intrinsics = np.asarray(
                [975.0954101562501, 975.0954101562501, 49.123456789, 39.987654321],
                dtype=np.float64,
            )
            depth_path = make_depth(root, intrinsics=tuple(official_intrinsics.tolist()))
            self.assertGreater(
                float(np.max(np.abs(official_intrinsics - official_intrinsics.astype(np.float32)))),
                1.0e-6,
            )
            sensor = root / "sensor.json"
            write_json(
                sensor,
                {
                    "image_width": 100,
                    "image_height": 80,
                    "intrinsics_fx_fy_cx_cy": official_intrinsics.tolist(),
                    "intrinsics_source": "synthetic non-float32 official sensor K",
                },
            )
            resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    sensor_calibration_contract=sensor,
                    sensor_source_video=manifest_source_video(manifest),
                    output_dir=root / "contract",
                )
            )
            contract_path = root / "contract" / "v19_camera_calibration_contract.json"
            output_dir = root / "adapted"
            report = depth_adapter.adapt(
                SimpleNamespace(
                    source_depth_npz=depth_path,
                    camera_contract=contract_path,
                    depth_plane="source_rgb",
                    output_dir=output_dir,
                    output_name="adapted.npz",
                    allow_implicit_depth_resize=False,
                    allow_metadata_only_ray_relabel=False,
                    replace=False,
                )
            )
            adapted_path = output_dir / "adapted.npz"
            with np.load(adapted_path, allow_pickle=False) as adapted:
                rows = np.asarray(adapted["intrinsics_fx_fy_cx_cy"])
            self.assertEqual(rows.dtype, np.dtype(np.float64))
            np.testing.assert_array_equal(
                rows,
                np.repeat(official_intrinsics[None, :], 2, axis=0),
            )
            self.assertTrue(report["array_invariants"]["output_intrinsics_float64"])
            self.assertEqual(report["active_intrinsics_dtype"], "float64")

            loaded_depth = visible_geometry.load_depth_npz(adapted_path)
            _, _, _, contract_v2 = visible_geometry.load_calibration_contract(
                contract_path, frame_ids=[0, 1]
            )
            binding = visible_geometry.validate_depth_camera_contract_binding(
                depth=loaded_depth,
                camera_contract_v2=contract_v2,
                calibration_contract_path=contract_path,
                depth_image_plane="source_rgb",
                allow_implicit_depth_resize=False,
            )
            self.assertEqual(
                binding["status"],
                "exact_camera_contract_hash_plane_intrinsics_and_affine_match",
            )

    def test_depth_adapter_changes_only_camera_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v19_depth_camera_adapter_") as temp:
            root = Path(temp)
            manifest = make_manifest(root / "input")
            depth_path = make_depth(root)
            sensor = root / "sensor.json"
            write_json(
                sensor,
                {
                    "image_width": 100,
                    "image_height": 80,
                    "intrinsics_fx_fy_cx_cy": [80.0, 82.0, 50.0, 40.0],
                    "intrinsics_source": "synthetic official sensor K",
                },
            )
            contract = resolver.resolve(
                resolve_args(
                    raw_frame_manifest=manifest,
                    sensor_calibration_contract=sensor,
                    sensor_source_video=manifest_source_video(manifest),
                    output_dir=root / "contract",
                )
            )
            contract_path = root / "contract" / "v19_camera_calibration_contract.json"
            output_dir = root / "adapted"
            with self.assertRaisesRegex(RuntimeError, "cannot reproject it onto new sensor rays"):
                depth_adapter.adapt(
                    SimpleNamespace(
                        source_depth_npz=depth_path,
                        camera_contract=contract_path,
                        depth_plane="source_rgb",
                        output_dir=output_dir,
                        output_name="adapted.npz",
                        allow_implicit_depth_resize=False,
                        allow_metadata_only_ray_relabel=False,
                        replace=False,
                    )
                )
            report = depth_adapter.adapt(
                SimpleNamespace(
                    source_depth_npz=depth_path,
                    camera_contract=contract_path,
                    depth_plane="source_rgb",
                    output_dir=output_dir,
                    output_name="adapted.npz",
                    allow_implicit_depth_resize=False,
                    allow_metadata_only_ray_relabel=True,
                    replace=False,
                )
            )
            self.assertTrue(report["metadata_only_ray_relabel_override"])
            self.assertFalse(report["depth_ray_geometry_reprojected"])
            with np.load(depth_path, allow_pickle=False) as source, np.load(output_dir / "adapted.npz", allow_pickle=False) as adapted:
                self.assertTrue(np.array_equal(source["depth"], adapted["depth"], equal_nan=True))
                np.testing.assert_allclose(adapted["intrinsics_fx_fy_cx_cy"], [[80.0, 82.0, 50.0, 40.0]] * 2)
                np.testing.assert_allclose(adapted["source_estimated_intrinsics_fx_fy_cx_cy"], source["intrinsics_fx_fy_cx_cy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
