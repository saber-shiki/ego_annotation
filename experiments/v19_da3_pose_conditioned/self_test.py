#!/usr/bin/env python3
"""CPU contract tests for the HOT3D pose-conditioned DA3 depth adapter."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import adapt_v19_depth_to_camera_contract as depth_adapter  # noqa: E402
import build_v19_visible_geometry_from_sam2_depth as visible_geometry  # noqa: E402
import run_da3_pose_conditioned_full_frame_v1 as da3  # noqa: E402


def write_contract(path: Path, source_depth_size: tuple[int, int], frame_ids: list[int]) -> None:
    width, height = source_depth_size
    intrinsics = [80.0, 82.0, width / 2.0, height / 2.0]
    payload = {
        "schema": "v19_camera_image_transform_contract_v2",
        "calibration_authority": "prediction_side_sensor_metadata",
        "intrinsics_model": "constant_pinhole_fx_fy_cx_cy",
        "intrinsics_coordinate_plane": "calibration",
        "intrinsics_fx_fy_cx_cy": intrinsics,
        "K": [[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]],
        "intrinsics_source": "synthetic_official_K",
        "calibration_plane": {"width": width, "height": height},
        "image_planes": {
            "source_rgb": {
                "plane_name": "source_rgb",
                "width": width,
                "height": height,
                "A_plane_from_calibration": np.eye(3).tolist(),
                "transform_kind": "identity",
                "interpolation": None,
                "pixel_center_convention": "integer_pixel_centers_opencv",
            }
        },
        "frame_ids": frame_ids,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class DA3PoseConditionedContractTest(unittest.TestCase):
    def test_windows_cover_timeline_with_bounded_overlap(self) -> None:
        windows = da3.make_windows(150, 16, 4)
        coverage = np.zeros(150, dtype=np.int32)
        for start, end in windows:
            self.assertEqual(end - start, 16)
            coverage[start:end] += 1
        self.assertTrue(np.all(coverage >= 1))
        self.assertLessEqual(int(coverage.max()), 2)
        self.assertEqual(windows[0], (0, 16))
        self.assertEqual(windows[-1], (134, 150))

    def test_windows_short_sequence_is_one_window(self) -> None:
        self.assertEqual(da3.make_windows(8, 16, 4), [(0, 8)])
        with self.assertRaises(ValueError):
            da3.make_windows(8, 4, 4)

    def test_ray_remap_identity_preserves_camera_z(self) -> None:
        depth = np.arange(48, dtype=np.float32).reshape(6, 8) + 1.0
        intrinsics = np.asarray([12.0, 13.0, 3.5, 2.5], dtype=np.float64)
        remapped, report = da3.remap_camera_z_to_exact_rays(
            depth,
            intrinsics,
            intrinsics,
            depth.shape,
        )
        np.testing.assert_allclose(remapped, depth, atol=1.0e-6)
        self.assertEqual(report["valid_fraction"], 1.0)
        self.assertLess(report["sampled_ray_angle_error_deg"]["max"], 2.0e-6)

    def test_ray_remap_marks_outside_source_fov_invalid(self) -> None:
        source = np.ones((7, 9), dtype=np.float32)
        source_intrinsics = np.asarray([20.0, 20.0, 4.0, 3.0])
        target_intrinsics = np.asarray([5.0, 5.0, 4.0, 3.0])
        remapped, report = da3.remap_camera_z_to_exact_rays(
            source,
            source_intrinsics,
            target_intrinsics,
            source.shape,
        )
        self.assertLess(report["valid_fraction"], 1.0)
        self.assertTrue(np.isnan(remapped[0, 0]))
        self.assertTrue(np.isfinite(remapped[3, 4]))

    def test_ray_remap_matches_pinhole_coordinate_change(self) -> None:
        height, width = 9, 11
        source_intrinsics = np.asarray([8.0, 9.0, 5.0, 4.0])
        target_intrinsics = np.asarray([10.0, 11.0, 5.0, 4.0])
        source_x = np.arange(width, dtype=np.float32)[None, :]
        source = np.repeat(source_x, height, axis=0)
        remapped, _ = da3.remap_camera_z_to_exact_rays(
            source,
            source_intrinsics,
            target_intrinsics,
            (height, width),
        )
        map_x, _ = da3.ray_remap_coordinates(
            source_intrinsics,
            target_intrinsics,
            (height, width),
        )
        valid = np.isfinite(remapped)
        # OpenCV's fixed-point bilinear remap quantizes fractional coordinates
        # to 1/32 pixel, so compare to the analytic map at that bound.
        np.testing.assert_allclose(remapped[valid], map_x[valid], atol=1.0 / 32.0)

    def test_camera_contract_accepts_ordered_frame_range_subset(self) -> None:
        da3.require_ordered_contract_subset([0, 1, 2, 3], [1, 2])
        da3.require_ordered_contract_subset([0, 2, 4, 6], [2, 6])
        with self.assertRaisesRegex(RuntimeError, "misses selected frames"):
            da3.require_ordered_contract_subset([0, 1, 2, 3], [1, 7])
        with self.assertRaisesRegex(RuntimeError, "ordered unique subset"):
            da3.require_ordered_contract_subset([0, 1, 2, 3], [2, 1])
        with self.assertRaisesRegex(RuntimeError, "ordered unique subset"):
            da3.require_ordered_contract_subset([0, 1, 2, 3], [1, 1])

    def test_hawor_c2w_is_inverted_to_opencv_w2c(self) -> None:
        with tempfile.TemporaryDirectory(prefix="da3_hawor_camera_") as temp:
            path = Path(temp) / "hawor.npz"
            rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
            translations = np.asarray([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]], dtype=np.float32)
            np.savez_compressed(
                path,
                frame_idx=np.asarray([4, 5], dtype=np.int32),
                R_c2w=rotations,
                t_c2w=translations,
            )
            w2c, report = da3.load_hawor_w2c(path, [4, 5])
            np.testing.assert_allclose(w2c[0], np.eye(4), atol=1.0e-7)
            np.testing.assert_allclose(w2c[1, :3, 3], -translations[1], atol=1.0e-7)
            self.assertEqual(report["output_convention"], "opencv_world_to_camera_4x4")

    def test_hawor_camera_rejects_bad_rotation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="da3_hawor_bad_camera_") as temp:
            path = Path(temp) / "hawor.npz"
            bad = np.eye(3, dtype=np.float32)
            bad[0, 0] = 2.0
            np.savez_compressed(
                path,
                frame_idx=np.asarray([0], dtype=np.int32),
                R_c2w=bad[None],
                t_c2w=np.zeros((1, 3), dtype=np.float32),
            )
            with self.assertRaisesRegex(RuntimeError, "proper rotation"):
                da3.load_hawor_w2c(path, [0])

    def test_confidence_contract_is_higher_is_worse_error_proxy(self) -> None:
        raw = np.asarray([0.25, 0.5, 2.0], dtype=np.float32)
        proxy = 1.0 / np.maximum(raw, 1.0e-6)
        self.assertGreater(proxy[0], proxy[1])
        self.assertGreater(proxy[1], proxy[2])
        self.assertEqual(
            da3.CONFIDENCE_SEMANTICS,
            "monotonic_inverse_DA3_confidence_error_proxy_higher_is_worse_not_metric_error",
        )

    def test_camera_adapter_accepts_pose_conditioned_da3_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="da3_camera_adapter_") as temp:
            root = Path(temp)
            contract = root / "camera.json"
            size = (10, 8)
            frames = [0, 1]
            write_contract(contract, size, frames)
            intrinsics = np.asarray([80.0, 82.0, 5.0, 4.0], dtype=np.float64)
            intrinsics_rows = np.repeat(intrinsics[None], len(frames), axis=0)
            source = root / "da3.npz"
            np.savez_compressed(
                source,
                frame_idx=np.asarray(frames, dtype=np.int32),
                depth=np.ones((2, 8, 10), dtype=np.float16),
                confidence=np.ones((2, 8, 10), dtype=np.float16),
                confidence_semantics=np.asarray(da3.CONFIDENCE_SEMANTICS),
                confidence_role=np.asarray("predicted_error_proxy_higher_is_worse"),
                source_size=np.asarray(size, dtype=np.int32),
                intrinsics_fx_fy_cx_cy=intrinsics_rows,
                source_estimated_intrinsics_fx_fy_cx_cy=intrinsics_rows,
                depth_provider=np.asarray("depth_anything_3"),
                camera_conditioning_mode=np.asarray("provided_pinhole_intrinsics_and_metric_extrinsics"),
                depth_ray_geometry_reprojected=np.asarray(True),
                depth_output_quantity=np.asarray("camera_z_m_ray_remapped_from_DA3_processed_K_to_exact_output_contract_K"),
                inference_camera_contract_sha256=np.asarray(depth_adapter.sha256_file(contract)),
                inference_camera_output_plane=np.asarray("source_rgb"),
                inference_camera_intrinsics_fx_fy_cx_cy=intrinsics_rows,
                inference_camera_extrinsics_w2c=np.repeat(np.eye(4)[None], 2, axis=0),
                pose_conditioning_mode=np.asarray(da3.POSE_CONDITIONING_MODE),
                camera_trajectory_source=np.asarray("synthetic_hawor.npz"),
            )
            output_dir = root / "bound"
            report = depth_adapter.adapt(
                SimpleNamespace(
                    source_depth_npz=source,
                    camera_contract=contract,
                    depth_plane="source_rgb",
                    output_dir=output_dir,
                    output_name="da3_bound.npz",
                    allow_implicit_depth_resize=False,
                    allow_metadata_only_ray_relabel=False,
                    replace=False,
                )
            )
            self.assertEqual(report["depth_provider"], "depth_anything_3")
            self.assertTrue(report["source_and_resolved_rays_match"])
            loaded = visible_geometry.load_depth_npz(output_dir / "da3_bound.npz")
            self.assertEqual(loaded["depth_provider"], "depth_anything_3")
            self.assertEqual(loaded["confidence_role"], "predicted_error_proxy_higher_is_worse")


if __name__ == "__main__":
    unittest.main()
