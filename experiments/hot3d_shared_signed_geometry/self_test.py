#!/usr/bin/env python3
"""CPU mechanism tests for backend-neutral signed geometry and full-MANO promotion."""
from __future__ import annotations

import hashlib
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
EXPERIMENT = REPO / "experiments/sam3d_p11_p12_branch"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EXPERIMENT))

import build_hot3d_shared_signed_geometry as signed_geometry  # noqa: E402
import build_v19_mano_surface_hypothesis_state as p18b  # noqa: E402
import render_p14_p15_layered_state as renderer  # noqa: E402


class SharedSignedGeometryTest(unittest.TestCase):
    def test_category_neutral_occupancy_extracts_closed_volume(self) -> None:
        occupancy = np.zeros((32, 40, 28), dtype=bool)
        occupancy[5:27, 6:34, 4:24] = True
        args = SimpleNamespace(sdf_pad_voxels=5, sdf_smooth_sigma_voxels=0.0)
        mesh = signed_geometry.occupancy_mesh(
            occupancy,
            np.asarray([-0.04, -0.05, -0.03], dtype=np.float64),
            0.0025,
            args,
        )
        topology = signed_geometry.mesh_topology(mesh)
        self.assertTrue(topology["watertight"])
        self.assertTrue(topology["winding_consistent"])
        self.assertTrue(topology["is_volume"])
        self.assertEqual(topology["boundary_edges"], 0)
        self.assertEqual(topology["nonmanifold_edges"], 0)

    def test_accepted_first_surface_depth_uses_exact_mask_affine(self) -> None:
        source_size = 1408
        mask_size = 960
        scale = mask_size / source_size
        A_mask_from_source = np.asarray(
            [
                [scale, 0.0, 0.5 * scale - 0.5],
                [0.0, scale, 0.5 * scale - 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        intrinsics = np.asarray([1000.0, 1000.0, 704.0, 704.0])
        uv_source = np.asarray(
            [[640.0 + (index % 8), 620.0 + (index // 8)] for index in range(64)],
            dtype=np.float64,
        )
        z = np.full((64,), 0.4, dtype=np.float64)
        camera = np.c_[
            (uv_source[:, 0] - intrinsics[2]) / intrinsics[0] * z,
            (uv_source[:, 1] - intrinsics[3]) / intrinsics[1] * z,
            z,
        ]
        candidate = {
            "camera_vertices_sample_m": camera.tolist(),
            "intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        }
        depth, report = signed_geometry.accepted_first_surface_depth_in_mask_plane(
            candidate,
            intrinsics,
            A_mask_from_source,
            (mask_size, mask_size),
            2.0,
        )
        transformed = signed_geometry.transform_uv(uv_source, A_mask_from_source)
        xy = np.rint(transformed).astype(int)
        self.assertTrue(np.isfinite(depth[xy[:, 1], xy[:, 0]]).all())
        self.assertEqual(report["accepted_first_surface_camera_point_count"], 64)
        self.assertGreaterEqual(report["accepted_first_surface_unique_seed_pixel_count"], 16)

    def test_renderer_consumes_accepted_full_778_archive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="signed_renderer_archive_") as temporary:
            root = Path(temporary)
            source_vertices = np.zeros((1, 778, 3), dtype=np.float32)
            accepted_vertices = source_vertices.copy()
            accepted_vertices[0, :, 0] = 0.01
            bridge = root / "bridge.npz"
            source = root / "source.npz"
            accepted = root / "accepted.npz"
            np.savez_compressed(
                bridge,
                vertices_world=source_vertices,
                frame_idx=np.asarray([0], dtype=np.int32),
                hand_side=np.asarray(["left"]),
            )
            np.savez_compressed(
                source,
                left_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            )
            np.savez_compressed(
                accepted,
                frame_idx=np.asarray([0], dtype=np.int32),
                hand_side=np.asarray(["left"]),
                vertices_world_m=accepted_vertices,
            )
            accepted_hash = hashlib.sha256(accepted.read_bytes()).hexdigest()
            temporal_row = {
                "frame_idx": 0,
                "hand_side": "left",
                "accepted_full_mano_state": "accepted_signed_p18_full_778",
                "accepted_full_mano_vertices_world_archive": str(accepted),
                "accepted_full_mano_vertices_world_archive_sha256": accepted_hash,
            }
            hand = {
                "hand_side": "left",
                "metric_mano_state": {
                    "vertices_reference": {
                        "bridge_npz": str(bridge),
                        "bridge_vertices_world_array": "vertices_world",
                        "bridge_row_index": 0,
                        "source_hawor_npz": str(source),
                    },
                    "vertices_sample_indices": [0, 1, 2],
                    "vertices_world_sample_m": source_vertices[0, :3].tolist(),
                },
            }
            cache = renderer.ManoArchiveCache([], {(0, "left"): temporal_row})
            try:
                vertices, faces, report = cache.hand_mesh(0, hand)
                np.testing.assert_allclose(vertices, accepted_vertices[0], atol=0.0)
                np.testing.assert_array_equal(faces, [[0, 1, 2]])
                self.assertEqual(report["mano_render_state"], "accepted_signed_p18_full_778")
                self.assertAlmostEqual(report["accepted_p18_max_vertex_delta_from_source_m"], 0.01)
                self.assertEqual(cache.summary()["accepted_signed_p18_full_surface_rows_read"], 1)
            finally:
                cache.close()

    def test_signed_full_mano_acceptance_and_fail_closed_object_motion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="shared_signed_mano_") as temporary:
            root = Path(temporary)
            archive = root / "full.npz"
            frame_idx = np.asarray([0, 0], dtype=np.int32)
            hand_side = np.asarray(["left", "right"])
            vertices = np.zeros((2, 778, 3), dtype=np.float32)
            joints = np.zeros((2, 21, 3), dtype=np.float32)
            vertices[1, :, 0] = 0.1
            np.savez_compressed(
                archive,
                frame_idx=frame_idx,
                hand_side=hand_side,
                vertices_world_m=vertices,
                joints_world_m=joints,
            )
            archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
            rows = []
            for side in ("left", "right"):
                rows.append(
                    {
                        "frame_idx": 0,
                        "hand_side": side,
                        "optimized_object_translation_world_m": [0.0, 0.0, 0.0],
                        "visible_joint_shift_px": {"max": 1.0},
                        "joint_camera_depth_shift_m": {"max": 0.001},
                        "final_active_constraint_residual_after_solver_m": {"max": 0.0},
                        "signed_object_surface_factor_state": "active_explicit_signed_geometry_ready",
                    }
                )
            state = {
                "physical_state_quarantined": False,
                "physical_surface_contract": {
                    "signed_object_surface_factor_active": True,
                    "signed_geometry_declared_ready": True,
                },
                "full_mano_vertices_world_archive": str(archive),
                "full_mano_vertices_world_archive_sha256": archive_hash,
                "intervals": [
                    {"hand_side": "left", "active_set_closed": True},
                    {"hand_side": "right", "active_set_closed": True},
                ],
            }
            args = SimpleNamespace(
                accept_signed_full_mano=True,
                max_accepted_visible_joint_shift_px=12.1,
                max_accepted_joint_depth_shift_m=0.035,
                max_accepted_penetration_residual_m=0.001,
            )
            accepted, report, archive_rows = p18b.signed_full_mano_acceptance(
                state, rows, args
            )
            self.assertTrue(accepted)
            self.assertTrue(report["accepted"])
            self.assertEqual(set(archive_rows), {(0, "left"), (0, "right")})
            contaminated = json.loads(json.dumps(rows))
            contaminated[0]["optimized_object_translation_world_m"] = [0.001, 0.0, 0.0]
            accepted_bad, report_bad, _ = p18b.signed_full_mano_acceptance(
                state, contaminated, args
            )
            self.assertFalse(accepted_bad)
            self.assertIn("nonzero_private_object_translation", report_bad["blockers"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
