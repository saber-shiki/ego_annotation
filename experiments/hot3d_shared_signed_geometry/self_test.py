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
import v19_signed_face_authority as face_authority  # noqa: E402
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

    def test_mesh_bound_face_authority_rejects_hash_mismatch_and_preserves_local_mask(self) -> None:
        with tempfile.TemporaryDirectory(prefix="signed_face_authority_") as temporary:
            root = Path(temporary)
            mesh_path = root / "mesh.ply"
            mesh = __import__("trimesh").creation.box(extents=[0.1, 0.08, 0.06])
            mesh.export(mesh_path)
            mesh_hash = face_authority.sha256_file(mesh_path)
            authority_path = root / "authority.npz"
            eligible = np.zeros((len(mesh.faces),), dtype=bool)
            eligible[:3] = True
            np.savez_compressed(
                authority_path,
                face_id=np.arange(len(mesh.faces), dtype=np.int32),
                signed_distance_eligible=eligible,
                provenance_code=np.where(eligible, 3, 0).astype(np.uint8),
                source_mesh_sha256=np.asarray(mesh_hash),
                source_mesh_face_count=np.asarray(len(mesh.faces), dtype=np.int64),
                authority_schema=np.asarray("v19_mesh_bound_signed_face_authority_v1"),
                authority_role=np.asarray("selected_collision_surface"),
                readiness_passed=np.asarray(True),
            )
            report = {
                "geometry_readiness": {
                    "signed_geometry_ready": True,
                    "signed_geometry_source": "shared_prediction_mask_depth_direct_pose_voxel_reconstruction",
                    "signed_face_authority_required": True,
                    "signed_geometry_consumer_policy": "local_observation_authority_faces_only",
                    "signed_face_authority_npz": str(authority_path),
                    "signed_face_authority_npz_sha256": face_authority.sha256_file(authority_path),
                    "signed_face_authority_mesh_sha256": mesh_hash,
                }
            }
            loaded = face_authority.load_signed_face_authority(
                report,
                source_report_path=root / "report.json",
                mesh_path=mesh_path,
                mesh_face_count=len(mesh.faces),
            )
            np.testing.assert_array_equal(loaded.pop("mask"), eligible)
            self.assertEqual(loaded["eligible_face_count"], 3)
            unsafe_legacy = {
                "geometry_readiness": {
                    "signed_geometry_ready": True,
                    "signed_geometry_source": "shared_prediction_mask_depth_direct_pose_voxel_reconstruction",
                    "signed_geometry_consumer_policy": "all_faces_of_validated_shared_proxy_signed_eligible",
                    "signed_face_authority_npz": str(authority_path),
                    "signed_face_authority_npz_sha256": face_authority.sha256_file(authority_path),
                    "signed_face_authority_mesh_sha256": mesh_hash,
                }
            }
            unsafe = face_authority.load_signed_face_authority(
                unsafe_legacy,
                source_report_path=root / "legacy.json",
                mesh_path=mesh_path,
                mesh_face_count=len(mesh.faces),
            )
            self.assertFalse(unsafe.pop("mask").any())
            self.assertFalse(unsafe["usable"])
            self.assertIn("requires_rebuild", unsafe["state"])
            contaminated = json.loads(json.dumps(report))
            contaminated["geometry_readiness"]["signed_face_authority_mesh_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "different topology mesh"):
                face_authority.load_signed_face_authority(
                    contaminated,
                    source_report_path=root / "report.json",
                    mesh_path=mesh_path,
                    mesh_face_count=len(mesh.faces),
                )

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

            gated = json.loads(json.dumps(rows))
            gated[0]["output_translation_gate"] = {"applied": True}
            accepted_gated, report_gated, _ = p18b.signed_full_mano_acceptance(
                state, gated, args
            )
            self.assertFalse(accepted_gated)
            self.assertIn(
                "published_candidate_contains_translation_gated_rows",
                report_gated["blockers"],
            )

            unauthorized = json.loads(json.dumps(rows))
            unauthorized[1][
                "full_unauthorized_surface_penetrating_vertex_count_after_solver"
            ] = 1
            unauthorized[1][
                "full_unauthorized_surface_penetration_after_solver_m"
            ] = {"max": 0.004}
            accepted_unauthorized, report_unauthorized, _ = (
                p18b.signed_full_mano_acceptance(state, unauthorized, args)
            )
            self.assertFalse(accepted_unauthorized)
            self.assertIn(
                "unresolved_penetration_nearest_unauthorized_closure_face",
                report_unauthorized["blockers"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
