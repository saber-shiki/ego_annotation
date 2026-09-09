import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from build_v19_demo_pose_binding import build_demo_pose_rows, readiness, scale_mesh


def pose(frame, time_s, angle=0.0, x=0.0):
    return {
        "frame_idx": frame,
        "time_s": time_s,
        "status": "fit_to_visible_depth_samples",
        "rigid_pose_observation_eligible": True,
        "rotation_world_from_completed_canonical_matrix": Rotation.from_euler("z", angle).as_matrix().tolist(),
        "translation_world_m": [x, 0.0, 1.0],
        "source_marker": f"direct-{frame}",
    }


class DemoPoseBinding(unittest.TestCase):
    def test_ready_p15_passes_rows_through_exactly(self):
        frames = [{"frame_idx": 30, "time_s": 2.0}, {"frame_idx": 10, "time_s": 1.0}]
        rows = [pose(30, 2.0, x=3), pose(10, 1.0, x=1)]
        p15 = {"annotation_ready": True, "graph_support": {"sufficient": True}, "pose_rows": rows}
        actual, summary = build_demo_pose_rows(p15, {"pose_rows": []}, frames)
        self.assertEqual(actual, rows)
        self.assertEqual(summary["mode"], "ready_P15_pass_through")
        self.assertIsNot(actual, rows)

    def test_unready_uses_direct_p14_and_slerps_only_missing_irregular_time(self):
        frames = [
            {"frame_idx": 89, "time_s": 10.25},
            {"frame_idx": 500, "time_s": 11.0},
            {"frame_idx": 88, "time_s": 10.0},
        ]
        direct0 = pose(88, 10.0, angle=0, x=0)
        direct1 = pose(500, 11.0, angle=np.pi, x=4)
        p15 = {"annotation_ready": False, "graph_support": {"sufficient": False}, "pose_rows": []}
        actual, summary = build_demo_pose_rows(p15, {"pose_rows": [direct1, direct0]}, frames)
        by_id = {row["frame_idx"]: row for row in actual}
        self.assertEqual(by_id[88]["source_marker"], "direct-88")
        self.assertEqual(by_id[500]["source_marker"], "direct-500")
        np.testing.assert_allclose(by_id[89]["translation_world_m"], [1, 0, 1], atol=1e-10)
        expected = Rotation.from_euler("z", np.pi / 4).as_matrix()
        np.testing.assert_allclose(by_id[89]["rotation_world_from_completed_canonical_matrix"], expected, atol=1e-8)
        self.assertFalse(by_id[89]["rigid_pose_observation_eligible"])
        self.assertEqual(by_id[89]["demo_visual_provenance"]["bracket_frame_ids"], [88, 500])
        self.assertEqual(summary["interpolated_frame_ids"], [89])

    def test_unknown_readiness_and_invalid_or_unbracketed_rows_fail_closed(self):
        frames = [{"frame_idx": idx, "time_s": float(idx)} for idx in (5, 8, 12)]
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_demo_pose_rows({"pose_rows": []}, {"pose_rows": []}, frames)
        unready = {"annotation_ready": False, "pose_rows": []}
        latent = pose(8, 8.0)
        latent["status"] = "latent"
        with self.assertRaisesRegex(ValueError, "not an eligible direct"):
            build_demo_pose_rows(unready, {"pose_rows": [pose(5, 5), latent, pose(12, 12)]}, frames)
        with self.assertRaisesRegex(ValueError, "not bracketed"):
            build_demo_pose_rows(unready, {"pose_rows": [pose(8, 8), pose(12, 12)]}, frames)

    def test_empty_placeholder_and_absent_row_share_irregular_time_interpolation(self):
        frames = [{"frame_idx": idx, "time_s": t} for idx, t in
                  [(7, 1.0), (21, 1.2), (34, 1.75), (99, 2.0)]]
        endpoints = [dict(pose(7, 1.0, x=0), object_id="selected"),
                     dict(pose(99, 2.0, angle=np.pi / 2, x=4), object_id="selected")]
        placeholder = {"frame_idx": 21, "object_id": "selected",
                       "status": "missing_accepted_metric_surface", "appearance_mask_path": "source.png",
                       "rotation_world_from_completed_canonical_matrix": None,
                       "translation_world_m": None, "rigid_pose_observation_eligible": False,
                       "generated_geometry_pose_evidence_consumed": False}
        p14 = {"object_id": "selected", "pose_rows": [endpoints[0], placeholder, endpoints[1]]}
        p15 = {"object_id": "selected", "annotation_ready": False, "pose_rows": []}
        original = copy.deepcopy(p14)
        rows, summary = build_demo_pose_rows(p15, p14, frames)
        self.assertEqual(p14, original)
        self.assertEqual(summary["interpolated_frame_ids"], [21, 34])
        for row, direct in zip((rows[0], rows[-1]), endpoints):
            self.assertEqual({k: v for k, v in row.items() if k != "demo_visual_provenance"}, direct)
        for row, alpha in zip(rows[1:3], (.2, .75)):
            np.testing.assert_allclose(row["translation_world_m"], [alpha * 4, 0, 1])
            np.testing.assert_allclose(row["rotation_world_from_completed_canonical_matrix"],
                                       Rotation.from_euler("z", alpha * np.pi / 2).as_matrix(), atol=1e-10)
            self.assertEqual(row["object_id"], "selected")
            self.assertFalse(row["annotation_ready"])
            self.assertFalse(row["direct_visible_measurement"])
            self.assertFalse(row["rigid_pose_observation_eligible"])
            self.assertTrue(row["demo_visual_provenance"]["latent_interpolated"])
        nested = rows[1]["demo_visual_provenance"]["source_placeholder_row"]
        self.assertEqual(nested, placeholder)
        self.assertIsNot(nested, placeholder)
        self.assertIsNone(rows[2]["demo_visual_provenance"]["source_placeholder_row"])

    def test_placeholder_contradictions_unknown_status_and_generated_evidence_rejected(self):
        frames = [{"frame_idx": i, "time_s": float(i)} for i in (1, 2, 3)]
        p15 = {"annotation_ready": False, "object_id": "selected", "pose_rows": []}
        missing = {"frame_idx": 2, "object_id": "selected", "status": "missing_accepted_metric_surface"}
        cases = [{key: value} for key, value in (
            ("rotation_world_from_completed_canonical_matrix", np.eye(3).tolist()),
            ("translation_world_m", [0, 0, 1]), ("translation_world_m", []),
            ("rigid_pose_observation_eligible", True), ("direct_visible_measurement", True),
            ("direct_metric_observation_added", True), ("observed", True), ("annotation_ready", True),
            ("generated_geometry_pose_evidence_consumed", True),
            ("generated_geometry_pose_evidence_consumed", 1),
            ("generated_geometry_pose_evidence", True),
            ("direct_pose_observation_source", "direct"),
            ("pose_measurement_status", "fit_to_visible_depth_samples"),
            ("status", "unknown"), ("status", "rigid_pose_observation_ineligible"),
            ("object_id", "wrong"))]
        for updates in cases:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                build_demo_pose_rows(p15, {"pose_rows": [pose(1, 1), {**missing, **updates}, pose(3, 3)]}, frames)
        for key in ("generated_geometry_pose_evidence", "generated_geometry_pose_evidence_consumed"):
            direct = dict(pose(2, 2), **{key: True})
            with self.subTest(direct_flag=key), self.assertRaisesRegex(ValueError, "not an eligible direct"):
                build_demo_pose_rows(p15, {"pose_rows": [pose(1, 1), direct, pose(3, 3)]}, frames)
            with self.subTest(report_flag=key), self.assertRaisesRegex(ValueError, "not an eligible direct"):
                build_demo_pose_rows(p15, {"pose_rows": [pose(1, 1), pose(3, 3)], key: True}, frames)

    def test_wrong_object_duplicates_nonfinite_and_missing_endpoints_fail_closed(self):
        frames = [{"frame_idx": i, "time_s": float(i)} for i in (1, 2, 3)]
        unready = {"annotation_ready": False, "object_id": "selected", "pose_rows": []}
        bad_rows = [
            [dict(pose(1, 1), object_id="wrong"), pose(3, 3)],
            [pose(1, 1), pose(1, 1), pose(3, 3)],
            [dict(pose(1, 1), translation_world_m=[np.nan, 0, 0]), pose(3, 3)],
            [dict(pose(1, 1), rotation_world_from_completed_canonical_matrix=np.full((3, 3), np.inf).tolist()), pose(3, 3)],
            [dict(pose(1, 1), rigid_pose_observation_eligible=False), pose(3, 3)],
        ]
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                build_demo_pose_rows(unready, {"pose_rows": rows}, frames)
        for endpoint in (1, 3):
            rows = [pose(i, i) for i in (1, 2, 3) if i != endpoint]
            placeholder = {"frame_idx": endpoint, "status": "missing_accepted_metric_surface"}
            for extra in ([], [placeholder]):
                with self.subTest(endpoint=endpoint, explicit=bool(extra)), self.assertRaisesRegex(ValueError, "not bracketed"):
                    build_demo_pose_rows(unready, {"pose_rows": rows + extra}, frames)

    def test_uniform_scale_preserves_topology_and_rejects_bad_bounds(self):
        import trimesh
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.ply"
            output = root / "scaled.ply"
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1]], float)
            faces = np.array([[0, 1, 2], [0, 3, 1]], int)
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(source)
            receipt = scale_mesh(source, output, {
                "uniform_scale": 1.1, "bounds": [0.8, 1.2],
                "pivot_canonical_m": [0, 0, 0], "selection_source": "synthetic silhouette record",
            })
            scaled = trimesh.load(output, process=False)
            np.testing.assert_array_equal(scaled.faces, faces)
            np.testing.assert_allclose(scaled.vertices, vertices * 1.1)
            self.assertTrue(receipt["faces_exactly_preserved"])
            self.assertFalse(receipt["generated_faces_contact_eligible"])
            with self.assertRaisesRegex(ValueError, "out-of-bound"):
                scale_mesh(source, root / "bad.ply", {
                    "uniform_scale": 1.5, "bounds": [0.8, 1.2],
                    "selection_source": "synthetic",
                })


class DemoReviewGeometry(unittest.TestCase):
    def test_source_bad_vertices_and_faces_and_serialized_mismatch(self):
        import trimesh
        V = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        F = np.array([[0, 1, 2]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.ply"
            source.write_bytes(b"fixture")
            bad_cases = [
                (V * np.nan, F), (V + np.inf, F),
                (V, np.array([[0., 1., 2.]])), (V, np.array([[0, -1, 2]])),
                (V, np.array([[0, 1, 3]])),
            ]
            for vertices, faces in bad_cases:
                with patch.object(trimesh, "load", return_value=SimpleNamespace(vertices=vertices, faces=faces)):
                    with self.assertRaisesRegex(ValueError, "finite vertices"):
                        scale_mesh(source, root / "bad.ply", None)
            for serialized in (V[::-1], V + .01, V[:2], V * np.nan):
                with patch.object(trimesh, "load", side_effect=[SimpleNamespace(vertices=V, faces=F),
                                                              SimpleNamespace(vertices=serialized, faces=F)]):
                    with self.assertRaisesRegex(ValueError, "serialized visual mesh vertices"):
                        scale_mesh(source, root / "bad.ply", None)
                    self.assertFalse((root / "bad.ply").exists())

    def test_legacy_ready_source_is_quarantined(self):
        import build_v19_demo_pose_binding as binding
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p15, completion, output = (root / name for name in ("p15.json", "completion.json", "demo.json"))
            source = {"annotation_ready": True, "diagnostic_only": False,
                      "pose_rows": [pose(9, 0), pose(13, .2)]}
            p15.write_text(json.dumps(source))
            completion.write_text(json.dumps({"render_mesh_contract": {"required_mesh_path": "unused", "required_mesh_sha256": "unused_hash"}}))
            with patch.object(sys, "argv", ["binding", "--pose-report", str(p15),
                      "--completion-report", str(completion), "--output", str(output),
                      "--legacy-ready-p15-binding"]):
                binding.main()
            result = json.loads(output.read_text())
            self.assertFalse(result["annotation_ready"])
            self.assertTrue(result["diagnostic_only"])
            self.assertFalse(result["physical_authority"])
            self.assertEqual(result["authority"], "demo_only_visual_hypothesis")
            self.assertTrue(result["core_source_identity"]["source_readiness"])
            self.assertEqual(json.loads(p15.read_text()), source)

    def test_visual_proximity_roles_manifest_binding_and_far_thumb(self):
        from fit_v20_visual_contact_demo import validate_visual_proximity_contract, visual_proximity_weights
        from v19_runtime_contracts import sha
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "appearance.json"
            path.write_text("{}")
            receipt = {
                "demo_visual_proximity_fit_eligible": True,
                "physical_contact_authority_eligible": False,
                "generated_faces_contact_eligible": False,
                "generated_faces_collision_eligible": False,
                "generated_faces_sign_eligible": False,
                "generated_faces_sdf_eligible": False,
                "generated_faces_nonpenetration_eligible": False,
            }
            receipt.update(output_mesh_path=str(path), output_mesh_sha256=sha(path))
            rep = {"appearance_mask_manifest": {"path": str(path), "sha256": sha(path)},
                   "render_mesh_contract": {"required_mesh_path": str(path), "required_mesh_sha256": sha(path)},
                   "visual_mesh_receipt": receipt}
            validate_visual_proximity_contract(rep, path)
            for bad in ({}, {**receipt, "demo_visual_proximity_fit_eligible": False},
                        {**receipt, "generated_faces_contact_eligible": True}):
                with self.assertRaisesRegex(ValueError, "role contract"):
                    validate_visual_proximity_contract({**rep, "visual_mesh_receipt": bad}, path)
            clone = Path(tmp) / "different.json"
            clone.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                validate_visual_proximity_contract(rep, clone)
            path.write_text("mutated")
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                validate_visual_proximity_contract(rep, path)
        result = visual_proximity_weights([True] * 5, [0.] * 5, [.5, .003, np.nan, np.inf, -.1])
        np.testing.assert_allclose(result, [0, .12, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
