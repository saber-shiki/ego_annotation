import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
