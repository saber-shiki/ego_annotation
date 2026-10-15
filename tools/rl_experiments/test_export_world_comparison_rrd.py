"""Portable RRD mesh/transform/timeline regressions (synthetic geometry only)."""
from pathlib import Path
import hashlib
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tools.rl_experiments.export_world_comparison_rrd import export_rrd, verify_roundtrip, validate_archives


def fixture(root):
    source = root / "source.bin"
    source.write_bytes(b"immutable synthetic test source")
    n = 3
    vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]], np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int64)
    items, archives = [], []
    for j, key in enumerate(("zero", "best", "last")):
        a = {"frame_idx": np.array([4, 5, 6]), "time_s": np.array([0., 1/30, 2/30]),
             "T_world_camera": np.broadcast_to(np.eye(4), (n, 4, 4)).copy(),
             "object_vertices_canonical": vertices, "object_faces": faces,
             "object_rotation": Rotation.from_rotvec(np.array([[.1, .2, .3], [.4, .1, .3], [.3, .2, -.1]]) + j * .01).as_matrix(),
             "object_translation_world": np.array([[.1, .2, .4], [.12, .2, .4], [.14, .2, .4]]) + j * .001,
             "object_valid": np.ones(n, bool)}
        for side in ("left", "right"):
            a.update({side+"_faces": faces, side+"_valid": np.ones(n, bool), side+"_display_valid": np.ones(n, bool),
                      side+"_vertices_world": np.broadcast_to(vertices, (n, 4, 3)).copy() + np.arange(n, dtype=np.float32)[:, None, None]*.01})
        archives.append(a)
        items.append({"key": key, "name": key, "steps": j*150, "objective": 2+j, "path": str(source),
                      "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    return items, archives


class WorldComparisonRrdTest(unittest.TestCase):
    def test_roundtrip_full_mesh_mano_nonidentity_rotation_and_blueprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); items, archives = fixture(root)
            output = root / "compare.rrd"
            report = export_rrd(items, archives, output)
            check = verify_roundtrip(output, items, archives)
            self.assertEqual(report["timeline"]["frames"], 3)
            self.assertEqual(report["native_triangles"], 4)
            self.assertEqual(len(check["mesh_frame_counts"]), 9)
            self.assertTrue(all(n == 3 for n in check["object_pose_frame_counts"].values()))
            self.assertFalse(report["world_offsets_applied"])
            with self.assertRaises(ValueError): export_rrd(items, archives, output)

    def test_clear_and_resume_preserves_mesh_latest_at(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); items, archives = fixture(root)
            for a in archives:
                a["object_valid"][1] = False
                a["left_valid"][1] = a["left_display_valid"][1] = False
            output = root / "compare.rrd"
            export_rrd(items, archives, output)
            check = verify_roundtrip(output, items, archives)
            self.assertTrue(all(n == 2 for n in check["object_pose_frame_counts"].values()))
            self.assertEqual(check["mesh_frame_counts"]["/comparison/zero/world/object"], 2)
            self.assertEqual(check["mesh_frame_counts"]["/comparison/zero/world/hands/left"], 2)

    def test_collision_highlight_keeps_complete_mesh_and_time(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); items, archives = fixture(root)
            reports = []
            for a in archives:
                reports.append({'frames': [{'frame_idx': int(fid), 'hands': {
                    side: {'inside_vertex_ids': [0], 'crossing_edge_vertex_ids': [[0, 1]]}
                    for side in ('left', 'right')}} for fid in a['frame_idx']]})
            output = root / 'collision.rrd'
            receipt = export_rrd(items, archives, output, collision_reports=reports)
            self.assertIn('red MANO vertices', receipt['collision_highlight'])
            self.assertTrue(verify_roundtrip(output, items, archives)['all_mesh_vertices_and_triangle_indices_equal'])

    def test_two_direct_candidates_without_fake_objective_and_triangle_outlines(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);items,archives=fixture(root);items=items[:2];archives=archives[:2]
            for item in items:item['objective']=None
            reports=[{'triangle_tolerance_m':1e-8,'frames':[{'frame_idx':int(fid),'hands':{
                side:{'inside_vertex_ids':[],'intersected_hand_face_ids':[0]} for side in ('left','right')}} for fid in a['frame_idx']]} for a in archives]
            output=root/'direct.rrd';receipt=export_rrd(items,archives,output,collision_reports=reports)
            self.assertIn('intersected hand triangles',receipt['collision_highlight'])
            check=verify_roundtrip(output,items,archives)
            self.assertEqual(len(check['mesh_frame_counts']),6)
            self.assertTrue(check['all_mesh_vertices_and_triangle_indices_equal'])

    def test_reject_changed_timeline_geometry_or_visibility(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, archives = fixture(root)
            archives[1]["time_s"][1] += .01
            with self.assertRaises(AssertionError): validate_archives(archives)
            _, archives = fixture(root)
            archives[1]["left_display_valid"][0] = False
            with self.assertRaises(AssertionError): validate_archives(archives)
            _, archives = fixture(root)
            archives[1]["right_faces"] = np.array([[0, 1, 100]])
            with self.assertRaises(ValueError): validate_archives(archives)


if __name__ == "__main__":
    unittest.main()
