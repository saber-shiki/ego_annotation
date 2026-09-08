"""Small CPU-only timeline/validity tests; no learned inference or licensed assets."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v19_demo_timeline import (annotation_timeline, fit_keyframes, hand_validity,
    interaction_states, interpolate_supported, join_indices, state_timeline)
from v20_demo_geometry import JOINT_ORDER, TIP_VERTICES, project, sha


def fixture():
    ids = np.array([42, 40, 43, 41])
    frames = [{"frame_idx": int(i), "time_s": (int(i) - 40) / 10,
               "source_width": 80, "source_height": 48,
               "camera": {"T_world_camera_metric": np.eye(4).tolist(),
                          "intrinsics_fx_fy_cx_cy": [60, 60, 39.5, 23.5]},
               "objects": []} for i in ids]
    poses = [{"frame_idx": int(i), "rotation_world_from_completed_canonical_matrix": np.eye(3).tolist(),
              "translation_world_m": [0, 0, 1]} for i in ids[::-1]]
    params = {"frame_idx": ids[::-1], "left_valid": np.array([False, True, True, True])}
    return {"frames": frames}, params, poses


class DemoTimeline(unittest.TestCase):
    def test_join_nonzero_reordered_ids_nonsquare_intrinsics(self):
        ann, params, poses = fixture()
        frames, aligned, rows, times, T, K, size, source = annotation_timeline(ann, params, poses, 40)
        np.testing.assert_array_equal(aligned["frame_idx"], [40, 41, 42, 43])
        np.testing.assert_array_equal(aligned["left_valid"], [True, False, True, True])
        self.assertEqual([r["frame_idx"] for r in rows], [40, 41, 42, 43])
        np.testing.assert_array_equal(size, [40, 24])
        points = np.array([[0.1, 0.2, 1.]])
        np.testing.assert_allclose(project(points, K[0]), (project(points, [60, 60, 39.5, 23.5]) + 0.5) / 2 - 0.5)

    def test_reject_ambiguous_joins_times_and_transforms(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            join_indices([1, 1], [1, 2])
        with self.assertRaisesRegex(ValueError, "coverage"):
            join_indices([1, 2], [1, 3])
        ann, params, poses = fixture()
        ann["frames"][0]["time_s"] = ann["frames"][1]["time_s"]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            annotation_timeline(ann, params, poses)
        ann, params, poses = fixture()
        ann["frames"][0]["camera"]["T_world_camera_metric"][0][0] = 2
        with self.assertRaisesRegex(ValueError, "rigid"):
            annotation_timeline(ann, params, poses)
        with self.assertRaisesRegex(ValueError, "coverage"):
            annotation_timeline(fixture()[0], fixture()[1], fixture()[2][:-1])

    def test_missing_side_and_invalid_nan_masks(self):
        z = {"left_valid": [1, 0, 1, 1], "left_vertices_world": np.ones((4, 778, 3))}
        z["left_vertices_world"][2] = np.nan
        np.testing.assert_array_equal(hand_validity(z, "left", 4, fitted=True), [1, 0, 0, 1])
        self.assertFalse(hand_validity(z, "right", 4, fitted=True).any())
        with self.assertRaisesRegex(ValueError, "lacks MANO"):
            hand_validity({"right_valid": [1, 1, 1, 1]}, "right", 4)

    def test_interpolation_does_not_cross_invalid_gap(self):
        times = np.array([0., .1, .3, .4, .7, 1.])
        valid = np.array([1, 1, 0, 1, 1, 1], bool)
        keys = fit_keyframes(6, {"left": valid}, stride=5)
        np.testing.assert_array_equal(keys, [0, 1, 3, 5])
        result = interpolate_supported(times, keys, np.array([[1.], [2.], [10.], [20.]]), valid)
        np.testing.assert_allclose(result[:, 0], [1, 2, 0, 10, 15, 20])

    def test_p17_states_gate_each_side_and_uncovered_frames(self):
        rows = [{"hand_side": "left", "target_entity_id": "object:item", "frame_start": 40,
                 "frame_end": 41, "contact_state": "no_contact"},
                {"side": "right", "object_id": "item", "start_frame": 41,
                 "end_frame": 42, "contact_state": "possible_contact"}]
        states = interaction_states({"interaction_judgments": rows}, np.array([42, 40, 41]), "item")
        self.assertEqual(states.tolist(), [["unresolved", "possible_contact"], ["no_contact", "unresolved"], ["no_contact", "possible_contact"]])
        self.assertEqual(interaction_states(None, [1], "item")[0, 0], "legacy_visual_hypothesis")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            interaction_states({"interaction_judgments": rows + rows}, np.array([40, 41, 42]), "item")

    def test_p17_rejects_malformed_rows_and_wrong_target(self):
        ids = np.array([40, 41, 42])
        valid = {"side": "left", "object_id": "item", "contact_state": "unresolved"}
        for bad in [None, [], "not a segment", 7]:
            with self.subTest(row=bad), self.assertRaisesRegex(ValueError, "segment is not an object"):
                interaction_states({"interaction_judgments": [bad]}, ids, "item")
        for fields in [{"contact_state": []}, {"frame_start": None},
                       {"frame_end": "bad"}, {"frame_start": 40.5},
                       {"object_id": {}}, {"side": {}}]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                interaction_states({"interaction_judgments": [{**valid, **fields}]}, ids, "item")
        with self.assertRaisesRegex(ValueError, "payload is not an object"):
            interaction_states([], ids, "item")
        for fields in [{"object_id": "another_item"}, {"frame_start": 90, "frame_end": 91}]:
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "no segments for target 'item'"):
                interaction_states({"interaction_judgments": [{**valid, **fields}]}, ids, "item")
        # A matching deliberately unresolved interval remains valid, including
        # the integer-string bounds accepted by the existing P17 loader.
        states = interaction_states({"interaction_judgments": [
            {**valid, "frame_start": "40", "frame_end": "42"}]}, ids, "item")
        self.assertTrue(np.all(states == "unresolved"))

    def test_render_timestamp_map_and_legacy_intrinsics(self):
        z = {"frame_idx": np.array([42, 40, 41]), "time_s": [.2, 0, .1],
             "render_size_wh": [80, 48], "intrinsics_render": np.tile([60, 60, 39.5, 23.5], (3, 1)),
             "T_world_camera": np.tile(np.eye(4), (3, 1, 1))}
        ids, times, order, size, K = state_timeline(z, 10, 3, 40)
        np.testing.assert_array_equal(order, [1, 2, 0])
        np.testing.assert_array_equal(size, [40, 24])
        z["time_s"] = [.4, 0, .1]
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            state_timeline(z, 10, 3)
        del z["time_s"]
        del z["render_size_wh"]
        z["frame_idx"] = np.array([2, 0, 1])
        z["intrinsics_960"] = z.pop("intrinsics_render")
        self.assertEqual(state_timeline(z, 10, 3)[3].tolist(), [960, 960])

    def test_real_triangle_renderer_missing_left_keeps_right_identity(self):
        from render_v20_visual_contact_demo import scene_for_frame, shade
        vertices = np.array([[-.4, -.4, 0], [.4, -.4, 0], [0, .4, 0]], np.float32)
        z = {"frame_idx": np.array([40]), "object_vertices_canonical": vertices,
             "object_faces": np.array([[0, 1, 2]]), "object_rotation": np.eye(3)[None],
             "object_translation_world": np.array([[0, 0, 2.]]),
             "T_world_camera": np.eye(4)[None], "right_valid": [True],
             "right_vertices_world": (vertices + [0, 0, 1])[None], "right_faces": np.array([[0, 1, 2]])}
        sc, verts, faces, normals, labels = scene_for_frame(z, 0)
        self.assertEqual(labels, [0, 2])
        image, mask, ids, depth = shade(sc, [20, 20, 19.5, 11.5], (40, 24), faces, normals, labels)
        self.assertEqual(image.shape, (24, 40, 3))
        self.assertGreater(np.count_nonzero(ids == 2), 0)
        self.assertEqual(np.count_nonzero(ids == 1), 0)
        z["right_valid"] = [False]
        self.assertEqual(scene_for_frame(z, 0)[4], [0])

    def test_small_fit_export_single_hand_and_p17_gating(self):
        import torch
        import trimesh
        import cv2
        import fit_v20_visual_contact_demo as fitter
        rng = np.random.default_rng(7)
        vertices = rng.normal(size=(778, 3)).astype("float32") * .01
        vertices[:, 2] += .9
        faces = np.array([[0, 1, 2]], np.int32)
        joints = np.concatenate([vertices[:16], vertices[TIP_VERTICES]])[JOINT_ORDER]

        class TestMano:
            """Differentiable test fixture, not an inference/model substitute in production."""
            def __init__(self):
                self.faces = faces
            def requires_grad_(self, enabled):
                pass
            def __call__(self, global_orient, hand_pose, betas):
                v = torch.tensor(vertices)[None].expand(len(hand_pose), -1, -1) + hand_pose[:, :1, None] * .001
                return SimpleNamespace(vertices=v, joints=v[:, :16])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ann, _, poses = fixture()
            mask = root / "mask.png"
            cv2.imwrite(str(mask), np.full((48, 80), 255, np.uint8))
            for frame in ann["frames"]:
                frame["objects"] = [{"object_id": "item", "mask_path": str(mask)}]
            (root / "ann.json").write_text(json.dumps(ann))
            mesh = root / "mesh.ply"
            trimesh.creation.box(extents=[.1, .1, .1]).export(mesh)
            (root / "poses.json").write_text(json.dumps({"pose_rows": poses, "render_mesh_contract": {
                "required_mesh_path": str(mesh), "required_mesh_sha256": sha(mesh)}}))
            ids = np.array([42, 40, 43, 41])
            valid = np.array([1, 1, 1, 0], bool)
            source_v = np.tile(vertices, (4, 1, 1)); source_v[~valid] = np.nan
            source_j = np.tile(joints, (4, 1, 1)); source_j[~valid] = np.nan
            np.savez(root / "params.npz", frame_idx=ids, left_valid=valid,
                     left_vertices_world_m=source_v, left_joints_world_m=source_j,
                     left_root_orient_axis_angle=np.zeros((4, 3)), left_hand_pose_axis_angle=np.zeros((4, 45)),
                     left_betas=np.zeros((4, 10)), left_trans_world_m=np.zeros((4, 3)), left_faces=faces)
            np.savez(root / "bridge.npz", frame_idx=ids[valid], hand_side=np.array(["left"] * 3),
                     vertices_current_v18_world_from_hawor_projection_relift_m=source_v[valid])
            (root / "judgments.json").write_text(json.dumps({"interaction_judgments": [
                {"side": "left", "frame_start": 40, "frame_end": 43, "contact_state": "no_contact"}]}))
            for side in ["LEFT", "RIGHT"]:
                (root / f"MANO_{side}.pkl").write_bytes(b"test fixture; no model")
            argv = ["fit", "--annotations", str(root / "ann.json"), "--pose-report", str(root / "poses.json"),
                    "--mano-bridge", str(root / "bridge.npz"), "--hawor-params", str(root / "params.npz"),
                    "--mano-models", str(root), "--output-dir", str(root / "out"), "--device", "cpu",
                    "--iterations", "1", "--object-id", "item", "--render-width", "40",
                    "--interaction-judgments", str(root / "judgments.json")]
            with patch.object(sys, "argv", argv), patch.object(fitter, "mano_models", return_value={s: TestMano() for s in ["left", "right"]}), contextlib.redirect_stdout(io.StringIO()):
                fitter.main()
            with np.load(root / "out/demo_state.npz") as state:
                np.testing.assert_array_equal(state["frame_idx"], [40, 41, 42, 43])
                np.testing.assert_array_equal(state["render_size_wh"], [40, 24])
                self.assertFalse(state["right_valid"].any())
                self.assertTrue(np.isnan(state["right_vertices_world"]).all())
                self.assertTrue(np.isnan(state["left_vertices_world"][1]).all())
                self.assertTrue(np.isfinite(state["object_translation_world"]).all())
                np.testing.assert_array_equal(state["left_source_betas"], np.zeros((4, 10)))
                np.testing.assert_array_equal(state["hand_articulation"][1], 0)
                np.testing.assert_allclose(state["left_base_vertices_world"][0], vertices)
            hypotheses = json.loads((root / "out/contact_hypotheses.json").read_text())
            self.assertFalse(np.asarray(hypotheses["weights"]).any())
            legacy_argv = argv[:-2]
            legacy_argv[legacy_argv.index("--output-dir") + 1] = str(root / "legacy")
            with patch.object(sys, "argv", legacy_argv), patch.object(fitter, "mano_models", return_value={s: TestMano() for s in ["left", "right"]}), contextlib.redirect_stdout(io.StringIO()):
                fitter.main()
            legacy = json.loads((root / "legacy/contact_hypotheses.json").read_text())
            self.assertEqual(legacy["interaction_source"], "legacy_visual_hypothesis")
            self.assertGreater(np.count_nonzero(legacy["weights"]), 0)

    def test_fit_contact_losses_and_contour_samples_are_render_width_invariant(self):
        import torch
        import trimesh
        import cv2
        import fit_v20_visual_contact_demo as fitter
        rng = np.random.default_rng(18)
        vertices = rng.normal(size=(778, 3)).astype("float32") * .008
        vertices[:, 2] += .945
        # Reference mask starts at x=480. Distances straddle the legacy 16px
        # thumb and 8px finger thresholds, not just an all-inside contact case.
        tip_u = np.array([460., 474., 470., 460., 490.])
        vertices[TIP_VERTICES, 0] = (tip_u - 479.5) * .945 / 720
        vertices[TIP_VERTICES, 1] = 0
        vertices[TIP_VERTICES, 2] = .945  # 5mm from the actual object front face
        faces = np.array([[0, 1, 2]], np.int32)
        joints = np.concatenate([vertices[:16], vertices[TIP_VERTICES]])[JOINT_ORDER]

        class TestMano:
            def __init__(self):
                self.faces = faces
            def requires_grad_(self, enabled):
                pass
            def __call__(self, global_orient, hand_pose, betas):
                v = torch.tensor(vertices)[None].expand(len(hand_pose), -1, -1) + hand_pose[:, :1, None] * .001
                return SimpleNamespace(vertices=v, joints=v[:, :16])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ann, _, poses = fixture()
            mask = np.zeros((48, 80), np.uint8)
            mask[15:33, 40:50] = 255
            cv2.imwrite(str(root / "mask.png"), mask)
            for frame in ann["frames"]:
                frame["objects"] = [{"object_id": "item", "mask_path": str(root / "mask.png")}]
            (root / "ann.json").write_text(json.dumps(ann))
            mesh = root / "mesh.ply"
            trimesh.creation.box(extents=[.1, .1, .1]).export(mesh)
            (root / "poses.json").write_text(json.dumps({"pose_rows": poses, "render_mesh_contract": {
                "required_mesh_path": str(mesh), "required_mesh_sha256": sha(mesh)}}))
            ids = np.array([42, 40, 43, 41])
            np.savez(root / "params.npz", frame_idx=ids, left_valid=np.ones(4, bool),
                     left_vertices_world_m=np.tile(vertices, (4, 1, 1)),
                     left_joints_world_m=np.tile(joints, (4, 1, 1)),
                     left_root_orient_axis_angle=np.zeros((4, 3)), left_hand_pose_axis_angle=np.zeros((4, 45)),
                     left_betas=np.zeros((4, 10)), left_trans_world_m=np.zeros((4, 3)), left_faces=faces)
            np.savez(root / "bridge.npz", frame_idx=ids, hand_side=np.array(["left"] * 4),
                     vertices_current_v18_world_from_hawor_projection_relift_m=np.tile(vertices, (4, 1, 1)))
            for side in ["LEFT", "RIGHT"]:
                (root / f"MANO_{side}.pkl").write_bytes(b"test fixture; no model")
            results = []
            real_tree = fitter.cKDTree
            for width in [960, 480]:
                samples = []
                def record_tree(points):
                    if np.shape(points) == (160, 2):
                        samples.append(np.asarray(points).copy())
                    return real_tree(points)
                out = root / str(width)
                argv = ["fit", "--annotations", str(root / "ann.json"), "--pose-report", str(root / "poses.json"),
                        "--mano-bridge", str(root / "bridge.npz"), "--hawor-params", str(root / "params.npz"),
                        "--mano-models", str(root), "--output-dir", str(out), "--device", "cpu",
                        "--iterations", "3", "--object-id", "item", "--render-width", str(width)]
                with (patch.object(sys, "argv", argv), patch.object(fitter, "cKDTree", side_effect=record_tree),
                      patch.object(fitter, "mano_models", return_value={s: TestMano() for s in ["left", "right"]}),
                      contextlib.redirect_stdout(io.StringIO())):
                    fitter.main()
                with np.load(out / "demo_state.npz") as archive:
                    state = dict(archive)
                report = json.loads((out / "fit_report.json").read_text())
                contact = json.loads((out / "contact_hypotheses.json").read_text())["weights"]
                results.append((state, report["trace"], np.asarray(contact), samples))
            full, half = results
            # Explicit legacy decisions: thumb20px -> off; finger6px -> on;
            # finger10px -> off. Unnormalized half-width gates violate these.
            np.testing.assert_allclose(full[2][:, 0], np.tile([0, .12, 0, 0, .12], (len(full[2]), 1)))
            np.testing.assert_array_equal(full[2], half[2])
            self.assertGreater(full[1][-1]["image"], 0)
            self.assertGreater(full[1][0]["boundary"], 0)
            self.assertGreater(full[1][0]["contact"], 0)
            for row_full, row_half in zip(full[1], half[1]):
                for term in ["total", "image", "boundary", "contact", "intrusion"]:
                    self.assertAlmostEqual(row_full[term], row_half[term], places=6)
            # Both observed contours and projected mesh boundary samples used
            # by closest-point matching must have identical positions/spacings.
            self.assertGreater(len(full[3]), 0)
            np.testing.assert_allclose(full[3], half[3], atol=1e-6)
            for key in ["object_delta", "object_rotation", "object_translation_world", "hand_articulation",
                        "hand_root", "hand_translation_camera", "hand_log_scale", "left_vertices_world"]:
                np.testing.assert_allclose(full[0][key], half[0][key], atol=1e-7)
            np.testing.assert_array_equal(full[0]["render_size_wh"], [960, 576])
            np.testing.assert_array_equal(half[0]["render_size_wh"], [480, 288])
            uv_full = project(full[0]["left_vertices_world"][0], full[0]["intrinsics_render"][0])
            uv_half = project(half[0]["left_vertices_world"][0], half[0]["intrinsics_render"][0])
            np.testing.assert_allclose(uv_half, (uv_full + .5) / 2 - .5, atol=1e-6)

    def test_short_rrd_logs_timestamp_camera_and_clears_missing_hand(self):
        import render_v20_visual_contact_demo as renderer
        vertices = np.array([[-.2, -.2, 0], [.2, -.2, 0], [0, .2, 0]], np.float32)
        # State storage is deliberately not chronological.
        z = {"frame_idx": np.array([42, 40, 41]), "time_s": [.2, 0, .1],
             "render_size_wh": [40, 24], "source_size_wh": [40, 24],
             "intrinsics_render": np.tile([20, 20, 19.5, 11.5], (3, 1)),
             "object_vertices_canonical": vertices, "object_faces": np.array([[0, 1, 2]]),
             "object_rotation": np.tile(np.eye(3), (3, 1, 1)),
             "object_translation_world": np.tile([0, 0, 2.], (3, 1)),
             "T_world_camera": np.tile(np.eye(4), (3, 1, 1)),
             "left_vertices_world": np.tile(vertices + [0, 0, 1], (3, 1, 1)),
             "left_faces": np.array([[0, 1, 2]]), "left_valid": [1, 1, 0]}
        positions = []

        class Video:
            def get(self, prop):
                return {renderer.cv2.CAP_PROP_FPS: 10, renderer.cv2.CAP_PROP_FRAME_COUNT: 3,
                        renderer.cv2.CAP_PROP_FRAME_WIDTH: 40, renderer.cv2.CAP_PROP_FRAME_HEIGHT: 24}[prop]
            def set(self, prop, value):
                positions.append(value)
            def read(self):
                return True, np.zeros((24, 40, 3), np.uint8)
            def release(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(root / "state.npz", **z)
            (root / "video").write_bytes(b"mock decoder input")
            argv = ["render", "--state", str(root / "state.npz"), "--source-video", str(root / "video"),
                    "--output-dir", str(root / "out")]
            process = lambda *args: SimpleNamespace(stdin=io.BytesIO(), wait=lambda: 0)
            with (patch.object(sys, "argv", argv), patch.object(renderer.cv2, "VideoCapture", return_value=Video()),
                  patch.object(renderer, "encode_process", side_effect=process),
                  patch.object(renderer.rr, "log", wraps=renderer.rr.log) as log,
                  patch.object(renderer.rr, "set_time", wraps=renderer.rr.set_time) as set_time,
                  contextlib.redirect_stdout(io.StringIO())):
                renderer.main()
            self.assertEqual(positions, [0, 1, 2])
            self.assertGreater((root / "out/interaction_demo.rrd").stat().st_size, 0)
            timestamp_calls = [c.kwargs["duration"] for c in set_time.call_args_list if c.args[0] == "time_s"]
            np.testing.assert_allclose(timestamp_calls, [0, .1, .2])
            left_logs = [c.args[1] for c in log.call_args_list if c.args[0] == "/world/hands/left"]
            self.assertIsInstance(left_logs[0], renderer.rr.Mesh3D)
            self.assertIsInstance(left_logs[1], renderer.rr.Clear)
            self.assertIsInstance(left_logs[2], renderer.rr.Mesh3D)
            right_logs = [c.args[1] for c in log.call_args_list if c.args[0] == "/world/hands/right"]
            self.assertTrue(all(isinstance(c, renderer.rr.Clear) for c in right_logs))
            cameras = [c.args[1] for c in log.call_args_list if c.args[0] == "/world/camera"]
            self.assertEqual(len(cameras), 3)
            self.assertTrue(all(isinstance(c, renderer.rr.Transform3D) for c in cameras))

    def test_runner_forwards_dynamic_options(self):
        import run_v20_visual_contact_demo as runner
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "video").touch(); (root / "state").touch()
            args = runner.parser().parse_args(["--mode", "replay", "--source-video", str(root / "video"),
                "--state", str(root / "state"), "--output-dir", str(root / "out"), "--render-width", "80", "--frames", "42,40"])
            command = runner.build_plan(args)[0][1]
            self.assertEqual(command[command.index("--render-width") + 1], "80")
            self.assertEqual(command[command.index("--frames") + 1], "42,40")
            self.assertEqual(command[command.index("--detail-yaw-deg") + 1], "20.0")


if __name__ == "__main__":
    unittest.main()
