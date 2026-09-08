"""Genuine short MP4 encoding/decoding from the layered synthetic package."""

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_v19_annotation_layers as fixtures
from assemble_v19_annotation_layers import assemble
from render_v19_annotation_layers_video import render, scene_for_layer
from render_v20_visual_contact_demo import encode_process, shade
from v19_video_timing import read_cfr_video_timing
from v19_annotation_layers import load_package, sha256


class AnnotationVideoTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AnnotationLayersTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.package = assemble(self.fixture.args)
        self.video = self.fixture.root / "source.mp4"
        process = encode_process(self.video, (640, 480), 30)
        for i in range(3):
            image = np.full((480, 640, 3), [40 + 10 * i, 80, 120], np.uint8)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        self.assertEqual(process.wait(), 0)

    def test_actual_video_from_package_with_missing_object_and_hand(self):
        before = sha256(self.package)
        output = self.fixture.root / "render"
        target = render(self.package, self.video, output, 160, sha256(self.video))
        report = json.loads((output / "render_report.json").read_text())
        self.assertEqual(report["frame_count"], 3)
        self.assertEqual(report["duration_s"], 0.1)
        self.assertEqual(report["layers"], ["prediction", "demo"])
        self.assertEqual(report["timing"]["source"]["fps_ratio"], "30")
        self.assertEqual(report["timing"]["encoded_output"]["max_cfr_grid_error_s"], 0)
        np.testing.assert_allclose(
            [r["output_video_pts_s"] for r in report["per_frame"]],
            np.arange(3) / 30,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            [r["source_video_pts_s"] for r in report["per_frame"]],
            np.arange(3) / 30,
            atol=1e-9,
        )
        middle = report["per_frame"][1]["layers"]
        self.assertFalse(middle["prediction"]["object_valid"])
        self.assertFalse(middle["prediction"]["left_valid"])
        self.assertEqual(middle["prediction"]["triangle_count"], 0)
        self.assertGreater(middle["demo"]["pixels_by_object_left_right"][0], 0)
        self.assertEqual(
            report["per_frame"][0]["layers"]["prediction"]["triangle_count"], 3
        )
        self.assertEqual(before, sha256(self.package))
        cap = cv2.VideoCapture(str(target))
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame)
        cap.release()
        self.assertEqual(len(decoded), 3)
        self.assertEqual(decoded[0].shape, (120, 480, 3))
        # Different meshes/poses cause actual pixels, not just a header change.
        original = decoded[0][30:, :160].astype(float)
        prediction = decoded[0][30:, 160:320].astype(float)
        demo = decoded[0][30:, 320:].astype(float)
        self.assertGreater(np.abs(original - prediction).sum(), 0)
        self.assertGreater(np.abs(prediction - demo).sum(), 0)

    def test_wrong_original_video_hash_rejected(self):
        out = self.fixture.root / "bad"
        with self.assertRaisesRegex(ValueError, "video hash mismatch"):
            render(self.package, self.video, out, 160, "0" * 64)
        self.assertFalse(out.exists())

    def test_right_only_geometry_identity_with_missing_object(self):
        _, layers = load_package(self.package)
        state = layers["prediction"]
        state["object_valid"][0] = False
        for name in list(state):
            if name.startswith("left_"):
                state[name.replace("left_", "right_", 1)] = state[name].copy()
        state["left_valid"][:] = False
        # Two non-coplanar connected triangles make smooth and primitive normals
        # observably different. A single planar triangle cannot catch the bug.
        state["right_vertices_world"][0, :4] = [
            [-0.08, -0.06, 1.0],
            [0.08, -0.06, 1.0],
            [-0.08, 0.08, 1.0],
            [0.08, 0.08, 1.08],
        ]
        state["right_faces"] = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        scene, faces, normals, labels = scene_for_layer(state, 0)
        self.assertEqual(labels, [2])
        np.testing.assert_array_equal(faces[0], state["right_faces"])
        self.assertEqual(normals[0].shape, (778, 3))
        K = np.array([100.0, 102.5, 79.625, 59.625])
        without_object = shade(scene, K, (160, 120), faces, normals, labels)
        self.assertGreater(np.count_nonzero(without_object[2] == 2), 0)
        # Add an off-screen object: hand geometry, illumination and visibility
        # are identical; only its insertion index changes from zero to one.
        state["object_valid"][0] = True
        state["object_translation_world"][0] = [10.0, 0.0, 1.0]
        scene, faces, normals, labels = scene_for_layer(state, 0)
        with_object = shade(scene, K, (160, 120), faces, normals, labels)
        mask = without_object[1]
        np.testing.assert_array_equal(mask, with_object[1])
        np.testing.assert_array_equal(without_object[2], with_object[2])
        np.testing.assert_array_equal(without_object[0][mask], with_object[0][mask])
        # No-label legacy scenes retain index-zero-as-object behavior.
        legacy = shade(scene, K, (160, 120), faces, normals)
        np.testing.assert_array_equal(mask, legacy[1])
        identity_labels = shade(scene, K, (160, 120), faces, normals, [0, 1])
        np.testing.assert_array_equal(legacy[0], identity_labels[0])

    def package_with_times(self, times, name):
        for frame, time in zip(self.fixture.annotations["frames"], times):
            frame["time_s"] = float(time)
        self.fixture.dump("annotations.json", self.fixture.annotations)
        self.fixture.demo["time_s"] = np.asarray(times)
        np.savez(self.fixture.args.demo_state, **self.fixture.demo)
        self.fixture.args.output_dir = self.fixture.root / name
        return assemble(self.fixture.args)

    def test_nonuniform_package_inside_old_half_frame_tolerance_is_rejected(self):
        times = [0.0, 0.045, 2 / 30]
        self.assertLess(abs(times[1] - 1 / 30), 0.5 / 30)
        package = self.package_with_times(times, "nonuniform")
        out = self.fixture.root / "bad_timing"
        with mock.patch("render_v19_annotation_layers_video.encode_process") as encode:
            with self.assertRaisesRegex(ValueError, "package constant-rate timeline"):
                render(package, self.video, out, 160, sha256(self.video))
            encode.assert_not_called()
        self.assertFalse(out.exists())

    def test_actual_source_pts_mismatch_rejected_despite_matching_aggregate_fps(self):
        vfr = self.fixture.root / "vfr.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(self.video),
                "-vf",
                r"settb=1/90000,setpts=if(eq(N\,1)\,4050\,N*3000)",
                "-vsync",
                "0",
                "-enc_time_base",
                "1:90000",
                "-c:v",
                "libx264",
                "-bf",
                "0",
                "-video_track_timescale",
                "90000",
                str(vfr),
            ],
            check=True,
            capture_output=True,
        )
        cap = cv2.VideoCapture(str(vfr))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
        self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 30, places=5)
        cap.release()
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_frames",
                    "-show_entries",
                    "frame=pts_time,pkt_pts_time",
                    "-of",
                    "json",
                    str(vfr),
                ],
                text=True,
            )
        )
        np.testing.assert_allclose(
            [float(f.get("pts_time", f.get("pkt_pts_time"))) for f in probe["frames"]],
            [0.0, 0.045, 2 / 30],
            atol=1e-6,
        )
        out = self.fixture.root / "bad_source_pts"
        with mock.patch("render_v19_annotation_layers_video.encode_process") as encode:
            with self.assertRaisesRegex(
                ValueError, "source presentation timestamps are not CFR"
            ):
                render(self.package, vfr, out, 160, sha256(vfr))
            encode.assert_not_called()
        self.assertFalse(out.exists())

    def test_fractional_rate_and_package_time_origin_are_preserved_explicitly(self):
        rate = Fraction(30000, 1001)
        source = self.fixture.root / "ntsc.mp4"
        process = encode_process(source, (640, 480), rate)
        for i in range(3):
            process.stdin.write(np.full((480, 640, 3), 50 + i, np.uint8).tobytes())
        process.stdin.close()
        self.assertEqual(process.wait(), 0)
        times = np.arange(3) / float(rate) + 12.5
        package = self.package_with_times(times, "fractional_rate")
        out = self.fixture.root / "ntsc_output"
        target = render(package, source, out, 160, sha256(source))
        self.assertEqual(read_cfr_video_timing(target).frame_rate, rate)
        report = json.loads((out / "render_report.json").read_text())
        self.assertEqual(report["timing"]["package_time_origin_s"], 12.5)
        self.assertEqual(report["timing"]["encoded_output"]["first_pts_s"], 0)
        np.testing.assert_allclose(
            [r["time_s"] for r in report["per_frame"]], times, atol=1e-9
        )
        np.testing.assert_allclose(
            [r["output_video_pts_s"] for r in report["per_frame"]],
            times - times[0],
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
