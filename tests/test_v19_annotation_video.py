"""Genuine short MP4 encoding/decoding from the layered synthetic package."""
import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import test_v19_annotation_layers as fixtures
from assemble_v19_annotation_layers import assemble
from render_v19_annotation_layers_video import render, scene_for_layer
from render_v20_visual_contact_demo import encode_process
from v19_annotation_layers import load_package, sha256


class AnnotationVideoTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AnnotationLayersTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.package = assemble(self.fixture.args)
        self.video = self.fixture.root / 'source.mp4'
        process = encode_process(self.video, (640, 480), 30)
        for i in range(3):
            image = np.full((480, 640, 3), [40 + 10*i, 80, 120], np.uint8)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        self.assertEqual(process.wait(), 0)

    def test_actual_video_from_package_with_missing_object_and_hand(self):
        before = sha256(self.package)
        output = self.fixture.root / 'render'
        target = render(self.package, self.video, output, 160, sha256(self.video))
        report = json.loads((output / 'render_report.json').read_text())
        self.assertEqual(report['frame_count'], 3)
        self.assertEqual(report['duration_s'], .1)
        self.assertEqual(report['layers'], ['prediction', 'demo'])
        middle = report['per_frame'][1]['layers']
        self.assertFalse(middle['prediction']['object_valid'])
        self.assertFalse(middle['prediction']['left_valid'])
        self.assertEqual(middle['prediction']['triangle_count'], 0)
        self.assertGreater(middle['demo']['pixels_by_object_left_right'][0], 0)
        self.assertEqual(report['per_frame'][0]['layers']['prediction']['triangle_count'], 3)
        self.assertEqual(before, sha256(self.package))
        cap = cv2.VideoCapture(str(target))
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:break
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
        out = self.fixture.root / 'bad'
        with self.assertRaisesRegex(ValueError, 'video hash mismatch'):
            render(self.package, self.video, out, 160, '0'*64)
        self.assertFalse(out.exists())

    def test_right_only_geometry_identity_with_missing_object(self):
        _, layers = load_package(self.package)
        state = layers['prediction']
        state['object_valid'][0] = False
        for name in list(state):
            if name.startswith('left_'):
                state[name.replace('left_', 'right_', 1)] = state[name].copy()
        state['left_valid'][:] = False
        _, faces, normals, labels = scene_for_layer(state, 0)
        self.assertEqual(labels, [2])
        np.testing.assert_array_equal(faces[0], state['right_faces'])
        self.assertEqual(normals[0].shape, (778, 3))


if __name__ == '__main__':unittest.main()
