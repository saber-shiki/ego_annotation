from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import render


class PolicyRenderEntryTests(unittest.TestCase):
    def test_policy_render_consumes_explicit_full_state_without_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame.jpg"
            cv2.imwrite(str(frame), np.zeros((8, 8, 3), dtype=np.uint8))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"frames": [{"frame_idx": 0, "raw_frame_path": str(frame)}]}))
            state = root / "policy_candidate.npz"
            vertices = np.zeros((1, 778, 3), dtype=np.float64)
            vertices[0, :, 2] = 1.0
            vertices[0, 0, 0] = 0.123
            arrays = {
                "frame_idx": np.asarray([0], dtype=np.int64),
                "time_s": np.asarray([0.0]),
                "T_world_camera": np.eye(4)[None],
                "intrinsics_render": np.asarray([[100.0, 100.0, 3.5, 3.5]]),
                "render_size_wh": np.asarray([8, 8], dtype=np.int64),
                "object_vertices_canonical": np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
                "object_faces": np.asarray([[0, 1, 2]], dtype=np.int64),
                "object_rotation": np.eye(3)[None],
                "object_translation_world": np.asarray([[0.0, 0.0, 1.0]]),
                "object_valid": np.asarray([True]),
            }
            for side in ("left", "right"):
                arrays[f"{side}_valid"] = np.asarray([True])
                arrays[f"{side}_vertices_world"] = vertices.copy()
                arrays[f"{side}_joints_world"] = np.zeros((1, 21, 3), dtype=np.float64)
                arrays[f"{side}_faces"] = np.asarray([[0, 1, 2]], dtype=np.int64)
            np.savez(state, **arrays)

            consumed = []

            def mesh_overlay(candidate, position, image):
                consumed.append(float(candidate["left_vertices_world"][position, 0, 0]))
                return image

            class Writer:
                def isOpened(self): return True
                def write(self, image): pass
                def release(self): pass

            def write_review(path, image):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"review")
                return True

            output = root / "policy.mp4"
            with patch.object(render, "mesh_overlay", side_effect=mesh_overlay), \
                 patch.object(cv2, "VideoWriter", return_value=Writer()), \
                 patch.object(cv2, "imwrite", side_effect=write_review):
                result = render.main(["policy", "--state", str(state), "--raw-manifest", str(manifest), "--output", str(output)])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["frame_idx"], 0)
            self.assertEqual(consumed, [0.123])


if __name__ == "__main__":
    unittest.main()
