#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = load_module(
    "depth_provider_pair_evaluator",
    ROOT / "scripts/evaluate_v19_hot3d_depth_provider_pair_p09.py",
)


class DepthProviderPairEvaluatorTest(unittest.TestCase):
    def test_explicit_depths_must_match_frozen_pair_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="depth_pair_evaluator_binding_") as temp:
            root = Path(temp)
            paths = {name: root / f"{name}.npz" for name in evaluator.BRANCHES}
            providers = {
                evaluator.BRANCHES[0]: "unidepth",
                evaluator.BRANCHES[1]: "depth_anything_3",
            }
            branches = {}
            for name, path in paths.items():
                path.write_bytes(name.encode())
                branches[name] = {
                    "path": str(path),
                    "sha256": evaluator.sha256_file(path),
                    "provider": providers[name],
                    "depth_array_sha256": f"depth-{name}",
                    "confidence_array_sha256": f"confidence-{name}",
                }
            snapshot = {"branches": branches}
            result = evaluator.bind_depth_paths_to_pair_snapshot(paths, snapshot)
            self.assertEqual(result[evaluator.BRANCHES[1]]["provider"], "depth_anything_3")
            paths[evaluator.BRANCHES[1]].write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "not the frozen paired archive"):
                evaluator.bind_depth_paths_to_pair_snapshot(paths, snapshot)

    def test_p09_world_camera_roundtrip_passes_and_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="depth_pair_evaluator_p09_") as temp:
            path = Path(temp) / "annotations.json"
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = [0.1, -0.2, 0.3]
            camera = np.asarray([[1.0, 2.0, 3.0], [0.2, 0.4, 0.8]])
            world = evaluator.transform_points(transform, camera)
            candidate = {
                "depth_provider": "unidepth",
                "camera_vertices_sample_m": camera.tolist(),
                "world_vertices_sample_m": world.tolist(),
            }
            payload = {
                "frames": [{
                    "frame_idx": 0,
                    "camera": {"T_world_camera_metric": transform.tolist()},
                    "objects": [{"object_id": "object", "visible_geometry_candidate": candidate}],
                }]
            }
            path.write_text(json.dumps(payload))
            _rows, closure = evaluator.load_p09_points(path, "object", "unidepth")
            self.assertEqual(closure["status"], "passed")
            self.assertLess(closure["error_m"]["max"], 1.0e-12)
            candidate["world_vertices_sample_m"][0][0] += 0.01
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "round-trip failed"):
                evaluator.load_p09_points(path, "object", "unidepth")

    def test_gt_rendered_depth_pose_cad_closure(self) -> None:
        mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        mesh.apply_translation([0.0, 0.0, 1.0])
        query = trimesh.proximity.ProximityQuery(mesh)
        depth = np.zeros((2, 3, 3), dtype=np.float32)
        mask = np.zeros_like(depth, dtype=bool)
        depth[:, 1, 1] = 0.9
        mask[:, 1, 1] = True
        gt_depth = {
            "depth_m": depth,
            "valid_mask": mask,
            "K": np.asarray([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]]),
            "frame_idx": np.asarray([0, 1]),
        }
        gt = {idx: {"T_camera_object": np.eye(4)} for idx in (0, 1)}
        result = evaluator.gt_depth_cad_closure(gt_depth, gt, query, 4)
        self.assertEqual(result["status"], "passed")
        self.assertLess(result["unsigned_distance_to_cad_m"]["max"], 1.0e-6)


if __name__ == "__main__":
    unittest.main()
