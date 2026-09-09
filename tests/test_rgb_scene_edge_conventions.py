"""Real RGB-D lifting/OpenCV PnP → pose-graph world-motion convention tests.

Only feature correspondence matching is a deterministic synthetic double. The
production source eligibility, lifting, PnP/refinement and edge residual run.
"""
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_rgb_scene_edges as builder
from rgb_scene_pose_kernels import RGBEdge, rgb_edge_residual


def fixture(moving=True):
    rng = np.random.default_rng(27)
    uv = np.array([(x, y) for y in range(170, 311, 28) for x in range(220, 421, 40)], float)
    wh = (640, 480)
    K4 = np.array([420., 410., 319.5, 239.5])
    depth = np.ones(wh[::-1], float)
    depth[uv[:, 1].astype(int), uv[:, 0].astype(int)] = rng.uniform(.8, 1.3, len(uv))
    Z = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    source_camera_points = np.c_[(uv[:, 0] - K4[2]) * Z / K4[0],
                                (uv[:, 1] - K4[3]) * Z / K4[1], Z]
    Cs, Ct = np.eye(4), np.eye(4)
    Cs[:3, :3] = Rotation.from_rotvec([.02, .04, -.03]).as_matrix()
    Cs[:3, 3] = [.1, -.1, .03]
    Ct[:3, :3] = Rotation.from_rotvec([-.03, .02, .04]).as_matrix()
    Ct[:3, 3] = [.05, -.04, .02]
    source_world = source_camera_points @ Cs[:3, :3].T + Cs[:3, 3]
    Rs = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
    ts = np.array([.1, .2, .8])
    Rt = Rotation.from_rotvec([.4, -.35, .2]).as_matrix() @ Rs if moving else Rs.copy()
    tt = ts + [.005, -.01, .03] if moving else ts.copy()
    # Independent material-point construction, not the builder's edge formula.
    canonical = (source_world - ts) @ Rs
    target_world = canonical @ Rt.T + tt
    target_camera_points = (target_world - Ct[:3, 3]) @ Ct[:3, :3]
    target_uv = target_camera_points[:, :2] / target_camera_points[:, 2:] * K4[:2] + K4[2:]
    assert (target_camera_points[:, 2] > 0).all()
    assert (target_uv > 10).all() and (target_uv < np.array(wh) - 10).all()
    geom = {
        "frame_idx": 7, "object_id": "tool", "rigid_pose_observation_eligible": True,
        "world_vertices_sample_m": source_world.tolist(),
        "camera_vertices_sample_m": source_camera_points.tolist(),
        "first_surface_depth_ownership": {
            "enabled": True, "fail_closed": False, "retained_fraction": 1., "failure_reasons": []},
    }
    obj = {"object_id": "tool", "rigid_pose_observation_eligible": True,
           "visible_geometry_candidate": geom}
    frames = {
        7: {"frame_idx": 7, "camera": {"T_world_camera_metric": Cs.tolist(),
            "intrinsics_fx_fy_cx_cy": K4.tolist()}, "objects": [obj]},
        31: {"frame_idx": 31, "camera": {"T_world_camera_metric": Ct.tolist(),
             "intrinsics_fx_fy_cx_cy": K4.tolist()}, "objects": [{"object_id": "tool"}]},
    }
    pairs = np.c_[np.arange(len(uv)), np.arange(len(uv))]
    def matches(_):
        return {"matches": pairs, "scores": np.ones(len(pairs))}
    args = SimpleNamespace(object_id="tool", min_confidence=.05, min_matches=20,
                           pnp_iterations=1000, reprojection_threshold_px=1.)
    kwargs = dict(source_idx=7, target_idx=31,
        features={7: {"keypoints": torch.tensor(uv[None])},
                  31: {"keypoints": torch.tensor(target_uv[None])}},
        masks={i: np.full(wh[::-1], 255, np.uint8) for i in frames}, frames=frames,
        depths=np.stack([depth, np.full_like(depth, np.nan)]),
        confidences=np.ones((2, *wh[::-1])), depth_pos={7: 0, 31: 1},
        depth_intrinsics=np.stack([K4, K4]), source_wh=wh, matcher=matches,
        device=torch.device("cpu"), args=args,
        initial_pose_by_frame={7: (Rs, ts), 31: (Rt, tt)},
        source_support_by_frame={7: np.ones(wh[::-1], bool)})
    return kwargs, source_world, target_world


def produce(kwargs):
    utils = ModuleType("lightglue.utils")
    utils.rbd = lambda value: value
    cv2.setRNGSeed(27)
    with patch.dict(sys.modules, {"lightglue.utils": utils}):
        return builder.feature_pair(**kwargs)


class ProductionRGBEdgeConvention(unittest.TestCase):
    def check_motion(self, moving):
        kwargs, source, target = fixture(moving)
        poses = kwargs["initial_pose_by_frame"]
        before = copy.deepcopy(poses)
        row, motion, evidence = produce(kwargs)
        self.assertEqual(row["status"], "accepted")
        Rs, ts = poses[7]
        Rt, tt = poses[31]
        expected_R = Rt @ Rs.T
        expected_t = tt - expected_R @ ts
        np.testing.assert_allclose(motion[0], expected_R, atol=1e-7, rtol=0)
        np.testing.assert_allclose(motion[1], expected_t, atol=1e-7, rtol=0)
        np.testing.assert_allclose(source @ motion[0].T + motion[1], target, atol=1e-7, rtol=0)
        rr, tr = rgb_edge_residual(RGBEdge(7, 31, *motion, 1.), poses)
        np.testing.assert_allclose(rr, 0, atol=1e-7, rtol=0)
        np.testing.assert_allclose(tr, 0, atol=1e-7, rtol=0)
        np.testing.assert_allclose(evidence["canonical_points"] @ Rs.T + ts, source, atol=1e-10, rtol=0)
        for i in poses:
            for a, b in zip(poses[i], before[i]):
                np.testing.assert_array_equal(a, b)
        return kwargs, motion, evidence

    def test_moving_object_nonidentity_poses_and_cameras(self):
        kwargs, motion, _ = self.check_motion(True)
        Rs, _ = kwargs["initial_pose_by_frame"][7]
        Rt, _ = kwargs["initial_pose_by_frame"][31]
        self.assertGreater(Rotation.from_matrix(motion[0]).magnitude(), .5)
        # Converting this world motion into the common canonical gauge would
        # erase the measured action, not fix the pose graph's convention.
        double_converted = Rt.T @ motion[0] @ Rs
        np.testing.assert_allclose(double_converted, np.eye(3), atol=1e-7)
        self.assertGreater(np.linalg.norm(double_converted - motion[0]), .5)

    def test_static_object_nonidentity_pose_with_moving_camera(self):
        _, motion, _ = self.check_motion(False)
        np.testing.assert_allclose(motion[0], np.eye(3), atol=1e-7, rtol=0)
        np.testing.assert_allclose(motion[1], 0, atol=1e-7, rtol=0)

    def test_arbitrary_canonical_gauge_does_not_change_emitted_world_motion(self):
        kwargs, motion, _ = self.check_motion(True)
        Q = Rotation.from_rotvec([.8, .2, -.4]).as_matrix()
        kwargs["initial_pose_by_frame"] = {
            i: (R @ Q, t.copy()) for i, (R, t) in kwargs["initial_pose_by_frame"].items()}
        row, changed, _ = produce(kwargs)
        self.assertEqual(row["status"], "accepted")
        np.testing.assert_allclose(changed[0], motion[0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(changed[1], motion[1], atol=1e-7, rtol=0)
        rr, tr = rgb_edge_residual(RGBEdge(7, 31, *changed, 1.), kwargs["initial_pose_by_frame"])
        np.testing.assert_allclose(rr, 0, atol=1e-7, rtol=0)
        np.testing.assert_allclose(tr, 0, atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
