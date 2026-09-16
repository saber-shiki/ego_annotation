from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from data.build_index import build_index
from data.bundle import EpisodeBundle, file_sha256, prepared_manifest, write_manifest
from data.dataset import EpisodeDataset


class SyntheticMano:
    """Explicit synthetic numeric fixture; never used by production assembly."""
    def __init__(self, side: str):
        import torch
        self.side = side
        self.template = torch.zeros((778, 3), dtype=torch.float32)
        self.template[:, 0] = torch.linspace(-0.04, 0.04, 778)
        self.template[:, 1] = (0.01 if side == "left" else -0.01)
        self.template[:, 2] = 1.0 + torch.linspace(0.0, 0.08, 778)
        self.faces = np.column_stack([np.arange(0, 776), np.arange(1, 777), np.arange(2, 778)]).astype(np.int64)

    def __call__(self, *, global_orient, hand_pose, betas, transl=None, **kwargs):
        import torch
        root = global_orient[:, 0] if global_orient.ndim == 4 else torch.linalg.matrix_exp(_skew(global_orient))
        if hand_pose.ndim == 4:
            articulation = hand_pose[:, :, 0, 1].sum(dim=1)
        else:
            articulation = hand_pose.reshape(len(hand_pose), -1).sum(dim=1)
        vertices = self.template.to(root.device)[None].expand(len(root), -1, -1)
        vertices = vertices @ root.transpose(1, 2)
        vertices = vertices.clone()
        vertices[:, :, 1] += articulation[:, None] * torch.linspace(0.0, 0.002, 778, device=root.device)[None]
        if transl is not None:
            vertices = vertices + transl[:, None]
        return SimpleNamespace(vertices=vertices, joints=vertices[:, :21])


def _skew(value):
    import torch
    x, y, z = value.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack([torch.stack([zero, -z, y], -1), torch.stack([z, zero, -x], -1),
                        torch.stack([-y, x, zero], -1)], -2)


def make_synthetic_bundle(root: Path, bundle_id: str = "synthetic-a", source_video_id: str = "video-a",
                          right_valid: tuple[bool, ...] | None = None, frame_count: int = 3) -> tuple[Path, dict]:
    import trimesh
    root.mkdir(parents=True, exist_ok=True)
    video = root / f"{bundle_id}.mp4"; video.write_bytes(b"synthetic-not-a-real-video")
    code = root / "code.txt"; code.write_text("synthetic code")
    config = root / "config.json"; config.write_text("{}")
    if frame_count < 2:
        raise ValueError("synthetic fixture needs at least two frames")
    if right_valid is None:
        right_valid = tuple((i != 1 for i in range(frame_count)))
    if len(right_valid) != frame_count:
        raise ValueError("right_valid must cover synthetic frame_count")
    frame_ids = np.arange(frame_count, dtype=np.int32)
    models = {side: SyntheticMano(side) for side in ("left", "right")}
    arrays = {"frame_idx": frame_ids}
    for side in ("left", "right"):
        roots = np.zeros((frame_count, 3), np.float32); poses = np.zeros((frame_count, 45), np.float32)
        betas = np.zeros((frame_count, 10), np.float32); trans = np.zeros((frame_count, 3), np.float32)
        vertices = []; joints = []
        for i in range(frame_count):
            output = models[side](global_orient=np_to_torch(roots[i:i+1]),
                                  hand_pose=np_to_torch(poses[i:i+1]), betas=np_to_torch(betas[i:i+1]),
                                  transl=np_to_torch(trans[i:i+1]))
            vertices.append(output.vertices[0].numpy()); joints.append(output.joints[0].numpy())
        arrays.update({f"{side}_root_orient_axis_angle": roots, f"{side}_hand_pose_axis_angle": poses,
                       f"{side}_betas": betas, f"{side}_trans_world_m": trans,
                       f"{side}_vertices_world_m": np.asarray(vertices), f"{side}_joints_world_m": np.asarray(joints),
                       f"{side}_valid": np.ones(frame_count, np.uint8) if side == "left" else np.asarray(right_valid, np.uint8),
                       f"{side}_faces": models[side].faces})
    mano = root / f"{bundle_id}_mano.npz"; np.savez(mano, **arrays)
    canonical = np.asarray([[-.04, -.04, -.04], [.04, -.04, -.04], [0, .04, -.04], [0, 0, .04]], np.float32)
    faces = np.asarray([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], np.int64)
    mesh = root / f"{bundle_id}_mesh.ply"; trimesh.Trimesh(canonical, faces, process=False).export(mesh)
    frames = []
    for i in range(frame_count):
        object_t = np.asarray([0.0 + 0.01 * i, 0.0, 1.04])
        observed = canonical + object_t
        hand_rows = [{"hand_side": side, "metric_mano_state": {
            "current_v18_camera_intrinsics_fx_fy_cx_cy": [60, 60, 32, 32],
            "camera_contract_alignment": {
                "active_contract_reinference_required": False,
                "source_hawor_state_intrinsics_match_active_contract": True,
            },
        }} for side in ("left", "right")]
        frames.append({"frame_idx": i, "time_s": i / 30.0, "source_width": 64, "source_height": 64,
            "camera": {"T_world_camera": np.eye(4).tolist(), "intrinsics_fx_fy_cx_cy": [60, 60, 32, 32]},
            "hands": hand_rows, "objects": [{"object_id": "object:synthetic", "rigid_pose_observation_eligible": True,
                "reconstructed_geometry_pose": {"rotation_world_from_canonical_matrix": np.eye(3).tolist(),
                    "translation_world_m": object_t.tolist()},
                "visible_geometry_candidate": {"world_vertices_sample_m": observed.tolist()}}]})
    annotations = root / f"{bundle_id}_annotations.json"; annotations.write_text(json.dumps({"frames": frames}))
    factor_rows = []; ownership_rows = []
    for frame in range(frame_count):
        for side in ("left", "right"):
            factor_rows.append({"factor_family": "contact_patch", "frame_idx": frame, "hand_side": side,
                "state": "active_contact_patch", "contact_patch_prior_probability": 0.7, "weight": 2.0,
                "contact_patch_band_m": 0.2, "contact_patch_target_margin_m": 0.002,
                "contact_patch_support_uncertainty_m": 0.003, "contact_anchor_residual_allowed": False})
    factors = root / f"{bundle_id}_factors.json"; factors.write_text(json.dumps({"factor_rows": factor_rows, "ownership_rows": ownership_rows}))
    manifest = prepared_manifest(bundle_id=bundle_id, clip_id=bundle_id, source_video_id=source_video_id,
        source_video=video, annotations=annotations, mano_archive=mano, object_mesh=mesh,
        contact_factors=factors, object_id="object:synthetic", frame_range=(0, frame_count - 1),
        code_identity=(str(code), file_sha256(code)), config_identity=(str(config), file_sha256(config)))
    path = write_manifest(root / f"{bundle_id}.manifest.json", manifest)
    return path, models


def np_to_torch(value):
    import torch
    return torch.as_tensor(value, dtype=torch.float32)


class DataLayerTest(unittest.TestCase):
    def test_index_split_reload_and_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p0, _ = make_synthetic_bundle(root / "a", "a", "video-a")
            p1, _ = make_synthetic_bundle(root / "b", "b", "video-b")
            index = build_index([root])
            self.assertEqual([row["bundle_id"] for row in index["episodes"]], ["a", "b"])
            train = EpisodeDataset(index, split="train", seed=4, test_fraction=0.5)
            test = EpisodeDataset(index, split="test", seed=4, test_fraction=0.5)
            self.assertFalse(train.source_video_ids & test.source_video_ids)
            self.assertEqual(train.source_video_ids | test.source_video_ids, {"video-a", "video-b"})
            bundle = EpisodeBundle.open(p0)
            first = bundle.load(); second = bundle.load()
            self.assertIs(first, second)
            self.assertFalse(first["frame_idx"].flags.writeable)
            with self.assertRaises(ValueError):
                first["frame_idx"][0] = 99
            reopened = EpisodeBundle.open(p0).load()
            np.testing.assert_array_equal(reopened["frame_idx"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
