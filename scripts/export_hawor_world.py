#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def run(args: argparse.Namespace) -> dict:
    hawor_root = args.hawor_root.resolve()
    sys.path.insert(0, str(hawor_root))
    os.chdir(hawor_root)

    from demo import hawor_infiller  # type: ignore
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left  # type: ignore
    from lib.eval_utils.custom_utils import load_slam_cam  # type: ignore
    from scripts.scripts_test_video.detect_track_video import detect_track_video  # type: ignore
    from scripts.scripts_test_video.hawor_slam import hawor_slam  # type: ignore
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation  # type: ignore

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    slam_path = Path(seq_folder) / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    if not slam_path.exists():
        hawor_slam(args, start_idx, end_idx)
    if not slam_path.exists():
        raise RuntimeError(f"HaWoR SLAM output missing: {slam_path}")

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
    _, _, R_c2w, t_c2w = load_slam_cam(str(slam_path))

    faces = np.asarray(get_mano_faces(), dtype=np.int32)
    hands = {}
    hand_to_idx = {"left": 0, "right": 1}
    for side, idx in hand_to_idx.items():
        trans = pred_trans[idx : idx + 1]
        rot = pred_rot[idx : idx + 1]
        pose = pred_hand_pose[idx : idx + 1]
        betas = pred_betas[idx : idx + 1]
        if side == "left":
            mano = run_mano_left(trans, rot, pose, betas=betas)
            hand_faces = faces[:, [0, 2, 1]]
        else:
            mano = run_mano(trans, rot, pose, betas=betas)
            hand_faces = faces
        hands[side] = {
            "vertices_world_m": as_numpy(mano["vertices"][0]).astype(np.float32),
            "joints_world_m": as_numpy(mano["joints"][0]).astype(np.float32),
            "trans_world_m": as_numpy(trans[0]).astype(np.float32),
            "root_orient_axis_angle": as_numpy(rot[0]).astype(np.float32),
            "hand_pose_axis_angle": as_numpy(pose[0]).astype(np.float32),
            "betas": as_numpy(betas[0]).astype(np.float32),
            "valid": as_numpy(pred_valid[idx]).astype(np.uint8),
            "faces": hand_faces.astype(np.int32),
        }

    frame_idx = np.arange(len(imgfiles), dtype=np.int32)
    out_npz = args.output_dir / "hawor_world_hands.npz"
    np.savez_compressed(
        out_npz,
        frame_idx=frame_idx,
        R_c2w=as_numpy(R_c2w).astype(np.float32),
        t_c2w=as_numpy(t_c2w).astype(np.float32),
        left_vertices_world_m=hands["left"]["vertices_world_m"],
        left_joints_world_m=hands["left"]["joints_world_m"],
        left_trans_world_m=hands["left"]["trans_world_m"],
        left_root_orient_axis_angle=hands["left"]["root_orient_axis_angle"],
        left_hand_pose_axis_angle=hands["left"]["hand_pose_axis_angle"],
        left_betas=hands["left"]["betas"],
        left_valid=hands["left"]["valid"],
        left_faces=hands["left"]["faces"],
        right_vertices_world_m=hands["right"]["vertices_world_m"],
        right_joints_world_m=hands["right"]["joints_world_m"],
        right_trans_world_m=hands["right"]["trans_world_m"],
        right_root_orient_axis_angle=hands["right"]["root_orient_axis_angle"],
        right_hand_pose_axis_angle=hands["right"]["hand_pose_axis_angle"],
        right_betas=hands["right"]["betas"],
        right_valid=hands["right"]["valid"],
        right_faces=hands["right"]["faces"],
        img_focal=np.asarray([float(img_focal)], dtype=np.float32),
        video_path=np.asarray([str(args.video_path)]),
        seq_folder=np.asarray([str(seq_folder)]),
    )
    valid_counts = {side: int(np.count_nonzero(hands[side]["valid"])) for side in hands}
    qc = {
        "status": "ok",
        "video_path": str(args.video_path),
        "seq_folder": str(seq_folder),
        "output_npz": str(out_npz),
        "frames": int(len(frame_idx)),
        "img_focal": float(img_focal),
        "valid_hand_frames": valid_counts,
        "slam_path": str(slam_path),
    }
    (args.output_dir / "qc_hawor_world_hands.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/hawor_work/third_party/HaWoR"))
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
