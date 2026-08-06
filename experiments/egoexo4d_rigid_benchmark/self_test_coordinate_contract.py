#!/usr/bin/env python3
"""Synthetic round-trip test for the Ego-Exo4D camera-coordinate adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_benchmark import evaluate_camera_trajectory, evaluate_hand_joint_map, load_json, write_json


def run_test(gt_dir: Path) -> dict[str, object]:
    hand_gt = load_json(gt_dir / "hand_pose_gt.json")
    camera_gt = load_json(gt_dir / "camera_pose_gt.json")
    annotation_to_extracted = np.asarray(
        camera_gt["coordinate_contract"]["annotation_camera_to_prediction_extracted_camera_3x3"],
        dtype=np.float64,
    )
    extrinsics = {
        int(row["local_frame_idx"]): np.asarray(row["world_to_camera_3x4"], dtype=np.float64)
        for row in camera_gt["frames"]
    }

    predictor_cameras: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rotations: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    frame_indices: list[int] = []
    for frame_idx in sorted(extrinsics):
        world_to_camera = extrinsics[frame_idx]
        rotation_annotation_to_world = world_to_camera[:, :3].T
        center_world = -rotation_annotation_to_world @ world_to_camera[:, 3]
        rotation_extracted_to_world = rotation_annotation_to_world @ annotation_to_extracted.T
        predictor_cameras[frame_idx] = (rotation_extracted_to_world, center_world)
        frame_indices.append(frame_idx)
        rotations.append(rotation_extracted_to_world)
        centers.append(center_world)

    synthetic_joints: dict[tuple[int, str], np.ndarray] = {}
    for frame in hand_gt["frames"]:
        frame_idx = int(frame["local_frame_idx"])
        for side, hand in frame["hands"].items():
            joints = np.zeros((21, 3), dtype=np.float64)
            for point in hand["joints"]:
                joints[int(point["hawor_openpose_index"])] = np.asarray(point["xyz_world_m"], dtype=np.float64)
            synthetic_joints[(frame_idx, side)] = joints

    hand_report = evaluate_hand_joint_map(
        "synthetic_exact_3d",
        synthetic_joints,
        predictor_cameras,
        hand_gt,
        camera_gt,
        min_num_views=2,
        max_gt_reprojection_px_rectified_512=20.0,
    )
    camera_report = evaluate_camera_trajectory(
        {
            "frame_idx": np.asarray(frame_indices, dtype=np.int32),
            "R_c2w": np.asarray(rotations, dtype=np.float64),
            "t_c2w": np.asarray(centers, dtype=np.float64),
        },
        camera_gt,
    )
    checks = {
        "adapter_orthonormal": bool(np.allclose(annotation_to_extracted.T @ annotation_to_extracted, np.eye(3), atol=1e-12)),
        "adapter_determinant": float(np.linalg.det(annotation_to_extracted)),
        "hand_absolute_mean_mm": hand_report["all_hands"]["absolute_joint_error_mm"]["mean"],
        "hand_root_relative_mean_mm": hand_report["all_hands"]["root_relative_joint_error_mm"]["mean"],
        "hand_pck_3d_50mm": hand_report["all_hands"]["pck_3d_50mm"],
        "rectified_reprojection_mean_px_includes_released_2d_annotation_noise": hand_report["all_hands"]["reprojection_error_px_rectified_512"]["mean"],
        "camera_se3_ate_rmse_mm": camera_report["alignments"]["se3_metric"]["ate_center_rmse_mm"],
        "camera_orientation_gauge_mean_deg": camera_report["orientation_gauge_alignment"]["orientation_error_deg"]["mean"],
        "camera_delta_1_rotation_mean_deg": camera_report["relative_pose"]["delta_1_frames"]["relative_rotation_error_deg"]["mean"],
    }
    passed = (
        checks["adapter_orthonormal"]
        and abs(float(checks["adapter_determinant"]) - 1.0) < 1e-12
        and float(checks["hand_absolute_mean_mm"]) < 1e-4
        and float(checks["hand_root_relative_mean_mm"]) < 1e-4
        and float(checks["hand_pck_3d_50mm"]) == 1.0
        and float(checks["camera_se3_ate_rmse_mm"]) < 1e-6
        and float(checks["camera_orientation_gauge_mean_deg"]) < 0.01
    )
    return {
        "status": "pass" if passed else "fail",
        "method": "synthetic_exact_3d_round_trip_through_official_extracted_view_axis_adapter",
        "ground_truth_dir": str(gt_dir),
        "fitted_from_evaluation_data": False,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_test(args.ground_truth_dir.resolve())
    if args.output:
        write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
