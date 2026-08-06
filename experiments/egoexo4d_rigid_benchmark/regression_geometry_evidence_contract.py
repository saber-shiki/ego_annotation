#!/usr/bin/env python3
"""Standalone regressions for P13/P14b physical-geometry evidence contracts.

No pytest installation or benchmark/NAS artifact is required. The regression
constructs synthetic meshes, masks, depths, cameras, and direct pose rows, then
runs the production CLIs in temporary directories.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
P13_SCRIPT = REPO_ROOT / "scripts" / "build_v18_compact_rigid_trellis_completion.py"
P14B_SCRIPT = REPO_ROOT / "scripts" / "filter_v19_rigid_completion_multiview_support.py"
P16_SCRIPT = REPO_ROOT / "scripts" / "build_v18_mano_object_constraint_state.py"
P18_SCRIPT = REPO_ROOT / "scripts" / "solve_v18_joint_mano_interval_trajectory.py"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")


def run_expect_failure(command: list[str], expected_message: str) -> None:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode == 0 or expected_message not in completed.stdout:
        raise AssertionError(
            f"expected command failure containing {expected_message!r}: {' '.join(command)}\n"
            f"returncode={completed.returncode}\n{completed.stdout}"
        )


def mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    assert isinstance(loaded, trimesh.Trimesh)
    return loaded


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_p13_inputs(root: Path) -> tuple[Path, Path]:
    input_dir = root / "p13_inputs"
    input_dir.mkdir(parents=True)
    x_values = np.linspace(-0.10, 0.10, 21)
    y_values = np.linspace(-0.05, 0.05, 11)
    points = np.asarray([[x, y, 0.0] for x in x_values for y in y_values], dtype=np.float64)
    points_path = input_dir / "observed_points.ply"
    trimesh.points.PointCloud(points).export(points_path)
    faces: list[list[int]] = []
    row_width = len(y_values)
    for x_idx in range(len(x_values) - 1):
        for y_idx in range(len(y_values) - 1):
            lower_left = x_idx * row_width + y_idx
            lower_right = (x_idx + 1) * row_width + y_idx
            upper_right = lower_right + 1
            upper_left = lower_left + 1
            faces.extend([[lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left]])
    observed_mesh_path = input_dir / "observed_surface.ply"
    trimesh.Trimesh(vertices=points, faces=np.asarray(faces), process=False).export(observed_mesh_path)
    trellis_mesh_path = input_dir / "trellis_mesh.ply"
    trimesh.creation.box(extents=[0.20, 0.10, 0.06]).export(trellis_mesh_path)
    evidence_path = input_dir / "evidence.json"
    write_json(
        evidence_path,
        {
            "case": "synthetic_geometry_contract",
            "object_id": "object:test_block",
            "partial_metric_geometry_paths": {
                "fused_point_cloud_path": str(points_path),
                "poisson_mesh_path": str(observed_mesh_path),
            },
            "depth_fused_object_row": {"mesh_reconstruction": {"voxel_size_m": 0.006}},
        },
    )
    trellis_report_path = input_dir / "trellis_report.json"
    write_json(trellis_report_path, {"mesh": str(trellis_mesh_path)})
    return evidence_path, trellis_report_path


def test_p13_split(root: Path) -> tuple[Path, Path]:
    evidence, trellis = make_p13_inputs(root)
    default_dir = root / "p13_default"
    override_dir = root / "p13_historical_override"
    common = [
        sys.executable,
        str(P13_SCRIPT),
        "--evidence-report",
        str(evidence),
        "--trellis-report",
        str(trellis),
        "--no-silhouette-free-space-filter",
        "--no-planar-slab-support-filter",
    ]
    run(common + ["--output-dir", str(default_dir)])
    run(common + ["--output-dir", str(override_dir), "--promote-single-view-hidden-prior-to-collision"])
    default_report_path = default_dir / "v18_compact_rigid_trellis_completion_report.json"
    override_report_path = override_dir / "v18_compact_rigid_trellis_completion_report.json"
    default = read_json(default_report_path)
    override = read_json(override_report_path)
    assert default["status"] == "ok_with_single_view_hidden_completion_quarantined_from_physical_geometry"
    assert default["annotation_ready"] is False
    assert default["geometry_readiness"]["collision_eligible_hidden_face_count"] == 0
    assert default["geometry_readiness"]["signed_geometry_ready"] is False
    observed_count = default["accepted_body_semantics"]["observed_depth_surface_faces_accepted"]
    hidden_count = default["accepted_body_semantics"]["trellis_hidden_surface_faces_in_pose_hypothesis"]
    assert observed_count > 0 and hidden_count > 0
    assert default["mesh_counts"]["completed_faces"] == observed_count + hidden_count
    assert default["mesh_counts"]["collision_eligible_faces"] == observed_count
    assert override["status"] == "ok_historical_override_single_view_hidden_prior_promoted_for_diagnostic"
    assert override["annotation_ready"] is False
    assert override["geometry_readiness"]["hidden_prior_collision_promotion_override"] is True
    assert override["geometry_readiness"]["signed_geometry_ready"] is False
    assert override["geometry_readiness"]["collision_eligible_hidden_face_count"] == hidden_count
    assert override["mesh_counts"]["collision_eligible_faces"] == override["mesh_counts"]["completed_faces"]
    default_pose = mesh(Path(default["outputs"]["pose_hypothesis_mesh_labeled"]))
    override_pose = mesh(Path(override["outputs"]["pose_hypothesis_mesh_labeled"]))
    override_collision = mesh(Path(override["outputs"]["collision_eligible_mesh_labeled"]))
    np.testing.assert_allclose(default_pose.vertices, override_pose.vertices, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(default_pose.faces, override_pose.faces)
    np.testing.assert_allclose(override_pose.vertices, override_collision.vertices, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(override_pose.faces, override_collision.faces)
    return default_report_path, override_report_path


def rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.asarray(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]],
        dtype=np.float64,
    )


def make_multiview_case(root: Path, name: str, second_view_degrees: float) -> tuple[Path, Path, Path, Path]:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    pose_mesh = trimesh.creation.box(extents=[0.10, 0.08, 0.06])
    pose_mesh_path = case_dir / "pose_hypothesis.ply"
    pose_mesh.export(pose_mesh_path)
    labels = ["observed_depth_surface"] + ["trellis_inferred_hidden_surface"] * (len(pose_mesh.faces) - 1)
    labels_path = case_dir / "completed_face_labels.json"
    write_json(labels_path, {"face_count": len(labels), "labels": labels})
    observed_mesh = pose_mesh.submesh([[0]], append=True, repair=False)
    collision_path = case_dir / "observed_collision_surface.ply"
    observed_mesh.export(collision_path)
    completion_path = case_dir / "completion_report.json"
    write_json(
        completion_path,
        {
            "method": "synthetic_p13_completion",
            "status": "ok_with_single_view_hidden_completion_quarantined_from_physical_geometry",
            "annotation_ready": False,
            "object_id": "test_block",
            "observed_band_m": 0.01,
            "outputs": {
                "completed_mesh_labeled": str(pose_mesh_path),
                "pose_hypothesis_mesh_labeled": str(pose_mesh_path),
                "collision_eligible_mesh_labeled": str(collision_path),
                "completed_face_labels": str(labels_path),
            },
            "geometry_readiness": {
                "pose_hypothesis_available": True,
                "collision_eligible_mesh": str(collision_path),
                "collision_eligible_hidden_face_count": 0,
                "collision_surface_watertight": False,
                "signed_geometry_ready": False,
                "annotation_ready": False,
            },
            "face_label_counts": {
                "completed_mesh": {
                    "observed_depth_surface": 1,
                    "trellis_inferred_hidden_surface": len(labels) - 1,
                }
            },
        },
    )
    mask_path = case_dir / "mask.png"
    cv2.imwrite(str(mask_path), np.full((128, 128), 255, dtype=np.uint8))
    raw_path = case_dir / "raw.jpg"
    cv2.imwrite(str(raw_path), np.full((128, 128, 3), 100, dtype=np.uint8))
    transform = np.eye(4, dtype=np.float64)
    frames = []
    rotations = [rotation_y(0.0), rotation_y(second_view_degrees)]
    for frame_idx in range(2):
        frames.append(
            {
                "frame_idx": frame_idx,
                "raw_frame_path": str(raw_path),
                "camera": {
                    "T_world_camera_metric": transform.tolist(),
                    "intrinsics_fx_fy_cx_cy": [100.0, 100.0, 63.5, 63.5],
                },
                "objects": [{"object_id": "test_block", "mask_path": str(mask_path)}],
            }
        )
    annotations_path = case_dir / "annotations.json"
    write_json(annotations_path, {"frames": frames})
    depth_path = case_dir / "depth.npz"
    np.savez_compressed(
        depth_path,
        frame_idx=np.asarray([0, 1], dtype=np.int32),
        depth=np.ones((2, 128, 128), dtype=np.float32),
        intrinsics_fx_fy_cx_cy=np.asarray([[100.0, 100.0, 63.5, 63.5]] * 2, dtype=np.float32),
    )
    pose_rows = []
    for frame_idx, rotation in enumerate(rotations):
        pose_rows.append(
            {
                "frame_idx": frame_idx,
                "status": "fit_to_visible_depth_samples",
                "rigid_pose_observation_eligible": True,
                "rotation_world_from_completed_canonical_matrix": rotation.tolist(),
                "translation_world_m": [0.0, 0.0, 1.0],
            }
        )
    # These rows deliberately resemble completion/hold hypotheses and must not
    # increase direct support, temporal coverage, or viewpoint diversity.
    for frame_idx in range(2, 12):
        pose_rows.append(
            {
                "frame_idx": frame_idx,
                "status": "completed_temporal_rigid_pose_uncertain",
                "rigid_pose_observation_eligible": True,
                "rotation_world_from_completed_canonical_matrix": rotation_y(60.0).tolist(),
                "translation_world_m": [0.0, 0.0, 1.0],
            }
        )
    pose_report_path = case_dir / "pose_report.json"
    write_json(
        pose_report_path,
        {
            "object_id": "test_block",
            "inputs": {
                "annotations": str(annotations_path),
                "completed_mesh": str(pose_mesh_path),
                "mesh_semantics": "pose_hypothesis_mesh_labeled",
            },
            "pose_rows": pose_rows,
        },
    )
    return annotations_path, completion_path, pose_report_path, depth_path


def run_multiview(root: Path, name: str, second_view_degrees: float) -> dict[str, Any]:
    annotations, completion, poses, depth = make_multiview_case(root, name, second_view_degrees)
    output_dir = root / f"{name}_output"
    run(
        [
            sys.executable,
            str(P14B_SCRIPT),
            "--annotations",
            str(annotations),
            "--completion-report",
            str(completion),
            "--pose-report",
            str(poses),
            "--depth-npz",
            str(depth),
            "--object-id",
            "test_block",
            "--output-dir",
            str(output_dir),
            "--depth-front-tolerance-m",
            "0.2",
            "--depth-surface-tolerance-m",
            "0.2",
            "--min-trusted-pose-frames-for-promotion",
            "2",
            "--temporal-bin-count",
            "2",
            "--min-occupied-temporal-bins-for-promotion",
            "2",
            "--min-pose-frame-span-fraction-for-promotion",
            "0.5",
            "--max-trusted-pose-gap-fraction-for-promotion",
            "0.5",
            "--viewpoint-bin-separation-deg",
            "15",
            "--no-render-qc",
        ]
    )
    return read_json(output_dir / "v19_multiview_supported_completion_report.json")


def test_multiview_support(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    same_view = run_multiview(root, "same_view", 0.0)
    distinct_view = run_multiview(root, "distinct_view", 20.0)
    same = same_view["multiview_support"]
    distinct = distinct_view["multiview_support"]
    assert same["trusted_pose_coverage"]["frame_count"] == 2
    assert same["viewpoint_diversity"]["distinct_viewpoint_bin_count"] == 1
    same_supported = same["face_counts"]["generated_hidden_repeated_same_view_support_only"]
    assert 0 < same_supported < 11
    assert same["face_counts"]["generated_hidden_distinct_viewpoint_visible_depth_supported_candidate"] == 0
    assert same["face_counts"]["generated_hidden_promoted_to_collision"] == 0
    assert same["face_counts"]["collision_eligible_total"] == 1
    assert same_view["geometry_readiness"]["signed_geometry_ready"] is False
    assert distinct["trusted_pose_coverage"]["frame_count"] == 2
    assert distinct["viewpoint_diversity"]["distinct_viewpoint_bin_count"] == 2
    assert distinct["pose_coverage_sufficient_for_hidden_collision_promotion"] is True
    distinct_supported = distinct["face_counts"]["generated_hidden_distinct_viewpoint_visible_depth_supported_candidate"]
    assert distinct_supported == same_supported
    assert distinct["face_counts"]["generated_hidden_promoted_to_collision"] == distinct_supported
    assert distinct["face_counts"]["collision_eligible_total"] == 1 + distinct_supported
    assert distinct_view["geometry_readiness"]["collision_surface_watertight"] is False
    assert distinct_view["geometry_readiness"]["signed_geometry_ready"] is False
    return same_view, distinct_view


def test_pose_canonical_binding(root: Path) -> None:
    annotations, completion, poses, depth = make_multiview_case(root, "pose_mesh_mismatch", 20.0)
    other_mesh_path = root / "pose_mesh_mismatch" / "other_canonical_mesh.ply"
    trimesh.creation.icosphere(subdivisions=1, radius=0.05).export(other_mesh_path)
    pose_payload = read_json(poses)
    pose_payload["inputs"]["completed_mesh"] = str(other_mesh_path)
    write_json(poses, pose_payload)
    run_expect_failure(
        [
            sys.executable,
            str(P14B_SCRIPT),
            "--annotations",
            str(annotations),
            "--completion-report",
            str(completion),
            "--pose-report",
            str(poses),
            "--depth-npz",
            str(depth),
            "--object-id",
            "test_block",
            "--output-dir",
            str(root / "pose_mesh_mismatch_output"),
            "--no-render-qc",
        ],
        "P14 pose/completion canonical-frame mismatch",
    )


def test_viewpoint_bin_boundary_exact_pair() -> None:
    p14b = import_script("p14b_viewpoint_pair_regression", P14B_SCRIPT)
    transform, selected_intrinsics, intrinsics_metadata = p14b.camera_contract(
        {
            "frame_idx": 0,
            "camera": {
                "T_world_camera_metric": np.eye(4, dtype=np.float64).tolist(),
                "intrinsics_fx_fy_cx_cy": [90.0, 91.0, 63.5, 63.5],
            },
        },
        {},
        np.asarray([100.0, 101.0, 63.5, 63.5], dtype=np.float64),
    )
    np.testing.assert_array_equal(transform, np.eye(4, dtype=np.float64))
    np.testing.assert_array_equal(selected_intrinsics, np.asarray([100.0, 101.0, 63.5, 63.5]))
    assert intrinsics_metadata["selected_intrinsics_source"] == "depth_npz_intrinsics_for_depth_grid_projection"
    frames = {
        frame_idx: {
            "frame_idx": frame_idx,
            "camera": {"T_world_camera_metric": np.eye(4, dtype=np.float64).tolist()},
        }
        for frame_idx in range(3)
    }
    pose_rows = [
        {
            "frame_idx": frame_idx,
            "rotation_world_from_completed_canonical_matrix": rotation_y(angle).tolist(),
            "translation_world_m": [0.0, 0.0, 1.0],
        }
        for frame_idx, angle in enumerate([10.0, 0.0, 20.0])
    ]
    info, bin_ids, separated_pairs = p14b.viewpoint_diversity(
        pose_rows,
        frames,
        np.zeros(3, dtype=np.float64),
        1.0,
        15.0,
    )
    # A greedy representative bin can contain all three views because each end
    # is 10 degrees from the first (middle) representative, while the two ends
    # are 20 degrees apart. Physical promotion uses the exact pair list.
    assert bin_ids == [0, 0, 0]
    assert info["distinct_viewpoint_bin_count"] == 1
    assert separated_pairs == [(1, 2)]
    masks = [
        np.asarray([True, False, True]),
        np.asarray([True, True, False]),
        np.asarray([True, False, False]),
    ]
    exact = p14b.faces_with_separated_view_support(masks, separated_pairs, 3)
    np.testing.assert_array_equal(exact, np.asarray([True, False, False]))


def test_consumer_selection(default_p13_report: Path, root: Path) -> None:
    p16 = import_script("p16_geometry_contract_regression", P16_SCRIPT)
    p18 = import_script("p18_geometry_contract_regression", P18_SCRIPT)
    report = read_json(default_p13_report)
    pose_path = Path(report["outputs"]["pose_hypothesis_mesh_labeled"])
    collision_path = Path(report["outputs"]["collision_eligible_mesh_labeled"])
    p16_path, p16_semantics, p16_readiness = p16.completion_surface_contract(report)
    assert p16_path == collision_path
    assert p16_semantics == "collision_eligible_mesh_labeled"
    assert p16_readiness["signed_geometry_ready"] is False
    p16_sign = p16.signed_geometry_source_readiness(p16_readiness, None, True)
    assert p16_sign["signed_geometry_ready"] is False
    signed_source_path = root / "signed_source_report.json"
    write_json(
        signed_source_path,
        {
            "geometry_readiness": {
                "signed_geometry_ready": True,
                "signed_geometry_mesh": str(collision_path),
            }
        },
    )
    bound_sign = p16.signed_geometry_source_readiness({}, signed_source_path, False, collision_path)
    assert bound_sign["signed_geometry_ready"] is True
    try:
        p16.signed_geometry_source_readiness({}, signed_source_path, False, pose_path)
    except RuntimeError as error:
        assert "sign mesh/source-report mismatch" in str(error)
    else:
        raise AssertionError("P16 accepted signed readiness from a report bound to a different mesh")
    unbound_source_path = root / "unbound_signed_source_report.json"
    write_json(unbound_source_path, {"geometry_readiness": {"signed_geometry_ready": True}})
    unbound_sign = p16.signed_geometry_source_readiness({}, unbound_source_path, False, collision_path)
    assert unbound_sign["signed_geometry_ready"] is None
    assert unbound_sign["source_mesh_contract_missing"] is True
    p18_contract = p18.resolve_physical_surface_contract(pose_path, None, default_p13_report)
    assert p18_contract["physical_surface_mesh"] == collision_path
    assert p18_contract["physical_surface_semantics"] == "collision_eligible_mesh_labeled"
    assert p18_contract["signed_geometry_ready"] is False
    try:
        p18.resolve_physical_surface_contract(pose_path, pose_path, default_p13_report)
    except RuntimeError as error:
        assert "physical surface mismatch" in str(error)
    else:
        raise AssertionError("P18 accepted the pose hypothesis as an explicit physical surface despite a collision-surface contract")
    legacy_report_path = root / "legacy_completion_report.json"
    write_json(legacy_report_path, {"outputs": {"completed_mesh_labeled": str(pose_path)}})
    legacy = read_json(legacy_report_path)
    legacy_p16_path, legacy_semantics, legacy_readiness = p16.completion_surface_contract(legacy)
    assert legacy_p16_path == pose_path
    assert legacy_semantics == "legacy_completed_mesh_labeled_unknown_collision_readiness"
    assert legacy_readiness == {}
    legacy_sign = p16.signed_geometry_source_readiness(legacy_readiness, None, True)
    assert legacy_sign["signed_geometry_ready"] is None
    assert legacy_sign["legacy_readiness_fields_missing"] is True
    legacy_p18 = p18.resolve_physical_surface_contract(pose_path, None, legacy_report_path)
    assert legacy_p18["physical_surface_semantics"] == "legacy_completed_mesh_labeled_unknown_collision_readiness"
    assert legacy_p18["signed_geometry_ready"] is None
    assert legacy_p18["legacy_geometry_readiness_fields_missing"] is True


def execute(root: Path) -> dict[str, Any]:
    default_report, override_report = test_p13_split(root)
    same_view, distinct_view = test_multiview_support(root)
    test_pose_canonical_binding(root)
    test_viewpoint_bin_boundary_exact_pair()
    test_consumer_selection(default_report, root)
    return {
        "status": "pass",
        "workdir": str(root),
        "tests": {
            "p13_default_single_view_hidden_quarantine": "pass",
            "p13_explicit_historical_override": "pass",
            "p13_pose_hypothesis_geometry_preserved": "pass",
            "p14b_completion_rows_not_counted_as_direct_support": "pass",
            "p14b_repeated_same_view_support_not_promoted": "pass",
            "p14b_self_occluded_hidden_faces_not_counted_as_visible_support": "pass",
            "p14b_distinct_viewpoint_and_temporal_support_promoted": "pass",
            "p14b_exact_viewpoint_pair_prevents_greedy_bin_boundary_error": "pass",
            "p14b_depth_grid_projection_uses_depth_npz_intrinsics": "pass",
            "p14b_pose_hypothesis_canonical_binding_mismatch_rejected": "pass",
            "p16_collision_surface_selected_and_signed_readiness_required": "pass",
            "p16_external_sign_readiness_bound_to_exact_mesh": "pass",
            "p18_collision_surface_selected_pose_hypothesis_rejected_as_physical_override": "pass",
            "legacy_missing_geometry_readiness_remains_unknown_not_ready": "pass",
        },
        "p13_default_report": str(default_report),
        "p13_override_report": str(override_report),
        "same_view_face_counts": same_view["multiview_support"]["face_counts"],
        "distinct_view_face_counts": distinct_view["multiview_support"]["face_counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=None, help="Optional persistent output directory; otherwise use a temporary directory")
    args = parser.parse_args()
    if args.workdir is not None:
        args.workdir.mkdir(parents=True, exist_ok=True)
        result = execute(args.workdir.resolve())
        print(json.dumps(result, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="geometry_evidence_contract_") as temporary:
        result = execute(Path(temporary).resolve())
        result["workdir"] = "temporary_directory_removed_after_success"
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
