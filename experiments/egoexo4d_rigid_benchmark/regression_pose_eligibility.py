#!/usr/bin/env python3
"""Dependency-light regression for the P09 -> P14 -> P15 eligibility contract.

This is intentionally a standalone assertion script rather than a pytest test.
It constructs a three-frame rigid cuboid fixture with one explicit eligible row,
one explicit ineligible row, and one legacy row whose eligibility is unspecified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    repo_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--work-dir", type=Path, default=None, help="Optional persistent work directory; otherwise use a temporary directory")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return payload


def write_cuboid_ply(path: Path) -> None:
    vertices = [
        (-0.02, -0.01, -0.005),
        (0.02, -0.01, -0.005),
        (0.02, 0.01, -0.005),
        (-0.02, 0.01, -0.005),
        (-0.02, -0.01, 0.005),
        (0.02, -0.01, 0.005),
        (0.02, 0.01, 0.005),
        (-0.02, 0.01, 0.005),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(" ".join(str(value) for value in vertex) for vertex in vertices)
    lines.extend("3 " + " ".join(str(value) for value in face) for face in faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cuboid_surface_points() -> np.ndarray:
    xs = np.linspace(-0.02, 0.02, 7)
    ys = np.linspace(-0.01, 0.01, 5)
    zs = np.linspace(-0.005, 0.005, 3)
    points: list[list[float]] = []
    for x in xs:
        for y in ys:
            points.extend(([float(x), float(y), -0.005], [float(x), float(y), 0.005]))
    for x in xs:
        for z in zs:
            points.extend(([float(x), -0.01, float(z)], [float(x), 0.01, float(z)]))
    for y in ys:
        for z in zs:
            points.extend(([-0.02, float(y), float(z)], [0.02, float(y), float(z)]))
    return np.unique(np.asarray(points, dtype=float), axis=0)


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    mesh = root / "fixture" / "cuboid.ply"
    completion = root / "fixture" / "completion.json"
    annotations = root / "fixture" / "annotations.json"
    write_cuboid_ply(mesh)
    write_json(completion, {"outputs": {"completed_mesh_labeled": str(mesh)}})

    canonical = cuboid_surface_points()
    frames: list[dict[str, Any]] = []
    eligibility = [True, False, None]
    for frame_idx, eligible in enumerate(eligibility):
        translation = np.asarray([0.01 * frame_idx, 0.0, 1.0], dtype=float)
        geometry: dict[str, Any] = {
            "world_vertices_sample_m": (canonical + translation[None, :]).tolist(),
        }
        obj: dict[str, Any] = {
            "object_id": "toy_object",
            "visible_geometry_candidate": geometry,
            "reconstructed_geometry_pose": {
                "rotation_world_from_canonical_matrix": np.eye(3).tolist(),
                "translation_world_m": translation.tolist(),
                "pose_source": "synthetic_exact_pose",
                "pose_observation_residual_norm": 0.0,
            },
        }
        if eligible is not None:
            obj["rigid_pose_observation_eligible"] = eligible
            geometry["rigid_pose_observation_eligible"] = eligible
            reason = "synthetic_pass" if eligible else "synthetic_explicit_rejection"
            obj["rigid_pose_observation_reason"] = reason
            geometry["rigid_pose_observation_reason"] = reason
        frames.append({"frame_idx": frame_idx, "objects": [obj]})
    write_json(
        annotations,
        {
            "case": "synthetic_pose_eligibility_contract",
            "raw_video": {"frame_count": 3, "fps": 30.0, "width": 64, "height": 64},
            "frames": frames,
        },
    )
    return annotations, completion, mesh


def run_command(command: list[str], *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(command)}")
    return result


def p14_command(python: Path, script: Path, annotations: Path, completion: Path, output: Path, *, override: bool) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), str(script),
        "--annotations", str(annotations),
        "--completion-report", str(completion),
        "--object-id", "toy_object",
        "--output-dir", str(output),
        "--sample-count", "400",
        "--iterations", "2",
    ]
    if override:
        command.append("--include-ineligible-rigid-pose-observations")
    return command


def p15_command(
    python: Path,
    script: Path,
    annotations: Path,
    completion: Path,
    pose_report: Path,
    output: Path,
    *,
    override: bool = False,
    completion_enabled: bool = True,
) -> list[str]:
    command = [
        str(python), str(script),
        "--annotations", str(annotations),
        "--pose-report", str(pose_report),
        "--completion-report", str(completion),
        "--object-id", "toy_object",
        "--output-dir", str(output),
        "--min-graph-frames", "3",
        "--min-visible-points", "20",
        "--surface-metric-sample-count", "400",
        "--max-nfev", "10",
        "--complete-full-timeline-rigid-pose" if completion_enabled else "--no-complete-full-timeline-rigid-pose",
    ]
    if override:
        command.append("--include-ineligible-rigid-pose-observations")
    return command


def run_regression(repo: Path, python: Path, root: Path) -> dict[str, Any]:
    p14_script = repo / "scripts" / "fit_v18_compact_rigid_object_pose.py"
    p15_script = repo / "scripts" / "solve_v19_rigid_object_pose_graph.py"
    for path in (python, p14_script, p15_script):
        if not path.exists():
            raise FileNotFoundError(path)

    annotations, completion, _mesh = build_fixture(root)
    p14_default_dir = root / "p14_default"
    p14_override_dir = root / "p14_override"
    run_command(p14_command(python, p14_script, annotations, completion, p14_default_dir, override=False))
    run_command(p14_command(python, p14_script, annotations, completion, p14_override_dir, override=True))

    p14_default_path = p14_default_dir / "v18_compact_rigid_object_pose_fit_report.json"
    p14_override_path = p14_override_dir / "v18_compact_rigid_object_pose_fit_report.json"
    p14_default = load_json(p14_default_path)
    p14_override = load_json(p14_override_path)
    default_fit_frames = [row["frame_idx"] for row in p14_default["pose_rows"] if row.get("status") == "fit_to_visible_depth_samples"]
    rejected_frames = [row["frame_idx"] for row in p14_default["pose_rows"] if row.get("status") == "rigid_pose_observation_ineligible"]
    override_fit_frames = [row["frame_idx"] for row in p14_override["pose_rows"] if row.get("status") == "fit_to_visible_depth_samples"]
    assert default_fit_frames == [0, 2], default_fit_frames
    assert rejected_frames == [1], rejected_frames
    assert p14_default["explicit_eligible_input_count"] == 1
    assert p14_default["eligibility_unspecified_input_count"] == 1
    assert p14_default["explicit_ineligible_input_count"] == 1
    assert p14_default["explicit_eligible_fit_count"] == 1
    assert p14_default["eligibility_unspecified_fit_count"] == 1
    assert p14_default["ineligible_observation_count"] == 1
    assert p14_default["ineligible_override_fit_count"] == 0
    assert override_fit_frames == [0, 1, 2], override_fit_frames
    assert p14_override["ineligible_override_fit_count"] == 1

    # Two trusted rows with a minimum of three may complete the timeline only as
    # an unresolved hypothesis; disabling completion must fail hard.
    p15_low_dir = root / "p15_low_support"
    run_command(p15_command(python, p15_script, annotations, completion, p14_default_path, p15_low_dir))
    p15_low = load_json(p15_low_dir / "v19_rigid_object_pose_graph_report.json")
    assert p15_low["graph_frame_count"] == 2
    assert p15_low["graph_support"]["sufficient"] is False
    assert p15_low["annotation_ready"] is False
    assert p15_low["status"] == "completed_uncertain_insufficient_trusted_pose_graph_support"
    low_pose_rows = [
        row for row in p15_low["pose_rows"]
        if row.get("status") in {"corrected_temporal_rigid_pose_graph", "completed_temporal_rigid_pose_uncertain"}
    ]
    assert len(low_pose_rows) == 3
    assert all(row.get("graph_support_sufficient") is False and row.get("annotation_ready") is False for row in low_pose_rows)
    failed = run_command(
        p15_command(
            python,
            p15_script,
            annotations,
            completion,
            p14_default_path,
            root / "p15_low_support_no_completion",
            completion_enabled=False,
        ),
        expect_success=False,
    )
    assert "below min_graph_frames=3" in (failed.stdout + failed.stderr)

    # A single direct observation must still produce only explicit nearest-hold
    # hypotheses rather than crashing Slerp construction.
    p14_single = json.loads(json.dumps(p14_default))
    for row in p14_single["pose_rows"]:
        if row.get("frame_idx") == 2:
            row["status"] = "synthetic_removed_second_pose_observation"
    p14_single_path = root / "p14_single_observation.json"
    write_json(p14_single_path, p14_single)
    p15_single_dir = root / "p15_single_observation"
    run_command(p15_command(python, p15_script, annotations, completion, p14_single_path, p15_single_dir))
    p15_single = load_json(p15_single_dir / "v19_rigid_object_pose_graph_report.json")
    assert p15_single["graph_frame_count"] == 1
    assert p15_single["annotation_ready"] is False
    assert p15_single["full_timeline_rigid_pose_completion"]["completed_row_count"] == 2
    assert p15_single["full_timeline_rigid_pose_completion"]["mode_counts"] == {"nearest_visible_pose_hold": 2}

    # P15 independently rejects the explicit-false row even if P14's historical
    # override fitted it.
    p15_defense_dir = root / "p15_defense"
    run_command(p15_command(python, p15_script, annotations, completion, p14_override_path, p15_defense_dir))
    p15_defense = load_json(p15_defense_dir / "v19_rigid_object_pose_graph_report.json")
    assert p15_defense["graph_frame_count"] == 2
    assert p15_defense["pose_observation_eligibility_policy"]["explicit_ineligible_candidate_count"] == 1
    assert p15_defense["pose_observation_eligibility_policy"]["explicit_ineligible_skipped_count"] == 1
    assert p15_defense["annotation_ready"] is False

    # Reproducing all three historical fits requires a second explicit P15
    # override and is fully audited.
    p15_override_dir = root / "p15_override"
    run_command(
        p15_command(
            python,
            p15_script,
            annotations,
            completion,
            p14_override_path,
            p15_override_dir,
            override=True,
        )
    )
    p15_override = load_json(p15_override_dir / "v19_rigid_object_pose_graph_report.json")
    assert p15_override["graph_frame_count"] == 3
    assert p15_override["graph_support"]["sufficient"] is True
    assert p15_override["annotation_ready"] is True
    assert p15_override["pose_observation_eligibility_policy"]["include_ineligible_override"] is True
    assert p15_override["pose_observation_eligibility_policy"]["explicit_ineligible_admitted_count"] == 1

    return {
        "status": "pass",
        "method": "standalone_p09_p14_p15_eligibility_regression",
        "fixture": {
            "frames": 3,
            "eligibility": {"0": True, "1": False, "2": None},
        },
        "assertions": {
            "p14_default_fit_frames": default_fit_frames,
            "p14_default_rejected_frames": rejected_frames,
            "p14_override_fit_frames": override_fit_frames,
            "p15_low_support_annotation_ready": p15_low["annotation_ready"],
            "p15_low_support_full_timeline_rows": len(low_pose_rows),
            "p15_no_completion_hard_rejection": True,
            "p15_single_observation_nearest_hold_rows": p15_single["full_timeline_rigid_pose_completion"]["completed_row_count"],
            "p15_defense_explicit_false_skipped": p15_defense["pose_observation_eligibility_policy"]["explicit_ineligible_skipped_count"],
            "p15_double_override_annotation_ready": p15_override["annotation_ready"],
        },
        "work_dir": str(root),
    }


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    # Do not resolve the venv launcher symlink: invoking its base interpreter path
    # directly would bypass the virtual environment's site-packages.
    python = args.python.expanduser().absolute()
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="v19_pose_eligibility_regression_")
        root = Path(temporary.name)
    else:
        root = args.work_dir.resolve()
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
    try:
        report = run_regression(repo, python, root)
        report["work_dir_persisted"] = temporary is None
        if temporary is not None:
            report["work_dir"] = None
        if args.output_json is not None:
            write_json(args.output_json.resolve(), report)
        print(json.dumps(report, indent=2))
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
