#!/usr/bin/env python3
"""Bind a frozen prediction-only pose report to an exact shape render mesh."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from v20_prediction_contracts import (
        append_prediction_stage,
        assert_prediction_only,
        require_equal_object_ids,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct test imports
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v20_prediction_contracts import (
        append_prediction_stage,
        assert_prediction_only,
        require_equal_object_ids,
    )


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_all_true(report: dict[str, Any], field: str, label: str) -> None:
    value = report.get(field)
    if isinstance(value, dict) and value and not all(bool(item) for item in value.values()):
        raise RuntimeError(f"{label} has failed {field}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--shape-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pose_path = args.pose_report.expanduser().resolve()
    shape_path = args.shape_report.expanduser().resolve()
    pose = load(pose_path)
    shape = load(shape_path)
    assert_prediction_only(
        pose,
        label="binder pose report",
        allowed_schema_prefixes=(
            "v20_prediction_camera_z_translation_graph_v2",
            "v20_prediction_rgb_rotation_pose_graph_v1",
            "v20_keyframe_periodic_global_observed_pose_graph",
            "v20_late_window_prediction_only_full_timeline_se3_v1",
        ),
        allowed_statuses=(
            "camera_z_translation_complete",
            "rgb_rotation_pose_graph_complete",
            "v20_keyframe_global_complete",
            "v20_late_window_se3_complete",
            "v20_late_window_se3_complete_signed_front_unchecked",
            "v20_late_window_se3_incomplete_signed_front_validation",
            "v20_late_window_se3_incomplete_validation_unavailable",
        ),
    )
    assert_prediction_only(
        shape,
        label="binder shape report",
        allowed_schema_prefixes=("v20_prediction_generated_shape_alignment_v1",),
        allowed_statuses=(
            "shape_alignment_complete",
            "shape_alignment_complete_signed_front_unchecked",
            "shape_alignment_incomplete_signed_front_validation",
        ),
    )
    object_id = require_equal_object_ids(("pose report", pose), ("shape report", shape))
    if not isinstance(pose.get("pose_rows"), list) or not pose["pose_rows"]:
        raise RuntimeError("pose report has no pose rows")
    require_all_true(pose, "final_gate_checks", "pose report")
    require_all_true(shape, "final_gate_checks", "shape report")

    shape_inputs = shape.get("inputs") or {}
    expected_pose_hash = (shape_inputs.get("sha256") or {}).get("pose_report")
    actual_pose_hash = sha(pose_path)
    if expected_pose_hash != actual_pose_hash:
        raise RuntimeError("shape report was not optimized against this exact pose report")

    contract = shape.get("render_mesh_contract") or {}
    mesh = Path(str(contract.get("required_mesh_path") or "")).expanduser().resolve()
    mesh_hash = str(contract.get("required_mesh_sha256") or "")
    if (
        not mesh.is_file()
        or sha(mesh) != mesh_hash
        or not bool(contract.get("renderer_must_match"))
    ):
        raise RuntimeError("shape render mesh contract is invalid")

    shape_signed_front = shape.get("signed_front_gate") or {}
    incomplete_signed_front = (
        str(pose.get("status") or "").endswith("incomplete_signed_front_validation")
        or str(shape.get("status") or "").endswith("incomplete_signed_front_validation")
        or (
            bool(shape_signed_front.get("configured"))
            and shape_signed_front.get("pass") is False
        )
    )
    signed_front_unchecked = (
        str(pose.get("status") or "").endswith("signed_front_unchecked")
        or str(shape.get("status") or "").endswith("signed_front_unchecked")
    )
    output: dict[str, Any] = dict(pose)
    output.update(
        {
            "schema": "v20_prediction_rgb_rotation_camera_z_shape_candidate_v2",
            "status": (
                "prediction_candidate_incomplete_signed_front_validation"
                if incomplete_signed_front
                else (
                    "prediction_candidate_signed_front_unchecked"
                    if signed_front_unchecked
                    else "prediction_candidate_frozen_for_visual_review"
                )
            ),
            "object_id": object_id,
            "claim_scope": (
                "Prediction-only RGB SO(3) rotation, camera-z translation, and "
                "shared generated visible-shape alignment. Generated mesh is "
                "visible-pose/render evidence only and has no collision/contact/"
                "SDF/sign/nonpenetration authority."
            ),
            "component_contract": {
                "pose_report": str(pose_path),
                "pose_report_sha256": actual_pose_hash,
                "shape_report": str(shape_path),
                "shape_report_sha256": sha(shape_path),
                "shape_input_pose_sha256_match": True,
                "object_id_match": True,
            },
            "surface_metrics": shape.get("final_metrics"),
            "signed_front_validation": shape_signed_front,
            "generated_shape_alignment": {
                "schema": shape.get("schema"),
                "status": shape.get("status"),
                "report": str(shape_path),
                "report_sha256": sha(shape_path),
                "initial_extent_m": shape.get("initial_extent_m"),
                "final_extent_m": shape.get("final_extent_m"),
                "accepted_outer_iterations": sum(
                    bool(item.get("accepted_update"))
                    for item in shape.get("outer_iterations", [])
                ),
            },
            "render_mesh_contract": {
                "required_mesh_path": str(mesh),
                "required_mesh_sha256": mesh_hash,
                "renderer_must_match": True,
            },
            "output": str(args.output.expanduser().resolve()),
        }
    )
    append_prediction_stage(
        output,
        stage="prediction_candidate_binder",
        input_hashes={
            "pose_report": actual_pose_hash,
            "shape_report": sha(shape_path),
            "render_mesh": mesh_hash,
        },
        notes={"object_id": object_id, "visual_review_only": True},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "gt_consumed": False,
                "object_id": object_id,
                "pose_rows": len(output["pose_rows"]),
                "mesh": str(mesh),
                "mesh_sha256": mesh_hash,
                "depth": (output.get("surface_metrics") or {}).get(
                    "true_first_hit_abs_depth_m"
                ),
                "coverage": (output.get("surface_metrics") or {}).get(
                    "true_first_hit_coverage_fraction_median"
                ),
                "iou": (output.get("surface_metrics") or {}).get(
                    "initial_silhouette_iou_median"
                ),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
