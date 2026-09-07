#!/usr/bin/env python3
"""Prediction-only shared generated-shape alignment under fixed object poses.

The solver keeps every per-frame object SE(3) fixed and estimates one shared
canonical anisotropic scale plus translation for the generated render mesh.
Correspondences are true first hits rebuilt at every outer iteration.  Exact
common-hit depth, full-point coverage, and silhouette gates backtrack each
candidate update.

HOT3D GT is not an input.  Generated geometry remains visible-pose/render-only
and is not promoted to collision, contact, SDF, or signed-volume authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

try:
    from v20_prediction_contracts import append_prediction_stage, assert_prediction_only
except ModuleNotFoundError:  # pragma: no cover - supports direct test imports
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v20_prediction_contracts import append_prediction_stage, assert_prediction_only


def import_module(path: Path, name: str) -> Any:
    resolved = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name}: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_digest(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    digest.update(str(vertices.shape).encode("ascii"))
    digest.update(vertices.tobytes())
    digest.update(str(faces.shape).encode("ascii"))
    digest.update(faces.tobytes())
    return digest.hexdigest()


def transform_points(
    points: np.ndarray, pivot: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape not in {(6,), (9,)}:
        raise RuntimeError(f"shape parameters must have shape (6,) or (9,), got {parameters.shape}")
    scale = np.exp(parameters[:3])
    centered = (np.asarray(points, dtype=np.float64) - pivot[None, :]) * scale[None, :]
    if len(parameters) == 9:
        rotation = Rotation.from_rotvec(parameters[3:6]).as_matrix()
        centered = centered @ rotation.T
        translation = parameters[6:9]
    else:
        translation = parameters[3:6]
    return centered + pivot[None, :] + translation[None, :]


def factor_with_transformed_points(
    factor: Any, pivot: np.ndarray, parameters: np.ndarray
) -> Any:
    from dataclasses import replace

    return replace(
        factor,
        depth_points_canonical=transform_points(
            factor.depth_points_canonical, pivot, parameters
        ),
        silhouette_points_canonical=transform_points(
            factor.silhouette_points_canonical, pivot, parameters
        ),
    )


def parameter_size(args: argparse.Namespace) -> int:
    return 9 if bool(getattr(args, "optimize_canonical_rotation", False)) else 6


def shape_regularization_blocks(parameters: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    blocks = [parameters[:3] / float(args.sigma_incremental_log_scale)]
    if len(parameters) == 9:
        blocks.append(parameters[3:6] / float(args.sigma_incremental_rotation_rad))
        blocks.append(parameters[6:9] / float(args.sigma_incremental_translation_m))
    else:
        blocks.append(parameters[3:6] / float(args.sigma_incremental_translation_m))
    return blocks


def residual_vector(
    parameters: np.ndarray,
    *,
    core: Any,
    periodic: Any,
    factors: dict[int, Any],
    pose_by_frame: dict[int, tuple[np.ndarray, np.ndarray]],
    pivot: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for frame_idx in sorted(factors):
        if frame_idx not in pose_by_frame:
            raise RuntimeError(f"fixed pose report lacks factor frame {frame_idx}")
        transformed = factor_with_transformed_points(
            factors[frame_idx], pivot, parameters
        )
        rotation, translation = pose_by_frame[frame_idx]
        depth, silhouette = periodic.generated_image_factor_blocks(
            core, transformed, rotation, translation, args
        )
        if depth is not None:
            blocks.append(depth)
        if silhouette is not None:
            blocks.append(silhouette)
    blocks.extend(shape_regularization_blocks(parameters, args))
    return np.concatenate([block.reshape(-1) for block in blocks]).astype(np.float64)


def gate_checks(deltas: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    return {
        "global_depth_median": deltas["global_abs_depth_median_delta_m"]
        <= float(args.max_global_depth_median_degradation_m),
        "global_depth_p95": deltas["global_abs_depth_p95_delta_m"]
        <= float(args.max_global_depth_p95_degradation_m),
        "segment_depth_median": deltas[
            "max_segment_common_hit_abs_depth_median_delta_m"
        ]
        <= float(args.max_segment_depth_median_degradation_m),
        "frame_depth_median": deltas[
            "max_frame_common_hit_abs_depth_median_delta_m"
        ]
        <= float(args.max_frame_depth_median_degradation_m),
        "frame_depth_p95": deltas["max_frame_common_hit_abs_depth_p95_delta_m"]
        <= float(args.max_frame_depth_p95_degradation_m),
        "frame_coverage": deltas["min_frame_first_hit_coverage_delta"]
        >= -float(args.max_frame_first_hit_coverage_degradation),
        "frame_silhouette": deltas["min_frame_silhouette_iou_delta"]
        >= -float(args.max_frame_silhouette_iou_degradation),
    }


def combined_gate_checks(
    step_deltas: dict[str, Any], initial_deltas: dict[str, Any], args: argparse.Namespace
) -> dict[str, bool]:
    """Require both local-step and fixed-initial exact gates."""
    checks = gate_checks(step_deltas, args)
    checks.update(
        {f"initial_{key}": value for key, value in gate_checks(initial_deltas, args).items()}
    )
    return checks


def signed_front_gate_for_shape(
    factors: dict[int, Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
    signed_helper: Any,
) -> tuple[dict[str, Any], bool, list[str]]:
    diagnostics = signed_helper.signed_front_bias_diagnostics(
        factors,
        metrics,
        negative_threshold_m=float(args.signed_front_negative_threshold_m),
    )
    passed, reasons = signed_helper.signed_front_bias_gate(
        diagnostics,
        max_front_bias_m=None,
        max_front_fraction=None,
        max_per_frame_front_bias_m=(
            float(args.max_per_frame_signed_front_bias_m)
            if args.max_per_frame_signed_front_bias_m is not None
            else None
        ),
        max_negative_front_segment_length=(
            int(args.max_negative_front_segment_length)
            if args.max_negative_front_segment_length is not None
            else None
        ),
    )
    return diagnostics, bool(passed), list(reasons)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-script",
        type=Path,
        default=root / "scripts" / "fit_v20_keyframe_observed_pose_graph.py",
    )
    parser.add_argument(
        "--periodic-script",
        type=Path,
        default=root / "scripts" / "fit_v20_keyframe_periodic_graph.py",
    )
    parser.add_argument(
        "--generated-visible-factor-module-script",
        type=Path,
        default=root / "scripts" / "v20_generated_first_hit_factors.py",
    )
    parser.add_argument(
        "--generated-visible-factor-builder-script",
        type=Path,
        default=(
            root
            / "experiments"
            / "sam3d_native_ghost_lite_20260826"
            / "build_p15_first_hit_silhouette_factors.py"
        ),
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--generated-visible-mesh", type=Path, required=True)
    parser.add_argument("--generated-visible-hand-npz", type=Path, required=True)
    parser.add_argument(
        "--generated-visible-mano-faces-pkl", type=Path, required=True
    )
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--outer-iterations", type=int, default=3)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--generated-visible-device", default="cuda:0")
    parser.add_argument("--generated-visible-source-size", type=int, default=1408)
    parser.add_argument("--generated-visible-raster-size", type=int, default=256)
    parser.add_argument("--generated-visible-render-batch-size", type=int, default=4)
    parser.add_argument("--generated-visible-min-frames", type=int, default=8)
    parser.add_argument("--generated-visible-min-observed-points", type=int, default=20)
    parser.add_argument(
        "--generated-visible-min-ownership-fraction", type=float, default=0.80
    )
    parser.add_argument(
        "--generated-visible-hand-unknown-dilation-px", type=int, default=1
    )
    parser.add_argument(
        "--generated-visible-boundary-downweight-radius-px",
        type=float,
        default=1.5,
    )
    parser.add_argument("--generated-visible-boundary-weight", type=float, default=0.65)
    parser.add_argument(
        "--generated-visible-min-depth-factor-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--generated-visible-max-removed-fraction-for-full-weight",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--generated-visible-max-observed-factors-per-frame",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--generated-visible-max-outside-factors-per-frame",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--generated-visible-max-missing-factors-per-frame",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--generated-visible-first-hit-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--generated-visible-silhouette-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--generated-visible-sigma-depth-m", type=float, default=0.008
    )
    parser.add_argument(
        "--generated-visible-sigma-silhouette-px", type=float, default=4.0
    )
    parser.add_argument(
        "--generated-visible-max-depth-residual-m", type=float, default=0.10
    )
    parser.add_argument(
        "--generated-visible-max-silhouette-residual-px",
        type=float,
        default=40.0,
    )
    parser.add_argument("--sigma-incremental-log-scale", type=float, default=0.12)
    parser.add_argument("--sigma-incremental-rotation-rad", type=float, default=0.12)
    parser.add_argument(
        "--sigma-incremental-translation-m", type=float, default=0.025
    )
    parser.add_argument("--max-incremental-log-scale", type=float, default=0.18)
    parser.add_argument("--max-incremental-rotation-rad", type=float, default=0.10)
    parser.add_argument("--max-cumulative-rotation-rad", type=float, default=0.15)
    parser.add_argument("--optimize-canonical-rotation", action="store_true", help="Allow one shared bounded canonical mesh-registration rotation; this is shape registration, not per-frame pose authority.")
    parser.add_argument(
        "--max-incremental-translation-m", type=float, default=0.035
    )
    parser.add_argument("--max-cumulative-log-scale", type=float, default=0.35)
    parser.add_argument(
        "--exact-gate-step-scales",
        type=str,
        default="1.0,0.5,0.25,0.1,0.05",
    )
    parser.add_argument("--min-cost-improvement", type=float, default=1.0e-4)
    parser.add_argument(
        "--max-global-depth-median-degradation-m", type=float, default=0.0005
    )
    parser.add_argument(
        "--max-global-depth-p95-degradation-m", type=float, default=0.001
    )
    parser.add_argument(
        "--max-segment-depth-median-degradation-m", type=float, default=0.001
    )
    parser.add_argument(
        "--max-frame-depth-median-degradation-m", type=float, default=0.003
    )
    parser.add_argument(
        "--max-frame-depth-p95-degradation-m", type=float, default=0.005
    )
    parser.add_argument(
        "--max-frame-first-hit-coverage-degradation",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--max-frame-silhouette-iou-degradation", type=float, default=0.005
    )
    parser.add_argument(
        "--signed-front-negative-threshold-m", type=float, default=0.005,
        help="Per-frame signed median below this is classified as front-biased.",
    )
    parser.add_argument(
        "--max-per-frame-signed-front-bias-m", type=float, default=None,
        help="Optional visible-pose/render gate for every frame's front bias.",
    )
    parser.add_argument(
        "--max-negative-front-segment-length", type=int, default=None,
        help="Optional maximum contiguous front-biased segment length.",
    )
    parser.add_argument(
        "--enforce-signed-front-gate", action="store_true",
        help="Reject shape updates/final status when the explicit per-frame/segment render gate fails.",
    )
    args = parser.parse_args()

    if int(args.outer_iterations) < 1:
        raise RuntimeError("outer iterations must be positive")
    step_scales = [
        float(value.strip())
        for value in str(args.exact_gate_step_scales).split(",")
        if value.strip()
    ]
    if (
        not step_scales
        or any(scale <= 0.0 or scale > 1.0 for scale in step_scales)
        or any(
            step_scales[index] <= step_scales[index + 1]
            for index in range(len(step_scales) - 1)
        )
    ):
        raise RuntimeError("exact gate step scales must be strictly descending in (0,1]")

    core = import_module(args.core_script, "v20_shape_core")
    periodic = import_module(args.periodic_script, "v20_shape_periodic")
    signed_helper = import_module(
        root / "scripts" / "fit_v20_late_window_prediction_only_se3.py",
        "v20_shape_signed_front_helpers",
    )
    generated = import_module(
        args.generated_visible_factor_module_script, "v20_shape_generated"
    )
    context = generated.create_context(args)
    if context is None:
        raise RuntimeError("generated factor context was not created")
    pose_report = periodic.load_json(args.pose_report)
    assert_prediction_only(
        pose_report,
        label="shape-alignment pose report",
        object_id=args.object_id,
    )
    fixed_poses = periodic.pose_map_from_report(pose_report)
    missing = sorted(
        int(frame.frame_idx)
        for frame in context.source_frames
        if int(frame.frame_idx) not in fixed_poses
    )
    if missing:
        raise RuntimeError(f"fixed pose report lacks factor frames: {missing[:20]}")

    initial_vertices = np.asarray(context.mesh_vertices, dtype=np.float64).copy()
    initial_extent = np.ptp(initial_vertices, axis=0)
    initial_input_hash = context.mesh_sha256
    initial_factors, initial_metrics = generated.rebuild_factors(
        context, fixed_poses, args
    )
    initial_metrics["mesh_sha256"] = geometry_digest(
        initial_vertices, context.mesh_faces
    )
    initial_signed_front, initial_signed_front_pass, initial_signed_front_reasons = signed_front_gate_for_shape(
        initial_factors, initial_metrics, args, signed_helper
    )
    outer_reports: list[dict[str, Any]] = []
    fixed_frame_ids = sorted(fixed_poses)

    for outer in range(int(args.outer_iterations)):
        current_vertices = np.asarray(context.mesh_vertices, dtype=np.float64).copy()
        current_digest = geometry_digest(current_vertices, context.mesh_faces)
        current_factors, current_info = generated.rebuild_factors(
            context, fixed_poses, args
        )
        current_info["mesh_sha256"] = current_digest
        pivot = np.median(current_vertices, axis=0)
        zero = np.zeros(parameter_size(args), dtype=np.float64)
        residual = lambda parameters: residual_vector(
            parameters,
            core=core,
            periodic=periodic,
            factors=current_factors,
            pose_by_frame=fixed_poses,
            pivot=pivot,
            args=args,
        )
        before = residual(zero)
        before_cost = float(np.dot(before, before))
        if bool(args.optimize_canonical_rotation):
            lower = np.asarray(
                [-float(args.max_incremental_log_scale)] * 3
                + [-float(args.max_incremental_rotation_rad)] * 3
                + [-float(args.max_incremental_translation_m)] * 3,
                dtype=np.float64,
            )
        else:
            lower = np.asarray(
                [-float(args.max_incremental_log_scale)] * 3
                + [-float(args.max_incremental_translation_m)] * 3,
                dtype=np.float64,
            )
        upper = -lower
        result = least_squares(
            residual,
            zero,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=int(args.max_nfev),
        )
        trials: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        for step_scale in step_scales:
            parameters = np.asarray(result.x, dtype=np.float64) * float(step_scale)
            candidate_vertices = transform_points(
                current_vertices, pivot, parameters
            )
            candidate_extent = np.ptp(candidate_vertices, axis=0)
            cumulative_log_scale = np.log(
                np.maximum(candidate_extent, 1.0e-12)
                / np.maximum(initial_extent, 1.0e-12)
            )
            cumulative_rotation = (
                parameters[3:6].copy() if bool(args.optimize_canonical_rotation) else np.zeros(3, dtype=np.float64)
            )
            trial_residual = residual(parameters)
            trial_cost = float(np.dot(trial_residual, trial_residual))
            context.mesh_vertices = candidate_vertices
            try:
                candidate_factors, candidate_info = generated.rebuild_factors(
                    context, fixed_poses, args
                )
            finally:
                context.mesh_vertices = current_vertices
            candidate_info["mesh_sha256"] = geometry_digest(
                candidate_vertices, context.mesh_faces
            )
            current_for_compare = dict(current_info)
            current_for_compare["mesh_sha256"] = "same_state_contract"
            candidate_for_compare = dict(candidate_info)
            candidate_for_compare["mesh_sha256"] = "same_state_contract"
            step_deltas = periodic.generated_metric_deltas(
                current_for_compare,
                candidate_for_compare,
                fixed_frame_ids,
                current_factors,
                candidate_factors,
            )
            initial_for_compare = dict(initial_metrics)
            initial_for_compare["mesh_sha256"] = "same_state_contract"
            global_deltas = periodic.generated_metric_deltas(
                initial_for_compare,
                candidate_for_compare,
                fixed_frame_ids,
                initial_factors,
                candidate_factors,
            )
            checks = combined_gate_checks(step_deltas, global_deltas, args)
            candidate_signed_front, candidate_signed_front_pass, candidate_signed_front_reasons = signed_front_gate_for_shape(
                candidate_factors, candidate_info, args, signed_helper
            )
            if bool(args.enforce_signed_front_gate):
                checks["signed_front_render_gate"] = bool(candidate_signed_front_pass)
            checks["optimizer_success"] = bool(result.success)
            checks["cost_improvement"] = trial_cost <= before_cost - float(
                args.min_cost_improvement
            ) * max(before_cost, 1.0)
            checks["cumulative_scale"] = bool(
                np.max(np.abs(cumulative_log_scale))
                <= float(args.max_cumulative_log_scale)
            )
            checks["cumulative_rotation"] = bool(
                np.linalg.norm(cumulative_rotation)
                <= float(args.max_cumulative_rotation_rad)
            )
            trial = {
                "step_scale": float(step_scale),
                "parameters": parameters.astype(float).tolist(),
                "anisotropic_scale": np.exp(parameters[:3]).astype(float).tolist(),
                "canonical_rotation_rotvec_rad": (parameters[3:6] if bool(args.optimize_canonical_rotation) else np.zeros(3, dtype=np.float64)).astype(float).tolist(),
                "canonical_translation_m": (parameters[6:9] if bool(args.optimize_canonical_rotation) else parameters[3:6]).astype(float).tolist(),
                "candidate_extent_m": candidate_extent.astype(float).tolist(),
                "cumulative_log_scale": cumulative_log_scale.astype(float).tolist(),
                "cumulative_rotation_rotvec_rad": cumulative_rotation.astype(float).tolist(),
                "cost": trial_cost,
                "gate_checks": checks,
                "metric_deltas": step_deltas,
                "initial_to_candidate_metric_deltas": global_deltas,
                "signed_front_diagnostics": candidate_signed_front,
                "signed_front_gate_pass": bool(candidate_signed_front_pass),
                "signed_front_gate_reasons": candidate_signed_front_reasons,
                "candidate_metrics": candidate_info,
                "accepted": bool(all(checks.values())),
                "candidate_vertices": candidate_vertices,
            }
            trials.append(trial)
            if trial["accepted"]:
                selected = trial
                break
        if selected is None:
            outer_reports.append(
                {
                    "outer_iteration": outer,
                    "accepted_update": False,
                    "optimizer_success": bool(result.success),
                    "optimizer_message": str(result.message),
                    "cost_before": before_cost,
                    "trials": [
                        {k: v for k, v in trial.items() if k != "candidate_vertices"}
                        for trial in trials
                    ],
                }
            )
            break
        context.mesh_vertices = np.asarray(
            selected["candidate_vertices"], dtype=np.float64
        )
        outer_reports.append(
            {
                "outer_iteration": outer,
                "accepted_update": True,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "cost_before": before_cost,
                "selected_step_scale": float(selected["step_scale"]),
                "selected_parameters": selected["parameters"],
                "selected_anisotropic_scale": selected["anisotropic_scale"],
                "selected_canonical_rotation_rotvec_rad": selected[
                    "canonical_rotation_rotvec_rad"
                ],
                "selected_canonical_translation_m": selected[
                    "canonical_translation_m"
                ],
                "selected_metric_deltas": selected["metric_deltas"],
                "selected_initial_to_candidate_metric_deltas": selected[
                    "initial_to_candidate_metric_deltas"
                ],
                "selected_metrics": selected["candidate_metrics"],
                "trials": [
                    {k: v for k, v in trial.items() if k != "candidate_vertices"}
                    for trial in trials
                ],
            }
        )
        print(
            f"[shape-align] outer={outer + 1} "
            f"step={selected['step_scale']} "
            f"scale={selected['anisotropic_scale']} "
            f"translation={selected['canonical_translation_m']}",
            flush=True,
        )

    final_vertices = np.asarray(context.mesh_vertices, dtype=np.float64)
    final_factors, final_metrics = generated.rebuild_factors(
        context, fixed_poses, args
    )
    final_digest = geometry_digest(final_vertices, context.mesh_faces)
    final_metrics["mesh_sha256"] = final_digest
    final_for_compare = dict(final_metrics)
    final_for_compare["mesh_sha256"] = "same_state_contract"
    initial_for_compare = dict(initial_metrics)
    initial_for_compare["mesh_sha256"] = "same_state_contract"
    final_metric_deltas = periodic.generated_metric_deltas(
        initial_for_compare,
        final_for_compare,
        fixed_frame_ids,
        initial_factors,
        final_factors,
    )
    final_gate_checks = gate_checks(final_metric_deltas, args)
    final_signed_front, final_signed_front_pass, final_signed_front_reasons = signed_front_gate_for_shape(
        final_factors, final_metrics, args, signed_helper
    )
    signed_front_configured = bool(
        args.max_per_frame_signed_front_bias_m is not None
        or args.max_negative_front_segment_length is not None
    )
    if bool(args.enforce_signed_front_gate):
        final_gate_checks["signed_front_render_gate"] = bool(final_signed_front_pass)
    accepted_outer_iterations = any(row.get("accepted_update") for row in outer_reports)
    all_metric_gates_pass = bool(all(final_gate_checks.values()))
    if not accepted_outer_iterations:
        shape_status = "shape_alignment_no_safe_update"
    elif signed_front_configured and not final_signed_front_pass:
        shape_status = "shape_alignment_incomplete_signed_front_validation"
    elif not all_metric_gates_pass:
        shape_status = "shape_alignment_no_safe_update"
    elif not signed_front_configured:
        shape_status = "shape_alignment_complete_signed_front_unchecked"
    else:
        shape_status = "shape_alignment_complete"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "generated_visible_shape_aligned_canonical.ply"
    trimesh.Trimesh(
        vertices=final_vertices,
        faces=np.asarray(context.mesh_faces, dtype=np.int64),
        process=False,
    ).export(mesh_path)
    report = {
        "schema": "v20_prediction_generated_shape_alignment_v1",
        "status": shape_status,
        "annotation_ready": False,
        "diagnostic_only": True,
        "gt_consumed": False,
        "object_id": args.object_id,
        "input_pose_report_schema": pose_report.get("schema"),
        "claim_scope": (
            "Shared prediction-only canonical anisotropic scale/translation (and "
            "optional bounded canonical registration rotation) under fixed object "
            "poses. Generated geometry remains render-only and has no "
            "collision/contact/SDF/sign authority."
        ),
        "inputs": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "pose_report": str(args.pose_report.expanduser().resolve()),
            "generated_visible_mesh": str(
                args.generated_visible_mesh.expanduser().resolve()
            ),
            "generated_visible_hand_npz": str(
                args.generated_visible_hand_npz.expanduser().resolve()
            ),
            "generated_visible_mano_faces_pkl": str(
                args.generated_visible_mano_faces_pkl.expanduser().resolve()
            ),
            "sha256": {
                "annotations": sha256_file(args.annotations),
                "pose_report": sha256_file(args.pose_report),
                "generated_visible_mesh": initial_input_hash,
                "generated_visible_hand_npz": sha256_file(
                    args.generated_visible_hand_npz
                ),
                "generated_visible_mano_faces_pkl": sha256_file(
                    args.generated_visible_mano_faces_pkl
                ),
            },
        },
        "fixed_pose_frame_count": len(fixed_poses),
        "factor_frame_count": len(final_factors),
        "initial_extent_m": initial_extent.astype(float).tolist(),
        "final_extent_m": np.ptp(final_vertices, axis=0).astype(float).tolist(),
        "shape_parameterization": "anisotropic_scale_plus_canonical_translation_plus_shared_bounded_rotation" if bool(args.optimize_canonical_rotation) else "anisotropic_scale_plus_canonical_translation",
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "initial_to_final_metric_deltas": final_metric_deltas,
        "final_gate_checks": final_gate_checks,
        "signed_front_gate": {
            "configured": signed_front_configured,
            "enforced": bool(args.enforce_signed_front_gate),
            "negative_threshold_m": float(args.signed_front_negative_threshold_m),
            "max_per_frame_front_bias_m": args.max_per_frame_signed_front_bias_m,
            "max_negative_front_segment_length": args.max_negative_front_segment_length,
            "initial_pass": bool(initial_signed_front_pass),
            "initial_reasons": initial_signed_front_reasons,
            "initial_diagnostics": initial_signed_front,
            "pass": bool(final_signed_front_pass),
            "reasons": final_signed_front_reasons,
            "diagnostics": final_signed_front,
            "scope": "visible-pose/render evidence only; not physical geometry authority",
        },
        "outer_iterations": outer_reports,
        "render_mesh_contract": {
            "required_mesh_path": str(mesh_path),
            "required_mesh_sha256": sha256_file(mesh_path),
            "renderer_must_match": True,
        },
        "outputs": {
            "aligned_mesh": str(mesh_path),
            "report": str(output_dir / "v20_generated_shape_alignment_report.json"),
        },
    }
    append_prediction_stage(
        report,
        stage="prediction_generated_shape_alignment",
        input_hashes=report["inputs"]["sha256"],
        notes={"fixed_pose_report_sha256": report["inputs"]["sha256"]["pose_report"]},
    )
    report_path = output_dir / "v20_generated_shape_alignment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "gt_consumed": False,
                "accepted_outer_iterations": sum(
                    bool(row.get("accepted_update")) for row in outer_reports
                ),
                "initial_depth": (initial_metrics or {}).get(
                    "true_first_hit_abs_depth_m"
                ),
                "final_depth": final_metrics.get("true_first_hit_abs_depth_m"),
                "initial_coverage": (initial_metrics or {}).get(
                    "true_first_hit_coverage_fraction_median"
                ),
                "final_coverage": final_metrics.get(
                    "true_first_hit_coverage_fraction_median"
                ),
                "initial_iou": (initial_metrics or {}).get(
                    "initial_silhouette_iou_median"
                ),
                "final_iou": final_metrics.get("initial_silhouette_iou_median"),
                "mesh": str(mesh_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
