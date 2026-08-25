#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh geometry in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f"not mesh: {path}")
    if len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"empty mesh: {path}")
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)


def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    rng = np.random.default_rng(1801)
    pts, _ = trimesh.sample.sample_surface(mesh, min(count, max(1, len(mesh.faces) * 2)), seed=rng)
    return np.asarray(pts, dtype=float)


def rigid_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(src) != len(dst) or len(src) < 3:
        raise RuntimeError("rigid fit requires matched arrays with >=3 points")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src
    xd = dst - mu_dst
    cov = xs.T @ xd / len(src)
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = mu_dst - (r @ mu_src)
    return r.astype(float), t.astype(float)


def apply_pose(points: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ r.T + t


def rigid_pose_observation_eligibility(obj: dict[str, Any], geom: dict[str, Any]) -> tuple[bool | None, list[str], list[str]]:
    """Resolve explicit P09 eligibility without rejecting legacy rows lacking the field."""
    values: list[bool] = []
    reasons: list[str] = []
    sources: list[str] = []
    for source, payload in (("object", obj), ("visible_geometry_candidate", geom)):
        value = payload.get("rigid_pose_observation_eligible")
        if isinstance(value, bool):
            values.append(value)
            sources.append(source)
        reason = payload.get("rigid_pose_observation_reason")
        if isinstance(reason, str) and reason and reason not in reasons:
            reasons.append(reason)
    if False in values:
        return False, reasons, sources
    if True in values:
        return True, reasons, sources
    return None, reasons, sources


def nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    if len(query) == 0 or len(target) == 0:
        return {"count": int(len(query)), "median_m": None, "p90_m": None, "p95_m": None, "mean_m": None, "max_m": None}
    d, _ = cKDTree(target).query(query, k=1, workers=-1)
    return {
        "count": int(len(query)),
        "median_m": float(np.median(d)),
        "p90_m": float(np.percentile(d, 90)),
        "p95_m": float(np.percentile(d, 95)),
        "mean_m": float(np.mean(d)),
        "max_m": float(np.max(d)),
    }


def fit_frame_pose(canonical_samples: np.ndarray, observed_world: np.ndarray, init_r: np.ndarray, init_t: np.ndarray, iterations: int) -> dict[str, Any]:
    r = init_r.copy()
    t = init_t.copy()
    trace = []
    for _ in range(iterations):
        model_world = apply_pose(canonical_samples, r, t)
        tree = cKDTree(model_world)
        _, idx = tree.query(observed_world, k=1, workers=-1)
        src = canonical_samples[idx]
        r_new, t_new = rigid_umeyama(src, observed_world)
        r, t = r_new, t_new
        trace.append(nearest_summary(observed_world, apply_pose(canonical_samples, r, t)))
    model_world = apply_pose(canonical_samples, r, t)
    init_model_world = apply_pose(canonical_samples, init_r, init_t)
    return {
        "rotation_world_from_completed_canonical_matrix": r.astype(float).tolist(),
        "translation_world_m": t.astype(float).tolist(),
        "observed_to_mesh_initial": nearest_summary(observed_world, init_model_world),
        "observed_to_mesh_final": nearest_summary(observed_world, model_world),
        "mesh_to_observed_final": nearest_summary(model_world, observed_world),
        "icp_trace": trace,
    }


def fit_quality_thresholds(mesh: trimesh.Trimesh, args: argparse.Namespace) -> dict[str, float]:
    object_diagonal_m = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)))
    median_threshold_m = float(
        np.clip(
            float(args.fit_quality_median_object_diag_fraction) * object_diagonal_m,
            float(args.fit_quality_median_floor_m),
            float(args.fit_quality_median_cap_m),
        )
    )
    p90_threshold_m = float(
        np.clip(
            float(args.fit_quality_p90_object_diag_fraction) * object_diagonal_m,
            float(args.fit_quality_p90_floor_m),
            float(args.fit_quality_p90_cap_m),
        )
    )
    return {
        "pose_hypothesis_object_diagonal_m": object_diagonal_m,
        "observed_to_mesh_final_median_threshold_m": median_threshold_m,
        "observed_to_mesh_final_p90_threshold_m": p90_threshold_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--completion-report", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=6000)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--fit-quality-median-object-diag-fraction", type=float, default=0.06)
    parser.add_argument("--fit-quality-p90-object-diag-fraction", type=float, default=0.15)
    parser.add_argument("--fit-quality-median-floor-m", type=float, default=0.004)
    parser.add_argument("--fit-quality-median-cap-m", type=float, default=0.012)
    parser.add_argument("--fit-quality-p90-floor-m", type=float, default=0.008)
    parser.add_argument("--fit-quality-p90-cap-m", type=float, default=0.025)
    parser.add_argument(
        "--include-ineligible-rigid-pose-observations",
        action="store_true",
        help="Historical-reproduction override: fit rows explicitly rejected by P09 rigid-pose eligibility",
    )
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    completion = load_json(args.completion_report)
    completion_outputs = completion.get("outputs") if isinstance(completion.get("outputs"), dict) else {}
    mesh_value = completion_outputs.get("pose_hypothesis_mesh_labeled") or completion_outputs.get("completed_mesh_labeled")
    if not mesh_value:
        raise RuntimeError("completion report lacks outputs.pose_hypothesis_mesh_labeled/completed_mesh_labeled")
    mesh_path = Path(mesh_value)
    mesh = load_mesh(mesh_path)
    if not 0.0 <= float(args.fit_quality_median_floor_m) <= float(args.fit_quality_median_cap_m):
        raise ValueError("fit-quality median floor/cap must satisfy 0 <= floor <= cap")
    if not 0.0 <= float(args.fit_quality_p90_floor_m) <= float(args.fit_quality_p90_cap_m):
        raise ValueError("fit-quality p90 floor/cap must satisfy 0 <= floor <= cap")
    if float(args.fit_quality_median_object_diag_fraction) < 0.0 or float(args.fit_quality_p90_object_diag_fraction) < 0.0:
        raise ValueError("fit-quality object-diagonal fractions must be non-negative")
    quality_thresholds = fit_quality_thresholds(mesh, args)
    canonical_samples = deterministic_sample_mesh(mesh, args.sample_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_pose = 0
    missing_observed = 0
    ineligible_observation_count = 0
    explicit_ineligible_input_count = 0
    explicit_eligible_input_count = 0
    eligibility_unspecified_input_count = 0
    ineligible_observation_frames: list[int] = []
    ineligible_reason_counts: dict[str, int] = {}
    explicit_eligible_fit_count = 0
    eligibility_unspecified_fit_count = 0
    ineligible_override_fit_count = 0
    fit_quality_eligible_count = 0
    fit_quality_ineligible_count = 0
    fit_quality_ineligible_frames: list[int] = []
    for frame in annotations.get("frames", []):
        frame_idx = int(frame.get("frame_idx"))
        obj = None
        for candidate in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            if candidate.get("object_id") == args.object_id:
                obj = candidate
                break
        if obj is None:
            continue
        geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        eligibility, eligibility_reasons, eligibility_sources = rigid_pose_observation_eligibility(obj, geom)
        observed = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=float)
        if eligibility is True:
            explicit_eligible_input_count += 1
        elif eligibility is False:
            explicit_ineligible_input_count += 1
        else:
            eligibility_unspecified_input_count += 1
        if eligibility is False and not args.include_ineligible_rigid_pose_observations:
            ineligible_observation_count += 1
            ineligible_observation_frames.append(frame_idx)
            for reason in eligibility_reasons or ["explicit_false_without_reason"]:
                ineligible_reason_counts[reason] = ineligible_reason_counts.get(reason, 0) + 1
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "status": "rigid_pose_observation_ineligible",
                    "object_id": args.object_id,
                    "rigid_pose_observation_eligible": False,
                    "rigid_pose_observation_reasons": eligibility_reasons,
                    "rigid_pose_observation_eligibility_sources": eligibility_sources,
                    "visible_sample_count": int(len(observed)) if observed.ndim == 2 else 0,
                    "policy": "explicit P09 false is a hard measurement-rejection gate",
                }
            )
            continue
        pose = obj.get("reconstructed_geometry_pose") if isinstance(obj.get("reconstructed_geometry_pose"), dict) else {}
        r = np.asarray(pose.get("rotation_world_from_canonical_matrix") or [], dtype=float)
        t = np.asarray(pose.get("translation_world_m") or [], dtype=float)
        if r.shape != (3, 3) or t.shape != (3,):
            missing_pose += 1
            rows.append({"frame_idx": frame_idx, "status": "missing_initial_graph_pose"})
            continue
        if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 3:
            missing_observed += 1
            rows.append({
                "frame_idx": frame_idx,
                "status": "no_current_visible_depth_samples_pose_carried_from_graph",
                "rotation_world_from_completed_canonical_matrix": r.astype(float).tolist(),
                "translation_world_m": t.astype(float).tolist(),
            })
            continue
        fit = fit_frame_pose(canonical_samples, observed, r, t, args.iterations)
        final_median = fit["observed_to_mesh_final"].get("median_m")
        final_p90 = fit["observed_to_mesh_final"].get("p90_m")
        fit_quality_eligible = bool(
            isinstance(final_median, (int, float))
            and isinstance(final_p90, (int, float))
            and np.isfinite(float(final_median))
            and np.isfinite(float(final_p90))
            and float(final_median) <= quality_thresholds["observed_to_mesh_final_median_threshold_m"]
            and float(final_p90) <= quality_thresholds["observed_to_mesh_final_p90_threshold_m"]
        )
        if fit_quality_eligible:
            fit_quality_eligible_count += 1
        else:
            fit_quality_ineligible_count += 1
            fit_quality_ineligible_frames.append(frame_idx)
        if eligibility is True:
            explicit_eligible_fit_count += 1
        elif eligibility is None:
            eligibility_unspecified_fit_count += 1
        else:
            ineligible_override_fit_count += 1
        fit.update({
            "frame_idx": frame_idx,
            "status": "fit_to_visible_depth_samples",
            "object_id": args.object_id,
            "visible_sample_count": int(len(observed)),
            "initial_pose_source": pose.get("pose_source"),
            "initial_pose_observation_residual_norm": pose.get("pose_observation_residual_norm"),
            "rigid_pose_observation_eligible": eligibility,
            "rigid_pose_observation_reasons": eligibility_reasons,
            "rigid_pose_observation_eligibility_sources": eligibility_sources,
            "ineligible_override_used": bool(eligibility is False and args.include_ineligible_rigid_pose_observations),
            "rigid_pose_fit_quality_eligible": fit_quality_eligible,
            "rigid_pose_fit_quality_reason": (
                "observed_to_pose_hypothesis_residual_within_scale_relative_median_and_p90_thresholds"
                if fit_quality_eligible
                else "observed_to_pose_hypothesis_residual_exceeds_scale_relative_median_or_p90_threshold"
            ),
            "rigid_pose_fit_quality_thresholds": quality_thresholds,
        })
        rows.append(fit)

    residuals = [row["observed_to_mesh_final"]["median_m"] for row in rows if row.get("status") == "fit_to_visible_depth_samples" and row["observed_to_mesh_final"]["median_m"] is not None]
    report = {
        "method": "fit_v18_compact_rigid_object_pose",
        "status": "ok",
        "claim_scope": "Per-frame pose is initialized from V18 graph SE3 and refit against current visible depth samples using the P13 pose hypothesis mesh. Hidden TRELLIS faces remain pose correspondences only; they do not become collision/sign geometry or create observations by themselves. Explicitly ineligible upstream rows are hard-rejected.",
        "object_id": args.object_id,
        "inputs": {
            "annotations": str(args.annotations),
            "completion_report": str(args.completion_report),
            "completed_mesh": str(mesh_path),
            "mesh_semantics": (
                "pose_hypothesis_mesh_labeled"
                if completion_outputs.get("pose_hypothesis_mesh_labeled")
                else "legacy_completed_mesh_labeled"
            ),
            "collision_eligible_mesh_not_used_for_pose_correspondence": completion_outputs.get("collision_eligible_mesh_labeled"),
            "completion_geometry_readiness": (
                completion.get("geometry_readiness")
                if isinstance(completion.get("geometry_readiness"), dict)
                else {}
            ),
        },
        "sample_count": int(len(canonical_samples)),
        "iterations": int(args.iterations),
        "eligibility_policy": {
            "explicit_false": "included_only_with_override" if args.include_ineligible_rigid_pose_observations else "rejected",
            "missing_field": "allowed_for_legacy_compatibility",
            "resolution": "false from either object or visible_geometry_candidate rejects the observation; otherwise explicit true is recorded",
            "include_ineligible_override": bool(args.include_ineligible_rigid_pose_observations),
        },
        "fit_quality_policy": {
            **quality_thresholds,
            "median_object_diag_fraction": float(args.fit_quality_median_object_diag_fraction),
            "p90_object_diag_fraction": float(args.fit_quality_p90_object_diag_fraction),
            "median_floor_m": float(args.fit_quality_median_floor_m),
            "median_cap_m": float(args.fit_quality_median_cap_m),
            "p90_floor_m": float(args.fit_quality_p90_floor_m),
            "p90_cap_m": float(args.fit_quality_p90_cap_m),
            "explicit_false": "hard rejected by P15/P14b",
            "missing_field": "legacy compatibility only; cannot support P14b hidden-geometry promotion",
            "claim_scope": "Post-fit residual gate over direct visible metric points and the pose hypothesis. Passing is necessary for a trusted pose observation, not sufficient for annotation readiness.",
        },
        "frame_count": len(rows),
        "fit_frame_count": sum(1 for r in rows if r.get("status") == "fit_to_visible_depth_samples"),
        "explicit_eligible_fit_count": explicit_eligible_fit_count,
        "eligibility_unspecified_fit_count": eligibility_unspecified_fit_count,
        "explicit_eligible_input_count": explicit_eligible_input_count,
        "eligibility_unspecified_input_count": eligibility_unspecified_input_count,
        "explicit_ineligible_input_count": explicit_ineligible_input_count,
        "ineligible_observation_count": ineligible_observation_count,
        "ineligible_override_fit_count": ineligible_override_fit_count,
        "fit_quality_eligible_count": fit_quality_eligible_count,
        "fit_quality_ineligible_count": fit_quality_ineligible_count,
        "fit_quality_ineligible_frames": fit_quality_ineligible_frames,
        "ineligible_observation_frames": ineligible_observation_frames,
        "ineligible_reason_counts": ineligible_reason_counts,
        "missing_pose_count": missing_pose,
        "missing_visible_depth_sample_count": missing_observed,
        "final_observed_to_mesh_median_summary_m": {
            "median": float(np.median(residuals)) if residuals else None,
            "p90": float(np.percentile(residuals, 90)) if residuals else None,
            "max": float(np.max(residuals)) if residuals else None,
        },
        "pose_rows": rows,
    }
    out_path = args.output_dir / "v18_compact_rigid_object_pose_fit_report.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "object_id", "frame_count", "fit_frame_count", "explicit_eligible_input_count", "eligibility_unspecified_input_count", "explicit_ineligible_input_count", "explicit_eligible_fit_count", "eligibility_unspecified_fit_count", "ineligible_observation_count", "ineligible_override_fit_count", "fit_quality_eligible_count", "fit_quality_ineligible_count", "fit_quality_ineligible_frames", "missing_pose_count", "missing_visible_depth_sample_count", "final_observed_to_mesh_median_summary_m"]}, indent=2))


if __name__ == "__main__":
    main()
