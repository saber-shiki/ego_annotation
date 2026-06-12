#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_articulation_fit_candidates"
CLAIM = (
    "This artifact fits bounded world-frame circle/hinge residual diagnostics for robust variable part-pair "
    "hypotheses. It transforms part visible-surface centers from metric-depth camera coordinates to V16 world "
    "coordinates using T_world_camera_metric, fits a 3D circle to relative part-center vectors, and records residual "
    "support or rejection. It does not estimate full part SE(3), hidden geometry, contact ownership, or final object pose."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def finite_vec3(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise RuntimeError(f"{label} must be a length-3 list")
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise RuntimeError(f"{label} must be finite length-3")
    return out


def stats(values: Any) -> dict[str, Any]:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return {"count": 0, "median": None, "p05": None, "p95": None, "min": None, "max": None}

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * p / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {"count": len(xs), "median": pct(50.0), "p05": pct(5.0), "p95": pct(95.0), "min": xs[0], "max": xs[-1]}


def finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def camera_to_world(point_camera: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform @ np.r_[point_camera, 1.0])[:3]


def frame_transforms(annotation_path: Path) -> dict[int, np.ndarray]:
    payload = require_dict(load_json(annotation_path), f"annotations {annotation_path}")
    out: dict[int, np.ndarray] = {}
    for raw in require_list(payload.get("frames"), "annotation frames"):
        frame = require_dict(raw, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        camera = require_dict(frame.get("camera"), "frame camera")
        transform = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"frame {frame_idx} has invalid T_world_camera_metric")
        out[frame_idx] = transform
    return out


def part_center_rows(surface_rows: list[dict[str, Any]], transforms: dict[int, np.ndarray]) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    out: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in surface_rows:
        frame_idx = require_int(row.get("frame_idx"), "surface frame_idx")
        transform = transforms.get(frame_idx)
        if transform is None:
            continue
        object_id = str(row.get("object_id"))
        label = str(row.get("part_track_label"))
        mn = finite_vec3(row.get("bbox_camera_min_m"), "bbox_camera_min_m")
        mx = finite_vec3(row.get("bbox_camera_max_m"), "bbox_camera_max_m")
        center_camera = 0.5 * (mn + mx)
        out[object_id][label][frame_idx] = {
            "frame_idx": frame_idx,
            "center_camera_m": center_camera,
            "center_world_m": camera_to_world(center_camera, transform),
            "vertices": row.get("vertices"),
            "faces": row.get("faces"),
            "part_containment_in_object": row.get("part_containment_in_object"),
        }
    return out


def fit_circle_3d(points: np.ndarray) -> dict[str, Any]:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise RuntimeError("circle fit requires at least three 3D points")
    mean = points.mean(axis=0)
    centered = points - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    normal = vt[2]
    projected = centered @ basis
    a = np.column_stack([projected[:, 0], projected[:, 1], np.ones(points.shape[0])])
    b = -(projected[:, 0] ** 2 + projected[:, 1] ** 2)
    d, e, f = np.linalg.lstsq(a, b, rcond=None)[0]
    center_2d = np.asarray([-d / 2.0, -e / 2.0], dtype=np.float64)
    radius_sq = max(0.0, float((d * d + e * e) / 4.0 - f))
    radius = math.sqrt(radius_sq)
    radial_residual = np.abs(np.linalg.norm(projected - center_2d, axis=1) - radius)
    plane_residual = np.abs(centered @ normal)
    angles = np.unwrap(np.arctan2(projected[:, 1] - center_2d[1], projected[:, 0] - center_2d[0]))
    angle_span = float(angles.max() - angles.min()) if angles.size else 0.0
    return {
        "circle_center_world_m": (mean + basis @ center_2d).astype(float).tolist(),
        "circle_plane_normal_world": normal.astype(float).tolist(),
        "circle_radius_m": float(radius),
        "circle_angle_span_deg": float(angle_span * 180.0 / math.pi),
        "radial_residual_m": stats(radial_residual),
        "plane_residual_m": stats(plane_residual),
        "singular_values": [float(v) for v in singular_values.tolist()],
    }


def classify_fit(fit: dict[str, Any], shared_frame_count: int, args: argparse.Namespace) -> tuple[str, list[str]]:
    blockers: list[str] = []
    radial_median = finite_float_or_none(require_dict(fit.get("radial_residual_m"), "radial residual").get("median"))
    radial_p95 = finite_float_or_none(require_dict(fit.get("radial_residual_m"), "radial residual").get("p95"))
    plane_p95 = finite_float_or_none(require_dict(fit.get("plane_residual_m"), "plane residual").get("p95"))
    radius = finite_float_or_none(fit.get("circle_radius_m"))
    angle_span = finite_float_or_none(fit.get("circle_angle_span_deg"))
    if shared_frame_count < int(args.min_shared_frames):
        blockers.append("too_few_shared_part_frames_for_articulation_fit")
    if radius is None or radius < float(args.min_circle_radius_m):
        blockers.append("circle_radius_too_small_for_articulation_fit")
    if angle_span is None or angle_span < float(args.min_angle_span_deg):
        blockers.append("articulation_angle_span_too_small")
    if radial_median is None or radial_median > float(args.max_radial_median_residual_m):
        blockers.append("radial_median_residual_above_threshold")
    if radial_p95 is None or radial_p95 > float(args.max_radial_p95_residual_m):
        blockers.append("radial_p95_residual_above_threshold")
    if plane_p95 is None or plane_p95 > float(args.max_plane_p95_residual_m):
        blockers.append("plane_p95_residual_above_threshold")
    if shared_frame_count < int(args.min_shared_frames):
        return "articulation_fit_underconstrained", blockers
    if blockers:
        return "articulation_fit_residual_rejected", blockers
    return "articulation_fit_residual_supported_visible_center_only_not_pose", blockers


def articulation_probes(model_report: dict[str, Any]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for raw in require_list(model_report.get("rejected_candidates"), "part model rejected candidates"):
        row = require_dict(raw, "candidate row")
        if row.get("candidate_state") == "articulation_hypothesis_not_fitted_no_pose":
            probes.append(row)
    return probes


def fit_probe(probe: dict[str, Any], centers: dict[str, dict[str, dict[int, dict[str, Any]]]], args: argparse.Namespace) -> dict[str, Any]:
    object_id = str(probe.get("object_id"))
    labels = [str(item) for item in require_list(probe.get("part_track_labels"), "part labels")]
    if len(labels) != 2:
        raise RuntimeError(f"{object_id}: articulation probe must have exactly two part labels")
    first = centers.get(object_id, {}).get(labels[0], {})
    second = centers.get(object_id, {}).get(labels[1], {})
    shared = sorted(set(first) & set(second))
    relative_vectors = np.asarray([second[frame]["center_world_m"] - first[frame]["center_world_m"] for frame in shared], dtype=np.float64)
    relative_norms = np.linalg.norm(relative_vectors, axis=1) if relative_vectors.size else np.asarray([], dtype=np.float64)
    adjacent_step = np.linalg.norm(np.diff(relative_vectors, axis=0), axis=1) if relative_vectors.shape[0] > 1 else np.asarray([], dtype=np.float64)
    if relative_vectors.shape[0] >= 3:
        fit = fit_circle_3d(relative_vectors)
        state, blockers = classify_fit(fit, len(shared), args)
    else:
        fit = {
            "circle_center_world_m": None,
            "circle_plane_normal_world": None,
            "circle_radius_m": None,
            "circle_angle_span_deg": None,
            "radial_residual_m": stats([]),
            "plane_residual_m": stats([]),
            "singular_values": [],
        }
        state, blockers = "articulation_fit_underconstrained", ["too_few_shared_part_frames_for_articulation_fit"]
    return {
        "object_id": object_id,
        "source_candidate_id": probe.get("candidate_id"),
        "part_track_labels": labels,
        "fit_type": "world_frame_relative_part_center_circle_fit",
        "fit_scope": "visible_part_surface_centers_only_not_part_pose",
        "coordinate_frame": "V16 T_world_camera_metric world frame",
        "shared_frame_count": len(shared),
        "frame_min": min(shared) if shared else None,
        "frame_max": max(shared) if shared else None,
        "relative_center_distance_m": stats(relative_norms),
        "adjacent_relative_vector_step_m": stats(adjacent_step),
        **fit,
        "articulation_fit_state": state,
        "fit_blockers": blockers,
        "acceptance_thresholds": {
            "min_shared_frames": int(args.min_shared_frames),
            "min_circle_radius_m": float(args.min_circle_radius_m),
            "min_angle_span_deg": float(args.min_angle_span_deg),
            "max_radial_median_residual_m": float(args.max_radial_median_residual_m),
            "max_radial_p95_residual_m": float(args.max_radial_p95_residual_m),
            "max_plane_p95_residual_m": float(args.max_plane_p95_residual_m),
        },
        "articulation_model_ready": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.part_model_candidates_root / case / "v18_part_model_candidates_report.json"
    surfaces_path = args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"
    annotation_path = args.v16_root / case / "annotations_v16_full.json"
    model_report = require_dict(load_json(model_path), f"{case} part model candidates")
    surfaces_report = require_dict(load_json(surfaces_path), f"{case} part surfaces")
    transforms = frame_transforms(annotation_path)
    surface_rows = [require_dict(raw, "surface row") for raw in require_list(surfaces_report.get("surface_rows"), "surface_rows")]
    centers = part_center_rows(surface_rows, transforms)
    rows = [fit_probe(probe, centers, args) for probe in articulation_probes(model_report)]
    state_counts = Counter(str(row.get("articulation_fit_state")) for row in rows)
    report = {
        "method": "build_v18_articulation_fit_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "part_model_candidates": str(model_path),
            "part_visible_surfaces": str(surfaces_path),
            "v16_annotations": str(annotation_path),
        },
        "articulation_fit_probe_count": len(rows),
        "articulation_fit_state_counts": dict(sorted(state_counts.items())),
        "articulation_fit_supported_count": state_counts.get("articulation_fit_residual_supported_visible_center_only_not_pose", 0),
        "articulation_fit_rejected_count": state_counts.get("articulation_fit_residual_rejected", 0),
        "articulation_fit_underconstrained_count": state_counts.get("articulation_fit_underconstrained", 0),
        "rows": rows,
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_articulation_fit_candidates_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(report["articulation_fit_state_counts"])
    summary = {
        "method": "build_v18_articulation_fit_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "articulation_fit_probe_count": sum(require_int(report.get("articulation_fit_probe_count"), "probe count") for report in reports),
        "articulation_fit_state_counts": dict(sorted(state_counts.items())),
        "articulation_fit_supported_count": state_counts.get("articulation_fit_residual_supported_visible_center_only_not_pose", 0),
        "articulation_fit_rejected_count": state_counts.get("articulation_fit_residual_rejected", 0),
        "articulation_fit_underconstrained_count": state_counts.get("articulation_fit_underconstrained", 0),
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_articulation_fit_candidates_report.json"),
                "articulation_fit_probe_count": report["articulation_fit_probe_count"],
                "articulation_fit_state_counts": report["articulation_fit_state_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_articulation_fit_candidates_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-model-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_model_candidates"))
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_articulation_fit_candidates"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--min-shared-frames", type=int, default=20)
    parser.add_argument("--min-circle-radius-m", type=float, default=0.02)
    parser.add_argument("--min-angle-span-deg", type=float, default=15.0)
    parser.add_argument("--max-radial-median-residual-m", type=float, default=0.02)
    parser.add_argument("--max-radial-p95-residual-m", type=float, default=0.06)
    parser.add_argument("--max-plane-p95-residual-m", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
