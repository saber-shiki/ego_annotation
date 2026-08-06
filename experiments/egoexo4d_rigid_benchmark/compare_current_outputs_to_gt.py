#!/usr/bin/env python3
"""Compare frozen and pose-gated V19 outputs with the available Ego-Exo4D GT.

The released shard has sparse visible-object masks, named hand joints, and a
camera trajectory.  It does not have object CAD/SE(3), MANO surfaces, contact,
or signed nonpenetration GT.  This evaluator therefore keeps a strict split:

* direct GT metrics: visible masks, named hand joints, camera trajectory;
* raw-view diagnostic: projected completed-mesh silhouette versus sparse masks;
* mechanism audit: frozen versus pose-gated reports without inventing GT.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from render_v19_rigid_state_artifact import (  # noqa: E402
    load_mesh,
    project_camera_points,
    scaled_intrinsics_for_frame,
    world_points_to_camera,
)

from evaluate_benchmark import (  # noqa: E402
    bbox_iou,
    bbox_xyxy,
    boundary_f1,
    camera_map,
    compare_hand_reports,
    evaluate_camera_trajectory,
    evaluate_hand_joint_map,
    evaluate_masks,
    finite_summary,
    load_hawor_joint_map,
    load_interval_joint_map,
    load_json,
    sha256_file,
    write_json,
)

POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "corrected_temporal_rigid_pose_graph",
    "completed_temporal_rigid_pose_uncertain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--fixed-p14-report", type=Path, required=True)
    parser.add_argument("--fixed-p15-report", type=Path, required=True)
    parser.add_argument("--fixed-p16-report", type=Path, required=True)
    parser.add_argument("--fixed-p18-state", type=Path, required=True)
    parser.add_argument("--fixed-p18b-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", default="tire_lever")
    parser.add_argument("--min-num-views", type=int, default=2)
    parser.add_argument("--max-gt-reprojection-px-rectified-512", type=float, default=20.0)
    parser.add_argument("--boundary-tolerance-px", type=int, default=5)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def pose_map(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in report.get("pose_rows", []) if isinstance(report.get("pose_rows"), list) else []:
        if not isinstance(row, dict) or row.get("status") not in POSE_STATUSES:
            continue
        rotation = np.asarray(row.get("rotation_world_from_completed_canonical_matrix") or [], dtype=np.float64)
        translation = np.asarray(row.get("translation_world_m") or [], dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            continue
        rows[int(row["frame_idx"])] = {
            "rotation": rotation,
            "translation": translation,
            "status": row.get("status"),
            "annotation_ready": row.get("annotation_ready"),
            "graph_support_sufficient": row.get("graph_support_sufficient"),
            "temporal_pose_graph": row.get("temporal_pose_graph"),
        }
    return rows


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def trajectory_comparison(frozen_report: dict[str, Any], fixed_report: dict[str, Any]) -> dict[str, Any]:
    frozen = pose_map(frozen_report)
    fixed = pose_map(fixed_report)
    common = sorted(set(frozen).intersection(fixed))
    translation_delta_m: list[float] = []
    rotation_delta_deg: list[float] = []
    rows: list[dict[str, Any]] = []
    for frame_idx in common:
        frozen_row = frozen[frame_idx]
        fixed_row = fixed[frame_idx]
        t_delta = float(np.linalg.norm(fixed_row["translation"] - frozen_row["translation"]))
        r_delta = rotation_angle_deg(fixed_row["rotation"] @ frozen_row["rotation"].T)
        translation_delta_m.append(t_delta)
        rotation_delta_deg.append(r_delta)
        rows.append(
            {
                "frame_idx": frame_idx,
                "translation_delta_mm": t_delta * 1000.0,
                "rotation_delta_deg": r_delta,
                "frozen_status": frozen_row["status"],
                "fixed_status": fixed_row["status"],
                "fixed_annotation_ready": fixed_row["annotation_ready"],
                "fixed_graph_support_sufficient": fixed_row["graph_support_sufficient"],
            }
        )
    return {
        "status": "compared" if common else "no_common_pose_rows",
        "claim_scope": "Frozen-versus-fixed state delta only; no object-pose GT exists, so smaller deltas are not accuracy evidence.",
        "common_frame_count": len(common),
        "translation_delta_mm": finite_summary(translation_delta_m, 1000.0),
        "rotation_delta_deg": finite_summary(rotation_delta_deg),
        "frozen_status_counts": dict(Counter(row["status"] for row in frozen.values())),
        "fixed_status_counts": dict(Counter(row["status"] for row in fixed.values())),
        "fixed_graph_support": fixed_report.get("graph_support"),
        "fixed_annotation_ready": fixed_report.get("annotation_ready"),
        "per_frame": rows,
    }


def projected_mesh_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    pose: dict[str, Any],
    frame: dict[str, Any],
    raw_video: dict[str, Any] | None,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices_world = vertices @ pose["rotation"].T + pose["translation"][None, :]
    transform = np.asarray((frame.get("camera") or {}).get("T_world_camera_metric") or [], dtype=np.float64)
    if transform.shape != (4, 4):
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks T_world_camera_metric")
    intrinsics, projection_report = scaled_intrinsics_for_frame(frame, width, height, raw_video)
    vertices_camera = world_points_to_camera(vertices_world, transform)
    u, v, z, _ = project_camera_points(vertices_camera, intrinsics, width, height)
    uv = np.c_[u, v]
    triangles = uv[faces]
    face_z = z[faces]
    valid = np.all(face_z > 0.01, axis=1) & np.isfinite(triangles).all(axis=(1, 2))
    triangles = triangles[valid]
    intersects = (
        (triangles[:, :, 0].max(axis=1) >= 0)
        & (triangles[:, :, 0].min(axis=1) < width)
        & (triangles[:, :, 1].max(axis=1) >= 0)
        & (triangles[:, :, 1].min(axis=1) < height)
    )
    triangles = triangles[intersects]
    polygons = np.rint(
        np.clip(triangles, np.asarray([-2 * width, -2 * height]), np.asarray([3 * width, 3 * height]))
    ).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    for start in range(0, len(polygons), 20000):
        cv2.fillPoly(mask, list(polygons[start : start + 20000]), 255, lineType=cv2.LINE_8)
    return mask > 0, {
        "candidate_face_count": int(len(faces)),
        "front_finite_intersecting_face_count": int(len(polygons)),
        "projection_contract": projection_report,
    }


def mask_metrics(prediction: np.ndarray, ground_truth: np.ndarray, boundary_tolerance_px: int) -> dict[str, Any]:
    intersection = int(np.count_nonzero(prediction & ground_truth))
    union = int(np.count_nonzero(prediction | ground_truth))
    pred_area = int(np.count_nonzero(prediction))
    gt_area = int(np.count_nonzero(ground_truth))
    pred_center = np.asarray(np.where(prediction)[::-1], dtype=np.float64).mean(axis=1) if pred_area else None
    gt_center = np.asarray(np.where(ground_truth)[::-1], dtype=np.float64).mean(axis=1) if gt_area else None
    boundary_precision, boundary_recall, boundary_score = boundary_f1(prediction, ground_truth, boundary_tolerance_px)
    return {
        "gt_area_px": gt_area,
        "prediction_area_px": pred_area,
        "intersection_px": intersection,
        "union_px": union,
        "iou": intersection / union if union else 1.0,
        "precision": intersection / pred_area if pred_area else 0.0,
        "recall": intersection / gt_area if gt_area else 0.0,
        "bbox_iou": bbox_iou(bbox_xyxy(prediction), bbox_xyxy(ground_truth)),
        "centroid_error_px": float(np.linalg.norm(pred_center - gt_center)) if pred_center is not None and gt_center is not None else None,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_score,
    }


def color_mask_overlay(image: np.ndarray, gt: np.ndarray, prediction: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    layer = image.copy()
    layer[gt] = (40, 210, 40)
    layer[prediction] = color
    layer[gt & prediction] = (245, 245, 245)
    result = cv2.addWeighted(image, 0.42, layer, 0.58, 0)
    cv2.rectangle(result, (0, 0), (result.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(result, label, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def projected_mesh_vs_sparse_gt(
    annotations_path: Path,
    completed_mesh_path: Path,
    gt_dir: Path,
    pose_reports: dict[str, dict[str, Any]],
    output_dir: Path,
    boundary_tolerance_px: int,
) -> dict[str, Any]:
    annotations = load_json(annotations_path)
    frames = {int(row["frame_idx"]): row for row in annotations["frames"]}
    raw_video = annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else None
    vertices, faces, mesh_summary = load_mesh(completed_mesh_path)
    poses = {label: pose_map(report) for label, report in pose_reports.items()}
    gt_report = load_json(gt_dir / "object_visible_mask_gt.json")
    gt_mask_dir = gt_dir / f"object_visible_masks_{gt_report['frames'][0]['runtime_grid'][0]}"
    method_rows: dict[str, list[dict[str, Any]]] = {label: [] for label in pose_reports}
    visualization_rows: list[np.ndarray] = []
    mask_root = output_dir / "projected_mesh_masks"

    for gt_row in gt_report["frames"]:
        frame_idx = int(gt_row["local_frame_idx"])
        frame = frames[frame_idx]
        gt_path = gt_mask_dir / gt_row["runtime_mask_file"]
        gt_image = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        raw_image = cv2.imread(str(frame["raw_frame_path"]), cv2.IMREAD_COLOR)
        if gt_image is None or raw_image is None:
            raise FileNotFoundError(gt_path if gt_image is None else frame["raw_frame_path"])
        if raw_image.shape[:2] != gt_image.shape:
            raw_image = cv2.resize(raw_image, (gt_image.shape[1], gt_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        gt_mask = gt_image > 0
        panels: list[np.ndarray] = []
        contour_image = raw_image.copy()
        gt_contours, _ = cv2.findContours(gt_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image, gt_contours, -1, (40, 255, 40), 3, cv2.LINE_AA)
        contour_labels = [f"f{frame_idx} green=GT"]
        method_colors = {"frozen_v1": (220, 40, 220), "pose_gate_fixed": (0, 165, 255)}
        for label, pose_by_frame in poses.items():
            pose = pose_by_frame.get(frame_idx)
            if pose is None:
                prediction = np.zeros_like(gt_mask)
                projection = {"status": "missing_pose"}
            else:
                prediction, projection = projected_mesh_mask(
                    vertices,
                    faces,
                    pose,
                    frame,
                    raw_video,
                    gt_image.shape[1],
                    gt_image.shape[0],
                )
            method_dir = mask_root / label
            method_dir.mkdir(parents=True, exist_ok=True)
            mask_path = method_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(mask_path), prediction.astype(np.uint8) * 255)
            metrics = mask_metrics(prediction, gt_mask, boundary_tolerance_px)
            method_rows[label].append(
                {
                    "local_frame_idx": frame_idx,
                    "source_frame_idx": int(gt_row["source_frame_idx"]),
                    "gt_mask": str(gt_path),
                    "projected_mesh_mask": str(mask_path),
                    "pose_status": pose.get("status") if pose is not None else None,
                    "pose_annotation_ready": pose.get("annotation_ready") if pose is not None else None,
                    "graph_support_sufficient": pose.get("graph_support_sufficient") if pose is not None else None,
                    "projection": projection,
                    **metrics,
                }
            )
            pred_contours, _ = cv2.findContours(prediction.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = method_colors.get(label, (255, 255, 0))
            cv2.drawContours(contour_image, pred_contours, -1, color, 2, cv2.LINE_AA)
            contour_labels.append(f"{label}={metrics['iou']:.3f}")
            panels.append(
                cv2.resize(
                    color_mask_overlay(
                        raw_image,
                        gt_mask,
                        prediction,
                        f"f{frame_idx} {label} IoU={metrics['iou']:.3f}",
                        color,
                    ),
                    (480, 480),
                    interpolation=cv2.INTER_AREA,
                )
            )
        cv2.rectangle(contour_image, (0, 0), (contour_image.shape[1], 58), (0, 0, 0), -1)
        cv2.putText(contour_image, " | ".join(contour_labels), (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        panels.insert(0, cv2.resize(contour_image, (480, 480), interpolation=cv2.INTER_AREA))
        visualization_rows.append(np.concatenate(panels, axis=1))

    visualization_path = output_dir / "projected_mesh_vs_sparse_gt.jpg"
    if visualization_rows:
        cv2.imwrite(str(visualization_path), np.concatenate(visualization_rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 94])

    methods: dict[str, Any] = {}
    for label, rows in method_rows.items():
        methods[label] = {
            "aggregate": {
                "iou": finite_summary([float(row["iou"]) for row in rows]),
                "precision": finite_summary([float(row["precision"]) for row in rows]),
                "recall": finite_summary([float(row["recall"]) for row in rows]),
                "bbox_iou": finite_summary([float(row["bbox_iou"]) for row in rows]),
                "centroid_error_px": finite_summary(
                    [float(row["centroid_error_px"]) for row in rows if row["centroid_error_px"] is not None]
                ),
                "boundary_f1": finite_summary([float(row["boundary_f1"]) for row in rows]),
            },
            "per_frame": rows,
        }
    return {
        "status": "scored_raw_view_diagnostic",
        "claim_scope": (
            "Projected completed-mesh silhouette versus five sparse raw-grid visible masks. This jointly tests runtime camera approximation, "
            "completed geometry, pose, and visible-occlusion mismatch; it is not object-SE(3) GT and cannot isolate any one error source."
        ),
        "important_caveats": [
            "Runtime projection uses an estimated pinhole over distorted raw Aria RGB because VRS calibration is unavailable.",
            "The projected completed mesh is amodal while released relation masks are visible masks and may exclude hand-occluded pixels.",
            "The same erroneous completed mesh is used for both trajectories, so the comparison does not validate canonical geometry.",
        ],
        "gt_frame_count": sum(len(rows) for rows in method_rows.values()) // max(1, len(method_rows)),
        "mesh": mesh_summary,
        "methods": methods,
        "visualization": str(visualization_path),
    }


def exact_joint_delta(reference: dict[tuple[int, str], np.ndarray], candidate: dict[tuple[int, str], np.ndarray]) -> dict[str, Any]:
    common = sorted(set(reference).intersection(candidate))
    values = [float(np.max(np.linalg.norm(candidate[key] - reference[key], axis=1))) for key in common]
    return {
        "reference_row_count": len(reference),
        "candidate_row_count": len(candidate),
        "common_row_count": len(common),
        "missing_candidate_rows": len(set(reference).difference(candidate)),
        "max_per_row_joint_delta_mm": finite_summary(values, 1000.0),
        "exact_within_1e_9_m": bool(values and max(values) <= 1.0e-9),
    }


def hand_and_camera_evaluation(
    run_root: Path,
    gt_dir: Path,
    frozen_p18: Path,
    frozen_p18b: Path,
    fixed_p18: Path,
    fixed_p18b: Path,
    min_num_views: int,
    max_gt_reprojection: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    hawor_path = run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz"
    hawor_joints, camera_payload = load_hawor_joint_map(hawor_path)
    cameras = camera_map(camera_payload)
    gt_hand = load_json(gt_dir / "hand_pose_gt.json")
    gt_camera = load_json(gt_dir / "camera_pose_gt.json")
    maps = {
        "hawor_baseline": hawor_joints,
        "frozen_p18_raw": load_interval_joint_map(frozen_p18),
        "frozen_p18b_canonical": load_interval_joint_map(frozen_p18b),
        "fixed_p18_quarantine": load_interval_joint_map(fixed_p18),
        "fixed_p18b_quarantine": load_interval_joint_map(fixed_p18b),
    }
    reports = {
        label: evaluate_hand_joint_map(
            label,
            joint_map,
            cameras,
            gt_hand,
            gt_camera,
            min_num_views,
            max_gt_reprojection,
        )
        for label, joint_map in maps.items()
    }
    exact_deltas = {label: exact_joint_delta(hawor_joints, joint_map) for label, joint_map in maps.items() if label != "hawor_baseline"}
    return reports, compare_hand_reports(reports), {
        "candidate_vs_hawor_exact_delta": exact_deltas,
        "camera_trajectory": evaluate_camera_trajectory(camera_payload, gt_camera),
    }


def event_runtime_report(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stamped = [row for row in rows if isinstance(row.get("timestamp_utc"), str)]
    if len(stamped) < 2:
        return {"status": "not_available"}
    start = datetime.fromisoformat(stamped[0]["timestamp_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(stamped[-1]["timestamp_utc"].replace("Z", "+00:00"))
    elapsed = (end - start).total_seconds()
    return {
        "status": "measured_from_first_and_last_harness_timestamp",
        "start": stamped[0]["timestamp_utc"],
        "end": stamped[-1]["timestamp_utc"],
        "elapsed_s": elapsed,
        "input_duration_s": 5.0,
        "realtime_ratio": elapsed / 5.0,
        "first_event": stamped[0].get("event"),
        "last_event": stamped[-1].get("event"),
    }


def mechanism_evidence(args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    frozen_p14 = load_json(run_root / "measurements" / "pose_fits" / f"{args.object_id}_visible_pose_fit" / "v18_compact_rigid_object_pose_fit_report.json")
    frozen_p15 = load_json(run_root / "measurements" / "pose_fits" / f"{args.object_id}_rigid_pose_graph" / "v19_rigid_object_pose_graph_report.json")
    frozen_p16 = load_json(run_root / "measurements" / "contact_nonpenetration" / f"{args.object_id}_mano_object_constraint" / "v18_mano_object_constraint_state.json")
    fixed_p14 = load_json(args.fixed_p14_report)
    fixed_p15 = load_json(args.fixed_p15_report)
    fixed_p16 = load_json(args.fixed_p16_report)
    completion = load_json(run_root / "measurements" / "geometry_completion" / f"compact_{args.object_id}_seed42" / "v18_compact_rigid_trellis_completion_report.json")
    p13_faces = completion.get("face_label_counts", {})
    frozen_p18_path = next(
        path
        for path in (run_root / "measurements" / "mano_interval_correction").glob(
            f"{args.object_id}_0_149/*/v18_joint_mano_interval_trajectory_state.json"
        )
    )
    frozen_p18 = load_json(frozen_p18_path)
    p18_by_side = {str(row.get("hand_side")): row for row in frozen_p18.get("intervals", [])}
    return {
        "p09_to_p14": {
            "frozen_fit_count": frozen_p14.get("fit_frame_count"),
            "fixed_fit_count": fixed_p14.get("fit_frame_count"),
            "fixed_explicit_ineligible_rejected": fixed_p14.get("ineligible_observation_count"),
            "fixed_missing_initial_pose_count": fixed_p14.get("missing_pose_count"),
            "frozen_final_observed_to_mesh_median_m": (frozen_p14.get("final_observed_to_mesh_median_summary_m") or {}).get("median"),
            "fixed_final_observed_to_mesh_median_m": (fixed_p14.get("final_observed_to_mesh_median_summary_m") or {}).get("median"),
        },
        "p15": {
            "frozen": {
                "status": frozen_p15.get("status"),
                "annotation_ready": frozen_p15.get("annotation_ready"),
                "graph_frame_count": frozen_p15.get("graph_frame_count"),
                "nonpenetration_target_frame_count": frozen_p15.get("nonpenetration_target_frame_count"),
                "optimizer": frozen_p15.get("optimizer"),
                "completion": frozen_p15.get("full_timeline_rigid_pose_completion"),
            },
            "fixed": {
                "status": fixed_p15.get("status"),
                "annotation_ready": fixed_p15.get("annotation_ready"),
                "graph_frame_count": fixed_p15.get("graph_frame_count"),
                "graph_support": fixed_p15.get("graph_support"),
                "completion": fixed_p15.get("full_timeline_rigid_pose_completion"),
            },
        },
        "p13_geometry": {
            "mesh_counts": completion.get("mesh_counts"),
            "face_label_counts": p13_faces,
            "silhouette_free_space_filter": completion.get("silhouette_free_space_filter"),
            "planar_slab_support_filter": completion.get("planar_slab_support_filter"),
        },
        "p16": {
            "frozen": {
                "surface_watertight": frozen_p16.get("completed_surface_mesh_watertight"),
                "sign_watertight": frozen_p16.get("sign_mesh_watertight"),
                "candidate_correction_count": frozen_p16.get("candidate_correction_count"),
            },
            "fixed": {
                "status": fixed_p16.get("status"),
                "physical_constraint_quarantined": fixed_p16.get("physical_constraint_quarantined"),
                "candidate_correction_count": fixed_p16.get("candidate_correction_count"),
            },
        },
        "p18_frozen": {
            side: {
                "optimizer_ran": row.get("optimizer_ran"),
                "translation_delta_norm_m": row.get("translation_delta_norm_m"),
                "output_translation_gate_applied_count": row.get("output_translation_gate_applied_count"),
                "output_translation_gate_selected_support_count": row.get("output_translation_gate_selected_support_count"),
                "active_set_closed": row.get("active_set_closed"),
                "contact_patch_factor_active_row_count": row.get("contact_patch_factor_active_row_count"),
                "contact_patch_anchor_coherence": row.get("contact_patch_anchor_coherence"),
            }
            for side, row in p18_by_side.items()
        },
        "p18_fixed": {
            "status": load_json(args.fixed_p18_state).get("status"),
            "optimization_skipped": load_json(args.fixed_p18_state).get("optimization_skipped"),
            "physical_state_quarantined": load_json(args.fixed_p18_state).get("physical_state_quarantined"),
        },
        "runtime": event_runtime_report(run_root / "logs" / "harness_events.jsonl"),
    }


def failure_taxonomy(report: dict[str, Any]) -> dict[str, Any]:
    mechanism = report["mechanism_evidence"]
    silhouette = report["projected_mesh_visible_mask_diagnostic"]["methods"]
    hand = report["hand_pose"]
    camera = report["camera_trajectory"]
    return {
        "status": "classified_from_partial_gt_and_mechanism_evidence",
        "classification_rule": {
            "confirmed_bug": "Implementation violated its own recorded trust/readiness/provenance contract and the same cached inputs are handled correctly by a code-only fix.",
            "pipeline_design_problem": "Code behaved as implemented, but the objective, phase graph, geometry representation, or acceptance policy is inadequate for the physical goal; redesign or new evidence is required.",
            "current_input_or_gt_limit": "The requested state is unobservable or unscorable from the released sensors/labels; a bug-free implementation cannot manufacture the missing information.",
        },
        "confirmed_implementation_bugs": [
            {
                "id": "BUG-01-p09-p14-eligibility-disconnect",
                "severity": "critical",
                "status": "fixed_in_b7b97e6",
                "evidence": mechanism["p09_to_p14"],
                "consequence": "27 explicitly rejected rows entered the frozen P14/P15 measurement set.",
            },
            {
                "id": "BUG-02-low-support-readiness-semantics",
                "severity": "high",
                "status": "fixed_in_b7b97e6",
                "evidence": mechanism["p15"],
                "consequence": "Sparse interpolation/nearest hold could be confused with observed trajectory coverage; fixed output is annotation_ready=false.",
            },
            {
                "id": "BUG-03-downstream-unready-state-consumption",
                "severity": "high",
                "status": "fixed_in_b7b97e6",
                "evidence": {
                    "fixed_p16": mechanism["p16"]["fixed"],
                    "fixed_p18": mechanism["p18_fixed"],
                },
                "consequence": "P16/P18/render could otherwise compute or display object-relative physical states from an unready object trajectory.",
            },
            {
                "id": "BUG-04-render-selected-frame-provenance",
                "severity": "low",
                "status": "fixed_in_b7b97e6",
                "evidence": "render manifest now records source-frame IDs and output-index mapping",
                "consequence": "Selected-frame artifact filenames previously encoded output order rather than source-frame identity without a manifest mapping.",
            },
            {
                "id": "BUG-05-stale-top-level-runtime-state",
                "severity": "medium",
                "status": "unresolved_provenance_bug",
                "evidence": "Frozen state/v19_physical_state.json and v19_uncertainty_state.json remained at P00 while renderer consumed later state.",
                "consequence": "Consumers can read a stale top-level state unless they know the renderer boundary and event provenance.",
            },
        ],
        "pipeline_design_or_architecture_problems": [
            {
                "id": "DESIGN-01-p15-before-p16-no-feedback-pass",
                "evidence": mechanism["p15"]["frozen"],
                "interpretation": "P15 had zero nonpenetration targets and stopped at x=0; P16 runs later and there is no required second P15 pass.",
            },
            {
                "id": "DESIGN-02-hidden-prior-dominates-completed-mesh",
                "evidence": mechanism["p13_geometry"],
                "interpretation": "Only 496 completed faces are observed-depth surface while 151335 are hidden prior; silhouette/slab gates rejected zero faces.",
            },
            {
                "id": "DESIGN-03-partial-view-icp-not-a-full-pose-certificate",
                "evidence": {
                    "frozen_projected_mesh_iou": silhouette["frozen_v1"]["aggregate"]["iou"],
                    "fixed_projected_mesh_iou": silhouette["pose_gate_fixed"]["aggregate"]["iou"],
                },
                "interpretation": "Low visible-surface residual does not imply that completed geometry and pose project onto the visible object.",
            },
            {
                "id": "DESIGN-04-p18-soft-solve-then-global-output-gate",
                "evidence": mechanism["p18_frozen"],
                "interpretation": "The optimizer can make large unsupported translations and then reset every row, instead of freezing unobservable variables inside the solve.",
            },
            {
                "id": "DESIGN-05-contact-has-no-stable-object-frame-anchor",
                "evidence": mechanism["p18_frozen"],
                "interpretation": "Semantic likely-contact priors do not create a metric contact patch or an object-frame anchor.",
            },
            {
                "id": "DESIGN-06-runtime-cost",
                "evidence": mechanism["runtime"],
                "interpretation": "The full mechanism is hundreds of times slower than the five-second input and needs profiling/vectorization, not reduced-frame accounting.",
            },
        ],
        "suspected_but_not_yet_proven_implementation_bugs": [
            {
                "id": "SUSPECT-01-p18-source-grid-factor-mask-grid",
                "evidence": "Audit found separate source-grid projection and factor-mask lookup paths; frozen P18 selected zero depth-order vertices.",
                "why_not_confirmed": "Zero selected vertices can also be caused by absent geometric support. A synthetic source-size-to-mask-size projection regression and real before/after remeasurement are still required.",
            },
            {
                "id": "SUSPECT-02-p13-filter-zero-rejection",
                "evidence": "Both silhouette and planar-slab filters rejected zero TRELLIS faces while the rendered shape is visibly wrong.",
                "why_not_confirmed": "This proves the acceptance policy was non-discriminative on this case, but not whether the implementation is mathematically wrong versus the thresholds/input prior being inadequate.",
            },
        ],
        "current_input_observability_or_ground_truth_limits": [
            {
                "id": "LIMIT-01-no-object-cad-or-se3-gt",
                "consequence": "Object canonical geometry and 6-DoF accuracy cannot be directly scored; projected silhouette is only a confounded diagnostic.",
            },
            {
                "id": "LIMIT-02-no-contact-or-signed-surface-gt",
                "consequence": "Metric contact and nonpenetration cannot be certified from narration, masks, or a non-watertight predicted mesh.",
            },
            {
                "id": "LIMIT-03-missing-aria-vrs-calibration",
                "consequence": "Raw distorted RGB cannot be exactly mapped to the released rectified camera; raw-view mesh projection remains an estimated-pinhole diagnostic.",
            },
            {
                "id": "LIMIT-04-monocular-scale-depth-and-occlusion-ambiguity",
                "evidence": {
                    "camera_se3_ate_rmse_mm": camera["alignments"]["se3_metric"]["ate_center_rmse_mm"],
                    "camera_sim3_scale": camera["alignments"]["sim3_diagnostic"]["scale"],
                    "hand_absolute_mpjpe_mm": hand["hawor_baseline"]["all_hands"]["absolute_joint_error_mm"]["mean"],
                    "hand_root_relative_mpjpe_mm": hand["hawor_baseline"]["all_hands"]["root_relative_joint_error_mm"]["mean"],
                },
                "consequence": "Single-view RGB cannot uniquely determine metric camera/hand/object translation through occlusion without stronger calibration or multi-view/depth evidence.",
            },
            {
                "id": "LIMIT-05-sparse-released-gt",
                "consequence": "Object masks cover 5/150 frames and hand joints 49/150 annotated frames; full-timeline failure modes remain only partially scored.",
            },
        ],
        "separate_benchmark_bug_already_fixed": {
            "id": "EVAL-01-rectified-k-treated-as-raw-k",
            "status": "fixed_and_old_metrics_invalidated",
            "scope": "This was an evaluator/coordinate-contract error, not evidence that the prediction pipeline consumed released raw sensor calibration.",
        },
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    gt_dir = args.ground_truth_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.replace:
            raise FileExistsError(f"output exists: {output_dir}; pass --replace")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    benchmark_manifest = load_json(gt_dir / "BENCHMARK_MANIFEST.json")
    case_id = str(benchmark_manifest["case_id"])
    benchmark_video = Path(benchmark_manifest["prediction_video"]["path"])
    runtime_input_contract_path = run_root / "input" / "runtime_input_contract.json"
    runtime_input_contract = load_json(runtime_input_contract_path) if runtime_input_contract_path.exists() else {}
    runtime_video = Path(runtime_input_contract["input_video"]) if runtime_input_contract.get("input_video") else None
    benchmark_hash = sha256_file(benchmark_video) if benchmark_video.exists() else None
    runtime_hash = sha256_file(runtime_video) if runtime_video is not None and runtime_video.exists() else None
    input_identity = {
        "status": "content_identical" if benchmark_hash is not None and runtime_hash == benchmark_hash else "content_mismatch_or_missing",
        "benchmark_video": str(benchmark_video),
        "benchmark_sha256": benchmark_hash,
        "manifest_declared_sha256": benchmark_manifest["prediction_video"].get("sha256"),
        "runtime_video": str(runtime_video) if runtime_video is not None else None,
        "runtime_sha256": runtime_hash,
        "byte_content_identical": benchmark_hash is not None and runtime_hash == benchmark_hash,
    }
    annotations_path = run_root / "measurements" / "object_geometry" / "visible_geometry" / args.object_id / "annotations_v19_visible_geometry.json"
    completion_report = load_json(
        run_root / "measurements" / "geometry_completion" / f"compact_{args.object_id}_seed42" / "v18_compact_rigid_trellis_completion_report.json"
    )
    completed_mesh = Path(completion_report["outputs"]["completed_mesh_labeled"])
    frozen_p15_path = run_root / "measurements" / "pose_fits" / f"{args.object_id}_rigid_pose_graph" / "v19_rigid_object_pose_graph_report.json"
    frozen_p15 = load_json(frozen_p15_path)
    fixed_p15 = load_json(args.fixed_p15_report)

    input_video = benchmark_video
    segmentation = evaluate_masks(
        run_root,
        input_video,
        gt_dir,
        args.object_id,
        output_dir,
        args.boundary_tolerance_px,
    )
    projected = projected_mesh_vs_sparse_gt(
        annotations_path,
        completed_mesh,
        gt_dir,
        {"frozen_v1": frozen_p15, "pose_gate_fixed": fixed_p15},
        output_dir,
        args.boundary_tolerance_px,
    )

    frozen_p18 = next(
        path
        for path in (run_root / "measurements" / "mano_interval_correction").glob(
            f"{args.object_id}_0_149/{case_id}/v18_joint_mano_interval_trajectory_state.json"
        )
    )
    frozen_p18b = next(
        path
        for path in (run_root / "measurements" / "mano_interval_correction").glob(
            f"{args.object_id}_0_149_surface_hypothesis_metric_mano/{case_id}/v18_joint_mano_interval_trajectory_state.json"
        )
    )
    hand_reports, hand_comparison, hand_camera_extra = hand_and_camera_evaluation(
        run_root,
        gt_dir,
        frozen_p18,
        frozen_p18b,
        args.fixed_p18_state,
        args.fixed_p18b_state,
        args.min_num_views,
        args.max_gt_reprojection_px_rectified_512,
    )
    mechanism = mechanism_evidence(args, run_root)
    report = {
        "status": "current_output_vs_available_gt_comparison_complete",
        "method": "compare_current_v19_outputs_to_egoexo4d_partial_gt",
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "case_id": case_id,
        "inputs": {
            "run_root": str(run_root),
            "ground_truth_dir": str(gt_dir),
            "frozen_p15_report": str(frozen_p15_path),
            "fixed_p14_report": str(args.fixed_p14_report.resolve()),
            "fixed_p15_report": str(args.fixed_p15_report.resolve()),
            "fixed_p16_report": str(args.fixed_p16_report.resolve()),
            "fixed_p18_state": str(args.fixed_p18_state.resolve()),
            "fixed_p18b_state": str(args.fixed_p18b_state.resolve()),
            "completed_mesh": str(completed_mesh),
            "sha256": {
                "frozen_p15_report": sha256_file(frozen_p15_path),
                "fixed_p14_report": sha256_file(args.fixed_p14_report.resolve()),
                "fixed_p15_report": sha256_file(args.fixed_p15_report.resolve()),
                "fixed_p16_report": sha256_file(args.fixed_p16_report.resolve()),
                "fixed_p18_state": sha256_file(args.fixed_p18_state.resolve()),
                "fixed_p18b_state": sha256_file(args.fixed_p18b_state.resolve()),
                "completed_mesh": sha256_file(completed_mesh),
            },
        },
        "input_identity": input_identity,
        "ground_truth_availability": load_json(gt_dir / "GROUND_TRUTH_AVAILABILITY.json"),
        "evaluation_scope": {
            "direct_gt": ["sparse raw-view visible object masks", "named hand joints", "camera trajectory"],
            "diagnostic_not_direct_object_pose_gt": ["projected completed-mesh silhouette versus sparse visible masks"],
            "unsupported": ["object CAD", "object per-frame SE(3)", "MANO surface", "metric contact", "signed nonpenetration", "dense metric depth"],
        },
        "object_visible_segmentation": segmentation,
        "projected_mesh_visible_mask_diagnostic": projected,
        "frozen_vs_fixed_pose_state_delta": trajectory_comparison(frozen_p15, fixed_p15),
        "hand_pose": hand_reports,
        "hand_pose_method_comparison": hand_comparison,
        "hand_state_exact_delta": hand_camera_extra["candidate_vs_hawor_exact_delta"],
        "camera_trajectory": hand_camera_extra["camera_trajectory"],
        "mechanism_evidence": mechanism,
    }
    taxonomy = failure_taxonomy(report)
    report["failure_taxonomy"] = taxonomy
    write_json(output_dir / "current_output_vs_gt_evaluation.json", report)
    write_json(output_dir / "failure_taxonomy.json", taxonomy)
    summary = {
        "status": report["status"],
        "input_content_identical": input_identity["byte_content_identical"],
        "input_sha256": benchmark_hash,
        "evaluator_sha256": report["evaluator"]["sha256"],
        "object_segmentation_mean_iou": segmentation["aggregate"]["iou"]["mean"],
        "frozen_projected_mesh_mean_iou": projected["methods"]["frozen_v1"]["aggregate"]["iou"]["mean"],
        "fixed_projected_mesh_mean_iou": projected["methods"]["pose_gate_fixed"]["aggregate"]["iou"]["mean"],
        "fixed_p15_annotation_ready": fixed_p15.get("annotation_ready"),
        "fixed_p15_graph_frames": fixed_p15.get("graph_frame_count"),
        "fixed_p15_min_graph_frames": (fixed_p15.get("graph_support") or {}).get("configured_min_graph_frames"),
        "hand_hawor_absolute_mpjpe_mm": hand_reports["hawor_baseline"]["all_hands"]["absolute_joint_error_mm"]["mean"],
        "hand_hawor_root_relative_mpjpe_mm": hand_reports["hawor_baseline"]["all_hands"]["root_relative_joint_error_mm"]["mean"],
        "fixed_p18b_absolute_mpjpe_mm": hand_reports["fixed_p18b_quarantine"]["all_hands"]["absolute_joint_error_mm"]["mean"],
        "fixed_p18b_exactly_hawor": hand_camera_extra["candidate_vs_hawor_exact_delta"]["fixed_p18b_quarantine"]["exact_within_1e_9_m"],
        "camera_se3_ate_rmse_mm": hand_camera_extra["camera_trajectory"]["alignments"]["se3_metric"]["ate_center_rmse_mm"],
        "camera_sim3_scale": hand_camera_extra["camera_trajectory"]["alignments"]["sim3_diagnostic"]["scale"],
        "confirmed_bug_count": len(taxonomy["confirmed_implementation_bugs"]),
        "pipeline_design_problem_count": len(taxonomy["pipeline_design_or_architecture_problems"]),
        "suspected_bug_requiring_test_count": len(taxonomy["suspected_but_not_yet_proven_implementation_bugs"]),
        "current_input_or_gt_limit_count": len(taxonomy["current_input_observability_or_ground_truth_limits"]),
        "report": str(output_dir / "current_output_vs_gt_evaluation.json"),
        "taxonomy": str(output_dir / "failure_taxonomy.json"),
        "visualization": projected["visualization"],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
