#!/usr/bin/env python3
"""Render and audit P14 pairwise-chain -> P14 regularized -> P15 completed states.

This is a read-only post-prediction diagnostic.  It never mutates finalized case
roots and never consumes generated geometry, GT masks/poses/MANO, or rejected
metric depth as pose evidence.  Every stage uses the same prediction-side camera,
full metric MANO surfaces, and canonical observed metric object surface.

For the current observed-only P14 schema, the raw pairwise pose is recovered
exactly from fields preserved by P14:

  R_p14 = Exp(c_regularized) @ R_pairwise
  t_measurement = t_pairwise
                  + preliminary fixed-R anchor-translation updates
                  + post-rotation fixed-R anchor-translation updates

P14's final translation then applies the saved temporal translation regularizer.
The reconstruction is fail-closed against every saved pairwise edge's rotation
angle and translation norm and against the atomic anchor pose.

Legacy P14 reports that did not preserve pairwise-chain fields are rendered with
an explicit unavailable panel; no modern algorithm is rerun or substituted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXPERIMENT_DIR = REPO / "experiments" / "sam3d_p11_p12_branch"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_DIR))
import render_v19_rigid_state_artifact as canonical  # noqa: E402
import render_p14_p15_layered_state as layered  # noqa: E402

SCHEMA = "hot3d_p14_pairwise_regularized_p15_visual_ab_v1"
CASE_ORDER = ("milk", "soup", "mug", "bbq", "spatula")
OBJECT_COLOR = layered.ROLE_COLORS["observed_metric_surface_overlay"]
MASK_COLOR = (245, 245, 245)
CHAIN_OUTLINE_COLOR = (255, 235, 40)
P14_OUTLINE_COLOR = (60, 255, 60)
P15_OUTLINE_COLOR = (255, 70, 255)


@dataclasses.dataclass(frozen=True)
class Pose:
    rotation: np.ndarray
    translation: np.ndarray
    provenance: str


@dataclasses.dataclass(frozen=True)
class HandMesh:
    side: str
    vertices_world: np.ndarray
    faces: np.ndarray
    provenance: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", choices=CASE_ORDER, default=list(CASE_ORDER))
    parser.add_argument("--display-width", type=int, default=704)
    parser.add_argument("--metric-mask-width", type=int, default=352)
    parser.add_argument("--world-width", type=int, default=640)
    parser.add_argument("--world-height", type=int, default=480)
    parser.add_argument("--object-face-budget", type=int, default=2400)
    parser.add_argument("--mano-face-budget", type=int, default=1000)
    parser.add_argument("--world-padding-m", type=float, default=0.06)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Independent case workers; each case writes a disjoint subtree.")
    parser.add_argument("--encode-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--export-keyframe-glb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export selected stage scenes as full-topology world-frame GLBs.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--render-frames",
        type=int,
        nargs="*",
        default=None,
        help="Diagnostic subset only. Omit for the required full 150-frame output.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path | str, description: str) -> Path:
    out = Path(path).expanduser().resolve(strict=True)
    if not out.is_file():
        raise RuntimeError(f"missing {description}: {out}")
    return out


def require_rotation(value: Any, description: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid rotation for {description}: shape={matrix.shape}")
    orth_error = float(np.max(np.abs(matrix @ matrix.T - np.eye(3))))
    determinant = float(np.linalg.det(matrix))
    if orth_error > 1.0e-5 or abs(determinant - 1.0) > 1.0e-5:
        raise RuntimeError(
            f"non-rigid rotation for {description}: orth_error={orth_error} det={determinant}"
        )
    return matrix


def require_translation(value: Any, description: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise RuntimeError(f"invalid translation for {description}: shape={vector.shape}")
    return vector


def rotation_angle_deg(relative_rotation: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(relative_rotation).magnitude()))


def pose_delta(a: Pose, b: Pose) -> tuple[float, float]:
    return (
        rotation_angle_deg(b.rotation @ a.rotation.T),
        float(np.linalg.norm(b.translation - a.translation)),
    )


def numeric_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if len(finite) == 0:
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def trace_translation_sum(report: Any, description: str) -> np.ndarray:
    if not isinstance(report, dict):
        raise RuntimeError(f"missing translation-fit report for {description}")
    trace = report.get("trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError(f"missing translation-fit trace for {description}")
    updates: list[np.ndarray] = []
    for index, row in enumerate(trace):
        if not isinstance(row, dict):
            raise RuntimeError(f"malformed translation trace row {description}[{index}]")
        updates.append(require_translation(row.get("translation_update_m"), f"{description}[{index}]"))
    return np.sum(np.asarray(updates, dtype=np.float64), axis=0)


def pose_from_row(row: dict[str, Any], provenance: str) -> Pose | None:
    if row.get("rotation_world_from_completed_canonical_matrix") is None:
        return None
    return Pose(
        require_rotation(row["rotation_world_from_completed_canonical_matrix"], provenance),
        require_translation(row.get("translation_world_m"), provenance),
        provenance,
    )


def frame_object(frame: dict[str, Any], object_id: str) -> dict[str, Any]:
    for row in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(row, dict) and row.get("object_id") == object_id:
            return row
    raise RuntimeError(f"frame {frame.get('frame_idx')} lacks object {object_id}")


def reconstruct_case_stages(
    case_name: str,
    p14: dict[str, Any],
    p15: dict[str, Any],
) -> tuple[dict[int, Pose], dict[int, Pose], dict[int, Pose], dict[int, dict[str, Any]], dict[str, Any]]:
    p14_rows = {
        int(row["frame_idx"]): row
        for row in p14.get("pose_rows", [])
        if isinstance(row, dict) and row.get("frame_idx") is not None
    }
    p15_rows = {
        int(row["frame_idx"]): row
        for row in p15.get("pose_rows", [])
        if isinstance(row, dict) and row.get("frame_idx") is not None
    }
    if sorted(p15_rows) != list(range(150)):
        raise RuntimeError(f"{case_name}: P15 must contain exactly frames 0..149")

    p14_poses: dict[int, Pose] = {}
    p15_poses: dict[int, Pose] = {}
    chain_poses: dict[int, Pose] = {}
    frame_audit: dict[int, dict[str, Any]] = {}
    for frame_idx in range(150):
        p14_row = p14_rows.get(frame_idx, {})
        p15_row = p15_rows[frame_idx]
        p14_pose = pose_from_row(p14_row, f"{case_name} P14 frame {frame_idx}")
        p15_pose = pose_from_row(p15_row, f"{case_name} P15 frame {frame_idx}")
        if p15_pose is None:
            raise RuntimeError(f"{case_name}: P15 frame {frame_idx} lacks completed pose")
        if p14_pose is not None:
            p14_poses[frame_idx] = p14_pose
        p15_poses[frame_idx] = p15_pose
        temporal = p15_row.get("temporal_pose_graph") if isinstance(p15_row.get("temporal_pose_graph"), dict) else {}
        direct = temporal.get("direct_visible_measurement") is True
        completed = not direct
        frame_audit[frame_idx] = {
            "frame_idx": frame_idx,
            "p14_pose_available": p14_pose is not None,
            "p15_pose_available": True,
            "p15_direct_visible_measurement": direct,
            "p15_completed_pose": completed,
            "p15_status": str(p15_row.get("status", "unknown")),
            "p15_completion_source": temporal.get("pose_source"),
            "p15_completion_bracket_visible_pose_frames": temporal.get("bracket_visible_pose_frames"),
            "p14_direct_pose_observation_source": p14_row.get("direct_pose_observation_source"),
            "p14_pose_measurement_status": p14_row.get("status"),
        }

    has_pairwise_schema = bool(
        isinstance(p14.get("pairwise_observed_surface_registration"), dict)
        and any(
            isinstance(row.get("rotation_correction_from_pairwise_chain"), dict)
            for row in p14_rows.values()
        )
    )
    reconstruction: dict[str, Any] = {
        "mode": (
            "exact_inverse_of_saved_pairwise_chain_and_translation_traces"
            if has_pairwise_schema
            else "unavailable_legacy_p14_schema_without_pairwise_chain_fields"
        ),
        "pairwise_chain_available": has_pairwise_schema,
        "legacy_chain_substitution_performed": False,
    }

    if has_pairwise_schema:
        for frame_idx, row in p14_rows.items():
            source = row.get("direct_pose_observation_source")
            if source != "adjacent_observed_metric_surfel_registration":
                if source == "object_owned_rgb_optical_flow_calibrated_pnp":
                    frame_audit[frame_idx]["pairwise_chain_unavailable_reason"] = (
                        "RGB-PnP is a direct appearance bridge, not a pairwise metric-surface chain row"
                    )
                continue
            final_pose = p14_poses.get(frame_idx)
            if final_pose is None:
                raise RuntimeError(f"{case_name}: direct metric P14 row {frame_idx} lacks pose")
            correction_block = row.get("rotation_correction_from_pairwise_chain")
            translation_block = row.get("translation_regularization")
            if not isinstance(correction_block, dict) or not isinstance(translation_block, dict):
                raise RuntimeError(f"{case_name}: frame {frame_idx} lacks reversible P14 fields")
            correction_rotvec = require_translation(
                correction_block.get("regularized_correction_rotvec_rad"),
                f"{case_name} frame {frame_idx} correction rotvec",
            )
            correction_rotation = Rotation.from_rotvec(correction_rotvec).as_matrix()
            chain_rotation = correction_rotation.T @ final_pose.rotation
            saved_correction_deg = float(correction_block.get("regularized_correction_deg"))
            recovered_correction_deg = rotation_angle_deg(final_pose.rotation @ chain_rotation.T)
            if abs(saved_correction_deg - recovered_correction_deg) > 1.0e-8:
                raise RuntimeError(
                    f"{case_name}: frame {frame_idx} rotation correction inversion mismatch"
                )
            measurement_translation = require_translation(
                translation_block.get("measurement_translation_world_m"),
                f"{case_name} frame {frame_idx} measurement translation",
            )
            regularized_translation = require_translation(
                translation_block.get("regularized_translation_world_m"),
                f"{case_name} frame {frame_idx} regularized translation",
            )
            if float(np.max(np.abs(regularized_translation - final_pose.translation))) > 1.0e-12:
                raise RuntimeError(f"{case_name}: P14 row/final regularized translation mismatch")
            post_rotation_fit_sum = trace_translation_sum(
                row.get("observed_anchor_translation_fit"),
                f"{case_name} frame {frame_idx} post-rotation translation fit",
            )
            stabilized_chain_translation = measurement_translation - post_rotation_fit_sum
            preliminary_fit_sum = trace_translation_sum(
                row.get("preliminary_chain_translation_stabilization"),
                f"{case_name} frame {frame_idx} preliminary translation fit",
            )
            raw_chain_translation = stabilized_chain_translation - preliminary_fit_sum
            chain_poses[frame_idx] = Pose(
                chain_rotation,
                raw_chain_translation,
                "saved_pairwise_observed_metric_surfel_chain_before_anchor_unary_temporal_regularization",
            )
            raw_to_final_rotation_deg, raw_to_final_translation_m = pose_delta(
                chain_poses[frame_idx], final_pose
            )
            frame_audit[frame_idx].update(
                {
                    "pairwise_chain_pose_available": True,
                    "chain_to_p14_rotation_deg": raw_to_final_rotation_deg,
                    "chain_to_p14_translation_m": raw_to_final_translation_m,
                    "preliminary_chain_translation_stabilization_m": float(
                        np.linalg.norm(preliminary_fit_sum)
                    ),
                    "post_rotation_anchor_translation_fit_m": float(
                        np.linalg.norm(post_rotation_fit_sum)
                    ),
                    "measurement_to_p14_temporal_translation_regularization_m": float(
                        np.linalg.norm(final_pose.translation - measurement_translation)
                    ),
                    "pairwise_chain_translation_world_m": raw_chain_translation.tolist(),
                    "pairwise_chain_rotation_world_from_canonical_matrix": chain_rotation.tolist(),
                    "stabilized_chain_translation_world_m": stabilized_chain_translation.tolist(),
                    "p14_measurement_translation_world_m": measurement_translation.tolist(),
                    "regularized_unary_rotation_world_from_canonical_matrix": final_pose.rotation.tolist(),
                }
            )
    else:
        for frame_idx in range(150):
            frame_audit[frame_idx]["pairwise_chain_pose_available"] = False
            frame_audit[frame_idx]["pairwise_chain_unavailable_reason"] = (
                "finalized legacy P14 did not preserve pairwise-chain/correction fields; no substitute was generated"
            )

    direct_rotation_differences: list[float] = []
    direct_translation_differences: list[float] = []
    direct_matrix_max_abs: list[float] = []
    p14_direct_frames: list[int] = []
    completed_frames: list[int] = []
    for frame_idx, row in frame_audit.items():
        if row["p15_direct_visible_measurement"]:
            if frame_idx not in p14_poses:
                raise RuntimeError(f"{case_name}: P15 direct frame {frame_idx} absent from P14")
            max_abs = float(
                np.max(np.abs(p14_poses[frame_idx].rotation - p15_poses[frame_idx].rotation))
            )
            translation_m = float(
                np.linalg.norm(
                    p15_poses[frame_idx].translation - p14_poses[frame_idx].translation
                )
            )
            rotation_deg = (
                0.0
                if max_abs == 0.0
                else rotation_angle_deg(
                    p15_poses[frame_idx].rotation
                    @ p14_poses[frame_idx].rotation.T
                )
            )
            direct_rotation_differences.append(rotation_deg)
            direct_translation_differences.append(translation_m)
            direct_matrix_max_abs.append(max_abs)
            p14_direct_frames.append(frame_idx)
            row.update(
                {
                    "p14_to_p15_rotation_deg": rotation_deg,
                    "p14_to_p15_translation_m": translation_m,
                    "p14_to_p15_rotation_matrix_max_abs": max_abs,
                    "p14_p15_direct_pose_exactly_equal": bool(
                        max_abs == 0.0 and translation_m == 0.0
                    ),
                }
            )
        else:
            completed_frames.append(frame_idx)

    if max(direct_matrix_max_abs, default=0.0) != 0.0 or max(
        direct_translation_differences, default=0.0
    ) != 0.0:
        raise RuntimeError(f"{case_name}: P15 modified a direct P14 pose")

    edge_validation: dict[str, Any] | None = None
    if has_pairwise_schema:
        pairwise = p14["pairwise_observed_surface_registration"]
        edges = pairwise.get("edges")
        if not isinstance(edges, list):
            raise RuntimeError(f"{case_name}: pairwise report lacks edges")
        edge_rotation_errors: list[float] = []
        edge_translation_errors: list[float] = []
        for edge in edges:
            source_idx = int(edge["source_frame_idx"])
            target_idx = int(edge["target_frame_idx"])
            if source_idx not in chain_poses or target_idx not in chain_poses:
                raise RuntimeError(
                    f"{case_name}: pairwise edge {source_idx}->{target_idx} lacks recovered endpoint"
                )
            source_pose = chain_poses[source_idx]
            target_pose = chain_poses[target_idx]
            delta_rotation = target_pose.rotation @ source_pose.rotation.T
            delta_translation = (
                target_pose.translation - delta_rotation @ source_pose.translation
            )
            edge_rotation_errors.append(
                abs(rotation_angle_deg(delta_rotation) - float(edge["delta_rotation_deg"]))
            )
            edge_translation_errors.append(
                abs(float(np.linalg.norm(delta_translation)) - float(edge["delta_translation_m"]))
            )
        anchor_idx = int(p14["anchor_frame_idx"])
        anchor_centroid = require_translation(
            p14["anchor_centroid_world_m"], f"{case_name} anchor centroid"
        )
        anchor_pose = chain_poses.get(anchor_idx)
        if anchor_pose is None:
            raise RuntimeError(f"{case_name}: recovered chain lacks anchor {anchor_idx}")
        anchor_rotation_error = float(np.max(np.abs(anchor_pose.rotation - np.eye(3))))
        anchor_translation_error = float(np.linalg.norm(anchor_pose.translation - anchor_centroid))
        edge_validation = {
            "edge_count": len(edges),
            "saved_edge_scalar_contract": (
                "Recovered target/source SE(3) must reproduce every saved edge rotation angle and translation norm"
            ),
            "maximum_edge_rotation_angle_error_deg": max(edge_rotation_errors, default=0.0),
            "maximum_edge_translation_norm_error_m": max(edge_translation_errors, default=0.0),
            "anchor_frame_idx": anchor_idx,
            "anchor_identity_rotation_max_abs_error": anchor_rotation_error,
            "anchor_centroid_translation_error_m": anchor_translation_error,
            "passed": bool(
                max(edge_rotation_errors, default=0.0) <= 1.0e-7
                and max(edge_translation_errors, default=0.0) <= 1.0e-10
                and anchor_rotation_error <= 1.0e-12
                and anchor_translation_error <= 1.0e-12
            ),
        }
        if not edge_validation["passed"]:
            raise RuntimeError(f"{case_name}: pairwise chain inversion failed: {edge_validation}")

    reconstruction.update(
        {
            "pairwise_chain_pose_count": len(chain_poses),
            "p14_pose_count": len(p14_poses),
            "p15_pose_count": len(p15_poses),
            "p15_direct_pose_count": len(p14_direct_frames),
            "p15_completed_pose_count": len(completed_frames),
            "p15_completed_frames": completed_frames,
            "p14_p15_direct_rotation_deg": numeric_summary(direct_rotation_differences),
            "p14_p15_direct_translation_m": numeric_summary(direct_translation_differences),
            "p14_p15_direct_rotation_matrix_max_abs": numeric_summary(direct_matrix_max_abs),
            "p14_p15_direct_se3_exact_identity": True,
            "edge_reconstruction_validation": edge_validation,
        }
    )
    return chain_poses, p14_poses, p15_poses, frame_audit, reconstruction


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vertices, faces, summary = canonical.load_mesh(path)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32), summary


def project_silhouette(
    vertices_canonical: np.ndarray,
    faces: np.ndarray,
    pose: Pose,
    T_world_camera: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    vertices_world = vertices_canonical @ pose.rotation.T + pose.translation[None, :]
    vertices_camera = canonical.world_points_to_camera(vertices_world, T_world_camera)
    u, v, depth, _ = canonical.project_camera_points(
        vertices_camera, intrinsics, width, height
    )
    uv = np.c_[u, v]
    valid = (
        np.all(np.isfinite(uv[faces]), axis=(1, 2))
        & np.all(depth[faces] > 0.01, axis=1)
        & np.any(uv[faces, 0] >= -width, axis=1)
        & np.any(uv[faces, 0] < 2 * width, axis=1)
        & np.any(uv[faces, 1] >= -height, axis=1)
        & np.any(uv[faces, 1] < 2 * height, axis=1)
    )
    output = np.zeros((height, width), dtype=np.uint8)
    if np.any(valid):
        polygons = np.round(uv[faces[valid]]).astype(np.int32)
        cv2.fillPoly(output, polygons, 255, lineType=cv2.LINE_8)
    return output


def binary_mask_metrics(projected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    p = projected > 0
    o = observed > 0
    intersection = int(np.count_nonzero(p & o))
    union = int(np.count_nonzero(p | o))
    p_count = int(np.count_nonzero(p))
    o_count = int(np.count_nonzero(o))
    if p_count:
        py, px = np.nonzero(p)
        p_centroid = np.asarray([float(np.mean(px)), float(np.mean(py))])
    else:
        p_centroid = np.asarray([np.nan, np.nan])
    if o_count:
        oy, ox = np.nonzero(o)
        o_centroid = np.asarray([float(np.mean(ox)), float(np.mean(oy))])
    else:
        o_centroid = np.asarray([np.nan, np.nan])
    centroid_error = (
        float(np.linalg.norm(p_centroid - o_centroid))
        if np.isfinite(p_centroid).all() and np.isfinite(o_centroid).all()
        else None
    )
    return {
        "metric_role": "appearance_silhouette_diagnostic_only_not_metric_pose_evidence",
        "projected_area_px": p_count,
        "object_owned_mask_area_px": o_count,
        "intersection_px": intersection,
        "union_px": union,
        "iou": float(intersection / union) if union else None,
        "projected_precision": float(intersection / p_count) if p_count else None,
        "mask_recall": float(intersection / o_count) if o_count else None,
        "centroid_error_px": centroid_error,
    }


def nearest_surface_metrics(observed_world: np.ndarray, model_world: np.ndarray) -> dict[str, Any]:
    if (
        observed_world.ndim != 2
        or observed_world.shape[1] != 3
        or model_world.ndim != 2
        or model_world.shape[1] != 3
        or len(observed_world) == 0
        or len(model_world) == 0
    ):
        raise RuntimeError("invalid point arrays for nearest-surface metrics")
    observed_to_model = cKDTree(model_world).query(observed_world, k=1, workers=-1)[0]
    model_to_observed = cKDTree(observed_world).query(model_world, k=1, workers=-1)[0]
    return {
        "metric_role": "accepted_direct_metric_surface_fit",
        "observed_to_anchor_model_m": numeric_summary(observed_to_model.tolist()),
        "anchor_model_to_observed_m": numeric_summary(model_to_observed.tolist()),
    }


def unsigned_hand_object_vertex_metrics(
    object_vertices_world: np.ndarray,
    hands: list[HandMesh],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "metric_role": (
            "unsigned_vertex_proximity_diagnostic_only; not signed contact, penetration, or nonpenetration"
        ),
        "hands": {},
    }
    object_tree = cKDTree(object_vertices_world)
    for hand in hands:
        hand_tree = cKDTree(hand.vertices_world)
        hand_to_object = object_tree.query(hand.vertices_world, k=1, workers=-1)[0]
        object_to_hand = hand_tree.query(object_vertices_world, k=1, workers=-1)[0]
        output["hands"][hand.side] = {
            "minimum_vertex_distance_m": float(
                min(np.min(hand_to_object), np.min(object_to_hand))
            ),
            "hand_to_object_vertex_distance_m": numeric_summary(hand_to_object.tolist()),
            "object_to_hand_vertex_distance_m": numeric_summary(object_to_hand.tolist()),
        }
    return output


def scene_layers(
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    pose: Pose | None,
    hands: list[HandMesh],
    object_face_budget: int,
    mano_face_budget: int,
) -> list[layered.SceneLayer]:
    rows: list[layered.SceneLayer] = []
    if pose is not None:
        rows.append(
            layered.SceneLayer(
                name="observed_metric_object_surface",
                role="observed_metric_surface_overlay",
                vertices_world=object_vertices @ pose.rotation.T + pose.translation[None, :],
                faces=object_faces,
                color=OBJECT_COLOR,
                alpha=0.66,
                face_budget=int(object_face_budget),
                depth_priority_bias_m=0.002,
            )
        )
    for hand in hands:
        role = f"mano_{hand.side}_full_surface"
        rows.append(
            layered.SceneLayer(
                name=role,
                role=role,
                vertices_world=hand.vertices_world,
                faces=hand.faces,
                color=layered.ROLE_COLORS[role],
                alpha=0.60,
                face_budget=int(mano_face_budget),
                depth_priority_bias_m=0.0,
            )
        )
    return rows


def draw_mask_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, int(thickness), cv2.LINE_AA)


def text(image: np.ndarray, value: str, y: int, color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.40) -> None:
    canonical.put_text_with_bg(
        image,
        value[:105],
        (12, int(y)),
        font_scale=float(scale),
        color=color,
        thickness=1,
        bg_alpha=0.72,
    )


def render_camera_stage(
    rgb_small: np.ndarray,
    frame: dict[str, Any],
    raw_video: dict[str, Any],
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    pose: Pose | None,
    hands: list[HandMesh],
    mask_small: np.ndarray,
    title: str,
    subtitle: str,
    unavailable_reason: str | None,
    object_face_budget: int,
    mano_face_budget: int,
) -> np.ndarray:
    height, width = rgb_small.shape[:2]
    intrinsics, _ = canonical.scaled_intrinsics_for_frame(frame, width, height, raw_video)
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    T_world_camera = np.asarray(
        camera.get("T_world_camera_metric") or camera.get("T_world_camera"), dtype=np.float64
    )
    if T_world_camera.shape != (4, 4):
        raise RuntimeError(f"frame {frame.get('frame_idx')}: invalid camera matrix")
    layers = scene_layers(
        object_vertices,
        object_faces,
        pose,
        hands,
        object_face_budget,
        mano_face_budget,
    )
    projections: list[tuple[np.ndarray, np.ndarray]] = []
    for layer in layers:
        camera_vertices = canonical.world_points_to_camera(layer.vertices_world, T_world_camera)
        u, v, depth, _ = canonical.project_camera_points(
            camera_vertices, intrinsics, width, height
        )
        projections.append((np.c_[u, v], depth))
    output, _, _ = layered.rasterize_scene(rgb_small, layers, projections)
    draw_mask_contour(output, mask_small, MASK_COLOR, 2)
    text(output, title, 27, (255, 255, 255), 0.46)
    text(output, subtitle, 51, (220, 220, 220), 0.35)
    if unavailable_reason:
        shade = output.copy()
        cv2.rectangle(shade, (0, height // 2 - 44), (width, height // 2 + 44), (0, 0, 0), -1)
        output = cv2.addWeighted(shade, 0.72, output, 0.28, 0.0)
        text(output, "OBJECT POSE N/A", height // 2 - 2, (60, 215, 255), 0.72)
        text(output, unavailable_reason, height // 2 + 29, (80, 220, 255), 0.35)
    text(
        output,
        "green=observed surface | blue/orange=unchanged full MANO | white=owned mask",
        height - 15,
        (220, 235, 220),
        0.32,
    )
    return output


def shared_world_bounds(
    object_vertices: np.ndarray,
    poses: list[Pose | None],
    hands: list[HandMesh],
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    step = max(1, len(object_vertices) // 1800)
    for pose in poses:
        if pose is not None:
            chunks.append(object_vertices[::step] @ pose.rotation.T + pose.translation[None, :])
    for hand in hands:
        chunks.append(hand.vertices_world)
    if not chunks:
        raise RuntimeError("cannot derive shared world bounds")
    points = np.vstack(chunks)
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    extent = np.maximum(high - low, 0.18)
    return center - 0.5 * extent - padding, center + 0.5 * extent + padding


def render_world_stage(
    frame_idx: int,
    frame: dict[str, Any],
    camera_path: np.ndarray,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    pose: Pose | None,
    hands: list[HandMesh],
    low: np.ndarray,
    high: np.ndarray,
    title: str,
    subtitle: str,
    unavailable_reason: str | None,
    width: int,
    height: int,
    object_face_budget: int,
    mano_face_budget: int,
    horizontal_axis: int,
    vertical_axis: int,
    depth_axis: int,
    view_label: str,
) -> np.ndarray:
    background = np.full((height, width, 3), 15, dtype=np.uint8)
    layers = scene_layers(
        object_vertices,
        object_faces,
        pose,
        hands,
        object_face_budget,
        mano_face_budget,
    )
    projected = [
        layered.project_world_axes(
            layer.vertices_world,
            low,
            high,
            width,
            height,
            int(horizontal_axis),
            int(vertical_axis),
            int(depth_axis),
        )
        for layer in layers
    ]
    output, _, _ = layered.rasterize_scene(background, layers, projected)
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    T_world_camera = np.asarray(
        camera.get("T_world_camera_metric") or camera.get("T_world_camera"), dtype=np.float64
    )
    layered.draw_camera_state(
        output,
        T_world_camera,
        camera_path,
        low,
        high,
        (int(horizontal_axis), int(vertical_axis)),
        0.05,
    )
    text(output, title, 25, (255, 255, 255), 0.44)
    text(output, subtitle, 49, (220, 220, 220), 0.34)
    if unavailable_reason:
        text(output, "OBJECT POSE N/A", height // 2, (60, 215, 255), 0.65)
        text(output, unavailable_reason, height // 2 + 27, (80, 220, 255), 0.32)
    text(
        output,
        f"frame {frame_idx:03d} | {view_label} | identical camera/MANO/bounds",
        height - 14,
        (220, 220, 220),
        0.31,
    )
    return output


def export_keyframe_stage_glbs(
    case_name: str,
    selected_frame_ids: list[int],
    frame_audit: dict[int, dict[str, Any]],
    frames: dict[int, dict[str, Any]],
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    chain_poses: dict[int, Pose],
    p14_poses: dict[int, Pose],
    p15_poses: dict[int, Pose],
    hands_by_frame: dict[int, list[HandMesh]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stage_pose_maps = {
        "pairwise_chain": chain_poses,
        "p14_regularized": p14_poses,
        "p15_completed": p15_poses,
    }
    for frame_idx in selected_frame_ids:
        hands = hands_by_frame.get(frame_idx)
        if hands is None or len(hands) != 2:
            raise RuntimeError(f"{case_name}: no fixed full-MANO scene cached for GLB frame {frame_idx}")
        camera = frames[frame_idx].get("camera") if isinstance(frames[frame_idx].get("camera"), dict) else {}
        T_world_camera = np.asarray(
            camera.get("T_world_camera_metric") or camera.get("T_world_camera"),
            dtype=np.float64,
        )
        if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all():
            raise RuntimeError(f"{case_name}: malformed GLB camera matrix at frame {frame_idx}")
        frame_rows: dict[str, dict[str, Any]] = {}
        for stage_name in ("pairwise_chain", "p14_regularized", "p15_completed"):
            pose = stage_pose_maps[stage_name].get(frame_idx)
            if pose is None:
                row = {
                    "frame_idx": frame_idx,
                    "stage": stage_name,
                    "available": False,
                    "reason": (
                        frame_audit[frame_idx].get("pairwise_chain_unavailable_reason")
                        if stage_name == "pairwise_chain"
                        else "no direct P14 pose; P15 completion must not be backfilled into an earlier stage"
                    ),
                }
                rows.append(row)
                frame_rows[stage_name] = row
                continue
            path = output_dir / f"frame_{frame_idx:06d}_{stage_name}.glb"
            copied_from: str | None = None
            if (
                stage_name == "p15_completed"
                and frame_audit[frame_idx].get("p15_direct_visible_measurement") is True
            ):
                p14_row = frame_rows.get("p14_regularized")
                if not isinstance(p14_row, dict) or p14_row.get("available") is not True:
                    raise RuntimeError(
                        f"{case_name}: direct P15 frame {frame_idx} has no P14 GLB to bind"
                    )
                source = Path(str(p14_row["path"])).resolve(strict=True)
                shutil.copyfile(source, path)
                copied_from = str(source)
                exported = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "layers": p14_row["layers"],
                }
            else:
                layers = scene_layers(
                    object_vertices,
                    object_faces,
                    pose,
                    hands,
                    object_face_budget=0,
                    mano_face_budget=0,
                )
                exported = layered.export_world_glb(path, layers)
            if not path.is_file() or path.stat().st_size <= 20:
                raise RuntimeError(f"{case_name}: invalid exported GLB {path}")
            with path.open("rb") as handle:
                if handle.read(4) != b"glTF":
                    raise RuntimeError(f"{case_name}: invalid GLB magic for {path}")
            roles = [str(layer["role"]) for layer in exported["layers"]]
            if roles != [
                "observed_metric_surface_overlay",
                "mano_left_full_surface",
                "mano_right_full_surface",
            ]:
                raise RuntimeError(f"{case_name}: GLB role contract failed at frame {frame_idx}: {roles}")
            row = {
                "frame_idx": frame_idx,
                "stage": stage_name,
                "available": True,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "copied_from_exact_direct_p14_glb": copied_from,
                "rotation_world_from_completed_canonical_matrix": pose.rotation.tolist(),
                "translation_world_m": pose.translation.tolist(),
                "T_world_camera_metric": T_world_camera.tolist(),
                "layers": exported["layers"],
                "full_topology_exported": True,
                "camera_MANO_and_canonical_surface_fixed_within_frame": True,
                "generated_geometry_loaded_or_exported": False,
            }
            rows.append(row)
            frame_rows[stage_name] = row
        if frame_audit[frame_idx].get("p15_direct_visible_measurement") is True:
            p14_row = frame_rows["p14_regularized"]
            p15_row = frame_rows["p15_completed"]
            if p14_row["sha256"] != p15_row["sha256"]:
                raise RuntimeError(
                    f"{case_name}: direct frame {frame_idx} P14/P15 GLBs are not byte-identical"
                )
    index_path = output_dir / "KEYFRAME_STAGE_GLB_INDEX.json"
    payload = {
        "schema": "hot3d_p14_stage_keyframe_glb_index_v1",
        "case": case_name,
        "claim_scope": (
            "Interactive world-frame snapshots of the same observed canonical object surface and "
            "same full metric MANO; only saved object SE(3) stage varies."
        ),
        "selected_frame_ids": selected_frame_ids,
        "entry_count": len(rows),
        "available_glb_count": sum(row.get("available") is True for row in rows),
        "unavailable_stage_count": sum(row.get("available") is False for row in rows),
        "direct_p14_p15_glbs_required_byte_identical": True,
        "generated_geometry_loaded_or_exported": False,
        "GT_consumed": False,
        "entries": rows,
    }
    write_json(index_path, payload)
    return {
        "directory": str(output_dir),
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "entry_count": len(rows),
        "available_glb_count": payload["available_glb_count"],
        "unavailable_stage_count": payload["unavailable_stage_count"],
    }


def raw_reference_panel(
    rgb_small: np.ndarray,
    mask_small: np.ndarray,
    silhouettes: dict[str, np.ndarray | None],
    frame_idx: int,
    status: str,
) -> np.ndarray:
    output = rgb_small.copy()
    draw_mask_contour(output, mask_small, MASK_COLOR, 2)
    chain = silhouettes.get("pairwise_chain")
    p14 = silhouettes.get("p14_regularized")
    p15 = silhouettes.get("p15_completed")
    if chain is not None:
        draw_mask_contour(output, chain, CHAIN_OUTLINE_COLOR, 2)
    if p14 is not None:
        draw_mask_contour(output, p14, P14_OUTLINE_COLOR, 2)
    if p15 is not None and p14 is None:
        draw_mask_contour(output, p15, P15_OUTLINE_COLOR, 2)
    text(output, f"RGB REFERENCE | frame {frame_idx:03d}", 27, (255, 255, 255), 0.46)
    text(output, status, 51, (220, 220, 220), 0.35)
    text(
        output,
        "white=owned mask | cyan=pairwise | green=P14 | magenta=P15-only",
        output.shape[0] - 15,
        (235, 235, 235),
        0.32,
    )
    return output


def stage_titles(
    case_name: str,
    frame_idx: int,
    audit: dict[str, Any],
    chain_available_globally: bool,
) -> dict[str, tuple[str, str, str | None]]:
    if audit.get("pairwise_chain_pose_available"):
        chain_sub = (
            f"before P14 regularization | to P14: {float(audit['chain_to_p14_rotation_deg']):.2f} deg, "
            f"{1000.0 * float(audit['chain_to_p14_translation_m']):.1f} mm"
        )
        chain_unavailable = None
    else:
        chain_sub = "no reconstructed pairwise object pose"
        chain_unavailable = str(audit.get("pairwise_chain_unavailable_reason") or "no direct metric chain row")
    if audit.get("p14_pose_available"):
        if audit.get("p14_direct_pose_observation_source") == "object_owned_rgb_optical_flow_calibrated_pnp":
            p14_sub = "direct calibrated RGB-PnP bridge; not a pairwise metric-surface row"
        elif chain_available_globally:
            p14_sub = "observed-only unary rotation + translation temporal regularization"
        else:
            p14_sub = "legacy independent per-frame P14 fit; pairwise stage was not saved"
        p14_unavailable = None
    else:
        p14_sub = "no accepted direct P14 pose"
        p14_unavailable = "P14 had no direct accepted pose for this frame"
    if audit["p15_direct_visible_measurement"]:
        p15_sub = "direct row | exact same SE(3) as P14 (zero optimizer correction)"
    else:
        source = str(audit.get("p15_completion_source") or "temporal completion")
        brackets = audit.get("p15_completion_bracket_visible_pose_frames")
        p15_sub = f"{source} | brackets={brackets} | uncertain, not direct evidence"
    return {
        "pairwise_chain": (f"{case_name.upper()} | P14 PAIRWISE CHAIN", chain_sub, chain_unavailable),
        "p14_regularized": (f"{case_name.upper()} | P14 REGULARIZED", p14_sub, p14_unavailable),
        "p15_completed": (f"{case_name.upper()} | P15 COMPLETED", p15_sub, None),
    }


def select_keyframes(
    frame_rows: list[dict[str, Any]], anchor_frame_idx: int | None
) -> tuple[list[int], dict[str, list[str]]]:
    reasons: dict[int, list[str]] = {}

    def add(frame_idx: int, reason: str) -> None:
        if 0 <= int(frame_idx) < 150:
            reasons.setdefault(int(frame_idx), [])
            if reason not in reasons[int(frame_idx)]:
                reasons[int(frame_idx)].append(reason)

    if anchor_frame_idx is not None:
        add(anchor_frame_idx, "atomic anchor")
    for index in (0, 30, 60, 90, 120, 149):
        add(index, "uniform timeline")
    with_rotation = sorted(
        [r for r in frame_rows if r.get("chain_to_p14_rotation_deg") is not None],
        key=lambda r: float(r["chain_to_p14_rotation_deg"]),
        reverse=True,
    )
    for row in with_rotation[:3]:
        idx = int(row["frame_idx"])
        add(idx, "largest chain-to-P14 rotation")
        add(idx - 1, "neighbor of largest rotation")
        add(idx + 1, "neighbor of largest rotation")
    with_translation = sorted(
        [r for r in frame_rows if r.get("chain_to_p14_translation_m") is not None],
        key=lambda r: float(r["chain_to_p14_translation_m"]),
        reverse=True,
    )
    for row in with_translation[:2]:
        idx = int(row["frame_idx"])
        add(idx, "largest chain-to-P14 translation")
        add(idx - 1, "neighbor of largest translation")
        add(idx + 1, "neighbor of largest translation")
    residual_delta = sorted(
        [
            r
            for r in frame_rows
            if r.get("chain_to_p14_observed_surface_median_delta_m") is not None
        ],
        key=lambda r: float(r["chain_to_p14_observed_surface_median_delta_m"]),
    )
    if residual_delta:
        add(int(residual_delta[0]["frame_idx"]), "largest direct-surface residual improvement")
        add(int(residual_delta[-1]["frame_idx"]), "largest direct-surface residual degradation")
    completed = [int(r["frame_idx"]) for r in frame_rows if r.get("p15_completed_pose")]
    if completed:
        add(completed[0], "first P15 completed frame")
        add(completed[len(completed) // 2], "middle P15 completed frame")
        add(completed[-1], "last P15 completed frame")
    bridges = [
        int(r["frame_idx"])
        for r in frame_rows
        if r.get("p14_direct_pose_observation_source")
        == "object_owned_rgb_optical_flow_calibrated_pnp"
    ]
    for frame_idx in bridges:
        add(frame_idx, "direct calibrated RGB-PnP bridge")
    return sorted(reasons), reasons


def build_contact_sheet(
    frame_dir: Path,
    selected: list[int],
    reasons: dict[str | int, list[str]],
    output_path: Path,
) -> None:
    tiles: list[np.ndarray] = []
    target_width = 1408
    for frame_idx in selected:
        path = frame_dir / f"{frame_idx:06d}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        target_height = int(round(image.shape[0] * target_width / image.shape[1]))
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        header = np.full((42, target_width, 3), 18, dtype=np.uint8)
        reason_values = reasons.get(frame_idx) or reasons.get(str(frame_idx)) or []
        cv2.putText(
            header,
            f"frame {frame_idx:03d}: {'; '.join(reason_values)}"[:180],
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        tiles.append(np.vstack([header, image]))
    if not tiles:
        raise RuntimeError(f"no selected frames available for contact sheet {output_path}")
    columns = 2
    rows: list[np.ndarray] = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), columns):
        chunk = tiles[start : start + columns]
        if len(chunk) < columns:
            chunk += [blank.copy()] * (columns - len(chunk))
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise RuntimeError(f"failed to write contact sheet {output_path}")


def plot_case_timeline(case_name: str, frame_rows: list[dict[str, Any]], output_path: Path) -> None:
    x = np.arange(150)

    def values(key: str) -> np.ndarray:
        out = np.full(150, np.nan, dtype=np.float64)
        for row in frame_rows:
            value = row.get(key)
            if value is not None:
                out[int(row["frame_idx"])] = float(value)
        return out

    completed = np.asarray([bool(r.get("p15_completed_pose")) for r in frame_rows], dtype=bool)
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
    rotation = values("chain_to_p14_rotation_deg")
    axes[0].plot(x, rotation, color="#00bcd4", linewidth=1.5, label="pairwise chain -> P14")
    axes[0].set_ylabel("rotation (deg)")
    axes[0].legend(loc="upper right")

    translation = 1000.0 * values("chain_to_p14_translation_m")
    temporal = 1000.0 * values("measurement_to_p14_temporal_translation_regularization_m")
    axes[1].plot(x, translation, color="#ff9800", linewidth=1.4, label="raw chain -> P14")
    axes[1].plot(x, temporal, color="#9c27b0", linewidth=1.1, label="measurement -> P14 temporal")
    axes[1].set_ylabel("translation (mm)")
    axes[1].legend(loc="upper right")

    chain_residual = 1000.0 * values("pairwise_chain_observed_surface_median_m")
    p14_residual = 1000.0 * values("p14_observed_surface_median_m")
    axes[2].plot(x, chain_residual, color="#00bcd4", linewidth=1.1, label="pairwise chain")
    axes[2].plot(x, p14_residual, color="#4caf50", linewidth=1.1, label="P14 regularized")
    axes[2].set_ylabel("accepted surface median (mm)")
    axes[2].legend(loc="upper right")

    axes[3].plot(x, values("pairwise_chain_mask_iou"), color="#00bcd4", linewidth=1.0, label="pairwise chain")
    axes[3].plot(x, values("p14_mask_iou"), color="#4caf50", linewidth=1.0, label="P14")
    axes[3].plot(x, values("p15_mask_iou"), color="#e040fb", linewidth=1.0, linestyle="--", label="P15")
    axes[3].scatter(x[completed], values("p15_mask_iou")[completed], s=14, color="#e040fb", label="P15 completed")
    axes[3].set_ylabel("appearance mask IoU")
    axes[3].set_xlabel("frame index")
    axes[3].legend(loc="upper right", ncol=2)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        for idx in x[completed]:
            axis.axvspan(idx - 0.45, idx + 0.45, color="#e040fb", alpha=0.045)
    fig.suptitle(
        f"{case_name}: P14 pairwise chain -> P14 regularized -> P15 completed\n"
        "Surface residual uses accepted metric rows only; mask IoU is appearance-only diagnostic",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def video_decode_qc(path: Path, expected_frames: int, expected_fps: float) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open encoded video {path}")
    metadata_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            raise RuntimeError(f"decoded empty frame {decoded} from {path}")
        decoded += 1
    capture.release()
    passed = bool(
        metadata_frames == expected_frames
        and decoded == expected_frames
        and abs(fps - expected_fps) <= 1.0e-3
    )
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "metadata_frame_count": metadata_frames,
        "decoded_frame_count": decoded,
        "fps": fps,
        "width": width,
        "height": height,
        "expected_frame_count": expected_frames,
        "expected_fps": expected_fps,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"encoded video QC failed: {report}")
    return report


def summary_from_frame_rows(frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(key: str) -> list[float]:
        return [float(row[key]) for row in frame_rows if row.get(key) is not None]

    direct_metric = [
        row
        for row in frame_rows
        if row.get("p14_direct_pose_observation_source")
        == "adjacent_observed_metric_surfel_registration"
    ]
    residual_deltas = collect("chain_to_p14_observed_surface_median_delta_m")
    mask_deltas = collect("chain_to_p14_mask_iou_delta")
    return {
        "frame_count": len(frame_rows),
        "direct_metric_surface_frame_count": len(direct_metric),
        "rgb_pnp_bridge_frame_count": sum(
            row.get("p14_direct_pose_observation_source")
            == "object_owned_rgb_optical_flow_calibrated_pnp"
            for row in frame_rows
        ),
        "p15_completed_frame_count": sum(bool(row.get("p15_completed_pose")) for row in frame_rows),
        "chain_to_p14_rotation_deg": numeric_summary(collect("chain_to_p14_rotation_deg")),
        "chain_to_p14_translation_m": numeric_summary(collect("chain_to_p14_translation_m")),
        "preliminary_chain_translation_stabilization_m": numeric_summary(
            collect("preliminary_chain_translation_stabilization_m")
        ),
        "post_rotation_anchor_translation_fit_m": numeric_summary(
            collect("post_rotation_anchor_translation_fit_m")
        ),
        "measurement_to_p14_temporal_translation_regularization_m": numeric_summary(
            collect("measurement_to_p14_temporal_translation_regularization_m")
        ),
        "pairwise_chain_observed_surface_median_m": numeric_summary(
            collect("pairwise_chain_observed_surface_median_m")
        ),
        "preliminary_translation_stabilized_chain_observed_surface_median_m": numeric_summary(
            collect("preliminary_translation_stabilized_chain_observed_surface_median_m")
        ),
        "regularized_rotation_at_stabilized_chain_translation_observed_surface_median_m": numeric_summary(
            collect("regularized_rotation_at_stabilized_chain_translation_observed_surface_median_m")
        ),
        "p14_measurement_before_temporal_translation_regularization_observed_surface_median_m": numeric_summary(
            collect("p14_measurement_before_temporal_translation_regularization_observed_surface_median_m")
        ),
        "p14_observed_surface_median_m": numeric_summary(
            collect("p14_observed_surface_median_m")
        ),
        "preliminary_translation_stabilization_residual_delta_m": numeric_summary(
            collect("preliminary_translation_stabilization_residual_delta_m")
        ),
        "regularized_rotation_residual_delta_m": numeric_summary(
            collect("regularized_rotation_residual_delta_m")
        ),
        "post_rotation_translation_fit_residual_delta_m": numeric_summary(
            collect("post_rotation_translation_fit_residual_delta_m")
        ),
        "temporal_translation_regularization_residual_delta_m": numeric_summary(
            collect("temporal_translation_regularization_residual_delta_m")
        ),
        "chain_to_p14_observed_surface_median_delta_m": numeric_summary(residual_deltas),
        "fraction_direct_metric_frames_with_lower_p14_median_residual": (
            float(sum(delta < 0.0 for delta in residual_deltas) / len(residual_deltas))
            if residual_deltas
            else None
        ),
        "pairwise_chain_mask_iou": numeric_summary(collect("pairwise_chain_mask_iou")),
        "p14_mask_iou": numeric_summary(collect("p14_mask_iou")),
        "p15_mask_iou": numeric_summary(collect("p15_mask_iou")),
        "chain_to_p14_mask_iou_delta": numeric_summary(mask_deltas),
        "fraction_comparable_frames_with_higher_p14_mask_iou": (
            float(sum(delta > 0.0 for delta in mask_deltas) / len(mask_deltas))
            if mask_deltas
            else None
        ),
        "interpretation": {
            "negative_surface_delta_means": "the named stage lowered accepted observed-metric surface residual",
            "positive_mask_iou_delta_means": "P14 increased appearance silhouette overlap",
            "mask_iou_is_pose_evidence": False,
            "unsigned_hand_object_vertex_distance_is_contact_evidence": False,
            "visual_review_required": True,
        },
    }


def render_case(
    case_name: str,
    run_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    p14_path = require_file(
        run_root
        / "experiments/sam3d_trellis_controlled/P14_observed_pose_fit/v18_compact_rigid_object_pose_fit_report.json",
        f"{case_name} P14 report",
    )
    p15_path = require_file(
        run_root
        / "experiments/sam3d_trellis_controlled/P15_observed_pose_graph/v19_rigid_object_pose_graph_report.json",
        f"{case_name} P15 report",
    )
    state_path = require_file(
        run_root / "final_results/sam3d/state/render_state.json",
        f"{case_name} final SAM3D render state",
    )
    p14 = load_json(p14_path)
    p15 = load_json(p15_path)
    state = load_json(state_path)
    object_id = str(p15.get("object_id") or p14.get("object_id"))
    if not object_id:
        raise RuntimeError(f"{case_name}: missing object id")
    adapter = state.get("experimental_p14_p15_adapter") if isinstance(state.get("experimental_p14_p15_adapter"), dict) else {}
    geometry = state.get("object_geometry") if isinstance(state.get("object_geometry"), dict) else {}
    physical = geometry.get("physical_surface") if isinstance(geometry.get("physical_surface"), dict) else {}
    if (
        physical.get("generated_faces_collision_eligible") is not False
        or physical.get("generated_faces_contact_eligible") is not False
        or adapter.get("generated_faces_collision_eligible") is not False
        or adapter.get("generated_faces_contact_eligible") is not False
    ):
        raise RuntimeError(f"{case_name}: final state does not preserve generated-face quarantine")
    object_mesh_path = require_file(physical.get("mesh"), f"{case_name} observed metric surface")
    p14_pose_mesh = p14.get("pose_hypothesis_mesh_labeled") or (p14.get("inputs") or {}).get("completed_mesh")
    if p14_pose_mesh is None or require_file(p14_pose_mesh, f"{case_name} P14 pose mesh") != object_mesh_path:
        raise RuntimeError(f"{case_name}: P14 and final state do not share the same observed canonical surface")
    object_vertices, object_faces, object_mesh_summary = load_mesh(object_mesh_path)

    annotation_path = require_file(
        (state.get("annotation_backbone") or {}).get("path"), f"{case_name} annotation backbone"
    )
    annotations = load_json(annotation_path)
    frames_list = annotations.get("frames") if isinstance(annotations.get("frames"), list) else []
    frames = {
        int(frame["frame_idx"]): frame
        for frame in frames_list
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }
    if sorted(frames) != list(range(150)):
        raise RuntimeError(f"{case_name}: annotations must contain exactly frames 0..149")
    raw_video = annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else {}
    fps = float(args.fps if args.fps is not None else raw_video.get("fps") or 30.0)
    if abs(fps - 30.0) > 1.0e-6:
        raise RuntimeError(f"{case_name}: expected 30 FPS, got {fps}")

    chain_poses, p14_poses, p15_poses, frame_audit, reconstruction = reconstruct_case_stages(
        case_name, p14, p15
    )
    p14_rows_by_idx = {
        int(row["frame_idx"]): row
        for row in p14.get("pose_rows", [])
        if isinstance(row, dict) and row.get("frame_idx") is not None
    }
    has_chain = bool(reconstruction["pairwise_chain_available"])
    anchor_frame_idx = int(p14["anchor_frame_idx"]) if has_chain else None
    anchor_canonical: np.ndarray | None = None
    if has_chain:
        anchor_object = frame_object(frames[anchor_frame_idx], object_id)
        anchor_geom = anchor_object.get("visible_geometry_candidate") if isinstance(anchor_object.get("visible_geometry_candidate"), dict) else {}
        anchor_observed = np.asarray(anchor_geom.get("world_vertices_sample_m") or [], dtype=np.float64)
        anchor_centroid = require_translation(
            p14.get("anchor_centroid_world_m"), f"{case_name} P14 anchor centroid"
        )
        if anchor_observed.ndim != 2 or anchor_observed.shape[1] != 3 or len(anchor_observed) < 6:
            raise RuntimeError(f"{case_name}: invalid anchor observed metric samples")
        anchor_canonical = anchor_observed - anchor_centroid[None, :]
        if float(np.linalg.norm(anchor_canonical.mean(axis=0))) > 1.0e-10:
            raise RuntimeError(f"{case_name}: atomic anchor canonical surface is not centered")

    case_dir = output_root / "cases" / case_name
    camera_dir = case_dir / "camera_frames"
    world_dir = case_dir / "world_frames"
    side_dir = case_dir / "side_frames"
    review_dir = case_dir / "review"
    for directory in (camera_dir, world_dir, side_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    requested_frames = (
        sorted(set(int(v) for v in args.render_frames))
        if args.render_frames is not None
        else list(range(150))
    )
    if any(idx < 0 or idx >= 150 for idx in requested_frames):
        raise RuntimeError(f"{case_name}: --render-frames outside 0..149")
    full_timeline = requested_frames == list(range(150))
    if args.encode_video and not full_timeline:
        raise RuntimeError("video encoding requires the full contiguous 150-frame timeline")

    camera_path = np.asarray(
        [
            ((frames[idx].get("camera") or {}).get("position_world_m") or np.asarray((frames[idx].get("camera") or {}).get("T_world_camera_metric"))[:3, 3].tolist())
            for idx in range(150)
        ],
        dtype=np.float64,
    )
    mano_cache = layered.ManoArchiveCache([])
    hands_by_frame: dict[int, list[HandMesh]] = {}
    rendered_frame_rows: list[dict[str, Any]] = []
    try:
        for count, frame_idx in enumerate(requested_frames):
            frame = frames[frame_idx]
            audit = frame_audit[frame_idx]
            raw_path = require_file(frame.get("raw_frame_path"), f"{case_name} frame {frame_idx} RGB")
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw is None:
                raise RuntimeError(f"failed to read RGB {raw_path}")
            source_h, source_w = raw.shape[:2]
            display_w = int(args.display_width)
            display_h = int(round(source_h * display_w / source_w))
            rgb_small = cv2.resize(raw, (display_w, display_h), interpolation=cv2.INTER_AREA)
            obj = frame_object(frame, object_id)
            mask_path = require_file(obj.get("mask_path"), f"{case_name} frame {frame_idx} owned mask")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or not np.any(mask > 0):
                raise RuntimeError(f"{case_name}: empty/unreadable owned mask frame {frame_idx}")
            mask_small = cv2.resize(mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST_EXACT)
            metric_w = int(args.metric_mask_width)
            metric_h = int(round(source_h * metric_w / source_w))
            mask_metric = cv2.resize(mask, (metric_w, metric_h), interpolation=cv2.INTER_NEAREST_EXACT)
            camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
            T_world_camera = np.asarray(
                camera.get("T_world_camera_metric") or camera.get("T_world_camera"), dtype=np.float64
            )
            metric_intrinsics, metric_intrinsics_report = canonical.scaled_intrinsics_for_frame(
                frame, metric_w, metric_h, raw_video
            )

            hands: list[HandMesh] = []
            for hand_row in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
                if not isinstance(hand_row, dict):
                    continue
                vertices, faces_hand, provenance = mano_cache.hand_mesh(frame_idx, hand_row)
                hands.append(
                    HandMesh(
                        str(provenance["side"]),
                        np.asarray(vertices, dtype=np.float64),
                        np.asarray(faces_hand, dtype=np.int32),
                        provenance,
                    )
                )
            if len(hands) != 2 or sorted(hand.side for hand in hands) != ["left", "right"]:
                raise RuntimeError(f"{case_name}: frame {frame_idx} lacks two complete MANO surfaces")
            hands_by_frame[frame_idx] = hands

            stage_poses = {
                "pairwise_chain": chain_poses.get(frame_idx),
                "p14_regularized": p14_poses.get(frame_idx),
                "p15_completed": p15_poses[frame_idx],
            }
            silhouettes_metric: dict[str, np.ndarray | None] = {}
            silhouette_metrics: dict[str, Any] = {}
            hand_object_metrics: dict[str, Any] = {}
            for stage_name, pose in stage_poses.items():
                if pose is None:
                    silhouettes_metric[stage_name] = None
                    silhouette_metrics[stage_name] = None
                    hand_object_metrics[stage_name] = None
                    continue
                if (
                    stage_name == "p15_completed"
                    and audit["p15_direct_visible_measurement"]
                    and silhouettes_metric.get("p14_regularized") is not None
                ):
                    silhouettes_metric[stage_name] = silhouettes_metric["p14_regularized"].copy()
                    silhouette_metrics[stage_name] = dict(silhouette_metrics["p14_regularized"])
                    silhouette_metrics[stage_name]["copied_because_p14_p15_direct_se3_exact_identity"] = True
                    hand_object_metrics[stage_name] = dict(hand_object_metrics["p14_regularized"])
                    hand_object_metrics[stage_name]["copied_because_p14_p15_direct_se3_exact_identity"] = True
                    continue
                silhouette = project_silhouette(
                    object_vertices,
                    object_faces,
                    pose,
                    T_world_camera,
                    metric_intrinsics,
                    metric_w,
                    metric_h,
                )
                silhouettes_metric[stage_name] = silhouette
                silhouette_metrics[stage_name] = binary_mask_metrics(silhouette, mask_metric)
                object_world = object_vertices @ pose.rotation.T + pose.translation[None, :]
                hand_object_metrics[stage_name] = unsigned_hand_object_vertex_metrics(
                    object_world, hands
                )

            surface_metrics: dict[str, Any] = {
                "pairwise_chain": None,
                "preliminary_translation_stabilized_chain": None,
                "regularized_rotation_at_stabilized_chain_translation": None,
                "p14_measurement_before_temporal_translation_regularization": None,
                "p14_regularized": None,
                "p15_completed": None,
            }
            if (
                anchor_canonical is not None
                and audit.get("p14_direct_pose_observation_source")
                == "adjacent_observed_metric_surfel_registration"
            ):
                geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
                observed = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float64)
                if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 6:
                    raise RuntimeError(f"{case_name}: direct metric row {frame_idx} lacks accepted observed samples")
                chain_pose = stage_poses["pairwise_chain"]
                p14_pose = stage_poses["p14_regularized"]
                if chain_pose is None or p14_pose is None:
                    raise RuntimeError(f"{case_name}: direct metric row {frame_idx} lacks chain/P14 pose")
                chain_model = anchor_canonical @ chain_pose.rotation.T + chain_pose.translation[None, :]
                stabilized_chain_translation = require_translation(
                    audit.get("stabilized_chain_translation_world_m"),
                    f"{case_name} frame {frame_idx} stabilized chain translation",
                )
                measurement_translation = require_translation(
                    audit.get("p14_measurement_translation_world_m"),
                    f"{case_name} frame {frame_idx} P14 measurement translation",
                )
                stabilized_chain_model = (
                    anchor_canonical @ chain_pose.rotation.T
                    + stabilized_chain_translation[None, :]
                )
                regularized_rotation_stabilized_translation_model = (
                    anchor_canonical @ p14_pose.rotation.T
                    + stabilized_chain_translation[None, :]
                )
                p14_measurement_model = (
                    anchor_canonical @ p14_pose.rotation.T
                    + measurement_translation[None, :]
                )
                p14_model = anchor_canonical @ p14_pose.rotation.T + p14_pose.translation[None, :]
                surface_metrics["pairwise_chain"] = nearest_surface_metrics(observed, chain_model)
                surface_metrics["preliminary_translation_stabilized_chain"] = nearest_surface_metrics(
                    observed, stabilized_chain_model
                )
                surface_metrics["regularized_rotation_at_stabilized_chain_translation"] = nearest_surface_metrics(
                    observed, regularized_rotation_stabilized_translation_model
                )
                surface_metrics["p14_measurement_before_temporal_translation_regularization"] = nearest_surface_metrics(
                    observed, p14_measurement_model
                )
                surface_metrics["p14_regularized"] = nearest_surface_metrics(observed, p14_model)
                surface_metrics["p15_completed"] = dict(surface_metrics["p14_regularized"])
                surface_metrics["p15_completed"]["copied_because_p14_p15_direct_se3_exact_identity"] = True
                saved_row = p14_rows_by_idx.get(frame_idx, {})
                saved = saved_row.get("observed_anchor_to_current_final")
                if isinstance(saved, dict) and saved.get("median_m") is not None:
                    independently_recomputed = float(
                        surface_metrics["p14_regularized"]["observed_to_anchor_model_m"]["median"]
                    )
                    if abs(independently_recomputed - float(saved["median_m"])) > 1.0e-12:
                        raise RuntimeError(
                            f"{case_name}: frame {frame_idx} independent P14 residual does not reproduce saved report"
                        )

            silhouettes_small = {
                key: (
                    cv2.resize(value, (display_w, display_h), interpolation=cv2.INTER_NEAREST_EXACT)
                    if value is not None
                    else None
                )
                for key, value in silhouettes_metric.items()
            }
            if audit["p15_direct_visible_measurement"]:
                silhouettes_small["p15_completed"] = silhouettes_small["p14_regularized"]
            raw_status = (
                "P15 completed pose (magenta) has no direct P14 row"
                if audit["p15_completed_pose"]
                else "P14 and P15 direct pose are exactly identical"
            )
            raw_panel = raw_reference_panel(
                rgb_small, mask_small, silhouettes_small, frame_idx, raw_status
            )
            titles = stage_titles(case_name, frame_idx, audit, has_chain)
            stage_panels: list[np.ndarray] = []
            for stage_name in ("pairwise_chain", "p14_regularized", "p15_completed"):
                title_value, subtitle, unavailable = titles[stage_name]
                stage_panels.append(
                    render_camera_stage(
                        rgb_small,
                        frame,
                        raw_video,
                        object_vertices,
                        object_faces,
                        stage_poses[stage_name],
                        hands,
                        mask_small,
                        title_value,
                        subtitle,
                        unavailable,
                        int(args.object_face_budget),
                        int(args.mano_face_budget),
                    )
                )
            camera_comparison = np.hstack([raw_panel, *stage_panels])
            camera_path_out = camera_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(
                str(camera_path_out), camera_comparison, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise RuntimeError(f"failed to write {camera_path_out}")

            low, high = shared_world_bounds(
                object_vertices,
                [stage_poses["pairwise_chain"], stage_poses["p14_regularized"], stage_poses["p15_completed"]],
                hands,
                float(args.world_padding_m),
            )
            world_panels: list[np.ndarray] = []
            for stage_name in ("pairwise_chain", "p14_regularized", "p15_completed"):
                title_value, subtitle, unavailable = titles[stage_name]
                world_panels.append(
                    render_world_stage(
                        frame_idx,
                        frame,
                        camera_path,
                        object_vertices,
                        object_faces,
                        stage_poses[stage_name],
                        hands,
                        low,
                        high,
                        title_value,
                        subtitle,
                        unavailable,
                        int(args.world_width),
                        int(args.world_height),
                        int(args.object_face_budget),
                        int(args.mano_face_budget),
                        0,
                        2,
                        1,
                        "WORLD X-Z (depth=Y)",
                    )
                )
            world_comparison = np.hstack(world_panels)
            world_path_out = world_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(
                str(world_path_out), world_comparison, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise RuntimeError(f"failed to write {world_path_out}")

            side_panels: list[np.ndarray] = []
            for stage_name in ("pairwise_chain", "p14_regularized", "p15_completed"):
                title_value, subtitle, unavailable = titles[stage_name]
                side_panels.append(
                    render_world_stage(
                        frame_idx,
                        frame,
                        camera_path,
                        object_vertices,
                        object_faces,
                        stage_poses[stage_name],
                        hands,
                        low,
                        high,
                        title_value,
                        subtitle,
                        unavailable,
                        int(args.world_width),
                        int(args.world_height),
                        int(args.object_face_budget),
                        int(args.mano_face_budget),
                        1,
                        2,
                        0,
                        "SIDE Y-Z (depth=X)",
                    )
                )
            side_comparison = np.hstack(side_panels)
            side_path_out = side_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(
                str(side_path_out), side_comparison, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise RuntimeError(f"failed to write {side_path_out}")

            audit["mask_metric_intrinsics"] = metric_intrinsics_report
            audit["appearance_silhouette_metrics"] = silhouette_metrics
            audit["accepted_metric_surface_metrics"] = surface_metrics
            audit["unsigned_hand_object_vertex_metrics"] = hand_object_metrics
            audit["mano_sides"] = [hand.side for hand in hands]
            audit["rendered_camera_frame"] = str(camera_path_out)
            audit["rendered_world_frame"] = str(world_path_out)
            audit["rendered_side_frame"] = str(side_path_out)
            for stage_name, prefix in (
                ("pairwise_chain", "pairwise_chain"),
                (
                    "preliminary_translation_stabilized_chain",
                    "preliminary_translation_stabilized_chain",
                ),
                (
                    "regularized_rotation_at_stabilized_chain_translation",
                    "regularized_rotation_at_stabilized_chain_translation",
                ),
                (
                    "p14_measurement_before_temporal_translation_regularization",
                    "p14_measurement_before_temporal_translation_regularization",
                ),
                ("p14_regularized", "p14"),
                ("p15_completed", "p15"),
            ):
                sm = silhouette_metrics.get(stage_name)
                if isinstance(sm, dict):
                    audit[f"{prefix}_mask_iou"] = sm.get("iou")
                surf = surface_metrics.get(stage_name)
                if isinstance(surf, dict):
                    audit[f"{prefix}_observed_surface_median_m"] = (
                        surf.get("observed_to_anchor_model_m") or {}
                    ).get("median")
            residual_stage_pairs = (
                (
                    "pairwise_chain_observed_surface_median_m",
                    "preliminary_translation_stabilized_chain_observed_surface_median_m",
                    "preliminary_translation_stabilization_residual_delta_m",
                ),
                (
                    "preliminary_translation_stabilized_chain_observed_surface_median_m",
                    "regularized_rotation_at_stabilized_chain_translation_observed_surface_median_m",
                    "regularized_rotation_residual_delta_m",
                ),
                (
                    "regularized_rotation_at_stabilized_chain_translation_observed_surface_median_m",
                    "p14_measurement_before_temporal_translation_regularization_observed_surface_median_m",
                    "post_rotation_translation_fit_residual_delta_m",
                ),
                (
                    "p14_measurement_before_temporal_translation_regularization_observed_surface_median_m",
                    "p14_observed_surface_median_m",
                    "temporal_translation_regularization_residual_delta_m",
                ),
                (
                    "pairwise_chain_observed_surface_median_m",
                    "p14_observed_surface_median_m",
                    "chain_to_p14_observed_surface_median_delta_m",
                ),
            )
            for before_key, after_key, delta_key in residual_stage_pairs:
                if audit.get(before_key) is not None and audit.get(after_key) is not None:
                    audit[delta_key] = float(audit[after_key] - audit[before_key])
            if audit.get("pairwise_chain_mask_iou") is not None and audit.get("p14_mask_iou") is not None:
                audit["chain_to_p14_mask_iou_delta"] = float(
                    audit["p14_mask_iou"] - audit["pairwise_chain_mask_iou"]
                )
            rendered_frame_rows.append(audit)
            if count % 15 == 0 or count + 1 == len(requested_frames):
                print(
                    f"{case_name}: rendered {count + 1}/{len(requested_frames)} frame={frame_idx}",
                    flush=True,
                )
    finally:
        mano_summary = mano_cache.summary()
        mano_cache.close()

    all_rows = [frame_audit[idx] for idx in range(150)]
    numeric = summary_from_frame_rows(all_rows)
    selected, reasons = select_keyframes(all_rows, anchor_frame_idx)
    selected_rendered = [idx for idx in selected if idx in requested_frames]
    keyframe_path = review_dir / f"{case_name}_keyframe_camera_comparison.jpg"
    keyframe_world_path = review_dir / f"{case_name}_keyframe_world_comparison.jpg"
    keyframe_side_path = review_dir / f"{case_name}_keyframe_side_comparison.jpg"
    if selected_rendered:
        build_contact_sheet(camera_dir, selected_rendered, reasons, keyframe_path)
        build_contact_sheet(world_dir, selected_rendered, reasons, keyframe_world_path)
        build_contact_sheet(side_dir, selected_rendered, reasons, keyframe_side_path)
    timeline_path = review_dir / f"{case_name}_stage_delta_timeline.png"
    plot_case_timeline(case_name, all_rows, timeline_path)
    write_json(
        review_dir / "keyframe_selection.json",
        {
            "selected_frame_ids": selected,
            "rendered_selected_frame_ids": selected_rendered,
            "reasons": {str(key): value for key, value in reasons.items()},
        },
    )

    glb_export = None
    if args.export_keyframe_glb and selected_rendered:
        glb_export = export_keyframe_stage_glbs(
            case_name,
            selected_rendered,
            frame_audit,
            frames,
            object_vertices,
            object_faces,
            chain_poses,
            p14_poses,
            p15_poses,
            hands_by_frame,
            output_root / "glb" / case_name,
        )

    outputs: dict[str, Any] = {
        "camera_frames": str(camera_dir),
        "world_frames": str(world_dir),
        "side_frames": str(side_dir),
        "keyframe_camera_comparison": str(keyframe_path) if keyframe_path.exists() else None,
        "keyframe_world_comparison": str(keyframe_world_path) if keyframe_world_path.exists() else None,
        "keyframe_side_comparison": str(keyframe_side_path) if keyframe_side_path.exists() else None,
        "keyframe_stage_glbs": glb_export,
        "stage_delta_timeline": str(timeline_path),
    }
    video_qc: dict[str, Any] = {}
    if args.encode_video:
        video_dir = output_root / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        camera_video = video_dir / f"{case_name}_camera_pairwise_p14_p15.mp4"
        world_video = video_dir / f"{case_name}_world_pairwise_p14_p15.mp4"
        side_video = video_dir / f"{case_name}_side_pairwise_p14_p15.mp4"
        canonical.encode_video(camera_dir, camera_video, fps, frame_count=150)
        canonical.encode_video(world_dir, world_video, fps, frame_count=150)
        canonical.encode_video(side_dir, side_video, fps, frame_count=150)
        video_qc["camera"] = video_decode_qc(camera_video, 150, fps)
        video_qc["world"] = video_decode_qc(world_video, 150, fps)
        video_qc["side"] = video_decode_qc(side_video, 150, fps)
        outputs["camera_video"] = str(camera_video)
        outputs["world_video"] = str(world_video)
        outputs["side_video"] = str(side_video)

    report = {
        "schema": SCHEMA,
        "status": "ok",
        "case": case_name,
        "run_root": str(run_root),
        "object_id": object_id,
        "claim_scope": (
            "Read-only post-prediction visual/numeric A/B of saved P14 pairwise chain, saved P14 regularized direct pose, "
            "and saved P15 completed timeline under one exact prediction-side camera, full metric MANO state, and "
            "canonical observed metric object surface. No pose values are optimized, clipped, smoothed, or replaced here."
        ),
        "inputs": {
            "p14_report": str(p14_path),
            "p14_report_sha256": sha256_file(p14_path),
            "p15_report": str(p15_path),
            "p15_report_sha256": sha256_file(p15_path),
            "render_state": str(state_path),
            "render_state_sha256": sha256_file(state_path),
            "annotations": str(annotation_path),
            "annotations_sha256": sha256_file(annotation_path),
            "canonical_observed_metric_surface": str(object_mesh_path),
            "canonical_observed_metric_surface_sha256": sha256_file(object_mesh_path),
            "diagnostic_renderer": str(Path(__file__).resolve()),
            "diagnostic_renderer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "geometry_contract": {
            "rendered_object_geometry": "prediction_side_observed_metric_surface_only",
            "generated_geometry_loaded": False,
            "generated_geometry_rendered": False,
            "generated_geometry_pose_evidence_consumed": False,
            "generated_faces_pose_eligible": False,
            "generated_faces_collision_eligible": False,
            "generated_faces_contact_eligible": False,
            "object_mesh": object_mesh_summary,
        },
        "shared_state_contract": {
            "camera": "same annotation camera.T_world_camera_metric and exact scaled intrinsics in all stages",
            "mano": "same full 778-vertex prediction-side metric MANO surfaces in all stages",
            "canonical_object_surface": "same observed metric surface vertices/faces in all stages",
            "only_stage_variable": "saved object SE(3) stage",
            "mano_vertices_modified": False,
            "camera_modified": False,
            "object_vertices_modified": False,
        },
        "stage_reconstruction": reconstruction,
        "numeric_comparison": numeric,
        "render_contract": {
            "rendered_frame_ids": requested_frames,
            "full_timeline": full_timeline,
            "display_size": [display_w, display_h],
            "mask_metric_size": [int(args.metric_mask_width), int(round(source_h * int(args.metric_mask_width) / source_w))],
            "world_size_per_stage": [int(args.world_width), int(args.world_height)],
            "world_projection_axes": {"horizontal": "world_x", "vertical": "world_z", "depth": "world_y"},
            "side_size_per_stage": [int(args.world_width), int(args.world_height)],
            "side_projection_axes": {"horizontal": "world_y", "vertical": "world_z", "depth": "world_x"},
            "object_face_budget": int(args.object_face_budget),
            "mano_face_budget": int(args.mano_face_budget),
            "mask_outline": "prediction-side MANO-subtracted object-owned appearance mask",
            "appearance_mask_is_metric_pose_evidence": False,
            "world_bounds_shared_across_all_three_stages_per_frame": True,
            "same_bounds_used_for_world_and_side_views": True,
            "keyframe_glbs_use_full_topology": bool(args.export_keyframe_glb),
            "visual_inspection_required": True,
        },
        "mano_full_surface": mano_summary,
        "keyframes": {
            "selected_frame_ids": selected,
            "reasons": {str(key): value for key, value in reasons.items()},
        },
        "outputs": outputs,
        "encoded_video_qc": video_qc,
        "frame_rows": all_rows,
        "elapsed_s": time.time() - started,
    }
    report_path = case_dir / "P14_PAIRWISE_REGULARIZED_P15_AB_REPORT.json"
    write_json(report_path, report)
    report["outputs"]["case_report"] = str(report_path)
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "case": case_name,
                "stage_reconstruction": reconstruction,
                "numeric_comparison": numeric,
                "outputs": outputs,
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def write_readme(output_root: Path, reports: list[dict[str, Any]]) -> Path:
    lines = [
        "# P14 pairwise chain → P14 regularized → P15 completed 可视化验证",
        "",
        "本目录是只读 post-prediction 诊断；未修改五个 finalized case root。",
        "",
        "## 固定量",
        "",
        "- 每个 stage 使用同一 prediction-side sensor camera。",
        "- 每个 stage 使用同一左右完整 778-vertex metric MANO。",
        "- 每个 stage 使用同一 canonical observed metric object surface。",
        "- 未加载或渲染 SAM3D/TRELLIS generated hidden geometry。",
        "- 唯一变化量是保存下来的 object SE(3) stage。",
        "",
        "## 面板",
        "",
        "1. RGB reference：白线是 object-owned appearance mask；青线为 pairwise，绿线为 P14，紫线仅标 P15 completed-only。",
        "2. P14 pairwise chain：anchor 起始、未经 P14 anchor/unary/temporal regularization 的 raw observed-surface chain。",
        "3. P14 regularized：P14 保存的 direct observed pose。",
        "4. P15 completed：direct 帧与 P14 SE(3) 完全相同；缺失帧显示 interpolation/hold completion。",
        "",
        "绿色是 observed metric object surface；蓝/橙是固定的完整左右 MANO；白线是 appearance mask。",
        "",
        "## 数值边界",
        "",
        "- accepted metric-surface residual 只在 P14 direct eligible metric rows 上计算。",
        "- mask IoU 仅为 appearance diagnostic，不恢复 rejected depth，也不是 metric pose evidence。",
        "- hand/object 距离是 unsigned vertex proximity，不是 signed contact、penetration 或 nonpenetration。",
        "- P15 completed rows 明确是 uncertain temporal completion，不冒充 direct evidence。",
        "",
        "## Case 概览",
        "",
        "| case | pairwise rows | P14 direct rows | P15 completed rows | P14→P15 direct SE(3) |",
        "|---|---:|---:|---:|---|",
    ]
    for report in reports:
        reconstruction = report["stage_reconstruction"]
        lines.append(
            f"| {report['case']} | {reconstruction['pairwise_chain_pose_count']} | "
            f"{reconstruction['p15_direct_pose_count']} | {reconstruction['p15_completed_pose_count']} | "
            f"{'exact identity' if reconstruction['p14_p15_direct_se3_exact_identity'] else 'DIFF'} |"
        )
    lines.extend(
        [
            "",
            "Milk 的 finalized P14 是旧版 independent per-frame schema，没有保存 pairwise chain/correction trace。",
            "其 pairwise 栏因此显式显示 unavailable；没有用新版算法重跑后冒充旧 P14 stage。",
            "",
            "## 入口",
            "",
            "- `videos/`：每例 camera/world/side 三个 150-frame、30 FPS 主视频。",
            "- `cases/<case>/review/*keyframe*.jpg`：camera/world/side 关键帧对比 sheet。",
            "- `glb/<case>/`：所选关键帧的 full-topology world-frame stage GLB；direct P14/P15 文件 byte-identical。",
            "- `cases/<case>/review/*timeline.png`：rotation、translation、surface residual、mask IoU timeline。",
            "- `cases/<case>/P14_PAIRWISE_REGULARIZED_P15_AB_REPORT.json`：逐帧可审计数值与 provenance。",
            "- `COLLECTION_P14_STAGE_AB_REPORT.json`：统一汇总。",
            "",
        ]
    )
    path = output_root / "README_ZH.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    started = time.time()
    collection_root = args.collection_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.replace:
            raise RuntimeError(f"refusing to overwrite non-empty output root: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_cases = [name for name in CASE_ORDER if name in set(args.cases)]
    if not selected_cases:
        raise RuntimeError("no cases selected")
    if int(args.workers) < 1:
        raise RuntimeError("--workers must be >=1")
    reports_by_case: dict[str, dict[str, Any]] = {}
    run_roots: dict[str, str] = {}
    resolved_run_roots: dict[str, Path] = {}
    for case_name in selected_cases:
        run_link = collection_root / "runs" / case_name
        if not run_link.is_symlink():
            raise RuntimeError(f"collection case is not a symlink: {run_link}")
        run_root = run_link.resolve(strict=True)
        run_roots[case_name] = str(run_root)
        resolved_run_roots[case_name] = run_root
    worker_count = min(int(args.workers), len(selected_cases))
    if worker_count == 1:
        for case_name in selected_cases:
            reports_by_case[case_name] = render_case(
                case_name, resolved_run_roots[case_name], output_root, args
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_case = {
                executor.submit(
                    render_case,
                    case_name,
                    resolved_run_roots[case_name],
                    output_root,
                    args,
                ): case_name
                for case_name in selected_cases
            }
            for future in concurrent.futures.as_completed(future_to_case):
                case_name = future_to_case[future]
                reports_by_case[case_name] = future.result()
                print(f"completed case worker: {case_name}", flush=True)
    reports = [reports_by_case[case_name] for case_name in selected_cases]

    readme_path = write_readme(output_root, reports)
    aggregate = {
        "schema": SCHEMA,
        "status": "ok",
        "collection_root": str(collection_root),
        "output_root": str(output_root),
        "cases": selected_cases,
        "case_worker_count": worker_count,
        "run_roots": run_roots,
        "claim_scope": (
            "Read-only visual and numeric stage audit. This output does not change or replace any finalized prediction artifact."
        ),
        "case_summaries": [
            {
                "case": report["case"],
                "object_id": report["object_id"],
                "stage_reconstruction": report["stage_reconstruction"],
                "numeric_comparison": report["numeric_comparison"],
                "encoded_video_qc": report["encoded_video_qc"],
                "case_report": report["outputs"]["case_report"],
                "keyframe_camera_comparison": report["outputs"]["keyframe_camera_comparison"],
                "keyframe_world_comparison": report["outputs"]["keyframe_world_comparison"],
                "keyframe_side_comparison": report["outputs"]["keyframe_side_comparison"],
                "keyframe_stage_glbs": report["outputs"]["keyframe_stage_glbs"],
                "stage_delta_timeline": report["outputs"]["stage_delta_timeline"],
            }
            for report in reports
        ],
        "global_contract": {
            "finalized_case_roots_modified": False,
            "generated_geometry_loaded_or_rendered": False,
            "generated_geometry_pose_evidence_consumed": False,
            "GT_consumed": False,
            "camera_MANO_and_canonical_surface_fixed_within_case": True,
            "P15_direct_rows_required_to_be_exactly_identical_to_P14": True,
            "legacy_missing_pairwise_state_substituted": False,
            "visual_inspection_required": True,
        },
        "outputs": {
            "readme_zh": str(readme_path),
            "videos": sorted(str(path) for path in (output_root / "videos").glob("*.mp4")),
        },
        "elapsed_s": time.time() - started,
    }
    aggregate_path = output_root / "COLLECTION_P14_STAGE_AB_REPORT.json"
    write_json(aggregate_path, aggregate)

    declared_files = [aggregate_path, readme_path]
    declared_files.extend(output_root.glob("videos/*.mp4"))
    declared_files.extend(output_root.glob("cases/*/P14_PAIRWISE_REGULARIZED_P15_AB_REPORT.json"))
    declared_files.extend(output_root.glob("cases/*/review/*.jpg"))
    declared_files.extend(output_root.glob("cases/*/review/*.png"))
    declared_files.extend(output_root.glob("cases/*/review/*.json"))
    declared_files.extend(output_root.glob("glb/*/*.glb"))
    declared_files.extend(output_root.glob("glb/*/*.json"))
    hash_rows = [
        {
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(set(declared_files))
        if path.is_file()
    ]
    hash_index = {
        "schema": "selected_deliverable_sha256_index_v1",
        "scope": "reports, videos, keyframe sheets, timelines, and review metadata; per-frame JPEG intermediates excluded",
        "entry_count": len(hash_rows),
        "entries": hash_rows,
    }
    hash_path = output_root / "ARTIFACT_SHA256_INDEX.json"
    write_json(hash_path, hash_index)
    done = {
        "schema": SCHEMA,
        "status": "complete_visual_review_pending",
        "case_count": len(reports),
        "video_count": len(aggregate["outputs"]["videos"]),
        "collection_report": str(aggregate_path),
        "collection_report_sha256": sha256_file(aggregate_path),
        "artifact_sha256_index": str(hash_path),
        "artifact_sha256_index_sha256": sha256_file(hash_path),
        "visual_review_pending": True,
        "elapsed_s": time.time() - started,
    }
    done_path = output_root / "STAGE_AB_RENDER_DONE.json"
    write_json(done_path, done)
    print(
        json.dumps(
            {
                "status": done["status"],
                "output_root": str(output_root),
                "cases": selected_cases,
                "videos": aggregate["outputs"]["videos"],
                "aggregate_report": str(aggregate_path),
                "elapsed_s": done["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
