#!/usr/bin/env python3
"""Evaluate a frozen UniDepth-vs-HOT3D-oracle-depth P09--P15 A/B.

Both branches are expected to share RGB, official pinhole K, HaWoR, OWLv2,
SAM2, anchor, raw TRELLIS output, and seed.  This evaluator scores (1) raw depth
against rendered target depth, (2) P09 camera-frame surfels against the released
CAD, (3) P14/P15 camera-relative object trajectories up to one fixed canonical
frame transform, and (4) the adapted pose-hypothesis mesh against the CAD.

Released HOT3D poses/CAD are consumed only here, after the baseline prediction
is frozen.  Results are conditional/oracle diagnostics, not RGB-only endpoint
accuracy and not contact/nonpenetration certification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import trimesh


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def summarize(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def distance_summary(values: Iterable[float]) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    out = summarize(vals.tolist())
    if vals.size:
        out["fraction_le_0p002m"] = float(np.mean(vals <= 0.002))
        out["fraction_le_0p005m"] = float(np.mean(vals <= 0.005))
        out["fraction_le_0p010m"] = float(np.mean(vals <= 0.010))
        out["rmse_m"] = float(np.sqrt(np.mean(vals * vals)))
    return out


def quat_wxyz_to_R(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        raise RuntimeError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(max(1e-15, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            q = np.asarray([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(max(1e-15, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            q = np.asarray([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = math.sqrt(max(1e-15, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            q = np.asarray([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def average_quaternions(quats: list[np.ndarray]) -> np.ndarray:
    if not quats:
        raise RuntimeError("cannot average zero quaternions")
    ref = quats[0] / np.linalg.norm(quats[0])
    A = np.zeros((4, 4), dtype=np.float64)
    for q in quats:
        q = q / np.linalg.norm(q)
        if float(np.dot(q, ref)) < 0.0:
            q = -q
        A += np.outer(q, q)
    _, vectors = np.linalg.eigh(A)
    q = vectors[:, -1]
    q /= np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def make_T(R: np.ndarray, t: Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(list(t), dtype=np.float64)
    return T


def transform_from_dict(payload: dict[str, Any]) -> np.ndarray:
    return make_T(quat_wxyz_to_R(payload["quaternion_wxyz"]), payload["translation_xyz"])


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_angle_rad(T: np.ndarray) -> float:
    return math.acos(float(np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(str(path), process=True, merge_primitives=True)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_mesh() if hasattr(loaded, "to_mesh") else trimesh.util.concatenate(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"unsupported mesh payload {type(loaded)} from {path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"empty mesh: {path}")
    return mesh


def load_gt(sidecar_path: Path, object_uid: str, stream_id: str) -> dict[int, dict[str, Any]]:
    data = load_json(sidecar_path)
    out: dict[int, dict[str, Any]] = {}
    for frame in data.get("frames") or []:
        idx = int(frame["frame_idx"])
        j = frame.get("json") or {}
        cam = (j.get("cameras.json") or {}).get(stream_id)
        match = None
        for entries in (j.get("objects.json") or {}).values():
            for entry in entries if isinstance(entries, list) else []:
                if str(entry.get("object_uid")) == object_uid:
                    match = entry
        if cam and match:
            T_wc = transform_from_dict(cam["T_world_from_camera"])
            T_wo = transform_from_dict(match["T_world_from_object"])
            out[idx] = {
                "T_world_camera": T_wc,
                "T_world_object": T_wo,
                "T_camera_object": np.linalg.inv(T_wc) @ T_wo,
                "visibility": (match.get("visibilities_modeled") or {}).get(stream_id),
                "object_name": match.get("object_name"),
                "object_bop_id": match.get("object_bop_id"),
            }
    if not out:
        raise RuntimeError(f"no GT rows for object UID {object_uid}")
    return out


def annotation_frames(path: Path) -> dict[int, dict[str, Any]]:
    data = load_json(path)
    return {int(frame.get("frame_idx", pos)): frame for pos, frame in enumerate(data.get("frames") or []) if isinstance(frame, dict)}


def annotation_camera(frame: dict[str, Any]) -> np.ndarray:
    camera = frame.get("camera") or {}
    value = camera.get("T_world_camera_metric") or camera.get("T_world_camera")
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        raise RuntimeError("annotation frame has invalid camera transform")
    return arr


def annotation_K(frame: dict[str, Any]) -> np.ndarray:
    value = np.asarray((frame.get("camera") or {}).get("intrinsics_fx_fy_cx_cy"), dtype=np.float64).reshape(-1)
    if value.shape != (4,):
        raise RuntimeError("annotation frame has invalid intrinsics")
    fx, fy, cx, cy = value
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def visible_candidate(frame: dict[str, Any], object_id: str | None) -> dict[str, Any] | None:
    objects = frame.get("objects") or []
    for obj in objects:
        if object_id is not None and str(obj.get("object_id")) != object_id:
            continue
        candidate = obj.get("visible_geometry_candidate")
        if isinstance(candidate, dict) and candidate.get("world_vertices_sample_m"):
            return candidate
    return None


def nearest_surface_distances(query: trimesh.proximity.ProximityQuery, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.asarray([], dtype=np.float64)
    _, distance, _ = query.on_surface(np.asarray(points, dtype=np.float64))
    distance = np.asarray(distance, dtype=np.float64)
    return distance[np.isfinite(distance)]


def evaluate_p09(
    annotations_path: Path,
    gt: dict[int, dict[str, Any]],
    cad_query: trimesh.proximity.ProximityQuery,
    object_id: str | None,
    max_points_per_frame: int,
    review_frames: set[int],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    frames = annotation_frames(annotations_path)
    rows: list[dict[str, Any]] = []
    global_distances: list[np.ndarray] = []
    review: dict[int, dict[str, Any]] = {}
    for idx in sorted(set(frames) & set(gt)):
        candidate = visible_candidate(frames[idx], object_id)
        if candidate is None:
            continue
        points_world = np.asarray(candidate["world_vertices_sample_m"], dtype=np.float64)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            continue
        if len(points_world) > max_points_per_frame:
            take = np.linspace(0, len(points_world) - 1, max_points_per_frame, dtype=int)
            points_world = points_world[take]
        T_pred_wc = annotation_camera(frames[idx])
        points_cam = transform_points(np.linalg.inv(T_pred_wc), points_world)
        points_gt_object = transform_points(np.linalg.inv(gt[idx]["T_camera_object"]), points_cam)
        distances = nearest_surface_distances(cad_query, points_gt_object)
        if distances.size == 0:
            continue
        global_distances.append(distances)
        row = {
            "frame_idx": idx,
            "point_count": int(len(distances)),
            "surfel_to_gt_cad_unsigned_m": distance_summary(distances.tolist()),
            "camera_z_m": summarize(points_cam[:, 2].tolist()),
            "gt_visibility": gt[idx].get("visibility"),
        }
        rows.append(row)
        if idx in review_frames:
            review[idx] = {
                "points_camera": points_cam,
                "distance_m": distances,
                "K": annotation_K(frames[idx]),
                "raw_frame_path": frames[idx].get("raw_frame_path"),
            }
    combined = np.concatenate(global_distances) if global_distances else np.asarray([], dtype=np.float64)
    medians = [r["surfel_to_gt_cad_unsigned_m"]["median"] for r in rows]
    return {
        "annotations": str(annotations_path),
        "frame_count": len(rows),
        "sample_count": int(len(combined)),
        "surfel_to_gt_cad_unsigned_m": distance_summary(combined.tolist()),
        "per_frame_median_m": summarize(medians),
        "rows": rows,
    }, review


def load_pose_rows(path: Path) -> dict[int, dict[str, Any]]:
    data = load_json(path)
    out = {}
    for row in data.get("pose_rows") or []:
        if not isinstance(row, dict) or "frame_idx" not in row:
            continue
        R = np.asarray(row.get("rotation_world_from_completed_canonical_matrix"), dtype=np.float64)
        t = np.asarray(row.get("translation_world_m"), dtype=np.float64)
        if R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all():
            out[int(row["frame_idx"])] = row
    return out


def pose_T(row: dict[str, Any]) -> np.ndarray:
    return make_T(row["rotation_world_from_completed_canonical_matrix"], row["translation_world_m"])


def fit_constant_transform(Xs: list[np.ndarray]) -> np.ndarray:
    q = average_quaternions([R_to_quat_wxyz(X[:3, :3]) for X in Xs])
    t = np.mean([X[:3, 3] for X in Xs], axis=0)
    return make_T(quat_wxyz_to_R(q), t)


def trajectory_residual_rows(
    pose_rows: dict[int, dict[str, Any]],
    annotations: dict[int, dict[str, Any]],
    gt: dict[int, dict[str, Any]],
    anchor_frame: int,
) -> dict[str, Any]:
    common = sorted(set(pose_rows) & set(annotations) & set(gt))
    if not common:
        return {"frame_count": 0, "rows": []}
    pred_cam_obj: dict[int, np.ndarray] = {}
    gt_cam_obj: dict[int, np.ndarray] = {}
    Xs: list[np.ndarray] = []
    for idx in common:
        P = np.linalg.inv(annotation_camera(annotations[idx])) @ pose_T(pose_rows[idx])
        G = gt[idx]["T_camera_object"]
        pred_cam_obj[idx] = P
        gt_cam_obj[idx] = G
        Xs.append(np.linalg.inv(P) @ G)
    X_best = fit_constant_transform(Xs)
    anchor_used = anchor_frame if anchor_frame in common else min(common, key=lambda i: abs(i - anchor_frame))
    X_anchor = np.linalg.inv(pred_cam_obj[anchor_used]) @ gt_cam_obj[anchor_used]

    def residuals(X: np.ndarray) -> tuple[list[dict[str, Any]], list[float], list[float], list[float]]:
        rows, trans, rot, depth = [], [], [], []
        for idx in common:
            residual = np.linalg.inv(gt_cam_obj[idx]) @ (pred_cam_obj[idx] @ X)
            translation = float(np.linalg.norm(residual[:3, 3]))
            rotation = rotation_angle_rad(residual)
            depth_delta = float((pred_cam_obj[idx] @ X)[2, 3] - gt_cam_obj[idx][2, 3])
            trans.append(translation)
            rot.append(rotation)
            depth.append(depth_delta)
            rows.append({
                "frame_idx": idx,
                "translation_residual_m": translation,
                "rotation_residual_rad": rotation,
                "rotation_residual_deg": math.degrees(rotation),
                "object_origin_depth_delta_m": depth_delta,
                "pose_status": pose_rows[idx].get("status"),
                "pose_measurement_status": pose_rows[idx].get("pose_measurement_status"),
            })
        return rows, trans, rot, depth

    anchor_rows, anchor_t, anchor_r, anchor_depth = residuals(X_anchor)
    best_rows, best_t, best_r, best_depth = residuals(X_best)
    variability_t = [float(np.linalg.norm(X[:3, 3] - X_best[:3, 3])) for X in Xs]
    variability_r = [rotation_angle_rad(np.linalg.inv(X_best) @ X) for X in Xs]
    return {
        "frame_count": len(common),
        "frame_idx_range": [common[0], common[-1]],
        "anchor_frame_requested": anchor_frame,
        "anchor_frame_used": anchor_used,
        "anchor_aligned": {
            "fixed_transform_gt_object_to_predicted_canonical": X_anchor.tolist(),
            "translation_residual_m": summarize(anchor_t),
            "rotation_residual_rad": summarize(anchor_r),
            "rotation_residual_deg": summarize([math.degrees(v) for v in anchor_r]),
            "object_origin_depth_delta_m": summarize(anchor_depth),
            "rows": anchor_rows,
        },
        "best_all_frame_fixed_transform": {
            "fixed_transform_gt_object_to_predicted_canonical": X_best.tolist(),
            "translation_residual_m": summarize(best_t),
            "rotation_residual_rad": summarize(best_r),
            "rotation_residual_deg": summarize([math.degrees(v) for v in best_r]),
            "object_origin_depth_delta_m": summarize(best_depth),
            "rows": best_rows,
        },
        "relative_transform_variability": {
            "translation_deviation_m": summarize(variability_t),
            "rotation_deviation_rad": summarize(variability_r),
            "rotation_deviation_deg": summarize([math.degrees(v) for v in variability_r]),
        },
    }


def evaluate_trajectory(report_path: Path, annotations_path: Path, gt: dict[int, dict[str, Any]], anchor_frame: int) -> dict[str, Any]:
    out = trajectory_residual_rows(load_pose_rows(report_path), annotation_frames(annotations_path), gt, anchor_frame)
    out["pose_report"] = str(report_path)
    out["annotations"] = str(annotations_path)
    return out


def resolve_pose_mesh(completion_report: Path) -> Path:
    data = load_json(completion_report)
    outputs = data.get("outputs") or {}
    value = outputs.get("pose_hypothesis_mesh_labeled") or outputs.get("completed_mesh_labeled") or outputs.get("completed_mesh")
    if not value:
        raise RuntimeError(f"completion report lacks pose mesh: {completion_report}")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def evaluate_mesh_shape(
    completion_report: Path,
    trajectory: dict[str, Any],
    cad: trimesh.Trimesh,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    mesh_path = resolve_pose_mesh(completion_report)
    pred = load_mesh(mesh_path)
    X = np.asarray(
        trajectory["best_all_frame_fixed_transform"]["fixed_transform_gt_object_to_predicted_canonical"],
        dtype=np.float64,
    )
    pred_to_gt = np.linalg.inv(X)
    pred_in_gt = pred.copy()
    pred_in_gt.apply_transform(pred_to_gt)
    np.random.seed(seed)
    pred_samples, _ = trimesh.sample.sample_surface(pred_in_gt, sample_count)
    np.random.seed(seed + 1)
    cad_samples, _ = trimesh.sample.sample_surface(cad, sample_count)
    pred_to_cad = nearest_surface_distances(trimesh.proximity.ProximityQuery(cad), pred_samples)
    cad_to_pred = nearest_surface_distances(trimesh.proximity.ProximityQuery(pred_in_gt), cad_samples)
    chamfer_mean = float(np.mean(pred_to_cad) + np.mean(cad_to_pred)) / 2.0
    fscore = {}
    for threshold in (0.002, 0.005, 0.010):
        precision = float(np.mean(pred_to_cad <= threshold))
        recall = float(np.mean(cad_to_pred <= threshold))
        score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        fscore[f"threshold_{threshold:.3f}m"] = {"precision": precision, "recall": recall, "fscore": score}
    return {
        "completion_report": str(completion_report),
        "pose_hypothesis_mesh": str(mesh_path),
        "predicted_mesh_watertight": bool(pred.is_watertight),
        "predicted_mesh_faces": int(len(pred.faces)),
        "alignment_source": "P15 best fixed transform between predicted canonical and HOT3D object frame",
        "sample_count_each_direction": sample_count,
        "predicted_surface_to_gt_cad_m": distance_summary(pred_to_cad.tolist()),
        "gt_cad_to_predicted_surface_m": distance_summary(cad_to_pred.tolist()),
        "symmetric_chamfer_mean_m": chamfer_mean,
        "fscore": fscore,
    }


def evaluate_raw_depth(unidepth_path: Path, oracle_path: Path, max_global_samples: int) -> dict[str, Any]:
    with np.load(unidepth_path) as ua, np.load(oracle_path) as ob:
        ua_frame_idx = np.asarray(ua["frame_idx"])
        ob_frame_idx = np.asarray(ob["frame_idx"])
        # NPZ members are compressed as whole arrays.  Materialize each once;
        # indexing ``archive['depth']`` inside the loop would repeatedly inflate it.
        ua_depth = np.asarray(ua["depth"])
        ob_depth = np.asarray(ob["depth"])
        ob_mask = np.asarray(ob["valid_target_surface_mask"], dtype=bool)
        ua_idx = {int(v): i for i, v in enumerate(ua_frame_idx.tolist())}
        ob_idx = {int(v): i for i, v in enumerate(ob_frame_idx.tolist())}
        common = sorted(set(ua_idx) & set(ob_idx))
        rows = []
        global_abs: list[np.ndarray] = []
        global_signed: list[np.ndarray] = []
        ratios: list[np.ndarray] = []
        for idx in common:
            u = np.asarray(ua_depth[ua_idx[idx]], dtype=np.float32)
            o = np.asarray(ob_depth[ob_idx[idx]], dtype=np.float32)
            mask = ob_mask[ob_idx[idx]]
            valid = mask & np.isfinite(u) & (u > 0.0) & np.isfinite(o) & (o > 0.0)
            if not np.any(valid):
                continue
            signed = (u[valid] - o[valid]).astype(np.float32)
            absolute = np.abs(signed)
            ratio = (u[valid] / o[valid]).astype(np.float32)
            rows.append({
                "frame_idx": idx,
                "valid_target_pixel_count": int(np.count_nonzero(valid)),
                "unidepth_minus_oracle_m": summarize(signed.tolist()),
                "absolute_depth_error_m": distance_summary(absolute.tolist()),
                "unidepth_over_oracle_ratio": summarize(ratio.tolist()),
            })
            step = max(1, int(math.ceil(len(absolute) / max(1, max_global_samples // max(1, len(common))))))
            global_abs.append(absolute[::step])
            global_signed.append(signed[::step])
            ratios.append(ratio[::step])
        abs_values = np.concatenate(global_abs) if global_abs else np.asarray([], dtype=np.float32)
        signed_values = np.concatenate(global_signed) if global_signed else np.asarray([], dtype=np.float32)
        ratio_values = np.concatenate(ratios) if ratios else np.asarray([], dtype=np.float32)
    return {
        "unidepth_npz": str(unidepth_path),
        "oracle_depth_npz": str(oracle_path),
        "frame_count": len(rows),
        "global_summary_sampling_policy": f"deterministic per-frame stride, <= approximately {max_global_samples} samples",
        "absolute_depth_error_m": distance_summary(abs_values.tolist()),
        "unidepth_minus_oracle_m": summarize(signed_values.tolist()),
        "unidepth_over_oracle_ratio": summarize(ratio_values.tolist()),
        "rows": rows,
    }


def lower_is_better(a: float | None, b: float | None) -> dict[str, Any]:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return {"available": False}
    return {
        "available": True,
        "unidepth": float(a),
        "oracle": float(b),
        "oracle_minus_unidepth": float(b - a),
        "relative_reduction": float((a - b) / a) if abs(a) > 1e-12 else None,
        "oracle_better": bool(b < a),
    }


def comparison_summary(branches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a = branches["unidepth_official_K"]
    b = branches["oracle_depth_official_K"]
    return {
        "p09_global_surfel_to_cad_median_m": lower_is_better(
            a["p09"]["surfel_to_gt_cad_unsigned_m"].get("median"),
            b["p09"]["surfel_to_gt_cad_unsigned_m"].get("median"),
        ),
        "p14_anchor_aligned_translation_median_m": lower_is_better(
            a["p14"]["anchor_aligned"]["translation_residual_m"].get("median"),
            b["p14"]["anchor_aligned"]["translation_residual_m"].get("median"),
        ),
        "p14_anchor_aligned_rotation_median_deg": lower_is_better(
            a["p14"]["anchor_aligned"]["rotation_residual_deg"].get("median"),
            b["p14"]["anchor_aligned"]["rotation_residual_deg"].get("median"),
        ),
        "p15_anchor_aligned_translation_median_m": lower_is_better(
            a["p15"]["anchor_aligned"]["translation_residual_m"].get("median"),
            b["p15"]["anchor_aligned"]["translation_residual_m"].get("median"),
        ),
        "p15_anchor_aligned_rotation_median_deg": lower_is_better(
            a["p15"]["anchor_aligned"]["rotation_residual_deg"].get("median"),
            b["p15"]["anchor_aligned"]["rotation_residual_deg"].get("median"),
        ),
        "adapted_mesh_symmetric_chamfer_mean_m": lower_is_better(
            a["mesh_shape"]["symmetric_chamfer_mean_m"],
            b["mesh_shape"]["symmetric_chamfer_mean_m"],
        ),
    }


def project_points(points_camera: np.ndarray, K: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points_camera, dtype=np.float64)
    valid = np.isfinite(p).all(axis=1) & (p[:, 2] > 1e-5)
    p = p[valid]
    if len(p) == 0:
        return np.empty((0, 2), dtype=np.int32), valid
    uv = np.column_stack((K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]))
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    original_valid_indices = np.where(valid)[0][inside]
    final_mask = np.zeros(len(valid), dtype=bool)
    final_mask[original_valid_indices] = True
    return np.rint(uv[inside]).astype(np.int32), final_mask


def paint_points(image: np.ndarray, uv: np.ndarray, colors: np.ndarray, radius: int = 2) -> None:
    for (u, v), color in zip(uv.tolist(), colors.tolist()):
        cv2.circle(image, (int(u), int(v)), radius, tuple(int(c) for c in color), -1, cv2.LINE_AA)


def p09_review(
    reviews: dict[str, dict[int, dict[str, Any]]],
    frames: list[int],
    output: Path,
) -> None:
    lines = []
    for idx in frames:
        panels = []
        for branch, title in (("unidepth_official_K", "A UniDepth + official K"), ("oracle_depth_official_K", "B HOT3D oracle depth + official K")):
            row = reviews.get(branch, {}).get(idx)
            if row is None:
                continue
            image = cv2.imread(str(row["raw_frame_path"]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            points = row["points_camera"]
            distances = row["distance_m"]
            if len(points) > 1000:
                take = np.linspace(0, len(points) - 1, 1000, dtype=int)
                points, distances = points[take], distances[take]
            uv, valid = project_points(points, row["K"], image.shape[1], image.shape[0])
            d = distances[valid]
            colors = np.zeros((len(d), 3), dtype=np.uint8)
            colors[:] = (20, 20, 230)
            colors[d <= 0.010] = (0, 210, 255)
            colors[d <= 0.005] = (30, 220, 30)
            paint_points(image, uv, colors, radius=2)
            banner = np.zeros((58, image.shape[1], 3), dtype=np.uint8)
            cv2.putText(banner, f"{title} | frame {idx} | CAD residual med={np.median(distances)*1000:.2f} mm p90={np.quantile(distances,.9)*1000:.2f} mm", (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255,255,255), 2, cv2.LINE_AA)
            panel = np.vstack([banner, image])
            panel = cv2.resize(panel, (640, int(round(panel.shape[0] * 640 / panel.shape[1]))), interpolation=cv2.INTER_AREA)
            panels.append(panel)
        if len(panels) == 2:
            lines.append(np.hstack(panels))
    if lines:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), np.vstack(lines), [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def sample_vertices(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) <= count:
        return vertices
    return vertices[np.linspace(0, len(vertices) - 1, count, dtype=int)]


def pose_review(
    branch_inputs: dict[str, dict[str, Any]],
    gt: dict[int, dict[str, Any]],
    cad: trimesh.Trimesh,
    frames: list[int],
    output: Path,
) -> None:
    gt_points = sample_vertices(cad, 5000)
    lines = []
    for idx in frames:
        panels = []
        for branch, title in (("unidepth_official_K", "A UniDepth P15"), ("oracle_depth_official_K", "B oracle-depth P15")):
            item = branch_inputs[branch]
            anns = item["annotations"]
            poses = item["poses"]
            if idx not in anns or idx not in poses or idx not in gt:
                continue
            frame = anns[idx]
            image = cv2.imread(str(frame.get("raw_frame_path")), cv2.IMREAD_COLOR)
            if image is None:
                continue
            K = annotation_K(frame)
            pred_mesh_points = item["mesh_points"]
            T_pred_cam_obj = np.linalg.inv(annotation_camera(frame)) @ pose_T(poses[idx])
            pred_cam = transform_points(T_pred_cam_obj, pred_mesh_points)
            gt_cam = transform_points(gt[idx]["T_camera_object"], gt_points)
            gt_uv, _ = project_points(gt_cam, K, image.shape[1], image.shape[0])
            pred_uv, _ = project_points(pred_cam, K, image.shape[1], image.shape[0])
            if len(gt_uv):
                paint_points(image, gt_uv[::2], np.repeat(np.asarray([[20,220,20]],dtype=np.uint8), len(gt_uv[::2]), axis=0), radius=1)
            if len(pred_uv):
                paint_points(image, pred_uv[::2], np.repeat(np.asarray([[230,30,220]],dtype=np.uint8), len(pred_uv[::2]), axis=0), radius=1)
            banner = np.zeros((58, image.shape[1], 3), dtype=np.uint8)
            cv2.putText(banner, f"{title} | frame {idx} | green=HOT3D CAD, magenta=predicted adapted mesh", (14,38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255,255,255), 2, cv2.LINE_AA)
            panel = np.vstack([banner,image])
            panel = cv2.resize(panel, (640, int(round(panel.shape[0]*640/panel.shape[1]))), interpolation=cv2.INTER_AREA)
            panels.append(panel)
        if len(panels)==2:
            lines.append(np.hstack(panels))
    if lines:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), np.vstack(lines), [int(cv2.IMWRITE_JPEG_QUALITY),94])


def timeline_plot(report: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    colors = {"unidepth_official_K": "tab:red", "oracle_depth_official_K": "tab:green"}
    labels = {"unidepth_official_K": "A UniDepth + official K", "oracle_depth_official_K": "B oracle depth + official K"}
    for branch, data in report["branches"].items():
        p09 = data["p09"]["rows"]
        axes[0].plot([r["frame_idx"] for r in p09], [1000*r["surfel_to_gt_cad_unsigned_m"]["median"] for r in p09], color=colors[branch], label=labels[branch])
        p15 = data["p15"]["anchor_aligned"]["rows"]
        axes[1].plot([r["frame_idx"] for r in p15], [1000*r["translation_residual_m"] for r in p15], color=colors[branch], label=labels[branch])
        axes[2].plot([r["frame_idx"] for r in p15], [r["rotation_residual_deg"] for r in p15], color=colors[branch], label=labels[branch])
    depth_rows = report["raw_depth_intervention"]["rows"]
    axes[3].plot([r["frame_idx"] for r in depth_rows], [1000*r["absolute_depth_error_m"]["median"] for r in depth_rows], color="tab:blue", label="UniDepth vs rendered target depth")
    axes[0].set_ylabel("P09 surfel→CAD median (mm)")
    axes[1].set_ylabel("P15 anchor-aligned translation (mm)")
    axes[2].set_ylabel("P15 anchor-aligned rotation (deg)")
    axes[3].set_ylabel("raw depth median abs error (mm)")
    axes[3].set_xlabel("frame")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    fig.suptitle("HOT3D frozen-upstream depth-source A/B (conditional oracle branch)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unidepth-depth", type=Path, required=True)
    p.add_argument("--oracle-depth", type=Path, required=True)
    p.add_argument("--unidepth-annotations", type=Path, required=True)
    p.add_argument("--oracle-annotations", type=Path, required=True)
    p.add_argument("--unidepth-p14", type=Path, required=True)
    p.add_argument("--oracle-p14", type=Path, required=True)
    p.add_argument("--unidepth-p15", type=Path, required=True)
    p.add_argument("--oracle-p15", type=Path, required=True)
    p.add_argument("--unidepth-completion", type=Path, required=True)
    p.add_argument("--oracle-completion", type=Path, required=True)
    p.add_argument("--hot3d-gt", type=Path, required=True)
    p.add_argument("--cad", type=Path, required=True)
    p.add_argument("--object-uid", required=True)
    p.add_argument("--object-id", default=None)
    p.add_argument("--stream-id", default="214-1")
    p.add_argument("--anchor-frame", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--review-frames", type=int, nargs="*", default=[0, 37, 75, 112, 149])
    p.add_argument("--max-p09-points-per-frame", type=int, default=1500)
    p.add_argument("--mesh-sample-count", type=int, default=20000)
    p.add_argument("--max-global-depth-samples", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cad = load_mesh(args.cad)
    cad_query = trimesh.proximity.ProximityQuery(cad)
    gt = load_gt(args.hot3d_gt, str(args.object_uid), args.stream_id)
    review_frames = set(args.review_frames)
    p09_a, review_a = evaluate_p09(args.unidepth_annotations, gt, cad_query, args.object_id, args.max_p09_points_per_frame, review_frames)
    p09_b, review_b = evaluate_p09(args.oracle_annotations, gt, cad_query, args.object_id, args.max_p09_points_per_frame, review_frames)
    p14_a = evaluate_trajectory(args.unidepth_p14, args.unidepth_annotations, gt, args.anchor_frame)
    p14_b = evaluate_trajectory(args.oracle_p14, args.oracle_annotations, gt, args.anchor_frame)
    p15_a = evaluate_trajectory(args.unidepth_p15, args.unidepth_annotations, gt, args.anchor_frame)
    p15_b = evaluate_trajectory(args.oracle_p15, args.oracle_annotations, gt, args.anchor_frame)
    mesh_a = evaluate_mesh_shape(args.unidepth_completion, p15_a, cad, args.mesh_sample_count, args.seed)
    mesh_b = evaluate_mesh_shape(args.oracle_completion, p15_b, cad, args.mesh_sample_count, args.seed)
    branches = {
        "unidepth_official_K": {"p09": p09_a, "p14": p14_a, "p15": p15_a, "mesh_shape": mesh_a},
        "oracle_depth_official_K": {"p09": p09_b, "p14": p14_b, "p15": p15_b, "mesh_shape": mesh_b},
    }
    raw_depth = evaluate_raw_depth(args.unidepth_depth, args.oracle_depth, args.max_global_depth_samples)
    report = {
        "status": "ok",
        "method": "evaluate_v19_hot3d_depth_source_ab",
        "claim_scope": "frozen-upstream downstream-conditional UniDepth-vs-rendered-oracle-depth A/B; evaluation-only",
        "target": {
            "object_uid": str(args.object_uid),
            "object_id": args.object_id,
            "object_name": next(iter(gt.values())).get("object_name"),
            "object_bop_id": next(iter(gt.values())).get("object_bop_id"),
            "cad": str(args.cad),
            "cad_watertight": bool(cad.is_watertight),
            "cad_faces": int(len(cad.faces)),
        },
        "controlled_invariants": [
            "same RGB and official fixed pinhole K",
            "same frozen HaWoR, OWLv2, SAM2, object plan, anchor, raw TRELLIS output, and seed",
            "only P09/P14b depth NPZ changes; P09-P15 depth-dependent state is rebuilt in an evaluator-only branch",
        ],
        "raw_depth_intervention": raw_depth,
        "branches": branches,
        "comparison": comparison_summary(branches),
        "scientific_boundaries": [
            "HOT3D GT camera/object pose and CAD are evaluator-only and do not enter baseline prediction state.",
            "Oracle-depth results test downstream conditional validity, not RGB-only end-to-end accuracy.",
            "Unsigned CAD residuals do not establish contact or signed nonpenetration, especially for non-watertight CADs.",
            "P14/P15 trajectory scoring fits one fixed canonical-frame transform; it tests temporal/camera-relative consistency rather than requiring identical mesh frame conventions.",
        ],
    }
    report_path = args.output_dir / "hot3d_depth_source_ab_report.json"
    write_json(report_path, report)
    p09_review({"unidepth_official_K": review_a, "oracle_depth_official_K": review_b}, args.review_frames, args.output_dir / "hot3d_depth_source_ab_p09_review.jpg")

    branch_pose_inputs = {}
    for name, ann_path, pose_path, completion in (
        ("unidepth_official_K", args.unidepth_annotations, args.unidepth_p15, args.unidepth_completion),
        ("oracle_depth_official_K", args.oracle_annotations, args.oracle_p15, args.oracle_completion),
    ):
        mesh = load_mesh(resolve_pose_mesh(completion))
        branch_pose_inputs[name] = {
            "annotations": annotation_frames(ann_path),
            "poses": load_pose_rows(pose_path),
            "mesh_points": sample_vertices(mesh, 5000),
        }
    pose_review(branch_pose_inputs, gt, cad, args.review_frames, args.output_dir / "hot3d_depth_source_ab_p15_overlay_review.jpg")
    timeline_plot(report, args.output_dir / "hot3d_depth_source_ab_timeline.png")
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "target": report["target"],
        "raw_depth": raw_depth["absolute_depth_error_m"],
        "comparison": report["comparison"],
    }, indent=2))


if __name__ == "__main__":
    main()
