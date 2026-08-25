#!/usr/bin/env python3
"""Evaluate a frozen UniDepth/DA3 pair against HOT3D depth/pose/CAD.

This script is evaluator-only. Released target depth, camera/object poses, and
CAD geometry are consumed only after paired prediction outputs are frozen. It
never writes into a prediction run root and is intentionally excluded from the
prediction runtime bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh

SCHEMA = "v19_hot3d_depth_provider_pair_p09_evaluation_v2"
BRANCHES = ("unidepth_official_K", "da3_nested_official_K_hawor_conditioned")
P09_ROUNDTRIP_MAX_M = 1.0e-9
GT_CAD_CLOSURE_MEDIAN_MAX_M = 5.0e-4
GT_CAD_CLOSURE_P95_MAX_M = 2.0e-3


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, role: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {role}: {path}")
    return path


def asset(path: Path, role: str) -> dict[str, Any]:
    path = require_file(path, role)
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def summarize(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
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


def error_summary(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    result = summarize(arr)
    if arr.size:
        result.update({
            "rmse": float(np.sqrt(np.mean(arr * arr))),
            "fraction_le_0p002m": float(np.mean(arr <= 0.002)),
            "fraction_le_0p005m": float(np.mean(arr <= 0.005)),
            "fraction_le_0p010m": float(np.mean(arr <= 0.010)),
            "fraction_le_0p020m": float(np.mean(arr <= 0.020)),
        })
    return result


def quat_wxyz_to_R(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(value) for value in q]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0 or not math.isfinite(norm):
        raise RuntimeError(f"invalid quaternion {q}")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_from_dict(value: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_R(value["quaternion_wxyz"])
    transform[:3, 3] = np.asarray(value["translation_xyz"], dtype=np.float64)
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def load_gt(sidecar_path: Path, object_uid: str, stream_id: str) -> dict[int, dict[str, Any]]:
    payload = load_json(sidecar_path)
    rows: dict[int, dict[str, Any]] = {}
    for frame in payload.get("frames") or []:
        idx = int(frame["frame_idx"])
        contents = frame.get("json") or {}
        camera = (contents.get("cameras.json") or {}).get(stream_id)
        matches = []
        for entries in (contents.get("objects.json") or {}).values():
            for entry in entries if isinstance(entries, list) else []:
                if str(entry.get("object_uid")) == object_uid:
                    matches.append(entry)
        if camera is None or len(matches) != 1:
            continue
        world_camera = transform_from_dict(camera["T_world_from_camera"])
        world_object = transform_from_dict(matches[0]["T_world_from_object"])
        rows[idx] = {
            "T_camera_object": np.linalg.inv(world_camera) @ world_object,
            "object_name": matches[0].get("object_name"),
            "object_bop_id": str(matches[0].get("object_bop_id")),
            "visibility": (matches[0].get("visibilities_modeled") or {}).get(stream_id),
        }
    if not rows:
        raise RuntimeError(f"no evaluator GT rows for object UID {object_uid}")
    return rows


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(str(path), process=True, merge_primitives=True)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_mesh() if hasattr(loaded, "to_mesh") else trimesh.util.concatenate(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"unsupported CAD payload {type(loaded)}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"empty CAD: {path}")
    return mesh


def validate_p09_pair(
    pair_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], Path, dict[str, Any]]:
    report = load_json(pair_report_path)
    if report.get("status") != "ready_for_depth_dependent_downstream_pair_after_p09":
        raise RuntimeError("paired P09 report is not frozen/ready")
    invariants = report.get("shared_output_invariants") or {}
    if (
        invariants.get("object_owned_masks_byte_identical") is not True
        or invariants.get("camera_hand_shared_frame_state_identical") is not True
        or report.get("sam3d_invoked") is not False
    ):
        raise RuntimeError("paired P09 shared-state invariants are not satisfied")
    branch_paths = {}
    for name in BRANCHES:
        row = (report.get("branches") or {}).get(name)
        if not isinstance(row, dict):
            raise RuntimeError(f"paired P09 report lacks {name}")
        path = require_file(Path(str((row.get("annotations") or {}).get("path"))), f"{name} annotations")
        if sha256_file(path) != (row.get("annotations") or {}).get("sha256"):
            raise RuntimeError(f"{name} annotations changed after paired P09")
        branch_paths[name] = path
    snapshot_path = require_file(
        Path(str(report.get("pair_verification_after") or "")), "post-P09 pair verification snapshot"
    )
    snapshot = load_json(snapshot_path)
    if (
        snapshot.get("status") != "ready_for_frozen_upstream_depth_provider_branches"
        or str((snapshot.get("freeze_contract") or {}).get("contract_id"))
        != str((report.get("freeze_contract") or {}).get("contract_id"))
    ):
        raise RuntimeError("post-P09 pair verification snapshot is invalid")
    invocations = {str(row.get("branch")): row for row in report.get("invocations") or []}
    for name in BRANCHES:
        branch = (snapshot.get("branches") or {}).get(name)
        invocation = invocations.get(name)
        if not isinstance(branch, dict) or not isinstance(invocation, dict):
            raise RuntimeError(f"paired P09 depth binding lacks {name}")
        if (
            str(branch.get("sha256")) != str(invocation.get("depth_archive_sha256"))
            or str(branch.get("provider")) != str(invocation.get("depth_provider"))
        ):
            raise RuntimeError(f"paired P09 invocation/snapshot depth binding differs for {name}")
    return report, branch_paths, snapshot_path, snapshot


def bind_depth_paths_to_pair_snapshot(
    depth_paths: dict[str, Path], snapshot: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    bindings = {}
    for name in BRANCHES:
        path = require_file(depth_paths[name], f"{name} camera-bound depth")
        row = (snapshot.get("branches") or {}).get(name)
        if not isinstance(row, dict):
            raise RuntimeError(f"pair snapshot lacks depth branch {name}")
        expected_path = require_file(Path(str(row.get("path") or "")), f"{name} snapshot depth")
        actual_sha256 = sha256_file(path)
        if path != expected_path or actual_sha256 != str(row.get("sha256")):
            raise RuntimeError(f"explicit evaluator depth is not the frozen paired archive for {name}")
        bindings[name] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": actual_sha256,
            "provider": str(row.get("provider")),
            "depth_array_sha256": str(row.get("depth_array_sha256")),
            "confidence_array_sha256": str(row.get("confidence_array_sha256")),
        }
    return bindings


def load_gt_depth(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"frame_idx", "depth_m", "valid_mask", "K", "image_size_wh", "target_object_uid", "target_object_name"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"GT target depth misses {missing}")
        result = {key: np.asarray(archive[key]) for key in archive.files}
    depth = np.asarray(result["depth_m"], dtype=np.float32)
    mask = np.asarray(result["valid_mask"], dtype=bool)
    frame_idx = np.asarray(result["frame_idx"], dtype=np.int64)
    if depth.shape != mask.shape or depth.ndim != 3 or frame_idx.shape != (depth.shape[0],):
        raise RuntimeError("GT target depth arrays disagree")
    if np.any(depth[~mask] != 0.0) or not np.isfinite(depth).all() or np.any(depth[mask] <= 0.0):
        raise RuntimeError("GT target depth validity contract failed")
    return result


def evaluate_raw_depth_branch(
    path: Path,
    expected_provider: str,
    gt: dict[str, Any],
    max_global_samples: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        frame_idx = np.asarray(archive["frame_idx"], dtype=np.int64)
        prediction = np.asarray(archive["depth"], dtype=np.float32)
        provider = str(np.asarray(archive["depth_provider"]).reshape(-1)[0]) if "depth_provider" in archive.files else "unidepth"
        intrinsics = np.asarray(archive["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        source_size = np.asarray(archive["source_size"], dtype=np.int64)
    gt_frames = np.asarray(gt["frame_idx"], dtype=np.int64)
    gt_depth = np.asarray(gt["depth_m"], dtype=np.float32)
    gt_mask = np.asarray(gt["valid_mask"], dtype=bool)
    gt_K = np.asarray(gt["K"], dtype=np.float64)
    gt_size = np.asarray(gt["image_size_wh"], dtype=np.int64)
    if provider != expected_provider:
        raise RuntimeError(f"raw depth provider mismatch: {provider!r} != {expected_provider!r}")
    if frame_idx.tolist() != gt_frames.tolist() or prediction.shape != gt_depth.shape:
        raise RuntimeError(f"{provider} raw depth timeline/raster differs from GT")
    if source_size.tolist() != gt_size.tolist() or not np.allclose(
        intrinsics, np.repeat(np.asarray([[gt_K[0, 0], gt_K[1, 1], gt_K[0, 2], gt_K[1, 2]]]), len(frame_idx), axis=0),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError(f"{provider} raw depth rays differ from evaluator GT camera")
    rows = []
    sampled_signed: list[np.ndarray] = []
    sampled_abs: list[np.ndarray] = []
    sampled_ratio: list[np.ndarray] = []
    sampled_aligned_abs: list[np.ndarray] = []
    valid_pixel_total = 0
    target_pixel_total = int(np.count_nonzero(gt_mask))
    per_frame_budget = max(1, max_global_samples // max(1, len(frame_idx)))
    for position, idx in enumerate(frame_idx.tolist()):
        valid = gt_mask[position] & np.isfinite(prediction[position]) & (prediction[position] > 0.0)
        valid_count = int(np.count_nonzero(valid))
        gt_count = int(np.count_nonzero(gt_mask[position]))
        valid_pixel_total += valid_count
        if valid_count == 0:
            rows.append({"frame_idx": idx, "gt_target_pixels": gt_count, "valid_prediction_pixels": 0})
            continue
        pred = prediction[position][valid]
        truth = gt_depth[position][valid]
        signed = pred - truth
        absolute = np.abs(signed)
        ratio = pred / truth
        scale_to_gt = float(np.median(truth / np.maximum(pred, 1.0e-8)))
        aligned_abs = np.abs(pred * scale_to_gt - truth)
        step = max(1, int(math.ceil(valid_count / per_frame_budget)))
        sampled_signed.append(signed[::step])
        sampled_abs.append(absolute[::step])
        sampled_ratio.append(ratio[::step])
        sampled_aligned_abs.append(aligned_abs[::step])
        rows.append({
            "frame_idx": idx,
            "gt_target_pixels": gt_count,
            "valid_prediction_pixels": valid_count,
            "prediction_coverage": float(valid_count / max(1, gt_count)),
            "signed_error_m": summarize(signed),
            "absolute_error_m": error_summary(absolute),
            "prediction_over_gt_ratio": summarize(ratio),
            "best_per_frame_scalar_to_gt": scale_to_gt,
            "best_scalar_aligned_absolute_error_m": error_summary(aligned_abs),
        })
    concat = lambda values: np.concatenate(values) if values else np.asarray([], dtype=np.float32)
    return {
        "depth_archive": asset(path, f"{expected_provider} camera-bound depth"),
        "provider": provider,
        "frame_count": len(frame_idx),
        "target_visible_pixel_count": target_pixel_total,
        "valid_prediction_pixel_count": valid_pixel_total,
        "prediction_coverage": float(valid_pixel_total / max(1, target_pixel_total)),
        "global_summary_sampling": f"deterministic per-frame stride, <= approximately {max_global_samples} samples",
        "signed_error_m": summarize(concat(sampled_signed)),
        "absolute_error_m": error_summary(concat(sampled_abs)),
        "prediction_over_gt_ratio": summarize(concat(sampled_ratio)),
        "best_per_frame_scalar_aligned_absolute_error_m": error_summary(concat(sampled_aligned_abs)),
        "per_frame_best_scalar_to_gt": summarize([row["best_per_frame_scalar_to_gt"] for row in rows if "best_per_frame_scalar_to_gt" in row]),
        "rows": rows,
    }


def annotation_camera(frame: dict[str, Any]) -> np.ndarray:
    camera = frame.get("camera") or {}
    value = camera.get("T_world_camera_metric") or camera.get("T_world_camera")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise RuntimeError("P09 annotation has invalid prediction camera")
    return result


def load_p09_points(
    path: Path, object_id: str, expected_provider: str
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    rows = {}
    roundtrip_errors = []
    roundtrip_point_count = 0
    for frame in payload.get("frames") or []:
        idx = int(frame["frame_idx"])
        objects = [row for row in frame.get("objects") or [] if str(row.get("object_id")) == object_id]
        if len(objects) != 1:
            raise RuntimeError(f"P09 frame {idx} target object count is {len(objects)}")
        candidate = objects[0].get("visible_geometry_candidate")
        if not isinstance(candidate, dict) or not candidate.get("world_vertices_sample_m"):
            continue
        provider = str(candidate.get("depth_provider"))
        if provider != expected_provider:
            raise RuntimeError(f"P09 frame {idx} provider mismatch: {provider!r} != {expected_provider!r}")
        points_world = np.asarray(candidate["world_vertices_sample_m"], dtype=np.float64)
        points_camera = np.asarray(candidate.get("camera_vertices_sample_m"), dtype=np.float64)
        transform = annotation_camera(frame)
        if points_world.shape != points_camera.shape or points_world.ndim != 2 or points_world.shape[1] != 3:
            raise RuntimeError(f"P09 frame {idx} camera/world point samples disagree")
        recovered_camera = transform_points(np.linalg.inv(transform), points_world)
        error = np.linalg.norm(recovered_camera - points_camera, axis=1)
        if not np.isfinite(error).all() or (error.size and float(np.max(error)) > P09_ROUNDTRIP_MAX_M):
            raise RuntimeError(f"P09 frame {idx} camera/world point round-trip failed")
        roundtrip_errors.append(error)
        roundtrip_point_count += int(error.size)
        rows[idx] = {
            "points_world": points_world,
            "T_world_camera": transform,
        }
    combined = np.concatenate(roundtrip_errors) if roundtrip_errors else np.asarray([], dtype=np.float64)
    closure = {
        "status": "passed",
        "frame_count": len(rows),
        "point_count": roundtrip_point_count,
        "maximum_allowed_error_m": P09_ROUNDTRIP_MAX_M,
        "error_m": summarize(combined),
        "contract": "inverse(T_world_camera) applied to stored world samples reproduces stored camera samples",
    }
    return rows, closure


def gt_depth_cad_closure(
    gt_depth: dict[str, Any],
    gt: dict[int, dict[str, Any]],
    query: trimesh.proximity.ProximityQuery,
    samples_per_frame: int,
) -> dict[str, Any]:
    depth = np.asarray(gt_depth["depth_m"], dtype=np.float32)
    valid_mask = np.asarray(gt_depth["valid_mask"], dtype=bool)
    intrinsics = np.asarray(gt_depth["K"], dtype=np.float64)
    frame_idx = np.asarray(gt_depth["frame_idx"], dtype=np.int64)
    distances = []
    rows = []
    count = max(1, int(samples_per_frame))
    for position, idx in enumerate(frame_idx.tolist()):
        ys, xs = np.nonzero(valid_mask[position])
        if len(xs) == 0:
            raise RuntimeError(f"GT target depth frame {idx} has no visible pixels")
        selected = np.linspace(0, len(xs) - 1, min(count, len(xs)), dtype=int)
        xs = xs[selected]
        ys = ys[selected]
        z = depth[position, ys, xs].astype(np.float64)
        camera = np.column_stack((
            (xs - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (ys - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ))
        object_points = transform_points(np.linalg.inv(gt[idx]["T_camera_object"]), camera)
        _, distance, _ = query.on_surface(object_points)
        distance = np.asarray(distance, dtype=np.float64)
        if not np.isfinite(distance).all():
            raise RuntimeError(f"GT depth/CAD closure frame {idx} returned non-finite distance")
        distances.append(distance)
        rows.append({
            "frame_idx": idx,
            "sample_count": int(distance.size),
            "unsigned_distance_to_cad_m": summarize(distance),
        })
    combined = np.concatenate(distances)
    summary = summarize(combined)
    if (
        float(summary["median"]) > GT_CAD_CLOSURE_MEDIAN_MAX_M
        or float(summary["p95"]) > GT_CAD_CLOSURE_P95_MAX_M
    ):
        raise RuntimeError(f"GT rendered-depth/pose/CAD closure failed: {summary}")
    return {
        "status": "passed",
        "contract": "rendered camera-Z depth backprojects through official K and GT camera/object pose to released CAD",
        "sample_selection": "deterministic linear index sampling within each target-visible mask",
        "samples_per_frame_limit": count,
        "frame_count": len(rows),
        "sample_count": int(combined.size),
        "median_maximum_allowed_m": GT_CAD_CLOSURE_MEDIAN_MAX_M,
        "p95_maximum_allowed_m": GT_CAD_CLOSURE_P95_MAX_M,
        "unsigned_distance_to_cad_m": summary,
        "rows": rows,
    }


def evaluate_da3_windows(
    qc_path: Path, bound_depth_path: Path, raw_depth: dict[str, Any]
) -> dict[str, Any]:
    qc = load_json(qc_path)
    windowing = qc.get("windowing") or {}
    pose = qc.get("pose_conditioning") or {}
    if (
        qc.get("status") != "ok"
        or pose.get("metric_scale_mode") != "nested_metric_branch"
        or pose.get("align_to_input_ext_scale") is not False
        or (windowing.get("overlap_consistency") or {}).get("passed") is not True
    ):
        raise RuntimeError("DA3 QC is not a passed Nested metric-branch run")
    qc_depth_path = require_file(Path(str((qc.get("outputs") or {}).get("depth_archive") or "")), "DA3 QC raw depth")
    qc_depth_sha256 = sha256_file(qc_depth_path)
    with np.load(bound_depth_path, allow_pickle=False) as archive:
        if "source_depth_archive_sha256" not in archive.files:
            raise RuntimeError("camera-bound DA3 depth lacks source archive binding")
        bound_source_sha256 = str(np.asarray(archive["source_depth_archive_sha256"]).reshape(-1)[0])
        bound_scale_mode = str(np.asarray(archive["metric_scale_mode"]).reshape(-1)[0])
        bound_overlap_passed = bool(np.asarray(archive["overlap_consistency_passed"]).reshape(-1)[0])
    if (
        qc_depth_sha256 != bound_source_sha256
        or bound_scale_mode != "nested_metric_branch"
        or not bound_overlap_passed
    ):
        raise RuntimeError("DA3 QC/raw depth does not bind the paired camera-bound DA3 archive")
    rows_by_frame = {int(row["frame_idx"]): row for row in raw_depth["rows"]}
    windows = []
    for window in windowing.get("windows") or []:
        frame_ids = list(range(int(window["frame_start"]), int(window["frame_end"]) + 1))
        if not frame_ids or any(idx not in rows_by_frame for idx in frame_ids):
            raise RuntimeError("DA3 QC window timeline differs from evaluator raw-depth timeline")
        selected = [rows_by_frame[idx] for idx in frame_ids]
        windows.append({
            "window_index": int(window["window_index"]),
            "frame_start": frame_ids[0],
            "frame_end": frame_ids[-1],
            "frame_count": len(frame_ids),
            "conditioned_camera_center_max_baseline_m": float(window["conditioned_camera_center_max_baseline_m"]),
            "model_prediction_scale_factor_diagnostic_not_applied": float(window["prediction_scale_factor_diagnostic"]),
            "evaluator_only_best_per_frame_scalar_to_gt": summarize(
                [row["best_per_frame_scalar_to_gt"] for row in selected]
            ),
            "prediction_over_gt_ratio_frame_medians": summarize(
                [row["prediction_over_gt_ratio"]["median"] for row in selected]
            ),
            "raw_absolute_error_frame_medians_m": summarize(
                [row["absolute_error_m"]["median"] for row in selected]
            ),
            "best_scalar_aligned_absolute_error_frame_medians_m": summarize(
                [row["best_scalar_aligned_absolute_error_m"]["median"] for row in selected]
            ),
        })
    if len(windows) != len(windowing.get("windows") or []):
        raise RuntimeError("DA3 window diagnosis is incomplete")
    return {
        "qc": asset(qc_path, "DA3 Nested metric-branch QC"),
        "qc_raw_depth": {
            "path": str(qc_depth_path),
            "bytes": int(qc_depth_path.stat().st_size),
            "sha256": qc_depth_sha256,
            "bound_by_camera_archive_source_depth_sha256": True,
        },
        "metric_scale_mode": pose["metric_scale_mode"],
        "input_trajectory_scale_alignment_applied": bool(pose["align_to_input_ext_scale"]),
        "overlap_consistency": windowing["overlap_consistency"],
        "overlap_relative_depth_median": windowing.get("overlap_relative_depth_median"),
        "overlap_scale_aligned_relative_depth_median": windowing.get("overlap_scale_aligned_relative_depth_median"),
        "window_count": len(windows),
        "windows": windows,
        "interpretation_boundary": "overlap consistency compares duplicate predictions, not absolute metric correctness against GT",
    }


def evaluate_p09_frames(
    rows: dict[int, dict[str, Any]],
    gt: dict[int, dict[str, Any]],
    query: trimesh.proximity.ProximityQuery,
    frame_filter: set[int] | None,
    max_points: int,
) -> dict[str, Any]:
    per_frame = []
    samples = []
    candidate_frames = sorted(set(rows) & set(gt))
    if frame_filter is not None:
        candidate_frames = [idx for idx in candidate_frames if idx in frame_filter]
    for idx in candidate_frames:
        points_world = rows[idx]["points_world"]
        if len(points_world) > max_points:
            points_world = points_world[np.linspace(0, len(points_world) - 1, max_points, dtype=int)]
        points_camera = transform_points(np.linalg.inv(rows[idx]["T_world_camera"]), points_world)
        points_object = transform_points(np.linalg.inv(gt[idx]["T_camera_object"]), points_camera)
        _, distance, _ = query.on_surface(points_object)
        distance = np.asarray(distance, dtype=np.float64)
        distance = distance[np.isfinite(distance)]
        if distance.size == 0:
            continue
        samples.append(distance)
        per_frame.append({
            "frame_idx": idx,
            "point_count": int(distance.size),
            "surfel_to_gt_cad_unsigned_m": error_summary(distance),
            "gt_visibility": gt[idx].get("visibility"),
        })
    combined = np.concatenate(samples) if samples else np.asarray([], dtype=np.float64)
    return {
        "frame_count": len(per_frame),
        "frame_ids": [row["frame_idx"] for row in per_frame],
        "sample_count": int(combined.size),
        "surfel_to_gt_cad_unsigned_m": error_summary(combined),
        "per_frame_median_m": summarize([
            row["surfel_to_gt_cad_unsigned_m"]["median"] for row in per_frame
        ]),
        "rows": per_frame,
    }


def comparison(branches: dict[str, dict[str, Any]], section: str, metric: str) -> dict[str, Any]:
    left = branches[BRANCHES[0]][section]
    right = branches[BRANCHES[1]][section]
    a = left
    b = right
    for part in metric.split("."):
        a = a[part]
        b = b[part]
    return {
        BRANCHES[0]: float(a),
        BRANCHES[1]: float(b),
        "da3_minus_unidepth": float(b - a),
        "lower_is_better": True,
        "da3_better": bool(b < a),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    p09_pair_path = require_file(args.p09_pair_report, "frozen paired P09 report")
    pair_report, annotations, pair_snapshot_path, pair_snapshot = validate_p09_pair(p09_pair_path)
    sidecar_path = require_file(args.hot3d_sidecar, "HOT3D evaluator sidecar")
    gt_depth_path = require_file(args.hot3d_target_depth, "HOT3D target-visible depth")
    cad_path = require_file(args.cad, "HOT3D target CAD")
    da3_qc_path = require_file(args.da3_qc, "DA3 Nested metric-branch QC")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"evaluator output must be fresh: {output_dir}")
    output_dir.mkdir(parents=True)
    gt_depth = load_gt_depth(gt_depth_path)
    object_uid = str(np.asarray(gt_depth["target_object_uid"]).reshape(-1)[0])
    object_name = str(np.asarray(gt_depth["target_object_name"]).reshape(-1)[0])
    if object_uid != str(args.object_uid) or object_name != str(pair_report["object_id"]):
        raise RuntimeError("GT depth target identity differs from paired P09 target")
    gt = load_gt(sidecar_path, object_uid, str(args.stream_id))
    if sorted(gt) != np.asarray(gt_depth["frame_idx"], dtype=np.int64).tolist():
        raise RuntimeError("GT sidecar timeline differs from GT target depth")
    mesh = load_mesh(cad_path)
    query = trimesh.proximity.ProximityQuery(mesh)
    gt_closure = gt_depth_cad_closure(
        gt_depth, gt, query, int(args.gt_cad_closure_samples_per_frame)
    )
    expected = {
        BRANCHES[0]: "unidepth",
        BRANCHES[1]: "depth_anything_3",
    }
    depth_paths = {
        BRANCHES[0]: require_file(args.unidepth_depth, "UniDepth camera-bound depth"),
        BRANCHES[1]: require_file(args.da3_depth, "DA3 camera-bound depth"),
    }
    depth_bindings = bind_depth_paths_to_pair_snapshot(depth_paths, pair_snapshot)
    branches: dict[str, dict[str, Any]] = {}
    p09_rows = {}
    for name in BRANCHES:
        raw = evaluate_raw_depth_branch(depth_paths[name], expected[name], gt_depth, int(args.max_global_depth_samples))
        rows, p09_closure = load_p09_points(
            annotations[name], str(pair_report["object_id"]), expected[name]
        )
        p09_rows[name] = rows
        branches[name] = {
            "provider": expected[name],
            "frozen_depth_binding": depth_bindings[name],
            "p09_world_camera_roundtrip": p09_closure,
            "raw_depth": raw,
            "p09_all_eligible_frames": evaluate_p09_frames(rows, gt, query, None, int(args.max_p09_points_per_frame)),
        }
    common_frames = set(p09_rows[BRANCHES[0]]) & set(p09_rows[BRANCHES[1]]) & set(gt)
    for name in BRANCHES:
        branches[name]["p09_common_frames"] = evaluate_p09_frames(
            p09_rows[name], gt, query, common_frames, int(args.max_p09_points_per_frame)
        )
    da3_window_diagnosis = evaluate_da3_windows(
        da3_qc_path, depth_paths[BRANCHES[1]], branches[BRANCHES[1]]["raw_depth"]
    )
    report = {
        "schema": SCHEMA,
        "status": "ok_evaluator_only_frozen_prediction_pair",
        "claim_scope": "evaluator-only raw depth and P09 surfel-to-CAD diagnosis; no GT asset enters prediction state",
        "prediction_inputs": {
            "p09_pair_report": asset(p09_pair_path, "frozen paired P09 report"),
            "post_p09_pair_verification": asset(pair_snapshot_path, "post-P09 pair verification snapshot"),
            "freeze_contract_id": pair_report["freeze_contract"]["contract_id"],
            "anchor_frame": int(pair_report["anchor_frame"]),
            "shared_output_invariants": pair_report["shared_output_invariants"],
            "depth_bindings": depth_bindings,
            "da3_qc": asset(da3_qc_path, "DA3 Nested metric-branch QC"),
        },
        "evaluation_inputs": {
            "hot3d_sidecar": asset(sidecar_path, "released HOT3D evaluator sidecar"),
            "hot3d_target_visible_depth": asset(gt_depth_path, "rendered GT target first-surface depth"),
            "cad": {**asset(cad_path, "released target CAD"), "vertices": len(mesh.vertices), "faces": len(mesh.faces), "watertight": bool(mesh.is_watertight)},
            "object_uid": object_uid,
            "object_name": object_name,
            "object_bop_id": next(iter(gt.values()))["object_bop_id"],
            "stream_id": str(args.stream_id),
        },
        "evaluator_geometry_contract_closure": {
            "gt_rendered_depth_pose_cad": gt_closure,
            "p09_world_camera_roundtrip": {
                name: branches[name]["p09_world_camera_roundtrip"] for name in BRANCHES
            },
        },
        "controlled_invariants": [
            "prediction pair frozen before evaluation",
            "same 150 RGB, official K, HaWoR, OWLv2, SAM2, base annotations, object-owned masks, camera/hand state, anchor, P09 script, parameters, and seed",
            "only external metric-depth provider and monotonic provider-specific confidence ranking differ",
            "GT depth/poses/CAD are evaluator-only and are not copied into prediction roots",
        ],
        "common_p09_frame_count": len(common_frames),
        "common_p09_frame_ids": sorted(common_frames),
        "branches": branches,
        "da3_window_scale_diagnosis": da3_window_diagnosis,
        "comparison": {
            "raw_depth_absolute_error_median_m": comparison(branches, "raw_depth", "absolute_error_m.median"),
            "raw_depth_absolute_error_rmse_m": comparison(branches, "raw_depth", "absolute_error_m.rmse"),
            "raw_depth_best_scalar_aligned_error_median_m": comparison(branches, "raw_depth", "best_per_frame_scalar_aligned_absolute_error_m.median"),
            "p09_common_surfel_to_cad_median_m": comparison(branches, "p09_common_frames", "surfel_to_gt_cad_unsigned_m.median"),
            "p09_common_surfel_to_cad_rmse_m": comparison(branches, "p09_common_frames", "surfel_to_gt_cad_unsigned_m.rmse"),
        },
        "scientific_boundaries": [
            "This technical pair reuses historical local29 shared prediction-side upstream and is not a formal fresh end-to-end A/B.",
            "Rendered GT depth covers the visible target CAD first surface only; background/unmodeled pixels are excluded.",
            "Unsigned surfel-to-CAD distance does not establish contact, occupancy, or signed nonpenetration.",
            "Best-scalar-aligned raw-depth residual is diagnostic only and must not rescale prediction outputs.",
            "Survivor-only P09 metrics can be biased by fail-closed coverage; common-frame and coverage results must be interpreted together.",
            "DA3 overlap consistency compares duplicate window predictions and is not an absolute metric-scale correctness guarantee.",
        ],
    }
    report_path = output_dir / "hot3d_depth_provider_pair_p09_evaluation.json"
    write_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "common_p09_frame_count": len(common_frames),
        "comparison": report["comparison"],
        "raw_depth_coverage": {name: branches[name]["raw_depth"]["prediction_coverage"] for name in BRANCHES},
        "p09_frame_count": {name: branches[name]["p09_all_eligible_frames"]["frame_count"] for name in BRANCHES},
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p09-pair-report", type=Path, required=True)
    parser.add_argument("--unidepth-depth", type=Path, required=True)
    parser.add_argument("--da3-depth", type=Path, required=True)
    parser.add_argument("--da3-qc", type=Path, required=True)
    parser.add_argument("--hot3d-sidecar", type=Path, required=True)
    parser.add_argument("--hot3d-target-depth", type=Path, required=True)
    parser.add_argument("--cad", type=Path, required=True)
    parser.add_argument("--object-uid", required=True)
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument("--max-global-depth-samples", type=int, default=1_000_000)
    parser.add_argument("--max-p09-points-per-frame", type=int, default=1500)
    parser.add_argument("--gt-cad-closure-samples-per-frame", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
