#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from compare_hand_streams_scale055_v3 import load_depth_archive, load_frame_window
from diagnose_contact_depth_conflict_v3 import summarize
from diagnose_hand_contact_reliability_v3 import (
    depth_patch_iqr_ratio,
    hand_bone_scale_m,
    hand_tip_spread_m,
    resize_mask_to_depth,
    summarize_key,
)
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, resize_bool_mask


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mesh_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(float)
    faces = blob["faces"].astype(np.int32)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, idx in enumerate(frame_idx.tolist()):
        if int(idx) in out:
            raise RuntimeError(f"mesh archive has duplicate frame {idx}")
        v0, v1 = int(vertex_offsets[i]), int(vertex_offsets[i + 1])
        f0, f1 = int(face_offsets[i]), int(face_offsets[i + 1])
        frame_vertices = vertices[v0:v1]
        frame_faces = faces[f0:f1]
        if len(frame_vertices) == 0 or len(frame_faces) == 0:
            raise RuntimeError(f"mesh archive frame {idx} is empty")
        if frame_faces.min() < 0 or frame_faces.max() >= len(frame_vertices):
            raise RuntimeError(f"mesh archive frame {idx} has invalid face indices")
        out[int(idx)] = (frame_vertices, frame_faces)
    return out


def camera_points(world_points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[world_points, np.ones(len(world_points), dtype=float)]
    return (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]


def source_size_for(frame: dict) -> np.ndarray:
    obj = frame.get("object", {})
    size = np.asarray(obj.get("source_image_size", []), dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} has invalid object source image size")
    return size


def object_mask_for(frame: dict, depth: np.ndarray, remote_output_root: Path | None, local_output_root: Path | None) -> np.ndarray:
    obj = frame.get("object", {})
    if not obj.get("mask_path"):
        raise RuntimeError(f"frame {frame.get('frame_idx')} has no object mask path")
    if not obj.get("mask_image_size"):
        raise RuntimeError(f"frame {frame.get('frame_idx')} object has no mask_image_size")
    mask_path = localize_path(str(obj["mask_path"]), remote_output_root, local_output_root)
    mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
    return resize_mask_to_depth(mask, depth)


def intrinsics_for(frame: dict, hand: dict, source: str, cli_intrinsics: list[float]) -> np.ndarray:
    if source == "annotation-vggt":
        intr = np.asarray(frame.get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", []), dtype=float)
    elif source == "hand":
        intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    elif source == "cli":
        intr = np.asarray(cli_intrinsics, dtype=float)
    else:
        raise RuntimeError(f"unsupported intrinsics source {source}")
    if intr.shape != (4,) or not np.isfinite(intr).all():
        raise RuntimeError(f"invalid {source} intrinsics for frame {frame.get('frame_idx')}")
    return intr


def hand_camera_joints(hand: dict) -> np.ndarray:
    joints = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
    if joints.shape != (21, 3):
        joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
    if joints.shape != (21, 3):
        raise RuntimeError("hand has no usable 21x3 camera joints")
    return joints


def hand_camera_vertices(hand: dict, T_world_camera: np.ndarray) -> np.ndarray:
    for key in ("vertices_source_camera_m", "vertices_source_camera_m_sample"):
        if key in hand:
            arr = np.asarray(hand[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
    for key in ("vertices_world_m", "vertices_world_m_sample"):
        if key in hand:
            arr = np.asarray(hand[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return camera_points(arr, T_world_camera)
    raise RuntimeError("hand has no usable MANO vertices")


def hand_local_vertices(hand: dict) -> np.ndarray:
    for key in ("vertices_camera", "vertices_camera_sample"):
        if key in hand:
            arr = np.asarray(hand[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
    vertices = hand_camera_vertices(hand, np.eye(4, dtype=float))
    cam_t = np.asarray(hand.get("cam_t", []), dtype=float)
    if cam_t.shape == (3,):
        return vertices - cam_t[None, :]
    return vertices


def hand_keypoints(hand: dict) -> np.ndarray:
    keypoints = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    if keypoints.shape != (21, 2):
        keypoints = np.asarray(hand.get("joints2d", []), dtype=float)
    if keypoints.shape != (21, 2):
        raise RuntimeError("hand has no usable 21x2 image keypoints")
    return keypoints


def oriented_vertex_normals(vertices_camera: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices_camera[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-9
    face_normals[valid] /= lengths[valid, None]
    face_normals[~valid] = 0.0
    normals = np.zeros_like(vertices_camera, dtype=float)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-9
    if int(np.count_nonzero(good)) < 10:
        raise RuntimeError("object mesh has too few vertices with face-derived normals")
    normals[good] /= lengths[good, None]
    toward_camera = np.einsum("ij,ij->i", normals, vertices_camera) < 0.0
    normals[good & ~toward_camera] *= -1.0
    return normals, good


def patch_summary(
    hand_vertices: np.ndarray,
    local_vertices: np.ndarray,
    object_vertices: np.ndarray,
    object_normals: np.ndarray,
    candidate_idx: np.ndarray,
    patch_size: int,
    tree: cKDTree,
) -> dict | None:
    if candidate_idx.size < patch_size:
        return None
    distances, nearest = tree.query(hand_vertices[candidate_idx], k=1)
    order = np.argsort(distances)[:patch_size]
    hand_idx = candidate_idx[order]
    nearest_idx = nearest[order]
    delta = hand_vertices[hand_idx] - object_vertices[nearest_idx]
    signed = np.einsum("ij,ij->i", delta, object_normals[nearest_idx])
    selected = hand_vertices[hand_idx]
    local = local_vertices[hand_idx] if len(local_vertices) == len(hand_vertices) else selected
    if len(selected) <= 1:
        spread = 0.0
        local_spread = 0.0
    else:
        spread = float(np.max(np.linalg.norm(selected[:, None, :] - selected[None, :, :], axis=2)))
        local_spread = float(np.max(np.linalg.norm(local[:, None, :] - local[None, :, :], axis=2)))
    return {
        "patch_vertices": int(patch_size),
        "patch_vertex_ids": [int(v) for v in hand_idx.tolist()],
        "patch_distance_median_m": float(np.median(distances[order])),
        "patch_distance_p95_m": float(np.percentile(distances[order], 95.0)),
        "patch_signed_gap_median_m": float(np.median(signed)),
        "patch_signed_gap_p95_abs_m": float(np.percentile(np.abs(signed), 95.0)),
        "patch_signed_gap_min_m": float(np.min(signed)),
        "patch_signed_gap_max_m": float(np.max(signed)),
        "patch_penetration_fraction_010m": float(np.mean(signed < -0.010)),
        "patch_front_separation_fraction_030m": float(np.mean(signed > 0.030)),
        "patch_spread_m": spread,
        "patch_local_spread_m": local_spread,
        "patch_local_center_m": [float(v) for v in np.mean(local, axis=0).tolist()],
    }


def best_patch(
    hand_vertices: np.ndarray,
    local_vertices: np.ndarray,
    object_vertices: np.ndarray,
    object_normals: np.ndarray,
    candidate_idx: np.ndarray,
    patch_sizes: list[int],
    tree: cKDTree,
) -> dict:
    reports = [
        report
        for size in patch_sizes
        if (report := patch_summary(hand_vertices, local_vertices, object_vertices, object_normals, candidate_idx, int(size), tree))
        is not None
    ]
    if not reports:
        return {
            "patch_reports": [],
            "best_patch_vertices": 0,
            "best_patch_vertex_ids": [],
            "best_patch_distance_median_m": None,
            "best_patch_distance_p95_m": None,
            "best_patch_signed_gap_median_m": None,
            "best_patch_signed_gap_p95_abs_m": None,
            "best_patch_penetration_fraction_010m": 0.0,
            "best_patch_front_separation_fraction_030m": 0.0,
            "best_patch_spread_m": None,
            "best_patch_local_spread_m": None,
            "best_patch_local_center_m": None,
        }

    def key(report: dict) -> tuple[float, float, float]:
        return (
            float(report["patch_distance_p95_m"]),
            abs(float(report["patch_signed_gap_median_m"])),
            float(report["patch_spread_m"]),
        )

    best = min(reports, key=key)
    return {
        "patch_reports": reports,
        "best_patch_vertices": int(best["patch_vertices"]),
        "best_patch_vertex_ids": best["patch_vertex_ids"],
        "best_patch_distance_median_m": best["patch_distance_median_m"],
        "best_patch_distance_p95_m": best["patch_distance_p95_m"],
        "best_patch_signed_gap_median_m": best["patch_signed_gap_median_m"],
        "best_patch_signed_gap_p95_abs_m": best["patch_signed_gap_p95_abs_m"],
        "best_patch_penetration_fraction_010m": best["patch_penetration_fraction_010m"],
        "best_patch_front_separation_fraction_030m": best["patch_front_separation_fraction_030m"],
        "best_patch_spread_m": best["patch_spread_m"],
        "best_patch_local_spread_m": best["patch_local_spread_m"],
        "best_patch_local_center_m": best["patch_local_center_m"],
    }


def row_for_hand(
    frame: dict,
    hand_i: int,
    hand: dict,
    depth: np.ndarray,
    mask_depth: np.ndarray,
    object_vertices_camera: np.ndarray,
    object_normals_camera: np.ndarray,
    tree: cKDTree,
    args: argparse.Namespace,
) -> dict:
    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    source_size = source_size_for(frame)
    joints = hand_camera_joints(hand)
    vertices = hand_camera_vertices(hand, T_world_camera)
    local_vertices = hand_local_vertices(hand)
    keypoints = hand_keypoints(hand)
    intr = intrinsics_for(frame, hand, args.intrinsics_source, args.intrinsics)
    if np.any(vertices[:, 2] <= 0.0) or np.any(joints[:, 2] <= 0.0):
        raise RuntimeError("hand geometry contains non-positive camera depth")
    projected_joints = project_points(joints, intr)
    reproj = np.linalg.norm(projected_joints - keypoints, axis=1)
    metric_depth = sample_depth(depth, keypoints, source_size)
    valid_depth = np.isfinite(metric_depth) & (metric_depth > 0.0)
    good_depth = valid_depth & (reproj <= float(args.good_joint_reprojection_px))
    mano_minus_metric = joints[good_depth, 2] - metric_depth[good_depth]
    depth_scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
    depth_xy = keypoints * depth_scale[None, :]
    patch_ratios = np.asarray(
        [depth_patch_iqr_ratio(depth, xy, int(args.patch_radius)) for xy in depth_xy[good_depth]],
        dtype=float,
    )
    stable_depth = patch_ratios[np.isfinite(patch_ratios)] <= float(args.max_depth_iqr_ratio)
    dist = mask_distance_map(mask_depth)
    uv = project_points(vertices, intr)
    xy = uv * depth_scale[None, :]
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    candidate_idx = np.flatnonzero(valid & (dist[y, x] <= float(args.mask_contact_distance_px)))
    patch = best_patch(
        vertices,
        local_vertices,
        object_vertices_camera,
        object_normals_camera,
        candidate_idx,
        [int(value) for value in args.patch_sizes],
        tree,
    )
    score = float(hand.get("detector_score", np.nan))
    measured = bool(hand.get("measurement_available", False))
    detector_ok = bool(np.isfinite(score) and score >= float(args.min_detector_score))
    projection_ok = bool(float(np.median(reproj)) <= float(args.max_good_median_reprojection_px))
    depth_ok = bool(
        len(mano_minus_metric) >= int(args.min_good_depth_joints)
        and abs(float(np.median(mano_minus_metric))) <= float(args.max_good_depth_bias_m)
    )
    stable_depth_ok = bool(
        len(stable_depth) >= int(args.min_good_depth_joints)
        and float(np.mean(stable_depth)) >= float(args.min_stable_depth_fraction)
    )
    bone_scale = hand_bone_scale_m(joints)
    bone_scale_ok = bool(float(args.min_bone_scale_m) <= bone_scale <= float(args.max_bone_scale_m))
    patch_available = int(patch["best_patch_vertices"]) >= int(args.min_patch_vertices)
    patch_distance_ok = bool(
        patch_available
        and patch["best_patch_distance_p95_m"] is not None
        and float(patch["best_patch_distance_p95_m"]) <= float(args.accept_patch_distance_p95_m)
    )
    patch_signed_ok = bool(
        patch_available
        and patch["best_patch_signed_gap_median_m"] is not None
        and abs(float(patch["best_patch_signed_gap_median_m"])) <= float(args.accept_patch_signed_gap_m)
        and float(patch["best_patch_signed_gap_p95_abs_m"]) <= float(args.accept_patch_signed_gap_p95_m)
    )
    patch_spread_ok = bool(
        patch_available
        and patch["best_patch_spread_m"] is not None
        and float(patch["best_patch_spread_m"]) <= float(args.accept_patch_spread_m)
    )
    penetration_ok = bool(
        float(patch["best_patch_penetration_fraction_010m"]) <= float(args.accept_patch_penetration_fraction)
    )
    contact_geometry_ok = bool(patch_distance_ok and patch_signed_ok and patch_spread_ok and penetration_ok)
    reliable = bool(
        measured
        and detector_ok
        and projection_ok
        and depth_ok
        and stable_depth_ok
        and bone_scale_ok
        and contact_geometry_ok
    )
    return {
        "frame_idx": int(frame["frame_idx"]),
        "hand_idx": int(hand_i),
        "side": str(hand.get("side", "unknown")),
        "track_id": hand.get("track_id"),
        "track_source": hand.get("track_source"),
        "filter_status": hand.get("filter_status"),
        "measurement_available": measured,
        "detector_score": score,
        "median_joint_reprojection_px": float(np.median(reproj)),
        "p95_joint_reprojection_px": float(np.percentile(reproj, 95.0)),
        "good_depth_joints": int(np.count_nonzero(good_depth)),
        "mano_minus_metric_depth_median_m": None if len(mano_minus_metric) == 0 else float(np.median(mano_minus_metric)),
        "mano_minus_metric_depth_p95_abs_m": None if len(mano_minus_metric) == 0 else float(np.percentile(np.abs(mano_minus_metric), 95.0)),
        "stable_depth_fraction": None if len(stable_depth) == 0 else float(np.mean(stable_depth)),
        "hand_bone_scale_m": float(bone_scale),
        "hand_tip_spread_m": float(hand_tip_spread_m(joints)),
        "mask_candidate_vertices": int(candidate_idx.size),
        **patch,
        "detector_ok": detector_ok,
        "projection_ok": projection_ok,
        "depth_ok": depth_ok,
        "stable_depth_ok": stable_depth_ok,
        "bone_scale_ok": bone_scale_ok,
        "patch_distance_ok": patch_distance_ok,
        "patch_signed_ok": patch_signed_ok,
        "patch_spread_ok": patch_spread_ok,
        "patch_penetration_ok": penetration_ok,
        "contact_geometry_ok": contact_geometry_ok,
        "patch_temporal_support_frames": 0,
        "patch_temporal_support_span_frames": 0,
        "patch_temporal_local_drift_m": None,
        "patch_temporal_local_drift_ok": False,
        "patch_temporal_support_ok": False,
        "reliable_geometry_contact": reliable,
        "reliable_for_contact": False,
    }


def point_extent(points: np.ndarray) -> float | None:
    if points.size == 0:
        return None
    if len(points) == 1:
        return 0.0
    return float(np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)))


def annotate_temporal_support(rows: list[dict], args: argparse.Namespace) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if bool(row.get("reliable_geometry_contact", False)):
            track = row.get("track_id")
            if track is None:
                key = ("side_hand", f"{row.get('side')}:{row.get('hand_idx')}")
            else:
                key = ("track", str(track))
            groups.setdefault(key, []).append(row)
    for candidates in groups.values():
        ordered = sorted(candidates, key=lambda row: int(row["frame_idx"]))
        clusters: list[list[dict]] = []
        cur: list[dict] = []
        for row in ordered:
            if not cur:
                cur = [row]
            elif int(row["frame_idx"]) - int(cur[-1]["frame_idx"]) <= int(args.max_temporal_patch_gap_frames):
                cur.append(row)
            else:
                clusters.append(cur)
                cur = [row]
        if cur:
            clusters.append(cur)
        for cluster in clusters:
            n = len(cluster)
            for start in range(n):
                for end in range(start + int(args.min_temporal_patch_frames), n + 1):
                    window = cluster[start:end]
                    frames = [int(row["frame_idx"]) for row in window]
                    centers = [
                        np.asarray(row.get("best_patch_local_center_m"), dtype=float)
                        for row in window
                        if row.get("best_patch_local_center_m") is not None
                    ]
                    centers = [center for center in centers if center.shape == (3,) and np.isfinite(center).all()]
                    if len(centers) != len(window):
                        continue
                    drift = point_extent(np.stack(centers, axis=0))
                    if drift is None or float(drift) > float(args.accept_temporal_patch_local_drift_m):
                        continue
                    for row in window:
                        row["patch_temporal_support_frames"] = int(max(row["patch_temporal_support_frames"], len(set(frames))))
                        row["patch_temporal_support_span_frames"] = int(
                            max(row["patch_temporal_support_span_frames"], max(frames) - min(frames))
                        )
                        previous = row.get("patch_temporal_local_drift_m")
                        if previous is None or float(drift) < float(previous):
                            row["patch_temporal_local_drift_m"] = float(drift)
                        row["patch_temporal_local_drift_ok"] = True
                        row["patch_temporal_support_ok"] = True
            for row in cluster:
                row["reliable_for_contact"] = bool(row["reliable_geometry_contact"] and row["patch_temporal_support_ok"])


def condition_counts(rows: list[dict]) -> dict:
    keys = [
        "measurement_available",
        "detector_ok",
        "projection_ok",
        "depth_ok",
        "stable_depth_ok",
        "bone_scale_ok",
        "patch_distance_ok",
        "patch_signed_ok",
        "patch_spread_ok",
        "patch_penetration_ok",
        "contact_geometry_ok",
        "patch_temporal_support_ok",
        "reliable_geometry_contact",
        "reliable_for_contact",
    ]
    return {key: int(sum(bool(row.get(key, False)) for row in rows)) for key in keys}


def compact_rows(rows: list[dict], limit: int) -> list[dict]:
    keys = [
        "frame_idx",
        "hand_idx",
        "side",
        "detector_score",
        "median_joint_reprojection_px",
        "mano_minus_metric_depth_median_m",
        "stable_depth_fraction",
        "mask_candidate_vertices",
        "best_patch_vertices",
        "best_patch_distance_p95_m",
        "best_patch_signed_gap_median_m",
        "best_patch_signed_gap_p95_abs_m",
        "best_patch_penetration_fraction_010m",
        "best_patch_spread_m",
        "best_patch_local_spread_m",
        "contact_geometry_ok",
        "patch_temporal_support_frames",
        "patch_temporal_local_drift_m",
        "patch_temporal_support_ok",
        "reliable_for_contact",
    ]

    def sort_key(row: dict) -> tuple[float, float, float]:
        p95 = row.get("best_patch_distance_p95_m")
        signed = row.get("best_patch_signed_gap_median_m")
        reproj = row.get("median_joint_reprojection_px")
        return (
            float(p95) if p95 is not None and np.isfinite(float(p95)) else float("inf"),
            abs(float(signed)) if signed is not None and np.isfinite(float(signed)) else float("inf"),
            float(reproj) if reproj is not None and np.isfinite(float(reproj)) else float("inf"),
        )

    return [{key: row.get(key) for key in keys} for row in sorted(rows, key=sort_key)[:limit]]


def summarize_rows(rows: list[dict]) -> dict:
    measured = [row for row in rows if row["measurement_available"]]
    measured_high = [row for row in measured if row["detector_ok"]]
    reliable_geometry = [row for row in rows if row["reliable_geometry_contact"]]
    reliable = [row for row in rows if row["reliable_for_contact"]]
    return {
        "rows": int(len(rows)),
        "measured_rows": int(len(measured)),
        "measured_high_score_rows": int(len(measured_high)),
        "reliable_geometry_contact_rows": int(len(reliable_geometry)),
        "reliable_temporal_contact_rows": int(len(reliable)),
        "condition_counts_all": condition_counts(rows),
        "condition_counts_measured_high_score": condition_counts(measured_high),
        "summary_measured_high_score": {
            "joint_reprojection_px": summarize_key(measured_high, "median_joint_reprojection_px"),
            "mano_minus_metric_depth_m": summarize_key(measured_high, "mano_minus_metric_depth_median_m"),
            "stable_depth_fraction": summarize_key(measured_high, "stable_depth_fraction"),
            "mask_candidate_vertices": summarize(np.asarray([row["mask_candidate_vertices"] for row in measured_high], dtype=float)),
            "best_patch_distance_p95_m": summarize_key(measured_high, "best_patch_distance_p95_m"),
            "best_patch_signed_gap_median_m": summarize_key(measured_high, "best_patch_signed_gap_median_m"),
            "best_patch_signed_gap_p95_abs_m": summarize_key(measured_high, "best_patch_signed_gap_p95_abs_m"),
            "best_patch_penetration_fraction_010m": summarize_key(measured_high, "best_patch_penetration_fraction_010m"),
            "best_patch_front_separation_fraction_030m": summarize_key(measured_high, "best_patch_front_separation_fraction_030m"),
            "best_patch_spread_m": summarize_key(measured_high, "best_patch_spread_m"),
            "best_patch_local_spread_m": summarize_key(measured_high, "best_patch_local_spread_m"),
            "hand_bone_scale_m": summarize_key(measured_high, "hand_bone_scale_m"),
        },
        "summary_reliable_temporal_contact": {
            "joint_reprojection_px": summarize_key(reliable, "median_joint_reprojection_px"),
            "mano_minus_metric_depth_m": summarize_key(reliable, "mano_minus_metric_depth_median_m"),
            "best_patch_distance_p95_m": summarize_key(reliable, "best_patch_distance_p95_m"),
            "best_patch_signed_gap_median_m": summarize_key(reliable, "best_patch_signed_gap_median_m"),
            "best_patch_signed_gap_p95_abs_m": summarize_key(reliable, "best_patch_signed_gap_p95_abs_m"),
            "best_patch_penetration_fraction_010m": summarize_key(reliable, "best_patch_penetration_fraction_010m"),
        },
        "rows_preview": compact_rows([row for row in measured_high if int(row["mask_candidate_vertices"]) > 0], 80),
    }


def run(args: argparse.Namespace) -> dict:
    frames = load_frame_window(args.annotations, int(args.frame_start), int(args.frame_end))
    frame_to_depth_i, depths = load_depth_archive(args.metric_depth_npz)
    meshes = load_mesh_archive(args.object_mesh_npz)
    rows: list[dict] = []
    skipped: list[dict] = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1, max(1, int(args.frame_stride))):
        frame = frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_annotation_frame"})
            continue
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            mask_depth = object_mask_for(frame, depth, args.remote_output_root, args.local_output_root)
            world_vertices, faces = meshes[frame_idx]
            T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
            object_vertices_camera = camera_points(world_vertices, T)
            positive = np.isfinite(object_vertices_camera).all(axis=1) & (object_vertices_camera[:, 2] > 0.0)
            if int(np.count_nonzero(positive)) < max(10, len(object_vertices_camera) // 20):
                raise RuntimeError("object mesh has too few positive-depth camera vertices")
            object_normals_camera, normal_valid = oriented_vertex_normals(object_vertices_camera, faces)
            surface_valid = positive & normal_valid
            if int(np.count_nonzero(surface_valid)) < max(10, len(object_vertices_camera) // 20):
                raise RuntimeError("object mesh has too few positive-depth vertices with face-derived normals")
            object_vertices_surface = object_vertices_camera[surface_valid]
            object_normals_surface = object_normals_camera[surface_valid]
            tree = cKDTree(object_vertices_surface)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        for hand_i, hand in enumerate(frame.get("hands", [])):
            try:
                rows.append(row_for_hand(frame, hand_i, hand, depth, mask_depth, object_vertices_surface, object_normals_surface, tree, args))
            except Exception as exc:
                skipped.append(
                    {
                        "frame_idx": frame_idx,
                        "hand_idx": hand_i,
                        "side": hand.get("side"),
                        "reason": str(exc),
                    }
                )
    if not rows:
        raise RuntimeError(f"no mesh-surface contact rows; skipped={skipped[:20]}")
    annotate_temporal_support(rows, args)
    summary = summarize_rows(rows)
    report = {
        "status": "diagnostic_mesh_surface_contact_rows_found"
        if summary["reliable_temporal_contact_rows"] > 0
        else "diagnostic_no_reliable_mesh_surface_contact_rows",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "diagnose_mesh_surface_contact_v3",
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "intrinsics_source": str(args.intrinsics_source),
        **summary,
        "skipped_count": int(len(skipped)),
        "skipped_preview": skipped[:120],
        "thresholds": {
            "min_detector_score": float(args.min_detector_score),
            "max_good_median_reprojection_px": float(args.max_good_median_reprojection_px),
            "good_joint_reprojection_px": float(args.good_joint_reprojection_px),
            "max_good_depth_bias_m": float(args.max_good_depth_bias_m),
            "min_good_depth_joints": int(args.min_good_depth_joints),
            "min_stable_depth_fraction": float(args.min_stable_depth_fraction),
            "min_bone_scale_m": float(args.min_bone_scale_m),
            "max_bone_scale_m": float(args.max_bone_scale_m),
            "mask_contact_distance_px": float(args.mask_contact_distance_px),
            "patch_sizes": [int(value) for value in args.patch_sizes],
            "min_patch_vertices": int(args.min_patch_vertices),
            "accept_patch_distance_p95_m": float(args.accept_patch_distance_p95_m),
            "accept_patch_signed_gap_m": float(args.accept_patch_signed_gap_m),
            "accept_patch_signed_gap_p95_m": float(args.accept_patch_signed_gap_p95_m),
            "accept_patch_spread_m": float(args.accept_patch_spread_m),
            "accept_patch_penetration_fraction": float(args.accept_patch_penetration_fraction),
            "min_temporal_patch_frames": int(args.min_temporal_patch_frames),
            "max_temporal_patch_gap_frames": int(args.max_temporal_patch_gap_frames),
            "accept_temporal_patch_local_drift_m": float(args.accept_temporal_patch_local_drift_m),
        },
        "interpretation": (
            "This diagnostic tests broad-mask candidate contacts against the actual object mesh surface. "
            "For each MANO vertex that projects near the object mask, it queries the nearest object-mesh surface vertex "
            "in the current camera frame and estimates signed separation using the mesh normal oriented toward the camera. "
            "Rows pass only when image reprojection, UniDepth hand depth, MANO bone scale, mesh-surface distance, signed "
            "gap, penetration fraction, local patch spread, and temporal support all agree. Passing rows are contact evidence "
            "for this mesh hypothesis; failing rows mean the broad-mask contact result is not sufficient physical evidence."
        ),
    }
    if args.keep_detail:
        report["rows_detail"] = rows
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_detail", "rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "hand", "cli"], default="annotation-vggt")
    parser.add_argument("--intrinsics", type=float, nargs=4, default=[1200.0, 1175.0, 960.0, 540.0])
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--good-joint-reprojection-px", type=float, default=20.0)
    parser.add_argument("--max-good-median-reprojection-px", type=float, default=12.0)
    parser.add_argument("--max-good-depth-bias-m", type=float, default=0.030)
    parser.add_argument("--min-good-depth-joints", type=int, default=12)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.040)
    parser.add_argument("--min-stable-depth-fraction", type=float, default=0.75)
    parser.add_argument("--min-bone-scale-m", type=float, default=0.120)
    parser.add_argument("--max-bone-scale-m", type=float, default=0.240)
    parser.add_argument("--mask-contact-distance-px", type=float, default=8.0)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--min-patch-vertices", type=int, default=8)
    parser.add_argument("--accept-patch-distance-p95-m", type=float, default=0.040)
    parser.add_argument("--accept-patch-signed-gap-m", type=float, default=0.020)
    parser.add_argument("--accept-patch-signed-gap-p95-m", type=float, default=0.040)
    parser.add_argument("--accept-patch-spread-m", type=float, default=0.050)
    parser.add_argument("--accept-patch-penetration-fraction", type=float, default=0.25)
    parser.add_argument("--min-temporal-patch-frames", type=int, default=2)
    parser.add_argument("--max-temporal-patch-gap-frames", type=int, default=8)
    parser.add_argument("--accept-temporal-patch-local-drift-m", type=float, default=0.030)
    parser.add_argument("--keep-detail", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
