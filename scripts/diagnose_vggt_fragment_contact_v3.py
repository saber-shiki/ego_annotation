#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from diagnose_intrinsics_focal_sweep_v3 import (
    hand_bone_scale_m,
    solve_source_camera_translation,
    source_local_vertices,
)
from diagnose_surface_fragment_contact_v3 import load_tracks, localize
from diagnose_vggt_focal_sweep_v3 import (
    source_focal_to_vggt_intrinsics,
    summarize,
    vggt_predicted_source_focals,
)
from diagnose_vggt_mano_contact_v3 import points_to_vggt_frame, resize_mask, vggt_affine_from_source
from diagnose_hand_reprojection_depth_v3 import project_points
from optimize_object_factor_graph_v3 import mask_distance_map


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fragment_depth(
    archive: np.lib.npyio.NpzFile,
    frame_i: int,
    mask: np.ndarray,
    min_depth_conf: float,
    conf_quantile: float,
    min_depth_pixels: int,
) -> tuple[dict, dict]:
    depth = archive["depth"].astype(float)[frame_i]
    conf = archive["depth_conf"].astype(float)[frame_i]
    valid = mask & np.isfinite(depth) & (depth > 0.0) & np.isfinite(conf)
    conf_values = conf[valid]
    if conf_values.size == 0:
        raise RuntimeError(f"frame index {frame_i} fragment mask has no finite VGGT depth/confidence")
    threshold = max(float(min_depth_conf), float(np.quantile(conf_values, float(conf_quantile))))
    valid &= conf >= threshold
    values = depth[valid]
    if values.size < int(min_depth_pixels):
        raise RuntimeError(
            f"frame index {frame_i} fragment mask has {values.size} VGGT depth pixels below minimum {min_depth_pixels}"
        )
    yx = np.column_stack(np.nonzero(valid)).astype(np.int32)
    surface = {
        "valid_mask": valid,
        "yx": yx,
        "depth_values": values.astype(float),
        "median_depth": float(np.median(values)),
    }
    return surface, {
        "vggt_mask_pixels": int(np.count_nonzero(mask)),
        "vggt_depth_pixels": int(values.size),
        "vggt_depth_conf_threshold": float(threshold),
        "vggt_depth_median": float(np.median(values)),
        "vggt_depth_p05": float(np.percentile(values, 5)),
        "vggt_depth_p95": float(np.percentile(values, 95)),
    }


def track_fragments(
    tracks: list[dict],
    archive_frames: np.ndarray,
    target_size: int,
    remote_output_root: Path | None,
    local_output_root: Path | None,
    args: argparse.Namespace,
) -> list[dict]:
    frame_to_i = {int(frame_idx): i for i, frame_idx in enumerate(archive_frames.astype(int).tolist())}
    archive = np.load(args.vggt_archive)
    fragments = []
    skipped = []
    for track in tracks:
        track_id = track["track_id"]
        for frame_idx, entry in sorted(track["track"].items()):
            if frame_idx not in frame_to_i:
                continue
            if not entry.get("visible") or not entry.get("mask_path"):
                continue
            mask_path = localize(str(entry["mask_path"]), remote_output_root, local_output_root)
            mask = resize_mask(mask_path, int(target_size))
            if int(np.count_nonzero(mask)) < int(args.min_vggt_mask_pixels):
                skipped.append(
                    {
                        "frame_idx": int(frame_idx),
                        "track_id": track_id,
                        "reason": "mask_too_small_after_vggt_resize",
                        "vggt_mask_pixels": int(np.count_nonzero(mask)),
                    }
                )
                continue
            try:
                surface, depth_report = fragment_depth(
                    archive,
                    frame_to_i[int(frame_idx)],
                    mask,
                    float(args.min_depth_conf),
                    float(args.conf_quantile),
                    int(args.min_depth_pixels),
                )
            except RuntimeError as exc:
                skipped.append({"frame_idx": int(frame_idx), "track_id": track_id, "reason": str(exc)})
                continue
            fragments.append(
                {
                    "frame_idx": int(frame_idx),
                    "frame_i": int(frame_to_i[int(frame_idx)]),
                    "track_id": track_id,
                    "mask_path": str(mask_path),
                    "mask": mask,
                    "surface": surface,
                    "object_depth_vggt": float(surface["median_depth"]),
                    **depth_report,
                }
            )
    if not fragments:
        raise RuntimeError(f"no usable SAM2 fragments overlap VGGT archive frames {archive_frames.tolist()}; skipped={skipped}")
    return fragments


def gap_stats(gap: np.ndarray) -> dict:
    if gap.size == 0:
        return {
            "near_vertices": 0,
            "gap_median_m": None,
            "gap_p95_abs_m": None,
            "contact_fraction_010m": 0.0,
            "contact_fraction_030m": 0.0,
            "hand_in_front_fraction_010m": 0.0,
            "hand_behind_fraction_010m": 0.0,
        }
    return {
        "near_vertices": int(gap.size),
        "gap_median_m": float(np.median(gap)),
        "gap_p95_abs_m": float(np.percentile(np.abs(gap), 95)),
        "contact_fraction_010m": float(np.mean(np.abs(gap) <= 0.010)),
        "contact_fraction_030m": float(np.mean(np.abs(gap) <= 0.030)),
        "hand_in_front_fraction_010m": float(np.mean(gap < -0.010)),
        "hand_behind_fraction_010m": float(np.mean(gap > 0.010)),
    }


def surface_stats(distance: np.ndarray) -> dict:
    if distance.size == 0:
        return {
            "surface_distance_median_m": None,
            "surface_distance_p95_m": None,
            "surface_distance_fraction_010m": 0.0,
            "surface_distance_fraction_030m": 0.0,
        }
    return {
        "surface_distance_median_m": float(np.median(distance)),
        "surface_distance_p95_m": float(np.percentile(distance, 95)),
        "surface_distance_fraction_010m": float(np.mean(distance <= 0.010)),
        "surface_distance_fraction_030m": float(np.mean(distance <= 0.030)),
    }


def nearest_surface_residuals(
    vertices: np.ndarray,
    uv: np.ndarray,
    fragment: dict,
    intrinsics_vggt: np.ndarray,
    target_size: int,
    max_distance_px: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    surface = fragment["surface"]
    yx = surface["yx"]
    z = surface["depth_values"]
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] <= target_size - 1) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= target_size - 1)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0 or yx.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), 0
    xy_surface = np.c_[yx[:, 1], yx[:, 0]].astype(float)
    xy = uv[valid_indices].astype(float)
    best = []
    best_d2 = []
    chunk = 128
    for start in range(0, len(xy), chunk):
        block = xy[start : start + chunk]
        d2 = np.sum((block[:, None, :] - xy_surface[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        best.append(idx)
        best_d2.append(d2[np.arange(len(block)), idx])
    best_idx = np.concatenate(best).astype(int)
    dist_px = np.sqrt(np.concatenate(best_d2))
    near_local = dist_px <= float(max_distance_px)
    if not np.any(near_local):
        return np.asarray([], dtype=float), np.asarray([], dtype=float), 0
    vertex_idx = valid_indices[near_local]
    surface_idx = best_idx[near_local]
    local_z = z[surface_idx]
    gap = vertices[vertex_idx, 2] - local_z
    fx, fy, cx, cy = intrinsics_vggt.astype(float)
    u = xy_surface[surface_idx, 0]
    v = xy_surface[surface_idx, 1]
    surface_points = np.c_[(u - cx) / fx * local_z, (v - cy) / fy * local_z, local_z]
    distance = np.linalg.norm(vertices[vertex_idx] - surface_points, axis=1)
    return gap.astype(float), distance.astype(float), int(vertex_idx.size)


def hand_fragment_row(
    frame: dict,
    hand_i: int,
    hand: dict,
    fragment: dict,
    intrinsics_vggt: np.ndarray,
    target_size: int,
    args: argparse.Namespace,
) -> dict | None:
    if not bool(hand.get("measurement_available", False)):
        return None
    local_joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
    raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    if local_joints.shape != (21, 3) or raw2d.shape != (21, 2):
        return None
    local_vertices = source_local_vertices(hand)
    raw2d_vggt = points_to_vggt_frame(raw2d, frame["object"]["source_image_size"], target_size)
    trans = solve_source_camera_translation(local_joints, raw2d_vggt, intrinsics_vggt)
    joints = local_joints + trans[None, :]
    vertices = local_vertices + trans[None, :]
    if np.any(joints[:, 2] <= 0.0) or np.any(vertices[:, 2] <= 0.0):
        return None
    projected_joints = project_points(joints, intrinsics_vggt)
    reproj = np.linalg.norm(projected_joints - raw2d_vggt, axis=1)
    uv = project_points(vertices, intrinsics_vggt)
    local_gap, surface_distance, near_count = nearest_surface_residuals(
        vertices,
        uv,
        fragment,
        intrinsics_vggt,
        target_size,
        float(args.contact_distance_px),
    )
    dist = mask_distance_map(fragment["mask"])
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    x = np.clip(np.rint(uv[:, 0]).astype(int), 0, target_size - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(int), 0, target_size - 1)
    near = valid & (dist[y, x] <= float(args.contact_distance_px))
    median_gap = vertices[near, 2] - float(fragment["object_depth_vggt"])
    score = float(hand.get("detector_score", np.nan))
    median_reproj = float(np.median(reproj))
    bone_scale = float(hand_bone_scale_m(joints))
    stats = gap_stats(local_gap)
    median_stats = gap_stats(median_gap)
    distance_stats = surface_stats(surface_distance)
    median_ok = bool(
        near_count >= int(args.min_near_vertices)
        and stats["gap_median_m"] is not None
        and abs(float(stats["gap_median_m"])) <= float(args.accept_gap_m)
    )
    p95_ok = bool(
        near_count >= int(args.min_near_vertices)
        and stats["gap_p95_abs_m"] is not None
        and float(stats["gap_p95_abs_m"]) <= float(args.accept_p95_gap_m)
    )
    distance_ok = bool(
        near_count >= int(args.min_near_vertices)
        and distance_stats["surface_distance_p95_m"] is not None
        and float(distance_stats["surface_distance_p95_m"]) <= float(args.accept_p95_distance_m)
    )
    projection_ok = bool(median_reproj <= float(args.accept_reprojection_px))
    detector_ok = bool(np.isfinite(score) and score >= float(args.min_detector_score))
    bone_scale_ok = bool(float(args.min_bone_scale_m) <= bone_scale <= float(args.max_bone_scale_m))
    no_large_violation = bool(
        stats["hand_in_front_fraction_010m"] <= float(args.accept_surface_violation_fraction)
        and stats["hand_behind_fraction_010m"] <= float(args.accept_surface_violation_fraction)
    )
    return {
        "frame_idx": int(fragment["frame_idx"]),
        "track_id": fragment["track_id"],
        "hand_index": int(hand_i),
        "side": hand.get("side", "unknown"),
        "detector_score": score,
        "median_joint_reprojection_px_vggt": median_reproj,
        "p95_joint_reprojection_px_vggt": float(np.percentile(reproj, 95)),
        "cam_t_z_vggt": float(trans[2]),
        "hand_bone_scale_m": bone_scale,
        "object_depth_vggt": float(fragment["object_depth_vggt"]),
        "vggt_mask_pixels": int(fragment["vggt_mask_pixels"]),
        "vggt_depth_pixels": int(fragment["vggt_depth_pixels"]),
        **stats,
        "median_depth_near_vertices": int(median_stats["near_vertices"]),
        "median_depth_gap_median_m": median_stats["gap_median_m"],
        "median_depth_gap_p95_abs_m": median_stats["gap_p95_abs_m"],
        **distance_stats,
        "detector_ok": detector_ok,
        "projection_ok": projection_ok,
        "bone_scale_ok": bone_scale_ok,
        "median_contact_ok": median_ok,
        "p95_contact_ok": p95_ok,
        "surface_distance_ok": distance_ok,
        "surface_violation_ok": no_large_violation,
        "reliable_for_contact": bool(
            detector_ok and projection_ok and bone_scale_ok and median_ok and p95_ok and distance_ok and no_large_violation
        ),
    }


def compact_rows(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []

    def sort_key(row: dict) -> tuple[float, float, float]:
        p95 = row.get("gap_p95_abs_m")
        median = row.get("gap_median_m")
        reproj = row.get("median_joint_reprojection_px_vggt")
        return (
            float(p95) if p95 is not None and np.isfinite(float(p95)) else math.inf,
            abs(float(median)) if median is not None and np.isfinite(float(median)) else math.inf,
            float(reproj) if reproj is not None and np.isfinite(float(reproj)) else math.inf,
        )

    keys = [
        "frame_idx",
        "track_id",
        "hand_index",
        "side",
        "detector_score",
        "median_joint_reprojection_px_vggt",
        "near_vertices",
        "gap_median_m",
        "gap_p95_abs_m",
        "contact_fraction_010m",
        "contact_fraction_030m",
        "hand_in_front_fraction_010m",
        "hand_behind_fraction_010m",
        "object_depth_vggt",
        "cam_t_z_vggt",
        "surface_distance_median_m",
        "surface_distance_p95_m",
        "surface_distance_fraction_030m",
        "median_depth_gap_median_m",
        "median_depth_gap_p95_abs_m",
        "reliable_for_contact",
    ]
    return [{key: row.get(key) for key in keys} for row in sorted(rows, key=sort_key)[:limit]]


def report_for_focal(rows: list[dict], focal_px: float, width: int, height: int, preview_limit: int) -> dict:
    high = [row for row in rows if row["detector_ok"]]
    near = [row for row in high if int(row["near_vertices"]) > 0]
    supported = [row for row in high if int(row["near_vertices"]) >= 20]
    reliable = [row for row in rows if row["reliable_for_contact"]]
    return {
        "source_focal_px": float(focal_px),
        "horizontal_fov_deg": float(2.0 * math.degrees(math.atan(width / (2.0 * float(focal_px))))),
        "vertical_fov_deg": float(2.0 * math.degrees(math.atan(height / (2.0 * float(focal_px))))),
        "rows": int(len(rows)),
        "high_score_rows": int(len(high)),
        "near_fragment_rows": int(len(near)),
        "supported_fragment_rows": int(len(supported)),
        "reliable_contact_rows": int(len(reliable)),
        "summary_high_score": {
            "joint_reprojection_px_vggt": summarize([row["median_joint_reprojection_px_vggt"] for row in high]),
            "near_vertices": summarize([row["near_vertices"] for row in high]),
            "gap_median_m": summarize([row["gap_median_m"] for row in supported]),
            "gap_p95_abs_m": summarize([row["gap_p95_abs_m"] for row in supported]),
            "median_depth_gap_median_m": summarize([row["median_depth_gap_median_m"] for row in supported]),
            "median_depth_gap_p95_abs_m": summarize([row["median_depth_gap_p95_abs_m"] for row in supported]),
            "surface_distance_median_m": summarize([row["surface_distance_median_m"] for row in supported]),
            "surface_distance_p95_m": summarize([row["surface_distance_p95_m"] for row in supported]),
            "surface_distance_fraction_010m": summarize([row["surface_distance_fraction_010m"] for row in supported]),
            "surface_distance_fraction_030m": summarize([row["surface_distance_fraction_030m"] for row in supported]),
            "contact_fraction_010m": summarize([row["contact_fraction_010m"] for row in supported]),
            "contact_fraction_030m": summarize([row["contact_fraction_030m"] for row in supported]),
            "hand_in_front_fraction_010m": summarize([row["hand_in_front_fraction_010m"] for row in supported]),
            "hand_behind_fraction_010m": summarize([row["hand_behind_fraction_010m"] for row in supported]),
            "object_depth_vggt": summarize([row["object_depth_vggt"] for row in high]),
            "hand_bone_scale_m": summarize([row["hand_bone_scale_m"] for row in high]),
        },
        "condition_counts": {
            "detector_ok": int(sum(row["detector_ok"] for row in rows)),
            "projection_ok": int(sum(row["projection_ok"] for row in rows)),
            "bone_scale_ok": int(sum(row["bone_scale_ok"] for row in rows)),
            "median_contact_ok": int(sum(row["median_contact_ok"] for row in rows)),
            "p95_contact_ok": int(sum(row["p95_contact_ok"] for row in rows)),
            "surface_distance_ok": int(sum(row["surface_distance_ok"] for row in rows)),
            "surface_violation_ok": int(sum(row["surface_violation_ok"] for row in rows)),
            "reliable_for_contact": int(sum(row["reliable_for_contact"] for row in rows)),
        },
        "rows_preview": compact_rows(near, preview_limit),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    archive = np.load(args.vggt_archive)
    archive_frames = archive["frame_idx"].astype(int)
    source_width, source_height = int(args.width), int(args.height)
    tracks = load_tracks(args.sam2_root)
    fragments = track_fragments(
        tracks,
        archive_frames,
        int(args.target_size),
        args.remote_output_root,
        args.local_output_root,
        args,
    )
    predicted = vggt_predicted_source_focals(archive, source_width, source_height, int(args.target_size))
    source_focals = [float(focal) for focal in args.source_focals]
    if args.include_vggt_predicted_focal:
        predicted_focal = float(predicted["median_mean_source_focal_px"])
        if all(abs(predicted_focal - existing) > 1e-6 for existing in source_focals):
            source_focals.append(predicted_focal)
    source_focals = sorted(source_focals)
    fragment_records = [{key: value for key, value in fragment.items() if key not in {"mask", "surface"}} for fragment in fragments]
    reports = []
    rows_by_focal: dict[str, list[dict]] = {}
    for focal_px in source_focals:
        K4 = source_focal_to_vggt_intrinsics(
            float(focal_px),
            source_width,
            source_height,
            float(args.cx),
            float(args.cy),
            int(args.target_size),
        )
        rows = []
        for fragment in fragments:
            frame = frames.get(int(fragment["frame_idx"]))
            if frame is None:
                raise RuntimeError(f"annotations missing frame {fragment['frame_idx']}")
            for hand_i, hand in enumerate(frame.get("hands", [])):
                row = hand_fragment_row(frame, hand_i, hand, fragment, K4, int(args.target_size), args)
                if row is not None:
                    rows.append(row)
        if not rows:
            raise RuntimeError(f"no valid VGGT fragment-contact rows for focal {focal_px}")
        key = f"{focal_px:.6f}"
        rows_by_focal[key] = rows
        reports.append(report_for_focal(rows, focal_px, source_width, source_height, int(args.row_preview_limit)))
    ranked_by_supported_gap = sorted(
        reports,
        key=lambda row: abs(float(row["summary_high_score"]["gap_median_m"].get("median", math.inf))),
    )
    report = {
        "status": "diagnostic_vggt_fragment_contact_rows_found"
        if any(row["reliable_contact_rows"] for row in reports)
        else "diagnostic_no_reliable_vggt_fragment_contact_rows",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "diagnose_vggt_fragment_contact_v3",
        "annotations": str(args.annotations),
        "vggt_archive": str(args.vggt_archive),
        "sam2_root": str(args.sam2_root),
        "vggt_frames": archive_frames.astype(int).tolist(),
        "target_size": int(args.target_size),
        "vggt_affine_source_to_target": {
            "scale_x_scale_y_pad_top": [float(v) for v in vggt_affine_from_source(source_width, source_height, int(args.target_size))],
        },
        "vggt_predicted_source_focal": predicted,
        "fragment_records": fragment_records,
        "focal_reports": reports,
        "ranked_by_supported_gap": [
            {
                "source_focal_px": row["source_focal_px"],
                "supported_fragment_rows": row["supported_fragment_rows"],
                "reliable_contact_rows": row["reliable_contact_rows"],
                "gap_median_m": row["summary_high_score"]["gap_median_m"],
                "gap_p95_abs_m": row["summary_high_score"]["gap_p95_abs_m"],
                "surface_distance_p95_m": row["summary_high_score"]["surface_distance_p95_m"],
                "joint_reprojection_px_vggt": row["summary_high_score"]["joint_reprojection_px_vggt"],
            }
            for row in ranked_by_supported_gap
        ],
        "acceptance": {
            "min_detector_score": float(args.min_detector_score),
            "min_near_vertices": int(args.min_near_vertices),
            "accept_gap_m": float(args.accept_gap_m),
            "accept_p95_gap_m": float(args.accept_p95_gap_m),
            "accept_p95_distance_m": float(args.accept_p95_distance_m),
            "accept_reprojection_px": float(args.accept_reprojection_px),
            "accept_surface_violation_fraction": float(args.accept_surface_violation_fraction),
        },
        "interpretation": (
            "This diagnostic replaces Depth Anything fragment depth with full-scene VGGT depth while keeping the same "
            "image-conditioned SAM2 fragments and MANO 2D reprojection solve. It compares each near hand vertex to the "
            "nearest valid VGGT surface pixel in the fragment, then reports local depth gap and 3D distance. Passing rows "
            "would indicate that the fragment-contact failure was mainly a monocular depth failure. Zero reliable rows "
            "leave the contradiction in the hand/object metric state or in fragment identity."
        ),
    }
    if args.keep_detail:
        report["rows_by_focal"] = rows_by_focal
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_by_focal"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-focals", type=float, nargs="*", default=[1400.0, 2304.0])
    parser.add_argument("--include-vggt-predicted-focal", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cx", type=float, default=960.0)
    parser.add_argument("--cy", type=float, default=540.0)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--remote-output-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/data"))
    parser.add_argument("--local-output-root", type=Path, default=Path("/data2/ego_annotation_outputs"))
    parser.add_argument("--contact-distance-px", type=float, default=18.0)
    parser.add_argument("--min-depth-conf", type=float, default=0.0)
    parser.add_argument("--conf-quantile", type=float, default=0.0)
    parser.add_argument("--min-vggt-mask-pixels", type=int, default=20)
    parser.add_argument("--min-depth-pixels", type=int, default=20)
    parser.add_argument("--min-near-vertices", type=int, default=20)
    parser.add_argument("--min-detector-score", type=float, default=0.45)
    parser.add_argument("--min-bone-scale-m", type=float, default=0.120)
    parser.add_argument("--max-bone-scale-m", type=float, default=0.240)
    parser.add_argument("--accept-gap-m", type=float, default=0.030)
    parser.add_argument("--accept-p95-gap-m", type=float, default=0.060)
    parser.add_argument("--accept-p95-distance-m", type=float, default=0.060)
    parser.add_argument("--accept-reprojection-px", type=float, default=18.0)
    parser.add_argument("--accept-surface-violation-fraction", type=float, default=0.10)
    parser.add_argument("--row-preview-limit", type=int, default=80)
    parser.add_argument("--keep-detail", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
