#!/usr/bin/env python3
"""Prediction-only global RGB orientation initializer for the V20 experiment.

This builder uses SuperPoint+LightGlue matches inside object-owned masks, lifts
only the source keypoints with prediction-side UniDepth, and solves a calibrated
source-world -> target-world PnP edge.  It is deliberately independent of
HOT3D GT, CAD, and generated meshes.  The output is an initializer/evidence
bundle; it is not a formal pose result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

try:
    from v20_prediction_contracts import append_prediction_stage, assert_prediction_only
except ModuleNotFoundError:  # pragma: no cover - supports direct test imports
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v20_prediction_contracts import append_prediction_stage, assert_prediction_only


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def resize_intrinsics(K: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    sx = float(target_wh[0]) / float(source_wh[0])
    sy = float(target_wh[1]) / float(source_wh[1])
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = sx * (out[0, 2] + 0.5) - 0.5
    out[1, 2] = sy * (out[1, 2] + 0.5) - 0.5
    return out


def validate_depth_contract(
    depth_ids: np.ndarray,
    depths: np.ndarray,
    confidences: np.ndarray,
    source_size: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[int, int]:
    source_size_array = np.asarray(source_size, dtype=np.int64).reshape(-1)
    if source_size_array.shape != (2,) or np.any(source_size_array <= 0):
        raise RuntimeError("depth source_size must be two positive dimensions")
    source_wh = (int(source_size_array[0]), int(source_size_array[1]))
    if depths.ndim != 3 or depths.shape[2] != source_wh[0] or depths.shape[1] != source_wh[1]:
        raise RuntimeError(f"depth raster {depths.shape} disagrees with source_size {source_wh}")
    if confidences.shape != depths.shape:
        raise RuntimeError("depth/confidence raster shapes disagree")
    if intrinsics.shape != (len(depth_ids), 4):
        raise RuntimeError("per-frame depth intrinsics shape is invalid")
    if len(np.unique(depth_ids)) != len(depth_ids):
        raise RuntimeError("depth frame_idx values are not unique")
    return source_wh


def find_case_root(annotation_path: Path) -> Path:
    # .../<case>/measurements/object_geometry/visible_geometry/<object>/annotations.json
    return annotation_path.expanduser().resolve().parents[4]


def object_for_frame(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(obj, dict) and obj.get("object_id") == object_id:
            return obj
    return None


def lift_source_world(
    frame_idx: int,
    uv: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv, dtype=np.float64)
    h, w = depth.shape[:2]
    x = np.clip(np.rint((uv[:, 0] + 0.5) * w / mask.shape[1] - 0.5).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint((uv[:, 1] + 0.5) * h / mask.shape[0] - 0.5).astype(np.int64), 0, h - 1)
    xi = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, mask.shape[1] - 1)
    yi = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, mask.shape[0] - 1)
    z = np.asarray(depth[y, x], dtype=np.float64)
    q = np.asarray(confidence[y, x], dtype=np.float64)
    valid = (mask[yi, xi] > 0) & np.isfinite(z) & (z > 1.0e-4) & np.isfinite(q) & (q >= float(min_confidence))
    fx, fy, cx, cy = np.asarray(K, dtype=np.float64).reshape(-1).tolist()
    camera = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
    homogeneous = np.column_stack((camera, np.ones(len(camera), dtype=np.float64)))
    world = (homogeneous @ np.asarray(T_world_camera, dtype=np.float64).T)[:, :3]
    valid &= np.isfinite(world).all(axis=1)
    return world, valid


def feature_pair(
    source_idx: int,
    target_idx: int,
    features: dict[int, dict[str, torch.Tensor]],
    masks: dict[int, np.ndarray],
    frames: dict[int, dict[str, Any]],
    depths: np.ndarray,
    confidences: np.ndarray,
    depth_pos: dict[int, int],
    depth_intrinsics: np.ndarray,
    source_wh: tuple[int, int],
    matcher: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    initial_pose_by_frame: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray] | None, dict[str, np.ndarray] | None]:
    from lightglue.utils import rbd
    with torch.inference_mode():
        matched = rbd(matcher({"image0": features[source_idx], "image1": features[target_idx]}))
    matched = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value for key, value in matched.items()}
    raw_pairs = matched.get("matches")
    pairs = np.asarray(raw_pairs if raw_pairs is not None else [], dtype=np.int64)
    raw_scores = matched.get("scores")
    scores = np.asarray(raw_scores if raw_scores is not None else [], dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        pairs = np.empty((0, 2), dtype=np.int64)
    kp0 = features[source_idx]["keypoints"].detach().cpu().numpy()
    kp1 = features[target_idx]["keypoints"].detach().cpu().numpy()
    if kp0.ndim == 3:
        kp0 = kp0[0]
    if kp1.ndim == 3:
        kp1 = kp1[0]
    uv0 = kp0[pairs[:, 0]] if len(pairs) else np.empty((0, 2))
    uv1 = kp1[pairs[:, 1]] if len(pairs) else np.empty((0, 2))
    mask0 = masks[source_idx]
    mask1 = masks[target_idx]
    if len(pairs):
        x0 = np.clip(np.rint(uv0[:, 0]).astype(np.int64), 0, mask0.shape[1] - 1)
        y0 = np.clip(np.rint(uv0[:, 1]).astype(np.int64), 0, mask0.shape[0] - 1)
        x1 = np.clip(np.rint(uv1[:, 0]).astype(np.int64), 0, mask1.shape[1] - 1)
        y1 = np.clip(np.rint(uv1[:, 1]).astype(np.int64), 0, mask1.shape[0] - 1)
        erode0 = cv2.erode((mask0 > 0).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
        erode1 = cv2.erode((mask1 > 0).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
        keep = (erode0[y0, x0] > 0) & (erode1[y1, x1] > 0)
        uv0 = uv0[keep]
        uv1 = uv1[keep]
        scores = scores[keep] if len(scores) == len(keep) else np.ones(len(uv0))
    source_frame = frames[source_idx]
    target_frame = frames[target_idx]
    source_camera = np.asarray(source_frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    target_camera = np.asarray(target_frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    source_intr = depth_intrinsics[depth_pos[source_idx]]
    target_intr = depth_intrinsics[depth_pos[target_idx]]
    source_world, valid = lift_source_world(source_idx, uv0, mask0, depths[depth_pos[source_idx]], confidences[depth_pos[source_idx]], source_intr, source_camera, float(args.min_confidence))
    source_world = source_world[valid]
    target_uv = uv1[valid]
    if len(source_world) < int(args.min_matches):
        return {"source_frame_idx": source_idx, "target_frame_idx": target_idx, "status": "rejected", "matches_total": int(len(pairs)), "matches_object": int(len(source_world)), "reason": "too_few_object_matches"}, None, None
    K_target = np.asarray([[target_intr[0], 0.0, target_intr[2]], [0.0, target_intr[1], target_intr[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_target = resize_intrinsics(
        K_target, source_wh, (mask1.shape[1], mask1.shape[0])
    )
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        source_world.astype(np.float64), target_uv.astype(np.float64), K_target,
        np.zeros(5, dtype=np.float64), iterationsCount=int(args.pnp_iterations),
        reprojectionError=float(args.reprojection_threshold_px), confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 3:
        return {"source_frame_idx": source_idx, "target_frame_idx": target_idx, "status": "rejected", "matches_total": int(len(pairs)), "matches_object": int(len(source_world)), "reason": "pnp_failed"}, None, None
    inliers = inliers[:, 0]
    try:
        rvec, tvec = cv2.solvePnPRefineLM(source_world[inliers], target_uv[inliers], K_target, np.zeros(5), rvec, tvec)
    except cv2.error:
        pass
    R_camera_source, _ = cv2.Rodrigues(rvec)
    T_wc_target = target_camera
    R_delta = T_wc_target[:3, :3] @ R_camera_source
    t_delta = T_wc_target[:3, :3] @ tvec[:, 0] + T_wc_target[:3, 3]
    projected, _ = cv2.projectPoints(source_world[inliers], rvec, tvec, K_target, np.zeros(5))
    reprojection = np.linalg.norm(projected.reshape(-1, 2) - target_uv[inliers], axis=1)
    covariance = np.cov(source_world[inliers].T) if len(inliers) >= 4 else np.zeros((3, 3))
    eig = np.linalg.eigvalsh(covariance) if np.isfinite(covariance).all() else np.zeros(3)
    eig = np.maximum(eig, 0.0)
    conditioning = float(eig[0] / max(eig[-1], 1.0e-12))
    inlier_fraction = float(len(inliers) / max(1, len(source_world)))
    rotation_deg = float(np.degrees(Rotation.from_matrix(R_delta).magnitude()))
    status = "accepted" if len(inliers) >= int(args.min_inliers) and inlier_fraction >= float(args.min_inlier_fraction) and float(np.median(reprojection)) <= float(args.max_reprojection_median_px) and rotation_deg <= float(args.max_edge_rotation_deg) else "rejected"
    reason = None if status == "accepted" else "quality_gate"
    row = {
        "source_frame_idx": source_idx,
        "target_frame_idx": target_idx,
        "frame_gap": target_idx - source_idx,
        "status": status,
        "reason": reason,
        "matches_total": int(len(pairs)),
        "matches_object": int(len(source_world)),
        "inliers": int(len(inliers)),
        "inlier_fraction": inlier_fraction,
        "matching_score_median": float(np.median(scores)) if len(scores) else None,
        "rotation_deg": rotation_deg,
        "translation_norm_m": float(np.linalg.norm(t_delta)),
        "reprojection_median_px": float(np.median(reprojection)),
        "reprojection_p95_px": float(np.percentile(reprojection, 95)),
        "source_3d_covariance_eigenvalues": eig.tolist(),
        "source_3d_conditioning": conditioning,
    }
    evidence = None
    if status == "accepted":
        if source_idx not in initial_pose_by_frame:
            raise RuntimeError(f"initial pose report lacks RGB source frame {source_idx}")
        source_R0, source_t0 = initial_pose_by_frame[source_idx]
        canonical_points = (source_world[inliers] - source_t0[None, :]) @ source_R0
        target_intrinsics = np.asarray(
            [K_target[0, 0], K_target[1, 1], K_target[0, 2], K_target[1, 2]],
            dtype=np.float64,
        )
        point_weights = np.clip(np.exp(-reprojection / 2.0), 0.25, 1.0)
        evidence = {
            "canonical_points": np.asarray(canonical_points, dtype=np.float64),
            "target_uv": np.asarray(target_uv[inliers], dtype=np.float64),
            "target_intrinsics": target_intrinsics,
            "target_camera": np.asarray(target_camera, dtype=np.float64),
            "weights": point_weights,
        }
        row["reprojection_evidence_count"] = int(len(canonical_points))
        row["reprojection_evidence_contract"] = "source_initial_canonical_points_to_target_uv"
    return row, (R_delta, t_delta) if status == "accepted" else None, evidence


def build_edge_pairs(
    selected: list[int], key_ids: list[int], keyframe_neighbor_span: int
) -> list[tuple[int, int]]:
    if int(keyframe_neighbor_span) < 1:
        raise RuntimeError("keyframe_neighbor_span must be positive")
    pairs = set(zip(selected[:-1], selected[1:]))
    for offset in range(1, int(keyframe_neighbor_span) + 1):
        for a, b in zip(key_ids[:-offset], key_ids[offset:]):
            pairs.add((a, b))
    if len(key_ids) >= 4:
        for a, b in (
            (key_ids[0], key_ids[len(key_ids) // 2]),
            (key_ids[len(key_ids) // 2], key_ids[-1]),
            (key_ids[0], key_ids[-1]),
        ):
            pairs.add((a, b))
    return sorted((a, b) for a, b in pairs if a < b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, required=True)
    parser.add_argument("--initial-pose-report", type=Path, required=True)
    parser.add_argument("--object-id", default="carton_milk")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=149)
    parser.add_argument("--keyframe-interval", type=int, default=5)
    parser.add_argument(
        "--keyframe-neighbor-span",
        type=int,
        default=1,
        help="Connect each keyframe to this many subsequent keyframes.",
    )
    parser.add_argument("--max-keyframes", type=int, default=40)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ratio-test", type=float, default=0.78)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--min-matches", type=int, default=30)
    parser.add_argument("--min-inliers", type=int, default=30)
    parser.add_argument("--min-inlier-fraction", type=float, default=0.20)
    parser.add_argument("--pnp-iterations", type=int, default=1000)
    parser.add_argument("--reprojection-threshold-px", type=float, default=2.5)
    parser.add_argument("--max-reprojection-median-px", type=float, default=1.5)
    parser.add_argument("--max-edge-rotation-deg", type=float, default=25.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lightglue-root", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-pose-report", type=Path, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lightglue_root.expanduser().resolve()))
    from lightglue import LightGlue, SuperPoint  # type: ignore

    device = torch.device(args.device)
    annotations = load_json(args.annotations)
    assert_prediction_only(
        annotations, label="RGB edge annotations", object_id=args.object_id
    )
    initial_report_path = args.initial_pose_report.expanduser().resolve()
    initial_report = load_json(initial_report_path)
    assert_prediction_only(
        initial_report,
        label="RGB edge initial pose report",
        object_id=args.object_id,
    )
    frames = {
        int(f["frame_idx"]): f
        for f in annotations.get("frames", [])
        if isinstance(f, dict) and f.get("frame_idx") is not None
    }
    annotation_root = args.annotations.expanduser().resolve()
    case_root = find_case_root(annotation_root)
    with np.load(args.depth_npz.expanduser().resolve(), allow_pickle=False) as data:
        required = {"frame_idx", "depth", "confidence", "source_size", "intrinsics_fx_fy_cx_cy"}
        missing_keys = sorted(required - set(data.files))
        if missing_keys:
            raise RuntimeError(f"depth archive lacks camera contract keys: {missing_keys}")
        depth_ids = np.asarray(data["frame_idx"], dtype=np.int64)
        depths = np.asarray(data["depth"], dtype=np.float32)
        confidences = np.asarray(data["confidence"], dtype=np.float32)
        source_size_array = np.asarray(data["source_size"], dtype=np.int64).reshape(-1)
        depth_intrinsics = np.asarray(data["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        source_wh = validate_depth_contract(
            depth_ids, depths, confidences, source_size_array, depth_intrinsics
        )
    depth_pos = {int(idx): i for i, idx in enumerate(depth_ids.tolist())}
    selected = [i for i in range(int(args.frame_start), int(args.frame_end) + 1) if i in frames and i in depth_pos]
    if len(selected) < 2:
        raise RuntimeError("not enough selected frames")

    extractor = SuperPoint(max_num_keypoints=int(args.max_keypoints)).eval().to(device)
    matcher = LightGlue(features="superpoint", depth_confidence=0.9, width_confidence=0.95, filter_threshold=0.2).eval().to(device)
    features: dict[int, dict[str, torch.Tensor]] = {}
    masks: dict[int, np.ndarray] = {}
    for n, idx in enumerate(selected, 1):
        obj = object_for_frame(frames[idx], args.object_id)
        if obj is None:
            continue
        mask = cv2.imread(str(obj.get("mask_path") or ""), cv2.IMREAD_GRAYSCALE)
        rgb_path = case_root / "input/raw_frame_manifest/rgb" / f"{idx:06d}.jpg"
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if mask is None or image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).to(device)
        with torch.inference_mode():
            features[idx] = {key: value.detach() for key, value in extractor.extract(tensor, resize=None).items()}
        masks[idx] = mask
        if n == 1 or n % 25 == 0 or n == len(selected):
            print(f"[global-rgb] extracted {n}/{len(selected)}", flush=True)

    key_ids = [idx for idx in selected if idx in features and (idx == selected[0] or (idx - selected[0]) % int(args.keyframe_interval) == 0)]
    if selected[-1] not in key_ids:
        key_ids.append(selected[-1])
    key_ids = sorted(set(key_ids))
    if len(key_ids) > int(args.max_keyframes):
        interior = key_ids[1:-1]
        take = max(0, int(args.max_keyframes) - 2)
        chosen = [interior[int(round(i * (len(interior) - 1) / max(1, take - 1)))] for i in range(take)] if take and interior else []
        key_ids = sorted(set([key_ids[0], *chosen, key_ids[-1]]))

    edge_pairs = [
        (a, b)
        for a, b in build_edge_pairs(
            selected, key_ids, int(args.keyframe_neighbor_span)
        )
        if a in features and b in features
    ]

    initial_rows = {
        int(row["frame_idx"]): row
        for row in initial_report.get("pose_rows", [])
        if isinstance(row, dict)
        and row.get("rotation_world_from_completed_canonical_matrix") is not None
    }
    initial_pose_by_frame = {
        idx: (
            np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64),
            np.asarray(row["translation_world_m"], dtype=np.float64),
        )
        for idx, row in initial_rows.items()
        if row.get("translation_world_m") is not None
    }
    if any(idx not in initial_pose_by_frame for idx in selected):
        missing_pose = sorted(idx for idx in selected if idx not in initial_pose_by_frame)
        raise RuntimeError(f"initial pose report lacks selected frames: {missing_pose[:20]}")

    edges: list[dict[str, Any]] = []
    edge_motions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    edge_reprojection_evidence: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for n, (a, b) in enumerate(edge_pairs, 1):
        row, motion, evidence = feature_pair(
            a, b, features, masks, frames, depths, confidences, depth_pos,
            depth_intrinsics, source_wh, matcher, device, args, initial_pose_by_frame
        )
        edges.append(row)
        if motion is not None:
            edge_motions[(a, b)] = motion
        if evidence is not None:
            edge_reprojection_evidence[(a, b)] = evidence
        print(f"[global-rgb] edge {n}/{len(edge_pairs)} {a}->{b} {row['status']} matches={row.get('matches_object')} inliers={row.get('inliers')}", flush=True)

    if selected[0] not in initial_rows:
        raise RuntimeError("initial pose report lacks anchor frame")
    anchor_R = np.asarray(initial_rows[selected[0]]["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    anchor_t = np.asarray(initial_rows[selected[0]]["translation_world_m"], dtype=np.float64)
    chain: dict[int, tuple[np.ndarray, np.ndarray]] = {selected[0]: (anchor_R.copy(), anchor_t.copy())}
    for a, b in zip(selected[:-1], selected[1:]):
        motion = edge_motions.get((a, b))
        if motion is None or a not in chain:
            continue
        R_delta, t_delta = motion
        R_prev, t_prev = chain[a]
        chain[b] = (R_delta @ R_prev, R_delta @ t_prev + t_delta)
    chain_rows = []
    for original in initial_report.get("pose_rows", []):
        if not isinstance(original, dict):
            chain_rows.append(original)
            continue
        row = dict(original)
        idx = int(row.get("frame_idx", -1))
        if idx in chain:
            row["rotation_world_from_completed_canonical_matrix"] = chain[idx][0].tolist()
            row["translation_world_m"] = chain[idx][1].tolist()
            row["pose_source"] = "v20_prediction_lightglue_rgb_depth_chain_diagnostic"
            row["diagnostic_only"] = True
            row["generated_geometry_pose_evidence_consumed"] = False
        chain_rows.append(row)

    metadata = {
        "schema": "v20_prediction_lightglue_rgb_depth_global_orientation_edges_v2",
        "annotations": str(args.annotations.expanduser().resolve()),
        "depth_npz": str(args.depth_npz.expanduser().resolve()),
        "initial_pose_report": str(args.initial_pose_report.expanduser().resolve()),
        "object_id": args.object_id,
        "prediction_side_only": True,
        "generated_geometry_consumed": False,
        "gt_used_as_solver_input": False,
        "reprojection_evidence_contract": {
            "present": True,
            "storage": "NPZ_flattened_points_with_edge_offsets",
            "canonical_frame": "initial_prediction_pose_source_object_frame",
            "target_frame": "camera_pixels_at_mask_resolution",
            "used_for_pose_objective_only_when_explicitly_enabled": True,
        },
        "camera_contract": {
            "source_size_wh": list(source_wh),
            "depth_raster_shape": list(depths.shape[1:]),
            "intrinsics_plane": "depth_source_plane",
            "target_intrinsics_resize_from_source_wh": list(source_wh),
        },
        "input_sha256": {
            "annotations": sha256_file(args.annotations),
            "depth_npz": sha256_file(args.depth_npz),
            "initial_pose_report": sha256_file(args.initial_pose_report),
        },
    }
    report = {
        **metadata,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "selected_frame_count": len(selected),
        "keyframes": key_ids,
        "edge_count": len(edges),
        "accepted_count": sum(row.get("status") == "accepted" for row in edges),
        "rejected_count": sum(row.get("status") != "accepted" for row in edges),
        "reprojection_evidence_edge_count": len(edge_reprojection_evidence),
        "reprojection_evidence_point_count": int(sum(len(value["weights"]) for value in edge_reprojection_evidence.values())),
        "chain_covered_frame_count": len(chain),
        "rows": edges,
        "parameters": {
            "keyframe_interval": int(args.keyframe_interval),
            "keyframe_neighbor_span": int(args.keyframe_neighbor_span),
            "max_keyframes": int(args.max_keyframes),
            "max_edge_rotation_deg": float(args.max_edge_rotation_deg),
            "min_matches": int(args.min_matches),
            "min_inliers": int(args.min_inliers),
            "min_inlier_fraction": float(args.min_inlier_fraction),
            "max_reprojection_median_px": float(
                args.max_reprojection_median_px
            ),
        },
    }
    append_prediction_stage(
        report,
        stage="prediction_rgb_depth_edge_builder",
        input_hashes=metadata["input_sha256"],
        notes={"source_size_wh": list(source_wh), "gt_consumed": False},
    )
    evidence_offsets = [0]
    evidence_points: list[np.ndarray] = []
    evidence_uv: list[np.ndarray] = []
    evidence_weights: list[np.ndarray] = []
    evidence_intrinsics: list[np.ndarray] = []
    evidence_camera: list[np.ndarray] = []
    for row in edges:
        key = (int(row["source_frame_idx"]), int(row["target_frame_idx"]))
        evidence = edge_reprojection_evidence.get(key)
        if evidence is None:
            evidence_points.append(np.empty((0, 3), dtype=np.float64))
            evidence_uv.append(np.empty((0, 2), dtype=np.float64))
            evidence_weights.append(np.empty((0,), dtype=np.float64))
            evidence_intrinsics.append(np.zeros((4,), dtype=np.float64))
            evidence_camera.append(np.zeros((4, 4), dtype=np.float64))
            evidence_offsets.append(evidence_offsets[-1])
        else:
            evidence_points.append(evidence["canonical_points"])
            evidence_uv.append(evidence["target_uv"])
            evidence_weights.append(evidence["weights"])
            evidence_intrinsics.append(evidence["target_intrinsics"])
            evidence_camera.append(evidence["target_camera"])
            evidence_offsets.append(evidence_offsets[-1] + len(evidence["weights"]))
    flat_points = np.concatenate(evidence_points, axis=0) if evidence_points else np.empty((0, 3), dtype=np.float64)
    flat_uv = np.concatenate(evidence_uv, axis=0) if evidence_uv else np.empty((0, 2), dtype=np.float64)
    flat_weights = np.concatenate(evidence_weights, axis=0) if evidence_weights else np.empty((0,), dtype=np.float64)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        metadata=np.asarray([json.dumps(metadata)]),
        source_frame_idx=np.asarray([row["source_frame_idx"] for row in edges], dtype=np.int64),
        target_frame_idx=np.asarray([row["target_frame_idx"] for row in edges], dtype=np.int64),
        rotation_source_to_target_world=np.asarray([edge_motions.get((row["source_frame_idx"], row["target_frame_idx"]), (np.eye(3), np.zeros(3)))[0] for row in edges]),
        translation_source_to_target_world_m=np.asarray([edge_motions.get((row["source_frame_idx"], row["target_frame_idx"]), (np.eye(3), np.zeros(3)))[1] for row in edges]),
        accepted=np.asarray([row.get("status") == "accepted" for row in edges], dtype=bool),
        source_size=np.asarray(source_wh, dtype=np.int64),
        reprojection_evidence_offsets=np.asarray(evidence_offsets, dtype=np.int64),
        reprojection_canonical_points=flat_points,
        reprojection_target_uv=flat_uv,
        reprojection_weights=flat_weights,
        reprojection_target_intrinsics=np.asarray(evidence_intrinsics, dtype=np.float64),
        reprojection_target_camera=np.asarray(evidence_camera, dtype=np.float64),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.output_pose_report is not None:
        chain_report = dict(initial_report)
        chain_report.update({"status": "v20_prediction_lightglue_rgb_depth_chain_diagnostic", "annotation_ready": False, "diagnostic_only": True, "generated_geometry_pose_evidence_consumed": False, "gt_used_as_solver_input": False, "pose_rows": chain_rows, "feature_edge_report": str(args.output_json.expanduser().resolve())})
        append_prediction_stage(
            chain_report,
            stage="prediction_rgb_depth_chain_builder",
            input_hashes=metadata["input_sha256"],
            notes={"source_size_wh": list(source_wh), "gt_consumed": False},
        )
        args.output_pose_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_pose_report.write_text(json.dumps(chain_report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "edge_count": len(edges), "accepted_count": report["accepted_count"], "rejected_count": report["rejected_count"], "keyframes": key_ids, "chain_covered_frame_count": len(chain), "output_json": str(args.output_json), "output_pose_report": str(args.output_pose_report) if args.output_pose_report else None}, indent=2))


if __name__ == "__main__":
    main()
