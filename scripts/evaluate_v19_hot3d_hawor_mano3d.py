#!/usr/bin/env python3
"""Evaluate HaWoR MANO predictions against HOT3D MANO annotations in camera 3D.

This evaluator is intentionally narrower than a full HOT3D physical benchmark:

- It consumes HOT3D hand GT only as evaluation data, never as V19 perception.
- It replays HOT3D MANO using the documented SMPLX convention
  (global_orient=wrist_xform[:3], transl=wrist_xform[3:], 15 PCA pose coeffs,
  side-specific MANO models, left shapedirs-x fix).
- It transforms both HOT3D GT and HaWoR predictions into their respective camera
  frames before comparing. This avoids falsely scoring unrelated HOT3D-world and
  HaWoR-SLAM-world origins.
- It reports absolute wrist/joint errors separately from root-aligned errors so
  global metric placement errors are not conflated with articulation errors.

The supported claim family is 3D hand/MANO localization on adapted HOT3D RGB
clips. This is still not a contact, occlusion, nonpenetration, or object-pose
metric.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

MANO_FINGERTIP_VERT_INDICES = [744, 320, 443, 554, 671]
# HaWoR's infiller/hand_utils/mano_wrapper.py appends MANO fingertips and then
# reorders joints with this mapping before export. HOT3D GT MANO must be mapped
# the same way before comparing to HaWoR's saved *_joints_world_m arrays.
HAWOR_MANO_TO_OPENPOSE_MAPPING = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def patch_legacy_numpy() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in [
        ("bool", np.bool_),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("unicode", str),
        ("str", str),
    ]:
        if not hasattr(np, name):
            setattr(np, name, value)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def quat_wxyz_to_matrix(q: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0:
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


def se3_from_hot3d_dict(d: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return quat_wxyz_to_matrix(d["quaternion_wxyz"]), np.asarray(d["translation_xyz"], dtype=np.float64)


def world_to_camera(points_world: np.ndarray, R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return (np.asarray(points_world, dtype=np.float64) - np.asarray(t_c2w, dtype=np.float64)[None, :]) @ np.asarray(R_c2w, dtype=np.float64)


def load_hand_shape(gt: dict[str, Any], sidecar_path: Path) -> tuple[np.ndarray, str]:
    if isinstance(gt.get("hand_shapes"), dict) and "mano" in gt["hand_shapes"]:
        return np.asarray(gt["hand_shapes"]["mano"], dtype=np.float32), "sidecar:hand_shapes"
    sibling = sidecar_path.parent / "hot3d_hand_shapes.json"
    if sibling.exists():
        shape = load_json(sibling)
        if "mano" not in shape:
            raise RuntimeError(f"{sibling} lacks mano hand shape")
        return np.asarray(shape["mano"], dtype=np.float32), str(sibling)
    tar_path = Path(str(gt.get("source_tar", ""))).expanduser()
    if tar_path.exists():
        with tarfile.open(tar_path, "r") as tar:
            if "__hand_shapes.json__" not in tar.getnames():
                raise RuntimeError(f"{tar_path} lacks __hand_shapes.json__")
            shape = json.load(tar.extractfile("__hand_shapes.json__"))  # type: ignore[arg-type]
        return np.asarray(shape["mano"], dtype=np.float32), str(tar_path)
    raise RuntimeError("HOT3D hand shape missing: sidecar has no hand_shapes, no sibling hot3d_hand_shapes.json, and source_tar is unavailable")


def load_smplx_mano(args: argparse.Namespace) -> dict[str, Any]:
    patch_legacy_numpy()
    try:
        import smplx  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "smplx is required to replay HOT3D MANO GT. Run this script in the HaWoR/remote environment or install smplx."
        ) from exc
    left_path = args.mano_left.expanduser()
    right_path = args.mano_right.expanduser()
    if not left_path.exists():
        raise FileNotFoundError(f"missing MANO_LEFT model: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"missing MANO_RIGHT model: {right_path}")
    left = smplx.create(str(left_path), "mano", use_pca=True, is_rhand=False, num_pca_comps=15)
    right = smplx.create(str(right_path), "mano", use_pca=True, is_rhand=True, num_pca_comps=15)
    # Same fix used by hand_tracking_toolkit and HaWoR for MANO_LEFT shapedirs.
    if torch.sum(torch.abs(left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])) < 1:
        left.shapedirs[:, 0, :] *= -1
    return {"left": left, "right": right}


def replay_hot3d_mano(
    layer: Any,
    beta: np.ndarray,
    theta: np.ndarray,
    wrist_xform: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta_t = torch.as_tensor(beta[None, :], dtype=torch.float32, device=device)
    theta_t = torch.as_tensor(theta[None, :], dtype=torch.float32, device=device)
    wrist_t = torch.as_tensor(wrist_xform[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = layer(
            betas=beta_t,
            global_orient=wrist_t[:, :3],
            hand_pose=theta_t,
            transl=wrist_t[:, 3:],
            return_verts=True,
        )
    vertices = out.vertices.detach().cpu().numpy()[0].astype(np.float64)
    joints = out.joints.detach().cpu().numpy()[0].astype(np.float64)
    if joints.shape[0] != 21:
        tips = vertices[np.asarray(MANO_FINGERTIP_VERT_INDICES, dtype=int)]
        joints = np.concatenate([joints, tips], axis=0)
    if joints.shape[0] != 21:
        raise RuntimeError(f"expected 21 GT MANO joints after fingertip append, got {joints.shape}")
    faces = np.asarray(layer.faces, dtype=np.int32)
    return vertices, joints, faces


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def project_pinhole(points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    z = points_cam[:, 2]
    out = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float64)
    valid = z > 1.0e-6
    out[valid, 0] = points_cam[valid, 0] / z[valid] * fx + cx
    out[valid, 1] = points_cam[valid, 1] / z[valid] * fy + cy
    return out


def get_k_for_frame(gt_row: dict[str, Any], stream_id: str) -> tuple[float, float, float, float]:
    params = gt_row["json"]["cameras.json"][stream_id]["calibration"]["projection_params"]
    f, cx, cy = [float(v) for v in params[:3]]
    return f, f, cx, cy


def per_joint_errors(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float, float, float]:
    e = np.linalg.norm(pred - gt, axis=1)
    wrist = float(e[0])
    mpjpe = float(np.mean(e))
    med = float(np.median(e))
    pred_rel = pred - pred[0:1]
    gt_rel = gt - gt[0:1]
    er = np.linalg.norm(pred_rel - gt_rel, axis=1)
    return wrist, mpjpe, med, float(np.mean(er)), float(np.median(er)), float(np.percentile(er, 95))


def draw_points(image: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    h, w = image.shape[:2]
    for p in pts:
        if not np.isfinite(p).all():
            continue
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        if -20 <= x < w + 20 and -20 <= y < h + 20:
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def render_review(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    gt_frames: dict[int, dict[str, Any]],
    pred_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
) -> str | None:
    if args.review_output is None or args.image_manifest is None:
        return None
    manifest = load_json(args.image_manifest)
    frame_paths = {int(r["frame_idx"]): Path(r["raw_frame_path"]) for r in manifest.get("frames", []) if isinstance(r, dict) and r.get("raw_frame_path")}
    by_frame_side = {(int(r["frame_idx"]), str(r["side"])): r for r in rows if r.get("matched")}
    tiles: list[np.ndarray] = []
    for frame_idx in args.review_frames:
        path = frame_paths.get(int(frame_idx))
        gt_row = gt_frames.get(int(frame_idx))
        if path is None or gt_row is None or not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        vis = image.copy()
        fx, fy, cx, cy = get_k_for_frame(gt_row, args.stream_id)
        lines = [f"frame {frame_idx:06d}  GT green, pred yellow/cyan"]
        for side, color in [("left", (0, 255, 255)), ("right", (255, 255, 0))]:
            row = by_frame_side.get((int(frame_idx), side))
            key = (int(frame_idx), side)
            if row is None or key not in pred_cache:
                lines.append(f"{side}: no matched 3D row")
                continue
            gt_cam, pred_cam = pred_cache[key]
            gt_uv = project_pinhole(gt_cam, fx, fy, cx, cy)
            pred_uv = project_pinhole(pred_cam, fx, fy, cx, cy)
            draw_points(vis, gt_uv, (0, 255, 0), 4)
            draw_points(vis, pred_uv, color, 3)
            lines.append(
                f"{side}: wrist {row['wrist_error_m']*1000:.1f}mm, MPJPE {row['joint_mpjpe_m']*1000:.1f}mm, root-align {row['root_aligned_mpjpe_m']*1000:.1f}mm"
            )
        banner = np.zeros((74, vis.shape[1], 3), dtype=np.uint8)
        banner[:] = (8, 8, 8)
        for j, text in enumerate(lines[:3]):
            cv2.putText(banner, text, (12, 23 + 23 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.vstack([banner, vis]))
    if not tiles:
        return None
    width = min(1600, max(t.shape[1] for t in tiles))
    resized = []
    for t in tiles:
        if t.shape[1] != width:
            scale = width / t.shape[1]
            t = cv2.resize(t, (width, int(round(t.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        resized.append(t)
    sheet = np.vstack(resized)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.review_output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"failed to write {args.review_output}")
    return str(args.review_output)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gt = load_json(args.hot3d_gt)
    frames_list = gt.get("frames")
    if not isinstance(frames_list, list) or not frames_list:
        raise RuntimeError("HOT3D GT sidecar has no frames")
    gt_frames = {int(row["frame_idx"]): row for row in frames_list if isinstance(row, dict)}
    beta, hand_shape_source = load_hand_shape(gt, args.hot3d_gt)
    npz = np.load(args.hawor_npz, allow_pickle=True)
    layers = load_smplx_mano(args)
    device = torch.device(args.device)
    for layer in layers.values():
        layer.to(device)
        layer.eval()

    frame_idx = np.asarray(npz["frame_idx"], dtype=int) if "frame_idx" in npz.files else np.arange(len(frames_list), dtype=int)
    rows: list[dict[str, Any]] = []
    pred_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for local_i, frame in enumerate(frame_idx.tolist()):
        gt_row = gt_frames.get(int(frame))
        if gt_row is None:
            continue
        hot3d_cam = gt_row.get("json", {}).get("cameras.json", {}).get(args.stream_id)
        if not isinstance(hot3d_cam, dict):
            continue
        R_hot3d_c2w, t_hot3d_c2w = se3_from_hot3d_dict(hot3d_cam["T_world_from_camera"])
        R_pred_c2w = np.asarray(npz["R_c2w"][local_i], dtype=np.float64)
        t_pred_c2w = np.asarray(npz["t_c2w"][local_i], dtype=np.float64)
        for side in ("left", "right"):
            hand = gt_row.get("json", {}).get("hands.json", {}).get(side)
            if not isinstance(hand, dict) or "mano_pose" not in hand:
                rows.append({"frame_idx": int(frame), "side": side, "measurable": False, "matched": False, "reason": "missing_hot3d_mano_pose"})
                continue
            pred_valid = bool(npz[f"{side}_valid"][local_i]) if f"{side}_valid" in npz.files else True
            if not pred_valid:
                rows.append({"frame_idx": int(frame), "side": side, "measurable": True, "matched": False, "reason": "missing_hawor_prediction"})
                continue
            mano = hand["mano_pose"]
            theta = np.asarray(mano["thetas"], dtype=np.float32)
            wrist_xform = np.asarray(mano["wrist_xform"], dtype=np.float32)
            gt_verts_w, gt_joints_raw_w, _ = replay_hot3d_mano(layers[side], beta, theta, wrist_xform, device)
            pred_joints_w = np.asarray(npz[f"{side}_joints_world_m"][local_i], dtype=np.float64)
            pred_verts_w = np.asarray(npz[f"{side}_vertices_world_m"][local_i], dtype=np.float64)
            gt_joints_raw_cam = world_to_camera(gt_joints_raw_w, R_hot3d_c2w, t_hot3d_c2w)
            gt_joints_hawor_order_cam = gt_joints_raw_cam[np.asarray(HAWOR_MANO_TO_OPENPOSE_MAPPING, dtype=int)]
            pred_joints_cam = world_to_camera(pred_joints_w, R_pred_c2w, t_pred_c2w)
            pred_verts_cam = world_to_camera(pred_verts_w, R_pred_c2w, t_pred_c2w)
            wrist, mpjpe, med, rel_mpjpe, rel_med, rel_p95 = per_joint_errors(pred_joints_cam, gt_joints_hawor_order_cam)
            vertex_centroid_err = float(np.linalg.norm(np.mean(pred_verts_cam, axis=0) - np.mean(world_to_camera(gt_verts_w, R_hot3d_c2w, t_hot3d_c2w), axis=0)))
            row = {
                "frame_idx": int(frame),
                "side": side,
                "measurable": True,
                "matched": True,
                "gt_visibility_modeled": hand.get("visibilities_modeled", {}).get(args.stream_id) if isinstance(hand.get("visibilities_modeled"), dict) else None,
                "pred_detected_same_frame": bool(npz[f"{side}_detected_same_frame"][local_i]) if f"{side}_detected_same_frame" in npz.files else None,
                "wrist_error_m": wrist,
                "joint_mpjpe_m": mpjpe,
                "joint_median_error_m": med,
                "root_aligned_mpjpe_m": rel_mpjpe,
                "root_aligned_median_error_m": rel_med,
                "root_aligned_p95_error_m": rel_p95,
                "vertex_centroid_error_m": vertex_centroid_err,
                "joint_order": "HaWoR 21-joint export order: MANO joints plus fingertips reordered by [0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]",
            }
            rows.append(row)
            if int(frame) in args.review_frames:
                pred_cache[(int(frame), side)] = (gt_joints_hawor_order_cam, pred_joints_cam)
    matched = [r for r in rows if r.get("matched")]
    measurable = [r for r in rows if r.get("measurable")]
    by_side: dict[str, Any] = {}
    for side in ("left", "right"):
        sm = [r for r in matched if r["side"] == side]
        sr = [r for r in measurable if r["side"] == side]
        by_side[side] = {
            "measurable_rows": len(sr),
            "matched_rows": len(sm),
            "match_rate": float(len(sm) / max(1, len(sr))),
            "wrist_error_m": summarize([float(r["wrist_error_m"]) for r in sm]),
            "joint_mpjpe_m": summarize([float(r["joint_mpjpe_m"]) for r in sm]),
            "root_aligned_mpjpe_m": summarize([float(r["root_aligned_mpjpe_m"]) for r in sm]),
            "vertex_centroid_error_m": summarize([float(r["vertex_centroid_error_m"]) for r in sm]),
        }
    review = render_review(args, rows, gt_frames, pred_cache)
    report = {
        "status": "ok",
        "method": "evaluate_v19_hot3d_hawor_mano3d",
        "claim_scope": "3D MANO hand localization/articulation against HOT3D MANO in camera coordinates; not contact, occlusion, nonpenetration, or object-pose scoring",
        "hot3d_gt": str(args.hot3d_gt),
        "hawor_npz": str(args.hawor_npz),
        "stream_id": args.stream_id,
        "mano_left": str(args.mano_left),
        "mano_right": str(args.mano_right),
        "hand_shape_source": hand_shape_source,
        "review": review,
        "summary": {
            "measurable_rows": len(measurable),
            "matched_rows": len(matched),
            "match_rate": float(len(matched) / max(1, len(measurable))),
            "wrist_error_m": summarize([float(r["wrist_error_m"]) for r in matched]),
            "joint_mpjpe_m": summarize([float(r["joint_mpjpe_m"]) for r in matched]),
            "joint_median_error_m": summarize([float(r["joint_median_error_m"]) for r in matched]),
            "root_aligned_mpjpe_m": summarize([float(r["root_aligned_mpjpe_m"]) for r in matched]),
            "root_aligned_median_error_m": summarize([float(r["root_aligned_median_error_m"]) for r in matched]),
            "vertex_centroid_error_m": summarize([float(r["vertex_centroid_error_m"]) for r in matched]),
            "by_side": by_side,
        },
        "rows": rows,
    }
    write_json(args.output_report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2)[:20000])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hot3d-gt", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument("--image-manifest", type=Path, default=None)
    parser.add_argument("--review-output", type=Path, default=None)
    parser.add_argument("--review-frames", type=int, nargs="*", default=[0, 50, 100, 149])
    parser.add_argument("--mano-left", type=Path, default=Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_LEFT.pkl"))
    parser.add_argument("--mano-right", type=Path, default=Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_RIGHT.pkl"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
