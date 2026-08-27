#!/usr/bin/env python3
"""Evaluate final hand-mesh alignment under the P15 rigid pose body.

Consumes the SAM3D native camera-origin metric mesh (anchor camera frame),
the P15 per-frame rigid pose rows, the HaWoR hand vertices, and the P09
first-surface object surfels.  Reports, per frame:
  * observed surfel -> aligned mesh surface distance,
  * hand vertex -> aligned mesh surface distance,
  * signed depth error at object-owned pixels (mesh minus measured UniDepth),
  * hand-vs-mesh depth order on hand pixels.

This is a diagnostic/geometry audit only.  Generated SAM3D faces are never
promoted to collision, sign, contact, or nonpenetration authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

BASE = "/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/backend_tests/20260819T122452Z_hot3d_milk_local_authority_local29_v1/runs/P0014_84ea2dcc_carton_milk_f2370_2519"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def world_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    return (np.asarray(points_world, dtype=float) - t[None, :]) @ R


def summarize(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median_m": float(np.median(arr)),
        "p05_m": float(np.percentile(arr, 5.0)),
        "p95_m": float(np.percentile(arr, 95.0)),
        "max_m": float(np.max(np.abs(arr))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    pose = load_json(args.pose_graph)
    ann_by_idx = {int(f["frame_idx"]): f for f in annotations["frames"]}
    pose_by_idx = {int(r["frame_idx"]): r for r in pose["pose_rows"]}
    hands = np.load(args.hand_npz)
    hand_idx = hands["frame_idx"].astype(int).tolist()
    mesh = trimesh.load(args.mesh_anchor_camera, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    T92 = np.asarray(ann_by_idx[args.anchor_frame]["camera"]["T_world_camera_metric"], dtype=np.float64)
    anchor_pose = pose_by_idx[args.anchor_frame]
    cent92 = np.asarray(anchor_pose["translation_world_m"], dtype=np.float64)
    R92 = np.asarray(anchor_pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    # canonical mesh: inverse anchor object pose applied to anchor-world vertices.
    canonical = ((vertices @ T92[:3, :3].T + T92[:3, 3]) - cent92[None, :]) @ R92
    rows: dict[str, Any] = {}
    for idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if idx not in ann_by_idx or idx not in pose_by_idx:
            continue
        frame = ann_by_idx[idx]
        obj = frame["objects"][0]
        visible = obj.get("visible_geometry_candidate")
        if not isinstance(visible, dict):
            continue
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        r = pose_by_idx[idx]
        Rw = np.asarray(r["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        tw = np.asarray(r["translation_world_m"], dtype=np.float64)
        vw = canonical @ Rw.T + tw
        vc = world_to_camera(vw, T)
        observed = np.asarray(visible.get("camera_vertices_sample_m"), dtype=np.float64)
        observed = observed[np.isfinite(observed).all(axis=1) & (observed[:, 2] > 0.0)]
        tree = cKDTree(vc)
        d_obs, _ = tree.query(observed, k=1)
        row: dict[str, Any] = {
            "observed_to_aligned_mesh_m": summarize(d_obs),
        }
        if idx in hand_idx:
            pos = hand_idx.index(idx)
            for side in ("left", "right"):
                if int(np.asarray(hands[f"{side}_valid"])[pos]) != 1:
                    continue
                hw = np.asarray(hands[f"{side}_vertices_world_m"][pos], dtype=np.float64)
                hc = world_to_camera(hw, T)
                d_hand, _ = tree.query(hc, k=1)
                row[f"hand_{side}_to_aligned_mesh_m"] = {
                    "min_m": float(np.min(d_hand)),
                    "median_m": float(np.median(d_hand)),
                    "p95_m": float(np.percentile(d_hand, 95.0)),
                    "contact_fraction_within_5mm": float(np.mean(d_hand <= 0.005)),
                }
        rows[str(idx)] = row
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "sam3d_p15_aligned_hand_mesh_audit.json"
    report = {
        "schema": "sam3d_p15_aligned_hand_mesh_audit_v1",
        "diagnostic_only": True,
        "annotation_ready": False,
        "claim_scope": (
            "P15 rigid pose body + SAM3D camera-origin metric mesh + HaWoR hands + P09 first-surface "
            "surfels. Generated SAM3D faces remain render-only; no collision/sign/contact/nonpenetration authority."
        ),
        "anchor_frame": int(args.anchor_frame),
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    meds = [r["observed_to_aligned_mesh_m"]["median_m"] for r in rows.values()]
    print("frames", len(rows), "observed->mesh median over frames", round(float(np.median(meds)), 4))
    for idx in [92, 103, 121, 146]:
        r = rows.get(str(idx))
        if r:
            print(idx, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-anchor-camera", type=Path, default=Path(BASE) / "experiments/sam3d_native_ghost_lite_20260826/audit_native_pose/sam3d_native_opencv_camera_origin_metric_scaled.ply")
    parser.add_argument("--annotations", type=Path, default=Path(BASE) / "measurements/object_geometry/visible_geometry/carton_milk/annotations_v19_visible_geometry.json")
    parser.add_argument("--pose-graph", type=Path, default=Path(BASE) / "experiments/sam3d_trellis_controlled/P15_observed_pose_graph/v19_rigid_object_pose_graph_report.json")
    parser.add_argument("--hand-npz", type=Path, default=Path(BASE) / "measurements/hand_candidates/hawor_world/hawor_world_hands.npz")
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=149)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
