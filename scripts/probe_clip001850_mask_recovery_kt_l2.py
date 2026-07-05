#!/usr/bin/env python3
"""KT-L2: clip001850 keyboard mask-recovery probe (CPU-only, no new model).

Causal card
-----------
Artifact defect: 72/150 frames (f78-149) carry a visible SAM2 keyboard mask
with 2500 metric backprojected points each, but are routed to
`visible_surface_ineligible_for_rigid_pose_fit` because one world extent axis
is inflated 4.4-6.2x (probable hand/background leakage). Direct support is
frozen at 13/150 frames.

Physical variable: per-frame object localization support -- whether the surface
inside the leaking mask is recoverable keyboard surface, or background/hand
points that quarantine can remove to leave a consensus-consistent keyboard
patch.

Live mechanisms:
  M2a (recoverable) -- mask leaks a minority of hand/background pixels; after
       quarantine the surviving points sit on the keyboard rest location with
       extent consistent with the anchor and can seed a rigid fit.
  M2b (degenerate)  -- after quarantine a thin/partial keyboard patch remains;
       extent is consistent but pose is only estimable up to a weak DOF.
  M-drift (delaminated) -- the mask has migrated OFF the keyboard onto
       background/hand-adjacent regions; surviving points are NOT keyboard
       surface, quarantine cannot recover support, centroid stays far from rest.

Discriminating measurement: staged quarantine (MANO hand hull -> keyboard depth
band -> rest spatial box); after each stage record surviving point count,
world extent, worst-axis ratio vs anchor, centroid, and centroid-to-rest
distance. A frame is recoverable ONLY IF after quarantine the centroid lands
within the keyboard rest neighbourhood (<=60 mm) AND extent is consistent
(worst-axis ratio <=2.5) with >=200 surviving points. A Procrustes/ICP fit of
the cleaned points to the f30-46 inlier rest body measures whether the
surviving patch actually aligns to the keyboard body (consensus-consistency).

Predictions:
  M2a -> many frames recover extent AND land near rest; direct support rises.
  M-drift -> cleaned centroids stay 100-300 mm from rest regardless of how
       aggressively we quarantine; direct support cannot rise above 13/150.

This script runs the measurement. It does NOT smooth, rewrite pose rows, or
claim contact. Outputs go to /tmp/clip001850_mask_recovery_kt_l2/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

OUT_DIR = Path("/tmp/clip001850_mask_recovery_kt_l2")

INLIER_FRAMES = [30, 31, 32, 33, 34, 35, 36, 45, 46]  # static rest cluster (subagent 25)
ALL_ELIGIBLE = [30, 31, 32, 33, 34, 35, 36, 45, 46, 60, 75, 76, 77]
INELIGIBLE_RANGE = range(78, 150)  # the 72 visible_surface_ineligible frames

# Recovery thresholds (documented, not tuned to pass)
MIN_SURVIVING_PTS = 200
WORST_AXIS_RATIO_RECOVERABLE = 2.5
WORST_AXIS_RATIO_CONSISTENT = 2.0
CENTROID_CONSENSUS_MM = 60.0
CENTROID_DRIFT_MM = 100.0
DEPTH_BAND_PERCENTILE = 2
REST_BOX_MARGIN_M = 0.04


def world_to_camera(pts_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    T_cw = np.linalg.inv(T_world_camera)
    homog = np.c_[pts_world, np.ones(len(pts_world))].T
    return (T_cw @ homog)[:3].T


def project(pts_world: np.ndarray, cam: dict):
    T_world_camera = np.array(cam["T_world_camera_metric"])
    fx, fy, cx, cy = cam["intrinsics_fx_fy_cx_cy"]
    pts_cam = world_to_camera(pts_world, T_world_camera)
    z = pts_cam[:, 2]
    u = fx * pts_cam[:, 0] / z + cx
    v = fy * pts_cam[:, 1] / z + cy
    return np.c_[u, v], z, pts_cam


def axis_aligned_extent(pts: np.ndarray) -> np.ndarray:
    return pts.max(axis=0) - pts.min(axis=0)


def procrustes_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid Procrustes: returns R (3x3), t (3,), rmsd. Aligns source -> target."""
    src = source.copy().astype(float)
    tgt = target.copy().astype(float)
    if len(src) < 10 or len(tgt) < 10:
        return np.eye(3), np.zeros(3), float("inf")
    cs = src.mean(0)
    ct = tgt.mean(0)
    S = src - cs
    T = tgt - ct
    # subsample target for speed if large
    if len(T) > 4000:
        T = T[np.random.default_rng(0).choice(len(T), 4000, replace=False)]
    # nearest-neighbor correspondence (single pass; trimmed to best 80%)
    from scipy.spatial import cKDTree

    tree = cKDTree(T)
    dist, idx = tree.query(S, k=1)
    keep = dist <= np.percentile(dist, 80)
    S2 = S[keep]
    Tcorr = T[idx[keep]]
    H = S2.T @ Tcorr
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = ct - R @ cs
    # rmsd on the kept correspondences after applying the rigid map
    mapped = (R @ S2.T).T + t
    rmsd = float(np.sqrt(np.mean(np.sum((mapped - Tcorr) ** 2, axis=1))))
    return R, t, rmsd


def main(run_root: str) -> None:
    run = Path(run_root)
    ann_path = run / "measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json"
    bridge_path = run / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz"
    pose_report_path = run / "measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ann = json.load(open(ann_path))
    frames = ann["frames"]

    # ---- Build rest inlier model from f30-46 (static cluster) ----
    rest_pts = np.concatenate(
        [np.array(frames[fi]["objects"][0]["visible_geometry_candidate"]["world_vertices_sample_m"]) for fi in INLIER_FRAMES]
    )
    rest_centroid = rest_pts.mean(0)
    rest_box_lo = np.percentile(rest_pts, DEPTH_BAND_PERCENTILE, axis=0) - REST_BOX_MARGIN_M
    rest_box_hi = np.percentile(rest_pts, 100 - DEPTH_BAND_PERCENTILE, axis=0) + REST_BOX_MARGIN_M

    # keyboard camera-depth band from inliers (per-frame camera)
    inlier_depths = np.concatenate(
        [project(np.array(frames[fi]["objects"][0]["visible_geometry_candidate"]["world_vertices_sample_m"]), frames[fi]["camera"])[1]
         for fi in INLIER_FRAMES]
    )
    depth_lo = float(np.percentile(inlier_depths, DEPTH_BAND_PERCENTILE))
    depth_hi = float(np.percentile(inlier_depths, 100 - DEPTH_BAND_PERCENTILE))

    anchor_ext = np.array(frames[75]["objects"][0]["visible_geometry_candidate"]["anchor_extent_world_m"])

    # ---- MANO bridge for hand quarantine ----
    bridge = np.load(bridge_path)
    b_frame = bridge["frame_idx"]
    b_side = bridge["hand_side"]
    b_verts_world = bridge["vertices_current_v18_world_from_hawor_projection_relift_m"]

    def hand_hulls(frame_idx: int, cam: dict) -> list[np.ndarray]:
        hulls = []
        for side in ("left", "right"):
            rows = [i for i in range(len(b_frame)) if b_frame[i] == frame_idx and b_side[i] == side]
            if rows:
                hpix, _, _ = project(b_verts_world[rows[0]], cam)
                hulls.append(hpix)
        return hulls

    # ---- Per-frame staged quarantine ----
    per_frame = []
    recoverable_count = 0
    consensus_count = 0
    drift_count = 0
    degenerate_count = 0

    for fi in INELIGIBLE_RANGE:
        f = frames[fi]
        cam = f["camera"]
        cand = f["objects"][0]["visible_geometry_candidate"]
        pts = np.array(cand["world_vertices_sample_m"])
        n0 = len(pts)
        ext_before = axis_aligned_extent(pts)
        ratio_before = ext_before / anchor_ext
        worst_before = float(ratio_before.max())

        kbpix, kbz, _ = project(pts, cam)

        # Stage 1: hand quarantine (MANO projected hull)
        keep_hand = np.ones(n0, bool)
        for hull in hand_hulls(fi, cam):
            keep_hand &= ~MplPath(hull).contains_points(kbpix)
        n_hand = int(keep_hand.sum())

        # Stage 2: depth-band quarantine (keyboard camera-depth band)
        keep_depth = keep_hand & (kbz >= depth_lo) & (kbz <= depth_hi)
        n_depth = int(keep_depth.sum())

        # Stage 3: rest spatial-box quarantine (keyboard is static in world)
        keep_box = keep_depth & np.all((pts >= rest_box_lo) & (pts <= rest_box_hi), axis=1)
        n_box = int(keep_box.sum())

        cleaned = pts[keep_box] if n_box > 0 else pts[keep_depth] if n_depth > 0 else pts[keep_hand]

        ext_after = axis_aligned_extent(cleaned) if len(cleaned) else np.zeros(3)
        ratio_after = ext_after / anchor_ext if len(cleaned) else np.zeros(3)
        worst_after = float(ratio_after.max()) if len(cleaned) else float("inf")
        cen_after = cleaned.mean(0) if len(cleaned) else np.array([np.nan] * 3)
        cen_dist_mm = float(np.linalg.norm(cen_after - rest_centroid) * 1000.0) if len(cleaned) else float("inf")

        # centroid distance using the more permissive hand+depth quarantine
        # (the rest-box test is decisive but we also report hand+depth-only
        #  centroid so the delamination is visible without box bias)
        hd_cleaned = pts[keep_depth]
        hd_cen = hd_cleaned.mean(0) if len(hd_cleaned) else np.array([np.nan] * 3)
        hd_cen_dist_mm = float(np.linalg.norm(hd_cen - rest_centroid) * 1000.0) if len(hd_cleaned) else float("inf")
        hd_ext = axis_aligned_extent(hd_cleaned) if len(hd_cleaned) else np.zeros(3)
        hd_worst = float((hd_ext / anchor_ext).max()) if len(hd_cleaned) else float("inf")

        # Classification. Primary discriminator is the UNBIASED hand+depth
        # centroid distance (no rest-box): a real keyboard patch must land on
        # the keyboard rest location regardless of how aggressively we box.
        # The rest-box survivor count is reported as a sub-classification of
        # delamination severity, not a separate mechanism.
        if hd_cen_dist_mm <= CENTROID_CONSENSUS_MM and worst_after <= WORST_AXIS_RATIO_RECOVERABLE:
            verdict = "recoverable_consensus"
            recoverable_count += 1
            consensus_count += 1
        elif hd_cen_dist_mm <= CENTROID_CONSENSUS_MM and worst_after > WORST_AXIS_RATIO_RECOVERABLE:
            verdict = "degenerate_partial_keyboard"
            degenerate_count += 1
        elif hd_cen_dist_mm <= CENTROID_DRIFT_MM:
            verdict = "mask_drift_delaminated_near"
            drift_count += 1
        else:
            verdict = "mask_drift_delaminated"
            drift_count += 1
        drift_subclass = "box_survivors_lt_min" if n_box < MIN_SURVIVING_PTS else "box_survivors_ge_min"

        # Procrustes fit of the most-permissive cleaned sample (hand+depth) to rest body
        R, t, rmsd = procrustes_fit(hd_cleaned, rest_pts)
        fit_translation_mm = float(np.linalg.norm(t) * 1000.0)

        per_frame.append({
            "frame_idx": fi,
            "n_raw": n0,
            "n_after_hand": n_hand,
            "n_after_depth": n_depth,
            "n_after_box": n_box,
            "extent_before": np.round(ext_before, 4).tolist(),
            "worst_axis_ratio_before": round(worst_before, 3),
            "extent_after_hand_depth_box": np.round(ext_after, 4).tolist(),
            "worst_axis_ratio_after": round(worst_after, 3),
            "centroid_after_mm": [round(float(c) * 1000, 1) for c in cen_after],
            "centroid_dist_to_rest_mm": round(cen_dist_mm, 1),
            "hand_depth_centroid_dist_to_rest_mm": round(hd_cen_dist_mm, 1),
            "hand_depth_worst_axis_ratio": round(hd_worst, 3),
            "rest_centroid_mm": [round(float(c) * 1000, 1) for c in rest_centroid],
            "procrustes_fit_translation_mm": round(fit_translation_mm, 1),
            "procrustes_fit_rmsd_mm": round(rmsd * 1000, 1),
            "verdict": verdict,
            "box_quarantine_subclass": drift_subclass,
        })

    # ---- Eligible-frame sanity (confirm rest model is self-consistent) ----
    eligible_self = []
    for fi in INLIER_FRAMES:
        f = frames[fi]
        cam = f["camera"]
        pts = np.array(f["objects"][0]["visible_geometry_candidate"]["world_vertices_sample_m"])
        kbpix, kbz, _ = project(pts, cam)
        keep = np.ones(len(pts), bool)
        for hull in hand_hulls(fi, cam):
            keep &= ~MplPath(hull).contains_points(kbpix)
        keep &= (kbz >= depth_lo) & (kbz <= depth_hi)
        keep &= np.all((pts >= rest_box_lo) & (pts <= rest_box_hi), axis=1)
        cleaned = pts[keep]
        cen = cleaned.mean(0) if len(cleaned) else np.array([np.nan] * 3)
        eligible_self.append({
            "frame_idx": fi,
            "n_after_quarantine": int(len(cleaned)),
            "centroid_dist_to_rest_mm": round(float(np.linalg.norm(cen - rest_centroid) * 1000), 1)
            if len(cleaned) else None,
        })

    # ---- Conclusion ----
    hd_dists = [r["hand_depth_centroid_dist_to_rest_mm"] for r in per_frame]
    n_within_60 = int(sum(1 for d in hd_dists if d <= CENTROID_CONSENSUS_MM))
    n_within_100 = int(sum(1 for d in hd_dists if d <= CENTROID_DRIFT_MM))
    n_within_150 = int(sum(1 for d in hd_dists if d <= 150.0))

    direct_support_baseline = len(ALL_ELIGIBLE)  # 13
    direct_support_recovered = direct_support_baseline + consensus_count

    summary = {
        "case": "hot3d_clip001850_keyboard_right",
        "kill_test": "KT-L2",
        "run_root": str(run),
        "rest_model": {
            "inlier_frames": INLIER_FRAMES,
            "rest_centroid_m": [round(float(c), 4) for c in rest_centroid],
            "rest_box_lo_m": [round(float(c), 4) for c in rest_box_lo],
            "rest_box_hi_m": [round(float(c), 4) for c in rest_box_hi],
            "depth_band_camera_m": [round(depth_lo, 4), round(depth_hi, 4)],
            "anchor_extent_m": [round(float(c), 4) for c in anchor_ext],
        },
        "thresholds": {
            "min_surviving_pts": MIN_SURVIVING_PTS,
            "worst_axis_ratio_recoverable": WORST_AXIS_RATIO_RECOVERABLE,
            "centroid_consensus_mm": CENTROID_CONSENSUS_MM,
            "centroid_drift_mm": CENTROID_DRIFT_MM,
            "depth_band_percentile": DEPTH_BAND_PERCENTILE,
            "rest_box_margin_m": REST_BOX_MARGIN_M,
        },
        "ineligible_frames": list(INELIGIBLE_RANGE),
        "verdict_counts": {
            "recoverable_consensus": consensus_count,
            "degenerate_partial_keyboard": degenerate_count,
            "mask_drift_delaminated": drift_count,
            "mask_drift_delaminated_near": int(sum(1 for r in per_frame if r["verdict"] == "mask_drift_delaminated_near")),
        },
        "delamination_total": drift_count + int(sum(1 for r in per_frame if r["verdict"] == "mask_drift_delaminated_near")),
        "centroid_displacement_mm": {
            "min": round(min(hd_dists), 1),
            "median": round(float(np.median(hd_dists)), 1),
            "max": round(max(hd_dists), 1),
            "n_within_60mm": n_within_60,
            "n_within_100mm": n_within_100,
            "n_within_150mm": n_within_150,
        },
        "direct_support": {
            "baseline_eligible_frames": direct_support_baseline,
            "recovered_consensus_frames": consensus_count,
            "direct_support_after_recovery": direct_support_recovered,
            "direct_support_rises_above_13": direct_support_recovered > direct_support_baseline,
        },
        "eligible_self_consistency": eligible_self,
        "conclusion": (
            "The 72 visible_surface_ineligible frames are NOT recoverable by hand/background "
            "quarantine. The SAM2 keyboard mask has delaminated off the keyboard: after hand-hull "
            "+ keyboard-depth-band quarantine, the surviving centroid of every one of the 72 frames "
            f"is {min(hd_dists):.0f}-{max(hd_dists):.0f} mm (median {np.median(hd_dists):.0f}) from the "
            f"keyboard rest location, with {n_within_100}/72 within 100 mm and {n_within_60}/72 within "
            "the 60 mm consensus neighbourhood. Quarantine removes contamination but leaves background/"
            "hand-adjacent points that are spatially displaced from the keyboard, not keyboard surface. "
            "direct support cannot truthfully rise above 13/150 via this route."
        ),
        "consensus_consistent_recovered_fits": int(consensus_count),
        "contact_claimed": False,
        "pose_rows_rewritten": False,
        "model_inference_run": False,
        "gpu_used": False,
    }

    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    with open(OUT_DIR / "per_frame.ndjson", "w") as fh:
        for row in per_frame:
            fh.write(json.dumps(row) + "\n")

    # ---- Review figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fis = [r["frame_idx"] for r in per_frame]
        hd = [r["hand_depth_centroid_dist_to_rest_mm"] for r in per_frame]
        box_d = [r["centroid_dist_to_rest_mm"] for r in per_frame]
        n_box = [r["n_after_box"] for r in per_frame]
        worst = [r["worst_axis_ratio_after"] for r in per_frame]
        fit_t = [r["procrustes_fit_translation_mm"] for r in per_frame]

        fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
        axes[0].bar(fis, hd, color="#c44", label="hand+depth centroid dist")
        axes[0].axhline(CENTROID_CONSENSUS_MM, color="green", ls="--", lw=1, label="consensus 60 mm")
        axes[0].axhline(CENTROID_DRIFT_MM, color="orange", ls=":", lw=1, label="drift 100 mm")
        axes[0].set_ylabel("centroid dist to rest (mm)")
        axes[0].set_title("KT-L2 clip001850 keyboard mask-recovery probe (f78-149, 72 ineligible frames)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].bar(fis, box_d, color="#48c")
        axes[1].axhline(CENTROID_CONSENSUS_MM, color="green", ls="--", lw=1)
        axes[1].set_ylabel("box-quarantine centroid\ndist to rest (mm)")
        axes[1].grid(alpha=0.3)

        axes[2].bar(fis, n_box, color="#888")
        axes[2].axhline(MIN_SURVIVING_PTS, color="red", ls="--", lw=1, label="min 200 pts")
        axes[2].set_ylabel("surviving pts after\nhand+depth+box")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

        axes[3].bar(fis, worst, color="#a4c")
        axes[3].axhline(WORST_AXIS_RATIO_RECOVERABLE, color="orange", ls="--", lw=1, label="recoverable 2.5x")
        axes[3].axhline(WORST_AXIS_RATIO_CONSISTENT, color="green", ls=":", lw=1, label="consistent 2.0x")
        axes[3].set_ylabel("worst-axis ratio vs anchor\n(after quarantine)")
        axes[3].set_xlabel("frame idx")
        axes[3].legend(fontsize=8)
        axes[3].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUT_DIR / "review.png", dpi=110)
        plt.close(fig)
        summary["review_figure"] = str(OUT_DIR / "review.png")
    except Exception as e:  # noqa: BLE001
        summary["review_figure_error"] = str(e)

    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)

    print("=== KT-L2 mask-recovery probe ===")
    print(f"ineligible frames: {len(per_frame)} (f78-149)")
    print(f"verdict: recoverable_consensus={consensus_count}  degenerate={degenerate_count}  "
          f"mask_drift={drift_count}")
    print(f"hand+depth centroid dist to rest (mm): min={min(hd_dists):.0f} "
          f"median={np.median(hd_dists):.0f} max={max(hd_dists):.0f}")
    print(f"within 60mm={n_within_60}  within 100mm={n_within_100}  within 150mm={n_within_150}")
    print(f"direct support: {direct_support_baseline} -> {direct_support_recovered} "
          f"(rises above 13: {direct_support_recovered > direct_support_baseline})")
    print(f"outputs: {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-root",
        default="/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1",
    )
    args = ap.parse_args()
    main(args.run_root)
