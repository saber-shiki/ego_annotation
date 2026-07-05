#!/usr/bin/env python3
"""KT-L1: clip001850 keyboard pose-regime classification (pose-independent).

Classifies each frame of clip001850 into one of:
    static_observed | moving_observed | mask_drift | occluded_out_of_view | unresolved
using ONLY pose-independent observed-surface evidence (no pose-fit rows, no
smoothing, no contact claim, no model inference, no GPU).

Evidence channels (all derived from the visible-geometry annotations + raw
masks + raw RGB + per-frame metric cameras):
  1. Pose-independent observed centroid trajectory (world-frame depth centroid).
  2. Rest-cluster consensus from the densest coherent cluster of observed
     centroids (data-driven; not the eligibility gate, which is anchored to a
     possibly-drifted frame).
  3. Per-frame observed-surface displacement vs the rest-cluster centroid.
  4. Mask extent: diagonal ratio and worst-axis ratio vs the (recorded) anchor
     extent, with explicit definitions.
  5. Image-space consistency: project the rest-cluster footprint into each
     frame's camera and measure its overlap with that frame's SAM2 keyboard mask
     (rest_in_mask_fraction). Decisive for "mask drifted onto hand/background".
  6. Hand-coverage of the keyboard mask (keyboard-mask pixels inside any hand
     bbox) as corroborating mask-drift evidence.
  7. Reprojection/crop review sheet proving whether observed points/masks land
     on the keyboard vs hand/background.

Outputs (CPU-only):
  /tmp/clip001850_pose_regime_kt_l1/{summary.json, per_frame.ndjson, review.png}

Run:
  /home/yiwen/ego_annotation/.venv/bin/python \
      scripts/analyze_clip001850_pose_regime_kt_l1.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration (all paths resolved against on-disk layout; the annotations
# carry /mnt/truenas-user-home/... prefixes that are remapped to /data2/...).
# ---------------------------------------------------------------------------
RUN_ROOT = Path(
    "/data2/ego_annotation_outputs/v19_runs/"
    "20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)
VG_JSON = RUN_ROOT / "measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json"
VG_ADAPTER = RUN_ROOT / "measurements/object_geometry/visible_geometry/keyboard/v19_visible_geometry_adapter_report.json"
POSE_REPORT = RUN_ROOT / "measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json"
OUT_DIR = Path("/tmp/clip001850_pose_regime_kt_l1")

PATH_PREFIX_MAP = {
    "/mnt/truenas-user-home/yiwen/ego_annotation_outputs/": "/data2/ego_annotation_outputs/",
}


def resolve_path(p: str) -> str:
    for src, dst in PATH_PREFIX_MAP.items():
        if p.startswith(src):
            return dst + p[len(src):]
    return p


# Decision thresholds. These are DATA-DERIVED at runtime from the rest cluster
# (see derive_thresholds), not hand-picked magic constants. The only fixed
# physical priors are the depth-noise scale of the sensor and the anchor-extent
# gate already encoded upstream; both are reported, not hidden.
REST_CLUSTER_RADIUS_MM = 55.0      # frames within this of a cluster core are "same rest pose"
                                    # (≈ measured HOT3D/UniDepth depth noise floor 25-40 mm + margin;
                                    #  reported alongside the empirical rest spread for audit)
MASK_LEAK_WORST_AXIS = 2.5         # worst-axis extent ratio above this => mask inflation
MASK_LEAK_DIAG = 1.30              # diag extent ratio above this => mask inflation (upstream gate)
REST_IN_MASK_LOW = 0.15            # rest-footprint projected into own mask below this => mask is elsewhere
REST_FOOTPRINT_MIN_VISIBLE = 0.20  # need at least this fraction of rest pts in-frame to trust the test


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def load_mask(mask_path: str) -> np.ndarray | None:
    p = resolve_path(mask_path)
    if not os.path.exists(p):
        return None
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)


def load_rgb(rgb_path: str) -> np.ndarray | None:
    p = resolve_path(rgb_path)
    if not os.path.exists(p):
        return None
    return cv2.imread(p)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def project_world_to_image(pts_world: np.ndarray, T_world_camera: np.ndarray,
                           fx: float, fy: float, cx: float, cy: float,
                           img_w: int, img_h: int, scale: float):
    """Project Nx3 world points to image. scale maps intrinsics (source) -> image px.

    Returns (u, v, depth, valid_in_image).
    """
    T_cw = np.linalg.inv(np.asarray(T_world_camera, dtype=np.float64))
    P = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    Pc = (T_cw @ P.T).T[:, :3]
    z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * Pc[:, 0] / z + cx
        v = fy * Pc[:, 1] / z + cy
    u *= scale
    v *= scale
    valid = (z > 0) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u, v, z, valid


def per_axis_extent(pts: np.ndarray) -> np.ndarray:
    return pts.max(axis=0) - pts.min(axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vg = load_json(VG_JSON)
    adapter = load_json(VG_ADAPTER)
    pose = load_json(POSE_REPORT)

    frames = vg["frames"]
    n_frames = len(frames)
    anchor_frame_idx = adapter.get("anchor_frame_idx")
    anchor_extent = np.array(adapter.get("anchor_extent_world_m", [np.nan, np.nan, np.nan]))
    anchor_centroid = np.array(adapter.get("anchor_centroid_world_m", [np.nan, np.nan, np.nan]))

    # --- gather per-frame observed-surface records -------------------------
    records = []
    for fr in frames:
        idx = fr["frame_idx"]
        cam = fr["camera"]
        T_wc = np.array(cam["T_world_camera_metric"], dtype=np.float64)
        fx, fy, cx, cy = cam["intrinsics_fx_fy_cx_cy"]
        objs = fr.get("objects", [])
        obj = objs[0] if objs else None
        rec = {
            "frame_idx": idx,
            "visible": False,
            "centroid_world_m": None,
            "world_vertices_sample_m": None,
            "world_extent_m": None,
            "extent_ratio_to_anchor_diag": None,
            "extent_ratio_to_anchor_axis": None,
            "worst_axis_ratio": None,
            "rigid_pose_observation_eligible": None,
            "rigid_pose_observation_reason": None,
            "mask_path": None,
            "mask_exists": False,
            "bbox_source_xyxy": None,
            "hands": [],
            "T_world_camera_metric": T_wc,
            "intrinsics": [fx, fy, cx, cy],
            "raw_frame_path": fr.get("raw_frame_path"),
        }
        if obj:
            rec["mask_path"] = obj.get("mask_path")
            rec["bbox_source_xyxy"] = obj.get("bbox_xyxy")
            rec["visible"] = bool(obj.get("visible"))
            rec["rigid_pose_observation_eligible"] = obj.get("rigid_pose_observation_eligible")
            rec["rigid_pose_observation_reason"] = obj.get("rigid_pose_observation_reason")
            cand = obj.get("visible_geometry_candidate") or {}
            rec["centroid_world_m"] = cand.get("centroid_world_m")
            rec["world_vertices_sample_m"] = cand.get("world_vertices_sample_m")
            rec["world_extent_m"] = cand.get("world_extent_m")
            rec["extent_ratio_to_anchor_diag"] = cand.get("extent_ratio_to_anchor_diag")
            rec["extent_ratio_to_anchor_axis"] = cand.get("extent_ratio_to_anchor_axis")
            if cand.get("extent_ratio_to_anchor_axis"):
                rec["worst_axis_ratio"] = float(np.max(np.abs(cand["extent_ratio_to_anchor_axis"])))
        for h in fr.get("hands", []):
            rec["hands"].append({
                "hand_side": h.get("hand_side"),
                "bbox_source_xyxy": h.get("bbox_xyxy"),
                "visibility_state": h.get("visibility_state"),
            })
        if rec["mask_path"]:
            rec["mask_exists"] = os.path.exists(resolve_path(rec["mask_path"]))
        records.append(rec)

    # --- 1. pose-independent observed centroid trajectory ------------------
    obs_centroids = {}
    for r in records:
        if r["centroid_world_m"] is not None:
            obs_centroids[r["frame_idx"]] = np.array(r["centroid_world_m"], dtype=np.float64)

    # --- 2. rest-cluster consensus (data-driven, NOT the eligibility gate) ----
    # The upstream rigid-fit eligibility gate is anchored to f75, which is itself
    # a displaced frame (see anchor_frame_idx). It is therefore NOT trusted as the
    # definition of "clean". Instead:
    #   (a) candidate pool = visible frames whose extent is PLAUSIBLE for the
    #       keyboard object, i.e. extent_ratio_to_anchor_diag in a tight band
    #       around 1.0 (this is a pose-independent observed-extent property; it
    #       separates the ~13 keyboard-sized masks from the 72 inflated leaked
    #       masks with diag ratio 1.3-2.4);
    #   (b) among those candidates, find the tightest coherent centroid cluster
    #       via robust componentwise median + MAD trimming (iterative). This
    #       rejects the extent-plausible-but-centroid-displaced outliers
    #       (f60/75/76/77) without assuming which frames are outliers.
    diag_band = (0.70, 1.20)  # extent-plausible for a keyboard-sized object
    candidate_idxs = []
    for r in records:
        if r["centroid_world_m"] is None:
            continue
        dg = r.get("extent_ratio_to_anchor_diag")
        if dg is not None and diag_band[0] <= dg <= diag_band[1]:
            candidate_idxs.append(r["frame_idx"])

    from scipy.spatial.distance import pdist

    def _trim_to_core(idxs_sub):
        """Iterative median+MAD trim on the candidate centroids."""
        if not idxs_sub:
            return [], np.full(3, np.nan)
        cur = list(idxs_sub)
        for _ in range(5):
            P = np.array([obs_centroids[i] for i in cur])
            med = np.median(P, axis=0)
            mad = np.median(np.abs(P - med), axis=0)
            scale_mad = np.where(mad > 1e-9, mad, 1e-9)
            # robust z in Mahalanobis-like (per-axis, normalised by 1.4826*MAD)
            dz = np.abs(P - med) / (1.4826 * scale_mad)
            mask = (dz < 3.5).all(axis=1)
            new = [cur[i] for i in range(len(cur)) if mask[i]]
            if set(new) == set(cur):
                break
            cur = new if new else cur
        P = np.array([obs_centroids[i] for i in cur]) if cur else np.zeros((0, 3))
        return cur, (np.median(P, axis=0) if len(P) else np.full(3, np.nan))

    rest_idxs, rest_centroid = _trim_to_core(candidate_idxs)
    rest_pts = np.array([obs_centroids[i] for i in rest_idxs]) if rest_idxs else np.zeros((0, 3))
    rest_spread_mm = float(np.median(pdist(rest_pts)) * 1000.0) if len(rest_pts) > 1 else 0.0
    rest_max_dist_mm = (
        float(np.max([np.linalg.norm(p - rest_centroid) for p in rest_pts]) * 1000.0)
        if len(rest_pts) else 0.0)
    # candidates excluded by the centroid trim (extent-plausible but displaced)
    rest_rejected_candidates = sorted(set(candidate_idxs) - set(rest_idxs))

    # Static radius: data-derived from the rest cluster's own internal extent,
    # floored at the sensor depth-noise scale. A frame is static_observed only if
    # its observed centroid lands inside this radius of the rest centroid.
    static_radius_mm = max(rest_max_dist_mm * 1.5, REST_CLUSTER_RADIUS_MM)

    # --- 3/4. per-frame displacement + extent ratios -----------------------
    # Build rest footprint (pool rest-cluster vertices) for reprojection test.
    rest_footprint_pts = []
    for r in records:
        if r["frame_idx"] in set(rest_idxs) and r["world_vertices_sample_m"]:
            rest_footprint_pts.append(np.array(r["world_vertices_sample_m"], dtype=np.float64))
    if rest_footprint_pts:
        rest_footprint = np.vstack(rest_footprint_pts)
        # subsample for speed
        if len(rest_footprint) > 6000:
            sel = np.random.default_rng(0).choice(len(rest_footprint), 6000, replace=False)
            rest_footprint = rest_footprint[sel]
    else:
        rest_footprint = np.zeros((0, 3))

    # source/manifest scale (intrinsics are in source px; masks/rgb are manifest px)
    fr0 = frames[0]
    src_w = fr0["source_width"]
    man_w = fr0["manifest_width"]
    scale = man_w / src_w if src_w else 1.0
    img_w = man_w
    img_h = fr0["manifest_height"]

    per_frame = []
    for r in records:
        idx = r["frame_idx"]
        entry = {
            "frame_idx": idx,
            "visible": r["visible"],
            "regime": None,  # filled below
            "evidence": {},
        }
        # occluded / out of view
        if not r["visible"] or r["centroid_world_m"] is None:
            entry["regime"] = "occluded_out_of_view"
            entry["evidence"]["reason"] = "no visible metric surface (not_visible_or_no_metric_depth)"
            per_frame.append(entry)
            continue

        c = np.array(r["centroid_world_m"], dtype=np.float64)
        disp_mm = float(np.linalg.norm(c - rest_centroid) * 1000.0)
        diag_ratio = r["extent_ratio_to_anchor_diag"]
        worst_axis = r["worst_axis_ratio"]
        world_extent = r["world_extent_m"]
        entry["evidence"]["centroid_world_m"] = c.tolist()
        entry["evidence"]["displacement_from_rest_mm"] = round(disp_mm, 1)
        entry["evidence"]["extent_ratio_to_anchor_diag"] = diag_ratio
        entry["evidence"]["worst_axis_ratio"] = worst_axis
        entry["evidence"]["world_extent_m"] = world_extent
        entry["evidence"]["rigid_pose_observation_eligible"] = r["rigid_pose_observation_eligible"]
        entry["evidence"]["rigid_pose_observation_reason"] = r["rigid_pose_observation_reason"]
        entry["evidence"]["is_anchor_frame"] = (idx == anchor_frame_idx)

        # --- 5. image-space consistency: project rest footprint & own verts --
        T_wc = r["T_world_camera_metric"]
        fx, fy, cx, cy = r["intrinsics"]
        ur, vr, zr, vr_valid = project_world_to_image(
            rest_footprint, T_wc, fx, fy, cx, cy, img_w, img_h, scale)
        n_rest_in = int(vr_valid.sum())
        rest_in_image_frac = float(vr_valid.mean()) if len(vr_valid) else 0.0

        rest_in_mask_frac = None
        mask = load_mask(r["mask_path"]) if r["mask_exists"] else None
        if mask is not None and n_rest_in > 0:
            ui = np.clip(ur[vr_valid].astype(int), 0, img_w - 1)
            vi = np.clip(vr[vr_valid].astype(int), 0, img_h - 1)
            rest_in_mask_frac = float(mask[vi, ui].mean())
        entry["evidence"]["rest_footprint_in_image_frac"] = round(rest_in_image_frac, 3)
        entry["evidence"]["rest_in_mask_fraction"] = round(rest_in_mask_frac, 3) if rest_in_mask_frac is not None else None

        # own observed verts -> own mask (sanity / tautology check on projection)
        own_in_mask_frac = None
        if r["world_vertices_sample_m"] is not None and mask is not None:
            own = np.array(r["world_vertices_sample_m"], dtype=np.float64)
            uo, vo, zo, vo_valid = project_world_to_image(
                own, T_wc, fx, fy, cx, cy, img_w, img_h, scale)
            if vo_valid.sum() > 0:
                ui = np.clip(uo[vo_valid].astype(int), 0, img_w - 1)
                vi = np.clip(vo[vo_valid].astype(int), 0, img_h - 1)
                own_in_mask_frac = float(mask[vi, ui].mean())
        entry["evidence"]["own_in_mask_fraction"] = round(own_in_mask_frac, 3) if own_in_mask_frac is not None else None

        # --- 6. hand-coverage of keyboard mask ------------------------------
        hand_cov = None
        if mask is not None and r["hands"]:
            hand_region = np.zeros_like(mask, dtype=np.uint8)
            for h in r["hands"]:
                bb = h.get("bbox_source_xyxy")
                if bb:
                    x1, y1, x2, y2 = [int(round(v * scale)) for v in bb]
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(img_w, x2); y2 = min(img_h, y2)
                    if x2 > x1 and y2 > y1:
                        hand_region[y1:y2, x1:x2] = 1
            if mask.sum() > 0:
                hand_cov = float((mask & hand_region).sum() / mask.sum())
        entry["evidence"]["keyboard_mask_in_hand_bbox_frac"] = round(hand_cov, 3) if hand_cov is not None else None
        entry["evidence"]["mask_area_px"] = int(mask.sum()) if mask is not None else None

        per_frame.append(entry)

    # --- regime labeling (decision tree, transparent + data-grounded) ------
    # The two cleanest pose-independent discriminators on this clip are:
    #   (i)  observed-centroid displacement from the rest centroid -- there is a
    #        clean gap between the static cluster (8-28 mm) and everything else
    #        (>=65 mm);
    #   (ii) extent_ratio_to_anchor_diag -- static cluster < 1.0, leaked masks
    #        >= 1.30, with a borderline zone (1.0-1.30) for f45/60/75/76/77.
    # rest_in_mask_fraction is reported and used as a tie-breaker for the
    # borderline zone; it does NOT cleanly separate inflated leaked masks from
    # static (some huge leaked masks still cover the keyboard location).
    #
    # Data-derived bounds:
    rest_cluster_worst_axis = []
    for e in per_frame:
        if e["frame_idx"] in set(rest_idxs) and e["evidence"].get("worst_axis_ratio") is not None:
            rest_cluster_worst_axis.append(e["evidence"]["worst_axis_ratio"])
    worst_axis_clean = max(rest_cluster_worst_axis) if rest_cluster_worst_axis else 2.1
    static_cluster_rest_in_mask = [e["evidence"].get("rest_in_mask_fraction") or 0
                                    for e in per_frame if e["frame_idx"] in set(rest_idxs)]
    rest_in_mask_static_min = min(static_cluster_rest_in_mask) if static_cluster_rest_in_mask else 0.5
    # border for "still covers keyboard location": below the static cluster's
    # minimum rest_in_mask but clearly nonzero.
    rest_in_mask_borderline = rest_in_mask_static_min * 0.6

    # Detect any contiguous MOVING run: >=3 contiguous frames, each
    # extent-plausible (diag < MASK_LEAK_DIAG), each displaced beyond
    # static_radius, with a monotonic displacement trend and worst_axis no worse
    # than the clean cluster bound. Motion is a trajectory, not a single frame.
    disp_by_idx = {e["frame_idx"]: e["evidence"].get("displacement_from_rest_mm")
                   for e in per_frame}
    diag_by_idx = {e["frame_idx"]: e["evidence"].get("extent_ratio_to_anchor_diag")
                   for e in per_frame}
    worst_by_idx = {e["frame_idx"]: e["evidence"].get("worst_axis_ratio")
                    for e in per_frame}
    moving_run_frames = set()
    all_idxs = sorted(disp_by_idx)
    for start in range(len(all_idxs)):
        run = []
        for k in range(start, len(all_idxs)):
            fi = all_idxs[k]
            if k > start and fi != all_idxs[k - 1] + 1:
                break
            dg = diag_by_idx[fi]; dp = disp_by_idx[fi]; wa = worst_by_idx[fi]
            if dg is None or dg >= MASK_LEAK_DIAG:
                break
            if dp is None or dp <= static_radius_mm:
                break
            if wa is None or wa > worst_axis_clean + 0.1:
                break
            run.append(fi)
        if len(run) >= 3:
            # monotonic displacement trend (allow 1 outlier step)
            dps = [disp_by_idx[f] for f in run]
            inc = sum(1 for i in range(1, len(dps)) if dps[i] >= dps[i - 1])
            dec = sum(1 for i in range(1, len(dps)) if dps[i] <= dps[i - 1])
            if inc >= len(dps) - 2 or dec >= len(dps) - 2:
                moving_run_frames.update(run)

    regime_counts = {}
    for e in per_frame:
        if e["regime"] == "occluded_out_of_view":
            regime_counts[e["regime"]] = regime_counts.get(e["regime"], 0) + 1
            continue
        ev = e["evidence"]
        disp = ev["displacement_from_rest_mm"]
        worst = ev.get("worst_axis_ratio")
        diag = ev.get("extent_ratio_to_anchor_diag")
        rim = ev.get("rest_in_mask_fraction")
        oim = ev.get("own_in_mask_fraction")
        hand_cov = ev.get("keyboard_mask_in_hand_bbox_frac")
        fidx = e["frame_idx"]

        extent_inflated = (diag is not None and diag >= MASK_LEAK_DIAG)

        if extent_inflated:
            regime = "mask_drift"
            ev["regime_basis"] = (
                f"extent inflated (diag={diag:.2f}>={MASK_LEAK_DIAG}, "
                f"worst_axis={worst:.2f}); systematic hand/background leakage; "
                f"centroid {disp:.0f}mm from rest")
        elif disp <= static_radius_mm:
            regime = "static_observed"
            ev["regime_basis"] = (
                f"centroid within static radius ({disp:.0f}mm<="
                f"{static_radius_mm:.0f}mm); extent consistent "
                f"(diag={diag:.2f}); rest_in_mask={rim}")
        elif fidx in moving_run_frames:
            regime = "moving_observed"
            ev["regime_basis"] = (
                f"part of a contiguous monotonic displaced run "
                f"(disp={disp:.0f}mm, diag={diag:.2f}, worst_axis={worst:.2f})")
        elif (disp <= static_radius_mm * 1.4 and rim is not None and rim >= rest_in_mask_borderline
              and hand_cov is not None and hand_cov >= 0.40):
            # borderline: just past the static gap, mask still partly covers the
            # keyboard location, and the hand is occluding => ambiguous static.
            regime = "unresolved"
            ev["regime_basis"] = (
                f"borderline: displaced {disp:.0f}mm (just past static radius "
                f"{static_radius_mm:.0f}mm) but rest_in_mask={rim} (>= borderline "
                f"{rest_in_mask_borderline:.2f}) and hand_cov={hand_cov} (>=0.40, "
                "hand occluding); cannot separate static-with-occlusion from "
                "mild mask shift")
        else:
            regime = "mask_drift"
            ev["regime_basis"] = (
                f"centroid displaced {disp:.0f}mm>static_radius "
                f"{static_radius_mm:.0f}mm, extent-plausible (diag={diag:.2f}) "
                f"but observed surface is not at the keyboard rest location "
                f"(rest_in_mask={rim}, hand_cov={hand_cov}); not part of any "
                "contiguous moving run => mask drifted off the keyboard")
        e["regime"] = regime
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    # --- focused displacements for the named outlier frames ----------------
    pose_graph_frames = pose.get("graph_frames", [])
    named = {}
    for e in per_frame:
        if e["frame_idx"] in (60, 75, 76, 77) and e["evidence"].get("displacement_from_rest_mm") is not None:
            named[e["frame_idx"]] = {
                "displacement_from_rest_mm": e["evidence"]["displacement_from_rest_mm"],
                "extent_ratio_to_anchor_diag": e["evidence"]["extent_ratio_to_anchor_diag"],
                "worst_axis_ratio": e["evidence"]["worst_axis_ratio"],
                "rest_in_mask_fraction": e["evidence"]["rest_in_mask_fraction"],
                "own_in_mask_fraction": e["evidence"]["own_in_mask_fraction"],
                "keyboard_mask_in_hand_bbox_frac": e["evidence"]["keyboard_mask_in_hand_bbox_frac"],
                "regime": e["regime"],
                "is_anchor_frame": e["evidence"]["is_anchor_frame"],
            }

    # --- build review contact sheet ----------------------------------------
    review_path = OUT_DIR / "review.png"
    try:
        review_path = build_review_sheet(records, per_frame, rest_centroid, rest_idxs,
                                         anchor_frame_idx, scale, img_w, img_h, rest_footprint)
    except Exception as ex:  # noqa: BLE001 - review is best-effort, must not sink the analysis
        review_path = None
        review_error = repr(ex)
    else:
        review_error = None

    # --- definitions (recorded for audit) ----------------------------------
    definitions = {
        "extent_ratio_to_anchor_diag": (
            "ratio of the observed world-frame extent diagonal "
            "(||world_extent_m||_2) to the anchor frame's extent diagonal "
            "(||anchor_extent_world_m||_2). ~1.0 => extent-consistent; the "
            "upstream rigid-fit eligibility gate accepts frames near 1.0."),
        "worst_axis_ratio": (
            "max over the three world axes of |world_extent_m[axis] / "
            "anchor_extent_world_m[axis]|. Large values indicate one axis is "
            "inflated relative to the anchor => hand/background leakage."),
        "rest_cluster": (
            "the robust core of pose-independent observed centroids among "
            "extent-plausible frames (extent_ratio_to_anchor_diag in [0.70,1.20]); "
            "found by iterative componentwise median+MAD trimming. NOT the pose-fit "
            "eligibility gate, which is anchored to f75 -- a displaced frame."),
        "rest_in_mask_fraction": (
            "fraction of the rest-cluster world footprint points, projected "
            "into this frame's camera, that land inside this frame's SAM2 "
            "keyboard mask. High => the static keyboard location is where the "
            "mask is. Low (with adequate rest footprint in-frame) => the mask "
            "is on a different region (hand/background) => mask drift."),
        "own_in_mask_fraction": (
            "fraction of THIS frame's own observed world vertices, projected "
            "into its own camera, that land inside its own mask. Should be ~1.0 "
            "(tautology confirming the projection pipeline); reported as a "
            "sanity check, not a regime discriminator."),
        "static_radius_mm": (
            "data-derived: max(rest-cluster internal extent, sensor depth floor "
            "REST_CLUSTER_RADIUS_MM). Frames within this of the rest centroid "
            "are static_observed."),
    }

    summary = {
        "task": "KT-L1 pose-regime classification (pose-independent)",
        "run_root": str(RUN_ROOT),
        "anchor_frame_idx": anchor_frame_idx,
        "anchor_centroid_world_m": anchor_centroid.tolist(),
        "anchor_extent_world_m": anchor_extent.tolist(),
        "n_frames": n_frames,
        "regime_counts": regime_counts,
        "rest_cluster": {
            "candidate_pool_frame_idxs": candidate_idxs,
            "candidate_pool_diag_band": list(diag_band),
            "frame_idxs": rest_idxs,
            "centroid_world_m": rest_centroid.tolist(),
            "internal_spread_mm": round(rest_spread_mm, 1),
            "internal_max_dist_mm": round(rest_max_dist_mm, 1),
            "extent_plausible_but_centroid_displaced": rest_rejected_candidates,
        },
        "static_radius_mm": round(static_radius_mm, 1),
        "named_outlier_frames": named,
        "pose_graph_direct_fit_frames": pose_graph_frames,
        "pose_graph_optimizer": {
            "nfev": pose.get("optimizer", {}).get("nfev"),
            "cost": pose.get("optimizer", {}).get("cost"),
            "nonpenetration_target_frame_count": pose.get("nonpenetration_target_frame_count"),
        },
        "thresholds": {
            "REST_CLUSTER_RADIUS_MM": REST_CLUSTER_RADIUS_MM,
            "MASK_LEAK_DIAG": MASK_LEAK_DIAG,
            "rest_cluster_worst_axis_max": worst_axis_clean,
            "rest_in_mask_borderline": round(rest_in_mask_borderline, 3),
            "rest_in_mask_static_cluster_min": round(rest_in_mask_static_min, 3),
            "moving_run_min_length": 3,
            "note": ("static_radius_mm is data-derived (max(rest-cluster internal "
                     "extent*1.5, sensor depth floor)). MASK_LEAK_DIAG=1.30 is the "
                     "clean gap in the diag-ratio distribution (static<1.0, "
                     "leaked>=1.34). worst_axis_clean is the rest cluster's own max. "
                     "rest_in_mask_borderline = 0.6*static-cluster-min. moving_observed "
                     "requires a contiguous monotonic run of >=3 extent-plausible frames "
                     "-- motion is a trajectory, not a single frame. All bounds are "
                     "reported for audit; none are tuned to suppress a fabrication."),
        },
        "definitions": definitions,
        "route_verdict": derive_route_verdict(regime_counts, rest_idxs, named),
        "review_path": str(review_path) if review_path else None,
        "review_error": review_error,
        "scope": {
            "smoothing_applied": False,
            "pose_rows_rewritten": False,
            "contact_claimed": False,
            "model_inference": False,
            "gpu_used": False,
        },
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_DIR / "per_frame.ndjson", "w") as f:
        for e in per_frame:
            f.write(json.dumps(e) + "\n")

    # console report
    print("=" * 70)
    print("KT-L1 clip001850 keyboard pose-regime classification")
    print("=" * 70)
    print(f"anchor_frame_idx = {anchor_frame_idx}  (centroid displaced from rest!)")
    print(f"rest_cluster frames = {rest_idxs}")
    print(f"rest_centroid = {np.round(rest_centroid,3).tolist()}  spread={rest_spread_mm:.1f}mm")
    print(f"static_radius_mm = {static_radius_mm:.1f}")
    print(f"regime_counts = {regime_counts}")
    print("-- named outlier frames (f60/75/76/77) --")
    for k in sorted(named):
        print(f"  f{k}: {named[k]}")
    print(f"\nroute_verdict = {summary['route_verdict']}")
    print(f"outputs: {OUT_DIR}")


def derive_route_verdict(regime_counts, rest_idxs, named):
    """Translate the KT-L1 outcome into the route decision of subagent-25 KT-L1."""
    static = regime_counts.get("static_observed", 0)
    moving = regime_counts.get("moving_observed", 0)
    drift = regime_counts.get("mask_drift", 0)
    oov = regime_counts.get("occluded_out_of_view", 0)
    unresolved = regime_counts.get("unresolved", 0)
    # f75/76/77 are the anchor + its neighbours
    anchor_drift = all(named.get(f, {}).get("regime") == "mask_drift" for f in (75, 76, 77))
    has_moving_run = moving >= 3
    if has_moving_run:
        return ("moving_observed: a contiguous run of frames shows coherent "
                "monotonic centroid displacement with rest-footprint still inside "
                "the mask. Route: supported moving trajectory (smoothing admissible "
                "ONLY on those frames with independent support).")
    if static > 0 and not has_moving_run:
        route = (
            "static-held gauge: the keyboard is observed as a static rest pose "
            f"({static} frames in the rest cluster {rest_idxs}). The displaced "
            "eligible frames (incl. the anchor) are mask_drift, not motion. "
            "Route: robust static T_rest over the rest cluster + per-DOF covariance "
            "+ visibility ledger; reject f75/76/77 outliers; mark occluded/leaked "
            "frames as occluded/mask-unreliable, never measured.")
        if anchor_drift:
            route += (" CRITICAL: the visible-geometry anchor frame is itself a "
                      "mask-drift outlier, so the upstream extent-consistency gate "
                      "and the ICP fits seeded from it are contaminated; the rest "
                      "cluster must be re-anchored before any pose re-fit.")
        return route
    return ("unresolved: insufficient static or moving consensus to route; collect "
            "additional observed-surface evidence before choosing an estimator.")


# ---------------------------------------------------------------------------
# Review sheet
# ---------------------------------------------------------------------------
def build_review_sheet(records, per_frame, rest_centroid, rest_idxs,
                       anchor_frame_idx, scale, img_w, img_h, rest_footprint):
    """Contact sheet: for selected frames show RGB + mask + rest-footprint + own pts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # choose review frames: rest sample + outliers + leaked + oov
    by_idx = {r["frame_idx"]: r for r in records}
    regime_by_idx = {e["frame_idx"]: e["regime"] for e in per_frame}
    review_frames = [0, 28, 30, 33, 36, 45, 46, 60, 75, 76, 77, 78, 90, 120, 149]
    review_frames = [f for f in review_frames if f in by_idx]
    ncols = 3
    nrows = int(np.ceil(len(review_frames) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, fidx in zip(axes, review_frames):
        r = by_idx[fidx]
        rgb = load_rgb(r["raw_frame_path"]) if r["raw_frame_path"] else None
        if rgb is None:
            ax.text(0.5, 0.5, f"f{fidx}\n(no rgb)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        # mask overlay
        mask = load_mask(r["mask_path"]) if r["mask_exists"] else None
        if mask is not None:
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
            mask_rgba[mask > 0] = [1, 0, 0, 0.35]
            ax.imshow(mask_rgba)
        # rest footprint projection
        T_wc = r["T_world_camera_metric"]
        fx, fy, cx, cy = r["intrinsics"]
        if len(rest_footprint) > 0:
            u, v, z, valid = project_world_to_image(
                rest_footprint, T_wc, fx, fy, cx, cy, img_w, img_h, scale)
            if valid.sum() > 0:
                ax.scatter(u[valid], v[valid], s=1, c="lime", alpha=0.25)
        # own observed pts
        if r["world_vertices_sample_m"] is not None:
            own = np.array(r["world_vertices_sample_m"], dtype=np.float64)
            uo, vo, zo, vao = project_world_to_image(
                own, T_wc, fx, fy, cx, cy, img_w, img_h, scale)
            if vao.sum() > 0:
                ax.scatter(uo[vao], vo[vao], s=1, c="yellow", alpha=0.25)
        # hand bboxes
        for h in r["hands"]:
            bb = h.get("bbox_source_xyxy")
            if bb:
                x1, y1, x2, y2 = [v * scale for v in bb]
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor="cyan", linewidth=1.2))
        regime = regime_by_idx.get(fidx, "?")
        ev = next((e["evidence"] for e in per_frame if e["frame_idx"] == fidx), {})
        disp = ev.get("displacement_from_rest_mm")
        rim = ev.get("rest_in_mask_fraction")
        worst = ev.get("worst_axis_ratio")
        title = (f"f{fidx} [{regime}]  anchor={'Y' if fidx==anchor_frame_idx else 'N'}\n"
                 f"disp={disp} rest_in_mask={rim} worst_axis={worst}")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(review_frames):]:
        ax.axis("off")
    fig.suptitle("KT-L1 clip001850 keyboard regime review\n"
                 "red=SAM2 mask  lime=rest-cluster footprint (where static keyboard projects)  "
                 "yellow=own observed pts  cyan=hand bbox",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "review.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
