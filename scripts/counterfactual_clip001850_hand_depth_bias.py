#!/usr/bin/env python3
"""Hand-depth-bias counterfactual for HOT3D clip001850 keyboard, right hand.

Causal question
---------------
The masked-depth kill-tests (subagent 07) show the MANO right hand sits 4-18 cm
*in front of* the keyboard-masked, hand-quarantined depth surface on every
mask-available frame, while the interval solver's completed-mesh penetration
channel reports large "inside" penetration. Subagent 07/10's open caveat (D4):
is the ~8 cm gap a MANO hand metric-depth bias (hand too close to camera), and
if corrected, would the fingertip vertices reach a contact_candidate without
breaking the existing projection tolerance?

This script answers it as a counterfactual: translate the *whole* MANO hand
along the camera ray (+camera z, away from camera) by the amount needed to bring
the fingertip/keyboard-overlap vertices into the keyboard-masked +
hand-quarantined depth band, then measure the consequences:

  1. required_shift_m        : along-ray translation (median over overlap verts)
  2. contact_after_shift     : does the shifted hand reach contact_candidate?
  3. reprojection_displacement_px : image-plane joint shift caused by the counterfactual
                               compared to the interval solver's already-tolerated
                               joint shift (visible_joint_shift_px).
  4. completed_mesh_distance : unsigned distance to the completed mesh after shift.

Decision: is ~8 cm hand-depth correction physically plausible (within the
combined metric floor and the interval solver's tolerated deltas) and would it
create contact_candidate without worsening projection beyond the existing
tolerance?

CPU-only. Reads only on-disk v19 artifacts. No model inference, no GPU.

Outputs (default /tmp/clip001850_hand_depth_bias_counterfactual/):
  per_frame.ndjson     -- one row per frame
  summary.json         -- aggregate decision + comparisons
  review_f32.jpg, review_f36.jpg, review_f45.jpg  (if RGB available)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

RUN_ROOT_DEFAULT = Path(
    "/data2/ego_annotation_outputs/v19_runs/"
    "20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)

# Soft-tissue / contact band (m). Within |delta| <= this counts as in-band.
SOFT_TISSUE_BAND_M = 0.015
# Clip metric floor (combined lateral + along-ray + rotation), from
# hand_metric_mechanisms.md Track J/M. Systematic bias, not averageable.
SIGMA_CLIP_M = (0.025, 0.030)        # min, max of the floor
# Existing projection tolerance anchor: interval solver's tolerated joint shift
# on the mask-available frames (visible_joint_shift_px median ~8-11px, max ~13px).
# We compare the counterfactual reprojection to this.
PROJECTION_TOLERANCE_PX = 13.0       # the interval solver already tolerated up to ~13px

DEFAULT_FRAMES = [30, 31, 32, 33, 34, 35, 36, 45, 46]


# --------------------------------------------------------------------------- #
def _remount(p: str | Path) -> Path:
    p = Path(p)
    if p.exists():
        return p
    s = str(p)
    for prefix in ("/mnt/truenas-user-home", "/mnt/user-home"):
        if s.startswith(prefix):
            local = Path("/data2" + s[len(prefix):])
            if local.exists():
                return local
    return p


def load_json(path: Path) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def load_mask_1408(path: Path) -> np.ndarray | None:
    path = _remount(path)
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = arr > 0
    if mask.shape != (1408, 1408):
        mask = cv2.resize(mask.astype(np.uint8), (1408, 1408),
                          interpolation=cv2.INTER_NEAREST) > 0
    return mask


MANO_FINGERTIP_JOINTS = np.array(sorted({4, 8, 12, 16, 20}))


def fingertip_vertex_indices(verts_world: np.ndarray,
                             joints_world: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(verts_world[:, None, :] - joints_world[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    return np.where(np.isin(nearest, MANO_FINGERTIP_JOINTS))[0]


def project(verts_world, R_c2w, t_c2w, intr):
    fx, fy, cx, cy = intr
    cam = (verts_world - t_c2w[None, :]) @ R_c2w
    cam_z = cam[:, 2]
    zsafe = np.maximum(cam_z, 1e-9)
    u = fx * cam[:, 0] / zsafe + cx
    v = fy * cam[:, 1] / zsafe + cy
    return u, v, cam_z


def numeric_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None,
                "max": None, "min": None, "mean": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
    }


@dataclass
class FrameData:
    frame_idx: int
    verts_world: np.ndarray       # (S,3) sample verts
    sample_ids: np.ndarray
    joints_world: np.ndarray      # (21,3)
    R_c2w: np.ndarray
    t_c2w: np.ndarray
    intr: tuple
    depth: np.ndarray             # (1408,1408)
    kb_mask: np.ndarray | None
    hand_support_mask: np.ndarray | None
    voo_mask: np.ndarray | None
    # interval solver deltas (existing tolerated)
    interval_cam_z_shift_m: float | None
    interval_lateral_norm_m: float | None
    interval_visible_joint_shift_px: dict | None
    interval_full_pen: dict | None
    # pose
    R_obj: np.ndarray
    t_obj: np.ndarray
    pose_source: str


def build_frame(run_root, f, ann_frames, mano_by_frame, frame_idx_arr, depth_arr,
                kb_mask_dir, own_dir, hand_side="right"):
    frame = ann_frames.get(f)
    if frame is None:
        return None
    mano = mano_by_frame.get(f)
    if mano is None:
        return None
    cam = frame.get("camera", {})
    T = np.asarray(cam.get("T_world_camera_metric") or [], dtype=float)
    if T.shape != (4, 4):
        return None
    R_c2w = T[:3, :3]
    t_c2w = T[:3, 3]
    intr_cam = cam.get("intrinsics_fx_fy_cx_cy")
    if not intr_cam or len(intr_cam) != 4:
        # fall back to hand metric state intrinsics
        h = next((h for h in frame.get("hands", []) if h.get("hand_side") == hand_side), None)
        ms = (h or {}).get("metric_mano_state", {}) or {}
        intr_cam = ms.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
    intr = tuple(float(x) for x in intr_cam)

    vw = np.asarray(mano["optimized_vertices_world_sample_m"], dtype=float)
    sids = np.asarray(mano["optimized_vertices_sample_ids"], dtype=int)
    jw = np.asarray(mano["optimized_joints_world_m"], dtype=float)

    di = np.where(frame_idx_arr == f)[0]
    if di.size == 0:
        return None
    depth = depth_arr[di[0]].astype(np.float32)

    kb = load_mask_1408(kb_mask_dir / f"{f:06d}.png")
    hs = load_mask_1408(own_dir / "right" / f"{f:06d}_projected_mano_hand_support.png")
    voo = load_mask_1408(own_dir / "right" / f"{f:06d}_visible_object_owned.png")

    # object pose (for completed-mesh distance context)
    # from pose report
    return FrameData(
        frame_idx=f, verts_world=vw, sample_ids=sids, joints_world=jw,
        R_c2w=R_c2w, t_c2w=t_c2w, intr=intr, depth=depth,
        kb_mask=kb, hand_support_mask=hs, voo_mask=voo,
        interval_cam_z_shift_m=mano.get("optimized_translation_camera_z_m"),
        interval_lateral_norm_m=mano.get("optimized_translation_lateral_norm_m"),
        interval_visible_joint_shift_px=mano.get("visible_joint_shift_px"),
        interval_full_pen=mano.get("full_observed_surface_penetration_after_solver_m"),
        R_obj=np.eye(3), t_obj=np.zeros(3),  # filled later from pose report
        pose_source="",
    )


def compute_overlap(fd: FrameData):
    """Identify fingertip verts that project onto the keyboard-masked +
    hand-quarantined (visible_object_owned) band, and the gap to the surface."""
    u, v, cam_z = project(fd.verts_world, fd.R_c2w, fd.t_c2w, fd.intr)
    H, W = fd.depth.shape
    ur = np.rint(u).astype(int)
    vr = np.rint(v).astype(int)
    inb = (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H)
    surf = np.full(len(fd.verts_world), np.nan)
    surf[inb] = fd.depth[vr[inb], ur[inb]].astype(float)
    finite = np.isfinite(surf) & inb

    kb_on = np.zeros(len(fd.verts_world), dtype=bool)
    voo_on = np.zeros(len(fd.verts_world), dtype=bool)
    if fd.kb_mask is not None:
        kb_on[finite] = fd.kb_mask[vr[finite], ur[finite]]
    if fd.voo_mask is not None:
        voo_on[finite] = fd.voo_mask[vr[finite], ur[finite]]

    tips = fingertip_vertex_indices(fd.verts_world, fd.joints_world)

    # delta = cam_z - surf : negative = hand in front (gap), positive = behind (pen)
    delta = cam_z - surf

    # overlap = fingertip verts projecting onto keyboard-masked, hand-quarantined band
    overlap_all = voo_on & finite
    overlap_tip = overlap_all & np.isin(np.arange(len(fd.verts_world)), tips)

    # Also keyboard-masked fingertip overlap (broader)
    kb_tip = kb_on & finite & np.isin(np.arange(len(fd.verts_world)), tips)

    return {
        "u": u, "v": v, "cam_z": cam_z, "surf": surf, "delta": delta,
        "inb": inb, "tips": tips,
        "overlap_all": overlap_all, "overlap_tip": overlap_tip, "kb_tip": kb_tip,
        "kb_on": kb_on, "voo_on": voo_on,
    }


def apply_ray_shift(fd: FrameData, shift_m: float):
    """Translate the whole hand along +camera z (away from camera) by shift_m.
    Returns shifted verts, joints, and the world translation vector applied."""
    z_world = fd.R_c2w[:, 2]                      # camera +z axis in world
    tw = shift_m * z_world
    verts2 = fd.verts_world + tw[None, :]
    joints2 = fd.joints_world + tw[None, :]
    return verts2, joints2, tw


def required_shift_to_surface(delta_tip_overlap: np.ndarray):
    """Given delta = cam_z - surf (negative = in front) for fingertip/overlap
    verts, the along-ray shift (away from camera) to bring each vert to the
    surface (delta -> 0) is shift = -delta = gap. Positive = move hand deeper."""
    return -delta_tip_overlap


def required_shift_for_band(delta_tip_overlap: np.ndarray,
                            band_m: float = SOFT_TISSUE_BAND_M):
    """Minimal shift to reach the near edge of the band (surf - band)."""
    return (-delta_tip_overlap) - band_m


def per_frame_counterfactual(fd: FrameData, completed_mesh_tree):
    ov = compute_overlap(fd)

    # representative gap over fingertip verts on keyboard-masked, hand-quarantined band
    tip_voo_delta = ov["delta"][ov["overlap_tip"]]
    tip_voo_delta = tip_voo_delta[np.isfinite(tip_voo_delta)]
    tip_kb_delta = ov["delta"][ov["kb_tip"]]
    tip_kb_delta = tip_kb_delta[np.isfinite(tip_kb_delta)]

    # required shift to bring each overlap fingertip to the SURFACE (gap).
    shifts_voo = required_shift_to_surface(tip_voo_delta) \
        if tip_voo_delta.size else np.array([])
    shifts_kb = required_shift_to_surface(tip_kb_delta) \
        if tip_kb_delta.size else np.array([])
    # also the band-edge shift (minimal to enter the band) for reporting
    band_voo = required_shift_for_band(tip_voo_delta, SOFT_TISSUE_BAND_M) \
        if tip_voo_delta.size else np.array([])
    band_kb = required_shift_for_band(tip_kb_delta, SOFT_TISSUE_BAND_M) \
        if tip_kb_delta.size else np.array([])

    # representative shift: median of positive shifts over VOO fingertip verts;
    # fall back to keyboard-masked fingertip verts if VOO set is empty.
    src = "keyboard_masked_hand_quarantined" if shifts_voo.size else (
        "keyboard_masked" if shifts_kb.size else "none")
    cand = shifts_voo if shifts_voo.size else shifts_kb
    if cand.size:
        pos = cand[cand > 0]
        required_shift = float(np.median(pos)) if pos.size else float(np.min(cand))
    else:
        required_shift = None

    # --- existing tolerance anchors ---
    vjsp = fd.interval_visible_joint_shift_px or {}
    existing_joint_shift_med_px = vjsp.get("median")
    existing_joint_shift_max_px = vjsp.get("max")
    interval_cz = fd.interval_cam_z_shift_m
    interval_lat = fd.interval_lateral_norm_m

    # --- apply the counterfactual shift ---
    result_after = None
    reprojection_summary = None
    contact_after = None
    completed_dist_after = None
    completed_dist_before = None
    if required_shift is not None and required_shift > 0:
        verts2, joints2, tw = apply_ray_shift(fd, required_shift)

        # reproject joints before/after
        u0, v0, _ = project(fd.joints_world, fd.R_c2w, fd.t_c2w, fd.intr)
        u1, v1, _ = project(joints2, fd.R_c2w, fd.t_c2w, fd.intr)
        dpx = np.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2)
        reprojection_summary = numeric_summary(dpx)

        # recompute overlap status after shift
        ov2 = compute_overlap(FrameData(
            frame_idx=fd.frame_idx, verts_world=verts2, sample_ids=fd.sample_ids,
            joints_world=joints2, R_c2w=fd.R_c2w, t_c2w=fd.t_c2w, intr=fd.intr,
            depth=fd.depth, kb_mask=fd.kb_mask, hand_support_mask=fd.hand_support_mask,
            voo_mask=fd.voo_mask,
            interval_cam_z_shift_m=None, interval_lateral_norm_m=None,
            interval_visible_joint_shift_px=None, interval_full_pen=None,
            R_obj=fd.R_obj, t_obj=fd.t_obj, pose_source=fd.pose_source))
        # post-shift contact check uses the KEYBOARD MASK directly (not the
        # stale VOO quarantine): the hand has moved, so the original
        # hand-support quarantine is no longer valid; the keyboard is still
        # where its mask says it is.
        delta_tip2 = ov2["delta"][ov2["kb_tip"]]
        delta_tip2 = delta_tip2[np.isfinite(delta_tip2)]
        # contact_candidate: at least one fingertip within band of keyboard surface
        in_band = np.abs(delta_tip2) <= SOFT_TISSUE_BAND_M
        # also allow modest behind-surface (soft penetration) within 2*band
        near = np.abs(delta_tip2) <= 2 * SOFT_TISSUE_BAND_M
        contact_after = {
            "fingertip_on_keyboard_count_after": int(delta_tip2.size),
            "in_soft_tissue_band_count": int(in_band.sum()) if delta_tip2.size else 0,
            "within_2band_count": int(near.sum()) if delta_tip2.size else 0,
            "delta_tip_summary_m": numeric_summary(delta_tip2),
            "contact_candidate": bool(delta_tip2.size and bool(in_band.any())),
        }

        # completed mesh distance before/after shift (optional, trimesh)
        if completed_mesh_tree is not None:
            import trimesh
            verts_obj0 = (fd.verts_world - fd.t_obj[None, :]) @ fd.R_obj
            _, dist0, _ = trimesh.proximity.closest_point(completed_mesh_tree, verts_obj0)
            completed_dist_before = numeric_summary(dist0)
            verts_obj = (verts2 - fd.t_obj[None, :]) @ fd.R_obj
            _, dist, _ = trimesh.proximity.closest_point(completed_mesh_tree, verts_obj)
            completed_dist_after = numeric_summary(dist)

        result_after = {
            "shift_world_translation_m": tw.tolist(),
            "reprojection_displacement_px": reprojection_summary,
            "contact_after_shift": contact_after,
            "completed_mesh_unsigned_dist_before_shift_m": completed_dist_before,
            "completed_mesh_unsigned_dist_after_shift_m": completed_dist_after,
        }

    # --- decision ---
    decision = decide(required_shift, reprojection_summary, contact_after,
                      interval_cz, existing_joint_shift_max_px)

    return {
        "frame_idx": fd.frame_idx,
        "fingertip_overlap_voo_count": int(ov["overlap_tip"].sum()),
        "fingertip_overlap_kb_count": int(ov["kb_tip"].sum()),
        "gap_delta_tip_voo_summary_m": numeric_summary(tip_voo_delta),
        "gap_delta_tip_kb_summary_m": numeric_summary(tip_kb_delta),
        "required_shifts_to_surface_voo_summary_m": numeric_summary(shifts_voo),
        "required_shifts_to_surface_kb_summary_m": numeric_summary(shifts_kb),
        "required_shifts_to_band_edge_voo_summary_m": numeric_summary(band_voo),
        "required_shifts_to_band_edge_kb_summary_m": numeric_summary(band_kb),
        "representative_required_shift_m": required_shift,
        "representative_shift_source": src,
        "interval_solver_cam_z_shift_m": interval_cz,
        "interval_solver_lateral_norm_m": interval_lat,
        "interval_solver_visible_joint_shift_px": {
            "median": existing_joint_shift_med_px,
            "max": existing_joint_shift_max_px,
        },
        "counterfactual_result": result_after,
        "decision": decision,
    }


def decide(required_shift, reprojection, contact_after, interval_cz,
           existing_joint_shift_max_px):
    if required_shift is None:
        return {
            "plausible_vs_metric_floor": "undetermined_no_overlap",
            "creates_contact_candidate": False,
            "projection_within_tolerance": None,
            "verdict": "no_overlap_fingertip_keyboard_masked",
        }
    smin, smax = SIGMA_CLIP_M
    # plausible: required shift not wildly beyond metric floor AND within the
    # order of magnitude of the interval solver's already-applied deltas (which
    # reached ~3.8 cm cam-z). We flag the ratio to the floor.
    ratio_to_floor_max = required_shift / smax if smax else None
    plausible = required_shift <= 5.0 * smax   # 5x floor (~15cm) is the hard implausibility line

    creates_contact = bool(contact_after and contact_after.get("contact_candidate"))

    rep_max = (reprojection or {}).get("max")
    rep_med = (reprojection or {}).get("median")
    tol = existing_joint_shift_max_px if existing_joint_shift_max_px else PROJECTION_TOLERANCE_PX
    within = (rep_max is not None and rep_max <= max(tol, PROJECTION_TOLERANCE_PX))

    if not plausible:
        verdict = "shift_implausibly_large_vs_metric_floor"
    elif creates_contact and within:
        verdict = "contact_candidate_created_projection_tolerated"
    elif creates_contact and not within:
        verdict = "contact_candidate_created_but_projection_exceeds_tolerance"
    elif not creates_contact:
        verdict = "shift_inadequate_for_contact_candidate"
    else:
        verdict = "indeterminate"

    return {
        "required_shift_m": required_shift,
        "ratio_to_metric_floor_max": ratio_to_floor_max,
        "ratio_to_interval_solver_cam_z_shift": (required_shift / interval_cz) if interval_cz else None,
        "plausible_vs_metric_floor": "plausible" if plausible else "implausible",
        "creates_contact_candidate": creates_contact,
        "reprojection_max_px": rep_max,
        "projection_tolerance_px": tol,
        "projection_within_tolerance": within,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Review image
# --------------------------------------------------------------------------- #
def make_review(fd, ov, required_shift, result_after, rgb_path, out_path, title):
    rgb_path = _remount(rgb_path)
    img = None
    if rgb_path.exists():
        img = cv2.imread(str(rgb_path))
    H, W = 1408, 1408
    if img is None:
        img = np.full((H, W, 3), 24, dtype=np.uint8)
    else:
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    overlay = img.copy()
    if fd.kb_mask is not None:
        ov_ = overlay.copy()
        ov_[fd.kb_mask] = (0.55 * ov_[fd.kb_mask] + 0.45 * np.array([40, 255, 80])).astype(np.uint8)
        overlay = ov_
    if fd.voo_mask is not None:
        ov_ = overlay.copy()
        ov_[fd.voo_mask] = (0.6 * ov_[fd.voo_mask] + 0.4 * np.array([0, 200, 255])).astype(np.uint8)
        overlay = ov_

    # original hand verts (cyan)
    u0, v0, _ = project(fd.verts_world, fd.R_c2w, fd.t_c2w, fd.intr)
    ok = (u0 >= 0) & (u0 < W) & (v0 >= 0) & (v0 < H)
    for uu, vv in zip(u0[ok], v0[ok]):
        cv2.circle(overlay, (int(uu), int(vv)), 2, (255, 200, 0), -1)
    # original fingertip (magenta)
    for i in ov["tips"]:
        if ok[i]:
            cv2.circle(overlay, (int(u0[i]), int(v0[i])), 6, (255, 0, 255), 2)

    # shifted hand verts (red)
    if required_shift is not None and required_shift > 0:
        verts2, joints2, _ = apply_ray_shift(fd, required_shift)
        u1, v1, _ = project(verts2, fd.R_c2w, fd.t_c2w, fd.intr)
        ok2 = (u1 >= 0) & (u1 < W) & (v1 >= 0) & (v1 < H)
        for uu, vv in zip(u1[ok2], v1[ok2]):
            cv2.circle(overlay, (int(uu), int(vv)), 2, (0, 0, 255), -1)
        for i in ov["tips"]:
            if ok2[i]:
                cv2.circle(overlay, (int(u1[i]), int(v1[i])), 6, (0, 0, 255), 2)
        # arrows for a few fingertip verts
        for i in ov["tips"][:12]:
            if ok[i] and ok2[i]:
                cv2.arrowedLine(overlay, (int(u0[i]), int(v0[i])),
                                (int(u1[i]), int(v1[i])), (255, 255, 255), 1, tipLength=0.2)

    # text panel
    lines = [title]
    lines.append(f"required shift = {required_shift*1000:.1f} mm along camera ray (away)" if required_shift else "no overlap")
    if result_after:
        rp = result_after["reprojection_displacement_px"]
        ca = result_after["contact_after_shift"]
        cd = result_after.get("completed_mesh_unsigned_dist_after_shift_m") or {}
        lines.append(f"reproj disp: med {rp['median']:.1f} max {rp['max']:.1f} px"
                     if rp.get("median") is not None else "reproj disp: n/a")
        lines.append(f"contact_candidate after: {ca['contact_candidate']}  in-band={ca['in_soft_tissue_band_count']}")
        if cd:
            lines.append(f"completed mesh dist after: med {cd.get('median')*1000:.1f} mm" if cd.get('median') else "completed mesh dist after: n/a")
    y = 36
    for ln in lines:
        cv2.putText(overlay, ln[:130], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument("--frames", type=int, nargs="*", default=DEFAULT_FRAMES)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/tmp/clip001850_hand_depth_bias_counterfactual"))
    ap.add_argument("--hand-side", default="right")
    ap.add_argument("--no-mesh", action="store_true",
                    help="skip completed-mesh distance (faster)")
    args = ap.parse_args()

    R = args.run_root
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cf] run_root={R}", flush=True)
    print(f"[cf] frames={args.frames}", flush=True)

    ann_path = R / "measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json"
    mano_path = R / ("measurements/mano_interval_correction/keyboard_0_149/"
                     "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/"
                     "v18_joint_mano_interval_trajectory_state.json")
    pose_path = R / "measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json"
    depth_npz = R / "measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz"
    completed_mesh_path = R / ("measurements/geometry_completion/compact_keyboard_seed42/"
                               "keyboard_compact_rigid_completed_mesh_labeled.ply")
    kb_mask_dir = R / "measurements/object_tracks/sam2_agent_points/keyboard/sam2/sam2_masks"
    own_dir = (R / "measurements/contact_visibility_factors/keyboard_0_149/"
               "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/ownership_masks")
    rgb_dir = R / "measurements/depth_slam/unidepth_full_frame/stills"

    ann = load_json(ann_path)
    ann_frames = {f["frame_idx"]: f for f in ann["frames"] if isinstance(f, dict)}
    mano = load_json(mano_path)
    mano_by_frame = {r["frame_idx"]: r for r in mano["per_frame_states"]
                     if r.get("hand_side") == args.hand_side}
    pose_rep = load_json(pose_path)
    pose_rows = {r["frame_idx"]: r for r in pose_rep["pose_rows"]}
    fi, depth_arr = (lambda nz: (np.asarray(nz["frame_idx"], dtype=np.int64),
                                  np.asarray(nz["depth"], dtype=np.float32)))(np.load(depth_npz))

    completed_tree = None
    if not args.no_mesh and completed_mesh_path.exists():
        import trimesh
        print("[cf] loading completed mesh (CPU)...", flush=True)
        mesh = trimesh.load(str(completed_mesh_path), process=False)
        completed_tree = mesh
        print(f"[cf] completed mesh verts={len(mesh.vertices)} faces={len(mesh.faces)} "
              f"watertight={mesh.is_watertight}", flush=True)

    frames = [f for f in args.frames]
    rows = []
    for f in frames:
        fd = build_frame(R, f, ann_frames, mano_by_frame, fi, depth_arr,
                         kb_mask_dir, own_dir, args.hand_side)
        if fd is None:
            print(f"[cf] f{f}: missing data, skipping", flush=True)
            continue
        # fill object pose for mesh context
        pr = pose_rows.get(f)
        if pr is not None:
            fd.R_obj = np.asarray(pr.get("rotation_world_from_completed_canonical_matrix",
                                         np.eye(3)), dtype=float)
            fd.t_obj = np.asarray(pr.get("translation_world_m", np.zeros(3)), dtype=float)
            fd.pose_source = (pr.get("temporal_pose_graph") or {}).get("pose_source", "")

        row = per_frame_counterfactual(fd, completed_tree)
        rows.append(row)
        d = row["decision"]
        print(f"  f{f:3d}  required_shift={_fmt(row['representative_required_shift_m'])} m  "
              f"gap_voo_med={_fmt((row['gap_delta_tip_voo_summary_m'] or {}).get('median'))}  "
              f"contact_after={d['creates_contact_candidate']}  "
              f"reproj_max={_fmt(d.get('reprojection_max_px'))}px  "
              f"verdict={d['verdict']}", flush=True)

    # write ndjson
    nd = out / "per_frame.ndjson"
    with open(nd, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[cf] wrote {nd} ({len(rows)} rows)", flush=True)

    # summary / decision
    decided = [r for r in rows if r["decision"]["verdict"] != "no_overlap_fingertip_keyboard_masked"]
    shifts = [r["representative_required_shift_m"] for r in decided
              if r["representative_required_shift_m"] is not None]
    contact_made = [r for r in decided if r["decision"]["creates_contact_candidate"]]
    proj_ok = [r for r in decided if r["decision"]["projection_within_tolerance"]]

    overall = {
        "case": "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1",
        "run_root": str(R),
        "frames": frames,
        "hand_side": args.hand_side,
        "parameters": {
            "soft_tissue_band_m": SOFT_TISSUE_BAND_M,
            "sigma_clip_m": list(SIGMA_CLIP_M),
            "projection_tolerance_px": PROJECTION_TOLERANCE_PX,
        },
        "n_frames_decided": len(decided),
        "required_shift_summary_m": numeric_summary(shifts) if shifts else {"count": 0},
        "frames_creating_contact_candidate": [r["frame_idx"] for r in contact_made],
        "frames_projection_within_tolerance": [r["frame_idx"] for r in proj_ok],
        "per_frame": [{"frame_idx": r["frame_idx"], "decision": r["decision"]}
                      for r in rows],
    }

    # overall verdict
    if shifts:
        med = float(np.median(shifts))
        smax = SIGMA_CLIP_M[1]
        ratio = med / smax
        if len(contact_made) == len(decided) and len(proj_ok) == len(decided):
            overall_verdict = (
                f"~{med*1000:.0f} mm along-ray correction is plausible (ratio to "
                f"sigma_clip_max {ratio:.1f}x) AND creates contact_candidate on all "
                f"decided frames without exceeding the existing projection tolerance.")
        elif len(contact_made) == len(decided):
            overall_verdict = (
                f"~{med*1000:.0f} mm correction creates contact_candidate but worsens "
                f"projection beyond the existing tolerance on "
                f"{len(decided)-len(proj_ok)}/{len(decided)} frames: the bias hypothesis "
                f"would trade contact for a reprojection error larger than the interval "
                f"solver already tolerated.")
        else:
            overall_verdict = (
                f"~{med*1000:.0f} mm correction does not uniformly produce "
                f"contact_candidate: the bias hypothesis is insufficient or the required "
                f"shift varies frame-to-frame beyond a single systematic offset.")
    else:
        overall_verdict = "no fingertip/keyboard-mask overlap on any decided frame."
    overall["overall_verdict"] = overall_verdict

    sp = out / "summary.json"
    with open(sp, "w") as fh:
        json.dump(overall, fh, indent=2)
    print(f"[cf] wrote {sp}", flush=True)
    print("\n===== COUNTERFACTUAL HEADLINE =====", flush=True)
    print(overall_verdict, flush=True)

    # review images for f32, f36, f45
    for f in (32, 36, 45):
        fd = build_frame(R, f, ann_frames, mano_by_frame, fi, depth_arr,
                         kb_mask_dir, own_dir, args.hand_side)
        if fd is None:
            print(f"[cf] f{f}: no data for review image", flush=True)
            continue
        ov = compute_overlap(fd)
        row = next((r for r in rows if r["frame_idx"] == f), None)
        rs = row["representative_required_shift_m"] if row else None
        ra = row["counterfactual_result"] if row else None
        rgb = rgb_dir / f"frame_{f:06d}.png"
        out_jpg = out / f"review_f{f}.jpg"
        title = f"f{f} right  keyboard-masked(green) voo(cyan)  hand[orig=cyan/tip=magenta -> shift=red]"
        try:
            make_review(fd, ov, rs, ra, rgb, out_jpg, title)
            print(f"[cf] wrote {out_jpg}", flush=True)
        except Exception as e:
            print(f"[cf] f{f} review image failed: {e}", flush=True)

    return 0


def _fmt(x):
    if x is None:
        return "  n/a"
    if isinstance(x, (int, float)):
        return f"{x:+.4f}" if abs(x) < 10 else f"{x:.2f}"
    return str(x)


if __name__ == "__main__":
    sys.exit(main())
