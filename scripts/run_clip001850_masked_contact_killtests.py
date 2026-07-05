#!/usr/bin/env python3
"""CPU-only masked-depth contact kill-tests (KT-1/KT-2/KT-3/KT-5) for HOT3D
clip001850 keyboard, right hand, frames 28-48.

Run root:
  /data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1

KT-1: recompute hand-vertex -> depth-map penetration restricted to (a) full-frame
      depth, (b) keyboard-masked depth, (c) keyboard-masked + hand-quarantined
      depth, plus a table/background-plane alternative. Proxy masks are marked.
KT-2: per-vertex penetrating sample-IDs + temporal IoU + anatomical region
      (no .max scalar as coherence).
KT-3: at f36, fingertip sample vertices vs full-frame depth, keyboard-masked
      depth, and completed mesh under one pose.
KT-5: completed-mesh distance / candidate counts stratified by pose provenance
      (fit vs missing/interpolated).

Outputs (under --out-dir, default /tmp/clip001850_masked_contact_killtests):
  contact_killtests_frame_detail.ndjson
  summary.json
  review_f32.jpg, review_f36.jpg   (if RGB/masks available)

No model inference, no GPU. Reads only on-disk v19 artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

RUN_ROOT_DEFAULT = Path(
    "/data2/ego_annotation_outputs/v19_runs/"
    "20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)

# Soft-tissue / contact band (meters). Within this |delta| counts as "near".
SOFT_TISSUE_BAND_M = 0.015
# Penetration threshold along camera ray (hand behind surface by more than this).
PENETRATION_MARGIN_M = 0.005
# Keyboard prior extents (m): long, short, thin sorted ascending.
KEYBOARD_PRIOR_EXTENTS_M = (0.03, 0.15, 0.45)

FRAME_LO_DEFAULT = 28
FRAME_HI_DEFAULT = 48
HAND_SIDE_DEFAULT = "right"
KT3_FRAME = 36


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _remount(p: str | Path) -> Path:
    """Map /mnt/truenas-user-home/... paths stored in artifacts to the local
    /data2/... mount when the truenas path is not visible."""
    p = Path(p)
    if p.exists():
        return p
    s = str(p)
    for prefix in (
        "/mnt/truenas-user-home",
        "/mnt/user-home",
    ):
        if s.startswith(prefix):
            local = Path("/data2" + s[len(prefix):])
            if local.exists():
                return local
            # ego_annotation_outputs lives directly under /data2
            local2 = Path("/data2" + s[len(prefix):])
            if local2.exists():
                return local2
    return p


def load_json(path: Path) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def load_depth_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    nz = np.load(path)
    frame_idx = np.asarray(nz["frame_idx"], dtype=np.int64)
    depth = np.asarray(nz["depth"], dtype=np.float32)  # (N, H, W)
    return frame_idx, depth


def load_mask_1408(path: Path) -> np.ndarray | None:
    """Load a boolean mask PNG and resize to 1408x1408 (nearest neighbour)."""
    path = _remount(path)
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = arr > 0
    if mask.shape != (1408, 1408):
        mask = cv2.resize(
            mask.astype(np.uint8), (1408, 1408), interpolation=cv2.INTER_NEAREST
        ) > 0
    return mask


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
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


# --------------------------------------------------------------------------- #
# Per-frame evidence bundle
# --------------------------------------------------------------------------- #
@dataclass
class FrameEvidence:
    frame_idx: int
    # MANO sample vertices (world) and their MANO sample ids
    verts_world: np.ndarray              # (S, 3)
    sample_ids: np.ndarray               # (S,)
    joints_world: np.ndarray             # (21, 3)
    # Camera
    R_c2w: np.ndarray                    # (3, 3)
    t_c2w: np.ndarray                    # (3,)
    intr: tuple[float, float, float, float]
    # Depth row
    depth: np.ndarray                    # (1408, 1408)
    # Masks (1408x1408 bool) or None
    keyboard_mask: np.ndarray | None
    keyboard_mask_source: str            # 'sam2' | 'proxy_temporal_hold' | 'missing'
    hand_support_mask: np.ndarray | None
    visible_object_owned_mask: np.ndarray | None
    mask_files_exist: bool
    # Published interval-solver numbers (for comparison)
    interval_full_penetration: dict | None
    interval_contact_patch_ids: list[int]
    interval_contact_patch_gap: dict | None
    interval_depth_order_selected_count: int
    # Pose
    object_rotation_world_from_canonical: np.ndarray   # (3, 3)
    object_translation_world: np.ndarray               # (3,)
    pose_source: str
    pose_direct_visible: bool
    pose_gap_frames: int


# --------------------------------------------------------------------------- #
# Mask resolution (KT-1 proxy logic)
# --------------------------------------------------------------------------- #
def resolve_masks(
    frame_idx: int,
    sam2_track: dict,
    sam2_mask_dir: Path,
    ownership_dir: Path,
    hand_side: str,
    available_mask_frames: list[int],
) -> tuple[np.ndarray | None, str, np.ndarray | None, np.ndarray | None, bool]:
    """Return (keyboard_mask_1408, keyboard_mask_source, hand_support_mask,
    visible_object_owned_mask, mask_files_exist)."""
    fname = f"{frame_idx:06d}.png"
    sam2_path = sam2_mask_dir / fname
    voo_path = ownership_dir / hand_side / f"{frame_idx:06d}_visible_object_owned.png"
    pms_path = ownership_dir / hand_side / f"{frame_idx:06d}_projected_mano_hand_support.png"

    kb_exists = sam2_path.exists()
    own_exists = voo_path.exists() and pms_path.exists()
    mask_files_exist = kb_exists and own_exists

    if mask_files_exist:
        kb = load_mask_1408(sam2_path)
        voo = load_mask_1408(voo_path)
        pms = load_mask_1408(pms_path)
        return kb, "sam2", pms, voo, True

    # No direct mask -> derive strongest available proxy.
    # 1) Try temporal-hold of nearest available SAM2 keyboard mask.
    if available_mask_frames:
        nearest = min(available_mask_frames, key=lambda f: abs(f - frame_idx))
        hold_path = sam2_mask_dir / f"{nearest:06d}.png"
        if hold_path.exists():
            kb_hold = load_mask_1408(hold_path)
            # ownership masks for the held frame (hand has moved, so these are weak)
            hold_voo = ownership_dir / hand_side / f"{nearest:06d}_visible_object_owned.png"
            hold_pms = ownership_dir / hand_side / f"{nearest:06d}_projected_mano_hand_support.png"
            voo = load_mask_1408(hold_voo) if hold_voo.exists() else None
            pms = load_mask_1408(hold_pms) if hold_pms.exists() else None
            return kb_hold, f"proxy_temporal_hold_f{nearest}", pms, voo, False

    # 2) Fall back to the SAM2 track bbox as a coarse proxy region.
    entry = sam2_track.get(str(frame_idx))
    if entry and entry.get("visible") and entry.get("bbox_xyxy"):
        bbox = entry["bbox_xyxy"]
        # bbox is in the 1408 grid (track was authored against full frames)
        x0, y0, x1, y1 = [float(c) for c in bbox]
        kb = np.zeros((1408, 1408), dtype=bool)
        kb[int(y0):int(y1), int(x0):int(x1)] = True
        return kb, "proxy_track_bbox", None, None, False

    return None, "missing", None, None, False


# --------------------------------------------------------------------------- #
# Projection + depth sampling
# --------------------------------------------------------------------------- #
def project_and_sample(
    verts_world: np.ndarray,
    R_c2w: np.ndarray,
    t_c2w: np.ndarray,
    intr: tuple[float, float, float, float],
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (cam_z, u, v, valid_pixel). depth sampled at (u,v) on valid."""
    fx, fy, cx, cy = intr
    cam = (verts_world - t_c2w[None, :]) @ R_c2w
    cam_z = cam[:, 2]
    zsafe = np.maximum(cam_z, 1e-9)
    u = fx * cam[:, 0] / zsafe + cx
    v = fy * cam[:, 1] / zsafe + cy
    ur = np.rint(u).astype(int)
    vr = np.rint(v).astype(int)
    H, W = depth.shape
    inbounds = (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H) & (cam_z > 1e-5)
    surf = np.full(verts_world.shape[0], np.nan, dtype=float)
    surf[inbounds] = depth[vr[inbounds], ur[inbounds]].astype(float)
    return cam_z, ur, vr, surf


def penetration_along_ray(cam_z: np.ndarray, surf: np.ndarray) -> np.ndarray:
    """Positive = hand behind surface (penetrating). NaN where no surface."""
    delta = cam_z - surf
    return delta


# --------------------------------------------------------------------------- #
# Table / background plane fit
# --------------------------------------------------------------------------- #
def fit_table_plane(
    depth: np.ndarray,
    keyboard_mask: np.ndarray | None,
    hand_mask: np.ndarray | None,
    intr: tuple[float, float, float, float],
    z_lo: float = 0.35,
    z_hi: float = 1.20,
    max_samples: int = 6000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Fit a plane (normal, point) to background depth pixels (non-keyboard,
    non-hand). Returns (normal(3,), centroid(3,), eigvals, n_used) or None.

    Backprojects depth pixels to camera-frame 3D using intrinsics."""
    fx, fy, cx, cy = intr
    H, W = depth.shape
    finite = np.isfinite(depth) & (depth > z_lo) & (depth < z_hi)
    if keyboard_mask is not None:
        finite &= ~keyboard_mask
    if hand_mask is not None:
        finite &= ~hand_mask
    ys, xs = np.where(finite)
    if xs.size < 50:
        return None
    rng = np.random.default_rng(seed)
    if xs.size > max_samples:
        sel = rng.choice(xs.size, max_samples, replace=False)
        xs = xs[sel]
        ys = ys[sel]
    z = depth[ys, xs].astype(float)
    x_cam = (xs.astype(float) - cx) * z / fx
    y_cam = (ys.astype(float) - cy) * z / fy
    pts = np.stack([x_cam, y_cam, z], axis=1)
    centroid = pts.mean(axis=0)
    centered = pts - centroid[None, :]
    # SVD plane fit
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    eigvals = np.linalg.svd(centered, compute_uv=False)
    return normal, centroid, eigvals, int(xs.size)


def signed_dist_to_plane(verts_world: np.ndarray, R_c2w: np.ndarray,
                         t_c2w: np.ndarray, normal: np.ndarray,
                         centroid: np.ndarray) -> np.ndarray:
    """Signed distance of hand vertices (world) to the table plane (camera frame).
    Plane defined in camera frame. Positive = above plane (in front, toward camera)
    when normal points toward camera (+z)."""
    cam = (verts_world - t_c2w[None, :]) @ R_c2w
    d = (cam - centroid[None, :]) @ normal
    return d


# --------------------------------------------------------------------------- #
# Anatomical region classification
# --------------------------------------------------------------------------- #
MANO_FINGERTIP_JOINTS = {4, 8, 12, 16, 20}     # thumb tip, index..pinky tips
MANO_PALM_JOINTS = {0}                          # wrist
MANO_THUMB_JOINTS = {1, 2, 3, 4}
MANO_INDEX_JOINTS = {5, 6, 7, 8}
MANO_MIDDLE_JOINTS = {9, 10, 11, 12}
MANO_RING_JOINTS = {13, 14, 15, 16}
MANO_PINKY_JOINTS = {17, 18, 19, 20}


def classify_anatomical(verts_world: np.ndarray,
                        joints_world: np.ndarray) -> list[str]:
    """Assign each vertex to nearest joint's finger region label."""
    # nearest joint
    d = np.linalg.norm(verts_world[:, None, :] - joints_world[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    labels = []
    for j in nearest:
        if j in MANO_THUMB_JOINTS:
            labels.append("thumb")
        elif j in MANO_INDEX_JOINTS:
            labels.append("index")
        elif j in MANO_MIDDLE_JOINTS:
            labels.append("middle")
        elif j in MANO_RING_JOINTS:
            labels.append("ring")
        elif j in MANO_PINKY_JOINTS:
            labels.append("pinky")
        else:
            labels.append("palm")
    return labels


def fingertip_vertex_mask(labels: list[str]) -> np.ndarray:
    is_tip = np.zeros(len(labels), dtype=bool)
    for i, lab in enumerate(labels):
        j = -1
        # a fingertip vertex is one whose nearest joint is a tip joint
        is_tip[i] = lab in {"thumb", "index", "middle", "ring", "pinky"} and False
    return is_tip


def fingertip_vertex_indices(verts_world: np.ndarray,
                             joints_world: np.ndarray) -> np.ndarray:
    """Vertices whose nearest joint is one of the 5 fingertip joints."""
    d = np.linalg.norm(verts_world[:, None, :] - joints_world[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    return np.where(np.isin(nearest, np.asarray(sorted(MANO_FINGERTIP_JOINTS))))[0]


# --------------------------------------------------------------------------- #
# Mesh distance helpers (KT-3 / KT-5)
# --------------------------------------------------------------------------- #
def closest_point_distances_canonical(
    verts_world: np.ndarray,
    R_obj: np.ndarray,
    t_obj: np.ndarray,
    mesh: trimesh.Trimesh,
    tree=None,
) -> np.ndarray:
    """Transform world verts -> object canonical frame, query mesh distance."""
    verts_obj = (verts_world - t_obj[None, :]) @ R_obj
    if tree is None:
        tree = mesh
    closest, dist, _ = trimesh.proximity.closest_point(tree, verts_obj)
    return np.asarray(dist, dtype=float), np.asarray(closest, dtype=float)


def best_effort_signed_distance_canonical(
    verts_world: np.ndarray,
    R_obj: np.ndarray,
    t_obj: np.ndarray,
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    """Best-effort signed distance. NOTE: unreliable on non-watertight meshes;
    reported only to expose the sign-channel failure."""
    verts_obj = (verts_world - t_obj[None, :]) @ R_obj
    try:
        sd = trimesh.proximity.signed_distance(mesh, verts_obj)
        # trimesh signed_distance: +inside, -outside. Penetration = inside.
        return np.asarray(sd, dtype=float)
    except Exception:
        return np.full(verts_obj.shape[0], np.nan, dtype=float)


# --------------------------------------------------------------------------- #
# Main per-frame KT-1/KT-2
# --------------------------------------------------------------------------- #
def run_frame(ev: FrameEvidence, labels: list[str]) -> dict[str, Any]:
    cam_z, ur, vr, surf = project_and_sample(
        ev.verts_world, ev.R_c2w, ev.t_c2w, ev.intr, ev.depth
    )
    delta = penetration_along_ray(cam_z, surf)  # +behind
    finite = np.isfinite(delta)

    H, W = ev.depth.shape
    inbounds = (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H)

    # ---- KT-1: three mask conditions ----
    def condition(mask: np.ndarray | None, require_hand_quarantine_mask: bool):
        if mask is None:
            return {
                "available": False,
                "missing_field": (
                    "keyboard_mask" if not require_hand_quarantine_mask
                    else "visible_object_owned_mask"
                ),
            }
        pix_ok = inbounds & finite & mask[vr, ur]
        d = delta[pix_ok]
        cz = cam_z[pix_ok]
        sf = surf[pix_ok]
        # penetration = hand behind surface beyond margin
        penetrating = d > PENETRATION_MARGIN_M
        near_band = np.abs(d) <= SOFT_TISSUE_BAND_M
        ids = ev.sample_ids[pix_ok][penetrating].tolist()
        near_ids = ev.sample_ids[pix_ok][near_band].tolist()
        return {
            "available": True,
            "n_vertices_in_region": int(pix_ok.sum()),
            "delta_summary_m": numeric_summary(d),
            "penetrating_vertex_count": int(penetrating.sum()),
            "penetrating_depth_summary_m": numeric_summary(d[penetrating]),
            "penetrating_sample_ids": ids,
            "near_band_vertex_count": int(near_band.sum()),
            "near_band_sample_ids": near_ids,
            "hand_cam_z_summary_m": numeric_summary(cz),
            "surface_depth_summary_m": numeric_summary(sf),
        }

    full = condition(np.ones((H, W), dtype=bool), False)
    kb = condition(ev.keyboard_mask, False) if ev.keyboard_mask is not None else {
        "available": False, "missing_field": "keyboard_mask",
        "keyboard_mask_source": ev.keyboard_mask_source,
    }
    if ev.keyboard_mask is not None:
        kb["keyboard_mask_source"] = ev.keyboard_mask_source
    voo = condition(ev.visible_object_owned_mask, True) if ev.visible_object_owned_mask is not None else {
        "available": False, "missing_field": "visible_object_owned_mask",
    }
    if ev.visible_object_owned_mask is not None:
        voo["keyboard_mask_source"] = ev.keyboard_mask_source
        voo["hand_quarantined"] = True

    # ---- Table / background plane ----
    table = fit_table_plane(ev.depth, ev.keyboard_mask, ev.hand_support_mask, ev.intr)
    table_block = {"available": False}
    if table is not None:
        normal, centroid, eigvals, n_used = table
        sd = signed_dist_to_plane(ev.verts_world, ev.R_c2w, ev.t_c2w, normal, centroid)
        # normal points toward camera (+z); positive sd => hand on camera side of
        # plane (in front of table/background); negative-large => hand well in
        # front; positive beyond margin => hand BEHIND plane (penetrating table).
        behind_plane = sd > PENETRATION_MARGIN_M
        in_front = sd < -PENETRATION_MARGIN_M
        table_block = {
            "available": True,
            "n_plane_samples": n_used,
            "plane_normal_camera": normal.tolist(),
            "plane_centroid_camera_m": centroid.tolist(),
            "signed_dist_summary_m": numeric_summary(sd),
            "in_front_of_plane_vertex_count": int(in_front.sum()),
            "behind_plane_vertex_count": int(behind_plane.sum()),
            "behind_plane_sample_ids": ev.sample_ids[behind_plane].tolist(),
        }

    # ---- KT-2: per-vertex penetrating IDs + anatomical region ----
    # Use the keyboard-masked + hand-quarantined condition as the primary
    # contact candidate; fall back to keyboard-masked if voo unavailable.
    primary = voo if voo.get("available") else kb
    kt2 = {
        "primary_condition": "keyboard_masked_hand_quarantined" if voo.get("available")
        else ("keyboard_masked" if kb.get("available") else "none"),
        "penetrating_sample_ids": primary.get("penetrating_sample_ids", []) if primary.get("available") else [],
        "near_band_sample_ids": primary.get("near_band_sample_ids", []) if primary.get("available") else [],
    }
    # anatomical region of penetrating ids
    sid_to_label = {int(sid): labels[i] for i, sid in enumerate(ev.sample_ids)}
    pen_labels = [sid_to_label.get(int(s), "?") for s in kt2["penetrating_sample_ids"]]
    from collections import Counter
    kt2["penetrating_anatomical_region_counts"] = dict(Counter(pen_labels))
    kt2["forbidden_max_scalar_as_coherence"] = True

    # ---- Route decision ----
    route = route_frame(ev, full, kb, voo, table_block, primary)
    kt2["route"] = route

    # ---- Published interval comparison ----
    interval_block = {
        "interval_full_observed_surface_penetration_m": ev.interval_full_penetration,
        "interval_contact_patch_vertex_ids": ev.interval_contact_patch_ids,
        "interval_contact_patch_gap_m": ev.interval_contact_patch_gap,
        "interval_depth_order_selected_vertex_count": ev.interval_depth_order_selected_count,
    }

    return {
        "frame_idx": ev.frame_idx,
        "hand_side": "right",
        "pose_source": ev.pose_source,
        "pose_direct_visible": ev.pose_direct_visible,
        "pose_gap_frames": ev.pose_gap_frames,
        "mask_files_exist": ev.mask_files_exist,
        "keyboard_mask_source": ev.keyboard_mask_source,
        "KT1_full_frame_depth": full,
        "KT1_keyboard_masked_depth": kb,
        "KT1_keyboard_masked_hand_quarantined_depth": voo,
        "KT1_table_background_plane": table_block,
        "KT2_vertex_coherence": kt2,
        "interval_solver_published": interval_block,
        "route": route,
    }


def route_frame(ev: FrameEvidence, full: dict, kb: dict, voo: dict,
                table: dict, primary: dict) -> str:
    """Decide the route label for this frame/subinterval.

    Routes:
      contact_candidate_keyboard_masked : masked+quarantined depth shows hand
                                           within/near or just behind keyboard
                                           surface (real contact evidence).
      full_frame_depth_leak             : full-frame penetration large but masked
                                           collapses -> raw barrier contaminated
                                           by non-keyboard (hand/table) surface.
      pose_unresolved                   : missing/interpolated object pose; distance
                                           untrustworthy.
      geometry_epoch_contaminated       : masked depth available but shows hand in
                                           front of keyboard (gap / no penetration);
                                           the interval solver's penetration claim
                                           rests on the inflated non-watertight
                                           completed mesh, uncorroborated by depth.
      evidence_missing                  : no mask and no proxy derivable.
    """
    # evidence missing entirely
    if ev.keyboard_mask is None and ev.keyboard_mask_source == "missing":
        return "evidence_missing"
    # pose unresolved dominates for missing/interpolated frames without direct mask
    if not ev.pose_direct_visible and not ev.mask_files_exist:
        return "pose_unresolved"

    fp = ((full.get("penetrating_depth_summary_m") or {}).get("max") or 0.0) if full.get("available") else 0.0
    fcnt = full.get("penetrating_vertex_count", 0) if full.get("available") else 0

    if primary.get("available"):
        mp = ((primary.get("penetrating_depth_summary_m") or {}).get("max") or 0.0)
        mcnt = primary.get("penetrating_vertex_count", 0)
        near = primary.get("near_band_vertex_count", 0)
        n_region = primary.get("n_vertices_in_region", 0)
        # real contact evidence: hand within soft-tissue band of keyboard surface
        # or modestly behind it, restricted to masked+quarantined pixels.
        if n_region > 0 and (near > 0 or (mcnt > 0 and mp <= 0.02)):
            return "contact_candidate_keyboard_masked"
        # full-frame leak: raw barrier large but masked collapses to nothing.
        if fp > 0.02 and fcnt > 5 and (mcnt == 0 or mp < 0.3 * fp):
            return "full_frame_depth_leak"
        # masked depth measured and shows hand clearly in front of keyboard (gap):
        # the published mesh-penetration is uncorroborated by depth -> the
        # penetration rests on the contaminated inflated mesh geometry epoch.
        if n_region > 0:
            return "geometry_epoch_contaminated"
    else:
        # no masked depth available; only completed-mesh distance (KT5/KT3)
        return "geometry_epoch_contaminated"

    # proxy-mask frames default to pose_unresolved (hand moved, held mask weak)
    if not ev.mask_files_exist:
        return "pose_unresolved"
    return "evidence_missing"


# --------------------------------------------------------------------------- #
# KT-2 temporal IoU across the window
# --------------------------------------------------------------------------- #
def temporal_coherence(rows: list[dict]) -> dict[str, Any]:
    """Compute temporal IoU of penetrating sample-id sets across consecutive frames."""
    sets = []
    frames = []
    for r in rows:
        kt2 = r["KT2_vertex_coherence"]
        s = set(int(x) for x in kt2.get("penetrating_sample_ids", []))
        sets.append(s)
        frames.append(r["frame_idx"])
    ious = []
    pairs = []
    for i in range(len(sets) - 1):
        a, b = sets[i], sets[i + 1]
        if not a and not b:
            ious.append(None)
            pairs.append((frames[i], frames[i + 1], None, 0, 0))
            continue
        inter = len(a & b)
        union = len(a | b)
        iou = inter / union if union else None
        ious.append(iou)
        pairs.append((frames[i], frames[i + 1], iou, len(a), len(b)))
    valid = [x for x in ious if x is not None]
    # also near-band coherence
    near_sets = []
    for r in rows:
        kt2 = r["KT2_vertex_coherence"]
        near_sets.append(set(int(x) for x in kt2.get("near_band_sample_ids", [])))
    near_ious = []
    for i in range(len(near_sets) - 1):
        a, b = near_sets[i], near_sets[i + 1]
        if not a and not b:
            continue
        union = len(a | b)
        near_ious.append(len(a & b) / union if union else None)
    near_valid = [x for x in near_ious if x is not None]
    # union of all penetrating ids
    union_all = set()
    for s in sets:
        union_all |= s
    return {
        "consecutive_iou_pairs": [
            {"f0": p[0], "f1": p[1], "iou": p[2], "set0_size": p[3], "set1_size": p[4]}
            for p in pairs
        ],
        "mean_iou": float(np.mean(valid)) if valid else None,
        "near_band_mean_iou": float(np.mean(near_valid)) if near_valid else None,
        "penetrating_id_union_size": len(union_all),
        "penetrating_id_union": sorted(union_all),
        "interpretation": interpret_coherence(valid, pairs),
    }


def interpret_coherence(ious: list[float], pairs) -> str:
    if not ious:
        return "no_penetrating_sets"
    mean = float(np.mean(ious))
    # count flicker (0 -> large -> 0)
    flicker = 0
    for p in pairs:
        if p[2] is not None:
            if (p[3] == 0 and p[4] > 20) or (p[3] > 20 and p[4] == 0):
                flicker += 1
    if mean >= 0.4 and flicker == 0:
        return "compact_persistent_set"
    if flicker >= 2:
        return "flickering_incoherent_set"
    if mean < 0.2:
        return "scattered_low_coherence"
    return "intermittent_moderate"


# --------------------------------------------------------------------------- #
# KT-3: f36 fingertip deep dive
# --------------------------------------------------------------------------- #
def run_kt3(ev: FrameEvidence, completed_mesh: trimesh.Trimesh,
            observed_mesh: trimesh.Trimesh, labels: list[str]) -> dict[str, Any]:
    tip_idx = fingertip_vertex_indices(ev.verts_world, ev.joints_world)
    tip_verts = ev.verts_world[tip_idx]
    tip_ids = ev.sample_ids[tip_idx]
    tip_labels = [labels[i] for i in tip_idx]

    # (a) full-frame depth along ray
    cam_z, ur, vr, surf = project_and_sample(
        tip_verts, ev.R_c2w, ev.t_c2w, ev.intr, ev.depth
    )
    delta_full = cam_z - surf

    # (b) keyboard-masked depth along ray
    delta_kb = delta_full.copy()
    if ev.keyboard_mask is not None:
        okpix = ev.keyboard_mask[vr, ur] if np.all((ur >= 0) & (ur < 1408) & (vr >= 0) & (vr < 1408)) else np.zeros(len(ur), dtype=bool)
        delta_kb = np.where(okpix, delta_full, np.nan)
    else:
        delta_kb = np.full(len(tip_verts), np.nan)

    # (b') keyboard-masked + hand-quarantined
    delta_voo = delta_full.copy()
    if ev.visible_object_owned_mask is not None:
        okpix = ev.visible_object_owned_mask[vr, ur] if np.all((ur >= 0) & (ur < 1408) & (vr >= 0) & (vr < 1408)) else np.zeros(len(ur), dtype=bool)
        delta_voo = np.where(okpix, delta_full, np.nan)
    else:
        delta_voo = np.full(len(tip_verts), np.nan)

    # (c) completed mesh (canonical frame)
    dist_c, _ = closest_point_distances_canonical(
        tip_verts, ev.object_rotation_world_from_canonical,
        ev.object_translation_world, completed_mesh
    )
    sd_c = best_effort_signed_distance_canonical(
        tip_verts, ev.object_rotation_world_from_canonical,
        ev.object_translation_world, completed_mesh
    )

    # (d) observed surface mesh (canonical frame)
    dist_o = np.full(len(tip_verts), np.nan)
    if observed_mesh is not None and len(observed_mesh.faces) > 0:
        dist_o, _ = closest_point_distances_canonical(
            tip_verts, ev.object_rotation_world_from_canonical,
            ev.object_translation_world, observed_mesh
        )

    per_vertex = []
    for i in range(len(tip_verts)):
        per_vertex.append({
            "sample_id": int(tip_ids[i]),
            "region": tip_labels[i],
            "cam_z_m": float(cam_z[i]),
            "depth_full_m": float(surf[i]) if np.isfinite(surf[i]) else None,
            "delta_full_m": float(delta_full[i]) if np.isfinite(delta_full[i]) else None,
            "delta_keyboard_masked_m": float(delta_kb[i]) if np.isfinite(delta_kb[i]) else None,
            "delta_keyboard_masked_hand_quarantined_m": float(delta_voo[i]) if np.isfinite(delta_voo[i]) else None,
            "completed_mesh_unsigned_m": float(dist_c[i]),
            "completed_mesh_best_effort_signed_m": float(sd_c[i]) if np.isfinite(sd_c[i]) else None,
            "observed_mesh_unsigned_m": float(dist_o[i]) if np.isfinite(dist_o[i]) else None,
        })

    return {
        "frame_idx": KT3_FRAME,
        "fingertip_vertex_count": int(len(tip_verts)),
        "completed_mesh_watertight": bool(completed_mesh.is_watertight),
        "observed_mesh_watertight": bool(observed_mesh.is_watertight) if observed_mesh is not None else None,
        "delta_full_summary_m": numeric_summary(delta_full[np.isfinite(delta_full)]),
        "delta_keyboard_masked_summary_m": numeric_summary(delta_kb[np.isfinite(delta_kb)]),
        "delta_keyboard_masked_hand_quarantined_summary_m": numeric_summary(delta_voo[np.isfinite(delta_voo)]),
        "completed_mesh_unsigned_summary_m": numeric_summary(dist_c),
        "observed_mesh_unsigned_summary_m": numeric_summary(dist_o[np.isfinite(dist_o)]),
        "within_soft_tissue_band_count": {
            "full_frame": int(np.sum(np.isfinite(delta_full) & (np.abs(delta_full) <= SOFT_TISSUE_BAND_M))),
            "keyboard_masked": int(np.sum(np.isfinite(delta_kb) & (np.abs(delta_kb) <= SOFT_TISSUE_BAND_M))),
            "completed_mesh_unsigned": int(np.sum(dist_c <= SOFT_TISSUE_BAND_M)),
        },
        "per_fingertip_vertex": per_vertex,
    }


# --------------------------------------------------------------------------- #
# KT-5: completed-mesh distance stratified by pose provenance
# --------------------------------------------------------------------------- #
def run_kt5(evs: list[FrameEvidence], completed_mesh: trimesh.Trimesh) -> dict[str, Any]:
    fit_rows = []
    missing_rows = []
    for ev in evs:
        verts_obj = (ev.verts_world - ev.object_translation_world[None, :]) @ \
                    ev.object_rotation_world_from_canonical
        _, dist, _ = trimesh.proximity.closest_point(completed_mesh, verts_obj)
        sd = best_effort_signed_distance_canonical(
            ev.verts_world, ev.object_rotation_world_from_canonical,
            ev.object_translation_world, completed_mesh
        )
        inside = sd > 0  # trimesh: +inside
        row = {
            "frame_idx": ev.frame_idx,
            "pose_source": ev.pose_source,
            "pose_direct_visible": ev.pose_direct_visible,
            "completed_unsigned_dist_median_m": float(np.median(dist)),
            "completed_unsigned_dist_p95_m": float(np.percentile(dist, 95)),
            "completed_unsigned_dist_max_m": float(np.max(dist)),
            "completed_best_effort_inside_count": int(np.sum(np.isfinite(sd) & inside)),
            "completed_best_effort_inside_median_m": float(np.median(sd[inside])) if np.any(np.isfinite(sd) & inside) else None,
            "interval_depth_order_selected_count": ev.interval_depth_order_selected_count,
        }
        if ev.pose_direct_visible:
            fit_rows.append(row)
        else:
            missing_rows.append(row)
    return {
        "completed_mesh_watertight": bool(completed_mesh.is_watertight),
        "completed_mesh_face_count": int(len(completed_mesh.faces)),
        "strata": {
            "fit_pose": {
                "frames": [r["frame_idx"] for r in fit_rows],
                "unsigned_dist_median_of_frame_medians_m": float(np.median([r["completed_unsigned_dist_median_m"] for r in fit_rows])) if fit_rows else None,
                "unsigned_dist_p95_of_frame_p95_m": float(np.median([r["completed_unsigned_dist_p95_m"] for r in fit_rows])) if fit_rows else None,
                "best_effort_inside_count_sum": sum(r["completed_best_effort_inside_count"] for r in fit_rows),
                "rows": fit_rows,
            },
            "missing_or_interpolated_pose": {
                "frames": [r["frame_idx"] for r in missing_rows],
                "unsigned_dist_median_of_frame_medians_m": float(np.median([r["completed_unsigned_dist_median_m"] for r in missing_rows])) if missing_rows else None,
                "unsigned_dist_p95_of_frame_p95_m": float(np.median([r["completed_unsigned_dist_p95_m"] for r in missing_rows])) if missing_rows else None,
                "best_effort_inside_count_sum": sum(r["completed_best_effort_inside_count"] for r in missing_rows),
                "rows": missing_rows,
            },
        },
    }


# --------------------------------------------------------------------------- #
# Review image
# --------------------------------------------------------------------------- #
def make_review_image(ev: FrameEvidence, rgb_path: Path, out_path: Path,
                      title: str) -> bool:
    rgb_path = _remount(rgb_path)
    if not rgb_path.exists():
        return False
    img = cv2.imread(str(rgb_path))
    if img is None:
        return False
    H, W = 1408, 1408
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    overlay = img.copy()
    if ev.keyboard_mask is not None:
        ov = overlay.copy()
        ov[ev.keyboard_mask] = (0.55 * ov[ev.keyboard_mask] + 0.45 * np.array([40, 255, 80])).astype(np.uint8)
        overlay = ov
    if ev.visible_object_owned_mask is not None:
        ov = overlay.copy()
        ov[ev.visible_object_owned_mask] = (0.6 * ov[ev.visible_object_owned_mask] + 0.4 * np.array([0, 200, 255])).astype(np.uint8)
        overlay = ov
    if ev.hand_support_mask is not None:
        ov = overlay.copy()
        ov[ev.hand_support_mask] = (0.6 * ov[ev.hand_support_mask] + 0.4 * np.array([255, 255, 0])).astype(np.uint8)
        overlay = ov
    # project hand verts
    cam_z, ur, vr, surf = project_and_sample(
        ev.verts_world, ev.R_c2w, ev.t_c2w, ev.intr, ev.depth
    )
    ok = (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H)
    for u, v in zip(ur[ok], vr[ok]):
        cv2.circle(overlay, (int(u), int(v)), 2, (0, 0, 255), -1)
    # fingertip verts bigger
    tips = fingertip_vertex_indices(ev.verts_world, ev.joints_world)
    for i in tips:
        if ok[i]:
            cv2.circle(overlay, (int(ur[i]), int(vr[i])), 5, (255, 0, 255), 2)
    cv2.putText(overlay, title[:120], (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_frame_evidence(args, frame_idx: int, ann_frames: dict,
                         mano_by_frame: dict, pose_rows: dict,
                         frame_idx_arr: np.ndarray, depth_arr: np.ndarray,
                         sam2_track: dict, sam2_mask_dir: Path,
                         ownership_dir: Path, available_mask_frames: list[int],
                         hand_side: str) -> FrameEvidence | None:
    frame = ann_frames.get(frame_idx)
    if frame is None:
        return None
    mano = mano_by_frame.get(frame_idx)
    if mano is None:
        return None
    pose = pose_rows.get(frame_idx)
    if pose is None:
        return None

    # camera
    cam = frame.get("camera", {})
    T = np.asarray(cam.get("T_world_camera_metric") or [], dtype=float)
    if T.shape != (4, 4):
        return None
    R_c2w = T[:3, :3]
    t_c2w = T[:3, 3]

    # intrinsics from hand metric state
    hand = None
    for h in frame.get("hands", []):
        if str(h.get("hand_side")) == hand_side:
            hand = h
            break
    if hand is None:
        return None
    ms = hand.get("metric_mano_state", {}) if isinstance(hand.get("metric_mano_state"), dict) else {}
    intr = hand.get("current_v18_camera_intrinsics_fx_fy_cx_cy") or \
           ms.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
    if not intr or len(intr) != 4:
        return None
    intr = tuple(float(x) for x in intr)

    # MANO sample verts + joints
    vw = np.asarray(mano["optimized_vertices_world_sample_m"], dtype=float)
    sids = np.asarray(mano["optimized_vertices_sample_ids"], dtype=int)
    jw = np.asarray(mano["optimized_joints_world_m"], dtype=float)

    # depth row
    depth_idx = np.where(frame_idx_arr == frame_idx)[0]
    if depth_idx.size == 0:
        return None
    depth = depth_arr[depth_idx[0]].astype(np.float32)

    # masks
    kb_mask, kb_source, pms, voo, mask_exists = resolve_masks(
        frame_idx, sam2_track, sam2_mask_dir, ownership_dir, hand_side,
        available_mask_frames
    )

    # pose provenance
    tpg = pose.get("temporal_pose_graph", {}) or {}
    pose_source = tpg.get("pose_source", pose.get("pose_measurement_status", "unknown"))
    # direct/fit visibility: pose_source marks direct visible observation; the
    # explicit boolean is only carried on missing-pose rows, so derive from source.
    direct = (pose_source == "direct_visible_pose_observation_corrected")
    gap = int(tpg.get("gap_frames", 0) or 0)
    R_obj = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=float)
    t_obj = np.asarray(pose["translation_world_m"], dtype=float)

    return FrameEvidence(
        frame_idx=frame_idx,
        verts_world=vw,
        sample_ids=sids,
        joints_world=jw,
        R_c2w=R_c2w,
        t_c2w=t_c2w,
        intr=intr,
        depth=depth,
        keyboard_mask=kb_mask,
        keyboard_mask_source=kb_source,
        hand_support_mask=pms,
        visible_object_owned_mask=voo,
        mask_files_exist=mask_exists,
        interval_full_penetration=mano.get("full_observed_surface_penetration_after_solver_m"),
        interval_contact_patch_ids=list(mano.get("contact_patch_vertex_ids") or []),
        interval_contact_patch_gap=mano.get("contact_patch_final_normal_gap_m"),
        interval_depth_order_selected_count=int(mano.get("visible_surface_depth_order_selected_vertex_count") or 0),
        object_rotation_world_from_canonical=R_obj,
        object_translation_world=t_obj,
        pose_source=pose_source,
        pose_direct_visible=direct,
        pose_gap_frames=gap,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument("--frame-lo", type=int, default=FRAME_LO_DEFAULT)
    ap.add_argument("--frame-hi", type=int, default=FRAME_HI_DEFAULT)
    ap.add_argument("--hand-side", default=HAND_SIDE_DEFAULT)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/tmp/clip001850_masked_contact_killtests"))
    args = ap.parse_args()

    R = args.run_root
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"[killtests] run_root={R}", flush=True)
    print(f"[killtests] frames {args.frame_lo}-{args.frame_hi} hand={args.hand_side}", flush=True)

    # ---- Load artifacts ----
    ann_path = R / "measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json"
    mano_path = R / ("measurements/mano_interval_correction/keyboard_0_149/"
                     "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/"
                     "v18_joint_mano_interval_trajectory_state.json")
    pose_path = R / "measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json"
    depth_npz_path = R / "measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz"
    completed_mesh_path = R / "measurements/geometry_completion/compact_keyboard_seed42/keyboard_compact_rigid_completed_mesh_labeled.ply"
    observed_mesh_path = R / "measurements/geometry_completion/compact_keyboard_seed42/keyboard_observed_depth_surface_labeled.ply"
    sam2_mask_dir = R / "measurements/object_tracks/sam2_agent_points/keyboard/sam2/sam2_masks"
    sam2_track_path = R / "measurements/object_tracks/sam2_agent_points/keyboard/sam2/sam2_track.json"
    ownership_dir = (R / "measurements/contact_visibility_factors/keyboard_0_149/"
                     "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/ownership_masks")
    rgb_dir = R / "measurements/depth_slam/unidepth_full_frame/stills"

    ann = load_json(ann_path)
    ann_frames = {f["frame_idx"]: f for f in ann["frames"] if isinstance(f, dict)}
    mano = load_json(mano_path)
    mano_by_frame = {}
    for r in mano["per_frame_states"]:
        if r.get("hand_side") == args.hand_side:
            mano_by_frame[r["frame_idx"]] = r
    pose_rep = load_json(pose_path)
    pose_rows = {r["frame_idx"]: r for r in pose_rep["pose_rows"]}
    frame_idx_arr, depth_arr = load_depth_npz(depth_npz_path)
    sam2_track = load_json(sam2_track_path)

    # available keyboard mask frames in the window
    available_mask_frames = []
    for f in range(args.frame_lo, args.frame_hi + 1):
        if (sam2_mask_dir / f"{f:06d}.png").exists():
            available_mask_frames.append(f)
    print(f"[killtests] keyboard mask files available in window: {available_mask_frames}",
          flush=True)

    # meshes
    print("[killtests] loading completed mesh (CPU)...", flush=True)
    completed_mesh = trimesh.load(str(completed_mesh_path), process=False)
    print(f"[killtests] completed mesh: verts={len(completed_mesh.vertices)} "
          f"faces={len(completed_mesh.faces)} watertight={completed_mesh.is_watertight}",
          flush=True)
    observed_mesh = None
    if observed_mesh_path.exists():
        observed_mesh = trimesh.load(str(observed_mesh_path), process=False)
        print(f"[killtests] observed mesh: verts={len(observed_mesh.vertices)} "
              f"faces={len(observed_mesh.faces)} watertight={observed_mesh.is_watertight}",
              flush=True)

    # completed mesh AABB (KT-d extent plausibility)
    bb = completed_mesh.bounding_box.extents
    bb_sorted = np.sort(bb)
    extent_ratio = np.sort(bb_sorted / np.sort(KEYBOARD_PRIOR_EXTENTS_M))
    face_labels_path = R / "measurements/geometry_completion/compact_keyboard_seed42/completed_mesh_face_labels.json"
    face_provenance = {}
    if face_labels_path.exists():
        fl = load_json(face_labels_path)
        counts = fl.get("label_counts") or fl.get("face_label_counts") or {}
        face_provenance = {
            "face_label_counts": counts,
            "trellis_fraction": _fraction(counts, "trellis_inferred_hidden_surface"),
            "observed_fraction": _fraction(counts, "observed_depth_surface"),
            "unsupported_fraction": _fraction(counts, "unsupported_uncertain"),
        }

    # ---- Build per-frame evidence ----
    evs = []
    for f in range(args.frame_lo, args.frame_hi + 1):
        ev = build_frame_evidence(
            args, f, ann_frames, mano_by_frame, pose_rows,
            frame_idx_arr, depth_arr, sam2_track, sam2_mask_dir,
            ownership_dir, available_mask_frames, args.hand_side
        )
        if ev is None:
            print(f"[killtests] f{f}: could not build evidence (missing frame/mano/pose)",
                  flush=True)
        else:
            evs.append(ev)
    print(f"[killtests] built evidence for {len(evs)} frames", flush=True)

    # labels (anatomical) computed from f36 joints as reference (MANO topology fixed)
    ref = evs[0]
    labels = classify_anatomical(ref.verts_world, ref.joints_world)

    # ---- KT-1 + KT-2 per frame ----
    rows = []
    for ev in evs:
        # recompute labels per frame (verts differ but topology is by sample id)
        lab = classify_anatomical(ev.verts_world, ev.joints_world)
        row = run_frame(ev, lab)
        rows.append(row)
        # terse log
        kt1f = row["KT1_full_frame_depth"]
        kt1k = row["KT1_keyboard_masked_depth"]
        kt1v = row["KT1_keyboard_masked_hand_quarantined_depth"]
        fp = (kt1f.get("penetrating_depth_summary_m") or {}).get("max")
        kp = (kt1k.get("penetrating_depth_summary_m") or {}).get("max") if kt1k.get("available") else None
        vp = (kt1v.get("penetrating_depth_summary_m") or {}).get("max") if kt1v.get("available") else None
        print(f"  f{ev.frame_idx:3d} mask={ev.keyboard_mask_source:24s} "
              f"full_pen_max={_fmt(fp)} kb_pen_max={_fmt(kp)} voo_pen_max={_fmt(vp)} "
              f"route={row['route']}", flush=True)

    # ---- KT-2 temporal coherence ----
    kt2_window = temporal_coherence(rows)

    # ---- KT-3 ----
    print(f"[killtests] KT-3 fingertip deep dive at f{KT3_FRAME}...", flush=True)
    kt3_ev = next((e for e in evs if e.frame_idx == KT3_FRAME), None)
    kt3 = None
    if kt3_ev is not None:
        kt3 = run_kt3(kt3_ev, completed_mesh, observed_mesh,
                      classify_anatomical(kt3_ev.verts_world, kt3_ev.joints_world))

    # ---- KT-5 ----
    print("[killtests] KT-5 completed-mesh distance stratified by pose provenance...",
          flush=True)
    kt5 = run_kt5(evs, completed_mesh)

    # ---- Review images ----
    review_paths = {}
    for f in (32, 36):
        ev_r = next((e for e in evs if e.frame_idx == f), None)
        if ev_r is None:
            continue
        rgb = rgb_dir / f"frame_{f:06d}.png"
        out_png = out / f"review_f{f}.jpg"
        title = (f"f{f} right  kb_mask={ev_r.keyboard_mask_source}  "
                 f"route={rows[[r['frame_idx'] for r in rows].index(f)]['route']}")
        ok = make_review_image(ev_r, rgb, out_png, title)
        if ok:
            review_paths[f] = str(out_png)
            print(f"[killtests] wrote review image f{f}: {out_png}", flush=True)

    # ---- Route summary ----
    from collections import Counter
    route_counts = dict(Counter(r["route"] for r in rows))

    # ---- Write NDJSON ----
    ndjson_path = out / "contact_killtests_frame_detail.ndjson"
    with open(ndjson_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[killtests] wrote {ndjson_path} ({len(rows)} rows)", flush=True)

    # ---- Write summary ----
    summary = {
        "case": "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1",
        "run_root": str(R),
        "window": {"frame_lo": args.frame_lo, "frame_hi": args.frame_hi,
                   "hand_side": args.hand_side},
        "parameters": {
            "soft_tissue_band_m": SOFT_TISSUE_BAND_M,
            "penetration_margin_m": PENETRATION_MARGIN_M,
            "keyboard_prior_extents_m": list(KEYBOARD_PRIOR_EXTENTS_M),
        },
        "mask_availability": {
            "keyboard_mask_frames_in_window": available_mask_frames,
            "ownership_mask_frames_in_window": available_mask_frames,
        },
        "geometry_epoch": {
            "completed_mesh_watertight": bool(completed_mesh.is_watertight),
            "completed_mesh_face_count": int(len(completed_mesh.faces)),
            "completed_mesh_aabb_extents_m": [float(x) for x in bb],
            "completed_mesh_extent_ratio_to_keyboard_prior": [float(x) for x in extent_ratio],
            "face_provenance": face_provenance,
        },
        "route_counts": route_counts,
        "KT2_temporal_coherence": kt2_window,
        "KT3_f36_fingertip": kt3,
        "KT5_pose_provenance_stratification": kt5,
        "review_images": review_paths,
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[killtests] wrote {summary_path}", flush=True)

    # ---- Console headline ----
    print("\n===== KILL-TEST HEADLINE =====", flush=True)
    print(f"routes: {route_counts}", flush=True)
    print(f"completed mesh watertight={completed_mesh.is_watertight} "
          f"extent_ratio={extent_ratio.tolist()}", flush=True)
    if face_provenance:
        print(f"face provenance: trellis={face_provenance.get('trellis_fraction')} "
              f"observed={face_provenance.get('observed_fraction')}", flush=True)
    print(f"KT2 mean IoU (penetrating sets): {kt2_window['mean_iou']} "
          f"-> {kt2_window['interpretation']}", flush=True)
    if kt3:
        print(f"KT3 f36 fingertip delta_full median="
              f"{(kt3['delta_full_summary_m'] or {}).get('median')} "
              f"delta_kb_masked median="
              f"{(kt3['delta_keyboard_masked_summary_m'] or {}).get('median')} "
              f"completed_unsigned median="
              f"{(kt3['completed_mesh_unsigned_summary_m'] or {}).get('median')}",
              flush=True)
    fit_med = kt5["strata"]["fit_pose"]["unsigned_dist_median_of_frame_medians_m"]
    miss_med = kt5["strata"]["missing_or_interpolated_pose"]["unsigned_dist_median_of_frame_medians_m"]
    print(f"KT5 completed-mesh unsigned dist median: fit={fit_med} missing={miss_med}",
          flush=True)

    return 0


def _fraction(counts: dict, key: str) -> Any:
    if not counts:
        return None
    total = sum(int(v) for v in counts.values() if isinstance(v, int))
    if total == 0:
        return None
    return round(int(counts.get(key, 0)) / total, 4)


def _fmt(x) -> str:
    if x is None:
        return "  n/a"
    return f"{x:+.4f}" if isinstance(x, (int, float)) else str(x)


if __name__ == "__main__":
    sys.exit(main())
