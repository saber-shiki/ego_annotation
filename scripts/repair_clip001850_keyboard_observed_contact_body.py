#!/usr/bin/env python3
"""CPU-only object-geometry repair for HOT3D clip001850 keyboard contact body.

Run root:
  /data2/ego_annotation_outputs/v19_runs/
  20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1

Mechanism (causal card):
  Defect: the completed TRELLIS keyboard body is 8.4x too thick in its thin axis,
  non-watertight, 95.4% TRELLIS-inferred. Its free space was never carved. Contact /
  nonpenetration queries against it report phantom 8-10 cm hand penetration that is
  an artefact of the inflated body enclosing the hand volume, not real contact.

  Physical variable: object geometry epoch — the keyboard body used for contact
  distance queries.

  Two competing mechanisms:
    M-A (geometry repair sufficient): once the TRELLIS bulge is carved away and the
        contact body is restricted to observed keyboard surface, the hand sits on
        the real surface and contact becomes decidable.
    M-B (hand-depth bias dominates): even with a clean observed body, the MANO hand
        is biased ~8 cm too close to camera (known clip001850 along-ray bias), so
        the hand-to-observed-surface gap remains large regardless of geometry repair.

  Discriminating measurement: compare hand-to-repaired-body distance vs
  hand-to-completed-body distance vs hand-to-keyboard-depth-surface distance on
  f32/f36/f45. If M-A, repaired-body distance ≈ 0-30 mm (within sigma_clip). If M-B,
  repaired-body distance ≈ 80 mm, matching the keyboard-depth gap.

Outputs (under --out-dir, default /tmp/clip001850_keyboard_body_repair):
  repaired_observed_contact_body.ply      — observed-only contact-eligible body
  repaired_carved_support_body.ply        — observed + free-space-carved TRELLIS
  hand_to_body_distances.ndjson           — per-frame distances to 3 bodies
  face_provenance_freespace_summary.json  — face provenance + free-space carve counts
  summary.json                            — headline decision
  review_f032.jpg / review_f036.jpg / review_f045.jpg

No model inference, no GPU. Reads only on-disk v19 artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

# clip001850 combined metric uncertainty floor (from hand_metric_mechanisms.md)
SIGMA_CLIP_M = 0.028  # 25-30 mm lateral + along-ray systematic bias
SOFT_TISSUE_BAND_M = 0.015

# Free-space carve margin: a TRELLIS face centroid closer to the camera than the
# observed depth by more than this margin is a free-space violation.
FREE_SPACE_CARVE_MARGIN_M = 0.010

KEYBOARD_PRIOR_EXTENTS_M = (0.03, 0.15, 0.45)

TARGET_FRAMES_DEFAULT = [30, 31, 32, 33, 34, 35, 36, 45, 46]
CARVE_FRAMES_DEFAULT = [30, 31, 32, 33, 34, 35, 36, 45, 46]
REVIEW_FRAMES = [32, 36, 45]


# --------------------------------------------------------------------------- #
# Helpers
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
    with open(path) as fh:
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
# Body construction
# --------------------------------------------------------------------------- #
def build_observed_only_body(
    completed_mesh: trimesh.Trimesh, face_labels: list[str]
) -> trimesh.Trimesh:
    """Extract the observed_depth_surface faces from the completed mesh."""
    labels = np.asarray(face_labels)
    observed_mask = labels == "observed_depth_surface"
    obs_faces = completed_mesh.faces[observed_mask]
    return trimesh.Trimesh(
        vertices=completed_mesh.vertices.copy(),
        faces=obs_faces,
        process=True,
    )


def free_space_carve_trellis(
    completed_mesh: trimesh.Trimesh,
    face_labels: list[str],
    carve_frames: list[int],
    frame_cameras: dict[int, dict],
    frame_depths: dict[int, np.ndarray],
    frame_kb_masks: dict[int, np.ndarray],
    sam2_mask_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """For each TRELLIS face, project its centroid into every available camera
    frame. If it projects onto a keyboard-mask pixel and its camera-z is closer
    than the observed depth by > margin, it violates free space -> carve.

    Returns (surviving_face_mask, carve_report).
    """
    labels = np.asarray(face_labels)
    is_trellis = labels == "trellis_inferred_hidden_surface"
    is_unsupported = labels == "unsupported_uncertain"
    carveable = is_trellis | is_unsupported

    # face centroids in canonical frame
    triangles = completed_mesh.triangles  # (F, 3, 3)
    face_centroids = triangles.mean(axis=1)  # (F, 3)

    n_carveable = int(carveable.sum())
    violated = np.zeros(len(completed_mesh.faces), dtype=bool)

    per_frame_stats = []
    for f in carve_frames:
        cam_info = frame_cameras.get(f)
        depth = frame_depths.get(f)
        kb_mask_path = sam2_mask_dir / f"{f:06d}.png"
        kb_mask = frame_kb_masks.get(f)
        if kb_mask is None:
            kb_mask = load_mask_1408(kb_mask_path)
        if cam_info is None or depth is None or kb_mask is None:
            per_frame_stats.append({"frame": f, "status": "skipped_missing_data"})
            continue

        R_c2w = cam_info["R_c2w"]
        t_c2w = cam_info["t_c2w"]
        fx, fy, cx, cy = cam_info["intr"]
        H, W = depth.shape

        # transform canonical centroids -> world -> camera
        R_obj = cam_info["R_obj"]  # rotation_world_from_canonical
        t_obj = cam_info["t_obj"]  # translation_world
        world = face_centroids @ R_obj.T + t_obj[None, :]
        cam = (world - t_c2w[None, :]) @ R_c2w
        cam_z = cam[:, 2]

        zsafe = np.maximum(cam_z, 1e-9)
        u = fx * cam[:, 0] / zsafe + cx
        v = fy * cam[:, 1] / zsafe + cy
        ur = np.rint(u).astype(int)
        vr = np.rint(v).astype(int)

        inbounds = (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H) & (cam_z > 1e-5)
        on_kb = np.zeros(len(completed_mesh.faces), dtype=bool)
        on_kb[inbounds] = kb_mask[vr[inbounds], ur[inbounds]]

        # depth at projected pixel
        surf_z = np.full(len(completed_mesh.faces), np.nan, dtype=float)
        surf_z[inbounds] = depth[vr[inbounds], ur[inbounds]].astype(float)

        # free-space violation: face is carveable, projects onto keyboard mask,
        # and is closer to camera than observed depth by more than margin
        valid_depth = np.isfinite(surf_z)
        is_violation = (
            carveable
            & on_kb
            & valid_depth
            & (cam_z < (surf_z - FREE_SPACE_CARVE_MARGIN_M))
        )
        newly_violated = is_violation & ~violated
        violated |= is_violation

        n_violated_this = int(is_violation.sum())
        n_carveable_on_kb = int((carveable & on_kb & valid_depth).sum())
        per_frame_stats.append({
            "frame": f,
            "status": "carved",
            "carveable_faces_on_kb_with_depth": n_carveable_on_kb,
            "violations_this_frame": n_violated_this,
            "newly_violated": int(newly_violated.sum()),
        })

    # observed faces always survive; carveable faces survive if not violated
    is_observed = labels == "observed_depth_surface"
    surviving = is_observed | (carveable & ~violated)

    report = {
        "total_faces": int(len(completed_mesh.faces)),
        "observed_faces": int(is_observed.sum()),
        "carveable_faces": int(n_carveable),
        "carveable_violated": int((carveable & violated).sum()),
        "carveable_surviving": int((carveable & ~violated).sum()),
        "surviving_total": int(surviving.sum()),
        "per_frame": per_frame_stats,
        "free_space_carve_margin_m": FREE_SPACE_CARVE_MARGIN_M,
    }
    return surviving, report


def build_carved_body(
    completed_mesh: trimesh.Trimesh, surviving_mask: np.ndarray
) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=completed_mesh.vertices.copy(),
        faces=completed_mesh.faces[surviving_mask],
        process=True,
    )


# --------------------------------------------------------------------------- #
# Distance computation
# --------------------------------------------------------------------------- #
def hand_to_body_distances(
    verts_world: np.ndarray,
    R_obj: np.ndarray,
    t_obj: np.ndarray,
    body_mesh: trimesh.Trimesh,
) -> np.ndarray:
    """Unsigned closest-surface distance from hand verts to body mesh in
    canonical frame. Returns per-vertex distances (m)."""
    verts_obj = (verts_world - t_obj[None, :]) @ R_obj
    _, dist, _ = trimesh.proximity.closest_point(body_mesh, verts_obj)
    return np.asarray(dist, dtype=float)


def hand_to_depth_surface_gap(
    verts_world: np.ndarray,
    R_c2w: np.ndarray,
    t_c2w: np.ndarray,
    intr: tuple[float, float, float, float],
    depth: np.ndarray,
    kb_mask: np.ndarray | None,
) -> dict[str, Any]:
    """Delta = cam_z_hand - depth_surface along camera ray (positive = behind)."""
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
    if kb_mask is not None:
        on_kb = np.zeros(verts_world.shape[0], dtype=bool)
        on_kb[inbounds] = kb_mask[vr[inbounds], ur[inbounds]]
        surf[~on_kb] = np.nan
    delta = cam_z - surf  # negative = hand in front
    return {
        "delta": delta,
        "summary": numeric_summary(delta),
    }


# --------------------------------------------------------------------------- #
# Review images
# --------------------------------------------------------------------------- #
def make_review_image(
    rgb_path: Path,
    out_path: Path,
    verts_world: np.ndarray,
    joints_world: np.ndarray,
    R_c2w: np.ndarray,
    t_c2w: np.ndarray,
    intr: tuple[float, float, float, float],
    depth: np.ndarray,
    kb_mask: np.ndarray | None,
    completed_dist_summary: dict,
    observed_dist_summary: dict,
    carved_dist_summary: dict,
    depth_gap_summary: dict,
    title: str,
) -> bool:
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
    if kb_mask is not None:
        ov = overlay.copy()
        blend = np.array([40, 255, 80], dtype=float)
        ov[kb_mask] = (0.55 * ov[kb_mask] + 0.45 * blend).astype(np.uint8)
        overlay = ov

    fx, fy, cx, cy = intr
    cam = (verts_world - t_c2w[None, :]) @ R_c2w
    cam_z = cam[:, 2]
    zsafe = np.maximum(cam_z, 1e-9)
    u = fx * cam[:, 0] / zsafe + cx
    v = fy * cam[:, 1] / zsafe + cy
    ur = np.rint(u).astype(int)
    vr = np.rint(v).astype(int)
    ok = (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H)
    for i in np.where(ok)[0]:
        cv2.circle(overlay, (int(ur[i]), int(vr[i])), 2, (0, 0, 255), -1)

    # fingertip verts
    d = np.linalg.norm(verts_world[:, None, :] - joints_world[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    tip_joints = {4, 8, 12, 16, 20}
    tips = np.array([i for i in range(len(verts_world)) if nearest[i] in tip_joints])
    for i in tips:
        if ok[i]:
            cv2.circle(overlay, (int(ur[i]), int(vr[i])), 5, (255, 0, 255), 2)

    # info panel
    lines = [
        title,
        f"completed_body  unsigned dist median={_fmtd(completed_dist_summary)}",
        f"observed_body   unsigned dist median={_fmtd(observed_dist_summary)}",
        f"carved_body     unsigned dist median={_fmtd(carved_dist_summary)}",
        f"kb_depth_surface gap median  ={_fmtd(depth_gap_summary)}  (-=hand in front)",
        f"sigma_clip={SIGMA_CLIP_M*1000:.0f}mm  soft_tissue={SOFT_TISSUE_BAND_M*1000:.0f}mm",
    ]
    y0 = 36
    for i, line in enumerate(lines):
        cv2.putText(overlay, line[:130], (16, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, line[:130], (16, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


def _fmtd(summary: dict | None) -> str:
    if not summary or summary.get("count", 0) == 0:
        return "  n/a"
    med = summary.get("median")
    return f"{med*1000:+.1f}mm" if med is not None else "  n/a"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/tmp/clip001850_keyboard_body_repair"))
    ap.add_argument("--hand-side", default="right")
    args = ap.parse_args()

    R = args.run_root
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    hand_side = args.hand_side

    target_frames = TARGET_FRAMES_DEFAULT
    carve_frames = CARVE_FRAMES_DEFAULT

    print(f"[repair] run_root={R}", flush=True)
    print(f"[repair] target frames={target_frames}  hand={hand_side}", flush=True)

    # ---- Load artifacts ----
    ann_path = R / "measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json"
    mano_path = R / ("measurements/mano_interval_correction/keyboard_0_149/"
                     "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/"
                     "v18_joint_mano_interval_trajectory_state.json")
    pose_path = R / "measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json"
    depth_npz_path = R / "measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz"
    completed_mesh_path = R / "measurements/geometry_completion/compact_keyboard_seed42/keyboard_compact_rigid_completed_mesh_labeled.ply"
    face_labels_path = R / "measurements/geometry_completion/compact_keyboard_seed42/completed_mesh_face_labels.json"
    sam2_mask_dir = R / "measurements/object_tracks/sam2_agent_points/keyboard/sam2/sam2_masks"
    rgb_dir = R / "measurements/depth_slam/unidepth_full_frame/stills"

    ann = load_json(ann_path)
    ann_frames = {f["frame_idx"]: f for f in ann["frames"] if isinstance(f, dict)}
    mano = load_json(mano_path)
    mano_by_frame = {}
    for r in mano["per_frame_states"]:
        if r.get("hand_side") == hand_side:
            mano_by_frame[r["frame_idx"]] = r
    pose_rep = load_json(pose_path)
    pose_rows = {r["frame_idx"]: r for r in pose_rep["pose_rows"]}
    nz = np.load(depth_npz_path)
    frame_idx_arr = np.asarray(nz["frame_idx"], dtype=np.int64)
    depth_arr = np.asarray(nz["depth"], dtype=np.float32)

    # ---- Load meshes ----
    print("[repair] loading completed mesh (CPU)...", flush=True)
    completed_mesh = trimesh.load(str(completed_mesh_path), process=False)
    fl_data = load_json(face_labels_path)
    face_labels = fl_data["labels"]
    print(f"[repair] completed mesh: faces={len(completed_mesh.faces)} "
          f"watertight={completed_mesh.is_watertight}", flush=True)
    print(f"[repair] face labels: {Counter(face_labels)}", flush=True)
    bb = completed_mesh.bounding_box.extents
    print(f"[repair] completed AABB={np.round(bb, 4)} "
          f"extent_ratio_to_prior={np.sort(bb) / np.sort(KEYBOARD_PRIOR_EXTENTS_M)}",
          flush=True)

    # ---- Build observed-only body ----
    print("[repair] building observed-only contact body...", flush=True)
    observed_body = build_observed_only_body(completed_mesh, face_labels)
    print(f"[repair] observed body: faces={len(observed_body.faces)} "
          f"verts={len(observed_body.vertices)} "
          f"watertight={observed_body.is_watertight}", flush=True)
    print(f"[repair] observed body AABB={np.round(observed_body.bounding_box.extents, 4)}",
          flush=True)
    obs_ply = out / "repaired_observed_contact_body.ply"
    observed_body.export(str(obs_ply))
    print(f"[repair] wrote {obs_ply}", flush=True)

    # ---- Build per-frame camera + mask dicts for carving ----
    frame_cameras: dict[int, dict] = {}
    frame_depths: dict[int, dict] = {}
    frame_kb_masks: dict[int, np.ndarray] = {}
    all_frames = sorted(set(target_frames) | set(carve_frames))
    for f in all_frames:
        af = ann_frames.get(f)
        pf = pose_rows.get(f)
        if af is None or pf is None:
            continue
        cam = af.get("camera", {})
        T = np.asarray(cam.get("T_world_camera_metric") or [], dtype=float)
        if T.shape != (4, 4):
            continue
        R_c2w = T[:3, :3]
        t_c2w = T[:3, 3]
        hand = None
        for h in af.get("hands", []):
            if str(h.get("hand_side")) == hand_side:
                hand = h
                break
        if hand is None:
            continue
        ms = hand.get("metric_mano_state", {}) if isinstance(hand.get("metric_mano_state"), dict) else {}
        intr = ms.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
        if not intr or len(intr) != 4:
            continue
        intr = tuple(float(x) for x in intr)
        R_obj = np.asarray(pf["rotation_world_from_completed_canonical_matrix"], dtype=float)
        t_obj = np.asarray(pf["translation_world_m"], dtype=float)
        frame_cameras[f] = {"R_c2w": R_c2w, "t_c2w": t_c2w, "intr": intr,
                            "R_obj": R_obj, "t_obj": t_obj}
        d_idx = np.where(frame_idx_arr == f)[0]
        if d_idx.size > 0:
            frame_depths[f] = depth_arr[d_idx[0]].astype(np.float32)
        kb = load_mask_1408(sam2_mask_dir / f"{f:06d}.png")
        if kb is not None:
            frame_kb_masks[f] = kb

    # ---- Free-space carve TRELLIS ----
    print(f"[repair] free-space carving over frames {carve_frames}...", flush=True)
    surviving_mask, carve_report = free_space_carve_trellis(
        completed_mesh, face_labels, carve_frames,
        frame_cameras, frame_depths, frame_kb_masks, sam2_mask_dir,
    )
    carved_body = build_carved_body(completed_mesh, surviving_mask)
    print(f"[repair] carved body: faces={len(carved_body.faces)} "
          f"verts={len(carved_body.vertices)} "
          f"watertight={carved_body.is_watertight}", flush=True)
    print(f"[repair] carved body AABB={np.round(carved_body.bounding_box.extents, 4)}",
          flush=True)
    carved_ply = out / "repaired_carved_support_body.ply"
    carved_body.export(str(carved_ply))
    print(f"[repair] wrote {carved_ply}", flush=True)

    # ---- Per-frame hand-to-body distances ----
    print("[repair] computing per-frame hand-to-body distances...", flush=True)
    ndjson_path = out / "hand_to_body_distances.ndjson"
    rows = []
    for f in target_frames:
        mr = mano_by_frame.get(f)
        af = ann_frames.get(f)
        pf = pose_rows.get(f)
        fc = frame_cameras.get(f)
        if mr is None or af is None or pf is None or fc is None:
            print(f"  f{f}: skipped (missing mano/ann/pose/camera)", flush=True)
            continue
        vw = np.asarray(mr["optimized_vertices_world_sample_m"], dtype=float)
        jw = np.asarray(mr["optimized_joints_world_m"], dtype=float)
        R_obj = fc["R_obj"]
        t_obj = fc["t_obj"]
        R_c2w = fc["R_c2w"]
        t_c2w = fc["t_c2w"]
        intr = fc["intr"]
        depth = frame_depths.get(f)
        kb_mask = frame_kb_masks.get(f)

        dist_completed = hand_to_body_distances(vw, R_obj, t_obj, completed_mesh)
        dist_observed = hand_to_body_distances(vw, R_obj, t_obj, observed_body)
        dist_carved = hand_to_body_distances(vw, R_obj, t_obj, carved_body)

        depth_gap = {"summary": {"count": 0}}
        if depth is not None and kb_mask is not None:
            depth_gap = hand_to_depth_surface_gap(vw, R_c2w, t_c2w, intr, depth, kb_mask)

        # interval solver published penetration
        interval_pen = mr.get("full_observed_surface_penetration_after_solver_m")

        # contact decision under each body
        obs_med = float(np.median(dist_observed))
        carved_med = float(np.median(dist_carved))
        depth_gap_med = depth_gap["summary"].get("median")

        def classify(med: float | None) -> str:
            if med is None:
                return "undecidable_no_data"
            if med <= SOFT_TISSUE_BAND_M:
                return "contact_candidate_within_soft_tissue"
            if med <= SIGMA_CLIP_M:
                return "contact_candidate_within_sigma_clip"
            if med <= 3 * SIGMA_CLIP_M:
                return "contact_candidate_uncertain_1to3sigma"
            return "no_contact_supported_beyond_3sigma"

        row = {
            "frame_idx": f,
            "hand_side": hand_side,
            "n_hand_verts": int(len(vw)),
            "completed_body_unsigned_dist_m": numeric_summary(dist_completed),
            "observed_body_unsigned_dist_m": numeric_summary(dist_observed),
            "carved_body_unsigned_dist_m": numeric_summary(dist_carved),
            "kb_depth_surface_gap_m": depth_gap["summary"],
            "interval_solver_published_penetration_m": interval_pen,
            "contact_state_if_observed_body": classify(obs_med),
            "contact_state_if_carved_body": classify(carved_med),
            "contact_state_if_kb_depth": (
                "contact_candidate" if (depth_gap_med is not None and abs(depth_gap_med) <= SIGMA_CLIP_M)
                else "no_contact_or_unresolved" if depth_gap_med is not None
                else "undecidable_no_data"
            ),
        }
        rows.append(row)
        print(f"  f{f}: completed_med={np.median(dist_completed)*1000:.1f}mm  "
              f"observed_med={obs_med*1000:.1f}mm  "
              f"carved_med={carved_med*1000:.1f}mm  "
              f"kb_depth_gap_med={(_fmm(depth_gap_med))}  "
              f"-> {row['contact_state_if_observed_body']}",
              flush=True)

    with open(ndjson_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[repair] wrote {ndjson_path} ({len(rows)} rows)", flush=True)

    # ---- Face provenance / free-space summary ----
    provenance_summary = {
        "completed_mesh": {
            "total_faces": int(len(completed_mesh.faces)),
            "watertight": bool(completed_mesh.is_watertight),
            "aabb_extents_m": [float(x) for x in bb],
            "extent_ratio_to_keyboard_prior": [
                float(x) for x in np.sort(bb) / np.sort(KEYBOARD_PRIOR_EXTENTS_M)
            ],
            "face_label_counts": dict(Counter(face_labels)),
        },
        "observed_only_body": {
            "faces": int(len(observed_body.faces)),
            "vertices": int(len(observed_body.vertices)),
            "watertight": bool(observed_body.is_watertight),
            "aabb_extents_m": [float(x) for x in observed_body.bounding_box.extents],
            "thin_axis_ratio_to_completed": float(
                np.min(observed_body.bounding_box.extents) / np.min(bb)
            ),
        },
        "carved_body": {
            "faces": int(len(carved_body.faces)),
            "vertices": int(len(carved_body.vertices)),
            "watertight": bool(carved_body.is_watertight),
            "aabb_extents_m": [float(x) for x in carved_body.bounding_box.extents],
            "thin_axis_ratio_to_completed": float(
                np.min(carved_body.bounding_box.extents) / np.min(bb)
            ),
        },
        "free_space_carve": carve_report,
        "watertight_sign_mesh_possible": bool(
            observed_body.is_watertight or carved_body.is_watertight
        ),
        "missing_topological_condition": (
            "Observed keyboard surface is a one-sided partial patch (4184 faces over "
            "a 0.25x0.66 m footprint). It captures the visible top/front of the keys "
            "but not the back, sides, or bottom. A watertight sign mesh requires a "
            "closed manifold enclosing a volume; the observed faces alone form an open "
            "surface with boundary edges. Free-space carving removes TRELLIS faces that "
            "violate observed depth, but the remaining carved body is still open "
            "(non-watertight) because the back/bottom was never observed. Signed "
            "penetration queries remain unreliable; only unsigned closest-surface "
            "distance is contact-admissible from existing observations."
        ),
    }
    prov_path = out / "face_provenance_freespace_summary.json"
    with open(prov_path, "w") as fh:
        json.dump(provenance_summary, fh, indent=2)
    print(f"[repair] wrote {prov_path}", flush=True)

    # ---- Review images ----
    for f in REVIEW_FRAMES:
        mr = mano_by_frame.get(f)
        fc = frame_cameras.get(f)
        if mr is None or fc is None:
            continue
        vw = np.asarray(mr["optimized_vertices_world_sample_m"], dtype=float)
        jw = np.asarray(mr["optimized_joints_world_m"], dtype=float)
        depth = frame_depths.get(f)
        kb_mask = frame_kb_masks.get(f)
        rgb = rgb_dir / f"frame_{f:06d}.png"
        out_jpg = out / f"review_f{f:03d}.jpg"

        dist_c = hand_to_body_distances(vw, fc["R_obj"], fc["t_obj"], completed_mesh)
        dist_o = hand_to_body_distances(vw, fc["R_obj"], fc["t_obj"], observed_body)
        dist_v = hand_to_body_distances(vw, fc["R_obj"], fc["t_obj"], carved_body)
        dg = {"summary": {"count": 0}}
        if depth is not None and kb_mask is not None:
            dg = hand_to_depth_surface_gap(vw, fc["R_c2w"], fc["t_c2w"], fc["intr"],
                                           depth, kb_mask)
        title = f"f{f} right  green=kb_mask  red=hand verts  magenta=fingertips"
        ok = make_review_image(
            rgb, out_jpg, vw, jw, fc["R_c2w"], fc["t_c2w"], fc["intr"],
            depth if depth is not None else np.zeros((1408, 1408), dtype=np.float32),
            kb_mask,
            numeric_summary(dist_c), numeric_summary(dist_o), numeric_summary(dist_v),
            dg["summary"], title,
        )
        if ok:
            print(f"[repair] wrote review image {out_jpg}", flush=True)

    # ---- Decision ----
    # Does geometry repair alone change contact_state, or does hand-depth bias
    # dominate?
    decision_lines = []
    geometry_changes_contact = False
    for row in rows:
        f = row["frame_idx"]
        obs_state = row["contact_state_if_observed_body"]
        completed_med = row["completed_body_unsigned_dist_m"]["median"]
        observed_med = row["observed_body_unsigned_dist_m"]["median"]
        carved_med = row["carved_body_unsigned_dist_m"]["median"]
        depth_gap = row["kb_depth_surface_gap_m"].get("median")

        # Does the repaired body put the hand within sigma_clip?
        repaired_within_band = (
            observed_med is not None and observed_med <= SIGMA_CLIP_M
        )
        # Does the repaired body gap match the keyboard depth gap?
        if depth_gap is not None and observed_med is not None:
            gap_vs_depth_match = abs(abs(depth_gap) - observed_med) < 0.015
        else:
            gap_vs_depth_match = None

        if repaired_within_band:
            geometry_changes_contact = True
            decision_lines.append(
                f"f{f}: geometry repair CHANGES contact_state -> {obs_state} "
                f"(observed_body_med={observed_med*1000:.1f}mm <= sigma_clip)"
            )
        else:
            reason = "hand_depth_bias_dominates" if gap_vs_depth_match else "gap_remains_large"
            decision_lines.append(
                f"f{f}: geometry repair does NOT change contact_state -> {obs_state} "
                f"(observed_body_med={observed_med*1000:.1f}mm > sigma_clip; "
                f"kb_depth_gap={_fmm(depth_gap)}; {reason})"
            )

    overall_decision = (
        "geometry_repair_sufficient_to_decide_contact"
        if geometry_changes_contact
        else "hand_depth_bias_dominates_geometry_repair_alone_insufficient"
    )

    summary = {
        "case": "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1",
        "run_root": str(R),
        "target_frames": target_frames,
        "hand_side": hand_side,
        "parameters": {
            "sigma_clip_m": SIGMA_CLIP_M,
            "soft_tissue_band_m": SOFT_TISSUE_BAND_M,
            "free_space_carve_margin_m": FREE_SPACE_CARVE_MARGIN_M,
            "keyboard_prior_extents_m": list(KEYBOARD_PRIOR_EXTENTS_M),
        },
        "decision": overall_decision,
        "per_frame_decision": decision_lines,
        "body_summaries": provenance_summary,
        "per_frame_distances": rows,
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[repair] wrote {summary_path}", flush=True)

    print("\n===== REPAIR HEADLINE =====", flush=True)
    print(f"DECISION: {overall_decision}", flush=True)
    for line in decision_lines:
        print(f"  {line}", flush=True)
    print(f"observed body faces: {len(observed_body.faces)}  "
          f"watertight: {observed_body.is_watertight}", flush=True)
    print(f"carved body faces: {len(carved_body.faces)}  "
          f"watertight: {carved_body.is_watertight}", flush=True)
    print(f"completed thin axis: {np.min(bb)*1000:.1f}mm -> "
          f"observed thin axis: {np.min(observed_body.bounding_box.extents)*1000:.1f}mm -> "
          f"carved thin axis: {np.min(carved_body.bounding_box.extents)*1000:.1f}mm",
          flush=True)

    return 0


def _fmm(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x*1000:+.1f}mm"


if __name__ == "__main__":
    sys.exit(main())
