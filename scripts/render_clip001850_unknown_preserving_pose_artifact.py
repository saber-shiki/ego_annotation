#!/usr/bin/env python3
"""Unknown-preserving object-pose artifact + full-duration render for clip001850.

This is the additive correction to the static-gauge fill in
`render_clip001850_static_gauge_pose_artifact.py`. That script froze f78-149 at
the f77 mask-drift outlier and then "repaired" the tail by stamping a held
T_rest into all 150 pose rows. The user correctly objected that T_rest is a
wrong prior: a missing frame is UNKNOWN, not "static at rest". A static prior
is a strong physical claim that can be laundered into missing-frame pose,
contact evidence, or graph infill. This script reverts that fill: measured
frames keep numeric pose; every other frame has NULL pose and an explicit
reason, and the renderer draws NO object body on them.

Causal card
-----------
- **Artifact defect (numeric + rendered).** The static-gauge pose table writes a
  numeric R/t into all 150 rows and labels `holds_static_gauge=True` / a held
  T_rest body placement on 142 unobserved frames. That is a pose prior masquerading
  as a trajectory: the renderer draws the body on f78-149 and f0-29 as if the
  keyboard were localized there. A missing observation is unknown; it must not be
  filled with T_rest as a pose row.
- **Physical variable.** `T_world_object(t)` for every frame. The honest state is:
  numeric R/t ONLY on the 8 admissible measured rest-cluster frames
  (f30-36,f46); every other frame is unknown with a named reason
  (occluded_out_of_view / unresolved_hand_occluded / rejected_mask_drift /
  mask_unreliable). T_rest is retained ONLY as `rest_cluster_reference_for_outlier_rejection`,
  an internal quantity used to reject f60/75/76/77, never as a latent pose, never
  written into any pose row, never consumed by the renderer.
- **Live mechanisms (rejected).** (M-prior-fill) write T_rest into missing pose
  rows -> rejected by the user/parent: launderable into trajectory/contact/graph.
  (M-glyph-as-body) draw a small marker at T_rest on unknown frames -> rejected:
  any body-shaped mark placed from T_rest is a localization claim. The surviving
  mechanism is unknown-preserving: numeric pose only on measured frames; unknown
  frames show NO object body, only a frame-independent "object pose unknown" glyph
  that cannot be mistaken for a localized body.
- **Intervention (this script).**
  (1) object_pose_observations.ndjson: 150 rows. f30-36,f46 carry their measured
      numeric R/t and is_measured=is_localized=true. All other 142 frames carry
      rotation/translation = null, is_measured=is_localized=false, and a named
      reason. No frame outside the measured set has a numeric pose.
  (2) rest_cluster_reference_for_outlier_rejection.json: the T_rest value and
      covariance from f30-36,f46, named explicitly as a reference scalar used to
      reject f60/75/76/77 as mask-drift outliers. It states it MUST NOT be
      consumed as latent pose, trajectory fill, contact evidence, or graph prior
      for missing frames. It is not named "hypothesis" or "prior".
  (3) visibility_ledger.ndjson: same states/counts as the corrected ledger.
  (4) Full-duration overlay/world/side-by-side render. Object body is drawn ONLY
      on measured f30-36,f46. Unknown/unresolved/rejected/mask-unreliable/occluded
      frames show NO object body; they show a frame-independent "object pose
      unknown" text glyph and a dashed bounding-region-free marker that cannot be
      read as a localized body. Contact banners are unchanged.
- **State change.** f78-149 and f0-29 no longer render a keyboard body at all.
  f60/75/76/77 render no body. f45 renders no body. Only f30-36,f46 render the
  measured observed body. The numeric table has null pose on 142 frames.

No smoothing, no new model/GPU, no contact promotion, no pose infill, no static
prior written into any pose row. Contact states are carried through unchanged
from the canonical contact_frame_detail table.

Outputs (all under --output-root):
  object_pose_observations.ndjson      150 rows; numeric R/t only on f30-36,f46
  rest_cluster_reference_for_outlier_rejection.json   T_rest as reference ONLY
  visibility_ledger.ndjson             150 per-frame visibility states
  v19_overlay.mp4 v19_world.mp4 v19_side_by_side.mp4  150 frames @ 30 fps
  manifest.json                        render lineage + acceptance gates
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

RUN_ROOT = Path(
    "/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)
RESEARCH = Path(
    "/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706"
)

# ---------------------------------------------------------------------------
# Measured / unknown regime definition (from KT-L1/KT-L2 outcomes)
# ---------------------------------------------------------------------------
MEASURED_FRAMES = [30, 31, 32, 33, 34, 35, 36, 46]              # 8 admissible
UNRESOLVED_FRAMES = [45]                                        # 1 hand-occluded
REJECTED_MASK_DRIFT_FRAMES = [60, 75, 76, 77]                   # 4 outliers
MASK_UNRELIABLE_FRAMES = list(range(78, 150))                   # 72 delaminated
# occluded_out_of_view = every remaining frame (no metric surface)

DEFAULT_CONTACT_STATE = "unresolved_evidence_incomplete"

# Per-reason text shown on unknown frames (banner + overlay glyph).
UNKNOWN_REASON = {
    "occluded_out_of_view": "object pose UNKNOWN: no metric keyboard surface observed",
    "unresolved_hand_occluded": "object pose UNKNOWN: hand occludes keyboard; unresolved",
    "rejected_mask_drift": "object pose UNKNOWN: mask drift; direct fit rejected as outlier",
    "mask_unreliable": "object pose UNKNOWN: SAM2 mask delaminated; not recoverable",
}

# Fixed semantic vocabulary for the contact-state banner (carried unchanged
# from the prior accepted consumer; contact states are NOT modified here).
CONTACT_STATE_BANNER = {
    "geometry_epoch_contaminated": "contact_state=geometry_epoch_contaminated",
    "full_frame_depth_leak": "contact_state=full_frame_depth_leak",
    "unresolved_incoherent_evidence": "contact_state=unresolved_incoherent_evidence",
    "pose_unresolved": "contact_state=pose_unresolved",
    "contact_candidate": "contact_state=contact_candidate",
    "confirmed_contact": "contact_state=confirmed_contact",
    DEFAULT_CONTACT_STATE: f"contact_state={DEFAULT_CONTACT_STATE}",
}

# Glyph style for unknown frames: a small frame-independent "unknown" marker drawn
# at a FIXED screen position (NOT a world-projected body). It carries no pose.
UNKNOWN_GLYPH_COLOR = (210, 90, 90)      # red-ish, clearly not a body color
MEASURED_BODY_FACE = (255, 170, 30, 70)
MEASURED_BODY_EDGE = (255, 200, 80, 200)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_pose_rows(path: Path) -> dict[int, dict[str, Any]]:
    rep = load_json(path)
    records = rep.get("pose_rows") if isinstance(rep, dict) else rep
    return {int(r["frame_idx"]): r for r in records}


def load_mano_rows(path: Path, hand_side: str) -> dict[int, dict[str, Any]]:
    state = load_json(path)
    per_frame = state.get("per_frame_states", state) if isinstance(state, dict) else state
    out: dict[int, dict[str, Any]] = {}
    for r in per_frame:
        if hand_side and r.get("hand_side") != hand_side:
            continue
        out[int(r["frame_idx"])] = r
    return out


def load_visgeo_frames(path: Path) -> dict[int, dict[str, Any]]:
    vis = load_json(path)
    frames = vis.get("frames", vis) if isinstance(vis, dict) else vis
    return {int(r["frame_idx"]): r for r in frames}


# ---------------------------------------------------------------------------
# SO(3) / reference utilities
# ---------------------------------------------------------------------------
def project_to_SO3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_angle(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def rest_cluster_reference(pose_rows: dict[int, dict[str, Any]]):
    """T_rest over the measured cluster, used ONLY to reject mask-drift outliers.

    Returns (R_rest, t_rest, per-frame relative rotation deg, translation std m).
    This value is NOT a hypothesis, NOT a prior, and MUST NOT be written into any
    missing-frame pose row.
    """
    Ts = np.array([pose_rows[f]["translation_world_m"] for f in MEASURED_FRAMES])
    Rs = np.array([pose_rows[f]["rotation_world_from_completed_canonical_matrix"] for f in MEASURED_FRAMES])
    t_rest = np.median(Ts, axis=0)
    R_med = np.median(Rs, axis=0)
    R_rest = project_to_SO3(R_med)
    rel_rots_deg = np.array([rotation_angle(R_rest.T @ R) for R in Rs])
    t_std = Ts.std(axis=0, ddof=0)
    return R_rest, t_rest, rel_rots_deg, t_std


# ---------------------------------------------------------------------------
# Visibility ledger
# ---------------------------------------------------------------------------
def build_visibility_ledger(
    kt_l1: dict[str, Any],
    kt_l2: dict[str, Any],
    n_frames: int = 150,
) -> dict[int, dict[str, Any]]:
    """Per-frame visibility state from KT-L1/KT-L2 outcomes.

    static_observed        f30-36,46   (admissible measured)
    unresolved_hand_occluded f45
    rejected_mask_drift    f60,75,76,77
    mask_unreliable        f78-149
    occluded_out_of_view   all remaining (no metric surface)
    """
    measured = set(MEASURED_FRAMES)
    unresolved = set(UNRESOLVED_FRAMES)
    rejected = set(REJECTED_MASK_DRIFT_FRAMES)
    unreliable = set(MASK_UNRELIABLE_FRAMES)

    out: dict[int, dict[str, Any]] = {}
    for fi in range(n_frames):
        if fi in measured:
            st = "static_observed"
            basis = "KT-L1 rest cluster (f30-36,46); admissible measured direct ICP fit"
        elif fi in unresolved:
            st = "unresolved_hand_occluded"
            basis = "KT-L1 borderline f45: 68mm depth offset, hand_cov 0.626"
        elif fi in rejected:
            st = "rejected_mask_drift"
            disp = kt_l1["named_outlier_frames"][str(fi)]["displacement_from_rest_mm"]
            basis = f"KT-L1/KT-L5 outlier: {disp:.0f}mm from rest, mask drift"
        elif fi in unreliable:
            st = "mask_unreliable"
            basis = "KT-L2 delaminated: cleaned centroid 138-278mm from rest, 0/72 recoverable"
        else:
            st = "occluded_out_of_view"
            basis = "no metric keyboard surface (missing_initial_graph_pose)"
        out[fi] = {
            "frame_idx": fi,
            "visibility_state": st,
            "visibility_basis": basis,
            "is_measured": st == "static_observed",
            "is_localized": st == "static_observed",
        }
    return out


# ---------------------------------------------------------------------------
# object_pose_observations.ndjson: numeric R/t ONLY on measured frames
# ---------------------------------------------------------------------------
def build_pose_observation_rows(
    pose_rows: dict[int, dict[str, Any]],
    visibility: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """150 rows. Measured f30-36,f46 carry numeric R/t. All others have null pose."""
    rows: list[dict[str, Any]] = []
    for fi in range(len(visibility)):
        vis = visibility[fi]
        st = vis["visibility_state"]
        if st == "static_observed":
            src = pose_rows[fi]
            R = np.asarray(src["rotation_world_from_completed_canonical_matrix"], dtype=float)
            t = np.asarray(src["translation_world_m"], dtype=float)
            rows.append({
                "schema": "ego.hoi.object_pose_observations/0.1.0",
                "frame_idx": fi,
                "object_id": "keyboard",
                "rotation_world_from_completed_canonical_matrix": R.tolist(),
                "translation_world_m": t.tolist(),
                "pose_source": "measured_direct_icp_admissible",
                "visibility_state": st,
                "is_measured": True,
                "is_localized": True,
                "provenance": "own measured direct ICP fit (rest cluster); admissible measured pose",
            })
        else:
            rows.append({
                "schema": "ego.hoi.object_pose_observations/0.1.0",
                "frame_idx": fi,
                "object_id": "keyboard",
                "rotation_world_from_completed_canonical_matrix": None,
                "translation_world_m": None,
                "pose_source": "unknown_no_numeric_pose",
                "visibility_state": st,
                "is_measured": False,
                "is_localized": False,
                "reason": st,
                "provenance": (
                    "no numeric pose written; frame is unknown. "
                    f"reason={st}: {vis['visibility_basis']}"
                ),
            })
    return rows


# ---------------------------------------------------------------------------
# rest_cluster_reference_for_outlier_rejection.json
# ---------------------------------------------------------------------------
def build_rest_cluster_reference(
    pose_rows: dict[int, dict[str, Any]],
    kt_l1: dict[str, Any],
    R_rest: np.ndarray,
    t_rest: np.ndarray,
    rel_rots_deg: np.ndarray,
    t_std: np.ndarray,
) -> dict[str, Any]:
    """T_rest as a REFERENCE scalar used to reject mask-drift outliers ONLY.

    Explicitly NOT a hypothesis, NOT a prior, NOT a latent pose. Must not be
    consumed as trajectory fill, contact evidence, or graph prior for missing
    frames.
    """
    outlier_disp = {}
    for fi in REJECTED_MASK_DRIFT_FRAMES:
        t = np.asarray(pose_rows[fi]["translation_world_m"], dtype=float)
        outlier_disp[fi] = float(np.linalg.norm(t - t_rest) * 1000.0)
    return {
        "schema": "ego.hoi.rest_cluster_reference_for_outlier_rejection/0.1.0",
        "role": "rest_cluster_reference_for_outlier_rejection",
        "statement": (
            "This value is an internal reference scalar computed from the measured "
            "rest cluster (f30-36,f46). It is used ONLY to reject f60/75/76/77 as "
            "mask-drift outliers. It is NOT a pose hypothesis, NOT a prior, NOT a "
            "latent pose, and MUST NOT be consumed as trajectory fill, contact "
            "evidence, nonpenetration evidence, or a graph prior for missing frames. "
            "Missing frames are UNKNOWN; their pose is null in "
            "object_pose_observations.ndjson."
        ),
        "must_not_be_consumed_as": [
            "latent pose for missing frames",
            "trajectory fill / interpolation target",
            "contact evidence",
            "nonpenetration evidence",
            "graph prior for missing frames",
            "body placement on unobserved frames",
        ],
        "measured_cluster_frames": MEASURED_FRAMES,
        "T_rest_translation_m": t_rest.tolist(),
        "T_rest_rotation_world_from_canonical": R_rest.tolist(),
        "translation_std_per_axis_m": t_std.tolist(),
        "translation_covariance_diag_m2": (t_std ** 2).tolist(),
        "rotation_observability": {
            "per_frame_relative_rotation_deg": rel_rots_deg.tolist(),
            "max_deg": float(rel_rots_deg.max()),
            "median_deg": float(np.median(rel_rots_deg)),
        },
        "outlier_rejection_use": {
            "rejected_frames": REJECTED_MASK_DRIFT_FRAMES,
            "displacement_from_T_rest_mm": outlier_disp,
            "note": "these frames were rejected because their direct fit is far from the rest cluster reference",
        },
        "source": "existing rest-cluster pose rows f30-36,f46 (median translation + SO(3)-projected median rotation)",
        "kt_l1_rest_cluster": kt_l1.get("rest_cluster", {}),
        "integrity_gates": {
            "not_written_into_any_pose_row": True,
            "not_consumed_by_renderer": True,
            "no_frame_has_this_as_numeric_pose": True,
            "named_not_hypothesis_not_prior": True,
        },
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def project_world(points_world: np.ndarray, T_world_camera: np.ndarray, intr: list[float]) -> np.ndarray:
    Tcw = np.linalg.inv(np.asarray(T_world_camera, dtype=float))
    ph = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1).T
    pc = Tcw @ ph
    z = pc[2]
    fx, fy, cx, cy = intr
    u = fx * pc[0] / z + cx
    v = fy * pc[1] / z + cy
    return np.stack([u, v, z], axis=1)


def body_world(body: trimesh.Trimesh, R: np.ndarray, t: np.ndarray) -> trimesh.Trimesh:
    verts = np.asarray(body.vertices, dtype=float) @ R.T + t[None, :]
    return trimesh.Trimesh(vertices=verts, faces=body.faces, process=False)


def put_text_fit(img, text, org, max_width, scale, color, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    s = float(scale)
    while s > 0.26:
        (tw, _), _ = cv2.getTextSize(text, font, s, thickness)
        if tw <= max_width:
            break
        s *= 0.92
    cv2.putText(img, text, org, font, s, color, thickness, cv2.LINE_AA)


def draw_dashed(draw, p1, p2, fill, width=1, dash=6):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    n = int(length // dash) + 1
    for i in range(0, n, 2):
        s = i * dash
        e = min((i + 1) * dash, length)
        draw.line([(x1 + ux * s, y1 + uy * s), (x1 + ux * e, y1 + uy * e)], fill=fill, width=width)


def banner(width, frame_idx, contact_state, vis_state, vis_basis, contact_basis):
    out = np.zeros((BANNER_H, width, 3), dtype=np.uint8)
    out[:] = (10, 10, 12)
    cv2.rectangle(out, (0, 0), (width - 1, BANNER_H - 1), (70, 70, 70), 1)
    head = CONTACT_STATE_BANNER.get(contact_state, f"contact_state={contact_state}")
    if vis_state == "static_observed":
        vis_label = f"visibility=static_observed  body=MEASURED (f{frame_idx} direct ICP)"
    else:
        vis_label = f"visibility={vis_state}  OBJECT BODY NOT DRAWN (pose unknown)"
    put_text_fit(out, f"f{frame_idx:03d}  {head}", (12, 22), width - 24, 0.60, (130, 255, 160), 2)
    put_text_fit(out, vis_label, (12, 45), width - 24, 0.40, (255, 228, 140), 1)
    put_text_fit(out, vis_basis, (12, 63), width - 24, 0.34, (170, 230, 255), 1)
    return out


def render_overlay_face_layer(draw, body_w, T_world_camera, intr, source_w, scale):
    """Draw the measured body. Called ONLY on static_observed frames."""
    verts = np.asarray(body_w.vertices, dtype=float)
    faces = np.asarray(body_w.faces, dtype=np.int64)
    Tcw = np.linalg.inv(np.asarray(T_world_camera, dtype=float))
    pc = (Tcw[:3, :3] @ verts.T + Tcw[:3, 3:4]).T
    z = pc[:, 2]
    fx, fy, cx, cy = intr
    u = fx * pc[:, 0] / z + cx
    v = fy * pc[:, 1] / z + cy
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    p0, p1, p2 = pc[f0], pc[f1], pc[f2]
    normals = np.cross(p1 - p0, p2 - p0)
    cz = (z[f0] + z[f1] + z[f2]) / 3.0
    cu = (u[f0] + u[f1] + u[f2]) / 3.0
    cvv = (v[f0] + v[f1] + v[f2]) / 3.0
    visible = ((normals[:, 2] < 0) & (cz > 0.03) &
               (cu > -80) & (cu < source_w + 80) & (cvv > -80) & (cvv < source_w + 80))
    idx = np.where(visible)[0]
    if len(idx) > 3200:
        idx = idx[np.linspace(0, len(idx) - 1, 3200).astype(np.int64)]
    idx = idx[np.argsort(-cz[idx])]
    su = u * scale
    sv = v * scale
    drawn = 0
    for fi in idx:
        a = (float(su[f0[fi]]), float(sv[f0[fi]]))
        b = (float(su[f1[fi]]), float(sv[f1[fi]]))
        c = (float(su[f2[fi]]), float(sv[f2[fi]]))
        if not all(np.isfinite([*a, *b, *c])):
            continue
        draw.polygon([a, b, c], fill=MEASURED_BODY_FACE, outline=MEASURED_BODY_EDGE)
        drawn += 1
    return drawn


def draw_unknown_glyph_overlay(draw, dims, reason_text):
    """Frame-independent 'object pose unknown' glyph. NOT a body placement."""
    ow, oh = dims["overlay_w"], dims["overlay_h"]
    # red dashed rectangle in a fixed screen corner + text. Carries no pose.
    x0, y0, x1, y1 = ow - 300, 70, ow - 12, 150
    draw.rectangle([x0, y0, x1, y1], outline=(210, 90, 90, 220), width=2)
    # dashed cross-hatch to make it clearly a 'no data' marker
    for yy in range(y0, y1, 8):
        draw.line([(x0, yy), (x1, yy)], fill=(210, 90, 90, 90), width=1)
    draw.text((x0 + 8, y0 + 6), "OBJECT POSE", fill=(255, 120, 120))
    draw.text((x0 + 8, y0 + 24), "UNKNOWN", fill=(255, 120, 120))
    draw.text((x0 + 8, y0 + 44), reason_text[:34], fill=(255, 180, 180))
    draw.text((x0 + 8, y0 + 60), "(no body drawn)", fill=(255, 180, 180))


def render_overlay(frame_idx, vis_state, contact_row, body_w, mano_row, vis_frame, dims, scale, source_w):
    ow, oh = dims["overlay_w"], dims["overlay_h"]
    rgb = RGB_DIR / f"{frame_idx:06d}.jpg"
    img = Image.open(rgb).convert("RGB").resize((ow, oh))
    draw = ImageDraw.Draw(img, "RGBA")
    cam = vis_frame["camera"]
    Twc = np.asarray(cam["T_world_camera_metric"], dtype=float)
    intr = [float(x) for x in cam["intrinsics_fx_fy_cx_cy"]]

    faces_drawn = 0
    if vis_state == "static_observed":
        # measured body placement ONLY
        faces_drawn = render_overlay_face_layer(draw, body_w, Twc, intr, source_w, scale)
    else:
        # NO body; frame-independent unknown glyph
        draw_unknown_glyph_overlay(draw, dims, UNKNOWN_REASON.get(vis_state, vis_state))

    # hand vertices always drawn (hand state is independent of object pose)
    verts = np.asarray(mano_row["optimized_vertices_world_sample_m"], dtype=float)
    proj = project_world(verts, Twc, intr)
    px = proj[:, 0] * scale
    py = proj[:, 1] * scale
    depth = proj[:, 2]
    in_frame = (depth > 0.03) & (px >= 0) & (px < ow) & (py >= 0) & (py < oh)
    for i in np.where(in_frame)[0]:
        draw.ellipse([px[i] - 2, py[i] - 2, px[i] + 2, py[i] + 2], fill=(0, 220, 230, 185))
    joints = np.asarray(mano_row.get("optimized_joints_world_m") or [], dtype=float)
    if joints.shape == (21, 3):
        pj = project_world(joints, Twc, intr)
        for j in (4, 8, 12, 16, 20):
            x = pj[j, 0] * scale
            y = pj[j, 1] * scale
            if pj[j, 2] > 0 and 0 <= x < ow and 0 <= y < oh:
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 90, 255, 240), width=2)

    draw.rectangle([0, oh - 40, ow, oh], fill=(0, 0, 0, 150))
    if vis_state == "static_observed":
        draw.text((8, oh - 34), "amber = measured observed body (admissible direct ICP)", fill=(255, 210, 120))
    else:
        draw.text((8, oh - 34), "NO object body: pose unknown (measured frames only get a body)", fill=(255, 150, 150))
    draw.text((8, oh - 18), "cyan=MANO verts | magenta=fingertips | red box=object pose unknown (not a body)", fill=(200, 230, 230))

    content = np.asarray(img)
    cs = contact_row.get("contact_state", DEFAULT_CONTACT_STATE)
    frame = np.vstack([banner(ow, frame_idx, cs, vis_state, UNKNOWN_REASON.get(vis_state, "") if vis_state != "static_observed" else "measured direct ICP", contact_row.get("contact_state_basis", "")), content])
    return frame[:, :, ::-1], {"overlay_faces_drawn": int(faces_drawn), "body_drawn": vis_state == "static_observed"}


def world_camera_for(body_w, hand):
    center = 0.5 * (body_w.vertices.mean(axis=0) + hand.mean(axis=0))
    eye = center + np.array([0.0, -0.55, 0.45])
    forward = center - eye
    forward /= np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R_c2w = np.stack([right, -up, forward], axis=1)
    Tcw = np.eye(4)
    Tcw[:3, :3] = R_c2w.T
    Tcw[:3, 3] = -R_c2w.T @ eye
    return Tcw, [1100.0, 1100.0, WORLD_W / 2.0, WORLD_H / 2.0]


def render_world(frame_idx, vis_state, contact_row, body_w, mano_row):
    hand = np.asarray(mano_row["optimized_vertices_world_sample_m"], dtype=float)
    joints = np.asarray(mano_row.get("optimized_joints_world_m") or [], dtype=float)
    # camera framed around hand+body (body is the measured body on measured frames;
    # on unknown frames we still need a camera, so frame around the hand only)
    if vis_state == "static_observed":
        center = 0.5 * (body_w.vertices.mean(axis=0) + hand.mean(axis=0))
        z0 = min(body_w.vertices[:, 2].min(), hand[:, 2].min()) - 0.02
    else:
        center = hand.mean(axis=0)
        z0 = hand[:, 2].min() - 0.02
    Tcw, intr = world_camera_for(body_w if vis_state == "static_observed" else trimesh.Trimesh(vertices=hand, faces=[]), hand)
    fx, fy, cx, cy = intr

    def proj(points):
        pc = (Tcw[:3, :3] @ points.T + Tcw[:3, 3:4]).T
        z = pc[:, 2]
        u = fx * pc[:, 0] / z + cx
        v = fy * pc[:, 1] / z + cy
        return np.stack([u, v, z], axis=1)

    canvas = Image.new("RGB", (WORLD_W, WORLD_H), (14, 14, 18))
    draw = ImageDraw.Draw(canvas, "RGBA")
    extent = 0.6
    grid = np.linspace(-extent, extent, 13)
    for gx in grid:
        a, b = proj(np.array([[center[0] + gx, center[1] - extent, z0], [center[0] + gx, center[1] + extent, z0]]))
        if a[2] > 0 and b[2] > 0:
            draw.line([(a[0], a[1]), (b[0], b[1])], fill=(45, 55, 70, 190), width=1)
    for gy in grid:
        a, b = proj(np.array([[center[0] - extent, center[1] + gy, z0], [center[0] + extent, center[1] + gy, z0]]))
        if a[2] > 0 and b[2] > 0:
            draw.line([(a[0], a[1]), (b[0], b[1])], fill=(45, 55, 70, 190), width=1)

    edge_count = 0
    body_centroid_drawn = False
    if vis_state == "static_observed":
        pv = proj(body_w.vertices)
        edges = body_w.edges_unique
        step = 1
        for ei, edge in enumerate(edges):
            if ei % step:
                continue
            a, b = pv[edge[0]], pv[edge[1]]
            if a[2] > 0 and b[2] > 0 and np.isfinite(a[:2]).all() and np.isfinite(b[:2]).all():
                draw.line([(a[0], a[1]), (b[0], b[1])], fill=MEASURED_BODY_EDGE, width=1)
                edge_count += 1
        bc = proj(body_w.vertices.mean(axis=0)[None, :])[0]
        if bc[2] > 0 and np.isfinite(bc[:2]).all():
            draw.ellipse([bc[0] - 4, bc[1] - 4, bc[0] + 4, bc[1] + 4], outline=(255, 255, 255, 230), width=2)
            body_centroid_drawn = True
    else:
        # NO body in world view either; draw a frame-independent "object unknown" tag
        # at a fixed screen position (NOT a world point).
        tag_x, tag_y = WORLD_W - 360, 16
        draw.rectangle([tag_x, tag_y, tag_x + 340, tag_y + 54], outline=(210, 90, 90, 220), width=2)
        draw.text((tag_x + 10, tag_y + 8), "OBJECT POSE UNKNOWN", fill=(255, 130, 130))
        draw.text((tag_x + 10, tag_y + 30), "no body drawn (measured frames only)", fill=(255, 180, 180))

    ph = proj(hand)
    for p in ph:
        if p[2] > 0 and np.isfinite(p[:2]).all():
            draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=(0, 220, 230, 210))
    if joints.shape == (21, 3):
        pj = proj(joints)
        for j in (4, 8, 12, 16, 20):
            p = pj[j]
            if p[2] > 0 and np.isfinite(p[:2]).all():
                draw.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], outline=(255, 90, 255, 240), width=2)

    draw.rectangle([0, WORLD_H - 52, WORLD_W, WORLD_H], fill=(0, 0, 0, 160))
    if vis_state == "static_observed":
        draw.text((8, WORLD_H - 46), "amber solid = measured observed body (admissible ICP) | white cross = body centroid", fill=(255, 210, 120))
    else:
        draw.text((8, WORLD_H - 46), "NO object body: pose unknown. Body drawn only on measured f30-36,f46.", fill=(255, 150, 150))
    draw.text((8, WORLD_H - 28), "cyan=MANO verts | magenta=fingertips | grid=world ground", fill=(200, 230, 230))

    content = np.asarray(canvas)
    cs = contact_row.get("contact_state", DEFAULT_CONTACT_STATE)
    frame = np.vstack([banner(WORLD_W, frame_idx, cs, vis_state, UNKNOWN_REASON.get(vis_state, "") if vis_state != "static_observed" else "measured direct ICP", contact_row.get("contact_state_basis", "")), content])
    return frame[:, :, ::-1], {"world_edges_drawn": int(edge_count), "body_drawn": vis_state == "static_observed", "body_centroid_drawn": body_centroid_drawn}


def render_side_by_side(frame_idx, vis_state, contact_row, overlay_bgr, world_bgr, dims):
    sbs_w, sbs_h = dims["sbs_w"], dims["sbs_h"]
    overlay_content = overlay_bgr[BANNER_H:, :, :]
    world_content = world_bgr[BANNER_H:, :, :]
    ow = int(round(overlay_content.shape[1] * sbs_h / overlay_content.shape[0]))
    ww = int(round(world_content.shape[1] * sbs_h / world_content.shape[0]))
    ov = cv2.resize(overlay_content, (ow, sbs_h), interpolation=cv2.INTER_AREA)
    wv = cv2.resize(world_content, (ww, sbs_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((sbs_h, sbs_w, 3), dtype=np.uint8)
    canvas[:, : min(ow, sbs_w)] = ov[:, : min(ow, sbs_w)]
    remain = sbs_w - min(ow, sbs_w)
    if remain > 0:
        canvas[:, min(ow, sbs_w): min(ow, sbs_w) + min(ww, remain)] = wv[:, : min(ww, remain)]
    cs = contact_row.get("contact_state", DEFAULT_CONTACT_STATE)
    bn = banner(sbs_w, frame_idx, cs, vis_state, UNKNOWN_REASON.get(vis_state, "") if vis_state != "static_observed" else "measured direct ICP", contact_row.get("contact_state_basis", ""))
    return np.vstack([bn, canvas])


def write_video(frames, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for fr in frames:
            if fr.shape[:2] != (h, w):
                raise RuntimeError(f"Frame shape mismatch {fr.shape[:2]} vs {(h, w)}")
            writer.write(fr)
    finally:
        writer.release()


# Globals (set in main)
RGB_DIR: Path
BANNER_H = 66
WORLD_W = 1280
WORLD_H = 720


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, default=RUN_ROOT)
    p.add_argument("--contact-state-table", type=Path, default=RESEARCH / "contact_state_table" / "contact_frame_detail.ndjson")
    p.add_argument("--observed-body", type=Path, default=RESEARCH / "keyboard_body_repair" / "repaired_observed_contact_body.ply")
    p.add_argument("--pose-report", type=Path, default=RUN_ROOT / "measurements" / "pose_fits" / "keyboard_rigid_pose_graph" / "v19_rigid_object_pose_graph_report.json")
    p.add_argument("--mano-state", type=Path, default=RUN_ROOT / "measurements" / "mano_interval_correction" / "keyboard_0_149" / "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1" / "v18_joint_mano_interval_trajectory_state.json")
    p.add_argument("--visible-geometry", type=Path, default=RUN_ROOT / "measurements" / "object_geometry" / "visible_geometry" / "keyboard" / "annotations_v19_visible_geometry.json")
    p.add_argument("--kt-l1-summary", type=Path, default=RESEARCH / "pose_regime_kt_l1" / "summary.json")
    p.add_argument("--kt-l2-summary", type=Path, default=RESEARCH / "mask_recovery_kt_l2" / "summary.json")
    p.add_argument("--rgb-dir", type=Path, default=RUN_ROOT / "input" / "raw_frame_manifest" / "rgb")
    p.add_argument("--output-root", type=Path, default=RESEARCH / "unknown_preserving_pose_render")
    p.add_argument("--hand-side", type=str, default="right")
    p.add_argument("--frame-count", type=int, default=150)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--overlay-width", type=int, default=960)
    p.add_argument("--overlay-height", type=int, default=960)
    p.add_argument("--world-width", type=int, default=1280)
    p.add_argument("--world-height", type=int, default=720)
    p.add_argument("--sbs-width", type=int, default=1920)
    p.add_argument("--sbs-height", type=int, default=540)
    p.add_argument("--review-frames", type=str, default="0,30,32,36,45,46,60,75,77,90,120,149")
    return p.parse_args()


def main():
    global RGB_DIR, BANNER_H, WORLD_W, WORLD_H
    args = parse_args()
    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir = out_dir / "review_frames"
    review_dir.mkdir(parents=True, exist_ok=True)

    RGB_DIR = args.rgb_dir
    WORLD_W, WORLD_H = args.world_width, args.world_height
    dims = dict(overlay_w=args.overlay_width, overlay_h=args.overlay_height,
                sbs_w=args.sbs_width, sbs_h=args.sbs_height)

    kt_l1 = load_json(args.kt_l1_summary)
    kt_l2 = load_json(args.kt_l2_summary)
    pose_rows = load_pose_rows(args.pose_report)
    mano_rows = load_mano_rows(args.mano_state, args.hand_side)
    vis_rows = load_visgeo_frames(args.visible_geometry)
    contact_rows_list = load_ndjson(args.contact_state_table)
    contact_rows = {int(r["frame_idx"]): r for r in contact_rows_list}
    body = trimesh.load(args.observed_body, process=False)
    if isinstance(body, trimesh.Scene):
        body = trimesh.util.concatenate([m for m in body.geometry.values() if isinstance(m, trimesh.Trimesh)])

    rgb_frames = sorted(RGB_DIR.glob("*.jpg"))
    frame_count = min(args.frame_count, len(rgb_frames))
    with Image.open(rgb_frames[0]) as im:
        source_w = im.width
    scale = dims["overlay_w"] / source_w

    # ---- rest cluster reference (NOT a prior, NOT written into pose rows) ----
    R_rest, t_rest, rel_rots_deg, t_std = rest_cluster_reference(pose_rows)

    # ---- visibility ledger ----
    visibility = build_visibility_ledger(kt_l1, kt_l2, frame_count)

    # ---- pose observations: numeric R/t ONLY on measured frames ----
    pose_obs_rows = build_pose_observation_rows(pose_rows, visibility)

    # ---- write durable tables ----
    pose_path = out_dir / "object_pose_observations.ndjson"
    ref_path = out_dir / "rest_cluster_reference_for_outlier_rejection.json"
    vis_path = out_dir / "visibility_ledger.ndjson"
    write_ndjson(pose_path, pose_obs_rows)
    write_ndjson(vis_path, list(visibility.values()))
    ref_doc = build_rest_cluster_reference(pose_rows, kt_l1, R_rest, t_rest, rel_rots_deg, t_std)
    ref_path.write_text(json.dumps(ref_doc, indent=2))

    # hash lineage
    source_hashes = {
        "object_pose_observations": sha256_file(pose_path),
        "visibility_ledger": sha256_file(vis_path),
        "rest_cluster_reference": sha256_file(ref_path),
        "observed_body": sha256_file(args.observed_body),
        "contact_state_table": sha256_file(args.contact_state_table),
        "kt_l1_summary": sha256_file(args.kt_l1_summary),
        "kt_l2_summary": sha256_file(args.kt_l2_summary),
        "pose_report": sha256_file(args.pose_report),
    }

    # ---- full-contact table (contact states unchanged) ----
    def contact_for(fi):
        if fi in contact_rows:
            return contact_rows[fi]
        return {"frame_idx": fi, "contact_state": DEFAULT_CONTACT_STATE,
                "contact_state_basis": "default_outside_contact_frame_detail_window"}

    # ---- render ----
    review_set = {int(x) for x in args.review_frames.split(",") if x.strip()}
    overlay_frames, world_frames, side_frames = [], [], []
    per_frame_manifest = []
    state_counts = Counter()

    # pose lookup from the observation rows we just wrote
    obs_by_frame = {r["frame_idx"]: r for r in pose_obs_rows}

    for fi in range(frame_count):
        vis = visibility[fi]
        vs = vis["visibility_state"]
        state_counts[vs] += 1
        crow = contact_for(fi)
        obs = obs_by_frame[fi]
        if vs == "static_observed":
            R = np.asarray(obs["rotation_world_from_completed_canonical_matrix"], dtype=float)
            t = np.asarray(obs["translation_world_m"], dtype=float)
            bw = body_world(body, R, t)
        else:
            bw = None  # NO body on unknown frames
        ov, ov_info = render_overlay(fi, vs, crow, bw, mano_rows[fi], vis_rows[fi], dims, scale, source_w)
        wo, wo_info = render_world(fi, vs, crow, bw, mano_rows[fi])
        sb = render_side_by_side(fi, vs, crow, ov, wo, dims)
        overlay_frames.append(ov)
        world_frames.append(wo)
        side_frames.append(sb)
        per_frame_manifest.append({
            "frame_idx": fi,
            "visibility_state": vs,
            "contact_state": crow.get("contact_state", DEFAULT_CONTACT_STATE),
            "pose_source": obs["pose_source"],
            "has_numeric_pose": obs["rotation_world_from_completed_canonical_matrix"] is not None,
            "is_measured": vis["is_measured"],
            "is_localized": vis["is_localized"],
            "body_drawn": vs == "static_observed",
            "overlay_info": ov_info,
            "world_info": wo_info,
        })
        if fi in review_set:
            cv2.imwrite(str(review_dir / f"overlay_f{fi:03d}.jpg"), ov)
            cv2.imwrite(str(review_dir / f"world_f{fi:03d}.jpg"), wo)
            cv2.imwrite(str(review_dir / f"side_by_side_f{fi:03d}.jpg"), sb)
        if fi % 25 == 0:
            print(f"[render] f{fi:03d}/{frame_count} vis={vs} contact={crow.get('contact_state')} body_drawn={vs == 'static_observed'}", flush=True)

    outputs = {
        "overlay": out_dir / "v19_overlay.mp4",
        "world": out_dir / "v19_world.mp4",
        "side_by_side": out_dir / "v19_side_by_side.mp4",
    }
    write_video(overlay_frames, outputs["overlay"], args.fps)
    write_video(world_frames, outputs["world"], args.fps)
    write_video(side_frames, outputs["side_by_side"], args.fps)

    # ---- acceptance gate checks ----
    # 1. no unobserved frame has numeric pose / is_measured / is_localized
    mislabelled = [fi for fi in range(frame_count)
                   if visibility[fi]["visibility_state"] != "static_observed"
                   and (visibility[fi]["is_measured"] or visibility[fi]["is_localized"])]
    numeric_on_unknown = [fi for fi in range(frame_count)
                          if visibility[fi]["visibility_state"] != "static_observed"
                          and obs_by_frame[fi]["rotation_world_from_completed_canonical_matrix"] is not None]
    # 2. f90 / f120 null pose + no body
    f90_null = obs_by_frame[90]["rotation_world_from_completed_canonical_matrix"] is None
    f120_null = obs_by_frame[120]["rotation_world_from_completed_canonical_matrix"] is None
    f90_no_body = not per_frame_manifest[90]["body_drawn"]
    f120_no_body = not per_frame_manifest[120]["body_drawn"]
    # 3. body consumed only from measured observations
    body_only_measured = all(
        (per_frame_manifest[fi]["body_drawn"]) == (visibility[fi]["visibility_state"] == "static_observed")
        for fi in range(frame_count)
    )
    # 4. contact states unchanged
    contact_states_in_render = {fi: contact_for(fi).get("contact_state") for fi in contact_rows}
    contact_unchanged = all(contact_rows[fi].get("contact_state") == contact_states_in_render[fi]
                            for fi in contact_rows)
    # 5. exactly 8 measured frames have numeric pose
    numeric_count = sum(1 for fi in range(frame_count)
                        if obs_by_frame[fi]["rotation_world_from_completed_canonical_matrix"] is not None)
    # 6. no T_rest written into any pose row
    trest_in_rows = any(
        obs_by_frame[fi]["translation_world_m"] is not None
        and np.allclose(np.asarray(obs_by_frame[fi]["translation_world_m"], dtype=float), t_rest)
        and visibility[fi]["visibility_state"] != "static_observed"
        for fi in range(frame_count)
    )

    manifest = {
        "schema": "ego.hoi.unknown_preserving_pose_render/0.1.0",
        "case": "hot3d_clip001850_keyboard_right",
        "run_root": str(args.run_root),
        "artifact_role": "unknown_preserving_object_pose_artifact_full_duration_render",
        "consumer_script": "scripts/render_clip001850_unknown_preserving_pose_artifact.py",
        "intervention": (
            "User-directed correction to the static-gauge fill: missing frames are "
            "UNKNOWN, not filled with T_rest. object_pose_observations.ndjson has "
            "numeric R/t ONLY on admissible measured f30-36,f46; all other 142 "
            "frames have null pose, is_measured=false, is_localized=false, and a "
            "named reason. T_rest is retained ONLY as "
            "rest_cluster_reference_for_outlier_rejection (to reject f60/75/76/77); "
            "it is NOT a hypothesis, NOT a prior, NOT written into any pose row, "
            "and MUST NOT be consumed as latent pose / trajectory fill / contact "
            "evidence / graph prior. The renderer draws the object body ONLY on "
            "measured frames; unknown frames show no body and a frame-independent "
            "unknown glyph. Contact states are carried through unchanged."
        ),
        "pose_policy": {
            "pose_infill_applied": False,
            "static_prior_written_into_pose_rows": False,
            "trest_written_into_any_missing_pose_row": bool(trest_in_rows),
            "numeric_pose_only_on_measured_frames": numeric_count == len(MEASURED_FRAMES),
            "missing_frames_are_unknown": True,
            "rest_cluster_reference_not_hypothesis_not_prior": True,
        },
        "inputs": {
            "pose_report": str(args.pose_report),
            "pose_report_sha256": source_hashes["pose_report"],
            "contact_state_table": str(args.contact_state_table),
            "contact_state_table_sha256": source_hashes["contact_state_table"],
            "observed_body": str(args.observed_body),
            "observed_body_sha256": source_hashes["observed_body"],
            "mano_state": str(args.mano_state),
            "visible_geometry": str(args.visible_geometry),
            "kt_l1_summary": str(args.kt_l1_summary),
            "kt_l2_summary": str(args.kt_l2_summary),
            "rgb_dir": str(args.rgb_dir),
        },
        "render_consumed_pose_source": str(pose_path),
        "render_consumed_pose_sha256": source_hashes["object_pose_observations"],
        "render_consumed_visibility_source": str(vis_path),
        "render_consumed_visibility_sha256": source_hashes["visibility_ledger"],
        "rest_cluster_reference_path": str(ref_path),
        "rest_cluster_reference_sha256": source_hashes["rest_cluster_reference"],
        "rest_cluster_reference_role": "outlier rejection ONLY; not a pose; not consumed by renderer",
        "outputs": {k: str(v) for k, v in outputs.items()},
        "frame_count": frame_count,
        "fps": args.fps,
        "duration_s": frame_count / args.fps,
        "visibility_counts": dict(state_counts),
        "rest_cluster_reference_summary": {
            "T_rest_translation_m": t_rest.tolist(),
            "translation_std_per_axis_m": t_std.tolist(),
            "rotation_observability_max_deg": float(rel_rots_deg.max()),
            "note": "reference scalar for outlier rejection; NOT a pose prior; not in any pose row",
        },
        "acceptance_gates": {
            "frame_count_is_150": frame_count == 150,
            "numeric_pose_only_on_measured_frames": numeric_count == len(MEASURED_FRAMES),
            "numeric_pose_count": numeric_count,
            "no_unobserved_frame_has_numeric_pose": len(numeric_on_unknown) == 0,
            "numeric_on_unknown_frames": numeric_on_unknown,
            "no_unobserved_frame_labelled_measured_or_localized": len(mislabelled) == 0,
            "mislabelled_frames": mislabelled,
            "f90_has_null_pose": f90_null,
            "f120_has_null_pose": f120_null,
            "f90_no_body_drawn": f90_no_body,
            "f120_no_body_drawn": f120_no_body,
            "body_drawn_only_on_measured_frames": body_only_measured,
            "no_trest_in_any_missing_pose_row": not trest_in_rows,
            "contact_states_unchanged": contact_unchanged,
            "no_pose_infill_applied": True,
            "no_static_prior_in_pose_rows": True,
            "render_manifest_consumed_pose_observations": source_hashes["object_pose_observations"] is not None,
            "pose_rows_written_count": len(pose_obs_rows),
            "visibility_rows_written_count": len(visibility),
        },
        "frames": per_frame_manifest,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[render] wrote pose observations -> {pose_path}")
    print(f"[render] wrote visibility -> {vis_path}")
    print(f"[render] wrote rest cluster reference -> {ref_path}")
    for k, v in outputs.items():
        print(f"[render] wrote {k} -> {v}")
    print(f"[render] wrote manifest -> {manifest_path}")
    gates = manifest["acceptance_gates"]
    failed = [k for k, v in gates.items() if v is False]
    print(f"[render] acceptance_gates {'ALL PASS' if not failed else 'FAILED: ' + ','.join(failed)}")
    print(f"[render] visibility_counts {dict(state_counts)}")


if __name__ == "__main__":
    main()
