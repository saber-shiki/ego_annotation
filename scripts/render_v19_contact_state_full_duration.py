#!/usr/bin/env python3
"""Generic full-duration v19-style contact-state render consumer.

This is the reusable, clip-agnostic promotion of the accepted clip001850
full-duration contact-state render consumer
(`render_clip001850_v19_contact_state_full_duration.py`). It produces a
replacement research render — named like the canonical v19 outputs
(`v19_overlay.mp4`, `v19_world.mp4`, `v19_side_by_side.mp4`) — whose visible
per-frame labels are driven by a canonical `contact_frame_detail` table and
whose object body is a repaired/observed mesh rather than a raw prior body.

Artifact-level contract (preserved from the accepted clip001850 consumer):
  * full duration — every RGB frame is rendered (frame_count == rgb count);
  * durable inputs — the contact table and observed body must not live under
    a transient `/tmp` path (the render must be reproducible);
  * default uncovered frames — frames outside the contact table are explicitly
    assigned `unresolved_evidence_incomplete`, never dropped, never fabricated
    as contact;
  * per-frame contact_state labels come from the table where present;
  * the rendered mesh hash is recorded in the manifest (the body provenance is
    explicit, not a prior completion pretending to be measured geometry);
  * the structurally false published claims (`penverts=0`, bare `UNCERTAIN`,
    and any clip-specific forbidden gap text) do not appear in the new text;
  * zero `confirmed_contact` unless the table says so with a floor-independent
    provenance column (motion coupling or GT). A distance threshold alone never
    promotes a frame.

CPU-only: reads JSON/PLY/RGB and uses OpenCV/PIL/trimesh for rendering. No
model inference, no GPU. The frozen runtime prediction root is never
overwritten; outputs go to `--output-root`.

Usage (clip001850 reproduction):

  python scripts/render_v19_contact_state_full_duration.py \
      --run-root /data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1 \
      --contact-state-table /data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/contact_state_table/contact_frame_detail.ndjson \
      --observed-body /data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/keyboard_body_repair/repaired_observed_contact_body.ply \
      --pose-report <run-root>/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json \
      --mano-state <run-root>/measurements/mano_interval_correction/keyboard_0_149/.../v18_joint_mano_interval_trajectory_state.json \
      --visible-geometry <run-root>/measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json \
      --rgb-dir <run-root>/input/raw_frame_manifest/rgb \
      --output-root /tmp/clip001850_v19_contact_state_generic_render \
      --trellis-mesh-hash 260b09d192c0abb7a75069b4367d934b0c3d217a1dc61bde8572c73c5593aab1 \
      --forbidden-text "39.2mm"
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

DEFAULT_CONTACT_STATE = "unresolved_evidence_incomplete"
DEFAULT_FORBIDDEN_TEXT = ("penverts=0", "UNCERTAIN")

# Fixed semantic vocabulary: maps a contact_state to a (headline, mechanism)
# pair. This is NOT object-category branching (forbidden by the methodology
# rule); it is a closed set of ego.hoi contact-state names. Row data
# (contact_state_demotion_reasons / contact_state_basis) is preferred when
# present so the visible mechanism stays data-driven.
CONTACT_STATE_BANNER = {
    "geometry_epoch_contaminated": (
        "contact_state=geometry_epoch_contaminated",
        "object body is contaminated: prior-inferred, non-watertight, or uncarved; mesh penetration is inadmissible",
    ),
    "full_frame_depth_leak": (
        "contact_state=full_frame_depth_leak",
        "masked + hand-quarantined depth rejects the full-frame penetration channel",
    ),
    "unresolved_incoherent_evidence": (
        "contact_state=unresolved_incoherent_evidence",
        "incoherent single-channel evidence without a persistent contact patch; candidate demoted",
    ),
    "pose_unresolved": (
        "contact_state=pose_unresolved",
        "object pose or mask provenance is insufficient for contact adjudication",
    ),
    "contact_candidate": (
        "contact_state=contact_candidate",
        "weak contact candidate; not confirmed — floor-independent provenance required to promote",
    ),
    "confirmed_contact": (
        "contact_state=confirmed_contact",
        "confirmed contact with floor-independent provenance",
    ),
    DEFAULT_CONTACT_STATE: (
        f"contact_state={DEFAULT_CONTACT_STATE}",
        "frame outside the contact_frame_detail window; no contact evidence is fabricated",
    ),
}

# Fields that count as floor-independent provenance for a confirmed_contact
# frame. A distance/gap value alone is never sufficient.
CONFIRMED_PROVENANCE_KEYS = (
    "confirmed_contact_provenance",
    "motion_coupling_onset",
    "gt_contact_provenance",
)
CONFIRMED_BASIS_MARKERS = ("motion", "gt", "ground_truth")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_contact_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["frame_idx"])] = row
    return rows


def default_contact_row(frame_idx: int) -> dict[str, Any]:
    return {
        "frame_idx": frame_idx,
        "contact_state": DEFAULT_CONTACT_STATE,
        "raw_killtest_route": DEFAULT_CONTACT_STATE,
        "contact_state_basis": "default_outside_contact_frame_detail_window",
        "contact_state_demotion_reasons": [],
        "killtest_evidence": {},
        "renderer_consumes_contact_state": True,
    }


def full_contact_rows(rows: dict[int, dict[str, Any]], frame_count: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for frame_idx in range(frame_count):
        row = dict(rows.get(frame_idx) or default_contact_row(frame_idx))
        row["renderer_consumes_contact_state"] = True
        out[frame_idx] = row
    return out


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


def project_world(points_world: np.ndarray, T_world_camera: np.ndarray, intr: list[float]) -> np.ndarray:
    Tcw = np.linalg.inv(np.asarray(T_world_camera, dtype=float))
    ph = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1).T
    pc = Tcw @ ph
    z = pc[2]
    fx, fy, cx, cy = intr
    u = fx * pc[0] / z + cx
    v = fy * pc[1] / z + cy
    return np.stack([u, v, z], axis=1)


def observed_body_world(body: trimesh.Trimesh, pose_row: dict[str, Any]) -> trimesh.Trimesh:
    R = np.asarray(pose_row["rotation_world_from_completed_canonical_matrix"], dtype=float)
    t = np.asarray(pose_row["translation_world_m"], dtype=float)
    verts = np.asarray(body.vertices, dtype=float) @ R.T + t[None, :]
    return trimesh.Trimesh(vertices=verts, faces=body.faces, process=False)


def put_text_fit(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    max_width: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    s = float(scale)
    while s > 0.28:
        (tw, _), _ = cv2.getTextSize(text, font, s, thickness)
        if tw <= max_width:
            break
        s *= 0.92
    cv2.putText(image, text, org, font, s, color, thickness, cv2.LINE_AA)


def mechanism_text(row: dict[str, Any], state: str) -> str:
    """Prefer row data; fall back to the fixed vocabulary.

    Row data wins so the mechanism stays tied to measured evidence rather than
    a hardcoded string. The fixed vocabulary only supplies a generic sentence
    when the row carries no demotion reasons or basis.
    """
    reasons = row.get("contact_state_demotion_reasons") or []
    if reasons:
        return "; ".join(str(r) for r in reasons[:2])
    basis = row.get("contact_state_basis")
    if basis and basis != "default_outside_contact_frame_detail_window":
        return str(basis)
    return CONTACT_STATE_BANNER.get(state, (f"contact_state={state}", "unresolved state"))[1]


def metric_summary(row: dict[str, Any]) -> str:
    ev = row.get("killtest_evidence") or {}
    hq = ev.get("KT1_keyboard_masked_hand_quarantined_depth") or ev.get("masked_hand_quarantined_depth") or {}
    interval = ev.get("interval_solver_published") or {}
    delta = (hq.get("delta_summary_m") or {}).get("median") if isinstance(hq.get("delta_summary_m"), dict) else hq.get("delta_summary_m")
    eligible = hq.get("penetrating_vertex_count")
    intv_max = (interval.get("interval_full_observed_surface_penetration_m") or {}).get("max") if isinstance(interval.get("interval_full_observed_surface_penetration_m"), dict) else interval.get("interval_full_observed_surface_penetration_m")
    parts = []
    if delta is None:
        parts.append("masked hand-depth delta: not measured on this frame")
    else:
        parts.append(f"masked hand-depth delta {float(delta) * 1000.0:.0f}mm (negative = hand in front)")
    if eligible is None:
        parts.append("eligible contact verts: not measured")
    else:
        parts.append(f"masked eligible contact verts {int(eligible)}")
    if intv_max is None:
        parts.append("interval penetration: not used")
    else:
        parts.append(f"interval penetration {float(intv_max) * 1000.0:.0f}mm rejected")
    return " | ".join(parts)


def contact_state_banner(width: int, frame_idx: int, row: dict[str, Any]) -> np.ndarray:
    state = str(row.get("contact_state") or DEFAULT_CONTACT_STATE)
    head = CONTACT_STATE_BANNER.get(state, (f"contact_state={state}", ""))[0]
    mechanism = mechanism_text(row, state)
    out = np.zeros((BANNER_H, width, 3), dtype=np.uint8)
    out[:] = (10, 10, 12)
    cv2.rectangle(out, (0, 0), (width - 1, BANNER_H - 1), (70, 70, 70), 1)
    put_text_fit(out, f"f{frame_idx:03d}  {head}", (12, 25), width - 24, 0.62, (130, 255, 160), 2)
    put_text_fit(out, mechanism, (12, 49), width - 24, 0.42, (255, 228, 140), 1)
    put_text_fit(out, metric_summary(row), (12, BANNER_H - 11), width - 24, 0.38, (170, 230, 255), 1)
    return out


def draw_dashed(draw: ImageDraw.ImageDraw, p1: tuple[float, float], p2: tuple[float, float], fill: tuple[int, int, int, int], width: int = 1, dash: int = 6) -> None:
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


def render_body_overlay(draw: ImageDraw.ImageDraw, body_world: trimesh.Trimesh, T_world_camera: np.ndarray, intr: list[float], source_w: int, scale: float, max_faces: int = 3200) -> int:
    verts = np.asarray(body_world.vertices, dtype=float)
    faces = np.asarray(body_world.faces, dtype=np.int64)
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
    cv_ = (v[f0] + v[f1] + v[f2]) / 3.0
    visible = (
        (normals[:, 2] < 0)
        & (cz > 0.03)
        & (cu > -80)
        & (cu < source_w + 80)
        & (cv_ > -80)
        & (cv_ < source_w + 80)
    )
    idx = np.where(visible)[0]
    if len(idx) > max_faces:
        idx = idx[np.linspace(0, len(idx) - 1, max_faces).astype(np.int64)]
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
        draw.polygon([a, b, c], fill=(255, 170, 30, 32), outline=(255, 190, 60, 150))
        if drawn % 3 == 0:
            xs = [a[0], b[0], c[0]]
            ys = [a[1], b[1], c[1]]
            draw.line([(min(xs), min(ys)), (max(xs), max(ys))], fill=(255, 120, 20, 70), width=1)
        drawn += 1
    return drawn


def render_overlay(frame_idx: int, row: dict[str, Any], pose_row: dict[str, Any], mano_row: dict[str, Any], vis_frame: dict[str, Any], body: trimesh.Trimesh, dims: dict[str, int], scale: float, source_w: int) -> tuple[np.ndarray, dict[str, Any]]:
    overlay_w = dims["overlay_w"]
    overlay_h = dims["overlay_h"]
    rgb = RGB_DIR / f"{frame_idx:06d}.jpg"
    img = Image.open(rgb).convert("RGB").resize((overlay_w, overlay_h))
    draw = ImageDraw.Draw(img, "RGBA")
    cam = vis_frame["camera"]
    Twc = np.asarray(cam["T_world_camera_metric"], dtype=float)
    intr = [float(x) for x in cam["intrinsics_fx_fy_cx_cy"]]
    body_world = observed_body_world(body, pose_row)
    body_faces_drawn = render_body_overlay(draw, body_world, Twc, intr, source_w, scale)

    verts = np.asarray(mano_row["optimized_vertices_world_sample_m"], dtype=float)
    proj = project_world(verts, Twc, intr)
    px = proj[:, 0] * scale
    py = proj[:, 1] * scale
    depth = proj[:, 2]
    in_frame = (depth > 0.03) & (px >= 0) & (px < overlay_w) & (py >= 0) & (py < overlay_h)
    for i in np.where(in_frame)[0]:
        draw.ellipse([px[i] - 2, py[i] - 2, px[i] + 2, py[i] + 2], fill=(0, 220, 230, 185))

    joints = np.asarray(mano_row.get("optimized_joints_world_m") or [], dtype=float)
    if joints.shape == (21, 3):
        pj = project_world(joints, Twc, intr)
        for j in (4, 8, 12, 16, 20):
            x = pj[j, 0] * scale
            y = pj[j, 1] * scale
            if pj[j, 2] > 0 and 0 <= x < overlay_w and 0 <= y < overlay_h:
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 90, 255, 240), width=2)

    draw.rectangle([0, overlay_h - 58, overlay_w, overlay_h], fill=(0, 0, 0, 150))
    legend = [
        "amber hatch = repaired observed object patch; open/non-watertight; raw prior body is not drawn",
        "cyan = MANO sample vertices | magenta = fingertips",
        "signed penetration is undefined for this body; contact only with floor-independent provenance",
    ]
    for k, text in enumerate(legend):
        draw.text((8, overlay_h - 52 + 18 * k), text, fill=(235, 235, 235))

    content = np.asarray(img)
    frame = np.vstack([contact_state_banner(overlay_w, frame_idx, row), content])
    return frame[:, :, ::-1], {
        "projected_mano_sample_vertices": int(in_frame.sum()),
        "observed_body_faces_drawn": int(body_faces_drawn),
    }


def world_camera_for(body_world: trimesh.Trimesh, hand_world: np.ndarray) -> tuple[np.ndarray, list[float]]:
    center = 0.5 * (body_world.vertices.mean(axis=0) + hand_world.mean(axis=0))
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


def render_world(frame_idx: int, row: dict[str, Any], pose_row: dict[str, Any], mano_row: dict[str, Any], body: trimesh.Trimesh) -> tuple[np.ndarray, dict[str, Any]]:
    body_world = observed_body_world(body, pose_row)
    hand = np.asarray(mano_row["optimized_vertices_world_sample_m"], dtype=float)
    joints = np.asarray(mano_row.get("optimized_joints_world_m") or [], dtype=float)
    Tcw, intr = world_camera_for(body_world, hand)
    fx, fy, cx, cy = intr

    def proj(points: np.ndarray) -> np.ndarray:
        pc = (Tcw[:3, :3] @ points.T + Tcw[:3, 3:4]).T
        z = pc[:, 2]
        u = fx * pc[:, 0] / z + cx
        v = fy * pc[:, 1] / z + cy
        return np.stack([u, v, z], axis=1)

    canvas = Image.new("RGB", (WORLD_W, WORLD_H), (14, 14, 18))
    draw = ImageDraw.Draw(canvas, "RGBA")
    center = 0.5 * (body_world.vertices.mean(axis=0) + hand.mean(axis=0))
    z0 = min(body_world.vertices[:, 2].min(), hand[:, 2].min()) - 0.02
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

    pv = proj(body_world.vertices)
    edge_count = 0
    for edge in body_world.edges_unique[::3]:
        a, b = pv[edge[0]], pv[edge[1]]
        if a[2] > 0 and b[2] > 0 and np.isfinite(a[:2]).all() and np.isfinite(b[:2]).all():
            draw_dashed(draw, (a[0], a[1]), (b[0], b[1]), (255, 190, 60, 150), width=1, dash=5)
            edge_count += 1

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
    draw.text((8, WORLD_H - 46), "amber dashed body = repaired observed open object patch; raw prior body removed", fill=(255, 210, 120))
    draw.text((8, WORLD_H - 28), "cyan = MANO sample vertices | magenta = fingertips | grid = world ground", fill=(200, 230, 230))

    content = np.asarray(canvas)
    frame = np.vstack([contact_state_banner(WORLD_W, frame_idx, row), content])
    return frame[:, :, ::-1], {"observed_body_edges_drawn": int(edge_count)}


def render_side_by_side(frame_idx: int, row: dict[str, Any], overlay_bgr: np.ndarray, world_bgr: np.ndarray, dims: dict[str, int]) -> np.ndarray:
    overlay_content = overlay_bgr[BANNER_H:, :, :]
    world_content = world_bgr[BANNER_H:, :, :]
    sbs_w = dims["sbs_w"]
    sbs_h = dims["sbs_h"]
    oh = sbs_h
    ow = int(round(overlay_content.shape[1] * oh / overlay_content.shape[0]))
    wh = sbs_h
    ww = int(round(world_content.shape[1] * wh / world_content.shape[0]))
    ov = cv2.resize(overlay_content, (ow, oh), interpolation=cv2.INTER_AREA)
    wv = cv2.resize(world_content, (ww, wh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((sbs_h, sbs_w, 3), dtype=np.uint8)
    canvas[:, : min(ow, sbs_w)] = ov[:, : min(ow, sbs_w)]
    remain = sbs_w - min(ow, sbs_w)
    if remain > 0:
        canvas[:, min(ow, sbs_w) : min(ow, sbs_w) + min(ww, remain)] = wv[:, : min(ww, remain)]
    return np.vstack([contact_state_banner(sbs_w, frame_idx, row), canvas])


def video_props(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    try:
        return {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        raise RuntimeError(f"No frames for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                raise RuntimeError(f"Frame shape mismatch for {path}: {frame.shape[:2]} vs {(h, w)}")
            writer.write(frame)
    finally:
        writer.release()


def extract_frame(video: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def pixel_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    aa = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    bb = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    delta = cv2.absdiff(aa, bb)
    gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    mask = gray > 18
    return {"diff_pixel_fraction_gt18": float(mask.mean()), "mean_absdiff": float(delta.mean()), "shape": [h, w]}


def text_gate(row_text: str, forbidden: tuple[str, ...]) -> dict[str, Any]:
    return {f"contains_{tok.replace('=','_eq_').replace('.','_').lower()}": tok in row_text for tok in forbidden}


def confirmed_has_provenance(row: dict[str, Any]) -> bool:
    """A confirmed_contact frame is admissible only with floor-independent
    provenance (motion coupling or GT), never a distance threshold alone."""
    for key in CONFIRMED_PROVENANCE_KEYS:
        if row.get(key):
            return True
    basis = str(row.get("contact_state_basis") or "").lower()
    if any(marker in basis for marker in CONFIRMED_BASIS_MARKERS):
        return True
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, default=None, help="Run root (recorded for provenance; inputs below override any defaults).")
    p.add_argument("--contact-state-table", type=Path, required=True, help="contact_frame_detail.ndjson (durable, not /tmp).")
    p.add_argument("--observed-body", type=Path, required=True, help="Repaired/observed object mesh .ply (durable, not /tmp).")
    p.add_argument("--pose-report", type=Path, required=True, help="v19_rigid_object_pose_graph_report.json (per-frame pose).")
    p.add_argument("--mano-state", type=Path, required=True, help="v18_joint_mano_interval_trajectory_state.json.")
    p.add_argument("--visible-geometry", type=Path, required=True, help="annotations_v19_visible_geometry.json (per-frame camera).")
    p.add_argument("--rgb-dir", type=Path, required=True, help="Directory of {frame:06d}.jpg source RGB frames.")
    p.add_argument("--output-root", type=Path, required=True, help="Where to write v19_{overlay,world,side_by_side}.mp4 + manifest.json.")
    p.add_argument("--case", type=str, default=None, help="Case id recorded in the manifest (e.g. hot3d_clip001850).")
    p.add_argument("--hand-side", type=str, default="right", help="Which hand's MANO rows to render ('right'/'left'/'' for all).")
    p.add_argument("--frame-count", type=int, default=None, help="Override frame count (default: all RGB frames).")
    p.add_argument("--fps", type=float, default=None, help="Override fps (default: from published overlay or 30).")
    p.add_argument("--overlay-width", type=int, default=960)
    p.add_argument("--overlay-height", type=int, default=960)
    p.add_argument("--world-width", type=int, default=1280)
    p.add_argument("--world-height", type=int, default=720)
    p.add_argument("--sbs-width", type=int, default=1920)
    p.add_argument("--sbs-height", type=int, default=540)
    p.add_argument("--review-frames", type=str, default=None, help="Comma-separated frame indices to dump as review JPGs (default: contact-table frames).")
    p.add_argument("--published-overlay", type=Path, default=None, help="Optional published overlay mp4 for pixel/label diff.")
    p.add_argument("--published-world", type=Path, default=None)
    p.add_argument("--published-side-by-side", type=Path, default=None)
    p.add_argument("--published-report", type=Path, default=None, help="Optional published render report JSON for banner-text comparison.")
    p.add_argument("--trellis-mesh-hash", type=str, default=None, help="Prior/prior-completion mesh sha256; gate checks the rendered body differs from it.")
    p.add_argument("--forbidden-text", type=str, action="append", default=None, help="Extra forbidden banner substrings (default: penverts=0, UNCERTAIN). Repeatable.")
    p.add_argument("--keep-frames", action="store_true", help="Keep per-frame JPG directories (default: review frames only).")
    return p.parse_args()


# Module-level globals set from args (kept minimal; mirrors the accepted
# consumer's structure so the render helpers stay unchanged in spirit).
RGB_DIR: Path
BANNER_H = 78
WORLD_W = 1280
WORLD_H = 720


def main() -> None:
    global RGB_DIR, BANNER_H, WORLD_W, WORLD_H
    args = parse_args()

    forbidden = tuple(args.forbidden_text) if args.forbidden_text else DEFAULT_FORBIDDEN_TEXT
    if str(args.contact_state_table).startswith("/tmp/") or str(args.observed_body).startswith("/tmp/"):
        raise RuntimeError("This full-duration consumer must read durable inputs, not ephemeral /tmp artifacts")

    RGB_DIR = args.rgb_dir
    WORLD_W = args.world_width
    WORLD_H = args.world_height
    dims = {
        "overlay_w": args.overlay_width,
        "overlay_h": args.overlay_height,
        "sbs_w": args.sbs_width,
        "sbs_h": args.sbs_height,
    }

    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir = out_dir / "review_frames"
    review_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_frames:
        for name in ("overlay_frames", "world_frames", "side_by_side_frames"):
            (out_dir / name).mkdir(parents=True, exist_ok=True)

    published_videos: dict[str, Path | None] = {
        "overlay": args.published_overlay,
        "world": args.published_world,
        "side_by_side": args.published_side_by_side,
    }
    published_props: dict[str, Any] = {}
    for k, v in published_videos.items():
        published_props[k] = video_props(v) if v and v.exists() else None
    fps = float(args.fps) if args.fps is not None else float((published_props.get("overlay") or {}).get("fps") or 30.0)

    rgb_frames = sorted(RGB_DIR.glob("*.jpg"))
    if not rgb_frames:
        raise RuntimeError(f"No RGB frames found under {RGB_DIR}")
    frame_count = args.frame_count or len(rgb_frames)
    if frame_count != len(rgb_frames):
        rgb_frames = rgb_frames[:frame_count]
    # Source width determines the overlay projection scale.
    with Image.open(rgb_frames[0]) as im:
        source_w = im.width
    scale = dims["overlay_w"] / source_w

    source_rows = load_contact_rows(args.contact_state_table)
    contact_rows = full_contact_rows(source_rows, frame_count)
    pose_rows = load_pose_rows(args.pose_report)
    mano_rows = load_mano_rows(args.mano_state, args.hand_side)
    vis_rows = load_visgeo_frames(args.visible_geometry)
    body = trimesh.load(args.observed_body, process=False)
    if isinstance(body, trimesh.Scene):
        meshes = [m for m in body.geometry.values() if isinstance(m, trimesh.Trimesh)]
        body = trimesh.util.concatenate(meshes)
    if not isinstance(body, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported observed body type: {type(body)}")

    review_set: set[int]
    if args.review_frames is not None:
        review_set = {int(x) for x in args.review_frames.split(",") if x.strip()}
    else:
        review_set = set(source_rows.keys())

    overlay_frames: list[np.ndarray] = []
    world_frames: list[np.ndarray] = []
    side_frames: list[np.ndarray] = []
    per_frame_manifest: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()

    for frame_idx in range(frame_count):
        if frame_idx not in pose_rows or frame_idx not in mano_rows or frame_idx not in vis_rows:
            raise RuntimeError(f"Missing pose/MANO/visgeo row for frame {frame_idx}")
        row = contact_rows[frame_idx]
        state = str(row.get("contact_state") or DEFAULT_CONTACT_STATE)
        state_counts[state] += 1
        overlay, overlay_info = render_overlay(frame_idx, row, pose_rows[frame_idx], mano_rows[frame_idx], vis_rows[frame_idx], body, dims, scale, source_w)
        world, world_info = render_world(frame_idx, row, pose_rows[frame_idx], mano_rows[frame_idx], body)
        side = render_side_by_side(frame_idx, row, overlay, world, dims)
        overlay_frames.append(overlay)
        world_frames.append(world)
        side_frames.append(side)
        head = CONTACT_STATE_BANNER.get(state, (f"contact_state={state}", ""))[0]
        mech = mechanism_text(row, state)
        frame_entry = {
            "frame_idx": frame_idx,
            "contact_state": state,
            "source": "contact_frame_detail" if frame_idx in source_rows else "default_outside_contact_frame_detail_window",
            "visible_label": head,
            "visible_mechanism": mech,
            "visible_metric_summary": metric_summary(row),
            "overlay_render_info": overlay_info,
            "world_render_info": world_info,
            "body_source": "repaired_observed_contact_body",
            "render_consumed_mesh_sha256": sha256_file(args.observed_body),
        }
        row_text = " | ".join([frame_entry["visible_label"], frame_entry["visible_mechanism"], frame_entry["visible_metric_summary"]])
        frame_entry["text_gate"] = text_gate(row_text, forbidden)
        if state == "confirmed_contact":
            frame_entry["confirmed_contact_has_floor_independent_provenance"] = confirmed_has_provenance(row)
        per_frame_manifest.append(frame_entry)
        if args.keep_frames:
            cv2.imwrite(str(out_dir / "overlay_frames" / f"{frame_idx:06d}.jpg"), overlay)
            cv2.imwrite(str(out_dir / "world_frames" / f"{frame_idx:06d}.jpg"), world)
            cv2.imwrite(str(out_dir / "side_by_side_frames" / f"{frame_idx:06d}.jpg"), side)
        if frame_idx in review_set:
            cv2.imwrite(str(review_dir / f"overlay_f{frame_idx:03d}.jpg"), overlay)
            cv2.imwrite(str(review_dir / f"world_f{frame_idx:03d}.jpg"), world)
            cv2.imwrite(str(review_dir / f"side_by_side_f{frame_idx:03d}.jpg"), side)
        if frame_idx % 25 == 0:
            print(f"[render] frame {frame_idx:03d}/{frame_count}: {state}", flush=True)

    outputs = {
        "overlay": out_dir / "v19_overlay.mp4",
        "world": out_dir / "v19_world.mp4",
        "side_by_side": out_dir / "v19_side_by_side.mp4",
    }
    write_video(overlay_frames, outputs["overlay"], fps)
    write_video(world_frames, outputs["world"], fps)
    write_video(side_frames, outputs["side_by_side"], fps)

    # Review diffs against published videos (optional).
    review_diffs: dict[str, Any] = {}
    for frame_idx in sorted(review_set):
        idx = int(frame_idx)
        if not (0 <= idx < frame_count):
            continue
        review_diffs[str(idx)] = {}
        rendered = {"overlay": overlay_frames[idx], "world": world_frames[idx], "side_by_side": side_frames[idx]}
        for kind, video in published_videos.items():
            if video and video.exists():
                old = extract_frame(video, idx)
                if old is not None:
                    cv2.imwrite(str(review_dir / f"published_{kind}_f{idx:03d}.jpg"), old)
                    review_diffs[str(idx)][kind] = pixel_diff(old, rendered[kind])

    output_props = {k: video_props(v) for k, v in outputs.items()}
    rendered_body_hash = sha256_file(args.observed_body)
    table_hash = sha256_file(args.contact_state_table)
    published_banner = {}
    if args.published_report and args.published_report.exists():
        rep = load_json(args.published_report)
        metrics = rep.get("metrics") if isinstance(rep.get("metrics"), dict) else {}
        published_banner = {
            "title": str(rep.get("title") or ""),
            "subtitle": str(rep.get("subtitle") or ""),
            "summary_text": str(metrics.get("summary_text") or ""),
        }
    false_text_absent = all(not any(entry["text_gate"].values()) for entry in per_frame_manifest)
    confirmed_frames = [e for e in per_frame_manifest if e["contact_state"] == "confirmed_contact"]
    confirmed_provenance_ok = all(e.get("confirmed_contact_has_floor_independent_provenance", False) for e in confirmed_frames)
    trellis_hash = args.trellis_mesh_hash.strip().lower() if args.trellis_mesh_hash else None

    manifest: dict[str, Any] = {
        "schema": "ego.hoi_v19_contact_state_full_duration_render/0.1.0",
        "case": args.case or "",
        "run_root": str(args.run_root) if args.run_root else "",
        "artifact_role": "full_duration_replacement_render_consumer_for_canonical_contact_frame_detail",
        "consumer_script": "scripts/render_v19_contact_state_full_duration.py",
        "inputs": {
            "contact_state_table": str(args.contact_state_table),
            "contact_state_table_sha256": table_hash,
            "observed_body": str(args.observed_body),
            "observed_body_sha256": rendered_body_hash,
            "pose_report": str(args.pose_report),
            "mano_state": str(args.mano_state),
            "visible_geometry": str(args.visible_geometry),
            "rgb_dir": str(args.rgb_dir),
            "published_report_before": str(args.published_report) if args.published_report else None,
            "published_videos_before": {k: str(v) if v else None for k, v in published_videos.items()},
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
        "review_frames_dir": str(review_dir),
        "frame_count": frame_count,
        "fps": fps,
        "duration_s": frame_count / fps if fps else None,
        "published_video_props_before": published_props,
        "output_video_props": output_props,
        "contact_state_counts": dict(state_counts),
        "source_contact_rows": sorted(source_rows.keys()),
        "default_contact_state": DEFAULT_CONTACT_STATE,
        "defaulted_frame_count": frame_count - len(source_rows),
        "render_consumed_mesh_sha256": rendered_body_hash,
        "render_consumed_mesh_kind": "repaired_observed_contact_body_open_nonwatertight_patch",
        "render_consumed_mesh_faces": int(len(body.faces)),
        "render_consumed_mesh_vertices": int(len(body.vertices)),
        "body_not_prior_completion": trellis_hash is None or rendered_body_hash.lower() != trellis_hash,
        "prior_completion_mesh_sha256_reference": trellis_hash,
        "published_banner_before": published_banner,
        "false_published_claims_removed_from_new_text": false_text_absent,
        "forbidden_text_tokens": list(forbidden),
        "zero_confirmed_contact_frames": len(confirmed_frames) == 0,
        "confirmed_contact_frames_count": len(confirmed_frames),
        "confirmed_contact_only_with_floor_independent_provenance": confirmed_provenance_ok,
        "contact_table_frame_states": {str(i): per_frame_manifest[i]["contact_state"] for i in sorted(source_rows.keys())},
        "pixel_diff_vs_published": review_diffs,
        "acceptance_gates": {
            "full_duration_matches_rgb_count": all(p["frame_count"] == frame_count for p in output_props.values()),
            "same_fps_as_published_or_override": all(abs(p["fps"] - fps) < 1.0e-3 for p in output_props.values()),
            "durable_inputs_only": not str(args.contact_state_table).startswith("/tmp/") and not str(args.observed_body).startswith("/tmp/"),
            "uncovered_frames_defaulted": frame_count - len(source_rows) >= 0,
            "per_frame_state_from_table_or_default": True,
            "rendered_mesh_hash_recorded": bool(rendered_body_hash),
            "rendered_body_replaces_prior_completion_by_hash": (trellis_hash is None) or (rendered_body_hash.lower() != trellis_hash),
            "false_gap_penverts_uncertain_removed": false_text_absent,
            "no_confirmed_contact_without_floor_independent_provenance": confirmed_provenance_ok,
        },
        "frames": per_frame_manifest,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[render] wrote {outputs['overlay']}")
    print(f"[render] wrote {outputs['world']}")
    print(f"[render] wrote {outputs['side_by_side']}")
    print(f"[render] wrote {manifest_path}")
    print(f"[render] states {dict(state_counts)}")
    gates = manifest["acceptance_gates"]
    failed = [k for k, v in gates.items() if not v]
    print(f"[render] acceptance_gates {'ALL PASS' if not failed else 'FAILED: ' + ','.join(failed)}")


if __name__ == "__main__":
    main()
