#!/usr/bin/env python3
"""Pose-liveness review artifact for HOT3D clip001850 keyboard.

Makes the inert rigid-object pose-graph defect VISIBLE as a review artifact,
not a status table. This implements the "object-pose graph liveness" frontier
card (subagent 21, causal card #1): the keyboard pose graph has
nfev=1 / cost=0 / zero active residuals, only 13/150 frames are direct fits,
65 frames are frozen at one identical pose, 35 are linear interpolation, and
the 13 fits jump up to 166 mm / 37 deg frame-to-frame.

The artifact consumes only existing CPU files:
  * the run-root rigid pose graph report (per-frame SE3 + provenance),
  * raw RGB frames,
  * per-frame visible geometry / cameras (T_world_camera_metric + intrinsics),
  * the durable repaired observed keyboard body (3221-face patch),
  * (optionally) the full-duration contact-state render for cross-reference.

It produces, under an output dir:
  * pose_liveness_contact_sheet.png  -- provenance-colored body overlay on the
    raw RGB for the key frames that expose frozen / direct / interpolated /
    jump provenance, each labelled with its pose source and frame-to-frame
    jump metric.
  * pose_trajectory_world.png        -- world-frame body-centroid trajectory
    over all 150 frames, coloured by provenance, exposing the frozen plateau
    and the teleport jumps.
  * jump_pair_f45_f46.png / jump_pair_f75_f76.png -- consecutive-frame strips
    that make the teleport visible.
  * manifest.json                    -- per-frame provenance + jump metrics +
    optimizer inertness summary + provenance counts.

The artifact does NOT claim contact. It labels measured vs held /
interpolated / ineligible pose provenance and shows object body placement.
No model inference, no GPU, no heavy compute. CPU-only OpenCV/PIL/trimesh/
matplotlib. Nothing staged, nothing committed.

Usage:
  python scripts/render_clip001850_pose_liveness_review.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/home/yiwen/ego_annotation")
RUN_ROOT = Path(
    "/data2/ego_annotation_outputs/v19_runs/"
    "20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)
DURABLE = Path(
    "/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706"
)

POSE_REPORT = RUN_ROOT / "measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json"
VISGEO = RUN_ROOT / "measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json"
RGB_DIR = RUN_ROOT / "input/raw_frame_manifest/rgb"
BODY = DURABLE / "keyboard_body_repair/repaired_observed_contact_body.ply"
CONTACT_STATE_VIDEO = DURABLE / "v19_contact_state_full_duration/v19_overlay.mp4"

OUTPUT_ROOT = Path("/tmp/clip001850_pose_liveness_review")

# Key frames that expose the four provenance defects.
REVIEW_FRAMES = [0, 11, 30, 36, 39, 45, 46, 75, 76, 90]

# ---------------------------------------------------------------------------
# Provenance classification
# ---------------------------------------------------------------------------

# Colour by pose source (what drives the SE3 this frame), keyed by a short tag.
SRC_COLOUR = {
    "direct": (60, 220, 90),     # green  -- measured, direct visible fit
    "held": (235, 70, 70),       # red    -- frozen / nearest-hold
    "interpolated": (70, 150, 245),  # blue -- linear interpolation
}
SRC_TAG = {
    "direct_visible_pose_observation_corrected": "direct",
    "nearest_visible_pose_hold": "held",
    "interpolated_between_visible_pose_observations": "interpolated",
}
SRC_LABEL = {
    "direct": "MEASURED (direct visible fit)",
    "held": "HELD (frozen nearest-hold)",
    "interpolated": "INTERPOLATED",
}
STATUS_SHORT = {
    "missing_initial_graph_pose": "missing_initial_graph_pose",
    "visible_surface_ineligible_for_rigid_pose_fit": "visible_surface_ineligible",
    "fit_to_visible_depth_samples": "fit_to_visible_depth_samples",
}


def classify(pose_row: dict) -> tuple[str, str]:
    """Return (prov_tag, status_short) for a pose row."""
    src = pose_row.get("temporal_pose_graph", {}).get("pose_source", "?")
    tag = SRC_TAG.get(src, "held")
    status = STATUS_SHORT.get(
        pose_row.get("pose_measurement_status", "?"), "?"
    )
    return tag, status


# ---------------------------------------------------------------------------
# Geometry helpers (same conventions as the accepted v19 render consumer)
# ---------------------------------------------------------------------------

def project_world(points_world, T_world_camera, intr):
    Tcw = np.linalg.inv(np.asarray(T_world_camera, dtype=float))
    ph = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1).T
    pc = Tcw @ ph
    z = pc[2]
    fx, fy, cx, cy = intr
    u = fx * pc[0] / z + cx
    v = fy * pc[1] / z + cy
    return np.stack([u, v, z], axis=1)


def body_world(body, pose_row):
    R = np.asarray(pose_row["rotation_world_from_completed_canonical_matrix"], dtype=float)
    t = np.asarray(pose_row["translation_world_m"], dtype=float)
    verts = np.asarray(body.vertices, dtype=float) @ R.T + t[None, :]
    return trimesh.Trimesh(vertices=verts, faces=body.faces, process=False)


def rotation_angle_deg(R_prev, R):
    dR = np.asarray(R_prev).T @ np.asarray(R)
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return float(ang)


# ---------------------------------------------------------------------------
# Body overlay drawing (provenance-coloured, backface-culled)
# ---------------------------------------------------------------------------

def draw_body_overlay(img_rgb, body, T_world_camera, intr, prov_tag, scale=1.0):
    """Draw the projected observed body faces, coloured by provenance.

    Fills are semi-transparent; edges are solid in the provenance colour so the
    body placement is readable even where the keyboard is small/occluded.
    """
    draw = ImageDraw.Draw(img_rgb, "RGBA")
    verts = np.asarray(body.vertices, dtype=float)
    faces = np.asarray(body.faces, dtype=np.int64)
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
        & (np.isfinite(cu))
        & (np.isfinite(cv_))
    )
    idx = np.where(visible)[0]
    idx = idx[np.argsort(-cz[idx])]  # far first
    col = SRC_COLOUR[prov_tag]
    fill = (col[0], col[1], col[2], 38)
    outline = (col[0], col[1], col[2], 210)
    su = u * scale
    sv = v * scale
    for fi in idx:
        a = (float(su[f0[fi]]), float(sv[f0[fi]]))
        b = (float(su[f1[fi]]), float(sv[f1[fi]]))
        c = (float(su[f2[fi]]), float(sv[f2[fi]]))
        if not all(np.isfinite([*a, *b, *c])):
            continue
        draw.polygon([a, b, c], fill=fill, outline=outline)
    # centroid marker
    cen = body.vertices.mean(axis=0)
    pcen = project_world(cen[None, :], T_world_camera, intr)[0]
    if pcen[2] > 0.03:
        x, y = pcen[0] * scale, pcen[1] * scale
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 255, 230), width=2)
        draw.line([x - 10, y, x + 10, y], fill=(255, 255, 255, 200), width=1)
        draw.line([x, y - 10, x, y + 10], fill=(255, 255, 255, 200), width=1)
    return img_rgb


def _font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Per-panel render
# ---------------------------------------------------------------------------

def render_panel(frame_idx, pose_row, vis_frame, body, jump, scale=0.5):
    """Render one review panel: RGB + provenance-coloured body + labels."""
    rgb_path = RGB_DIR / f"{frame_idx:06d}.jpg"
    img = Image.open(rgb_path).convert("RGB")
    W, H = img.size
    img = img.resize((int(W * scale), int(H * scale)))
    cam = vis_frame["camera"]
    Twc = np.asarray(cam["T_world_camera_metric"], dtype=float)
    intr = [float(x) for x in cam["intrinsics_fx_fy_cx_cy"]]
    bw = body_world(body, pose_row)
    prov_tag, status = classify(pose_row)
    img = draw_body_overlay(img, bw, Twc, intr, prov_tag, scale=scale)

    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    # provenance banner (top)
    font_big = _font(int(22 * scale / 0.5) if scale < 0.5 else 22)
    font_big = _font(max(18, int(22 * scale * 2)))
    font_sm = _font(max(13, int(15 * scale * 2)))
    col = SRC_COLOUR[prov_tag]
    banner_h = int(58 * scale * 2)
    draw.rectangle([0, 0, w, banner_h], fill=(0, 0, 0, 200))
    line1 = f"f{frame_idx}  |  {SRC_LABEL[prov_tag]}"
    draw.text((8, 4), line1, fill=col + (255,) if len(col) == 3 else col, font=font_big)
    line2 = f"status: {status}"
    draw.text((8, int(26 * scale * 2)), line2, fill=(225, 225, 225, 255), font=font_sm)

    # jump metric banner (bottom)
    dtrans = jump["dtrans_mm"]
    drot = jump["drot_deg"]
    jump_flag = dtrans >= 30.0 or drot >= 10.0
    jcol = (255, 90, 90) if jump_flag else (200, 200, 200)
    bh = int(34 * scale * 2)
    draw.rectangle([0, h - bh, w, h], fill=(0, 0, 0, 200))
    jtxt = f"d_frame = {dtrans:7.1f} mm / {drot:6.2f} deg"
    if jump_flag:
        jtxt += "   << JUMP"
    draw.text((8, h - bh + 6), jtxt, fill=jcol + (255,) if len(jcol) == 3 else jcol, font=font_sm)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory(pose_rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.array([r["translation_world_m"] for r in pose_rows], dtype=float)
    tags = [classify(r)[0] for r in pose_rows]
    n = len(pose_rows)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    # top: world XY plan view
    ax = axes[0]
    cmap = {"direct": "#3cdc5a", "held": "#eb4646", "interpolated": "#4696f5"}
    for tag in ("held", "interpolated", "direct"):
        m = [i for i in range(n) if tags[i] == tag]
        if not m:
            continue
        ax.scatter(ts[m, 0], ts[m, 1], c=cmap[tag], s=14 if tag == "direct" else 7,
                   label=SRC_LABEL[tag], zorder=3 if tag == "direct" else 2,
                   edgecolors="white", linewidths=0.3 if tag == "direct" else 0)
    ax.plot(ts[:, 0], ts[:, 1], color=(0.5, 0.5, 0.5, 0.4), lw=0.8, zorder=1)
    # annotate jumps
    dtrans = np.zeros(n)
    for i in range(1, n):
        dtrans[i] = np.linalg.norm(ts[i] - ts[i - 1]) * 1000
    for i in np.where(dtrans >= 30)[0]:
        ax.annotate(f"f{i} {dtrans[i]:.0f}mm", (ts[i, 0], ts[i, 1]),
                    fontsize=7, color="#cc0000",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_title("Keyboard body centroid trajectory (world frame) -- clip001850\n"
                 "frozen plateau + teleport jumps; pose graph is inert (nfev=1, cost=0)")
    ax.set_xlabel("world X [m]")
    ax.set_ylabel("world Y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # bottom: per-frame translation jump
    ax = axes[1]
    colors = [cmap[t] for t in tags]
    ax.bar(range(n), dtrans, color=colors, width=1.0)
    ax.axhline(30, color="#cc0000", lw=1, ls="--", label="jump flag (30mm)")
    ax.set_xlabel("frame idx")
    ax.set_ylabel("frame-to-frame dtrans [mm]")
    ax.set_title("Per-frame body-centroid jump (green=direct fit, red=held, blue=interpolated)\n"
                 "max jump 166 mm at f76; 65 frames frozen at 0 mm (held)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    for i in np.where(dtrans >= 30)[0]:
        ax.annotate(f"{dtrans[i]:.0f}", (i, dtrans[i]), fontsize=7,
                    color="#cc0000", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def build_contact_sheet(frames, panels, out_path, scale=0.42):
    """Tile per-frame panels into a contact sheet with a legend header."""
    n = len(panels)
    cols = 3
    rows = math.ceil(n / cols)
    ph, pw = panels[0].shape[:2]
    # legend header
    legend_h = 70
    sheet = Image.new("RGB", (cols * pw, rows * ph + legend_h), (15, 15, 15))
    draw = ImageDraw.Draw(sheet, "RGBA")
    font = _font(20)
    font_sm = _font(14)
    title = ("clip001850 keyboard pose-liveness review  --  inert pose graph: "
             "nfev=1, cost=0, 13/150 direct fits, 65 frozen, 35 interpolated, "
             "jumps up to 166 mm")
    draw.text((10, 6), title, fill=(240, 240, 240), font=font)
    # colour legend
    lx = 10
    ly = 38
    for tag, lab in (("direct", "MEASURED (direct fit)"),
                     ("held", "HELD (frozen)"),
                     ("interpolated", "INTERPOLATED")):
        c = SRC_COLOUR[tag]
        draw.rectangle([lx, ly, lx + 18, ly + 14], fill=c + (255,) if len(c) == 3 else c)
        draw.text((lx + 24, ly - 1), lab, fill=(220, 220, 220), font=font_sm)
        lx += 230
    draw.text((lx, ly - 1), "white crosshair = body centroid", fill=(200, 200, 200), font=font_sm)

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        sheet.paste(Image.fromarray(panel), (c * pw, r * ph + legend_h))
    sheet.save(out_path)


def build_jump_pair(fa, pa, va, fb, pb, vb, body, out_path, scale=0.5):
    """Side-by-side consecutive frames to expose a teleport jump."""
    ja = {"dtrans_mm": 0.0, "drot_deg": 0.0}
    jb = {"dtrans_mm": 0.0, "drot_deg": 0.0}
    # compute real jump for b
    Rb = np.asarray(pb["rotation_world_from_completed_canonical_matrix"])
    Ra = np.asarray(pa["rotation_world_from_completed_canonical_matrix"])
    jb["dtrans_mm"] = float(np.linalg.norm(
        np.asarray(pb["translation_world_m"]) - np.asarray(pa["translation_world_m"])) * 1000)
    jb["drot_deg"] = rotation_angle_deg(Ra, Rb)
    img_a = render_panel(fa, pa, va, body, ja, scale=scale)
    img_b = render_panel(fb, pb, vb, body, jb, scale=scale)
    h = max(img_a.shape[0], img_b.shape[0])
    w = img_a.shape[1] + img_b.shape[1] + 20
    canvas = Image.new("RGB", (w, h + 36), (12, 12, 12))
    d = ImageDraw.Draw(canvas, "RGBA")
    d.text((10, 6), f"JUMP f{fa} -> f{fb}:  dtrans={jb['dtrans_mm']:.1f} mm / "
          f"drot={jb['drot_deg']:.2f} deg  (physically impossible for a desk keyboard)",
          fill=(255, 100, 100), font=_font(18))
    canvas.paste(Image.fromarray(img_a), (0, 36))
    canvas.paste(Image.fromarray(img_b), (img_a.shape[1] + 20, 36))
    canvas.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = json.load(open(POSE_REPORT))
    pose_rows = {r["frame_idx"]: r for r in report["pose_rows"]}
    visgeo = json.load(open(VISGEO))
    vis_frames = {f["frame_idx"]: f for f in visgeo["frames"]}
    body = trimesh.load(str(BODY))
    n = len(report["pose_rows"])

    # per-frame jump metrics (consecutive frames)
    jumps = {}
    prev = None
    for fi in sorted(pose_rows):
        r = pose_rows[fi]
        if prev is None:
            jumps[fi] = {"dtrans_mm": 0.0, "drot_deg": 0.0, "from_frame": None}
        else:
            pr = pose_rows[prev]
            dt = float(np.linalg.norm(
                np.asarray(r["translation_world_m"]) - np.asarray(pr["translation_world_m"])) * 1000)
            dr = rotation_angle_deg(
                pr["rotation_world_from_completed_canonical_matrix"],
                r["rotation_world_from_completed_canonical_matrix"])
            jumps[fi] = {"dtrans_mm": dt, "drot_deg": dr, "from_frame": prev}
        prev = fi

    # ---- contact sheet ----
    panels = [render_panel(fi, pose_rows[fi], vis_frames[fi], body, jumps[fi])
              for fi in REVIEW_FRAMES]
    cs_path = OUTPUT_ROOT / "pose_liveness_contact_sheet.png"
    build_contact_sheet(REVIEW_FRAMES, panels, cs_path)

    # ---- trajectory plot ----
    traj_path = OUTPUT_ROOT / "pose_trajectory_world.png"
    plot_trajectory([pose_rows[i] for i in sorted(pose_rows)], traj_path)

    # ---- jump pairs ----
    build_jump_pair(45, pose_rows[45], vis_frames[45],
                    46, pose_rows[46], vis_frames[46], body,
                    OUTPUT_ROOT / "jump_pair_f45_f46.png")
    build_jump_pair(75, pose_rows[75], vis_frames[75],
                    76, pose_rows[76], vis_frames[76], body,
                    OUTPUT_ROOT / "jump_pair_f75_f76.png")

    # ---- manifest ----
    import collections
    status_counts = collections.Counter(
        pose_rows[i]["pose_measurement_status"] for i in sorted(pose_rows))
    src_counts = collections.Counter(
        pose_rows[i]["temporal_pose_graph"].get("pose_source", "?")
        for i in sorted(pose_rows))
    all_jumps = [jumps[i]["dtrans_mm"] for i in sorted(jumps) if jumps[i]["from_frame"] is not None]
    opt = report.get("optimizer", {})
    manifest = {
        "artifact": "clip001850 keyboard pose-liveness review",
        "purpose": ("Expose the inert rigid-object pose-graph defect (D5 graph "
                    "inertness + D1 pose unobserved) as a visible review artifact. "
                    "Labels measured vs held/interpolated/ineligible pose provenance "
                    "and shows object body placement. Does NOT claim contact."),
        "inputs": {
            "pose_report": str(POSE_REPORT),
            "visible_geometry": str(VISGEO),
            "rgb_dir": str(RGB_DIR),
            "observed_body": str(BODY),
            "body_faces": int(len(body.faces)),
            "body_verts": int(len(body.vertices)),
        },
        "optimizer_inertness": {
            "nfev": opt.get("nfev"),
            "cost": opt.get("cost"),
            "residual_rms_before": opt.get("residual_rms_before"),
            "residual_rms_after": opt.get("residual_rms_after"),
            "nonpenetration_target_frame_count": report.get("nonpenetration_target_frame_count"),
            "graph_frame_count": report.get("graph_frame_count"),
            "graph_frames": report.get("graph_frames"),
            "verdict": ("inert: zero active residuals; temporal/nonpenetration factors "
                        "have no support; pose is per-frame depth fit or held/interp fill, "
                        "not a graph solve"),
        },
        "provenance_counts": {
            "pose_measurement_status": dict(status_counts),
            "pose_source": dict(src_counts),
            "total_frames": n,
            "direct_fit_frames": int(status_counts.get("fit_to_visible_depth_samples", 0)),
            "missing_initial_graph_pose": int(status_counts.get("missing_initial_graph_pose", 0)),
            "visible_surface_ineligible": int(status_counts.get("visible_surface_ineligible_for_rigid_pose_fit", 0)),
        },
        "jump_metrics_summary": {
            "median_mm": float(np.median(all_jumps)),
            "p90_mm": float(np.percentile(all_jumps, 90)),
            "max_mm": float(np.max(all_jumps)),
            "max_jump_frame": int(sorted(pose_rows)[int(np.argmax(all_jumps)) + 1]),
            "jump_flag_threshold_mm": 30.0,
            "jump_frame_count_>=30mm": int(np.sum(np.array(all_jumps) >= 30.0)),
        },
        "review_frames": [],
        "all_frame_states": [],
        "outputs": {
            "contact_sheet": str(cs_path),
            "trajectory_plot": str(traj_path),
            "jump_pair_f45_f46": str(OUTPUT_ROOT / "jump_pair_f45_f46.png"),
            "jump_pair_f75_f76": str(OUTPUT_ROOT / "jump_pair_f75_f76.png"),
            "manifest": str(OUTPUT_ROOT / "manifest.json"),
        },
        "scope": {
            "contact_claimed": False,
            "gpu_used": False,
            "model_inference": False,
            "cpu_only": True,
        },
    }
    for fi in REVIEW_FRAMES:
        r = pose_rows[fi]
        tag, status = classify(r)
        j = jumps[fi]
        manifest["review_frames"].append({
            "frame_idx": fi,
            "provenance_tag": tag,
            "provenance_label": SRC_LABEL[tag],
            "pose_measurement_status": r["pose_measurement_status"],
            "pose_source": r["temporal_pose_graph"].get("pose_source"),
            "translation_world_m": r["translation_world_m"],
            "dtrans_mm": round(j["dtrans_mm"], 2),
            "drot_deg": round(j["drot_deg"], 3),
            "jump": bool(j["dtrans_mm"] >= 30.0 or j["drot_deg"] >= 10.0),
        })
    for fi in sorted(pose_rows):
        r = pose_rows[fi]
        tag, status = classify(r)
        j = jumps[fi]
        manifest["all_frame_states"].append({
            "frame_idx": fi,
            "provenance_tag": tag,
            "status": status,
            "dtrans_mm": round(j["dtrans_mm"], 2),
            "drot_deg": round(j["drot_deg"], 3),
        })
    with open(OUTPUT_ROOT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[pose-liveness] wrote {OUTPUT_ROOT}")
    for p in sorted(OUTPUT_ROOT.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")
    print(f"[pose-liveness] optimizer: nfev={opt.get('nfev')} cost={opt.get('cost')} "
          f"direct={status_counts.get('fit_to_visible_depth_samples',0)} "
          f"frozen={status_counts.get('missing_initial_graph_pose',0)} "
          f"max_jump={np.max(all_jumps):.1f}mm")


if __name__ == "__main__":
    main()
