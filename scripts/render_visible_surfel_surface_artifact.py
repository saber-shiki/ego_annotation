#!/usr/bin/env python3
"""Render a full-duration diagnostic video of the prediction-side *visible surfel
surface* for one object track.

This is a *lower-cost artifact-changing render*: it does NOT run any model
inference. It consumes already-produced prediction-side visible metric surfel
samples (the compatibility archive: 375k world-frame vertices, 150 frames x 2500)
plus per-frame camera poses, intrinsics, RGB frames and SAM2 masks, and renders:

  * ``v19_overlay.mp4``  : surfel points projected over RGB + SAM2 mask tint.
  * ``v19_world.mp4``    : camera-relative 3D scatter of the surfel cloud, per
                           frame (NOT a global/world-pose view).
  * ``v19_side_by_side.mp4`` : overlay | world panel.
  * ``v19_still_sheet.png``  : representative frames contact sheet.
  * ``v19_render_manifest.json`` + ``v19_render_report.json``.

Every visible mark in every frame is driven by real surfel vertices: each frame's
own 2500 world-frame vertices are transformed into that frame's camera frame with
``inv(T_world_camera_metric)`` and projected with the constant pinhole intrinsics.

CLAIM SCOPE (enforced in the manifest):
  observed_partial_visible_surface / diagnostic_render.
  annotation_ready = false for pose, contact, occlusion, nonpenetration.
  No completed/TRELLIS mesh, no faces as evidence (NPZ faces are non-evidential
  loader padding), no absolute object pose, no GT.

CPU-only; safe to run on the A800 host. No GPU, no model inference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# --------------------------------------------------------------------------- IO

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def by_frame(frames: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(r["frame_idx"]): r for r in frames if isinstance(r, dict) and "frame_idx" in r}


# ---------------------------------------------------------------- geometry core

def world_to_camera(world: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """world-frame points -> camera-frame points via inverse camera-to-world pose."""
    Tcw = np.linalg.inv(np.asarray(T_wc, dtype=np.float64))
    hom = np.c_[world, np.ones(len(world), dtype=np.float64)]
    return (hom @ Tcw.T)[:, :3]


def project_to_pixels(cam_xyz: np.ndarray, intr: np.ndarray, src_size: int, out_size: int) -> np.ndarray:
    """Pinhole project camera-frame points to output-resolution integer pixels.

    intr = [fx, fy, cx, cy] at src_size. Points with cam_z<=0 are dropped.
    Returns int array (N,2) of (x,y) in [0,out_size)."""
    fx, fy, cx, cy = intr
    z = cam_xyz[:, 2]
    front = z > 0
    u = fx * cam_xyz[front, 0] / z[front] + cx
    v = fy * cam_xyz[front, 1] / z[front] + cy
    scale = float(out_size) / float(src_size)
    xs = np.rint(u * scale).astype(np.int64)
    ys = np.rint(v * scale).astype(np.int64)
    keep = (xs >= 0) & (xs < out_size) & (ys >= 0) & (ys < out_size)
    return np.stack([xs[keep], ys[keep]], axis=1), int(front.sum())


# ---------------------------------------------------------------- rendering

def make_depth_cmap() -> LinearSegmentedColormap:
    # near=warm/bright, far=cool; reads well over RGB and on black 3D panel
    return plt.get_cmap("turbo")


def depth_color(values: np.ndarray, vlo: float, vhi: float, cmap) -> np.ndarray:
    t = np.clip((values - vlo) / max(1e-9, (vhi - vlo)), 0.0, 1.0)
    rgba = cmap(t)
    return (rgba[:, :3] * 255.0).astype(np.uint8)


def draw_overlay(
    rgb: np.ndarray,
    sam2_mask: np.ndarray,
    pts_xy: np.ndarray,
    pt_colors: np.ndarray,
    frame_idx: int,
    time_s: float,
    on_sam2_frac: float,
    n_front: int,
    n_total: int,
    median_depth_m: float,
) -> np.ndarray:
    """RGB + translucent SAM2 tint + projected depth-colored surfel points + HUD."""
    out = rgb.copy()
    H, W = out.shape[:2]
    # --- SAM2 keyboard mask: green translucent fill + outline ---
    mbool = sam2_mask > 127
    if mbool.any():
        tint = np.zeros_like(out)
        tint[mbool] = (0, 230, 120)  # green BGR
        out = cv2.addWeighted(out, 1.0, tint, 0.22, 0.0)
        contours, _ = cv2.findContours(mbool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 230, 120), 1, cv2.LINE_8)
    # --- surfel points: depth-colored dots with thin dark ring for contrast ---
    overlay = np.zeros_like(out)
    for (x, y), col in zip(pts_xy.tolist(), pt_colors.tolist()):
        cv2.circle(overlay, (int(x), int(y)), 2, (int(col[0]), int(col[1]), int(col[2])), -1, cv2.LINE_AA)
    # dark ring pass
    ring = np.zeros_like(out)
    for (x, y) in pts_xy.tolist():
        cv2.circle(ring, (int(x), int(y)), 3, (0, 0, 0), 1, cv2.LINE_AA)
    out = cv2.addWeighted(out, 1.0, ring, 0.55, 0.0)
    out = cv2.addWeighted(out, 1.0, overlay, 1.0, 0.0)
    # --- HUD ---
    hud_lines = [
        "observed_partial_visible_surface | diagnostic_render",
        "observation_only=true; downstream state claims disabled",
        f"frame {frame_idx:03d}/149  t={time_s:.3f}s  surfels {n_front}/{n_total}",
        f"on-SAM2-mask {on_sam2_frac*100:4.1f}%   median cam-depth {median_depth_m:.3f} m",
    ]
    y0 = 8
    for i, line in enumerate(hud_lines):
        y = y0 + i * 20
        cv2.rectangle(out, (4, y - 2), (min(W - 4, 720), y + 16), (0, 0, 0), -1)
        cv2.putText(out, line, (8, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    # depth-colorbar legend
    cv2.putText(out, "surfels colored by camera depth (near->far)",
                (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def render_3d_panel(
    cam_xyz: np.ndarray,
    ax_lim,
    z_lo: float,
    z_hi: float,
    cmap,
    frame_idx: int,
    time_s: float,
    fig,
    ax,
    out_size: int,
) -> np.ndarray:
    """Camera-relative 3D scatter -> RGBA numpy image (out_size x out_size)."""
    ax.clear()
    if len(cam_xyz):
        z = cam_xyz[:, 2]
        t = np.clip((z - z_lo) / max(1e-9, (z_hi - z_lo)), 0.0, 1.0)
        cols = cmap(t)
        # sort by depth so nearer points draw on top
        order = np.argsort(-z)
        cdx = cam_xyz[order]
        ax.scatter(cdx[:, 0], cdx[:, 1], cdx[:, 2], c=cols[order], s=4, edgecolors="none", depthshade=False)
    # camera origin
    ax.scatter([0], [0], [0], marker="o", s=30, c="white", edgecolors="black")
    ax.set_xlim(*ax_lim[0]); ax.set_ylim(*ax_lim[1]); ax.set_zlim(*ax_lim[2])
    ax.set_xlabel("cam X (m) -> right")
    ax.set_ylabel("cam Y (m) -> down")
    ax.set_zlabel("cam Z (m) -> forward")
    ax.set_title(f"camera-relative surfel cloud  frame {frame_idx:03d}  t={time_s:.2f}s\n"
                 "observed samples only; no cross-frame placement", fontsize=8)
    ax.view_init(elev=18, azim=-64)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # ARGB -> RGB
    argb = buf.reshape((int(fig.bbox.height), int(fig.bbox.width), 4))
    rgba = argb[..., [1, 2, 3, 0]]
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if rgb.shape[0] != out_size or rgb.shape[1] != out_size:
        rgb = cv2.resize(rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return rgb


# ---------------------------------------------------------------- encoding

def encode_mp4(frame_dir: Path, pattern: str, out_mp4: Path, fps: float) -> None:
    """PNG sequence -> mp4 via system ffmpeg (yuv420p, widely playable)."""
    if out_mp4.exists():
        out_mp4.unlink()
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_dir / pattern),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_mp4}:\n{proc.stderr[-2000:]}")


def count_frames_png(d: Path) -> int:
    return len([p for p in d.glob("*.png")])


# ---------------------------------------------------------------- main

def build(args: argparse.Namespace) -> None:
    run = Path(args.run_root)
    base = Path(args.base_run)
    out_dir = run / "render_visible_surfel_surface"
    tmp = run / ".render_tmp_surfel"
    for s in ["overlay", "world", "side"]:
        (tmp / s).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    PY = sys.executable
    print(f"[render] python={PY}  run={run}")

    # ---- load inputs ----
    arc = np.load(run / "r3_visible_surfel_compat" / "visible_surfel_compat_archive.npz", allow_pickle=True)
    V = np.ascontiguousarray(arc["vertices"], dtype=np.float64)
    voff = arc["vertex_offsets"]
    frame_idx_arc = arc["frame_idx"]
    faces_non_evidential = bool(arc["faces_non_evidential"])

    ann = load_json(base / "state" / "base_annotations" / "annotations_v19_base.json")
    ann_frames = by_frame(ann["frames"])
    rgb_manifest = load_json(base / "input" / "raw_frame_manifest" / "manifest.json")
    rgb_frames = by_frame(rgb_manifest["frames"])
    sam2 = load_json(run / "prediction_masks" / "keyboard_sam2_manifest.json")
    sam2_frames = by_frame(sam2["frames"])
    calib = load_json(base / "state" / "calibration" / "v19_camera_calibration_contract.json")

    intr_src = np.array(calib["intrinsics_fx_fy_cx_cy"], dtype=np.float64)  # at source size
    src_size = int(calib["source_size"]["width"])
    fps = float(ann["raw_video"]["fps"])
    n_frames = int(ann["raw_video"]["frame_count"])

    # output overlay resolution = manifest RGB resolution
    out_size = int(rgb_manifest["manifest_width"])
    assert int(rgb_manifest["manifest_height"]) == out_size, "RGB manifest not square"

    frame_order = list(range(n_frames))

    # ---- precompute per-frame camera coords + projections ----
    pre = []
    all_camz = []
    for f in frame_order:
        world = V[voff[f]:voff[f + 1]]
        T = ann_frames[f]["camera"]["T_world_camera_metric"]
        cam = world_to_camera(world, T)
        pts, n_front = project_to_pixels(cam, intr_src, src_size, out_size)
        front = cam[:, 2] > 0
        camf = cam[front]
        pre.append({"world": world, "cam": cam, "cam_front": camf, "pts": pts, "n_front": n_front})
        if len(camf):
            all_camz.append(camf[:, 2])
    allz = np.concatenate(all_camz) if all_camz else np.array([0.5])
    z_lo, z_hi = float(np.percentile(allz, 2)), float(np.percentile(allz, 98))

    # camera-relative axis limits (consistent across frames -> motion visible)
    allx = np.concatenate([p["cam_front"][:, 0] for p in pre if len(p["cam_front"])])
    ally = np.concatenate([p["cam_front"][:, 1] for p in pre if len(p["cam_front"])])
    pad = 0.02
    ax_lim = (
        (float(np.percentile(allx, 1)) - pad, float(np.percentile(allx, 99)) + pad),
        (float(np.percentile(ally, 1)) - pad, float(np.percentile(ally, 99)) + pad),
        (max(0.0, z_lo - pad), z_hi + pad),
    )

    cmap = make_depth_cmap()
    DPI = 150
    FS = out_size / DPI  # inches -> out_size px
    fig = plt.figure(figsize=(FS, FS), dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")
    plt.tight_layout(pad=0.5)

    # ---- per-frame render ----
    rows = []
    for idx, f in enumerate(frame_order):
        p = pre[f]
        rgb_path = rgb_frames[f]["rgb"]
        sam2_path = sam2_frames[f]["mask"]
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise RuntimeError(f"cannot read RGB {rgb_path}")
        if rgb.shape[0] != out_size:
            rgb = cv2.resize(rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)
        sam2_mask = cv2.imread(sam2_path, cv2.IMREAD_GRAYSCALE)
        if sam2_mask is None:
            raise RuntimeError(f"cannot read SAM2 mask {sam2_path}")
        if sam2_mask.shape[0] != out_size:
            sam2_mask = cv2.resize(sam2_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        sam2_bool = sam2_mask > 127

        camf = p["cam_front"]
        if len(camf):
            colors = depth_color(camf[:, 2], z_lo, z_hi, cmap)
            pts_for_color = p["pts"]  # already filtered to in-bounds; align length below
        else:
            colors = np.zeros((0, 3), np.uint8)

        # on-SAM2 fraction over projected (in-bounds) points
        if len(p["pts"]):
            on = sam2_bool[p["pts"][:, 1], p["pts"][:, 0]].mean()
        else:
            on = 0.0
        med_z = float(np.median(camf[:, 2])) if len(camf) else float("nan")

        overlay_img = draw_overlay(
            rgb, sam2_mask, p["pts"], colors[:len(p["pts"])], f, f / fps, float(on),
            p["n_front"], int(len(p["world"])), med_z,
        )
        world_img = render_3d_panel(camf, ax_lim, z_lo, z_hi, cmap, f, f / fps, fig, ax, out_size)
        side = np.full((out_size, out_size * 2 + 4, 3), 24, np.uint8)
        side[:, :out_size] = overlay_img
        side[:, out_size + 4:] = world_img

        cv2.imwrite(str(tmp / "overlay" / f"{f:06d}.png"), overlay_img)
        cv2.imwrite(str(tmp / "world" / f"{f:06d}.png"), world_img)
        cv2.imwrite(str(tmp / "side" / f"{f:06d}.png"), side)

        rows.append({
            "frame_idx": f,
            "time_s": f / fps,
            "surfels_total": int(len(p["world"])),
            "surfels_in_front_of_camera": p["n_front"],
            "projected_in_bounds_pts": int(len(p["pts"])),
            "projected_on_sam2_mask_frac": round(float(on), 4),
            "cam_depth_m": {
                "median": round(med_z, 5),
                "p05": round(float(np.percentile(camf[:, 2], 5)), 5) if len(camf) else None,
                "p95": round(float(np.percentile(camf[:, 2], 95)), 5) if len(camf) else None,
            },
        })
        if idx % 25 == 0 or idx == n_frames - 1:
            print(f"[render] frame {f:03d}/{n_frames-1} on_sam2={on*100:4.1f}% medZ={med_z:.3f}")

    plt.close(fig)

    # ---- encode ----
    overlay_mp4 = out_dir / "v19_overlay.mp4"
    world_mp4 = out_dir / "v19_world.mp4"
    side_mp4 = out_dir / "v19_side_by_side.mp4"
    for d, pat, mp4 in [("overlay", "%06d.png", overlay_mp4), ("world", "%06d.png", world_mp4), ("side", "%06d.png", side_mp4)]:
        print(f"[encode] {mp4.name} ...")
        encode_mp4(tmp / d, pat, mp4, fps)

    # ---- still sheet: 8 frames spread across timeline ----
    pick = [int(round(i)) for i in np.linspace(0, n_frames - 1, 8)]
    cells = []
    for f in pick:
        ov = cv2.imread(str(tmp / "overlay" / f"{f:06d}.png"))
        wo = cv2.imread(str(tmp / "world" / f"{f:06d}.png"))
        cap_h = out_size // 2
        ov_s = cv2.resize(ov, (cap_h, cap_h))
        wo_s = cv2.resize(wo, (cap_h, cap_h))
        cells.append(np.hstack([ov_s, wo_s]))
    sheet = np.vstack(cells)
    cv2.putText(sheet, "v19 observed_partial_visible_surface still sheet  (left: overlay | right: camera-relative 3D)",
                (8, sheet.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    sheet_path = out_dir / "v19_still_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)

    # ---- validate encoded frame counts ----
    val = {}
    for mp4 in [overlay_mp4, world_mp4, side_mp4]:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height,duration",
             "-of", "json", str(mp4)], capture_output=True, text=True)
        st = json.loads(probe.stdout)["streams"][0]
        val[mp4.name] = {
            "nb_read_frames": int(st.get("nb_read_frames", -1)),
            "r_frame_rate": st.get("r_frame_rate"),
            "width": int(st["width"]), "height": int(st["height"]),
            "duration_s_reported": float(st.get("duration", -1)),
        }

    on_fracs = [r["projected_on_sam2_mask_frac"] for r in rows]
    medz = [r["cam_depth_m"]["median"] for r in rows]
    manifest = {
        "schema": "v19_visible_surfel_surface_render_manifest_v1",
        "render_method": "render_visible_surfel_surface_artifact",
        "claim_scope": "observed_partial_visible_surface diagnostic render; substrate only; not pose/contact/occlusion/nonpenetration",
        "labels": {
            "surface_kind": "observed_partial_visible_surface",
            "render_kind": "diagnostic_render",
            "annotation_ready": {
                "pose": False,
                "contact": False,
                "occlusion": False,
                "nonpenetration": False,
                "surface_substrate": True,
            },
        },
        "marks_driven_by": "real prediction-side visible metric surfel vertices (per-frame own 2500 world vertices -> camera frame via inv(T_world_camera_metric) -> pinhole project at constant intrinsics)",
        "provenance": {
            "run_root": str(run),
            "base_run": str(base),
            "surfels_npz": str(run / "r3_visible_surfel_compat" / "visible_surfel_compat_archive.npz"),
            "surfels_source_field": "objects[*].visible_geometry_candidate.world_vertices_sample_m",
            "camera_poses": "annotations_v19_base.frames[*].camera.T_world_camera_metric (hawor_npz_R_c2w_t_c2w)",
            "intrinsics": "v19_camera_calibration_contract.json constant pinhole fx=fy=537.021 cx=722.288 cy=720.896 @1408",
            "rgb_manifest": str(base / "input" / "raw_frame_manifest" / "manifest.json"),
            "sam2_manifest": str(run / "prediction_masks" / "keyboard_sam2_manifest.json"),
            "faces_non_evidential_loader_padding": faces_non_evidential,
        },
        "forbidden_in_this_artifact": [
            "completed/TRELLIS mesh", "faces as geometry evidence", "absolute object pose rows",
            "contact", "occlusion ownership", "nonpenetration", "GT"],
        "video": {
            "fps": fps, "frame_count": n_frames, "duration_s": n_frames / fps,
            "source_size": [src_size, src_size], "overlay_size": [out_size, out_size],
            "duration_matches_raw": abs((n_frames / fps) - float(ann["raw_video"]["duration_s"])) < 1e-6,
        },
        "projection": {
            "convention": "world -> camera via inv(T_world_camera_metric); u=fx*X/Z+cx, v=fy*Y/Z+cy at src_size; scaled x out/src",
            "intrinsics_fx_fy_cx_cy_at_src": [float(x) for x in intr_src],
            "src_to_out_scale": out_size / src_size,
            "depth_color_range_m": [z_lo, z_hi],
        },
        "files": {
            "v19_overlay": str(overlay_mp4),
            "v19_world": str(world_mp4),
            "v19_side_by_side": str(side_mp4),
            "v19_still_sheet": str(sheet_path),
        },
        "validation": val,
        "surfel_on_sam2_mask_frac": {
            "median": float(np.median(on_fracs)), "min": float(np.min(on_fracs)),
            "max": float(np.max(on_fracs)),
        },
        "cam_depth_m_summary": {"median_of_medians": float(np.median(medz)),
                                "min": float(np.min(medz)), "max": float(np.max(medz))},
        "rows": rows,
    }
    (out_dir / "v19_render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "status": "ok",
        "method": "render_visible_surfel_surface_artifact",
        "labels": manifest["labels"],
        "marks_driven_by": manifest["marks_driven_by"],
        "frame_count": n_frames,
        "duration_s": n_frames / fps,
        "duration_matches_raw": manifest["video"]["duration_matches_raw"],
        "validation": val,
        "all_videos_150_frames": all(v["nb_read_frames"] == n_frames for v in val.values()),
        "surfel_on_sam2_mask_frac_median": manifest["surfel_on_sam2_mask_frac"]["median"],
        "files": manifest["files"],
        "still_sheet": str(sheet_path),
    }
    (out_dir / "v19_render_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # cleanup tmp
    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    print(json.dumps({k: v for k, v in report.items() if k not in ("validation",)}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--base-run", required=True,
                   help="the supportgate base run holding base_annotations, raw_frame_manifest, calibration")
    p.add_argument("--keep-tmp", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
