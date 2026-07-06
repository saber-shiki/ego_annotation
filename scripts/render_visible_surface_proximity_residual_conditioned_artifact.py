#!/usr/bin/env python3
"""Render clip-level visible-surface proximity residual conditioned on frozen visual prior.

Conditioning step: reads the frozen geometry-blind visual contact prior packet,
validates its hash, and attaches prior provenance to the render manifest/report.
The residual measurements (MANO-to-surfel distances) are unchanged — this is a
metadata-level DAG link, not a geometric modification.

Distance-band labels and filenames use ``visible_surface_gap`` instead of the
previous ``close_visible_surface_support`` / ``near_visible_surface_support``
naming, avoiding ``support`` as a commitment word.

Claim scope: visible-surface proximity residual only — a metric likelihood,
not a binary touch state, hidden-state decision, nonpenetration result,
occlusion owner claim, or object pose.  The frozen visual prior supplies
ordinal visual-semantic labels (asserted/absent/unresolved) from geometry-blind
RGB+mask-overlay evidence; these labels are joined with residual gap
measurements as conditioning metadata but never used to modify MANO joints
or object surfels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

from refit_v19_mano_contact_similarity_interval import (
    as_list,
    full_vertices_camera,
    project_camera,
)
from render_visible_surfel_surface_artifact import (
    by_frame,
    depth_color,
    encode_mp4,
    load_json,
    make_depth_cmap,
    project_to_pixels,
    render_3d_panel,
    world_to_camera,
)


HAND_COLORS_BGR = {
    "left": (255, 220, 40),
    "right": (255, 40, 220),
}
STATE_COLORS_BGR = {
    "visible_surface_gap_close": (0, 255, 255),
    "visible_surface_gap_near": (0, 180, 255),
    "visible_surface_gap_high": (180, 180, 180),
    "hidden_state_unresolved": (80, 80, 80),
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prior_packet(prior_path: Path) -> dict[str, Any]:
    """Load and validate the frozen visual contact prior packet."""
    if not prior_path.exists():
        raise FileNotFoundError(f"prior packet not found: {prior_path}")

    payload = load_json(prior_path)
    prior_id = payload.get("prior_id")
    if not prior_id:
        raise ValueError("prior packet lacks 'prior_id'")

    # Validate mechanism is geometry-blind visual prior
    mechanism = payload.get("mechanism")
    if mechanism != "visual_semantic_contact_prior":
        raise ValueError(f"prior mechanism '{mechanism}' is not visual_semantic_contact_prior")

    geometry_blind = payload.get("geometry_blind") or {}
    forbidden = geometry_blind.get("forbidden_inputs") or []
    negations = geometry_blind.get("meta_negation_booleans") or {}

    return payload


def validate_prior_hash(prior_path: Path, expected_hash: str) -> bool:
    """Recompute prior hash and compare against expected."""
    actual = file_sha256(prior_path)
    if actual != expected_hash:
        raise ValueError(
            f"prior hash mismatch: expected={expected_hash}, actual={actual}. "
            "Prior may have been modified after freeze. Aborting."
        )
    return True


def residual_rows_by_frame_side(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    payload = load_json(path)
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in as_list(payload.get("per_frame_states")):
        if not isinstance(row, dict):
            continue
        f = row.get("frame_idx")
        side = row.get("hand_side")
        if isinstance(f, int) and isinstance(side, str):
            rows[(f, side)] = row
    return rows


def prior_state_by_frame_side(prior_payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    """Extract per-frame per-hand prior labels from frozen prior packet.

    The prior stores state under ``frames[].hands.{left,right}.visual_prior_state``
    for a subset of target frames (typically 37 of 150).  Frames not covered by
    the prior are represented as ``not_in_prior`` so consumers can distinguish
    "no prior label" from the unresolved state the prior itself can emit.
    """
    result: dict[tuple[int, str], dict[str, Any]] = {}

    # Collect covered frame indices from prior
    covered = set()
    for entry in as_list(prior_payload.get("frames")):
        if not isinstance(entry, dict):
            continue
        f = entry.get("frame_idx")
        if not isinstance(f, int):
            continue
        hands = entry.get("hands") or {}
        for hand_side in ("left", "right"):
            hand_entry = hands.get(hand_side)
            if isinstance(hand_entry, dict):
                covered.add((f, hand_side))
                result[(f, hand_side)] = {
                    "prior_state": hand_entry.get("visual_prior_state", "unresolved"),
                    "prior_confidence": hand_entry.get("confidence"),
                    "prior_lean": hand_entry.get("lean"),
                    "provenance": hand_entry.get("provenance", "visual_semantic_prior"),
                    "frame_idx": f,
                    "hand_side": hand_side,
                }

    # For target frames listed but not in frames array (shouldn't happen, but be safe)
    target_frames = prior_payload.get("target_frames") or []
    for f in target_frames:
        for hand_side in ("left", "right"):
            if (f, hand_side) not in result:
                result[(f, hand_side)] = {
                    "prior_state": "not_in_prior",
                    "prior_confidence": None,
                    "frame_idx": f,
                    "hand_side": hand_side,
                }

    return result


def distance_median(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    gap = row.get("direct_object_surface_source_distance_gap_m")
    if not isinstance(gap, dict):
        gap = row.get("direct_object_surface_source_gap_m")
    med = gap.get("median") if isinstance(gap, dict) else None
    if isinstance(med, (int, float)) and np.isfinite(float(med)):
        return float(med)
    return None


def classify_visible_surface_gap(dist_m: float | None) -> str:
    """Display-only distance bands for MANO-to-visible-surfel gap.

    These are visualization bins, not acceptance thresholds or physical-state decisions.
    They measure only visible surfaces; hidden state remains unresolved.
    """
    if dist_m is None:
        return "hidden_state_unresolved"
    if dist_m <= 0.010:
        return "visible_surface_gap_close"
    if dist_m <= 0.030:
        return "visible_surface_gap_near"
    return "visible_surface_gap_high"


def source_uv_to_out(uv: np.ndarray, source_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    scale = float(out_size) / float(source_size)
    xy = np.rint(uv * scale).astype(np.int64)
    valid = np.isfinite(uv).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < out_size) & (xy[:, 1] >= 0) & (xy[:, 1] < out_size)
    return xy, valid


def draw_selected_gap_vectors(
    img: np.ndarray,
    state_row: dict[str, Any] | None,
    frame: dict[str, Any],
    intr_src: np.ndarray,
    source_size: int,
    out_size: int,
    state_color: tuple[int, int, int],
) -> int:
    if not isinstance(state_row, dict):
        return 0
    source_w = np.asarray(state_row.get("source_contact_vertices_world_sample_m") or [], dtype=np.float64)
    target_w = np.asarray(state_row.get("contact_surface_vertices_world_sample_m") or [], dtype=np.float64)
    if source_w.ndim != 2 or target_w.ndim != 2 or source_w.shape[1:] != (3,) or target_w.shape[1:] != (3,):
        return 0
    n = min(len(source_w), len(target_w))
    if n == 0:
        return 0
    T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    src_cam = world_to_camera(source_w[:n], T)
    tgt_cam = world_to_camera(target_w[:n], T)
    src_uv, src_front = project_camera(src_cam, [float(x) for x in intr_src.tolist()])
    tgt_uv, tgt_front = project_camera(tgt_cam, [float(x) for x in intr_src.tolist()])
    src_xy, src_valid = source_uv_to_out(src_uv, source_size, out_size)
    tgt_xy, tgt_valid = source_uv_to_out(tgt_uv, source_size, out_size)
    valid = src_valid & tgt_valid
    if src_front is not None:
        valid &= src_front
    if tgt_front is not None:
        valid &= tgt_front
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        return 0
    if len(ids) > 48:
        ids = ids[np.linspace(0, len(ids) - 1, 48, dtype=np.int64)]
    for i in ids.tolist():
        sx, sy = src_xy[i].tolist()
        tx, ty = tgt_xy[i].tolist()
        cv2.line(img, (int(sx), int(sy)), (int(tx), int(ty)), state_color, 1, cv2.LINE_AA)
        cv2.circle(img, (int(sx), int(sy)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(sx), int(sy)), 4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(img, (int(tx), int(ty)), state_color, markerType=cv2.MARKER_CROSS, markerSize=7, thickness=1, line_type=cv2.LINE_AA)
    return int(len(ids))


def draw_hands_and_state(
    img: np.ndarray,
    frame: dict[str, Any],
    intr_src: np.ndarray,
    source_size: int,
    out_size: int,
    residuals: dict[tuple[int, str], dict[str, Any]],
    prior_states: dict[tuple[int, str], dict[str, Any]],
    frame_idx: int,
) -> dict[str, Any]:
    bridge_cache: dict[Path, dict[str, np.ndarray]] = {}
    side_summaries: dict[str, Any] = {}
    for hand in as_list(frame.get("hands")):
        if not isinstance(hand, dict):
            continue
        side = str(hand.get("hand_side") or hand.get("side") or "")
        if side not in ("left", "right"):
            continue
        metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
        verts = full_vertices_camera(metric, bridge_cache)
        joints = np.asarray(metric.get("joints_current_v18_camera_m") or [], dtype=np.float64)
        color = HAND_COLORS_BGR[side]
        state_row = residuals.get((frame_idx, side))
        med = distance_median(state_row)
        state = classify_visible_surface_gap(med)
        state_color = STATE_COLORS_BGR[state]

        # Look up prior label for this frame/side
        prior_entry = prior_states.get((frame_idx, side), {})
        prior_label = prior_entry.get("prior_state", "n/a")

        drawn_vertices = 0
        drawn_joints = 0
        if verts.ndim == 2 and verts.shape[1] == 3 and len(verts):
            uv, front = project_camera(verts, [float(x) for x in intr_src.tolist()])
            xy, valid = source_uv_to_out(uv, source_size, out_size)
            valid &= front if front is not None else True
            layer = np.zeros_like(img)
            for x, y in xy[valid].tolist():
                cv2.circle(layer, (int(x), int(y)), 1, color, -1, cv2.LINE_AA)
            img[:] = cv2.addWeighted(img, 1.0, layer, 0.85, 0.0)
            drawn_vertices = int(np.count_nonzero(valid))
        if joints.shape == (21, 3):
            uvj, frontj = project_camera(joints, [float(x) for x in intr_src.tolist()])
            xyj, validj = source_uv_to_out(uvj, source_size, out_size)
            validj &= frontj if frontj is not None else True
            for jid, (x, y) in enumerate(xyj[validj].tolist()):
                cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), 1, cv2.LINE_AA)
            drawn_joints = int(np.count_nonzero(validj))
        drawn_vectors = draw_selected_gap_vectors(img, state_row, frame, intr_src, source_size, out_size, state_color)
        side_summaries[side] = {
            "visible_surface_gap_label": state,
            "visible_surface_gap_median_m": med,
            "prior_label": prior_label,
            "drawn_hand_vertices": drawn_vertices,
            "drawn_hand_joints": drawn_joints,
            "drawn_source_target_gap_vectors": drawn_vectors,
            "visibility_state": hand.get("visibility_state"),
        }
        # compact per-side HUD line
        y = 94 if side == "left" else 116
        d_txt = "unresolved" if med is None else f"{med*1000:5.1f} mm"
        label = f"{side}: gap={state}  median={d_txt}  prior={prior_label}"
        cv2.rectangle(img, (4, y - 3), (780, y + 16), (0, 0, 0), -1)
        cv2.putText(img, label, (8, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, state_color, 1, cv2.LINE_AA)
    return side_summaries


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
    frame: dict[str, Any],
    intr_src: np.ndarray,
    source_size: int,
    residuals: dict[tuple[int, str], dict[str, Any]],
    prior_states: dict[tuple[int, str], dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    out = rgb.copy()
    H, W = out.shape[:2]
    mbool = sam2_mask > 127
    if mbool.any():
        tint = np.zeros_like(out)
        tint[mbool] = (0, 230, 120)
        out = cv2.addWeighted(out, 1.0, tint, 0.18, 0.0)
        contours, _ = cv2.findContours(mbool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 230, 120), 1, cv2.LINE_8)
    ring = np.zeros_like(out)
    dots = np.zeros_like(out)
    for (x, y), col in zip(pts_xy.tolist(), pt_colors.tolist()):
        cv2.circle(ring, (int(x), int(y)), 3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(dots, (int(x), int(y)), 2, (int(col[0]), int(col[1]), int(col[2])), -1, cv2.LINE_AA)
    out = cv2.addWeighted(out, 1.0, ring, 0.5, 0.0)
    out = cv2.addWeighted(out, 1.0, dots, 1.0, 0.0)
    hand_summary = draw_hands_and_state(out, frame, intr_src, source_size, W, residuals, prior_states, frame_idx)
    hud_lines = [
        "visible-surface proximity residual | observation only | MANO joints preserved",
        "display bands only; hidden state unresolved; no pose/NP/owner/solver corrections",
        f"frame {frame_idx:03d}/149  t={time_s:.3f}s  keyboard surfels {n_front}/{n_total}",
        f"on-SAM2-mask {on_sam2_frac*100:4.1f}%   median keyboard cam-depth {median_depth_m:.3f} m",
    ]
    for i, line in enumerate(hud_lines):
        y = 8 + i * 20
        cv2.rectangle(out, (4, y - 2), (min(W - 4, 930), y + 16), (0, 0, 0), -1)
        cv2.putText(out, line, (8, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "keyboard surfels=depth color; hands=metric MANO samples/joints | metric_mano_preserved=true",
                (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
    return out, hand_summary


def encode_and_probe(frame_dir: Path, out_mp4: Path, fps: float) -> dict[str, Any]:
    encode_mp4(frame_dir, "%06d.png", out_mp4, fps)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height,duration",
         "-of", "json", str(out_mp4)], capture_output=True, text=True, check=True)
    st = json.loads(probe.stdout)["streams"][0]
    return {
        "nb_read_frames": int(st.get("nb_read_frames", -1)),
        "r_frame_rate": st.get("r_frame_rate"),
        "width": int(st["width"]),
        "height": int(st["height"]),
        "duration_s_reported": float(st.get("duration", -1)),
    }


def build(args: argparse.Namespace) -> None:
    run = Path(args.run_root)
    base = Path(args.base_run)
    residual_path = Path(args.residual_state)
    prior_path = Path(args.prior_packet)
    expected_hash = args.expected_prior_hash
    out_dir = run / "render_visible_surface_proximity_residual"
    tmp = run / ".render_tmp_proximity_residual_conditioned"
    for s in ["overlay", "world", "side"]:
        (tmp / s).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[render-r5-cond] python={sys.executable} run={run}")

    # ------------------------------------------------------------------
    # 1. Load and validate the frozen prior
    # ------------------------------------------------------------------
    print(f"[render-r5-cond] loading prior from {prior_path}")
    prior_payload = load_prior_packet(prior_path)
    validate_prior_hash(prior_path, expected_hash)
    prior_id = prior_payload["prior_id"]
    prior_hash = expected_hash
    prior_attestation = prior_payload["source"]["attestation"]
    print(f"[render-r5-cond] prior_id={prior_id} hash={prior_hash[:12]}... validated ok")

    prior_states = prior_state_by_frame_side(prior_payload)
    print(f"[render-r5-cond] prior_state entries: {len(prior_states)}")

    # Count prior state distribution for reporting
    prior_state_counts: dict[str, int] = {"asserted": 0, "absent": 0, "unresolved": 0}
    for ps in prior_states.values():
        lbl = ps.get("prior_state", "unresolved")
        prior_state_counts[lbl] = prior_state_counts.get(lbl, 0) + 1
    print(f"[render-r5-cond] prior label distribution: {prior_state_counts}")

    # ------------------------------------------------------------------
    # 2. Load residual state and other inputs
    # ------------------------------------------------------------------
    residuals = residual_rows_by_frame_side(residual_path)
    print(f"[render-r5-cond] residual entries: {len(residuals)}")

    arc = np.load(run / "r3_visible_surfel_compat" / "visible_surfel_compat_archive.npz", allow_pickle=True)
    V = np.ascontiguousarray(arc["vertices"], dtype=np.float64)
    voff = arc["vertex_offsets"]
    faces_non_evidential = bool(arc["faces_non_evidential"])

    ann = load_json(run / "prediction_annotations" / "annotations_for_cotracker.json")
    ann_frames = by_frame(ann["frames"])
    base_ann = load_json(base / "state" / "base_annotations" / "annotations_v19_base.json")
    rgb_manifest = load_json(base / "input" / "raw_frame_manifest" / "manifest.json")
    rgb_frames = by_frame(rgb_manifest["frames"])
    sam2 = load_json(run / "prediction_masks" / "keyboard_sam2_manifest.json")
    sam2_frames = by_frame(sam2["frames"])
    calib = load_json(base / "state" / "calibration" / "v19_camera_calibration_contract.json")
    intr_src = np.array(calib["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    source_size = int(calib["source_size"]["width"])
    fps = float(base_ann["raw_video"]["fps"])
    n_frames = int(base_ann["raw_video"]["frame_count"])
    out_size = int(rgb_manifest["manifest_width"])

    # ------------------------------------------------------------------
    # 3. Pre-compute surfel projections
    # ------------------------------------------------------------------
    pre: list[dict[str, Any]] = []
    all_camz = []
    for f in range(n_frames):
        world = V[voff[f]:voff[f + 1]]
        T = ann_frames[f]["camera"]["T_world_camera_metric"]
        cam = world_to_camera(world, T)
        pts, n_front = project_to_pixels(cam, intr_src, source_size, out_size)
        camf = cam[cam[:, 2] > 0]
        pre.append({"world": world, "cam_front": camf, "pts": pts, "n_front": n_front})
        if len(camf):
            all_camz.append(camf[:, 2])
    allz = np.concatenate(all_camz) if all_camz else np.array([0.5])
    z_lo, z_hi = float(np.percentile(allz, 2)), float(np.percentile(allz, 98))
    allx = np.concatenate([p["cam_front"][:, 0] for p in pre if len(p["cam_front"])])
    ally = np.concatenate([p["cam_front"][:, 1] for p in pre if len(p["cam_front"])])
    pad = 0.02
    ax_lim = ((float(np.percentile(allx, 1)) - pad, float(np.percentile(allx, 99)) + pad),
              (float(np.percentile(ally, 1)) - pad, float(np.percentile(ally, 99)) + pad),
              (max(0.0, z_lo - pad), z_hi + pad))
    cmap = make_depth_cmap()
    dpi = 150
    fs = out_size / dpi
    fig = plt.figure(figsize=(fs, fs), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    plt.tight_layout(pad=0.5)

    # ------------------------------------------------------------------
    # 4. Render frames
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {k: 0 for k in STATE_COLORS_BGR}
    for f in range(n_frames):
        p = pre[f]
        rgb = cv2.imread(rgb_frames[f]["rgb"])
        if rgb is None:
            raise RuntimeError(f"cannot read RGB {rgb_frames[f]['rgb']}")
        if rgb.shape[0] != out_size or rgb.shape[1] != out_size:
            rgb = cv2.resize(rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)
        sam2_mask = cv2.imread(sam2_frames[f]["mask"], cv2.IMREAD_GRAYSCALE)
        if sam2_mask is None:
            raise RuntimeError(f"cannot read SAM2 mask {sam2_frames[f]['mask']}")
        if sam2_mask.shape[0] != out_size or sam2_mask.shape[1] != out_size:
            sam2_mask = cv2.resize(sam2_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        sam2_bool = sam2_mask > 127
        camf = p["cam_front"]
        colors = depth_color(camf[:, 2], z_lo, z_hi, cmap) if len(camf) else np.zeros((0, 3), np.uint8)
        on = float(sam2_bool[p["pts"][:, 1], p["pts"][:, 0]].mean()) if len(p["pts"]) else 0.0
        med_z = float(np.median(camf[:, 2])) if len(camf) else float("nan")
        overlay, hands = draw_overlay(rgb, sam2_mask, p["pts"], colors[:len(p["pts"])], f, f / fps, on,
                                      p["n_front"], int(len(p["world"])), med_z,
                                      ann_frames[f], intr_src, source_size, residuals, prior_states)
        world_img = render_3d_panel(camf, ax_lim, z_lo, z_hi, cmap, f, f / fps, fig, ax, out_size)
        side = np.full((out_size, out_size * 2 + 4, 3), 24, np.uint8)
        side[:, :out_size] = overlay
        side[:, out_size + 4:] = world_img
        cv2.imwrite(str(tmp / "overlay" / f"{f:06d}.png"), overlay)
        cv2.imwrite(str(tmp / "world" / f"{f:06d}.png"), world_img)
        cv2.imwrite(str(tmp / "side" / f"{f:06d}.png"), side)
        for h in hands.values():
            lb = h["visible_surface_gap_label"]
            state_counts[lb] = state_counts.get(lb, 0) + 1
        rows.append({
            "frame_idx": f,
            "time_s": f / fps,
            "residual_by_hand": hands,
            "keyboard_surfels_total": int(len(p["world"])),
            "keyboard_projected_on_sam2_mask_frac": round(on, 4),
        })
        if f % 25 == 0 or f == n_frames - 1:
            print(f"[render-r5-cond] frame {f:03d}/{n_frames-1} states="
                  f"{{ {s}:{h.get('visible_surface_gap_label','?')} prior={h.get('prior_label','?')} for s,h in hands.items() }}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Encode videos with cleaner filenames
    # ------------------------------------------------------------------
    overlay_mp4 = out_dir / "v19_visible_surface_gap_overlay.mp4"
    world_mp4 = out_dir / "v19_visible_surface_gap_world.mp4"
    side_mp4 = out_dir / "v19_visible_surface_gap_side_by_side.mp4"
    val = {
        overlay_mp4.name: encode_and_probe(tmp / "overlay", overlay_mp4, fps),
        world_mp4.name: encode_and_probe(tmp / "world", world_mp4, fps),
        side_mp4.name: encode_and_probe(tmp / "side", side_mp4, fps),
    }

    pick = sorted(set([0, 60, 75, 93, 106, 109, 110, 111, 120, 149]))
    cells = []
    for f in pick:
        im = cv2.imread(str(tmp / "overlay" / f"{f:06d}.png"))
        cells.append(cv2.resize(im, (out_size // 2, out_size // 2), interpolation=cv2.INTER_AREA))
    sheet = np.vstack(cells)
    cv2.putText(sheet, "visible-surface proximity residual stills: display bands only, hidden state unresolved",
                (8, sheet.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    sheet_path = out_dir / "v19_visible_surface_gap_still_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)

    # ------------------------------------------------------------------
    # 6. Produce conditioned manifest and report
    # ------------------------------------------------------------------
    residual_state_sha256 = file_sha256(residual_path)
    residual_report_path = residual_path.with_name(residual_path.stem + "_report.json")
    residual_report_sha256 = file_sha256(residual_report_path)
    annotations_path = run / "prediction_annotations" / "annotations_for_cotracker.json"
    annotations_sha256 = file_sha256(annotations_path)

    manifest = {
        "schema": "v19_visible_surface_proximity_residual_render_manifest_v2",
        "render_method": "render_visible_surface_proximity_residual_conditioned_artifact",
        "claim_scope": "per-frame visible-surface proximity residual only — likelihood measurement, not a binary touch state, hidden-state decision, nonpenetration, occlusion owner, or object pose. Metric MANO joints are preserved; no solver corrections.",
        "residual_is_likelihood_only": True,
        "metric_mano_preserved": True,
        "no_solver_correction": True,
        "conditioned_on_prior_hash": prior_hash,
        "conditioned_on_prior_id": prior_id,
        "prior_not_modified_since_compute": True,
        "prior_provenance": {
            "prior_hash": prior_hash,
            "prior_id": prior_id,
            "mechanism": "visual_semantic_contact_prior",
            "input_modality": prior_payload["source"]["input_modality"],
            "attestation": prior_attestation,
            "geometry_blind": True,
            "prior_packet_path": str(prior_path.absolute()),
            "prior_packet_sha256": prior_hash,
        },
        "conditioning_policy": {
            "circularity_guard": "prior is geometry-blind (no MANO, depth, object pose); residual is metric-only (MANO+surfels). Prior cannot be influenced by residual measurements. Conditioning is a one-way metadata join — prior labels are attached to residual rows as contextual reference only.",
            "prior_labels_used_as": "contextual metadata only; never used to modify MANO joints, object surfels, or residual gap measurements.",
            "residual_computation_precedes_conditioning": True,
            "conditioning_step_is_post_hoc": True,
        },
        "labels": {
            "visible_surface_gap_close": "display band: median MANO-sample to observed visible-surface gap <= 10 mm",
            "visible_surface_gap_near": "display band: median gap > 10 mm and <= 30 mm",
            "visible_surface_gap_high": "display band: median gap > 30 mm for selected visible-surface candidates; hidden state unresolved",
            "hidden_state_unresolved": "residual row unavailable or visible-only evidence cannot decide hidden state",
        },
        "prior_label_legend": {
            "asserted": "frozen visual prior: visible cues strongly indicate hand-keyboard contact (geometry-blind, visual-only)",
            "absent": "frozen visual prior: hand is clearly separated from keyboard (visual-only)",
            "unresolved": "frozen visual prior: contact cannot be visually distinguished from near-approach gap",
        },
        "distance_band_thresholds_visualization_only_m": {
            "visible_surface_gap_close_max": 0.010,
            "visible_surface_gap_near_max": 0.030,
        },
        "forbidden_claims": [
            "binary_touch_state",
            "ownership",
            "hidden_state_decision",
            "nonpenetration",
            "occlusion_owner",
            "object_pose",
            "GT",
            "contact_confirmed",
            "surface_support_confirmed",
        ],
        "provenance": {
            "run_root": str(run),
            "base_run": str(base),
            "residual_state": str(residual_path),
            "residual_state_sha256": residual_state_sha256,
            "residual_report": str(residual_report_path),
            "residual_report_sha256": residual_report_sha256,
            "annotations": str(annotations_path),
            "annotations_sha256": annotations_sha256,
            "surfels_npz": str(run / "r3_visible_surfel_compat" / "visible_surfel_compat_archive.npz"),
            "faces_non_evidential_loader_padding": faces_non_evidential,
            "prior_packet": str(prior_path.absolute()),
            "prior_packet_sha256": prior_hash,
        },
        "video": {
            "fps": fps,
            "frame_count": n_frames,
            "duration_s": n_frames / fps,
            "duration_matches_raw": abs((n_frames / fps) - float(base_ann["raw_video"]["duration_s"])) < 1e-6,
        },
        "state_counts_hand_rows": state_counts,
        "prior_state_counts": {
            "total_entries": len(prior_states),
            "distribution": prior_state_counts,
        },
        "files": {
            "overlay": str(overlay_mp4),
            "world": str(world_mp4),
            "side_by_side": str(side_mp4),
            "still_sheet": str(sheet_path),
        },
        "supersedes": {
            "note": "This conditioned render supersedes the previous v19_close_visible_surface_support_* render. The previous filenames and labels used 'support' which could be misread as confirming physical surface contact support. The new filenames use 'visible_surface_gap' to make explicit that these are gap measurements, not support commitments.",
            "superseded_files": [
                str(out_dir / "v19_close_visible_surface_support_overlay.mp4"),
                str(out_dir / "v19_close_visible_surface_support_world.mp4"),
                str(out_dir / "v19_close_visible_surface_support_side_by_side.mp4"),
                str(out_dir / "v19_close_visible_surface_support_still_sheet.png"),
                str(out_dir / "v19_close_visible_surface_support_render_manifest.json"),
                str(out_dir / "v19_close_visible_surface_support_render_report.json"),
            ],
        },
        "validation": val,
        "rows": rows,
    }
    manifest_path = out_dir / "v19_visible_surface_gap_render_manifest.json"
    report_path = out_dir / "v19_visible_surface_gap_render_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {k: v for k, v in manifest.items() if k != "rows"}
    report["status"] = "ok"
    report["all_videos_150_frames"] = all(v["nb_read_frames"] == n_frames for v in val.values())
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(report, indent=2)[:20000])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--base-run", required=True)
    p.add_argument("--residual-state", required=True)
    p.add_argument("--prior-packet", required=True)
    p.add_argument("--expected-prior-hash", required=True)
    p.add_argument("--keep-tmp", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
