#!/usr/bin/env python3
"""Frame-level ego.hoi contact review driven by clip001850 kill-test rows.

This is the smallest viewer-consumable artifact for the current mechanism state:
frames 32 and 36 are rendered from the source RGB plus actual projected MANO
sample vertices, but the visible contact labels come from
/tmp/clip001850_masked_contact_killtests/contact_killtests_frame_detail.ndjson.

The point is not presentation polish. The artifact must change the visible
annotation state from the published v19 banner (gap/penverts/UNCERTAIN) to the
post-kill-test state:
  f32 -> geometry_epoch_contaminated
  f36 -> full_frame_depth_leak
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUN_ROOT = Path(
    "/data2/ego_annotation_outputs/v19_runs/"
    "20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)
INTV_STATE = RUN_ROOT / (
    "measurements/mano_interval_correction/keyboard_0_149/"
    "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/"
    "v18_joint_mano_interval_trajectory_state.json"
)
VIS_GEO = RUN_ROOT / (
    "measurements/object_geometry/visible_geometry/keyboard/"
    "annotations_v19_visible_geometry.json"
)
KILL_DIR = Path("/tmp/clip001850_masked_contact_killtests")
RGB_DIR = RUN_ROOT / "input/raw_frame_manifest/rgb"

SOURCE_W, SOURCE_H = 1408, 1408
MANIFEST_W, MANIFEST_H = 960, 960
SCALE = MANIFEST_W / SOURCE_W

PUBLISHED_BANNER = (
    "PUBLISHED v19 banner (current video state):\n"
    "R: gap 39.2mm, shift 0.0px, closed False | penverts=0 | UNCERTAIN\n"
    "annotation_ready=True despite graph nfev=1/cost=0/NP targets=0"
)

ROUTE_LABELS = {
    "geometry_epoch_contaminated": "UNRESOLVED — object body contaminated, not contact-eligible",
    "full_frame_depth_leak": "UNRESOLVED — full-frame depth leak; keyboard mask refutes penetration",
    "pose_unresolved": "UNRESOLVED — object pose/mask proxy gap",
    "contact_candidate_keyboard_masked": "CONTACT CANDIDATE — one masked keyboard-depth vertex within band",
    "evidence_missing": "UNRESOLVED — evidence missing",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _mm(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v) * 1000:.1f}mm"
    except (TypeError, ValueError):
        return "n/a"


def _stat(row: dict[str, Any], path: list[str], key: str = "median") -> Any:
    cur: Any = row
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    if isinstance(cur, dict):
        return cur.get(key)
    return cur


def load_interval() -> dict[str, Any]:
    return json.loads(INTV_STATE.read_text())


def load_visgeo() -> dict[str, Any]:
    return json.loads(VIS_GEO.read_text())


def load_kill_rows() -> dict[int, dict[str, Any]]:
    nd = KILL_DIR / "contact_killtests_frame_detail.ndjson"
    if not nd.exists():
        raise FileNotFoundError(f"{nd} missing; run run_clip001850_masked_contact_killtests.py")
    rows: dict[int, dict[str, Any]] = {}
    for line in nd.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[int(r["frame_idx"])] = r
    return rows


def load_kill_summary() -> dict[str, Any]:
    return json.loads((KILL_DIR / "summary.json").read_text())


def project_vertices(verts_world: np.ndarray, T_world_camera: np.ndarray, intr: list[float]) -> np.ndarray:
    Tcw = np.linalg.inv(T_world_camera)
    Ph = np.concatenate([verts_world, np.ones((len(verts_world), 1))], axis=1).T
    Pc = Tcw @ Ph
    z = Pc[2]
    fx, fy, cx, cy = intr
    x = fx * Pc[0] / z + cx
    y = fy * Pc[1] / z + cy
    return np.stack([x, y, z], axis=1)


def interval_frame(intv: dict[str, Any], frame: int) -> dict[str, Any]:
    return next(x for x in intv["per_frame_states"] if x["frame_idx"] == frame and x["hand_side"] == "right")


def build_panel_a(rgb_path: Path) -> Image.Image:
    img = Image.open(rgb_path).convert("RGB").resize((MANIFEST_W, MANIFEST_H))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, MANIFEST_W, 124], fill=(0, 0, 0, 180))
    d.text((12, 8), "PANEL A — PUBLISHED v19 STATE", font=_font(21), fill=(255, 120, 120))
    for i, ln in enumerate(PUBLISHED_BANNER.splitlines()):
        d.text((12, 38 + i * 23), ln, font=_font(18), fill=(255, 235, 235))
    return img


def build_panel_b(rgb_path: Path, intv: dict[str, Any], visgeo: dict[str, Any], row: dict[str, Any], frame: int) -> tuple[Image.Image, dict[str, Any]]:
    entry = interval_frame(intv, frame)
    cam = visgeo["frames"][frame]["camera"]
    Twc = np.array(cam["T_world_camera"])
    intr = cam["intrinsics_fx_fy_cx_cy"]
    verts = np.array(entry["optimized_vertices_world_sample_m"])
    sids = np.array(entry["optimized_vertices_sample_ids"])
    proj = project_vertices(verts, Twc, intr)
    px = proj[:, 0] * SCALE
    py = proj[:, 1] * SCALE
    depth = proj[:, 2]
    in_frame = (px >= 0) & (px < MANIFEST_W) & (py >= 0) & (py < MANIFEST_H) & (depth > 0)

    full_ids = np.array(row["KT1_full_frame_depth"].get("penetrating_sample_ids") or [])
    hq_ids = np.array(row["KT1_keyboard_masked_hand_quarantined_depth"].get("penetrating_sample_ids") or [])
    near_ids = np.array(row["KT1_keyboard_masked_hand_quarantined_depth"].get("near_band_sample_ids") or [])
    full_mask = np.isin(sids, full_ids) & in_frame
    hq_mask = np.isin(sids, hq_ids) & in_frame
    near_mask = np.isin(sids, near_ids) & in_frame

    img = Image.open(rgb_path).convert("RGB").resize((MANIFEST_W, MANIFEST_H))
    d = ImageDraw.Draw(img, "RGBA")

    for i in np.where(in_frame)[0]:
        d.ellipse([px[i] - 2, py[i] - 2, px[i] + 2, py[i] + 2], fill=(0, 210, 220, 175))
    for i in np.where(full_mask)[0]:
        d.ellipse([px[i] - 5, py[i] - 5, px[i] + 5, py[i] + 5], fill=(255, 150, 20, 220))
    for i in np.where(near_mask & ~hq_mask)[0]:
        d.ellipse([px[i] - 6, py[i] - 6, px[i] + 6, py[i] + 6], outline=(80, 255, 80, 240), width=2)
    for i in np.where(hq_mask)[0]:
        d.ellipse([px[i] - 7, py[i] - 7, px[i] + 7, py[i] + 7], outline=(255, 255, 255, 255), width=2, fill=(255, 40, 40, 240))

    d.rectangle([0, MANIFEST_H - 116, MANIFEST_W, MANIFEST_H], fill=(0, 0, 0, 165))
    legend = [
        "PANEL B — kill-test geometry overlay",
        "cyan: MANO sample vertices projected from interval state",
        "orange: full-frame-depth penetrating ids (leak-prone)",
        "red: keyboard-masked+hand-quarantined penetrating ids (eligible contact)",
        f"eligible red count={int(hq_mask.sum())}; route={row['route']}",
    ]
    for j, ln in enumerate(legend):
        d.text((10, MANIFEST_H - 110 + j * 21), ln, font=_font(17), fill=(235, 255, 235))

    return img, {
        "n_sample_verts_projected": int(in_frame.sum()),
        "full_frame_depth_penetrating_projected": int(full_mask.sum()),
        "keyboard_hq_penetrating_projected": int(hq_mask.sum()),
        "keyboard_hq_near_projected": int(near_mask.sum()),
        "depth_range_m": [float(depth[in_frame].min()), float(depth[in_frame].max())] if in_frame.any() else None,
    }


def build_panel_c(row: dict[str, Any], summary: dict[str, Any], proj_info: dict[str, Any]) -> Image.Image:
    W, H = 760, 960
    img = Image.new("RGB", (W, H), (17, 17, 22))
    d = ImageDraw.Draw(img)
    mono = _font(18)
    mono_b = _font(19)
    title = _font(22)

    route = row["route"]
    geom = summary.get("geometry_epoch", {})
    prov = geom.get("face_provenance", {})
    kt2 = row["KT2_vertex_coherence"]
    hq = row["KT1_keyboard_masked_hand_quarantined_depth"]
    full = row["KT1_full_frame_depth"]
    kb = row["KT1_keyboard_masked_depth"]
    table = row["KT1_table_background_plane"]
    interval = row["interval_solver_published"]

    y = 14
    d.text((14, y), "PANEL C — ego.hoi CONTACT STATE FROM KILL TESTS", font=title, fill=(255, 235, 120))
    y += 36

    sections = [
        ("visible contact_state", [
            f"contact_state = {route}",
            ROUTE_LABELS.get(route, "UNRESOLVED — route not mapped"),
            f"mask source = {row['keyboard_mask_source']}  mask_file_exists={row['mask_files_exist']}",
            f"pose = {row['pose_source']}  direct_visible={row['pose_direct_visible']} gap={row['pose_gap_frames']}",
        ]),
        ("KT-1: keyboard-masked + hand-quarantined depth", [
            f"eligible vertices over keyboard mask = {hq.get('n_vertices_in_region')}",
            f"median hand-depth delta = {_mm(_stat(row, ['KT1_keyboard_masked_hand_quarantined_depth','delta_summary_m']))}",
            f"max hand-depth delta = {_mm(_stat(row, ['KT1_keyboard_masked_hand_quarantined_depth','delta_summary_m'], 'max'))}",
            f"penetrating count = {hq.get('penetrating_vertex_count')}   near-band count = {hq.get('near_band_vertex_count')}",
            "negative delta = hand in front of keyboard depth surface",
        ]),
        ("full-frame depth leak check", [
            f"full-frame penetrating count = {full.get('penetrating_vertex_count')}",
            f"full-frame max delta = {_mm(_stat(row, ['KT1_full_frame_depth','delta_summary_m'], 'max'))}",
            f"keyboard-masked penetrating count = {kb.get('penetrating_vertex_count')}",
            f"table/background plane max signed = {_mm(_stat(row, ['KT1_table_background_plane','signed_dist_summary_m'], 'max'))}",
        ]),
        ("interval solver published mesh-penetration channel", [
            f"published interval penetration max = {_mm(_stat(interval, ['interval_full_observed_surface_penetration_m'], 'max'))}",
            f"published interval penetration median = {_mm(_stat(interval, ['interval_full_observed_surface_penetration_m']))}",
            f"published penetrating count = {interval.get('interval_full_observed_surface_penetrating_vertex_count')}",
            "kill-test result: not transferable as keyboard contact evidence",
        ]),
        ("KT-2: vertex-set coherence", [
            f"route = {kt2.get('route')}",
            f"eligible penetrating ids = {kt2.get('penetrating_sample_ids')}",
            f"near-band ids = {kt2.get('near_band_sample_ids')}",
            f"forbidden max-scalar coherence = {kt2.get('forbidden_max_scalar_as_coherence')}",
            f"window mean IoU = {summary.get('KT2_temporal_coherence',{}).get('mean_iou')}",
        ]),
        ("geometry epoch", [
            f"completed mesh watertight = {geom.get('completed_mesh_watertight')}",
            f"AABB extents m = {[round(x,3) for x in geom.get('completed_mesh_aabb_extents_m', [])]}",
            f"extent ratio to keyboard prior = {[round(x,2) for x in geom.get('completed_mesh_extent_ratio_to_keyboard_prior', [])]}",
            f"TRELLIS face fraction = {prov.get('trellis_fraction')}  observed = {prov.get('observed_fraction')}",
            "contact/NP zero penetration = disabled geometry query, not no-contact",
        ]),
        ("overlay stats", [
            f"projected MANO sample verts = {proj_info['n_sample_verts_projected']}",
            f"orange full-frame leak verts = {proj_info['full_frame_depth_penetrating_projected']}",
            f"red keyboard-HQ contact verts = {proj_info['keyboard_hq_penetrating_projected']}",
        ]),
    ]

    for name, lines in sections:
        d.text((14, y), name, font=mono_b, fill=(120, 220, 255))
        y += 25
        for ln in lines:
            d.text((24, y), str(ln)[:84], font=mono, fill=(226, 226, 232))
            y += 22
        y += 7
    return img


def compose_review(frame: int, rows: dict[int, dict[str, Any]], summary: dict[str, Any], intv: dict[str, Any], visgeo: dict[str, Any], out: Path) -> dict[str, Any]:
    rgb = RGB_DIR / f"{frame:06d}.jpg"
    if not rgb.exists():
        raise FileNotFoundError(rgb)
    row = rows[frame]
    panel_a = build_panel_a(rgb)
    panel_b, proj_info = build_panel_b(rgb, intv, visgeo, row, frame)
    panel_c = build_panel_c(row, summary, proj_info)

    gap = 12
    label_h = 26
    total_w = panel_a.width + gap + panel_b.width + gap + panel_c.width
    total_h = MANIFEST_H + label_h
    canvas = Image.new("RGB", (total_w, total_h), (8, 8, 10))
    d = ImageDraw.Draw(canvas)
    x = 0
    for label, panel in (
        ("A: published v19", panel_a),
        ("B: projected kill-test geometry", panel_b),
        ("C: ego.hoi contact_state", panel_c),
    ):
        d.rectangle([x, 0, x + panel.width, label_h], fill=(40, 40, 48))
        d.text((x + 8, 3), label, font=_font(20), fill=(235, 235, 150))
        canvas.paste(panel, (x, label_h))
        x += panel.width + gap
    canvas.save(out, quality=92)

    hq = row["KT1_keyboard_masked_hand_quarantined_depth"]
    return {
        "frame": frame,
        "out": str(out),
        "contact_state": row["route"],
        "panel_b_geometry": proj_info,
        "keyboard_hq_delta_median_m": _stat(row, ["KT1_keyboard_masked_hand_quarantined_depth", "delta_summary_m"]),
        "keyboard_hq_penetrating_count": hq.get("penetrating_vertex_count"),
        "interval_published_penetration_max_m": _stat(row["interval_solver_published"], ["interval_full_observed_surface_penetration_m"], "max"),
        "pose_source": row["pose_source"],
        "mask_source": row["keyboard_mask_source"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[32, 36])
    ap.add_argument("--out-dir", default="/tmp/clip001850_egohoi_contact_review")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intv = load_interval()
    visgeo = load_visgeo()
    rows = load_kill_rows()
    summary = load_kill_summary()

    manifest = {
        "schema": "ego.hoi_contact_review/0.1.0",
        "source_rows": str(KILL_DIR / "contact_killtests_frame_detail.ndjson"),
        "source_summary": str(KILL_DIR / "summary.json"),
        "meaning": (
            "Panel C is driven by KT-1/2/3/5 contact_state rows. The published video says "
            "gap/penverts/UNCERTAIN; this review says the specific mechanism: geometry_epoch_contaminated "
            "or full_frame_depth_leak."
        ),
        "route_counts": summary.get("route_counts"),
        "frames": [],
    }
    for f in args.frames:
        if f not in rows:
            raise SystemExit(f"frame {f} not found in kill-test rows")
        manifest["frames"].append(compose_review(f, rows, summary, intv, visgeo, out_dir / f"egohoi_review_f{f:03d}.jpg"))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir}")
    for frame in manifest["frames"]:
        print(frame["frame"], frame["contact_state"], frame["out"])


if __name__ == "__main__":
    main()
