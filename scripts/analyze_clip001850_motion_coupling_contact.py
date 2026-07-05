#!/usr/bin/env python3
"""
Motion-coupling contact-promotion adjudication for HOT3D clip001850.

This is the NEXT admissible contact-promotion channel after signed-distance /
geometry-epoch / hand-depth channels were all refuted (see subagents 07,10-17).
It is floor-independent: it does NOT use any distance threshold to create contact.
It measures whether the keyboard's RIGID MOTION (translation onset / rotation
impulse) is time-locked to right-hand motion or fingertip approach in frames
28-48, which is the only kind of evidence that could legitimately promote a
frame beyond `unresolved` GT-free (KT-U in subagent 14).

Inputs (existing CPU files only, no heavy inference / GPU):
  RUN/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json
  RUN/measurements/mano_interval_correction/.../v18_joint_mano_interval_trajectory_state.json
  DURABLE/contact_state_table/contact_frame_detail.ndjson
  RUN/input/raw_frame_manifest/rgb/*.jpg   (review image only)

Outputs (additive; no staging / commit):
  /tmp/clip001850_motion_coupling_contact/{summary.json, per_frame.ndjson, review.png}
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUN = Path(
    "/data2/ego_annotation_outputs/v19_runs/"
    "20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)
DURABLE = Path(
    "/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706"
)
OUT = Path("/tmp/clip001850_motion_coupling_contact")
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = (28, 48)  # inclusive
FPS = 30.0

POSE_REPORT = RUN / "measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json"
MANO_STATE = (
    RUN
    / "measurements/mano_interval_correction/keyboard_0_149/"
    "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/"
    "v18_joint_mano_interval_trajectory_state.json"
)
CFD = DURABLE / "contact_state_table/contact_frame_detail.ndjson"
RGB = RUN / "input/raw_frame_manifest/rgb"

# MANO joint indices for fingertips (right hand, same as left topology)
TIP_IDX = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def rotmat_to_angle_delta(R0, R1):
    """Relative rotation angle (rad) between two rotation matrices."""
    dR = R1 @ R0.T
    cos = (np.trace(dR) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return math.acos(cos)


def load():
    pr = json.loads(POSE_REPORT.read_text())
    ms = json.loads(MANO_STATE.read_text())
    pose_rows = pr["pose_rows"]
    graph_frames = set(pr["graph_frames"])
    # map frame->pose row
    prow = {r["frame_idx"]: r for r in pose_rows}
    n = max(prow) + 1
    # per_frame mano
    mfp = {(r["frame_idx"], r["hand_side"]): r for r in ms["per_frame_states"]}
    # contact_frame_detail
    cfd = {}
    if CFD.exists():
        for line in CFD.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cfd[r["frame_idx"]] = r
    return pr, ms, prow, graph_frames, n, mfp, cfd


def main():
    pr, ms, prow, graph_frames, n, mfp, cfd = load()

    # --- Object (keyboard) per-frame trajectory ---
    obj_t = np.full((n, 3), np.nan)
    obj_direct = np.zeros(n, dtype=bool)
    obj_status = [""] * n
    obj_pose_src = [""] * n
    for f in range(n):
        r = prow[f]
        obj_t[f] = np.array(r["translation_world_m"], float)
        tg = r.get("temporal_pose_graph", {})
        obj_direct[f] = bool(tg.get("direct_visible_measurement", False)) or (f in graph_frames)
        obj_pose_src[f] = tg.get("pose_source", "")
        obj_status[f] = r.get("pose_measurement_status", "")
    obj_R = np.full((n, 3, 3), np.nan)
    for f in range(n):
        obj_R[f] = np.array(prow[f]["rotation_world_from_completed_canonical_matrix"], float)

    # per-frame deltas (frame f vs f-1)
    obj_dtrans = np.zeros(n)
    obj_drot = np.zeros(n)
    # whether the transition f-1 -> f is between two DIRECT measurements
    obj_trans_direct_edge = np.zeros(n, dtype=bool)
    for f in range(1, n):
        if not np.isnan(obj_t[f]).any() and not np.isnan(obj_t[f - 1]).any():
            obj_dtrans[f] = float(np.linalg.norm(obj_t[f] - obj_t[f - 1]))
            obj_drot[f] = rotmat_to_angle_delta(obj_R[f - 1], obj_R[f])
            obj_trans_direct_edge[f] = bool(obj_direct[f] and obj_direct[f - 1])

    # --- Right hand trajectory ---
    rh_root = np.full((n, 3), np.nan)
    rh_tips = {k: np.full((n, 3), np.nan) for k in TIP_IDX}
    rh_present = np.zeros(n, dtype=bool)
    for f in range(n):
        r = mfp.get((f, "right"))
        if r is None:
            continue
        rh_present[f] = True
        rh_root[f] = np.array(r["optimized_translation_world_m"], float)
        j = np.array(r["optimized_joints_world_m"], float)
        for name, idx in TIP_IDX.items():
            rh_tips[name][f] = j[idx]

    rh_droot = np.zeros(n)
    for f in range(1, n):
        if rh_present[f] and rh_present[f - 1]:
            rh_droot[f] = float(np.linalg.norm(rh_root[f] - rh_root[f - 1]))

    # fingertip approach to keyboard CENTROID (not surface: centroid is
    # observable, surface is the contaminated epoch). Approach SPEED defined
    # as the per-frame decrease of min tip->centroid distance.
    tip_to_c = np.full((n, 5), np.nan)
    for f in range(n):
        if rh_present[f] and not np.isnan(obj_t[f]).any():
            d = [float(np.linalg.norm(rh_tips[k][f] - obj_t[f])) for k in TIP_IDX]
            tip_to_c[f] = d
    min_tip_c = np.nanmin(tip_to_c, axis=1)  # closest fingertip to centroid
    approach = np.zeros(n)  # positive = fingertip approaching centroid
    for f in range(1, n):
        if not np.isnan(min_tip_c[f]) and not np.isnan(min_tip_c[f - 1]):
            approach[f] = min_tip_c[f - 1] - min_tip_c[f]

    # --- Motion-coupling statistics ---
    # object motion floor: robust scale of object per-frame translation delta
    # over DIRECT->DIRECT transitions across the whole timeline (these are the
    # only independent object-motion measurements; held/interp transitions are
    # artificially zero by construction and would deflate the floor).
    direct_edges = np.where(obj_trans_direct_edge)[0]
    direct_edges = direct_edges[direct_edges > 0]
    obj_dtrans_direct = obj_dtrans[direct_edges] if len(direct_edges) else np.array([])
    obj_drot_direct = obj_drot[direct_edges] if len(direct_edges) else np.array([])

    def robust(arr):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return None
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826 or 1e-9
        return dict(median=med, mad=mad, p90=float(np.percentile(arr, 90)),
                    max=float(np.max(arr)), n=int(arr.size))

    obj_trans_floor = robust(obj_dtrans_direct)
    obj_rot_floor = robust(obj_drot_direct)

    # window slice
    w = np.arange(WINDOW[0], WINDOW[1] + 1)

    # Object total motion amplitude across window (first direct -> last direct in window)
    wdirect = [f for f in w if obj_direct[f]]
    if len(wdirect) >= 2:
        a, b = wdirect[0], wdirect[-1]
        obj_window_amp_mm = float(np.linalg.norm(obj_t[b] - obj_t[a]) * 1000.0)
        obj_window_rot_deg = math.degrees(rotmat_to_angle_delta(obj_R[a], obj_R[b]))
    else:
        obj_window_amp_mm = None
        obj_window_rot_deg = None
    # right hand total motion amplitude across window (root)
    rh_window_amp_mm = (
        float(np.linalg.norm(rh_root[w[-1]] - rh_root[w[0]]) * 1000.0)
        if rh_present[w[0]] and rh_present[w[-1]] else None
    )
    # fingertip approach total (min tip->centroid change window start->end)
    tip_c_drop_mm = None
    if not np.isnan(min_tip_c[w[0]]) and not np.isnan(min_tip_c[w[-1]]):
        tip_c_drop_mm = float((min_tip_c[w[0]] - min_tip_c[w[-1]]) * 1000.0)

    # Per-frame coupling: peak object-translation impulse (direct edge) vs
    # peak fingertip approach speed and vs min tip->centroid frame.
    w_direct_edges = [f for f in w if obj_trans_direct_edge[f]]
    obj_trans_in_window = [(f, obj_dtrans[f] * 1000.0) for f in w_direct_edges]
    obj_rot_in_window = [(f, math.degrees(obj_drot[f])) for f in w_direct_edges]
    approach_mm = [(f, approach[f] * 1000.0) for f in w if approach[f] > 0]

    # candidate coupling frames: a DIRECT object-motion edge where dtrans exceeds
    # 3x its robust direct-edge floor AND coincides (same frame or +/-1) with a
    # fingertip-approach peak.
    coupling_floor_mm = (obj_trans_floor["median"] + 3 * obj_trans_floor["mad"]) * 1000.0 if obj_trans_floor else float("inf")
    coupling_floor_rot_deg = (obj_rot_floor["median"] + 3 * obj_rot_floor["mad"]) * 180 / math.pi if obj_rot_floor else float("inf")

    # the dominant object-motion edge in the window
    dom_obj_edge = max(obj_trans_in_window, key=lambda kv: kv[1]) if obj_trans_in_window else None
    dom_approach = max(approach_mm, key=lambda kv: kv[1]) if approach_mm else None
    # min tip->centroid frame in window (closest approach)
    min_tip_frame = int(w[np.nanargmin(min_tip_c[w])]) if not np.isnan(min_tip_c[w]).all() else None

    # time-lock test: does the dominant object-motion edge fall within +/-1 of
    # the closest-approach frame OR the dominant approach-speed frame?
    def within(f1, f2, tol=1):
        return f1 is not None and f2 is not None and abs(f1 - f2) <= tol

    timelock_closest = within(dom_obj_edge[0] if dom_obj_edge else None, min_tip_frame)
    timelock_approach = within(dom_obj_edge[0] if dom_obj_edge else None,
                               dom_approach[0] if dom_approach else None)

    # Does the dominant object edge actually clear the motion floor?
    dom_clears_floor = (dom_obj_edge is not None and dom_obj_edge[1] > coupling_floor_mm)
    dom_rot_clears = False
    if dom_obj_edge:
        rf = next((rr[1] for rr in obj_rot_in_window if rr[0] == dom_obj_edge[0]), 0.0)
        dom_rot_clears = rf > coupling_floor_rot_deg

    # --- Verdict logic ---
    # A frame can be promoted beyond `unresolved` ONLY if there is a real
    # object-motion impulse (clears the independent motion floor) that is
    # time-locked (<=1 frame) to a fingertip approach, at a direct-edge frame.
    # This is the floor-independent KT-U release valve; distance is never used.
    promotes = []
    if dom_clears_floor and (timelock_closest or timelock_approach):
        promotes.append(int(dom_obj_edge[0]))
    # any additional direct edges that clear floor AND coincide with approach peak frame?
    for (f, dmm) in obj_trans_in_window:
        if dmm > coupling_floor_mm and f != (dom_obj_edge[0] if dom_obj_edge else -1):
            if within(f, min_tip_frame) or within(f, dom_approach[0] if dom_approach else None):
                promotes.append(int(f))

    # --- Localized impulse test at the fingertip-approach frame ---
    # For a table-rested keyboard, a hand-driven typing impulse is expected to
    # be a LOCALIZED spike at the approach/contact frame (key travel ~2-4mm +
    # minor rocking, i.e. a few mm) that is LARGER than the deltas at frames
    # where the hand is far. We test the converse: is the keyboard delta at the
    # closest-approach frame among the LARGEST in the window, or is it
    # comparable-to-smaller (indicating pose-fit jitter, not coupling)?
    PHYS_TYPING_PRIOR_MM = 5.0  # physical prior: hand-driven typing translation <= few mm
    window_direct_dtrans = [obj_dtrans[f] for f in w_direct_edges]
    approach_frame_kb_response_mm = None
    localized_impulse = False
    localized_impulse_reason = None
    # candidate coupling probe frames: the closest-approach frame and the
    # approach-speed peak frame (the two places coupling would manifest).
    probe_frames = []
    if min_tip_frame is not None:
        probe_frames.append(("closest_approach", min_tip_frame))
    if dom_approach:
        probe_frames.append(("approach_speed_peak", int(dom_approach[0])))
    for label, pf in probe_frames:
        if not obj_trans_direct_edge[pf]:
            continue
        d = obj_dtrans[pf] * 1000.0
        if approach_frame_kb_response_mm is None or d > approach_frame_kb_response_mm:
            approach_frame_kb_response_mm = d
            probe_label = label
            probe_pf = pf
    if approach_frame_kb_response_mm is not None and len(window_direct_dtrans) >= 3:
        wmed = float(np.median(window_direct_dtrans) * 1000.0)
        wmax = float(np.max(window_direct_dtrans) * 1000.0)
        localized_impulse = (
            approach_frame_kb_response_mm > PHYS_TYPING_PRIOR_MM
            and approach_frame_kb_response_mm >= wmax - 1e-6
            and approach_frame_kb_response_mm > wmed
        )
        localized_impulse_reason = (
            f"probe-frame(f{probe_pf}, {probe_label}) kb delta={approach_frame_kb_response_mm:.1f}mm "
            f"vs window median={wmed:.1f}mm max={wmax:.1f}mm, typing prior={PHYS_TYPING_PRIOR_MM}mm"
        )
    elif approach_frame_kb_response_mm is None:
        localized_impulse_reason = (
            f"no probe frame (closest-approach f{min_tip_frame}, approach-peak "
            f"f{dom_approach[0] if dom_approach else None}) is a DIRECT pose edge; "
            f"keyboard motion at the approach is not independently measurable"
        )
    # Spearman: object dtrans vs fingertip approach speed on window direct edges
    spearman = None
    try:
        from scipy.stats import spearmanr
        xs = [obj_dtrans[f] for f in w_direct_edges]
        ys = [approach[f] for f in w_direct_edges]
        if len(xs) >= 4:
            rho, p = spearmanr(xs, ys)
            spearman = {"rho": float(rho), "p": float(p), "n": len(xs)}
    except Exception as e:
        spearman = {"error": str(e)}

    # overall verdict
    if not promotes:
        if dom_obj_edge is None or obj_window_amp_mm is None:
            verdict = "motion_coupling_absent_no_independent_object_motion_in_window"
        elif obj_window_amp_mm < 5.0:
            verdict = "motion_coupling_absent_keyboard_motion_negligible"
        elif not dom_clears_floor:
            verdict = "motion_coupling_absent_no_impulse_above_motion_floor"
        else:
            verdict = "motion_coupling_present_but_not_time_locked_to_hand_approach"
    else:
        verdict = "motion_coupling_candidate_promotion_frames"

    # --- per-frame ndjson ---
    per_frame = []
    for f in range(n):
        row = {
            "frame_idx": f,
            "in_window": WINDOW[0] <= f <= WINDOW[1],
            "object": {
                "direct_visible": bool(obj_direct[f]),
                "pose_source": obj_pose_src[f],
                "status": obj_status[f],
                "centroid_world_m": obj_t[f].tolist() if not np.isnan(obj_t[f]).any() else None,
                "dtrans_mm": None if f == 0 or np.isnan(obj_dtrans[f]) else obj_dtrans[f] * 1000.0,
                "drot_deg": None if f == 0 or np.isnan(obj_drot[f]) else math.degrees(obj_drot[f]),
                "edge_direct_to_direct": bool(obj_trans_direct_edge[f]),
            },
            "right_hand": {
                "present": bool(rh_present[f]),
                "root_world_m": rh_root[f].tolist() if rh_present[f] else None,
                "droot_mm": rh_droot[f] * 1000.0 if rh_present[f] and f > 0 else None,
                "min_tip_to_centroid_mm": (min_tip_c[f] * 1000.0) if not np.isnan(min_tip_c[f]) else None,
                "approach_speed_mm_per_frame": (approach[f] * 1000.0) if f > 0 else None,
            },
            "contact_state_canonical": cfd.get(f, {}).get("contact_state") if cfd else None,
        }
        if WINDOW[0] <= f <= WINDOW[1]:
            row["motion_coupling"] = {
                "coupling_floor_trans_mm": coupling_floor_mm,
                "coupling_floor_rot_deg": coupling_floor_rot_deg,
                "promoted_by_motion_coupling": f in promotes,
            }
        per_frame.append(row)
    with (OUT / "per_frame.ndjson").open("w") as fh:
        for r in per_frame:
            fh.write(json.dumps(r) + "\n")

    # --- summary ---
    summary = {
        "schema": "org.ego.hoi.motion_coupling_adjudication/0.1.0",
        "case": "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1",
        "channel": "object_motion_coupling_to_hand_approach",
        "window": list(WINDOW),
        "no_distance_threshold_used": True,
        "object_motion_floor": {
            "basis": "robust median + 3*MAD of per-frame keyboard translation delta over DIRECT->DIRECT pose edges across full timeline (only independent object-motion measurements; held/interp transitions are artificially zero)",
            "direct_edge_count": int(len(direct_edges)),
            "translation_floor_mm": coupling_floor_mm,
            "rotation_floor_deg": coupling_floor_rot_deg,
            "translation_direct_stats_mm": {k: v * 1000.0 if k != "n" else v for k, v in obj_trans_floor.items()} if obj_trans_floor else None,
            "rotation_direct_stats_deg": {k: math.degrees(v) if k != "n" else v for k, v in obj_rot_floor.items()} if obj_rot_floor else None,
        },
        "window_object_motion": {
            "first_direct_frame": int(wdirect[0]) if wdirect else None,
            "last_direct_frame": int(wdirect[-1]) if wdirect else None,
            "total_translation_amplitude_mm": obj_window_amp_mm,
            "total_rotation_amplitude_deg": obj_window_rot_deg,
            "right_hand_total_root_translation_mm": rh_window_amp_mm,
            "min_tip_to_centroid_drop_mm": tip_c_drop_mm,
            "object_to_hand_motion_ratio": (obj_window_amp_mm / rh_window_amp_mm) if (obj_window_amp_mm and rh_window_amp_mm) else None,
        },
        "dominant_object_motion_edge": {
            "frame": int(dom_obj_edge[0]) if dom_obj_edge else None,
            "dtrans_mm": float(dom_obj_edge[1]) if dom_obj_edge else None,
            "clears_translation_floor": bool(dom_clears_floor),
            "dom_edge_clears_rotation_floor": bool(dom_rot_clears),
        },
        "dominant_fingertip_approach": {
            "frame": int(dom_approach[0]) if dom_approach else None,
            "approach_speed_mm_per_frame": float(dom_approach[1]) if dom_approach else None,
        },
        "closest_fingertip_to_centroid_frame": min_tip_frame,
        "time_lock": {
            "dom_obj_edge_vs_closest_approach_within_1_frame": bool(timelock_closest),
            "dom_obj_edge_vs_approach_speed_peak_within_1_frame": bool(timelock_approach),
        },
        "physical_typing_prior_mm": PHYS_TYPING_PRIOR_MM,
        "approach_frame_keyboard_response": {
            "closest_approach_frame": min_tip_frame,
            "is_direct_edge": bool(min_tip_frame and obj_trans_direct_edge[min_tip_frame]),
            "keyboard_translation_mm": approach_frame_kb_response_mm,
            "localized_impulse": bool(localized_impulse),
            "reason": localized_impulse_reason,
        },
        "spearman_object_dtrans_vs_approach_speed_window": spearman,
        "promoted_frames": sorted(set(int(x) for x in promotes)),
        "verdict": verdict,
        "interpretation": {
            "motion_coupling_absent_keyboard_motion_negligible":
                "Total keyboard translation across window <5 mm while hand moves cm-to-dm; no object motion exists to couple. confirmed_contact remains unreachable GT-free.",
            "motion_coupling_absent_no_impulse_above_motion_floor":
                "No keyboard per-frame motion edge exceeds the independent direct-edge motion floor (median+3*MAD); apparent deltas are pose-measurement jitter, not a hand-driven impulse.",
            "motion_coupling_present_but_not_time_locked_to_hand_approach":
                "A keyboard motion impulse clears the floor but does not coincide (+/-1 frame) with the fingertip approach peak or closest-approach frame, so it is not attributable to hand contact.",
            "motion_coupling_candidate_promotion_frames":
                "A keyboard motion impulse clears the floor AND is time-locked (+/-1 frame) to a fingertip approach; listed frames are candidate promotions requiring GT (R8) confirmation.",
            "motion_coupling_absent_no_independent_object_motion_in_window":
                "Window contains no direct->direct object pose edges, so no independent object motion is measurable; coupling is untestable here.",
        }.get(verdict, verdict),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- review plot ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fwin = np.arange(max(0, WINDOW[0] - 4), min(n, WINDOW[1] + 5))
    # 1. object translation delta (direct edges emphasized)
    ax = axes[0]
    dth = coupling_floor_mm
    ax.bar(fwin, [obj_dtrans[f] * 1000 for f in fwin], width=0.6,
           color=["crimson" if obj_trans_direct_edge[f] else "lightgray" for f in fwin])
    if dth and dth < 1e3:
        ax.axhline(dth, color="black", ls="--", lw=1, label=f"motion floor {dth:.2f}mm")
    ax.axvspan(WINDOW[0], WINDOW[1], color="yellow", alpha=0.08)
    ax.set_ylabel("keyboard dtrans\n[mm/frame]")
    ax.set_title("clip001850 motion-coupling adjudication (right hand, frames 28-48)")
    ax.legend(loc="upper right", fontsize=8)
    # 2. object rotation delta
    ax = axes[1]
    ax.bar(fwin, [math.degrees(obj_drot[f]) for f in fwin], width=0.6,
           color=["crimson" if obj_trans_direct_edge[f] else "lightgray" for f in fwin])
    if coupling_floor_rot_deg < 180:
        ax.axhline(coupling_floor_rot_deg, color="black", ls="--", lw=1,
                   label=f"rot floor {coupling_floor_rot_deg:.2f}deg")
    ax.axvspan(WINDOW[0], WINDOW[1], color="yellow", alpha=0.08)
    ax.set_ylabel("keyboard drot\n[deg/frame]")
    ax.legend(loc="upper right", fontsize=8)
    # 3. right hand root motion + fingertip approach speed
    ax = axes[2]
    ax.bar(fwin, [rh_droot[f] * 1000 for f in fwin], width=0.6, color="steelblue", alpha=0.6, label="R hand droot")
    ax2 = ax.twinx()
    ax2.plot(fwin, [approach[f] * 1000 for f in fwin], "g.-", lw=1, ms=4, label="fingertip approach speed")
    ax.axvspan(WINDOW[0], WINDOW[1], color="yellow", alpha=0.08)
    ax.set_ylabel("R hand droot [mm]", color="steelblue")
    ax2.set_ylabel("approach [mm/frame]", color="green")
    # 4. min tip->centroid distance
    ax = axes[3]
    ax.plot(fwin, [min_tip_c[f] * 1000 if not np.isnan(min_tip_c[f]) else np.nan for f in fwin], ".-", color="purple")
    ax.axvspan(WINDOW[0], WINDOW[1], color="yellow", alpha=0.08)
    if min_tip_frame is not None:
        ax.axvline(min_tip_frame, color="purple", ls=":", lw=1, label=f"closest f{min_tip_frame}")
    if dom_obj_edge:
        ax.axvline(dom_obj_edge[0], color="crimson", ls=":", lw=1, label=f"obj impulse f{dom_obj_edge[0]}")
    ax.set_ylabel("min tip->centroid\n[mm]")
    ax.set_xlabel("frame")
    ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xticks(fwin[::2])
    fig.tight_layout()
    fig.savefig(OUT / "review.png", dpi=120)
    plt.close(fig)

    # console digest
    print("VERDICT:", verdict)
    print("window object total translation: ", summary["window_object_motion"]["total_translation_amplitude_mm"], "mm")
    print("window right hand root translation:", summary["window_object_motion"]["right_hand_total_root_translation_mm"], "mm")
    print("object/hand motion ratio:", summary["window_object_motion"]["object_to_hand_motion_ratio"])
    print("dominant object motion edge:", summary["dominant_object_motion_edge"])
    print("dominant approach:", summary["dominant_fingertip_approach"])
    print("closest tip->centroid frame:", min_tip_frame)
    print("time_lock:", summary["time_lock"])
    print("promoted_frames:", summary["promoted_frames"])
    print("OUT:", OUT)


if __name__ == "__main__":
    main()
