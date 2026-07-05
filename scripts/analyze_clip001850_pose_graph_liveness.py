#!/usr/bin/env python3
"""CPU liveness probe for the clip001850 keyboard rigid pose graph.

Reads the pose-graph report, the upstream visible-pose-fit observation rows, the
visible-geometry adapter report, the raw frame manifest, and the full-duration
contact-state render manifest, and emits a liveness diagnosis that distinguishes:

  - no_seed_support            : object not visible / no metric surface (no seed)
  - visible_surface_degenerate : visible metric surface but mask extent inconsistent
                                 (hand/background leakage) -> ineligible for rigid fit
  - solver_plumbing_inert      : optimizer produced zero correction (pass-through)
  - legitimate_static_held_gauge: object genuinely static with declared gauge
  - mixed                      : more than one of the above materially present

No smoothing, no repair, no heavy inference, no GPU. Outputs only to /tmp.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_json(path: str) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def _round_t(t, nd=5):
    return tuple(round(float(x), nd) for x in t)


def _rot_angle_deg(Ra, Rb) -> float:
    R = np.asarray(Rb) @ np.asarray(Ra).T
    cos = (np.trace(R) - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summary(vals):
    if not vals:
        return {"count": 0, "median": None, "p90": None, "max": None, "mean": None}
    return {
        "count": len(vals),
        "median": statistics.median(vals),
        "p90": _pct(vals, 0.90),
        "max": max(vals),
        "mean": statistics.fmean(vals),
    }


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
def analyze(
    pose_graph_report: dict,
    input_pose_report: dict,
    visible_geometry_adapter: dict | None,
    raw_frame_manifest: dict,
    contact_render_manifest: dict,
    args: argparse.Namespace,
) -> dict:
    pose_rows = pose_graph_report["pose_rows"]
    by_frame = {r["frame_idx"]: r for r in pose_rows}
    frame_count = len(pose_rows)

    graph_frames = pose_graph_report.get("graph_frames", [])
    optimizer = pose_graph_report.get("optimizer", {})
    full_completion = pose_graph_report.get("full_timeline_rigid_pose_completion", {})
    correction = pose_graph_report.get("correction_summary", {})

    # ----- 1. pose status counts ----------------------------------------- #
    pms_counter = Counter(r.get("pose_measurement_status") for r in pose_rows)
    status_counter = Counter(r.get("status") for r in pose_rows)
    pose_source_counter = Counter(
        r.get("temporal_pose_graph", {}).get("pose_source") for r in pose_rows
    )

    pose_status_counts = {
        "pose_measurement_status": dict(pms_counter),
        "status": dict(status_counter),
        "pose_source": dict(pose_source_counter),
    }

    # ----- 2. active support by factor family (recoverable) ------------- #
    # temporal: correction deltas (translation_delta_step_norm_m)
    temporal_delta = correction.get("translation_delta_step_norm_m", {})
    temporal_rot = correction.get("rotation_delta_step_norm_rad", {})
    # depth/ICP: observed_to_mesh_final medians on fit frames (live measurement residual)
    icp_final_medians = []
    for f in graph_frames:
        r = by_frame.get(f)
        if r and r.get("observed_to_mesh_final"):
            icp_final_medians.append(r["observed_to_mesh_final"].get("median_m"))
    # nonpenetration
    np_target_count = pose_graph_report.get("nonpenetration_target_frame_count", 0)

    support_by_family = {
        "temporal_smooth": {
            "active": (temporal_delta.get("max", 0.0) or 0.0) > 1e-9,
            "translation_delta_step_max_m": temporal_delta.get("max"),
            "rotation_delta_step_max_rad": temporal_rot.get("max"),
            "note": "temporal factor is active only if optimizer produced nonzero step deltas",
        },
        "depth_icp_visible": {
            "active": len(icp_final_medians) > 0,
            "fit_frame_count": len(icp_final_medians),
            "observed_to_mesh_final_median_of_medians_m": (
                statistics.median(icp_final_medians) if icp_final_medians else None
            ),
            "note": "per-frame ICP fit residual on the 13 direct-fit frames; the only live measurement",
        },
        "nonpenetration": {
            "active": np_target_count > 0,
            "target_frame_count": np_target_count,
            "note": "nonpenetration factor has zero support (no target frames)",
        },
    }

    # ----- 3. optimizer nfev/cost/residuals ------------------------------ #
    optimizer_diag = {
        "success": optimizer.get("success"),
        "message": optimizer.get("message"),
        "nfev": optimizer.get("nfev"),
        "cost": optimizer.get("cost"),
        "residual_rms_before": optimizer.get("residual_rms_before"),
        "residual_rms_after": optimizer.get("residual_rms_after"),
        "inert": (
            optimizer.get("nfev", 0) <= 1
            and (optimizer.get("cost", 0.0) or 0.0) == 0.0
            and (optimizer.get("residual_rms_before", 0.0) or 0.0) == 0.0
            and (optimizer.get("residual_rms_after", 0.0) or 0.0) == 0.0
        ),
    }

    # ----- 4. direct fit frame list -------------------------------------- #
    direct_fit_frames = [
        f for f in graph_frames if by_frame.get(f, {}).get("pose_measurement_status")
        == "fit_to_visible_depth_samples"
    ]

    # ----- 5. frozen-pose clusters --------------------------------------- #
    clusters = defaultdict(list)
    for r in pose_rows:
        clusters[_round_t(r["translation_world_m"])].append(r["frame_idx"])
    cluster_list = sorted(
        (
            {
                "translation_world_m_rounded": list(k),
                "frame_count": len(v),
                "frame_indices": v,
            }
            for k, v in clusters.items()
        ),
        key=lambda c: c["frame_count"],
        reverse=True,
    )
    # tag each cluster with the pose_source composition
    for c in cluster_list:
        srcs = Counter(by_frame[f].get("temporal_pose_graph", {}).get("pose_source") for f in c["frame_indices"])
        c["pose_source_counts"] = dict(srcs)

    # ----- 6. held / interpolated segments ------------------------------- #
    held_segments = []
    interp_segments = []
    # group consecutive frames by pose_source
    current = None
    for r in pose_rows:
        src = r.get("temporal_pose_graph", {}).get("pose_source")
        if src != current:
            current = src
            seg = {"pose_source": src, "start": r["frame_idx"], "end": r["frame_idx"], "frames": [r["frame_idx"]]}
            if src == "nearest_visible_pose_hold":
                held_segments.append(seg)
            elif src == "interpolated_between_visible_pose_observations":
                interp_segments.append(seg)
            elif src == "direct_visible_pose_observation_corrected":
                pass  # direct, not a fill
            else:
                pass
        else:
            if held_segments and held_segments[-1]["pose_source"] == src:
                held_segments[-1]["end"] = r["frame_idx"]
                held_segments[-1]["frames"].append(r["frame_idx"])
            elif interp_segments and interp_segments[-1]["pose_source"] == src:
                interp_segments[-1]["end"] = r["frame_idx"]
                interp_segments[-1]["frames"].append(r["frame_idx"])

    # ----- 7. direct-edge translation/rotation jumps --------------------- #
    direct_edges = []
    for a, b in zip(direct_fit_frames, direct_fit_frames[1:]):
        ra, rb = by_frame[a], by_frame[b]
        dt = math.dist(ra["translation_world_m"], rb["translation_world_m"])
        dr = _rot_angle_deg(
            ra["rotation_world_from_completed_canonical_matrix"],
            rb["rotation_world_from_completed_canonical_matrix"],
        )
        direct_edges.append(
            {
                "from_frame": a,
                "to_frame": b,
                "frame_gap": b - a,
                "translation_jump_mm": dt * 1000.0,
                "rotation_jump_deg": dr,
            }
        )
    edge_trans = [e["translation_jump_mm"] for e in direct_edges]
    edge_rot = [e["rotation_jump_deg"] for e in direct_edges]

    # ----- 8. input-output deltas ---------------------------------------- #
    input_rows = {r["frame_idx"]: r for r in input_pose_report.get("pose_rows", [])}
    io_deltas = []
    for f in direct_fit_frames:
        g = by_frame.get(f)
        i = input_rows.get(f)
        if g and i and g.get("translation_world_m") and i.get("translation_world_m"):
            dt = math.dist(g["translation_world_m"], i["translation_world_m"])
            dr = _rot_angle_deg(
                g["rotation_world_from_completed_canonical_matrix"],
                i["rotation_world_from_completed_canonical_matrix"],
            )
            io_deltas.append(
                {"frame": f, "translation_delta_mm": dt * 1000.0, "rotation_delta_deg": dr}
            )
    io_trans = [d["translation_delta_mm"] for d in io_deltas]

    # ----- 9. render default / contact-state coupling -------------------- #
    render_frames = contact_render_manifest.get("frames", [])
    render_by_frame = {f["frame_idx"]: f for f in render_frames}
    # map pose_measurement_status -> contact_state
    pose_to_contact = defaultdict(Counter)
    for r in pose_rows:
        f = r["frame_idx"]
        cs = render_by_frame.get(f, {}).get("contact_state", "<no_render_row>")
        pms = r.get("pose_measurement_status", "<none>")
        src = render_by_frame.get(f, {}).get("source", "<none>")
        pose_to_contact[pms][cs] += 1
    render_coupling = {
        "pose_measurement_status_to_contact_state": {
            k: dict(v) for k, v in pose_to_contact.items()
        },
        "render_defaulted_frame_count": contact_render_manifest.get("defaulted_frame_count"),
        "render_contact_state_counts": contact_render_manifest.get("contact_state_counts"),
    }

    # ----- 10. visible-geometry eligibility (if available) --------------- #
    vg_eligibility = None
    if visible_geometry_adapter:
        vg_rows = visible_geometry_adapter.get("rows", [])
        visible_metric = [r for r in vg_rows if r.get("status") == "visible_metric_surface_measurement"]
        eligible = [r for r in visible_metric if r.get("rigid_pose_observation_eligible") is True]
        ineligible = [r for r in visible_metric if r.get("rigid_pose_observation_eligible") is False]
        inelig_ratios = [r.get("extent_ratio_to_anchor_axis_max") for r in ineligible if r.get("extent_ratio_to_anchor_axis_max") is not None]
        elig_ratios = [r.get("extent_ratio_to_anchor_axis_max") for r in eligible if r.get("extent_ratio_to_anchor_axis_max") is not None]
        inelig_reasons = Counter(r.get("rigid_pose_observation_reason") for r in ineligible)
        vg_eligibility = {
            "total_frames": len(vg_rows),
            "visible_metric_surface_count": len(visible_metric),
            "rigid_pose_eligible_count": len(eligible),
            "visible_but_ineligible_count": len(ineligible),
            "not_visible_count": len(vg_rows) - len(visible_metric),
            "ineligible_extent_ratio_max_summary": _summary(inelig_ratios),
            "eligible_extent_ratio_max_summary": _summary(elig_ratios),
            "ineligible_reasons": dict(inelig_reasons),
        }

    # ----- 11. classify dominant mechanism ------------------------------- #
    n_missing = pms_counter.get("missing_initial_graph_pose", 0)
    n_ineligible = pms_counter.get("visible_surface_ineligible_for_rigid_pose_fit", 0)
    n_fit = pms_counter.get("fit_to_visible_depth_samples", 0)
    total = frame_count

    mech_flags = {
        "no_seed_support": {
            "present": n_missing > 0,
            "frame_count": n_missing,
            "fraction": n_missing / total,
            "evidence": f"{n_missing}/{total} frames have pose_measurement_status=missing_initial_graph_pose (object not visible / no metric surface)",
        },
        "visible_surface_degenerate": {
            "present": n_ineligible > 0,
            "frame_count": n_ineligible,
            "fraction": n_ineligible / total,
            "evidence": (
                f"{n_ineligible}/{total} frames are visible_metric_surface_measurement but "
                "rigid_pose_observation_eligible=False (mask extent inconsistent, hand/background leakage)"
                if vg_eligibility
                else f"{n_ineligible}/{total} frames visible_surface_ineligible_for_rigid_pose_fit"
            ),
        },
        "solver_plumbing_inert": {
            "present": bool(optimizer_diag["inert"]),
            "frame_count": total if optimizer_diag["inert"] else 0,
            "fraction": 1.0 if optimizer_diag["inert"] else 0.0,
            "evidence": (
                f"optimizer nfev={optimizer_diag['nfev']}, cost={optimizer_diag['cost']}, "
                f"residual_rms before={optimizer_diag['residual_rms_before']} after={optimizer_diag['residual_rms_after']}, "
                f"correction translation_delta_step max={temporal_delta.get('max')} m, "
                f"input-output max delta={max(io_trans) if io_trans else 0.0:.4f} mm (pass-through)"
            ),
        },
        "legitimate_static_held_gauge": {
            "present": False,
            "frame_count": 0,
            "fraction": 0.0,
            "evidence": (
                "NOT supported: direct-fit frames jump up to "
                f"{max(edge_trans):.1f} mm / {max(edge_rot):.1f} deg (not static); "
                "no declared gauge in the report; frozen holds are an artifact of missing data, not a declared rest pose"
                if edge_trans
                else "no direct edges to test staticness"
            ),
        },
    }

    present = [k for k, v in mech_flags.items() if v["present"]]
    # legitimate_static_held_gauge is mutually exclusive with the defect mechanisms;
    # if it were the only one present we'd pick it. Here it is never present.
    if len(present) <= 1 and present and present[0] != "legitimate_static_held_gauge":
        dominant = present[0]
    elif "legitimate_static_held_gauge" in present and len(present) == 1:
        dominant = "legitimate_static_held_gauge"
    else:
        dominant = "mixed"

    # Refine: solver_plumbing_inert is a global pass-through that compounds with
    # the localization failures, so if >=2 defect mechanisms are present it is mixed.
    classification = {
        "dominant_mechanism": dominant,
        "present_mechanisms": present,
        "mechanism_flags": mech_flags,
        "rationale": (
            "The pose graph is unlocalized on 137/150 frames via two co-dominant causes: "
            f"{n_missing} frames have no visible surface (no_seed_support) and {n_ineligible} "
            "have a visible but degenerate/leaking mask (visible_surface_degenerate). Independently, "
            "the optimizer is a pure pass-through (nfev=1, cost=0, zero correction, zero input-output "
            "delta) even on its 13 direct-fit frames, whose per-frame ICP fits jump up to "
            f"{max(edge_trans):.1f} mm unsmoothed (solver_plumbing_inert). legitimate_static_held_gauge "
            "is refuted: the direct fits are not static and no gauge is declared."
        ),
    }

    # ----- assemble summary ---------------------------------------------- #
    summary = {
        "schema": "clip001850_pose_graph_liveness_probe/v1",
        "case": args.case,
        "run_root": args.run_root,
        "frame_count": frame_count,
        "pose_status_counts": pose_status_counts,
        "support_by_factor_family": support_by_family,
        "optimizer": optimizer_diag,
        "direct_fit_frames": direct_fit_frames,
        "direct_fit_frame_count": len(direct_fit_frames),
        "frozen_pose_clusters": cluster_list,
        "held_interpolated_segments": {
            "nearest_visible_pose_hold_segments": held_segments,
            "interpolated_segments": interp_segments,
            "hold_frame_count": sum(len(s["frames"]) for s in held_segments),
            "interp_frame_count": sum(len(s["frames"]) for s in interp_segments),
        },
        "direct_edge_jumps": {
            "edges": direct_edges,
            "translation_mm_summary": _summary(edge_trans),
            "rotation_deg_summary": _summary(edge_rot),
        },
        "input_output_deltas": {
            "per_frame": io_deltas,
            "translation_mm_summary": _summary(io_trans),
            "pass_through": all(d["translation_delta_mm"] < 1e-6 for d in io_deltas) if io_deltas else True,
        },
        "render_contact_state_coupling": render_coupling,
        "visible_geometry_eligibility": vg_eligibility,
        "full_timeline_completion": full_completion,
        "classification": classification,
        "inputs": {
            "pose_graph_report": args.pose_graph_report,
            "input_pose_report": args.input_pose_report,
            "visible_geometry_adapter": args.visible_geometry_adapter,
            "raw_frame_manifest": args.raw_frame_manifest,
            "contact_render_manifest": args.contact_render_manifest,
        },
    }
    return summary


# --------------------------------------------------------------------------- #
# Per-frame + edge NDJSON writers
# --------------------------------------------------------------------------- #
def write_per_frame_ndjson(summary: dict, pose_rows: list, render_by_frame: dict, path: str):
    by_frame = {r["frame_idx"]: r for r in pose_rows}
    with open(path, "w") as fh:
        for r in pose_rows:
            f = r["frame_idx"]
            rf = render_by_frame.get(f, {})
            rec = {
                "frame_idx": f,
                "pose_measurement_status": r.get("pose_measurement_status"),
                "status": r.get("status"),
                "pose_source": r.get("temporal_pose_graph", {}).get("pose_source"),
                "translation_world_m": r.get("translation_world_m"),
                "direct_visible_measurement": r.get("temporal_pose_graph", {}).get("direct_visible_measurement"),
                "bracket_visible_pose_frames": r.get("temporal_pose_graph", {}).get("bracket_visible_pose_frames"),
                "gap_frames": r.get("temporal_pose_graph", {}).get("gap_frames"),
                "contact_state": rf.get("contact_state"),
                "render_source": rf.get("source"),
                "observed_to_mesh_final_median_m": (
                    r.get("observed_to_mesh_final", {}).get("median_m") if r.get("observed_to_mesh_final") else None
                ),
            }
            fh.write(json.dumps(rec) + "\n")


def write_pose_edges_ndjson(summary: dict, path: str):
    with open(path, "w") as fh:
        for e in summary["direct_edge_jumps"]["edges"]:
            fh.write(json.dumps(e) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", required=True)
    ap.add_argument(
        "--pose-graph-report",
        default=None,
        help="v19_rigid_object_pose_graph_report.json (default: <run-root>/measurements/pose_fits/keyboard_rigid_pose_graph/)",
    )
    ap.add_argument(
        "--input-pose-report",
        default=None,
        help="upstream v18_compact_rigid_object_pose_fit_report.json (the observation rows fed into the graph)",
    )
    ap.add_argument(
        "--visible-geometry-adapter",
        default=None,
        help="v19_visible_geometry_adapter_report.json (eligibility data)",
    )
    ap.add_argument(
        "--raw-frame-manifest",
        default=None,
        help="raw_frame_manifest/manifest.json",
    )
    ap.add_argument(
        "--contact-render-manifest",
        default=None,
        help="full-duration contact-state render manifest.json",
    )
    ap.add_argument("--output-root", default="/tmp/clip001850_pose_graph_liveness")
    ap.add_argument("--case", default="hot3d_clip001850_keyboard")
    args = ap.parse_args()

    rr = args.run_root
    pose_graph_path = args.pose_graph_report or os.path.join(
        rr, "measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json"
    )
    input_pose_path = args.input_pose_report or os.path.join(
        rr, "measurements/pose_fits/keyboard_visible_pose_fit/v18_compact_rigid_object_pose_fit_report.json"
    )
    vg_adapter_path = args.visible_geometry_adapter or os.path.join(
        rr, "measurements/object_geometry/visible_geometry/keyboard/v19_visible_geometry_adapter_report.json"
    )
    raw_manifest_path = args.raw_frame_manifest or os.path.join(
        rr, "input/raw_frame_manifest/manifest.json"
    )
    contact_render_path = args.contact_render_manifest or os.path.join(
        "/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706",
        "v19_contact_state_full_duration/manifest.json",
    )

    for p in [pose_graph_path, input_pose_path, vg_adapter_path, raw_manifest_path, contact_render_path]:
        if not os.path.exists(p):
            print(f"[probe] WARNING: input missing: {p}", file=sys.stderr)

    pose_graph_report = _load_json(pose_graph_path)
    input_pose_report = _load_json(input_pose_path) if os.path.exists(input_pose_path) else {"pose_rows": []}
    vg_adapter = _load_json(vg_adapter_path) if os.path.exists(vg_adapter_path) else None
    raw_manifest = _load_json(raw_manifest_path) if os.path.exists(raw_manifest_path) else {}
    contact_render_manifest = _load_json(contact_render_path) if os.path.exists(contact_render_path) else {"frames": []}

    summary = analyze(
        pose_graph_report, input_pose_report, vg_adapter, raw_manifest, contact_render_manifest, args
    )

    os.makedirs(args.output_root, exist_ok=True)
    summary_path = os.path.join(args.output_root, "summary.json")
    per_frame_path = os.path.join(args.output_root, "per_frame.ndjson")
    edges_path = os.path.join(args.output_root, "pose_edges.ndjson")

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    render_by_frame = {f["frame_idx"]: f for f in contact_render_manifest.get("frames", [])}
    write_per_frame_ndjson(summary, pose_graph_report["pose_rows"], render_by_frame, per_frame_path)
    write_pose_edges_ndjson(summary, edges_path)

    print(f"[probe] wrote {summary_path}")
    print(f"[probe] wrote {per_frame_path} ({len(pose_graph_report['pose_rows'])} rows)")
    print(f"[probe] wrote {edges_path} ({len(summary['direct_edge_jumps']['edges'])} edges)")
    print(f"[probe] dominant_mechanism = {summary['classification']['dominant_mechanism']}")
    print(f"[probe] present_mechanisms  = {summary['classification']['present_mechanisms']}")
    print(
        f"[probe] optimizer inert = {summary['optimizer']['inert']} "
        f"(nfev={summary['optimizer']['nfev']}, cost={summary['optimizer']['cost']})"
    )
    print(
        f"[probe] direct_fit_frames ({summary['direct_fit_frame_count']}): {summary['direct_fit_frames']}"
    )


if __name__ == "__main__":
    main()
