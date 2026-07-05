#!/usr/bin/env python3
"""Reconcile triple-valued hand-object contact distance for HOT3D clip001850
keyboard, right hand, frames 26-46.

This is the CPU-only D-a/D-b/D-d reconciliation experiment named by the first
research slice causal card
(.memory/tasks/2026-07-06-research-track-execution/subagents/02_first_slice_causal_card.md).

Artifact defect being repaired
------------------------------
The same right-hand contact relation over frames 26-46 is triple-valued:
  * interval MANO           : ~0.10 m PENETRATION into observed depth surface
  * contact/nonpenetration  : 0 penetrating vertices against completed TRELLIS
                              mesh (non-watertight -> signed query disabled)
  * published render        : +0.039 m GAP

Mechanisms discriminated
------------------------
  M1/M5 : contact/NP disabled by non-watertight sign mesh while a coherent
          observed-surface penetration signal sits stranded in the interval
          solver (cross-solver geometry-source decoupling).
  M2    : geometry epoch contamination -- the 10 cm "penetration" is the hand
          pressing into hand/table points mis-labelled as object surface, or
          the completed mesh is too inflated/contaminated to be trusted.

What this script measures (all CPU, trimesh closest_point on existing PLYs)
---------------------------------------------------------------------------
  D-a : for the SAME per-frame hand-vertex set under the SAME per-frame object
        pose, recompute closest-point distance to (i) the observed depth surface
        mesh and (ii) the completed mesh, in the completed-mesh canonical frame.
        Cross-check the interval solver's observed-surface penetration and the
        contact/NP nearest-surface-unsigned number. Report the geometry-source
        disagreement.
  D-b : temporal coherence (IoU) of the interval solver's contact-patch
        penetrating MANO vertex ids across consecutive frames.
  D-d : completed-mesh AABB extent vs a keyboard prior (~0.45 x 0.15 x 0.03 m)
        and observed-surface extent; free-space note from the completion report.

Output
------
A joined per-frame cross-source ledger (ego.hoi-style contact_frame_detail +
geometry provenance) written as NDJSON + a summary JSON. Every required field is
emitted; where a value cannot be recomputed the exact blocking reason is stated.

No heavy inference. No GPU. No edits to run-root artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "trimesh is required (available in the project venv: "
        ".venv/bin/python). " + str(exc) + "\n"
    )
    raise

# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #

DEFAULT_RUN_ROOT = (
    "/data2/ego_annotation_outputs/v19_runs/"
    "20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)

# Keyboard physical-extent prior used for the D-d plausibility test (metres).
KEYBOARD_PRIOR_EXTENTS_M = [0.45, 0.15, 0.03]
KEYBOARD_PRIOR_SOURCE = "first-slice causal card D-d prior (~0.45 x 0.15 x 0.03 m)"

FRAME_LO = 26
FRAME_HI = 46
HAND_SIDE = "right"

# Relative paths inside the run root.
REL = {
    "pose_graph": "measurements/pose_fits/keyboard_rigid_pose_graph/"
                  "v19_rigid_object_pose_graph_report.json",
    "interval": "measurements/mano_interval_correction/keyboard_0_149/"
                "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/"
                "v18_joint_mano_interval_trajectory_state.json",
    "contact_np": "measurements/contact_nonpenetration/"
                  "keyboard_mano_object_constraint/"
                  "v18_mano_object_constraint_state.json",
    "completion": "measurements/geometry_completion/compact_keyboard_seed42/"
                  "v18_compact_rigid_trellis_completion_report.json",
    "completed_ply": "measurements/geometry_completion/compact_keyboard_seed42/"
                     "keyboard_compact_rigid_completed_mesh_labeled.ply",
    "observed_ply": "measurements/geometry_completion/compact_keyboard_seed42/"
                    "keyboard_observed_depth_surface_labeled.ply",
    "render_report": "renders/v19_published_runtime/v19_published_render_report.json",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load_json(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


def stat_of(dist):
    """Reduce a distribution dict or scalar to a compact JSON-able summary."""
    if isinstance(dist, dict):
        return {k: dist.get(k) for k in ("count", "min", "median", "p90", "p95", "max", "mean")}
    return {"value": dist}


def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def transform_points_to_canonical(world_pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map world-frame points into the completed-mesh canonical frame.

    The pose row stores R = rotation_world_from_completed_canonical and
    t = translation_world_m, i.e. x_world = R @ x_canonical + t.
    Therefore x_canonical = R^T @ (x_world - t).
    """
    return (world_pts - t) @ R.T


def closest_point_unsigned(mesh, points: np.ndarray):
    """Unsigned closest-surface distance for each point (metres)."""
    if len(points) == 0:
        return np.empty((0,))
    (closest, dist, _tid) = trimesh.proximity.closest_point(mesh, points)
    return dist.astype(float)


def signed_distance_best_effort(mesh, points: np.ndarray):
    """Best-effort signed distance. trimesh uses ray-based winding which is only
    reliable on watertight meshes; for non-watertight meshes the sign is
    unreliable, so we return a confidence flag alongside the signed values.
    """
    if len(points) == 0:
        return np.empty((0,)), False
    try:
        sd = trimesh.proximity.signed_distance(mesh, points)
        return sd.astype(float), bool(mesh.is_watertight)
    except Exception:  # pragma: no cover - defensive
        return np.empty((0,)), False


def temporal_iou_sets(set_list):
    """IoU of consecutive non-empty vertex-id sets; returns list of (i, i+1, iou)."""
    out = []
    for i in range(len(set_list) - 1):
        a, b = set_list[i], set_list[i + 1]
        if not a or not b:
            continue
        u = a | b
        iou = len(a & b) / len(u) if u else 0.0
        out.append((i, i + 1, iou))
    return out


# --------------------------------------------------------------------------- #
# main reconciliation
# --------------------------------------------------------------------------- #

def reconcile(run_root: str, frame_lo: int, frame_hi: int, hand_side: str,
              out_dir: str):
    run_root = os.path.abspath(run_root)
    paths = {k: os.path.join(run_root, v) for k, v in REL.items()}

    pose_graph = load_json(paths["pose_graph"])
    interval = load_json(paths["interval"])
    contact_np = load_json(paths["contact_np"])
    completion = load_json(paths["completion"])
    render_report = load_json(paths["render_report"])

    # ---- meshes (canonical frame) -----------------------------------------
    completed_mesh = trimesh.load(paths["completed_ply"], process=False)
    observed_mesh = trimesh.load(paths["observed_ply"], process=False)

    completed_extents = [float(x) for x in completed_mesh.bounding_box.extents]
    observed_extents = [float(x) for x in observed_mesh.bounding_box.extents]
    # sorted-descending so the largest dimension is first regardless of axis
    completed_extents_sorted = sorted(completed_extents, reverse=True)
    observed_extents_sorted = sorted(observed_extents, reverse=True)
    prior_sorted = sorted(KEYBOARD_PRIOR_EXTENTS_M, reverse=True)
    completed_extent_ratio = [c / p for c, p in zip(completed_extents_sorted, prior_sorted)]

    # ---- completion / geometry epoch provenance ---------------------------
    face_label_counts = completion.get("face_label_counts", {})
    mesh_counts = completion.get("mesh_counts", {})
    completed_face_counts = face_label_counts.get("completed_mesh", {})
    completed_total = sum(int(v) for v in completed_face_counts.values()) if completed_face_counts else 0
    face_provenance_summary = {}
    if completed_total > 0:
        for label, cnt in completed_face_counts.items():
            face_provenance_summary[label] = {
                "count": int(cnt),
                "fraction": round(int(cnt) / completed_total, 4),
            }
    observed_depth_faces = int(completed_face_counts.get("observed_depth_surface", 0))
    unsupported_faces = int(completed_face_counts.get("unsupported_uncertain", 0))
    trellis_faces = int(completed_face_counts.get("trellis_inferred_hidden_surface", 0))
    observed_fraction = observed_depth_faces / completed_total if completed_total else 0.0
    trellis_fraction = trellis_faces / completed_total if completed_total else 0.0

    watertight_flags = {
        "completed_surface_mesh_watertight": completion.get(
            "completed_surface_mesh_watertight",
            bool(getattr(completed_mesh, "is_watertight", False))),
        "sign_mesh_watertight": completion.get("sign_mesh_watertight"),
        "completed_mesh_trimesh_is_watertight": bool(getattr(completed_mesh, "is_watertight", False)),
        "observed_mesh_trimesh_is_watertight": bool(getattr(observed_mesh, "is_watertight", False)),
    }
    free_space_rejection_state = completion.get("free_space_rejection_state")
    free_space_rejected = face_label_counts.get("free_space_rejected", 0)

    # ---- pose rows: index by frame ----------------------------------------
    pose_rows = {r["frame_idx"]: r for r in pose_graph.get("pose_rows", [])}
    optimizer = pose_graph.get("optimizer", {})
    np_target_frames = pose_graph.get("nonpenetration_target_frame_count")
    graph_inert = (
        optimizer.get("nfev") == 1
        and float(optimizer.get("cost", 1.0)) == 0.0
        and np_target_frames in (0, None)
    )

    # ---- interval per-frame states: index by (frame, side) ---------------
    interval_pf = {}
    for f in interval.get("per_frame_states", []):
        interval_pf[(f["frame_idx"], f["hand_side"])] = f
    interval_side = {}
    for iv in interval.get("intervals", []):
        if iv.get("hand_side") == hand_side:
            interval_side = iv

    # ---- contact/NP rows: index by (frame, side) -------------------------
    cn_rows = {}
    for r in contact_np.get("constraint_rows", []):
        cn_rows[(r["frame_idx"], r["hand_side"])] = r
    cn_side_summary = contact_np.get("summary_by_side", {}).get(hand_side, {})

    # ---- render published gap (summary only; per-frame comes from interval)
    render_sides = safe_get(render_report, "metrics", {}).get("sides", {}) if render_report else {}
    render_side_summary = render_sides.get(hand_side, {})
    render_summary_text = safe_get(render_report, "metrics", {}).get("summary_text")

    frames = list(range(frame_lo, frame_hi + 1))

    contact_patch_vertex_sets = []  # parallel to frames, for D-b IoU
    per_frame_rows = []

    for frame_idx in frames:
        ipf = interval_pf.get((frame_idx, hand_side), {})
        cn = cn_rows.get((frame_idx, hand_side), {})
        pose_row = pose_rows.get(frame_idx, {})

        # --- observed-surface penetration (interval solver, published) -----
        obs_pen = ipf.get("full_observed_surface_penetration_after_solver_m")
        obs_pen_initial = ipf.get("initial_observed_surface_penetration_m")
        # contact patch gap (this is what the render republishes as the gap)
        cp_gap = ipf.get("contact_patch_final_normal_gap_m")
        cp_vertex_ids = ipf.get("contact_patch_vertex_ids", []) or []
        cp_vertex_set = set(int(v) for v in cp_vertex_ids) if cp_vertex_ids else set()
        contact_patch_vertex_sets.append(cp_vertex_set)

        hand_verts_world = ipf.get("optimized_vertices_world_sample_m")
        hand_sample_ids = ipf.get("optimized_vertices_sample_ids", [])

        # --- contact/NP completed-mesh fields ------------------------------
        cn_signed_candidates = cn.get("signed_query_candidate_vertex_count")
        cn_penetrating = cn.get("penetrating_vertex_count")
        cn_penetration_depth = cn.get("penetration_depth_m")
        cn_signed_distance = cn.get("signed_distance_m")
        cn_nearest_unsigned = cn.get("nearest_surface_unsigned_m")
        cn_sign_wt = cn.get("sign_mesh_watertight")
        cn_near_surface = cn.get("near_surface_vertex_count")
        cn_aabb_cand = cn.get("surface_aabb_candidate_vertex_count")
        cn_hand_vertex_count = cn.get("hand_vertex_count")
        cn_reason = cn.get("reason")
        cn_sign_mesh_path = cn.get("sign_mesh_path")
        cn_surface_mesh_path = cn.get("surface_mesh_path")

        # --- pose provenance -----------------------------------------------
        tpg = pose_row.get("temporal_pose_graph", {})
        pose_provenance = {
            "pose_source": tpg.get("pose_source") or pose_row.get("status"),
            "pose_measurement_status": pose_row.get("pose_measurement_status"),
            "direct_visible_measurement": tpg.get("direct_visible_measurement"),
            "gap_frames": tpg.get("gap_frames"),
            "initial_pose_source": pose_row.get("initial_pose_source"),
            "visible_sample_count": pose_row.get("visible_sample_count"),
            "rotation_delta_rotvec_rad": tpg.get("rotation_delta_rotvec_rad"),
            "translation_delta_world_m": tpg.get("translation_delta_world_m"),
            "nonpenetration_target_world_m": tpg.get("nonpenetration_target_world_m"),
            "nonpenetration_weight": tpg.get("nonpenetration_weight"),
        }

        # --- D-a: recompute distances for the same hand-vertex set ---------
        recompute = {
            "hand_vertex_count_used": 0,
            "observed_surface_unsigned_m_recomputed": None,
            "completed_mesh_unsigned_m_recomputed": None,
            "completed_mesh_signed_m_best_effort": None,
            "completed_mesh_signed_reliable": False,
            "blocked_reason": None,
        }
        disagreement = None
        disagreement_reason = None

        R_np = np.asarray(pose_row.get("rotation_world_from_completed_canonical_matrix"),
                          dtype=float).reshape(3, 3) if pose_row.get(
            "rotation_world_from_completed_canonical_matrix") is not None else None
        t_np = np.asarray(pose_row.get("translation_world_m"), dtype=float).reshape(3) if \
            pose_row.get("translation_world_m") is not None else None

        if (hand_verts_world is not None and len(hand_verts_world) > 0
                and R_np is not None and t_np is not None):
            pts_world = np.asarray(hand_verts_world, dtype=float).reshape(-1, 3)
            pts_canon = transform_points_to_canonical(pts_world, R_np, t_np)
            recompute["hand_vertex_count_used"] = int(len(pts_world))

            d_obs = closest_point_unsigned(observed_mesh, pts_canon)
            d_comp = closest_point_unsigned(completed_mesh, pts_canon)
            recompute["observed_surface_unsigned_m_recomputed"] = _summarize(d_obs)
            recompute["completed_mesh_unsigned_m_recomputed"] = _summarize(d_comp)

            sd, reliable = signed_distance_best_effort(completed_mesh, pts_canon)
            if sd.size:
                recompute["completed_mesh_signed_m_best_effort"] = _summarize(sd)
                recompute["completed_mesh_signed_reliable"] = bool(reliable)

            # D-a geometry-source disagreement:
            #   observed penetration (max, interval solver) vs
            #   recomputed completed-mesh unsigned distance (median).
            obs_pen_max = (obs_pen.get("max") if isinstance(obs_pen, dict) else obs_pen)
            comp_dist_median = (recompute["completed_mesh_unsigned_m_recomputed"] or {}).get("median")
            if obs_pen_max is not None and comp_dist_median is not None:
                # observed penetration is positive-inside; completed unsigned
                # distance is positive-outside. Their signed sum is the
                # cross-source gap that the render/graph cannot reconcile.
                disagreement = float(obs_pen_max) + float(comp_dist_median)
            else:
                disagreement_reason = (
                    "observed penetration max or recomputed completed-mesh "
                    "median unavailable"
                )
        else:
            missing = []
            if hand_verts_world is None or len(hand_verts_world) == 0:
                missing.append("optimized_vertices_world_sample_m")
            if R_np is None:
                missing.append("rotation_world_from_completed_canonical_matrix")
            if t_np is None:
                missing.append("translation_world_m")
            recompute["blocked_reason"] = (
                "D-a/D-b mesh-distance recomputation blocked by missing: "
                + ", ".join(missing)
            )
            disagreement_reason = recompute["blocked_reason"]

        # render-published gap for this frame. The render report only carries a
        # per-side median; the per-frame value it republishes is the interval
        # solver's contact_patch_final_abs_normal_gap_m distribution, so we use
        # the interval per-frame contact_patch_final_normal_gap_m.
        render_gap_m = None
        if isinstance(cp_gap, dict):
            render_gap_m = cp_gap.get("median")
        elif cp_gap is not None:
            render_gap_m = cp_gap

        row = {
            "schema": "ego.hoi/0.1.0/contact_frame_detail",
            "run_root": run_root,
            "frame_index": frame_idx,
            "hand_side": hand_side,
            "object_id": "keyboard",
            # ---- observed-surface (interval) evidence -------------------
            "observed_surface_penetration_m": stat_of(obs_pen),
            "observed_surface_penetration_initial_m": stat_of(obs_pen_initial),
            "observed_surface_penetrating_vertex_count_after_solver": ipf.get(
                "full_observed_supported_penetrating_vertex_count_after_solver"),
            "interval_dense_observed_constraint_count_final": ipf.get(
                "dense_observed_constraint_count_final"),
            "contact_patch_vertex_ids_count": len(cp_vertex_set),
            "contact_patch_vertex_ids_sample": sorted(cp_vertex_set)[:20],
            # ---- completed-mesh / contact-NP fields ---------------------
            "contact_np_hand_vertex_count": cn_hand_vertex_count,
            "contact_np_near_surface_vertex_count": cn_near_surface,
            "contact_np_surface_aabb_candidate_vertex_count": cn_aabb_cand,
            "contact_np_signed_query_candidate_vertex_count": cn_signed_candidates,
            "contact_np_penetrating_vertex_count": cn_penetrating,
            "contact_np_penetration_depth_m": stat_of(cn_penetration_depth),
            "contact_np_signed_distance_m": stat_of(cn_signed_distance),
            "contact_np_nearest_surface_unsigned_m": stat_of(cn_nearest_unsigned),
            "contact_np_sign_mesh_watertight": cn_sign_wt,
            "contact_np_sign_mesh_path": cn_sign_mesh_path,
            "contact_np_surface_mesh_path": cn_surface_mesh_path,
            "contact_np_reason": cn_reason,
            # ---- render published gap -----------------------------------
            "render_gap_m": render_gap_m,
            "render_active_set_closed": render_side_summary.get("active_set_closed"),
            # ---- geometry-source disagreement (D-a) ---------------------
            "geometry_source_disagreement_m": disagreement,
            "geometry_source_disagreement_reason": disagreement_reason,
            "recomputed_distances": recompute,
            # ---- penetrating-set coherence (D-b) ------------------------
            "penetrating_vertex_coherence": {
                "has_vertex_ids": bool(cp_vertex_set),
                "vertex_ids_are_mano_sample_ids": True,
                "note": "temporal IoU computed across the whole window in the "
                        "summary; per-frame set sizes are in "
                        "contact_patch_vertex_ids_count.",
            },
            # ---- pose provenance ----------------------------------------
            "pose_provenance": pose_provenance,
            # ---- face provenance / watertight / epoch -------------------
            "face_provenance_summary": face_provenance_summary,
            "watertight_flags": watertight_flags,
            "free_space_rejection_state": free_space_rejection_state,
            "free_space_rejected": free_space_rejected,
        }

        # ---- decision route ----------------------------------------------
        row["decision_route"] = _decision_route(row, graph_inert)
        per_frame_rows.append(row)

    # ---- D-b: temporal coherence across the window -----------------------
    iou_pairs = temporal_iou_sets(contact_patch_vertex_sets)
    mean_iou = float(np.mean([p[2] for p in iou_pairs])) if iou_pairs else None
    nonzero_sets = sum(1 for s in contact_patch_vertex_sets if s)
    # spatial compactness proxy: do the vertex ids cluster on fingertips/palm?
    # MANO fingertip ids are small (<~200); palm/thumb ids vary. Report id range.
    all_ids = sorted(set().union(*contact_patch_vertex_sets)) if any(contact_patch_vertex_sets) else []
    coherence_summary = {
        "frames_in_window": len(frames),
        "frames_with_contact_patch_vertex_ids": nonzero_sets,
        "temporal_iou_pairs_evaluated": len(iou_pairs),
        "temporal_iou_mean": mean_iou,
        "temporal_iou_per_pair": [
            {"i_frame": frames[i], "j_frame": frames[j], "iou": round(iou, 4)}
            for i, j, iou in iou_pairs
        ],
        "penetrating_vertex_id_min": all_ids[0] if all_ids else None,
        "penetrating_vertex_id_max": all_ids[-1] if all_ids else None,
        "penetrating_vertex_id_union_size": len(all_ids),
        "coherence_interpretation": (
            "compact_persistent_patch" if (mean_iou is not None and mean_iou >= 0.3 and nonzero_sets >= 3)
            else ("partially_persistent" if mean_iou is not None and mean_iou > 0.0
                  else "scattered_or_absent")
        ),
    }

    # ---- D-d: extent / free-space plausibility ---------------------------
    extent_plausibility = {
        "completed_mesh_aabb_extents_m": completed_extents,
        "completed_mesh_aabb_extents_sorted_m": completed_extents_sorted,
        "observed_mesh_aabb_extents_m": observed_extents,
        "observed_mesh_aabb_extents_sorted_m": observed_extents_sorted,
        "keyboard_prior_extents_sorted_m": prior_sorted,
        "keyboard_prior_source": KEYBOARD_PRIOR_SOURCE,
        "completed_to_prior_extent_ratio": [round(r, 3) for r in completed_extent_ratio],
        "observed_to_prior_extent_ratio": [
            round(o / p, 3) for o, p in zip(observed_extents_sorted, prior_sorted)
        ],
        "free_space_rejection_state": free_space_rejection_state,
        "free_space_rejected": free_space_rejected,
        "plausibility_interpretation": _extent_interpretation(completed_extents_sorted, prior_sorted),
    }

    # ---- window-level aggregates -----------------------------------------
    obs_pen_max_vals = [
        (r["observed_surface_penetration_m"] or {}).get("max") for r in per_frame_rows
    ]
    obs_pen_max_vals = [v for v in obs_pen_max_vals if v is not None]
    comp_dist_med_vals = [
        (r["recomputed_distances"].get("completed_mesh_unsigned_m_recomputed") or {}).get("median")
        for r in per_frame_rows
    ]
    comp_dist_med_vals = [v for v in comp_dist_med_vals if v is not None]
    disagreement_vals = [r["geometry_source_disagreement_m"] for r in per_frame_rows
                         if r["geometry_source_disagreement_m"] is not None]

    window_summary = {
        "schema": "ego.hoi/0.1.0/graph_solutions",
        "run_root": run_root,
        "window": [frame_lo, frame_hi],
        "hand_side": hand_side,
        "object_id": "keyboard",
        "n_frames": len(frames),
        # triple-valued contact contradiction
        "interval_observed_surface_penetration_max_m_window": _summarize(np.array(obs_pen_max_vals)) if obs_pen_max_vals else None,
        "completed_mesh_unsigned_distance_median_m_window": _summarize(np.array(comp_dist_med_vals)) if comp_dist_med_vals else None,
        "render_published_gap_median_m": render_side_summary.get("contact_patch_final_abs_normal_gap_m_median"),
        "render_published_active_set_closed": render_side_summary.get("active_set_closed"),
        "render_published_summary_text": render_summary_text,
        "contact_np_signed_query_candidate_vertex_count_window_sum": sum(
            (r["contact_np_signed_query_candidate_vertex_count"] or 0) for r in per_frame_rows),
        "contact_np_penetrating_vertex_count_window_sum": sum(
            (r["contact_np_penetrating_vertex_count"] or 0) for r in per_frame_rows),
        "contact_np_sign_mesh_watertight_any": any(
            r["contact_np_sign_mesh_watertight"] for r in per_frame_rows
            if r["contact_np_sign_mesh_watertight"] is not None),
        "geometry_source_disagreement_m_window": _summarize(np.array(disagreement_vals)) if disagreement_vals else None,
        # graph liveness
        "pose_graph_optimizer": optimizer,
        "pose_graph_nonpenetration_target_frame_count": np_target_frames,
        "pose_graph_graph_inert": graph_inert,
        "pose_graph_status": pose_graph.get("status"),
        "pose_graph_annotation_ready": pose_graph.get("annotation_ready"),
        # geometry epoch
        "geometry_epoch": {
            "object_id": "keyboard",
            "face_label_counts_completed_mesh": completed_face_counts,
            "mesh_counts": mesh_counts,
            "observed_depth_face_fraction": round(observed_fraction, 4),
            "trellis_inferred_face_fraction": round(trellis_fraction, 4),
            "watertight_flags": watertight_flags,
            "free_space_rejection_state": free_space_rejection_state,
            "extent_plausibility": extent_plausibility,
        },
        # coherence
        "penetrating_vertex_coherence": coherence_summary,
        # contact/NP side summary
        "contact_np_side_summary": cn_side_summary,
        # interval side summary (key fields)
        "interval_side_key": {
            "corrected_frame_count": interval_side.get("corrected_frame_count"),
            "initial_observed_surface_penetration_max_m": interval_side.get(
                "initial_observed_surface_penetration_max_m"),
            "full_observed_surface_penetration_after_solver_max_m": interval_side.get(
                "full_observed_surface_penetration_after_solver_max_m"),
            "translation_delta_norm_m": interval_side.get("translation_delta_norm_m"),
            "object_translation_delta_norm_m": interval_side.get("object_translation_delta_norm_m"),
            "contact_patch_final_abs_normal_gap_m": interval_side.get(
                "contact_patch_final_abs_normal_gap_m"),
        },
        "mechanism_decision": _mechanism_decision(
            per_frame_rows, coherence_summary, extent_plausibility, graph_inert),
    }

    # ---- write outputs ----------------------------------------------------
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ndjson_path = os.path.join(out_dir, "contact_frame_detail.ndjson")
    with open(ndjson_path, "w") as fh:
        for r in per_frame_rows:
            fh.write(json.dumps(r) + "\n")
    summary_path = os.path.join(out_dir, "geometry_reconciliation_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(window_summary, fh, indent=2)
    # a compact human-readable table
    table_path = os.path.join(out_dir, "contact_frame_detail_table.tsv")
    with open(table_path, "w") as fh:
        fh.write("\t".join([
            "frame", "pose_source", "obs_pen_max_m", "obs_pen_count",
            "completed_unsigned_med_m", "completed_signed_med_m",
            "cn_signed_candidates", "cn_penetrating", "cn_nearest_unsigned_med_m",
            "render_gap_m", "disagreement_m", "cp_vertex_ids_count",
            "cp_vertex_id_min", "cp_vertex_id_max",
            "cn_sign_watertight", "decision_route\n"]))
        for r in per_frame_rows:
            rd = r["recomputed_distances"]
            comp_u = (rd.get("completed_mesh_unsigned_m_recomputed") or {})
            comp_s = (rd.get("completed_mesh_signed_m_best_effort") or {})
            op = r["observed_surface_penetration_m"] or {}
            nu = r["contact_np_nearest_surface_unsigned_m"] or {}
            ids = r["contact_patch_vertex_ids_sample"]
            coh = r["penetrating_vertex_coherence"]
            fh.write("\t".join([
                str(r["frame_index"]),
                str((r["pose_provenance"] or {}).get("pose_source")),
                _fmt(op.get("max")),
                str(op.get("count")),
                _fmt(comp_u.get("median")),
                _fmt(comp_s.get("median")),
                str(r["contact_np_signed_query_candidate_vertex_count"]),
                str(r["contact_np_penetrating_vertex_count"]),
                _fmt(nu.get("median")),
                _fmt(r["render_gap_m"]),
                _fmt(r["geometry_source_disagreement_m"]),
                str(r["contact_patch_vertex_ids_count"]),
                str(min(ids)) if ids else "",
                str(max(ids)) if ids else "",
                str(r["contact_np_sign_mesh_watertight"]),
                r["decision_route"],
            ]) + "\n")

    return {
        "ndjson": ndjson_path,
        "summary": summary_path,
        "table": table_path,
        "window_summary": window_summary,
    }


def _summarize(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return None
    qs = np.quantile(arr, [0.5, 0.9, 0.95]) if arr.size >= 2 else [float(np.median(arr))] * 3
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "median": round(float(qs[0]), 6),
        "p90": round(float(qs[1]), 6),
        "p95": round(float(qs[2]), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
    }


def _fmt(v):
    if v is None:
        return ""
    try:
        return f"{float(v):.5f}"
    except (TypeError, ValueError):
        return str(v)


def _decision_route(row, graph_inert):
    """Per-frame decision route: M1/M5 coherent observed-surface stranded
    across solvers vs M2 geometry contamination vs unreconciled-pose."""
    obs_pen = row.get("observed_surface_penetration_m") or {}
    obs_pen_max = obs_pen.get("max")
    cn_signed_cand = row.get("contact_np_signed_query_candidate_vertex_count") or 0
    cn_penetrating = row.get("contact_np_penetrating_vertex_count") or 0
    sign_wt = row.get("contact_np_sign_mesh_watertight")
    has_obs_pen = obs_pen_max is not None and obs_pen_max > 0.01
    cp_ids = row.get("contact_patch_vertex_ids_count", 0) or 0

    if has_obs_pen and cn_signed_cand == 0 and cn_penetrating == 0:
        # observed penetration stranded while contact/NP cannot sign.
        # if the completed-mesh extent is inflated (contamination), still route
        # here but flag M2 as a co-condition.
        if sign_wt is False:
            return "M1_M5_cross_solver_geometry_decoupled_sign_mesh_disabled"
        return "M5_cross_solver_geometry_decoupled"
    if not has_obs_pen and cp_ids == 0:
        return "no_observed_surface_penetration_this_frame"
    return "unresolved"


def _extent_interpretation(completed_sorted, prior_sorted):
    ratios = [c / p for c, p in zip(completed_sorted, prior_sorted) if p > 0]
    if not ratios:
        return "unknown"
    over = [r for r in ratios if r > 1.5]
    if len(over) >= 1 and max(ratios) > 4.0:
        return "completed_mesh_inflated_relative_to_keyboard_prior_M2_geometry_contamination"
    if len(over) >= 1:
        return "completed_mesh_larger_than_prior_partial_M2"
    return "completed_mesh_extent_consistent_with_prior"


def _mechanism_decision(rows, coherence, extent, graph_inert):
    obs_pen_max_vals = [
        (r["observed_surface_penetration_m"] or {}).get("max") for r in rows
    ]
    obs_pen_max_vals = [v for v in obs_pen_max_vals if v is not None]
    frames_with_pen = sum(1 for v in obs_pen_max_vals if v and v > 0.01)
    # per-frame route counts (majority route is what matters, not unanimity)
    route_counts = {}
    for r in rows:
        route_counts[r["decision_route"]] = route_counts.get(r["decision_route"], 0) + 1
    m1m5_frames = sum(v for k, v in route_counts.items()
                      if k.startswith("M1_M5") or k.startswith("M5"))
    coh_interp = coherence.get("coherence_interpretation")
    extent_interp = extent.get("plausibility_interpretation", "")
    contaminated = "contamination" in extent_interp

    decision = []
    if m1m5_frames > 0:
        decision.append(
            "M1/M5_confirmed on %d/%d frames: observed-surface penetration is "
            "stranded in the interval solver while contact/NP emits zero "
            "penetrating vertices for the same frames (non-watertight sign mesh "
            "disables the signed query) -> cross-solver geometry-source "
            "decoupling, not measurement absence. (per-frame route counts: %s)"
            % (m1m5_frames, len(rows), route_counts))
    if contaminated:
        decision.append(
            "M2_co-condition: completed-mesh extent is inflated relative to the "
            "keyboard prior (" + extent_interp + "); geometry epoch contamination "
            "must be repaired before the completed mesh is trusted as a contact "
            "body, but the observed-surface penetration lives on a different "
            "geometry source so M1/M5 stands as the proximate route.")
    if graph_inert:
        decision.append(
            "M4_confirmed: pose graph inert (nfev=1, cost=0, "
            "nonpenetration_target_frame_count=0); no live contact/NP factor was "
            "coupled into the graph even though a penetration family has support.")
    if coh_interp == "compact_persistent_patch":
        decision.append(
            "D-b: penetrating vertex set is compact/persistent (IoU mean >= 0.3) "
            "-> supports real fingertip/palm contact rather than scattered leakage.")
    elif coh_interp in ("partially_persistent", "scattered_or_absent"):
        decision.append(
            "D-b: penetrating vertex set is " + coh_interp +
            " -> observed-surface signal is intermittent; does not by itself "
            "refute M1/M5 but weakens the strength of the stranded evidence.")
    if not decision:
        decision.append("no_decisive_signal")
    return decision


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--frame-lo", type=int, default=FRAME_LO)
    ap.add_argument("--frame-hi", type=int, default=FRAME_HI)
    ap.add_argument("--hand-side", default=HAND_SIDE, choices=["left", "right"])
    ap.add_argument("--out-dir", default="/tmp/clip001850_geometry_reconciliation")
    args = ap.parse_args(argv)
    res = reconcile(args.run_root, args.frame_lo, args.frame_hi,
                    args.hand_side, args.out_dir)
    print("NDJSON   :", res["ndjson"])
    print("Summary  :", res["summary"])
    print("TSV table:", res["table"])
    ws = res["window_summary"]
    print()
    print("=== window summary (key) ===")
    print(json.dumps({
        "interval_observed_surface_penetration_max_m_window": ws.get("interval_observed_surface_penetration_max_m_window"),
        "completed_mesh_unsigned_distance_median_m_window": ws.get("completed_mesh_unsigned_distance_median_m_window"),
        "render_published_gap_median_m": ws.get("render_published_gap_median_m"),
        "contact_np_signed_query_candidate_vertex_count_window_sum": ws.get("contact_np_signed_query_candidate_vertex_count_window_sum"),
        "contact_np_penetrating_vertex_count_window_sum": ws.get("contact_np_penetrating_vertex_count_window_sum"),
        "pose_graph_graph_inert": ws.get("pose_graph_graph_inert"),
        "geometry_source_disagreement_m_window": ws.get("geometry_source_disagreement_m_window"),
        "penetrating_vertex_coherence": ws.get("penetrating_vertex_coherence"),
        "mechanism_decision": ws.get("mechanism_decision"),
    }, indent=2))


if __name__ == "__main__":
    main()
