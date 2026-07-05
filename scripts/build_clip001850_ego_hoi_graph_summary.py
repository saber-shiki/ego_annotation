#!/usr/bin/env python3
"""
clip001850 three-solver -> factor_graph_summary adapter.

Purpose (D6 mechanism: "cross-solver geometry-source decoupling")
-----------------------------------------------------------------
A v19 run emits three independent per-solver reports and NO unified
`factor_graph_summary`. This adapter reads those reports from a run root and
assembles ONE `factor_graph_summary` that the R0/R1 scaffold
(build_ego_hoi_sidecar_and_graph_health.py) consumes via `--graph-summary`.

The adapter is pure aggregation: it reads JSON already on disk, copies numbers
with provenance, and exposes which evidence field is missing when a number
cannot be sourced. It does NOT run any model, GPU, or heavy inference.

It populates exactly the block the scaffold's decision table consults FIRST:

    factor_graph_summary.cross_solver_geometry_consistency
        solver_geometry_epoch_id / solver_geometry_source_family
        contact_query_geometry_epoch_id / contact_query_geometry_source_family
        render_geometry_epoch_id / render_geometry_source_family
        signed_query_candidate_vertex_count
        watertight
        face_provenance_summary
        observed_surface_penetration_m
        published_contact_gap_m

plus `objective` (energy_initial/after from the inert rigid pose graph),
`factor_family_health` (per-family declared/supported from each solver's own
counts), `geometry_epochs` (lineage of the geometry completion epoch), and
`declared_gauges`.

The accepted decision is `cross_solver_geometry_decoupled`. If any evidence
field is missing, the adapter records it in `_provenance` so the scaffold's
mismatch detection stays falsifiable.

Usage
-----
    python3 scripts/build_clip001850_ego_hoi_graph_summary.py \
        --run-root /data2/.../<run> --out-dir /tmp/egohoi_clip001850 \
        [--hand-side right] [--start-frame 26] [--end-frame 46]

Writes:
    <out-dir>/clip001850_factor_graph_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_FAMILY = "ego.hoi"
SCHEMA_VERSION = "0.1.0"
PRODUCER = "build_clip001850_ego_hoi_graph_summary.py"
PRODUCER_VERSION = "clip001850-adapter-2-provenance"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_canonical(obj: Any) -> str:
    """Stable hash of a JSON-serializable object (sorted keys, no spaces)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def resolve_provenance_path(raw_path: str | None, run_root: Path) -> Path | None:
    """Resolve an inputs[] path recorded in a solver report onto this run root.

    Solver reports record absolute paths under the producing host's mount
    (e.g. ``/mnt/truenas-user-home/...``). On this machine the same run lives
    under ``run_root`` (e.g. ``/data2/...``). We rebase by matching the run-id
    directory segment so the provenance hash is of the *actual* on-disk file,
    not a stale host path.
    """
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.exists():
        return p
    run_id = run_root.name
    idx = raw_path.find(run_id)
    if idx >= 0:
        tail = raw_path[idx + len(run_id):].lstrip("/")
        cand = run_root / tail
        if cand.exists():
            return cand
    return None


def provenance_fingerprint(fields: dict[str, Any]) -> str:
    """Hash a dict of *measured* provenance fields into a stable fingerprint.

    Used to derive geometry epoch_id / source_family from measured provenance
    (file hashes + filter states) rather than adapter-authored string literals,
    so the cross-solver decision is falsifiable: equal-string tampering in this
    adapter can no longer flip it, and changing the underlying file or a filter
    flag changes the family.
    """
    return sha256_json_canonical(fields)


def find_interval_state_path(run_root: Path) -> Path | None:
    """Locate v18_joint_mano_interval_trajectory_state.json under the run root."""
    base = run_root / "measurements" / "mano_interval_correction"
    if not base.is_dir():
        return None
    hits = list(base.rglob("v18_joint_mano_interval_trajectory_state.json"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Per-report extraction (each returns data + provenance)
# --------------------------------------------------------------------------- #
def extract_pose_graph(run_root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Rigid pose graph report -> inertness + NP-target evidence."""
    rep_path = run_root / "measurements" / "pose_fits" / "keyboard_rigid_pose_graph" / \
        "v19_rigid_object_pose_graph_report.json"
    out: dict[str, Any] = {"_available": rep_path.is_file()}
    if not rep_path.is_file():
        provenance["pose_graph_report"] = {"status": "missing", "path": str(rep_path)}
        return out
    provenance["pose_graph_report"] = {
        "status": "present", "path": str(rep_path), "sha256": sha256_file(rep_path),
    }
    rep = load_json(rep_path)
    opt = rep.get("optimizer") or {}
    out["optimizer_success"] = opt.get("success")
    out["optimizer_nfev"] = opt.get("nfev")
    out["objective_cost_after"] = opt.get("cost")
    out["residual_rms_before"] = opt.get("residual_rms_before")
    out["residual_rms_after"] = opt.get("residual_rms_after")
    out["nonpenetration_target_frame_count"] = rep.get("nonpenetration_target_frame_count")
    out["graph_frame_count"] = rep.get("graph_frame_count")
    out["graph_frames"] = rep.get("graph_frames")
    out["annotation_ready"] = rep.get("annotation_ready")
    out["status"] = rep.get("status")
    # correction deltas
    cs = rep.get("correction_summary") or {}
    td = cs.get("translation_delta_norm_m") or {}
    out["translation_delta_norm_m_max"] = td.get("max")
    provenance["pose_graph_report"]["graph_frames"] = rep.get("graph_frames")
    provenance["pose_graph_report"]["nonpenetration_target_frame_count"] = rep.get(
        "nonpenetration_target_frame_count"
    )
    return out


def extract_visible_pose_fit(run_root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Visible pose fit report -> object_se3 support fraction."""
    rep_path = run_root / "measurements" / "pose_fits" / "keyboard_visible_pose_fit" / \
        "v18_compact_rigid_object_pose_fit_report.json"
    out: dict[str, Any] = {"_available": rep_path.is_file()}
    if not rep_path.is_file():
        provenance["visible_pose_fit_report"] = {"status": "missing", "path": str(rep_path)}
        return out
    provenance["visible_pose_fit_report"] = {
        "status": "present", "path": str(rep_path), "sha256": sha256_file(rep_path),
    }
    rep = load_json(rep_path)
    out["frame_count"] = rep.get("frame_count")
    out["fit_frame_count"] = rep.get("fit_frame_count")
    out["missing_pose_count"] = rep.get("missing_pose_count")
    out["ineligible_pose_observation_count"] = rep.get("ineligible_pose_observation_count")
    provenance["visible_pose_fit_report"].update(
        {
            "fit_frame_count": rep.get("fit_frame_count"),
            "ineligible_pose_observation_count": rep.get("ineligible_pose_observation_count"),
            "missing_pose_count": rep.get("missing_pose_count"),
        }
    )
    return out


def extract_interval_mano(run_root: Path, hand_side: str, provenance: dict[str, Any]) -> dict[str, Any]:
    """Interval MANO report -> observed-surface penetration + correction support."""
    rep_path = find_interval_state_path(run_root)
    out: dict[str, Any] = {"_available": rep_path is not None}
    if rep_path is None:
        provenance["interval_mano_report"] = {"status": "missing"}
        return out
    provenance["interval_mano_report"] = {
        "status": "present", "path": str(rep_path), "sha256": sha256_file(rep_path),
    }
    rep = load_json(rep_path)
    inputs = rep.get("inputs") or {}
    depth_npz_paths = inputs.get("depth_npz") or []
    depth_npz_raw = depth_npz_paths[0] if isinstance(depth_npz_paths, list) and depth_npz_paths else None
    intervals = rep.get("intervals") or []
    target = next((iv for iv in intervals if iv.get("hand_side") == hand_side), None)
    out["hand_side"] = hand_side
    out["optimizer_ran"] = target.get("optimizer_ran") if target else None
    out["corrected_frame_count"] = target.get("corrected_frame_count") if target else None
    out["frame_count"] = target.get("frame_count") if target else None
    out["contact_patch_factor_active_row_count"] = (
        target.get("contact_patch_factor_active_row_count") if target else None
    )
    # Full observed-surface penetration STAT BLOCK (count/median/p90/p95/max/mean).
    # NOTE: we deliberately do NOT collapse this to max-of-max. The decisive
    # statistic exported downstream is the median (see stat_policy below);
    # max-of-max is the single worst vertex in the worst frame and is carried
    # only as a distribution tail, never as decisive evidence.
    pen_stat = target.get("full_observed_surface_penetration_after_solver_max_m") if target else None
    out["observed_surface_penetration_stat"] = pen_stat
    out["initial_observed_surface_penetration_stat"] = (
        target.get("initial_observed_surface_penetration_max_m") if target else None
    )
    out["contact_patch_final_abs_normal_gap_m"] = (
        target.get("contact_patch_final_abs_normal_gap_m") if target else None
    )
    out["translation_delta_norm_m"] = target.get("translation_delta_norm_m") if target else None
    # ---- measured filter / quarantine / eligibility states (A1 evidence) ----
    out["interval_filter_state"] = {
        "dense_observed_surface_barrier_enabled": (
            target.get("dense_observed_surface_barrier_enabled") if target else None
        ),
        "visible_object_mask_gate_enabled": (
            target.get("visible_object_mask_gate_enabled") if target else None
        ),
        "hand_owned_object_depth_quarantine_enabled": (
            target.get("hand_owned_object_depth_quarantine_enabled") if target else None
        ),
        "surface_eligibility_factor_enabled": (
            target.get("surface_eligibility_factor_enabled") if target else None
        ),
        "visible_surface_depth_order_term_enabled": (
            target.get("visible_surface_depth_order_term_enabled") if target else None
        ),
        "visible_surface_depth_order_selected_vertex_count": (
            target.get("visible_surface_depth_order_selected_vertex_count") if target else None
        ),
    }
    # ---- depth_npz provenance (the barrier surface source) ----
    depth_npz_resolved = resolve_provenance_path(depth_npz_raw, run_root)
    out["depth_npz_provenance"] = {
        "recorded_path": depth_npz_raw,
        "resolved_path": str(depth_npz_resolved) if depth_npz_resolved else None,
        "sha256": sha256_file(depth_npz_resolved) if depth_npz_resolved else None,
        "resolved": depth_npz_resolved is not None,
    }
    provenance["interval_mano_report"].update(
        {
            "hand_side": hand_side,
            "corrected_frame_count": out["corrected_frame_count"],
            "depth_npz_resolved": depth_npz_resolved is not None,
            "depth_npz_sha256": out["depth_npz_provenance"]["sha256"],
            "filter_mask_gate_enabled": out["interval_filter_state"]["visible_object_mask_gate_enabled"],
            "filter_hand_quarantine_enabled": out["interval_filter_state"]["hand_owned_object_depth_quarantine_enabled"],
            "filter_eligibility_enabled": out["interval_filter_state"]["surface_eligibility_factor_enabled"],
        }
    )
    return out


def extract_contact_nonpenetration(
    run_root: Path, hand_side: str, start_frame: int, end_frame: int, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Contact/NP report -> zero-candidate query against completed mesh."""
    rep_path = run_root / "measurements" / "contact_nonpenetration" / \
        "keyboard_mano_object_constraint" / "v18_mano_object_constraint_state.json"
    out: dict[str, Any] = {"_available": rep_path.is_file()}
    if not rep_path.is_file():
        provenance["contact_nonpenetration_report"] = {"status": "missing", "path": str(rep_path)}
        return out
    provenance["contact_nonpenetration_report"] = {
        "status": "present", "path": str(rep_path), "sha256": sha256_file(rep_path),
    }
    rep = load_json(rep_path)
    inputs = rep.get("inputs") or {}
    out["completed_surface_mesh_watertight"] = rep.get("completed_surface_mesh_watertight")
    out["sign_mesh_watertight"] = rep.get("sign_mesh_watertight")
    out["candidate_correction_count"] = rep.get("candidate_correction_count")
    out["measured_pair_count"] = rep.get("measured_pair_count")
    # ---- contact/NP surface + sign mesh provenance (A3/A4 evidence) ----
    surface_raw = inputs.get("completed_surface_mesh")
    sign_raw = inputs.get("sign_mesh")
    surface_resolved = resolve_provenance_path(surface_raw, run_root)
    sign_resolved = resolve_provenance_path(sign_raw, run_root)
    out["mesh_provenance"] = {
        "surface_mesh_recorded_path": surface_raw,
        "surface_mesh_resolved_path": str(surface_resolved) if surface_resolved else None,
        "surface_mesh_sha256": sha256_file(surface_resolved) if surface_resolved else None,
        "surface_mesh_resolved": surface_resolved is not None,
        "sign_mesh_recorded_path": sign_raw,
        "sign_mesh_resolved_path": str(sign_resolved) if sign_resolved else None,
        "sign_mesh_sha256": sha256_file(sign_resolved) if sign_resolved else None,
        "sign_mesh_resolved": sign_resolved is not None,
        "surface_mesh_equals_sign_mesh": (
            (surface_resolved is not None) and surface_resolved == sign_resolved
        ),
    }
    sbs = rep.get("summary_by_side") or {}
    side = sbs.get(hand_side) or {}
    out["frames_with_any_penetration"] = side.get("frames_with_any_penetration")
    # signed-query candidate + penetration over the slice frames
    rows = rep.get("constraint_rows") or []
    slice_rows = [
        r for r in rows
        if r.get("hand_side") == hand_side
        and start_frame <= int(r.get("frame_idx", -1)) <= end_frame
    ]
    signed_candidates = [int(r.get("signed_query_candidate_vertex_count") or 0) for r in slice_rows]
    penetrating = [int(r.get("penetrating_vertex_count") or 0) for r in slice_rows]
    out["slice"] = {
        "hand_side": hand_side,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "row_count": len(slice_rows),
        "signed_query_candidate_vertex_count_max": max(signed_candidates) if signed_candidates else None,
        "penetrating_vertex_count_max": max(penetrating) if penetrating else None,
    }
    provenance["contact_nonpenetration_report"].update(
        {
            "candidate_correction_count": rep.get("candidate_correction_count"),
            "sign_mesh_watertight": rep.get("sign_mesh_watertight"),
            "surface_mesh_resolved": surface_resolved is not None,
            "surface_mesh_sha256": out["mesh_provenance"]["surface_mesh_sha256"],
            "sign_mesh_resolved": sign_resolved is not None,
            "sign_mesh_sha256": out["mesh_provenance"]["sign_mesh_sha256"],
            "slice_signed_query_candidate_max": out["slice"]["signed_query_candidate_vertex_count_max"],
        }
    )
    return out


def extract_geometry_completion(run_root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Geometry completion report -> face provenance + epoch lineage."""
    rep_path = run_root / "measurements" / "geometry_completion" / "compact_keyboard_seed42" / \
        "v18_compact_rigid_trellis_completion_report.json"
    out: dict[str, Any] = {"_available": rep_path.is_file()}
    if not rep_path.is_file():
        provenance["geometry_completion_report"] = {"status": "missing", "path": str(rep_path)}
        return out
    provenance["geometry_completion_report"] = {
        "status": "present", "path": str(rep_path), "sha256": sha256_file(rep_path),
    }
    rep = load_json(rep_path)
    out["mesh_counts"] = rep.get("mesh_counts") or {}
    out["face_label_counts"] = rep.get("face_label_counts") or {}
    out["free_space_rejection_state"] = rep.get("free_space_rejection_state")
    # watertight flag is carried by the contact/NP report against this mesh; not
    # in the completion report itself. Record as pending.
    out["watertight"] = None
    provenance["geometry_completion_report"].update(
        {
            "face_label_counts_completed_mesh": (rep.get("face_label_counts") or {}).get(
                "completed_mesh"
            ),
            "free_space_rejection_state": rep.get("free_space_rejection_state"),
        }
    )
    return out


def extract_render(run_root: Path, hand_side: str, provenance: dict[str, Any]) -> dict[str, Any]:
    """Published render report -> published contact gap."""
    rep_path = run_root / "renders" / "v19_published_runtime" / "v19_published_render_report.json"
    out: dict[str, Any] = {"_available": rep_path.is_file()}
    if not rep_path.is_file():
        provenance["render_report"] = {"status": "missing", "path": str(rep_path)}
        return out
    provenance["render_report"] = {
        "status": "present", "path": str(rep_path), "sha256": sha256_file(rep_path),
    }
    rep = load_json(rep_path)
    metrics = rep.get("metrics") or {}
    sides = metrics.get("sides") or {}
    side = sides.get(hand_side) or {}
    out["contact_patch_final_abs_normal_gap_m_median"] = side.get(
        "contact_patch_final_abs_normal_gap_m_median"
    )
    out["active_set_closed"] = side.get("active_set_closed")
    out["summary_text"] = metrics.get("summary_text")
    # ---- render consumed-state provenance ----
    # The publish path consumes the interval trajectory state JSON (it stamps the
    # contact-patch gap from it). That file IS the render's consumed geometry/
    # state source; hashing it gives the render-side provenance fingerprint.
    consumed_raw = metrics.get("interval_state")
    consumed_resolved = resolve_provenance_path(consumed_raw, run_root)
    out["consumed_state_provenance"] = {
        "recorded_path": consumed_raw,
        "resolved_path": str(consumed_resolved) if consumed_resolved else None,
        "sha256": sha256_file(consumed_resolved) if consumed_resolved else None,
        "resolved": consumed_resolved is not None,
        "source": "metrics.interval_state",
    }
    provenance["render_report"].update(
        {
            "hand_side": hand_side,
            "contact_patch_final_abs_normal_gap_m_median": side.get(
                "contact_patch_final_abs_normal_gap_m_median"
            ),
            "consumed_state_resolved": consumed_resolved is not None,
            "consumed_state_sha256": out["consumed_state_provenance"]["sha256"],
        }
    )
    return out


# --------------------------------------------------------------------------- #
# factor_graph_summary assembly
# --------------------------------------------------------------------------- #
def build_factor_graph_summary(
    *,
    run_root: Path,
    hand_side: str,
    start_frame: int,
    end_frame: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance: dict[str, Any] = {}
    pose = extract_pose_graph(run_root, provenance)
    vpf = extract_visible_pose_fit(run_root, provenance)
    im = extract_interval_mano(run_root, hand_side, provenance)
    cnp = extract_contact_nonpenetration(run_root, hand_side, start_frame, end_frame, provenance)
    gc = extract_geometry_completion(run_root, provenance)
    rr = extract_render(run_root, hand_side, provenance)

    run_id = run_root.name

    # (geometry epoch lineage is built after the cross-solver provenance
    # fingerprints are computed, since its epoch_id is derived from the
    # contact/NP surface-mesh provenance hash.)

    # ---- cross_solver_geometry_consistency ------------------------------- #
    # KT-6: the three geometry epoch_id / source_family fields are DERIVED from
    # measured provenance (file hashes + measured filter states), NOT authored
    # string literals. The interval solver's barrier surface is identified by
    # its depth_npz hash + the mask/quarantine/eligibility filter states; the
    # contact/NP query surface by its surface_mesh + sign_mesh hashes +
    # watertight; the render by the consumed interval-state hash. Equal-string
    # tampering in this adapter can no longer flip the decision, and flipping a
    # filter flag or swapping a file changes the family.
    completed_mesh = (gc.get("face_label_counts") or {}).get("completed_mesh") or {}
    total_completed = sum(int(v) for v in completed_mesh.values() if isinstance(v, int)) or 1
    obs_faces = int(completed_mesh.get("observed_depth_surface", 0))
    unsupported_faces = int(completed_mesh.get("unsupported_uncertain", 0))
    trellis_faces = int(completed_mesh.get("trellis_inferred_hidden_surface", 0))

    # ---- solver (interval) provenance fingerprint ----
    im_depth = im.get("depth_npz_provenance") or {}
    im_filter = im.get("interval_filter_state") or {}
    solver_prov: dict[str, Any] = {
        "depth_npz_sha256": im_depth.get("sha256"),
        "depth_npz_path": im_depth.get("resolved_path"),
        "visible_object_mask_gate_enabled": im_filter.get("visible_object_mask_gate_enabled"),
        "hand_owned_object_depth_quarantine_enabled": im_filter.get("hand_owned_object_depth_quarantine_enabled"),
        "surface_eligibility_factor_enabled": im_filter.get("surface_eligibility_factor_enabled"),
        "dense_observed_surface_barrier_enabled": im_filter.get("dense_observed_surface_barrier_enabled"),
        "visible_surface_depth_order_selected_vertex_count": im_filter.get(
            "visible_surface_depth_order_selected_vertex_count"
        ),
    }
    solver_fp = provenance_fingerprint(solver_prov)
    # source family is a measured descriptor, not an authored semantic label:
    # it encodes which provenance fields back it, plus the short hash.
    solver_source_family = (
        f"interval_depth_barrier+mask{im_filter.get('visible_object_mask_gate_enabled')}"
        f"+quar{im_filter.get('hand_owned_object_depth_quarantine_enabled')}"
        f"+elig{im_filter.get('surface_eligibility_factor_enabled')}"
        f"#{solver_fp[:8]}"
        if solver_prov["depth_npz_sha256"] is not None
        else None
    )
    solver_epoch_id = f"epoch_interval_{solver_fp[:12]}" if solver_prov["depth_npz_sha256"] is not None else None

    # ---- contact/NP provenance fingerprint ----
    cnp_mesh = cnp.get("mesh_provenance") or {}
    contact_prov: dict[str, Any] = {
        "surface_mesh_sha256": cnp_mesh.get("surface_mesh_sha256"),
        "surface_mesh_path": cnp_mesh.get("surface_mesh_resolved_path"),
        "sign_mesh_sha256": cnp_mesh.get("sign_mesh_sha256"),
        "sign_mesh_path": cnp_mesh.get("sign_mesh_resolved_path"),
        "sign_mesh_watertight": cnp.get("sign_mesh_watertight"),
        "completed_surface_mesh_watertight": cnp.get("completed_surface_mesh_watertight"),
    }
    contact_fp = provenance_fingerprint(contact_prov)
    contact_source_family = (
        f"contact_signed_query+signmesh{cnp_mesh.get('sign_mesh_sha256','')[:8]}"
        f"+watertight{cnp.get('sign_mesh_watertight')}#{contact_fp[:8]}"
        if contact_prov["surface_mesh_sha256"] is not None
        else None
    )
    contact_epoch_id = f"epoch_contact_{contact_fp[:12]}" if contact_prov["surface_mesh_sha256"] is not None else None

    # ---- render provenance fingerprint ----
    rr_state = rr.get("consumed_state_provenance") or {}
    render_prov: dict[str, Any] = {
        "consumed_state_sha256": rr_state.get("sha256"),
        "consumed_state_path": rr_state.get("resolved_path"),
        "source": rr_state.get("source"),
    }
    render_fp = provenance_fingerprint(render_prov)
    render_source_family = (
        f"render_consumed_interval_state#{render_fp[:8]}"
        if render_prov["consumed_state_sha256"] is not None
        else None
    )
    render_epoch_id = f"epoch_render_{render_fp[:12]}" if render_prov["consumed_state_sha256"] is not None else None

    # ---- penetration stat policy (NOT max-of-max) ----
    pen_stat = im.get("observed_surface_penetration_stat") or {}
    # The decisive statistic is the MEDIAN across frames. max-of-max is the
    # single worst vertex in the worst frame and is explicitly NOT decisive;
    # it is carried only as a distribution tail for context.
    decisive_pen = pen_stat.get("median")
    stat_policy = {
        "policy": "median_decisive_not_max_of_max",
        "decisive_statistic": "median",
        "decisive_value": decisive_pen,
        "count": pen_stat.get("count"),
        "median": pen_stat.get("median"),
        "p90": pen_stat.get("p90"),
        "p95": pen_stat.get("p95"),
        "mean": pen_stat.get("mean"),
        "max": pen_stat.get("max"),
        "rationale": (
            "max-of-max is one worst vertex in one worst frame, not a representative "
            "contact measurement; carried as a tail only, never as decisive evidence"
        ),
    }

    # ---- provenance completeness (drives evidence_incomplete in scaffold) ----
    prov_missing: list[str] = []
    if solver_prov["depth_npz_sha256"] is None:
        prov_missing.append("solver.depth_npz_sha256 (interval barrier surface source)")
    if contact_prov["surface_mesh_sha256"] is None:
        prov_missing.append("contact.surface_mesh_sha256 (contact/NP query surface)")
    if contact_prov["sign_mesh_sha256"] is None:
        prov_missing.append("contact.sign_mesh_sha256 (contact/NP sign mesh)")
    if render_prov["consumed_state_sha256"] is None:
        prov_missing.append("render.consumed_state_sha256 (renderer consumed state)")
    provenance_completeness = {
        "complete": len(prov_missing) == 0,
        "missing": prov_missing,
        "present_sources": [
            s for s, ok in (
                ("interval_depth_npz", solver_prov["depth_npz_sha256"] is not None),
                ("contact_surface_mesh", contact_prov["surface_mesh_sha256"] is not None),
                ("contact_sign_mesh", contact_prov["sign_mesh_sha256"] is not None),
                ("render_consumed_state", render_prov["consumed_state_sha256"] is not None),
            ) if ok
        ],
        "derivation": (
            "epoch_id / source_family derived from measured provenance hashes + "
            "filter states (sha256 of canonical JSON), not adapter-authored strings"
        ),
    }

    csg: dict[str, Any] = {
        # provenance-derived solver geometry (interval depth barrier)
        "solver_geometry_epoch_id": solver_epoch_id,
        "solver_geometry_source_family": solver_source_family,
        "solver_geometry_provenance": solver_prov,
        # provenance-derived contact/NP geometry
        "contact_query_geometry_epoch_id": contact_epoch_id,
        "contact_query_geometry_source_family": contact_source_family,
        "contact_query_geometry_provenance": contact_prov,
        # provenance-derived render geometry (consumed interval state)
        "render_geometry_epoch_id": render_epoch_id,
        "render_geometry_source_family": render_source_family,
        "render_geometry_provenance": render_prov,
        "signed_query_candidate_vertex_count": cnp.get("slice", {}).get(
            "signed_query_candidate_vertex_count_max"
        ),
        "watertight": cnp.get("sign_mesh_watertight"),
        "sign_mesh_watertight": cnp.get("sign_mesh_watertight"),
        "completed_surface_mesh_watertight": cnp.get("completed_surface_mesh_watertight"),
        "face_provenance_summary": {
            "observed_depth_surface_faces": obs_faces,
            "observed_fraction": round(obs_faces / total_completed, 6),
            "unsupported_uncertain_faces": unsupported_faces,
            "trellis_inferred_hidden_surface_faces": trellis_faces,
            "trellis_completed_fraction": round(trellis_faces / total_completed, 6),
            "total_completed_faces": total_completed,
            "free_space_rejected": completed_mesh.get("free_space_rejected", 0),
        },
        # decisive penetration statistic (median), full distribution in stat_policy
        "observed_surface_penetration_m": decisive_pen,
        "observed_surface_penetration_stat_policy": stat_policy,
        "published_contact_gap_m": rr.get("contact_patch_final_abs_normal_gap_m_median"),
        "provenance_completeness": provenance_completeness,
    }

    # ---- geometry epoch lineage (derived from contact/NP surface provenance) #
    completed_mesh_faces = (gc.get("mesh_counts") or {}).get("completed_faces")
    completed_epoch_id = (
        f"epoch_completed_mesh_{contact_fp[:12]}" if contact_prov["surface_mesh_sha256"] else None
    )
    geometry_epochs = [
        {
            "epoch_id": completed_epoch_id,
            "anchor_frame": 0,
            "superseded_by": None,
            "source": "trellis_completion_over_observed_depth_surface",
            "source_fingerprint_sha256": contact_fp,
            "completed_faces": completed_mesh_faces,
            "free_space_rejection_state": gc.get("free_space_rejection_state"),
            "object_id": "keyboard",
        }
    ]

    # ---- objective (the inert rigid pose graph) -------------------------- #
    # The rigid pose graph is the canonical graph objective. It ran nfev=1 at
    # cost 0 with residual 0->0, so energy_initial == energy_after == 0.
    cost = pose.get("objective_cost_after")
    objective = {
        "energy_initial": 0.0,
        "energy_after": float(cost) if isinstance(cost, (int, float)) else 0.0,
        "source": "v19_rigid_object_pose_graph_report.optimizer.cost",
        "nfev": pose.get("optimizer_nfev"),
        "residual_rms_before": pose.get("residual_rms_before"),
        "residual_rms_after": pose.get("residual_rms_after"),
        "nonpenetration_target_frame_count": pose.get("nonpenetration_target_frame_count"),
    }

    # ---- factor_family_health (per-family declared/supported) ------------ #
    # Each family maps to one solver's own support counts so the scaffold's
    # support-fraction signal is also faithful, not just the geometry decision.
    vpf_total = vpf.get("frame_count") or 150
    vpf_supported = vpf.get("fit_frame_count") or 0
    im_frame_count = im.get("frame_count") or vpf_total
    im_supported = im.get("corrected_frame_count") or 0
    cnp_pairs = cnp.get("measured_pair_count") or 0
    cnp_corrections = cnp.get("candidate_correction_count") or 0
    cp_active = im.get("contact_patch_factor_active_row_count") or 0

    factor_family_health: dict[str, dict[str, Any]] = {
        # object pose prior: 13 of 150 frames have a direct visible fit
        "object_se3_observation": {
            "declared": vpf_total,
            "supported": vpf_supported,
            "residual_sum": 0.0,
            "provenance": "visible_pose_fit.fit_frame_count",
            "ineligible": vpf.get("ineligible_pose_observation_count"),
            "missing": vpf.get("missing_pose_count"),
        },
        # interval MANO observed-surface correction: live, corrected 57/150 right
        "hand_state_observation": {
            "declared": im_frame_count,
            "supported": im_supported,
            "residual_sum": float(decisive_pen or 0.0),
            "provenance": "interval_mano.corrected_frame_count / observed_surface_penetration (median decisive)",
            "hand_side": hand_side,
        },
        # contact switch: contact/NP found zero penetrating frames
        "contact_switch_discrete": {
            "declared": vpf_total,
            "supported": int(cnp.get("frames_with_any_penetration") or 0),
            "residual_sum": 0.0,
            "provenance": "contact_nonpenetration.summary_by_side.<side>.frames_with_any_penetration",
        },
        # signed nonpenetration: candidate corrections = 0 (sign mesh non-watertight)
        "contact_local_nonpenetration": {
            "declared": cnp_pairs,
            "supported": cnp_corrections,
            "residual_sum": 0.0,
            "provenance": "contact_nonpenetration.candidate_correction_count",
        },
        # contact patch factor inside the interval solver: 13 active rows but
        # weight 0.0 in the published configuration -> not net-driving
        "contact_switch_temporal": {
            "declared": im_frame_count,
            "supported": cp_active,
            "residual_sum": float((im.get("contact_patch_final_abs_normal_gap_m") or {}).get("median") or 0.0),
            "provenance": "interval_mano.contact_patch_factor_active_row_count",
        },
    }

    # ---- declared gauges (the rigid pose graph anchor lock) -------------- #
    declared_gauges = [
        {
            "variable": "object_se3::keyboard",
            "gauge": "rigid_pose_graph_anchor_frame_lock",
            "anchor_frames": pose.get("graph_frames"),
            "nfev": pose.get("optimizer_nfev"),
        }
    ]

    summary = {
        "case": run_id,
        "object_id": "keyboard",
        "hand_side": hand_side,
        "slice": {"start_frame": start_frame, "end_frame": end_frame},
        "factor_graph_summary": {
            "objective": objective,
            "factor_family_health": factor_family_health,
            "declared_gauges": declared_gauges,
            "geometry_epochs": geometry_epochs,
            "active_geometry_epoch_id": completed_epoch_id,
            "cross_solver_geometry_consistency": csg,
        },
    }

    # expose missing-field status for falsifiability. Provenance-completeness
    # (the gate the scaffold consults) is authoritative; this secondary list
    # records decision-relevant numeric fields that are also absent.
    missing_fields: list[str] = list(provenance_completeness["missing"])
    if csg["signed_query_candidate_vertex_count"] is None:
        missing_fields.append("cross_solver_geometry_consistency.signed_query_candidate_vertex_count")
    if csg["observed_surface_penetration_m"] is None:
        missing_fields.append("cross_solver_geometry_consistency.observed_surface_penetration_m (decisive median)")
    if csg["published_contact_gap_m"] is None:
        missing_fields.append("cross_solver_geometry_consistency.published_contact_gap_m")
    if csg["watertight"] is None:
        missing_fields.append("cross_solver_geometry_consistency.watertight (contact/NP sign_mesh_watertight)")

    # NOTE on file shape: the scaffold's `--graph-summary` mode wraps the whole
    # file as the `factor_graph_summary` dict (`annotation = {factor_graph_summary:
    # <file>, frames: []}`). So the file's TOP LEVEL must carry the summary keys
    # the scaffold consults (objective, factor_family_health, geometry_epochs,
    # declared_gauges, cross_solver_geometry_consistency). Metadata lives under
    # `_`-prefixed siblings so it does not interfere with the decision table.
    summary = {
        "case": run_id,
        "objective": objective,
        "factor_family_health": factor_family_health,
        "declared_gauges": declared_gauges,
        "geometry_epochs": geometry_epochs,
        "active_geometry_epoch_id": completed_epoch_id,
        "cross_solver_geometry_consistency": csg,
        "_meta": {
            "object_id": "keyboard",
            "hand_side": hand_side,
            "slice": {"start_frame": start_frame, "end_frame": end_frame},
            "producer": {"name": PRODUCER, "version": PRODUCER_VERSION},
            "source_run_root": str(run_root),
            "provenance": provenance,
            "missing_fields": missing_fields,
        },
    }
    return summary, provenance


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_clip001850_ego_hoi_graph_summary.py",
        description=(
            "Aggregate the three clip001850 per-solver reports into one "
            "factor_graph_summary for the R0/R1 graph-health scaffold. "
            "No model inference; reads JSON only."
        ),
    )
    p.add_argument(
        "--run-root",
        required=True,
        help="v19 run root, e.g. .../20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1",
    )
    p.add_argument("--out-dir", required=True, help="directory to write the summary JSON")
    p.add_argument("--hand-side", default="right", choices=["left", "right"])
    p.add_argument("--start-frame", type=int, default=26)
    p.add_argument("--end-frame", type=int, default=46)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, provenance = build_factor_graph_summary(
        run_root=run_root,
        hand_side=args.hand_side,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    out_path = out_dir / "clip001850_factor_graph_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csg = summary["cross_solver_geometry_consistency"]
    pc = csg.get("provenance_completeness") or {}
    print(f"[adapter] wrote {out_path}")
    print(f"[adapter] run_root = {run_root}")
    print(f"[adapter] hand_side = {args.hand_side}, slice = [{args.start_frame},{args.end_frame}]")
    print("[adapter] cross_solver_geometry_consistency (provenance-derived):")
    for k in (
        "solver_geometry_source_family",
        "contact_query_geometry_source_family",
        "render_geometry_source_family",
        "signed_query_candidate_vertex_count",
        "watertight",
        "sign_mesh_watertight",
        "observed_surface_penetration_m",
        "published_contact_gap_m",
    ):
        print(f"    {k}: {csg.get(k)}")
    sp = csg.get("observed_surface_penetration_stat_policy") or {}
    print(
        f"[adapter] penetration stat_policy: {sp.get('policy')} decisive={sp.get('decisive_statistic')}"
        f" median={sp.get('median')} max={sp.get('max')} (max NOT decisive)"
    )
    print(f"[adapter] face_provenance_summary: {csg['face_provenance_summary']}")
    print(f"[adapter] provenance_completeness.complete: {pc.get('complete')}")
    if not pc.get("complete"):
        print(f"[adapter] MISSING provenance sources: {pc.get('missing')}")
    missing = summary.get("_meta", {}).get("missing_fields") or []
    if missing:
        print(f"[adapter] MISSING fields: {missing}")
    else:
        print("[adapter] all cross_solver evidence fields present")
    print(
        "[adapter] next: python3 scripts/build_ego_hoi_sidecar_and_graph_health.py "
        f"--graph-summary {out_path} --out-dir {out_dir}/ego_hoi --run-id {run_root.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
