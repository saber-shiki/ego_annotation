#!/usr/bin/env python3
"""
R0/R1 scaffold: emit an `ego.hoi` 0.1.0 sidecar manifest and a graph_health
table skeleton from a v18-style annotation/graph summary or a synthetic input.

Purpose (D5 mechanism: "factor graph can be inert while looking implemented")
--------------------------------------------------------------------------------
This is instrumentation, not a solver. It turns a graph summary into the three
independent signals that separate the live mechanisms for D5:

  1. cross_solver_geometry -> "geometry-source decoupling" (different geometry queried)
  2. support_fraction      -> "measurement support absent"  (no extraction)
  3. input_output_delta    -> "graph inert"                 (wiring/plumbing)
  4. render_state_hash     -> "stale render dependency"     (render chain)
  plus stale_dependency_count + geometry_epoch lineage -> "stale geometry join"

Every graph_health row carries a `decision` chosen by a fixed decision table
(see `decide_mechanism`). The decision is the discriminating output: it routes
the next implementation step rather than scoring it.

No heavy inference runs here. It hashes inputs/outputs, reads counts already
present in the summary, and emits JSON/NDJSON. The per-variable input/output
delta and render_state_hash are explicitly pending downstream differencers and
the renderer; where a real signal exists today (objective energy delta) it is
populated, never faked.

Inputs
------
  --annotation-json PATH   v18 annotation with top-level `factor_graph_summary`
                           and per-frame `frames[*].factor_graph_solution`.
  --graph-summary PATH     just the `factor_graph_summary` dict (JSON).
  --synthetic [--synthetic-defect MODE]
                           build an in-memory summary; MODE in
                           {none,support,inert,geometry_mismatch,stale_render,stale_join}.
                           Demonstrates each decision branch with no model runs.

Outputs
-------
  --out-dir PATH           writes:
                             manifest.json            (ego.hoi 0.1.0 sidecar)
                             graph_health.json        (table skeleton, all rows)
                             graph_health.ndjson      (one row per solution)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_FAMILY = "ego.hoi"
SCHEMA_VERSION = "0.1.0"
NAMESPACE = "org.ego.hoi"
PRODUCER = "build_ego_hoi_sidecar_and_graph_health.py"
PRODUCER_VERSION = "r0r1-scaffold-1"

# Factor families instrumented. These are the v18 families that already appear
# in factor_graph_summary / per-frame factor counts. The scaffold is
# category-agnostic: it reads whatever counts the summary provides.
DEFAULT_FACTOR_FAMILIES: tuple[str, ...] = (
    "contact_switch_discrete",
    "contact_switch_temporal",
    "contact_local_nonpenetration",
    "occlusion_owner_discrete",
    "hand_state_observation",
    "object_se3_observation",
    "part_se3_observation",
    "camera_depth_correction_observation",
)

# Decision thresholds. These are instrumentation thresholds, not acceptance
# gates: they route the next diagnostic, they do not block output.
SUPPORT_FRACTION_THRESHOLD = 0.05   # <5% support on a critical family => suspect
RESIDUAL_EPSILON = 1.0e-9           # |residual| above this counts as "active"
ENERGY_DELTA_INERT_EPS = 1.0e-6     # objective energy moved less than this => inert


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_canonical(obj: Any) -> str:
    """Stable hash of a JSON-serializable object (sorted keys, no spaces)."""
    return sha256_bytes(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def short(h: str) -> str:
    return h[:12]


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def synthetic_summary(defect: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a minimal v18-shaped summary that exercises each decision branch.

    The synthetic summary is intentionally tiny and contains only the fields the
    scaffold consumes. It is NOT a measurement: it exists so the dry run proves
    the decision table fires for each defect mode without any model runtime.
    """
    base_families = {fam: {"declared": 100, "supported": 100, "residual_sum": 5.0} for fam in DEFAULT_FACTOR_FAMILIES}
    energy_initial = 10.0
    if defect == "support":
        # M1: measurement support absent on a critical family.
        base_families["contact_switch_discrete"] = {"declared": 100, "supported": 1, "residual_sum": 0.0}
        base_families["object_se3_observation"] = {"declared": 100, "supported": 2, "residual_sum": 0.0}
        energy_after = 9.5  # optimizer moved a little, but support is the defect
    elif defect == "inert":
        # M2: graph inert. Active residuals but objective did not move.
        energy_after = energy_initial  # exactly no movement
    else:
        energy_after = 4.0  # healthy movement

    frames: list[dict[str, Any]] = []
    for i in range(3):
        frames.append(
            {
                "frame_idx": i,
                "factor_graph_solution": {
                    "objective": {"energy": energy_after},
                    "factors": {fam: v["supported"] for fam, v in base_families.items()},
                    "variables": {
                        "object_se3": [{"object_id": "obj_0", "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}],
                        "contact_switch": [
                            {"hand_side": "right", "object_id": "obj_0", "state": "contact"}
                        ],
                    },
                },
            }
        )

    summary = {
        "case": f"synthetic_{defect}",
        "factor_graph_summary": {
            "objective": {"energy_initial": energy_initial, "energy_after": energy_after},
            "factor_family_health": base_families,
            "declared_gauges": [
                {"variable": "object_se3::obj_0", "gauge": "anchor_frame_lock", "anchor_frame": 0}
            ],
            "geometry_epochs": [
                {"epoch_id": "geo_0", "anchor_frame": 0, "superseded_by": None, "source": "fusion"},
                {"epoch_id": "geo_1", "anchor_frame": 1, "superseded_by": None, "source": "fusion"},
            ],
            "active_geometry_epoch_id": "geo_1",
        },
    }

    if defect == "geometry_mismatch":
        summary["factor_graph_summary"]["cross_solver_geometry_consistency"] = {
            "solver_geometry_epoch_id": "geo_observed_keyboard_surface",
            "solver_geometry_source_family": "observed_surface",
            "contact_query_geometry_epoch_id": "geo_completed_keyboard_trellis",
            "contact_query_geometry_source_family": "trellis_completed",
            "render_geometry_epoch_id": "geo_completed_keyboard_trellis",
            "render_geometry_source_family": "trellis_completed",
            "signed_query_candidate_vertex_count": 0,
            "watertight": False,
            "face_provenance_summary": {
                "observed_fraction": 0.046,
                "trellis_completed_fraction": 0.954,
            },
            "observed_surface_penetration_m": 0.102,
            "published_contact_gap_m": 0.039,
        }

    if defect == "stale_join":
        # M3 (stale joins): variables reference a superseded geometry epoch.
        for fr in frames:
            fr["factor_graph_solution"]["variables"]["object_se3"] = [
                {"object_id": "obj_0", "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "geometry_epoch_id": "geo_0"}
            ]
        summary["factor_graph_summary"]["geometry_epochs"][0]["superseded_by"] = "geo_1"

    if defect == "stale_render":
        # Render-chain defect is signalled at the manifest level: the renderer
        # has not refreshed, so render_state_hash is null/unchanged even though
        # the graph moved. We represent this by leaving render_state_hash unset.
        pass

    return summary, frames


def resolve_input(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Return (annotation_dict, frames, input_hashes). input_hashes maps role->sha256."""
    input_hashes: dict[str, str] = {}
    if args.synthetic:
        annotation, frames = synthetic_summary(args.synthetic_defect)
        input_hashes["annotation_json"] = sha256_json_canonical(annotation)
        input_hashes["synthetic_seed"] = sha256_bytes(f"{args.synthetic_defect}".encode("utf-8"))
        return annotation, frames, input_hashes

    if args.annotation_json:
        path = Path(args.annotation_json)
        annotation = load_json(path)
        input_hashes["annotation_json"] = sha256_file(path)
        frames = annotation.get("frames") or []
        return annotation, frames, input_hashes

    if args.graph_summary:
        path = Path(args.graph_summary)
        summary_obj = load_json(path)
        annotation = {"factor_graph_summary": summary_obj, "frames": []}
        input_hashes["graph_summary"] = sha256_file(path)
        return annotation, [], input_hashes

    raise RuntimeError("no input provided (use --synthetic, --annotation-json, or --graph-summary)")


# --------------------------------------------------------------------------- #
# Graph-health derivation
# --------------------------------------------------------------------------- #
def _family_health_entry(declared: int, supported: int, residual_sum: float) -> dict[str, Any]:
    declared = int(declared or 0)
    supported = int(supported or 0)
    residual_sum = float(residual_sum or 0.0)
    support_fraction = (supported / declared) if declared > 0 else 0.0
    active_residual_count = supported if residual_sum > RESIDUAL_EPSILON else 0
    return {
        "declared_count": declared,
        "supported_count": supported,
        "support_fraction": round(support_fraction, 6),
        "active_residual_count": active_residual_count,
        "residual_sum": round(residual_sum, 6),
    }


def derive_family_health(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per factor family: support fraction + active residual count.

    Reads `factor_family_health` from the summary if present (synthetic /
    future producers). Falls back to declared-only counts from
    `factor_counts` so the scaffold still emits a row for real v18 summaries
    that only carry counts (residual_sum unknown -> treated as 0 / pending).
    """
    fg = summary.get("factor_graph_summary") or summary
    declared = fg.get("factor_family_health")
    out: dict[str, dict[str, Any]] = {}
    if isinstance(declared, dict) and declared:
        for fam, entry in declared.items():
            if not isinstance(entry, dict):
                continue
            out[fam] = _family_health_entry(
                entry.get("declared", 0), entry.get("supported", 0), entry.get("residual_sum", 0.0)
            )
        return out

    factor_counts = fg.get("factor_counts") or {}
    for fam in DEFAULT_FACTOR_FAMILIES:
        cnt = int(factor_counts.get(fam, 0) or 0)
        out[fam] = {
            "declared_count": cnt,
            "supported_count": None,          # not reported by v18 counts alone
            "support_fraction": None,
            "active_residual_count": None,
            "residual_sum": None,
            "pending": "v18 factor_counts carry declarations only; support/residual pending graph-health producer",
        }
    return out


def derive_geometry_epoch_lineage(summary: dict[str, Any]) -> dict[str, Any]:
    fg = summary.get("factor_graph_summary") or summary
    epochs = fg.get("geometry_epochs") or []
    active = fg.get("active_geometry_epoch_id")
    lineage: list[dict[str, Any]] = []
    superseded_ids: set[str] = set()
    for ep in epochs:
        if not isinstance(ep, dict):
            continue
        eid = str(ep.get("epoch_id"))
        sup = ep.get("superseded_by")
        if sup is not None:
            superseded_ids.add(eid)
        lineage.append(
            {
                "epoch_id": eid,
                "anchor_frame": ep.get("anchor_frame"),
                "source": ep.get("source", "unknown"),
                "superseded_by": sup,
                "is_active": eid == active,
            }
        )
    return {
        "active_geometry_epoch_id": active,
        "lineage": lineage,
        "superseded_epoch_ids": sorted(superseded_ids),
    }


def derive_stale_dependency_count(frames: list[dict[str, Any]], superseded_ids: set[str]) -> int:
    """Count variables joined to a superseded geometry epoch (stale joins)."""
    if not superseded_ids:
        return 0
    stale = 0
    for fr in frames:
        g = fr.get("factor_graph_solution")
        if not isinstance(g, dict):
            continue
        variables = g.get("variables") or {}
        for key in ("object_se3", "part_se3"):
            for row in variables.get(key, []) or []:
                if isinstance(row, dict) and str(row.get("geometry_epoch_id")) in superseded_ids:
                    stale += 1
    return stale


def derive_gauge_declaration_count(summary: dict[str, Any]) -> int:
    fg = summary.get("factor_graph_summary") or summary
    gauges = fg.get("declared_gauges") or []
    return len(gauges) if isinstance(gauges, list) else 0


def derive_cross_solver_geometry_consistency(summary: dict[str, Any]) -> dict[str, Any]:
    """Compare geometry lineage used by solver, contact/NP query, and render.

    This discriminates HOT3D clip001850: interval MANO measured penetration
    against an observed keyboard surface, contact/nonpenetration queried a
    different mostly-completed non-watertight mesh with zero signed-query
    candidates, and the renderer published a positive gap. That is different
    from generic missing support: the measurement exists, but it is stranded on
    a different geometry source.
    """
    fg = summary.get("factor_graph_summary") or summary
    raw = (
        fg.get("cross_solver_geometry_consistency")
        or fg.get("geometry_source_consistency")
        or fg.get("geometry_source_checks")
        or {}
    )
    if not isinstance(raw, dict):
        raw = {}

    solver_epoch = raw.get("solver_geometry_epoch_id") or raw.get("solver_epoch_id")
    contact_epoch = raw.get("contact_query_geometry_epoch_id") or raw.get("contact_epoch_id")
    render_epoch = raw.get("render_geometry_epoch_id") or raw.get("render_epoch_id")
    solver_source = raw.get("solver_geometry_source_family") or raw.get("solver_source_family")
    contact_source = raw.get("contact_query_geometry_source_family") or raw.get("contact_source_family")
    render_source = raw.get("render_geometry_source_family") or raw.get("render_source_family")

    epochs = [v for v in (solver_epoch, contact_epoch, render_epoch) if v not in (None, "")]
    sources = [v for v in (solver_source, contact_source, render_source) if v not in (None, "")]
    epoch_consistent = len(set(map(str, epochs))) <= 1 if epochs else None
    source_consistent = len(set(map(str, sources))) <= 1 if sources else None

    candidate_count = raw.get("signed_query_candidate_vertex_count")
    try:
        candidate_count_int = None if candidate_count is None else int(candidate_count)
    except (TypeError, ValueError):
        candidate_count_int = None

    # Metric statistic policy (KT-6): do NOT treat max-of-max as the decisive
    # penetration statistic. The summary may carry an explicit stat_policy
    # block; if present, observed_surface_penetration_m is already the decisive
    # (non-max) value. We also surface the policy so the decision rationale can
    # cite it.
    stat_policy = raw.get("observed_surface_penetration_stat_policy") or {}
    observed_penetration = raw.get("observed_surface_penetration_m")
    published_gap = raw.get("published_contact_gap_m")
    try:
        observed_penetration_float = None if observed_penetration is None else float(observed_penetration)
    except (TypeError, ValueError):
        observed_penetration_float = None
    try:
        published_gap_float = None if published_gap is None else float(published_gap)
    except (TypeError, ValueError):
        published_gap_float = None

    # provenance completeness gate (KT-6): the three geometry epoch/source
    # families must be derived from measured provenance; if any provenance
    # source is missing, the decision routes to evidence_incomplete rather
    # than pretending measured cross-solver consistency.
    completeness = raw.get("provenance_completeness") or {}
    provenance_complete = bool(completeness.get("complete", True))
    provenance_missing = list(completeness.get("missing") or [])

    mismatch_reasons: list[str] = []
    if epoch_consistent is False:
        mismatch_reasons.append("solver/contact/render geometry_epoch_id differ")
    if source_consistent is False:
        mismatch_reasons.append("solver/contact/render geometry_source_family differ")
    if candidate_count_int == 0 and (epoch_consistent is False or source_consistent is False):
        mismatch_reasons.append("signed query used zero candidate vertices on a different geometry source")
    # Deliberately do not add a mismatch reason merely because a penetration
    # scalar and a render gap are both nonzero. That coexistence is a symptom
    # after source provenance is already known to differ; by itself it is a
    # tautology across channels and was the KT-6 failure mode.

    return {
        "solver_geometry_epoch_id": solver_epoch,
        "solver_geometry_source_family": solver_source,
        "solver_geometry_provenance": raw.get("solver_geometry_provenance") or {},
        "contact_query_geometry_epoch_id": contact_epoch,
        "contact_query_geometry_source_family": contact_source,
        "contact_query_geometry_provenance": raw.get("contact_query_geometry_provenance") or {},
        "render_geometry_epoch_id": render_epoch,
        "render_geometry_source_family": render_source,
        "render_geometry_provenance": raw.get("render_geometry_provenance") or {},
        "sign_mesh_watertight": raw.get("sign_mesh_watertight"),
        "completed_surface_mesh_watertight": raw.get("completed_surface_mesh_watertight"),
        "geometry_epoch_consistent": epoch_consistent,
        "geometry_source_consistent": source_consistent,
        "signed_query_candidate_vertex_count": candidate_count_int,
        "watertight": raw.get("watertight"),
        "face_provenance_summary": raw.get("face_provenance_summary") or {},
        "observed_surface_penetration_m": observed_penetration_float,
        "observed_surface_penetration_stat_policy": stat_policy,
        "published_contact_gap_m": published_gap_float,
        "provenance_complete": provenance_complete,
        "provenance_missing": provenance_missing,
        "mismatch_reasons": mismatch_reasons,
    }


def cross_solver_geometry_mismatch(check: dict[str, Any]) -> bool:
    return bool(check.get("mismatch_reasons"))


def derive_objective_delta(summary: dict[str, Any]) -> dict[str, Any]:
    fg = summary.get("factor_graph_summary") or summary
    obj = fg.get("objective") or {}
    e_init = obj.get("energy_initial")
    e_after = obj.get("energy_after")
    if e_init is None or e_after is None:
        return {"energy_initial": None, "energy_after": None, "energy_delta": None, "pending": "objective missing"}
    e_init = float(e_init)
    e_after = float(e_after)
    return {
        "energy_initial": e_init,
        "energy_after": e_after,
        "energy_delta": round(e_init - e_after, 9),
    }


def output_hash_from_frames(frames: list[dict[str, Any]]) -> str:
    """Hash the solved variables across all frames -> graph output fingerprint.

    This is the graph-side output hash. The renderer fingerprint
    (render_state_hash) is a separate, downstream hash that the renderer must
    populate; comparing the two is what exposes a stale render chain.
    """
    variables_only = []
    for fr in frames:
        g = fr.get("factor_graph_solution")
        if isinstance(g, dict):
            variables_only.append({"frame_idx": fr.get("frame_idx"), "variables": g.get("variables")})
    return sha256_json_canonical(variables_only) if variables_only else sha256_bytes(b"empty")


# --------------------------------------------------------------------------- #
# Decision table (the discriminator)
# --------------------------------------------------------------------------- #
def decide_mechanism(
    family_health: dict[str, dict[str, Any]],
    objective_delta: dict[str, Any],
    stale_dependency_count: int,
    cross_solver_geometry: dict[str, Any],
    render_state_hash: str | None,
    graph_output_hash: str,
    geometry_lineage: dict[str, Any],
) -> dict[str, Any]:
    """Map the three independent signals to a mechanism decision.

    Ordering is deliberate and reflects causal precedence: you cannot diagnose
    wiring before you confirm measurements exist, and you cannot diagnose the
    render chain before you confirm the graph moved.
    """
    rationale: list[str] = []

    # 0. evidence_incomplete (KT-6): if the cross-solver geometry fields were
    #    not derivable from measured provenance (a source hash/filter state is
    #    missing), do NOT pretend measured cross-solver consistency. Route to
    #    evidence_incomplete so the missing provenance is surfaced rather than
    #    hidden behind an authored-string mismatch.
    prov_complete = cross_solver_geometry.get("provenance_complete", True)
    prov_missing = cross_solver_geometry.get("provenance_missing") or []
    if not prov_complete:
        rationale.append(
            "cross_solver_geometry provenance incomplete; cannot derive measured "
            f"geometry-source consistency. missing={prov_missing}"
        )
        return {
            "decision": "evidence_incomplete",
            "mechanism": "KT-6: a geometry provenance source (depth_npz/mesh/sign-mesh/render-state hash) is missing; cross-solver consistency is asserted, not measured",
            "next_intervention": "re-emit the cross_solver_geometry_consistency block from measured provenance (file hashes + filter states) before any cross-solver decision is trusted",
            "rationale": rationale,
            "provenance_missing": prov_missing,
            "cross_solver_geometry": cross_solver_geometry,
        }

    # 1. cross-solver geometry-source decoupling. This precedes generic
    # support checks because the selected HOT3D slice has support in one solver
    # and zero candidates in another due to different geometry sources.
    if cross_solver_geometry_mismatch(cross_solver_geometry):
        rationale.extend(cross_solver_geometry.get("mismatch_reasons") or [])
        return {
            "decision": "cross_solver_geometry_decoupled",
            "mechanism": "M5: solver/contact/render stages query different geometry epochs or source families",
            "next_intervention": "instrument and assert geometry lineage across solver_geometry_epoch_id, contact_query_geometry_epoch_id, and render_geometry_epoch_id before adding contact factors",
            "rationale": rationale,
            "cross_solver_geometry": cross_solver_geometry,
        }

    # 2. measurement support (M1)
    critical_families = ("contact_switch_discrete", "object_se3_observation")
    worst_support = None
    for fam in critical_families:
        entry = family_health.get(fam) or {}
        sf = entry.get("support_fraction")
        if sf is None:
            continue
        if worst_support is None or sf < worst_support:
            worst_support = sf
    if worst_support is not None and worst_support < SUPPORT_FRACTION_THRESHOLD:
        rationale.append(
            f"support_fraction {worst_support:.3f} < {SUPPORT_FRACTION_THRESHOLD} on a critical family"
        )
        return {
            "decision": "measurement_support_absent",
            "mechanism": "M1: measurement extraction produced no support before the optimizer",
            "next_intervention": "repair measurement extraction (masks/depth/tracks/correspondences); do NOT retune optimizer weights",
            "rationale": rationale,
        }

    # 3. graph inert (M2): objective did not move
    e_delta = objective_delta.get("energy_delta")
    if e_delta is not None and abs(e_delta) <= ENERGY_DELTA_INERT_EPS:
        rationale.append(f"objective energy_delta {e_delta} ~ 0 (optimizer did not move)")
        return {
            "decision": "graph_inert",
            "mechanism": "M2: variables not wired to residuals or solver plumbing broken",
            "next_intervention": "repair variable wiring / solver plumbing; check Jacobian and gauge lock",
            "rationale": rationale,
        }

    # 4. stale geometry join (M3 stale deps)
    if stale_dependency_count > 0:
        rationale.append(f"{stale_dependency_count} variables join a superseded geometry epoch")
        return {
            "decision": "stale_geometry_join",
            "mechanism": "M3: stale object/hand/geometry ids joined into the solve",
            "next_intervention": "re-bind variables to active geometry_epoch_id; refresh dependency ids",
            "rationale": rationale,
        }

    # 5. stale render chain: graph moved but render fingerprint is missing/unchanged
    if render_state_hash in (None, ""):
        rationale.append("graph_output_hash populated but render_state_hash is null (renderer has not refreshed)")
        return {
            "decision": "stale_render_dependency",
            "mechanism": "render chain: state-to-render dependency broken or not yet run",
            "next_intervention": "repair state-to-render dependency chain; run renderer from graph output rows",
            "rationale": rationale,
            "graph_output_hash": graph_output_hash,
        }

    return {
        "decision": "active_healthy",
        "mechanism": "no D5 defect detected at instrumentation resolution",
        "next_intervention": "proceed to per-variable input/output delta differencer and render hash lineage",
        "rationale": rationale or ["all instrumentation signals nominal"],
    }


# --------------------------------------------------------------------------- #
# Row + manifest construction
# --------------------------------------------------------------------------- #
def build_graph_health_row(
    *,
    run_id: str,
    base_job_id: str,
    base_manifest_sha256: str,
    solution_id: str,
    summary: dict[str, Any],
    frames: list[dict[str, Any]],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    family_health = derive_family_health(summary)
    objective_delta = derive_objective_delta(summary)
    geometry_lineage = derive_geometry_epoch_lineage(summary)
    stale_dep_count = derive_stale_dependency_count(frames, set(geometry_lineage["superseded_epoch_ids"]))
    gauge_count = derive_gauge_declaration_count(summary)
    cross_solver_geometry = derive_cross_solver_geometry_consistency(summary)
    graph_output_hash = output_hash_from_frames(frames)

    # input/output deltas placeholder: objective energy delta is real today;
    # per-variable family deltas are explicitly pending a downstream differencer.
    input_output_deltas: dict[str, Any] = {
        "objective_energy_delta": objective_delta.get("energy_delta"),
        "per_family": {
            fam: {
                "delta_norm": None,
                "max_abs_delta": None,
                "pending": "populated by graph_input_output_differencer (R1 follow-on)",
            }
            for fam in family_health
        },
    }

    row: dict[str, Any] = {
        "schema_family": SCHEMA_FAMILY,
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "run_id": run_id,
        "base_job_id": base_job_id,
        "base_manifest_sha256": base_manifest_sha256,
        "solution_id": solution_id,
        "factor_family_health": family_health,
        "support_fraction_min_critical": _min_critical_support(family_health),
        "active_residual_count_total": _active_residual_total(family_health),
        "objective_delta": objective_delta,
        "input_output_deltas": input_output_deltas,
        "stale_dependency_count": stale_dep_count,
        "gauge_declaration_count": gauge_count,
        "geometry_epoch_id_active": geometry_lineage["active_geometry_epoch_id"],
        "geometry_epoch_lineage": geometry_lineage["lineage"],
        "cross_solver_geometry_consistency": cross_solver_geometry,
        "input_hashes": input_hashes,
        "output_hashes": {
            "graph_output": graph_output_hash,
            "object_pose": None,
            "hand_state": None,
            "contact": None,
            "pending": "per-family output hashes populated by table writers",
        },
        "render_state_hash": None,  # placeholder: populated by renderer
        "render_state_hash_lineage": {
            "graph_output_hash": graph_output_hash,
            "render_state_hash": None,
            "pending": "renderer must write its consumed-state fingerprint here",
        },
    }
    row["mechanism_decision"] = decide_mechanism(
        family_health,
        objective_delta,
        stale_dep_count,
        cross_solver_geometry,
        row["render_state_hash"],
        graph_output_hash,
        geometry_lineage,
    )
    return row


def _min_critical_support(family_health: dict[str, dict[str, Any]]) -> Any:
    vals = [
        (family_health[fam] or {}).get("support_fraction")
        for fam in ("contact_switch_discrete", "object_se3_observation")
        if (family_health.get(fam) or {}).get("support_fraction") is not None
    ]
    return round(min(vals), 6) if vals else None


def _active_residual_total(family_health: dict[str, dict[str, Any]]) -> Any:
    total = 0
    seen = False
    for entry in family_health.values():
        v = (entry or {}).get("active_residual_count")
        if v is None:
            continue
        seen = True
        total += int(v)
    return total if seen else None


def build_manifest(
    *,
    run_id: str,
    base_job_id: str,
    base_manifest_sha256: str,
    out_dir: Path,
    graph_health_rows: list[dict[str, Any]],
    input_hashes: dict[str, str],
    annotation_source: str,
) -> dict[str, Any]:
    decisions = [r["mechanism_decision"]["decision"] for r in graph_health_rows]
    return {
        "schema_family": SCHEMA_FAMILY,
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "run_id": run_id,
        "provenance": {
            "base_job_id": base_job_id,
            "base_manifest_sha256": base_manifest_sha256,
            "producer": PRODUCER,
            "producer_version": PRODUCER_VERSION,
            "created_at_unix": time.time(),
            "annotation_source": annotation_source,
        },
        "inputs": [{"role": role, "sha256": h} for role, h in sorted(input_hashes.items())],
        "outputs": [
            {"path": "manifest.json", "role": "sidecar_manifest"},
            {"path": "graph_health.json", "role": "graph_health_table"},
            {"path": "graph_health.ndjson", "role": "graph_health_stream"},
        ],
        "declared_tables": [
            "objects.parquet",
            "object_parts.parquet",
            "geometry_epochs.parquet",
            "object_pose.parquet",
            "object_visible_surface.parquet",
            "rigidity_windows.parquet",
            "contact_hypotheses.parquet",
            "contact_frame_detail.parquet",
            "visibility_states.parquet",
            "hoi_hand_corrections.parquet",
            "graph_solutions.parquet",
            "hoi_validation_metrics.parquet",
        ],
        "graph_health_summary": {
            "row_count": len(graph_health_rows),
            "decisions": decisions,
            "support_fraction_min_critical": [r["support_fraction_min_critical"] for r in graph_health_rows],
            "stale_dependency_count_total": sum(r["stale_dependency_count"] for r in graph_health_rows),
            "cross_solver_geometry_mismatch_count": sum(
                1 for r in graph_health_rows if cross_solver_geometry_mismatch(r.get("cross_solver_geometry_consistency") or {})
            ),
            "gauge_declaration_count": [r["gauge_declaration_count"] for r in graph_health_rows],
        },
        "render_state_hash": None,  # placeholder: populated by renderer
        "out_dir": str(out_dir),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_ego_hoi_sidecar_and_graph_health.py",
        description=(
            "Emit an ego.hoi 0.1.0 sidecar manifest and a graph_health table skeleton "
            "from a v18 annotation/graph summary or a synthetic input. Instrumentation "
            "for graph-inertness and geometry-lineage mechanisms: routes the next implementation step, "
            "does not score or gate."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="use an in-memory synthetic summary (no model runs)")
    src.add_argument("--annotation-json", metavar="PATH", help="path to a v18 annotation JSON (with factor_graph_summary)")
    src.add_argument("--graph-summary", metavar="PATH", help="path to a standalone factor_graph_summary JSON")

    p.add_argument(
        "--synthetic-defect",
        choices=["none", "support", "inert", "geometry_mismatch", "stale_render", "stale_join"],
        default="none",
        help="defect mode to inject into the synthetic summary (default: none -> active_healthy)",
    )
    p.add_argument("--out-dir", metavar="PATH", required=True, help="directory to write manifest.json + graph_health.*")
    p.add_argument("--run-id", default="r0r1_scaffold_run", help="ego.hoi run id")
    p.add_argument("--base-job-id", default="", help="upstream base job id")
    p.add_argument("--base-manifest-sha256", default="", help="upstream base manifest sha256")
    p.add_argument("--solution-id", default="sol_0", help="graph solution id for the health row")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    annotation, frames, input_hashes = resolve_input(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = build_graph_health_row(
        run_id=args.run_id,
        base_job_id=args.base_job_id,
        base_manifest_sha256=args.base_manifest_sha256,
        solution_id=args.solution_id,
        summary=annotation,
        frames=frames,
        input_hashes=input_hashes,
    )

    manifest = build_manifest(
        run_id=args.run_id,
        base_job_id=args.base_job_id,
        base_manifest_sha256=args.base_manifest_sha256,
        out_dir=out_dir,
        graph_health_rows=[row],
        input_hashes=input_hashes,
        annotation_source=(
            f"synthetic:{args.synthetic_defect}" if args.synthetic
            else (args.annotation_json or args.graph_summary or "unknown")
        ),
    )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "graph_health.json").write_text(json.dumps([row], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "graph_health.ndjson").open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    decision = row["mechanism_decision"]["decision"]
    print(f"[ego.hoi] wrote {out_dir}/manifest.json")
    print(f"[ego.hoi] wrote {out_dir}/graph_health.json (+ .ndjson)")
    print(f"[ego.hoi] decision={decision}")
    print(f"[ego.hoi] rationale={row['mechanism_decision']['rationale']}")
    print(f"[ego.hoi] next_intervention={row['mechanism_decision']['next_intervention']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
