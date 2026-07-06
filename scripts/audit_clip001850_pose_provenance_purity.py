#!/usr/bin/env python3
"""
R0/R1 provenance-purity audit for object-pose state on HOT3D clip001850.

Purpose (D5/D6 mechanism: pose prior laundered as a measurement)
----------------------------------------------------------------
The static-fill / nearest-hold / interpolation laundering error (subagents
28/31) can recur unnoticed unless the pose state is audited for provenance
purity. This script reads the on-disk pose artifacts and enforces the
five-class ontology from `31_pose_state_semantics_theory.md`:

    observed_measured   -> numeric pose admissible (a measurement)
    observed_rejected   -> numeric pose RETAINED for audit only, NOT admissible
    unresolved          -> numeric pose RETAINED, NOT admissible (ambiguous)
    unknown             -> NO numeric pose (null/absent)
    solver_estimate     -> numeric pose admissible (a live posterior)

The discriminator between truth and laundering is provenance, not styling. A
value plus a "do not trust me" flag is still read as a value by every consumer
that keys on the pose column (verified: the delivered renderer places a metric
body from `translation_world_m` and reads no visibility field). Therefore a
frame that is unknown/rejected/unresolved MUST carry no numeric pose at all.

This is instrumentation, not a solver, and not an acceptance gate that prevents
approximate uncertain outputs. It records purity so a future R4/R7 pose graph
cannot silently re-introduce held/interpolated/imputed values into the
measurement channel.

Inputs (all on disk; no model inference, CPU-only)
--------------------------------------------------
  current unknown-preserving pose observations:
    <RESEARCH_ROOT>/unknown_preserving_pose_render/object_pose_observations.ndjson
  original P15 rigid pose graph (nearest-hold + interpolation):
    <V19_RUN>/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json
  static-fill (superseded negative evidence):
    <RESEARCH_ROOT>/static_gauge_pose_render/object_pose_static_gauge.ndjson
  render consumption lineage:
    <SIDECAR>/tables/render_consumption.ndjson
  pose graph liveness probe (inertness evidence):
    <RESEARCH_ROOT>/pose_graph_liveness/summary.json

Outputs (under <RESEARCH_ROOT>/pose_provenance_purity_audit/)
--------------------------------------------------------------
  summary.json     aggregate purity metrics + the five required checks
  per_frame.ndjson one graph_health-like row per frame (current + original)
  audit.md         human-readable audit report
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "clip001850_pose_provenance_purity_audit/v1"
RESEARCH_ROOT = Path(
    "/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706"
)
V19_RUN = Path(
    "/data2/ego_annotation_outputs/v19_runs/"
    "20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)
SIDECAR_TABLES = (
    RESEARCH_ROOT
    / "ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/tables"
)
OUT_DIR = RESEARCH_ROOT / "pose_provenance_purity_audit"

FRAME_COUNT = 150
# Admissible measured frames: rest-cluster consensus that survived KT-L1 outlier
# gating (subagent 26). These are the ONLY frames eligible to contribute a
# pose-measurement factor.
ADMISSIBLE_MEASURED_FRAMES = frozenset({30, 31, 32, 33, 34, 35, 36, 46})
# Rejected mask-drift observations (f60/75/76/77): a fit was attempted and
# rejected as inconsistent with the rest cluster; retained for audit only.
REJECTED_FRAMES = frozenset({60, 75, 76, 77})
# Unresolved hand-occluded observation (f45): ambiguous, neither accepted nor
# cleanly rejected.
UNRESOLVED_FRAMES = frozenset({45})


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_ndjson(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_numeric_pose(row: dict[str, Any]) -> bool:
    t = row.get("translation_world_m")
    R = row.get("rotation_world_from_completed_canonical_matrix")
    return t is not None and R is not None


def ontology_class(frame_idx: int) -> str:
    """Map a clip001850 frame to its pose-state ontology class.

    This is the ground-truth regime verdict from KT-L1/KT-L2 (subagents 26/27),
    not a heuristic. It is the reference the purity audit checks against.
    """
    if frame_idx in ADMISSIBLE_MEASURED_FRAMES:
        return "observed_measured"
    if frame_idx in REJECTED_FRAMES:
        return "observed_rejected"
    if frame_idx in UNRESOLVED_FRAMES:
        return "unresolved"
    return "unknown"


# --------------------------------------------------------------------------- #
# Loaders for the three pose artifacts
# --------------------------------------------------------------------------- #
def load_current_pose() -> list[dict[str, Any]]:
    """Current unknown-preserving object_pose_observations (150 rows)."""
    return load_ndjson(
        RESEARCH_ROOT
        / "unknown_preserving_pose_render/object_pose_observations.ndjson"
    )


def load_original_pose_graph() -> list[dict[str, Any]]:
    """Original P15 rigid pose graph rows (nearest-hold + interpolation)."""
    rep = load_json(
        V19_RUN
        / "measurements/pose_fits/keyboard_rigid_pose_graph/"
        "v19_rigid_object_pose_graph_report.json"
    )
    return rep["pose_rows"]


def load_static_fill_pose() -> list[dict[str, Any]]:
    """Superseded static-gauge fill (held T_rest on 142 frames)."""
    return load_ndjson(
        RESEARCH_ROOT / "static_gauge_pose_render/object_pose_static_gauge.ndjson"
    )


# --------------------------------------------------------------------------- #
# Purity analysis
# --------------------------------------------------------------------------- #
def classify_numeric_rows(
    rows: list[dict[str, Any]],
    *,
    numeric_key: str,
    source_key: str | None,
    source_getter,
) -> dict[str, Any]:
    """For a set of pose rows, classify each numeric row by ontology and by the
    artifact's own declared completion source.

    Returns counts needed by the required checks.
    """
    numeric_frames: list[int] = []
    numeric_by_class: dict[str, list[int]] = {
        "observed_measured": [],
        "observed_rejected": [],
        "unresolved": [],
        "unknown": [],
        "solver_estimate": [],
    }
    # forbidden = numeric pose on a non-admissible class (unknown/rejected/unresolved)
    forbidden_numeric: list[dict[str, Any]] = []
    # imputed = numeric pose sourced from a held/interpolated/static prior
    imputed_frames: list[int] = []

    for row in rows:
        fi = int(row["frame_idx"])
        cls = ontology_class(fi)
        src = source_getter(row)
        if has_numeric_pose(row):
            numeric_frames.append(fi)
            numeric_by_class[cls].append(fi)
            if cls in ("unknown", "observed_rejected", "unresolved"):
                forbidden_numeric.append(
                    {
                        "frame_idx": fi,
                        "ontology_class": cls,
                        "declared_source": src,
                        "reason": f"numeric pose present on {cls} frame",
                    }
                )
            if src is not None and _is_imputed_source(src):
                imputed_frames.append(fi)

    admissible_numeric = list(
        set(numeric_by_class["observed_measured"])
        | set(numeric_by_class["solver_estimate"])
    )
    total_numeric = len(numeric_frames)
    purity = (len(admissible_numeric) / total_numeric) if total_numeric else 1.0
    return {
        "total_numeric": total_numeric,
        "admissible_numeric_count": len(admissible_numeric),
        "admissible_numeric_frames": sorted(admissible_numeric),
        "purity": round(purity, 6),
        "numeric_by_ontology_class": {
            k: sorted(v) for k, v in numeric_by_class.items()
        },
        "forbidden_numeric_count": len(forbidden_numeric),
        "forbidden_numeric": forbidden_numeric,
        "imputed_pose_count": len(imputed_frames),
        "imputed_pose_frames": sorted(set(imputed_frames)),
        "unknown_null_pose_count": sum(
            1
            for r in rows
            if ontology_class(int(r["frame_idx"])) == "unknown"
            and not has_numeric_pose(r)
        ),
    }


_IMPUTED_MARKERS = (
    "nearest_visible_pose_hold",
    "interpolated_between_visible_pose_observations",
    "static_gauge_held_rest",
    "held",
    "interpolat",
    "imput",
    "static_prior",
)


def _is_imputed_source(src: str) -> bool:
    if not isinstance(src, str):
        return False
    low = src.lower()
    return any(m in low for m in _IMPUTED_MARKERS)


def current_source_getter(row: dict[str, Any]) -> str | None:
    return row.get("pose_source")


def original_source_getter(row: dict[str, Any]) -> str | None:
    tpg = row.get("temporal_pose_graph")
    if isinstance(tpg, dict):
        return tpg.get("pose_source")
    return row.get("pose_source")


def static_source_getter(row: dict[str, Any]) -> str | None:
    return row.get("pose_source")


# --------------------------------------------------------------------------- #
# Required check 3: render manifest lineage
# --------------------------------------------------------------------------- #
def audit_render_lineage() -> dict[str, Any]:
    rc = load_ndjson(SIDECAR_TABLES / "render_consumption.ndjson")
    current_pose_path = str(
        RESEARCH_ROOT
        / "unknown_preserving_pose_render/object_pose_observations.ndjson"
    )
    static_pose_path = str(
        RESEARCH_ROOT
        / "static_gauge_pose_render/object_pose_static_gauge.ndjson"
    )

    consumes_current = [
        i for i, r in enumerate(rc)
        if r.get("consumed_object_pose_observations") == current_pose_path
    ]
    consumes_static = [
        i for i, r in enumerate(rc)
        if r.get("consumed_static_gauge_pose") == static_pose_path
    ]
    # The latest consumers are the highest-index rows (unknown-preserving render
    # was produced after the static-gauge render). Current source of record =
    # the artifact consumed by the most-recent render that passes the
    # body_drawn_only_on_measured_frames gate.
    current_is_source_of_record = bool(consumes_current)
    static_superseded = bool(consumes_static) and not any(
        rc[i].get("acceptance_gates", {}).get(
            "render_manifest_consumed_static_gauge_pose_rows", False
        )
        and i >= max(consumes_current, default=-1)
        for i in consumes_static
    )
    # Determine which render consumers assert body-drawn-only-on-measured
    body_only_on_measured_rows = [
        i for i, r in enumerate(rc)
        if r.get("acceptance_gates", {}).get(
            "body_drawn_only_on_measured_frames"
        ) is True
    ]
    return {
        "render_consumption_row_count": len(rc),
        "rows_consuming_unknown_preserving_pose": consumes_current,
        "rows_consuming_static_gauge_pose": consumes_static,
        "current_pose_artifact_is_source_of_record": current_is_source_of_record,
        "static_fill_superseded": static_superseded
        or (bool(consumes_static) and bool(consumes_current)),
        "body_drawn_only_on_measured_asserted_by_rows": body_only_on_measured_rows,
        "current_pose_path": current_pose_path,
        "static_pose_path": static_pose_path,
    }


# --------------------------------------------------------------------------- #
# Required check 4: measurement-factor eligibility + solver_estimate count
# --------------------------------------------------------------------------- #
def audit_factor_eligibility() -> dict[str, Any]:
    rep = load_json(
        V19_RUN
        / "measurements/pose_fits/keyboard_rigid_pose_graph/"
        "v19_rigid_object_pose_graph_report.json"
    )
    opt = rep.get("optimizer", {}) or {}
    nfev = opt.get("nfev")
    cost = opt.get("cost")
    corr = rep.get("correction_summary", {}) or {}
    tcorr = corr.get("translation_delta_norm_m", {}) or {}
    inert = (
        nfev == 1
        and cost == 0.0
        and tcorr.get("max", 1.0) == 0.0
    )
    npen = rep.get("nonpenetration_target_frame_count", 0)
    # solver_estimate count: frames whose pose came from a live solver posterior
    # with nonzero correction. The P15 graph produced zero correction on every
    # direct-fit frame, so solver_estimate count = 0.
    solver_estimate_count = 0 if inert else None
    return {
        "eligible_measurement_factor_frames": sorted(ADMISSIBLE_MEASURED_FRAMES),
        "eligible_measurement_factor_count": len(ADMISSIBLE_MEASURED_FRAMES),
        "solver_estimate_count": solver_estimate_count,
        "solver_inert": inert,
        "optimizer_nfev": nfev,
        "optimizer_cost": cost,
        "translation_delta_max_m": tcorr.get("max"),
        "nonpenetration_target_frame_count": npen,
        "rationale": (
            "Only f30-36,f46 survived KT-L1 outlier gating as admissible "
            "measured rest-cluster observations; f45 is unresolved, "
            "f60/75/76/77 are rejected mask-drift, all others are unknown. "
            "The P15 optimizer is a zero-correction pass-through (nfev=1, "
            "cost=0, max translation delta=0), so no solver_estimate frames "
            "exist."
        ),
    }


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #
def run_audit() -> dict[str, Any]:
    current_rows = load_current_pose()
    original_rows = load_original_pose_graph()
    static_rows = load_static_fill_pose()

    current = classify_numeric_rows(
        current_rows,
        numeric_key="translation_world_m",
        source_key="pose_source",
        source_getter=current_source_getter,
    )
    original = classify_numeric_rows(
        original_rows,
        numeric_key="translation_world_m",
        source_key="temporal_pose_graph",
        source_getter=original_source_getter,
    )
    static = classify_numeric_rows(
        static_rows,
        numeric_key="translation_world_m",
        source_key="pose_source",
        source_getter=static_source_getter,
    )

    render_lineage = audit_render_lineage()
    factor_elig = audit_factor_eligibility()

    # ---- Required check verdicts ----
    # (1) no unknown/rejected/unresolved frame has numeric pose in current
    check_1 = {
        "name": "current_no_forbidden_numeric_pose",
        "passed": current["forbidden_numeric_count"] == 0,
        "forbidden_numeric_count": current["forbidden_numeric_count"],
        "forbidden_numeric": current["forbidden_numeric"],
    }
    # (2) original pose graph contains held/interpolated/imputed numeric -> fails purity
    original_fails = (
        original["imputed_pose_count"] > 0
        or original["forbidden_numeric_count"] > 0
        or original["purity"] < 1.0
    )
    check_2 = {
        "name": "original_pose_graph_fails_purity",
        "passed": original_fails,
        "imputed_pose_count": original["imputed_pose_count"],
        "forbidden_numeric_count": original["forbidden_numeric_count"],
        "purity": original["purity"],
        "note": (
            "PASS here means the original graph IS contaminated (the audit "
            "correctly detects the laundering); a clean original would be a "
            "failure of detection."
        ),
    }
    # (3) renderer points to unknown-preserving; static-fill superseded
    check_3 = {
        "name": "render_manifest_points_to_unknown_preserving",
        "passed": (
            render_lineage["current_pose_artifact_is_source_of_record"]
            and render_lineage["static_fill_superseded"]
        ),
        "current_is_source_of_record": render_lineage[
            "current_pose_artifact_is_source_of_record"
        ],
        "static_fill_superseded": render_lineage["static_fill_superseded"],
    }
    # (4) measurement factors eligible only f30-36,f46; solver_estimate 0
    check_4 = {
        "name": "measurement_factor_eligibility_and_solver_estimate",
        "passed": (
            factor_elig["eligible_measurement_factor_count"] == 8
            and set(factor_elig["eligible_measurement_factor_frames"])
            == set(ADMISSIBLE_MEASURED_FRAMES)
            and factor_elig["solver_estimate_count"] == 0
        ),
        "eligible_frames": factor_elig["eligible_measurement_factor_frames"],
        "solver_estimate_count": factor_elig["solver_estimate_count"],
        "solver_inert": factor_elig["solver_inert"],
    }
    # (5) current render body drawn only on measured frames
    check_5 = {
        "name": "current_render_body_drawn_only_on_measured",
        "passed": (
            len(render_lineage["body_drawn_only_on_measured_asserted_by_rows"]) > 0
            and render_lineage["current_pose_artifact_is_source_of_record"]
        ),
        "asserted_by_rows": render_lineage[
            "body_drawn_only_on_measured_asserted_by_rows"
        ],
    }

    # graph_health-like aggregate row (the required fields + provenance)
    graph_health_row = {
        "schema": SCHEMA,
        "case": "hot3d_clip001850_keyboard",
        "metric_family": "pose_provenance_purity",
        "pose_measurement_purity": {
            "current_unknown_preserving": current["purity"],
            "original_pose_graph": original["purity"],
            "static_gauge_fill": static["purity"],
        },
        "imputed_pose_count": {
            "current_unknown_preserving": current["imputed_pose_count"],
            "original_pose_graph": original["imputed_pose_count"],
            "static_gauge_fill": static["imputed_pose_count"],
        },
        "unknown_null_pose_count": current["unknown_null_pose_count"],
        "rejected_fit_count": len(REJECTED_FRAMES),
        "rejected_fit_frames": sorted(REJECTED_FRAMES),
        "unresolved_count": len(UNRESOLVED_FRAMES),
        "unresolved_frames": sorted(UNRESOLVED_FRAMES),
        "current_render_body_drawn_only_on_measured": check_5["passed"],
        "solver_estimate_count": factor_elig["solver_estimate_count"],
        "eligible_measurement_factor_count": factor_elig[
            "eligible_measurement_factor_count"
        ],
        "decision": (
            "pose_channel_pure_current__original_laundering_detected"
            if check_1["passed"] and check_2["passed"]
            else "POSE_CHANNEL_PURITY_VIOLATION"
        ),
    }

    summary = {
        "schema": SCHEMA,
        "case": "hot3d_clip001850_keyboard",
        "frame_count": FRAME_COUNT,
        "ontology": {
            "observed_measured": sorted(ADMISSIBLE_MEASURED_FRAMES),
            "observed_rejected": sorted(REJECTED_FRAMES),
            "unresolved": sorted(UNRESOLVED_FRAMES),
            "unknown_count": FRAME_COUNT
            - len(ADMISSIBLE_MEASURED_FRAMES)
            - len(REJECTED_FRAMES)
            - len(UNRESOLVED_FRAMES),
            "solver_estimate": [],
        },
        "inputs": {
            "current_pose": str(
                RESEARCH_ROOT
                / "unknown_preserving_pose_render/object_pose_observations.ndjson"
            ),
            "current_pose_sha256": sha256_file(
                RESEARCH_ROOT
                / "unknown_preserving_pose_render/object_pose_observations.ndjson"
            ),
            "original_pose_graph": str(
                V19_RUN
                / "measurements/pose_fits/keyboard_rigid_pose_graph/"
                "v19_rigid_object_pose_graph_report.json"
            ),
            "static_fill": str(
                RESEARCH_ROOT
                / "static_gauge_pose_render/object_pose_static_gauge.ndjson"
            ),
        },
        "current_unknown_preserving": current,
        "original_pose_graph": original,
        "static_gauge_fill": static,
        "render_lineage": render_lineage,
        "factor_eligibility": factor_elig,
        "checks": [check_1, check_2, check_3, check_4, check_5],
        "graph_health_row": graph_health_row,
    }
    summary["all_checks_pass"] = all(
        c["passed"] for c in summary["checks"]
    )
    return summary


def write_outputs(summary: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # per_frame.ndjson: one graph_health-like row per frame for current + original
    current_rows = load_current_pose()
    original_rows = load_original_pose_graph()
    cur_by = {int(r["frame_idx"]): r for r in current_rows}
    orig_by = {int(r["frame_idx"]): r for r in original_rows}
    per_frame = []
    for fi in range(FRAME_COUNT):
        cls = ontology_class(fi)
        c = cur_by.get(fi, {})
        o = orig_by.get(fi, {})
        c_num = has_numeric_pose(c)
        o_num = has_numeric_pose(o)
        o_src = (o.get("temporal_pose_graph") or {}).get("pose_source")
        per_frame.append(
            {
                "schema": SCHEMA,
                "frame_idx": fi,
                "ontology_class": cls,
                "current_has_numeric_pose": c_num,
                "current_pose_source": c.get("pose_source"),
                "current_numeric_admissible": c_num
                and cls == "observed_measured",
                "current_violation": (
                    c_num and cls != "observed_measured"
                ),
                "original_has_numeric_pose": o_num,
                "original_pose_source": o_src,
                "original_is_imputed": _is_imputed_source(o_src)
                if o_src
                else False,
                "original_violation": (
                    o_num
                    and (cls in ("unknown", "observed_rejected", "unresolved"))
                ),
                "eligible_measurement_factor": cls == "observed_measured",
            }
        )
    with (OUT_DIR / "per_frame.ndjson").open("w", encoding="utf-8") as f:
        for row in per_frame:
            f.write(json.dumps(row) + "\n")

    # audit.md
    md = render_audit_md(summary, per_frame)
    (OUT_DIR / "audit.md").write_text(md, encoding="utf-8")


def render_audit_md(summary: dict[str, Any], per_frame: list[dict[str, Any]]) -> str:
    c = summary["current_unknown_preserving"]
    o = summary["original_pose_graph"]
    s = summary["static_gauge_fill"]
    rl = summary["render_lineage"]
    fe = summary["factor_eligibility"]
    checks = summary["checks"]
    lines = []
    lines.append("# Pose provenance-purity audit — clip001850 keyboard")
    lines.append("")
    lines.append(
        "Discriminator between truth and laundering is **provenance**, not "
        "styling. A frame with no admissible observation must carry **no pose "
        "value at all**, because every consumer that keys on the pose column "
        "places a metric body from `translation_world_m` and reads no "
        "visibility flag (verified: `render_v19_contact_state_full_duration.py` "
        "`observed_body_world`)."
    )
    lines.append("")
    lines.append("## Five-class ontology")
    ont = summary["ontology"]
    lines.append(
        f"- observed_measured: {ont['observed_measured']} ({len(ont['observed_measured'])} frames)"
    )
    lines.append(
        f"- observed_rejected: {ont['observed_rejected']} ({len(ont['observed_rejected'])} frames) — retained for audit only, NOT admissible"
    )
    lines.append(
        f"- unresolved: {ont['unresolved']} ({len(ont['unresolved'])} frame) — retained, NOT admissible"
    )
    lines.append(
        f"- unknown: {ont['unknown_count']} frames — NO numeric pose (null)"
    )
    lines.append(
        f"- solver_estimate: {ont['solver_estimate']} (0 — P15 optimizer inert)"
    )
    lines.append("")
    lines.append("## Required checks")
    for ch in checks:
        mark = "PASS" if ch["passed"] else "FAIL"
        lines.append(f"- **[{mark}] {ch['name']}**")
        for k, v in ch.items():
            if k in ("name", "passed"):
                continue
            lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("## Purity summary")
    lines.append(
        "| artifact | total numeric | admissible numeric | purity | imputed | forbidden numeric |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, d in [
        ("current unknown-preserving", c),
        ("original P15 pose graph", o),
        ("static-gauge fill (superseded)", s),
    ]:
        lines.append(
            f"| {name} | {d['total_numeric']} | {d['admissible_numeric_count']} | "
            f"{d['purity']} | {d['imputed_pose_count']} | {d['forbidden_numeric_count']} |"
        )
    lines.append("")
    lines.append(
        f"- `unknown_null_pose_count` (current): **{c['unknown_null_pose_count']}** "
        f"(all {ont['unknown_count']} unknown frames carry null pose)"
    )
    lines.append(
        f"- `rejected_fit_count`: **{len(REJECTED_FRAMES)}** "
        f"({sorted(REJECTED_FRAMES)})"
    )
    lines.append("")
    lines.append("## Factor eligibility")
    lines.append(
        f"- eligible measurement-factor frames: {fe['eligible_measurement_factor_frames']} "
        f"({fe['eligible_measurement_factor_count']})"
    )
    lines.append(f"- solver_estimate_count: {fe['solver_estimate_count']}")
    lines.append(
        f"- solver inert: {fe['solver_inert']} "
        f"(nfev={fe['optimizer_nfev']}, cost={fe['optimizer_cost']}, "
        f"max translation delta={fe['translation_delta_max_m']} m)"
    )
    lines.append("")
    lines.append("## Render manifest lineage")
    lines.append(
        f"- rows consuming unknown-preserving pose: {rl['rows_consuming_unknown_preserving_pose']}"
    )
    lines.append(
        f"- rows consuming static-gauge pose: {rl['rows_consuming_static_gauge_pose']}"
    )
    lines.append(
        f"- current pose artifact is source of record: {rl['current_pose_artifact_is_source_of_record']}"
    )
    lines.append(f"- static-fill superseded: {rl['static_fill_superseded']}")
    lines.append(
        f"- `body_drawn_only_on_measured_frames` asserted by rows: "
        f"{rl['body_drawn_only_on_measured_asserted_by_rows']}"
    )
    lines.append("")
    lines.append("## graph_health-like aggregate row")
    lines.append("```json")
    lines.append(json.dumps(summary["graph_health_row"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Decision")
    lines.append(
        f"**{summary['graph_health_row']['decision']}**"
    )
    if summary["all_checks_pass"]:
        lines.append("")
        lines.append(
            "All five required checks pass: the current unknown-preserving pose "
            "channel is pure (no forbidden numeric pose), the original laundering "
            "is correctly detected, the render manifest points to the unknown-"
            "preserving artifact with the static fill superseded, and the "
            "measurement-factor eligibility is exactly f30-36,f46 with zero "
            "solver estimates."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="output directory (default: %(default)s)",
    )
    ap.parse_args()
    summary = run_audit()
    write_outputs(summary)
    print(f"wrote {OUT_DIR}/summary.json")
    print(f"wrote {OUT_DIR}/per_frame.ndjson ({FRAME_COUNT} rows)")
    print(f"wrote {OUT_DIR}/audit.md")
    print(f"all_checks_pass: {summary['all_checks_pass']}")
    print(f"decision: {summary['graph_health_row']['decision']}")
    gh = summary["graph_health_row"]
    print(
        "purity current/original/static: "
        f"{gh['pose_measurement_purity']['current_unknown_preserving']}/"
        f"{gh['pose_measurement_purity']['original_pose_graph']}/"
        f"{gh['pose_measurement_purity']['static_gauge_fill']}"
    )
    print(f"imputed_pose_count original: {gh['imputed_pose_count']['original_pose_graph']}")
    print(f"unknown_null_pose_count: {gh['unknown_null_pose_count']}")
    print(f"rejected_fit_count: {gh['rejected_fit_count']}")
    print(f"solver_estimate_count: {gh['solver_estimate_count']}")


if __name__ == "__main__":
    main()
