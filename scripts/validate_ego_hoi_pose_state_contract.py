#!/usr/bin/env python3
"""
Reusable pose-state schema contract validator for `ego.hoi` packages.

Purpose (D5/D6 mechanism: prior-laundering recurrence prevention)
----------------------------------------------------------------
The pose-state doctrine (`31_pose_state_semantics_theory.md`) separates three
kinds of object-pose record that must never be conflated:

  * observation facts   -> numeric R/t admissible, only on `observed_measured`
                           frames (own visible surface produced a metric fit
                           that survived gating).
  * unknown rows        -> NO numeric pose (null/absent). The discriminator
                           between truth and laundering is provenance, not
                           styling: a value plus a "do not trust me" flag is
                           still read as a value by every consumer that keys on
                           the pose column, so absence of a value is the only
                           representation of unknown that cannot be laundered.
  * solver estimates    -> a distinct record type carrying a live posterior
                           with declared priors and posterior covariance. Never
                           a re-consumed measurement; never mixed into the
                           observation table as a clean measurement.

This script makes that contract machine-checkable for ANY `ego.hoi` package, not
just clip001850. It takes a package root (the directory holding `manifest.json`
plus a `tables/` tree, e.g. `.../org.ego.hoi/0.1.0/<run_id>/`) and enforces the
seven-contract checks below. It is instrumentation that routes an intervention:
it does not run a solver, does not block approximate uncertain outputs, and does
not edit the package. A legitimate `solver_estimate` posterior with covariance
passes; a held/interpolated/static-gauge value on an unknown frame fails.

The seven contract checks
-------------------------
  1. pose_table_purity
     The current object-pose observations table exists and carries numeric R/t
     ONLY on rows that declare an admissible measurement (`observed_measured`);
     unknown / rejected / unresolved rows carry null pose. A measured row that is
     missing its pose is also a violation.
  2. render_consumes_current_pose
     The current (primary) `render_artifacts` entries consume the current pose
     observations (their declared consumed-pose sha256 equals the on-disk pose
     table sha256) and assert `body_drawn_only_on_measured_frames=true`.
  3. no_current_table_names_static_gauge_or_held
     No CURRENT table descriptor (an entry in `manifest.tables[]`) names a
     static-gauge / held-pose / static-pose-hypothesis object. Superseded
     negative evidence recorded under `superseded_artifacts` / `input_hashes` is
     allowed (it is not a current table descriptor).
  4. no_current_static_pose_hypothesis_reference
     No CURRENT artifact or manifest reference to `static_pose_hypothesis`, and
     no `*static_pose_hypothesis*` file under the package root. (Superseded
     references are allowed but noted.)
  5. graph_health_render_lineage_and_purity
     `tables/graph_health.ndjson` exists and its row carries a non-null
     `render_state_hash_lineage` (with a non-null `render_state_hash` and a
     consumed pose-source sha256) plus a pose-provenance-purity signal.
  6. tables_rowcount_and_hash_integrity
     For every `manifest.tables[]` entry, the referenced file exists; when a
     numeric `row_count` is declared it matches the actual ndjson line count;
     any declared on-table sha256 matches the file digest.
  7. solver_estimate_distinct_type_with_covariance_if_present
     IF any solver-estimate pose record exists (a row that carries numeric pose
     AND declares `solver_estimate` provenance), THEN it must live in a record
     type/table distinct from the observations table AND carry a posterior
     covariance. When no solver-estimate record exists the check passes
     trivially (forward-looking: future R4/R7 solvers).

Inputs
------
  --package-root PATH   ego.hoi package root (holds manifest.json + tables/)
  --output-dir  PATH    where to write summary.json + audit.md

CPU-only (stdlib json/hashlib/pathlib). No model inference, no GPU.

Outputs
-------
  <output-dir>/summary.json   per-check verdicts, counts, and the package digest
  <output-dir>/audit.md       human-readable audit report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

VALIDATOR = "ego_hoi_pose_state_contract_validator/v1"

# Patterns that mark a record as a prior/hold artefact, never a current table.
PRIOR_TABLE_PATTERNS = ("static_gauge", "static_pose_hypothesis", "held_pose", "held_rest")
HYPOTHESIS_PATTERNS = ("static_pose_hypothesis",)

# Patterns that mark a pose row as a forbidden (non-admissible) provenance, even
# if it somehow carries numeric pose in the observations table.
FORBIDDEN_POSE_SOURCE_PATTERNS = (
    "static_gauge",
    "held",
    "nearest",
    "interp",
    "imputed",
    "unknown_no_numeric_pose",
)

# Provenance patterns that mark a numeric pose as an admissible solver estimate.
SOLVER_ESTIMATE_PATTERNS = ("solver_estimate", "solver estimate", "posterior")


# --------------------------------------------------------------------------- IO
def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def file_nonblank_lines(path: Path) -> int:
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# --------------------------------------------------------------- classification
def pose_rotation(row: dict[str, Any]) -> Any:
    """Return the rotation value (matrix) declared on the row, or None."""
    for k in (
        "rotation_world_from_completed_canonical_matrix",
        "rotation_world_m",
        "rotation_world",
        "rotation",
        "R",
    ):
        if row.get(k) is not None:
            return row.get(k)
    return None


def pose_translation(row: dict[str, Any]) -> Any:
    for k in (
        "translation_world_m",
        "translation_world",
        "translation",
        "t",
    ):
        if row.get(k) is not None:
            return row.get(k)
    return None


def has_numeric_pose(row: dict[str, Any]) -> bool:
    return pose_rotation(row) is not None and pose_translation(row) is not None


def is_solver_estimate_record(row: dict[str, Any]) -> bool:
    """A row that declares a solver-estimate provenance (not an audit scalar)."""
    schema = str(row.get("schema", ""))
    src = str(row.get("pose_source", "")) + " " + str(row.get("provenance", ""))
    if any(p in schema.lower() for p in SOLVER_ESTIMATE_PATTERNS):
        return True
    if any(p in src.lower() for p in SOLVER_ESTIMATE_PATTERNS):
        return True
    return False


def declares_measured(row: dict[str, Any]) -> bool:
    """A row that declares an admissible measurement (observed_measured).

    The cleanest contract fields are is_measured / is_localized; fall back to a
    measured-flavoured pose_source that is not forbidden and not solver.
    """
    if row.get("is_measured") is True:
        return True
    if row.get("is_localized") is True:
        return True
    src = str(row.get("pose_source", "")).lower()
    if "measured" in src or "observed_icp" in src:
        if not any(p in src for p in FORBIDDEN_POSE_SOURCE_PATTERNS):
            return True
    return False


def covariance_value(row: dict[str, Any]) -> Any:
    for k in row:
        kl = k.lower()
        if "covariance" in kl or kl in ("posterior_cov", "cov", "sigma"):
            v = row[k]
            if v is not None:
                return v
    return None


# --------------------------------------------------------- current-table lookup
def _is_current_status(status: str | None) -> bool:
    if not status:
        return True
    s = str(status).lower()
    return not (
        s.startswith("superseded")
        or "moved_out" in s
        or s == "placeholder_not_produced"
        or s == "rejected"
    )


def find_current_pose_table(manifest: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Identify the current object-pose observations table descriptor.

    Generic rule over manifest.tables[]: name contains 'object_pose', status is
    current, name is not a prior/held/hypothesis artefact. Prefer a name
    containing 'observations'. Returns (name, descriptor) or (None, None).
    """
    candidates: list[tuple[str, dict[str, Any], int]] = []
    for t in manifest.get("tables", []):
        name = str(t.get("name", ""))
        if "object_pose" not in name:
            continue
        if not _is_current_status(t.get("status")):
            continue
        lname = name.lower()
        if any(p in lname for p in PRIOR_TABLE_PATTERNS):
            continue
        # rank: prefer 'observations'
        rank = 0 if "observations" in lname else 1
        candidates.append((name, t, rank))
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[2])
    name, desc, _ = candidates[0]
    return name, desc


# --------------------------------------------------------------------- checks
def check_1_pose_table_purity(
    pkg: Path, pose_name: str | None, pose_desc: dict[str, Any] | None
) -> dict[str, Any]:
    cur = pose_desc is not None and pose_name is not None
    detail: dict[str, Any] = {
        "check": "1_pose_table_purity",
        "pose_table": pose_name,
        "current_descriptor_found": cur,
    }
    if not cur:
        return _fail(detail, "no current object-pose observations table descriptor found")

    rel = str(pose_desc.get("path", f"tables/{pose_name}.ndjson"))
    pose_file = _resolve_path(pkg, rel)
    if not pose_file.exists():
        return _fail(detail, f"pose table file missing: {pose_file}")
    rows = load_ndjson(pose_file)
    numeric_rows: list[int] = []
    forbidden_numeric: list[int] = []
    measured_missing_pose: list[int] = []
    for r in rows:
        fi = r.get("frame_idx")
        num = has_numeric_pose(r)
        measured = declares_measured(r)
        solver = is_solver_estimate_record(r)
        if num:
            numeric_rows.append(fi)
            if not (measured or solver):
                forbidden_numeric.append(fi)
        else:
            # no numeric pose
            if measured and not solver:
                # a measured observation must carry its pose
                measured_missing_pose.append(fi)
    detail.update(
        {
            "row_count": len(rows),
            "numeric_pose_rows": numeric_rows,
            "numeric_pose_count": len(numeric_rows),
            "forbidden_numeric_pose_rows": forbidden_numeric,
            "measured_rows_missing_pose": measured_missing_pose,
        }
    )
    ok = (len(forbidden_numeric) == 0) and (len(measured_missing_pose) == 0)
    return _pass(detail) if ok else _fail(
        detail,
        f"forbidden numeric pose on {len(forbidden_numeric)} non-measured row(s)"
        + (f"; measured row(s) missing pose: {measured_missing_pose}" if measured_missing_pose else ""),
    )


def check_2_render_consumes_current_pose(
    pkg: Path, manifest: dict[str, Any], pose_desc: dict[str, Any] | None, pose_name: str | None
) -> dict[str, Any]:
    ra = manifest.get("render_artifacts")
    detail: dict[str, Any] = {"check": "2_render_consumes_current_pose"}
    if not isinstance(ra, dict) or not ra:
        return _fail(detail, "manifest.render_artifacts missing or not a dict")
    if pose_desc is None or pose_name is None:
        return _fail(detail, "cannot verify consumption: no current pose table")
    rel = str(pose_desc.get("path", f"tables/{pose_name}.ndjson"))
    pose_file = _resolve_path(pkg, rel)
    if not pose_file.exists():
        return _fail(detail, f"pose table file missing: {pose_file}")
    pose_sha = sha256_file(pose_file)
    body_flag_ok: list[str] = []
    body_flag_bad: list[str] = []
    consume_ok: list[str] = []
    consume_bad: list[str] = []
    for view, ent in ra.items():
        if not isinstance(ent, dict):
            continue
        # body-draw flag
        bflag = ent.get("body_drawn_only_on_measured_frames")
        if bflag is True:
            body_flag_ok.append(view)
        else:
            body_flag_bad.append(view)
        # consumed pose sha: any key tying object_pose to a sha256
        consumed_sha: str | None = None
        for k, v in ent.items():
            if "object_pose" in k and "sha256" in k and isinstance(v, str):
                consumed_sha = v
                break
        if consumed_sha is None and isinstance(ent.get("consumed_object_pose_observations_sha256"), str):
            consumed_sha = ent["consumed_object_pose_observations_sha256"]
        if consumed_sha is None:
            consume_bad.append(f"{view}:no_consumed_pose_sha")
        elif consumed_sha == pose_sha:
            consume_ok.append(view)
        else:
            consume_bad.append(f"{view}:sha_mismatch({consumed_sha[:12]}!=pose {pose_sha[:12]})")
    detail.update(
        {
            "pose_table_sha256": pose_sha,
            "body_drawn_only_on_measured_views_ok": body_flag_ok,
            "body_drawn_only_on_measured_views_bad": body_flag_bad,
            "consume_sha_ok": consume_ok,
            "consume_sha_bad": consume_bad,
        }
    )
    ok = (not body_flag_bad) and (not consume_bad)
    return _pass(detail) if ok else _fail(
        detail,
        f"body flag bad on {body_flag_bad}" + f"; consume mismatch: {consume_bad}" if consume_bad else "",
    )


def check_3_no_current_table_names_static_gauge(manifest: dict[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {"check": "3_no_current_table_names_static_gauge_or_held"}
    bad: list[str] = []
    for t in manifest.get("tables", []):
        name = str(t.get("name", ""))
        status = t.get("status")
        lname = name.lower()
        if any(p in lname for p in PRIOR_TABLE_PATTERNS) and _is_current_status(status):
            bad.append(f"{name}(status={status})")
    detail["violating_current_table_descriptors"] = bad
    return _pass(detail) if not bad else _fail(detail, f"current tables[] name prior/held artefacts: {bad}")


def check_4_no_current_static_pose_hypothesis(pkg: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {"check": "4_no_current_static_pose_hypothesis_reference"}
    # (a) no file under package root named *static_pose_hypothesis*
    stray_files = [str(p.relative_to(pkg)) for p in pkg.rglob("*") if any(q in p.name.lower() for q in HYPOTHESIS_PATTERNS)]
    # (b) no current (non-superseded) manifest reference
    current_refs: list[str] = []

    def _scan(obj: Any, path: str, in_superseded: bool) -> None:
        if isinstance(obj, dict):
            sub_super = in_superseded
            # entering a clearly-superseded container
            if any(tok in path.lower() for tok in ("superseded_artifacts", "input_hashes", "superseded_render")):
                sub_super = True
            for k, v in obj.items():
                if not sub_super and any(q in str(k).lower() for q in HYPOTHESIS_PATTERNS):
                    current_refs.append(f"{path}.{k}")
                if not sub_super and isinstance(v, str) and any(q in v.lower() for q in HYPOTHESIS_PATTERNS):
                    current_refs.append(f"{path}.{k}=str")
                _scan(v, f"{path}.{k}", sub_super)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _scan(v, f"{path}[{i}]", in_superseded)

    _scan(manifest, "manifest", in_superseded=False)
    detail["stray_hypothesis_files"] = stray_files
    detail["current_manifest_hypothesis_refs"] = current_refs
    ok = (not stray_files) and (not current_refs)
    return _pass(detail) if ok else _fail(
        detail,
        f"stray files: {stray_files}; current refs: {current_refs}",
    )


def check_5_graph_health_lineage_and_purity(pkg: Path) -> dict[str, Any]:
    gh = pkg / "tables" / "graph_health.ndjson"
    detail: dict[str, Any] = {"check": "5_graph_health_render_lineage_and_purity", "graph_health_path": str(gh)}
    if not gh.exists():
        return _fail(detail, "tables/graph_health.ndjson missing")
    rows = load_ndjson(gh)
    if not rows:
        return _fail(detail, "graph_health.ndjson empty")
    row = rows[0]
    lineage = row.get("render_state_hash_lineage")
    problems: list[str] = []
    if lineage is None:
        problems.append("render_state_hash_lineage is null/absent")
        lineage_ok = False
    else:
        rsh = lineage.get("render_state_hash") if isinstance(lineage, dict) else None
        csha = (
            lineage.get("consumed_pose_source_sha256")
            if isinstance(lineage, dict)
            else None
        )
        lineage_ok = (rsh is not None) and (csha is not None)
        if rsh is None:
            problems.append("render_state_hash null")
        if csha is None:
            problems.append("consumed_pose_source_sha256 null")
    # purity signal
    purity_present = False
    if isinstance(lineage, dict):
        if lineage.get("pose_provenance_purity_summary") or lineage.get("pose_purity_decision"):
            purity_present = True
    if not purity_present:
        for k in row:
            if "pur" in k.lower() and "pose" in k.lower():
                purity_present = True
                break
    detail.update(
        {
            "render_state_hash_lineage_nonnull": lineage is not None,
            "render_state_hash_nonnull": isinstance(lineage, dict) and lineage.get("render_state_hash") is not None,
            "consumed_pose_source_sha256_nonnull": isinstance(lineage, dict) and lineage.get("consumed_pose_source_sha256") is not None,
            "pose_provenance_purity_signal_present": purity_present,
        }
    )
    ok = lineage_ok and purity_present
    return _pass(detail) if ok else _fail(detail, "; ".join(problems) + ("; no purity signal" if not purity_present else ""))


def check_6_tables_rowcount_hash_integrity(pkg: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {"check": "6_tables_rowcount_and_hash_integrity", "per_table": []}
    problems: list[str] = []
    for t in manifest.get("tables", []):
        name = str(t.get("name", ""))
        rel = str(t.get("path", f"tables/{name}.ndjson"))
        f = _resolve_path(pkg, rel)
        entry: dict[str, Any] = {"name": name, "path": rel}
        if not f.exists():
            entry["exists"] = False
            problems.append(f"{name}: file missing ({rel})")
            detail["per_table"].append(entry)
            continue
        entry["exists"] = True
        rc = t.get("row_count")
        if isinstance(rc, int):
            actual = file_nonblank_lines(f)
            entry["declared_row_count"] = rc
            entry["actual_row_count"] = actual
            if actual != rc:
                problems.append(f"{name}: row_count {rc} != actual {actual}")
        else:
            entry["declared_row_count"] = rc  # e.g. "see file"
        # any on-table sha256 field (key ending in sha256 whose value points at this file)
        sha_keys = {k: v for k, v in t.items() if k.lower().endswith("sha256") and isinstance(v, str)}
        if sha_keys:
            actual_sha = sha256_file(f)
            entry["sha_checks"] = {}
            for k, v in sha_keys.items():
                ok_sha = v == actual_sha
                entry["sha_checks"][k] = {"declared": v, "match": ok_sha}
                if not ok_sha:
                    problems.append(f"{name}: {k} mismatch")
        detail["per_table"].append(entry)
    return _pass(detail) if not problems else _fail(detail, "; ".join(problems))


def check_7_solver_estimate_distinct_with_covariance(pkg: Path, manifest: dict[str, Any], pose_name: str | None) -> dict[str, Any]:
    detail: dict[str, Any] = {"check": "7_solver_estimate_distinct_type_with_covariance_if_present"}
    solver_records: list[dict[str, Any]] = []
    for t in manifest.get("tables", []):
        name = str(t.get("name", ""))
        rel = str(t.get("path", f"tables/{name}.ndjson"))
        f = _resolve_path(pkg, rel)
        if not f.exists() or not f.suffix == ".ndjson":
            continue
        for r in load_ndjson(f):
            if is_solver_estimate_record(r) and has_numeric_pose(r):
                solver_records.append({"table": name, "frame_idx": r.get("frame_idx"), "schema": r.get("schema")})
    detail["solver_estimate_pose_records"] = solver_records
    if not solver_records:
        detail["triggered"] = False
        detail["note"] = "no solver_estimate pose record present; check passes trivially (forward-looking)"
        return _pass(detail)
    detail["triggered"] = True
    problems: list[str] = []
    pose_schema = None
    if pose_name:
        pf = _resolve_path(pkg, f"tables/{pose_name}.ndjson")
        if pf.exists():
            prows = load_ndjson(pf)
            if prows:
                pose_schema = prows[0].get("schema")
    detail["observations_schema"] = pose_schema
    for rec in solver_records:
        # locate the actual row to test covariance + distinct schema
        tbl = rec["table"]
        tf = _resolve_path(pkg, f"tables/{tbl}.ndjson")
        target = None
        for r in load_ndjson(tf):
            if r.get("frame_idx") == rec["frame_idx"] and is_solver_estimate_record(r):
                target = r
                break
        if target is None:
            continue
        cov = covariance_value(target)
        rec["has_posterior_covariance"] = cov is not None
        rec["record_schema"] = target.get("schema")
        rec["distinct_table_from_observations"] = tbl != pose_name
        rec["distinct_schema_from_observations"] = (pose_schema is None) or (target.get("schema") != pose_schema)
        if cov is None:
            problems.append(f"{tbl}[{rec['frame_idx']}]: no posterior covariance")
        if pose_schema is not None and target.get("schema") == pose_schema and tbl == pose_name:
            problems.append(f"{tbl}[{rec['frame_idx']}]: solver_estimate mixed into observations table")
    return _pass(detail) if not problems else _fail(detail, "; ".join(problems))


# --------------------------------------------------------------------- helpers
def _resolve_path(pkg: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return pkg / rel


def _pass(detail: dict[str, Any]) -> dict[str, Any]:
    detail["status"] = "PASS"
    return detail


def _fail(detail: dict[str, Any], reason: str) -> dict[str, Any]:
    detail["status"] = "FAIL"
    detail["reason"] = reason
    return detail


# ----------------------------------------------------------------------- main
def validate(package_root: Path) -> dict[str, Any]:
    pkg = package_root.resolve()
    manifest_path = pkg / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found under package root: {pkg}")
    manifest = load_json(manifest_path)
    pose_name, pose_desc = find_current_pose_table(manifest)

    checks = [
        check_1_pose_table_purity(pkg, pose_name, pose_desc),
        check_2_render_consumes_current_pose(pkg, manifest, pose_desc, pose_name),
        check_3_no_current_table_names_static_gauge(manifest),
        check_4_no_current_static_pose_hypothesis(pkg, manifest),
        check_5_graph_health_lineage_and_purity(pkg),
        check_6_tables_rowcount_hash_integrity(pkg, manifest),
        check_7_solver_estimate_distinct_with_covariance(pkg, manifest, pose_name),
    ]
    all_pass = all(c["status"] == "PASS" for c in checks)
    result = {
        "validator": VALIDATOR,
        "package_root": str(pkg),
        "manifest_sha256": sha256_file(manifest_path),
        "current_pose_table": pose_name,
        "all_checks_pass": all_pass,
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c["status"] == "PASS"),
        "n_fail": sum(1 for c in checks if c["status"] == "FAIL"),
        "decision": "pose_state_contract_satisfied" if all_pass else "POSE_STATE_CONTRACT_VIOLATION",
        "checks": checks,
    }
    return result


def render_audit(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# ego.hoi pose-state schema contract audit")
    lines.append("")
    lines.append(f"- validator: `{result['validator']}`")
    lines.append(f"- package root: `{result['package_root']}`")
    lines.append(f"- manifest sha256: `{result['manifest_sha256']}`")
    lines.append(f"- current pose table: `{result['current_pose_table']}`")
    lines.append(f"- decision: **{result['decision']}** ({result['n_pass']}/{result['n_checks']} checks PASS)")
    lines.append("")
    for c in result["checks"]:
        icon = "PASS" if c["status"] == "PASS" else "FAIL"
        lines.append(f"## [{icon}] {c['check']}")
        for k, v in c.items():
            if k in ("check", "status", "reason"):
                continue
            lines.append(f"- {k}: `{_truncate(v)}`")
        if c["status"] == "FAIL":
            lines.append(f"- reason: {c.get('reason')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _truncate(v: Any, limit: int = 120) -> str:
    s = json.dumps(v) if not isinstance(v, str) else v
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate an ego.hoi package against the pose-state schema contract.")
    ap.add_argument(
        "--package-root",
        required=True,
        help="ego.hoi package root (holds manifest.json + tables/)",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="where to write summary.json + audit.md",
    )
    args = ap.parse_args()

    pkg = Path(args.package_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = validate(pkg)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "audit.md").write_text(render_audit(result))

    print(f"[{result['decision']}] {result['n_pass']}/{result['n_checks']} checks PASS")
    for c in result["checks"]:
        print(f"  [{c['status']}] {c['check']}" + (f" -- {c.get('reason')}" if c["status"] == "FAIL" else ""))
    print(f"summary: {out / 'summary.json'}")
    print(f"audit:   {out / 'audit.md'}")


if __name__ == "__main__":
    main()
