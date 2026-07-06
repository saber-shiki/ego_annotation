#!/usr/bin/env python3
"""
Reusable post-build hook: run the R0 pose-state contract validator on an
`ego.hoi` package and record its decision + summary hash into a dedicated,
non-validated evidence log under the package.

Purpose (R0 promotion: contract reusable in the code path, not only manual)
---------------------------------------------------------------------------
`validate_ego_hoi_pose_state_contract.py` (subagent 34) is the package-generic
machine-checkable form of the pose-state doctrine (subagent 31): it separates
observation facts, unknown rows, and solver estimates, and refuses held /
nearest-hold / interpolated / static-gauge values in a measured-pose column.
Until now it was a manual CLI step. This wrapper makes it part of the build
path: any sidecar builder calls `run(package_root)` (or the CLI) after producing
the package, and the contract decision plus a content-addressed summary hash are
written durably into `<package-root>/pose_state_schema_contract/validation_log.ndjson`.

IMPORTANT (invariant): the validation record is written to a SEPARATE evidence
file, NEVER appended to a table the validator itself checks. The validator's
check 6 enforces that every declared `manifest.tables[]` row_count matches the
on-disk line count, so appending to `tables/graph_health.ndjson` would break the
very contract being validated (a self-inflicted row-count violation). The log
lives under `pose_state_schema_contract/` alongside `summary.json`/`audit.md`,
outside the validated `tables/` set.

This is instrumentation that routes an intervention. It does NOT run a solver,
does NOT edit the package's pose/render/tables, and does NOT block approximate
uncertain outputs: a future R4/R7 `solver_estimate` with posterior covariance
passes the contract and is recorded as such. A violation is surfaced as a log
row + non-zero exit code, never a silent skip.

What it records (append-only validation_log.ndjson row)
-------------------------------------------------------
  schema                         ego_hoi_pose_state_contract_validation/v1
  metric_family                  pose_state_contract_validation
  validator                      ego_hoi_pose_state_contract_validator/v1
  package_root                   absolute path of the validated package
  manifest_sha256                sha256 of the package manifest.json (from validator)
  decision                       pose_state_contract_satisfied | POSE_STATE_CONTRACT_VIOLATION
  all_checks_pass                bool
  n_pass / n_fail / n_checks     int
  summary_sha256                 sha256 of the validator summary.json file
                                 (content address of the full per-check evidence)
  per_check_status               {check_name: PASS|FAIL}
  validator_output_dir           where summary.json + audit.md were written
  produced_at_unix               wall-clock timestamp
  producer                       this script + version

Usage
-----
  # CLI (post-build, automatic in a build pipeline):
  python3 scripts/run_ego_hoi_pose_state_contract_on_build.py \
      --package-root .../org.ego.hoi/0.1.0/<run_id>/

  # programmatic (from inside a sidecar builder):
  from run_ego_hoi_pose_state_contract_on_build import run
  rec = run(package_root=Path(".../<run_id>/"))

Inputs
------
  --package-root PATH     ego.hoi package root (holds manifest.json + tables/)
  --output-dir PATH       where the validator writes summary.json + audit.md AND
                          the append-only validation_log.ndjson
                          (default: <package-root>/pose_state_schema_contract)

CPU-only (stdlib json/hashlib/pathlib/time). No model inference, no GPU.

Exit code: 0 if the contract is satisfied, 1 on violation, 2 on validator error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "ego_hoi_pose_state_contract_validation/v1"
PRODUCER = "run_ego_hoi_pose_state_contract_on_build.py"
PRODUCER_VERSION = "r0-build-hook-1"

# Import the validator from the sibling module in scripts/. We add this file's
# directory to sys.path so the import works whether invoked as a script or
# imported from another module.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Late import inside run() so `--help` and import-time errors are clear.
_VALIDATOR_MODULE = "validate_ego_hoi_pose_state_contract"


def _load_validator():
    """Import the validator module from the scripts/ directory."""
    try:
        mod = __import__(_VALIDATOR_MODULE)
    except ImportError as e:  # pragma: no cover - environment error path
        raise SystemExit(
            f"could not import {_VALIDATOR_MODULE} from {_SCRIPTS_DIR}: {e}"
        )
    return mod


def _sha256_canonical(obj: Any) -> str:
    """Stable content hash of a JSON-serializable object (sorted keys)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_record(
    *,
    result: dict[str, Any],
    package_root: Path,
    output_dir: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Construct the validation log record from a validator result dict."""
    per_check_status = {c["check"]: c["status"] for c in result.get("checks", [])}
    return {
        "schema": SCHEMA,
        "metric_family": "pose_state_contract_validation",
        "producer": PRODUCER,
        "producer_version": PRODUCER_VERSION,
        "validator": result.get("validator", "ego_hoi_pose_state_contract_validator/v1"),
        "package_root": str(package_root.resolve()),
        "manifest_sha256": result.get("manifest_sha256"),
        "current_pose_table": result.get("current_pose_table"),
        "decision": result.get("decision"),
        "all_checks_pass": bool(result.get("all_checks_pass")),
        "n_pass": int(result.get("n_pass", 0)),
        "n_fail": int(result.get("n_fail", 0)),
        "n_checks": int(result.get("n_checks", 0)),
        "summary_sha256": _sha256_file(summary_path),
        "summary_json_sha256_canonical": _sha256_canonical(result),
        "per_check_status": per_check_status,
        "validator_output_dir": str(output_dir.resolve()),
        "produced_at_unix": time.time(),
        "note": (
            "R0 contract validation record. Append-only evidence: does not edit "
            "pose/render tables and does not gate approximate uncertain outputs. "
            "A solver_estimate posterior with covariance passes the contract."
        ),
    }


def run(
    package_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the pose-state contract validator on a package and record the result.

    Writes validator summary.json + audit.md and appends one record row to
    ``validation_log.ndjson`` in ``output_dir`` (default
    ``<package-root>/pose_state_schema_contract``). The log is deliberately
    kept OUTSIDE ``tables/`` so it cannot perturb the row-count/hash invariants
    the validator's check 6 enforces on declared tables.

    Returns the record dict. Does NOT modify any file under ``tables/``.
    """
    pkg = package_root.resolve()
    if not (pkg / "manifest.json").exists():
        raise SystemExit(f"manifest.json not found under package root: {pkg}")

    out = (output_dir or (pkg / "pose_state_schema_contract")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    mod = _load_validator()
    result = mod.validate(pkg)

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    audit_path = out / "audit.md"
    if hasattr(mod, "render_audit"):
        audit_path.write_text(mod.render_audit(result), encoding="utf-8")

    record = build_record(
        result=result, package_root=pkg, output_dir=out, summary_path=summary_path
    )

    # Append-only evidence log, OUTSIDE tables/ so it never perturbs the
    # row-count/hash invariants the validator checks on declared tables.
    log_path = out / "validation_log.ndjson"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    return record


def _print_report(record: dict[str, Any]) -> None:
    print(f"[{record['decision']}] {record['n_pass']}/{record['n_checks']} checks PASS")
    for check, status in record["per_check_status"].items():
        print(f"  [{status}] {check}")
    print(f"summary_sha256: {record['summary_sha256']}")
    print(f"validator output: {record['validator_output_dir']}")
    print(f"package_root: {record['package_root']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_ego_hoi_pose_state_contract_on_build.py",
        description=(
            "Run the R0 pose-state contract validator on an ego.hoi package and "
            "record the decision + summary hash into an append-only "
            "validation_log.ndjson under the package (outside tables/, so it "
            "never perturbs the contract's row-count/hash invariants). "
            "Reusable post-build hook: instrumentation, not a gate."
        ),
    )
    ap.add_argument(
        "--package-root",
        required=True,
        help="ego.hoi package root (holds manifest.json + tables/)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="where the validator writes summary.json + audit.md AND the "
        "append-only validation_log.ndjson "
        "(default: <package-root>/pose_state_schema_contract)",
    )
    args = ap.parse_args(argv)

    record = run(
        package_root=Path(args.package_root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    _print_report(record)

    if not record["all_checks_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - validator/import error path
        print(f"ERROR: validator could not complete: {exc}", file=sys.stderr)
        sys.exit(2)
