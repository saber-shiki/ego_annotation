# 38 — R0 contract reusable in the build path (wrapper/runbook)

Branch `yiwen_research`. One new additive untracked script
(`scripts/run_ego_hoi_pose_state_contract_on_build.py`). Nothing staged, nothing
committed. CPU-only (stdlib `json`/`hashlib`/`pathlib`/`time`; no model
inference, no GPU). Existing scripts untouched.

## 0. What this is and why

`validate_ego_hoi_pose_state_contract.py` (subagent 34) is the package-generic,
machine-checkable form of the pose-state doctrine (subagent 31): it refuses
held / nearest-hold / interpolated / static-gauge values in a measured-pose
column and separates observation facts, unknown rows, and solver estimates.
Until now it was a **manual** CLI step. The task asked for the smallest additive
integration that makes the contract reusable in the code path so future sidecar
builds automatically run it and record summary hashes.

I evaluated the two options the task offered:

1. **Wire the validator into `build_ego_hoi_sidecar_and_graph_health.py`.**
   Rejected as unsafe/ambiguous. That script is the R0/R1 *scaffold* builder: its
   manifest uses `declared_tables` (a list of bare strings) and emits **no**
   `tables[]` dicts and **no** `render_artifacts`. The validator's checks 1/2/5/6
   require `manifest.tables[]` as dicts with `name`/`status`/`path` plus
   `render_artifacts`, so the scaffold's own output can never pass the contract.
   Running the validator on the scaffold's `out_dir` would always FAIL — not a
   useful or safe integration, and editing the scaffold to satisfy the validator
   is a refactor outside the "smallest additive" scope.
2. **A reusable wrapper/runbook script.** Chosen. The wrapper imports the existing
   validator, runs it on any package root, and records the decision + a
   content-addressed summary hash. Any sidecar builder calls `run(package_root)`
   after producing the package (the runbook is the module docstring).

## 1. The wrapper

`scripts/run_ego_hoi_pose_state_contract_on_build.py`:

- `run(package_root, output_dir=None) -> record` — programmatic entry point
  importable from any builder. Adds `scripts/` to `sys.path` and imports the
  validator module, so it works invoked as a script or imported.
- CLI mirrors the validator: `--package-root`, `--output-dir`.
- Writes `summary.json` + `audit.md` (regenerated from the validator) and appends
  one record to a **dedicated** `validation_log.ndjson` under `output_dir`
  (default `<package-root>/pose_state_schema_contract/`).
- Exit code: 0 satisfied, 1 violation, 2 validator/import error.

The record carries: `schema`, `metric_family`, `validator`, `package_root`,
`manifest_sha256`, `current_pose_table`, `decision`, `all_checks_pass`,
`n_pass`/`n_fail`/`n_checks`, `summary_sha256` (sha256 of the on-disk
`summary.json`), `summary_json_sha256_canonical`, `per_check_status`,
`validator_output_dir`, `produced_at_unix`, `producer`.

## 2. The invariant that drove the design (defect found and fixed during this run)

The validation record is written to a **separate** file under
`pose_state_schema_contract/validation_log.ndjson`, **never** appended to
`tables/graph_health.ndjson`.

Reason: `graph_health.ndjson` is a **manifest-declared table**
(`manifest.tables[]` name=`graph_health`, `row_count=4`). The validator's
check 6 (`tables_rowcount_and_hash_integrity`) enforces that every declared
`row_count` equals the on-disk line count. Appending a contract-validation row
to `graph_health.ndjson` would make actual=5 while declared=4 → a
**self-inflicted `POSE_STATE_CONTRACT_VIOLATION`** on check 6, on the very
package being validated. (I confirmed this empirically: an earlier iteration of
the wrapper did append to `graph_health.ndjson`; the package's own contract then
failed check 6, and the wrapper recorded a violation it had itself caused. The
fix is to keep the log outside the validated `tables/` set.)

This is the report-35 lesson (structure beats flags) applied to evidence files:
the validation log must not live in a location the contract measures.

## 3. Result on clip001850: 7/7 PASS, package unmutated

Package root:
`.../ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/`
(manifest sha256 `916f2a66...`).

| signal | value |
|---|---|
| decision | `pose_state_contract_satisfied` |
| checks | 7/7 PASS |
| summary_sha256 | `82ab13ed272d04aceee7a554ecb4733bb4a4306f23e313dd49dba2fcc4f1734f` |
| graph_health.ndjson lines before/after | 4 / 4 (declared 4) |
| all `tables/*.ndjson` sha256 before/after | byte-identical |
| validator re-run AFTER wrapper | 7/7 PASS (wrapper did not break the package's own contract) |
| reproducibility (summary_sha256 across 2 runs) | identical |

The wrapper writes only under `<package-root>/pose_state_schema_contract/`:
`summary.json`, `audit.md` (regenerated — same validator, equivalent content to
subagent 34's outputs), and `validation_log.ndjson` (new append-only evidence).

## 4. Negative-case discrimination (not a tautology)

A synthetic package with a numeric pose on a non-`is_measured` row (exactly the
nearest-hold/static-gauge laundering of subagents 28/31/33):

- wrapper exit code = 1
- decision = `POSE_STATE_CONTRACT_VIOLATION`, n_fail=3
- check 1 (`1_pose_table_purity`) = FAIL
- record written to `validation_log.ndjson`, **nothing** written into `tables/`

So the wrapper surfaces violations rather than silently passing, and it records
both outcomes durably.

## 5. Scope discipline / honest limits

- **One new additive script.** Zero edits to `validate_ego_hoi_pose_state_contract.py`,
  `build_ego_hoi_sidecar_and_graph_health.py`, or any existing script. The other
  untracked `scripts/*.py` on the branch belong to prior subagents, not this task.
- **No staging, no commits.** Script untracked (`??`); `git diff --cached` empty.
- **No model inference, no GPU, no solver, no pose infill.** Stdlib only.
- The wrapper **does not raise localization coverage** and does not edit pose/
  render state. It records the contract decision on whatever the package declares.
- The wrapper regenerates `summary.json`/`audit.md` in the package's existing
  `pose_state_schema_contract/` dir (same validator → equivalent content to
  subagent 34's manual run). This is the intended behaviour of a build hook. The
  new durable artifact is `validation_log.ndjson`.
- The contract assumes the package's row provenance fields (`is_measured`,
  `pose_source`, `schema`) are honest; a package that lies would pass check 1.
  This is the same boundary as subagent 34 — the wrapper does not change it.
- **Boundary on build integration:** the wrapper is deliberately NOT wired into
  `build_ego_hoi_sidecar_and_graph_health.py`, because that builder's output is a
  graph-health scaffold, not a contract-validatable package (its manifest has no
  `tables[]` dicts / `render_artifacts`). Wiring it there would be a refactor and
  would always fail the contract. The reusable hook is the wrapper itself; the
  clip-specific package builder (`build_clip001850_ego_hoi_sidecar_package.py`)
  or any future generic builder can call `run(package_root)` after producing the
  package.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
PKG=/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research

# CLI (post-build hook)
python3 scripts/run_ego_hoi_pose_state_contract_on_build.py --package-root "$PKG"
# -> pose_state_contract_satisfied (7/7); exit 0; writes $PKG/pose_state_schema_contract/{summary.json,audit.md,validation_log.ndjson}

# programmatic
python3 - <<'PY'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
from run_ego_hoi_pose_state_contract_on_build import run
rec = run(Path("$PKG"))
print(rec["decision"], rec["summary_sha256"])
PY

# prove tables/ unmutated by the wrapper
for f in "$PKG"/tables/*.ndjson; do sha256sum "$f"; done   # compare before/after
```
