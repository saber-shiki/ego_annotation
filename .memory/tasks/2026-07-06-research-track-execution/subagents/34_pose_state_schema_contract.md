# 34 — Reusable pose-state schema contract validator (R0 → package-generic)

Branch `yiwen_research`. One new additive script (`scripts/validate_ego_hoi_pose_state_contract.py`),
nothing staged, nothing committed. CPU-only (stdlib `json`/`hashlib`/`pathlib`; no
numpy, no model inference, no GPU). Durable outputs under
`/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/pose_state_schema_contract/`.

## 0. What this is and why

R0's clip-local cleanup (subagents 30–33) produced a clean `object_pose_observations`
table (numeric pose only on the 8 `observed_measured` frames, null pose on the
142 unknown/rejected/unresolved frames) and a clean unknown-preserving render.
But that cleanup was clip-specific and enforced by reading specific frame sets.
The recurrence risk named in the doctrine (`31_pose_state_semantics_theory.md`)
and surfaced by the fresh-context attack (`33_pose_semantics_attack.md`) is
structural: a held-T_rest stamp, a nearest-hold copy, a Slerp/lerp interpolation,
or an orphan `static_pose_hypothesis` object can re-enter the package because
nothing machine-checks the contract that separates observation facts, unknown
rows, and solver estimates.

This task advances R0 from clip-local cleanup to a **reusable, package-generic
contract validator**. It takes any `ego.hoi` package root and enforces the seven
contract checks without any clip-specific frame set or hardcoded path. It is
instrumentation that routes an intervention: it does not run a solver, does not
block approximate uncertain outputs, and does not edit the package. A legitimate
`solver_estimate` posterior with covariance passes; a held/interpolated/
static-gauge value on an unknown frame fails.

## 1. The validator and the seven contract checks

`scripts/validate_ego_hoi_pose_state_contract.py` is package-root-driven
(`--package-root PATH --output-dir PATH`). It loads `manifest.json` + `tables/`,
identifies the current pose table generically (an entry in `manifest.tables[]`
whose name contains `object_pose`, status is current, and name is not a
prior/held/hypothesis artefact; preferring `observations`), then runs:

1. **pose_table_purity** — the current pose observations table carries numeric
   R/t ONLY on rows that declare an admissible measurement (`is_measured`/`is_localized`
   true, or a measured-flavoured `pose_source` that is not forbidden); unknown/
   rejected/unresolved rows carry null pose. A measured row missing its pose is
   also a violation.
2. **render_consumes_current_pose** — the primary `render_artifacts` entries
   assert `body_drawn_only_on_measured_frames=true` and their declared consumed
   object-pose sha256 equals the on-disk pose table sha256.
3. **no_current_table_names_static_gauge_or_held** — no CURRENT `tables[]`
   descriptor names a static-gauge / held-pose / hypothesis object. (Superseded
   negative evidence under `superseded_artifacts` / `input_hashes` is allowed.)
4. **no_current_static_pose_hypothesis_reference** — no current manifest
   reference to `static_pose_hypothesis` and no `*static_pose_hypothesis*` file
   under the package root.
5. **graph_health_render_lineage_and_purity** — `tables/graph_health.ndjson`
   exists with a non-null `render_state_hash_lineage` (non-null
   `render_state_hash` + `consumed_pose_source_sha256`) and a pose-provenance-
   purity signal.
6. **tables_rowcount_and_hash_integrity** — every `tables[]` file exists; a
   declared numeric `row_count` matches the actual ndjson line count; any declared
   on-table sha256 matches the file digest.
7. **solver_estimate_distinct_type_with_covariance_if_present** — IF any solver-
   estimate pose record exists, it must live in a record type/table distinct from
   the observations table AND carry a posterior covariance. Passes trivially when
   no solver-estimate record is present (forward-looking for future R4/R7 solvers).

## 2. Result on the clip001850 package: 7/7 PASS

Package root: `.../ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/`
(manifest sha256 `70f333b8...`). Decision: **pose_state_contract_satisfied**.

| # | check | status | evidence |
|---|---|---|---|
| 1 | pose_table_purity | PASS | 150 rows; numeric pose only on `[30,31,32,33,34,35,36,46]` (8); forbidden_numeric=[]; measured_missing_pose=[] |
| 2 | render_consumes_current_pose | PASS | pose table sha256 `67dabc79...`; overlay/world/side_by_side all consume it; body_drawn_only_on_measured=true on all three |
| 3 | no_current_table_names_static_gauge_or_held | PASS | violating_current_table_descriptors=[] (7 current tables, none prior/held) |
| 4 | no_current_static_pose_hypothesis_reference | PASS | stray_hypothesis_files=[]; current_manifest_refs=[] |
| 5 | graph_health_render_lineage_and_purity | PASS | render_state_hash_lineage non-null; render_state_hash non-null; consumed_pose_source_sha256 non-null; purity signal present |
| 6 | tables_rowcount_and_hash_integrity | PASS | all 7 tables exist; declared row_counts match (contact_frame_detail 150, render_consumption 9, motion_coupling 1, object_pose_observations 150, object_pose_visibility_ledger 150); "see file" entries exist |
| 7 | solver_estimate_distinct_type_with_covariance | PASS | triggered=False; no solver_estimate pose record present |

## 3. Negative-case discrimination (the validator is not a tautology)

A validator that only ever passes is not a contract. I ran 8 synthetic probes
(temp packages built in-memory, then deleted) where each contract is violated:

| probe | contract violated | expected | got |
|---|---|---|---|
| neg1 forbidden numeric pose on `is_measured=false` (nearest-hold) | check 1 | FAIL | FAIL |
| neg1b measured row missing its pose | check 1 | FAIL | FAIL |
| neg2 render consumes mismatched pose sha | check 2 | FAIL | FAIL |
| neg3 current table named `object_pose_static_gauge` | check 3 | FAIL | FAIL |
| neg4 current manifest reference to static_pose_hypothesis | check 4 | FAIL | FAIL |
| neg5 graph_health null render_state_hash_lineage | check 5 | FAIL | FAIL |
| neg6 declared row_count != actual | check 6 | FAIL | FAIL |
| neg7 solver_estimate record without covariance | check 7 | FAIL | FAIL |
| pos7 solver_estimate WITH covariance in distinct type | check 7 | PASS | PASS |

Every violation FAILS; a legitimate covariance-bearing solver estimate in a
distinct record type PASSES. The check 1 forbidden-numeric case is exactly the
nearest-hold/static-gauge laundering that subagents 28/31/33 found, so the
validator would have flagged the prior-laundering package as a contract
violation had it existed then.

## 4. How this advances R0

R0's output-substrate milestone (`TASK_PACK.md`) requires the `ego.hoi` sidecar
to encode the pose-state ontology: `unknown` is a first-class null-pose row,
observation rows are separated from solver-estimate rows, and a reference
statistic / held value is forbidden in a measured-pose column. This validator
makes that ontology **enforceable beyond clip001850**: any future package is
scored by the same generic rule, and any numeric pose on a non-measured/non-
solver frame, any static-gauge current table, any orphan hypothesis reference,
any null graph-health lineage, any row-count/hash mismatch, or any solver
estimate lacking covariance flips the decision to `POSE_STATE_CONTRACT_VIOLATION`.

It is instrumentation, not an acceptance gate that blocks approximate uncertain
outputs: a legitimate R4/R7 `solver_estimate` with posterior covariance passes,
which is the one honest way to populate previously-unknown frames.

## 5. Scope discipline / honest limits

- **One new additive script.** Zero edits to existing scripts. The pre-existing
  dirty `scripts/*.py` on the branch are not mine.
- **No staging, no commits.** Script untracked (`??`); `git diff --cached` empty;
  on `yiwen_research`.
- **No model inference, no GPU, no smoothing, no pose infill.** Stdlib only.
- The validator **does not raise localization coverage**. It checks the contract
  on whatever the package declares; it does not add observations. clip001850
  remains 8/150 observed, 137/150 unknown.
- The validator reads the package's own declared fields (`is_measured`,
  `is_localized`, `pose_source`, `schema`) to classify rows. A malicious package
  that lies (`is_measured=true` on a held value) would pass check 1 — the
  contract assumes the row's provenance fields are honest. The provenance-purity
  audit (subagent 32) cross-checks the *source* of the rows against an external
  reference; this contract validator is complementary, not a replacement for it.
- The clip001850 PASS reflects that the parent applied the corrections surfaced
  by subagent 33 (repoint primary render to unknown-preserving; move static-gauge
  tables out of `tables[]` into `superseded_prior_laundering/`; delete orphan
  `static_pose_hypothesis.json`; wire graph-health render lineage + purity). The
  validator confirms those corrections hold as a generic contract, not just a
  clip-local state.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
PKG=/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research
OUT=/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/pose_state_schema_contract
python3 scripts/validate_ego_hoi_pose_state_contract.py --package-root "$PKG" --output-dir "$OUT"
# -> pose_state_contract_satisfied (7/7); writes $OUT/summary.json + $OUT/audit.md
```
