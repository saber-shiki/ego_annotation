# 32 — Pose provenance-purity audit for clip001850 (R0/R1)

Branch `yiwen_research`. One new additive script, nothing staged, nothing
committed. CPU-only (stdlib `json`/`hashlib`/`pathlib`; no numpy, no model
inference, no GPU).

Script: `scripts/audit_clip001850_pose_provenance_purity.py`
Outputs (all under
`/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/pose_provenance_purity_audit/`):
`summary.json`, `per_frame.ndjson` (150 rows), `audit.md`.

## 0. What this is and why

The static-fill / nearest-hold / interpolation laundering error (subagents
28/31) can recur unnoticed unless the pose state is audited for provenance
purity. The discriminator between truth and laundering is **provenance**, not
styling: a frame with no admissible observation must carry **no pose value at
all**, because every consumer that keys on the pose column places a metric body
from `translation_world_m` and reads no visibility flag (verified:
`render_v19_contact_state_full_duration.py::observed_body_world` reads only
`rotation_world_from_completed_canonical_matrix` and `translation_world_m`;
grep for `visibility_state|is_measured|is_localized` across both delivered
consumers returns empty — subagent 31 finding A).

This audit reads the on-disk pose artifacts and enforces the five-class
ontology from `31_pose_state_semantics_theory.md`:

| class | numeric pose stored? | clip001850 frames | count |
|---|---|---|---|
| observed_measured | **yes** — the fit | 30,31,32,33,34,35,36,46 | 8 |
| observed_rejected | retained for audit only, NOT admissible | 60,75,76,77 | 4 |
| unresolved | retained, NOT admissible | 45 | 1 |
| unknown | **no** — null/absent | 0-29,37-44,47-59,61-74,78-149 | 137 |
| solver_estimate | yes — a live posterior | (none — P15 inert) | 0 |

Sum = 150. The audit treats `observed_rejected`/`unresolved`/`unknown` as
**forbidden** to carry numeric pose in the current state: a value on any of
them is laundering.

## 1. The five required checks (all PASS)

```
[1 PASS] current_no_forbidden_numeric_pose
    forbidden_numeric_count = 0   forbidden_numeric = []
[2 PASS] original_pose_graph_fails_purity
    imputed_pose_count = 137      forbidden_numeric_count = 142
    purity = 0.053333
    (PASS = the audit correctly DETECTS the laundering; a clean original
     would be a failure of detection.)
[3 PASS] render_manifest_points_to_unknown_preserving
    current_is_source_of_record = True   static_fill_superseded = True
[4 PASS] measurement_factor_eligibility_and_solver_estimate
    eligible_frames = [30,31,32,33,34,35,36,46]   solver_estimate_count = 0
    solver_inert = True (nfev=1, cost=0.0, max translation delta=0.0 m)
[5 PASS] current_render_body_drawn_only_on_measured
    asserted_by_rows = [6,7,8]   (body_drawn_only_on_measured_frames=True)
```

`all_checks_pass = True`. Decision:
`pose_channel_pure_current__original_laundering_detected`.

## 2. Purity by artifact

| artifact | total numeric | admissible numeric | purity | imputed | forbidden numeric |
|---|---|---|---|---|---|
| current unknown-preserving | 8 | 8 | **1.0** | 0 | 0 |
| original P15 pose graph | 150 | 8 | **0.0533** | 137 | 142 |
| static-gauge fill (superseded) | 150 | 8 | **0.0533** | 142 | 142 |

- The current unknown-preserving pose channel is **pure**: numeric R/t exists
  only on the 8 admissible measured frames (f30-36,46); all 137 unknown frames
  carry null pose; f45/f60/f75/f76/f77 carry null pose (no value laundered as a
  measurement).
- The original P15 pose graph is **contaminated**: 137 frames are
  nearest_visible_pose_hold (102) or interpolated_between_visible_pose_observations
  (35) — inherited local patches written into `translation_world_m`. A further
  5 non-admissible direct-fit frames (f45 unresolved, f60/75/76/77 rejected)
  carry numeric pose they are not entitled to. Purity 8/150 = 0.0533.
- The static-gauge fill is contaminated identically (142 held-T_rest + 8
  measured) and is superseded.

## 3. Measurement-factor eligibility and solver inertness

- Eligible measurement-factor frames: exactly f30-36,f46 (8). Only these
  survived KT-L1 outlier gating as admissible rest-cluster observations.
  f45 is unresolved; f60/75/76/77 are rejected mask-drift; all others are
  unknown.
- `solver_estimate_count = 0`. The P15 optimizer is a zero-correction
  pass-through: `nfev=1, cost=0.0`, `translation_delta_norm_m.max=0.0`,
  `nonpenetration_target_frame_count=0`. There is no live posterior on any
  frame, so no frame legitimately carries a solver-estimate pose. Until an
  R4/R7 graph adds a pose-level prior + measurement terms with posterior
  covariance, `unknown` is the only honest label for the 137 unobserved
  frames.

## 4. Render manifest lineage

- Render-consumption rows 6,7,8 (the latest consumers) consume
  `unknown_preserving_pose_render/object_pose_observations.ndjson` and assert
  `body_drawn_only_on_measured_frames=True`,
  `numeric_pose_only_on_measured_frames=True`, `unknown_frames_have_null_pose=True`.
- Rows 3,4,5 consume the static-gauge pose (the earlier, superseded render).
- `current_pose_artifact_is_source_of_record=True`;
  `static_fill_superseded=True`. The current render draws the keyboard body
  only on f30-36,f46; unknown/rejected/unresolved frames show no body.

## 5. graph_health-like aggregate row (in summary.json `graph_health_row`)

Carries the required fields:

```
pose_measurement_purity        = {current: 1.0, original: 0.0533, static: 0.0533}
imputed_pose_count             = {current: 0, original: 137, static: 142}
unknown_null_pose_count        = 137
rejected_fit_count             = 4   ([60,75,76,77])
current_render_body_drawn_only_on_measured = true
solver_estimate_count          = 0
eligible_measurement_factor_count = 8
decision = pose_channel_pure_current__original_laundering_detected
```

`per_frame.ndjson` (150 rows) gives the per-frame breakdown: `ontology_class`,
`current_has_numeric_pose`, `current_pose_source`, `current_violation`,
`original_has_numeric_pose`, `original_pose_source`, `original_is_imputed`,
`original_violation`, `eligible_measurement_factor`. This is the
discriminating substrate: a future R4/R7 pose graph that writes a value into
an `unknown`/`rejected`/`unresolved` frame will flip `current_violation` to
`true` and the decision to `POSE_CHANNEL_PURITY_VIOLATION`.

## 6. How this prevents recurrence

The laundering error went unnoticed because no instrumentation classified pose
rows by provenance against an ontology. A held-T_rest stamp, a nearest-hold
copy, or a Slerp/lerp interpolation all wrote into the same
`translation_world_m` column the renderer localizes from, with an unread
`is_measured=false` flag. This audit makes the contract machine-checkable:
any future pose artifact is scored against the five-class ontology, and any
numeric pose on a non-`observed_measured`/non-`solver_estimate` frame is
flagged as a violation. The audit is instrumentation (routes an
intervention), not an acceptance gate that blocks approximate uncertain
outputs — a legitimate R4 `solver_estimate` with posterior covariance passes.

## 7. Scope discipline / honest limits

- **One new additive script.** Zero edits to existing scripts. Pre-existing
  dirty `scripts/*.py` on the branch are not mine.
- **No staging, no commits.** Script untracked (`??`); `git diff --cached`
  empty; on `yiwen_research`.
- **No model inference, no GPU, no smoothing, no pose infill.** Stdlib only.
- The audit **does not raise localization coverage**. Direct support stays
  8/150; the 137 unknown frames remain unknown. Raising coverage requires new
  perception/GT evidence, not a stored prior.
- The ontology frame sets (admissible=f30-36,46; rejected=f60,75,76,77;
  unresolved=f45) are the GT-free KT-L1/KT-L2 regime verdicts. Only HOT3D
  GT/R8 (bundle not on local disk) can adjudicate the 137 unknown frames; the
  purity audit keeps them honest until then.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
.venv/bin/python scripts/audit_clip001850_pose_provenance_purity.py
# -> /data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/pose_provenance_purity_audit/
#    summary.json   (checks, purity, graph_health_row, inputs + sha256)
#    per_frame.ndjson (150 rows; per-frame ontology + current/original violation flags)
#    audit.md       (human-readable report)
```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Implemented the pose provenance-purity audit as one new additive script scripts/audit_clip001850_pose_provenance_purity.py (untracked, not staged). It audits both the current unknown-preserving object_pose_observations and the original P15 pose graph rows against the five-class ontology (observed_measured f30-36,46=8; observed_rejected f60,75,76,77=4; unresolved f45=1; unknown=137; solver_estimate=0). All five required checks PASS: (1) current_no_forbidden_numeric_pose forbidden_numeric_count=0; (2) original_pose_graph_fails_purity imputed_pose_count=137, forbidden_numeric_count=142, purity=0.0533 (PASS = laundering correctly detected); (3) render_manifest_points_to_unknown_preserving current_is_source_of_record=True static_fill_superseded=True; (4) eligible measurement-factor frames exactly f30-36,46, solver_estimate_count=0, solver_inert=True (nfev=1 cost=0); (5) current_render_body_drawn_only_on_measured asserted by render-consumption rows 6,7,8. Produces summary.json, per_frame.ndjson (150 rows), audit.md under pose_provenance_purity_audit/. The graph_health-like row carries pose_measurement_purity, imputed_pose_count, unknown_null_pose_count=137, rejected_fit_count=4, current_render_body_drawn_only_on_measured=true. Decision: pose_channel_pure_current__original_laundering_detected. No scope widening: one additive instrumentation script, no solver, no gate, no edits to existing scripts."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Evidence sufficient for independent review: summary.json records inputs with sha256 (current pose object_pose_observations.ndjson sha256 67dabc79..., original pose graph report path, static-gauge fill path), the per-frame ontology classification, all five check verdicts with their raw counts, and the graph_health_row. per_frame.ndjson gives 150 per-frame rows with current_violation and original_violation flags (current_violation count=0; original_violation count=142). audit.md renders the report. All commands reproducible from the script header. Git state: script untracked, nothing staged, branch yiwen_research."
    }
  ],
  "changedFiles": [
    "scripts/audit_clip001850_pose_provenance_purity.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": ".venv/bin/python scripts/audit_clip001850_pose_provenance_purity.py",
      "result": "passed",
      "summary": "Wrote summary.json (5 checks ALL PASS, graph_health_row), per_frame.ndjson (150 rows), audit.md. purity current/original/static = 1.0/0.053333/0.053333; imputed_pose_count original=137; unknown_null_pose_count=137; rejected_fit_count=4; solver_estimate_count=0; decision=pose_channel_pure_current__original_laundering_detected."
    },
    {
      "command": "inspection of summary.json checks + graph_health_row required fields",
      "result": "passed",
      "summary": "All five required fields present in graph_health_row (pose_measurement_purity, imputed_pose_count, unknown_null_pose_count=137, rejected_fit_count=4, current_render_body_drawn_only_on_measured=true). Check1 forbidden_numeric=[]; Check2 imputed=137 forbidden=142 purity=0.0533; Check3 current_is_source_of_record=True static_fill_superseded=True; Check4 eligible=[30-36,46] solver_estimate=0 solver_inert=True; Check5 body_drawn_only_on_measured rows=[6,7,8]."
    },
    {
      "command": "per_frame.ndjson violation inspection (current vs original)",
      "result": "passed",
      "summary": "current_violation count=0 (no forbidden numeric on current); original_violation count=142 (137 imputed nearest-hold/interp + f45 unresolved + f60/75/76/77 rejected direct fits). f0/f90 original_violation=True (nearest_visible_pose_hold), current_violation=False (null pose). f45/f60 original_violation=True, current_violation=False."
    },
    {
      "command": "git status --short scripts/audit_clip001850_pose_provenance_purity.py && git diff --cached --name-only && git branch --show-current",
      "result": "passed",
      "summary": "Script untracked (??); nothing staged (git diff --cached empty); on branch yiwen_research."
    }
  ],
  "validationOutput": [
    "summary.json: all_checks_pass=True; five checks [PASS current_no_forbidden_numeric_pose (forbidden=0); PASS original_pose_graph_fails_purity (imputed=137, forbidden=142, purity=0.0533); PASS render_manifest_points_to_unknown_preserving (current source of record, static superseded); PASS measurement_factor_eligibility (eligible f30-36,46, solver_estimate=0, solver_inert nfev=1 cost=0); PASS current_render_body_drawn_only_on_measured (rows 6,7,8)].",
    "graph_health_row carries required fields: pose_measurement_purity {current:1.0, original:0.0533, static:0.0533}; imputed_pose_count {current:0, original:137, static:142}; unknown_null_pose_count 137; rejected_fit_count 4 ([60,75,76,77]); current_render_body_drawn_only_on_measured true; solver_estimate_count 0.",
    "per_frame.ndjson: 150 rows; current_violation count 0; original_violation count 142; each row carries ontology_class, current/original has_numeric_pose, current/original violation, eligible_measurement_factor.",
    "Ontology sum verified = 150: observed_measured 8 + observed_rejected 4 + unresolved 1 + unknown 137 + solver_estimate 0.",
    "Decision: pose_channel_pure_current__original_laundering_detected."
  ],
  "residualRisks": [
    "The audit does not raise localization coverage: direct support stays 8/150 (measured rest cluster). The 137 unknown frames remain unknown; raising support requires new perception/GT evidence (R2 correspondences that survive, or HOT3D GT/R8), not a stored prior.",
    "The ontology frame sets (admissible f30-36,46; rejected f60,75,76,77; unresolved f45) are GT-free KT-L1/KT-L2 verdicts. Only HOT3D GT (bundle not on local disk) can adjudicate the 137 unknown frames; the purity audit keeps them honest until then.",
    "The audit is instrumentation that detects laundering; it does not prevent a future R4/R7 solver from producing over-confident solver_estimate posteriors if its priors are miscalibrated. The solver_estimate class is the one still-abusable channel; honest posterior covariance checked against the rendered ghost is the remaining guard.",
    "Original pose graph 'forbidden_numeric_count' is 142 (137 imputed nearest-hold/interp + 5 non-admissible direct: f45 unresolved + f60/75/76/77 rejected), while imputed_pose_count is 137; both are reported separately in summary.json."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new untracked additive script scripts/audit_clip001850_pose_provenance_purity.py (~470 lines, stdlib only). No existing scripts edited. Outputs under the research output namespace: pose_provenance_purity_audit/{summary.json, per_frame.ndjson 150 rows, audit.md}. Nothing staged, nothing committed.",
  "reviewFindings": [
    "no blockers: all five required checks PASS with correct semantics (check 2 PASS means the audit DETECTS the original laundering, not that the original is clean).",
    "finding: the current unknown-preserving pose channel is pure (numeric pose only on f30-36,46; 137 unknown frames null pose; f45/f60/75/76/77 null pose).",
    "finding: the original P15 pose graph is correctly detected as contaminated (137 nearest-hold/interp imputed + 5 non-admissible direct-fit numeric = 142 forbidden numeric; purity 0.0533).",
    "finding: renderer/consumer lineage confirms rows 6,7,8 consume the unknown-preserving pose artifact and assert body_drawn_only_on_measured_frames=True; static-gauge rows 3,4,5 are superseded.",
    "finding: solver_estimate_count=0 confirmed via P15 optimizer inertness (nfev=1, cost=0, max translation delta=0)."
  ],
  "manualNotes": "Core result for the parent: the clip001850 pose provenance-purity audit is implemented and all five required checks pass. The current unknown-preserving pose channel is pure (purity 1.0); the original P15 nearest-hold/interpolation laundering is correctly detected (purity 0.0533, 137 imputed, 142 forbidden numeric); the static-gauge fill is superseded. The graph_health-like row carries pose_measurement_purity, imputed_pose_count, unknown_null_pose_count=137, rejected_fit_count=4, current_render_body_drawn_only_on_measured=true, solver_estimate_count=0. This makes the laundering contract machine-checkable: any future pose artifact that writes a numeric value into an unknown/rejected/unresolved frame flips current_violation to true and the decision to POSE_CHANNEL_PURITY_VIOLATION. The audit is instrumentation (routes an intervention), not an acceptance gate that blocks approximate uncertain outputs -- a legitimate R4 solver_estimate with posterior covariance passes. Open .../pose_provenance_purity_audit/audit.md for the human-readable report."
}
```
