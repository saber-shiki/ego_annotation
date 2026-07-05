# 20 — clip001850 ego.hoi sidecar package

Packages the clip001850 research slice into an `ego.hoi` 0.1.0 sidecar extension
that a downstream evaluator/render consumer can read. Packaging layer only: no
model, no GPU, no heavy inference. Reads durable artifacts under
`/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/` and
emits a package whose manifest points at the rendered full-duration artifact and
whose tables reproduce f32/f36/f45 contact state.

Script: `scripts/build_clip001850_ego_hoi_sidecar_package.py` (untracked, nothing staged).
Output: `/tmp/clip001850_ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/`

## What the package contains

```
manifest.json
tables/
  contact_frame_detail.ndjson              150 rows (one per frame)
  geometry_epochs_body_provenance.ndjson   6 rows (one per geometry epoch / body)
  render_consumption.ndjson                3 rows (overlay / world / side_by_side)
  graph_health.ndjson                      1 row (carried verbatim from source)
  motion_coupling.ndjson                   1 row (placeholder: not_produced)
```

## Manifest contract (the consumer-facing gate)

The task names the strict condition: do not claim schema progress unless the
sidecar points to the rendered full-duration artifact and can reproduce
f32/f36/f45 state. Both hold, verified programmatically:

- `frame_count = 150` (full duration of the raw clip).
- `zero_confirmed_contact_frames = True`; `confirmed_contact_count = 0`.
- `reproduce_f32_f36_f45_from_table = True`, with `reproduced_frame_states =
  {32: geometry_epoch_contaminated, 36: full_frame_depth_leak, 45:
  unresolved_incoherent_evidence}` — byte-derived from the
  `contact_frame_detail` table, not hardcoded.
- `render_artifacts.{overlay,world,side_by_side}` each carry the durable path,
  a sha256 that matches the durable `artifact_manifest.json` exactly, and
  `frame_count=150 / fps=30.0`. All three files exist on disk.
- `render_consumed_mesh_sha256 = 6d4bcebfb76149a4…` (the 3221-face observed
  body, not the TRELLIS completed mesh), `body_not_trellis = True`.
- `defaulted_frame_policy`: the 129 frames outside the kill-test window
  (28–48) carry `unresolved_evidence_incomplete` from
  `default_policy_uncovered`; `defaulted_frame_count = 129`. They are not
  dropped and not fabricated as contact.
- `input_hashes`: sha256 of 16 durable source files (artifact manifest,
  contact table + summary, body repair summary + face-provenance + both PLYs,
  hand-depth counterfactual summary + per-frame, graph-health NDJSON +
  manifest + factor-graph summary, full-duration render manifest + the three
  mp4s). No `MISSING` entries.

## Table fidelity

### contact_frame_detail.ndjson (150 rows)

- Frames 28–48 carry the canonical `contact_state` and the measured evidence
  that makes the state reproducible (`signed_gap_m`, `source_gap_z`,
  `observed_masked_penetrating_count`, `raw_penetrating_count`,
  `interval_published_penetration_max_m`, `render_consumed_mesh_sha256`,
  sigma_clip, pose provenance, face-provenance global).
- f32: `geometry_epoch_contaminated`, signed_gap −0.144 m, masked_pen=0,
  interval_pen_max=0.082 m (refuted by masked depth).
- f36: `full_frame_depth_leak`, signed_gap −0.081 m, masked_pen=0,
  interval_pen_max=0.107 m.
- f45: `unresolved_incoherent_evidence`, signed_gap −0.085 m, masked_pen=1,
  interval_pen_max=0.080 m (one thumb vertex on a depth outlier, demoted).
- Frames outside the window: `unresolved_evidence_incomplete`,
  `source=default_policy_uncovered`.

### geometry_epochs_body_provenance.ndjson (6 rows)

- `completed_trellis_mesh` (epoch_contact_e44915052d53): 90892 faces,
  non-watertight, `admissible_for_contact=False` (95.4% TRELLIS-inferred,
  uncarved at query time; mesh penetration not contact-admissible).
- `observed_only_body` (sha 6d4bcebfb76149a4…): 3221 faces, non-watertight,
  the full-duration render-consumed body; `admissible_for_contact=False`
  (open patch, unsigned distance only, 35.6–115.2 mm observed gap).
- `carved_support_body` (sha 52e8bbfac911…): 41670 faces, non-watertight,
  research-audit only, not rendered into the artifact.
- Three cross-solver geometry epochs (solver / contact_query / render) carried
  from the graph-health `cross_solver_geometry_consistency` block, each
  `admissible_for_contact=False`, with `mismatch_reasons` and
  `geometry_source_family`.

### render_consumption.ndjson (3 rows)

One per full-duration output. Each row carries the path, sha256 (matches
durable manifest), frame_count=150, fps=30, the consumed contact table path +
sha256, the consumed observed body sha256 + face count, `body_not_trellis`,
the default-state policy, `contact_state_counts`, `zero_confirmed_contact_frames`,
the decisive frame states, acceptance gates, and the per-panel
`pixel_diff_vs_published`.

### graph_health.ndjson (1 row)

Carried verbatim from the durable graph-health stream, annotated with
`sidecar_run_id` and `source_path`. Decision = `cross_solver_geometry_decoupled`
(solver/contact/render geometry epoch ids and source families differ).

### motion_coupling.ndjson (1 row, placeholder)

`status=not_produced`. The motion-coupling channel (object-motion onset
time-locked to hand kinematics) is the named floor-independent route to
promote a frame to `confirmed_contact`. It is not yet produced on this slice;
expected signal is weak (keyboard barely moves). The row records the contract:
admissible promotion only with a populated motion_coupling provenance column;
signed-distance thresholds / gap shifts / 2D-reprojection changes are
inadmissible.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
/home/yiwen/ego_annotation/.venv/bin/python scripts/build_clip001850_ego_hoi_sidecar_package.py
# -> /tmp/clip001850_ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/
#    manifest.json + tables/*.ndjson
```

Reads durable artifacts only. No torch, no CUDA, no model inference, no network.
Rerun is deterministic except for `provenance.created_at_unix`.

## Honest limits

- This is a packaging layer. It does not produce new contact, geometry, or
  motion evidence; it surfaces the existing durable evidence in a consumer-
  readable sidecar. The contact verdict (zero admissible frames) is inherited
  from subagents 07/10–16 and carried unchanged.
- The motion-coupling channel is a documented placeholder, not produced.
  Promoting any frame to `confirmed_contact` requires that channel or R8 GT;
  the package records this contract explicitly.
- The graph_health row carries the cross-solver geometry decoupling decision;
  it does not re-derive support/liveness beyond what the durable graph-health
  producer already measured.
- f32/f36/f45 reproducibility is verified against the `contact_frame_detail`
  table, which is the authority. The manifest's `reproduce_f32_f36_f45_from_table`
  gate is computed from the emitted table, not asserted.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Single additive script scripts/build_clip001850_ego_hoi_sidecar_package.py packages the clip001850 research slice into an ego.hoi 0.1.0 sidecar extension. Output at /tmp/clip001850_ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/ with manifest.json + 5 NDJSON tables (contact_frame_detail 150 rows, geometry_epochs_body_provenance 6 rows, render_consumption 3 rows, graph_health 1 row, motion_coupling 1 placeholder). No scope widening: one new script, no edits to existing scripts. Reads durable artifacts only; no model/GPU/heavy inference."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "manifest.json programmatically verified: frame_count=150; zero_confirmed_contact_frames=True (confirmed_contact_count=0); reproduce_f32_f36_f45_from_table=True with reproduced_frame_states={32:geometry_epoch_contaminated, 36:full_frame_depth_leak, 45:unresolved_incoherent_evidence}; render_artifacts overlay/world/side_by_side each carry durable path + sha256 that MATCH the durable artifact_manifest.json exactly (overlay=451ce259, world=9687c649, side_by_side=9a5724b4), frame_count=150, fps=30, all files exist; render_consumed_mesh_sha256=6d4bcebfb76149a4 (observed body, not TRELLIS), body_not_trellis=True; input_hashes for 16 durable sources, zero MISSING; defaulted_frame_policy.defaulted_frame_count=129, default_contact_state=unresolved_evidence_incomplete. Rerun deterministic except created_at_unix."
    }
  ],
  "changedFiles": [
    "scripts/build_clip001850_ego_hoi_sidecar_package.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "/home/yiwen/ego_annotation/.venv/bin/python scripts/build_clip001850_ego_hoi_sidecar_package.py",
      "result": "passed",
      "summary": "Wrote manifest.json + 5 NDJSON tables (contact_frame_detail 150, geometry_epochs_body_provenance 6, render_consumption 3, graph_health 1, motion_coupling 1). frame_count=150, zero_confirmed_contact_frames=True, reproduce_f32_f36_f45_from_table=True. Exit 0."
    },
    {
      "command": "manifest field validation (python3 -c on manifest.json + table inspection)",
      "result": "passed",
      "summary": "Verified frame_count=150, confirmed_contact_count=0, f32/f36/f45 reproduced from table, render hashes match durable artifact_manifest.json for all 3 outputs, 129 defaulted frames source=default_policy_uncovered, window frames source=contact_frame_detail."
    },
    {
      "command": "reproducibility rerun (sha256 compare manifest before/after)",
      "result": "passed",
      "summary": "Rerun produces identical manifest fields; only provenance.created_at_unix differs (expected timestamp)."
    },
    {
      "command": "git status --short scripts/build_clip001850_ego_hoi_sidecar_package.py + git diff --cached --name-only",
      "result": "passed",
      "summary": "Script is untracked (??); nothing staged; on branch yiwen_research."
    }
  ],
  "validationOutput": [
    "manifest.frame_count=150 == raw clip duration",
    "manifest.zero_confirmed_contact_frames=True; confirmed_contact_count=0",
    "manifest.reproduce_f32_f36_f45_from_table=True; reproduced={32:geometry_epoch_contaminated, 36:full_frame_depth_leak, 45:unresolved_incoherent_evidence}",
    "render_artifacts overlay/world/side_by_side sha256 match durable artifact_manifest.json (overlay=451ce259, world=9687c649, side_by_side=9a5724b4); all 3 files exist; frame_count=150 fps=30",
    "render_consumed_mesh_sha256=6d4bcebfb76149a4 (3221-face observed body), body_not_trellis=True",
    "defaulted_frame_policy: 129 frames default_policy_uncovered -> unresolved_evidence_incomplete; 21 window frames carry canonical contact_state from contact_frame_detail",
    "input_hashes: 16 durable sources hashed, zero MISSING",
    "contact_frame_detail table: f32 signed_gap=-0.144m masked_pen=0; f36 -0.081m masked_pen=0; f45 -0.085m masked_pen=1 (demoted); matches durable canonical rows",
    "geometry_epochs_body_provenance: all 6 bodies admissible_for_contact=False (none reach contact); TRELLIS=90892 faces, observed=3221, carved=41670",
    "motion_coupling: status=not_produced placeholder with documented promote-to-confirmed_contact contract",
    "git status: script untracked (??), nothing staged"
  ],
  "residualRisks": [
    "motion_coupling channel is a documented placeholder, not produced. confirmed_contact promotion requires that channel or R8 GT; the package records the contract but does not produce the signal.",
    "This is a packaging layer; it does not produce new contact/geometry/hand evidence. The contact verdict (zero admissible frames) is inherited from subagents 07/10-16 and carried unchanged.",
    "graph_health row is carried verbatim from the durable stream; it does not re-derive support/liveness beyond the existing producer.",
    "Output lives under /tmp/clip001850_ego_hoi_sidecar. It is reproducible from the durable artifact root by rerunning the script; a durable copy of the sidecar itself is not written by this script."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new untracked script scripts/build_clip001850_ego_hoi_sidecar_package.py (~480 lines). No edits to existing scripts. Outputs go to /tmp (not the repo). Nothing staged or committed.",
  "reviewFindings": [
    "no blockers: sidecar manifest points at the rendered full-duration artifact (all 3 mp4 paths + sha256 match the durable manifest) and the contact_frame_detail table reproduces f32/f36/f45 state (reproduce_f32_f36_f45_from_table=True, verified against the emitted table).",
    "no blockers: frame_count=150, zero_confirmed_contact_frames=True, defaulted-frame policy documented (129 frames unresolved_evidence_incomplete), input hashes for 16 durable sources with zero MISSING.",
    "note: motion_coupling is a documented placeholder; not a regression, the channel is not yet produced on this slice and the contract for its admissible use is recorded."
  ],
  "manualNotes": "The package satisfies the task's strict condition: the sidecar is not a schema container — it points at the rendered full-duration v19_overlay/world/side_by_side mp4s (durable, hashed, 150 frames) and the contact_frame_detail table byte-reproduces the f32/f36/f45 decisive states. A downstream evaluator/render consumer can read manifest.json to find the render artifacts + their hashes, read contact_frame_detail.ndjson for per-frame state, read geometry_epochs_body_provenance.ndjson for body admissibility, read render_consumption.ndjson for the render lineage, read graph_health.ndjson for the cross-solver decoupling decision, and read motion_coupling.ndjson for the documented absence of the floor-independent channel. The contact verdict stays unresolved with zero confirmed_contact frames."
}
```
