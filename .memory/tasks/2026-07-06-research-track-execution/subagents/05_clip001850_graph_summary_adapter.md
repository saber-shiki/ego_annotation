# Subagent 05 — clip001850 graph-summary adapter

## Result
`mechanism_decision = cross_solver_geometry_decoupled` on real three-solver
evidence from run
`20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`,
right hand, frames 26–46. The R0/R1 scaffold fires the D6 branch (checked first
in the decision table) instead of the generic `measurement_support_absent` /
`graph_inert` routes.

## Output paths (exact)
- Adapter summary (graph-summary input): `/tmp/egohoi_clip001850/clip001850_factor_graph_summary.json`
- Scaffold output: `/tmp/egohoi_clip001850/ego_hoi/graph_health.json`, `graph_health.ndjson`, `manifest.json`

## What was built
New untracked script `scripts/build_clip001850_ego_hoi_graph_summary.py` (no
heavy inference; reads JSON only). It aggregates the five per-solver reports
under the run root into ONE `factor_graph_summary` the scaffold consumes via
`--graph-summary`:

1. `measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`
   → objective (energy_initial/after, nfev, NP-target count), declared gauge.
2. `measurements/pose_fits/keyboard_visible_pose_fit/v18_compact_rigid_object_pose_fit_report.json`
   → `object_se3_observation` support fraction (13/150).
3. `measurements/mano_interval_correction/.../v18_joint_mano_interval_trajectory_state.json`
   → `hand_state_observation` support (57/150 right), `observed_surface_penetration_m`.
4. `measurements/contact_nonpenetration/keyboard_mano_object_constraint/v18_mano_object_constraint_state.json`
   → `contact_local_nonpenetration` (0/300), `signed_query_candidate_vertex_count`,
   `watertight`, contact-source family.
5. `measurements/geometry_completion/compact_keyboard_seed42/v18_compact_rigid_trellis_completion_report.json`
   → face provenance (observed 3221 / unsupported 963 / trellis 86708), epoch lineage.
Plus `renders/v19_published_runtime/v19_published_render_report.json` →
`published_contact_gap_m`.

### Key contract gotcha found and fixed
The scaffold's `--graph-summary` mode wraps the **whole file** as the
`factor_graph_summary` dict. The adapter therefore writes the summary keys
(`objective`, `factor_family_health`, `cross_solver_geometry_consistency`, …)
at the file's **top level**, with metadata under a `_meta` sibling. A first
attempt nested them one level too deep, which silently produced an all-null
`cross_solver_geometry_consistency` block and a wrong `stale_render_dependency`
decision — corrected by flattening to the contract.

## Decision evidence (from graph_health.json, real numbers)
- `decision`: `cross_solver_geometry_decoupled`
- `rationale`:
  - solver/contact/render geometry_epoch_id differ
    (`geo_observed_keyboard_depth_surface` vs `geo_completed_keyboard_trellis_seed42`)
  - solver/contact/render geometry_source_family differ
    (`observed_depth_surface` vs `trellis_completed`)
  - observed-surface penetration (0.107 m) and published positive gap (0.039 m) coexist
- `signed_query_candidate_vertex_count`: 4 (slice max 26–46; non-watertight sign mesh)
- `watertight`: false
- `face_provenance_summary`: observed_fraction 0.0354, trellis_completed_fraction 0.9540
- `observed_surface_penetration_m`: 0.1068 (right, after solver max)
- `published_contact_gap_m`: 0.0392
- `objective_delta`: energy_initial 0.0 → energy_after 0.0, delta 0.0 (rigid pose graph inert: nfev=1, cost 0)
- factor_family_health support fractions:
  hand_state_observation 0.38 (57/150), object_se3_observation 0.087 (13/150),
  contact_local_nonpenetration 0.0 (0/300), contact_switch_discrete 0.0 (0/150),
  contact_switch_temporal 0.087 (13/150)

The cross-solver branch fires *before* the support/inert checks by design, so
the "support absent" mislabel (the defect the causal card flagged in
`02_first_slice_causal_card.md` §3 M5) is avoided: the measurement is not
absent globally — it is stranded on a different geometry source in a sibling
solver.

## Falsifiability / missing-field handling
The adapter records `_meta.missing_fields` listing every cross-solver evidence
field it could not source. Verified against a partial run root missing the
contact/NP, interval, geometry, and render reports: the adapter emitted
`MISSING fields preventing a clean cross_solver decision:
[signed_query_candidate_vertex_count, observed_surface_penetration_m,
published_contact_gap_m, watertight]`. Each report also carries a per-file
`_meta.provenance` entry with `present`/`missing` status + sha256.

## Acceptance report
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Added exactly one new script scripts/build_clip001850_ego_hoi_graph_summary.py that reads the five named reports from the run root and constructs a factor_graph_summary with cross_solver_geometry_consistency populated, then runs build_ego_hoi_sidecar_and_graph_health.py --graph-summary to emit graph_health.json. No existing files touched; no scope widening; no heavy inference (JSON reads only). mechanism_decision = cross_solver_geometry_decoupled on real evidence."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "graph_health.json decision block reproduced above cites exact on-disk numbers: observed_surface_penetration_m=0.1068, published_contact_gap_m=0.0392, signed_query_candidate_vertex_count=4, watertight=false, face provenance observed 3221/trellis 86708, objective_delta 0.0 (nfev=1), support fractions 57/150 hand, 13/150 pose, 0/300 NP. Adapter provenance + missing_fields list expose source/absence of each field. Reproducible via the two commands in commandsRun."
    }
  ],
  "changedFiles": [
    "scripts/build_clip001850_ego_hoi_graph_summary.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 scripts/build_clip001850_ego_hoi_graph_summary.py --run-root /data2/.../20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1 --out-dir /tmp/egohoi_clip001850 --hand-side right --start-frame 26 --end-frame 46",
      "result": "passed",
      "summary": "wrote /tmp/egohoi_clip001850/clip001850_factor_graph_summary.json; all cross_solver evidence fields present"
    },
    {
      "command": "python3 scripts/build_ego_hoi_sidecar_and_graph_health.py --graph-summary /tmp/egohoi_clip001850/clip001850_factor_graph_summary.json --out-dir /tmp/egohoi_clip001850/ego_hoi --run-id 20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1 --solution-id sol_clip001850_rh_26_46",
      "result": "passed",
      "summary": "decision=cross_solver_geometry_decoupled; rationale: epoch/source differ + penetration(0.107) and positive gap(0.039) coexist"
    },
    {
      "command": "adapter run against a partial run root missing contact/NP+interval+geometry+render reports",
      "result": "passed",
      "summary": "missing-field path verified: _meta.missing_fields lists signed_query_candidate_vertex_count, observed_surface_penetration_m, published_contact_gap_m, watertight"
    },
    {
      "command": "git diff --cached --name-only; git status --short scripts/build_clip001850_ego_hoi_graph_summary.py",
      "result": "passed",
      "summary": "new script untracked only; nothing staged"
    }
  ],
  "validationOutput": [
    "graph_health.json mechanism_decision = cross_solver_geometry_decoupled (required decision met).",
    "cross_solver_geometry_consistency populated from real on-disk fields: observed_surface_penetration_m=0.1068, published_contact_gap_m=0.0392, signed_query_candidate_vertex_count=4, watertight=false, observed_fraction=0.0354, trellis_completed_fraction=0.9540.",
    "objective_delta = 0.0 (rigid pose graph inert: nfev=1, cost 0.0, NP targets 0) captured in objective block.",
    "factor_family_health support fractions faithful: hand_state_observation 57/150, object_se3_observation 13/150, contact_local_nonpenetration 0/300.",
    "Contract gotcha surfaced and fixed: scaffold --graph-summary wraps whole file as factor_graph_summary; adapter now writes summary keys at file top level (first attempt nested too deep -> silent all-null block -> wrong stale_render decision).",
    "Missing-field exposure verified on partial run root."
  ],
  "residualRisks": [
    "The slice-max signed_query_candidate_vertex_count over frames 26-46 is 4, not 0 (frame 37 alone was 0 per the causal card); this does not change the decision (epoch/source differ + penetration/gap coexist still fire), but the candidate-count==0 sub-reason did not trigger for the wider slice.",
    "The epoch/source-family labels for the three solvers are structural facts (interval solver queries observed surface, contact/NP queries completed TRELLIS mesh by design), so the epoch/source mismatch fires even if numeric fields are missing; numeric-field absence is still exposed via _meta.missing_fields.",
    "Adapter is read-only aggregation; it does not yet emit per-frame contact_frame_detail rows (R0 table) — that is the next R0 build step routed by this decision, out of scope here."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new untracked script (~300 lines) scripts/build_clip001850_ego_hoi_graph_summary.py that aggregates five clip001850 per-solver JSON reports into a scaffold-consumable factor_graph_summary with cross_solver_geometry_consistency. No other files modified.",
  "reviewFindings": [
    "no blockers: decision is the required cross_solver_geometry_decoupled on real evidence; all numeric claims cite on-disk report fields; nothing staged.",
    "note: the adapter writes summary keys at file top level to match the scaffold's --graph-summary contract (whole-file-as-factor_graph_summary); documented inline in the script."
  ],
  "manualNotes": "Required result achieved: mechanism_decision=cross_solver_geometry_decoupled. Exact output path: /tmp/egohoi_clip001850/ego_hoi/graph_health.json (+ .ndjson, manifest.json). Adapter input: /tmp/egohoi_clip001850/clip001850_factor_graph_summary.json. To re-run end-to-end: (1) python3 scripts/build_clip001850_ego_hoi_graph_summary.py --run-root <RUN> --out-dir /tmp/egohoi_clip001850 --hand-side right --start-frame 26 --end-frame 46; (2) python3 scripts/build_ego_hoi_sidecar_and_graph_health.py --graph-summary /tmp/egohoi_clip001850/clip001850_factor_graph_summary.json --out-dir /tmp/egohoi_clip001850/ego_hoi --run-id <RUN_NAME>. The decision routes the next intervention: wire observed-surface contact evidence into contact/NP before adding contact factors (per causal card M1/M5). Not committed."
}
```
