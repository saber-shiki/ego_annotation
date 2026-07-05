# R1 Graph / Render Lineage Instrumentation Inventory

Scope: concrete, file/function/field-level inventory for the R1 deliverable
(support / liveness / gauge / freshness / input-output deltas / render-state
hash lineage) against the live v18/v19 code paths. Read-only. No edits, no
staging, no commits. Branch `yiwen_research` (44 modified + 7 untracked files
in working tree; nothing staged).

The inventory is organized so each proposed field is tied to (i) the defect
mechanism it discriminates, (ii) what already exists, (iii) what is missing,
and (iv) the smallest safe patch that makes the signal real rather than
`None`/`pending`.

---

## 0. Anchor: the R1 instrumentation target already has a scaffold

`scripts/build_ego_hoi_sidecar_and_graph_health.py` (untracked, 657 lines) is
the in-flight R0/R1 producer. It defines the `ego.hoi` 0.1.0 sidecar manifest
and a `graph_health` row whose schema matches TASK_PACK R1 exactly:

- `factor_family_health[fam] = {declared_count, supported_count, support_fraction, active_residual_count, residual_sum}`
- `support_fraction_min_critical`, `active_residual_count_total`
- `objective_delta = {energy_initial, energy_after, energy_delta}`
- `input_output_deltas = {objective_energy_delta, per_family:{delta_norm, max_abs_delta}}`
- `stale_dependency_count`, `gauge_declaration_count`
- `geometry_epoch_lineage`, `geometry_epoch_id_active`
- `input_hashes`, `output_hashes.{graph_output, object_pose, hand_state, contact}`
- `render_state_hash`, `render_state_hash_lineage.{graph_output_hash, render_state_hash}`
- `mechanism_decision = {decision, mechanism, next_intervention, rationale}`
  produced by `decide_mechanism(...)` with a fixed, ordered decision table:
  `measurement_support_absent` → `graph_inert` → `stale_geometry_join` →
  `stale_render_dependency` → `active_healthy`.

The scaffold is correct as a **schema + decision router**, but for real v18
summaries almost every per-family signal is explicitly `None` / `pending`,
because it only reads `factor_graph_summary` / `factor_counts` that v18 already
produces. The R1 work is therefore **not "design the schema"** — it is **wire
the real signals from the solvers/state/renderer into the existing row**.

This re-frames R1 from a green-field design task into a concrete wiring task:
each field below has a producer location in code where the real value already
exists or can be computed with a minimal, non-blocking change.

---

## 1. Code surface inventory (files / functions / state fields)

### 1a. Solvers that produce graph output

| Solver | File | Optimizer | Variables (per solver self-report) | Per-family health emitted today |
|---|---|---|---|---|
| V17 full-timeline sparse graph | `scripts/solve_v17_full_timeline_factor_graph.py` | SciPy `sparse.linalg.lsqr` (bounded `least_squares` for correspondences) | per-frame object translation (3) + small-angle object rotvec (3) + per-(frame,side) hand camera-ray shift (1) | NO per-family support; reports aggregate `weighted_residual_rms_before/after`, `correction_summary`, `contact_before/after`, `linear_system_shape`, `contact_selection_digest` |
| V19 rigid pose graph | `scripts/solve_v19_rigid_object_pose_graph.py` | SciPy `least_squares` (soft_l1, jac_sparsity) | per-observation rot_delta(3)+trans_delta(3); full-timeline interpolation/extrapolation fill | partial: `nonpenetration_target_frame_count`, `correction_summary`, `surface_before/after`, `residual_rms_before/after`. NO per-family residual decomposition, NO gauge declaration |
| V18 joint MANO interval | `scripts/solve_v18_joint_mano_interval_trajectory.py` | torch LBFGS over `root_delta, pose_delta, trans_delta, object_trans_delta, contact_logit` | per-frame MANO root orient, 15-finger pose, global trans, object trans, contact logit | partial: per-frame `delta_norms`, `corrected_frame_count`, `active_set_added_constraint_counts`, `contact_patch_*count`, `visible_surface_depth_order_selected_vertex_count`, `dense_observed_constraint_count_final`. NO aggregated per-family `support_fraction` / `gradient_share` |

Key functions (where R1 instrumentation would attach):

- `solve_v17_full_timeline_factor_graph.py`
  - `build_linear_system(graph, args, correspondence_params)` (line 579): the
    single point where every factor family is instantiated as rows. Returns
    `LinearSystem(matrix, target, contact_correspondence_count,
    contact_selection_digest)`. **This is the natural place to emit a
    per-family row-count map** (object prior rows, object-step rows, object-accel
    rows, hand-ray prior rows, hand-ray-step rows, contact rows).
  - `solve_contact_correspondence_system(graph, args)` (line 851): runs the
    alternating correspondence + least-squares. Returns `(system, result,
    solution, correspondence_history)`.
  - `solve_case(...)` (line 1204): writes the report; already computes
    `before_residual = initial_system.matrix @ x0 - initial_system.target` and
    `after_residual = system.matrix @ solution - system.target` — so a
    per-family residual-norm split is a `(matrix @ solution - target)` sliced
    by the row ranges recorded in `build_linear_system`.

- `solve_v19_rigid_object_pose_graph.py`
  - `residual_vector(x, observations, args)` (line 197): assembles the residual
    block-by-block: per-obs translation prior (3) + rotation prior (3) +
    optional nonpenetration target (3) + temporal step (6) + temporal accel (6).
    Each block has a known source family. **Today the families are implicit in
    code order, not recorded.**
  - `residual_sparsity(observations)` (line 246): mirrors the same block order
    for the Jacobian. The two together are the canonical place to emit a
    `factor_family_row_ranges` map.
  - `run(args)` (line ~437): writes the report. `result.cost`,
    `residual_rms_before/after`, `correction_summary` exist; **no gauge
    declaration, no per-family active residual count**.

- `solve_v18_joint_mano_interval_trajectory.py`
  - `optimize_rows(...)` (line 1613): builds the loss in `closure()` (line
    1813). The loss is accumulated family-by-family (observed penetration,
    dense barrier, contact patch, visible shift, depth shift, hand-ray-shift
    prior, translation/root/pose priors, temporal smoothness, bound hinge).
    **Each term is added to a scalar `loss`; the per-family contribution is
    discarded at backward.**
  - The post-solve per-frame `states[i]` dict already carries
    `final_active_constraint_residual_after_solver_m`,
    `full_observed_surface_penetration_after_solver_m`,
    `contact_patch_final_normal_gap_m`, `visible_joint_shift_px`,
    `joint_camera_depth_shift_m`, `delta_norms`. These are exactly the
    per-family output signals; they are not yet aggregated into a
    `factor_family_health` map.

### 1b. State materialization (the render-state boundary)

- `scripts/build_v19_rigid_render_state.py` — `build(args)` (line ~250).
  - **Already enforces path-equivalence freshness** via `require_matching_path`
    for pose_report / constraint_report / temporal_mano_state inputs against
    the current annotations / completion_report / completed_mesh. This is a
    structural stale-dependency guard, **but it is path-based, not hash-based**,
    and only covers the four named sidecars. It does NOT cover the render
    manifest, and it does not emit a `stale_dependency_count` field.
  - Already rejects completion reports with `unsupported_uncertain` faces or
    un-run free-space filtering (`validate_completion_body_contract`).
  - State written to disk (`state` dict) carries `inputs.*` paths and
    `completion_outputs`, but **no SHA256 hashes** of any input.

- `scripts/render_v19_rigid_state_artifact.py` — `render(args)` (line 737).
  - Consumes the render state; writes `v19_rigid_state_render_manifest.json`
    with `inputs.{render_state, annotations, completed_mesh}` paths, plus
    `evidence.frames_rendered / frames_with_object_pose /
    frames_with_rasterized_body_pixels`.
  - **No content hash of the consumed render state.** No
    `render_state_hash_lineage` field. The manifest records that the renderer
    consumed a path, not that it consumed a specific byte content.

### 1c. Existing validators (R1 consumers, not producers)

- `scripts/validate_v18_factor_graph.py` (531 lines): validates a v18
  `annotations_v18_full.json`'s `factor_graph_summary`. Checks
  `objective.energy_after <= energy_initial`, variable/factor counts, family
  presence, implemented_variable_status, spec_factor_gaps, contact anchor
  fixed-point, per-frame variable/factor consistency. **No check on
  support_fraction, liveness, gauge declaration, or render-state hash.**
- `scripts/validate_v18_full_pipeline_artifact.py` (1131 lines): validates
  frame count, fps, monotonicity, overlay/world/side-by-side draw counts. No
  graph-health check.

### 1d. Existing reusable hashing helper

- `scripts/build_v18_hand_baseline_branch.py:247 file_sha256(path)`: chunked
  SHA256 of a file. The scaffold already has its own
  `sha256_file / sha256_json_canonical` (lines 67–84). Use the scaffold's
  versions; do not duplicate.

---

## 2. Field-by-field status against the scaffold

For each R1 field: present state in the scaffold, real-signal producer
location, gap, smallest patch.

### 2.1 `factor_family_health[fam].support_fraction` / `supported_count` / `active_residual_count` / `residual_sum`

- **Scaffold behavior**: for real v18 summaries, falls back to
  `factor_counts[fam]` declarations and sets `supported_count=None,
  support_fraction=None, active_residual_count=None, residual_sum=None,
  pending=...` (see `derive_family_health`, line 251). For synthetic
  summaries, reads the synthetic `factor_family_health` block.
- **Real signal location**:
  - V17 graph: `build_linear_system` (line 579) knows exactly how many rows
    belong to each family (object prior, object step, object accel, hand-ray
    prior, hand-ray step, contact). "Supported" = rows whose target is finite
    and whose weight is non-zero; "active residual" = `|matrix @ solution -
    target| > RESIDUAL_EPSILON`.
  - V19 pose graph: `residual_vector` (line 197) + `residual_sparsity` (line
    246) know family block boundaries.
  - V18 MANO solver: `closure()` (line 1813) family terms + post-solve
    `states[i]` residuals.
- **Defect discriminated**: D5-M1 (measurement extraction produced no support
  before optimizer). D5-M4 (one residual channel dominates). Without this
  field, "factor present in `factor_counts`" is indistinguishable from "factor
  actually constrained the solve".
- **Gap**: no solver writes a per-family `{declared, supported, residual_sum}`
  block into its report, and v18 `factor_graph_summary` has no
  `factor_family_health` key (only `factor_counts`).
- **Smallest safe patch**:
  1. In `solve_v17_full_timeline_factor_graph.py::build_linear_system`, return
     a `family_row_ranges: dict[str, tuple[int,int]]` alongside the linear
     system. Compute `supported = nonzero target count in range`,
     `residual_sum = sum(|residual_i|)` after the solve in `solve_case`.
     Write `factor_family_health` into the report.
  2. Have `build_ego_hoi_sidecar_and_graph_health.py::derive_family_health`
     prefer a real `factor_family_health` block when present (it already does
     this); the only change is that real producers must emit it.
  3. For V19 pose graph and V18 MANO solver, add the same
     `factor_family_health` block to their report dicts. Each already knows the
     family boundaries; the patch is bookkeeping, not new modeling.

### 2.2 `input_output_deltas.per_family.{delta_norm, max_abs_delta}`

- **Scaffold behavior**: `objective_energy_delta` is populated from
  `objective.energy_initial - energy_after`; `per_family.delta_norm` is
  explicitly `None` with `pending="populated by
  graph_input_output_differencer (R1 follow-on)"` (line ~440).
- **Real signal location**:
  - V17: `summarize_shifts(solution, graph)` (line 912) already returns
    `object_shift_norm_m`, `object_rotvec_norm_rad`, `hand_ray_shift_abs_m`
    summaries. These ARE the per-family input-output deltas; they are just not
    labeled under `input_output_deltas.per_family`.
  - V19 pose graph: `correction_summary(result.x, observations)` (line 339)
    returns `translation_delta_norm_m`, `rotation_delta_norm_rad`, plus step
    norms. Same situation.
  - V18 MANO: per-frame `delta_norms = {translation_m, object_translation_m,
    root_rad, max_pose_joint_rad}` plus `contact_patch_posterior_minus_prior`.
- **Defect discriminated**: D5-M2 (nonzero residual with zero state delta ⇒
  wiring/plumbing broken, gauge locked, or Jacobian zero). This is the single
  most important field for the "inert graph" mechanism and it is the one the
  scaffold most visibly lacks today.
- **Gap**: producers compute the magnitudes but do not slot them into a
  uniform `input_output_deltas.per_family` schema.
- **Smallest safe patch**:
  1. In each solver report, add an `input_output_deltas` block that re-uses
     the already-computed `correction_summary` / `summarize_shifts` /
     `delta_norms` values, keyed by the same family names used in
     `factor_family_health`.
  2. Teach `derive_...` in the scaffold to read producer-side
     `input_output_deltas.per_family` when present (currently it always emits
     `pending`). One-line prefer-real-over-template change.

### 2.3 `gauge_declaration_count` and `declared_gauges`

- **Scaffold behavior**: `derive_gauge_declaration_count(summary)` reads
  `factor_graph_summary.declared_gauges`. Real v18 summaries do NOT have this
  key (confirmed against
  `/data2/ego_annotation_outputs/v18_full_pipeline/task5_tomato_960/annotations_v18_full.json`),
  so the count is always 0 for real runs.
- **Real signal location**:
  - V17 graph: gauge is implicitly "all corrections are zero-mean priors
    anchored at the input pose" (`x0 = np.zeros(...)`, line 1217). The free
    DOF are the per-frame object translation/rotation and per-(frame,side)
    hand ray shift. The fixed DOF are camera trajectory, MANO articulation,
    mesh topology, contact labels. These are listed in the report's
    `semantics.optimized_variables` / `semantics.fixed_variables` (line
    ~1335) but not as a formal gauge declaration.
  - V19 pose graph: gauge is "rot_delta=0, trans_delta=0 at every observation"
    (x0 zeros, line 437), i.e. the ICP pose is the gauge origin. Object
    radius used for sigma_r is in `pose_row_sigma`.
  - V18 MANO solver: gauge is the source HaWoR MANO replay
    (`zero_surface_mode == "similarity_mapped_raw"` maps raw vertices through
     a fixed similarity), with `translation_allowed_t` masking the
     translation gauge on low-support frames.
- **Defect discriminated**: D1-M3 (camera/depth gauge error absorbed by
  object pose) and D5-M2 (gauge lock masquerading as inert graph). A
  declared gauge makes "which DOF are free" explicit, exposing the case
  where a residual is non-zero but its variable is gauge-locked.
- **Gap**: no solver emits a `declared_gauges` list. The semantics are in
  prose.
- **Smallest safe patch**: each solver writes a small
  `declared_gauges: [{variable, gauge, anchor_frame, free_dof, fixed_dof}]`
  block (for V17/V19/V18). Pure documentation-of-fact; no optimization change.
  Scaffold already consumes it via `derive_gauge_declaration_count`.

### 2.4 `stale_dependency_count` and `geometry_epoch_lineage`

- **Scaffold behavior**: `derive_geometry_epoch_lineage` reads
  `factor_graph_summary.geometry_epochs` + `active_geometry_epoch_id`;
  `derive_stale_dependency_count` counts variables whose
  `geometry_epoch_id` is in `superseded_epoch_ids`. For real v18 summaries,
  `geometry_epochs` is absent ⇒ lineage is empty and count is 0.
- **Real signal location**:
  - `scripts/build_v19_rigid_render_state.py::require_matching_path` (line
    ~175) already detects stale path-level dependencies across pose /
    constraint / temporal-MANO sidecars. This is a stronger freshness check
    than the scaffold's, but it raises on violation rather than counting.
  - `scripts/build_v18_compact_rigid_trellis_completion.py` and
    `build_v18_depth_fused_reconstruction.py` produce completion reports with
    face-provenance / free-space state; these are the natural
    geometry-epoch sources but they do not assign `epoch_id` /
    `superseded_by`.
  - V18 annotations carry per-frame `object_se3` rows but no
    `geometry_epoch_id` field on those rows.
- **Defect discriminated**: D2-M1/M3 (geometry contamination / prior
  completion inventing surfaces) and D5-M3 (stale object/hand/geometry ids
  joined into the solve). Today these are invisible at the graph level.
- **Gap**: there is no `geometry_epoch_id` carried on variable rows, and no
  `geometry_epochs` lineage list in the summary.
- **Smallest safe patch** (two layers, independent):
  1. **Render-state layer (cheapest, highest-value)**: in
     `build_v19_rigid_render_state.py`, count the number of failed
     `require_matching_path` checks (convert from raise-on-first to
     collect-then-raise) and write `stale_dependency_count` +
     `stale_dependency_details` into the state. The path-equivalence guard
     already exists; this only changes how violations are reported.
  2. **Graph-layer (requires geometry-epoch plumbing)**: when R3 (geometry
     epochs) lands, completion reports will assign `epoch_id`. Then
     `object_se3` rows in the annotation get a `geometry_epoch_id`, and the
     scaffold's existing `derive_stale_dependency_count` starts working
     unmodified. R1 should land layer (1); layer (2) is R3-coupled.

### 2.5 `render_state_hash` and `render_state_hash_lineage`

- **Scaffold behavior**: `render_state_hash = None` placeholder; lineage says
  `pending="renderer must write its consumed-state fingerprint here"`.
- **Real signal location**:
  - `build_v19_rigid_render_state.py::build(args)` writes the canonical
    render-state JSON to disk. Its content hash is the natural
    `render_state_hash`.
  - `render_v19_rigid_state_artifact.py::render(args)` reads that file and
    produces the manifest. The renderer is the place that records "I consumed
    hash H".
- **Defect discriminated**: D5 "correct graph output but stale render" — the
  case where the solve moved but the renderer consumed an older cached state.
  Without this, a re-run that silently skips rendering is indistinguishable
  from a fresh render.
- **Gap**: no SHA256 anywhere in the render chain. The closest existing
  mechanism is path-equivalence in P19a, which catches a different defect
  (sidecar built against a different upstream path) — not the
  same-content-different-bytes case.
- **Smallest safe patch**:
  1. In `build_v19_rigid_render_state.py::build(args)`, after `write_json`,
     compute `sha256_file(output)` and add `render_state_sha256` to the state
     dict (re-write once so the hash is over the final bytes, or store the
     hash in a sibling `<output>.sha256` file to avoid the self-reference).
  2. In `render_v19_rigid_state_artifact.py::render(args)`, read the state
     file, compute its `sha256_file`, and write `render_state_hash` +
     `render_state_hash_lineage` into the manifest under
     `inputs.render_state_sha256`.
  3. The scaffold's `graph_health` row already has a slot for
     `render_state_hash`; teach the renderer (or a thin post-render step) to
     fill it. No solver change needed.

### 2.6 `output_hashes.{object_pose, hand_state, contact}`

- **Scaffold behavior**: only `graph_output` is computed
     (`output_hash_from_frames`); the three per-family output hashes are
     `None` / `pending`.
- **Real signal location**: each solver already produces per-family solution
  arrays (V17: `summarize_shifts`; V19: `pose_rows` + `correction_summary`;
  V18 MANO: `per_frame_states[].optimized_*`).
- **Defect discriminated**: D5-M2 at finer granularity than the aggregate
  graph_output hash. Lets the renderer assert per-family freshness.
- **Smallest safe patch**: hash the per-family solution sub-document
  (canonical JSON) in each producer. One helper call per family; no
  modeling change.

---

## 3. Patch plan — ordered by mechanism leverage, smallest-first

Each patch is independent and additive. None changes any optimization
behavior. None adds a heuristic threshold to the solver path; the two
thresholds that exist live in the scaffold's decision table
(`SUPPORT_FRACTION_THRESHOLD = 0.05`, `ENERGY_DELTA_INERT_EPS = 1e-6`) and
route diagnostics rather than gate output.

### Patch P1 — V17 graph: emit real `factor_family_health` + `input_output_deltas` + `declared_gauges`
- File: `scripts/solve_v17_full_timeline_factor_graph.py`
- In `build_linear_system` (line 579): record `family_row_ranges` as each
  `add_row` call is made. Return it on `LinearSystem`.
- In `solve_case` (line 1204): after `after_residual` is computed, slice the
  residual by family range and emit `{declared, supported, residual_sum,
  active_residual_count}` per family into `report["factor_family_health"]`.
- Re-use the already-computed `shifts` (line 1264) to populate
  `report["input_output_deltas"]["per_family"]`.
- Add `report["declared_gauges"]` from the existing
  `semantics.optimized_variables` / `fixed_variables` (line ~1335) — this is
  re-labelling, not new information.
- Discriminates: D5-M1, D5-M2, D5-M4, D1-M3.

### Patch P2 — V19 pose graph: same three fields
- File: `scripts/solve_v19_rigid_object_pose_graph.py`
- In `residual_vector` / `residual_sparsity` (lines 197, 246): the family
  block order is fixed by construction; capture it once as
  `family_row_ranges` and return alongside the residual.
- In `run(args)` (line ~437): emit `factor_family_health`,
  `input_output_deltas.per_family` (from `correction_summary` +
  `nonpenetration_target_*`), and `declared_gauges` (gauge origin = ICP
  pose at x0=0; free DOF = rot_delta+trans_delta per observation; fixed DOF
  = camera, MANO, mesh).
- Discriminates: D1-M2 (symmetry/weak texture ⇒ rotation low-observability,
  visible as low `active_residual_count` on rotation family), D5-M2.

### Patch P3 — V18 MANO interval solver: same three fields, aggregated
- File: `scripts/solve_v18_joint_mano_interval_trajectory.py`
- In `closure()` (line 1813): accumulate per-family loss contributions into
  a dict captured by nonlocal (or refactor closure to return a named-tuple
  breakdown). The family list is already enumerable: `observed_penetration,
  dense_barrier, contact_patch, visible_shift, depth_shift, hand_ray_shift,
  trans_prior, root_prior, pose_prior, temporal_trans, temporal_root,
  temporal_pose, bound_hinge`.
- In `optimize_rows` post-solve (line ~1980, where `states` is built):
  aggregate per-family `support_fraction` (frames with non-empty constraint
  set), `active_residual_count`, and `delta_norm` (from `delta_norms` and
  per-family residual summaries already computed).
- Add the three blocks to the `interval` summary dict (line ~2050).
- Discriminates: D3-M1 (hand/object source gap wrong), D3-M4 (contact
  flicker), D4-M2 (source-switch jitter), D5-M2/M4.

### Patch P4 — render-state builder: emit `render_state_sha256` + `stale_dependency_count`
- File: `scripts/build_v19_rigid_render_state.py`
- In `build(args)`: after `write_json(output, state)`, compute
  `sha256_file(output)` and write a sibling `<output>.sha256` (avoids
  self-referential hash). Add `render_state_sha256_path` to the state's
  `outputs`.
- Convert the four `require_matching_path` calls (lines ~250–290) from
  raise-on-first to collect-then-raise; record
  `stale_dependency_count` + `stale_dependency_details` in the state. Still
  raise (preserves today's hard freshness contract) but expose the count.
- Discriminates: D5 stale-render, D2 stale geometry.

### Patch P5 — renderer: write `render_state_hash` lineage into manifest
- File: `scripts/render_v19_rigid_state_artifact.py`
- In `render(args)` (line 737): read the render-state file bytes, compute
  `sha256_file(render_state_path)`, add to manifest under
  `inputs.render_state_sha256` and a new top-level
  `render_state_hash_lineage = {render_state_hash, graph_output_hash (if
  present in the state), annotations_sha256, completed_mesh_sha256}`.
- Discriminates: D5 "correct graph output, stale render".

### Patch P6 — scaffold: prefer real producer fields over `pending` templates
- File: `scripts/build_ego_hoi_sidecar_and_graph_health.py`
- `derive_family_health` (line 251): already prefers real
  `factor_family_health`; no change.
- `build_graph_health_row` (line ~430): when the producer summary includes a
  real `input_output_deltas.per_family`, copy it through instead of the
  `pending` template. When `declared_gauges` is present, surface the actual
  gauge list, not just the count. When `render_state_hash` is supplied via
  CLI / manifest read, populate it instead of leaving `None`.
- Add an optional `--render-manifest PATH` arg so the scaffold can read the
  renderer's `render_state_hash` and complete the lineage.
- This patch is what makes P1–P5 visible in the `graph_health` row.

### Patch ordering rationale
P6 is the consumer; P1–P5 are producers. Land P4 + P5 + P6 first: they close
the **stale-render** branch of the decision table with the smallest changes
(no solver touches). Then P1 (V17 graph) because V17 is the simplest graph
and proves the per-family wiring end-to-end. Then P2, P3.

---

## 4. What this inventory deliberately does NOT propose

- No new threshold tuning, no new acceptance gate. The two thresholds in the
  scaffold route diagnostics; they do not block output. AGENTS.md forbids
  heuristic gates; this instrumentation must remain non-gating.
- No change to any optimizer objective, noise model, or residual weighting.
- No geometry-epoch plumbing at the graph layer — that is R3 work and is
  listed as layer (2) under §2.4, intentionally deferred.
- No new hand-written if/else for object categories, materials, or contact
  phrases (AGENTS.md methodology).
- No local GPU / heavy inference. Every patch is CPU bookkeeping over
  already-computed residuals and JSON files.

---

## 5. Residual unknowns / open questions for the parent

1. **Is the render-state self-hash acceptable as a sibling `.sha256` file, or
   must it be embedded?** Embedding requires a two-pass write (hash then
   re-serialize with the hash field included, then the hash changes). Sibling
   file is cleaner. Decision needed before P4 lands.
2. **Should the V18 `annotations_v18_full.json` `factor_graph_summary` itself
   grow a `factor_family_health` block, or only the per-solver reports?** The
   scaffold reads either; the cleaner answer is "per-solver reports produce
   it; the annotation aggregator composes it". Parent should confirm.
3. **Gauge declaration scope.** The scaffold counts any entry in
   `declared_gauges`. Should "free DOF / fixed DOF" be a structured list per
   gauge, or is the count + prose semantics enough for R1? Recommendation:
   structured list, since D1-M3 discrimination depends on knowing which DOF
   are gauge-locked.

---

## 6. Acceptance report

This was a read-only inventory task. No files were edited, staged, or
committed. The deliverable is this document at the authoritative path.
