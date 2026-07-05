# First research slice — causal card (R0/R1)

Read-only selection + causal card. No repo edits, nothing staged. Branch
`yiwen_research`. All evidence below is from reading existing v19 artifacts
(JSON) plus one already-rendered still. Zero model inference, zero GPU, zero
re-run of any pipeline stage.

This card is the bridge between the raw v19 solver outputs and the two sibling
deliverables: the R1 field inventory (`01_graph_render_inventory.md`) and the
R0/R1 scaffold (`03_r0_r1_scaffold_implementation.md`,
`scripts/build_ego_hoi_sidecar_and_graph_health.py`). It selects the one narrow
slice on which the scaffold's decision table must fire on *real* data, and names
the mechanism that the current 4-branch table does not yet route.

---

## 0. Selected slice

| field | value |
|---|---|
| run root | `/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1` |
| clip | HOT3D `clip001850` pinhole, 150 frames, 30fps (development set: HOT3D GT hand+object pose exists for later R8 adjudication) |
| object | `keyboard` (rigid) |
| hand / window | **right hand, frames ~26–46** (grasp/type window; contained in the run's own `freeze_validation_frames_0032_0074` QC window) |
| why this run | most recent *complete* v19 run with the full three-solver stack (rigid pose graph + interval MANO + contact/nonpenetration) actually materialized; demo-regime v18 clips (tomato/trash) predate the v19 pose-graph/contact-graph structure and cannot exercise graph liveness |

Three coupled solver artifacts, all under the run root, are the entire evidence
surface for this slice:

- **Object pose graph:** `measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`
- **Hand interval MANO:** `measurements/mano_interval_correction/keyboard_0_149/hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/v18_joint_mano_interval_trajectory_state.json`
- **Contact / nonpenetration:** `measurements/contact_nonpenetration/keyboard_mano_object_constraint/v18_mano_object_constraint_state.json`
- Geometry epoch source: `measurements/geometry_completion/compact_keyboard_seed42/v18_compact_rigid_trellis_completion_report.json`
- Published render: `renders/v19_published_runtime/{v19_overlay,v19_world,v19_side_by_side}.mp4` + `v19_published_render_report.json`; still `state/freeze_validation_frames_0032_0074/side_by_side_frame_000032.jpg`

---

## 1. Artifact defect

The physical scene at frames ~26–46 is a person's **right hand typing on / resting
on a keyboard** — a genuine hand–object contact interaction. The published
annotation and the three solvers give **three mutually contradictory answers**
for the same hand–object relation over the same frames:

| measurement (right hand, frame 36–37 representative) | value | geometry it queries | claim |
|---|---|---|---|
| interval MANO `full_observed_surface_penetration_after_solver_m.max` | **0.102–0.107 m** | observed depth surface (hard barrier) | hand is 10 cm **inside** the object |
| contact/NP `penetrating_vertex_count` / `penetration_depth_m` | **0** | completed TRELLIS mesh (signed) | hand is **outside**, no penetration |
| render `contact_patch_final_abs_normal_gap_m` (right median) | **+0.039 m** (gap) | contact-patch target | hand is 39 mm **away** (a gap) |

Visible defect (still `side_by_side_frame_000032.jpg`): the overlay prints
`R: gap 39.1mm … penverts=0 … closed True` and the banner
`UNCERTAIN = not accepted contact closure`; the world panel shows the keyboard
as a **diffuse, over-sized green point cloud** (labelled `keyboard 56736 verts`,
the raw TRELLIS vertex count) with the cyan MANO hands floating beside it, not a
compact ~45×15×3 cm keyboard body. The published-report median is even worse:
`L gap 98.6mm closed False | R gap 39.2mm closed False`.

Two independent inertness/decoupling facts underlie this:

1. **The rigid pose graph is provably inert.** Its optimizer block:
   `success: true, nfev: 1, cost: 0.0, residual_rms_before: 0.0, residual_rms_after: 0.0`;
   every entry of `correction_summary` (`translation_delta_norm_m`,
   `rotation_delta_norm_rad`, step norms) is `median=p90=p95=max=mean=0.0`;
   `nonpenetration_target_frame_count: 0`. Output ≡ input. It "ran" and changed
   nothing.
2. **The measured penetration never leaves the interval solver.** The interval
   MANO solver *is* live (`optimizer_ran: true`, right-hand
   `corrected_frame_count: 57/150`, initial obs-surface penetration max
   `0.141 m` → after `0.107 m`), yet its net render-facing correction is ~0
   (`translation_delta_norm_m_median`: right `5.2e-20`, left `1.46e-7`) and its
   10 cm penetration signal is not consumed by the pose graph, the contact/NP
   state, or the render. `contact_patch_weight: 0.0` in the interval solver — the
   contact-patch factor is present but carries zero weight.

Net: the annotation the user consumes says "no contact / 39 mm gap / uncertain,"
driven by the **blind** contact/NP number, while a sibling solver's own state
says the hand is 10 cm inside the object. The graph that should reconcile them
does nothing.

---

## 2. Physical variable

Primary: **hand–object signed contact distance** for the right hand over frames
26–46 (the quantity that grounds contact state, ownership, and nonpenetration).
It is currently triple-valued (−107 mm, 0 mm, +39 mm) because it is computed
three times against **three different object geometries in different poses**, and
none of the three is propagated into the graph or reconciled.

Coupled variables that determine which of the three (if any) is physical:
- **Object geometry epoch identity** consumed by each contact test (observed
  depth surface vs 95.4%-TRELLIS completed mesh).
- **Object pose observability/freshness** (13/150 eligible frames; 72 rejected
  for mask extent 4–6× anchor; frozen at f77 for 78–149).
- **Hand metric depth** (HOT3D clip001850 has a known along-ray depth bias and a
  25–30 mm lateral floor per `hand_metric_mechanisms.md`).
- **Graph factor coupling** — whether the observed-surface penetration is wired
  as a live factor at all.

---

## 3. Live mechanisms (why the 10 cm penetration is invisible to the render/graph)

**M1 — Contact/NP silently disabled by non-watertight, wrong-epoch geometry.**
Contact/NP queries `keyboard_compact_rigid_completed_mesh_labeled.ply` with
`completed_surface_mesh_watertight: false`, `sign_mesh_watertight: false`, and
`sign_mesh_path == surface_mesh_path`. At frame 37 (right) the signed test
processed **`signed_query_candidate_vertex_count: 0`** of 64 hand vertices →
`penetrating_vertex_count: 0` by construction, not by physics. Open3D signed
raycasting on a non-watertight mesh returns unreliable inside/outside, so the
nonpenetration channel is effectively off.

**M2 — Geometry epoch contamination (observed *and* completed).** Completion
face provenance: `observed_depth_surface: 3221`, `unsupported_uncertain: 963`,
`trellis_inferred_hidden_surface: 86708` → the completed mesh consumed by
contact/NP is **95.4 % TRELLIS-inferred hidden surface, 3.5 % observed**, with
`free_space_rejected: 0` and `free_space_rejection_state:
not_evaluated_in_this_revision`. Meanwhile the interval solver's observed-surface
barrier may itself be contaminated (the pose report flags mask extent 4–6×
anchor as `probable_hand_background_leakage`). So the interval's 10 cm
"penetration" could be the hand pressing into hand/table points mis-labelled as
object surface, not real interpenetration. This is the same failure class named
in `pipeline_invariants.md` (V19 clip001851: `unsupported_uncertain` faces
concatenated into the accepted body).

**M3 — Object pose wrong / frozen.** Only 13/150 frames have a direct visible
pose fit; 72 are `visible_surface_ineligible_for_rigid_pose_fit`; frames 78–149
hold f77's pose bit-identically (`nearest_visible_pose_hold`,
`direct_visible_measurement: false`, gap up to 72 frames). Every hand–object
distance is computed against a pose that is either rejected-as-leaked or stale,
so all three contact numbers may be measuring against a wrong object placement.

**M4 — Missing factor coupling / gauge lock (graph inertness proper).** The pose
graph received `nonpenetration_target_frame_count: 0` because contact/NP emitted
`candidate_correction_count: 0`; the only live factors were visible-pose priors
that are trivially satisfied at their own measured values (cost 0, `nfev 1`).
The interval solver's `contact_patch_weight: 0.0` means its own contact factor is
zero-weighted. The graph is inert because **no live nonpenetration/contact factor
was ever wired in**, not because the physics is null.

**M5 — Cross-solver measurement decoupling (the mechanism the scaffold does not
yet route).** Two solvers measure the same contact against two geometries and
disagree by 10 cm, and neither result reaches the third solver or the render.
The scaffold's per-family `support_fraction` would show the contact/NP family at
~0 support and route `measurement_support_absent` — but that label *hides* the
fact that a strong penetration signal already exists in a sibling solver against
a different geometry. The defect is not "no measurement"; it is "the measurement
exists, is unreconciled across geometry sources, and is not coupled into the
graph or render."

These are not exclusive: current evidence already **confirms M1 + M4 as
proximate causes** (non-watertight sign mesh; zero NP targets; zero-weight
contact patch; nfev=1). M2 and M3 remain live as the reason the interval
solver's 10 cm signal may or may not be trustworthy once coupled. M5 is the
framing that selects the intervention.

---

## 4. Discriminating measurement

One lightweight experiment, no heavy inference, that separates "wire the existing
signal in" (M1/M4/M5) from "repair geometry/pose first" (M2/M3):

**Align all contact measurements onto a common (geometry, pose) and test agreement + spatial/temporal coherence of the penetrating set.**

Concretely, over right-hand frames 26–46, using only fields already on disk
(interval `per_frame_states[].optimized_vertices_world_sample_m`,
`optimized_joints_world_m`; the observed depth surface PLY; the completed mesh
PLY; the pose rows; the contact/NP per-row candidate counts):

- **D-a (geometry-source disagreement):** for each frame, recompute hand→surface
  signed distance of the same hand-vertex set against (i) observed depth surface
  and (ii) completed mesh, under the *same* per-frame object pose. Report the
  distribution of the difference.
- **D-b (penetrating-set coherence):** for the interval solver's
  observed-surface penetrating vertices, check whether they form a compact patch
  on fingertips/palm that persists across consecutive frames (temporal IoU of
  penetrating vertex ids), vs. a scattered/jumping set.
- **D-c (pose stratification):** stratify penetration/gap by pose provenance —
  observed-pose frames {30–36} vs held/frozen frames {37–46}.
- **D-d (free-space plausibility):** carve the observed depth points against the
  completed-mesh occupancy (light CPU, existing depth NPZ) and report the
  fraction of completed-mesh volume that free space contradicts; report the
  object AABB extent vs a keyboard prior (~0.45×0.15×0.03 m).
- **D-e (factor support/liveness audit):** run the R0/R1 scaffold's decision
  table on a v19 graph-summary adapter built from the three per-solver reports
  (see §6). Report per-family `support_fraction`, `active_residual_count`,
  `input_output_delta`.

---

## 5. Predictions per mechanism

| mechanism | D-a geometry disagreement | D-b coherence | D-c pose strata | D-d free-space / extent | D-e scaffold |
|---|---|---|---|---|---|
| **M1** sign-mesh disabled | large: signed-vs-completed ≈ 0 everywhere, observed shows −0.1 m; difference ≈ the full penetration | coherent (real fingertips) | high in both strata | extent inflated but penetration real | contact family support≈0 while interval penetration family support high → *decoupling*, not absent |
| **M2** geometry contamination | both surfaces disagree even after pose-align; observed surface itself has stray points near hand | penetrating set scattered / co-located with leaked points | uncorrelated with pose provenance | free-space rejects large fraction; extent 4–6× anchor | support present but on contaminated faces |
| **M3** pose wrong/frozen | disagreement tracks pose gap_frames | penetration jumps at pose holds | **systematically larger in held frames {37–46}** | extent anomaly concentrated where pose rejected | pose family low active residual on rotation |
| **M4** missing coupling / gauge lock | n/a (measurement exists, just unrouted) | coherent | either | plausible | pose graph energy_delta=0 with support present → `graph_inert`; NP targets=0 |
| **M5** cross-solver decoupling | interval and contact/NP each internally consistent but mutually contradictory | interval coherent, contact/NP empty | either | either | scaffold routes `measurement_support_absent` for contact family, MISSING that interval family carries the signal → card adds a `cross_source_disagreement` branch |

**Already-observed partial adjudication** (from §1–§3 numbers, no new run):
- D-c is *already answered*: penetration is high in **both** observed (f36:
  0.107 m) and held (f37: 0.102 m) frames → **pose freeze alone is not the
  driver** (weakens M3 as sole cause).
- D-e is *already answered* for the pose graph: energy_delta=0 with 13 supported
  prior frames and `nfev=1` → **`graph_inert` via missing NP factor** (confirms
  M4), and contact/NP `candidate_correction_count=0` → contact family support ≈ 0
  (confirms M1/M5).

So the remaining live fork is **M1/M5 (wire it in) vs M2 (geometry too
contaminated to trust)**, adjudicated by D-a + D-b + D-d.

---

## 6. Exact lightweight commands / files to inspect

All read-only; write any scratch to `/tmp`, never the repo or run root.

```bash
R=/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1

# (1) Pose-graph inertness + support (nfev/cost/deltas/NP targets/observed frames)
python3 - <<'PY'
import json
pg=json.load(open(f"{__import__('os').environ['R']}/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json"))
print('optimizer', pg['optimizer'])
print('NP target frames', pg['nonpenetration_target_frame_count'])
print('graph_frames', pg['graph_frames'])
print('corrections nonzero?', any(pg['correction_summary'][k]['max']>0 for k in pg['correction_summary']))
PY

# (2) Interval solver liveness + per-frame right-hand observed-surface penetration
#     fields: full_observed_surface_penetration_after_solver_m, delta_norms, contact_patch_weight
#     file:  .../mano_interval_correction/keyboard_0_149/.../v18_joint_mano_interval_trajectory_state.json

# (3) Contact/NP blindness: candidate_correction_count, sign_mesh_watertight,
#     signed_query_candidate_vertex_count, penetrating_vertex_count per row
#     file:  .../contact_nonpenetration/keyboard_mano_object_constraint/v18_mano_object_constraint_state.json

# (4) Geometry epoch provenance: face_label_counts, mesh_counts, free_space_rejection_state, watertight
#     file:  .../geometry_completion/compact_keyboard_seed42/v18_compact_rigid_trellis_completion_report.json

# (5) Visible-pose support fraction: fit_frame_count / missing_pose_count / ineligible_pose_observation_count
#     file:  .../pose_fits/keyboard_visible_pose_fit/v18_compact_rigid_object_pose_fit_report.json

# (6) Render truth: v19_published_render_report.json -> metrics.sides.{left,right}
#     still: state/freeze_validation_frames_0032_0074/side_by_side_frame_000032.jpg

# (7) Run the R0/R1 scaffold on this slice via a v19 graph-summary adapter (light, /tmp out).
#     The base annotation has NO factor_graph_summary (confirmed), so build one:
#     adapter reads the three per-solver reports and emits a factor_graph_summary with
#     factor_family={visible_pose_prior:{declared:13,supported:13},
#                    nonpenetration:{declared:0,supported:0},
#                    contact_patch:{declared:300,supported:0},
#                    interval_observed_penetration:{declared:150,supported:57}},
#     objective={energy_initial:0.0,energy_after:0.0} for the pose graph.
#     Then: python3 scripts/build_ego_hoi_sidecar_and_graph_health.py \
#             --graph-summary /tmp/clip001850_v19_graph_summary.json --out-dir /tmp/egohoi_slice
#     Predicted decision: measurement_support_absent (contact family) — which this card
#     upgrades to cross_source_disagreement because the interval family DOES have support.
```

The D-a/D-b/D-d geometry work needs only `trimesh`/`open3d` distance queries on
already-materialized PLYs + the existing depth NPZ under
`measurements/depth_slam/` — CPU-only, seconds, no model, no GPU.

---

## 7. Implementation route per outcome

The routes populate the **ego.hoi R0 tables** and the **R1 graph-health row**,
and select which of the sibling patch set (`01_graph_render_inventory.md` P1–P6)
to land first.

**If D-a large + D-b coherent + D-d extent inflated but penetration real → M1/M5
(wire the existing signal in).**
- R0: `contact_frame_detail.parquet` carries *both* source distances per
  frame/hand (`observed_surface_signed_m`, `completed_mesh_signed_m`) with a
  `source_disagreement_m` column and a `contact_state` posterior derived from the
  observed-surface source (source of truth) conditioned on visibility.
- R1: add a `cross_source_disagreement` branch to the scaffold decision table
  (between `measurement_support_absent` and `graph_inert`): fires when one factor
  family has support and a sibling family measuring the same variable has ~0
  support against a different geometry epoch. Route: reconcile geometry source,
  do not label "support absent."
- Patch: P3 (interval solver emits per-family support so the observed-penetration
  family is visible) + P2 (pose graph accepts an observed-surface nonpenetration
  target so `nonpenetration_target_frame_count` > 0) + contact/NP switches its
  signed query to the observed depth surface (or a watertight sign mesh).
- Expected state change: pose graph `nfev>1`, `energy_delta<0`, nonzero
  `translation_delta`; render right-hand `active_set_closed: true` with a small
  signed gap (mm-scale), `penverts` reflecting the observed-surface count; world
  panel object body sized to the observed keyboard, not the diffuse TRELLIS cloud.

**If D-a disagreement persists after pose-align + D-b scattered + D-d free-space
rejects large fraction → M2 (repair geometry epoch first).**
- R0: `geometry_epochs.parquet` records face provenance (`observed 3221 /
  unsupported 963 / trellis 86708`), `free_space_rejected` count (currently 0 →
  must be evaluated), watertight flags, and an `accepted_body_face_ids` subset
  that **excludes** `unsupported_uncertain` + un-carved TRELLIS from the contact
  body (per `pipeline_invariants.md`).
- R1: `stale_dependency_count` / geometry-epoch lineage on the contact rows;
  contact posterior routed to `unresolved` while the epoch is contaminated
  (uncertainty as a state variable, not omission).
- Patch: P4 (render-state builder already rejects `unsupported_uncertain`; extend
  to require free-space evaluation before a mesh is contact-eligible) + R3
  geometry-epoch plumbing. Do **not** wire the interval penetration into the
  graph until the observed surface is decontaminated — that would optimize the
  hand into leaked points.
- Expected state change: contact rows flip to `unresolved` with provenance;
  world panel object shrinks to the carved observed body; the 10 cm penetration
  either collapses (was leakage) or survives (was real) after carving —
  either way the number becomes trustworthy.

**If D-c had shown held-frame penetration ≫ observed-frame → M3 (pose first).**
(Currently *disfavored*: penetration is high in both strata.) Route would be:
repair pose observability (D1 workstream) — more visible-pose frames, per-DOF
covariance, gauge declaration — before trusting any contact number; R1 emits
`declared_gauges` + pose-family `active_residual_count` (Patch P2).

**Confirmed-now: M4 (graph inert via missing factor).** Independent of the
fork, the pose graph must stop reporting `annotation_ready: true` while
`nfev=1, cost=0, NP targets=0`. R1 graph-health row records
`objective_energy_delta=0` + `nonpenetration_target_frame_count=0` → decision
`graph_inert` → route "repair variable wiring": the pose graph needs a live
nonpenetration/contact factor sourced from the reconciled measurement above.

---

## 8. Expected table / render state change (acceptance target for the first build)

- `ego.hoi/geometry_epochs.parquet`: one keyboard epoch row with real face
  provenance, `free_space_rejected` populated (not `not_evaluated`), watertight
  flags, extent-ratio-to-anchor, `accepted_body_face_ids`.
- `ego.hoi/object_pose.parquet`: 150 rows with `pose_source`
  (visible/held/interp), `direct_visible_measurement`, `gap_frames`,
  per-DOF observability; support fraction 13/150 exposed, 72 ineligible flagged.
- `ego.hoi/contact_frame_detail.parquet`: right-hand frames 26–46 carry
  `observed_surface_signed_m`, `completed_mesh_signed_m`, `source_disagreement_m`,
  `visibility_state`, and a `contact_state` posterior — the triple-valued
  contradiction becomes one reconciled row with explicit uncertainty.
- `ego.hoi/graph_solutions.parquet` (graph-health row): pose-graph
  `objective_energy_delta`, per-family `support_fraction`
  (visible_pose_prior 1.0, nonpenetration 0.0→>0 after fix, interval_observed
  0.38), `nfev`, `nonpenetration_target_frame_count`, `mechanism_decision`.
- Render (`v19_overlay/world/side_by_side`): right-hand contact label driven by
  the reconciled observed-surface contact state — `closed`/`penverts` reflect the
  observed geometry, and the world object body is keyboard-sized. The current
  `gap 39mm / penverts=0 / not accepted` (blind contact/NP) must be replaced by
  or reconciled with the observed-surface evidence, with uncertainty styling
  where the epoch is contaminated.

---

## 9. Why this slice drives R0/R1 (not a survey)

- It forces the R0 `ego.hoi` sidecar to carry the exact columns that make the
  contradiction representable (dual-source signed distance, epoch provenance,
  pose provenance) — R0 tables designed from a real defect, not a schema guess.
- It forces the R1 graph-health instrumentation to distinguish "no measurement"
  from "measurement exists but is decoupled across geometry sources" — a branch
  the current scaffold lacks, discovered because this slice has a *live,
  supported* penetration family sitting next to a *zero-support* contact family
  for the same variable.
- It fires the scaffold decision table on real v19 data for the first time
  (via the three-report adapter), and immediately exposes that the v19 run emits
  **no unified `factor_graph_summary`** — the concrete R1 wiring gap.
- Every subsequent build is measured against the frame-26–46 render + the
  reconciled `contact_frame_detail` rows, not against a validator or row count.

---

## 10. Provenance / freshness flags observed (for R1 to capture, not to fix here)

- Interval solver `scientific_test` string says **"tomato geometry"** inside a
  *keyboard* clip → template copy-paste; a freshness/provenance smell the R1
  instrumentation should catch (state text not specialized to the active object).
- `sign_mesh_path == surface_mesh_path` with `sign_mesh_watertight: false` →
  the signed nonpenetration channel is nominally present but physically inert; a
  support-fraction audit must mark it unsupported rather than "measured".
- Pose report `status: corrected_pose_graph_surface_preserved_no_nonpenetration_gain`
  and `annotation_ready: true` while the optimizer did nothing (`nfev: 1`) →
  "annotation_ready" must not be assertable without a nonzero input-output delta.

---

## Acceptance report

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Delivered exactly one causal card at the authoritative path selecting one narrow slice (HOT3D clip001850 keyboard, right hand, frames 26-46, run v5_focalfix_coordrigid_v1). Card contains all requested sections: selected slice, artifact defect, physical variable, live mechanisms (M1-M5), discriminating measurements (D-a..D-e), predictions per mechanism, exact lightweight read-only commands/files, and implementation route per outcome tied to ego.hoi R0 tables + R1 scaffold decision table + sibling patch set. No repo edits; no scope widening beyond selection + card."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Every claim is backed by a cited field from a named on-disk artifact with exact numbers: pose graph nfev=1/cost=0.0/all deltas 0.0/NP targets 0; interval right-hand penetration 0.141->0.107 m, corrected 57/150, contact_patch_weight 0.0, render translation_delta 5.2e-20; contact/NP candidate_correction_count 0, penetrating_vertex_count 0, sign_mesh_watertight false, signed_query_candidate_vertex_count 0; completion face provenance observed 3221/unsupported 963/trellis 86708 (95.4% TRELLIS), free_space_rejected 0/not_evaluated; visible pose fit 13 eligible/72 ineligible/65 missing of 150; render report right gap 39.2mm active_set_closed false. All reproducible via the read-only commands in section 6."
    }
  ],
  "changedFiles": [
    ".memory/tasks/2026-07-06-research-track-execution/subagents/02_first_slice_causal_card.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "ls/find on /data2/ego_annotation_outputs/v19_runs + v18_* (artifact discovery)", "result": "passed", "summary": "located most-recent complete v19 run: 20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"},
    {"command": "read v19_rigid_object_pose_graph_report.json (optimizer/correction_summary/graph_frames/frozen frames)", "result": "passed", "summary": "pose graph inert: nfev=1, cost=0.0, all corrections 0.0, NP targets 0, 13/150 frames, 78-149 frozen at f77"},
    {"command": "read v18_joint_mano_interval_trajectory_state.json (summary/intervals/per_frame_states)", "result": "passed", "summary": "interval solver live but decoupled: right penetration 0.141->0.107 m, corrected 57/150, contact_patch_weight 0.0"},
    {"command": "read v18_mano_object_constraint_state.json (summary_by_side/constraint_rows)", "result": "passed", "summary": "contact/NP blind: candidate_correction_count 0, penetrating_vertex_count 0, sign_mesh non-watertight, signed_query_candidate 0"},
    {"command": "read v18_compact_rigid_trellis_completion_report.json (face_label_counts/mesh_counts/free_space)", "result": "passed", "summary": "completed mesh 95.4% TRELLIS hidden surface, free_space_rejected 0 / not_evaluated"},
    {"command": "read v19_published_render_report.json + side_by_side_frame_000032.jpg", "result": "passed", "summary": "render publishes blind contact/NP: right gap 39.2mm, active_set_closed false, penverts=0; world object a diffuse oversized cloud"},
    {"command": "python3 scripts/build_ego_hoi_sidecar_and_graph_health.py --help; inspect base annotation for factor_graph_summary", "result": "passed", "summary": "scaffold CLI confirmed; base annotation has NO factor_graph_summary -> v19 emits only per-solver reports, exposing the R1 aggregation gap"}
  ],
  "validationOutput": [
    "Slice contradiction reproduced numerically: same right-hand contact at f36-37 is -107mm (interval, observed surface), 0mm (contact/NP, completed TRELLIS mesh), +39mm (render, contact-patch) -> three geometries, three answers.",
    "Graph inertness confirmed from optimizer block (nfev=1, cost 0.0, residual 0->0, all deltas 0.0) with nonpenetration_target_frame_count=0.",
    "Pose support fraction 13/150 with 72 ineligible (mask extent 4-6x anchor, hand/bg leakage) and 78-149 frozen at f77 confirmed.",
    "Mechanism M4 (graph inert via missing factor) and M1 (sign-mesh disabled) confirmed now; M2 vs M1/M5 fork left as the designed discriminating experiment (D-a/D-b/D-d), all runnable CPU-only with zero model inference."
  ],
  "residualRisks": [
    "The card selects and diagnoses; it does not run the D-a/D-b/D-d geometry experiment (that is the first implementation step it routes to). The experiment is specified as CPU-only, no heavy inference.",
    "HOT3D GT hand/object pose exists for this clip but is intentionally not consumed at R0/R1 (GT-free consistency first, per task constraint); GT adjudication is deferred to R8.",
    "Running the scaffold on real v19 data requires a small three-report -> factor_graph_summary adapter (the base annotation lacks factor_graph_summary); specified in section 6, not yet built."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new task-memory document (the causal card) at the authoritative subagents path. No source files touched; git working tree otherwise unchanged.",
  "reviewFindings": [
    "no blockers: read-only selection + card; all numeric claims cite exact on-disk fields; no edits, nothing staged.",
    "note: card intentionally proposes a 5th decision-table branch (cross_source_disagreement) not present in 03's scaffold; this is a routed R1 follow-on, flagged for the parent, not an edit to the scaffold."
  ],
  "manualNotes": "This card is the measurement-design bridge between 01_graph_render_inventory.md (producer field locations / patches P1-P6) and 03_r0_r1_scaffold_implementation.md (decision-table scaffold). Key cross-cutting finding for the parent: the v19 run emits three independent per-solver reports and NO unified factor_graph_summary, and two of them (interval MANO vs contact/NP) measure the same hand-object contact against different geometry epochs and disagree by ~10cm while the pose graph that should reconcile them is provably inert. The first artifact-changing build is the CPU-only D-a/D-b/D-d reconciliation experiment feeding the R0 contact_frame_detail + geometry_epochs tables; land pose-graph liveness (Patch P2) + interval per-family support (Patch P3) so the reconciled penetration can become a live nonpenetration factor."
}
```
