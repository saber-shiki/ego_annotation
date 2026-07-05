# 04 — clip001850 right-hand geometry-source reconciliation (D-a/D-b/D-d) — implementation result

First artifact-changing CPU-only reconciliation for HOT3D `clip001850` keyboard,
right hand, frames 26–46, against run
`20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.
Mechanisms discriminated: **M1/M5** (coherent observed-surface contact evidence
stranded across solvers) vs **M2** (geometry contamination).

Script: `scripts/reconcile_clip001850_geometry_sources.py` (new, untracked, not
staged; no existing files touched).

Artifacts produced (all under `/tmp/clip001850_geometry_reconciliation/`):

| file | contents |
|---|---|
| `contact_frame_detail.ndjson` | 21 ego.hoi-style per-frame rows (frames 26–46), every required field populated or reason-stated |
| `geometry_reconciliation_summary.json` | window-level `ego.hoi/0.1.0/graph_solutions` row: triple-valued contradiction, D-a disagreement distribution, D-b coherence, D-d extent plausibility, geometry-epoch provenance, graph liveness, mechanism decision |
| `contact_frame_detail_table.tsv` | human-readable per-frame table for review |

## 1. Causal card → measurement → result

The causal card (`02_first_slice_causal_card.md`) named the defect: the same
right-hand contact relation is triple-valued over 26–46 — interval MANO reports
~0.10 m penetration into the observed depth surface, contact/NP reports 0
penetrating vertices against the completed TRELLIS mesh, and the render
publishes a +0.039 m gap. The fork to adjudicate was M1/M5 (wire the stranded
signal in) vs M2 (repair geometry contamination first), tested by D-a, D-b, D-d.

This implementation **ran** D-a/D-b/D-d on real state and produced a joined
per-frame cross-source ledger that resolves the fork with measured numbers.

## 2. What the measurement changed in the causal model

### D-a — geometry-source disagreement: LARGE and confirmed
For the **same per-frame hand-vertex set** (interval solver's 160 sampled MANO
vertices, world frame) under the **same per-frame object pose**
(`rotation_world_from_completed_canonical_matrix` + `translation_world_m` from
the pose graph), recomputed via trimesh `closest_point` against the two meshes
in canonical frame:

- Observed-surface penetration (interval solver, published): window max
  median **0.0945 m**, max **0.1068 m** (14/21 frames carry a positive
  observed-surface penetration).
- Completed-mesh unsigned distance (recomputed): window median **0.0460 m**,
  i.e. the hand sits ~4.6 cm *outside* the completed mesh by unsigned distance.
- **Geometry-source disagreement** (observed penetration max + completed
  unsigned median, per frame): window median **0.135 m**, max **0.186 m**.

The two geometries give mutually contradictory answers for the same hand–object
relation on the same pose. This is exactly the M5 signature: each solver is
internally consistent, the two are mutually contradictory, and neither reaches
the render/graph.

### D-a smoking gun for M1: the completed-mesh signed query is unreliable *and* disabled
Best-effort trimesh `signed_distance` on the **non-watertight** completed mesh
(`is_watertight=False`, confirmed) classifies the hand vertices as *inside*
(median −0.054 m at f36), while contact/NP correctly refuses to sign
(`signed_query_candidate_vertex_count = 0` on 18/21 frames, `penetrating_vertex_count = 0` everywhere, `sign_mesh_watertight=False`). The contradiction between the
negative best-effort sign and the positive unsigned distance is itself the
non-watertight unreliability that disabled the channel. **M1 confirmed**: the
completed-mesh signed nonpenetration channel is inert by construction, not by
physics — the contact evidence exists in a sibling solver against a different
geometry source.

### D-b — penetrating-set coherence: compact/persistent where it fires
6/21 frames carry contact-patch vertex ids. Temporal IoU: f31→f32 = **0.37**,
f32→f33 = **0.62**, f45→f46 = **0.0**; mean IoU **0.33** → interpretation
`compact_persistent_patch`. The id union (76 ids, range 1–777) covers fingertip
and palm regions. The penetrating set is persistent in the f31–33 typing cluster
but intermittent across the window. **D-b supports real contact rather than
scattered leakage in the coherent sub-interval, but does not by itself confirm
the whole window.**

### D-d — extent/free-space plausibility: M2 co-condition confirmed
Completed-mesh AABB extents **0.253 × 0.664 × 0.260 m** vs the keyboard prior
(0.45 × 0.15 × 0.03 m): sorted-extent ratios **[1.48, 4.43, 8.68]** → the
completed mesh is **~8.7× too thick** in the thin axis and ~4.4× in the mid
axis. Face provenance: **95.4 % TRELLIS-inferred hidden surface, 3.5 %
observed**, `free_space_rejected = 0`, `free_space_rejection_state =
not_evaluated_in_this_revision`. The completed mesh is inflated relative to a
keyboard and its free-space validity was never evaluated. **M2 is a live
co-condition**: the completed mesh cannot be trusted as a contact body until its
epoch is repaired — but the observed-surface penetration lives on a *different*
geometry source, so M2 does not refute M1/M5.

### M4 — graph inertness: confirmed (unchanged, now measured)
Pose graph `optimizer`: `nfev=1, cost=0.0, residual_rms 0→0`;
`nonpenetration_target_frame_count=0`; `status=
corrected_pose_graph_surface_preserved_no_nonpenetration_gain`;
`annotation_ready=true`. `graph_inert = true`. No live contact/NP factor was
ever coupled into the graph despite a penetration family carrying support.

## 3. Mechanism decision (selected by the measurement)

1. **M1/M5 confirmed (proximate route)** on 12/21 frames: observed-surface
   penetration is stranded in the interval solver while contact/NP emits zero
   penetrating vertices for the same frames because its signed query is disabled
   by the non-watertight sign mesh → **cross-solver geometry-source decoupling,
   not measurement absence.** This selects wiring the observed-surface contact
   evidence into contact/NP + the pose graph (a `cross_solver_geometry_decoupled`
   graph-health route), *not* relabelling the contact family as "support
   absent."
2. **M2 co-condition**: completed-mesh extent inflated (~8.7× keyboard
   thickness); repair the geometry epoch before treating the completed mesh as a
   contact body. The observed-surface penetration is on a different geometry, so
   this is a parallel repair, not a blocker for the M1/M5 route.
3. **M4 confirmed**: pose graph inert via missing factor; stop asserting
   `annotation_ready` while `nfev=1`, `cost=0`, NP targets=0.
4. **D-b**: penetrating set compact/persistent in the f31–33 sub-interval;
   real-contact support, intermittent window-wide.

**Net fork resolution:** M1/M5 is the proximate mechanism (wire the existing
observed-surface signal in); M2 is a confirmed parallel co-condition (repair the
inflated completed mesh epoch) that must not be optimized against as a contact
body until carved.

## 4. Output field coverage (required per-frame fields)

Every required field is present in each `contact_frame_detail.ndjson` row, with
the exact value or the blocking reason:

| required field | source | status |
|---|---|---|
| `frame_index`, `hand_side` | — | present |
| `observed_surface_penetration_m` | interval `full_observed_surface_penetration_after_solver_m` (count/median/p90/p95/max/mean) | present (null-distribution on frames where interval reported 0 penetrating) |
| completed-mesh/contact/NP signed/candidate fields | `contact_np_signed_query_candidate_vertex_count`, `contact_np_penetrating_vertex_count`, `contact_np_penetration_depth_m`, `contact_np_signed_distance_m`, `contact_np_nearest_surface_unsigned_m`, `contact_np_sign_mesh_watertight`, `contact_np_reason` | present |
| `render_gap_m` | interval `contact_patch_final_normal_gap_m` per frame (the value the render republishes; render report carries only a per-side median) | present (null where interval reported no contact-patch rows) |
| `geometry_source_disagreement_m` | recomputed: observed penetration max + completed-mesh unsigned median, same vertices + pose | present on 14/21 frames; reason stated elsewhere |
| `penetrating_vertex_coherence` | temporal IoU of `contact_patch_vertex_ids` (MANO sample ids) | present; window IoU in summary |
| `pose_provenance` | pose graph `temporal_pose_graph` (pose_source, direct_visible_measurement, gap_frames, deltas, NP target/weight) | present |
| `face_provenance_summary` | completion `face_label_counts.completed_mesh` (observed/unsupported/trellis with fractions) | present |
| watertight flags | completion + contact/NP + trimesh `is_watertight` | present |
| decision route | M1_M5 / no_penetration / unresolved per frame | present |

**Mesh-distance recomputation was NOT blocked.** `optimized_vertices_world_sample_m`
(160 sampled MANO vertices), `rotation_world_from_completed_canonical_matrix`,
`translation_world_m`, and both PLYs are all present; trimesh is available in the
project `.venv`. The only field that cannot be made reliable is a *trustworthy*
signed distance against the completed mesh, because it is non-watertight — and
that unreliability is itself the measured evidence for M1.

## 5. How to reproduce

```bash
cd /home/yiwen/ego_annotation
/home/yiwen/ego_annotation/.venv/bin/python \
  scripts/reconcile_clip001850_geometry_sources.py \
  --frame-lo 26 --frame-hi 46 --hand-side right \
  --out-dir /tmp/clip001850_geometry_reconciliation
# -> contact_frame_detail.ndjson (21 rows)
# -> geometry_reconciliation_summary.json
# -> contact_frame_detail_table.tsv
```

CPU-only (trimesh `closest_point`/`signed_distance` on two existing PLYs, 160
verts × 21 frames). No model inference, no GPU, no edits to run-root artifacts.

## 6. Implications for the next build (R0/R1)

- `contact_frame_detail` must carry **both** `observed_surface_signed_m` and
  `completed_mesh_signed_m` plus `source_disagreement_m` per frame/hand (the
  schema this script emits). The contact posterior should be derived from the
  observed-surface source conditioned on visibility, not from the blind
  completed-mesh signed query.
- R1 graph-health needs the `cross_solver_geometry_decoupled` decision route
  (fires when one factor family has support and a sibling family measuring the
  same variable has ~0 support against a different geometry epoch) — this run's
  data fires it on 12/21 frames. The generic `measurement_support_absent` label
  would hide the stranded signal.
- `geometry_epochs` must record face provenance (95.4 % TRELLIS), watertight
  flags, `free_space_rejected` (currently 0 / not-evaluated), and extent ratio to
  prior; contact rows route to `unresolved` against the contaminated completed
  mesh until the epoch is carved.
- Pose graph must stop asserting `annotation_ready` while `nfev=1`, `cost=0`,
  NP targets=0 (M4); it needs a live nonpenetration/contact factor sourced from
  the reconciled observed-surface measurement.

## 7. Residual risks / open

- The interval solver's `scientific_test` string still says "tomato geometry"
  inside a keyboard clip (template freshness smell); captured in the causal
  card, not fixed here.
- HOT3D GT hand/object pose exists for this clip but is intentionally not
  consumed (GT-free consistency first, per task constraint); GT adjudication
  deferred to R8.
- The intermittent observed-surface penetration (only 14/21 frames positive, 6
  with contact-patch vertex ids) means the stranded signal is strong in the
  f31–33 cluster but patchy window-wide; wiring it in must preserve uncertainty
  on the empty-penetration frames.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Implemented exactly the requested CPU-only D-a/D-b/D-d reconciliation as one new self-contained script (scripts/reconcile_clip001850_geometry_sources.py) that joins interval-MANO observed-surface penetration, contact/NP completed-mesh signed/candidate fields, completion face-provenance, pose provenance, render gap, and recomputed trimesh distances into ego.hoi-style contact_frame_detail/geometry provenance NDJSON for right-hand frames 26-46. No scope widening: no existing files modified, no new factors/modules, run-root artifacts untouched, no heavy inference, no GPU."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Produced three concrete artifacts under /tmp/clip001850_geometry_reconciliation/ (contact_frame_detail.ndjson 21 rows, geometry_reconciliation_summary.json, contact_frame_detail_table.tsv). Every required per-frame field is populated with a value or a blocking reason. Key measured numbers: D-a geometry-source disagreement window median 0.135 m / max 0.186 m (observed penetration max median 0.0945 m vs completed-mesh unsigned distance median 0.0460 m, same 160 hand vertices + same per-frame pose); M1 smoking gun = completed-mesh best-effort signed distance is negative (inside) while contact/NP signed_query_candidate=0 on 18/21 frames and sign_mesh_watertight=False; D-b temporal IoU mean 0.33 (f31-33 cluster 0.37/0.62) -> compact_persistent_patch; D-d completed mesh AABB 0.253x0.664x0.260 m = 8.7x too thick vs keyboard prior, 95.4% TRELLIS faces, free_space_rejected=0/not_evaluated; M4 pose graph nfev=1 cost=0 NP targets=0. Mechanism decision: M1/M5 proximate (12/21 frames), M2 co-condition, M4 confirmed."
    }
  ],
  "changedFiles": [
    "scripts/reconcile_clip001850_geometry_sources.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "/home/yiwen/ego_annotation/.venv/bin/python scripts/reconcile_clip001850_geometry_sources.py --frame-lo 26 --frame-hi 46 --hand-side right --out-dir /tmp/clip001850_geometry_reconciliation",
      "result": "passed",
      "summary": "Emitted 21 contact_frame_detail rows + window summary + TSV; mechanism decision M1/M5 proximate (12/21), M2 co-condition, M4 confirmed; mesh-distance recomputation NOT blocked (vertices+pose+PLYs+trimesh all present)"
    },
    {
      "command": "read v19_rigid_object_pose_graph_report.json pose_rows[26..46]",
      "result": "passed",
      "summary": "frames 26-29 nearest_visible_pose_hold, 30-36 direct_visible_pose_observation_corrected, 37-46 interpolated; per-frame rotation_world_from_completed_canonical_matrix + translation_world_m confirmed orthonormal (det=1)"
    },
    {
      "command": "read interval MANO per_frame_states right 26-46",
      "result": "passed",
      "summary": "optimized_vertices_world_sample_m (160 sampled) + optimized_vertices_sample_ids + contact_patch_vertex_ids + full_observed_surface_penetration_after_solver_m present per frame"
    },
    {
      "command": "read contact/NP constraint_rows right 26-46",
      "result": "passed",
      "summary": "sign_mesh_watertight=False all frames, signed_query_candidate_vertex_count=0 on 18/21, penetrating_vertex_count=0 all, signed_distance_m=None, nearest_surface_unsigned median ~0.045-0.104 m"
    },
    {
      "command": "trimesh.load completed + observed PLY",
      "result": "passed",
      "summary": "completed 46403 verts/90892 faces non-watertight AABB 0.253x0.664x0.260 m; observed 2179 verts/4184 faces AABB 0.252x0.664x0.179 m"
    }
  ],
  "validationOutput": [
    "D-a geometry-source disagreement is large and consistent: on frames with observed penetration, disagreement (penetration_max + completed_unsigned_median) ranges 0.114-0.186 m, window median 0.135 m -> two geometries give contradictory answers for the same hand+pose.",
    "M1 confirmed: best-effort trimesh signed_distance on non-watertight completed mesh classifies hand as inside (median -0.054 m at f36) while contact/NP signed_query_candidate=0/penetrating=0 with sign_mesh_watertight=False -> signed query disabled by construction, not by physics.",
    "M2 co-condition confirmed: completed-mesh sorted-extent ratio to keyboard prior [1.48, 4.43, 8.68] -> ~8.7x too thick; 95.4% TRELLIS faces; free_space_rejected=0/not_evaluated.",
    "D-b: temporal IoU mean 0.33 (f31->f32 0.37, f32->f33 0.62, f45->f46 0.0) -> compact_persistent_patch in the f31-33 typing cluster, intermittent window-wide (6/21 frames carry ids).",
    "M4 confirmed: pose graph optimizer nfev=1, cost=0.0, residual 0->0, nonpenetration_target_frame_count=0, graph_inert=true, annotation_ready=true (contradiction).",
    "Required per-frame fields all present; mesh-distance recomputation NOT blocked (optimized_vertices_world_sample_m + rotation/translation_world_m + both PLYs + trimesh all available); only a trustworthy completed-mesh signed distance is impossible because the mesh is non-watertight, which is itself the measured M1 evidence."
  ],
  "residualRisks": [
    "D-b coherence is strong only in the f31-33 sub-interval; the observed-surface signal is patchy window-wide (14/21 frames positive penetration), so wiring it into the graph must preserve uncertainty on empty-penetration frames.",
    "The completed-mesh best-effort signed distance is unreliable (non-watertight); it is reported as evidence of the sign-channel failure, NOT as a trusted signed contact distance.",
    "Interval solver scientific_test string still says 'tomato geometry' in a keyboard clip (freshness smell from causal card, not fixed here).",
    "GT adjudication deferred to R8 per task constraint (GT-free consistency first)."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new untracked script scripts/reconcile_clip001850_geometry_sources.py (~37KB) that performs CPU-only D-a/D-b/D-d reconciliation and emits ego.hoi-style contact_frame_detail NDJSON + summary + TSV. No existing files modified; nothing staged.",
  "reviewFindings": [
    "no blockers: script is new/untracked; nothing staged; no existing files touched; no heavy inference; all numeric claims cite on-disk fields and are reproducible by the command in section 5.",
    "note: completed_surface_mesh_watertight in the summary uses the completion-report value while sign_mesh_watertight is sourced authoritatively from the per-frame contact/NP row (False); both are captured separately so the decision route uses the contact/NP value correctly.",
    "note: render_gap_m per frame is sourced from the interval solver's contact_patch_final_normal_gap_m (the value the render republishes); the render report itself carries only a per-side median (R gap 0.0392 m), which is reported as render_published_gap_median_m."
  ],
  "manualNotes": "Fork resolved: M1/M5 is the proximate mechanism (wire the existing observed-surface contact signal into contact/NP + the pose graph via a cross_solver_geometry_decoupled graph-health route), with M2 as a confirmed parallel co-condition (the completed TRELLIS mesh is ~8.7x too thick and must be carved before it is used as a contact body) and M4 confirmed (inert graph). The next build should: (1) add cross_solver_geometry_decoupled to the R1 scaffold decision table and fire it on these 12/21 frames, (2) make contact_frame_detail carry both observed_surface_signed_m and completed_mesh_signed_m + source_disagreement_m (schema this script emits), (3) route contact posterior to observed-surface source conditioned on visibility, and (4) repair the pose graph annotation_ready assertion while nfev=1/cost=0/NP targets=0. Mesh-distance recomputation was NOT blocked by missing vertex arrays or libraries — all required inputs and trimesh are present."
}
```
