# clip001850 reconciliation — artifact-level attack (R0/R1)

Read-only. No repo edits, nothing staged. Branch `yiwen_research`. Every number
below is read from on-disk v19 artifacts and the two new scaffold scripts. Zero
model inference, zero GPU.

Run root (`$R`):
`/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`

Target of attack: the planned CPU-only reconciliation in
`subagents/02_first_slice_causal_card.md` §7 (route "M1/M5 — wire the existing
observed-surface penetration into contact/NP") + the `cross_solver_geometry_decoupled`
graph-health branch produced by `scripts/build_clip001850_ego_hoi_graph_summary.py`
→ `scripts/build_ego_hoi_sidecar_and_graph_health.py`.

---

## Top-line

The reconciliation as planned collapses **three numbers that are each not a
trustworthy keyboard-contact measurement** into one well-provenanced row plus a
decision, and then routes to "wire the observed-surface penetration in." Under
the actual artifact:

- the interval "10 cm penetration into observed keyboard surface" is penetration
  into the **full-frame UniDepth map** (table + the hand's own monocular depth +
  background), with hand-owned quarantine **off**, object-mask gating **off**,
  eligibility **off**, and the physically-correct depth-order-vs-object term
  selecting **0 vertices** across all 150 frames;
- the contact/NP "0 penetration" is **0 by construction** — a signed query with
  0 candidate vertices against a 95.4 %-TRELLIS **non-watertight** mesh whose
  free space was never evaluated;
- the render "+39 mm gap" is the interval **contact-patch** channel (a third
  geometry/target), and the same interval solver reports **+68 mm gap** and
  **−107 mm penetration on the same hand at the same frame f36**.

Wiring the observed-surface penetration in (the plan's primary route) optimizes
the MANO hand against the table plane and its own depth. The reconciled
`contact_frame_detail`/`graph_solutions` rows are **not read by any renderer**,
so even a perfect reconciliation leaves the overlay/world/side-by-side video
byte-identical. The `cross_solver_geometry_decoupled` decision fires on **three
hard-coded source-family strings the adapter author wrote**, not on measured
geometry provenance.

Each risk below is stated as mechanism → observable symptom in rows/render →
discriminating kill-test that forces the next implementation. Route uncertainty
to `contact_state=unresolved` with provenance; do not omit the module and do not
"wire in" an unverified surface.

---

## Attack summary

| # | Risk | Severity | Kill-test |
|---|---|---|---|
| A1 | Interval penetration is against full-frame depth (table/hand leak), not the keyboard | **CRITICAL** | KT-1 keyboard-masked + hand-quarantined re-query |
| A2 | Planned route "wire the signal in" is the wrong route; D-a/D-b as specified don't discriminate | **CRITICAL** | KT-2 per-vertex-ID coherence (not `.max`) |
| A3 | Completed mesh too contaminated / non-watertight → contact/NP "0" is by construction | **HIGH** | KT-4 free-space eval + observed-face-only sign mesh |
| A4 | `cross_solver_geometry_decoupled` gameable by adapter-authored strings + `.max`-of-`.max` | **HIGH** | KT-6 provenance-hash-derived source families |
| A5 | ego.hoi rows change no renderable annotation (no consumer) | **CRITICAL** | KT-7 re-render through an ego.hoi-consuming path and pixel-diff |
| A6 | Pose-provenance dismissal (card D-c) is invalid; f37–44 are missing-pose | **MEDIUM** | KT-5 completed-mesh gap stratified by pose provenance |
| A7 | Within-solver contradiction (−107 mm vs +68 mm at f36) unreconciled | **MEDIUM** | KT-3 fingertip signed distance to all three surfaces under one pose |

---

## A1 — The interval "observed keyboard surface" is the full-frame depth (CRITICAL)

**Mechanism.** The interval solver's observed-surface barrier is the dense
full-frame monocular depth map, not a keyboard-segmented surface, and the filters
that would remove non-keyboard and hand-owned pixels are disabled.

**Evidence (`.../v18_joint_mano_interval_trajectory_state.json`, right interval + params):**
- input `depth_npz = measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz`, shape **(150, 1408, 1408)** — a full image depth map.
- `dense_observed_surface_barrier_enabled: True`, `dense_observed_penetration_weight: 300000`, `dense_observed_constraint_count_final` median 13.5 / max 722 per frame.
- `hand_owned_object_depth_quarantine_enabled: False`, `hand_owned_quarantined_face_count` = 0 on every frame → the hand's own monocular-depth pixels remain in the barrier.
- `visible_object_mask_gate_enabled: False`, `surface_eligibility_factor_enabled: False`, per-frame `surface_eligible_face_count: 0` → barrier not restricted to keyboard faces.
- `visible_surface_depth_order_selected_vertex_count` = **0 across all 150 frames** — the one term that actually checks hand-vs-object depth order found nothing.
- `scientific_test` string says *"observed **tomato** geometry"* inside a keyboard clip → the observed-surface path is a copied template, not specialized to this object.

**Symptom in rows/render.** `full_observed_surface_penetration_after_solver_max_m`
right = max 0.107 m but **median 0.0**, p90 0.047, p95 0.092 — the "10 cm" is the
single worst vertex in the single worst frame. The solver's own `translation_delta_norm_m`
right has median ~0 but **p90 4.5 cm, p95 7.2 cm, max 10.8 cm**: the barrier
dragged the hand up to 10.8 cm in the contact window (hidden by the median-0
headline the render publishes).

**Kill-test KT-1.** Recompute observed-surface penetration for right f28–48
restricted to (i) keyboard-mask pixels only and (ii) with hand-owned depth
pixels quarantined, and separately against a table-plane fit. *Prediction:* if
the penetration is table/hand leak, keyboard-masking + hand-quarantine collapses
count and depth toward 0; if it survives on keyboard-only masked depth it is
real. *Route:* only the keyboard-masked, hand-quarantined penetration may become
a contact/NP factor; the raw dense-barrier `.max` is demoted to non-evidential
debugging. This kills the plan's "wire the existing signal in" unless the signal
survives masking.

---

## A2 — Planned route and discriminator do not separate "real" from "leaked" (CRITICAL)

**Mechanism.** Card §7 routes to "wire in" when `D-a large + D-b coherent + D-d
extent inflated but penetration real`. D-a (geometry disagreement) is **large by
construction** — the two solvers query surfaces that are 10–25 cm apart — so it
discriminates nothing. D-b as specified (temporal IoU of penetrating vertex ids)
is the real discriminator, and the current data already predicts **incoherence**.

**Evidence (right, `full_observed_surface_penetration_after_solver_m` per frame):**
penetrating count = f31:105, f32:69, f33:43, f34:0, f35:0, f36:78, f37:89,
f38:130, f39:190, f40:0, f42:70, f43:52, f44:195, f45:156, f46:0. The count
**flickers 0→78→…→190→0** across consecutive frames and **grows during the
solve** (f38 initial 1 → after 130; f39 initial 22 → after 190). A hand resting
on a keyboard does not gain 130 penetrating vertices while a nonpenetration
barrier of weight 3e5 is active; a broad plane the palm spreads against does.

**Symptom in rows/render.** If the reconciliation reports "D-b coherent" from the
`.max` scalar it is wrong — `.max` cannot express set coherence. The adapter
already reduces this to `observed_surface_penetration_m = .max-of-.max = 0.107`,
discarding the per-vertex evidence entirely.

**Kill-test KT-2.** Emit per-frame penetrating **vertex IDs** (from
`optimized_vertices_sample_ids` filtered by signed distance), and report temporal
IoU across f28–48 and the anatomical region (fingertip vs palm vs dorsal).
*Prediction:* real contact → stable compact fingertip set, high IoU; leak → scattered,
low IoU, count non-monotone. *Route:* incoherent set → M2 (repair surface source
first, `contact_state=unresolved`), never "wire in." Forbid `.max` as the
reconciliation's penetration statistic.

---

## A3 — Completed mesh is contaminated and non-watertight; contact/NP "0" is not physics (HIGH)

**Mechanism.** The contact/NP signed query runs against a mesh that is 95.4 %
TRELLIS-inferred hidden surface and non-watertight, so Open3D signed raycasting
returns 0 usable candidates by construction; free space was never carved.

**Evidence (`.../v18_compact_rigid_trellis_completion_report.json`,
`.../v18_mano_object_constraint_state.json`):**
- completed-mesh faces: `observed_depth_surface 3221` / `unsupported_uncertain 963` / `trellis_inferred_hidden_surface 86708` = 90 892 → **95.4 % TRELLIS, 3.5 % observed**.
- `free_space_rejection_state: not_evaluated_in_this_revision`, `free_space_rejected: 0`.
- `completed_surface_mesh_watertight: False`, `sign_mesh_watertight: False`, `sign_mesh_path == surface_mesh_path`.
- right f36 row: `surface_aabb_candidate_vertex_count: 0`, `signed_query_candidate_vertex_count: 0`, `penetrating_vertex_count: 0`; f37–39: `surface_aabb_candidate` 2–3, `signed_query_candidate 0`; reason `uncertainty_only_nonwatertight_mesh_no_signed_correction`.
- contact/NP hand geometry = `delivered_annotation_metric_mano_vertices_world_sample_m`, **64 verts** (interval uses 160) → the two solvers do not even share the hand vertex set.

**Symptom in rows/render.** `frames_with_any_penetration: 0` both sides,
`candidate_correction_count: 0`. World panel draws `keyboard 56736 verts` =
the raw TRELLIS vertex count (`mesh_counts.trellis_vertices 56736`), i.e. the
render body is the contaminated completion, not the ~4 k-face observed body.

**Kill-test KT-4.** Before any completed-mesh distance is called contact evidence:
(a) evaluate free space against the observed depth points and report the rejected
fraction; (b) build a watertight sign mesh from **observed faces only** and re-run
the signed query; (c) report the TRELLIS-inferred fraction inside the queried
region. *Prediction:* the "0 penetration" is an artifact of the non-watertight
95.4 %-TRELLIS mesh, not a physical clearance. *Route:* until an observed-face
sign mesh exists, `contact_state=unresolved` with `geometry_epoch_contaminated`
provenance; do not publish "no contact / +39 mm gap" as an accepted annotation.

---

## A4 — `cross_solver_geometry_decoupled` fires on adapter-authored fields (HIGH)

**Mechanism.** The decision is a string comparison plus a sign check on values
the adapter hard-codes or copies, with no verification that any solver actually
queried the named geometry.

**Evidence (`scripts/build_clip001850_ego_hoi_graph_summary.py` →
`scripts/build_ego_hoi_sidecar_and_graph_health.py`):**
- The adapter sets `solver_geometry_source_family = "observed_depth_surface"`,
  `contact_query_geometry_source_family = "trellis_completed"`,
  `render_geometry_source_family = "trellis_completed"`, and epoch ids
  `"geo_observed_keyboard_depth_surface"` / `"geo_completed_keyboard_trellis_seed42"`
  as **string literals**. The scaffold's `derive_cross_solver_geometry_consistency`
  compares these strings and fires on inequality. Making the three literals equal
  would flip the decision with no change to the run.
- `observed_surface_penetration_m = _stat_max(full_observed_surface_penetration_after_solver_max_m)`
  = **0.107 (max-of-max)**; `published_contact_gap_m = 0.039` (contact-patch abs
  median, a different geometry). The mismatch reason *"observed-surface
  penetration and published positive gap coexist"* fires whenever both > 0 — a
  **tautology**: two channels computed against different surfaces will always
  both be nonzero.
- No field carries the actual queried-geometry provenance (depth-source path hash +
  mask/quarantine state for the interval barrier; `surface_mesh_path`/`sign_mesh_path`
  hashes for contact/NP; consumed-mesh hash for the renderer). The scaffold's
  `watertight`, `signed_query_candidate_vertex_count`, and face-provenance are
  copied but never used to *gate* the decision.

**Symptom in rows/render.** `graph_health.json` reports
`decision = cross_solver_geometry_decoupled` with `mismatch_reasons` that are true
of any run with two nonzero channels and three authored labels — it would "fire
correctly" on a fabricated summary.

**Kill-test KT-6.** Derive the three source-family/epoch fields from measured
provenance only: interval barrier = hash of `depth_npz` + `{mask_gate,
hand_quarantine, eligibility}` state; contact/NP = hashes of `surface_mesh_path`
and `sign_mesh_path` + `sign_mesh_watertight`; render = hash of the mesh the
publish path consumed. Route the decision to `evidence_incomplete` when any
provenance hash is absent, and forbid `.max`-of-`.max` as the penetration
statistic. *Prediction:* with real provenance the decision still fires (the paths
genuinely differ) but becomes falsifiable; equal-string tampering can no longer
flip it. Until then, treat `cross_solver_geometry_decoupled` as authored, not
measured.

---

## A5 — The reconciled rows change no renderable annotation (CRITICAL)

**Mechanism.** The overlay/world/side-by-side videos the user consumes are
produced by the v19 render path from v19 measurements; nothing reads ego.hoi rows.

**Evidence.** `grep` across `scripts/render_*.py`, `build_v19_rigid_render_state.py`,
`publish_v19_render_artifact.py`: **no consumer** of `contact_frame_detail`,
`geometry_epochs.parquet`, `graph_solutions.parquet`, or `org.ego.hoi`. Only
`build_ego_hoi_sidecar_and_graph_health.py` and `build_clip001850_ego_hoi_graph_summary.py`
mention them. `publish_v19_render_artifact.py` consumes `--interval-state` (the
interval JSON) + pre-rendered `--overlay/--world/--side-by-side` mp4s and stamps
the contact-patch gap into the banner (`v19_published_render_report.json`:
right gap 39.2 mm, `active_set_closed: false`, `translation_delta_norm_m_median
5.2e-20`).

**Symptom in rows/render.** After the CPU reconciliation writes parquet rows +
a graph_health decision, `renders/v19_published_runtime/*.mp4` and the stills
(`side_by_side_frame_000032.jpg`, f36) are unchanged. The user's annotation still
reads `R: gap 39.2mm … penverts=0 … UNCERTAIN`.

**Kill-test KT-7.** Re-render right f32/f36 **through a path that sources contact/
pose/geometry from the ego.hoi rows** and pixel/label-diff against the current
published stills. *Prediction:* current path → identical pixels → no artifact
change → not progress (per AGENTS.md: rows are backing data only). *Route:* either
rewire the render-state builder to consume ego.hoi rows (then the reconciled
`contact_state` and observed-sized body appear), or classify the rows as
non-evidential until a consumer exists. A graph_health decision + parquet with an
unchanged video is a schema artifact, not a renderable annotation change.

---

## A6 — Pose-provenance dismissal is invalid; f37–44 have no graph pose (MEDIUM)

**Mechanism.** Card D-c claims "penetration high in both observed f36 and held
f37 → pose freeze not the driver." But (a) the interval penetration is
**pose-independent** — it is measured against full-frame depth, not the posed
keyboard, so equality across pose provenance is expected and tells us nothing;
(b) f37 is not a "held f77 freeze" — the freeze is frames 78–149.

**Evidence (`.../v18_compact_rigid_object_pose_fit_report.json`,
`.../v19_rigid_object_pose_graph_report.json`):**
- `graph_frames: [30,31,32,33,34,35,36,45,46,60,75,76,77]`, `fit_frame_count 13`.
- f30–36, 45, 46 = `fit_to_visible_depth_samples`; **f37–44 = `missing_initial_graph_pose`**. The completed mesh at f37–44 is interpolated/extrapolated across the **9-frame graph gap 36→45**.

**Symptom in rows/render.** contact/NP AABB candidate count for right is 0 at f36
(a fit frame) but 2–3 at f37–39 (missing-pose frames) — the posed mesh moves
relative to the hand exactly where the pose is interpolated.

**Kill-test KT-5.** Stratify the **completed-mesh** signed distance/gap (which is
posed, unlike the dense barrier) by pose provenance: fit {30–36,45,46} vs
missing-pose {37–44}. *Prediction:* if candidate count/gap jumps across the 36→45
gap, the mesh pose at 37–44 is untrustworthy. *Route:* `contact_state=unresolved`
on interpolated-pose frames; the claim "pose is not the driver" is not established
by the pose-independent dense barrier.

---

## A7 — Within-solver contradiction is unreconciled (MEDIUM)

**Mechanism.** The same interval solver reports, for right f36: dense-barrier
penetration **+107 mm** and contact-patch normal gap **+68 mm** (positive =
away) on the fingertip vertices. A ~175 mm split on one hand at one frame means
the barrier surface and the contact-patch target are 14–25 cm apart in depth.

**Evidence.** f36 right: `full_observed_surface_penetration_after_solver_m.max
0.107`; `contact_patch_final_normal_gap_m.median +0.068` (count 12, weight
25000). Interval-level right `contact_patch_final_abs_normal_gap_m` median 0.039,
p90 0.13, p95 0.137, max 0.15 — the "contact" fingertips are 4–15 cm from target.

**Symptom in rows/render.** The reconciliation would carry both an
observed-surface source (deep penetration) and a completed-mesh contact-patch
source (large gap) for the same hand; averaging or picking either without
locating the true touched surface produces an arbitrary `contact_state`.

**Kill-test KT-3.** Under the f36 fit pose, compute the fingertip vertices'
signed distance to (a) full-frame depth, (b) keyboard-masked depth, (c)
completed mesh. *Prediction:* the fingertips lie within a soft-tissue band of at
most one surface. *Route:* source contact evidence from that surface only; if
none, `contact_state=unresolved`.

---

## Required kill-test gate for the next implementation

The reconciliation may claim progress only after **all** of the following pass on
right f28–48 and are reflected in the rendered stills:

1. **KT-1** observed-surface penetration recomputed with keyboard-mask + hand-owned
   quarantine; only the masked/quarantined signal is eligible as a factor.
2. **KT-2** per-vertex-ID temporal coherence reported (not `.max`); incoherent set
   routes to geometry repair, not "wire in."
3. **KT-4** completed-mesh free space evaluated and an observed-face-only watertight
   sign mesh built before any completed-mesh distance is called contact evidence.
4. **KT-6** `cross_solver_geometry_consistency` fields derived from measured
   provenance hashes; decision routes to `evidence_incomplete` when absent;
   `.max`-of-`.max` forbidden as the penetration statistic.
5. **KT-7** the reconciled `contact_state`/observed-sized body is re-rendered
   through an ego.hoi-consuming path and pixel/label-diffed against the current
   published stills — a nonzero diff is the acceptance evidence.

KT-3 and KT-5 are the discriminators that decide *which* surface/pose the contact
evidence is sourced from; they must run before KT-7.

## What "real progress" requires (route, not stop)

- Contact evidence for keyboard f28–48 comes from a **keyboard-masked observed
  surface under a fit pose**, or is `unresolved` with provenance. The full-frame
  dense barrier is non-evidential.
- The completed mesh is contact-eligible only after free-space carving + an
  observed-face watertight sign mesh; otherwise `contact_state=unresolved`,
  `geometry_epoch_contaminated`.
- The pose graph stops asserting `annotation_ready: True` while `nfev=1`,
  `cost=0`, `nonpenetration_target_frame_count=0` (confirmed inert); a live
  nonpenetration factor must come from the masked observed evidence above.
- A renderer consumes the ego.hoi rows so the reconciliation changes the video.

A reconciliation that faithfully records three disagreeing numbers, fires
`cross_solver_geometry_decoupled`, and leaves the video unchanged is **true but
useless**: it does not answer whether the hand touches the keyboard and does not
change the artifact the user consumes.

## Residual risks

- HOT3D GT hand+object pose exists for this clip (development set) and could
  adjudicate KT-1/KT-3 directly; it is deliberately deferred to R8 (GT-free
  first). If the CPU kill-tests are ambiguous, GT is the tiebreaker.
- KT-7 presumes an ego.hoi-consuming render path exists; today it does not, so
  KT-7 cannot pass without new render wiring. Until then, treat the reconciliation
  as backing-data only, not a rendered-annotation change.
- The interval solver's large in-window hand moves (right translation delta max
  10.8 cm) were driven by the suspect barrier; if KT-1 invalidates the barrier,
  those corrections must be reverted, not preserved.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Seven concrete attacks (A1-A7) each with mechanism, observable symptom in rows/render, severity, and a discriminating kill-test, all backed by exact fields from named on-disk artifacts: interval barrier is full-frame depth (1408x1408) with hand_owned_object_depth_quarantine_enabled=False, visible_object_mask_gate_enabled=False, surface_eligibility_factor_enabled=False, visible_surface_depth_order_selected_vertex_count=0; penetration .max 0.107 with median 0.0 and count flicker 0->190->0 growing during solve; contact/NP candidate_correction_count 0, sign_mesh_watertight False, signed_query_candidate 0, 95.4% TRELLIS non-watertight mesh, free_space not_evaluated; adapter hard-codes three geometry-source-family strings and copies .max-of-.max; no renderer consumes ego.hoi rows (grep across scripts/render_*.py, publish, build_v19_rigid_render_state); pose graph nfev=1/cost0/NP targets 0 and f37-44 missing_initial_graph_pose across a 9-frame graph gap. File paths and severities provided for each."
    }
  ],
  "changedFiles": [
    ".memory/tasks/2026-07-06-research-track-execution/subagents/06_reconciliation_attack.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "read PROMPT/EPISTEMIC/TASK_PACK/02_first_slice_causal_card.md + 01/03 + OPS", "result": "passed", "summary": "loaded task spec, epistemic model, and the reconciliation plan under attack"},
    {"command": "read scripts/build_ego_hoi_sidecar_and_graph_health.py + scripts/build_clip001850_ego_hoi_graph_summary.py", "result": "passed", "summary": "confirmed cross_solver_geometry_decoupled fires on adapter-authored string literals + .max-of-.max copy + tautological coexistence reason"},
    {"command": "python3 inspect v18_joint_mano_interval_trajectory_state.json (params + right interval + per-frame f28-48)", "result": "passed", "summary": "dense full-frame-depth barrier, quarantine/mask/eligibility OFF, depth-order selected 0 verts, penetration .max 0.107 median 0.0, count flickers/grows, translation delta max 10.8cm, f36 within-solver -107mm vs +68mm"},
    {"command": "python3 inspect v18_mano_object_constraint_state.json (summary + constraint_rows f34-40 right)", "result": "passed", "summary": "candidate_correction_count 0, frames_with_any_penetration 0, sign non-watertight, signed_query_candidate 0, hand geom = delivered 64 verts"},
    {"command": "python3 inspect v18_compact_rigid_trellis_completion_report.json", "result": "passed", "summary": "completed mesh 95.4% TRELLIS / 3.5% observed, free_space not_evaluated, trellis_vertices 56736 (= world panel label)"},
    {"command": "python3 inspect v19_rigid_object_pose_graph_report.json + visible_pose_fit report", "result": "passed", "summary": "pose graph inert nfev=1 cost0 NP targets 0; graph_frames [30-36,45,46,60,75,76,77]; f37-44 missing_initial_graph_pose"},
    {"command": "python3 inspect unidepth_full_frame_depth_v3.npz shape + v19_published_render_report.json", "result": "passed", "summary": "depth (150,1408,1408) full frame; render publishes right gap 39.2mm from interval contact-patch, translation median 5.2e-20"},
    {"command": "grep ego.hoi/contact_frame_detail/graph_health consumers across renderers", "result": "passed", "summary": "no v19 renderer/state-builder reads ego.hoi rows; publish consumes interval-state JSON + pre-rendered mp4s"}
  ],
  "validationOutput": [
    "A1 CRITICAL: interval observed-surface penetration is against full-frame UniDepth (table/hand leak), quarantine+mask+eligibility off, depth-order term 0 verts, .max-of-.max headline.",
    "A2 CRITICAL: planned 'wire in' route + D-a/D-b as specified do not separate real from leaked; count flicker 0->190->0 and growth during solve predict incoherence -> geometry repair, not wire-in.",
    "A3 HIGH: contact/NP '0 penetration' is 0-by-construction on a 95.4% TRELLIS non-watertight mesh with free space not evaluated.",
    "A4 HIGH: cross_solver_geometry_decoupled fires on three adapter-authored source-family strings + a tautological >0/>0 coexistence check; gameable.",
    "A5 CRITICAL: no renderer consumes ego.hoi rows; reconciliation leaves the published video byte-identical.",
    "A6 MEDIUM: card D-c pose dismissal invalid (penetration is pose-independent; f37-44 are missing-pose across a 9-frame graph gap).",
    "A7 MEDIUM: same solver reports -107mm and +68mm on right f36; touched surface unlocated."
  ],
  "residualRisks": [
    "KT-7 cannot pass until an ego.hoi-consuming render path exists; until then the reconciliation is backing-data only, not a rendered-annotation change.",
    "HOT3D GT (hand+object pose) exists for this clip and can adjudicate KT-1/KT-3 if CPU kill-tests are ambiguous; deferred to R8 by the GT-free-first constraint.",
    "If KT-1 invalidates the dense barrier, the interval solver's large in-window hand corrections (right translation delta up to 10.8cm) must be reverted, not preserved."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one task-memory findings document (this attack plan) at the authoritative subagents path. No source files touched; git working tree otherwise unchanged (pre-existing dirty files not mine).",
  "reviewFindings": [
    "blocker: interval 'observed keyboard surface' penetration is against full-frame depth with hand-quarantine/mask/eligibility OFF (v18_joint_mano_interval_trajectory_state.json) - the plan's primary route 'wire the signal in' would optimize the hand into the table/its-own-depth. Require KT-1 before any wiring.",
    "blocker: no renderer consumes ego.hoi rows (scripts/publish_v19_render_artifact.py + render_v19_*.py) - reconciliation changes no renderable annotation. Require KT-7.",
    "blocker: cross_solver_geometry_decoupled decision fires on adapter-authored string literals + .max-of-.max in scripts/build_clip001850_ego_hoi_graph_summary.py:~380-420 and scripts/build_ego_hoi_sidecar_and_graph_health.py derive_cross_solver_geometry_consistency - not measured provenance. Require KT-6.",
    "major: completed mesh 95.4% TRELLIS non-watertight, free_space not_evaluated (v18_compact_rigid_trellis_completion_report.json) - contact/NP '0' is by construction. Require KT-4.",
    "major: card D-c dismissal of pose mechanism is invalid; f37-44 are missing_initial_graph_pose across a 9-frame graph gap (v18_compact_rigid_object_pose_fit_report.json). Require KT-5.",
    "note: within-solver -107mm vs +68mm on right f36 must be reconciled before crossing solvers (KT-3)."
  ],
  "manualNotes": "Strongest single finding for the parent: the reconciliation reconciles three numbers none of which is a trustworthy keyboard-contact measurement (interval=full-frame-depth penetration, contact/NP=0-candidate query on contaminated non-watertight mesh, render=contact-patch gap on same mesh), and its output rows are read by no renderer. The plan's primary route ('wire the observed-surface penetration in') is the wrong route given the barrier is unmasked full-frame depth with hand-quarantine off. Gate the next implementation on KT-1 (mask+quarantine re-query), KT-2 (per-vertex coherence, not .max), KT-4 (free-space + observed-face sign mesh), KT-6 (provenance-hashed cross-solver decision), KT-7 (re-render through an ego.hoi consumer and pixel-diff). Route unresolved contact as a state variable with provenance; do not omit and do not wire an unverified surface."
}
```
