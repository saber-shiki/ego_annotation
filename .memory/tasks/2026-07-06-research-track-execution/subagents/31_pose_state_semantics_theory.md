# 31 — Object pose state semantics doctrine (corrected after user objection)

Theory/doctrine task. Branch `yiwen_research`. **No edits**, nothing staged.
Concrete example throughout: HOT3D **clip001850** keyboard, run
`/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.

This supersedes the earlier framing (subagent 28) in which `T_rest` was written
into the pose rows for 142 non-observed frames as a "held static gauge" and
rendered as a de-rated body. The parent's contract correction is: **not every
hypothesis belongs in state.** `T_rest` may exist as a rest-cluster *reference*
(outlier rejection + cluster description) but must never become a stored pose
value, a downstream factor prior, an imputed trajectory, contact evidence, or a
render-localized body on missing frames.

All load-bearing dataflow claims below were verified on disk (see §7), not taken
from secondary reports.

---

## 0. The correction in one line

The unit that separates truth from laundering is **provenance**, not styling. A
frame with no admissible observation must carry **no pose value at all**
(`unknown`), because a value plus a "do not trust me" flag is still read as a
value by every consumer that keys on the pose column — verified: the delivered
renderers place a metric body from `translation_world_m` **and read no
visibility/`is_measured` field whatsoever** (§7-A). Absence of a value is the
only representation of `unknown` that cannot be laundered.

---

## 1. Core principle — state is a commitment, priors are not measurements

**State is the set of claims the system commits to and that downstream stages are
entitled to rely on.** Placing a value in state grants it the epistemic status of
a commitment. A hypothesis ("the keyboard is probably still at rest on f90") is a
guess the system does *not* commit to. Writing it into the pose column promotes a
guess to a commitment; every downstream stage then relies on it *correctly per
the contract* — that is the laundering, and it is a contract bug, not a rendering
bug.

The mechanism is the standard double-counting failure of a Bayesian estimator.
For a pose variable `x_t`, the honest posterior is

```
p(x_t | evidence) ∝ prior(x_t) · Π_k likelihood_k(measurement_k | x_t)
```

The prior and the measurements are **separate terms with separate covariances**.
Injecting a prior mean (`T_rest`) into the *measurement* channel — a factor that
says "a measurement observed `x_t = T_rest` with measurement covariance" —
(a) fabricates information that does not exist (there was no measurement on f90),
and (b) double-counts the prior if a real prior factor is also present. The
posterior collapses toward `T_rest` with **falsely tight covariance**. Contact,
nonpenetration, and every HOI claim then inherit false confidence in the object's
location. This is precisely "laundering a prior as a measurement": a guess
acquires the status of an observation by being placed in the observation channel.

Corollary — **not every hypothesis belongs in state.** Three kinds of thing are
computed and used transiently but must never be committed to the pose state:
reference statistics (`T_rest`), gating decisions (outlier rejection), and local
patches (nearest-hold, interpolation). State admits only: observation facts,
explicit unknowns, and model-produced estimates with declared priors and
posterior covariance.

---

## 2. The pose-state ontology (five classes) + T_rest's demoted role

Every per-frame object-pose variable is in exactly one class. The class — not a
styling flag — determines what may be stored and what any consumer may do.

| Class | Meaning | Numeric pose stored? | Provenance tag | clip001850 frames | count |
|---|---|---|---|---|---|
| **observed_measured** | own visible surface produced a metric fit that survived outlier/consistency gating | **yes** — the fit | `observed_icp` + mask/depth/geom hashes | 30,31,32,33,34,35,36,46 | 8 |
| **observed_rejected** | a fit was attempted and **rejected** as inconsistent (mask drift / >Nσ from cluster) | value **retained for audit only**, flagged rejected | `observed_icp_rejected` + reason | 60, 75, 76, 77 | 4 |
| **unresolved** | a fit exists but is ambiguous; neither accepted nor cleanly rejected | value retained, flagged unresolved | `observed_icp_unresolved` + reason | 45 (hand-occluded, 44 mm/17°) | 1 |
| **unknown** | **no admissible observation** exists | **no** — pose is **null/absent** | `no_admissible_observation` + subreason | 0–29,37–44,47–59,61–74 (occluded_out_of_view, 65) + 78–149 (mask_delaminated, 72) | 137 |
| **solver_estimate** | a live solver inferred the variable from explicit evidence + model priors | **yes** — the posterior | `solver_estimate` + solver id + priors used + posterior covariance | none yet (P15 inert, §7-C) | 0 |

Sum: 8 + 4 + 1 + 137 + 0 = 150. ✔

**T_rest is not a state class.** It is a *reference statistic* of the
`observed_measured` cluster (pose-translation median `[0.581,-0.064,0.345] m`
over f30–36,46; per-axis spread 29/40/35 mm; max pairwise rotation 9.2°). Its
**only** two legitimate roles:

1. **Outlier-rejection gate** — deciding f60/75/76/77 are too far from the cluster
   to be `observed_measured` (they become `observed_rejected`).
2. **Cluster description** — the factual statement "the observed rest cluster is
   at `T_rest` ± spread," attached to the observed frames and to a gauge report.

T_rest is a statistic **about** the observed frames, never a value **for** the
unobserved frames. (Note: the observed *centroid* rest `[0.599,-0.094,0.339]`
differs from the pose-translation median by the canonical-origin offset; do not
conflate them or check one against the other.)

**Forbidden uses of T_rest** (each is a laundering path): a pose value for any
`unknown` frame; a downstream factor prior; an imputed trajectory; contact /
nonpenetration geometry evidence; a render-localized body on missing frames.

---

## 3. Storage rules (substrate / schema level)

1. **`unknown` is a first-class row with a null pose.** The schema must permit,
   and a consumer must expect, a pose row whose rotation/translation are absent.
   `unknown` is represented by the *absence* of a value, plus a provenance reason
   — never by a placeholder value with a flag. (This is the direct fix to §7-B.)

2. **Observation rows and solver-estimate rows are different record types.** An
   `object_pose` measurement row (an observation fact, immutable given the
   measurement) must not share a column with, or be silently overwritten by, a
   `graph_solutions` estimate. A single `translation_world_m` column that holds
   "measured fit here, held T_rest there, interpolation elsewhere" (the current
   `object_pose_static_gauge.ndjson`, §7-B) is the structural defect.

3. **`observed_rejected` retains its value but in a rejected slot**, with the
   rejection reason and the gate that fired. It is kept for provenance/audit; it
   is not a pose the object had.

4. **Provenance travels with every value.** A downstream stage must be able to
   read, for any pose it consumes, whether it is a measured fact, a rejected
   measurement, a model prior, or a prior solver estimate. Provenance is
   admissibility metadata, not decoration.

5. **Reference statistics live outside the per-frame pose table.** `T_rest` and
   its covariance belong in a gauge/cluster-description record scoped to the
   observed frames, explicitly not indexed as a per-frame pose.

6. **The anchor/canonical seed must itself be an `observed_measured` frame.** In
   clip001850 the visible-geometry `anchor_frame_idx=75` is an `observed_rejected`
   mask-drift outlier 199.5 mm off rest (§7-D) — a rejected observation laundered
   into the geometry substrate and the eligibility gate. A rejected observation
   must never seed the substrate, the canonical mesh, or the gate.

---

## 4. Renderer rules — when a ghost/prior may be shown

The renderer is a **consumer of state**; it must not manufacture localization the
state does not contain.

- **Place a metric object body at frame `t` only if** `state(t)` is
  `observed_measured` (solid/confident styling) **or** `solver_estimate` (ghost
  styling: faded/wireframe, opacity/spread scaled to posterior covariance,
  labeled "estimate").
- **On `unknown`, `observed_rejected`, `unresolved`: draw no metric body.** Show
  a non-metric state indicator ("keyboard: pose unknown / occluded / rejected /
  unresolved") and the rest of the scene (the hand has its own state — MANO is
  present on all 150 frames, so an honest `unknown` frame is a hand with **no**
  keyboard body).
- **A ghost is allowed only as the visualization of a `solver_estimate`** — a
  model posterior with covariance. It may **never** be sourced from a reference
  statistic (`T_rest`) or an inherited local patch (nearest-hold, interpolation).
  On deeply unobserved frames (f78–149) a legitimate `solver_estimate` would have
  huge posterior covariance, so its ghost would be very diffuse/faded — honestly
  saying "the model guesses roughly here and is very unsure." That is legitimate
  because it is the model's posterior; a crisp body at `T_rest` is not.
- **`T_rest` may appear only in a QC/debug overlay** explicitly labeled "rest-
  cluster reference (describes f30–36,46 only)," never as a per-frame metric body
  on other frames and never in the delivered annotation.

This overrules subagent 28's render: drawing the body at `T_rest` on the 142
non-observed frames — even dashed/dimmed with `is_measured=false` — is a
localization claim, because the renderer reads `translation_world_m` and ignores
the flag (§7-A). The honest render draws **no** keyboard body on those frames.

---

## 5. Downstream factor-graph rules — anti-laundering

The later factor graph must **operate on unknown variables**, with explicit
evidence and **priors chosen by the model, not inherited local patches.**

1. **Measurement factors only on `observed_measured`.** Only f30–36,46 contribute
   a pose-measurement factor, each with its fit covariance. Nothing else enters
   the measurement channel.

2. **`unknown` frames are free latent variables with no measurement factor.** The
   graph instantiates a pose variable and estimates it from priors + whatever
   *indirect* evidence couples to it (temporal smoothness to observed neighbors,
   rigidity, motion coupling, correspondence tracks that bridge the gap).

3. **Priors are explicit model factors with calibrated noise.** A declared
   temporal process model (e.g. constant-position/constant-velocity with process
   noise) or a learned object-motion prior — applied uniformly by the solver, not
   hand-stamped per frame. If the object truly is static, that must **emerge as
   the posterior** under a motion prior + weak evidence; it must not be injected
   as `T_rest`.

4. **No inherited local patch enters the graph.** An *inherited local patch* is a
   value copied/frozen into a variable from a neighbor or cluster, bypassing the
   prior machinery, with no calibrated noise and no posterior. In clip001850 these
   are `nearest_visible_pose_hold` (f0–29←f30, f78–149←f77), Slerp/lerp
   interpolation (f37–44 etc.), and the `T_rest` stamp. All are forbidden as
   graph inputs; the frames they cover are `unknown`.

5. **`observed_rejected` contributes no clean measurement.** At most a robust
   factor with an explicit outlier process may use it; it is never a clean
   measurement and never a prior.

6. **A `solver_estimate` is never re-consumed as a measurement** in a later solve
   (that is laundering across iterations — the same double-counting as §1). It may
   seed/initialize; it may not constrain.

7. **The graph-health table must audit provenance purity** (extends §7-E R0/R1
   scaffold): count measurement factors whose input provenance is not
   `observed_measured`; count pose variables that received a value from a held/
   interpolated/`T_rest` source. Both counts must be zero for the pose channel;
   any nonzero value flags laundering and routes an intervention.

**The clip001850 consequence is honest and stable:** with correspondences
unrecoverable on f78–149 (KT-L2: 0/72) and only a weak temporal prior anchored
40+ frames away, the posterior on the unobserved frames has huge covariance —
correctly "we do not know where the keyboard is here." Contact therefore stays
`unresolved` there: **object pose unknown ⇒ no contact geometry ⇒ contact
unknown.** That is why contact is unresolved on the 129 frames outside the
observed window — a feature of the doctrine, not a gap to be patched.

---

## 6. Mapping to the R0–R7 roadmap (`TASK_PACK.md`)

- **R0 — output substrate (`ego.hoi` sidecar).** Encode §3 in the schema: make
  `unknown` a first-class row with null pose; separate observation rows from
  `graph_solutions` estimate rows; forbid a reference statistic or held value in a
  measured-pose column; move `T_rest`/covariance to a gauge record scoped to the
  observed frames. This is the substrate-level fix that makes the subagent-28
  representation (`object_pose_static_gauge.ndjson`, held T_rest in
  `translation_world_m`) schema-illegal.

- **R1 — causal cards + graph instrumentation.** Add the §5.7 provenance-purity
  signals to `build_ego_hoi_sidecar_and_graph_health.py` (§7-E): measurement-
  factor provenance count, held/imputed-pose count, and a render-lineage check
  that flags any body drawn from a non-`observed_measured`/non-`solver_estimate`
  pose. Declare `T_rest` in the gauge block as *outlier-rejection statistic*, not
  a per-frame prior.

- **R2 — correspondence & rigidity.** The *principled alternative* to holding
  `T_rest`. Temporal tracks / rigidity windows are the real evidence that could
  legitimately constrain the `unknown` frames; robust Procrustes seeds do the
  outlier rejection (the same f60/75/76/77) that `T_rest` was misused for. If
  tracks bridge the occluded gaps, they upgrade `unknown` → `solver_estimate`
  honestly; on clip001850 KT-L2 shows they do not survive on f78–149.

- **R3 — geometry epochs.** Same doctrine for surfaces: never-observed geometry is
  `prior_completed` provenance, not `observed`; contact routes to observed faces;
  the canonical mesh must not be seeded from the f75 outlier (§3.6). The 3221-face
  observed-only body is the `observed_measured` analogue for geometry.

- **R4 — pose optimizer.** The one stage that *legitimately produces
  `solver_estimate` values* for `unknown` frames. It must add a **pose-level**
  trajectory prior (model-chosen, calibrated) plus measurement terms, so unknown
  frames are actually estimated with posterior covariance, and outliers are
  rejected by robust factors — not pre-filtered by `T_rest`. This directly
  replaces the current inert P15 residual (§7-C), which has no pose-measurement
  term and no live prior, so it estimates nothing.

- **R5 — HOI hand correction.** Same rule: hand drift latents are estimates, not
  measurements; occlusion intervals are `unknown`, not smoothly imputed.

- **R6 — contact / occlusion channels.** Enforce "object pose unknown ⇒ contact
  unknown." Contact must never be computed against a held/imputed object pose;
  `T_rest` must never be contact geometry. The visibility states
  (occluded_out_of_view / mask_delaminated / unresolved) are the honest carrier of
  the `unknown` pose downstream.

- **R7 — live factor graph.** The synthesis where the full anti-laundering
  discipline of §5 must hold: measurement factors only on observed facts; free
  latents on unknown frames; explicit model priors; posterior estimates tagged
  `solver_estimate` and rendered as ghosts; graph-health (R1) proving no prior is
  laundered as a measurement and that the graph is live (not a P15-style
  pass-through).

---

## 7. Concrete findings (file paths + severity)

- **A — doctrine-blocker (verified).** `scripts/render_v19_contact_state_full_duration.py:187-191`
  `observed_body_world()` places the body from **only** `pose_row["rotation_world_from_completed_canonical_matrix"]`
  and `pose_row["translation_world_m"]`. Grep for `visibility_state|is_measured|
  is_localized|T_rest|static_gauge|mask_unreliable|occluded_out_of_view` across
  both delivered consumers (`render_v19_contact_state_full_duration.py`,
  `render_clip001850_v19_contact_state_full_duration.py`) returns **empty**. The
  renderer structurally ignores every "do not localize me" flag; any value in the
  pose column becomes a placed body. ⇒ flag-based discipline cannot prevent
  laundering; only a null pose can.

- **B — doctrine-blocker (verified).** `scripts/render_clip001850_static_gauge_pose_artifact.py:300,312`
  writes `pose_source="static_gauge_held_rest"` with `translation_world_m = T_rest`
  for all 142 non-observed frames — the *same column* the renderer localizes from
  (consumed :864,:877). `is_measured=false` is set (:264,:319) but never read.
  Line 109 labels `occluded_out_of_view` frames `body=held static gauge T_rest`,
  i.e. a metric body drawn where the object was not observed. This is the exact
  laundering the correction rejects; the artifact's held-T_rest pose rows are
  superseded. Corrected representation: those 142 frames are `unknown` with null
  pose and no rendered body.

- **C — high (verified).** `scripts/solve_v19_rigid_object_pose_graph.py:295-333`
  residual operates only on correction deltas (`trans_delta`/`rot_delta` anchor,
  step, accel) plus a nonpenetration target that is absent for this clip; there is
  **no pose-measurement or pose-trajectory term**. Result: `nfev=1, cost=0`, zero
  `solver_estimate` frames. R4 must add the pose-level prior + measurement terms
  so `unknown` frames become genuine estimates. Until then there is no legitimate
  value for any unobserved frame.

- **D — high (verified via report 26 §2 + adapter report).** Visible-geometry
  `anchor_frame_idx=75` is an `observed_rejected` mask-drift outlier 199.5 mm off
  rest, yet it seeds the ICP canonical mesh and the `rigid_pose_observation_eligible`
  gate. A rejected observation is laundered into the substrate (§3.6). Any pose
  refit must re-anchor to the `observed_measured` cluster first.

- **E — medium (R0/R1 gap).** `scripts/build_ego_hoi_sidecar_and_graph_health.py`
  (subagent 03) has support/liveness/stale-join routing but no provenance-purity
  channel. It must gain the §5.7 counts (measurement-factor provenance; held/
  imputed-pose count; render-lineage flag) so laundering is detectable, and the
  `object_pose` table must gain the null-pose `unknown` row type (§3.1).

- **F — high (build_pose_rows, verified).** `scripts/solve_v19_rigid_object_pose_graph.py:377-463`
  fills 137 non-direct frames by `nearest_visible_pose_hold` (:61-67, copies a
  neighbor's pose) and Slerp/lerp interpolation (:52-58) — inherited local
  patches. 102 frames collapse to two poses (f30-hold, f77-hold). These frames are
  `unknown`; the hold/interpolation must not enter state or the graph (§5.4).

---

## 8. Residual risks / boundaries

- **This is doctrine, not an implemented artifact.** No render or table was
  changed. Applying it requires the R0 schema change (null-pose `unknown`), the
  renderer branch (no body on `unknown`), and the R4 live optimizer. The subagent-28
  held-T_rest artifact remains on disk and must be treated as superseded, not as
  the pose source of record.
- **"Static" for clip001850 remains a GT-free inference.** Even with the correct
  ontology, the claim that the keyboard is static over the observed support rests
  on the rest-cluster consensus + motion-coupling negative + KT-L1 no-moving-run —
  all GT-free. Only HOT3D GT/R8 (bundle not on local disk) can adjudicate the 137
  unobserved frames. The doctrine's `unknown` label is exactly what keeps those
  frames honest until then.
- **The doctrine does not raise localization coverage.** It correctly leaves
  8/150 observed and 137/150 unknown; raising real coverage requires new evidence
  (R2 correspondences that survive, or a new perception pass), not a stored prior.
- **`solver_estimate` is the one class that can still be abused.** A live R4/R7
  solver could produce over-confident posteriors if its priors are miscalibrated
  or if it silently re-consumes its own estimates (§5.6). The R1 provenance-purity
  audit and honest posterior covariance are the guards; they must be checked
  against the rendered ghost, not just the table.
- **Verification scope.** Findings A/B/C/F were verified by reading the exact code
  regions on disk this session; D relies on report 26 + the adapter report; E is a
  design gap in the R0/R1 scaffold. No dynamic run was executed (theory task).

---

## Reproduce (read-only, evidence for §7)

```bash
cd /home/yiwen/ego_annotation
sed -n '187,191p' scripts/render_v19_contact_state_full_duration.py            # A: body = f(R,t) only
grep -n "visibility_state\|is_measured\|T_rest\|static_gauge" \
  scripts/render_v19_contact_state_full_duration.py \
  scripts/render_clip001850_v19_contact_state_full_duration.py                  # A: empty
grep -n "static_gauge_held_rest\|translation_world_m\|is_measured" \
  scripts/render_clip001850_static_gauge_pose_artifact.py                       # B: held T_rest in pose col
sed -n '295,333p' scripts/solve_v19_rigid_object_pose_graph.py                  # C: no pose-measurement term
sed -n '377,463p' scripts/solve_v19_rigid_object_pose_graph.py                  # F: nearest-hold / interp
```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Delivered the corrected pose-state-semantics doctrine at the authoritative path .memory/tasks/2026-07-06-research-track-execution/subagents/31_pose_state_semantics_theory.md, grounded in on-disk verification. Core correction encoded: not every hypothesis belongs in state; T_rest is a rest-cluster REFERENCE (outlier rejection + cluster description only), never a stored pose value, factor prior, imputed trajectory, contact evidence, or render-localized body on missing frames. Defined a five-class ontology (observed_measured f30-36,46=8; observed_rejected f60,75,76,77=4; unresolved f45=1; unknown 137 with NULL pose; solver_estimate 0 currently) summing to 150 for clip001850, plus storage rules (unknown=absent value, observation vs solver-estimate separated, provenance travels with every value, anchor must be observed_measured), renderer rules (metric body only for observed_measured or solver_estimate; ghost only as a solver posterior with covariance, never from T_rest/held/interp), anti-laundering graph rules (measurement factors only on observed facts; unknown=free latent; priors are explicit model factors not inherited local patches; solver_estimate never re-consumed as measurement; provenance-purity audit), and a full R0-R7 mapping. Concrete findings with file paths and severity: A doctrine-blocker render_v19_contact_state_full_duration.py:187-191 places body from pose R/t only and reads no visibility flag (grep empty, verified); B doctrine-blocker render_clip001850_static_gauge_pose_artifact.py:300,312 writes T_rest into translation_world_m for 142 held frames (verified); C high solve_v19_rigid_object_pose_graph.py:295-333 has no pose-measurement term nfev=1 cost=0 (verified); D high anchor_frame_idx=75 is a rejected outlier seeding the substrate; E medium build_ego_hoi_sidecar_and_graph_health.py lacks provenance-purity + null-pose unknown row; F high build_pose_rows:377-463 nearest-hold/interp inherited local patches (verified)."
    }
  ],
  "changedFiles": [
    ".memory/tasks/2026-07-06-research-track-execution/subagents/31_pose_state_semantics_theory.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "sed -n '187,191p' scripts/render_v19_contact_state_full_duration.py + grep visibility/is_measured/T_rest across both delivered render consumers",
      "result": "passed",
      "summary": "observed_body_world places body from pose_row rotation+translation_world_m ONLY; grep for any visibility/is_measured/T_rest/static_gauge field returns empty in both consumers -> renderer structurally ignores 'do not localize' flags (finding A)."
    },
    {
      "command": "grep -n static_gauge_held_rest/translation_world_m/is_measured scripts/render_clip001850_static_gauge_pose_artifact.py",
      "result": "passed",
      "summary": "Held frames get pose_source=static_gauge_held_rest with translation_world_m=T_rest (:300,:312) written into the same column the renderer localizes from (:864,:877); is_measured=false set (:264,:319) but never read; line 109 labels occluded_out_of_view body=held static gauge T_rest (finding B)."
    },
    {
      "command": "sed -n '295,333p' + '377,463p' scripts/solve_v19_rigid_object_pose_graph.py",
      "result": "passed",
      "summary": "residual has only correction-delta anchor/step/accel + absent nonpenetration target, no pose-measurement/trajectory term (nfev=1 cost=0, finding C); build_pose_rows uses nearest_visible_pose_hold (:61-67) + Slerp/lerp (:52-58) inherited local patches, 102 frames collapse to 2 poses (finding F)."
    }
  ],
  "validationOutput": [
    "Correction encoded: T_rest is a reference statistic (median [0.581,-0.064,0.345] m, spread 29/40/35 mm, rot 9.2deg over f30-36,46), legitimate only for outlier rejection + cluster description; forbidden as stored pose, factor prior, imputed trajectory, contact evidence, or render-localized body on missing frames.",
    "Five-class ontology for clip001850 sums to 150: observed_measured 8 (f30-36,46), observed_rejected 4 (f60,75,76,77), unresolved 1 (f45), unknown 137 (65 occluded_out_of_view + 72 mask_delaminated, NULL pose), solver_estimate 0 (P15 inert).",
    "Storage rule: unknown = absence of a value (not a flagged placeholder), because the delivered renderers read translation_world_m and no visibility/is_measured field (verified) -- structural prevention beats flag-based discipline.",
    "Anti-laundering: measurement factors only on observed_measured; unknown frames are free latent variables; priors are explicit model factors with calibrated noise (not inherited nearest-hold/interp/T_rest patches); solver_estimate never re-consumed as a measurement; R1 must add a provenance-purity audit.",
    "Renderer: metric body only for observed_measured (solid) or solver_estimate (ghost scaled to posterior covariance); no body on unknown/rejected/unresolved; T_rest never a per-frame body, only a labeled QC overlay.",
    "R0-R7 mapping delivered; consequence for clip001850: object pose unknown on 137 frames => contact unknown there (feature, not gap); static-ness must emerge as an R4 posterior, not be injected as T_rest."
  ],
  "residualRisks": [
    "Doctrine only; no schema/render/optimizer change was made. The subagent-28 held-T_rest artifact (object_pose_static_gauge.ndjson + its render) remains on disk and must be treated as superseded, not as the pose source of record.",
    "Static verdict for clip001850 stays GT-free; only HOT3D GT/R8 (bundle not on local disk) can adjudicate the 137 unknown frames. The unknown label is what keeps them honest until then.",
    "The doctrine does not raise localization coverage (8/150 observed, 137/150 unknown); raising it requires new surviving evidence (R2 correspondences or a new perception pass), not a stored prior.",
    "solver_estimate is the one class still abusable: an R4/R7 solver with miscalibrated priors or self-re-consumed estimates can produce over-confident posteriors; the R1 provenance-purity audit + honest posterior covariance checked against the rendered ghost are the guards.",
    "Findings A/B/C/F verified by reading exact code regions this session; D relies on report 26 + adapter report; E is a design gap. No dynamic run executed (theory task)."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one task-memory doctrine document (31_pose_state_semantics_theory.md) at the authoritative subagents path. No source files touched; read-only verification of render/solver/static-gauge code; nothing staged or committed.",
  "reviewFindings": [
    "doctrine-blocker (A, high): scripts/render_v19_contact_state_full_duration.py:187-191 places the object body from pose_row rotation+translation_world_m ONLY, and neither delivered consumer reads any visibility/is_measured/gauge field (grep empty, verified). Any value in the pose column becomes a localized body -> a flagged placeholder cannot represent unknown; only a null pose can.",
    "doctrine-blocker (B, high): scripts/render_clip001850_static_gauge_pose_artifact.py:300,312 writes T_rest into translation_world_m for all 142 non-observed frames (pose_source=static_gauge_held_rest), the same column the renderer localizes from; is_measured=false is set but never read; line 109 draws a metric body at T_rest on occluded_out_of_view frames. This held-T_rest representation is superseded: those frames must be unknown with null pose and no body.",
    "high (C): scripts/solve_v19_rigid_object_pose_graph.py:295-333 has no pose-measurement/trajectory term (only correction-delta anchor/step/accel + absent nonpenetration target), so nfev=1 cost=0 and zero solver_estimate frames exist. R4 must add a model pose prior + measurement terms so unknown frames become genuine estimates with posterior covariance.",
    "high (D): visible-geometry anchor_frame_idx=75 is an observed_rejected mask-drift outlier 199.5 mm off rest (report 26 sec 2) yet seeds the ICP canonical mesh and eligibility gate -- a rejected observation laundered into the substrate; any refit must re-anchor to the observed_measured cluster.",
    "high (F): scripts/solve_v19_rigid_object_pose_graph.py:377-463 fills 137 non-direct frames by nearest_visible_pose_hold and Slerp/lerp (inherited local patches), 102 collapse to 2 poses; these frames are unknown and the hold/interp must not enter state or the graph.",
    "medium (E): scripts/build_ego_hoi_sidecar_and_graph_health.py lacks a provenance-purity channel and the object_pose schema lacks a null-pose unknown row type; R0/R1 must add both so prior-laundering is schema-illegal and detectable."
  ],
  "manualNotes": "Core doctrine for the parent: the discriminator between truth and laundering is provenance, not styling. Represent unknown frames by ABSENCE of a pose value (verified necessary: the delivered renderers read translation_world_m and ignore every visibility flag). T_rest is demoted to a rest-cluster reference used only to reject outliers (f60/75/76/77) and describe the observed cluster (f30-36,46); it is never a stored pose, factor prior, imputed trajectory, contact evidence, or rendered body on missing frames. Five state classes: observed_measured (8), observed_rejected (4), unresolved (1, f45), unknown (137, null pose), solver_estimate (0 now; P15 inert). A ghost may be rendered ONLY as an R4/R7 solver posterior with covariance, never from T_rest/held/interp. The factor graph infills unknown variables with explicit evidence + model-chosen priors (declared noise), never inherited nearest-hold/interp/T_rest patches, and never re-consumes a solver_estimate as a measurement. Consequence for clip001850: pose is unknown on 137/150 frames, so contact is correctly unknown there; static-ness must emerge as an R4 posterior. The subagent-28 held-T_rest artifact is superseded, not the pose source of record."
}
```
