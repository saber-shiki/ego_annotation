# 21 — Next-mechanism roadmap after the clip001850 contact-state render

Read-only analysis. No repo edits, nothing staged. Branch `yiwen_research`.
Inputs read: `PROMPT.md`, `EPISTEMIC.md`, `OPS.md`, `TASK_PACK.md`, `subagents/10–17`
(and 01–09 for grounding), project memory (`hand_metric_mechanisms.md`,
`object_geometry_and_render_lessons.md`, `pipeline_invariants.md`,
`self_consistency_metrics.md`). All load-bearing numbers re-verified against the
on-disk run
`/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`
and the durable artifact
`/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/`.

---

## 0. Where the task actually is (verified state, not narrative)

The clip001850 contact-state render is **done and committed**
(`6ea2f35 research: render clip001850 contact states full duration`,
`scripts/render_clip001850_v19_contact_state_full_duration.py`). Verified from the
durable manifest: 150 frames @ 30 fps; body drawn as the 3221-face
`repaired_observed_contact_body` on **all 150** frames (`body_not_trellis=true`,
`render_consumed_mesh_faces=3221`); every acceptance gate true
(`false_gap_penverts_uncertain_removed`, `no_confirmed_contact_in_28_48`,
`rendered_body_replaces_trellis_by_hash`, `defaulted_uncovered_frames`); pixel diff
vs published overlay 0.12 / world 0.08 / sbs 0.32 at f32. That artifact is not the
question. The question is which mechanism moves the HOI state next.

Three facts, measured today, govern the choice:

**Fact A — the object pose graph is inert and the keyboard is unlocalized on 91% of
frames.** From `v19_rigid_object_pose_graph_report.json`: `optimizer.nfev=1`,
`optimizer.cost=0.0`, `residual_rms_before=residual_rms_after=0.0`,
`nonpenetration_target_frame_count=0`. Per-frame `pose_measurement_status`:
`missing_initial_graph_pose=65`, `visible_surface_ineligible_for_rigid_pose_fit=72`,
`fit_to_visible_depth_samples=13` (= 150). Only 13 frames carry a real object pose
(f30–36, 45, 46, 60, 75, 76, 77); the other 137 are missing or ineligible. The 65
missing frames all share one **identical frozen translation** `[0.578,-0.096,0.339] m`
(f0–11 verified: 1 unique row of 12). The 13 fit frames are per-frame depth fits that
jump up to **166.4 mm frame-to-frame** (f76) — physically impossible for a keyboard on
a desk. So the graph is not "a static object correctly held"; it has **zero active
residuals** (cost 0) and the object is either frozen or teleporting. This is D5 (graph
inertness) **and** D1 (pose unobserved), live and measured.

**Fact B — motion coupling is dead on this slice.** Per-frame object translation step:
median **0.00 mm**, p90 **8.13 mm**, max **166.4 mm**. The nonzero motion is pose-fit
noise (Fact A), not keyboard displacement; a keyboard under typing moves sub-mm. There
is no coherent object trajectory to time-lock against hand kinematics, and there is no
trajectory at all on 137/150 frames. Motion coupling cannot confirm contact here — and
it is structurally downstream of Fact A (it needs a live trajectory first).

**Fact C — HOT3D GT machinery exists but the GT data is not on local disk.** Two
evaluators are present and runnable: `scripts/evaluate_v19_hot3d_gt_hand_to_aligned_object_mesh.py`
and `scripts/evaluate_v19_hot3d_hand_to_object_gap.py`. They require
`--hot3d-gt <objects.json/hands.json/cameras.json bundle>`, `--completed-mesh`,
`--object-alignment-report`, `--object-bop-id`, `--mano-left`, `--stream-id 214-1`.
Searches across `/data2 /data /mnt /home/yiwen` found **no** HOT3D GT json, no
clip001850 GT sidecar, and **no evaluator output ever produced for clip001850**. GT is
the only channel that can adjudicate the standing verdict (hand 80–180 mm in front of
the keyboard: genuine hovering vs keyboard-depth bias), but the blocker is **data
location/acquisition**, almost certainly server-side, not the mechanism.

These three facts, plus the standing GT-free contact verdict (zero admissible contact
frames; distance-only ceiling is `contact_candidate` because σ_clip 25–30 mm ≫
soft-tissue 2–5 mm; hand-depth counterfactual refuted at 83–202 mm shifts / 62–169 px
reprojection; geometry repair necessary-not-sufficient), fully determine the ranking.

---

## 1. Decision: the next mechanism is graph-factor liveness → object-pose trajectory (R1/D5+D1)

Ranked, with the one-line reason each earns its slot:

1. **Graph-factor liveness + object-pose trajectory (R1 → D5+D1).** The hardest
   *essential root* blocker that is *actionable now*: the graph has zero active
   residuals and the object is unlocalized on 137/150 frames. It gates motion coupling
   (needs a trajectory), full-window contact (needs posed geometry on >21 frames), and
   correct render body placement (currently a frozen/teleporting keyboard). Changes both
   numeric (graph_health rows + pose trajectory) and rendered (world-panel body) state.
   This is the milestone PROMPT.md ordered *first* ("the current graph can be inert while
   appearing implemented … must be measurable before adding more factors") and it was
   skipped straight to contact (D6). **Do this next.**

2. **HOT3D GT / R8 adjudication.** The single decisive measurement of the physical
   variable everything else has been arguing about GT-free (is the hand on the keyboard
   or hovering?). It grades the entire contact policy and becomes the instrument that
   scores every future mechanism change. Ranked #2, not #1, because it is **data-gated**
   (Fact C: GT not on local disk) and the design constraint is GT-free-first — GT is the
   tiebreaker, not the construction path. Its first step is a scoped GT-locate preflight.

3. **Motion coupling on a mechanism-live slice.** The *only* GT-free route to
   `confirmed_contact`, so it must exist in the system — but it is dead on clip001850
   (Fact B) and downstream of #1. Its correct next execution is on a **moving-object**
   demo clip (tomato / trash / scissors / putty knife), after #1 produces a live
   trajectory. This card doubles as the fix for the deeper "system only ever run on one
   degenerate slice" root blocker.

**Deprioritized — reusable render consumer (R11 delivery integration).** Explicitly *not*
next, with mechanism. The committed full-duration consumer is bespoke (7 hardcoded
clip001850 / RUN_ROOT references). Generalizing it now would build a consumer with
**nothing new to consume**: no other clip has a `contact_frame_detail` table or a
repaired body, and the one clip that does is fed by an inert pose graph (Fact A). Per
AGENTS.md, render styling / plumbing / refactor is a *support* action, admissible only to
verify a just-implemented mechanism or expose a failure — it reduces no strict physical
blocker and changes no rendered/numeric HOI state on any clip. It is the R11 bridge, and
R11 fires only "when a research output improves delivery utility and runtime." That
precondition is not met until a mechanism (card 1/2/3) produces contact/pose rows on more
than one clip and more than 21 frames. Reconsider it *after* card 1 makes the pose
trajectory live and card 3 produces a positive (non-`unresolved`) HOI state on a second
clip — then a single generalized `render_ego_hoi_contact_state(run_root, table, body)`
becomes the correct R11 move.

Delivery/research split, stated once: cards 1–3 are all **research-track** (ego.hoi
mechanism resolution, offline/CPU-light, no delivery-monotonicity obligation yet). GT
(card 2) is research-only forever — GT is unavailable at inference and exists only to
grade. The **delivery** track is the shipped v19 render + frozen runtime prediction;
nothing here touches it until an R11 integration card, which the reusable consumer will
become once fed.

---

## 2. Causal card #1 — Graph-factor liveness + object-pose trajectory (R1 / D5 + D1)

**Artifact defect (rendered + numeric).**
- *Rendered:* in `v19_world.mp4` / `v19_side_by_side.mp4`, the keyboard body is drawn at
  a **frozen** pose (`[0.578,-0.096,0.339] m`, identical across the 65
  `missing_initial_graph_pose` frames) on most of the video, and **teleports** up to
  166 mm on the 13 fit frames. The object neither tracks the scene nor holds a physical
  rest pose; on 137/150 frames the drawn body's placement is not a measurement.
- *Numeric:* `optimizer.nfev=1`, `cost=0.0`, `residual_rms=0.0`,
  `nonpenetration_target_frame_count=0`; `pose_measurement_status` =
  missing 65 / ineligible 72 / fit 13. The pose graph carries factors (temporal,
  nonpenetration) with **zero support** while the report presents as an implemented
  optimizer. This is exactly the D5 failure PROMPT.md warned about: "graph outputs can
  equal inputs … include factors with no support while tables imply optimization occurred."

**Physical variable.** The rigid object pose trajectory `T_world_object(t)` (SE3 per
frame) and its per-DOF observability — i.e. *whether the keyboard is localized in metric
world space through the clip*, and *which factor families actually constrain it*.

**Live mechanisms (why the object is unlocalized).**
- M1 — **No seed / correspondence support** (D5-M1, D1-M1): the 65 `missing_initial_graph_pose`
  frames never received an initial pose; the graph had nothing to propagate, so no
  temporal factor could fire. Cause upstream of the solver: absent inter-frame
  correspondences / seeds.
- M2 — **Visible surface degenerate / ineligible** (D1-M2): the 72
  `visible_surface_ineligible_for_rigid_pose_fit` frames had visible keyboard pixels that
  failed the rigid-fit eligibility test (too few points, near-planar, low texture) →
  low-observability rotation/translation DOFs.
- M3 — **Solver plumbing inert** (D5-M2/M3): even where seeds exist (13 fit frames), the
  *temporal graph* did not run (nfev=1, cost 0); the poses are independent per-frame depth
  fits, not a graph solve, so nothing smooths the 166 mm jumps or fills gaps. The
  variables may not be wired to a live objective.
- M4 — **Static-object null result (steelman)**: the keyboard truly barely moves, per-frame
  depth fit is adequate, and the "graph" is legitimately trivial. (This must be excluded,
  not assumed.)

**Discriminating measurement.** Run the R1 graph-health instrumentation
(`scripts/build_ego_hoi_sidecar_and_graph_health.py`, already extended with the
provenance-hash routes) against this pose graph, and add three cheap CPU probes on the
13 fit frames + gap boundaries:
- (a) **Factor support fraction & active-residual count by family** (temporal-smooth,
  depth-fit, nonpenetration): distinguishes M3 (zero support everywhere) from M4 (nonzero
  depth-fit support, zero temporal need).
- (b) **Held-out inter-frame reprojection / depth residual** on a candidate correspondence
  track over the ineligible frames: distinguishes M1 (no track exists) from M2 (track
  exists but rotation DOF has flat residual = low observability).
- (c) **Per-DOF residual curvature (Hessian diagonal)** on the fit frames: distinguishes a
  well-constrained static pose (M4: sharp in all 6 DOF) from a degenerate one (M2: flat in
  ≥1 DOF).

**Predicted outcomes (measurement pattern per mechanism).**
- If **M3 (inert plumbing)** dominates: support fraction ≈ 0 on *all* families even on the
  13 fit frames, and the 166 mm jumps have no temporal residual opposing them. → the graph
  is a pass-through; variables not wired.
- If **M1 (no seed)** dominates: correspondence track over the 65 missing frames is
  *absent* (probe (b) returns no track), while eligible frames have sharp Hessians. →
  need seeds/correspondences, not a new factor.
- If **M2 (degenerate surface)** dominates: tracks exist but probe (c) shows a flat DOF
  (typically yaw about the desk normal or along-keyboard translation) and held-out residual
  is insensitive to that DOF. → observability limit; represent as high covariance, do not
  fabricate a pose.
- If **M4 (true static)** holds: sharp Hessians on all 6 DOF at the fit frames, depth-fit
  support nonzero, and the "missing" frames are genuinely occluded (object out of view). →
  the correct artifact is a held pose *with declared gauge*, and the defect reduces to the
  render honestly labeling held vs measured.

**Intervention per outcome.**
- M3 → repair the variable-to-objective wiring / solver plumbing in the rigid pose graph so
  the temporal factor actually opposes the 166 mm jumps; re-solve; the 13 fits become a
  smoothed trajectory. (Fix plumbing *before* touching noise weights.)
- M1 → build the R2 temporal correspondences (2D/3D mask tracks) that seed the missing
  frames; feed robust-Procrustes seeds into the graph so the 65 missing frames get a pose.
- M2 → keep the pose but attach per-DOF covariance; route the flat DOF as
  `low_observability` in the graph-health row and render that DOF's body edge as uncertain.
- M4 → stop treating "missing" as a defect; emit an explicit `pose_held` /
  `pose_out_of_view` state with declared gauge, and make the render draw held-pose frames
  distinctly from measured-pose frames.

**Expected table / render state change.**
- *Tables:* `graph_solutions` / graph-health rows gain real `support_fraction`,
  `active_residual_count`, `input_output_delta`, per-DOF `covariance`, and
  `stale_dependency`/`gauge` per factor family (today all implicitly zero/absent).
  `object_pose` rows change from 65 frozen-identical + 13 jumpy to a
  smoothed-or-declared-held trajectory with covariance.
- *Render:* the world-panel keyboard stops freezing/teleporting; frames gain a visible
  measured-vs-held distinction; and — the downstream payoff — with pose available on more
  than 13 frames, `contact_frame_detail` can extend beyond the 21-frame window (the 129
  `unresolved_evidence_incomplete` frames shrink), because contact can be tested against
  posed geometry where pose becomes valid.

**Why now / blocker reduced.** This is the strict root blocker under the anti-avoidance
invariant: motion coupling and full-window contact both name "need a live object
trajectory" as their precondition, and the render defaults 129/150 frames to unresolved
partly because pose exists on only 13. It is CPU-only (Procrustes / residual probes /
re-solve), runs in seconds-to-minutes, respects the runtime and heavy-compute-placement
invariants, and needs no new data. Research track.

---

## 3. Causal card #2 — HOT3D GT / R8 contact adjudication

**Artifact defect (numeric, adjudication gap).** The standing clip001850 verdict is
`unresolved` with **zero admissible contact frames**, but the causal model still holds
two live, GT-free-inseparable hypotheses (subagent 14 §0): the hand is genuinely hovering
80–180 mm above the keys, *or* the keyboard depth surface is biased far. Every rendered
`geometry_epoch_contaminated` / `full_frame_depth_leak` / `unresolved_incoherent_evidence`
label, and the σ_clip-driven `contact_candidate` ceiling, are **unvalidated** — no
measurement of the true physical gap exists. The evaluators that would produce it have
never run for this clip (Fact C).

**Physical variable.** The ground-truth hand-to-keyboard metric gap and contact interval:
GT MANO surface vertices vs GT object pose (BOP id) in camera/world frame — the quantity
σ_clip and the whole contact policy are proxies for.

**Live mechanisms (what GT would discriminate).**
- M1 — **Hand genuinely hovering/reaching**: GT gap ≈ the measured 80–180 mm; the pipeline
  is correct; `unresolved`/`no_contact_supported` is the right terminal artifact.
- M2 — **Keyboard-depth bias**: GT gap ≪ 80–180 mm (near 0 on typing frames); the SAM2/UniDepth
  keyboard surface is biased far; the defect is object-depth, not hand or contact.
- M3 — **Hand-metric bias larger than measured**: GT wrist places the hand closer than
  HaWoR did; the 25–45 mm error scale (Track J/M) is under-stated for this clip.
- M4 (data) — **GT is not locally available**: the evaluator cannot run without a
  GT-acquisition step.

**Discriminating measurement.**
- Step 0 (**preflight, decides everything**): locate the HOT3D GT bundle for clip001850
  (objects.json / hands.json / cameras.json, stream `214-1`, object BOP id) and the
  SMPLX/MANO layer files the evaluator imports. Search server/A800 and the HOT3D download
  cache, not just the local workstation.
- Step 1 (if present): run `evaluate_v19_hot3d_gt_hand_to_aligned_object_mesh.py` and
  `evaluate_v19_hot3d_hand_to_object_gap.py` on f28–48 → GT hand-to-object gap median /
  p10 / min per frame, and `hot3d_gt_contact_compatibility` at σ.

**Predicted outcomes.**
- M4 first: if the preflight finds **no** GT bundle → the card's deliverable *is* the
  recorded acquisition blocker (which files, which host), and R8 stays deferred. Honest
  stop, not a fake pass.
- M1: GT gap ≈ 80–180 mm, `contact_compatibility` low → the render's `unresolved`/no-contact
  labels are **confirmed**; the policy is validated; publish GT as the R8 reference.
- M2: GT gap ≈ 0 on typing frames while masked depth said 80–180 mm → **keyboard-depth
  bias** is the real defect; reroute effort to object-surface depth repair, and the
  `full_frame_depth_leak`/`geometry_epoch_contaminated` labels are *right for the wrong
  reason*.
- M3: GT hand closer than HaWoR → update the σ_clip / Track-J error scale for this clip
  from an independent channel (never from the keyboard gap — that circularity is the trap).

**Intervention per outcome.**
- M4 → record the GT-locate blocker with exact expected paths/host; request server-side GT
  provisioning; keep GT-free path (card 1) as the primary.
- M1 → freeze the contact policy as validated; promote GT to the frozen R8 development-set
  reference that grades all later mechanism changes.
- M2 → open an object-depth-repair experiment (keyboard surface bias), demote the
  masked-depth surface from "trusted channel," and re-derive source-gap against a
  GT-anchored surface.
- M3 → revise `combined_metric_uncertainty_sigma_m` from the GT-measured along-ray bias;
  the contact bins shift accordingly.

**Expected table / render state change.**
- *Tables:* a new `hoi_validation_metrics` row per frame with `hot3d_gt_hand_to_object_gap_m`,
  `gt_contact_compatibility`, and `verdict_agreement` (does GT agree with the rendered
  `contact_state`?). If M1, `contact_frame_detail` gains a `gt_adjudicated=true` provenance
  column and — only here — a frame may legitimately reach `confirmed_contact` via the GT
  channel (the floor-independent release valve, never a signed-distance threshold).
- *Render:* labels that GT confirms get a `GT-adjudicated` provenance tag; a
  keyboard-depth-bias finding (M2) would flip the world-panel body correction and the gap
  banner. Under M4, nothing renders — the artifact is the recorded blocker.

**Why now / blocker reduced.** GT resolves the deepest *physical* uncertainty and becomes
the measurement instrument for the entire research track (R8 grades R2–R7, R9, R10). It is
ranked below card 1 only because it is data-gated and the design is GT-free-first; its
preflight is the cheap first move and its result either validates months of GT-free
reasoning or redirects it. Research track; runs offline / server-side (heavy-compute
placement invariant); never enters delivery.

---

## 4. Causal card #3 — Motion coupling on a mechanism-live slice (D3/workstream F)

**Artifact defect.** The HOI system possesses **no floor-independent contact channel** —
the only path to `confirmed_contact` GT-free (policy §0-F2) is object-motion onset
time-locked to hand kinematics, and it has never been built or exercised. Consequently
every clip is capped at `contact_candidate`/`unresolved`, and the system has only ever
been demonstrated on **one degenerate slice** (clip001850 keyboard: static object,
contact-ambiguous, GT-free-unresolvable). The deeper defect is that the mechanisms have
never been run on a case where they can produce a *positive* HOI state.

**Physical variable.** The temporal correlation between object rigid-body velocity/acceleration
onset and hand velocity — the causal signature "the object started moving because the hand
moved it," which is present *only when the object actually moves*.

**Live mechanisms.**
- M1 — **Genuine motion coupling**: on a manipulated moving object, object-motion onset
  lags/locks to hand kinematics → contact is confirmable independent of the metric floor.
- M2 — **No coupling because the object is static** (clip001850): real object motion is
  sub-mm, below the pose noise floor (Fact B: median step 0, jumps are pose-fit artifacts)
  → the channel is legitimately silent; contact stays `unresolved`.
- M3 — **Spurious coupling from pose noise**: the 166 mm pose-fit jumps (Fact A) correlate
  with nothing physical; naive motion coupling on clip001850 would read them as "motion" →
  a false confirmed_contact. This is the failure mode that makes clip001850 the wrong slice
  and makes card 1 a hard prerequisite (a live smoothed trajectory removes the artifact).

**Discriminating measurement.** Select a **moving-object** demo clip (tomato / trash /
origami / scissors / putty knife — TASK_PACK workstream H demo-regime set) where the object
is picked up and displaced, produce a live object trajectory for it (card-1 machinery),
then compute per-interval: object speed/acceleration onset time, hand speed at the object
surface, and their cross-correlation lag; compare against the `contact_candidate` source-gap
intervals.

**Predicted outcomes.**
- M1 on a moving clip: a compact interval where object acceleration onset locks to hand
  velocity within a small lag *and* source-gap is within σ_clip → the channel fires; that
  interval is promotable to `confirmed_contact`.
- M2 (re-verified on clip001850): object speed ≈ 0 within the trajectory noise; the channel
  returns silence; contact remains `unresolved`. Correct null.
- M3 (guard): if "motion" tracks the pose-fit jump timestamps rather than hand kinematics,
  the coupling is an artifact → reject; do not confirm contact; card 1 must fix the
  trajectory first.

**Intervention per outcome.**
- M1 → add the motion-coupling channel to `contact_frame_detail` as the floor-independent
  corroboration column; promote qualifying intervals to `confirmed_contact` with
  `motion_coupled=true` provenance; render a solid contact mark *only* on those frames.
- M2 → record the null for clip001850 (a real information limit, carried as uncertainty,
  not engineered away) and keep the channel for objects that move.
- M3 → block promotion; feed back to card 1 that the trajectory is still noise-dominated.

**Expected table / render state change.**
- *Tables:* `contact_frame_detail` gains `motion_coupling_lag`, `object_speed`,
  `motion_coupled` (bool); on a moving clip, some intervals move from `contact_candidate`
  to `confirmed_contact` with that provenance populated. On clip001850, the column is added
  and honestly reads null/`no_object_motion`.
- *Render:* on a moving-object clip, the overlay/world video shows a **solid** contact mark
  and event label on motion-confirmed intervals — the first positive (non-`unresolved`) HOI
  contact state the system will have produced. On clip001850 the render is unchanged
  (correct: nothing to confirm).

**Why now / blocker reduced.** This is the only mechanism that lifts the system's GT-free
ceiling off `contact_candidate`, and it simultaneously breaks the single-degenerate-slice
trap by forcing execution on a case where mechanisms fire. It is ranked #3 because it is
strictly downstream of card 1 (needs a live, de-noised trajectory or it produces M3 false
positives) and because clip001850 is the wrong object for it. Research track; CPU-light
(velocity/correlation), within runtime invariants.

---

## 5. One-paragraph summary for the parent

The clip001850 contact render is committed and correct; do not treat "make it reusable" as
next — that is R11 plumbing with nothing new to consume, and it reduces no physical blocker.
The measured root blocker is the object-pose graph: it is inert (`nfev=1`, `cost=0`, zero
active residuals, zero nonpenetration support) and the keyboard is unlocalized on 137/150
frames (65 frozen-identical, 72 ineligible, only 13 real fits that jump up to 166 mm). Fix
graph-factor liveness / the pose trajectory first (card 1) — it is CPU-only, actionable now,
and it unlocks motion coupling, full-window contact, and correct render body placement.
Second, run HOT3D GT/R8 (card 2) — the machinery exists but the GT bundle is not on local
disk, so its first step is a server-side GT-locate preflight; GT is the only adjudicator of
the hovering-vs-keyboard-depth circularity and becomes the instrument that grades everything
after. Third, build motion coupling on a *moving-object* demo clip (card 3), after card 1
gives it a clean trajectory — it is the only GT-free path off the `contact_candidate` ceiling
and it retires the single-degenerate-slice problem. Motion coupling is dead on clip001850
(the keyboard doesn't move; its "motion" is pose noise), so do not run it there.

---

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Delivered a mechanism-ranked roadmap with top-3 causal cards at the authoritative path, without widening scope (analysis only, no code/mechanism executed). Decided the next mechanism = graph-factor liveness + object-pose trajectory (R1/D5+D1), ranked HOT3D GT/R8 #2, motion coupling #3, and explicitly deprioritized the reusable render consumer as R11 plumbing that reduces no physical blocker. Each card carries all seven TASK_PACK analysis-contract fields (artifact defect, physical variable, live mechanisms, discriminating measurement, predictions, intervention-per-outcome, expected table/render state change) plus the delivery/research split, and every route is tied to an on-disk rendered/numeric artifact (pose graph report, contact_frame_detail, v19 render manifest, HOT3D evaluators)."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Every load-bearing claim re-verified against on-disk state with commands below: pose graph inert (nfev=1/cost=0/np_target=0) and 137/150 frames unlocalized (missing 65 / ineligible 72 / fit 13, fit indices 30-36,45,46,60,75,76,77); object 'motion' median 0.00mm / p90 8.13mm / max 166.4mm with 65 missing frames frozen at identical [0.578,-0.096,0.339]; full-duration render committed (6ea2f35), 150 frames, body=repaired_observed_contact_body 3221 faces on all frames, all acceptance gates true, 7 hardcoded clip001850 refs; HOT3D GT evaluators present but no GT bundle/output found on /data2 /data /mnt /home. Independent reviewer can rerun the cited commands to confirm."
    }
  ],
  "changedFiles": [
    ".memory/tasks/2026-07-06-research-track-execution/subagents/21_next_research_mechanism_roadmap.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "read PROMPT/EPISTEMIC/OPS/TASK_PACK + subagents 10-17 (+01-09 grounding) + project memory", "result": "passed", "summary": "Loaded task spec, D1-D7 mechanism map, contact policy, the two closed repair branches, and both render-consumer attacks (G0/A1-A8/KT-U)."},
    {"command": "python3 inspect v19_rigid_object_pose_graph_report.json (optimizer + pose_measurement_status + per-frame translation)", "result": "passed", "summary": "nfev=1, cost=0.0, residual_rms=0.0, nonpenetration_target_frame_count=0; status missing=65/ineligible=72/fit=13; fit frames 30-36,45,46,60,75,76,77; motion step median 0.00 p90 8.13 max 166.4 mm; f0-11 frozen identical [0.578,-0.096,0.339]."},
    {"command": "python3 inspect research_clip001850_contact_state_20260706/v19_contact_state_full_duration/manifest.json", "result": "passed", "summary": "150 frames @30fps; contact_state_counts unresolved_evidence_incomplete=129/pose_unresolved=12/geometry_epoch_contaminated=7/full_frame_depth_leak=1/unresolved_incoherent_evidence=1; body=repaired_observed_contact_body 3221 faces on all 150 (body_not_trellis=true); all acceptance_gates true; pixel diff overlay 0.12/world 0.08/sbs 0.32."},
    {"command": "git log/ls-files + grep hardcodes on render_clip001850_v19_contact_state_full_duration.py", "result": "passed", "summary": "Committed 6ea2f35, tracked and clean; 7 hardcoded clip001850/RUN_ROOT references -> bespoke one-clip script (reusability = R11 plumbing)."},
    {"command": "find /data2 /data /mnt /home for HOT3D GT + grep evaluate_v19_hot3d_gt_hand_to_aligned_object_mesh.py args", "result": "passed", "summary": "Evaluators evaluate_v19_hot3d_gt_hand_to_aligned_object_mesh.py + evaluate_v19_hot3d_hand_to_object_gap.py present (require --hot3d-gt/--completed-mesh/--object-alignment-report/--object-bop-id/--mano-left/--stream-id 214-1); NO GT bundle or clip001850 evaluator output found anywhere on local disk."},
    {"command": "git status --short (context only, no edits)", "result": "passed", "summary": "On yiwen_research; pre-existing dirty files (v18/v19 scripts, docs, AGENTS.md) are not mine; only new file is this findings doc at the authoritative path; nothing staged."}
  ],
  "validationOutput": [
    "Next mechanism = graph-factor liveness/object-pose trajectory: root blocker, measured live (nfev=1/cost=0, 137/150 unlocalized), actionable CPU-only now, gates cards 2/3 and full-window contact + render body placement.",
    "HOT3D GT/R8 = decisive adjudicator (hovering vs keyboard-depth circularity) and the instrument that grades all future mechanisms; ranked #2 because data-gated (GT not on local disk) -> first step is a server-side GT-locate preflight.",
    "Motion coupling = only GT-free path off the contact_candidate ceiling; dead on clip001850 (object motion is pose noise: median 0.00mm, jumps 166mm are pose-fit artifacts) -> relocate to a moving-object demo clip, downstream of card 1.",
    "Reusable render consumer deprioritized: bespoke committed script (7 hardcodes); generalizing now = R11 plumbing consuming nothing new (one clip has a table, fed by an inert graph); reduces no physical blocker; reconsider after cards 1+3 produce rows on a 2nd clip.",
    "Every card tied to on-disk numeric/rendered artifacts and the delivery(v19 shipped render)/research(ego.hoi) split; all cards are research-track, CPU-light or server-side, within runtime and heavy-compute-placement invariants."
  ],
  "residualRisks": [
    "HOT3D GT may exist on the server/A800 or a HOT3D download cache not visible from the local workstation; card 2's M4 branch (record acquisition blocker) covers this, and the GT-locate preflight must search server-side before concluding GT is unavailable.",
    "sigma_clip / 25-45mm hand-error scale and the geometry/hand-depth verdicts are inherited from subagents 07/10/11/12/14 (measured, not re-derived here); card 2 (GT) is the intended validator and may revise them.",
    "Card 1's static-object steelman (M4) is possible: if the keyboard is genuinely static and only the render's held-vs-measured labeling is wrong, the fix is smaller than a full graph repair - the discriminating probes (support fraction, Hessian, held-out residual) are designed to catch this before large intervention.",
    "This is decision/roadmap analysis; it executes none of the three cards. The discriminating measurements and interventions are the acceptance gate for whoever runs them.",
    "Pre-existing untracked scripts (analyze_v19_contact_patch_support_posterior.py, apply_v19_contact_support_translation_gate.py, build_v19_contact_support_object_pose_candidate.py; Jun27/Jul3) are prior contact-to-object-pose explorations, not part of this task's chain; noted, not relied upon."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one task-memory findings document at the authoritative subagents path (21_next_research_mechanism_roadmap.md). No source files touched; nothing staged.",
  "reviewFindings": [
    "no blockers in this deliverable (analysis-only, no-edit task honored).",
    "primary finding (root blocker): the object pose graph is inert (nfev=1, cost=0.0, residual_rms=0.0, nonpenetration_target_frame_count=0) and the keyboard is unlocalized on 137/150 frames (missing 65 / ineligible 72 / fit 13); 65 missing frames are frozen at one identical pose and the 13 fits jump up to 166mm - this is the measured D5+D1 defect that should be fixed before motion coupling or a reusable consumer. Fix graph-factor liveness first.",
    "finding: HOT3D GT is data-gated - the evaluators exist but no GT bundle or clip001850 evaluator output is on local disk; R8's first move is a server-side GT-locate preflight, not a render change.",
    "finding: motion coupling is dead on clip001850 (object motion = pose noise) and downstream of the graph fix; run it on a moving-object demo clip to avoid false confirmed_contact from pose-fit jumps.",
    "finding: the reusable render consumer is R11 delivery plumbing that reduces no physical blocker and has nothing new to consume; deprioritize until a second clip carries contact/pose rows."
  ],
  "manualNotes": "Core reframe: the committed clip001850 render is not the frontier - the pose graph feeding it is inert and the object is unlocalized on 91% of frames (verified nfev=1/cost=0 and 137/150 missing-or-ineligible, 65 frozen at [0.578,-0.096,0.339], 13 fits jumping to 166mm). Do graph-factor liveness/pose-trajectory next (card 1, CPU-only, actionable, unlocks the rest). Then GT/R8 (card 2) but budget a server-side GT-locate preflight first because no GT bundle is on local disk. Then motion coupling on a moving-object clip (card 3), never on clip001850 where 'motion' is pose noise. Do NOT make the render consumer reusable next - it is R11 plumbing consuming nothing new and reduces no physical blocker; it becomes correct only after cards 1+3 produce contact/pose rows on more than one clip and more than 21 frames."
}
```
