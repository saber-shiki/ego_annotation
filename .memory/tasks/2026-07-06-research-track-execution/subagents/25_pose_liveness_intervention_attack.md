# 25 — Adversarial acceptance criteria for the graph-liveness intervention

Read-only attack. Branch `yiwen_research`. No repo edits, nothing staged. One
CPU-only demonstration run of the *pre-existing* smoother was executed to
/tmp and deleted; no repo file was touched.

Inputs read: `PROMPT.md`, `EPISTEMIC.md`, `OPS.md`, `TASK_PACK.md`,
`.memory/project/graph_liveness_and_motion_coupling.md`, subagents 18–22, and
the on-disk run
`/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.
Source read directly: `scripts/solve_v19_rigid_object_pose_graph.py`,
`scripts/smooth_v19_rigid_object_pose_trajectory.py`,
`scripts/render_v19_contact_state_full_duration.py`,
`scripts/render_clip001850_pose_liveness_review.py`,
`scripts/fit_v18_compact_rigid_object_pose.py`.

This document does **not** execute the liveness intervention. It defines the
kill-tests that separate a *real* liveness intervention from the two
substitutions the task names (support audit; pose-smoothing heuristic), grounds
each test in verified on-disk numbers, and states the artifact contract an
intervention must satisfy to count.

---

## 0. Headline — the proposed next step is one keyword away from a category error

Subagent 21 Card 1 and subagent 22 §5/§8 both converge on: "make the graph a
trajectory smoother — add a smoothness/measurement term on the pose itself, or
replace P15 with `smooth_v19_rigid_object_pose_trajectory.py`." That framing
contains a hidden premise: **that there is an object trajectory to repair.**

The verified evidence says there is not. On clip001850 the keyboard is a
static, desk-rested object. Its "trajectory" is 9 tightly-clustered
measurements of one rest pose plus 4 outlier/degenerate fits, over a background
of 137 frames with no usable measurement. Framing the intervention as
*trajectory smoothing* pre-commits to the wrong physical model and licenses the
exact heuristic the task warns against: an acceleration prior that connects the
static cluster to the outliers with a fabricated slide. **I ran the pre-existing
smoother to confirm this** (§7, FM1): it produced a ~180 mm fabricated keyboard
slide, added zero localization support (still 13/150 direct), applied a 78 mm
correction that *moves a measured pose to fit the prior*, and left the dominant
rendered defect (72 trailing frames frozen ~200 mm off the rest pose) essentially
unchanged (206.8 → 199.2 mm).

The physical variable is **`T_world_object(t)` and its per-DOF observability**,
not "a smoother trajectory." The primary experiment is not smoothing — it is
**deciding whether the object is static, moving, or unobserved, per frame**, then
choosing the estimator that matches. The kill-tests below force that decision
before any solver runs and make every downstream route falsifiable at the
artifact.

---

## 1. Verified mechanism ground truth (the facts that reframe the problem)

All re-verified against the on-disk report
`measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`
and the visible-geometry annotations
`measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json`.

**F1 — The graph is inert by construction, not by broken wiring.** The optimizer
variables in `solve_v19_rigid_object_pose_graph.py` are per-frame *correction
deltas* seeded at zero (`x0 = np.zeros(...)`). The residual
(`residual_vector`) contains a delta→0 anchor, a temporal-step and acceleration
prior **on the correction field** (comment, verbatim: *"Smooth the correction
field, not the physical object trajectory, so real object motion measured by ICP
is preserved."*), and a nonpenetration target term that is *conditional* on a
constraint report. This run passed no `--constraint-report`
(`inputs.constraint_report = null`) and the P16 constraint state has 0/300 rows
in the accepted state (non-watertight mesh) — so `nonpenetration_target_frame_count = 0`.
Every remaining term is homogeneous in the deltas → `r(x0)=0`, `‖Jᵀr‖=0`, and
`least_squares` terminates at the seed: `nfev=1, cost=0.0, residual_rms=0.0`.
The 13 ICP poses pass through unchanged. **"Inert plumbing / unwired variables"
(Card-1 M3) is a misdiagnosis: the solver is wired and correctly at its global
optimum; the objective simply has no trajectory term and no active driver.**
Corroborated independently by subagent 22 §0–§4.

**F2 — The object is static; the "jumps" are outliers/degeneracy, not motion.**
The 13 direct fits split cleanly:

| group | frames | dist from cluster median | rot from mean | obs→mesh median |
|---|---|---|---|---|
| static cluster | f30–f46 (9) | 14.6–40.8 mm | 8.6–12.6° | 11.4–26.3 mm |
| outliers | f60, f75, f76, f77 (4) | 74.0, 189.9, 175.5, 182.7 mm | 30.2, 35.5, 11.2, 10.4° | 11.6, **5.6**, 31.8, 35.8 mm |

A 166 mm frame-to-frame step (f75→f76) is ~5.7 m/s for a keyboard mid-typing —
physically impossible. The 9-frame cluster (≈25 mm / ≈10° spread) is one rest
pose measured under ICP/depth noise. Subagent 19 independently found the object
motion floor is pure jitter (median 21 mm, no hand-coupling, Spearman ρ=−0.25).

**F3 — Low surface residual does NOT certify pose (planar/partial degeneracy).**
f75 sits **189.9 mm and 35.5° from the rest pose yet has the best obs→mesh
fit of all 13 frames (5.6 mm).** A partial, near-planar keyboard-top patch fits
the mesh at many rotations/translations along the weak DOFs. Any estimator that
trusts the surface-fit residual (ICP, the smoother's measurement anchor) will be
*confidently wrong* here. This is visible-surface degeneracy, measured.

**F4 — Two distinct "unlocalized" populations, not one.** `pose_measurement_status`:

| status | count | visible kb points | meaning |
|---|---|---|---|
| `fit_to_visible_depth_samples` | 13 | 2500 each | clean extent-consistent fits |
| `missing_initial_graph_pose` | 65 | **0 each** | keyboard not visible (occluded / out of frame) |
| `visible_surface_ineligible_for_rigid_pose_fit` | 72 | **2500 each** | surface present but **extent ratio 1.66× (max 2.43×)** → `systematic_mask_extent_inconsistent…probable_hand_background_leakage` |

The eligibility flag comes from `rigid_pose_observation_eligible` set upstream in
`build_v19_visible_geometry_from_sam2_depth.py` on metric-extent consistency
(eligible frames: extent ratio 0.83–1.15). **The 65 "missing" frames have no
seed possible (no surface). The 72 "ineligible" frames have abundant surface that
is discarded because the SAM2 mask leaked onto hand/background — latent support
that a real intervention could recover by cleaning the mask, not by smoothing.**

**F5 — The observed surface itself moves ~180 mm at f75–77 (pose-independent).**
World-frame observed keyboard centroid (straight from depth backprojection, no
pose fit): f30–46 cluster at x≈0.59–0.61; f75/76/77 at x≈0.40–0.46. So the
outliers are not just bad ICP on stable points — the underlying "keyboard" points
relocated ~180 mm. Combined with f78–149 all being extent-leaked, the parsimonious
account is **the mask drifts off the keyboard onto hand/background starting ~f60
and fully delaminates by f78.** This is *not* settled by pose-fit residuals; it
requires the adjudication in KT-L1. It is exactly the ambiguity a smoother
silently resolves as "the keyboard slid."

**F6 — Body placement is driven by the raw pose rows; graph-health rows reach no
pixel.** `render_v19_contact_state_full_duration.py` places the object body from
`pose_row["rotation_world_from_completed_canonical_matrix"]` and
`pose_row["translation_world_m"]` (lines 188–189). It reads no
support_fraction / covariance / graph_health. **Any liveness deliverable that
lands in a separate graph_health table changes zero pixels** unless it rewrites
the pose rows the renderer consumes. The current render therefore draws the body
frozen at the f77 outlier pose ([0.3897,−0.1101,0.3444], ~180 mm from rest) for
all 72 frames f78–149, and ramps toward it across the 35 interpolated frames
f37–74 — i.e. **~107/150 frames render the body materially off its true rest
location.** That is the dominant, viewer-visible defect.

**F7 — Correction to prior counts.** Subagent 21 / the project memory note say
"65 frozen missing-pose frames share one identical translation." Verified false
in detail (subagent 22 §7 concurs): 30 frames (f0–29) hold at the f30 pose
[0.5776,−0.0958,0.3385]; 72 frames (f78–149) hold at the *different* f77 pose
[0.3897,−0.1101,0.3444]; 35 frames interpolate. 102 nearest-hold frames collapse
to **two** unique translations. The qualitative verdict (unlocalized, jumpy)
stands; the number the render is frozen at (the f77 outlier) matters because it
is the one that is ~180 mm wrong.

---

## 2. The physical variable and the primary discriminator

Physical variable: `T_world_object(t)` (SE3 per frame) **and its per-DOF
observability/gauge**. The correct estimator depends on which of three regimes
each frame is in:

- **STATIC + observed** — one rest pose, measured. Estimate a single robust
  `T_rest` with covariance; per-frame state = measured/held-static.
- **MOVING + observed** — a real trajectory. Smoothing/interp is admissible with
  per-frame support.
- **UNOBSERVED** (no surface, or mask leaked) — no localization claim. State =
  occluded / out-of-view / mask-unreliable; render as uncertain or held-with-gauge,
  never as measured.

**The primary experiment is not "smooth the poses." It is KT-L1: classify each
frame into {static-observed, moving-observed, unobserved} using pose-independent
evidence, before choosing an estimator.** Skipping this step is precisely how a
support audit or a smoother gets substituted for the physical variable: both
assume the regime instead of measuring it.

---

## 3. Artifact-level kill-tests (discriminate the four diagnoses)

Each test states the observation, the discriminating prediction per mechanism,
and the routing. All are CPU-only on existing artifacts. A "real liveness
intervention" is admissible only if it runs KT-L1 and KT-L2 first and its route
matches their outcome.

### KT-L1 — Static vs moving vs mask-drift (pose-independent) — **run first**
- **Observation:** world-frame observed keyboard centroid + point cloud per
  direct frame (from `centroid_world_m` / `world_vertices_sample_m`, no pose
  fit); plus per-frame mask IoU against a temporally-propagated keyboard track,
  and reprojection of the observed points into the image to check they land on
  the keyboard (not the hand/desk).
- **Predictions:**
  - *Static* → observed centroid stable within depth noise (≈25–40 mm) across
    all frames flagged "observed"; no coherent monotonic displacement.
  - *Moving* → observed centroid traces a **coherent, monotonic, low-acceleration
    path**, and the moving frames still project onto a keyboard-consistent mask.
  - *Mask-drift* → observed centroid jumps to a new location that coincides with
    the hand/forearm/desk region and the mask extent inflates (→ KT-L2).
- **Measured now (partial):** f30–46 centroid stable (~30 mm spread); f75–77
  centroid ~180 mm away with mask extent about to delaminate (F5) → *mask-drift*
  is the leading hypothesis but the IoU/reprojection check is the deciding
  observation and must be produced as the test's artifact.
- **Routing:** static → §4 static-held gauge; moving → §4 smoothing (only for
  the frames that pass); mask-drift/unobserved → §4 unobserved routing +
  seed-rebuild attempt (KT-L2).

### KT-L2 — Latent support recoverable vs genuinely absent (the 72 vs 65 split)
- **Observation:** for the 72 extent-ineligible frames, re-segment/clean the
  keyboard mask (remove hand/background using MANO occupancy + the anchor extent
  box) and recompute the extent ratio and a fresh ICP fit; for the 65 zero-point
  frames, confirm the object is truly out of view.
- **Predictions:**
  - *Recoverable (M2a)* → after mask cleaning, extent ratio returns to ≈1.0 on a
    subset of the 72 frames and ICP produces a pose consistent with the rest
    cluster → real added support; `direct_row_count` rises above 13.
  - *Degenerate (M2b)* → surface survives cleaning but only a planar/partial
    patch remains; pose is estimable only up to a weak DOF → covariance-only
    routing (KT-L4).
  - *Absent (M1)* → 65 zero-point frames stay zero → honest occluded/out-of-view.
- **Routing:** distinguishes "no seed support" (M1, honest hole) from "visible
  surface being thrown away" (M2a, recoverable) from "degenerate surface"
  (M2b, covariance). **Neither smoothing nor median touches this axis** — this is
  the test that separates real support gain from cosmetic gap-fill.

### KT-L3 — Inert wiring vs no driver (proves the graph is not a plumbing bug)
- **Observation:** inject one synthetic nonpenetration target (or one
  pose-measurement residual on a held-out clean frame) into `residual_vector` and
  re-solve; observe `nfev`, `cost`, and whether the corresponding correction
  becomes nonzero.
- **Predictions:**
  - *Wired-but-undriven (F1)* → the injected term immediately produces `nfev>1`,
    nonzero cost, and a nonzero correction on that frame only → the graph is a
    correct corrector with no driver, **not** broken plumbing.
  - *Genuinely broken* → the injected term still yields `nfev=1`, cost 0 → then
    and only then is there a wiring bug to fix.
- **Routing:** this kill-test exists to *prevent* the misdiagnosis "fix the
  plumbing." Expected outcome is "wired-but-undriven," which redirects effort to
  the estimator/objective, not to solver plumbing. If it comes back
  "wired-but-undriven," any PR whose description is "repaired variable-to-objective
  wiring" is rejected as attacking a non-existent bug.

### KT-L4 — Per-DOF observability (which DOFs are real, which are gauge)
- **Observation:** on each clean fit frame, compute the Hessian diagonal / local
  curvature of the surface-fit objective per SE3 DOF (or the spread of ICP
  restarts from perturbed inits).
- **Predictions:**
  - *Well-observed DOF* → sharp curvature; ICP restarts converge (translation
    in-plane; normal-axis position).
  - *Degenerate DOF* → flat curvature; restarts scatter (expected: yaw about the
    desk normal and/or along-plane translation on partial views — the f75
    signature).
- **Routing:** well-observed DOFs feed the rest-pose estimate; degenerate DOFs
  get high covariance and **uncertain render styling**, never a fabricated value.

### KT-L5 — Outlier vs real measurement (before any estimate consumes the 13 fits)
- **Observation:** robust consensus (e.g. median/MAD or RANSAC on SE3) over the
  13 fits; flag frames whose pose is > (motion-floor) from consensus AND whose
  observed points fail KT-L1 reprojection.
- **Predictions:**
  - f60/f75/f76/f77 exceed consensus by 74–190 mm and (per KT-L1) their observed
    points fall off the keyboard → **rejected outliers**, excluded from the
    rest-pose estimate and from any trajectory.
  - f30–46 form the consensus → the rest pose.
- **Routing:** the rest-pose estimator and any smoother must consume only the
  consensus set; **outliers are relabeled `rejected_degenerate_fit`, not smoothed
  in and not medianed in.** This is the guard against both FM1 and FM2.

---

## 4. What evidence justifies each route (decision table)

An intervention must name which route it is taking and cite the kill-test
outcome that licenses it. Routes chosen without the cited evidence are
substitutions.

| Route | Admissible **only if** | Forbidden when | Verified status on clip001850 |
|---|---|---|---|
| **Trajectory smoothing** (accel prior on pose) | KT-L1 = *moving-observed* on a contiguous run of frames, each with independent support surviving KT-L2/KT-L5 | KT-L1 = static or mask-drift; support is interpolated across gaps; outliers not first removed | **NOT justified.** Object is static; the "motion" is 4 outliers + mask drift. Smoothing here fabricates motion (§7 FM1, demonstrated). |
| **Static held gauge** (single `T_rest` + covariance + visibility ledger) | KT-L1 = static; KT-L5 consensus exists; KT-L4 covariance attached; a per-frame visibility state distinguishes measured / held-visible / occluded / mask-unreliable | rest pose computed over unfiltered fits; constant stamped on unobserved frames without a visibility state (that is FM2) | **Justified for the observed frames**, provided outliers rejected (KT-L5), weak DOFs carry covariance (KT-L4), and f78–149 are marked occluded/mask-unreliable not "held-measured". |
| **Seed / correspondence building** (clean masks, temporal tracks) | KT-L2 = *recoverable*: mask cleaning restores extent≈1.0 and yields consensus-consistent fits on some of the 72 ineligible frames | claimed as "support added" when `direct_row_count` does not actually rise, or when the new fits are degenerate (KT-L4) | **The highest-value real intervention.** 72 frames carry full surface being discarded for mask leakage; recovering even part raises support above 13/150. Must be validated by KT-L2 + KT-L5, not asserted. |
| **Covariance-only routing** (keep pose, widen weak DOF) | KT-L4 = degenerate DOF on a frame whose surface is otherwise clean (KT-L2 = degenerate) | used to launder an outlier (KT-L5 reject) as "high uncertainty" instead of rejecting it | Applies to the weak yaw/normal DOF of the planar keyboard; **not** a home for the f75/f77 outliers (those are rejects, not wide-covariance keeps). |

Cross-cutting rule: **no route may claim `direct`/`measured`/`localized` on a
frame with zero visible surface (65 frames) or extent-leaked mask (72 frames)
unless KT-L2 recovered it.** Localization claimed on unobserved frames is
fabrication regardless of which estimator produced it.

---

## 5. Required rendered/numeric state change (the artifact contract)

A liveness intervention is complete only if the following change and are
inspected *in the rendered artifact*, not in a side table.

**Numeric (must change in the pose rows the renderer consumes, or a table that
provably rewrites them):**
1. Outlier frames f60/f75/f76/f77 relabeled from `direct` to
   `rejected_degenerate_fit` (or `mask_drift`), with the KT-L1/KT-L5 evidence.
2. Either (moving route) `direct_row_count` rises above 13 via KT-L2 recovered
   seeds, **or** (static route) a `T_rest` + per-DOF covariance row exists and the
   67+ "held" frames carry an explicit visibility state
   (`occluded` / `out_of_view` / `mask_unreliable` / `held_static_visible`),
   not the current opaque `nearest_visible_pose_hold`.
3. Per-DOF observability/covariance present for the rest pose or each fit (KT-L4).
4. A graph-health row that reports support_fraction, active_residual_count,
   input→output delta, and gauge — **and a hash lineage proving the render
   consumed the changed pose rows** (guards FM3).

**Rendered (must be visible in `v19_world.mp4` / `v19_side_by_side.mp4`):**
5. The body on f78–149 must **stop being drawn ~180 mm off** at the f77 outlier
   pose. Under the static verdict it moves to `T_rest` (~[0.57,−0.09,0.34]); under
   mask-drift/occlusion it is styled uncertain/occluded. Either way the world-panel
   body placement on those 72 frames changes by ~180 mm or changes styling — a
   large, viewer-obvious delta. A pixel diff on f78–149 must be non-zero and
   attributable to pose, not to a banner.
6. Measured / held-static / occluded / mask-unreliable / rejected-outlier frames
   must be visually distinguishable (the review script
   `render_clip001850_pose_liveness_review.py` already has the color vocabulary
   direct/held/interpolated — but it is a QC still-set that reads the same raw
   rows; it visualizes the defect, it does not fix the body placement).

**Acceptance gate:** if steps 5–6 do not change the full-duration world/side-by-side
video, the intervention has not touched the artifact — it is a support audit
(FM3), regardless of how rich the graph-health table is.

---

## 6. The three named failure modes — detection tests

### FM1 — Pose smoothing that hides jumps without support  *(demonstrated)*
**What it looks like:** the max frame-to-frame jump statistic drops; the world
body moves smoothly; it looks fixed.
**Why it is false:** on a static object the "trajectory" is noise + outliers;
an acceleration prior connects the static cluster to the outliers as a smooth
slide and does not add localization.
**Demonstration (I ran the pre-existing `smooth_v19_rigid_object_pose_trajectory.py`
on this run, CPU-only, output to /tmp, deleted):**
- Max direct-edge step: 166.4 mm → **121.8 mm** (cosmetic drop, still physically
  impossible for a keyboard).
- f75 distance from rest pose: 207.0 mm → **206.8 mm** (outlier **not** fixed —
  the accel prior, gap-scaled over the 14–15 frame gaps, accommodates the outlier
  as slow drift instead of rejecting it).
- Correction applied to f76: **78.6 mm** — a *measured* pose relocated to satisfy
  the prior.
- `direct_row_count`: **13 → 13**; completion still 102 hold + 35 interp. **Zero
  support added.**
- f78–149 still frozen at the (smoothed) f77 outlier, ~199 mm from rest — the
  dominant rendered defect (F6) is **not** fixed.
**Detection test:** require (a) `direct_row_count` strictly increases OR a
`T_rest`+visibility-ledger replaces the hold frames; (b) outliers appear as
`rejected_*`, not as smoothed-through points; (c) no correction exceeds the
frame's own measurement covariance without that frame being an adjudicated
outlier. The smoother fails all three → it is FM1, not a liveness fix.

### FM2 — Static median pose that fabricates localization
**What it looks like:** "the keyboard is static, so stamp the median/mean pose on
all 150 frames — now it's localized everywhere."
**Why it is false:** (a) a median over the *unfiltered* 13 fits is dragged toward
the f60/75/76/77 outliers (the outliers pull the estimate ~20–40 mm and rotate
it); (b) stamping a constant on the 65 zero-surface frames and 72 mask-leaked
frames claims the object is at that pose when it was never observed there —
the object could have been moved off-camera. A constant is only valid where the
object is *observed to be static*.
**Detection test:** the rest pose must be a robust consensus over the KT-L5
inlier set (f30–46), not a median over all 13; and every frame the constant is
applied to must carry a visibility state — `held_static_visible` (object seen,
consistent with rest) vs `occluded`/`out_of_view`/`mask_unreliable` (constant
shown only as a gauge with declared uncertainty, never as `measured`). If any
zero-surface or extent-leaked frame is labeled `direct`/`localized`, it is FM2.

### FM3 — Graph-health rows that do not affect render/body placement
**What it looks like:** a rich `graph_health` / `graph_solutions` table with
support_fraction, active_residual_count, covariance, gauge — task "done."
**Why it is false:** `render_v19_contact_state_full_duration.py` places the body
from `pose_row["translation_world_m"]` (F6); a parallel health table changes no
pixel. Per AGENTS.md this is a support action masquerading as mechanism progress.
**Detection test:** (a) a render-state hash lineage must show the render consumed
the *changed* pose rows; (b) a pixel diff on the corrected frames (esp. f78–149)
must be non-zero and attributable to body placement; (c) the health row's
input→output delta must be non-zero (the current report's `correction_summary`
is all zeros — a health row that faithfully reports "delta=0" is honest but
proves the artifact did not change). A health table with all-zero deltas and an
unchanged render is FM3.

---

## 7. Attack summary (what the parent must not let happen)

1. **Do not frame the next step as "trajectory smoothing / de-jump the
   trajectory."** That presupposes motion that the evidence contradicts (F2, F5)
   and licenses the demonstrated FM1 heuristic. Frame it as *regime
   classification → matched estimator* (KT-L1 first).
2. **Do not accept a graph-health table as the deliverable.** The render reads raw
   pose rows (F6); a health table is FM3 unless it rewrites those rows and the
   video changes.
3. **Do not accept "the graph was inert, I fixed the wiring."** KT-L3 will show
   the graph is wired and undriven (F1); there is no plumbing bug to fix.
4. **The single highest-value real intervention is KT-L2 seed recovery** on the 72
   extent-leaked frames (full surface currently discarded) plus a robust static
   `T_rest` (KT-L5) with covariance (KT-L4) and an honest visibility ledger — and
   its proof is that the world body stops rendering ~180 mm off on f78–149 (F6,
   contract step 5). Everything else is audit or cosmetics.

---

## Residual risks

- **KT-L1 not yet fully run.** The static verdict rests on pose-cluster + observed-
  centroid + subagent-19 motion-floor evidence (strong), but the deciding
  mask-IoU/reprojection observation for f60–77 (mask-drift vs real motion) is
  specified, not executed here. If f60–77 turn out to be real motion, the *moving*
  route opens for those frames — but the smoother is still wrong there because it
  connects them to the static cluster without per-frame support.
- **KT-L2 recoverability is a hypothesis.** The 72 ineligible frames *have* surface
  and a leakage reason; whether mask cleaning restores consensus-consistent fits is
  unmeasured. If unrecoverable, they are honest occlusion holes, and the ceiling on
  localization is the ~13 clean frames — which makes the static-held gauge (not
  smoothing) the only correct artifact.
- **The demonstration used the smoother's defaults** (accel sigma = median pose
  sigma). A hand-tuned sigma could reduce the fabricated slide, but tuning a prior
  to suppress a fabrication is itself an uneducated-heuristic path (AGENTS.md); the
  fix is regime classification + outlier rejection, not sigma tuning.
- **This is adversarial acceptance design, not the intervention.** It executes no
  fix. The kill-tests and the §5 contract are the gate for whoever implements it.
- **Line numbers** for `solve_v19_rigid_object_pose_graph.py` internals are cited
  from subagent 22 (re-read and consistent with the source); render body-placement
  lines (188–189) verified directly.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Delivered an artifact-level attack on the graph-liveness next step at the authoritative path. Concrete findings with file paths + severity: (BLOCKER-framing) the proposed 'trajectory smoothing' route is a category error because the keyboard is static (verified: 9 fits cluster within 14.6-40.8mm/8.6-12.6deg at rest pose ~[0.57,-0.09,0.34]; 4 outliers f60/75/76/77 at 74-190mm are degenerate/mask-drift, not motion; f75 has best surface fit 5.6mm at 190mm/35deg wrong pose). (BLOCKER) render_v19_contact_state_full_duration.py:188-189 places the body from raw pose_row translation/rotation, so graph-health tables reach no pixel (FM3), and the body is currently drawn ~180mm off on 72 frames f78-149 frozen at the f77 outlier. Defined 6 kill-tests (KT-L1..L5) that discriminate inert-wiring vs no-seed vs degeneracy vs static-held, a route-justification decision table (smoothing/static-held/seed-building/covariance-only with admit/forbid conditions), the required numeric+rendered artifact contract, and detection tests for all three named failure modes. FM1 (smoothing hides jumps without support) was empirically demonstrated by running the pre-existing scripts/smooth_v19_rigid_object_pose_trajectory.py: max step 166.4->121.8mm cosmetic, f75 outlier 207.0->206.8mm not fixed, 78.6mm fabricating correction on f76, direct support unchanged 13/150, f78-149 still ~199mm off."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "read PROMPT/EPISTEMIC/OPS/TASK_PACK + graph_liveness_and_motion_coupling.md + subagents 18-22", "result": "passed", "summary": "Loaded task spec, the two named substitution risks (support audit; pose-smoothing heuristic), and the prior inventory (subagent 22 confirms inert-by-construction)."},
    {"command": "python3 inspect v19_rigid_object_pose_graph_report.json (optimizer, pose_rows, correction_summary, completion)", "result": "passed", "summary": "nfev=1 cost=0.0 residual_rms=0.0 np_target=0; 13 direct fits split into 9-frame static cluster (14.6-40.8mm) + 4 outliers (74-190mm); 102 nearest-hold collapse to TWO translations (f30 pose x30 frames, f77 outlier pose x72 frames), 35 interp; max frame step 166.4mm at f75->f76."},
    {"command": "read scripts/solve_v19_rigid_object_pose_graph.py + smooth_v19_rigid_object_pose_trajectory.py + render_v19_contact_state_full_duration.py + fit_v18_compact_rigid_object_pose.py", "result": "passed", "summary": "Confirmed: solver variables are correction-deltas seeded at zero, smoothness on correction field not trajectory, NP term conditional and empty -> inert by construction not a bug. Render body from pose_row translation/rotation lines 188-189. Eligibility from rigid_pose_observation_eligible (metric-extent consistency)."},
    {"command": "python3 join annotations visible_geometry with pose status; extent ratios; observed centroids", "result": "passed", "summary": "65 missing frames have 0 visible kb points (no seed); 72 ineligible frames have 2500 points each but extent ratio 1.66x (max 2.43x) = mask leakage = latent recoverable support; observed centroid drifts ~180mm at f75-77 pose-independently (mask-drift vs motion ambiguity)."},
    {"command": ".venv/bin/python scripts/smooth_v19_rigid_object_pose_trajectory.py --annotations ... --pose-report P14 --completed-mesh ... --object-id keyboard --output-report /tmp/kb_smoothed_traj_probe.json (then deleted)", "result": "passed", "summary": "Demonstrated FM1: max step 166.4->121.8mm cosmetic; f75 outlier 207.0->206.8mm NOT fixed; 78.6mm correction moves measured f76 to fit prior; direct_row_count 13->13 zero support added; f78-149 still frozen ~199mm off rest pose."},
    {"command": "git status --short (context) + rm /tmp probe", "result": "passed", "summary": "On yiwen_research; 48 pre-existing dirty files are not mine; only new file is this findings doc; /tmp demonstration artifact deleted; nothing staged."}
  ],
  "validationOutput": [
    "Primary attack: 'trajectory smoothing' presupposes motion the evidence refutes (static keyboard: 9-frame cluster ~25mm/10deg, 4 outliers 74-190mm, f75 best surface fit 5.6mm at 190mm/35deg wrong pose). The intervention must be regime-classification-first (KT-L1), not smoothing.",
    "FM1 demonstrated empirically by running the pre-existing smoother: cosmetic jump reduction (166->122mm), outlier retained (207->207mm), fabricating 78mm correction, zero support gain (13/150), dominant 72-frame render defect unchanged.",
    "FM3 verified structurally: render places body from raw pose rows (render_v19_contact_state_full_duration.py:188-189); a graph-health table changes no pixel unless it rewrites pose rows and the video changes.",
    "M1/M2 discriminated by data: 65 frames 0 visible points (no seed, honest hole); 72 frames 2500 points but extent 1.66x (mask leakage, recoverable latent support). Seed recovery on the 72 is the highest-value real intervention.",
    "KT-L3 predicts the graph is wired-but-undriven (inert by construction), so 'I fixed the wiring' is a misdiagnosis to reject.",
    "Artifact contract: outliers relabeled rejected; either direct_row_count rises (moving/recovered) or T_rest+covariance+visibility-ledger replaces holds (static); and the world body must stop rendering ~180mm off on f78-149 (viewer-visible pixel change)."
  ],
  "residualRisks": [
    "KT-L1 mask-IoU/reprojection adjudication for f60-77 (mask-drift vs real motion) is specified but not executed here; static verdict rests on strong but not conclusive pose-cluster + observed-centroid + subagent-19 motion-floor evidence.",
    "KT-L2 recoverability of the 72 extent-leaked frames is a hypothesis; if mask cleaning fails, they are honest occlusion holes and static-held gauge (not smoothing) is the only correct artifact.",
    "The FM1 demonstration used the smoother's default accel sigma; tuning it could reduce the fabricated slide, but sigma-tuning to suppress a fabrication is itself an uneducated-heuristic path -- the fix is regime classification + outlier rejection.",
    "This is adversarial acceptance design; it executes no fix. The kill-tests and section-5 contract are the gate for the implementer.",
    "solve_v19 internal line numbers cited from subagent 22 (re-read, consistent); render lines 188-189 verified directly."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one task-memory findings document (25_pose_liveness_intervention_attack.md) at the authoritative subagents path. No source files touched; one CPU-only demonstration run of the pre-existing smoother written to /tmp and deleted; nothing staged.",
  "reviewFindings": [
    "blocker (framing): the proposed 'object-pose trajectory smoothing' next step is a category error -- the keyboard is static (verified 9-frame cluster ~25mm/10deg vs 4 outliers 74-190mm), so smoothing fabricates motion. Reframe as regime-classification-first (KT-L1). Severity: high -- it licenses the exact heuristic the task warns against.",
    "blocker (FM3): render_v19_contact_state_full_duration.py:188-189 draws the body from raw pose_row translation/rotation; any graph-health table is a support audit that changes zero pixels unless it rewrites pose rows and the world/side-by-side video changes. Severity: high.",
    "blocker (FM1, demonstrated): running the pre-existing scripts/smooth_v19_rigid_object_pose_trajectory.py cosmetically drops max jump 166->122mm but retains the f75 outlier (207->207mm), applies a 78mm fabricating correction, adds zero support (13/150), and leaves f78-149 frozen ~199mm off rest. Severity: high -- this is the ready-to-run trap.",
    "finding (highest-value real fix): 72 'ineligible' frames carry full visible surface (2500 pts) discarded for mask leakage (extent 1.66x); KT-L2 mask cleaning could recover real support and is the only route that raises localization above 13/150. Severity: medium (opportunity).",
    "finding (FM2 guard): a static rest pose must be a robust consensus over KT-L5 inliers (f30-46), not a median over all 13 (outliers drag it), and must never be stamped as 'measured' on the 65 zero-surface or 72 leaked frames without a visibility state. Severity: medium.",
    "correction to prior memory/subagent-21: the 102 nearest-hold frames collapse to TWO translations (f30 pose x30, f77 outlier pose x72), not one; the render is frozen ~180mm off at the f77 outlier for 72 frames -- the dominant viewer-visible defect. Severity: low (factual precision)."
  ],
  "manualNotes": "Core reframe for the parent: the liveness frontier is correct, but 'trajectory smoothing' is the wrong verb. clip001850's keyboard is static; its pose problem is (a) 4 outlier/degenerate fits to reject, (b) a robust static rest pose + per-DOF covariance to estimate, (c) 72 mask-leaked frames whose surface is being thrown away (recoverable support), and (d) 65 genuinely-unobserved frames + all trailing frames the render currently draws ~180mm off at the f77 outlier. I RAN the pre-existing smoother to prove it is FM1: cosmetic jump drop, outlier retained, 78mm fabricating correction, zero support gained, dominant render defect unchanged. Gate any liveness PR on: KT-L1 (regime classification first), KT-L3 (proves no plumbing bug), KT-L5 (outlier rejection), and the section-5 artifact contract (the world body must visibly stop rendering ~180mm off on f78-149). A graph-health table with all-zero deltas and an unchanged video is FM3, not progress."
}
```
