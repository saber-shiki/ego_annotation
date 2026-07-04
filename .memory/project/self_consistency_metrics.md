# Self-consistency (GT-free) metrics — proven decisive during the 2026-07-04 demo sprint

User ruling: these metrics matter beyond the demo; they should become a standard per-clip QC
stage of the real pipeline. Every metric below located or predicted a user-visible defect
BEFORE any ground truth was consulted. Provenance: `.memory/tasks/2026-07-04-demo-pack/OPS.md`
+ `demo_pack_20260704/review/track{F2,G,I,J,K,L}_*.md`.

## The metric family (with measured discriminative power)

1. **Cross-detector keypoint residual** (per frame, per side): median 2D distance between an
   independent 2D pose detector (rtmlib) and the primary hand stream (WiLoR/HaWoR projection).
   - Field: `rtmlib_wilor_median_keypoint_delta_px`. On trash_1050: clip-side median 284px /
     p95 761px flagged 479/1050 frames; per-frame values separated genuinely-wrong 2D states
     from fine ones.
   - **Known blind spot (critical):** 2D-consistent-but-3D-wrong states pass it (trash f231:
     delta 38px, every confidence field endorsed a hand with ~25% depth error). 2D agreement
     bounds lateral error only; pair it with a size/depth consistency check.

2. **Projected-size vs detected-size consistency** (per frame, per side): rendered hand bbox
   height vs detector bbox height. Ratio directly measures relative depth error
   (s_true/s_rend = z_rend/z_true). Caught the f231 depth-too-deep hand (117px vs ~150px →
   ~22% deep). Cheap; closes the blind spot of metric 1.

3. **Silhouette IoU: posed mesh vs segmentation mask** (per object, sampled frames): the
   single most decisive object metric of the sprint.
   - Caught the wrong tomato mesh (scene patch, IoU 0.08), selected the correct completed
     mesh (0.40–0.53), and drove evidence-based scale selection for the v19 round-body swap
     (peak mean IoU 0.553 vs 0.375; scale sweep, `renderer/tomato_v19_swap_transform.json`).
   - Measured fused trash meshes at 0.11–0.47 → predicted exactly the "bag mesh inconsistent
     with video" complaint. Threshold intuition: <0.2 = image-space claims must come from the
     per-frame mask, not the fused body.

4. **Temporal jitter**: frame-to-frame wrist displacement (mm) and root-orientation angular
   velocity (deg/frame), median + p95. Per-frame independent fits produce p95 spikes the user
   sees immediately as jitter (Track L's acceptance metric). Also: hand-mesh acceleration RMS
   per clip (Track I self-consistency table).

5. **World-frame physical plausibility**: fraction of frames with camera above hands along
   estimated gravity; camera-height span (p95−p5). Diagnosed non-gravity-aligned SLAM worlds
   (trash camera "height" span 4.4m = walking direction leaking into the vertical axis) and
   measured the honest ceiling (72% even with per-frame head-up → world is drift-bent, not
   rotation-correctable).

6. **Reprojection residual of the fitted translation** (per frame): median px of hand-joint
   reprojection under the target intrinsics. Transfers poorly across capture regimes — see
   caveats below — but within one regime it is the right per-frame confidence driver
   (renderer alpha gating uses it).

7. **Penetration volume / contact-state timeline coherence** (from v18 machinery): kept in
   the family; less exercised this sprint.

## Why this family works (mechanism)

Each metric is a *disagreement between two independent measurement paths* of the same
physical quantity (two detectors; mesh vs mask; model vs temporal prior; geometry vs
gravity). Disagreement upper-bounds correctness without GT. The sprint's empirical lesson:
**every user-visible defect corresponded to a large value in at least one of these metrics,
and each metric's blind spot was covered by another one.** The pipeline version of this: a
per-clip QC report emitting all of them per frame/object, consumed by (a) confidence tiers,
(b) automatic re-estimation triggers (e.g., hand re-run when metric 1+2 disagree with the
state's confidence), (c) buyer-facing verification (the protocol already sells exactly these).

## Consumption rules learned the hard way

- **Gate on the CURRENT layer's provenance, not stale upstream labels.** trash f850: the
  occlusion module's `unresolved` label hid a hand the current detector saw
  (`wilor_visible`, 21px), while an inferred infill drew solid. Superseded fields must not
  outrank the replacing layer's own evidence.
- **Uncertainty is represented, not omitted**: inferred/fallback states render as reduced-
  alpha ghosts; never silently hidden, never drawn solid.
- **Confidence flicker is a presentation defect**: temporal median (5f) on display alpha;
  source switches need hysteresis (≥5 consecutive frames) to avoid popping.
