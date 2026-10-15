# Mask alignment experiment R6 — September 29, 2026

**Result: useful output-mask repair, not a complete pose solution and not promoted to default.**

## What actually changed

The opt-in solver now jointly fits full-mesh rotation/translation to fixed visible boundaries and raw UniDepth surface samples, with Gaussian neighbor factors on the correction relative to the reproduced seed. One shared scale, full mesh topology, source depth, metric MANO and observations remain unchanged. Every exported world pose is re-rendered and checked against a frozen same-frame seed. The default producer and original conflicting-depth/mask tests are restored unchanged.

## Five complete cases

Common scoring width: 192 pixels. Visible domain is fixed by source MANO/ownership; raw SAM2 is visible, not amodal. No GT or CAD was read.

| Case | raw IoU median: baseline → R6 | visible IoU median: baseline → R6 | raw IoU < .5: baseline → R6 | changed frames |
|---|---:|---:|---:|---:|
| milk | 0.8420 → 0.8466 | 0.8569 → 0.8628 | 0 → 0 | 103/150 |
| soup | 0.8566 → 0.8568 | 0.9021 → 0.9079 | 0 → 0 | 110/150 |
| BBQ | 0.6527 → 0.7522 | 0.7355 → 0.8300 | 17 → 12 | 144/150 |
| mug | 0.5991 → 0.6215 | 0.6678 → 0.6847 | 18 → 13 | 104/150 |
| spatula | 0.4403 → 0.4635 | 0.4333 → 0.4632 | 102 → 93 | 89/150 |

All 750 exported poses passed the frozen-reference mask checks at 192 px. No exported-score/report discrepancies remain. Every reproduced seed was bitwise equal to jump_r2 in rotation, translation, vertices, faces and uniform scale.

The independent 320-pixel audit found **zero visible-IoU regressions greater than .015** across all 750 frames. This does not mean every higher-resolution boundary score improved: at a stricter 2-pixel boundary tolerance at 320 px, spatula median boundary recall fell from 0.640 to 0.614. Resolution-dependent silhouette/boundary uncertainty remains; no resolution-invariant guarantee is claimed.

## Motion: the newly introduced catastrophic flips were repaired, but not every motion statistic improved

| Case | rotation step p95 baseline → R6 (deg) | maximum baseline → R6 (deg) |
|---|---:|---:|
| milk | 9.73 → 10.06 | 28.04 → 23.47 |
| soup | 13.96 → 15.10 | 25.20 → 32.56 |
| BBQ | 24.32 → 33.16 | 50.25 → 43.70 |
| mug | 38.50 → 42.02 | 93.94 → 95.33 |
| spatula | 22.06 → 21.26 | 99.52 → 99.34 |

- BBQ f40: baseline 22.49°, independent R4 113.86°, temporally coupled R6 32.98°. The isolated large branch change is gone, not every frame-to-frame rotation reduced.
- Spatula f137: baseline 4.27°, R4 83.81°, R6 4.27°. The new isolated flip is gone; the inherited roughly 99° worst step elsewhere remains.
- Soup and mug still have larger maximum rotation steps than baseline. This is NOT certified physical non-regression. Silhouette evidence and generated-shape ambiguity remain insufficient to replace the default path.

## Milk checks

| Frame | visible IoU baseline → R6 | translation step baseline → R6 (mm) | rotation step baseline → R6 (deg) |
|---|---:|---:|---:|
| 143 | 0.7912 → 0.8440 | 5.588 → 6.132 | 3.785 → 1.883 |
| 148 | 0.7705 → 0.8023 | 7.796 → 6.725 | 4.835 → 3.581 |
| 149 | 0.7723 → 0.8374 | 3.510 → 3.946 | 0.945 → 4.366 |

Mask support improves on all three; the old large jump failure is not reintroduced. f149 rotation step nevertheless increases from about .945° to 4.366°, so do not describe every motion quantity as unchanged or improved.

## Visual inspection and limits

- Decoded all 20 baseline/new overlay/world videos: 3,000 frames. Inspected the 15 contiguous timeline sheets covering all 750 R6 frames, plus world-space failure/repair sequences and full-sized best/worst examples. This is visual QC of actual triangle/MANO renders, not a metric-only pass.
- Encoded and decoded 15 full-length H.264 comparison videos (2,250 frames), including green raw-SAM2 outlines. All have 150 frames. World views are independently framed; pixel alignment is judged in the shared RGB view.
- Milk/soup changes are generally small. BBQ’s tilted and upright body alignment is visibly better over many frames; some heavily occluded/misaligned views remain poor.
- Mug body alignment is only modestly better; handle/orientation disagreement persists. Spatula’s handle/head still have widespread disagreement; 93/150 frames remain below raw IoU .5 at 192 px. The full-mesh requirement is preserved, not replaced by silhouettes or primitives.
- Comparing final repair with RGB bootstrap still shows many earlier registration regressions. This experiment guards **output repair relative to the reproduced final seed**, not every internal seed-stage RGB/PnP/scale update. The original end-to-end mask-constrained shared-scale problem remains open.

## Next causal work

The fixed-domain contract prevents output refinements from spending mask support and exposes rotation/shape ambiguity. The remaining hard blocker is a trustworthy motion/rotation likelihood not inherited from the same generated mesh bindings. A next artifact-producing experiment should couple source RGB motion and fixed observed surfaces across frames, retain multiple orientation hypotheses under occlusion, and evaluate whether poor silhouettes arise from orientation or the unchanged generated shape. It must keep this output reference and complete mesh/MANO videos; do not simply tighten scalar gates or declare IoU to be 3D accuracy.

## Tests and preservation

- 221 regression tests: 217 passed, 4 skipped, in the existing ego_skrl environment. No packages installed. Numerical runs used the existing ego_annotation interpreter, CPU-only, one thread per process.
- The ego_annotation environment alone lacks `gymnasium` for one broad environment test; the initial full-suite attempt there is retained, not reported as a passing suite.
- Source receipts, immutable snapshot hashes, and old/new prediction hashes checked. Source meshes, MANO, depth, observations, S3 and RL/reward code unchanged. Original dirty worktree untouched. No merge/push.
- R1/R2 infrastructure failures, R3 regressions, R4/R5 independent-fit failures and R4 one-pixel report discrepancy remain preserved. R4 480-pixel audit timed out after two completed cases; it is not a five-case result. R6 320-pixel audit completed all five.

## Artifacts

- Prediction root: `/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_mask_alignment_20260929_r6`
- Snapshot: `/mnt/user-home/kupingxin/ego_annotation_runtime/monocular_mask_alignment_20260929_r6`
- Browser review: `/mnt/user-home/kupingxin/ego_annotation_experiments/observation_reliability_20260928/mask_alignment_repair_r6/review_192/index.html`
- Actual mask/motion evidence: `/mnt/user-home/kupingxin/ego_annotation_experiments/observation_reliability_20260928/mask_alignment_repair_r6/review_192/summary.json`
- Higher-resolution audit: `/mnt/user-home/kupingxin/ego_annotation_experiments/observation_reliability_20260928/mask_alignment_repair_r6/audit_320/summary.json`
- Full mesh/MANO failure → repair: `/mnt/user-home/kupingxin/ego_annotation_experiments/observation_reliability_20260928/mask_alignment_repair_r6/review_192/rotation_failure_and_repair.jpg`

Reproduce predictions with `python -m preprocessing.mask_repair --source-observations SOURCE --output-dir FRESH_OUTPUT --registration-width 192 --max-nfev 24`. Runtime metadata lists no previous prediction or evaluation file as an input.
