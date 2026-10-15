# Mask repair: failed first implementation and corrected experiment

Updated September 29, 2026. This is an experiment, **not a default-pipeline replacement**.

## Actual first-attempt outcome (`monocular_mask_alignment_20260928_r3`)

All five 150-frame predictions completed. The evaluator decoded every frame of the
old/new full triangle/MANO overlay and world videos (3,000 decoded frames total).
At a common 192-pixel scoring width, raw IoU medians were:

| case | jump_r2 | r3 |
|---|---:|---:|
| milk | .842 | .852 |
| soup | .857 | .852 |
| BBQ | .653 | .590 |
| mug | .599 | .618 |
| spatula | .440 | .202 |

The spatula timeline renders show the corresponding large silhouette displacement.
R3 is rejected. Its immutable run/snapshot and the earlier failed r1/r2 attempts
are retained. R1 failed code-identity checking after an in-run snapshot update;
r2 failed import preflight (`data` omitted). Neither is evidence of a valid batch.

## Concrete defects

1. Comparing each step with the previous step spends another .015 tolerance each
   iteration. Shared-scale keyframes lost as much as .064 IoU within a single fit.
2. `t += ta[a] - t[a]` propagates an anchor translation to every frame without a
   mask check. A subsequent timeline guard cannot repair the already damaged seed.
3. Constraining the recursive contour proposal changes later LK mesh bindings.
   R3 therefore did not isolate output non-regression from tracker perturbation.
4. The previous attempt changed the default producer and weakened the existing
   conflicting-mask/depth test by replacing the conflicting mask with a consistent
   one. Both edits have been restored exactly to committed behavior/tests.
5. The unfinished evaluator used a *scaled final mesh* with bootstrap translations.
   The corrected evaluator uses original source vertices for both bootstraps.

## Corrective intervention and discriminating predictions

**Named blocker:** post-registration silhouettes are visibly wrong, while changing
recursive tracking sacrifices previously useful rotations and jump repairs.

The new `python -m preprocessing.mask_repair` entry is opt-in. It reproduces the
unchanged fixed-UniDepth producer from source observations in a fresh `seed/`
subdirectory, then refines output poses only. It never reads a previous run,
category condition, frame exception, GT, CAD, or evaluation targets as inputs.

- Real full triangle mesh and metric MANO are preserved.
- The one seed-derived uniform scale is frozen; no per-frame scaling or depth rewrite.
- Same-frame rotation and translation are optimized jointly with bidirectional
  physical contours and source first-surface point-to-plane terms.
- Missing-depth frames retain the seed camera-depth value; a silhouette cannot
  supply a fake metric-depth observation.
- Source raw mask, ownership and MANO define a fixed domain. Hand-only regions can
  contain amodal mesh. No candidate depth changes the comparison population.
- Raw positive recall, visible/owned IoU/precision/recall, boundary precision/recall,
  spill, and allowed-occlusion occupancy are recorded. Boundary tolerance is 2 px
  at the declared scoring width, and artificial exclusion-boundary contours are
  removed. Background occluders without source ownership evidence are not invented.
- Every output update must pass against BOTH the unchanged seed and the current
  candidate. Visible IoU may not drop below the seed at all; .015 overlap/recall/
  precision tolerances and .03 boundary tolerances cannot accumulate below the seed.
- Source surface samples stay fixed. Mean surface distance may not worsen by more
  than 3 mm, p90 by more than 5 mm, or first-hit coverage by more than .03. These are
  experimental guards, not calibrated measurement covariance or accuracy claims.
- No absolute orientation or angular-velocity clamp. Soft local priors retain
  source evidence rather than forbidding large supported rotations.

If mask error is dominated by local R/t misalignment, the actual rendered boundary
should improve without sacrificing depth support. If only translations move, or
boundary updates stall, the result identifies rotational ambiguity/shape mismatch
or conflicting depth as the remaining problem. If pose jitter increases, independent
frame fitting needs a temporally coupled correction objective; mask scores alone
cannot certify the repair. Milk f143/f148/f149 are evaluation checks only.

## Scope limitation (intentional, not hidden)

This corrected experiment constrains the **output repair**, not the internal seed
proposal generator. It is NOT evidence that every original RGB/PnP/shared-scale
iteration is mask-safe. Shared scale stays fixed during repair. An end-to-end
constrained scale solve remains future work. The source mask is not amodal, and
silhouette agreement does not establish 3D rotation, contact, or hidden shape.

## Reproduction and evidence

Use an immutable curated snapshot with the baseline runtime dependencies (`data`,
`observations`, etc.) plus current preprocessing files. Preflight imports and the
new/original tests before launch. CPU: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`, interpreter
`/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`.

Prediction command:

```
python -m preprocessing.mask_repair --source-observations SOURCE --output-dir FRESH_OUTPUT
```

Evaluation: `tools/preprocessing_analysis/evaluate_mask_repair.py` reads frozen
old/new outputs, checks source/snapshot/output hashes, evaluates stage-correct
meshes, decodes all videos, and writes old/new full-panel comparison videos plus
contiguous timeline sheets into a fresh review directory. No predictions are
modified by evaluation. Corrected-experiment results must be appended only after
running and inspecting the actual artifacts.

## R4/R5 causal update: silhouette success is not motion success

R4 output-mask medians improved on all five cases (192 px: milk .860, soup
.870, BBQ .754, mug .620, spatula .466). Exported poses satisfy the frozen mask
contract. One spatula frame (117) differs from its camera-local solver report by
one grazing raster pixel; this is preserved in evaluation evidence. R5 verifies
and scores the exact exported world pose, retaining exact same-frame seeds for
unchanged outputs. No numerical tolerance was widened.

However, independent per-frame rotation refinement created symmetry/ambiguity
flips: BBQ f40 changed from a 22.49° inter-frame rotation to 113.86°, and spatula
f137 from 4.27° to 83.81°. These are evaluation observations, never special cases
in the solver. R4 and its export-only R5 successor are NOT promoted.

**R6 intervention:** add neighbor factors on the *correction* relative to the
unchanged RGB/depth seed. Let `D_i = R_i R_seed_i^T` and
`d_i = t_i - t_seed_i` in the world frame. Gaussian edges penalize
`log(D_i D_j^T)` and `d_i-d_j`, weighted by timestamp spacing. They do not penalize
absolute angle or impose an angular-velocity limit. A coherent 120° correction
has zero rotational edge cost; an isolated 90° symmetry flip does not.

Observation populations are normalized, so increasing the contour sample count
cannot silently overwhelm the temporal factors. Their Gaussian loss does not
saturate as a robust outlier. Forward/reverse coordinate sweeps accept a proposal
only when its image merit gain exceeds its added neighbor correction energy,
while retaining every frozen mask and surface guard. The seed reference is not
reset at the second sweep. The final world pose is rechecked before serialization.

Prediction: isolated new symmetry flips should disappear without losing the
previous mask improvements; real supported seed rotations remain unconstrained.
If the correction prior suppresses useful alignment broadly, that is a concrete
measurement/temporal conflict requiring a richer motion likelihood—not grounds
for claiming success from IoU alone. Inspect full videos and per-frame motion.
