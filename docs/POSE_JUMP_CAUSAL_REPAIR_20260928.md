# Pose jump causal repair — 2026-09-28

## Reproduced defect and evidence

Frozen input/output: milk `monocular_visible_surface_20260927_r2`. The original video/masks and camera are continuous near the end; the uniform mesh scale is one constant. Actual before/after video frames reproduce the mismatch, so this is not H.264, a final-frame index shift or a varying-scale renderer bug.

A read-only instrumented replay of the frozen tracker reproduces every stored R/t to <=1e-9:
- frame148: contour-only refinement adds31.42deg /35.86mm, turns the SAME PnP-point median reprojection1.29px into12.52px, and whole-mask IoU0.856 into0.574;
- frame149: it adds45.71deg /36.40mm, turning1.30px into18.57px and IoU0.843 into0.471, although its own selected contour residual decreases;
- similar corruption occurs at interior frames10 and13, so a last-frame patch is not a solution.

A second mechanism produces the143/144 jump: frame143 has no qualified P09 visible object surface, not a missing RGB image or necessarily missing raw UniDepth. The solver gives it an independent Z variable and strong image terms but weak temporal correction terms. Its estimated camera-Z moves to0.3988m between0.4386m and0.4360m, creating40.46/41.54mm world-position steps. Existing regularization smooths corrections rather than actual motion; large priors are also robustified by the global loss.

Evidence: `/mnt/user-home/kupingxin/ego_annotation_experiments/milk_jump_diagnosis_20260928/` (original frames, trace and numerical witness). GT is not needed for these causal comparisons and is not a prediction input.

## Generic mechanism changes

1. Local image refinement jointly consumes the current frame's accepted RGB model-point observations and contour evidence. Candidate updates are evaluated on the SAME point set and fresh whole-object projection on a fixed visibility domain. A contour objective alone may not override valid tracking. Rejected updates return the CURRENT frame's PnP estimate, not another frame's pose. No angle/frame-specific clamp.
2. In full-timeline registration, a frame lacking qualified metric support has no independently free depth-correction variable. Its latent camera-Z correction is obtained from neighboring metric-anchored world corrections using actual timestamps, while its own RGB proposal motion and independent lateral fit remain. Missing observations are still labeled missing. Leading/trailing gaps use a one-sided correction estimate, not a copied frame or manufactured depth.
3. Add a soft actual-world acceleration term (timestamp aware) in addition to correction continuity. Observation residuals remain robust; motion/priors remain quadratic so an unsupported impulse is not made cheap by robust-loss saturation. Raw depths, point clouds, global mesh scale and MANO remain protected.

## Discriminating regression criteria

- A low contour residual with bad RGB reprojection must not be accepted.
- A large real image-supported rotation remains allowed; do not cap absolute orientation or valid motion.
- One-frame and multi-frame metric gaps cannot reset the depth reference; actual RGB motion is not flattened.
- Endpoint gaps have the same generic inference semantics as other gaps, no `frame==149` or milk branch.
- Sparse Jacobian covers cross-frame dependencies introduced by latent Z.
- Fit/render every frame, retain original validity flags, and verify all five clips under one configuration.

Run changes only in the independent preprocessing worktree and new frozen output roots. Preserve old failures and source bytes. Any GT comparison follows completed prediction and cannot select parameters or frames in the producer.

## First full-batch result and causal amendment

The first guarded-tracker run (`monocular_visible_surface_20260928_jump_r1`) repaired milk but regressed BBQ visibly: its later bottle orientation was wrong and median fixed-domain IoU fell 0.733 -> 0.326. The failure is retained in the full raw/old/repaired comparison videos. Do not deploy that revision. Point binding is a model-dependent hypothesis, not infallible truth: preserving low point reprojection can lock a wrong shape/pose branch. Changing a local refinement also changes future seeds and makes this error recursive.

The amended implementation separates proposal generation from output repair. The original serial LK/PnP/contour proposal state and its replenishment are retained verbatim. Only an output contour overwrite contradicted by BOTH fixed current-frame tracking pixels AND fresh whole-visible-mask evidence is repaired with the joint step. A mask-improving contour update is not forbidden solely for disagreeing with stale 3D bindings. Repaired outputs never silently modify the legacy proposal-generation chain. This is not per-object selection, cached-result ingestion, an angle clamp or frame copying: every proposal is regenerated from source RGB with one path, and both conflicting observations and the unfiltered proposal are retained. There are no baseline/GT evaluator reads in the producer. The latent metric-gap depth and actual-motion factors remain the same.

Regression must now verify exact reproduction of the unfiltered original proposal timeline, plus no recurring feedback change caused by a repaired frame, before comparing the second complete five-case run.
