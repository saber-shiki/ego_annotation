# Milk full-action reward ablation screen — 2026-10-12

## Question and discriminating outcomes

The operator requests ablations to identify which reward families influence which training outcomes. Existing full 118D PPO with the trusted SAM3D collision hypothesis has large hand projection regressions and modest/nonmonotonic collision changes. This study is not another reward/PPO optimization attempt: remove exactly one family per variant, hold all other weights, reward scale, action decoder and PPO hyperparameters fixed, and evaluate every candidate against the SAME unablated full objective and geometric metrics.

If removing a family changes its intended physical outcome consistently in paired seeds, that is evidence of its influence in this bounded configuration. If a zero/inactive family has identical training/parameters/outcomes, it is a negative control, not proof the family is generally useless. If effects reverse across seeds or unrelated outputs also change, report instability/shared-network/return-credit effects rather than claiming a clean semantic causal decomposition. A lower ablated training J alone is never success.

## Fixed experiment

- Same immutable R6 native SAM3D object (239430 vertices/478880 triangles), camera/world scale, source v2 measurements and metric MANO; no GT/CAD, perception rerun, old learned policy or proxy geometry.
- Both hands unfrozen. Same standard118D action space,551D observations, masks, MANO betas/object shape fixed, all Gaussian heads trainable.
- Same original full-action PPO: initial log_std=-1, rollout128, epochs/minibatches8/8, lr3e-4, clip.2, gradient clip.5, gamma1, lambda.95, reward scale1e-5, no KL early stop. Keep the generated collision v1 positive-count reduction unchanged; do not fix it during this ablation screen.
- **13 variants × seeds0/1 =26 fresh policies**, each256 training steps (two PPO updates). One independent deterministic complete150-frame evaluation at256. This is a short screening study, not convergence/generalization, and two seeds do not support significance claims.
- Four one-thread CPU worker processes on the current A800 server, each with fresh output subdirectory and no source writes. 12-minute timeout per job; controller1-hour hard cap. Estimate30–45min including full evaluations and native/MANO rendering. No automatic budget increase or survivor-only omission.
- Common fresh zero replay/audit and source hashes; no old RL zero artifact is a training input. Each run resets its own seeds and environment. Verify matching initial policy parameters across variants within each seed.

## Single-family removals

1. `hand_pixel`: joint_pixel_deadband (2D projected-joint preservation).
2. `hand_joint_depth`: joint_depth_deadband (camera-Z source joint-depth preservation).
3. `hand_surface_depth`: visible_surface_depth_order (independent P09 first-surface order).
4. `hand_priors`: root/articulation/translation priors, plus optional hand-ray prior if active.
5. `hand_temporal`: hand correction velocity/acceleration.
6. `hand_bounds`: hand root/articulation/translation soft magnitude bounds.
7. `object_temporal`: object rotation/translation correction velocity/acceleration.
8. `object_geometry`: observed point-to-plane/point-to-point geometry factors.
9. `object_rgb_rotation`: qualified source-world RGB rotation factors (translation edges disabled in current adapter).
10. `object_rgb_reprojection`: qualified RGB reprojection factors.
11. `contact`: contact normal proximity, probability prior/likelihood/continuity as one interaction-state family.
12. `generated_collision`: the new full-native-mesh inside-depth hypothesis.

`full` removes nothing. Optional formal object sigma/translation-target, measured signed barriers and alternative visual-profile families are currently inactive: enumerate separately, not artificially activate them or claim trained effects. Classify factor names strictly; unknown active factor names fail instead of silently escaping accounting.

## Accounting/metrics

Compute all full factors on exactly the same candidate, then select the training subset. That preserves candidate caching, original geometric reductions and observations. Training return telescopes to `-scale*J_train(final)`, while each evaluation independently reports `J_full`, original formal J, collision J, dropped cost and every grouped cost. Keep full raw evaluation logs and compatibility-bound256 checkpoint/NPZ/JSON.

Measure all150 frames: generated-mesh detection frames, inside vertex/centroid counts, maximum inside depth, summed squared inside residual and actual hand-edge/object-triangle crossings; joint pixel/depth shift median/p95/max versus fresh zero; residual hand/object translation/rotation change; residual velocity/acceleration; source contact-prior deviation; action-block magnitudes; official update count and finite parameter changes. Compute identical-raster object silhouette IoU from complete triangles, not GT/amodal.

Deliver full-native/MANO zero/seed0/seed1 RGB/fixed-world videos for each variant and a cross-variant review. Selected comparisons can be exported as RRD with actual penetration red vertices/edges. Preserve every failed run/timeout. Summarize paired seed deltas, report signs that agree/disagree, no overinterpretation of cross-metric tradeoffs. Original observations and prior frozen experiments remain unchanged; no default promotion, merge or push.
