# V20 material RGB-D repair — causal patch, not a new pipeline version

Base: f22876a9e2e1c792e3941b31f7ee3a20ffecbaf1. Separate worktree. Preserve v16 archive and all formal states. No GT in implementation experiments, evidence selection, or candidate generation.

## Defect being addressed

Frozen same-ray diagnosis shows f149 generated first-hit depth -44.16mm vs P09, even though nearest-mesh distance is ~8mm. Existing NN factors associate newly visible surface patches with a different anchor patch. Earlier pixel experiments are invalid: target pose has zero Jacobian in the source-to-target pixel residual. Low-conditioning "translation-only" PnP terms also still have rotation Jacobians.

## Correct observation equations

Column notation T_i = world_from_object at time i. Source and target observations are fixed world points X_s/X_t, not immutable-pose canonical landmarks.

- q_s(T_s) = inverse(T_s) X_s
- Xhat_t(T_s,T_t) = T_t q_s(T_s)
- Forward image residual = project(inverse(C_t) Xhat_t) - u_t.
- Reverse residual, where target depth is eligible: project(inverse(C_s) T_s inverse(T_t) X_t) - u_s.
- Metric residual where both endpoints have eligible depth: inverse(T_t) X_t - inverse(T_s) X_s, equivalently forward transported 3D disagreement. Fixed measurements must not be recomputed from a candidate pose.

Implementation will use per-match metric XYZ error in target-camera coordinates, plus the forward pixel residual (and image-only rows when target depth is absent). XYZ and UV are correlated; pair-normalized conservative weights, not a claim of independent sensor likelihood. Do not count the compressed PnP SE(3) residual again for pairs whose raw material factors are active.

For a deliberately translation-only PnP edge, freeze rotation in its residual. Use a fixed source support point c_s (mean of valid source P09 points): measured displacement d = E_measured(c_s)-c_s and predicted displacement = t_t-t_s+(R_t0 R_s0^T-I)(c_s-t_s). R_i0 is fixed to this run's input orientations. This expression has exactly zero derivative w.r.t. all candidate rotations and is invariant to a common translation of the world origin. It is a fixed-orientation conditional constraint, not a full uncertain SE(3) likelihood. When no support point is available, do not silently trust the PnP translation.

## Measurement/provenance contract

Reuse the already frozen SuperPoint/LightGlue matching evidence (no new model inference required). Validate annotation, source depth and source initial-pose hashes in the edge NPZ and JSON. Recover the *fixed* source world observation from the legacy canonical serialization using its named initial pose only once. Extract target depth from the named prediction-side depth camera contract at the matched target pixel. Enforce object-owned masks, explicit pixel planes and camera roundtrip. UniDepth confidence is predicted error (larger is worse); never weight it as precision directly.

Do not infer metric eligibility from merely positive depth. f143 has no accepted P09: allow 2D target matches and temporal state, but do not add source or target metric depth from that frame. Record observation masks, uncertainty and missing-depth reasons. Use actual RGB material matches, not shape nearest-neighbors. Validate non-collinearity/coverage and forward/reverse material residuals; planar point covariance alone is not a reason to zero every 3D–3D rotation direction.

## Scope and ablation plan (fixed before runtime)

Window 110–149, anchor120; outside-window poses unchanged. Full SE(3), additive translation. Generated mesh never enters the pose residual. No small cumulative rotation cap. Preserve existing renderer/mesh hash contract.

A. Regression: old reprojection has zero target Jacobian; corrected residual has nonzero source+target Jacobians and zero residual for a nontrivial true relative motion. Translation-only rotation Jacobian must be zero, including under global-frame rebasing.
B. Corrected pixel-only control: same cached measurements and predefined controls; all other settings fixed. It tests whether the earlier failure was a miswired measurement.
C. Material RGB-D graph: use material correspondences in place of anchor/adjacent NN and disable duplicate PnP rows. Fixed conservative metric/UV scales, node and temporal priors recorded. Not an arbitrary parameter sweep.

Outcomes: target pixel and metric agreement improve together => real measurement-to-state coupling; pixel improves but metric worsens => depth/pose/shape conflict remains, do not chase silhouette by moving Z. Material errors improve but generated first hit remains wrong => isolate shared generated geometry mismatch for pose-frozen shape stage. If correspondence graph lacks coverage, retain uncertainty/initial base, do not claim the graph recovered rotation.

Run prediction first without generated acceptance. Freeze each numerical result before post-solve same-ray generated validation. Improvement-only safety checks and final quality diagnosis are separate: an absolute -5mm target must not reject every intermediate step and freeze a poor initial state. Do not relax prior validation thresholds to claim success.

## Deliverables

Regression tests, hash-bound material NPZ+JSON, frozen controlled pose reports, same-ray common-hit + coverage/tail comparison, representative geometry/UV overlays. If a sane candidate changes the delivered annotation, fixed-pose shape adaptation and full 150-frame MP4/RRD. Report uncorrected portions honestly; do not replace actual geometry with labels. No push/merge without owner approval.
