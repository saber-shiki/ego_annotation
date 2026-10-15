# Visible-surface motion and envelope ablation — 2026-10-07

## Causal mechanism (declared before running)

The remaining artifact defect is incorrect complete-mesh orientation/placement in BBQ,
mug and spatula, not missing input masks. Old material points bound by an erroneous
mesh pose can validate that same pose. R6 output repair helps masks but does not supply
independent material-motion evidence. A loose bounding box alone previously had nearly
zero increment; a biased metric surface can make a tightly constrained box wrong.

Implement a source-observation motion layer: fixed masked RGB tracklets, local measured
surface lifting on their own camera rays, observation-derived proper rigid-motion
proposals, and sparse joint SE3/shared latent material-point fitting. Latent points are
motion nuisance variables, never a replacement for the full reconstructed mesh. Add
fixed observed normals and physical visible boundary constraints on the actual first
mesh surface. An open surface envelope is a weak, uncertainty-padded prior, not a
completed object, measured hidden dimension, collision surface or statistical interval.

## Frozen protocol

A: memoized, hash-verified complete 2026-09-29 R6 scenes for all five clips.
B: A plus observation-driven motion, surface normals and visible boundaries.
C: identical B plus soft envelope (lambda=0.15). Same candidate starts, samples,
tracklet split, numerical budgets, source ownership, camera/K, mesh topology, shared
uniform scale, source MANO and immutable source observations. No GT, CAD, new model,
new camera, depth correction, per-frame scale or learned policy in prediction.

The observation-only motion solve uses RGB pixels and source depth/normal support,
NOT mesh-ray intersection bindings. A supplies the one fixed canonical/gauge anchor
and initialization, not observed material coordinates. Depthless tracklet observations
remain 2D; unsupported frames retain inferred pose and explicit observation validity.
Overlapping tracklets and derived normals/envelopes share source errors; they are not
independent sensors. Whole tracklets are held out, with an explicit caveat that
neighboring feature redetection can repeat a physical point.

Numerical full-mesh factors re-evaluate first-hit visibility at outer iterations. All
candidate scores use fixed target samples/populations and fixed hand exclusion domains.
Missing ray hits retain a distance/coverage penalty, never silently disappear. Competing
trajectory starts (A, observation-motion, intermediate rotation correction) are compared
using the same score. Candidate disagreement is not calibrated covariance. Box-free and
box arms start independently from identical proposals. Missing surface frames have no
new envelope observation or box weight; existing world corrections handle depth gaps.

## Discriminating predictions

- If biased mesh bindings dominate, motion supported by raw RGBD material identity should
  differ from A and improve orientation and complete-mesh projections at difficult frames.
- If material support/metric depth is too weak, tracklet reprojection/depth conflict or
  insufficient edge coverage will remain. Do not call a smooth unsupported path recovery.
- If motion improves but mesh silhouette/surface still disagrees, fixed generated shape or
  initial canonical alignment dominates; do not absorb it with per-frame scale.
- If C and B are nearly identical and box residuals inactive, loose envelope is redundant.
- If C reduces gross escapes without hurting observed support and frozen-posthoc physical
  metrics, that supports a robustness benefit, not independent metric calibration.
- If predicted depth agreement improves while GT surface error worsens, shared depth bias
  dominates; increasing depth/box weight is not the next intervention.

## Execution, budgets and deliverables

Server: dexrobot-Standard-PC-Q35-ICH9-2009 (A800), provisioned ego_annotation Python;
CUDA_VISIBLE_DEVICES='', numerical/FFmpeg threads=1. No local workstation inference.
Separate development worktree; immutable curated code snapshot for execution, fresh NAS
output root. Parent launches numerical experiment commands, no raw-video runtime agent
or manually assembled predictions. No delegation was requested.

Budget: approximately 60–90 min code/tests, 15–30 min five-case numerical/render batch,
15–25 min review/evaluation; hard batch limit 45 min. This is an offline geometry study,
not a new top-level pipeline version or real-time default. Log exact runtime. Limit
tracklets/points/candidates uniformly; do not substitute boxes or lower-topology objects.

All five clips: 150 frames, 30 fps, complete mesh+source MANO A/B/C RGB overlays and common
fixed-world views, representative/timeline images, per-frame raw/owned mask/boundary,
first-hit depth/coverage, normal, envelope and motion evidence. Independently decode
videos. Freeze prediction hashes before any evaluation. Post-freeze evaluator applies
official QA; uses one A-derived object/world gauge shared across arms, reports all eligible
frames, actual-camera full-surface error and rotation/translation trajectory error.
Raw rotation is not symmetry-adjusted; gauge-aligned motion is not absolute initial pose.
No evaluation files/GT feedback in prediction snapshot, no per-case picking or tuning.
No default promotion unless real rendered/geometric results justify it.
