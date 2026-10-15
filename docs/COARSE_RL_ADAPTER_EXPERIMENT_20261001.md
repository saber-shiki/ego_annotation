# Explicit coarse-preprocessed milk → PPO experiment (2026-10-01)

## Causal question and scope

The user requests connecting the latest milk preprocessing result to the current RL training backend. The strict blocker is not GPU budget: the v2/coarse result has no explicit trainable dataset/state/observation contract. The existing v1 guards must remain unchanged. This patch adds a separately selected adapter, not a reclassification of optimized preprocessing as raw observations.

Latest input: `monocular_mask_alignment_20260929_r6/milk_observation_v2_20260921_r2/coarse_scene.json` and its SHA-bound NPZ; original frozen v2 bundle remains the measurement source. R6 is experimental output-mask repair, NOT a production promotion or physical accuracy certificate. No GT, CAD, audit output, old checkpoint, or prior RL candidate may be an input.

## State, action and reward contract

- A new versioned coarse-training index explicitly names the derived initializer and original v2 source.
- Preserve every frame, timestamp, camera/K, original source validity, qualified depth/P09 clouds, complete native triangles and both full-778 MANO hands. Source MANO parameters/betas remain original; the new object mesh/SE3 is the residual-action reference. Inferred poses remain latent initial state, never upgraded into metric observations.
- Keep the existing 118-action residual decoder and original Gaussian actor/value architecture. Append explicit v2 interaction/visibility/source-validity and causal RGB summaries plus candidate object residual history. This requires a fresh checkpoint contract; old 480-feature checkpoints must reject.
- Keep `formal_annotation` factor weights, reductions, reward_scale and PPO hyperparameters. Rebuild unsigned contact membership on the actual fixed coarse mesh against source MANO once; preserve source contact priors, uncertainty and first-surface evidence. Generated mesh cannot acquire signed authority.
- Wire independent source-world RGB rotation/reprojection factors into the existing declared-factor kernels, using the existing observation-side qualification weights and no generated canonical RGB labels. RGB factors use both current endpoint poses; no future policy state enters an observation/reward prefix.
- Source sigma/image/nonpenetration factors absent in the source remain absent. Fixed object scale/shape and cameras are not trainable actions. This is not a new global scale solver or full legacy-fit equivalence.

## Discriminating predictions

1. Zero action must reproduce the R6 complete object mesh/trajectory and original MANO within real replay tolerance. Failure means adapter/coordinates/replay are wrong; do not train.
2. Controlled physical-unit actions must change full MANO/object geometry and the corresponding named factors without changing source arrays. Prefix rewards must telescope to `-reward_scale*J(final)` at gamma=1. Failure means decoder/reward coupling is wrong; fix the mechanism.
3. A fresh 1024-step PPO run is compared with zero on exactly the same new objective. Decreased final objective plus sane complete-video geometry indicates usable learning signal on this one video only. Worse projection/depth/RGB terms identify where updates spend evidence; do not call finite checkpoints success or blindly extend runtime.
4. All-source observations and code are frozen before runtime. No perception rerun, package install, GT-side evaluation, or online numerical fitting occurs inside training.

## Runtime and review

Compute target is the current A800 server (`dexrobot-Standard-PC-Q35-ICH9-2009`), not a workstation. Existing `/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python`, CPU-only and one numerical thread. Bounded first pass: synthetic/mechanical tests, one complete zero rollout, small controlled-action checks, then 1024 training steps with deterministic complete-video evaluation at 256-step boundaries. Keep diagnostics/checkpoints/export receipts and full-length zero/best/final mesh/MANO overlays. Dataset split is `all` because this is same-video optimization, explicitly NOT held-out generalization.

Work in the separate `ego_annotation_rl_preprocessed_milk` worktree; leave the original dirty preprocessing branch and immutable prediction roots untouched. Install no packages. Preserve the original guard tests. This document records an adapter experiment within the current pipeline, not a new top-level pipeline version.
