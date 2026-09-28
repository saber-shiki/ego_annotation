# Monocular coarse preprocessing: implementation contract

Date: 2026-09-27. Base 790893c. Independent branch `feature/monocular-coarse-preprocess-20260927`.

## Scope and causal objective

Input remains one monocular RGB video and its frozen prediction-side v2 observations. Never consume CAD, GT poses/depth, extra cameras, old optimized trajectories, or the evaluation audit directory. Do not change the raw observation producer, source bundle, S3 packages, hand decoder, RL training or rewards. New derived coarse state is explicitly optimized and is not relabeled source evidence.

The hardest actionable defect is missing rigid rotation and accumulated translation drift: disconnected RGB components fall back to identity, and source-depth errors enter chained translations. This implementation couples actual native mesh surface points to tracked RGB pixels and visible silhouette constraints. It does not merely smooth the old SE3 chain.

## Mechanisms and order

1. Load hash-bound annotations, camera/K, complete visual mesh, source MANO, appearance masks and RGB evidence. Use native appearance-camera geometry centered at one fixed mesh center as the object coordinate system. Preserve a fixed transform back to the original appearance camera. Produce all input frame IDs/timestamps without filling observations.
2. Add prediction-side masked forward/backward LK tracks to bridge existing RGB gaps. Bind valid track pixels to actual native triangles at the current candidate; PnP provides rigid-pose proposals. Retain keyframe/model correspondences rather than replacing missing rotation with I. Existing pairwise RGB rotations are auxiliary uncertain factors; their noisy metric translations do not get chained into the final state.
3. Joint sparse robust pose refinement: tracked-point reprojection, multi-keyframe source/target correspondences, occlusion-aware visible contour correspondences, retained front-depth support, separate RGB rotation factors and temporal correction regularity. Candidate transforms remain SO(3)/SE(3). Frames without usable measurements retain a temporally inferred state with explicit provenance, never promoted to measured.
4. Active R2: estimate *relative* per-frame depth scale from rigid material-point chord lengths in existing RGB/P09 correspondences, with one fixed source reference gauge and smoothness. Remove R1's free additive-bias variables from production passes. Add observed-to-actual-triangle 3D registration so missed image rays do not remove all depth support. Absolute scale stays inherited from the source prediction; this is internal consistency, not independent metric calibration. The causal amendment and failed R1 are preserved in `MONOCULAR_COARSE_R1_CAUSAL_REPAIR.md`.
5. Fit one bounded, shared, volume-preserving low-rank shape correction to multi-frame image/depth evidence. Preserve native topology and strong shape prior. This implements a limited shared-geometry refinement, not a claimed recovery of missing topology or unseen physical surfaces. No per-frame shape/scale. Contact/collision authority stays false.
6. Export `coarse_scene.json`, `coarse_scene.npz`, immutable factor/solver diagnostics and full-video before/after mesh+MANO overlays. Keep source, bootstrap and refined states separately. Neither quality flags nor temporal interpolation substitute for mesh reconstruction.

## Numerical contract

- meters and radians; pinhole rays use exact half-pixel resized intrinsics.
- world-from-camera and world-from-object are explicit; rendering uses their composition, not a camera/world AABB substitution.
- robust residuals are scaled in pixels, meters and radians separately. No absolute 45-degree limit on real object motion.
- absolute scale is fixed to the source visual mesh gauge; `absolute_metric_accuracy_certified=false`.
- outer iterations re-query actual triangle correspondences. Within each least-squares pass the correspondences are fixed and Jacobian sparsity reflects real dependencies.
- all accepted updates are evaluated by prediction-side residuals. Never select configurations or frames using GT.
- when feature support fails, use temporal/candidate inference with uncertainty, not zero geometry or a fabricated observation.

## Validation and discriminating outcomes

Synthetic tests must distinguish true rotation from identity, show pixel factors move a displaced object, preserve a single global gauge, bridge missing observations without changing their validity, preserve mesh/hand topology, reject input/hash/timeline leakage, and verify sparse residual/Jacobian dimensions.

Frozen five-case development runs use one shared configuration, fresh run roots, CPU numerical kernels on the A800 server and no learned-model calls. Freeze code before each batch. Output all 150 frames. First inspect source-side diagnostics and complete mesh/MANO videos. Only after prediction outputs are frozen, an external evaluator may read GT and report improvements/regressions; evaluation must not write into the candidate.

Predictions: if source-depth translation chaining dominates, model/image factors should reduce image misalignment and drift while recovering non-identity rotations. If depth bias is global/unobservable, image consistency may improve without absolute metric improvement; retain this limit. If native shape is wrong, shared low-rank refinement may be insufficient; report the remaining geometric failure rather than enlarge contact tolerances.

## Resource and compatibility boundary

No local workstation GPU inference; the current compute host is the A800 server. Restrict CPU numerical threads and iterations. No new dependency installation. A long-running batch gets its own job window inside the existing `ego_annotation` tmux session, never adopts or injects commands into another pane. Existing source/dataset training guards remain intact. New CLI is separate (`ego-preprocess` / `python -m preprocessing`).
