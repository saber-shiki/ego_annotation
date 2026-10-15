# Monocular derived coarse scenes

## Explicit surface-motion/envelope experiment (2026-10-07)

```bash
# Server only; use its provisioned interpreter and an immutable snapshot.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /ABS/PROVISIONED/PYTHON -m preprocessing.surface_ablation \
  --source-observations /ABS/source_observations.json \
  --baseline /ABS/FROZEN_R6/coarse_scene.json --output-dir /ABS/NEW_OUTPUT
```

This runs A (memoized R6), B (mesh-independent RGBD material motion plus observed surface/
boundary factors), and C (B plus a soft open envelope). Run from an immutable
snapshot using the provisioned server CPU interpreter; the output must be new.
It does not run perception or PPO and does not change the default producer.
Full mesh, source MANO, camera/K and the R6 shared scale remain frozen.

The five-case study produced local improvements but persistent physical errors
and new motion regressions, particularly on spatula. **Do not promote B/C to the
RL/default baseline.** See [design](../docs/VISIBLE_ENVELOPE_ABLATION_20261007.md)
and [complete results/videos](../docs/VISIBLE_ENVELOPE_RESULTS_20261007.md).

This is a separate preprocessing consumer of frozen **v2 monocular-RGB observations**. It does not run observation models, use CAD/GT/additional cameras, change source MANO, load a policy, or bypass the old training guards.

## Fixed-depth surface alignment profile

The new user-directed operation is available as `--profile visible-surface`: keep raw UniDepth/P09 surfaces fixed and fit the generated mesh's shared uniform scale and independent translations. It does not unify MoGe, rescale depth, or use volume-preserving deformation. See [VISIBLE_SURFACE.md](VISIBLE_SURFACE.md) for its specific command, contracts and outputs. The legacy profile below remains available for controlled comparison.

## Run

Use a provisioned environment with the project numerical dependencies, Open3D and ffmpeg. No installation is performed by this command.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /path/to/python -m preprocessing \
  --source-observations /absolute/original/run/source_observations.json \
  --output-dir /absolute/new/coarse_scene_run
```

Installed entry point: `ego-preprocess`. The output directory must be new and outside the immutable observation run. Source receipts contain absolute paths: use the original frozen run rather than rewriting them to a downloaded partial evaluation copy.

Defaults: 480-pixel tracking plane, 128-pixel contour plane, up to 18 optimizer evaluations per pass, 320-pixel review videos. `--no-render` is useful for bounded numerical regression only; a complete reviewed delivery needs full-video rendering.

## Implemented mechanisms

1. Actual native-triangle ray intersections bind persistent, forward/backward checked masked LK tracks to one object frame. The R2 robust local PnP tracker is preserved, not replaced by greedy multi-solver selection. Missing observations remain missing in source validity.
2. **Disabled by default:** `--experimental-geometry-relocalization` exposes the failed-validation low-texture experiment. It scores 24 proper mesh-axis rotations with actual triangle silhouettes and a temporal path, but the five-case evaluation showed that better silhouettes can select a worse physical orientation. It must not be described as an accepted precision fix. When explicitly enabled, changed poses rebuild stale bindings as weak visual support, not material tracks.
3. A temporally diverse RootSIFT keyframe bank supplies reciprocal long-range image matches. Pose candidates use robust training support and are all judged on the same held-out pixel identities. Accepted frames replace their stale local factor rows and add actual source/target endpoint factors. The failed R3–R6 greedy scoring and 25-degree/60-mm correction rule are not used.
4. A sparse robust joint SE3 objective combines persistent image tracks, existing multi-frame RGB source/target correspondences, separate RGB rotation factors, visible contour correspondences and temporal regularity. Existing noisy RGB translations are not chained into the result.
5. The inherited R2 relative-depth stage estimates per-frame **relative scale** from rigid chord lengths of matched material points in eligible source depth. It anchors one scale and smooths weak support. R1's free additive-depth-bias mechanism was removed from production passes because it could explain away displaced geometry.
6. Front-depth and trimmed observed-to-actual-triangle registration constrain translation even when projected rays miss the current model. They are uncertain monocular measurements, not GT or signed collision constraints.
7. One bounded, volume-preserving, low-rank shared shape correction is optimized across the clip. Native topology remains unchanged. This is not reconstruction of missing holes/hidden topology, and it cannot certify physically accurate shape.

## Outputs

- `coarse_scene.json`: derived-state manifest, source/code hashes, scale/gauge and authority boundaries.
- `coarse_scene.npz`: complete mesh, world object rotations/translations, camera/K, frame IDs/times, untouched source MANO, original source validity, relative depth scales and shared shape parameters.
- `bootstrap.npz`, `relocalized.npz`, `tracking.json`, `shape_relocalization.json`, `relocalization.json`, `solver.json`, `relative_depth_calibration.json`, `image_diagnostics.json`: mechanism traces and source-side measurements.
- The default enables only common-held-out descriptor relocalization. `--no-relocalization` runs the local-tracker-only ablation; `--experimental-geometry-relocalization` opts into the failed geometry experiment. These two flags are mutually incompatible. No mode loads historical candidate outputs.
- `renders/before_after_overlay.mp4`: complete original timeline, native object and MANO triangles before/after.
- `renders/coarse_world.mp4`: complete fixed-world view of actual geometry.
- `completed.json` or `failed.json`: terminal outcome. A terminal-success record is not an accuracy certificate.

For row `i`, geometry is `vertices_canonical @ object_rotation[i].T + object_translation_world[i]`. Arrays use meters/radians and world-from-object/world-from-camera semantics. `relative_depth_bias_m` is only the equivalent median-Z correction diagnostic derived from the estimated scale; it is **not** a free fitted bias in R2.

## Limits that callers must preserve

`contains_optimized_state=true`; `absolute_metric_accuracy_certified=false`; `signed_geometry_ready=false`. Absolute metric scale remains tied to monocular source predictions. Image consistency does not imply centimeter or millimeter physical accuracy. Do not feed this new manifest to the old v1/P13 actor by deleting its guards. A separately reviewed coarse-state RL adapter is still required.

Evaluation is outside this package and only reads GT after outputs are frozen. The evaluation directory is not a valid producer input. Five development cases are not evidence of held-out generalization.

## Tests

```bash
python -m unittest -v preprocessing.test_coarse preprocessing.test_relocalization preprocessing.test_shape_relocalization
python -m unittest test_pipeline observations.test_observations \
  observations.test_manifest rewards.test_terms envs.test_qualification \
  legacy_optimize.test_legacy_optimize preprocessing.test_coarse \
  preprocessing.test_relocalization preprocessing.test_shape_relocalization
```

The full grouped suite needs the pre-existing RL/Gym environment even though this preprocessing command does not use RL. Tests cover actual point/triangle queries, coordinate transforms, non-identity PnP, pixel-driven translation/rotation recovery, sparse Jacobian dependencies, relative scale from rigid tracks, 3D re-anchoring and input/hash protection.
