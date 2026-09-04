# Milk P14/P15 pose repair experiment

This directory contains an isolated prediction-side experiment for
`P0014_84ea2dcc_carton_milk_f2370_2519`.

## Boundary

- Formal input/output root: `/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_depth_order_corrected_fullrun_20260903T101508Z`.
- Formal P14/P15/D19 state was read-only; `FORMAL_STATE_INTEGRITY.json` records the final hashes.
- The clean formal source at commit `36e87ecdad656939ced4ac0b8c5ca4198848fd2e` was not modified.
- Experimental source is the separate worktree `/mnt/user-home/kupingxin/ego_annotation_worktrees/milk_pose_repair_experiment_20260904`, branch `experiment/milk-pose-repair-20260904`, commit `3cad519af05e75e10eda318addc24923c77a3087`.

## Changes

1. Make the calibration/depth (1408×1408) → RGB/SAM2 mask (960×960) → factor raster (256×256) contract explicit, including half-pixel affine validation.
2. Rasterize the complete projected MANO triangle silhouette as unknown support.
3. Exclude observed-to-rendered coverage factors whose observed pixel is in projected hand-unknown support.
4. Add per-first-hit ownership/confidence/boundary weights.
5. Require exact path/hash/frame-camera/K/object contracts for strict image factors.
6. Add an observed-only global P14 graph with anchor loop closures, local surface edges, all accepted adjacent RGB/PnP edges, RGB-vs-depth conflict weighting, and temporal correction priors.
7. Keep all generated SAM3D faces out of pose optimization.

## Results

See `P15_POSE_REPAIR_AB.md` and `P15_POSE_REPAIR_AB.json`.

The final global+image candidate is diagnostic-only. At 256-pixel evaluation it improves mask IoU median from `0.830797` (formal P15) to `0.837756` and mask centroid error median from `2.922 px` to `2.446 px`; image-factor RMS decreases by `0.0057697`, and observed-mesh median degradation is `-0.000318 m`. It is not copied into formal P15, P16–P18, D18, D19, or `SUITE_DONE.json`.

## Reproduction

```bash
bash experiments/sam3d_native_ghost_lite_20260826/run_milk_pose_repair_experiment.sh
```

The runner is idempotent for existing experiment outputs and writes only below
`experiments/pose_repair_20260904`.
