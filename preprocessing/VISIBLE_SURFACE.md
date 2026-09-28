# Fixed-UniDepth visible-surface alignment

Use this profile for the user-selected contract: **trust the eligible, de-occluded UniDepth/P09 visible surface as the reference; adapt the SAM3D mesh, not the depth.** SAM3D/MoGe generation and source observations are not rerun or edited.

```bash
cd /mnt/user-home/kupingxin/ego_annotation_preprocess_monocular
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/provisioned/python -m preprocessing \
  --profile visible-surface \
  --registration-width 192 --scale-keyframes 5 --max-nfev 24 \
  --source-observations /ABS/FROZEN_RUN/source_observations.json \
  --output-dir /ABS/NEW_OUTPUT
```

The profile is explicit; `--profile legacy` retains the previous preprocessing route for comparison. Neither route replaces the raw observation producer or enables RL training.

## What is fitted

`p_camera = R_camera_from_object @ (s * v_centered_native) + t_camera`

- One shared positive uniform object scale `s`, with translations separate from scale. Metric-observed frames have independent XYZ variables; in a metric gap, XY remains free and Z uses an inferred neighboring world-correction bridge rather than an independent depth reset.
- Native topology and proportions preserved; no per-axis stretching or per-frame size.
- Actual first-hit triangles and currently visible mesh samples, not projected-vertex z buffers or full-model/partial-cloud bounding-box ratios.
- Point-to-plane residuals for ray-hit surface correspondences; tangential point-to-point terms would incorrectly lock translation on planar surfaces. Missed rays retain visible-surface/coverage constraints.
- De-occlusion-aware contour constraints and coverage over a fixed observed population.
- Original UniDepth and eligible camera-space points are immutable. No depth bias, relative-depth rescaling, CAD, GT or extra camera inputs.

The v2 source mesh already contains its old camera-origin scene scale. The manifest records that old scale and the NEW residual object scale separately. The combined scalar is a **length-scale record**; translations are refitted independently, not multiplied twice.

## Rotation and timeline

The original-template RGB tracker first supplies a rotation proposal. The 2026-09-28 patch preserves that serial proposal/rebinding state, but repairs an output contour overwrite when both fixed current-frame tracking pixels and a fresh fixed-domain whole-mask projection contradict it. Mask-improving updates are not blocked solely for disagreeing with stale 3D bindings. Unfiltered proposals and the evidence are saved in `tracking.json`; this is not a full orientation/ambiguity solution. See `../docs/POSE_JUMP_CAUSAL_REPAIR_20260928.md` and `../docs/POSE_JUMP_REPAIR_RESULTS_20260928.md` for the rejected first attempt and final five-case results.

The resulting rotations are frozen during scale/translation registration and saved as `alignment_initial_rotation`. Scaling the template does not trigger another rotation branch.

Anchor calibration uses the exact same appearance-anchor frame/camera/K as the generated mesh. Up to five eligible, temporally distributed frames refine a single shared scale; it is then frozen for full-timeline translation fitting. Unavailable metric frames remain unavailable as observations: `translation_depth_inferred` and `missing_metric_depth_inference` expose their inference. A timestamp-aware world-correction bridge preserves current RGB motion without a frame-local Z reset; endpoints use a one-sided correction. Soft actual-world acceleration terms discourage unsupported impulses without imposing a hard motion cap. Original UniDepth data is never interpolated or replaced.

## Evidence and outputs

- `anchor_alignment.json`: same-frame scale/translation fit and fixed-population checks.
- `shared_scale_alignment.json`: one scale across selected frames; no per-frame sizes.
- `timeline_surface_alignment.json`: raw-depth translation fit with frozen scale/rotation.
- `visible_surface_metrics.json`: matched before/after using the same RGB rotation proposal and raw source population.
- `coarse_scene.{json,npz}`: complete mesh and full timeline. `uniform_object_scale` and `shared_shape_matrix = s I`; unlike the legacy volume-preserving mode, determinant is `s^3`.
- `relative_depth_scale` is exactly one and `relative_depth_bias_m` exactly zero (inactive compatibility arrays).
- Before/after videos compare the pre-alignment RGB pose proposal with the aligned output. Source MANO is unchanged.

A lower fit residual means consistency with the chosen UniDepth reference, not independent physical GT accuracy. Hidden dimensions remain SAM3D priors; wrong shape/rotation cannot always be fixed by uniform scale. Generated surfaces do not acquire signed collision/contact authority.

## Tests

```bash
python -m unittest -v preprocessing.test_visible_surface preprocessing.test_pose_jump
```

Actual-triangle tests cover independent scale/translation recovery, partial occlusion without whole-object collapse, first-visible-surface rather than hidden-backface fitting, shared scale, unchanged observations and routing without legacy depth correction.
