# Ordered shape / depth / MANO repair: branch guide

This branch preserves the frozen R6 mask-repair dependency and the subsequent
2026-10-15 opt-in repair study. It **does not replace the default producer or R6**,
change the RL actor contract, or certify absolute metric scale/contact/zero
penetration. Do not interpret this publication as promotion of the experimental
MANO state.

## Contents

- R6 dependency: `preprocessing.mask_repair`, fixed mask domains and regression tests.
- New worker: `python -m preprocessing.sequential_repair`.
- Shared complete-mesh deformation: `preprocessing/shared_repair.py`.
- Anchored relative depth curve: `preprocessing/relative_repair.py`.
- Full MANO FK with shared per-hand shape and image/motion evidence:
  `preprocessing/mano_repair.py`.
- Complete native-mesh/MANO rendering: `preprocessing/repair_render.py`.
- Post-freeze local analysis helpers: `tools/preprocessing_analysis/*sequential*`,
  `export_repaired_rrd.py`.

See [design](SHAPE_DEPTH_HAND_REPAIR_DESIGN_20261015.md) and
[actual results and failures](SHAPE_DEPTH_HAND_REPAIR_RESULTS_20261015.md).
The result report records the execution-time state. Its statements that no
commit/push had yet occurred describe that earlier prediction freeze; the owner
subsequently authorized publication to new personal and team branches. No
prediction, evaluator input, frozen runtime snapshot or existing artifact was
changed by publication.

## Provisioned environment and input contract

Use an already provisioned CPU interpreter with the project numerical/visual
requirements and ffmpeg; licensed MANO assets must already be present at the
hash-bound paths declared by the source observation bundle. No command below
installs packages or downloads data. Heavy inference is not part of this study.

`SOURCE` is the original immutable `ego.annotation.observation_bundle.v2` manifest,
not a relocated partial copy. `BASE` is the matching frozen R6 directory containing
`coarse_scene.json` and its hash-bound NPZ. Memoized observation/R6 stages must
remain available. This worker consumes their results; it is not a standalone raw
video perception replacement. Prediction must not receive reference annotation or
CAD paths. Use a curated immutable snapshot of the worker and its required
`preprocessing`, `data`, `observations`, and `utils` modules, not a mutable runtime
copy of this development worktree.

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PY=/path/to/provisioned/python
SOURCE=/absolute/original/source_observations.json
BASE=/absolute/frozen_R6_directory
OUT=/absolute/new_study_root
```

Each stage output directory must be new. Commands are sequential; inspect each
complete rendered result before advancing or promoting a state.

```bash
# 1. Shared shape, original raw depth; full timeline.
"$PY" -B -u -m preprocessing.sequential_repair \
  --source "$SOURCE" --baseline "$BASE" \
  --stage shape --output "$OUT/shape"

# 2a. Isolated relative-depth control: unchanged R6 shape and rotations.
"$PY" -B -u -m preprocessing.sequential_repair \
  --source "$SOURCE" --baseline "$BASE" \
  --stage depth_only --output "$OUT/depth_only"

# 2b. Ordered composition: retain preceding shared shape/rotations;
#     calibrate depth and refit timeline translations, not absolute dimensions.
"$PY" -B -u -m preprocessing.sequential_repair \
  --source "$SOURCE" --baseline "$BASE" --initial "$OUT/shape" \
  --stage shape_depth --output "$OUT/shape_depth"

# 3. Full MANO + object correction on the combined state.
"$PY" -B -u -m preprocessing.sequential_repair \
  --source "$SOURCE" --baseline "$BASE" --initial "$OUT/shape_depth" \
  --stage hands --output "$OUT/hands"
```

Every successful stage emits `coarse_scene.{json,npz}`, source/solver diagnostics,
`completed.json`, and complete `renders/before_after_{overlay,world}.mp4` videos.
Source raw depth, cameras, intrinsics and timing remain unchanged. Shared geometry
changes are actual full-mesh transforms; hand outputs are reproducible full MANO
FK with source validity/uncertainty, not skeleton or box proxies.

## Tests

From the repository root, with the provisioned environment:

```bash
"$PY" -B -m unittest preprocessing.test_shared_repair \
  preprocessing.test_relative_repair preprocessing.test_mask_repair \
  preprocessing.test_mask_export preprocessing.test_mask_temporal

"$PY" -B -m unittest test_pipeline observations.test_observations \
  observations.test_manifest rewards.test_terms envs.test_qualification \
  legacy_optimize.test_legacy_optimize
```

`preprocessing.smoke_mano_repair` is the retained original-host asset replay smoke;
it intentionally uses the recorded local source path. Synthetic tests need no
HOT3D reference assets. Do not treat passing tests as physical improvement.

## Evaluation / artifacts are not standalone portable tooling

The analysis scripts preserve the exact original-host study layout under
`/mnt/user-home/kupingxin` and `/mnt/truenas-user-home/kupingxin`. In particular,
some import helper code from the earlier external
`ego_annotation_experiments/milk_R6_vs_CAD_GT_20261014` and
`hot3d_s3_pose_audit_20260927` directories. Those external helpers and datasets are
**not included in this branch**. A fresh checkout can use the worker/tests with
its provisioned prediction inputs, but cannot reproduce the historical evaluation
or launch its RRD review scripts without those local dependencies. They are local
post-freeze analysis records, not a portable evaluator release.

Original rendered result directory:

```text
/mnt/user-home/kupingxin/ego_annotation_experiments/milk_preprocess_shape_depth_hands_20261015_r1/final_review/
```

Interactive RRD:

```text
/mnt/user-home/kupingxin/ego_annotation_experiments/milk_preprocess_shape_depth_hands_20261015_r1/final_review/baseline_repair_GT.rrd
```

Videos, RRDs, NPZ arrays, source images, licensed models, credentials and local
execution logs are deliberately not Git contents. Relative artifact links in the
historical results report refer to that local review directory, not GitHub-hosted
files. Known residuals (approximately +10% long transverse dimension, last-frame
retreat, and limited absolute hand-depth improvement) remain explicitly open.
