# Read-only observation consistency diagnostics

These are analysis tools, not a new prediction pipeline or runtime dependency.

`diagnose_observations.py` imports a specified **frozen prediction snapshot**, instruments its existing tracker without changing its outputs, checks exact equality against the saved bootstrap, and records source-side evidence. It writes only to a fresh external analysis directory. Source and frozen prediction inputs must remain outside the output directory. It reads no GT/CAD and runs no model inference. The original 20260928 run used the already provisioned `ego_annotation/.venv/bin/python`, CUDA disabled and one CPU thread. No installs are required.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python \
  tools/preprocessing_analysis/diagnose_observations.py \
  --run /mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_visible_surface_20260928_jump_r2 \
  --snapshot /mnt/user-home/kupingxin/ego_annotation_runtime/monocular_visible_surface_20260928_jump_r2 \
  --output /ABS/NEW_EXTERNAL_ANALYSIS_DIRECTORY
```

The original output is in `/mnt/user-home/kupingxin/ego_annotation_experiments/observation_reliability_20260928/prediction_only_diagnostics`. Its zero exit sentinel and log are in the parent directory.

`summarize_evidence.py` is a separate, **post-prediction evaluation** step. It joins the above diagnostics with previously frozen GT evaluation JSON to demonstrate the limits of a low-reprojection/high-inlier proxy. It never produces or changes predictions, and does not establish whether PnP itself or a later stage caused any individual final error.

```bash
python tools/preprocessing_analysis/summarize_evidence.py \
  --diagnostics /ABS/PREDICTION_ONLY_DIAGNOSTICS \
  --run /ABS/FROZEN_RUN \
  --evaluation /ABS/EXISTING_POST_PREDICTION_EVALUATION \
  --output /ABS/NEW_SUMMARY.json
```

Interpretation limits: hull/mask area is not a bounded probability; point covariance is a model-dependent conditioning diagnostic; the 5mm self-visibility test is insensitive to very thin objects; sparse background selection may include moving regions and epipolar checks do not test scale; 128px MANO depth checks do not measure contact correctness. All thresholds are diagnosis settings, not production rejection gates. See the two 20260928 reliability/plan documents under `docs/`.
