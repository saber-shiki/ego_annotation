# Coarse preprocessing → annotation PPO

This is an **explicit experimental training adapter**, not removal of the old v2 guards and not promotion of R6 preprocessing to the default pipeline. The raw observation-only `main.py` and `data.dataset.EpisodeDataset` are unchanged.

## Inputs and compatibility

Use a prediction-only `ego.annotation.monocular_coarse_scene.v1` result with `canonical_frame=centered_native_appearance_camera_axes`, complete native topology, one fixed uniform scale, unchanged source MANO, and its original hash-bound v2 observation bundle. Optimized preprocessing is the **initial-state reference**, not raw measurement or a supervised target. Source camera/K/timestamps, MANO, original pose/metric validity and independent RGB/depth/contact declarations are checked separately.

The new environment has **551 observation features and 118 residual actions** with the standard one-object/five-contact-slots configuration. It adds typed visibility/contact/pad declarations, source-vs-inferred state flags, causal incoming RGB summaries, and current/previous candidate object corrections. It does not add an 8-frame temporal encoder. Checkpoints need a new compatibility contract; old 480-feature policies reject.

The default `formal_annotation` ratios and PPO hyperparameters remain unchanged. Independent qualified source RGB rotation and source-world reprojection factors now drive the existing object reward kernels. No noisy RGB translation edge is chained or enabled. Unsigned contact membership is rebound once on the complete coarse native mesh; generated faces have **no signed authority**. Scale, shape, cameras and MANO betas remain fixed. Object image/first-hit/sigma priors absent in the source remain absent.

## Commands

Use a pre-provisioned environment; no package/model installs. Heavy work belongs on the server, not the workstation. Freeze code before a real run and create new outputs.

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python
$PY -B -m data.coarse_training /absolute/preprocessing/coarse_scene.json --output /fresh/index.json
$PY -B train.py --config configs/ppo_coarse_annotation.yaml --dataset /fresh/index.json \
  --split all --output /fresh/training --device cpu
$PY -B rollout.py --config configs/ppo_coarse_annotation.yaml --dataset /fresh/index.json \
  --split all --checkpoint /fresh/training/agent.pt \
  --checkpoint-provenance /fresh/training/agent.provenance.json \
  --output /fresh/rollout/manifest.json --device cpu
```

For a bounded **same-video** experiment, use the immutable snapshot's `tools/rl_experiments/preprocessed_learning.py --coarse ... --output FRESH_RUN`. It performs:

1. complete zero-action rollout and full-mesh/MANO export;
2. controlled checks on a three-frame prefix (relative object likelihoods are intentionally absent on a one-frame prefix without source sigma priors);
3. source/target RGB coupling and old-checkpoint rejection;
4. continuous seed-0 official PPO training, 1024 steps, independent deterministic full-video evaluations at 256/512/768/1024;
5. full-length zero/best-trained/last mesh/MANO RGB and fixed-world comparison videos, with decoded frame counts.

The zero baseline is not subtracted from or used to change the reward. Compare all checkpoints against zero on this **same new objective**; numeric objectives from old observation runs are not a controlled comparison. "Best trained" means the lowest objective among saved trained checkpoints and need not beat zero. Raw SAM2 silhouette IoU is an additional same-raster visual diagnostic, not an amodal mask or GT/3D accuracy metric.

Posthoc, read frozen exports with `tools/rl_experiments/inspect_preprocessed_results.py --run RUN --output FRESH_REVIEW`. It reports per-hand physical correction, actual projected joint shifts and positive-depth checks, so large weighted pixel penalties are not confused with measured pixel or physical errors.

Tests: `python -m unittest data.test_coarse_training tools.rl_experiments.test_preprocessed_learning data.test_observation_alignment envs.test_annotation_env models.test_skrl_runtime`. Retain the existing grouped source/reward/domain suite. Synthetic tests are mechanism tests, not quality evidence.
