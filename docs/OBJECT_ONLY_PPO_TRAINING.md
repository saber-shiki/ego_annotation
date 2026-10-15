# Object-only safe-residual training entry

`configs/ppo_object_only_coarse.yaml` selects `envs.object_only_coarse_env:ObjectOnlyCoarseEnv` over the existing explicit R6/v2 `CoarseTrainingDataset`. It is a separate experimental contract; old v1/v2 guards and full 118-action policies are unchanged.

## What is different

- 551 source/causal observation features, **six** raw Gaussian latent actions: object rotation xyz, world translation xyz.
- No sampled hand or contact coordinates. Full source MANO/triangles still drive every export/render and all hand factors remain in the unchanged objective.
- Latents map through the existing .05rad/.01m unit scales and a smooth radial-tanh ball. Each frame's correction is bounded relative to its **frozen R6 initializer** by rotation-vector norm .5° and translation norm 2mm. This is not a frame-to-frame velocity bound.
- PPO log probabilities/ratios are over the six sampled raw latents. Do not recompute a Gaussian probability on mapped physical deltas. A deterministic environment map is valid; it is not a claim that all post-sample filtering was an earlier PPO bug.
- Initial log std is [-2.8]*3+[-3.0]*3, with [-5,-2.5] bounds, zero mean. No implicit std annealing or warm-start checkpoint.
- Zero-reference prefix objective J0 is evaluated once with the exact same episode and zero decoder. Centered reward is `-c*((J_t-J0_t)-(J_prev-J0_prev))`; c=1e-5, gamma1. All original factor weights/reductions remain. Formal evaluation always reports **raw final J** and its named factors. Centering is an episode-wise action-independent constant shift, not a physical improvement and not guaranteed variance reduction.

## Bounded experiment

Use the pre-provisioned server environment and immutable code snapshot. No model/API inference, numerical optimizer or installation occurs in the training run.

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -B -u \
  tools/rl_experiments/object_only_learning.py \
  --coarse /absolute/R6/coarse_scene.json --output /fresh/experiment/root
```

The supported experiment is 600 steps, 150-step rollouts, 4 epochs/4 minibatches, lr1e-4, ratio/value clips .1, grad norm .25, KL threshold .01. It saves independently evaluated complete 150-frame exports at 150/300/450/600 after the official skrl update. It checks exact hand geometry preservation, unchanged hand factor totals, trust-ball limits, return telescope and old 118D checkpoint rejection. The value model still consumes the current 551D observation, not new privileged critic summaries.

Diagnostics include raw/relative J, six raw/executed traces, actual Adam step counts, post-update approximate KL/clip fraction and official tracked policy/value losses. Post-update likelihood diagnostics restore Torch RNG (GaussianMixin can sample even with taken_actions). Their KL is not the internal minibatch early-stop statistic; zeros in an epoch's tracked average may include skipped minibatches under skrl's stock reduction.

Outputs include complete native/MANO zero/best-trained/last RGB overlay and fixed-world videos. A "best trained" checkpoint can still be worse than zero; never replace R6 solely because training ran. Raw SAM2/complete-mesh IoU is a same-raster visual diagnostic, not an amodal mask, contact authority, GT or 3D accuracy metric. Positive results on one video are not generalization.

This is a declared **joint intervention** (action subspace, noise, mapping, centered reward and PPO settings); improvement cannot attribute causality to one hyperparameter. Only after raw J/geometry/render inspection should a component ablation or hand-unfreezing be considered.

Tests: `python -m unittest envs.test_object_only_coarse_env data.test_coarse_training models.test_skrl_runtime tools.rl_experiments.test_preprocessed_learning`. Retain the grouped pipeline/source/reward/domain suite.
