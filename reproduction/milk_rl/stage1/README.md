# Milk stage1 learning-curve experiment (not a source change)

The three Python files in this directory are a control harness and bounded API tests.
They explicitly import the frozen snapshot at
`/mnt/user-home/kupingxin/ego_annotation_runtime/rl_milk_minfix_v1_20260918`.
No real milk training has been run by the implementation step. The destination
`milk_learning_stage1_1024_v1` must not exist before the parent launches it.

## Exact launch command — parent only, after review

```bash
cd /mnt/user-home/kupingxin/ego_skrl/milk-learning-stage1-9q9skT
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
  timeout --signal=TERM --kill-after=30s 7200s \
  /mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -B -u learning_harness.py \
  --evaluator-timeout 1800 > launch.log 2>&1
```

This runs CPU/one-thread, seed0 Python/NumPy/Torch and seeded Gym RNG, `split=all`,
1024 uninterrupted training interactions. It creates the fixed experiment directory
exclusively, and refuses alternate real-data config/index/output paths. An external
7200s wall limit bounds training plus evaluations; each evaluator has its own 1800s
limit. Any nonzero exit must be treated as failure, including `timeout` exit 124.
Do not rerun into a partially populated experiment directory or delete its evidence.
A hard external timeout can leave partial logs without a final failure JSON.

The effective config differs from original YAML only in trainer.timesteps=1024,
disable_progressbar=true, and experiment write/checkpoint intervals=0. The last two
avoid TensorBoard noise and skrl's automatic raw-reward-selected checkpoints; the
harness saves full agent checkpoints instead. PPO rollouts=128, epochs=8, minibatches=8,
gamma=1, learning rate, preprocessing, architecture, rewards, scales, actions and data
remain original. No manual optimizer update or chunked trainer restart is used.

## Outputs

At `checkpoints/step_{0256,0512,0768,1024}/`:

- `agent.pt`, `agent.provenance.json`: full skrl checkpoint (including optimizer and
  preprocessors), finite and policy-change checks, effective config SHA and checkpoint SHA.
- `evaluation.manifest.json`, companion provenance, and `.episodes/`: original
  deterministic Gaussian mean-action rollout exports, including full geometry NPZ,
  candidate JSON and integrity receipt for all 150 frames. Source assertions for finite
  actions, no truncation, natural completion and frozen preprocessors are retained.
- `evaluation.diagnostics.json`: every step's raw/executed actions, physical delta,
  gate mask/reasons, reward, objective and prefix ledger; ONLY terminal ledgers are
  aggregated. Named-term groups collapse hand side/object slot/frame IDs, while the
  complete original ledger terms remain available.
- `receipt.json`: original checkpoint hash, evaluator outputs and hashes, elapsed
  evaluation time, training Python/NumPy/Torch RNG hashes and environment position,
  episode, elapsed steps and Gym RNG state before/after evaluation. Subprocess isolation
  avoids changing training models/RNG/env; equality is asserted rather than masking a
  mutation with RNG restoration.
- `evaluator.stdout.log`, `evaluator.stderr.log`; timeout log on evaluator timeout.

At experiment root:

- `runtime_identity.json`: package versions, Python executable/platform, thread settings,
  script/source tree hashes, index and source manifest hashes.
- `run_config.json`, `effective_config.yaml`: exact effective configuration and original
  config hash; source YAML is never rewritten.
- `zero_baseline.json`: existing zero's FINAL ledger, families and hashes. Prefix costs
  are not summed. Read-only validation gives objective **1821.4214886534671**.
- `training_diagnostics.json`: per-step action vectors, per-kind normalized raw/executed
  and physical-unit distributions, fraction filtered by mask, fraction nonzero raw
  coordinates zeroed (denominator: all coordinates), and training episode-end objectives
  separate from deterministic evaluations. Captures original tracked policy/value loss
  and policy std. KL is explicitly unavailable because skrl2.1 retains it locally.
- `checkpoints.json`, `summary.json`: four deterministic evaluations, minimum FINAL
  objective checkpoint (not last or reward sum), deltas vs zero and supplied rounded
  previous256 reference **8900.215**, including final term-family differences.
- `run.failure.json` for failures inside the training loop. Initial construction failures
  and hard timeout instead leave a nonzero launch exit and partial launch/output evidence.

## Bounded validation

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -B -m unittest -v test_learning_harness 2>&1 | tee tests.log
```

Tests use fakes and the same-user baseline API fixture (loaded read-only), with a tiny
local diagnostic adapter. One test performs only 12 synthetic training interactions
(three true skrl updates), exports two synthetic three-frame episodes per successful
checkpoint, verifies RNG/model/env noninterference and full optimizer/preprocessor
checkpoint contents, then intentionally fails an evaluator at step12 to prove training
never reaches step13. No MANO construction or real-data training is involved. All test
artifacts are disposable under `/tmp`; `tests.log` and `validation.json` are retained here.

## Scope and possible stage2

This is **same-video optimization**, not convergence, generalization, or annotation-quality
acceptance. The zero ledger is the baseline; lower measured objective alone is not an
annotation-improvement claim. Previous256 is a user-supplied rounded reference, not a
newly validated checkpoint. No automatic 4096 decision is made.

The saved full agent checkpoints do not include all state needed for exact mid-video
continuation: training environment/history, Gym/global RNG states, trainer cursor,
rollout memory bookkeeping and any non-checkpoint runtime state would need capture and
validation. The hashes logged here prove noninterference but cannot restore RNG state.
There is intentionally no `--resume` or 4096 option. After measured stage1 review and
approval, either implement and test exact continuation against an uninterrupted synthetic
run, or explicitly authorize a fresh seeded 4096-step experiment and verify its first1024
trajectory/checkpoints against stage1. Do not call `agent.load` alone an exact resume.
