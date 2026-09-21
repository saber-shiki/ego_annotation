# Milk RL local working baseline

This branch preserves the local milk pipeline that completed observation production,
zero-action export, official skrl PPO training, deterministic rollout and Rerun export.
It is a runnable baseline, **not a convergence or annotation-quality claim**.
It does not include the separate unfinished V7 work or change the mesh/reward design.

## Source identity

- Team base: `823a906da71a03bfbb185fa5045d982f2684e7c8` (`case/rl-refactor`).
- Executed snapshot: `/mnt/user-home/kupingxin/ego_annotation_runtime/rl_milk_minfix_v1_20260918`.
- Core change: the observation-owned interaction compatibility adapter described in
  [MILK_INTERACTION_COMPATIBILITY.md](MILK_INTERACTION_COMPATIBILITY.md), its tests,
  and its connection in `main.py`.
- All 96 files listed in
  [`runtime_snapshot.json`](../reproduction/milk_rl/runtime_snapshot.json) match
  this branch's source/configuration bytes. This is a file-level identity check,
  not a claim that the archived directory itself was a Git commit.
- Private `assets.json`, credential files, model weights, videos, observation data,
  checkpoints, NPZs and RRDs are external and deliberately not committed.

## What completed

The source clip is `hot3d_P0014_84ea2dcc_carton_milk_f2370_2519`, 150 frames at
30 FPS. Existing server artifacts are at:

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/
  hot3d_milk_823a906_minfix_v1_20260918/source_observations.json
  indexes/milk_823a906_minfix_v1.json
  milk_rl_smoke_minfix_v1/zero/
  milk_learning_stage1_1024_v1/checkpoints/
  milk_learning_stage1_1024_v1/rrd/
```

The 1024-interaction experiment used official skrl PPO with deterministic evaluations
at 256/512/768/1024 interactions. The zero-action final objective was
1821.4214886534671; the best trained checkpoint (256) was 8900.21514216587, worse
than zero. Do not interpret successful execution as improved annotation quality.
Checkpoint hashes and objectives are retained in
[`run_reference.json`](../reproduction/milk_rl/run_reference.json).

## Runtime and reproduction

Use the existing [README](../README.md) for `main.py observe`, asset preflight,
index construction, `train.py`, `rollout.py`, and explicit rendering. Model workers
require separately provisioned interpreters/assets and an external credentials
manifest; a clone alone does not contain them. Do not run heavy observation/model
jobs on a workstation without explicit authorization.

The existing dedicated RL environment was Python 3.11 with torch 2.6.0,
skrl 2.1.0, Gymnasium 1.3.0 and rerun-sdk 0.37.2. The historical
[`environment.freeze.txt`](../reproduction/milk_rl/environment.freeze.txt) records
that environment; its editable-install path is machine-specific, not a portable
requirements file. For a new checkout follow `pip install .[rl,visual]` in the
main README and provision external model assets independently.

For a single-clip smoke use `--split all`, not a train/test split that might omit
this one episode. For example, with pre-provisioned data/runtime:

```sh
python -m data.build_index /PATH/TO/RUN/source_observations.json --output /NEW/episode_index.json
python train.py --config configs/ppo_annotation.yaml --dataset /NEW/episode_index.json \
  --split all --output /NEW/training --device cpu
python rollout.py --config configs/ppo_annotation.yaml --dataset /NEW/episode_index.json \
  --split all --checkpoint /NEW/training/agent.pt \
  --checkpoint-provenance /NEW/training/agent.provenance.json \
  --output /NEW/rollout/manifest.json --device cpu
```

Those are ordinary entrypoint examples, **not an exact replay of the 1024-step
learning-curve experiment**. The latter used the unchanged controller archived in
[`reproduction/milk_rl/stage1/`](../reproduction/milk_rl/stage1/), with its saved
`effective_config.yaml`. The original RRD exporter is archived in
[`reproduction/milk_rl/rrd/`](../reproduction/milk_rl/rrd/).

These archival scripts intentionally preserve the bytes of the executed scripts,
including original absolute source/data/output paths. Their READMEs are historical
run instructions, not current status. The synthetic harness also imports an
external baseline fixture. They are not installed package entrypoints or a portable
one-command launcher. Do not invoke them against existing output directories;
inspect and explicitly rebind paths in a new experiment before replaying. Source
observations themselves contain hash-bound references, so moving only an index
or NPZ is insufficient. No exact mid-video training resume is claimed.

## Known geometry and training limitations (preserved, not fixed here)

- Upstream generation is SAM3D Objects, despite historical `trellis` field names
  in the shared face-support helper.
- Complete canonical SAM3D: 239478 vertices / 478976 triangles.
- The observation bundle binds the mixed pose mesh: 138150 vertices / 270345
  triangles (7027 observed triangles plus 263318 retained generated triangles).
- Contact geometry uses the observed-only surface: 3666 vertices / 7027 triangles.
- The completion report asks for the intact SAM3D render prior, but the RL NPZ/RRD
  path currently exports/displays the mixed pose mesh. This known binding mismatch
  is retained to preserve the executed baseline. The RRD exporter uses flat colors,
  not SAM3D texture/Gaussian appearance.
- Policy object inputs use source poses and observed points; observed geometry
  rewards align observed points across frames, rather than fitting a complete
  SAM3D shell. Contact anchors are frozen on the observed surface at assembly.
- The inspected full zero-action ledger has no declared object-image evidence and
  inactive signed penetration due to non-watertight/missing local face authority.
  A complete-body collision or silhouette constraint must not be inferred from
  the presence of a complete generated mesh.

## Lightweight source validation

From the checkout root, with the pre-provisioned RL environment:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest observations.test_interaction_assembly
```

Publishing validation in the pre-provisioned CPU RL environment:

- `observations.test_interaction_assembly`: 8 tests, OK, 1 skipped (optional
  external saved-judgment batch not supplied).
- Grouped suite (`test_pipeline observations.test_observations
  observations.test_manifest rewards.test_terms envs.test_qualification
  legacy_optimize.test_legacy_optimize`): 143 tests, OK, 4 skipped.
- All 96 snapshot file hashes match; staged diff whitespace, Python syntax,
  large-file and credential-pattern checks passed. A pattern scan is not a
  guarantee of the absence of every possible secret.

Tests do not establish physical annotation quality. Publishing this baseline does
not rerun perception or training and does not mutate the existing frozen outputs.
