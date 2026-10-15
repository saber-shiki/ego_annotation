# RL V7 parity execution — 2026-09-17

## Owner-approved order

User explicitly authorized remembering and implementing the plan in this order:
1. Complete real source-evidence chain.
2. Restore offline object trajectory refinement with PPO.
3. Restore complete mesh/scale/MANO/per-digit visual fitting.
4. Complete portable policy annotation JSON/RRD delivery.
5. Real multi-video training, ablation, full-timeline review and performance validation.

Authoritative implementation design: `docs/RL_V7_IMPLEMENTATION_PLAN.md`.
Repo: `/mnt/user-home/kupingxin/ego_annotation_rl_refactor`.
Branch: `case/rl-refactor`; starting HEAD `823a906da71a03bfbb185fa5045d982f2684e7c8`.
No commits/push, changes to original checkout, model environment mutation, or local heavy jobs authorized.

## Current checkpoint

- Stage 1A implementation and the three confirmed 1A/1B review-blocker fixes are present in the working tree: independent-judgment adapter → real P17 loader, typed immutable interaction/pad/visibility evidence, explicit sparse timeline handling, plus strict adapted enum/prior-provenance validation at both typed and P17 consumer seams. The blocker fixes pass bounded synthetic/regression checks and await reviewer acceptance.
- Reduced blocker: current strict contact responses now reach P17 without invented VLM metric fields. No rendered/geometric quality improvement is claimed from this data-path change.
- Remaining stage-1 blocker: 1C appearance-native/full observation integration. Stage 1 is not complete.
- Stage 1B is implemented; its independent review found target-reprojection, adapted-schema, and code-identity integrity gaps. All three are now patched and synthetically checked (46 final focused/regression tests); reviewer acceptance and real separately provisioned worker validation remain pending. Source-world RGB measurements now reach cold main/recovery → manifest → dataset without P14.
- Next bounded work after review: stage 1C native appearance/data integration.
- Stages 2–5 remain not started. A partial stage 1 fix must not close the whole stage.

## Validation environment boundary

Lightweight synthetic/contract CPU tests may use the existing lexical interpreter:
`/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`
with `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.
It has numerical/domain imports but no gymnasium/skrl; Rerun is 0.37.0 rather than project pin 0.37.2. Do not install into it or claim unified runtime readiness.
The separate `ego_skrl/.venv` has skrl 2.1.0 but lacks domain dependencies; do not splice its site-packages into the former environment.
No real models, licensed MANO inference, PPO training or heavy whole-video jobs in local checks.

## Work log

### 2026-09-17 — preparation

- Confirmed clean target branch before creating this plan and memory.
- Persisted approved sequence and mechanism-specific requirements.
- Read-only interpreter package metadata checks only; no dependency installation.
- Documentation/memory are support work, not an implemented annotation capability.

### 2026-09-17 — stage 1A implementation (checked synthetically; review pending)

- Added `observations/interaction/adapter.py`: current strict schemas joined on explicit frame/hand/object identity; rejects duplicate/overlapping/foreign judgments; keeps full individual responses, request provenance and path/hash receipts. Explicit four-value occlusion mapping; conservative depth vocabulary reduction keeps both original entity rows. Missing observations stay unresolved. Anatomical declarations do not grant reachability/contact authority.
- Main now assembles inside the recoverable stage. Recovery contract binds raw batches/receipts, adapter schema and the 0.06 m configuration prior. Numeric support uncertainty is explicitly **not** a VLM metric measurement; text uncertainty is unchanged. P17 factor provenance retains the prior source. Existing diagnostic/timeline export views are retained.
- P17 loader/build now use declared frame IDs, not manufactured intermediate IDs. Added strict adapter identity checks while preserving legacy input support.
- Added `data/interaction.py`: frozen typed contact declarations, five per-digit declarations with unknown distinct from false, and five-state visibility/occluder/uncertainty. `EpisodeBundle.load()['interaction_evidence']` is separate from existing contact/factor and geometric-visibility fields. v1 bundles without judgments remain valid with explicit missing declarations; unversioned v1 evidence is not silently upgraded.
- EpisodeBundle honors explicit sparse timelines; legacy manifests retain their consecutive range contract. Deep-frozen manifest/factor data and bytes-backed immutable arrays prevent source mutation. Supervisor-approved compatibility fixes: `thaw_evidence` at manifest validation/write and existing policy identity JSON boundary; sparse index counts use explicit IDs. No new policy export feature was added.
- Added 12 synthetic tests in `observations/test_interaction_adapter.py` and `data/test_interaction.py`. External Astra transport is stubbed; real judge wrappers/current schemas, actual assembly/P17 loader, P17 build with explicitly absent geometry, and synthetic EpisodeBundle loading are exercised. No real MANO/model/provider calls.

Exact validation commands/results:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_interaction_adapter data.test_interaction observations.test_manifest data.test_rl_data
# Initial 14 tests passed, 2.241 s; /tmp/ego-rl-v7-stage1a-focused.log

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_interaction_adapter data.test_interaction observations.test_manifest data.test_rl_data test_pipeline.StageRecoveryTests test_pipeline.AstraTransportTests test_pipeline.ExportTests.test_visibility_rejects_missing_duplicate_incomplete_bad_identity_and_states test_pipeline.ExportTests.test_production_fit_wrapper_derives_visibility_before_export envs.test_qualification
# Intermediate 32 tests passed, 1.328 s; /tmp/ego-rl-v7-stage1a-regressions.log

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_interaction_adapter data.test_interaction observations.test_manifest data.test_rl_data test_pipeline.StageRecoveryTests test_pipeline.AstraTransportTests test_pipeline.ExportTests.test_visibility_rejects_missing_duplicate_incomplete_bad_identity_and_states test_pipeline.ExportTests.test_production_fit_wrapper_derives_visibility_before_export envs.test_qualification rewards.test_terms.PrimaryPads
# Final 40 tests passed, 3.388 s; /tmp/ego-rl-v7-stage1a-final-tests.log
```

- CPU-only AST script parsed all 10 changed Python files and verified the real policy identity export statically calls `thaw_evidence`; `importlib.util.find_spec` confirmed `gymnasium=False`, `skrl=False`. Actual Gymnasium export/domain integration and full grouped import suite were **not run** because those dependencies are unavailable. No package installs or site-package splicing. Open3D emitted the expected no-CUDA-device import warning under `CUDA_VISIBLE_DEVICES=''`; the bounded synthetic triangle regressions passed.
- Infrastructure incident: first standard-library edit helper used bare `python - <<'PY'` and failed before any edit with `/bin/bash: python: command not found`, exit 127. Paused for supervisor decision; supervisor explicitly approved continuing this same run using `/usr/bin/python3` for standard-library-only edits. Corrected helper `/usr/bin/python3 - <<'PY'` succeeded. No environment mutation; authorized domain interpreter used only for lightweight tests/static checks.
- `git diff --check` passed; `git diff --cached --name-only` empty. No staging, commits, push, installs, heavy local runs, training, real MANO loading or API inference. Parent plan/memory additions preserved.
- This is evidence-chain repair only, **not** stage-1 closure or demonstrated annotation quality. 1B: independent source-world RGB measurement producer/load/recovery contract remains pending. 1C: appearance-native worker/result binding, complete native faces and versioned data integration remain pending. Candidate reachability/reward changes, V7 visual fitting, full export and real server validation remain later-stage work.

### 2026-09-17 — stage 1B implementation and same-protocol timeout recovery (synthetically checked; review pending)

Scope and mechanism:
- Preserved all 1A edits and parent plan/memory additions. Only 1B source/data/test files and this task log changed in this slice; no 1C or later-stage implementation.
- Added explicit `--observation-mode source-world` to `observations.object.rgb_tracking`. It uses current P09 retained depth replay, current camera/K, owned masks and LightGlue correspondences, not P14 canonical poses. No initializer, trajectory optimizer, or post-RANSAC LM refinement is called in this mode. The legacy default mode retains its pose input/canonical output and LM branch. Added the missing module CLI entrypoint used by worker commands.
- New serializer `observations/object/rgb_evidence.py` saves actual fixed source-world points, source/target UV, both camera/K/time/image-plane contracts, relative world motion `D=Rt Rs^T`, `d=tt-D ts`, inlier indices, match scores, reprojection diagnostics and conditioning. Generated geometry is not a measurement input. Empty results have zero-length motion/evidence arrays, not identity/zero-motion placeholders. Rejections retain exact reasons.
- Supervisor clarified the ambiguous original “weak means empty” instruction: retain mechanically usable noisy measurements, including 4–7 unique inliers, low fraction and underconditioned geometry. `min_matches` is pre-PnP only. Successful finite positive-depth PnP needs >=4 unique inliers; success is not confidence. Existing min-inliers/fraction/median settings are explicitly diagnostic references/weak reasons, not gates. Point weights remain explicitly **uncalibrated reprojection heuristics**, not probabilities. Legacy numerical acceptance decisions remain unchanged.
- `main.py` runs a separately provisioned RGB worker immediately after P09 surface production, inside its real recovery sequence. Its manifest binds NPZ/report with roles and a versioned `extensions.rgb_source_measurements = ego.annotation.rgb_source_measurements.v1` declaration; the outer v1 bundle is not silently upgraded. 1C can extend the outer bundle separately. Unextended prepared/v1 bundles return `rgb_evidence=None`; measured-empty extension returns a typed empty `RGBSourceEvidence` with diagnostics. Legacy canonical NPZ needs its legacy consumer and is not reinterpreted.
- `data/rgb.py` loads hash-bound NPZ with `allow_pickle=False`, checks complete embedded JSON pairing, exact input bytes/identity, sparse timeline/entity, shapes, finite transforms, K/planes, eligibility, source-world projection and cheirality. `EpisodeBundle.load()['rgb_evidence']` exposes frozen dataclasses and bytes-backed immutable arrays. Arrays and diagnostics are ready for stage 2; no seed/trajectory/refinement implementation is claimed.

Timeout/failure preservation:
- Original worker hit the exact **1800000 ms** execution timeout before final validation/report. Parent verified no lingering rgb/unittest process and no staged files; recovered the same branch/HEAD/worktree from preserved context without reverting edits. Parent snapshot `/tmp/ego-rl-v7-stage1b-timeout-Bl6dvY.patch`, SHA256 `d604b60e846f2e49311df6e015656e96955f5cff508e9f0f61d7b28cf9a31d10` includes tracked+untracked work before recovery.
- Inspected earlier failure logs rather than restarting: `/tmp/ego-rl-v7-stage1b-focused.log` (5 tests: no-source path incorrectly called time_keyframes on empty features; fixed to produce an empty evidence result); `/tmp/ego-rl-v7-stage1b-chain.log` (6 tests: missing synthetic candidate identity fixture); `/tmp/ego-rl-v7-stage1b-cold.log` (missing synthetic anchor review fixture); `/tmp/ego-rl-v7-stage1b-cold2.log` (incorrect test expectation that tampered stage must raise rather than quarantine/rerun). These failures were code/test contract failures, not environment provisioning failures.
- Recovery semantics were read and left unchanged: immutable baseline changes fail, generated RGB report changes invalidate/quarantine the RGB/dependent suffix and rerun. The final cold test proves unchanged receipts are reused, tampered bytes plus old source manifest are archived, prefix receipts remain identical, RGB reruns exactly once, and the new manifest loads valid evidence. After quarantine, the test indexes the delivered manifest explicitly, not historical manifests in quarantine. No silent tamper reuse or weakened verification.

Validation commands (all CPU-only; no models, real MANO, API, installs or training):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_rgb_source
# Earlier run: 5 tests, 1 error; /tmp/ego-rl-v7-stage1b-focused.log

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_rgb_source observations.test_rgb_cold_main
# Earlier run: 6 tests, 1 fixture error; /tmp/ego-rl-v7-stage1b-chain.log

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_rgb_cold_main
# Earlier runs: 1 fixture error then 1 recovery-expectation failure;
# /tmp/ego-rl-v7-stage1b-cold.log, /tmp/ego-rl-v7-stage1b-cold2.log

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_rgb_source data.test_rgb observations.test_rgb_cold_main observations.test_observations.RotationChainTests
# Recovery focused: 16 tests passed, 2.585 s; /tmp/ego-rl-v7-stage1b-recovery-focused.log

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_manifest data.test_rl_data observations.test_interaction_adapter data.test_interaction test_pipeline.StageRecoveryTests observations.test_observations.RGBMetricReplayTests
# Regression: 30 tests passed, 0.880 s; /tmp/ego-rl-v7-stage1b-recovery-regression.log

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_rgb_source data.test_rgb observations.test_rgb_cold_main observations.test_observations.RotationChainTests observations.test_manifest data.test_rl_data observations.test_interaction_adapter data.test_interaction test_pipeline.StageRecoveryTests observations.test_observations.RGBMetricReplayTests
# Final: 46 tests passed, 3.471 s; /tmp/ego-rl-v7-stage1b-final.log
```

- Added 11 tests across `observations/test_rgb_source.py`, `data/test_rgb.py`, and `observations/test_rgb_cold_main.py`; subcases include nonconsecutive IDs, target without metric depth/pose, real support replay + real PnP, retained 4–7 inliers/low fraction/underconditioning, insufficient input, failed PnP, duplicate/foreign/<4 inliers, invalid transform, negative depth, tampering/pair mismatch/foreign IDs, immutable consumer arrays, optional v1 capability, and real recovery behavior. LightGlue extraction/matching and upstream cold-model/geometry/judgment callbacks are explicitly mocked; cold `main.annotate`, RGB production, recovery, manifest and dataset consumers execute. Boundary-policy tests additionally mock PnP to inject exact inlier outcomes. Synthetic MANO numeric fixtures only.
- AST parsed all 9 stage-1B Python files with the authorized interpreter and same CPU environment; passed. `git diff --check` passed after removal of extra EOF blank line. `git diff --cached --name-only` was empty. Final diff inspected; tracked diff inspection copy is `/tmp/ego-rl-v7-stage1b-final-tracked.diff` (new files inspected separately).
- Not run: actual separately provisioned LightGlue/SuperPoint, upstream raw-video model stages, real licensed MANO, Gymnasium/skrl policy/export, training, full grouped suite, full-video rendering/visual quality/runtime. Existing domain interpreter lacks gymnasium/skrl and has rerun 0.37.0; no environment repair or dependency splicing.
- Residual risks/next step: required independent review, then 1C full appearance/native extension. Real server worker smoke remains necessary. Source evidence intentionally binds absolute input paths and exact current bytes; portability/export is later-stage work. No rendered-quality claim, no whole-stage or roadmap completion, no commits/staging/push.


### 2026-09-17 — stage 1A/1B independent-review blocker fixes (synthetically checked; rereview pending)

Dispositions and source evidence:
- **Confirmed — target endpoint integrity (`data/rgb.py`).** The loader previously checked target cheirality but projected only the source endpoint. It now transforms fixed source-world support by the serialized relative world motion, projects it with the validated target camera/intrinsics, requires finite in-plane target pixels, requires every accepted residual to satisfy the serialized RANSAC inlier threshold, and recomputes/compares median and p95 diagnostics. Repacked `target_uv`, median, p95, and threshold tampering are negative tests through the real producer/typed loader boundary.
- **Confirmed — adapted-schema integrity (`data/interaction.py`, `actions/filters/contact.py`).** Adapted v1 rows previously allowed arbitrary mapped occlusion/depth strings and did not bind the support uncertainty to its declared conservative prior. A shared consumer validator now enforces exact mapped vocabularies and requires `kind=conservative_configuration_prior`, parameter `contact_support_uncertainty_prior_m`, exact numeric agreement with `contact_support_uncertainty_m`, and `is_metric_measurement is False`. Both EpisodeBundle's typed loader and the real P17 loader enforce it. Unversioned legacy/v1 behavior remains unchanged.
- **Confirmed — cold-run code identity (`main.py`).** The manifest covered `main.py`, `utils`, `observations`, `actions`, and `envs`, but omitted the new authoritative `data` consumers. `data/**/*.py` is now included. The real cold/resume test temporarily changes `data/rgb.py`, verifies resume refuses cross-version recovery before stage reuse, restores the exact bytes, and then continues the existing artifact-tamper recovery test.
- No 1C or later-stage mechanism was implemented. No optimizer/renderer was added to the observation path.

Exact validation commands/results (CPU-only; no model/API/training/MANO execution):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_rgb_source data.test_rgb observations.test_interaction_adapter data.test_interaction observations.test_rgb_cold_main
# First post-edit run: 23 tests, 7 errors. The new diagnostic equality tolerance was too strict for the small numerical difference between OpenCV's direct projection and projection through the serialized world transform. /tmp/ego-rl-v7-stage1ab-review-fixes-focused.log was later overwritten by the second focused run below.

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_rgb_source data.test_rgb observations.test_interaction_adapter data.test_interaction observations.test_rgb_cold_main
# Second post-edit run: 23 tests, 2 failures. The target-UV test changed one of 30 pixels, which correctly stayed below p95 and the RANSAC threshold; the threshold comparison's absolute numeric allowance was also too broad for a deliberately repacked near-zero threshold. Test changed to repack all target x coordinates and threshold allowance changed to scale with the declared threshold. /tmp/ego-rl-v7-stage1ab-review-fixes-focused.log

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest -v observations.test_rgb_source data.test_rgb observations.test_rgb_cold_main observations.test_observations.RotationChainTests observations.test_manifest data.test_rl_data observations.test_interaction_adapter data.test_interaction test_pipeline.StageRecoveryTests observations.test_observations.RGBMetricReplayTests
# Final prior A/B regression: 46 tests passed in 3.820 s. /tmp/ego-rl-v7-stage1ab-review-fixes-regression.log; SHA256 5ba929a4388267a0fce6b0c84c9a639c8200b865ad2d8751e50a4681c9c292d6

git diff --check
git diff --cached --name-only
git status --short
# diff check passed; cached/staged file list empty; status contains only the preserved intentional working-tree implementation/plan/memory plus these blocker fixes.
```

Residual limits:
- These fixes establish source/data integrity only. They do not complete stage 1, implement 1C, demonstrate rendered quality, or establish production readiness.
- Actual separately provisioned LightGlue/SuperPoint and raw-video workers remain unvalidated; server-side cold smoke is still required.
- Gymnasium/skrl and the project-pinned Rerun boundary remain untested in the available interpreter. No environment was installed into or spliced.
- No real MANO, inference, training, full-video render, visual comparison, or throughput validation was run.
