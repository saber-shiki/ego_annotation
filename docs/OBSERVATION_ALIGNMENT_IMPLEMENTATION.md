# Observation / preprocessing alignment implementation — 2026-09-21

> Update 2026-09-22: the earlier "real workers not run" notes below describe the
> initial code-delivery stage. Real r2 observation production/readback and
> controlled environment-side consumption have now been executed. The old policy
> factory/checkpoint remains incompatible. See [REAL_V2_MILESTONES_RESULTS.md](REAL_V2_MILESTONES_RESULTS.md).


## Scope and source identities

- User authorized changes on an independent RL branch, limited to observation and preprocessing, with per-part reports. No training changes.
- Branch: `feature/rl-observation-v7-alignment-20260921`.
- Worktree: `/mnt/user-home/kupingxin/ego_annotation_rl_observation_alignment`.
- Base: `823a906da71a03bfbb185fa5045d982f2684e7c8`.
- V7 reference remains read-only: `research/rgb-scene-v7-baseline` / `67c40e6`.
- Existing dirty worktree was not reset or modified. Its 20 changed/untracked files were backed up under `/mnt/truenas-user-home/kupingxin/rl_observation_alignment_implementation_20260921/` before branching.
- The earlier uncommitted observation-side interaction/RGB patch was read, selectively reused, and retested. Its action-filter/env/export changes were NOT imported. The new observation-side P17 loader is under `observations/interaction/loader.py`; existing policy filters remain unchanged.
- No commits, pushes, package installations, model inference, Astra requests, real MANO weight loading, training or real-video production run were performed.

## What changed and why

### Part 1 — source roles, timeline, immutable loading

Root defect: implicit consecutive IDs could manufacture positions absent from sparse input, and source-derived initializers were easy to confuse with measurements.

- `observations/contracts.py`: v2 schema, capabilities and R0–R5 artifact roles; classification is explicitly not an accuracy certificate. P09 geometry is R2 while its centroid initializer is R4.
- `data/timeline.py`: exact ID join and timestamp validation. V1 without explicit IDs retains its consecutive contract.
- `data/immutable.py`: bytes-backed read-only arrays and JSON-compatible frozen nested mappings. This protects normal consumer APIs, not malicious Python reflection or calls to dict base-class mutation methods.
- `data/bundle.py`: sparse IDs and timeline matching, immutable source values.
- `data/build_index.py`: counts actual declared frames; v2 preprocessing packages are not silently indexed into unchanged training.

Part-specific check: 14 bounded CPU tests passed, including unchanged synthetic env geometry/export tests. No annotation-quality claim.

### Part 2 — independent judgments → typed interaction observations

Root defect: strict contact responses lacked P17's occlusion and metric-prior fields; saved pads/visibility were not available as typed data.

- `observations/interaction/adapter.py`: joins independent contact/pads/occlusion/depth/visibility by explicit frame/hand/entity. Preserves raw results, receipts and missing/unknown states; validates supplied source JSON bytes against receipts.
- Maps occlusion/depth vocabularies explicitly. A supported-object/mixed-hand combination may become `hand_depth_unreliable`; missing object support does not become object-depth authority.
- `0.06 m` is labeled `conservative_configuration_prior`, never an image-derived metric measurement.
- `observations/interaction/loader.py` is the observation-side P17 consumer; sparse IDs and adapted vocabularies are validated without changing policy action filters.
- `data/interaction.py`: frozen typed declarations and full visibility timeline.
- `data/interaction_features.py`: `[N,2,5]` primary/known arrays, `[N,3]` visibility, and source-only visual/display eligibility. These are not final display-valid flags and do not mutate source MANO.
- Cold `main.py` connects the typed assembly to the existing P17 geometry-factor producer.

Part-specific check: 26 tests passed. Read-only existing milk judgment loading retained 300 declarations and the left90–95/right84–89 out-of-frame declarations. Source hash remained unchanged. The policy renderer still uses its original logic; it has NOT been updated to consume these new arrays.

### Part 3 — independent RGB measurements and local motion preprocessing

Root defect: source RGB evidence depended on P14/legacy pose inputs and was not produced by the cold observation entry.

- `observations/object/rgb_tracking.py`: explicit `--observation-mode source-world`; forbids initial/output pose reports for this mode. Legacy-canonical mode remains available for the reference command.
- Source points are lifted only through retained P09 support; a target may lack depth/source pose. Missing edges are empty evidence with reasons, not zero/identity-motion observations.
- Separate model boundary for SuperPoint/LightGlue. Tests replace that boundary, not the actual PnP/support replay.
- Local PnP LM is explicitly a measurement refinement (default `--pnp-refinement lm`), not global trajectory optimization. `ransac-only` remains explicit for controlled fixtures. It never invokes the legacy full initializer.
- `observations/object/rgb_evidence.py`: paired NPZ/JSON, exact input identity, fixed source-world points, both camera/K/time/pixel-plane records and quality diagnostics.
- `data/rgb.py`: immutable typed loading and source/target reprojection checks. Corrected a reused-patch defect: diagnostics must be compared in the matcher/manifest pixel plane, not the potentially differently sized serialized mask plane. LM output is checked against recorded diagnostics, not assumed to remain under the original RANSAC inlier cutoff.
- `data/motion.py`: V7 local-tree/cycle and conditioning semantics; separate measurement-propagated SE(3) seed marked R4. No least-squares or candidate update. A disconnected nonmetric frame remains unresolved; a supported nonmetric target may have an uncertain seed.
- Main/recovery/manifest/bundle now connect the RGB worker; local motion preprocessing is derived when annotations declare an anchor.

Part-specific check: 44 focused/current-and-prior tests passed. Real OpenCV PnP, both local refinement modes, nonzero residual plane conversion, sparse IDs and cold/resume contracts covered. Actual pretrained LightGlue/SuperPoint was NOT executed.

### Part 4 — full appearance-native geometry observation

Root defect: complete native geometry existed but had no independent typed source path for the RL observation consumer; appearance reconstruction was confined to the legacy suffix.

- `observations/object/visual_geometry.py`: creates a source-only complete native observation from current appearance RGB/mask and retained metric support. Replays native camera transforms, preserves every face, applies source scene similarity, and saves `visual_full_metric_camera.ply` plus its report.
- The mesh remains in OpenCV camera coordinates at the appearance anchor. There is no automatic canonical binding to a legacy posterior and no 21-point image-scale search.
- Cold main supports a new appearance-native worker or strictly same-run, same-frame, same-byte, same-seed native reuse. Original model/config/receipt authority comes from the immutable recovery prefix; the reuse-decision receipt is NOT a new model execution receipt.
- V2 names visual-native/report/metric-camera mesh separately from pose/collision meshes.
- `data/geometry.py`: validates and loads complete geometry, its anchor camera/K, input identity and native topology. `bind_visual_to_candidate()` is a pure explicit transform API requiring a candidate revision; no env or decoder integration is implied.
- Generated geometry remains non-collision/non-signed-authoritative.
- `observe` defaults to v2. Explicit version1 preserves the old bundle layout without the new complete visual extension; it still benefits from the observation-side interface fixes.

Part-specific check: 41 related tests passed. Real main/recovery/manifest/data consumers execute on synthetic upstream producers, including the new native geometry preparation. No real SAM3D inference or full-video rendering was performed.

### Part 5 — source MANO and appearance static views

Root defect: source full-MANO/appearance data had no common typed preprocessing view for future consumers.

- `data/visual_preprocessing.py`: source vertices/joints in camera coordinates and a fixed960-wide fitting plane; positive-depth/in-frame masks remain distinct from source validity.
- Per-frame, per-digit 12-vertex source MANO membership uses the actual full source vertex array; absent hands carry `-1` IDs and a false validity mask, not fabricated pads.
- Lazy appearance reader validates byte-identical raw-mask copies and produces the fixed-plane mask, 160-point contour and distance transform per requested frame. Does not allocate whole-video distance transforms by default.
- These are source observations. No MANO fitting, camera-origin hand-scale action, policy feature layout or nearest-triangle candidate update was introduced.
- Full licensed MANO LBS replay of every real frame has not been executed; the preprocessing view explicitly states `mano_lbs_replay_performed=False`.

Part-specific check: 10 tests passed, including 3 differential tests that extract only named pure function definitions from frozen V7 `git show 67c40e6:...`. No V7 module-level entry, model loader or optimizer is executed. Checks cover pixel-center K, primary visible declaration semantics and native/canonical geometry transformation.

## API boundary and use

```python
from data.bundle import EpisodeBundle

bundle = EpisodeBundle.open('/ABS/NEW_RUN/source_observations.json')
observations = bundle.load()

# Existing source fields remain separately present.
observations['hands']
observations['objects']                   # legacy pose/collision roles unchanged

# New observation/preprocessing views.
observations['interaction_evidence']      # declarations incl. provenance/uncertainty
observations['interaction_timelines']     # entity -> source typed arrays
observations['rgb_evidence']              # None if capability absent, empty edges if measured-empty
observations['motion_preprocessing']      # R2 qualification + separate R4 initialization
observations['visual_geometry']           # full metric-camera native mesh, not canonical candidate
observations['source_visual_preprocessing']
observations['appearance_evidence']       # lazy .prepare_frame(actual_frame_id)
```

V1 bundles preserve missing capabilities explicitly (e.g. `rgb_evidence=None`). They are not silently promoted to v2. V2 advertises actual available capabilities and requires explicit timeline and role metadata.

The existing training-index/dataset interface rejects `preprocessing_only` v2, including a hand-authored index. Inspect v2 through `EpisodeBundle`, not unchanged `train.py`. A caller manually bypassing the dataset into an old environment is outside the new observation interface contract; this change does not purport to make the old actor consume new fields.

Cold CLI (NOT executed on real data in this task):

```bash
python main.py observe --video INPUT --case CASE --run-root NEW_RUN \
  --assets-manifest EXISTING_PROVISIONED_ASSETS --gpu EXPLICIT_GPU \
  --observation-bundle-version 2
```

Future production requires a fresh immutable source snapshot and new run root. A checkpoint/version change cannot resume an old run under its old identity. No asset/environment provisioning is implied.

## Validation commands

All local tests were CPU-only, single-threaded and used synthetic geometry/numeric fixtures. Existing real-model optional switches were unset. Neither PPO tests nor model inference were run.

```bash
cd /mnt/user-home/kupingxin/ego_annotation_rl_observation_alignment
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
EGO_V7_REFERENCE=/mnt/user-home/kupingxin/ego_annotation_worktrees/standalone_v7_baseline_20260912 \
env -u EGO_MANO_MODELS -u EGO_OWLV2_MODEL -u EGO_IMAGE_TARGET_PYTHON \
/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -B -m unittest \
  test_pipeline observations.test_observations observations.test_manifest \
  observations.test_cold_contracts observations.test_interaction_adapter \
  observations.test_rgb_source observations.test_rgb_cold_main \
  observations.test_visual_geometry observations.test_v7_reference_preprocessing \
  data.test_rl_data data.test_observation_alignment data.test_interaction \
  data.test_rgb data.test_visual_preprocessing rewards.test_terms \
  envs.test_qualification envs.test_annotation_env envs.test_domain_semantics \
  legacy_optimize.test_legacy_optimize -v
```

The first combined run discovered 199 tests: 195 passed, 4 intentionally skipped (real licensed MANO, real tokenizer/provider prerequisites). Reference reward/solver synthetic tests are regressions of unchanged code, not a production optimizer run or training session. The final rerun after boundary hardening is recorded under the evidence directory.

## Preserved failures and fixes

- First cold visual integration fixture pre-created an interaction file in an earlier stage, then the real assembly tried to replace it. StageRecovery correctly rejected mutation of immutable earlier artifacts. Fixed the fixture to let the real assembly create its own output; recovery protection was not weakened.
- First read-only V7 canonical differential fixture omitted the pure `_pose` helper. Added that named definition to the isolated AST extraction; no V7 module execution was introduced.
- Removed duplicated unittest discovery caused by importing a TestCase class directly into the first new test module; the reported first-part count is the deduplicated 14.

Logs, the original dirty-worktree backup, source-range check and final patch snapshot are under:

`/mnt/truenas-user-home/kupingxin/rl_observation_alignment_implementation_20260921/`

## Explicit remaining boundaries

1. Training/PPO, policy/value encoders, actions/filters, reward profiles, env state/dynamics/observation packing and rendering remain at823a906. New observation fields do not automatically reach the old480-dimensional actor or change its display-valid behavior.
2. Candidate-dependent nearest triangles, image-scale learning, 52-dimensional MANO visual decoder integration, history encoding and offline revisiting remain later work. The pure canonical binding API is provided, not wired into state updates.
3. P17 retains current raw-image batching and depth-review inputs; richer multi-panel source MANO/appearance/temporal context for Astra was not added or quality-validated. Typed evidence assembly and source qualification are now explicit, but image-judgment accuracy is unproven.
4. Camera/depth/HaWoR/ownership/P09 producer math is preserved, not rewritten. New all-frame licensed MANO LBS validation and a separate dense per-surfel provenance archive were not added. RGB matched points carry their pixel/depth-support contracts; source P09 retains its existing representation.
5. Real isolated LightGlue/SAM3D workers, actual raw-video-to-v2 completion, full-video geometry/render inspection, throughput and annotation quality remain unvalidated. These need an explicitly authorized compute target and fresh immutable runtime; CPU fixtures do not substitute for them.
6. Local ICP-only registration as a separately serialized new measurement family is not added here; the existing observed-surface source and pure local RGB/loop processing remain separate from the retained full P14 reference.
7. No claim of physical contact, collision, signed distance or completed V7 annotation parity is made. This delivery repairs observation production/loading/preprocessing mechanisms within the user-limited scope.


## Follow-up: independent observation acceptance (same branch)

The user explicitly selected four acceptance criteria: full v2 readback, actual
preprocessing consumption, independent outputs without PPO, and an explicit list
of pending downstream consumers. Implemented `python -m observations.inspect`
and `EpisodeBundle.load_observations()` with depth streaming, full mask/raster
readback, P09 replay, source MANO/bridge/pad/projection, lazy appearance processing,
typed interaction, RGB reprojection, motion seed, full visual geometry binding,
and independent first-surface queries. It emits NPZ/CSV and data-driven QC images,
not just capability flags. It deliberately does not prepare old solver contact
membership or change policy/render consumers.

Final synthetic interface acceptance: 29 artifacts, 11 groups, 47 actually read
source files rechecked unchanged, 1,556 first-surface query samples, full1,152-face
visual fixture, and 4 review images. Fixture is clearly synthetic, NOT milk or a
licensed MANO/model prediction. Existing real milk is v1 and is correctly refused
rather than silently restamped. Real cold v2/quality verification remains pending.

New tests: 9. Combined regression: 208 discovered, 204 passed, 4 intentionally
skipped. Details, exact CLI, consumers and outputs:
[OBSERVATION_INSPECTION.md](OBSERVATION_INSPECTION.md).
Evidence: `/mnt/truenas-user-home/kupingxin/rl_v2_observation_acceptance_20260921/`.
