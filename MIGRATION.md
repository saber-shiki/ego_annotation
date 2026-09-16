# RL-first source ownership and migration

This document describes the active owners after the RL-first refactor. The
implementation baseline is `8b17958a0bccfba4b2951ce50fc9c8bc60a24c31`.
Historical source remains available in Git; production code does not retain
compatibility wrappers at the old `factors`, `gates`, or `optimization` paths.

## Entrypoints

| Entry | Ownership and output |
|---|---|
| `python main.py observe ...` | Runs raw-video observation providers and writes hash-bound `source_observations.json` (`ego.annotation.observation_bundle.v1`). Stops before P14/P15/P18 pose solving, visual MANO fitting, policy inference, and rendering. |
| `python main.py annotate ...` | Compatibility command name for the same observation-only behavior. It does not publish an annotation. |
| `python main.py resume ...` | Resumes only a same-code/same-input observation producer prefix. |
| `python main.py prepare ...` | Builds retained observation-side image/surface/visibility inputs. |
| `python main.py judge ...` | Runs one independent Astra observation/evidence judgment. |
| `python optimize.py legacy-reference ...` | Strict source-manifest reference composition: P14 observed pose, RGB edges/P15 refinement, fresh appearance-native QC, shape binding, and visual fitting. Outputs are create-only reference candidates under the source run; data/train/rollout do not consume them. |
| `python optimize.py observed-pose ...` | Preserved direct observed-surface P14 numerical reference. Its result is downstream of the source bundle and is never an RL data prerequisite. |
| `python optimize.py refine-pose ...` | Preserved RGB/observed P15 reference refinement. |
| `python optimize.py fit-visual ...` | Preserved visual Adam hand/object fitting reference. |
| `python optimize.py fit-formal ...` | Preserved named formal MANO interval solve. |
| `python optimize.py formal-tail ...` | Preserved P15/P18/acceptance/render reference tail. |
| `python train.py ...` | Official skrl 2.1 PPO training over read-only source observation episodes; saves full agent/preprocessor checkpoint and provenance. |
| `python rollout.py ...` | Deterministic no-update checkpoint rollout using the same builders. It evaluates every selected episode exactly once, exports each complete natural termination before reset, and binds candidate JSON, geometry NPZ, and integrity receipts in a top-level manifest. Compatible new data indexes are accepted; training identity stays in provenance. |
| `python render.py policy ...` | Renders an explicit complete policy-candidate NPZ directly. It does not require or forge formal MANO acceptance. |
| `python render.py dual ...` | Preserved dual display-geometry render for an already-frozen legacy package. |
| `python evaluate.py ...` | Preserved HOT3D box, MANO, object, and aggregation schemas. Evaluation remains outside prediction/observation main. |

Installed console scripts are `ego-observe`, `ego-annotate`, `ego-optimize`,
`ego-render`, `ego-evaluate`, `ego-train`, and `ego-rollout`.

## Source-observation manifest

The source manifest is create-only and contains:

- raw video and raw-frame manifest;
- camera calibration/world transforms and conditioned camera-z depth;
- raw HaWoR world archive and MANO bridge parameters/surfaces;
- object plan, detector identity, raw/owned masks, and metric eligibility;
- P09 visible annotations, including the uncertain per-frame
  `reconstructed_geometry_pose` initializer;
- reconstructed shape contract and mesh;
- appearance masks plus the selected independent appearance-anchor decision;
- independent contact/visibility/occlusion judgments and P17 factor rows.

Every artifact path is relative to the run root and byte-hashed. The manifest
sets `contains_optimized_state=false` and `contains_policy_results=false`. It has
no `pose_fits/**`, P15/P18 result, accepted MANO archive, checkpoint, rollout,
or render. `observations.manifest.load_source_observation_manifest` validates the
schema, path containment, roles, and hashes.

## Physical path migration

| Previous owner | Active owner | Notes |
|---|---|---|
| `factors/common.py` | `rewards/terms/common.py` | Pure cost kernels; no reward aggregation or agent dependency. |
| `factors/hand/*` | `rewards/terms/hand/*` | Names, activation inputs, units, and reductions preserved. |
| `factors/interaction/*` | `rewards/terms/interaction/*` | Signed/unsigned authority remains an input contract. |
| `factors/motion/*` | `rewards/terms/motion/*` | Temporal kernels remain pure. |
| `factors/object/*` | `rewards/terms/object/*` | Residual kernels remain independent of skrl. |
| `gates/camera/calibration.py` | `envs/qualification/camera.py` | Camera/image-plane contract qualification. |
| `gates/hand/observation.py` | `envs/qualification/hand_observation.py` | Hand observation support and full-K qualification. H2 remains translation-specific. |
| `gates/hand/acceptance.py` | `envs/qualification/hand_acceptance.py` | Legacy output acceptance, separate from policy candidate export. |
| `gates/object/{geometry,observation,tracking,trajectory}.py` | `envs/qualification/object_*.py` | Evidence/readiness qualification; no action clipping. |
| `gates/output/acceptance.py` | `envs/qualification/output.py` | Complete geometry/timeline validation. |
| `gates/interaction/contact.py` | `actions/filters/contact.py` | Contact evidence interpretation and action-affecting row filters. |
| `gates/interaction/visibility.py` | `actions/filters/visibility.py` | Visibility/quarantine/display filtering. |
| `optimization/{hand,interaction,object,solve,problems}.py` | `legacy_optimize/*` | Numerical loops and reference compositions only. |
| `optimization/common/correspondences.py` | `utils/candidate_geometry.py` | Candidate-dependent geometry/active-set refresh shared by legacy and RL. |
| `optimization/common/solvers.py` | `utils/numerics.py` | Pure registration/numerical helpers used by observations. |
| `optimization/common/variables.py` | `utils/candidate_state.py` | Shared candidate SE(3) transforms. |
| `optimization/hand/mano.py` | `utils/mano.py` | Shared licensed MANO loading/replay and topology. |
| P14 direct observed pose implementation | `observations/object/pose.py` | Callable by the legacy `observed-pose` command; not executed by source main. |
| main numerical functions | `optimize.py` | Existing signatures retained where possible. |
| main render dispatch | `render.py` | Policy candidate render is explicit and acceptance-independent. |

`migration-map.json` contains the machine-readable path map under
`rl_first_refactor` in addition to the older symbol-provenance ledger.

## Invariants preserved

- Reward terms are mathematical costs. Legacy optimization reuses those costs but
  never imports RL episode accounting or reward normalization.
- `rewards/profiles.py` selects one reviewed formal or visual composition;
  `reward_scale` is a single post-objective PPO-return scale and never changes
  selected factor ratios, normalizers, units, activation, or robust loss.
- Soft magnitude hinges remain soft; no new hard clipping replaces them.
- Candidate-dependent correspondence and active-set refresh remains live.
- Generated faces do not gain signed/contact/collision authority through relocation.
- H2 translation support is not generalized into a universal gate.
- Full-MANO physical acceptance and display validity remain separate.
- Policy candidate rendering consumes explicit full hand/object arrays directly;
  a false legacy acceptance flag cannot discard policy actions.
- Training/data/rollout modules must not import `legacy_optimize` or require its
  outputs/success flags.

## Validation boundaries

The grouped numerical parity tests moved with their owners:

```bash
python -m unittest rewards.test_terms envs.test_qualification \
  legacy_optimize.test_legacy_optimize observations.test_observations
```

Manifest and entry split tests are:

```bash
python -m unittest observations.test_manifest tests.test_entrypoints \
  test_pipeline.GroupedContractTests
```

These CPU tests validate ownership, kernels, source-only dispatch, manifest
identity, and explicit policy-state render consumption. They do not establish
RL learning quality, real-video accuracy, or physical acceptance. Those claims
require the integrated skrl checkpoint/reload/rollout smoke and bounded real
prepared observation fixture described in the task acceptance contract.
