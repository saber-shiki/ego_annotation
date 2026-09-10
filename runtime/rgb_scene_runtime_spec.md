# Standalone RGB scene runtime spec

This is the sole execution specification. Read `runtime/rgb_scene_system_prompt.md`, `runtime/runtime_config.json`, and the supplied immutable `input/runtime_input_contract.json`. Execute only `scripts/run_rgb_scene_stage.py`. Its bundled lower-level producers are dependencies, not alternative entrypoints. No environment provisioning, installs, downloads or code/config changes. All tools are serial; every bash stage call has explicit **timeout=1200**. Preserve the named failure/log and stop on infrastructure or mechanical contract errors. Never poll, wait with sleep, switch devices or retry a failed command. Owned producers have an 1100-second outer deadline; reconstruction shares a 1000-second inner budget across phases, leaving cleanup/recording margin below the tool deadline. Nested timeout records are in the geometry output `execution.json` and propagated into the outer blocker. Cleanup uses owned process supervision, including detached descendants; any unresolved process forbids retry. A tool timeout needs process reconciliation before any further managed-shell activity.

## Scope, inputs and assets

One selected rigid object, two genuine MANO hands, full decoded CFR RGB timeline. The frontend emits zero-based decoded IDs; scene/graph consumers join IDs separately from timestamps. No silent truncation or resampling. Aspect ratio is preserved. HaWoR requires square focal; supplied unequal fx/fy is an explicit unsupported sensor/model combination, never averaged. An optional `target_description` is in the prepared input contract; otherwise choose the manipulated rigid object from current images. Optional supplied camera metadata has already been copied and byte-bound; absent calibration triggers camera-conditioned UniDepth estimation with uncertainty.

The config supplies host, physical GPU, pinned Python environments, HaWoR, UniDepth, OWLv2, SAM2, LightGlue, native SAM3D, MANO assets and the offline checkpoint inventories. Physical GPU maps to logical cuda:0. The scene GPU policy is either exclusive or explicitly authorized shared_idle with pinned UUID, utilization at most 5%, and a declared memory floor. Every heavy producer rechecks, including nested native reconstruction. Never relax that policy. Environments, native tool provider and MANO license/assets must be preflighted before execution.

## Command form

Bind `PYTHON` from config `main_python`, `GPU` from config `gpu`, and `RUN` from the launch. Use absolute paths. For every stage:

```
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage {STAGE} --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU}
```

For frontend stages add `--phase {PHASE}`. A verified-resume launch additionally supplies `--resume-contract {RESUME_CONTRACT}` on every command. Only byte-verified successful producers are skipped; a failed or partial producer is not resumable. Do not rewrite decisions in a resumed run. Existing logs and results are immutable.

### Fixed P08 depth-order recovery mode

A launch with `mode: p08_depth_order_recovery` supplies the already-applied absolute `recovery_contract`. The immutable birth input contract and original `logs/integration/events.jsonl` remain unchanged. Invalid P09-propose-and-later files have been hash-archived under `state/runtime_recovery/P09_DEPTH_ORDER/invalidated/`; do not read, restore, or reuse them. New producer logs/events are append-only under `logs/recovery/P09_DEPTH_ORDER/`. Every executable command must add:

```
--recovery-contract "{RUN}/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json"
```

The only allowed recovery commands, in order with the same decision/inspection pauses described below, are:

```
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage frontend --phase P09_DEPTH_ORDER --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU} --recovery-contract "{RUN}/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json"
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage frontend --phase P09-propose --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU} --recovery-contract "{RUN}/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json"
# inspect and author a new P09 anchor decision; never copy the archived anchor
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage frontend --phase P09 --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU} --recovery-contract "{RUN}/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json"
# inspect and author new P10 rigid/object-selection decisions
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage frontend --phase P11 --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU} --recovery-contract "{RUN}/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json"
```

Then invoke stages `geometry`, `initial-pose`, `refined-pose`, `appearance`, `native-shape`, `shape`, `visual-fit`, and `export` in that order, each with the same `--run-root`, `--runtime-config`, `--gpu`, and `--recovery-contract` arguments. Perform the new appearance and interaction decision/inspection pauses at their normal points. Code rejects `P01`, `camera`, `P06`, `P07`, and `P08` in recovery mode. A recovery failure is create-only at `state/runtime_recovery/P09_DEPTH_ORDER/failure.json`; stop rather than retry. The recovery actor handoff is separate at the supplied `state/runtime_recovery/P09_DEPTH_ORDER/native_recovery_handoff.md`; do not overwrite or claim a missing old actor handoff. The final JSON and RRD retain the birth identity and add execution-bundle, recovery-contract, retained-prefix, original-event, and stop identities.

## Sequence and actual visual decisions

1. `frontend --phase P01`: raw frame manifest at `input/raw_frame_manifest/manifest.json`. Inspect representative images named by that manifest.
2. `camera`: supplied/estimated intrinsics, conditioned full depth, camera binding and full HaWoR MANO surfaces/parameters in `measurements/hand_candidates/hawor_world/hawor_world_hands.npz`.
3. Author `measurements/object_candidates/object_plan_agent.json` from inspected images. Root has exactly one `objects` entry with dynamic `object_id`, `track_id`, `description`, `physical_branch_hypotheses`, `evidence_frames`, `expected_visible_intervals`, `uncertainty_notes`, `detector_text_prompts`, `detector_prompt_frames`, `detector_active_interval: [first_id,last_id]`. Prompts are open-vocabulary text, not manual clicks. Do not manufacture a category-specific planner.
4. `frontend --phase P06` runs OWLv2; `frontend --phase P07` runs SAM2. Inspect current detector and mask reviews for identity, missing support and contamination. Weak local gaps remain uncertain; a different tracked object is a mechanical identity failure.
5. `frontend --phase P08` binds actual MANO/camera into base annotations. `frontend --phase P09_DEPTH_ORDER` uses the active K/half-pixel affines, metric depth, and all full P08 MANO triangles to derive a hash-bound owned SAM2 track. It removes only supported first hits clearly in front; behind, near, no-hit, and unsupported-depth pixels retain raw appearance support without becoming metric eligible. `frontend --phase P09-propose` consumes that owned track with `--no-exclude-hand-bboxes` and writes `measurements/object_geometry/anchor_candidates/{OBJECT}/anchor_candidate_proposals.json` and `anchor_candidate_review.jpg`; P09 never performs a second blind subtraction.
6. Inspect candidate/raw/mask/metric evidence. Author `state/anchor_decisions/{OBJECT}.json` with `object_id`, `selected_anchor_frame_idx`, `selected_candidate_rank`, `visual_rationale`, `rejected_candidate_observations`, `uncertainties`, `candidate_report_path`. A proposal score is not a model judgment. Run `frontend --phase P09`. Its retained object-owned depth is separate from raw appearance; quarantined support cannot be used for RGB source lifting.
7. Inspect retained geometry and author `state/physical_branch_decisions/{OBJECT}.json` with `object_id`, `physical_branch: rigid`, rationale and uncertainty. If object is not rigid, record the scope blocker rather than run a rigid reconstruction. Author `state/object_selection.json` with `case` from input, selected `object_id`, `track_id`, `anchor_frame` matching the actual decision. Run `frontend --phase P11`, then `geometry`, then `initial-pose`. These are fresh same-run native reconstruction and initial rigid measurements.
8. `refined-pose` runs current-video SuperPoint/LightGlue, retained-source-depth calibrated PnP, time-selected adjacent/multihop/loop edges, robust global SO3 and refreshed observed/RGB six-DoF refinement. The immutable source prior is separate from `state/scene/{OBJECT}/refined_pose.json`. Missing metric frames remain latent uncertain nodes; no copied pose evidence is supplied. Temporal terms use seconds; one retained-metric anchor fixes gauge. Rotation updates may accumulate beyond the small per-step trust region.
9. `appearance` validates raw-track, owned-track, preprocessor and corrected-P09 identities, then writes `measurements/demo_visual/{OBJECT}/appearance_evidence/appearance_mask_manifest.json` from immutable raw SAM2 masks plus corrected metric annotations. Raw appearance and metric eligibility remain distinct. Inspect actual appearance images/masks and retained metric support, then author `state/appearance_decisions/{OBJECT}.json` with `object_id`, `selected_frame_idx`, `visual_rationale`, `uncertainties`. The selected frame needs available raw appearance and same-frame retained metric depth for scene scale. This model-assisted selection need not equal the numerical proposal or the initial canonical anchor.
10. `native-shape` reconstructs the selected raw appearance with native SAM3D; `shape` replays full native triangles, transforms into the refined canonical gauge, estimates one static image-supported scale and writes `measurements/scene_shape/{OBJECT}/pose_mesh.json`. It never resets to initial poses based on a readiness flag. Image scale is a bounded visual hypothesis, not metric ground truth.
11. Inspect raw RGB, appearance and refined trajectory across visible/hidden intervals. Author `state/agent_interaction_judgments/{OBJECT}.json` with `object_id: {OBJECT}`, `status: ok`, `interaction_judgments` (non-overlapping intervals per side; each has `target_entity_id: object:{OBJECT}`, `hand_side: left|right`, `frame_start`, `frame_end`, `contact_state: likely_contact|possible_contact|no_contact|unresolved`, evidence/rationale/uncertainty), and `frames` containing every frame ID exactly once as an integer and `visibility` entries for `object`, `left`, `right`. Each visibility entry has `state: visible|partially_visible|occluded|out_of_frame|unresolved`, `occluder` identity or null, and nonempty uncertainty text. No missing entity/frame judgments are synthesized. Claims through occlusion require temporal/depth-order evidence; otherwise unresolved.
12. `visual-fit` executes full MANO LBS, fixed beta, articulation/root and camera-origin similarity, projection priors, temporal regularization and unsigned nearest-triangle fingertip pad proximity. Standalone display validity combines fitted pose/source MANO validity with these explicit judgments: `out_of_frame` clears the entity; supported finite occluded poses retain uncertainty. Source poses and MANO validity/arrays remain unchanged and separate from display validity. It does not force all fingers into contact or assign inside/outside meaning to generated normals.
13. `export` creates exactly `final/annotation.json` and `final/world.rrd`, with complete object/MANO triangles, camera/K/time, trajectory and embedded source RGB. No annotation MP4 or detail-image renderer. The portable JSON and RRD identity metadata preserve run UUID, raw-video/bundle/input-contract hashes, source timing, supplied calibration hash (null when estimated), and optional target description. Internal measurements/meshes/fit state remain separate for reproduction. Inspect full-timeline object orientation, both hands, appearance, visibility and identity; counts alone are not quality evidence.

## Decision provenance (required on all five authored judgment kinds)

Add top-level `visual_judgment_provenance`:

```
{"kind":"current_run_visual_judgment","phase":"P05|P09_anchor|P10|appearance_anchor|P17_judgment",
 "run_identity":{"run_id":"...","run_root":"...","input_video_sha256":"...","bundle_manifest_sha256":"..."},
 "evidence_sha256":{"absolute current evidence path":"sha256"}}
```

Copy identity fields from the input contract. Hash current files using pinned Python; only claim visual inspection for images actually read. P05 requires the current P01 manifest and at least one manifest-owned current P01 image; P09_anchor requires the selected-object current proposal and its producer-bound review image containing the selected frame; P10 binds retained geometry; appearance_anchor requires the selected-object current appearance manifest, the selected-frame producer-bound mask and its current P01 raw RGB; P17_judgment binds refined_pose.json and inspected raw frames. These checks bind evidence identity, not an attestation that pixels were seen. Author actual visual judgments, not fabricated model execution records. Do not copy old predictions or author successful producer receipts yourself.

## Failure and result

Failed producers preserve logs and named `state/runtime_blockers` entries. If failure occurs before a producer can log it, write a create-only blocker naming phase, exact command, error, affected state variable and current evidence, then stop. No automatic repair or fallback reconstruction. No signed collision/nonpenetration claim from this visual scene. Return final pair paths, representative visual observations and genuine remaining uncertainties only after the corresponding commands and inspection occurred.
