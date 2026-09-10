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

A fixed validator-only continuation is the sole exception to starting with `P09_DEPTH_ORDER`. Its launch supplies `validator_revision_contract` at `state/runtime_recovery/P09_DEPTH_ORDER/validator_revisions/{NEW_BUNDLE_MANIFEST_SHA}.json` and a new handoff path distinct from the immutable old `native_recovery_handoff.md`. The runtime validator independently resolves and verifies that sidecar, the terminal failed-actor snapshot, exact one-event recovery prefix, accepted P09_DEPTH_ORDER receipt/outputs/ancestors, old and new bundle identities, and validator-only bundle diff. Start the command list at `P09-propose`; do not rerun P09_DEPTH_ORDER and do not resume the old actor. All new receipts are stamped with the new bundle, original recovery contract, and validator revision SHA.

```
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage frontend --phase P09_DEPTH_ORDER --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU} --recovery-contract "{RUN}/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json"
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage frontend --phase P09-propose --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU} --recovery-contract "{RUN}/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json"
# inspect and author a new P09 anchor decision; never copy the archived anchor
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage frontend --phase P09 --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU} --recovery-contract "{RUN}/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json"
# inspect and author new P10 rigid/object-selection decisions
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage frontend --phase P11 --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU} --recovery-contract "{RUN}/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json"
```

Then invoke stages `geometry`, `initial-pose`, `refined-pose`, `appearance`, `native-shape`, `shape`, `visual-fit`, and `export` in that order, each with the same `--run-root`, `--runtime-config`, `--gpu`, and `--recovery-contract` arguments. Perform the new appearance and interaction decision/inspection pauses at their normal points. Code rejects `P01`, `camera`, `P06`, `P07`, and `P08` in recovery mode. A recovery failure is create-only at `state/runtime_recovery/P09_DEPTH_ORDER/failure.json`; stop rather than retry. The recovery actor handoff is separate at the supplied `state/runtime_recovery/P09_DEPTH_ORDER/native_recovery_handoff.md`; do not overwrite or claim a missing old actor handoff. The final JSON and RRD retain the birth identity and add execution-bundle, recovery-contract, retained-prefix, original-event, and stop identities.

## Generic candidate continuation

A continuation is an isolated create-only namespace below the prepared run. The
parent prepares it (outside prediction execution) with the pinned interpreter:

```bash
"$PYTHON" scripts/prepare_rgb_scene_runtime.py prepare-candidate --run-root "$RUN" --candidate-id "$CANDIDATE" --native-mode new
```

For a ranking-first default omit `--frame`; an explicit `--frame` is allowed
only after inspecting that current RGB/mask and must be a frame in the current
proposal list. Candidate-local P09/P11 decisions must be authored and visually
confirmed with current provenance; source P01--P08 and anchor files are never
copied or relabeled. Run each normal suffix stage with
`--candidate-context "$RUN/candidates/$CANDIDATE/candidate_contract.json"`.
Candidate IDs are create-only and failed/partial candidates stop without retry.

Verified native reuse is explicit and byte-bound. First prepare with
`--native-mode reuse`, then run `prepare_rgb_scene_runtime.py import-native`
with a complete successful native QC, zero-return model execution receipt,
source control contract, source RGB/mask, and model/config/seed identity. The importer writes a derived
QC/path mapping and preserves source wrapper failure evidence. A mesh file
without these matching receipts is not cacheable. `native-shape` consumes only
that imported QC or runs the pinned native producer once for `new`; shape,
posterior, MANO fit, and export stay on the same candidate mesh and frame.
Outputs/events under `candidates/<candidate_id>` are distinct from source
outputs and are included in final identity metadata.


1. `frontend --phase P01`: raw frame manifest at `input/raw_frame_manifest/manifest.json`. Inspect representative images named by that manifest.
2. `camera`: supplied/estimated intrinsics, conditioned full depth, camera binding and full HaWoR MANO surfaces/parameters in `measurements/hand_candidates/hawor_world/hawor_world_hands.npz`.
3. Author `measurements/object_candidates/object_plan_agent.json` from inspected images. Root has exactly one `objects` entry with dynamic `object_id`, `track_id`, `description`, `physical_branch_hypotheses`, `evidence_frames`, `expected_visible_intervals`, `uncertainty_notes`, `detector_text_prompts`, `detector_prompt_frames`, `detector_active_interval: [first_id,last_id]`. Prompts are open-vocabulary text, not manual clicks. Do not manufacture a category-specific planner.
4. `frontend --phase P06` runs OWLv2; `frontend --phase P07` runs SAM2. Inspect current detector and mask reviews for identity, missing support and contamination. Weak local gaps remain uncertain; a different tracked object is a mechanical identity failure.
5. `frontend --phase P08` binds actual MANO/camera into base annotations. `frontend --phase P09_DEPTH_ORDER` uses the active K/half-pixel affines, metric depth, and all full P08 MANO triangles to derive a hash-bound owned SAM2 track plus a typed native-depth classification/metric-candidate raster. It removes only supported first hits clearly in front from appearance. Near-depth, unsupported, invalid-coordinate, unresolved-hand, and no-hit fringe pixels retain appearance but are metric-ineligible. Clear ordinary object body outside validated hand coverage may enter the existing robust depth/confidence filter; no-hit alone never makes it trusted. `frontend --phase P09-propose` and `P09` explicitly consume the eligibility manifest with `--no-exclude-hand-bboxes`; P09 never performs a second blind subtraction.
6. Apply the ranking-first anchor policy below: retain the existing preferred-group/score/frame-ID ordering, inspect its first candidate's actual RGB and conditioning mask, and choose it unless a concrete disqualifying defect is evidenced. Author `state/anchor_decisions/{OBJECT}.json` with `object_id`, `selected_anchor_frame_idx`, `selected_candidate_rank`, `visual_rationale`, `rejected_candidate_observations`, `uncertainties`, `candidate_report_path`. A score does not replace image inspection, but subjective neatness must not override the default ranking. Distinguish global rank from review-panel position in the rationale. Run `frontend --phase P09`. Its retained object-owned depth is separate from raw appearance; quarantined support cannot be used for RGB source lifting.
7. Inspect retained geometry and author `state/physical_branch_decisions/{OBJECT}.json` with `object_id`, `physical_branch: rigid`, rationale and uncertainty. If object is not rigid, record the scope blocker rather than run a rigid reconstruction. Author `state/object_selection.json` with `case` from input, selected `object_id`, `track_id`, `anchor_frame` matching the actual decision. Both P10 files require P10 visual provenance bound to current P09 annotations and at least one current P09-produced owned-mask image. Run `frontend --phase P11`, then `geometry`, then `initial-pose`. These are fresh same-run native reconstruction and initial rigid measurements.
8. `refined-pose` runs current-video SuperPoint/LightGlue, retained-source-depth calibrated PnP, time-selected adjacent/multihop/loop edges, robust global SO3 and refreshed observed/RGB six-DoF refinement. The immutable source prior is separate from `state/scene/{OBJECT}/refined_pose.json`. Missing metric frames remain latent uncertain nodes; no copied pose evidence is supplied. Temporal terms use seconds; one retained-metric anchor fixes gauge. Rotation updates may accumulate beyond the small per-step trust region.
9. `appearance` validates raw-track, owned-track, preprocessor and corrected-P09 identities, then writes `measurements/demo_visual/{OBJECT}/appearance_evidence/appearance_mask_manifest.json` from immutable raw SAM2 masks plus corrected metric annotations. Raw appearance and metric eligibility remain distinct. Preserve this stage's existing appearance ranking and apply the same ranking-first inspection/exception policy below; do not substitute the P09 score. Inspect actual appearance images/masks and retained metric support, then author `state/appearance_decisions/{OBJECT}.json` with `object_id`, `selected_frame_idx`, `visual_rationale`, `uncertainties`. The selected frame needs available raw appearance and same-frame retained metric depth for scene scale. This selection need not equal the initial canonical anchor, but an override of this stage's top-ranked eligible candidate requires concrete evidence, not aesthetic preference.
10. `native-shape` reconstructs the selected raw appearance with native SAM3D; `shape` replays full native triangles, transforms into the refined canonical gauge, estimates one static image-supported scale and writes `measurements/scene_shape/{OBJECT}/pose_mesh.json`. It never resets to initial poses based on a readiness flag. Image scale is a bounded visual hypothesis, not metric ground truth.
11. Inspect raw RGB, appearance and refined trajectory across visible/hidden intervals. Author `state/agent_interaction_judgments/{OBJECT}.json` with `object_id: {OBJECT}`, `status: ok`, `interaction_judgments` (non-overlapping intervals per side; each has `target_entity_id: object:{OBJECT}`, `hand_side: left|right`, `frame_start`, `frame_end`, `contact_state: likely_contact|possible_contact|no_contact|unresolved`, evidence/rationale/uncertainty), and `frames` containing every frame ID exactly once as an integer and `visibility` entries for `object`, `left`, `right`. Each visibility entry has `state: visible|partially_visible|occluded|out_of_frame|unresolved`, `occluder` identity or null, and nonempty uncertainty text. No missing entity/frame judgments are synthesized. Claims through occlusion require temporal/depth-order evidence; otherwise unresolved.
12. `visual-fit` executes full MANO LBS, fixed beta, articulation/root and camera-origin similarity, projection priors, temporal regularization and unsigned nearest-triangle fingertip pad proximity. Standalone display validity combines fitted pose/source MANO validity with these explicit judgments: `out_of_frame` clears the entity; supported finite occluded poses retain uncertainty. Source poses and MANO validity/arrays remain unchanged and separate from display validity. It does not force all fingers into contact or assign inside/outside meaning to generated normals.
13. `export` creates exactly `final/annotation.json` and `final/world.rrd`, with complete object/MANO triangles, camera/K/time, trajectory and embedded source RGB. No annotation MP4 or detail-image renderer. The portable JSON and RRD identity metadata preserve run UUID, raw-video/bundle/input-contract hashes, source timing, supplied calibration hash (null when estimated), and optional target description. Internal measurements/meshes/fit state remain separate for reproduction. Inspect full-timeline object orientation, both hands, appearance, visibility and identity; counts alone are not quality evidence.

## Ranking-first anchor selection policy

- Keep the existing P09 numeric mechanism unchanged: supported non-border single-component candidates are preferred, then score descending, then frame ID ascending. The existing score weights are area 0.24, visible-depth samples 0.16, hand-removal term 0.18, border safety 0.16, extent consistency 0.14, connectedness 0.07, depth stability 0.05. Do not invent replacement weights or a subjective second ranking. Preferred grouping is review priority, not proof of correctness; other frames remain useful observations.
- Default to `ranked_candidates[0]` after inspecting its actual RGB, intended SAM3D mask and metric support. Only override for an evidenced wrong target/ownership, severe blur or exposure destroying useful structure, substantial truncation/occlusion of necessary structure, severe mask contamination, or unusable same-frame metric binding. A local hand-contact notch, a slightly less smooth boundary, a more frontal alternative, or a smaller depth spread alone is not a veto. Weak but usable measurements continue with uncertainty.
- If an override is necessary, inspect alternatives in the original order and select the first usable one. Read both the default and challenger RGB/masks individually at comparable display scale; do not compare a mosaic-only default against an enlarged challenger. Retain original pixel-area information so enlargement does not hide lower input resolution. State the default's specific defect, affected structure, inspected evidence and alternative's costs in the existing judgment fields. Global rank and review-panel rank must not be confused; temporal diversification of a panel does not reprioritize its candidates.
- Unresolved selection conflicts or visible unsupported structures in a reconstructed mesh may warrant a bounded two-candidate control, not unrestricted reruns. Default budget is at most two candidates, one inference each, with identical model/seed/settings and independent outputs; an already completed candidate counts toward the budget. Use only a separately authorized control entry/launch. If none exists, ask the supervisor with the exact alternatives and uncertainty; do not invent producer commands, change seeds, overwrite an anchor, or copy prior predictions. Without additional budget, retain the ranking-based usable choice and its uncertainty rather than blocking all approximate output.
- Judge a control by current image/shape evidence, not watertightness, face count, a single silhouette score or a visually convenient imagined object. Do not fit one candidate differently to make it win. Hidden geometry remains a hypothesis. No category/frame-number special cases.
- Later appearance selection keeps its own existing score (area 0.45, sharpness 0.35, safe border clearance 0.20), eligibility and same-frame metric binding. Apply ranking-first review to that list without mixing the two score definitions. Changing the shape-conditioning anchor requires the existing explicit native-camera/canonical binding; it is not a permission to replace an already-bound scene mesh in place.

## Decision provenance (required on all five authored judgment kinds)

Add top-level `visual_judgment_provenance`:

```
{"kind":"current_run_visual_judgment","phase":"P05|P09_anchor|P10|appearance_anchor|P17_judgment",
 "run_identity":{"run_id":"...","run_root":"...","input_video_sha256":"...","bundle_manifest_sha256":"..."},
 "evidence_sha256":{"absolute current evidence path":"sha256"}}
```

Copy identity fields from the input contract. Hash current files using pinned Python; only claim visual inspection for images actually read. P05 requires the current P01 manifest and at least one manifest-owned current P01 image; P09_anchor requires the selected-object current proposal and its producer-bound review image containing the selected frame; both the physical branch and object selection use P10 and require current P09 annotations plus a current P09-produced owned-mask image; appearance_anchor requires the selected-object current appearance manifest, the selected-frame producer-bound mask and its current P01 raw RGB; P17_judgment binds current recovery refined_pose.json and inspected retained P01 raw frames. These checks bind evidence identity, not an attestation that pixels were seen. Author actual visual judgments, not fabricated model execution records. Do not copy old predictions or author successful producer receipts yourself.

## Failure and result

Failed producers preserve logs and named `state/runtime_blockers` entries. If failure occurs before a producer can log it, write a create-only blocker naming phase, exact command, error, affected state variable and current evidence, then stop. No automatic repair or fallback reconstruction. No signed collision/nonpenetration claim from this visual scene. Return final pair paths, representative visual observations and genuine remaining uncertainties only after the corresponding commands and inspection occurred.
