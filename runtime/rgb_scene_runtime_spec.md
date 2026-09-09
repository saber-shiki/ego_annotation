# Standalone RGB scene runtime spec

This is the sole execution specification. Read `runtime/rgb_scene_system_prompt.md`, `runtime/runtime_config.json`, and the supplied immutable `input/runtime_input_contract.json`. Execute only `scripts/run_rgb_scene_stage.py`. Its bundled lower-level producers are dependencies, not alternative entrypoints. No environment provisioning, installs, downloads or code/config changes. All tools are serial; every bash stage call has explicit **timeout=1200**. Preserve the named failure/log and stop on infrastructure or mechanical contract errors. Never poll, wait with sleep, switch devices or retry a failed command. A tool timeout needs process reconciliation before any further managed-shell activity.

## Scope, inputs and assets

One selected rigid object, two genuine MANO hands, full decoded CFR RGB timeline. The frontend emits zero-based decoded IDs; scene/graph consumers join IDs separately from timestamps. No silent truncation or resampling. Aspect ratio is preserved. HaWoR requires square focal; supplied unequal fx/fy is an explicit unsupported sensor/model combination, never averaged. An optional `target_description` is in the prepared input contract; otherwise choose the manipulated rigid object from current images. Optional supplied camera metadata has already been copied and byte-bound; absent calibration triggers camera-conditioned UniDepth estimation with uncertainty.

The config supplies host, physical GPU, pinned Python environments, HaWoR, UniDepth, OWLv2, SAM2, LightGlue, native SAM3D, MANO assets and the offline checkpoint inventories. Physical GPU maps to logical cuda:0. The scene GPU policy is either exclusive or explicitly authorized shared_idle with pinned UUID, utilization at most 5%, and a declared memory floor. Every heavy producer rechecks, including nested native reconstruction. Never relax that policy. Environments, native tool provider and MANO license/assets must be preflighted before execution.

## Command form

Bind `PYTHON` from config `main_python`, `GPU` from config `gpu`, and `RUN` from the launch. Use absolute paths. For every stage:

```
"{PYTHON}" scripts/run_rgb_scene_stage.py --stage {STAGE} --run-root "{RUN}" --runtime-config runtime/runtime_config.json --gpu {GPU}
```

For frontend stages add `--phase {PHASE}`. A verified-resume launch additionally supplies `--resume-contract {RESUME_CONTRACT}` on every command. Only byte-verified successful producers are skipped; a failed or partial producer is not resumable. Do not rewrite decisions in a resumed run. Existing logs and results are immutable.

## Sequence and actual visual decisions

1. `frontend --phase P01`: raw frame manifest at `input/raw_frame_manifest/manifest.json`. Inspect representative images named by that manifest.
2. `camera`: supplied/estimated intrinsics, conditioned full depth, camera binding and full HaWoR MANO surfaces/parameters in `measurements/hand_candidates/hawor_world/hawor_world_hands.npz`.
3. Author `measurements/object_candidates/object_plan_agent.json` from inspected images. Root has exactly one `objects` entry with dynamic `object_id`, `track_id`, `description`, `physical_branch_hypotheses`, `evidence_frames`, `expected_visible_intervals`, `uncertainty_notes`, `detector_text_prompts`, `detector_prompt_frames`, `detector_active_interval: [first_id,last_id]`. Prompts are open-vocabulary text, not manual clicks. Do not manufacture a category-specific planner.
4. `frontend --phase P06` runs OWLv2; `frontend --phase P07` runs SAM2. Inspect current detector and mask reviews for identity, missing support and contamination. Weak local gaps remain uncertain; a different tracked object is a mechanical identity failure.
5. `frontend --phase P08` binds actual MANO/camera into base annotations. `frontend --phase P09-propose` writes `measurements/object_geometry/anchor_candidates/{OBJECT}/anchor_candidate_proposals.json` and `anchor_candidate_review.jpg`.
6. Inspect candidate/raw/mask/metric evidence. Author `state/anchor_decisions/{OBJECT}.json` with `object_id`, `selected_anchor_frame_idx`, `selected_candidate_rank`, `visual_rationale`, `rejected_candidate_observations`, `uncertainties`, `candidate_report_path`. A proposal score is not a model judgment. Run `frontend --phase P09`. Its retained object-owned depth is separate from raw appearance; quarantined support cannot be used for RGB source lifting.
7. Inspect retained geometry and author `state/physical_branch_decisions/{OBJECT}.json` with `object_id`, `physical_branch: rigid`, rationale and uncertainty. If object is not rigid, record the scope blocker rather than run a rigid reconstruction. Author `state/object_selection.json` with `case` from input, selected `object_id`, `track_id`, `anchor_frame` matching the actual decision. Run `frontend --phase P11`, then `geometry`, then `initial-pose`. These are fresh same-run native reconstruction and initial rigid measurements.
8. `refined-pose` runs current-video SuperPoint/LightGlue, retained-source-depth calibrated PnP, time-selected adjacent/multihop/loop edges, robust global SO3 and refreshed observed/RGB six-DoF refinement. The immutable source prior is separate from `state/scene/{OBJECT}/refined_pose.json`. Missing metric frames remain latent uncertain nodes; no copied pose evidence is supplied. Temporal terms use seconds; one retained-metric anchor fixes gauge. Rotation updates may accumulate beyond the small per-step trust region.
9. `appearance` writes `measurements/demo_visual/{OBJECT}/appearance_evidence/appearance_mask_manifest.json` plus a numerical candidate proposal. Inspect actual appearance images/masks and retained metric support, then author `state/appearance_decisions/{OBJECT}.json` with `object_id`, `selected_frame_idx`, `visual_rationale`, `uncertainties`. The selected frame needs available raw appearance and same-frame retained metric depth for scene scale. This model-assisted selection need not equal the numerical proposal or the initial canonical anchor.
10. `native-shape` reconstructs the selected raw appearance with native SAM3D; `shape` replays full native triangles, transforms into the refined canonical gauge, estimates one static image-supported scale and writes `measurements/scene_shape/{OBJECT}/pose_mesh.json`. It never resets to initial poses based on a readiness flag. Image scale is a bounded visual hypothesis, not metric ground truth.
11. Inspect raw RGB, appearance and refined trajectory across visible/hidden intervals. Author `state/agent_interaction_judgments/{OBJECT}.json` with `status: ok`, `interaction_judgments` (non-overlapping intervals per side; each has `target_entity_id: object:{OBJECT}`, `hand_side: left|right`, `frame_start`, `frame_end`, `contact_state: likely_contact|possible_contact|no_contact|unresolved`, evidence/rationale/uncertainty), and `frames` containing every frame ID and `visibility` entries for `object`, `left`, `right`. Each visibility entry has `state: visible|partially_visible|occluded|out_of_frame|unresolved`, `occluder` identity or null, and uncertainty. Claims through occlusion require temporal/depth-order evidence; otherwise unresolved.
12. `visual-fit` executes full MANO LBS, fixed beta, articulation/root and camera-origin similarity, projection priors, temporal regularization and unsigned nearest-triangle fingertip pad proximity. It does not force all fingers into contact or assign inside/outside meaning to generated normals.
13. `export` creates exactly `final/annotation.json` and `final/world.rrd`, with complete object/MANO triangles, camera/K/time, trajectory and embedded source RGB. No annotation MP4 or detail-image renderer. Internal measurements/meshes/fit state remain separate for reproduction. Inspect full-timeline object orientation, both hands, appearance, visibility and identity; counts alone are not quality evidence.

## Decision provenance (required on all five authored judgment kinds)

Add top-level `visual_judgment_provenance`:

```
{"kind":"current_run_visual_judgment","phase":"P05|P09_anchor|P10|appearance_anchor|P17_judgment",
 "run_identity":{"run_id":"...","run_root":"...","input_video_sha256":"...","bundle_manifest_sha256":"..."},
 "evidence_sha256":{"absolute current evidence path":"sha256"}}
```

Copy identity fields from the input contract. Hash current files using pinned Python; only claim visual inspection for images actually read. P05 binds manifest/inspected images; P09_anchor binds proposal/review; P10 binds retained geometry; appearance_anchor binds appearance manifest and inspected RGB/masks; P17_judgment binds refined_pose.json and inspected raw frames. These are actual visual judgments, not fabricated model execution records. Do not copy old predictions or author successful producer receipts yourself.

## Failure and result

Failed producers preserve logs and named `state/runtime_blockers` entries. If failure occurs before a producer can log it, write a create-only blocker naming phase, exact command, error, affected state variable and current evidence, then stop. No automatic repair or fallback reconstruction. No signed collision/nonpenetration claim from this visual scene. Return final pair paths, representative visual observations and genuine remaining uncertainties only after the corresponding commands and inspection occurred.
