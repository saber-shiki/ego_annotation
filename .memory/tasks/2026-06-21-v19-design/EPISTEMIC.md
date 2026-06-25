# EPISTEMIC — V19 Design

## Current model

V19 should be a pipeline/evaluation release, not another V18-style accumulation of script outputs. The design authority should move to a Pi agent harness: the agent owns physical-state judgment, evidence requests, uncertainty assignment, and final artifact writing; scripts remain measurement/render/evaluation instruments.

The user-facing claim should be self-contained physical annotation from raw egocentric video to renderable videos and benchmark reports. The primary implementation route is a Pi SDK harness with GPT-5.5 through the existing `occ` provider in `~/.pi/agent/models.json`, using a custom V19 system prompt to replace the default prompt for the harness session.

The agent is not a replacement for the physical optimizer. The final design requires explicit full-timeline state variables (`H`, camera/head pose, object geometry/pose, visibility, contact, uncertainty) and a robust objective whose state is consumed by the renderer. Agent authority is physical-state selection, measurement planning, and contradiction repair; solver/state authority is numerical consistency and artifact-driving variables.

## Corrected rigid-object mechanism

The prior error was treating manifest `claim_scope`/wording as the source of truth and then trying to make the design rhetorically defensible. That is the wrong model. The source of truth is the artifact-producing mechanism selected by physical state.

Correct rigid-object logic:

1. VLM/agent-harness decides an object instance is rigid over a frame span.
2. That decision commits the pipeline to the rigid-object branch; it is not an optional hypothesis that can be replaced by a point cloud/status label after the fact.
3. TRELLIS completes the per-instance mesh from object crops/masks. Local names include `remote_run_trellis_shape_v3.py`, `build_v18_compact_rigid_trellis_completion.py`, and `microsoft/TRELLIS-image-large`. Observed depth-fused surfaces remain the metric anchor; TRELLIS supplies hidden-surface completion.
4. The completed mesh pose is estimated for every visible frame with mask/depth support. Local stage type: `fit_v18_compact_rigid_object_pose.py` and `v18_compact_rigid_object_pose_fit_report.json`.
5. The pose trajectory enters the temporal factor graph with camera/depth, MANO hand state, contact, occlusion, and nonpenetration terms. The graph corrects pose and hand/object relation jointly.
6. The renderer consumes the corrected rigid mesh pose in overlay/world/side-by-side videos. Occluded/out-of-view frames carry temporal/occlusion uncertainty for the rigid object rather than reverting to visible-surface-only delivery.

Residuals after rigid commitment are not excuses to abandon object pose. They measure pose quality, widen uncertainty, identify repair targets, and drive additional measurements/optimization. If the physical-state decision was wrong, that is a state-classification error to revise explicitly, not a silent fallback.

The current proposal now removes residual early-state ambiguity: visible surfaces are named as measurements/evidence, while `G_i` and `O_{i,t}` preserve branch-selected geometry and corrected rigid/articulated pose trajectories with attached residual uncertainty.

## Corrected benchmark scope

A broad dataset inventory is not an evaluation plan. V19 must have an executable benchmark slice:

- Primary external dataset: HOT3D, 3-5 curated clips.
- Optional secondary dataset: H2O, 2-3 clips.
- Fallback: DexYCB may replace H2O if H2O access/tooling blocks implementation, but H2O and DexYCB must not both be active in initial V19.
- No additional hand-only, 2D occlusion, state-change, or HOI datasets are V19 acceptance targets.
- Baselines are limited to what actually runs on selected clips: V18 v5 on project videos, HaWoR on HOT3D, and official/reference evaluators for active HOT3D/H2O/DexYCB clips. WiLoR/HaMeR can be internal component ablations only if actually used, not full-paper benchmark claims.

The current proposal no longer contains active references to HO3D, Ego-Exo4D, AssemblyHands, FreiHAND, HInt, or Ego4D Hands & Objects; it also avoids `benchmark matrix`/`benchmark suite` framing.

## Supported design claims

- Pi supports the required harness mechanics: `models.json` can register an `occ` provider/model, CLI can select provider/model and override system prompt, and the SDK can create sessions with `DefaultResourceLoader.systemPromptOverride`.
- HOT3D is the primary V19 benchmark because it is the best single anchor for joint egocentric hand/object/head/world claims.
- H2O is the optional secondary benchmark because it supplies egocentric RGB-D hand-object sequences with two-hand interaction and 6D object poses.
- DexYCB is only a fallback for the same secondary slot if H2O blocks; it is not an additional active dataset.
- The representative project raw clips are verified through V16 raw-frame manifests: `task5_tomato_960` uses `/data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_5/20260118_1257_Rec3db6_P0_Sc6ab88_task_5.mp4`; `trash_1050` uses `/data2/egoscale_demo_30h/egoscale_tasks/20260108_1057_Recf94e_P0_S994da4_task_9/20260108_1057_Recf94e_P0_S994da4_task_9.mp4`.

## Key mechanism choices

- Agent-native visual judgment should replace isolated VLM JSON calls. The harness must expose frames, crops, render sheets, depth overlays, 3D snapshots, and benchmark comparison images directly to the agent, and the agent must cite the visual/geometric artifact behind any accepted state change.
- Object discovery should be category-agnostic: sample frames, agent proposes candidate manipulated objects, open-vocabulary detection/segmentation/tracking gathers evidence, and a uniform object schema handles all objects. No tomato/bowl/pot if/else pipeline.
- Object geometry must be per-instance. Retrieved/category/TRELLIS/dataset meshes are priors only until adaptation/fitting to video/depth/dataset-instance evidence; they cannot ground object pose by themselves.
- Object state-type decisions commit to branch-specific state writers. Visible surfaces are evidence/anchors for rigid pose, not a weaker deliverable replacing pose after a rigid decision.
- Runtime and compute placement are design invariants: default path should be same order of magnitude as input duration on server-class compute; heavy inference/reconstruction/rendering should not run locally by accident; long per-instance optimization belongs in offline branches.
- Quantitative evaluation must be small and executable. A fixed few-clip slice is stronger than an aspirational list of datasets that will not run.
- Autoresearch is valid only against the fixed small benchmark slice with held-out clips, predictions before runs, committed configs, and explicit failure ledgers; otherwise it becomes benchmark overfitting or status theater.

## Current status

`docs/pipeline_v19_design_proposal.md` was updated, clean-room reviewed twice, committed as `2796a79 Design V19 agent harness evaluation release`, and pushed. Subsequent corrections: `3ec0106 Clarify V18 tomato rigid scope`, `ac9358b Tighten V19 mechanism and benchmark scope`, and `8820d82 Remove residual V19 ambiguity`. The latest commit removes residual early-state ambiguity and defensive wording after a full re-read. No implementation was started. Unrelated pre-existing dirty files remain unstaged.
