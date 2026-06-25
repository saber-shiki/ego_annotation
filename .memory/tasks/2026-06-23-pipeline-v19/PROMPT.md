## Objective

Implement V19 as a Pi-native physical annotation and evaluation pipeline. Use `docs/pipeline_v19_design_proposal.md` for physical/object/benchmark design, with `docs/v19_run_contract.md` superseding the older SDK-created-session route for harness launch and control. Pi itself is the harness.

V19 is complete only when a Pi command/session with the V19 system prompt can take an egocentric video input, produce full-duration rendered physical annotation artifacts, write render-consumed state/evidence, and run the bounded external evaluation slice. The artifact must show real hand/object/contact/occlusion mechanisms in overlay/world/side-by-side videos; JSON, validators, status fields, or reports are backing evidence only.

## Workbench

User-corrected active Workbench. Follow these items in order; do not substitute prompt scaffolding, state containers, validators, or V18 artifact repackaging for pipeline development. V18 may be mined for reusable code, contracts, and algorithms, but the V19 pipeline must not depend on V18/V17 cached output roots as default inputs.

1. Extract valuable components from the V18 mess into V19-usable components. The extraction is complete only when the reusable mechanisms needed for a fresh-video run are callable without depending on cached V18/V17 artifacts. At minimum this includes fresh raw-frame manifest generation and fresh base annotation/state construction.
2. Orchestrate those components using English as the executable pipeline. The runbook is successful only if a Pi session can follow it on a fresh input video end-to-end, producing regenerated measurements/state/renders under the V19 run root. Concrete commands/scripts, required inputs/outputs, and agent judgment points must be sufficient to run; fake numbered scripts, JSON registries, old cached roots, and local-only plumbing do not count.
3. Enforce rigid body branches. When the agent/workflow decides an object is rigid, the pipeline must execute completion/adaptation, visible-frame pose fitting, factor-graph correction, and corrected mesh-pose rendering from V19-produced measurements. Visible surfaces remain measurements, not replacements for rigid pose.
4. Test and iterate the full pipeline. Run the English-orchestrated pipeline end to end on representative fresh video(s), inspect physical failures, repair the actual mechanism, and rerun until the artifact improves or a concrete missing implementation is exposed.
5. Improve visualization. Make overlay/world/side-by-side videos clear, full-duration, and audience-readable, with visible marks driven by physical state and unresolved uncertainty shown honestly.
6. Run ablation/comparison on open-source data. Use the bounded benchmark scope: HOT3D primary, optional H2O or DexYCB fallback as previously constrained. Compare against relevant baselines/components only for claim families the data supports.
7. Autoresearch. After the pipeline and bounded comparisons exist, run controlled research/optimization loops over fixed clips/metrics, preserving rendered samples, metrics, failure clusters, and causal interpretations.

## Context

Repo root: `/home/yiwen/ego_annotation`.

Design source: `docs/pipeline_v19_design_proposal.md`.

Latest design commit at task start: `8820d82 Remove residual V19 ambiguity`.

V18 baseline artifact: `/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5/`.

V18 baseline manifest: `/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5/v18_current_frontier_interval_mano_artifact_manifest.json`.

Project representative raw videos:
- `task5_tomato_960`: `/data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_5/20260118_1257_Rec3db6_P0_Sc6ab88_task_5.mp4`, 960 frames, 30 fps, 1920x1080.
- `trash_1050`: `/data2/egoscale_demo_30h/egoscale_tasks/20260108_1057_Recf94e_P0_S994da4_task_9/20260108_1057_Recf94e_P0_S994da4_task_9.mp4`, 1050 frames, 30 fps, 1920x1080.

Pi provider route: `~/.pi/agent/models.json` defines provider `occ` and model `gpt-5.5` with image input/reasoning support.

Server / remote execution context recovered from prior V17/V18 runs and scripts, to be re-verified before V19 use:
- A800 SSH target: `yiwen@192.168.11.220`.
- Remote repo root used by V18 jobs: `/mnt/user-home/yiwen/ego_annotation_remote/repo`.
- Remote working root used by setup scripts: `/mnt/user-home/yiwen/ego_annotation_remote/`.
- Large remote output/data root used by V18 jobs: `/mnt/truenas-user-home/yiwen/ego_annotation_outputs`.
- Prior storage observation: `/mnt/user-home` was near/full during V17, while `/mnt/truenas-user-home` had large free capacity; large V19 inputs/outputs should default to truenas-backed paths after a fresh `df -h` check.
- Prior A800 GPU pattern: jobs set `GPU_ID` and export `CUDA_VISIBLE_DEVICES="$GPU_ID"`; V19 must probe current free GPUs before launch and record the chosen GPU in the run manifest.
- Prior environment anchors include `/mnt/user-home/yiwen/ego_annotation_remote/hawor_work/.venv_hawor/bin/python`, `/mnt/user-home/yiwen/ego_annotation_remote/data/sam2.1_hiera_small.pt`, and `/mnt/user-home/yiwen/ego_annotation_remote/trellis_work`; these paths are evidence from earlier runs, not fresh V19 guarantees.
- Prior TRELLIS observation in V18 OPS: the A800 host was reachable and the TRELLIS stack could import via `/usr/bin/python3.10` plus the migrated virtualenv site-packages even though a venv python symlink was broken. Re-check before using TRELLIS for V19.

Installed Pi docs used by the active Pi-native route:
- `/home/yiwen/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent/docs/models.md`
- `/home/yiwen/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent/docs/usage.md`
- `/home/yiwen/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent/docs/prompt-templates.md`

Rigid object branch local anchors:
- TRELLIS route: `scripts/remote_run_trellis_shape_v3.py`, `scripts/build_v18_compact_rigid_trellis_completion.py`, `microsoft/TRELLIS-image-large`.
- Visible-frame pose stage type: `scripts/fit_v18_compact_rigid_object_pose.py`, `v18_compact_rigid_object_pose_fit_report.json`.
- Factor-graph / MANO-object correction anchors include V18 temporal factor-graph state, `scripts/build_v18_mano_object_constraint_state.py`, `scripts/solve_v18_joint_mano_interval_trajectory.py`, and `scripts/run_v18_interval_mano_canonical_artifact.py`.

## Task specifications

The V19 entry point is a Pi command/session with the V19 system prompt loaded. Pi itself is the harness, not a top-level script or wrapper acting as annotation authority. Scripts are measurement, render, optimization, and evaluation instruments. The Pi session owns the evidence loop, physical-state selection, uncertainty assignment, repair decisions, artifact-root contract, and final state handoff to the renderer.

### Parent/runtime/evaluator separation

Keep three roles separate and auditable.

- The parent/development agent edits code, docs, prompt templates, runbooks, task memory, and launch contracts. It may launch or monitor a runtime agent, but it must not manually execute V19 measurement stages, assemble run-root prediction artifacts, choose runtime object/contact decisions, repair prediction outputs, or score GT inside the runtime run. Parent-run component staging is not runtime evidence.
- The runtime agent is a prediction executor. Its input contract is only the raw/input video, a fresh run root, case id, repository code/runbook, and prediction-side sensor metadata needed to run the pipeline. It must not receive HOT3D/H2O/DexYCB GT, evaluator targets, raw-vs-V19 ablation goals, Workbench/task-memory instructions, parent-session concerns, or prompts explaining what the parent must not do. It produces prediction artifacts under the run root: `input/`, `measurements/`, `state/`, `renders/`, and `logs/`.
- The evaluator/ablation phase is separate and starts only after the runtime prediction output is frozen. It may consume GT sidecars, baselines, and evaluator scripts, and must write evaluation outputs without modifying prediction state.

A valid runtime launch must use a curated runtime workspace, not the parent agent's active working directory and not a full project worktree. The runtime workspace may contain only the runtime system prompt, runtime prompt template, runtime-only ontology/runbook, and scripts/config/assets required by that runbook. It must not contain `.memory/`, `AGENTS.md`, parent/development docs, evaluator/benchmark docs, historical task reports, or unrelated project documentation. The runtime agent must run in a dedicated tmux session/window whose purpose is visible to the user, and the launch command must pass only the fixed runtime entrypoint arguments. If any runtime-visible prompt/runbook/ontology mentions Workbench, `.memory/tasks`, OPS/EPISTEMIC, ablation, GT, parent sessions, evaluator roles, or development checklists as runtime instructions, stop and repair the runtime bundle before launching. Workbench item 2 is not closed until a runtime agent launched under this curated-workspace separation follows the runbook end-to-end on a fresh input/run root.

The V19 harness must use a V19-specific system prompt that overrides the generic Pi prompt. The intended route is provider `occ`, model `gpt-5.5`, high or xhigh reasoning, and a narrow tool set. A safe smoke check must be non-interactive, tool-disabled, session-disabled, and isolated from discovered context/skills/extensions.

A completed V19 run must write a stable run directory containing `input/`, `measurements/`, `state/`, `renders/`, `evaluation/`, and `logs/`. The required rendered videos are full-duration `v19_overlay.mp4`, `v19_world.mp4`, and `v19_side_by_side.mp4`; optional story or interactive viewers cannot replace those MP4s.

The renderer consumes `state/`, not private measurement files. Every visible annotation layer must be driven by explicit state variables and must be reproducible from the run directory.

Full-timeline state variables must include MANO hand state, camera/head pose, object geometry state, object/part pose trajectory, masks/depth measurements, visibility/occlusion ownership, contact state, residuals, and uncertainty.

Object discovery must be category-agnostic. Build a roster of manipulated objects, containers, tools, support surfaces, and likely occluders. Use model-produced visual evidence, open-vocabulary detection, segmentation/tracking, depth/SLAM, and agent-native visual review. Do not encode tomato, bowl, pot, trash, color, material, or action phrase as Python branch logic.

Rigid-object logic is mandatory after rigid physical-state selection. If the VLM/agent-harness decides an object instance is rigid over a frame span, the pipeline must run the rigid branch: TRELLIS per-instance mesh completion, visible-frame SE(3)/Sim(3) pose estimation, temporal factor-graph pose correction with camera/depth/MANO/contact/occlusion/nonpenetration terms, and canonical overlay/world/side-by-side rendering of the corrected rigid object state. Visible surfaces are evidence and metric anchors; they are not a weaker deliverable replacing pose after rigid classification.

For `task5_tomato_960`, the correct object-pose mechanism is: rigid decision -> TRELLIS completion -> pose on visible frames -> factor-graph pose correction -> rendered rigid tomato state. Residuals after rigid commitment measure pose quality, uncertainty, and repair targets; they are not an excuse to abandon object pose.

Hand state means metric 3D MANO state with frame semantics, camera/world transforms, visibility, provenance, and uncertainty. Framewise hand detectors, 2D tracks, keypoints, boxes, or overlays are measurements only.

Object pose means reconstructed or adapted per-instance geometry with pose trajectory. Category priors, retrieved meshes, TRELLIS raw outputs, centroids, primitives, boxes, masks, point clouds, and labels cannot ground object pose until fitted/adapted to observed instance evidence and consumed by the state/renderer.

Contact, occlusion, and nonpenetration must be explicit state variables/factors with uncertainty. Weak measurements continue downstream with uncertainty; they do not block artifact production and they do not disappear.

The active benchmark scope is intentionally small. V19 uses 3-5 HOT3D clips as the primary external benchmark. V19 may use 2-3 H2O clips as an optional secondary benchmark. DexYCB may replace H2O only if H2O access/tooling blocks the secondary slot. H2O and DexYCB must not both be active in the initial V19 gate. No additional hand-only, 2D occlusion, state-change, or HOI dataset is part of V19 acceptance.

Baselines are limited to what will run on selected clips: V18 v5 on project representatives, HaWoR on selected HOT3D clips, and official/reference evaluators for active HOT3D/H2O/DexYCB clips. WiLoR and HaMeR may be internal component ablations only if they are actually used in a V19 run.

Clip lists, metrics, and any held-out clips must be fixed before tuning. Autoresearch may optimize only against the fixed small benchmark slice and must record predictions before runs, interventions, metrics, rendered samples, failure clusters, and clean-room review.

Runtime is a design invariant. The default path should produce a first complete full-video artifact in the same order of magnitude as input duration, target roughly `10x` input duration on declared server-class compute, excluding optional external-evaluation runs. Heavy inference, TRELLIS, hand models, SLAM/depth, video segmentation, rendering batches, and benchmarks belong on the declared server/A800 target or an explicit remote target, not accidental local execution.

Acceptance requires full-duration project representative renders, state-to-render provenance, rigid-object branch execution where rigidity is selected, MANO provenance, contact/occlusion evidence or uncertainty, runtime/compute records, bounded benchmark metrics, ablations that change the artifact or a claim-specific metric, and clean-room review that consumes rendered videos as a user would.

## Constraints

Do not treat validators, JSON fields, row counts, reports, status overlays, or clean schemas as V19 progress unless they drive the rendered physical annotation.
Do not substitute visible surfaces, masks, point clouds, centroids, primitives, boxes, labels, or category priors for object pose after a rigid-object decision.
Do not broaden the benchmark list beyond HOT3D plus optional H2O or DexYCB fallback without a design amendment.
Do not run heavy model inference locally unless the local machine is explicitly declared as the intended compute target for that tool.
Do not use sleep, polling loops, or idle waits; use tmux/job sentinels for long-running work.
Do not add silent fallbacks; broken contracts must fail loudly or produce explicit uncertainty states.
Do not use category/name/color/material/action-specific if/else logic as the perception strategy.
Do not stage unrelated dirty files.
Do not use `git add -A`, `git add .`, or broad directory staging.
Do not commit without inspecting the staged diff.
Do not claim completion until the rendered videos and backing state have been consumed as physical annotations.
Do not replace implementation with audits, ledgers, or readiness summaries.
