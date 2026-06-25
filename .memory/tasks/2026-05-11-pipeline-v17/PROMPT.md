## Objective
// State the overall goal, final deliverable, and what done means.

Build an egocentric loco-manipulation video annotation pipeline that improves over open-source baselines in accuracy and physical consistency.

The annotation target is:

1. Head camera pose.
2. MANO hand pose and keypoints.
3. Semantic captioning of the video.
4. Manipulated object pose, where object pose means reconstructed manipulated-object geometry over time, not a centroid, sphere, bounding box, category primitive, or visual patch.

The target accuracy is about 5mm where measurable. Ground truth is unavailable for the target data, so accuracy claims need sanity checks, physical consistency, and representative evidence instead of metric overfitting.

Done means a full raw-video pipeline version that produces:

1. A full-duration video overlaying MANO hand annotation and object annotation.
2. A full-duration 3D animation of head camera, MANO hands, and manipulated objects in world coordinates.
3. A full-duration side-by-side presentation of annotated video and reconstructed 3D with semantic captions.

For v16 and later, every deliverable must have the same frame count and duration as the original raw video. Short windows, selected-frame renders, contact slices, contact sheets, and debug clips are QC artifacts only.

## Workbench
// Maintain the short-term steering state: current status, selected next tasks, observed failures, and open blockers.

Current version: V17.

Current status: V17 is formally FAILED and closed as of 2026-06-12. Further work moves to V18. V17 evidence may be inspected but V17 compute should not continue.

Former status: V17 was open. The current readiness flags remain false:

- v3_solver_complete=false
- annotation_ready=false
- deliverable_ready=false
- accuracy_target_met=false
- object_geometry_complete=false
- object_pose_requirement_met=false
- rigid_pose_requirement_met=false

The user corrected the working model: progress should be measured as pipeline movement, not by row counts or local diagnostic improvements. Measurement trust is essential, but row acceptance belongs in the evidence layer. It can justify a solver decision, expose a measurement bug, or falsify a claim. It must not become the headline progress metric.

The pipeline dependency model is:

1. Full-video annotation requires object geometry and pose over the full interval.
2. It requires MANO hand state in the same metric camera/depth frame.
3. It requires contact ownership tying the correct hand to the correct manipulated object.
4. It requires one solver or coherent optimization path that can absorb noisy measurements, reject bad measurements, and expose systematic failures without hand-tuning every frame.

Current V17 progress and epistemic state:

- Hand side: solved-mechanism state. Interior-owned full-residual graph (3ff2121) closed the depth-ownership question; corrected hand states are baked into full-timeline annotations with solver-camera intrinsics (98e8620, a659a8c): 1,926 baked + 1,841 kept-prior hands, spans 11.6/13.5cm, visual projection QC passed.
- Object side: full-interval BundleSDF jobs per contiguous active segment (86895b0..9db4da1, 9 jobs all solver-ready). First results: tomato peel ACCEPTED full-interval (217f, IoU 0.84); merged-gap faucet job falsified (split jobs queued); tomato 649f and trash jobs running/queued on A800 pass-2.
- Deliverable chain implemented end-to-end: multi-object world mesh archive (c248030) -> deliverable-state manifests (0588ba9) -> full-duration render driver with honest partial-state banner (c363625). Smoke render (tomato, peel-only stream) validates the path.
- Automation armed in tmux session ego_annotation: remote pass-2 queue -> outsync chain (sync->eval->archive->state) -> final window (state->render both cases). Scripts preserved in .memory/local/.
- Known open mechanisms: camera-model triplet (V16 nominal/DROID world, UniDepth metric, BundleSDF rectified) bridged by adapters, not unified — V18 design input; bag deformation may reject rigid meshes (expected, evidence not failure); contact ownership still zero factor-ready.

- Full-video requirement: V16 produced full-length trash and tomato renders, but user review found single-object state, missing objects, bad object-hand states, tomato deformation, and weak world rendering. V16 is packaging evidence, not quality closure.
- Multi-object evidence: V17 added VLM object plans, SAM2 multi-object masks, visible surfaces, material tracks, partial material/surface diagnostics, legacy/local contact-patch QC, and short BundleSDF reconstructions. These are evidence layers, not complete object pose or physical contact factors.
- Object geometry state: accepted BundleSDF reconstructions cover short pink-lid trash-can segments only. Current depth-contact audit found zero shared-depth/contact-ready frames for those accepted meshes, so they do not attach to the current MANO/contact state. Full-interval, contact-compatible manipulated-object geometry is still absent.
- Contact ownership state: image-plane hand/object contact evidence exists, but metric hand depth, object depth, and object geometry disagree. Physical contact factors are not ready.
- Hand state: RTMLib, HaMeR, HaWoR, VLM-box hand repair, temporal depth graphs, MANO refits, relinearized graphs, and full-residual graphs have been tested. The 2026-06-12 depth-owner diagnostic chain found the dominant residual mechanism: UniDepth pixels at hand silhouette boundaries (depth-edge bleed) contaminated both depth factors and the all-pixel acceptance predicate, violating the depth-edge visibility invariant. The interior-owned full-residual hand graph (commit 3ff2121) re-solves ray shifts against edge-band-excluded UniDepth factors: interior-compatible variable rows 1,270/1,926; interior-predicate accepted hand rows 1,996 (trash 1,056, tomato 940) vs 1,072 prior legacy. Remaining hand-side residual classes: 504 depth-tail, 74 hand-in-front, 49 hand-behind, 29 projection-untrusted rows. Open hand-side caveat: the interior predicate's validity rests on edge-band parameters (7px window, 0.10m range, 6px dilation) not yet validated against an independent hand-silhouette source.

Current highest-leverage blocker:

Full-interval manipulated-object mesh reconstruction and contact-compatible object ownership. The hand depth-observation ownership mechanism is solved pending edge-band validity; accepted BundleSDF meshes remain short-segment with zero shared-depth/contact-ready frames, so object geometry/pose and contact factors are the binding dependency for V17 closure.

High-priority blockers in dependency order:

1. Full-interval manipulated-object mesh reconstruction and pose for active objects.
2. Contact-compatible object ownership linking the correct hand, object, depth field, and reconstructed geometry under the interior-owned depth model.
3. Hand-side residual cleanup: 656 interior-incompatible rows and edge-band predicate validation against an independent hand-silhouette source.

Immediate next actions:

1. Stop treating every suspicious measurement as a blocker before the next pipeline iteration.
2. Design the next object-side intervention: full-interval object geometry (extend BundleSDF segment coverage or replace backend) and contact ownership that consumes the interior-owned hand/depth state.
3. Falsify it on `trash_1050` and `task5_tomato_960` with full-interval object pose/geometry checks. Keep readiness flags false unless full-video object/contact state also closes.
3. Timebox any measurement diagnostic. It must end in one of three decisions: tolerate as solver noise, repair as a measurement bug, or bypass as irrelevant to the current bottleneck.
4. If continuing the currently untracked depth-owner diagnostic, first answer this yes/no question: does it choose between depth-observation ownership, camera-depth calibration, MANO surface/pose ownership, or object-contact ownership? If no, leave it untracked and do not cite it.
5. Report progress as: pipeline state, highest-leverage blocker, next intervention, and expected falsification signal. Keep row counts as supporting evidence only.

Current uncommitted local state:

- None. The former untracked depth-owner diagnostic draft was verified, executed, and committed (bd72f09); follow-ups: edge-ownership counterfactual (2cf1827) and interior-owned full-residual hand graph wired into the joint contract (3ff2121).

## Context
// List the code, docs, resources, and prior evidence the agent must consult or may optionally use.

Repository: `/home/yiwen/ego_annotation`

Required local reads before continuing:

- `PROMPT.md`
- `AGENTS.md`
- `LOG.md`
- `docs/pipeline_v17.md`
- Current git status

Raw data:

- `/data2/egoscale_demo_30h/`

Important output roots:

- `docs/pipeline_v17.md` is the authoritative full evidence-root list. Check it before rerunning or duplicating V17 diagnostics.
- `/data2/ego_annotation_outputs/v16_full_pipeline/`
- `/data2/ego_annotation_outputs/v17_measurement_store/`
- `/data2/ego_annotation_outputs/v17_full_state/`
- `/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/`
- `/data2/ego_annotation_outputs/v17_joint_solver_problem/`
- `/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces/`
- `/data2/ego_annotation_outputs/v17_geometry_source_audit/`
- `/data2/ego_annotation_outputs/v17_object_geometry_factor_problem/`
- `/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed/`
- `/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/`
- `/data2/ego_annotation_outputs/v17_geometry_reconstruction_results/`
- `/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit/`
- `/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap/`
- `/data2/ego_annotation_outputs/v17_hand_metric_depth_state/`
- `/data2/ego_annotation_outputs/v17_hand_depth_repair_graph/`
- `/data2/ego_annotation_outputs/v17_relinearized_residual_object_contact_state/`
- `/data2/ego_annotation_outputs/v17_relinearized_residual_factor_coverage/`
- `/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph/`
- `/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph_pose/`
- `/data2/ego_annotation_outputs/v17_full_residual_surface_tail_diagnostic/`
- `/data2/ego_annotation_outputs/v17_full_residual_depth_owner_diagnostic/`
- `/data2/ego_annotation_outputs/v17_depth_edge_ownership_counterfactual/`
- `/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph/`

Representative raw-video cases:

- `trash_1050`
- `task5_tomato_960`

Compute resources:

- Local 3080 for relatively light workloads.
- A800 server via WireGuard: `192.168.11.220`, user `yiwen`.
- 4090 server via WireGuard: `192.168.9.220`, user `yiwen`.
- Use tmux on remote servers.
- Check GPU availability before launching heavy jobs.

Environment:

- Use `.venv/bin/python` for V17 scripts. System Python lacks `cv2`.
- Pyright path observed in prior work: `/home/yiwen/.npm-global/bin/pyright`.
- General OpenAI API key: use `OPENAI_API_KEY` from `.env` with caution.
- Lower-cost Responses/agent harness key: use `OCC_BASEURL` and `OCC_API_KEY` from `.env`; intended model provider includes `gpt-5.5`.

Important corrected facts:

- The handoff template used here is `/home/yiwen/PROMPT.template.md`; `/home/yiwen/PROMPT.md.template` was absent.
- Prior V3 design named a full-video joint hand-object-camera-depth-contact factor graph. Earlier graph artifacts were real component graphs, but they did not complete the V3 full joint solver.
- v2-v15 have no true deliverables under the full raw-video rule. Their clips and renders are QC evidence.
- Starting at v16, every version must be a complete full-video pipeline version with an upfront design document.

## Task specifications
// Define the relatively stable task requirements, expected behavior, edge cases, and verification criteria.

Constitution of work:

1. Do not substitute an easier representation for the required component. Object pose requires reconstructed manipulated-object geometry.
2. Do not encode visual case variation with object-class, color, material, or action if/else branches.
3. Use open-source vision models, open-vocabulary detectors, segmentation/tracking models, and LLM/VLM calls to produce object plans and segmentation evidence.
4. Keep downstream reconstruction, filtering, contact reasoning, and rendering category-agnostic unless a physical domain discontinuity is explicitly represented in data.
5. Pipeline planning and implementation must interleave. A version needs an upfront design, then implementation and evaluation.
6. Do not close a version from component evidence, a short window, selected frames, or a partial render.
7. Do not spend days optimizing local measurement acceptance if a robust solver can absorb the noise or if the measurement does not change the pipeline-level decision.

Pipeline iteration rule:

1. Implement a complete version path for representative full raw videos.
2. Inspect the outputs, including visual render quality and physical plausibility.
3. Decide the highest-leverage blocker at pipeline level.
4. Research and design the next intervention only for that blocker.
5. Run the next full-pipeline iteration.
6. Use diagnostics to explain failures, not to replace pipeline movement.

Measurement decision rule:

For each suspicious measurement source, decide:

1. Tolerate as solver noise if robust losses, priors, and switch variables can absorb it.
2. Repair as a measurement bug if it changes the pipeline-level decision or violates an invariant such as coordinate frame, scale, visibility, or identity.
3. Bypass if it is irrelevant to the current bottleneck.

Progress reporting rule:

Report:

1. Pipeline state.
2. Current highest-leverage blocker.
3. Next intervention.
4. Expected falsification signal.
5. Evidence limits.

Do not report progress as rows accepted, factors ready, or residual counts except as supporting evidence for a pipeline decision.

Verification requirements:

- For code changes: run `.venv/bin/python -m py_compile` on edited Python files.
- Run pyright on edited Python files where practical.
- Run `git diff --check`.
- Regenerate only the artifacts necessary to verify the claim.
- After regenerating V17 artifacts, audit `v3_solver_complete=false`, `annotation_ready=false`, `deliverable_ready=false`, `object_geometry_complete=false`, and `object_pose_requirement_met=false` in the joint solver summary and any rendered artifact manifests unless the work truly closes those requirements.
- Inspect user-facing videos/sheets when render quality or demo quality is part of the claim.
- Do not claim completion from tests or JSON equality alone.

Result demo requirement:

- Planned form is side-by-side annotated video plus 3D world reconstruction with real-time text annotations.
- The 3D presentation must be stakeholder-facing and visually legible, not diagnostic-looking.
- Include local close-up views of dexterous manipulation where needed.

## Constraints
// Define hard rules the agent must never violate while executing the task.

Never use blocking sleeps to wait for long-running jobs. Launch jobs in tmux, then immediately continue with parallelizable work: reasoning, ledger updates, next-artifact preparation, wiring, docs, or review. Poll job state only as a quick non-blocking check between other actions. Arm dependent follow-up steps inside tmux (e.g., `cmd1 && cmd2` chains) instead of waiting in the foreground.

Do not close V17 until full-video deliverables are real and supported by the required hand, object, camera, contact, and caption states.

Do not turn QC artifacts into deliverables.

Do not claim object pose from centroids, primitives, visual patches, or partial short-window reconstructions.

Do not allow measurement tables, row counts, or report status names to become the progress narrative.

Do not spend unbounded time improving local measurement accuracy. Exercise judgment: distinguish measurement error that a solver should tolerate from a mistake that invalidates the pipeline state.

Do not use silent fallbacks. Missing evidence or contract violations should surface as explicit false-readiness, errors, or rejected states.

Do not use category-specific branches as a perception strategy.

Do not create a new top-level version for bug fixes, threshold changes, renderer fixes, server setup, or local mechanism studies. Those belong inside the current version.

Keep `PROMPT.md` and `LOG.md` out of commits. They are ignored by `.gitignore`.

Before any durable code/doc commit:

1. Classify the working tree.
2. Stage only task-owned source/doc paths.
3. Read the staged diff.
4. Commit one coherent logical change.

Lessons and working style corrections:

1. Measurement trust is essential, but it is not the progress metric.
2. Row-count framing confused the user and hid pipeline status.
3. The user needs pipeline-level logic: what dependency is blocked, why it matters, what action will falsify or unblock it.
4. Artifact-honesty work should be capped. Mark incomplete artifacts as QC early, then return to solver/pipeline iteration.
5. If a diagnostic will only produce a table and no pipeline decision, do not do it.
6. Alternate between hand and object only when the dependency being unblocked is explicit.
7. Always explain whether an issue is a measurement bug, tolerable measurement noise, or a real model/pipeline deficiency.
8. Keep scientific judgment ahead of implementation churn.
