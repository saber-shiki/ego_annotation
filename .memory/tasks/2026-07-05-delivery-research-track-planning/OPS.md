# OPS — append-only

## 2026-07-05 — Task initialized
Stakeholder feedback converted into delivery/research split planning task. First-wave subagents planned: minimal modules, API-serving research, correctness fixes, delivery output format, self-consistency metrics, v19 bottleneck research. Dependent waves: delivery auto-research guide + HOI outputs/metrics, research auto-research guide + RL factor-graph reasoning, end-to-end distillation, then synthesis.

## 2026-07-05 — First-wave subagents launched
Parallel async run fc1a7524-8af7-4606-b164-4bdfa1ccacfd launched for independent topics: 01 minimal delivery modules, 02 PyTorch/API serving research, 03 blunt correctness fixes, 04 delivery API output format, 05 delivery self-consistency metrics, 07 v19 bottlenecks/HOI research. Output directory: .memory/tasks/2026-07-05-delivery-research-track-planning/subagents/.
Dependent plan:
- Wave 2 after 04+05(+07): subagent 6 delivery auto-research guide (needs 4/5), subagent 8 HOI outputs/metrics (needs 4/5/7).
- Wave 3 after 6+8(+7): subagent 9 research auto-research guide, subagent 10 RL/factor-graph reasoning.
- Wave 4 after 10(+7/8): subagent 11 end-to-end distillation.
- Final: Claude-opus synthesis assistant + main synthesis into two task-definition packs and discussion memo.

## 2026-07-05 — First-wave partial consumption
Completed first-wave outputs 02 and 05 consumed. Implications recorded in EPISTEMIC: delivery serving must be video-aware Ray-first, Triton later; throughput arithmetic is 59.5 realtime streams/module for 10k video-h/week; GT-free camera metrics cannot certify 5mm head/camera without GT/fiducial/IMU; full-rate SAM2 is likely throughput blocker.

## 2026-07-05 — Dependent wave partially launched
Subagent 6 launched async (29a6cd3b-54d4-4a73-b6f3-f989b7377262) after outputs 04 and 05 landed. It will produce delivery-track auto-research guide using API output schema + self-consistency metrics + web research. Waiting on first-wave outputs 03 and 07 before launching subagent 8.

## 2026-07-05 — Subagent 6 retry
Initial communicator run 29a6cd3b-54d4-4a73-b6f3-f989b7377262 failed with upstream 400 before content. Retried as communicator on nearby Claude Opus route: 1c555144-b741-406c-952f-823988bc0112. First-wave still waiting on outputs 03 and 07 before subagent 8.

## 2026-07-05 — First-wave complete; subagent 8 launched
First-wave run fc1a7524-8af7-4606-b164-4bdfa1ccacfd completed. Outputs 01/02/03/04/05/07 saved under subagents/. Key evidence: tomato drift quantified (p95 31.6/24.5px, fallback ~100px), delivery minimal modules D1-D11, Ray-first serving design, delivery schema v1, metrics suite, v19 HOI bottleneck taxonomy and E1-E9 research experiment plan. Subagent 8 launched as 2fdf43f7-70d7-427b-bd95-44af47ee1e3f. Subagent 6 retry still running.

## 2026-07-05 — Subagent 6 integrated; subagent 8 OOM and relaunch
Subagent 6 completed and wrote `subagents/06_delivery_autoresearch_guide.md`. Mechanism added to delivery plan: auto-research requires a protected vector evaluator over hand/camera/drift/caption/throughput plus visual-veto; a single scalar would optimize gameable self-consistency proxies. Updated `DELIVERY_TRACK_TASK_PACK_DRAFT.md` accordingly.

Subagent 8 run `2fdf43f7-70d7-427b-bd95-44af47ee1e3f` failed from Node heap OOM after producing only a partial tool-call stream. No target file was written. Salvaged mechanism-level implications from the partial stream: `ego.research_hoi` 0.x extension, geometry epochs, per-DOF observability, contact posterior, full-timeline contact hypotheses, graph liveness/stale-join/gauge metrics, and contact thresholds grounded in 30.48 mm source-gap uncertainty. Relaunched constrained subagent 8 as `75fa11b0-1702-452c-b075-7e9efa77c30b` using communicator/zai with concise output target `subagents/08_hoi_output_and_metrics.md`.

Created `RESEARCH_TRACK_TASK_PACK_DRAFT.md` from subagent 7 plus salvaged subagent 8 mechanisms; slots remain for 8b/9/10/11.

## 2026-07-05 — Subagent 8b artifact normalized
Constrained subagent 8b run `75fa11b0-1702-452c-b075-7e9efa77c30b` failed only acceptance wrapping (`criterion-2` not reported), but wrote a complete 255-line HOI schema/metrics artifact to `subagents/08b_relaunch_summary.md`. Copied it to canonical dependency path `subagents/08_hoi_output_and_metrics.md` for downstream wave 3. Content includes `org.ego.research.hoi` / `ego.research.hoi` 0.1.0, isolated extension file layout, full-duration HOI renders, object/geometry/pose/rigidity/contact/occlusion/hand-correction tables, GT and GT-free metrics, validation protocols, uncertainty boundaries, and promotion gate.

## 2026-07-05 — Wave 3 launched
Launched wave 3 parallel subagents as run `ef8663cf-a5f6-497b-bf9a-d5feb7a6530f`: subagent 9 (`communicator`, `zai/glm-5.2`) writes `subagents/09_research_autoresearch_guide.md`; subagent 10 (`theorist`, `zai/glm-5.2`) writes `subagents/10_rl_factor_graph_approximation.md`. Both are constrained to file outputs and no edits beyond output capture.

## 2026-07-05 — Wave 3 substantive implications integrated
Subagent 9 and 10 outputs were usable despite acceptance-wrapper failures. Integrated their substantive claims into `RESEARCH_TRACK_TASK_PACK_DRAFT.md`: research auto-improvement is blocked until a protected HOI evaluator exists; fixed-slice-only wins are overfit; proxy-only wins are capture; RL is only an inference-side approximator of validated graph objectives and cannot supply missing measurements. Output files are `subagents/09_research_autoresearch_guide.md` and `subagents/10_rl_factor_graph_approximation.md`.

## 2026-07-05 — User correction to delivery problem model
User rejected prior summaries that over-centered renderer/QC, SAM2, and claim-policing language. Corrected `DELIVERY_TRACK_TASK_PACK_DRAFT.md` and `EPISTEMIC.md`: delivery's main body is numeric outputs; renderer is QC/demo only; SAM2 is not in the default delivery path when HOI is skipped; reaching ~5mm head/camera requires a metric pose source plus calibration/evaluator, not RGB-only monocular inference.

## 2026-07-05 — Removed invented local-VLM delivery option
User identified that local VLM deployment was never part of the requirement. Corrected delivery plan and epistemic model: captioning throughput is counted as existing action-caption alignment or external/agent caption-call throughput, not local GPU-hours or vLLM/TGI deployment.

## 2026-07-06 00:11 — Public API endpoint naming correction
User identified endpoint names such as `/v1/delivery/jobs` as a category error: delivery/research are internal planning tracks, not customer API concepts. Updated current planning artifacts to use domain-facing names: `POST /v1/annotation-jobs`, base schema `ego.annotation.output`, and HOI extension namespace `org.ego.hoi` / `ego.hoi` in task packs. Preserved historical subagent outputs as evidence rather than rewriting them. Added project invariant that public API endpoint paths must use product/domain nouns, not internal track labels.

## 2026-07-06 00:28 — Reframed 5mm as positive optimization target
User identified that delivery planning still negated the 5mm goal by over-emphasizing unsupported claim boundaries. Updated the delivery task pack to define uniform 5mm as the ideal across head/camera, hand wrist/root, all-joint MPJPE, hand surface/MPVPE, projection, visibility, and jitter. Rewrote head/camera and hand sections so measurement limits select interventions and reported frontier, not target deletion. Updated EPISTEMIC and SYNTHESIS_WORKING accordingly; added project invariant that accuracy targets are optimization objectives.
