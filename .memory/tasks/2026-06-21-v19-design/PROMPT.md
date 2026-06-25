## Objective

Design V19 as a pipeline/evaluation version without starting implementation. V19 should improve the V18 deliverable by changing orchestration, evaluation, object coverage, visualization, and iterative research/evaluation discipline rather than by another aggressive algorithm update.

Concrete deliverable: a detailed V19 design proposal in `docs/pipeline_v19_design_proposal.md`, grounded in current research and the user's comments.

## Workbench

Current status: V18 v5 is the scoped bounded MANO closure artifact under current evidence. V19 design starts from the user critique that V18 is physically useful but not audience-ready, not self-contained enough, not evaluated quantitatively enough, and not agent-native enough.

Current task: research and reason deeply, then update the V19 design document only. Do not implement code, run model inference, or start a new pipeline.

## Context

Relevant existing design file: `docs/pipeline_v19_design_proposal.md`.
Current V18 deliverable root: `/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5/`.
User comments to address:
1. Entry point should be an agent harness, not a script. Separate VLM calls should be replaced by internal judgment of the agent. Implementation route: use GPT-5.5 in Pi agent, override default system prompt, use `occ` provider in `~/.pi/agent/models.json` for test.
2. Pipeline must be self-contained: input any video and get results.
3. Object detection is incomplete; task5 manipulates tomato, bowl, pot, etc., but current detection only detects tomato.
4. Explain why tomato is still point cloud instead of rigid body in V18, and design V19 to present manipulated objects better.
5. Visualization/presentation must be audience-oriented: hand mesh, head/camera pose, manipulated object, key message; avoid internal-state text dumps and ugly line-plot world view. Learn from HaWoR website 3D style and extend it.
6. Quantitative comparison and ablation: find open-source ego datasets with ground truth, including those used by WiLoR, HaMeR, and HaWoR; compare against HaWoR and other baselines.
7. Autoresearch: with harness, algorithm, and benchmark set up, tune iteratively to optimize open-source benchmark performance.

## Task specifications

Research current papers/datasets as needed. Make falsifiable design claims. Distinguish pipeline design, evaluation plan, presentation design, and future implementation routes. Preserve no-implementation boundary.

## Constraints

Do not start implementation. Do not run heavy local inference. Do not use local validator loops. Do not stage unrelated dirty paths. If committing the design, stage only the design doc and task-memory if tracked; task memory is likely untracked/ignored. Use web/current sources for papers/datasets.
