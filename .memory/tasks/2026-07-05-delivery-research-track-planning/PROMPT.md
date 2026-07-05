## Objective
Produce a two-track plan after stakeholder feedback: a delivery track that drops current HOI/factor-graph optimization and ships a usable high-throughput API pipeline for head/camera + hand accuracy + fine-grained semantic video captioning, and a research track that continues HOI/physical-state extraction toward accurate non-end-to-end pipeline and future distillation.

## Workbench
1. Orchestrate eleven requested specialist subagents with dependency-aware waves.
2. Synthesize their outputs into new task-definition packs for delivery and research tracks.
3. Discuss the proposed split, deliverables, metrics, APIs, and unresolved user decisions with the user.

## Context
Repo root: /home/yiwen/ego_annotation
Demo pack: /data2/ego_annotation_outputs/demo_pack_20260704
Relevant durable memory: .memory/project/{pipeline_invariants.md,hand_metric_mechanisms.md,object_geometry_and_render_lessons.md,self_consistency_metrics.md}; demo final state in .memory/tasks/2026-07-04-demo-pack/EPISTEMIC.md.
Stakeholder feedback after demo: separate head/camera and hand error metrics (~5mm expected for both); visible hand vs annotation drift in tomato possibly intrinsics/reprojection/metric-space; fine-grained video captioning into semantic 2-3s clips; directly callable API with ~10000 video-hours/week throughput.

## Task specifications
Delivery track should exclude current HOI/factor graph optimization unless required for head/camera/hand/captioning. It needs minimal vision modules, API-ified batched services, correctness fixes, simplified agent harness for semantics + limited per-clip tuning, output schema, realistic self-consistency/benchmark metrics, and auto-research guides.
Research track should continue HOI physical extraction, v19 bottleneck analysis, HOI outputs/metrics, dataset/metric reuse, factor-graph/RL approximation reasoning, and end-to-end distillation reasoning.

## Constraints
Do not invent progress via reports alone; final output must be usable task packs and a decision memo.
Separate delivery claims from research claims.
Favor mechanisms and falsifiable metrics over broad roadmap wording.
Do not require user decisions unless choices materially affect scope/resource allocation.
