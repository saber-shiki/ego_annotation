---
description: Run V19 Pi-harness physical annotation on a video
argument-hint: "<input-video> <run-root> [case-id]"
---
Run V19 on this input using Pi itself as the annotation harness.

Input video: `$1`
Run root: `$2`
Case id or label: `${@:3}`

## Non-negotiable architecture

Pi is the harness. Do not create or call an outer script that controls Pi. Python scripts may be called only as measurement, optimization, rendering, export, or evaluation tools.

Use the active V19 system prompt from `configs/v19_agent_system_prompt.md`. If this prompt was not loaded through `--system-prompt`, stop before annotation work and report the launch command that should be used.

## First actions

1. Read `.memory/tasks/2026-06-23-pipeline-v19/PROMPT.md`, `.memory/tasks/2026-06-23-pipeline-v19/EPISTEMIC.md`, `.memory/tasks/2026-06-23-pipeline-v19/OPS.md`, `docs/v19_component_extraction.md`, `docs/v19_english_orchestration.md`, and `docs/v19_run_contract.md`.
2. Inspect `git status --short`; identify unrelated dirty files and do not stage or modify them.
3. Verify the input video exists and identify frame count, FPS, resolution, and duration without changing the video.
4. Verify local/Pi route and current A800/server target state before any heavy tool call.
5. Create the run root only after confirming it does not overwrite an existing V19 run.
6. Write `input/input_manifest.json`, `logs/harness_events.jsonl`, and unresolved initial `state/` files before launching measurement tools.
7. Continue through the Workbench in `.memory/tasks/2026-06-23-pipeline-v19/PROMPT.md`; for the current orchestration step, follow `docs/v19_english_orchestration.md` and do not invent missing scripts.

## Physical goal

The first project-video goal is interval-level metric MANO correction over the full input timeline with explicit camera/world semantics, object geometry/pose state, contact/occlusion/nonpenetration uncertainty, clear overlay/world/side-by-side visualization, and later bounded quantitative evaluation.

Do not claim progress from created directories, JSON validity, row counts, validators, or launch commands. Progress requires a changed physical state, a rendered physical artifact, a physically discriminating measurement, or a concrete mechanism failure that determines the next intervention.

## Rigid-object reminder

After an object is classified rigid, visible surfaces are evidence only. The required branch is TRELLIS completion -> visible-frame pose -> factor-graph correction -> corrected mesh-pose render.
