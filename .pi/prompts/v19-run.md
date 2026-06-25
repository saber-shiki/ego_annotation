---
description: Run V19 Pi-harness physical annotation on a video
argument-hint: "<input-video> <run-root> [case-id]"
---
Run V19 prediction on this input using Pi itself as the annotation harness.

Input video: `$1`
Run root: `$2`
Case id or label: `${@:3}`

## Runtime input contract

The runtime inputs are the input video, the fresh run root, this case id, the single runtime spec in this workspace, and prediction-side sensor metadata needed by the pipeline.

Pi is the harness. Do not create or call an outer script that controls Pi. Python scripts may be called only as measurement, optimization, rendering, or export tools. Launch preflight is complete before start; execute prediction phases only.

## Start actions

1. Read `runtime/v19_runtime_spec.md` and no other instruction document. Do not inspect or mention any file that is not named by that spec.
2. Verify the input video exists and identify frame count, FPS, resolution, and duration without changing the video.
3. Confirm the run root does not overwrite an existing completed V19 run.
4. Create initial `input/`, `logs/`, and unresolved `state/` records before launching measurement tools.
5. Probe the declared server route before heavy work and record the selected compute target.
6. Execute the phase graph in `runtime/v19_runtime_spec.md` in order. Do not discover alternate scripts. If a required component is missing, write the concrete missing implementation and blocked physical variable under the run root rather than fabricating outputs.

## Output contract

Produce a V19 prediction run root with render-consumed `state/`, prediction-side `measurements/`, durable `logs/`, and full-duration `renders/v19_overlay.mp4`, `renders/v19_world.mp4`, and `renders/v19_side_by_side.mp4` when the spec reaches render publication.

Progress requires a changed physical state, a rendered physical artifact, a physically discriminating measurement, or a concrete mechanism failure that determines the next intervention. Created directories, JSON validity, row counts, validators, or launch commands are not progress by themselves.

## Physical target

The target is interval-level metric MANO correction over the full input timeline with explicit camera/world semantics, object geometry/pose state, contact/occlusion/nonpenetration uncertainty, and clear overlay/world/side-by-side visualization.

After an object is classified rigid, visible surfaces are evidence only. The required branch is completion/adaptation -> visible-frame pose -> factor-graph correction -> corrected mesh-pose render.
