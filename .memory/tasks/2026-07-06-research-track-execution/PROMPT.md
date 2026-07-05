## Objective
Build the research-track HOI/physical-state extraction system: full-duration renderable annotations for object geometry, object pose, contact, occlusion ownership, nonpenetration, articulation/deformation, and hand-object interaction events. Progress means resolving mechanisms: connect artifact defects to physical variables, test competing mechanisms with discriminating measurements, implement the selected intervention, and change the rendered/numeric HOI state. Done when R0-R11 in `TASK_PACK.md` have produced a live `ego.hoi` sidecar, graph instrumentation, correspondence/rigidity, geometry epochs, pose optimizer, HOI hand corrections, contact/occlusion channels, live factor graph, evaluator, RL approximation study, distillation study, and delivery integration route.

## Workbench
1. Start R0/R1: build `ego.hoi` sidecar skeleton and graph instrumentation plan for support/liveness/gauge/freshness/input-output deltas.
2. Choose first concrete research slice from current evidence: object pose/geometry/contact failure on representative v19/demo clips, with causal cards before runtime.
3. Delegate concrete workstreams to subagents: code inventory for existing v18/v19 graph/render/state paths, measurement-design critic, and implementation plan for graph-health instrumentation.
4. Implement the first artifact-changing mechanism selected by the discriminating analysis.
5. Render/inspect the resulting HOI state and update EPISTEMIC with what the measurement changed.

## Context
Repo root: `/home/yiwen/ego_annotation`.
Canonical research task pack: `.memory/tasks/2026-07-06-research-track-execution/TASK_PACK.md`.
Planning source: `.memory/tasks/2026-07-05-delivery-research-track-planning/RESEARCH_TRACK_TASK_PACK_DRAFT.md`.
Project invariants: `.memory/project/pipeline_invariants.md`, `.memory/project/object_geometry_and_render_lessons.md`, `.memory/project/hand_metric_mechanisms.md`.
Relevant prior systems: v18/v19 scripts under `scripts/`, runtime specs under `runtime/`, v19 artifacts under `/data2/ego_annotation_outputs/v19_runs/`, v18 artifacts under `/data2/ego_annotation_outputs/v18_*`.
Output namespace: `org.ego.hoi`; schema family `ego.hoi` 0.1.0.
Research branch for actual work: `yiwen_research`.

## Task specifications
Every experiment must start with a causal card: artifact defect, physical variable, live mechanisms, discriminating measurement, predictions, intervention per outcome, and expected table/render state change.
Start with R0/R1 because the current graph can be inert while appearing implemented; graph support, liveness, stale dependencies, gauge declarations, and render-state lineage must be measurable before adding more factors.
Use full-duration renderable HOI artifacts as the research output: tables/assets/provenance plus overlay/world/side-by-side videos driven by state rows.
Do not replace analysis with activity: a build task is valid only when tied to a mechanism and discriminating measurement.
Do not replace analysis with caution: uncertainty is a state variable and routing signal, not a reason to omit the mechanism.
Initial mechanism families are object-pose null spaces, geometry epoch contamination/prior completion, contact evidence under occlusion, smooth HOI hand drift, graph inertness/stale joins, cross-solver geometry-source decoupling, and runtime-heavy teacher/student separation.
The first slice is HOT3D clip001850 keyboard, right hand, frames 28-48 in run `/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`. KT-1/2/3/5 plus the repair/counterfactual branches show the interval mesh-penetration channel is not admissible keyboard contact: f32 routes to `geometry_epoch_contaminated`, f36 routes to `full_frame_depth_leak`, and f45 demotes to `unresolved_incoherent_evidence` because it is one depth-outlier thumb vertex without temporal support. R0/R1 must expose the interval channel, contact/NP completed-mesh query, keyboard-masked depth, canonical contact state, and render state as one reconciled graph/contact state without wiring contaminated penetration into contact.
Graph-health rows must include provenance-hash-derived solver/contact/render geometry epoch ids, geometry source families, signed query candidate count, watertight flag, face provenance summary, non-max penetration statistics, published contact gap, incomplete-provenance routing, and a `cross_solver_geometry_decoupled` decision route when the measured provenance differs.
RL work begins only after live graph variables and evaluator exist; distillation begins only from validated teacher outputs.
All heavyweight inference/runtime work runs on server/A800 or approved non-local compute.

## Constraints
Do not write boundary/overclaim/permission framing into task artifacts.
Do not call schema rows progress unless they drive rendered HOI annotations or expose a mechanism that selects the next intervention.
Do not add factors before measuring support/liveness and the variable they control.
Do not use prior-completed or stale geometry as contact/nonpenetration evidence without provenance routing.
Do not run heavy local GPU/model inference on the workstation.
Do not stage unrelated repository changes.
