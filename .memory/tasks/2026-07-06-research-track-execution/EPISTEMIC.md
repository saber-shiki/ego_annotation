# Current epistemic model

The research task is a mechanism-resolution problem. The required artifact is full-duration renderable HOI annotation driven by measured geometry, pose, contact, occlusion, and hand-correction state.

The current deepest blocker is graph observability/liveness, not adding another factor. Prior v19 failures can be explained by object pose null spaces, geometry epochs that mix observed and prior-completed surfaces, contact evidence disappearing under occlusion, smooth hand metric drift, stale joins, and solver outputs that can equal inputs while appearing implemented.

The first research intervention should make these mechanisms measurable. R0/R1 must create the `ego.hoi` sidecar skeleton and graph-health instrumentation so each solve exposes support fraction, active residuals, input-output state deltas, stale dependency counts, gauge declarations, geometry epoch lineage, and final render-state hashes.

The first concrete slice is HOT3D clip001850 keyboard, right hand, frames ~26-46 in run `/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`. The artifact defect is triple-valued contact distance for the same hand/object relation: interval MANO reports ~10 cm penetration into observed keyboard surface, contact/nonpenetration reports zero penetrating vertices against a 95.4%-TRELLIS non-watertight completed mesh with zero signed-query candidates, and the published render reports a +39 mm gap. The rigid pose graph is inert (`nfev=1`, cost 0, deltas 0, nonpenetration targets 0). The mechanism is cross-solver geometry-source decoupling plus missing factor coupling: the measurement exists in one solver, is absent in another because it queries different geometry, and is not reconciled by the graph or renderer.

R0/R1 now needs a fifth graph-health route beyond support/inert/stale-join/stale-render: `cross_solver_geometry_decoupled`. Required fields are solver/contact/render geometry epoch ids, geometry source families, signed query candidate count, watertight flag, face provenance summary, observed-surface penetration, and published contact gap. This route must fire before generic `measurement_support_absent` because the absence is solver-local, not measurement-global.

The next discriminating experiment is CPU-only D-a/D-b/D-d on frames 26-46: compare hand-to-surface distances against observed surface and completed mesh under common pose, test temporal coherence of penetrating vertices, and evaluate free-space/extent plausibility of the completed keyboard mesh. The result selects either wiring observed-surface evidence into contact/NP or repairing geometry epoch contamination first.

Subagents should handle concrete inventory/review tasks while the main agent owns the causal roadmap and final mechanism selection.
