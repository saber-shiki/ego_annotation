# Current epistemic model

The research task is a mechanism-resolution problem. The required artifact is full-duration renderable HOI annotation driven by measured geometry, pose, contact, occlusion, and hand-correction state.

The current deepest blocker is graph observability/liveness, not adding another factor. Prior v19 failures can be explained by object pose null spaces, geometry epochs that mix observed and prior-completed surfaces, contact evidence disappearing under occlusion, smooth hand metric drift, stale joins, and solver outputs that can equal inputs while appearing implemented.

The first research intervention should make these mechanisms measurable. R0/R1 must create the `ego.hoi` sidecar skeleton and graph-health instrumentation so each solve exposes support fraction, active residuals, input-output state deltas, stale dependency counts, gauge declarations, geometry epoch lineage, and final render-state hashes.

The first discriminating experiment should run on a narrow representative slice where existing v19 artifacts show object/contact/hand inconsistency. If nonzero residuals produce zero state delta, the variable wiring or optimizer plumbing is the root defect. If graph output changes but render state remains stale, the state-to-render dependency chain is the root defect. If support is near zero, measurement extraction is the root defect.

Subagents should handle concrete inventory/review tasks while the main agent owns the causal roadmap and final mechanism selection.
