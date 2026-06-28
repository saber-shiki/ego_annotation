# Pipeline V19 current epistemic state

## Current supported claim

The prior claim that Workbench item 6 was complete remains retracted. Earlier HOT3D metrics/renders are diagnostic evidence only because the final object layer did not show a correct rigid keyboard body.

The current supported claim is narrower and updated: the known final-renderer mechanism has been repaired in code/spec, not yet accepted in a runtime deliverable. V19 now has a render-state builder (`scripts/build_v19_rigid_render_state.py`) and a V19 rigid-body renderer (`scripts/render_v19_rigid_state_artifact.py`) that consume explicit state, rasterize mesh faces as a rigid object body, and scale source-coordinate intrinsics to the decoded render frame size. A local smoke on the old `clip001850` run showed nonzero filled keyboard-body pixels and recorded 1408x1408 -> 960x960 K scaling. This proves the renderer no longer hides object geometry as sampled vertices, but it does not prove the physical keyboard pose/mesh is sane.

## Current causal model

The original active failure had two coupled mechanisms:

1. P19 routed to `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py`, which loaded only mesh vertices and drew sampled `cv2.circle` points. It did not load mesh faces or rasterize a rigid body, and it consumed measurement reports directly rather than a render-consumed state boundary.
2. Projection semantics were implicit and wrong-prone. HOT3D frames decoded/rendered at 960x960 carried source-size 1408x1408 intrinsics; the old renderer scaled by `width/(2*cx)` rather than by an explicit source-size/render-size contract.

The repair changes the mechanism: P19a now writes `state/render_state/{OBJECT_ID}_rigid_render_state.json` containing completed mesh path, accepted full-timeline pose rows, constraint rows, optional temporal MANO payload, and the projection contract. P19b renders from that state, rasterizes mesh faces, and records scaled intrinsics examples. The runtime spec and English orchestration now forbid the old V18-named point renderer as a final P19 renderer.

The local diagnostic smoke is also negative evidence about the existing old prediction state: with all faces on frame 75, the keyboard appears as a large green rigid body but spills over hand/table regions. That means the new renderer can reveal object-state/pose/mesh problems that the point renderer obscured. It does not by itself fix mesh completion or pose fitting.

## Rejected mechanisms and claims

- Rejected: “Workbench item 6 complete” for the fixed HOT3D slice. The final object artifact failed strict rendered-annotation consumption.
- Rejected: “Workbench item 7 active.” Autoresearch must wait until renderer wiring, projection, full runtime rerun, subjective sanity, render quality, and quantitative comparison are done in order.
- Rejected: treating full-duration videos, render manifests, residual reports, or green object points as evidence of a rigid keyboard body.
- Rejected: treating SAM2 keyboard segmentation correctness as sufficient for object-pose/render correctness. The failure occurs downstream of segmentation.
- Rejected: using `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py` as final V19 P19 output. It is historical/diagnostic only because it renders sampled vertices, not a body.

## Live uncertainties

1. Whether a full runtime Pi rerun follows the corrected P19a/P19b state-render contract end-to-end without parent assembly.
2. Whether the corrected runtime artifact passes subjective physical sanity in 2D overlay and 3D/world view when the filled rigid body exposes the actual mesh/pose state.
3. Whether the existing keyboard mesh completion/pose branch is physically too broad or misregistered even after correct rendering and K scaling.
4. Whether the MANO/contact/occlusion state remains coherent when viewed against a real filled keyboard body rather than point samples.
5. How to preserve real subjective judgment rather than checklist compliance: every subjective review must surface at least one artifact-specific observation/anomaly/risk hypothesis not already named by the protocol, and that observation must affect the next repair decision or explicitly rule out an obvious unlisted fault.

## Next action

Commit the scoped renderer/spec/task-memory repair, sync the corrected runtime code/spec into the runtime workspace, rerun the full pipeline through the runtime Pi agent, then inspect the final overlay/world/side-by-side videos as a user would. Do not resume HOT3D quantitative comparison or autoresearch until the runtime-rendered keyboard example is free of obvious first-glance faults or its failure mechanism is preserved.
