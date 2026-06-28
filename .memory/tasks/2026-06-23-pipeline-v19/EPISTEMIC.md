# Pipeline V19 current epistemic state

## Current supported claim

Workbench item 3 remains unfinished. The renderer/projection reset has been repaired in code and runtime spec, but the accepted V19 milestone still requires a runtime-generated full-duration visual artifact whose physical keyboard body, MANO hand state, and world relationship pass subjective inspection.

The supported mechanism claim is now sharper: the failed `clip001851` P19 artifact is not primarily a P19 rasterization/projection failure. P19 rendered a filled rigid body from state, but the body was physically wrong because upstream object-support evidence admitted hand/table/occluder pixels as keyboard surface. The current repair changes the upstream evidence mechanism: P09 now writes object-owned masks after hand-owned support subtraction and propagates those masks as the object `mask_path`, so P11/P12 TRELLIS conditioning no longer consumes raw SAM2 hand/table pixels as object appearance. P13 now rejects TRELLIS hidden faces outside the evidence-frame object-owned silhouette and conditionally rejects hidden faces far off a planar observed support slab. Commit `78e610d Propagate object-owned support into V19 geometry` preserves this mechanism.

## Current causal model

The original renderer failure had two solved mechanisms: the legacy renderer drew sampled vertices instead of mesh faces, and HOT3D projection used fragile source/render-size semantics. P19a/P19b now use explicit render-consumed state, mesh-face rasterization, and source-size-to-render-size K scaling.

The remaining visible failure is upstream object-state contamination. Earlier diagnostic evidence showed the keyboard mask/visible surface can include broad non-keyboard regions; once P19 renders faces honestly, the contaminated state appears as a large sheet-like body. A diagnostic P13 run with silhouette filtering rejected many hidden faces but still left a ~0.9 m planar extent. A depth-layer probe showed depth trimming alone keeps the large footprint. Therefore the live mechanism is not mainly hidden thickness; it is object-owned support selection before visible-geometry lifting and TRELLIS conditioning.

The current intervention targets that mechanism directly. If the causal model is right, the repaired P09->P13 rerun should produce P11 crops whose alpha mask is the object-owned keyboard support rather than raw SAM2 support, a smaller/cleaner completed keyboard mesh, and a P19 overlay/world view where the green body aligns with the keyboard rather than spanning hands/table. If the repaired render still spills, then the next mechanism is likely open-vocabulary/SAM2 keyboard mask identity/support (P06/P07) or evidence-frame selection, not renderer/P19.

## Rejected mechanisms and claims

- Rejected: “final V19 keyboard render is acceptable because P19 outputs exist.” The artifact was visually wrong.
- Rejected: “rerun only P19 to fix the current failure.” P19 is exposing a bad state; the bad state is built before TRELLIS/pose fitting.
- Rejected: “silhouette hidden-face filtering alone fixes the keyboard.” The diagnostic completion still had ~0.9 m extent, so the observed support itself was too broad.
- Rejected: “depth-band trimming inside the raw mask fixes support.” Representative frames still retained the oversized footprint.
- Rejected: “quantitative HOT3D comparison/autoresearch can resume now.” Workbench item 3 subjective artifact sanity still blocks metrics.

## Live uncertainties

1. Whether the runtime Pi continuation can complete supportrepair_v1 P09->P19b under the curated runtime bundle without provider/tool failure.
2. Whether hand-owned bbox subtraction is strong enough for this clip, or whether P06/P07 SAM2 support remains too broad even after hand boxes are removed.
3. Whether the repaired P11/TRELLIS crop selects the physical keyboard key grid/body rather than table/hand support.
4. Whether the completed keyboard mesh and pose are physically coherent in world view after support repair.
5. Whether MANO interval correction remains visibly coherent once the object body is repaired.

## Next action

Let the active tmux Pi continuation run supportrepair_v1 from P09 through P19b on `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260628_hot3d_clip001851_pinhole_a800_native_v2_renderstate_rerun`. Do not run metrics or autoresearch. When P19b exists, inspect the rendered overlay/world/side-by-side as physical annotation, including an open-ended anomaly search. If the keyboard body remains wrong, repair the next exposed physical mechanism rather than adding validators or ledgers.
