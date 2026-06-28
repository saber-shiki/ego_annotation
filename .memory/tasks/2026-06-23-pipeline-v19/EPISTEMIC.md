# Pipeline V19 current epistemic state

## Current supported claim

Workbench item 3 is accepted for the fresh Pi-runtime `supportrepair_v2` P19b artifact on `hot3d_clip001851_pinhole_a800_native_v2_renderstate_rerun`. The claim is scoped to subjective physical sanity of the runtime-rendered keyboard/MANO artifact, not to render polish, quantitative accuracy, or general benchmark performance.

The evidence is visual and causal, not container-level: the old rejected P19 rendered the keyboard as a large green sheet/curtain over the hand/table; the fresh P19b overlay/world/side-by-side videos keep the object as a compact keyboard-scale rigid body across early, middle, late, anchor, and terminal frames. The guarded fetch also proved the inspected videos/manifest were fresh post-P18/P19b `supportrepair_v2` runtime outputs, but freshness is supporting evidence only.

The next unfinished Workbench item is item 4: improve render quality according to existing instructions. P20/P21/HOT3D metrics and autoresearch remain blocked until item 4 has produced an audience-readable artifact or a deliberate Workbench decision advances to item 5.

## Current causal model

1. P19 renderer/projection mechanics render mesh faces from explicit render state with scaled source intrinsics. Therefore P19 output reveals the render-consumed physical state rather than a point-cloud/projection proxy.
2. The old curtain failure was caused primarily by upstream object-support/anchor contamination. Evidence: the repaired renderer exposed broad sheet geometry in the stale branch; `supportrepair_v1` object-owned mask paths alone still chose frame 60 with implausible extent; `supportrepair_v2` changed the evidence anchor to frame 140, where the object-owned crop shows a keyboard/key-grid and compact metric extent.
3. `supportrepair_v2` propagated causally through P12/P13/P14/P15/P18/P19: fresh TRELLIS/crop provenance from frame 140, compact completed mesh, full 150-frame object pose completion, fresh interval MANO state, and fresh render-consumed state. The decisive observation is that P19b video content changed from curtain to compact keyboard body.
4. Contact/MANO remains weak and uncertain by design. P17 judgments used possible-contact priors with hand-projected ownership quarantine; P18 should not be expected to snap both hands onto the keyboard. The fresh renders show near/offset/intersecting hand skeletons in some world-view intervals but label the MANO state as uncertain rather than accepted contact.
5. The remaining visible gap is not item-3 object-support sanity. It is item-4 presentation/legibility and uncertainty rendering: the keyboard appears as a coarse filled green slab/mesh, text overlays clutter the video, world view does not yet communicate contact/occlusion/depth order cleanly, and audience-readable uncertainty semantics are absent.

## Rejected mechanisms and claims

- Rejected: “P19 output existence means V19 artifact sanity.” The old P19 existed and was visibly wrong.
- Rejected: “rerun only P19 fixes the failure.” P19 exposed bad upstream object state.
- Rejected: “object-owned mask path contract alone fixes object support.” v1 wrote object-owned masks but kept an invalid frame-60 anchor.
- Rejected: “raw SAM2 provenance in `source_mask_path` is itself a failure.” Operative P11/P09 mask paths matter; provenance can preserve raw evidence.
- Rejected: “old P17/P18/P19 events without `marker=supportrepair_v2` are current completion evidence.” They belong to the rejected pre-repair branch.
- Rejected: “P18 runtime length indicates a left/right temporal coupling bug.” The solver calls `build_rows`/`optimize_rows` separately per side.
- Rejected: metrics/autoresearch before subjective runtime artifact sanity.
- Rejected: “the fresh P19b artifact is fully polished or quantitatively validated.” It only closes Workbench item 3’s physical sanity gate.

## Live uncertainties

1. How to make item-4 renders audience-readable while preserving honest uncertainty and not hiding the coarse/uncertain physics.
2. Whether render-quality changes can be made without weakening the now-correct object-support/body mechanism.
3. How the repaired prediction will score on fixed HOT3D/open-source metrics after a render-quality freeze.
4. Whether the MANO/contact uncertainty is normal measurement uncertainty for this clip or a systematic hand/world alignment error that must be repaired before item 5.

## Next action

Proceed to Workbench item 4 only. The first item-4 target should be render readability and honest uncertainty display for the already fresh P19b branch: reduce clutter/overpaint, make overlay/world/side-by-side interpretable to a stakeholder, preserve full-duration videos, preserve the compact keyboard body, and expose weak MANO/contact as uncertainty rather than pretending closure. Do not run P20/P21/metrics/autoresearch until item 4 has produced and visually passed an audience-readable artifact or the Workbench is explicitly advanced to item 5.
