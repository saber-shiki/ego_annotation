# Pipeline V19 current epistemic state

## Current supported claim

Workbench item 3 is still active. Renderer/projection repair and supportrepair_v2 upstream object-support repair have advanced the runtime to a stronger physical state, but no V19 milestone is accepted until a fresh supportrepair_v2 P19b overlay/world/side-by-side artifact is visually consumed as an annotation.

The rejected counterfactual is the old runtime P19 artifact: its green keyboard body rendered as a large vertical sheet/curtain over the hand/table, not as a keyboard-scale body. That failure was not caused primarily by P19 projection; it came from upstream object-support/anchor contamination exposed by the repaired face renderer.

## Current causal model

1. P19 renderer/projection mechanics are capable of rendering mesh faces from explicit render state with scaled source intrinsics. Therefore P19 output is now diagnostic of the render-consumed physical state rather than just a point-cloud/projection bug.
2. supportrepair_v1 proved object-owned mask paths alone are insufficient: the frame-60 anchor still had implausible metric extent `[0.389, 1.134, 0.746]` m.
3. supportrepair_v2 changed the causal evidence source by anchoring P09/P11 at frame 140. The repaired anchor extent is compact for the keyboard case (`~[0.124, 0.388, 0.208]` m), P11 conditions TRELLIS on an object-owned keyboard crop showing key-grid/body, P12/P13 are fresh from that crop, P14 has 150 pose rows with 146 fitted frames, and P15 has full 150-frame rigid pose completion. This supports the mechanism that evidence-frame selection, not just mask-path provenance, was the dominant object-body failure.
4. P16/P17 reintroduced hand/object/contact evidence on top of the repaired object. The P17 judgments are intentionally weak possible-contact priors with hand-projected pixels quarantined from object support, so P18 should not be interpreted as a force that must snap MANO hands onto the keyboard. It should preserve uncertainty unless visible-surface support exists.
5. P18 is currently the live transforming mechanism: it is solving the full 0-149 MANO interval state from repaired object pose/mesh plus the weak contact/visibility factor. Existing P18/P19 files are stale pre-v2 artifacts until replaced by a fresh supportrepair_v2 event.

## Rejected mechanisms and claims

- Rejected: “P19 output existence means V19 artifact sanity.” The old artifact was visibly wrong.
- Rejected: “rerun only P19 fixes the failure.” P19 exposed bad upstream object state.
- Rejected: “object-owned mask path contract alone fixes object support.” v1 wrote object-owned masks but kept an invalid frame-60 anchor.
- Rejected: “raw SAM2 provenance in `source_mask_path` is itself a failure.” Operative P11/P09 mask paths matter; provenance can preserve raw evidence.
- Rejected: “old P17/P18/P19 events without `marker=supportrepair_v2` are current completion evidence.” They belong to the rejected pre-repair branch.
- Rejected: metrics/autoresearch before subjective runtime artifact sanity.

## Live uncertainties

1. Whether P18 supportrepair_v2 completes without failure and writes a fresh interval MANO state.
2. Whether P19a/P19b after P18 consume the fresh supportrepair_v2 state rather than stale pre-repair state.
3. Whether fresh P19b shows a keyboard-scale rigid body aligned with the visible keyboard in overlay, world, and side-by-side views.
4. Whether MANO state remains visually coherent relative to the repaired keyboard, or remains separated/penetrating/floating with honest uncertainty.
5. What open-ended anomaly appears in visual review beyond the predeclared dimensions.

## Next action

Let the single runtime tmux continue P18. When fresh supportrepair_v2 P18/P19 events appear, copy the P19b videos locally, build the prepared visual review sheets, and inspect overlay/world/side-by-side frames as physical annotation. The primary falsifier is the old rejection cause: if any of the three fresh views still shows the keyboard as a large green sheet/curtain over the hand/table, Workbench item 3 remains failed regardless of solver status, row counts, or metrics. If the repaired artifact passes subjective physical sanity, proceed to Workbench item 4 render-quality polish. If it fails, localize the mechanism from visual evidence before any metrics or publication.
