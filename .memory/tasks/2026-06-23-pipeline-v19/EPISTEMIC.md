# Pipeline V19 current epistemic state

## Current supported claim

Workbench item 3 remains unfinished. The renderer/projection reset is implemented, and P19 can render filled rigid bodies from explicit state, but the accepted V19 milestone still requires a runtime-generated full-duration visual artifact whose physical keyboard body, MANO hand state, and world relationship pass subjective inspection.

The current supported mechanism claim is: the failed runtime keyboard artifact is upstream of P19. `supportrepair_v1` proved that P09 can propagate object-owned masks after hand-owned bbox subtraction, but it also showed that formal mask-path ownership is not enough. The v1 anchor frame 60 retained an implausible metric visible surface extent `[0.389, 1.134, 0.746]` m. Visual review of `/tmp/v19_clip001851_p09_supportrepair_review.jpg` revised the earlier explanation: the repaired 2D masks are mostly keyboard-shaped after hand subtraction; the dominant remaining failure is evidence-frame/metric-anchor selection and metric lifting scale, not simply raw SAM2 table/hand leakage.

## Current causal model

Solved renderer mechanisms: the legacy point renderer and fragile K scaling have been replaced by P19a/P19b state rendering, mesh-face rasterization, and explicit source-size-to-render-size projection scaling.

Remaining physical failure chain:

1. P06/P07 plus P09 hand-owned subtraction can produce object-owned masks that visually isolate the keyboard key grid/body in many frames.
2. The old P09 anchor selection used a poor frame (frame 60) whose metric visible surfels are much too large. P13 consumes the P09 anchor visible mesh, so a bad anchor poisons TRELLIS alignment/completion even if the P11 crop mask is object-owned.
3. Candidate review found later non-border frames, especially frame 140/143, with visually object-dominant keyboard masks and much smaller extents (~0.39–0.44 m max axis). This supports a general evidence-frame selection repair: choose a visually object-dominant, non-border, compact metric support frame rather than a maximum-mask/maximum-vertices frame.

Predictions now live for `supportrepair_v2`: P09 rerun with anchor frame 140 should produce object-owned masks and a smaller anchor visible mesh. If v2 passes physical anchor validation, P11 frame-140 crop should condition TRELLIS on keyboard body/key grid rather than frame-60 distorted support. If v2 still produces an oversized anchor or visually bad crop, the next mechanism is metric depth/calibration or P07 mask support, and P12/P19 must not run.

## Rejected mechanisms and claims

- Rejected: “P19 output existence means V19 artifact sanity.” The rendered body was visually wrong.
- Rejected: “rerun only P19 to fix the failure.” P19 exposes a bad upstream state.
- Rejected: “P09 mask-path contract alone fixes object support.” v1 wrote object-owned masks, but the physical anchor was still invalid.
- Rejected: “raw SAM2 provenance in `source_mask_path` is itself a P11 failure.” It is provenance; the operative P11 crop mask/selected mask path must be object-owned.
- Rejected: “frame 60 is a valid keyboard anchor because it has many visible pixels.” Its metric extent is implausibly large.
- Rejected: metrics/autoresearch before subjective runtime artifact sanity.

## Live uncertainties

1. Whether active `supportrepair_v2` completes P09 with frame-140 anchor under the runtime Pi without provider/tool failure.
2. Whether frame 140 remains physically better after full P09 rerun and runtime validation, not only in parent inspection of v1 annotations.
3. Whether the frame-140 P11 crop is sufficient for TRELLIS completion and downstream pose fitting.
4. Whether final P19 after v2 shows a keyboard body in the correct 2D/world relationship, or whether metric depth/calibration still scales the body incorrectly.
5. Whether MANO interval correction remains visually coherent once the object body is physically repaired.

## Next action

Let the active tmux Pi continuation run `supportrepair_v2` from P09 with anchor frame 140. If P09/P11 pass physical validation, continue through P19b and then inspect overlay/world/side-by-side as physical annotation. Do not run P20/P21, metrics, or autoresearch before parent visual inspection of the repaired P19 artifact. If v2 fails, repair the next exposed physical mechanism rather than adding validators.
