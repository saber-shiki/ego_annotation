# Pipeline V19 current epistemic state

## Current supported claim

The HOT3D `clip-001850` pinhole V19 v5 mesh-frame-repaired freeze is rejected as a correctness artifact. The latest canonical contact sheet and independent diagnostics showed the rendered green keyboard/object geometry was visibly stretched and displaced because the upstream object mask/visible-surface measurement was contaminated by hands/table/arm. Both v5 freeze hashes are evidence only, not deliverables.

Runtime-owned OWLv2 P06/P07 now supplies a cleaner keyboard mask stream, and the downstream rerun has progressed through P17 from `sam2_owlv2_box_points`: P08 wrote base MANO/object state, P09 lifted visible geometry using the corrected mask root and canonical focal contract, P13 produced a completed mesh from the new anchor support, P14 fit 98 visible-frame keyboard poses at about 5 mm median observed-to-mesh residual, and P15 emitted full-timeline rigid poses with 98 direct corrected poses plus 52 uncertain temporal completions. This supports the claim that the previous mask-support mechanism has been materially repaired, but it still does not prove final render registration or MANO/contact correctness; P18 is actively running and P19/P20/P21 visual consumption remains decisive.

Rejected boundaries:
- Wrong raw-mesh freeze: `daa3978f641691ef148720a75a9ec3bea35a22e55b6c01eb6a0eea0ed85f2a3c`.
- Mesh-frame-repaired but still visually wrong freeze: `2d999fbc4f51e375b4a9eb7277943bba74ce542cf0c613ece3479c888a267080`.

## What remains supported

The P12 raw TRELLIS-vs-P13 completed-mesh mismatch was real, and commit `1b0aa0f Require completed mesh provenance for V19 P18/P19` remains a valid guard. It is insufficient because the P13 completed mesh and P14/P15 poses were built from contaminated P07/P09 measurements.

## Current root mechanism

The rejected v5 freeze failed because P07/P09 accepted wrong object support. The earlier SAM2 masks for representative frames included non-object regions: keyboard plus hands/table/arm. P09 lifted those mask pixels with depth into visible surfels; P13 completed geometry and P14/P15 pose fitting then used that contaminated surface. Renderer-equivalent projection of P13/P15 reproduced the bad canonical overlay, so P19 publication/projection was not the primary root.

Positive-click-derived and agent-authored box mechanisms were rejected because they still selected tabletop/hand support. The current replacement mechanism is OWLv2 text-grounded detection (`keyboard.` / `computer keyboard.`) producing object boxes that seed SAM2. Runtime-owned OWLv2 P06/P07 masks are visually localized to the keyboard footprint on inspected frames, with only small disconnected mask noise rather than broad table/hand/arm support. P09-P15 from that branch now produce many direct visible pose fits instead of sparse/contaminated pose support, so the live causal question has moved downstream: whether cleaner visible support yields a final canonical render registered to the physical keyboard, or whether P09 surfel lifting, P13 completion/adaptation, P14/P15 pose frame, camera conventions, or P19 rendering still introduce a systematic geometric error.

Evidence:
- `/tmp/v19_wrong_registration_diagnostics/renderer_equiv_projection_sheet.jpg`: red SAM2 masks include non-object support; blue P09 surfels follow that contaminated support; green P13/P15 mesh follows the same contaminated support.
- `/tmp/v19_mask_refine_probe/hand_subtract_probe_sheet.jpg`: subtracting hand boxes alone leaves large tabletop components.
- `/tmp/v19_mask_refine_probe/prompt_refine_probe_sheet.jpg`: prompt-envelope refinement reduces some contamination but still leaves table support; the existing P06/P07 prompt/mask mechanism is not sufficient as-is.

## Strict blocker

The final canonical overlay/world/side-by-side must render object geometry that coincides with the physical keyboard/key field. A mask, mesh, or pose that includes broad tabletop/hand/sleeve support is a hard failure, not an uncertainty label. Small disconnected mask noise may be carried or filtered as measurement uncertainty only if it does not determine the rigid geometry/pose.

## Live repair mechanisms

1. If the final render is still wrong while P07 masks are clean, localize the first broken transition by projecting: P07 mask contours -> P09 lifted surfels -> P13 completed mesh under P15 pose -> P19 rendered frames. This distinguishes depth/intrinsics/backprojection, completion/pose-frame, and renderer/projection mechanisms without blind detector retries.
2. If P13/P14 overfit remaining small mask noise, add a generic P09/P14 visible-surface refinement/rejection mechanism using prompt support, component filtering, hand occluder exclusion, depth/extent consistency, and visual QC; contaminated frames must become missing/uncertain observations rather than geometry-completion anchors.
3. If the object registers but MANO/contact remains incoherent, repair the interval hand state/contact mechanism separately; do not use contact uncertainty labels to excuse object-registration failure.

## Next action

Do not run HOT3D scoring and do not claim any freeze accepted. Let the runtime finish P18-P21 from the accepted OWLv2/SAM2 mask branch, then consume the canonical overlay/world/side-by-side as physical annotations. The decisive falsifiable claim is: if broad mask-support contamination was the dominant mechanism, the new render should move the keyboard geometry onto the physical key field; if it remains stretched/displaced, the remaining systematic error lies downstream in P09 surfel lifting, P13 completion/adaptation, P14/P15 pose fitting, camera conventions, or P19 rendering.
