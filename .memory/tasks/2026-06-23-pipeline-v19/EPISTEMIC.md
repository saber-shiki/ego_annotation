# Pipeline V19 current epistemic state

## Current supported claim

The HOT3D `clip-001850` pinhole V19 v5 mesh-frame-repaired freeze is rejected as a correctness artifact. The latest canonical contact sheet and independent diagnostics show the rendered green keyboard/object geometry is visibly stretched and displaced because the upstream object mask/visible-surface measurement is contaminated by hands/table/arm. Both v5 freeze hashes are evidence only, not deliverables.

Rejected boundaries:
- Wrong raw-mesh freeze: `daa3978f641691ef148720a75a9ec3bea35a22e55b6c01eb6a0eea0ed85f2a3c`.
- Mesh-frame-repaired but still visually wrong freeze: `2d999fbc4f51e375b4a9eb7277943bba74ce542cf0c613ece3479c888a267080`.

## What remains supported

The P12 raw TRELLIS-vs-P13 completed-mesh mismatch was real, and commit `1b0aa0f Require completed mesh provenance for V19 P18/P19` remains a valid guard. It is insufficient because the P13 completed mesh and P14/P15 poses were built from contaminated P07/P09 measurements.

## Current root mechanism

P07/P09 accepted wrong object support. The SAM2 masks for representative frames already include non-object regions: keyboard plus hands/table/arm. P09 lifted those mask pixels with depth into visible surfels; P13 completed geometry and P14/P15 pose fitting then used that contaminated surface. Renderer-equivalent projection of P13/P15 reproduces the bad canonical overlay, so P19 publication/projection is not the primary root.

Evidence:
- `/tmp/v19_wrong_registration_diagnostics/renderer_equiv_projection_sheet.jpg`: red SAM2 masks include non-object support; blue P09 surfels follow that contaminated support; green P13/P15 mesh follows the same contaminated support.
- `/tmp/v19_mask_refine_probe/hand_subtract_probe_sheet.jpg`: subtracting hand boxes alone leaves large tabletop components.
- `/tmp/v19_mask_refine_probe/prompt_refine_probe_sheet.jpg`: prompt-envelope refinement reduces some contamination but still leaves table support; the existing P06/P07 prompt/mask mechanism is not sufficient as-is.

## Strict blocker

The final canonical overlay/world/side-by-side must render object geometry that coincides with the physical keyboard/key field. A mask, mesh, or pose that includes tabletop/hand/sleeve support is a hard failure, not an uncertainty label.

## Live repair mechanisms

1. Rerun P07 with stricter object-only prompts / negative prompts so SAM2 separates the dark key grid and immediate keyboard body from hands/table.
2. Add a generic P09 visible-surface refinement/rejection mechanism using prompt support, hand occluder exclusion, negative prompt neighborhoods, depth/extent consistency, and visual QC; contaminated frames must become missing/uncertain observations rather than geometry-completion anchors.
3. If no mask-only repair is clean enough, use a model-produced segmentation branch or VLM-guided mask plan for the object core, then feed the same generic P09-P15-P19 path.

## Next action

Do not run HOT3D scoring and do not claim any freeze accepted. The next artifact-changing attempt must rerun/refine P07/P09 before P13/P14/P19. P18/P19 mesh-frame provenance is already repaired; further work there will not fix the visible contradiction.
