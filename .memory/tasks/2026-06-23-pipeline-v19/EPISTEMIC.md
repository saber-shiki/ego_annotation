# Pipeline V19 current epistemic state

## Current supported claim

Workbench item 3 is **not accepted** for clip001851. `anchorreview_v2` completed P13–P19b and produced full-duration overlay/world/side-by-side videos, but rendered physical inspection rejected the artifact because MANO hands remain physically incoherent relative to the keyboard.

Supported improvements:

1. The object branch is materially better than the revoked `supportrepair_v2` curtain/sheet artifact. Runtime anchor review selected frame 106, and accepted-body semantics exclude `unsupported_uncertain` Poisson fill from `outputs.completed_mesh_labeled`. The v2 completed mesh contains accepted `{observed_depth_surface, trellis_inferred_hidden_surface}` faces and zero accepted `unsupported_uncertain` faces. See OPS entries around anchorreview_v2 P13.
2. Fresh P19b v2 is mechanically complete and fresh: 150 overlay/world/side-by-side frames, 5.0 s videos, render state references the v2 mesh and v2 MANO interval state.
3. Visual review of frames 95/106/120/140/149 shows the green object is roughly keyboard-scale/key-grid registered, but the MANO hands remain detached or wrongly ordered in world/side-by-side views.
4. The earlier conclusion that visible hand-mask support is absent on the rejected frames is **retracted**. It was produced by an implementation bug in `build_v19_mano_mask_depth_refit_inputs.py`: the adapter hardcoded a 960x540 target and 0.5 projection scale, while HOT3D clip001851 masks are 960x960 from 1408x1408 source. After repair, frames 95–149 recover 55/55 visible filtered masks for both hands. See OPS 2026-06-30 mask-shape repair entries.

Therefore the remaining blocker is not P19 production, unsupported Poisson promotion, or absent raw hand masks. The blocker is the coupled physical MANO/object relation: available hand support can improve image/depth alignment, but current mechanisms do not produce a coherent hand-on-keyboard state.

## Current causal model

### Object-registration mechanism

The rejected supportrepair artifact failed partly because uncertainty labels did not control downstream render semantics and because the old anchor/frame choice promoted contaminated support into the accepted body. `anchorreview_v2` repaired those mechanisms by selecting frame 106 through candidate visual review and excluding `unsupported_uncertain` observed Poisson fill from accepted/rendered object body.

Rendered evidence supports only a scoped object-side claim: the branch no longer renders the broad curtain/table-sized body, and the object body is generally keyboard-scale. The filled mesh remains broad/solid over the key area and can overpaint beyond visible keys, so object refinement may be needed later. But relative to the current hand failure, object pose/completion is not the primary root blocker.

### Hand support and depth mechanism

The mask/depth support pipeline had a real coordinate bug. Repairing the adapter changed support counts from prior projection-only `left_visible_filtered_masks=0`, `right_visible_filtered_masks=5` to 55 visible masks for each hand over frames 95–149. This means direct hand support exists and can drive a refit; the previous zero-support conclusion is invalid.

A fixed-scale MANO mask/depth refit using repaired support partially works as a measurement repair:

- left hand selected 36 rows, preserving scale near 1, but failed frames 95–113 due a zero-state MANO/bridge mismatch around 3 cm;
- right hand selected all 55 rows, preserving scale near 1 but needing larger translation changes (~3.3 cm median) and pose changes (p95 around 1.1 rad);
- promoted candidate state had 91 promoted hand-frame rows and 19 skipped rows.

However visual render of `maskdepth_shape_v1` rejects this as a physical correction. The overlay alignment improves in places, but local-world hands remain detached/misordered around the keyboard. Direct nearest-object distance is mixed: some later rows improve, but several right-hand inspected frames move farther from the keyboard surface.

### Contact-coupling mechanism

A contact/object-coupled correction is still missing. Running contact-similarity on the promoted state exposed two more mechanisms:

- sparse 64-vertex annotation samples are inadequate for contact target selection; using them produced only six contact rows and left ~11 cm median contact residual;
- using full MANO vertices from the promoted bridge improves target selection and reduces median contact residual to ~4.9 cm on ten rows, but observations are still sparse, scale pins at the ±2% bounds, and rendered yellow hypotheses appear only on a few frames and remain visually incoherent.

Thus the current contact-similarity mechanism is not a valid interval correction. It can reduce a local residual where image-adjacent vertices exist, but it cannot reason through frames where the projected hand/object do not overlap, occlusion hides the true contact, or object/hand depth-order is ambiguous.

## Rejected mechanisms and claims

- Rejected: “P19b/P19c supportrepair artifacts are accepted physical annotation.” They failed final visual physical registration.
- Rejected: “anchorreview_v2 closes Workbench item 3.” It completes videos but fails MANO/object physical sanity.
- Rejected: “unsupported_uncertain Poisson fill is acceptable object body.” It is diagnostic uncertainty and is now excluded from accepted mesh semantics.
- Rejected: “translation-only contact/depth correction can repair the v2 hand state.” The focused subset candidate preserved scale/projection but left centimeter-scale separation.
- Rejected: “visible hand-mask/depth refit is unavailable on the rejected frames.” That claim was an artifact of the hardcoded 960x540/0.5 adapter bug.
- Rejected: “fixed-scale mask/depth MANO refit alone solves the hand-object relation.” It improves support/image/depth evidence but remains physically incoherent in world render.
- Rejected: “current contact-similarity Sim(3) refit solves the interval.” Full-vertex contact improves residuals on sparse rows but does not produce coherent rendered hand-keyboard geometry.
- Rejected: metrics/P20/P21/autoresearch/item 4. They remain blocked until Workbench item 3 visual physical sanity passes.

## Live uncertainties

1. Whether the broad/solid accepted TRELLIS keyboard body needs further object-side refinement after the hand blocker is repaired.
2. Why left early frames 95–113 have a MANO zero-state/bridge mismatch that prevents the articulated mask-depth refitter from fitting them.
3. How to build a coupled hand/object interval mechanism that uses recovered hand masks/depth, full MANO surfaces, keyboard surface geometry, occlusion state, and temporal continuity without silently snapping MANO to the object.
4. Whether the object-mask adjacency assumption is too strict for keyboard manipulation under occlusion, requiring contact candidate generation from depth-order/nearest-surface hypotheses rather than projected mask overlap alone.

## Next action

Do not run metrics/P20/P21/autoresearch. Workbench item 3 remains active. The next causal intervention must be a coupled MANO/object interval repair: use the recovered hand masks/depth and full MANO surface, but replace sparse object-mask-overlap contact selection with an occlusion-aware nearest-surface/contact candidate mechanism that can operate when projected overlap is absent and can be rejected visually if it creates false contact.
