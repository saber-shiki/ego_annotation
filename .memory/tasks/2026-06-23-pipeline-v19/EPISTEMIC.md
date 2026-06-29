# Pipeline V19 current epistemic state

## Current supported claim

Workbench item 3 is **not accepted** for clip001851. The fresh Pi-runtime `anchorreview_v2` branch completed P13–P19b and produced full-duration overlay/world/side-by-side videos, but rendered physical inspection rejected the artifact.

Supported improvements:

1. The object branch is materially better than the revoked `supportrepair_v2` curtain/sheet artifact. Runtime anchor review selected frame 106, and P13 accepted-body semantics excluded `unsupported_uncertain` Poisson fill from `outputs.completed_mesh_labeled`. The fresh v2 completed mesh contains only `{observed_depth_surface: 5444, trellis_inferred_hidden_surface: 51915}` accepted faces and zero `unsupported_uncertain` accepted faces; 299 unsupported observed faces remain diagnostic only.
2. Fresh P19b v2 is mechanically complete and fresh: 150 overlay/world/side-by-side frames, 5.0 s videos, render state references the v2 mesh and v2 MANO interval state. See OPS 2026-06-30 P19b entries.
3. Visual review of frames 95/106/120/140/149 shows the green object is keyboard-scale and roughly key-grid registered, especially at anchor frame 106, but the MANO hands remain physically incoherent relative to the keyboard in world/side-by-side views.

Therefore the remaining blocker is not P19 video production or unsupported Poisson promotion. The blocker is the physical MANO/object relation: the artifact still shows centimeter-scale detached or misordered hands around a keyboard that the video implies they are manipulating.

## Current causal model

### Object-registration mechanism

The rejected supportrepair artifact failed partly because uncertainty labels did not control downstream render semantics and because the old anchor/frame choice promoted contaminated support into the accepted body. `anchorreview_v2` directly repaired those two mechanisms:

- frame 106 was chosen through candidate visual review rather than defaulting to frame 140;
- `unsupported_uncertain` observed Poisson fill is no longer accepted/rendered as object body.

Rendered evidence now supports a scoped object-side claim: the branch no longer renders the broad curtain/table-sized body, and the object body is generally keyboard-scale. Remaining object imperfections still exist: the filled mesh is broad/solid over the key area and can overpaint beyond visible keys, so object registration is not perfect. But compared with the hand failure, object pose/completion is no longer the primary root blocker for Workbench item 3.

If object-side work resumes later, the live mechanisms are accepted TRELLIS hidden-surface placement, silhouette/key-grid pose constraints, and visibility-aware object rendering—not unsupported Poisson labels.

### Hand-object separation mechanism

The hand-object gap is present in the measurement/state, not a renderer coordinate bug. HaWoR MANO hand depth and UniDepth/SAM keyboard surface depth disagree at contact scale. P18 conservative output gating preserves HaWoR wrist/root translation because selected visible-surface support is zero, so the final v2 render carries the original MANO-depth problem.

Fresh v2 evidence:

- P18 v2 wrote 300 rows for both hands over 150 frames, but all final translations are effectively gated/preserved because there is no selected visible-surface support.
- Representative P19b frames show MANO skeletons separated from or wrongly ordered around the keyboard in world view; overlay labels still show centimeter residuals (e.g. around 7–12 cm on anchor/mid frames and larger in late frames).
- A focused `contact_translation_subset_v1` candidate constrained scale to ~1.0 and kept median 2D joint shift small (~1.6 px), but contact distance only improved from median ~11.2 cm to ~8.6 cm and visual inspection still showed large separation. This rejects “small global translation/depth offset with current articulation” as sufficient.

The next hand repair must add or infer a stronger MANO pose/depth/contact state. It cannot be another output gate, a silent snap, a metrics selector, or a scale-changing mask-depth fit. It likely needs explicit hand-owned surface/occlusion/contact reasoning that can update MANO pose/depth under occlusion while carrying uncertainty.

## Rejected mechanisms and claims

- Rejected: “P19b/P19c supportrepair artifacts are accepted physical annotation.” They failed final visual physical registration.
- Rejected: “anchorreview_v2 closes Workbench item 3.” It completes videos but fails MANO/object physical sanity.
- Rejected: “unsupported_uncertain Poisson fill is acceptable object body.” It is diagnostic uncertainty and is now excluded from accepted mesh semantics.
- Rejected: “a high contact prior or output-gate bypass justifies snapping MANO to the object.” Current gaps are too large and support selection is absent.
- Rejected: “translation-only contact/depth correction can repair the v2 hand state.” The focused subset candidate preserved scale/projection but left ~6–21 cm contact distances and visible separation.
- Rejected for the inspected frames: “visible hand-mask/depth refit is available after simply omitting object-mask subtraction.” Projection-only filtering recovered no usable support on frames 95/106/120/140/149.
- Rejected: metrics/P20/P21/autoresearch/item 4. They remain blocked until Workbench item 3 visual physical sanity passes.

## Live uncertainties

1. Whether the broad/solid accepted TRELLIS keyboard body needs further object-side refinement after the hand blocker is repaired.
2. What hand evidence can support a metric MANO correction when filtered support is absent on the rejected frames; projection-only filtering over frames 95/106/120/140/149 also produced no usable left/right support.
3. Whether an occlusion-aware MANO pose/depth refit using keyboard plane/contact constraints can improve hand relation without false contact certainty.
4. Whether available HaWoR/MANO parameters permit a fast enough runtime repair rather than an offline per-clip optimization.

## Next action

Do not run metrics/P20/P21/autoresearch. Workbench item 3 remains active. The next causal intervention must target the hand state: design and test a stronger prediction-side MANO pose/depth/contact repair that goes beyond global translation while preserving scale, image plausibility, and explicit uncertainty. The already-completed `anchorreview_v2` P19b render and the negative `contact_translation_subset_v1` render are comparison artifacts, not accepted deliverables.
