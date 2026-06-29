# Pipeline V19 current epistemic state

## Current supported claim

Workbench items 3, 4, 5, and one Workbench-6 iteration are complete for HOT3D `clip001851`, with a strict scope.

The current best mechanism is **source metric MANO plus a separate uncertain contact-surface hypothesis**. The metric MANO joint/root state remains the selected source hand state; the point-to-plane surface samples are rendered as an explicit uncertain contact/contact-surface posterior. This preserves 21-joint MANO evaluation while still visualizing the physically useful near-keyboard surface evidence. Contact ownership, signed nonpenetration, and certain hand-object contact remain unresolved.

Canonical evidence for the prior v4 branch showed the failure mechanism: v4 point-to-plane fitting made the rendered hand-keyboard relation plausible, but when its fitted transform was promoted into `optimized_joints_world_m`, HOT3D 21-joint localization regressed badly. On 109 candidate rows, runtime-HaWoR wrist median was `0.03547 m` and v4 was `0.10211 m`; joint MPJPE median was `0.04721 m` and v4 was `0.11305 m`.

The item-6 split-state branch confirms the causal model. On the same 109 candidate rows, source-MANO/contact-surface candidate and runtime HaWoR are exactly equal for wrist error, joint MPJPE, joint median error, root-aligned MPJPE, root-aligned median error, and root-aligned p95 error; max absolute row delta is `0.0 m`. Visual inspection still shows cyan surface samples near the keyboard/visible hand, but labels make clear that the cyan surface is separate from metric joints and contact is not accepted.

Key current branch artifacts:

- Split state: `$RUN/measurements/mano_interval_correction_maskdepth_nearsurface_plane_v4_surfacehyp_hawor_metric/$CASE/v18_joint_mano_interval_trajectory_state.json`.
- Render state: `$RUN/state/render_state/keyboard_rigid_render_state_maskdepth_nearsurface_plane_v4_surfacehyp_hawor_metric.json`.
- Full render: `$RUN/renders/keyboard_rigid_state_runtime_maskdepth_nearsurface_plane_v4_surfacehyp_hawor_metric_presentation/$CASE/`.
- Freeze: `$RUN/state/v19_prediction_freeze_manifest_surfacehyp_hawor_metric.json`.
- Evaluation: `$RUN/evaluation/hot3d_mano3d_surfacehyp_hawor_metric/`.
- Fair matched-row comparison SHA256: `1a95947d37edb2b9c6b3a0e5e3ebe1cffed004e47b3ecee999b38e413eee3163`.

## Current causal model

### Object-registration mechanism

The rejected `supportrepair_v2` artifact failed partly because unsupported Poisson fill and poor anchor/semantics promoted non-object surface into the accepted body. `anchorreview_v2` repaired object-side state by selecting frame 106 via visual candidate review and excluding `unsupported_uncertain` observed Poisson fill from the accepted/rendered body. Mesh-vs-UniDepth checks around frames `95/106/120/140/149` show keyboard mesh depth residuals near 0–4 mm median, with abs medians mostly below about 1.5 cm. Object registration is not the current root blocker, although the keyboard mesh is broad/solid and can overpaint keys.

### Hand support and depth mechanism

The earlier no-support conclusion was false. `build_v19_mano_mask_depth_refit_inputs.py` had hardcoded 960x540 / 0.5 projection scaling while clip001851 masks are 960x960 from 1408x1408 source. After repair, frames 95–149 recover 55/55 filtered masks for each hand. Fixed-scale mask/depth refit alone improves image/depth evidence but remains visually incoherent in world view, so it is a measurement repair, not the final physical mechanism.

### Contact-coupling mechanism

Object-mask-overlap-only contact selection was too sparse. v3 expanded contact candidates using full MANO vertices, explicit source-size projection, projected object mesh proximity, nearest object surface distance, and depth-order plausibility; this produced 109 rows but point-to-point residuals traded surface-normal improvement for tangential drift and scale pressure. v4 replaced point-to-point contact with a point-to-plane surface-normal residual, which matches the geometric failure mechanism: the dominant visual error was normal separation from the keyboard surface, while exact nearest-sampled-vertex equality was an invalid tangential target under occlusion and sparse contact.

The important correction is representational: surface-normal evidence is a **local uncertain contact-surface variable**, not a license to move the whole metric hand/root. A global Sim(3) contact fit couples local contact to all 21 joints and root translation, so it can improve a rendered surface relation while destroying metric MANO localization. The split-state mechanism breaks that bad coupling: keep metric MANO joints/root from a source with known evaluation behavior, and render the contact-surface posterior separately with explicit uncertainty.

### Metric mechanism

The HOT3D MANO3D evaluator scores 21-joint localization in camera 3D. It does not score object pose, contact, occlusion, nonpenetration, or surface-to-keyboard plausibility. The source-MANO/contact-surface branch intentionally makes the evaluator see the preserved metric MANO state, so its 21-joint metrics are equal to source HaWoR on candidate rows. This is not metric cheating if, and only if, the artifact labels the contact surface as a separate uncertain hypothesis rather than as the accepted metric MANO body.

## Rejected mechanisms and claims

- Rejected: “P19b/P19c supportrepair artifacts are accepted physical annotation.” They failed final visual physical registration.
- Rejected: “`anchorreview_v2` alone closes Workbench item 3.” It fixes object rendering but fails MANO/object physical sanity.
- Rejected: “unsupported_uncertain Poisson fill is acceptable object body.” It is diagnostic uncertainty and is now excluded from accepted mesh semantics.
- Rejected: “translation-only contact/depth correction can repair the v2 hand state.” It preserved scale/projection but left centimeter-scale separation.
- Rejected: “visible hand-mask/depth refit is unavailable on the rejected frames.” That was caused by the mask-size adapter bug.
- Rejected: “fixed-scale mask/depth MANO refit alone solves the hand-object relation.” It improves support/image/depth evidence but remains physically incoherent in world render.
- Rejected: “sparse object-mask-overlap contact similarity solves the interval.” It produced too few rows and incoherent rendered hypotheses.
- Rejected: “point-to-point nearest surface contact is the right residual.” It reduces distance but creates tangential artifacts and scale pressure.
- Rejected: “v4 point-to-plane fitted joints are a MANO metric improvement.” They are visually useful but worsen HOT3D 21-joint localization.
- Rejected: “surface-hypothesis rows prove contact.” They only prove an uncertain geometric surface posterior near the object; contact ownership/nonpenetration remain unresolved.

## Live uncertainties

1. Generalization beyond clip001851: the source-MANO/contact-surface split is mechanically general, but it must be rerun on the fixed HOT3D slice or next representative clip to test whether visual clarity and metric preservation hold across cases.
2. Contact semantics: the current branch visualizes a near-surface posterior, not accepted contact. A future factor should estimate contact ownership probability without moving metric MANO joints unless support is strong.
3. Mesh appearance: the keyboard mesh is physically usable but broad/solid. Presentation opacity repairs readability for clip001851, but object-side refinement may still be needed later.
4. Hand-surface posterior geometry: the cyan surface is not a full MANO mesh; it is sampled surface evidence. A future artifact may need better surface sampling/uncertainty visualization while preserving the metric hand state.

## Next action

Commit the item-6 mechanism and memory/spec updates after staged-diff review. Then continue Workbench item 6 by running the generalized P18b/P19/P20/evaluator path on the next fixed-slice clip or representative runtime branch, not by retuning clip001851 thresholds.
