# Pipeline V19 current epistemic state

## Current supported claim

Workbench items 3, 4, and 5 are complete for the corrected HOT3D keyboard runtime artifact. Workbench item 6 now has a two-clip controlled mechanism result on HOT3D `clip001851` and `clip001850`, with a strict scope.

The current best mechanism is **source metric MANO plus a separate uncertain contact-surface hypothesis**. Metric MANO joint/root state remains the selected source hand state; point-to-plane MANO surface samples are rendered as an explicit uncertain contact-surface posterior. This preserves 21-joint MANO evaluation while still visualizing near-object surface evidence. Contact ownership, signed nonpenetration, certain hand-object contact, and broad non-keyboard generalization remain unresolved.

Evidence from `clip001851`: the raw v4 point-to-plane branch made the rendered hand-keyboard relation plausible but corrupted HOT3D 21-joint localization when promoted into `optimized_joints_world_m` (`wrist median 0.03547 -> 0.10211 m`; `joint MPJPE median 0.04721 -> 0.11305 m` on 109 matched candidate rows). The split-state branch kept the same surface evidence but restored exact HaWoR metric equality on the same 109 rows: max absolute per-row delta `0.0 m` for wrist, joint MPJPE, joint median error, root-aligned MPJPE, root-aligned median error, and root-aligned p95.

Evidence from `clip001850`: the same point-to-plane mechanism, without retuned thresholds, built 196 surface rows. Surface-normal separation improved from `27.4 mm` to `10.5 mm` median while scale stayed pinned near 1.0. The split-state branch rendered full-duration overlay/world/side-by-side videos and preserved metric MANO exactly on the 196 candidate rows. Matched-row HOT3D comparison has max absolute per-row delta `0.0 m` for wrist, joint MPJPE, joint median error, root-aligned MPJPE, root-aligned median error, and root-aligned p95. Visual review showed cyan surface samples near the hand/keyboard interaction in frames 32, 50, 74, and 75, while frames 120 and 149 honestly rendered mostly source-MANO-only state when no contact-surface posterior was present.

Key current clip001850 artifacts:

- Split state: `$RUN/measurements/mano_interval_correction/keyboard_0_149_surface_hypothesis_metric_mano/$CASE/v18_joint_mano_interval_trajectory_state.json`.
- Render branch: `$RUN/renders/keyboard_rigid_state_runtime_surfacehyp_metric_generalization_presentation/$CASE/`.
- Published branch: `$RUN/renders/v19_published_surfacehyp_metric_generalization_presentation/` and canonical `$RUN/renders/v19_overlay.mp4`, `$RUN/renders/v19_world.mp4`, `$RUN/renders/v19_side_by_side.mp4`.
- Render manifest SHA256: `f31b12aac0ce433d495c471da54fdbbe7e2c9c3fdb96bbeb05c9f92ab95ffc78`.
- Matched-row comparison SHA256: `d77a23c7b323aad3e37ac41e699beedc15ae2053fc2b5eeb08829c6c7c592e9a`.
- Local visual review sheet: `/tmp/v19_clip001850_surfacehyp_metric_generalization/clip001850_surfacehyp_metric_generalization_full_review_sheet.jpg`.

## Current causal model

### Object-registration mechanism

The rejected `supportrepair_v2` artifact failed partly because unsupported Poisson fill and poor anchor/semantics promoted non-object surface into the accepted body. `anchorreview_v2` repaired object-side state by selecting frame 106 via visual candidate review and excluding `unsupported_uncertain` observed Poisson fill from the accepted/rendered body. Mesh-vs-UniDepth checks around frames `95/106/120/140/149` show keyboard mesh depth residuals near 0–4 mm median, with abs medians mostly below about 1.5 cm. Object registration is no longer the active root blocker for the keyboard slices, although the mesh remains broad/solid and visually over-covers keys/table in some views.

### Hand support and depth mechanism

The earlier no-support conclusion was false. `build_v19_mano_mask_depth_refit_inputs.py` had hardcoded 960x540 / 0.5 projection scaling while clip001851 masks are 960x960 from 1408x1408 source. After repair, frames 95–149 recover 55/55 filtered masks for each hand. Fixed-scale mask/depth refit alone improves image/depth evidence but remains visually incoherent in world view, so it is a measurement repair, not the final physical mechanism.

### Contact-coupling mechanism

Object-mask-overlap-only contact selection was too sparse. v3 expanded contact candidates using full MANO vertices, explicit source-size projection, projected object mesh proximity, nearest object surface distance, and depth-order plausibility. Point-to-point residuals reduced Euclidean distance but created tangential artifacts and scale pressure. v4 point-to-plane residual targets the actual observed failure mode: surface-normal separation from the keyboard; tangential equality to a sampled nearest vertex is invalid under sparse/occluded contact.

The critical representation rule is now supported on two HOT3D clips: surface-normal evidence is a **local uncertain contact-surface variable**, not a license to move the whole metric hand/root. A global Sim(3) contact fit couples local contact to all 21 joints and root translation, so it can improve a rendered surface relation while damaging metric MANO localization. The split-state mechanism breaks that bad coupling: keep metric MANO joints/root from a source with known evaluation behavior, and render the contact-surface posterior separately with explicit uncertainty.

### Metric mechanism

The HOT3D MANO3D evaluator scores 21-joint localization in camera 3D. It does not score object pose, contact, occlusion, nonpenetration, or surface-to-keyboard plausibility. The source-MANO/contact-surface branch intentionally makes the evaluator see the preserved metric MANO state; therefore equality to HaWoR is the expected metric result and confirms preservation, not improvement. This is legitimate only because the render labels and visuals keep the cyan surface separate from metric MANO and state that contact remains uncertain.

## Rejected mechanisms and claims

- Rejected: “P19b/P19c supportrepair artifacts are accepted physical annotation.” They failed final visual physical registration.
- Rejected: “`anchorreview_v2` alone closes Workbench item 3.” It fixes object rendering but fails MANO/object physical sanity.
- Rejected: “unsupported_uncertain Poisson fill is acceptable object body.” It is diagnostic uncertainty and is now excluded from accepted mesh semantics.
- Rejected: “translation-only contact/depth correction can repair the v2 hand state.” It preserved scale/projection but left centimeter-scale separation.
- Rejected: “visible hand-mask/depth refit is unavailable on the rejected frames.” That was caused by the mask-size adapter bug.
- Rejected: “fixed-scale mask/depth MANO refit alone solves the hand-object relation.” It improves support/image/depth evidence but remains physically incoherent in world render.
- Rejected: “sparse object-mask-overlap contact similarity solves the interval.” It produced too few rows and incoherent rendered hypotheses.
- Rejected: “point-to-point nearest surface contact is the right residual.” It reduces distance but creates tangential artifacts and scale pressure.
- Rejected: “v4 point-to-plane fitted joints are a MANO metric improvement.” They are visually useful but worsen HOT3D 21-joint localization when promoted into metric joint/root state.
- Rejected: “surface-hypothesis rows prove contact.” They only prove an uncertain geometric surface posterior near the object; contact ownership/nonpenetration remain unresolved.
- Rejected: “summary report comparison across different row counts is causal metric evidence.” Use exact frame/side matched-row comparison for metric preservation claims.

## Live uncertainties

1. Generalization beyond keyboard HOT3D slices: the source-MANO/contact-surface split is mechanically general, but evidence so far is keyboard-only. Another fixed-slice clip or a project representative object is needed for broader scope.
2. Contact semantics: the current branch visualizes a near-surface posterior, not accepted contact. A future factor should estimate contact ownership probability without moving metric MANO joints unless support is strong.
3. Mesh appearance: the keyboard mesh is physically usable but broad/solid. Presentation opacity repairs readability, but object-side refinement may still be needed.
4. Hand-surface posterior geometry: cyan samples are not a full MANO mesh and can scatter broadly in world view. The artifact remains honest because the samples are rendered as uncertainty, but better surface sampling/uncertainty visualization may be needed.
5. Runtime: full-vertex point-to-plane fitting plus full rendering is slow and poorly observable while running. The current implementation is acceptable as Workbench research evidence, but it may need acceleration/incremental logging before it becomes the default runtime path.

## Next action

Continue Workbench item 6 by testing the split-state mechanism on the next fixed-slice/generalization target without threshold retuning, or by converting the slow full-vertex point-to-plane branch into a bounded/default-runtime mechanism. Any next run must preserve the same causal invariant: metric MANO joints remain separate from uncertain contact-surface hypotheses until a mechanism proves joint/root improvement without visual regression.
