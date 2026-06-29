# Pipeline V19 current epistemic state

## Current supported claim

Workbench items 3, 4, and 5 are complete for HOT3D `clip001851` on the `maskdepth_nearsurface_plane_v4` branch, with a strict scope.

Supported physical-artifact claim: the frozen V19 artifact is full-duration and audience-readable. It renders a real state-consumed rigid keyboard body plus an uncertain optimized MANO surface hypothesis. In the interaction interval, cyan optimized MANO surface samples align with the visible hands and lie on/near the keyboard surface, removing the previous obvious detached/misordered hand-keyboard failure. Contact ownership, signed nonpenetration, and metric-perfect MANO are **not** accepted.

Supported evaluation claim: the v4 point-to-plane correction is **not** a HOT3D 21-joint MANO accuracy improvement. On the fair 109 candidate rows where v4 has optimized joints, it worsens runtime-HaWoR wrist and MPJPE metrics. This metric result bounds the claim: v4 improves the rendered uncertain hand-surface/keyboard relationship but corrupts evaluated joint localization.

Canonical frozen prediction artifacts for this branch:

- Freeze manifest: `$RUN/state/v19_prediction_freeze_manifest.json`, SHA256 `2e829cdc16ee6602e5893cac91ed29a40e30eef64dda887e279fd515a8f5b7b7`, status `frozen`, `hot3d_scoring_run=false` at freeze.
- Canonical videos: `$RUN/renders/v19_overlay.mp4`, `$RUN/renders/v19_world.mp4`, `$RUN/renders/v19_side_by_side.mp4`.
- Presentation branch: `$RUN/renders/keyboard_rigid_state_runtime_maskdepth_nearsurface_plane_v4_surface_presentation/$CASE/`.
- Evaluation: `$RUN/evaluation/hot3d_mano3d_maskdepth_nearsurface_plane_v4/`.

## Current causal model

### Object-registration mechanism

The rejected `supportrepair_v2` artifact failed partly because unsupported Poisson fill and poor anchor/semantics promoted non-object surface into the accepted body. `anchorreview_v2` repaired object-side state by selecting frame 106 via visual candidate review and excluding `unsupported_uncertain` observed Poisson fill from the accepted/rendered body. Mesh-vs-UniDepth checks around frames `95/106/120/140/149` show keyboard mesh depth residuals near 0–4 mm median, with abs medians mostly below about 1.5 cm. Object registration is therefore not the primary root of the remaining MANO metric failure, although the keyboard mesh is broad/solid and can overpaint keys.

### Hand support and depth mechanism

The earlier no-support conclusion was false. `build_v19_mano_mask_depth_refit_inputs.py` had hardcoded 960x540 / 0.5 projection scaling while clip001851 masks are 960x960 from 1408x1408 source. After repair, frames 95–149 recover 55/55 filtered masks for each hand. Fixed-scale mask/depth refit alone improves image/depth evidence but remains visually incoherent in world view, so it is a measurement repair, not the final physical mechanism.

### Contact-coupling mechanism

Object-mask-overlap-only contact selection was too sparse. v3 expanded contact candidates using full MANO vertices, explicit source-size projection, projected object mesh proximity, nearest object surface distance, and depth-order plausibility; this produced 109 rows but point-to-point residuals traded surface-normal improvement for tangential drift and scale pressure. v4 replaced point-to-point contact with a point-to-plane surface-normal residual. This matches the observed visual failure mechanism: before-correction errors were mostly normal separation from the keyboard surface, while exact nearest-sampled-vertex equality was an invalid tangential target under occlusion and sparse contact.

The visual artifact accepted by item 3/4 is therefore an uncertainty-preserving surface hypothesis, not a snap and not contact closure. It uses a bounded similarity correction, preserves scale, retains baseline skeleton provenance in diagnostic renders, and marks contact as not accepted. The cyan optimized surface samples are the physical surface hypothesis to inspect; orange/white skeletons are provenance/baseline.

### Metric mechanism

The HOT3D MANO3D evaluator scores 21-joint localization in camera 3D. It does not score object pose, contact, occlusion, nonpenetration, or surface-to-keyboard plausibility. Evaluation after freeze required an evaluation-only copy of the v4 state with `source_hawor_npz` added for camera trajectory; the frozen prediction state was not modified.

Metric observation on the fair 109 candidate rows: runtime HaWoR baseline wrist median `0.03547 m`, v4 `0.10211 m`; baseline joint MPJPE median `0.04721 m`, v4 `0.11305 m`; baseline root-aligned MPJPE median `0.02122 m`, v4 `0.04036 m`. Evaluator review projections land on visible hands, so this is not an obvious camera/review artifact. The mechanism is that the surface-normal correction moves the MANO joint/root state to make a plausible uncertain surface relation to the keyboard, but that movement damages the 21-joint state against HOT3D GT.

## Rejected mechanisms and claims

- Rejected: “P19b/P19c supportrepair artifacts are accepted physical annotation.” They failed final visual physical registration.
- Rejected: “`anchorreview_v2` alone closes Workbench item 3.” It fixes object rendering but fails MANO/object physical sanity.
- Rejected: “unsupported_uncertain Poisson fill is acceptable object body.” It is diagnostic uncertainty and is now excluded from accepted mesh semantics.
- Rejected: “translation-only contact/depth correction can repair the v2 hand state.” It preserved scale/projection but left centimeter-scale separation.
- Rejected: “visible hand-mask/depth refit is unavailable on the rejected frames.” That was caused by the mask-size adapter bug.
- Rejected: “fixed-scale mask/depth MANO refit alone solves the hand-object relation.” It improves support/image/depth evidence but remains physically incoherent in world render.
- Rejected: “sparse object-mask-overlap contact similarity solves the interval.” It produced too few rows and incoherent rendered hypotheses.
- Rejected: “point-to-point nearest surface contact is the right residual.” It reduces distance but creates tangential artifacts and scale pressure.
- Rejected: “v4 point-to-plane is a MANO metric improvement.” It is visually useful for uncertain surface/contact rendering but worsens HOT3D 21-joint localization on candidate rows.

## Live uncertainties

1. Representation split: whether the correct next design is to keep baseline/metric MANO joints as the evaluated hand state while rendering a separate uncertain near-surface MANO surface/contact hypothesis, instead of replacing joints with the surface-fit state.
2. Contact semantics: v4 supports near-surface uncertain interaction, not accepted contact ownership. Any future artifact must avoid collapsing this into a hard contact claim.
3. Mesh appearance: the keyboard mesh is physically usable but broad/solid. Presentation opacity repairs readability for clip001851, but object-side refinement may still be needed later.
4. Left frame 141 lacks a v4 temporal MANO row, and early-left rows have larger visible shifts. These are carried as uncertainty.

## Next action

A scoped evidence checkpoint should be committed: near-surface/point-to-plane interval refitter, render-surface/label improvements, and task memory. The next Workbench phase after item 5 is research/iteration, but it must start from the negative metric mechanism: preserve the visually sane uncertain surface relation while preventing surface contact correction from degrading the evaluated 21-joint MANO state.
