# Pipeline V19 current epistemic state

## Current supported claim

Workbench items 3, 4, and 5 are complete for the corrected HOT3D keyboard runtime artifact. Workbench item 6 has a controlled mechanism chain on HOT3D `clip001851`, `clip001850`, and `clip001849`, with a strict scope:

**Metric MANO must remain separate from contact-surface hypotheses.** The best supported representation is source metric MANO joints/root plus separate uncertain surface/posterior variables. A local hand-object coupling can improve rendered surface relation, but it must not be promoted into the scored metric MANO state unless matched-row HOT3D and visual evidence prove improvement. Contact ownership, signed nonpenetration, and certain contact are not supported.

Evidence from `clip001851`: the raw v4 point-to-plane branch made the rendered hand-keyboard relation plausible but corrupted HOT3D 21-joint localization when fitted joints were promoted (`wrist median 0.03547 -> 0.10211 m`; `joint MPJPE median 0.04721 -> 0.11305 m` on 109 matched rows). The split-state branch kept the same surface evidence but restored exact HaWoR metric equality on those 109 rows: max absolute per-row delta `0.0 m` for wrist, joint MPJPE, joint median error, root-aligned MPJPE, root-aligned median error, and root-aligned p95.

Evidence from `clip001850`: the same point-to-plane mechanism, without retuned thresholds, built 196 surface rows. Surface-normal separation improved from `27.4 mm` to `10.5 mm` median while scale stayed pinned near 1.0. The split-state branch rendered full-duration overlay/world/side-by-side videos and preserved metric MANO exactly on the 196 matched rows.

Evidence from `clip001849`: the same v4 point-to-plane mechanism, without retuned thresholds, built 240 rows. Surface-normal separation improved from about `98.5 mm` to `9.7 mm` median, while tangent median increased from about `13.7 mm` to `51.6 mm`; this supports a normal-gap surface posterior, not tangential contact correspondence. The split-state branch preserved metric MANO exactly on the 240 matched rows. Visual review showed useful surface evidence around hand/key regions but persistent broad keyboard mesh/body and scattered world-view samples.

The direct object-surface posterior branch on `clip001849` reused v4 candidate selection, skipped the hand Sim(3), and wrote object-surface targets while preserving source metric MANO. It preserved metrics exactly on the same ordered 240 rows and confirmed the object-surface dots lie on the pose-transformed sampled keyboard mesh. It did **not** produce a near-contact relation: source hand-to-object distance median was about `103.5 mm`, normal gap median about `98.0 mm`, and visuals showed separated hands/object dots. This branch is evidence that object-surface provenance without hand/object registration is insufficient for near-contact annotation.

The source-gap correspondence renderer on `clip001849` consumed the direct branch’s stored source hand vertices and object-surface targets, rendered magenta source endpoints, yellow object endpoints, and orange source-gap links, then republished/froze/evaluated the full video. Matched-row HOT3D comparison again preserved metric MANO exactly on 240 candidate rows: all max absolute per-row deltas were `0.0 m`. Visual review across frames `0/30/75/120/149` and every-15-frame side-by-side sheets showed the layer is readable and honest, but the orange links mostly span from source hands to broad keyboard-body targets rather than to local contact patches. The source-gap renderer is therefore a diagnostic visualization of deterministic proximity links, not a contact mechanism, not an uncertainty interval in a probabilistic sense, and not Workbench-6 closure.

## Current causal model

### Object-registration mechanism

The rejected `supportrepair_v2` artifact failed partly because unsupported Poisson fill and poor anchor/semantics promoted non-object surface into accepted body. `anchorreview_v2` repaired object-side state by selecting a better anchor and excluding `unsupported_uncertain` observed Poisson fill from accepted/rendered body. Mesh-vs-UniDepth checks around frames `95/106/120/140/149` show keyboard mesh depth residuals near 0–4 mm median, with abs medians mostly below about 1.5 cm. Object registration is no longer the active root blocker for the keyboard slices, although the mesh remains broad/solid and can over-cover keys/table in world view.

### Hand support and depth mechanism

The earlier no-support conclusion was false. `build_v19_mano_mask_depth_refit_inputs.py` had hardcoded 960x540 / 0.5 projection scaling while clip001851 masks are 960x960 from 1408x1408 source. After repair, frames 95–149 recovered 55/55 filtered masks for each hand. Fixed-scale mask/depth refit alone improves image/depth evidence but remains visually incoherent in world view, so it is a measurement repair, not the final physical mechanism.

### Contact-coupling mechanism

Object-mask-overlap-only contact selection was too sparse. v3 expanded contact candidates using full MANO vertices, explicit source-size projection, projected object mesh proximity, nearest object surface distance, and depth-order plausibility. Point-to-point residuals reduced Euclidean distance but created tangential artifacts and scale pressure. v4 point-to-plane residual targeted surface-normal separation and made useful local surface posterior evidence, but promoting the fitted joints corrupted metric MANO. The split-state mechanism breaks that bad coupling: keep metric MANO from the source and render the contact-surface posterior separately.

The direct and sourcegap branches isolate a second failure mode: the current target-selection mechanism uses deterministic global nearest-neighbor object surface points. On a broad keyboard mesh, those targets can spread over the body and produce long source-gap links. The failure is systematic and mostly normal/depth separation with broad-surface attachment, not random per-frame jitter. The next causal intervention must change target selection or contact probability estimation, not just draw more links or globally move the hand.

### Metric mechanism

The HOT3D MANO3D evaluator scores 21-joint localization in camera 3D. It does not score object pose, contact, occlusion, nonpenetration, surface patch plausibility, or whether an orange source-gap link is physically meaningful. Equality to HaWoR confirms metric preservation only. It is legitimate evidence for the metric-MANO invariant and irrelevant to contact correctness except by ruling out accidental joint/root motion.

## Rejected mechanisms and claims

- Rejected: “P19b/P19c supportrepair artifacts are accepted physical annotation.” They failed final visual physical registration.
- Rejected: “`anchorreview_v2` alone closes Workbench item 3.” It fixes object rendering but fails MANO/object physical sanity.
- Rejected: “unsupported_uncertain Poisson fill is acceptable object body.” It is diagnostic uncertainty and is now excluded from accepted mesh semantics.
- Rejected: “translation-only contact/depth correction can repair the hand state.” It preserved scale/projection but left centimeter-scale separation.
- Rejected: “visible hand-mask/depth refit is unavailable on the rejected frames.” That was caused by the mask-size adapter bug.
- Rejected: “fixed-scale mask/depth MANO refit alone solves the hand-object relation.” It improves support/image/depth evidence but remains physically incoherent in world render.
- Rejected: “sparse object-mask-overlap contact similarity solves the interval.” It produced too few rows and incoherent rendered hypotheses.
- Rejected: “point-to-point nearest surface contact is the right residual.” It reduces distance but creates tangential artifacts and scale pressure.
- Rejected: “v4 point-to-plane fitted joints are a MANO metric improvement.” They are visually useful but worsen HOT3D 21-joint localization when promoted into metric joint/root state.
- Rejected: “surface-hypothesis rows prove contact.” They only prove an uncertain geometric surface posterior near the object; contact ownership/nonpenetration remain unresolved.
- Rejected: “direct object-surface posterior can replace v4 near-surface posterior.” It preserves metrics and semantics but leaves a `~103 mm` source hand-to-object gap and visually separated hands/object dots.
- Rejected: “source-gap correspondence rendering is a causal contact mechanism or Workbench-6 closure.” It renders pre-existing deterministic nearest-neighbor source/target pairs and exposes broad/wrong target attachment; it does not close the gap.
- Rejected: “orange source-gap links are uncertainty intervals.” They are single deterministic 1:1 proximity links, not uncertainty regions or probability bounds.
- Rejected: “summary report comparison across different row counts is causal metric evidence.” Use exact frame/side matched-row comparison for metric preservation claims.

## Live uncertainties

1. Generalization beyond keyboard HOT3D slices: the split-state invariant is mechanically general, but physical evidence remains keyboard-only.
2. Contact semantics: no current branch supports accepted contact ownership or nonpenetration. Future output must render probability/uncertainty rather than pretend contact.
3. Target selection: global nearest-neighbor object targets are a live root failure on broad objects. A locality-constrained target query or contact-probability factor is the next discriminating intervention.
4. Mesh appearance: the keyboard mesh is usable but broad/solid. Local target restriction may reduce wrong links, but it will not by itself refine object geometry.
5. Runtime: v4 point-to-plane fitting and full rendering are slow. Direct/local-patch posterior branches avoid hand optimization and should be preferred for diagnostic target-selection experiments.

## Next action

Run the prepared local-patch object-surface posterior branch after syncing the committed target-locality code. Prediction: if global nearest-neighbor target selection caused the broad sourcegap links, restricting each object target to projected object mesh samples near the source MANO vertex should localize/shorten links while preserving metric MANO exactly. Falsifiers: too few rows/vertices survive, links remain broad/long despite locality, targets attach to visibly wrong object regions, or any matched-row MANO delta becomes nonzero. This branch should still be reported as a posterior/contact-probability diagnostic, not accepted contact.
