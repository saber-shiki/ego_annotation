# V18 Factor-Graph Design Contract

This note is the current design target. It defines physical graph semantics that implementation and artifact inspection must be judged against.

## Desired Physical State

V18 estimates a full-video physical annotation state, not a collection of labels. The primary solved state is the metric MANO hand annotation. Object/part pose, latent contact, nonpenetration, and occlusion variables are useful only insofar as they constrain, improve, validate, or falsify that hand state. For each frame, the solved state must explain visible evidence and carry uncertainty where sensors do not determine the state.

The graph state is the source of truth for user-facing physical claims, with this priority:

- metric MANO hand state `H_{t,h}` as the primary delivered annotation;
- object or part pose/state as contact/occlusion/nonpenetration context for hand-state inference;
- latent contact mode and contact patch/anchor state as constraints on hand state, not deliverables by themselves;
- nonpenetration state;
- occlusion ownership and visibility state;
- uncertainty/residual state.

Raw masks, VLM schema, detector tracks, contact proposals, temporal episodes, depth rows, audit measurements, and rendered marks are measurements/provenance. They are not final physical state.

## Variable Families

### Hand State

`H_{t,h}` is the metric MANO hand state for frame `t` and hand `h`, with MANO surface or reproducible MANO parameters, camera/world-frame semantics, provenance, and uncertainty. It is the primary variable the graph must improve or validate.

HaWoR metric MANO is the primary physical hand source. WiLoR/RTMLib/boxes may support association but must not replace metric MANO for contact or pose-fill claims. A graph/contact/occlusion change that leaves the final MANO joints, vertices/parameters, and uncertainty unchanged is not principled progress on the primary objective.

### Rigid Object State

For rigid or compact pose-eligible object `o`:

- `G_o`: object-frame geometry with explicit scope: complete, visible-surface-only, or uncertain completion.
- `T_{t,o} ∈ SE(3)`: object pose in V18 world/camera convention.

A rigid pose claim must attach to `G_o` transformed by `T_{t,o}`. A box, mask, centroid, class label, VLM physical type, or frame-local visible surface alone is not rigid body state.

If only visible surfaces are available, pose and geometry claims must be scoped as visible-surface-only or uncertain.

#### Compact Rigid Full-Shape / Pose Inference

For a compact rigid manipulated object, full shape may be inferred from partial observations, but only as an object-agnostic posterior over shape and pose. The method must not use category primitives, object-specific rules, or hand-picked proxy gates.

Variables for object instance `o`:

- `S_o`: canonical closed surface, mesh, or implicit field with uncertainty over unseen regions.
- `T_{t,o} ∈ SE(3)`: per-frame object pose.
- `V_{t,o}`: visibility/ownership state for observed object surface, occlusion, and out-of-frame regions.
- Coupled hand state `H'_{t,h}` only after `S_o`/`T_{t,o}` provide a physically valid contact/nonpenetration factor.

Admissible observations/factors:

- model-produced object masks and track identity;
- metric depth points on visible object pixels;
- camera intrinsics and world/camera transforms;
- silhouette consistency between projected `T_{t,o} S_o` and the object mask;
- depth consistency between projected/visible `T_{t,o} S_o` and observed object-owned depth;
- free-space constraints along camera rays before observed object depth;
- temporal smoothness of `T_{t,o}` for a rigid object;
- contact and nonpenetration factors only after hand evidence is valid;
- VLM/model judgment only as an uncertain prior over physical type/action context, never as geometry.

Optimization objective:

`S_o`, `T_{t,o}`, and visibility are estimated jointly or by a bounded alternating solver that monotonically reduces a stated robust energy. The energy must include visible depth residuals, silhouette residuals, free-space/occupancy constraints, temporal pose smoothness, and shape regularization such as closure/smoothness/minimal unsupported surface area. Robust residual scales should come from sensor/model uncertainty or be estimated from residual distributions, not selected per object/example.

Required causal behavior:

- Visible-surface PCA/centroid may initialize pose but is not the final pose estimate.
- Multi-frame visible surfaces must be fused in the canonical object frame under the current pose estimate, then poses must be re-estimated against the inferred shape; one pass of visible-surface fusion is not enough.
- Unobserved regions may be completed only as uncertain shape posterior support consistent with observed depth, masks, free space, and compact rigid priors. Unknown hidden surfaces must not be presented as certain geometry.
- A full-shape claim must report which surface regions are observed, inferred, or unresolved.
- A posed full-shape claim must be rejected or marked uncertain in frames where projected depth/silhouette contradicts observations.
- The solved `S_o`/`T_{t,o}` matters for V18 only if it constrains or validates `H'_{t,h}`; object-only reconstruction without hand-state effect is support work, not the primary deliverable.

Tractability requirement:

The default path must be feed-forward or bounded iterative at the object-instance level: downsampled object-owned depth/ray factors, sparse SE(3) trajectory variables, and a compact mesh/implicit representation. Per-instance neural-field optimization, BundleSDF, NeRF, or long training loops are not the default path.

### Part and Articulation State

For part-required object `o` with part `p`:

- `G_{o,p}`: part geometry with explicit scope.
- `T_{t,o,p} ∈ SE(3)`: part pose.
- `A_{t,o}` or `θ_{t,o}`: articulation parameter where fitted and supported.

Part contact claims must attach to part contact state and validated part geometry/pose. Current-frame visible parts are observations; accepted global part tracks define required/missing parts.

### Articulated / Part Local Contact State

For an articulated or part-required object, full-object contact must not be inferred by applying the rigid local patch rule to the parent object. The graph needs a part-scoped contact state `P^part_{t,h,o,p}` / `C_{t,h,o,p}` before active contact can be claimed.

Required support:

- the object schema requires part/relative-motion modeling or has an accepted part track;
- a same-frame part state exists or an explicit occlusion/temporal part-state variable represents the part at time `t` with uncertainty;
- current observed MANO and independent image/VLM association support the hand/part interaction;
- MANO-to-part visible-surface or part-patch residual is locally close under the represented part pose/geometry;
- hand-footprint-excluded object/part-owned depth is compatible, or unresolved depth is carried as uncertainty rather than converted to active contact;
- no nonpenetration conflict;
- contact may constrain `T_{t,o,p}` only through an emitted stable part pose-anchor factor with validated part geometry. A part contact state alone must not move the parent object `T_{t,o}`.

If an articulated object has compatible parent-object depth and image association but no part state at that frame, the correct state is missing/unresolved part contact evidence, not active parent-object contact.

#### Dominant Visible-Part Surface State

A prior proposed dominant visible-part mechanism was falsified: it reinterpreted the parent object's visible surface as a lid/rim part surface, used VLM keyword text and hand-picked proxy gates, and could label body contact as part contact. That mechanism is invalid.

V18 may instantiate a part-scoped dominant visible-surface state `G^dom_{t,o,p}` / `T^dom_{t,o,p}` only from model-produced or geometry-measured part-specific evidence, not from the parent object mask/visible surface alone. Required support:

- the object schema requires part/relative-motion modeling and has an accepted global part label for `p`;
- current-frame or temporally propagated model-produced part mask/surface evidence exists for `p`, with measured dominance or measured local support for the actual part surface;
- any text/VLM evidence is provenance for proposing the part, not a keyword gate that substitutes for part geometry;
- part/contact association uses MANO surface, segmentation/depth, reconstructed geometry, or model-produced perception tied to the part surface, not hand-picked scalar proxies;
- the dominant state is serialized as a part-scoped variable and must not be treated as parent object `SE(3)` or complete object geometry;
- it may support part-local contact only if it constrains or validates the MANO hand state with explicit uncertainty and no parent-object pose correction;
- if the object mask includes inseparable can/body geometry or the part identity is ambiguous, the state remains unresolved rather than active contact.

This mechanism exists only to avoid a false missing-part state when the current observation truly measures the moving part. It must not relabel parent-object geometry as part geometry.

### Deformable / Nonrigid State

For deformable objects, V18 must instantiate a real nonrigid state or explicitly carry missing mechanism uncertainty.

Current scoped variable allowed: local visible-surface patch state at solved active contact. It is not whole-object pose, hidden geometry, or full nonrigid reconstruction.

### Latent Contact State

For each candidate hand/object, hand/part, or hand/deformable interaction:

- `C_{t,h,x}`: latent contact mode / contact phase state.
- `P_{t,h,x}`: local contact patch, object-frame anchor, or contact manifold state where applicable.

`C` is the physical contact relation. `P` is the local geometric support for contact. Neither is identical to raw distance, proposal rows, temporal episode labels, rendered lines, or emitted pose-anchor measurements.

A pose-anchor factor is an intermittent constraint handle generated by contact evidence. It may constrain `T_{t,o}` or `T_{t,o,p}`, but per-frame pose-anchor emission is not the existence condition for contact.

Active contact is a posterior graph state. It must be explained by contact mode/patch state, association evidence, solved geometric residuals, temporal continuity, and absence of unresolved depth/nonpenetration contradictions. It is not a success metric by itself; it must constrain, update, validate, or falsify `H_{t,h}` to count as progress on V18.

### Nonpenetration State

`N_{t,h,x}` represents local nonpenetration evidence/constraint between MANO and object/part/deformable geometry. It may veto/penalize contact or expose uncertainty; it must not substitute for object geometry or contact state.

### Visibility and Occlusion State

For each relevant hand/object/part:

- visibility state: visible, partially visible, occluded, out-of-frame, unresolved;
- occlusion owner where supported;
- depth-order evidence and uncertainty.

Occlusion ownership is a graph variable supported by visibility/depth/temporal evidence, not a missing-data label. Pose fill through occlusion requires accepted owner plus metric evidence; ownership alone is insufficient.

## Measurement and Factor Families

### Hand Observation Factors

Constrain `H_{t,h}` to HaWoR metric MANO observations with weights from depth-scale support, sample count, reprojection/depth support, and visibility state. Contact, nonpenetration, occlusion, and temporal factors must enter the objective as constraints or penalties on `H_{t,h}` when physically valid. Inferred rows may regularize continuity but must not create certain contact or pose-fill claims.

### Object Geometry and Pose Factors

Constrain `G_o` and `T_{t,o}` using mask/silhouette agreement, depth point alignment, temporal pose continuity, multiframe geometry consistency, nonpenetration, and contact factors when contact is solved and physically eligible.

Visible-surface PCA/centroid observations can initialize or weakly constrain visible pose, but they are not full object pose.

### Part and Articulation Factors

Constrain part pose and articulation from part mask/depth agreement, visible-surface geometry, temporal continuity, articulation residuals, and contact factors to validated part geometry. Underconstrained articulation remains uncertain.

### Contact Factors

Contact factors constrain the latent contact mode/patch state and optionally continuous pose variables:

- visual/VLM prior factor on `C_{t,h,x}` from 2D frame/action context;
- MANO-to-geometry distance likelihood factor, with measurement uncertainty;
- local contact patch residual factor on `P_{t,h,x}`;
- temporal contact persistence and make/break factors on `C`;
- slip/relative-motion consistency where available;
- nonpenetration/depth-order compatibility factors;
- optional pose-anchor factor from `C/P` to `T_{t,o}` or `T_{t,o,p}`.

Raw distance is a likelihood/capture signal, not a hard final prerequisite. Independent association can propose contact over the measurement uncertainty scale. Final active contact is accepted only when the posterior solved state gives a plausible contact explanation and no stronger no-contact/depth/nonpenetration explanation dominates.

Temporal episode support may preserve contact continuity only when anchored by direct association/visual/metric evidence and not contradicted by separation, slip, depth order, or nonpenetration. Episode continuity alone is not active contact.

Implementation-level distinction required by the current Workbench: a temporal episode row is only a persistence factor, not a solved contact variable. A posterior rigid temporal contact state may become active only when the solved contact switch `C_{t,h,o}` is on, the frame lies inside an eligible anchor-supported episode, independent image/VLM association remains positive, observed metric hand support is present, and no stronger separation, raw depth-order, or nonpenetration explanation dominates. `accepted_contact_owner` is conditional evidence about a candidate contact explanation and must not be used as independent evidence that contact exists. Bridge frames require previous and next anchors within the configured temporal bound; one-sided nearest-anchor dilation is only a hypothesis/tail, not active persistence. Raw depth contradictions must remain immutable observations with magnitude/uncertainty provenance. A raw depth conflict may support active contact only when a represented physical explanation accounts for it: numerically weak raw depth evidence plus direct visual/metric contact prior, a represented rigid occluded-contact patch/depth interval with uncertainty, or a local deformable patch contact with direct pre-patch association and close local residual. Episode membership, accepted owner, proximity, high overlap, high mesh-distance support, an `occluded_contact_patch_anchor` label, or renderer state must not erase a depth contradiction. Strong raw depth contradictions must demote rigid temporal contacts unless an actual occluded-patch state or local-deformable-patch mechanism explains them. This posterior state may support an active contact claim but must be scoped as contact-mode state only; it must not emit or substitute for full-object `SE(3)` correction. Same-frame visible-surface anchors and pose-anchor factors remain observations/constraints around `C`, not the definition of contact. Temporal-only `C_t` without a patch/anchor `P_t` must render as a non-spatial state/uncertainty mark, not as a metric hand-object contact edge.

### Local Rigid Visible-Surface Contact State

For rigid or compact objects whose full same-frame object pose is weak, V18 may still instantiate a local visible-surface contact state `P^vis_{t,h,o}`. In the current bounded graph this is serialized as `local_rigid_visible_contact_patch::{frame_idx}::{hand_side}::{object_id}` plus the `rigid_local_visible_surface_contact_state` support path. This is a contact-manifold/patch feasibility state, not a full-object pose claim.

Required support:

- current metric MANO hand state is observed and depth-scaled;
- independent association such as pair-contact image evidence or a visual/VLM contact prior, not `accepted_contact_owner` alone;
- current hand-to-visible-surface residual is locally close (millimetre/centimetre scale) on same-frame visible geometry;
- current hand-footprint-excluded object-owned local depth is compatible; any in-front/tail/insufficient local depth contradiction blocks this state unless a separate represented hidden-patch/deformable mechanism applies;
- no nonpenetration conflict;
- the state explicitly records that it affects latent contact `C_t` only and does not affect `object_se3`, part pose, or hidden object geometry.

This state exists to avoid a false dependency between local contact evidence and full rigid pose support. It must not be used to move full object pose, and it must not upgrade weak visible-surface geometry into reconstructed object pose.

### Rigid Occluded-Contact Patch Depth State

For strong raw hand-behind-object depth contradictions, V18 must instantiate a represented hidden-patch feasibility state before any rigid occluded contact can remain active. The minimal bounded state is `P^occ_{t,h,o}` with:

- current-hand raw hand-minus-object depth-gap statistics computed from the graph's current metric MANO hand state `H_{t,h}` against object-owned depth. The object depth must come from pixels owned by the object mask/visible surface after excluding a dilated projection of the same current hand, or from a posed object visible-surface ray/patch with equivalent ownership semantics. Depth at the hand-projection pixel is diagnostic only because it may be hand/foreground depth; it must not serve as object contact depth. If hand-footprint exclusion leaves no nearby object-owned depth, the measurement is unresolved or contradictory according to its local statistics, not silently replaced by hand-pixel scene depth;
- local contact-patch depth statistics are distinct from broad near-mask hand-cloud statistics. A broad p95 tail over all hand vertices near an object mask can diagnose uncertainty, but active contact support must come from a represented local patch statistic with bounded mask distance, bounded median gap, bounded p95 gap, enough vertices, independent association, and no nonpenetration conflict;
- any legacy/stale pairwise depth-gap rows may be retained only as provenance or audit contrast, not as the admissibility/raw-contradiction source when a current object-owned measurement exists;
- a posed-object camera-depth interval derived from transformed object geometry (`world_bbox_corners_m` and/or visible/depth-fused world vertices) under the current camera transform;
- an explicit uncertainty bound combining object-depth validation residual and minimum depth uncertainty;
- an independent association requirement from image/VLM contact evidence, not `accepted_contact_owner` alone;
- nonpenetration and observed-hand-support compatibility;
- a Boolean support estimate plus a falsifiable state such as `supported_occluded_patch_depth_interval_compatible`, `physically_incompatible_raw_depth_gap_exceeds_reliable_object_depth_interval`, `unresolved_unreliable_object_depth_interval`, or `unresolved_missing_object_depth_interval`.

Mechanism prediction: if the raw hand-behind-object gap exceeds a reliable posed object's hidden depth interval plus uncertainty, the represented occluded patch is physically incompatible and must block active rigid contact even when visual overlap, temporal manipulation labels, and owner/proximity evidence are strong. User-facing `depth_occluded_contact_possible` projection/rendering is allowed only when a represented hidden-depth/patch state is supported or remains unresolved; raw contact energy, near visible geometry, overlap, owner, or a renderer branch are not enough. A reliable physical impossibility, missing association, missing hand support, nonpenetration conflict, or absent represented patch state is a contradicted/nonrendered noncontact state, not an unresolved possible contact. If the interval itself is unreliable because object pose/visible-depth silhouette support is weak, the correct state is unresolved, not a proof of real-world impossibility and not active contact. If the interval is compatible in a future case, the same represented patch state may explain the depth conflict, but it still does not by itself move full object `SE(3)` unless a separate pose-eligible contact factor is emitted and residual/correction bounds are satisfied.

### Nonpenetration Factors

Compare MANO surface and physically eligible object/part/deformable geometry. They may block active contact, penalize poses, or mark local geometry insufficient. They must not move pose without a physically meaningful geometry model.

### Occlusion Factors

Couple visibility, owner variable, depth order, and temporal evidence. Accepted ownership requires positive evidence against unresolved/unowned alternatives. Contact or pose fill through occlusion requires additional metric geometry evidence.

## Objective and Solve Schedule

The design target is a joint hybrid physical graph. Runtime constraints allow bounded alternating implementation if semantics are preserved.

Valid bounded schedule:

1. Initialize hand/object/part/deformable/contact variables from measurements with uncertainty.
2. Estimate continuous pose/state variables from observation and temporal factors.
3. Estimate latent contact, contact patch, nonpenetration, visibility, and occlusion variables from current geometry and measurements.
4. Feed solved contact/nonpenetration/occlusion factors back into eligible continuous variables.
5. Iterate until stable, or carry explicit uncertainty.
6. Write final physical state only from solved variables and factor evidence.

If a fixed point oscillates, preserve uncertainty or stable latent contact state with uncertainty. Do not choose whichever pass gives a desired label.

## Uncertainty Semantics

Uncertainty is graph state, not omission.

Examples:

- missing depth samples: invalid/unavailable measurement;
- visible-surface-only object: pose hypothesis without complete hidden geometry;
- underconstrained articulation: part evidence exists but articulation parameter unresolved;
- contact episode without posterior latent contact support: possible/nonactive evidence;
- active temporal contact state through weak/missing same-frame pose-anchor observations: allowed only when the posterior `C` state is anchor-bounded, associated, metric-hand-supported, and conflict-free; uncertainty stays on the contact state and does not become object-pose correction;
- occluded hand without accepted owner and metric support: unresolved/occluded, not pose-filled.

## Final-State Semantics

Final annotation state is graph-derived.

A user-facing physical claim is valid only if it names the solved variable/factor evidence that produced it. Provenance/debug channels may expose raw proposals, schema, masks, episodes, and measurements, but those fields must not be alternate sources of truth.

## V18 Spec Consistency Checklist

- Metric MANO hand state: physical hand variable and contact surface source.
- Object geometry/pose: rigid objects require object-frame geometry plus scoped per-frame `SE(3)`.
- Part/articulation: part-required objects use part geometry/pose/articulation variables; missing parts remain uncertainty.
- Contact: latent contact state/patch explains evidence; labels, raw distance, overlap, episodes, and pose-anchor emission are not contact.
- Nonpenetration: physical constraint/veto, not pose fallback.
- Occlusion: explicit visibility/owner/depth-order/uncertainty variables.
- Runtime: bounded alternating solve is allowed only if unstable states remain uncertain.
- Consumed artifacts: videos communicate solved graph state, not all internal evidence.

## Current-Code Mismatches To Resolve

- Current `contact_switch` is being promoted toward latent `C_t`, but it remains a bounded discrete posterior rather than a full continuous contact-manifold optimizer.
- Current rigid contact handling now has parent-side support for hand-footprint-excluded object-owned local depth, final-anchor temporal bracketing, and represented hidden-patch explanation. It still requires fresh clean-room review and remains an approximation rather than a solved continuous contact manifold.
- Current positive hidden-patch evidence is limited to task5 frames 824, 825, 826, and 834. Frame 823 is direct compatible visible-surface contact under the stricter hand-excluded depth measurement. Hidden-patch rows are scoped as contact feasibility/contact state only; they must not become object-pose correction without a separate stable pose-anchor factor and residual bound.
- Local rigid visible-surface contact is represented as `local_rigid_visible_contact_patch` and accepted scoped after the 2 cm cap repair. The remaining close/direct/compatible inactive contacts are articulated/part-required trash lid/can rows where same-frame represented part state is absent. The next mismatch is not a rigid local-contact relaxation; it is missing `articulated_part_contact_patch` / `P^part_{t,h,o,p}` state or missing part-track evidence.
- Current visual priors are hand-engineered; VLM contact prior should become an observation factor on latent `C`, not a final label.
- Render/export must expose posterior contact state and uncertainty, not proposal fields or anchor availability.

## Accepted Negative Evidence

- Distance-only contact activation is invalid: it creates proximity false positives.
- Contact-only visible surface must not move full rigid `object_se3` by itself.
- Post-coupled distance alone cannot justify the contact that created the coupling.
- Per-frame emitted pose-anchor is invalid as a necessary condition for contact existence.
- Owner-only temporal mesh-distance support is insufficient for deformable local patch activation.
- Part pose availability is not part contact.
