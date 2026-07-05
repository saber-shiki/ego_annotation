# Research track task pack draft

## Objective

Build a HOI/physical-state extraction system that turns egocentric video into full-duration, renderable annotations for object geometry, object pose, contact, occlusion ownership, nonpenetration, articulation/deformation, and hand-object interaction events.

Progress is mechanism resolution. Each change must connect an observed artifact defect to a physical variable, test the mechanism that could explain it, implement the intervention selected by the result, and change the rendered/numeric HOI state.

The research artifact is full-duration HOI annotation: tables, assets, and overlay/world/side-by-side videos whose visible marks are driven by measured geometry, pose, contact, occlusion, and hand-correction state. Schema rows exist to drive and reproduce those annotations.

## Analysis contract for every experiment

Every research experiment starts with a causal card:

1. **Artifact defect:** the visible or numeric HOI failure being repaired.
2. **Physical variable:** object pose, geometry, contact state, occlusion owner, hand drift, graph liveness, or runtime variable that is wrong.
3. **Live mechanisms:** two or more concrete mechanisms that could produce the defect.
4. **Discriminating measurement:** the observation that separates those mechanisms.
5. **Predictions:** expected measurement pattern under each mechanism.
6. **Intervention:** the implementation change selected by each possible outcome.
7. **State change:** the table fields, assets, and rendered marks expected to change.

An experiment that only produces a score without selecting the next implementation is bookkeeping. An implementation that lacks a discriminating measurement is blind activity.

## Causal map of current HOI defects

### D1 — Object pose moves through unobserved directions

Observed defect: object pose can change while silhouette/depth residuals stay similar, so the rendered object can look plausible in one view and be physically wrong in 3D.

Live mechanisms:
- M1: temporal correspondences are missing, so pose lacks point-level constraints.
- M2: object symmetry or weak texture leaves some rotations low-observability.
- M3: camera/depth gauge error is being absorbed by object pose.
- M4: optimizer weights let one residual channel dominate the others.

Discriminating measurements:
- Held-out temporal track reprojection residual.
- Visible-depth residual along camera ray.
- Silhouette residual separated by image-plane direction.
- Per-DOF Hessian/covariance and residual contribution by factor family.

Predictions and interventions:
- High track residual with good silhouette → build/repair temporal correspondences and robust Procrustes seeds.
- High depth residual with good 2D residual → add metric depth/camera constraints and depth-axis pose factors.
- Low rotational observability on symmetric object → add part/texture correspondences or represent that DOF with high covariance.
- One channel dominates gradient share → recalibrate factor noise and normalize residual scales.

### D2 — Geometry encodes a single-frame prior as object body

Observed defect: a mesh built from one anchor frame or prior completion can be treated as a stable physical object even where surfaces were never observed.

Live mechanisms:
- M1: source masks include background or hand pixels.
- M2: depth support is sparse or wrong at the anchor.
- M3: prior completion invents hidden surfaces that later contact logic consumes.
- M4: pose error during fusion smears observed surfaces.

Discriminating measurements:
- Face-provenance fractions by region: observed, interpolated, prior-completed, unknown.
- Free-space violation volume.
- Held-out silhouette/depth residual.
- Multi-frame fusion residual after pose alignment.

Predictions and interventions:
- High contamination rate → repair segmentation/source prompts before geometry fusion.
- High free-space violation → prune mesh and refit scale/pose.
- High prior-completed fraction in active contact region → collect additional views or route contact to visible-surface evidence.
- Fusion residual grows with additional frames → repair pose alignment before adding geometry.

### D3 — Contact evidence disappears when contact matters

Observed defect: hand-object contact often occurs under partial occlusion; earlier factors encoded empty support as if contact had been measured.

Live mechanisms:
- M1: MANO/object source gap is wrong because hand or object pose is wrong.
- M2: contact is visible only through motion coupling, not direct surface visibility.
- M3: depth order is available but not linked to contact state.
- M4: temporal contact state flickers because every frame is solved independently.

Discriminating measurements:
- Source-gap posterior at MANO region/object surface.
- Depth-order render-and-compare residual.
- Object motion onset relative to hand velocity/acceleration.
- Contact-map response where learned channel exists.
- Contact flicker rate and interval continuity.

Predictions and interventions:
- Source gap and motion agree but depth is missing → add visibility-conditioned contact posterior and temporal interval smoothing.
- Motion without source-gap support → inspect hand/object pose and association before adding contact factors.
- Depth order contradicts source gap → repair geometry, K, depth, or hand/object pose.
- High flicker with stable evidence → add temporal contact mode variable and hysteresis.

### D4 — Hand state has smooth metric drift inside HOI

Observed defect: delivery hands can be locally plausible in 2D while metric wrist/root, joints, or surface are biased enough to corrupt contact and nonpenetration.

Live mechanisms:
- M1: per-clip/per-side translation and rotation bias from calibration/crop/K mismatch.
- M2: hand source switching creates discontinuities.
- M3: occlusion intervals are over-smoothed and exit at wrong position.
- M4: root is corrected while all-joint MPJPE or surface remains wrong.

Discriminating measurements:
- Wrist/root error and all-joint MPJPE where GT exists.
- Final-layer reprojection residual against independent 2D detector evidence.
- Projected MANO size versus detected hand size.
- Source-switch jitter and occlusion-exit residual.
- Source gap residual under high-confidence contact candidates.

Predictions and interventions:
- Reprojection improves while MPJPE worsens → repair metric root/depth calibration.
- MPJPE improves while render drifts → repair projection/crop/K adapter.
- Jitter spikes at source switches → add source hysteresis and detector-bounded fusion.
- Surface/contact residual remains after root repair → optimize all-joint/surface state, not only wrist/root.

### D5 — Factor graph can be inert while looking implemented

Observed defect: graph outputs can equal inputs, use stale dependencies, or include factors with no support while tables imply optimization occurred.

Live mechanisms:
- M1: factor support is zero or near zero on critical intervals.
- M2: graph variables do not control the rendered state.
- M3: stale object/hand/geometry ids are joined into the solve.
- M4: residual scales make one factor family invisible.

Discriminating measurements:
- Active residual count and support fraction by factor family.
- Input-output deltas for every state family.
- Render-state hash lineage from graph output to final video.
- Stale dependency count and geometry epoch freshness.
- Gradient/share by factor family.

Predictions and interventions:
- Zero support → repair measurement extraction before changing optimizer weights.
- Nonzero residual with zero state delta → repair variable wiring or solver plumbing.
- Correct graph output but stale render → repair state-to-render dependency chain.
- Dominant factor suppresses others → recalibrate noise model and residual normalization.

### D6 — Cross-solver geometry-source decoupling and contact-source admissibility

Observed defect: in HOT3D clip001850 keyboard frames 28-48, the interval MANO mesh-penetration channel reports large right-hand penetration while keyboard-masked, hand-quarantined depth places the hand in front of the keyboard surface on every mask-available frame. Contact/nonpenetration reports zero penetrating vertices against a non-watertight, 95.4%-TRELLIS completed mesh whose free space was never carved. The published render collapses this into a vague gap/penverts/UNCERTAIN banner.

Live mechanisms:
- M1: solver/contact/render stages consume different measured provenance hashes, so generic support-absent labels hide a source-lineage defect.
- M2: the completed keyboard geometry epoch is contact-ineligible: inflated, mostly inferred, non-watertight, and uncarved.
- M3: the interval mesh-penetration channel is not an admissible keyboard-contact factor unless keyboard-masked, hand-quarantined depth and per-vertex coherence preserve it.
- M4: clip001850 hand metric depth bias is large enough that distance-only evidence can reach `contact_candidate` at most, not `confirmed_contact`.
- M5: the graph/render path can carry rows without changing what the viewer sees unless `contact_frame_detail` is consumed by a render artifact.

Discriminating measurements:
- Provenance-hash-derived solver/contact/render geometry epoch ids and source families, plus explicit incomplete-provenance routing.
- Keyboard-masked and hand-quarantined depth deltas per frame, with penetrating/near vertex ids rather than max scalar summaries.
- Per-vertex temporal IoU and anatomical localization of surviving contact candidates.
- Completed-mesh watertightness, face provenance, free-space state, and observed-face sign-mesh eligibility.
- Pose provenance for posed-geometry fallback frames.
- Viewer-visible f32/f36 ego.hoi review whose labels come from `contact_frame_detail` route states.

Measured result and interventions:
- f32 routes to `geometry_epoch_contaminated`: interval penetration max 82 mm, keyboard-HQ median delta -144 mm, zero eligible penetrating vertices.
- f36 routes to `full_frame_depth_leak`: interval penetration max 107 mm, keyboard-HQ median delta -81 mm, zero eligible penetrating vertices; full-frame leak is only ~22 mm on non-keyboard pixels.
- f45 demotes to `unresolved_incoherent_evidence`: one thumb vertex at +5.86 mm on a near-depth outlier, no temporally persistent contact patch.
- Observed-body repair removes the false TRELLIS penetration but leaves observed-only hand distances at 35.6-115.2 mm; it is a render/body-provenance repair, not a contact recovery route.
- Hand-depth contact-forcing is refuted: required along-ray shifts are 83-202 mm, produce 62-169 px median reprojection displacement, and create a candidate on only f36. Do not use keyboard gap as a hand-depth correction target.
- Keep `cross_solver_geometry_decoupled` as a provenance-hash graph-health route, but not as permission to use the contaminated contact source.
- Production-style render consumption now exists as a full-duration replacement artifact under `/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/v19_contact_state_full_duration/`: the durable canonical `contact_frame_detail` drives f32/f36/f45 labels, uncovered frames default to `unresolved_evidence_incomplete`, and the 3221-face observed body is rendered as an open/hatched patch instead of the raw TRELLIS body. The still-only consumer is QC evidence, not the consumer. Next artifact-changing work is promoting this consumer into the reusable runtime path or adding motion/GT adjudication for confirmed contact.

### D7 — Runtime-heavy mechanisms need teacher/student separation

Observed defect: some mechanisms may improve HOI state but run too slowly for delivery-scale use.

Live mechanisms:
- M1: heavy stage is needed only to create teacher labels.
- M2: heavy stage can be replaced by an amortized initializer or scheduler.
- M3: heavy stage consumes video-local context that batching fails to preserve.

Discriminating measurements:
- GPU-hours/video-hour by stage.
- Batch fill, model residency, and queue wait.
- Quality loss when stage is approximated.
- Student/teacher parity by variable family.

Predictions and interventions:
- Teacher improves state but is slow → retain as offline teacher and start distillation.
- Approximation preserves parity → package as fast research path.
- Approximation loses one variable family → split model or add missing measurement input.

## Output substrate: `ego.hoi`

Namespace: `org.ego.hoi`; schema family: `ego.hoi` 0.1.0.

Default file layout:

```text
extensions/org.ego.hoi/0.1.0/{run_id}/
  manifest.json
  tables/
    objects.parquet
    object_parts.parquet
    geometry_epochs.parquet
    object_pose.parquet
    object_visible_surface.parquet
    rigidity_windows.parquet
    contact_hypotheses.parquet
    contact_frame_detail.parquet
    visibility_states.parquet
    hoi_hand_corrections.parquet
    graph_solutions.parquet
    hoi_validation_metrics.parquet
  streams/
    hoi_events.ndjson
    hoi_overlay_events.ndjson
    provenance.ndjson
    errors.ndjson
  assets/
    meshes/ points/ masks/ contact_maps/
  renders/
    hoi_overlay.mp4
    hoi_side_by_side.mp4
    hoi_world.mp4
```

Rows carry `base_job_id`, `base_manifest_sha256`, frame/interval, object id, geometry epoch id, solution id, input hashes, output hashes, coordinate frame, gauge, covariance/observability, and provenance. Graph-health rows additionally carry solver/contact/render geometry epoch ids, geometry source families, signed query candidate count, watertight flag, face provenance summary, observed-surface penetration, published contact gap, and `cross_solver_geometry_decoupled` decision state. The renderer consumes these rows to draw hands, object bodies, visible surfaces, contact state, occlusion ownership, event labels, and uncertainty styling.

## Workstream A — object hypotheses and temporal correspondences

Mechanism being tested: object pose is underconstrained because the object lacks stable temporal point/surface correspondences.

Build:
- Open-vocabulary object hypotheses from semantic plans, captions, boxes, and masks.
- Motion-discovered rigid-body candidates from 2D tracks, 3D/depth tracks, and temporal mask correspondences.
- Track-quality tables with lifetime, occlusion gaps, reprojection residual, depth residual, and association confidence.

Discriminating experiment:
- Hold out frames from each candidate interval.
- Fit correspondence tracks on the remaining frames.
- Predict held-out 2D positions and depth support.

Outcome routing:
- Low held-out residual → feed tracks to rigidity and pose.
- High residual with identity swaps → improve association or split object hypotheses.
- High depth inconsistency → repair depth/camera substrate before pose fitting.

## Workstream B — rigidity, articulation, and deformation

Mechanism being tested: the object state family is wrong if rigid pose is forced onto articulated or deformable motion.

Build:
- Rigidity windows from pairwise distance conservation and robust Procrustes residuals.
- Motion-class posterior over rigid, articulated, deformable, support surface, unresolved body.
- Articulation candidates with revolute/prismatic/free-joint residuals.
- Deformation intervals from visible surface flow and event labels.

Discriminating experiment:
- Compare rigid Procrustes residual against articulated and deformable alternatives over the same interval.
- Normalize residuals by depth noise and track uncertainty.

Outcome routing:
- Stable rigidity → feed pose graph and geometry fusion.
- Articulation residual lower than rigid residual → split parts and fit joint model.
- Deformation residual dominates → use visible-surface/surfel state for that interval.

## Workstream C — geometry epochs

Mechanism being tested: contact and pose fail because the geometry epoch is contaminated, stale, or dominated by unobserved completion.

Build:
- Pose-aligned multi-frame visible-surface fusion.
- Geometry epochs with source masks, depth sources, anchor frames, scale basis, supersession links, and per-face provenance.
- Free-space carving and silhouette/depth validation against held-out frames.
- Prior completion only for never-seen regions, with separate provenance labels.

Discriminating experiment:
- Render the fused geometry into held-out frames.
- Compare silhouette, visible-depth residual, and free-space violations by face provenance class.

Outcome routing:
- High observed-face residual → repair masks, depth, pose, or fusion.
- High free-space violation → prune geometry and refit scale/pose.
- High completion fraction in active interaction region → acquire more visible surface evidence or route contact to observed faces.

## Workstream D — object pose and trajectory

Mechanism being tested: object trajectory error comes from weak pose seeds, wrong residual weighting, or camera/depth gauge leakage.

Build:
- Pose seeds from robust Procrustes over temporal correspondences.
- Pose refinement using silhouette, visible depth, surface-track reprojection, rigidity, and smooth motion priors.
- Per-DOF covariance/observability and declared gauge per trajectory.
- Pose routes for rigid objects, articulated parts, deformables, and support surfaces.

Discriminating experiment:
- Run ablations with correspondence-only, depth-only, silhouette-only, and fused residuals.
- Compare GT ADD/ADD-S where available and GT-free held-out residuals everywhere.

Outcome routing:
- Depth-axis residual dominates → add metric depth/camera constraints and depth-axis factors.
- Rotation weak on symmetric shape → add texture/part correspondence and represent covariance.
- Good masks with bad pose → repair optimizer and correspondence weighting.
- Good pose with bad masks/depth → repair perception substrate.

## Workstream E — HOI hand corrections

Mechanism being tested: HOI contact is wrong because delivery hand state contains smooth drift or source-switch artifacts.

Build:
- `hoi_hand_corrections` rows as drift latents relative to delivery hands.
- Per-side smooth translation/rotation correction families tied to visible 2D, MANO geometry, depth/size, and object-contact channels.
- Source hysteresis and occlusion-aware uncertainty for hand state used inside HOI.

Discriminating experiment:
- Fit correction family on development clips using GT-free residuals.
- Evaluate wrist/root, all-joint MPJPE, reprojection, projected size, source gap, and jitter on held-out clips.

Outcome routing:
- Reprojection improves while MPJPE worsens → repair metric root/depth calibration.
- MPJPE improves while rendered hand drifts → repair projection/crop/K adapter.
- Occlusion-exit drift remains high → strengthen detector-bounded fusion and temporal reset.
- Root improves but all-joint MPJPE stays high → optimize articulation/surface, not only global hand transform.

## Workstream F — contact, near-contact, and occlusion ownership

Mechanism being tested: contact state is recoverable by combining weak channels that fail at different times.

Build:
- Contact posterior over `contact`, `near`, `none`, and `unresolved` per hand/object/part interval.
- Source-gap channel from MANO region/object surface distance with soft-tissue bands.
- Depth-order channel from render-and-compare and local depth evidence.
- Motion-coupling channel from object motion onset, hand velocity, and relative acceleration.
- Learned contact-map channel where training/evaluation data exists.
- Visibility states for hand/object with occluder candidates and confidence.

Discriminating experiment:
- For each candidate contact interval, score source gap, depth order, motion coupling, learned contact, and visibility independently.
- Compare channel combinations against annotated/proximity-derived contact and render review.

Outcome routing:
- Source gap and motion agree while depth is missing → add visibility-conditioned posterior and interval smoothing.
- Motion without source-gap support → inspect hand/object pose and association.
- Depth order contradicts source gap → repair geometry, K, depth, or hand/object pose.
- High flicker with stable evidence → add temporal contact mode variable and hysteresis.

## Workstream G — graph redesign

Mechanism being tested: the graph succeeds only when every factor family has live measurement support and controls rendered variables.

Build:
- Variables: object pose, geometry epoch, rigidity state, contact mode, occlusion owner, hand drift latent, gauge, and per-channel noise.
- Factors: correspondence reprojection, visible-depth residual, silhouette residual, rigidity, contact source-gap, depth-order, motion coupling, nonpenetration, hand correction, and temporal smoothness.
- Switchable contact mixtures and visibility-conditioned factor activation.
- Graph health table with support fraction, active residual count, gradient/share, liveness, input-output deltas, stale dependency count, and gauge declarations.

Discriminating experiment:
- Add one factor family at a time on the same frozen clips.
- Measure target metric movement, cross-family regressions, factor support, gradient share, liveness, and rendered state deltas.

Outcome routing:
- Factor support near zero → repair measurement extraction.
- Nonzero residual with zero state delta → repair variable wiring or solver plumbing.
- Correct graph output but stale render → repair state-to-render dependency chain.
- One family improves while another degrades → inspect shared variables and noise scaling.

## Workstream H — evaluator and datasets

Mechanism being tested: a metric is useful only if it selects the same implementation direction as GT or visual HOI consumption on held-out data.

Build:
- Frozen development set from HOT3D clips with MANO/object CAD/pose where available.
- Demo-regime GT-free set: tomato, trash, origami, scissors, putty knife, phone/calculator.
- In-house HOI lockbox with synchronized camera/head/hand/object/contact annotations for promotion studies.
- External dataset adapters after regime/noise decomposition: EgoPressure, ARCTIC, HOI4D, DexYCB.
- Full-duration visual review renderer plus representative sheets for every metric run.

Discriminating experiment:
- Compare proxy metric movement against GT metric movement where GT exists.
- Compare both against visual annotation consumption: object identity, pose, contact interval, occlusion state, and event timing.

Outcome routing:
- GT metric and proxy metric move together → keep metric and expand clips.
- Proxy moves alone → redesign proxy channel or weighting.
- Render contradicts table state → repair state construction or renderer wiring.
- Runtime dominates → isolate teacher path and start approximation study.

## Workstream I — RL approximation

Mechanism being tested: validated graph inference can be accelerated by learned initialization, scheduling, or projection without losing measurement coupling.

Build after live graph variables and evaluator exist:
- Amortized MAP initializer for graph variables.
- Discrete hypothesis scheduler for object/part/contact modes.
- Feasibility projection over measurement-derived states.
- Policy/value model trained against validated graph objective and evaluator outputs.

Discriminating experiment:
- Compare learned approximation to graph MAP on held-out clips.
- Ablate measurement channels to verify dependence on pose/geometry/contact evidence.
- Measure runtime reduction and posterior calibration.

Outcome routing:
- High parity and lower runtime → use as initializer or fast path.
- Low parity localized to one variable family → improve teacher labels or policy inputs for that family.
- Runtime still high → distill smaller student or restrict RL to scheduling.

## Workstream J — distillation and delivery handoff

Mechanism being tested: validated HOI teachers can produce a fast student that preserves useful physical state and calibrated uncertainty.

Build:
- Teacher label export from validated graph/evaluator runs: pose, geometry provenance, contact posterior, occlusion state, hand correction, and uncertainty.
- Student model trained to predict teacher state and calibrated uncertainty from delivery-available inputs.
- Abstention head for low-observability intervals.
- Runtime implementation as a Ray actor or later PyTriton service once tensor ABI is stable.

Discriminating experiment:
- Compare student to teacher, GT/proxy metrics, visual render consumption, and runtime.
- Ablate teacher state families to identify which ones create delivery utility.

Outcome routing:
- Student matches teacher and cuts runtime → package as research fast path.
- Student fails a family → return to teacher measurement or split the student by variable family.
- Delivery utility appears → define the smallest delivery extension, QC lane, or model service that consumes it.

## Milestones

R0 — Output substrate: write `ego.hoi` sidecar manifest, tables, provenance streams, and full-duration render from rows.

R1 — Causal cards and graph instrumentation: add support/liveness/gauge/freshness metrics and cross-solver geometry-source consistency to current v19-style graph runs; every experiment records mechanisms and predictions before runtime.

R2 — Correspondence and rigidity: produce temporal tracks, rigidity windows, Procrustes pose seeds, and residual reports.

R3 — Geometry epochs: build pose-aligned multi-frame visible-surface fusion with face provenance and free-space checks.

R4 — Pose optimizer: refine object/part pose with correspondence, silhouette, depth, rigidity, and observability outputs.

R5 — HOI hand correction: add drift latents and all-joint hand metrics inside HOI state.

R6 — Contact/occlusion channels: implement source gap, depth order, motion coupling, learned maps, visibility states, and temporal contact smoothing.

R7 — Live factor graph: combine R2-R6 variables with calibrated noise, switchable contact modes, and graph-health reporting.

R8 — HOI evaluator: run GT, GT-free, render, lockbox, and runtime metrics across development, demo-regime, and in-house promotion clips.

R9 — RL approximation: train initializer/scheduler/projection models against validated graph outputs and measure parity/runtime.

R10 — Distillation: train student models from validated teacher outputs, measure parity/calibration/runtime, and package fast research path.

R11 — Delivery integration route: when a research output improves delivery utility and runtime, convert it into the smallest delivery extension, QC lane, or model service that preserves base output semantics and measured accuracy.
