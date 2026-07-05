# Research track task pack draft

## Objective

Continue HOI/physical-state extraction as a separate research program. The research track may build object geometry, object pose, contact, occlusion ownership, nonpenetration, factor graphs, RL approximations, and distillation experiments, but none of these semantics enter the delivery API base contract until promoted through explicit evidence gates.

The research artifact is still renderable annotation: full-duration HOI overlays/world views whose visible object/contact/occlusion marks are driven by represented mechanisms and uncertainty. Schema rows are backing data, not the artifact.

## Track boundary

- Base delivery outputs remain `ego.delivery.output` v1 and contain head/camera, hands, semantic clips, QC, and renders only.
- Research outputs use an isolated extension namespace, currently reverse-DNS `org.ego.research.hoi` with schema family `ego.research.hoi` 0.x, as a sidecar or attached extension that references a pinned delivery manifest by id and hash.
- The extension may read delivery hand/camera/semantic rows; it cannot mutate delivery rows, delivery captions, delivery status, or delivery metrics.
- Promotion back to delivery requires measured utility, runtime budget, stable schema, and no contamination of the base accuracy claims.

## Core causal diagnosis

The v19 HOI path failed because the graph asked unobserved variables to explain the video:

1. Object pose was under-measured: silhouette/depth channels did not observe several failing directions, and pose could move through null spaces without changing residuals.
2. Object shape was weak: single-anchor completion baked appearance defects and hidden-region priors into a mesh treated as if metric.
3. Contact was under-observed: contact evidence often disappeared exactly under occlusion, and previous rows encoded empty support as if a factor existed.
4. Hand state needed drift latents: hand errors were smooth per-clip/per-side biases not represented in the solver state.
5. Factor graph liveness was not guaranteed: stale joins, zero-support terms, and passthrough solved states could look like implemented optimization.

Therefore the research program starts with measurement channels and state representation, then uses graphs, RL, and distillation after variables are observable.

## Research-state extension sketch

Namespace: `org.ego.research.hoi`; schema family: `ego.research.hoi` 0.1.0.

Default file layout:

```text
extensions/org.ego.research.hoi/0.1.0/{research_run_id}/
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

Required invariants:

- Every dependent row carries `base_job_id`, `base_manifest_sha256`, `frame_index` or half-open interval, and solution/epoch ids.
- Geometry freshness is represented by mandatory `geometry_epoch_id`; stale joins are queryable defects.
- Pose rows include per-DOF covariance/observability and declared gauge. Unobserved DOFs are uncertain, not silently smoothed.
- Solved tables carry input hashes and `differs_from_input`/liveness bits.
- Contact is a posterior over `{contact, near, none, unresolved}`, not a boolean.
- Signed distance is valid only against watertight geometry epochs with per-face provenance.
- Per-face labels distinguish observed, interpolated, prior-completed, and unknown regions.
- Deformable objects use visible-surface/surfel state; rigid pose is `not_applicable_deformable`.

## Ontology

Entities:

- Hands: delivery hand tracks (`left_primary`, `right_primary`) plus optional research correction rows; MANO vertex/region ownership references coarse anatomy (`thumb_tip`, `index_tip`, `fingers`, `palm`, etc.).
- Objects: open-vocabulary semantic hypotheses, motion-discovered rigid bodies, support surfaces, articulated bodies, deformables, unresolved bodies.
- Geometry epochs: time-scoped representations of object/surface geometry with method, source masks, anchor frames, scale basis, face provenance, and supersession links.
- Poses: per-frame object/camera/world transforms with gauge and observability.
- Rigidity windows: measured rigidity/deformation state over intervals via pairwise distance conservation and Procrustes residuals.
- Contacts: hand-object/part intervals with source-gap posterior, depth-order evidence, motion-onset evidence, learned-contact evidence if available, and ownership uncertainty.
- Occlusion: per-frame visibility state for hands/objects with occluder candidates and confidence.
- Events: grasp onset/release, object motion, deformation, part articulation, occlusion transitions.

## Metric families

Object pose and trajectory:

- GT: ADD/ADD-S, translation error mm, rotation error deg, trajectory ATE/RPE under declared gauge. Contact-usable target: ≤20–30 mm median translation and ≤6 deg rotation; strong: ≤10 mm and ≤3 deg.
- GT-free: held-out surface-track reprojection, silhouette IoU, visible-depth residual, per-DOF observability, temporal jitter normalized by image motion.
- Blind spots: symmetric objects hide rotation; silhouette misses depth-axis motion; global gauge can make a wrong trajectory look locally consistent.
- Routing: pose residual bad with shape/mask good → tracking/pose; pose residual good with silhouette/depth bad → shape/mask/camera substrate.

Object shape/geometry:

- GT: visible-region Chamfer to CAD, face-provenance-specific distance, free-space violation volume.
- GT-free: silhouette reprojection, depth support residual, free-space carving violations, mask contamination rate, watertightness/self-intersection.
- Threshold basis: visible shape should be comparable to or better than depth noise (roughly 15–45 mm depending channel); prior-completed faces are never used as hard contact evidence.
- Routing: high prior-face fraction means hidden-region uncertainty, not pose failure.

Contact and near-contact:

- GT/proxy-GT: interval AUROC/balanced accuracy against annotated/proximity-derived contact; onset/offset timing; signed-distance residual to GT mesh where available.
- GT-free: channel agreement between source gap, depth order, hand/object motion coupling, learned contact map, and occlusion state; contact flicker rate; object-motion-without-grasp contradiction.
- State semantics: `contact` when source gap is within soft-tissue band and channels support touch; `near` when within about 2σ combined uncertainty (~60 mm) without touch support; `none` when gap/channels reject; `unresolved` when occlusion or geometry makes the claim unmeasurable.
- Blind spots: contact through full occlusion remains uncertain; pressure/force is unobserved unless pressure data exists.

Nonpenetration:

- Valid only on watertight/observed-enough geometry. Use soft-tissue bands: finger pads about 2–3 mm, palm about 5 mm. Penetration beyond band into observed faces is a defect; penetration into prior-completed hidden faces is uncertainty.

Rigidity/articulation/deformation:

- GT-free: pairwise distance conservation, Procrustes residual over windows, depth noise floor, rigidity SNR, motion-class posterior.
- GT: known rigid objects and articulated datasets (HOT3D/ARCTIC/DexYCB/HOI4D where adopted after noise decomposition).
- Articulation: fitted revolute/prismatic/free model residual; scissors and similar objects validate part-motion semantics.
- Deformation: visible surface flow/residual and human event labels for fold/cut/bend intervals.

Occlusion and ownership:

- GT: rendered visibility from CAD/pose where available.
- GT-free: depth-order render-and-compare, mask boundary consistency, occluder overlap/frontness, contradiction with detector evidence.
- Claim boundary: ownership through full occlusion is posterior/uncertain unless depth-order and temporal evidence constrain it.

Hand correction inside HOI:

- Research may maintain `hoi_hand_corrections` as drift latents/corrections relative to base delivery hands.
- These rows cannot overwrite base hand states. Promotion to delivery requires GT-free anchors validated against HOT3D/held-out GT and final-layer visual drift.

Graph health:

- Term support fraction, zero-support active factor count, term gradient/share, solver liveness (`differs_from_input`), input-output hash difference, stale join count, gauge declaration count.
- Zero-support factors, stale joins, and unlabeled gauges are blocking defects for graph-derived claims.

Runtime:

- Research may be offline, but records GPU-hours/video-hour and wall-clock. Any method requiring hours per one-minute clip is research-only by default and cannot be promoted to delivery without a runtime redesign.

## Validation protocols

1. Fixed HOT3D slices with MANO and object CAD/pose where available; lockbox clips selected before tuning.
2. Demo-regime clips for GT-free metrics and human-labeled event boundaries: tomato, trash, origami, scissors, putty knife, phone/calculator.
3. External datasets only after noise/regime-transfer decomposition: EgoPressure for pressure/contact, ARCTIC for articulation, HOI4D for scale/part statistics, DexYCB for tabletop grasping.
4. Every experiment records predictions before running: which metric should move, which metric should not move, and what each outcome implies.
5. Dual reporting: proxy metric plus GT metric where GT exists. Proxy improvement without GT improvement is proxy capture.
6. Visual consumption is required: inspect full-duration renders or representative sheets as HOI annotations, not as file existence.

## Research automation invariant

Research auto-improvement cannot start as an unattended loop until the evaluator exists. The protected evaluator is not a single scalar; it is a vector over object pose, shape, rigidity, contact, occlusion, hand correction, graph health, and runtime. Acceptance requires target-family improvement, GT/proxy co-motion where GT exists, no protected-family regression, lockbox transfer, visual-render veto, and graph liveness/gauge/freshness asserts.

The honest experiment sequence is dependency-constrained:

1. Instrument graph health first: term support, liveness, gauge, stale joins, freshness.
2. Measure rigidity and correspondence before pose optimization.
3. Use pose-stabilized multi-frame geometry before nonpenetration/contact claims.
4. Validate contact channels before switchable contact factors.
5. Develop drift latents in parallel, but never fit them to GT for deployable claims.
6. Treat every negative result as a redirect within the same mechanism family unless it falsifies that family.

A fixed-slice win without lockbox transfer is overfit. A proxy win without GT/render agreement is proxy capture. A solved state that equals its input is inert. A nonwatertight or prior-completed face cannot support signed nonpenetration.

## RL approximation boundary

RL is useful only after the graph's variables and measurements are live. It may amortize a validated graph MAP solve, schedule discrete hypotheses/switches, choose initializers, or project a measurement-derived state into a feasibility set. It cannot recover information absent from the sensors and cannot replace object pose, contact, or drift measurements.

Admissible RL outputs must preserve posterior uncertainty, provenance, gauge declaration, measurement coupling, and liveness. Rewards are the validated graph objective or a measurement-coupled simulator residual; silhouette/source-gap/proximity/render style rewards alone are invalid because they reproduce known proxy-capture failures. Online simulation is research-only and only a feasibility regularizer; mass/friction/soft-tissue parameters are unmeasured variables, not truth.

RL stop conditions: R4 graph redesign not closed; channel ablation shows ignored measurements; reward improves while GT/proxy/render contradict; posterior calibration fails; simulator sensitivity dominates; runtime misses the delivery budget; disagreement with graph MAP persists beyond a declared band.

## Research milestones

R0 — Schema/output harness: `ego.research.hoi` sidecar manifest, tables, and full-duration HOI render driven by rows.

R1 — Correspondence-first rigid-body extraction: temporal 2D/3D tracks, rigidity windows, robust Procrustes, pose observability, and held-out track residuals.

R2 — Multi-frame geometry epochs: pose-aligned visible surface fusion, face provenance, free-space checks, prior completion only for never-seen regions.

R3 — Contact measurement channels: source-gap posterior, depth-order evidence, object-motion coupling, contact-map integration, and unresolved state rendering.

R4 — Factor graph redesign: drift latents, switchable contact mixtures, calibrated noise, liveness audit, gauge declarations, stale-join prevention.

R5 — HOI evaluation suite: GT + GT-free metrics, routing rules, lockbox protocols, visual-veto process, and protected evaluator bundle. No autonomous research loop runs before R5 exists.

R6 — RL approximation study: approximate the factor-graph inference policy only after R1-R4 produce live variables and a validated graph objective; evaluate graph-MAP parity, GT/GT-free metrics, posterior calibration, measurement-coupling ablations, and runtime.

R7 — End-to-end distillation study: train a model to imitate validated research outputs and uncertainty, with delivery promotion gates for runtime and accuracy. Distillation never becomes a source of HOI truth; it is a fast student of validated teachers.

## Remaining synthesis slot

The remaining unresolved planning question is distillation/promotion: which validated research outputs become trainable teacher labels, what uncertainty must be distilled, which failure modes require abstention, and what evidence permits a distilled model to affect delivery.

## Uncertainty boundaries

Must remain uncertain unless new evidence exists:

- Contact ownership through full occlusion.
- Signed distance on prior-completed/hallucinated mesh faces.
- Pose DOFs below observability/noise threshold.
- Hidden-region object shape.
- Deformable object full state from monocular video.
- Contact versus hover when the source gap is within combined uncertainty but channels disagree.
- Grasp force/pressure without pressure or force evidence.
- Object identity completeness from open-vocabulary plans alone.
- World gravity/camera trajectory on drift-bent worlds without calibrated camera evidence.
- Occluded hand pose beyond the temporal/measurement horizon.

## Promotion gate to delivery

A research mechanism can enter the delivery track only if:

1. It improves a delivery-protected metric or caption utility on a frozen eval set.
2. It does not regress hand/camera/drift/caption/throughput protected metrics.
3. It fits the delivery runtime budget or has a redesigned fast approximation.
4. Its uncertainty can be represented in `ego.delivery.output` without changing base semantics.
5. The rendered delivery artifact visibly improves or the mechanism remains research-only.
