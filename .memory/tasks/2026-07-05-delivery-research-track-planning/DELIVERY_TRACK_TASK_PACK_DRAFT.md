# Delivery track task pack draft

## Objective

Build a separate delivery pipeline, independent of the v1-v19 HOI/factor-graph version line, that exposes a directly callable API for egocentric-video head/camera numeric state, metric hand numeric state, semantic clip captions, provenance, and validation metrics. Renders/overlays are QC and demonstration views of those numbers, not the primary delivery result. The delivery pipeline intentionally excludes object-pose/contact/nonpenetration/HOI optimization from its default path.

## Success criteria

1. Customer can submit videos through `/v1/delivery/jobs` and receive a manifest-driven artifact bundle (`ego.delivery.output` v1) with frame-aligned numeric tables, semantic clips, QC metrics, provenance, explicit errors, and optional QC/demo renders.
2. Head/camera and hand metrics are reported separately. The head/camera path to ~5mm is a measurement-source problem: ingest device VIO/SLAM/IMU/fiducial/GT-capable pose where available, calibrate video↔pose synchronization and camera/head extrinsics, and evaluate ATE/RPE under a fixed metric gauge. If input is RGB-only monocular video with no metric anchor, the input contract cannot support ~5mm translation.
3. Hand metric claim is wrist/root in camera frame unless the user explicitly chooses a stricter joint-level target. The deployable route is GT-free drift self-calibration anchored by cross-detector residuals, projected-size/depth consistency, crop/intrinsics correctness, and held-out HOT3D-style evaluation.
4. QC overlays must not contradict numeric states: if the rendered hand appears 20-100 px away from independent 2D evidence, diagnose whether the numeric hand state, projection/crop/K adapter, or renderer is wrong. Overlay correctness is a QC constraint, not the primary result.
5. Fine-grained semantic clips cover the full video timeline with mostly 2-3s segments; captions are grounded in visible entities/actions and carry confidence/evidence frames.
6. Throughput path demonstrates measured capacity toward ~10,000 video-hours/week, using module-speed instrumentation and a Ray-first video-aware scheduler. Dense object segmentation/SAM2 is not in the delivery default path when HOI is skipped.

## Non-goals

- No default HOI factor graph, object pose optimization, contact ownership, signed nonpenetration, object mesh reconstruction, TRELLIS completion, or per-instance neural reconstruction.
- No dense object segmentation/SAM2/object tracking lane unless a delivery metric or caption-grounding requirement specifically needs it.
- No reliance on RGB-only monocular reconstruction for ~5mm head/camera translation; add or ingest a metric pose source instead.

## Required module set

D1. Ingestion + raw frame manifest: frame paths, frame count, timestamps, resolution, input hash.

D2. Camera calibration contract: one canonical K (+ distortion/rectification, axis convention, source) per clip/session; all consumers cite the same contract; missing intrinsics is a hard error.

D3. Metric depth/intrinsics support: conditional lane for uncalibrated ingest diagnostics, semantic grounding, and camera-scale QC; not default hand-depth truth and not a substitute for a metric head/camera source.

D4. Head/camera trajectory: first-class numeric output with validity/confidence, gauge declaration, and separated metrics. Preferred source is device VIO/SLAM/IMU or fiducial/mocap-derived pose. RGB-only tracking may provide QC/relative motion but cannot by itself establish ~5mm metric translation in arbitrary egocentric video.

D5. HaWoR metric MANO: current metric wrist/root translation source, with focal-cache invalidation.

D6. WiLoR visible-hand geometry: root-relative geometry, 2D/crop evidence, presence/absence evidence, raw crop metadata preserved.

D7. Hybrid + temporal fusion hand layer: HaWoR metric translation + WiLoR visible geometry; detector-fidelity-bounded fusion; robust capped relative-motion prior; source hysteresis; occluded/fallback states ghosted.

D8. GT-free smooth-drift self-calibration: fixed-capacity per-clip correction family (R(t)+per-side b(t)) anchored by GT-free residuals (cross-detector, size/depth, static-scene). This is the deployable hand-accuracy bridge.

D9. QC/demo renderer: deterministic projection of the numeric state and captions; pure function of state/layer hashes; no silent K fallback; frame count equals input. It diagnoses state/projection contradictions but does not define the numeric result.

D10. Self-consistency QC: per-frame and per-clip metrics over camera, hands, overlay drift, semantic captions, and throughput.

D11. Offline evaluator harness: HOT3D hand evaluator + new camera/head evaluator + ray/lateral decomposition and reconciliation checks.

## Blunt correctness fixes before API beta

1. Establish one calibrated camera model per clip/device; use it everywhere; validate UniDepth-vs-SfM/capture calibration agreement.
2. Eliminate per-frame elastic depth scale; one rigid metric space per clip.
3. Constrain temporal hand fusion by detector residual; do not allow smoothness priors to drag hands 20-100 px off evidence.
4. Recompute QC confidence from the final fused layer, not pre-fusion residuals.
5. Make renderer deterministic and content-hash/resume safe so QC views cannot hide numeric/projection mismatches; remove silent intrinsics defaults.
6. Build separated benchmark tables: camera/head ATE/RPE/scale under fixed metric gauge and hand wrist/MPJPE/root-aligned metrics.

## API/output contract

Use `ego.delivery.output` v1:

- `manifest.json` as source of truth and artifact index.
- Parquet: `frames`, `head_camera`, `hand_states`, `semantic_clips`, `validation_metrics`.
- NDJSON: `overlay_events`, `caption_events`, `provenance`, `errors`.
- Renders: overlay, side-by-side/low-res optional, thumbnails.
- Coordinate frames: `image_px`, `camera_t`, `world_w0`, optional `head_t`, `mano_left/right`.
- Extensions: HOI research outputs must use an extension namespace and cannot alter delivery semantics.

## Serving architecture

Phase 0: benchmark current modules with cold/warm timings, module_speed_x, GPU utilization, queue wait, batch fill, and artifact correctness metrics.

Phase 1: FastAPI async job service + custom video-aware coalescer + Ray Serve/Ray GPU actors. Public API accepts video URIs and emits job ids; internal scheduler groups adjacent chunks by video/time/state affinity.

Phase 2: migrate stable stateless modules to PyTriton/Triton when tensor contracts settle; VLM/captioning to vLLM/TGI when supported; keep stateful video modules as Ray actors.

Phase 3: KubeRay/KServe only as outer fleet control if needed; managed GPU platforms only after cost/data-locality benchmarks.

Avoid TorchServe for new production.

## Validation/metrics

Camera/head:
- Static-scene reprojection residual, 3D closure residual, gravity/world drift plausibility, and GT ATE/RPE/scale when GT exists.

Hands:
- HOT3D wrist/root camera-frame error, visible-joint/root-relative MPJPE, final-layer reprojection residual, cross-detector keypoint residual, projected-size-vs-detected-size, jitter/source-switch metrics.

Overlay drift:
- Burst score on final rendered state; intrinsics/crop residual-reduction sweep; render provenance contradiction count.

Semantics:
- Segment coverage/duration compliance, caption entity/action grounding, caption consensus under overlapping windows, boundary stability, caption-to-annotation agreement.

Throughput/API:
- GPU-hours/video-hour by lane, worker residency/load amortization, job success and explicit failure rate, output completeness, lane regression, capacity/backpressure forecast.

## Delivery auto-research operating model

Auto-research is allowed only after the protected evaluator exists. The evaluator couples the numbers customers care about: head/camera metric error, hand metric error, caption grounding, throughput, and projection/QC contradictions. Each axis has paired guard metrics so a fix cannot improve one numeric table while damaging another.

Three surfaces:

- `axis_spec.md`: human-edited program for one axis (`head_camera`, `hand`, `drift`, `caption`, `throughput`) naming defect, protected metric, eval clips, accept rule, and anti-patterns.
- Editable stage code: only the delivery pipeline stage for that axis.
- Protected evaluation bundle: frozen eval clips, GT sidecars, metric harness, visual-review checklist, and any calibration used for scoring. The loop cannot edit these.

Acceptance rule:

1. Target-axis metric improves beyond measured rerun noise.
2. No protected metric regresses beyond noise. Examples: H3 reprojection improvement with worse H5 size/depth is rejected; throughput improvement with worse caption grounding is rejected.
3. Numeric-output consistency passes: improved head/camera or hand numbers do not create impossible projection/crop/size/caption contradictions in QC views.
4. Provenance contradiction count stays zero: no current detector or pose evidence hidden behind stale inferred states.

Run discipline:

- Each experiment starts with a causal card: numeric defect, wrong physical variable, mechanism hypothesis, coupling, predictions, accept/reject logic, and next action per outcome.
- One mechanism change per iteration; keep-if-better, else reset; preserve rejected rows as negative information.
- Cross-axis monotonicity is enforced at merge by recomputing the full vector on the frozen eval set.
- Human opens bounded family experiments when a real mechanism requires worse-before-better steps, such as replacing an intrinsics/crop adapter.
- All model/pipeline runtime executes on server/A800 or approved non-local compute; local machine remains orchestration only.

Initial eval set:

- GT-anchored: HOT3D 001849/001850/001851 for hand H1/H2; camera GT only where sidecars exist.
- Self-consistency: task5_tomato_960, trash_1050, window_putty_knife, phone_calculator, cut_cloth_scissors, origami_paper.
- `trash_1050` remains a known-fail probe; a cheap improvement without a plausible mechanism is suspected metric gaming.

Axis gates:

- Hand: H1 wrist/root camera-frame error primary; H2 MPJPE secondary; guard left/right asymmetry and H5 size/depth.
- Drift: R1 burst + H3 reprojection paired with H5 size ratio; reject 2D-only wins.
- Head/camera: HC1/HC2/HC3 for routing only; HC4 GT required for any 5 mm claim.
- Caption: S1 coverage/duration, S2 grounding, S4 boundary stability; periodic human audit for VLM shared hallucination.
- Throughput: T1 GPU-hours/video-hour by active lane; guard lane-specific quality. SAM2/object segmentation is absent from the default delivery lane unless explicitly enabled for caption grounding or a future requirement.

Anti-patterns that invalidate a run:

- Fitting calibration or corrections to eval GT and calling the number deployable.
- Improving one customer-facing number while another required number or QC/projection consistency gets worse.
- Shipping schema/ledger/validator changes as progress when head/camera, hand, caption, or throughput numbers did not improve.
- Global reweighting to hide localized hand-source failures.
- Hand-coded category/action if/else paths.
- Emitting 5 mm camera/head accuracy from self-consistency metrics alone.
- Letting HOI/factor-graph machinery enter the delivery default path.

## First implementation milestones

M0 — Decision lock: define hand metric target (wrist/root vs full joint), camera source policy, camera benchmark source, deployment assumption.

M1 — Measurement harness: module-speed benchmark + camera/head evaluator + final-layer QC recomputation on tomato.

M2 — Correctness patch tranche: calibration contract, no elastic scale, detector-bounded fusion, deterministic renderer.

M3 — API alpha: manifest + tables + overlay events for one clip; no HOI fields.

M4 — Caption alpha: full-timeline semantic clips with grounded captions and metrics.

M5 — Throughput alpha: Ray scheduler running at batch load on representative clips, module budgets reported.

M6 — Pilot release: 100+ video-hours processed through API, all metrics emitted, failures explicit, capacity forecast for 10k h/week.

## Parked decisions for user

1. Does customer input include device calibration/VIO/head pose metadata?
2. Is the hand accuracy target wrist/root-only or full MANO joint/surface accuracy?
3. Which camera/head benchmark or fiducial capture should define the 5mm camera claim?
4. Is the initial deployment a private Ray fleet or Kubernetes-first platform?
