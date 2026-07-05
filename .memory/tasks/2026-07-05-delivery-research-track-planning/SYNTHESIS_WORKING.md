# Working synthesis from first-wave outputs

## Delivery track commitments emerging

1. **Split the stakeholder's "drift" complaint into three mechanisms.**
   - Overlay 2D drift: our projection/renderer/fusion chain; tomato defect is real and quantified.
   - Hand metric error: current sub-10 wrist/root result is GT-fitted smooth drift correction; the deployable path replaces the GT fit with GT-free smooth self-calibration while keeping all-joint MPJPE and surface/MPVPE in the optimization vector.
   - Head/camera trajectory error: current evidence is weak and likely decimeter-class under HaWoR/DROID world; the first deliverables are a fixed-gauge camera evaluator, stronger metric pose source, synchronization/extrinsics calibration, and trajectory repair toward the same 5mm ideal.

2. **Minimal delivery module set (no HOI):** ingestion, calibration contract, conditional UniDepth, camera/head trajectory, HaWoR metric MANO, WiLoR visible geometry, hybrid+temporal fusion, GT-free smooth-drift self-calibration, deterministic renderer, self-consistency QC, offline evaluators. Object planning/SAM2/TRELLIS/object pose/contact/factor graph are out of the default delivery lane except semantics/captioning may use object evidence as a separate lane.

3. **Blunt correctness fixes before API:** one calibrated K (+ distortion) per clip/device used everywhere; eliminate per-frame elastic depth scale; fusion constrained by detector residual not only smoothness; QC/chips must measure final fused layer, not pre-fusion rows; deterministic renderer with no silent intrinsics fallback; separate camera/head vs hand benchmark tables.

4. **Serving architecture:** chosen default is async job API + video-aware coalescer + Ray Serve/Ray GPU actors on a private GPU fleet; this is the simplest solution for arbitrary PyTorch functions, persistent residency, batching, and video-affinity scheduling. Triton/PyTriton follows only after tensor contracts stabilize; KServe is later outer orchestration; avoid TorchServe. Measure `module_speed_x`, GPU utilization, queue wait, batch fill. 10k video-hours/week = 59.5 realtime streams continuously per module.

5. **API output:** public API endpoints use domain nouns such as `POST /v1/annotation-jobs`, not internal track names. Base artifact schema is `ego.annotation.output` v1 with manifest, Parquet tables (`frames`, `head_camera`, `hand_states`, `semantic_clips`, `validation_metrics`), NDJSON overlay/caption/provenance/errors, renders. HOI is explicitly excluded from base schema and must live in a domain extension namespace such as `org.ego.hoi`.

6. **Metrics:** GT-free metrics are disagreement between independent measurement paths and route error-reduction work; they do not redefine the target. Required protected vector: camera/head ATE/RPE/rotation/scale plus static reprojection/3D closure; hand wrist/root, all-joint MPJPE, MPVPE/surface, reprojection, cross-detector, size-depth, visibility, jitter; rendered-overlay drift; intrinsics sweep; semantic segment/caption metrics; throughput/API health.

## Research track commitments emerging

1. V19 HOI failure diagnosis: graph is starved of the right measurements. Object pose is fit against channels that do not observe failing directions; object shape is single-frame prior; contact factors ask for visible evidence removed by occlusion; hand factor graph is functionally passthrough+gate.
2. Right research backbone is correspondence-first rigid-body extraction: temporal 2D/3D tracks -> rigidity statistic -> robust Procrustes pose -> pose-aligned multi-frame fusion -> prior completion only for never-seen regions with uncertainty labels.
3. Factor graph remains useful only after measurements exist; it needs drift latents, switchable contact mixtures, contact likelihoods from occlusion-surviving channels, fitted noise models, liveness/identifiability audit, and gauge declarations.
4. Detection + factor graph is not wrong, but semantic detection must become one object-hypothesis source alongside motion discovery. RL/physics is a feasibility projection experiment; end-to-end is a distillation endpoint, not current pipeline replacement.
5. Research experiments E1-E9 from subagent 7 form the research backbone; E5 GT-free drift latent is the designed bridge back into delivery.

## Resolved operating choices

- Customer/API inputs may include device calibration, VIO/SLAM/IMU, or head-pose metadata, but the pipeline must not depend on them. The schema reserves optional fields; metadata-absent jobs use the calibration resolver and video-derived trajectory with explicit gauge uncertainty.
- The first hand optimization priority after wrist/root is all-joint MPJPE. MPVPE/surface, visibility under occlusion, projection, and temporal stability remain protected metrics.
- The camera/head promotion benchmark is an in-house fixed-gauge fiducial/mocap lockbox with hidden GT, calibrated sync, and known camera/head extrinsics. Public camera-sidecar datasets are development/regression checks.
- The simplest initial deployment is FastAPI job ingress plus Ray Serve/Ray GPU actors on a private GPU fleet. Kubernetes/KServe is deferred until the Ray fleet needs outer orchestration.
