# Working synthesis from first-wave outputs

## Delivery track commitments emerging

1. **Split the stakeholder's "drift" complaint into three mechanisms.**
   - Overlay 2D drift: our projection/renderer/fusion chain; tomato defect is real and quantified.
   - Hand metric error in camera frame: current sub-10 result is GT-fitted smooth drift correction; deployable claim needs GT-free smooth self-calibration.
   - Head/camera trajectory error: currently unproven and likely decimeter-class under HaWoR/DROID world; first deliverable is a camera evaluator + calibration/trajectory measurement, not a promise.

2. **Minimal delivery module set (no HOI):** ingestion, calibration contract, conditional UniDepth, camera/head trajectory, HaWoR metric MANO, WiLoR visible geometry, hybrid+temporal fusion, GT-free smooth-drift self-calibration, deterministic renderer, self-consistency QC, offline evaluators. Object planning/SAM2/TRELLIS/object pose/contact/factor graph are out of the default delivery lane except semantics/captioning may use object evidence as a separate lane.

3. **Blunt correctness fixes before API:** one calibrated K (+ distortion) per clip/device used everywhere; eliminate per-frame elastic depth scale; fusion constrained by detector residual not only smoothness; QC/chips must measure final fused layer, not pre-fusion rows; deterministic renderer with no silent intrinsics fallback; separate camera/head vs hand benchmark tables.

4. **Serving architecture:** async job API + video-aware coalescer + Ray Serve/Ray GPU actors first; Triton/PyTriton once tensor contracts stabilize; vLLM/TGI for caption/VLM; KServe later as outer platform; avoid TorchServe. Measure `module_speed_x`, GPU utilization, queue wait, batch fill. 10k video-hours/week = 59.5 realtime streams continuously per module.

5. **API output:** public API endpoints use domain nouns such as `POST /v1/annotation-jobs`, not internal track names. Base artifact schema is `ego.annotation.output` v1 with manifest, Parquet tables (`frames`, `head_camera`, `hand_states`, `semantic_clips`, `validation_metrics`), NDJSON overlay/caption/provenance/errors, renders. HOI is explicitly excluded from base schema and must live in a domain extension namespace such as `org.ego.hoi`.

6. **Metrics:** GT-free metrics are disagreement between independent measurement paths. Required: camera static reprojection/3D closure/gravity plausibility + GT ATE/RPE when available; hand wrist GT, visible-joint/root-relative, reprojection, cross-detector, size-depth, jitter; rendered-overlay drift; intrinsics sweep; semantic segment/caption metrics; throughput/API health.

## Research track commitments emerging

1. V19 HOI failure diagnosis: graph is starved of the right measurements. Object pose is fit against channels that do not observe failing directions; object shape is single-frame prior; contact factors ask for visible evidence removed by occlusion; hand factor graph is functionally passthrough+gate.
2. Right research backbone is correspondence-first rigid-body extraction: temporal 2D/3D tracks -> rigidity statistic -> robust Procrustes pose -> pose-aligned multi-frame fusion -> prior completion only for never-seen regions with uncertainty labels.
3. Factor graph remains useful only after measurements exist; it needs drift latents, switchable contact mixtures, contact likelihoods from occlusion-surviving channels, fitted noise models, liveness/identifiability audit, and gauge declarations.
4. Detection + factor graph is not wrong, but semantic detection must become one object-hypothesis source alongside motion discovery. RL/physics is a feasibility projection experiment; end-to-end is a distillation endpoint, not current pipeline replacement.
5. Research experiments E1-E9 from subagent 7 form the research backbone; E5 GT-free drift latent is the designed bridge back into delivery.

## User decisions likely needed

- Metric definition for the stakeholder's "~5mm hand": wrist/root camera-frame (reachable) vs root-aligned/full-joint MPJPE (not currently reachable).
- Camera source policy: will customer/API inputs include device VIO/SLAM/calibration metadata? If yes, delivery head/camera track is ingest+cross-check; if no, we must build/validate camera trajectory from video and cannot claim 5mm until evaluator passes.
- Camera benchmark source beyond HOT3D near-static: Aria/ADT/Nymeria/MPS sidecars or an in-house fiducial capture.
- API deployment assumption: in-house Ray fleet first vs Kubernetes platform from day one; managed GPU only as overflow unless user wants vendor route.
