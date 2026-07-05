## Objective
Build the delivery-track execution system: a high-throughput egocentric-video annotation API that returns numeric head/camera state, metric hand state, fine-grained semantic clips/captions, validation metrics, provenance, explicit errors, and optional QC/demo renders. The target is uniform 5mm-level error minimization across head/camera, wrist/root, all-joint MPJPE, hand surface/MPVPE, projection, visibility, and temporal stability. Done when the pipeline has an API alpha, metric vector/evaluator, correctness fixes, caption alpha, throughput alpha, and pilot-capacity forecast as defined in `TASK_PACK.md`.

## Workbench
1. Lock M0 metric vector, optional calibration/VIO/head-pose interface, in-house fixed-gauge head/camera promotion benchmark, and FastAPI+Ray Serve deployment baseline.
2. Build M1 measurement harness: module-speed benchmark, camera/head evaluator scaffold, HOT3D hand all-joint/wrist metrics, and final-layer QC recomputation on tomato/trash.
3. Implement M2 correctness patches: calibration resolver, no elastic depth scale, detector-bounded hand fusion, deterministic renderer with no silent intrinsics fallback.
4. Build M3 API alpha around `POST /v1/annotation-jobs` and `ego.annotation.output` v1 manifest/tables/errors/provenance.
5. Build M4-M6 caption alpha, throughput alpha, and pilot release forecast.

## Context
Repo root: `/home/yiwen/ego_annotation`.
Canonical delivery task pack: `.memory/tasks/2026-07-06-delivery-track-execution/TASK_PACK.md`.
Planning source: `.memory/tasks/2026-07-05-delivery-research-track-planning/DELIVERY_TRACK_TASK_PACK_DRAFT.md`.
Project invariants: `.memory/project/pipeline_invariants.md`, `.memory/project/hand_metric_mechanisms.md`, `.memory/project/self_consistency_metrics.md`.
Demo pack reference: `/data2/ego_annotation_outputs/demo_pack_20260704/release/ego_annotation_demo_pack_20260704.zip`.
Default serving choice: FastAPI job ingress + Ray Serve/Ray GPU actors on a private GPU fleet.
Head/camera promotion benchmark choice: in-house fixed-gauge fiducial/mocap lockbox with hidden GT, calibrated sync, and known camera/head extrinsics.
Hand priority after wrist/root: all-joint MPJPE.

## Task specifications
Use product/domain API names only: public endpoint `POST /v1/annotation-jobs`, base schema `ego.annotation.output` v1, optional HOI extension namespace separate from base delivery output.
Customer inputs may include device calibration, VIO/SLAM/IMU, or head-pose metadata; the API reserves those fields, but jobs must also run when metadata is absent.
Calibration resolver must produce one canonical K plus distortion/rectification/source/uncertainty per clip/session; silent fallback intrinsics are invalid.
Default delivery excludes HOI factor graph, object pose optimization, contact ownership, signed nonpenetration, object mesh reconstruction, TRELLIS completion, and full-rate dense SAM2/object segmentation.
Captioning lane uses existing task/action captions or batched external/agent caption review; do not budget it as a local GPU/vLLM lane unless scope changes.
Primary customer product is numeric/tables/metrics/provenance/errors; renders are QC/demo projections of the numeric state and cannot change numeric results.
Every delivery experiment needs a causal card: numeric defect, physical variable, mechanism, discriminating measurement, predictions, intervention, and expected metric/render state change.
Metric vector must report p50/p95/RMSE or appropriate distribution summaries for head/camera ATE/RPE/rotation/scale, wrist/root, all-joint MPJPE, MPVPE/surface where available, reprojection, visibility, jitter, semantic grounding, and throughput.
Throughput target is 10,000 video-hours/week, i.e. 59.5 realtime aggregate per active module; measure module_speed_x, GPU-hours/video-hour, queue wait, batch fill, worker residency, and explicit failure rates.

## Constraints
Do not shrink the 5mm ideal to the easiest current metric.
Do not treat self-consistency metrics as fixed-gauge metric evaluation.
Do not let HOI/factor-graph machinery enter the delivery default path.
Do not claim progress from schema, reports, validators, or renders unless a required numeric/semantic/throughput mechanism changed.
Do not run heavy local GPU/model inference on the workstation; use server/A800 or approved non-local compute.
Do not use public API names containing delivery/research/internal track labels.
Do not stage unrelated repository changes.
