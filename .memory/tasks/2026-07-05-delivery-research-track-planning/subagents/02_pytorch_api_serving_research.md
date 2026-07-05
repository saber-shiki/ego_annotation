# Subagent 2 — PyTorch vision model serving/API research

## Scope and bottom line

Mode: synthesis. The question is not which serving library can expose a PyTorch model over HTTP; many can. The decisive mechanism is whether the serving layer can keep GPUs saturated while preserving video-temporal state: adjacent chunks from the same video should share decode, cached frames, masks/tracks, and boundary states instead of being treated as unrelated HTTP requests.

Recommendation:

1. **Implement the first production path as a video-aware async API + custom scheduler + Ray Serve/Ray actors on GPU nodes.** This gives the fastest path for arbitrary existing PyTorch modules (SAM2, UniDepth, WiLoR, caption/VLM, camera tracker) while supporting GPU placement, autoscaling, dynamic batching, observability hooks, and stateful per-video actors. Ray Serve explicitly supports GPU-bound deployments, `@serve.batch(max_batch_size, batch_wait_timeout_s)`, custom batch sizing by cost, FastAPI ingress, autoscaling, and video-analysis examples using GPU replicas and batched frame encoding.[^ray-batch][^ray-video]
2. **Use Triton/PyTriton selectively after tensor contracts stabilize.** Triton is the strongest low-level inference server for stateless or sequence-shaped tensor calls: dynamic batching, sequence batching, model instance groups, per-model statistics, Prometheus metrics, perf_analyzer/model-analyzer, and explicit GPU instance placement.[^triton-batcher][^triton-stats][^pytriton] It is a poor first wrapper for research-stage modules with Python-heavy preprocessing/postprocessing and mutable temporal state unless PyTriton is used.
3. **Serve VLM/captioning with a specialized LLM/VLM server where supported: vLLM or Hugging Face TGI.** These are the actual vLLM-style analogs for captioning because they optimize token scheduling/KV cache and expose OpenAI-compatible multimodal APIs. vLLM supports multimodal inputs via OpenAI-compatible Chat Completions and media UUID caching; TGI documents VLM image inputs, chat completions, streaming, Prometheus/Grafana monitoring, tensor parallelism, and guided JSON output.[^vllm-mm][^tgi-vlm]
4. **Do not choose TorchServe for new production.** Official PyTorch Serve docs now state TorchServe is no longer actively maintained and has no planned updates, bug fixes, new features, or security patches.[^torchserve-maint]
5. **Treat KServe/Knative as a later Kubernetes control plane, not the inner scheduler.** KServe gives InferenceService CRDs, scale-to-zero, KEDA/Knative autoscaling, transformers, inference graphs, GPU scheduling, and Prometheus-driven scaling.[^kserve-autoscale] It does not by itself solve adjacent video chunk colocation or module-specific temporal state.
6. **Use Modal/Runpod/Baseten for burst capacity or deployment-speed experiments, not as the default 10k video-hour/week backbone until cost/data-locality benchmarks prove it.** Modal and Runpod are excellent for fast serverless GPU deployment, cold-start reduction, autoscaling, and queue-backed jobs; they add vendor queue semantics, data egress/storage questions, and weaker control over cross-request colocation.[^modal-batch][^runpod]

## Target throughput arithmetic

Project target: **~10,000 video-hours/week**.

Continuous equivalent video throughput required:

- `10,000 video-hours / 7 days = 1,428.6 video-hours/day`
- `= 59.5 video-hours/hour`
- The system must process **~59.5 realtime video streams continuously per module** if each module runs at 1× realtime per GPU.

Capacity formula per module:

```text
required_gpus = weekly_video_hours * module_wall_seconds_per_video_second_per_gpu / 604800
              = 59.5 / module_speed_x

module_speed_x = processed_video_seconds / wall_seconds / gpu_count
```

Examples per module:

| Measured module speed per GPU | Continuous GPUs for 10k video-hours/week |
|---:|---:|
| 20× realtime | ~3 GPUs |
| 10× realtime | ~6 GPUs |
| 2× realtime | ~30 GPUs |
| 1× realtime | ~60 GPUs |
| 0.5× realtime | ~119 GPUs |
| 0.25× realtime | ~238 GPUs |

This makes throughput instrumentation a design dependency. Any recommendation without measured `module_speed_x`, GPU utilization, queue wait, and batch fill is guessing.

## Recommendation matrix

Legend: **Strong** = use directly; **Good** = useful with caveats; **Weak** = support role only; **Avoid** = not suitable as default.

| Option | HTTP/API speed | Arbitrary PyTorch fit | Dynamic batching | Adjacent video colocation/state | Multi-GPU scheduling | Observability | Deployment speed | 10k h/week fit | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Ray Serve / Ray actors** | Good | **Strong** | **Strong** via `@serve.batch`; custom batch size by pixels/tokens/frames | **Strong if custom scheduler/sticky actors are added** | **Strong** via `num_gpus`, actors, autoscaling, KubeRay | Good: dashboard, metrics, logs; add app metrics | Good | **Strong first path** | Default control plane and module wrapper. |
| **NVIDIA Triton** | Good once model repo exists | Good for TorchScript/ONNX/TensorRT/Python backend; weaker for messy Python state | **Strong** dynamic/ragged/sequence batching | Medium: sequence batching helps but video colocation logic still external | **Strong** instance groups and GPU placement | **Strong** model stats, Prometheus, perf_analyzer/model-analyzer | Medium; export/config cost | **Strong optimization target** | Use for stable stateless modules and PyTriton wrappers. |
| **PyTriton** | Good | **Strong** for Python functions | Good through Triton features + `@batch` | Medium; still needs external video scheduler | Good | Good through Triton, perf_analyzer, OTEL guide | **Good** | Good | Bridge from Python modules into Triton without full export. |
| **BentoML** | Good | Strong for Python services | Good adaptive batching with `batchable=True`, `max_batch_size`, `max_latency_ms` | Weak/Medium; no native video-temporal scheduler | Medium: workers/resources; less cluster scheduling than Ray | Good: Prometheus/monitoring APIs | **Strong** | Medium | Good for single services and quick packaging; less suited as global scheduler. |
| **KServe/Knative** | Medium once Kubernetes exists | Medium; depends on runtime/container | Depends on underlying server | Weak/Medium; inference graphs are request DAGs, not temporal colocation | Strong at Kubernetes resource level | Good with Prometheus/KEDA/Grafana | Weak initially; strong after platform exists | Good later | Use as outer platform after scheduler and module contracts settle. |
| **FastAPI + custom batcher** | **Strong fastest local start** | **Strong** | Strong if implemented carefully | **Strong** because custom logic can match video IDs/time ranges | Weak alone; add Ray/K8s | Weak unless instrumented | **Strong** | Medium alone, strong with Ray | Good first ingress/prototype; avoid hand-rolled GPU fleet alone. |
| **Celery/Redis/RabbitMQ queues** | Good for async job submission | Strong | Weak; task queues do not batch tensors by themselves | Medium for durable per-video queues, weak for GPU batching | Weak; no GPU-aware scheduler | Medium queue depth/retry metrics | Good | Weak as serving core | Use only for durable job queue/control-plane tasks, not GPU inference batching. |
| **Modal** | **Strong** | Strong | Good via `@modal.batched(max_batch_size, wait_ms)` | Weak/Medium; serverless workers obscure sticky video state | Good managed GPU options/fallbacks | Medium/Good endpoint metrics | **Strong** | Medium if cost/data locality works | Best for burst and rapid experiments; benchmark before default. |
| **Runpod serverless** | Strong | Strong Docker/handler | Queue-level, custom in worker | Weak/Medium; request queue not video-aware by default | Good managed GPU workers | Medium logs/debug/queue | **Strong** | Medium if cost/data locality works | Burst/offload option; better for coarse jobs than fine temporal colocation. |
| **Baseten/Truss** | Strong managed endpoint | Strong Python/custom Docker | Possible/custom; managed scaling | Weak/Medium | Good managed | Good managed logs/metrics/tracing | **Strong** | Medium if vendor acceptable | Good for rapid external managed deployments, especially VLM/custom models. |
| **vLLM / TGI** | **Strong for VLM text generation** | Narrow: supported VLM/LLM models only | **Strong for token scheduling**, not SAM2/UniDepth | Weak except media/image cache and prompt batching | Strong tensor/data parallel for LLM/VLM | Good Prometheus/Grafana in TGI; vLLM ecosystem metrics | Good | Strong for captioning only | Use for semantic captioning/VLM, not general CV modules. |
| **NVIDIA Dynamo** | Strong for generative AI fleets | Narrow LLM/reasoning engines | LLM-specific | Weak for SAM2/UniDepth | Strong for LLM fleet orchestration | Good LLM ops | Medium | Narrow | Not a general vision-serving answer; revisit only for large VLM fleet. |
| **TorchServe** | Historically good | PyTorch-native | Batch size + max batch delay | Weak | Medium workers/GPU limit | Medium metrics API | Medium | Avoid | Official limited maintenance makes it unsuitable for new production. |

## Why vLLM has no exact general-vision equivalent

vLLM works because LLM serving has a common execution object: autoregressive token streams with KV cache, prefill/decode phases, and schedulable token budgets. SAM2, UniDepth, WiLoR, DROID/camera tracking, and VLM captioning do not share one scheduler state:

- SAM2/video segmentation benefits from temporal propagation, masks, object IDs, and adjacent-frame memory.
- UniDepth is mostly stateless dense image inference plus preprocessing/resolution decisions.
- WiLoR/hand models combine detector/crop geometry, hand-specific postprocessing, and temporal smoothing.
- DROID/camera tracking is stateful over long sequences and has boundary conditions unlike frame classifiers.
- VLM/captioning is LLM-like after visual encoding and should use vLLM/TGI if model compatibility is good.

The closest equivalent architecture is therefore **two-level serving**:

1. a **video-aware scheduler** that understands video IDs, time ranges, frame rates, temporal overlap, state ownership, priorities, and downstream artifact paths;
2. specialized **per-module executors** that use the best batching/scheduling primitive for that module: Ray batch, Triton dynamic batch, Triton sequence batch, vLLM/TGI token scheduler, or a long-running stateful actor.

## Batching and colocation design

### Public API contract

Use an asynchronous job API, not synchronous frame upload:

```text
POST /v1/jobs
  input: video_uri, optional frame_range, requested_modules, priority, deadline, output_uri
  output: job_id

GET /v1/jobs/{job_id}
  output: status, module statuses, progress, artifact URIs, metrics summary

POST /v1/modules/{module}/runs    internal only
  input: normalized clip bundle references, not raw frame blobs when possible
```

Video data should be referenced by object-store URI or shared filesystem path. Large frame payloads over HTTP will move the bottleneck from GPU to network/serialization.

### Scheduler objects

Each unit of work should carry enough structure for colocation:

```text
TaskKey = (
  module, model_version, video_id, camera_id,
  frame_start, frame_end, fps, resolution_bucket,
  temporal_context_left, temporal_context_right,
  state_affinity_key, priority, deadline
)
```

### Two-tier batching

**Tier 1: temporal coalescer.** Groups adjacent chunks before model batching.

Rules:

- Sort pending tasks by `(module, model_version, video_id, camera_id, frame_start)`.
- Merge adjacent or overlapping chunks when `(gap_frames <= G)` and combined duration stays under module memory/time budget.
- Preserve overlap halos for modules that need boundary context: e.g., `left/right_context_frames` for SAM2 propagation, camera tracking, temporal hand smoothing.
- Route stateful tasks with the same `state_affinity_key = module:model_version:video_id[:track_id]` to the same actor until the actor emits a checkpointed boundary state.
- Emit a `collocation_savings` metric: decoded frames reused, model state reused, duplicate overlap frames avoided.

**Tier 2: GPU microbatcher.** Batches compatible tensor work inside each module.

Rules:

- Batch by **cost**, not request count. Cost should be frames × pixels for image models, total tokens/images for VLMs, active masks/tracks for SAM2, and estimated graph/keyframe count for camera trackers.
- Keep separate queues by model, device type, dtype, resolution bucket, and memory profile.
- Use short wait windows for online calls (10–100 ms) and larger windows for offline throughput jobs (0.5–5 s) when latency is not user-visible.
- Use admission control: if `estimated_vram_bytes + resident_model_bytes > safety_fraction * device_vram`, split before dispatch.
- Batch boundaries must be logged. Silent padding/truncation can corrupt physical timelines.

### Module-specific scheduling

| Module | Best initial executor | Batching policy | State/colocation rule |
|---|---|---|---|
| SAM2 | Ray actor or PyTriton if stable | Batch prompts/frames by total pixels and active objects; longer temporal bundles for propagation | Sticky per video/object track; preserve mask state and boundary halos. |
| UniDepth | Triton/PyTriton or Ray batch | Stateless frame/image microbatches by resolution bucket | No temporal stickiness required; colocate only to reuse decoded frames. |
| WiLoR/hand | Ray actor first; Triton later if tensorized | Batch crops/frames by total crop pixels; separate detector and hand model if detector bottlenecks | Use temporal bundle for smoothing and source-switch hysteresis; avoid per-frame independent API calls as final output. |
| VLM/captioning | vLLM/TGI if model supported; Ray wrapper otherwise | Batch by image count + prompt token budget; use 2–3s clip units | Cache visual embeddings/keyframes by clip; use structured JSON output where possible. |
| DROID/camera tracker | Long-running Ray actor/job, not microbatched request server | Parallelize by video/segment; avoid small request batching | Stateful sequence processing; chunks need boundary state and loop-closure/global refinement handling. |
| Pre/postprocess/decode | CPU Ray actors or local workers near GPU | Batch reads and decode frame ranges | Co-locate decoded frame cache with GPU node; avoid repeated object-store pulls. |

## Multi-GPU scheduling design

Initial design:

- One long-lived worker process/actor per `(module, model_version, gpu_id)` for heavy models. Load model once; pin CUDA device explicitly.
- Use data parallelism for SAM2/UniDepth/WiLoR: many independent GPU actors, each with one model replica.
- Use tensor/pipeline parallelism only for large VLM/captioning models that require it; vLLM/TGI already have the right abstractions for this class.
- Keep memory-heavy models from co-residing unless measured. Multi-model colocation can improve utilization when one model is CPU/preprocess-bound, but it can also create VRAM fragmentation and OOM oscillation.
- Reserve CPU decode/preprocess workers per GPU worker. Underfeeding GPU because ffmpeg/JPEG/depth loading is serialized will look like “model slow” unless measured separately.

Triton path:

- Use `instance_group` for per-GPU model instances and multiple instances per GPU when profiling proves input/output overlap helps.[^triton-instance]
- Enable `dynamic_batching { preferred_batch_size: [...] max_queue_delay_microseconds: ... }` for stateless modules.
- Use sequence batching only for model-level state machines with clean start/end/correlation semantics; do not force whole video pipeline state into Triton if the state spans decode/cache/artifact side effects.

Ray path:

- Use `ray_actor_options={"num_gpus": 1}` per GPU model replica and autoscaling configs for replicas.[^ray-video]
- For GPU memory variable workloads, implement `batch_size_fn` based on total pixels/frames/tokens rather than number of requests.[^ray-batch]
- Use Ray object store or node-local cache for decoded frames and intermediate tensors, but record eviction/staleness explicitly.

## Observability requirements

Expose metrics at four layers.

### API/job layer

- `jobs_submitted_total`, `jobs_completed_total`, `jobs_failed_total`
- `job_e2e_seconds{module_set,priority}`
- `artifact_bytes_written`, `artifact_write_seconds`
- per-video status: submitted, decoding, queued, running, writing, failed, complete

### Scheduler layer

- `queue_depth{module,priority,resolution_bucket}`
- `queue_wait_seconds{module}` p50/p95/p99
- `batch_fill_ratio{module}` = actual cost / max cost
- `batch_wait_seconds{module}`
- `collocated_chunks_total`, `adjacent_merge_ratio`, `decoded_frame_reuse_ratio`
- `state_affinity_misses_total` for stateful modules
- `backpressure_rejections_total` with cause

### Model/GPU layer

- `processed_video_seconds_total{module,model_version}`
- `wall_seconds_total{module}` and `gpu_seconds_total{module}`
- `module_speed_x = processed_video_seconds / wall_seconds / gpu_count`
- GPU utilization, SM occupancy proxy, memory used/free, OOM count, CUDA retry count
- split timers: decode, preprocess, host→device copy, inference, postprocess, write
- batch stats: frames/images/tokens/pixels per execution

Triton already exposes model statistics including inference count, execution count, queue/compute durations, batch stats, and Prometheus metrics such as inference queue summaries.[^triton-stats] BentoML exposes Prometheus-style histograms and monitoring hooks; Modal exposes endpoint latency/throughput/running/queued metrics; KServe can autoscale from Prometheus/KEDA metrics.[^bentoml-monitor][^modal-metrics][^kserve-autoscale]

### Artifact correctness layer

Because this project's target is renderable video annotations rather than API bookkeeping, serving metrics must join to output sanity metrics:

- frame count/duration preservation
- hand reprojection residual and projected-size consistency
- caption clip coverage and overlap gaps
- camera/head trajectory continuity
- per-module confidence/uncertainty summaries
- stale dependency detection: output module version and input artifact hashes must match the run graph

## Minimal implementation path

### Phase 0 — measure current modules without changing algorithms

Deliverable: benchmark harness around existing module commands.

For each module on representative videos, record:

```text
module, model_version, gpu_type, gpu_count
input_video_seconds, input_frames, resolution
wall_seconds_cold, wall_seconds_warm
decode_seconds, preprocess_seconds, inference_seconds, postprocess_seconds, write_seconds
processed_video_seconds_per_gpu_second
peak_vram_gb, mean_gpu_util, p95_gpu_util
failure_type if any
```

This phase answers the capacity question before platform work. If a module is 0.25× realtime, no serving framework will save the 10k h/week target without many GPUs or algorithmic thinning.

### Phase 1 — first API with custom scheduler and Ray workers

- Build async job API around video URI and artifact URI.
- Implement module registry: command/wrapper, model version, resource class, batch policy, state policy.
- Implement temporal coalescer and GPU worker actors.
- Add per-module workers for UniDepth, SAM2, WiLoR, VLM/captioning as wrappers around current code paths.
- Produce one end-to-end run that returns full-length artifacts and metrics, even if module outputs are approximate.

Why Ray first: it avoids export friction and supports arbitrary Python, GPU resource placement, dynamic batching, and actor state. It keeps the hard mechanism—video-aware scheduling—in project code rather than burying it in a model server that does not know video semantics.

### Phase 2 — stabilize high-throughput modules

- Convert stateless/frame-level modules to PyTriton/Triton where profiling shows throughput improvement.
- Keep stateful temporal modules as Ray actors unless sequence semantics are clean enough for Triton sequence batching.
- Move VLM/captioning to vLLM/TGI when the chosen VLM is supported and output schema can be constrained.
- Add load tests at target concurrency and batch windows.

### Phase 3 — fleet/platform hardening

- If operating Kubernetes, package Ray Serve/KubeRay and/or KServe InferenceServices.
- Add Prometheus/Grafana dashboards and OpenTelemetry trace IDs across job→chunk→module→artifact.
- Add autoscaling policies based on queue depth, GPU utilization, and deadline miss rate.
- Add canary model versions and rollback by module version.

### Phase 4 — optional managed overflow

- Add Modal/Runpod/Baseten backends for burst/offline overflow only after the same benchmark harness measures cost per processed video-hour, cold-start tax, artifact transfer time, and retry behavior.

## Failure modes and mitigations

| Failure mode | Mechanism | Observable symptom | Mitigation |
|---|---|---|---|
| Adjacent chunks processed independently | HTTP request batching ignores video time | duplicate decode, boundary discontinuities, SAM2/DROID drift at chunk edges | temporal coalescer, state affinity keys, overlap halos, boundary state checkpoints |
| GPU OOM from variable clips | batch size by request count ignores pixels/frames/objects/tokens | sporadic CUDA OOM at high-res or many-object clips | cost-based batch sizing, VRAM estimator, split before dispatch, record batch cost |
| Queue starvation | large offline jobs fill wait windows and block small urgent jobs | high p95 latency despite free average capacity | priority queues, aging, max bundle duration, per-priority worker shares |
| Cold-start/model-load tax | scale-to-zero or serverless workers unload models | first jobs miss deadline; metrics look worse at low traffic | warm pools for required throughput; measure cold and warm separately |
| Non-idempotent retries | retried task writes partial artifacts or advances state twice | corrupt outputs, duplicated rows, mismatched manifests | idempotent artifact paths with run IDs; atomic writes; retry only from checkpoint |
| Hidden preprocessing bottleneck | GPU model is fast but decode/crop/depth load serializes | low GPU utilization, growing queues | node-local frame cache, CPU worker pool, split timing metrics |
| Multi-model colocation hurts | two models fight for VRAM/cache/PCIe | lower throughput than isolated models | isolation by default; colocate only after benchmark matrix proves gain |
| Triton export mismatch | TorchScript/ONNX changes preprocessing or dynamic shape semantics | fast but wrong outputs | PyTriton first; compare outputs to Python baseline on fixed clips |
| Scheduler reports success while artifact is stale | module output hashes not connected to final render state | API says complete; visual annotation uses old data | artifact dependency hashes, stale-state validation, final artifact sanity metrics |
| Vendor queue obscures state | Modal/Runpod distributes requests to arbitrary workers | state cache misses, repeated model load/decode, inconsistent temporal chunks | use vendor only for coarse full-video/large-chunk jobs or stateless modules |
| VLM batching dominates caption quality | overly large visual/text batches force truncation or low-res sampling | captions miss fine-grained 2–3s events | token/image budget metrics, structured output validation, clip coverage metric |

## Practical defaults to start

These are starting points for measurement, not acceptance thresholds.

| Module class | Initial chunk | Overlap halo | Batch wait | Batch cost cap |
|---|---:|---:|---:|---|
| UniDepth/stateless frames | 32–128 frames | 0 | 50–500 ms | total pixels / VRAM |
| WiLoR/hand | 2–5 s | 0.5 s | 50–500 ms | crop pixels + detected hands |
| SAM2 | 5–20 s | 1–2 s | 0.5–2 s offline | frames × pixels × objects |
| VLM/captioning | 2–3 s semantic clip | optional neighboring keyframes | 100 ms–2 s | images + prompt tokens + max output tokens |
| DROID/camera | whole video or long segments | large boundary/keyframe state | job-level, not microbatch | sequence length/keyframes |

## Source-grounded observations

- Ray Serve has first-class dynamic request batching: `@serve.batch(max_batch_size, batch_wait_timeout_s)`, optional `max_concurrent_batches`, and `batch_size_fn` for custom cost metrics like total tokens/nodes. Its docs explicitly warn that downstream batch sizes should align with upstream batch sizes in deployment graphs.[^ray-batch]
- Ray Serve's video-analysis tutorial uses GPU replicas, FastAPI ingress, autoscaling, and batched frame encoding by concatenating frames across requests and splitting outputs afterward.[^ray-video]
- Triton dynamic batching combines inference requests server-side for stateless models; sequence batching exists for stateful models with start/end/correlation controls; instance groups place multiple model instances across GPUs; stats expose inference count, execution count, queue/compute timings, and batch stats.[^triton-batcher][^triton-stats]
- PyTriton exposes Python functions through Triton HTTP/gRPC and preserves Triton features such as dynamic batching and response cache without changing the Python model environment.[^pytriton]
- BentoML supports adaptive batching via `@bentoml.api(batchable=True, max_batch_size, max_latency_ms)`, GPU resources/workers, and monitoring/Prometheus-style metrics.[^bentoml-batch][^bentoml-monitor]
- KServe supports GPU InferenceServices, transformer/predictor component autoscaling, scale-to-zero via `minReplicas: 0`, inference graphs with Sequence/Switch/Ensemble/Splitter routers, and KEDA/Prometheus autoscaling.[^kserve-autoscale]
- TorchServe supports batch inference via `batch_size` and `max_batch_delay`, but the official docs state it is no longer actively maintained.[^torchserve-batch][^torchserve-maint]
- Modal supports GPU functions with specific GPU types/counts and dynamic batching via `@modal.batched(max_batch_size, wait_ms)`; endpoint metrics show latency, throughput, running, and queued requests.[^modal-gpu][^modal-batch][^modal-metrics]
- Runpod Serverless exposes endpoint/worker/handler abstractions, queues requests when no worker is available, cold-starts workers, and charges for compute time used. This is useful for burst, but the queue is not a video-aware scheduler.[^runpod]
- TGI VLM docs support image URLs/base64 in prompts, chat-completion style image inputs, streaming, OpenAI client compatibility, and JSON schema-guided generation for image-derived structured outputs.[^tgi-vlm]

## Residual risks

1. **Actual module speeds are unknown in this research artifact.** The capacity estimate is formulaic until measured on A800/server hardware with representative videos.
2. **SAM2 and camera-tracker state contracts must be specified before safe chunking.** Incorrect boundary state design can create visually plausible but physically inconsistent outputs.
3. **VLM/captioning server choice depends on the selected model.** vLLM/TGI support varies by architecture and modality; unsupported models fall back to a Ray/PyTorch wrapper.
4. **Data movement may dominate at 10k h/week.** Object-store reads, video decode, frame cache misses, and artifact writes must be measured alongside inference.
5. **Managed GPU platforms may fail cost or locality constraints.** Modal/Runpod/Baseten should be benchmarked with real video artifact transfer, not just model inference latency.

## Minimal decision for parent synthesis

Use this serving stack in the delivery plan:

```text
Public API: FastAPI async job service
Scheduler: custom video-aware coalescer + durable job metadata
Execution: Ray Serve/Ray actors on GPU nodes
Optimized model servers: Triton/PyTriton for stable stateless vision modules; vLLM/TGI for VLM/captioning
Platform later: KubeRay/KServe if Kubernetes fleet management becomes necessary
Overflow: Modal/Runpod/Baseten only after cost/locality benchmark
Avoid: TorchServe for new production
```

This path keeps the project-specific mechanism—temporal colocation and artifact-aware observability—inside the system while using mature serving primitives where they fit.

[^ray-batch]: Ray Serve dynamic request batching docs: https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html
[^ray-video]: Ray Serve video-analysis tutorial snippets retrieved from Context7 `/ray-project/ray`, showing GPU `VideoEncoder`, FastAPI ingress, autoscaling, and batched frame encoding.
[^triton-batcher]: NVIDIA Triton batching docs / Context7 `/triton-inference-server/server`, dynamic batching, sequence batching, ragged batching, and queue delay configuration: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
[^triton-stats]: Triton statistics and optimization docs / Context7 `/triton-inference-server/server`, including inference count, execution count, queue/compute timings, batch stats, perf_analyzer/model-analyzer.
[^triton-instance]: Triton model configuration docs for `instance_group` and GPU placement, retrieved via Context7 `/triton-inference-server/server`.
[^pytriton]: PyTriton overview: https://triton-inference-server.github.io/pytriton/latest/
[^bentoml-batch]: BentoML adaptive batching docs / Context7 `/bentoml/bentoml`, `@bentoml.api(batchable=True)`, `max_batch_size`, `max_latency_ms`.
[^bentoml-monitor]: BentoML monitoring and metrics docs / Context7 `/bentoml/bentoml`, `bentoml.monitor`, Prometheus histograms, GPU worker resources.
[^kserve-autoscale]: KServe docs / Context7 `/websites/kserve_github_io_website`, GPU InferenceService, KPA/KEDA autoscaling, transformers, InferenceGraph, Prometheus scraping.
[^torchserve-maint]: TorchServe official docs, limited maintenance notice: https://docs.pytorch.org/serve/README.html
[^torchserve-batch]: TorchServe batch inference docs: https://docs.pytorch.org/serve/batch_inference_with_ts.html
[^modal-gpu]: Modal GPU docs: https://modal.com/docs/guide/gpu
[^modal-batch]: Modal dynamic batching docs: https://modal.com/docs/guide/dynamic-batching
[^modal-metrics]: Modal endpoint metrics docs: https://modal.com/docs/guide/endpoint-metrics
[^runpod]: Runpod Serverless overview: https://docs.runpod.io/serverless/overview
[^vllm-mm]: vLLM multimodal input docs discovered via web search: https://docs.vllm.ai/en/latest/features/multimodal_inputs/
[^tgi-vlm]: Hugging Face TGI VLM docs: https://huggingface.co/docs/text-generation-inference/basic_tutorials/visual_language_models
