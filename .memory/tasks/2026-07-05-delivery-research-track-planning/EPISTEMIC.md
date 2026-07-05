# Current epistemic model

The stakeholder is not asking for more HOI demo polish. The delivery problem is a productization problem: separated head/camera and hand accuracy near 5mm, drift-free reprojection in visible videos, semantic clip segmentation/captioning, and a directly callable high-throughput API. The research problem remains HOI physical extraction and should not be allowed to slow the delivery pipeline.

## First commitments from completed subagents

Serving/API mechanism: the inner serving problem is video-temporal colocation, not merely HTTP. First production path should be a video-aware async API + custom scheduler + Ray Serve/Ray GPU actors; Triton/PyTriton comes after tensor contracts stabilize; vLLM/TGI is for caption/VLM only; TorchServe should be avoided.

Throughput mechanism: 10k video-hours/week requires about 59.5 realtime streams continuously per module. Every module needs measured `module_speed_x`, GPU utilization, queue wait, batch fill, and load-amortization metrics. Full-rate SAM2 is already the likely budget breaker; hands are not the dominant throughput problem.

Metric mechanism: a useful GT-free metric is disagreement between independent measurement paths. Delivery metrics must route reruns/tuning, not hide outputs. A true ~5mm head/camera accuracy claim requires camera GT/fiducial/IMU validation; current GT-free camera metrics can find drift but cannot certify absolute head accuracy. Current HOT3D sub-10 hand result proves smooth correctability under disclosed GT calibration; deployable hand accuracy still needs GT-free self-calibration.

## Live uncertainties
- Which exact minimal module set is necessary and sufficient for sub-centimeter head/camera + hand accuracy without HOI.
- Whether visible drift is dominated by intrinsics, crop/pinhole adapter, coordinate conventions, metric scaling, or stale render layer wiring.
- What API output schema makes provenance, coordinate frames, semantic clips, and QC metrics implementable without HOI baggage.
- Which HOI research bottlenecks are real physical-model limits versus implementation mistakes.

## First-wave synthesis additions

The tomato drift complaint is no longer generic: shipped tomato left/right fused hands have p95 31.6/24.5 px versus WiLoR 2D, with fallback rows around 100 px off and fusion doubling the tail relative to pre-fusion fit. The dominant product fixes are calibrated K, rigid metric space, detector-bounded fusion, final-layer QC, and deterministic renderer.

Delivery can drop HOI/object/factor-graph work without losing hand/camera accuracy because the metric hand path is already independent of the HOI graph. The hard delivery unknown is head/camera 5 mm; current evidence says no evaluator exists and HaWoR/DROID world is decimeter-class, so the first deliverable is measurement + camera-source policy.

Research should not add more HOI factors until it adds measurements: temporal surface correspondence, rigidity statistics, multi-frame fusion, and contact channels surviving occlusion.
