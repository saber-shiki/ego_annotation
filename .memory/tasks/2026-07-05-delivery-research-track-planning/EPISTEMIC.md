# Current epistemic model

The stakeholder is not asking for more HOI demo polish. The delivery problem is a productization problem: separated head/camera and hand accuracy near 5mm, drift-free reprojection in visible videos, semantic clip segmentation/captioning, and a directly callable high-throughput API. The research problem remains HOI physical extraction and should not be allowed to slow the delivery pipeline.

Live uncertainties:
- Which vision modules are necessary and sufficient for sub-centimeter head/camera + hand accuracy without HOI.
- Whether visible drift is dominated by intrinsics, crop/pinhole adapter, coordinate conventions, metric scaling, or stale render layer wiring.
- Which API/batching stack gives fastest path to 10k video-hours/week while preserving per-module observability.
- Which GT-free self-consistency metrics can drive iteration before full labels exist.
- Which HOI research bottlenecks are real physical-model limits versus implementation mistakes.
