# RGB→SAM3D annotation integration (owner-approved, additive demo layer)

Base: bf2d2ed146518b6e566d7fd4da5f8d032d20a8b4. New isolated integration branch. No writes to previously published demo, existing outputs, or formal state.

## Four mandatory priorities
1. Close fresh pose production, never load historical RGB/pose initializer caches.
2. Use a SAM3D-only mesh branch; TRELLIS comparison is optional research, not required input.
3. Demo consumes THIS run's predictions. Licensed weights/source assets may be cached; clip-specific predictions may not masquerade as fresh.
4. One final annotation package holds core metric/physical state and additive demo state, driving corresponding MP4/world-RRD without mixing layers.

## Phase retention

P00 input/run creation; P01 frame/time/size; P03 camera contract/intrinsics estimate and depth; P04 camera/HaWoR MANO; P05 model/agent object plan; P06 grounded detector; P07 segmentation; P08 base annotations; P09 eligible visible geometry and model/agent anchor choice; P10 physical branch; P11 native evidence; P12 SAM3D; P13 native→canonical adaptation; P14 visible pose; RGB full-timeline initializer/refinement; P14b multi-view support; P15 temporal graph; P16 constraint measurement; P17 contact/occlusion prior; P18 interval correction; P18b preserve metric MANO/contact-surface split; P19 canonical physical state/render; additional demo repair; final annotation and paired-layer world RRD; publication/visual review. No stage may be silently replaced by a flag or ledger.

Original sign/physical-authority checks remain intact: uncertain partial geometry may yield unsigned/unresolved state; this does not suppress demo output but cannot promote it to physical truth. The optional observed-only signed-geometry producer remains available as an explicit evidence path, never a generated full mesh sign fallback.

## Initial scope and contracts

One model-selected manipulated rigid object; one or two predicted hands; full original frame count/duration. Dynamic frame IDs/timestamps, image dimensions and hand validity. Object plans/anchor/interaction judgments are produced by the existing visual runtime agent from raw images, not supplied as secret per-case scripts or hard-coded category state machines. Unknown/nonrigid branch stays explicit rather than inventing a rigid mesh substitute. Multi-object expansion is outside first integration slice.

Use raw RGB plus optional prediction-side calibration (never camera/object/hand GT). Pure RGB requires predicted intrinsics/depth; world is sequence-local with an estimated metric scale, not externally surveyed ground truth.

## Implementation seams

- Parent owns SAM3D-only adapter, same-run pose seed/refine, phase spec/run planning and integration tests.
- Isolated demo writer owns existing demo fitter/renderer/runner/geometry plus new timeline utilities/tests. Generalize N/dimensions/missing-hand handling; preserve actual MANO LBS and all triangles. New state fields: frame_idx,time_s,render_size_wh,intrinsics_render (keep intrinsics_960 legacy alias only when appropriate),left_valid,right_valid, plus existing meshes/poses. Missing sides are not zero meshes rendered at origin.
- Isolated state writer owns new `assemble_v19_annotation_layers.py` and `render_v19_annotation_layers_rrd.py` plus their tests. Assembly consumes P19 physical render state + original annotations/HaWoR + optional demo NPZ/report; preserve P16/P17/P18/P18b verbatim under metric_prediction, no promotion by demo. Store relative self-contained asset paths/hashes in a new final directory, frame-wise camera/objects/hands/interaction state, and additive demo layers. RRD reads only final package, logs both layers separately, one shared camera/trajectory and timestamp timeline. Never join a demo object pose to core metric hands accidentally.

## Artifact validation

Small synthetic tests must fail for wrong/missing transforms, reordered frame IDs, missing hands, same-run path escapes or changed source assets. Test core/demo state separation and actual RRD generation on a short synthetic rigid mesh + hand geometry. Real milk replay is permitted as an explicitly labeled integration replay to exercise new packaging/rendering; it is NOT a fresh prediction run. Fresh runtime setup/execution requires curated workspace and preflight before inference. Report actual blocker if external assets/launch are not ready rather than pretending a plan is a completed fresh run.

No push/merge in this task without new authorization. Preserve named mechanisms and actual video output; docs/phase plans are only backing evidence.
