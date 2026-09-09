# Standalone RGB scene annotation/reconstruction — design amendment

## User request (2026-09-10)

Extract the current visual demo as a complete reusable/general video annotation and reconstruction pipeline. Final deliverables are only `annotation.json` and `world.rrd`; do not render overlay/detail MP4 by default. Investigate and restore the mechanisms behind the exact old published `milk_hand_object_demo.rrd`, especially its superior rotation. Pause the previous core continuation and dual-layer P19-export objective while doing this task. Never overwrite old/core artifacts or feed GT poses into production solving.

## Exact legacy reference, not another RRD

```
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/demos/milk_visual_contact_20260908/milk_hand_object_demo.rrd
SHA256 138dca4d9cd05866f1e56c9c7e15e762466aa908501ace35e53d1c3d0270cb10
```

Parent read back actual RRD chunks: all150object/left/right/camera samples match its `state/demo_state.npz` (SHA48f3a9da8057d5a0bfebe549598f470b0325c4162125336f52a9cb4a48677246). Its base object pose arrays exactly equal the V16 candidate cited by fit_v2 (SHA6dcc2cf3532a9b3e326b777fdd38be552a435028027479d0b7404ce4f3784108). Final visual fit only rotates V16 by median5.373°, maximum6.139°; better global rotation is substantially inherited from the upstream pose repair.

Reference evidence report (developer/evaluator only):
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/old_milk_hand_object_demo_reference/exact_rrd_source_binding.json`

Old mechanisms:
- global keyframe/multihop RGB orientation followed by robust SO(3) graph;
- late-window full SE(3): relative RGB rotations/translations, repeated observed surface correspondences, point-to-plane/point-to-point residuals, temporal terms, soft source-pose priors, one gauge anchor, outer correspondence refresh; no small total rotation cap;
- generated-shape alignment on top of that corrected pose;
- full MANO LBS, fixed beta, corrected left shapedirs/flat mean, hand articulation/root/translation, explicit camera-origin similarity, hand image projection priors, full-triangle pad proximity and temporal parameter regularization;
- legacy soft normal intrusion was an appearance heuristic, not true signed physics; any reuse must not grant sign/collision/nonpenetration authority;
- full-mesh z-buffer and RRD geometry; no display-face deletion.

Case-specific choices (window110–149, special143 metadata, anchor120, fixed150frames, hardcoded paths/IDs) must NOT become generic production rules. Reuse the numerical ideas and validated implementations, not old clip predictions or frame hacks. Official camera calibration used in old run is prediction-side metadata, not GT object pose; accept supplied calibration explicitly, otherwise preserve estimated-calibration uncertainty.

## Why current version regressed

`visual_v2_reconditioned` repaired mesh and silhouette but initialized from original P14 (136direct+14interpolated), without old global SO(3)/full-SE(3) repair. D02 still applies small ±0.08rad-per-component object corrections, so its silhouette objective cannot recover large orientation bias/material-face correspondence. Silhouette IoU and adjacent-step smoothness do not prove correct rotation. Direct P14 must remain an immutable **source observation/prior**, not a hard freeze on the later visual posterior.

## Intended independent dataflow

```
RGB video + optional target description + optional calibration + pinned runtime/assets
  -> raw frames/timestamps
  -> camera/depth/HaWoR metric MANO
  -> model-produced object plan/detection/SAM2 appearance track
  -> retained metric support and initial rigid poses (reuse tested producers)
  -> robust global RGB rotation + observed/RGB full-SE(3) temporal refinement
  -> independent appearance-anchor native SAM3D in the same canonical gauge
  -> image-supported static shape scale + bounded visual MANO/mesh fitting
  -> annotation.json + world.rrd
```

The pipeline may reuse existing lower-level scripts as libraries in its own fresh workspace. It must not require an already-completed external core run, P16–P19, or old demo assets. A native model-assisted executor may supply visual decisions as it already does; document that execution dependency honestly rather than claiming an unattended plain-Python planner that does not exist.

## Inputs and lifecycle

- One clear documented entry/preparation path for raw video, run/output root, asset config, GPU and optional calibration/target prompt. Dynamic object/case/frame IDs, real timestamps, frame count and aspect ratio. Explicitly state format constraints (e.g. CFR if inherited frontend requires it); never resample or truncate silently.
- Fresh run by default. Internal intermediates/NPZ/logs/receipts remain under a work/cache tree for reproducibility/resumption. An explicit verified resume may reuse only same-video/model/code-bound stages; do not stamp old clip artifacts as fresh.
- Core and prior published artifacts stay immutable. Historical outputs and GT/evaluation paths never enter the native production context.
- Keep production actor execution in the curated native subagent protocol, with sequential ShellGate tools, correct immutable `input/runtime_input_contract.json`, pinned Python/assets, no installations, timeout1200, and preserved named-stage failures. Native agents do not repair their environment.
- Heavy inference executes only on the authorized server/A800. Preserve core's original GPU policy; any explicit shared-idle policy belongs to this separate pipeline and must be recorded and preflighted.

## Pose mechanism and uncertainty

- Initial P14 and camera/depth data are noisy priors, not global rotation truth. Preserve them separately; refined poses are new hypothesis outputs.
- Use actual current-video RGB correspondences/PnP and retained metric surface support. Keep appearance masks distinct from metric-eligible lifting support; quarantined depth never becomes source3D merely because an appearance pixel exists.
- Robust relative-rotation/loop constraints plus full SE(3) temporal/observed consistency must actually enter the objective, not only initialization. Validate source/target pose and camera transforms; retain corrected fixed-source reprojection/Jacobian semantics.
- Generic all-frame graph or time-based overlapping windows with boundary consistency, not a fixed late-frame interval. One justified gauge anchor; time-aware priors/regularization. Explicit uncertainty for missing/occluded/weakly constrained nodes. No nearest-hold masquerading as measurement.
- Use bounded per-update trust regions, not a small total rotation cap that prevents fixing a biased baseline. Preserve the original observation field even when a posterior changes it.
- Shape/native-gauge conversion and downstream MANO fitting must consume the refined pose stage rather than silently resetting to P14/P15 via old binder readiness logic.
- Full SAM3D triangles and time-indexed MANO surfaces/parameters remain real. Generated surfaces may support unsigned visual proximity, never physical contact/collision/SDF/sign/nonpenetration authority. Do not force every finger into contact.

## JSON/RRD-only final export

- Dedicated geometry/time RRD writer; no MP4 encoder, overlay renderer, detail rasterization or V19 physical/P19 requirement in default final export.
- RRD contains complete object meshes, complete left/right MANO, per-frame object/hand state, camera transforms/intrinsics, camera trajectory and source RGB timeline as appropriate. Invalid/missing entities explicitly clear/become uncertain.
- JSON contains scene/object identity, frame/time/calibration, source-vs-refined poses, hand state/parameter provenance, visibility/occlusion/interaction hypotheses, uncertainties, and exact RRD/entity/file identity.
- Final directory contains only `annotation.json` and `world.rrd`. These two files must be usable for inspection without external runtime NPZ/mesh paths; embed render geometry in RRD and use typed entity references in JSON. Internal intermediates are not deleted or presented as additional final artifacts.
- GT only after frozen prediction, in isolated evaluation; compare rotations only after canonical/camera gauge alignment. Never claim GT-level accuracy from silhouette overlap.

## Validation and acceptance

Tests must exercise nonzero/noncontiguous IDs, non-square images, different frame counts/timestamps/objects, missing hands/poses, real active rotation factors that correct synthetic biased rotations, robust bad-edge behavior, source/target camera invariants, no hidden legacy frame-specific logic, native/current-run source binding, and no final MP4 encoding. RRD arrays/topology/time and JSON identity must read back consistently after moving the two final files away from internal artifacts.

After independent review, parent launches real current-video prediction, checks full-timeline orientation against source and exact old demo, and optionally evaluates against available GT under one explicit coordinate alignment. Any remaining orientation drift must be reported and drive the next targeted mechanism correction; do not call a smoother trajectory or clean JSON a recovered orientation.

No new standalone implementation or model execution has yet occurred at this design commit.
