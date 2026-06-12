# Pipeline V18: Real-Time-Scale Occlusion-Aware HOI Annotation

## Status

V18 is open as a redesign after formal V17 failure. Current V18 implementation has a bounded fixed-pass status deliverable with full-duration 2D overlay, abstract world/status, side-by-side videos, and visible-surface geometry evidence, but it is not a final pose-complete annotation pipeline. Any accepted V18 implementation must preserve this design and obey the runtime and occlusion constraints below.

V17 failed as a pipeline design, not merely as an unfinished run:

1. Runtime violated the product constraint. A path that can spend tens of hours on a roughly one-minute clip is unacceptable; pipeline wall time must be the same order of magnitude as input duration.
2. BundleSDF was the wrong default object mechanism. It performs per-instance/test-time neural optimization for a rigid canonical mesh plus per-frame SE(3). Using GPU-hours to discover whether a manipulated object is rigid is bad methodology; physical state type must be inferred cheaply from model-produced semantics and fast residual checks.
3. Occlusion was not a first-class state. V17 mostly marked rows as unobserved, repaired selected hand-depth measurements, or used temporal candidates; it did not maintain explicit hand/object visibility, occluder ownership, depth ordering, uncertainty, and bounded infill through occlusion.
4. V17 over-optimized evidence ledgers and component diagnostics instead of delivering an efficient full-video annotation path.
5. V17 produced useful evidence, especially depth-edge ownership for MANO/depth measurement validity, but it did not deliver an acceptable V17 annotation result.

All V17 readiness flags remain false: `v3_solver_complete=false`, `annotation_ready=false`, `deliverable_ready=false`, `accuracy_target_met=false`, `object_geometry_complete=false`, `object_pose_requirement_met=false`, and `rigid_pose_requirement_met=false`.

## Implementation Checkpoint 1: Runtime And Visibility Scaffold

V18 now has the first bounded scaffold artifacts, generated without heavy perception or reconstruction:

```text
/data2/ego_annotation_outputs/v18_runtime_manifest/
/data2/ego_annotation_outputs/v18_visibility_occlusion_state/
```

`build_v18_runtime_manifest.py` writes the default V18 DAG and hard budget contract before any heavy stage runs. The planned default critical path is 7.75x real time, under the initial 10x hard ceiling: 271.25 seconds for `trash_1050` (35.0 s raw video) and 248.0 seconds for `task5_tomato_960` (32.0 s raw video). The default DAG contains no BundleSDF, NeRF, neural-field training, or all-face CPU raster stage. This is a plan/budget gate, not proof that later implementation will meet the budget.

`build_v18_visibility_occlusion_state.py` writes full-timeline hand and object visibility rows from existing fast evidence. It does not infer certain poses through occlusion. Unobserved hands are marked unresolved, with possible short occlusion recorded only as an unowned hypothesis when bounded detector gaps overlap visible active objects. Object geometry is scoped as visible depth-backed surface, visible mask with rejected surface, or no visible geometry; every object row keeps `object_geometry_complete=false` and `object_pose_requirement_met=false`. Physical state types are recovered from model-produced VLM physical notes rather than object-name branches. Current counts:

- `trash_1050`: 2,100 hand-state rows with 1,593 visible, 8 partially visible, and 499 unresolved; 4,200 object-state rows with 1,604 visible, 29 unresolved active-mask gaps, and 2,567 out-of-frame/inactive rows; object geometry scopes are 1,417 visible depth-backed surfaces, 187 visible masks with rejected surfaces, and 2,596 no-visible-geometry rows. Model physical states: 2 deformable, 1 articulated, 1 rigid.
- `task5_tomato_960`: 1,920 hand-state rows with 1,722 visible, 11 partially visible, and 187 unresolved; 8,640 object-state rows with 1,109 visible, 40 unresolved active-mask gaps, and 7,491 out-of-frame/inactive rows; object geometry scopes are 694 visible depth-backed surfaces, 415 visible masks with rejected surfaces, and 7,531 no-visible-geometry rows. Model physical states: 5 rigid, 2 deformable, 1 articulated, 1 unknown.

Readiness remains false. The next V18 step is fast object-surface/motion state and a bounded consistency graph; the scaffold only makes runtime and occlusion state explicit.

## Implementation Checkpoint 2: Fast Object Motion State

V18 now writes a cheap object surface/motion reducer:

```text
/data2/ego_annotation_outputs/v18_fast_motion_state/
```

`build_v18_fast_motion_state.py` consumes the V18 visibility/occlusion state plus existing V17 visible-surface/material-track/surface-replay evidence. It runs in under a second on the representative cases and does not run BundleSDF, NeRF, or any new reconstruction backend. The reducer preserves the distinction between visible-surface motion evidence and complete object pose.

Current fast motion-state counts across both cases: 1 partial rigid visible-surface motion support, 1 local rigid-motion-only-not-pose, 3 deformable visible-surface/surface-motion states, 1 deformable unresolved/no-surface state, 2 articulated visible-surface unresolved states, 1 visible-surface-only motion unresolved state, and 4 motion-unresolved/no-surface states. Per-case observations:

- `trash_1050`: black trash bag and white trash bag are deformable visible-surface states; off-white can is articulated/visible-surface unresolved; pink-lid can is the only partial rigid visible-surface motion-supported object, with 44 rigid-ready material pairs and 2 visible-surface replay-ready partial segments.
- `task5_tomato_960`: faucet handle is articulated/visible-surface unresolved; tomato has local rigid motion only, not pose; tomato peel is deformable visible-surface/motion; several context objects have no usable surface/motion evidence and stay unresolved.

This confirms the V18 design premise: fast model-produced physical state plus cheap residual summaries can prevent wasting GPU-hours on rigid reconstruction for deformable or unresolved objects. It still does not close object geometry, contact ownership, full consistency optimization, or rendering.

## Implementation Checkpoint 3: Consistency/Contact Scaffold

V18 now writes a bounded consistency/contact reducer:

```text
/data2/ego_annotation_outputs/v18_consistency_graph/
```

`build_v18_consistency_graph.py` joins the V18 visibility/occlusion state, V18 fast motion state, V17 pairwise image contact, and V17 pairwise metric depth-gap evidence. It does not run a nonlinear optimizer and does not fill occluded poses. It exposes blocker classes that the future bounded optimizer must address.

Across the two representative cases, the reducer materializes 5,564 hand-object pair rows: 619 image-contact candidates are rejected by metric-depth contradiction, 1,490 are image-overlap-only, 3,107 have no contact image evidence, and 348 are unobserved pairs. Contact-factor-ready rows remain zero. Blockers include 619 metric-depth contradictions, 619 incomplete-object-geometry rows, 679 hand-visibility-unresolved rows, 138 object-visibility-unresolved rows, and 1,490 image-overlap-is-not-contact rows.

Per-case counts:

- `trash_1050`: 97 image-contact candidates rejected by metric depth, 977 image-overlap-only rows, 1,940 no-contact-image rows, and 252 unobserved pairs.
- `task5_tomato_960`: 522 image-contact candidates rejected by metric depth, 513 image-overlap-only rows, 1,167 no-contact-image rows, and 96 unobserved pairs.

This preserves the V17 lesson in V18 form: projected hand/object overlap is not physical contact. The next implementation step is a bounded optimizer or state update that uses interior-owned hand depth, visible object surfaces, occlusion ownership, and fast motion state together; it must still keep unresolved states explicit when metric depth or geometry do not support contact.

## Implementation Checkpoint 4: Renderable Annotation State And Status Overlay

V18 now writes a full-timeline renderable annotation state and full-duration status overlay videos:

```text
/data2/ego_annotation_outputs/v18_annotation_state/
/data2/ego_annotation_outputs/v18_renders/
```

`build_v18_annotation_state.py` joins V18 visibility/occlusion rows, fast motion state, consistency/contact rows, V17 timeline boxes/masks, and V16 raw-frame manifests into one per-frame state. Both representative cases match the raw frame count: `trash_1050` has 1,050 state frames and `task5_tomato_960` has 960 state frames. The joined state has 3,334 renderable hand-box rows and 2,713 renderable object-mask rows across both cases. It keeps `annotation_ready=false`, `object_geometry_complete=false`, and `object_pose_requirement_met=false`.

`render_v18_status_overlay.py` renders full-duration MP4 status overlays from raw frames and verified object masks. The renderer completed both cases in 61.55 seconds total and the QC manifests report exact frame-count matches:

- `trash_1050`: `/data2/ego_annotation_outputs/v18_renders/trash_1050/v18_status_overlay.mp4`, 1,050/1,050 frames, 1,601 hand boxes, 1,604 object masks, 499 unresolved-hand labels, and 29 unresolved-object labels.
- `task5_tomato_960`: `/data2/ego_annotation_outputs/v18_renders/task5_tomato_960/v18_status_overlay.mp4`, 960/960 frames, 1,733 hand boxes, 1,109 object masks, 187 unresolved-hand labels, and 40 unresolved-object labels.

These videos are a real V18 status output, not a final pose-complete annotation. They satisfy the full-duration/same-frame-count render constraint for the 2D status overlay, and they make the missing geometry/contact state visible instead of hiding it. The remaining V18 gap is the actual bounded optimizer/geometry path and full 3D/side-by-side outputs.

## Implementation Checkpoint 5: Bounded State Solution, World/Status, And Side-By-Side Status Deliverable

V18 now writes a bounded fixed-pass state solution and a status-deliverable manifest:

```text
/data2/ego_annotation_outputs/v18_bounded_state_solution/
/data2/ego_annotation_outputs/v18_status_deliverable_manifest/
```

`build_v18_bounded_state_solution.py` classifies hand observation gaps, object geometry scope, and contact modes without filling poses through occlusion or promoting image overlap to physical contact. Across both representative cases it reports:

- Hand states: 1,257 observed depth-consistent, 2,058 observed depth-unchecked, 19 partial observations, 113 short-gap possible occlusion candidates left unfilled, 25 short gaps with no visible occluder evidence, and 548 long/open unresolved gaps.
- Object states: 2,111 visible-surface-only rows with hidden geometry unresolved, 602 visible-mask-only/surface-rejected rows, 69 active-object visibility-unresolved rows, and 10,058 inactive/out-of-frame rows.
- Contact states: 619 rejected under current metric-depth evidence, 1,490 image-overlap-only/near rows, 3,107 no-contact-image rows, and 348 unobserved pairs. Contact-factor-ready rows remain zero. Pose-filled-through-occlusion rows remain zero.

V18 also now renders full-duration abstract world/status and side-by-side status videos:

- `trash_1050`: `/data2/ego_annotation_outputs/v18_renders/trash_1050/v18_world_status.mp4` and `/data2/ego_annotation_outputs/v18_renders/trash_1050/v18_status_side_by_side.mp4`, both 1,050/1,050 frames.
- `task5_tomato_960`: `/data2/ego_annotation_outputs/v18_renders/task5_tomato_960/v18_world_status.mp4` and `/data2/ego_annotation_outputs/v18_renders/task5_tomato_960/v18_status_side_by_side.mp4`, both 960/960 frames.

The world/status render is deliberately image-normalized abstract status geometry, not a metric 3D reconstruction. The side-by-side videos place the raw-frame status overlay next to that abstract status view. Visual sheets were extracted for both cases for non-corruption checks.

`build_v18_status_deliverable_manifest.py` writes `/data2/ego_annotation_outputs/v18_status_deliverable_manifest/v18_status_deliverable_manifest.json`. The manifest marks `status_deliverable_ready=true` and `final_pose_complete_deliverable_ready=false`. Measured render time for the status outputs is 123.92 seconds for 67 seconds of source video, or 1.85x real time, under the 10x V18 status-render budget. This closes a V18 status deliverable, not the final geometry/pose/contact deliverable.

Remaining gap after this checkpoint: implement or integrate a bounded object geometry/pose path that can actually reconstruct manipulated-object geometry where evidence supports it, validate contact ownership with metric depth and complete/appropriate geometry, and only then upgrade final annotation readiness.

## Implementation Checkpoint 6: Visible-Surface Geometry Archive

V18 now stages actual depth-backed visible-surface geometry evidence:

```text
/data2/ego_annotation_outputs/v18_visible_geometry_archive/
```

`build_v18_visible_geometry_archive.py` validates and copies the compact visible-surface NPZ archives into V18, then writes per-case reports with per-frame surface offsets, vertex/face counts, bounded object state, and explicit geometry claims. It preserves the V17 visible-surface measurement evidence as V18 geometry evidence, but it does not reconstruct hidden geometry, canonical meshes, or complete object pose.

Across both representative cases the archive contains 2,111 accepted depth-backed visible-surface frame rows, 602 rejected visible-mask rows, 1,212,570 vertices, and 2,107,754 faces. Object-level status counts are: 1 partial rigid visible-surface archive ready but not complete pose, 7 visible-surface archives with hidden geometry unresolved, and 5 visible masks without accepted surface. Geometry claims are 8 depth-backed visible-surface-only objects and 5 objects with no accepted visible surface.

The status manifest now points to these archives and reports `visible_geometry_archive_ready=true`, while keeping `hidden_geometry_reconstructed=false`, `canonical_mesh_ready=false`, `complete_object_pose_ready=false`, `object_geometry_complete=false`, and `object_pose_requirement_met=false`. This is the first V18 object-geometry artifact, scoped to observed visible surfaces only.

Remaining gap after this checkpoint: a bounded method that completes manipulated-object geometry where warranted, estimates object pose only under that geometry support, and validates contact ownership.

## Design Goal

Build a full raw-video hand-object interaction annotation pipeline whose default path is fast, occlusion-aware, and honest about unresolved geometry. The direction remains hand detection + object detection + consistency optimization, but every required stage must have bounded cost and every inferred state must carry visibility and uncertainty.

The pipeline must output full-duration videos with the same frame count and duration as the original raw video:

1. Annotated raw-video overlay with hands, object instances, depth-backed object surfaces or unresolved-object markers, contact/occlusion status, uncertainty, and semantic captions.
2. Full-duration 3D world animation with head camera, MANO hands, object visible/deformable surfaces, contact hypotheses, occlusion ghosts, and uncertainty.
3. Full-duration side-by-side presentation of annotated video and 3D reconstruction with captions and explicit status flags.

A V18 result is not allowed to call object pose complete unless manipulated-object geometry is actually reconstructed. When only visible depth-backed surface exists, the artifact must say `visible_surface_only`, `hidden_geometry_unresolved`, and keep `object_geometry_complete=false`. This is an honest pipeline output, not closure of the object-pose requirement.

## Runtime Contract

Runtime is a design invariant, not an afterthought.

- Target end-to-end wall time: same order of magnitude as video duration.
- Initial V18 hard ceiling for representative clips: no more than 10x real time end-to-end on available workstation/A800 resources. A 60-second clip should finish in minutes, not hours.
- Stretch target after the first working prototype: 1-3x real time for a 60-second clip.
- Any required stage whose expected runtime exceeds the hard ceiling is not in the default pipeline.
- Per-instance neural-field optimization, NeRF/SDF training loops, BundleSDF-style optimization, and all-face CPU raster sweeps are disallowed in the default path. They may exist only as offline research branches that cannot block delivery or be used to discover obvious physical state types.
- No foreground sleep/poll loops are allowed for progress waiting. Long work must run in tmux or a job system with sentinel files/logs while the agent does useful parallel work or returns control.

Every run writes a runtime manifest with per-stage wall time, device, frame count, object-frame count, and whether each stage stayed within budget.

## Raw-Video Input Contract

For each case the pipeline consumes:

- Raw RGB video or source-resolution frame manifest.
- Frame count, fps, timestamps, and source image dimensions.
- Camera intrinsics if known; otherwise an explicit estimated-intrinsics state with uncertainty.
- Optional existing head/camera trajectory; if absent, a fast visual-inertial/visual-only trajectory estimate with uncertainty.
- Optional metric depth archive; if absent, a fast monocular metric-depth pass with cached per-frame intrinsics/scale confidence.
- Representative V18 cases: `trash_1050` and `task5_tomato_960` from `/data2/egoscale_demo_30h/` and the existing V16/V17 artifacts only as reusable evidence, not as accepted annotations.

## Full-Timeline State Variables

V18 state is per frame unless otherwise noted.

### Camera and Depth

- `T_world_camera[t]` with uncertainty.
- `K[t]` or selected camera-intrinsics model with uncertainty.
- Metric depth map `D[t]` with confidence, depth-edge bands, and invalid/low-confidence masks.

### Hands

For each hand side/track:

- Existence state: `visible`, `partially_visible`, `occluded`, `out_of_frame`, or `unresolved`.
- 2D keypoints, boxes, and hand masks with source/confidence.
- MANO pose/shape/translation when observed or inferred.
- Camera-depth alignment variables and uncertainty.
- Temporal track identity and infill source.
- Occluder owner when hidden by an object or another hand.

### Objects

For each model-produced object id:

- Semantic label/caption and physical state type proposed by VLM/LLM/perception: `rigid`, `deformable`, `articulated`, `container/context`, `fluid/loose`, or `unknown`.
- Existence/visibility state: `visible`, `partially_visible`, `occluded`, `out_of_frame`, or `unresolved`.
- Open-vocabulary detection boxes, SAM2 masks/tracks, mask confidence, and identity continuity.
- Depth-backed visible surface mesh/surfels for observed pixels.
- Material/feature tracks on visible surfaces.
- Rigid SE(3) only when cheap motion residuals support a rigid state.
- Deformation field or per-frame surfel motion for non-rigid states.
- Hidden/completed geometry state only if produced by a bounded feed-forward prior or directly supported by observations; otherwise explicitly unresolved.

### Contact and Occlusion

- Pairwise hand-object contact candidates and owner variables.
- Contact mode per hand-object pair: `contact`, `near`, `not_contact`, `occluded_contact_possible`, or `unresolved`.
- Occlusion ownership graph with depth-order evidence: what hides what, and for how long.
- Uncertainty/covariance for every inferred-through-occlusion hand/object state.

## Perception Sources

Differences between cases must come from model outputs, not hand-written visual case branches.

Default sources:

- VLM/LLM object-plan pass for object roster, active intervals, physical state type, and likely occlusion/contact semantics.
- Open-vocabulary detector or grounding model for object boxes.
- SAM2 or equivalent video segmentation/tracking for object masks and identity continuity.
- RTMLib-style 2D hand keypoints and boxes.
- HaMeR/WiLoR-style per-frame MANO measurements where visible.
- HaWoR or another temporal hand-motion model only after a validation contract proves infill behavior; until then it is a candidate temporal measurement, not an accepted occlusion solution.
- Fast metric depth and depth-edge ownership masks.
- Optical-flow/material point trackers for object surface motion and occlusion consistency.

Disallowed default sources:

- BundleSDF/NeRF/test-time neural reconstruction loops.
- Category-specific if/else code for bags, cans, tomatoes, peels, faucets, colors, or action phrases.
- Silent fallback from missing geometry to centroids, spheres, boxes, patches, or masks as if object pose were solved.

## Fast Object Geometry Path

V18 object geometry is state-aware and bounded-cost.

1. Always build a per-frame visible surface from mask + metric depth when available.
2. Track surface points/features across adjacent frames with bounded optical flow/material tracking.
3. Use VLM physical state type as a prior, then revise it with cheap residuals:
   - rigid hypothesis: adjacent-frame Procrustes/ICP residuals, surface-track consistency, scale stability;
   - deformable hypothesis: local surface tracks explain motion but no single SE(3) explains the object;
   - occluded/unresolved hypothesis: insufficient visible surface or depth.
4. For rigid-supported objects, build a lightweight surfel/TSDF-style canonical visible geometry and per-frame pose under a strict iteration/time cap.
5. For deformable objects, maintain per-frame visible geometry plus surface motion/deformation variables. Do not force a canonical rigid pose.
6. For hidden geometry, use only bounded feed-forward priors or observed multi-view fusion. If hidden geometry is not supported, mark it unresolved and render uncertainty.

This path is allowed to produce useful partial geometry quickly. It is not allowed to claim complete object pose when the object geometry is only visible surface.

## Occlusion Model

Occlusion is a first-class inference target.

For each hand/object track, V18 estimates a visibility state and occluder owner. The model uses:

- Detector confidence drop or mask disappearance.
- Depth ordering: foreground object/hand depth relative to the predicted hidden track.
- Boundary/depth-edge ownership masks.
- Before/after temporal continuity.
- Contact persistence when supported by pre/post evidence.
- Nonpenetration constraints.
- Maximum occlusion duration thresholds; long occlusions become unresolved, not hallucinated exact pose.

Short occlusion infill is permitted only with uncertainty. Rendered hidden states must be visually distinct from observed states, e.g. translucent/ghosted MANO or object surfaces with covariance bands and labels.

## Optimization Objective

The V18 optimizer is a bounded robust factor graph or equivalent fixed-iteration smoother. It must run under the runtime contract.

Variables:

- Camera/depth scale/intrinsics adjustments within small priors.
- Hand pose/translation/depth alignment and visibility state.
- Object visible-surface state, rigid pose or deformation state, and visibility state.
- Contact ownership/mode and occlusion ownership.

Factors:

- 2D hand keypoint/box/mask reprojection.
- MANO depth against interior-owned metric depth, excluding depth-edge bands.
- Object mask/depth visible-surface residuals.
- Surface-track temporal consistency.
- Rigid/deformable state residuals from cheap motion tests.
- Occlusion depth ordering and visibility transition costs.
- Contact proximity, contact persistence, and nonpenetration.
- Temporal smoothness with robust losses and switch variables.

The optimizer must expose unresolved rows and rejected factors. It must not hide broken contracts with silent fallbacks.

## Pipeline DAG and Parallelism

V18 is parallel by construction:

1. Decode/cache frames.
2. In parallel: hand detection, object planning/detection, depth estimation, camera trajectory, captions.
3. In parallel per object: SAM2 tracking, visible surface extraction, material/flow tracking, cheap rigidity/deformation test.
4. In parallel per hand track: MANO measurement assembly and visibility/occlusion evidence.
5. Single-writer reducers assemble measurement tables.
6. Bounded optimizer runs once per case or in small temporal chunks with explicit overlap/reducer semantics.
7. Full-duration render runs by frame chunks and concatenates with frame-count verification.

Every parallel worker writes isolated artifacts. Reducers are the only writers to summary JSONs.

## Acceptance Checks

A V18 run is acceptable as a pipeline artifact only if:

1. Full-duration outputs match the raw video frame count, fps, and duration.
2. Runtime manifest satisfies the hard runtime ceiling, or the run is explicitly marked failed for runtime.
3. No default stage uses BundleSDF/NeRF/test-time neural reconstruction.
4. Hand/object visibility states are present for every frame/track.
5. Occluded intervals are labeled with occluder owner or unresolved status; inferred poses through occlusion carry uncertainty.
6. Object geometry claims are scoped: visible-surface, rigid visible canonical, deformable surface, feed-forward completed, or unresolved.
7. Contact labels are tied to a hand side, object id, geometry source, and visibility/depth evidence.
8. Renders distinguish observed from inferred/occluded/unresolved states.
9. Readiness flags remain false unless object geometry, contact ownership, and visual QC actually support them.
10. Representative visual review covers named V16/V17 failure frames and any new occlusion intervals.

## Expected Failure Modes

- Long or complete hand occlusion without reliable before/after state: output unresolved, not hallucinated.
- Deformable or transparent objects with poor depth: visible-surface uncertainty grows; no rigid pose claim.
- SAM2 identity switches under occlusion: object identity marked ambiguous until re-associated.
- Metric depth failure near hand/object edges: interior/depth-edge ownership masks prevent false residual claims.
- Camera/world scale drift: report trajectory uncertainty and avoid metric-contact claims that depend on unvalidated scale.
- Fast model failures: if feed-forward hand/object detectors miss a visible state, V18 may fail the case quickly rather than launch expensive reconstruction.

## Deliverable Layout

Default output roots:

```text
/data2/ego_annotation_outputs/v18_runtime_manifest/
/data2/ego_annotation_outputs/v18_measurements/
/data2/ego_annotation_outputs/v18_visibility_occlusion_state/
/data2/ego_annotation_outputs/v18_object_surfaces/
/data2/ego_annotation_outputs/v18_fast_motion_state/
/data2/ego_annotation_outputs/v18_consistency_graph/
/data2/ego_annotation_outputs/v18_full_state/
/data2/ego_annotation_outputs/v18_renders/
```

Required videos per case:

```text
v18_overlay_mano_object_occlusion.mp4
v18_reconstruction_3d_world_occlusion.mp4
v18_side_by_side_occlusion.mp4
```

Every summary JSON must include runtime, frame-count equality, readiness flags, unresolved-state counts, and evidence-source counts.

## Immediate Implementation Order

1. Build the V18 runtime/DAG manifest and hard-stop budget checks before adding perception logic.
2. Build full-timeline visibility/occlusion schema for hands and objects.
3. Assemble fast measurements from existing frame/depth/hand/object artifacts where available; do not run BundleSDF.
4. Implement visible object surface extraction + cheap motion-state residuals under per-object time caps.
5. Implement bounded hand/object/contact/occlusion consistency graph.
6. Render full-duration outputs with uncertainty/occlusion status.
7. Evaluate on `trash_1050` and `task5_tomato_960`; if runtime or occlusion handling fails, mark V18 failed quickly and preserve the causal evidence.
