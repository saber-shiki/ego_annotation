# Pipeline V18: Real-Time-Scale Occlusion-Aware HOI Annotation

## Status

V18 is open as a redesign after formal V17 failure. No V18 implementation is accepted until this design is preserved and the first implementation obeys the runtime and occlusion constraints below.

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
