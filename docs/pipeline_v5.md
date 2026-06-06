# Pipeline V5: Visibility-Aware Dynamic Object Mesh

## Starting Evidence

V4 has an accepted bounded result for wild-rice frames 2520 to 2550:

- per-frame measured/completed active-stem mesh archive with frame 2550 completed by SAM2 mask-seed propagation;
- overlay, 3D world animation, and side-by-side videos;
- z-buffer median silhouette IoU 0.964 and median z-buffer absolute p95 3.42 mm over 31 frames;
- 15 reliable temporal hand-object contact rows;
- selected-contact penetration 0 percent and full-hand penetration 0 percent under 1 mm SDF QC.

V4 also falsified a rigid canonical object map on the same frames. ICP/BPA fusion of the completed per-frame meshes produced an open canonical surface spanning 0.239 x 0.449 x 0.309 m. Replaying that single map through per-frame poses gave median silhouette IoU 0.373 and median z-buffer p95 80.1 mm. Visual QC showed a broad merged surface covering multiple stem/sheath regions. The cause is object-state change under peeling plus partial visibility near parallel plant material.

## Research Readout

Target-aligned hand-object reconstruction work exists, but direct drop-in code is limited:

- ForeHOI matches the daily hand-object video problem and combines 2D mask inpainting with 3D object completion. Its public repository currently lists inference and training code as unreleased.
- AGILE targets watertight object reconstruction with contact-aware tracking, but its repository currently says full code will be available by August 2026.
- HOLD is released and reconstructs category-agnostic hand/object surfaces from monocular video without object templates. It is a viable baseline/integration candidate, though its neural-field workflow is heavier than the accepted V4 measured-sheet evidence path.
- FoundationPose and RGBTrack are useful after a real mesh prior exists; they are pose trackers, not object mesh reconstructors.
- Mesh4D and ViSER support the representation direction of dynamic/deformable surfaces, but they are not immediate replacements for this project's full annotation contract.

V5 should therefore implement a project-owned dynamic object-surface graph while testing released baselines where they can be made to run.

## Representation

For each object track, V5 stores:

- per-frame measured visible surface mesh from mask, metric depth, camera pose, and intrinsics;
- per-frame source label: measured, model-propagated completion, rejected, or ambiguous;
- canonical dynamic surface coordinates for object material points when tracking is reliable;
- per-frame deformation field from canonical surface to observed frame;
- per-region visibility and measurement confidence;
- optional full-mesh prior from an image-to-3D or HOI reconstruction model;
- MANO vertices, keypoints, hand confidence, and contact patches;
- full-hand nonpenetration state and active-contact equality state.

The core state is a mesh or neural surface with visibility weights, deformable surface support, and explicit full-mesh uncertainty.

## Objective

The V5 graph minimizes robust residuals:

- measured visible-surface residual: observed per-frame mesh vertices must lie on the deformed object surface where visibility confidence is high;
- mask and z-buffer residual: rendered deformed surface must match the accepted mask and metric depth;
- temporal deformation residual: neighboring frames prefer low acceleration and local as-rigid-as-possible deformation;
- correspondence residual: model/track evidence links surface regions across frames when confidence is high;
- full-hand nonpenetration residual: non-contact MANO surface samples stay outside the object SDF;
- active-contact residual: contact-supported MANO patches lie on the object surface with near-zero signed distance;
- full-shape prior residual: generated or learned object mesh priors constrain unobserved regions with lower weight than measured depth/mask evidence.

The solver must expose residuals by frame and by source. A generated full mesh that fails z-buffer, contact, or nonpenetration QC is evidence of a bad prior, not a substitute object annotation.

## Implementation Order

1. Build a V5 dynamic-surface diagnostic on top of V4 accepted meshes.
   - Input: completed V4 mesh archive, manifest, annotations, contact rows, and SDF reports.
   - Output: per-frame local surface descriptors, visibility masks, temporal correspondence candidates, and a report explaining where a dynamic map is observable.
   - Acceptance: the diagnostic must identify stable surface regions and reject ambiguous merged surfaces such as frame 2539.

2. Add a deformable surface fit on a short high-quality subwindow.
   - Start with frames 2531 to 2537, where V3/V4 contact and z-buffer evidence are strongest.
   - Optimize per-frame deformations with ARAP and z-buffer residuals.
   - Acceptance: replayed mesh must preserve per-frame z-buffer p95 near the measured-sheet baseline and keep full-hand penetration at 0 percent.

3. Integrate a released full-object prior path.
   - First baseline: test HOLD on a cropped representative sequence if dependencies are runnable.
   - Second baseline: use an image-to-3D mesh prior from selected clean keyframes, then fit it to V4 measured surfaces and contact residuals.
   - Acceptance: generated prior must improve unobserved/full-shape plausibility without degrading measured-frame z-buffer/contact/SDF QC.

4. Upgrade the presentation renderer.
   - Replace diagnostic line-plot styling with a stakeholder 3D scene: fixed camera choreography, shaded MANO meshes, shaded object mesh, head-camera frustum, contact close-up inset, and readable semantic captions.
   - Acceptance: side-by-side and standalone 3D must immediately communicate head pose, two hands, manipulated object mesh, and contact state.

## Current V5 Binary State

V5 is open. The rigid canonical-map branch has been falsified. The next concrete implementation target is the dynamic-surface observability diagnostic, followed by a deformable short-window fit only if the diagnostic finds stable surface regions.

## Dynamic-Surface Observability Result

The first V5 diagnostic measured which frames in the completed V4 archive are observable enough for dynamic surface fitting. A frame must pass three tests: accepted z-buffer residuals, PCA extent consistency against the track median, and stable nearest-neighbor surface overlap with at least one adjacent frame after local ICP.

The report marks 13 observable frames: 2522, 2523, 2525 to 2529, 2531, 2532, 2542, 2545, 2549, and 2550. The only contiguous window of length at least four is 2525 to 2529. This window has median z-buffer p95 1.71 mm, stable neighbor support for every frame, and one reliable contact row.

The contact-rich 2533 to 2537 interval is rejected for dynamic-map anchoring despite good per-frame z-buffer residuals. Frames 2534 to 2537 are extent outliers, and frame 2533 lacks stable temporal overlap. This matches the visual failure mode: the selected measured masks include parallel stem/sheath surfaces, so the frames are contact evidence with ambiguity, not a stable object-map anchor.

Diagnostic artifact:

- `/data2/ego_annotation_outputs/representative_wild_rice/v5_dynamic_surface_observability_completed_2520_2550/qc_dynamic_surface_observability_v5.json`

The next V5 implementation should fit the first dynamic surface on frames 2525 to 2529 as a geometry-first window. Contact-rich ambiguous frames should enter later with uncertainty weights or segmentation repair.
