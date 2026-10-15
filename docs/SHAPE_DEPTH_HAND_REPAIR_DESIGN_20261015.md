# Sequential shape / relative-depth / MANO repair study

## Scope and isolation

This is an opt-in numerical repair study of a frozen monocular observation bundle, not a new default pipeline version and not an RL training run. The raw-video/perception contract remains the existing v2 observation producer. Frozen outputs of that exact logical producer and R6 preprocessing may be reused as memoized initialization; the original files are never edited. All stages emit the full 150-frame/30fps/5-second timeline. An output becoming complete is not proof that it is more accurate.

Development worktree: `/mnt/user-home/kupingxin/ego_annotation_preprocess_shape_depth_hands`, branch `feature/milk-preprocess-shape-depth-hands-20261015`, base `6c4fbab`. Original R6 uncommitted preprocessing changes are inherited explicitly and source hashes preserved externally. No original-tree edits, commit, merge or push.

Compute: host `dexrobot-Standard-PC-Q35-ICH9-2009`, existing `ego_skrl/.venv/bin/python`, CUDA disabled, CPU numerical operations bounded to one thread. No new perception/model inference, PPO training, installation or environment provisioning. Prediction executes in a curated immutable code snapshot via a dedicated command worker; it receives only prediction inputs, stage config and a fresh output root. Development/evaluation documentation and evaluators are not in that snapshot. Evaluation/render-comparison with independent reference annotations occurs only after terminal prediction freeze, separately from prediction.

## Explicit contract amendments

1. The former uniform-scale-only template cannot resolve anisotropic source shape errors. Permit a **single shared positive full-mesh principal-axis deformation** estimated from image/depth evidence. All original vertices and faces are preserved; no category primitive, box replacement, target dimension, reflection or per-frame shape.
2. In the depth stage, the raw UniDepth measurements are immutable but are no longer declared error-free metric authority. Add a distinct, anchored **relative multiplicative depth calibration** from rigid RGB material correspondences. No per-frame additive bias. The first-frame scale remains the source gauge; absolute metric correctness remains uncertified.
3. In the hand stage, source HaWoR is an uncertain prior, not an immutable final hand state. Permit one beta vector per side plus time-indexed MANO root translation, orientation and articulation updates driven by pixel/temporal/interaction evidence. Exact licensed MANO FK must replay the source before any update. Export actual FK surfaces and reproducible parameters, not moved skeletons or fabricated detector geometry.
4. Existing source contact/occlusion declarations remain uncertain. Geometric proximity cannot manufacture contact ownership. No signed-collision or zero-penetration certification; the generated mesh is not certified physical occupancy. Contact terms operate only on image-declared digits and must allow uncertain gap, not force all hands to the object.

## State, measurements and objective

Fixed per-frame camera/K/time and source validity/visibility, raw RGB, masks and depth; source full native mesh and MANO assets. R6 supplies initialization, not independent observations.

- Shared object deformation `S=B diag(exp(q)) B^T`, with B from source full-mesh area geometry, positive Jacobian and bounded q. Each frame retains world object R,t, refined by actual triangle image/depth queries and pixel tracks.
- Relative depth: trusted bidirectional image matches, independently read source/target depth samples, rigid chord-length ratios; fit an anchored temporal log-scale curve with source-confidence/support weights, no additive depth freedom. A missing or weak observation stays missing/weak. Both calibrated cloud and calibration uncertainty are recorded separately from raw source.
- Hands: shared beta and full MANO parameters; image-space track evidence originates in source RGB and is distinguished from source-regressor projected-joint priors. Detection boxes are weak support, not keypoint truth. Use actual FK derivatives, time-indexed camera projection, actual world acceleration, conservative source proximity and documented occlusion states. No blanket backward-motion cancellation.

Normalized robust source costs: de-occlusion-aware full-triangle contour correspondences, source-point first-surface normal/depth registration, visible model-crease to image-gradient evidence, bidirectional material reprojection, bounded deformation/source pose priors, world-motion regularity. Optimize using fixed correspondence populations per outer iteration, then re-query actual triangles; check image and surface residuals separately, not only aggregate cost. Shape is fitted jointly with keyframe pose before being frozen for full-timeline pose refinement. Depth-only control fixes the native shape. Joint stage reuses the same raw population with the derived calibration.

Hand sources in this bundle contain HaWoR and detection boxes, not an independent full 2D joint detector. Source RGB LK tracks provide temporal support, not independent absolute pose truth; nearest mesh proximity is only a weak interaction hypothesis. These limitations must remain visible, including when the result is worse.

## Ordered interventions and discriminating predictions

1. **shape_fixed_depth**: shape + object pose only; raw depth/camera/MANO unchanged. If shape evidence couples correctly, held-out contour/face consistency should improve and different principal dimensions can change independently. If shape changes only to follow biased depth or misassigned contours, independent geometric quality may deteriorate; do not promote the result, expose which source family drove it and correct that coupling.
2. **depth_only_control**: fixed R6 shape, anchored relative-depth calibration + object pose. Detect whether rigid image correspondences actually support a scale trend. A negligible curve despite source depth inconsistency means insufficient/mispaired material support, not that depth is correct. Preserve failed correspondence evidence and improve the measurement rather than manufacture a trend.
3. **shape_depth**: same shape mechanism + calibrated depth, then full pose. If the two mechanisms conflict, inspect per-surface image/depth residuals and metric gauge. Do not pick outputs by independent-reference scores or insert reference-derived corrections.
4. **shape_depth_hands**: initialize from preceding frozen state, replay/fix beta constancy and optimize full MANO state with image/temporal/declared interaction support; keep real backward/forward movement where observed. If independent depth or articulation evidence is weak, report the uncertainty instead of claiming a centimeter guarantee.

The user requested sequential repair and visual review. Stage implementations run and are inspected before the next mechanism is promoted; unsuccessful stage outputs remain available for controlled comparison, but do not automatically replace baseline. A combined mechanism may consume a weak-but-usable earlier state with explicit uncertainty, never silently claim the failure resolved.

## Deliverables and review

Every stage: complete coarse_scene NPZ/manifest, calibration/shape/MANO mechanism trace, source/model/code hashes, uncertainty/missingness, complete source-RGB overlays and world full-mesh video. Separate evaluator: dimensions, same-time camera-space depth errors, world joint/surface error, motion and projection curves; comparison always includes frozen R6. Reference data are not producer inputs and cannot set weights, target sizes, correction curves or frame-specific fixes. Decode complete videos, read representative entire meshes and inspect early/middle/late frames. Preserve regressions explicitly. No RL run until a new source state is demonstrably better and separately approved for its downstream contract.

Bounded initial configuration (not reference-tuned): 480-pixel source evidence / 160-pixel triangle scoring; 10 temporally spread shape keyframes, up to 4 correspondence updates and 24 numerical steps per fit; full-timeline sparse pose max24 steps per outer pass; shape principal log-length prior0.12 with bounds+-0.25; depth anchor fixed, log-scale prior0.20, support-weighted trend. Source residuals carry observation-scale uncertainty, not arbitrary physical-success thresholds. Runtime cost and solver termination are reported; a slow offline study is not advertised as a near-realtime default.
