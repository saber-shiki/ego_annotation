# Pipeline V17: Measurement-Grounded Multi-Object HOI Annotation

## Status

V16 is closed only as the first full raw-video delivery. It produced full-length videos for two raw clips, but the annotations do not meet the quality requirement. V17 treats every detector output as a measurement with residuals, confidence, and source evidence before the solver can accept an annotation state.

V17 implementation has started with the measurement store. The first implementation slice reads V16 full-video outputs and prior HaWoR/WiLoR artifacts as measurements, then emits anchor QC that exposes the known V16 failure frames before any graph solver can accept or repair them.

The current measurement-store implementation is the evidence layer for the full V17 solver. Model outputs remain traceable measurements with confidence, residual, source, and failure fields; missing hands, missing objects, missing contact states, and incomplete HaWoR/WiLoR coverage become explicit QC failures.

The current HaWoR evidence path uses a compact full-video adapter input generated from V16 annotations. The compact file preserves frame indices, timestamps, source camera transforms, source intrinsics, measured V16 hand 2D keypoints, detector scores, and hand boxes, then reruns the HaWoR camera-local adapter against the full 0-1049 HaWoR NPZ. The adapter input therefore contains only the fields read by the HaWoR residual calculation.

## V16 Failure Analysis

The failures are not isolated rendering bugs.

Trash frame 0182:
- The right hand is visible in the raw image, but the delivered MANO state drifts away from it.
- The object mesh covers disconnected or incomplete bag/can regions.
- The final annotation record contains hand boxes but no retained confidence, source, or residual fields, so a bad hand measurement looks like an accepted hand state.

Trash frame 0260:
- Both rendered hands are attached to the wrong image region near the frame edge.
- This is a single-frame hand hypothesis failure under truncation and occlusion.
- The temporal state did not reject the hypothesis against mask, keypoint, depth, or motion evidence.

Trash frame 0764:
- The visible hand and bag are in contact, but the V16 manipulation view reports them as separated.
- Nearest hand-mesh distance is an insufficient contact model. Contact must be a latent state supported by image adjacency, depth ordering, 3D gap, surface motion, and temporal consistency.

Trash frame 0856:
- The object mask/mesh and both MANO hands are visibly wrong.
- The contact label becomes numerically small even though the underlying hand/object states are bad. A contact residual cannot rescue invalid state estimates.

Trash frame 0949:
- The hand annotation is wrong even without requiring object reasoning.
- V16 lacks an independent hand-state rejection path.

Trash frame 0970:
- Both hands are visible in the raw frame, but the delivered hand list is empty.
- V16 drops the state when measurements disappear; it does not maintain a prediction/update filter with uncertainty through occlusion or detector failure.

Tomato frames around 0480 to 0760:
- The tomato mesh changes shape frame to frame because V16 reconstructs a visible depth patch each frame.
- The tomato requires a persistent near-rigid canonical shape with per-frame pose and small allowed deformation.
- Bowls, plates, tray, sink, and containers can participate in manipulation context, but V16 only carries one object stream.

World reconstruction:
- Skeleton-only hands do not visually read as hands.
- The camera frustum and trajectory do not explain the camera-hand-object relation.
- The 3D panel looks like a diagnostic plot. V17 must render shaded MANO surfaces, shaded object meshes, the current camera image plane/frustum, a local manipulation close-up, and a stable world/camera relationship so the view reads as a reconstruction.

## Root Causes

1. V16 promoted measurements into annotations.

WiLoR hands, object masks, and depth-derived meshes enter the delivered timeline without a strong measurement residual contract. The final JSON often lacks score/source/residual fields for active hand states. The renderer cannot distinguish accepted state, low-confidence measurement, prediction, or rejection.

2. V16 is single-object.

The user-facing task is object pose annotation for manipulated objects. A clip can involve several manipulated or contact-relevant objects. V17 must keep a multi-object timeline where each object has identity, active interval, role, geometry state, and evidence status.

3. V16 has no persistent object shape state.

A tomato, bowl, plate, lid, and trash can are not independent depth patches per frame. Near-rigid objects need a canonical mesh plus per-frame pose. Deformable objects need a canonical or reference surface plus deformation state and temporal regularization.

4. V16 does not implement occlusion smoothing.

Missing hands or objects become predicted states with uncertainty. Bad measurements are rejected or downweighted by image, mask, keypoint, depth, and motion residuals.

5. V16 contact is a diagnostic distance, not contact reasoning.

Contact requires a state variable. A small nearest distance can be false when hand/object states are bad; a large distance can be false when depth or pose is wrong. V17 must estimate contact jointly with hand and object state.

6. V16's 3D view is not designed as a visual explanation.

The 3D panel must show what the annotation means. A line plot with labels cannot satisfy the V17 rendering contract.

## Research Conclusions

The current literature supports a measurement-and-state design over a single replacement model.

HaWoR is directly relevant because it targets egocentric world-space hand motion, combines camera trajectory estimation with hand reconstruction, and includes a motion infiller for missing frames. Its official repository also depends on masked DROID-SLAM and Metric3D, which matches the failure mode of moving egocentric cameras and missing hands. Source: https://github.com/ThunderVVV/HaWoR and https://arxiv.org/abs/2501.02973.

The previous HaWoR branch did not prove HaWoR wrong. It proved the integration was incomplete: the raw camera-local HaWoR hands were partly plausible, while the tested bridge into the existing DROID/object world used a global Sim(3) alignment that produced severe reprojection and hand-scale errors. V17 therefore treats HaWoR as a primary measurement source for world-space hand motion and missing-frame infilling, while making the coordinate bridge itself a residual-checked graph variable.

SAM 2 supports promptable video segmentation and mask propagation, including multi-object video tracking support in the official repository. V17 uses it as one segmentation measurement source; the graph estimates the final object state from masks, depth, tracks, geometry, and contact evidence. Source: https://github.com/facebookresearch/sam2 and https://arxiv.org/abs/2408.00714.

VGGT predicts camera parameters, depth maps, point maps, and 3D point tracks from multiple views. V17 uses it as a geometry and track source for camera/object consistency checks, especially when DROID or monocular depth is unstable. Source: https://github.com/facebookresearch/vggt and https://arxiv.org/abs/2503.11651.

FoundationPose and BundleSDF are relevant for rigid or near-rigid object pose and reconstruction. FoundationPose handles model-based and model-free 6D pose estimation/tracking for novel objects; BundleSDF reconstructs and tracks unknown rigid objects from RGBD sequences with pose-graph optimization. These methods are not suitable as a universal solution for deformable bags, but they are the right class of method for tomato, bowl, plate, lid, can, and container-like objects after masks and depth are available. Sources: https://github.com/NVlabs/FoundationPose, https://arxiv.org/abs/2312.08344, https://github.com/NVlabs/BundleSDF, and https://arxiv.org/abs/2303.14158.

TRELLIS, Hunyuan3D, and PartCrafter can propose complete meshes or structured object priors from images. V17 may use them as prior proposal sources, but any generated mesh must pass visible replay, temporal track support, and physical consistency before entering the delivered annotation. Sources: https://github.com/microsoft/TRELLIS, https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1, and https://github.com/wgsxm/PartCrafter.

WHOLE and EgoGrasp point toward the correct formulation: world-space hand-object interaction reconstruction must model hands and objects jointly over time, especially under occlusion and object entries/exits. V17 implements this principle with available components, explicit residuals, and a joint hand-object graph. Sources: https://arxiv.org/abs/2602.22209 and https://arxiv.org/abs/2601.01050.

## V17 Pipeline Definition

### Stage 0: Representative Inputs

V17 will run on at least the same two full raw clips as V16:

- task5 tomato, 960 frames;
- trash, 1050 frames.

The named failure frames become required QC anchors for trash: 0182, 0260, 0764, 0856, 0949, 0970. Tomato must include frames around 0480, 0720, and 0760, plus frames where bowls/plates/trays are visible.

### Stage 1: Measurement Store

Every model output enters as a measurement record. The graph solver creates annotation states from residual-checked measurements.

Each measurement stores:

```text
frame_idx
entity_id
measurement_type
source_model
source_checkpoint
coordinate_frame
value
confidence
covariance_or_scale
residuals_against_other_sources
visibility_state
failure_reason
```

No hand, object, camera, or contact state can enter the delivered annotation unless its source measurements and residuals remain traceable.

### Stage 2: Camera And Metric Geometry

V17 will estimate camera/geometry with at least two independent sources:

- masked DROID/HaWoR-style egocentric SLAM for full trajectory continuity;
- VGGT for camera, depth, point maps, and tracks on selected windows;
- UniDepth or Metric3D as dense metric depth measurement.

Dynamic hand/object masks are excluded from static-background SLAM where the method supports masking. The camera state stores uncertainty and residual spikes. The renderer must project accepted hand/object states back through the same camera model, so an impossible camera-hand relation becomes visible in QC.

### Stage 3: Multi-Object Plan

V17 replaces the single object stream with an object roster.

A VLM/video-review agent proposes:

```text
object_id
natural_language_name
active_intervals
role_distribution
prompt_points_or_boxes
expected_physical_behavior
```

Roles are data, not code branches. Examples of roles are manipulated object, support object, container, tool, target surface, and distractor. The downstream pipeline treats all object records uniformly.

Open-vocabulary detection plus SAM 2/Cutie-style video segmentation produces masks for every active object. Object identities are checked by temporal mask propagation, point tracks, depth, and VLM review. A clip can deliver multiple object meshes and poses.

### Stage 4: Hand Measurement And State Estimation

V17 uses a fixed measurement set:

- HaWoR world-space hand motion and infilled hand trajectory measurements as the primary temporal hand-motion source;
- WiLoR per-frame MANO measurements as an independent image-conditioned hand source;
- HaMeR per-frame MANO measurements from RTMLib crop evidence, with explicit source-coordinate intrinsics and metric-translation residuals;
- RTMLib 2D keypoints;
- SAM 2 hand masks or another hand-mask source;
- metric depth over visible hand regions;
- learned hand-motion and hand-object interaction priors.

The delivered hand state is a fixed-lag smoothed MANO trajectory. The solver can choose among measurements because the objective defines residuals, not because code branches on visual cases.

HaWoR contributes motion continuity, world-space trajectory, and missing-frame infill. WiLoR and HaMeR contribute per-frame MANO image evidence through separate model families and crop contracts. The accepted hand state is the graph solution that best satisfies projection, mask, depth, temporal, and contact residuals. When hand sources disagree, V17 records the disagreement and either repairs the state or marks it unresolved.

For each hand and frame, the state can be:

```text
measured_accepted
measured_repaired
predicted_occluded
predicted_detector_miss
rejected_unresolved
outside_view
```

A visible hand cannot silently disappear. If all hand reconstructor measurements fail but the raw image/mask/keypoint evidence shows a hand, V17 emits a predicted state with high uncertainty and a QC flag, then tries local repair from neighboring frames.

### Stage 5: Object Geometry State

Each object has a canonical geometry state and a per-frame state.

Canonical geometry can come from:

- visible RGBD/depth fusion over accepted masks;
- BundleSDF-style rigid object reconstruction when the rigidity residual supports it;
- FoundationPose tracking when a reconstructed/generated mesh is accepted;
- TRELLIS/Hunyuan3D/PartCrafter mesh priors after replay acceptance.

Per-frame object state contains:

```text
T_wo_t        pose for rigid or near-rigid component
D_t           deformation field or surface offsets when needed
M_t           mask support
P_t           point-track support
Q_t           geometry and replay residuals
```

Rigidity is inferred from evidence. If a surface track set can be explained by one SE(3) transform with small residual, the object is near-rigid. If it cannot, the object uses a deformable surface state with smoothness and local area regularization. This is a physical distinction, not category branching.

For tomato, V17 enforces a near-rigid shape prior: one canonical tomato mesh with pose and small deformation, plus visible-surface updates only when they do not contradict the canonical shape.

For trash bag, V17 uses deformable surface state and avoids pretending there is a rigid 6D pose.

### Stage 6: Contact State And Physical Consistency

Contact is a latent state per hand patch and object surface region:

```text
no_contact
candidate_contact
sticking
sliding
supporting
occluded_contact
rejected_unresolved
```

The graph estimates contact from:

- 2D hand/object mask adjacency;
- depth ordering and depth gap;
- 3D hand-object SDF gap;
- nonpenetration;
- relative motion of hand patches and object surface tracks;
- object acceleration consistency when object mass/rigidity evidence makes the test meaningful;
- deformation consistency for nonrigid surfaces.

Contact cannot be asserted from nearest distance alone. Contact cannot be rejected when image/depth/track evidence supports contact but the current 3D state is inconsistent; that case becomes a state repair target.

### Stage 7: Learned-Prior Full-Timeline Smoother

V17 implements the prediction/update idea as a fixed-lag nonlinear factor graph. A simple constant-velocity or constant-acceleration prior is not the process model for hand-object manipulation. It can appear only as a weak local smoothness regularizer. The actual process terms are learned priors and physically grounded residuals.

State variables:

```text
T_wc_t                    camera pose
H_h,t                     MANO hand pose/shape/global transform
V_h,t                     hand velocity latent
Z_h,t                     learned hand-motion latent
G_o                       object canonical geometry
T_wo,t                    object pose for near-rigid components
D_o,t                     object deformation state
C_h,o,t                   contact mode variables
U_*                       uncertainty/covariance variables
```

Residuals:

```text
camera motion and reprojection
hand MANO prior
hand 2D keypoint reprojection
hand mask silhouette
hand metric-depth support
hand temporal velocity/acceleration
object mask replay
object depth replay
object point-track consistency
rigidity or deformation energy
object shape-prior consistency
hand-object nonpenetration
contact equality / sliding / support residuals
measurement confidence calibration
HaWoR motion-infill prior
hand-object correspondence prior
hand-object generative plausibility prior
```

The learned process model predicts through missing measurements with growing uncertainty. For hands, HaWoR-style motion infilling provides the primary learned temporal proposal. For contact, TOCH-style spatio-temporal object-to-hand correspondence provides a learned contact refinement prior. For broader hand-object plausibility, a G-HOP-style diffusion prior can propose or score physically plausible hand-object states. These learned priors enter as factors in the objective; they do not override image, mask, depth, object-track, or nonpenetration evidence.

Measurements update the state only when their residuals are plausible. Outlier measurements remain in the measurement store but do not become accepted states.

### Stage 8: QC

V17 QC is frame-local and timeline-local.

Required frame anchors:

- trash 0182: right hand must stay on the visible hand or be flagged unresolved;
- trash 0260: wrong edge hand hypotheses must be rejected or repaired;
- trash 0764: hand-bag contact must be represented as contact or unresolved repair target, not isolated separation;
- trash 0856: bad hand/object state must not produce a confident contact label;
- trash 0949: visible hands must pass hand residual checks;
- trash 0970: visible hands must not disappear silently;
- tomato 0480/0720/0760: tomato mesh must preserve persistent near-rigid shape;
- tomato context: bowls/plates/trays must appear in the multi-object roster when visible and relevant to the manipulation.

Aggregate QC is insufficient. Every deliverable manifest must include:

```text
named_anchor_status
measurement_source_coverage
accepted_state_coverage
rejected_unresolved_frames
hand_residual_summary
object_residual_summary
contact_residual_summary
render_frame_count_qc
```

### Stage 9: Rendering

The V17 world view must be rebuilt.

The 3D panel must show:

- shaded MANO hand meshes with skeleton overlays as secondary cues;
- shaded object meshes with separate colors per object;
- current camera frustum with textured image plane or raw-frame thumbnail plane;
- a line of sight from camera to hand-object region;
- stable world axes and scale;
- head trajectory as a subtle path;
- local manipulation close-up with contact patches;
- uncertainty/rejection status when state is predicted or unresolved.

The side-by-side video remains full-length and synchronized with the raw video. Debug plots can exist as QC artifacts, but the deliverable must read as a reconstruction.

### Stage 10: Deliverables And Closure

For every V17 sample:

```text
overlay_mano_object_multi.mp4
world_reconstruction_3d_v17.mp4
side_by_side_v17.mp4
annotations_v17_full.json
measurements_v17_full/
object_meshes_v17_full/
v17_manifest.json
v17_anchor_qc.json
```

Closure requires:

- output frame count and fps equal raw video;
- all named V16 failure frames pass or are explicitly marked unresolved with correct evidence;
- no visible hand disappears silently;
- no per-frame object depth patch is delivered as a persistent near-rigid object;
- multiple manipulated/context objects are represented where visible and relevant;
- contact labels are state estimates with supporting residuals;
- world reconstruction is visually legible as a 3D scene.
