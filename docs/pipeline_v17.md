# Pipeline V17: Measurement-Grounded Multi-Object HOI Annotation

## Status

V16 closed only as full-duration packaging. It produced full-length videos for two raw clips, but it did not satisfy the original V3 joint graph requirement and its annotations do not meet the quality requirement. V17 treats every detector output as a measurement with residuals, confidence, and source evidence before the solver can accept an annotation state.

V17 implementation has produced the measurement store, full-state integration, a latent contact-mode graph, and sparse full-timeline evidence-consistency graphs. The measurement store reads V16 full-video outputs, prior HaWoR/WiLoR artifacts, HaMeR repairs, SAM2 object masks, contact-state rows, local deformable contact patches, and tomato persistent-shape state as measurements, then emits anchor QC before any graph solver can accept or repair them. The full-state integration writes full-length V17 evidence/QC JSONs, and the current render step verifies duration only. The contact-mode graph estimates per-frame per-hand contact/no-contact/unobserved modes from graph-corrected hand-object gaps, object-mask proximity, selected anchors, and temporal switch cost. The sparse geometry graph optimizes per-active-frame object translation corrections, per-active-frame small-angle object rotation corrections, and per-valid-hand camera-ray depth corrections against either selected anchor contacts or contact-mode factor-ready rows, object priors, pose smoothness, and hand-ray smoothness. Its contact equality terms use a local nearest MANO surface patch and report broader hand-object distances separately. It keeps camera trajectory, MANO articulation and shape, object mesh topology, and contact mode labels fixed, so it is an integrated consistency solver for the current V17 evidence layer. The complete nonlinear V3 joint solver remains open.

The current generated graph annotations still carry one manipulated object stream per frame. The V17 multi-object roster and multi-object pose timeline remain an open requirement: object-plan and SAM2 evidence record additional objects, but the contact-mode-factor graph outputs do not yet emit simultaneous object states for bowls, plates, trays, lids, trash-can parts, and other contact-relevant objects. Any artifact whose schema exposes only `object` rather than simultaneous `objects` is a single-manipulated-object QC artifact.

V17 also corrects a version-accounting problem. V3 already identified the core requirement: solve or expose the metric contradiction between MANO hands and object geometry through a joint factor graph. Later versions implemented real component graphs, including object-pose, sparse object-track, and contact-dynamics graphs. Their scope stayed at selected windows or selected state variables. V16 then closed as a full-length delivery artifact with QC flags while the original joint graph requirement remained open. V17 therefore treats prior graph outputs as evidence modules and reports the current sparse full-timeline graph separately from the still-unimplemented complete nonlinear V3 solver.

The current measurement-store implementation is the evidence layer for V17 state estimation. Model outputs remain traceable measurements with confidence, residual, source, and failure fields; missing hands, missing objects, missing contact states, and incomplete HaWoR/WiLoR coverage become explicit QC failures.

The current HaWoR evidence path uses a compact full-video adapter input generated from V16 annotations. The compact file preserves frame indices, timestamps, source camera transforms, source intrinsics, measured V16 hand 2D keypoints, detector scores, and hand boxes, then reruns the HaWoR camera-local adapter against the full 0-1049 HaWoR NPZ. The adapter input therefore contains only the fields read by the HaWoR residual calculation.

HaWoR rows without current-frame observed hand support are stored as `hawor_motion_infill_candidate` measurements. They can support an occluded or detector-miss state after temporal, projection-contradiction, contact, and nonpenetration checks. Current-frame 2D evidence is required for observed-visible measurements, not for motion-infill candidates.

The trash hand-evidence path now includes VLM-localized visible hand boxes for anchor frames where RTMLib or full-frame HaMeR crops were broad, missing, or attached to the wrong region. The VLM boxes enter the measurement store as image-localization evidence and HaMeR crop inputs. Synthetic keypoints derived from those boxes have no metric meaning; the usable 3D evidence is the HaMeR MANO output and its source-camera reprojection residual.

The trash anchor repair path now materializes selected VLM-box HaMeR hypotheses as V17 repair hand states for frames 0182, 0260, 0764, 0856, 0949, and 0970. The measurement store also ingests contact evidence recomputed from those repaired states. This clears the hand-state blockers at the named trash anchors.

Frames 0182 and 0856 exposed the wrong object variable. Whole-object depth re-anchoring can force a hand-bag metric contact, but temporal validation rejects that variable because it shifts the entire deformable bag surface. V17 now records separate local deformable contact-patch meshes for those frames. Frame 0182 uses a 206-vertex, 302-face black-bag patch centered at the repaired right-hand contact support; the patch occupies 1.43 percent of the SAM2 bag mask and has a 2.47 mm nearest hand-surface distance. Frame 0856 uses a 638-vertex, 1,086-face white-bag patch; the patch occupies 12.77 percent of the SAM2 bag mask and has a 0.97 mm nearest hand-surface distance. The anchor contact graph v4 selects those local patch rows for 0182 and 0856, accepts contact at 0182, 0260, 0764, and 0856, and accepts no-contact at 0949 and 0970. The rejected whole-object depth candidates remain in the measurement store as failed evidence, while the accepted state is the local deformable surface geometry.

The tomato measurement path now has a persistent visible-surface mesh state for `object:obj_tomato`. The solver fuses SAM2 object masks with V16 metric-depth surface extraction over the full active mask interval, rejects 20 temporal surface-scale outliers, and writes a canonical mesh with 84,318 vertices and 136,902 faces. The robust 1-99 percent canonical extents are 9.85 cm, 9.67 cm, and 9.56 cm; raw min-max extent is reported separately because sparse tails can overstate object scale. Tomato anchors 0480, 0720, and 0760 pass persistent-shape QC with surface-to-canonical p95 residuals of 1.7 mm, 5.6 mm, and 1.3 mm. The full-state builder translates the canonical-local visible surfaces by their per-frame `object_center_world_m` before saving the V17 mesh archive, so the archive carries world-coordinate meshes. The tomato contact graph v1 selects left-hand contact rows at 0480, 0720, and 0760.

The current measurement store passes the named trash and tomato anchors. The evidence-layer full-state integration also renders full raw-video outputs for both representative clips: trash has 1,050 raw frames and 1,050 frames in overlay, world, and side-by-side renders; tomato has 960 raw frames and 960 frames in overlay, world, and side-by-side renders. The anchor-only sparse evidence-consistency graph remains a limited QC branch. Trash uses four selected contact factors at 0182, 0260, 0764, and 0856; after optimization its local contact p95-of-p95 is 4.56 mm, broader 80-nearest-point p95-of-p95 is 11.94 mm, max object translation correction is 6.89 mm, max object rotation correction is 0.0259 rad, and max hand camera-ray correction is 3.18 mm. Tomato uses three selected contact factors at 0480, 0720, and 0760; after optimization its local contact p95-of-p95 is 6.08 mm, broader 80-nearest-point p95-of-p95 is 22.44 mm, max object translation correction is 7.62 mm, max object rotation correction is 0.0105 rad, and max hand camera-ray correction is 10.69 mm. The tomato anchor-only branch therefore remains partial even under the sparse evidence-consistency metric. The graph uses local nearest MANO patches for the current contact correspondence selection; a previous 160-point stochastic hand subsample made tomato frame 0480 fail at 36.1 mm, which exposed a sampling artifact in the sparse contact linearization.

The contact-mode graph is implemented as a V17 contact-state layer; the full V3 solver remains open. Its default manifests read the anchor-only sparse graph outputs, so it estimates contact modes from fixed graph-corrected geometry while hand geometry, object geometry, object pose, and contact labels remain fixed inputs. It accepts both representative clips with zero anchor contradictions and explicit `v3_solver_complete=false` metadata. Trash has 2,100 hand-side rows, 1,625 active geometry observations, 473 unobserved rows, 172 contact-mode rows, and 81 contact-factor-ready rows after requiring positive contact evidence, explicit mask-distance evidence, and sparse-graph hand residual compatibility. Tomato has 1,920 hand-side rows, 1,340 active observations, 580 unobserved rows, 527 contact-mode rows, and 382 contact-factor-ready rows. Contact-mode reports now store row-level `contact_factor_readiness_checks`, including hand residual values, mask-distance availability, and threshold predicates, so readiness can be audited from the JSON. Contact-mode interval sheets are regenerated from solved rows with visible QC banners and report per-side timelines, counts, interval previews, and readiness state; they are contact-state QC summaries, and visual-quality rendering remains open. The long tomato left-hand interval from frame 0714 to 0939 matches hand-held washing/rinsing behavior in the graph-corrected side-by-side video, while the trash contact bursts match bag/trash-can manipulation. Rows without hand/object geometry are marked `unobserved` and cannot inherit temporal contact. Rows whose unary evidence opposes contact can remain temporally labeled as contact, but they cannot become geometry factors.

The contact-mode-factor sparse graph consumes only accepted contact-mode `contact_factor_ready` rows through an explicit `--contact-mode-graph-root` input. It structurally accepts both representative clips with converged local contact-patch correspondences. The modeled contact patch now uses 16 nearest MANO surface vertices. A support-size sweep showed why the broader metric cannot be the local contact predicate: trash/tomato p95-of-p95 stays below 5 mm through 18 nearest vertices, then rises to 12.84/13.22 mm at 80 vertices because non-contact hand surface is being included. The 80-nearest-point report remains a support-size sensitivity diagnostic, not a requirement that the whole nearby hand surface touch the object. Trash uses all 81 corrected contact-mode factors, with 1,296 linearized local-patch correspondences, contact-patch p95-of-p95 4.75 mm, broader 80-nearest-point p95-of-p95 12.84 mm, max object translation correction 6.32 mm, max object rotation correction 0.0295 rad, and max hand ray correction 3.36 mm. Tomato uses all 382 corrected contact-mode factors, with 6,112 local-patch correspondences, contact-patch p95-of-p95 5.00 mm, broader 80-nearest-point p95-of-p95 13.22 mm, max object translation correction 7.15 mm, max object rotation correction 0.0294 rad, and max hand ray correction 12.21 mm. The 5 mm numbers are local evidence-consistency diagnostics under the sparse graph's fixed camera, fixed MANO articulation, fixed object topology, fixed contact-label, and nearest-vertex support assumptions. They do not prove a physically valid contact patch until patch identity, image/depth support, and anatomical/contact-area stability are also estimated. The generated reports set `accuracy_target_met=false`, `annotation_ready=false`, and `deliverable_ready=false` because the complete V3-class nonlinear solver remains open.

Two contact-support experiments were rejected before this interpretation was adopted. A single contiguous MANO patch made the graph nonconvergent and raised p95-of-p95 to 15.8 mm for trash and 10.3 mm for tomato. A small multi-site anatomical support model remained nonconvergent at 8.77 mm and 8.07 mm while increasing runtime. These negative results show that support selection alone is not the remaining mechanism; full MANO articulation/surface fitting and object geometry remain fixed in the current sparse graph.

The graph output materializes the optimized state in the annotation files. Object translation and small-angle rotation corrections move `center_world_m`, nested V17 surface centers, local-patch world vertex arrays when present, and the corrected mesh archive around the solved object center. Hand camera-ray corrections move world-space MANO vertices and joints along the solved camera optical axis. Source-camera measurements remain unchanged as evidence.

Graph-corrected full-length renders pass duration QC only. Trash graph renders contain 1,050 frames for overlay, world, and side-by-side outputs; tomato graph renders contain 960 frames for all three outputs. The visual inspection sheets are sampled from the final graph-corrected side-by-side videos and recorded in the render summary. The current trash sheet shows selected contact frames 0182, 0260, 0764, and 0856 with displayed nearest gaps of 0.5 mm, 1.7 mm, 2.0 mm, and 0.9 mm. The current tomato sheet shows selected contact frames 0480, 0720, and 0760 with displayed nearest gaps of 2.4 mm, 0.5 mm, and 2.3 mm. The regenerated contact-mode-factor renders also pass duration QC: trash contains 1,050 frames for overlay, world, and side-by-side outputs; tomato contains 960 frames for all three outputs. New QC renders use `qc_` filenames and their render summary carries `render_qc_scope=duration_only_not_visual_quality`, `visual_quality_qc_pass=false`, `stage9_visual_deliverable_ready=false`, `annotation_ready=false`, `deliverable_ready=false`, and `accuracy_target_met=false`. These videos are duration evidence/QC renders rather than V17 closure deliverables. The current world videos and sheets still use the diagnostic V16/V17 renderer; the Stage 9 audience renderer with shaded MANO/object surfaces, image-plane context, close-up manipulation views, and uncertainty/rejection status remains open.

Current V17 evidence outputs:

```text
/data2/ego_annotation_outputs/v17_full_state/
/data2/ego_annotation_outputs/v17_full_timeline_factor_graph/
/data2/ego_annotation_outputs/v17_full_timeline_factor_graph_renders/
/data2/ego_annotation_outputs/v17_contact_mode_graph/trash_1050/v17_contact_mode_graph_report.json
/data2/ego_annotation_outputs/v17_contact_mode_graph/trash_1050/contact_mode_interval_review_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_graph/task5_tomato_960/v17_contact_mode_graph_report.json
/data2/ego_annotation_outputs/v17_contact_mode_graph/task5_tomato_960/contact_mode_interval_review_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_graph/v17_contact_mode_graph_summary.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/v17_full_timeline_factor_graph_summary.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/{trash_1050,task5_tomato_960}/v17_full_timeline_factor_graph_report.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/{trash_1050,task5_tomato_960}/annotations_v17_full_timeline_graph.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/{trash_1050,task5_tomato_960}/object_meshes_v17_full_timeline_graph.npz
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/renders/qc_overlay_mano_object_multi.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/renders/qc_world_reconstruction_3d_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/renders/qc_side_by_side_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/v17_contact_mode_factor_side_by_side_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/renders/qc_overlay_mano_object_multi.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/renders/qc_world_reconstruction_3d_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/renders/qc_side_by_side_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/v17_contact_mode_factor_side_by_side_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/v17_render_summary.json
```

The first three roots above are legacy V17 evidence/QC outputs whose filenames predate the QC naming correction. The current contact-mode-factor render manifests use `qc_` render filenames.

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
- A hand region is visible in the raw frame, but the delivered hand list is empty.
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

WHOLE and EgoGrasp point toward the correct formulation: world-space hand-object interaction reconstruction must model hands and objects jointly over time, especially under occlusion and object entries/exits. V17 uses this formulation as the missing target. Current V17 artifacts provide measurement layers, contact-mode QC, and a sparse evidence-consistency graph, while the full joint hand-object-camera-depth-contact solver remains open. Sources: https://arxiv.org/abs/2602.22209 and https://arxiv.org/abs/2601.01050.

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
- VLM-localized visible hand boxes for detector-miss or bad-crop anchors, used as HaMeR crop localization and as image-level hand evidence;
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

### Stage 7: Full Nonlinear Graph Target

This section defines the still-open full nonlinear solver target. The implemented V17 graphs are the contact-mode graph and sparse evidence-consistency graphs described in the Status section: the contact-mode graph estimates binary contact/no-contact/unobserved modes from fixed V17 geometry and image evidence, and the sparse geometry graph optimizes object translation, small-angle object rotation, local contact-patch correspondences, and hand camera-ray depth corrections. Camera trajectory, MANO articulation and shape, object mesh topology, full object geometry, and nonlinear contact physics stay fixed. The current sparse graph only shows consistency of the accepted evidence layer under that limited variable set; it is not full annotation closure.

The full V3-class solver should implement the prediction/update idea as a fixed-lag nonlinear factor graph. A simple constant-velocity or constant-acceleration prior is not the process model for hand-object manipulation. It can appear only as a weak local smoothness regularizer. The actual process terms are learned priors and physically grounded residuals.

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
- trash 0970: visible hand evidence requires an accepted, predicted, or unresolved hand state;
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
