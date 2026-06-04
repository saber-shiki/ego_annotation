# Pipeline V3: Referring Segmentation and Joint Metric Contact

## Why V3 Exists

V2 replaced object proxies with model-planned masks and observed-surface meshes. It proved a real object mesh path for the pink-lid track, but it also exposed a larger 3D inconsistency:

- The accepted pink-lid mask is visually stable.
- Metric-depth meshing covers the contact window.
- MANO vertices often project near the object mask in 2D.
- The same vertices are hundreds of millimeters away from the object surface in camera depth.

The compact contact-depth report for frames 840 to 930 has 84 frames where hand vertices project near the object mask:

- median hand-minus-object depth gap: 0.385 m
- p95 hand-minus-object depth gap: 0.667 m
- median object-over-hand depth ratio: 0.741
- p05/p95 object-over-hand depth ratio: 0.472 / 0.937

This magnitude cannot be fixed by a Kalman smoother or an object-pose-only optimizer. V3 must jointly reason about object identity, object mesh, MANO metric scale/depth, camera pose scale, metric depth reliability, and contact state.

A lightweight 1D contact-depth diagnostic then solved only the depth gap for frames with at least 80 near-mask hand vertices. This diagnostic is intentionally underdetermined: each frame has one contact-depth equation and four correction variables before priors. It reduced median corrected depth gap to 0.16 mm, but the inferred correction was not small:

- median hand-depth scale: 0.939
- median object-depth scale: 1.217
- p95 object-depth scale: 1.581
- median object-depth shift: 43 mm

This is useful as a causal diagnostic, not as an annotation result. It shows that contact can be made numerically true when scale and shift corrections are explicit variables, while the true faulty subsystem remains unidentified. A hidden correction would destroy the evidence about which subsystem is wrong.

## Perception Branch

The white-liner track remains the hardest object-perception case in the representative trash clip. The rejected evidence is explicit:

- OWLv2 plus SAM selected the pink lid or background for the white-bag track.
- VLM point prompts plus SAM improved selected frames but spilled onto floor glare, clothing, and the pink lid.
- Metric-depth component cleanup failed because the wrong lid/background components can have plausible monocular depth.
- SAM2 propagation from weak point-prompt seeds tracked floor glare or the pink lid.

The next valid mask model must be language-conditioned across video, not another hand-written cleanup rule.

### SAM3

SAM3/SAM3.1 is the preferred direct interface because the official repository supports image and video segmentation from text prompts. SAM3.1 adds multiplex video tracking for multiple objects. Current environment access blocks this path:

- `facebook/sam3` metadata is visible, but `sam3.pt` download returns Hugging Face gated-access 403.
- `facebook/sam3.1` metadata is visible, but `sam3.1_multiplex.pt` download returns gated-access 403.

Implemented preparation:

- `scripts/run_sam3_referring_track.py`

This runner should be used after checkpoint access is granted. It consumes a video window, text prompt, optional VLM point prompts, and writes mask PNGs, review stills, overlay video, and QC JSON.

### SAMWISE

SAMWISE is the next unverified executable candidate because it is a CVPR 2025 text-driven video segmentation model built on SAM2 and provides arbitrary-video/frame-folder inference from natural-language prompts.

Implemented preparation:

- `scripts/run_samwise_referring_masks.py`
- `scripts/remote_setup_samwise.sh`

Planned run:

- input: frames 678 to 918 of the representative trash clip
- prompt: `translucent white plastic trash bag liner being opened and placed inside the small trash can`
- expected raw output: binary mask PNGs, overlay video, review stills at 678, 720, 797, 858, 880, 900, 918, and QC JSON
- acceptance requirement: visual or VLM semantic verification must accept the masks before mesh reconstruction

The GPU hosts became unreachable over SSH before setup/checkpoint verification completed. This is an external connectivity blocker for the SAMWISE execution branch, not a model-selection decision.

### SOLA

SOLA is a secondary fallback. It generates SAM2 tracks and selects tracks by language alignment. Its public instructions are organized around MeViS and Ref-Youtube-VOS dataset-format track generation, so it is less direct than SAMWISE for immediate custom-video inference.

## Mesh Branch

V2 observed-surface meshes are necessary but incomplete: they reconstruct only the visible surface from mask and depth. V3 needs an object-centric complete mesh state.

Current evidence:

- TripoSR produced a complete mesh prior for frame 858 from the accepted pink-lid crop.
- Strict frame-858 alignment to the observed metric-depth surface reached median prior-to-observed and observed-to-prior distances near 10 mm.
- The 858-930 window optimizer reduced object surface residuals but failed contact consistency.
- VGGT masked multiview geometry over frames 858 to 880 produced noncollapsed raw object points, but camera-motion Sim3 scaling collapsed the object to millimeters. Rescaling from the observed surface extent produced a 0.57 m by 0.66 m by 0.42 m rounded mesh with 71 mm median point-to-observed distance and a visual blob around the lid, so it is rejected as an annotation mesh.
- Hunyuan3D-2mv multiview generation from four depth-grown object crops ran successfully on A800 and produced a 369k-vertex mesh. Frame 858/878 alignment again reached low nearest-surface medians, but visual QC rejects the result as a generic oval shell with fuzzy side growth rather than scene-faithful lid/can geometry.
- SAMWISE text-conditioned segmentation produced clean object-level lid masks for frames 858 to 880. A dense mask-depth height-field mesh from those masks reached median silhouette-mask IoU 0.9965 and median vertex-depth error 0.24 mm across 23 frames. This establishes a valid visible-surface mesh observation for the lid.
- The same height-field archive fails temporal object consistency. Its robust camera-frame Z extent changes up to 2.50x relative to the median, and the world center speed reaches 8.79 m/s between adjacent video frames. This is a metric-depth/camera-scale inconsistency, because the per-frame projection evidence is already strong.
- A one-variable-per-frame depth-scale graph regularized object XY extent to within about 6 percent of the median, but projection-depth error rose to 36.7 mm median and 293 mm on frame 880. The tradeoff exposes the missing constraint: V3 needs independent metric camera/depth scale evidence before rigid object pose can be trusted.
- Full-scene VGGT was run on frames 858 to 880 using the original scene frames and SAMWISE masks only for object-point selection. This is stronger evidence than the rejected object-crop VGGT branch because the model sees the room and camera motion. VGGT camera centers align to the DROID trajectory with 10.5 mm median error after a Sim3, but the required VGGT-to-DROID scale is 0.129 and the VGGT-predicted focal maps back to about 1200 px in the 1920-wide source frame, far from the DROID prior of 2304 px.
- VGGT object points project entirely inside the SAMWISE mask after correcting the aspect-ratio-preserving resize and padding. They expose a concrete Depth Anything failure: in frames 878 to 880, VGGT places the lid surface at about 0.69 to 0.75 m while Depth Anything places the same mask at about 1.01 to 1.12 m. This explains the late-frame heightfield expansion without blaming the mask.
- A VGGT observed-surface mesh built from the selected points is temporally coherent: robust camera extents are 0.050 x 0.034 x 0.044 m, extent-ratio max-log median is 0.049, and pairwise center speed median is 0.056 m/s. This is still only a compact visible-surface patch, not a full lid mesh or watertight object.
- Re-solving MANO translations under VGGT intrinsics reduces the high-score contact gap to 86 mm median, but does not close it. A bounded temporal Z-shift/contact optimizer can drive median contact gap near zero only by hitting the 35 cm shift bound and producing frame-880 median keypoint reprojection error of 80 px with an implausibly small hand scale. VGGT reduces the object-depth error, but current MANO measurements remain inconsistent with 5 mm contact annotation.

Interpretation:

Object-complete asset priors are feasible, but the current object crops underconstrain hidden geometry. Mesh alignment alone can fit the partial observed surface while inventing a plausible but wrong backside. The scene-derived visible surface is now stronger evidence than the generative complete priors for this clip. The next mesh branch must add an independent metric-scale source and then use the graph to decide whether the rigid lid trajectory, MANO trajectory, and camera trajectory can satisfy surface projection, temporal motion, and contact together.

Implemented current mesh tools:

- `scripts/complete_object_heightfield_from_mask_depth_v3.py`
- `scripts/diagnose_object_mesh_temporal_consistency_v3.py`
- `scripts/reconstruct_scaled_observed_object_mesh_v3.py`
- `scripts/regularize_heightfield_depth_scale_v3.py`

## Required V3 Solver

V3 should model one joint state over the clip:

- camera trajectory scale and per-frame pose corrections;
- MANO hand pose, translation, and metric scale correction;
- object mesh pose and, for deformable objects, low-dimensional deformation;
- metric-depth scale/bias and per-frame reliability;
- binary or probabilistic contact state per hand/object region.

Factors:

- 2D MANO reprojection to WiLoR keypoints;
- verified mask silhouette agreement;
- object mesh to metric-depth surface residuals;
- MANO/object non-penetration signed-distance residuals;
- contact attraction only when contact is inferred from image evidence and temporal continuity;
- contact attraction downweighted or disabled for predicted or low-confidence hands whose reprojection residual is large;
- temporal smoothness on camera, hands, object pose, and deformation;
- prior terms on physically plausible hand size, object rigidity/deformation, and depth scale.

The graph must expose residual conflicts. A low object-depth residual with a 0.4 m hand/object depth gap is a failed joint annotation, not a success.

After the full-scene VGGT branch, the next graph should treat VGGT scene geometry as an independent depth/camera factor rather than replacing all other sources with it. The current factor graph needs:

- camera intrinsics and Sim3 scale variables constrained by DROID camera motion, VGGT camera motion, and any real calibration if it becomes available;
- object visible-surface factors from SAMWISE plus VGGT, with Depth Anything downweighted or rejected on frames where it contradicts VGGT by hundreds of millimeters;
- MANO depth variables constrained by 2D reprojection, temporal velocity/acceleration, hand-size priors, and detector confidence;
- contact variables that can turn off or mark a hand observation unreliable when satisfying contact would require large reprojection error, hand-scale collapse, or bound-saturated depth shifts.

The failure to close contact after VGGT is a useful V3 result because it separates two mechanisms: Depth Anything creates large object-depth outliers in late frames, and WiLoR/MANO still places some measured hands at incompatible depths even under VGGT intrinsics. V3 cannot close until a stronger temporal hand model or direct egocentric hand-depth model repairs that second mechanism.

## Surface-Specific Contact Branch

The corrected V2-mask reliability diagnostic used the real pink-lid mask source and still found zero reliable contact rows. The residual comes from more than mask identity: a single lid mask collapses several physical surfaces that can have different depth, visibility, and contact state.

Implemented current branch:

- `scripts/build_contact_surface_plan_v3.py`
- `scripts/render_point_prompt_review_v3.py`
- `scripts/adapt_sam2_track_to_annotations_v3.py`

Model-produced surface plan:

`/data2/ego_annotation_outputs/representative_trash/v3_contact_surface_plan_840_930/contact_surface_plan_vlm.json`

The plan splits the 840 to 930 contact window into five visible surface tracks:

- `pink_lid_top_dished_panel_visible`
- `pink_lid_raised_annular_rim`
- `pink_lid_outer_vertical_flange_edge`
- `second_can_exposed_opening_rim`
- `white_liner_draped_edge_second_can`

Point prompts and review stills:

`/data2/ego_annotation_outputs/representative_trash/v3_contact_surface_points_840_930/`

Inspection status:

- the central lid panel, annular rim, outer flange, and early liner prompts are visually credible;
- the exposed can rim is lower confidence and correctly marks frame 858 invisible;
- later liner prompts are plausible but boundary-sensitive because liner, rim, hand, and lid pixels overlap near the perimeter.

This branch supplies surface-specific mask targets so SAM2 or a referring video segmentation model can produce measured masks, metric-depth observed-surface meshes, and per-surface contact reliability. Contact factors should activate only after the surface mask, metric depth, MANO projection, and temporal support agree. V3 closes only when the resulting hand/object state passes per-surface contact reliability; zero reliable rows would strengthen the falsification of the current hand/camera/depth state.

Current diagnostics narrow the remaining missing evidence. The next solver cannot be another local smoother over the same variables. It must add at least one nonlocal source of metric information:

- a calibrated camera/head scale source, such as measured camera intrinsics plus a known-size object or scene measurement;
- a stronger 3D hand-depth model that predicts metric hand state directly from egocentric images and is checked against 2D RTMLib/WiLoR agreement;
- a temporal hand reconstruction stage that optimizes MANO pose and depth over many frames with metric-depth observations, then uses contact as a checked residual rather than a mandatory attraction;
- object support and force checks only after the hand/object trajectory is geometrically credible.

Until one of these sources is added, V3 should report the contradiction instead of rendering corrected contact. The current evidence rejects these approaches as closure mechanisms:

- object-only mesh pose refinement;
- translation-only, similarity-only, rigid, and local MANO pose refits;
- HaWoR direct replacement or camera-local adaptation;
- Kalman smoothing over the current hand state;
- forcing contact when image-supported contact rows are sparse or inconsistent.

Operationally, the version sequence is:

- V1: prove the full annotation plumbing with dense head trajectory, WiLoR MANO, captions, and an initial 3D presentation.
- V2: remove object proxies by using VLM object plans, open-vocabulary detection, SAM masks, metric depth, and observed-surface object meshes.
- V3: solve or expose the metric contradiction between the observed object mesh and MANO hands through a joint factor graph. V3 cannot close by producing a nicer render while contact stays hundreds of millimeters wrong.

QC in this project means falsification of the annotation, not only file checks. Structural QC checks that videos, frame counts, JSON records, and mesh archives exist. Geometric QC checks surface fit, silhouette fit, MANO reprojection, hand/object contact distance, penetration support, temporal smoothness, and whether any residual improves only by pushing hidden variables to implausible values. Visual QC checks that the overlay and 3D presentation show the intended object and hand state rather than a diagnostic-looking plot or a wrong object.

### MANO Reprojection Diagnostic

The contact window also shows that the MANO source-camera placement has nontrivial 2D disagreement with the hand detector boxes:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_reprojection_depth_840_930.json`

Result:

- hand rows: 182;
- MANO median depth: 1.326 m;
- projected MANO bbox versus detector bbox median L2 residual: 103 px;
- projected MANO bbox versus detector bbox p95 L2 residual: 231 px;
- projected MANO bbox versus detector bbox median max-axis residual: 75 px.
- per-side median L2 residual: left 119 px, right 67 px;
- worst residuals occur on the right hand around frames 875 to 884, where several rows are predicted or low-confidence hand states.

Interpretation: the hand side of the geometry is not a fixed metric oracle. V3 needs explicit MANO translation/depth variables constrained by 2D reprojection, temporal motion, hand-size priors, and hand observation confidence before contact factors can be trusted.

### Joint Depth Contact Probe

The first low-dimensional joint probe uses the same 840 to 930 contact-depth rows, but constrains the correction to one shared MANO depth scale, one shared object depth scale, and smooth per-frame depth shifts:

`/data2/ego_annotation_outputs/representative_trash/v3_joint_depth_contact_840_930.json`

Result:

- rows: 79 frames with at least 80 near-mask hand vertices;
- raw hand-object depth gap median/p95: 0.390 m / 0.667 m;
- corrected absolute gap median/p95: 0.009 m / 0.059 m;
- contact solved threshold: 0.010 m p95;
- hand depth scale: 0.750, exactly the lower bound;
- object depth scale: 1.052;
- hand and object depth shifts both hit their absolute bounds.

Status: `diagnostic_contact_depth_conflict_remains`.

Interpretation: shared depth scale plus smooth shifts cannot explain the contact conflict within the current bounds. The next solver must add image reprojection and silhouette terms for MANO/camera/object state, because the depth-only contact rows alone force implausible corrections and still leave 59 mm p95 contact-depth error.

### Confidence-Gated Object Factor Graph

The object-pose factor graph was rerun on frames 858 to 880 with contact residuals weighted by hand detector score and MANO reprojection fit:

`/data2/ego_annotation_outputs/representative_trash/v3_factor_graph_858_880_confidence/qc_object_factor_graph_v3.json`

Result:

- used frames: 23;
- contact weight median/p95: 0.360 / 0.609;
- observed-to-prior median surface distance: 56.4 mm to 18.0 mm;
- prior-to-observed median surface distance: 60.6 mm to 16.7 mm;
- contact median distance: 608 mm to 560 mm;
- contact p95 distance: 665 mm to 659 mm;
- depth-axis offset median/p95: 23 mm / 281 mm;
- optimizer status: hit `max_nfev=45`.

Interpretation: confidence gating reduces the influence of weak hand states, but object-pose optimization still cannot repair contact. V3 must add MANO/camera depth variables with reprojection and temporal constraints rather than treating the hand mesh as fixed.

### MANO Contact-Reprojection Tradeoff

The next diagnostic asks what would happen if MANO were moved far enough to satisfy the object contact depth:

`/data2/ego_annotation_outputs/representative_trash/v3_mano_contact_reprojection_tradeoff_840_930.json`

Result:

- per-hand rows: 106;
- median absolute hand/object contact-depth gap: 0.423 m;
- measured hand rows: 65, with median gap 0.381 m;
- measured rows with detector score at least 0.50: 58, with median gap 0.393 m;
- median center-ray hand translation required to match object depth: 0.427 m;
- current MANO bbox residual median: 122 px L2;
- median center-ray translation changes bbox residual by 18 px, with p95 change 151 px;
- projection-preserving camera-origin scale median: 0.717;
- median implied metric hand-extent change under that scale: 78.6 mm.

Interpretation: the large gap remains when predicted and low-score hands are excluded. There is no small harmless MANO correction that makes the current object surface contact-consistent. Rigidly moving the hand to the object requires roughly 0.4 m translation. Preserving the 2D projection while changing depth implies a hand-size change around 80 mm at the median. The conflict belongs to joint MANO/camera/depth scale estimation, not to object-pose refinement alone and not mainly to predicted-hand artifacts.

### Joint MANO-Object Graph Probe

Implemented:

- `scripts/optimize_joint_mano_object_graph_v3.py`

The first probe uses frames 858 to 862 with the strict pink-lid observed mesh and the frame-858 TripoSR mesh prior:

`/data2/ego_annotation_outputs/representative_trash/v3_joint_mano_object_858_862/qc_joint_mano_object_graph_v3.json`

State:

- per-frame object rotation, translation, and camera-axis depth offset;
- one global hand metric scale;
- one per-hand center-ray depth shift.

Factors:

- observed object surface to complete-mesh prior;
- complete-mesh prior to observed object surface;
- object silhouette inside the accepted mask;
- hand-object contact proximity from near-mask MANO vertices;
- MANO bbox reprojection to detector boxes;
- temporal smoothness and anchor priors.

Result:

- used frames: 5;
- hand factors: 3;
- optimizer hit `max_nfev=25`, so this is still a diagnostic probe;
- observed-to-prior median surface distance: 93.4 mm to 24.2 mm;
- prior-to-observed median surface distance: 84.1 mm to 20.2 mm;
- contact median distance: 679 mm to 533 mm;
- contact p95 distance: 698 mm to 572 mm;
- MANO bbox residual median: 129 px to 128 px;
- hand scale: 0.99994;
- median hand ray shift: -8.0 mm;
- median object depth-axis offset: 65 mm.

Status: `diagnostic_joint_surface_improved_contact_remains_large`.

Interpretation: adding explicit MANO scale and ray-shift variables does not solve contact under reprojection and hand-size priors. The graph improves object surface fit but leaves more than 0.5 m median hand/object contact distance. This is the correct failure signal: V3 needs a stronger hand/camera/depth estimation stage, not looser contact weights or a hidden smoothing correction.

### Metric-Depth Alignment Diagnostic

Implemented:

- `scripts/diagnose_metric_depth_alignment_v3.py`

The diagnostic compares the current MANO source-camera depth and object mesh source-camera depth against the independent Depth Anything V2 metric-depth map used for observed-surface meshing:

`/data2/ego_annotation_outputs/representative_trash/v3_metric_depth_alignment_840_930.json`

Result:

- rows: 182;
- high-confidence measured hand rows: 124;
- object mesh depth minus metric-depth median: -0.98 mm;
- high-confidence measured MANO depth minus metric-depth median: 165 mm;
- high-confidence measured MANO over metric-depth median ratio: 1.136;
- high-confidence measured MANO over metric-depth p95 ratio: 1.784.

Interpretation: the object mesh sits at the metric-depth surface because V2 meshed that surface, while MANO is systematically deeper than metric depth at the hand joint projections. This does not prove Depth Anything is metrically exact, but it localizes the current pink-lid contact conflict to MANO/camera-depth alignment more strongly than object-mesh depth. The next v3 component must refit measured MANO depth against metric depth and 2D keypoints before contact can become a physically meaningful factor.

### Independent 2D Hand-Keypoint Evidence

Implemented:

- `scripts/run_rtmlib_hand2d_v3.py`
- `scripts/diagnose_rtmlib_wilor_hand2d_v3.py`

RTMLib was run on the A800 host for frames 840 to 930:

`/data2/ego_annotation_outputs/representative_trash/v3_rtmlib_hand2d_840_930/`

Runtime note: the job was launched under tmux on the GPU server, but ONNXRuntime could not load the CUDA provider because `libcudnn.so.9` was missing from the runtime library path. RTMLib still completed with the ONNXRuntime backend and wrote 91-frame outputs. This affects speed, not the semantics of the 2D keypoint model output.

RTMLib output:

- processed frames: 91;
- frames with hands: 91;
- median hands per frame: 2;
- median RTMLib hand mean score: 0.403;
- overlay video: `rtmlib_hand2d_overlay.mp4`.

The first all-pairs comparison to WiLoR was misleading because correct and crossed hand pairings were mixed. The one-to-one association diagnostic gives the actual 2D agreement:

`/data2/ego_annotation_outputs/representative_trash/v3_rtmlib_hand2d_840_930/qc_rtmlib_wilor_association.json`

Result:

- frames: 91;
- frames with at least one RTMLib/WiLoR match: 65;
- frames with at least one match below 30 px median keypoint error: 54;
- matched hand instances: 113;
- good matched hand instances: 101;
- matched median keypoint delta: 16.3 px;
- good-match p95 keypoint delta: 21.8 px.

Visual review:

- frame 840 rejects RTMLib as a direct replacement because it marks the lower body/leg region as a low-confidence hand while the visible hand is near the can;
- frame 880 gives one useful left-hand match at 12.3 px median keypoint delta but misses the right hand;
- frames 903 and 930 give visually plausible two-hand RTMLib detections and one-to-one matches to WiLoR at roughly 9 to 22 px median keypoint delta.

Interpretation: RTMLib provides an independent live 2D hand-keypoint observation for many contact-window frames. It does not explain the hundreds-of-millimeters hand/object depth conflict, and it cannot be converted into contact constraints when detections are unmatched, low-confidence, or visually false. The next graph should use RTMLib and WiLoR as associated 2D evidence, while metric depth and contact terms own the depth disagreement.

### Hand Depth, Keypoint, and Contact Graph

Implemented:

- `scripts/optimize_hand_depth_keypoint_contact_v3.py`

The graph uses only hand observations with associated RTMLib/WiLoR 2D landmarks. Its state is deliberately small:

- one global hand scale;
- per-hand camera-ray depth shift;
- per-row object depth shift;
- no hidden fallback to predicted hands.

Factors:

- 2D keypoint reprojection to the WiLoR keypoints when RTMLib and WiLoR agree below 30 px;
- MANO joint depth against the metric-depth map;
- hand/object contact depth for hand vertices that project near the object mask;
- hand bone-scale prior;
- temporal smoothness on hand and object depth shifts.

Strict contact-supported run:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_keypoint_contact_840_930/qc_hand_depth_keypoint_contact_loose_near.json`

Result:

- observations: 3, all left-hand frames 886, 888, and 889;
- before fitting: keypoint reprojection median 14.5 px, MANO-minus-metric-depth median -62 mm, hand-object depth median 85 mm;
- depth-only fit: MANO-minus-metric-depth median 13 mm, but hand-object depth median worsens to 155 mm;
- contact fit: hand-object depth median becomes 3.2 mm, but object shift saturates the 80 mm bound and MANO-minus-metric-depth median remains -54 mm;
- status: `diagnostic_metric_depth_residual_remains`.

Matched-2D run with no required near-mask contact:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_keypoint_contact_840_930/qc_hand_depth_keypoint_contact_no_near_minrows1.json`

Result:

- observations: 6;
- before fitting: keypoint reprojection median 19.7 px, MANO-minus-metric-depth median 49 mm;
- depth-only fit: MANO-minus-metric-depth median 0.45 mm, but hand shifts reach the 150 mm bound;
- contact fit: contact-supported rows reach 2.4 mm median hand-object depth, but hand scale drops to 0.883, hand shifts reach the 150 mm bound, object shift reaches the 80 mm bound, and keypoint reprojection median rises to 20.6 px;
- status: `diagnostic_keypoint_reprojection_residual_too_large`.

Interpretation: independent 2D keypoints make the contradiction sharper. In rows where 2D hands are live, metric depth and contact can be made individually plausible, but the joint fit requires bounded depth shifts or loses keypoint quality. This rejects a Kalman-only or smoothing-only fix. The next v3 mechanism must estimate hand depth from stronger 3D evidence, external scale, or a richer MANO/depth/camera state before contact can serve as a physical regularizer.

HaWoR was also tested in the same diagnostic instead of assuming it was unusable from the earlier adapter failure.

Camera-local HaWoR contact-supported run:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_keypoint_contact_840_930/qc_hawor_camera_local_hand_depth_keypoint_contact.json`

Result:

- observations: 3;
- before fitting: keypoint reprojection median 26.2 px, MANO-minus-metric-depth median -385 mm, hand-object depth median -220 mm;
- contact fit: hand-object depth median becomes 3.9 mm, but hand scale hits the 1.15 upper bound, hand shift hits the 150 mm bound, and keypoint reprojection median rises to 28.1 px;
- status: `diagnostic_keypoint_reprojection_residual_too_large`.

Camera-local HaWoR without required near-mask contact:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_keypoint_contact_840_930/qc_hawor_camera_local_hand_depth_keypoint_contact_no_near.json`

Result:

- observations: 10;
- before fitting: keypoint reprojection median 29.0 px, MANO-minus-metric-depth median -284 mm;
- contact fit: MANO-minus-metric-depth median becomes -10.8 mm, but hand scale hits the 1.15 upper bound and hand shift reaches the 150 mm bound;
- status: `diagnostic_keypoint_reprojection_residual_too_large`.

The HaWoR translation-refit annotation produced zero contact-supported observations under the same matched-2D/contact criteria. HaWoR therefore does not close the v3 hand state on this slice. It is still useful evidence that a different hand backend changes the error direction: HaWoR is too shallow relative to metric depth, while WiLoR’s accepted contact rows have a smaller but still inconsistent depth/contact tradeoff.

### EgoForce Pose-Head Branch

EgoForce was tested because it directly targets egocentric camera-space hand pose, the missing quantity in the current failure. The diagnostic used annotation-derived hand boxes only as crops and disabled Kalman filtering:

`/data2/ego_annotation_outputs/representative_trash/v3_egoforce_posehead_840_930/`

Runtime repairs on the A800 host:

- downloaded and size-verified `_DATA/model_weights.pth` from the public EgoForce Hugging Face repository;
- copied MANO left/right pickle files from `/data/dex_home/yiwen/mano_assets/mano/models`;
- installed `chumpy` with no build isolation and restored legacy NumPy aliases before MANO pickle loading;
- bypassed unused EgoForce package imports that pulled in PyTorch3D and depth-model wrappers before the pose head could run;
- replaced EgoForce's PyTorch3D-dependent camera-space solve with the same pinhole ray-translation least-squares equation over the pose-head 2D/3D outputs and crop metadata.

Output:

- requested frames: 91;
- hand rows: 131;
- skipped rows/frames: 43;
- overlay video: `egoforce_posehead_overlay.mp4`, 83 frames at 1920 x 1080;
- median joint reprojection to the source observed keypoints: 127 px;
- p95 joint reprojection: 463 px.

The shared contact-reliability diagnostic gives:

`/data2/ego_annotation_outputs/representative_trash/v3_egoforce_posehead_840_930/qc_contact_reliability_bonescale.json`

Result:

- rows with object masks and meshes: 31;
- high-score measured rows: 27;
- reliable contact rows: 0;
- high-score median joint reprojection: 238 px;
- depth-consistent rows: 0;
- contact-consistent rows: 2, but these rows fail projection and depth checks;
- median hand bone scale: 163 mm.

Visual review of `overlay_probe_001.jpg`, `overlay_probe_002.jpg`, and `overlay_probe_003.jpg` confirms the metric failure: EgoForce points are displaced onto the lid, floor, or arm rather than the visible hand. This branch therefore does not supply usable contact evidence on the representative slice. Full EgoForce detector mode remains a separate test after the full detector setup finishes, because the pose-head diagnostic used annotation boxes and pseudo-arm boxes as crop evidence.

### MANO Metric-Depth Refit Probe

Implemented:

- `scripts/refit_mano_metric_depth_v3.py`

This probe uses measured hands with detector score at least 0.50. It scans one global MANO local-geometry scale and computes a bounded per-row center-ray shift to match metric-depth samples at WiLoR 2D joints:

`/data2/ego_annotation_outputs/representative_trash/v3_refit_mano_metric_depth_840_930.json`

Result:

- rows: 124;
- global hand scale: 0.9975;
- median MANO-minus-metric-depth residual: 154 mm to 0 mm;
- p95 absolute ray shift hits the 220 mm bound;
- median absolute ray shift: 170 mm;
- median 2D joint reprojection residual: 10.9 px to 13.3 px;
- median hand span changes by -0.4 mm.
- good-keypoint subset, defined by initial median reprojection at most 20 px: 90 rows;
- good-keypoint subset median depth residual: 144 mm to 0 mm;
- good-keypoint subset median 2D reprojection: 9.1 px to 11.5 px;
- good-keypoint subset p95 depth residual after refit: 140 mm.

Status: `diagnostic_mano_reprojection_residual_too_large`.

Interpretation: metric-depth evidence can pull MANO to the depth surface without shrinking the hand. On rows with good initial 2D keypoints, the median reprojection remains below the 12 px threshold after refit. The required shifts are still large, many rows hit the shift bound, and the good-keypoint p95 depth residual remains 140 mm. V3 needs a hand-depth estimator that models metric-depth reliability, hand occlusion, and temporal consistency before applying contact factors to the final annotation.

### Hand-Depth Reliability Diagnostic

Implemented:

- `scripts/diagnose_hand_depth_reliability_v3.py`

This diagnostic samples Depth Anything metric depth at measured hand joints and records local depth-patch stability, keypoint reprojection quality, and proximity to the accepted object mask:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_reliability_840_930.json`

Result:

- joint rows: 2604;
- good-keypoint rows: 1853;
- stable good-keypoint rows: 1740;
- stable good-keypoint MANO-minus-metric-depth median: 161 mm;
- stable good-keypoint depth-patch IQR ratio median/p95: 0.0035 / 0.0187;
- stable good points near the object mask: 198;
- stable good near-object MANO-minus-metric-depth median: 230 mm.

Interpretation: local metric-depth instability does not explain the main MANO-depth excess. Even stable depth patches with good 2D keypoints place current MANO substantially deeper than the depth surface. The near-object subset is worse, so contact/occlusion regions need special treatment, but the broad mechanism is a MANO/camera-depth alignment error rather than only edge noise in metric depth.

### Hand-Contact Reliability Diagnostic

Implemented:

- `scripts/diagnose_hand_contact_reliability_v3.py`

This diagnostic asks whether current WiLoR hands can be used as physical contact observations for the accepted pink-lid object mesh. A hand row is reliable only when the hand is measured, detector score is high, raw 2D keypoint reprojection is good, metric-depth samples at hand joints agree with MANO depth, local depth patches are stable, MANO bone scale is plausible, and near-mask hand vertices are already close to the object surface:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_contact_reliability_840_930.json`

Result:

- hand rows: 50;
- measured high-score hand rows: 27;
- reliable contact rows: 0;
- measured high-score median 2D keypoint reprojection: 23.6 px;
- measured high-score median MANO-minus-metric-depth residual: 257 mm;
- measured high-score median hand/object contact-depth gap: 115 mm, computed only on the 5 rows with near-mask contact samples;
- measured high-score median bone scale: 213 mm;
- measured high-score median fingertip spread: 101 mm;
- condition counts for measured high-score rows after the corrected bone-scale QC: 5 pass projection, 0 pass depth, 7 pass stable-depth, 27 pass bone scale, 0 pass contact.

Interpretation: the current WiLoR hand stream cannot provide valid contact factors in this slice. Detector score alone is misleading under occlusion. The original tip-spread size check was wrong because a grasping hand can be closed; the corrected bone-scale check shows that hand scale is not the active bottleneck. The failures are depth agreement and hand/object contact support.

### MANO Depth-Refit Candidate Render

Implemented:

- `scripts/apply_mano_depth_refit_v3.py`

This script creates a candidate annotation JSON by shifting only measured hands with good initial 2D reprojection toward metric-depth samples, then recomputes source-camera and world MANO vertices:

`/data2/ego_annotation_outputs/representative_trash/v3_mano_depth_refit_candidate_840_930/annotations_mano_depth_refit.json`

Rendered videos:

- `/data2/ego_annotation_outputs/representative_trash/v3_mano_depth_refit_candidate_840_930/render/overlay_mano_object.mp4`
- `/data2/ego_annotation_outputs/representative_trash/v3_mano_depth_refit_candidate_840_930/render/reconstruction_3d_world.mp4`
- `/data2/ego_annotation_outputs/representative_trash/v3_mano_depth_refit_candidate_840_930/render/side_by_side.mp4`

Result:

- applied hand corrections: 86;
- median applied shift: -155 mm;
- structural video check: 91 frames, overlay and reconstruction 960 by 540, side-by-side 1920 by 540;
- contact-depth median: 385 mm to 249 mm;
- high-confidence measured contact-depth median: 393 mm to 234 mm;
- high-confidence measured contact-depth p95 remains 598 mm.

Visual review:

- frame 858: image overlay remains plausible, but the 3D hands are still separated from the object mesh;
- frame 880: right-hand overlay collapses into a narrow vertical strip on the lid;
- frame 903: hands remain below or beside the lid in 3D.

Interpretation: depth translation alone is not an acceptable v3 annotation stage. It improves the median depth residual but does not produce contact-consistent MANO geometry and degrades some visible hand overlays. The next hand branch must refit MANO pose/translation jointly, or replace the hand backend with a model whose metric hand placement is better conditioned under egocentric occlusion.

### Hand Translation and MANO-Layer Refit Diagnostics

Implemented:

- `scripts/optimize_hand_translation_contact_v3.py`
- `scripts/refit_mano_pose_contact_v3.py`
- `scripts/optimize_hand_similarity_contact_v3.py`

The translation-only diagnostic uses the fused WiLoR local geometry as the source of truth and optimizes per-hand source-camera translation against 2D keypoints, metric depth, temporal smoothness, hand-size limits, and object contact depth.

Smoke result on frames 880 to 889:

- independent contact-reliability rows: 18;
- measured high-score rows: 12;
- reliable contact rows after translation refit: 0;
- measured high-score median 2D reprojection: 45.4 px;
- measured high-score median MANO-minus-metric-depth residual: 13.8 mm;
- measured high-score median bone scale after corrected QC: 164 mm;
- measured high-score median fingertip spread: 32.5 mm.

Interpretation: translation can reduce some depth residuals, but it does not create reliable contact evidence. The small fingertip spread is a closed-hand pose descriptor, not a size failure. The refit is rejected because projection support remains weak and contact rows still fail.

The MANO-layer pose/contact refit attempted to optimize MANO pose, global orientation, translation, and local scale from the saved `mano_params`.

Smoke result on frames 880 to 889:

- fit rows: 12;
- independent contact-reliability rows: 18;
- measured high-score rows: 12;
- reliable contact rows after pose refit: 0;
- measured high-score median 2D reprojection: 41.2 px;
- measured high-score median MANO-minus-metric-depth residual: -68.4 mm;
- measured high-score median contact gap on available near-mask rows: 160 mm;
- measured high-score median bone scale after corrected QC: 191 mm;
- measured high-score median fingertip spread: 112 mm.

This refit changes the visible hand pose but still fails contact and projection reliability. The cause is partly a representation mismatch: the fused annotations store WiLoR local geometry after a global scale and a source-camera translation solve, while `mano_params` remain in the raw WiLoR MANO frame. Reconstructing fused geometry from a plain SMPLX MANO layer and the saved params gives tens to hundreds of millimeters of geometry error. Using WiLoR's own MANO wrapper and the recovered raw-to-metric scale of 1.341 improves reproduction, but the right hand in frames 886 to 889 still has median joint errors from 14 mm to 61 mm.

Interpretation: saved `mano_params` are a useful pose prior, not the current source of truth for metric hand geometry. The next v3 hand solver must operate on the fused local vertex/joint stream or a stronger hand backend, and must treat tip spread as pose state rather than hand-size evidence.

The fused-geometry similarity refit then tested whether a local isotropic scale and source-camera translation can resolve the contact conflict.

Smoke result on frames 880 to 889:

- fit observations: 12;
- independent contact-reliability rows: 18;
- measured high-score rows: 12;
- reliable contact rows after similarity refit: 0;
- measured high-score median 2D reprojection: 44.5 px;
- measured high-score median MANO-minus-metric-depth residual: 13.8 mm;
- measured high-score median bone scale after corrected QC: 189 mm;
- measured high-score median fingertip spread: 38.4 mm;
- measured high-score contact gap on the one available near-mask row: 27.7 mm.

Interpretation: similarity refit improves depth and one contact-depth row, but still yields zero reliable contact rows because projection support and contact support remain weak. The limiting variable is the coupled image/depth/contact hand state, not scalar hand size.

## Implemented Diagnostics

The implemented v3 code is diagnostic, not the required solver above:

- `scripts/summarize_contact_depth_scale_v3.py`
- `scripts/optimize_contact_depth_scale_v3.py`
- `scripts/diagnose_hand_reprojection_depth_v3.py`
- `scripts/diagnose_mano_contact_reprojection_tradeoff_v3.py`
- `scripts/diagnose_metric_depth_alignment_v3.py`
- `scripts/refit_mano_metric_depth_v3.py`
- `scripts/diagnose_hand_depth_reliability_v3.py`
- `scripts/diagnose_hand_contact_reliability_v3.py`
- `scripts/apply_mano_depth_refit_v3.py`
- `scripts/optimize_hand_translation_contact_v3.py`
- `scripts/refit_mano_pose_contact_v3.py`
- `scripts/optimize_hand_similarity_contact_v3.py`
- `scripts/optimize_temporal_hand_contact_v3.py`
- `scripts/optimize_joint_depth_contact_v3.py`
- `scripts/optimize_object_factor_graph_v3.py`
- `scripts/optimize_joint_mano_object_graph_v3.py`
- `scripts/remote_setup_hawor.sh`
- `scripts/export_hawor_world.py`
- `scripts/adapt_hawor_to_annotations_v3.py`
- `scripts/adapt_hawor_camera_local_v3.py`
- `scripts/optimize_hand_rigid_contact_v3.py`
- `scripts/remote_setup_handdgp.sh`
- `scripts/run_handdgp_export_v3.py`

### HaWoR World-Hand Branch

HaWoR was run on the representative trash clip on the A800 GPU server through tmux:

`/data2/ego_annotation_outputs/representative_trash/v3_hawor_world/hawor_world_hands.npz`

Export QC:

- frames: 1050;
- image focal used by HaWoR: 2304;
- valid hand frames: 1049 left and 1049 right;
- HaWoR metric scale estimate: 0.872534;
- SLAM DBA errors reported by HaWoR: 1.098 and 0.797.

The raw HaWoR hand output has plausible surface extent before alignment:

- left-hand fingertip-spread median: 86 mm;
- right-hand fingertip-spread median: 79 mm;
- vertex bounding-box diagonal median: about 221 to 226 mm.

The camera-path Sim(3) alignment to the existing DROID-derived annotation world for frames 840 to 930 reached:

`/data2/ego_annotation_outputs/representative_trash/v3_hawor_world_adapted_840_930/qc_adapt_hawor_to_annotations_v3.json`

- alignment frames: 19;
- camera-position error median/p95/max: 14.9 mm / 26.4 mm / 30.5 mm;
- Sim(3) scale: 0.350569;
- adapted hands: 182;
- skipped hands: 0.

That camera alignment is not physically valid for the hands. It shrinks the HaWoR hand span to about 26 mm and gives hundreds of pixels of 2D error against the original observed keypoints. The shared contact-reliability diagnostic reports:

`/data2/ego_annotation_outputs/representative_trash/v3_hawor_contact_reliability_840_930.json`

- rows: 50;
- measured high-score rows: 27;
- reliable contact rows: 0;
- measured high-score median joint reprojection error: 481.5 px;
- measured high-score median fingertip spread after Sim(3): 26.2 mm;
- near-mask contact-gap median on available rows: -847.8 mm.

Alternative bridge checks do not rescue the branch. Raw HaWoR projection under HaWoR's own camera convention has a better median reprojection error of 25.8 px, but p95 remains 715.9 px. Whole-clip and near-window camera alignment variants still have hundreds of pixels of median reprojection error after adaptation. A reprojection-aware Sim(3) compromise improves median reprojection to about 39 px, but preserves a tiny 29 mm hand span and increases camera error to roughly 35 to 54 mm. Tightening the hand-size prior only raises the hand span to about 43 mm while pushing camera error toward 58 to 93 mm.

Interpretation: HaWoR is useful evidence, but it is not a direct v3 replacement for WiLoR in this clip. HaWoR's own hand-camera projection is partly plausible, while a single world-frame Sim(3) that aligns HaWoR cameras to the existing DROID trajectory makes the hands physically impossible. The next v3 implementation must optimize hand state in the target metric frame with explicit hand-size, 2D keypoint, metric-depth, temporal, and contact factors. It must not accept HaWoR contact factors through a hidden scale correction.

### HaWoR Camera-Local Branch

Implemented:

- `scripts/adapt_hawor_camera_local_v3.py`

The camera-local adapter preserves HaWoR's per-frame hand geometry in HaWoR camera coordinates, then uses the existing annotation camera pose only to place that camera-local hand in the DROID/object world. This avoids the global camera-path Sim(3) that shrank HaWoR hands.

Result on frames 840 to 930:

- adapted hands: 182;
- median hand bone scale: 165 mm;
- median fingertip spread: 76 mm;
- median camera depth: 1.01 m;
- median joint reprojection: 22.6 px over all adapted hands.

Corrected contact reliability:

- measured high-score rows: 27;
- reliable contact rows: 0;
- measured high-score median reprojection: 30.7 px;
- measured high-score median MANO-minus-metric-depth residual: -147 mm;
- measured high-score median contact gap on available near-mask rows: -137 mm;
- all measured high-score rows pass bone scale.

A translation-only refit of HaWoR camera-local hands improves metric-depth residual but still fails contact reliability:

- measured high-score median reprojection: 31.4 px;
- measured high-score median MANO-minus-metric-depth residual: -13 mm;
- measured high-score contact rows: only 3 available, with median gap -19 mm;
- reliable contact rows: 0.

Interpretation: HaWoR camera-local geometry is more plausible than the Sim(3) world bridge, but translation alone cannot satisfy projection, depth, and contact. The next solver needs at least per-frame rotation/depth/contact-state variables, or stronger 2D hand keypoints, before rendering a candidate.

### HaWoR Rigid Hand-State Probe

Implemented:

- `scripts/optimize_hand_rigid_contact_v3.py`

This probe adds per-frame hand rotation to the HaWoR camera-local translation variables. It uses reprojection, metric-depth, temporal, bone-scale, and object contact-depth factors.

Result on frames 840 to 930:

- observations: 27;
- median reprojection: 30.7 px to 20.2 px;
- median MANO-minus-metric-depth residual under the reliability rows: -147 mm to -24 mm;
- median contact gap on available measured high-score rows: -137 mm to -386 mm;
- median rotation correction: 0.36 rad;
- median translation correction: 167 mm;
- reliable contact rows after corrected reliability QC: 0.

Interpretation: rigid per-frame freedom improves some reprojection and depth residuals, but it worsens contact and uses large hidden corrections. This is a failed diagnostic, not a candidate for rendering. The next hand stage needs contact-state inference and robust keypoint selection, or a stronger hand keypoint backend, rather than looser rigid optimization.

### HandDGP Camera-Space Branch

Implemented:

- `scripts/remote_setup_handdgp.sh`
- `scripts/run_handdgp_export_v3.py`

HandDGP was tested as an independent camera-space hand mesh source because its DGP module solves global camera translation from cropped hand images and crop intrinsics. The adapter runs on measured WiLoR hand crops, transforms the source camera intrinsics into crop coordinates, runs the official FreiHAND checkpoint, writes camera-space vertices and joints into the annotation schema as diagnostic full-vertex hand geometry, and reuses the same contact reliability diagnostic.

Result on frames 840 to 930:

- exported hands: 124;
- skipped hands: 0;
- export median reprojection: 26.5 px;
- export p95 reprojection: 234.9 px;
- export median hand depth: 0.785 m.

Corrected contact reliability:

`/data2/ego_annotation_outputs/representative_trash/v3_handdgp_840_930/qc_contact_reliability_bonescale.json`

- reliability rows: 27;
- measured high-score rows: 27;
- reliable contact rows: 0;
- measured high-score median reprojection: 36.5 px;
- measured high-score median MANO-minus-metric-depth residual: -244 mm;
- measured high-score median contact gap on available near-mask rows: -330 mm;
- measured high-score median bone scale: 126 mm.

Interpretation: HandDGP does not solve the current v3 hand/object metric contradiction on this egocentric contact window. It places the hand shallower than metric depth and the object mesh, while the projection error remains above the contact-reliability threshold for nearly all rows. This branch strengthens the current diagnosis: replacing WiLoR with a generic camera-space hand mesh model is insufficient. HandDGP also does not provide MANO pose parameters, so it cannot satisfy the final MANO deliverable by itself. The missing mechanism is a clip-specific joint hand/depth/contact estimation stage or a more egocentric metric hand backend whose output passes the same residual checks.

### Temporal Hand Contact Graph Probe

Implemented:

- `scripts/optimize_temporal_hand_contact_v3.py`

This probe keeps the fused WiLoR local hand geometry and optimizes a temporal source-camera translation, velocity, and continuous contact probability for measured hands. It uses separate WiLoR and RTMLib 2D keypoint residuals, metric-depth residuals, object-depth contact residuals only for near-mask vertices, non-penetration residuals, bone-scale priors, and temporal motion/contact smoothness. It also checks the annotation representation contract before fitting: `joints3d_camera + cam_t` and `vertices_camera + cam_t` must match the fused source-camera geometry within 25 mm, or the row is rejected.

Smoke result on frames 880 to 889:

`/data2/ego_annotation_outputs/representative_trash/v3_temporal_hand_contact_split2d_880_889/qc_temporal_hand_contact.json`

- observations: 12;
- variables: 84;
- solver RMS: 7.60 to 5.59;
- median translation shift: 158 mm;
- median contact probability: 0.050 to 0.039;
- median WiLoR keypoint reprojection: 305 px to 46 px;
- median RTMLib keypoint reprojection on matched rows: 167 px to 45 px;
- median MANO-minus-metric-depth: -58 mm to 3.4 mm;
- median contact gap: 120 mm to 143 mm.

External contact reliability on the candidate:

`/data2/ego_annotation_outputs/representative_trash/v3_temporal_hand_contact_split2d_880_889/qc_contact_reliability_bonescale_after.json`

- rows: 18;
- measured high-score rows: 12;
- reliable contact rows: 0;
- measured high-score median reprojection: 46.4 px;
- measured high-score median MANO-minus-metric-depth: 14 mm;
- measured high-score contact gap: 13.7 mm on only one near-mask row;
- contact-ok rows: 0 because the near-mask support is below the reliability threshold.

Interpretation: the graph improves the depth residual without forcing contact. This is the right failure mode, because it exposes that the available 2D and near-mask contact support cannot justify a physically reliable contact factor. Temporal translation, velocity, and contact-state inference over the existing fused hand stream are therefore insufficient to close V3.

### Corrected V2-Mask Contact And Focal Sweep

Implemented:

- `scripts/merge_v2_object_masks_with_hands_v3.py`
- `scripts/diagnose_intrinsics_focal_sweep_v3.py`

The earlier contact diagnostics used the older full-annotation object track. For the V2 pink-lid mesh, the object source of truth is:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_pink_lid_masks/annotations_plan_masks.local.json`

The merge script combines that V2 object-mask source with the WiLoR hand stream:

`/data2/ego_annotation_outputs/representative_trash/v3_v2pink_masks_wilor_hands_merged.json`

Corrected contact reliability for frames 840 to 930:

`/data2/ego_annotation_outputs/representative_trash/v3_v2pink_wilor_contact_reliability_840_930.json`

- rows: 182;
- measured high-score rows: 124;
- reliable contact rows: 0;
- measured high-score median joint reprojection: 10.9 px;
- measured high-score median MANO-minus-metric-depth residual: 173 mm;
- measured high-score median hand-lid contact gap: 269 mm;
- measured high-score contact-ok rows: 1.

This corrects a source-mixing error in the diagnostic workflow. The bad mask seen in the old contact probe came from the stale white-bag track, not from the V2 pink-lid mask source. Visual probes with the V2 masks show that the pink-lid observed-surface mask is semantically correct in frames 857 and 886. Those frames still do not give reliable physical contact evidence: the right hand interacts with liner or rim context, and the left-hand MANO fit near the lid edge is not reliable enough to serve as a contact factor.

Focal-length sweep:

`/data2/ego_annotation_outputs/representative_trash/v3_v2pink_intrinsics_focal_sweep_fine_840_930.json`

- tested focal range: 1800 to 2400 px with principal point 960, 540;
- best median contact gap focal: 1800 px, with median contact gap -1.4 mm;
- at 1800 px, reliable contact rows remain 0, depth-ok rows are 5 out of 124, and contact-ok rows are 1 out of 124;
- at 2304 px, reliable contact rows remain 0, measured high-score median contact gap is 269 mm, and measured high-score median depth residual is 173 mm.

Interpretation: focal length is a real sensitivity, but focal-only correction does not close V3. Lowering focal can align the median hand-lid depth gap, yet it fails the per-row reliability tests. The missing mechanism is a joint object-context and hand-state model that represents which surface is being contacted: lid, rim, liner, or no contact.

## Immediate Execution Plan

1. Build a multi-surface object-context annotation for the pink-lid window: lid, rim, liner, and visible non-contact states must be model-produced masks or verified VLM/SAM outputs, then consumed by one uniform geometry path.
2. Run the contact reliability diagnostic per surface. Contact factors may activate only for rows that pass 2D projection, metric-depth, bone-scale, and near-surface support on the same surface.
3. Add a MANO-parametric temporal refit only after the contacted surface is identified. The refit must preserve the annotation representation contract and must abort if regenerated local joints/vertices disagree with the stream.
4. Keep SAMWISE/SAM3-style referring segmentation as the white-liner recovery branch. Any recovered liner masks must pass visual or VLM verification before meshing.
5. Render a candidate only after reliability rows become nonzero under the corrected V2-mask/object-context diagnostics.
