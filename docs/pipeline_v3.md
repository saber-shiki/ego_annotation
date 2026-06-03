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

Interpretation:

Object-complete mesh priors are feasible, but mesh alignment alone cannot fix hand/object contact while MANO/camera/depth scale disagree.

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

This diagnostic asks whether current WiLoR hands can be used as physical contact observations for the accepted pink-lid object mesh. A hand row is reliable only when the hand is measured, detector score is high, raw 2D keypoint reprojection is good, metric-depth samples at hand joints agree with MANO depth, local depth patches are stable, MANO hand span is plausible, and near-mask hand vertices are already close to the object surface:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_contact_reliability_840_930.json`

Result:

- hand rows: 50;
- measured high-score hand rows: 27;
- reliable contact rows: 0;
- measured high-score median 2D keypoint reprojection: 23.6 px;
- measured high-score median MANO-minus-metric-depth residual: 257 mm;
- measured high-score median hand/object contact-depth gap: 115 mm, computed only on the 5 rows with near-mask contact samples;
- measured high-score median hand span: 101 mm;
- condition counts for measured high-score rows: 5 pass projection, 0 pass depth, 7 pass stable-depth, 9 pass span, 0 pass contact.

Interpretation: the current WiLoR hand stream cannot provide valid contact factors in this slice. Detector score alone is misleading under occlusion. The graph must either use a stronger egocentric world-hand backend or refit MANO pose/translation with 2D keypoints, metric depth, hand-size priors, and temporal constraints before contact residuals become physical evidence.

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
- `scripts/optimize_joint_depth_contact_v3.py`
- `scripts/optimize_object_factor_graph_v3.py`
- `scripts/optimize_joint_mano_object_graph_v3.py`
- `scripts/remote_setup_hawor.sh`
- `scripts/export_hawor_world.py`
- `scripts/adapt_hawor_to_annotations_v3.py`

### HaWoR World-Hand Branch

HaWoR was run on the representative trash clip on the A800 GPU server through tmux:

`/data2/ego_annotation_outputs/representative_trash/v3_hawor_world/hawor_world_hands.npz`

Export QC:

- frames: 1050;
- image focal used by HaWoR: 2304;
- valid hand frames: 1049 left and 1049 right;
- HaWoR metric scale estimate: 0.872534;
- SLAM DBA errors reported by HaWoR: 1.098 and 0.797.

The raw HaWoR hand output has plausible hand size before alignment:

- left-hand joint-span median: 86 mm;
- right-hand joint-span median: 79 mm;
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
- measured high-score median hand span: 26.2 mm;
- near-mask contact-gap median on available rows: -847.8 mm.

Alternative bridge checks do not rescue the branch. Raw HaWoR projection under HaWoR's own camera convention has a better median reprojection error of 25.8 px, but p95 remains 715.9 px. Whole-clip and near-window camera alignment variants still have hundreds of pixels of median reprojection error after adaptation. A reprojection-aware Sim(3) compromise improves median reprojection to about 39 px, but preserves a tiny 29 mm hand span and increases camera error to roughly 35 to 54 mm. Tightening the hand-size prior only raises the hand span to about 43 mm while pushing camera error toward 58 to 93 mm.

Interpretation: HaWoR is useful evidence, but it is not a direct v3 replacement for WiLoR in this clip. HaWoR's own hand-camera projection is partly plausible, while a single world-frame Sim(3) that aligns HaWoR cameras to the existing DROID trajectory makes the hands physically impossible. The next v3 implementation must optimize hand state in the target metric frame with explicit hand-size, 2D keypoint, metric-depth, temporal, and contact factors. It must not accept HaWoR contact factors through a hidden scale correction.

## Immediate Execution Plan

1. Implement a MANO-layer refit on the 840 to 930 contact window. Variables: per-hand MANO translation, global hand scale, limited pose correction if the MANO layer is available in the active environment, and temporal velocity state. Observations: raw 2D keypoints, metric-depth samples at reliable joints, existing MANO pose prior, hand-size prior, object mesh non-penetration, and contact attraction only for image-supported contact rows.
2. Produce a candidate annotation JSON and rerun `scripts/diagnose_hand_contact_reliability_v3.py`. The candidate is accepted only if reliable contact rows become nonzero, hand spans are plausible, and projection/depth/contact residuals pass the documented thresholds.
3. Render the candidate videos only after the reliability diagnostic is not already falsified.
4. Keep SAMWISE as the parallel white-liner perception branch: run `scripts/run_samwise_referring_masks.py` on frames 678 to 918 only after setup is verified in tmux, then visually reject or accept masks before meshing.
