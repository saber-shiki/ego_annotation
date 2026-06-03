# Pipeline V2: VLM-Planned Object Mesh Reconstruction

## Logic

V1 established full-frame plumbing with DROID camera poses, WiLoR MANO hands, captions, object masks, and side-by-side rendering. Its object stage still encoded visual variation in hand-written category logic and rendered object geometry as a proxy. That failed the central requirement: manipulated objects must be reconstructed as geometry, and visual object selection must be model-produced.

V2 replaces that object path:

1. A VLM reads sampled video frames and action metadata, then writes an object plan with track IDs, descriptions, open-vocabulary prompts, active intervals, and physical notes.
2. OWLv2 proposes boxes from the VLM prompts; SAM produces masks from those boxes.
3. Visual QC and optional VLM mask verification reject masks that cover a different planned object or background.
4. Depth Anything V2 metric indoor estimates dense per-frame depth for verified mask frames.
5. The mesh stage back-projects verified mask pixels through the camera intrinsics and head-camera pose to build a dynamic observed-surface mesh in world coordinates. Each mesh archive now uses one declared depth source so DROID-relative depth and monocular metric depth are not mixed inside the same object track.
6. Contact-depth correction is an explicit ablation with reported shift values. The default V2 mesh archive leaves monocular depth unchanged, then reports the resulting hand-mesh distances.
7. The renderer draws MANO hands, head camera frustum, trajectory, and the object mesh in the world reconstruction panel.

V2 scope: reconstruct the observed surface for a manipulated object and show contact-frame object geometry. V3 scope starts at complete watertight geometry and a single object-centric mesh state across the whole clip.

## Implemented Components

- `scripts/build_object_plan_vlm.py`: calls the OpenAI Responses API with sampled frames and action metadata, returning a structured object plan.
- `scripts/build_object_point_prompts_vlm.py`: asks a VLM for positive and negative SAM point prompts on selected target-object frames, in image coordinates.
- `scripts/segment_object_plan_v2.py`: runs plan-driven OWLv2 plus SAM and writes full-timeline annotations with object masks.
- `scripts/segment_object_points_v2.py`: runs SAM from VLM point prompts when detector boxes localize the wrong object.
- `scripts/verify_plan_masks_vlm.py`: verifies proposed masks against the target object description using a VLM review sheet.
- `scripts/estimate_metric_depth_v2.py`: runs Depth Anything V2 metric indoor on measured object-mask frames and stores dense metric depth maps.
- `scripts/reconstruct_object_mesh_v2.py`: builds a per-frame dynamic mesh from masks, one selected depth source, and head-camera pose; optional contact-depth correction is disabled by default and reported when enabled.
- `scripts/refine_mask_by_vlm_points_depth_v3.py`: filters point-prompt masks by connected components and metric-depth compatibility with VLM positive points. This is a diagnostic refinement, not a substitute for visual tracking.
- `scripts/run_sam2_vlm_points_track.py`: propagates a target object through video using SAM2 from VLM point prompts on clean seed frames.
- `scripts/fuse_v1_full_fidelity.py`: renders `--object-mesh-npz` archives in the 3D world panel.

## Representative Trash Clip

Clip:

`/data2/egoscale_demo_30h/egoscale_tasks/20260108_1057_Recf94e_P0_S994da4_task_9/20260108_1057_Recf94e_P0_S994da4_task_9.mp4`

VLM plan:

`/data2/ego_annotation_outputs/representative_trash/v2_object_plan/object_plan_vlm.json`

The VLM identified four tracks:

- `black_trash_bag`
- `white_trash_bag`
- `off_white_trash_can_first`
- `pink_lid_trash_can_second`

### Failed White-Bag Track

The white-bag segmentation run processed 471 planned frames and wrote masks for all 471 frames:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_white_bag_masks/`

Visual QC rejected this result. The masks selected the pink lid in frames 880 to 918 and a wall/door-like surface around frame 534. The detector score and detection rate were live signals for prompt match; the review sheet showed identity drift. This mask set is quarantined, and mesh reconstruction consumes accepted object masks only.

Review sheet:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_white_bag_masks/review_sheets/white_bag_plan_masks_000_918.jpg`

### White-Liner Point-Prompt Recovery

The white liner is the manipulated deformable object in the second half of this clip. The rejected OWLv2-box masks did not support object mesh reconstruction, so v3 adds a model-driven point-prompt path.

VLM point prompts:

`/data2/ego_annotation_outputs/representative_trash/v3_white_liner_point_prompts/object_point_prompts_vlm.json`

Selected prompted frames:

- 534, 602, 671, 678, 720, 797, 858, 880, 900, 918

The VLM placed positive points on visible liner sheet, rim, or folds and negative points on hands, the pink lid, the can interior, floor, wall, plant, and clothing.

SAM point masks:

`/data2/ego_annotation_outputs/representative_trash/v3_white_liner_points_sam/`

The point masks are semantically better than the OWLv2-box masks. VLM verification accepted all 10 selected masks as the white-liner track:

`/data2/ego_annotation_outputs/representative_trash/v3_white_liner_points_vlm_verify/qc_vlm_mask_verification.json`

Visual inspection still found spillover: frame 720 included floor glare and clothing fragments, and frame 918 covered the pink lid. Metric-depth component filtering kept 7 of 10 prompted frames but did not fix frame 918 because the false lid/background component had compatible monocular depth:

`/data2/ego_annotation_outputs/representative_trash/v3_white_liner_points_depth_refined/qc_depth_refined_masks.json`

This falsifies depth-only cleanup for translucent liner masks. The next valid path is video identity tracking or a stronger referring segmentation model, not meshing the raw point masks.

SAM2 propagation from VLM point seeds:

`/mnt/user-home/yiwen/ego_annotation_remote/outputs/v3_white_liner_sam2_points_678_918/`

The 4090 server run used seed frames 678, 720, 797, and 858, and completed 241 frames from 678 to 918. QC reports 241 visible frames. Local visual review is still pending because SSH transfer from the 4090 host became unstable after the run. This remote result is evidence that the video propagation stage executes; it is not accepted mask evidence until selected propagated masks are pulled and inspected.

### Accepted Pink-Lid Track

The pink-lid/trash-can track is visibly manipulated and the mask sheet matches the target object.

Mask output:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_pink_lid_masks/`

Mask QC:

- planned frames processed: 372
- detected frames: 372
- target object: `pink_lid_trash_can_second`
- active frame span: 678 to 1049

Review sheet:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_pink_lid_masks/review_sheets/pink_lid_plan_masks_678_1049.jpg`

The accepted sheet shows a consistent mask on the pink lid/trash-can assembly from approach through close-up handling.

## Mesh Reconstruction

DROID-only mesh reconstruction produced a real observed-surface mesh, but it missed the most important contact interval because DROID keyframes jumped from frame 818 to frame 980. That left frames 840 to 930 without nearby DROID depth.

DROID-only mesh QC:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh/qc_object_mesh_v2.json`

- mesh frames: 186
- valid vertices: 74,638
- valid triangles: 126,956
- missing near-DROID-keyframe frames: 153
- hand-mesh distance median: 0.284 m

Depth Anything V2 metric indoor filled that observability gap:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_metric_depth/qc_metric_depth_v2.json`

- selected mask frames: 372
- dense depth resolution: 960 by 540
- depth median: 1.109 m
- depth p05/p95: 0.372 m / 2.234 m

An adversarial review exposed that the earlier metric-depth archive mixed 186 DROID-depth frames with 153 metric-depth frames. The corrected archive uses only `metric_depth`.

Strict metric-depth mesh QC:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict/qc_object_mesh_v2.json`

- depth source: `metric_depth`
- mesh frames: 372
- valid vertices: 550,893
- valid triangles: 1,047,857
- frames outside the object interval: 678
- hand-mesh distance frame count: 281
- hand-mesh distance median over frames with hands: 0.202 m
- hand-mesh distance p05/p95 over frames with hands: 0.0049 m / 0.572 m
- worst hand-mesh distance: 0.832 m at frame 817

Bounded contact-depth ablation:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict_contact03/qc_object_mesh_v2.json`

- contact-depth correction cap: 0.03 m
- corrected frames: 159
- median reported shift: 0.03 m
- hand-mesh distance median over frames with hands: 0.201 m
- hand-mesh distance p95 over frames with hands: 0.566 m

The ablation saturated the 30 mm cap on most corrected frames and barely changed the distribution. That falsifies a simple global contact-depth shift as the scale/contact fix.

For the contact window 840 to 930:

- mesh frames: 91
- hand-mesh distance median: 0.092 m
- hand-mesh distance p05/p95: 0.0015 m / 0.544 m
- worst distance in this window: 0.597 m at frame 875
- frames 848 and 885 have near-contact distances below 10 mm, while frames 840 to 847 and 875 show large hand-object depth disagreement

## Current Deliverable Slice

Contact-window side-by-side render:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict_render_840_930/side_by_side.mp4`

Frame count and size:

- side-by-side: 91 frames, 30 fps, 1920 by 540
- overlay: 91 frames, 30 fps, 960 by 540
- 3D reconstruction: 91 frames, 30 fps, 960 by 540

Inspected stills:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict_render_840_930/review_stills/`

Visual inspection:

- frame 840: the 2D mask is correct, but the metric-depth surface is visibly separated from the hands in the world panel;
- frame 858: the object mask and observed surface are plausible, while the left hand remains separated in 3D;
- frame 875: the 2D mask covers the lid, but the observed surface is far from both hands;
- frame 880: the surface stays present and detailed, with remaining depth mismatch against the MANO hands.

## Evidence Status

The current v2 result supports these mechanisms on one representative non-kitchen clip:

- VLM object planning can identify manipulated object tracks.
- Open-vocabulary detection plus SAM can produce correct object masks when the target is visually unambiguous.
- The same path can fail when the prompt is ambiguous, as shown by the rejected white-bag masks.
- Dense metric monocular depth supplies full mask-interval coverage when DROID keyframes skip the contact interval.
- Dynamic observed-surface mesh reconstruction gives a real object mesh in the world panel.
- The corrected strict run exposes the open scale/contact problem instead of hiding it: the hand-mesh distance distribution remains far above the 5 mm target.

Evidence still required:

- complete mesh reconstruction for the full object, including the unseen backside;
- a single temporally consistent object-centric mesh identity;
- deformable white-bag reconstruction;
- external scale and ground-truth-style validation before any absolute 5 mm claim;
- joint optimization of depth scale, hand pose, camera pose, object pose, and contact state;
- physical force consistency with explicit force, mass, inertia, and object acceleration estimates.

## V3 Design Direction

V3 should make the object the state variable across the clip, beyond per-frame observed surfaces.

Masking upgrade:

- Use referring video segmentation or mask tracking after VLM object selection. Good candidates are Grounded-SAM variants, Florence-2/Florence-style referring detection, SAM2 video propagation, XMem/Cutie-style memory tracking, and VLM verification for ambiguous frames.
- The current white-bag failure is the test case for this upgrade: a detector confidence score can stay high while object identity drifts. The v3 mask stage should carry identity through time and use VLM review sheets to reject drift.

Mesh-prior upgrade:

- Generate a complete object mesh from a clean object crop and mask. SAM 3D Objects is the best matched current model because its input contract is image plus mask and its output is full 3D object geometry, texture, and layout for cluttered natural images. Its public setup requires Hugging Face checkpoint access and a GPU with at least 32 GB VRAM.
- Use TripoSR as the immediate public executable mesh-prior probe. It produces a complete single image mesh quickly and runs on 4090 class GPUs. TripoSR consumes an isolated object image; SAM 3D consumes the full scene plus mask.
- Treat the generated mesh as a prior. The optimized object state must fit multi-frame mask silhouettes, metric depth surfaces, DROID camera poses, and MANO contact evidence.

Factor-graph upgrade:

- Variables: camera poses, MANO hand states, object pose, complete object mesh or deformation state, per-frame contact state, and depth/scale corrections.
- Vision factors: silhouette overlap between rendered mesh and verified masks, depth residuals against observed mask-depth surfaces, temporal pose/deformation smoothness, and object identity consistency across mask-track embeddings or VLM verdicts.
- Contact factors: non-penetration between MANO mesh and object mesh, contact attraction only for observed or inferred contact states, and contact persistence during grasp-like phases.
- Force and acceleration factors should enter after object mass, inertia, and contact mode are explicit variables. Before that point, contact residuals constrain geometry while force claims remain underdetermined.

Depth/contact diagnostic:

- `scripts/diagnose_contact_depth_conflict_v3.py` measures the camera-frame depth of the observed object mesh and the MANO vertices that project near the object mask.
- On frames 840 to 930, 84 of 91 frames have hand vertices near the mask.
- Median hand-minus-object depth gap for those near-mask vertices is 0.385 m; p95 is 0.667 m.
- This explains why object pose optimization alone cannot enforce contact: the visual mask says the hand and object overlap in 2D, while the current MANO and monocular metric-depth estimates place them far apart in camera depth.

Current v3 execution target:

1. Use frame 858 of the trash clip as the first mesh-prior test because the accepted mask is clean and both hands interact with the pink lid.
2. Run TripoSR on the accepted crop to obtain a complete mesh prior.
3. Align that mesh to the frame-858 metric-depth object surface using similarity ICP plus silhouette scale initialization.
4. Extend alignment through frames 858 to 930 with an optimizer that can test whether shared mesh geometry explains multiple observed surfaces and MANO contact evidence.
5. Render the aligned complete mesh together with the v2 observed-surface mesh to expose failure modes visually.

## V3 First Evidence

Input crop and mask:

`/data2/ego_annotation_outputs/representative_trash/v3_mesh_prior_triposr_input/`

The crop comes from source frame 858 and uses the accepted pink-lid mask. The object fills a 478 by 478 neutral-background crop.

TripoSR execution:

- model: `stabilityai/TripoSR`
- server: `192.168.11.220`, A800 GPU, launched through `tmux`
- local mesh output: `/data2/ego_annotation_outputs/representative_trash/v3_mesh_prior_triposr_frame858/0/mesh.obj`
- mesh size: 28,417 vertices and 56,696 faces

The A800 run was needed because `torchmcubes` requires a CUDA toolkit for build. The 4090 server had Python and torch but lacked `nvcc`; the A800 server had `/usr/local/cuda` and built the dependency. The TripoSR run also required installing `onnxruntime` because `rembg` imports it even when `--no-remove-bg` is used.

Frame-858 prior alignment on the strict metric-depth surface:

`/data2/ego_annotation_outputs/representative_trash/v3_mesh_prior_aligned_frame858_strict/qc_align_mesh_prior_v3.json`

- observed surface: 1,849 vertices and 3,539 faces
- aligned complete prior: 28,417 vertices and 56,696 faces
- prior-to-observed median distance: 10.8 mm
- observed-to-prior median distance: 8.9 mm
- prior-to-observed p95 distance: 29.2 mm
- observed-to-prior p95 distance: 34.4 mm

Tripanel visual review:

`/data2/ego_annotation_outputs/representative_trash/v3_mesh_prior_aligned_frame858_strict/alignment_review_tripanel.png`

The review shows a round complete prior aligned over a wider irregular observed surface. The numeric surface distance is lower after the strict metric-depth rebuild, but the visual still indicates that the observed mask-depth surface includes more than the compact lid prior. V3 must separate manipulated object identity from adjacent support/trash-can geometry before treating this as object pose.

Window optimization prototype:

`/data2/ego_annotation_outputs/representative_trash/v3_mesh_prior_window_858_930/qc_optimize_mesh_prior_window_v3.json`

- window: frames 858 to 930
- used measured mesh frames: 73
- variables: 7 shared similarity parameters
- residual RMS: 3.10 to 1.40
- status: hit `max_nfev=80`
- observed-to-prior median distances: 24.5 to 55.9 mm, median 35.0 mm
- hand-to-prior minimum distances: 0.36 to 39.9 mm, median 2.28 mm

Window visual review:

`/data2/ego_annotation_outputs/representative_trash/v3_mesh_prior_window_858_930/window_alignment_review_tripanel.png`

The window optimizer is a failed prototype. The tripanel shows the complete prior inflated into a large oval around the observed surface. The objective reduced numeric residuals by changing scale and rotation without preserving the visible object identity. This failure identifies the next v3 requirement: silhouette rendering factors, per-frame object pose variables, bidirectional surface terms, depth-scale variables, and explicit separation between the manipulated lid and nearby support/trash-can geometry.

Per-frame pose factor-graph probe:

`/data2/ego_annotation_outputs/representative_trash/v3_factor_graph_858_880_depth_probe/qc_object_factor_graph_v3.json`

- window: frames 858 to 880
- variables: 161, including per-frame rotation, translation, and camera-axis depth offset
- status: hit `max_nfev=35`
- residual RMS: 3.22 to 2.71
- observed-to-prior median surface distance: 58.4 mm to 19.2 mm
- prior-to-observed median surface distance: 62.1 mm to 18.6 mm
- contact median distance: 566 mm to 577 mm
- contact p95 distance: 598 mm to 623 mm
- depth-axis offsets reached 0.39 m

This probe improves mesh-to-surface fit but leaves contact wrong. That result shifts the next v3 implementation from object-pose-only optimization to joint scale/depth optimization over MANO, object depth, and camera trajectory. The current data says the hand-object contact conflict is upstream of object mesh fitting.
