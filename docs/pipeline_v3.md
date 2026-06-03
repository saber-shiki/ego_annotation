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

A lightweight 1D contact-depth diagnostic then solved only the depth gap for frames with at least 80 near-mask hand vertices. It reduced median corrected depth gap to 0.16 mm, but the inferred correction was not small:

- median hand-depth scale: 0.939
- median object-depth scale: 1.217
- p95 object-depth scale: 1.581
- median object-depth shift: 43 mm

This is useful as a causal diagnostic, not as an annotation result. It shows that contact can be made numerically true only by exposing scale and shift corrections as explicit variables with priors. A hidden correction would destroy the evidence about which subsystem is wrong.

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

SAMWISE is the next executable fallback because it is a CVPR 2025 text-driven video segmentation model built on SAM2 and provides arbitrary-video/frame-folder inference from natural-language prompts.

Implemented preparation:

- `scripts/run_samwise_referring_masks.py`
- `scripts/remote_setup_samwise.sh`

Planned run:

- input: frames 678 to 918 of the representative trash clip
- prompt: `translucent white plastic trash bag liner being opened and placed inside the small trash can`
- expected output: binary mask PNGs, overlay video, review stills at 678, 720, 797, 858, 880, 900, 918, and QC JSON

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

## Factor Graph

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
- temporal smoothness on camera, hands, object pose, and deformation;
- prior terms on physically plausible hand size, object rigidity/deformation, and depth scale.

The graph must expose residual conflicts. A low object-depth residual with a 0.4 m hand/object depth gap is a failed joint annotation, not a success.

Implemented diagnostics:

- `scripts/summarize_contact_depth_scale_v3.py`
- `scripts/optimize_contact_depth_scale_v3.py`

## Immediate Execution Plan

1. Restore GPU-server connectivity and inspect any existing `ego_samwise_setup` tmux session before launching duplicate work.
2. Complete SAMWISE setup in tmux on a GPU host and verify the checkpoint file.
3. Run `scripts/run_samwise_referring_masks.py` on frames 678 to 918.
4. Pull review stills and mask QC; visually reject or accept before meshing.
5. If SAMWISE masks pass, reconstruct a white-liner observed-surface mesh and test whether depth/contact conflict resembles the pink-lid case.
6. If SAMWISE masks fail, move to SOLA or image-level referring segmentation instead of writing visual if/else cleanup.
7. Implement a v3 joint optimizer over a short pink-lid contact window using the existing strict metric mesh and contact-depth report, with explicit residual reporting for reprojection, depth, silhouette, penetration, and contact.
