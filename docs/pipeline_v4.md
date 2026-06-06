# Pipeline V4: Residual-Gated Object Tracks and Temporal Completion

## V3 Closure State

V3 now has two mesh-backed representative results:

- Trash/lid, frames 858 to 880: 23-frame measured object mesh, MANO hands, contact rows, full-hand nonpenetration, overlay video, standalone 3D world animation, and side-by-side presentation.
- Wild-rice stem, frames 2531 to 2537: seven-frame continuous mesh-backed evidence window with the same deliverable types and the same geometry/contact/SDF checks.

The remaining limitation is object-track completeness under ambiguous visual evidence. The wild-rice branch exposed two different causes:

- some frames need temporal completion because the manipulated object is partially occluded or identity-ambiguous;
- some frames were lost before completion because hand-written component-count pruning rejected VLM-accepted masks.

V4 therefore separates residual-gated measured recovery from temporal completion. A model-selected mask should first be tested against rendered depth, silhouette, hand contact, and SDF residuals. Completion is introduced only for frames where model evidence is absent or fails those residual checks.

## Research-Backed Design Decision

The strongest direct research target is ForeHOI: it is designed for daily hand-object interaction videos and combines 2D mask inpainting with 3D shape completion. Its public repository is reachable, but the README currently marks inference and training code as unreleased. ForeHOI should shape the V4 target contract, but V4 cannot depend on unreleased inference code.

Released components support a practical V4:

- SAM2, Cutie, and DEVA-style video object segmentation and memory propagation can propose temporally complete mask tracks.
- Existing VLM point-prompt and SAM2 candidate selection code in this repo can provide per-frame mask hypotheses and rejection evidence.
- V3 measured-sheet reconstruction can convert accepted masks and metric depth into watertight object meshes.
- V3 z-buffer, mesh-surface contact, selected-contact SDF, full-hand SDF, and visual review already falsify wrong geometry.

V4 should therefore make object-track construction a residual-gated map problem. Category-specific rule systems and single-frame generative replacements remain invalid for this project.

Source links checked during V4 design:

- ForeHOI repository: `https://github.com/Tao-11-chen/ForeHOI`
- ForeHOI project page: `https://tao-11-chen.github.io/project_pages/ForeHOI/`
- SAM2 repository: `https://github.com/facebookresearch/sam2`
- Cutie repository: `https://github.com/hkchengrex/Cutie`
- DEVA repository: `https://github.com/hkchengrex/Tracking-Anything-with-DEVA`
- ArtHOI project page: `https://arthoi-reconstruction.github.io/`

## State

For each manipulated object track, V4 keeps:

- per-frame object masks with model identity, confidence, and source provenance;
- per-frame measured surface sheets from accepted mask/depth evidence;
- a canonical object map in metric object coordinates;
- per-frame object pose and optional low-dimensional deformation;
- per-frame completion confidence, split into measured, propagated, and hallucinated regions;
- MANO hand pose, scale, and per-hand measurement confidence;
- head-camera pose and metric intrinsics/depth source reliability;
- contact state per hand region and object surface region.

The downstream geometry code stays category-agnostic. Visual differences enter as model-produced masks, tracks, depths, confidences, and captions.

## Edges and Objective

The V4 graph minimizes a robust weighted sum of residuals:

- mask and silhouette residuals between rendered object mesh and model-produced object masks;
- z-buffer residuals between rendered mesh depth and metric-depth observations on measured regions;
- surface fusion residuals between the canonical map and per-frame measured sheets;
- temporal object pose and deformation smoothness;
- mask-track identity residuals from SAM2/Cutie/DEVA memories or VLM-selected track hypotheses;
- MANO 2D reprojection and metric-depth residuals;
- hand-object nonpenetration SDF residuals over full MANO surface samples;
- contact equality residuals only for contact states supported by image/depth evidence and temporal continuity;
- contact force-motion consistency residuals only when contact is active and object acceleration is observable;
- completion-prior residuals from learned object completion, weighted lower than measured depth/silhouette evidence.

The graph should use robust losses and explicit measurement weights. A frame with unsupported object evidence should increase uncertainty or fail completion QC; it should not silently copy a neighboring mesh.

## Implementation Plan

1. Build a temporal object-track hypothesis module.
   - Inputs: VLM points, SAM2 image masks, SAM2/Cutie/DEVA propagation masks when available, existing captions, and previous accepted V3 masks.
   - Output: one uniform per-frame mask hypothesis table with mask source, confidence, model identity, frame index, and rejection reason when no mask is accepted.
   - Acceptance: visual contact sheets plus z-buffer feasibility after measured-sheet reconstruction.

2. Add measured-sheet temporal map fitting.
   - Inputs: accepted masks, UniDepth/VGGT intrinsics, camera poses, and per-frame measured sheets.
   - Output: canonical object map plus per-frame pose/deformation and uncertainty.
   - Acceptance: measured frames preserve V3 z-buffer p95 below 5 mm where depth evidence is clean, and no measured frame gets worse than the per-frame sheet baseline without an explained residual tradeoff.

3. Add missing-frame completion.
   - First implementation: map-propagated mesh poses with uncertainty and silhouette/depth checks against propagated masks.
   - Optional learned prior path: integrate released completion code if available; ForeHOI becomes first choice once inference is released.
   - Acceptance: completed frames must pass rendered-mask consistency, temporal motion plausibility, and hand nonpenetration. They remain labeled as completed, not measured, in deliverables.

4. Add contact and motion consistency.
   - Use V3 contact rows for active contact detection.
   - Add non-contact clearance for full MANO samples.
   - Add force-motion checks only when object acceleration and contact normals are observable enough to make the residual meaningful. Low-observability frames should carry uncertainty rather than forced physics.

5. Render measured vs completed geometry visibly.
   - Measured mesh regions and completed mesh regions should use distinct but non-distracting material styles.
   - Captions must state whether a clip is a dense run, a mesh-backed evidence window, or a completed sequence.
   - Final visual review must inspect overlays, standalone 3D, and side-by-side outputs.

## First V4 Experiment

Use wild-rice frames 2520 to 2550 because V3 already localized the missing mechanism there:

- accepted measured frames have strong dense-sheet QC;
- rejected frames expose both over-pruning and the real temporal-completion gap;
- hand detections are lower confidence, so contact weighting must be honest.

Measured-recovery experiment:

1. Use the existing dense measured masks and meshes as fixed evidence.
2. Re-admit VLM-selected masks that rule pruning rejected, then test them with the V3 residual checks.
3. Run V3 z-buffer/contact/SDF QC on measured frames and identify frames that still need temporal completion.
4. Render the measured sequence with status visible in the caption.

Temporal map fitting and completion follow this measured-recovery branch. They should consume measured/rejected/ambiguous status as data, not recreate the pruning rules.

## First V4 Experiment Result

The first V4 experiment used the VLM-selected SAM2 track for wild-rice frames 2520 to 2550 before the rule-pruning layer. The VLM selector accepted all 31 frames. The relaxed pruning report rejected ten of those frames, mostly through `too_many_components` or `large_secondary_component`; frame 2550 was the semantic rejection, where the VLM identified the mask as detached peel rather than the active stem.

Residual-gated geometry recovered a continuous 30-frame measured run, frames 2520 to 2549. The all-face z-buffer report over frames 2520 to 2550 has median silhouette IoU 0.964, median visible-silhouette-inside-mask fraction 0.989, median z-buffer absolute median 0.42 mm, and median z-buffer absolute p95 3.91 mm. Formerly pruned frames 2520, 2521, 2524, 2528, 2530, 2538, 2539, and 2540 mostly pass the geometry residuals; frame 2539 remains weak with IoU 0.510 and should be treated as low-confidence measured evidence.

Contact and SDF checks on the 30-frame measured run match the dense V3 branch: 15 reliable temporal contact rows, median contact patch p95 0.70 mm, median signed-gap p95 absolute 0.58 mm, selected-contact SDF penetration 0 percent, full-hand SDF penetration 0 percent, and selected-contact SDF p95 3.45 mm at 1 mm pitch.

Final V4 measured-run deliverables:

- Overlay video: `/data2/ego_annotation_outputs/representative_wild_rice/v4_mesh_surface_contact_review_unpruned_residual_gated_2520_2549/mesh_surface_contact_review.mp4`
- Side-by-side video: `/data2/ego_annotation_outputs/representative_wild_rice/v4_world_reconstruction_unpruned_residual_gated_finalvis_2520_2549/world_reconstruction_side_by_side.mp4`
- Standalone 3D video: `/data2/ego_annotation_outputs/representative_wild_rice/v4_world_reconstruction_unpruned_residual_gated_finalvis_2520_2549/world_reconstruction_3d.mp4`
- Evidence manifest: `/data2/ego_annotation_outputs/v4_unpruned_wild_rice_evidence_manifest_20260606.json`

This closes the residual-gated measured-recovery branch of V4 for the wild-rice clip. Temporal map fitting and completion remain open for frames where the model evidence is absent, semantically rejected, or too ambiguous after residual checks. The next V4 branch is a tracker/completion module that carries measured/completed/rejected status explicitly instead of pruning by visual-rule thresholds.
