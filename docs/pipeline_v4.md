# Pipeline V4: Temporal Object Completion with Residual-Gated Geometry

## V3 Closure State

V3 now has two mesh-backed representative results:

- Trash/lid, frames 858 to 880: 23-frame measured object mesh, MANO hands, contact rows, full-hand nonpenetration, overlay video, standalone 3D world animation, and side-by-side presentation.
- Wild-rice stem, frames 2531 to 2537: seven-frame continuous mesh-backed evidence window with the same deliverable types and the same geometry/contact/SDF checks.

The remaining limitation is temporal completeness. Wild-rice does not have an object mesh for every frame from 2520 to 2550 because the VLM/SAM/depth evidence rejects several frames. V3 correctly refuses to fabricate object geometry for those frames. V4 must solve that missing-frame problem with model-backed temporal completion and then preserve the V3 checks as the acceptance contract.

## Research-Backed Design Decision

The strongest direct research target is ForeHOI: it is designed for daily hand-object interaction videos and combines 2D mask inpainting with 3D shape completion. Its public repository is reachable, but the README currently marks inference and training code as unreleased. ForeHOI should shape the V4 target contract, but V4 cannot depend on unreleased inference code.

Released components support a practical V4:

- SAM2, Cutie, and DEVA-style video object segmentation and memory propagation can propose temporally complete mask tracks.
- Existing VLM point-prompt and SAM2 candidate selection code in this repo can provide per-frame mask hypotheses and rejection evidence.
- V3 measured-sheet reconstruction can convert accepted masks and metric depth into watertight object meshes.
- V3 z-buffer, mesh-surface contact, selected-contact SDF, full-hand SDF, and visual review already falsify wrong geometry.

V4 should therefore make temporal object completion a residual-gated map problem, not a category-specific rule system and not a single generative-prior replacement.

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
- rejected frames expose the temporal-completion gap;
- hand detections are lower confidence, so contact weighting must be honest.

Experiment:

1. Use the existing dense measured masks and meshes as fixed evidence.
2. Generate temporal mask proposals for the missing frames with the strongest available released tracker.
3. Fit a temporally smooth object map/pose sequence over all frames, with measured frames as hard evidence and propagated frames as weaker evidence.
4. Run V3 z-buffer/contact/SDF QC on measured frames and completion-specific silhouette/depth/motion QC on completed frames.
5. Render two videos: measured-only evidence window and completed-sequence view, with completion status visible in the caption.

V4 can close only when the completed frames are model-backed, residual-checked, and visually inspected. A copied mesh, primitive, or unverified generative fill remains rejected.
