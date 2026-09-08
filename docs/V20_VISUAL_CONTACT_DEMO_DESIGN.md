# Visual-contact demo patch (owner-approved objective change)

Base c49e028; keep all prior artifacts and formal states immutable. New demo-only branch. The deliverable is the original 150-frame/5s video plus clear reconstructed SAM3D/MANO interaction and contact close-ups. No GT input, no contact/SDF/collision authority claim. P09 exact per-frame depth is NOT a hard acceptance gate. Full genuine object/hand meshes retained.

## Mechanism and rationale

The old mesh/pointcloud metric objective is no longer the demo target. Preserve the v16 visual object pose/shape as initial trajectory. Reconstruct both hands from the hash-bound HaWoR MANO axis-angle parameters; verify zero-state vs existing 778-vertex geometry. Correct left MANO shapedirs X sign and flat-hand-mean convention before any fit (right agrees to submicron precision; left agreement ~microns). Existing focused RRD uses `if side in hands` where hands is keyed by frame, causing missing world hands; new renderer must log per-frame meshes and use actual scene z-buffering, not skeleton overlays.

Optimize actual object SE(3), MANO root translation/orientation, and bounded hand articulation corrections on a keyframe timeline. Rendered state is reconstructed with MANO LBS at every frame, not interpolation of arbitrary deformed surface vertices. Keep beta fixed. A small per-hand camera-origin scale/depth adjustment may be used only as an explicit bounded demo similarity if needed, not hidden geometry edits.

Visual objective: projected object silhouette support from its real mesh vs object-owned image mask; MANO joint/vertex projection prior to existing prediction; selected visible thumb/finger pad-to-triangle contact hypotheses; soft normal-based intrusion discouragement; priors and time regularization. Contact correspondence comes from actual hand surface patches and nearest/ray-hit object triangles, not a centroid or 2D marker. Only hypotheses with image-near support and bounded changes activate; no assumption that every finger touches.

Fit clear raw-video thumb contact around stable grasp windows; use temporal interpolation of parameter corrections for all frames. Other fingers maintain source MANO pose unless image-supported proximity warrants a weak constraint. Start with 0/35/65/95/120/140/149 raw frame inspection. Any manual image observation is saved as transparent demo keyframe annotation, not machine GT.

## Render

Use full triangle surface z-buffer (CPU BVH or GPU rasterizer), combined hand/object occlusion. Clean colors, flat/smooth diffuse shading, no dense wireframe/P09 cloud in main demo. Side-by-side original vs overlay, plus a clearly labeled fixed +20-degree yawed 3D inspection close-up (use --detail-yaw-deg 0 for a camera-aligned comparison); no composited raw video pasted into object geometry. RRD records actual world object + both MANO meshes, camera, demo fitting transforms and uncertainty. Contact markers only if fitted surface gap supports them; video content must show geometry contact independently of a label.

## Controlled iteration

1. Verify MANO reconstruction and source/raw tip overlays; measure real pad-to-mesh gaps.
2. Fit with bounded changes and fixed silhouette/projection priors. Keyframe and full-timeline gap/projection checks plus visually inspect first renders. No pixel/depth arbitrary gate can replace visual judgment.
3. If contact pulls hand away from raw video, strengthen local image constraints or reject that contact hypothesis, rather than forcing every finger to touch.
4. If object contour deteriorates, preserve base pose and re-fit hands around it, retaining full MANO.
5. Export full-duration demo, keyframe views, NPZ state, exact source hashes and failure limitations. No push/merge without owner authorization.

This is an offline visual fitting demonstration, not the default annotation runtime performance benchmark or a claim of measured contact.

## Preflight results and fixed controls

Both hands replay through SMPL-X MANO with `flat_hand_mean=True`; left shapedirs[:,0,:] sign correction is required (ordinary left model was ~4mm wrong; corrected model agrees at micrometre scale). Use the saved 778-vertex identity to verify all-frame replay. The hand contact tips are vertices744/320/443/554/671 (thumb/index/middle/ring/little).

Raw frames show clear thumb grasp at35/65/95/120; save these as explicit demo visual contact hypotheses, not automatically measured contact. Primary targets are thumb pads from both hands where image support is near the object mask; other near fingers only weakly. Initial thumb surface-point distances range ~6–42mm. A frame-smoothed per-hand camera-origin similarity scale [0.85,1.18] is explicitly authorized by this demo design, with fixed beta and true MANO articulation. This largely preserves the hand image projection while permitting uncertain depth/size reconciliation. Bound root adjustment and joint articulation and protect original image projections. Model-contact distance uses actual triangles and cannot certify physical nonpenetration.
