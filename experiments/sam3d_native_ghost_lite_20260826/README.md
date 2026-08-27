# SAM3D Native × GHOST-lite Alignment — milk (carton) instance

Research worktree: `ego_annotation_worktrees/sam3d_native_ghost_lite`
Branch: `research/sam3d-native-ghost-lite-alignment-20260826`
Instance: `P0014_84ea2dcc_carton_milk_f2370_2519` (150 frames, anchor 92)
Run: `20260819T122452Z_hot3d_milk_local_authority_local29_v1`

This study verifies the SAM3D object mesh coordinate contract against the
sensor metric frames and evaluates how well the generated complete mesh
adheres to the observed surface and to the HaWoR hands — before any hand
refinement is attempted. It is a research audit only: **generated SAM3D
faces are never promoted to collision/sign/contact/nonpenetration authority.**

## Coordinate contract (verified)

SAM3D native output (`output["mesh"][0]`) is a **local mesh**; the true
metric camera mesh is obtained by

    mesh_camera = compose_transform(scale).rotate(quat_l2c).translate(trans_l2c)

with quaternion in PyTorch3D real-first (wxyz) order and OpenCV camera axes
(+x right, +y down, +z forward). The local mesh alone (without the native
pose) does **not** lie on the observed surface.

A/B audit (`scripts/audit_sam3d_native_pose_contract_v3.py`) compares five
hypotheses by convex-projection IoU against the object-owned mask:

| hypothesis | projection IoU |
|---|---|
| raw local mesh | 0.041 |
| pytorch3d, no flip | 0.006 |
| pytorch3d + declared flip | (wrong axis) |
| native OpenCV camera | **0.702** |
| native + metric scale | 0.702 (projection invariant) |

`native_pose_contract_supported` verifies the axis/quaternion/projection
contract only. The first implementation then made an invalid metric-scale
assumption: it matched observed depth to the median depth of **all complete
mesh vertices** projecting into the mask. That mixes front, back, and hidden
faces and placed the measured visible surface near the object's middle.

The corrected audit rasterizes the true first hit and estimates the
camera-origin scale from `observed_z / native_first_hit_z` at trusted surfel
pixels. It exports:

`.../audit_native_pose_first_hit_v2/sam3d_native_opencv_camera_origin_first_hit_metric_scaled.ply`

At frame 92 this changed the front-hit residual from −25.7 mm (mesh front in
front of observation) to approximately 0 mm. The corrected scale is 0.41112;
the rejected depth-centroid scale was approximately 0.3826.

## Rejected global-static Sim(3) experiment

`scripts/optimize_sam3d_multi_frame_sim3_ghost_lite.py` fits a **single
global Sim(3)** (uniform scale + rotation + translation in the anchor camera
frame) to 31 sampled frames using: one-way observed-surfel → mesh nearest
distance, first-hit depth (behind-observed + front-poke with tolerance),
silhouette outside-distance, and scale/rotation/translation priors.

Result: scale 1.232, rotation R, translation t. At the anchor and mid frames
the mesh adheres to the observed surface (median 5–14 mm), but **late frames
drift off** — frame 146 convex silhouette IoU = 0.00, signed depth ≈ −1.39 m.
Cause: the carton is *carried* by the hands, so its world position moves
~13 cm over the clip; a single static Sim(3) cannot cover that motion.

Conclusion: **a single static world/anchor Sim(3) is not the right contract.**
Per-frame object SE(3) is required, and the P15 observed pose graph provides
that motion body. This old output is retained only as a negative control.

## Corrected P15-aware true-first-hit alignment

`scripts/optimize_sam3d_p15_first_hit_sim3_v3.py` fixes the failed mechanism:

1. one small canonical Sim(3) correction is shared across frames;
2. each frame applies its own P15 object SE(3), then `T_world_camera_metric`;
3. depth uses the rasterized nearest z-buffer hit at trusted surfel pixels;
4. silhouette is bidirectional and hand-projected regions are explicit unknown;
5. scale/rotation/translation priors are zero-centred on the incremental
   correction, not the absolute camera translation;
6. nearest-neighbour/Chamfer is diagnostic only and does not drive alignment.

The first corrected run (`p15_first_hit_alignment_v2`) reduced objective RMS
from 0.0581 to 0.0332 and improved sampled-frame median silhouette IoU from
0.672 to 0.825. Full-topology 1408×1408 audit gave first-hit median residuals
between −3.7 and +7.3 mm on frames 30/50/70/92/103/110/121/130/146.
Most importantly, at frame 92 the measured surface moved from **48.5% through
the old mesh thickness** (depth-centred failure) to about **−6.6% from the
front boundary**, i.e. slightly outside/at the first hit rather than inside
the object's middle. The sign convention is `mesh_z - observed_z`: negative
means the generated front is closer to the camera, positive means it is
behind the measurement.

The recommended reproducible depth-authority configuration is
`p15_first_hit_depth_authority_bounded_v4`. It bounds the incremental Sim(3),
keeps shared scale essentially unchanged (`1.00074`), and uses only small
canonical rotation/translation updates. At the 704-pixel optimization plane,
sampled-frame median silhouette IoU improves from 0.672 to 0.759 while median
absolute per-frame first-hit error is 1.76 mm. Independent full-topology
1408×1408 audit on frames 30/50/70/92/103/110/121/130/146 reports first-hit
median residuals:

`−1.1, −2.2, −2.5, +3.1, −0.7, −4.9, +2.5, −4.1, +4.7 mm`.

At frame 92 the measured surface is at approximately −5.7% of the captured
front-to-last thickness (slightly outside/in front of the generated first
hit), compared with +48.5% for the rejected depth-centred mesh. The optimizer
hit its finite evaluation budget rather than a smooth-gradient convergence
condition, so the QC remains `optimizer_incomplete`; the mesh is recommended
because the independent full-resolution geometry audit passes the intended
mechanism, not because SciPy reported success.

## Why not a hand-refinement optimizer yet

1. Axis/quaternion/projection and true-first-hit metric scale are now verified.
2. The generated mesh is complete, but its hidden/back surfaces still have no
   signed collision authority. Hands must be driven by **observed-surface**
   constraints plus conservative contact proxies, never by raw generated faces.
3. Old unsigned nearest-neighbour hand/mesh gaps are not evidence of valid
   contact because they can hide a visible surface embedded inside the mesh.
   Any later hand refinement must use the corrected first-hit mesh plus an
   independent signed/conservative collision proxy.

## Artifacts

- Corrected native/metric audit: `.../audit_native_pose_first_hit_v2/sam3d_native_pose_contract_audit.json`
- Corrected first-hit metric mesh: `.../audit_native_pose_first_hit_v2/sam3d_native_opencv_camera_origin_first_hit_metric_scaled.ply`
- Recommended alignment QC: `.../p15_first_hit_depth_authority_bounded_v4/qc_sam3d_p15_first_hit_alignment.json`
- Recommended binary mesh: `.../p15_first_hit_depth_authority_bounded_v4/SAM3D_FIRST_HIT_ALIGNED_RECOMMENDED.binary.ply`
- Recommended ASCII mesh: `.../p15_first_hit_depth_authority_bounded_v4/SAM3D_FIRST_HIT_ALIGNED_RECOMMENDED.ascii.ply`
- Recommended OBJ: `.../p15_first_hit_depth_authority_bounded_v4/SAM3D_FIRST_HIT_ALIGNED_RECOMMENDED.obj`
- Full-resolution first-hit audit: `.../p15_first_hit_depth_authority_bounded_v4/fullres_audit/sam3d_first_hit_surface_position_audit.json`
- Old-vs-fixed depth slice: `.../p15_first_hit_depth_authority_bounded_v4/frame92_REJECTED_vs_RECOMMENDED_depth_slice.png`
- Corrected overlays: `.../p15_first_hit_depth_authority_bounded_v4/renders_overlay/overlay_*.png`
- Visible-surface overlays: `.../p15_first_hit_depth_authority_bounded_v4/renders_visible_surface_fit/fit_*.png`
- Combined frame-92 GLB scene: `.../p15_first_hit_depth_authority_bounded_v4/combined_scene_frame92/frame_000092_object_hands_visible_surface.glb`
- Combined frame-92 colored PLY: `.../combined_scene_frame92/frame_000092_object_hands_visible_surface.binary.ply` (ASCII sibling also provided)
- Combined-scene previews: `.../combined_scene_frame92/preview_front.png`, `preview_side.png`, `preview_angle.png`
- Rejected negative controls: `.../audit_native_pose/` and `.../sim3_alignment/`

## Environment

- Scripts run with the `sam3d-objects` conda env (pytorch3d + trimesh + open3d).
- Render uses `ego_annotation/.venv/bin/python` (open3d 0.19, cv2 4.11).

## Status

Local research only. Not pushed.
