# SAM3D Native × GHOST-lite Alignment — milk study and frozen cross-video benchmark

Research worktree: `ego_annotation_worktrees/sam3d_native_ghost_lite`
Branch: `research/sam3d-native-ghost-lite-alignment-20260826`
Initial instance: `P0014_84ea2dcc_carton_milk_f2370_2519` (150 frames, anchor 92)
Initial run: `20260819T122452Z_hot3d_milk_local_authority_local29_v1`

This study verifies the SAM3D object mesh coordinate contract against the
sensor metric frames and evaluates how well the generated complete mesh
adheres to the observed surface and to the HaWoR hands — before any hand
refinement is attempted. After the initial milk investigation, the same
first-hit/shared-Sim(3) configuration is evaluated without category-specific
parameters on four pre-existing non-milk clips (spatula, bottle, can, mug).
It is a research audit only: **generated SAM3D faces are never promoted to
collision/sign/contact/nonpenetration authority.**

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

## Frozen cross-video benchmark

The benchmark definition is frozen in `cross_video_benchmark_v1.json` before
reviewing optimization results. It includes four 150-frame clips selected to
span different geometry, not to maximize the score:

- thin elongated spatula, anchor 149;
- near-axisymmetric bottle, anchor 139;
- strongly axisymmetric can, anchor 90;
- nonconvex mug with handle, anchor 32.

All cases use their own same-run P12 native mesh, P15 pose graph, visible
surface, object-owned masks, HaWoR hands, and camera K. They share the milk
bounded-v4 objective, bounds, 704 raster plane, 20k-face QEM optimization
proxy, and priors. The only temporal generalization is replacing milk's
hard-coded frame window with 16 frames sampled uniformly from each clip's
eligible visible-surface observations while always including the anchor.
There are no object-label branches in the optimizer or renderer.

All native contracts pass independently. Native OpenCV convex-projection IoU
and first-hit camera-origin metric scale are:

| case | native IoU | first-hit metric scale |
|---|---:|---:|
| spatula | 0.638 | 0.561 |
| bottle | 0.719 | 0.415 |
| can | 0.780 | 0.381 |
| mug | 0.514 | 0.396 |

The frozen shared-Sim(3) sampled-frame results are not uniformly beneficial:

| case | silhouette IoU before→after | median absolute first-hit before→after |
|---|---:|---:|
| spatula | 0.197→0.202 | 4.62→5.64 mm |
| bottle | 0.364→0.458 | 1.54→1.00 mm |
| can | 0.682→0.702 | 8.74→2.30 mm |
| mug | 0.439→0.436 | 5.21→6.23 mm |

All four cases then completed full-timeline 150-frame before/after renders at
the 960 review plane, using the complete mesh topology and true MANO z-buffer
occlusion. The delivery-grade metric is `hand_occluded_render_vs_owned_mask`,
medians over all 150 frames:

| case | IoU before→after | observed coverage | centroid px | boundary px | frames improved/degraded |
|---|---:|---:|---:|---:|---:|
| spatula | 0.187→0.190 | 0.217→0.221 | 57.4→58.0 | 26.00→26.17 | 145 / 5 |
| bottle | 0.329→0.388 | 0.832→0.848 | 43.6→30.6 | 20.29→16.00 | 132 / 18 |
| can | 0.621→**0.592** | 0.934→**0.858** | 19.3→**32.5** | 14.08→**15.38** | 45 / **105** |
| mug | 0.406→0.422 | 0.701→0.694 | 63.3→59.1 | 16.58→15.39 | 134 / 16 |

This rejects any claim that one shared canonical Sim(3) is a generally
sufficient video alignment optimizer:

1. **It is net harmful on can.** The sampled proxy objective predicted an
   improvement (0.682→0.702), but on the full timeline with complete topology
   the median IoU *drops* and 105/150 frames degrade.
2. **The optimization proxy is not a faithful stand-in for the delivery
   audit.** Restricted to the same 16 used frames, can's proxy delta median is
   `+0.011` while its full-topology delta median is `-0.024`, with 5/16 frames
   flipping sign (correlation 0.66). Spatula and bottle agree well (corr 0.77 /
   0.97), so the mismatch is case-dependent and cannot be assumed away. The
   differing contracts are 20k QEM proxy vs full topology, 704 vs 960 px, and
   hand convex-hull unknown vs true MANO first-hit occlusion.
3. **It trades away good anchors.** bottle anchor 139 falls 0.848→0.678 and
   can anchor 90 falls 0.822→0.722 while mid-sequence frames improve. A single
   global 7-DoF correction cannot protect well-posed frames and repair
   badly-posed ones at once.
4. **It cannot repair temporal pose drift.** can frame 146 is grossly
   mislocated both before and after; a shared Sim(3) changes projected shape
   and depth but not per-frame P15 translation error.

Spatula's low IoU is not merely a small-object metric artifact. Normalized by
`sqrt(median owned-mask area)`, its boundary error is 0.218 vs can 0.097 and
bottle 0.115, i.e. genuinely about twice as bad in relative terms.

Full-timeline before/after true-raster videos, per-frame metrics, hstacked A/B
videos (1920×960, h264, 150 frames each) and contact sheets are under
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/sam3d_native_ghost_lite_cross_video_benchmark_20260828/`,
aggregated in `cross_video_benchmark_summary.json`.

### Root cause and what the next solver must change

Per `solve_v19_rigid_object_pose_graph.py` and
`fit_v18_compact_rigid_object_pose.py`: P14 trajectories come from accumulated
adjacent partial-surfel ICP plus an anchor unary fit, and the P15 graph's data
terms are only a zero-delta prior, temporal delta smoothing, and optional
nonpenetration. There is **no image-space first-hit or silhouette factor**, and
these runs' P15 optimizers essentially no-op (`nfev=1, cost=0`). Residual
per-frame error is therefore structural, not a missing smoother — P15 already
carries correction velocity/acceleration residuals.

The next round should extend that existing sparse pose graph with per-frame
SE(3) deltas (scale still shared), adding first-hit depth + bidirectional
silhouette + hand-unknown data terms to the residuals it already has, with
rotation priors weighted by geometric observability, the anchor as a
high-confidence prior rather than a hard lock, and bounds calibrated from these
four cases rather than milk's ±30 mm / ±5°. Because of finding (2) it must
either demonstrate per-frame proxy/full-topology equivalence or accept steps
via a full-topology trust region, and must keep the hand-unknown contract
identical between optimization and evaluation.

## Depth-order front-only mask correction (S0–S5)

The cross-video benchmark above concluded that a shared Sim(3) is the wrong
contract and that P15 needs per-frame SE(3) plus image factors. Following that,
this worktree traced the milk case's apparent mask/surface misalignment to its
actual root cause. Full write-up: `docs/research_pipeline_version_tree.md` §4.

**S0 — depth hypothesis rejected.** `diagnose_mask_depth_alignment_three_experiments.py`
shows same-frame `depth → 3D → pixel` closure is 0 px and
`camera→world→camera` is 9.34e-08 m. Millimetre axial depth residuals cannot
produce tens-of-pixels lateral offset.

**S1 — root cause.** `diagnose_hand_depth_order_ownership.py` reconstructs V19
ownership exactly (146/146 frames) and rasterizes per-hand first-hit depth.
Current ownership is `raw_mask − dilate(projected_full_MANO_silhouette, 4 px)`
with **no depth-order test**. Of validly-removed pixels, **78.29% have the hand
behind the measured object surface**; only 0.67% are genuinely in front.

**S2 — front-only counterfactual mask.** `build_depth_order_counterfactual_masks.py`
removes only `hand_z < object_z − 5 mm`, preserving behind-object, ambiguous and
padding-only pixels. With the **same old GHOST mesh and no re-optimization**,
swapping only the evaluation mask moves full-video median IoU `0.574 → 0.785`
and centroid `43.29 → 9.30 px`. Most apparent misalignment was a contaminated
evaluation target, not bad geometry.

**S3 — rebuilt P14/P15.** `build_depth_order_counterfactual_pose_reference.py`
rebinds the corrected visible geometry to the same observed-only P14/P15
contract. The new trajectory shifts translation by median 19.37 mm and rotation
by 2.75°, and repairs the tail: frames 132–149 gain ΔIoU +0.0736 and −10.75 px
centroid, dropping centroid max `38.6 → 27.0 px`, at the cost of slight
mid-segment degradation.

**S4 — first-hit/silhouette image factors in P15.**
`build_p15_first_hit_silhouette_factors.py` freezes local correspondences from a
single low-resolution raster of the observed-only pose-hypothesis mesh;
`scripts/solve_v19_rigid_object_pose_graph.py` (+294 lines) re-linearizes those
canonical points under each per-frame SE(3) via `--image-factor-npz`. Hand
regions stay explicit unknown support; the NPZ is contract-bound to
annotations/pose_report/mesh/object_id. P15 stops being degenerate
(`nfev=1, cost=0` → optimizer success `nfev=10`), and full-video improves to
IoU median `0.7912` with centroid max `19.31 px`. Frames 84–127 still degrade.

**S5 — temporal lag rejected.** `diagnose_temporal_lag_mesh_mask.py` finds best
integer lag = 0 in all three configurations; P14 translation regularization tau
is only 0.0091 s (0.27 frames). Residual error is per-frame pose/shape, not lag.

Not yet merged into the P09 mainline: ownership must become per-pixel
depth-order, and 4 px padding must be a review band rather than a deletion.
Only the milk case has been validated; can already degraded under shared Sim(3).

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
- Correct-intrinsics full 150-frame optimized-object overlay: `.../p15_first_hit_depth_authority_bounded_v4/full_video_correct_intrinsics_v2/optimized_object_camera_overlay.mp4`
- Correct-intrinsics full 150-frame raw-vs-overlay comparison: `.../p15_first_hit_depth_authority_bounded_v4/full_video_correct_intrinsics_v2/optimized_object_side_by_side.mp4`
- Correct-intrinsics full-video QC, including per-frame raster-vs-mask metrics: `.../p15_first_hit_depth_authority_bounded_v4/full_video_correct_intrinsics_v2/qc_optimized_object_full_video.json`
- Rejected wrong-intrinsics videos: `.../p15_first_hit_depth_authority_bounded_v4/full_video/`; the renderer used 1408×1408 source-plane K directly on 960×960 manifest RGB, giving zero median hull-mask IoU and about 349 px median centroid error.
- Rejected negative controls: `.../audit_native_pose/` and `.../sim3_alignment/`

## Environment

- Scripts run with the `sam3d-objects` conda env (pytorch3d + trimesh + open3d).
- Render uses `ego_annotation/.venv/bin/python` (open3d 0.19, cv2 4.11).

## Status

Committed on `research/sam3d-native-ghost-lite-alignment-20260826` and pushed to
the `personal` remote. Research audit only — no generated face has been promoted
to collision/sign/contact authority, and the P09 ownership mainline is unchanged.
