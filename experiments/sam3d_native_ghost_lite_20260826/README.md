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

`native_pose_contract_supported`. The metric camera mesh is exported as
`.../audit_native_pose/sam3d_native_opencv_camera_origin_metric_scaled.ply`
(anchor camera frame, metres).

## Multi-frame Sim(3) experiment (negative result, still useful)

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

Conclusion: **a single global Sim(3) is not the right contract.** Per-frame
SE(3) is required, and the P15 observed pose graph already provides exactly
that.

## Final alignment (P15 per-frame pose + SAM3D metric mesh)

`scripts/audit_sam3d_p15_aligned_hand_mesh_v3.py` consumes
`sam3d_native_opencv_camera_origin_metric_scaled.ply` (canonical =
anchor-world − anchor-centroid) + P15 per-frame
`rotation_world_from_completed_canonical` / `translation_world_m` + HaWoR
hands + P09 first-surface surfels:

| frame | observed→mesh median | P95 | hand→mesh min (L/R) | contact <5mm (R) |
|---|---|---|---|---|
| 92  | 10.5 mm | 25.3 mm | 0.17 / 16.1 mm | 0 % |
| 103 | 11.3 mm | 34.8 mm | 2.6 / 0.06 mm | 12.6 % |
| 121 | 11.7 mm | 34.0 mm | 4.1 / 0.3 mm | 6.3 % |
| 146 | 10.0 mm | 20.5 mm | 13.2 / 0.1 mm | 4.2 % |

Full 146-frame median observed→mesh: **11.9 mm** (P95 ≈ 20–43 mm).  Hand
min-gaps are mostly < 5 mm, i.e. consistent with physical contact (hand
thickness ≈ 15–20 mm) — no systematic floating or penetration at this
evidence level. Render overlays (`--mode p15`) confirm the cyan SAM3D hull
tracks the red owned mask across frames 30–146.

## Why not a hand-refinement optimizer yet

1. Coordinate contract is now verified; a hand optimizer can trust the mesh.
2. The generated mesh is complete, but its hidden/back surfaces still have no
   signed collision authority. Hands must be driven by **observed-surface**
   constraints plus *conservative* contact proxies, never by the raw generated
   faces.
3. P15 + SAM3D metric mesh already put hands within a few mm of the surface in
   most contact frames; refinement risk is low but requires signed geometry
   before it can claim nonpenetration.

## Artifacts

- Audit: `experiments/sam3d_native_ghost_lite_20260826/audit_native_pose/sam3d_native_pose_contract_audit.json`
- Metric mesh: `.../audit_native_pose/sam3d_native_opencv_camera_origin_metric_scaled.ply`
- Sim(3) QC: `.../sim3_alignment/qc_sam3d_ghost_lite_sim3_alignment.json`
- Sim(3) mesh: `.../sim3_alignment/sam3d_ghost_lite_sim3_aligned_anchor_camera.ply`
- Hand/mesh audit: `.../diagnostic/sam3d_p15_aligned_hand_mesh_audit.json`
- Renders: `.../renders_p15/overlay_*.png` (+ `mosaic_p15.png`)

## Environment

- Scripts run with the `sam3d-objects` conda env (pytorch3d + trimesh + open3d).
- Render uses `ego_annotation/.venv/bin/python` (open3d 0.19, cv2 4.11).

## Status

Local research only. Not pushed.
