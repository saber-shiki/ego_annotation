# Pipeline V7: Replayable Mesh Priors and Visibility-Aware Surface State

## Starting Point

V6 has a repaired 31-frame wild-rice active-stem archive:

- object mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/solidified_sheet_object_meshes_world.npz`
- overlay video: `/data2/ego_annotation_outputs/representative_wild_rice/v6_mesh_surface_contact_review_repaired2539_2520_2550/mesh_surface_contact_review.mp4`
- side-by-side video: `/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/world_reconstruction_side_by_side.mp4`
- 3D video: `/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/world_reconstruction_3d.mp4`

The V6 object geometry is still a per-frame measured or repaired mesh stream. V6 also supplies sparse CoTracker factors with a continuous ready chain from frame 2534 to 2550, but transporting full meshes through those factors fails image replay. V7 therefore treats sparse factors as temporal evidence and tests explicit object priors through the same replay checks used for delivered meshes.

## Research Grounding

The relevant open-source frontier is split across four mechanism classes:

- SAM 3 provides open-vocabulary concept and exemplar segmentation in images and video, including text prompts, points, boxes, masks, and a detector-tracker design. Source: https://github.com/facebookresearch/sam3
- SAM 3D Objects reconstructs full 3D shape, texture, pose, and layout from masked objects in natural images. Source: https://github.com/facebookresearch/sam-3d-objects
- CoTracker3 tracks arbitrary or quasi-dense points through video. V6 already validated that its mask- and mesh-attached tracks can become sparse temporal factors. Source: https://github.com/facebookresearch/co-tracker
- 4DTAM and Shape of Motion support the representation choice: dynamic surfaces should be driven by depth, long-range tracks, low-dimensional motion, and visibility. Sources: https://github.com/muskie82/4dtam and https://shape-of-motion.github.io/

The research implication is precise: generated 3D models are candidate priors, and dynamic reconstruction papers inform temporal state variables. A generated or dynamic prior becomes delivered object annotation only after it replays against the measured video evidence.

## Representation

V7 keeps the V6 per-frame object meshes as measured observations. It adds two candidate state layers.

### Mesh-Prior Candidate

For each generated prior mesh `P`, V7 estimates a per-frame metric similarity transform to the measured object surface:

```text
X_t = s_t R_t P + p_t
```

where `s_t` is scale, `R_t` is a proper rotation, and `p_t` is translation in world coordinates. The first harness uses robust PCA and nearest-neighbor distances to align the prior to each measured frame. It reports both visible-surface coverage and hidden-surface conflict. Visible coverage asks whether the measured camera-visible sheet lies on the prior. Hidden conflict asks how much generated prior surface is unsupported by the measured sheet. Hidden conflict is diagnostic for completion quality, while delivery readiness comes from visible coverage plus image-depth, contact, SDF, and visual replay.

### Visibility-Aware Surface State

The temporal state is a set of object surfels or vertices with frame-local visibility and uncertainty:

```text
S_k = canonical surfel position and normal
M_tk = low-dimensional motion for surfel k at frame t
V_tk = model-produced visibility/support confidence
C_th = contact likelihood for hand region h at frame t
```

The surfel state is category-agnostic. SAM3/SAM2 masks, VLM verifier points, CoTracker tracks, UniDepth depth, VGGT cameras, MANO vertices, and contact rows all enter as data. Model evidence, geometry, and physics decide object behavior.

## Factor Graph

The graph has these node classes:

- `T_wc_t`: head camera pose at frame `t`, initialized from the accepted VGGT world trajectory and held fixed unless a later version tests camera refinement.
- `H_t`: MANO hand pose and vertices, initialized from the accepted hand stream and depth/contact refit.
- `O_t`: delivered per-frame observed object mesh, fixed as measurement.
- `P`: optional generated prior mesh, such as SAM 3D Objects or Hunyuan3D output.
- `A_t`: similarity or low-dimensional deformation variables mapping `P` or canonical surfels into frame `t`.
- `S_k`: canonical surfel variables for temporally supported surface patches.
- `z_tk`: continuous contact likelihood variables between hand regions and object surface patches.

The graph has these edge classes:

- image-depth observation edge: projected object surface must match the model-produced mask silhouette and metric depth under `T_wc_t`;
- prior alignment edge: `A_t(P)` or `A_t(S_k)` should stay near the measured object surface where that surface is visible;
- CoTracker correspondence edge: a tracked 2D point lifted by depth and attached to the mesh should remain near the same moving surfel across neighboring frames;
- local surface regularity edge: neighboring surfels should preserve short-range shape where both are visible, using robust penalties so occlusion and changing support can fail locally;
- temporal motion edge: `A_t` should change smoothly under a constant-velocity or low-dimensional SE(3) motion model;
- nonpenetration edge: hand vertices outside active contact patches should stay outside the object signed-distance field;
- contact equality edge: only hand patches with visual, geometric, and temporal support receive near-zero signed-distance residuals;
- contact dynamics edge: when object acceleration, contact patch, and hand motion are all observable, the contact impulse direction must be consistent with the object's measured acceleration up to unknown mass/friction bounds.

The contact variable `z_tk` controls attraction. A frame lacking contact evidence contributes nonpenetration and observation edges, while a contact-supported patch contributes a near-surface equality edge.

## Objective

V7 minimizes a robust weighted least-squares objective:

```text
min_X
  sum_t rho_mask_depth(E_replay(O_t or A_t(P), I_t, D_t, K_t, T_wc_t))
+ sum_t rho_align(d(A_t(P), O_t_visible))
+ sum_(t,k) rho_track(||A_t(S_k) - lift(track_tk)||)
+ sum_(t,k,j) rho_arap(||(S_k - S_j)_t - (S_k - S_j)_(t-1)||)
+ sum_t rho_motion(||A_t - predict(A_t | A_(t-1), A_(t-2))||)
+ sum_(t,h) rho_nonpen(max(0, -sdf_object(H_t,h)))
+ sum_(t,h) z_th rho_contact(|sdf_object(H_t,h)|)
```

`rho` is robust, because tracks, masks, monocular depth, and hand fits all fail locally. Replay terms are delivery checks as well as optimization residuals: a solution that improves the objective but fails silhouette/depth/contact/SDF replay is rejected.

## Solver

V7 uses two solver tiers:

1. Mesh-prior replay harness: align each candidate mesh to measured surfaces, archive it, then run existing z-buffer/contact/SDF QC. Complete priors are judged by visible-surface coverage and image-depth replay; hidden-surface conflict remains in the report because a full mesh can contain unseen geometry that the video cannot refute in one view.
2. Dynamic surfel graph: optimize low-dimensional per-frame motion and surfel offsets with SciPy or PyTorch least squares on bounded frame windows. The first target is the V6 wild-rice 2534 to 2550 interval, because it has continuous sparse track factors and verified measured geometry.

The solver keeps measurements fixed while testing prior completion only where measured evidence and visibility allow it.

## Acceptance Checks

For any V7 candidate archive:

- z-buffer replay must preserve silhouette and metric depth against the object mask and UniDepth depth;
- mesh-surface contact must be recomputed on the candidate archive;
- selected-contact SDF must show near-surface contact and object exterior consistency;
- full-hand SDF must show hand/object exterior consistency;
- sparse track residuals must remain within the V6 factor tolerance on ready pairs;
- stakeholder render must show head trajectory, MANO hands, object mesh, contact markers, and captions clearly.

## First Implemented Artifact

### Generated-Prior Replay Harness

`scripts/archive_aligned_mesh_prior_v7.py` creates a replayable mesh-prior archive:

```bash
.venv/bin/python scripts/archive_aligned_mesh_prior_v7.py \
  --mesh-prior <candidate_mesh.obj-or-ply> \
  --observed-mesh-archive /data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/solidified_sheet_object_meshes_world.npz \
  --frame-start 2534 \
  --frame-end 2550 \
  --output-mesh-archive <out>/aligned_prior_meshes_world.npz \
  --output-json <out>/qc_aligned_mesh_prior_v7.json
```

The output mesh archive can be passed directly to `render_mesh_zbuffer_qc_v3.py`, `diagnose_mesh_surface_contact_v3.py`, `diagnose_volume_sdf_contact_v3.py`, and `diagnose_hand_object_sdf_penetration_v3.py`.

This first V7 artifact is a falsification harness. It decides whether a real generated mesh prior deserves to enter the dynamic graph.

Negative-control run:

```bash
.venv/bin/python scripts/archive_aligned_mesh_prior_v7.py \
  --mesh-prior /data2/ego_annotation_outputs/representative_wild_rice/v3_trellis_frame2548_raw/trellis_mesh.ply \
  --observed-mesh-archive /data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/solidified_sheet_object_meshes_world.npz \
  --frame-start 2538 \
  --frame-end 2540 \
  --output-mesh-archive /data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/aligned_prior_meshes_world.npz \
  --output-json /data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/qc_aligned_mesh_prior_v7.json
```

The TRELLIS prior is rejected before contact checks:

- alignment bidirectional p95 median: 0.842 m;
- visible-surface coverage p95 median: 0.022 m;
- alignment threshold: 0.010 m;
- z-buffer replay median silhouette IoU: 0.152;
- z-buffer replay median depth absolute error: 0.083 m;
- z-buffer replay p95 depth absolute error median: 0.124 m.

Artifacts:

- alignment report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/qc_aligned_mesh_prior_v7.json`
- z-buffer report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/zbuffer_qc/qc_mesh_zbuffer_projection_v3.json`
- z-buffer video: `/data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/zbuffer_qc/mesh_zbuffer_projection_qc.mp4`

The rejected result proves the harness is live: a visually plausible generated mesh becomes object pose only after visible-surface coverage and image-depth replay agree with the observed video.

`scripts/run_v7_generated_prior_replay_qc.py` wraps the same acceptance logic for future SAM 3D Objects, Hunyuan3D, TRELLIS, or Mesh4D outputs. It first replays the observed target archive against the supplied mask, metric depth, camera pose, and intrinsics contract. If that measured target fails visible-inside or depth thresholds, the wrapper returns `invalid_observed_target` before prior alignment, because prior acceptance would otherwise mix target-contract failure with generated-mesh failure. After the observed target passes, the wrapper runs prior alignment, z-buffer replay, and a single accept/reject report. The TRELLIS negative-control wrapper report rejects the prior because visible-surface coverage and every replay check fail:

- alignment bidirectional p95: 0.906 m, threshold 0.010 m;
- visible-surface coverage p95: 0.023 m, threshold 0.010 m;
- median silhouette IoU: 0.152, threshold 0.900;
- median visible-inside fraction: 0.152, threshold 0.900;
- median z-buffer p95 depth error: 0.124 m, threshold 0.010 m.

Wrapper artifact:

- report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_generated_prior_replay_trellis_negative_control_2538_2540/qc_v7_generated_prior_replay.json`

Corrected visible-surface replay artifact:

- report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_generated_prior_replay_trellis_negative_control_visible_semantics_2538_2540/qc_v7_generated_prior_replay.json`
- z-buffer video: `/data2/ego_annotation_outputs/representative_wild_rice/v7_generated_prior_replay_trellis_negative_control_visible_semantics_2538_2540/zbuffer_qc/mesh_zbuffer_projection_qc.mp4`

The corrected report keeps strict full-surface alignment as a diagnostic and uses these delivery checks: visible-surface coverage p95, silhouette IoU, visible-inside fraction, and z-buffer p95 depth. The TRELLIS prior remains rejected:

- visible-surface coverage p95: 0.025 m, threshold 0.010 m;
- median silhouette IoU: 0.158, threshold 0.900;
- median visible-inside fraction: 0.159, threshold 0.900;
- median z-buffer p95 depth error: 0.123 m, threshold 0.010 m.

### Representative Trash Prior Replay

V7 also tests the generated-prior replay contract on the non-kitchen trash-lid representative. The measured input is the existing SAMWISE/UniDepth/VGGT-K solidified sheet archive for frames 865 to 870:

- measured archive: `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_perframe_thick001_865_870/solidified_sheet_object_meshes_world.npz`
- manifest: `/data2/ego_annotation_outputs/representative_trash/v3_samwise_pink_lid_bundlesdf_dataset_858_880/manifest.json`
- hand/camera annotations: `/data2/ego_annotation_outputs/representative_trash/v3_mano_side_metric_refit_858_880/annotations_side_metric_refit.json`
- metric depth: `/data2/ego_annotation_outputs/representative_trash/v3_unidepth_metric_source_858_880/unidepth_metric_depth_v3.npz`

This measured sheet archive is visible-surface evidence only. It is not treated as closed object pose.

TRELLIS frame-880 prior replay:

- report: `/data2/ego_annotation_outputs/representative_trash/v7_generated_prior_replay_trellis_frame880_865_870/qc_v7_generated_prior_replay.json`
- visual check: `/data2/ego_annotation_outputs/representative_trash/v7_generated_prior_replay_trellis_frame880_865_870/visual_check/contact_sheet.png`
- visible-surface coverage p95: 0.0248 m;
- median silhouette IoU: 0.404;
- median visible-inside fraction: 0.918;
- median z-buffer p95 depth error: 0.0531 m.

Hunyuan3D-mv prior replay:

- report: `/data2/ego_annotation_outputs/representative_trash/v7_generated_prior_replay_hunyuan_mv_depthgrown_865_870/qc_v7_generated_prior_replay.json`
- visual check: `/data2/ego_annotation_outputs/representative_trash/v7_generated_prior_replay_hunyuan_mv_depthgrown_865_870/visual_check/contact_sheet.png`
- visible-surface coverage p95: 0.0549 m;
- median silhouette IoU: 0.315;
- median visible-inside fraction: 0.885;
- median z-buffer p95 depth error: 0.0940 m.

Visual inspection shows both generated priors become smooth cap or bowl-like surfaces over the lid and miss the rim/interior depth structure. V7 therefore rejects both as object-pose annotations.

### Representative Mop Prior Replay

V7 tests a third representative class: a long-handled mop. This stresses long, thin tool geometry and large perspective changes.

Measured evidence:

- measured archive: `/data2/ego_annotation_outputs/representative_mop/v7_mop_observed_surface_contract_unidepth_vggt_759_765/observed_mask_depth_meshes_world.npz`
- manifest: `/data2/ego_annotation_outputs/representative_mop/v3_mop_depth_manifold_dataset_735_765/manifest.json`
- hand/camera annotations: `/data2/ego_annotation_outputs/representative_mop/v3_vggt_object_skeleton_735_765/annotations_v3_vggt_object_skeleton.json`
- metric depth: `/data2/ego_annotation_outputs/representative_mop/v3_unidepth_dense_735_765/unidepth_full_frame_depth_v3.npz`
- measured replay baseline: `/data2/ego_annotation_outputs/representative_mop/v7_mop_observed_surface_contract_unidepth_vggt_zbuffer_759_765/qc_mesh_zbuffer_projection_v3.json`

The earlier mop baseline mixed a per-row depth PNG and fixed intrinsics at export time with full-frame UniDepth and annotation-VGGT intrinsics at replay time, producing a false failure. Re-exporting the same model-produced masks with the same UniDepth and VGGT contract used by replay gives live measured visible-surface evidence: median silhouette IoU is 0.808, median visible-inside fraction is 1.000, and median z-buffer p95 depth error is 0.0016 m. This remains visible-surface evidence, not closed object-pose delivery.

The guarded wrapper now reproduces this diagnosis: running the stale archive through the corrected replay contract returns `invalid_observed_target` with median visible-inside fraction 0.352 and median z-buffer p95 depth error 0.048 m. Running the corrected archive passes observed-target replay and continues to prior rejection.

TRELLIS frame-750 prior replay:

- report: `/data2/ego_annotation_outputs/representative_mop/v7_generated_prior_replay_trellis_frame750_contract_unidepth_vggt_759_765/qc_v7_generated_prior_replay.json`
- z-buffer video: `/data2/ego_annotation_outputs/representative_mop/v7_generated_prior_replay_trellis_frame750_contract_unidepth_vggt_759_765/zbuffer_qc/mesh_zbuffer_projection_qc.mp4`
- visible-surface coverage p95: 0.137 m;
- hidden-surface conflict p95: 0.744 m;
- median silhouette IoU: 0.126;
- median visible-inside fraction: 0.142;
- median z-buffer p95 depth error: 0.0730 m.

Visual inspection shows the generated prior becomes a long diagonal plank crossing the room instead of the mop head and handle. V7 rejects it as object-pose annotation.

### Visibility-Aware Surfel Graph

`scripts/fit_visibility_surfel_graph_v7.py` builds temporal surfel nodes from mesh-attached learned point tracks. It consumes the V6 merged pair-factor report and the accepted V6 mesh archive, rejects duplicate observation conflicts explicitly, and solves:

```text
min_X
  ||X - X_measured|| / sigma_obs
+ ||R_t X_i + p_t - X_j|| / sigma_track
+ ||X_{t+1} - 2 X_t + X_{t-1}|| / sigma_smooth
```

The solve changes only the surfel state package and leaves the delivered mesh archive fixed.

Command:

```bash
.venv/bin/python scripts/fit_visibility_surfel_graph_v7.py \
  --mesh-archive /data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/solidified_sheet_object_meshes_world.npz \
  --pair-factors-json /data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor5_merged_pair_factors_2532_2550/qc_merged_pair_factors_v6.json \
  --frame-start 2534 \
  --frame-end 2550 \
  --output-dir /data2/ego_annotation_outputs/representative_wild_rice/v7_visibility_surfel_graph_multianchor5_2534_2550
```

Result:

- surfel nodes: 695;
- accepted CoTracker edges: 509;
- smooth triples: 325;
- rejected pairs: 0;
- pair residual p95 before solve: 8.36 mm;
- pair residual p95 after solve: 6.93 mm;
- correction displacement p95: 0.91 mm;
- solver evaluations: 5;
- status: `annotation_ready: false`.

Artifacts:

- repaired-archive sparse edges for frame-2539 anchor: `/data2/ego_annotation_outputs/representative_wild_rice/v7_cotracker_anchor2539_sparse_edges_repaired_archive_2532_2550/cotracker_sparse_correspondence_edges_v7.json`
- repaired-archive pair factors: `/data2/ego_annotation_outputs/representative_wild_rice/v7_cotracker_anchor2539_pairwise_repaired_archive_2532_2550/qc_cotracker_pairwise_rigid_factors_v7.json`
- merged pair factors: `/data2/ego_annotation_outputs/representative_wild_rice/v7_cotracker_multianchor5_repaired_archive_merged_pair_factors_2532_2550/qc_merged_pair_factors_v7.json`
- report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_visibility_surfel_graph_repaired_archive_multianchor5_2534_2550/qc_visibility_surfel_graph_v7.json`
- solved surfel positions: `/data2/ego_annotation_outputs/representative_wild_rice/v7_visibility_surfel_graph_repaired_archive_multianchor5_2534_2550/visibility_surfel_positions_v7.npz`
- visual review video: `/data2/ego_annotation_outputs/representative_wild_rice/v7_visibility_surfel_review_repaired_archive_multianchor5_2534_2550/visibility_surfel_review_v7.mp4`

The first surfel run exposed a provenance bug: frame 2539 disappeared because gap sparse-edge vertex indices came from the pre-repair factor-graph mesh archive, while V7 consumed the repaired V6 mesh archive. V7 now fails fast when sparse-edge provenance points to a different mesh archive. Reattaching the frame-2539 CoTracker world points to the repaired archive restores continuous surfel support through 2539.

This closes the first V7 temporal-state evidence: the repaired-archive CoTracker factors support a visibility-aware surfel graph over observed patches with sub-millimeter correction magnitude. The graph represents observed temporal patches and serves as a state prior for later mesh completion.
