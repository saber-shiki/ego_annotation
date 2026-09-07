# V20 Late-Window Prediction-Only SE(3) Repair Design

## Objective

Repair the late-frame object trajectory for `P0014_84ea2dcc_carton_milk_f2370_2519 / carton_milk`, emphasizing frames `115–149`, where the generated visible mesh is observed in front of the P09 visible surface. Preserve prediction/ground-truth separation and never promote generated geometry to physical authority.

## Inputs

- V19 annotation contract and raw video.
- P09 visible surface observations from `visible_geometry_candidate.camera_vertices_sample_m` (camera frame) or contract-equivalent `world_vertices_sample_m` (world frame).
- Prediction-only base full-timeline pose report.
- Prediction-only LightGlue/SuperPoint plus source-depth PnP relative edges, weighted by inlier fraction, reprojection error, source-3D conditioning, and cycle consistency.
- Optional generated mesh only for post-solve render/first-hit validation and later shape adaptation.
- HOT3D GT is forbidden during candidate generation and may be used only by a separate post-freeze evaluator.

## Coordinate contract

For `T_world_camera=[R,t]`:

```text
p_world  = p_camera @ R.T + t
p_camera = (p_world - t) @ R
```

Per-frame correction uses additive translation and left rotation:

```text
R_i = Exp(delta_rotation_i) @ R_initial_i
t_i = t_initial_i + delta_translation_i
```

Frame 143 has no direct metric visible-surface observation. Initialize/output it by continuous SE(3) interpolation between neighboring direct frames, with explicit uncertain provenance.

## Pose objective

Optimize only prediction-side evidence:

1. robust SO(3) geodesic residuals for accepted RGB relative edges;
2. RGB/PnP relative translation residuals with conditioning-aware reduced confidence;
3. observed P09 point-to-plane and point-to-point residuals from correspondences rebuilt at every outer pass;
4. absolute correction priors;
5. absolute trajectory velocity/acceleration priors;
6. correction smoothness and immutable anchor gauge.

Generated-mesh first-hit/silhouette factors are excluded from the pose objective. They are rebuilt only to validate candidates against common-hit depth, coverage, and silhouette gates.

## Outer loop and gates

- Rebuild observed point correspondences at every outer iteration.
- Keep an immutable initial pose/metric baseline.
- Compare every candidate with both current and immutable initial states.
- Apply per-frame, segment, and global common-hit depth gates, plus coverage and silhouette gates when generated validation is enabled.
- Never leak rejected candidates.
- Enforce total rotation/translation trust bounds relative to the immutable initial state.
- Preserve rows outside the requested window exactly.

## Shape stage

After pose freeze, optimize only shared low-dimensional canonical shape (anisotropic scale and canonical translation, with optional bounded deformation). Rebuild true first-hit factors per outer pass; keep pose fixed and require step plus fixed-initial gates. Generated geometry remains render-only.

## Rendering stage

- Convert P09 camera points to world exactly once for Rerun world views.
- Project world points to camera exactly once for 2D overlays.
- Preserve all objective mesh faces when path/hash contract is active.
- Render the complete original frame count/duration.
- Mark metric-missing frames uncertain.
- Never use display-only pruning to hide mismatch.

## Required artifacts

- late-window pose report with hashes/provenance;
- optional shape-adapted mesh/report bound to exact pose hash;
- full 150-frame MP4 and RRD;
- representative frames `120,130,135,140,142,143,144,149`;
- signed/absolute first-hit metrics;
- edge/cycle/conditioning diagnostics;
- separate post-freeze GT evaluation.

## Non-goals and prohibitions

- No GT input to solver, initialization, edge selection, or shape fitting.
- No formal P14/P15/D19 state modification.
- No face pruning as pose repair.
- No blind global camera-z shift as substitute for SE(3).
- No collision/contact/SDF/sign/nonpenetration claim from generated mesh.
- No annotation-ready claim from JSON, validators, or aggregate medians alone.

## Rotation observability amendment

The late-window graph must not inherit a small cumulative-rotation correction
cap from the old local solver. Only the selected anchor frame is gauge-fixed;
all other late-window rotations remain free state variables. Per-update trust
regions may prevent numerical jumps, but the allowed total trajectory rotation
must be broad enough to contain the RGB-observed motion and must be reported
when clipping occurs. RGB adjacent/multihop geodesic edges are the primary
rotation evidence; P09 partial depth and generated mesh are not rotation
authority. If the graph is rotation-bound or the evidence is unobservable, the
run must report an uncertain/incomplete candidate rather than silently flatten
or replace the trajectory.

## Implementation invariants

- Frame 143 is excluded from the interpolation source when it lacks P09 metric geometry; it remains a variable node with temporal/RGB/prior evidence and uncertainty.
- RGB NPZ and JSON rows are merged by `(source_frame_idx,target_frame_idx)` exactly once. JSON diagnostics do not replace an NPZ numeric transform when JSON omits the transform; if both serialize a transform, their SE(3) identity is checked.
- `source_3d_conditioning` is normalized by `source_conditioning_full_weight_scale` before weighting; raw conditioning and effective weight are preserved in diagnostics.
- Generated validation is not passed to `pose_residual`. Its current-state and immutable-initial comparisons are separate from the observed/RGB pose objective.
- The renderer converts `camera_vertices_sample_m` to world exactly once before Rerun logging or world-to-camera overlay projection.
- Edges below `min_rgb_rotation_conditioning` retain their prediction-side translation evidence but set `rotation_weight_scale=0`; low-conditioning planar PnP is never allowed to silently act as rotation authority.
- Signed-front validation now records per-frame and contiguous negative segments in addition to aggregate statistics. Optional strict limits are explicit CLI gates and only affect visible-pose/render candidate status.
- Optional observed-only RGB source-canonical→target-UV factors may be serialized as flattened NPZ arrays with offsets. They are disabled by default, and low-conditioning factors freeze rotation while retaining weak translation evidence.
- Post-freeze shape may optionally include one bounded shared canonical registration rotation, explicitly reported as shape registration rather than per-frame pose; it must not be used to claim recovered object rotation.
