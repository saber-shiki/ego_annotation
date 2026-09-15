# HOT3D evaluation

A CPU-only evaluator for exported annotation JSONs. It reads recorded input
provenance, loads HOT3D ground truth, validates the GT projection chain, and writes
hand/camera metrics with per-hand coverage and reproducibility hashes. It does
not import the annotation pipeline, run inference, download assets, or modify
input runs.

## Run

Use a Python environment with NumPy, SciPy, CPU PyTorch, and `smplx`. Licensed
MANO assets and the HOT3D dataset must already be available.

```bash
python hot3d_eval/run_eval.py \
  --annotations /path/to/run/final/annotation.json /path/to/another/annotation.json \
  --model-config /path/to/runtime/build/config/model_envs.json \
  --output report.md

python -m hot3d_eval.run_eval \
  --manifest hot3d_eval/annotations_manifest.example.txt \
  --mano-left /path/to/MANO_LEFT.pkl --mano-right /path/to/MANO_RIGHT.pkl \
  --dataset-root /path/to/hot3d/dataset --output report.md

python -m unittest hot3d_eval.test_hot3d_eval -v
```

The dataset root defaults to `/mnt/nas-106/hot3d/dataset`. The manifest is UTF-8,
one JSON path per line; blank lines and lines starting with `#` are ignored.
Relative manifest paths resolve against the manifest directory. Repeated input
paths are rejected. Source overrides `--source-video` and `--sensor-contract`
are available for **one input** at a time.

MANO lookup, independently for each hand:
1. Explicit `--mano-left` / `--mano-right`.
2. `--model-config`: `mano_models/MANO_{LEFT,RIGHT}.pkl`; if unreadable,
   `hawor_root/_DATA/data_left/mano_left/MANO_LEFT.pkl` and
   `hawor_root/_DATA/data/mano/MANO_RIGHT.pkl`.
3. When no config is supplied, environment variable `EGO_MANO_MODELS` names the
   directory containing `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`.

Missing or ambiguous source records are errors, not invitations to infer a
sequence from directory names. Asset absence is an explicit error explaining
these three configuration options. The report records the actual MANO file
hashes and whether the left shapedirs fix was needed.

## Source records and timestamp domain

For each annotation:

1. Read `input_identity.source_video_path` and
   `input_identity.source_sensor_calibration_path`. A single-input CLI override
   takes precedence. For legacy exports without these fields, read
   `../../input_source/DERIVATION.json` relative to the annotation's directory;
   use its `source_video` and adjacent `sensor_calibration_derived.json`.
2. Read `INPUT_PROVENANCE.json` beside the original source video. Use its
   `source_sequence_id`, `source_frame_idx_start/end`, and explicit upstream
   video path embedded in `test_provenance`. An existing upstream
   `input/raw_frame_manifest/manifest.json` is authoritative for per-frame
   `timestamp_ns` and `source_frame_idx`. The derived contract's recorded
   `source_frame_start` supplies the offset into the parent clip.
3. Require manifest `source_time_domain=TIME_CODE`. Never substitute
   `device_timestamp_ns`. When there is no manifest, index the sorted unique
   `214-1` timestamps in `box2d_hands.csv` by the recorded source frame index.
   The report labels this fallback.
4. Read exact-time `mano_hand_pose_trajectory.jsonl` hand records and combine
   their presence with `masks/mask_hand_pose_available.csv` for stream `214-1`.
   Missing/masked hand records are unavailable; GT hand poses are not
   interpolated. Headset poses use exact rows, otherwise nearest ≤1 ms,
   otherwise bracketing linear-translation/SO(3)-SLERP interpolation. The report
   includes each timestamp lookup method, offset, or interpolation bracket.

Frame IDs must be strictly increasing. Consecutiveness for temporal metrics is
based on **source frame indices**, so gaps are never joined into derivative or
relative-pose windows. Acceleration deliberately uses the fixed requested
`dt=1/30`, not empirical timestamp differences.

## Coordinate chain and GT validation

Transforms act on column vectors; arrays of points are stored as row vectors.
All geometry is in metres internally. Quaternion storage is `w,x,y,z`.

```
T_gt_world_native_camera(ts) = T_world_device(ts) @ T_Device_Camera(214-1)
T_gt_world_output_camera(ts) = T_gt_world_native_camera(ts) @ inv(Q)
Q[:3,:3] = R_output_camera_from_native_camera
```

The orientation contract must explicitly declare
`T_world_camera_native @ inv(R_output_camera_from_native_camera) in homogeneous form`.
Other semantics are rejected rather than guessed. The native FISHEYE624
parameters from the sensor contract are cross-checked against `camera_models.json`.

GT FK follows the official hand-tracking-toolkit convention:
`smplx.create(..., 'mano', use_pca=True, num_pca_comps=15, is_rhand=...)`, default
non-flat hand mean, record-specific β. The left x shapedirs are flipped when
left/right shapedirs exhibit the known MANO bug, as in the official toolkit.
The wrist quaternion is converted to a rotation vector and passed as
`global_orient`; wrist translation is passed as `transl`. This is MANO's
root-pivot transform, **not** an additional post-hoc rotation of the output
vertices about the world origin. The 16 joints plus tip vertices
`[744,320,443,554,671]` supply the 21 landmarks.

Before scoring, select three temporally spread **GT-visible** samples per hand
(independent of prediction validity). Project their 21 GT joints into the
**native FISHEYE624 image** and compare with the corresponding native RGB box,
expanded by 1.2 about its center. Every sample must have ≥80% of its joints
inside, with positive native camera depth. Fewer than three visible samples
for either hand, or any failing sample, aborts evaluation with exit code 2.
The command prints full per-frame projection points, boxes and offset vectors,
and saves `report.md.gate-failure.json`; it does not write a new score report.
A previously existing report is not overwritten on failure and must not be
mistaken for the failed invocation's output. Passing reports embed all gate
samples and camera timestamp diagnostics. Native projection validates the
native FK/extrinsic/time chain; the output-axis conversion is additionally
checked by the explicit contract and synthetic rotation test.

## Prediction authority and world registration

New exports use only `frames[i].hands.left/right.joints_world`. Null means
unavailable; finite non-null data must be 21×3. No FK reconstruction of predicted
hands is performed. Missing hands count against coverage.

For an annotation **entirely predating the joints_world field** (neither hand
has the key in any frame), an explicitly documented legacy recovery reads
`run/measurements/scene_fit/<object_id>/demo_state.npz` from the same run.
It loads only the `left/right_joints_world` state, matches archive `frame_idx`,
and requires archive camera poses to agree with the annotation to 1e-6.
The archive is opened read-only, with pickle disabled; its SHA256 appears in
report provenance. Reports and metric-table footnotes say:
**“joints source: state archive (annotation predates joints_world field)”**.
The presence of even one JSON `joints_world` key, including null, disables this
recovery for the entire annotation. If a legacy archive is absent, missing
predictions are explicitly reported as `预测缺 joints_world` and scores are N/A.

```
A = T_gt_world_output_camera(frames[0]) @ inv(T_pred_world_camera(frames[0]))
```

Apply this single SE(3) to all predicted cameras, joints and available
`object.visual_fitted_pose` transforms. There is **no scale fit and no per-frame
camera registration** at this stage. The first annotation frame anchors the
world even when no hand is valid there. Object poses are registered but no
object metric is specified by this protocol.

## Landmark mapping (zero-based)

The official canonical20 order is
`[tip thumb,index,middle,ring,pinky,wrist,thumb MCP,thumb IP, ...]`.
Removing canonical index 5 (wrist) yields official19. Ours21 additionally has
thumb CMC, which is excluded together with our wrist:

| Ours21 index | Landmark | Official19 index |
| --- | --- | --- |
| 0 | wrist | excluded |
| 1 | thumb CMC | excluded |
| 2 | thumb MCP | 5 |
| 3 | thumb IP | 6 |
| 4 | thumb tip | 0 |
| 5 | index MCP | 7 |
| 6 | index PIP | 8 |
| 7 | index DIP | 9 |
| 8 | index tip | 1 |
| 9 | middle MCP | 10 |
| 10 | middle PIP | 11 |
| 11 | middle DIP | 12 |
| 12 | middle tip | 2 |
| 13 | ring MCP | 13 |
| 14 | ring PIP | 14 |
| 15 | ring DIP | 15 |
| 16 | ring tip | 3 |
| 17 | pinky MCP | 16 |
| 18 | pinky PIP | 17 |
| 19 | pinky DIP | 18 |
| 20 | pinky tip | 4 |

`OURS21_TO_OFFICIAL19 = [4,8,12,16,20,2,3,5,6,7,9,10,11,13,14,15,17,18,19]`.

## Metric protocol

Protocol version: `hot3d-eval-v1-first-camera-se3`.
“Valid” means available prediction and GT for that hand at the matched time.
Acceptance/visibility labels from the prediction do not select scoring frames.

| Metric | Definition |
| --- | --- |
| MPJPE | Mean Euclidean error of official19, after A only, mm. |
| PCK_AUC | PCK with error ≤τ, τ=0:0.5:50 mm (101 thresholds), trapezoid integral /50×100. |
| FINGERTIP_PCK_AUC | Same, only first five official19 points. |
| MPVPE (official) | 778 rest-mesh vertices: `v_template + shapedirs @ β`, zero actual articulation/wrist. Prediction β is the first valid frame's `source_betas`, required constant over the whole annotation. GT β comes from each valid matched record. Millimetres; sequence-equal summary. |
| PA-MPJPE | Per-frame, per-hand Sim3 Umeyama fit on all 21 points, then mean error in mm. |
| W-MPJPE | WHAM/HaWoR-style consecutive valid runs split into ≤100-frame chunks; fit Sim3 using first two frames (42 paired points), score all 21 points over the chunk. |
| WA-MPJPE | Same chunks; fit Sim3 on every point in the chunk (2100 pairs for a full chunk). |
| RTE | Entire valid wrist trajectory SE(3) fit without scale; mean Euclidean wrist residual / GT wrist path length ×100. Path length only sums consecutive valid edges. Zero path length is N/A. |
| Accel | Mean norm of prediction−GT second differences divided by `(1/30)²`, m/s². Every three-frame window must be valid and consecutive. |
| Jerk (3rd-diff, mm) | Mean norm of predicted third position differences, mm; four consecutive valid frames. **Not divided by dt³.** Not directly comparable with Dyn-HaMR's paper scaling. Multiply by `30³/1000` for m/s³. |
| ATE first / SE3 / Sim3 | Camera-center error RMSE in mm after A only / a full-trajectory rigid fit / a full-trajectory similarity fit. |
| RPE-T / RPE-R | Adjacent-frame `E=inv(inv(G_i)@G_{i+1}) @ (inv(P_i)@P_{i+1})`; translation norm (mm), rotation angle (degrees), then per-sequence RMSE. |

**W/WA tails of 2–99 frames are retained.** Singletons are discarded. The report
lists chunk counts and lengths and all excluded slots. Fit-degenerate samples
are N/A with a reason; no arbitrary scale is inserted.

A violated shape-constancy contract emits `ERROR MPVPE(official)` on stderr and
makes **only** official MPVPE N/A, including the first drifting frame and maximum
β difference from the first valid β. Other metrics still run. An additional
**MPVPE(mean-β, diagnostic)** uses the per-coordinate mean β over valid frames,
labels this non-protocol choice, and stays out of the official summary. This
reveals temporal shape inconsistency without silently pretending a constant β
was exported. The prediction pipeline must export canonical β for official
shape comparability; this evaluator never modifies that pipeline.

Report coverage is shown per hand and metric as scored sample slots / all
annotation slots: 19N/21N points, 778N vertices for shape, 21(N−2) acceleration
slots, 21(N−3) jerk slots, N camera poses, and N−1 RPE pairs. RTE uses N wrist
slots. Summaries first average available hand/clip results within each source
sequence, then weight source sequences equally. Camera rows additionally report
cross-sequence median. N/A entries are never treated as zero.

## Verification boundary

The unittest suite includes closed-form zero/translation/Sim3/RPE/AUC cases,
landmark invariants, derivative units, missing-window/tail accounting, shape
expansion, legacy source and archive authority, camera timestamp lookup,
synthetic end-to-end reporting, and a deliberately failing projection gate.
It needs no network, GPU, HOT3D files, MANO assets, or model inference.
