# Visual-contact demo: replay, refit and supported scope

## What is verified

The published demo geometry was produced by commit
`5ef89331c8174764415c23381a0a0e773f320ee9`, using the previously frozen V16
pose/shape and HaWoR/MANO predictions. Later publication cleanup formats code,
replaces wildcard imports and adds a launcher/documentation without changing
the fitter or renderer's numerical statements. See
[V20_VISUAL_CONTACT_DEMO_RESULTS.md](V20_VISUAL_CONTACT_DEMO_RESULTS.md).

Three operations must not be confused:

1. **Render replay**: use the frozen `demo_state.npz` and original video. No
   upstream predictions or contact fitting are recomputed.
2. **Demo refit**: recompute MANO/object fitting from named prediction inputs,
   then render. This still reuses V19/V20 predictions.
3. **V19 fresh + demo**: regenerate the prediction inputs from raw video first,
   then apply a separately declared demo stage. This has **not yet been
   validated end-to-end for this demo branch**.

## Current case limits and transferability

The SE(3), MANO LBS, image projection, pad/triangle distance and temporal
correction mechanisms are reusable. Current scripts nevertheless implement a
validated **case profile**, not a general demo generator:

- 150 consecutive frame IDs `0..149` and complete prediction poses;
- a single rigid object, with a hash-bound canonical mesh in the pose report;
- square 960×960 object masks/render plane; source-camera K is explicitly
  resized using the half-pixel convention;
- both hands present through the clip; HaWoR world axis-angle/beta arrays and
  matching 778-vertex MANO bridge/topology;
- flat-hand-mean MANO and the verified left `shapedirs[:, 0, :]` convention for
  the supplied model assets;
- thumb-first grasp contact hypotheses, not automatically verified contact
  for every action or object;
- keyframe corrections every five frames, replayed through actual MANO at every
  output frame.

Changing input paths/object ID is not proof of transfer. Other frame counts,
single-hand/occluded intervals, different MANO conventions, articulated objects
and other actions require explicit adapters/hypotheses and rendered validation.
No physical/SDF/collision authority is inferred from generated mesh contact.

## Runtime environment

The environment observed when preparing this branch for publication:

| Dependency | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.12.0 (delivered run reported `2.12.0+cu130`) |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| smplx | 0.1.28 |
| chumpy | 0.70 |
| trimesh | 4.12.2 |
| open3d | 0.19.0 |
| opencv-python / opencv-python-headless | 4.11.0.86 |
| rerun-sdk | 0.37.0 |

The server interpreter is
`/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`; GPU fitting used
physical GPU5, logical `cuda:0`. This is a recorded environment, **not a tested
clean-install lockfile**. The broader `pyproject.toml` constraints differ from
this historical runtime (notably NumPy), so do not claim `pip install .` alone
recreates it. The runner records actual package versions each time.

External assets: compatible licensed `MANO_LEFT.pkl`/`MANO_RIGHT.pkl`, the source
prediction arrays and video, plus ffmpeg/ffprobe. Render replay needs no MANO
model files because full geometry is in `demo_state.npz`. No installer or
runtime dependency provisioning is performed by the launcher.

## Fastest visual reproduction: render replay

Run these commands in the preflighted compute environment. `--dry-run` only
validates paths and prints argv; it does not validate decoder/model compatibility
and creates no output directory. Remove it to execute.

```bash
REPO=/path/to/checkout
PY=/path/to/preflighted/python
DELIVERY=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/demos/milk_visual_contact_20260908
VIDEO=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_unidepth_sam3d_ghost_lite_full_20260903T035100Z/input/input.mp4

cd "$REPO"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" scripts/run_v20_visual_contact_demo.py \
  --mode replay \
  --state "$DELIVERY/state/demo_state.npz" \
  --source-video "$VIDEO" \
  --detail-yaw-deg 20 \
  --output-dir /path/to/new/demo_replay \
  --dry-run
```

The original source-video SHA256 is:

```text
43513ed7534f8d32ede3ec13ad8c5b92d41487705064081491ab7831ee0e1b59
```

Verify the intended video and delivered `SHA256SUMS` before replay. Matching
frame count alone does not establish video identity.

## Recompute demo fitting from the same cached predictions

This reproduces the postprocessing mechanism, **not a V19 fresh run**. Supply
paths explicitly, use the expected source/model assets, and run fitting on the
selected server/GPU in managed tmux.

```bash
REPO=/path/to/checkout
PY=/path/to/preflighted/python
CASE=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_depth_order_corrected_fullrun_20260903T101508Z/run/P0014_84ea2dcc_carton_milk_f2370_2519
SOURCE_CASE=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_unidepth_sam3d_ghost_lite_full_20260903T035100Z/run/P0014_84ea2dcc_carton_milk_f2370_2519
VIDEO=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_unidepth_sam3d_ghost_lite_full_20260903T035100Z/input/input.mp4
MANO_MODELS=/path/to/licensed/mano_data

cd "$REPO"
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$PY" scripts/run_v20_visual_contact_demo.py \
  --mode fit-render \
  --source-video "$VIDEO" \
  --annotations "$CASE/measurements/object_geometry/visible_geometry/carton_milk/annotations_v19_visible_geometry.json" \
  --pose-report "$CASE/experiments/pose_repair_v20_late_window_se3_20260907/final_candidate_late_window_v16/v20_late_window_pose_shape_candidate.json" \
  --mano-bridge "$CASE/state/base_annotations/v19_mano_bridge_from_hawor_world.npz" \
  --hawor-params "$SOURCE_CASE/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --mano-models "$MANO_MODELS" \
  --object-id carton_milk --device cuda:0 --iterations 240 \
  --detail-yaw-deg 20 \
  --output-dir /path/to/new/demo_refit \
  --dry-run
```

The pose report must name the correct mesh path/hash; do not rename or recenter
the mesh independently. The fitter verifies source/bridge vertices and MANO
zero-state vertices/joints. These inputs are deliberately outside Git; no model
weights or licensed assets are included in the branch.

Expected output layout:

```text
new_run/
  run_report.json             # actual argv, hashes, environment, step exits
  fit.log                    # fit-render only
  fit/                       # fit-render only
    demo_state.npz
    fit_report.json
    contact_hypotheses.json
  render.log
  renders/
    interaction_demo.mp4
    contact_detail.mp4
    interaction_demo.rrd
    render_report.json
    frame_120_detail.png      # plus other representative frames
```

The runner refuses an existing output directory. A subprocess failure stops the
chain and records the exit/log; it does not silently fall back or use an earlier
run. `completed_pending_visual_review` is an execution status, not contact or
annotation readiness.

## Validation

```bash
"$PY" -m unittest discover -s tests -p 'test_v20_visual_contact_demo.py' -v
"$PY" -m unittest discover -s tests -p 'test_v20_demo_runner.py' -v
"$PY" -m py_compile scripts/v20_demo_geometry.py scripts/fit_v20_visual_contact_demo.py \
  scripts/render_v20_visual_contact_demo.py scripts/run_v20_visual_contact_demo.py
ffprobe -v error -show_entries stream=width,height,nb_frames,r_frame_rate:format=duration \
  -of json /path/to/new/demo_refit/renders/interaction_demo.mp4
ffmpeg -v error -i /path/to/new/demo_refit/renders/interaction_demo.mp4 -f null -
```

Inspect actual full videos and at least frames35/65/95/120/140/149. Use the
camera-aligned overlay for image fidelity and the labeled inspection view for
surface gaps. Check both world MANO meshes in RRD; do not infer successful
contact from millimetre figures or row counts alone. Independent reruns need
not be byte-identical: GPU/library/encoding and RRD metadata can differ. Exact
hashes establish input identity; visible/geometric equivalence establishes
functional reproduction.
