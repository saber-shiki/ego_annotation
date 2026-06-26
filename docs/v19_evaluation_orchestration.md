# V19 Post-Run Quantitative Comparison Orchestration

This document is for the evaluator phase after prediction outputs are frozen. It is not part of the runtime prediction prompt or runtime runbook.

## 14. Bounded quantitative comparison

Current command truth: `scripts/build_v19_hot3d_clip_adapter.py` adapts public HOT3D-Clips WebDataset tars into a V19 input video/frame manifest and an evaluation-only HOT3D GT sidecar. H2O and DexYCB adapters are not implemented. Workbench item 6 must run real adapters before any external quantitative claim.

Minimal HOT3D-Clips adaptation command:

```bash
python "$REPO_ROOT/scripts/build_v19_hot3d_clip_adapter.py" \
  --tar "$BENCH_ROOT/hot3d_clips/raw/train_aria/clip-001849.tar" \
  --output-root "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849" \
  --clip-id clip-001849 \
  --split train_aria \
  --image-field image_214-1.jpg
```

The adapter's HOT3D hand/object/MANO/object-pose annotations under `evaluation/hot3d_gt/` are scoring-only and must not feed object prompts, hand state, contact, occlusion, or physical-state selection. HOT3D `cameras.json` calibration is sensor metadata, not a perception label; it may feed the camera adapter below.

Current fixed HOT3D slice v1 for the initial gate is the first three `train_aria` clip tars in HuggingFace repository path order, selected before scoring clips beyond `clip-001849`: `clip-001849`, `clip-001850`, and `clip-001851`. The run artifact records this at `$BENCH_ROOT/hot3d_clips/evaluation/v19_hot3d_fixed_slice_v1.json`.

Minimal fisheye-to-pinhole camera adaptation command:

```bash
python "$REPO_ROOT/scripts/build_v19_hot3d_pinhole_adapter.py" \
  --input-root "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849" \
  --output-root "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849" \
  --stream-id 214-1
```

This camera adapter writes a new V19 input video/manifest plus `state/calibration/v19_hot3d_pinhole_camera_calibration_contract.json`. It is not a prediction and does not score 3D state; it only makes later V19/HaWoR runs consume a pinhole camera instead of raw fisheye frames.

Initial HOT3D hand-box comparison command, valid only for 2D localization claims:

```bash
python "$REPO_ROOT/scripts/evaluate_v19_hot3d_hawor_boxes.py" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --hawor-npz "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/measurements/hawor_world_f609/hawor_world_hands.npz" \
  --output-report "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_hawor_box_eval.json" \
  --stream-id 214-1

python "$REPO_ROOT/scripts/render_v19_hot3d_hawor_box_review.py" \
  --manifest "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/input/raw_frame_manifest/manifest.json" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --hawor-npz "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/measurements/hawor_world_f609/hawor_world_hands.npz" \
  --output "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_hawor_box_review.jpg"

python "$REPO_ROOT/scripts/aggregate_v19_hot3d_box_evals.py" \
  --reports "$BENCH_ROOT"/hot3d_clips/v19_inputs/clip-*/evaluation/hot3d_hawor_box_eval.json \
  --output-report "$BENCH_ROOT/hot3d_clips/evaluation/hot3d_hawor_box_eval_aggregate.json"
```

After HaWoR has been run on the pinhole input with the pinhole focal from the calibration contract, score 3D MANO hand state in camera coordinates:

```bash
python "$REPO_ROOT/scripts/evaluate_v19_hot3d_hawor_mano3d.py" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --hawor-npz "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849/measurements/hawor_world_pinhole_f610/hawor_world_hands.npz" \
  --image-manifest "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849/input/raw_frame_manifest/manifest.json" \
  --output-report "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849/evaluation/hot3d_hawor_mano3d_eval.json" \
  --review-output "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849/evaluation/hot3d_hawor_mano3d_review.jpg"

python "$REPO_ROOT/scripts/aggregate_v19_hot3d_mano3d_evals.py" \
  --reports "$BENCH_ROOT"/hot3d_clips/v19_inputs_pinhole/clip-*/evaluation/hot3d_hawor_mano3d_eval.json \
  --output-report "$BENCH_ROOT/hot3d_clips/evaluation/hot3d_hawor_mano3d_eval_aggregate.json"
```

The 3D evaluator must run in an environment with `smplx` and side-specific MANO assets; the current fixed-slice run used the remote HaWoR environment and explicit `--mano-left/--mano-right` paths. It replays HOT3D GT MANO with HaWoR's 21-joint ordering, then compares HOT3D GT and HaWoR predictions in camera coordinates. It reports absolute wrist/joint errors separately from wrist-subtracted translation-aligned errors; no rotation or scale Procrustes alignment is applied, so this is not a pure articulation metric. The aggregate also splits same-frame detector-supported rows from infilled rows. It still does not score contact, occlusion, nonpenetration, or object pose.

After a V19 runtime prediction boundary is frozen, the same evaluator may score a P18 interval-state JSON directly:

```bash
python "$REPO_ROOT/scripts/evaluate_v19_hot3d_hawor_mano3d.py" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001850/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --interval-state "$RUN_ROOT/measurements/mano_interval_correction/keyboard_0_149/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --output-report "$RUN_ROOT/evaluation/hot3d_mano3d_interval/hot3d_v19_interval_mano3d_eval.json" \
  --image-manifest "$RUN_ROOT/evaluation/hot3d_mano3d_interval/review_frames_1408/manifest.json" \
  --review-output "$RUN_ROOT/evaluation/hot3d_mano3d_interval/hot3d_v19_interval_mano3d_review.jpg" \
  --mano-left "$BUNDLE_ROOT/third_party/WiLoR/mano_data/MANO_LEFT.pkl" \
  --mano-right "$BUNDLE_ROOT/third_party/WiLoR/mano_data/MANO_RIGHT.pkl"
```

Interval-state mode evaluates `optimized_joints_world_m` only. It obtains camera trajectory from the row `source_hawor_npz`, and it must not report full-vertex MANO metrics unless a future interval state stores full predicted vertices. The comparison to the HaWoR NPZ baseline must be interpreted mechanistically: if MPJPE worsens while wrist-subtracted MPJPE is unchanged, the correction changed global wrist/root placement more than articulation, so the next intervention should gate translation/contact constraints rather than treat the interval correction as a hand-accuracy improvement.

Workbench item 7 autoresearch compares MANO correction mechanisms against that baseline, not camera-adapter variants. The first supported correction target is low-support/occluded intervals where HaWoR keeps plausible boxes but hallucinates MANO articulation. Build a prediction-side repaired NPZ or interval state, then score it with the same evaluator:

```bash
python "$REPO_ROOT/scripts/repair_v19_mano_support_temporal.py" \
  --input-npz "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/measurements/hawor_world_pinhole_f610/hawor_world_hands.npz" \
  --output-npz "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/measurements/hawor_world_pinhole_f610_left_support_temporal_repair_v5_wristrel/hawor_world_hands.npz" \
  --report "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/measurements/hawor_world_pinhole_f610_left_support_temporal_repair_v5_wristrel/v19_mano_support_temporal_repair_report.json" \
  --sides left --repair-mode wrist_relative \
  --min-score 0.45 --min-area-ratio 0.35 --pose-norm-z 3.0 --pose-norm-mad-floor 0.25 \
  --fill-gap-frames 3 --pre-dilate-frames 4 --post-dilate-frames 4 \
  --long-run-min-frames 5 --long-run-post-dilate-frames 22 \
  --min-interval-frames 3 --max-anchor-gap-frames 55 --min-raw-bad-frames-per-interval 12

python "$REPO_ROOT/scripts/evaluate_v19_hot3d_hawor_mano3d.py" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001850/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --hawor-npz "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/measurements/hawor_world_pinhole_f610_left_support_temporal_repair_v5_wristrel/hawor_world_hands.npz" \
  --output-report "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/evaluation/hot3d_hawor_mano3d_left_support_temporal_repair_v5_eval.json"
```

This repair consumes no HOT3D hand labels. It uses prediction-side support signals only: same-frame support, detector score/area, hand-pose magnitude, and temporal continuity. Its necessary comparison is HaWoR pinhole baseline versus repaired MANO on the fixed slice, with target-interval metrics and review sheets preserved. If it improves only wrist-subtracted error but not MPJPE, the mechanism supports an articulation-prior component but does not close MANO state; the next intervention must use stronger hand-owned visible-surface/mask-depth evidence rather than more temporal smoothing.

The runbook still fixes the evaluation discipline now:

- Primary benchmark: 3-5 HOT3D clips.
- Optional secondary: 2-3 H2O clips, or DexYCB only if H2O is blocked. Do not run both H2O and DexYCB in the initial V19 gate.
- Baselines: V18 v5 on project representatives; HaWoR on selected HOT3D clips; official/reference metrics only where the dataset annotates the claim.
- Required ablations: MANO candidate source/refit, depth/camera source, contact/occlusion/nonpenetration factors, rigid branch enabled versus visible-surface-only after a rigid decision.

If benchmark evaluation is requested before adapters exist, stop with `missing_benchmark_adapter` and name the blocked physical metric family. Do not fabricate `metrics.json`.
