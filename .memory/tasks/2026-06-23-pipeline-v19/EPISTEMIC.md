# Pipeline V19 current epistemic state

## Current supported claim

The HOT3D `clip-001850` pinhole V19 v5 run is the current frozen prediction boundary. It supersedes the older v3 frozen snapshot, which remains immutable but rejected for mechanism failure. V5 was executed by runtime agents on A800/truenas with parent code/spec/infrastructure repairs only; parent did not manually run prediction phases or copy canonical renders. Provenance: OPS 2026-06-26T18:21.

The two user-identified failures are repaired for the frozen v5 artifact at the level the evidence supports:

- Frame 0032 no longer exhibits the rejected source-coordinate SAM2 failure where the object annotation tracked only the right-edge sleeve/table region. The canonical overlay shows the green keyboard/object mesh crossing the keyboard body/key field, and P21 ties that render to a direct corrected P15 pose row. Provenance: OPS 2026-06-26T17:02, 2026-06-26T17:35, 2026-06-26T18:21.
- Frame 0074 no longer loses the keyboard in the canonical render. P15 emits a `completed_temporal_rigid_pose_uncertain` row interpolated between visible frames `[60,75]`; P19/P20/P21 consume it so the green keyboard/object mesh remains visible at frame 0074. This is an uncertain temporal completion, not a direct local mask/depth observation. Provenance: OPS 2026-06-26T17:35, 2026-06-26T18:21.

The frozen v5 run root is `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`. Its local mirror is `/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`. The freeze manifest is `state/v19_prediction_freeze_manifest.json`; HOT3D scoring is explicitly deferred and has not been run. Provenance: OPS 2026-06-26T18:21.

## Mechanisms now supported

1. **Canonical focal extraction is fixed for v5.** V5 P04 used `559.1640985505215` from canonical calibration fields, not the diagnostics/outlier focal `780.8843383789062` that invalidated v4. This supports the v5 hand/camera backbone under the current input contract. Provenance: OPS 2026-06-26T16:32, 2026-06-26T17:02.

2. **Source-coordinate object prompts are interpreted correctly.** P07 now records prompt/source/SAM2 coordinate sizes and scales source-frame prompt points into the SAM2 frame instead of treating them as 960-pixel prompt coordinates. The accepted repaired P07 masks preserve keyboard identity on reviewed usable frames; remaining local gaps are carried as uncertain/missing observations into P15, not treated as object absence. Provenance: OPS 2026-06-26T14:45, 2026-06-26T17:02, 2026-06-26T17:23.

3. **Rigid pose completion is full-timeline.** P15 v5 has 150 pose rows with full-timeline rigid completion enabled: 13 direct corrected rows and 137 uncertain completed rows. Frame 0032 is direct; frame 0074 is uncertain interpolation between frames 60 and 75. This repairs the state-level disappearance mechanism from rejected v3. Provenance: OPS 2026-06-26T17:35, 2026-06-26T18:21.

4. **Canonical render publication is now real-copy based.** The earlier symlink-preferred P20 path produced zero-byte canonical files on truenas despite valid published-runtime copies. `scripts/publish_v19_render_artifact.py` now publishes canonical outputs as verified non-empty copies. Runtime reran P20 and produced openable canonical MP4s: overlay 15,913,485 bytes, world 14,389,614 bytes, side-by-side 9,840,582 bytes. Provenance: OPS 2026-06-26T18:21.

5. **The final artifact has been consumed as an annotation.** Parent and runtime both consumed the canonical renders, not just schemas or logs. P21 reports 150-frame canonical videos and render/state linkage at frames 0032/0074; parent inspected the extracted contact sheet/stills and confirmed the two rejected visible failures are absent. Provenance: OPS 2026-06-26T18:21.

## Remaining uncertainty and scope control

- The rendered green keyboard geometry is visibly broad/noisy relative to the physical keyboard. The freeze supports identity preservation, timeline continuity, and render consumption; it does not support a claim of precise keyboard geometry.
- Frame 0074 remains an uncertain temporal rigid completion because local SAM2/depth evidence is missing there. The artifact correctly carries that uncertainty instead of omitting the object.
- Contact is not closed with certainty. P17/P18 provide posterior/gap state and ownership quarantine; the evidence supports uncertain hand/object interaction state, not certain contact ownership.
- The v5 freeze is a prediction boundary for later evaluation. It is not a HOT3D score and should not be described as benchmark success until evaluator/ablation runs consume the frozen prediction without modifying it.

## Next action

Commit the parent-owned publisher repair and task-memory updates with narrow staging. Do not run HOT3D scoring in the prediction/runtime run. If evaluation is requested next, start a separate evaluator phase that consumes the frozen v5 run root and writes evaluation outputs without modifying prediction state.
