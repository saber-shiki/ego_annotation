# V19 Runtime Spec

This is the only runtime instruction document. It defines state ontology, execution policy, exact phase order, scripts, command templates, required outputs, and stop conditions. Do not inspect or mention any file that is not named by this spec.

## Runtime inputs

The launch provides:

- `{INPUT_VIDEO}`: egocentric input video;
- `{RUN_ROOT}`: fresh output run root;
- `{CASE_ID}`: case id;
- this runtime workspace;
- prediction-side sensor metadata, if present next to the input.

## Runtime outputs

The runtime output is a prediction run root containing `input/`, `measurements/`, `state/`, `renders/`, and `logs/`. The renderer consumes `state/`. Logs and measurements are provenance, not final annotations.

## State ontology

- `camera`: intrinsics, camera/head pose, depth/scale provenance, frame/time semantics, uncertainty.
- `hands`: metric 3D MANO state over time, side, camera/world transforms, visibility, provenance, uncertainty.
- `objects`: object instances, masks/tracks, physical branch, reconstructed or adapted geometry, pose/posterior, provenance, uncertainty.
- `visibility_occlusion`: visible, partially visible, occluded, out-of-frame, or unresolved state for hands and objects, with occluder ownership when inferable.
- `contact`: contact, near-contact, non-contact, or unresolved state with patch/distance evidence and uncertainty.
- `nonpenetration`: hand/object geometry residuals and uncertainty; absence of a valid signed volume is unresolved, not success.
- `renders`: visible overlay/world/side-by-side annotations caused by state variables.

## Evidence rules

- A detector box, keypoint track, mask, depth map, point cloud, centroid, label, or JSON row is a measurement, not physical state by itself.
- Object pose requires object geometry adapted or fitted to observed instance evidence and a pose trajectory/posterior.
- Hand state requires metric MANO surface or reproducible MANO parameters with camera/world semantics.
- Contact and occlusion require geometric, depth-order, temporal, or explicitly uncertain evidence. Do not make them certain from a semantic label alone.
- Weak measurements continue downstream with uncertainty. Broken contracts, wrong frame alignment, wrong coordinate frame, wrong object mask, side swap, missing geometry, or invalid units must be fixed or represented as unresolved.

## Execution policy

1. Execute phases in order.
2. Do not discover or substitute scripts. Each script phase names the script to run.
3. For an agent-write phase, write only the specified JSON/Markdown artifact and preserve uncertainty.
4. Bind placeholders from launch arguments, phase outputs, or this spec. If a placeholder cannot be bound without searching outside the bundle, record the unresolved placeholder as a blocker.
5. Heavy model phases run on the declared server target after probe and bundle sync. Light metadata/state phases may run locally.
6. Infrastructure is out of scope for runtime. Parent preflight is complete before launch. Execute prediction phases only; if a named phase command fails, record that phase blocker and stop.
7. Do not run scoring or comparisons inside this runtime run.
8. Do not use sleep, polling loops, or idle waits. Long-running jobs need durable command logs/status files and inspectable job handles.

## Declared compute and asset targets

- `{REMOTE}`: `yiwen@192.168.11.220`
- `{REMOTE_BUNDLE}`: `/mnt/user-home/yiwen/ego_annotation_runtime/v19_bundle`
- `{REMOTE_OUTPUT}`: `/mnt/truenas-user-home/yiwen/ego_annotation_outputs`
- HaWoR work root: `/mnt/user-home/yiwen/ego_annotation_remote/hawor_work`
- HaWoR Python: `/mnt/user-home/yiwen/ego_annotation_remote/hawor_work/.venv_hawor/bin/python`
- SAM2 checkpoint: `/mnt/user-home/yiwen/ego_annotation_remote/data/sam2.1_hiera_small.pt`
- UniDepth checkout: `/mnt/truenas-user-home/yiwen/a800_migrated_home/ego_annotation_remote/unidepth_work/UniDepth`
- Remote model Python for UniDepth/SAM2: `/mnt/user-home/yiwen/ego_annotation_remote/model_envs/unidepth_sam2/bin/python`; this is a parent-preflight launch contract.

## Stop condition

If a phase cannot run because an input, script, model asset, or environment is missing, write:

`{RUN_ROOT}/state/runtime_blockers/<PHASE_ID>.json`

with phase id, missing component, blocked state variable, evidence, and next required repair. Stop that branch rather than inventing substitute outputs.

## Placeholders

- `{FRAME_END}`: last frame index from P01 manifest.
- `{SOURCE_WIDTH}`, `{SOURCE_HEIGHT}`: source video resolution from P01 manifest.
- `{GPU_ID}`: selected server GPU from P02.
- `{REMOTE_MODEL_PYTHON}`: `/mnt/user-home/yiwen/ego_annotation_remote/model_envs/unidepth_sam2/bin/python`, a parent-preflighted remote model interpreter used for remote UniDepth/SAM2 Python phases.
- `{OBJECT_ID}`: object id chosen in P05.
- `{TRACK_ID}`: SAM2 track id for `{OBJECT_ID}`.
- `{ANCHOR_FRAME}`: selected clean object evidence frame.
- `{INTERVAL_START}`, `{INTERVAL_END}`: selected physical interval for MANO/object correction.
- `<calibration_contract>`: chosen calibration contract JSON filename under `{RUN_ROOT}/state/calibration/`.
- `<completed_mesh_ply>`: completed mesh path from P13.
- `<visible_contact_ownership_factor_report>`: factor report from P17.
- `<render_branch_overlay_mp4>`, `<render_branch_world_mp4>`, `<render_branch_side_by_side_mp4>`: P19 render outputs.

# Phase graph

## P00 startup records

Type: agent writes JSON/Markdown.

Outputs:

- `{RUN_ROOT}/input/runtime_input_contract.json`
- `{RUN_ROOT}/logs/harness_events.jsonl`
- `{RUN_ROOT}/state/v19_physical_state.json`
- `{RUN_ROOT}/state/v19_uncertainty_state.json`
- `{RUN_ROOT}/state/v19_agent_evidence.md`

State after phase: unresolved camera, hands, objects, contact, occlusion, and nonpenetration.

## P01 raw frame manifest

Script: `scripts/build_v19_raw_frame_manifest.py`

```bash
python scripts/build_v19_raw_frame_manifest.py \
  --video "{INPUT_VIDEO}" \
  --output-dir "{RUN_ROOT}/input/raw_frame_manifest" \
  --render-width 960
```

Required output: `{RUN_ROOT}/input/raw_frame_manifest/manifest.json`.

## P02 server probe, bundle sync, and remote input staging

Type: bash command.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 "{REMOTE}" \
  "set -euo pipefail; mkdir -p '{REMOTE_BUNDLE}' '{REMOTE_OUTPUT}/runtime_inputs/{CASE_ID}/raw_frame_manifest' '{REMOTE_OUTPUT}/v19_runs/{CASE_ID}'; hostname; df -h '{REMOTE_OUTPUT}'; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
rsync -a --delete ./ "{REMOTE}:{REMOTE_BUNDLE}/"
rsync -a --delete "{RUN_ROOT}/input/raw_frame_manifest/" "{REMOTE}:{REMOTE_OUTPUT}/runtime_inputs/{CASE_ID}/raw_frame_manifest/"
```

Required output: log event in `{RUN_ROOT}/logs/harness_events.jsonl` with selected `{GPU_ID}`, successful bundle sync, and successful raw-frame-manifest staging.

## P03 depth and intrinsics measurement

Script: `scripts/run_unidepth_full_frame_v3.py`

```bash
ssh "{REMOTE}" "set -euo pipefail; cd '{REMOTE_BUNDLE}'; CUDA_VISIBLE_DEVICES='{GPU_ID}' '{REMOTE_MODEL_PYTHON}' scripts/run_unidepth_full_frame_v3.py \
  --manifest '{REMOTE_OUTPUT}/runtime_inputs/{CASE_ID}/raw_frame_manifest/manifest.json' \
  --output-dir '{REMOTE_OUTPUT}/v19_runs/{CASE_ID}/measurements/depth_slam/unidepth_full_frame' \
  --frame-start 0 \
  --frame-end {FRAME_END} \
  --unidepth-repo /mnt/truenas-user-home/yiwen/a800_migrated_home/ego_annotation_remote/unidepth_work/UniDepth \
  --remote-root '{REMOTE_OUTPUT}/runtime_inputs/{CASE_ID}/raw_frame_manifest' \
  --local-root '{RUN_ROOT}/input/raw_frame_manifest' \
  --source-width {SOURCE_WIDTH} \
  --source-height {SOURCE_HEIGHT}"
rsync -a "{REMOTE}:{REMOTE_OUTPUT}/v19_runs/{CASE_ID}/measurements/depth_slam/unidepth_full_frame/" "{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/"
```

Required output: `{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz` and `qc_unidepth_full_frame_v3.json` copied back from the remote prediction output.

## P03b calibration contract

If prediction-side calibration metadata is present next to the input, copy it to `{RUN_ROOT}/state/calibration/` and record the source. Otherwise run:

Script: `scripts/build_v19_calibration_contract.py`

```bash
python scripts/build_v19_calibration_contract.py \
  --case "{CASE_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --unidepth-npz "{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-dir "{RUN_ROOT}/state/calibration" \
  --aggregation median \
  --square-focal
```

Required output: one calibration contract JSON under `{RUN_ROOT}/state/calibration/`.

## P04 MANO hand measurement

Script: `scripts/remote_run_hawor_export.sh` (calls `scripts/export_hawor_world.py`)

```bash
rsync -a "{INPUT_VIDEO}" "{REMOTE}:{REMOTE_OUTPUT}/runtime_inputs/{CASE_ID}/input_video.mp4"
ssh "{REMOTE}" "set -euo pipefail; cd '{REMOTE_BUNDLE}'; EGO_HAWOR_ROOT=/mnt/user-home/yiwen/ego_annotation_remote/hawor_work EGO_HAWOR_CASE='{CASE_ID}' EGO_HAWOR_CLIP='{REMOTE_OUTPUT}/runtime_inputs/{CASE_ID}/input_video.mp4' EGO_HAWOR_OUTPUT_DIR='{REMOTE_OUTPUT}/v19_runs/{CASE_ID}/measurements/hand_candidates/hawor_world' EGO_HAWOR_IMG_FOCAL='<focal_from_calibration_contract>' EGO_HAWOR_FORCE_FOCAL_CACHE_REFRESH=1 bash scripts/remote_run_hawor_export.sh"
rsync -a "{REMOTE}:{REMOTE_OUTPUT}/v19_runs/{CASE_ID}/measurements/hand_candidates/hawor_world/" "{RUN_ROOT}/measurements/hand_candidates/hawor_world/"
```

Required output: `{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz` and `qc_hawor_world_hands.json`.

## P05 object plan

Type: agent writes JSON from visual evidence.

Output: `{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json`.

Minimum fields per object: `object_id`, `description`, `physical_branch_hypotheses`, `evidence_frames`, `expected_visible_intervals`, `uncertainty_notes`.

## P06 object point prompts

Type: agent writes JSON from visual evidence.

Output for each object: `{RUN_ROOT}/measurements/object_candidates/object_point_prompts_agent/{OBJECT_ID}/object_point_prompts_vlm.json`.

Minimum fields: object id, prompt frame ids, positive points, negative points, active intervals, point coordinate frame.

## P07 object masks/tracks

Script: `scripts/run_sam2_vlm_points_multiobject.py`

```bash
rsync -a "{RUN_ROOT}/measurements/object_candidates/object_point_prompts_agent/" "{REMOTE}:{REMOTE_OUTPUT}/v19_runs/{CASE_ID}/measurements/object_candidates/object_point_prompts_agent/"
ssh "{REMOTE}" "set -euo pipefail; cd '{REMOTE_BUNDLE}'; CUDA_VISIBLE_DEVICES='{GPU_ID}' '{REMOTE_MODEL_PYTHON}' scripts/run_sam2_vlm_points_multiobject.py \
  --clip '{REMOTE_OUTPUT}/runtime_inputs/{CASE_ID}/input_video.mp4' \
  --point-root '{REMOTE_OUTPUT}/v19_runs/{CASE_ID}/measurements/object_candidates/object_point_prompts_agent' \
  --output-root '{REMOTE_OUTPUT}/v19_runs/{CASE_ID}/measurements/object_tracks/sam2_agent_points' \
  --checkpoint /mnt/user-home/yiwen/ego_annotation_remote/data/sam2.1_hiera_small.pt \
  --frame-start 0 \
  --frame-end {FRAME_END} \
  --sam2-image-width 960 \
  --render-width 960"
rsync -a "{REMOTE}:{REMOTE_OUTPUT}/v19_runs/{CASE_ID}/measurements/object_tracks/sam2_agent_points/" "{RUN_ROOT}/measurements/object_tracks/sam2_agent_points/"
```

Required output for each object: `{RUN_ROOT}/measurements/object_tracks/sam2_agent_points/{TRACK_ID}/sam2/sam2_track.json`.

## P08 base annotations

Script: `scripts/build_v19_base_annotations.py`

```bash
python scripts/build_v19_base_annotations.py \
  --case "{CASE_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --hawor-npz "{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --object-plan "{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json" \
  --sam2-output-root "{RUN_ROOT}/measurements/object_tracks/sam2_agent_points" \
  --calibration-contract "{RUN_ROOT}/state/calibration/<calibration_contract>.json" \
  --output-dir "{RUN_ROOT}/state/base_annotations"
```

Required output: `{RUN_ROOT}/state/base_annotations/annotations_v19_base.json`, `v19_base_physical_state.json`, and `v19_mano_bridge_from_hawor_world.npz`.

## P09 visible metric geometry

Script: `scripts/build_v19_visible_geometry_from_sam2_depth.py`

```bash
python scripts/build_v19_visible_geometry_from_sam2_depth.py \
  --case "{CASE_ID}" \
  --track-id "{TRACK_ID}" \
  --object-id "{OBJECT_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --sam2-root "{RUN_ROOT}/measurements/object_tracks/sam2_agent_points" \
  --depth-npz "{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-dir "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}" \
  --base-annotations "{RUN_ROOT}/state/base_annotations/annotations_v19_base.json" \
  --calibration-contract "{RUN_ROOT}/state/calibration/<calibration_contract>.json" \
  --object-plan "{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json" \
  --anchor-frame "{ANCHOR_FRAME}" \
  --preserve-source-index
```

Required output: `v19_visible_geometry_depth_fused_report.json` and visible-geometry annotations.

## P10 branch decision

Type: agent writes JSON/Markdown.

Output: `{RUN_ROOT}/state/physical_branch_decisions/{OBJECT_ID}.json` and evidence text appended to `state/v19_agent_evidence.md`.

If branch is not rigid, stop rigid path and render uncertainty from available state. If branch is rigid, continue.

## P11 rigid evidence bundle

Script: `scripts/build_v18_compact_rigid_evidence_bundle.py`

```bash
python scripts/build_v18_compact_rigid_evidence_bundle.py \
  --case "{CASE_ID}" \
  --object-id "{OBJECT_ID}" \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --depth-fused-report "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/v19_visible_geometry_depth_fused_report.json" \
  --output-root "{RUN_ROOT}/measurements/geometry_completion/rigid_evidence" \
  --selected-frame-idx "{ANCHOR_FRAME}" \
  --selection-note "runtime selected clean object evidence frame"
```

Required output: evidence bundle report and crop image path.

## P12 mesh prior

Script: `scripts/remote_run_trellis_shape_v3.py`

```bash
python scripts/remote_run_trellis_shape_v3.py \
  --repo /mnt/user-home/yiwen/ego_annotation_remote/trellis_work \
  --image "<evidence_crop_rgba>" \
  --output-dir "{RUN_ROOT}/measurements/geometry_completion/trellis_{OBJECT_ID}_seed42" \
  --seed 42
```

Required output: TRELLIS mesh report and mesh path.

## P13 mesh adaptation/completion

Script: `scripts/build_v18_compact_rigid_trellis_completion.py`

```bash
python scripts/build_v18_compact_rigid_trellis_completion.py \
  --evidence-report "<evidence_bundle_report>" \
  --trellis-report "<trellis_report>" \
  --output-dir "{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42"
```

Required output: completion report and completed mesh.

## P14 visible-frame pose fit

Script: `scripts/fit_v18_compact_rigid_object_pose.py`

```bash
python scripts/fit_v18_compact_rigid_object_pose.py \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --completion-report "{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42/v18_compact_rigid_trellis_completion_report.json" \
  --object-id "{OBJECT_ID}" \
  --output-dir "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_visible_pose_fit"
```

Required output: object pose fit report.

## P15 temporal rigid pose graph

Script: `scripts/solve_v19_rigid_object_pose_graph.py`

```bash
python scripts/solve_v19_rigid_object_pose_graph.py \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --pose-report "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_visible_pose_fit/v18_compact_rigid_object_pose_fit_report.json" \
  --completion-report "{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42/v18_compact_rigid_trellis_completion_report.json" \
  --object-id "{OBJECT_ID}" \
  --output-dir "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_rigid_pose_graph"
```

Required output: rigid pose graph report.

## P16 MANO/object constraint measurement

Script: `scripts/build_v18_mano_object_constraint_state.py`

```bash
python scripts/build_v18_mano_object_constraint_state.py \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --hawor-npz "{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --pose-report "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_rigid_pose_graph/v19_rigid_object_pose_graph_report.json" \
  --completion-report "{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42/v18_compact_rigid_trellis_completion_report.json" \
  --output-dir "{RUN_ROOT}/measurements/contact_nonpenetration/{OBJECT_ID}_mano_object_constraint" \
  --object-id "{OBJECT_ID}"
```

Required output: MANO/object constraint state.

## P17 contact/occlusion prior rows

Type: agent writes interval judgment JSON, then script consumes it.

Agent output: `{RUN_ROOT}/state/agent_interaction_judgments/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}.json`.

Script: `scripts/build_v19_visible_contact_ownership_factor.py`

```bash
python scripts/build_v19_visible_contact_ownership_factor.py \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --case "{CASE_ID}" \
  --target-entity-id "object:{OBJECT_ID}" \
  --frame-span "{INTERVAL_START}" "{INTERVAL_END}" \
  --output-root "{RUN_ROOT}/measurements/contact_visibility_factors/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}" \
  --agent-interaction-judgment "{RUN_ROOT}/state/agent_interaction_judgments/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}.json"
```

Required output: visible contact/ownership factor report.

## P18 interval MANO correction

Script: `scripts/solve_v18_joint_mano_interval_trajectory.py`

```bash
python scripts/solve_v18_joint_mano_interval_trajectory.py \
  --case "{CASE_ID}" \
  --object-id "object:{OBJECT_ID}" \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --pose-report "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_rigid_pose_graph/v19_rigid_object_pose_graph_report.json" \
  --completed-mesh "<completed_mesh_ply>" \
  --depth-npz "{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-dir "{RUN_ROOT}/measurements/mano_interval_correction/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}" \
  --start-frame "{INTERVAL_START}" \
  --end-frame "{INTERVAL_END}" \
  --sides left right \
  --factor-report "<visible_contact_ownership_factor_report>" \
  --optimize-contact-state \
  --visible-surface-depth-order-term
```

Required output: interval MANO trajectory state.

## P19 full-duration render

Script: `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py`

```bash
python scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py \
  --case "{CASE_ID}" \
  --object-label "{OBJECT_ID}" \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --pose-report "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_rigid_pose_graph/v19_rigid_object_pose_graph_report.json" \
  --completed-mesh "<completed_mesh_ply>" \
  --temporal-mano-state "{RUN_ROOT}/measurements/mano_interval_correction/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}/{CASE_ID}/v18_joint_mano_interval_trajectory_state.json" \
  --output-root "{RUN_ROOT}/renders/{OBJECT_ID}_rigid_mano_runtime"
```

Required output: full-duration overlay/world/side-by-side render branch.

## P20 publish canonical render names

Script: `scripts/publish_v19_render_artifact.py`

```bash
python scripts/publish_v19_render_artifact.py \
  --overlay "<render_branch_overlay_mp4>" \
  --world "<render_branch_world_mp4>" \
  --side-by-side "<render_branch_side_by_side_mp4>" \
  --interval-state "{RUN_ROOT}/measurements/mano_interval_correction/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}/{CASE_ID}/v18_joint_mano_interval_trajectory_state.json" \
  --output-dir "{RUN_ROOT}/renders/v19_published_runtime" \
  --canonical-dir "{RUN_ROOT}/renders" \
  --replace-canonical \
  --title "V19 runtime prediction {CASE_ID}"
```

Required output:

- `{RUN_ROOT}/renders/v19_overlay.mp4`
- `{RUN_ROOT}/renders/v19_world.mp4`
- `{RUN_ROOT}/renders/v19_side_by_side.mp4`

## P21 visual consumption

Type: agent inspects the rendered videos and state rows.

Required output: append to `{RUN_ROOT}/state/v19_agent_evidence.md` a concrete statement of which physical mechanisms worked, which failed, and which state variables remain unresolved.
