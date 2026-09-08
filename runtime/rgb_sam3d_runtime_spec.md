# V19 RGB / SAM3D Runtime Spec

This is the only runtime execution spec; `runtime/rgb_sam3d_system_prompt.md` establishes the prediction-executor role. Internal dependencies and hashes are in `runtime/bundle_manifest.json`. It defines state ontology, execution policy, exact phase order, scripts, command templates, required outputs, and stop conditions. Do not inspect or mention any file that is not named by this spec.

## Runtime inputs

The launch provides:

- `{INPUT_VIDEO}`: original egocentric RGB bytes copied to `{RUN_ROOT}/input/source.mp4` before launch;
- `{RUN_ROOT}`: fresh output run root;
- `{CASE_ID}`: case id;
- this runtime workspace;
- prediction-side sensor metadata, if present next to the input.

## Runtime outputs

The runtime output is a prediction run root containing `input/`, `measurements/`, `state/`, `renders/`, and `logs/`. The renderer consumes `state/`. Logs and measurements are provenance, not final annotations. Treat `{RUN_ROOT}/logs/harness_events.jsonl` as append-only JSONL: append one complete JSON object per event and never overwrite prior events.

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

1. Execute in dependency order: P00/P01/P02; optional intrinsics probe; P03b camera; final conditioned P03; P03c; P04 onward.
2. Do not discover or substitute scripts. Each script phase names the script to run.
3. For an agent-write phase, write only the specified JSON/Markdown artifact and preserve uncertainty.
4. Bind placeholders from launch arguments, phase outputs, or this spec. If a placeholder cannot be bound without searching outside the bundle, record the unresolved placeholder as a blocker.
5. The runtime itself runs on the A800 compute host. Heavy model phases and light metadata/state phases both read and write the same A800/truenas run root; do not copy phase outputs to any other machine during prediction.
6. Infrastructure is out of scope for runtime. Launch preflight is complete before start. Execute prediction phases only; if a named phase command fails, record that phase blocker and stop.
7. Do not run scoring or comparisons inside this runtime run.
8. Before each GPU-heavy phase, recheck live GPU memory/utilization and compute processes; if the configured GPU is busy, record the live process/memory evidence and stop that launch; do not take a different GPU without launch authorization.
9. Do not use sleep, polling loops, or idle waits. Long-running jobs need durable command logs/status files and inspectable job handles.

## Declared compute and asset targets

The launch-preflighted `runtime/runtime_config.json` binds model/checkpoint and interpreter paths. No installation, downloading, environment repair, historical clip predictions, or alternate reconstruction model is part of a prediction phase. All clip-specific inputs and outputs stay inside `{RUN_ROOT}`; only weights and model-source assets are reusable.

- Host: `{COMPUTE_HOST}`; physical GPU `{GPU_ID}` maps to logical `cuda:0`.
- General Python `{MAIN_PYTHON}`, depth Python `{UNIDEPTH_PYTHON}`, detector Python `{OWLV2_PYTHON}`, demo Python `{DEMO_PYTHON}`, RGB-feature Python `{RGB_PYTHON}`.
- HaWoR checkout `{HAWOR_ROOT}`, UniDepth source `{UNIDEPTH_REPO}`, model `{UNIDEPTH_MODEL}`, OWLv2 model `{OWLV2_MODEL}`, SAM2 checkpoint `{SAM2_CHECKPOINT}`.
- SAM3D source/config and offline MoGe/DINOv2 assets are declared in `runtime/runtime_config.json`. The SAM3D runner verifies the pinned assets before inference.
- MANO model assets `{MANO_MODELS}` and WiLoR source `{WILOR_ROOT}` are read-only model assets.

## Current-run source binding

`input/runtime_input_contract.json` supplies a unique prepared `run_id`, absolute `run_root`, original video hash and bundle identity. All named frontend and later producer commands must run through the fixed stage entries. Do not execute producer scripts directly or copy a prior prediction into the new run root. A successful command records input/output/code hashes tied to this identity; downstream stages verify the chain and all generated files, including masks/meshes. There is no cache-registration command. These records show command provenance, not reconstruction quality or physical authority.

P05, the P09 anchor decision, P10 and P17 judgments are authored from current visual evidence, not model-execution records. Add a top-level `visual_judgment_provenance` object with:
- `kind: current_run_visual_judgment`;
- `phase`: `P05`, `P09_anchor`, `P10` or `P17_judgment` respectively;
- `run_identity`: exact `run_id`, `run_root`, `input_video_sha256`, `bundle_manifest_sha256` copied from the prepared contract;
- `evidence_sha256`: absolute current evidence paths mapped to their SHA-256 values. P05 includes inspected P01 frames/manifest; anchor includes the P09 proposal/review image; P10 includes P09 visible geometry; P17 includes the current P15 pose report and inspected raw frames.

Do not author this provenance on behalf of unseen images or relabel copied judgments. It distinguishes visual judgment from measurement execution and does not make a judgment certain. The fixed frontend reads detector prompts/frames/intervals and optional `track_id` from the single selected P05 object. P10 must explicitly state `physical_branch: rigid` to enter P11. Alternative branches remain unresolved/nonrigid rather than entering rigid reconstruction.

External model/source/interpreter identities are bound in the bundle manifest and rechecked at preparation and stage use. Optional environment overlays cannot enable GPU/network access during import preflight.

## Stop condition

If a phase cannot run because an input, script, model asset, or environment is missing, write:

`{RUN_ROOT}/state/runtime_blockers/<PHASE_ID>.json`

with phase id, missing component, blocked state variable, evidence, and next required repair. Stop that branch rather than inventing substitute outputs.

## Placeholders

- `{FRAME_END}`: last frame index from P01 manifest.
- `{SOURCE_WIDTH}`, `{SOURCE_HEIGHT}`: source video resolution from P01 manifest.
- `{GPU_ID}`: selected A800 GPU from P02.
- `{SENSOR_CALIBRATION_METADATA}`: launcher-supplied prediction-side camera metadata path, or the empty string when the dataset/input provides none. The runtime must not search unrelated data roots for it.
- `{SENSOR_SOURCE_VIDEO}`: launcher-supplied local path to the exact RGB video associated with `{SENSOR_CALIBRATION_METADATA}`. Sensor-first resolution hashes it against `{INPUT_VIDEO}` and rejects a mismatch.
- `{SENSOR_CALIBRATION_AUTHORITY}`: one of `dataset_sensor_calibration`, `prediction_side_sensor_metadata`, or `explicit_user_calibration` when metadata is supplied.
- `{SENSOR_FRAME_INTRINSICS_KEY}`: optional per-frame `[fx,fy,cx,cy]` field name; empty when the supplied metadata has a top-level K.
- `{OBJECT_ID}`: object id chosen in P05.
- `{TRACK_ID}`: SAM2 track id for `{OBJECT_ID}`.
- `{ANCHOR_FRAME}`: selected clean object evidence frame.
- `{INTERVAL_START}`, `{INTERVAL_END}`: selected physical interval for MANO/object correction.
- `<completed_mesh_ply>`: completed mesh path from P13.
- `<visible_contact_ownership_factor_report>`: factor report from P17.
- `<render_branch_overlay_mp4>`, `<render_branch_world_mp4>`, `<render_branch_side_by_side_mp4>`: P19 render outputs.

# Phase graph

## P00 startup records

Type: agent writes JSON/Markdown; preserve the prepared input contract without overwriting it.

Inputs/outputs:

- `{RUN_ROOT}/input/runtime_input_contract.json`
- `{RUN_ROOT}/logs/harness_events.jsonl`
- `{RUN_ROOT}/state/v19_physical_state.json`
- `{RUN_ROOT}/state/v19_uncertainty_state.json`
- `{RUN_ROOT}/state/v19_agent_evidence.md`

The input contract must preserve the launch-provided `input_video_sha256` and input path. State after phase: unresolved camera, hands, objects, contact, occlusion, and nonpenetration.

## P01 raw frame manifest

Script: `scripts/build_v19_raw_frame_manifest.py`

```bash
"{MAIN_PYTHON}" scripts/run_v19_frontend_stage.py --phase P01 \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

Required output: `{RUN_ROOT}/input/raw_frame_manifest/manifest.json`.

## P02 A800 host probe and GPU selection

Type: bash command.

```bash
set -euo pipefail
hostname
df -h "{RUN_ROOT}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Required output: append a log event to `{RUN_ROOT}/logs/harness_events.jsonl` with selected `{GPU_ID}` and successful A800 host probe. There is no per-phase bundle sync or raw-frame-manifest staging because Pi, the input video, and the run root are already on the A800 host.

## P03a/P03b/P03/P03c/P04 camera, depth and MANO

After P01 and the live P02 probe, execute exactly:

```bash
"{MAIN_PYTHON}" scripts/run_v19_sam3d_integration_stage.py --stage camera \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

The command binds original input identity and the complete P01 frame/time map before any measurement. Its internal commands use:

1. `scripts/run_unidepth_full_frame_v3.py` unconditioned into `measurements/depth_slam/unidepth_intrinsics_probe` **only if no calibration was supplied** in the prepared input contract.
2. `scripts/resolve_v19_camera_contract.py` (P03b) into `state/calibration/v19_camera_calibration_contract.json`. Supplied sensor metadata must match the copied RGB bytes and complete timeline. It never falls back after a supplied calibration error. Without supplied calibration it aggregates the probe, with square focal for the supported HaWoR model API.
3. `scripts/run_unidepth_full_frame_v3.py` conditioned on that contract (`--camera-input-plane manifest_rgb --camera-output-plane source_rgb`) into `measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz`. Radius-to-exact-source-ray reprojection is required by this producer; the camera-head K remains diagnostic.
4. `scripts/adapt_v19_depth_to_camera_contract.py` with `--depth-plane source_rgb` into `state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz`. Metadata-only ray relabeling is forbidden; depth values/raster/frame IDs and the exact conditioned K/contract must agree.
5. `scripts/remote_run_hawor_export.sh` / `scripts/export_hawor_world.py` with an isolated byte-identical `input/hawor_sequence/{CASE_ID}.mp4`, the full source `[fx,fy,cx,cy]`, and exact source focal. The released HaWoR API requires square focal: non-square supplied K is a concrete model/API blocker, never averaged or relabeled. Required outputs are `measurements/hand_candidates/hawor_world/hawor_world_hands.npz` and `qc_hawor_world_hands.json`.

Each subcommand has a fresh output destination and durable `logs/integration/` command/event records. Failure stops subsequent commands. Heavy subcommands recheck the selected GPU and stop rather than switching devices. No machine `paths.env` is read. Model/checkpoint placement comes only from the bound config.

## P05 object plan

Type: agent writes JSON from visual evidence.

Output: `{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json`.

The JSON root must contain an `objects` list. Minimum fields per object: `object_id`, `description`, `physical_branch_hypotheses`, `evidence_frames`, `expected_visible_intervals`, `uncertainty_notes`, `detector_text_prompts`, `detector_prompt_frames`, and `detector_active_interval`. `detector_text_prompts` are open-vocabulary object queries such as `keyboard.` / `computer keyboard.`; they are not pixel coordinates. `detector_prompt_frames` must be representative visible frames selected from the raw video, and `detector_active_interval` must be a frame span over which the object is expected to be physically present or uncertain.

## P06 object grounded detector box prompts

Script: `scripts/build_v19_owlv2_object_box_prompts.py`

The default P06 source is OWLv2 text-conditioned detection, not VLM/agent pixel clicks. A VLM/agent may name the target object, write open-vocabulary text prompts, and choose representative frames in P05, but it must not be treated as the source of pixel-accurate click coordinates. For each rigid/manipulated object selected in P05, run OWLv2 with the object's `detector_text_prompts` on `detector_prompt_frames`, then write SAM2 prompt JSON containing `box_xyxy` in the detector image coordinate frame. Do not hard-code a category-specific object id, prompt string, or active interval in the runtime command; keyboard prompts are an example P05 object-plan value, not the P06 contract.

```bash
"{MAIN_PYTHON}" scripts/run_v19_frontend_stage.py --phase P06 \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

Required output for each object: `{RUN_ROOT}/measurements/object_candidates/object_box_prompts_owlv2/{OBJECT_ID}/object_point_prompts_vlm.json` and `v19_owlv2_object_box_prompt_report.json`. The prompt JSON must contain `prompt_source=owlv2_text_grounded_detector_boxes`, `box_xyxy` on visible prompt frames, and a coordinate declaration matching the frame images used by the detector. If OWLv2 produces no usable box for the target object, write a P06 blocker and stop; do not replace it with VLM/agent click coordinates as the default path.

## P07 object masks/tracks

Script: `scripts/run_sam2_vlm_points_multiobject.py`

```bash
"{MAIN_PYTHON}" scripts/run_v19_frontend_stage.py --phase P07 \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

Required output for each object: `{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points/{TRACK_ID}/sam2/sam2_track.json` written directly under the A800/truenas run root.

Required P07 self-check before P08: inspect `qc_sam2_multiobject_points.json`, P06 OWLv2 box review frames, prompt contract reports, and the SAM2 overlay/mask review for representative prompted frames, visible gaps inside expected active intervals, and any frames later used for rigid fitting/evidence. A mask that tracks a hand/sleeve/table edge while the object is visible is a hard P07 failure, not noisy-but-usable evidence. A mask that contains the target object but also broad hand/table/arm support is still a wrong object-support mask and must not become a rigid pose or geometry-completion observation. If OWLv2 boxes are loose or wrong, repair the grounded detector query/frame set/threshold or use another text-grounded detector; do not return to VLM/agent pixel-click prompting as the default. After at least one grounded detector rerun, remaining local mask gaps or low-confidence frames are not by themselves a stop condition when the accepted masks preserve the object identity on usable evidence frames; record those gaps as missing/uncertain mask observations and continue so the rigid branch can complete the full timeline in P15. The pipeline must not let an obvious wrong object track become a rigid pose observation, and it also must not prevent a rigid object from reaching P15 merely because local SAM2 evidence is missing in some visible frames.

## P08 base annotations

Script: `scripts/build_v19_base_annotations.py`

```bash
"{MAIN_PYTHON}" scripts/run_v19_frontend_stage.py --phase P08 \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

Required output: `{RUN_ROOT}/state/base_annotations/annotations_v19_base.json`, `v19_base_physical_state.json`, and `v19_mano_bridge_from_hawor_world.npz`.

P08 must prove that P04 HaWoR consumed the same full `source_rgb` camera contract. Retain source inference-plane K and its affine separately from the active source K; replacing K metadata does not recalibrate MANO or camera trajectories. A contract mismatch must remain explicitly unresolved and cannot be promoted by later image fitting.

## P09 visible metric geometry and anchor proposal

Script: `scripts/build_v19_visible_geometry_from_sam2_depth.py`

P09 is intentionally two-step. First propose anchor candidates from the same SAM2/depth/camera evidence without committing to a canonical object frame:

```bash
"{MAIN_PYTHON}" scripts/run_v19_frontend_stage.py --phase P09-propose \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

Required proposal outputs: `anchor_candidate_proposals.json` and `anchor_candidate_review.jpg`. The numeric proposal score is not an acceptance gate. The runtime agent must inspect the review sheet as an image, compare raw appearance/mask/depth/hand-removal summaries, and write an explicit anchor decision:

```text
{RUN_ROOT}/state/anchor_decisions/{OBJECT_ID}.json
```

Minimum decision fields: `object_id`, `selected_anchor_frame_idx`, `selected_candidate_rank`, `visual_rationale`, `rejected_candidate_observations`, `uncertainties`, and `candidate_report_path`. Favor the cleanest full-object evidence: large visible support, low hand overlap, non-border mask, coherent object outline/key-grid or texture, reliable depth, and metric extent consistent with neighboring plausible frames. Do not choose a frame solely because it has the largest mask or most sampled points.

Then run canonical visible geometry with the selected anchor. This second command must use `--require-anchor-frame` so the pipeline cannot silently fall back to the max-point frame:

```bash
"{MAIN_PYTHON}" scripts/run_v19_frontend_stage.py --phase P09 \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

Required output: `v19_visible_geometry_depth_fused_report.json` and visible-geometry
annotations. P09 must treat hand-owned pixels as occlusion/uncertainty, not visible object
surface. It must project the full prediction-side HaWoR/MANO camera vertices and triangle
faces referenced by P08, subtract that triangle silhouette, and record the result under
`object_surface_ownership_filter`. A hand bbox is diagnostic only: it may never remove or
retain surfels, and missing/malformed MANO geometry fails that frame closed. The per-object
`mask_path` written to annotations must be this object-owned mask, so P11 native inputs do
not include hand pixels.

P09 must also record `first_surface_depth_ownership` before backprojection. A global
percentile/MAD crop is forbidden because it can delete a real sloped face or a disconnected
rigid part. The accepted surfels must come from per-component MAD seeds expanded through
UniDepth-confidence and local-depth geodesic support. Support below 90%, missing confidence,
or small-component loss above the fixed gate fails the frame closed. Rejection farther than
10 px inside the owned mask is not automatically background: it passes only for at most five
isolated raster samples, or when it is at most 1.5% of valid owned depth and at least 95% of
those pixels have a UniDepth predicted-error proxy above their component seed threshold.
Otherwise the frame fails closed, preserving possible real thickness as unresolved. A
raw/accepted extent ratio above 2 is allowed only when the report explicitly proves this
high-support boundary/interior uncertainty quarantine; it must never influence voxel/RMS
scale. Do not lower these gates or choose a contaminated anchor to keep the run moving. When mask and depth rasters differ, use `cv2.INTER_NEAREST_EXACT` so
discrete mask samples follow the V2 OpenCV half-pixel affine; legacy
`cv2.INTER_NEAREST` is invalid for this contract.

## P10 branch decision

Type: agent writes JSON/Markdown.

Output: `{RUN_ROOT}/state/physical_branch_decisions/{OBJECT_ID}.json` and evidence text appended to `state/v19_agent_evidence.md`.

If branch is not rigid, stop rigid path and render uncertainty from available state. If branch is rigid, continue.

## P11 rigid evidence bundle

Script: `scripts/build_v18_compact_rigid_evidence_bundle.py`

```bash
"{MAIN_PYTHON}" scripts/run_v19_frontend_stage.py --phase P11 \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

Required output: evidence bundle report and object crop image path. P11 must use the same `selected_anchor_frame_idx` recorded in P09's anchor-decision JSON; it must not independently rerank evidence frames by mask area or point count. SAM3D consumes `selected.raw_frame_path` and the matching object-owned `selected.mask_path` as full-scene RGB plus mask, with no external pointmap or independent crop/rotation. The legacy RGBA crop is preserved as P11 evidence only; no second reconstruction backend runs.

## P12/P13 SAM3D native reconstruction and canonical geometry

After the visual P09 anchor decision and rigid P10 decision, write `{RUN_ROOT}/state/object_selection.json` with exactly the selected `case`, bare `object_id`, and `anchor_frame` from those same decisions. The object ID must equal P09's explicitly passed object ID; do not select a new object or anchor here.

```bash
"{MAIN_PYTHON}" scripts/run_v19_sam3d_integration_stage.py --stage geometry \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

This runs `scripts/run_v19_sam3d_reconstruction.py`, native inputs, `scripts/remote_run_sam3d_objects_mesh_v7.py`, the native-to-predicted-metric canonical bridge, and face support labeling. Required output: `{RUN_ROOT}/measurements/geometry_completion/sam3d_{OBJECT_ID}/completion_report.json`. Preserve its original intact generated render mesh and separate observed/support-qualified physical surface. No single-view generated mesh becomes sign/collision authority.

## P14/RGB/P14b/P15 full-timeline object pose

```bash
"{MAIN_PYTHON}" scripts/run_v19_sam3d_integration_stage.py --stage pose \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

The named mechanisms are visible-surface P14, object-owned SuperPoint/LightGlue calibrated RGB-D edges from THIS run through `scripts/run_v19_offline_rgb_edges.py` / `scripts/build_v20_global_rgb_orientation_edges.py`, weighted SE3 spanning initialization plus robust SO3 loop refinement, P14b multiview support, and the original P15 temporal graph. P14 direct/latent measurement distinctions remain; no historical pose file is an input. Generated geometry does not provide the RGB/P14 pose observations. Missing observations retain uncertainty, not direct-metric status. The initializer `scripts/build_v19_rgb_pose_initializer.py` preserves P14 direct/latent eligibility and binds `inputs.completed_mesh` for `scripts/filter_v19_rigid_completion_multiview_support.py`; `scripts/solve_v19_rigid_object_pose_graph.py` receives the full timeline. Required P15 output: `{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}/temporal/v19_rigid_object_pose_graph_report.json`.

## P16/P17/P18/P18b physical interaction and metric MANO preservation

Type: visual agent judgment followed by deterministic scripts. Inspect raw frames and the current object/hand geometry across the full sequence. Write `{RUN_ROOT}/state/agent_interaction_judgments/{OBJECT_ID}.json` with `case`, `target_entity_id: object:{OBJECT_ID}`, `status: ok`, and `interaction_judgments`.

Each judgment has `start_frame`, `end_frame`, `hand_side`, `target_entity_id`, `contact_state` (`likely_contact`, `possible_contact`, `no_contact`, `unresolved`), `occlusion_relation` (`hand_in_front_of_object`, `object_in_front_of_hand`, `object_partially_occluded_by_hand`, `no_visible_occlusion`, `unresolved`), `confidence`, `visual_evidence`, and `uncertainty`. The P17 consumer also requires numeric `contact_prior_probability` in [0,1] and nonnegative `contact_support_uncertainty_m`. Supply `depth_reliability` as `hand_depth_unreliable`, `object_depth_reliable`, `mixed_or_unresolved`, or `not_evaluated`; optional `contact_weight_multiplier` must be nonnegative. These are agent-produced uncertain priors, not measured contact. Missing or malformed fields stop the phase; the dispatcher does not invent them. Use explicit intervals for both available hands; do not force contact through unknown/occluded frames. Do not use fitted demo geometry to make these judgments.

```bash
"{MAIN_PYTHON}" scripts/run_v19_sam3d_integration_stage.py --stage interaction \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

This executes original P16 measurement, P17 visible contact/ownership factors, full-interval P18 MANO correction and P18b metric-MANO/contact-surface separation. Original geometric support/readiness/sign-authority checks are not relaxed. P18 enables `--require-active-full-K-mano-contract`; unready P15 takes the existing source-MANO quarantine path rather than relabeling readiness. Quarantined physical state remains quarantined; this does not silently promote an unsigned surface to a signed volume. The optional observed-only signed-geometry producer remains an explicit evidence route rather than generated-shape fallback.

## P19 physical state and original renderer

```bash
"{MAIN_PYTHON}" scripts/run_v19_sam3d_integration_stage.py --stage physical-render \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

P19 is preserved independently at `{RUN_ROOT}/state/render_state/{OBJECT_ID}_rigid_render_state.json`. Its source metric MANO, corrected contact-surface hypotheses, support and uncertainty are not rewritten by subsequent visual fitting.

## D01/D02 additive visual fit

```bash
"{MAIN_PYTHON}" scripts/run_v19_sam3d_integration_stage.py --stage demo \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

Fit the same run's intact SAM3D triangles, P15 object pose and real HaWoR/MANO to image and eligible P17 contact hypotheses. Actual MANO articulation/root and bounded visual similarity adjustments are replayed; no arbitrary hand-vertex morph. Missing hands stay invalid and are not rendered. A ready or explicitly non-diagnostic P15 prediction may seed this additive fit without changing its source readiness; recursive prediction-source checks still apply. The fitted state is `demo_only`; source metric state and physical uncertainty are untouched. Changing display resolution does not change the fitter's 960-width reference pixel units.

## P20 shared final annotation, MP4 and world RRD

```bash
"{MAIN_PYTHON}" scripts/run_v19_sam3d_integration_stage.py --stage export \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

Required artifacts:
- `{RUN_ROOT}/final/annotation_layers.json` and registered relative assets;
- `{RUN_ROOT}/final/world.rrd`;
- `{RUN_ROOT}/final/video/annotation_layers.mp4` and render report.

The MP4 and RRD both consume this final package's normalized world geometry, shared camera and time axis. `metric_prediction` retains original P19/annotations/P17 data; `demo_visual` has its own object and hand meshes. Distinct canonical meshes require the reconstruction report's explicit identity/hash binding. Do not join a demo object with prediction hands or treat fitted contact as a measurement.

## P21 full-video inspection

Inspect the full MP4 plus representative beginning/middle/end and interaction transitions, not only metadata counts. Confirm real object and full MANO surfaces, visible alignment, camera/world convention, invalid-hand handling, layer identities and original duration. Record limitations and inspection evidence under `{RUN_ROOT}/state/final_visual_review.json`. Missing or nonsensical visible geometry is not success. No external publication occurs in this runtime.
