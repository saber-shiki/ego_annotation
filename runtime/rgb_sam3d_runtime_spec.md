# V19 RGB / SAM3D Runtime Spec

This is the only runtime instruction document. It defines state ontology, execution policy, exact phase order, scripts, command templates, required outputs, and stop conditions. Do not inspect or mention any file that is not named by this spec.

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

Type: agent writes JSON/Markdown.

Outputs:

- `{RUN_ROOT}/input/runtime_input_contract.json`
- `{RUN_ROOT}/logs/harness_events.jsonl`
- `{RUN_ROOT}/state/v19_physical_state.json`
- `{RUN_ROOT}/state/v19_uncertainty_state.json`
- `{RUN_ROOT}/state/v19_agent_evidence.md`

The input contract must preserve the launch-provided `input_video_sha256` and input path. State after phase: unresolved camera, hands, objects, contact, occlusion, and nonpenetration.

## P01 raw frame manifest

Script: `scripts/build_v19_raw_frame_manifest.py`

```bash
"{MAIN_PYTHON}" scripts/build_v19_raw_frame_manifest.py \
  --video "{INPUT_VIDEO}" \
  --output-dir "{RUN_ROOT}/input/raw_frame_manifest" \
  --render-width 960
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

## P03 camera-conditioned depth measurement

Script: `scripts/run_unidepth_full_frame_v3.py`

**Execution dependency:** when prediction-side sensor metadata is supplied (for a calibrated RGB input), execute P03b first, then run the final P03 command below. UniDepth
must consume the resolved `manifest_rgb` rays in its depth head. Its predicted camera-head K
is diagnostic only: the final source-plane camera-z raster is formed by resizing the metric
radius and explicitly projecting it onto exact `source_rgb` contract rays. P03c is not allowed
to relabel a disagreeing unchanged z raster after the fact. If no sensor metadata is supplied, run one
unconditioned intrinsics-probe pass into
`{RUN_ROOT}/measurements/depth_slam/unidepth_intrinsics_probe`, resolve P03b from that probe,
then run this final camera-conditioned P03 pass into the path below.

```bash
CUDA_VISIBLE_DEVICES='{GPU_ID}' '{UNIDEPTH_PYTHON}' scripts/run_unidepth_full_frame_v3.py \
  --manifest '{RUN_ROOT}/input/raw_frame_manifest/manifest.json' \
  --output-dir '{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame' \
  --camera-contract '{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json' \
  --camera-input-plane manifest_rgb \
  --camera-output-plane source_rgb \
  --frame-start 0 \
  --frame-end {FRAME_END} \
  --unidepth-repo {UNIDEPTH_REPO} \
  --model-id {UNIDEPTH_MODEL} \
  --remote-root '{RUN_ROOT}/input/raw_frame_manifest' \
  --local-root '{RUN_ROOT}/input/raw_frame_manifest' \
  --source-width {SOURCE_WIDTH} \
  --source-height {SOURCE_HEIGHT}
```

Required output: `{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz` and `qc_unidepth_full_frame_v3.json` written directly under the A800/truenas run root. The report must declare `depth_ray_geometry_reprojected:true`, the radius-to-exact-ray conversion, and a maximum internal conditioned-ray discrepancy no greater than the fixed 1° gate; camera-head K disagreement is retained as a diagnostic rather than made active.

### P03a intrinsics probe (only without supplied calibration)

When no sensor metadata is supplied, run this once before P03b. It is an estimated-camera probe, NOT the final conditioned depth archive.

```bash
CUDA_VISIBLE_DEVICES='{GPU_ID}' '{UNIDEPTH_PYTHON}' scripts/run_unidepth_full_frame_v3.py \
  --manifest '{RUN_ROOT}/input/raw_frame_manifest/manifest.json' \
  --output-dir '{RUN_ROOT}/measurements/depth_slam/unidepth_intrinsics_probe' \
  --frame-start 0 --frame-end {FRAME_END} \
  --unidepth-repo '{UNIDEPTH_REPO}' --model-id '{UNIDEPTH_MODEL}' \
  --remote-root '{RUN_ROOT}/input/raw_frame_manifest' --local-root '{RUN_ROOT}/input/raw_frame_manifest' \
  --source-width {SOURCE_WIDTH} --source-height {SOURCE_HEIGHT}
```

## P03b source-neutral camera/image-transform contract

Script: `scripts/resolve_v19_camera_contract.py`

The launcher binds `{SENSOR_CALIBRATION_METADATA}`; the runtime must not discover calibration outside the declared prediction input. If prediction-side sensor metadata is supplied, resolve it into the canonical V2 contract. A supplied-but-missing, malformed, variable-K, wrong-size, or wrong-timeline sensor contract is a P03b implementation/input failure and must not silently fall back. Only when the launcher explicitly supplies no sensor metadata may the original robust UniDepth aggregation be used as fallback.

```bash
SENSOR_METADATA='{SENSOR_CALIBRATION_METADATA}'
SENSOR_INTRINSICS_KEY='{SENSOR_FRAME_INTRINSICS_KEY}'
CAMERA_ARGS=()
if [[ -n "$SENSOR_METADATA" ]]; then
  CAMERA_ARGS+=(
    --sensor-calibration-contract "$SENSOR_METADATA"
    --sensor-calibration-authority '{SENSOR_CALIBRATION_AUTHORITY}'
    --prediction-source-video '{INPUT_VIDEO}'
    --sensor-source-video '{SENSOR_SOURCE_VIDEO}'
  )
  if [[ -n "$SENSOR_INTRINSICS_KEY" ]]; then
    CAMERA_ARGS+=(--sensor-frame-intrinsics-key "$SENSOR_INTRINSICS_KEY")
  fi
else
  CAMERA_ARGS+=(
    --unidepth-npz "{RUN_ROOT}/measurements/depth_slam/unidepth_intrinsics_probe/unidepth_full_frame_depth_v3.npz"
    --fallback-reason "launcher supplied no prediction-side sensor calibration"
    --aggregation median
    --square-focal
  )
fi

"{MAIN_PYTHON}" scripts/resolve_v19_camera_contract.py \
  --case "{CASE_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --output-dir "{RUN_ROOT}/state/calibration" \
  "${CAMERA_ARGS[@]}"
```

Required outputs:

- `{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json`
- `{RUN_ROOT}/state/calibration/v19_camera_calibration_intrinsics.npz`

Both sensor-first and estimated-fallback modes use schema `v19_camera_image_transform_contract_v2`. In sensor-first mode, the resolver must prove exact prediction/sensor source-video SHA256 identity and matching frame/time timeline before accepting K; same frame count alone is insufficient. The contract declares native calibration K plus explicit `calibration`, `source_rgb`, `manifest_rgb`, `sam2_mask`, and `render` planes under `K_plane = A_plane_from_calibration @ K_calibration`. The fallback mode reuses `build_v19_calibration_contract.py` aggregation functions; the legacy builder remains available but is no longer the source-neutral runtime entry point.

## P03c bind metric depth to the resolved camera plane

Script: `scripts/adapt_v19_depth_to_camera_contract.py`

```bash
"{MAIN_PYTHON}" scripts/adapt_v19_depth_to_camera_contract.py \
  --source-depth-npz "{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --camera-contract "{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json" \
  --depth-plane source_rgb \
  --output-dir "{RUN_ROOT}/state/calibration/depth_camera_contract"
```

Required output:

`{RUN_ROOT}/state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz`

The adapter must prove that depth values, dtype, frame IDs, and raster size are unchanged,
that source K already equals the requested `source_rgb` K, and—when P03 declares camera
conditioning—that the inference contract hash and output plane match. It preserves the
source K under `source_estimated_intrinsics_fx_fy_cx_cy` and embeds the resolved contract
SHA256, plane, K, and `A_depth_from_calibration`. A metadata-only K relabel is a hard blocker:
the forensic `--allow-metadata-only-ray-relabel` option is forbidden in production and P09
rejects such an archive. All metric backprojection must consume this camera-bound archive.

## P04 MANO hand measurement

Script: `scripts/remote_run_hawor_export.sh` (calls `scripts/export_hawor_world.py`)

Extract the HaWoR focal from the chosen contract's `source_rgb` image plane, never from native calibration-plane K when an image affine is present and never from diagnostics/outlier tables. Prefer top-level `source_plane_intrinsics_fx_fy_cx_cy`; legacy contracts may fall back to top-level `focal_px`, `focal_geom_px`, `intrinsics_fx_fy_cx_cy[0]`, then `intrinsics.fx`. If none exists, stop with a P04 blocker before running HaWoR.

```bash
CONTRACT='{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json'
HAWOR_K=$("{MAIN_PYTHON}" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(" ".join(map(str,c["source_plane_intrinsics_fx_fy_cx_cy"])))' "$CONTRACT")
FOCAL=$("{MAIN_PYTHON}" - "$CONTRACT" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
value = None
source_plane = data.get("source_plane_intrinsics_fx_fy_cx_cy")
if isinstance(source_plane, list) and len(source_plane) >= 2 and all(isinstance(v, (int, float)) for v in source_plane[:2]):
    value = float((float(source_plane[0]) * float(source_plane[1])) ** 0.5)
for key in ("focal_px", "focal_geom_px"):
    if value is None and isinstance(data.get(key), (int, float)):
        value = float(data[key])
        break
if value is None:
    intr = data.get("intrinsics_fx_fy_cx_cy")
    if isinstance(intr, list) and intr and isinstance(intr[0], (int, float)):
        value = float(intr[0])
if value is None:
    intr = data.get("intrinsics")
    if isinstance(intr, dict) and isinstance(intr.get("fx"), (int, float)):
        value = float(intr["fx"])
if value is None:
    raise SystemExit(f"missing canonical focal in {path}; do not scan diagnostics")
print(value)
PY
)
HAWOR_CLIP='{RUN_ROOT}/input/hawor_sequence/{CASE_ID}.mp4'
mkdir -p "$(dirname "$HAWOR_CLIP")"
if [ ! -s "$HAWOR_CLIP" ]; then cp -- '{INPUT_VIDEO}' "$HAWOR_CLIP"; fi
[ "$(sha256sum '{INPUT_VIDEO}' | awk '{print $1}')" = "$(sha256sum "$HAWOR_CLIP" | awk '{print $1}')" ]
CUDA_VISIBLE_DEVICES='{GPU_ID}' \
EGO_HAWOR_ROOT="{HAWOR_ROOT}" \
EGO_HAWOR_CASE='{CASE_ID}' \
EGO_HAWOR_CLIP="$HAWOR_CLIP" \
EGO_HAWOR_OUTPUT_DIR='{RUN_ROOT}/measurements/hand_candidates/hawor_world' \
EGO_HAWOR_IMG_FOCAL="$FOCAL" \
EGO_HAWOR_CAMERA_INTRINSICS_FX_FY_CX_CY="$HAWOR_K" \
EGO_HAWOR_FORCE_FOCAL_CACHE_REFRESH=1 \
bash scripts/remote_run_hawor_export.sh
```

Required output: `{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz` and `qc_hawor_world_hands.json` written directly under the A800/truenas run root.

## P05 object plan

Type: agent writes JSON from visual evidence.

Output: `{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json`.

Minimum fields per object: `object_id`, `description`, `physical_branch_hypotheses`, `evidence_frames`, `expected_visible_intervals`, `uncertainty_notes`, `detector_text_prompts`, `detector_prompt_frames`, and `detector_active_interval`. `detector_text_prompts` are open-vocabulary object queries such as `keyboard.` / `computer keyboard.`; they are not pixel coordinates. `detector_prompt_frames` must be representative visible frames selected from the raw video, and `detector_active_interval` must be a frame span over which the object is expected to be physically present or uncertain.

## P06 object grounded detector box prompts

Script: `scripts/build_v19_owlv2_object_box_prompts.py`

The default P06 source is OWLv2 text-conditioned detection, not VLM/agent pixel clicks. A VLM/agent may name the target object, write open-vocabulary text prompts, and choose representative frames in P05, but it must not be treated as the source of pixel-accurate click coordinates. For each rigid/manipulated object selected in P05, run OWLv2 with the object's `detector_text_prompts` on `detector_prompt_frames`, then write SAM2 prompt JSON containing `box_xyxy` in the detector image coordinate frame. Do not hard-code a category-specific object id, prompt string, or active interval in the runtime command; keyboard prompts are an example P05 object-plan value, not the P06 contract.

```bash
mkdir -p "{RUN_ROOT}/measurements/object_candidates/object_box_prompts_owlv2" "{RUN_ROOT}/renders/review_frames/P06_owlv2_object_boxes"
TEXT_PROMPT_ARGS=()
for PROMPT in "${DETECTOR_TEXT_PROMPTS[@]}"; do
  TEXT_PROMPT_ARGS+=(--text-prompt "$PROMPT")
done
CUDA_VISIBLE_DEVICES="{GPU_ID}" "{OWLV2_PYTHON}" scripts/build_v19_owlv2_object_box_prompts.py \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --output-root "{RUN_ROOT}/measurements/object_candidates/object_box_prompts_owlv2" \
  --review-dir "{RUN_ROOT}/renders/review_frames/P06_owlv2_object_boxes" \
  --case-id "{CASE_ID}" \
  --object-id "{OBJECT_ID}" \
  --track-id "{TRACK_ID}" \
  --description "{OBJECT_DESCRIPTION}" \
  "${TEXT_PROMPT_ARGS[@]}" \
  --prompt-frames "{DETECTOR_PROMPT_FRAMES_CSV}" \
  --active-start "{DETECTOR_ACTIVE_START}" \
  --active-end "{DETECTOR_ACTIVE_END}" \
  --owlv2-model {OWLV2_MODEL} \
  --box-threshold 0.03 \
  --device cuda \
  --box-only
```

Required output for each object: `{RUN_ROOT}/measurements/object_candidates/object_box_prompts_owlv2/{OBJECT_ID}/object_point_prompts_vlm.json` and `v19_owlv2_object_box_prompt_report.json`. The prompt JSON must contain `prompt_source=owlv2_text_grounded_detector_boxes`, `box_xyxy` on visible prompt frames, and a coordinate declaration matching the frame images used by the detector. If OWLv2 produces no usable box for the target object, write a P06 blocker and stop; do not replace it with VLM/agent click coordinates as the default path.

## P07 object masks/tracks

Script: `scripts/run_sam2_vlm_points_multiobject.py`

```bash
mkdir -p '{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points'
CUDA_VISIBLE_DEVICES='{GPU_ID}' '{MAIN_PYTHON}' scripts/run_sam2_vlm_points_multiobject.py \
  --clip '{INPUT_VIDEO}' \
  --point-root '{RUN_ROOT}/measurements/object_candidates/object_box_prompts_owlv2' \
  --output-root '{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points' \
  --checkpoint {SAM2_CHECKPOINT} \
  --frame-start 0 \
  --frame-end {FRAME_END} \
  --sam2-image-width 960 \
  --render-width 960 \
  --use-positive-prompt-box \
  --prompt-box-pad-ratio 0.18 \
  --prompt-box-min-pad-px 24
```

Required output for each object: `{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points/{TRACK_ID}/sam2/sam2_track.json` written directly under the A800/truenas run root.

Required P07 self-check before P08: inspect `qc_sam2_multiobject_points.json`, P06 OWLv2 box review frames, prompt contract reports, and the SAM2 overlay/mask review for representative prompted frames, visible gaps inside expected active intervals, and any frames later used for rigid fitting/evidence. A mask that tracks a hand/sleeve/table edge while the object is visible is a hard P07 failure, not noisy-but-usable evidence. A mask that contains the target object but also broad hand/table/arm support is still a wrong object-support mask and must not become a rigid pose or geometry-completion observation. If OWLv2 boxes are loose or wrong, repair the grounded detector query/frame set/threshold or use another text-grounded detector; do not return to VLM/agent pixel-click prompting as the default. After at least one grounded detector rerun, remaining local mask gaps or low-confidence frames are not by themselves a stop condition when the accepted masks preserve the object identity on usable evidence frames; record those gaps as missing/uncertain mask observations and continue so the rigid branch can complete the full timeline in P15. The pipeline must not let an obvious wrong object track become a rigid pose observation, and it also must not prevent a rigid object from reaching P15 merely because local SAM2 evidence is missing in some visible frames.

## P08 base annotations

Script: `scripts/build_v19_base_annotations.py`

```bash
"{MAIN_PYTHON}" scripts/build_v19_base_annotations.py \
  --case "{CASE_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --hawor-npz "{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --object-plan "{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json" \
  --sam2-output-root "{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points" \
  --calibration-contract "{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json" \
  --output-dir "{RUN_ROOT}/state/base_annotations"
```

Required output: `{RUN_ROOT}/state/base_annotations/annotations_v19_base.json`, `v19_base_physical_state.json`, and `v19_mano_bridge_from_hawor_world.npz`.

If P04 HaWoR was not rerun under the active camera contract (for example, an additive camera-only experiment intentionally inherits a frozen HaWoR archive), P08 must compare the inherited full-image HaWoR K with the active `source_rgb` K. For legacy HaWoR archives, reconstruct the source K as `[img_focal,img_focal,width/2,height/2]`, matching upstream HaWoR's full-image center convention. Any focal or principal-point mismatch must set `active_contract_reinference_required=true` in the base report and camera/hand rows. Hand state must retain the source HaWoR K separately from the active V19 K; replacing K metadata does not recalibrate frozen MANO/camera trajectories. Downstream MANO/contact promotion remains blocked until P04 is rerun or independently aligned.

## P09 visible metric geometry and anchor proposal

Script: `scripts/build_v19_visible_geometry_from_sam2_depth.py`

P09 is intentionally two-step. First propose anchor candidates from the same SAM2/depth/camera evidence without committing to a canonical object frame:

```bash
"{MAIN_PYTHON}" scripts/build_v19_visible_geometry_from_sam2_depth.py \
  --case "{CASE_ID}" \
  --track-id "{TRACK_ID}" \
  --object-id "{OBJECT_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --sam2-root "{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points" \
  --depth-npz "{RUN_ROOT}/state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz" \
  --output-dir "{RUN_ROOT}/measurements/object_geometry/anchor_candidates/{OBJECT_ID}" \
  --base-annotations "{RUN_ROOT}/state/base_annotations/annotations_v19_base.json" \
  --calibration-contract "{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json" \
  --depth-image-plane source_rgb \
  --mask-image-plane sam2_mask \
  --object-plan "{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json" \
  --preserve-source-index \
  --exclude-hand-regions \
  --hand-bbox-exclusion-pad-px 4 \
  --propose-anchor-candidates-only \
  --anchor-candidate-count 12
```

Required proposal outputs: `anchor_candidate_proposals.json` and `anchor_candidate_review.jpg`. The numeric proposal score is not an acceptance gate. The runtime agent must inspect the review sheet as an image, compare raw appearance/mask/depth/hand-removal summaries, and write an explicit anchor decision:

```text
{RUN_ROOT}/state/anchor_decisions/{OBJECT_ID}.json
```

Minimum decision fields: `object_id`, `selected_anchor_frame_idx`, `selected_candidate_rank`, `visual_rationale`, `rejected_candidate_observations`, `uncertainties`, and `candidate_report_path`. Favor the cleanest full-object evidence: large visible support, low hand overlap, non-border mask, coherent object outline/key-grid or texture, reliable depth, and metric extent consistent with neighboring plausible frames. Do not choose a frame solely because it has the largest mask or most sampled points.

Then run canonical visible geometry with the selected anchor. This second command must use `--require-anchor-frame` so the pipeline cannot silently fall back to the max-point frame:

```bash
"{MAIN_PYTHON}" scripts/build_v19_visible_geometry_from_sam2_depth.py \
  --case "{CASE_ID}" \
  --track-id "{TRACK_ID}" \
  --object-id "{OBJECT_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --sam2-root "{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points" \
  --depth-npz "{RUN_ROOT}/state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz" \
  --output-dir "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}" \
  --base-annotations "{RUN_ROOT}/state/base_annotations/annotations_v19_base.json" \
  --calibration-contract "{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json" \
  --depth-image-plane source_rgb \
  --mask-image-plane sam2_mask \
  --object-plan "{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json" \
  --anchor-frame "{ANCHOR_FRAME}" \
  --require-anchor-frame \
  --preserve-source-index \
  --exclude-hand-regions \
  --hand-bbox-exclusion-pad-px 4
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
"{MAIN_PYTHON}" scripts/build_v18_compact_rigid_evidence_bundle.py \
  --case "{CASE_ID}" \
  --object-id "{OBJECT_ID}" \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --depth-fused-report "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/v19_visible_geometry_depth_fused_report.json" \
  --output-root "{RUN_ROOT}/measurements/geometry_completion/rigid_evidence" \
  --selected-frame-idx "{ANCHOR_FRAME}" \
  --selection-note "agent-selected anchor from state/anchor_decisions/{OBJECT_ID}.json"
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

The named mechanisms are visible-surface P14, object-owned SuperPoint/LightGlue calibrated RGB-D edges from THIS run, weighted SE3 spanning initialization plus robust SO3 loop refinement, P14b multiview support, and the original P15 temporal graph. P14 direct/latent measurement distinctions remain; no historical pose file is an input. Generated geometry does not provide the RGB/P14 pose observations. Missing observations retain uncertainty, not direct-metric status. Required P15 output: `{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}/temporal/v19_rigid_object_pose_graph_report.json`.

## P16/P17/P18/P18b physical interaction and metric MANO preservation

Type: visual agent judgment followed by deterministic scripts. Inspect raw frames and the current object/hand geometry across the full sequence. Write `{RUN_ROOT}/state/agent_interaction_judgments/{OBJECT_ID}.json` with `case`, `target_entity_id: object:{OBJECT_ID}`, `status: ok`, and `interaction_judgments`.

Each judgment has `start_frame`, `end_frame`, `hand_side`, `target_entity_id`, `contact_state` (`likely_contact`, `possible_contact`, `no_contact`, `unresolved`), `occlusion_relation` (`hand_in_front_of_object`, `object_in_front_of_hand`, `object_partially_occluded_by_hand`, `no_visible_occlusion`, `unresolved`), `confidence`, `visual_evidence`, and `uncertainty`. Use explicit intervals for both available hands; do not force contact through unknown/occluded frames. Do not use fitted demo geometry to make these judgments.

```bash
"{MAIN_PYTHON}" scripts/run_v19_sam3d_integration_stage.py --stage interaction \
  --run-root '{RUN_ROOT}' --runtime-config runtime/runtime_config.json --gpu {GPU_ID}
```

This executes original P16 measurement, P17 visible contact/ownership factors, full-interval P18 MANO correction and P18b metric-MANO/contact-surface separation. Original geometric support/readiness/sign-authority checks are not relaxed. Quarantined physical state remains quarantined; this does not silently promote an unsigned surface to a signed volume. The optional observed-only signed-geometry producer remains an explicit evidence route rather than generated-shape fallback.

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

Fit the same run's intact SAM3D triangles, P15 object pose and real HaWoR/MANO to image and eligible P17 contact hypotheses. Actual MANO articulation/root and bounded visual similarity adjustments are replayed; no arbitrary hand-vertex morph. Missing hands stay invalid and are not rendered. The fitted state is `demo_only`; source metric state and physical uncertainty are untouched. Changing display resolution does not change the fitter's 960-width reference pixel units.

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
