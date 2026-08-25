# HOT3D five-clip controlled SAM3D / TRELLIS runtime spec

This document is authoritative for the controlled dual-backend run.  Execute one
case per fresh run root.  The geometry backend is the only branch variable.

## Bound launch values

The launch prompt binds all of these values explicitly:

- `{INPUT_VIDEO}`: the exact 150-frame, 1408x1408, 30 FPS pinhole RGB clip.
- `{RUN_ROOT}`: fresh prediction output root.
- `{CASE_ID}`: stable clip id.
- `{OBJECT_ID}` and `{TRACK_ID}`: stable target object id.
- `{GPU_ID}`: dedicated physical A800 id for this case.
- `{SENSOR_CALIBRATION_METADATA}`: prediction-side official pinhole camera contract.
- `{TARGET_HINT}` and `{TARGET_EXCLUSIONS}`: semantic hints only, never masks or poses.
- optional `{ANCHOR_GUIDANCE}`: a previously diagnosed frame to scrutinize, never an artifact or permission to skip fresh P09 review.

The launch also binds:

- `{SENSOR_SOURCE_VIDEO}={INPUT_VIDEO}`
- `{SENSOR_CALIBRATION_AUTHORITY}=prediction_side_sensor_metadata`
- `{SENSOR_FRAME_INTRINSICS_KEY}` is empty because the supplied contract has one top-level K.
- general Python: `/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`
- SAM3D Python: resolved from `/mnt/user-home/kupingxin/sam3d-objects/activate.sh` as `$CONDA_PREFIX/bin/python`.
- SAM3D repository: `/mnt/user-home/kupingxin/sam3d-objects/src`.
- SAM3D config: `/mnt/nas-222-project/kupingxin/sam3d-objects/checkpoints/modelscope/pipeline.yaml`.
- Launch preflight must source the activation script and prove `from inference import Inference` with the declared repository before any case is launched.
- TRELLIS Python/repository/model and all common assets are those declared by `runtime/v19_runtime_spec.md`.
- DA3 Python/repository/model are launcher-preflighted local assets bound as `{DA3_PYTHON}`, `{DA3_REPO}`, and `{DA3_MODEL_PATH}`. The controlled checkpoint is `depth-anything/DA3NESTED-GIANT-LARGE-1.1` (CC BY-NC 4.0; non-commercial research use only).

## Isolation and fairness rules

1. Do not inspect or consume reference-label state directories, CAD models, reference poses,
   foreground reference depth, MANO reference state, or any sibling run output.
2. Official K is allowed only through the launcher-supplied prediction-side sensor contract.
3. Execute common phases from `runtime/v19_runtime_spec.md`, but apply the DA3 intervention order:
   P00, P01, P02, **P03b, P04, P03d DA3, P03c**, then P05 through P11. P03 UniDepth is an A/B
   baseline only and is not the active depth source in this branch. Do not execute the canonical
   P12 through P21; replace that tail with this document.
4. P05 must inspect the raw contact sheet as an image.  P07 must inspect OWLv2/SAM2
   review imagery.  P09 must inspect the anchor-candidate review image and write the
   explicit anchor decision. When the proposal report exposes supported
   `conditioning_coherence_preferred` rows, select among those one-component owned-mask
   candidates unless image inspection finds wrong ownership or inadequate target identity;
   disconnected rows remain valid metric/appearance evidence but are ambiguous native
   single-image completion anchors. The shared P09 contract is further constrained here: an
   object-owned appearance mask may survive depth rejection only after full projected-MANO
   triangle-silhouette subtraction; rejected depth remains ineligible, and per-frame extent
   eligibility must use the orientation-invariant visible-population reference rather than
   one anchor's camera-axis AABB. The launch target hint must be visually confirmed.
5. Use one shared P11 evidence report, anchor RGB, object-owned mask, DA3 depth, observed metric
   surface, camera/HaWoR state, and observed-only object trajectory for both geometry branches.
6. SAM3D receives full RGB plus the binary object-owned mask and no external pointmap. DA3 is an
   external depth measurement only; SAM3D internal MoGe, config, runner, and `pointmap=None` remain frozen.
   TRELLIS receives its native P11 object-isolated RGBA crop.  These native conditioning
   formats are intentional; neither backend may receive another hidden source.
7. Generated faces are render-only.  They are never collision, sign, contact, or pose
   observations.  Preserve uncertainty in the final state.
8. Run commands from the isolated bundle.  Do not edit scripts during a run.  If a named
   command fails, write `{RUN_ROOT}/state/runtime_blockers/<PHASE>.json` and stop.
9. Keep `{RUN_ROOT}/logs/harness_events.jsonl` append-only.  After every completed phase,
   append a timestamped event and update `{RUN_ROOT}/state/suite_case_progress.json`.
10. Do not run scoring or backend ranking in the prediction process.

## Common P00-P11

Read `runtime/v19_runtime_spec.md`, bind the launch values above, and execute only P00,
P01, P02, **P03b, P04, P03d DA3, P03c**, P05, P06, P07, P08, P09, P10, and P11 in that
order. P03d must pass official K and the P04 HaWoR metric W2C trajectory to DA3. It writes
camera-z metric depth on exact source rays; metadata-only K relabel and any released/reference
camera pose are forbidden. P03 UniDepth is not the active depth source and must not feed P09 in
this branch.
Use the dedicated `{GPU_ID}` unless a live probe shows it is no longer safe; do not take
another case's declared GPU.  The target should remain rigid even when local evidence is
missing; record missing evidence as uncertainty rather than broadening the object mask.

After P04 and before P05, run the exact P03d intervention. All DA3 source/checkpoint paths are
launcher/preflight bindings; runtime discovery or download is forbidden:

```bash
CUDA_VISIBLE_DEVICES='{GPU_ID}' '{DA3_PYTHON}' scripts/run_da3_pose_conditioned_full_frame_v1.py \
  --manifest '{RUN_ROOT}/input/raw_frame_manifest/manifest.json' \
  --camera-contract '{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json' \
  --hawor-npz '{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz' \
  --da3-repo '{DA3_REPO}' \
  --model-path '{DA3_MODEL_PATH}' \
  --model-id depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --output-dir '{RUN_ROOT}/measurements/depth_slam/da3_pose_conditioned' \
  --camera-input-plane manifest_rgb \
  --camera-output-plane source_rgb \
  --frame-start 0 \
  --frame-end {FRAME_END} \
  --source-width {SOURCE_WIDTH} \
  --source-height {SOURCE_HEIGHT} \
  --window-size 16 \
  --window-overlap 4 \
  --ref-view-strategy middle
```

Then run P03c before P05 with the provider-specific output name:

```bash
MAIN_PYTHON=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
DA3_RAW_DEPTH='{RUN_ROOT}/measurements/depth_slam/da3_pose_conditioned/da3_pose_conditioned_full_frame_depth_v1.npz'
DEPTH_NPZ='{RUN_ROOT}/state/calibration/depth_camera_contract/da3_pose_conditioned_depth_camera_contract_v2.npz'
"$MAIN_PYTHON" scripts/adapt_v19_depth_to_camera_contract.py \
  --source-depth-npz "$DA3_RAW_DEPTH" \
  --camera-contract '{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json' \
  --depth-plane source_rgb \
  --output-dir '{RUN_ROOT}/state/calibration/depth_camera_contract' \
  --output-name da3_pose_conditioned_depth_camera_contract_v2.npz
```

In every common P08/P09/P14b/P18 command that names the canonical UniDepth camera-bound archive,
substitute exactly `$DEPTH_NPZ`; do not rename the DA3 file to a UniDepth filename. P09's
first-surface logic consumes the archive-declared provider confidence semantics. P03d converts
DA3 raw higher-is-better confidence to a monotonic higher-is-worse error proxy; it is not calibrated
metric uncertainty.

After P11, bind and validate:

```bash
set -euo pipefail
MAIN_PYTHON=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
EXP_ROOT='{RUN_ROOT}/experiments/sam3d_trellis_controlled'
EVIDENCE_REPORT='{RUN_ROOT}/measurements/geometry_completion/rigid_evidence/{CASE_ID}/{OBJECT_ID}/evidence_bundle/evidence_bundle_report.json'
ANNOTATIONS='{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json'
HAWOR_NPZ='{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz'
DA3_RAW_DEPTH='{RUN_ROOT}/measurements/depth_slam/da3_pose_conditioned/da3_pose_conditioned_full_frame_depth_v1.npz'
DEPTH_NPZ='{RUN_ROOT}/state/calibration/depth_camera_contract/da3_pose_conditioned_depth_camera_contract_v2.npz'
test -s "$EVIDENCE_REPORT"
test -s "$ANNOTATIONS"
test -s "$HAWOR_NPZ"
test -s "$DA3_RAW_DEPTH"
test -s "$DEPTH_NPZ"
"$MAIN_PYTHON" - "$EVIDENCE_REPORT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
b = d.get('selected_anchor_atomic_binding')
f = int(d.get('selected_frame_idx', -1))
if not isinstance(b, dict) or b.get('validated') is not True:
    raise SystemExit('P11 selected anchor is not atomically bound')
if any(int(b.get(k, -2)) != f for k in ('selected_frame_idx', 'visible_geometry_frame_idx', 'canonical_surface_frame_idx')):
    raise SystemExit(f'P11 selected anchor frame mismatch: selected={f} binding={b}')
PY
mkdir -p "$EXP_ROOT"
```

## D11 native dual-conditioning contract

Materialize byte-bound native inputs from the one selected P11 evidence row:

```bash
"$MAIN_PYTHON" experiments/sam3d_p11_p12_branch/build_p11_dual_geometry_inputs.py \
  --evidence-report "$EVIDENCE_REPORT" \
  --output-dir "$EXP_ROOT/P11_dual_inputs"

P11_DUAL_REPORT="$EXP_ROOT/P11_dual_inputs/p11_dual_geometry_inputs_report.json"
test -s "$P11_DUAL_REPORT"
```

Inspect `P11_dual_inputs/review/sam3d_native_input_mask_review.png` as an image.  Stop if
it contains broad hand, sleeve, table, or unrelated-object ownership.

## D12 TRELLIS and SAM3D raw geometry priors

Resolve and run TRELLIS with the native isolated crop:

```bash
EVIDENCE_CROP_RGBA=$("$MAIN_PYTHON" scripts/resolve_v19_trellis_conditioning_image.py \
  --evidence-report "$EVIDENCE_REPORT")
test -s "$EVIDENCE_CROP_RGBA"

CUDA_VISIBLE_DEVICES='{GPU_ID}' \
TORCH_HOME=/mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub \
ATTN_BACKEND=xformers SPCONV_ALGO=native \
/mnt/user-home/kupingxin/ego_annotation/.runtime/trellis_work/.venv_trellis/bin/python \
  scripts/remote_run_trellis_shape_v3.py \
  --repo /mnt/user-home/kupingxin/ego_annotation/.runtime/trellis_work/TRELLIS \
  --model /mnt/truenas-user-home/kupingxin/ego_annotation_models/trellis-image-large-25e0d31f \
  --dinov2-repo /mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub/hub/facebookresearch_dinov2_main \
  --dinov2-checkpoint /mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth \
  --image "$EVIDENCE_CROP_RGBA" \
  --output-dir "$EXP_ROOT/P12_trellis" \
  --seed 42

TRELLIS_REPORT="$EXP_ROOT/P12_trellis/qc_trellis_shape_v3.json"
test -s "$TRELLIS_REPORT"
```

Run the frozen SAM3D Objects runner through the native P11 full-RGB + owned-mask
contract. SAM3D checkpoint loading is fail-closed offline: MoGe
`Ruicheng/moge-vitl/model.pt` and the DINOv2 source/checkpoint must be local,
hash-bound, and reported with `network_resolution_allowed:false`; any undeclared Hugging
Face resolution is a D12 blocker. `pointmap=None` is enforced by the D11/D12 report adapter:

```bash
BUNDLE_ROOT=$(pwd)
set +u
source /mnt/user-home/kupingxin/sam3d-objects/activate.sh
set -u
SAM3D_PYTHON="$CONDA_PREFIX/bin/python"
cd "$BUNDLE_ROOT"

"$MAIN_PYTHON" experiments/sam3d_p11_p12_branch/run_p12_parallel_geometry_priors.py \
  --p11-report "$P11_DUAL_REPORT" \
  --trellis-report "$TRELLIS_REPORT" \
  --output-dir "$EXP_ROOT/P12_parallel" \
  --case-name '{CASE_ID}_{OBJECT_ID}_anchor' \
  --seed 42 \
  --sam3d-python "$SAM3D_PYTHON" \
  --sam3d-runner scripts/remote_run_sam3d_objects_mesh_v7.py \
  --sam3d-repo /mnt/user-home/kupingxin/sam3d-objects/src \
  --sam3d-config /mnt/nas-222-project/kupingxin/sam3d-objects/checkpoints/modelscope/pipeline.yaml \
  --sam3d-moge-checkpoint /mnt/user-home/kupingxin/sam3d-objects/hf-cache/hub/models--Ruicheng--moge-vitl/blobs/da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f \
  --sam3d-dinov2-repo /mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub/hub/facebookresearch_dinov2_main \
  --sam3d-dinov2-checkpoint /mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth \
  --cuda-visible-device '{GPU_ID}' \
  --min-free-mib 30000

P12_REPORT="$EXP_ROOT/P12_parallel/p12_parallel_geometry_priors_report.json"
test -s "$P12_REPORT"
```

Bind exact raw mesh paths from reports, never by globbing:

```bash
eval "$("$MAIN_PYTHON" - "$P12_REPORT" "$TRELLIS_REPORT" <<'PY'
import json, shlex, sys
from pathlib import Path
p12 = json.loads(Path(sys.argv[1]).read_text())
trellis = json.loads(Path(sys.argv[2]).read_text())
sam = (((p12.get('candidates') or {}).get('sam3d_objects') or {}).get('native_outputs') or {}).get('raw_mesh') or {}
values = {
    'SAM3D_RAW_MESH': sam.get('path'),
    'TRELLIS_RAW_MESH': trellis.get('mesh'),
}
for key, value in values.items():
    if not value or not Path(value).is_file() or Path(value).stat().st_size <= 0:
        raise SystemExit(f'missing {key}: {value!r}')
    print(f'{key}={shlex.quote(str(value))}')
PY
)"
```

## D13 controlled common adaptation

Run both raw priors through the same unmodified metric alignment/completion adapter and
one shared P11 evidence report:

```bash
"$MAIN_PYTHON" experiments/sam3d_p11_p12_branch/run_p13_controlled_geometry_prior_ab.py \
  --evidence-report "$EVIDENCE_REPORT" \
  --builder-script scripts/build_v18_compact_rigid_trellis_completion.py \
  --sam3d-bridge-script experiments/sam3d_p11_p12_branch/build_p13_sam3d_native_metric_bridge.py \
  --python "$MAIN_PYTHON" \
  --candidate "sam3d_new_object_owned_mask|sam3d_objects|$SAM3D_RAW_MESH|$P12_REPORT" \
  --candidate "trellis_frozen|trellis|$TRELLIS_RAW_MESH|$TRELLIS_REPORT" \
  --output-dir "$EXP_ROOT/P13_controlled" \
  --silhouette-dilate-px 16 \
  --planar-slab-eigenvalue-ratio-max 0.04 \
  --planar-slab-min-band-m 0.018 \
  --planar-slab-max-band-m 0.055

CONTROLLED_REPORT="$EXP_ROOT/P13_controlled/p13_controlled_geometry_prior_ab_report.json"
test -s "$CONTROLLED_REPORT"
```

For SAM3D, D13 must preserve the native quaternion/local-to-camera pose, bridge the whole
camera-origin scene similarity to robust sensor depth, and then use identity canonical
alignment. P13 must fail closed unless selected RGB, owned mask, camera, metric surfels,
centroid, and observed canonical surface are atomically bound to the same P11 frame. Native
projection overlap is necessary but not sufficient: the generated render prior must also cover
the same frame's measured front surface. The median/extent limit remains strict. P95/extent
uses a strict tier at `0.15`; a bounded conditional tail tier through `0.18` is permitted only
when the strict median still passes and native convex projection IoU is at least `0.25`. Such a
row must be labeled `conditional_p95_tail_uncertain_native_projection_supported` in the bridge
report and carried as render-quality uncertainty; it does not relax projection, pose, contact,
collision, or signed-geometry eligibility. This quality check remains diagnostic/render
eligibility only and never promotes generated faces to pose, contact, or collision evidence. P13 must
not fall back to TRELLIS RMS/PCA permutations/ICP. Preserve the resulting
metric-canonical SAM3D topology as a separate render underlay while keeping the observed
metric surface physically authoritative:

```bash
"$MAIN_PYTHON" experiments/sam3d_p11_p12_branch/build_p13_dual_mesh_geometry_prior.py \
  --controlled-report "$CONTROLLED_REPORT" \
  --candidate sam3d_new_object_owned_mask \
  --output-dir "$EXP_ROOT/P13_sam3d_dual"

DUAL_REPORT="$EXP_ROOT/P13_sam3d_dual/p13_dual_mesh_geometry_prior_report.json"
DUAL_STATE="$EXP_ROOT/P13_sam3d_dual/dual_mesh_render_state.json"
OBSERVED_MESH="$EXP_ROOT/P13_sam3d_dual/collision_eligible_observed_surface.ply"
test -s "$DUAL_REPORT"
test -s "$DUAL_STATE"
test -s "$OBSERVED_MESH"
```

## D14-D16 shared observed-only object trajectory and unsigned MANO state

Write the common observed-only pose body contract:

```bash
OBSERVED_COMPLETION="$EXP_ROOT/P14_observed_completion/observed_only_completion_report.json"
"$MAIN_PYTHON" scripts/build_v19_observed_only_completion_reference.py \
  --case '{CASE_ID}' \
  --object-id '{OBJECT_ID}' \
  --controlled-report "$CONTROLLED_REPORT" \
  --candidate sam3d_new_object_owned_mask \
  --observed-mesh "$OBSERVED_MESH" \
  --output "$OBSERVED_COMPLETION"
```

Fit direct poses from observed evidence and complete one shared full timeline. Rotation must
come from adjacent accepted metric-surface registration plus the selected anchor's observed
surfels, never from generated completion faces. Translation uses the same observed anchor
surface and the fixed temporal prior. A long strict-depth gap may be bridged only by the
script's fail-closed projected-MANO-subtracted RGB optical-flow + exact-camera PnP path; this
does not reinstate rejected depth. Each bridge records its own rotation observability, but a
line-like bridge may remain explicitly underobservable only when the unchanged downstream
score/fraction timeline gate still passes. Do not lower the default support, gap, reprojection,
observability-fraction thresholds and do not include explicitly ineligible metric rows. The
strict rotation-step tier remains `15°`. User-authorized recovery permits only the named sparse
conditional tail below: at most two adjacent transitions (`<=1.5%` of timeline steps), each at
most `18°` with translation step `<=0.020 m`, both endpoints direct eligible
`adjacent_observed_metric_surfel_registration` rows, no generated pose evidence, and endpoint
rotation-observability scores `<=0.030`. The solver must preserve the original matrices without
clipping and record `conditional_sparse_underobservable_rotation_tail` on the report and target
rows. Any transition outside those conditions remains a D15 blocker:

```bash
"$MAIN_PYTHON" scripts/fit_v18_compact_rigid_object_pose.py \
  --annotations "$ANNOTATIONS" \
  --completion-report "$OBSERVED_COMPLETION" \
  --object-id '{OBJECT_ID}' \
  --output-dir "$EXP_ROOT/P14_observed_pose_fit"

POSE_FIT="$EXP_ROOT/P14_observed_pose_fit/v18_compact_rigid_object_pose_fit_report.json"
"$MAIN_PYTHON" scripts/solve_v19_rigid_object_pose_graph.py \
  --annotations "$ANNOTATIONS" \
  --pose-report "$POSE_FIT" \
  --completion-report "$OBSERVED_COMPLETION" \
  --object-id '{OBJECT_ID}' \
  --complete-full-timeline-rigid-pose \
  --allow-sparse-conditional-rotation-tail \
  --conditional-max-rotation-step-deg 18 \
  --conditional-max-rotation-step-count 2 \
  --conditional-max-rotation-step-fraction 0.015 \
  --conditional-max-rotation-step-translation-m 0.020 \
  --conditional-max-endpoint-rotation-observability-score 0.030 \
  --output-dir "$EXP_ROOT/P15_observed_pose_graph"

POSE_GRAPH="$EXP_ROOT/P15_observed_pose_graph/v19_rigid_object_pose_graph_report.json"
test -s "$POSE_GRAPH"
```

Read both pose reports. P14 must state `generated_faces_pose_eligible:false`, identify every
direct metric-surface and RGB-PnP row, and prove that generated geometry was diagnostic-only.
The layered-state adapter requires `annotation_ready:true`, `graph_support.sufficient:true`,
`temporal_readiness.ready:true`, exactly 150 accepted pose rows, and zero
`nonpenetration_target_frame_count`. The temporal gate enforces direct-pose fraction, bounded
direct/interpolation/hold gaps, rotation observability, and bounded per-frame SE(3) jumps. If
any gate fails, preserve the report, write a D15 blocker, and stop; do not relabel interpolation,
nearest holds, optimizer success, or a rejected depth row as evidence. If the sparse conditional
rotation tier is applied, record its exact transitions in `state/v19_agent_evidence.md`, inspect
those transition frames during D18, and retain them as low-confidence trajectory uncertainty in
the final publication; do not call the estimated step a ground-truth angular velocity.

Build an unsigned observed-surface MANO/object measurement state:

```bash
"$MAIN_PYTHON" scripts/build_v18_mano_object_constraint_state.py \
  --annotations "$ANNOTATIONS" \
  --hawor-npz "$HAWOR_NPZ" \
  --pose-report "$POSE_GRAPH" \
  --completion-report "$OBSERVED_COMPLETION" \
  --skip-signed-distance \
  --output-dir "$EXP_ROOT/P16_unsigned_mano_object" \
  --object-id '{OBJECT_ID}'

CONSTRAINT_REPORT="$EXP_ROOT/P16_unsigned_mano_object/v18_mano_object_constraint_state.json"
test -s "$CONSTRAINT_REPORT"
```

## D17 branch render states

Build the shared source state without applying geometry-dependent hand correction, then
clone only the geometry layers for SAM3D and TRELLIS:

```bash
SOURCE_STATE="$EXP_ROOT/P17_source_state/observed_only_rigid_render_state.json"
"$MAIN_PYTHON" scripts/build_v19_rigid_render_state.py \
  --case '{CASE_ID}' \
  --object-id '{OBJECT_ID}' \
  --object-label '{OBJECT_ID} shared observed pose' \
  --annotations "$ANNOTATIONS" \
  --pose-report "$POSE_GRAPH" \
  --completion-report "$OBSERVED_COMPLETION" \
  --completed-mesh "$OBSERVED_MESH" \
  --constraint-report "$CONSTRAINT_REPORT" \
  --output "$SOURCE_STATE"

"$MAIN_PYTHON" experiments/sam3d_p11_p12_branch/build_p14_p15_layered_render_states.py \
  --source-render-state "$SOURCE_STATE" \
  --dual-mesh-state "$DUAL_STATE" \
  --dual-mesh-report "$DUAL_REPORT" \
  --controlled-report "$CONTROLLED_REPORT" \
  --sam-candidate sam3d_new_object_owned_mask \
  --trellis-candidate trellis_frozen \
  --output-dir "$EXP_ROOT/P15_layered_states"

STATE_ADAPTER_REPORT="$EXP_ROOT/P15_layered_states/p14_p15_layered_render_state_adapter_report.json"
SAM3D_STATE="$EXP_ROOT/P15_layered_states/sam3d_owned_dual_mesh/experimental_layered_render_state.json"
TRELLIS_STATE="$EXP_ROOT/P15_layered_states/trellis_frozen_legacy_cut/experimental_layered_render_state.json"
test -s "$STATE_ADAPTER_REPORT"
test -s "$SAM3D_STATE"
test -s "$TRELLIS_STATE"
```

## D18 complete-duration final renders

Read `{RUN_ROOT}/state/anchor_decisions/{OBJECT_ID}.json` to bind the exact
`{ANCHOR_FRAME}`.  Run the single bundled D18 wrapper exactly as written. It invokes the
same bundled layered renderer for both states, validates four `150-frame / 30 FPS` videos
per backend, and writes one dual-render report. Do not invent an alternative renderer name
or manually reconstruct this command:

```bash
"$MAIN_PYTHON" scripts/run_hot3d_dual_backend_d18_renders.py \
  --run-root '{RUN_ROOT}' \
  --anchor-frame '{ANCHOR_FRAME}' \
  --expected-frame-count 150 \
  --expected-fps 30 \
  --generated-face-budget 12000 \
  --replace

D18_RENDER_REPORT="$EXP_ROOT/D18_dual_backend_render_report.json"
SAM3D_RENDER_MANIFEST="$EXP_ROOT/renders/sam3d/p14_p15_layered_full_mano_render_manifest.json"
TRELLIS_RENDER_MANIFEST="$EXP_ROOT/renders/trellis/p14_p15_layered_full_mano_render_manifest.json"
test -s "$D18_RENDER_REPORT"
test -s "$SAM3D_RENDER_MANIFEST"
test -s "$TRELLIS_RENDER_MANIFEST"
```

Inspect each anchor review PNG as an image.  Confirm that the object follows the same
observed trajectory in both branches, the generated geometry appears as an underlay,
the green observed surface owns measured support, and full MANO surfaces remain visibly
labeled.  Record any drift, handedness concern, or unresolved contact in
`{RUN_ROOT}/state/v19_agent_evidence.md`; do not change state to hide it.

## D19 final result publication

This finalizer validates all four 150-frame, 30 FPS videos per backend, publishes stable
names, links meshes/state/reports, and writes `SUITE_DONE.json` last:

```bash
"$MAIN_PYTHON" scripts/finalize_hot3d_dual_backend_case.py \
  --case '{CASE_ID}' \
  --object-id '{OBJECT_ID}' \
  --run-root '{RUN_ROOT}' \
  --sam3d-render-manifest "$SAM3D_RENDER_MANIFEST" \
  --trellis-render-manifest "$TRELLIS_RENDER_MANIFEST" \
  --controlled-report "$CONTROLLED_REPORT" \
  --dual-report "$DUAL_REPORT" \
  --state-adapter-report "$STATE_ADAPTER_REPORT" \
  --output-dir '{RUN_ROOT}/final_results' \
  --expected-frame-count 150 \
  --expected-fps 30

test -s '{RUN_ROOT}/SUITE_DONE.json'
test -s '{RUN_ROOT}/final_results/case_result_manifest.json'
```

Final stable outputs are:

- `{RUN_ROOT}/final_results/sam3d/videos/{camera_overlay,world_view,side_world_view,side_by_side}.mp4`
- `{RUN_ROOT}/final_results/trellis/videos/{camera_overlay,world_view,side_world_view,side_by_side}.mp4`
- per-backend `geometry/`, `state/`, `reports/`, and `backend_result.json`
- `{RUN_ROOT}/final_results/case_result_manifest.json`

Append one final harness event and a concise uncertainty statement.  Do not run any
additional backend, rerank geometry, or consume reference-label sidecars after D19.
