# Standalone RGB candidate continuation

This workflow adds a create-only suffix below an existing prepared run. It is
clip-agnostic: object IDs, frame IDs, dimensions, timestamps, model identities,
and source hashes are read from the run rather than embedded in the command.
The source P01--P08 outputs and original selection remain untouched.

## Prepare and select

After the source prefix and current `anchor_candidate_proposals.json` have been
inspected, create a fresh candidate (use a new ID for every failed or partial
attempt):

```bash
PYTHON=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
$PYTHON scripts/prepare_rgb_scene_runtime.py prepare-candidate \
  --run-root "$RUN" --candidate-id "$CANDIDATE" --native-mode new
# or explicitly override the ranking-first frame after inspecting RGB/mask:
$PYTHON scripts/prepare_rgb_scene_runtime.py prepare-candidate \
  --run-root "$RUN" --candidate-id "$CANDIDATE" --frame "$FRAME" --native-mode new
```

The default is `ranked_candidates[0]`; an explicit frame must be in the same
current proposal list and is only a selection request, not visual confirmation.
The runtime author must write candidate-local `state/anchor_decisions/`,
`state/physical_branch_decisions/`, `state/object_selection.json`, and later
`state/appearance_decisions/` / interaction judgment files with current-run
visual provenance. Their P09 and P11 anchor frame must equal the candidate
frame. Candidate events are append-only at
`candidates/$CANDIDATE/logs/events.jsonl`.

## Stages

Run the candidate stages through the single command boundary, pausing for each
required inspection/decision rather than attempting to automate human judgment:

```bash
CTX="$RUN/candidates/$CANDIDATE/candidate_contract.json"
# 1. Generate candidate-owned P09 depth-order evidence, then inspect ownership
$PYTHON scripts/run_rgb_scene_stage.py --stage frontend --phase P09_DEPTH_ORDER \
  --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
# 2. Generate ranking proposals; inspect ranked_candidates[0] RGB, SAM2 mask and metric support.
$PYTHON scripts/run_rgb_scene_stage.py --stage frontend --phase P09-propose \
  --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
# Author candidate-local state/anchor_decisions/<object>.json with provenance.
# 3. Materialize same-frame P09 geometry, inspect it, and author candidate-local P10
# physical_branch_decisions/<object>.json and object_selection.json.
$PYTHON scripts/run_rgb_scene_stage.py --stage frontend --phase P09 \
  --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
$PYTHON scripts/run_rgb_scene_stage.py --stage frontend --phase P11 \
  --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
# 4. Continue the candidate suffix. Inspect appearance output and author its
# appearance decision; inspect trajectory/visibility and author P17 judgment before visual-fit.
$PYTHON scripts/run_rgb_scene_stage.py --stage geometry --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
$PYTHON scripts/run_rgb_scene_stage.py --stage initial-pose --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
$PYTHON scripts/run_rgb_scene_stage.py --stage refined-pose --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
$PYTHON scripts/run_rgb_scene_stage.py --stage appearance --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
# Author/inspect state/appearance_decisions/<object>.json (same candidate frame).
$PYTHON scripts/run_rgb_scene_stage.py --stage native-shape --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
$PYTHON scripts/run_rgb_scene_stage.py --stage shape --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
# Author/inspect complete state/agent_interaction_judgments/<object>.json now.
$PYTHON scripts/run_rgb_scene_stage.py --stage visual-fit --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
$PYTHON scripts/run_rgb_scene_stage.py --stage export --run-root "$RUN" --runtime-config "$BUNDLE/runtime/runtime_config.json" --gpu "$GPU" --candidate-context "$CTX"
```

A failed stage leaves its command and blocker; it is not retried or overwritten
under the same candidate ID. A candidate with `native-mode new` invokes the
pinned native SAM3D producer once. All later pose, global RGB SO(3)/full SE(3),
appearance/shape binding, MANO fitting, and export stages consume that
candidate's bound mesh and refined posterior.

## Verified native reuse

Reuse is a materialization operation, not a PLY swap. Prepare a separate reuse
candidate and provide the complete successful native QC plus model execution
receipt, with the exact RGB/mask bytes used by that receipt:

```bash
$PYTHON scripts/prepare_rgb_scene_runtime.py prepare-candidate \
  --run-root "$RUN" --candidate-id "$CANDIDATE_REUSE" --frame "$FRAME" --native-mode reuse \
  --source-native /absolute/source/native/output
$PYTHON scripts/prepare_rgb_scene_runtime.py import-native \
  --candidate-context "$RUN/candidates/$CANDIDATE_REUSE/candidate_contract.json" \
  --source-qc /absolute/control/native_reconstruction/<case>/qc_sam3d_objects_mesh_v7.json \
  --execution-receipt /absolute/control/execution_receipt.json \
  --control-contract /absolute/control/control_contract.json \
  --image /absolute/source/rgb/frame.jpg --mask /absolute/source/rgb/mask.png \
  --model-identity '{"config_sha256":"...","seed":42}'
```

The importer requires `status: ok`, a zero-return execution receipt, complete
raw/OpenCV-camera/PyTorch3D-camera mesh outputs and the exact image/mask paths.
It copies bytes into the candidate, writes a derived QC mapping and preserves
any wrapper `failure.json`; it never restamps the source receipt. Any mutation
of the source mesh, RGB, mask, frame, or configuration must invalidate reuse.
The normal `native-shape` stage then consumes the imported QC path and the same
candidate frame.

Final `candidates/$CANDIDATE/final/annotation.json` and `world.rrd` are the only
published pair. They embed source birth identity plus candidate/native reuse
identity and retain complete geometry, full timestamps, visibility, camera and
MANO surfaces. The source run's `final/` and all source anchors remain
unchanged. `prepare_rgb_scene_runtime.py bundle` is still required to create a
fresh curated runtime bundle before prediction; it does not execute inference.
