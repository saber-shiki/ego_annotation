# Standalone RGB scene v7 baseline

This branch publishes the frozen numerical research v7 source as an independent
baseline. The publication commit changes packaging/documentation and the
parent-side launch preparation helper only; it does not change inference,
geometry, pose-graph, MANO, or export math. The frozen source is
`fa9a84cbfb6836b92e35a6316073a6855e6b4d22`; the executed bundle is
`/mnt/user-home/kupingxin/ego_annotation_runtime/rgb_scene_chain_contact_gpu4_20260911_v4`
and the candidate is
`.../candidates/rgb_chain_contact_repair_v7` in the external integration output.

## Reproduction contract

A new raw video starts with `prepare_rgb_scene_runtime.py bundle` from a clean
reviewed source, then `prepare-run`, and uses the pinned native
`rgb-scene-runtime` provider. The current validated v7 path instead reuses the
verified P01–P08 source prefix and native frame-0 mesh receipt through the
candidate contract (`native_mode: reuse`); it still executes candidate-local
P09/P11, canonical geometry/pose graph, appearance/shape, MANO visual fit and
export. Both paths are full timeline, full mesh and complete time-indexed MANO;
selection is ranked-first, with explicit visual confirmation. Raw appearance,
depth-owned masks and metric eligibility remain separate. P17 uses exact
integer `frame_idx`; optional real-evidence primary pads use one shared
transform. Only the final annotation/world pair is published.

Preparation is create-only and producer-free (from the reviewed source checkout):
```bash
PREP_PYTHON=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
"$PREP_PYTHON" scripts/prepare_rgb_scene_candidate_launch.py \
  --bundle "$BUNDLE" --candidate-context "$CANDIDATE/candidate_contract.json" \
  --output "$NEW_PLAN" --gpu "$GPU" --python "$PREP_PYTHON"
```
It derives all 12 stage scripts and three bounded inspection recipes directly
from the current bundle/candidate contracts, without previous candidate templates.
The physical GPU must match the bundle configuration; the internal logical
`--device` is fixed to `cuda:0`. `--image-width` / `--image-height` are optional
**metadata only**, not stage or image-resolution overrides. Valid identifiers,
selected frame indices, and paths containing spaces are supported. The helper
checks source contract/context/event-snapshot bindings before writing, validates
all shell bodies, rejects existing output directories, and never chmods an
output mount. Runtime still validates model/assets/producer lineage at execution;
this is not a replacement for full preflight or a VLM planner.

`launch_manifest.json` includes exact startup, judgment and final paths plus
quoted `/bin/bash` commands. Run startup/proposal/P09 inspection commands only
after the corresponding inputs exist. Proposal summaries show selected scalar
ranking fields; P09 summaries hash both the adapter report and the complete
annotation file using streaming SHA256. Output is bounded to 12,001 bytes
including the final newline. These recipes **do not inspect pixels or author
judgments**. The native runtime must inspect the actual current RGB/mask evidence
and author P09, P10, appearance and full-timeline P17 judgments.

See [raw-video build/prepare-run instructions](STANDALONE_RGB_SCENE_README.md)
and [prepare-candidate/import-native commands](STANDALONE_RGB_SCENE_CANDIDATE_WORKFLOW.md).
These are distinct paths: a fresh video requires its own perception/model outputs;
validated prefix/native reuse requires exact current-video receipts, not another
video's cache or a copied old mesh. Execute producers only via the isolated native
project agent and its pinned ShellGate provider, not by running these scripts as
a development-agent prediction substitute.

The external baseline depends on preflighted pinned Python/model environments,
MANO and HaWoR assets, LightGlue, SAM3D, and the native ShellGate provider.
There is no one-command pip install and no unattended VLM planner. Resume and
GPU sharing remain current limitations, not solved capabilities.

## Quality and scope limits

The candidate events all returned rc0 and the final JSON/RRD hashes are recorded
in the [developer-only publication record](STANDALONE_RGB_SCENE_V7_RECORD.md).
This is not a quality pass:
`annotation_ready=false`, `demo_only=true`, and contact, collision, sign/SDF,
and nonpenetration were not measured. No MP4 is shipped. Numerical comparison
is post-freeze evaluation only; GT is never a prediction input. Birth bundle
identity and execution bundle identity are preserved distinctly.

Design note: the helper materializes literal `/bin/bash` scripts from parsed
stage arguments, rather than copying historical v2–v6 templates, because
historical template substitution can silently retain stale bundle, candidate,
GPU, frame, or inspection paths. Source bindings, configuration arguments and
all generated contents are validated before create-only output creation; any
existing output is rejected. No inference/math or frozen runtime changes are
part of this helper. Runtime/model asset verification remains with runtime
preflight. This is publication support, not improved physical annotation.
