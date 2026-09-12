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

Preparation is create-only and producer-free:
`python scripts/prepare_rgb_scene_candidate_launch.py --bundle BUNDLE
--candidate-context CANDIDATE/candidate_contract.json --output PLAN --gpu N
--device cuda:0 --python /absolute/pinned/python`. It derives all 12 stage
scripts and inspection recipes directly from the current bundle/CLI and
candidate contract. It accepts arbitrary identifiers, frame indices, dimensions,
device and paths containing spaces, rejects collisions before writing, validates
bash syntax, and never chmods an output mount or rewrites old paths. The helper
is intentionally not a VLM planner and does not author visual judgments.

The external baseline depends on preflighted pinned Python/model environments,
MANO and HaWoR assets, LightGlue, SAM3D, and the native ShellGate provider.
There is no one-command pip install and no unattended VLM planner. Resume and
GPU sharing remain current limitations, not solved capabilities.

## Quality and scope limits

The candidate events all returned rc0 and the final JSON/RRD hashes are recorded
in the developer-only publication record. This is not a quality pass:
`annotation_ready=false`, `demo_only=true`, and contact, collision, sign/SDF,
and nonpenetration were not measured. No MP4 is shipped. Numerical comparison
is post-freeze evaluation only; GT is never a prediction input. Birth bundle
identity and execution bundle identity are preserved distinctly.

Design note: the helper materializes literal `/bin/bash` scripts from parsed
stage arguments, rather than copying historical v2–v6 templates, because
historical template substitution can silently retain stale bundle, candidate,
GPU, frame, or inspection paths. Complete input contracts are validated before
create-only output creation; any existing output is rejected.
