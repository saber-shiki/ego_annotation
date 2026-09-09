# Generic additive demo visual v2

This branch is an isolated, demo-only visual hypothesis. It does not modify or
promote P14/P15, completion, annotations, P17, MANO, or final annotation state.
All generated files are confined to:

- `measurements/demo_visual/<object>/<demo-id>/`
- `state/demo_visual/<object>/<demo-id>/`
- `renders/demo_visual/<object>/<demo-id>/`
- `logs/demo_visual/<object>/<demo-id>/`

## Mechanisms

1. `build_v19_demo_visual_inputs.py` validates exact raw/annotation/SAM2 frame
   coverage, copies every available raw appearance mask byte-for-byte, and
   chooses an appearance anchor using only normalized area, border clearance,
   and crop sharpness. The selected frame must independently contain same-frame
   camera, centroid, and finite camera/world surfels. Appearance and metric
   planes remain separate in `anchor_decision.json`.
2. `build_v19_demo_pose_binding.py` passes explicitly ready P15 rows through.
   For explicitly unready P15 only, it retains eligible direct P14 rows and
   interpolates absent, bracketed rows with linear translation and SO(3) Slerp
   on annotation time. The supported P14 `missing_accepted_metric_surface`
   placeholder is also an interpolation target **only** when it carries no
   transform values or positive direct/eligibility/generated evidence. The
   complete original placeholder (including object/status/mask evidence) is
   nested in `demo_visual_provenance.source_placeholder_row`; synthesized rows
   are explicitly latent/interpolated, not direct, eligible, observed, or ready.
   Direct transforms/statuses and all original direct fields remain unchanged.
   Unknown readiness/statuses, contradictory placeholders, ineligible transformed
   rows, positive `generated_geometry_pose_evidence_consumed` (or legacy
   `generated_geometry_pose_evidence`), object mismatches, nonfinite rigid poses,
   extrapolation, and duplicate rows fail closed. Missing endpoints are rejected,
   never nearest-held. This is a precise amendment to the absent-row-only contract.
3. The same stage writes a separate visual mesh. It applies at most one static
   uniform scale about a declared pivot, checks the configured bound, and
   verifies finite source/scaled vertices, valid indexed triangles, exact face-array
   preservation and serialized vertex order/coordinates (absolute PLY tolerance
   `1e-6 m`). Identity scale is used when there is no externally produced
   visual-scale record. The mesh-bound role contract explicitly enables only
   `demo_visual_proximity_fit_eligible=true`; measured/physical contact,
   collision, sign, SDF, and nonpenetration authority remain false.
4. `fit_v20_visual_contact_demo.py` requires the appearance manifest in visual-v2
   mode and checks its resolved path and SHA against the pose binding. It does
   not fall back to object-owned masks. P17 **visual judgment** likely/possible
   states permit forming fingertip hypotheses, not measured contact;
   **every fingertip, including the thumb**, requires finite actual
   triangle distance below `0.012 m` plus image-mask proximity. The generated
   mesh is a visual proximity reference only: signed intrusion/nonpenetration
   loss is disabled, and unsigned triangle distances drive proximity fitting.
   The output remains `annotation_ready=false`.

The old direct integration-stage demo is retained only as explicit legacy mode:
`--legacy-ready-p15-binding` and `--legacy-owned-mask`. It cannot consume an
unready P15 and is not used by the visual-v2 runner. Even a ready source produces
an unready diagnostic demo binding; source readiness is retained only in
`core_source_identity`.

## Prepare and launch

Build a fresh runtime bundle from the committed source first; runtime closure
now recursively includes the additive runner and its dependencies. Then prepare
an external native request (the SAM2 track is explicit, never discovered by
label heuristics):

```bash
PY=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
"$PY" scripts/prepare_v19_additive_demo.py \
  --run-root "$RUN_ROOT" \
  --source-bundle "$SOURCE_BUNDLE" \
  --demo-bundle "$FRESH_DEMO_BUNDLE" \
  --output-dir "$NEW_REQUEST_DIR" \
  --sam2-track "$RUN_ROOT/measurements/object_tracks/sam2_agent_points/<track>/sam2/sam2_track.json" \
  --demo-id visual_v2 \
  --render-width 960 \
  --gpu 5
```

`--object-id` is optional but, when supplied, must match
`state/object_selection.json`. The prepare step byte-binds object selection,
raw manifest and every RGB, SAM2 track and every available mask, P14, P15,
completion and mesh, HaWoR/MANO bridge, P17 visual judgment, video, and both
bundle manifests. Current P15 + HaWoR/MANO + SAM3D + P17 visual judgment are
sufficient upstream stages: preparation and the default runner neither discover
nor require core P16/P17 factor/P18/P19 outputs. The fitter receives the byte-bound
`judgment` input via `--interaction-judgments`. No optional factor-input mode is
introduced by this patch. Visual-judgment lineage still requires the current
selected-object P15 and current P01 frame evidence.
The case must equal the live runtime-input case and be a safe
path component. The track path, track ID and selected object must agree with both
the byte-bound annotation adapter and its producer report/object plan. Invisible
track rows with stale masks are provenance only, never appearance or fit evidence.

Before probing the GPU, the runner verifies the live run identity and the curated
bundle's files/assets, and requires the exact bundle-local runtime config. Every
original input, raw RGB and mask is checked against its prepared digest before
and after each phase. Generated consumer inputs are checked against their prior
successful producer digests, never a newly accepted post-consumption digest.

Each phase writes to a create-only staging directory; required outputs and source
hashes must validate before atomic publication. The binding now publishes at
`state/demo_visual/<object>/<demo-id>/visual_binding/`. Embedded RGB/mask/mesh
references name published paths rather than staging locations. Failures, including
config/interpreter/GPU preflight errors, have a local failure receipt; partial
staging is preserved under hidden `.failed` debug namespaces, not final phase
paths. D00 can be retried at an absent final destination while preserving its
prior failed debug tree. Full branch receipts remain create-only: use a fresh
`--demo-id` for a new branch attempt.

A visual scale record is optional and must provide:

```json
{
  "uniform_scale": 1.0,
  "bounds": [0.8, 1.2],
  "pivot_canonical_m": [0.0, 0.0, 0.0],
  "selection_source": "model-produced or silhouette-review record identity"
}
```

## Remaining SAM3D stage

This code slice intentionally does **not** launch model inference. The anchor
report contains a fail-closed optional reconditioning contract: a future
separate demo-only SAM3D producer must condition on the selected raw RGB plus
copied raw SAM2 mask, use same-frame metric surfels only in its metric bridge,
preserve native face indices exactly, and write a new demo completion under the
same demo namespace. It must not weaken `RunLineage`, overwrite core P13 or
completion, or promote generated geometry to physical authority. Until that
producer is integrated, visual v2 retains the complete current completion mesh
plus the bounded static alignment; uniform scale cannot recover missing shape
or topology. Dynamic visual scale optimization also remains unimplemented; an
external static scale record is not that mechanism.

Completion of the branch means only `completed_pending_visual_review`. Full
source-duration videos must be compared against the prior demo, including the
P14/P15 discontinuity window. A remaining visible jump is a bounded demo
failure, not annotation readiness.
