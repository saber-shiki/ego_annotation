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
   on annotation time. Unknown readiness, invalid/latent P14 rows, extrapolation,
   and duplicate rows fail closed.
3. The same stage writes a separate visual mesh. It applies at most one static
   uniform scale about a declared pivot, checks the configured bound, and
   verifies exact face-array preservation. Identity scale is used when there is
   no externally produced visual-scale record. Generated faces remain
   ineligible for contact, collision, SDF, or nonpenetration authority.
4. `fit_v20_visual_contact_demo.py` requires the appearance manifest in visual-v2
   mode. It does not fall back to object-owned masks. P17 likely/possible states
   gate fingertip hypotheses, and weak non-thumb weights require both image-mask
   proximity and actual triangle-surface gap. The output remains
   `annotation_ready=false`.

The old direct integration-stage demo is retained only as explicit legacy mode:
`--legacy-ready-p15-binding` and `--legacy-owned-mask`. It cannot consume an
unready P15 and is not used by the visual-v2 runner.

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
completion and mesh, HaWoR/MANO bridge, P17, visual judgment, video, and both
bundle manifests. The runner rechecks those hashes before any subprocess.

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
or topology.

Completion of the branch means only `completed_pending_visual_review`. Full
source-duration videos must be compared against the prior demo, including the
P14/P15 discontinuity window. A remaining visible jump is a bounded demo
failure, not annotation readiness.
