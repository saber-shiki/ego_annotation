# Optional demo native reconditioning (current-version patch)

This source slice adds a callable demo-only mechanism, not a new complete pipeline version or a visual acceptance claim. Core producers, core readiness/physical gates, original MANO, and the final assembler are unchanged.

## Mechanism and discriminating observations

The defect addressed is that the demo previously retained the core frontal mesh even when a different appearance anchor was available. Native SAM3D now receives the selected source RGB and full SAM2 appearance mask. Only that same frame's retained positive metric surfels supply scene depth; appearance pixels never become trusted depth. The unchanged native local-to-OpenCV transform and P13 front-depth numerical helper are reused without manufacturing core P11/P13 lineage.

The full mesh is mapped through the selected world-from-camera pose into the **original core canonical gauge**, using the unchanged `build_demo_pose_rows` initializer. Explicitly unready P15 requires a genuine direct P14 anchor, not an interpolated row. Ready P15 uses its own selected pose. Both native pose meshes and all serialized canonical vertices/faces are replay-checked. D00 clearance scoring now saturates above a declared 5% short-edge safety margin; larger safely interior appearance support is not penalized for being less centered.

D00c optionally searches one static uniform scale about the canonical vertex-mean pivot. Its objective is mean `1-IoU` between full-triangle silhouette rasters and deterministic multi-frame appearance masks with fixed cameras/initializers. Defaults: bounds `[0.8,1.2]`, 41 grid values plus identity, at most seven direct/ready evidence frames, 192-pixel-wide aspect-preserving rasters. At least three usable frames are required. Flat/invalid objectives fail rather than claim alignment. Bound hits are recorded without expansion or retry. Occlusion, camera/pose error and finite raster/grid resolution remain explicit uncertainty; this is not metric scale or contact evidence.

For the authorized model run, distinguish:
- A different complete native mesh and improved silhouette after correct gauge replay would support appearance conditioning as the limiting mechanism.
- A model/native-support failure leaves native execution logs/staging and no fallback mesh; inspect the actual native conditioning/projection before changing the mechanism.
- A valid new mesh with poor silhouettes or a scale bound hit indicates remaining mesh/pose/camera/occlusion ambiguity. Do not widen scale bounds automatically or reinterpret visual proximity as measured contact.

## Parent preparation and launch

Only after source review, prepare a **new curated execution bundle** from this commit using the normal parent-owned bundle preparation. Do not modify an old deployed bundle. The closure automatically includes the new producers/helper and unchanged native runner/depth helper. SAM3D uses pinned `sam3d_python`, offline config/weights/source assets checked with the existing `geometry` asset role (including checkpoint bytes before launch), one contract-selected physical GPU, forced offline flags and a 1200-second native subprocess timeout. No installation or download is part of prediction.

Using the pinned main Python and this reviewed development source for the parent-only preparer (which is deliberately not in the curated runtime closure), the new preparer options are:

```bash
"$MAIN_PYTHON" "$REVIEWED_SOURCE/scripts/prepare_v19_additive_demo.py" \
  --run-root "$RUN_ROOT" --source-bundle "$CORE_SOURCE_BUNDLE" \
  --demo-bundle "$NEW_DEMO_BUNDLE" --output-dir "$FRESH_REQUEST_DIR" \
  --sam2-track "$CURRENT_SELECTED_SAM2_TRACK" \
  --object-id "$OBJECT_ID" --demo-id "$FRESH_DEMO_ID" \
  --gpu "$AUTHORIZED_PHYSICAL_GPU" --minimum-free-mib 20000 \
  --reconstruct-visual-mesh --sam3d-seed 42 \
  --estimate-visual-scale --visual-scale-bounds 0.8 1.2 \
  --visual-scale-samples 41 --visual-scale-max-frames 7 \
  --visual-scale-raster-width 192 --render-width 960
```

Use the generated `request.json` with the existing parent launch mechanism; the native executor reads and executes the launch contract's exact `command` array, serial managed tools, timeout1200 and stop-on-failure. The source runtime profile spells out these stages. To compare reconstruction alone, omit `--estimate-visual-scale`. To retain the old mesh path, omit `--reconstruct-visual-mesh`; scaling is independently optional. A supplied `--visual-scale-record` cannot be combined with estimation. The old explicit scale-record behavior remains available for backward comparison, not mislabeled as optimization.

## Published outputs and failure semantics

All paths are beneath `<run>/{measurements,state,logs,renders}/demo_visual/<object>/<demo-id>`:

- `measurements/.../appearance_evidence/{appearance_mask_manifest,anchor_decision}.json`
- `measurements/.../visual_sam3d/visual_completion.json`, `original_canonical_mesh.ply`, native raw/OpenCV/PyTorch3D meshes, QC and execution log/receipt
- `measurements/.../visual_scale/visual_scale_record.json`
- `state/.../visual_binding/{pose_binding.json,visual_mesh.ply}`
- unchanged D02 `measurements/.../fit/{demo_state.npz,fit_report.json,contact_hypotheses.json}` and D03 render outputs
- `state/.../receipts/{events.jsonl,branch_result.json}` or `failure.json`

The branch hashes staged output bytes **before** atomic rename and checks those exact hashes after publication and at every consumer/closure boundary. Native QC/argv retain original staging paths as execution provenance only; authoritative consumers resolve through explicit published path+SHA records in `visual_completion.json`. A failed phase preserves `.D00b_visual_sam3d.failed` / `.D00c_visual_scale.failed` when staging exists and stops before binding/fit/render. No enabled reconstruction failure substitutes the core mesh.

The D01 `core_source_identity` still names original P14/P15/completion. `visual_completion_provenance`, `visual_mesh_receipt`, and `demo_pose_initializer` are separate typed visual sources. Fit reports now snapshot both initializer and mesh provenance; `base_object_*` remains the visual initializer, **not** a claim of equality to core P15.

## Validation and remaining boundary

CPU tests mock only native subprocess/GPU/asset placement, exercise the actual numerical reconstruction, serialization, optional scale estimator and binder, and add enabled/disabled phase routing, failure, asset tamper and closure coverage. Synthetic recovery uses nonidentity scale and multiple aspect ratios. No GPU/model or native-agent launch was performed in this implementation lane, and no existing run/bundle was written.

The final assembler/package is intentionally not changed. Its known legacy core-mesh/base-P15 assumptions will reject a distinct visual mesh/initializer until a separately reviewed typed binding amendment is implemented. Parent must run the authorized native stage and inspect full-timeline MANO/mesh render outputs before claiming visible improvement. Runtime of real native inference and dense-mesh silhouette alignment remains to be measured on the designated server.
