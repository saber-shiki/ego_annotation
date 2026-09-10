# Restore depth-ordered ownership and rewind this run to P08

## User authorization and exact boundary

The user explicitly asks to restore the previous behind-hand mask fix and, since it affects downstream inputs, rewind the current bundle/run to the last unaffected phase and rerun with fixed code. This is NOT permission to overwrite old published demo/core artifacts or to relabel reused observations as fresh.

Sole writer worktree:
`/mnt/user-home/kupingxin/ego_annotation_worktrees/standalone_rgb_scene_depth_order_20260910`
Branch `pipeline/standalone-rgb-scene-depth-order-20260910`, base `9b829c017f169964cb191b8a9fe4ec3649786a91`.

Code/tests/docs and local commits only. No current run changes/moves, model/native execution, GPU, environment installation, deployed bundle edits or push/merge. Parent will stop/reconcile actor, apply the reviewed migration and run the new native continuation. CPU synthetic tests and read-only current/historical evidence allowed. Use deployed `/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`.

## Confirmed old fix (reuse mechanism, not historical outputs)

Old clean worktree `/mnt/user-home/kupingxin/ego_annotation_worktrees/sam3d_native_ghost_lite`, branch `research/sam3d-native-ghost-lite-alignment-20260826`, HEAD61c4f1674a613edffd354740484f644c7200f5d2.
Fix commit **637be2a7b429e06254070bdfe173ac87dd93b4b0**.
Files:
- `experiments/sam3d_native_ghost_lite_20260826/diagnose_hand_depth_order_ownership.py` SHA d5fe6cdf3dea9bf28116247f81e335bf995eff5cfb5e2fa0ce12aeb1954d3358
- `experiments/sam3d_native_ghost_lite_20260826/build_depth_order_counterfactual_masks.py` SHA02f65ad92ec56ca9b92ac2437b58291a709234ebf567018eb27603c1d4844ed7

These match the old corrected-run frozen code_snapshot/depth_order_patch copies. The old corrected run first generated `front_only` masks and then passed `--no-exclude-hand-bboxes` to P09, avoiding the second depth-blind removal. Actual old P09 log records exclude_hand_regions=false. Reference evidence:
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_depth_order_corrected_fullrun_20260903T101508Z/code_snapshot/depth_order_patch/MASK_DEFECT_PROVENANCE.json`
and `.../logs/03_p09_visible_geometry.log`.

Old rule (5mm declared tolerance): compare nearest actual MANO surface camera-z and same-pixel measured depth. Remove only z_hand < z_depth - tolerance. Preserve z_hand > z_depth + tolerance (behind), near-equal ambiguous, and padding/no-hit in the appearance mask. No bbox/centroid/depth-independent silhouette ownership.

Important hardening while extracting: old CPU diagnostic uses screen-space linear z interpolation. Use perspective-correct first-hit depth (inverse-z barycentrics or tested actual-ray triangle intersection) in the new reusable numerical helper. Do not blindly inherit this approximation, fixed milk paths/frame assumptions, replace/rmtree flag, or the need for an already-broken P09 annotation to bootstrap. Build directly from current P08 MANO, current depth contract and current SAM2 masks.

## Confirmed current defect / evidence

Current fresh run (read-only during code work):
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/cold_raw_rgb_v1`
UUIDded84007-0cd4-4717-b16a-2865e62eaea1, casergb_scene_20260910_v1, selected milk_carton/track milk_carton_track. These names are evidence only, NEVER special code branches.

Current source `run_rgb_scene_frontend.py` passes --exclude-hand-regions for both P09-propose andP09. `build_v19_visible_geometry_from_sam2_depth.py` subtract_hand_owned_bbox_regions simply applies `out[silhouette]=False` before depth filtering, no hand-vs-object depth test. The separate robust_first_surface_depth_ownership cannot restore already-erased pixels.

Parent ran the old diagnostic on current completed P09 without changing inputs:
`.../standalone_pipeline_20260910/hand_depth_order_audit/current_frames_legacy_depth_diagnostic.json`
`.../hand_depth_order_audit/current_frames_depth_order_review.jpg`

Current removed counts (diagnostic z is predicted depth, not GT):
- f0:30169removed,25479behind,0front,4690no-hit; median behind gap~90mm;
- f65:14568removed,11695behind,0front,42ambiguous,2831no-hit;
- f92:7457removed,5291behind,10front,23ambiguous,2133no-hit;
- f132:13993removed,10577behind,0front,6ambiguous,3410no-hit;
- f149:21142removed,18814behind,0front,2328no-hit.
This is a large upstream evidence loss, not a cosmetic final-overlay defect. Core P09 errors also bias initial pose and later RGB/observed factors. Need rerun suffix, not just switch appearance mask during final fit.

## A. Real reusable P09 ownership correction

Keep prior core/shared producer entrypoints unchanged by default. Add a standalone-owned depth-order helper/adapter or an explicit new opt-in core-compatible input seam if strictly necessary; only standalone new path should activate it. Avoid a wholesale duplicate2300line P09 implementation or post-hoc relabeling of P09 outputs. Prefer direct fresh P08/SAM2/depth consumer before both P09proposal and P09, with exact full-MANO first-hit classification.

Requirements:
- Match camera-space bridge vertices to source hand/frame identity and topology, apply full active source K and explicit half-pixel source→mask/depth transforms. Handle non-square/different raster sizes. Compare depths only in the same camera/units/pixel axis.
- Per-pixel nearest hand over full actual left/right MANO triangles; positive-depth/truncation/missing geometry diagnostics; no face deletion for visual convenience and no bbox fallback. Small confidence/depth-ambiguous/no-hit regions preserve appearance and explicit uncertainty; they do not automatically become trusted metric support. Retain the existing robust-depth quarantine before lifting and fail only genuine malformed contracts/mechanics.
- Raw SAM2 appearance track remains immutable and separately bound. Depth-owned output is a new declared measurement with full hashes/report. No second silhouette-only subtraction after correction.
- Both P09proposal and P09 consume depth-ordered support. P11/P14/RGB source lifting and downstream refined pose must consistently consume corrected P09; appearance branch still has the ORIGINAL raw SAM2 appearance masks, not depth-pruned masks mislabeled raw.
- Use explicit source/owned mask identities in adapter reports and update standalone consumers/checks necessary to validate both tracks instead of faking `sam2_track` identity.
- Preserve valid existing source MANO arrays and old first-surface policy. No generated mesh/sign/collision/contact authority in ownership.
- Fresh standalone full run from RGB must include the corrected mechanism automatically, not only this recovery.

Tests: behind triangles preserve object pixels; front triangles remove only supported first-hit pixels; ambiguous/no-hit/padding cannot erase raw appearance; slanted triangle uses correct perspective z; overlapping hands use nearest hit; correct camera/pixel resize; malformed/NaN/near-plane/missing-hand policies; arbitrary object IDs/time/raster; exact proposed+actualP09 wiring and raw appearance retained; no duplicated blind subtraction. Reuse old implementation concepts and test with fresh current P08 inputs read-only to show defect actually reduced.

## B. Explicit same-run rewind after P08, with invalid suffix archived

Current native03526946 is being told to finish its current command then stop. Worker MUST NOT operate on or migrate the run. Parent handles it only after terminal/process reconciliation.

Need implement a developer-side recovery prepare/apply path and native mode. Bounded single transition `P09_DEPTH_ORDER` for this implementation, not arbitrary stage edits. Use immutable original input/runtime_input_contract.json and run UUID. Old bundle remains intact. Bind new reviewed bundle and exact original event prefix ending in successfulP08 (and current P05 judgment). Reuse ONLY P01, camera/depth binding,HaWoR,P05,P06,P07,P08 and their verified ancestors. Old suffix P09-propose/P09/P11/geometry/initial/refined/appearance/shape/fit/export is INVALID for this continuation even if command returned0.

Suggested safe implementation (adapt only if stronger):
1. Preparer requires original bundle/source identity, stopped-launch evidence and no active owned model/stage process, complete current event file, successfulP08 prefix. Verify preserved prefix bytes/outputs and P05 visual judgment/raw identity with OLD code's strict lineage.
2. Inventory/hash every affected suffix file and decision. Create a new recovery directory and archive/move ONLY named affected outputs (P09proposal/visiblegeometry, P09anchor/P10rigid/object_selection, P11 selected evidence, old SAM3D/posefits/scene/appearance/fit/final if present) into explicit `invalidated/` namespace with original→archive mapping/hashes. Do NOT move raw/SAM2/P08/HaWoR/depth/P05, old failures or unrelated paths. Original events/logs remain frozen evidence; do not rewrite/re-stamp them. Record safe partial-apply failure and refuse launch if all moves not completed/verified.
3. Recovery-specific lineage recognizes only exact preserved oldP08-prefix event digests plus newevents in a separate append-only recovery event path, validates old code identity on retained receipts and new code identity/recovery-contract SHA on new receipts. It must reject stale invalidated suffix paths even if restored manually. The birth input contract is not edited to pretend it was born under newcode. Changed code/assets allowlist must be narrow: corrected ownership/standalone routing/recovery helpers/profile/docs/spec/tests; no upstream weight/calibration/video/model changes. Path differences inside newbundle runtime config normalized deliberately, never permit external model swaps.
4. New native continuation starts exactly with corrected ownership/P09-propose, requires NEW current candidate image inspection/anchor decision (may differ from old92), then P09/P10/selection/P11 and all downstream standalone stages. No prefix rerun, no reuse old anchor/geometry/poseRGB even if hashes still exist in archive. New logs/handoff/failure under recovery namespace, old blocker's bytes unchanged. Stage allowlist enforced in code; no camera/P01/P06/P07/P08 with recovery mode.
5. Final JSON portable identity retains birth run/video/calibration and explicitly adds recovery execution bundle/contract hashes. It must not require birth bundle hash == current bundle by overwriting either; both identities preserved. Fresh mode and existing same-code successful-resume remain tested and unaffected.

This migration is required to honor the user's requested rewind rather than waste recomputing unaffected models or falsify fresh results. Keep it narrow and testable. If the suggested path needs substantial design change, ask supervisor before coding an unbounded invalidation framework. Do not relax hash verification to make old predictions pass.

Recovery fixtures: valid retained-prefix + invalid suffix, preservation hashes, correct archive mapping and no upstream mutation, refusing active/incomplete/foreign identity, partialmove failure, rejecting oldsuffix as active producer, per-stage old/newcode identity, fresh+samecode resume compatibility, proper continuation stages/defaultnoMP4/final birth+execution identities. Model tests mocked, no real-run writes.

## C. Validation and finish

Run focused numerical/ownership/lineage/recovery tests, full unittest/compile/diffcheck, local commits and clean handoff. Include exact parent oldbundle→newbundle preparation/apply and native launch payload. No actual run migration or model execution by worker. IndependentAstra/Sol review follows, then parent stops/reconciles oldactor, applies recovery and launches corrected suffix.

Be concise in docs/receipt scope. The artifact-changing point is the real first-hit depth comparison preserving visible object support and making fresh corrected P09 drive the entire suffix, not a new uncertainty label alone.

## Part B implementation contract (bounded P08 recovery)

Part B is implemented by `scripts/rgb_scene_recovery.py` and the three developer-only subcommands in `scripts/prepare_rgb_scene_runtime.py`. Preparation and verification are read-only with respect to the run and write only a new external package directory. Apply is the sole run-mutating command. It hashes and moves the explicit P09-propose-and-later file inventory into `state/runtime_recovery/P09_DEPTH_ORDER/invalidated/`, appends intent/completion rows to `apply_journal.jsonl`, removes emptied invalid suffix directories, and installs `recovery.json` plus `apply_complete.json` only after every move verifies. An interrupted apply has no completion contract and runtime launch refuses it.

The old `logs/integration/events.jsonl` and `logs/integration/*.log` files stay byte-unchanged. Runtime lineage loads only the snapshotted exact P01-through-P08 event prefix plus new `logs/recovery/P09_DEPTH_ORDER/events.jsonl`; merely restoring an archived old suffix path cannot make it a producer. Retained receipts require the original bundle code identity. New receipts require both the new bundle manifest SHA and recovery contract SHA. Runtime code rejects all recovery frontend phases except `P09_DEPTH_ORDER`, `P09-propose`, `P09`, and `P11`, and rejects the `camera` stage; the remaining fixed standalone suffix stages are allowlisted.

The stopped-run evidence is an owner stop/reconciliation record, not a prediction actor receipt. Its exact run UUID, actor/workflow UUIDs, terminal-observed/non-resumable state, empty owned-process snapshot, immutable input SHA, full original-event SHA/length, and P08 prefix SHA/length are checked. No nonexistent actor handoff is required or synthesized. Preparation and apply also perform a live run-root-bound owned-process check; the run owner must independently reconcile owned processes immediately before apply.

Developer command sequence from this reviewed source checkout (the target bundle and package paths must be new):

```bash
PY=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
SRC=/mnt/user-home/kupingxin/ego_annotation_worktrees/standalone_rgb_scene_depth_order_20260910
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/cold_raw_rgb_v1
OLD=/mnt/user-home/kupingxin/ego_annotation_runtime/rgb_scene_standalone_candidate_9b829c0_gpu5_20260910
STOP=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/hand_depth_order_audit/owner_stop_03526946/stopped_run_evidence.json
NEW=/mnt/user-home/kupingxin/ego_annotation_runtime/rgb_scene_standalone_depth_order_recovery_<reviewed-commit>_gpu5_20260910
PKG=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/hand_depth_order_audit/P08_recovery_<reviewed-commit>

# First build NEW with the existing bundle command and the preflighted asset-path JSON.
"$PY" "$SRC/scripts/prepare_rgb_scene_runtime.py" prepare-recovery --run-root "$RUN" --source-bundle "$OLD" --target-bundle "$NEW" --stop-evidence "$STOP" --package-dir "$PKG"
"$PY" "$SRC/scripts/prepare_rgb_scene_runtime.py" verify-recovery --package-dir "$PKG" --run-root "$RUN" --target-bundle "$NEW"
# Reconcile owned processes again, inspect/review the package, then and only then:
"$PY" "$SRC/scripts/prepare_rgb_scene_runtime.py" apply-recovery --package-dir "$PKG"
"$PY" "$SRC/scripts/prepare_rgb_scene_runtime.py" verify-applied-recovery --run-root "$RUN" --target-bundle "$NEW"
```

The new native task payload is exactly:

```json
{"mode":"p08_depth_order_recovery","runtime_cwd":"<NEW>","run_root":"<RUN>","recovery_contract":"<RUN>/state/runtime_recovery/P09_DEPTH_ORDER/recovery.json","handoff_path":"<RUN>/state/runtime_recovery/P09_DEPTH_ORDER/native_recovery_handoff.md","tmux_target":"<new dedicated visible target>"}
```

The runtime profile and sole runtime spec define the executable continuation commands. No agent CLI is invoked by these developer helpers. Fresh prediction and same-code verified resume remain on their existing paths.
