# Pipeline V19 current epistemic state

## Current supported claim

HOT3D `clip-001850` pinhole V19 has a frozen A800-native prediction through full physical-state generation and full-duration user-facing renders. The freeze is anchored by remote and local run root `20260626_hot3d_clip001850_pinhole_a800_native_v3`, canonical videos `renders/v19_overlay.mp4`, `renders/v19_world.mp4`, and `renders/v19_side_by_side.mp4`, and `state/v19_prediction_freeze_manifest.json`. The freeze manifest records non-empty 150-frame / 5.0 s / 30 FPS canonical renders and hashes for key state artifacts. HOT3D scoring has not been run and remains outside this prediction freeze. Provenance: OPS 2026-06-26T14:03.

## Mechanism that produced the artifact

The delivered render is driven by the V19 runtime chain: raw frame/depth/calibration, HaWoR MANO candidates, keyboard object plan/prompts, SAM2 visible-surface track, metric visible geometry, rigid branch decision, TRELLIS prior mesh, compact rigid completion, visible-frame and temporal rigid pose graph, MANO/object constraint state, visible contact/ownership factor, interval MANO correction, and canonical publication. P21 visual consumption confirmed the overlay/world/side-by-side samples show the same keyboard geometry and hand hypotheses rather than empty containers.

## Important caveats

The frozen prediction is not a hard-contact proof. P21 evidence and interval metrics show both hand intervals are still labeled `INTERVAL MANO UNCERTAIN`; left active-set closure failed, right closed, and median contact normal gaps are about 21 mm. Treat contact, occlusion, and nonpenetration as uncertain physical annotations, not final ground-truth contact closure. P13 event metadata still has a stale `completed_mesh` field pointing at raw TRELLIS, but the actual compact mesh used downstream is `measurements/geometry_completion/compact_keyboard_seed42/keyboard_compact_rigid_completed_mesh_labeled.ply`.

## Ruled out / repaired failure mechanisms

P12 did not fail physically after the TRELLIS bridge; the failure was an output-contract bug that globbed for `*report*.json` and risked selecting the Gaussian `.ply`. The correct P12 outputs are `qc_trellis_shape_v3.json` and `trellis_mesh.ply`. P17 failure was schema mismatch in agent interaction judgments, repaired by requiring non-empty `interaction_judgments` rows. P18 failure was dependency/bundle closure for WiLoR MANO assets and helper scripts, repaired with bundle-local WiLoR/MANO paths. P19 failure was a stale renderer default constraint-report path, repaired by passing the P16 constraint state. P20 false success was caused by truenas symlink semantics creating zero-byte canonical files, repaired by copying valid published MP4s and patching the publisher to fall back to copies.

## Next decision point

If evaluation is desired, use the frozen manifest and canonical videos as the immutable prediction boundary before running any HOT3D scoring. Do not mutate the frozen prediction while evaluating.
