# Pipeline V19 current epistemic state

## Current supported claim

The HOT3D `clip-001850` pinhole V19 repaired v5 run is now frozen as the accepted prediction boundary for the mesh-frame repair scope. The accepted boundary is the A800/truenas run root `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`, mirrored locally at `/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.

The supported mechanism claim is narrow: the prior wrong-pose artifact was caused by P18/P19 applying P13 completed-canonical poses to raw P12 TRELLIS mesh vertices. Rebinding P18/P19 to P13 `outputs.completed_mesh_labeled` repaired visible keyboard registration in the decisive canonical frames 0032 and 0074. The freeze does not claim exact keyboard geometry, direct visible pose at frame 0074, certain contact closure, or HOT3D score quality. Provenance: OPS mesh-frame localization, repaired P18-P21/freeze, and local sync entries.

## Accepted freeze evidence

- Freeze manifest: `state/v19_prediction_freeze_manifest.json`, sha256 `2d999fbc4f51e375b4a9eb7277943bba74ce542cf0c613ece3479c888a267080`.
- Status fields: `status=frozen`, lineage `v5_focalfix_coordrigid_mesh_frame_repair_after_rejected_wrong_mesh_freeze`, `scoring_status=not_run_deferred_by_contract`, `hot3d_scoring_run=false`.
- Canonical video hashes:
  - `renders/v19_overlay.mp4`: `d223bce316bb88d90e179bab8813fa043f5a57c241750fd924479762fadd7137`.
  - `renders/v19_world.mp4`: `d5e430b25f0e43bcfd2d660e5f9dc2b4db4635143b1fdce0b27dd0485db0245d`.
  - `renders/v19_side_by_side.mp4`: `73ecf78b24a29994c3ba1ba276c78fb701c2ddecda3eb722fae7dab2f9cbbf64`.
- Remote and local ffprobe both show all three canonical videos are real full-duration outputs: 150 frames, 5.0 s.
- Runtime P21 evidence/harness event `visual_consumption_completed_mesh_frame_repair` records frame 0032 as direct corrected visible pose registration and frame 0074 as completed/interpolated pose registration.
- Parent visual inspection of runtime QC frames at `/tmp/v19_hot3d_clip001850_meshfix_qc/` agrees with runtime P21: the green mesh lies on the physical keyboard/key field at frames 0032 and 0074 rather than on the rejected right-edge/sleeve/table region.

## Rejected boundary superseded

The earlier v5 freeze manifest hash `daa3978f641691ef148720a75a9ec3bea35a22e55b6c01eb6a0eea0ed85f2a3c` is superseded and should not be used as a correctness artifact. It remains only an immutable record of the rejected wrong-mesh state.

The rejected mechanism was not ordinary uncertainty: P15 serialized transforms in the P13 completed-canonical frame (`rotation_world_from_completed_canonical_matrix`, `translation_world_m`), while P18/P19 consumed `measurements/geometry_completion/trellis_keyboard_seed42/trellis_mesh.ply`. Raw TRELLIS and completed metric meshes have different frames/scales, so the rendered keyboard-like mesh was predictably misregistered.

## Durable source/spec repair

Commit `1b0aa0f Require completed mesh provenance for V19 P18/P19` preserves the mesh-frame contract:

- `runtime/v19_runtime_spec.md` resolves downstream `COMPLETED_MESH_PLY` from P13 `outputs.completed_mesh_labeled` and forbids raw P12 `trellis_mesh.ply` for P18/P19.
- `scripts/solve_v18_joint_mano_interval_trajectory.py` and `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py` accept `--completion-report` and fail loudly if `--completed-mesh` disagrees with P13 source of truth.
- Py-compile passed for both changed scripts before commit.

## Remaining scope limits

- Geometry is approximate and elongated/noisy; the repaired freeze supports registration/timeline continuity, not exact keyboard reconstruction.
- Frame 0074 is still `completed_temporal_rigid_pose_uncertain` / interpolated between visible pose observations, not a direct local mask/depth pose observation.
- Contact closure remains posterior/uncertain; it is not an accepted exact contact/nonpenetration solution.
- HOT3D evaluation has not been run. Any HOT3D scoring must be a separate evaluator phase that consumes the frozen repaired run root without modifying prediction state.

## Next action if the user requests evaluation

Run HOT3D scoring only as an evaluator phase against the repaired frozen run root and report it as evaluation of the frozen prediction, not as a new prediction phase. If future visual review finds a new registration failure, the next live mechanisms are camera/image-size projection scaling or pose-fit/world-camera convention; do not patch uncertainty labels to cover a visible geometric contradiction.
