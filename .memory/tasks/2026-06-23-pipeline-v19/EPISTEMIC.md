# Pipeline V19 current epistemic state

## Current supported claim

The previously frozen HOT3D `clip-001850` pinhole V19 artifact is an immutable rejected snapshot, not an acceptable prediction. `Frozen` means the run root and key outputs were hashed so later evaluation could have a fixed boundary; it did not mean the physical annotation content was correct. User inspection of frames 0032 and 0074 exposed mechanism-level failures, so the pipeline must be iterated and rerun self-contained. Provenance: OPS 2026-06-26T14:45.

## Live failure mechanisms

1. **Object prompt coordinate contract bug.** Runtime P06 wrote keyboard prompts in source-video coordinates (`source_video_pixels_1408x1408`) while also storing `prompt_image_width=960`. P07 SAM2 previously scaled points as if they were already in 960-pixel prompt coordinates. Source-frame points such as x=975..1223 were therefore outside/near the SAM2 frame edge, causing SAM2 to track the right sleeve/table edge instead of the keyboard. Frame 0032 confirms this: the raw keyboard is visible, but the local SAM2 mask is mostly right image edge/sleeve/table, and the rigid mesh fit follows that wrong measurement.

2. **Rigid branch not enforced full-timeline.** The keyboard was classified rigid, but P15 only rewrote frames with accepted visible pose observations. Missing/ineligible frames remained `missing_initial_graph_pose`; frame 0074 confirms this by rendering no keyboard even though the raw keyboard is visible. A rigid object must carry an explicit uncertain pose through local mask failures; disappearing is an implementation bug, not uncertainty.

## Repairs in source/spec

`run_sam2_vlm_points_multiobject.py` now infers prompt coordinate size from `point_coordinate_frame`, validates points within the declared frame, and records prompt/source/SAM2 coordinate sizes. `solve_v19_rigid_object_pose_graph.py` now supports full-timeline rigid completion with status `completed_temporal_rigid_pose_uncertain` for interpolated/held poses; render/constraint helpers accept that explicit uncertain rigid pose status. `runtime/v19_runtime_spec.md` now requires P07 object-track self-check/repair before P08 and requires P15 full-timeline rigid pose completion. These are pipeline repairs, not manual output edits.

## Next required intervention

A runtime-owned A800 rerun is active from the repaired curated bundle: `v19_hot3d_001850_a800_v4_coordrigid`, run root `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v4_coordrigid_v1`. The first launch script was invalid because it inlined the multi-line system prompt; the corrected launcher passes the prompt as a quoted runtime variable and created no prediction artifacts before repair. Next evidence must come from runtime-owned outputs: P07 regenerated keyboard masks should no longer edge-clip source-coordinate prompts, P15 should emit full-timeline rigid poses including uncertain completions, and rendered frame 0032/0074 must be consumed before any new freeze/evaluation claim.
