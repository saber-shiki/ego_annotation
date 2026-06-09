# Version Delivery Audit

This audit uses the delivery standard from `PROMPT.md`: a deliverable video must cover a meaningful continuous action interval and show the annotated video, the 3D world reconstruction, MANO hands, head camera, object mesh, and semantic caption. Short clips of three to seven frames count as QC evidence for individual mechanisms.

## Current Delivery Status

v1 produced full-clip videos. Its object representation is physically weak and uses proxy object state in key cases, so it stands as a baseline artifact.

```text
/data2/ego_annotation_outputs/fullmesh_task7/fused/side_by_side.mp4
/data2/ego_annotation_outputs/fullmesh_task5/fused/side_by_side.mp4
/data2/ego_annotation_outputs/representative_trash/fused_bagprompt_full_final/side_by_side.mp4
```

v2 produced a 91-frame continuous contact-window video with observed object surface geometry. It is a milestone demo for observed-surface meshing.

```text
/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict_render_840_930/side_by_side.mp4
```

v3 produced investigation clips. It failed as a clean pipeline version because scope expanded during implementation around segmentation, mesh completion, depth scale, hand refit, and contact.

v4 through v6 produced 31-frame wild-rice videos. These are useful continuous evidence windows for dynamic surface and sparse correspondence work, but they are still short relative to a complete action and represent selected problem intervals.

```text
/data2/ego_annotation_outputs/representative_wild_rice/v4_world_reconstruction_completed_measurement_plus_sam2seed_finalvis_2520_2550/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/representative_wild_rice/v5_world_reconstruction_state_presentation_2520_2550/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/world_reconstruction_side_by_side.mp4
```

v7 through v15 produced short QC/evidence clips, mostly three to seven frames. Those clips support inspection of mechanisms such as mesh replay, MANO repair, hidden-surface fusion, and contact dynamics. They remain below the task delivery standard.

Representative short evidence clips:

```text
/data2/ego_annotation_outputs/representative_box_books/v7_box_books_similarity_refit_box_with_books_616_618/deliverables/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/representative_box_books/v8_probe_box_books_612_617/deliverables_tail_v1/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/representative_trash/v9_partcrafter_fused_prior_trash_865_870/observed_plus_hidden_prior_iterativeraster_60k/deliverables/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v10_mesh4d_consecutive_outputs/fused_hidden/wild_rice_mesh4d_hidden_compact_2538_2543/deliverables/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered/deliverables_preferred/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/deliverables_dynamics_final/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/deliverables_mode_dynamics_final/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/deliverables_handoff_final/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/deliverables_switch_final/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/deliverables_switch_final/world/world_reconstruction_side_by_side.mp4
```

## Versioning Correction

Future pipeline versions must begin with a design document before implementation. The design document must define:

- the action interval length and representative samples;
- state variables for head pose, MANO hands, object mesh, contact, and caption;
- model-produced perception inputs;
- optimization objective;
- physical consistency terms;
- acceptance predicates;
- render deliverables;
- expected failure modes.

Bug fixes, threshold corrections, renderer improvements, and short diagnostic studies belong inside the current version as patches or sub-experiments. A new top-level version number requires a new upfront pipeline definition.

## Immediate Consequence

The current repository contains many validated components and short evidence clips. The missing artifact is a v15-quality full action-window deliverable. The next work item is to define the next real pipeline version upfront, then run it over meaningful continuous intervals on representative samples.
