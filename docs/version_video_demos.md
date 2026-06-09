# Version Video Demo Index

This file separates demo artifacts from pipeline maturity. A video path means a rendered artifact exists for that version label. It means the version label had a clean prior pipeline definition only when the status says so.

## Versioning Audit

v1 and v2 were pipeline milestones with explicit scope before the final renders.

v3 became an investigation package. It produced several useful videos and diagnostics, but it failed as a clean pipeline milestone because the scope expanded around contact, segmentation, depth, camera scale, and hand refit during implementation.

v4 through v7 regained a clearer design thread around dynamic object surfaces, sparse temporal factors, and replayable mesh acceptance.

v8 through v11 were object/hand repair and mesh-completion milestones. They have concrete design docs and deliverables, but the numbering also reflects rapidly discovered failure mechanisms.

v12 through v15 are best read as one contact-dynamics research thread with sub-versions: sliding contact, contact mode, same-point handoff, and contact switch. Future work should avoid bumping a new top-level version for each bug fix. A new version should start from an upfront design document with state variables, evidence sources, objective, acceptance predicates, representative samples, and expected demo outputs.

## Demo Videos

### v1: End-to-End RGB Baseline

Status: delivered baseline, physically weak object representation.

Side-by-side:

```text
/data2/ego_annotation_outputs/fullmesh_task7/fused/side_by_side.mp4
```

Video facts: 1920 by 540, 2040 frames, 30 fps.

### v2: VLM-Planned Observed Object Surface

Status: closed observed-surface milestone, overall pipeline still open.

Side-by-side:

```text
/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict_render_840_930/side_by_side.mp4
```

Video facts: 1920 by 540, 91 frames, 30 fps.

### v3: Referring Segmentation and Joint Metric Contact

Status: failed as a clean pipeline milestone. Use these videos as evidence of what was tested, not as a closed v3 pipeline.

Dense observed-surface side-by-side:

```text
/data2/ego_annotation_outputs/representative_trash/v3_world_reconstruction_dense_measured_sheet_finalvis2_858_880/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 23 frames, 6 fps.

Closed-SDF contact slice:

```text
/data2/ego_annotation_outputs/representative_trash/v3_world_reconstruction_closed_sdf_levelset_865_870/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 792, 6 frames, 2 fps.

### v4: Residual-Gated Object Tracks and Temporal Completion

Status: designed dynamic-object surface milestone with wild-rice delivery.

Side-by-side:

```text
/data2/ego_annotation_outputs/representative_wild_rice/v4_world_reconstruction_completed_measurement_plus_sam2seed_finalvis_2520_2550/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 31 frames, 6 fps.

### v5: Visibility-Aware Dynamic Object Mesh

Status: designed dynamic-object state and presentation upgrade, with dense-map hypotheses rejected and sparse-track evidence introduced.

Side-by-side:

```text
/data2/ego_annotation_outputs/representative_wild_rice/v5_world_reconstruction_state_presentation_2520_2550/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 31 frames, 6 fps.

### v6: Robust Sparse Correspondence Factors

Status: designed sparse-factor milestone with repaired wild-rice frame 2539.

Side-by-side:

```text
/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 31 frames, 6 fps.

### v7: Replayable Mesh Priors and Candidate Acceptance

Status: designed acceptance harness and representative measured-mesh delivery.

Side-by-side:

```text
/data2/ego_annotation_outputs/representative_box_books/v7_box_books_similarity_refit_box_with_books_616_618/deliverables/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 3 frames, 6 fps.

### v8: Contact-Aware MANO Repair

Status: hand/contact repair milestone on box-books.

Side-by-side:

```text
/data2/ego_annotation_outputs/representative_box_books/v8_probe_box_books_612_617/deliverables_tail_v1/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 792, 7 frames, 6 fps.

### v9: Evidence-Gated Hidden Prior Fusion

Status: generated hidden-geometry path accepted only after visible replay and physics checks.

Side-by-side:

```text
/data2/ego_annotation_outputs/representative_trash/v9_partcrafter_fused_prior_trash_865_870/observed_plus_hidden_prior_iterativeraster_60k/deliverables/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 6 frames, 6 fps.

### v10: Video-Conditioned Mesh4D Evidence

Status: video-conditioned object-prior test. Raw generated meshes failed visible-surface replay; filtered hidden evidence delivered a wild-rice result.

Side-by-side:

```text
/data2/ego_annotation_outputs/v10_mesh4d_consecutive_outputs/fused_hidden/wild_rice_mesh4d_hidden_compact_2538_2543/deliverables/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 6 frames, 6 fps.

### v11: Temporal Hidden-Surface Fusion

Status: temporal hidden-geometry milestone with contact-rich trash delivery.

Side-by-side:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered/deliverables_preferred/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 6 frames, 6 fps.

### v12: Sliding Contact Dynamics

Status: contact-dynamics sub-version over accepted V11 geometry.

Side-by-side:

```text
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/deliverables_dynamics_final/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 6 frames, 6 fps.

### v13: Contact-Mode Dynamics

Status: mode-aware contact-dynamics sub-version.

Side-by-side:

```text
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/deliverables_mode_dynamics_final/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 6 frames, 6 fps.

### v14: Contact Handoff Dynamics

Status: same-material-point handoff test. Trash accepted; box-books rejected that physical claim.

Side-by-side:

```text
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/deliverables_handoff_final/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 6 frames, 6 fps.

### v15: Contact-Switch Surface Dynamics

Status: contact-switch surface test. Box-books and trash accepted.

Box-books side-by-side:

```text
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/deliverables_switch_final/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 7 frames, 6 fps.

Trash control side-by-side:

```text
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/deliverables_switch_final/world/world_reconstruction_side_by_side.mp4
```

Video facts: 1920 by 778, 6 frames, 6 fps.
