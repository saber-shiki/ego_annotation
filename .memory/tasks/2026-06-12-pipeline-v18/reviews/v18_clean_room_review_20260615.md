# V18 clean-room adversarial review — 2026-06-15

PASS: I did not find a must-fix false-contact, final-support-gate, render/count, or readiness-claim violation in the inspected V18 committed state and latest final artifacts.

## Evidence inspected

- Commits reviewed: `74f6fab Add bounded visual contact prior` and `62a0824 Support local surface-changing contacts` against the prior V18 code.
- Files reviewed: `scripts/run_v18_full_pipeline.py`, `scripts/validate_v18_full_pipeline_artifact.py`, `scripts/validate_v18_factor_graph.py`, `docs/pipeline_v18.md`, `docs/pipeline_v19_design_proposal.md`.
- Latest artifacts reviewed under `/data2/ego_annotation_outputs/v18_full_pipeline/` for `trash_1050` and `task5_tomato_960`.
- Validators rerun successfully:
  - `.venv/bin/python -m py_compile scripts/run_v18_full_pipeline.py scripts/validate_v18_full_pipeline_artifact.py scripts/validate_v18_factor_graph.py`
  - `.venv/bin/python scripts/validate_v18_full_pipeline_artifact.py --report /data2/ego_annotation_outputs/v18_full_pipeline/v18_full_pipeline_report.json`
  - `.venv/bin/python scripts/validate_v18_factor_graph.py --root /data2/ego_annotation_outputs/v18_full_pipeline`
  - `git diff --check`

## Contact activation review

The two active task5 tomato contacts are the only active depth-contradicted contacts in the final JSON.

- Frame 926, left hand / `object:obj_tomato`:
  - `physical_contact_mode=active_physical_contact`, `estimate=true`.
  - `depth_contradiction=true`, but `depth_conflict_blocks_active_contact=false` and `visual_contact_prior_overrode_weak_depth_conflict=true`.
  - `visual_contact_prior.contact_prior_supported=true` with image contact, `mesh_contact_support_score=0.9975301277197833`, `effective_metric_contact_distance_m=0.0275895738745561`, observed HaWoR hand support, and `nonpenetration_conflict=false`.
  - Final support path is `surface_changing_visible_depth_silhouette_pose`; `post_graph_final_support_allows_active_contact=true`.
  - Object validation keeps `object_geometry_complete=false` and `object_pose_requirement_met=false`.

- Frame 927, left hand / `object:obj_tomato`:
  - `physical_contact_mode=active_physical_contact`, `estimate=true`.
  - `depth_contradiction=true`, demoted only by the bounded visual prior; `depth_conflict_blocks_active_contact=false` and `visual_contact_prior_overrode_weak_depth_conflict=true`.
  - `mesh_contact_support_score=0.9952582902556185`, `effective_metric_contact_distance_m=0.059176477598701385`, observed HaWoR hand support, and `nonpenetration_conflict=false`.
  - Final support path is `surface_changing_local_visible_contact_surface`; local support records scope `contact_support_only_not_full_object_pose_or_hidden_geometry_completion`, observed projection inside-mask fraction `0.875`, and observed-to-predicted median residual `0.06691063021499072 m`.
  - The same object validation explicitly rejects full compact visible pose on this frame (`surface_changing_compact_visible_pose_supported=false`, blocker `projected_mesh_vertices_have_weak_mask_support`) and still keeps `object_geometry_complete=false` / `object_pose_requirement_met=false`.

Other depth-contradicted rows remain blocked/non-active unless they are explicitly non-active uncertainty modes. In task5, there are 522 depth-contradicted/override rows, but only the two rows above have a supported visual prior and active mode. In trash, there are no visual-prior active depth-contradicted contacts; active contacts are 8 deformable same-frame visible-surface contacts without depth contradiction.

## Final support and render/count consistency

- Every active contact in both cases has `post_graph_final_support_paths_present=true`, `post_graph_final_support_allows_active_contact=true`, and a non-empty `physical_contact_mode_support_paths` list.
- Non-active renderable modes are not counted as active:
  - `trash_1050`: 8 active contacts, 5 depth-occluded possible rows, 33 supported-near noncontacts. Overlay/world active counts are 8 contact lines/edges; non-active lines/edges are counted separately.
  - `task5_tomato_960`: 2 active contacts, 0 depth-occluded possible rows, 1 supported-near noncontact. Overlay/world active counts are 2 contact lines/edges; the supported-near row is counted separately.
- World active contact rendering reports no missing metric endpoints.

## Constraint/invariant review

- The visual prior is not an unconstrained oracle in the current artifact: active depth-contradicted contacts require image contact, high mesh support, <=7 cm effective metric distance, observed HaWoR, no nonpenetration conflict, and final post-graph support.
- Ordinary depth contradictions still block active contact: rows without visual-prior override retain `depth_conflict_blocks_active_contact=true` and non-active modes such as `depth_contradicted_noncontact` or `depth_occluded_contact_possible`.
- Local surface-changing support is contact-scoped only. It does not promote full object pose, hidden geometry, object geometry completion, or global occlusion ownership.
- No new category/frame-specific activation branch was found in the reviewed code path; the new local support predicate is schema/evidence driven.
- Global occlusion owner remains unsupported in the artifacts: `occlusion_owner_supported_vars=0` for both cases. Contact depth-order evidence remains local to contact pairs.
- `docs/pipeline_v18.md` accurately documents the active task5 frames and preserved limitations. `docs/pipeline_v19_design_proposal.md` states that V19 is a proposal and that V18 remains the executable script-based pipeline.

## Residual risks / non-blocking concerns

- The review verifies JSON/code invariants and reported render counts, not raw-video visual truth frame by frame. The local surface contact at task5 frame 927 is intentionally weaker than full pose support because predicted projected mesh support is weak; this is acceptable only under the documented contact-only scope.
- The validators are necessary but not sufficient for visual quality. They would not replace manual inspection of the output MP4s; they do, however, catch the specific false-active/depth-veto/final-support/readiness-count failures reviewed here.
- Overlay contact line endpoints are 2D hand/object box centers, while world contact edges use metric nearest endpoints. This is consistent with the current reported counts and docs emphasis on world metric endpoints, but overlay line geometry should not be interpreted as the metric contact location.
