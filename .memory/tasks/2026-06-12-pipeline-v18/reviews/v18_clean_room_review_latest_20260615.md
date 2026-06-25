PASS — bounded clean-room adversarial review of V18 HEAD 62a0824.

Scope inspected: latest committed code, final artifact under `/data2/ego_annotation_outputs/v18_full_pipeline`, latest run log, validators, and `docs/pipeline_v18.md`, restricted to the five listed risks.

## Findings

1. **No active contact without post-graph final support path found.**
   - Code: `scripts/run_v18_full_pipeline.py::attach_contact_physical_modes` computes `support_paths`, `near_supported`, and `final_support_allows_active`; it records `post_graph_final_support_paths_present` / `post_graph_final_support_allows_active_contact`, demotes any existing `estimate=true` when final support is absent, and only sets `active_physical_contact` when `final_support_allows_active` is true (lines 2151-2169).
   - Artifact: task5 active rows are only frames 926 and 927, both with `post_graph_final_support_allows_active_contact=true`; frame 926 has `surface_changing_visible_depth_silhouette_pose`, frame 927 has `surface_changing_local_visible_contact_surface`.

2. **Local surface support is contact-scoped and does not claim object pose/hidden geometry completion.**
   - Code: `final_contact_support_paths_for_mode` only appends `surface_changing_local_visible_contact_surface` under visual prior + mask/residual/distance/nonpenetration conditions and writes scope `contact_support_only_not_full_object_pose_or_hidden_geometry_completion` (lines 2047-2067).
   - Artifact: recursive scan of completion-like keys (`object_geometry_complete`, `object_pose_requirement_met`, `hidden_geometry_reconstructed`, `complete_object_pose_ready`, `canonical_mesh_ready`) found zero `true` values in both final annotation JSONs. Task5 frame 927 tomato validation/reconstruction keeps `surface_changing_compact_visible_pose_supported=false`, `object_geometry_complete=false`, and `object_pose_requirement_met=false`.

3. **Validators still reject unconstrained active depth contradictions when the configured validator set is run.**
   - Code: `validate_v18_full_pipeline_artifact.py` requires active depth-contradicted contacts to have no blocking depth conflict, explicit visual-prior override, supported prior, effective distance <= 7 cm, mesh support >= 0.90, and no nonpenetration conflict (lines 282-289). For active surface-changing contacts it also requires compact or local final support; local support must have visual prior, observed mask fraction >= 0.80, residual <= 0.075 m, and effective distance <= 0.07 m (lines 292-304).
   - Code: `validate_v18_factor_graph.py` additionally checks active contacts for geometry evidence, physical support, visual-prior override on depth contradiction, and no nonpenetration conflict (lines 161-169).
   - Runtime verification passed both configured validators.

4. **Task5 active rows are bounded and specific, not broad false activations.**
   - Artifact query: `task5_tomato_960` has 2,298 contact-switch rows with modes `{separated_or_unresolved_noncontact: 1774, depth_contradicted_noncontact: 520, active_physical_contact: 2, raw_contact_proposal_without_final_validated_physical_support: 1, supported_near_noncontact: 1}`.
   - The only active rows are `(926, left, object:obj_tomato)` and `(927, left, object:obj_tomato)`. Both are depth-contradicted but have `depth_conflict_blocks_active_contact=false`, visual-prior override, observed HaWoR, no nonpenetration conflict, high mesh support, and close effective metric distance. The remaining 520 depth-contradicted task5 rows stay non-active.

5. **Docs are not misleading for the reviewed limitations.**
   - `docs/pipeline_v18.md` lines 955-967 state that local surface-changing support is contact-only, does not complete hidden geometry or full object pose, unsupported proposals are not drawn as solved/possible contact, and global occlusion owner remains unsupported with object geometry/pose completion false.

## Residual risks / non-blocking caveats

- The factor-graph validator alone is weaker than the full artifact validator for final support-path thresholds; the configured runtime validation runs both, and the full validator carries the strict local-surface/depth-contradiction checks.
- Some top-level object rows omit root `object_geometry_complete` / `object_pose_requirement_met` fields rather than storing explicit `false`; nested validation and reconstruction fields are false and no completion-like key is true. This is absence, not an overclaim, but consumers should use the validated nested fields/counts.
