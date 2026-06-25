# V18 compact multiview object-geometry completion review

MUST-FIX.

## Demonstrated failure: schema blocker is being treated as schema eligibility

`scripts/run_v18_full_pipeline.py:2725-2773`, especially the `schema_eligible` condition at lines 2735-2740, allows completion when either `physical == "rigid"` **or** `surface_change_without_pose_state is True`:

```python
schema_eligible = bool(
    (physical == "rigid" or schema.get("surface_change_without_pose_state") is True)
    and schema.get("requires_part_or_relative_motion_model") is not True
    and schema.get("secondary_deformable_or_surface_component") is not True
    and physical != "deformable"
)
```

That second branch is too broad for a field named `object_geometry_complete` / `object_pose_requirement_met`. In the current accepted output, all 10 true completion rows are `object:obj_tomato`, but its own `physical_state_schema` says:

- `model_physical_state_type: "unknown"`
- `schema_confidence: "low"`
- `schema_blockers: ["no_explicit_primary_physical_state_term", "surface_change_without_pose_model"]`
- `surface_change_without_pose_state: true`

The completion assessment therefore promotes a schema row that explicitly says there is no pose model (`surface_change_without_pose_model`) into `schema_eligible_compact_object=true`, then sets both `object_geometry_complete=true` and `object_pose_requirement_met=true` on frames 281, 849, 852, 925, 926, 928, 936, 937, 938, and 939. The mesh counts are real, but the schema gate is not actually requiring a non-blocked compact-object schema.

Why this breaks the contract: the docs at `docs/pipeline_v18.md:965-968` say completion is only for "schema-eligible compact objects". The current rows are not schema-clean; they are low-confidence unknown/surface-change rows with explicit schema blockers. That is a false object-pose/geometry readiness promotion, even though it is not a centroid/category primitive proxy.

Smallest corrective patch:

1. In `compact_multiview_geometry_completion_assessment`, block completion when `schema_blockers` contains pose/model blockers such as `surface_change_without_pose_model` or `no_explicit_primary_physical_state_term`, and block non-rigid primary states unless a new explicit model/residual-produced compact-geometry eligibility field exists.
2. If the intended tomato path should remain eligible, create an explicit upstream schema/residual field (for example `compact_single_object_geometry_eligible=true` with blockers empty) rather than reusing `surface_change_without_pose_state`, which currently means "surface changed but no pose model".
3. Update `docs/pipeline_v18.md:968` after rerun; under the current schema it should not claim "10 compact multiview tomato object-geometry/object-pose rows" as completed.

## Validator weakness that lets this pass

`scripts/validate_v18_full_pipeline_artifact.py:423-431` checks only the self-reported nested assessment fields. It does not independently recompute schema eligibility from `obj["physical_state_schema"]`, does not reject `schema_blockers`, and does not require an explicit compact-object eligibility field. As a result the current invalid tomato rows pass validation because `assessment["schema_eligible_compact_object"]` is already set true by the over-broad producer predicate.

Smallest validator fix: inside the `if validation.get("object_geometry_complete") is True or validation.get("object_pose_requirement_met") is True:` block, derive `schema = obj.get("physical_state_schema", {})` and require the same corrected eligibility predicate directly from the object row, including no relevant schema blockers and no articulated/deformable/unknown-without-explicit-compact-eligibility state. Also require `assessment.get("blockers")` to be empty when assessment true.

## Non-issues checked

- I found no tomato/frame/trash-specific branch inside `compact_multiview_geometry_completion_assessment`; the problematic promotion is schema-field-driven, not a hardcoded frame list.
- Sparse/invisible-frame protections exist in the producer through current-frame visible depth/silhouette validation plus source-frame/point/hull/Poisson thresholds. The validator still should recompute rather than trust the assessment booleans.
- Current validators preserve root/nested consistency for rows where the root booleans are present.
- Contact/occlusion/render frame counts did not obviously change in this patch: validators pass; report frame counts match all videos; current validated counts remain trash 8 active contacts / task5 2 active contacts / accepted global occlusion owner 0.

## Commands run

- `git log --oneline -3 && git status --short`
- `.venv/bin/python scripts/validate_v18_full_pipeline_artifact.py --report /data2/ego_annotation_outputs/v18_full_pipeline/v18_full_pipeline_report.json && .venv/bin/python scripts/validate_v18_factor_graph.py --root /data2/ego_annotation_outputs/v18_full_pipeline`
- JSON inspection of `/data2/ego_annotation_outputs/v18_full_pipeline/{trash_1050,task5_tomato_960}/annotations_v18_full.json` confirming 0 trash true rows and 10 task5 tomato true rows with low-confidence unknown schema blockers.
