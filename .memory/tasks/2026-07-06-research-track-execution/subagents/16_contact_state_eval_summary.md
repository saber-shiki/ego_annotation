# 16 — clip001850 contact-state evaluation summary

Task: produce a concise machine-readable evaluation summary for clip001850
contact-state resolution, suitable for downstream benchmark/evaluator work,
from existing artifacts only (no inference, no GPU).

Outputs (no staging, no commit):
- `/tmp/clip001850_contact_state_evaluation/summary.json` — schema
  `clip001850.contact_state_evaluation/0.1.0`
- `/tmp/clip001850_contact_state_evaluation/summary.md` — human companion

## Inputs consumed

- `EPISTEMIC.md`, `subagents/11..14`
- `/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/`
  (canonical `contact_frame_detail` rows + summary, keyboard body repair
  summary, hand-depth counterfactual summary, egohoi review manifest,
  artifact manifest)
- `/tmp/clip001850_masked_contact_killtests/` (kill-test rows/summary)
- `scripts/publish_v19_render_artifact.py`, `scripts/build_v19_rigid_render_state.py`
  (grep-verified: neither references `contact_frame_detail` / `contact_state` /
  `ego.hoi`)

## Canonical contact_state distribution (f28–f48, 21 frames)

| state | count | frames |
|---|---:|---|
| `pose_unresolved` | 12 | 28, 29, 37–44, 47, 48 |
| `geometry_epoch_contaminated` | 7 | 30–35, 46 |
| `full_frame_depth_leak` | 1 | 36 |
| `unresolved_incoherent_evidence` | 1 | 45 |
| `confirmed_contact` | 0 | — |

Every frame is accounted for; union == f28–f48. The three review-rendered
frames are f32=`geometry_epoch_contaminated`, f36=`full_frame_depth_leak`,
f45=`unresolved_incoherent_evidence`.

## Geometry repair decision

`hand_depth_bias_dominates_geometry_repair_alone_insufficient`.

Necessary (removes the false 8–10 cm interval penetration from the 95.4 %
TRELLIS-inferred, 8.42×-too-thick completed mesh) but not sufficient (hand
35.6–115.2 mm from the observed-only body, keyboard depth gap −74.7 to
−179.3 mm). A watertight sign mesh is not constructible from existing
observations (singly-observed keyboard top sheet only); closing it would
reintroduce inferred interior faces — the exact artifact the repair deletes.

## Hand-depth counterfactual decision

Refuted. Required along-ray shift 83–202 mm (median 134 mm), 2.8–6.7× the
25–30 mm sigma_clip floor and frame-varying (not a single systematic offset).
Catastrophic reprojection (62–169 px median, 110–285 px max vs 13 px
tolerance); only f36 reaches `contact_candidate`, and even there projection
breaks.

## f45 demotion reason

`contact_candidate_keyboard_masked` → `unresolved_incoherent_evidence`.
One thumb vertex at +5.86 mm (0.20 σ) riding a 393.6 mm depth outlier
(keyboard median 520 mm), temporal IoU 0.15 (< 3-consecutive-frame
requirement), no compact multi-vertex patch. The kill-test script promoted
on a bare ≥1-vertex-within-6 mm presence rule; the policy forbids this.

## Promote to confirmed_contact contract

Floor-independent channel only: motion coupling (object-motion onset
time-locked to hand kinematics; expected weak — keyboard barely moves) or
HOT3D GT (deferred to R8). Signed-distance thresholds, gap-magnitude hand
shifts, and 2D-reprojection changes are explicitly inadmissible. GT-free
ceiling is `unresolved` / `contact_candidate` at most.

## Production render consumes table?

No. Review renderer consumes for f32/f36/f45; production
`publish_v19_render_artifact.py` / `build_v19_rigid_render_state.py` do not
reference the table (grep-verified). Open KT-7 / policy §9 consumer gap:
all canonical rows are currently backing-data-only.

## Test contract

`summary.json` carries `evaluation_schema_for_downstream_tests`
(required top-level fields, required enums, required counts) and an
`assertions` array (A1–A10, all pass). Validated: schema name, all required
fields present, distribution sums to 21, frame union == f28–f48, all 10
assertions pass.

## Validation run

```
missing top-level: []
dist sum: 21 == 21 ? True
frames: [28..48]
match: True
assertions all pass: True
num assertions: 10
```

No files staged; no commits. New files only at `/tmp/...` and this findings
doc at the authoritative subagents path.
