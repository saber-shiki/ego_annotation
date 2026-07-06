# 30 — Unknown-preserving object-pose artifact for clip001850 (correction to static-gauge fill)

Branch `yiwen_research`. One new additive script, nothing staged, nothing
committed. CPU-only (numpy / OpenCV / PIL / trimesh). No model inference, no
GPU, no smoothing, no contact promotion, **no pose infill, no static prior in
any pose row**.

Script: `scripts/render_clip001850_unknown_preserving_pose_artifact.py`
Outputs (all under
`/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/unknown_preserving_pose_render/`):
`object_pose_observations.ndjson` (150 rows),
`rest_cluster_reference_for_outlier_rejection.json`,
`visibility_ledger.ndjson` (150 rows),
`v19_overlay.mp4` / `v19_world.mp4` / `v19_side_by_side.mp4` (150 frames @ 30 fps),
`manifest.json`, `review_frames/`.

## 0. What this is and why

The user objected to the static-gauge fill in subagent 28 as a wrong prior: the
static-gauge script wrote a numeric R/t into all 150 pose rows and placed a
held-`T_rest` body on 142 unobserved frames. A static prior is a strong physical
claim that can be laundered into missing-frame pose, contact evidence, or graph
infill. **A missing frame is unknown, not "static at rest".**

This script reverts that fill and replaces it with an unknown-preserving
artifact:

- `object_pose_observations.ndjson` has numeric R/t **ONLY** on the 8 admissible
  measured rest-cluster frames f30-36,f46. The other 142 frames have
  `rotation=translation=None`, `is_measured=is_localized=false`, and a named
  reason (`occluded_out_of_view` / `unresolved_hand_occluded` /
  `rejected_mask_drift` / `mask_unreliable`).
- `T_rest` is retained **only** as `rest_cluster_reference_for_outlier_rejection`,
  an internal reference scalar used to reject f60/75/76/77 as mask-drift
  outliers. It is explicitly **not** named "hypothesis" or "prior"; it states it
  MUST NOT be consumed as latent pose, trajectory fill, contact evidence, graph
  prior, or body placement on unobserved frames. It is not written into any pose
  row and is not consumed by the renderer.
- The renderer draws the object body **ONLY** on measured frames. Unknown frames
  show **no object body** — only a frame-independent red "OBJECT POSE UNKNOWN"
  glyph at a fixed screen corner (not a world point, not a body shape). The hand
  is always drawn because hand state is independent of object pose.
- Contact states on f28-48 are carried through unchanged from
  `contact_frame_detail.ndjson` (byte-identical, sha verified).

## 1. Causal card

- **Artifact defect (numeric + rendered).** The static-gauge pose table writes a
  numeric R/t into all 150 rows and labels a held-`T_rest` body placement on 142
  unobserved frames. That is a pose prior masquerading as a trajectory: the
  renderer draws the body on f78-149 and f0-29 as if the keyboard were localized
  there.
- **Physical variable.** `T_world_object(t)` for every frame. The honest state
  is: numeric R/t ONLY on the 8 admissible measured frames (f30-36,f46); every
  other frame is unknown with a named reason. `T_rest` is retained only as a
  reference scalar for outlier rejection, never as a latent pose.
- **Live mechanisms (rejected).** (M-prior-fill) write T_rest into missing pose
  rows → rejected by the user/parent: launderable into trajectory/contact/graph.
  (M-glyph-as-body) draw a marker at T_rest on unknown frames → rejected: any
  body-shaped mark placed from T_rest is a localization claim.
- **Intervention.** Numeric R/t only on measured frames; unknown frames have
  null pose and a named reason; renderer draws body only on measured frames;
  unknown frames show no body, only a fixed-screen unknown glyph.
- **State change.** f78-149 and f0-29 no longer render a keyboard body. f60/75/
  76/77 render no body. f45 renders no body. Only f30-36,f46 render the measured
  body. The numeric table has null pose on 142 frames.

## 2. The required outputs

**(1) `object_pose_observations.ndjson` — 150 rows; numeric R/t ONLY on f30-36,f46.**
8 measured rows carry `rotation_world_from_completed_canonical_matrix`,
`translation_world_m`, `is_measured=is_localized=true`,
`pose_source=measured_direct_icp_admissible`. 142 unknown rows carry
`rotation=translation=None`, `is_measured=is_localized=false`,
`pose_source=unknown_no_numeric_pose`, and a `reason` field naming the
visibility state. Verified: exactly 8 numeric rows, 142 null rows.

**(2) `rest_cluster_reference_for_outlier_rejection.json` — T_rest as reference ONLY.**
`role=rest_cluster_reference_for_outlier_rejection`. `statement` explicitly says
it is NOT a hypothesis/prior/latent pose and MUST NOT be consumed as trajectory
fill, contact evidence, nonpenetration evidence, graph prior, or body placement.
`must_not_be_consumed_as` lists the six forbidden uses. Integrity gates:
`named_not_hypothesis_not_prior=true`, `not_written_into_any_pose_row=true`,
`not_consumed_by_renderer=true`. `T_rest=[0.581,-0.064,0.345] m`,
translation std `[9.2,15.8,11.5] mm`, rotation observability max 6.65°.
Outlier rejection evidence: f60/75/76/77 displacements from T_rest are
documented (f75 anchor 199.5mm off).

**(3) `visibility_ledger.ndjson` — 150 per-frame states.** Exactly the required
states/counts: `static_observed` (8: f30-36,46), `unresolved_hand_occluded`
(1: f45), `rejected_mask_drift` (4: f60,75,76,77), `mask_unreliable`
(72: f78-149), `occluded_out_of_view` (65: rest). `is_measured/is_localized`
true ONLY for static_observed.

**(4) Full-duration videos.** `v19_overlay.mp4` (960×1026), `v19_world.mp4`
(1280×786), `v19_side_by_side.mp4` (1920×606) — all 150 frames @ 30 fps. Body is
drawn ONLY on f30-36,f46 (measured amber solid). Unknown frames: no body, red
"OBJECT POSE UNKNOWN" glyph + banner. Contact banners unchanged.

## 3. Acceptance evidence (all gates pass)

```
frame_count_is_150                                  : True
numeric_pose_only_on_measured_frames                : True   (numeric_pose_count=8)
no_unobserved_frame_has_numeric_pose                : True   (numeric_on_unknown_frames=[])
no_unobserved_frame_labelled_measured_or_localized  : True   (mislabelled_frames=[])
f90_has_null_pose                                   : True
f120_has_null_pose                                  : True
f90_no_body_drawn                                   : True
f120_no_body_drawn                                  : True
body_drawn_only_on_measured_frames                  : True
no_trest_in_any_missing_pose_row                    : True
contact_states_unchanged                            : True
no_pose_infill_applied                              : True
no_static_prior_in_pose_rows                        : True
render_manifest_consumed_pose_observations          : True
pose_rows_written_count                             : 150
visibility_rows_written_count                       : 150
```

- **150 frames:** all three videos have frame_count == 150 (960×1026 /
  1280×786 / 1920×606).
- **f90/f120 null pose + no body:** f90/f120 rows have
  `rotation=translation=None`, `is_measured=is_localized=false`;
  `body_drawn=false` in the per-frame manifest.
- **No unobserved frame has numeric pose:** exactly 8 numeric rows (f30-36,46);
  `numeric_on_unknown_frames=[]`.
- **No unobserved frame is measured/localized:** `is_measured/is_localized`
  true only on the 8 static_observed; `mislabelled_frames=[]`.
- **Body consumed only from measured observations:** `body_drawn_only_on_measured_frames=true`;
  renderer builds body from `obs_by_frame[fi]` only when `vis_state==static_observed`.
- **No T_rest in any missing pose row:** `no_trest_in_any_missing_pose_row=true`.
- **Contact table byte-identical:** sha256 of
  `contact_frame_detail.ndjson` = `de7ebc27de68ab0b...` (unchanged; 21 rows,
  f32 geometry_epoch_contaminated / f36 full_frame_depth_leak / f45
  unresolved_incoherent_evidence).
- **Render manifest proves body only from measured observations:**
  `render_consumed_pose_source=object_pose_observations.ndjson` with sha;
  per-frame `body_drawn` true only for static_observed.
- **Pixel-level confirmation (world view):** measured frames f30/36/46 have
  ~1.5% amber body pixels; all unknown frames (f0/45/60/90/120) have ~0.017%
  amber — essentially zero, confirming no body on unknown frames.

## 4. Why this is the correct correction (mechanism)

The user's objection is mechanistically correct: writing `T_rest` into missing
pose rows is imputation, not measurement. The rest-cluster statistic licenses
only outlier rejection and cluster description; it does not license a static
hypothesis, a graph prior, or missing-frame pose fill. The honest representation
of a missing observation is null pose + named reason. The renderer must reflect
that: a body drawn on an unobserved frame is a localization claim, regardless of
how it is captioned.

The unknown-preserving artifact keeps `T_rest` available as an internal
reference for outlier rejection (the only use that does not require the value to
be a pose), while removing it from every pose row and every render pixel on
unobserved frames. This preserves the measured/unknown separation the user
asked for and leaves the missing frames for a later factor graph to handle.

## 5. Scope discipline

- **One new additive script.** Zero edits to any existing script (including the
  static-gauge script, which remains as-is). Pre-existing dirty `scripts/*.py`
  on the branch are not mine.
- **No staging, no commits.** Script is untracked (`??`); `git diff --cached`
  empty.
- **No smoothing, no new model, no GPU, no contact promotion, no pose infill, no
  static prior in any pose row.** CPU-only numpy/OpenCV/PIL/trimesh.
- Outputs under the research output namespace (not the repo, not `/tmp`).

## 6. Honest limits / residual risks

- **The unknown-preserving artifact does not add localization.** Direct support
  stays at 8/150 (the measured rest cluster). The 142 unknown frames remain
  unknown; raising support requires new perception/GT evidence, not a prior.
- **`T_rest` still exists on disk as `rest_cluster_reference_for_outlier_rejection`.**
  Although the JSON explicitly forbids its use as pose/prior/infill and it is
  not wired into any pose row or render pixel, a downstream consumer could
  ignore the stated role. The statement field and `must_not_be_consumed_as` list
  make this contract explicit; the parent should enforce it at consumption time.
  If the parent prefers T_rest not exist at all, the reference file can be
  deleted without affecting the pose/render artifact (outlier rejection is
  already baked into the visibility ledger).
- **f45 (unresolved_hand_occluded) has null pose.** It is genuinely ambiguous
  (68mm depth offset, hand-occluded); the render shows no body, only the unknown
  glyph. This is honest.
- **The reviewer cannot see pixels through this text.** Open the review JPGs
  under `.../unknown_preserving_pose_render/review_frames/` (f0,30,32,36,45,46,
  60,75,77,90,120,149) and the mp4s. Clearest evidence: `world_f090.jpg` shows
  the hand with no keyboard body and a red "OBJECT POSE UNKNOWN" tag, vs
  `world_f030.jpg` which shows the measured amber body.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
.venv/bin/python scripts/render_clip001850_unknown_preserving_pose_artifact.py
# -> /data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/unknown_preserving_pose_render/
#    object_pose_observations.ndjson (150 rows; 8 numeric, 142 null)
#    rest_cluster_reference_for_outlier_rejection.json
#    visibility_ledger.ndjson (150 rows)
#    v19_overlay/world/side_by_side.mp4 (150 frames @ 30fps)
#    manifest.json (acceptance_gates ALL PASS)
#    review_frames/*.jpg
```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Implemented the unknown-preserving object-pose artifact as one new additive script scripts/render_clip001850_unknown_preserving_pose_artifact.py (untracked, not staged). Numeric R/t written ONLY on the 8 admissible measured frames f30-36,f46; the other 142 frames have null rotation/translation, is_measured=is_localized=false, and a named reason (occluded_out_of_view/unresolved_hand_occluded/rejected_mask_drift/mask_unreliable). T_rest is retained ONLY as rest_cluster_reference_for_outlier_rejection (explicitly not hypothesis/prior; must_not_be_consumed_as latent pose/trajectory fill/contact evidence/nonpenetration/graph prior/body placement; not written into any pose row; not consumed by renderer). Renderer draws object body ONLY on measured f30-36,f46; unknown frames show no body and a fixed-screen red OBJECT POSE UNKNOWN glyph. Contact states carried through unchanged. All acceptance gates pass: frame_count_is_150=True; numeric_pose_only_on_measured_frames=True (count=8); no_unobserved_frame_has_numeric_pose=True ([]); no_unobserved_frame_labelled_measured_or_localized=True ([]); f90/f120_has_null_pose=True; f90/f120_no_body_drawn=True; body_drawn_only_on_measured_frames=True; no_trest_in_any_missing_pose_row=True; contact_states_unchanged=True; no_pose_infill_applied=True; no_static_prior_in_pose_rows=True. Three videos 150 frames @ 30fps. Contact table sha256 unchanged (de7ebc27de68ab0b). World-view pixel check: measured frames ~1.5% amber body, unknown frames ~0.017% (no body)."
    }
  ],
  "changedFiles": [
    "scripts/render_clip001850_unknown_preserving_pose_artifact.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": ".venv/bin/python scripts/render_clip001850_unknown_preserving_pose_artifact.py",
      "result": "passed",
      "summary": "Wrote .../unknown_preserving_pose_render/{object_pose_observations.ndjson 150 rows (8 numeric, 142 null), rest_cluster_reference_for_outlier_rejection.json, visibility_ledger.ndjson 150 rows, v19_overlay.mp4 150f 960x1026, v19_world.mp4 150f 1280x786, v19_side_by_side.mp4 150f 1920x606, manifest.json, review_frames/}. All acceptance_gates ALL PASS. visibility_counts {occluded_out_of_view:65, static_observed:8, unresolved_hand_occluded:1, rejected_mask_drift:4, mask_unreliable:72}."
    },
    {
      "command": ".venv/bin/python manifest gate + pose-row inspection (f30/45/60/90/120)",
      "result": "passed",
      "summary": "8 numeric rows on f30-36,46; 142 null rows. f90/f120 rotation=translation=None, is_measured=is_localized=false. f45/f60 null pose with reason. f30 measured own pose. numeric_on_unknown_frames=[], mislabelled_frames=[]."
    },
    {
      "command": "sha256sum contact_frame_detail.ndjson",
      "result": "passed",
      "summary": "Contact table sha256 de7ebc27de68ab0b... unchanged (21 rows f28-48; f32 geometry_epoch_contaminated, f36 full_frame_depth_leak, f45 unresolved_incoherent_evidence)."
    },
    {
      "command": ".venv/bin/python world-view amber pixel fraction check (measured vs unknown)",
      "result": "passed",
      "summary": "World view: measured f30/36/46 ~1.5% amber body pixels; unknown f0/45/60/90/120 ~0.017% amber (no body). Confirms body drawn only on measured frames."
    },
    {
      "command": "git status --short scripts/render_clip001850_unknown_preserving_pose_artifact.py && git diff --cached --name-only && git branch --show-current",
      "result": "passed",
      "summary": "Script untracked (??); nothing staged (git diff --cached empty); on branch yiwen_research."
    }
  ],
  "validationOutput": [
    "object_pose_observations.ndjson: 150 rows; exactly 8 numeric R/t on f30-36,46; 142 null pose rows with named reasons. f90/f120 null pose, is_measured=is_localized=false.",
    "rest_cluster_reference_for_outlier_rejection.json: T_rest=[0.581,-0.064,0.345]; role=rest_cluster_reference_for_outlier_rejection; named_not_hypothesis_not_prior=true; not_written_into_any_pose_row=true; not_consumed_by_renderer=true; must_not_be_consumed_as lists latent pose/trajectory fill/contact/nonpenetration/graph prior/body placement.",
    "visibility_ledger.ndjson: 150 rows; static_observed 8 (f30-36,46), unresolved_hand_occluded 1 (f45), rejected_mask_drift 4 (f60,75,76,77), mask_unreliable 72 (f78-149), occluded_out_of_view 65; is_measured/is_localized true only for static_observed.",
    "Render: body drawn ONLY on f30-36,f46; unknown frames show no body + red OBJECT POSE UNKNOWN glyph. World-view pixel check confirms (~1.5% amber on measured, ~0.017% on unknown).",
    "Contact table sha256 unchanged (de7ebc27de68ab0b); 21 rows f28-48; f32/f36/f45 states preserved.",
    "Three full-duration videos 150 frames @ 30fps each (overlay/world/side_by_side).",
    "Manifest pose_policy: pose_infill_applied=false, static_prior_written_into_pose_rows=false, no_trest_in_any_missing_pose_row=true, numeric_pose_only_on_measured_frames=true, missing_frames_are_unknown=true, rest_cluster_reference_not_hypothesis_not_prior=true."
  ],
  "residualRisks": [
    "Unknown-preserving artifact does not add localization: direct support stays 8/150 (measured rest cluster). The 142 unknown frames remain unknown; raising support requires new perception/GT evidence, not a prior.",
    "T_rest still exists on disk as rest_cluster_reference_for_outlier_rejection.json. Although the JSON explicitly forbids its use as pose/prior/infill and it is not wired into any pose row or render pixel, a downstream consumer could ignore the stated role. If the parent prefers T_rest not exist at all, deleting the reference file does not affect the pose/render artifact (outlier rejection is already baked into the visibility ledger).",
    "f45 (unresolved_hand_occluded) has null pose and no body; genuinely ambiguous (68mm depth offset, hand-occluded). Honest but means f45 contributes no object-pose evidence.",
    "Reviewer must open review_frames/*.jpg (f0,30,32,36,45,46,60,75,77,90,120,149) and the mp4s to consume the artifact visually; clearest evidence is world_f090.jpg (hand, no keyboard body, red OBJECT POSE UNKNOWN tag) vs world_f030.jpg (measured amber body)."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new untracked additive script scripts/render_clip001850_unknown_preserving_pose_artifact.py (~560 lines). No existing scripts edited. Outputs under the research output namespace (object_pose_observations.ndjson 150 rows with numeric R/t only on f30-36,46; rest_cluster_reference_for_outlier_rejection.json; visibility_ledger.ndjson 150 rows; v19_overlay/world/side_by_side.mp4 150 frames each; manifest.json; review_frames/). Nothing staged, nothing committed.",
  "reviewFindings": [
    "no blockers: all acceptance gates pass (150 frames; numeric pose only on 8 measured f30-36,46; no unobserved frame has numeric pose / is_measured / is_localized; f90/f120 null pose + no body; body drawn only on measured; no T_rest in any missing pose row; contact table sha unchanged; no pose infill; no static prior in pose rows).",
    "finding: the intervention is the user-directed unknown-preserving correction to the static-gauge fill; T_rest is retained only as rest_cluster_reference_for_outlier_rejection (explicitly not hypothesis/prior; not in any pose row; not consumed by renderer).",
    "finding: direct support stays 8/150 -- the honest measured ceiling; the 142 unknown frames are null-pose with named reasons, left for a later factor graph, not filled with a static prior."
  ],
  "manualNotes": "Core result for the parent: the clip001850 keyboard unknown-preserving object-pose artifact is implemented and rendered. object_pose_observations.ndjson has numeric R/t ONLY on the 8 admissible measured frames f30-36,f46; all other 142 frames have null pose, is_measured=is_localized=false, and a named reason (occluded_out_of_view/unresolved_hand_occluded/rejected_mask_drift/mask_unreliable). T_rest is kept only as rest_cluster_reference_for_outlier_rejection (to reject f60/75/76/77), explicitly NOT a hypothesis/prior, not in any pose row, not consumed by the renderer. The renderer draws the object body ONLY on measured frames; unknown frames show no body and a fixed-screen red OBJECT POSE UNKNOWN glyph. Contact states on f28-48 are preserved unchanged (sha verified). Open .../unknown_preserving_pose_render/world_f090.jpg (hand, no keyboard body, OBJECT POSE UNKNOWN tag) vs world_f030.jpg (measured amber body) to see the correction. If the parent wants T_rest removed entirely, delete rest_cluster_reference_for_outlier_rejection.json; it does not affect the pose/render artifact."
}
```
