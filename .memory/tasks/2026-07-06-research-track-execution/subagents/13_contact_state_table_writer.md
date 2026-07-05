# 13 — clip001850 canonical ego.hoi contact_state table writer

Slice: HOT3D `clip001850` keyboard, right hand, frames 28-48, run
`20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.

Implements the canonical `ego.hoi.contact_frame_detail/0.1.0` table by consuming
the kill-test rows (subagent 07) under the policy (subagent 10), then rerenders
the f32/f36 review artifact from the canonical table instead of raw kill-test rows.

## What changed

- **New** `scripts/build_clip001850_contact_state_table.py` — reads
  `/tmp/clip001850_masked_contact_killtests/contact_killtests_frame_detail.ndjson`
  (reruns the kill-test script if missing), computes provenance hashes, applies
  the policy's route→contact_state identity mapping, and writes
  `/tmp/clip001850_contact_state_table/{contact_frame_detail.ndjson, summary.json}`.
- **Modified** `scripts/render_clip001850_egohoi_contact_review.py` — now prefers
  the canonical table (`--contact-state-dir`, default
  `/tmp/clip001850_contact_state_table`) when present, unwraps each canonical
  row's `killtest_evidence`, and binds the visible label to the canonical
  `contact_state`. Falls back to raw kill-test rows monotonically when the
  canonical table is absent.

Nothing staged, nothing committed. Both scripts live in the working tree only.

## Canonical row schema (per frame / hand)

Top-level fields (policy §5 provenance, all present on every row):

- `schema`, `case`, `frame_idx`, `hand_side`
- `contact_state` (identity from kill-test route on this clip) +
  `contact_state_basis` + `object_body_provenance` (`geometry_epoch_contaminated` clip-wide)
- `depth_source`: `sha256` of `unidepth_full_frame_depth_v3.npz` + barrier flags
  (`barrier_mask_gate=False`, `barrier_hand_quarantine=False`,
  `barrier_eligibility=False`, `barrier_weight=300000.0`,
  `barrier_enabled=True`, `barrier_target=completed_mesh_observed_face_filtered`)
- `keyboard_mask`: per-frame `sha256` of `sam2_masks/000XXX.png`, or
  `missing_reason` (`proxy_temporal_hold:...` / `no_keyboard_mask_file_and_no_proxy`)
- `contact_surface_mesh`: completed-mesh `sha256`, `watertight=False`,
  `face_count`, AABB extents + ratio to keyboard prior
- `contact_sign_mesh`: `sha256=None`, `watertight=False`, explicit `missing_reason`
  (no observed-face watertight sign mesh built; free space not evaluated; mesh non-watertight)
- `free_space`: `evaluated=False`, `rejected_state=not_evaluated`
- `nearest_surface_face_provenance`: `not_evaluated_per_query` (kill-test does
  not resolve per-query nearest-face provenance; global distribution carried in
  `face_provenance_global`: observed 3.54%, TRELLIS 95.4%, unsupported 1.06%)
- `pose`: `source`, `direct_visible_measurement`, `gap_frames`, `provenance`
  (`direct_visible_measurement` / `interpolated_or_held`)
- `combined_metric_uncertainty_sigma_m=0.028`, `sigma_clip_range_m=[0.025,0.030]`,
  `sigma_clip_basis` (Track J 25-30 mm + 3.40° rotation, systematic)
- `source_gap`: `signed_gap_m` (keyboard-masked+HQ delta median),
  `sigma_clip_m`, `source_gap_z` (computable only on mask-available frames),
  sign convention documented
- `observed_masked_penetrating_count`, `observed_masked_near_count`,
  `raw_penetrating_count`, `survival_fraction`, `penetrating_vertex_ids`,
  `near_band_vertex_ids`, `temporal_iou`
- `render_consumed_mesh_sha256` (= completed-mesh hash) +
  `renderer_consumes_contact_state` (`true` for the f32/f36 review frames, else `false`)
- `interval_solver_published` + `interval_published_penetration_max_m` for
  cross-solver comparison
- `killtest_evidence`: the original kill-test row, verbatim (audit + renderer)

## Required-state verification (from summary.json `required_state_check`)

| required state | frame(s) | verified |
|---|---|---|
| `geometry_epoch_contaminated` | 32 | ✅ |
| `full_frame_depth_leak` | 36 | ✅ |
| `contact_candidate_keyboard_masked` | 45 | ✅ |
| `pose_unresolved` for proxy/interpolated | 28,29,37-44,47,48 | ✅ (12 frames) |

Full distribution: `pose_unresolved=12, geometry_epoch_contaminated=7,
full_frame_depth_leak=1, contact_candidate_keyboard_masked=1`.

Headline numbers (canonical rows):
- f32: `signed_gap_m=-0.144` (hand 14.4 cm in front of keyboard depth), `z=-5.15`,
  masked_pen=0, raw_pen=0 → interval penetration uncorroborated by masked depth.
- f36: `signed_gap_m=-0.081`, `z=-2.89`, masked_pen=0, raw_pen=11, survival=0.0
  → full-frame leak collapses to zero under keyboard mask.
- f45: `signed_gap_m=-0.085`, `z=-3.05`, masked_pen=1, raw_pen=60, survival=0.017
  → one masked keyboard-depth vertex within the soft-tissue band (contact_candidate ceiling).

## Rerender

`render_clip001850_egohoi_contact_review.py` rerun, consuming the canonical
table. Manifest `source_label = "canonical ego.hoi contact_frame_detail"`,
`source_rows = /tmp/clip001850_contact_state_table/contact_frame_detail.ndjson`.
f32 → `geometry_epoch_contaminated`, f36 → `full_frame_depth_leak` in the
review artifact (`/tmp/clip001850_egohoi_contact_review/egohoi_review_f032.jpg`,
`egohoi_review_f036.jpg`).

## Residual risks / honest limits

- `nearest_surface_face_provenance` is `not_evaluated_per_query` — the kill-test
  measures keyboard-masked depth-surface delta along the camera ray, not the
  nearest completed-mesh face per hand vertex. Per-query face provenance would
  require a separate signed-distance face-attribution query (KT-4 rebuild),
  which is the named next blocker, not something this writer can fabricate.
  Global face provenance (95.4% TRELLIS) is carried so the contaminated epoch is
  still evidenced.
- `confirmed_contact` is not produced on any frame — consistent with policy §4
  (distance evidence tops out at `contact_candidate`; the 25-30 mm systematic
  floor swamps the 2-5 mm soft-tissue band). f45 sits at the ceiling.
- `renderer_consumes_contact_state=true` covers only the review renderer for
  f32/f36. The production `publish_v19_render_artifact.py` still does **not**
  consume this table (KT-7 consumer gap, policy §9); the summary records this.
- The canonical `contact_state` is the identity of the kill-test route on this
  clip. If a future KT-4 rebuild / motion-coupling channel promotes states, the
  writer's mapping is the single place to extend.
