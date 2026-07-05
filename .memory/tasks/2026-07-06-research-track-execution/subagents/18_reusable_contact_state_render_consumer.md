# 18 — Reusable v19 contact-state full-duration render consumer

Task: promote the accepted clip001850 full-duration contact-state render
consumer (`scripts/render_clip001850_v19_contact_state_full_duration.py`)
into a reusable, runtime-style consumer **without touching the many unrelated
dirty files** in the working tree, **without** resurrecting the rejected
still-only QC script (`render_clip001850_v19_contact_state_artifact.py`).

Implemented: `scripts/render_v19_contact_state_full_duration.py` — a new
additive, parameterized script. Validated by re-rendering clip001850 into
`/tmp/clip001850_v19_contact_state_generic_render/` and comparing every
artifact-level gate against the accepted clip-specific output.

No existing scripts edited. No files staged or committed. CPU-only render.

---

## 1. What the generic consumer does (mechanism preserved, I/O parameterized)

The accepted clip-specific script hardcoded the run root, durable root, and the
four runtime-state JSON paths as module constants. The generic script keeps the
accepted rendering mechanism byte-for-byte (same projection, same hatched
observed-body overlay, same world orbit camera, same banner geometry, same
default-uncovered-frames policy, same artifact-level gates) but lifts every
input to a required CLI argument:

`--run-root --contact-state-table --observed-body --pose-report --mano-state
--visible-geometry --rgb-dir --output-root`

plus optional `--case --hand-side --frame-count --fps --overlay/world/sbs
dimensions --review-frames --published-{overlay,world,side-by-side,report}
--trellis-mesh-hash --forbidden-text --keep-frames`.

### Artifact-level gates preserved (the strict contract)

Every gate from the accepted clip001850 consumer is present in the generic
manifest `acceptance_gates`, generalized so they hold for any clip:

| accepted clip-specific gate | generic gate | invariant |
|---|---|---|
| `full_duration_150_frames` | `full_duration_matches_rgb_count` | output frame count == RGB frame count |
| `same_fps_as_published_overlay` | `same_fps_as_published_or_override` | output fps == published or `--fps` |
| `durable_inputs_only` | `durable_inputs_only` | contact table + body not under `/tmp` (runtime guard raises, exit 1) |
| `defaulted_uncovered_frames` (==129) | `uncovered_frames_defaulted` | frames outside the table default to `unresolved_evidence_incomplete` |
| `f32_f36_f45_from_canonical_table` | `per_frame_state_from_table_or_default` | every frame's visible label is table-derived or the documented default |
| `false_gap_penverts_uncertain_removed` | `false_gap_penverts_uncertain_removed` | no frame banner contains `penverts=0`, bare `UNCERTAIN`, or clip-specific `--forbidden-text` |
| `rendered_body_replaces_trellis_by_hash` | `rendered_body_replaces_prior_completion_by_hash` | rendered mesh sha256 != prior-completion hash (when `--trellis-mesh-hash` given) |
| `no_confirmed_contact_in_28_48` | `no_confirmed_contact_without_floor_independent_provenance` | any `confirmed_contact` frame must carry a floor-independent provenance column (motion/GT), never a distance threshold |

The `confirmed_contact` gate is the strictest generalization: the accepted
clip had zero confirmed frames, so the clip gate was a windowed absence check.
The generic gate enforces the actual contract — `confirmed_contact` is
admissible only when the table row carries `confirmed_contact_provenance` /
`motion_coupling_onset` / `gt_contact_provenance`, or a `contact_state_basis`
containing a motion/GT marker. A gap threshold alone never promotes a frame.

### Two intentional, evidence-faithful generalizations

1. **Data-driven banner mechanism text.** The accepted script used a fixed
   per-state sentence. The generic `mechanism_text()` prefers the table row's
   `contact_state_demotion_reasons`, then `contact_state_basis`, falling back
   to the fixed vocabulary only when the row carries neither. For clip001850
   f45 this surfaces the actual demotion reasons ("only one eligible
   keyboard-masked penetrating vertex; temporal IoU below persistent-patch
   requirement") rather than a hardcoded sentence. The fixed vocabulary is a
   closed set of `ego.hoi` contact-state names — not object-category branching
   (forbidden by the methodology rule).
2. **`contact_table_frame_states` supersedes `decisive_frame_states`.** The
   accepted manifest recorded only f32/f36/f45. The generic records every
   contact-table frame's state (a superset); the decisive f32/f36/f45 values
   are identical.

---

## 2. Validation: re-render clip001850 and compare to the accepted artifact

Command run (CPU-only, no model inference):

```
python scripts/render_v19_contact_state_full_duration.py \
  --run-root <v19_runs/20260626_hot3d_clip001850_...> \
  --contact-state-table <durable>/contact_state_table/contact_frame_detail.ndjson \
  --observed-body <durable>/keyboard_body_repair/repaired_observed_contact_body.ply \
  --pose-report <run>/.../v19_rigid_object_pose_graph_report.json \
  --mano-state <run>/.../v18_joint_mano_interval_trajectory_state.json \
  --visible-geometry <run>/.../annotations_v19_visible_geometry.json \
  --rgb-dir <run>/input/raw_frame_manifest/rgb \
  --output-root /tmp/clip001850_v19_contact_state_generic_render \
  --case hot3d_clip001850 --hand-side right \
  --published-{overlay,world,side-by-side} <run>/renders/v19_*.mp4 \
  --published-report <run>/renders/v19_published_runtime/v19_published_render_report.json \
  --trellis-mesh-hash 260b09d1... \
  --forbidden-text "39.2mm" --review-frames "32,36,45"
```

Result: `[render] acceptance_gates ALL PASS`.

### Gate equivalence (accepted clip-specific ↔ generic)

All 7 artifact-level gates match (both True):

| accepted gate | generic gate | match |
|---|---|---|
| full_duration_150_frames | full_duration_matches_rgb_count | ✓ |
| same_fps_as_published_overlay | same_fps_as_published_or_override | ✓ |
| durable_inputs_only | durable_inputs_only | ✓ |
| defaulted_uncovered_frames | uncovered_frames_defaulted | ✓ |
| false_gap_penverts_uncertain_removed | false_gap_penverts_uncertain_removed | ✓ |
| rendered_body_replaces_trellis_by_hash | rendered_body_replaces_prior_completion_by_hash | ✓ |
| no_confirmed_contact_in_28_48 | no_confirmed_contact_without_floor_independent_provenance | ✓ |

### Key-field equivalence

- `contact_state_counts` **identical**: `{unresolved_evidence_incomplete:129,
  pose_unresolved:12, geometry_epoch_contaminated:7, full_frame_depth_leak:1,
  unresolved_incoherent_evidence:1}`
- decisive f32/f36/f45 states **identical**: `geometry_epoch_contaminated` /
  `full_frame_depth_leak` / `unresolved_incoherent_evidence`
- `defaulted_frame_count` **identical**: 129
- `frame_count` **identical**: 150
- `render_consumed_mesh_sha256` **identical**:
  `6d4bcebfb76149a48f9487bceeb4b516de1a887ae1ef901ccc9a952bde49b666`
  (3221 faces, 1726 verts; ≠ trellis `260b09d1…`)
- `output_video_props` **identical**: overlay 960×1038, world 1280×798,
  side_by_side 1920×618, all 150 frames @ 30 fps
- `false_published_claims_removed_from_new_text`: True
- `zero_confirmed_contact_frames`: True

### Pixel diff vs published (nonzero on all decisive frames, as required)

| frame | overlay diff frac | world diff frac | sbs diff frac |
|------:|------------------:|----------------:|--------------:|
| 32 | 0.107 | 0.081 | 0.317 |
| 36 | 0.107 | 0.080 | 0.318 |
| 45 | 0.122 | 0.085 | 0.327 |

These are in the same range as the accepted clip-specific render
(0.12 / 0.08 / 0.33). The small delta vs the accepted output comes entirely
from the data-driven banner mechanism text (generalization #1 above) — a
banner-pixel difference, not a body/hand/geometry difference.

### Guard verification

- `/tmp` durable-input guard: passing a `/tmp` contact table or body raises
  `RuntimeError` and the process exits 1.

---

## 3. Scope discipline

- **One new additive file**: `scripts/render_v19_contact_state_full_duration.py`.
- **Zero edits** to any existing script, including the accepted
  `render_clip001850_v19_contact_state_full_duration.py` (left in place as the
  clip-specific record) and the rejected `render_clip001850_v19_contact_state_artifact.py`
  (confirmed absent — not resurrected).
- **No staging, no commits.** `git status --short` shows the new script as `??`
  (untracked); `git diff --cached --name-only` is empty.
- **No heavy inference / GPU**: CPU-only OpenCV/PIL/trimesh render.
- Outputs went to `/tmp/clip001850_v19_contact_state_generic_render/` (not the
  repo, not the frozen runtime prediction root).

## 4. Residual risks

- The world-view camera is a synthetic orbit viewpoint framed on the scene
  centroid, identical to the accepted consumer (the published world renderer's
  exact viewpoint is not exported as data). This is a visibility/provenance
  styling, not a metric reconstruction; the overlay uses the real per-frame
  metric camera.
- The generic consumer assumes the input JSON field names
  (`rotation_world_from_completed_canonical_matrix`, `translation_world_m`,
  `optimized_vertices_world_sample_m`, `optimized_joints_world_m`,
  `T_world_camera_metric`, `intrinsics_fx_fy_cx_cy`) match the v19 runtime
  convention. Clips using a different schema would need an adapter, not a code
  branch.
- The `confirmed_contact` provenance gate checks for the named
  floor-independent fields/basis markers; a future schema that adds a new
  provenance column would need that column added to
  `CONFIRMED_PROVENANCE_KEYS`.
