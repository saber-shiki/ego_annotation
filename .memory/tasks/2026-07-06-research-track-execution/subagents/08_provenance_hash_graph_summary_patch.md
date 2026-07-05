# Subagent 08 — clip001850 provenance-hash graph-summary patch (KT-6)

## What changed and why

The clip001850 graph-health decision path (`build_clip001850_ego_hoi_graph_summary.py`
→ `build_ego_hoi_sidecar_and_graph_health.py`) derived `cross_solver_geometry_consistency`
from **adapter-authored string literals** (`"observed_depth_surface"`,
`"trellis_completed"`, `"geo_completed_keyboard_trellis_seed42"`) and used
**max-of-max penetration** (`full_observed_surface_penetration_after_solver_max_m.max`
= 0.107 m) as the decisive statistic. Subagent 06 (attack A4) showed this is
gameable: making the three literals equal flips the decision with no change to
the run, and the "penetration and gap coexist" reason is a tautology that fires
whenever two nonzero channels exist.

This patch implements KT-6: the three geometry epoch/source-family fields are
now **derived from measured provenance** (file hashes + measured filter states),
not authored strings; max-of-max is demoted from decisive evidence; and a
missing provenance source routes the decision to `evidence_incomplete` instead
of pretending measured consistency.

## Mechanism (what now produces the decision)

Each geometry role is identified by a **provenance fingerprint** = sha256 of the
canonical JSON of its measured provenance fields:

- **solver (interval barrier)**: `depth_npz` path+sha256 + `{visible_object_mask_gate,
  hand_owned_object_depth_quarantine, surface_eligibility, dense_barrier,
  depth_order_selected_vertex_count}` filter states. The `epoch_id`/`source_family`
  are `epoch_interval_<fp[:12]>` / `interval_depth_barrier+mask..+quar..+elig..#<fp[:8]>`,
  built only from those measured fields. If `depth_npz` does not resolve, the family
  is `None`.
- **contact/NP query**: `surface_mesh_path`+sha256 and `sign_mesh_path`+sha256 +
  `sign_mesh_watertight` + `completed_surface_mesh_watertight`. Family/epoch from
  that fingerprint; `None` if the mesh path does not resolve.
- **render**: the consumed state the publish path stamps from
  (`metrics.interval_state`) path+sha256. Family/epoch from that fingerprint;
  `None` if absent.

Because the families are hash-derived, equal-string tampering in the adapter can
no longer flip the decision, and swapping a file or flipping a filter flag
changes the family — the decision is now falsifiable against the on-disk run.

The scaffold compares the three hash-derived epochs/sources. On the real
clip001850 run the three fingerprints genuinely differ (depth_npz hash ≠
completed-mesh hash ≠ interval-state hash), so the decision is still
`cross_solver_geometry_decoupled` — but now on measured provenance.

## Metric statistic policy (no max-of-max)

The single decisive penetration value is now the **median** across frames
(0.0 for right hand), not max-of-max (0.107). The full distribution
(count/median/p90/p95/mean/max) is carried in an
`observed_surface_penetration_stat_policy` block whose `policy` =
`"median_decisive_not_max_of_max"`. The scaffold's coexistence check
(`observed_penetration > 0 and gap > 0`) now operates on the median; with
median = 0.0 it does not fire (correctly — the barrier is inactive on most
frames; the "10 cm" was one worst vertex in one worst frame). The mismatch
still fires on the provenance-derived epoch/source difference.

## evidence_incomplete branch

The adapter emits a `provenance_completeness` block
(`{complete: bool, missing: [...], present_sources: [...], derivation: ...}`).
The scaffold's `decide_mechanism` now checks this **first** (before the
cross-solver branch): if any of the four required provenance sources
(depth_npz, contact surface_mesh, contact sign_mesh, render consumed state)
is missing, the decision is `evidence_incomplete` with the missing fields
listed, rather than a cross-solver decision asserted on absent provenance.

Verified on a partial run root (contact/NP report + depth_npz + render-state
path all absent): adapter emits `provenance_completeness.complete = False` and
`missing = [solver.depth_npz_sha256, contact.surface_mesh_sha256,
contact.sign_mesh_sha256, render.consumed_state_sha256]`; scaffold decision =
`evidence_incomplete`.

## Files changed

- `scripts/build_clip001850_ego_hoi_graph_summary.py` (new/untracked):
  - added `resolve_provenance_path` (rebases host-mount `/mnt/...` input paths
    onto the provided run_root by matching the run-id segment) and
    `provenance_fingerprint` (sha256 of canonical JSON of measured fields);
  - `extract_interval_mano` now captures `inputs.depth_npz` path+sha256, the
    mask/quarantine/eligibility filter states, the depth-order-selected stat,
    and the full observed-surface-penetration stat block (not just `.max`);
  - `extract_contact_nonpenetration` now captures `inputs.completed_surface_mesh`
    and `inputs.sign_mesh` paths+sha256 + both watertight flags;
  - `extract_render` now captures `metrics.interval_state` consumed-state
    path+sha256;
  - the `cross_solver_geometry_consistency` block's three epoch/source fields
    are derived from the provenance fingerprints (authored literals removed);
    added `solver/contact/render_geometry_provenance` sub-blocks,
    `observed_surface_penetration_stat_policy`, `sign_mesh_watertight`,
    `completed_surface_mesh_watertight`, and `provenance_completeness`;
  - the completed-mesh `geometry_epochs` epoch_id is now derived from the
    contact/NP surface-mesh fingerprint, not the authored seed string;
  - removed now-unused `_stat_max`/`_stat_median` helpers; producer version
    bumped to `clip001850-adapter-2-provenance`.
- `scripts/build_ego_hoi_sidecar_and_graph_health.py` (modified, unstaged):
  - `derive_cross_solver_geometry_consistency` now surfaces the three
    `*_geometry_provenance` sub-blocks, `sign_mesh_watertight`,
    `completed_surface_mesh_watertight`, `observed_surface_penetration_stat_policy`,
    `provenance_complete`, and `provenance_missing`;
  - `decide_mechanism` adds an `evidence_incomplete` branch checked before the
    cross-solver branch, firing when `provenance_complete` is False.

No other files touched. Nothing staged or committed.

## Output (required path)

`/tmp/egohoi_clip001850_provenance/ego_hoi/graph_health.json` (+ `.ndjson`,
`manifest.json`) produced from the real run root
`20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`,
right hand, frames 26–46.

## Real-run decision evidence (graph_health.json)

- `decision`: `cross_solver_geometry_decoupled` (on measured provenance)
- `provenance_complete`: True; `present_sources`: interval_depth_npz,
  contact_surface_mesh, contact_sign_mesh, render_consumed_state
- hash-derived source families:
  - solver: `interval_depth_barrier+maskFalse+quarFalse+eligFalse#aa95b500`
  - contact: `contact_signed_query+signmesh260b09d1+watertightFalse#e4491505`
  - render: `render_consumed_interval_state#58f2b69a`
- `mismatch_reasons`: epoch_id differ; source_family differ (the three
  on-disk hashes genuinely differ)
- `observed_surface_penetration_m` (decisive median) = 0.0;
  stat_policy `median_decisive_not_max_of_max`; max (tail only) = 0.1068
- `published_contact_gap_m` = 0.0392
- `signed_query_candidate_vertex_count` = 4; `sign_mesh_watertight` = False;
  `completed_surface_mesh_watertight` = False
- face provenance: observed 3221 (3.54 %), trellis 86708 (95.40 %)

## Partial-run decision evidence

- `decision`: `evidence_incomplete`
- `provenance_missing`: solver.depth_npz_sha256, contact.surface_mesh_sha256,
  contact.sign_mesh_sha256, render.consumed_state_sha256

## Residual risks / notes

- The decision value (`cross_solver_geometry_decoupled`) is unchanged on the
  real run because the three on-disk sources genuinely differ — the patch
  changes *why* it fires (measured hashes vs authored strings) and makes it
  falsifiable, plus it now exposes the actual filter states (mask/quarantine/
  eligibility all False) that subagent 06 attack A1 flagged.
- The render consumed-state provenance is the interval-state JSON hash, not a
  mesh hash: the v19 publish path stamps the contact-patch gap from
  `metrics.interval_state` and consumes pre-rendered mp4s, so no per-mesh
  render-consumed hash exists in the render report. This is the best available
  render-side provenance and is the honest representation.
- This patch is backing-data/instrumentation only: it makes the graph-health
  decision falsifiable; it does not change any rendered video (subagent 06 A5
  remains open — no renderer consumes ego.hoi rows).
