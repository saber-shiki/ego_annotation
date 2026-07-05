# R0/R1 scaffold: ego.hoi sidecar + graph-health instrumentation

## What changed

One new untracked file, no edits to any pre-existing (dirty) file, nothing staged:

- `scripts/build_ego_hoi_sidecar_and_graph_health.py` (stdlib-only, `--help` CLI)

The scaffold consumes a v18-style annotation (`--annotation-json`, with top-level
`factor_graph_summary` and per-frame `factor_graph_solution`), a standalone
summary (`--graph-summary`), or an in-memory synthetic summary (`--synthetic`,
no model runtime). It emits `manifest.json` (ego.hoi 0.1.0 sidecar) plus
`graph_health.json` / `graph_health.ndjson` (table skeleton, one row per
solution).

## Why this is instrumentation, not a solver

The script addresses D5 ("the factor graph can be inert while looking
implemented"). It produces the three *independent* signals that separate the
live D5 mechanisms, plus a fixed decision table that routes the next
implementation step. It does not score, gate, or invent measurements.

## Required fields — all present in every graph_health row

| field | source | status |
|---|---|---|
| support_fraction (per family + `support_fraction_min_critical`) | `factor_family_health.supported/declared`, else `null` pending | real where summary provides it; honestly null + `pending` note on v18 `factor_counts`-only summaries |
| active_residual_count (per family + `active_residual_count_total`) | supported count where residual_sum > epsilon | real where available; null + pending otherwise |
| input_hashes | sha256 of annotation_json / graph_summary / synthetic seed | real |
| output_hashes | `graph_output` = sha256 of solved variables across frames; per-family output hashes null pending table writers | graph_output real now; family outputs pending |
| input_output_deltas | `objective_energy_delta` real (from `energy_initial`/`energy_after`); per-family `delta_norm`/`max_abs_delta` null pending differencer | energy delta real; per-family pending (explicitly labelled) |
| stale_dependency_count | count of variables joined to a superseded geometry_epoch_id | real |
| gauge_declaration_count | len(`declared_gauges`) | real |
| geometry_epoch_id lineage | `geometry_epochs[*]` with active flag + superseded ids | real |
| render_state_hash (placeholder) | null until renderer writes it | placeholder by design |

## How the scaffold discriminates the three D5 mechanisms

The decision table runs in causal-precedence order (you cannot diagnose wiring
before confirming measurements exist; you cannot diagnose the render chain
before confirming the graph moved). Verified across all synthetic modes:

| defect mode | signal that fires | decision | next intervention routed |
|---|---|---|---|
| `support` | support_fraction_min_critical = 0.01 < 0.05 | `measurement_support_absent` (M1) | repair measurement extraction; do NOT retune optimizer |
| `inert` | objective energy_delta = 0.0 | `graph_inert` (M2) | repair variable wiring / solver plumbing; check Jacobian + gauge lock |
| `stale_join` | stale_dependency_count = 3 (joins superseded geo_0) | `stale_geometry_join` (M3) | re-bind variables to active geometry_epoch_id |
| `stale_render` / `none` | graph_output_hash set, render_state_hash null | `stale_render_dependency` (render chain) | run renderer from graph output rows |

### The three independent mechanisms map to three independent signals

1. **Measurement-support failure** is isolated by `support_fraction` alone. The
   `support` mode keeps the optimizer moving (energy_delta = 0.5) yet still
   routes to "repair measurement extraction" — proving the support signal is
   not confounded with liveness.
2. **Graph inertness** is isolated by `input_output_delta` (objective energy
   delta today). The `inert` mode has full support (1.0) and active residuals
   (800) yet routes to "repair variable wiring" because energy_delta = 0 —
   proving liveness is not confounded with support.
3. **Stale render** is isolated by comparing `graph_output_hash` to
   `render_state_hash`. The `stale_render` mode has support=1.0, energy_delta=6.0,
   stale_deps=0, yet routes to "repair state-to-render chain" because the
   renderer fingerprint is null. This is the channel that distinguishes a correct
   graph from a stale video.

### Honest limitation surfaced by the scaffold (not a bug)

At R0/R1 no renderer is wired, so `render_state_hash` is null for every run —
including the synthetic "healthy" mode. The scaffold therefore correctly reports
`stale_render_dependency` for `none` rather than falsely claiming
`active_healthy`. The `active_healthy` branch only fires once a real renderer
populates `render_state_hash`. This is the correct epistemic state: the
scaffold refuses to claim health it has not measured. The discriminating value
is that once a renderer *does* populate the hash, the same decision table will
flip to `active_healthy` with no code change.

## Commands run (validation)

```
python3 scripts/build_ego_hoi_sidecar_and_graph_health.py --help
# loop: --synthetic --synthetic-defect {none,support,inert,stale_render,stale_join}
python3 scripts/build_ego_hoi_sidecar_and_graph_health.py --graph-summary <realstyle> ...
```

All outputs parsed as valid JSON; all required fields present; decision table
fires a distinct branch for each of the four defect mechanisms; fallback path on
`factor_counts`-only summaries marks support/residual as `null` + `pending`
rather than faking values.

## Residual risks / next steps

- `input_output_deltas.per_family` and `render_state_hash` are deliberately
  pending: they require (a) a per-variable graph input/output differencer and
  (b) the renderer writing its consumed-state fingerprint. The scaffold exposes
  exactly where each must plug in.
- The decision thresholds (SUPPORT_FRACTION_THRESHOLD=0.05,
  ENERGY_DELTA_INERT_EPS=1e-6) are instrumentation routing thresholds, not
  acceptance gates — explicitly labelled in-code and used only to route the next
  diagnostic. They are not uneducated heuristics grounding physical claims.
- Stale-dependency detection currently keys on `geometry_epoch_id` joins; once
  hand/object id staleness is recorded in summaries, the same counter extends to
  those families with no decision-table change.

## Acceptance

New file only; nothing staged; not committed. Full structured report in the
`acceptance-report` block below.
