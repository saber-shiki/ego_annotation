# Contact-state decision policy for clip001850 — replaces "wire observed-surface penetration in"

Slice: HOT3D `clip001850` keyboard, right hand, frames 26–46, run
`/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.
Consumes kill-tests KT-1..KT-7 (defined in `subagents/06_reconciliation_attack.md`).
Read-only formulation. No edits made.

---

## 0. The two physical facts that kill "wire it in" and set the ceiling

**F1 — the "observed keyboard surface" is not the keyboard.** The interval
solver's observed-surface barrier queries `depth_slam/unidepth_full_frame/…_v3.npz`
shape `(150,1408,1408)` — the full-frame monocular depth (table + hand + background) —
with `visible_object_mask_gate_enabled=False`, `hand_owned_object_depth_quarantine_enabled=False`,
`surface_eligibility_factor_enabled=False`, and the one physically-correct term
`visible_surface_depth_order_selected_vertex_count=0` on all 150 frames.
`.memory/project/hand_metric_mechanisms.md` independently rules UniDepth at hand
pixels **26–45 mm abs-median, "too noisy"**. Penetration into this map is not
penetration into an object. The raw `.max` (0.107 m, median 0.0) is one worst
vertex; its hard barrier (weight 3e5) already dragged the right hand up to
**10.8 cm** (`translation_delta_norm_m` p95 7.2 cm, max 10.8 cm). Adding it as a
graph factor optimizes the MANO hand into the table plane and its own depth.

**F2 — clip001850's metric floor swamps the soft-tissue band, so signed distance
cannot CONFIRM contact.** Track J decomposition (`hand_metric_mechanisms.md`):
clip001850 lateral floor **25–30 mm** (binding constraint), plus range-dependent
along-ray bias, plus a **real 3.40° camera-frame rotation** (≈15–25 mm at 0.35 m),
even/odd-stable and identical across prediction variants — i.e. **systematic bias,
not zero-mean noise, so temporal averaging does not reduce it.** Touch-vs-no-touch
requires resolving the finger-pad soft-tissue band (~2–5 mm). Because
σ_clip ≈ 30 mm ≫ 5 mm, a 1σ "compatible" gap band spans ~0–30 mm — it cannot
distinguish a hand pressing keys from a hand hovering 3 cm above them.
**Consequence: source-gap / signed-distance evidence tops out at `contact_candidate`;
`confirmed_contact` is physically unreachable from distance alone on this clip and
requires a channel independent of the metric floor (temporal motion-coupling, or
R8 GT).**

The old route ("wire the observed-surface penetration in") assumed the signal was
admissible and metric-resolving. Both assumptions are false. The policy below makes
the penetration an **input to an admissibility test**, not a factor, and caps its
promotion at `contact_candidate`.

---

## 1. State set (per hand / per frame, table `contact_frame_detail`)

| `contact_state` | meaning | gate that produces it | KT |
|---|---|---|---|
| `unresolved_evidence_incomplete` | a required provenance hash or the KT-1 admissible statistic is absent; contact undecidable | provenance gate G0 fails | KT-6 |
| `unresolved_full_frame_depth_leak` | observed-surface penetration collapses to ~0 once restricted to keyboard-mask pixels with hand-owned depth quarantined → the signal was table/hand, not keyboard | observed channel dies by leak (G_obs) | KT-1 |
| `unresolved_incoherent_evidence` | surviving penetrating/near set is temporally scattered and/or anatomically diffuse and/or not localizable to a single surface | coherence/localization fails (G_obs) | KT-2, KT-3 |
| `unresolved_pose_gap` | contact can only be tested against posed geometry, but the object pose at this frame is interpolated/held across a graph gap (not `direct_visible_measurement`) | posed-geometry fallback under bad pose | KT-5 |
| `geometry_epoch_contaminated` | contact can only be tested against the completed mesh, and that mesh is not contact-eligible (free space uncarved / non-watertight sign mesh / queried face is TRELLIS-inferred) | posed-geometry fallback under contaminated body | KT-4 |
| `no_contact_supported` | admissible, coherent, localized on an observed surface; source-gap exceeds 3σ_clip | z-decision on admissible channel | KT-1/2/3 |
| `contact_candidate` | admissible, coherent, localized on an observed surface; source-gap within 3σ_clip (compatible ≤1σ, uncertain 1–3σ). **Ceiling for distance-only evidence on this clip (F2).** | z-decision on admissible channel | KT-1/2/3 |
| `confirmed_contact` | `contact_candidate` **plus** an orthogonal, floor-independent corroboration (object-motion onset locked to hand kinematics, or R8 GT) | orthogonal channel | (motion / GT) |

Object-body provenance is a **separate per-row field** (`object_body_provenance ∈
{observed_carved, geometry_epoch_contaminated}`), not the contact_state; on this
clip it is `geometry_epoch_contaminated` for every frame (95.4 % TRELLIS,
`free_space_rejection_state=not_evaluated`) and drives the world-panel body styling
regardless of the contact_state.

---

## 2. Decision procedure (per hand, per frame)

Not a flat cascade. Evaluate **channel admissibility first**, decide on the best
admissible channel, and fall to a named `unresolved_*` reason only when no channel
is admissible. On clip001850 the completed-mesh channel is contaminated clip-wide
(KT-4 fails globally), so the observed-masked channel is the sole contact channel
except where the hand fully occludes the keyboard.

```
decide_contact_state(frame f, hand h):

  # G0 — provenance completeness (KT-6). No decision without knowing what was queried.
  if any_required_provenance_hash_missing(f, h)  # see §5
     or kt1_admissible_stat_absent(f, h):
     return unresolved_evidence_incomplete

  # --- Channel A: observed-masked depth (pose-independent; the trusted channel) ---
  A = kt1_observed_masked_quarantined(f, h)      # KT-1
  if A.available:                                # keyboard-mask pixels exist under hand
     if A.penetrating_count == 0 and A.near_count == 0:
        # nothing survives mask+quarantine that was present pre-mask
        return unresolved_full_frame_depth_leak      # KT-1 leak
     if not kt2_coherent(f, h) or not kt3_localized_single_surface(f, h):
        return unresolved_incoherent_evidence        # KT-2 / KT-3
     z = source_gap_z(A.signed_m, sigma_clip)        # §4
     if z > 3:  return no_contact_supported
     if motion_coupling_corroborates(f, h) and z <= 1:
        return confirmed_contact                      # orthogonal channel (F2)
     return contact_candidate                         # distance-only ceiling

  # --- Channel A unavailable (hand fully occludes keyboard). Fallback: completed mesh ---
  # precedence: pose validity precedes geometry validity precedes contact.
  if not pose_is_direct_visible(f):                # KT-5
     return unresolved_pose_gap
  if not kt4_completed_mesh_contact_eligible():    # KT-4 (clip-wide: fails)
     return geometry_epoch_contaminated
  # (only reached if a carved observed-face watertight sign mesh exists under a fit pose)
  z = source_gap_z(completed_signed_m, sigma_clip)
  return no_contact_supported if z > 3 else contact_candidate
```

**What this yields for clip001850 right 26–46 today (before any KT run):** every
frame → `unresolved_evidence_incomplete` (KT-1 admissible statistic + provenance
hashes not yet produced). That is the correct honest replacement for the premature
wire-in. After KT-1/2/3/5 run, the projected assignment (from on-disk numbers) is:

- f26–29 (`nearest_visible_pose_hold`), f37–44 (`missing_initial_graph_pose`,
  9-frame gap 36→45, raw count spikes f38:130/f39:190 grown during solve): the
  observed channel, if keyboard pixels survive under the hand, is likely
  `unresolved_full_frame_depth_leak` or `unresolved_incoherent_evidence`; where the
  keyboard is occluded, `unresolved_pose_gap`.
- f30–36, 45–46 (`direct_visible_pose_observation`), f31–33 typing cluster
  (KT-2 IoU 0.37/0.62, compact patch): candidates for `contact_candidate` **iff**
  KT-1 survival is non-empty and KT-3 localizes on the keyboard-masked surface.
- No frame reaches `confirmed_contact` without motion-coupling (deferred) or GT (R8).

---

## 3. Promotion / demotion — exact evidence per state

- **→ `unresolved_evidence_incomplete`**: any of §5's hashes null, or KT-1 not run.
  **Promoted out** only when all provenance is present and the KT-1 admissible
  statistic exists. Never skip to a decided state on partial provenance.
- **→ `unresolved_full_frame_depth_leak`**: pre-mask penetration existed but
  KT-1 keyboard-masked + hand-quarantined penetrating_count = 0 AND near_count = 0.
  **Promoted to `contact_candidate`** only if ≥1 vertex survives on keyboard-masked,
  hand-quarantined depth (survival is the discriminator — a keyboard-contact vertex
  projects onto keyboard pixels; a table/hand vertex does not). **Demoted from any
  decided state back to leak** if a provenance re-check shows the barrier ran on
  unmasked depth.
- **→ `unresolved_incoherent_evidence`**: KT-2 mean pairwise temporal IoU of
  penetrating vertex-id sets near 0, or count non-monotone (0→190→0 flicker / growth
  during solve), or KT-3 fingertips sit within the soft-tissue band of >1 surface or
  of none. **Promoted to candidate** when the surviving set is a compact
  fingertip/palm patch persistent across ≥3 consecutive frames AND KT-3 localizes it
  to exactly one surface. **Demoted** if the set scatters or splits across surfaces.
- **→ `unresolved_pose_gap`**: observed channel unavailable AND
  `direct_visible_measurement=false` (interpolated/held). **Promoted** when a direct
  visible pose fit becomes available for the frame (more visible-pose frames / R4).
  Never dismiss the pose mechanism using the pose-independent dense barrier (that was
  card D-c's error).
- **→ `geometry_epoch_contaminated`**: observed channel unavailable, pose valid, but
  completed mesh fails KT-4 (free space `not_evaluated`, `sign_mesh_watertight=false`,
  or the nearest queried face is `trellis_inferred_hidden_surface`/`unsupported_uncertain`).
  **Promoted** only after KT-4 rebuild: free-space carving run + observed-face-only
  watertight sign mesh + the signed query's nearest face is an observed face. This
  state also fixes the invalid "0 penetration = no contact" (`penetrating_vertex_count=0`
  is zero-by-construction on a non-watertight 95.4 %-TRELLIS mesh, not physical
  clearance).
- **→ `no_contact_supported`**: admissible + coherent + localized, source-gap z > 3σ_clip
  (maps to existing `contact_unlikely_source_gap_exceeds_3sigma`). **Demoted to candidate**
  if z drops ≤ 3 after a pose/geometry repair.
- **→ `contact_candidate`**: admissible + coherent + localized, z ≤ 3σ_clip. Split for
  rendering into ≤1σ ("compatible") vs 1–3σ ("uncertain"). This is the **hard ceiling**
  for distance-only evidence on clip001850 (F2). **Promoted to `confirmed_contact`**
  only by the orthogonal channel below; **demoted to unresolved** if any admissibility
  gate later fails.
- **→ `confirmed_contact`**: `contact_candidate` (z ≤ 1) AND object-motion onset is
  time-locked to hand kinematics (a correlation/timing signal, floor-independent), OR
  R8 GT adjudication. **Demoted to candidate** if motion-coupling weakens or GT is
  withdrawn. No signed-distance threshold promotes to this state.

---

## 4. Thresholds — physical basis or explicitly none

- **σ_clip (combined metric uncertainty) = 25–30 mm lateral + along-ray bias + 3.40° rotation**
  for clip001850. Basis: Track J oracle decomposition and Track M rotation calibration
  in `hand_metric_mechanisms.md`, measured, per-clip, not tuned. This is the ONLY scale
  used to bin source-gap. Reuse the existing `source_gap_z` = signed_gap / σ_clip already
  computed by `build_v19_source_gap_contact_likelihood_state.py`.
- **z ≤ 1 → compatible; 1 < z ≤ 3 → uncertain; z > 3 → unlikely.** Basis: standard
  Gaussian compatibility at the measured σ; identical to the existing
  `contact_likelihood_state_counts` bins. Not a new heuristic — the existing renderer
  contract.
- **`confirmed_contact`: NO signed-distance threshold.** Physically unresolvable from
  distance alone because σ_clip (30 mm) ≫ soft-tissue band (2–5 mm) and the floor is
  systematic bias (not reducible by averaging). Gated only on the orthogonal channel.
- **KT-1 leak discriminator: NO tuned threshold.** The test is survival: does ≥1
  penetrating/near vertex remain after restricting to keyboard-mask pixels and removing
  hand-owned depth. Count > 0 vs 0. Report `survival_fraction = masked_count / raw_count`
  as evidence, but the gate is presence, not a magic cutoff.
- **KT-2 coherence: report distribution; routing threshold flagged, not physical.**
  Report mean temporal IoU and count-monotonicity. If automation needs a cutoff, use
  IoU ≥ 0.5 over ≥3 consecutive frames (basis: >50% set overlap frame-to-frame at 30 fps
  for a quasi-static contact patch) — labelled a routing threshold, never a physical
  contact claim.
- **KT-4 contact-eligibility: per-query provenance, NO global fraction cutoff.** The
  signed distance is admissible only if the nearest surface face used is an OBSERVED face
  (per-face provenance), the sign mesh is watertight, and free space has been evaluated.
  Avoids a tuned "% observed" gate.
- **KT-5 pose: boolean, no threshold.** `direct_visible_measurement` and `gap_frames`
  are on disk (`v19_rigid_object_pose_graph_report.json`).
- **Nonpenetration factor (if promoted): SOFT, scaled to σ_clip, never a hard barrier.**
  Basis: a hard 3e5 barrier against a 30 mm-noisy surface is the mechanism that dragged
  the hand 10.8 cm. Any NP factor sourced from observed depth uses a soft cost with
  scale σ_clip and only over KT-1-admissible, KT-2-coherent vertices.

---

## 5. Provenance requirements (KT-6) — mandatory columns on every `contact_frame_detail` row

Missing any of these ⇒ `unresolved_evidence_incomplete` (the scaffold must route to
`evidence_incomplete`, never fire the cross-solver decision on absent provenance):

- `depth_source_sha256` (of `unidepth_full_frame_depth_v3.npz`) + `barrier_mask_gate`,
  `barrier_hand_quarantine`, `barrier_eligibility` (the three flags, currently all `false`)
- `keyboard_mask_sha256` (per-frame SAM2 mask from
  `measurements/object_tracks/sam2_agent_points/keyboard/sam2/sam2_masks` — confirmed
  present, so KT-1 is runnable)
- `contact_surface_mesh_sha256`, `contact_sign_mesh_sha256`, `sign_mesh_watertight`,
  `free_space_evaluated`, `free_space_rejected_fraction`
- `nearest_surface_face_provenance ∈ {observed, unsupported_uncertain, trellis}` for the
  face used in the signed distance (per-query, not global)
- `pose_source`, `direct_visible_measurement`, `gap_frames`
- `combined_metric_uncertainty_sigma_m` (the 25–30 mm floor) + `source_gap_z`
- `render_consumed_mesh_sha256` + `renderer_consumes_contact_state` (bool, KT-7)
- `observed_masked_penetrating_count`, `observed_masked_near_count`, `raw_penetrating_count`,
  `survival_fraction`, `penetrating_vertex_ids`, `temporal_iou`

The three cross-solver source-family/epoch fields the scaffold compares must be **derived
from these hashes**, not authored strings (KT-6 kills the current adapter's hard-coded
`"observed_depth_surface"`/`"trellis_completed"` literals and the tautological
"penetration>0 AND gap>0 coexist" reason).

---

## 6. Renderer label mapping (extend `publish_v19_render_artifact.py:contact_semantics()`; monotonic)

`contact_semantics()` already emits, at the combined-metric-uncertainty scale, and
never claims ownership/NP. Map states onto it and extend only the unresolved family:

| `contact_state` | render wording | styling |
|---|---|---|
| `confirmed_contact` | `contact — motion-corroborated; NP unresolved` (or `— GT` at R8) | solid contact mark; hand mesh solid |
| `contact_candidate` (z≤1) | existing `near-contact compatible; ownership/NP unresolved` | dashed/amber contact mark |
| `contact_candidate` (1<z≤3) | existing `near-contact uncertain; ownership/NP unresolved` | dotted/amber |
| `no_contact_supported` | existing `contact unlikely by source gap` | no contact mark |
| `unresolved_full_frame_depth_leak` | `unresolved — depth not object-localized` | hatched "unresolved" tag; **suppress the old "gap 39mm / penverts=0 / closed"** (that number came from the contaminated completed mesh) |
| `unresolved_pose_gap` | `unresolved — object pose interpolated (Nf gap)` | hatched; show gap length |
| `geometry_epoch_contaminated` | `unresolved — object body 95% inferred, not carved` | hatched; **world-panel body drawn as uncertain (hatched), not solid green** — fixes the current "diffuse oversized green TRELLIS cloud rendered as accepted body" (`object_body_provenance` drives this on every frame) |
| `unresolved_incoherent_evidence` | `unresolved — contact evidence incoherent` | hatched |
| `unresolved_evidence_incomplete` | `unresolved — provenance incomplete` | hatched |

The single vague banner `UNCERTAIN = not accepted contact closure` is replaced by the
specific machine-readable reason. No prior valid wording is removed (monotonic).

---

## 7. Scaffold patch (`build_ego_hoi_sidecar_and_graph_health.py` + adapter) — KT-6

`decide_mechanism` currently fires `cross_solver_geometry_decoupled` first on
`mismatch_reasons`, one of which is the tautology `observed_penetration>0 AND
published_gap>0` (two channels on different surfaces are always both nonzero), and the
adapter hard-codes the three source-family strings. Patch:

1. `derive_cross_solver_geometry_consistency`: drop the coexistence tautology; require
   the three epoch/source fields to be **hash-derived** (§5). If any provenance hash is
   absent → return a `provenance_incomplete=true` block.
2. `decide_mechanism`: add branch **before** `cross_solver_geometry_decoupled`:
   `if cross_solver_geometry.provenance_incomplete: decision = "evidence_incomplete"`.
   Keep `cross_solver_geometry_decoupled` only when hash-derived epochs genuinely differ.
3. Forbid `.max`-of-`.max` as `observed_surface_penetration_m`; require the KT-1
   keyboard-masked, hand-quarantined statistic as the input.
4. `graph_health` remains a **solver-level** diagnosis (which channel is admissible); the
   per-frame `contact_state` is the **new `contact_frame_detail`** table driven by §2.
   The graph decision routes which channel the per-frame policy may consume.

---

## 8. Next artifact-changing patch (the root blocker, not more schema)

The hardest essential blocker is **channel admissibility**: until KT-1 runs we do not
know whether any real keyboard-contact signal exists, so no rendered state can be
anything but `unresolved_evidence_incomplete`. One integrated patch:

1. **KT-1 script** (CPU-only, new): for right f26–46, restrict the observed-surface
   penetration to keyboard SAM2-mask pixels, quarantine hand-owned depth, and separately
   test a table-plane fit. Emit per-frame `observed_masked_penetrating_count`,
   `survival_fraction`, `penetrating_vertex_ids`. Reuses the already-present depth NPZ +
   keyboard masks + `optimized_vertices_world_sample_m`/`optimized_vertices_sample_ids`.
2. **`contact_frame_detail` writer** applying §2 with the §5 provenance columns → real
   per-frame `contact_state` for right 26–46 (default `unresolved_evidence_incomplete`,
   promoting where KT-1/2/3 pass).
3. **KT-7 consumer** (the delivery gate): make `build_v19_rigid_render_state.py` source
   contact from `contact_frame_detail`, and `publish_v19_render_artifact.py:contact_semantics()`
   read the per-frame `contact_state` (§6 mapping). Re-render right f32/f36 and pixel/label-diff
   the published stills. A nonzero diff (leak frames lose the false "gap 39mm"; contaminated
   body drawn hatched) is the acceptance evidence. Set `artifact_change_verified` from the diff.

Do **not** add the observed-surface penetration as a hard graph factor. If, after
KT-1/2/3 promotion, a nonpenetration factor is added, it is soft and σ_clip-scaled over
admissible vertices only (§4), and the interval solver's prior in-window drags (up to
10.8 cm) driven by the unmasked barrier must be reverted, not preserved.

---

## 9. Residual risks

- **KT-7 has no consumer today** (confirmed: only the three ego.hoi scaffold scripts
  reference the rows; `build_v19_rigid_render_state.py` / `publish_v19_render_artifact.py`
  do not). Until step 8.3 lands, every contact_state is backing-data-only, not a rendered
  annotation change — `artifact_change_verified=false`.
- **`confirmed_contact` is unreachable GT-free** on this clip until motion-coupling is
  implemented; expect the whole window to sit at `contact_candidate` / `unresolved_*`.
  HOT3D GT exists for this clip and is the R8 tiebreaker (deferred by the GT-free-first
  constraint).
- **KT-1 could itself return leak on most frames**, in which case the honest result is a
  window of `unresolved_full_frame_depth_leak` — a correct non-contact-evidence artifact,
  not a failure to hide.
- **Motion-coupling on clip001850 may be weak** (keyboard barely moves under typing), so
  even the orthogonal channel may not lift candidates to confirmed GT-free; that is a real
  information limit to represent as uncertainty, not to paper over.
- **`build_v19_rigid_render_state.py` already guards** against `unsupported_uncertain`
  promotion and stale layers, yet the shipped world panel still shows 56 736 TRELLIS verts
  — the guard was satisfied by a semantics flag without free-space actually run (KT-4
  contamination passing the guard). The `object_body_provenance` field + KT-4 rebuild must
  gate the body, not a declared flag.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Delivered a complete, implementable per-frame contact-state decision policy at the authoritative path that replaces 'wire observed-surface penetration in'. It consumes KT-1..KT-7 (defined in subagents/06_reconciliation_attack.md) and produces the requested ego.hoi contact_states (confirmed_contact, contact_candidate, unresolved_full_frame_depth_leak, unresolved_pose_gap, geometry_epoch_contaminated, no_contact_supported, plus unresolved_incoherent_evidence and unresolved_evidence_incomplete). Sections give: the per-channel decision procedure (pseudocode), exact promote/demote evidence per state, renderer label mapping onto the existing publish_v19_render_artifact.py:contact_semantics() vocabulary with styling, mandatory provenance columns (KT-6), thresholds each with a physical basis (sigma_clip 25-30mm from Track J/M in hand_metric_mechanisms.md) or explicitly no threshold (confirmed_contact, KT-1 survival), a concrete scaffold patch to decide_mechanism/derive_cross_solver_geometry_consistency, and the next artifact-changing patch (KT-1 + contact_frame_detail writer + KT-7 render consumer + pixel-diff). Load-bearing claims verified against on-disk fields: full-frame depth barrier with mask/quarantine/eligibility OFF; sigma floor 25-30mm; keyboard SAM2 masks present (KT-1 runnable); no renderer consumes ego.hoi rows (KT-7 critical); contact_semantics() already implements the source-gap vocabulary to extend."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read PROMPT.md EPISTEMIC.md OPS.md TASK_PACK.md + subagents/01-06",
      "result": "passed",
      "summary": "Loaded task spec, epistemic model, KT-1..KT-7 definitions (06), reconciliation impl (04), adapter (05), causal card (02), scaffold impl (03)."
    },
    {
      "command": "read scripts/build_ego_hoi_sidecar_and_graph_health.py (decide_mechanism, derive_cross_solver_geometry_consistency, thresholds)",
      "result": "passed",
      "summary": "Confirmed decision-table ordering, SUPPORT_FRACTION_THRESHOLD=0.05/ENERGY_DELTA_INERT_EPS=1e-6, and the tautological coexistence mismatch reason KT-6 attacks."
    },
    {
      "command": "grep hand_metric_mechanisms.md + pipeline_invariants.md; grep publish_v19_render_artifact.py contact_semantics",
      "result": "passed",
      "summary": "sigma_clip 25-30mm lateral + 3.40deg rotation + UniDepth 26-45mm 'too noisy'; existing contact-wording contract at combined-uncertainty scale; unsupported_uncertain-must-be-consumed invariant."
    },
    {
      "command": "find run root for keyboard masks; grep ego.hoi consumers across scripts/render_*.py + publish",
      "result": "passed",
      "summary": "keyboard SAM2 masks present (KT-1 runnable); no renderer consumes contact_frame_detail/graph_health (KT-7 confirmed critical)."
    }
  ],
  "validationOutput": [
    "F1: interval observed-surface barrier queries full-frame UniDepth (150,1408,1408) with mask_gate/hand_quarantine/eligibility=False and depth-order selected 0 verts -> 'wire it in' optimizes hand into table/own-depth; already dragged hand up to 10.8cm.",
    "F2: sigma_clip 25-30mm (systematic bias, not averageable) >> soft-tissue band 2-5mm -> signed distance cannot CONFIRM contact; ceiling is contact_candidate; confirmed_contact needs floor-independent motion-coupling or R8 GT.",
    "Policy default before kill-tests run = unresolved_evidence_incomplete on every frame -> correct honest replacement for premature wire-in.",
    "State machine validated per-channel (not flat cascade) so a contaminated completed-mesh channel does not mislabel a frame that has a real KT-1 observed channel.",
    "Renderer mapping is monotonic: contact_candidate/no_contact_supported reuse existing contact_semantics() wording verbatim; only the unresolved family is extended into machine-readable reasons; geometry_epoch_contaminated additionally hatches the world body (fixes green TRELLIS cloud).",
    "KT-6 scaffold patch removes the coexistence tautology and hard-coded source-family strings, routing to evidence_incomplete on missing provenance hashes."
  ],
  "residualRisks": [
    "No renderer consumes ego.hoi rows today; every contact_state is backing-data-only until the KT-7 consumer patch (build_v19_rigid_render_state.py + publish_v19_render_artifact.py) lands and a pixel/label-diff confirms artifact change.",
    "confirmed_contact is GT-free-unreachable on clip001850 until motion-coupling is implemented; motion-coupling may itself be weak (keyboard barely moves under typing) -> a real information limit to carry as uncertainty, HOT3D GT is the R8 tiebreaker.",
    "KT-1 may return leak on most frames -> a correct window of unresolved_full_frame_depth_leak, not a hidden failure.",
    "build_v19_rigid_render_state.py's unsupported_uncertain/free-space guard was satisfied by a semantics flag without free-space actually run (56736 TRELLIS verts still shipped); object_body_provenance + KT-4 rebuild must gate the body, not a declared flag.",
    "Policy is a formulation over existing kill-tests; the kill-tests themselves (KT-1..KT-5) still need to be executed to assign real post-promotion states (projected assignment given, not measured here)."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one task-memory document (this policy) at the authoritative subagents path. No source files touched; nothing staged.",
  "reviewFindings": [
    "blocker: subagents/06_reconciliation_attack.md A5 + confirmed here - no renderer consumes contact_frame_detail/graph_health (scripts/build_v19_rigid_render_state.py, scripts/publish_v19_render_artifact.py); the policy's states are non-evidential until the KT-7 consumer patch (section 8.3) lands. Severity: critical for delivery.",
    "blocker: 'wire observed-surface penetration in' (04/05 route, TASK_PACK D6) is invalid - the observed surface is full-frame UniDepth with mask/quarantine/eligibility OFF (v18_joint_mano_interval_trajectory_state.json); it must be gated by KT-1 and capped at contact_candidate. Severity: critical for correctness.",
    "major: scripts/build_ego_hoi_sidecar_and_graph_health.py derive_cross_solver_geometry_consistency fires on a tautological 'penetration>0 AND gap>0' reason and adapter-authored source-family strings (build_clip001850_ego_hoi_graph_summary.py); section 7 patch required (KT-6). Severity: major - decision is gameable.",
    "major: completed mesh 95.4% TRELLIS non-watertight, free_space not_evaluated (v18_compact_rigid_trellis_completion_report.json) - contact/NP '0 penetration' is zero-by-construction; geometry_epoch_contaminated state + KT-4 rebuild required before the completed mesh is contact-eligible. Severity: major.",
    "note: confirmed_contact has no signed-distance threshold by physical necessity (sigma_clip 30mm >> soft-tissue 5mm); requires orthogonal motion-coupling or GT - this is a design constraint, not a gap to fill."
  ],
  "manualNotes": "The policy's core reframe for the parent: 'wire the observed-surface penetration in' is the wrong route because the observed surface is full-frame UniDepth (memory rates it 26-45mm 'too noisy') and clip001850's 25-30mm metric floor swamps the 2-5mm soft-tissue band. The penetration becomes an INPUT to KT-1's admissibility test, not a graph factor; distance evidence tops out at contact_candidate; confirmed_contact needs floor-independent motion-coupling or R8 GT. Default state is unresolved_evidence_incomplete until the kill-tests run. The single highest-leverage next patch is KT-1 (keyboard-masked + hand-quarantined re-query, runnable now - masks are on disk) bundled with the contact_frame_detail writer and the KT-7 render consumer + pixel-diff, because until then no rendered annotation changes and no state is decidable. Any nonpenetration factor added after promotion must be soft and sigma_clip-scaled, never the hard 3e5 barrier that dragged the hand 10.8cm."
}
```
