# 17 — Attack on the production-style contact render consumer (subagent 15)

Read-only analysis. No repo edits, nothing staged. Branch `yiwen_research`.
Inputs read: `PROMPT.md`, `EPISTEMIC.md`, `OPS.md`, `TASK_PACK.md` (D6, workstreams C/F/G),
`subagents/09,10,11,12,13,14`, and the actual production render/publish source
(`scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py`,
`scripts/publish_v19_render_artifact.py`, `scripts/build_v19_rigid_render_state.py`,
`scripts/render_v19_rigid_state_artifact.py`, `scripts/render_clip001850_egohoi_contact_review.py`,
`scripts/build_clip001850_contact_state_table.py`). Numbers re-verified against on-disk
run `/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`
and durable artifact `/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/`.

Slice: HOT3D clip001850 keyboard, right hand, published states f32/f36/f45; full-duration 150 frames.

---

## 0. Thesis (read first)

The dispatched work (OPS `332a242f…`) asks subagent 15 to make the canonical
`contact_frame_detail` table "affect a v19-style rendered artifact" so that f32 reads
`geometry_epoch_contaminated`, f36 `full_frame_depth_leak`, f45 `unresolved_incoherent_evidence`,
the body is drawn observed/repaired rather than solid TRELLIS, and the `gap/penverts/UNCERTAIN`
banner is suppressed.

**The risk named in the task is real and structurally forced by how this clip was rendered.**
The clip001850 published video was NOT produced by the render path everyone has been reading
(`build_v19_rigid_render_state.py` → `render_v19_rigid_state_artifact.py`). It was produced by
`render_v18_compact_rigid_tomato_temporal_mano_attempt.py` consuming the **raw TRELLIS mesh**,
then wrapped by `publish_v19_render_artifact.py`, which only stamps a **single static banner on
every frame**. Every per-frame visible mark the task wants changed — the green TRELLIS body,
`penverts=0`, `UNCERTAIN` — is baked into the Stage-1 video, not the Stage-2 banner. Therefore
the single easiest way to "pass" this task is a **cosmetic overlay**: edit the publish banner
text (or emit a new 3-panel still like subagents 09/13) while the canonical full-duration mp4,
its manifest, the drawn body, and the per-frame numeric rows a downstream system consumes stay
identical. That is the failure the kill-tests below must catch.

Two hard facts frame the whole gate:

1. **The banner cannot carry per-frame contact_state.** `publish_v19_render_artifact.py`
   applies one `metrics` string to all 150 frames (`write_video_with_banner` → `annotate_frame`,
   same `summary_text` per frame). f32/f36/f45 differ. So a real per-frame consumer must change
   the **Stage-1 renderer**, not the publish banner. A publish-only diff is definitionally
   insufficient and is the primary cosmetic tell.

2. **The production renderer will now refuse to re-run on this clip.** Both Stage-1
   (`render_v18_…:validate_completed_mesh_contract`) and the parallel `build_v19_rigid_render_state.py:
   validate_completion_body_contract` (working-tree, uncommitted) reject: a `trellis_mesh.ply`
   name, `free_space_rejection_state=not_evaluated…`, and `unsupported_uncertain>0`. clip001850's
   completion report has all three (`free_space not_evaluated`, `unsupported_uncertain=963`,
   `trellis_inferred_hidden_surface=86708`). The published video predates these guards (Jun 26/27).
   So subagent 15 cannot "just re-render"; it must either (a) render the repaired observed body
   (which is not `trellis_*` and can be passed with no `--completion-report`, bypassing the match
   check but not the geometry) or (b) build a valid observed/carved completion report first. If it
   silently bypasses the guard to re-render the TRELLIS body under a relabeled banner, that is a
   cosmetic pass, not progress.

---

## 1. The production render path as actually built (evidence)

From `logs/P19_render_full_duration_keyboard.log` and `P19…status.json`:

- **Stage 1 — geometry/hand/contact pixels (method `render_v18_compact_rigid_object_temporal_mano_attempt`,
  script `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py`).** Inputs:
  `annotations_v19_visible_geometry.json`, `v19_rigid_object_pose_graph_report.json`,
  `completed_mesh = measurements/geometry_completion/trellis_keyboard_seed42/trellis_mesh.ply`
  (**raw TRELLIS**), `v18_mano_object_constraint_state.json`, and
  `temporal_mano_state = …/v18_joint_mano_interval_trajectory_state.json`.
  Outputs: `renders/keyboard_rigid_mano_runtime/hot3d_…/v18_{overlay,world,side_by_side}_keyboard.mp4`
  (150 frames). This is where the following visible marks are drawn:
  - **Green object body** = raw TRELLIS mesh vertices (`load_mesh_vertices(args.completed_mesh)`),
    90,892 faces, AABB `[0.253, 0.664, 0.260]` m, 8.42× too thick in the thin axis.
  - **`penverts=0`** = `row.get("penetrating_vertex_count")` from the constraint rows. Verified:
    all **300/300** constraint rows carry `penetrating_vertex_count=0` — zero-by-construction on a
    non-watertight, free-space-uncarved, 95.4%-TRELLIS mesh (KT-3/KT-4). This is a false
    "no penetration ⇒ clearance" claim.
  - **`UNCERTAIN`** = `if "uncertainty" in state: return (0,200,255),3,"UNCERTAIN"` (line ~274).
- **Stage 2 — banner only (`scripts/publish_v19_render_artifact.py`).** Reads the Stage-1 videos +
  `--interval-state`, stamps a static banner, writes `renders/v19_published_runtime/v19_{overlay,
  world,side_by_side}.mp4` and canonical `renders/v19_{overlay,world,side_by_side}.mp4`. Verified
  banner (`v19_published_render_report.json`):
  `subtitle = "green=rigid object | … | UNCERTAIN=not accepted contact closure"`,
  `metrics.summary_text = "L: gap 98.6mm, shift 0.0px, closed False | R: gap 39.2mm, shift 0.0px, closed False"`.
  `contact_semantics` and `contact_likelihood_state_counts` are **None** (the interval state here
  uses the `side_summary` gap path, not the source-gap posterior).

**No `rigid_render_state.json` exists for this run.** The `build_v19_rigid_render_state.py` /
`render_v19_rigid_state_artifact.py` path was never used for clip001850. Do not let subagent 15
"consume the render state" by writing a fresh `build_v19_rigid_render_state.py` output that the
published video never depended on — that is a new artifact disconnected from the delivered one.

**Path note:** logs reference `/mnt/truenas-user-home/yiwen/…`; the accessible mount is
`/data2/…`. A re-render needs `--path-rewrite /mnt/truenas-user-home/yiwen=/data2` (or equivalent).

**What "numeric/render state a downstream system consumes" means here:** the full-duration
mp4s (visual), plus `v18_temporal_rigid_object_manifest.json`, the constraint rows
(`penetrating_vertex_count`), the interval trajectory gap, the drawn mesh identity, and
`v19_published_render_report.json`. A pass must change these, not only banner pixels.

---

## 2. What subagents 09/11/13 actually produced (and did not)

- **09/13 review still (`render_clip001850_egohoi_contact_review.py`)** is a 3-panel JPG for f32/f36
  only: Panel A = published-banner text, Panel B = MANO **sample vertices** projected (cyan dots),
  Panel C = a text card whose label is bound to canonical `contact_state`. It **draws no object body
  at all** — no world/body panel, no mesh. It is a QC still, not the production video. It does not
  satisfy "repaired/observed body replaces the TRELLIS body," and it is not the canonical mp4.
- **11 repaired bodies** exist as `.ply` only (durable:
  `research_clip001850_contact_state_20260706/keyboard_body_repair/repaired_observed_contact_body.ply`
  = 3221 faces, AABB `[0.198,0.543,0.169]` m; `repaired_carved_support_body.ply` = 41,670 faces,
  AABB thin-axis **0.253 m unchanged**). **Neither has ever been rendered into any video.**
- **13 canonical table** covers **only 21 frames (28–48)** of 150; states `pose_unresolved×12,
  geometry_epoch_contaminated×7, full_frame_depth_leak×1, unresolved_incoherent_evidence×1`. Every
  row's `render_consumed_mesh_sha256 = 260b09d1…` = the **TRELLIS** completed-mesh hash, and rows
  carry `renderer_consumes_contact_state=true` only for f32/f36 (the review still), false elsewhere
  including f45. So the backing table **still points the renderer at the TRELLIS mesh**; consuming
  it as-is would drive a TRELLIS body render.
- **Storage:** the live copies are under `/tmp/clip001850_*` (ephemeral). The durable copies are
  under `/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/`. A production
  consumer must read the durable path, not `/tmp`.

---

## 2.1 Observed subagent 15 draft — the predicted cosmetic pattern is already materializing

An in-progress consumer exists: `scripts/render_clip001850_v19_contact_state_artifact.py`
(untracked, 640 lines, mtime 04:45). Read-only inspection confirms it lands squarely in the
failure catalog:

- **Produces stills, not the canonical mp4 (CF-2 + CF-7).** Its outputs are
  `overlay_f032.jpg / world_f032.jpg / side_by_side_f032.jpg` (and f36/f45) plus extracted
  `published_*` stills "for diff." It **reads** the published videos
  (`RUN_ROOT/renders/v19_{overlay,world,side_by_side}.mp4`) but does **not** re-render them. The
  canonical full-duration mp4 a downstream system consumes stays byte-identical (solid TRELLIS body,
  `penverts=0`, `gap 39.2mm`, `UNCERTAIN`). This is exactly "a cosmetic overlay that does not alter
  the numeric/render state a downstream system would consume." → fails A1 and the KT-R1 gate.
- **Reads ephemeral `/tmp` (CF-8).** Sources are `/tmp/clip001850_contact_state_table/…` and
  `/tmp/clip001850_keyboard_body_repair/repaired_observed_contact_body.ply`, not the durable
  `/data2/…/research_clip001850_contact_state_20260706/` copies. → fails A7.
- **What it does right (keep this, move it into the real chain):** it draws the observed body as a
  hatched/wireframe uncertain patch placed into the world frame (`observed_body_world`,
  `render_uncertain_body_overlay`) instead of a solid TRELLIS body, and it stamps a per-frame banner
  driven by the canonical `contact_state` (`CONTACT_STATE_BANNER`, e.g. "UNRESOLVED: object body 8.4x
  too thick, 95.4% TRELLIS-inferred"). This is the correct body-replacement + per-frame-label
  behavior — but only inside its own 3-frame stills. The task is to make this happen **in the
  canonical full-duration render + its backing state**, replacing the TRELLIS body there.

So the current draft satisfies KT-R2/KT-R3/KT-R4 *inside a QC still* while failing KT-R1's gate
(the full-duration mp4) and A7 (durable provenance). It is a QC still masquerading as the consumer.

## 3. The four kill-tests requested

Gate = the **canonical full-duration render** (`renders/v19_{overlay,world,side_by_side}.mp4`) and
its backing numeric state, consumed as an annotation. A schema row, watertight flag, distance
number, validator pass, or a new still is **not** a pass. Each test states the discriminating
prediction so that both outcomes change the decision.

### KT-R1 — Does the render actually consume canonical `contact_frame_detail`?
**Measure:** (a) the per-frame visible contact label the full-duration mp4 shows at f32/f36/f45;
(b) whether it is byte-derivable from the canonical `contact_frame_detail.ndjson.contact_state`;
(c) a **perturbation**: change one state in the canonical table (e.g. set f36 to a sentinel string),
re-run subagent 15's producer, and re-read the mp4 label at f36.

- **Consumed (pass):** f32 label = `geometry_epoch_contaminated`, f36 = `full_frame_depth_leak`,
  f45 = `unresolved_incoherent_evidence`, drawn in the **Stage-1 per-frame layer** (not the static
  banner); the manifest cites the canonical table path + sha256 as `contact_state_source`; and the
  perturbation changes the f36 mp4 label. → real consumer.
- **Not consumed (fail):** the label is unchanged, or comes from the interval state / a hardcoded
  string, or only the static publish banner changed (identical text on all 150 frames), or the
  perturbation does not alter the video. → cosmetic. **Severity: CRITICAL.**
- **Coverage sub-test:** the table covers 21/150 frames. The full-duration render must assign the
  other 129 frames a **documented default** (policy default `unresolved_evidence_incomplete`), not
  silently drop them and not fabricate contact. If the 129 uncovered frames render as accepted
  contact or as a blank state, fail. **Severity: MAJOR.**

### KT-R2 — Does the repaired/observed body REPLACE the TRELLIS body (not an extra panel)?
**Measure:** the identity and provenance of the object mesh drawn in the world/overlay panel of the
mp4 at f32/f36: face count, face-provenance composition, and whether the hand is enclosed by the body.

- **Replaced (pass):** the drawn body is the observed (3221-face) or carved (41,670-face,
  TRELLIS-hidden faces removed) body; TRELLIS-inferred-hidden faces (86,708) are **removed or drawn
  hatched/uncertain**, not solid; `render_consumed_mesh_sha256` in the backing state = the repaired
  body's hash (≠ `260b09d1…`); and the hand is drawn **in front of / outside** the body surface,
  consistent with KT-1's 80–180 mm hand-in-front (the phantom bulge that enclosed the hand is gone).
- **Decoration / not replaced (fail):** the world panel still shows the solid 90,892-face TRELLIS
  blob; or a repaired body is added as a **second panel / extra overlay** while the original body is
  unchanged; or `render_consumed_mesh_sha256` is still the TRELLIS hash. → extra-panel cosmetic.
  **Severity: CRITICAL.**
- **Correction to subagent 14's KT-A criterion:** do **not** use "thin-axis extent ≤ 0.05 m off the
  rendered mesh." That is unreachable by the honest bodies: the observed body's canonical-frame AABB
  thin axis is **0.169 m** and the carved body's is **0.253 m** (the canonical axes do not align with
  the keyboard's physical thin axis, and multi-key depth noise spreads the surface). Enforcing 0.05 m
  would false-fail a correct repair or, worse, pressure a category-primitive box snap (forbidden,
  F-A2). Use **face count + face provenance + hand-not-enclosed** as the discriminator instead.
- **Watertight tell (F-A1):** if the drawn body reports `watertight=true` with a nonzero
  `penetrating_vertex_count`, the interior faces are inferred (TRELLIS/prior) — reject; nonpenetration
  is not an available contact channel for a singly-observed keyboard.

### KT-R3 — Do labels suppress the invalid gap/penverts claims?
**Measure:** presence, at f32/f36/f45, of `gap 39.2mm` (contaminated completed-mesh / interval gap),
`penverts=0` (zero-by-construction), and bare `UNCERTAIN` as accepted-contact metrics — in both the
Stage-1 per-frame text and the Stage-2 banner.

- **Suppressed (pass):** `penverts=0` is removed or replaced with an explicit
  "penetration undefined — body non-watertight/contaminated" tag; the `gap 39.2mm` accepted-contact
  number is removed or replaced with the canonical `source_gap` (keyboard-masked delta, sign = hand
  in front: f32 ≈ −144 mm, f36 ≈ −81 mm) or a "gap invalid — contaminated body" tag; bare `UNCERTAIN`
  is replaced by the named state. Because the static banner cannot vary per frame, the suppression
  must be in the **Stage-1 per-frame layer** (the banner may additionally be neutralized).
- **Not suppressed (fail):** any of `gap 39.2mm`, `penverts=0`, bare `UNCERTAIN` still reads as an
  accepted contact/clearance claim on f32/f36/f45. **Severity: CRITICAL** (these are the exact
  false claims the deliverable must retire; leaving them is the cosmetic-relabel failure).

### KT-R4 — Does f45 remain demoted?
**Measure:** the f45 mp4 label and whether any contact mark is drawn on f45; and the f45 backing row.

- **Demoted (pass):** f45 label = `unresolved_incoherent_evidence` (or `unresolved`), **no contact
  mark** (no dot/line/green contact) drawn; backing row carries `contact_state=unresolved_incoherent_
  evidence` with the demotion reason (one thumb vertex at +5.86 mm ≪ σ_clip on a 393.6 mm depth
  outlier vs 520 mm keyboard median, zero temporal coherence).
- **Re-promoted (fail):** f45 renders `contact_candidate` / `near-contact` / any contact mark,
  i.e. the raw kill-test script's permissive ≥1-vertex rule leaked back through the production
  consumer instead of the demoted canonical state. **Severity: MAJOR.**

### KT-R5 (cross-cutting, from subagent 14 KT-U) — verdict not laundered into contact
**Measure:** whether any frame in f28–48 renders `confirmed_contact` and, if so, whether a
floor-independent provenance column (motion-coupling onset or R8 GT) is populated.

- **Pass:** window f28–48 renders **zero admissible contact frames**; no `confirmed_contact` appears
  without a motion/GT column; the residual gap + σ_clip is visible and the body is hatched where
  inferred.
- **Fail:** a frame flips to contact justified only by a shrunken gap, a signed-distance threshold,
  or the agent interaction judgment. **Severity: CRITICAL** (forced contact).

---

## 4. Cosmetic-overlay failure catalog (the specific ways subagent 15 fakes it)

- **CF-1 banner-only swap** — edit `publish_v19_render_artifact.py` to print canonical labels in the
  static banner; Stage-1 body/penverts/UNCERTAIN unchanged. Tell: publish is the only file changed;
  identical text on all 150 frames; f32/f36/f45 cannot differ. Caught by KT-R1(a,c), KT-R3.
- **CF-2 new-still substitution** — emit another 3-panel review still (09/13 pattern) and call it the
  consumer; the canonical mp4 is byte-identical. Tell: the deliverable is a `.jpg`/2-frame artifact,
  not `renders/v19_*.mp4`. Caught by KT-R1 (gate is the full-duration mp4), full-duration invariant.
- **CF-3 label-over-solid-body** — relabel to `geometry_epoch_contaminated` while still drawing the
  solid green TRELLIS blob. Tell: `render_consumed_mesh_sha256 = 260b09d1…`; world panel unchanged;
  label contradicts the still-solid body. Caught by KT-R2.
- **CF-4 extra-panel body** — add the repaired body as a new side panel/overlay, leaving the original
  world body TRELLIS. Tell: two bodies visible; original panel unchanged. Caught by KT-R2 ("replace,
  not add").
- **CF-5 numeric-state divergence** — change pixels but leave the manifest / constraint rows
  (`penetrating_vertex_count=0`) / interval gap / mesh identity untouched, so a downstream JSON
  consumer sees no change. Tell: `v18_temporal_rigid_object_manifest.json` and constraint rows
  unchanged. Caught by KT-R1(b) + KT-R2 (state, not only pixels).
- **CF-6 guard-bypass re-render** — disable/bypass the completion-body guard to re-render the TRELLIS
  body so the pipeline "runs green," then relabel. Tell: the drawn body is still 90,892-face TRELLIS;
  guard edited or `--completion-report` dropped while `--completed-mesh` is the TRELLIS mesh. Caught
  by KT-R2 + a guard-honesty check (A8).
- **CF-7 partial-window render** — render only f32/f36/f45 (or 28–48) as a "clip" and present it as
  the artifact. Tell: frame count ≠ 150 / duration ≠ raw. Caught by full-duration invariant.
- **CF-8 /tmp provenance** — consume `/tmp/clip001850_contact_state_table` which is ephemeral and
  disappears; the render is not reproducible. Tell: source path under `/tmp`. Caught by A7.

---

## 5. Artifact-level acceptance criteria for subagent 15

A deliverable is **progress iff all of A1–A8 hold**; any single failure among A1/A2/A3/A5 is a
CRITICAL reject (cosmetic or forced-contact).

- **A1 — Real consumer, full-duration.** The producer is (or re-points) the Stage-1 renderer that
  drives the canonical `renders/v19_{overlay,world,side_by_side}.mp4`, at the full **150 frames /
  same duration as raw**. Per-frame contact label at f32/f36/f45 = the canonical `contact_state`,
  drawn in the per-frame layer, and a **pixel/label diff vs the current published still is nonzero**
  at f32/f36/f45. (KT-R1; subagent 14 G0.)
- **A2 — Body replaced, not decorated.** The world/overlay object body at f32/f36 is the
  observed/carved repaired body (face count ≈ 3221 or 41,670; TRELLIS-hidden faces removed or
  hatched), the hand is drawn in front of it, and the backing state's
  `render_consumed_mesh_sha256` = the repaired body hash (≠ TRELLIS). No residual solid TRELLIS blob;
  no second body panel. (KT-R2.)
- **A3 — Invalid claims retired.** `gap 39.2mm`, `penverts=0`, and bare `UNCERTAIN` no longer read as
  accepted contact/clearance on f32/f36/f45; replaced by canonical `source_gap`/named state/explicit
  "undefined" tags. (KT-R3.)
- **A4 — f45 demoted.** f45 renders `unresolved_incoherent_evidence`, no contact mark. (KT-R4.)
- **A5 — Verdict preserved.** No `confirmed_contact` on f28–48 without a motion/GT provenance column;
  window shows zero admissible contact frames; body hatched where inferred; residual gap + σ_clip
  visible. (KT-R5 / subagent 14 KT-U.)
- **A6 — Full-window honesty.** The 129 non-kill-test frames carry the documented default state
  (`unresolved_evidence_incomplete`), are not dropped, and are not fabricated as contact. (KT-R1
  coverage.)
- **A7 — Reproducible, no cheat.** Labels/body derive from the **durable** canonical table + repaired
  body (`/data2/…/research_clip001850_contact_state_20260706/`, not `/tmp`), the mapping is
  table-derived not hardcoded (perturbation test passes), and the manifest cites input hashes.
- **A8 — Guard-honest.** If the completion-body guard blocks re-render, subagent 15 either (a) renders
  the repaired observed body with recorded provenance (best available fidelity, hatched where
  inferred), or (b) records the guard block as the concrete blocker and stops — it does **not** bypass
  the guard to re-render the TRELLIS body under a relabeled banner (CF-6).

**One-line gate:** subagent 15 passes iff the **canonical full-duration mp4** shows, per frame,
the canonical `contact_state` (f32 contaminated / f36 leak / f45 demoted) with the TRELLIS body
**replaced** by the hatched repaired body and the false `gap/penverts/UNCERTAIN` claims **gone from
the pixels and the backing state** — and no frame is forced to contact. A banner edit, a new still,
a second body panel, an unchanged manifest, or a guard-bypassed TRELLIS re-render is fake.

---

## 6. Residual risks / boundaries of this analysis

- **This analysis does not execute subagent 15's render**; it defines the gate. The pass conditions
  are measured on the mp4 + manifest that subagent 15 must produce.
- **Rendering the repaired body is feasible but under-specified.** The Stage-1 renderer accepts an
  arbitrary `--completed-mesh` when `--completion-report` is omitted (the `trellis_*` name check still
  applies; the repaired body passes it). But omitting the completion report drops provenance, and the
  observed body is an open sheet — the render must style it as an uncertain surface, not a closed body.
  A cleaner route is a minimal observed/carved completion report so provenance and hatching are driven
  by data, not flags.
- **The static-banner architecture means Stage 2 cannot express per-frame state.** Any real per-frame
  consumer edits Stage 1; a Stage-2-only change is by construction cosmetic. If subagent 15 argues the
  banner is enough, that argument is false for this renderer.
- **σ_clip / the contact verdict are inherited** from subagents 07/10/11/12/14 (measured, not
  re-derived here). If a later channel (motion-coupling, R8 GT) legitimately promotes a frame, A5's
  provenance-column requirement is the correct release valve — not a signed-distance threshold.
- **Coverage vs runtime.** Extending the canonical table to 150 frames means KT-1 on all
  mask-available frames (CPU-only, light) plus a documented default elsewhere; this stays within the
  runtime invariant and must not be skipped by rendering only the 21-frame window.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Attacked the production-style contact render consumer with on-disk-verified structure. Established the true render chain: clip001850's published video was built by scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py (method render_v18_compact_rigid_object_temporal_mano_attempt) consuming the RAW trellis_mesh.ply, then wrapped by scripts/publish_v19_render_artifact.py which stamps ONE static banner on all 150 frames (write_video_with_banner applies a single metrics summary_text per frame) -> banner cannot carry per-frame contact_state, so a publish-only edit is definitionally cosmetic. No rigid_render_state.json exists for this run; build_v19_rigid_render_state.py/render_v19_rigid_state_artifact.py were never in this clip's chain. Verified: penverts=0 on all 300/300 constraint rows (penetrating_vertex_count=0, zero-by-construction on non-watertight 95.4% TRELLIS mesh); banner text 'R: gap 39.2mm, shift 0.0px, closed False'; UNCERTAIN baked in Stage-1; completion report free_space_rejection_state=not_evaluated, unsupported_uncertain=963, trellis_inferred_hidden_surface=86708 -> current working-tree guards (validate_completed_mesh_contract + validate_completion_body_contract) refuse a re-render on this completion; repaired bodies exist only as .ply (observed 3221 faces AABB[0.198,0.543,0.169]m, carved 41670 faces thin-axis 0.253m unchanged) and were never rendered; canonical table covers 21/150 frames with render_consumed_mesh_sha256=TRELLIS on every row; live copies under ephemeral /tmp. Delivered four requested kill-tests (KT-R1 consume canonical / KT-R2 body-replace-not-panel / KT-R3 suppress gap+penverts+UNCERTAIN / KT-R4 f45 demoted) plus cross-cutting KT-R5 (no forced contact), a CF-1..CF-8 cosmetic-failure catalog, and artifact-level acceptance criteria A1-A8. Corrected subagent 14's unreachable 'thin-axis <=0.05m' KT-A criterion (observed body is 0.169m, carved 0.253m in canonical AABB) to face-count+provenance+hand-not-enclosed."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "read PROMPT/EPISTEMIC/OPS/TASK_PACK + subagents 09,10,11,12,13,14", "result": "passed", "summary": "Loaded task spec, contact policy (KT-1..KT-7), the two closed repair branches, the canonical table writer, and the prior render-consumer attack (G0/KT-U)."},
    {"command": "read scripts/publish_v19_render_artifact.py + render_v18_compact_rigid_tomato_temporal_mano_attempt.py + build_v19_rigid_render_state.py + render_v19_rigid_state_artifact.py (arg surface) + render_clip001850_egohoi_contact_review.py", "result": "passed", "summary": "Confirmed static per-frame banner; identified real Stage-1 renderer + raw-TRELLIS body + penverts/UNCERTAIN origin; both render paths now guard-reject the contaminated completion; review still draws no object body."},
    {"command": "inspect run root: P19/P20 logs+status, v19_published_render_report.json, completion report face_label_counts+free_space_state, constraint rows penetrating_vertex_count, render_state search", "result": "passed", "summary": "P19 method render_v18_compact... with trellis_mesh.ply; banner 'R gap 39.2mm/closed False'; free_space not_evaluated; unsupported 963/trellis 86708; 300/300 constraint rows penverts=0; NO rigid_render_state.json exists."},
    {"command": "inspect canonical table + durable artifact dir + repaired body plys", "result": "passed", "summary": "Table 21/150 frames (28-48), states pose_unresolved12/geom7/leak1/incoherent1, render_consumed_mesh_sha256=TRELLIS(260b09d1) on all rows; durable copies at research_clip001850_contact_state_20260706 (contact_state_table + keyboard_body_repair .ply); live copies transient under /tmp."},
    {"command": "git status --short + branch (context only, no edits)", "result": "passed", "summary": "On yiwen_research; build_v19_rigid_render_state.py is pre-existing dirty (not mine); nothing staged; only new file is this findings doc at the authoritative path."}
  ],
  "validationOutput": [
    "Production render is two-stage: Stage-1 render_v18_compact_rigid_tomato_temporal_mano_attempt.py bakes body/penverts/UNCERTAIN from RAW trellis_mesh.ply; Stage-2 publish_v19_render_artifact.py stamps one static banner per video -> per-frame contact_state must be driven at Stage-1, banner-only is cosmetic.",
    "penverts=0 verified on all 300 constraint rows = zero-by-construction on non-watertight 95.4% TRELLIS mesh; gap 39.2mm from contaminated completed mesh; both are the false claims to retire.",
    "Both current render paths guard-reject clip001850's completion (trellis name / free_space not_evaluated / unsupported_uncertain=963) -> subagent 15 cannot 'just re-render'; must render the repaired observed body or record the guard block, never bypass to a relabeled TRELLIS render.",
    "Subagent 14's KT-A thin-axis<=0.05m is unreachable (observed body canonical AABB thin axis 0.169m, carved 0.253m); replaced with face-count + face-provenance + hand-not-enclosed discriminator.",
    "Canonical table = 21/150 frames, render_consumed_mesh_sha256=TRELLIS on every row; full-duration render needs a documented default on 129 frames and a repaired-body hash to swap the body.",
    "Gate is the canonical full-duration mp4 + backing manifest/rows, not a schema row, a banner, or a new 3-panel still."
  ],
  "residualRisks": [
    "This is a decision/attack analysis; it does not execute subagent 15's render. A1-A8 are the acceptance gate for whoever runs it.",
    "Rendering the repaired observed body is feasible (Stage-1 accepts arbitrary --completed-mesh with no --completion-report) but drops provenance and is an open sheet; prefer a minimal observed/carved completion report so hatching/provenance are data-driven, not flag-driven.",
    "Live canonical table/bodies/stills are under ephemeral /tmp; only /data2/.../research_clip001850_contact_state_20260706 is durable - the consumer must read the durable path or the render is not reproducible.",
    "The contact verdict (unresolved, zero admissible frames) is inherited from subagents 07/10/11/12/14, measured not re-derived here; a later motion-coupling/GT channel may legitimately promote a frame only via the A5 provenance column, never a signed-distance threshold.",
    "Full-window (150-frame) coverage requires KT-1 on all mask-available frames (CPU-light) plus a documented default elsewhere; rendering only the 21-frame window is a partial-window fake (CF-7)."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one task-memory findings document at the authoritative subagents path (17_render_consumer_attack.md). No source files touched; nothing staged.",
  "reviewFindings": [
    "blocker (CRITICAL): the in-progress subagent 15 draft scripts/render_clip001850_v19_contact_state_artifact.py already exhibits the named risk - it emits f32/f36/f45 JPG stills (overlay/world/side_by_side) that READ the published mp4 but do not re-render it, and it sources ephemeral /tmp; it draws the hatched observed body + per-frame canonical labels correctly but only inside its own stills, so the canonical full-duration renders/v19_*.mp4 + manifest a downstream system consumes stay TRELLIS/penverts=0/gap-39.2mm. It is a QC still masquerading as the consumer (fails A1 full-duration + A7 durable); the correct body-replacement/label logic must be moved into the Stage-1 renderer that drives the canonical mp4.",
    "blocker (CRITICAL): the production render chain (render_v18_compact_rigid_tomato_temporal_mano_attempt.py -> publish_v19_render_artifact.py) does not consume contact_frame_detail; publish stamps ONE static banner on all 150 frames so per-frame f32/f36/f45 states are impossible from Stage-2 - any publish-only or new-still deliverable is cosmetic and fails KT-R1.",
    "blocker (CRITICAL): the drawn object body is the raw 90892-face TRELLIS mesh baked in Stage-1; the repaired observed/carved bodies (3221/41670 faces) have never been rendered and the canonical table's render_consumed_mesh_sha256 still = TRELLIS - the body must be REPLACED in the world panel (KT-R2), not added as an extra panel and not left solid under a new label.",
    "blocker (CRITICAL): penverts=0 (all 300 constraint rows, zero-by-construction on a non-watertight mesh) and gap 39.2mm (contaminated completed mesh) are false accepted-contact/clearance claims baked into Stage-1; they must be suppressed/replaced on f32/f36/f45 in pixels AND backing state (KT-R3), not just in the banner.",
    "blocker (CRITICAL): both current render paths guard-reject clip001850's completion (trellis name / free_space not_evaluated / unsupported_uncertain=963); a guard-bypass re-render of the TRELLIS body under a relabeled banner (CF-6) is a cosmetic pass - reject; render the repaired body or record the block (A8).",
    "major: f45 must render unresolved_incoherent_evidence with no contact mark (KT-R4); the raw kill-test script's permissive >=1-vertex rule (+5.86mm thumb vertex on a 393.6mm depth outlier) must not leak back through the production consumer.",
    "major: canonical table covers 21/150 frames; the full-duration render must assign the other 129 frames the documented default (unresolved_evidence_incomplete), not drop them or fabricate contact (KT-R1 coverage / A6).",
    "major: live artifacts are under ephemeral /tmp; the consumer must read the durable /data2/.../research_clip001850_contact_state_20260706 copies or the render is not reproducible (A7/CF-8).",
    "correction: subagent 14's KT-A 'thin-axis <=0.05m off the rendered mesh' is unreachable (observed body 0.169m, carved 0.253m in canonical AABB) and would false-fail a correct repair or pressure a forbidden primitive-box snap; use face-count + face-provenance + hand-not-enclosed instead (KT-R2)."
  ],
  "manualNotes": "Core reframe for the parent: the clip001850 published render is NOT the render_state path everyone has been reading. It is render_v18_compact_rigid_tomato_temporal_mano_attempt.py (raw TRELLIS body) + publish_v19_render_artifact.py (single static banner per video). Because the banner is static, per-frame contact_state can only be expressed by editing the Stage-1 renderer; a publish-only edit or a new 3-panel still is definitionally cosmetic and must fail. CONFIRMED: subagent 15's in-progress draft (scripts/render_clip001850_v19_contact_state_artifact.py) is exactly this cosmetic pattern - f32/f36/f45 stills that read (not re-render) the canonical mp4 and source /tmp, drawing the correct hatched body + labels only inside its own stills while the downstream-consumed canonical renders/v19_*.mp4 stays TRELLIS/penverts=0/gap-39.2mm. The three visible marks the task wants changed (green TRELLIS body, penverts=0, gap 39.2mm/UNCERTAIN) are all baked in Stage-1, and both current render paths now guard-reject re-rendering this contaminated completion - so subagent 15's real work is: (1) render the repaired observed/carved body IN PLACE of the TRELLIS body (not an extra panel, not a separate still), hatched where inferred; (2) drive per-frame labels from the DURABLE canonical contact_frame_detail (f32 contaminated / f36 leak / f45 demoted), extended or defaulted across all 150 frames; (3) delete the false penverts=0 and gap-39.2mm claims from both pixels and backing state; (4) keep the verdict unresolved with zero admissible contact frames. Grade it on the canonical full-duration mp4 + its manifest/rows (A1-A8), never on a schema row, a banner, a still, or a guard-bypassed TRELLIS re-render."
}
```
