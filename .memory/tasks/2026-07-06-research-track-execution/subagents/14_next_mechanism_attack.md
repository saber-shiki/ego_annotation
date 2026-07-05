# 14 — Attack on the next two root mechanisms: object-geometry repair & hand-depth-bias correction

Read-only analysis. No repo edits, nothing staged. Branch `yiwen_research`.
Inputs read: `PROMPT.md`, `EPISTEMIC.md`, `OPS.md`, `TASK_PACK.md` (D6, workstreams C/E/F),
`subagents/06,07,08,09,10`, project memory (`hand_metric_mechanisms.md`,
`object_geometry_and_render_lessons.md`, `pipeline_invariants.md`,
`self_consistency_metrics.md`). Numbers re-verified against on-disk
`/tmp/clip001850_masked_contact_killtests/{summary.json,contact_killtests_frame_detail.ndjson}`
and the run root `20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.

Slice under attack: HOT3D clip001850 keyboard, **right hand, frames 28–48**.

---

## 0. The single fact that governs both branches (read this first)

**Both proposed mechanisms, executed honestly, converge on the same verdict: right-hand
keyboard contact on clip001850 stays UNRESOLVED. Neither branch can produce
`confirmed_contact` GT-free.** They differ only in *which* fake-progress story would be
told if the branch is graded on a proxy instead of the artifact.

The physical geometry that forces this:

- The keyboard-masked, hand-quarantined depth surface (the physical first surface) sits at
  cam_z ≈ **0.50–0.52 m**. The right-hand MANO vertices sit at cam_z ≈ **0.39–0.44 m**.
  The hand is **80–180 mm in front of the keyboard surface** on every mask-available frame
  (f30 −176 mm, f32 −144 mm, f36 −81 mm, f45 −85 mm), with **0 eligible penetrating
  vertices** (KT-1, subagent 07).
- clip001850's measured hand-metric error scale is **25–45 mm** (lateral floor 25–30 mm,
  systematic; UniDepth at hand pixels 26–45 mm; camera-frame rotation 3.40° ≈ 15–25 mm at
  0.35 m; raw HaWoR wrist medians 20–37 mm). This is **3–5× smaller than the 80–180 mm
  gap** (`hand_metric_mechanisms.md`).

Consequences that are not negotiable by either branch:

1. **Object-geometry repair cannot move the contact verdict.** The contact-relevant surface
   is the observed keyboard top. KT-1 already queried it and found the hand 80–180 mm in
   front. A correctly carved observed body measures *the same top surface*, so it returns
   *the same gap*. Geometry repair changes the **render body** and **invalidates the false
   interval penetration** — real, monotonic wins — but it does **not** create contact.
2. **Hand-depth correction cannot close the gap.** Bounded by the measured 25–45 mm error
   scale, a correct along-ray depth fix removes at most ≈ a quarter to a half of the
   80–180 mm gap. A residual 40–140 mm gap remains → hand still not on the keyboard →
   contact still unresolved. Pushing the hand the full 80–180 mm to reach the keyboard would
   inject an 80–180 mm wrist error where the measured error is 20–37 mm — i.e. it forces
   contact by breaking the hand.
3. Because HaWoR wrist is within 20–37 mm of GT (camera frame) yet the hand renders 80–180 mm
   in front of the keyboard, **GT itself likely places the hand ~50–160 mm in front of the
   keyboard surface**. The honest live hypotheses are therefore *hand genuinely hovering /
   reaching* OR *keyboard depth biased far* — **not** "obvious contact the pipeline is
   missing." Any branch that assumes contact and works backward is scientifically invalid on
   this clip.

The fake-progress trap is identical for both branches: **reporting a render change (branch 1)
or a metric-accuracy change (branch 2) as "resolved contact."** Neither is contact. The
checklist in §5 gates exactly that substitution.

---

## 1. Branch A — object-geometry repair (free-space carving / observed-face watertight sign mesh)

Intended by D6/workstream C: rebuild the keyboard epoch so the body is contact-eligible.
Current epoch: `completed_mesh` **90 892 faces, 95.40 % TRELLIS-inferred**, 3.54 % observed,
non-watertight, `free_space_rejection_state=not_evaluated`, AABB **0.253 × 0.664 × 0.260 m**
vs implied keyboard prior 0.171 × 0.383 × **0.031 m** → **8.4× too thick in the thin axis**
(re-verified from `summary.json`). This phantom 26 cm body is what the interval barrier
"penetrates" (82–107 mm), because the hand at 0.40 m sits inside a body that bulges from the
keyboard surface toward the camera.

### 1.1 What Branch A can legitimately change (the real deliverable)

- **World-panel body**: replace the 56 736-vertex TRELLIS blob (drawn today as accepted
  green body — a pipeline-invariant violation) with an observed keyboard top surface whose
  thin-axis extent ≈ the depth-measured keyboard thickness (~2–4 cm), styled as uncertain
  where inferred. Monotonic render win (`object_geometry_and_render_lessons.md`,
  `pipeline_invariants.md`).
- **Invalidate the false interval penetration**: once free space in front of the keyboard is
  carved, the hand is shown in **carved free space**, so the 82–107 mm interval penetration
  and the barrier-driven **10.8 cm hand drag** (subagent 06 A1) are revealed as artifacts and
  reverted.

Neither of these is a contact channel. They are a geometry/render repair.

### 1.2 Fake-progress modes (what would make Branch A a proxy win)

- **F-A1 — Watertight-by-completion.** To get signed penetration you need a *closed* body.
  A keyboard seen from one head-cam viewpoint is observed only on its **top sheet**; sides,
  bottom, and interior are never observed. Closing that sheet into a watertight body requires
  **inferred faces (TRELLIS or a category primitive)** — reintroducing exactly the phantom
  surface the repair was meant to delete. "Observed-face **watertight** sign mesh" is a
  contradiction for a singly-observed keyboard: observed-only ⇒ open sheet ⇒ signed distance
  undefined; watertight ⇒ inferred interior ⇒ contaminated. **Nonpenetration as a contact
  channel is simply not available for this object from this viewpoint.** Any Branch A output
  that reports a `sign_mesh_watertight=true` keyboard and a nonzero `penetrating_vertex_count`
  is asserting contact against invented geometry — fake.
- **F-A2 — Category-primitive substitution.** Pruning/refitting the mesh to the keyboard
  prior box (0.45 × 0.15 × 0.03) and calling that "carved geometry" is a category primitive
  standing in for reconstruction — forbidden by AGENTS.md. Observable tell: the repaired
  body's extents snap to the prior rather than to per-frame depth.
- **F-A3 — Container win.** Emitting `free_space_rejected_fraction > 0`,
  `object_body_provenance=observed_carved`, or a schema/graph-health row while the **rendered
  world body is unchanged** and the **contact query's nearest face is still TRELLIS**. This is
  the exact failure the pipeline invariant names: an `unsupported_uncertain`/carved *flag*
  satisfied without free space actually driving the body (subagent 10 §9: guard passed, 56 736
  TRELLIS verts still shipped).
- **F-A4 — Verdict laundering.** Reporting "geometry repaired → penetration gone → therefore
  no contact / contact resolved." The penetration was never contact evidence; removing a
  phantom does not resolve contact. The verdict was and remains hand-in-front / unresolved.

### 1.3 Exact observable artifact that proves Branch A changed the problem (not a proxy)

A **re-rendered clip001850 world/side-by-side still (f32 and f36)** in which **all** hold:

1. The keyboard body's **thin-axis extent, measured off the rendered mesh, ≤ ~0.05 m**
   (down from 0.26 m), and its shape reads as a keyboard top surface, not a 26 cm slab or a
   prior box.
2. The **contact query's per-query nearest-face provenance at the hand region = `observed_depth_surface`**
   (not `trellis_inferred_hidden_surface`), recorded per vertex, not as a global fraction.
3. The **interval barrier penetration recomputed against the carved body = ~0** at f32/f36
   (the phantom is gone), and the hand is drawn in carved free space; the previously published
   `penverts`/barrier hand-drag is reverted in the state the render consumes.
4. The frame's `contact_state` **remains `unresolved`/`no_contact_supported`** with a visible
   residual gap and hatched (uncertain) body — it does **not** flip to contact.

Proxy that does NOT count: any `free_space_rejected_fraction`, `sign_mesh_watertight`, face-
fraction, or graph-health-row change while the rendered body or per-query nearest-face is
unchanged.

### 1.4 Kill-test KT-A — observed-only keyboard body topology / free-space

Build the keyboard epoch from **observed depth faces only**, run free-space carving against
observed depth points, and attempt the signed query at the hand region on f30–36,45,46 (the
`direct_visible_pose_observation` frames).

Discriminating predictions:
- **Observed-only ⇒ open sheet.** `sign_mesh_watertight=false` by construction; signed-distance
  candidate count stays ~0; the only defined measurement is **unsigned gap to the top sheet**.
  Expected result: gap ≈ **the KT-1 keyboard-masked value (hand 80–180 mm in front)**. ⇒ the
  honest observed body *reduces to the KT-1 masked-depth surface*; Branch A adds a corrected
  render body but **no new contact channel**. Route: contact stays source-gap-capped
  (`contact_candidate` ceiling), render body repaired, penetration invalidated.
- **If someone forces watertightness** (F-A1): the interior faces used by the signed query
  are `trellis_inferred`/prior; record per-query provenance. Any resulting `penetrating_vertex_count>0`
  is against inferred geometry ⇒ inadmissible. Route: reject as F-A1, keep unresolved.
- **If free-space carving removes the phantom bulge**: the interval barrier penetration at
  f32/f36 recomputes to ~0. ⇒ real invalidation of the false signal (a genuine state change),
  but the contact verdict is unchanged (gap still says hand-in-front).

KT-A passes as a **render/invalidation** win only. It can never pass as a **contact** win on
this clip. If a Branch A report claims contact, it failed KT-A regardless of its schema.

---

## 2. Branch B — hand-depth-bias correction (D4 / workstream E)

Intended: correct the MANO metric depth so the hand can be compared to the keyboard surface.
The observed signature is the 80–180 mm hand-in-front offset plus the agent-judgment
"likely_contact."

### 2.1 What Branch B can legitimately change

Improve hand metric accuracy (wrist/all-joint depth) toward the pipeline's uniform-accuracy
target — a valid D4 goal in its own right. It is **not** valid to justify Branch B *as the
clip001850 contact fix*, because §0 shows the correctable magnitude (25–45 mm) cannot close
the 80–180 mm gap.

### 2.2 Fake-progress modes

- **F-B1 — Circular / contact-forcing correction.** Deriving the depth shift from the
  hand-to-keyboard gap (or magnitude = the gap), then "discovering" the hand now touches the
  keyboard. This uses the contact assumption to build the correction and then reports the
  assumption as a finding. Observable tell: correction magnitude ≈ the keyboard gap
  (80–180 mm), which is 3–5× the measured bias.
- **F-B2 — 2D-blind depth shift.** Reporting the correction validated because 2D reprojection
  stayed low. Per `self_consistency_metrics.md` #1, 2D reprojection is invariant to along-ray
  depth error (its documented blind spot: f231, 38 px with ~25 % depth error). 2D consistency
  is **necessary but not sufficient**; alone it endorses any along-ray shift, including a wrong
  one.
- **F-B3 — Overshoot past the measured bias.** Applying a >45 mm along-ray correction on a
  clip whose measured error scale is 25–45 mm and whose total wrist error is 20–37 mm. This
  injects error larger than the thing being corrected.
- **F-B4 — Verdict laundering.** "Depth corrected → gap smaller → contact." A smaller gap that
  is still > σ_clip is still not contact; and if the correction is unbounded it is F-B1.

### 2.3 Exact observable artifact that proves Branch B changed the problem (not a proxy)

A **re-rendered clip001850 overlay still (f32/f36)** plus its backing per-frame numbers in
which **all** hold:

1. **Projected-size/detected-size ratio moves toward 1.0** (self-consistency metric #2), where
   the correction was derived from **size + 2D + temporal** channels, **not** from the keyboard
   gap.
2. **Cross-detector 2D reprojection residual ≤ its pre-correction value** (metric #1): a pure
   along-ray depth fix must preserve 2D placement; the overlay hand reprojects onto the same
   skin pixels at corrected apparent size.
3. **Correction magnitude ≤ the independently measured bias (≈25–45 mm)**, recorded with its
   derivation, and demonstrably **not equal to the keyboard gap**.
4. **`contact_state` after correction is decided by the residual gap vs σ_clip**, and on this
   clip **remains `contact_candidate`/`unresolved`** (residual gap 40–140 mm) — it does not
   flip to contact.

Proxy that does NOT count: the hand-to-keyboard gap shrinking. That is the quantity being
explained, not evidence the correction is physical. If the reported success metric is "gap
closed," it is F-B1/F-B4.

### 2.4 Kill-test KT-B — hand-depth shift vs 2D reprojection consistency (paired with size)

Measure, **before** applying any correction, on f30–36,45,46: (i) cross-detector 2D
reprojection residual (rtmlib vs MANO projection), and (ii) projected-size/detected-size ratio.
Note z_hand/z_keyboard ≈ 0.435/0.52 ≈ 0.84, so a hand-too-close-by-the-gap would render
**≈1.2× too big**.

Discriminating predictions:
- **Ratio ≈ 1.2 (rendered hand bigger than detected) AND 2D residual already low** ⇒ hand is
  genuinely too close along the ray ⇒ real depth bias with the sign/magnitude to matter.
  A correct along-ray push preserves 2D and drives the ratio → 1.0. This is the *only* pattern
  under which Branch B is a real correction. (Even then it is bounded by the measured
  25–45 mm; it narrows the gap, and contact remains gated on the residual.)
- **Ratio ≈ 1.0 already (rendered size matches detected)** ⇒ the hand is at the correct depth
  *for its apparent size* ⇒ **no along-ray bias of the required magnitude** ⇒ the 80–180 mm
  gap is real hovering or keyboard-depth error. Any depth push then breaks the size ratio
  toward 0.84 to manufacture contact. **Kill the correction; contact = no-contact/unresolved.**
- **2D residual worsens under the shift** ⇒ the "depth fix" carries a lateral component (it is
  chasing the keyboard, not the ray) ⇒ F-B1. Reject.

KT-B is executable now with existing metrics (self_consistency #1 + #2) and does not need GT.
GT (R8) is the tiebreaker if size and 2D are ambiguous.

---

## 3. Cross-cutting kill-test — preserving uncertainty instead of forcing contact

The shared failure surface: after either repair, the residual belief "the hand is obviously on
the keyboard" (agent interaction judgment) pressures a flip to `confirmed_contact`.
`pipeline_invariants.md` is explicit: "A clip can be a correct non-contact artifact; forcing
contact on a slice whose metric source-gap evidence is contact-unlikely is a physical error."

Kill-test KT-U (must pass after **any** Branch A/B change):
- **Provenance-driven, not belief-driven.** `contact_state` may reach `confirmed_contact` only
  via a **floor-independent** channel (object-motion onset time-locked to hand kinematics, or
  R8 GT) — never via a signed-distance threshold (subagent 10 §3/§4). On clip001850 the
  keyboard barely moves under typing, so motion-coupling is expected weak ⇒ the honest ceiling
  is `contact_candidate`/`unresolved` GT-free.
- **Render must show the residual.** The world body stays **hatched/uncertain** where inferred;
  the banner shows the **residual gap and σ_clip**, not a binary "closed." If the re-render
  shows a solid contact mark or a solid green body without a motion/GT row driving it, KT-U
  fails (forced contact).
- **Symmetric outcome legitimacy.** `unresolved_full_frame_depth_leak`, `no_contact_supported`,
  and a persistent `contact_candidate` window are **correct terminal artifacts**, not failures
  to be engineered away.

Observable tell of a violation: `contact_state` transitions to contact with no new
motion-coupling/GT provenance column populated, or the rendered banner drops the residual gap.

---

## 4. Kill-test — is f45 `contact_candidate` a real signal or one-vertex noise?

**Verdict: one-vertex noise on a depth outlier. It is not keyboard contact and must not be
cited as evidence contact exists on this clip.** Re-verified from the on-disk f45 row:

- Under the keyboard-masked + hand-quarantined condition, **12** hand vertices fall in the
  keyboard region; median delta **−85.3 mm** (hand in front). **Exactly one** vertex
  "penetrates," at **+5.86 mm** — sample id 4, a **thumb** vertex (KT-2:
  `penetrating_anatomical_region_counts={thumb:1}`).
- **+5.86 mm ≪ σ_clip (25–30 mm)** → z ≈ 0.2σ. Statistically indistinguishable from zero.
- **The single vertex rides a depth outlier, not the keyboard.** The f45 keyboard-masked
  surface has `surface_depth` min **393.6 mm** vs median **520.0 mm** — a 12.6 cm-closer pixel.
  The penetrating thumb vertex sits at hand cam_z min **399.4 mm**, i.e. it matched that
  393.6 mm outlier pixel (both ~12 cm in front of the true keyboard body at 520 mm). The
  hand-quarantine failed to remove that near-hand/mask-edge pixel; the "penetration" is the
  thumb touching a **depth artifact 12 cm in front of the keyboard**, not the keyboard.
- **It violates subagent 10's own promotion rule.** §2/§3 require, for `contact_candidate`,
  KT-2 temporal coherence (a compact patch persistent across **≥3 consecutive frames**) and
  KT-3 single-surface localization. f45 is a **single vertex, single frame, single anatomical
  point** — zero temporal coherence. By the stated policy it must route to
  `unresolved_incoherent_evidence`, not `contact_candidate`. The kill-test *script* (subagent
  07) promoted it on a bare presence rule (≥1 vertex within 6 mm), which is exactly the
  `.max`/single-vertex artifact the attack forbids.

Route correction: **f45 → `unresolved_incoherent_evidence`** (or `unresolved` with a
`single_vertex_depth_outlier` provenance note). The window's honest contact summary over
f28–48 is **zero admissible contact frames**: `pose_unresolved` ×12,
`geometry_epoch_contaminated` ×7, `full_frame_depth_leak` ×1, and the lone
`contact_candidate` demoted to incoherent. This *strengthens* §0: there is no GT-free contact
signal on this window for either branch to "recover."

---

## 5. Artifact-level decision checklist for subagents 11–13

Apply per branch. **Gate = the rendered still / consumed state, never a schema row.** Any
"support" output (schema, graph-health row, watertight flag, distance number, validator pass)
is inadmissible as the pass criterion (AGENTS.md anti-avoidance invariant).

### G0 — Consumer exists (blocks everything; subagent 06 A5, subagent 10 §9)
- [ ] A renderer (`build_v19_rigid_render_state.py` / `publish_v19_render_artifact.py`, or an
      ego.hoi-consuming path) **reads the new rows and re-renders f32/f36**. A pixel/label
      **diff vs the current published still is nonzero**. If the video is byte-identical, the
      work is backing-data-only → **not progress**, regardless of branch.

### G-A — Object-geometry repair (Branch A)
- [ ] **Render body changed**: keyboard thin-axis extent measured off the rendered mesh
      **≤ ~0.05 m** (from 0.26 m); body reads as a keyboard, not a slab or a prior box.
- [ ] **Not a primitive** (¬F-A2): extents trace to per-frame observed depth, not the prior
      box; verify the fit is not the keyboard prior snapped in.
- [ ] **Per-query provenance**: contact query's nearest face at the hand region =
      `observed_depth_surface`, recorded per vertex (¬F-A1). If `sign_mesh_watertight=true`,
      confirm the interior faces are **not** TRELLIS/prior; if they are, reject.
- [ ] **False penetration invalidated**: interval barrier penetration vs the carved body ≈ 0
      at f32/f36; the barrier's 10.8 cm hand-drag is reverted in the consumed state.
- [ ] **Verdict preserved** (¬F-A4, KT-U): `contact_state` stays `unresolved`/`no_contact_supported`;
      body drawn hatched; residual gap shown. **Contact does not flip.**
- [ ] **KT-A honesty**: report explicitly states Branch A delivered a render/invalidation win,
      **not** a contact channel (nonpenetration is unavailable for this object/viewpoint).

### G-B — Hand-depth-bias correction (Branch B)
- [ ] **Independent derivation** (¬F-B1): correction derived from size + 2D + temporal
      channels; magnitude recorded and **≠ keyboard gap**.
- [ ] **Magnitude bound** (¬F-B3): |along-ray correction| **≤ ~45 mm** (the measured error
      scale); flag and reject anything larger.
- [ ] **KT-B size test**: pre-correction projected-size/detected-size ratio reported. Correction
      applied **only** if ratio ≈ 1.2 (hand genuinely too close) and it drives ratio → 1.0.
      If ratio ≈ 1.0, **no correction** — the gap is hovering/keyboard-depth, contact = no/unresolved.
- [ ] **2D preserved** (¬F-B2): cross-detector 2D reprojection residual after correction
      **≤** before; overlay hand on same skin pixels at corrected size.
- [ ] **Verdict preserved** (¬F-B4, KT-U): residual gap (40–140 mm) vs σ_clip keeps
      `contact_state` at `contact_candidate`/`unresolved`. **Contact does not flip.**

### G-U — Uncertainty preservation (both branches; KT-U)
- [ ] `confirmed_contact` appears **only** with a populated motion-coupling or GT provenance
      column — never from a signed-distance threshold.
- [ ] Rendered banner shows residual gap + σ_clip (not binary "closed"); uncertain body hatched.
- [ ] `unresolved_*` / `no_contact_supported` are accepted as correct terminal outputs.

### G-f45 — Signal admissibility
- [ ] f45 is **not** cited as contact evidence; it routes to `unresolved_incoherent_evidence`
      (single thumb vertex, +5.86 mm ≪ σ_clip, on a 393.6 mm depth outlier, no temporal
      coherence). Window contact summary over f28–48 = **zero admissible contact frames**.

**One-line gate for the main agent:** a Branch-11/12/13 deliverable is progress **iff** the
re-rendered f32/f36 still changes (G0) *and* the change is the branch's real deliverable
(G-A render/invalidation, or G-B size-validated bounded depth fix) *and* `contact_state` did
**not** flip to contact without a floor-independent channel (G-U). A schema/flag/distance
change with an unchanged render, or a contact flip justified by a shrunken gap, is fake.

---

## 6. Residual risks / boundaries of this analysis

- **σ_clip and the 25–45 mm bias scale are from Tracks J/M/V** (`hand_metric_mechanisms.md`),
  measured on clips 1849/1850/1851, not re-derived here. If a Branch B run measures a
  larger *along-ray* clip001850 bias from an independent channel, the §0 magnitude bound
  should be updated from that measurement — but it must be an independent channel, not the
  keyboard gap.
- **The keyboard-depth-error hypothesis is not excluded.** §0 leaves "keyboard depth biased
  far" live alongside "hand hovering." KT-B (size) and, if ambiguous, R8 GT discriminate. Do
  not let Branch A/B silently assume the keyboard depth is correct to justify a hand shift, or
  assume the hand is correct to justify a keyboard shift — that circularity is the core trap.
- **Multi-frame fusion could slightly sharpen the observed top surface** vs single-frame masked
  depth; if it moves the surface by ≫ measured depth noise it is worth re-checking, but it will
  not move it by the 80–180 mm needed to change the verdict.
- **GT exists for this clip** (HOT3D development set) and can adjudicate all of the above
  directly; deferred to R8 by the GT-free-first constraint.
- This is an attack/decision analysis over existing kill-test artifacts; it does not itself run
  Branch A/B. The checklist is the acceptance gate for whoever does.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Attacked both next mechanisms with on-disk-verified numbers. Object-geometry repair: keyboard completed mesh 90892 faces / 95.40% TRELLIS / 3.54% observed / non-watertight / free_space not_evaluated / AABB thin axis 0.260m vs 0.031m prior = 8.4x too thick (summary.json); named fake-progress modes F-A1..F-A4 (watertight-by-completion, primitive substitution, container win, verdict laundering), the exact render-still artifact that proves change, and KT-A (observed-only body reduces to KT-1 masked-depth surface -> render/invalidation win only, never contact). Hand-depth correction: measured error scale 25-45mm << 80-180mm gap; F-B1..F-B4; KT-B pairs 2D-reprojection invariance (metric #1 blind spot) with projected-size/detected-size ratio (metric #2, ~1.2 if too close) as the discriminator; bound |correction|<=45mm. Cross-cutting KT-U (uncertainty vs forced contact) and f45 verdict: one thumb vertex +5.86mm << sigma_clip on a 393.6mm depth outlier (median 520mm), zero temporal coherence -> unresolved_incoherent_evidence, over-promoted by presence rule. Delivered an artifact-level G0/G-A/G-B/G-U/G-f45 checklist for subagents 11-13 with file paths and a one-line gate. Severities: G0 no-consumer CRITICAL; F-A1/F-B1 contact-forcing CRITICAL."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "read PROMPT/EPISTEMIC/OPS/TASK_PACK + subagents 06-10 + project memory (hand_metric_mechanisms, object_geometry_and_render_lessons, pipeline_invariants, self_consistency_metrics)", "result": "passed", "summary": "Loaded task spec, D6 mechanism map, KT-1..KT-7 defs, contact policy, and the two branches under attack."},
    {"command": "python3 inspect /tmp/clip001850_masked_contact_killtests/{summary.json,contact_killtests_frame_detail.ndjson}", "result": "passed", "summary": "Re-verified: 95.40% TRELLIS non-watertight mesh, 8.4x thin-axis inflation; f45 = 1 thumb vertex +5.86mm on 393.6mm depth outlier vs 520mm keyboard median, 12 region verts median -85.3mm; route counts pose_unresolved12/geom_contam7/leak1/candidate1."},
    {"command": "python3 derive keyboard prior extents + gap-vs-bias magnitudes", "result": "passed", "summary": "Confirmed hand 80-180mm in front of keyboard surface vs 25-45mm measured hand-metric scale => 3-5x; neither branch closes contact GT-free."},
    {"command": "git status --short (context only, no edits)", "result": "passed", "summary": "On branch yiwen_research; pre-existing dirty files are not mine; nothing staged; only new file is this findings doc at the authoritative path."}
  ],
  "validationOutput": [
    "Convergence: both branches -> contact UNRESOLVED GT-free; fake-progress trap identical (render/metric change reported as resolved contact).",
    "Branch A honest ceiling: render body repair + false-penetration invalidation; nonpenetration channel unavailable for singly-observed keyboard (observed-only=open sheet, watertight=inferred interior).",
    "Branch B honest ceiling: bounded (<=45mm) depth fix narrows gap by <=~1/2; KT-B size ratio ~1.2 vs ~1.0 discriminates real-bias from forced-contact; 2D reprojection alone insufficient (documented blind spot).",
    "f45 contact_candidate = one-vertex noise on a depth outlier; violates subagent 10's own >=3-consecutive-frame coherence rule; re-route to unresolved_incoherent_evidence.",
    "Checklist gate is the re-rendered f32/f36 still (G0), never a schema/flag/distance proxy."
  ],
  "residualRisks": [
    "sigma_clip / 25-45mm bias scale are prior measurements (Tracks J/M/V), not re-derived here; a larger independently-measured along-ray bias would relax the Branch B magnitude bound but must not be sourced from the keyboard gap.",
    "keyboard-depth-error hypothesis remains live alongside hand-hovering; KT-B size and R8 GT discriminate; branches must not assume one surface correct to justify shifting the other (core circularity).",
    "KT-7/G0 consumer still does not exist in the v19 render path per subagents 06/10; until it lands every Branch 11-13 contact_state is backing-data-only.",
    "This is decision analysis over existing kill-test artifacts; it does not execute Branch A/B - the checklist is the acceptance gate for whoever runs them.",
    "HOT3D GT exists for clip001850 and can adjudicate directly; deferred to R8 by the GT-free-first constraint."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one task-memory findings document at the authoritative subagents path (14_next_mechanism_attack.md). No source files touched; nothing staged.",
  "reviewFindings": [
    "blocker (CRITICAL): no renderer consumes ego.hoi rows (subagents 06 A5 / 10 s9); until a consumer re-renders f32/f36 with a nonzero pixel/label diff, both branches are backing-data-only, not progress. Gate G0.",
    "blocker (CRITICAL): Branch A 'observed-face watertight sign mesh' is self-contradictory for a singly-observed keyboard - watertightness requires inferred interior faces (F-A1); any resulting penetrating_vertex_count>0 is contact against invented geometry. Nonpenetration is not an available contact channel here.",
    "blocker (CRITICAL): Branch B correction magnitude must be <=~45mm (measured error scale) and derived from size/2D/temporal, never the 80-180mm keyboard gap (F-B1 contact-forcing); the gap shrinking is not evidence the correction is physical.",
    "major: f45 contact_candidate is one thumb vertex (+5.86mm << sigma_clip 25-30mm) on a 393.6mm depth outlier (keyboard median 520mm) with zero temporal coherence; it violates subagent 10's >=3-consecutive-frame promotion rule and must route to unresolved_incoherent_evidence. Window f28-48 has zero admissible contact frames.",
    "major: both branches converge on contact UNRESOLVED; grading either on a render change (A) or metric-accuracy change (B) as 'resolved contact' is verdict laundering (F-A4/F-B4). contact_state may reach confirmed_contact only via motion-coupling or GT (KT-U).",
    "note: object-geometry repair done honestly reduces to the KT-1 keyboard-masked depth surface already computed (hand 80-180mm in front); its real deliverable is a corrected render body + invalidation of the false interval penetration, not a new contact channel."
  ],
  "manualNotes": "Core reframe for the parent: the next two mechanisms are being positioned as 'the thing that finally resolves clip001850 contact.' They cannot. Object-geometry repair delivers a corrected render body and kills the false 82-107mm interval penetration, but its contact-relevant surface is the observed keyboard top that KT-1 already queried (hand 80-180mm in front) - nonpenetration is simply unavailable for a singly-observed keyboard because watertightness demands inferred interior faces. Hand-depth correction is bounded by the measured 25-45mm error scale, 3-5x too small to close the 80-180mm gap; the only correct correction pattern is when projected-size/detected-size ratio is ~1.2 (hand genuinely too close), validated by 2D-reprojection invariance and driven to ~1.0 - if the ratio is already ~1.0, the gap is real hovering or keyboard-depth error and any shift forces contact. f45 is one-vertex noise. Grade subagents 11-13 on the re-rendered f32/f36 still (G0) and whether contact_state stayed unresolved without a floor-independent channel (G-U), not on schema rows, watertight flags, distances, or a shrunken gap."
}
```
