# Subagent 9 — Research-track auto-research operating guide

## Scope and reader

Durable operating guide for **Karpathy-style auto-research on the research track only** — HOI /
physical-state extraction: object geometry, object pose, rigidity, contact, occlusion ownership,
nonpenetration, factor-graph redesign, and the offline approximation/distillation families
(R6/R7, subagents 10/11). It is **distinct from the delivery-track guide**
(`06_delivery_autoresearch_guide.md`): delivery ratchets a five-axis GT-free/self-consistency +
HOT3D-hand vector on a frozen eval set under a same-order-of-magnitude runtime invariant; the
research track ratchets a **different metric vector** (object pose/shape/contact/rigidity/
occlusion), mostly open scientific problems on a tiny HOT3D fixed slice plus lockbox, on
**offline A800 branches under its own run root**, behind a promotion gate nothing crosses until
five evidence conditions close. Anchored to `07_v19_bottlenecks_and_hoi_research.md`
(B1–B19, RQ1–RQ10, E1–E9), `08_hoi_output_and_metrics.md` (ontology + metric vector), and
`RESEARCH_TRACK_TASK_PACK_DRAFT.md` (R0–R7).

**Central adaptation problem.** AutoResearch is safe because it optimizes a single scalar
(`val_bpb`) on a read-only evaluator over held-out data. The research track has neither a single
scalar nor a clean held-out set. Its load-bearing metrics are GT residuals on a **three-clip
benchmark that overfits even/odd** (B18) and a family of GT-free self-consistency quantities
that are the **same currency as delivery** — and therefore just as gameable, plus structurally
empty under manipulation occlusion for the contact term (B3: 900/900 rows zero support). The
field's SOTA for the headline claim (category-agnostic monocular object pose) is itself open
(BOP-H3 novel-object RGB: GigaPose 9.4 AP), so any "improvement" can be movement along a
symmetry/gauge null space rather than real error reduction. Every design choice below exists to
keep the ratchet honest under those modes. Skip the protected-evaluator, lockbox, proxy-capture,
and visual-veto design in §2 and the loop will tighten a residual that the held-out clip
contradicts — the exact failure the v19 ledger (P46/P50 proxy-capture, P48/P49 falsified pose
repairs) already documents twice.

## 1. Philosophy translated to the research track

**1.1 Program the spec in English.** The durable artifact of a run is a **mechanism spec**
(R1–R7): a named mechanism family, the bottleneck it attacks (B#), the experiments that test it
(E#), the discriminating predictions declared before each run, and the redirect rule per
outcome. The research code is editable; the spec is the control plane.

**1.2 Become one with the render before touching code.** The HOI artifact is the full-duration
`hoi_overlay.mp4`/`hoi_side_by_side.mp4` driven by `hoi_overlay_events.ndjson` layers pointing
at table rows (subagent 8). Before any change, consume the current HOI render as an annotation:
which object is rigid, where the contact patch sits, whether the drift arrow matches the visible
slip, whether pose axes lie flat on the keyboard or float. A residual read without watching the
render is the research version of training without looking at the data.

**1.3 Ratchet: propose → offline run → measure protected vector → keep-if-better else `git reset`.** Same core loop, three research modifications (§2): the protected metric is a **vector over the HOI ontology**, not a scalar; acceptance requires **GT residual and proxy residual to move together** (proxy-only motion is proxy capture — P46/P50); runs are **offline, hours-per-clip allowed** on A800 branches under the runtime invariant's research exemption.

**1.4 Protect the evaluator; never let the loop edit the thing it is scored on.** Protected surface: frozen fixed slice (HOT3D 001849/50/51), **≥2 lockbox clips named before any tuning** (B18), GT sidecars (HOT3D CAD + poses + proximity-derived contact), the P37 constant-gauge evaluator, and the GT-free self-consistency harness **shared one-implementation with delivery**. The **drift-latent b(t) spline that closes sub-10 mm when GT-fit (Track V)** is itself an evaluator-cheat risk: any loop edit that reads GT to fit per-clip spline knots is editing the evaluator through a side channel. GT is read-only, even for variables GT can represent.

**1.5 Add one measurement channel at a time; prefer the simpler mechanism.** The v19 diagnosis (subagent 7 §0): the joint layer is *starved of measurements*, not short of factors. Sequencing: priority-1 drift latents and priority-4 noise calibration start immediately; priorities 2–3 (contact mixture, occlusion-surviving contact channels) only pay off after correspondence exists. Do not ship a switchable contact factor before E6 certifies that any contact channel transfers to the non-grasp regime.

**1.6 Never stop, but stay scoped — proxy capture and benchmark overfit are the dominant failures.** Cerebras' "environment and task framing > model choice" maps directly. Autonomy is throttled (Karpathy's slider): run unattended on tightly scoped, low-regime-risk families (E1 rigidity probe, E4 term-efficacy audit); keep the advisor in the loop where the claim is load-bearing (E2 pose, E5 drift latent, E7 contact switch) or where the metric is most gameable (contact AUROC on proximity-derived truth).

## 2. The research auto-research operating loop

### 2.1 Three-surface architecture

| AutoResearch surface | Research-track equivalent | Rule |
|---|---|---|
| `program.md` | **`mechanism_spec.md`** — one per R1–R7 family. Names B#, RQ, E# experiments, protected vector entries, discriminating predictions, lockbox, redirect rule. | Human + agent edit. |
| `train.py` | **Research stage code** for that family (track lifter, Procrustes solver, TSDF fuser, contact switch, drift-latent block, FoundationPose/SpatialTrackerV2 probe). Editable; category-agnostic. | Editable. |
| `prepare.py` + data | **Protected evaluation bundle** — frozen fixed slice, frozen ≥2 lockbox, frozen HOT3D GT, P37 constant-gauge evaluator, shared self-consistency harness, mandatory HOI render-consumption checklist. | Read-only to the loop. |

### 2.2 The protected metric vector (research gate; from subagent 8)

Acceptance is a **vector** over the HOI ontology, not a single residual.

| Family | Primary gate metric | Anchor | Mandatory complement (anti-gaming) |
|---|---|---|---|
| Object pose (R1/E2/E9) | `object_pose_translation_residual_gt_mm`, `_rotation_residual_gt_deg` | HOT3D GT, P37 gauge | `object_silhouette_iou`; `primary_observation_channel==texture_tracks` for previously null-space DOFs (B1) |
| Object shape (R2/E3) | `object_chamfer_visible_mm`, `silhouette_iou_p50` | HOT3D CAD | `observed_face_fraction`; no chamfer claim on prior faces |
| Rigidity (R1/E1) | `rigidity_procrustes_residual_mm`, `pairwise_distance_drift_mm` | GT-free | depth-noise floor (B16); `declared_vs_measured_agreement` |
| Contact (R3/E6/E7) | `contact_auroc_vs_proximity`, switch posterior | HOT3D proximity | `contact_evidence_support_fraction`; per-channel σ (B7); domain-transfer (B17) |
| Occlusion (R3) | depth-order render-and-compare; `visibility_state` consistency | GT-free | ownership through full occlusion stays `unresolved` unless depth+temporal |
| Hand correction (R4/E5) | `post_correction_gt_mm`; `anchor_residual_*` | HOT3D GT (never fit) | `anchor_residual_correlation_with_gt_drift`; `applied_to_delivery=false` |
| Graph health (R4/E4) | `solver_liveness_diff_count`; term support fraction; gauge count | structural | zero-diff ⇒ inert; missing gauge ⇒ unclaimed |

**Acceptance rule (replaces single `val_bpb` compare):**

1. **Target family gate metric improves on the fixed slice** beyond a noise band measured from ≥3 baseline reruns (E1/E2 baselines 38.85/77.1 mm are the standing reference).
2. **GT residual and proxy residual move together.** Proxy-only motion (silhouette down, GT flat/up) is **proxy capture** — rejected, routed by §5.4. The P46/P50 lesson encoded as the ratchet.
3. **No other protected family regresses** beyond its noise band.
4. **Lockbox holds.** Improvement must reproduce on the ≥2 lockbox clips named before tuning (B18). Fixed-slice-only is overfit.
5. **Visual-truth veto.** Human (or VLM verifier as second path, never sole truth) reviews full-duration `hoi_overlay.mp4`. If rendered HOI marks contradict visible evidence — contact patch floating, pose axes through the keyboard, drift arrow opposite the slip — reject regardless of metrics.
6. **Gauge and liveness invariants hold.** Declared gauge; `solver_liveness_diff_count > 0`; `promotion_status` unchanged (`not_promoted`).

Only when 1–6 hold does the change advance the branch. Otherwise `git reset`.

### 2.3 The loop (per family, per A800 branch)

All model inference, optimization, and offline reconstruction on server/A800, never the user
workstation; orchestrate via tmux/job sentinels; **no `sleep`/polling** (AGENTS runtime).

```
SETUP (human + agent, once per family):
  - Agree run tag; branch research-ar/<family>-<date> from current research base.
  - Read mechanism_spec.md + editable stage code + protected bundle.
  - Confirm fixed slice + >=2 lockbox resolve; confirm GT sidecars + P37 evaluator;
    confirm shared self-consistency harness version matches delivery (the bridge).
  - Baseline: run current research pipeline UNMODIFIED on fixed slice + lockbox;
    compute full HOI metric vector + capture hoi_overlay.mp4. Write baseline ledger row.

LOOP FOREVER (until human interrupt or family close):
  1. Read git state + last N ledger rows (near-misses, discards, crashes, proxy-capture).
  2. Write causal experiment card (section 4) BEFORE editing: bottleneck -> physical variable
     -> mechanism -> discriminating predictions (>=3) -> accept/reject -> redirect per outcome.
  3. Edit ONLY editable stage code for this family. One measurement channel per iteration.
  4. git commit the code change.
  5. Launch on A800 in tmux, output -> run.log, durable sentinel, own run root.
     Do NOT sleep-wait: start next card or do immediate non-blocking status check.
  6. Read hoi_validation_metrics.parquet + GT evaluator. Missing/empty -> crash (tail log);
     solver_liveness_diff_count==0 -> flag inert (B10), do NOT trust objective.
  7. Apply section-2.2 rule (vector + GT/proxy co-motion + lockbox + visual veto + gauge/liveness).
  8. Log ledger row (keep/discard/crash/inert/proxy-capture). Ledger git-untracked.
  9. Accepted -> advance branch. Rejected -> apply card redirect rule WITHIN family;
     git reset only the code, preserve negative-information ledger row.
  10. Family close: predictions met on fixed slice + lockbox -> promotion-gate evidence;
      or redirect tree exhausted -> record falsification, do NOT hop families.
```

**When stuck, think harder before rewinding.** Re-read the mechanism spec and E# predictions;
combine near-misses; escalate to a more radical **in-family** mechanism (translation-only →
add rotation; 2D → 3D tracks; soft → switchable contact). Never abandon a mechanism because one
implementation was inert (AGENTS); never hop to an unrelated family (Cerebras).

**Worse-before-better exception.** Multi-frame TSDF fusion (E3) may transiently worsen chamfer
while E2 poses stabilize; switchable contact (E7) may flip 001849 from `none` to `near` before σ
calibration lands. Handle explicitly: human opens a **named family experiment** with bounded
transient-regression budget on a declared subset, evaluated on the **full vector at family
close**, not per micro-step.

### 2.4 Distributed families

R1–R7 are sequenced, not independent. Parallelism is *within* the dependency DAG: E4 (instrument)
runs first; E1→E2→E3 form the rigid-body chain; E5 (drift latent) is parallel and
delivery-coupled; E6→E7 form the contact chain. Cross-family monotonicity enforced at merge:
recompute full HOI vector across all families on fixed slice + lockbox before any merge.

## 3. Dataset / task queue templates

### 3.1 Protected eval bundle

```yaml
eval_bundle:
  id: research_hoi_eval_v1
  frozen: true
  fixed_slice: [hot3d_001849, hot3d_001850, hot3d_001851]  # P37 constant-gauge template
    # 001849: 0 proximity rows (P51) contact-negative anchor
    # 001851: 205/300 proximity rows <1 sigma (P52) near-contact anchor
  lockbox: [hot3d_lockbox_A, hot3d_lockbox_B]   # named BEFORE any tuning (B18)
  regime_transfer: [task5_tomato_960, trash_1050, phone_calculator]  # GT-free only
  multi_view_probe: [hot3d_001851_streamB]      # B19
  contact_gt_sources: [hot3d_proximity_derived, egopressure_external]  # B17
  shared_self_consistency_harness: { version: pinned_with_delivery }
  metric_coverage_check: true
```

### 3.2 Experiment queue (tied to E1–E9)

Priority follows AGENTS anti-avoidance: hardest essential root blocker first (B1–B4).

| ID | Family | Exp | Pri | Bottleneck | Gate metric | Monotonic guard | Predictions-redirect (summary) | Deps |
|---|---|---|---|---|---|---|---|---|
| RR-004 | graph_health | E4 | 0 | B7/B10 | solver_liveness_diff_count, term_support_fraction | (instrument) | instrument everything consumes | — |
| RR-001 | rigidity | E1 | 1 | B5/B16 | rigidity_procrustes_residual_mm | depth_noise_floor | keyboard le15mm→adopt; all≈noise→E8 first; short tracks→re-detection | — |
| RR-002 | object_pose | E2 | 2 | B1 | pose_translation_residual_gt_mm | silhouette_iou, primary_observation_channel | drops below 38.85/77.1→correspondence; along-plane only→E5-object; no-improve→gauge/symmetry | RR-001 |
| RR-005 | hand_correction | E5 | 2 | B4 | post_correction_gt_mm | applied_to_delivery=false, anchor_residual_correlation | 20-37→10-15→delivery mechanism; lateral-only→add depth anchor; no-improve→measure anchor/GT-drift correlation | — |
| RR-003 | object_shape | E3 | 3 | B2/B11 | object_chamfer_visible_mm | observed_face_fraction | fusion beats anchor→adopt; hidden error persists→uncertainty band; fusion worse→loop E2 | RR-002 |
| RR-006 | contact | E6 | 3 | B3/B17 | contact_auroc_vs_proximity | contact_evidence_support_fraction, domain_transfer | high AUROC→E7; FP on 001849→training data; chance→depth-order+motion-onset only | — |
| RR-007 | contact | E7 | 4 | B6 | switch_posterior_reproduces_P51_P52 | metric_MANO preserved (P18b), contact_flicker_rate | reproduces→adopt; collapses→rescale prior; flips→hysteresis | RR-006 |
| RR-008 | substrate_probe | E8 | 4 | B12/B19 | runtime_table, camera_vs_gt | — | parity→D1 viable; worse on egocentric→teacher-only | — |
| RR-009 | object_pose | E9 | 4 | B15 | pose_translation_residual_gt_mm | primary_observation_channel | beats baselines→our stage is limit; comparable→single-view info budget→E8/multi-view | — |

### 3.3 Experiment ledger (git-untracked `results.tsv`)

```
commit  family   exp  pose_t_mm  chamfer_mm  rigidity_mm  contact_auroc  wrist_post_mm  liveness  lockbox  visual_veto  status      description
a1b2c3d graph    E4   77.1       n/a         n/a          n/a            37.0           0         n/a      n/a          inert       baseline; zero-diff solver flagged
b2c3d4e rigidity E1   77.1       n/a         12.4         n/a            37.0           1         pass     pass         keep        keyboard le 15mm; build into branch
c3d4e5f pose     E2   41.2       n/a         12.4         n/a            37.0           1         FAIL     pass         discard     fixed-slice only; lockbox regressed -> overfit
d4e5f6g pose     E2   33.8       n/a         12.4         n/a            37.0           1         pass     pass         keep        tracks observe null-space; co-motion GT+silhouette
e5f6a7  contact  E6   n/a        n/a         n/a          0.61           37.0           1         n/a      n/a          proxy-only  chance-level on 001849 -> route depth-order+motion-onset
```

Row `c3d4e5f`: canonical lockbox rejection. Row `e5f6a7`: canonical proxy-capture redirect
(image channel uninformative, mechanism preserved, redirect in-chain).

## 4. Metric-first experiment templates

### 4.1 Card skeleton (all families)

```
CARD id / family / exp / priority
BOTTLENECK (B#): <named blocker>
DEFECT (in rendered HOI artifact): <what hoi_overlay/rows get wrong>
PHYSICAL VARIABLE: <null-space pose, single-anchor shape, drift latent, switch>
MECHANISM HYPOTHESIS: <cause->effect chain>
COUPLING: <why the edit touches that variable>
GATE METRIC (subagent-8 field): <primary>
MONOTONIC GUARD SET: <families that must not regress>
LOCKBOX: <clips that must hold>
DISCRIMINATING PREDICTIONS (>=3):
  - mechanism dominant: <gate improves AND lockbox holds AND proxy co-moves AND render improves>
  - measurement weak/miscoupled: <gate flat, or proxy-only motion, or fixed-slice-only>
  - another variable dominates: <named other metric moves; named redirect>
ACCEPT/REJECT: section-2.2 rule.
REDIRECT PER OUTCOME (within family): <next card per outcome>
EVAL CLIPS / COMPUTE / DEPENDENCIES / DEFAULT_PATH_VS_OFFLINE
```

### 4.2 Object pose (E2/E9 — load-bearing scientific blocker)

BOTTLENECK B1. DEFECT: keyboard 38.85 mm (001851)/77.1 mm 4.93° (001849); null-space motion
invisible to depth-ICP. GATE: pose_translation_residual_gt_mm (P37 gauge); rotation_residual_gt_deg.
GUARD: object_silhouette_iou; primary_observation_channel==texture_tracks for previously null
DOFs. LOCKBOX: lockbox_A/B. PREDICTIONS: (i) correspondence-dominant → translation drops below
baselines AND rotation stabilizes without P48 penalty AND IoU holds AND lockbox holds; (ii)
depth-scale-bias-dominant → along-image-plane improves, depth-axis persists → couple E5-object;
(iii) no-improvement with healthy tracks → gauge/symmetry → inspect residual axes vs keyboard
symmetry. REDIRECT: each outcome maps to a named next card within R1; never hop to contact or
distillation.

### 4.3 Drift latent (E5 — delivery-coupled)

BOTTLENECK B4. DEFECT: wrist 20–37 mm; GT-fit b(t) closes sub-10 but is not deployable. GATE:
post_correction_gt_mm (HOT3D wrist, NEVER fit); anchor_residual_px. GUARD:
applied_to_delivery==false; anchor_residual_correlation_with_gt_drift. LOCKBOX: lockbox_A/B.
PREDICTIONS: (i) anchors observe drift → 20–37→10–15 mm AND correlation>0.5 AND lockbox holds →
this is the delivery hand-accuracy mechanism; queue promotion-gate evidence; (ii) lateral-only →
add size-consistency-weighted depth anchor; (iii) no-improvement vs Track V sub-10 → anchors
noise-dominated → measure anchor/GT-drift correlation to find which anchor has signal. REDIRECT:
stay in R4; this experiment's success criterion is simultaneously the delivery promotion
pipeline's first test.

### 4.4 Contact (E6/E7 — most gameable)

BOTTLENECK B3+B6+B17. DEFECT: 900/900 rows zero support; manual slice attribution. GATE (E6):
contact_auroc_vs_proximity; (E7): switch_posterior_reproduces_P51_P52. GUARD:
contact_evidence_support_fraction; per-channel σ (B7); metric_MANO preserved (P18b). LOCKBOX:
lockbox_A/B. PREDICTIONS (E6): (i) high AUROC both clips → E7 justified; (ii) systematic FP on
001849 (hover misread) → grasp priors don't transfer → training-data work (EgoPressure) before
any factor ships; (iii) chance-level → image channel uninformative → depth-order+motion-onset
only; mechanism preserved, redirect in-chain. PREDICTIONS (E7): (i) reproduces P51/P52 → adopt
discrete-continuous; (ii) collapses to none → prior mis-scaled, E4/E6 σ say which; (iii) flips
per-frame → hysteresis prior. REDIRECT: never ship a contact factor before E6 certifies transfer
(B17); never claim nonpenetration on non-watertight meshes (B11).

## 5. Failure triage rules

**5.1 Crash / missing artifacts.** Read `tail -n 50 run.log`. Trivial → fix/rerun. Mechanism
broken → log `crash`, revert. **Infrastructure failure (env/CUDA/symlink/stale-sidecar) is NOT
an experiment result** — record phase blocker, stop lane; do not let the loop provision
infrastructure (AGENTS role separation; Cerebras' biggest sink was sandbox/GPU permissions, not
research).

**5.2 Solver liveness == 0 (`differs_from_input` false).** Inert (B10). Flag, do not trust the
objective, do not compare metrics. Route: inspect term support rows; if zero-support, variable
is in a null space → add missing measurement channel, do not retune weights.

**5.3 Gate improved, GT flat or up → proxy capture (most important triage).** Silhouette down,
chamfer up (P47); source-gap down, GT up (P46); contact AUROC up on 001849 only via hover FP.
**Reject.** Route by §5.7. External echo: DexMan on proximity-reward proxy capture.

**5.4 Fixed-slice improves, lockbox fails → overfit (B18).** **Reject.** Route: either mechanism
overfit (simplify; reduce dof) or regime doesn't transfer (B15) — declare which before retrying.
Never claim promotion on fixed-slice-only evidence.

**5.5 Gate flat, nothing moved → measurement weak/miscoupled, not "it doesn't work."** Did the
edit reach the physical variable? Did the solver consume it (liveness)? Did the render expose
it? Re-examine coupling before discarding the *mechanism* — only the *implementation* failed.

**5.6 Improvement only on a known structural defect (zero-support contact rows "fixed"
cheaply).** Without a named cause, gaming the aggregate (Cerebras). Require the mechanism;
keep the negative-information ledger row.

**5.7 Routing table:**

| Symptom | Route to | Do NOT |
|---|---|---|
| Pose residual bad, silhouette/mask good | tracking/pose stage (E2/E9) | retune shape prior |
| Pose residual good, silhouette bad | shape/mask/camera substrate | claim pose win |
| Along-image-plane improves, depth-axis persists | depth-scale bias; couple E5-object | declare translation win |
| No improvement with healthy tracks | gauge/symmetry; inspect residual axes | add more pose terms |
| Contact AUROC up, 001849 hover FP up | grasp-prior domain gap (B17); training data | ship contact factor |
| Switch flips per-frame | hysteresis prior on switch | smooth in post |
| Rigidity residual ≈ lifted noise on all objects | depth substrate (E8 prerequisite) | declare rigid/deformable |
| Chamfer visible good, hidden bad | irreducible uncertainty; render as band | claim full-body chamfer |
| Wrist improves fixed-slice only | overfit (B18); simplify latent dof | promote to delivery |
| Liveness == 0 | add measurement channel / fix solver | trust objective |
| Non-watertight mesh | nonpenetration `unresolved` (B11) | sign-clamp penetration |

**5.8 Stuck.** Think harder within the family before rewinding: re-read mechanism spec + E#
predictions, combine near-misses, escalate to a more radical in-family mechanism. Rewinding is
a last resort. Never jump to an unrelated family.

## 6. Anti-patterns specific to the research track

Severity: **[blocker]** invalidates the run; **[high]** corrupts the metric; **[medium]** wastes
compute or violates scope.

**R1 [blocker] Proxy capture — proxy improves while GT regresses or stays flat.** P46/P50/P47.
Guard: §2.2 rule 2; every card declares the proxy complement.

**R2 [blocker] Fitting the drift latent / calibration to GT (evaluator cheat).** Track V's GT-fit
b(t) closes sub-10 mm; any loop edit that reads GT to set spline knots or per-clip scale edits
the evaluator through a side channel. Guard: GT sidecars read-only; deployable claims require
GT-free anchors validated on held-out clips (E5).

**R3 [blocker] Fixed-slice-only improvement without lockbox (overfit).** B18. Guard: §2.2 rule
4; ≥2 lockbox clips named before any tuning.

**R4 [blocker] Inert solver accepted as inference.** B10. Guard: standing
`solver_liveness_diff_count > 0` assert from E4; zero-diff runs flagged, not compared.

**R5 [blocker] Nonpenetration claim on non-watertight / prior-completed meshes.** B11. Guard:
`min_signed_penetration_mm` null + nonpenetration `unresolved` when `!watertight` or face label
is `prior-completed`.

**R6 [blocker] HOI tables without full-duration render.** Subagent 8: short windows/contact
slices/debug clips are QC only. Guard: full-duration `hoi_overlay.mp4`/`hoi_side_by_side.mp4`
mandatory.

**R7 [high] Gauge motion masquerading as improvement.** Track B Umeyama lesson; per-clip world
similarity, object canonical frame, camera-convention rotation are unfixed freedoms. Guard:
declared gauge per experiment; P37 constant-transform evaluator template.

**R8 [high] Offline-method creep into the default path.** HOLD/MagicHOI/BundleSDF-class runtimes
violate the runtime invariant. Guard: every card declares DEFAULT_PATH_VS_OFFLINE; teachers never
ship as stages; promotion requires runtime within delivery budget.

**R9 [high] Contact factor shipped before transfer test.** B17. Guard: E6 AUROC gate; EgoPressure
external reference before any contact factor ships.

**R10 [high] Declaring contact/nonpenetration through full occlusion.** B3. Guard: occlusion row
exposes channels; `visibility_state` stays `unresolved` unless depth-order + temporal evidence.

**R11 [high] Abandoning a mechanism because one implementation was inert.** AGENTS. Guard: §2.3
"when stuck"; redirect within family; preserve negative-information ledger row before reset.

**R12 [high] Hopping to an unrelated family to manufacture motion.** Cerebras. Guard: one family
per loop; pinned gate metric; redirect rule stays in-family.

**R13 [medium] Category/if-else branches to pass specific clips.** AGENTS methodology. Guard:
case variation in model outputs, one category-agnostic path.

**R14 [medium] Heavy compute on the local workstation; `sleep`/polling as progress.** AGENTS
runtime. Guard: A800/offline branches only; tmux/job sentinels; no `sleep`.

**R15 [medium] Reporting ledger/keep-count as the deliverable.** AGENTS work-progress. Guard:
report which physical mechanism changed, which HOI rendered/geometric evidence changed, which
protected vector family improved on fixed slice + lockbox.

**R16 [medium] Mutating delivery semantics through the research sidecar.** Subagent 8 isolation.
Guard: research reads delivery substrates read-only; absence of extension leaves delivery
byte-identical; promotion is the only door (§7).

## 7. Promotion gate (the only door back to delivery)

A research mechanism enters `ego.delivery.output` only with ALL of (subagent 8 §10; task pack):

1. **GT win** on fixed slice + ≥2 lockbox clips, declared predictions met (not narrated after).
2. **GT-free self-consistency win** on delivery-regime clips, shared metric family (one
   implementation, versioned with delivery — the bridge).
3. **Runtime within delivery budget** (≤2.45 GPU-h/video-h lane share; same-order-of-magnitude
   as input duration as default path).
4. **Regime-transfer evidence** (HOT3D↔egoscale; B15/B18).
5. **Implementation-class review**: solver liveness, freshness boundary, gauge declaration, mask
   ownership, watertightness (B9/B10/B11/B13).

Until all five close, the mechanism ships nowhere, `promotion_status` stays `not_promoted`, and
**delivery milestones never depend on a research outcome**. Promotion triggers a delivery
minor-version bump with the new field under a delivery-owned namespace (no base-field mutation).
The drift latent (E5) is the designed first test of this pipeline.

## 8. How negative results redirect within the same mechanism chain

The ratchet's honesty depends on negative results being informative, not terminal. Each E#
declares ≥3 discriminating outcomes **before** the run, and each maps to a named next card
within the same family:

- **E1 rigidity.** All-objects-≈-lifted-noise → depth substrate is the blocker → E8 becomes
  prerequisite; rigidity mechanism preserved, not abandoned.
- **E2 pose.** Along-image-plane-only → depth-scale bias → couple E5-object variant (same pose
  family, not a hop to contact).
- **E3 shape.** Fusion worse than anchor → E2 poses insufficient → loop back to E2 with the
  measured pose-error budget fusion requires.
- **E5 drift latent.** Lateral-only → add size-consistency-weighted depth anchor (same family).
  No improvement vs Track V sub-10 → measure anchor/GT-drift correlation (same family, sharper
  instrument).
- **E6 contact.** Chance-level AUROC → image channel uninformative → depth-order + motion-onset
  channels (same family, different channel). Systematic FP on 001849 → training-data work item
  (B17), not mechanism abandonment.
- **E7 switch.** Collapses to none → prior mis-scaled → E4/E6 σ recalibration (same family).
  Flips per-frame → add hysteresis prior (same family).

A negative result that **cannot** map to a named next card within the family is recorded as a
falsification (the v19 ledger discipline: P46/P48/P49/P50 each falsified a specific pose repair
and each redirected the model). It never justifies hopping to an unrelated mechanism.

## 9. Residual risks

1. **Protected evaluator bundle does not yet exist end-to-end.** Building frozen fixed-slice +
   lockbox + GT + P37 evaluator + shared self-consistency harness is itself research work
   (R0/R5); until it exists, the loop has no honest gate and must not run unattended.
2. **Lockbox clip provenance unresolved.** ≥2 HOT3D lockbox clips must be named and GT-curated
   before any tuning (B18); until then every fixed-slice win is overfit-suspect.
3. **Contact GT is proximity-derived, not patch/pressure truth.** AUROC against proximity is a
   proxy until EgoPressure (or equivalent) is consumed; contact claims inherit that ceiling (B17).
4. **Novel-object egocentric RGB pose is open** (BOP-H3 GigaPose 9.4 AP). Pose "wins" may reflect
   symmetry null-space motion; gauge/observability fields expose this, not eliminate it.
5. **Greedy ratchet under-explores.** Loop favors local wins and will not invent a novel
   mechanism a human researcher would reach; direction-setting stays human, especially for R6
   (RL) and R7 (distillation), which are gated on validated research outputs.
6. **Offline teacher creep.** HOLD/MagicHOI/FoundationPose-class runtimes can silently become
   default-path substrates if the runtime declaration is not enforced per card (R8).

## References

- Karpathy, "A Recipe for Training Neural Networks," 2019. https://karpathy.github.io/2019/04/25/recipe/
- karpathy/autoresearch, `program.md` (March 2026). https://github.com/karpathy/autoresearch/blob/master/program.md
- Fortune, "Why everyone is talking about Andrej Karpathy's autonomous AI research agent," Mar 2026. https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/
- NextBigFuture, "Andrej Karpathy on Code Agents, AutoResearch and the Self Improvement Loopy Era," Mar 2026. https://www.nextbigfuture.com/2026/03/andrej-karpathy-on-code-agents-autoresearch-and-the-self-improvement-loopy-era-of-ai.html
- DataCamp, "A Guide to Andrej Karpathy's AutoResearch." https://www.datacamp.com/tutorial/guide-to-autoresearch
- Cerebras, "How to stop your autoresearch loop from cheating," Mar 2026. https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating
- Karpathy, "Software Is Changing Again" / autonomy-slider, YC AI Startup School, Jun 2025.
