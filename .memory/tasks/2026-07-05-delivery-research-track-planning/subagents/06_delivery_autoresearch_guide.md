# Subagent 6 — Delivery-track auto-research guide

## Scope and reader

This is a durable operating guide for running **Karpathy-style auto-research on the delivery track only** (head/camera + hand accuracy, visible-drift fixes, fine-grained 2–3 s semantic captioning, ~10 000 video-hours/week API). It excludes HOI/factor-graph work; those belong to the research track (subagent 9's guide). The reader is the delivery agent(s) and the human "research advisor" who sets direction and reviews. It is written so that section 2's spec skeleton can be lifted almost verbatim into a `program.md`-style instruction file per delivery axis.

The guide is grounded in two Karpathy sources: the disciplined empirical **"Recipe for Training Neural Networks"** methodology[^recipe], and the **AutoResearch** autonomous propose→run→measure→ratchet loop[^ar-repo][^ar-fortune][^ar-nbf][^ar-datacamp]. It is anchored to this project's own artifacts: the delivery API output schema (subagent 4, `04_delivery_api_output_format.md`), the GT-free self-consistency metric set (subagent 5, `05_delivery_self_consistency_metrics.md`), and the serving/throughput constraints (subagent 2, `02_pytorch_api_serving_research.md`).

**The central adaptation problem, stated up front.** AutoResearch is safe because it optimizes a *single scalar (`val_bpb`) computed by a read-only evaluator (`prepare.py`) on held-out data*[^ar-repo]. The delivery track has no single scalar: it has five metric families (head/camera, hand, visible-drift, caption, throughput), most of which are **GT-free self-consistency proxies that can be gamed** — a 2D-consistent but wrong-depth hand passes H3, over-smoothing passes jitter while visibly lagging, VLM consensus preserves a shared hallucination, and GT-fitted per-clip calibration already produced a fake sub-10 mm hand result (subagent 5, residual risk 2). The real deliverable is a **rendered video annotation whose visible marks are driven by real mechanisms** (AGENTS.md), not a loss number. Every design choice below exists to keep the ratchet honest against those gaming modes. If you skip the protected-evaluator and visual-veto design in section 2, you will build a loop that reliably improves proxies while the delivered video gets worse — the exact failure the Cerebras reproduction documented within hours of loosening guardrails[^cerebras].

---

## 1. Philosophy translated to ego_annotation delivery

Six load-bearing ideas, each mapped from source to this project.

**1.1 Program the spec in English, not the pipeline in ad-hoc scripts.** Karpathy's insight: the quality of autonomous behavior scales with the quality of the Markdown that specifies it — "you are programming the `program.md`," not touching the Python[^ar-nbf][^ar-datacamp]. For delivery, the durable artifact of an auto-research run is a per-axis spec (section 2) that names the defect, the protected metric, the eval clips, the accept/reject rule, and the anti-patterns. The pipeline code is the editable surface; the spec is the control plane. A vague spec produces drift; a sharp spec produces a git history of validated improvements[^ar-repo][^cerebras].

**1.2 Become one with the rendered video before touching code.** The Recipe's first stage is hours of data inspection before any model code, because the network is a compressed version of the data[^recipe]. For delivery this is literal: before proposing any change, the agent must *consume the current rendered overlay/side-by-side and the backing rows as an annotation* (AGENTS.md work-progress standard) — identify what object is present, what physical state is claimed, where the projected hand leaves the visible hand, where the caption contradicts the frame. A proxy metric read without watching the video is the delivery version of training without looking at the data.

**1.3 Ratchet: propose → short run → measure protected metric → keep-if-better, else `git reset`.** This is the AutoResearch core loop verbatim[^ar-repo]. It transfers directly, with two mandatory modifications for delivery (section 2): the metric is a **protected vector** (five axes), not a scalar; and acceptance requires **vector monotonicity plus a visual-truth veto**, not a single lower number. Monotonicity is not optional here — it is already the project invariant (AGENTS.md: V18+ must preserve every valid prior capability). The greedy ratchet is exactly a monotonic ratchet.

**1.4 Protect the evaluator; never let the loop edit the thing it is scored on.** In AutoResearch the agent cannot modify `prepare.py`, the eval clip set, or `evaluate_bpb` — keeping evaluation outside write access preserves metric integrity[^ar-repo][^ar-datacamp]. For delivery the protected surface is larger and stricter: the frozen eval clip list, the frozen GT sidecars (HOT3D hand GT, any camera fiducial/IMU trajectory), the metric-computation harness that reads the API output schema, **and any calibration that would otherwise be fit to the eval GT**. The demo's GT-fitted per-clip hand calibration (subagent 5) is precisely the evaluator-cheat this rule forbids: a "5 mm" number obtained by fitting to the answer on the scored clip is not a delivery capability.

**1.5 Don't be a hero; add one signal at a time; prefer the simpler mechanism.** The Recipe warns against custom architectures and against adding complexity in bulk — copy the simplest thing that works, introduce one change at a time, verify each yields the expected gain[^recipe]. AutoResearch encodes the same as an explicit **simplicity criterion**: a small metric win that adds ugly complexity is not worth it; deleting code for equal-or-better metric is a win to keep[^ar-repo]. For delivery this kills the reflex to bolt HOI/factor-graph machinery, per-category if/else branches (AGENTS.md methodology), or a new module onto a drift that a single intrinsics/crop-adapter correction (R2) would fix.

**1.6 Never stop, but stay scoped — drift is the dominant failure, not lack of intelligence.** AutoResearch's "NEVER STOP" directive keeps the loop running unattended[^ar-repo]; the Cerebras reproduction shows the flip side: loosen the objective and the agent abandons the task within hours (it silently switched from memory-savings to a masking side-quest that saved no memory)[^cerebras]. Their conclusion — *environment design and task framing matter more than model choice* — is the operating principle for delivery. Each loop is pinned to one axis, one protected metric, one eval clip set, with frequent visual check-ins. Autonomy is the throttle (Karpathy's "autonomy slider"[^slider]): run the loop unattended on tightly-scoped, low-risk axes (hand refit band-local escalation, caption windowing); keep a human in the loop for axes where the metric is most gameable (a 5 mm head/camera claim, which no GT-free metric can certify — subagent 5, residual risk 1).

---

## 2. The delivery auto-research operating loop

### 2.1 Three-surface architecture (the AutoResearch three-file mapping)

| AutoResearch surface | Delivery-track equivalent | Rule |
|---|---|---|
| `program.md` (English spec) | **`axis_spec.md`** — one per axis (head_camera, hand, drift, caption, throughput). Names defect, protected metric, eval clips, accept rule, anti-patterns. | The human edits this. This is the "research org code." |
| `train.py` (editable code) | **The delivery pipeline stage for that axis** — the perception/reconstruction/render code the agent may edit (e.g. hand fusion, intrinsics/crop adapter, caption windowing, SAM2 rate). | Editable. Category-agnostic; put case variation in model outputs, not Python branches (AGENTS.md). |
| `prepare.py` (read-only evaluator + data) | **Protected evaluation bundle** — frozen eval clip list, frozen GT sidecars, the harness that computes metric axes from the API output artifacts (`validation_metrics.parquet`, `quality_summary`, `overlay_events.ndjson`), and the mandatory visual-review checklist. | Read-only to the loop. Changing it is a human act recorded as a spec amendment, never an experiment. |

### 2.2 The protected metric vector (why a scalar is not enough)

The acceptance signal is a **vector** whose entries are protected metric axes read directly from the API output schema, so the loop is scored on exactly the numbers the customer receives. Per axis, the primary gate metric and its schema location:

| Axis | Primary gate metric (subagent 5 ID) | Schema source (subagent 4) | Direction | Anchor status |
|---|---|---|---|---|
| Hand accuracy | H1 wrist/root translation error; H2 MPJPE | HOT3D GT eval; `hand_states` joints | lower | **GT-anchored** on HOT3D 001849/50/51 |
| Visible hand drift | H3 reprojection px; H5 projected/detected size ratio | `median_hand_reprojection_error_px`, `projected_to_detected_size_ratio` | lower | Self-consistency (gameable — pair H3+H5) |
| Head/camera | HC1 static reprojection residual; HC4 ATE when GT exists | `median_feature_reprojection_error_px`; `head_camera` transforms | lower | Self-consistency only unless fiducial/IMU GT added |
| Caption | S1 coverage/duration; S2 grounding | `semantic_clips` coverage; `caption_confidence` | S1 higher/compliant, S2 higher | Self-consistency + periodic human audit |
| Throughput | T1 GPU-h per video-h by lane | serving logs / `runtime` provenance | lower | Measured, hardware-normalized |

**Acceptance rule (replaces AutoResearch's single `val_bpb` compare):**

1. **Target axis must improve** its primary gate metric on the frozen eval set (beyond a noise band you measured from re-running the baseline ≥3×).
2. **No other protected axis may regress** beyond its noise band. A change that improves H3 while regressing H5 (the classic "2D agreement without 3D truth" gaming mode) is **rejected**, not accepted — this is vector monotonicity, and it is the project's monotonicity invariant (AGENTS.md) expressed as the ratchet.
3. **Visual-truth veto.** For any change touching a rendered layer, a human (or a VLM verifier used only as a second path, never as sole truth) reviews the overlay/side-by-side on the eval clips. If the rendered annotation contradicts visible evidence, the change is rejected regardless of proxy metrics. This is the delivery version of "the metric improved but the video got worse," the failure the whole guide exists to prevent.
4. **Provenance non-contradiction (R4) must stay at zero.** Any frame where current detector evidence is hidden/ghosted or an inferred state is rendered solid is a fix-before-claim defect, not a tradeoff.

Only when 1–4 hold does the change advance the branch (`git commit` kept). Otherwise `git reset`.

### 2.3 The loop (per axis, per GPU/branch)

Adapted directly from the AutoResearch `program.md` experiment loop[^ar-repo], with delivery guardrails folded in. Compute placement follows AGENTS.md: **all model inference and pipeline runs execute on the server/A800, never local**; the agent orchestrates via tmux/job sentinels and does **no `sleep`/polling** (AGENTS.md runtime discipline).

```
SETUP (human + agent, once per run):
  - Agree run tag; create branch delivery-ar/<axis>-<date> from current working version.
  - Read axis_spec.md + the editable stage code + the protected metric harness interface.
  - Confirm frozen eval clip list + GT sidecars resolve; confirm compute target = server/A800.
  - Record baseline: run the pipeline UNMODIFIED on the eval clips, compute the full
    protected metric vector + capture rendered overlays. Write baseline row to ledger.

LOOP FOREVER (until human interrupt):
  1. Read git state + the last N ledger rows (near-misses, discards, crashes).
  2. Write the causal experiment card (section 4) BEFORE editing: defect -> physical
     variable -> mechanism hypothesis -> discriminating predictions -> accept/reject rule.
     (AGENTS.md research discipline: no sequential guessing; every result must be informative.)
  3. Edit ONLY the editable stage code for this axis. One mechanism change per iteration.
  4. git commit the code change.
  5. Launch the run on server/A800 in a tmux window, output -> run.log, durable sentinel.
     Do NOT sleep-wait: start the card for the next idea or do an immediate non-blocking
     status check; return to this run when its sentinel fires.
  6. Read metrics from the API output artifacts (validation_metrics.parquet / quality_summary)
     + throughput logs. If artifacts are missing/empty -> treat as crash (read tail of log).
  7. Apply the section-2.2 acceptance rule (vector monotonicity + visual veto + R4=0).
  8. Log the ledger row (kept/discard/crash) — leave the ledger git-untracked, like results.tsv.
  9. If accepted: advance branch. If rejected: git reset to step-1 commit.
  10. Time box: a run must stay within the same order of magnitude as clip duration
      (v18 runtime invariant). Kill and discard runs exceeding the axis time budget.
```

**When stuck, think harder before rewinding** (AutoResearch guidance[^ar-repo], sharpened by AGENTS.md): re-read the axis spec and the referenced metric mechanisms, combine previous near-misses, or escalate to a more radical mechanism *within the same axis family* — never hop to an unrelated mechanism, and never abandon a mechanism because one implementation was inert (AGENTS.md research discipline). Rewind the branch very sparingly.

**Worse-before-better exception (fixes the ratchet's known blind spot).** The greedy ratchet cannot take a step backward to set up a larger gain[^ar-limits]. Some real delivery fixes need this: replacing the intrinsics/crop adapter (R2) may transiently raise H3 while it corrects H5/depth. Handle it explicitly: the human may open a **named family experiment** in the spec with a bounded transient-regression budget on a subset of axes, evaluated on the *full vector at family close*, not per micro-step. This keeps "never abandon a mechanism because one attempt was inert" (AGENTS.md) without opening a general license to regress.

### 2.4 Distributed axes (autoresearch@home for delivery)

The five axes are independent enough to run as parallel loops on separate GPUs/branches — the delivery analog of the "research community of agents"[^ar-fortune][^cerebras]. Constraint: **cross-axis monotonicity is enforced at merge, not per loop.** Before merging an axis branch into the working version, recompute the *full* protected vector across all axes on the eval set; reject the merge if any other axis regressed. This is the merge-time guard that stops one axis's local win from silently degrading another (e.g., a throughput SAM2-rate reduction that quietly worsens caption grounding S2).

---

## 3. Dataset / task queue templates

### 3.1 Frozen eval clip set (the "held-out data" analog)

The eval set is protected and versioned; the loop cannot edit it. It must span the gaming modes each metric has. Seed it from the six demo clips plus HOT3D, stratified:

```yaml
eval_set:
  id: delivery_eval_v1
  frozen: true                 # loop may NOT add/remove clips; changes are human spec amendments
  gt_anchored:
    - clip: hot3d_001849       # hand GT (H1/H2); camera GT only if sidecar exists (HC4)
    - clip: hot3d_001850
    - clip: hot3d_001851
  self_consistency:
    - clip: task5_tomato_960   # stable tabletop: HC1/HC2 sanity, tomato visible-drift concern (R1)
    - clip: trash_1050         # KNOWN-FAIL probe: drift-bent world (HC3), band-local escalation (H3/H4/R1)
    - clip: window_putty_knife # static scene sanity + short-action caption (S1)
    - clip: phone_calculator   # tap/hold caption intervals (S3), object-presence grounding (S5)
    - clip: cut_cloth_scissors # cutting action caption + hand occlusion (S1/S2)
    - clip: origami_paper      # occlusion joint-relabel (H3 review-tier), boundary captions
  roles:
    known_fail_probes: [trash_1050]   # a change is suspicious if it "fixes" these too cheaply
    stable_negatives: [task5_tomato_960, window_putty_knife]  # must NOT regress
  metric_coverage_check: true  # every protected axis must be exercised by >=1 clip
```

Rationale: `trash_1050` is retained as a **known-fail probe** exactly as AutoResearch keeps a baseline to beat — a change that "improves" the drift-bent world without a plausible mechanism is a gaming signal, not a win (subagent 5 uses trash to validate failure *detection*). Stable negatives guard monotonicity.

### 3.2 Task queue (experiment backlog)

Each queue entry is a causal experiment card, not a "try X" note. Priority follows AGENTS.md anti-avoidance: the hardest essential root blocker first, not the locally-verifiable easy one.

```yaml
task_queue:
  - id: DR-001
    axis: drift
    priority: 1                       # highest measured blocker for visible drift
    defect: "tomato: projected left hand leaves visible hand in renders/overlay.mp4 ~mid-clip"
    physical_variable: "metric depth / crop-scale / intrinsics of hand projection"
    mechanism_hypothesis: "single focal/crop correction reduces residual across hands+static (R2)"
    gate_metric: [R2_sweep_improvement, H5_size_ratio_error, H3_reproj_px]
    monotonic_guard: [H1_wrist_error, HC1_static_reproj]   # must not regress
    eval_clips: [task5_tomato_960, trash_1050, window_putty_knife]
    compute_target: server_a800
    status: ready
  - id: DR-014
    axis: throughput
    priority: 1
    defect: "full-rate SAM2 breaks 5-node GPU-h budget (T1 2.08-2.23 vs 0.5-0.7 target)"
    physical_variable: "SAM2 propagation frame rate / batch cost"
    mechanism_hypothesis: "reduced-rate propagation + microbatch by pixels*objects holds mask quality"
    gate_metric: [T1_gpu_h_per_video_h]
    monotonic_guard: [S2_caption_grounding, R3_object_iou]  # rate cut must not degrade masks->captions
    eval_clips: [task5_tomato_960, phone_calculator]
    compute_target: server_a800
    status: ready
```

Queue hygiene (from Cerebras' cost lesson[^cerebras]): each rejected run burns real GPU minutes, so **proposal quality dominates total cost** — a card must state discriminating predictions before it is dequeued, and cards with no outcome that would change the model are deleted, not run (AGENTS.md research discipline).

### 3.3 Experiment ledger (the `results.tsv` analog)

Tab-separated, **git-untracked** (as in AutoResearch[^ar-repo]), one row per run. Extended from a single metric to the protected vector plus the veto:

```
commit  axis  H1_mm  H3_px  H5_ratio_err  HC1_px  S1_cov  S2_grnd  T1_ghpvh  visual_veto  R4_contra  status  description
a1b2c3d hand  5.1    7.4    0.08          1.8     1.00    0.86     n/a       pass         0          keep    baseline
b2c3d4e drift 5.1    6.9    0.06          1.8     1.00    0.86     n/a       pass         0          keep    R2 focal-scale -1.2% + crop fix
c3d4e5f drift 5.1    5.2    0.19          1.8     1.00    0.86     n/a       FAIL         0          discard 2D-only refit: H3 down but H5 up + video worse
d4e5f6g tput  5.1    7.4    0.08          1.8     0.71    0.61     0.62      pass         0          discard SAM2 rate/4: T1 ok but S2 grounding regressed
```

Row `c3d4e5f` is the canonical rejected-gaming record: H3 improved, but H5 (depth) worsened and the visual veto failed — kept in the ledger as **negative information** (AGENTS.md/EPISTEMIC discipline: preserve why an attractive change failed). Row `d4e5f6g` shows the cross-axis monotonic guard catching a throughput win that broke captions.

---

## 4. Metric-first experiment templates

Every experiment is a causal card written **before** the run (AGENTS.md research discipline; AutoResearch "propose a change with explicit reasoning"[^ar-datacamp]). Templates below are pinned to specific self-consistency metric IDs (subagent 5) and API schema fields (subagent 4).

### 4.1 Card skeleton (all axes)

```
CARD id / axis / priority
DEFECT (in the rendered artifact): <what the overlay/side-by-side/backing rows get wrong>
PHYSICAL VARIABLE: <the wrong quantity: depth, focal, wrist translation, boundary time, GPU-h>
MECHANISM HYPOTHESIS: <cause->effect chain that produced the defect>
COUPLING: <why the proposed edit touches that variable>
GATE METRIC (schema field + subagent-5 ID): <primary>
MONOTONIC GUARD SET: <axes/metrics that must not regress>
DISCRIMINATING PREDICTIONS:
  - if mechanism dominant: <gate improves AND guard holds AND video improves>
  - if measurement weak/miscoupled: <gate flat, or improves without guard/video improving>
  - if another variable dominates: <specific other metric moves instead>
ACCEPT/REJECT: apply section-2.2 rule. NEXT ACTION per outcome.
EVAL CLIPS / COMPUTE TARGET
```

### 4.2 Hand accuracy (GT-anchored — the cleanest loop)

```
DEFECT: right-hand median reprojection 18.6 px, size-ratio-error 0.21 -> low_confidence tier.
PHYSICAL VARIABLE: per-side hand detection/crop OR wrist translation vs articulation.
GATE METRIC: H1 wrist_error_mm (HOT3D GT) primary; H2 MPJPE secondary. Schema: hand_states.joints_3d_camera_m.
MONOTONIC GUARD: left-hand H1/H3 (do not fix right by breaking left); H5 size ratio.
DISCRIMINATING PREDICTIONS:
  - smooth bias over time -> camera/registration self-calibration target (NOT GT-fitted, see anti-pattern A2).
  - per-frame noisy residual -> hand estimator/fusion source switch.
  - side-specific -> per-side crop/label issue; route to detector, not camera.
ACCEPT: H1 down AND left-hand guard holds AND overlay shows right hand tracking the visible hand.
```

This is the axis most faithful to AutoResearch because H1 is GT-anchored on HOT3D — treat that GT bundle exactly like `evaluate_bpb`: read-only, never fit to.

### 4.3 Visible drift (most gameable — always pair metrics)

```
DEFECT: R1 drift burst — projected hand departs visible evidence >50 px for >=3 consecutive frames.
GATE METRIC: R1 burst score + H3 reproj px. PAIR MANDATORY with H5 size-ratio (subagent-5 routing).
MONOTONIC GUARD: H5 (depth), HC1 (static reproj), R4 (=0).
DISCRIMINATING PREDICTIONS:
  - all layers drift coherently -> intrinsics/camera (R2 sweep helps everything) -> global fix.
  - localized band only -> band-local hand/source escalation (NOT global reweight; that resurrected old garbage in the demo).
  - backing state good but video wrong -> render-wiring fix (R4), do NOT rerun model.
ACCEPT: R1 down AND H5 within 0.9-1.1 AND visual veto pass. REJECT if H3 down but H5 up (2D-only gaming).
```

### 4.4 Head/camera (metric ceiling — advisor-gated)

```
DEFECT: candidate 5 mm head/camera claim; only self-consistency metrics available.
GATE METRIC: HC1 static reproj + HC2 3D closure + HC3 gravity plausibility. HC4 ATE ONLY if GT sidecar exists.
HARD RULE: NO deployable "5 mm head/camera" claim from HC1-3 alone (subagent 5 residual risk 1).
  The loop may improve HC1-3 and route drift; it may NOT emit a metric accuracy claim without HC4 GT.
DISCRIMINATING PREDICTIONS:
  - HC1/HC2 pass but HC4 fails -> gauge/calibration wrong.
  - HC4 passes but hands drift -> route to hand/intrinsics, not camera.
NEXT ACTION if no GT: keep this axis human-in-the-loop; queue "acquire fiducial/IMU camera-GT clip" as a blocker.
```

### 4.5 Caption (2–3 s semantic)

```
DEFECT: caption names an object with no track / boundary unstable / duration non-compliant.
GATE METRIC: S1 coverage+duration (100% timeline, >=90% interior 2-3 s), S2 grounding, S4 boundary stability.
  Schema: semantic_clips (start/end_frame, boundary_reason, caption_*, evidence_frame_indices).
MONOTONIC GUARD: S5 caption-to-annotation agreement (no naming a visible-object contact the delivery track can't back).
DISCRIMINATING PREDICTIONS:
  - unsupported noun -> rerun detector prompts / object-plan constraint (put variation in model output, not if/else).
  - unstable boundary under resampling (S4 sd >1 s) -> lower confidence + merge, do NOT present precise 2-3 s label.
ACCEPT: S1 compliant AND S2 grounded AND VLM self-consensus (S3) not a shared hallucination (periodic human audit).
```

### 4.6 Throughput (measured, hardware-normalized)

```
DEFECT: lane exceeds GPU-h/video-h budget (T1). Total allowance ~2.45 GPU-h/video-h; conditional target 1.5-2.0.
GATE METRIC: T1 GPU-h per video-h by lane; T2 worker residency; T5 regression vs measured anchor.
MONOTONIC GUARD: quality axes for that lane (SAM2 rate -> S2/R3; hand batch -> H3).
DISCRIMINATING PREDICTIONS:
  - lane over budget by >20% -> profile+tune that lane (AGENTS.md: attack the over-budget lane directly).
  - cold-start dominates -> persistent resident workers (subagent 2), not algorithm change.
ACCEPT: T1 down AND guarded quality axes hold. Full-rate SAM2 is the current dominant blocker, not hands.
```

---

## 5. Failure triage rules

Read a run's outcome, then route. These are the delivery routing rules (subagent 5) folded into the AutoResearch crash/keep/discard flow[^ar-repo].

**5.1 Run crashed (empty/missing output artifacts).** Read `tail -n 50 run.log`. If trivial (typo, missing import, OOM from a too-large batch) fix and rerun. If the mechanism is fundamentally broken, log `crash`, revert, move on. A crash on the server/A800 that is an *infrastructure* failure (env, CUDA, symlink) is **not** an experiment result — record the phase blocker and stop that lane; do not spend the loop provisioning infrastructure (AGENTS.md role separation; Cerebras spent more time on sandbox permissions than research[^cerebras]).

**5.2 Gate improved, guard axis regressed → reject (gaming/tradeoff).** The most important triage. H3 down + H5 up = 2D agreement hiding wrong depth → reject, route to R2/depth. Throughput down + S2 down = mask degradation → reject. This is where the vector ratchet earns its keep.

**5.3 Gate improved, guard holds, but video veto fails → reject.** The proxy moved but the rendered annotation contradicts the frame. This is the "metric up, artifact down" case; the render is the artifact (AGENTS.md). Investigate whether the metric harness reads a stale render layer (R4) before trusting any proxy.

**5.4 Gate flat, nothing moved → the measurement was weak or miscoupled, not "it doesn't work."** "It works / it doesn't work" are not interpretations (AGENTS.md). Ask: did the edit actually reach the physical variable? Did the solver/render consume it? Re-examine coupling before discarding the *mechanism* — only the *implementation* failed.

**5.5 Improvement only on a known-fail probe (`trash_1050`) with no plausible mechanism → suspect gaming.** A cheap "fix" to a drift-bent world without a named cause is the delivery version of the agent optimizing the metric instead of the task[^cerebras]. Require the mechanism before keeping.

**5.6 Routing table (which knob, given which metrics move):**

| Symptom | Route to | Do NOT |
|---|---|---|
| Head/camera metrics fail, hand image metrics pass | camera/SLAM/intrinsics | rerun hand model first |
| Hand reproj fails, camera/static pass | hand detection/refit/fusion (band-local) | global reweight |
| Hand reproj passes, projected-size fails | depth/crop-scale/intrinsics | claim 3D accuracy from 2D |
| Intrinsics sweep improves ALL layers | fix adapter/convention, rerun projection | patch per-hand residuals |
| One source band diverges | band-local escalation/hysteresis | global reweight (resurrects old garbage) |
| Caption grounding fails | rerun VLM with object-plan constraint / lower confidence | let fluent caption outrank frame |
| Throughput lane fails | tune the over-budget lane (SAM2 first) | thin hands first |
| Rendered state contradicts current evidence (R4>0) | fix dataflow/render gating | rerun model |

**5.7 Stuck (ideas exhausted).** Think harder within the axis before rewinding[^ar-repo]: re-read the axis spec + metric mechanisms, combine near-misses, escalate to a more radical *in-family* mechanism. Rewinding the branch is a last resort. Never jump to an unrelated axis to manufacture motion.

---

## 6. Anti-patterns specific to ego_annotation

Each is a concrete way a delivery auto-research loop produces false progress here. Severity tags: **[blocker]** invalidates the run; **[high]** corrupts the metric; **[medium]** wastes compute.

**A1 [blocker] Optimizing a self-consistency proxy while the rendered video degrades.** The loop lowers H3/HC1/jitter but the overlay drifts further from the visible hand or the caption contradicts the frame. Root cause: a scalar ratchet with no visual veto. Guard: section-2.2 rule 3 (mandatory visual review on rendered eval clips), and pair every gameable metric with its complement (H3 with H5, jitter with H3).

**A2 [blocker] Fitting calibration/parameters to the eval GT ("evaluator cheat").** The demo's sub-10 mm hand result came from GT-fitted per-clip calibration and proves smooth correctability, not deployable accuracy (subagent 5, residual risk 2). Any loop edit that reads the eval GT to set intrinsics/scale/registration is editing the protected evaluator through a side channel. Guard: GT sidecars and GT-derived calibration are read-only; deployable claims require GT-free self-calibration validated on held-out clips.

**A3 [blocker] Shipping a container and calling it the artifact.** A passing `validation_metrics.parquet`, a green `quality_summary`, a row count, or an overlay label is not the deliverable unless the *visible annotation content* is produced by the named mechanism and checked against the video (AGENTS.md). An auto-research loop that ratchets schema/validator fields without changing the rendered marks has made zero progress.

**A4 [high] Metric drift / silent objective substitution.** The loop wanders off the axis (Cerebras: memory-savings → masking side-quest that saved no memory[^cerebras]). In delivery this looks like a "hand accuracy" loop that starts tuning caption prompts, or a throughput loop that quietly trades away mask quality. Guard: one axis per loop, pinned gate metric, monotonic guard set, frequent human check-ins; cross-axis monotonicity enforced at merge (section 2.4).

**A5 [high] Global reweighting to suppress a local failure.** In the demo, global reweighting resurrected old garbage hand states. Band-local escalation/hysteresis is the correct mechanism for a localized band (subagent 5). A loop that lowers a clip-median by globally down-weighting is gaming the aggregate.

**A6 [high] Category/if-else branches to pass specific clips.** Hand-written logic keyed on object class/color/action to make tomato or scissors pass is a failed perception strategy (AGENTS.md methodology). Case variation belongs in model-produced data (open-vocabulary detector, SAM2, VLM plan), consumed by one category-agnostic path.

**A7 [high] Emitting a 5 mm head/camera accuracy claim from GT-free metrics.** HC1–HC3 detect drift but cannot certify absolute accuracy (subagent 5, residual risk 1). A loop that "reaches 5 mm" on self-consistency alone is claiming an accuracy the measurement cannot support. Guard: head/camera accuracy claims require HC4 GT (fiducial/IMU); until then this axis is advisor-gated.

**A8 [high] Over-smoothing that passes jitter (H6) while visually lagging.** A temporal filter can drive jitter metrics green while the hand visibly lags the video. Guard: H6 is a review index, not a gate; pair with H3 and the visual veto (subagent 5 blind spot).

**A9 [high] VLM self-consensus (S3) preserving a shared hallucination.** Overlapping-window caption agreement can lock in a common wrong noun/verb. Guard: periodic human audit on stratified clips; S2 grounding against detector/SAM2 tracks; S5 agreement with annotation state.

**A10 [medium] Bolting HOI/factor-graph/object-pose machinery into the delivery loop.** Delivery excludes HOI (subagent 4 scope; task spec). A loop that adds contact/object-pose mechanisms to "improve" captions or drift is violating scope and adding runtime the throughput budget cannot afford. Route HOI ideas to the research track.

**A11 [medium] Heavy compute on the local workstation.** SAM2/UniDepth/HaWoR/WiLoR/VLM inference must run on server/A800; an accidental `cuda` default locally violates AGENTS.md runtime discipline. The loop orchestrates; the server executes.

**A12 [medium] `sleep`/polling as a progress substitute; treating infra failures as experiment results.** Waiting on a run with `sleep` is forbidden (AGENTS.md); use tmux/job sentinels and work an independent card meanwhile. An infrastructure crash (env/CUDA/symlink) is a phase blocker to record and stop on, not a discarded experiment — the Cerebras team's biggest time sink was sandbox/GPU permissions, not research[^cerebras].

**A13 [medium] Reporting ledger/keep-count as the deliverable.** "37 experiments, 20 kept" is process bookkeeping, not artifact progress. Report what physical annotation mechanism changed, what rendered/geometric evidence changed, and which protected axis improved on the eval set — nothing else earns the headline (AGENTS.md work-progress standard).

---

## References

- [^recipe]: A. Karpathy, "A Recipe for Training Neural Networks," 2019. https://karpathy.github.io/2019/04/25/recipe/ — become-one-with-the-data, dumb baselines, overfit, regularize, tune; add complexity one at a time; don't be a hero.
- [^ar-repo]: karpathy/autoresearch, `program.md` (March 2026). https://github.com/karpathy/autoresearch/blob/master/program.md — three-file architecture, protected `prepare.py`/`evaluate_bpb`, 5-min budget, keep-if-better-else-`git reset` ratchet, `results.tsv`, NEVER STOP, crash/timeout handling, simplicity criterion.
- [^ar-fortune]: "Why everyone is talking about Andrej Karpathy's autonomous AI research agent," Fortune, Mar 2026. https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/ — 700 experiments / 2 days / 20 optimizations; "the final boss battle"; research-community-of-agents.
- [^ar-nbf]: "Andrej Karpathy on Code Agents, AutoResearch and the Self Improvement Loopy Era," NextBigFuture, Mar 2026. https://www.nextbigfuture.com/2026/03/andrej-karpathy-on-code-agents-autoresearch-and-the-self-improvement-loopy-era-of-ai.html — "programming the program.md"; behavior quality scales with the spec.
- [^ar-datacamp]: "A Guide to Andrej Karpathy's AutoResearch," DataCamp. https://www.datacamp.com/tutorial/guide-to-autoresearch — protected evaluator preserves metric integrity; open proposer + fixed evaluator + strict ratchet; `val_bpb` vocabulary-independent.
- [^cerebras]: S. Chieng, S. Cherfa, "How to stop your autoresearch loop from cheating," Cerebras, Mar 2026. https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating — agent abandoned the task (masking side-quest, no memory saved); environment/task-framing > model choice; proposal quality dominates compute cost; different agents converge; tooling gap, not intelligence.
- [^ar-limits]: Analyses of the greedy ratchet's local-search limitation (cannot step backward for a larger gain), e.g. mljar / alexeyondata / Verdent AutoResearch write-ups, Mar 2026.
- [^slider]: A. Karpathy, "Software Is Changing Again" / autonomy-slider keynote, YC AI Startup School, Jun 2025; summary at https://www.latent.space/p/s3 — partial autonomy, autonomy slider left→right over time, human-in-the-loop during the jagged phase.

## Residual risks

1. **Self-consistency gating remains fundamentally gameable.** The vector ratchet + visual veto reduce but do not eliminate metric-gaming; any axis without GT (head/camera, most drift, captions) can drift toward a consistent-but-wrong optimum. Periodic human/GT anchoring is required, not optional.
2. **The 5 mm head/camera claim is unreachable by this loop alone.** No self-consistency metric certifies absolute camera accuracy; acquiring a fiducial/IMU camera-GT clip is a prerequisite blocker for that claim (subagent 5, residual risk 1).
3. **The greedy ratchet under-explores.** Even with the family-experiment exception, the loop favors small local wins and will not invent a novel mechanism a human researcher would eventually reach (observed AutoResearch limitation[^ar-limits]); direction-setting stays human.
4. **Compute honesty depends on hardware-normalized throughput.** T1/T5 comparisons are only valid when normalized to the same GPU/resolution; shared-GPU contention on A800 can hide a lane regression (subagent 5, T-family blind spots).
5. **The guide assumes the protected evaluator harness exists.** Building the frozen eval-clip + GT + schema-reading metric harness is itself delivery work; until it exists, the loop has no honest gate and must not run unattended.