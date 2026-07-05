# Subagent 10 — RL approximation of the HOI factor graph and physical-state inference

Role: research-track input for task-pack synthesis. Defines if/how RL may approximate the
factor-graph inference objective for HOI extraction without becoming a shortcut that
substitutes learned bias for measurement. Consumers: parent synthesis, subagent 9 (research
loop), subagent 11 (distillation). Evidence base: Sub7 (v19 bottlenecks, objective redesign
§4–5), Sub8 (HOI outputs/metrics/uncertainty), Sub5 (self-consistency metrics), R6. No code edited.

## 0. Category check and verdict

The v19 factor graph failed because it was starved of **measurements**, not because the
optimizer was weak (Sub7 §0, §4.1): contact had zero support in 900/900 rows, object pose was
fit against channels blind to the failing directions, hand terminal error lived in a latent
the state did not contain. An RL policy is a deterministic function of the same observations
the graph sees; it cannot recover information absent from the sensors. Applied to the
measurement side, RL manufactures learned prior bias and labels it inference — the exact
failure mode AGENTS integrity/monotonicity and Sub7 §8.1 prohibit.

**Verdict.** RL is admissible only as an **inference-side approximator** of an already-validated
factor graph: it may amortize the solver, schedule discrete hypotheses, choose initializers,
or project a measurement-derived estimate onto a feasibility set. It may never replace the
likelihood, never replace a measurement channel, never replace the per-claim posterior with a
point estimate. R6 (RL approximation) is gated by R4 (factor-graph redesign) being closed and
R1–R3 (correspondence, geometry, contact channels) producing live variables. Approximating an
inert graph is approximating nothing (B10 liveness failure).

## 1. Admissible vs inadmissible roles

| Role | Approximates | Admissible? | Gate |
|---|---|---|---|
| **A. Amortized MAP** | graph argmin of validated objective over the measurement bundle | yes | R4 closed; teacher = graph MAP, liveness asserted |
| **B. Hypothesis/switch scheduler** | switch init (contact/near/none, rigid/articulated/deformable), factor-subset activation, initializer choice | yes | R3 + R1 live |
| **C. Physics-feasibility projection** | residual policy projecting a measurement-derived estimate onto a simulator-feasible set (ManipTrans/DexMachina/SPIDER/QuasiSim lineage) | yes, research-only | sim-fidelity gap bounded by measurement; never contact truth |
| **D. End-to-end video→state RL** | whole HOI state from pixels | no here | distillation target (Sub11) |
| **E. Measurement hallucination** | state along unobserved null-space directions (in-plane planar pose, contact under full occlusion, drift latent without GT-free anchor) | no | outputs must be `unresolved` |
| **F. Reward-proxy optimization** | silhouette IoU / source gap / proximity as a stand-alone objective | no | P46/P50/DexMan proxy capture |

A–C are the admissible surface. D is deferred to Sub11; E and F are falsifications (§6).

## 2. State / action / reward design per admissible role

Shared principle: the reward is computed by evaluating the **actual factor-graph objective**
(A/B) or the **measurement-coupled simulator residual** (C), never a proxy. The measurement
bundle is the graph's pinned input contract (Sub8 isolation).

**Role A — Amortized MAP.** *State/context:* per-clip encoded measurement bundle — depth,
masks, SAM2/CoTracker tracks, HaWoR/WiLoR hand streams, contact-channel posteriors (E6),
drift-anchor residuals (E5), rigidity statistic (E1), declared gauge block. *Action:* full
continuous state — hand drift latents (spline coefficients), per-frame object SE(3), per-
interval contact switch, occlusion states; used as warm start or direct proposal. *Reward:*
negative graph free energy E(x)=−Σ factors, evaluated by the graph's own objective code
(calibrated σ from B7, switchable contact from R4), plus a small imitation term against the
graph MAP. *Constraint:* the policy may emit `no_correction` (identity); a frozen-output run
is a liveness failure (B10), not a solution.

**Role B — Hypothesis/switch scheduler.** *State:* clip-level summary — motion-coupling
timeline, rigidity-residual distribution, contact-channel agreement matrix, occlusion-state
timeline, track-density profile. *Action:* per-interval switch init over {contact, near, none}
(R4/B6), rigidity-class init (B5), factor-subset activation mask (well-supported terms per E4),
initializer/solver selection. *Reward:* graph objective reached after a fixed compute budget −
switch-flicker penalty (temporal hysteresis) − unsupported-factor-activation penalty. This
attacks the v19 failure "contact truth became a manual slice-selection mechanism" (Sub7 §4.1,
B6): the scheduler automates discrete structure, but the **posterior** still comes from the
switchable mixture solve, not from the policy.

**Role C — Physics-feasibility projection.** *State:* noisy measurement-derived hand+object
state (from A/B or feedforward), watertight object mesh with face provenance (B11), contact
hypothesis with channel evidence. *Action:* residual SE(3)/MANO corrections projecting into
the feasible set (no penetration beyond soft-tissue band, friction cone, gravity/support).
*Reward:* simulator feasibility residual − measurement-fidelity penalty **in observed
directions only**. *Critical:* unobserved null-space directions are free for the simulator to
set and the result there is labeled `feasibility_hypothesis`, not measurement. Sim-fidelity
hazard (mass/friction/soft tissue) is a declared variable, not a hidden default (§6 F-RL4,
§9 E_RL4).

## 3. Offline teacher vs online environment

**Offline teacher is the default and only research-sanctioned mode for Roles A and B.** The
teacher is the validated factor-graph solver, emitting state + posterior + per-claim
provenance. Training is imitation (behavioral cloning of the graph MAP) then RL fine-tuning
where the reward is the graph objective. This anchors the policy to the graph's evidence chain
and makes disagreement with the teacher a measurable, debuggable quantity.

**Online environment is admissible only for Role C**, where the simulator is the environment
(model-based RL). It is **inadmissible as default-path training for A/B**: (a) no accurate HOI
simulator exists (gravity/world drift B8, soft tissue, mass/friction unknown, non-watertight
geometry B11), so an online env silently encodes wrong physics as truth; (b) per-clip training
loops violate the runtime invariant (AGENTS; Sub7 §8.4). Online RL runs only on offline A800
branches under declared run roots, never on the delivery path or workstation.

**No reward shaping from renders.** A render is an inspection instrument (Sub8), not a
reward signal; shaping on overlay appearance is proxy capture (§6 F-RL1/F-RL6).

## 4. Uncertainty and posterior outputs

The graph's unique value is calibrated per-claim uncertainty (Sub7 §5.2, Sub8 uncertainty
boundaries). A point-estimate policy is a regression, not an approximation. Required for every
admissible role:

- The policy emits a **posterior**: parametric head (Gaussian over continuous DOFs, categorical
  over switches) or calibrated ensemble. Role A matches the graph's posterior family
  (translation/rotation σ from fitted noise B7; contact `posterior_p_contact`/`posterior_p_near`).
- **Calibration is a promotion prerequisite.** Expected calibration error on held-out GT
  (fixed slice + lockbox) must close a declared band before any research-default or promotion
  claim (§9 E_RL5). Uncalibrated ⇒ non-evidential.
- **Provenance preserved.** The output carries which channels drove the correction (ablatable,
  §9 E_RL2). A black-box correction ignoring measurements is a learned prior; its null-space
  components are marked `unresolved` per Sub8 object_pose semantics.
- **Gauge declaration travels with the output.** Every transform declares its gauge (per-clip
  world similarity, object canonical frame, camera-convention rotation). An "improvement"
  along an ungauged freedom is gauge motion (Track B Umeyama; Sub7 §8.3).

## 5. Constraints inherited from metric, gauge, and liveness

RL inherits every graph invariant; none relax because the optimizer changed form.

- **Liveness (B10, Sub8 solver_liveness_diff_count).** The RL-solved state must bit-differ
  from input, or the policy must emit identity with a recorded reason. Frozen-output with good
  objective is the same inert-solver false positive already caught (Track B 7.25 mm row).
- **Gauge (Sub7 §4.1, §8.3).** Declared per output; P37 constant-transform evaluator is the
  template. RL "improvements" are scored only after gauge fixation.
- **Self-consistency family (Sub5).** RL outputs are scored on the **same** GT-free family as
  the graph: HC1–HC6, R1–R4, contact evidence support fraction, anchor residual correlation.
  Reward is not a metric; the policy is evaluated by metrics, not its training signal.
- **Term efficacy (Sub7 E4).** The policy is auditable for measurement coupling: channels
  used/ignored, gradient share per input. Same instrument the graph redesign consumes.
- **Runtime (AGENTS).** Inference-time policy use only on any default path; training offline.
  Hours-per-clip inference ⇒ offline-only, cannot promote (Sub8 gate 3).
- **Freshness (B9).** Input contract pins substrate versions; a policy trained on one
  mesh/pose substrate cannot silently consume another (Sub8 base-job-id/manifest-sha pinning).

## 6. Invalid shortcuts (named, with the falsification that catches each)

- **F-RL1 — Reward-proxy capture.** Training on silhouette IoU / source gap / proximity as the
  objective. Prediction: reward↑ GT↓ — the measured P46/P50 pattern, echoed by DexMan's
  proximity-reward critique (Sub7 §5.3, §8.1). Any reward ≠ graph objective (or, for C,
  measurement-coupled simulator residual) is rejected at design time.
- **F-RL2 — Prior bias in unobserved null spaces.** A policy producing confident state along
  unobservable directions (in-plane planar rotation w/o texture tracks B1; contact under full
  occlusion B3; drift latent with no GT-free anchor B4) emits a learned prior. Detection:
  channel ablation (§9 E_RL2); output invariant to removing a measurement ⇒ that component is
  prior, relabel `unresolved`.
- **F-RL3 — Point estimate replacing posterior.** A policy without calibrated uncertainty
  cannot ground contact/pose/occlusion/nonpenetration (AGENTS heuristic rule; Sub8 boundary 1);
  rejected at the gate regardless of accuracy.
- **F-RL4 — Simulator-as-truth (Role C).** Sim mass/friction/soft-tissue are not measurements.
  Simulator feasibility is a regularizer; outputs in unobserved directions are
  `feasibility_hypothesis`. Detection: sim-parameter sweep (§9 E_RL4).
- **F-RL5 — End-to-end replacement of the graph.** Video→state RL without the graph teacher
  is distillation, not approximation — out of scope, routed to Sub11 (teacher must emit
  calibrated uncertainty, Sub7 §8.10). **F-RL6 — Reward shaping from rendered appearance**
  injects renderer styling into inference. Both reject.

## 7. Evaluation: against graph, GT, GT-free

RL is evaluated on three axes; reward is none of them.

1. **Graph-MAP agreement (primary — RL is an approximator).** Fixed slice + lockbox:
   translation/rotation/contact-switch agreement within a declared band; objective gap to the
   graph MAP; cases where RL finds a *better* objective than the iterative solver (escaped
   local minimum) feed back to graph redesign, not claimed as RL wins.
2. **GT (Sub8 metric table).** `object_pose_translation_residual_gt_mm` (green ≤20),
   `..._rotation_..._deg` (green ≤5), `object_chamfer_visible_mm` (green ≤15),
   `hand_wrist_residual_gt_mm` post-correction (target ≤5), `contact_auroc_vs_proximity`
   (green ≥0.8), `rigidity_classification_accuracy` (green ≥0.9). Dual proxy+GT reporting
   (Sub7 §8.1); proxy-only improvement is rejection.
3. **GT-free (Sub5 shared currency).** HC1–HC6, R1–R4, `contact_evidence_support_fraction`,
   `solver_liveness_diff_count`, `anchor_residual_correlation_with_gt_drift`. Blind-spot
   routing mirrors Sub5/Sub8: hand fail + camera-static pass ⇒ hand source, not RL retune;
   silhouette-IoU fail with hand good ⇒ separate object confidence, not global reweight.
4. **Calibration.** ECE on held-out GT per output family; reliability diagrams per claim type.
5. **Visual consumption.** Full-duration `hoi_overlay.mp4`/`hoi_side_by_side.mp4` driven by
   the policy output, inspected as an annotation (Sub8). File existence is not consumption.

## 8. Stop conditions

- R4 not closed ⇒ R6 does not start; nothing valid to approximate (RESEARCH_TRACK R6 wording).
- Channel ablation (E_RL2) shows the policy ignores a measurement it should depend on ⇒ it is
  a learned prior; stop, re-examine input encoding and reward coupling.
- Proxy test (E_RL3) confirms reward↑/GT↓ ⇒ stop; reward is wrong, redesign to graph objective.
- Calibration (E_RL5) fails the band ⇒ uncertainty invalid; stop, recalibrate or demote to
  non-evidential.
- Sim sweep (E_RL4) shows Role-C output varies with unmeasured sim params ⇒ demote to
  `feasibility_hypothesis`; do not promote.
- Inference exceeds delivery budget ⇒ offline-only; do not promote.
- Disagreement with graph MAP persists outside band after imitation+RL ⇒ policy class too
  weak; report as a negative result bounding approximability, not a graph failure.

## 9. Falsification experiments (predictions declared before runs; A800/offline, own run root)

**E_RL1 — Amortized-inference parity (Role A).** Train on graph MAP, fixed slice + ≥2 lockbox.
Predictions: (i) objective parity AND GT parity ⇒ Role A viable as research-default
accelerator; (ii) objective parity but GT fails ⇒ graph objective mis-specified — loop to R4,
RL exposed a graph defect; (iii) both fail ⇒ policy class too weak, report approximability
bound.

**E_RL2 — Measurement-coupling ablation (A/B).** Remove texture tracks / a contact channel /
a drift anchor in turn. Per channel: (i) observed direction ⇒ output changes materially;
(ii) unobserved null-space ⇒ output invariant ⇒ that component is prior, relabel `unresolved`
(B1, B3, B4). Sharpens provenance labels on the output.

**E_RL3 — Proxy vs graph-objective reward (Role A).** Two policies, identical architecture;
one silhouette-only reward, one full graph objective. Prediction: proxy shows reward↑ GT↓
(P46/P50/DexMan); graph-objective shows reward↓ GT↓ or unchanged. Rejects F-RL1 empirically
and reproduces the project's proxy-capture evidence inside the RL frame.

**E_RL4 — Role-C sim-fidelity bound.** Physics projection across a mass/friction/soft-tissue
sweep. Predictions: (i) measurement-dominated ⇒ output stable across configs and in observed
directions; (ii) simulator-dominated ⇒ output varies with params ⇒ label
`feasibility_hypothesis`, never contact truth; (iii) GT improves only inside the soft-tissue
band (2–3 mm finger pad, 5 mm palm, ContactOpt semantics) ⇒ useful regularizer; outside that
band any "improvement" is suspect.

**E_RL5 — Posterior calibration.** ECE + reliability diagrams per claim family on held-out GT.
Predictions: (i) calibrated ensemble passes ⇒ uncertainty admissible; (ii) point-estimate or
miscalibrated head fails ⇒ add/recalibrate before any promotion claim. Gates F-RL3.

Sequencing: E_RL1/2/3 require R4 + R1–R3 closed; E_RL4 once watertight geometry (B11) and a
contact hypothesis exist; E_RL5 on any policy claiming a posterior. Negative results redirect
within the chain (redesign reward, relabel null-space outputs, recalibrate); they do not
justify hopping to Role D or an unrelated mechanism (AGENTS research discipline).

## 10. Promotion boundaries

RL **never ships to delivery as a stage** (teachers never ship as stages; Sub7 §8.4, Sub8
gate). Two distinct boundaries:

- **Into the research default path.** A Role-A/B policy may replace the iterative solver as
  research-default inference only after: graph-objective parity on fixed slice + lockbox, GT
  parity, GT-free self-consistency parity, posterior calibration closed, liveness/gauge/
  freshness asserts passing, and a measured runtime win. The graph remains teacher of record,
  re-run on a sample for drift monitoring.
- **Into delivery.** The Sub8 five-part gate in full: (1) GT win on fixed slice + ≥2 lockbox
  with declared predictions; (2) GT-free self-consistency win on delivery-regime clips via the
  shared metric family; (3) runtime within budget (≤2.45 GPU-h/video-h lane share,
  same-order-of-magnitude as input duration); (4) regime-transfer (HOT3D↔egoscale, B15/B18);
  (5) implementation-class review (liveness, freshness, gauge, mask ownership). Plus
  RL-specific: posterior-calibration evidence (E_RL5) and a measurement-coupling audit (E_RL2)
  proving the promoted output is inference over measurements, not a learned prior.

Role C does not promote to delivery as a contact/pose source; at most as a feasibility
regularizer with its sim-fidelity gap represented as uncertainty.

## 11. Sequencing, coupling, residual risks

R4 (drift latents, switchable contact, calibrated noise, liveness, gauge) ⇒ R6 (this
subagent). R1–R3 must produce live variables first. The drift-latent overlap (Sub7 §9.7) is
the first honest test: an RL amortizer for the drift latent is only buildable once E5's GT-free
anchors are validated.

Couplings: Sub9 consumes the E_RL experiments as protected-eval-bundle entries with proxy-
capture guards; Sub11 consumes a calibrated Role-A/B policy only as a fast-path student, graph
as teacher of record — never the reverse.

Residual risks: (1) the graph objective itself may be mis-specified, so RL parity can encode
its defects — E_RL1(ii) is the detector; (2) simulator-fidelity gap (B8, B11) bounds Role C's
contact claims a priori; (3) benchmark overfit on three clips (B18) — lockbox named before any
policy tuning; (4) offline-teacher quality — if the graph MAP is wrong the policy inherits the
error, so the teacher's calibrated uncertainty (B7) is a hard prerequisite; (5) runtime — an
amortizer not faster than the iterative solver has no reason to exist and is itself negative.
