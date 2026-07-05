# Subagent 7 — V19 bottlenecks and HOI-extraction research analysis

Role: research-track input for task-pack synthesis. Consumers: parent synthesis, subagent 8
(HOI outputs/metrics), subagent 9 (research auto-research guide), subagent 10 (RL/factor-graph),
subagent 11 (distillation).

Evidence base: `.memory/tasks/2026-06-23-pipeline-v19/EPISTEMIC.md` (P34–P57 falsification
ledger), `docs/v19_hot3d_fixed_slice_v1_results.md`, `runtime/v19_runtime_spec.md` (P00–P21),
`docs/pipeline_v19_design_proposal.md`, `.memory/project/{pipeline_invariants,
hand_metric_mechanisms, object_geometry_and_render_lessons, self_consistency_metrics}.md`,
demo-pack final state, plus targeted web research (sources at end). No code was edited.

---

## 0. The one-sentence diagnosis

V19's joint HOI layer is starved of the measurements that would let it work: object pose is
fit against depth/silhouette channels that do not observe the failing directions, object shape
comes from a single-frame generative prior rather than accumulated video evidence, and the
contact/support factors ask for visible evidence that manipulation occlusion physically
removes — so the factor graph, as built, is measurably inert on hands (900/900 rows with zero
support; wrist deltas exactly 0.0) and one-sided on objects. The decomposition
(detection → per-object geometry/pose → joint graph) is not refuted by the evidence; the
current *placement of authority* inside it is. The missing cross-cutting measurement family is
temporal surface correspondence (2D/3D tracks), and the missing state family is low-dimensional
per-clip drift latents.

---

## 1. What v19 actually does today (mechanism summary, as-built)

Per `runtime/v19_runtime_spec.md` P00–P21: UniDepth metric depth + intrinsics contract → HaWoR
metric world hands + WiLoR visible-geometry hybrid (P04/P04b) → agent object plan → OWLv2
text-grounded boxes → SAM2 masks/tracks → anchor-frame visible geometry fusion (P09, agent
anchor review) → TRELLIS single-image mesh prior (P12) → completion + silhouette/free-space/
planar-slab filters (P13) → per-frame visible pose fit against depth (P14) → temporal rigid
pose graph with full-timeline completion (P15) → MANO/object constraint measurement (P16) →
agent contact/occlusion interval priors (P17) → interval MANO solve with contact/depth-order
terms and translation gate (P18) → **state split that preserves metric MANO and demotes contact
output to an uncertain surface hypothesis (P18b)** → state-driven full-duration render (P19)
→ publish (P20) → agent visual consumption (P21).

What is real and working: full-duration state-driven renders; mask tracking on the fixed slice
(150/150 visible, correct object identity); metric-MANO preservation discipline; contact-regime
wording from a Gaussian source-gap model; honest non-contact attribution (clip001849) vs
near-contact attribution (clip001851) validated against GT (P51/P52); the freshness boundary
(P19a); the falsification ledger itself.

What is inert or invalid: the hand-side factor graph acts as passthrough+gate (fixed-slice
wrist deltas 0.0, articulation deltas ±1 mm mixed sign); the object pose trajectory carries
centimeter/degree time-varying residuals not explained by camera (camera residual 3.7 mm
median, correlation with object residual r=−0.036, P44); contact ownership and signed
nonpenetration are unresolved everywhere (correctly so, given non-watertight meshes and zero
support evidence).

---

## 2. Bottleneck taxonomy

Classes: **IMPL** = engineering defect/fragility, mechanism known, fix buildable;
**SCI** = open scientific problem, mechanism or method not yet established;
**DATA** = missing/limited evidence in the data itself, or a domain information limit.

| # | Class | Bottleneck | Mechanism | Evidence | Owner |
|---|---|---|---|---|---|
| B1 | SCI | Rigid-object pose observability | Depth-ICP + silhouette channels do not observe in-plane rotation/translation of a mostly-planar visible surface; pose fits are one-sided; falsified fixes: constant rotation (P48), acceleration smoothing (P49), silhouette-only refit (P50), stationary translation generalization (P46) | v19 EPISTEMIC: 77.1 mm/4.93° median residuals (P37); P44 rules out camera | research |
| B2 | SCI | Instance-faithful object shape | Single anchor-frame TRELLIS completion hallucinates hidden geometry and bakes one moment's appearance into the permanent body; filters (silhouette/free-space/planar-slab) are post-hoc guards, not a shape posterior; no multi-frame evidence fusion | tomato f929 peel-horn vs f270 (IoU 0.375→0.553, same trajectory); clip001851 90,318 TRELLIS hidden faces, 72,449 free-space-rejected | research |
| B3 | SCI | Contact evidence under manipulation occlusion | The hand covers exactly the surface it touches; visible-mask-overlap support is structurally empty at contact time; contact must come from latent-surface, learned-prior, depth-order, or temporal channels | fixed-slice: 900/900 rows zero visible-surface support; "keys occluded by hand" failure cluster | research |
| B4 | SCI | Hand terminal residual = smooth per-clip drift | Wrist error is a slowly varying camera-frame SE(3)-ish drift (~8 effective dof/clip, lag-1 autocorr 0.66–0.77); not representable in current graph variables; no GT-free anchor proven yet | Track V: b(t) spline closes sub-10 mm on all HOT3D clips; per-frame oracle fails | research (delivery consumes result) |
| B5 | SCI | Rigidity/branch decision is declared, not measured | VLM decides rigid/deformable; no motion-based rigidity statistic verifies it, segments parts, or detects deformation events | design §6.3; no rigidity test exists in any phase | research |
| B6 | SCI | Discrete contact structure handled manually | Contact-present vs contact-absent is decided by human slice attribution (P51/P52) before applying contact-coupled machinery; the graph has no switch/mixture machinery to infer it | v19 EPISTEMIC "contact truth has become a slice-selection mechanism" | research |
| B7 | SCI | Noise scales and uncertainty calibration | Factor σ's are hand-set (combined contact σ=30.48 mm is the one modeled scale); no fitted mapping from self-consistency residuals to factor noise; "uncertainty calibration" term in design is unimplemented | design §5.2 vs fixed-slice behavior (broad priors push cm-scale unsupported shifts) | research |
| B8 | SCI | Gravity/world alignment on drift-bent worlds | SLAM worlds are not gravity-aligned and not rotation-correctable when drift-bent (72% ceiling measured); real fix is upstream (IMU/per-frame floor tracking) | object_geometry lessons: trash 4.4 m "height" span | research (upstream) |
| B9 | IMPL | Stale-sidecar/freshness fragility | 22 phases hand off via JSON path conventions; dependent states can silently consume artifacts built against different upstream meshes/poses; P19a boundary added but the class recurred twice | pipeline_invariants.md; v18 f850 stale-validity bug; P57 preview invalidity | both (shared lib) |
| B10 | IMPL | Solver liveness not asserted | A solver can freeze variables and emit input bit-identical to output while reporting a good objective | Track B 7.25 mm frozen-translation row | both (shared lib) |
| B11 | IMPL | Non-watertight completed meshes | Signed distance/nonpenetration is undefined by construction; watertight-with-uncertainty-labels is buildable (TSDF/Poisson + trim labels) | v19 EPISTEMIC: "non-watertight … cannot support signed nonpenetration" | research |
| B12 | IMPL | Runtime hot spots | P18 tens of minutes on 5 s clips; renderer 1053 s/5 s before face budgets; visible-geometry builder 4.8 GB RAM/41 min | fixed-slice failure cluster 5; P41/P43/P45 | both |
| B13 | IMPL | Mask ownership contamination | Hand/sleeve/table pixels enter object masks; guard is bbox-pad subtraction, not per-pixel ownership | P07 self-check contract; P09 hand-bbox exclusion | research (delivery shares masks) |
| B14 | IMPL | Camera-convention/adapter calibration | Constant per-clip camera-frame rotation (3.4–3.8°) at the pinhole-adapter level unified several "hand" errors; fix belongs in adapter, not estimator | Track M/P resolution | delivery |
| B15 | DATA | No object GT in delivery regime; regime non-transfer | HOT3D (close-range, large hands, fisheye→pinhole) ≠ egoscale delivery regime; WiLoR 2D refit works at trash conditions (7–10 px) and fails on HOT3D (23–26 px floor) | Track K | research discipline |
| B16 | DATA | Monocular metric depth noise floor | UniDepth at hand pixels 26–45 mm abs-median; every downstream metric claim inherits it | Track J | research |
| B17 | DATA | Contact GT scarcity | Only proximity-derived contact from HOT3D poses; no patch/pressure labels in-house (EgoPressure exists externally); contact prevalence per slice must be estimated before contact machinery is applied | P51/P52; web: EgoPressure | research |
| B18 | DATA | Tiny benchmark, overfit pressure | 3 clips/900 rows; even/odd holdout proves smoothness, not transfer; no lockbox clips yet | hand_metric_mechanisms Track V caveat | research discipline |
| B19 | DATA | Single-view use of a multi-view dataset | HOT3D ships synchronized multi-view egocentric streams; v19 consumes one stream; HOT3D paper reports multi-view methods significantly outperform single-view on exactly our claim families | web: HOT3D CVPR'25 | research (cheap win) |

Severity ordering for the research track: B1–B4 are the load-bearing scientific blockers (they
directly cause the visible artifact defects: bad object body, inert graph, unresolved contact).
B5–B7 are the design blockers that determine whether the joint layer can ever be more than a
smoother. B9/B10/B11 are cheap engineering with outsized effect on trustworthiness of every
future experiment.

---

## 3. Topic 1 — Extracting rigid bodies from video more accurately

### 3.1 First-principles restatement

A rigid body over a frame span is exactly: a subset of scene surface whose pairwise geodesic/
Euclidean distances are conserved, plus an SE(3) trajectory. Its two state components have
different natural estimators:

- **Trajectory** is observable from *correspondence over time* (point p seen at t and t′ gives
  one constraint on T_t→t′). It does not require complete shape.
- **Shape** is observable from *pose-aligned accumulation* of all visible evidence across the
  clip. It does not require a generative prior except for never-seen regions.

V19 inverts this: shape is fixed first from one frame plus a generative prior (P12/P13), and
trajectory is then fit per-frame against depth/silhouette of that possibly-wrong shape
(P14/P15). Three measured consequences follow necessarily from that ordering:

1. **Shape error masquerades as pose error and vice versa.** P47: observed-only geometry
   improved visible mesh support (7.72 mm) but worsened source gap, residual, compatibility,
   and row count — shape and pose are jointly underdetermined under the current channels.
2. **Unobservable pose directions.** For a mostly-planar visible surface (keyboard), depth
   residuals are invariant to in-plane translation and rotation about the surface normal;
   silhouette constrains only extent. P48/P49/P50 falsified constant-rotation, smoothing, and
   silhouette repairs because none of them adds an observation of the null-space directions.
   The only channel that observes them is **surface texture correspondence over time** — which
   no phase currently measures.
3. **Anchor-frame appearance bakes into permanent geometry.** Tomato peel-horn; IoU
   0.375→0.553 from anchor choice alone with the same trajectory.

### 3.2 What the field says (and what transfers)

- **Per-video neural optimization** (HOLD CVPR'24; MagicHOI ICCV'25 adds NVS-diffusion priors
  for unseen regions + visible-contact alignment; BIGS CVPR'25 Gaussian-splat bimanual with SDS)
  is the accuracy SOTA for category-agnostic hand+object from monocular video — and is
  hours-per-clip. Under the project runtime invariant it is an **offline research/teacher
  branch only** (same status as BundleSDF). Its value here: a per-clip pseudo-GT teacher for
  distillation and for auditing our fast path.
- **Model-free 6D pose trackers** (FoundationPose CVPR'24: model-based or ~16 reference views,
  RGBD, near-real-time tracking) are the strongest drop-in pose baselines. Note the honest
  external evidence: on BOP-H3/HOT3D RGB-only, novel-object pose methods score low
  (GigaPose 9.4 AP, +GenFlow 31.2 AP) — novel-object egocentric RGB pose is an open problem,
  not a pick-a-model errand. With UniDepth pseudo-depth, FoundationPose becomes runnable on
  our clips and directly tests whether our pose stage or our information budget is the limit.
- **Feedforward 3D tracking/geometry** (SpatialTrackerV2 ICCV'25: joint video depth + ego-motion
  + object motion, 10–20 s/sequence; TAPIP3D NeurIPS'25: world-frame track refinement;
  VGGT CVPR'25: sub-second multi-view geometry/tracks; MegaSaM: robust dynamic-scene
  camera+depth) makes **correspondence-first rigid-body extraction compatible with the runtime
  invariant** for the first time. This is the key enabling shift since the v19 design was
  written.
- **Agentic/contact-aware reconstruction** (AGILE 2025: VLM-guided QC + generative meshes +
  "anchor-and-track" optimization replacing brittle SfM; CHOIR: staged open-world analysis →
  generative rectification → contact-aware optimization) — independent convergent evidence that
  (a) anchor+track is the right pose backbone and (b) watertight simulation-ready meshes are
  the right shape deliverable.

### 3.3 Proposed mechanism (research-track candidate, "correspondence-first rigid body")

1. Track dense/quasi-dense points on the object mask through the clip (2D tracker + per-frame
   metric depth lift, or a 3D tracker directly; camera-stabilized world frame).
2. **Rigidity statistic**: for candidate point subsets, residual of the best per-frame SE(3)
   Procrustes fit. This single quantity (i) verifies/falsifies the VLM rigid decision with a
   measurement (B5), (ii) segments articulated parts (multiple low-residual clusters with
   distinct trajectories), (iii) time-localizes deformation events (residual bursts, e.g.
   tomato peel), (iv) yields the pose trajectory itself as a by-product.
3. Pose from robust Procrustes/factor over tracks (texture correspondences observe exactly the
   directions depth-ICP cannot); depth/silhouette become *secondary* consistency channels.
4. Shape from pose-aligned multi-frame TSDF/surfel fusion of all visible evidence
   (hand-ownership-subtracted), watertight by construction with per-face provenance labels
   (`observed`, `interpolated`, `prior-completed`); TRELLIS (single- or multi-image
   conditioning) fills only never-seen regions and stays labeled uncertain — this is the
   pipeline-invariant-compliant version of completion.
5. The existing free-space/silhouette filters become *acceptance tests* evaluated on all
   frames, not post-hoc surgery on a single-anchor mesh.

This keeps every named v19 mechanism (TRELLIS completion, pose graph, render consumption)
and re-grounds each on the measurement that actually constrains it. It is monotonic: visible
fused surfaces remain the metric anchor, as the design already demands.

### 3.4 Known risks of the proposed mechanism

Track drift on textureless/blurred egocentric video; depth noise (B16) swamping the rigidity
statistic (this is decided by experiment E1, not by argument); hand-occlusion killing track
lifetime exactly during manipulation (mitigation: re-detection + track stitching; multi-view
HOT3D streams, B19); keyboards/symmetric objects still ambiguous when texture is repetitive
(mitigation: global appearance-template terms, OPFormer-style descriptor matching).

---

## 4. Topic 2 — Defining objectives in the factor graph

### 4.1 First-principles restatement

A factor graph is MAP inference in an explicit generative model: p(measurements | state) ×
p(state). Every measured v19 pathology maps to a specific modeling error, which makes the
redesign concrete rather than aesthetic:

| Measured pathology | Modeling error |
|---|---|
| Contact/support term has zero support in 900/900 rows; gate acts only as safety boundary | **Likelihood asks for evidence occlusion physically removes.** The measurement channel (visible-mask-overlap depth-order support under projected MANO) does not exist in the regime where contact happens. A likelihood must be defined over channels that exist: learned per-vertex contact priors from images (ContactOpt/DeepContact family, S²Contact), depth-order render-and-compare, temporal kinematic coupling (object moves iff grasped), latent adjacent-surface support |
| Raw optimizer attempts cm-scale unsupported wrist shifts; the *out-of-graph* output gate is what protects the artifact | **The declared model's MAP is wrong and is being overridden from outside.** The prior/data-term balance is not the fitted noise structure of the actual measurements; safety-by-override is standing evidence of misspecification |
| Contact-coupled MANO correction reduced surface residual by moving the hand ~10 cm and worsened GT by 8–11 cm | **A proxy objective with a huge basin captured a variable it does not observe.** Contact distance observes relative hand-object geometry only; letting it move absolute hand state without an absolute-channel counterweight guarantees this failure |
| Terminal hand error is smooth per-clip drift (b(t)/R(t), ~8 effective dof, autocorr 0.66–0.77); graph optimizes per-frame articulation/translation | **Missing latent variable.** No term can absorb error along a direction that is not in the state. The graph needs explicit low-dimensional per-clip drift latents (spline/GP over SE(3)-ish corrections) — and the falsified interventions (rotation-only, scale-only, per-frame) map out exactly which parameterizations are wrong |
| Contact-present vs absent decided by manual slice attribution (P51/P52) | **Missing discrete structure.** Contact is a switch variable; the standard machinery is switchable constraints (Sünderhauf & Protzel 2012), DCS, max-mixtures (Olson & Agarwal), GNC — jointly inferring the discrete hypothesis with the continuous state instead of assuming it |
| σ's hand-set; one modeled scale (30.48 mm combined contact σ) | **Uncalibrated noise models.** The self-consistency metric family is an unexploited *empirical noise source*: cross-detector keypoint deltas, size-consistency ratios, and reprojection residual distributions are per-frame samples of exactly the measurement noise the factors need |
| Umeyama world alignment gauge-invalid (Track B); object canonical frame vs BOP frame needs one fitted constant transform (P37) | **Unfixed gauge freedoms.** Per-clip world similarity, object canonical frame, and camera-convention rotation must be explicit variables with declared gauges, or "improvements" are gauge motion |
| 7.25 mm interval-solver row with output bit-identical to input | **Optimization artifact accepted as inference.** Solver liveness must be an asserted invariant (diff solved vs input) |

### 4.2 Objective redesign priorities (ranked by evidence of unlocked value)

1. **Drift latents with GT-free anchors** (couples to B4; delivery-relevant). Add per-clip,
   per-side slowly varying translation/rotation correction variables (raised-cosine spline,
   ~8–24 dof) to the hand stream. Anchor them only with GT-free residuals: cross-detector 2D
   reprojection (rtmlib vs WiLoR/HaWoR projection), projected-size vs detected-size, static-
   scene feature consistency, and (research) contact events. Track V proves the target error
   is representable; the open question is anchor observability (experiment E5).
2. **Contact as discrete-continuous mixture** (B6). Per-interval switch over
   {contact, near, none} with max-mixture/switchable-constraint semantics; the solved switch
   posterior replaces manual slice attribution and *is itself* the reportable contact state
   with uncertainty.
3. **Contact likelihoods from channels that exist under occlusion** (B3). Learned per-vertex
   contact maps (DeepContact-style, with the caveat that grasp-trained priors may mispredict
   flat-surface pressing — validate on HOT3D-derived proximity truth first); depth-order
   render-and-compare (does the hypothesized hand-in-front-of-object ordering reproduce the
   observed masks/depth?); object-motion-onset coupling (object trajectory begins moving ⇒
   grasp interval; a purely temporal channel with no occlusion problem). ContactOpt's
   soft-tissue observation (2–3 mm finger-pad, 5 mm palm deformation) also sets the *correct
   sign convention*: small penetration is expected at true contact, so a hard nonpenetration
   loss at 0 is a wrong likelihood.
4. **Fitted noise models** (B7). Fit per-channel σ (and heavy-tail shape) from self-consistency
   residual distributions per clip; factor weights stop being hand-set constants and become
   measured quantities with provenance. This is the designated bridge already named in
   `self_consistency_metrics.md`.
5. **Identifiability/liveness audit as a standing instrument** (B10 + observability). For each
   term: number of rows with measurement support, gradient magnitude relative to priors, and
   effect-on-GT of term removal ("term efficacy table", experiment E4). The fixed-slice
   zero-support finding was a null-space discovery made by accident; make it a tool.
6. **Gauge declaration.** One explicit per-clip world-gauge block (similarity to any external
   frame), one per-object canonical-frame transform, camera-convention rotation as a
   calibration variable (Track M showed it is real and constant per clip).

### 4.3 The honest status statement the research pack should carry

Today's hand-side "factor graph" is functionally HaWoR-passthrough + safety gate; the object
pose graph is a smoother over one-sided depth fits. That is the measured starting point, not a
verdict. The sequencing implication: **the graph needs measurements before it needs more
factors.** Objective work in priorities 2–3 only pays off after Topic-1 correspondence
measurements and contact channels exist; priority 1 and 4–6 can start immediately.

---

## 5. Topic 3 — Is detection + factor graph the right decomposition for HOI extraction?

### 5.1 The candidate decompositions

- **D0 (current):** semantic detection → instance masks/tracks → per-instance geometry
  (prior-completed) → per-instance pose (depth fit) → joint factor graph → render.
- **D1 (geometry/correspondence-first):** scene-level 4D backbone (video depth + camera +
  dense 3D tracks: SpatialTrackerV2/MegaSaM/VGGT-class) → rigid grouping by motion (the
  rigidity statistic of §3.3) → semantics attached to discovered bodies afterwards (open-vocab
  labels on track clusters) → same joint layer on top.
- **D2 (end-to-end feedforward HOI):** learn video → (hand, object, contact) directly. Current
  category-agnostic SOTA (HOLD/MagicHOI/BIGS) is per-sequence optimization, i.e. offline;
  feedforward HOI is nascent. D2 is the distillation *target* (subagent 11), not a
  pipeline-now option.
- **D3 (physics/simulator layer):** RL/residual policy tracks noisy kinematic estimates in a
  physics simulator, projecting them onto the dynamically feasible set (ManipTrans CVPR'25,
  DexMachina, SPIDER, QuasiSim lineage). This replaces E_contact/E_nonpenetration soft terms
  with hard simulator constraints — subagent 10's topic; the interface note below is what
  matters here.

### 5.2 Assessment against the local evidence

The measured v19 failures localize inside three D0 *components* — geometry-from-one-frame,
pose-from-depth-only, contact-from-visible-overlap — none of which is intrinsic to
"detection + factor graph". So the honest answer decomposes into two independent questions:

**(a) Is semantic detection the right geometric entry point?** Partly. Detection/tracking was
*not* the bottleneck on the fixed slice (SAM2 held the keyboard 150/150). But: (i) roster
completeness on real scenes (bowl/pot/surfaces/occluders) is unexercised and is a known miss
class; (ii) mask ownership contamination is a hard failure class (B13); (iii) detection cannot
answer "what is rigidly moving", which is the physically decisive property of a manipulated
object. Motion-based rigid grouping finds manipulated objects by that property and discovers
unknown-unknowns — but it cannot find physically relevant objects that never move (supports,
containers), where semantics is the only channel. **Conclusion: detection and motion discovery
are two evidence channels for the same object-instance state, not competing pipelines.** The
research pack should reframe "object detection" as "object instance hypothesis sources"
(semantic + kinematic), fused in the same roster.

**(b) Is a factor graph the right joint layer?** Yes for the annotation product, with the §4
repairs, because it is the only layer that (i) carries explicit calibrated uncertainty per
claim, (ii) preserves gauge/provenance bookkeeping, (iii) admits heterogeneous measurement
channels with per-channel noise, and (iv) produces the interpretable intermediate state a
distillation teacher needs. The alternatives do not currently deliver these: RL/physics gives
feasibility but no calibrated posterior and imports a sim-fidelity gap (unknown mass/friction/
soft tissue); end-to-end nets give speed but no per-claim evidence chain until distilled from
a trustworthy teacher. The graph's *justification* — joint contact/occlusion/nonpenetration
coupling — only becomes informative once object pose is trustworthy at ~cm and contact events
are detectable; before that the joint layer optimizes noise (which is exactly what the
fixed-slice showed). **Conclusion: keep D0's skeleton; import D1 as the measurement backbone
(tracks feed hands, objects, rigidity, and contact kinematics alike); treat D3 as a
feasibility-projection experiment family and D2 as the distillation endpoint.**

### 5.3 Interface notes for subagents 10/11

- For RL-as-factor-graph-approximation (10): the graph defines the posterior; a physics policy
  is an amortized projector onto the feasible set. The clean experiment interface is: same
  input kinematic streams, compare (graph MAP with soft physics terms) vs (simulator rollout
  tracking) on HOT3D-derived proximity truth + penetration/jitter self-consistency. The known
  hazard is reward proxy capture — DexMan's observation that nearest-surface proximity rewards
  encourage contact anywhere mirrors our P46/P50 proxy-capture falsifications exactly.
- For distillation (11): the teacher must emit *state + uncertainty + per-claim evidence
  provenance* (which is what the factor-graph layer uniquely provides); HOLD/MagicHOI-class
  offline reconstructions are complementary slow teachers for shape/pose pseudo-GT on clips
  where our fast path is weak. GraG-style temporally coherent long-sequence reconstruction
  (reported 6.4× faster than prior optimization work) is worth an offline evaluation as a
  teacher-cost reducer.

---

## 6. Concrete research questions

RQ1. At UniDepth/VGGT-class depth quality, what is the SNR of the pairwise-distance rigidity
statistic on egocentric manipulation clips — i.e., can rigid vs deformable vs articulated be
*measured* rather than declared? (B5, B16)

RQ2. Does object pose from temporal texture correspondence (robust Procrustes over tracks)
reduce the HOT3D keyboard trajectory residuals (clip001851 runtime baseline 38.85 mm median,
P46; clip001849 77.1 mm/4.93° median, P37) where depth-ICP variants were falsified (P46–P50)? Which residual directions remain, and are they symmetry-explained?
(B1)

RQ3. Does pose-aligned multi-frame fusion + prior completion restricted to never-seen regions
beat single-anchor TRELLIS completion on (i) GT chamfer of visible regions, (ii) all-frame
silhouette IoU, (iii) watertightness for signed distance? What fraction of the body remains
prior-dependent (irreducible shape uncertainty to be labeled, not hidden)? (B2, B11)

RQ4. Which GT-free anchors observe the per-clip smooth drift latent, and along which components
(lateral vs along-ray vs rotational)? Candidates: cross-detector reprojection, size
consistency, static-scene features, contact events. (B4)

RQ5. Can contact presence/absence per interval be inferred as a switch posterior from channels
that survive occlusion (learned contact maps, depth-order render-and-compare, object-motion
onset), reproducing the P51/P52 manual attribution on clips 001849 (none) and 001851 (near)?
(B3, B6)

RQ6. Do grasp-trained contact priors (ContactOpt/S²Contact family) transfer to non-grasp
contact regimes (flat-surface pressing, typing) — measured as AUROC against HOT3D
proximity-derived contact on the fixed slice? (B3, B17)

RQ7. What does each existing factor term contribute — support rows, gradient share, effect-on-GT
of removal? (Term efficacy table; B7, B10)

RQ8. Does multi-view egocentric input (HOT3D's other streams), used only as additional
measurement channels in the same graph, materially reduce hand lateral floors (8.9–30 mm) and
object pose residuals — quantifying how much of the current error is a single-view information
limit? (B15, B16, B19)

RQ9. What is the runtime/accuracy frontier of feedforward 4D backbones (SpatialTrackerV2,
VGGT+tracks, MegaSaM) on our clips versus the current UniDepth+HaWoR stack — are they viable
default-path substrates under the same-order-of-magnitude runtime invariant? (B12, enabling D1)

RQ10. Can an offline teacher (HOLD/MagicHOI-class, FoundationPose-with-pseudo-depth) produce
per-clip object pose/shape pseudo-GT good enough to audit and train the fast path on
delivery-regime clips that have no GT? (B15, distillation input)

---

## 7. Recommended experiments

All experiments follow the project research discipline: causal account first, discriminating
predictions declared before runs, every outcome must revise the model or force the next action.
All run on A800/offline branches with own run roots. Fixed slice = HOT3D clips 001849/001850/
001851 (+ add ≥2 lockbox clips before any tuning, per B18). Costs are rough A800 wall-clock.

**E1 — Rigidity-by-motion feasibility probe** (RQ1; ~0.5 day)
Track ≥200 points on existing SAM2 object masks (CoTracker-class 2D + UniDepth lift, and
SpatialTrackerV2 3D as second condition) on keyboard clips + tomato + trash bag. Compute
per-pair distance-conservation residuals and best-SE(3) Procrustes residual over time.
Predictions: (i) if rigidity is measurable at current depth quality: keyboard residual ≪
tomato-during-peel ≪ trash bag, with keyboard ≤ ~10–15 mm; (ii) if depth noise dominates: all
objects show similar residuals ≈ lifted depth noise (~30–45 mm) → motion-first classification
needs a better depth substrate (E8 becomes prerequisite); (iii) if tracking fails (short track
lifetimes under hand occlusion): the blocker is correspondence lifetime, not noise →
re-detection/stitching work item. Every outcome redirects: (i) → build the statistic into the
branch decision; (ii) → depth first; (iii) → tracker robustness first.

**E2 — Track-based object pose vs depth-ICP on the fixed slice** (RQ2; ~1 day)
Fit per-frame SE(3) from persistent tracks (robust Procrustes with track-confidence weights),
same mask/depth inputs otherwise; evaluate against HOT3D object GT with the existing P37
constant-gauge evaluator. Predictions: (i) correspondence-is-the-missing-measurement ⇒
translation residual drops materially below the per-clip baselines (38.85 mm on 001851,
77.1 mm on 001849) and rotation stabilizes without the P48 rotation penalty; (ii) depth-scale-bias-dominates ⇒ translation improves along-image-plane but
depth-axis residual persists → couple with drift/scale latent (E5-object variant); (iii) no
improvement with healthy tracks ⇒ the P37 residual is gauge/symmetry structure —
inspect residual axis alignment against keyboard symmetry before any further pose work.

**E3 — Multi-frame fused shape vs single-anchor completion** (RQ3; ~1–2 days)
Using E2 poses, fuse all-frame hand-subtracted object surfels into TSDF; complete never-seen
regions (TRELLIS conditioned on best crop; also try multi-image conditioning); label faces by
provenance; make watertight. Score: GT chamfer (HOT3D CAD) split by visible/hidden regions,
all-frame silhouette IoU, watertight yes/no. Predictions: (i) fusion beats anchor mesh on
visible-region chamfer and IoU (expected from tomato anchor evidence); (ii) hidden-region error
persists ⇒ quantified irreducible completion uncertainty → render as uncertainty, feeds B11
signed-distance semantics; (iii) fusion is worse ⇒ E2 poses are not accurate enough for
accumulation → loop back to E2 with the measured pose-error budget that fusion requires.

**E4 — Factor-term efficacy audit + solver-liveness invariant** (RQ7; ~0.5 day)
For each term in the P18 objective on the fixed slice: rows with measurement support, gradient
magnitude share, GT effect of ablating the term. Add a standing assert: solved state must
differ from input state (bit-diff) or the run is marked inert. Prediction: contact/depth-order
terms show ~zero support (known), hand-image/prior dominate; the output is the redesign
priority list with numbers. This experiment cannot fail to be informative; it is the
instrument the redesign consumes.

**E5 — GT-free drift-latent graph variable** (RQ4; ~2 days)
Add per-side spline drift latent (start translation-only, K_b≈24 raised-cosine, per Track V) to
the hand stream; anchor with GT-free residuals only (rtmlib cross-detector reprojection +
size-consistency); solve; evaluate against HOT3D GT (never fit to it). Predictions: (i) anchors
observe the drift ⇒ wrist medians move from 20–37 mm toward 10–15 mm; (ii) anchors observe only
lateral components ⇒ lateral improves, along-ray unchanged → add size-consistency-weighted
depth anchor next; (iii) no improvement while Track V's GT-fit b(t) closes sub-10 on the same
rows ⇒ the anchors are noise-dominated → measure anchor-residual vs GT-drift correlation
directly to find which anchor family has signal. This is the highest delivery-relevant research
experiment; its success criterion is also the delivery hand-accuracy mechanism.

**E6 — Contact-prior transfer test** (RQ6; ~1 day)
Run an image/geometry-based contact predictor (DeepContact-style per-vertex maps; plus a simple
2D hand-contact-state classifier) on the fixed slice; score AUROC against HOT3D
proximity-derived contact rows (001851: 205/300 rows <1σ; 001849: 0 rows). Predictions:
(i) high AUROC ⇒ contact becomes a measured channel → E7 is justified; (ii) systematic
false-positives on 001849 (hands hovering over keyboard) ⇒ grasp-trained priors do not
transfer to press/hover discrimination → training-data work item (EgoPressure, HOT3D-derived
labels) before any contact factor ships; (iii) chance-level ⇒ image channel is uninformative
here → depth-order + motion-onset channels are the only viable contact evidence.

**E7 — Switchable-contact mixture solve** (RQ5; ~1–2 days, after E6)
Implement per-interval contact switch (switchable constraint / max-mixture) over
{contact, near, none} with channels from E6 + object-motion onset + depth-order; solve jointly
with hand/object state on both clips. Predictions: (i) switch posterior reproduces P51/P52
attribution without manual slice selection ⇒ discrete-continuous machinery adopted; hands must
remain metric-MANO-preserved (P18b split stays); (ii) switch collapses to "none" everywhere ⇒
switch prior mis-scaled or channels too weak — the fitted σ's from E4/E6 say which; (iii)
switch flips frame-to-frame ⇒ missing temporal coupling on the switch variable (hysteresis
prior, as the renderer already learned for display).

**E8 — Feedforward 4D backbone probe** (RQ9; ~1 day, offline branch)
Run SpatialTrackerV2 and MegaSaM (and VGGT for geometry/intrinsics) on one HOT3D clip + one
egoscale clip. Measure: camera vs GT (HOT3D), depth vs UniDepth at hand/object pixels, track
density/lifetime on object surfaces, wall-clock. Predictions: (i) camera/depth at parity or
better with 10–20 s/clip runtimes ⇒ D1 substrate is viable for v20 default path; (ii) worse on
egocentric (fisheye, close hands, blur) ⇒ these models' training regime doesn't transfer →
they remain teacher/offline tools; either way the runtime table feeds the delivery/research
boundary decision.

**E9 — External pose-tracker baseline with pseudo-depth** (RQ10, supports RQ2; ~1 day)
FoundationPose model-free (reference views from the clip's clean frames; UniDepth pseudo-RGBD)
on the keyboard slice; score with the P37 evaluator. Predictions: (i) beats the per-clip
baselines (38.85/77.1 mm) ⇒ our pose stage is the limit, adopt/learn from its render-and-compare refinement; (ii) comparable or
worse ⇒ the single-view information budget is the limit → prioritize E8/multi-view (RQ8) over
pose-stage engineering. Also run the RQ8 multi-view arm here if HOT3D stream extraction is
cheap: same graph, add second-stream reprojection factors.

Sequencing: E4 immediately (instrument); E1→E2→E3 as the rigid-body chain; E5 parallel
(delivery-coupled); E6→E7 as the contact chain; E8/E9 as offline probes that can run anytime.
Kill/continue rules are the predictions above — a negative result redirects within the chain;
it does not justify hopping to an unrelated mechanism (AGENTS research discipline).

---

## 8. Failure modes of this research program (named, with countermeasures)

1. **Proxy capture** — optimizing silhouette/source-gap/support proxies that anti-correlate
   with GT (measured twice: P46, P50; external echo: DexMan on proximity rewards). Counter:
   every experiment reports GT residual *and* the proxy; a proxy-only win is a rejection.
2. **Benchmark overfit on 3 clips** — even/odd holdout proves smoothness, not transfer
   (Track V caveat). Counter: lockbox clips named before tuning; regime-transfer check
   (HOT3D↔egoscale) required for any promotion claim.
3. **Gauge motion masquerading as improvement** — Track B Umeyama lesson. Counter: declared
   gauges per experiment; the P37 constant-transform evaluator is the template.
4. **Offline-method creep into the default path** — HOLD/BundleSDF-class runtimes violating
   the runtime invariant. Counter: every experiment declares default-path vs offline-branch
   status up front; teachers never ship as stages.
5. **Inert-solver false positives** — accepting objective improvements from frozen or
   disconnected solves. Counter: liveness assert from E4 becomes a standing invariant.
6. **Contact-prior domain mismatch** — grasp-trained priors mislabeling hover-over-keyboard as
   contact. Counter: E6 transfer test gates any contact factor.
7. **Track failure modes** — drift on textureless surfaces, death under occlusion, WiLoR-style
   joint-label unreliability (labeled vs point-set residual 21.5 vs 11.5 px). Counter: track
   confidence enters as fitted noise; point-set (label-free) losses where labels are unreliable.
8. **VLM narrative overconfidence** — rigid/contact decisions without measurement backing
   (design's own expected failure mode). Counter: the rigidity statistic (E1) and switch
   posterior (E7) convert declarations into measured quantities; VLM remains hypothesis source.
9. **Depth-scale bias contaminating everything downstream** — B16. Counter: E2/E5 explicitly
   separate along-ray vs lateral residuals (the Track J decomposition is the standing first
   analysis on any new benchmark).
10. **Distillation from an untrustworthy teacher** — end-to-end training on states whose error
    structure is unmodeled. Counter: teacher states must carry calibrated uncertainty (B7) and
    pass the self-consistency family before entering any training set.

---

## 9. Keeping this research separate from delivery

Delivery track (per stakeholder): head/camera + hand accuracy (~5 mm target), semantic clip
captioning, high-throughput API — explicitly without the HOI factor-graph layer. The
separation contract:

1. **Read-only substrate sharing.** Research consumes delivery's measurement substrate
   (frames, depth, masks, hand streams, calibration contracts) as read-only inputs with
   pinned versions. Research may not demand delivery-side feature changes; requests go through
   the parent as explicit substrate contracts.
2. **Compute and state isolation.** Research runs on A800/offline branches under its own run
   roots; no shared mutable state; runtime-invariant exemptions are labeled offline-research
   per run. No research job on the delivery serving path or the user workstation.
3. **Promotion gate (the only door between tracks).** A research mechanism enters delivery
   only with all of: (a) fixed-slice GT win with declared predictions, (b) GT-free
   self-consistency win on delivery-regime clips, (c) runtime within the delivery budget,
   (d) regime-transfer evidence, (e) implementation-class review (liveness/freshness/gauge).
   Until then it ships nowhere, and delivery milestones never depend on a research outcome.
4. **Shared currency, two uses.** The self-consistency metric family is delivery's QC *and*
   research's GT-free acceptance instrument. One implementation, versioned together, so
   transfer claims are comparable across tracks (this is the designated bridge in
   `self_consistency_metrics.md`).
5. **Bug-class flowback.** Implementation-class discoveries made inside research (solver
   liveness, freshness boundaries, gauge/adapters, mask ownership) flow into shared libraries
   through normal review as isolated patches — never as wholesale research-branch merges.
6. **Claims hygiene.** Research reports state claims against the fixed slice + lockbox with
   provenance; delivery reports state claims against the delivery metrics/SLA. No claim
   migrates tracks without re-evaluation under the destination track's evidence standard.
7. **Drift-latent overlap is the one designed coupling.** E5's GT-free drift correction, if it
   works, is simultaneously the research graph's first live latent and delivery's hand-accuracy
   mechanism. It should be developed research-side and promoted through the gate — it is the
   deliberate first test of the promotion pipeline.

---

## 10. Sources

Local: `runtime/v19_runtime_spec.md`; `docs/pipeline_v19_design_proposal.md`;
`docs/v19_hot3d_fixed_slice_v1_results.md`; `.memory/tasks/2026-06-23-pipeline-v19/EPISTEMIC.md`;
`.memory/tasks/2026-06-21-v19-design/EPISTEMIC.md`; `.memory/project/pipeline_invariants.md`;
`.memory/project/hand_metric_mechanisms.md`; `.memory/project/object_geometry_and_render_lessons.md`;
`.memory/project/self_consistency_metrics.md`; `.memory/tasks/2026-07-04-demo-pack/EPISTEMIC.md`.

External (web, 2026-07-05):
- HOLD, Fan et al., CVPR 2024 (highlight); MagicHOI, ICCV 2025 (arXiv:2508.05506); BIGS,
  CVPR 2025 (arXiv:2504.09097); Follow My Hold (arXiv:2508.18213) — per-video HOI
  reconstruction lineage and generative-prior trend.
- FoundationPose, Wen et al., CVPR 2024 (arXiv:2312.08344) — model-based/model-free 6D pose +
  tracking; ~16 reference views, RGBD.
- SpatialTrackerV2, ICCV 2025 (arXiv:2507.12462) — feedforward video depth + ego-motion +
  object motion, 10–20 s/sequence; TAPIP3D, NeurIPS 2025 (arXiv:2504.14717) — world-frame 3D
  track refinement.
- VGGT, CVPR 2025 best paper (arXiv:2503.11651); MonST3R; MegaSaM — feedforward/robust
  dynamic-scene geometry backbones.
- HOT3D, CVPR 2025 highlight (arXiv:2411.19167) — multi-view egocentric hand+object GT;
  multi-view ≫ single-view; BOP-H3 2024/2025 novel-object RGB pose results (GigaPose 9.4 AP,
  +GenFlow 31.2 AP; OPFormer) — novel-object egocentric RGB pose remains open.
- ContactOpt, Grady et al., CVPR 2021 (arXiv:2104.07267) — learned contact maps (DeepContact),
  soft-tissue penetration semantics; S²Contact, CPF, NL2Contact follow-ups; EgoPressure —
  egocentric contact-pressure benchmark.
- Switchable constraints (Sünderhauf & Protzel, IROS 2012); DCS (Agarwal et al., ICRA 2013);
  max-mixtures (Olson & Agarwal); GNC (Yang et al., 2020) — discrete-continuous robust factor
  graph machinery.
- ManipTrans, CVPR 2025; DexMachina; SPIDER (arXiv:2511.09484); QuasiSim, ECCV 2024;
  Garcia-Hernando et al. residual-RL — physics-based tracking/retargeting; DexMan's
  proximity-reward critique; AGILE / GHOST / CHOIR (2025) — agentic, Gaussian, and
  contact-aware monocular HOI reconstruction; GraG — long-sequence temporally coherent HOI
  reconstruction (6.4× speedup claim).
