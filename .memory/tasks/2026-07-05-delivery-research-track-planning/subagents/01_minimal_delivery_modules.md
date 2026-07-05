# Subagent 01 — Minimal delivery-track vision modules (HOI/factor-graph/object-physical optimization dropped)

Scope: which vision modules the delivery track must keep to ship (a) separate head/camera and hand
error metrics both targeting the ~5mm band, (b) visible overlay without hand drift, (c) demonstrable
superiority over raw HaWoR/WiLoR — and which modules to delete. Evidence base:
`.memory/project/{hand_metric_mechanisms,object_geometry_and_render_lessons,self_consistency_metrics}.md`,
`.memory/tasks/2026-07-04-demo-pack/EPISTEMIC.md`, `runtime/v19_runtime_spec.md` (P00–P21),
demo-pack reports `review/track{B,I,J,K,L,M,V}_*.md` and `throughput/throughput_model.md` under
`/data2/ego_annotation_outputs/demo_pack_20260704`.

---

## 0. Claim architecture — what the stakeholder asks decomposes into three different "drifts"

The three requirements are governed by three physically distinct mechanisms. Conflating them is how
the current pipeline's evidence gets misread; separating them is exactly what the stakeholder's
"separate head/camera and hand metrics" request enables.

1. **Overlay 2D drift** (hand marks sliding off the visible hand in the rendered video). This is
   reprojection mismatch inside our *own* camera image: intrinsics contract errors, WiLoR
   weak-perspective/crop-convention mismatch, stale render layers, render-scale bugs. It does *not*
   involve GT registration at all. Proven fix stack exists (detector-anchored hybrid + Track L
   temporal fusion + projection contract): tomato 6.5/4.2px, phone 8.4/13.7px median reprojection
   residual in the demo. The stakeholder's tomato-drift hypothesis (intrinsics/reprojection/
   metric-space) points at the calibration-contract → renderer chain (modules D2, D7, D9 below).
2. **Metric hand error in camera frame** (the "hand ~5mm" metric). Raw HaWoR wrist medians are
   20–37mm on HOT3D. Track V decomposed this error: it is a *smooth, low-dimensional drift* —
   constant per-clip camera-frame rotation (2.4–4.0° ≈ 15–25mm at 0.35m, adapter/extrinsics-level)
   plus per-side slowly-varying translation bias b(t) (lag-1 autocorrelation 0.66–0.77) — sitting on
   a per-frame localization noise floor of **3.5–4.7mm** (oracle interpolation floor). A fixed
   a-priori spline family (K_rot=8, K_b=24, ≈8 effective dof) reaches **4.7–7.6mm** on all six
   clip+side cells under even/odd holdout. So the ~5mm hand band is *physically reachable with the
   current hand estimators* — but only demonstrated with GT-fitted per-clip calibration, a disclosed
   demo device. The deployable version of this claim requires one new module: **GT-free per-clip
   self-calibration of the smooth drift (D8)**. Everything else in the hand chain is already proven.
3. **Head/camera world trajectory error** (the "head/camera ~5mm" metric). This is the one
   requirement with **no supporting evidence today and no evaluator in the repo**. Track B measured
   predicted-world vs GT-world wrist trajectories: 100–250mm residual after best global SE3/Umeyama
   fit, HaWoR-SLAM metric scale off by 10–15% (Umeyama s = 0.85–0.92), trajectory *shapes*
   incompatible — i.e., the current camera source (HaWoR's masked-DROID SLAM, exported as
   R_c2w/t_c2w) is decimeter-class in absolute world terms on HOT3D near-static clips. Hand metrics
   survived because they are evaluated in camera frame. A separate head/camera metric makes the
   camera module a first-class deliverable for the first time; it needs its own evaluator, its own
   error budget, and probably its own upgrade path (D4).

Consequence for "better than HaWoR/WiLoR": superiority is already *proven* for articulation
(visible-joint MPJPE −21%…−48% held-out vs HaWoR, zero per-clip parameters), for metric placement
(WiLoR's own canonical-focal depth is 142mm wrist median — unusable), and for temporal/overlay
stability (fusion vs per-frame fits). Wrist-translation superiority over HaWoR is *pending D8* —
today the hybrid's wrist vector is bit-identical to HaWoR by construction (Track V, Axis 2).

### Error budget (hand metric, camera frame — from Tracks J/M/P/V)

| Term | Magnitude | Owning module | GT-free handle |
|---|---|---|---|
| Constant per-clip camera-frame rotation (conventions/adapter) | 2.4–4.0° ≈ 15–25mm @0.35m (real on 2/3 clips) | stream adapter / calibration contract (D2) | fix once per capture rig; static-scene epipolar/reprojection check |
| Shared smooth camera/registration drift | part of b(t) | camera module (D4) | static-scene feature consistency |
| Per-side smooth hand-branch drift (depth/lateral) | dominant b(t) share; lag-1 0.66–0.77 | hand chain (D5–D7) + D8 | 2D cross-detector reprojection (lateral), bbox size-ratio (depth) |
| Per-frame hand localization noise | 3.5–4.7mm median floor | HaWoR/WiLoR hybrid (D7) | temporal fusion smooths; this floor bounds the claim |
| Articulation (root-aligned MPJPE) | 22–27mm today; **not in the 5mm band by any current mechanism** | WiLoR visible geometry (D6) | rtmlib cross-detector residual |

The last row forces a metric-definition decision (§6): "hand ~5mm" is honest only for wrist/root
translation in camera frame; joint-level articulation at 5mm is beyond every current method and must
be quoted separately (root-aligned MPJPE ≈ 22mm on 1849).

---

## 1. Mandatory modules (the minimal set), and why each survives

Naming: D-numbers are the delivery module set; provenance in parentheses maps to v19 phases/scripts.

**D1. Ingestion + raw frame manifest** (P01). Decode, frame indexing, fps/resolution/timestamps.
Trivial cost; it is the contract root — every downstream claim ("same frame count and duration as
raw video") and every time-alignment question anchors here. Timestamps must be first-class: a
constant time offset between frames and any device pose stream is a candidate mechanism for smooth
drift.

**D2. Camera calibration contract** (P03b, `build_v19_calibration_contract.py` or capture-side
metadata copy). Single source of truth for fx/fy/cx/cy, distortion/rectification, camera model, and
axis conventions, consumed by HaWoR (focal extraction priority chain), the hybrid builder, the
renderer, and both evaluators. Why mandatory: the overlay-drift hypothesis and the Track M constant
rotation both live at this layer. Track M showed a 3.40/3.81° constant camera-frame rotation on
1850/1851, identical across prediction variants → adapter/convention-level, not hand-model-level;
+R alone halves 1851's wrist error. The delivery fix is a per-capture-rig convention audit in this
contract, not per-clip fitting.

**D3. Metric monocular depth — UniDepth** (P03, `run_unidepth_full_frame_v3.py`). Conditionally
mandatory: **mandatory for uncalibrated ingest** (it is the intrinsics source feeding D2) and for
SLAM scale anchoring/cross-check (HaWoR-SLAM scale is measurably 10–15% off; a metric depth prior is
the concrete correction mechanism); **QC-only when the capture rig supplies calibration**. Explicitly
*not* a hand-depth source (UniDepth at hand pixels: 26–45mm abs-median — ruled out, Track J).
Measured cost 0.51 GPU-h/video-h at 5fps (can drop further at keyframe rate for intrinsics/scale
only).

**D4. Head/camera trajectory module** (today: HaWoR's masked-DROID SLAM w/ scale, exported
R_c2w/t_c2w by `export_hawor_world.py`; standalone `run_droid_full_frame.py` exists). Promoted from
internal byproduct to first-class deliverable — it *is* the head/camera product. Roles: head/camera
metric itself, world-frame hand export, relative-motion prior inside fusion (HaWoR frame-to-frame
deltas), world-view rendering. Known state: locally usable relative motion (with clip-dependent
discontinuities — trash bridge had 7–11 m/frame jumps; always cap/robustify delta priors), but
decimeter-class absolute trajectories and 10–15% scale bias on HOT3D. Upgrade ladder (in expected
order of leverage): (a) ingest device VIO/SLAM trajectory as primary camera source when the customer
rig provides one (production-legitimate sensor metadata; the ~5mm camera claim then inherits the
device spec and our module becomes cross-check), (b) hand-masked DROID/DPVO + UniDepth metric scale
anchoring + static-scene bundle adjustment, (c) report RPE/scale rather than ATE if near-static
clips make absolute 5mm ill-posed. Which rung is reachable is *unmeasured* — first validation item
(V1).

**D5. HaWoR metric MANO** (P04, `remote_run_hawor_export.sh` → `export_hawor_world.py`). The metric
wrist/root translation source and the world-frame bridge. Nothing else in the stack provides metric
camera-frame hand translation: WiLoR's canonical-focal depth is unusable (142mm), UniDepth-at-hand
is ruled out, head-motion triangulation is ruled out (2–3mm baselines). Keep the focal-cache
invalidation guard (HaWoR caches focal-dependent artifacts keyed by pathname only).

**D6. WiLoR full-frame visible-hand evidence** (P04b, `run_wilor_full_frame.py`). Supplies (a)
root-relative visible MANO geometry — the proven articulation win: visible-joint MPJPE 47.2→33.0
(1850) and 46.7→36.9 (1851) held-out, left hands ≈−48%, zero per-clip parameters; (b) the 2D UV
anchor + detection confidence that ground overlay-no-drift and the lateral GT-free anchor; (c)
hand-presence evidence for absence/occlusion states. Measured optimized cost: YOLO detector 42.5
ms/frame at 2fps keyframes + batched regressor 5.27 ms/crop fp16 → 0.19–0.31 GPU-h/video-h (the
hand lane is no longer the throughput blocker).

**D7. Hybrid + temporal fusion hand layer** (P04b `build_v19_wilor_hawor_hybrid_hand_npz.py`
with `--translation-policy hawor_wrist_aligned`, plus the Track L fusion recipe productized).
This module *is* the delivered hand state. Required mechanisms, each tied to a measured failure it
prevents: source-switch hysteresis ≥5 frames (popping); banded least-squares translation with
per-frame reprojection-weighted WiLoR observations (per-frame independent fits jitter at p95);
HaWoR *relative-motion* deltas as smoothness prior, capped/robustified (bridge discontinuities);
quaternion smoothing for root orientation; never blend articulations across sources within a frame;
residual-driven band-local escalation (|fused−raw| > 50px for ≥3 consecutive detected frames) rather
than global re-weighting (global ×400 fixed one band and resurrected garbage accepts elsewhere);
per-frame visibility/absence states from detector evidence rendered as ghosts/absent (occlusion
discipline without SAM2).

**D8. GT-free per-clip self-calibration of smooth drift — NEW, the deployable-claim maker.**
Estimates the slowly-varying correction family that Track V fitted with GT: constant-to-slow R(t)
plus per-side translation bias b(t), ≈8 effective dof per clip. Why it is well-posed GT-free: every
dof of the fitted family has a named GT-free measurement that couples to it — shared rotation/camera
terms ← static-scene feature/epipolar consistency (D4); per-side lateral drift ← cross-detector 2D
reprojection consistency (rtmlib + WiLoR UV; residual 7–10px at 0.5–0.7m with f≈610px ⇒ ~10mm/frame
lateral constraint, aggregated over ≥300 frames against 8 dof); per-side depth drift ← projected-vs-
detected bbox size ratio (directly measures relative depth error; caught a ~22% depth error the 2D
residual endorsed). Without D8 the honest headline stays "wrist 20–37mm raw; sub-10 only under
disclosed GT calibration"; with D8 at GT-fit quality the band is 4.7–7.6mm (floor 3.5–4.7mm).
Regime caveat on the critical path: on HOT3D close-range hands the 2D anchor itself is broken
(23–26px floor from weak-perspective/crop-convention mismatch, Track K) — the full-perspective
crop-aware refit is therefore a *delivery-track prerequisite for proving GT-free 5mm on the only
GT benchmark we have*, even though the anchor already works in the egoscale delivery regime.

**D9. Overlay renderer with projection contract** (delivery subset of
`renderer/render_demo_presentation.py`). The visible artifact. Required mechanisms: skeleton +
shaded hand mesh (reads far better than mesh alone); per-frame confidence → alpha with 5-frame
median (flicker); provenance-of-current-layer outranks stale upstream validity fields (two shipped
bugs of this class: v18 `unresolved` hiding a WiLoR-supported override; trash f850); explicit
`scale_xy = render_size/source_size` projection rule in the manifest (960/1408-class bugs);
uncertainty rendered as ghosts, absence as absence. World view optional-tier: equal-scale oblique
projection, hand-anchored center, content-anchored grid, no camera-space positive-depth culling
(ghost-hands bug); gravity estimation stays presentation-only (worlds are drift-bent; 72%
correction ceiling measured).

**D10. Self-consistency QC stage** (per-clip, GT-free; from `self_consistency_metrics.md`). Emits
per frame/side: rtmlib↔primary-stream median keypoint delta (px), projected-vs-detected size ratio,
reprojection residual of fitted translation, temporal jitter (wrist mm/frame, root deg/frame,
median+p95), camera QC (static-scene residual, scale cross-check, delta-discontinuity count).
Why mandatory: every user-visible defect in the sprint corresponded to a large value in ≥1 of these
metrics, and each metric's blind spot is covered by another (2D-consistent-but-3D-wrong passes
metric 1 and is caught by metric 2). In production (no GT) this stage is the *entire* per-clip
accuracy story: it drives confidence tiers, re-estimation triggers, and the buyer-facing
verification protocol. rtmlib is CPU-cheap.

**D11. Benchmark evaluation harness** (offline, not per-customer-clip): HOT3D pinhole adapter +
existing hand evaluator (`evaluate_v19_hot3d_hawor_mano3d.py`: camera-frame wrist, MPJPE,
root-aligned MPJPE; claim-scope string already correct) + **new camera evaluator** (ATE-SE3, RPE@1s,
scale% vs GT poses; does not exist yet) + Track J ray/lateral decomposition as a standard diagnostic
+ the reconciliation check (decomposition must reproduce the evaluator's medians exactly before any
derived number is trusted).

Minimal-set cost check (why this is also the throughput answer): the accuracy lane is
UniDepth-keyframe (≤0.51) + hand lane (0.19–0.31, measured) + HaWoR/camera keyframe lane
(0.3–1.0, unmeasured) GPU-h/video-h — inside the 2.45 GPU-h/video-h budget *only because* the
deleted stack below (SAM2 full-rate 2.08–2.23 measured, TRELLIS per-object, 23.9min CPU solver per
5s clip) is gone from the default path.

---

## 2. Modules deleted from the delivery track, and why

Deletion is causally safe for the three delivery requirements because **the metric hand path is
already independent of the HOI stack by the pipeline's own design**: P18b exists precisely because
P18's surface-contact factors moved metric MANO wrongly, so v19's default runtime *preserves P04
metric MANO joints and demotes HOI output to a separate uncertain hypothesis*. Deleting P16–P18b
therefore costs zero hand accuracy. Object modules never fed the hand/camera metrics at all.

| Deleted (v19 phase) | What it is | Why delete |
|---|---|---|
| P05 object plan (VLM) | object naming/prompt planning | object-only; no coupling to head/hand metrics or overlay-drift |
| P06 OWLv2 box prompts | text-grounded object detection | same |
| P07 SAM2 object masks/tracks | object segmentation/tracking | same; also the single largest measured throughput item (2.08–2.23 GPU-h/video-h full-rate) |
| P09 visible metric geometry + anchor proposal | mask+depth lifting, anchor decision | serves object completion only; also RAM-heavy (4.8GB/41min per 450f) |
| P10–P13 branch decision, evidence bundle, TRELLIS, completion | per-instance object mesh reconstruction | object-only; TRELLIS ~2.8 min/object; anchor-frame policy burden |
| P14–P15 pose fit + temporal rigid pose graph | object 6DoF | object-physical optimization, explicitly out of scope |
| P16 MANO/object constraint measurement | HOI factors | HOI; quarantined from metric MANO by P18b anyway |
| P17 contact/occlusion prior rows | contact/ownership factors | HOI; hand visibility/absence is re-derived from detector evidence in D7 |
| P18/P18b interval MANO correction + split | factor-graph interval solver | the interval solver froze wrist translation when it "improved" the objective (bit-identical outputs, Track B); 23.9 min pure CPU per 5s clip; its only safe mode is "don't move metric MANO" — i.e., a no-op for delivery |
| v17 factor-problem builders (`build_v17_*`) | prior-generation factor machinery | superseded; research-track archaeology |
| BundleSDF/NeRF branches | per-instance neural reconstruction | already research-only by project invariant (runtime discipline) |
| Gravity estimation ladder | world-view leveling | presentation-only; measured 72% correction ceiling on drift-bent worlds; keep as optional render flag, not accuracy path |
| World-trajectory Umeyama/SE3 alignment as a *hand* metric | gauge test | gauge-invalid for hands (100–250mm residuals); survives only inside the camera evaluator (D11) where trajectory alignment is the correct object |

Two scope notes. (a) The captioning/semantic deliverable may re-import SAM2/OWLv2 for *semantics*;
that is a different lane with its own budget — nothing in the head/hand accuracy chain consumes
them. (b) Everything deleted here continues in the research track; this report only asserts the
delivery track's default path must not execute or wait on any of it.

---

## 3. Module I/O contracts and calibration requirements

Contract style: every module declares inputs with provenance, outputs with schema, the calibration
fields it consumes, and its failure semantics (blocker vs uncertainty-carrying row). Paths follow
the v19 run-root layout unless renamed by the API subagent.

**D1 ingestion** — In: raw video (+ optional capture metadata sidecar). Out:
`input/raw_frame_manifest/manifest.json` (frame paths, count, fps, width/height, per-frame
timestamps, decode provenance). Requirement: timestamps mandatory even if synthesized from fps —
declared as such. Failure: any decode gap is a blocker (frame-count contract roots here).

**D2 calibration contract** — In: capture metadata if present, else D3 output. Out: one canonical
`state/calibration/v19_camera_calibration_contract.json`: `focal_px` (+`intrinsics_fx_fy_cx_cy`),
principal point, camera model + distortion/rectification declaration, **axis-convention
declaration**, source (`capture_metadata` | `unidepth_estimated`), and per-rig convention-audit
record (the Track M rotation check result). Consumers must use the exact focal-extraction priority
chain (`focal_px` → `focal_geom_px` → `intrinsics_fx_fy_cx_cy[0]` → `intrinsics.fx`) and never read
diagnostics tables. One contract per clip; every consumer records which file it read. This is the
single leverage point for the stakeholder's intrinsics/reprojection drift hypothesis.

**D3 UniDepth** — In: manifest frames (keyframe rate acceptable for intrinsics/scale roles). Out:
`measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz` + QC JSON (metric
depth, estimated intrinsics, per-frame validity). Calibration: none in, candidate intrinsics out.
Failure: missing output is a blocker only for uncalibrated ingest; otherwise QC-degraded.

**D4 camera trajectory** — In: manifest, D2 focal, hand masks from the HaWoR pipeline (dynamic-
region masking), optional D3 depth (scale anchor), optional device VIO stream. Out: per-frame
`R_c2w`, `t_c2w`, scale + scale-source, per-frame validity/confidence, delta-discontinuity flags
(cap threshold recorded), SLAM provenance (`hawor_slam_w_scale_*.npz` or successor). Contract
change vs today: this NPZ becomes a first-class published artifact with its own QC row, not a
byproduct inside the hand NPZ. Calibration: consumes D2 exactly; declares the world-frame gauge
(arbitrary origin; gravity NOT guaranteed). Failure: SLAM divergence → per-frame invalid rows +
clip-level camera-tier downgrade, never silent identity poses.

**D5 HaWoR** — In: video/frames, D2 focal (explicit; forced cache refresh on focal change). Out:
`measurements/hand_candidates/hawor_world/hawor_world_hands.npz` + QC (MANO params, camera- and
world-frame joints, per-frame validity, R_c2w/t_c2w passthrough until D4 is split out). Calibration:
focal cache invalidation guard is part of the contract. Failure: missing NPZ is a blocker (no
metric translation source exists without it).

**D6 WiLoR** — In: video/frames; detector at keyframe rate + batched regressor on crops. Out:
`measurements/hand_candidates/wilor_full_frame/wilor_raw.json` (per frame/side: MANO geometry,
crop/UV, detection confidence, crop-convention fields needed by the full-perspective refit). Out
must preserve raw crop parameters — the Track K fix is impossible without them. Failure: model-asset
missing is a blocker; low-confidence rows are uncertainty, not failure.

**D7 hybrid+fusion** — In: D5 NPZ, D6 raw, D2 contract. Out:
`measurements/hand_candidates/wilor_hawor_hybrid/wilor_visible_hawor_wrist_hybrid_hands.npz` +
report with per-frame/side provenance (`wilor_visible` | `hawor_fallback` | `absent` |
`occluded_ghost`), fusion diagnostics (band escalations, hysteresis switches, delta caps hit), and
`translation_policy=hawor_wrist_aligned`. Invariants: wrist translation from D5 (+D8 correction
when present); articulation never blended within a frame; solver-liveness proof required (diff
solved vs input — the interval solver shipped bit-identical output while reporting success).
Failure: WiLoR absent → blocker; per-row gaps → fallback rows with provenance.

**D8 self-calibration** — In: D7 state, D10 raw measurements (cross-detector residuals, size
ratios), D4 static-scene residuals. Out: per-clip correction record `state/self_calibration.json`:
family (R(t)+b(t) per side), knots/effective dof, per-dof GT-free observability report, correction
applied/not-applied decision with residual evidence, and the *uncorrected* stream retained as
provenance. Calibration requirement: capacity control fixed a-priori (K_rot=8, K_b=24-class,
knot spacing ≈ 2× measured residual correlation time) — per-clip capacity selection is the
overfitting boundary Track V documented; never fit per-clip capacity in production. Failure:
low-observability clip (few detected hand frames, no static scene) → no correction + explicit tier
downgrade, never a silent identity correction.

**D9 renderer** — In: D7 (+D8) state, D2 contract, manifest. Out: full-duration overlay (+optional
world/side-by-side) + manifest recording `scale_xy`, style, per-layer provenance. Invariants:
current-layer provenance gates display; confidence alpha (5f median); uncertainty=ghost,
absence=absent; frame count == raw video. Failure: any manifest field empty → render failure, not a
warning.

**D10 QC** — In: D6/D7 states, rtmlib CPU pass over frames, D4 diagnostics. Out: per-clip QC JSON +
per-frame rows: `rtmlib_wilor_median_keypoint_delta_px`, size-ratio, reprojection px, jitter
median/p95, camera QC, tier assignment with the pairing rule (metric-1 large OR metric-2 large →
degraded; both small → tier A). Contract: QC consumes the *current* delivered layer, never stale
upstream labels.

**D11 evaluators** — In: benchmark GT (HOT3D MANO + device poses), prediction NPZs, D2 contract.
Out: per-clip hand metrics (wrist median, MPJPE, root-aligned MPJPE — definitions frozen as in the
existing script), camera metrics (ATE-SE3, RPE@1s, scale%), ray/lateral decomposition, and a
mandatory reconciliation assertion between decomposition and evaluator medians. Gauge handling
declared per metric (camera ATE alignment is legitimate; hand metrics stay camera-frame, never
world-aligned).

Cross-cutting calibration requirements: (i) one contract file per clip, all consumers cite it;
(ii) per-rig convention audit (rotation check) happens once at rig onboarding, recorded in D2, so
Track M-class constant rotations are configuration, not per-clip fitting; (iii) any time-offset
between frame timestamps and device pose streams is estimated and recorded at onboarding;
(iv) focal-dependent caches must key on focal.

---

## 4. Accuracy bottlenecks, ranked, with expected measurable contribution

1. **Camera/adapter convention rotation (D2).** Evidence: constant 3.40/3.81° on 1850/1851,
   even/odd-stable, prediction-variant-invariant; +R alone halves 1851 wrist error. Expected
   contribution: −40…−60% wrist median on affected rigs, from a one-time per-rig audit. Cheapest
   item on the list; do first.
2. **GT-free smooth-drift estimation (D8).** The gap between raw (20–37mm) and calibrated
   (4.7–7.6mm) wrist medians is entirely this term. Expected contribution scales linearly with the
   fraction of drift the GT-free anchors recover; ceiling = GT-fit quality (floor 3.5–4.7mm).
   Discriminating risk: if the GT-free estimate fitted on even frames does not track the GT-fitted
   b(t) (correlation test, V5), the anchors are too weak and the ~5mm wrist claim degrades to
   "sub-10 under disclosed calibration" — a stakeholder-visible difference. This is the delivery
   track's hardest essential module.
3. **WiLoR crop-convention/full-perspective refit (D6/D8 prerequisite).** Evidence: 2D refit floor
   is 7–10px in the egoscale regime but 23–26px on HOT3D close-range hands; mechanism is
   weak-perspective/crop mismatch growing with hand_size/depth. Contribution: unlocks the lateral
   GT-free anchor *on the GT benchmark* (without it, D8 cannot be validated at 5mm where GT exists)
   and removes a regime boundary from the overlay-no-drift claim.
4. **Head/camera absolute accuracy (D4).** Current evidence says decimeter-class world trajectories
   and 10–15% scale bias; ATE has literally never been measured (no evaluator). Expected
   contributions, by mechanism: depth-anchored scale correction removes the 10–15% scale term;
   static-scene BA attacks trajectory shape; device VIO ingestion (where available) likely jumps
   straight to the target band. Until V1 runs, "~5mm head/camera" is an open commitment — flag to
   stakeholder as measured-unknown, not as promised.
5. **Per-frame hand localization floor (D5–D7).** 3.5–4.7mm median floor bounds every wrist claim;
   approaching it needs a better per-frame estimator (research track), not more calibration.
   Lateral floors from static families (8.9–10.9mm best case) are already absorbed by time-varying
   b(t); do not re-spend on static per-clip scalar/curve families — ruled out (plateau 12.0mm;
   frozen scalars regress held-out 32.8→40.1).
6. **Articulation (D6).** Hybrid already banks −21…−48% visible-joint MPJPE vs HaWoR held-out.
   Root-aligned MPJPE (~22mm) will not reach 5mm; report honestly, park improvement in research.
7. **Temporal jitter / overlay stability (D7/D9).** p95 jitter and band-local divergences are
   presentation-visible; recipe measured (v3c band-local: p95 81/90 vs 78/89 baseline while fixing
   the divergent band). Contribution: keeps the overlay claim true at all frames, not on average.

---

## 5. Minimal validation suite — head/camera and hand proven separately

Design rule (project research discipline): every experiment states discriminating predictions
before running; every metric must reconcile with the evaluator medians before derived claims.

### Head/camera (new capability — currently zero evidence)

- **V1. Camera trajectory evaluator on HOT3D** (build in D11): per clip — ATE after declared SE3
  alignment, RPE@1s, scale%. Predictions: if SLAM is locally sound but globally drifting, RPE is
  small while ATE carries the drift (fixable by BA/anchoring); if scale-dominated, scale% ≈ 10–15%
  and depth-anchoring (D3→D4) removes it; if trajectory shape diverges even at 1s horizon, the
  near-static regime is SLAM-hostile and the camera claim must move to device-VIO ingestion or a
  moving-camera benchmark. Each outcome selects a different D4 rung — that is the point.
- **V2. Drift-identity test** (one-shot, decisive): correlate GT-fitted per-side b(t) (Track V
  pairs NPZs, reusable) with the camera registration error (GT camera ⊖ predicted camera) over
  time, per clip. Prediction if drift is shared camera registration: high correlation, side-shared
  component dominates → D8's camera anchors matter most. Prediction if per-side hand-branch drift
  dominates (Track V's shared-rotation oracle failure suggests this): low camera correlation,
  per-side residual dominates → D8's detector/size anchors matter most, and camera and hand claims
  decouple cleanly (good for the stakeholder's separated-metric framing).
- **V3. GT-free camera QC transfer** (production gate): static-scene feature reprojection RMS,
  scale cross-check vs D3 at static regions, delta-discontinuity count. Calibrate these gates on
  HOT3D against V1 outcomes (gate value ↔ measured ATE/RPE band), then apply as the per-clip camera
  tier in production.
- Regime coverage: HOT3D is near-static (2–3mm inter-frame baselines) — adversarial for absolute
  SLAM. Add one moving-camera GT source before hardening the camera band (candidates: Aria ADT /
  Nymeria-class MPS-trajectory sets, or a one-day in-house rig capture with fiducials; choose in
  the task pack — do not promise 5mm from HOT3D alone).

### Hand (extend the proven protocol)

- **V4. Camera-frame hand evaluator** (exists): wrist median, MPJPE, root-aligned MPJPE per
  clip/side, claim scope frozen ("camera coordinates; wrist-subtracted alignment only; not
  contact/occlusion/object"). This stays the primary hand table.
- **V5. D8 acceptance**: fit GT-free self-calibration on all frames, evaluate on GT with V4, and
  compare against (a) uncorrected, (b) GT-fitted fixed-config spline (the 4.7–7.6mm reference),
  (c) oracle floor (3.5–4.7mm). Also verify solver liveness (corrected ≠ input bitwise) and
  correlation between GT-free and GT-fitted b(t). Predictions: recovers ≥70% of drift → wrist
  lands ≤8mm and the deployable claim holds; tracks poorly → anchors too weak, escalate the Track K
  refit or add static-scene anchors before re-attempting; improves even frames only → capacity
  leak, tighten the fixed-capacity rule.
- **V6. Baseline-superiority table** (the "better than HaWoR/WiLoR" artifact): identical frames,
  identical protocol — raw HaWoR; raw WiLoR (canonical-focal metricfit, documented 142mm, plus
  2D-only column); hybrid; hybrid+D8. Columns: wrist median, MPJPE, root-aligned MPJPE, overlay
  reprojection px p50/p95, jitter p95. Rules learned: even/odd holdout wherever anything is fitted;
  state explicitly that interleaved holdout proves smoothness, not cross-clip transfer; cross-clip
  transfer is claimed only from the fixed a-priori config applied to untouched clips.
- **V7. Ray/lateral decomposition + reconciliation** (Track J protocol) on every new benchmark
  clip before any tuning: identifies which mechanism dominates that clip (depth-scale vs lateral
  vs rotation) and catches evaluator/adapter bugs by forcing exact reconciliation with V4 medians.

### Overlay no-drift (per delivered clip, GT-free)

- **V8. Per-clip QC bundle** (D10): reprojection residual p50/p95 per side (target: egoscale-regime
  7–10px class), size-ratio band (catches depth-wrong-but-2D-right), rtmlib delta p50/p95, jitter
  p95, absence/ghost accounting — plus the demo-proven human protocol: QC video with per-frame
  provenance chips, N-frame spot-check on the final rendered artifact (consumption, not existence).
  Acceptance is judged on the rendered video; the QC numbers select which frames a human inspects.

Minimal suite total: two new evaluators (camera V1, D8-acceptance V5), one decisive one-shot
analysis (V2), one table (V6), and the QC bundle (V8) that already exists in demo form. Everything
else reuses frozen demo-pack machinery (Track J decomposition, Track V pair NPZs, evaluator).

---

## 6. Decisions the task pack must pin (material to scope)

1. **Metric definitions for the 5mm band**: wrist/root camera-frame median (reachable) vs
   root-aligned MPJPE (not reachable by any current mechanism, ~22mm) for hands; ATE vs RPE vs
   scale for head/camera. The stakeholder's "~5mm for both" must be bound to specific definitions
   before the delivery contract is written — this changes whether the target is engineering
   (D8 + D2) or research (new hand estimator).
2. **Camera source policy**: is device VIO/SLAM metadata available on customer rigs? If yes, D4
   becomes ingest+cross-check and the camera band is inherited; if no, D4 is a build item with V1
   deciding feasibility.
3. **Regime scope of the overlay claim**: egoscale-regime (0.5–0.7m hands) is proven at 7–10px;
   close-range large-hand regimes need the Track K refit first.
4. **Benchmark set for the camera claim**: HOT3D alone is insufficient (near-static); pick the
   moving-camera GT source.

Bottom line: the minimal delivery set is **D1–D11 ≈ ingestion, calibration contract, UniDepth
(conditional), camera trajectory, HaWoR, WiLoR, hybrid+fusion, GT-free self-calibration, renderer,
QC, evaluators**. Deleting the object/HOI stack costs zero hand/camera accuracy (P18b already
quarantined it) and is what brings the compute lane inside the throughput budget. Of the three
stakeholder requirements: overlay-no-drift is engineering of proven mechanisms; hand-5mm is one new
well-posed module (D8) plus one known fix (crop refit) away, with the floor measured at 3.5–4.7mm;
head/camera-5mm is the only requirement with no current evidence in its favor — its first
deliverable is a measurement (V1), not a promise.
