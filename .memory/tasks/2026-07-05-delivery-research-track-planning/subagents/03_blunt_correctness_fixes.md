# Subagent 03 — Blunt mechanism-level correctness fixes before the product API

Scope: the tomato "visible hand vs annotation drift" the stakeholder saw, plus every
coordinate/intrinsics/crop-scale/camera-frame/head-metric/renderer-path defect that must be
fixed before an API ships. All numbers below were measured on 2026-07-05 from the shipped
demo state (read-only analysis; no code edited). HOI/contact/factor-graph is out of scope.

---

## 0. The drift is real, quantified, and its mechanism is known

I reprojected the **shipped** tomato hand layer
(`demo_pack_20260704/tomato_hands_y/hybrid/wilor_temporal_fused_hybrid_hands.npz`, the
exact override rendered into `tomato_grasp_demo.mp4`) against WiLoR's own full-frame 2D
joints, per frame, under the render intrinsics and render camera poses:

| side | frames measured | median px | p90 | p95 | frames >20px | worst runs |
|---|---:|---:|---:|---:|---:|---|
| left | 852 | 8.0 | 20.0 | 31.6 | 85 (10.0%) | f150–159, f168–176, f182–194, f202–211 (contiguous multi-second block f147–f321) |
| right | 874 | 7.2 | 17.2 | 24.5 | 60 (6.9%) | f194–206 (13 consecutive), f926–946 |

Breakdown by provenance of the shipped rows:

- `hawor_bridge_fallback_temporal_fused` rows that coincide with a WiLoR detection:
  **median 99.7px off (left), max 164px** — the fallback hands sit a tenth of the frame away
  from the visible hand. 118 left / 76 right frames are fallback.
- `wilor_visible_temporal_fused` (detector-anchored!) rows: median 8.0px but **max 104.6px
  (left) / 255.6px (right)** — the temporal fusion drags even detector-anchored frames off
  the evidence.
- The pre-fusion crop-scale fit had accepted-row median 6.54/4.25px, p95 14.0/11.6px
  (trackY report). The fusion stage **doubled the tail** (p95 31.6/24.5px measured above).

A 20–100px offset on a 960px frame with a ~120px hand is exactly "the annotation drifts
around the hand." The stakeholder saw a real defect that the demo QC understated (see FIX-5).

---

## 1. Ranked root-cause hypotheses (with status and discriminating tests)

### H1 — CONFIRMED (dominant visible mechanism): temporal fusion drags hands off detector evidence
**Mechanism.** The v3c fusion solves banded least-squares translation with HaWoR bridge
*relative-motion* deltas as the smoothness prior. Those deltas live in the bridge's metric
space, which is rescaled **per frame** by `hawor_to_v18_depth_scale` (measured on tomato:
median 0.508, p5–p95 0.426–0.679, frame-to-frame |Δscale| p95 = **0.38**). A prior whose
translation deltas are elastically wrong per frame systematically fights the WiLoR data
term; band escalation only fires at |fused−raw|>50px for ≥3 consecutive frames, so
20–50px divergence persists everywhere else and 50px+ divergence persists in short bursts.
**Evidence.** Table above; band-escalation spec in
`.memory/project/hand_metric_mechanisms.md` (Tracks T/v3c: fused-vs-raw p95 81/90px on
trash); Track V: bridge relative motion contained 7–11 m/frame discontinuities on trash.
**Discriminating test (already run, positive).** Fused-vs-WiLoR2D per frame, by source:
accepted-source frames alone reach 100–255px; fallback frames ~100px. Both sub-mechanisms live.

### H2 — CONFIRMED (secondary, worst-magnitude frames): fallback rows are metric-inconsistent bridge relifts
**Mechanism.** When WiLoR has no accepted row, the shipped row is the HaWoR bridge relift
(`hawor_bridge_candidates_current_v18_camera_local.npz`): HaWoR vertices projected and
relifted through **per-frame UniDepth depth + per-frame UniDepth intrinsics + per-frame
depth scale**, then rendered under the constant frame-0 intrinsics. Three inconsistencies
stack: elastic depth scale (0.43–0.68), per-frame vs frame-0 intrinsics (radial expansion
error ≈ (1−fx_f/fx_0)·r px), and HaWoR's wrong focal assumption (below).
**Evidence.** Measured 40–164px offsets on fallback rows; trackY report itself: "old bridge
hands were much smaller and/or displaced."

### H3 — CONFIRMED (root of both, the stakeholder's named suspicion): no calibrated camera model exists
Three incompatible intrinsics conventions coexist on one clip:

| consumer | intrinsics | implied HFOV @960×540 |
|---|---|---|
| HaWoR (hand depth + **camera trajectory SLAM**) | default `[2304,2304,960,540]` @1080p ≡ fx 1152 | ~45° |
| UniDepth (depth + the "v18 camera intrinsics") | per-frame estimate, fx **385–653** across the tomato clip (median 541), fx≠fy by ~9% | ~77–103° |
| Renderer + trackY refit | frame-0 sample `[441.6, 486.5, 486.8, 271.4]` (18% below clip median) | ~95° |

- The per-frame `hawor_to_v18_depth_scale` median 0.508 ≈ 541/1152 = 0.47: the "metric
  bridge scale" is **mostly a patch over the focal-assumption mismatch**, with UniDepth
  per-frame noise on top. The metric space is per-frame elastic by construction.
- fx≠fy by 9% is estimator noise, not physics. Prediction it makes: a 3-dof translation
  fit of a rigid hand under anisotropic-wrong K leaves an irreducible residual ≈ 4.5% of
  hand pixel extent ≈ 4–7px — which matches the observed accepted-fit medians 4.25/6.54px.
- Absolute metric depth of the accepted hands scales with the arbitrary fx choice
  (z ≈ fx·scale): fx 441.6 vs clip median 541 ⇒ shipped hand depths/sizes carry ~±20%
  unvalidated metric error even where the 2D overlay looks aligned. This is the
  "metric-space" half of the stakeholder's remark.
- The **camera trajectory** (`T_world_camera_metric`) comes from HaWoR's SLAM run under the
  ~2.1× wrong focal. SLAM under badly wrong K misattributes rotation vs translation. Head
  pose at ~5mm is impossible on this foundation; it also plants slow camera-frame drift that
  later shows up as "hand error" (Track M/V: R(t) 3.4–3.8°, b(t) drift — the same error
  class, observed even on HOT3D where K is exact, i.e., SLAM/registration drift exists on
  top of the K problem).
- No lens distortion handling exists for ego clips (only the HOT3D FISHEYE624 adapter).
  At ~95° HFOV, an uncorrected consumer lens puts tens of px of radial error near edges.

**Discriminating tests (cheap, run before building):**
- **T1 (K refit test):** re-run the trackG crop-scale fit under fx=fy=541 (clip-median K).
  Predictions: accepted-frame median reprojection drops toward ~2–4px (anisotropy was fake)
  and fitted depths shift ~+22% coherently. If residuals do NOT drop, pixels are genuinely
  anisotropic (implausible) or per-frame K noise dominates → needs T3 anchor.
- **T3 (independent K):** SfM self-calibration (COLMAP/GLOMAP on keyframes, or DROID with
  intrinsics optimization) on tomato. Agreement with UniDepth clip-median within a few % ⇒
  robust per-clip self-calibration is viable; disagreement ⇒ device calibration required.
- **T-dist (distortion):** fit a 1-parameter radial model in the SfM step; if k1 is
  significant, undistortion at ingest becomes part of FIX-1.

### H4 — RULED OUT: global frame/time misalignment
Shifting the shipped layer by k ∈ {−2..+2} frames against per-frame WiLoR 2D minimizes
error exactly at k=0 for both sides (8.05/7.19px at k=0 vs 9.4–13.1px at |k|=1,2).
No off-by-one indexing between annotation rows and decoded frames. Do not spend time here.

### H5 — LIVE PRODUCT RISK (not the shipped-video mechanism): renderer stale-layer/resume paths
TrackY rendered with `--no-resume`, so the shipped tomato frames are current. But this bug
class fired **twice** already (v18 `unresolved` label hid a WiLoR-supported override at
trash f850; stale-validity gating in v18 renders), the renderer resumes by default, and
`load_v18_annotations` silently falls back to `[500,500,480,270]` intrinsics when the
annotation lacks the field — a silent-fallback that would produce exactly "annotation
drift" with no error. These are structural product defects even though they didn't produce
this particular complaint.

### H6 — BOUNDED, affects measurement not mechanism: WiLoR 2D is a self-anchor, not ground truth
Everything above is measured against WiLoR's own 2D joints. Under occlusion WiLoR joint
*labels* are unreliable (origami: labeled vs point-set residual 21.5 vs 11.5px), and part of
the f147–f321 left block is the tray/bowl occlusion period. Some of the ">20px" there may be
WiLoR 2D being wrong rather than the fused layer — but that does not rescue the artifact:
in those frames the shipped video draws a confident hand that matches *neither* the detector
nor the visible evidence. The independent anchor for acceptance is rtmlib (metric #1 of
`self_consistency_metrics.md`), which must be run per delivery clip (see FIX-5/T2).

---

## 2. Fixes required before the product API (ranked; each names the mechanism and its acceptance evidence)

### FIX-1 — One calibrated camera model per clip, used everywhere (root fix)
Establish a single constant pinhole K (+ distortion) per clip/device before any model runs:
- Source ladder: (a) device/session calibration or video metadata when available;
  (b) robust per-clip self-calibration = UniDepth intrinsics aggregated over ALL frames
  (trimmed median, never a frame-0 sample) **cross-validated against an SfM/SLAM
  self-calibration**; accept only on few-% agreement, else flag the clip.
- Enforce fx=fy unless the capture chain is proven anamorphic.
- If distortion is significant (T-dist), undistort at ingest; pinhole-only downstream.
- Feed the same K to: UniDepth (known-K mode — also improves its depth), the SLAM/camera
  trajectory, WiLoR/HaWoR translation refits, and the renderer. Per-frame intrinsics fields
  become diagnostics only and must not parameterize any transform.
**Acceptance evidence:** one K per clip recorded in the output schema with provenance;
T1-style refit residual drop demonstrated on tomato; UniDepth-vs-SfM K agreement logged
per clip; zero code paths reading per-frame intrinsics for projection (grep-clean +
runtime assert).

### FIX-2 — One rigid metric space; kill the per-frame elastic depth scale
`hawor_to_v18_depth_scale` (per-frame ratio, 0.28–1.30 range, ±38% frame jumps) must die.
With FIX-1 in place its main cause (focal mismatch) disappears; the residual
depth-model-vs-hand-stream scale becomes a **per-clip constant** (or an explicit
slowly-varying spline with disclosed dof), fitted jointly over the clip with outlier
rejection — never a per-frame ratio. Camera trajectory must be re-run under the calibrated
K (DPVO/DROID with known K) so world scale, hand depth, and camera motion live in one
rigid metric space.
**Acceptance evidence:** scale field is constant-per-clip in the schema; size-consistency
metric (rendered vs detected hand bbox height, metric #2) within ±10% at p90 over the clip;
fallback-row projections within the same residual budget as detector rows.

### FIX-3 — Detector-fidelity-bounded temporal fusion
The fusion objective is currently allowed to trade detector agreement for smoothness
without bound below 50px. Change the contract:
- Hard residual bound: whenever a validated detection exists, |fused−raw| ≤ ~10–15px
  (constraint or escalation on ANY sustained divergence, not only >50px/≥3f).
- Robustify/cap the relative-motion prior (Track V lesson: bridge deltas can contain
  m/frame discontinuities) and weight it by its own scale-status; elastic-scale fallback
  rows get ~zero prior weight and must not pull the solution.
- Fallback/infill frames are occlusion/uncertainty states: rendered as reduced-alpha ghosts
  positioned only when temporally interpolatable, never as confident hands 100px off.
**Acceptance evidence:** re-measure the exact metric of §0 on the rebuilt layer — target
median ≤8px, p95 ≤20px vs an **independent** detector (rtmlib), and zero >20px runs longer
than ~0.5s on delivery clips; visual scrub of f147–f321 and f926–f946 shows the hand
annotation locked to the visible hand or explicitly ghosted.

### FIX-4 — Separate head/camera and hand metrics (stakeholder's explicit ask)
Nothing in the current eval isolates camera error; the Track M/V camera-frame drift
(R(t) 3.4–3.8°, smooth b(t)) currently lives inside the "hand" number.
- **Camera/head metric:** on GT benchmarks (HOT3D device poses; Aria MPS), report camera
  ATE and RPE (position mm + rotation deg over declared horizons, e.g., 100ms/1s) under an
  explicitly declared gauge (Track B lesson: naive world-Umeyama is gauge-invalid; the
  convention must be part of the metric definition).
- **Hand metric:** wrist/MPJPE **in the GT camera frame** (camera error excluded by
  construction) — this is what the HOT3D eval already does; keep it as the hand number.
- **Expectation setting for the ~5mm ask:** 5mm hand-in-camera is at the measured frontier
  (per-side localization floor after camera-drift removal was the terminal HOT3D residual);
  5mm camera is deliverable as short-horizon RPE, not absolute monocular world position.
  Say this in the API spec, with the decomposition evidence.
- GT-free per-clip proxies (production QC): rtmlib cross-detector residual (lateral),
  size-consistency ratio (depth), K-stability, camera-above-hands plausibility.
**Acceptance evidence:** benchmark report with two separated tables (camera ATE/RPE; hand
camera-frame MPJPE) on HOT3D + at least one Aria-type clip; discriminating experiment: under
FIX-1/FIX-2 the fitted per-clip R(t)/b(t) camera-drift terms should shrink — if they do not,
the residual is SLAM registration and the camera metric will show it directly.

### FIX-5 — QC/confidence must measure the shipped layer, not an upstream stage
The renderer's alpha/chips consume `*_wilor_fit_reprojection_median_px` — the **pre-fusion
fit residual** — so drifted fused frames still display "visible/high". That is why the demo
QC did not flag what the stakeholder saw.
- Recompute reprojection residual against the FINAL fused vertices (the §0 computation) and
  drive alpha, confidence tiers, and chips from that.
- Add the independent cross-detector residual (rtmlib) per frame to the per-clip QC report.
**Acceptance evidence:** on the current tomato layer this recomputed QC must flag the
f147–f321 left block — that is the regression test proving the QC now sees what users see.

### FIX-6 — Deterministic renderer, no silent fallbacks, single provenance authority
- Render = pure function of (state hash, layer hashes); `--resume` off in the product path
  or keyed by content hash so stale frame mixing is impossible.
- Delete the silent intrinsics default (`[500,500,480,270]`) — missing intrinsics is a hard
  error, not a guess.
- Current-layer provenance is the only visibility/validity authority (the stale-validity
  supersede bug fired twice; make it structural: superseded fields are not readable by the
  render path).
**Acceptance evidence:** double render → frame-hash equality; a state with missing
intrinsics fails loudly; grep/assert shows no reader of superseded validity fields.

Order of execution: FIX-1 → FIX-2 → FIX-3 (they nest causally); FIX-5/FIX-6 are small and
parallel; FIX-4 defines the acceptance frame and should be specified first, implemented on
the benchmark as soon as FIX-1/2 land.

---

## 3. What NOT to touch

- **No HOI/factor-graph/contact/occlusion-ownership work** in the delivery track — excluded
  by the stakeholder; nothing above needs it.
- **Do not build more per-clip calibration families** (R(t)/s(t)/g(z)/b(t) knot ladders).
  They are disclosed demo/oracle devices; the product-side answer to the same error class is
  the calibrated camera + rigid metric space + (research-track) GT-free self-calibration.
- **Do not replace WiLoR/HaWoR.** The error budget is camera-model/registration + fusion
  policy, not the hand networks: in the correct-K regime the 2D refit reaches 4–7px, and
  HaWoR relative motion is a usable prior once capped. Swapping estimators attacks the
  wrong variable.
- **Do not re-select hand base streams** for wrist metrics (hybrid vs HaWoR wrist vectors
  are identical — measured inert).
- **Do not chase frame/time alignment** (H4 ruled out, k=0 optimal).
- **Do not attempt rotation-only world de-drifting** (72% measured ceiling; drift-bent
  worlds are an upstream SLAM problem, addressed via FIX-1/FIX-2 rerun).
- **Do not touch the v16 MANO/mesh/depth render backbone or the mask-based image-space
  object claims** — they are correct mechanisms per monotonicity; only the paths named in
  FIX-5/FIX-6 change.

---

## 4. Validation evidence required for delivery sign-off (per clip, automated)

1. **Camera model record:** single K (+distortion flag) with provenance and
   UniDepth-vs-SfM agreement number.
2. **Overlay agreement vs independent detector (rtmlib):** median ≤8px, p95 ≤20px, no
   >20px run longer than ~0.5s; per-frame series shipped in the QC report.
3. **Size consistency:** rendered vs detected hand bbox height ratio within ±10% p90
   (closes the 2D-consistent-but-3D-wrong blind spot).
4. **Metric rigidity:** per-clip constant depth-scale field; no per-frame projective
   parameter anywhere in the transform chain.
5. **Separated benchmark metrics:** camera ATE/RPE table + hand camera-frame MPJPE table
   (HOT3D + one Aria-type clip), gauge convention declared.
6. **Renderer determinism:** double-render hash equality; QC chips derived from final-layer
   residuals (tomato f147–f321 must flag on the old layer).
7. **Human scrub** of the previously-failing segments (tomato f147–f321, f926–f946;
   trash bands) — the annotation visibly locks to the hand or explicitly ghosts.

## Provenance of the measurements in this memo
- Shipped-layer reprojection audit, shift test, depth-scale distribution: computed
  2026-07-05 from `demo_pack_20260704/tomato_hands_y/hybrid/wilor_temporal_fused_hybrid_hands.npz`,
  `tomato_hands_y/wilor_full_frame_960/wilor_raw.json`,
  `/data2/ego_annotation_outputs/v18_corrective_1600/hawor_bridge_state/task5_tomato_960/hawor_bridge_candidates_current_v18_camera_local.npz`,
  `/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/task5_tomato_960/annotations_v18_full.json`.
- Intrinsics provenance: `scripts/run_v18_full_pipeline.py` (depth-NPZ per-frame intrinsics
  → `current_v18_camera_intrinsics_fx_fy_cx_cy`; HaWoR `source_intrinsics
  [2304,2304,960,540]`), UniDepth source
  `/data2/ego_annotation_outputs/v18_unidepth_extension/complete_depth_root/task5_tomato_960/unidepth_metric/unidepth_metric_depth_v3.npz`.
- Renderer paths: `demo_pack_20260704/renderer/render_demo_presentation.py` (frame-0
  intrinsics, `[500,500,480,270]` fallback, fit-stage residual consumption).
- Prior mechanism evidence: `.memory/project/hand_metric_mechanisms.md`,
  `object_geometry_and_render_lessons.md`, `self_consistency_metrics.md`,
  `demo_pack_20260704/review/track{K,Y}_*.md`, demo final state
  `.memory/tasks/2026-07-04-demo-pack/EPISTEMIC.md`.
