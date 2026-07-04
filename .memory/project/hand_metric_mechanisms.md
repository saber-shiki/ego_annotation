# Hand metric-accuracy mechanisms — HOT3D evidence base (2026-07-04 sprint)

What was measured, what worked, what is ruled out, and why. Provenance: demo-pack Tracks
B/I/J/K/(M) reports under `demo_pack_20260704/review/` + full-precision rows in
`demo_pack_20260704/quant/ledger.csv`. Baselines: HaWoR wrist medians 20–37mm per clip/side
(camera frame, no trajectory alignment).

## The error structure (Track J oracle decomposition — do this FIRST on any new benchmark)

Decompose wrist error into along-prediction-ray and perpendicular components against GT.
On HOT3D clips 1849/1850/1851: perpendicular (lateral) floors are 8.9/10.9mm (1849 L/R) and
25–30mm (1850/1851) — **the binding constraint**; along-ray bias is range-dependent and
clip-specific. Clips differ in *dominant mechanism*, not parameters of one mechanism
(1849-right depth-dominated; 1850/1851 lateral-dominated). This explains every calibration
non-transfer. The decomposition reconciled exactly with the evaluator's published medians —
always verify that reconciliation before trusting derived numbers.

## Ruled out (with the falsifying evidence)

- **Per-side scalar ray scales**: plateau at 12.0mm on a disclosed 39-frame slice of 1849;
  frozen scalars do not transfer (1851-right worsens 32.8→40.1) — Track I.
- **Per-clip range-dependent depth curves** (even-fit/odd-eval): best odd-frame 11.4mm
  (1849-L); cannot cross the lateral floor — Track J.
- **Translation-only WiLoR 2D refit on HOT3D**: 23–26px median reprojection floor (vs
  7–10px in the trash-demo regime) → lateral fix does not transfer to close-range large
  hands; mechanism: weak-perspective/crop-convention mismatch grows with hand_size/depth —
  Track K (P3). At trash conditions (hands 130–180px, 0.5–0.7m) the same refit reaches
  7–10px and is the correct lateral anchor.
- **UniDepth at hand pixels for wrist depth**: 26–45mm abs-median — too noisy — Track J.
- **Head-motion triangulation on HOT3D**: inter-frame baselines 2–3mm → useless — Track J.
- **World-trajectory Umeyama/SE3 alignment as a metric**: gauge-invalid (SLAM world vs GT
  world incommensurable; 100–250mm residuals) — Track B.
- **Interval-solver 7.25mm row**: solver froze wrist translation (outputs bit-identical to
  input baseline). Lesson: verify solver liveness by diffing solved vs input state, never
  trust the objective value — Track B.

## What actually generalizes

- **WiLoR-visible geometry + HaWoR wrist (structural hybrid)**: visible-joint MPJPE
  47.2→33.0 (1850) and 46.7→36.9 (1851) held-out, left hands ≈−48%, zero per-clip
  parameters. Wrist translation unchanged by construction.
- Per-clip calibration (any family) is a *disclosed demo/cherry-pick device*, not a
  deployable stage, until a GT-free self-calibration source exists.

## Open mechanism (Track M, running at sprint end)

Constant per-clip rotation between prediction and GT camera frames (adapter/extrinsics
level): would unify the range-dependent "depth bias", the lateral floor, and calibration
non-transfer (2.4–4.0° ray-angle medians ≈ 15–25mm at 0.35m). If confirmed → fix belongs in
the HOT3D pinhole adapter / camera-convention mapping, not the hand estimator. Check
`review/trackM_rotation_calib_report.md`.

## Hybrid hand-layer construction (egoscale demo regime)

- WiLoR UV + HaWoR depth: fixes image placement; depth can be ~20–25% deep → rendered-size
  mismatch (caught by size-consistency metric).
- WiLoR crop-scale depth + reprojection gate (≤20px): right scale, but per-frame
  independent fits jitter; requires temporal fusion.
- **Temporal fusion recipe (Track L)**: source-switch hysteresis (≥5 frames) + banded
  least-squares translation with per-frame reprojection-weighted WiLoR observations +
  HaWoR *relative-motion* deltas as the smoothness prior (HaWoR frame-to-frame motion is
  reliable even where absolute position is biased) + quaternion smoothing for root
  orientation; never blend articulations across sources within a frame.

## Bench/infra caveats

- WiLoR needs `ultralytics` → use `hawor_work/.venv_hawor` python on the A800; bare
  `python3` there lacks numpy; `model_envs/unidepth_sam2` python for UniDepth/SAM2 work.
- `build_v19_visible_geometry_from_sam2_depth.py` materializes compressed depth NPZ in RAM
  (~4.8GB, 41min on 450f) — memmap fallback exists in the Track E run dir; upstream fix
  still pending.
- HOT3D WiLoR raws for clips 1849/1850/1851 exist under the A800 demo-pack
  `quant_tuning{,/trackI_heldout}` dirs — reuse, don't re-run.

## Resolution (Tracks M/P, end of sprint)

- **Constant per-clip camera-frame rotation is REAL** on 1850/1851 (3.40/3.81 deg, even/odd-stable,
  identical across prediction variants) — an adapter/extrinsics-level term; +R alone halves 1851.
  1849's mechanism is depth-scale instead (0.72 deg). Fitted via origin-anchored Procrustes on
  matched camera-frame wrist pairs; matched-pair NPZs dumped for reuse (trackM/pairs/).
- **Time-varying per-clip calibration (windowed R(t)+s(t)+g(z), even-fit/odd-report) reaches
  sub-10mm on 1849 both hands (8.78/8.90mm)**; 1850/1851 stop at 10.5–18.5mm with documented
  overfitting boundaries (odd-frame regressions per added knot). Demo-only device; the capacity
  ladder + knot sweep is the reusable METHODOLOGY: it measures how much error is slow drift vs
  per-frame noise, and where calibration capacity stops generalizing.
- Per-frame shared-rotation oracle still fails (13–30mm) → the terminal residual is per-side
  hand localization, not camera mapping. That is the pipeline's real frontier.
- Trash-clip hand layer: HaWoR bridge relative motion contained 7–11 m/frame discontinuities
  (the "smooth relative motion" premise is clip-dependent — always cap/robustify delta priors).
