# 26 — clip001850 keyboard pose-regime classification (KT-L1, pose-independent)

Branch `yiwen_research`. One new additive script, nothing staged, nothing committed.
Script: `scripts/analyze_clip001850_pose_regime_kt_l1.py`
Outputs: `/tmp/clip001850_pose_regime_kt_l1/{summary.json, per_frame.ndjson, review.png}`
CPU-only (numpy / scipy.pdist / cv2 / matplotlib). No model inference, no GPU, no smoothing,
no pose-row rewrite, no contact claim.

---

## 0. Causal card

- **Artifact defect.** The keyboard rigid-object pose graph is inert (`nfev=1, cost=0`)
  and unlocalizes 137/150 frames. The 13 direct ICP fits include a 4-frame outlier group
  (f60/75/76/77) up to 200 mm from the rest cluster, and the renderer freezes f78–149 at
  the f77 outlier pose ~180 mm off the keyboard's true location (subagents 22–25). Before
  any estimator is chosen, the per-frame physical regime must be measured: is each frame a
  static observation, a real moving observation, a mask-drift artifact, occluded/out-of-view,
  or unresolved? Smoothing or a static median applied without this adjudication fabricates
  motion or localization (FM1/FM2 in subagent 25).
- **Physical variable.** `T_world_object(t)` and its per-frame observability/regime —
  decided from **pose-independent observed-surface evidence only** (depth centroids, mask
  extents, image-space reprojection), not from the pose-fit rows.
- **Live mechanisms.** M1 no-seed (object not visible); M2 mask-drift (SAM2 mask leaked
  onto hand/background); M3 real object motion; M4 static-held gauge (the steelman).
- **Discriminating measurement (this artifact).** Pose-independent observed centroid
  trajectory; rest-cluster consensus; per-frame centroid displacement vs rest; mask extent
  diag + worst-axis ratios; image-space rest-footprint-vs-mask overlap; hand-coverage of
  the keyboard mask; reprojection/crop review.
- **Predictions.** Static → tight centroid cluster, extent-consistent, rest-footprint
  projects into mask. Moving → contiguous monotonic displaced run. Mask-drift → centroid
  displaced and/or extent inflated, rest-footprint does not track the mask. Unobserved → no
  metric surface.
- **Intervention selected by outcome.** Route decision (static-held gauge / supported
  moving trajectory / support recovery / unresolved) — stated at the end, not implemented
  here (this is KT-L1 classification only).

## 1. Headline result

**Route verdict: static-held gauge.** The keyboard is observed as a static rest pose.
There is **no moving-observed frame** and **no contiguous moving run**. The 4 displaced
"eligible" direct-fit frames (f60/75/76/77) — including the visible-geometry **anchor frame
f75 itself** — are mask-drift, not motion.

| regime | count | frames |
|---|--:|---|
| static_observed | 8 | 30,31,32,33,34,35,36,46 |
| unresolved | 1 | 45 (borderline; hand-occluded, 68 mm depth offset) |
| mask_drift | 76 | 60, 75, 76, 77, 78–149 |
| occluded_out_of_view | 65 | 0–29, 37–44, 47–59, 61–74 (no metric surface) |
| moving_observed | 0 | — |

Rest cluster: centroid `[0.599, −0.094, 0.339] m`, internal spread **21.6 mm**, max internal
distance **28.5 mm**. Static radius (data-derived) = **55 mm**. There is a clean gap in
observed-centroid displacement between the static cluster (8–28 mm) and everything else
(≥65 mm).

## 2. The critical new finding — the anchor frame is itself a mask-drift outlier

`v19_visible_geometry_adapter_report.json` records `anchor_frame_idx = 75`. The anchor
centroid is `[0.456, −0.181, 0.230] m`, which is **199.5 mm from the rest-cluster centroid**
`[0.599, −0.094, 0.339] m`. The frame chosen as the canonical keyboard anchor (whose extent
defines the eligibility gate for every other frame, and whose visible surface seeds the ICP
canonical mesh) is itself a mask-drift frame:

| frame | disp vs rest | diag ratio | worst-axis | rest_in_mask | hand_cov | regime | anchor? |
|---|--:|--:|--:|--:|--:|---|:--:|
| 60 | 70 mm | 1.15 | 2.37 | 0.306 | 0.13 | mask_drift | |
| **75** | **200 mm** | 1.00 | 1.00 | 0.301 | 0.20 | **mask_drift** | **YES** |
| 76 | 189 mm | 1.07 | 3.13 | 0.610 | 0.02 | mask_drift | |
| 77 | 202 mm | 1.06 | 3.01 | 0.615 | 0.02 | mask_drift | |

f75 has `extent_ratio = [1.0, 1.0, 1.0]` **trivially** (it IS the anchor, so it matches
itself). Its observed surface projects to image region `(812, 600)` while the static
keyboard rest location projects to `(744, 523)` — 106 px apart. This means:

1. The upstream `rigid_pose_observation_eligible` gate compares every frame's extent to a
   **drifted** anchor. The 13 "eligible" frames are those whose extent happens to match
   f75's (leaked) extent — not a clean keyboard filter.
2. The ICP canonical mesh and pose seed derive from a frame whose observed surface is not at
   the keyboard rest location.
3. **Any pose re-fit must re-anchor to the rest cluster (f30–36,46) before it is trustworthy.**
   This is why this script determines the rest cluster from observed-centroid coherence
   (median+MAD over extent-plausible frames), NOT from the eligibility gate.

## 3. Measurements produced (all pose-independent)

**(1) Observed centroid trajectory.** Per-frame `centroid_world_m` from the visible-geometry
candidate (depth backprojection, no pose fit). Static cluster f30–36,46: 8–28 mm spread.
All other visible frames: 65–232 mm from rest.

**(2) Rest-cluster consensus.** Candidate pool = frames with
`extent_ratio_to_anchor_diag ∈ [0.70, 1.20]` (13 frames). Iterative componentwise median+MAD
trim (3.5σ) → core `[30,31,32,33,34,35,36,46]`, centroid `[0.599,−0.094,0.339]`, spread
21.6 mm. Rejected extent-plausible candidates (centroid-displaced): `[45, 60, 75, 76, 77]`.

**(3) f60/75/76/77 observed-surface displacement vs rest.** Table in §2: 70 / 200 / 189 /
202 mm. All far beyond the 55 mm static radius and beyond any depth-noise floor.

**(4) Mask extent ratios (with definitions, recorded in summary.definitions).**
- `extent_ratio_to_anchor_diag` = `‖world_extent_m‖₂ / ‖anchor_extent_world_m‖₂`.
  Static cluster 0.83–0.95; leaked f78+ ≥ 1.34; borderline f45/60/75/76/77 1.0–1.15.
- `worst_axis_ratio` = `max_axis |world_extent_m[axis]/anchor_extent_world_m[axis]|`.
  Static cluster ≤ 2.06; f78+ ≥ 4.1; f76/77 = 3.0–3.13 (inflated one axis).

**(5) Image-space consistency (rest-footprint reprojection).** The rest-cluster world
vertices are pooled and projected into each frame's camera; `rest_in_mask_fraction` =
fraction landing inside that frame's SAM2 mask. Static cluster: 0.71–0.83 (the keyboard
projects onto its own mask). Drifted frames: 0.20–0.75 — note this does **not** cleanly
separate inflated leaked masks from static, because huge leaked masks often still cover the
keyboard location while also covering hand/background. It is used as a tie-breaker, not the
primary discriminator. `own_in_mask_fraction` (own observed verts → own mask) is ~0.998 on
every frame — a tautology that confirms the projection pipeline is correct.

**(6) Hand-coverage.** `keyboard_mask_in_hand_bbox_frac` = fraction of the keyboard mask
inside any hand bbox. f45 = 0.626 (hand occluding the keyboard → borderline), f30–35 rises
0.24→0.88 (hand approaches during the typing window), f76/77 ≈ 0.02 (leakage is onto
background/desk, not hand).

**(7) Reprojection/crop review.** `review.png` is a 15-panel contact sheet (f0,28,30,33,36,
45,46,60,75,76,77,78,90,120,149) showing RGB + SAM2 mask (red) + rest-cluster footprint
projection (lime) + own observed points (yellow) + hand bbox (cyan), each labelled with
regime, displacement, rest_in_mask, worst_axis. The static frames show lime+yellow+mask
coincident on the keyboard; the drifted frames show the lime rest-keyboard location
displaced from the red mask.

## 4. Why moving_observed is empty (the steelman, tested and rejected)

A real object move requires a **contiguous monotonic trajectory**. The moving-run detector
requires ≥3 contiguous frames, each extent-plausible (diag < 1.30), each displaced beyond
the static radius, with monotonic displacement and worst-axis no worse than the clean
cluster bound (2.06). No such run exists: f45 and f60 are isolated; f75–77 are contiguous
but f76/77 have worst-axis 3.0+ (inflated) and the displacement sequence 200→189→202 is
non-monotonic; f78+ are all extent-inflated. Combined with the rest cluster's 21.6 mm
spread over 9 frames, the keyboard is demonstrably static. A 200 mm "move" of a desk
keyboard mid-clip that returns to rest is not physically supported by any contiguous
observed-surface evidence.

## 5. Route decision (clear route, per task requirement)

**Static-held gauge.** Concrete next-step contract (consistent with subagent 25 §4):
1. Estimate a robust `T_rest` over the rest cluster f30–36,46 (median+MAD centroid
   `[0.599,−0.094,0.339]`), with per-DOF covariance from the cluster spread (KT-L4).
2. **Re-anchor the visible-geometry / ICP canonical mesh to the rest cluster** — the current
   f75 anchor is a mask-drift outlier and contaminates the eligibility gate and pose seeds.
3. Reject f60/75/76/77 as `rejected_degenerate_fit` / `mask_drift` (KT-L5); exclude from any
   rest-pose or trajectory estimate.
4. Mark the 65 no-surface frames `occluded`/`out_of_view` and the 72 extent-leaked frames
   `mask_unreliable` in a visibility ledger — never as `measured`/`localized` (guards FM2).
5. KT-L2 (support recovery on the 72 leaked frames via mask cleaning) is the highest-value
   follow-up: they carry full visible surface currently discarded for leakage; recovering
   any subset raises localization above 8/150. Must be validated by re-anchored refits, not
   asserted.

The **support-recovery** and **unresolved** routes are not the primary verdict: support
recovery is a follow-up (KT-L2), and only f45 is unresolved (borderline, hand-occluded).

## 6. Scope discipline

- **One new additive script** (`scripts/analyze_clip001850_pose_regime_kt_l1.py`). Zero
  edits to existing scripts.
- **No smoothing, no pose-row rewrite, no contact claim** (`scope` block in summary.json:
  `smoothing_applied=false, pose_rows_rewritten=false, contact_claimed=false`).
- **No model inference / GPU** — imports are json/os/re/pathlib/cv2/numpy/scipy.pdist/
  matplotlib only.
- **Nothing staged or committed** — script is untracked (`??`); `git diff --cached` empty.
- Outputs under `/tmp/clip001850_pose_regime_kt_l1/` (not the repo).

## 7. Honest limits / residual risks

- **rest_in_mask_fraction is not a universal discriminator.** Inflated leaked masks (f78+)
  often still cover the keyboard location, so their rest_in_mask reaches 0.5–0.75,
  overlapping the static cluster (0.71–0.83). The primary discriminator on this clip is
  observed-centroid displacement (clean 8–28 mm vs ≥65 mm gap) + diag extent ratio. The
  script reports rest_in_mask honestly and uses it only as a borderline tie-breaker (f45).
- **f45 (unresolved) is genuinely ambiguous.** 68 mm depth offset, rest_in_mask 0.466,
  hand_cov 0.626 — consistent with either a hand-occluded static observation or mild mask
  shift. Classified unresolved rather than forced into static or drift.
- **The 65 no-surface frames are labelled `occluded_out_of_view` without distinguishing
  occlusion from out-of-view** — pose-independently only the absence of metric surface is
  known; the cause requires the lost mask track.
- **KT-L1 does not itself rewrite pose rows or render.** Per subagent 25 §5, the
  artifact-changing intervention (static `T_rest` + visibility ledger + re-render so f78–149
  stop drawing ~180 mm off) is the next step that this classification licenses.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
/home/yiwen/ego_annotation/.venv/bin/python scripts/analyze_clip001850_pose_regime_kt_l1.py
# -> /tmp/clip001850_pose_regime_kt_l1/{summary.json, per_frame.ndjson, review.png}
```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Implemented KT-L1 as one new additive script scripts/analyze_clip001850_pose_regime_kt_l1.py (untracked, not staged). Consumes only existing CPU files: visible_geometry annotations + adapter report (per-frame centroid_world_m, world_vertices_sample_m, world_extent_m, extent_ratio_to_anchor_diag/axis, rigid_pose_observation_eligible, mask_path, camera T_world_camera_metric + intrinsics, hands bbox), the pose-graph report (graph_frames + optimizer for cross-ref), raw RGB, and SAM2 mask PNGs. Produces all 7 required measurements: (1) pose-independent observed centroid trajectory; (2) rest-cluster consensus via median+MAD over extent-plausible frames (NOT the eligibility gate); (3) f60/75/76/77 displacement vs rest (70/200/189/202 mm); (4) mask extent diag + worst-axis ratios with explicit definitions in summary.definitions; (5) rest-footprint reprojection overlap (rest_in_mask_fraction) + own_in_mask_fraction sanity; (6) keyboard-mask hand-coverage; (7) 15-panel reprojection/crop review.png. Per-frame regime labels with evidence basis for all 150 frames. No smoothing, no pose-row rewrite, no contact claim, no model inference, no GPU."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Every load-bearing claim verified against on-disk state and reproducible: rest cluster f30-36,46 centroid [0.599,-0.094,0.339] spread 21.6mm (clean gap to >=65mm for all other frames); anchor_frame_idx=75 read from adapter report, anchor centroid [0.456,-0.181,0.230] = 199.5mm from rest (the anchor is a mask-drift outlier); f60/75/76/77 all classified mask_drift with displacement/extent/rest_in_mask evidence; regime_counts {static_observed:8, unresolved:1 (f45), mask_drift:76, occluded_out_of_view:65, moving_observed:0} sum=150; moving-run detector finds no contiguous monotonic extent-plausible run (steelman tested+rejected); own_in_mask_fraction ~0.998 on all frames confirms projection pipeline; image-space rest-centroid projection vs mask-bbox-center delta confirms drift (f30 21px, f60 89px, f75 106px, f76 84px); summary.scope {smoothing_applied:false, pose_rows_rewritten:false, contact_claimed:false, model_inference:false, gpu_used:false}; imports limited to json/os/re/pathlib/cv2/numpy/scipy.pdist/matplotlib; git diff --cached empty, script untracked. Route verdict = static-held gauge with re-anchor-to-rest-cluster requirement stated."
    }
  ],
  "changedFiles": [
    "scripts/analyze_clip001850_pose_regime_kt_l1.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "/home/yiwen/ego_annotation/.venv/bin/python scripts/analyze_clip001850_pose_regime_kt_l1.py",
      "result": "passed",
      "summary": "Wrote /tmp/clip001850_pose_regime_kt_l1/{summary.json 6KB, per_frame.ndjson 84KB (150 rows), review.png 4.6MB}. regime_counts static=8/unresolved=1/mask_drift=76/occluded=65/moving=0. rest cluster f30-36,46 spread 21.6mm. anchor f75 = mask_drift (199.5mm from rest). Route = static-held gauge."
    },
    {
      "command": "python3 reprojection sanity (own verts -> own mask)",
      "result": "passed",
      "summary": "Verified projection pipeline with scale=960/1408: f30 own_in_mask_fraction=0.998, projected range [614-848,183-633] matches manifest bbox [612-851,132-633]."
    },
    {
      "command": "python3 inspect adapter report for anchor_frame_idx",
      "result": "passed",
      "summary": "anchor_frame_idx=75; anchor_centroid=[0.456,-0.181,0.230]; anchor_extent=[0.2526,0.7023,0.1792]. Confirms the eligibility gate is anchored to a displaced frame."
    },
    {
      "command": "python3 per-frame table inspection (regime/disp/diag/worst/rest_in_mask/hand_cov)",
      "result": "passed",
      "summary": "Full 85-visible-frame table confirms clean displacement gap (static 8-28mm vs >=65mm); f78+ all diag>=1.34; f76/77 worst_axis 3.0+; f45 borderline (68mm, rest_in_mask 0.466, hand_cov 0.626)."
    },
    {
      "command": "git status --short + git diff --cached --name-only + import grep",
      "result": "passed",
      "summary": "Script untracked (??); nothing staged (git diff --cached empty); imports json/os/re/pathlib/cv2/numpy/scipy/matplotlib only (no torch/cuda/model)."
    }
  ],
  "validationOutput": [
    "Route verdict = static-held gauge: keyboard observed static (rest cluster f30-36,46, 21.6mm spread); no moving-observed frame and no contiguous moving run; f60/75/76/77 (incl. the anchor) are mask_drift.",
    "Critical finding: visible-geometry anchor_frame_idx=75 is itself a mask-drift outlier (centroid 199.5mm from rest cluster), contaminating the upstream eligibility gate and ICP canonical seed; any pose re-fit must re-anchor to the rest cluster first.",
    "Required measurements all produced and auditable in summary.json/per_frame.ndjson: centroid trajectory, rest-cluster consensus, f60/75/76/77 displacement (70/200/189/202mm), extent diag+worst-axis ratios with definitions, rest-footprint reprojection overlap, hand-coverage, 15-panel review.png.",
    "Decision tree is data-grounded: static_radius=55mm from rest-cluster internal extent; MASK_LEAK_DIAG=1.30 is the clean gap in the diag distribution; worst_axis_clean = rest cluster's own max (2.06); moving_observed requires a contiguous monotonic run of >=3 extent-plausible frames. No tuned suppression heuristics.",
    "scope flags all false: no smoothing, no pose-row rewrite, no contact claim, no model inference, no GPU.",
    "Counts sum to 150 (8+1+76+65); moving_observed absent (=0)."
  ],
  "residualRisks": [
    "rest_in_mask_fraction does not cleanly separate inflated leaked masks from static (huge leaked masks often still cover the keyboard location, rest_in_mask 0.5-0.75); it is used only as a borderline tie-breaker (f45). The primary discriminator is centroid displacement + diag extent ratio.",
    "f45 (unresolved) is genuinely ambiguous: 68mm depth offset with hand_cov 0.626 (hand occluding) — could be static-with-occlusion or mild mask shift; labelled unresolved rather than forced.",
    "The 65 no-surface frames are labelled occluded_out_of_view without distinguishing occlusion from out-of-view (pose-independently only the absence of metric surface is known).",
    "KT-L1 classifies only; it does not rewrite pose rows or re-render. The artifact-changing intervention (static T_rest + visibility ledger + re-anchor + re-render so f78-149 stop drawing ~180mm off) is the next step this classification licenses, per subagent 25 section 5.",
    "KT-L2 (support recovery on the 72 extent-leaked frames via mask cleaning) is unmeasured; if those frames are unrecoverable they are honest occlusion holes and static-held gauge is the ceiling."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new additive analysis script scripts/analyze_clip001850_pose_regime_kt_l1.py (CPU-only pose-independent regime classifier). No existing files touched; outputs under /tmp; nothing staged or committed.",
  "reviewFindings": [
    "no blockers",
    "finding (high value): the visible-geometry anchor frame (f75) is itself a mask-drift outlier 199.5mm from the rest cluster; the upstream rigid_pose_observation_eligible gate and ICP canonical seed are contaminated by it. Any pose re-fit must re-anchor to the rest cluster (f30-36,46).",
    "finding (route): KT-L1 verdict is static-held gauge, not smoothing and not support-recovery-as-primary; moving_observed is empty because no contiguous monotonic displaced run exists (the steelman is tested and rejected)."
  ],
  "manualNotes": "Core result for the parent: clip001850 keyboard is static; route = static-held gauge over the rest cluster f30-36,46 (centroid [0.599,-0.094,0.339], 21.6mm spread). The single most important new discovery is that the visible-geometry anchor_frame_idx=75 is itself a mask-drift frame (199.5mm off rest) — so the eligibility gate that produced the 13 'eligible' fits and the ICP canonical mesh is anchored to a drifted frame and must be re-anchored before any pose refit is trusted. f60/75/76/77 are all mask_drift; f45 is the one unresolved (hand-occluded borderline); 65 frames have no metric surface; 72 frames (f78-149) are extent-leaked mask_drift. KT-L2 mask-cleaning on those 72 is the highest-value follow-up. This classification licenses the static T_rest + visibility-ledger + re-render intervention of subagent 25 section 5; it does not itself rewrite pose rows."
}
