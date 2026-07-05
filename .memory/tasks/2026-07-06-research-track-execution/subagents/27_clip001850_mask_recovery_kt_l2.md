# 27 — clip001850 KT-L2 mask-recovery probe

Branch `yiwen_research`. One additive script only; nothing staged, nothing committed.
Script: `scripts/probe_clip001850_mask_recovery_kt_l2.py`
Outputs: `/tmp/clip001850_mask_recovery_kt_l2/{summary.json, per_frame.ndjson, review.png}`
CPU-only. No model inference, no GPU, no smoothing, no pose-row rewrite, no contact claim.

---

## 0. Causal card (the question)

**Artifact defect.** 72/150 frames (f78-149) carry a visible SAM2 keyboard mask
with 2500 metric backprojected points each, but are routed to
`visible_surface_ineligible_for_rigid_pose_fit` because one world extent axis is
inflated 4.4-6.2x vs the anchor (`systematic_mask_extent_inconsistent…probable_hand_background_leakage`).
Direct object-pose support is frozen at 13/150 frames (subagents 22-25).

**Physical variable.** Per-frame object-localization support: whether the surface
inside the leaking mask is *recoverable* keyboard surface (quarantine removes a
minority of hand/background pixels and leaves a keyboard patch at the rest
location) or *delaminated* (the mask has migrated off the keyboard onto
background/hand-adjacent regions, so quarantine cannot recover keyboard support).

**Live mechanisms.**
- M2a (recoverable): mask leaks a minority of hand/background pixels; after
  quarantine the surviving points sit on the keyboard rest location with extent
  consistent with the anchor and can seed a rigid fit.
- M2b (degenerate): after quarantine a thin/partial keyboard patch remains;
  extent is consistent but pose is only estimable up to a weak DOF.
- M-drift (delaminated): the mask has migrated OFF the keyboard; surviving
  points are background/hand-adjacent, quarantine cannot recover support.

**Discriminating measurement.** Staged quarantine — (1) MANO hand projected hull,
(2) keyboard camera-depth band derived from the f30-46 inlier frames, (3) rest
spatial box — then measure surviving point count, world extent, worst-axis ratio
vs anchor, centroid, and centroid-to-rest distance. A frame is recoverable ONLY
if the *unbiased* hand+depth-cleaned centroid lands within 60 mm of the keyboard
rest location AND the extent is consistent (worst-axis ≤2.5x). A trimmed
Procrustes fit of the cleaned sample to the f30-46 inlier rest body measures
whether the surviving patch actually aligns to the keyboard body.

**Prediction.**
- M2a → many frames recover extent AND land near rest; direct support rises
  above 13/150.
- M-drift → cleaned centroids stay 100-300 mm from rest regardless of
  quarantine aggressiveness; direct support cannot rise above 13/150.

## 1. Result — verdict: **mask_drift_delaminated (all 72)**

The SAM2 keyboard mask has fully delaminated off the keyboard by f78. After
hand-hull + keyboard-depth-band quarantine (the unbiased test — no rest box),
the cleaned centroid of every one of the 72 frames is **138-278 mm
(median 166 mm)** from the keyboard rest location. **Zero** frames land within
the 60 mm consensus neighbourhood; **zero** within 100 mm; only 2 within 150 mm.

```
hand+depth centroid dist to rest (mm): min=138  median=166  max=278
  n_within_60mm  = 0 / 72
  n_within_100mm = 0 / 72
  n_within_150mm = 2 / 72

verdict_counts:
  recoverable_consensus        = 0
  degenerate_partial_keyboard  = 0
  mask_drift_delaminated       = 72   (delamination_total = 72)

direct_support:
  baseline_eligible_frames       = 13
  recovered_consensus_frames     = 0
  direct_support_after_recovery  = 13
  direct_support_rises_above_13  = False
```

**Trimmed Procrustes fit** of each hand+depth-cleaned sample to the f30-46 rest
body: required translation **149-284 mm (median 175 mm)**, RMSD 149-284 mm.
The cleaned samples do not align to the keyboard body. This is consistent with
the centroid displacement: the surviving surface is background, not a keyboard
patch.

## 2. Why quarantine cannot recover support (the mechanism)

The leak is not a minority of stray pixels. On f78 the raw 2500 points span
camera depth 0.32-1.01 m; only 1633/2500 survive the keyboard depth band
(0.40-0.64 m), and those survivors have a world centroid at **[0.405, -0.152, 0.321]**
vs the keyboard rest centroid **[0.596, -0.095, 0.345]** — a **190 mm**
displacement in the x-y plane with the z roughly right. The pattern is uniform
across f78-149.

The rest-box quarantine (keep only points inside the keyboard's static world
extent) drops most frames below the 200-point minimum because the surviving
depth-band points are background that falls outside the keyboard's true spatial
location. For the frames that retain ≥200 box-survivors, the box-cleaned
centroid is still 100-150 mm from rest. Quarantine removes contamination but
leaves background that is spatially displaced from the keyboard — there is no
keyboard surface left inside the mask to recover. The mask has tracked onto the
hand/forearm/desk region that sits in the same depth band as the keyboard but
~166 mm away in the world.

## 3. Rest-model validity check (the negative control)

Running the *same* quarantine on the 9 eligible inlier frames (f30-46, where the
mask is on the keyboard) retains ~2000-2400 points each with centroid-to-rest
distance **6.9-54.7 mm** (median ~15 mm):

```
f30: 2414 pts,  14.9 mm
f31: 2408 pts,  27.6 mm
f32: 2417 pts,  18.6 mm
f33: 2345 pts,  14.6 mm
f34: 2380 pts,   7.4 mm
f35: 2409 pts,   6.9 mm
f36: 2376 pts,   8.8 mm
f45: 1987 pts,  54.7 mm
f46: 2145 pts,  21.3 mm
```

This is the discriminating evidence: when the mask is actually on the keyboard,
the cleaned centroid lands within consensus (≤55 mm); when it has delaminated
(f78-149), it lands 138-278 mm off. The quarantine/rest-model is not the cause
of the displacement — it resolves it correctly in both regimes.

## 4. Required conclusions (per task)

| question | answer |
|---|---|
| How many of the 72 frames become eligible support? | **0** |
| How many remain mask_drift / unrecoverable? | **72** (all) |
| Is any recovered fit consensus-consistent? | **No** — 0/72 land within the 60 mm consensus neighbourhood; Procrustes translation 149-284 mm |
| Can direct support truthfully rise above 13/150? | **No** — direct support stays at 13/150 via this route |

## 5. What this implies for the next intervention

The delamination is complete and begins around f60 (consistent with subagent 25
F5: observed centroid drifts ~180 mm pose-independently at f75-77, and the
eligible fits f60/75/76/77 are the mask-drift outliers). Quarantine cannot
recover the 72 frames because the mask is not leaking — it has migrated. The
honest artifact state for f78-149 is `mask_unreliable` / `occluded` (per
subagent 25's static-held-gauge route with a visibility ledger), not
`measurable` or `localized`. Recovering these frames would require re-running a
keyboard segmenter/tracker (a new perception pass, out of scope here and
forbidden by the no-new-model constraint), not a geometric quarantine.

This closes the KT-L2 branch negatively: the 72 frames are a hard perception
hole on this run, and the localization ceiling for clip001850 via the existing
SAM2 track is the ~13 clean frames (of which only the f30-46 cluster of 9 is
the genuine static rest consensus; f60/75/76/77 are drift outliers to reject
per KT-L5). The correct artifact-changing route is the static-held gauge with
covariance + visibility ledger described in subagent 25 §4 — *not* smoothing
and *not* mask quarantine.

## 6. Scope discipline

- One new additive script: `scripts/probe_clip001850_mask_recovery_kt_l2.py`.
  Zero edits to existing scripts.
- No staging, no commits. `git status --short` shows the script as `??`
  (untracked); `git diff --cached --name-only` is empty.
- No heavy inference / GPU / model: CPU-only numpy/scipy/matplotlib. Consumes
  the visible-geometry annotations (world points, camera, intrinsics), the MANO
  bridge npz, and computes extents/centroids/Procrustes fits.
- No smoothing, no pose-row rewrite, no contact claim
  (`pose_rows_rewritten=False`, `contact_claimed=False`).
- Outputs to `/tmp/clip001850_mask_recovery_kt_l2/` only.

## 7. Residual risks

- The keyboard-depth band and rest box are derived from the f30-46 inlier
  frames; if the keyboard genuinely moved after f46 (refuted by subagent 25 F2
  and the motion-coupling negative of subagent 19, but GT-free), the rest box
  could be mis-placed. The negative control (§3) shows the model resolves
  correctly where the mask is on the keyboard, so the displacement on f78-149
  is not a box artifact.
- The MANO hand hull uses the HaWoR bridge vertices (fresh, not interval-corrected);
  if the hand state is drifted on f78-149 the hull is approximate. The hand hull
  removes only ~1-45 points per frame in any case, so the result does not depend
  on hand-quarantine precision — the depth band and rest location do the work.
- This probe measures recoverability under quarantine; it does not run a new
  segmenter. Recovering the 72 frames would require re-segmentation, which is a
  perception intervention, not a geometry one.
- Procrustes uses a single nearest-neighbour pass with 80% trimming; a full
  ICP refinement could lower RMSD marginally but cannot close a 166 mm
  centroid gap — the gap is the signal, not the fit quality.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
.venv/bin/python scripts/probe_clip001850_mask_recovery_kt_l2.py
# -> /tmp/clip001850_mask_recovery_kt_l2/{summary.json, per_frame.ndjson (72 rows), review.png}
```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Implemented the requested KT-L2 mask-recovery probe as one additive script scripts/probe_clip001850_mask_recovery_kt_l2.py (untracked, not staged). Consumes only existing CPU files: visible_geometry annotations (per-frame world_vertices_sample_m, camera T_world_camera_metric, intrinsics, anchor_extent), MANO bridge npz (vertices_current_v18_world_from_hawor_projection_relift_m per frame/side). Implements conservative staged quarantine: (1) MANO projected hand-hull removal, (2) keyboard camera-depth band from f30-46 inliers, (3) rest spatial box. Reports per-frame surviving point counts at each stage, world extent and worst-axis ratio vs anchor before/after, centroid, centroid-to-rest distance (both box-quarantined and unbiased hand+depth), and a trimmed Procrustes/ICP fit of the cleaned sample to the f30-46 rest body. Outputs /tmp/clip001850_mask_recovery_kt_l2/{summary.json, per_frame.ndjson (72 rows), review.png}. No new model, no GPU, no smoothing, no pose-row rewrite, no contact claim. No edits to existing scripts; no scope widening."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Decisive, independently verifiable result. Negative control: applying the same quarantine to the 9 eligible inlier frames f30-46 retains 1987-2417 pts with centroid-to-rest 6.9-54.7 mm (rest model is valid). On the 72 ineligible frames f78-149 the unbiased hand+depth-cleaned centroid is 138-278 mm (median 166 mm) from the keyboard rest location; 0/72 within 60 mm, 0/72 within 100 mm, 2/72 within 150 mm. Trimmed Procrustes fit to the rest body requires 149-284 mm translation (median 175 mm) — cleaned samples do not align to the keyboard body. Verdict: all 72 frames = mask_drift_delaminated; 0 recoverable, 0 degenerate, 0 consensus-consistent; direct support 13 -> 13 (does not rise above 13/150). Conclusion answers all four required questions with measured evidence in summary.json and per_frame.ndjson."
    }
  ],
  "changedFiles": [
    "scripts/probe_clip001850_mask_recovery_kt_l2.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": ".venv/bin/python scripts/probe_clip001850_mask_recovery_kt_l2.py",
      "result": "passed",
      "summary": "Wrote /tmp/clip001850_mask_recovery_kt_l2/{summary.json, per_frame.ndjson (72 rows), review.png (1430x1210)}. verdict_counts: recoverable_consensus=0, degenerate=0, mask_drift_delaminated=72; hand+depth centroid dist min=138/median=166/max=278 mm; direct_support 13->13 (rises_above_13=False)."
    },
    {
      "command": "python3 inspect summary.json verdict_counts/centroid_displacement/direct_support + eligible_self_consistency + per_frame verdict distribution + procrustes translation stats",
      "result": "passed",
      "summary": "Confirmed: 0 recoverable, 72 delaminated; 0/72 within 100mm of rest; eligible self-consistency 6.9-54.7mm (valid rest model); procrustes translation 149-284mm (no alignment)."
    },
    {
      "command": "git status --short scripts/probe_clip001850_mask_recovery_kt_l2.py && git diff --cached --name-only && git branch --show-current",
      "result": "passed",
      "summary": "Script untracked (??); nothing staged (git diff --cached empty); on branch yiwen_research."
    }
  ],
  "validationOutput": [
    "KT-L2 verdict: all 72 visible_surface_ineligible frames (f78-149) are mask_drift_delaminated. 0 recoverable_consensus, 0 degenerate_partial_keyboard, 72 mask_drift.",
    "Decisive metric: after MANO hand-hull + keyboard-depth-band quarantine (no rest-box bias), cleaned centroid is 138-278 mm (median 166 mm) from keyboard rest [0.596,-0.095,0.345]. 0/72 within 60mm consensus, 0/72 within 100mm, 2/72 within 150mm.",
    "Negative control: same quarantine on eligible inliers f30-46 retains 1987-2417 pts with centroid-to-rest 6.9-54.7mm -> rest model valid, displacement on f78-149 is real delamination not a box artifact.",
    "Trimmed Procrustes fit of cleaned sample to f30-46 rest body: translation 149-284 mm (median 175), RMSD 149-284 mm -> surviving surface is background, not keyboard patch.",
    "direct_support: baseline 13 eligible -> 13 after recovery (recovered_consensus=0). direct_support_rises_above_13 = False.",
    "Mechanism: mask migrated off keyboard onto hand/forearm/desk region in same depth band but ~166mm away in world; quarantine removes contamination but leaves background, no keyboard surface to recover.",
    "No contact claimed (contact_claimed=false); no pose rows rewritten (pose_rows_rewritten=false); no model inference (model_inference_run=false); no GPU (gpu_used=false)."
  ],
  "residualRisks": [
    "Rest model (depth band + box) derived from f30-46 inliers; if keyboard genuinely moved after f46 the box could be misplaced, but subagent 25 F2 + subagent 19 motion-coupling negative establish static, and the negative control (6.9-54.7mm on inliers) shows the model resolves correctly where the mask is on the keyboard.",
    "MANO hand hull uses fresh HaWoR bridge (not interval-corrected); hand-quarantine removes only 1-45 pts/frame so the result does not depend on hand-hull precision.",
    "This probe measures recoverability under geometric quarantine; recovering the 72 frames would require re-segmentation (new perception pass, out of scope and forbidden by the no-new-model constraint).",
    "Procrustes uses single nearest-neighbour pass with 80% trimming; full ICP could lower RMSD marginally but cannot close a 166mm centroid gap."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one untracked additive probe script scripts/probe_clip001850_mask_recovery_kt_l2.py (~280 lines). No source/state/render/publish code edited. Outputs to /tmp (summary.json, per_frame.ndjson 72 rows, review.png). Nothing staged, nothing committed.",
  "reviewFindings": [
    "no blockers: the probe implements the requested conservative recovery (MANO hand quarantine, anchor/rest image-space + metric extent consistency, optional robust Procrustes/ICP fit to inlier rest body) and reports extent_diag/worst_axis before/after plus centroid displacement and consensus-consistency.",
    "no blockers: the conclusion answers all four required questions with measured evidence (0/72 recoverable, 72/72 delaminated, 0 consensus-consistent, direct support cannot rise above 13/150), backed by a negative control that validates the rest model.",
    "note: the result closes the KT-L2 branch negatively and routes the next intervention to the static-held gauge + visibility ledger (subagent 25 §4), not smoothing and not mask quarantine. Recovering the 72 frames needs a new perception pass."
  ],
  "manualNotes": "The 72 visible_surface_ineligible frames are a hard perception hole, not a recoverable-by-quarantine population. The SAM2 keyboard mask delaminated off the keyboard by f78: after hand+depth quarantine the cleaned centroid of every frame is 138-278mm (median 166mm) from where the keyboard actually rests. The negative control (eligible inliers f30-46 land 6.9-54.7mm from rest under the same quarantine) proves the displacement is real delamination, not a model artifact. direct support stays at 13/150; the localization ceiling for clip001850 via the existing SAM2 track is the f30-46 static cluster (9 genuine rest-consensus frames; f60/75/76/77 are drift outliers to reject per KT-L5). The correct next artifact-changing route is the static-held gauge with covariance + visibility ledger (f78-149 = mask_unreliable/occluded), which is the subject of a separate intervention, not this probe."
}
```
