# 23 — clip001850 keyboard pose-graph liveness probe

Branch `yiwen_research`. One additive script only; nothing staged, nothing committed.
Script: `scripts/analyze_clip001850_pose_graph_liveness.py`
Outputs: `/tmp/clip001850_pose_graph_liveness/{summary.json, per_frame.ndjson, pose_edges.ndjson}`

CPU-only. No smoothing, no repair, no heavy inference, no GPU.

---

## 0. Causal card

**Artifact defect.** The keyboard rigid-object pose graph is the measured root
blocker for clip001850 (subagent 21, Fact A): it is inert (`nfev=1`, `cost=0`,
zero residuals) and the keyboard is unlocalized on 137/150 frames. This gates
motion coupling, full-window contact, and render body placement.

**Physical variable.** `T_world_object(t)` (per-frame SE3) and which factor
families actually constrain it — i.e. whether the keyboard is localized in
metric world space through the clip.

**Live mechanisms (from subagent 21 card 1).**
- M1 no_seed_support — object not visible / no metric surface; no seed to propagate.
- M2 visible_surface_degenerate — visible pixels exist but the mask is leaking
  (hand/background), so the surface is ineligible for a rigid fit.
- M3 solver_plumbing_inert — even where seeds exist, the temporal graph did not run;
  variables not wired to a live objective.
- M4 legitimate_static_held_gauge — the keyboard truly barely moves and a held
  pose is the correct artifact (steelman; must be excluded, not assumed).

**Discriminating measurement.** Consume the pose-graph report, the upstream
visible-pose-fit observation rows, the visible-geometry adapter report, the raw
frame manifest, and the contact-state render manifest; measure pose status
counts, factor-family support, optimizer nfev/cost/residuals, direct-fit frame
list, frozen-pose clusters, held/interpolated segments, direct-edge jumps,
input-output deltas, and render/contact-state coupling. No smoothing.

**Predictions per mechanism.**
- M1 → large `missing_initial_graph_pose` fraction, zero visible surface there.
- M2 → visible metric surface but `rigid_pose_observation_eligible=False`, mask
  extent ratio inflated.
- M3 → optimizer inert (nfev≤1, cost 0), zero correction deltas, zero
  input-output delta (pass-through), direct fits unsmoothed.
- M4 → sharp, near-constant direct fits with a declared gauge.

## 1. Result — verdict: **mixed** (no_seed_support + visible_surface_degenerate + solver_plumbing_inert)

```
dominant_mechanism = mixed
present_mechanisms  = [no_seed_support, visible_surface_degenerate, solver_plumbing_inert]
```

### 1.1 Pose status counts (150 frames)

| pose_measurement_status | count | fraction |
|---|--:|--:|
| missing_initial_graph_pose | 65 | 0.43 |
| visible_surface_ineligible_for_rigid_pose_fit | 72 | 0.48 |
| fit_to_visible_depth_samples | 13 | 0.09 |

pose_source: `nearest_visible_pose_hold` 102, `interpolated_between_visible_pose_observations` 35,
`direct_visible_pose_observation_corrected` 13.

### 1.2 Active support by factor family

| family | active | evidence |
|---|---|---|
| temporal_smooth | **no** | correction translation_delta_step max = 0.0 m; rotation max = 0.0 rad |
| depth_icp_visible | yes | 13 fit frames, observed_to_mesh_final median-of-medians ≈ 13.8 mm (the only live measurement) |
| nonpenetration | **no** | nonpenetration_target_frame_count = 0 |

### 1.3 Optimizer (inert = True)

`nfev=1`, `cost=0.0`, `residual_rms_before=0.0`, `residual_rms_after=0.0`,
message "`gtol` termination condition is satisfied." The graph satisfied gtol on
the first evaluation because the objective was already zero — there was nothing
to optimize.

### 1.4 Direct fit frames (13)

`[30, 31, 32, 33, 34, 35, 36, 45, 46, 60, 75, 76, 77]`

### 1.5 Frozen-pose clusters (top 2)

- **73 frames** at translation `[0.390, -0.110, 0.344] m` — frames 77–149
  (1 direct fit at f77 + 72 nearest-hold). This is the tail half of the clip frozen.
- **31 frames** at translation `[0.578, -0.096, 0.339] m` — frames 0–30
  (1 direct fit at f30 + 30 nearest-hold). The head of the clip frozen before
  the object is ever seen.

### 1.6 Held / interpolated segments

- nearest_visible_pose_hold: **102 frames** across two long frozen segments.
- interpolated_between_visible_pose_observations: **35 frames** bridging f36→f45
  (the contact window interior).

### 1.7 Direct-edge translation/rotation jumps (12 edges between consecutive direct fits)

| | median | p90 | max | mean |
|---|--:|--:|--:|--:|
| translation (mm) | 25.6 | 117.1 | **166.4** | 47.5 |
| rotation (deg) | 5.3 | 35.3 | **42.1** | 12.0 |

The direct fits are physically impossible for a keyboard on a desk: f75→f76 jumps
166 mm / 36.8°, f46→f60 jumps 74 mm / 42.1°. These are independent per-frame ICP
fits with no temporal smoothing — exactly what an inert graph leaves behind.

### 1.8 Input-output deltas (pass-through = True)

For all 13 direct-fit frames, the graph output translation equals the input
visible-pose-fit translation to within 1e-6 mm (max delta = 0.0 mm). The graph
is a pure pass-through: it does not modify a single pose.

### 1.9 Render default / contact-state coupling

| pose_measurement_status | → contact_state (count) |
|---|---|
| missing_initial_graph_pose (65) | unresolved_evidence_incomplete 53, pose_unresolved 12 |
| visible_surface_ineligible_for_rigid_pose_fit (72) | unresolved_evidence_incomplete 72 |
| fit_to_visible_depth_samples (13) | unresolved_evidence_incomplete 4, geometry_epoch_contaminated 7, full_frame_depth_leak 1, unresolved_incoherent_evidence 1 |

The render defaults 129/150 frames to `unresolved_evidence_incomplete`. Of those,
72 come directly from visible-surface-ineligible frames (the degenerate-mask
population) and 53 from not-visible frames. The render's `defaulted_frame_count=129`
is structurally downstream of the pose graph's 137 unlocalized frames: fixing pose
localization is the lever that shrinks the render's defaulted population.

### 1.10 Visible-geometry eligibility (the M2 discriminator)

The visible-geometry adapter reports **85** frames with a visible metric surface,
but only **13** pass `rigid_pose_observation_eligible=True`. The other **72** are
ineligible, all with the same reason:
`systematic_mask_extent_inconsistent_with_selected_anchor_rigid_object_probable_hand_background_leakage`.

The discriminating measurement is the mask extent ratio to the anchor axis:
- ineligible frames: median **4.72×** (range 4.10–7.18) — one axis is 4–7× too large.
- eligible frames: median **1.84×** (range 1.0–3.13).

This is not a near-planar or low-texture degeneracy (M2 as originally framed in
card 1); it is a **mask segmentation failure**: the SAM2 keyboard mask is leaking
into the hand/background on 72/85 visible frames, inflating the depth-axis extent
4–7×, which the rigid-fit eligibility gate correctly rejects. The visible surface
is present but the mask is contaminated.

## 2. Mechanism adjudication

**legitimate_static_held_gauge is refuted.** The 13 direct-fit frames jump up to
166 mm / 42° frame-to-frame — not static. No gauge is declared in the report. The
frozen holds (73 + 31 frames) are an artifact of missing data (nearest-hold fill),
not a declared rest pose. A static object under a held gauge would show near-zero
direct-fit deltas with a declared gauge; neither holds.

**The defect is mixed, with three concurrently present mechanisms:**
1. **no_seed_support (65/150, 43%)** — the object is genuinely not visible (no
   SAM2 mask, no metric depth) on the first 30 frames and scattered gaps. Hard
   information limit; cannot seed a pose where there is no observation.
2. **visible_surface_degenerate (72/150, 48%)** — the object IS visible with a
   metric surface, but the SAM2 mask leaks into hand/background, inflating the
   extent ratio 4–7×, and the eligibility gate rejects it. This is the largest
   single cause of unlocalization and it is a **perception/segmentation defect**,
   not a geometry or solver defect.
3. **solver_plumbing_inert (150/150, global)** — even on the 13 frames that do
   get a direct ICP fit, the temporal graph is a pure pass-through (nfev=1, cost 0,
   zero correction, zero input-output delta). The 166 mm direct-fit jumps go
   unsmoothed. The "graph" is a relabeling of independent per-frame fits, not a
   trajectory solve.

Mechanisms 1 and 2 are the two co-dominant causes of the 137 unlocalized frames
(91%). Mechanism 3 is independently and globally true: even if localization were
fixed, the graph would still not produce a coherent trajectory because it does not
run. The classification is `mixed` because no single mechanism accounts for the
defect alone, and the next intervention differs per mechanism.

## 3. Implications for the next intervention (not implemented here)

- **M2 (visible_surface_degenerate) is the highest-leverage perception target:**
  72 frames have a visible metric surface that is discarded purely because the
  SAM2 mask leaks. Repairing the mask (hand/background quarantine, or a tighter
  open-vocabulary keyboard prompt) on these 72 frames would move them from
  ineligible to eligible, potentially multiplying the direct-fit count from 13 to
  ~85. This is the single intervention with the largest frame-count payoff.
- **M3 (solver_plumbing_inert) must be fixed regardless:** even with more seeds,
  the temporal graph does not run. The correction_summary deltas are identically
  zero, meaning the temporal factor, nonpenetration factor, and any smoothing are
  all inactive. The optimizer wiring must be repaired before trajectory quality
  can improve.
- **M1 (no_seed_support) is a hard information limit** on 65 frames (object out of
  view); these can only be held/interpolated, never directly measured. The honest
  artifact is a declared `pose_held`/`pose_out_of_view` state, not a fabricated
  measurement.

The required artifact change (per EPISTEMIC frontier) is numeric graph-health /
object-pose rows with support fractions, active residuals, input-output deltas,
and a measured-vs-held render state. This probe produces those measurements; the
next step is the repair.

## 4. Scope discipline

- One new additive script: `scripts/analyze_clip001850_pose_graph_liveness.py`.
- Zero edits to existing scripts.
- No staging, no commits. `git status --short` shows the script as `??` (untracked);
  `git diff --cached --name-only` is empty.
- No heavy inference / GPU: CPU-only JSON/numpy.
- Outputs to `/tmp/clip001850_pose_graph_liveness/` only.

## 5. Residual risks

- The eligibility gate (`extent_ratio_to_anchor_axis_max` with an implicit
  threshold near ~3.1×, the max among eligible frames) is a heuristic in the
  upstream adapter, not validated here. If the threshold is too strict, some of
  the 72 ineligible frames may be fit-able with a looser gate or a repaired mask.
  This probe measures the gate's effect; it does not validate the threshold.
- The extent-ratio signal (4–7× on one axis) strongly indicates mask leakage, but
  confirming the leak requires inspecting the actual SAM2 masks against the RGB
  frames — not done here (CPU-only, no mask rendering). The reason string in the
  adapter ("probable_hand_background_leakage") is the upstream diagnosis, carried
  as evidence.
- The frozen-pose cluster at `[0.390, -0.110, 0.344]` (73 frames, f77–149) is the
  f77 direct fit held forward. f77 is itself a direct ICP fit with a 36° rotation
  jump from f76; if f77 is a fit artifact, the entire tail is frozen on an
  artifact. This is consistent with the solver_plumbing_inert finding.

## Acceptance report

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Implemented the requested CPU liveness probe as one additive script scripts/analyze_clip001850_pose_graph_liveness.py (untracked, not staged). Consumes the pose graph report, upstream visible-pose-fit observation rows, visible-geometry adapter report, raw frame manifest, and full-duration contact-state render manifest. Outputs /tmp/clip001850_pose_graph_liveness/{summary.json, per_frame.ndjson (150 rows), pose_edges.ndjson (12 edges)}. Measures all required quantities: pose status counts (missing 65 / ineligible 72 / fit 13), active support by factor family (temporal=no, depth_icp=yes 13 frames, nonpenetration=no), optimizer nfev=1/cost=0/residuals=0 (inert=true), direct fit frame list [30-36,45,46,60,75,76,77], frozen-pose clusters (73-frame + 31-frame), held/interpolated segments (102 hold / 35 interp), direct-edge jumps (trans max 166.4mm / rot max 42.1deg), input-output deltas (pass-through=true, max 0.0mm), render/contact-state coupling (129 defaulted). Classifies dominant mechanism = mixed (no_seed_support 43% + visible_surface_degenerate 48% + solver_plumbing_inert global; legitimate_static_held_gauge refuted). No smoothing/repair. No scope widening: no edits to existing scripts."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "All load-bearing numbers re-derived from on-disk inputs and cross-checked against subagent 21 Fact A and subagent 19 direct-edge measurements. Optimizer inert verified (nfev=1, cost=0.0, residual_rms 0.0/0.0, correction deltas all 0.0, input-output delta max 0.0mm over 13 fit frames = pass-through). Pose status verified (missing 65 / ineligible 72 / fit 13 = 150). Visible-geometry eligibility verified (85 visible metric, 13 eligible, 72 ineligible with extent_ratio median 4.72x vs eligible 1.84x, all 72 same leakage reason). Direct edges verified (f75->f76 166.4mm/36.8deg max). Frozen clusters verified (73 frames at [0.390,-0.110,0.344] f77-149; 31 frames at [0.578,-0.096,0.339] f0-30). Render coupling verified (129 defaulted = 72 ineligible + 53 missing + 4 fit-outside-window). Independent reviewer can rerun the script with the documented command and inspect the three output files."
    }
  ],
  "changedFiles": [
    "scripts/analyze_clip001850_pose_graph_liveness.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "/home/yiwen/ego_annotation/.venv/bin/python scripts/analyze_clip001850_pose_graph_liveness.py --run-root /data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1",
      "result": "passed",
      "summary": "Wrote /tmp/clip001850_pose_graph_liveness/{summary.json, per_frame.ndjson (150 rows), pose_edges.ndjson (12 edges)}. dominant_mechanism=mixed; present=[no_seed_support, visible_surface_degenerate, solver_plumbing_inert]; optimizer inert=true (nfev=1, cost=0.0); direct_fit_frames=13."
    },
    {
      "command": "python3 inspect summary.json fields + per_frame.ndjson + pose_edges.ndjson",
      "result": "passed",
      "summary": "Verified all required measurements present in summary.json: pose_status_counts, support_by_factor_family, optimizer, direct_fit_frames, frozen_pose_clusters, held_interpolated_segments, direct_edge_jumps, input_output_deltas (pass_through=true), render_contact_state_coupling, visible_geometry_eligibility, classification. per_frame.ndjson=150 rows; pose_edges.ndjson=12 edges with translation/rotation jumps."
    },
    {
      "command": "git status --short scripts/analyze_clip001850_pose_graph_liveness.py && git diff --cached --name-only",
      "result": "passed",
      "summary": "Script untracked (??); nothing staged (git diff --cached empty). On branch yiwen_research."
    }
  ],
  "validationOutput": [
    "dominant_mechanism = mixed; present = [no_seed_support (65/150, 43%), visible_surface_degenerate (72/150, 48%), solver_plumbing_inert (150/150, global)]; legitimate_static_held_gauge refuted (direct fits jump 166mm/42deg, no declared gauge).",
    "optimizer inert = true: nfev=1, cost=0.0, residual_rms before=0.0 after=0.0, message 'gtol termination condition satisfied'.",
    "support_by_factor_family: temporal_smooth active=false (correction delta max 0.0m); depth_icp_visible active=true (13 fit frames, observed_to_mesh_final median 13.8mm); nonpenetration active=false (0 target frames).",
    "direct_fit_frames (13): [30,31,32,33,34,35,36,45,46,60,75,76,77].",
    "frozen_pose_clusters: 73 frames at [0.390,-0.110,0.344] (f77-149, 1 direct + 72 hold); 31 frames at [0.578,-0.096,0.339] (f0-30, 1 direct + 30 hold).",
    "held/interpolated: 102 nearest_visible_pose_hold + 35 interpolated.",
    "direct_edge_jumps: translation median 25.6mm / max 166.4mm (f75->f76); rotation median 5.3deg / max 42.1deg (f46->f60).",
    "input_output_deltas: pass_through=true; max translation delta 0.0mm over 13 fit frames (graph does not modify any pose).",
    "render_coupling: 129/150 frames default to unresolved_evidence_incomplete (72 from ineligible + 53 from missing + 4 fit-outside-window).",
    "visible_geometry_eligibility: 85 visible metric surface, 13 eligible, 72 ineligible (extent_ratio median 4.72x vs eligible 1.84x; all 72 reason = systematic_mask_extent_inconsistent probable hand/background leakage)."
  ],
  "residualRisks": [
    "The rigid-fit eligibility gate (extent_ratio_to_anchor_axis_max, implicit threshold ~3.1x) is an upstream heuristic not validated here; some of the 72 ineligible frames may be fit-able with a repaired SAM2 mask or looser gate. This probe measures the gate's effect, not the threshold's correctness.",
    "The extent-ratio signal (4-7x on one axis) strongly indicates SAM2 mask leakage, but confirming the leak requires inspecting masks against RGB frames (not done; CPU-only probe). The adapter's reason string is carried as upstream evidence.",
    "The 73-frame frozen cluster at [0.390,-0.110,0.344] is the f77 direct fit held forward; f77 itself is a direct ICP fit with a 36.8deg rotation jump from f76. If f77 is a fit artifact, the entire clip tail is frozen on an artifact (consistent with solver_plumbing_inert).",
    "This probe measures liveness; it does not smooth, repair, or re-solve. The next intervention (mask repair for M2, solver wiring for M3) is not implemented here."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one untracked additive analysis script (scripts/analyze_clip001850_pose_graph_liveness.py) and /tmp outputs (summary.json, per_frame.ndjson, pose_edges.ndjson). No source/state/render/publish code edited. Nothing staged, nothing committed.",
  "reviewFindings": [
    "no blockers: the probe consumes all five named input sources (pose graph report, visible geometry, raw frame manifest, contact-state render manifest, pose-fit observation rows) and produces all required measurements in the three named output files.",
    "no blockers: classification is mixed with three concurrently-present mechanisms, each backed by measured evidence (frame counts, optimizer state, input-output delta, extent ratios); legitimate_static_held_gauge is explicitly refuted with the 166mm direct-fit jumps.",
    "note: the highest-leverage next intervention is M2 (visible_surface_degenerate): 72 frames have a visible metric surface discarded purely due to SAM2 mask leakage (extent ratio 4-7x). Repairing the mask could move direct-fit coverage from 13 to ~85 frames — the single largest payoff. M3 (solver wiring) must be fixed regardless since the graph is a pass-through."
  ],
  "manualNotes": "The pose graph is unlocalized on 137/150 frames via two co-dominant perception causes (65 not-visible + 72 mask-leak-ineligible) and is globally inert as a solver (nfev=1, cost=0, zero correction, pass-through confirmed by zero input-output delta). The 13 direct fits jump up to 166mm/42deg unsmoothed. The render defaults 129 frames to unresolved_evidence_incomplete, structurally downstream of the 137 unlocalized pose frames. legitimate_static_held_gauge is refuted. The next artifact-changing work is: (1) repair the SAM2 keyboard mask on the 72 leaking frames (M2, highest frame-count payoff) and (2) repair the temporal graph wiring so it actually smooths/fills (M3). This probe provides the numeric graph-health/object-pose measurements that gate those interventions; it does not perform them."
}
```
