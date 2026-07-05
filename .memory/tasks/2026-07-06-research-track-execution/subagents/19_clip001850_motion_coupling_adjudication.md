# 19 — Motion-coupling contact-promotion adjudication for clip001850

Branch `yiwen_research`. Additive script only; nothing staged, nothing committed.
Script: `scripts/analyze_clip001850_motion_coupling_contact.py`
Outputs: `/tmp/clip001850_motion_coupling_contact/{summary.json, per_frame.ndjson, review.png, review_raw_frames.jpg}`

Slice: HOT3D clip001850 keyboard, right hand, frames 28-48, run
`20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.

---

## 0. Causal card (read first)

This is the **next admissible contact-promotion channel** after the signed-distance /
geometry-epoch / hand-depth channels were all refuted (subagents 07, 10-17). KT-U
(subagent 14) names the only two floor-independent channels that may legitimately promote
a frame beyond `unresolved` GT-free: **object-motion onset time-locked to hand kinematics**,
or R8 GT. This task tests the first one.

- **Artifact defect:** zero admissible contact frames in window 28-48. The canonical
  `contact_frame_detail` routes 12 frames to `pose_unresolved`, 7 to
  `geometry_epoch_contaminated`, 1 to `full_frame_depth_leak`, 1 to
  `unresolved_incoherent_evidence`.
- **Physical variable under test:** keyboard rigid motion (per-frame translation delta +
  rotation delta) vs right-hand root motion and fingertip approach to the keyboard.
- **Live mechanisms:**
  1. *Hand-driven object impulse* — if the hand contacts/pushes the keyboard, the
     keyboard should exhibit a localized translation/rotation spike at the fingertip
     approach frame, exceeding its independent motion floor. Predicts a positive
     time-locked spike at f~36-39 (closest approach).
  2. *Negligible/jitter-dominated object motion* — the keyboard is table-rested; per-frame
     deltas are dominated by head-cam ego-motion + ICP/depth-fit jitter, with no localized
     impulse at the approach frame. Predicts deltas of the same order everywhere, no
     time-lock.
- **Discriminating measurement (no distance threshold):** per-frame keyboard translation/
  rotation delta restricted to DIRECT→DIRECT pose edges (the only independent object-motion
  measurements; held/interp transitions are artificially zero or linear-fill), compared
  against (a) an independent motion floor (median + 3·MAD over all 9 direct edges), (b) a
  physical typing prior (≤5 mm), (c) a localized-impulse test at the approach-speed peak
  frame, and (d) Spearman correlation of object delta vs fingertip approach speed.
- **Predictions before running:**
  - If mechanism 1 dominates → a keyboard delta clears the floor AND is the window max AND
    coincides (±1 frame) with the fingertip approach peak/closest approach → candidate
    promotion frame, requires R8 GT confirmation.
  - If mechanism 2 dominates → no edge clears the floor, the approach-frame keyboard
    response is ≤ window median, correlation is non-significant → **confirmed_contact
    remains unreachable GT-free; motion coupling cannot promote any frame.**

## 1. Inputs consumed (existing CPU files only; no GPU / heavy inference)

- `RUN/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`
  — 150 keyboard pose rows (rotation matrix + translation_world_m); only 13 are direct
  visible measurements (`graph_frames = [30,31,32,33,34,35,36,45,46,60,75,76,77]`), the
  other 137 are nearest-hold or linear interpolation.
- `RUN/measurements/mano_interval_correction/.../v18_joint_mano_interval_trajectory_state.json`
  — 300 per-frame MANO rows (left+right), continuous trajectory; right-hand
  `optimized_translation_world_m` + `optimized_joints_world_m` (21 joints, fingertips at
  4/8/12/16/20). Hand and keyboard share the same world frame (verified: right wrist
  ≈[0.46-0.54, -0.17..-0.29, 0.15-0.18], keyboard centroid ≈[0.56-0.59, -0.06..-0.10,
  0.33-0.39]).
- `DURABLE/contact_state_table/contact_frame_detail.ndjson` — canonical per-frame
  contact_state for cross-reference.
- `RUN/input/raw_frame_manifest/rgb/*.jpg` — review contact sheet only.

## 2. Result — verdict: **motion coupling absent; no frame can be promoted**

`summary.json` `verdict = motion_coupling_absent_no_impulse_above_motion_floor`.

Five independent tests all converge on the negative:

### 2.1 No object-motion edge clears the independent floor
The keyboard's per-frame translation delta over the 9 DIRECT→DIRECT edges across the whole
timeline has median **21.1 mm**, MAD-scaled σ **18.6 mm** → motion floor (median + 3·MAD) =
**76.9 mm**. The largest window edge (f46, **57.2 mm**) does **not** clear it. Rotation
floor = **18.2°**; f46's 21.5° is the only edge near it, but f46 is a withdrawal-frame
re-fit, not a contact impulse (see 2.4).

The 9-sample floor is itself inflated by exactly the two non-contact re-fit jumps (f46,
f76/f77) where the SAM2 mask re-acquires the object — so the floor is conservative (biased
*against* finding a contact impulse), and the verdict still holds.

### 2.2 The approach-frame keyboard response is BELOW the window median (anti-correlation)
At the **fingertip approach-speed peak f36** (approach = **76.4 mm/frame**, the strongest
hand motion toward the keyboard), the keyboard translation delta is **8.6 mm** — *smaller*
than the window direct-edge median (**20.2 mm**) and far below the window max (57.2 mm).
The keyboard moves *less* at the moment of strongest hand approach than it does at f31-f33
when the hand is 143-157 mm from the centroid. This is the opposite of a hand-driven
impulse.

### 2.3 Spearman correlation is non-significant and slightly negative
Object dtrans vs fingertip approach speed over the 7 window direct edges: **ρ = −0.25,
p = 0.59**. No positive coupling; if anything, weakly anti-correlated.

### 2.4 The dominant object-motion edge is a withdrawal re-fit, not contact
The largest keyboard delta (f46: 57.2 mm, 21.5°) occurs when the **hand has already
withdrawn** (min tip→centroid = 158.8 mm, approach speed = −25.6 mm/frame, i.e. hand moving
*away*). It coincides with a large right-hand root motion (48.9 mm) — consistent with a
head-cam ego-motion + ICP re-initialization when the mask re-acquires, not a contact
impulse. It is not time-locked to the approach (closest approach f39, approach peak f36;
both >1 frame away).

### 2.5 The closest-approach frame has no independent keyboard measurement
The closest fingertip-to-centroid frame is **f39 (60.3 mm)**, but f39 is an *interpolated*
pose (`interpolated_between_visible_pose_observations`), so its keyboard delta (constant
4.53 mm linear fill) is not an independent motion measurement. The honest statement is: at
the frame of closest approach, no independent keyboard motion is measurable, and the two
bracketing direct edges (f36, f45) both show keyboard motion at or below the jitter median.

### 2.6 Window motion amplitudes
- Keyboard total translation (f30→f46, first→last direct): **41.4 mm** over 16 frames.
- Right-hand root total translation (f28→f48): **40.2 mm**.
- Ratio ≈ **1.03** — but this net-displacement ratio is misleading: the keyboard's per-frame
  deltas are scattered 4-57 mm with no localization at the approach, while the hand executes
  a clear approach-and-withdraw arc (approach peaks +76 mm/frame at f36, withdrawal peaks
  −50 mm/frame at f40). The keyboard shows no corresponding impulse. A table-rested keyboard
  under typing should move a few mm at contact and be near-zero otherwise; the measured 4-57
  mm scatter is jitter, not typing.

## 3. Conclusion

**Motion coupling does not promote any frame in 28-48 beyond `unresolved`.** There is no
hand-driven object-motion impulse that (a) clears the independent direct-edge motion floor,
(b) is the localized window maximum, and (c) is time-locked (±1 frame) to fingertip
approach. The dominant keyboard motion edge (f46) is a withdrawal-time pose re-fit; the
approach-peak frame (f36) shows keyboard motion *below* the jitter median; the closest
approach (f39) is interpolated and has no independent keyboard measurement. Spearman ρ =
−0.25 (p=0.59).

This is consistent with the keyboard being a **table-rested object whose per-frame pose
deltas are dominated by head-mounted-camera ego-motion + ICP/depth-fit jitter**, not by
hand-driven impulses. It is also consistent with subagent 14's §0 finding that the right
hand is 80-180 mm in front of the keyboard-masked depth surface on every mask-available
frame (i.e. the hand likely hovers / reaches without firm contact, or keyboard depth is
biased far).

**`confirmed_contact` remains unreachable GT-free on clip001850 window 28-48.** The only
remaining admissible route to a confirmed verdict is R8 (HOT3D GT), which exists for this
development-set clip but is deferred by the GT-free-first constraint. No frame should be
promoted by a distance threshold, a shrunken gap, a signed-distance sign flip, or an agent
interaction judgment.

## 4. Required provenance if a frame were ever promoted

If a future channel (e.g. R8 GT) does promote a frame, the canonical
`contact_frame_detail` row for that frame must carry a populated floor-independent
provenance column — `motion_coupling_onset` (the direct-edge frame, dtrans/drot, the
approach frame it time-locks to, and the clearance over the motion floor) or `hot3d_gt_r8`.
The motion-coupling evidence above shows **no such column can be populated truthfully for
any frame in 28-48 from the existing CPU pose/MANO artifacts.**

## 5. Method / honesty notes

- **No distance threshold was used to create contact.** The fingertip-to-centroid distance
  is reported only as the *coupling target* (where approach peaks occur); it is never
  thresholded to declare contact.
- The motion floor is computed only from DIRECT→DIRECT pose edges. Held/interp transitions
  are artificially zero (nearest-hold) or constant linear fill (interp) and would either
  deflate the floor or inject fake structure; both are excluded.
- The 9-edge floor is small and inflated by two mask-reacquire jumps; the verdict is
  therefore conservative. Even under a permissive physical typing prior (≤5 mm), the
  approach-frame keyboard response (8.6 mm) is not a localized maximum and is not
  time-locked — so the negative holds under either floor definition.
- Outputs are additive (`/tmp/...` + one untracked script). Nothing staged, nothing
  committed. No heavy inference / GPU.

## 6. Residual risks / boundaries

- The keyboard pose graph has only 13 direct measurements in 150 frames (9 direct→direct
  edges total, 7 in the window). Sub-frame or sub-cm typing impulses are below the pose
  graph's temporal/metric resolution and cannot be resolved by this channel — a genuinely
  soft touch would be invisible here. This does not contradict the verdict; it bounds the
  channel's sensitivity.
- The fingertip-to-centroid distance uses the keyboard *centroid* (observable), not the
  contaminated completed-mesh surface. This is intentional and does not affect the
  motion-coupling verdict, which depends only on the *temporal coincidence* of motion
  onsets, not on absolute distance.
- R8 GT can adjudicate definitively and remains the only path to a confirmed verdict.

## Acceptance report

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Implemented the requested motion-coupling channel as an additive script scripts/analyze_clip001850_motion_coupling_contact.py (untracked, not staged). Consumes only existing CPU files: keyboard pose graph report (150 rows, 13 direct visible), MANO interval trajectory state (right hand continuous), canonical contact_frame_detail, raw rgb frames for the review contact sheet. Measures keyboard rigid motion (per-frame translation + rotation delta restricted to DIRECT->DIRECT pose edges) vs right-hand root motion and fingertip approach, with NO distance threshold used to create contact (fingertip-to-centroid distance is only the coupling target, never thresholded). Five independent tests: (1) motion floor median+3*MAD over 9 direct edges = 76.9mm trans / 18.2deg rot, largest window edge f46 57.2mm does not clear it; (2) approach-speed peak f36 keyboard response 8.6mm < window median 20.2mm (anti-correlation); (3) Spearman object-dtrans vs approach-speed rho=-0.25 p=0.59 n=7; (4) dominant edge f46 is a withdrawal re-fit (hand 158mm away, moving away), not time-locked to approach (closest f39, peak f36); (5) closest-approach f39 is interpolated, no independent keyboard measurement. Verdict: motion_coupling_absent_no_impulse_above_motion_floor; promoted_frames=[]; confirmed_contact unreachable GT-free. Scope not widened: no edits to render/publish/state code, no GT, no heavy inference."
    }
  ],
  "changedFiles": [
    "scripts/analyze_clip001850_motion_coupling_contact.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 scripts/analyze_clip001850_motion_coupling_contact.py",
      "result": "passed",
      "summary": "Wrote /tmp/clip001850_motion_coupling_contact/{summary.json,per_frame.ndjson(150 rows),review.png,review_raw_frames.jpg}. Verdict motion_coupling_absent_no_impulse_above_motion_floor; promoted_frames=[]; floor trans 76.9mm/rot 18.2deg; approach-peak f36 kb response 8.6mm < median 20.2mm; Spearman rho=-0.25 p=0.59."
    },
    {
      "command": "python3 -c json.load summary.json + per_frame.ndjson window inspection",
      "result": "passed",
      "summary": "Verified per_frame rows 28-48: direct edges f31(20.2mm) f32(28.2) f33(21.1) f34(4.8) f35(4.1) f36(8.6) f46(57.2); interpolated f37-45 constant 4.53mm linear fill (correctly excluded); approach peak f36 +76mm/frame; closest approach f39 60.3mm interpolated; f46 withdrawal hand 158mm away."
    },
    {
      "command": "git status --short scripts/analyze_clip001850_motion_coupling_contact.py && git diff --cached --stat",
      "result": "passed",
      "summary": "Script untracked (??); nothing staged (git diff --cached empty). Other modified files in tree are pre-existing dirty, not mine."
    }
  ],
  "validationOutput": [
    "Verdict: motion_coupling_absent_no_impulse_above_motion_floor. No frame in 28-48 promotable by motion coupling.",
    "Object motion floor (median+3*MAD over 9 DIRECT->DIRECT edges, full timeline) = 76.9mm translation / 18.2deg rotation; largest window edge f46=57.2mm does not clear it.",
    "Approach-speed peak f36 (fingertip approach +76.4mm/frame): keyboard delta 8.6mm, BELOW window direct-edge median 20.2mm and max 57.2mm -> anti-correlation, not coupling.",
    "Spearman(object dtrans, approach speed) over 7 window direct edges: rho=-0.25, p=0.59 -> no significant positive coupling.",
    "Dominant object edge f46 (57.2mm, 21.5deg) occurs at hand withdrawal (min tip->centroid 158.8mm, approach -25.6mm/frame): mask-reacquire/ego-motion re-fit, not contact; not time-locked to closest approach f39 or approach peak f36 (>1 frame).",
    "Closest approach f39 (60.3mm) is an interpolated pose -> no independent keyboard motion measurable there; bracketing direct edges f36/f45 both at/below jitter median.",
    "No distance threshold used to create contact; fingertip-to-centroid distance reported only as coupling target. confirmed_contact remains unreachable GT-free; only R8 (HOT3D GT) can adjudicate definitively."
  ],
  "residualRisks": [
    "Pose graph has only 13 direct measurements in 150 frames (9 direct->direct edges, 7 in window); sub-frame or sub-cm typing impulses are below the pose graph's temporal/metric resolution and would be invisible to this channel. Does not contradict the verdict; bounds channel sensitivity.",
    "9-edge motion floor is small and inflated by two mask-reacquire jumps (f46, f76/f77); the verdict is conservative (biased against finding a contact impulse) and still holds; even under a permissive 5mm physical typing prior the approach-frame response (8.6mm) is not a localized maximum and is not time-locked.",
    "Fingertip-to-keyboard distance uses keyboard centroid (observable) not the contaminated completed-mesh surface; intentional and does not affect the motion-onset-coincidence verdict.",
    "R8 HOT3D GT exists for this development-set clip and remains the only path to a confirmed verdict; deferred by the GT-free-first constraint."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one untracked additive analysis script (scripts/analyze_clip001850_motion_coupling_contact.py) and /tmp outputs (summary.json, per_frame.ndjson, review.png, review_raw_frames.jpg). No source/state/render/publish code edited. Nothing staged, nothing committed.",
  "reviewFindings": [
    "no blockers: the motion-coupling channel is the next admissible floor-independent contact-promotion route (KT-U) and was tested honestly with no distance-threshold contact creation; the negative verdict is supported by five independent convergent tests (motion floor, approach-frame response, Spearman, time-lock, closest-approach interpolation).",
    "note (sensitivity bound): the keyboard pose graph's 13-direct-measurement / 9-direct-edge resolution means a genuinely soft/sub-cm typing touch would be below this channel's sensitivity; this is a bound on the channel, not evidence of contact. R8 GT is the definitive adjudicator."
  ],
  "manualNotes": "Motion coupling is negative and the negative is well-evidenced: at the fingertip approach-speed peak (f36) the keyboard moves 8.6mm, LESS than its 20.2mm jitter median at frames where the hand is far (f31-f33) - i.e. the keyboard moves less when the hand approaches most. The largest keyboard motion (f46, 57mm/21.5deg) happens when the hand has withdrawn to 158mm and is moving away, i.e. a mask-reacquire/ego-motion re-fit. Spearman rho=-0.25 p=0.59. The closest approach (f39, 60mm) is an interpolated pose with no independent keyboard measurement. Net: this is a table-rested keyboard whose per-frame deltas are dominated by head-cam ego-motion + ICP/depth-fit jitter, not hand-driven impulses. confirmed_contact stays unreachable GT-free on clip001850 28-48; the only remaining admissible route is R8 HOT3D GT. If a future channel ever promotes a frame, the contact_frame_detail row must populate a floor-independent provenance column (motion_coupling_onset or hot3d_gt_r8); the evidence here shows no such column can be truthfully populated for any frame in 28-48 from the existing CPU pose/MANO artifacts."
}
```
