# 12 — Hand-depth-bias counterfactual: clip001850 keyboard, right hand

Slice: HOT3D `clip001850`, right hand, frames 30–36 & 45–46, run
`20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.
CPU-only. No model inference, no GPU. New untracked script, nothing staged.

Script: `scripts/counterfactual_clip001850_hand_depth_bias.py`
Outputs: `/tmp/clip001850_hand_depth_bias_counterfactual/{per_frame.ndjson, summary.json, review_f32.jpg, review_f36.jpg, review_f45.jpg}`

## Causal question

Subagent 07's kill-tests left an open caveat (D4): the keyboard-masked,
hand-quarantined depth surface shows the MANO right hand 4–18 cm *in front of*
the keyboard on every mask-available frame. Two live explanations: (a) the MANO
hand metric depth is biased ~8 cm too close to camera (a single systematic
along-ray offset that, if corrected, would land the hand on the keyboard and
produce contact), or (b) the gap is not a correctable systematic offset.

This counterfactual tests (a) directly: translate the *whole* MANO hand along
the camera ray (+camera z, away from camera) by exactly the amount needed to
bring the fingertip/keyboard-overlap vertices to the keyboard-masked depth
surface, then measure (i) whether contact_candidate results, (ii) the
reprojection cost, (iii) the completed-mesh distance change.

## Method

For each frame, using on-disk MANO sample verts (world), camera pose
(`T_world_camera_metric`), intrinsics (fx=fy=559.16, cx=720.9, cy=719.5),
UniDepth full-frame depth (1408²), SAM2 keyboard mask, and ownership masks
(visible_object_owned = keyboard − projected hand support):

1. Identify fingertip verts (nearest joint ∈ {thumb,index,middle,ring,pinky
   tips}) that project onto keyboard-masked, hand-quarantined pixels.
2. Compute the gap = `surface_depth − cam_z` for those verts (negative = hand
   in front). The required along-ray shift to bring each to the surface = gap.
3. Apply the representative shift (median over overlap fingertip verts) as a
   pure world translation along `R_c2w[:,2]` to the whole hand.
4. After shift, re-project and re-sample depth at the **keyboard mask** (the
   VOO quarantine is stale once the hand moves). Count fingertips within the
   15 mm soft-tissue band → contact_candidate.
5. Measure 21-joint image-plane reprojection displacement (px) before/after.
6. Measure completed-mesh unsigned distance before/after (trimesh closest_point).

Comparisons: required shift vs σ_clip floor (25–30 mm, from
`hand_metric_mechanisms.md` Track J/M); vs interval solver
`optimized_translation_camera_z_m` (the already-tolerated in-window drag);
reprojection vs interval solver `visible_joint_shift_px` (the already-tolerated
joint displacement, median 8–11 px, max ~13 px); completed-mesh distance before
vs after.

## Headline result

**The ~8 cm hand-depth-bias hypothesis is refuted.** The required shift is not a
uniform ~8 cm; it ranges **83–202 mm (median 134 mm)** across the 9 frames — a
2.4× span. A single systematic along-ray offset cannot explain it.

| frame | req shift mm | gap VOO med mm | reproj med px | reproj max px | interval cz mm | interval jointshift max px | cmesh before med mm | cmesh after med mm | contact? |
|------:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|:--|
| 30 | 161 | −161 | 169 | 285 | 0 | 0.1 | 81 | 13 | no |
| 31 | 134 | −134 | 136 | 224 | 28 | 12.8 | 40 | 14 | no |
| 32 | 125 | −125 | 123 | 200 | 28 | 12.7 | 52 | 13 | no |
| 33 | 169 | −169 | 141 | 240 | 27 | 12.9 | 64 | 23 | no |
| 34 | 202 | −202 | 156 | 280 | 0 | 0.2 | 103 | 33 | no |
| 35 | 156 | −156 | 123 | 232 | 0 | 0.2 | 95 | 20 | no |
| **36** | **83** | **−83** | **69** | **135** | 21 | 13.5 | 61 | 19 | **yes (8/34 in band)** |
| 45 | 85 | −85 | 62 | 110 | 38 | 12.6 | 30 | 12 | no |
| 46 | 103 | −103 | 78 | 139 | 1 | 0.4 | 44 | 13 | no |

### Two independent failures of the hypothesis

**Failure 1 — projection cost is catastrophic on every frame.** The
counterfactual shift causes 62–169 px median (110–285 px max) joint reprojection
displacement. The interval solver already tolerated only ~13 px max
(`visible_joint_shift_px`). The shift is **5–22× the existing projection
tolerance**. A hand at 0.4 m depth translating 8–20 cm along the ray contracts
its image footprint dramatically — the fingertips sweep inward off their image
evidence. On **0/9 frames** does the shift stay within the tolerated
reprojection. The hand would detach from its 2D detection entirely.

**Failure 2 — the shift does not even produce contact on 8/9 frames.** After
applying each frame's representative shift, only f36 reaches contact_candidate
(8 of 34 keyboard-projected fingertips within the 15 mm band). On f45, f46,
f30–35 the median fingertip is *still 42–58 mm in front of* the keyboard surface
after the shift. Mechanism: the keyboard surface is tilted and non-flat; a pure
along-ray translation moves each fingertip to a *different* surface point (image
projection contracts inward), so the single median-gap shift overshoots some
fingertips and undershoots others. No single translation lands the hand on the
surface. The required shift varies fingertip-to-fingertip and frame-to-frame by
more than the entire soft-tissue band.

### Completed-mesh distance confirms the shift is into the phantom bulge, not real contact

Completed-mesh unsigned distance drops from 30–103 mm (before) to 12–33 mm
(after). This is *not* contact evidence: the completed mesh is 8.4× too thick in
its thin axis (AABB 0.253×0.664×0.260 vs keyboard prior 0.45×0.15×0.03), 95.4%
TRELLIS-inferred, non-watertight (subagent 07). Moving the hand 8–20 cm deeper
drives it into the inflated mesh volume — exactly the artifact that produced the
interval solver's spurious "10.7 cm penetration" in the first place. The
distance drop corroborates that the mesh bulges toward the camera; it does not
corroborate contact.

## Decision

**An ~8 cm hand-depth correction is not physically plausible and would not
create contact_candidate without worsening projection beyond the existing
tolerance.**

1. **Not plausible as a single systematic bias.** The required shift is 83–202
   mm (median 134 mm), 2.8–6.7× the 25–30 mm σ_clip metric floor, and varies
   2.4× across frames. The metric floor is itself a *systematic* (non-averageable)
   bias per `hand_metric_mechanisms.md`; a correction 3–7× larger than the floor,
   and frame-varying, is not the same physical quantity.

2. **No contact without destroying projection.** Only 1/9 frames (f36) reaches
   contact_candidate, and even there the reprojection (135 px max) is 10× the
   tolerated 13 px. 0/9 frames stay within projection tolerance. The hand-depth
   bias hypothesis trades a depth gap for a 2D-alignment break an order of
   magnitude larger than anything the solver already accepted — and still fails
   to produce contact on 8/9 frames.

**Implication for the contact-state policy (subagent 10):** the open caveat D4
resolves against the "correct the hand metric depth bias to recover contact"
route. The 8–20 cm gap is not a correctable offset; it is a depth-surface vs
MANO-depth inconsistency whose correction is neither metrically admissible nor
projection-preserving. Clip001850 contact on these frames remains explicit
`unresolved` (depth-gap + hand-depth-bias-refuted + mesh-contaminated), not
no-contact and not confirmed contact. The next artifact-changing route is
repairing the object geometry epoch (free-space carving / observed-face
watertight sign mesh) so the keyboard body becomes contact-eligible, *not*
shifting the hand.

## Required outputs (all present)

- `per_frame.ndjson` — 9 rows (f30–36, 45, 46): gap summaries, required shifts
  (to-surface and to-band-edge, VOO and keyboard-masked), counterfactual result
  (shift vector, reprojection displacement, contact_after, completed-mesh dist
  before/after), decision.
- `summary.json` — parameters, required-shift summary (median 134 mm),
  frames-creating-contact-candidate [36], frames-within-projection-tolerance [],
  per-frame decisions, overall_verdict.
- `review_f32.jpg`, `review_f36.jpg`, `review_f45.jpg` — RGB + keyboard mask
  (green) + VOO (cyan) + original hand verts (cyan, fingertip magenta) + shifted
  hand verts (red, fingertip red) + shift arrows.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
/home/yiwen/ego_annotation/.venv/bin/python scripts/counterfactual_clip001850_hand_depth_bias.py
# -> /tmp/clip001850_hand_depth_bias_counterfactual/{per_frame.ndjson,summary.json,review_f32.jpg,review_f36.jpg,review_f45.jpg}
```

![f36 review](/tmp/clip001850_hand_depth_bias_counterfactual/review_f36.jpg)

![f45 review](/tmp/clip001850_hand_depth_bias_counterfactual/review_f45.jpg)

![f32 review](/tmp/clip001850_hand_depth_bias_counterfactual/review_f32.jpg)
