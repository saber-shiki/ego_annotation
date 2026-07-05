# 07 — clip001850 masked-depth contact kill-tests (KT-1/KT-2/KT-3/KT-5)

CPU-only kill-tests for HOT3D clip001850 keyboard, right hand, frames 28-48, run
`20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.
Zero model inference, zero GPU. New untracked script, nothing staged.

Script: `scripts/run_clip001850_masked_contact_killtests.py`
Outputs: `/tmp/clip001850_masked_contact_killtests/{contact_killtests_frame_detail.ndjson, summary.json, review_f32.jpg, review_f36.jpg}`

## Causal question

Subagent 06 (A1-A7) claimed the interval solver's 10.7 cm right-hand "observed-surface
penetration" at f36 is a full-frame-depth leak (table/hand pixels), and subagent 04
claimed it is coherent stranded contact evidence (M1/M5) worth wiring into contact/NP.
The fork: is the penetration **real keyboard contact** (wire it in) or an **artifact of
the inflated completed mesh** (repair geometry first)? KT-1/KT-2/KT-3/KT-5 discriminate.

A key correction to both priors: the interval solver's
`full_observed_surface_penetration_after_solver_m` is **not** the raw depth map — it is a
signed-distance query against the **completed mesh** (Open3D RaycastScene) filtered to
depth-supported observed faces (`full_observed_surface_measure`, solver line 1537). So the
"10.7 cm penetration" = hand vertices inside the completed-mesh volume whose nearest face
is observed-supported. KT-1 asks the independent question: where is the hand relative to
the **keyboard-masked depth surface** along the camera ray?

## Discriminating result (the headline)

**The keyboard-masked, hand-quarantined depth surface refutes the interval solver's
penetration on every mask-available frame.** The hand sits 4-18 cm **in front of** the
keyboard depth surface; the "penetration" is the 8.4×-too-thick completed mesh enclosing
the hand.

| frame | interval published pen `.max` | full-frame delta `.max` | **kb-masked+HQ delta median** | kb-masked+HQ pen count |
|------:|-----:|-----:|-----:|-----:|
| 30 | (none) | −0.043 | **−0.176** | 0 |
| 32 | +0.082 | −0.024 | **−0.144** | 0 |
| 36 | **+0.107** | +0.022 (leak) | **−0.081** | 0 |
| 45 | +0.080 | +0.034 | **−0.085** (1 vertex +0.006) | 1 |
| 46 | (none) | +0.005 | (n_region 0) | 0 |

`delta = cam_z_hand − depth_surface`; negative = hand in front of surface (gap), positive
= behind (penetrating). At f36 the interval solver says 78 vertices penetrate 7.6 cm
median / 10.7 cm max into the observed mesh surface; the keyboard-masked+hand-quarantined
depth says the same 12 keyboard-projected hand vertices are **8.1 cm in front** of the
keyboard surface. These are mutually exclusive for the same hand at the same frame.

## KT-1 — masked depth + hand quarantine + table plane

Three conditions per frame (delta along camera ray, positive = behind surface):
- **full-frame depth**: at f36, 11/160 vertices show delta>0 (max +0.022) — these project
  onto **non-keyboard** foreground pixels (the hand's own closer fingers / near objects).
  After keyboard masking they all vanish → this is the leak 06 predicted, but it is small
  (2.2 cm), not the 10.7 cm headline.
- **keyboard-masked depth** (SAM2 keyboard mask, 960² resized to 1408² nearest-neighbour):
  on all 9 mask-available frames delta is negative (hand in front). f32 worst: median
  −0.157 m.
- **keyboard-masked + hand-quarantined** (`visible_object_owned` = keyboard minus projected
  MANO hand support): identical sign, hand 4.6-17.6 cm in front. Penetrating count = 0 on
  8/9 frames; f45 has 1 vertex at +0.006 m (within soft-tissue band).
- **table/background plane** (fit to non-keyboard, non-hand depth): hand is 0.35-0.40 m in
  front of the background plane on all frames — no table penetration. The hand is in front
  of every surface the depth offers.

Proxy masks: frames 28,29,37-44,47,48 have **no keyboard mask files** (SAM2 only saved
masks at f30-36,45,46 in this window). For these the script holds the nearest available
mask (`proxy_temporal_hold`) and marks `mask_files_exist=false`; all 12 route to
`pose_unresolved` because the object pose is also missing/interpolated there. The exact
missing field per frame is in each NDJSON row (`missing_field`).

## KT-2 — per-vertex penetrating IDs + temporal IoU (no `.max`)

Per-vertex penetrating MANO sample IDs under the keyboard-masked+hand-quarantined
condition are **empty on 8/9 mask-available frames** (f45 has 1 id). The interval solver's
78 f36 penetrating vertices (ids from the mesh query) do not appear in the keyboard-depth
penetrating set at all — the two surfaces disagree about which vertices are "inside".

- Temporal IoU of keyboard-masked penetrating sets: mean **0.15**, interpretation
  `scattered_low_coherence`. Near-band IoU: 0.06. Penetrating-id union: 14 ids.
- `forbidden_max_scalar_as_coherence = true` on every row; coherence is reported as set
  IoU + anatomical region, never as a `.max`.

This kills the "wire the signal in" route: the mesh-penetration set is incoherent with the
keyboard depth surface, so it is not transferable contact evidence.

## KT-3 — f36 fingertip vertices vs three surfaces under one pose

37 fingertip vertices (nearest joint ∈ {thumb,index,middle,ring,pinky tips}), f36 fit pose:

| surface | delta/unsigned median | within 15 mm band |
|---|---|---|
| full-frame depth (ray) | −0.039 m (front) | 4/37 |
| keyboard-masked depth (ray) | **−0.086 m** (front, 8 verts) | **0/8** |
| keyboard-masked+HQ depth (ray) | −0.083 m (front, 5 verts) | 0/5 |
| completed mesh unsigned | +0.055 m (outside) | 3/37 |
| observed mesh unsigned | +0.071 m (outside) | — |

No fingertip is within the keyboard depth surface band. The completed-mesh unsigned
distance says fingertips are 5.5 cm **outside** the mesh — directly contradicting the
interval solver's signed claim that they are 7.6 cm **inside**. The contradiction is the
non-watertight mesh: best-effort `trimesh.signed_distance` is unreliable on it (exactly
the sign-channel failure 06/A3 and 04 identified), so the interval solver's "inside"
classification and the KT-3 unsigned "outside" are both artifacts of the same broken
geometry, not a real contact measurement.

## KT-5 — completed-mesh distance stratified by pose provenance

- fit-pose frames {30-36,45,46}: completed-mesh unsigned dist median-of-medians **0.061 m**;
  best-effort inside-count sum 91.
- missing/interpolated frames {28,29,37-44,47,48}: unsigned dist median **0.060 m**;
  best-effort inside-count sum 124.

Unsigned distance is **not** discriminative by pose provenance (0.061 vs 0.060). The
best-effort signed inside-count is **higher** on interpolated-pose frames (124 vs 91),
consistent with the interpolated pose drifting the mesh relative to the hand — but because
the mesh is non-watertight the signed count is not trustworthy either. Conclusion: pose
provenance is not the dominant driver of the mesh-distance contradiction; the mesh
geometry epoch is.

## Route distribution (frames 28-48, right hand)

| route | count | frames |
|---|---|---|
| `pose_unresolved` | 12 | 28,29,37-44,47,48 (missing/interpolated pose, proxy mask) |
| `geometry_epoch_contaminated` | 7 | 30-35,46 (masked depth shows hand in front; mesh pen uncorroborated) |
| `full_frame_depth_leak` | 1 | 36 (full-frame +0.022 from 11 non-keyboard verts, masked→0) |
| `contact_candidate_keyboard_masked` | 1 | 45 (1 fingertip within 6 mm of keyboard-masked+HQ surface) |
| `evidence_missing` | 0 | — |

## Mechanism decision

**M2 (geometry epoch contaminated) is the proximate mechanism, not M1/M5.** The interval
solver's 10.7 cm penetration is against a completed mesh that is 8.4× too thick in the
thin axis (AABB 0.253×0.664×0.260 vs keyboard prior 0.45×0.15×0.03), 95.4% TRELLIS,
non-watertight, free-space never carved. The keyboard-masked depth surface — the physical
first surface — shows the hand 4-18 cm in front of the keyboard on every measurable frame.
Wiring the interval penetration into contact/NP would pull the hand toward a phantom
surface that bulges 8 cm toward the camera.

**Do not wire the interval-solver observed-surface penetration in.** This overturns
subagent 04's M1/M5 recommendation, which was based on mesh-vs-mesh agreement without
checking the masked depth surface.

**Caveat — the kill-tests expose but do not resolve hand-depth bias (D4).** The agent
interaction judgment says "likely_contact" (right hand on keyboard) yet the depth shows an
8 cm gap. Two live explanations: (a) the MANO hand metric depth is biased ~8 cm too close
to camera (known HOT3D clip001850 along-ray bias), or (b) the keyboard depth is wrong. The
keyboard-masked depth surface is at ~0.50-0.52 m (physically sensible for a keyboard on a
desk viewed from a head cam), and the hand vertices are at cam_z ~0.40 m. If the hand is
truly resting on the keyboard, the hand vertices should be at ~0.50 m too — so the 8 cm
offset is most consistent with MANO hand-depth bias, not a keyboard-depth error. Resolving
this requires the GT-free hand-correction workstream (R5), not more depth masking. Until
then, contact state on these frames is `unresolved` with provenance: depth-gap +
hand-depth-bias-suspected + mesh-contaminated.

## Required outputs (all present)

- `contact_killtests_frame_detail.ndjson` — 21 rows (f28-48), each with KT1 (4 conditions
  + table plane), KT2 (per-vertex ids, IoU, anatomical region, forbidden-.max flag),
  interval-solver published numbers, pose provenance, mask source, route.
- `summary.json` — geometry epoch (watertight/extents/face provenance), route counts,
  KT2 window coherence, KT3 f36 fingertip table, KT5 pose-provenance stratification.
- `review_f32.jpg`, `review_f36.jpg` — RGB + keyboard mask (green) + visible-object-owned
  (cyan) + hand support (yellow) + MANO sample verts (red) + fingertip verts (magenta).

## Reproduce

```bash
cd /home/yiwen/ego_annotation
/home/yiwen/ego_annotation/.venv/bin/python scripts/run_clip001850_masked_contact_killtests.py
# -> /tmp/clip001850_masked_contact_killtests/{contact_killtests_frame_detail.ndjson,summary.json,review_f32.jpg,review_f36.jpg}
```
