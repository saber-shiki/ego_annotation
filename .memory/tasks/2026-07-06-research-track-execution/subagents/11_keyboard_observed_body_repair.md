# 11 — clip001850 keyboard observed-body geometry repair

CPU-only object-geometry repair for HOT3D clip001850 keyboard contact body, right
hand, frames 30-36/45/46. Zero model inference, zero GPU. New untracked script,
nothing staged.

Script: `scripts/repair_clip001850_keyboard_observed_contact_body.py`
Outputs: `/tmp/clip001850_keyboard_body_repair/{repaired_observed_contact_body.ply,
repaired_carved_support_body.ply, hand_to_body_distances.ndjson,
face_provenance_freespace_summary.json, summary.json, review_f032.jpg,
review_f036.jpg, review_f045.jpg}`

## Causal question

The completed TRELLIS keyboard body is 8.4× too thick in its thin axis (AABB
0.253×0.664×0.260 m vs keyboard prior 0.45×0.15×0.03 m), non-watertight, 95.4 %
TRELLIS-inferred, free space never carved. The interval solver's "observed-surface
penetration" channel queries signed distance against this inflated body, reporting
8-10 cm phantom hand penetration (f32 max 82 mm, f36 max 107 mm, f45 max 80 mm).

Subagents 07 and 10 established that the keyboard-masked, hand-quarantined depth
surface refutes this penetration on every frame: the hand sits 4-18 cm in front of
the real keyboard depth surface. The fork: is the phantom penetration a pure
geometry-epoch artefact that disappears once the body is restricted to observed
surface (M-A: geometry repair sufficient), or does the hand remain too far from the
real surface even after repair because the MANO hand-depth bias dominates the gap
(M-B)?

## Mechanism implemented

Three bodies are constructed from existing on-disk observations only:

1. **Observed-only contact body** — 3221 `observed_depth_surface` faces extracted
   from the completed mesh by face-label filter. This is the real observed keyboard
   front surface, no TRELLIS completion, no unsupported faces. 1726 vertices,
   non-watertight (open surface patch).

2. **Free-space-carved support body** — starts from the completed mesh and carves
   every `trellis_inferred_hidden_surface` + `unsupported_uncertain` face whose
   centroid projects onto a keyboard-mask pixel in any of the 9 available frames
   and whose camera-z is closer than the observed depth by >10 mm. Carved 49 222 of
   87 671 carveable faces (56 %). Resulting body: 41 670 faces, 21 511 vertices,
   still non-watertight (back/bottom was never observed).

3. **Completed body** (baseline) — the original 90 892-face TRELLIS-completed mesh,
   for comparison.

For each target frame, unsigned closest-surface distance from the 160 MANO sample
vertices to each body is computed in canonical object frame (hand verts transformed
by inverse pose). The keyboard-masked depth-surface gap along the camera ray is
computed independently for directional context (negative = hand in front).

## Discriminating result (the headline)

**Hand-depth bias dominates. Geometry repair removes the false penetration signal
but does not change the contact state from unresolved to contact.** On every target
frame, the hand is 36-115 mm from the nearest observed keyboard surface (unsigned 3D
distance), well above the clip's 28 mm sigma_clip floor. The keyboard depth-surface
gap confirms the hand is 75-179 mm in front of the real surface along the camera ray.

| frame | interval published pen `.max` | completed body unsigned median | **observed body unsigned median** | carved body unsigned median | kb-depth gap median |
|------:|-----:|-----:|-----:|-----:|-----:|
| 30 | (none) | 80.9 mm | **84.1 mm** | 84.1 mm | −172.7 mm |
| 31 | (none) | 40.1 mm | **45.8 mm** | 41.8 mm | −146.9 mm |
| 32 | +82 mm | 52.1 mm | **59.9 mm** | 54.4 mm | −157.3 mm |
| 33 | (none) | 63.6 mm | **70.9 mm** | 65.0 mm | −165.2 mm |
| 34 | (none) | 103.2 mm | **115.2 mm** | 106.1 mm | −179.3 mm |
| 35 | (none) | 95.5 mm | **110.9 mm** | 98.6 mm | −148.9 mm |
| 36 | +107 mm | 61.0 mm | **78.3 mm** | 65.9 mm | −83.2 mm |
| 45 | +80 mm | 30.4 mm | **35.6 mm** | 32.0 mm | −74.7 mm |
| 46 | (none) | 44.4 mm | **56.1 mm** | 45.3 mm | −116.9 mm |

The interval solver's "penetration" is the TRELLIS bulge enclosing the hand. At f36,
the solver reports 78 vertices penetrating 76 mm median / 107 mm max into the
completed mesh. The observed-only body says the same hand is 78 mm away from the
real surface. The TRELLIS bulge accounts for the entire phantom: removing it
*increases* the hand-to-body distance (completed 61 mm → observed 78 mm at f36),
confirming the inflated body was manufacturing false proximity.

## Why geometry repair does not change contact_state

- The observed-only body is the strongest contact-eligible body from existing
  observations. It contains no TRELLIS completion. Its unsigned closest-surface
  distance to the hand is the cleanest geometric measurement available.
- Even against this clean body, no frame reaches the 28 mm sigma_clip band. The best
  frame (f45) is 35.6 mm — 7.6 mm above the floor. The median across all frames is
  ~60-80 mm.
- The keyboard depth-surface gap (75-179 mm along camera ray) independently confirms
  the hand is far from the keyboard. This gap is the known clip001850 MANO hand-depth
  bias (~8 cm along-ray, systematic, non-averageable per hand_metric_mechanisms.md).
- Therefore: geometry repair is *necessary* (it eliminates the false penetration that
  would otherwise contaminate contact reasoning) but *not sufficient* (the hand-depth
  bias still places the hand outside the contact band). Resolving contact requires the
  R5 hand-depth-bias repair, not more geometry work.

## Watertight sign mesh — the exact missing condition

A watertight sign mesh is **impossible from existing observations**. The observed
keyboard surface is a one-sided partial patch: 3221 faces over a 0.20×0.54 m
footprint that captures the visible top/front of the keys but not the back, sides,
or bottom. It has boundary edges and encloses no volume. Free-space carving removes
TRELLIS faces that violate observed depth, but the remaining carved body (41 670
faces) is still open because the occluded back/bottom surfaces were never observed
by any camera frame in this clip. The exact missing topological/free-space condition:

> **A watertight sign mesh requires either (a) multi-view coverage that observes the
> keyboard's full closed surface (back, sides, bottom) — not available in this
> single-clip egocentric video where the keyboard rests on a desk and only its top
> face is ever visible — or (b) a parametric keyboard CAD model registered to the
> observed surface, providing the hidden surfaces with explicit prior provenance
> rather than TRELLIS free-space-uncarved inference.** Until one of these exists,
> signed penetration queries remain unreliable; only unsigned closest-surface
> distance to the observed patch is contact-admissible.

This is a fundamental information limit, not an implementation gap. The script does
not fake a watertight mesh. The observed-only body and carved body are both reported
as non-watertight with their face provenance intact.

## Free-space carving detail

Across the 9 carve frames, 49 222 of 87 671 carveable faces were violated (projected
onto keyboard mask pixels and closer to camera than observed depth by >10 mm). Frame
f30 alone violated 42 055 faces (the camera was most head-on). The surviving 38 449
carveable faces plus 3221 observed faces form the carved support body. The carved
body's canonical-frame AABB thin axis did not shrink (252.6 mm vs 252.6 mm completed)
because the canonical axes do not align with the camera viewing direction, and
surviving back/side faces still span the full canonical extent. The observed-only
body thin axis is 169.4 mm (67 % of completed) — thinner but still far above the 30 mm
keyboard prior, reflecting depth noise across multiple key surfaces rather than
TRELLIS inflation.

## Per-frame contact_state under the observed body

| frame | observed body state | meaning |
|------:|---|---|
| 30 | `no_contact_supported_beyond_3sigma` | 84.1 mm > 3×28 mm |
| 31 | `contact_candidate_uncertain_1to3sigma` | 45.8 mm, 1-3σ |
| 32 | `contact_candidate_uncertain_1to3sigma` | 59.9 mm, 1-3σ |
| 33 | `contact_candidate_uncertain_1to3sigma` | 70.9 mm, 1-3σ |
| 34 | `no_contact_supported_beyond_3sigma` | 115.2 mm |
| 35 | `no_contact_supported_beyond_3sigma` | 110.9 mm |
| 36 | `contact_candidate_uncertain_1to3sigma` | 78.3 mm, 1-3σ |
| 45 | `contact_candidate_uncertain_1to3sigma` | 35.6 mm, 1-3σ |
| 46 | `contact_candidate_uncertain_1to3sigma` | 56.1 mm, 1-3σ |

No frame reaches `contact_candidate_within_sigma_clip` (≤28 mm) or
`contact_candidate_within_soft_tissue` (≤15 mm). The `contact_candidate_uncertain`
label on 6/9 frames is an artifact of the unsigned-distance classification ceiling:
distance alone can at most reach `contact_candidate` (per subagent 10's policy, F2).
Even that ceiling is not reached at sigma_clip resolution.

## Decision

**`hand_depth_bias_dominates_geometry_repair_alone_insufficient`**

Geometry repair alone does not change f32/f36/f45 contact_state. The TRELLIS body
repair removes the false 8-10 cm penetration signal (necessary work), but the hand
remains 36-115 mm from the real observed keyboard surface — a gap dominated by the
clip001850 MANO hand-depth bias (~8 cm along-ray systematic). The next required
intervention is R5 hand-depth-bias repair, not further geometry work.

## Required outputs (all present)

- `repaired_observed_contact_body.ply` — 3221-face observed-only contact body
- `repaired_carved_support_body.ply` — 41 670-face free-space-carved body
- `hand_to_body_distances.ndjson` — 9 rows (f30-36, 45, 46), per-frame distances to
  3 bodies + keyboard depth gap + interval published penetration + contact_state
- `face_provenance_freespace_summary.json` — face provenance, watertight flags, carve
  counts, missing topological condition statement
- `summary.json` — headline decision + per-frame decisions + all body summaries
- `review_f032.jpg`, `review_f036.jpg`, `review_f045.jpg` — RGB + keyboard mask
  (green) + hand sample verts (red) + fingertip verts (magenta) + distance panels

## Reproduce

```bash
cd /home/yiwen/ego_annotation
/home/yiwen/ego_annotation/.venv/bin/python scripts/repair_clip001850_keyboard_observed_contact_body.py
# -> /tmp/clip001850_keyboard_body_repair/
```
