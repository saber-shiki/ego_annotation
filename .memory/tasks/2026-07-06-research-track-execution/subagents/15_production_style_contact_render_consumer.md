# 15 — Production-style v19 contact-state render consumer (clip001850 f32/f36/f45)

Implements the smallest production-shaped consumer of the canonical
`ego.hoi.contact_frame_detail/0.1.0` table and the repaired observed keyboard
contact body. Renders overlay / world / side-by-side stills whose visible
labels are driven by the canonical `contact_state` and whose keyboard body is
drawn as an uncertain / hatched observed-surface patch, not a solid accepted
TRELLIS body.

CPU-only. No model inference, no GPU. New untracked script, nothing staged.

Script: `scripts/render_clip001850_v19_contact_state_artifact.py`
Outputs: `/tmp/clip001850_v19_contact_state_render/`

## Causal question (G0 — the consumer gap)

Subagents 06/10/13/14 all name the same blocker: no renderer in the v19
production path consumes the canonical `contact_frame_detail` table, so every
canonical `contact_state` row is backing-data-only. The published v19 banner
collapses f32/f36/f45 into a single vague
`gap 39.2mm | shift 0.0px | closed False | UNCERTAIN` line and draws the
keyboard as a solid green accepted rigid body. The artifact defect: the
delivered annotation does not visibly carry the named unresolved mechanism
(`geometry_epoch_contaminated` / `full_frame_depth_leak` /
`unresolved_incoherent_evidence`) and does not expose the body as the
contaminated, non-watertight observed patch it actually is.

Intervention: a new render path that (a) reads the canonical table, (b) reads
the repaired observed contact body, (c) draws the body as hatched/uncertain,
and (d) puts the canonical `contact_state` in the banner. This closes G0
nonzero-pixel/label-diff requirement.

## What the script consumes

- `/tmp/clip001850_contact_state_table/contact_frame_detail.ndjson` (canonical
  rows; `contact_state` is the authoritative visible label).
- `/tmp/clip001850_keyboard_body_repair/repaired_observed_contact_body.ply`
  (3221-face observed-only non-watertight patch, canonical object frame).
- `<RUN_ROOT>/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`
  (`rotation_world_from_completed_canonical_matrix` + `translation_world_m` per
  frame, to place the body in world).
- `<RUN_ROOT>/.../v18_joint_mano_interval_trajectory_state.json` (right-hand
  optimized sample verts + joints per frame, for the hand overlay).
- `<RUN_ROOT>/.../annotations_v19_visible_geometry.json` (per-frame camera
  `T_world_camera_metric` + intrinsics).
- `<RUN_ROOT>/input/raw_frame_manifest/rgb/{frame:06d}.jpg` (source RGB).
- Published v19 videos `renders/v19_{overlay,world,side_by_side}.mp4` and
  `renders/v19_published_runtime/v19_published_render_report.json` (the
  "before" state for pixel + label diff).

## What it produces (per frame f32/f36/f45)

- `overlay_f0XX.jpg` (960×1038): source RGB + projected MANO sample verts
  (cyan, fingertips magenta) + the observed contact body projected as a
  translucent amber hatched wireframe (back-face-culled, depth-sorted) with a
  legend line stating it is the UNCERTAIN non-watertight patch, NOT solid
  TRELLIS. Top banner carries the canonical `contact_state` headline + the
  mechanism sentence + the canonical keyboard-HQ delta / masked-eligible pen
  count / interval published pen max (marked REFUTED).
- `world_f0XX.jpg` (1280×798): a 3D world view (synthetic orbit camera framed
  on the body+hand centroid) showing the observed body as sparse amber dashed
  edges (uncertain), hand verts (cyan), fingertip joints (magenta), and a
  world ground grid. Same canonical banner.
- `side_by_side_f0XX.jpg` (1920×540): overlay (left) + world (right) at
  published side-by-side content geometry.
- `published_v19_{overlay,world,side_by_side}_f0XX.jpg`: extracted published
  stills, kept for the diff.
- `manifest.json`: per-frame `contact_state`, output paths, overlay render
  info (body face/vert counts, projected MANO vert count),
  `pixel_diff_vs_published` (diff fraction > 18 absdiff, mean absdiff, common
  shape) for all three panels, and `label_diff_vs_published` (published banner
  text, new banner text, whether published carried UNCERTAIN/gap/penverts,
  whether the new banner carries the canonical contact_state and drops
  UNCERTAIN, label_changed bool).

## Visible replacement of the published banner (label diff, verified)

Published banner (from `v19_published_render_report.json`):
```
title:    V19 runtime prediction hot3d_clip001850_...
subtitle: ...UNCERTAIN=not accepted contact closure
metrics:  L: gap 98.6mm, shift 0.0px, closed False | R: gap 39.2mm, shift 0.0px, closed False
```

New banner per frame (carries the canonical `contact_state`):
- f32 → `contact_state = geometry_epoch_contaminated` — "object body 8.4x too
  thick, 95.4% TRELLIS-inferred, non-watertight, uncarved -> mesh penetration
  NOT contact-admissible"
- f36 → `contact_state = full_frame_depth_leak` — "keyboard-masked+hand-
  quarantined depth refutes interval penetration (0 eligible verts)"
- f45 → `contact_state = unresolved_incoherent_evidence` — "single thumb
  vertex on a depth outlier (+5.86mm << sigma_clip), no temporally persistent
  contact patch -> demoted from candidate"

Manifest `label_diff_vs_published` for all three frames:
`label_changed=true`, `new_drops_uncertain=true`,
`new_carries_canonical_contact_state=true`, `published_has_uncertain=true`,
`published_has_gap=true`.

## Body rendered as uncertain/hatched (not solid accepted TRELLIS)

- The observed body (3221 faces, 1726 verts, AABB 0.198×0.543×0.169 m,
  non-watertight) is placed in world via the per-frame pose and projected with
  back-face culling + depth sort. It is drawn as a faint amber translucent
  fill + dashed amber edges + sparse diagonal hatch. The legend on every panel
  reads "observed contact body (UNCERTAIN, non-watertight patch; NOT solid
  TRELLIS)".
- Programmatic color check on the published vs new overlay f32: published
  solid-green-body pixels (G high, B/R low) = 4567; new = 498 (negligible — the
  solid accepted green body is gone). New amber-wireframe pixels in the
  keyboard region (right half) = 3330; cyan MANO verts = 872.
- The world view renders the same body as sparse amber dashed edges (730–979
  amber px across f32/f36/f45) plus cyan hand verts and magenta fingertips.

## Pixel diff vs published (nonzero, confirmed)

| frame | overlay diff frac | world diff frac | side_by_side diff frac |
|------:|------------------:|----------------:|-----------------------:|
| 32 | 0.120 | 0.083 | 0.333 |
| 36 | 0.120 | 0.082 | 0.333 |
| 45 | 0.134 | 0.086 | 0.343 |

(diff fraction = pixels with absdiff > 18 over the resized common shape; mean
absdiff overlay ~13, world ~22, side_by_side ~32.)

## What this is and is not

- It IS a production-shaped consumer that closes G0: the canonical
  `contact_frame_detail` table now drives visible banner labels, and the body
  provenance is rendered as uncertain/observed instead of solid/accepted. The
  diff vs published is nonzero on every panel.
- It does NOT flip any `contact_state` to contact. f32/f36/f45 stay unresolved
  with their named mechanisms; the body stays hatched/uncertain; the banner
  shows the keyboard-HQ delta (hand in front) and marks the interval
  published penetration as REFUTED. This satisfies KT-U (uncertainty
  preserved, no forced contact).
- It is a still-image consumer for the three review frames, not a full-duration
  video re-render. Full-duration video production would run this render path
  across all 150 frames inside the runtime; that is the next integration step,
  not this slice.
- The world-view camera is a synthetic orbit viewpoint framed on the scene
  centroid (not the published world camera), because the published world
  renderer's exact viewpoint is not exported as data. The overlay camera is
  the real per-frame metric camera.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
/home/yiwen/ego_annotation/.venv/bin/python scripts/render_clip001850_v19_contact_state_artifact.py
# -> /tmp/clip001850_v19_contact_state_render/{overlay,world,side_by_side}_f0{32,36,45}.jpg
#    + published_v19_*_f0{32,36,45}.jpg + manifest.json
```

Requires the canonical table and repaired body to already exist (subagents
13/11). Re-running is deterministic (exit 0).

![overlay f32 new vs published](/tmp/clip001850_v19_contact_state_render/comparison_overlay_f032_published_vs_new.jpg)

![world f32 new vs published](/tmp/clip001850_v19_contact_state_render/comparison_world_f032_published_vs_new.jpg)

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "New untracked script scripts/render_clip001850_v19_contact_state_artifact.py implements the smallest production-shaped consumer: reads /tmp/clip001850_contact_state_table/contact_frame_detail.ndjson + /tmp/clip001850_keyboard_body_repair/repaired_observed_contact_body.ply + pose/interval/visgeo state, renders overlay/world/side-by-side stills for f32/f36/f45 under /tmp/clip001850_v19_contact_state_render/. Banner visibly replaces published gap/penverts/UNCERTAIN with canonical contact_state labels (f32 geometry_epoch_contaminated, f36 full_frame_depth_leak, f45 unresolved_incoherent_evidence) - verified via manifest label_diff (label_changed=true, new_drops_uncertain=true, new_carries_canonical_contact_state=true). Body drawn as amber hatched wireframe observed patch, not solid green TRELLIS (programmatic color check: published solid-green 4567px -> new 498px; new amber wireframe 3330px in keyboard region). Manifest carries pixel diff (overlay 0.12-0.13, world 0.08-0.09, side_by_side 0.33-0.34) and label diff vs published. No scope widening: single new script, no edits to existing scripts."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Reproducible: rm -rf output + rerun exits 0 with identical diff fractions. Outputs inspected programmatically (color analysis, manifest fields). Comparison composites written. Nothing staged (git status shows script as untracked ??)."
    }
  ],
  "changedFiles": [
    "scripts/render_clip001850_v19_contact_state_artifact.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "/home/yiwen/ego_annotation/.venv/bin/python scripts/render_clip001850_v19_contact_state_artifact.py",
      "result": "passed",
      "summary": "Rendered 9 stills (overlay/world/side_by_side x f32/f36/f45) + 9 published stills + manifest.json. Exit 0. label_changed=true on all frames; overlay pixel diff frac 0.120/0.120/0.134."
    },
    {
      "command": "rm -rf /tmp/clip001850_v19_contact_state_render && <rerun>",
      "result": "passed",
      "summary": "Clean reproducibility check: identical outputs and diff fractions."
    },
    {
      "command": "programmatic color/diff analysis on rendered vs published stills",
      "result": "passed",
      "summary": "Published solid-green body 4567px -> new 498px; new amber wireframe 3330px (overlay f32 right half); world view amber 730-979px + cyan MANO. Banner green contact_state text present."
    },
    {
      "command": "git status --short scripts/render_clip001850_v19_contact_state_artifact.py",
      "result": "passed",
      "summary": "?? (untracked); nothing staged; on branch yiwen_research."
    }
  ],
  "validationOutput": [
    "f32 contact_state=geometry_epoch_contaminated; label_changed=true; drops_uncertain=true; pixel diff overlay=0.120 world=0.083 sbs=0.333",
    "f36 contact_state=full_frame_depth_leak; label_changed=true; drops_uncertain=true; pixel diff overlay=0.120 world=0.082 sbs=0.333",
    "f45 contact_state=unresolved_incoherent_evidence; label_changed=true; drops_uncertain=true; pixel diff overlay=0.134 world=0.086 sbs=0.343",
    "Body provenance: observed-only 3221 faces / 1726 verts / non-watertight, drawn as amber hatched wireframe; solid green TRELLIS body absent (4567px -> 498px).",
    "KT-U preserved: no contact_state flipped to contact; banner marks interval published penetration as REFUTED and shows keyboard-HQ delta (hand in front)."
  ],
  "residualRisks": [
    "World-view camera is a synthetic orbit viewpoint framed on the scene centroid, not the published world renderer's exported viewpoint (the published world camera is not exported as data). Overlay uses the real per-frame metric camera. A future full-duration integration should reuse the production world renderer's viewpoint for exact visual continuity.",
    "This is a still-image consumer for f32/f36/f45, not a full 150-frame video re-render. Full-duration production requires running this render path across all frames inside the runtime.",
    "Hatched body render subsamples faces/edges for performance and visual sparsity (max 4000 faces, edges stride 3); it is a visibility/provenance styling, not a metric reconstruction.",
    "Label diff is textual (banner strings), not OCR of the rasterized banner; the manifest records the exact published and new banner text."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new untracked script scripts/render_clip001850_v19_contact_state_artifact.py (~530 lines). No edits to existing scripts. Outputs go to /tmp (not the repo). Nothing staged or committed.",
  "reviewFindings": [
    "no blockers: G0 consumer gap closed for f32/f36/f45 - canonical contact_frame_detail now drives visible banner labels with nonzero pixel/label diff vs published on all three panels.",
    "note: world-view viewpoint is synthetic; acceptable for a review-frame consumer, should be reconciled with the production world renderer if promoted to full-duration."
  ],
  "manualNotes": "This closes the G0 blocker named by subagents 06/10/13/14 for the three review frames: the canonical ego.hoi contact_frame_detail table now has a production-shaped render consumer. The delivered stills visibly replace the published gap/penverts/UNCERTAIN banner with the named unresolved mechanisms (geometry_epoch_contaminated / full_frame_depth_leak / unresolved_incoherent_evidence) and draw the keyboard body as an uncertain hatched observed patch rather than a solid accepted TRELLIS body. Contact verdict is preserved as unresolved on all three frames (KT-U). Next integration step is full-duration video production across all 150 frames, which belongs in the runtime, not this slice."
}
```
