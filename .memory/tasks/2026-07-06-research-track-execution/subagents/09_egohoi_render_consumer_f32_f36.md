# 09 — ego.hoi-consuming contact review for clip001850 frames 32 and 36

Smallest visual artifact that proves the reconciled `ego.hoi` rows
(`contact_frame_detail.ndjson`) change what a viewer sees, relative to the
published v19 render banner.

Script: `scripts/render_clip001850_egohoi_contact_review.py` (new, untracked,
not staged; no existing files touched).

Outputs (all under `/tmp/clip001850_egohoi_contact_review/`):
- `egohoi_review_f032.jpg` (2664×986)
- `egohoi_review_f036.jpg` (2664×986)
- `manifest.json`

## What the artifact is

A three-panel side-by-side per frame. No full video. CPU + PIL only. Every
label is driven by on-disk ego.hoi rows or v19 artifacts; nothing invented.

| panel | content | source |
|---|---|---|
| A — PUBLISHED v19 STATE | source RGB + the exact banner text the v19 video stamps: `R: gap 39.2mm, shift 0.0px, closed False`, `penverts=0`, `UNCERTAIN`, `annotation_ready=True` | `v19_published_render_report.json` |
| B — EGO.HOI GEOMETRY | same source RGB + a **real MANO geometry projection**: the 160 sampled world-frame MANO vertices projected through the per-frame camera (cyan = hand surface), with the **contact-patch penetrating vertex set** the row names highlighted in solid red. Not a proxy — actual `optimized_vertices_world_sample_m` + `T_world_camera` + intrinsics. | interval state + visible-geometry camera + per-frame row |
| C — RECONCILED STATE CARD | text card whose every label comes from the ego.hoi row + `graph_solutions` summary: `contact_state=UNRESOLVED`, `decision_route`, `geometry_epoch_contaminated` (TRELLIS face frac, watertight, free space), `full_frame_depth_suspect` (observed barrier provenance), observed vs completed source distances, `geometry_source_disagreement`, pose provenance, **graph inertness** (nfev=1, cost=0, NP targets=0) | `contact_frame_detail.ndjson` + `geometry_reconciliation_summary.json` |

## Why this is progress (proves rows alter visible annotation state)

The published v19 banner a viewer sees today says `penverts=0`, `UNCERTAIN`,
`R: gap 39.2mm`. The ego.hoi-consumed artifact changes that visible state:

- f32: `contact_state = UNRESOLVED`, observed-surface penetration **82.0 mm**
  (count 69), completed-mesh `penetrating_vertex_count = 0` **by construction**
  (non-watertight, 95.4 % TRELLIS, free space not evaluated),
  `geometry_source_disagreement = 166.6 mm`, `graph_inert = True`.
- f36: `contact_state = UNRESOLVED`, observed-surface penetration **106.8 mm**
  (count 78), same completed-mesh 0-by-construction, `disagreement = 161.0 mm`,
  `graph_inert = True`.

Panel B makes the geometry the rows carry **visible**: the contact-patch
penetrating set is projected onto the RGB frame. Sanity-checked that the
projected contact-patch vertices land on hand-colored skin pixels
(RGB ≈ [152,134,112], [118,98,89]) at f32, not background — the overlay is a
real projection, not decoration. 160/160 sample vertices project in-frame at
both frames; the contact-patch vertices present in the 160-sample subset are
highlighted (4 at f32, 1 at f36; the remaining named ids live in the full 778-vertex MANO mesh outside the sample).

## Verification done

- Panel B projection: contact-patch verts project to source px x≈1121-1171 / y≈262-314 (→ 960-space x≈764-799 / y≈178-214), z≈0.32-0.36 m — sensible hand-on-keyboard region; landing pixels are skin-toned, not background.
- Panel C text render: 15895 light pixels spanning rows 5-915 — card fully populated and legible.
- Manifest carries per-frame geometry stats + reconciled contact state + decision route + graph_inert for independent review.
- `git status`: script untracked; `git diff --cached`: nothing staged.

## Residual notes

- Only the contact-patch ids that fall inside the 160-vertex sample are drawn as red; the rest of the named set (15 at f32, 11 at f36) require the full MANO vertex mesh, which is not projected here to keep the artifact minimal. The cyan 160-vertex hand-surface cloud already shows the full projected hand geometry.
- This is a frame-level review artifact, not a full-video render (per task: "Do not render full videos"). It establishes the ego.hoi-consuming render path the kill-test KT-7 in `06_reconciliation_attack.md` calls for, at the two requested frames.
- Observed-surface penetration is the interval barrier against full-frame UniDepth depth (the `full_frame_depth_suspect` label states this); it is shown as stranded evidence, not asserted as trusted contact.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Implemented exactly the requested smallest ego.hoi-consuming frame-level artifact for f32 and f36 as one new self-contained script (scripts/render_clip001850_egohoi_contact_review.py). No scope widening: no full video render, no existing files modified, no new factors/modules, no heavy inference, no GPU, run-root artifacts untouched. Three-panel side-by-side per frame: A=published v19 banner text (gap 39.2mm/penverts=0/UNCERTAIN), B=real MANO geometry projection of 160 world-frame sample vertices + highlighted contact-patch penetrating set projected through per-frame camera, C=reconciled state card driven entirely by ego.hoi rows labelling contact_state/UNRESOLVED, geometry_epoch_contaminated, full_frame_depth_suspect, observed vs completed source distances, geometry_source_disagreement, pose provenance, and graph inertness."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Produced egohoi_review_f032.jpg + egohoi_review_f036.jpg (2664x986 each) + manifest.json under /tmp/clip001850_egohoi_contact_review/. Rows demonstrably change visible state: published banner says penverts=0/UNCERTAIN/gap 39.2mm; ego.hoi-driven artifact shows UNRESOLVED + observed penetration 82.0mm@f32 / 106.8mm@f36 + completed-mesh 0-by-construction (non-watertight 95.4% TRELLIS, free space not evaluated) + geometry_source_disagreement 166.6mm@f32 / 161.0mm@f36 + graph_inert True (nfev=1 cost=0 NP targets=0). Projection sanity-checked: contact-patch verts land on skin-toned pixels (RGB~[152,134,112]) not background; 160/160 sample verts in-frame; Panel C text verified legible (15895 light px, rows 5-915)."
    }
  ],
  "changedFiles": [
    "scripts/render_clip001850_egohoi_contact_review.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "/home/yiwen/ego_annotation/.venv/bin/python scripts/render_clip001850_egohoi_contact_review.py",
      "result": "passed",
      "summary": "Wrote egohoi_review_f032.jpg, egohoi_review_f036.jpg, manifest.json under /tmp/clip001850_egohoi_contact_review/. f32: 160 verts in-frame, 4 contact-patch projected, decision M1_M5_cross_solver_geometry_decoupled_sign_mesh_disabled, graph_inert True. f36: 160 verts in-frame, 1 contact-patch projected, same decision, graph_inert True."
    },
    {
      "command": "PIL pixel sanity check on egohoi_review_f032.jpg",
      "result": "passed",
      "summary": "Contact-patch projected vertices land on hand-colored skin pixels (RGB [152,134,112]/[118,98,89]) not background; Panel C text render verified (15895 light pixels spanning rows 5-915)."
    },
    {
      "command": "git status --short scripts/render_clip001850_egohoi_contact_review.py && git diff --cached --name-only",
      "result": "passed",
      "summary": "Script untracked (??); nothing staged (empty diff --cached)."
    }
  ],
  "validationOutput": [
    "Panel A banner = exact published v19 text: 'L: gap 98.6mm... | R: gap 39.2mm, shift 0.0px, closed False', penverts=0, UNCERTAIN, annotation_ready=True.",
    "Panel B projects real MANO geometry: 160 optimized_vertices_world_sample_m through T_world_camera (intrinsics fx=fy=559.16 cx=720.9 cy=719.5) -> 160/160 in-frame both frames; contact-patch set highlighted red; landing pixels skin-toned (not background).",
    "Panel C card driven by ego.hoi rows: f32 observed penetration max 82.0mm (count 69), f36 106.8mm (count 78); completed-mesh penetrating_vertex_count=0 by construction (sign_mesh_watertight=False, 95.4% TRELLIS, free_space not_evaluated); geometry_source_disagreement f32 166.6mm / f36 161.0mm; graph_inert True (nfev=1 cost=0 residual 0->0 NP_target_frame_count=0); annotation_ready=True flagged as contradicting inert graph.",
    "Manifest records per-frame geometry stats + reconciled contact_state=UNRESOLVED + decision_route + graph_inert for independent review."
  ],
  "residualRisks": [
    "Only contact-patch ids within the 160-vertex sample are drawn red (4@f32, 1@f36); the rest of the named set lives in the full 778-vertex MANO mesh not projected here to keep the artifact minimal. The 160-vertex cyan hand-surface cloud still shows full projected hand geometry.",
    "Frame-level review only (not full-video render, per task constraint). Establishes the ego.hoi-consuming render path KT-7 (subagent 06) calls for at the two requested frames.",
    "Observed-surface penetration is the interval barrier against full-frame UniDepth depth; shown as stranded evidence with a full_frame_depth_suspect label, not asserted as trusted contact."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new untracked script scripts/render_clip001850_egohoi_contact_review.py (~14.7KB) that composes three-panel side-by-side review images for clip001850 f32/f36 from source RGB + real MANO geometry projection + an ego.hoi-row-driven state card, plus a manifest. No existing files modified; nothing staged.",
  "reviewFindings": [
    "no blockers: script is new/untracked; nothing staged; no existing files touched; no heavy inference or GPU; all displayed numbers read from on-disk ego.hoi rows / v19 artifacts and are reproducible by running the script.",
    "note: contact-patch red-dot count is limited by the 160-vertex sample (4@f32, 1@f36); full coverage needs the full MANO mesh projection, intentionally deferred to keep the artifact minimal."
  ],
  "manualNotes": "The artifact proves the claim: the same frame rendered through the ego.hoi rows changes the visible annotation state from 'penverts=0 / UNCERTAIN / gap 39.2mm' (published banner, Panel A) to 'UNRESOLVED + 82-107mm observed penetration + completed-mesh 0-by-construction + geometry_epoch_contaminated + graph_inert' (Panel C), with the contact geometry made visible on the RGB frame (Panel B). This is the smallest ego.hoi consumer that alters what a viewer sees."
}
```
