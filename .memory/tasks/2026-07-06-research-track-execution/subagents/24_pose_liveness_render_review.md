# 24 — clip001850 pose-liveness review render

Makes the inert rigid-object pose-graph defect (D5 graph inertness + D1 pose
unobserved) **visible as a review artifact**, not a status table. This
implements the "object-pose graph liveness" frontier card from subagent 21
(causal card #1): the keyboard pose graph has `nfev=1 / cost=0 / zero active
residuals`, only 13/150 frames are direct fits, 65 frames are frozen at one
identical pose, 35 are linear interpolation, and the 13 fits jump up to
166 mm / 37° frame-to-frame.

Branch `yiwen_research`. One new additive script, nothing staged, nothing
committed. CPU-only (OpenCV / PIL / trimesh / matplotlib). No model inference,
no GPU.

---

## 0. Causal card (read first)

- **Artifact defect (rendered + numeric):** the keyboard body in the existing
  world/overlay videos is drawn at a *frozen* pose on 65 frames (one identical
  translation `[0.5776,-0.0958,0.3385] m`) and *teleports* up to 166 mm on the
  13 fit frames. The pose graph reports `nfev=1, cost=0.0, residual_rms=0.0,
  nonpenetration_target_frame_count=0` — it is a pass-through, not a solve. The
  defect is currently buried in a JSON report; it is not visible to a viewer.
- **Physical variable:** the rigid object pose trajectory `T_world_object(t)`
  and its per-frame provenance — whether each frame's SE3 is a *measured* depth
  fit, a *held* nearest-neighbour fill, an *interpolated* linear fill, or an
  *ineligible* surface that failed the rigid-fit test.
- **Live mechanisms:** (M1) no seed/correspondence support → 65 missing frames
  never got a pose; (M2) degenerate visible surface → 72 ineligible frames;
  (M3) inert solver plumbing → temporal/nonpenetration factors have zero
  support so nothing smooths the 166 mm jumps or fills gaps; (M4) legitimate
  static-held gauge (steelman, excluded by the jumps).
- **Discriminating measurement (this artifact):** per-frame projected body
  placement labelled by provenance + frame-to-frame jump metric, so a viewer
  can see directly that the keyboard is frozen on most frames and teleports on
  the fit frames.
- **Intervention:** a review render that surfaces the defect visibly. It does
  not fix the trajectory (that is the next mechanism step); it makes the
  defect and its provenance auditable in the artifact the user consumes.

## 1. What was built

`scripts/render_clip001850_pose_liveness_review.py` (untracked, additive) reads
only existing CPU files and produces a review artifact under
`/tmp/clip001850_pose_liveness_review/`:

| output | what it exposes |
|---|---|
| `pose_liveness_contact_sheet.png` | 10 key-frame panels: RGB + provenance-coloured projected observed body (green=measured, red=held/frozen, blue=interpolated) + white centroid crosshair, each labelled with pose source, status, and frame-to-frame jump. Covers frozen (f0,f11), direct fit (f30,f36,f45,f46,f75,f76), interpolated (f39), held-ineligible (f90). |
| `pose_trajectory_world.png` | World-frame body-centroid trajectory (top: XY plan view coloured by provenance with jump annotations; bottom: per-frame dtrans bar chart). Exposes the frozen plateau, the linear-interp segments, and the two teleport jumps. |
| `jump_pair_f45_f46.png` | Side-by-side consecutive frames: f45→f46 = 57.2 mm / 21.5° teleport. |
| `jump_pair_f75_f76.png` | Side-by-side consecutive frames: f75→f76 = 166.4 mm / 36.9° teleport. |
| `manifest.json` | Per-frame provenance + jump metrics for all 150 frames; optimizer inertness summary; provenance counts; review-frame states. |

### Provenance classification

The pose graph carries two independent fields per frame:
`pose_measurement_status` (why a frame is or is not a direct fit) and
`temporal_pose_graph.pose_source` (how the frame's SE3 was filled). The
render classifies by `pose_source` and shows `status` in the banner:

| pose_source | tag | colour | count |
|---|---|---|---|
| `direct_visible_pose_observation_corrected` | MEASURED (direct fit) | green | 13 |
| `nearest_visible_pose_hold` | HELD (frozen) | red | 102 |
| `interpolated_between_visible_pose_observations` | INTERPOLATED | blue | 35 |

Cross-tabulated against status: the 13 direct fits are all `fit_to_visible_depth_samples`;
the 35 interpolated are all `missing_initial_graph_pose`; the 72
`visible_surface_ineligible_for_rigid_pose_fit` frames are all held; the other
30 held frames are `missing_initial_graph_pose`.

## 2. Evidence the artifact surfaces

### Optimizer inertness (from the pose graph report)

- `optimizer.nfev = 1`, `cost = 0.0`, `residual_rms_before = residual_rms_after = 0.0`
- `nonpenetration_target_frame_count = 0`; `graph_frame_count = 13`
- `graph_frames = [30,31,32,33,34,35,36,45,46,60,75,76,77]`
- Verdict: inert — zero active residuals; temporal/nonpenetration factors have
  no support; the 13 "fits" are independent per-frame depth fits, not a graph
  solve, so nothing smooths the jumps or fills gaps.

### Frozen defect (f0 = f11, identical pose)

- f0 translation `[0.5776,-0.0958,0.3385]` == f11 translation (identical).
- The 65 `missing_initial_graph_pose` frames are all frozen at the nearest fit
  frame's pose; f0–f29 share f30's pose.
- Per-frame jump over all 150 frames: median **0.0 mm**, p90 **8.1 mm**, max
  **166.4 mm**. The 0.0 mm median is the frozen plateau.

### Jump defect

- f46: **57.2 mm / 21.5°** (direct-fit refit; mask re-acquire / ego-motion).
- f76: **166.4 mm / 36.9°** (the largest teleport).
- Both are direct fits (`fit_to_visible_depth_samples`), so they are drawn as
  "measured" — the label honestly says MEASURED while the jump metric flags the
  teleport. This is the core defect: the only frames that ARE measured are the
  ones that jump, because there is no temporal factor opposing them.

### Review frames (from manifest)

| frame | provenance | status | dtrans | drot | jump |
|------:|---|---|---:|---:|:--:|
| 0 | held (frozen) | missing_initial_graph_pose | 0.0 mm | 0.0° | |
| 11 | held (frozen) | missing_initial_graph_pose | 0.0 mm | 0.0° | |
| 30 | measured | fit_to_visible_depth_samples | 0.0 mm | 0.0° | |
| 36 | measured | fit_to_visible_depth_samples | 8.6 mm | 1.0° | |
| 39 | interpolated | missing_initial_graph_pose | 4.5 mm | 1.9° | |
| 45 | measured | fit_to_visible_depth_samples | 4.5 mm | 1.9° | |
| 46 | measured | fit_to_visible_depth_samples | 57.2 mm | 21.5° | **JUMP** |
| 75 | measured | fit_to_visible_depth_samples | 8.1 mm | 0.4° | |
| 76 | measured | fit_to_visible_depth_samples | 166.4 mm | 36.9° | **JUMP** |
| 90 | held (frozen) | visible_surface_ineligible | 0.0 mm | 0.0° | |

## 3. Scope discipline

- **One new additive script**, `scripts/render_clip001850_pose_liveness_review.py`.
  Zero edits to any existing script.
- **No staging, no commits.** `git status --short` shows the script as `??`
  (untracked); `git diff --cached --name-only` is empty.
- **No heavy inference / GPU / model.** CPU-only: reads JSON/PLY/JPG and uses
  OpenCV/PIL/trimesh/matplotlib. No torch, no CUDA, no network.
- **Does NOT claim contact.** `manifest.scope.contact_claimed = false`. The
  artifact labels pose provenance and body placement only.
- Outputs go to `/tmp/clip001850_pose_liveness_review/` (not the repo).

## 4. Honest limits

- This artifact makes the defect *visible*; it does not repair the trajectory.
  The next mechanism step (card 1 intervention) is to repair the
  variable-to-objective wiring / build temporal correspondences so the
  temporal factor opposes the jumps and fills the 137 non-fit frames — then
  re-render this review to show a smoothed trajectory.
- The body overlay uses the 3221-face repaired observed patch (open,
  non-watertight). It is drawn for placement/provenance, not as a contact
  surface. Backface-culled faces with `z > 0.03 m` only.
- f39 is interpolated AND `missing_initial_graph_pose` status — the status
  field is orthogonal to the source field; the panel shows both honestly.
- The world-trajectory plot uses the body centroid (observable), not a
  completed-mesh surface.

## Reproduce

```bash
cd /home/yiwen/ego_annotation
/home/yiwen/ego_annotation/.venv/bin/python scripts/render_clip001850_pose_liveness_review.py
# -> /tmp/clip001850_pose_liveness_review/{contact_sheet,trajectory,jump_pairs,manifest}
```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Implemented the requested pose-liveness review artifact as one new additive script scripts/render_clip001850_pose_liveness_review.py (untracked, not staged). Consumes only existing CPU files: run-root rigid pose graph report (per-frame SE3 + provenance), raw RGB frames, per-frame visible geometry/cameras (T_world_camera_metric + intrinsics), and the durable repaired observed keyboard body (3221-face patch). Produces a contact sheet of key frames (f0/f11 frozen, f30/f36 direct fit, f39 interpolated, f45/f46 jump, f75/f76 jump, f90 held-ineligible), a world-trajectory plot, two jump-pair strips, and a manifest with per-frame states + jump metrics + optimizer inertness summary. The contact sheet visibly labels measured vs held/interpolated/ineligible provenance (green/red/blue colouring + banner text) and shows object body placement via provenance-coloured projected faces + centroid crosshair. Does NOT claim contact (contact_claimed=false). No GPU, no model inference. Nothing staged or committed."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Every load-bearing claim verified against on-disk state: optimizer inert (nfev=1/cost=0/residual_rms=0/nonpenetration_target=0, graph_frame_count=13); provenance counts direct=13/held=102/interpolated=35 (missing_initial_graph_pose=65, ineligible=72, fit=13); f0 translation==f11 translation (frozen defect); jump metrics max=166.4mm@f76, 57.2mm@f46, median=0.0mm, only 2 frames>=30mm; all 4 provenance classes present in review (direct f30/f36/f45/f46/f75/f76, held-missing f0/f11, interpolated f39, held-ineligible f90); contact sheet pixel check confirms green/red/blue provenance colours + white centroid markers render; contact_claimed=false, gpu_used=false, model_inference=false; git diff --cached empty, script untracked. Independent reviewer can rerun the script and inspect the 5 output files + manifest.json."
    }
  ],
  "changedFiles": [
    "scripts/render_clip001850_pose_liveness_review.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "/home/yiwen/ego_annotation/.venv/bin/python scripts/render_clip001850_pose_liveness_review.py",
      "result": "passed",
      "summary": "Wrote /tmp/clip001850_pose_liveness_review/{pose_liveness_contact_sheet.png 2.96MB, pose_trajectory_world.png 94KB, jump_pair_f45_f46.png 610KB, jump_pair_f75_f76.png 624KB, manifest.json 32KB}. Contact sheet 1990x1440 (10 panels, 4 rows x 3 cols + legend). optimizer nfev=1 cost=0.0 direct=13 frozen=65 max_jump=166.4mm."
    },
    {
      "command": "python3 -c manifest inspection (review_frames, provenance_counts, optimizer_inertness, jump_metrics)",
      "result": "passed",
      "summary": "Verified: optimizer nfev=1/cost=0/residual_rms=0/np_target=0/graph_frames=[30-36,45,46,60,75,76,77]; provenance direct=13/held=102/interpolated=35; f0==f11 translation; max_jump 166.4mm@f76, 57.2mm@f46, median=0.0mm; all 4 provenance classes in review frames; contact_claimed=false."
    },
    {
      "command": "python3 -c contact sheet pixel colour check",
      "result": "passed",
      "summary": "Contact sheet contains green (direct) 18049px, red (held) 8987px, blue (interp) 2056px, white centroid 9260px. All provenance colours render."
    },
    {
      "command": "git status --short scripts/render_clip001850_pose_liveness_review.py && git diff --cached --name-only",
      "result": "passed",
      "summary": "Script untracked (??); nothing staged (git diff --cached empty). On branch yiwen_research."
    }
  ],
  "validationOutput": [
    "Artifact exposes the inert pose-graph defect visibly: contact sheet shows frozen body (f0/f11 red held, identical placement), direct fits (f30/f36 green), interpolated (f39 blue), held-ineligible (f90 red), and jumps (f46 57mm/21.5deg, f76 166mm/37deg flagged JUMP).",
    "Trajectory plot shows world-frame centroid frozen plateau (median 0.0mm), linear-interp segments, and two teleport jumps annotated.",
    "Jump-pair strips make f45->f46 (57mm) and f75->f76 (166mm) teleports visible as side-by-side consecutive frames.",
    "Manifest carries optimizer inertness (nfev=1/cost=0/np_target=0), provenance counts (direct 13/held 102/interp 35), jump summary (max 166.4mm@f76, 2 frames>=30mm), and per-frame states for all 150 frames.",
    "Does NOT claim contact: manifest.scope.contact_claimed=false; no contact state, gap, or signed-distance text anywhere in the artifact.",
    "CPU-only: no torch/cuda/gpu/model-inference imports; gpu_used=false, model_inference=false.",
    "Nothing staged or committed; script is untracked (??); outputs under /tmp."
  ],
  "residualRisks": [
    "This artifact makes the defect visible; it does not repair the trajectory. The next mechanism step is to repair the pose-graph variable-to-objective wiring / build temporal correspondences so the temporal factor opposes the jumps and fills the 137 non-fit frames, then re-render this review.",
    "The body overlay uses the open non-watertight 3221-face observed patch for placement/provenance only; it is not a contact surface.",
    "f39 is interpolated AND missing_initial_graph_pose status (status and source are orthogonal fields); the panel shows both honestly.",
    "The reviewer cannot see the rendered images through this report text; they must open the PNGs under /tmp/clip001850_pose_liveness_review/ to consume the artifact as a viewer."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new untracked script scripts/render_clip001850_pose_liveness_review.py (~330 lines). No edits to existing scripts. Outputs go to /tmp (not the repo). Nothing staged or committed.",
  "reviewFindings": [
    "no blockers: the artifact visibly labels measured vs held/interpolated/ineligible pose provenance (green/red/blue + banner text), shows object body placement (projected faces + centroid), exposes frozen (f0/f11), direct fit (f30/f36), interpolation (f39), and jumps (f46 57mm, f76 166mm), includes a manifest with frame states and jump metrics, and does not claim contact.",
    "note: the artifact surfaces the defect but does not fix it; the next step is the trajectory-repair intervention (card 1), after which this review should be re-rendered to show a smoothed trajectory."
  ],
  "manualNotes": "The pose graph is inert (nfev=1, cost=0, zero active residuals) and the keyboard is unlocalized on 137/150 frames (65 frozen at one identical pose, 72 ineligible, 35 interpolated, only 13 direct fits that jump up to 166mm). This review makes that defect visible: open /tmp/clip001850_pose_liveness_review/pose_liveness_contact_sheet.png to see the frozen plateau (red held frames f0/f11/f90 at identical placement), the measured fits (green f30/f36), the interpolated fill (blue f39), and the teleport jumps (f46 57mm/21.5deg, f76 166mm/37deg, flagged JUMP). The trajectory plot and jump-pair strips make the teleports visceral. The manifest records per-frame provenance + jump metrics for all 150 frames. The artifact does not claim contact and runs CPU-only. Open the PNGs to consume the artifact as a viewer."
}
