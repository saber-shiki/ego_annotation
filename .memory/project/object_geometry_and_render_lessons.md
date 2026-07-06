# Object geometry + presentation-render lessons (2026-07-04 demo sprint)

Provenance: demo-pack OPS + `review/track{A,F1,F2,F3}_*.md`, tomato investigation, renderer
`demo_pack_20260704/renderer/render_demo_presentation.py` (reference implementation of all
mechanisms below).

## Object completion

- **TRELLIS conditioning-frame selection is a first-class pipeline decision.** The same
  object completed from a mid-manipulation frame (tomato f929, mid-peel) grows artifacts of
  that moment (peel-flap horn) into the rigid body used for all frames; a clean
  maximum-visibility intact-state anchor (v19 f270) yields a round 8cm body. Anchor choice
  changed silhouette IoU 0.375→0.553 with the SAME pose trajectory.
- **Fusion can absorb scene slabs**: v18 f806/f929 "completed" meshes carry counter-surface
  slabs + debris. Silhouette-support filtering against per-frame masks is the needed guard.
- **Always render canonical shaded views of a mesh before using it** (some exports are
  degenerate: the f806 TRELLIS export is a flat slab; v17 BundleSDF is a fragmented shell).
- **Registering mesh generations**: similarity ICP degenerates (shrinks the clean body into
  the artifact-bearing one, s=0.496). Use fixed-scale rigid trimmed ICP over a scale sweep,
  select scale by silhouette-IoU against masks (evidence-based, buyer-explainable).

## Presentation renderer mechanisms (all implemented + verified)

- **Image-space object claims come from per-frame masks, not fused-mesh projection.** Fused
  bodies flood regions the object never covers (worst for deformables). Mask fills are
  tight by construction; when the mask is missing, draw the mesh silhouette contour only
  (a thin outline misleads far less than a wrong fill). Mask dir resolves from the state's
  `source_mask_path` with frame substitution.
- **Orthographic world view**: camera-space positive-depth culling is INVALID there (culled
  hand fills while contours survived — "ghost hands"); anchor the view center on hands
  (object centroids drag the midpoint); grid must be content-anchored, never world-origin.
- **Gravity**: SLAM/HaWoR worlds are not gravity-aligned (trash camera "height" span 4.4m =
  walking leakage). Estimator ladder with measured outcomes: mean head-up (posture-biased),
  global support-plane RANSAC among low content points (works on stable clips: tomato 100%
  camera-above-hands), station segmentation (collapses when camera translation is jittery),
  per-frame low-pass head-up (cannot separate long bends from drift). Drift-bent worlds are
  NOT fully rotation-correctable (measured 72% ceiling with per-frame instant head-up) —
  the real fix is upstream (gravity/IMU or per-frame floor tracking in SLAM).
- **Adaptive shading by measured mesh property**: lambert-shade meshes with median dihedral
  angle <15° (smooth completed bodies, tomato 0.8°); flat-fill rough poisson blobs
  (23–31°) — shading turns cratered geometry into visual noise. Vertex-normal smoothing
  helps but cannot fix cratered geometry.
- **Hand display**: skeleton (21-joint) + shaded mesh reads far better than mesh alone;
  per-frame confidence drives alpha; 5-frame median kills flicker; provenance of the
  CURRENT hand layer outranks stale visibility labels.
- **Concurrent renderer editing across agents works** with additive-only contracts (new
  flag + new helpers, no changes to existing paths, py_compile after each edit) — used
  successfully by 5 agents in one day on one file.

## V19 HOI render-consumer invariant

- **Per-frame HOI/contact state must be consumed by the Stage-1 renderer that draws object/hand/contact pixels.** `publish_v19_render_artifact.py` stamps one static banner on every frame and cannot express frame-varying states such as f32 `geometry_epoch_contaminated`, f36 `full_frame_depth_leak`, and f45 `unresolved_incoherent_evidence`. A publish-banner edit or selected-frame still is QC only; a real consumer changes the full-duration mp4/backing manifest so the drawn body hash, contact-state label, and invalid-metric suppression are state-derived per frame.
- **Do not leave raw TRELLIS bodies in pixels under repaired labels.** For clip001850 the false state was baked by a Stage-1 render of the raw 90,892-face TRELLIS keyboard body plus zero-by-construction `penverts=0`. The repair uses the 3221-face observed open body as an amber/hatched uncertain patch and removes the false `gap 39.2mm`, `penverts=0`, and bare `UNCERTAIN` strings from the replacement render. Future geometry-source repairs need the same body-replacement proof: consumed mesh hash changes away from TRELLIS, the original solid body disappears, and uncovered frames carry explicit unresolved defaults rather than fabricated contact.

## Demo-vs-pipeline boundary (user policy, 2026-07-04)

Demo may use ad-hoc tuning, cherry-picking, per-clip overfitting — disclosed in metric
definitions and reproducible. The full pipeline may not: anything per-clip must become
GT-free self-calibration, and every heuristic here must either trace to a measured physical
mechanism or be replaced. The self-consistency metric family
(`self_consistency_metrics.md`) is the designated bridge: demo QC today, pipeline
confidence/re-estimation triggers tomorrow.
- Visible surfel renders are admissible as observation-only substrate artifacts when they draw per-frame prediction-side surfel samples as points/splats, avoid triangulated/shaded faces, avoid cross-frame accumulation, and label camera-relative panels as observations rather than object pose. NPZ faces in visible-surfel compatibility archives are loader padding only and must never drive rendering, contact, pose, occlusion, nonpenetration, normals, or surface continuity. A surfel render can improve the delivered visual substrate while leaving pose/contact blockers unresolved.
