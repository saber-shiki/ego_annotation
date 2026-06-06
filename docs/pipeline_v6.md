# Pipeline V6: Robust Sparse Correspondence Factors

## Starting State

V5 has two accepted wild-rice geometry streams:

- completed 31-frame active-stem archive, frames 2520 to 2550, with measured/completed state labels and stakeholder-readable render videos;
- repaired six-frame active-stem archive, frames 2532 to 2537, created from a SAM2 seed track and verified by z-buffer, mesh-surface contact, selected-contact SDF, full-hand SDF, and visual render QC.

V5 also falsified three dense temporal object-map hypotheses:

- a rigid canonical mesh over the completed 31-frame sequence;
- a shared-topology dynamic mesh over the observable 2525 to 2529 window;
- dense transport edges over both the observable window and the repaired 2532 to 2537 window.

The repaired sequence has strong per-frame object geometry and contact evidence, while the dense material map remains unobservable from nearest-neighbor geometry alone. V6 therefore moves temporal object reasoning from dense correspondence to sparse learned correspondence factors.

## Representation

V6 keeps the V5 repaired per-frame mesh archive as delivered geometry. CoTracker points become auxiliary graph observations:

- each track has frame-local image coordinates, visibility, mask support, and lifted world coordinates;
- each accepted world point is attached to the nearest repaired mesh surface in each visible frame;
- each neighboring frame pair gets a candidate SE3 factor only from tracks visible in both frames;
- each factor stores its source tracks, robust fit residuals, inlier set, and acceptance state.

This representation keeps visual case variation out of hand-written logic. The point tracks come from a learned video tracker, masks come from the repaired SAM2 object track, and the graph consumes masks, depths, mesh attachments, and residuals through one category-agnostic path.

## Objective

For a neighboring frame pair with accepted track positions `x_i` and `y_i`, V6 estimates a robust rigid transform:

```text
min_R,t sum_i rho(||x_i R + t - y_i||_2)
```

where `R` is a proper rotation, `t` is translation in meters, and `rho` is the Huber loss implemented by iterative reweighted least squares. Each IRLS step solves a weighted Kabsch problem, recomputes point residuals, and updates weights by:

```text
w_i = min(1, delta / max(residual_i, eps))
```

The factor is ready only when enough inlier tracks remain and the inlier p95 residual stays under the chosen tolerance. The current strict settings are:

- at least 12 pair tracks;
- at least 12 inlier tracks;
- Huber delta 10 mm;
- inlier residual threshold 12 mm;
- accepted inlier p95 threshold 10 mm.

The graph can later use ready factors as sparse temporal priors on object pose or local deformation. Rejected factors stay in the report as evidence about occlusion, object bending, bad tracks, or changing visible support.

## First Diagnostic

Input artifacts:

- CoTracker archive: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/cotracker_object_tracks_v5.npz`
- mesh-anchored sparse edges: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_sparse_edges_midquery2535_2532_2537/cotracker_sparse_correspondence_edges_v5.json`

Command:

```bash
.venv/bin/python scripts/fit_cotracker_pairwise_rigid_factors_v6.py \
  --cotracker-npz /data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/cotracker_object_tracks_v5.npz \
  --sparse-edges-json /data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_sparse_edges_midquery2535_2532_2537/cotracker_sparse_correspondence_edges_v5.json \
  --output-json /data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_pairwise_rigid_factors_2532_2537/qc_cotracker_pairwise_rigid_factors_v6.json \
  --min-pair-tracks 12 \
  --min-inlier-tracks 12 \
  --huber-delta-m 0.010 \
  --max-inlier-residual-m 0.012 \
  --accept-inlier-p95-m 0.010 \
  --irls-iterations 8
```

Report:

- `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_pairwise_rigid_factors_2532_2537/qc_cotracker_pairwise_rigid_factors_v6.json`

Result summary:

- usable tracks: 69
- neighboring frame pairs: 5
- rigid-factor-ready pairs: 2
- ready-pair inlier residual median: 3.36 mm
- ready-pair inlier residual p95: 8.27 mm
- all clipped inlier residual p95 across every pair: 10.06 mm

Pair results:

| Pair | Tracks | Inliers | Ready | Median residual | P95 residual | Inlier p95 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 2532 to 2533 | 31 | 25 | no | 8.56 mm | 27.77 mm | 11.53 mm |
| 2533 to 2534 | 46 | 39 | no | 6.91 mm | 19.36 mm | 11.65 mm |
| 2534 to 2535 | 66 | 61 | no | 5.38 mm | 13.33 mm | 10.06 mm |
| 2535 to 2536 | 69 | 69 | yes | 2.91 mm | 6.53 mm | 6.53 mm |
| 2536 to 2537 | 69 | 67 | yes | 3.98 mm | 11.42 mm | 8.50 mm |

## Interpretation

The learned point tracks produce useful sparse graph factors on the late repaired interval, especially 2535 to 2537. Earlier pairs have enough visible tracks, but their tail residuals exceed the 10 mm inlier-p95 tolerance. Those early pairs should enter the next graph as weak or rejected evidence, because a hard temporal material constraint would distort the accepted per-frame mesh geometry.

V6 has therefore established the first pair-local temporal correspondence factors with millimeter-scale residuals on part of the repaired window. The delivered object mesh remains the V5 repaired per-frame mesh archive until a graph that uses these factors is solved and then replayed through z-buffer, contact, selected-contact SDF, full-hand SDF, and visual QC.

## Next Implementation

The next V6 graph should add these pair-local factors without changing the accepted geometry by default:

1. load V5 repaired meshes, MANO vertices, contact rows, CoTracker tracks, and robust pair factors;
2. optimize per-frame object pose and optional low-dimensional local deformation, weighted by ready-pair factors only where the factor report marks them ready;
3. preserve the per-frame z-buffer residual as a hard delivery check after solve;
4. recompute mesh-surface contact, selected-contact SDF, and full-hand SDF on the solved archive;
5. render only graph states that pass the same visual and metric QC used by V4/V5.

The first solve should target frames 2535 to 2537, because both neighboring factors pass the strict residual criterion. Frames 2532 to 2535 need weaker correspondence handling or additional perception evidence before they can support temporal smoothing.

## Transport Replay Test

The first graph-use test transports the accepted source-frame mesh through each ready CoTracker SE3 factor and indexes the transported mesh by the target frame. It then compares the transported surface to the accepted target-frame mesh before any image replay.

Artifacts:

- transported mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_transport_ready_pairs_2535_2537/transported_ready_pair_meshes_world.npz`
- transport residual report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_transport_ready_pairs_2535_2537/qc_transport_ready_pair_meshes_v6.json`
- z-buffer replay report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_transport_ready_pairs_zbuffer_qc_2536_2537/qc_mesh_zbuffer_projection_v3.json`

Surface transport succeeds on the common visible surface:

| Pair | Bidirectional median | Bidirectional p95 |
| --- | ---: | ---: |
| 2535 to 2536 | 0.85 mm | 3.14 mm |
| 2536 to 2537 | 1.69 mm | 5.34 mm |

Target-frame z-buffer replay gives low depth residual but poor full-silhouette agreement:

| Target frame | Silhouette IoU | Visible inside mask | Z-buffer p95 |
| --- | ---: | ---: | ---: |
| 2536 | 0.710 | 0.939 | 6.45 mm |
| 2537 | 0.629 | 0.912 | 9.54 mm |

This is the expected distinction between material-patch tracking and full object-mask propagation. The ready CoTracker factors are valid sparse temporal factors for a stable common surface region. They are not a full propagated mesh annotation, because the visible support changes enough that the transported source mesh misses or overdraws target-frame silhouette regions. V6 should use these factors for local pose/deformation regularization and missing-patch support, then keep measured target masks/depth as the authority for delivered mesh coverage.

## Wider Ambiguity-Bridge Test

The wider CoTracker run uses the completed V4 sequence from 2532 to 2550 and queries the same clean source frame 2535. This tests whether learned sparse correspondence can carry object evidence from the repaired contact interval into later ambiguous or completed frames.

Artifacts:

- wide CoTracker run: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_midquery2535_2532_2550/`
- wide mesh-anchored sparse edges: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_sparse_edges_midquery2535_2532_2550/cotracker_sparse_correspondence_edges_v6.json`
- wide pairwise factor report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_pairwise_rigid_factors_meshanchored_2532_2550/qc_cotracker_pairwise_rigid_factors_v6.json`
- wide transport residual report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_transport_ready_pairs_2534_2538/qc_transport_ready_pair_meshes_v6.json`
- wide transported-mesh z-buffer replay: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_transport_ready_pairs_zbuffer_qc_2535_2538/qc_mesh_zbuffer_projection_v3.json`

The tracker does not produce long-range all-frame tracks. Accepted track count is strong from 2533 to 2537, drops to 39 at 2538, and falls below the 12-track factor threshold after 2541. After mesh-surface anchoring, 88 tracks remain usable and 421 neighboring-frame edges survive.

Mesh-anchored pairwise factor readiness:

| Pair | Tracks | Inliers | Ready | Inlier p95 |
| --- | ---: | ---: | --- | ---: |
| 2532 to 2533 | 37 | 24 | no | 11.00 mm |
| 2533 to 2534 | 72 | 53 | no | 10.48 mm |
| 2534 to 2535 | 85 | 79 | yes | 9.59 mm |
| 2535 to 2536 | 87 | 83 | yes | 6.90 mm |
| 2536 to 2537 | 74 | 70 | yes | 7.64 mm |
| 2537 to 2538 | 32 | 29 | yes | 9.67 mm |

The new useful fact is the 2537 to 2538 bridge: visual inspection of the 2538 overlay shows retained tracks on the active stem mask, and the pair passes the strict residual criterion. This extends sparse temporal evidence into the first later ambiguous measured frame.

Transport replay still rejects the transported source meshes as full target annotations. The four ready-pair transported surfaces have bidirectional surface medians between 1.32 and 1.99 mm, and p95 residuals between 4.89 and 8.29 mm. However, full z-buffer replay over target frames 2535 to 2538 gives median silhouette IoU 0.681 and median z-buffer p95 20.46 mm. Frame 2538 has IoU 0.701, visible-inside-mask 0.856, and z-buffer p95 21.55 mm.

V6 conclusion after the wide run: learned sparse factors can bridge stable local surface patches from the repaired contact interval into frame 2538. They cannot fill the rest of 2539 to 2550 because track support collapses, and they cannot replace per-frame mask/depth geometry even where the pair factor is ready. The next valid solver should use these factors as local temporal priors with measured target masks/depth as hard replay checks.
