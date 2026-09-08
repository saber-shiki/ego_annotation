# Material RGB-D dependency repair: results and scope

Base f22876a. New isolated branch `research/milk-v20-material-rgbd-repair-clean-20260908`.

## Mechanism changes

- Legacy RGB canonical serialization is recovered ONCE to fixed world observations via its hash-bound producer pose. Candidate source inverse + candidate target transform now enter pixel residual; both endpoint Jacobians are nonzero.
- Low-conditioning PnP translation-only factors condition on immutable reference orientation and a source support point. The residual has zero derivative w.r.t. candidate rotations and is invariant to a common shift of the world origin. Missing source support rejects that conditional factor.
- New cached-material adapter extracts target prediction depth with source/target P09 eligibility, owned-mask support, exact pixel/intrinsics/pose/hash checks and decreasing weights for larger UniDepth predicted error. No inference/GT/generated geometry.
- Material mode uses fixed matched world points in place of adjacent/anchor NN. Compressed PnP edges are removed wherever their raw material/pixel observations are active. No duplicated PnP and raw factor likelihood.
- f143 remains a latent SE(3) node: 142->143 image-only; 143->144 source metric omitted. No invented depth.
- Renderer caption now says hash-bound to pose *report*, not falsely claiming the mesh entered the pose objective. Face retention/coordinates unchanged.

## Controlled execution (GT not loaded)

Root:
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_depth_order_corrected_fullrun_20260903T101508Z/run/P0014_84ea2dcc_carton_milk_f2370_2519/experiments/pose_repair_v20_material_rgbd_20260908`

`CONTROL_PARAMETERS.json` and each command.json record all explicit settings. Both new controls use the same window110–149, anchor120, four outer iterations, initial pose, rotation prior sigma0.5rad and translation prior sigma0.2m. They use no generated validation during optimization. The old frozen diagnostic is a historical comparator, not an otherwise-identical ablation. New point arrays and fixed seed are reused; no parameter sweep or GT ranking.

Packed input: 71 pairs, 8145 image points, 8047 eligible metric pairs before solver window/subsampling. f143 has97 image points/0 target metric observations. 143->144 is excluded due to source P09 absence.

On identical FULL material observations after freeze (not just optimization subsamples):

|candidate|pixel median px|pixel p95 px|3D median mm|3D p95 mm|
|---|---:|---:|---:|---:|
|old frozen diagnostic v25|2.219|7.591|1.933|7.561|
|corrected pixel control|1.323|4.298|1.593|6.656|
|material RGB-D control|1.150|2.879|1.462|5.457|

All original rows outside window are identical. Both controls accepted 4 outers and did not hit update trust bounds. Solver times: ~72.3s pixel control, ~27.0s material control on the server, bounded CPU4 threads.

## Generated geometry comparison: not a successful replacement

Full 476616-face BVH, same original P09 rays and same frozen mesh for all pose controls, before the new shape pass:

|frame|old signed mm / coverage|corrected pixel signed mm / coverage|material signed mm / coverage|
|---:|---:|---:|---:|
|130|-2.22 / .731|-0.62 / .724|-1.54 / .730|
|135|-8.37 / .776|-5.97 / .770|-18.53 / .758|
|140|-29.27 / .827|-24.77 / .764|-24.05 / .670|
|142|-37.94 / .830|-32.59 / .769|-23.94 / .609|
|144|-40.40 / .873|-37.82 / .835|-25.26 / .564|
|149|-44.16 / .824|-42.02 / .810|-23.11 / .525|

Common-hit per-frame comparisons are also saved in `frozen_comparison_v1/comparison.json` alongside separate coverage changes. The depth median over fewer hits is NOT sufficient to select the material result. It loses coverage and worsens f135 depth.

One fixed-pose shared anisotropic shape attempt (2 outers, physical GPU5 logical cuda:0, original step/fixed-initial limits) accepted two small updates. Its raster metrics at f149 are signed median -21.34mm, coverage .5392, silhouette IoU .458. This does not restore old coverage. Status is `shape_alignment_incomplete_signed_front_validation`.

**Decision: preserve v16 archive and prior display baseline. Neither new control is promoted as annotation-ready.** The cause-directed factors now improve their actual observation metrics, but absolute pose / complete mesh binding / multi-view visible geometry remain inconsistent. This experiment does not prove which of learned depth error, absolute pose anchoring, or unobserved generated geometry dominates the residual. It does disprove treating correct material consistency as sufficient evidence for full visible-mesh alignment.

## Real full-video evidence

- `visual_review/corrected_pixels/side_by_side.mp4`, `full.rrd`, `full.json`
- `visual_review/material_rgbd_shape/side_by_side.mp4`, `full.rrd`, `full.json`
- `visual_review/comparison_montage.png` and representative stills 120/130/135/140/142/143/144/149

Both renders: 150frames, 1920x960,30FPS,5s; exact mesh path/hash; zero display front-face pruning. Checked representative stills: material variant has a visible lateral/coverage mismatch late. Diagnostic bindings preserve numerical pose rows and make no alignment/authority claim.

A managed-shell multi-line input stalled before dispatch and was cleared after verifying no job existed. Later shape/render bash waiting windows timed out while real managed jobs continued; logs, exit sentinels and processes were checked instead of duplicating jobs. All dispatched controls/shape/renders eventually exited0. Expensive shape/raster/render is offline research, not a claim of an input-duration default pipeline.

## Validation and independent review

- Standard-library unittest mechanism suite: 11 tests, including nontrivial transforms, both endpoint Jacobians, rotation isolation, rebasing, hash tampering, missing metric eligibility, and source NaN/zero/negative/Inf.
- Existing direct focused suites:22 late-window +12 keyframe tests.
- Python compilation and git diff check.
- Independent factor-math review: no issues found.
- Independent input-contract review: producer identity and timeline rejection diagnostics hardened; source-depth validity made explicit. Recheck withdrew the original NaN-survivor claim (the old comparison already rejected NaN), confirmed fixes, verdict OK for named source-contract changes. Reviewers did not certify visual replacement quality.

## Next mechanism, not another weight sweep

Reconcile absolute pose and shared canonical visible patches using material identity over stable and newly exposed surfaces, with bidirectional/cycle and uncertainty checks. Validate the generated object's *visible face*, not only relative matched-point motion. Current patch has not implemented a persistent multi-view surface reconstruction. Until that is established, keep mesh fixed for diagnosis and do not blind-shift camera Z, prune front faces, or use GT/generated geometry as physical pose authority.
