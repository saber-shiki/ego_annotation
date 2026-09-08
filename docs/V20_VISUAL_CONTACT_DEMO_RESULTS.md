# Visual contact demo — offline fitted demonstration

Owner explicitly approved visual plausibility/contact detail rather than strict P09 metric annotation. Base c49e028; preserve v16 archive, material studies, formal states. No GT input/evaluation in this demo. Generated shape is a demo contact/appearance reference, not measured collision/contact/SDF truth.

## Delivered mechanism

The same reconstructed 476616-face SAM3D surface and full 778-vertex/1538-face MANO hands are used. Real SMPL-X MANO LBS replays source HaWoR parameters, fixes left shapedirs-X and flat-mean conventions, and optimizes bounded articulation/root/camera-origin-similarity corrections plus object SE(3) on 31 keyframes. Corrections are interpolated through all 150 frames and the actual MANO model is evaluated per frame. The exported surface is not a 2D patch or vertex morph.

The primary demo contact hypotheses are visible thumbs near the object-owned mask. Actual pad samples query the intact object's closest triangles. Soft normal intrusion discouragement is NOT a physical signed-distance certificate. Hand image projection priors and object silhouette support preserve visible motion; P09 mismatch is not a hard gate.

A first preview exposed a genuine loss implementation problem: reverse contour matching to only hull corner vertices penalized long edges incorrectly. The second run samples 160 interpolated points along the projected actual-mesh hull edges; the object boundary loss dropped from ~27.2 to ~3.05. This is an image support approximation of the supplied real mesh, not replacement geometry.

## Runtime and numerical observations

Fit v2 took ~111s on physical GPU5 (logical cuda:0). Model zero-state max vertex errors: left8.14e-8m / right7.60e-8m; source/bridge vertices and source joint order are checked by current fitter. Full combined-scene rendering took ~264s on bounded CPU4 threads. This is offline demo fitting, not an input-duration production annotation claim.

Pad gap = mean unsigned point-to-triangle distance of the nearest3 samples in a12-vertex fingertip neighborhood. This measures the fitted reference geometry, NOT sensor-confirmed contact.

|frame|left pad before→after mm|right pad before→after mm|joint projection median change L/R px|
|---:|---:|---:|---:|
|35|1.21→0.27|23.58→0.04|1.11/0.85|
|65|2.62→0.24|21.65→0.22|1.68/0.91|
|95|4.02→0.35|33.94→0.48|0.46/1.14|
|120|23.93→0.63|37.89→0.21|0.39/0.31|
|140|15.21→15.48|5.46→0.84|0.64/0.94|
|149|13.18→10.10|21.05→0.41|1.24/1.25|

Explicit fitted camera-origin hand similarity scales range0.9167–0.9841. Hand root rotation correction max~0.96deg; translation max~5.34mm. Object rotation correction max~6.14deg and translation normmax~38.08mm. These are demo registration adjustments, not claims of recovered absolute physical pose. In particular late left-hand contacts are NOT forced and not claimed precise.

## Visual delivery

Experiment root:
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_depth_order_corrected_fullrun_20260903T101508Z/run/P0014_84ea2dcc_carton_milk_f2370_2519/experiments/visual_contact_demo_20260908`

- `full_demo_v2/interaction_demo.mp4`: full original vs smooth-surface overlay, no dense point cloud/wireframe.
- `full_demo_v2/contact_detail.mp4`: full-duration original crop vs solid-surface +20deg yaw inspection; angle is labeled. `--detail-yaw-deg 0` yields camera-aligned detail.
- `full_demo_v2/interaction_demo.rrd`: actual world object/left/right MANO mesh and source camera. Object canonical mesh is logged once plus per-frame transform; no missing world-hand dictionary lookup.
- `full_demo_v2/contact_keyframes_montage.jpg` and `contact_before_after_120.jpg`.
- `fit_v2/demo_state.npz`: full original mesh, frame-wise object poses, fitted/base MANO vertices/joints, camera calibration and parameter corrections.

Both MP4s:1920x960,30FPS,150frames,5s. Full-video ffmpeg decoding passed. Combined-scene ray casting uses all original triangles and closest-depth surface occlusion; no display-only front-face deletion or source-video texture pasted into object geometry. Smooth hand shading interpolates vertex normals only; geometry remains original MANO.

Inspected: raw tip overlays 65/120, previews35/65/120/149, before/after120 novel-view surfaces and final keyframes. Clearer thumb attachment at120; shape/hand contour approximations and wrist-only MANO truncation remain visible. This is a stylized reconstruction, not photorealistic resimulation.

## Validation

5 standard unittest cases: camera roundtrip/halfpixel intrinsics; camera-origin scale projection invariance; nonzero rotation gradient at identity; actual triangle contact/shared-scene occlusion; nonzero torch/NumPy export-transform parity. Fitter verifies MANO zero-state vertices and joint indexing. Independent read-only review: OK with notes; novel-view description drift fixed, no mathematical/render-dataflow defect found. Review is not a guarantee of physical contact or of Rerun GUI playback. Both MP4s decoded fully; RRD generated without error and actual meshes explicitly logged.

The final-source input identity/bridge assertions were added after fit v2 as hardening, without changing optimizer/render math. Model replay checks and parameters persisted from run. Formal P14/P15/D19 and prior v16 assets untouched. No push/merge.
