# Pipeline V19 current epistemic state

## Current supported claim

The HOT3D `clip-001850` pinhole V19 v5 mesh-frame-repaired freeze is rejected as a correctness artifact. The latest canonical contact sheet shown to the user falsified the previous acceptance claim: the green keyboard/object geometry is visibly stretched and displaced relative to the physical keyboard/key field across typical frames. The run root and manifest remain immutable evidence, not a deliverable.

Rejected boundary:
- Run root: `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.
- Local mirror: `/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.
- Manifest hash: `2d999fbc4f51e375b4a9eb7277943bba74ce542cf0c613ece3479c888a267080`.
- Canonical frames sheet: `/tmp/v19_typical_render_frames/typical_repaired_v19_frames_contact_sheet.jpg`.

## What remains supported

The raw P12 TRELLIS-vs-P13 completed-mesh mismatch was a real bug. Commit `1b0aa0f Require completed mesh provenance for V19 P18/P19` is still a valid guard against applying P13 completed-canonical poses to raw P12 model-frame vertices. It is not sufficient to make the rendered object physically registered.

The earlier wrong-mesh freeze hash `daa3978f641691ef148720a75a9ec3bea35a22e55b6c01eb6a0eea0ed85f2a3c` remains rejected. The newer mesh-frame freeze hash `2d999fbc4f51e375b4a9eb7277943bba74ce542cf0c613ece3479c888a267080` is also rejected as a deliverable because visual registration is still wrong.

## Strict blocker

The final canonical overlay/world/side-by-side must render object geometry that coincides with the physical keyboard in the video. Green pixels on or near the key field are not enough. A stretched/displaced keyboard-like shape is a failed physical annotation.

## Live mechanisms

1. **P13 completed geometry is wrong in shape/scale**: the completed keyboard mesh may be too elongated/broad or contaminated by table/hand surfaces. Prediction: rendering the P13 mesh at any pose will not match the key field extents; P13/P11/P09 evidence will show overlarge support or wrong anchor surface.
2. **P14/P15 pose fit is using wrong visible support**: the pose may align the completed mesh to a contaminated point set or wrong subset of the keyboard/table. Prediction: P14 residuals/eligible frames will look numerically plausible while projected mesh overlays wrong table/hand-supported regions; direct projection of P13 mesh with P14/P15 pose will reproduce the canonical mismatch.
3. **P19 projection/camera convention is wrong**: the world pose could be reasonable but rendered with incorrect camera transform, scale, handedness, or image-size intrinsics. Prediction: world view may be internally coherent while 2D overlay is displaced; projecting object vertices with independent camera/intrinsics code will disagree with P19 overlay.
4. **P13 completed-canonical frame still mismatches P14 pose frame**: path provenance now points to P13, but the pose fitter and renderer may disagree on which P13 mesh coordinate transform is canonical. Prediction: P14 fitting inputs or P13 report contain an additional transform that P19 does not apply.

## Next action

Do not run HOT3D scoring and do not claim any freeze accepted. Localize the visible mismatch against final canonical frames by comparing projected P13 mesh, P14/P15 pose rows, P09/P11 observed surfels, and P19 renderer projection. The next intervention must change the rendered physical annotation, not labels, ledgers, or uncertainty wording.
