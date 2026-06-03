# Pipeline V2: Physics-Aware Object Refinement

## Scope

Pipeline v2 keeps the v1 measurement stack and adds a physical consistency layer over the object state. It consumes `annotations_v1_full.json`, preserves the object image ray for every active frame, and refines the object depth and extent in the DROID/WiLoR world coordinate system.

The design target follows from v1's residual error. The object already has a full-frame mask track, a source-camera ray, DROID depth rows, and hand-contact anchors. The next observable correction is whether the 3D object proxy is physically consistent with the MANO hands.

## Implementation

`scripts/refine_v2_physics.py` builds a sparse factor graph with one depth and one radius variable per active object frame.

Each frame contributes:

- depth prior: keep the v1 DROID/contact optimized depth unless physical terms justify moving it;
- mask-radius prior: keep the 3D radius consistent with the visible mask area and current depth;
- contact-surface residuals: fingertips near the object in image space should lie near the object surface in 3D;
- non-penetration residuals: MANO surface samples near the object in image space should not sit inside the object extent;
- temporal center acceleration: object centers should move smoothly along the fixed source rays;
- temporal radius acceleration: object extent should vary smoothly except where the mask evidence supports change.

The optimizer uses `scipy.optimize.least_squares` with a sparse Jacobian pattern over the chain graph. A short-budget run that does not converge raises with the diagnostic residual table instead of writing a partial v2 annotation.

## Outputs

For each clip, v2 writes:

- `annotations_v2_physics.json`
- `qc_v2_physics.json`
- `overlay_mano_object.mp4`
- `reconstruction_3d_world.mp4`
- `side_by_side.mp4`

The renderer is the same visual contract as v1. The overlay is unchanged in image space because v2 refines only the 3D object state. The 3D panel draws the head camera frustum, local trajectory, MANO hands, and v2 object centroid/extent in a camera-up aligned world view. The JSON remains in DROID world coordinates; the display view avoids treating DROID's raw z coordinate as physical height.

The reported contact and penetration values are internal residual metrics against the spherical object proxy and sampled MANO surface. They measure physical consistency within this model. They are not ground-truth pose error.

## Validated Samples

Task7 tomato chopping/preparation, full-MANO rerun:

`/data2/ego_annotation_outputs/fullmesh_task7/v2_physics_contact/`

- videos: 2040 frames, 30 fps; overlay/reconstruction 960x540; side-by-side 1920x540;
- optimizer: 1590 active object frames, 105,349 residuals, 215,452 sparse Jacobian nonzeros, 183 function evaluations;
- MANO surface: full 778-vertex mesh per hand when WiLoR detects a hand;
- object semantic interval: 1600 frames; 1590 v2 object pose frames and 10 explicitly unobserved degenerate edge/occlusion frames;
- contact median internal residual: 14.6 mm to 5.9 mm;
- contact p95 internal residual: 50.7 mm to 25.2 mm;
- max sampled hand-object penetration: 57.5 mm to 17.9 mm;
- p95 sampled penetration: 20.3 mm to 2.9 mm.

Task5 tomato washing/peeling, full-MANO rerun:

`/data2/ego_annotation_outputs/fullmesh_task5/v2_physics_contact/`

- videos: 960 frames, 30 fps; overlay/reconstruction 960x540; side-by-side 1920x540;
- optimizer: 670 active object frames, 48,632 residuals, 99,266 sparse Jacobian nonzeros, 208 function evaluations;
- MANO surface: full 778-vertex mesh per hand when WiLoR detects a hand;
- object semantic interval: 670 frames; 670 v2 object pose frames;
- contact median internal residual: 13.2 mm to 7.0 mm;
- contact p95 internal residual: 82.7 mm to 44.0 mm;
- max sampled hand-object penetration: 45.4 mm to 29.8 mm;
- p95 sampled penetration: 25.8 mm to 5.4 mm.

Residual RMS before and after v2:

| clip | residual group | before | after |
| --- | --- | ---: | ---: |
| task7 | depth prior | 0.000 | 0.389 |
| task7 | radius mask | 0.000 | 0.654 |
| task7 | center acceleration | 0.089 | 0.209 |
| task7 | radius acceleration | 0.139 | 0.240 |
| task7 | contact surface | 2.619 | 1.289 |
| task7 | non-penetration | 0.411 | 0.066 |
| task5 | depth prior | 0.000 | 0.714 |
| task5 | radius mask | 0.000 | 1.179 |
| task5 | center acceleration | 0.102 | 0.340 |
| task5 | radius acceleration | 0.064 | 0.300 |
| task5 | contact surface | 4.073 | 2.005 |
| task5 | non-penetration | 0.535 | 0.112 |

The physical refinement moves object depth/radius away from the v1 depth and mask priors to reduce contact and penetration contradictions. Task7 depth IQR changed by -1.1 mm to 29.5 mm; task5 depth IQR changed by -22.6 mm to 12.8 mm.

Fresh stills inspected after rendering:

- task7 frames 600 and 1910: object remains on visible tomato material; 3D object extent stays close to active hands;
- task7 frames 334-343: degenerate edge/occlusion states remain unobserved rather than forcing a 3D object pose;
- task7 frame 1980: object annotation is absent after the tomato semantic interval; only hand annotations remain;
- task5 frame 270: predicted pre-contact object state remains on visible tomato with weak hand-contact influence;
- task5 frame 274: first measured tomato state remains on the visible tomato;
- task5 frame 480: sink contact frame remains visually coherent after v2 depth/radius refinement.

The full-MANO rerun changed the v2 tuning. The earlier sampled-surface setting over-prioritized non-penetration after 778-vertex MANO became available and slightly worsened task5 contact p95. The current default uses `contact_sigma_m = 0.010` and `penetration_sigma_m = 0.020`, which reduced both contact error and sampled penetration on task5 and task7.

## Evidence Limits

V2 improves internal physical consistency under the v1 measurement model. Absolute 5 mm certification still requires an external metric reference.

The remaining limit is observability. The dataset package inspected so far has RGB video and action JSON; depth, IMU, camera calibration, fiducials, CAD model, object size, and ground-truth pose are absent. DROID scale is still anchored by WiLoR hand geometry and DROID relative depth. The v2 factor graph can reduce contradictions between hand contact, object depth, object extent, and temporal motion. A calibrated metric reference must come from additional evidence.

## V3 Direction

The next improvement should target the missing observability:

- camera and scale: evaluate VGGT or MASt3R-style dense geometry as an additional depth/correspondence prior against DROID on the same clips;
- object masks: replace tomato-specific color proposals with action-segment object profiles, promptable OWLv2/SAM proposals, hand-contact scoring, and video-memory tracking;
- hands: fit MANO through temporal/contact residuals in addition to per-frame detector output;
- calibration: add an explicit scale source, such as measured hand size, known object/tool size, AprilTag/Charuco calibration, table plane measurement, or depth/IMU if available.

SAM2 check on task7 frames 312-360:

- Input interval included all 49 source frames, not only old v2 measured frames.
- Prompt frame: 312, from the existing v2 object box.
- SAM2 produced visible masks on 38/49 frames and lost frames 335-345, the heavy occlusion span.
- This falsifies a segmentation-only v3 for occlusions. The object state still needs contact-aware prediction and physical consistency during full occlusion.
