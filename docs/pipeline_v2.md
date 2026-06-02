# Pipeline V2: Physics-Aware Object Refinement

## Scope

Pipeline v2 keeps the v1 measurement stack and adds a physical consistency layer over the object state. It does not rerun DROID, WiLoR, OWLv2, or SAM. It consumes `annotations_v1_full.json`, preserves the object image ray for every active frame, and refines only the object depth and extent in the DROID/WiLoR world coordinate system.

The reason is causal: v1's worst remaining object error is not a missing renderer or a missing Kalman pass. The object already has a full-frame mask track, a source-camera ray, DROID depth rows, and hand-contact anchors. The next observable correction is whether the 3D object proxy is physically consistent with the MANO hands.

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

The renderer is the same visual contract as v1. The overlay is unchanged in image space because v2 refines only the 3D object state. The 3D panel draws the v2 object centroid and spherical extent.

## Validated Samples

Task7 tomato chopping/preparation:

`outputs/examples/tomato_v2_physics_task7/`

- videos: 2040 frames, 30 fps; overlay/reconstruction 960x540; side-by-side 1920x540;
- optimizer: 1590 active object frames, 53,343 residuals, 111,440 sparse Jacobian nonzeros, 152 function evaluations;
- contact median error: 14.6 mm to 8.4 mm;
- contact p95 error: 50.7 mm to 26.4 mm;
- max sampled hand-object penetration: 57.0 mm to 18.0 mm;
- p95 sampled penetration: 4.8 mm to 0.0 mm.

Task5 tomato washing/peeling:

`outputs/examples/tomato_v2_physics_task5/`

- videos: 960 frames, 30 fps; overlay/reconstruction 960x540; side-by-side 1920x540;
- optimizer: 670 active object frames, 47,825 residuals, 97,652 sparse Jacobian nonzeros, 113 function evaluations;
- contact median error: 13.2 mm to 10.4 mm;
- contact p95 error: 82.7 mm to 47.5 mm;
- max sampled hand-object penetration: 43.5 mm to 18.5 mm;
- p95 sampled penetration: 8.9 mm to 0.064 mm.

Fresh stills inspected after rendering:

- task7 frames 600 and 1910: object remains on visible tomato material; 3D object extent stays close to active hands;
- task5 frame 270: predicted pre-contact object state remains on visible tomato and is not over-constrained by hand contact;
- task5 frame 480: sink contact frame remains visually coherent after v2 depth/radius refinement.

## Evidence Limits

V2 improves internal physical consistency under the v1 measurement model. It still does not certify 5 mm absolute accuracy.

The remaining limit is observability. The dataset package inspected so far has RGB video and action JSON, with no depth, IMU, camera calibration, fiducials, CAD model, object size, or ground-truth pose. DROID scale is still anchored by WiLoR hand geometry and DROID relative depth. The v2 factor graph can reduce contradictions between hand contact, object depth, object extent, and temporal motion; it cannot create a calibrated metric reference.

## V3 Direction

The next improvement should target the missing observability rather than adding another smoother:

- camera and scale: evaluate VGGT or MASt3R-style dense geometry as an additional depth/correspondence prior against DROID on the same clips;
- object masks: replace tomato-specific color proposals with SAM2 video memory and promptable object identity;
- hands: export full MANO vertices for all inspected samples and fit MANO through temporal/contact residuals, not only per-frame detector output;
- calibration: add an explicit scale source, such as measured hand size, known object/tool size, AprilTag/Charuco calibration, table plane measurement, or depth/IMU if available.
