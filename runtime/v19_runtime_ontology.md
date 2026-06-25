# V19 Runtime Ontology

This file defines the runtime prediction ontology. Use it to decide what a state variable means and what evidence may support it.

## Inputs

Runtime inputs are an egocentric input video, a fresh run root, a case id, this runtime ontology/runbook, the scripts named by the runbook, and prediction-side sensor metadata such as camera calibration.

## Outputs

The runtime output is a prediction run root. It contains `input/`, `measurements/`, `state/`, `renders/`, and `logs/`. The renderer consumes `state/`; logs and measurements are provenance, not the final annotation.

## Physical state variables

- `camera`: intrinsics, camera/head pose, depth/scale provenance, frame/time semantics, and uncertainty.
- `hands`: metric 3D MANO state over time, side, camera/world transforms, visibility, provenance, and uncertainty.
- `objects`: object instances, masks/tracks, physical branch, reconstructed or adapted geometry, pose/posterior, provenance, and uncertainty.
- `visibility_occlusion`: visible, partially visible, occluded, out-of-frame, or unresolved state for hands and objects, with occluder ownership when inferable.
- `contact`: contact, near-contact, non-contact, or unresolved state with patch/distance evidence and uncertainty.
- `nonpenetration`: residuals between hand/object geometry and uncertainty; absence of a valid signed volume is unresolved, not success.
- `renders`: visible overlay/world/side-by-side annotations caused by the state variables above.

## Evidence rules

- A detector box, keypoint track, mask, depth map, point cloud, centroid, label, or JSON row is a measurement, not physical state by itself.
- Object pose requires object geometry adapted or fitted to observed instance evidence and a pose trajectory/posterior.
- Hand state requires metric MANO surface or reproducible MANO parameters with camera/world semantics.
- Contact and occlusion require geometric, depth-order, temporal, or explicitly uncertain evidence. Do not make them certain from a semantic label alone.
- Weak measurements continue downstream with uncertainty. Broken contracts, wrong frame alignment, wrong coordinate frame, wrong object mask, side swap, missing geometry, or invalid units must be fixed or represented as unresolved.

## Runtime discipline

Use the runbook files in this workspace. Do not search for project history or development instructions. If the bundle lacks a required script, metadata file, or model asset path, write a concrete missing-component record in the run root and stop that branch.
