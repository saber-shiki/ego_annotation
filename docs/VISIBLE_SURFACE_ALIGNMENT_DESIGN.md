# Trusted-UniDepth visible-surface alignment profile

Date: 2026-09-27. User-directed change of reference: preserve raw UniDepth and eligible de-occluded P09 surfaces. Do not force SAM3D's internal MoGe to match, do not modify the observation producer, and do not use CAD or GT inputs.

## Physical operation

The existing source visual mesh already includes a camera-origin scene scale. Solve a NEW residual uniform object scale and independent translation:

`p_camera = R_camera_from_object @ (s * v_centered_native) + t_camera`

Scale is one positive scalar per rigid object, not a per-frame scale, not a volume-preserving stretch, and not `s * (R v + t)`. Record the old scene scale and new residual length scale separately to prevent double application. Initial registration uses the appearance-anchor mesh and the SAME frame's retained visible surface. Rotation is held at the existing hypothesis during this module; scale cannot be claimed to fix orientation ambiguity.

## Registration mechanism

- Actual triangle first-hit queries and view-dependent visible mesh samples.
- All fixed eligible observed samples participate; missed rays get a visible-surface correspondence and explicit coverage cost, not deletion from the score.
- Match observed points to the mesh's visible part, never the full mesh bounding box to a partial point cloud and never a symmetric full-mesh/partial-cloud Chamfer objective.
- Use independent object scale and translation, robust 3D residuals, de-occlusion-aware contour support and a weak scale prior. Do not manufacture hidden-surface dimensions from a small visible patch.
- Refresh correspondences between iterations and score updates on the same fixed source population. Preserve a trace of scale/translation/coverage/first-hit residual changes.
- Anchor calibration precedes tracking; multiple eligible, temporally distributed frames may refine the single shared scale using fixed tracked rotations and separate translations. Then freeze scale and fit the full translation timeline to RAW visible surfaces with image/temporal support.

## Boundaries

Raw depth and point clouds remain byte/value-identical. No relative depth scale or bias fitting in this profile. No anisotropic deformation, no geometry relocalization experiment, no hand fitting, no RL/reward changes. Complete mesh faces/vertex identity and source MANO are preserved. Missing metric observations remain missing; trajectories can still be inferred from image/time evidence.

The profile trusts UniDepth operationally; it does not certify absolute physical accuracy or hidden mesh geometry. The previous CAD+pose rendered reference is evaluation-only, not a measured depth sensor stream and never a registration target.

## Discriminating tests and real runs

Tests: recover independent scale/translation on actual synthetic triangles; do not shrink the full object into an occluded patch; do not fit a hidden backface instead of the first visible surface; preserve raw points, mesh faces and a single scale across frames; reject mismatched coordinate/time input. Weakly observable scale must be reported, not made certain by a prior.

Real runs: the same five frozen S3-derived source bundles, one shared configuration, new immutable snapshot and output roots on the CPU server. Report pre/post first-surface 3D/depth residuals, coverage, visible contour support, shared scale, and full 150-frame object/MANO videos. Inspect those before independent GT evaluation. Do not pick a scale or parameter per case using GT/CAD results.

## R2 isolation correction after the first real batch

The first batch proved that the visible-reference operation can improve fit, but recalculating mesh-bound RGB tracks after scaling changed some rotation branches. A scalar size-registration experiment should not be confounded by a new rotation estimate. The active profile therefore computes the existing RGB rotation proposal once using the original template and its established 128px tracker contour plane, freezes all R matrices, and then registers shared scale plus translations against raw UniDepth. It stores `alignment_initial_rotation/translation_world` for an exactly matched before/after comparison and asserts rotation equality. Projection-preserving translation scaling is only an initializer; t remains independently optimized.
