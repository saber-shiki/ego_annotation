# V20 keyframe pose-graph experiment

This directory contains a diagnostic-only implementation for the milk-carton case.
It is intentionally separate from the formal corrected run.

## Compute contract

- Optimize only keyframe SE(3) nodes (`--keyframe-interval 5` or `10`).
- Use the sparse linearized Gauss--Newton path by default (`--optimizer-mode
  linearized_gn`); `least_squares` is retained only as a slow comparison.
- Non-keyframes are output with the left correction composition
  `R = DeltaR @ R0`, `t = DeltaR @ t0 + Deltat`.
- P15 first-hit/silhouette factors must be contract v2 and are path/hash
  checked against the exact annotation, P14 pose report, and observed-surface
  mesh. Factors are frozen for this prototype and evaluated nonlinearly at each
  outer pass; this is not a claim of GPU correspondence regeneration.
- When enabled, observed-surface 3-D correspondences are rebuilt per outer pass.
  Point-to-plane/rotation-covariance edges are evidence, not authority.
- Generated SAM3D/TRELLIS faces are never consumed as pose, collision, contact,
  signed-distance, or nonpenetration authority.

## Acceptance gates

An update is kept only when the objective decreases and all of the following
remain within bounds: observed-to-mesh median degradation, cumulative SE(3)
correction, exact face-union mask IoU/centroid proxy (both median and mean).
The outputs remain `diagnostic_only` and `annotation_ready=false` even when a
candidate passes these gates.

## Observed result

For the formal milk input, image-only keyframe GN runs completed in about 56 s
(K=5, three outer passes) and 29 s (K=10, one outer pass), compared with the
previously aborted 146-node sparse finite-difference graph. Exact 256-pixel
mask evaluation is recorded in the case output directory under
`experiments/pose_repair_v20_keyframe_20260904/`.

The 3-D-only and weak-3-D candidates that conflicted with the observed mask
were rejected fail-closed. Do not use the V20 pose reports as formal P15/P16--P18,
D18, D19, or formal RRD inputs without a separate review and promotion step.
