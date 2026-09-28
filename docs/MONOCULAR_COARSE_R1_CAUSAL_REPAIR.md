# R1 causal repair before R2

R1 completed five frozen inputs and full videos. It recovered non-identity rotations and improved source-mask alignment, but it is not acceptable as a near-GT coarse initializer. Image consistency alone did not prevent large 3D deviations. A free per-frame additive depth bias plus weak magnitude prior can absorb geometric errors; ray-only depth factors also disappear when a displaced mesh misses the measured rays.

R2 changes the coupling, not the evaluation thresholds:

1. Estimate relative per-frame monocular depth scale from robust intra-object pairwise distances at existing RGB correspondences. The same material-point distances must remain constant under rigid motion. Use only eligible source P09/RGB measurements; anchor one scale at the appearance frame and keep disconnected support uncertain. No GT, CAD, extra view or per-case parameter.
2. Remove the free additive-bias variables from the production refinement passes. Consume corrected source clouds as uncertain derived measurements.
3. Add trimmed observed-point to actual-model-triangle correspondences, refreshed between passes, so a mesh that has moved away still receives a 3D registration signal rather than losing all depth factors. This is an unsigned shape prior, not signed collision authority.
4. Preserve raw clouds/poses/receipts, retain the R1 attempt, rerun every case under one R2 configuration. GT may be read only by the external evaluator after the new candidate is frozen.

Discriminating outcome: smaller candidate-to-source surface residual and fewer unsupported depth jumps would support the causal account. Remaining global metric offset is still unobservable with these inputs; unchanged or worse rotation implies correspondence/orientation ambiguity, not a reason to hide errors with clipping or selective frame removal.
