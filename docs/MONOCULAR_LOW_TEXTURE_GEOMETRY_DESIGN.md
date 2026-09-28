# Low-texture geometry relocalization (after R7 descriptor diagnosis)

2026-09-27. Prediction-only monocular extension. No object-family branches and no GT/CAD inputs.

Measured mechanism blocker: the white-cup appearance keyframes contain only 3–9 interior RootSIFT features before pose/hand filtering. Lowering PnP support thresholds would not establish stable 6DoF. The hand occlusion mask is not the primary cause in this case. A geometric, rather than invented texture, observation is required.

For clips with a measured low median interior-feature count, propose a bounded set of proper SO(3) rotations around the existing mesh principal axes. Score actual full-triangle renders against the existing appearance masks in a fixed hand-occlusion domain, refine the best few hypotheses, and select a temporally consistent keyframe path. Preserve the baseline hypothesis. Interpolate corrections, not fabricated observations, over the full timeline. Apply only a demonstrated prediction-side contour improvement, with temporal and measured-surface support retained.

A changed geometric pose invalidates its old model-bound 3D points. Rebind or remove those derived rows explicitly; do not keep the old canonical points after changing orientation. Rebound per-frame visual supports are labeled as such, not promoted to material tracks. Fresh descriptor banks and existing RGB pair factors are then built on the new pose.

Discriminating tests: proper-rotation candidate set, trajectory selection avoids alternating ambiguous flips, correction interpolation preserves all frame IDs, and an asymmetric synthetic triangle mesh can select a 90-degree correct silhouette over an incorrect baseline. Symmetric or unsupported silhouettes should preserve uncertainty, not manufacture orientation certainty. This does not repair missing topology or solve the single-view absolute-scale gauge.
