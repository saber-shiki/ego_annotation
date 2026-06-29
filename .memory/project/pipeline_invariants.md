# Pipeline invariants

## Uncertainty labels must control state/render semantics

A label such as `unsupported_uncertain` is not an uncertainty mechanism unless downstream state construction and rendering consume it. In V19 clip001851, Poisson observed-surface fill faces were labeled `unsupported_uncertain` but still concatenated into the completed object mesh and rendered as accepted green object body. That is invalid.

Untrusted Poisson fill, unsupported TRELLIS faces, and any other uncertain completion surfaces must either be excluded from accepted canonical object geometry or rendered as a separate uncertainty hypothesis. They must not be silently promoted to the object body just because they are present in a mesh container.

## Rigid anchor frames require candidate review

A rigid-object anchor frame defines canonical visible surface, metric scale/extent checks, TRELLIS conditioning, and later pose refits. It is a physical-state decision, not a bookkeeping default. Scripts may rank/propose anchor candidates, but they must expose visual review evidence and raw score factors; the runtime agent must inspect candidate frames and record the chosen anchor with rationale. Do not silently choose the largest mask, most sampled points, earliest frame, or a contact-near frame when a less-occluded full-object frame gives stronger geometry evidence.
