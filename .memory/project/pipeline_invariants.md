# Pipeline invariants

## Public API names are product/domain names

Internal planning tracks such as delivery, research, demo, v18, or v19 are not product concepts and must not appear in customer-facing API endpoint paths. Endpoint names should name the customer operation or data domain, e.g. annotation jobs, video analysis jobs, hand state, camera pose, or HOI extension resources. Track labels may remain in task memory, branch names, or internal run roots, but public schemas and endpoints should use stable domain nouns such as `ego.annotation.output` and optional domain extensions such as `ego.hoi`.

## Uncertainty labels must control state/render semantics

A label such as `unsupported_uncertain` is not an uncertainty mechanism unless downstream state construction and rendering consume it. In V19 clip001851, Poisson observed-surface fill faces were labeled `unsupported_uncertain` but still concatenated into the completed object mesh and rendered as accepted green object body. That is invalid.

Untrusted Poisson fill, unsupported TRELLIS faces, and any other uncertain completion surfaces must either be excluded from accepted canonical object geometry or rendered as a separate uncertainty hypothesis. They must not be silently promoted to the object body just because they are present in a mesh container.

## Rigid anchor frames require candidate review

A rigid-object anchor frame defines canonical visible surface, metric scale/extent checks, TRELLIS conditioning, and later pose refits. It is a physical-state decision, not a bookkeeping default. Scripts may rank/propose anchor candidates, but they must expose visual review evidence and raw score factors; the runtime agent must inspect candidate frames and record the chosen anchor with rationale. Do not silently choose the largest mask, most sampled points, earliest frame, or a contact-near frame when a less-occluded full-object frame gives stronger geometry evidence.

## Contact wording follows the metric source-gap model

V19 rendered contact wording must be generated from the physical state summary, not a generic banner phrase. Use the Gaussian contact compatibility score and `contact_likelihood_state_counts` as a compatibility residual at the combined metric uncertainty scale. The wording may distinguish regimes such as `near-contact compatible; ownership/NP unresolved`, `near-contact uncertain; ownership/NP unresolved`, and `contact unlikely by source gap`.

These labels are publication semantics over the existing state. They do not accept contact ownership, do not prove signed nonpenetration, and do not authorize moving metric MANO or object pose. A clip can be a correct non-contact artifact; forcing contact on a slice whose GT/metric source-gap evidence is contact-unlikely is a physical error.
