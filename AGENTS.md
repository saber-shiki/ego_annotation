# Ego Annotation Operating Rules

## Work Ethics

Do not satisfy a requirement with an obviously false simplification. Difficulty cannot justify an incorrect substitute. If a required component is hard, keep the requirement intact, expose the unsolved part as an unsolved part, and work on the real mechanism.

Object pose in this project means reconstructed object geometry when the object is manipulated. A centroid, sphere, bounding box, category-specific primitive, or visual patch is not an acceptable replacement for object mesh reconstruction.

## Methodology

Do not encode visual case variation with hand-written if/else logic or object-family state machines. Python branches for categories such as color, object class, material, or action phrase are a failed perception strategy.

Use open-source vision models, open-vocabulary detectors, segmentation/tracking models, or LLM/VLM calls to produce object plans and segmentation evidence. The geometry pipeline should consume model outputs such as masks, tracks, depths, poses, confidences, and captions through one uniform reconstruction path.

When different cases require different treatment, put the difference in model-produced data or learned/open-vocabulary perception outputs. Keep the downstream reconstruction, filtering, contact reasoning, and rendering code category-agnostic unless a domain discontinuity is physically real and represented explicitly in the data.

## Versioning And Delivery

Starting at v16, every pipeline version is a complete pipeline version. A version cannot close with component evidence, a short window, or a partial render.

Each v16-or-later version must begin with an upfront design document before implementation. The design must define the raw-video input contract, full-timeline state variables, perception sources, optimization objective, physical consistency terms, acceptance checks, representative raw videos, expected failure modes, and complete render outputs.

Every v16-or-later deliverable must have the same frame count and duration as the original raw video. Short windows, contact slices, debug clips, contact sheets, and selected-frame renders are QC artifacts only.

Bug fixes, threshold changes, renderer fixes, server setup, and local mechanism studies belong inside the current version as patches or experiments. They do not create a new top-level version number.
