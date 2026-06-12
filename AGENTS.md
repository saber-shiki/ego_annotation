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

Do not silently substitute pipeline components after an upfront design exists. If a design names a baseline component, that component remains required until a design amendment records the evidence that rejects it or replaces it. Cached outputs may be reused as memoized results of the same logical stage, but the pipeline must remain self-contained from raw video and named model/config inputs.

Each v16-or-later version must begin with an upfront design document before implementation. The design must define the raw-video input contract, full-timeline state variables, perception sources, optimization objective, physical consistency terms, acceptance checks, representative raw videos, expected failure modes, and complete render outputs.

Every v16-or-later deliverable must have the same frame count and duration as the original raw video. Short windows, contact slices, debug clips, contact sheets, and selected-frame renders are QC artifacts only.

Bug fixes, threshold changes, renderer fixes, server setup, and local mechanism studies belong inside the current version as patches or experiments. They do not create a new top-level version number.

## V18 Binding Baseline

Unless an explicit evidence-backed design amendment changes it, V18 is the following pipeline:

1. Camera/depth backbone: run DROID-SLAM-style camera tracking and the project metric-depth backend on the full raw video. Existing V16/V17 results may be reused only as memoized outputs of this same stage, keyed by raw video, model, and configuration.
2. Hand branch: run HaWoR, WiLoR, and RTMLib on the full video. HaWoR is the required temporal/occlusion hand baseline; WiLoR is the visible-frame MANO candidate; RTMLib is the independent 2D keypoint anchor. For visible frames, compare hand candidates by a fixed residual score: 2D keypoint reprojection, metric-depth disagreement, temporal acceleration, and hand bone-scale error. Accept a hand state only when the residual thresholds pass; use HaWoR through occlusion only when boundary-frame residuals and temporal continuation pass. Otherwise mark the hand unresolved/uncertain.
3. Object/part perception: a VLM planner must produce the object roster, physical-state proposal, and object/part prompts. OWLv2 provides text-conditioned keyframe boxes. SAM2 is the baseline video segmentation/tracking model that turns prompts into temporal object/part masks. SAM v1 is not the default tracking path.
4. Physical-state decision: VLM physical proposals are hypotheses. Rigid, articulated, deformable, and unresolved states are accepted by residual tests, not category/name/color branches.
5. Geometry/reconstruction: accepted masks plus metric depth produce visible point clouds/surfaces. Rigid objects or rigid parts are reconstructed by multi-frame depth fusion with SE(3) registration and silhouette/depth residual checks. Articulated objects reconstruct parts separately and accept an articulation only if its residuals beat the single-rigid model. Deformable or under-observed objects remain visible-surface-only unless stronger evidence supports more.
6. Factor graph: optimize only these variable families unless amended: bounded camera/depth correction, hand state, object/part SE(3), articulation parameter, contact switch, and occlusion owner. Factors are hand observation residuals, object mask/depth/registration residuals, temporal/rigid/articulation consistency, occlusion depth ordering, and contact/nonpenetration. Contact factors are allowed only when both hand and object/part geometry are valid.

## Work Progress Standard

Understanding is required before risky intervention, but understanding must produce action. Once the missing mechanism is identified, progress means implementing the mechanism, running the experiment, producing accepted evidence, or preserving a concrete failure that changes the next decision. Repeating readiness summaries, blocker manifests, audits, or render/status updates without a new evidence-producing attempt is not progress.

Maintain ledgers and audits at full standard, but they should be thin records of actual work. They must not become a separate workstream or an excuse for not advancing the causal pipeline. A ledger entry that only says the same thing is still not done is evidence of task steering failure.

## Runtime And Occlusion Discipline

Starting at v18, runtime is a design invariant. The default pipeline for a raw video must run in the same order of magnitude as the input duration. A method that takes hours for a roughly one-minute clip is a failed default design, even if its intermediate evidence is interesting. Per-instance neural reconstruction or training loops such as BundleSDF/NeRF-style optimization may be used only as offline research branches, never as the default path or as a way to discover obvious physical state types.

Occlusion must be represented explicitly. Hands and objects need per-frame visibility states such as visible, partially visible, occluded, out-of-frame, and unresolved, with occluder ownership and uncertainty for inferred states. Do not treat occlusion as only missing data, do not silently fill occluded hands/objects with certain poses, and do not claim contact or object pose through occlusion without depth-order and temporal evidence.

Do not block on foreground sleep/poll loops while waiting for long jobs. Long-running work must run in tmux or a job system with durable sentinels/logs while the agent does useful parallel work or returns control.
