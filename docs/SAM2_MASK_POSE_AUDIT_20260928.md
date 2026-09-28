# SAM2 mask-to-pose audit — 2026-09-28

## Answer

The original SAM2 mask transport is not the failure: all 150 frames in every case have byte-identical `source_mask_path` and `appearance_mask_path` content, and the preprocessing source receipt verifies those files. The final repair mesh can nevertheless be visibly outside that mask.

Compared against the saved `monocular_visible_surface_20260928_jump_r2` full mesh, raw visible-mask IoU medians were:

| case | bootstrap RGB pose | final repair | final frames <0.5 IoU |
|---|---:|---:|---:|
| milk |0.808|0.848|0|
| soup |0.871|0.858|0|
| BBQ |0.678|0.653|17|
| mug |0.593|0.598|19|
| spatula |0.521|0.440|103|

Raw IoU is conservative for an amodal full mesh: hands occlude visible SAM2 pixels, so a full mesh can legitimately extend outside a visible mask. The fixed MANO/owned visible-domain medians were milk .858, soup .898, BBQ .733, mug .665, spatula .435. The low spatula result therefore remains after the known hand-domain qualification; it is not explained by occlusion alone.

Inspection: milk/soup generally align; BBQ has visible outliers (for example f37); mug has cup/handle mismatches; spatula is repeatedly far outside the SAM2 support. Evidence image: `/mnt/user-home/kupingxin/ego_annotation_experiments/observation_reliability_20260928/sam2_projection_check/five_case_mask_projection.jpg`.

## Actual pose data flow

### RGB bootstrap

`preprocessing/tracking.py:109-128` seeds features **inside the eroded SAM2 appearance mask** and only keeps LK endpoints that remain in the next mask. This constrains feature locations to the mask but does not constrain the complete mesh silhouette.

At each frame (`tracking.py:164-187`):

1. start from the previous pose plus source RGB rotation edge;
2. flow existing 2D pixels;
3. run `solvePnPRansac`/LM on 3D points bound by the current mesh at an earlier pose;
4. run `contour_pose_proposal`.

The old contour proposal (`tracking.py:75-87`) uses only 32 sampled current rendered boundary points and 32 sampled target mask boundary points. It minimizes a robust 2D contour residual, plus rotation and translation priors. It has no hard mask containment/IoU term.

The final guarded path (`tracking.py:90-106`) checks a candidate only if the same tracking points get worse **and** the candidate also lowers a mask IoU computed in a hand-excluded domain. If tracking points get worse but mask improves, the candidate is retained. This is intentionally an output-repair compromise after the first repair caused BBQ recursive drift, but it means a current frame can still receive a poor raw-SAM2/full-mesh fit when the tracking and contour evidence disagree.

### Fixed-UniDepth registration

`surface_pipeline.py:40` computes RGB bootstrap once and then `visible_surface.py` receives the resulting `R,t`.

`surface_pipeline.py:77` asserts registration did not change R. Therefore the final visible-mask errors in these examples are already present in the frozen RGB rotation hypothesis or are produced by translation/scale afterward; the UniDepth stage cannot repair a wrong rotation.

`VisibleSurfaceRegistration` does include mask-derived terms, but not as a hard invariant:

- `visible_surface.py:153-163`: contour correspondences use only 64 sampled model/target boundary points and exclude uncertain hand pixels;
- `visible_surface.py:196-212`: contour residuals are soft 2px terms, mixed with 5mm surface residuals, cheirality, translation priors and temporal terms;
- `visible_surface.py:237-255`: a step is accepted when the aggregate merit decreases. It does not require every frame's raw SAM2 IoU to stay above baseline or even above a minimum.

The `measure()` IoU is an acceptance/diagnostic population score, not a hard per-frame mask constraint. In addition, current visible surface correspondence uses `observed_correspondences()` (`visible_surface.py:141-151`): missed rays are assigned nearest currently visible model points. That is appropriate for incomplete depth coverage but cannot recover a wrong orientation. The final `R` remains frozen.

## Where the current logic violates the intuitive contract

The intuitive contract would be:

> If a pose update makes the rendered full mesh leave a reliable visible-object mask, that update cannot be accepted.

The actual contract is weaker:

> If an aggregate robust objective improves, accept the shared translation/scale update; if a local contour update does not simultaneously contradict selected tracking evidence and its hand-domain IoU, preserve it.

This distinction explains why a low mask IoU can survive. It is not an input routing bug; it is a missing hard/conditional containment invariant combined with an upstream pose/shape ambiguity.

There are also two different visual claims that must not be conflated:

- mask consistency is necessary for a plausible visible pose;
- mask consistency is not sufficient to identify 3D rotation, because different orientations or wrong shape can project to similar silhouettes.

## Required next correction before any new pose search

Before implementing more PnP candidates, add a diagnostic/selection contract that is explicit about the mask:

1. Store three separate mask scores per frame and candidate: raw SAM2 IoU, MANO/owned-domain IoU, and boundary precision/recall. Never report only the hand-excluded score.
2. Keep a fixed raw-SAM2 population for candidate comparison. A candidate may not improve its score by changing the hand exclusion domain.
3. Do not make full-mesh containment an unconditional hard rule under occlusion. Instead distinguish: visible owned region must be covered; mesh outside raw mask is allowed only in a hand/background occlusion corridor or low-confidence shape region. The allowed corridor must come from source ownership evidence, not candidate depth.
4. For each candidate, require non-increase of raw/owned boundary error relative to the current pose before accepting translation/scale; if a shared scale/timeline update worsens a frame strongly, reject or downweight that update rather than hiding it in an aggregate robust merit.
5. For the upstream RGB pose, separate “track point remains inside mask” from “complete mesh silhouette fits mask”. Current tracking can satisfy the former while violating the latter. The latter must be a retained candidate score, not only a post-hoc report.
6. Keep rotations open for the next pose experiment. Adding a hard mask constraint only to the frozen-R visible-surface backend cannot fix BBQ/mug/spatula; it would only expose that their RGB bootstrap R is already wrong.

The minimal safe next experiment is therefore a **mask-contract audit/rejection probe**, not an immediate production change: replay the five RGB bootstraps and report where candidate/accepted updates violate fixed raw/owned boundary scores, then test a candidate that jointly optimizes 2D tracks plus fixed-domain silhouette with explicit non-regression. It must preserve milk f143/f148/f149, generate full videos, and report whether bad spatula frames are rejected before or after the R proposal. No source mask/depth/MANO changes and no GT selection.
