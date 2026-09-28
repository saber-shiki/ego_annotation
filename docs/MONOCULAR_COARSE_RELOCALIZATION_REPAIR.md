# Descriptor-keyframe relocalization repair (R7)

The R3–R6 attempts are retained, not promoted. They exposed implementation defects, not an established accuracy improvement:

- candidates were scored on different self-selected inlier sets;
- the optimized contour score did not belong to the pose actually returned;
- accepted poses could retain incompatible old 3D track bindings;
- selecting the nearest seven keys could remove genuine long-baseline references;
- the R5 25-degree/60-mm rule compared a correction to the current estimate, not actual adjacent-frame motion; its documentation was inaccurate. It is removed, not renamed a physical guarantee.

## Necessary intervention

Retain the R2 robust local tracking mechanism unchanged as a baseline. Add a separate offline relocalization stage using mutually matched interior RootSIFT descriptors across a temporally diverse keyframe bank (always retaining the appearance anchor). These are newly measured monocular image correspondences, not GT or historical optimized poses.

Each candidate is generated from training correspondences and evaluated on exactly the same held-out correspondences as the current pose. Positive depth, full-evidence reprojection, fixed pixel-domain silhouette support, and observed-surface disagreement are checked without fitting away a different evaluation set. Large corrective rotations are not forbidden merely by magnitude.

Accepted frame poses replace their stale local factor rows and receive explicit paired source/target keyframe reprojection factors. Rejected or unsupported relocalization leaves the local baseline untouched. Serialize only primitive diagnostics, retain all input frames, and preserve source measurement validity.

## Discriminating tests and claims

Tests must reject a wrong pose that explains a small subset while rejecting the rest, recover a large correct rotation, preserve input arrays, retain anchor/long-baseline keys, replace stale factors only at accepted frames, and exercise end-to-end export/JSON behavior. A low inlier-only median is not acceptance evidence.

Run all five frozen clips with one configuration, on the CPU server. Inspect complete videos and external evaluation only after prediction freezes. Improvements on these development clips are not held-out generalization or a physical millimeter certificate. No change to RL, source MANO, S3 or observation producers.
