# Pipeline V19 current epistemic state

## Current supported claim

For HOT3D `clip-001850`, the current accepted prediction boundary is the runtime-owned OWLv2/SAM2 branch with integrated support-gated P18 interval MANO, frozen after regenerated full-duration renders. The freeze manifest is `state/v19_prediction_freeze_manifest.json` with SHA256 `06c8ff45a955421b341ceafb0548b27d2d791e9ea6495f1fc6d17ee4db67c594` under run root `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1` (OPS 2026-06-27T05:48:12+00:00). It supersedes the earlier OWLv2 freeze `f0b563c7e...` for MANO evaluation because the earlier P18 allowed unsupported global hand translation.

The physical render claim is bounded: OWLv2/SAM2 repaired the prior broad table/hand/arm mask-support failure enough for the keyboard mesh to register to the visible key field/rim in direct frames such as 0032 and 0074, and support-gated P18 prevents zero-support contact/temporal terms from moving global hand translation. Current canonical render hashes are overlay `1cd1acf8f100908ad0cad42c6ef981ee65418f4a23fd2705697403ceded0d54c`, world `5de53e313fa738b9993fcd564bf22c67cb46752a65a8c55ef1d648579aa062be`, and side-by-side `bfc4c048766e618b590fc251ea417409b595c271f899e15e755e74621e7b4c31` (OPS 2026-06-27T05:48:12+00:00). This does not prove contact, signed nonpenetration, precise late-frame temporal completion, or full object-pose accuracy.

The current HOT3D 21-joint MANO quantitative claim is also bounded to `clip-001850`: support-gated runtime P18 restores wrist/root translation to HaWoR baseline and leaves only a small wrist-relative articulation effect. Over 300 hand/frame rows, HaWoR baseline median wrist error is `0.033586475 m` and support-gated runtime is exactly `0.033586475 m`; baseline median joint MPJPE is `0.047187152 m` and support-gated runtime is `0.046050225 m`; baseline median root-aligned MPJPE is `0.022665785 m` and support-gated runtime is `0.022970178 m`. The evaluator review sheet projects GT and predicted joints onto the visible hands, so the result is not a gross review-coordinate artifact (OPS 2026-06-27T05:52:43+00:00).

## Current causal model

Object registration and interval MANO are separate mechanisms. Object registration originally failed through segmentation-support contamination: VLM/click/box SAM2 prompts selected table/hand/arm support; OWLv2 text-grounded boxes changed the upstream evidence and produced localized keyboard masks. The remaining object-state uncertainty is late temporal completion and approximate broad geometry, not a proven contact/nonpenetration state.

The original P18 MANO failure was systematic, not normal measurement noise: contact/temporal/object terms moved the global wrist/root translation even when selected visible-surface support count was zero. That changed absolute MPJPE while leaving root-aligned MPJPE nearly unchanged. The support gate implements the physically narrower rule: global wrist/root translation may be changed only when direct visible-surface/depth-order support or another calibrated metric support factor exists; otherwise the source HaWoR wrist/root anchor is preserved and the optimizer may only contribute wrist-relative articulation.

The integrated runtime result matches that mechanism: after gating, wrist correction norm is zero for all scored rows, wrist error returns exactly to baseline, and the residual metric changes are small articulation/root-aligned changes. The gate is not a rule that translation is always wrong; it is a rule that unsupported translation is not physically grounded.

## Rejected boundaries and mechanisms

- Wrong raw-mesh freeze `daa3978f641691ef148720a75a9ec3bea35a22e55b6c01eb6a0eea0ed85f2a3c`: rejected because P18/P19 consumed raw P12 TRELLIS mesh while P15 poses were in P13 completed-canonical mesh coordinates.
- Mesh-frame-repaired freeze `2d999fbc4f51e375b4a9eb7277943bba74ce542cf0c613ece3479c888a267080`: rejected because the object still visibly followed contaminated mask/surfel support.
- Positive-click-derived prompt boxes and explicit agent-authored box prompts: rejected because they still allowed SAM2 to select broad tabletop/hand/background support.
- P09 960/1408 intrinsics/depth-size suspicion: ruled out for the OWLv2 branch; the earlier cyan surfel mismatch was diagnostic display scaling.
- HOT3D evaluator coordinate/review failure: ruled out by 1408-frame evaluator review sheets showing GT and predictions on visible hands.
- Pure articulation-error explanation for the original P18 regression: ruled down by decomposition and support-gated runtime scoring. Root-aligned errors stayed effectively fixed while absolute errors followed wrist/root translation.

## Live uncertainties

1. Generalization: the support gate is confirmed on fixed `clip-001850` only. Prediction: on intervals with zero selected visible-surface support, the gate should prevent MPJPE regressions by blocking unsupported wrist/root translation; on intervals with positive support, translation corrections may still be useful and need separate evaluation.
2. Broader HOT3D slice: Workbench item 6 requires the fixed 3-clip HOT3D slice, but only `clip-001850` currently has full runtime object/MANO artifacts and integrated support-gated scoring. `clip-001849` and `clip-001851` have baseline/pinhole evaluation inputs but not V19 support-gated runtime freezes.
3. Contact/nonpenetration: still uncertain. The gate prevents unsupported hand translation; it does not prove contact ownership, signed nonpenetration, or object-hand force consistency.
4. Full MANO surface metrics: unavailable for P18 interval states because the solver stores optimized joints and sampled vertices, not full optimized MANO vertices.
5. Object metrics: current HOT3D scoring covers 21-joint MANO only. Object pose/contact/occlusion metrics remain unimplemented or unsupported by the current evaluator path.

## Next action

Preserve the integrated support-gated evidence in docs/memory and commit the evidence checkpoint without staging unrelated dirty files. Then continue Workbench item 6 by launching the next fixed HOT3D clip through the same Pi-runtime pathway if the curated runtime bundle has the required input contract, rather than tuning further on `clip-001850`.
