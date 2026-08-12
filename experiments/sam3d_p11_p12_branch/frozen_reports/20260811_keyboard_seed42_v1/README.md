# Frozen SAM3D P11–P15 keyboard/seed42 experiment

This directory freezes the **additive, render-only** experiment that evaluates
SAM3D Objects as a P12 generated geometry prior against frozen TRELLIS.

## Scope

- Case: `hot3d_clip001851_keyboard_pinhole`
- Anchor: frame 109
- Seed: 42
- SAM3D upstream commit: `f91db411c50efee93d8db7aeb323885650f6f722`
- Geometry candidate: object-owned-mask SAM3D Objects
- Integration candidate: topology-preserving dual mesh
- Physical eligibility: generated faces are not collision/contact eligible and
  signed geometry remains disabled.

## Contents

- `reports/`: selected machine-readable reports and the full Chinese summary.
- `runbooks/`: exact shell launchers used for the major stages/finalization.
- `checksums/experiment_code.sha256`: hashes of experiment source files at freeze time.
- `checksums/output_artifacts.sha256`: hashes of retained external artifacts,
  including videos, meshes, reports, and review images. Large binary outputs are
  intentionally not copied into Git; their absolute source paths remain recorded
  in reports and this checksum manifest.

## Important limitation discovered during post-freeze review

The P15 green overlay is a projected observed metric mesh, not the raw per-frame
SAM2 mask. Any visible drift therefore requires separating SAM2 tracking error
from object-pose/camera/canonical-projection error. Likewise, handedness labels
are inherited from HaWoR detector classes and need an independent side audit.
The baseline reports are preserved unchanged; follow-up diagnostics are frozen
separately under `post_freeze_diagnostics/` and do not alter the baseline
prediction or its original artifact hashes.

## Post-freeze video diagnostics

`post_freeze_diagnostics/P15_MASK_DRIFT_AND_HANDEDNESS_ZH.md` and the paired
machine-readable reports record the evaluator-only follow-up requested during
video review.  HOT3D GT was accessed only after the prediction freeze.

Main findings:

- raw SAM2 keyboard tracking is stable against the official modal mask
  (150-frame median IoU `0.97055`); the conspicuous green P15 drift belongs to
  the projected observed-mesh/object-pose/camera chain (median IoU `0.62297`),
  not to the raw tracker mask;
- the object-owned mask is severely reduced by whole-hand-box subtraction
  (median retained raw-mask area `34.75%`), which is a separate ownership issue;
- the MANO bridge, source arrays, renderer, and camera transform do not swap
  left/right.  HOT3D box assignment favors the original labels in all 113
  paired two-hand frames, and a full side-swap counterfactual greatly worsens
  HOT3D MANO 3D errors.
