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
The baseline reports are preserved unchanged; follow-up diagnostics belong to a
separate commit/report.
