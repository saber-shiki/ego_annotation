# RGB manifest-plane correction

The RGB missing-pose mechanism was not yet executed when review caught a second
actual input contract error: P01 JPEGs are960×960, but the new dimension check
used P09's original-video source dimensions1408×1408. This would stop atframe0
after constructing the feature models. Recovery itself was independently
approved; the RGB mechanism remained blocked until this correction.

## Corrected planes and dataflow

`preflight_manifest_images` reads P01's declared frame rows before any feature
model constructor. It validates IDs/timestamps, original source/depth size,
actual decoded JPEG size against **manifest** size, and source/owned-mask
contracts where P09 exists. Missing-P14 target frames need no P09 geometry row
just to load their image. All selected JPEGs are preflighted before inference.

Untouched native masks feed exact P09 retained-depth replay. Separate masks
resized to the manifest raster feed feature matching. Source UVs are lifted into
the original depth raster using the existing half-pixel mapping and retained
support. Target PnP K is scaled to the manifest/matching raster. When serializing
reprojection evidence, UV and K are explicitly scaled back to the **original
owned-mask raster**, preserving the current material consumer's expected plane.
No source JPEG is blindly upscaled to original-video dimensions.

P01 manifest and P09 adapter report hashes join the RGB input provenance. The
named dispatcher passes both into actual command lineage. No P14 pose, geometry,
mask, depth, model weights or core physical/readiness gate is modified.

## Regression and actual-input evidence

- An exact1408source/960JPEG/960mask fixture includes a missing-P14 target and
  passes preflight without constructing models; a wrong manifest dimension is
  rejected.
- A second fixture distinguishes160source/depth,80manifest,40mask dimensions.
  Its depth has a known sparse quarantined outlier and smooth nonplanar surface.
  It calls the actual P09 replay and asserts the complete retained boolean
  raster, then feeds that support through feature-plane filtering, lifting and
  real OpenCV PnP. Expected world points/UV/K use independent pinhole equations,
  not the production lift/resize helpers. The outlier never reaches PnP; missing
  target pose is not invented; serialized UV/K match the40-pixel mask plane.
- The earlier PnP fixture now also computes expected coordinates independently.
- Read-only preflight of the actual current run passed **all150 JPEGs in2.83s**:
  source/depth1408×1408, manifest960×960, owned masks960×960. Frame0 and missing-pose
  frame30 both pass. Original annotation/manifest hashes remain unchanged.

Evidence: integration output `rgb_recovery_preparation/manifest_plane_tests.log`
and `actual_manifest_preflight.json`. This fixes the executable raster contract;
it is not a claim of new RGB edges, successful fitting or final annotation.
The runtime still needs to execute `pose-rgb` after the scoped re-review.
