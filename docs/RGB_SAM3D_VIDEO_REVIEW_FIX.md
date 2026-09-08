# Package-video review corrections

Scope: `render_v19_annotation_layers_video.py`, a stdlib-only `v19_video_timing.py`, a small shared `shade()` correction, and focused tests. No fitter, mesh, pose, camera or source annotation updates.

## P1: actual CFR mapping

The previous half-frame annotation tolerance could accept `[0, 0.045, 0.0666667]` at30FPS and encode a different middle timestamp. It also trusted average FPS/frame count without reading source PTS.

The CFR-only writer now:
- probes every source frame's explicit integer PTS (`pts`, or FFmpeg4.x's `pkt_pts`) and final frame duration;
- rejects missing/inferred PTS, duplicates and non-CFR timing; `best_effort_timestamp` is not used as proof;
- checks source PTS against the exact rational frame cadence, permitting only floor/ceil clock-tick rounding when a cadence is not exactly representable; at integer tick positions no one-tick tolerance is granted;
- checks the package's nominal CFR grid within1microsecond, independent of source clock resolution;
- passes the exact rational rate to ffmpeg, reads the encoded output PTS back, and records source/package/output origins explicitly instead of claiming their absolute timestamps are identical;
- performs input timing rejection before creating output or launching the encoder.

Tests include the reviewed nonuniform package, a genuine generated VFR MP4 whose average FPS/frame count still equal30/3, exact30000/1001 output, nonzero package time origin, quantized/coarse clocks, missing/conflicting PTS aliases and incompatible final-frame duration. The initial run exposed installed ffprobe4.4's `pkt_pts` naming; explicit alias support fixed this without falling back to inferred timestamps.

## P2: stable hand shading

When semantic labels are present, smooth hand normals are now selected by labels1/2 rather than BVH insertion index. An object may be missing while the valid hand occupies index0. No-label callers retain the old object-at-index0 convention.

The right-only regression uses two connected noncoplanar triangles, casts actual rays, then adds an off-screen object. Hand pixels must remain identical when its insertion index changes; geometry and visibility do not change. A single planar triangle would not expose this defect.

## Validation

-12focused video/timing tests pass, including real ffmpeg encoding, VFR construction, ffprobe PTS inspection and triangle shading.
-86unittests plus22late/12keyframe function tests pass: **120focused cases**. Not a full pytest claim.
- Three mechanism regressions run against pre-fix e420f51 code loaded into memory produced the expected three assertion failures. Repository files were not reverted for this negative control.
- Black25.1.0 formatting of the new video/timing modules/tests was checked AST-identical before/after formatting. Shared shader change remained a small targeted edit.

## Existing replay is not retimed or overwritten

The actual milk source and existing150-frame MP4 were both decoded by ffprobe. Both have30FPS, timebase1/15360, firstPTS0,150frames, duration5s and **zero** CFR grid error. Their complete PTS sequences match; package grid error is also0. Thus the general P1 was reachable but does not invalidate the timing of that specific replay. Existing source/package/MP4/RRD bytes remain unchanged.

Evidence under:
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/video_review_fix/`

This patch does not establish fresh prediction, contact accuracy or RRD GUI acceptance. Independent re-review is required before generalized exporter acceptance.
