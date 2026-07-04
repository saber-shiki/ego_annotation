# Demo pack — final epistemic state (19:40 CST, deadline 21:00)

## Delivered artifact
Release zip (227MB, 15 videos): 6 presentation (tomato/trash/origami polished + phone/window/scissors same-day), 6 QC (every presentation clip covered, per-frame provenance chips, each verified truthful incl. absent-hand/rejected suppression), 3 HOT3D benchmark GT-comparison videos (FIT/HELD-OUT badges), hero sheet, EN/CN docs incl. presenter brief + acceptance dry-run appendix, quant with all-clips-under-10 headline (regenerable, 0.0mm recompute deviation), throughput model with 3 measured same-day datapoints.

## Claims now supported (mechanism + evidence)
- <10mm on all 3 HOT3D clips both hands (5.1/6.2, 6.5/7.6, 4.7/7.3) under disclosed even/odd per-clip calibration, fixed a-priori config; mechanism = error is smooth camera-frame drift (rotation + translation-bias b(t)); disclosed-best 3.5-4.7 = interpolation floor.
- Same-day processing: 3 fresh clips end-to-end on one busy consumer 4090 (SAM2 48-108s, depth 110-209s, hands 77-128s per clip).
- Hand layers: all four ego clips detector-anchored (tomato 6.5/4.2px 845/871 accepts; trash v3c band-local; origami 410/450 after joint-label relabel; phone 8.4/13.7px); occlusion = ghosts, absence = absent.
- World views: trash drift-bent world corrected by R(t) (44.6->61.5% camera-above-hands, no wobble) — same mechanism class as the metric calibration.

## Decisive mechanisms found today (in repo memory, commits 8ce7b42/7b8ec13/3fa0358)
- Smooth-drift correctability (b(t)/R(t)) spans metric AND rendering domains.
- WiLoR joint-label unreliability under occlusion (labeled vs point-set residual 21.5 vs 11.5px) — new failure mode, per-frame assignment relabel fixes.
- Stale-validity gating bug class (2nd instance); band-local vs global fusion re-weighting.
- sh-4090 provisioning traps (zero-byte pythons); volcano A800 bare but 7 idle 80GB GPUs for future batch.

## Boundaries (disclosed in-pack)
Cross-clip calibration transfer undemonstrated (GT-free self-calibration = roadmap); same-day clips use reduced single-camera recipe; scissors object = sparse held-tool surfels; frozen-scalar repro needs licensed externals (HOT3D GT, MANO).

## Remaining
Phase 4 critic's 3 BLOCKERs + 1 HIGH fixed and re-verified in release; AF closed QC/metric coverage to 6/6; appendix scope-stamped. Remaining: delta gate-review -> delivery message. LAN A800 still down (v19 resume post-demo).
