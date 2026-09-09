# Same-run RGB observation continuation after successful P14

The current run completed the anchor-bound P14 fit:136 direct pose observations
and14 explicit missing-accepted-surface rows over150frames. The subsequent RGB
producer extracted150frames of features and failed because it required valid
initial poses for every selected frame. The camera, HaWoR, SAM3D mesh and P14 fit
are retained; no successful RGB edge bundle was produced.

## Observation mechanism

Calibrated source-world-to-target-image PnP does not require a target initial
object pose. The corrected RGB builder must use only eligible P09 metric source
support with a valid P14 source pose, while allowing missing-pose frames as
image-only targets. Existing rejected per-pixel P09 depth support must stay
excluded. Appearance matching and accepted source-depth support are different
inputs; a permissive appearance mask is not metric authority.

Missing nodes remain unresolved/latent until supported by RGB edges or the
existing temporal graph. No pose is fabricated or marked as a new direct depth
measurement merely to satisfy the old completeness assertion.

## Bounded continuation

`prepare_v19_rgb_recovery.py` supports only the explicit successor of the
verified P14 transition for this actual failure. It requires:
- the preceding recovery record, source execution bundle and original run;
- the successful bound completion and P14 pose receipts and bytes;
- the exact RGB missing-selected-frames failure log and exit1;
- absent old RGB bundle, corrected RGB directory and later pose outputs.

The new `state/runtime_recovery/P14_RGB/` record snapshots the existing event
prefix and binds all current completion/pose/annotations/anchor and failure
inputs. Prior birth/P09/P14 event identities are recursively verified before
accepting exact prefix events. New events must identify the current code and
recovery hash. No prefix or prior output is rewritten. Model settings/assets and
core P14/P15 solvers cannot change; only the scoped RGB observation, dispatcher,
verification and launch-instruction repairs are permitted.

The new native recovery launch starts the fixed `pose-rgb` dispatcher. It does
not rerun P14 fitting, anchor normalization, SAM3D, hands or perception. Corrected
RGB outputs use `rgb_full_timeline/` and a new producer log; old failure evidence
stays intact. It continues the existing initializer, multiview support, P15,
physical interaction, additive demo and final exports after successful RGB
production. Any new named failure stops that launch and is preserved separately
under the P14_RGB recovery directory.

This is same-run fault recovery, not a new cold run, automatic model replacement
or a generic cache-registration interface. Preparation and unit tests establish
only the intended continuation boundary; actual RGB edges, completed trajectory
and final mesh-driven videos/world recording remain required.
