# SAM2 prefix propagation repair

## Observed failure and causal mechanism

Frozen integrated source25f28ee, full150-frame milk run
`rlopt_fixed_25f28ee_20260916/milk_full` failed candidate-mask identity.
All27 rejected frames were exactly0..26, with no mask file. Selected prompt
frames were27,54,81,121,135,149. All123 remaining masks had detector-box
IoU>=0.7453. The production wrapper called `propagate_in_video(state)` once.
The deployed predictor defaults to forward traversal beginning at the earliest
conditioning frame, not video frame0. Thus the prefix was never inferred and
was incorrectly serialized as invisible. This is a missing computation, not
permission to weaken identity thresholds or reinterpret detector boxes as masks.

## Minimal intervention

Keep the original forward traversal and all of its outputs. If the earliest
prompt's local extracted-frame index is greater than0, call the same predictor's
real reverse traversal from that prompt through local frame0. Merge only prefix
outputs; preserve the existing forward anchor result. Translate local output
indices through the existing selected source-frame list (including nonzero
source offsets). Record propagation direction/counts in the producer summary.
A missing returned frame is a mechanical producer failure, distinct from a
returned empty mask, which remains an uncertain/nonvisible measurement.

No new prompts, frame0 reseeding, detector changes, model substitution, mask
copying, identity-threshold changes or edits to old prediction artifacts.

## Validation and claims

CPU fake-provider tests exercise the helper and the actual worker run with
normal mask/report/video writers: late first prompt, zero prompt, nonzero source
indices, boundary overlap preservation, missing or invalid provider outputs.
A separate server SAM2 diagnostic may consume the frozen run's prompt evidence
as read-only input but writes new diagnostic masks outside it, with no new
production receipt. Check actual prefix overlays/identity afterward.
Only then freeze new code and start another full raw-video run. No cross-code
resume/import of old observation receipts; complete annotation feasibility is
not established by prefix repair alone.
