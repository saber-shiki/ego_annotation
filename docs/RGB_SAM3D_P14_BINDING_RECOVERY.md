# Same-run P14 report-binding repair

The current run completed P09, P11 and fresh native SAM3D P12/P13. P14 then failed
before fitting because the SAM3D-only completion report supplied
`inputs.evidence_report` plus `input_hashes.evidence_report`, whereas the existing
strict P14 consumer requires `inputs.candidate_evidence_report`,
`candidate_evidence_report_sha256` and `selected_anchor_atomic_binding`.

The model, mesh, anchor and hash-bound P11 evidence already exist. The correction
must normalize this interface, not rerun SAM3D or remove its strict anchor gate.
A new deterministic binding producer writes an independent completion copy;
existing reports/meshes and physical authority remain unchanged. P14 uses a new
`visible_anchor_bound` output and phase log so the old empty failed directory and
traceback are retained.

## Second explicit transition, same birth run

The previous P09 recovery is immutable. `prepare_v19_p14_recovery.py` supports one
specific successor: a recorded P14 binding failure, completed current P09/P11/
SAM3D, an empty failed P14 output, and absent corrected pose outputs. It does not
provide arbitrary model-change/retry/cache-registration operations.

The new `state/runtime_recovery/P14/recovery.json` pins the predecessor P09 record,
previous execution manifest/config, current event prefix, original completion/
annotations/plan/anchor and P14 failure log/blocker. Existing events are checked
against the predecessor context before the new prefix is accepted. This handles
both birth-code events and P09-recovery-code events without rewriting either.
New events require the new execution code and P14 recovery hashes while keeping
the original run identity. Prefix changes, wrong predecessor, unrelated stage,
model/solver changes, or changed protected geometry fail.

Allowed differences from the previous execution bundle are limited to the
reviewed report normalizer, reconstruction/pose-dispatch wiring, verification
transition logic, failure-reporting logic and native instructions. External
model/tool identities and normalized execution configuration must remain equal.
The core P14/P15 solvers and their physical constraints are unchanged.

The native actor's explicit `recovery` mode starts at the record's P14 pose stage,
not P09 or P00. It retains all completed geometry, reads current-run evidence,
then continues P14b/P15, physical interaction, additive demo and final exports.
Any new failure during this launch is recorded exclusively at
`state/runtime_recovery/P14/failure.json`; original P09/P14 blockers remain
untouched. Original outer `logs/pose.log` is preserved; the continuation uses
`logs/P14_recovery.log` and a separate recovery handoff.

## Evidence boundaries

Unit fixtures exercise a birth -> P09 -> P14 chain with original event identities,
new successful receipts, a real subprocess failure, preserved original blockers,
wrong-predecessor/prefix/model negatives and separate native handoff routing.
Preparation and launcher fixtures do not execute prediction models.

The report-binding producer has a separate real-input check against the actual
failed completion/P11/P09 data. The current P14 consumer and P15 anchor gate must
accept the normalized copy without weakening checks; this is not evidence that
P14 optimization or full-video annotation has completed.

No original run identity, event prefix, source report, mesh, plan, anchor or
previous recovery record may be overwritten. The goal of the repair is to resume
actual pose fitting and finish the current full-video annotation, not to count
these support records as a reconstruction result.
