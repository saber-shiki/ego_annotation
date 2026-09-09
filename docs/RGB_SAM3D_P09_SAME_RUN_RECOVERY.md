# P09 same-run recovery (developer operation)

The active RGB case already completed P00–P08 and P09-propose, including genuine
UniDepth, HaWoR and SAM2 outputs. Canonical P09 never started: a just-written
anchor JSON caused the strict first-read metadata check to reject it. The old
recursive verifier also expands473 paths to about1.21million calls.

The repaired verifier accepts a metadata transition only after one immediate
full reread proves identical bytes, stable second metadata, and unchanged
inode/device. Per-operation DAG validation prevents repeated dependency
expansion, while independent calls revisit ancestors and post-command checks
force fresh byte reads. The read-only actual-anchor check completed in6.61s,
with no producer invoked or prediction data modified.

## Why an explicit continuation is needed

The stopped native executor's role required termination after a named phase
failure. A supervisor tool message cannot override that role. A fresh-start
launcher also correctly rejects this partially completed run. We therefore
create a narrowly scoped **P09 recovery launch**, not a second fresh run and not
permission to disregard future failures.

The birth `input/runtime_input_contract.json`, source video, current object plan,
anchor, original command logs, producer events and outputs are unchanged. The
new execution bundle must differ only in verification mechanics and runtime
launch instructions. Model source, checkpoint identities, consumer/solver code
and execution parameters must match. Copied model asset paths may relocate only
with otherwise identical configuration and byte-verified bundle assets.

`prepare_v19_run_recovery.py`:
1. verifies source and target bundles and the immutable birth manifest binding;
2. validates the actual existing anchor/producer chain with the corrected
   verifier against the original source bundle;
3. requires the recorded P09 verification blocker, completed upstream phases,
   and absence of canonical P09 output directory/producer log;
4. creates a **new** `state/runtime_recovery/P09/` directory containing byte
   snapshots of the original bundle manifest/config and original event prefix,
   plus a recovery record binding target bundle, original run identity, protected
   plan/anchor/blocker hashes and permitted phase.

It does not execute a producer, modify old records or provide an arbitrary
cache-registration interface. A failed preparation is retained, not overwritten.

## Runtime enforcement

`RunLineage` accepts the original code identity only for exact events from the
bound immutable prefix. It still verifies their outputs and recursive input
bytes using the original expected hashes. New events keep the same birth run
identity but additionally record their actual execution-bundle and recovery
contract hashes. Re-stamped old events, altered prefix/anchor/model assets,
unrelated phase/run/bundle identities or unbound new-code events fail.

The native launch helper validates the prepared recovery record, full target
asset inventory, unchanged raw video/CFR and absence of P09 execution before
emitting `mode: recovery`. The actor receives only this current-run recovery
contract and the curated runtime. It starts at P09, writes a new outer recovery
log and separate recovery handoff, and continues P10–P21 normally. It never
restarts P00–P08, re-authors the existing anchor, or treats this continuation as
another cold run. Managed file/command tools must be strictly sequential.

Any further named phase failure still stops that launch. Another failure is not
authorization to create additional transitions automatically. This first slice
supports only the present pre-invocation P09 verification failure, not arbitrary
phase resume or model changes.

## Commands (not an automatic launch)

```sh
python scripts/prepare_v19_run_recovery.py \
  --run-root /absolute/existing_current_run \
  --source-bundle /absolute/frozen_birth_bundle \
  --target-bundle /absolute/reviewed_corrected_bundle

python scripts/prepare_v19_native_launch.py recovery-request \
  --bundle-root /absolute/reviewed_corrected_bundle \
  --run-root /absolute/existing_current_run \
  --output-dir /absolute/new_developer_request_directory
```

The generated workflow is executed only through the native subagent API after
capability discovery and offline validation. There is no Pi/agent CLI fallback.

These checks preserve provenance across a known implementation repair; they are
not adversarial filesystem attestation, proof of visual inspection or successful
reconstruction. The goal remains actual SAM3D/core/demo execution and full-video
MP4/world-RRD delivery. Recovery tests and records are support evidence only.
