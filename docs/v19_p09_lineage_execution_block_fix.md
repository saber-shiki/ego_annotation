# P09 lineage execution-block patch (2026-09-09)

## Defect and bounded intervention

The current `cold_raw_rgb_v3` run stopped before the P09 producer: the first
anchor byte read changed its CIFS metadata fingerprint. Subsequent preserved
reads have SHA-256 `4bc2cf425ea39c476161b327b4929b18b8cff9a1b9c261a90d4b63cf4296fa24`.
Separately, the recorded 473-file ancestor DAG implied 1,209,004 recursive
`require` calls; a prior unchanged-verifier observation timed out at 180 seconds
without a verdict. Cached bytes did not prevent repeated producer expansion.
These are execution defects preventing visible geometry from running, not
reasons to change existing predictions or their provenance authority.

Only `scripts/v19_run_lineage.py` changes production behavior:

- Stat before/after a full SHA read. A changed fingerprint permits exactly one
  immediate full reread. Accept only equal hashes and a stable second read,
  without device/inode replacement. Diagnostics retain all four fingerprints
  and both hashes. No waits, timestamp normalization, hash substitution, or
  unbounded retries. Original prepared/event/evidence hashes still apply.
- One synchronous verification epoch wraps each public `require()` or `inputs()`
  batch, including nested calls. Completed paths and producer closures are
  deduplicated separately. Each producer input edge still compares its own
  expected digest; each output still compares its recorded output digest.
  Active path and active producer stacks reject cycles, including sibling-output
  cycles. A producer is complete only after all its dependencies pass.
- Epoch digest lookups avoid repeated stats within the operation. All completion
  sets and epoch lookups expire on exit or exception. Only the existing
  fingerprint-checked byte cache survives within a verifier instance; subsequent
  public calls traverse ancestors again. `success()` still force-reads input
  bytes after a command, and raw-copy verification is unchanged.

This is not concurrent filesystem attestation: no command or concurrent writer
belongs inside an epoch. Stat-stable adversarial mutation is not solved by this
patch. The existing post-command force-byte check remains essential.

## Discriminating validation

If shared expansion dominates, producer-shaped DAG calls should scale with
unique producer-input edges plus requested roots, not multiplicative mask fanout.
If source bytes actually differ or metadata never stabilizes, reconciliation
must fail; it must not overrule a historical expected hash. If real verification
still fails, preserve that exact mismatch instead of modifying source artifacts.

`tests/test_v19_run_lineage.py` adds nine tests: metadata-only refresh, changed
bytes, identical-byte replacement, continuous metadata mutation, mutation during
reread, preserved raw/output/evidence expectations, conflicting producer input
expectations, sibling-output cycle, and a real tiny-subprocess P07-mask → P08 →
P09-proposal DAG. Existing changed-ancestor, cycle, copy, and post-command mutation
regressions remain. The DAG has 76 files, 52 producer-input edges, 25 roots:
77 calls (old recorded recursion: 32,525), 76 byte reads, 152 fingerprint stats,
509 total `Path.stat` visits.

Use the deployed interpreter, without installing anything:

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
"$PY" -m unittest discover -s tests -p 'test_v19_run_lineage.py' -v
"$PY" -m unittest discover -s tests -p 'test_v19_sam3d_runtime_closure.py' -v
```

Results: 23 lineage and 24 adjacent closure tests pass. Initial system-Python
closure checks lacked cv2/scipy; the deployed interpreter passes both, with no
dependency changes.

## Actual failed-anchor read-only check

Evidence directory:
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/p09_lineage_fix/`

`check_anchor_readonly.py` loads the corrected source with frozen-v5 helper
imports, and passes the frozen bundle/root/config with `asset_role='P09'`, exactly
the frontend's asset responsibility. `anchor_verification.json` records accepted
hashes and per-path byte-read counts. The script never invokes a producer.

Measured on `dexrobot-Standard-PC-Q35-ICH9-2009`: constructor 0.109 s; anchor
verification 6.611 s; 473 completed paths / 10 producers; 1,572 calls; 472 byte
reads (1,135,259,574 bytes), at most one per path; 945 fingerprint stats / 5,663
all-path stats. The raw-video digest was already checked during construction.
The anchor matches the preserved hash, the events SHA is unchanged, and P09
producer outputs/log remain absent. This timing is not a cold-cache benchmark.
The actual anchor was stable; the first-read refresh branch is exercised by the
regression fixture, not asserted to have recurred during this check.

Budget: hour 1 code/tiny tests; hour 2 read-only validation/review/commit, up to two
correction iterations. GPU/model runtime, full-pipeline/render runtime, and visual
review are zero in this patch. No annotation completion or visible improvement is
claimed. The remaining step is focused review followed by the parent's separately
authorized native same-run P09 continuation; no launcher, receipt, bundle, or
prediction changes are part of this patch.
