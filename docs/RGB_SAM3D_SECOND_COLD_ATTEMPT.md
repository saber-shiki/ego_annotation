# Second cold RGB attempt: verified input copy and large-write transport

Run `390d3665-9c77-42e6-b36a-b38ef615c125`, directory `cold_raw_rgb_v2`, is frozen and **failed before HaWoR invocation**. P01 and all four P03 subcommands returned0. P04 input-lineage collection failed after0.202s, before a model return code existed:

```text
ValueError: input changed during byte verification: .../input/hawor_sequence/rgb_case_20260909_v2.mp4
```

The square-focal/source-plane correction from the first cold attempt worked. No P04 hand prediction, object reconstruction, physical interaction or final video/RRD is claimed by this attempt.

## Actual storage diagnosis

Parent read-only inspection found that both the source RGB and HaWoR copy have the expected original SHA256 `43513ed7534f8d32ede3ec13ad8c5b92d41487705064081491ab7831ee0e1b59`.

Five new diagnostic copies on the same CIFS filesystem (outside the frozen prediction run) reproduced a metadata transition during the first immediate read: mtime and ctime moved backwards by approximately12–16ms, while inode, size and SHA256 were unchanged. Each second read was stable. The original runtime exception did not record its before/after fingerprints, so this is a reproduced storage behavior consistent with the failure, not a claim that the executor itself measured those values.

## Minimal producer correction — strict verifier unchanged

`copy_verified_raw_video` now hashes the **new destination copy** and compares it with the immutable prepared input SHA before the copy enters lineage validation. Previously the code copied the bytes but computed the expected environment hash from the original source only, leaving the new file's first read to the stat-sensitive lineage cache.

This is an explicit byte-identical copy check, not ignoring ctime/mtime or accepting unverified bytes. `v19_run_lineage.py` is unchanged. A corrupt copy still fails, later content changes still fail, output reuse still fails, and post-command source checks remain.

On the actual CIFS mount, three paired reproductions using the unchanged strict `RunLineage.checked_digest` showed:
- old unverified copy: 3/3 rejected despite matching SHA;
- destination-verified copy: 3/3 accepted with matching SHA.

Focused tests also assert that the destination—not merely the source—is hashed, deliberate corruption stops before P04, and modifying the verified copy remains rejected. These tests do not run HaWoR.

Evidence is under integration output `cold_v2_failure/`: `existing_copy_checks.json`, `stat_copy_repro/results.json`, `verified_copy_repro/results.json`, `camera.log`, and `verified_copy_tests.log`. The diagnostic copies are not inputs to a future prediction run.

## Separate managed input-transport blocker

The stopped native agent then tried to write a6149-byte handoff. That tool operation timed out and initially produced no file. Parent inspection found the owned broker and idle shell alive, with no model jobs. The blocker JSONs from the Python dispatcher were already present. After explicit cleanup approval, the agent shut down its broker and returned a final handoff for host persistence; no alternate route or prediction retry occurred.

The provider previously wrote an encoded command block to a nonblocking PTY with one unchecked `os.write`. The separate isolated provider repair is now committed at `8940bb17952e7e5dcfb4c039c397164de60a1f1d` after a focused deadline correction and independent OK review. It drains pending input with writable readiness, retains partial writes, and distinguishes incomplete-delivery failure from an already running command timeout. Real source-broker/bash/PTY cases reproduce the old6149-byte and64KiB failures and pass exact-byte/hash/follow-up checks with the repair; all32 transport tests pass. Old provider paths and frozen candidates remain untouched. This tool repair still requires an actual native large-write probe before another cold run.

No cached prediction, model replacement, threshold relaxation, metadata-only acceptance or completed-reconstruction claim is authorized by either repair.
