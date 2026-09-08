# Native RGB / SAM3D launch boundary

This developer document and `scripts/prepare_v19_native_launch.py` prepare **native subagent API requests**. They do not execute Pi through a CLI, start prediction scripts, or turn a prepared bundle into a claimed reconstruction.

## Registered actor

The project-discovered profile is:

`.pi/agents/ego-annotation.rgb-sam3d-runtime.md`

Runtime name: `ego-annotation.rgb-sam3d-runtime`.

The profile has explicit fresh context, `systemPromptMode: replace`, and disabled project/global/skill inheritance. Its tool allowlist is read/write/bash/ShellGate/supervisor, with the installed ShellGate provider explicitly loaded through `subagentOnlyExtensions`. A tool name alone is not a provider loader.

The profile is copied and hashed as part of a new curated candidate, not installed globally. The management `create` tool unexpectedly wrote to user scope despite a project-scope request during development; that newly created global profile was removed. Project-file discovery in the isolated launcher worktree and actual v2 candidate was then verified. The final workflow does not depend on that global registration.

This isolates **supplied agent context and role instructions**, not operating-system filesystem authority. Shell/read tools are not advertised as an OS sandbox. The actor is instructed to use only declared workspace/run/model inputs and never to modify code/assets, provision environments, download weights or spawn another agent.

## Tool-only probe

Use a new probe root and a separate developer request directory:

```sh
python scripts/prepare_v19_native_launch.py tool-probe \
  --bundle-root /absolute/curated_candidate \
  --run-root /absolute/new_probe_root \
  --output-dir /absolute/new_native_request
```

This creates a synthetic, non-case image and marker, then emits:

For a large-write stress probe, use `--probe-min-write-bytes 8192`: keep the short marker identity exact, and author non-measurement padding in the result. Transport verification compares the actual native `write` argument bytes with the file and native readback. `--probe-marker-bytes 8192` remains available for a separate large-marker echo test, but that also tests model transcription and must not be conflated with byte transport.
- `request.json`: exact native tool-call parameters;
- `native.workflow.js`: a single `runs.run` invocation of the registered actor;
- `registration.json`: profile/provider/bundle/contract/request fingerprints and **execution pending**.

The native supervisor must first discover the executable project agent with `subagent({action:'list',capabilities:true,cwd:...,agentScope:'project'})`, validate the workflow with `action:'validate'`, and then execute the emitted request with the native `subagent` tool. It must not launch a Pi/Claude/Codex/Cursor CLI or a foreground fallback.

The actor's `tool_probe` mode checks real cwd, creates a new non-adopted broker-managed tmux session, runs a short marker command, reads a marker and synthetic image with the actual file/image tools, writes and reads a result, and shuts down only its completed probe broker. No SAM3D, HaWoR, UniDepth, detector, segmentation, fitting or other prediction model is invoked. A tool/setup failure is a blocker to retain and report, not a reason to switch modes silently.

The probe's actual result and native transcript—not the generated request or an agent-list entry—are required to accept the launch/tool boundary. Offline script validation alone checks syntax and does not establish provider startup or image transport.

The workflow explicitly declares the contract's **absolute** `tool_probe_result.json` as its child output. Prediction uses `RUN_ROOT/state/native_prediction_handoff.md` for the final handoff, distinct from all phase outputs. This prevents the native harness's report-routing instruction from overriding the required probe/phase artifact location.

## Prediction request (separate authorization and execution)

First use the accepted runtime preparation entry to create a new raw-input run with its random current-run identity, source-video hash and bundle binding. Then:

```sh
python scripts/prepare_v19_native_launch.py prediction-request \
  --bundle-root /absolute/curated_candidate \
  --run-root /absolute/prepared_raw_run \
  --output-dir /absolute/new_prediction_request
```

The helper validates registered copied files/profile, a parsed prepared UUID, exact canonical run-root/bundle/raw-video binding and source CFR. An initial prediction request requires exactly the untouched preparation layout (raw input/contract, optional declared calibration, and the five empty output directories). P00 logs/state/receipts, P01 generated inputs, predictions, symlinks and undeclared files or directories are refused; recovery is a separate decision. Developer request directories must be disjoint from run/probe and bundle trees in both directions. The helper validates native provider identities against the prepared inventory and emits the same native-harness request shape, now in `prediction` mode. Authorized prediction-request preparation invokes the runtime asset verifier's full default path; scoped runtime checks remain separate. The synthetic tool-only probe checks only its actual tool-provider dependencies, not unrelated neural weights.

The actor receives only its mode, curated cwd, declared input-contract/run root and new dedicated tmux target. CASE_ID, copied INPUT_VIDEO and calibration binding are read from the immutable prepared input contract. It follows the authoritative runtime prompt/spec and produces all visual judgments from the current evidence. A completed tool probe is not permission to skip actual P00–P21 or to reuse previous clip predictions.

Initial launches and interrupted-run recovery are distinct: do not regenerate a new 'fresh' request around partially produced outputs. Any recovery must be an explicit same-protocol decision preserving the original run/receipts and partial state.

## Current evidence

Eleven small request/profile/isolation planning tests pass. The exact v2 request was offline-validated and the project agent was discovered as executable. The real native tool-only probe ran as workflow `9f7b2998-a5f4-4f22-8e28-feab145690bc`, child `932ee60f-b481-48c4-bd95-30081cb918f1`, and is **BLOCKED**, not accepted. Cwd, managed-shell marker and text read succeeded; the synthetic-image read failed with `managed socket client exited null`. Image description and output write/readback were not completed. No prediction model was invoked or alternate backend used.

The failure was reproduced against the installed broker: a nonblocking `sendall()` silently delivered only8064 of2000028 response bytes, with no final newline. The separate ShellGate worktree now has FIFO/write-readiness correction `dd4e03b8c438ee43b76c7ede4577fdb2119033fa`, with13 parent-rerun tests and independent **OK with notes** for the current native client. The global installed plugin remains unchanged. The profile and builder pin now target that isolated repaired provider; a new candidate/native probe is required before acceptance. The P2 write-half-close compatibility note is retained; the current native socket client does not half-close its request side. The failed probe's idle broker was inspected and then shut down through ShellGate solely for cleanup. No real prediction launch is claimed here.

The fixed-provider probe `fa32c276-88a3-4bd0-a831-69b93b0d3d6e` / child `8dff194d-9d40-45e3-ab6c-157321a71cff` subsequently read the large image, correctly described its pixels, and completed managed write/readback plus broker cleanup. However, the relative workflow output caused the harness's authoritative report path to override the contract's JSON target. That measured routing issue is now corrected with an absolute declared child output; a final exact-path probe is pending. Launcher review also found malformed/mismatched run identity, early-interruption freshness and one-direction directory-separation gaps; the current nine-test boundary correction covers them. No prediction model or cold pipeline has been launched.

The exact-path probe subsequently passed (see `native_launch/NATIVE_TOOL_PROBE_ACCEPTANCE.json`). Two real cold attempts then executed fresh depth/camera stages but stopped before HaWoR; their distinct causes and preserved evidence are in `RGB_SAM3D_FIRST_COLD_ATTEMPT.md` and `RGB_SAM3D_SECOND_COLD_ATTEMPT.md`. A larger handoff write exposed a separate PTY input short-write path. That repair is independently reviewed at isolated provider commit `8940bb17952e7e5dcfb4c039c397164de60a1f1d`, with32 transport tests passed. The current profile/builder pin target this new write-transport worktree; previous provider paths are unchanged. Native8192-byte marker write/readback remains required before the next cold attempt. No complete cold reconstruction is established.

The first native large-marker probe (`aae258f1`, child `3d834ff2`) wrote/read9524bytes exactly as submitted to the native write tool; the parent compared all three byte sequences and their SHA256. However, the model expanded the8192-byte input marker to8544bytes while constructing that write argument. The original full-marker echo requirement **failed** and is not relabeled passed. Its frozen result and independent transport comparison are retained in `native_launch/large_write_transport_comparison.json`. A new predefined probe separates short identity echo from authored large padding and validates the actual submitted payload; no measured/model arrays are entrusted to LLM transcription.

Developer evidence is under:
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/native_launch/`
