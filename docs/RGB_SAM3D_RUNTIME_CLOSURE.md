# RGB / SAM3D curated runtime closure

This is developer-side preparation and contract validation, **not a fresh reconstruction**. No agent or model is launched by the commands below. The accepted integration replay remains separately documented; old predictions are not permitted inputs to a fresh run.

## Scope and execution chain

`run_v19_sam3d_integration_stage.py` retains geometry → pose → interaction → physical-render → additive demo → package export. Its camera entry now executes the actual producer dependency order:

- no supplied calibration: unconditioned P03a intrinsics probe → P03b resolver → conditioned P03 → P03c binding;
- supplied calibration: P03b resolver directly, with no fallback after a sensor error → conditioned P03 → P03c;
- P04 HaWoR receives full source `[fx,fy,cx,cy]`, a byte-identical isolated sequence and the source focal. Non-square supplied focal remains a model/API blocker, not an averaged substitute.

Source CFR is checked using decoded rational PTS before measurement, with the shared stdlib `v19_video_timing.py`. The raw manifest must match the source size, full frame/time timeline and video hash. `ffprobe` and `ffmpeg` are required before preparing a candidate.

P05 object plans, P09 visual anchor decisions and P11 case/object/anchor identity must agree. P17 requires agent-produced probability and support-uncertainty fields as declared by the existing consumer; the dispatcher does not fill them in. P14's mesh binding is normalized to `inputs.completed_mesh` for P14b without replacing geometry or converting latent poses into observations. P15 sees the full timeline. P16, P17, P18, P18b and P19 retain their original physical/sign and readiness rules. P18 requests the existing full-K MANO check.

The demo may consume ready or non-diagnostic **prediction** inputs through a narrowly opt-in allowance; shared provenance checks remain strict by default. No copied P15 readiness/diagnostic field is changed. The demo binding preserves the source hash and original state; only the newly fitted outputs have demo-only, non-physical authority.

Fresh output directories/files are refused on reuse. Each executed subcommand has a durable log, append-only event record and failure blocker; a failed command prevents subsequent commands. Clip paths, including embedded report references, must stay in the same run root, with symlink/relative escapes rejected. Heavy stages check the configured host/GPU and recheck the selected GPU instead of switching devices.

## Workspace preparation (developer only)

From the isolated runtime lane, using the existing environment:

```bash
cd /mnt/user-home/kupingxin/ego_annotation_worktrees/rgb_sam3d_integration_runtime_lane_20260908
/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python scripts/prepare_v19_sam3d_runtime.py bundle \
  --source-root "$PWD" \
  --asset-paths /mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/closure_inputs/runtime_asset_paths.json \
  --bundle-root /mnt/user-home/kupingxin/ego_annotation_runtime/rgb_sam3d_integration_closure_candidate_v2_20260908 \
  --report-dir /mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/runtime_closure
```

The destination must not exist. There is deliberately no replace/install option. The builder:

1. Validates the explicit asset-only config and required local model/checkpoint files.
2. Computes a script closure from the authoritative spec, Python imports, literal `script('name.py')`, path/`with_name`/importlib dependencies and the three required SAM3D experiment helper modules. The late-window initializer imports only its dependency-light helpers: its optional generated-validation CLI is not executed and does not require a historical candidate. No TRELLIS inference or A/B report is an executed dependency.
3. Copies only this code closure, the runtime prompt/spec/config and required SAM2/WiLoR/LightGlue code/model assets. It excludes developer instructions, task memory, tests, reports, full worktrees and old clip predictions.
4. Copies WiLoR model source from the explicitly authorized prior **curated model-source subtree**, not prior predictions. The P18 consumer loads `wilor/models/mano_wrapper.py`. MANO weights come from the canonical shared assets.
5. Pins existing SuperPoint/LightGlue Torch cache file hashes. `run_v19_offline_rgb_edges.py` intercepts URL checkpoint resolution and loads only those declared local bytes; missing/changed files raise rather than download. `TORCH_HOME` is explicit.
6. Preserves the supplied SAM3D environment overlay, without sourcing machine `paths.env`. Offline MoGe/DINOv2 binding remains enforced by the existing SAM3D runner.
7. Compiles the copied code and checks real imports from the isolated bundle/interpreters with `CUDA_VISIBLE_DEVICES=''`. No models are instantiated. It writes `runtime/bundle_manifest.json` and a developer-side `preflight.json` with exact import commands/results.

Shared external asset paths remain explicit read-only model/checkpoint/interpreter dependencies. This is not a relocatable environment installer. The bundle's manifest records copied-file hashes and source revision; it does not claim model numerical correctness or a successful prediction.

## Prepare a new raw input (developer only; no prediction execution)

Once a specific new raw video and case are authorized:

```bash
/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python scripts/prepare_v19_sam3d_runtime.py prepare-run \
  --bundle-root /mnt/user-home/kupingxin/ego_annotation_runtime/rgb_sam3d_integration_closure_candidate_v2_20260908 \
  --run-root /absolute/new/run_root \
  --video /absolute/authorized/raw_video.mp4 \
  --case new_case
```

Optional prediction-side calibration uses `--sensor-metadata`, `--sensor-source-video`, `--sensor-authority`, and optionally `--sensor-frame-intrinsics-key`. The associated source video's bytes must match the supplied raw video. No calibration is discovered from neighboring or historical directories. Sensor timeline/K validity is then checked by P03b. Supplied errors never fall back to an estimated camera.

Preparation verifies copied bundle hashes, decoded source CFR, and fresh-root refusal, then copies raw bytes and optional metadata into `input/`. It writes the launch contract without generating object plans, anchors, poses, contact judgments or other predictions. A failed/unchecked import preflight is not launchable. No raw video was supplied for a new prediction in this closure task, so no real case input/run root is created merely to demonstrate preparation.

## Native-agent launch contract (not executed here)

The supervising launcher must use the existing native agent harness, not a Python/CLI loop controlling an agent:

- **cwd:** the curated candidate directory above;
- **system prompt file:** `runtime/rgb_sam3d_system_prompt.md`;
- **execution spec:** `runtime/rgb_sam3d_runtime_spec.md`;
- **user input:** exact new `RUN_ROOT`, `CASE_ID`, copied `INPUT_VIDEO`, and `input/runtime_input_contract.json`;
- **configuration:** `runtime/runtime_config.json`, including the declared compute host and GPU;
- **workspace visibility:** only the curated bundle, new run root and declared read-only model assets;
- **containment:** a dedicated visible runtime session tied to this isolated cwd, with durable stage logs;
- **authority:** prediction execution only. Infrastructure repairs, installation and alternative model selection are not runtime tasks.

Do not expose developer docs, this reproduction guide, evaluation assets or old predictions to the native runtime agent. The agent follows P00–P21, makes the visual judgment artifacts, and stops on phase infrastructure/contract failures. It does not rewrite input readiness. No agent was launched by this task.

## Validation and remaining limits

Focused CPU tests cover the emitted arguments against actual parser declarations (including loop-declared options), camera dependency order/full-K binding, P14/P14b/P15 direct/latent rows, P19/P18b path/state preservation, failure stopping/logs, fresh output refusal, outside-root/history references, offline checkpoint behavior, ready/unready prediction-input authority, actual small demo fitting, and clean-bundle imports/closure.

Validation commands and actual results are retained under the integration output `runtime_closure/` and the worker handoff. The correction includes the separately authorized parent video-PTS/semantic-shader commit before real workspace preparation; see `RGB_SAM3D_VIDEO_REVIEW_FIX.md` for that independent change.

Remaining limits: no cold P00–P21 run, neural constructor/inference validation, GPU availability acceptance at future launch time, or new full-video physical annotation has been established. Approximate sensor/geometry failures remain uncertainties under the original mechanisms. Preparation and a validated script graph must never be labeled completed fresh reconstruction.

## Review correction: current invocation lineage and external identity

The v1 candidate stays frozen. The v2 candidate adds the following bounded anti-misuse checks; none is evidence of successful reconstruction or adversarial filesystem attestation.

- Bare clip filenames (`candidate.json`, `mesh.ply`, `clip.mp4`) in nested source references are rejected, as are relative paths and symlink escapes. Multiword explanatory prose and model configuration/weight names are not interpreted as clip filenames.
- `prepare-run` creates a new random UUID, binds the absolute run root, original video hash, optional sensor hash and exact bundle manifest. Copying old reports and their old execution records into this root does not produce the new identity.
- `run_v19_frontend_stage.py` is a **fixed** P01/P06/P07/P08/P09-proposal/P09/P11 dispatcher. It derives command inputs from the same-run manifest and visually authored P05/anchor/P10 records and exposes no arbitrary command, registration or cached-prediction option. The spec routes these phases through it rather than executing producers directly.
- Existing `run_command` records current run identity, code closure hashes, actual input hashes and output hashes only after subprocess success and required output existence. It covers fresh output trees (including masks, meshes and NPZs), excluding mutable command logs. P04 also captures its isolated sequence intermediates. Later consumers require and recursively verify the successful chain. Repeated file verification is deduplicated only while its stat fingerprint remains unchanged.
- Demo pose/mesh binding is now a named deterministic subprocess so its output has real invocation lineage; source readiness, geometry and physical state remain untouched.
- P05/anchor/P10/P17 documents instead declare `visual_judgment_provenance`, current run identity and hashes of the current phase evidence. No command record is fabricated for visual judgment. These declarations cannot prove the images were inspected; that remains the native agent's responsibility and the later visual/cold-run check.
- `v19_runtime_assets.py` enumerates declared external source/config/model files and records hashes plus resolved paths/tree membership. This includes UniDepth and OWLv2 local model files, UniDepth code, HaWoR source/dependencies/checkpoints/config/MANO data and Metric3D checkpoint, SAM2 checkpoint, SAM3D source and pipeline-referenced configs/checkpoints, MoGe, DINOv2 source/checkpoint, the two LightGlue weights, and interpreter executable/venv identity. It does not hash unrelated prediction directories, Torch caches, or all site-packages. Copied SAM2/WiLoR/LightGlue code and copied MANO assets stay protected by bundle hashes without dependencies on unused originals.
- Asset hashes and membership are revalidated at prepare-run and each stage entry before use; the separate nested SAM3D/RGB producers also verify their run inputs and assets. Same-path changed bytes or interpreter retargeting fail. This is not a portable environment installer or complete binary-library freeze.
- Import preflight forcibly disables CUDA visibility and Hugging Face networking after optional environment overlays and removes inherited `PYTHONPATH`. No model inference is invoked.

External asset verification reads approximately the declared model footprint, not arbitrary caches. Runtime cost must be judged on the actual cold pipeline; no runtime performance success follows from these records. The native registration/launch/tool-probe seam belongs to the separate supervising lane, with the approved project profile `.pi/agents/ego-annotation.rgb-sam3d-runtime.md` copied and hashed by the builder. Its exact fresh-context/tool/provider fields are validated, and the ShellGate provider, broker, package manifest and lockfile enter the external identity inventory. This inclusion is not a native launch/tool-probe result; those remain pending in the separate lane. This patch does not implement or invoke an agent launcher.

## Separate runtime-cost correction (candidate v2 remains frozen)

The v2 full external identity check measured 109.4 seconds and was not an acceptable default check for every phase. The follow-up keeps full verification at preparation/authorized launch (`verify_bundle` with no runtime role) and assigns fixed consumption responsibilities to runtime entrypoints:

- P01/P08/P09/P11, pose/initializer, interaction, physical-render, demo and export verify their interpreter identities; copied code/MANO/model assets remain covered by the bundle hashes. P06 additionally checks OWLv2 assets; P07 checks the SAM2 checkpoint; camera checks UniDepth and HaWoR assets.
- The geometry dispatcher checks only its interpreter. The nested SAM3D reconstruction entry alone checks SAM3D source/config/model bytes and DINOv2 source; it does not repeat a dispatcher-wide inventory. Existing fixed-hash MoGe/DINOv2 loader checks cover those checkpoints at consumption.
- The pose dispatcher never scans SAM3D/HaWoR weights. Its offline RGB loader retains actual per-load LightGlue hashes and does not rescan unrelated model trees. The initializer checks only its interpreter.
- Runtime roles are assigned by the fixed script/phase implementation. There is no user skip-cache CLI or mtime-only identity replacement. Every selected asset is still byte-hashed; full preparation checks still detect mutations of assets not consumed by the current stage.

Measured with the corrected source against the frozen v2 inventory, full verification took 95.971 s. Geometry dispatch took 0.055 s, pose 0.062 s, initializer 0.054 s, RGB entry 0.035 s, and export 0.052 s. The nested geometry verification took 52.940 s for its 12.50 GB/298-file subset; model-consumer checks still have an explicit substantial cost, but it is no longer repeated by the dispatcher or irrelevant phases. The measurement ran full verification first, then the scoped roles; it is not a cold-cache performance guarantee. Copied bundle bytes were verified in every measurement. This does not establish whole-pipeline runtime acceptance.

The cost fix is a separate source commit and review diff. Candidate v2 was read only and remains available to the separately owned native no-model tool probe. No new candidate or inference run was built/launched for this follow-up; incorporate the cost fix into a new candidate only after review.
