# WiLoR-only full preprocessing run record (in progress)

Date2026-10-15. The owner explicitly requires no HaWoR prediction residue and
subsequently authorizes per-stage resource-aware shared GPU use. This record is
not a completed-result report.

## Independent run, not a hand-only display substitution

- Worktree `/home/kupingxin/ego_annotation_wilor_only_preprocessing`.
- Branch `experiment/rl-wilor-only-preprocessing-20261015`, base4d8bb65.
- Frozen producer snapshot
  `/home/kupingxin/ego_annotation_runtime/wilor_only_full_preprocess_20261015_r1`.
- Actual source prediction root
  `/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/wilor_only_milk_observation_20261015_r1`.
- Full R6 / ordered-repair root
  `/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/wilor_only_milk_20261015_r1`.
- Compute host `dexrobot-Standard-PC-Q35-ICH9-2009`, A800 server. Device selected
  before each subprocess; explicit UUID and resource snapshot saved to stage logs.
  No other jobs terminated. Exclusive default behavior remains unchanged.
- Project tmux job `ego_annotation:wilor_only_full`. Separate post-freeze analysis
  job `ego_annotation:wilor_only_postprocess` waits on the terminal event, not a
  polling/sleep loop. Analysis refuses to run unless ALL prediction stages finish.
- Runtime log and exit sentinel:
  `/home/kupingxin/ego_annotation_experiments/wilor_only_full_preprocess_20261015/runtime_r1/prediction.{log,exit}`.

Raw input is the original150-frame/5second milk clip and its prediction-side sensor
K. No old observation bundle, hand archive, camera trajectory, metric scale,
object/hand mask, old generated mesh, ownership/contact factor, or coarse state is
an input. No case-specific prediction was copied into the runtime. HaWoR worker
module is absent; `hawor_root` and `hawor_python` are absent from assets. MANO,
DROID and WiLoR detector weights are independent model assets, not HaWoR state.
The DROID/detector model bytes were provisioned under neutral paths before the run;
their historical storage directory name is not a runtime dependency.

The existing schema has `hawor_world`/`source_hawor_npz` and bridge-array aliases.
Those names alone are not producer identity: WiLoR outputs use a distinct archive
path, model tag, own camera/MANO contract, and true canonical left/right FK. No
HaWoR proof flag is fabricated. New consumers validate the independently named
WiLoR contract; default HaWoR guards remain unchanged. The bridge file itself is
`mano_bridge_wilor_world.npz`. The final source manifest explicitly identifies the
legacy artifact slot as a schema alias, not a HaWoR prediction.

## What the command actually schedules

Fresh raw frames / Astra object plan / supplied-K contract / full UniDepth;
WiLoR full-frame camera-local MANO and own-image perspective translation;
OWLv2 / SAM2; independent standalone DROID + current UniDepth background scale;
canonical MANO world conversion; new base / full-surface ownership / eligibility /
P09 / RGB source motion / new native SAM3D / geometry binding / interaction
judgments and factors; newv2 source bundle. Then full R6 mask repair, shared shape,
relative depth and full MANO/object joint stages with complete renders.

No prior HaWoR/GT comparisons are loaded until the wrapper writes
`full_preprocessing.completed`. A new camera means this tests complete replacement,
not a controlled one-variable articulation ablation. Metric scale and dynamic
scene effects remain uncertain and are not hidden by importing an old camera.

The SAM3D stage resource floor is20GB on the owner-authorized shared policy; the
previous generic exclusive30GB threshold is not treated as a model requirement.
If actual model allocation or execution fails, preserve that real failure rather
than silently loading an old mesh or changing the producer.

## Pre-launch mechanism checks (not final pipeline accuracy)

- Existing grouped regression152 tests:148pass /4skip.
- Additional new independent-backend tests9pass.
- Camera/hand downstream consumer regression19pass.
- True licensed MANO random-pose left/right reflection + independent camera
  rotation replay through the unchanged `preprocessing.mano_repair.fk`:
  maximum vertex error left6.82e-7m, right7.31e-8m. Synthetic states only; not case
  predictions or model inference.
- Synthetic full-surface coupling uses real MANO assets, canonical export,
  base annotations, interaction/P18/data camera guards and depth-order ownership.
  All guards pass; actual hand triangles remove5388front pixels from49152object
  mask pixels. This verifies physical dataflow, not accuracy on the milk clip.
- Independent provider imports passed in provisioned interpreters, including
  standalone DROID core. No inference environment is provisioned inside the run.

Logs, exact snapshot hashes and smoke evidence reside at
`/home/kupingxin/ego_annotation_experiments/wilor_only_full_preprocess_20261015/`.

## Owner-authorized numerical replay allowance and R3

After observing R2's world conversion residual, the owner explicitly authorized
ignoring this micrometre-scale deviation and continuing. The independent WiLoR
canonical FK tolerance is now5e-6m (5micrometres), shared by the exporter, downstream
contract verifier and the WiLoR path of MANO repair. The HaWoR default remains at
2e-6m. Values remain logged; nonfinite values and larger discrepancies still fail.
No model weights, geometry, hand scale, source measurements, cost weights or GT
metric definitions changed.

Read-only diagnosis of all150R2frames found left vertex/joint residuals
2.195593e-6 /2.100436e-6m, right7.376290e-8 /5.787387e-8m. Axis-angle parameters
replayed through the existing solver FK differ by only5.960464e-8m for both hands.
This check wrote numerical diagnostic evidence only, no replacement predictions,
and did not read HaWoR or GT. Eleven numerical/independent-backend tests pass,
including acceptance of the reported residual, rejection above5micrometres, and
rejection of nonfinite/negative residuals.

Immutable R1/R2 code and failed outputs remain unchanged. A new R3 cold command
uses `ego_annotation_runtime/wilor_only_full_preprocess_20261015_r3`, creates
`ego_annotation_rl_runs/wilor_only_milk_observation_20261015_r3` and then
`ego_annotation_preprocess_runs/wilor_only_milk_20261015_r3`. It does not falsify
cross-code resume receipts or copy old prediction states. Jobs:
`ego_annotation:wilor_only_full_r3` and
`ego_annotation:wilor_only_postprocess_r3`. Control log:
`.../wilor_only_full_preprocess_20261015/runtime_r3/prediction.log`.

Evaluation waits for all prediction stages to finish, with separate R3 output
names (`r3_evaluation_r6`, `r3_evaluation_hands`, `r3_full_comparison`). **R3 is
running, not yet a complete full-pipeline result.**

## Analysis registered after successful freeze

Compare new R6 and final joint states with old HaWoR R6/final and GT. Produce
`evaluation_r6`, `evaluation_hands` and complete
`full_comparison/full_pipeline_{overlay,world}.mp4`. Every state uses full native
object triangles and778-vertex hands; one common world observer; one first-camera
SE3 per independent run, scale1, no per-frame/object fitting. Read-only evaluator
helpers are original-host dependencies, not prediction inputs. No contact or
zero-penetration certification; no RL training or default promotion.

At this writing, fresh depth, WiLoR, OWLv2 and SAM2 have produced real outputs;
independent camera and downstream stages still require execution. **No full-run
success or replacement conclusion is claimed yet.** Code is not committed,
merged or pushed. The previous component-study branch/artifacts remain untouched.

## R1 actual failure and R2 corrective execution

R1 exited1 at `independent_droid` after successfully tracking all150frames and
running its backend. Metric scale preparation then failed with OpenCV4.11:
`resize.cpp:4086: (-215:Assertion failed) func != 0`. The actual bound depth archive
is float16, which OpenCV resize cannot consume. This is an implementation dtype
boundary, not an accuracy verdict and not a reason to restore a HaWoR camera.
All R1 predictions/logs/snapshot remain unchanged. Its queued analysis stopped
without reading references.

Corrective mechanism: `resize_metric_depth` explicitly converts the current depth
raster toFP32 before interpolation. Tested on a syntheticFP16 raster AND the
actual first depth frame fromR1; dimensions, finite positive metre values and
outputFP32 were verified. Independent tests now10/10pass. No depth scale,
trajectory, or model result was substituted.

Fresh cold R2 uses `wilor_only_full_preprocess_20261015_r2` snapshot and new source
`wilor_only_milk_observation_20261015_r2`, with new derived root
`wilor_only_milk_20261015_r2`. It re-executes the whole raw-video observation path;
no cross-version receipts are restamped and noR1 predictions are copied intoR2.
Jobs are `ego_annotation:wilor_only_full_r2` and
`ego_annotation:wilor_only_postprocess_r2`; control log/exit now live under
`.../wilor_only_full_preprocess_20261015/runtime_r2/`. The new run starts only after
its provider preflight. **R2 is still in progress; full comparison is not ready.**

## R3 same-code continuation authorized at 2026-10-15 22:30 +08

R3's initial command stopped at21:28:15 with exit1, after20successful stages
through `surface`. The next `rgb_source_measurements` worker never started:
GPU utilization/free-memory changed between selection and launch verification.
Its failed log and initial control files are preserved. This was not an OOM or
MANO-replay failure; independent world export completed both150-frame hands
(left vertex residual2.211571e-6m, right6.662335e-8m).

Owner requested selecting available GPU resources and continuing from the saved
prefix. The producer snapshot, raw video, asset config and executable identity
were verified unchanged. `main.py resume` uses the existing same-run recovery
mechanism; no code/receipt restamping or manual prediction assembly was used.
Twenty original receipt hashes were retained and emitted `reuse` events. Failed
unreceipted files moved to the native recovery quarantine:
`quarantine/20261015T144256.100929Z-e3b5a092` inside the source run.

Two launcher/preflight attempts are preserved in `runtime_r3_resume1/2`. OpenCV
prepends its bundled library directory to `LD_LIBRARY_PATH` at import time; a
preflight must reproduce that import in its own child, not export its side effect
and then prepend it a second time in the main process. Attempt2 was stopped
before any producer stage. The final launcher restores the original pre-import
environment, checks the post-import fingerprint in a separate preflight child,
and invokes the unchanged main entry, which imports OpenCV once normally.

Active continuation: `ego_annotation:wilor_r3_resume3`; wrapper/log/exit live in
`.../wilor_only_full_preprocess_20261015/runtime_r3_resume3/`. At22:44, RGB source
measurements completed on physicalGPU4 (UUID
`GPU-6e6f03e5-1be9-e688-6141-e494294d389d`), launch free22629MiB/utilization0%,
required6000MiB/max85%. `rigidity_evidence` also completed. The active next stage
is `judge_rigidity`; complete reconstruction, R6 and joint repair remain pending.
GPU choice remains per-stage under the unchanged shared resource policy; other
processes were not stopped. Only our own environment-mismatched launch was ended.

`postprocess_r3_resume3.sh` waits on this continuation's completion event and
requires exit0 plus its `full_preprocessing.completed` sentinel. Its outside-run
HaWoR-final path typo was corrected to the existing `...20261015_r1/stage3_hands_r2`
using filesystem metadata only; no baseline/GT prediction content was loaded.
No complete-pipeline accuracy or replacement conclusion is yet available.
