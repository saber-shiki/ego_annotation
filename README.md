# Grouped ego annotation

The grouped entry is `python main.py`. Numerical stages are owned by
`optimization/solve.py`; observation eligibility and physical acceptance remain
separate from candidate estimates.

## Cold raw-video entry

```bash
python main.py annotate --video INPUT_VIDEO --case CASE_ID --run-root NEW_RUN_DIRECTORY \
  --assets-manifest LOCAL_MODEL_ENVIRONMENTS_JSON --gpu EXPLICIT_GPU_INDEX
```

The run directory must not exist. Use the separately provisioned model-environment
manifest; the runner does not install models or download missing weights. It
copies raw input, produces new camera/depth/MANO/mask observations, executes the
independent Astra decisions, builds native complete geometry and observed/RGB
pose evidence, fits the visual scene, and writes `final/annotation.json`,
`final/world.rrd`, and `renders/annotation.mp4`. Per-frame visibility/contact
judgments receive every corresponding raw frame in bounded batches; final
inspection receives every rendered frame. Stage logs and producer receipts remain
inside this run. Do not use a GPU occupied by another job.

The cold entry is implemented and its CPU contracts are tested. A full model run
of this checkpoint has not yet been executed; it is not evidence of three completed
annotations. Standalone results explicitly retain `annotation_ready=false` and
`physical_contact_authority=false`. The complete exporter requires current decoded
RGB bytes; `mode="geometry_debug"` is explicitly partial and is never used by the
cold entry.

Raw-to-metric geometry keeps the producer boundary explicit: face-support reads
and hashes the exact P11 evidence bytes before computation and writes
`input_hashes.evidence_report`. The normalization gate verifies that declaration
against the selected P09 frame/surfels before adding P14-compatible aliases; it
never invents a missing producer hash. Existing completion reports missing that
field remain invalid. Generated render faces remain excluded from collision/sign
geometry.

## Same-run recovery and candidate attempts

The first attempt of each case must still use `annotate` with a new run directory.
For a transient failure, continue the existing run with the **same installed code,
model configuration and declared assets**:

```bash
python main.py resume --run-root RUN_DIRECTORY \
  --assets-manifest SAME_MODEL_ENVIRONMENTS_JSON --gpu SAME_GPU_INDEX
```

`resume` reads the original case, target, raw video and sensor identity from the
run contract. It verifies the full raw/input hashes, source hashes, model config,
declared model/source trees and successful stage receipts before skipping any
producer. Each v2 stage receipt also compares the **current planned contract**:
producer identity/type, executable and argv, cwd, critical execution environment,
GPU index/UUID request, output roles, explicit stage parameters and immutable input
prefix. Astra stages additionally bind the prompt, schema, fixed model/endpoint,
evidence JSON and ordered image paths/hashes. Planning never executes a producer.
A changed contract refuses reuse and requires explicit rewind at the first changed
stage or earlier. Reuse requires the same working directory/environment and GPU
request (including whether `--gpu-uuid` was supplied); changing `--gpu` requires
rewind at the first affected model stage, not merely at a later fit.
The run has one writer lock. The original Astra response and producer
ID are reused; no replacement API receipt is minted. The original GPU UUID is
retained and the next actual worker still performs the normal availability check.
No GPU is allocated for a fully cached replay.

Inspect `state/recovery/stages.json` for exact stage names and the complete ordered
prefix. To intentionally recompute a candidate suffix while retaining its verified
observation/native prefix:

```bash
python main.py resume --run-root RUN_DIRECTORY \
  --assets-manifest SAME_MODEL_ENVIRONMENTS_JSON --gpu SAME_GPU_INDEX \
  --rewind visual_fit --reason 'repeat the fit after a transient execution failure' \
  --candidate fit-attempt-b
```

Rewind invalidates the named stage and its ordered dependent suffix, leaving the
prefix unchanged. `quarantine/ATTEMPT/` keeps prior suffix artifacts, worker logs,
producer manifests, before/after stage manifests and the explicit reason. Missing,
changed or unreceipted partial output is also quarantined before the corresponding
stage is rerun. `state/recovery/events.jsonl` is append-only and records success,
failure and retained-producer reuse. Candidate identity and retained-versus-new
execution provenance appear in the final annotation. Native reuse means reusing
this run's successful byte-bound native producer, including its mesh and inputs;
there is no arbitrary old-mesh import.

Code/config/asset changes are rejected, including with `--rewind`; install a new
frozen version and start a fresh run instead. Pre-journal and v1 journals (including
8252 integration attempts without planned contracts) are refused without restamping,
even with rewind. Existing cases must keep using their own frozen build. Asset verification
reads all declared cold-path model/source files (and interpreter binaries), so
startup includes real checkpoint hashing; installed site-packages must remain
frozen separately. Credential file contents are not hashed or copied into the run. These rules
preserve the requirement that all three cases receive their own fresh observations.

Recovery CPU tests: `python -m unittest test_pipeline.StageRecoveryTests
 test_pipeline.MainResumeCLITests test_pipeline.RecoveryAssetTests -v` (join the
command onto one line). The CLI fixture uses clearly labeled synthetic producers
and a real three-frame decoded video, not a production annotation or API call.

## Prepared numerical entries

`python main.py refine-pose --annotations ANNOTATIONS --source-pose SOURCE
--rgb-edges EDGES_NPZ --rgb-report EDGES_JSON --object-id OBJECT --anchor FRAME
--run-root RUN --output POSTERIOR` consumes hash-bound, same-run metric/RGB
observations and runs local RGB seeding, robust SO(3), and refreshed full SE(3).
The posterior is create-only and explicitly lacks physical/contact authority.
Source arrays and their provenance remain unchanged. Fresh inputs require the raw identity
contract `input/runtime_input_contract.json` and role-bound producer receipts in
`state/producer_receipts.json`; RGB NPZ/report must come from one producer and
carry identical metadata. Foreign-run files are rejected even if byte-identical.
Inputs are prepared
prediction observations, not raw video or ground truth.

CPU regression: `python -m unittest test_pipeline optimization.test_optimization
observations.test_observations -v` (NumPy and SciPy required).

`python main.py fit-visual --help` exposes the prepared complete-MANO/object
optimizer. It preserves two-hand Adam variables, camera-origin hand scale,
unsigned top-three pad distances, source replay, supported-segment interpolation,
and the 12-iteration correspondence refresh. `render.export` produces a portable
JSON/RRD pair with complete triangles and explicit visibility clears. Native-shape
binding uses extracted front-depth helpers, not a dynamic research-script loader.

Full CPU regression: `python -m unittest test_pipeline factors.test_factors
optimization.test_optimization observations.test_observations -v`.
Set `EGO_MANO_MODELS` to licensed MANO assets to exercise the real two-hand CPU
optimizer smoke test; model assets are never downloaded by tests.

Thirteen independent judgments are available with `python main.py judge NAME
--evidence EVIDENCE_JSON --image IMAGE --output NEW_JSON`. Each carries its own
prompt and schema. `EGO_ASTRA_CREDENTIAL_FILE` must identify a user-owned,
owner-only credential file. Requests and response model/group are locked to
`gpt-6-astra`; `disable_fallbacks=true` is mandatory. Missing routes or mismatched
response models fail rather than choosing another model. The transport sends
actual image bytes and records their hashes. Credential contents are never stored
in output or configuration.

Main-environment package installation: `pip install --no-deps .`; provision the
listed core/visual dependencies in the isolated runtime first. Heavy observation
providers will use their separately declared environments.

Model workers now contain the actual HaWoR, UniDepth, OWLv2, SAM2, SAM3D and
TRELLIS implementations. Heavy dependencies are imported in the worker functions,
not at pipeline import time. Every worker requires provisioned model paths;
HaWoR gets a fresh private raw-video copy before any upstream detector/SLAM call.

`python main.py fit-formal --problem INPUTS_JSON` invokes the extracted full
per-hand interval solver through a named `FormalManoInterval` profile. The
HOT3D profile fixes object translation and enables its original full-K and
translation-support gates. Optional compositions are named, and numeric tuning
cannot switch Boolean factor families. This candidate still requires the formal
full-MANO acceptance stage before physical publication.

## Prepared formal/shared physical tail and evaluation

`python main.py formal-tail --spec INPUTS_JSON` consumes prepared prediction
observations, source pose, completion, depth, licensed MANO assets and independent
interaction judgments under one run root. Profiles are `formal`, `rgb_sam3d`, and
`hot3d_shared`. It executes the formal object graph, P17 evidence assembly,
per-hand interval solve, full-MANO acceptance, and source-named formal JSON/RRD/MP4
export. The HOT3D profile builds a shared observed-derived signed proxy; readiness
and local face authority still control every signed factor. This prepared entry
has not yet been model-run end to end; its CPU math and package contracts are
covered by migrated tests.

`python evaluate.py {boxes,mano,object,aggregate-boxes,aggregate-mano} --help`
exposes the five retained HOT3D evaluation/summary tools. Evaluation reads frozen
prediction plus explicitly supplied evaluation input; prediction `main.py` never
imports evaluation. `python -m observations.hand.wilor --help` provides optional
WiLoR prediction in its pre-provisioned environment; the default remains HaWoR.

The shared formal renderer consumes a bound accepted full-778 MANO archive only
when the actual P18b acceptance says accepted, signed-ready and not quarantined;
source surfaces remain separately stored. `render.render_dual_backend` branches
only complete SAM3D/TRELLIS display geometry after that shared solve and records an
identical shared-state hash for both outputs. Geometry records must bind the
shared package hash, mesh hash, canonical frame and `generated_display_only` role.


## Source organization and historical scope

The active package has four business directories (`observations`, `gates`,
`factors`, `optimization`) and five utility modules. Historical version chains,
runpy dispatchers, old batch scripts, copied runtime prompts, and experimental
standalone CLIs have been retired from the active checkout; baseline `12aede8`
remains available in Git and the original worktree. `migration-map.json` records
relative source symbols used during extraction; subsequent moves are in Git.

Default prediction remains HaWoR/UniDepth/OWLv2/SAM2/SAM3D with observed/RGB pose
optimization and unsigned visual fitting. Optional WiLoR and TRELLIS workers,
formal object/MANO profiles, observed-derived signed geometry, surface/visibility
inputs, full-MANO acceptance, complete layered exports and independent evaluation
are retained. Historical alternative depth/SLAM/hand/shape providers, articulated
part/deformable/variable-camera solvers, global/keyframe/late-window experimental
CLIs and GT benchmark suites are outside this version; their code remains in Git.

Use `main.py prepare --help` for optional observation builders and
`main.py preflight --assets-manifest ENVIRONMENTS_JSON` to validate actual
provider-package imports before allocating a run. No worker installs missing
assets or silently replaces a failed producer with empty observations.

Validation distinguishes CPU mechanism tests from actual model runs. The suite
includes full-RGB producer binding, OpenCV PnP and source/target Jacobians, observed
first-surface eligibility, exact K/radius rays, formal contact-state gradients,
active/dense normalization, real MANO replay when assets are supplied, shared
accepted-state export, full triangle RRD and MP4 rendering primitives. A passing
CPU suite does not establish cold-video accuracy or three completed annotations.


OWLv2 detector queries use its actual tokenizer/model positional budget, including
BOS/EOS tokens. `judge_object` requests short object noun phrases while preserving
the detailed object description separately. Before depth/MANO inference, the cold
entry tokenizes all phrases in the OWLv2 environment; over-budget phrases trigger
up to two same-Astra shortening judgments, each with original/effective text,
token IDs/counts, rationale and image provenance saved under `state/detector_text`.
No detector text is truncated. The inference worker rejects any remaining overflow
before loading model weights and pads valid mixed-length queries to the model's
actual positional budget.

Formal tail specifications may include `object_problem` with `profile` equal to
`observed_only`, `image_assisted` plus `image_factors`, or
`nonpenetration_assisted` plus `nonpenetration_targets`. `mano_problem` selects a
named interval profile, current-run `factor_reports`, optional `hand_ray_prior`,
and scalar `numeric_options`. These selectors do not accept arbitrary Boolean
factor combinations. `prepare hand-visibility` produces the actual report accepted
by the formal loader; image targets, visible-surface track and face-eligibility
builders expose the corresponding preserved argument schemas.

`python main.py render-dual --spec DUAL_JSON` renders two already-bound display
geometries against one immutable physical package. The specification names
`package`, `source_video`, `output`, and exactly two `geometries` records
(`sam3d`/`trellis`, mesh path/hash, package hash,
`canonical_frame=shared_completed_canonical_meters`,
`role=generated_display_only`). This entry does no second physical optimization.
The formal renderer accepts a distinct physical-completion report so display mesh
identity cannot overwrite signed-proxy readiness or collision-surface provenance.
WiLoR's optional worker is a raw hand-provider API, not a replacement for HaWoR's
world-camera trajectory and downstream archive contract.

See [MIGRATION.md](MIGRATION.md) for the callable capability matrix, final symbol
provenance, and explicit remaining equivalence gaps. The first raw entry is
create-only; public same-code resume and candidate rewind are described above.
Old-version or foreign-run prefix import remains unsupported. Do not claim
three-case completion from CPU/build evidence alone.

Rigidity review now judges the selected tracked whole, including attached parts,
using its object plan and paired raw/selected-track/object-owned crops for every
frame in ten-consecutive-frame sheets. The crop window is fixed within each
sequence; images are not motion-aligned. The judgment distinguishes observed
relative component motion from whole-object rotation, perspective and occlusion,
and preserves attachment/mechanism uncertainty. The standalone rigid gate remains:
articulated, deformable, rigid-part or unresolved results stop the run with the
original Astra answer saved. A rigid label that contradicts observed component
motion also stops. Richer evidence never licenses rewriting a previous answer.
