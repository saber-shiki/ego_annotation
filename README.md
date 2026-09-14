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
