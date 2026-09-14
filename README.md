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

Formal object/shared-tail acceptance, optional WiLoR, evaluation consolidation,
remaining owner deduplication and final legacy removal are still under migration. The historical documentation below
is retained only during extraction and does not describe the new CLI.

---

# Ego Annotation — visual-contact demo branch

This branch preserves the V19/V20 research pipeline and adds a **demo-only
hand–object visual fitting and rendering path**. The delivered example is a
150-frame milk-carton clip. It is not a claim of measured physical contact,
annotation readiness, or arbitrary-video support.

- **[Standalone RGB scene v7 baseline](docs/STANDALONE_RGB_SCENE_V7_BASELINE.md)**: frozen research baseline, external artifacts and create-only launch preparation.
- [Developer-only v7 publication evidence](docs/STANDALONE_RGB_SCENE_V7_RECORD.md): exact frozen identities, prerequisites, actual validation and known limits.
- **[Demo replay and refitting](docs/V20_VISUAL_CONTACT_DEMO_REPRODUCE.md)**:
  required inputs, verified case limits, environment notes, commands and outputs.
- [Demo design](docs/V20_VISUAL_CONTACT_DEMO_DESIGN.md) and
  [visual/numerical results](docs/V20_VISUAL_CONTACT_DEMO_RESULTS.md).
- **[V19 fresh-run reproducibility plan](docs/V19_FRESH_TO_DEMO_VALIDATION.md)**:
  distinguishes raw-video prediction, cached demo refitting, render replay and
  cross-video transfer.

The independent delivered artifacts are stored outside Git:

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/demos/milk_visual_contact_20260908/
  milk_hand_object_demo.mp4
  milk_contact_detail.mp4
  milk_hand_object_demo.rrd
  contact_keyframes.jpg
  contact_before_after_120.jpg
  state/demo_state.npz
  SHA256SUMS
```

Both videos are 1920×960, 30 FPS, 150 frames / 5 seconds. The detail video's
reconstruction is explicitly a +20° inspection view, not a second observed
camera. The original V16 archive and formal annotation state remain unchanged.
Raw videos, predictions, licensed MANO models, model weights and MP4/RRD files
are **not distributed through this branch**.

## Code entry points

```text
scripts/run_v20_visual_contact_demo.py     # explicit replay or fit-render launcher
scripts/fit_v20_visual_contact_demo.py     # actual MANO/object fitting
scripts/render_v20_visual_contact_demo.py  # combined triangle-depth rendering + RRD
scripts/v20_demo_geometry.py               # camera, MANO and triangle utilities
```

The launcher is **not** a replacement for the V19 phase graph. It never
installs environments or starts an inference agent. Fitting must run on an
explicitly selected server/GPU in managed tmux; use a fresh output directory.

## Focused tests

Run with the preflighted demo Python environment:

```bash
python -m unittest discover -s tests -p 'test_v20_visual_contact_demo.py' -v
python -m unittest discover -s tests -p 'test_v20_demo_runner.py' -v
```

The first suite checks geometry/projection/occlusion and Torch–NumPy transform
parity. The second checks launcher arguments, fresh-output protection, dry-run
behavior and subprocess failure reporting. Tests are necessary but do not
replace inspection of full videos and contact close-ups.

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
