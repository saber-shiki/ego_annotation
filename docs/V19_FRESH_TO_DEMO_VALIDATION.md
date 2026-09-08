# V19 fresh → visual demo: reproducibility and transfer validation

This is a **development-side plan**, not an instruction packet for a runtime
agent. No fresh model inference was launched as part of branch publication.

## Four different claims

| Test | Allowed reuse | What it demonstrates |
|---|---|---|
| Frozen render replay | Published `demo_state.npz` + exact original video | Renderer reproduces the fitted geometry and presentation |
| Demo refit | Named frozen annotations/poses/mesh/HaWoR predictions, fixed MANO assets | Contact/visual postprocessing can be rerun |
| Complete V19 fresh + demo | Raw video, prediction-side calibration, code, fixed model weights/assets only | Upstream predictions and final demo can be regenerated without old prediction caches |
| Cross-video test | Same code/config policy and model assets, new raw videos | Transferability beyond the one validated milk case |

Success on the first two does not prove the third or fourth. A visually good
demo also does not imply physically certified contact/annotation readiness.

## Existing entry points, verified by source inspection

- `runtime/v19_runtime_spec.md`: named phase graph P00–P21. Its generic P12
  defaults to TRELLIS, not the SAM3D mesh used in this demo.
- `scripts/build_local_v19_runtime_bundle.py`: curated source closure builder
  with current-user path adaptation. This is an older base-bundle builder;
  do not assume its manifest alone satisfies the current stronger preflight.
- `scripts/build_hot3d_dual_backend_runtime_bundle.py`: clean-source,
  hash-bound overlay builder for the dual-backend runtime profile, including
  SAM3D native bridge scripts.
- `scripts/preflight_local_v19_runtime.py`: verifies bundle files, clean-source
  declaration, offline model assets, source-video identity, environments and
  CLI/self-tests. It is a real preflight, not proof that inference succeeded.
- `scripts/launch_local_v19_runtime_agent.sh`: legacy foreground launcher with
  machine/provider/model defaults. Treat it as a reference only; use the
  current approved agent protocol and explicit launch contract rather than
  silently executing old defaults.

The historical milk bundle at
`/mnt/user-home/kupingxin/ego_annotation_runtime/milk_unidepth_sam3d_ghost_lite_bundle_20260903T035100Z`
declares source revision `36e87ecdad656939ced4ac0b8c5ca4198848fd2e`, not this
published demo branch. Its script manifest does not include the new visual
contact fitter/renderer. Reusing it unchanged is not a clean rerun of the
current source revision.

## Recommended first fresh test: the same milk video

### 1. Freeze the configuration before running

- Pin the new published branch **by commit**, including transitive V19/V20
  scripts and selected runtime spec.
- Choose whether the target is default V19/TRELLIS or the **same SAM3D + demo
  mechanism**. Use the latter to reproduce this particular demo; do not replace
  mesh backends without recording a design/config change.
- Bind exact raw-video hash, source dimensions/FPS/frame IDs, sensor intrinsics
  and any declared image affine. Camera/object/hand GT sidecars must not be
  supplied as prediction inputs.
- Freeze model checkpoints, third-party source revisions, licenses, Python/CUDA
  environments, seeds, agent model/prompt and permitted judgment policy.

### 2. Create a clean curated runtime workspace

Build from the pinned clean source, not the developer's active working tree.
Include only the runtime prompt, template, one authoritative phase spec and
required script/config/assets. Exclude task memories, developer docs,
evaluators, GT files, old prediction outputs and research ablation objectives.
Licensed model files stay outside Git. Reuse of **model weight caches** is
permitted; reuse of **this clip's prediction caches** is not.

Use a brand-new run root and reject overwrite. Inspect every symlink/source
reference: no old masks, depth, HaWoR results, generated mesh, repaired pose
report, or demo state may be silently loaded from the historical run.

### 3. Complete launch preflight before the executor starts

Verify model paths/hashes, package/CLI imports, offline assets, storage, input
identity, GPU availability and spec closure. The current source preflight
requires `source_worktree_dirty: false` and explicit offline model assets;
older bundle manifests may not meet this contract. Resolve bundle/environment
problems outside the prediction run, not by asking the runtime executor to
install packages or patch itself.

Heavy phases run in an inspectable managed server tmux environment on an
explicitly allocated GPU (previously GPU5; recheck availability). A failing
agent launch/tool setup must be reported and retried through the same approved
protocol, not silently replaced by a foreground CLI.

### 4. Regenerate the complete prediction path

Execute the selected spec's actual dependency order:

- Raw-frame manifest; sensor-first camera contract when supplied (or the
  documented prediction-side intrinsics probe), then camera-conditioned depth.
- Fresh HaWoR camera/metric MANO inference and export.
- Fresh object plan, grounded detection and segmentation/tracking.
- Visible-surface observations and source/anchor selection.
- Fresh selected mesh reconstruction and canonical metric adaptation.
- Visible-frame pose fitting, temporal/full-timeline pose graph.
- Contact/occlusion/nonpenetration measurements and uncertain states under their
  original authority contracts, followed by render-state materialization.
- Actual full-duration V19 overlay/world/side-by-side rendering and inspection.

Follow the selected spec's subphase dependencies; do not execute numeric phase
labels mechanically (e.g. sensor calibration precedes conditioned depth).
Weak observations must retain uncertainty and continue where permitted; broken
coordinate/identity contracts are implementation failures, not alternate
sources of a successful-looking output.

### 5. Freeze V19 predictions, then run the separately labeled demo layer

The demo currently starts from a **V16 repaired pose/shape report**. A fresh
P15 output is not automatically that report. Before claiming fresh-to-demo
reproduction, implement/declare the adapter that binds the freshly generated
canonical mesh to the fresh pose report, and decide whether the tested pose
repair is a required logical stage. If required, run it on the fresh inputs;
never fetch the historical repaired pose as a shortcut.

Use fresh matching HaWoR parameters and MANO bridge. Check the demo input
profile (150 consecutive frames,960-square mask plane,both hands,compatible
MANO convention). Keep demo contact-fitting updates separate from canonical
V19 metric/physical state. Generated triangles may guide the owner-approved
visual fitting hypothesis, not become certified physical geometry.

### 6. Judge reproducibility on actual outputs

Preserve:

- input/code/model/environment/prompt hashes and each command/log/exit;
- evidence that all clip-specific predictions were regenerated;
- original-length V19 videos and final demo main/detail MP4 + world RRD;
- representative raw/overlay/geometry-contact frames, including failure frames;
- before/after contour, projection, continuity and pad-surface diagnostics.

Compare frozen results with the existing demo **after** prediction completes.
Do not tune fresh predictions against GT. Optional GT evaluation is a third,
post-freeze process and cannot modify prediction/demo state.

GPU and model nondeterminism, agent choices, encoder builds and RRD metadata
can prevent byte-identical output. Report separately:

1. same-input/same-code **functional and visual reproduction**;
2. numerical variation (pose/mesh/hand/contact deltas);
3. byte equality, if it happens to hold.

Do not call a render-only replay a fresh run, or a successfully completed phase
ledger a visually reproduced demo.

## Transfer tests after the first fresh run

The core method is reusable but arbitrary-video support is unverified. First
externalize/generalize frame counts, mask/render dimensions and hand-validity
handling. Keep actual object mesh and MANO; select contact hypotheses from each
new action's image evidence rather than forcing thumb contact universally.

Then run at least a second clear rigid-object grasp and a case with different
hand visibility/occlusion. Use one frozen config policy, record any manual
intervention, and compare full videos rather than only successful frames.
Articulated/deformable objects require explicit geometry/state models beyond
this single-rigid-object demo layer.

## Current status

- Demo fitting/rendering and focused tests: previously executed for milk.
- Publication cleanup runner: command planning, dry-run and failure reporting
  are tested separately; it does not claim a new full fit was run.
- Complete current-branch V19 fresh + demo: **not yet executed**.
- Cross-video generality: **not yet validated**.

A full fresh run is feasible as the next dedicated task, once the chosen mesh
profile, fresh pose-to-demo adapter and exact compute/environment preflight are
fixed. No estimate of guaranteed output quality or runtime follows from this
source inspection alone.
