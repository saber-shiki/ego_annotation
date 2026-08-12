# Experimental P11/P12 SAM3D Geometry Backend Branch

This directory is an additive experiment. It does **not** modify or replace the
canonical V19 P11/P12 scripts, runtime specification, TRELLIS output, or SAM3D
upstream source.

## Mask provenance (no dataset-GT leakage)

SAM3D Objects does not predict its own mask: its public API is
`Inference(image, mask, seed, pointmap=None)` and embeds the caller-supplied mask
in RGBA alpha. For this experiment the supplied mask is prediction-side:

1. text-grounded OWLv2 boxes from the runtime RGB;
2. SAM2 video propagation (`sam2.1_hiera_small`);
3. V19 hand-ownership subtraction using the prediction-side hand box, removing
   4,056 of 67,917 pixels at frame 109;
4. the resulting 63,861-pixel object-owned mask is shared by TRELLIS P11 and the
   corrected SAM3D P12 branch.

No HOT3D object mask, CAD silhouette, released pose, or evaluator GT is consumed
by P11/P12/P13 prediction. Therefore this is a fair shared-upstream generator
comparison, but it is not a claim that SAM3D alone performs segmentation.

## Strict blocker reduced

The archived SAM3D/TRELLIS comparison did not have one authoritative P11 input
contract:

- canonical TRELLIS consumed the P11 object-isolated RGBA crop built from the
  object-owned mask;
- the archived SAM3D invocation consumed full RGB with the pre-ownership SAM2
  mask.

That confounds geometry-backend attribution. The two models also require
intentionally different native conditioning formats, so forcing both to consume
the same crop would be another invalid simplification.

## Mechanism

### Experimental P11

`build_p11_dual_geometry_inputs.py` consumes an existing canonical P11 evidence
report and materializes an immutable, side-by-side input bundle:

- TRELLIS: the existing object-isolated RGBA crop, copied byte-for-byte;
- SAM3D Objects: the selected full runtime RGB and the exact object-owned mask,
  both copied byte-for-byte;
- no SAM3D crop, rotation, rectification, pointmap injection, or RGB masking;
- hashes, dimensions, mask provenance, and a visible mask-review image.

By default it rejects a selected mask that is not byte-identical to
`selected.visible_geometry_candidate.mask_path`. That prevents a raw SAM2 mask
from silently entering the SAM3D branch again.

### Experimental P12

`run_p12_parallel_geometry_priors.py` consumes only the experimental P11 report.
It:

- runs the existing external SAM3D runner in the unmodified SAM3D environment;
- passes full RGB + object-owned mask and leaves `pointmap=None`;
- does not rerun or mutate TRELLIS; an existing TRELLIS report is a frozen
  reference;
- emits a backend-neutral report containing raw priors only;
- explicitly marks both candidates as non-metric, non-canonical, and not
  collision-ready.

P13 alignment/fusion, pose fitting, and hand-object integration are out of scope
for this patch.

## Causal predictions

1. If the corrected object-owned-mask SAM3D output removes front/hand-associated
   geometry while retaining its prior silhouette/completeness advantage, the
   previous input-mask mismatch was a material confounder and SAM3D remains the
   preferred P12 candidate.
2. If geometry quality degrades substantially, the mask semantics require a
   controlled raw-SAM2 versus object-owned ablation; it does not justify silently
   switching masks in production.
3. If the raw SAM3D mesh is good but later P13 integration is poor, the blocker is
   the geometry adapter rather than P12 generation. The raw P12 artifact remains
   frozen for that diagnosis.

## Self-test

```bash
cd /mnt/user-home/kupingxin/ego_annotation
.venv/bin/python experiments/sam3d_p11_p12_branch/self_test.py
```

The self-test is CPU-only and uses synthetic files plus reused fake P12 reports.

## Keyboard experiment: P11

```bash
REPO=/mnt/user-home/kupingxin/ego_annotation
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
EXP=$RUN/experiments/sam3d_p11_p12_native_owned_mask_v1

$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/build_p11_dual_geometry_inputs.py \
  --evidence-report \
    $RUN/measurements/geometry_completion/rigid_evidence/hot3d_clip001851_keyboard_pinhole/keyboard/evidence_bundle/evidence_bundle_report.json \
  --comparison-mask \
    $RUN/measurements/object_tracks/sam2_owlv2_box_points/keyboard/sam2/sam2_masks/000109.png \
  --output-dir $EXP/p11_dual_inputs
```

## Keyboard experiment: P12

Run from the A800/server host. The command below invokes SAM3D through its
separate micromamba Python and references the existing TRELLIS result read-only.
Choose a GPU only after a one-shot `nvidia-smi` preflight; the wrapper requires
30,000 MiB free by default and never waits or polls.

```bash
source /mnt/user-home/kupingxin/sam3d-objects/activate.sh
SAM_PYTHON=$CONDA_PREFIX/bin/python
GPU_ID=7  # example only; replace after checking for at least 30000 MiB free

REPO=/mnt/user-home/kupingxin/ego_annotation
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
EXP=$RUN/experiments/sam3d_p11_p12_native_owned_mask_v1

$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/run_p12_parallel_geometry_priors.py \
  --p11-report $EXP/p11_dual_inputs/p11_dual_geometry_inputs_report.json \
  --output-dir $EXP/p12_parallel_priors \
  --sam3d-python $SAM_PYTHON \
  --sam3d-runner $REPO/scripts/remote_run_sam3d_objects_mesh_v7.py \
  --sam3d-repo /mnt/user-home/kupingxin/sam3d-objects/src \
  --sam3d-config /mnt/user-home/kupingxin/sam3d-objects/src/checkpoints/modelscope/pipeline.yaml \
  --cuda-visible-device $GPU_ID \
  --trellis-report \
    $RUN/measurements/geometry_completion/trellis_keyboard_seed42/qc_trellis_shape_v3.json \
  --seed 42
```

The SAM3D upstream checkout is not edited by either command.

## P12 raw-mask A/B

After P12 succeeds, compare the archived raw-SAM2-mask output with the corrected
object-owned-mask output before any P13 alignment or fusion:

```bash
$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/render_p12_raw_sam_mask_ab.py \
  --old-sam-report \
    $RUN/experiments/sam3d_vs_trellis_20260803/sam3d_raw/keyboard_anchor/qc_sam3d_objects_mesh_v7.json \
  --new-p12-report \
    $EXP/p12_parallel_priors/p12_parallel_geometry_priors_report.json \
  --output-dir $EXP/p12_raw_mask_ab \
  --target-faces 60000 \
  --surface-samples 60000 \
  --profile-bins 24 \
  --panel-size 560 \
  --seed 42
```

The three-view sheet independently centers and longest-axis-normalizes each raw
local mesh. Native pose/translation/scale deltas are reported separately, so a
layout change cannot be misreported as a shape improvement.

## Controlled P13 comparison

`run_p13_controlled_geometry_prior_ab.py` runs frozen TRELLIS, archived
raw-SAM2-mask SAM3D, and corrected object-owned-mask SAM3D through the exact same
canonical evidence and unmodified legacy P13 builder. Compatibility `trellis_*`
fields stay isolated in each candidate directory; the top-level report restores
source-neutral aliases and records the evidence/builder hashes.

The first attempted run is intentionally retained if environment setup fails.
For virtualenv interpreters, the orchestrator validates but does not resolve the
`bin/python` symlink, because resolving it can discard `pyvenv.cfg` discovery.

After a successful controlled report, render every aligned prior and pose
hypothesis in one observed-surface PCA frame:

```bash
$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/render_p13_controlled_geometry_prior_ab.py \
  --controlled-report \
    $EXP/p13_controlled_geometry_prior_ab_retry1/p13_controlled_geometry_prior_ab_report.json \
  --output-dir $EXP/p13_controlled_render \
  --target-faces 60000 \
  --panel-size 520
```

These are adapter-level P13 results, not SAM3D-native pose scores or physical
collision claims.

The pose-hypothesis render can expose gaps created by cutting generated faces
near observed support. Quantify those gaps without turning them into an arbitrary
gate:

```bash
$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/evaluate_p13_controlled_seams.py \
  --controlled-report \
    $EXP/p13_controlled_geometry_prior_ab_retry1/p13_controlled_geometry_prior_ab_report.json \
  --output-dir $EXP/p13_controlled_seams
```

The seam diagnostic reports both boundary directions and CDFs at 1/2/5/10/20/30
mm. It remains unsigned and cannot promote collision readiness.

## Dual-mesh P13

The legacy builder deletes every generated face center inside the observed band
(12.119 mm here), which explains the approximately 11 mm SAM seam and destroys a
watertight raw prior. Build a source-neutral experimental state that applies only
the global P13 similarity to the raw SAM mesh and keeps observation/physics in
separate layers:

```bash
$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/build_p13_dual_mesh_geometry_prior.py \
  --controlled-report \
    $EXP/p13_controlled_geometry_prior_ab_retry1/p13_controlled_geometry_prior_ab_report.json \
  --candidate sam3d_new_object_owned_mask \
  --output-dir $EXP/p13_dual_mesh_new_owned \
  --target-faces 60000 \
  --panel-size 540
```

The generated mesh is a complete render underlay only. The observed metric mesh
is rendered on top and remains the only collision-eligible surface; signed
geometry stays disabled.

## Experimental P14/P15 layered render adapter

The P13 PCA review cannot show whether an intact prior stays sane under the full
object trajectory, camera projection, and hand motion.  The next discriminating
test therefore freezes the existing observed-only 150-frame object pose and all
camera/MANO state, and changes only the geometry backend/integration:

1. corrected object-owned-mask SAM3D with the intact topology-preserving dual
   mesh;
2. the same corrected SAM3D prior after the legacy observed-band face deletion;
3. frozen TRELLIS after the same legacy integration.

`build_p14_p15_layered_render_states.py` clones the existing observed-only render
state into three additive experiment states.  It verifies that the pose graph was
solved against the same observed canonical mesh and had zero nonpenetration
pose targets.  Across the three states, the annotation/camera backbone, pose
rows, MANO constraint payload, temporal MANO payload, hidden-volume payload,
and projection contract must have identical value hashes.  The inherited
geometry-dependent constraint payload is retained for provenance but is not
rendered or recomputed.  Every branch uses one shared observed-only physical
surface; generated contact, collision, and signed geometry remain disabled.

```bash
SOURCE_STATE=$RUN/experiments/sam3d_vs_trellis_20260803/triangle_front_only_rotate_only/model_b_sam3d/fixed_observed_pose_physical/render_state.json
AB=$EXP/p14_p15_layered_geometry_ab_v1

$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/build_p14_p15_layered_render_states.py \
  --source-render-state $SOURCE_STATE \
  --dual-mesh-state $EXP/p13_dual_mesh_new_owned/dual_mesh_render_state.json \
  --dual-mesh-report $EXP/p13_dual_mesh_new_owned/p13_dual_mesh_geometry_prior_report.json \
  --controlled-report $EXP/p13_controlled_geometry_prior_ab_retry1/p13_controlled_geometry_prior_ab_report.json \
  --output-dir $AB
```

`render_p14_p15_layered_state.py` uses the canonical V19 source-to-render K
scaling and camera transform helpers.  Unlike the old P19 diagnostic renderer,
it resolves every annotation `vertices_reference` to the actual full 778-vertex
metric MANO surface and the 1,538-face left/right topology.  Inline 64-vertex
samples must reproduce exactly.  Generated, observed, and MANO triangles share
a deterministic mean-triangle-depth review painter.  Observed pixels own object
support so the hidden prior cannot punch holes through measured geometry, while
MANO triangles that win the shared depth painter remain visible.  This is still
a review renderer rather than a metric z-buffer evaluator.

First render frame 109 at a high display budget and export a combined world GLB:

```bash
BRANCH=sam3d_owned_dual_mesh  # repeat for sam3d_owned_legacy_cut and trellis_frozen_legacy_cut
$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/render_p14_p15_layered_state.py \
  --render-state $AB/$BRANCH/experimental_layered_render_state.json \
  --output-dir $AB/$BRANCH/frame109_review \
  --frames 109 \
  --generated-face-budget 30000 \
  --observed-face-budget 0 \
  --mano-face-budget 0 \
  --export-glb-frame 109 \
  --no-encode-video
```

Then render all 150 frames with one shared display budget:

```bash
$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/render_p14_p15_layered_state.py \
  --render-state $AB/$BRANCH/experimental_layered_render_state.json \
  --output-dir $AB/$BRANCH/full_video \
  --generated-face-budget 8000 \
  --observed-face-budget 0 \
  --mano-face-budget 0 \
  --export-glb-frame 109
```

Finally validate the shared state/budgets/intrinsics/world framing and compose
three-column camera-overlay plus overlay/world/side comparison videos:

```bash
$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/compose_p15_layered_geometry_ab.py \
  --adapter-report $AB/p14_p15_layered_render_state_adapter_report.json \
  --manifests \
    $AB/sam3d_owned_dual_mesh/full_video/p14_p15_layered_full_mano_render_manifest.json \
    $AB/sam3d_owned_legacy_cut/full_video/p14_p15_layered_full_mano_render_manifest.json \
    $AB/trellis_frozen_legacy_cut/full_video/p14_p15_layered_full_mano_render_manifest.json \
  --output-dir $AB/full_video_three_branch_ab \
  --review-source-frame 109
```

These videos test temporal render behavior only.  They do not validate SAM3D's
native camera pose/scale convention, establish a watertight metric collision
body, or authorize signed MANO contact/nonpenetration.  This archived clip also
uses the earlier 1408-to-960 K scaling contract; formal promotion still requires
repeating the benchmark on the official HOT3D pinhole/P03c camera contract.

As a separate unsigned diagnostic, recover all 778 MANO vertices for all 300
frame/side rows, invert the same observed-only object trajectory into canonical
coordinates, and query exact point-to-triangle distance against the shared
observed surface plus each generated render layer:

```bash
$REPO/.venv/bin/python \
  $REPO/experiments/sam3d_p11_p12_branch/evaluate_p15_full_mano_unsigned_distance.py \
  --adapter-report $AB/p14_p15_layered_render_state_adapter_report.json \
  --render-states \
    $AB/sam3d_owned_dual_mesh/experimental_layered_render_state.json \
    $AB/sam3d_owned_legacy_cut/experimental_layered_render_state.json \
    $AB/trellis_frozen_legacy_cut/experimental_layered_render_state.json \
  --output-dir $AB/full_mano_unsigned_distance
```

Only distance to the shared prediction-side observed mesh is physically eligible,
and even that partial non-watertight surface cannot supply a sign.  Generated
layer distances are layout diagnostics only; near-zero values must not be called
contact or penetration.
