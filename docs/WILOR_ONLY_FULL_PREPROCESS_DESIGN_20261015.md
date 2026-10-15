# WiLoR-only complete preprocessing experiment (2026-10-15)

## Owner requirements and causal scope

Re-run the complete raw-video observation + R6 preprocessing + ordered
shape/depth/MANO repair path. **No HaWoR predictions, camera, metric scale, hand
initialization, infilling, masks, ownership, contact, old object geometry or
preprocessing output may enter the new prediction.** Existing HaWoR results are
readable only by a separate evaluator after the new outputs freeze. Original raw
RGB and prediction-side sensor calibration remain valid inputs; licensed MANO,
DROID, UniDepth, WiLoR, SAM2/OWLv2/SAM3D model weights are model assets, not case
predictions. No GT/CAD/evaluator sidecars are prediction inputs.

New branch experiment/rl-wilor-only-preprocessing-20261015, base4d8bb65. Prior
component-study branch and artifacts stay unchanged. Source defaults for HaWoR
remain available for other users, but the WiLoR runtime excludes its module and
has no hawor_root/hawor_python assets. Historical `hawor_npz`/bridge schema names
must not be interpreted as producer identity; new backend contracts identify the
actual model and prove camera/FK binding. Prefer neutral names in new records.

## Independent mechanisms

1. Full original RGB timeline + exact supplied K; fresh UniDepth inference.
2. WiLoR detector and full MANO on every frame. No old camera track: network
   measurement lives in camera coordinates. Preserve native 2D; camera translation
   may use a declared own-2D perspective fit, never GT/hand-size scaling.
3. Independent DROID-SLAM from its standalone provisioned source/checkpoint,
   original images and K, never the HaWoR SLAM wrapper or hand masks. Estimate a
   single metric translation scale from DROID inverse-depth vs current UniDepth
   on static-background candidates excluding current object masks and WiLoR boxes.
   This scale remains uncertain; no fallback to old trajectories or GT scale.
4. Convert right-network/reflected-left MANO to canonical side-specific layer
   rotation matrices and world transforms. Correct the shaped wrist pivot when
   composing camera rotation. Replay all actual vertices/joints with the existing
   canonical FK before supplying them to the unchanged solver variables. Missing
   detections are missing, not filled with old hands.
5. Fresh object plan/detection/SAM2, then regenerate base/bridge, hand depth-order
   ownership, eligibility, P09 geometry/anchor, RGB source motion, native SAM3D
   mesh, geometry binding and independent interaction judgments/factors. Fresh
   v2 source manifest; no edited legacy output pretending to be a fresh producer.
6. Full R6 mask-repair solve and ordered shared-shape / relative-depth / full-MANO
   stages on the new source, with complete videos, retaining every failure.

Per-stage resource-aware GPU selection is explicitly owner-authorized, allowing
sharing only when measured free memory and utilization satisfy the declared stage
budget. Do not kill other jobs. Keep the old exclusive policy as default. A model
OOM or invalid inference is an actual failed attempt, not silent fallback.

## Discriminating predictions and outputs

The strict uncertainty is whether a genuinely independent WiLoR foundation changes
hand depth, hand-derived object masks/geometry, and final joint state after the
same complete consistency pipeline. If 2D improves but metric depth fails after
joint fitting, report that ambiguity. If object state changes through ownership,
show actual masks and full native mesh videos; do not attribute all changes to
articulation. If independent camera motion dominates, score and expose it rather
than importing a better HaWoR camera. A full no-HaWoR substitution changes both
camera and hand systems, so it is NOT a one-variable hand-only causal comparison.

All stages cover the complete150-frame milk clip. Render native object +778-vertex
hands in RGB and one shared fixed-world observer against the frozen HaWoR result;
one first-camera SE3/scale1 for display/evaluation, no per-frame registration. Read
HOT3D only after final prediction freeze. Report hand2D/3D, posed mesh errors,
camera errors, object geometry/pose, coverage, temporal errors and actual runtime.
No contact/zero-penetration or held-out generalization certification. No RL
training or default replacement in this experiment.
