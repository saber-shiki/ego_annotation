# HaWoR / WiLoR hand measurement study (2026-10-15)

This is an opt-in component study on the current RL inputs, not a new pipeline
version or promotion of a different default. Base: 4d8bb65. Existing predictions,
training, source bundles and sibling worktrees remain read-only.

## Causal question and discriminating predictions

The current hand state can have incorrect articulation, image alignment, absolute
depth, and temporal jitter. HaWoR combines hand regression, infilling and camera
motion; WiLoR is single-frame hand regression. A lower world error alone cannot
identify which of these mechanisms improved. The old WiLoR worker also labels
root-local vertices/joints as camera coordinates and assumes its own focal.

Use exactly the frozen milk observation RGB and prediction-side full pinhole K.
Keep the existing prediction-side camera trajectory for both hand alternatives.
Freeze WiLoR outputs before loading any GT. No GT boxes, shapes, trajectories,
scale, optimization targets or hand-side filtering go to prediction. Select the
highest detector-score hand per side, retain all raw detections/counts, and do not
fill missing frames. Export full 778-vertex MANO surfaces and reproducible right
MANO rotation matrices + explicit left reflection + translation, not skeletons.
The left mesh has reversed winding. No GT-derived hand-size scaling.

Evaluation separately loads timestamp-matched HOT3D MANO and availability/QA
masks. Validate FK against the provided GT vertex cache. Compare official19
joints, matched common hand/frame slots as well as own coverage:

* source-pixel projection error (px); derived MANO GT projection, NOT independent
  manually annotated 2D landmarks; report in-image/positive-depth subset separately;
* camera absolute MPJPE and posed MPVPE (mm), wrist bias and depth;
* camera wrist-relative MPJPE (translation removed only); per-hand/frame Sim3 PA
  as diagnostic, never absolute metric accuracy;
* world error after one first-camera SE3 (scale=1), using the SAME camera for both;
* acceleration error on consecutive valid triplets, miss/duplicate rates, and
  actual worker runtime. No bridging missing frames.

If 2D/root-relative errors improve but depth worsens, WiLoR is a useful image/pose
measurement rather than a drop-in metric-world replacement. If own coverage looks
better than paired coverage, investigate missing-hand selection bias. If accuracy
improves but jitter worsens, retain temporal inference/uncertainty rather than
promoting raw single-frame results. If neither improves, the result rejects this
specific substitution, not independent hand measurements in general. Comparing
native default-focal and source-K translations from the SAME network predictions
separates a camera-interface defect from model quality.

## Camera-coupling amendment (before the second prediction-only experiment)

The first frozen run has native network 2D errors 20.73/20.19px, but the same
predicted anatomy after direct square-focal conversion projects at 47.42/45.19px
and has camera MPJPE39.56/40.33mm. Thus the direct weak-perspective camera
conversion is demonstrably not preserving the useful image measurement. Its
native full-image focal is 27500px versus the physical 974.34px. This is a
specific camera-model mismatch, not evidence that WiLoR's original 2D joints
are worse than HaWoR.

A second prediction-only experiment consumes only the frozen network measurement
and source K, with unchanged full MANO shape, articulation and orientation. Fit
three translation coordinates per hand/frame by least-squares perspective
projection to the SAME network's native 21 image points. No GT, rigid rotation,
scale, or hand-size target. Keep the direct-conversion branch and trace image
residuals. If image consistency and GT metric depth improve, camera coupling was
a substantial cause; if only image consistency improves, pose/shape/depth
ambiguity remains; if neither improves, fixed anatomy/orientation cannot explain
the native image evidence under true K. No temporal filter is introduced, so
jitter changes remain exposed. This experiment does not claim independent 2D/3D
evidence or production RL-FK compatibility.

## Initial scope and execution

Primary case: P0014_84ea2dcc, source frames 2370–2519, 150 frames / 30 Hz / 5 s.
This is a development case, not held-out generalization. Expand only after the
coordinate interface and primary experiment are verified. Prediction runs use a
curated immutable worker snapshot and fresh output directory, no evaluator/GT
files. Compute host dexrobot-Standard-PC-Q35-ICH9-2009 is the A800 server. Explicit
GPU selection, one CPU numerical thread. Provision missing public WiLoR weights
before prediction. Existing licensed MANO files are reused, never downloaded or
committed. Keep full-timeline mesh overlays and fixed-camera geometry comparison;
selected images are QC only. No contact/zero-penetration certification.

A production swap additionally requires canonical left/right MANO replay through
RL's existing FK, source/bridge/manifest provenance changes, ownership/masks and
contact re-computation, and end-to-end RL consumption tests. Merely renaming a
WiLoR NPZ as `hawor_world_hands.npz` is not integration.
