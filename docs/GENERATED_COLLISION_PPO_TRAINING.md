# Full-action PPO under a trusted generated-object collision hypothesis

This is an explicitly selected experimental environment, **not the default producer**:

- `configs/ppo_generated_collision_coarse.yaml`
- `envs.generated_collision_coarse_env:GeneratedCollisionCoarseEnv`
- `utils/generated_collision.py`
- `tools/rl_experiments/generated_collision_learning.py`

The input is the same R6 initializer/native SAM3D topology plus hash-bound source v2 measurements. The normal 118D action decoder/official skrl 2.1.0 PPO is restored. Both MANO hands' root, articulation and translation are trainable; contact logits and object SE3 are also trainable. Original absent-entity/inactive-contact masks remain, but no extra hand freeze, object-only radial mapping, or hard action mask is installed. Source MANO betas, object shape/scale and camera are fixed as in the ordinary full-action pipeline.

## Collision hypothesis rather than authority promotion

The operator explicitly trusts the generated object body for this experiment. The actual retained mesh must be closed, winding-consistent and have positive enclosed volume for the selected signed parity query; it is never repaired, remeshed, voxelized or replaced by a proxy. Original measured signed flags/face-authority masks remain false/zero. The new additive term is named `hypothesis/object:0/{left,right}/penetration/frame=…` and is logged as operator-trusted generated geometry, not observed physical sign.

The current actual hand and object state feed the query at every accepted frame. Samples are all MANO vertices and every MANO triangle centroid, transformed through the current world_from_object SE3 into the actual native canonical mesh. Cached scene geometry is fixed; the per-frame candidate cache is invalidated by the standard action decoder. Current cost at each hand/frame is

```
weight * sum(max(inside_depth - 0.1mm, 0)^2) / max(number of positive samples, 1)
```

with weight300000 and five parity rays. This v1 reduction mirrors the original dense barrier's positive-residual-count normalization. It is not a count/volume metric: changes in the positive denominator can make a lower mean penalty coexist with more aggregate penetration. Always examine the independently reported summed squared depths, maximum depth, inside sample/vertex counts and actual crossing edges too. The barrier is not multiplied by contact probability; a contact logit cannot switch it off. Existing 64mm source contact deadband remains unchanged in this first controlled intervention.

`J_total = J_original_formal + J_generated_collision`; all original factor weights/reductions remain. Full returns telescope to `-1e-5*J_total` with gamma1. The old no-collision checkpoints are rejected by environment/config compatibility. No learned checkpoint is substituted for zero solely because its collision cost decreases.

## Invocation and output

Use the provisioned server interpreter, CPU numerical execution, one thread and a fresh immutable snapshot/output root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -B -u \
  tools/rl_experiments/generated_collision_learning.py \
  --coarse /absolute/R6/coarse_scene.json --output /fresh/run
```

The configuration deliberately restores the prior ordinary PPO settings (initial log_std -1, 128-step rollouts, 8 epochs/8 minibatches, lr3e-4, clip.2, no KL early stop), not the object-only experiment's hyperparameters. The declared test is 1024steps with full independent 150-frame evaluations at256/512/768/1024. This controls the normal-flow comparison but may repeat the known high-noise hand projection failure; retain and report it rather than calling the trust hypothesis successful or silently changing settings in flight.

Each zero/checkpoint evaluation folder contains `collision_audit.json` and `collision_frames.csv`, including every frame's left/right inside depths, actual vertex and centroid IDs, and actual hand-edge/native-object triangle crossings. The independent audit uses eleven parity rays; a separate actual-zero comparison found no 5-vs-11 sign disagreements across all 300 hand-frames. Detected intersections are with the generated mesh, not physical GT; absence of these samples/edge crossings is not a formal proof of triangle disjointness.

Posthoc commands:

```bash
python -B tools/rl_experiments/inspect_preprocessed_results.py --run /run --output /fresh/geometry_review
python -B tools/rl_experiments/inspect_generated_collision_results.py --run /run --output /fresh/collision_review
python -B tools/rl_experiments/review_comparison_videos.py --run /run --output /fresh/video_review
```

The collision review exports `renders/zero_best_last_world_collision.rrd`: complete object/MANO geometry, identical world gauge/initial eye, shared full timeline, red actual MANO vertices >0.1mm inside the hypothesis body and red actual crossing hand edges. Object visibility can be toggled in Rerun to see the occluded red regions. The RRD contains actual surfaces, not video billboards. Default zero/best-trained/last MP4s remain unmodified.

This is offline optimization of a single video, not production/generalization or physical-accuracy certification. Independent metric/report/render inspection is required before concluding what changed.
