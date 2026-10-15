# Milk R6: full 118D PPO trusting the native SAM3D mesh — results 2026-10-12

## Verdict

**The generated mesh provides a real, action-coupled nonpenetration hypothesis; the restored ordinary full-action PPO did not eliminate penetration or improve the annotation. Do not promote any checkpoint.**

This run fulfilled the operator's request to unfreeze both hands and use the available generated geometry for interaction training. It restored the original full 118D PPO configuration rather than silently keeping the six-action policy. It finished 1024 steps, four independent complete 150-frame checkpoint evaluations, geometry audits and full native/MANO renders; process exit 0. Execution/checkpoint success is not quality success.

## Inputs and mechanism

- Immutable runtime: `/mnt/user-home/kupingxin/ego_annotation_runtime/rl_generated_collision_milk_20261012_r1`
- Output run: `/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_generated_collision_r6_learning1024_20261012_r1`
- Logs/reviews: `/mnt/user-home/kupingxin/ego_annotation_experiments/generated_collision_rl_milk_20261012_r1`
- Existing Python: `/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python`
- Host: `dexrobot-Standard-PC-Q35-ICH9-2009`; CPU numerical execution, one thread; no package installs or perception rerun.
- Same frozen milk R6 initializer and original v2 source observations; no GT/CAD or old policy input.
- Native object: 239430 vertices / 478880 triangles. Source `visual_native_mesh` explicitly points to `sam3d_mesh.ply`. No mesh repair, proxy, decimation, shape/scale change or camera change.
- Actual topology: closed, consistently wound, positive-volume mesh. This enables queries **relative to that generated body**, not independent physical shape/sign certification.
- Both MANO roots/45D articulation/translations, object SE3 and ten contact logits are policy coordinates. No extra hand-freeze mask. Source shape/betas and ordinary absent/inactive gates remain unchanged.
- Additive independent term: `hypothesis/object:0/{hand}/penetration/frame=…`, computed from all MANO vertices plus every MANO triangle centroid against the current transformed complete object mesh. Weight300000, 0.1mm tolerance, squared inside residual with positive-count normalization. Contact probability cannot deactivate it. Original contact attraction/deadband and all formal factors remain unchanged.
- Original source `signed_geometry_ready=false` and observed-face authority masks remain unchanged. Operator trust is explicit in a separate contract, `generated_mesh_collision_hypothesis.v1`.
- Original ordinary PPO settings restored: Gaussian log_std=-1, rollouts128, epochs/minibatches8/8, lr3e-4, clip.2, gradient clip.5, KL threshold0. No old checkpoint is loaded; the old no-collision contract was explicitly rejected.

## Zero-action penetration frames

Criterion: an actual hand vertex or triangle centroid is more than 0.1mm inside the generated object, **or** an interior hand-edge segment intersects an actual object triangle. Ray endpoint touches are excluded at the tolerance. These are generated-mesh-consistency diagnostics; absence of these sampled/edge tests is not a proof that all triangle pairs are disjoint.

Frame indices are zero-based, complete clip0–149:

- Right hand: **0–149** (150 frames).
- Left hand: **0–143, 145** (145 frames).
- Union: **all150 frames**.
- Maximum sampled inside depth: **1.8791mm** (left); right maximum1.4106mm.
- More-than-1mm penetration samples present in143 left frames and134 right frames.

Baseline exact records: `zero/collision_frames.csv`, `zero/collision_audit.json`.

The detector uses actual mesh inside/outside queries and actual hand-edge/object-triangle casts. Five-ray training and eleven-ray independent audit were compared on all300 zero hand-frames: **no threshold-sign disagreements**.

## Complete deterministic checkpoint outcomes

`J_total = J_original + J_generated_collision`, lower is better. Raw factor weights and original J were not replaced by a new success metric.

| Candidate | J_original | Added collision cost | J_total | Frames with any detection | Max inside depth mm | Summed squared inside hinge* | Crossing edges summed over hand-frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero R6 | 2067.838616 | 33.748504 | 2101.587120 | 150 | 1.8791 | 0.004097724 | 26237 |
| 256 | 36519.248714 | 32.599094 | 36551.847808 | 150 | 2.0007 | 0.004162122 | 27024 |
| 512 | 50850.652752 | 33.631372 | 50884.284124 | 150 | 1.9221 | 0.003664697 | 25104 |
| 768 | 372611.113746 | 33.627806 | 372644.741552 | 150 | 1.8254 | 0.003402412 | 23017 |
| 1024 | 1902688.430886 | 33.565031 | 1902721.995916 | 150 | 1.9327 | 0.003769997 | 24816 |

`*` Sum over the same full vertex/centroid sample set and timeline, in m²; not penetration volume, not mean physical depth, and not GT.

At768, summed squared inside residuals fall **16.97%** and edge-crossing count falls **12.27%**. At1024, reductions are only **8.00% / 5.42%**, and maximum inside depth increases **2.85%**. No checkpoint has a completely penetration-free full frame by the union criterion. The best trained checkpoint by total J is256, but it is much worse than zero and even has worse aggregate collision metrics. The768 collision improvement does not compensate for the large violation of image constraints.

At1024, left affected frames are0–142 and144; right remains0–149. Exact checkpoint records are under `checkpoints/step_*/evaluation/collision_frames.csv` and `collision_audit.json`.

## Action coupling and why this is not enough

Controlled frame0 ± translations/root/articulation were run before PPO. Both hands' geometry changes under their coordinates. There are locally improving translations:

- Left +2mm world-Z: collision objective change approximately−0.01417; original objective change+0.00267, so combined objective improves.
- Right +2mm world-Y: collision change approximately−0.00604; original objective change+0.00267, again combined objective improves.
- Lowering contact logits does not change the body barrier.

Therefore the barrier is not disconnected, and a better local direction exists without freezing hands. It does not follow that high-dimensional PPO will find it reliably.

The final joint-pixel cost is **1900094.40**, about **99.86%** of total J; this is weighted cost, not pixels. Independently measured joint pixel-shift p95 is14.61px left /21.96px right, with maximum23.30 /30.86px. Mean MANO vertex changes are3.26 /5.77mm. All evaluated valid hand joints retain positive camera depth; the losses are not simply behind-camera explosions.

The added collision cost is only on the scale of33 compared with the accumulated pixel costs. The ordinary exploration starts with about3.68mm translation sigma and1.05deg rotation-coordinate sigma, while the measured generated-body inside depths are around1–2mm. This scale mismatch is a candidate explanation, not proof of sole causality.

Post-update PPO diagnostics show clip fractions0.984–1.0 and extremely large empirical approximate KL. Each128-step update actually performs64 optimizer steps (512 overall); parameters are finite and stochastic std barely changes (~.368→.3675). Those KL values are post-update estimates over sampled latents and current normalization, not internal minibatch KL instrumentation or a formal divergence certificate. They strongly motivate isolating normalizer/update accounting before claiming stable full-action learning; current data do not distinguish parameter movement, normalization movement and critic/advantage effects.

## A concrete reduction defect exposed

The first implementation reused the dense barrier's normalization by **the number of currently penetrating samples**. That is not monotonic in aggregate collision amount:

- At256 the added collision cost decreases3.41%, yet squared inside sum increases1.57% and crossing-edge count increases3.00%.
- Adding shallow positive samples can dilute an average over positive samples; removing them can hide real count/depth improvement.

Thus collision reward alone is not a reliable evaluation measure. A next intervention should fix the per-frame normalization (fixed sample/area weights or explicit total/count terms, with physical scaling) and verify monotonic inward/outward/count behavior. Do not just increase steps or silently drop projection costs. Keep hands unfrozen and geometry unchanged, but bound exploration/updates in physical units rather than repeating this high-noise configuration.

## Render and RRD deliverables

All are complete150-frame native mesh/MANO comparisons. Panels: Zero R6 / best trained256 / last1024.

- `renders/zero_best_last_overlay.mp4`
- `renders/zero_best_last_world.mp4`
- **`renders/zero_best_last_world_collision.rrd`** (29687818bytes, Rerun0.37.2)

The RRD is self-contained. Actual MANO vertices >0.1mm inside the generated body are red; actual crossing hand edges are red line segments. Toggle object visibility or rotate the view to inspect occluded red regions; no geometry is offset or replaced. The MP4 caption's “no signed claim” refers to absence of certified physical authority, not absence of the experimental generated-body signed queries.

Raw same-256px SAM2/full-mesh silhouette IoU median: zero.832334, trained256.848723, last1024.811851. These are not amodal/GT or physically accurate collision metrics; improved IoU at256 does not make that checkpoint an annotation improvement.

Both MP4s independently decode150frames/30fps. All150 RGB/fixed-world frames were inspected through six all-frame sheets plus representative full-size images. Native Rerun screenshots of the three final views were obtained and inspected. The RRD passed official `rerun rrd verify`; every complete object topology, all object transforms and all900 hand mesh frames were read back and compared to their frozen arrays.

## Regression and preservation

- Final grouped+domain/real regression: **198 tests, 194 passed, 4 skipped**.
- Frozen snapshot:211 source/config hashes rechecked, **0 mismatches**.
- R6 coarse input hash unchanged; input source/NPZ reverified during execution.
- Four checkpoint/export artifact hashes independently checked.
- Training process exit0, bounded1024 steps; no extension, merge, push or baseline replacement.

Detailed reports: `final_collision_review/assessment.json`, `final_geometry_review/geometry_diagnostics.json`, `final_video_review/verification.json`, `delivery_verification.json` under the experiment log directory.

## Decision

Keep the unmodified R6 zero output as the baseline. Generated SAM3D triangles can indeed drive interaction/collision queries and useful controlled residual directions, but **trusting that geometry alone does not make the current full PPO a successful hand-object nonpenetration optimizer**. The next causal correction is collision-reduction monotonicity and full-action exploration/update accounting, not renewed hand freezing, a proxy object, fake physical-authority flags, or a larger training budget.
