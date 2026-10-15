# Trusted generated-mesh full-action PPO experiment — 2026-10-12

## User-selected intervention and causal prediction

The operator requests restoring normal RL (both MANO hands unfrozen) and using only the available SAM3D-generated object geometry as a trusted interaction hypothesis. Previous R6 experiments had no active signed collision cost, a 64mm contact deadband, and the most recent object-only experiment could not move hands. A generated-mesh-dependent repulsive cost plus the normal full action space is needed to test whether the delivered MANO/native-mesh intersections can actually decrease.

Predictions: (1) if the missing repulsive reward was sufficient, full-episode inside depth/count and actual hand-edge/object-triangle crossings decline without large joint-image deterioration; (2) collision declines but image/MANO consistency fails => competing evidence/scales or unsafe update, not annotation success; (3) controlled outward motions improve collision cost but PPO does not => reward is coupled, optimization failed; (4) no baseline 3D intersections => the apparent visual overlap was occlusion, not the tested mesh penetration mechanism.

## Input and authority

Same immutable R6 milk coarse manifest/NPZ and original v2 source measurements, full 150 frames, original world/camera/K and native metric MANO. Original coarse/source `signed_geometry_ready=false` and qualification guards remain intact. No GT/CAD, old policy, perception rerun, box/primitive geometry, mesh repair, remeshing, scaling, hand-shape change or installation.

Actual native mesh inspected: 239430 vertices / 478880 faces; trimesh reports watertight, consistent winding and positive enclosed volume (~0.00011716m3). This supports using inside/outside for **this generated mesh**, not claiming real object shape/depth/physical certification. A separate explicit `generated_mesh_collision_hypothesis.v1` runtime contract logs an additive hypothetical collision term. It does not set measured signed-authority flags to true.

## Objective and actions

Use standard 118D CoarseAnnotationEnv decoder, official skrl 2.1.0 PPO, original physical scales, source priors and formal factors. Both hands' roots/45D articulation/translation, object SE3 and contact logits are sampled and executed subject only to existing absent/inactive gates. No new hand freeze or object-only radial mapping.

Primary new collision cost queries the current MANO vertices **and every MANO triangle centroid** against the full actual canonical native object mesh under current object SE3. It penalizes positive inside depth beyond 0.1mm, with squared-hinge average over positive samples at weight300000 per hand/frame. This is independent of contact probability, so lowering a contact logit cannot bypass collision. The original contact prior/deadband stays unchanged; this experiment isolates nonpenetration rather than also changing attraction/uncertainty. Report raw original `J_original`, additive `J_generated_collision`, and their training sum `J_total` separately. Original reward convention is -1e-5*dJ_total, gamma1 (no centered reward).

Original baseline PPO hyperparameters restored: log_std=-1, rollouts128, epochs8, minibatches8, lr3e-4, ratio/value clips.2, grad norm.5, KL threshold0. Existing Gaussian raw-action/likelihood contract preserved. This may still exhibit the previous high-noise projection failure; if so retain it, do not hide it or extend budget. Zero policy mean initial state unchanged.

## Detection, execution, budget and outputs

1. Before PPO, detect the full zero trajectory with 0.1mm and 1mm inside-depth thresholds for both hands; record all frame IDs/intervals, vertex/face-center sample counts, depths and closest triangles. Also query actual MANO edge segments against object triangles, detecting surface crossings even when sample points miss an intersection. These are generated-mesh-consistency metrics, not GT physical penetration or volume estimates.
2. Controlled ± hand translations and articulation/contact execution verify both hands respond; collision has improving directions, old/checkpoint contracts fail and zero return telescopes.
3. Freeze fresh snapshot and output root; run CPU-only one numerical thread on A800 server, 1024steps, full independent deterministic 150-frame checkpoint evaluation at256/512/768/1024. 35-minute wall limit per attempt, no automatic extension. Exact artifact identity and finite policy checks.
4. Independent posthoc query collision at more parity rays; compare complete original/full objectives, all-frame collision metrics, image residuals and native/MANO RGB/fixed-world videos. Produce a same-candidate RRD if requested/feasible. Selection criterion is not 'lowest trained J' alone; preserve zero in every comparison and label collisions/image tradeoffs or failure explicitly.

This is an offline single-video experimental configuration, not a production capability or generalization claim. The additive reward/detection is a visualization-and-training hypothesis under the operator's trust choice; source metric/observational authority remains separate.
