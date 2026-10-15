# Object-first safe-residual PPO experiment — 2026-10-05

## Named mechanism and causal question

The prior R6→118D PPO run executed real geometry but spent almost all loss on hand image displacement. At 256 steps object NN/outline had small gains; by 1024 raw J was 370758 vs zero 2067.84. This experiment tests whether a six-dimensional, low-noise, locally bounded object policy can improve the frozen R6 state without moving either source hand. It does NOT prove which of several training changes caused any improvement; it is one declared object-first configuration, not a single-factor ablation.

## Inputs and frozen invariants

Same hash-bound 2026-09-29 R6 milk coarse manifest/NPZ and original v2 measurement bundle; all 150 frames, full native triangles, fixed uniform scale, camera/K, metric MANO, raw P09/depth, qualified independent RGB factors and source visibility/contact declarations remain. No GT/CAD, new providers, old policy weights, numerical fit, generated signed authority or perception rerun. Existing v1/v2 guards and old 118D environment remain available.

## Action / probability contract

New explicit object-only environment exposes exactly six unbounded Gaussian **latent** actions: rotation xyz then world translation xyz. Existing .05 rad/.01 m local unit conversion is followed by a smooth radial-tanh ball map with maximum rotation-vector norm .5 degrees and maximum translation norm .002 m **relative to each frame's fixed R6 initializer**, not an accumulated motion-step bound. Only the object blocks enter the full decoder; hands/contact remain zero.

PPO stores the six sampled latent coordinates and their six-dimensional Gaussian likelihood. There are no sampled hand/contact dimensions to mask. The environment's deterministic map is part of the transition; likelihood is never computed by pretending executed/squashed values are unbounded Gaussian samples. Clarification of previous advice: deterministic post-sample filtering is not inherently an invalid PPO transition when raw action/likelihood are paired; the old design's concern was wasted exploration/credit assignment, not a proven probability bug.

Initial latent log std is [-2.8]*3 + [-3.0]*3; min -5, max -2.5, zero mean. No silent std annealing, actor warm start, critic changes or extra factor weights in this pass.

## Reward / PPO contract

Keep the same formal_annotation mathematical objective J, factor ratios and reward_scale=1e-5. Cache one complete zero-action prefix objective table J0 for each episode with the identical decoder/replay. Reward is `-c*((J_t-J0_t)-(J_prev-J0_prev))`, J(empty)=J0(empty)=0. Gamma remains 1. Therefore complete return is `-c*(J_final-J0_final)` and candidate ordering on the same episode is unchanged. This is action-independent centering, not a new physical factor; it does not guarantee variance reduction or fix credit assignment. Only current/accepted-prefix state enters observations; no future policy results are consumed.

Official skrl 2.1 PPO/Adam; 150 rollout steps (one complete episode), 4 epochs, 4 minibatches, lr1e-4, ratio/value clips .1, grad norm .25, KL threshold .01, GAE lambda .95. 600 continuous timesteps, seed0, deterministic evaluations after updates at 150/300/450/600. Training still uses current 551D actor/value observation encoding; critic prefix features and later hand/contact unfreezing are outside this experiment.

## Discriminating tests and results

1. Zero policy must reproduce R6 object trajectory and full native mesh; hands/contact must match the zero replay at all frames. Both raw and centered returns must telescope. Failures are implementation errors, not noisy measurements.
2. Controlled positive/negative object actions on a three-frame prefix must drive object terms, preserve hand terms and respect both radial limits. Old 118D checkpoints reject; stored actions/log probabilities are six-dimensional raw latents.
3. Success requires decreased **raw complete J** versus zero, without moving the hands or violating actual rotation/translation limits; full RGB/world video and same-raster silhouette diagnostics are separate evidence. "Best trained" must not be substituted for zero if no improvement exists.
4. If actions move geometry but all checkpoints are worse, excessive hand exploration is removed yet object objective/update direction remains insufficient. Diagnose factor changes/advantage/value/update statistics next; do not extend budget or delete costly factors.
5. If there is no appreciable action response, inspect likelihood, clipping/saturation, and objective coupling rather than declaring convergence.

## Compute and delivery

Current session is on A800 server `dexrobot-Standard-PC-Q35-ICH9-2009`; reuse `/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python`, CPU-only numerical threads=1. Parent edits/preflights and launches the frozen training command; no manually assembled runtime predictions. New immutable snapshot and fresh output root, durable launch log/exit sentinel in a window of canonical `ego_annotation` tmux. Produce four full 150-frame deterministic exports plus full 150-frame zero/best/last overlay and fixed-world videos. Preserve old run/dirty worktree. No package install, merge or push. Expected budget: implementation/tests roughly 20–35 min, numeric training/evaluations roughly 10–20 min, renders/review roughly 5–10 min; per-run hard limit 35 min. This is offline single-video experimental training, not a real-time production or generalization claim.
