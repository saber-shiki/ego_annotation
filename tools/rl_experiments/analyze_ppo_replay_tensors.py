"""Analyze captured PPO tensors without environment stepping or optimizer updates.

The first Gaussian draws are reconstructed bitwise. Only the FIRST minibatch
before optimization can be attributed to the original update. Learned-actor
normalizer probes are endpoint sensitivity tests, not historical PPO KL.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gymnasium as gym
import numpy as np
import torch
from skrl.agents.torch.ppo.ppo import compute_gae
from skrl.resources.preprocessors.torch import RunningStandardScaler

from models.factory import build_models
from tools.rl_experiments.diagnose_ppo_update_invariants import compare
from tools.rl_experiments.reward_ablation_learning import parameter_hash


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))


def advantages(rewards, values, last_values, tensors, cfg):
    return compute_gae(
        rewards=rewards, values=values, last_values=last_values,
        terminated=tensors['terminated'], truncated=tensors['truncated'],
        discount_factor=cfg['discount_factor'], lambda_coefficient=cfg['gae_lambda'],
        time_limit_bootstrap=cfg['time_limit_bootstrap'],
    )


def flat_gradient(parameters):
    return torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten() for p in parameters])


def gaussian_log_probability(policy, observations, actions):
    mean, extra = policy.compute({'observations': observations}, role='policy')
    log_std = extra['log_std']
    if policy._g_clip_log_std:
        log_std = log_std.clamp(policy._g_min_log_std, policy._g_max_log_std)
    return torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1, keepdim=True)


def first_minibatch_probe(models, tensors, rewards, last_values, cfg, indexes, convention):
    """Compute gradients only. No optimizer is instantiated or used.

    legacy: exact installed normalization/clipping ordering before step #1.
    frozen_observation_only / frozen_value_only: isolate each statistical
    transition, without changing learned parameters or updating a policy.
    fixed_identity: diagnostic counterfactual freezes BOTH scalers at their
    collection values. It is not a trained candidate or a performance claim.
    """
    policy, value = models['policy'], models['value']
    for model in models.values():
        model.zero_grad(set_to_none=True)
    returns, adv = advantages(rewards, tensors['values'], last_values, tensors, cfg)
    obs_scaler = RunningStandardScaler(size=tensors['observations'].shape[-1], device='cpu')
    value_scaler = RunningStandardScaler(size=1, device='cpu')
    if convention not in ('legacy', 'fixed_identity', 'frozen_observation_only', 'frozen_value_only'):
        raise ValueError(convention)
    update_value = convention in ('legacy', 'frozen_observation_only')
    update_observation = convention in ('legacy', 'frozen_value_only')
    old_values = value_scaler(tensors['values'], train=update_value)
    targets = value_scaler(returns, train=update_value)
    observation = obs_scaler(tensors['observations'].reshape(128, -1)[indexes], train=update_observation)
    actions = tensors['actions'].reshape(128, -1)[indexes]
    log_prob = gaussian_log_probability(policy, observation, actions)
    recorded = tensors['log_prob'].reshape(128, 1)[indexes]
    ratio = (log_prob - recorded).exp()
    sampled_adv = adv.reshape(128, 1)[indexes]
    policy_loss = -torch.minimum(sampled_adv * ratio, sampled_adv * ratio.clamp(1 - cfg['ratio_clip'], 1 + cfg['ratio_clip'])).mean()
    predicted, _ = value.compute({'observations': observation}, role='value')
    anchor = old_values.reshape(128, 1)[indexes]
    target = targets.reshape(128, 1)[indexes]
    difference = predicted - anchor
    clipped = anchor + difference.clamp(-cfg['value_clip'], cfg['value_clip']) if cfg['value_clip'] > 0 else predicted
    value_loss = cfg['value_loss_scale'] * torch.nn.functional.mse_loss(target, clipped)
    (policy_loss + value_loss).backward()
    actor_grad = flat_gradient(list(policy.parameters()))
    critic_grad = flat_gradient(list(value.parameters()))
    actor_norm, critic_norm = float(actor_grad.norm()), float(critic_grad.norm())
    joint_norm = float(np.hypot(actor_norm, critic_norm))
    row = {
        'convention': convention, 'policy_loss': float(policy_loss.detach()), 'value_loss': float(value_loss.detach()),
        'actor_gradient_norm': actor_norm, 'critic_gradient_norm': critic_norm,
        'joint_gradient_norm': joint_norm,
        'joint_clip_multiplier': min(1., cfg['grad_norm_clip'] / (joint_norm + 1e-6)) if cfg['grad_norm_clip'] > 0 else 1.,
        'value_clip_saturated_fraction': float((difference.abs() > cfg['value_clip']).float().mean()) if cfg['value_clip'] > 0 else 0.,
        'oldvalue_predicted_abs_difference_mean': float(difference.detach().abs().mean()),
        'raw_logprob_difference_max': float((log_prob - recorded).detach().abs().max()),
        'optimizer_steps': 0,
    }
    return row, actor_grad, critic_grad


def analyze(run, captured, output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((run / 'matrix_contract.json').read_text())['base_config']
    cfg = config['ppo']
    report = {'scope': 'captured replay loss/gradient diagnostic; no optimizer, training, new rollout or annotation',
              'host_device': 'dexrobot-Standard-PC-Q35-ICH9-2009 / CPU single-thread', 'optimizer_steps': 0, 'seeds': []}
    inputs = {}
    for seed in (0, 1):
        tensor_path = captured / f'seed_{seed}' / 'first128_replay_tensors.npz'
        inputs[str(tensor_path)] = sha256(tensor_path)
        with np.load(tensor_path, allow_pickle=False) as archive:
            tensors = {k: torch.from_numpy(archive[k].copy()) for k in archive.files}
        torch.manual_seed(seed)
        env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (551,), np.float32),
                              action_space=gym.spaces.Box(-np.inf, np.inf, (118,), np.float32),
                              state_space=None, device='cpu')
        models = build_models(env, config)
        initial_file = run / 'runs/full' / f'seed_{seed}' / 'initial_parameters.json'
        expected = json.loads(initial_file.read_text())
        assert parameter_hash(models['policy']) == expected['actor']
        assert parameter_hash(models['value']) == expected['critic']
        # Identical zero-mean Gaussian sampling consumes identical RNG. This
        # confirms draw reconstruction, but firstbatch index attribution also
        # relies on the inspected trainer not drawing elsewhere before update.
        mean = torch.zeros(1, 118)
        std = models['policy'].log_std_parameter.detach().exp()
        distribution = torch.distributions.Normal(mean, std)
        reconstructed_actions = torch.stack([distribution.rsample() for _ in range(128)])
        assert torch.equal(reconstructed_actions, tensors['actions'])
        indexes = torch.randperm(128)[:16]
        with torch.no_grad():
            raw_next, _ = models['value'].compute({'observations': tensors['last_observations'].clamp(-5, 5)}, role='value')
        initial_scaler = RunningStandardScaler(size=551, device='cpu')
        initial_values_scaler = RunningStandardScaler(size=1, device='cpu')
        last_values = initial_values_scaler(raw_next, inverse=True)
        _, full_adv = advantages(tensors['rewards'], tensors['values'], last_values, tensors, cfg)
        _, critic_only_adv = advantages(torch.zeros_like(tensors['rewards']), tensors['values'], last_values, tensors, cfg)
        groups = []
        gradients = {}
        for folder in sorted((run / 'runs').iterdir()):
            group = folder.name
            path = folder / f'seed_{seed}' / 'training_trace.json'
            inputs[str(path)] = sha256(path)
            trace = json.loads(path.read_text())[:128]
            rewards = torch.tensor([r['reward'] for r in trace], dtype=torch.float32).reshape(128, 1, 1)
            if group == 'full':
                assert torch.equal(rewards, tensors['rewards'])
            _, adv = advantages(rewards, tensors['values'], last_values, tensors, cfg)
            _, reward_only_adv = advantages(rewards, torch.zeros_like(tensors['values']), torch.zeros_like(last_values), tensors, cfg)
            legacy, actor_grad, critic_grad = first_minibatch_probe(models, tensors, rewards, last_values, cfg, indexes, 'legacy')
            consistent, fixed_actor_grad, _ = first_minibatch_probe(models, tensors, rewards, last_values, cfg, indexes, 'fixed_identity')
            groups.append({
                'group': group, 'advantage_to_full_cosine': cosine(adv, full_adv),
                'advantage_to_critic_only_cosine': cosine(adv, critic_only_adv),
                'advantage_to_reward_only_cosine': cosine(adv, reward_only_adv),
                'reward_std': float(rewards.std()), 'initial_value_std': float(tensors['values'].std()),
                'legacy_first_minibatch': legacy, 'frozen_scalers_counterfactual_first_minibatch': consistent,
                'legacy_actor_gradient_vs_fixed_scaler_cosine': cosine(actor_grad, fixed_actor_grad),
            })
            gradients[group] = actor_grad
        for row in groups:
            row['legacy_actor_gradient_vs_full_cosine'] = cosine(gradients[row['group']], gradients['full'])
            row['legacy_actor_gradient_vs_full_difference_norm'] = float((gradients[row['group']] - gradients['full']).norm())
        isolated_scalers = []
        for convention in ('legacy', 'frozen_observation_only', 'frozen_value_only', 'fixed_identity'):
            row, _, _ = first_minibatch_probe(models, tensors, tensors['rewards'], last_values, cfg, indexes, convention)
            isolated_scalers.append(row)
        # Reward-free actor gradient with identical randomized initial critic.
        critic_row, critic_actor_grad, _ = first_minibatch_probe(models, tensors, torch.zeros_like(tensors['rewards']), last_values, cfg, indexes, 'legacy')
        critic_row['hand_pixel_removed_actor_gradient_cosine_to_critic_only'] = cosine(gradients['hand_pixel'], critic_actor_grad)
        assert parameter_hash(models['policy']) == expected['actor']
        assert parameter_hash(models['value']) == expected['critic']
        checkpoint_path = run / 'runs/full' / f'seed_{seed}' / 'agent.pt'
        inputs[str(checkpoint_path)] = sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        learned = deepcopy(models['policy']); learned.load_state_dict(checkpoint['policy'])
        final_scaler = deepcopy(initial_scaler); final_scaler.load_state_dict(checkpoint['observation_preprocessor'])
        x = tensors['observations'].reshape(128, -1); a = tensors['actions'].reshape(128, -1)
        # Deliberately paired frozen learned actor; not sampled from it and not
        # the historical second rollout. Only exact analytic conditional KL
        # has the KL interpretation here.
        changing = deepcopy(final_scaler)
        changing(x[indexes], train=True)
        near_endpoint_sensitivity = compare(learned, final_scaler, changing, x[indexes], a[indexes])
        frozen_control = compare(learned, final_scaler, deepcopy(final_scaler), x[indexes], a[indexes])
        report['seeds'].append({
            'seed': seed, 'raw_gaussian_first128_reconstructed_exactly': True,
            'initial_actor_critic_verified': True, 'initial_actor_critic_unchanged': True,
            'first_minibatch_index_order': indexes.tolist(),
            'first_minibatch_index_scope': 'reconstructed RNG sequence; verified zero-mean draws and inspected library call order',
            'group_learning_signal': groups, 'reward_zero_random_critic_first_minibatch': critic_row,
            'full_first_minibatch_scaler_transition_2x2': isolated_scalers,
            'learned_actor_endpoint_single_batch_normalizer_sensitivity': near_endpoint_sensitivity,
            'learned_actor_frozen_scaler_negative_control': frozen_control,
            'learned_actor_sensitivity_is_not_historical_PPO_KL': True,
        })
    assert all(sha256(p) == h for p, h in inputs.items())
    report['input_sha256'] = inputs
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for s in report['seeds']:
        print('SEED', s['seed'])
        for g in s['group_learning_signal']:
            if g['group'] in ('full', 'hand_pixel', 'generated_collision', 'hand_surface_depth', 'object_temporal'):
                print(g)
        print('critic_only', s['reward_zero_random_critic_first_minibatch'])
        print('scaler_2x2', s['full_first_minibatch_scaler_transition_2x2'])
        print('normalizer_probe', s['learned_actor_endpoint_single_batch_normalizer_sensitivity'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--captured', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    analyze(args.run, args.captured, args.output)
