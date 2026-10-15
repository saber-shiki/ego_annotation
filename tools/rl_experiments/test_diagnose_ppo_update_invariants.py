"""Regression tests for no-optimizer PPO replay diagnostic calculations."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, main

import gymnasium as gym
import numpy as np
import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler

from models.factory import build_models
from tools.rl_experiments.analyze_ppo_replay_tensors import (
    advantages, cosine, first_minibatch_probe, gaussian_log_probability,
)
from tools.rl_experiments.diagnose_ppo_update_invariants import compare, probability
from tools.rl_experiments.reward_ablation_learning import parameter_hash


class PPOReplayDiagnosticTest(TestCase):
    def models(self):
        torch.manual_seed(7)
        env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
                              action_space=gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                              state_space=None, device='cpu')
        cfg = {'model': {'encoder': {'hidden_dims': [8], 'layer_norm': True},
                         'policy': {'zero_mean_init': True, 'min_log_std': -3., 'max_log_std': -.5}}}
        return build_models(env, cfg)

    def tensors(self, models):
        x = torch.randn(128, 1, 6)
        a = torch.randn(128, 1, 3) * np.exp(-1.)
        with torch.no_grad():
            values, _ = models['value'].compute({'observations': x.reshape(128, 6)}, role='value')
            log_prob = gaussian_log_probability(models['policy'], x.reshape(128, 6), a.reshape(128, 3))
        return {'observations': x, 'actions': a, 'values': values.reshape(128, 1, 1),
                'log_prob': log_prob.reshape(128, 1, 1), 'rewards': torch.randn(128, 1, 1),
                'terminated': torch.zeros(128, 1, 1, dtype=torch.bool),
                'truncated': torch.zeros(128, 1, 1, dtype=torch.bool)}

    def test_probability_respects_policy_bounds_and_taken_raw_action(self):
        policy = self.models()['policy']
        with torch.no_grad():
            policy.log_std_parameter.fill_(-10.)
        x, a = torch.randn(10, 6), torch.randn(10, 3)
        scaler = RunningStandardScaler(size=6, device='cpu')
        _, std, log = probability(policy, scaler, x, a)
        self.assertTrue(torch.equal(std, torch.ones_like(std) * np.exp(-3.)))
        _, actual = policy.act({'observations': scaler(x), 'taken_actions': a}, role='policy')
        torch.testing.assert_close(log, actual['log_prob'], rtol=0, atol=0)

    def test_frozen_normalizer_compare_is_exact_negative_control(self):
        policy = self.models()['policy']
        with torch.no_grad():
            policy.mean_head.weight.fill_(.1)
        scaler = RunningStandardScaler(size=6, device='cpu')
        x, a = torch.randn(128, 6), torch.randn(128, 3)
        before = parameter_hash(policy)
        result = compare(policy, scaler, deepcopy(scaler), x, a)
        for name in ('analytic_conditional_KL_mean', 'logprob_change_abs_max', 'mean_action_difference_rms'):
            self.assertEqual(result[name], 0.)
        self.assertEqual(before, parameter_hash(policy))

    def test_zero_mean_initial_actor_not_changed_by_statistics(self):
        policy = self.models()['policy']
        x, a = torch.randn(128, 6) + 3., torch.randn(128, 3)
        original = RunningStandardScaler(size=6, device='cpu')
        changed = deepcopy(original); changed(x, train=True)
        result = compare(policy, original, changed, x, a)
        self.assertGreater(result['feature_normalized_difference_rms'], 0.)
        self.assertEqual(result['analytic_conditional_KL_mean'], 0.)
        self.assertEqual(result['logprob_change_abs_max'], 0.)

    def test_value_scaler_sequential_coordinate_mismatch(self):
        values = torch.linspace(-.2, .2, 128).reshape(128, 1, 1) + .5
        returns = torch.linspace(-20., -2., 128).reshape(128, 1, 1)
        scaler = RunningStandardScaler(size=1, device='cpu')
        encoded_before = scaler(values, train=True)
        scaler(returns, train=True)
        encoded_after = scaler(values)
        self.assertGreater(float((encoded_before - encoded_after).abs().max()), .2)
        frozen = RunningStandardScaler(size=1, device='cpu')
        stored = frozen(values); frozen(returns)
        torch.testing.assert_close(stored, frozen(values), rtol=0, atol=0)

    def test_first_minibatch_gradient_probe_cannot_change_parameters(self):
        models = self.models(); tensors = self.tensors(models)
        cfg = {'discount_factor': 1., 'gae_lambda': .95, 'time_limit_bootstrap': True,
               'ratio_clip': .2, 'value_clip': .2, 'value_loss_scale': 1., 'grad_norm_clip': .5}
        before = {name: parameter_hash(model) for name, model in models.items()}
        for convention in ('legacy', 'fixed_identity', 'frozen_value_only', 'frozen_observation_only'):
            row, actor_gradient, _ = first_minibatch_probe(
                models, tensors, tensors['rewards'], torch.zeros(1, 1), cfg, torch.arange(16), convention)
            self.assertEqual(row['optimizer_steps'], 0)
            self.assertGreater(float(actor_gradient.norm()), 0.)
            self.assertEqual(row['raw_logprob_difference_max'], 0.)
            if convention == 'fixed_identity':
                self.assertEqual(row['value_clip_saturated_fraction'], 0.)
        self.assertEqual(before, {name: parameter_hash(model) for name, model in models.items()})

    def test_zero_reward_advantage_can_be_random_critic_signal(self):
        models = self.models(); tensors = self.tensors(models)
        cfg = {'discount_factor': 1., 'gae_lambda': .95, 'time_limit_bootstrap': True}
        zero = torch.zeros_like(tensors['rewards'])
        _, critic_only = advantages(zero, tensors['values'], torch.zeros(1, 1), tensors, cfg)
        _, weak_reward = advantages(tensors['rewards'] * 1e-8, tensors['values'], torch.zeros(1, 1), tensors, cfg)
        self.assertGreater(cosine(critic_only, weak_reward), .99999)
        # No critic/no reward does not create a spurious normalized signal.
        _, no_signal = advantages(zero, torch.zeros_like(tensors['values']), torch.zeros(1, 1), tensors, cfg)
        self.assertEqual(float(no_signal.abs().max()), 0.)


if __name__ == '__main__':
    main()
