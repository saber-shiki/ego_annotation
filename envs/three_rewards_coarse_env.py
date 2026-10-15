"""Explicit full-action experiment using only three geometric reward groups."""
from dataclasses import replace
from data.immutable import freeze
from envs.coarse_annotation_env import CoarseAnnotationEnv
from rewards.three_rewards import ThreeRewardEvaluator, CONTRACT


class ThreeRewardsCoarseEnv(CoarseAnnotationEnv):
    def __init__(self, dataset, cfg, device='cpu'):
        super().__init__(dataset, cfg, device)
        if cfg.get('hard_action_mask') is not None or cfg.get('freeze_translation_without_visible_surface_support', False):
            raise ValueError('three reward experiment preserves both trainable hands')
        if self.action_space.shape != (118,): raise ValueError('requires normal full 118D action flow')
        self.evaluator = ThreeRewardEvaluator(cfg, device)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        e = self.state.episode
        topology = {slot: self.evaluator.scene_for(obj).topology for slot, obj in e.objects.items() if obj.get('entity_id')}
        adapter = {**dict(e.provenance['preprocessed_state_adapter']), 'three_rewards': self.cfg['three_rewards'],
                   'training_contract': self.cfg['training_contract'], 'collision_topology': topology,
                   'reference_is_GT': False, 'hands_frozen': False, 'policy_action_dim': 118,
                   'source_signed_authority_promoted': False}
        self.state.episode = replace(e, provenance=freeze({**dict(e.provenance), 'preprocessed_state_adapter': adapter}))
        self.evaluator.zero_for(self.state)
        info.update(three_reward_contract=CONTRACT, reference_is_GT=False, hands_frozen=False)
        return obs, info

    def step(self, action):
        result = super().step(action)
        info = result[-1]; baseline = self.evaluator.zero_for(self.state)[self.state.position]
        info.update(zero_reference_objective=float(baseline), zero_relative_objective=info['objective']-baseline,
                    return_convention='sum(reward)=-(J_three(final)-J_three_zero(final)); gamma=1')
        return result
