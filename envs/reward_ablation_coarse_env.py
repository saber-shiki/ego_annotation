"""Full 118D PPO ablations, with invariant full evaluation on every state."""
from dataclasses import replace
from data.immutable import freeze
from envs.generated_collision_coarse_env import GeneratedCollisionCoarseEnv,GeneratedCollisionReward
from rewards.ablation import CONTRACT,GROUPS,grouped,select_ledger


class RewardAblationEvaluator(GeneratedCollisionReward):
    def __init__(self,cfg,device='cpu'):
        super().__init__(cfg,device)
        options=cfg.get('reward_ablation',{})
        if options.get('contract')!=CONTRACT or options.get('drop_group') not in ('full',*GROUPS):
            raise ValueError('explicit reviewed single-family reward ablation required')
        self.drop_group=options['drop_group']
        self.last_full_ledger=None
        self.last_removed={}

    def objective(self,state,end_position):
        full=super().objective(state,end_position)
        selected,removed=select_ledger(full,self.drop_group)
        self.last_full_ledger,self.last_removed=full,removed
        return selected


class RewardAblationCoarseEnv(GeneratedCollisionCoarseEnv):
    def __init__(self,dataset,cfg,device='cpu'):
        super().__init__(dataset,cfg,device)
        self.evaluator=RewardAblationEvaluator(cfg,device)

    def reset(self,**kwargs):
        observation,info=super().reset(**kwargs)
        episode=self.state.episode
        adapter={**dict(episode.provenance['preprocessed_state_adapter']),
                 'reward_ablation':dict(self.cfg['reward_ablation']),
                 'evaluation_objective':'all original formal factors plus generated-mesh collision; no dropped costs'}
        self.state.episode=replace(episode,provenance=freeze({**dict(episode.provenance),'preprocessed_state_adapter':adapter}))
        info.update(reward_ablation=dict(self.cfg['reward_ablation']))
        return observation,info

    def step(self,action):
        result=super().step(action)
        full=self.evaluator.last_full_ledger
        info=result[-1]
        full_groups=grouped(full.terms)
        collision=full_groups['generated_collision']
        info.update(training_objective=info['objective'],full_evaluation_objective=full.objective,
                    original_formal_objective=full.objective-collision,generated_collision_objective=collision,
                    full_factor_ledger=full.document(),full_group_costs=full_groups,
                    removed_objective=sum(self.evaluator.last_removed.values()),
                    reward_ablation=dict(self.cfg['reward_ablation']))
        return result
