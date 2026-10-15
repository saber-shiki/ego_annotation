"""Geometric response and PPO coordinate regressions, synthetic fixtures only."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from data.immutable import freeze
from envs.test_generated_collision_coarse_env import closed_fixture
from envs.test_object_only_coarse_env import fixture_config
from envs.three_rewards_coarse_env import ThreeRewardsCoarseEnv
from rewards.three_rewards import DEFAULTS, GROUPS, NativeMoments, penetration_cost, continuity_samples, squared_norm_mean


def config(models):
    cfg=fixture_config(models);cfg.pop('object_only');cfg.update(reward_scale=1.,three_rewards=deepcopy(DEFAULTS),training_contract='fixed_asinh_raw_value_zero_critic.v1')
    return cfg


class ThreeRewardTests(unittest.TestCase):
    def test_native_moments_exact_all_vertex_not_proxy(self):
        rng=np.random.default_rng(3);v=rng.normal(size=(1234,3));A=rng.normal(size=(3,3));b=rng.normal(size=3)
        moments=NativeMoments.from_vertices(v)
        self.assertAlmostEqual(moments.displacement_squared(A,b),squared_norm_mean(v@A+b),places=12)
        self.assertEqual(moments.count,len(v))

    def test_fixed_penetration_denominator_monotone_and_zero_outside(self):
        kw=dict(epsilon_m=.0001,sigma_m=.0005,deepest_weight=.1)
        a=np.array([.001,.002,0.,-.001]);b=a.copy();b[2]=.0005
        self.assertGreater(penetration_cost(b,**kw),penetration_cost(a,**kw))
        self.assertEqual(penetration_cost(np.full(4,-.001),**kw),0.)

    def test_temporal_constant_vs_jump_and_no_gap_bridge(self):
        t=np.arange(5)/30.;f=np.arange(5);valid=np.ones(5,bool)
        same=np.full((5,2,3),.002);jump=same.copy();jump[2]*=-1
        kwargs=(t,f,valid,.03,1.,squared_norm_mean)
        self.assertEqual(continuity_samples(same,*kwargs),0.)
        self.assertGreater(continuity_samples(jump,*kwargs),0.)
        valid[2]=False
        self.assertEqual(continuity_samples(jump,t,f,valid,.03,1.,squared_norm_mean),0.)

    def test_full_actions_three_groups_zero_relative_and_native_export(self):
        with TemporaryDirectory() as td:
            b,m=closed_fixture(Path(td));env=ThreeRewardsCoarseEnv([b],config(m));obs,_=env.reset(seed=0)
            self.assertEqual(obs.shape,(551,));self.assertEqual(env.action_space.shape,(118,))
            rewards=[]
            for _ in range(3):_,r,done,_,info=env.step(np.zeros(118,np.float32));rewards.append(r)
            np.testing.assert_allclose(rewards,0.,atol=1e-12);self.assertTrue(done)
            self.assertEqual(set(info['factor_ledger']['terms']),set(GROUPS))
            export=env.export_annotation(Path(td)/'candidate.json')
            self.assertFalse(export['document']['preprocessed_state_adapter']['reference_is_GT'])
            source=env.state.episode.objects['object:0'];self.assertFalse(source['signed_geometry_ready'])
            with np.load(export['npz']) as a:np.testing.assert_array_equal(a['object_faces'],source['faces'])
            env.reset(seed=0);a=np.zeros(118,np.float32);layout=env.state.episode.layout
            for s in ['left','right']:a[layout.block(f'hand.{s}.translation').start]=.2
            total=0.
            for _ in range(3):_,r,done,_,info=env.step(a);total+=r
            self.assertAlmostEqual(total,-info['zero_relative_objective'],places=8)
            self.assertGreater(info['factor_ledger']['terms']['reference_alignment'],0.)
            from envs.dynamics import decode_candidate
            changed=env.state.copy();changed.position=0
            before=env.evaluator.objective(changed,3)
            changed.contact_logits[0,0]-=20
            changed.term_cache.pop(0,None)
            after=env.evaluator.objective(changed,3)
            self.assertAlmostEqual(before.terms['nonpenetration_contact'],after.terms['nonpenetration_contact'],places=7)
            env.close()

    def test_fixed_preprocessor_values_and_official_training_smoke(self):
        import torch,json,train
        from models.fixed_preprocessor import FixedAsinh
        pre=FixedAsinh();x=torch.randn(50,551)*200.;before=deepcopy(pre.state_dict())
        torch.testing.assert_close(pre(x),pre(x,train=True),rtol=0,atol=0)
        self.assertTrue(all(torch.equal(v,pre.state_dict()[k]) for k,v in before.items()))
        with TemporaryDirectory() as td:
            b,m=closed_fixture(Path(td));root=Path(td);ec=config(m)
            cfg={'runtime':{'dataset_class':'data.coarse_training:CoarseTrainingDataset','environment_class':'envs.three_rewards_coarse_env:ThreeRewardsCoarseEnv','training_contract':ec['training_contract']},
                 'environment':ec,'model':{'encoder':{'hidden_dims':[16,16]},'policy':{'initial_log_std':-2.5,'max_log_std':-2.},'value':{'zero_value_init':True}},
                 'ppo':{'rollouts':3,'learning_epochs':1,'mini_batches':1,'discount_factor':1.,'gae_lambda':1.,'learning_rate':1e-4,'random_timesteps':0,'value_clip':.2},
                 'trainer':{'timesteps':6,'disable_progressbar':True},'experiment':{'write_interval':0,'checkpoint_interval':0}}
            index=root/'index.json';index.write_text(json.dumps({'schema':'synthetic.coarse.v1'}))
            env=train.build_env(cfg,dataset_path=index,split='all',dataset=[b]);agent=train.build_agent(env,cfg,training=True,experiment_dir=root/'runs',experiment_name='test')
            o,_=env.reset();v,_=agent.value.act({'observations':agent._observation_preprocessor(o)},role='value');self.assertEqual(float(v.abs().max()),0.)
            self.assertEqual(agent._value_preprocessor(v,inverse=True).shape,v.shape)
            initial={k:v.detach().clone() for k,v in agent.policy.state_dict().items()}
            train.build_trainer(env,agent,cfg).train();self.assertTrue(train.finite_policy_parameter_changed(agent,initial))
            agent.save(str(root/'agent.pt'));new=train.build_agent(env,cfg,training=False,experiment_dir=root/'reload',experiment_name='test');new.load(str(root/'agent.pt'))
            for k,v in agent.policy.state_dict().items():torch.testing.assert_close(v,new.policy.state_dict()[k])
            cfg['environment'].pop('mano_models');self.assertEqual(train.compatibility_contract(env,cfg)['training_contract'],ec['training_contract'])
            env.close()


if __name__=='__main__':unittest.main()
