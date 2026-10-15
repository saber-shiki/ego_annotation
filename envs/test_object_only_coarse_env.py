"""Synthetic mechanism/likelihood regressions for object-only safe residual PPO."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from data.test_coarse_training import fixture
from data.coarse_training import ADAPTER
from envs.object_only_coarse_env import CONTRACT, ObjectOnlyCoarseEnv, physical_delta, radial_tanh


def fixture_config(models):
    return {"max_objects":1,"contact_variables_per_hand":5,"mano_models":models,
            "preprocessed_state_adapter":ADAPTER,"reward_profile":"formal_annotation","reward_scale":1e-5,
            "action_scales":{"hand_root_rad":.05,"hand_articulation_rad":.05,"hand_translation_m":.01,
                             "object_rotation_rad":.05,"object_translation_m":.01,"contact_logit":1.},
            "object_only":{"contract":CONTRACT,"max_rotation_deg":.5,"max_translation_m":.002}}


class ObjectOnlyTests(unittest.TestCase):
    def test_radial_map_zero_odd_derivative_and_norm_bounds(self):
        np.testing.assert_array_equal(radial_tanh(np.zeros(3),.002),np.zeros(3))
        v=np.array([.1,-.2,.3]);d=radial_tanh(v,.002)
        self.assertLessEqual(np.linalg.norm(d),.002+1e-14)
        np.testing.assert_allclose(radial_tanh(-v,.002),-d)
        v=np.array([1e-8,2e-8,0.]);np.testing.assert_allclose(radial_tanh(v,.002),v,atol=1e-16)
        c=fixture_config({})
        for a in (np.ones(6)*1000.,np.r_[[20.,0,0],[.2,-.3,.4]]):
            d=physical_delta(a,c)
            self.assertLessEqual(np.linalg.norm(d[:3]),np.deg2rad(.5)+1e-12)
            self.assertLessEqual(np.linalg.norm(d[3:]),.002+1e-12)
        with self.assertRaises(ValueError):physical_delta(np.zeros(118),c)
        with self.assertRaises(ValueError):physical_delta(np.full(6,np.nan),c)

    def test_zero_relative_return_controlled_object_and_frozen_hands_contact(self):
        with tempfile.TemporaryDirectory() as td:
            b,_,archive,models=fixture(Path(td));cfg=fixture_config(models)
            env=ObjectOnlyCoarseEnv([b],cfg);obs,_=env.reset(seed=0)
            self.assertEqual(env.action_space.shape,(6,));self.assertEqual(obs.shape,(551,))
            initial=env.state.copy();zero_hand={side:None for side in ("left","right")}
            rewards=[];raw_rewards=[]
            for _ in range(3):
                _,r,done,truncated,info=env.step(np.zeros(6,np.float32));rewards.append(r);raw_rewards.append(info['original_reward'])
            self.assertTrue(done);self.assertFalse(truncated)
            np.testing.assert_allclose(rewards,np.zeros(3),atol=1e-13)
            self.assertAlmostEqual(sum(raw_rewards),-1e-5*info['objective'],places=10)
            zero_cost=info['objective'];zero_hand={s:env.state.hand_vertices[s].copy() for s in zero_hand}
            zero_contact=env.state.contact_logits.copy();zero_ledger=info['factor_ledger']
            env.reset(seed=0);cache=env.zero_prefix_objectives
            rewards=[]
            for i in range(3):
                a=np.array([.5,-.2,.1,.8,-.1,.2],np.float32) if i==0 else np.zeros(6,np.float32)
                _,r,done,_,info=env.step(a);rewards.append(r)
                np.testing.assert_array_equal(info['raw_action'],a)
                self.assertEqual(info['executed_action'].shape,(6,))
            self.assertAlmostEqual(sum(rewards),-1e-5*(info['objective']-zero_cost),places=10)
            self.assertNotEqual(info['objective'],zero_cost)
            self.assertIs(cache,env.zero_prefix_objectives)
            for s in zero_hand:np.testing.assert_array_equal(env.state.hand_vertices[s],zero_hand[s])
            np.testing.assert_array_equal(env.state.contact_logits,zero_contact)
            for k,v in zero_ledger['terms'].items():
                if k.startswith('hand/'):self.assertAlmostEqual(info['factor_ledger']['terms'][k],v,places=9)
            obj=initial.episode.objects['object:0']
            rotation=Rotation.from_matrix(env.state.object_rotation['object:0']@obj['rotation'].transpose(0,2,1)).magnitude()
            trans=np.linalg.norm(env.state.object_translation['object:0']-obj['translation'],axis=1)
            self.assertLessEqual(rotation.max(),np.deg2rad(.5)+1e-6)
            self.assertLessEqual(trans.max(),.002+1e-8)
            export=env.export_annotation(Path(td)/'candidate.json')
            self.assertEqual(export['document']['preprocessed_state_adapter']['policy_subspace'],CONTRACT)
            with np.load(export['npz'],allow_pickle=False) as z:np.testing.assert_array_equal(z['object_faces'],archive['faces'])
            with self.assertRaises(RuntimeError):env.step(np.zeros(6,np.float32))

    def test_six_dimensional_raw_gaussian_likelihood_and_vector_std(self):
        import torch
        from models.policy import ResidualGaussianPolicy
        import gymnasium as gym
        policy=ResidualGaussianPolicy(gym.spaces.Box(-np.inf,np.inf,(551,),np.float32),None,
            gym.spaces.Box(-np.inf,np.inf,(6,),np.float32),'cpu',initial_log_std=[-2.8]*3+[-3.]*3,max_log_std=-2.5)
        obs=torch.zeros((1,551));action=torch.tensor([[.5,-.2,.1,.8,-.1,.2]])
        _,out=policy.act({'observations':obs,'taken_actions':action},role='policy')
        expected=torch.distributions.Normal(torch.zeros_like(action),torch.exp(policy.log_std_parameter)).log_prob(action).sum(-1,keepdim=True)
        torch.testing.assert_close(out['log_prob'],expected)
        delta=physical_delta(action[0].numpy(),fixture_config({}))
        mapped=torch.tensor(delta/.05,dtype=torch.float32).reshape(1,6)
        self.assertFalse(torch.isclose(expected,torch.distributions.Normal(torch.zeros_like(action),torch.exp(policy.log_std_parameter)).log_prob(mapped).sum(-1,keepdim=True)).all())
        with self.assertRaises(ValueError):ResidualGaussianPolicy(policy.observation_space,None,policy.action_space,'cpu',initial_log_std=[-3.]*5)

    def test_official_ppo_update_checkpoint_reload_rejects_118D(self):
        import torch
        import train
        import rollout
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);b,_,_,models=fixture(root)
            cfg={"runtime":{"dataset_class":"data.coarse_training:CoarseTrainingDataset","environment_class":"envs.object_only_coarse_env:ObjectOnlyCoarseEnv"},
                 "environment":fixture_config(models),"model":{"encoder":{"hidden_dims":[16,16]},
                 "policy":{"initial_log_std":[-2.8]*3+[-3.]*3,"max_log_std":-2.5},"value":{"use_states":False}},
                 "ppo":{"rollouts":3,"learning_epochs":1,"mini_batches":1,"discount_factor":1.,"random_timesteps":0,"learning_rate":1e-4},
                 "trainer":{"timesteps":6,"disable_progressbar":True},"experiment":{"write_interval":0,"checkpoint_interval":0}}
            idx=root/'index.json';idx.write_text(json.dumps({'schema':'synthetic.coarse_training.v1'}))
            env=train.build_env(cfg,dataset_path=idx,split='all',device='cpu',dataset=[b])
            agent=train.build_agent(env,cfg,training=True,experiment_dir=root/'runs',experiment_name='test')
            before={k:v.detach().clone() for k,v in agent.policy.state_dict().items()}
            train.build_trainer(env,agent,cfg).train()
            self.assertTrue(train.finite_policy_parameter_changed(agent,before))
            train.assert_agent_checkpoint_finite(agent)
            checkpoint=root/'agent.pt';agent.save(str(checkpoint))
            reload=train.build_agent(env,cfg,training=False,experiment_dir=root/'reload',experiment_name='test')
            reload.load(str(checkpoint))
            for k,v in agent.policy.state_dict().items():torch.testing.assert_close(v,reload.policy.state_dict()[k])
            # Compatibility includes the explicit six-action environment and contract.
            cfg['environment'].pop('mano_models')
            contract=train.compatibility_contract(env,cfg)
            self.assertIn('(6,)',contract['action_space'])
            provenance={'skrl_version':'2.1.0','checkpoint':{'sha256':train._sha256(checkpoint)},'compatibility':{**contract,'action_space':'Box(-inf, inf, (118,), float32)'}}
            with self.assertRaisesRegex(ValueError,'compatibility'):rollout._verify_checkpoint_contract(provenance=provenance,checkpoint=checkpoint,cfg=cfg,env=env)
            env.close()


if __name__=='__main__':unittest.main()
