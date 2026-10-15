from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from envs.test_generated_collision_coarse_env import config,closed_fixture
from envs.reward_ablation_coarse_env import RewardAblationCoarseEnv
from rewards.ablation import CONTRACT,GROUPS
from tools.rl_experiments.reward_ablation_learning import metrics,parameter_hash,VARIANTS


class AblationHarnessTest(unittest.TestCase):
    def test_matrix_coverage_and_zero_metrics(self):
        self.assertEqual(set(VARIANTS),{'full',*GROUPS})
        self.assertEqual(len(VARIANTS),13)
        with TemporaryDirectory() as td:
            b,m=closed_fixture(Path(td));cfg=config(m);cfg['reward_ablation']={'contract':CONTRACT,'drop_group':'full'}
            env=RewardAblationCoarseEnv([b],cfg);env.reset(seed=0)
            zero_prob=(1/(1+np.exp(-env.state.contact_logits))).mean(axis=-1)
            rows=[]
            for i in range(3):
                _,_,_,_,info=env.step(np.zeros(118,np.float32))
                rows.append({'executed_delta':info['executed_delta'].tolist(),'contact_mean_probability':zero_prob[i].tolist()})
            a=env.export_annotation()['arrays']
            value=metrics(a,a,rows,zero_prob)
            for side in ['left','right']:
                self.assertEqual(value[side]['joint_pixel_p95'],0.)
                self.assertEqual(value[side]['vertex_correction_max_mm'],0.)
                self.assertEqual(value[side]['correction_acceleration_p95_mm_s2'],0.)
            self.assertEqual(value['object']['translation_max_mm'],0.)
            self.assertEqual(value['contact_prior_deviation_mean_abs'],[0.,0.])
            self.assertEqual(len(value['action_blocks']),10)

    def test_parameter_hash_detects_real_parameter_change(self):
        import torch
        model=torch.nn.Linear(3,2)
        old=parameter_hash(model)
        self.assertEqual(old,parameter_hash(model))
        with torch.no_grad():model.weight[0,0]+=.001
        self.assertNotEqual(old,parameter_hash(model))


if __name__=='__main__':unittest.main()
