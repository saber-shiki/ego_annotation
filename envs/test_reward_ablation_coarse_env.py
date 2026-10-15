"""Mechanism tests for exact reward selection, invariant full scores and states."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from rewards.ablation import CONTRACT,GROUPS,classify,grouped,select_ledger
from rewards.evaluator import FactorLedger
from envs.test_generated_collision_coarse_env import config,closed_fixture
from envs.reward_ablation_coarse_env import RewardAblationCoarseEnv

NAMES = [
 'hand/left/joint_pixel_deadband/frame=0','hand/right/joint_depth_deadband/frame=0',
 'hand/left/visible_surface_depth_order','hand/right/root_prior','hand/left/translation_velocity',
 'hand/right/articulation_soft_bound','object/object:0/rotation_acceleration',
 'object/object:0/observed_point_to_plane','object/object:0/rgb_edge/factor=0/rotation',
 'object/object:0/rgb_reprojection/factor=0','contact/left/state_prior',
 'hypothesis/object:0/right/penetration/frame=0']


class RewardAblationTest(unittest.TestCase):
    def test_classifier_exact_groups_unknowns_and_optional(self):
        self.assertEqual(tuple(map(classify,NAMES)),GROUPS)
        self.assertEqual(classify('signed/object:0/left/dense/frame=0'),'optional_measured_signed')
        self.assertEqual(classify('object/object:0/translation_sigma_prior'),'optional_object_priors')
        for name in ['object/object:0/rgb_edge/factor=0/translation','visual/hand/left/root_prior','hand/left/unknown']:
            with self.assertRaises(ValueError):classify(name)

    def test_only_one_family_removed_no_weight_or_denominator_changes(self):
        terms={k:float(i+1) for i,k in enumerate(NAMES)}
        full=FactorLedger(1,terms,{k:{'denominator':7} for k in terms},reward_scale=1e-5)
        for name,group in zip(NAMES,GROUPS):
            selected,removed=select_ledger(full,group)
            self.assertEqual(removed,{name:terms[name]})
            self.assertAlmostEqual(selected.objective,full.objective-terms[name])
            for kept in selected.terms:
                self.assertEqual(selected.terms[kept],full.terms[kept]);self.assertEqual(selected.statistics[kept],full.statistics[kept])
            self.assertEqual(sum(grouped(full.terms).values()),full.objective)
        self.assertEqual(select_ledger(full,'full')[0].objective,full.objective)
        with self.assertRaises(ValueError):select_ledger(full,'hand_pixel,generated_collision')

    def test_same_actions_same_observations_full_scores_and_geometry_all_variants(self):
        with TemporaryDirectory() as td:
            b,models=closed_fixture(Path(td));reference=None
            for drop in ('full',*GROUPS):
                cfg=config(models);cfg['reward_ablation']={'contract':CONTRACT,'drop_group':drop}
                env=RewardAblationCoarseEnv([b],cfg);obs,_=env.reset(seed=0);returns=[];observations=[obs]
                layout=env.state.episode.layout
                for i in range(3):
                    action=np.zeros(118,np.float32)
                    action[layout.block('hand.left.translation').start]=float(i+1)*4
                    action[layout.block('object.object:0.translation').start+1]=.05
                    obs,reward,done,truncated,info=env.step(action);returns.append(reward);observations.append(obs)
                    self.assertFalse(truncated)
                    self.assertAlmostEqual(info['full_evaluation_objective']-info['removed_objective'],info['training_objective'],places=6)
                    self.assertTrue(all(classify(k)!=drop for k in info['factor_ledger']['terms']))
                self.assertTrue(done)
                self.assertAlmostEqual(sum(returns),-1e-5*info['training_objective'],places=9)
                data=(np.stack(observations),info['full_evaluation_objective'],env.state.hand_vertices['left'].copy(),env.state.object_translation['object:0'].copy())
                if reference is None:reference=data
                else:
                    np.testing.assert_array_equal(data[0],reference[0]);self.assertEqual(data[1],reference[1]);np.testing.assert_array_equal(data[2],reference[2]);np.testing.assert_array_equal(data[3],reference[3])
                self.assertFalse(env.state.episode.objects['object:0']['signed_geometry_ready'])

    def test_explicit_contract_required(self):
        with TemporaryDirectory() as td:
            b,m=closed_fixture(Path(td))
            with self.assertRaises(ValueError):RewardAblationCoarseEnv([b],config(m))


if __name__=='__main__':unittest.main()
