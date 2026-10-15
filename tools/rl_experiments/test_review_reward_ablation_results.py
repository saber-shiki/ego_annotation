"""Small regressions for read-only ablation-review calculations."""
from copy import deepcopy
from unittest import TestCase,main
import torch
from tools.rl_experiments.review_reward_ablation_results import scalar_metrics,state_hash


class ReviewAblationTest(TestCase):
    def test_state_hash_exact_and_tensor_change(self):
        a={'weight':torch.tensor([[1.,2.]]),'count':torch.tensor(4)}
        self.assertEqual(state_hash(a),state_hash(deepcopy(a)))
        b=deepcopy(a);b['weight'][0,0]+=.001
        self.assertNotEqual(state_hash(a),state_hash(b))

    def test_video_seed_panel_and_metrics_not_training_target(self):
        hand={k:0. for k in ['joint_pixel_p95','joint_pixel_max','joint_depth_shift_p95_mm',
                             'vertex_correction_mean_mm','correction_velocity_p95_mm_s',
                             'correction_acceleration_p95_mm_s2','nonpositive_joint_depth_count']}
        r={'objective':10.,'training_objective':6.,'removed_objective':4.,'original_formal_objective':9.,'generated_collision_objective':1.,'seed':1,
           'geometry':{'left':hand,'right':hand,'object':{'translation_max_mm':1.,'rotation_max_deg':2.,'correction_acceleration_p95_mm_s2':3.,'correction_rotvec_rate_p95_deg_s':4.},
                       'contact_prior_deviation_mean_abs':[.1,.2],'action_blocks':{}},
           'collision':{'inside_squared_hinge_sum_m2':.3,'max_inside_depth_mm':2.,'total_crossing_edges_over_all_hand_frames':9,'affected_frame_count':5},'full_group_costs':{'hand_pixel':4.}}
        render={'candidate_measurements':[{'raw_SAM2_silhouette_IoU_median':v} for v in [.8,.6,.7]],'per_frame_raw_IoU':[[.8],[.6],[.7,.9]]}
        m=scalar_metrics(r,render)
        self.assertEqual(m['J_full'],10.)
        self.assertEqual(m['J_train'],6.)
        self.assertEqual(m['J_removed'],4.)
        self.assertEqual(m['IoU_median'],.7)
        self.assertAlmostEqual(m['IoU_mean'],.8)
        self.assertEqual(m['right_contact_prob_abs_shift'],.2)


if __name__=='__main__':main()
