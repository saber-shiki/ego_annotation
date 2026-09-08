import sys,unittest,json,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_v19_rgb_pose_initializer import initialize

class RGBInitializer(unittest.TestCase):
    def fixture(self):
        R=Rotation.from_euler('y',.7).as_matrix();t0=np.array([.1,.2,.8]);t1=np.array([.2,.15,.6]);E=t1-R@t0
        poses={'object_id':'thing','pose_rows':[{'frame_idx':i,'status':'fit_to_visible_depth_samples','rigid_pose_observation_eligible':True,
            'rotation_world_from_completed_canonical_matrix':np.eye(3).tolist(),'translation_world_m':t.tolist(),'observed_to_mesh_final':{'median_m':.001}} for i,t in [(0,t0),(3,t0),(7,t0)]]}
        ann={'frames':[{'frame_idx':i,'objects':[]} for i in [0,3,7]]}
        arrays={'source_frame_idx':np.array([0]),'target_frame_idx':np.array([3]),'rotation_source_to_target_world':np.array([R]),'translation_source_to_target_world_m':np.array([E]),'accepted':np.array([True])}
        quality=[{'source_frame_idx':0,'target_frame_idx':3,'status':'accepted','source_3d_conditioning':.1,'inlier_fraction':1.,'reprojection_median_px':.1}]
        return poses,ann,arrays,quality,R,t1
    def test_relative_motion_not_flattened_and_original_untouched(self):
        base,ann,arrays,q,R,t=self.fixture();before=json.dumps(base,sort_keys=True);out=initialize(base,ann,'thing',arrays,q,0);rows={r['frame_idx']:r for r in out['pose_rows']}
        np.testing.assert_allclose(rows[3]['rotation_world_from_completed_canonical_matrix'],R,atol=1e-9)
        np.testing.assert_allclose(rows[3]['translation_world_m'],t,atol=1e-9)
        self.assertEqual(before,json.dumps(base,sort_keys=True));self.assertTrue(rows[7]['rgb_initializer']['uncertain'])
        self.assertNotIn('observed_to_mesh_final',rows[3]);self.assertIn('source_P14_distance_diagnostics',rows[3])
    def test_rejected_edge_not_consumed(self):
        base,ann,arrays,q,R,t=self.fixture();arrays['accepted'][:]=False;q[0]['status']='rejected';out=initialize(base,ann,'thing',arrays,q,0)
        self.assertEqual(out['rgb_full_timeline_initialization']['accepted_edge_count'],0)
        np.testing.assert_allclose(out['pose_rows'][1]['rotation_world_from_completed_canonical_matrix'],np.eye(3))
    def test_conflicting_acceptance_fails(self):
        base,ann,arrays,q,_,_=self.fixture();q[0]['status']='rejected'
        with self.assertRaisesRegex(ValueError,'acceptance mismatch'):initialize(base,ann,'thing',arrays,q,0)
    def test_invalid_rotation_fails(self):
        base,ann,arrays,q,_,_=self.fixture();arrays['rotation_source_to_target_world'][:]=0
        with self.assertRaisesRegex(ValueError,'invalid world-relative'):initialize(base,ann,'thing',arrays,q,0)

    def test_quality_nan_rejected(self):
        base,ann,arrays,q,_,_=self.fixture();q[0]['inlier_fraction']=float('nan')
        with self.assertRaisesRegex(ValueError,'quality evidence'):initialize(base,ann,'thing',arrays,q,0)

    def test_direct_and_latent_rows_survive_P14b_P15_without_promotion(self):
        from types import SimpleNamespace
        from filter_v19_rigid_completion_multiview_support import trusted_pose_rows
        from solve_v19_rigid_object_pose_graph import build_pose_rows
        base,ann,arrays,q,_,_=self.fixture()
        base['pose_rows'][-1].update(status='interpolated',rigid_pose_observation_eligible=False)
        ann['frames'].append({'frame_idx':8,'objects':[]})
        out=initialize(base,ann,'thing',arrays,q,0)
        self.assertEqual([r['frame_idx'] for r in trusted_pose_rows(out)],[0,3])
        args=SimpleNamespace(complete_full_timeline_rigid_pose=True,
            max_rigid_pose_interpolation_gap_frames=10,max_rigid_pose_extrapolation_gap_frames=10)
        rows,summary=build_pose_rows(out['pose_rows'],[],np.zeros(0),args,False)
        self.assertEqual([r['frame_idx'] for r in rows],[0,3,7,8])
        self.assertFalse(rows[2]['rigid_pose_observation_eligible'])
        self.assertEqual(rows[3]['status'],'rgb_initialized_uncertain')
        self.assertEqual(rows,out['pose_rows'])

    def test_actual_p14_schema_reaches_p14b_and_preserves_measurements(self):
        from filter_v19_rigid_completion_multiview_support import trusted_pose_rows
        base,ann,arrays,q,_,_=self.fixture()
        base.update(method='fit_v18_compact_rigid_object_pose',pose_hypothesis_mesh_labeled='/fresh/pose.ply',frame_fit_rows=base['pose_rows'])
        out=initialize(base,ann,'thing',arrays,q,0)
        self.assertEqual(out['inputs']['completed_mesh'],'/fresh/pose.ply')
        self.assertEqual(out['frame_fit_rows'],out['pose_rows'])
        self.assertEqual(out['source_P14_frame_fit_rows'],base['frame_fit_rows'])
        self.assertEqual(len(trusted_pose_rows(out)),3)
        self.assertFalse(out['pose_rows'][1]['rgb_initializer']['direct_metric_observation_added'])

if __name__=='__main__':unittest.main()
