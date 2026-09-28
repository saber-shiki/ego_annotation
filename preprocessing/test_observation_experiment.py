import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from preprocessing.test_visible_surface import fixture
from preprocessing.geometry import TriangleMesh,project
from preprocessing.observation_experiment import Evidence,binding_visibility,fixed_rgb_score,choose,observability,normalized_weights,distinct


class ObservationExperimentTests(unittest.TestCase):
    def test_holdout_is_identity_not_candidate_inlier(self):
        e=Evidence(np.zeros((4,3)),np.zeros((4,2)),np.array([3,4,3,5]),np.array([0,0,8,8]),np.ones(4,bool))
        np.testing.assert_array_equal(e.hold,[True,False,True,False])
    def test_missed_binding_remains_in_score_population(self):
        e=Evidence(np.array([[0.,0.,0.],[0.,0.,0.]]),np.array([[10.,10.],[10.,10.]]),np.array([0,3]),np.array([0,0]),np.array([True,False]))
        cost=fixed_rgb_score(e,np.eye(3),np.array([0.,0.,1.]),np.array([100.,100.,10.,10.]),e.hold)
        self.assertAlmostEqual(cost,1.5)
    def test_backface_binding_not_a_visible_material_point(self):
        inp,_,_,_,truth=fixture(scale=1.)
        mesh=TriangleMesh(inp.vertices,inp.faces);t=truth[0]
        X=np.array([[0.,0.,-.04],[0.,0.,.04]]);uv=project(X+t,inp.K[0])
        np.testing.assert_array_equal(binding_visibility(mesh,X,uv,inp.K[0],np.eye(3),t),[True,False])
    def test_leave_one_changes_only_specified_family(self):
        scores=[{'rgb':0.,'mask':.1,'raw_mask':.8,'surface':1.}, {'rgb':1.,'mask':.2,'raw_mask':.1,'surface':0.}]
        self.assertEqual(choose(scores,'multi')[0],0)
        self.assertEqual(choose(scores,'no_rgb')[0],1)
        self.assertEqual(choose(scores,'no_surface')[0],0)
        self.assertEqual(choose(scores,'no_mano')[0],choose(scores,'raw_mask')[0])
    def test_missing_depth_not_manufactured(self):
        score={'rgb':None,'mask':.4,'raw_mask':.5,'surface':None}
        self.assertAlmostEqual(choose([score],'multi')[1][0],.4)
    def test_reference_populations_balanced(self):
        w=normalized_weights(np.array([0,0,0,1]))
        self.assertAlmostEqual(w[:3].sum(),.5);self.assertAlmostEqual(w[3],.5)
    def test_large_real_angle_is_not_clamped(self):
        R=Rotation.from_euler('y',100,degrees=True).as_matrix()
        self.assertTrue(distinct(R,np.zeros(3),[(np.eye(3),np.zeros(3),'base')]))
    def test_observability_reports_six_scaled_directions(self):
        rng=np.random.default_rng(47);X=rng.normal(0,.04,(30,3));t=np.array([0.,0.,.7]);K=np.array([300.,300.,96.,96.])
        e=Evidence(X,project(X+t,K),np.arange(30),np.zeros(30,int),np.ones(30,bool))
        result=observability(e,np.eye(3),t,K,.05)
        self.assertTrue(result['available']);self.assertEqual(len(result['singular_values']),6);self.assertFalse(result['covariance_calibrated'])

if __name__=='__main__':unittest.main()
