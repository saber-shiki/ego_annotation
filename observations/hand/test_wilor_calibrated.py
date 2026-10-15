import unittest
import numpy as np
from observations.hand.wilor_calibrated import calibrated_translation
from hot3d_eval.compare_hand_backends import apply, score


class CalibrationTests(unittest.TestCase):
    def test_identity(self):
        t = np.array([.1, -.2, .7])
        np.testing.assert_allclose(calibrated_translation(t, 1000, [800,600], [1000,1000,400,300]), t)

    def test_direct_weak_camera_equivalence(self):
        scale, a, b, box = 1.3, .02, -.03, 180.
        center = np.array([207., 319.])
        size, focal0, K = np.array([800.,600.]), 3500., np.array([920.,920.,397.,306.])
        bs = box * scale
        native = np.r_[2*(center-size/2)/bs + [a,b], 2*focal0/bs]
        expected = np.r_[2*(center-K[2:])/bs + [a,b], 2*K[0]/bs]
        np.testing.assert_allclose(calibrated_translation(native, focal0, size, K), expected)
        # Metric anatomy is NOT multiplied by the focal ratio.
        np.testing.assert_allclose(calibrated_translation(native, focal0, size, K)[2], expected[2])

    def test_anisotropic_refused(self):
        with self.assertRaises(ValueError):
            calibrated_translation([0,0,1], 1000, [800,600], [900,950,400,300])

    def test_invalid_camera_refused(self):
        for f in [0, -1, np.nan]:
            with self.assertRaises(ValueError):
                calibrated_translation([0,0,1], f, [800,600], [900,900,400,300])

    def test_batched_transform(self):
        T = np.tile(np.eye(4), (5,1,1)); T[:,:3,3] = [1,2,3]
        points = np.zeros((5,21,3))
        np.testing.assert_allclose(apply(T, points), np.broadcast_to([1,2,3], points.shape))

    def test_missing_temporal_windows(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(5,21,3)) * .01 + [0,0,.6]
        p = dict(camera_joints=x,world_joints=x,camera_vertices=np.tile(x[:,:1],(1,778,1)))
        m, _ = score(p,p,np.array([True,True,False,True,True]),np.arange(5),np.tile([900,900,400,300],(5,1)),[800,600])
        self.assertEqual(m['MPJPE19_camera_mm']['mean'], 0)
        self.assertEqual(m['valid_consecutive_triplets'], 0)
        self.assertIsNone(m['acceleration_error_world_m_s2']['mean'])

    def test_perspective_translation_recovery(self):
        from observations.hand.wilor_perspective import fit_translation
        from observations.hand.wilor_calibrated import project
        rng=np.random.default_rng(21)
        local=rng.normal(size=(21,3))*.025
        true_t=np.array([.04,-.02,.48]);K=np.array([900,900,400,300])
        t,meta=fit_translation(local,project(local+true_t,K),K,[0,0,.7])
        np.testing.assert_allclose(t,true_t,atol=1e-7)
        self.assertLess(meta['after_mean_px'],1e-5)
        self.assertTrue(meta['success'])

    def test_native_and_physical_projection_are_distinct(self):
        rng=np.random.default_rng(27)
        local=rng.normal(size=(5,21,3))*.02
        native_t=np.array([0,0,5.]);source_t=np.array([0,0,.5])
        x=local+source_t
        g=dict(camera_joints=x,world_joints=x,camera_vertices=np.tile(x[:,:1],(1,778,1)))
        p=dict(g,native_camera_joints=local+native_t,native_projection_K=np.tile([9000,9000,400,300],(5,1)))
        m,_=score(p,g,np.ones(5,bool),np.arange(5),np.tile([900,900,400,300],(5,1)),[800,600])
        self.assertEqual(m['projection19_px']['mean'],0)
        self.assertGreater(m['native_network_projection19_px']['mean'],.1)

    def test_joint_order_and_translation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(5,21,3)) * .01 + [0,0,.6]
        g = dict(camera_joints=x,world_joints=x,camera_vertices=np.tile(x[:,:1],(1,778,1)))
        p = {k:v+[0,0,.02] for k,v in g.items()}
        m,_ = score(p,g,np.ones(5,bool),np.arange(5),np.tile([900,900,400,300],(5,1)),[800,600])
        self.assertAlmostEqual(m['MPJPE19_camera_mm']['mean'],20)
        self.assertAlmostEqual(m['wrist_relative_MPJPE19_camera_mm']['mean'],0)
        self.assertAlmostEqual(m['PA_MPJPE21_mm']['mean'],0)


if __name__ == '__main__':
    unittest.main()
