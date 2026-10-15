"""Numerical read-only regression for complete native motion audit."""
import unittest
import numpy as np
from tools.rl_experiments.audit_three_rewards_motion_contact import full_native_motion, stats


class NativeMotionAuditTest(unittest.TestCase):
    def test_full_moments_match_all_explicit_vertices_nonuniform_time(self):
        rng=np.random.default_rng(38);vertices=rng.normal(size=(127,3))*.03
        times=np.array([0.,.04,.08,.13,.19,.25])
        from scipy.spatial.transform import Rotation
        R=Rotation.from_rotvec(rng.normal(size=(len(times),3))*.1).as_matrix()
        R0=Rotation.from_rotvec(rng.normal(size=(len(times),3))*.1).as_matrix()
        t=rng.normal(size=(len(times),3))*.02;t0=rng.normal(size=(len(times),3))*.02
        a={'object_vertices_canonical':vertices,'time_s':times,'object_rotation':R,'object_translation_world':t}
        base={**a,'object_rotation':R0,'object_translation_world':t0}
        result=full_native_motion(a,base)
        native=np.einsum('nj,tij->tni',vertices,R)+t[:,None]
        reference=np.einsum('nj,tij->tni',vertices,R0)+t0[:,None]
        for key,x in [('actual',native),('correction',native-reference)]:
            v=np.diff(x,axis=0)/np.diff(times)[:,None,None]
            acc=np.diff(v,axis=0)/((np.diff(times)[1:]+np.diff(times)[:-1])*.5)[:,None,None]
            for name,array in [('per_edge_full_surface_RMS_speed_mm_s',v),('per_triple_full_surface_RMS_acceleration_mm_s2',acc)]:
                explicit=np.sqrt(np.square(array).sum(-1).mean(-1))*1000
                for metric in ('mean','p50','p95','max'):
                    self.assertAlmostEqual(result[key][name][metric],stats(explicit)[metric],places=8)

    def test_zero_corrections_and_constant_translation(self):
        a={'object_vertices_canonical':np.array([[0.,0.,0.],[.04,.03,0.],[0.,.08,.01]]),
           'time_s':np.array([0.,.03,.07,.11]),'object_rotation':np.repeat(np.eye(3)[None],4,axis=0),
           'object_translation_world':np.c_[np.array([0.,.03,.07,.11])*.01,np.zeros((4,2))]}
        r=full_native_motion(a,a)
        self.assertAlmostEqual(r['actual']['per_edge_full_surface_RMS_speed_mm_s']['mean'],10.)
        self.assertLess(r['actual']['per_triple_full_surface_RMS_acceleration_mm_s2']['max'],1e-12)
        self.assertEqual(r['correction']['per_edge_full_surface_RMS_speed_mm_s']['max'],0.)
        self.assertEqual(r['correction']['per_triple_full_surface_RMS_acceleration_mm_s2']['max'],0.)


if __name__=='__main__':unittest.main()
