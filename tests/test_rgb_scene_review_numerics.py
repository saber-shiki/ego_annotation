"""Accepted numerical review counterexamples; CPU geometry, not video quality."""
import copy
from dataclasses import replace
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rgb_scene_pose_kernels import PoseNode, RGBEdge, RGBReprojectionFactor
from rgb_scene_pose_graph import GraphConfig, solve, residual_blocks, reprojection_depths, depth_state


def node(i, z=0):
    return PoseNode(i,np.eye(3),np.array([0.,0.,z]),None,'test',False,1.)


def factor():
    return RGBReprojectionFactor(0,1,0,1,np.array([[0.,0.,0.],[.005,0,.005],[-.005,0,.005]]),
        np.array([[0,0],[.005/.305,0],[-.005/.305,0]]),np.array([1.,1.,0.,0.]),np.eye(3),np.zeros(3),np.ones(3),1.)


class RestorationTests(unittest.TestCase):
    def test_bounded_restoration_and_independent_rotation(self):
        nodes=[node(0),node(1,-.2),node(2,1.)];original=copy.deepcopy(nodes)
        rotation=Rot.from_euler('z',45,degrees=True).as_matrix()
        edges=[RGBEdge(0,1,np.eye(3),np.array([0.,0.,.3]),1.),RGBEdge(0,2,rotation,np.array([0.,0.,1.]),1.)]
        cfg=GraphConfig(outer_iterations=14,inner_evaluations=60,observed_weight=0,
            correction_velocity_rad_s=100,correction_velocity_m_s=100,acceleration_rad_s2=100,acceleration_m_s2=100)
        poses,report=solve(nodes,[0,1,2],edges,0,cfg,[factor()])
        self.assertGreater(poses[1][1][2],0)
        self.assertLess(Rot.from_matrix(poses[2][0]@rotation.T).magnitude(),.03)
        restored=[r for r in report['trace'] if r['stage']=='SE3' and r['accepted'] and r['depth_violation_before_m']>0]
        self.assertGreaterEqual(len(restored),4)
        for r in restored:
            self.assertLess(r['depth_violation_after_m'],r['depth_violation_before_m'])
            self.assertLessEqual(r['max_translation_step_component_m'],.05)
        self.assertEqual(report['depth_restoration']['remaining_invalid'],0)
        for a,b in zip(nodes,original):
            np.testing.assert_array_equal(a.initial_translation,b.initial_translation)
            np.testing.assert_array_equal(a.initial_rotation,b.initial_rotation)
        np.testing.assert_array_equal(poses[0][1],nodes[0].initial_translation)

    def test_feasible_samples_preserved_and_disabled_samples_have_no_authority(self):
        nodes=[node(0),node(1,.02)];f=factor()
        edges=[RGBEdge(0,1,np.eye(3),np.array([0.,0.,-.3]),1.)]
        poses,report=solve(nodes,[0,1],edges,0,GraphConfig(outer_iterations=4),[f])
        self.assertTrue(np.all(reprojection_depths(f,poses,nodes)>=1e-6))
        bad=replace(f,source_world=np.full((3,3),np.nan))
        for cfg,disabled in [(GraphConfig(reprojection_weight=0),bad),(GraphConfig(),replace(bad,edge_weight=0)),(GraphConfig(),replace(bad,weights=np.zeros(3)))]:
            result,_=solve(nodes,[0,1],edges,0,cfg,[disabled])
            self.assertLess(result[1][1][2],0)
        mixed=replace(f,source_world=np.vstack([f.source_world,np.full((1,3),np.nan)]),target_uv=np.vstack([f.target_uv,[[0,0]]]),weights=np.array([1.,1.,1.,0.]))
        solve(nodes,[0,1],edges,0,GraphConfig(outer_iterations=2),[mixed])
        with self.assertRaisesRegex(ValueError,'nonfinite'):solve(nodes,[0,1],edges,0,GraphConfig(),[bad])

    def test_depth_endpoint_support_rotation_eligibility_and_unrecoverable(self):
        nodes=[node(0),node(1,-.2)];poses={n.frame_idx:(n.initial_rotation,n.initial_translation) for n in nodes};f=factor()
        baseline=reprojection_depths(f,poses,nodes)
        for idx in (0,1):
            changed=copy.deepcopy(poses);r,t=changed[idx];changed[idx]=(r,t+[0,0,.001])
            self.assertGreater(np.linalg.norm(reprojection_depths(f,changed,nodes)-baseline),.001)
            changed[idx]=(Rot.from_euler('y',.1).as_matrix()@r,t)
            np.testing.assert_array_equal(reprojection_depths(replace(f,rotation_eligible=False),changed,nodes),baseline)
        # Self-endpoint material projection cannot be restored by object updates.
        fixed=replace(f,source_frame_idx=0,target_frame_idx=0,source_pos=0,target_pos=0,source_world=f.source_world+[0,0,-1])
        result,report=solve(nodes,[0,1],[RGBEdge(0,1,np.eye(3),np.array([0.,0.,.3]),1.)],0,GraphConfig(outer_iterations=12),[fixed])
        self.assertEqual(report['depth_restoration']['remaining_invalid'],3)
        self.assertGreater(result[1][1][2],.1)
        self.assertIn(0,report['depth_restoration']['affected_frame_ids'])


class CorrectionTests(unittest.TestCase):
    def test_branch_cut_nonuniform_time_different_initial_and_endpoint_jacobians(self):
        for angles in ([179,181],[181,179]):
            nodes=[node(0),node(1)]
            nodes[0].initial_rotation=Rot.from_euler('x',.7).as_matrix()
            nodes[1].initial_rotation=Rot.from_euler('y',-.9).as_matrix()
            poses={i:(Rot.from_euler('z',a,degrees=True).as_matrix()@nodes[i].initial_rotation,np.zeros(3)) for i,a in enumerate(angles)}
            for gap in (.2,1.7):
                cfg=GraphConfig()
                def temporal(p):return residual_blocks(p,nodes,[0,gap],[],[],[],cfg,True)[2]
                deps,value=temporal(poses);self.assertEqual(deps,(0,1))
                self.assertAlmostEqual(np.linalg.norm(value),np.deg2rad(2)/np.sqrt(gap)/cfg.correction_velocity_rad_s)
                for idx in (0,1):
                    changed=copy.deepcopy(poses);r,t=changed[idx];changed[idx]=(Rot.from_euler('x',1e-5).as_matrix()@r,t)
                    self.assertGreater(np.linalg.norm(temporal(changed)[1]-value),1e-7)

    def test_large_rotation_recovery_crosses_principal_cut(self):
        nodes=[node(0),node(1),node(2)]
        nodes.append(node(3))
        truth=[np.eye(3),Rot.from_euler('z',179,degrees=True).as_matrix(),Rot.from_euler('z',181,degrees=True).as_matrix(),Rot.from_euler('z',90,degrees=True).as_matrix()]
        edges=[RGBEdge(0,3,truth[3],np.zeros(3),20.),RGBEdge(3,1,truth[1]@truth[3].T,np.zeros(3),20.)]+[RGBEdge(1,2,truth[2]@truth[1].T,np.zeros(3),20.)]
        cfg=GraphConfig(outer_iterations=16,inner_evaluations=80,prior_rotation_rad=20,acceleration_rad_s2=100,correction_velocity_rad_s=20)
        poses,_=solve(nodes,[0,1,1.3,2],edges,0,cfg)
        for i in (1,2):self.assertLess(Rot.from_matrix(poses[i][0]@truth[i].T).magnitude(),.02)

if __name__=='__main__':unittest.main()
