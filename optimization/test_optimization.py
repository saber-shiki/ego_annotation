"""Mechanism tests for rgb_scene_pose_graph, not visual quality claims."""
import copy
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from utils.data import PoseNode,RGBEdge,RGBReprojectionFactor,PointFactor,SolverConfig
from factors.object.pose import rgb_edge_residual
from factors.object.image import rgb_reprojection_residual
from optimization.object.pose import solve_scene_pose as solve,GraphConfig,residual_blocks
from observations.common import build_nodes
from utils.data import time_source_pairs,time_keyframes,RKEY,TKEY
from optimization.object.shape import rebind_canonical


def fixture(ids=None):
    ids=ids or [7,19,33,103,204,510]
    times=np.array([0,.25,.6,1.,1.5,2.])
    truth={i:(Rotation.from_rotvec([.12*t,-.08*t,.35*t]).as_matrix(),np.array([.03*t,.01*t,1+.02*t])) for i,t in zip(ids,times)}
    rng=np.random.default_rng(4);surface=rng.normal(size=(50,3))*[.045,.025,.015]
    nodes=[]
    for k,(i,t) in enumerate(zip(ids,times)):
        r,p=truth[i];biased=Rotation.from_rotvec([.15*t,.05*t,.65*t]).as_matrix()@r
        observed=None if k==2 else surface@r.T+p
        nodes.append(PoseNode(i,biased,p+np.array([.025*t,-.005*t,.01*t]),observed,'synthetic',observed is not None,.95 if observed is None else .1,{}))
    edges=[]
    for ai,a in enumerate(ids):
        for b in ids[ai+1:]:
            ra,ta=truth[a];rb,tb=truth[b];dr=rb@ra.T
            edges.append(RGBEdge(a,b,dr,tb-dr@ta,1.))
    return nodes,times,edges,truth


def error(poses,truth):
    return np.mean([Rotation.from_matrix(poses[i][0]@truth[i][0].T).magnitude() for i in truth])


class PoseGraphTests(unittest.TestCase):
    def test_large_rotation_recovery_requires_evidence_and_ids_do_not_matter(self):
        n,t,e,truth=fixture();source=copy.deepcopy(n)
        cfg=GraphConfig(outer_iterations=5,inner_evaluations=45,observed_weight=0,reprojection_weight=0)
        fitted,report=solve(n,t,e,n[0].frame_idx,cfg)
        baseline={v.frame_idx:(v.initial_rotation,v.initial_translation) for v in n}
        self.assertLess(error(fitted,truth),error(baseline,truth)*.2)
        self.assertGreater(Rotation.from_matrix(fitted[n[-1].frame_idx][0]@n[-1].initial_rotation.T).magnitude(),.7)
        zero,_=solve(n,t,[RGBEdge(x.source_frame_idx,x.target_frame_idx,x.rotation_source_to_target_world,x.translation_source_to_target_world_m,0.) for x in e],n[0].frame_idx,cfg)
        self.assertGreater(error(zero,truth),error(fitted,truth)*3)
        for a,b in zip(n,source):np.testing.assert_array_equal(a.initial_rotation,b.initial_rotation)
        nn,tt,ee,truth2=fixture([2,9,17,28,42,900])
        other,_=solve(nn,tt,ee,nn[0].frame_idx,cfg)
        for a,b in zip(n,nn):np.testing.assert_allclose(fitted[a.frame_idx][0],other[b.frame_idx][0],atol=1e-6)
        self.assertTrue(any(r['stage']=='SE3' and r['observed']['factor_count'] for r in report['trace']))

    def test_robust_long_loop(self):
        n,t,e,truth=fixture();a,b=n[0].frame_idx,n[-1].frame_idx
        e.append(RGBEdge(a,b,Rotation.from_rotvec([2.,.3,-.5]).as_matrix(),np.array([.5,0,0]),1.))
        cfg=GraphConfig(outer_iterations=5,inner_evaluations=50,observed_weight=0,reprojection_weight=0)
        robust,diag=solve(n,t,e,a,cfg)
        linear,_=solve(n,t,e,a,GraphConfig(**{**cfg.__dict__,'loss':'linear'}))
        self.assertLess(error(robust,truth),error(linear,truth))
        self.assertLess(error(robust,truth),.13)
        self.assertEqual(diag['local_RGB_rotation']['edges'][-1]['weight_scale'],0.)
        self.assertEqual(diag['local_RGB_rotation']['edges'][-1]['reason'],'rejected_independent_local_cycle')

    def test_relative_reversed_and_reprojection_source_target_jacobians(self):
        rs=Rotation.from_rotvec([.2,.1,-.1]).as_matrix();rt=Rotation.from_rotvec([-.3,.2,.5]).as_matrix()
        ts=np.array([.1,.2,1.]);tt=np.array([.3,.1,1.2]);poses={7:(rs,ts),19:(rt,tt)}
        d=rt@rs.T;e=RGBEdge(7,19,d,tt-d@ts,1.)
        for x in rgb_edge_residual(e,poses):np.testing.assert_allclose(x,0,atol=1e-12)
        reverse=RGBEdge(19,7,d.T,-d.T@e.translation_source_to_target_world_m,1.)
        for x in rgb_edge_residual(reverse,poses):np.testing.assert_allclose(x,0,atol=1e-12)
        world=np.array([[.1,.05,.02],[.04,-.03,.08],[-.03,.04,.09]])@rs.T+ts
        rc=Rotation.from_rotvec([.1,-.15,.02]).as_matrix();tc=np.array([-.1,.02,-.4]);K=np.array([630,510,300,190])
        camera=(((world-ts)@rs)@rt.T+tt-tc)@rc
        uv=camera[:,:2]/camera[:,2,None]*K[:2]+K[2:]
        f=RGBReprojectionFactor(7,19,0,1,world,uv,K,rc,tc,np.ones(3),1.)
        nodes=[PoseNode(i,*poses[i],None,'test',False,1.) for i in poses];cfg=SolverConfig(rgb_reprojection_weight_scale=1.)
        np.testing.assert_allclose(rgb_reprojection_residual(f,poses,nodes,cfg),0,atol=1e-10)
        for idx in poses:
            changed=dict(poses);r,t=poses[idx];changed[idx]=(Rotation.from_rotvec([1e-5,0,0]).as_matrix()@r,t)
            self.assertGreater(np.linalg.norm(rgb_reprojection_residual(f,changed,nodes,cfg)),1e-5)
        e=RGBEdge(7,19,d,tt-d@ts,1.,{'translation_only':True},d,world.mean(axis=0))
        before=rgb_edge_residual(e,poses)
        changed={7:(rt,ts),19:(rs,tt)}
        for a,b in zip(before,rgb_edge_residual(e,changed)):np.testing.assert_array_equal(a,b)

    def test_observed_factors_have_rotation_and_translation_derivatives(self):
        n,t,e,truth=fixture();poses=dict(truth);a,b=0,1
        sw=n[a].observed_world;tw=n[b].observed_world
        f=PointFactor(a,b,sw,tw,np.tile([0,0,1.],(len(sw),1)),np.ones(len(sw)))
        cfg=GraphConfig(rotation_weight=0,translation_weight=0)
        def obs(p):return np.concatenate([v for _,v in residual_blocks(p,n,t,[],[f],[],cfg)][2*len(n):2*len(n)+2])
        np.testing.assert_allclose(obs(poses),0,atol=1e-10)
        for rotation in (True,False):
            changed=dict(poses);r,p=poses[n[a].frame_idx]
            changed[n[a].frame_idx]=(Rotation.from_rotvec([.01,0,0]).as_matrix()@r,p) if rotation else (r,p+[.001,0,0])
            self.assertGreater(np.linalg.norm(obs(changed)),.001)

    def test_time_pair_planning_and_canonical_rebinding(self):
        ids=[7,19,103,211];times=dict(zip(ids,[0,.1,.8,1.5]))
        keys=time_keyframes(ids,times,.5);self.assertEqual(keys,[7,103,211])
        pairs,_=time_source_pairs(ids,keys,{19,211},3,times,1.)
        self.assertIn((19,7),pairs);self.assertIn((211,103),pairs)
        n,t,e,truth=fixture();rows=[{'frame_idx':i,RKEY:r.tolist(),TKEY:p.tolist()} for i,(r,p) in truth.items()]
        v=np.array([[.01,.02,0],[.03,.02,.04],[-.03,.01,.01]])
        q=Rotation.from_rotvec([.6,.2,-.1]).as_matrix();b=np.array([.03,-.02,.1])
        new,newrows=rebind_canonical(v,rows,q,b)
        for old,row in zip(rows,newrows):np.testing.assert_allclose(v@np.array(old[RKEY]).T+old[TKEY],new@np.array(row[RKEY]).T+row[TKEY],atol=1e-12)

    def test_missing_metric_keeps_current_hypothesis_and_invalid_pose_fails(self):
        frames=[{'frame_idx':i,'time_s':t,'objects':[{'object_id':'cup','rigid_pose_observation_eligible':False}]} for i,t in [(7,0),(19,.3),(103,1)]]
        row={'frame_idx':19,RKEY:Rotation.from_rotvec([0,0,.9]).as_matrix().tolist(),TKEY:[0,0,1]}
        source={'pose_rows':[row]};original=copy.deepcopy(source)
        nodes,t=build_nodes({'frames':frames},source,'cup')
        np.testing.assert_array_equal(nodes[1].initial_rotation,row[RKEY]);self.assertFalse(nodes[1].metric_observation)
        self.assertEqual(source,original)
        source['pose_rows'][0][RKEY]=np.zeros((3,3)).tolist()
        with self.assertRaises(ValueError):build_nodes({'frames':frames},source,'cup')


"""Accepted numerical review counterexamples; CPU geometry, not video quality."""
import copy
from dataclasses import replace
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from utils.data import PoseNode, RGBEdge, RGBReprojectionFactor
from optimization.object.pose import GraphConfig, solve_scene_pose as solve, residual_blocks, reprojection_depths, depth_state


def node(i, z=0):
    return PoseNode(i,np.eye(3),np.array([0.,0.,z]),None,'test',False,1.)


def restoration_factor():
    return RGBReprojectionFactor(0,1,0,1,np.array([[0.,0.,0.],[.005,0,.005],[-.005,0,.005]]),
        np.array([[0,0],[.005/.305,0],[-.005/.305,0]]),np.array([1.,1.,0.,0.]),np.eye(3),np.zeros(3),np.ones(3),1.)


class RestorationTests(unittest.TestCase):
    def test_bounded_restoration_and_independent_rotation(self):
        nodes=[node(0),node(1,-.2),node(2,1.)];original=copy.deepcopy(nodes)
        rotation=Rot.from_euler('z',45,degrees=True).as_matrix()
        edges=[RGBEdge(0,1,np.eye(3),np.array([0.,0.,.3]),1.),RGBEdge(0,2,rotation,np.array([0.,0.,1.]),1.)]
        cfg=GraphConfig(outer_iterations=14,inner_evaluations=60,observed_weight=0,
            correction_velocity_rad_s=100,correction_velocity_m_s=100,acceleration_rad_s2=100,acceleration_m_s2=100)
        poses,report=solve(nodes,[0,1,2],edges,0,cfg,[restoration_factor()])
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
        nodes=[node(0),node(1,.02)];f=restoration_factor()
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
        nodes=[node(0),node(1,-.2)];poses={n.frame_idx:(n.initial_rotation,n.initial_translation) for n in nodes};f=restoration_factor()
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


class FormalProblemTests(unittest.TestCase):
    def problem(self, **kwargs):
        from optimization.problems import FormalManoInterval
        from pathlib import Path
        p=Path('fixture')
        return FormalManoInterval('case','object',p,p,p,p,[p],p,p,p,0,2,**kwargs)

    def test_hot3d_fixes_object_and_preserves_translation_support_gates(self):
        args=self.problem(profile='hot3d_shared',factor_reports=[Path('factor')]).arguments()
        self.assertFalse(args.optimize_object_translation)
        self.assertTrue(args.optimize_contact_state)
        self.assertTrue(args.require_active_full_K_mano_contract)
        self.assertTrue(args.visible_surface_depth_order_term)
        self.assertTrue(args.freeze_translation_without_visible_surface_support)
        self.assertTrue(args.gate_translation_with_visible_surface_support)

    def test_optional_named_factors_and_forbidden_boolean_combinations(self):
        self.assertTrue(self.problem(profile='contact_geometry').arguments().contact_state_geometry_likelihood)
        self.assertTrue(self.problem(profile='object_delta').arguments().optimize_object_translation)
        self.assertFalse(self.problem(profile='no_dense_barrier').arguments().dense_observed_surface_barrier)
        with self.assertRaises(ValueError):self.problem(profile='hand_ray_prior').arguments()
        with self.assertRaises(ValueError):self.problem(profile='unknown').arguments()
        with self.assertRaises(ValueError):self.problem(numeric_options={'optimize_object_translation':True}).arguments()

    def test_empty_formal_interval_does_not_invent_geometry(self):
        from optimization.hand.fit import optimize_rows
        result,states,archive=optimize_rows([],None,None,None,None)
        self.assertEqual(result['status'],'no_rows');self.assertEqual(states,[])
        self.assertEqual(archive['vertices_world_m'].shape,(0,778,3))

if __name__ == '__main__':
    unittest.main()
