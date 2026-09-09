"""Mechanism tests for rgb_scene_pose_graph, not visual quality claims."""
import copy
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rgb_scene_pose_kernels import PoseNode,RGBEdge,RGBReprojectionFactor,PointFactor,rgb_edge_residual,rgb_reprojection_residual,SolverConfig
from rgb_scene_pose_graph import solve,GraphConfig,residual_blocks
from rgb_scene_pose_io import build_nodes,time_source_pairs,time_keyframes,RKEY,TKEY
from bind_rgb_scene_shape import rebind_canonical


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
        self.assertLess(min(diag['edges'][-1]['robust_rotation_precision']),.1)

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

if __name__=='__main__':unittest.main()
