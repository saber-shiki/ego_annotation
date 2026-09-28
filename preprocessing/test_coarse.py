"""Discriminating CPU tests; geometry/optimization, not only file/schema checks."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from preprocessing.geometry import (resized_intrinsics,project,matrix_k,TriangleMesh,shape_matrix,camera_pose,world_pose)
from preprocessing.factors import ImageFactors
from preprocessing.optimizer import SceneProblem
from preprocessing.inputs import reject_evaluation_path,InputReceipt
from preprocessing.tracking import solve_pnp


def synthetic(n=4,points=60):
    rng=np.random.default_rng(83);x=rng.uniform(-.07,.07,(points,3))
    C=np.repeat(np.eye(4)[None],n,axis=0);K=np.tile([450.,450.,239.5,239.5],(n,1))
    inp=SimpleNamespace(ids=np.arange(n),times=np.arange(n)/30.,cameras=C,K=K,shape_basis=np.eye(3),anchor=0,rgb=SimpleNamespace(edges=[]))
    return inp,x


class CoarseTests(unittest.TestCase):
    def test_half_pixel_intrinsics(self):
        k=np.array([800,700,310,240.]);a=resized_intrinsics(k,[640,480],[320,240])
        np.testing.assert_allclose(a,[400,350,154.75,119.75])
    def test_camera_world_roundtrip(self):
        C=np.eye(4);C[:3,:3]=Rotation.from_rotvec([.3,.1,-.2]).as_matrix();C[:3,3]=[.3,-.2,.1]
        R=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();t=np.array([.1,.2,.7]);r,tt=camera_pose(R,t,C);r2,t2=world_pose(r,tt,C)
        np.testing.assert_allclose(r2,R,atol=1e-12);np.testing.assert_allclose(t2,t,atol=1e-12)
    def test_native_triangle_ray_is_camera_z(self):
        m=TriangleMesh([[-1,-1,1],[1,-1,1],[0,1,1]],[[0,1,2]])
        p,z,ok=m.query([[50,50],[60,50]],[100,100,50,50],np.eye(3),np.zeros(3))
        self.assertTrue(ok.all());np.testing.assert_allclose(z,[1,1],atol=1e-6);np.testing.assert_allclose(p[:,2],1,atol=1e-6)
    def test_uniform_scale_is_projection_gauge(self):
        p=np.array([[.1,.2,1],[.05,-.1,.4]]);k=np.array([500,500,240,240])
        np.testing.assert_allclose(project(p,k),project(p*1.5,k))
    def test_shared_shape_preserves_volume_and_is_not_per_frame(self):
        S=shape_matrix([.08,-.03],Rotation.from_rotvec([.2,.1,.3]).as_matrix())
        self.assertAlmostEqual(np.linalg.det(S),1.,places=12)
        self.assertGreater(np.linalg.norm(S-np.eye(3)),.05)
    def test_pixel_factors_recover_translation_not_just_smooth(self):
        inp,x=synthetic();n=len(inp.ids);target=np.array([.025,-.018,.55]);R=np.repeat(np.eye(3)[None],n,axis=0)
        f=np.repeat(np.arange(n),len(x));pts=np.tile(x,(n,1));uv=project(pts+target,inp.K[f])
        factors=ImageFactors(f,pts,uv,np.ones(len(f)),'synthetic')
        original=np.tile([0.,0.,.55],(n,1));problem=SceneProblem(inp,R,original,factors,(np.zeros(0,int),np.zeros((0,3)),np.zeros(0)))
        (_,t,_,_),diag=problem.solve(50)
        self.assertLess(np.linalg.norm(t-target,axis=1).max(),.001)
        self.assertLess(diag['residuals']['image']['rms_after'],diag['residuals']['image']['rms_before']/20)
        np.testing.assert_array_equal(original,np.tile([0.,0.,.55],(n,1)))
    def test_model_pnp_recovers_real_large_rotation(self):
        inp,x=synthetic();R=Rotation.from_euler('y',65,degrees=True).as_matrix();t=np.array([.01,-.02,.5]);uv=project(x@R.T+t,inp.K[0])
        result=solve_pnp(x,uv,inp.K[0],np.eye(3),np.array([0,0,.5]))
        self.assertIsNotNone(result);r,tt,ii,error=result
        self.assertLess(Rotation.from_matrix(r@R.T).magnitude(),1e-4);np.testing.assert_allclose(tt,t,atol=1e-5)
    def test_joint_solver_uses_rotation_observations(self):
        inp,x=synthetic(n=3);n=3;true=Rotation.from_euler('z',25,degrees=True).as_matrix()
        Rs=np.repeat(np.eye(3)[None],n,axis=0);t=np.tile([0,0,.6],(n,1));f=np.repeat(np.arange(n),len(x));pts=np.tile(x,(n,1));uv=project(pts@true.T+t[f],inp.K[f])
        problem=SceneProblem(inp,Rs,t,ImageFactors(f,pts,uv,np.ones(len(f)),'test'),(np.zeros(0,int),np.zeros((0,3)),np.zeros(0)))
        (R,_,_,_),_=problem.solve(45)
        self.assertLess(np.degrees(Rotation.from_matrix(R[1]@true.T).magnitude()),.2)
    def test_sparse_pattern_covers_numeric_derivatives(self):
        inp,x=synthetic(n=3,points=8);f=np.repeat(np.arange(3),8);pts=np.tile(x,(3,1));R=np.repeat(np.eye(3)[None],3,axis=0);t=np.tile([0,0,.6],(3,1));uv=project(pts+t[f],inp.K[f]);depth=(np.arange(3),x[:3],np.ones(3)*.6)
        p=SceneProblem(inp,R,t,ImageFactors(f,pts,uv,np.ones(len(f)),'test'),depth,fit_bias=True,fit_shape=True)
        q=np.zeros(p.size);q[0]=.02;mask=p.sparsity().toarray()!=0;v=p.residual(q)
        for j in range(p.size):
            test=q.copy();test[j]+=1e-6;der=np.abs(p.residual(test)-v)/1e-6
            self.assertFalse(np.any((der>1e-4)&~mask[:,j]),f'missing derivative column {j}')
    def test_relative_depth_scale_from_rigid_chords(self):
        from preprocessing.metric_alignment import estimate_relative_depth
        inp,x=synthetic(n=3,points=30);base=x+[0,0,.6];scales=[1.,1.15,1.3]
        inp.wh=(480,480);inp.clouds=[base*s for s in scales];inp.cloud_valid=np.ones(3,bool)
        edges=[]
        for a,b in [(0,1),(1,2),(0,2)]:
            edges.append(SimpleNamespace(source_frame_idx=a,target_frame_idx=b,source_world=inp.clouds[a],target_uv=project(inp.clouds[b],inp.K[b]),target_size_wh=np.array(inp.wh),diagnostics={'inlier_fraction':1.}))
        inp.rgb=SimpleNamespace(edges=edges)
        result=estimate_relative_depth(inp)
        self.assertGreater(len(result['report']['edges']),0)
        np.testing.assert_allclose(result['scale'],1/np.array(scales),atol=.015)
        np.testing.assert_allclose(inp.clouds[2],base*1.3)
    def test_surface_factors_reanchor_missed_ray_depth(self):
        inp,x=synthetic(n=3,points=20);R=np.repeat(np.eye(3)[None],3,axis=0);initial=np.tile([0,0,.9],(3,1));target=np.array([0,0,.5])
        frames=np.repeat(np.arange(3),len(x));points=np.tile(x,(3,1))
        surface=(frames,points,points+target)
        p=SceneProblem(inp,R,initial,ImageFactors.empty(),(np.zeros(0,int),np.zeros((0,3)),np.zeros(0)),use_depth=True,surface=surface)
        (_,t,_,_),_=p.solve(50)
        self.assertLess(np.linalg.norm(t-target,axis=1).max(),.002)
    def test_triangle_closest_stays_active_outside_rays(self):
        m=TriangleMesh([[-1,-1,1],[1,-1,1],[0,1,1]],[[0,1,2]])
        points,d=m.closest([[0,0,3]])
        np.testing.assert_allclose(points,[[0,0,1]],atol=1e-6);self.assertAlmostEqual(d[0],2.,places=6)
    def test_multi_hypothesis_pnp_contains_large_rotation_solution(self):
        from preprocessing.tracking import solve_pnp_hypotheses
        inp,x=synthetic();R=Rotation.from_euler('xyz',[65,-20,25],degrees=True).as_matrix();t=np.array([.02,-.01,.55]);uv=project(x@R.T+t,inp.K[0])
        hs=solve_pnp_hypotheses(x,uv,inp.K[0],np.eye(3),np.array([0,0,.5]),max_candidates=8)
        self.assertTrue(hs)
        self.assertLess(min(Rotation.from_matrix(h[0]@R.T).magnitude() for h in hs),1e-3)
    def test_forbid_evaluation_inputs(self):
        for path in ['/x/evaluation/a.json','/x/hot3d_gt/a.json','/x/depth_gt/a.png','/x/hot3d_s3_pose_audit_20260927/a.json']:
            with self.assertRaises(ValueError):reject_evaluation_path(path)
        reject_evaluation_path('/x/runs/hot3d_clip/source_observations.json')
    def test_hash_receipt_detects_changed_input(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.json';p.write_text('{}');r=InputReceipt();r.add(p);r.check();p.write_text('{"changed":true}')
            with self.assertRaises(RuntimeError):r.check()
    def test_bias_gauge_is_explicit(self):
        inp,x=synthetic(n=3,points=3);R=np.repeat(np.eye(3)[None],3,axis=0);t=np.tile([0,0,.6],(3,1))
        p=SceneProblem(inp,R,t,ImageFactors.empty(),(np.arange(3),x,np.ones(3)*.6),fit_bias=True)
        z=np.zeros(p.size);z[6]=.02
        self.assertAlmostEqual(p.groups(z)['reference_bias_gauge'][0],200.)

if __name__=='__main__':unittest.main()
