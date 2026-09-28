"""Actual triangle tests for fixed-depth scale/translation and partial visibility."""
import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
from preprocessing.geometry import TriangleMesh
from preprocessing.visible_surface import VisibleSurfaceRegistration,cloud_digest,select_scale_keyframes


def fixture(scale=1.4,n=1,occluded=False,initial_z=.75,target_z=.75):
    m=trimesh.creation.box(extents=[.16,.10,.08]);v=np.asarray(m.vertices,float);f=np.asarray(m.faces,int)
    mesh=TriangleMesh(v*scale,f);wh=(192,192);K=np.array([300.,300.,95.5,95.5]);R=np.repeat(np.eye(3)[None],n,axis=0)
    truth=np.asarray([[-.015+.025*i,.012-.015*i,target_z+.03*i] for i in range(n)])
    initial=np.asarray([[.025+.025*i,-.012-.015*i,initial_z+.03*i] for i in range(n)])
    masks=[];clouds=[];hd=[]
    for i in range(n):
        X,z,mask=mesh.raster(K,np.eye(3),truth[i],wh);hand=np.full(wh[::-1],np.inf)
        if occluded:
            hand[:,96:]=.3;mask[:,96:]=False
        q=X[mask]+truth[i];q.setflags(write=False);clouds.append(q);masks.append(mask);hd.append(hand)
    inputs=SimpleNamespace(ids=np.arange(n),times=np.arange(n)/30.,anchor=n-1,vertices=v,faces=f,
        cameras=np.repeat(np.eye(4)[None],n,axis=0),K=np.tile(K,(n,1)),wh=wh,masks=np.asarray(masks),owned_masks=np.asarray(masks),
        clouds=clouds,cloud_valid=np.ones(n,bool),source_R=R,source_t=initial)
    return inputs,R,initial,np.asarray(hd),truth


class VisibleSurfaceTests(unittest.TestCase):
    def test_independent_object_scale_and_translation(self):
        inp,R,t,hd,truth=fixture();before=cloud_digest(inp.clouds)
        fit=VisibleSurfaceRegistration(inp,R,t,hd,[0],fit_scale=True,max_points=768)
        s,tc,report=fit.solve(outer_iterations=7,max_nfev=45)
        self.assertAlmostEqual(s,1.4,delta=.035)
        np.testing.assert_allclose(tc,truth,atol=.006)
        self.assertLess(report['after']['merit_m'],report['before']['merit_m']*.3)
        self.assertEqual(cloud_digest(inp.clouds),before)
        self.assertGreater(np.linalg.det(np.eye(3)*s),2.)
    def test_partial_surface_does_not_shrink_whole_object(self):
        inp,R,t,hd,truth=fixture(scale=1.,occluded=True)
        fit=VisibleSurfaceRegistration(inp,R,truth.copy(),hd,[0],fit_scale=True,max_points=768)
        s,tc,report=fit.solve(outer_iterations=5,max_nfev=30)
        self.assertGreater(s,.94);self.assertLess(s,1.06)
        self.assertFalse(report['full_object_dimensions_directly_observed'])
        self.assertFalse(report['scale_uniqueness_certified'])
    def test_hidden_backface_is_not_zero_depth_error(self):
        inp,R,t,hd,truth=fixture(scale=1.,initial_z=.75,target_z=.83)
        fit=VisibleSurfaceRegistration(inp,R,t,hd,[0],fit_scale=False,max_points=768)
        before=fit.measure(1.,fit.t0)
        self.assertGreater(before['frames'][0]['first_hit_abs_depth_m']['median'],.06)
        s,tc,report=fit.solve(outer_iterations=6,max_nfev=40)
        self.assertEqual(s,1.)
        self.assertLess(report['after']['frames'][0]['first_hit_abs_depth_m']['median'],.004)
        self.assertAlmostEqual(tc[0,2],truth[0,2],delta=.004)
    def test_multiple_frames_share_one_scale(self):
        inp,R,t,hd,truth=fixture(scale=1.25,n=3)
        fit=VisibleSurfaceRegistration(inp,R,t,hd,[0,1,2],fit_scale=True,max_points=512)
        s,tc,report=fit.solve(outer_iterations=12,max_nfev=40)
        self.assertIsInstance(s,float);self.assertAlmostEqual(s,1.25,delta=.04)
        self.assertEqual(tc.shape,(3,3));np.testing.assert_allclose(tc,truth,atol=.008)
        self.assertEqual(len(report['after']['frames']),3)
    def test_single_keyframe_keeps_appearance_anchor(self):
        inp,_,_,_,_=fixture(n=3)
        self.assertEqual(select_scale_keyframes(inp,[{}]*3,maximum=1),[2])
    def test_fixed_scale_has_no_depth_bias_or_scale_variables(self):
        inp,R,t,hd,_=fixture(n=3)
        fit=VisibleSurfaceRegistration(inp,R,t,hd,[0,1,2],fit_scale=False,initial_scale=1.2,temporal=True)
        self.assertEqual(fit.size,9)
        x=fit.pack(1.2,fit.t0);s,_=fit.unpack(x);self.assertEqual(s,1.2)
    def test_visible_surface_profile_routes_without_legacy_depth_correction(self):
        from unittest.mock import patch
        from preprocessing.__main__ import run
        with patch('preprocessing.surface_pipeline.run_visible_surface',return_value={'depth_fixed':True}) as runner:
            result=run('/a/source_observations.json','/b/new',profile='visible-surface')
        self.assertTrue(result['depth_fixed']);runner.assert_called_once()
    def test_invalid_qualified_surface_is_rejected(self):
        inp,R,t,hd,_=fixture();inp.clouds=[np.array([[0.,0.,-1.]])]
        with self.assertRaises(ValueError):VisibleSurfaceRegistration(inp,R,t,hd,[0])

if __name__=='__main__':unittest.main()
