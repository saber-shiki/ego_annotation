"""Regression tests for fair candidate validation and actual correspondence wiring."""
import json,unittest
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from preprocessing.geometry import TriangleMesh,project
from preprocessing.factors import ImageFactors
from preprocessing.relocalization import (reprojection_score,pnp_hypotheses,diverse_keys,
    mutual_matches,replace_frame_factors,relocalize)


def fixture(texture=True):
    size=240;rng=np.random.default_rng(49);im=np.zeros((size,size),np.uint8)
    if texture:
        im[35:205,35:205]=rng.integers(30,220,(170,170),dtype=np.uint8)
        for _ in range(70):cv2.circle(im,tuple(rng.integers(45,195,2)),int(rng.integers(2,7)),int(rng.integers(20,240)),-1)
    mask=np.zeros_like(im,bool);mask[40:201,40:201]=True
    R=np.repeat(np.eye(3)[None],3,axis=0);t=np.tile([0.,0.,1.],(3,1))
    inputs=SimpleNamespace(ids=np.array([10,20,30]),anchor=1,wh=(size,size),K=np.tile([200.,200.,120.,120.],(3,1)),
        cameras=np.repeat(np.eye(4)[None],3,axis=0),grays=np.repeat(im[None],3,axis=0),masks=np.repeat(mask[None],3,axis=0),clouds=[np.empty((0,3)) for _ in range(3)])
    mesh=TriangleMesh([[-.4,-.4,0],[.4,-.4,0],[.4,.4,0],[-.4,.4,0]],[[0,1,2],[0,2,3]])
    return inputs,mesh,np.full((3,120,120),np.inf),R,t


class RelocalizationTests(unittest.TestCase):
    def test_failed_geometry_experiment_is_opt_in(self):
        import inspect
        from preprocessing.__main__ import run
        params=inspect.signature(run).parameters
        self.assertFalse(params['enable_geometry'].default)
        self.assertTrue(params['enable_relocalization'].default)
    def test_full_evidence_defeats_tiny_perfect_inlier_subset(self):
        rng=np.random.default_rng(5);X=rng.uniform(-.06,.06,(30,3));k=np.array([450.,450.,240.,240.]);R=np.eye(3);true=np.array([0,0,.6]);bad=true+[.05,0,0]
        uv=project(X+true,k);uv[:4]=project(X[:4]+bad,k)
        a=reprojection_score(R,true,X,uv,k);b=reprojection_score(R,bad,X,uv,k)
        self.assertGreater(b['cost_px'],a['cost_px']*3)
        self.assertGreater(a['inlier_fraction'],b['inlier_fraction'])
    def test_negative_depth_cannot_get_good_score(self):
        q=np.zeros((20,3));uv=np.zeros((20,2));s=reprojection_score(np.eye(3),[0,0,-1],q,uv,[100,100,0,0])
        self.assertEqual(s['inlier_fraction'],0);self.assertEqual(s['cost_px'],12.)
    def test_large_rotation_with_outliers_and_no_input_mutation(self):
        rng=np.random.default_rng(11);X=rng.uniform(-.1,.1,(70,3));R=Rotation.from_euler('y',70,degrees=True).as_matrix();t=np.array([.01,.02,.7]);k=np.array([500.,500.,240.,240.]);uv=project(X@R.T+t,k)
        uv[:12]=rng.uniform(60,410,(12,2));r0=np.eye(3);t0=np.array([0.,0.,.65]);before=[x.copy() for x in [X,uv,r0,t0]]
        hs=pnp_hypotheses(X,uv,k,r0,t0)
        self.assertTrue(hs);self.assertLess(min(Rotation.from_matrix(h[0]@R.T).magnitude() for h in hs),.01)
        for actual,expected in zip([X,uv,r0,t0],before):np.testing.assert_array_equal(actual,expected)
    def test_anchor_and_long_baselines_not_evicted_by_recent_keys(self):
        keys=diverse_keys(range(150),0,148,maximum=7)
        self.assertIn(0,keys);self.assertIn(149,keys);self.assertNotIn(148,keys);self.assertEqual(len(keys),7)
        self.assertTrue(any(40<k<110 for k in keys))
    def test_reciprocal_descriptor_identity(self):
        rng=np.random.default_rng(13);a=rng.random((25,128)).astype('float32');order=rng.permutation(25);b=a[order].copy()
        pairs=mutual_matches(a,b)
        self.assertEqual(len(pairs),25)
        for i,j in pairs:self.assertEqual(i,order[j])
    def test_accepted_frames_remove_all_stale_bindings(self):
        old=ImageFactors(np.array([0,1,1,2]),np.ones((4,3)),np.zeros((4,2)),np.ones(4),'old')
        new=ImageFactors(np.array([1]),np.full((1,3),9.),np.zeros((1,2)),np.ones(1),'new')
        out=replace_frame_factors(old,{1},new)
        self.assertEqual(len(out.frames),3);np.testing.assert_array_equal(out.points[out.frames==1],[[9,9,9]])
        np.testing.assert_array_equal(old.points,np.ones((4,3)))
    def test_actual_descriptor_stage_wires_pairs_and_serializes(self):
        inp,mesh,hd,R,t=fixture();tracks=ImageFactors.empty();rows=[{'kind':'baseline'} for _ in range(3)]
        rr,tt,ff,output,report=relocalize(inp,mesh,hd,R,t,tracks,rows)
        self.assertGreater(report['backend_factor_rows'],0);self.assertGreater(len(ff.frames),0)
        json.dumps(report,allow_nan=False);json.dumps(output,allow_nan=False)
        np.testing.assert_allclose(rr,R);np.testing.assert_allclose(tt,t)
        self.assertEqual(report['accepted_frames'],[])
    def test_no_descriptors_preserves_full_baseline(self):
        inp,mesh,hd,R,t=fixture(False);tracks=ImageFactors(np.arange(3),np.zeros((3,3)),np.ones((3,2)),np.ones(3),'baseline')
        rr,tt,ff,rows,report=relocalize(inp,mesh,hd,R,t,tracks,[{'kind':'baseline'}]*3)
        np.testing.assert_array_equal(rr,R);np.testing.assert_array_equal(tt,t);np.testing.assert_array_equal(ff.frames,tracks.frames)
        self.assertEqual(len(report['frames']),3);self.assertEqual(report['accepted_frames'],[])

if __name__=='__main__':unittest.main()
