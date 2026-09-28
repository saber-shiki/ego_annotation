"""Category/frame independent regressions for observed pose-jump mechanisms."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from preprocessing.test_visible_surface import fixture
from preprocessing.geometry import TriangleMesh,camera_pose,project
from preprocessing.factors import ImageFactors
from preprocessing.tracking import small_image_fit,refinement_consistent
from preprocessing.visible_surface import VisibleSurfaceRegistration,cloud_digest


class PoseJumpTests(unittest.TestCase):
    def test_bad_contour_score_cannot_override_fixed_tracking_evidence(self):
        before={'rms_px':1.5,'p90_px':2.,'positive_fraction':1.}
        after={'rms_px':19.,'p90_px':39.,'positive_fraction':1.}
        self.assertFalse(refinement_consistent(before,after,.84,.47)[0])
        self.assertFalse(refinement_consistent(before,before,.84,.47)[0])

    def test_joint_refinement_keeps_large_real_image_supported_rotation(self):
        inp,R,t,hd,truth=fixture(scale=1.)
        real_R=Rotation.from_euler('y',65,degrees=True).as_matrix();real_t=truth[0]
        mesh=TriangleMesh(inp.vertices,inp.faces);X,z,mask=mesh.raster(inp.K[0],real_R,real_t,inp.wh)
        inp.masks[0]=mask;inp.owned_masks[0]=mask
        native=X[mask][::max(1,np.count_nonzero(mask)//80)]
        uv=project(native@real_R.T+real_t,inp.K[0])
        bad=ImageFactors(np.zeros(len(native),int),native,uv+np.array([25.,-20.]),np.ones(len(native))*.35,'bad_partial_contour')
        with patch('preprocessing.tracking.contour_factors',return_value=(bad,[])):
            got_R,got_t,report=small_image_fit(real_R,real_t,inp,mesh,hd,0,20,support_points=native,support_uv=uv)
        self.assertGreater(np.degrees(Rotation.from_matrix(got_R).magnitude()),60.)
        self.assertLessEqual(report['tracked_reprojection_after']['rms_px'],.750001)
        self.assertGreaterEqual(report['fixed_mask_IoU_after'],report['fixed_mask_IoU_before']-1e-6)
        self.assertEqual(report['joint_tracking_points'],len(native))

    def test_mask_improving_correction_is_not_blocked_by_stale_bindings(self):
        from preprocessing.tracking import guarded_contour_output
        inp,R,t,hd,_=fixture(scale=1.);mesh=TriangleMesh(inp.vertices,inp.faces)
        good={'rms_px':1.,'p90_px':2.,'positive_fraction':1.}
        bad={'rms_px':20.,'p90_px':30.,'positive_fraction':1.}
        after_R=Rotation.from_euler('y',50,degrees=True).as_matrix()
        with patch('preprocessing.tracking.projection_quality',side_effect=[good,bad]), patch('preprocessing.tracking.fresh_mask_iou',side_effect=[.3,.8]), patch('preprocessing.tracking.small_image_fit') as joint:
            selected,_,report=guarded_contour_output(R[0],t[0],after_R,t[0],inp,mesh,hd,0,np.zeros((8,3)),np.zeros((8,2)),8,{'available':True})
        joint.assert_not_called();np.testing.assert_array_equal(selected,after_R)
        self.assertFalse(report['contour_candidate_rejected'])

    def test_output_repair_does_not_change_next_proposal_or_rebinding_state(self):
        from types import SimpleNamespace
        from preprocessing.tracking import bootstrap
        inp,_,_,hd,_=fixture(scale=1.,n=3);inp.anchor=0;inp.rgb=SimpleNamespace(edges=[])
        uv=np.zeros((8,2));X=np.zeros((8,3));turn=Rotation.from_euler('z',20,degrees=True).as_matrix()
        def pnp(X,uv,K,R,t):return R,t,np.arange(8),0.
        def contour(R,t,*args):return turn@R,t,{'available':True}
        def guard(R,t,after_R,after_t,*args):return R,t,{'available':True}
        with patch('preprocessing.tracking.seed_features',return_value=(uv,X)), patch('preprocessing.tracking.flow_track',return_value=(uv,np.ones(8,bool))), patch('preprocessing.tracking.solve_pnp',side_effect=pnp), patch('preprocessing.tracking.contour_pose_proposal',side_effect=contour), patch('preprocessing.tracking.guarded_contour_output',side_effect=guard):
            R,t,_,rows=bootstrap(inp,hd,max_features=8,refine_anchor=False,guarded_output=True)
        np.testing.assert_allclose(R[1],np.eye(3),atol=1e-12)
        np.testing.assert_allclose(R[2],turn,atol=1e-12)
        np.testing.assert_allclose(rows[2]['unfiltered_contour_pose']['rotation_world'],turn@turn,atol=1e-12)

    def make_gap(self,missing,n=5):
        inp,R,t,hd,truth=fixture(scale=1.,n=n)
        for j in missing:inp.clouds[j]=np.empty((0,3));inp.cloud_valid[j]=False
        return inp,R,t,hd,truth

    def test_gap_preserves_nonlinear_RGB_motion_and_metric_bias(self):
        inp,R,reference,hd,_=self.make_gap([1,3])
        reference[1,2]-=.04
        warm=reference.copy();warm[[0,2,4],2]+=.05
        fit=VisibleSurfaceRegistration(inp,R,warm,hd,range(5),fit_scale=False,temporal=True,motion_reference_world=reference)
        _,t=fit.unpack(fit.pack(1.,warm))
        np.testing.assert_allclose(t[:,2],reference[:,2]+.05,atol=1e-12)
        self.assertEqual(fit.size,13)
        self.assertNotAlmostEqual(t[1,2],(t[0,2]+t[2,2])*.5)
        self.assertFalse(inp.cloud_valid[1]);self.assertEqual(len(inp.clouds[1]),0)

    def test_leading_and_trailing_gaps_do_not_copy_pose(self):
        inp,R,reference,hd,_=self.make_gap([0,1,3,4])
        warm=reference.copy();warm[2,2]+=.05
        fit=VisibleSurfaceRegistration(inp,R,warm,hd,range(5),fit_scale=False,temporal=True,motion_reference_world=reference)
        _,t=fit.unpack(fit.pack(1.,warm))
        np.testing.assert_allclose(t[:,2],reference[:,2]+.05)
        self.assertGreater(abs(t[4,2]-t[3,2]),.02)
        self.assertEqual(len(fit.depth_bridges),4)

    def test_world_correction_bridge_compensates_camera_rotation(self):
        inp,R,reference,hd,_=self.make_gap([1,3])
        inp.cameras[:,:3,:3]=Rotation.from_euler('y',np.array([0,3,-4,5,-2])[:,None],degrees=True).as_matrix()
        delta=np.array([.01,-.005,.05]);warm=reference.copy();warm[[0,2,4]]+=delta
        fit=VisibleSurfaceRegistration(inp,R,warm,hd,range(5),fit_scale=False,temporal=True,motion_reference_world=reference)
        _,t=fit.unpack(fit.pack(1.,fit.t0))
        for j in [1,3]:
            expected=camera_pose(R[j],reference[j]+delta,inp.cameras[j])[1][2]
            self.assertAlmostEqual(t[j,2],expected,places=12)

    def test_sparse_jacobian_includes_cross_frame_latent_depth(self):
        inp,R,t,hd,_=self.make_gap([1,3])
        fit=VisibleSurfaceRegistration(inp,R,t,hd,range(5),fit_scale=False,temporal=True,max_points=20)
        x=fit.pack(1.,fit.t0);s,tc=fit.unpack(x);depth,contour=fit.correspondences(s,tc)
        groups=fit.residual_groups(x,depth,contour);pattern=fit.sparsity(groups).toarray().astype(bool)
        def f(xx):return np.concatenate([v for _,v in fit.residual_groups(xx,depth,contour)])
        base=f(x)
        for k in range(len(x)):
            trial=x.copy();trial[k]+=1e-7;derivative=(f(trial)-base)/1e-7
            self.assertFalse(np.any((np.abs(derivative)>1e-4)&~pattern[:,k]),msg=f'column {k}')

    def test_constant_fast_motion_not_penalized_as_acceleration(self):
        inp,R,t,hd,_=self.make_gap([])
        inp.times=np.array([0.,.02,.055,.1,.14]);t[:,0]=.1+2.*inp.times;t[:,1]=0.;t[:,2]=.8
        fit=VisibleSurfaceRegistration(inp,R,t,hd,range(5),fit_scale=False,temporal=True)
        groups=dict(fit.motion_groups(t))
        np.testing.assert_allclose(groups['motion_acceleration'],0.,atol=1e-10)

    def test_motion_prior_does_not_saturate_as_robust_outlier(self):
        inp,R,t,hd,_=self.make_gap([])
        fit=VisibleSurfaceRegistration(inp,R,t,hd,range(5),fit_scale=False,temporal=True)
        rho=fit.loss_for_groups([('surface:0',np.zeros(1)),('motion_acceleration',np.zeros(1))])(np.array([100.,100.]))
        self.assertLess(rho[0,0],20.);self.assertEqual(rho[0,1],100.);self.assertEqual(rho[1,1],1.)

    def test_solver_cannot_reset_missing_depth_to_conflicting_silhouette(self):
        inp,R,t,hd,truth=self.make_gap([2])
        t=truth.copy();t[:,2]-=.05;mesh=TriangleMesh(inp.vertices,inp.faces)
        for j in range(5):
            _,_,mask=mesh.raster(inp.K[j],R[j],t[j],inp.wh);inp.masks[j]=mask;inp.owned_masks[j]=mask
        digest=cloud_digest(inp.clouds)
        fit=VisibleSurfaceRegistration(inp,R,t,hd,range(5),fit_scale=False,temporal=True,max_points=160)
        s,tc,report=fit.solve(outer_iterations=6,max_nfev=30)
        self.assertEqual(s,1.);self.assertAlmostEqual(tc[2,2]-t[2,2],((tc[1,2]-t[1,2])+(tc[3,2]-t[3,2]))/2.,places=12)
        self.assertAlmostEqual(tc[2,2],truth[2,2],delta=.012)
        self.assertEqual(cloud_digest(inp.clouds),digest)
        self.assertEqual(report['missing_metric_depth_inference'][0]['frame_idx'],2)
        self.assertEqual(report['after']['frames'][2]['metric_samples'],0)

if __name__=='__main__':unittest.main()
