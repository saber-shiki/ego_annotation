"""Actual-triangle low-texture and temporal mode-selection tests."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from preprocessing.geometry import TriangleMesh
from preprocessing.factors import ImageFactors
from preprocessing.shape_relocalization import cube_rotations,select_path,interpolate_corrections,geometry_relocalize,dense_acceptance


class ShapeRelocalizationTests(unittest.TestCase):
    def test_rotation_bank_contains_24_proper_modes(self):
        bank=cube_rotations();self.assertEqual(len(bank),24)
        self.assertEqual(len({tuple(R.ravel()) for R in bank}),24)
        for R in bank:
            np.testing.assert_allclose(R@R.T,np.eye(3));self.assertAlmostEqual(np.linalg.det(R),1.)
    def test_path_avoids_alternating_ambiguous_flips(self):
        I=np.eye(3);flip=Rotation.from_euler('y',180,degrees=True).as_matrix();states=[]
        for i in range(6):
            states.append([{'R':I,'t':np.array([0,0,.5]),'cost':0. if i%2==0 else .05},
                           {'R':flip,'t':np.array([0,0,.5]),'cost':.05 if i%2==0 else 0.}])
        path=select_path(states,np.arange(6)/30.)
        matrices=[s[j]['R'] for s,j in zip(states,path)]
        for R in matrices:np.testing.assert_allclose(R,matrices[0],atol=1e-12)
    def test_interpolation_preserves_timeline_and_endpoints(self):
        base=np.repeat(np.eye(3)[None],5,axis=0);trans=np.zeros((5,3));times=np.array([0.,.1,.3,.7,1.])
        end=Rotation.from_euler('z',90,degrees=True).as_matrix()
        R,t=interpolate_corrections(base,trans,times,[0,4],[{'R':np.eye(3),'t':np.zeros(3)},{'R':end,'t':np.array([.1,0,0])}])
        self.assertEqual(R.shape,(5,3,3));np.testing.assert_allclose(R[-1],end,atol=1e-12);np.testing.assert_allclose(t[-1],[.1,0,0])
        np.testing.assert_allclose(R@R.transpose(0,2,1),np.broadcast_to(np.eye(3),R.shape),atol=1e-12)
    def test_small_real_improvement_is_not_called_regression(self):
        ok,reason=dense_acceptance([.60,.70],[.61,.71],[.4,.3],[.39,.29],[.01,.02],[.01,.02])
        self.assertTrue(ok,reason)
    def test_image_gain_cannot_hide_surface_regression(self):
        ok,reason=dense_acceptance([.60,.70],[.80,.90],[.4,.3],[.2,.1],[.01,.02],[.08,.09])
        self.assertFalse(ok);self.assertEqual(reason,'rejected_source_surface_regression')
    def test_asymmetric_real_mesh_corrects_wrong_90_degree_silhouette(self):
        import trimesh
        a=trimesh.creation.box(extents=[.24,.045,.025]);b=trimesh.creation.box(extents=[.045,.16,.025]);b.apply_translation([-.0975,.0575,0])
        m=trimesh.util.concatenate([a,b]);mesh=TriangleMesh(m.vertices,m.faces)
        K=np.array([200.,200.,64.,64.]);true=Rotation.from_euler('z',90,degrees=True).as_matrix();trans=np.array([0.,0.,.6]);_,_,mask=mesh.raster(K,true,trans,(128,128))
        n=3;R=np.repeat(np.eye(3)[None],n,axis=0);t=np.tile(trans,(n,1));hd=np.full((n,128,128),np.inf)
        inp=SimpleNamespace(ids=np.arange(n),times=np.arange(n)/30.,anchor=0,shape_basis=np.eye(3),
            cameras=np.repeat(np.eye(4)[None],n,axis=0),K=np.tile(K,(n,1)),wh=(128,128),
            masks=np.repeat(mask[None],n,axis=0),grays=np.zeros((n,128,128),np.uint8),clouds=[np.empty((0,3))]*n,source_R=R.copy(),source_t=t.copy())
        rr,tt,_,_,report=geometry_relocalize(inp,mesh,hd,R,t,ImageFactors.empty(),[{'kind':'baseline'}]*n)
        self.assertTrue(report['applied'],report)
        self.assertGreater(report['dense_median_IoU_after'],.85)
        self.assertGreater(report['dense_median_IoU_after']-report['dense_median_IoU_before'],.3)
        np.testing.assert_array_equal(R,inp.source_R);np.testing.assert_array_equal(t,inp.source_t)

if __name__=='__main__':unittest.main()
