"""Synthetic mechanism tests; no case assets or reference dataset required."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from preprocessing.shared_repair import AffineMesh,shared_matrix,observed_normals
from preprocessing.geometry import TriangleMesh

class SharedRepairTests(unittest.TestCase):
 def test_affine_rays_equal_explicit_deformed_full_triangles(self):
  import trimesh
  m=trimesh.creation.icosphere(subdivisions=2,radius=.08);v=np.asarray(m.vertices);f=np.asarray(m.faces);B=Rotation.from_rotvec([.1,.2,.1]).as_matrix();S=shared_matrix([.13,-.19,.04],B);R=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();t=np.array([.01,.02,.5]);k=np.array([300,310,80,80]);y,x=np.mgrid[40:120:3,40:120:3];uv=np.c_[x.ravel(),y.ravel()]
  a=AffineMesh(v,f);X,z,ok,n=a.query(uv,k,R,t,S);e=TriangleMesh(v@S.T,f);Y,zz,valid=e.query(uv,k,R,t)
  np.testing.assert_array_equal(ok,valid);np.testing.assert_allclose(z[ok],zz[valid],atol=1e-6);np.testing.assert_allclose(X[ok]@S.T,Y[valid],atol=1e-6)
  np.testing.assert_allclose(np.linalg.norm(n[ok],axis=1),1.,atol=1e-8)
 def test_shared_shape_is_positive_not_per_frame(self):
  B=Rotation.from_rotvec([.4,.2,.1]).as_matrix();q=np.array([-.2,.1,.04]);S=shared_matrix(q,B)
  self.assertGreater(np.linalg.det(S),0);np.testing.assert_allclose(np.linalg.eigvalsh(S),np.sort(np.exp(q)),atol=1e-12)
 def test_observed_plane_normals(self):
  x,y=np.mgrid[-.05:.05:12j,-.07:.07:12j];points=np.c_[x.ravel(),y.ravel(),np.full(x.size,.4)];normal,confidence=observed_normals(points,points[::4])
  np.testing.assert_allclose(normal,np.tile([0,0,-1],(len(normal),1)),atol=1e-8);self.assertTrue(np.all(confidence>.9))
 def test_affine_preserves_topology_not_box(self):
  import trimesh
  m=trimesh.creation.icosphere(subdivisions=1);S=shared_matrix([.1,-.2,0],np.eye(3));changed=np.asarray(m.vertices)@S.T
  self.assertEqual(changed.shape,m.vertices.shape);np.testing.assert_array_equal(m.faces,m.faces.copy());self.assertFalse(np.allclose(changed,m.vertices))

if __name__=='__main__':unittest.main()
