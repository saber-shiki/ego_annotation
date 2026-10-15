"""Full mesh nonpenetration mathematical tests, synthetic triangles only."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from utils.complete_triangle_collision import CompleteTriangleScene,triangles_overlap,intersection_barycentric
from solvers.direct_nonpenetration import linear_feasible_step


class CompleteTrianglesTest(unittest.TestCase):
    def test_coplanar_containment_touch_and_separation(self):
        a=np.array([[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]])
        inside=np.array([[[.2,.2,0.],[.4,.2,0.],[.2,.4,0.]]])
        self.assertTrue(triangles_overlap(a,inside)[0])
        self.assertTrue(triangles_overlap(a,inside[:,::-1])[0])
        above=inside.copy();above[:,:,2]=1e-4
        self.assertFalse(triangles_overlap(a,above)[0])
        separate=inside+np.array([3.,0.,0.]);self.assertFalse(triangles_overlap(a,separate)[0])
        touch=np.array([[[2.,0.,0.],[3.,0.,0.],[2.,1.,0.]]]);self.assertTrue(triangles_overlap(a,touch)[0])

    def test_non_coplanar_reverse_direction_intersection_missed_by_hand_edges(self):
        import trimesh
        cube=trimesh.creation.box(extents=[.4,.4,.4]);scene=CompleteTriangleScene(cube.vertices,cube.faces)
        # The object's edges pierce the interior of this large triangle; ALL hand
        # vertices and its centroid are outside, and ALL hand edges miss cube.
        v=np.array([[-3.,-3.,0.],[3.,-3.,0.],[0.,3.,0.]]);f=np.array([[0,1,2]])
        old=scene.query(v,f,np.eye(3),np.zeros(3),edges=True,nsamples=11)
        self.assertEqual(old['edge_crossings'],0);self.assertLessEqual(old['depth_m'].max(),0.)
        new=scene.intersections(v,f,np.eye(3),np.zeros(3));self.assertGreater(len(new['hand_face_ids']),0)
        R=Rotation.from_rotvec([.2,.5,-.3]).as_matrix();t=np.array([.7,-.8,.5])
        moved=scene.intersections(v@R.T+t,f,R,t)
        np.testing.assert_array_equal(np.sort(new['object_face_ids']),np.sort(moved['object_face_ids']))

    def test_conservative_broad_phase_matches_all_triangle_pairs(self):
        import trimesh
        cube=trimesh.creation.icosphere(subdivisions=2,radius=.01);scene=CompleteTriangleScene(cube.vertices,cube.faces)
        rng=np.random.default_rng(123);tri=rng.normal(size=(63,3,3))*.009;vertices=tri.reshape(-1,3);faces=np.arange(len(vertices)).reshape(-1,3)
        found=scene.intersections(vertices,faces,np.eye(3),np.zeros(3))
        explicit=set()
        for i,a in enumerate(tri):
            keep=triangles_overlap(np.repeat(a[None],len(scene.triangles),axis=0),scene.triangles)
            explicit.update((i,int(j)) for j in np.flatnonzero(keep))
        actual=set(zip(found['hand_face_ids'].tolist(),found['object_face_ids'].tolist()))
        self.assertSetEqual(actual,explicit)

    def test_intersection_witness_is_actual_hand_point_not_far_plane_vertex(self):
        a=np.array([[[-3.,-3.,0.],[3.,-3.,0.],[0.,3.,0.]]])
        b=np.array([[[.1,-.1,-.2],[.1,.1,-.2],[.1,0.,.2]]])
        bar,fallback=intersection_barycentric(a,b)
        p=np.einsum('nv,nvc->nc',bar,a)
        self.assertFalse(fallback[0]);self.assertTrue((bar>=0).all());self.assertAlmostEqual(bar.sum(),1.)
        self.assertLess(np.linalg.norm(p[0]-np.array([.1,0.,0.])),1e-7)

    def test_directed_exit_chooses_same_actual_wall_for_opposite_inside_points(self):
        import trimesh
        cube=trimesh.creation.box(extents=[.4,.4,.4]);scene=CompleteTriangleScene(cube.vertices,cube.faces)
        p=np.array([[-.19,0.,0.],[.19,0.,0.],[-.2,.02,0.]])
        exit=scene.directed_exit(p,np.array([1.,0.,0.]),np.eye(3),np.zeros(3))
        np.testing.assert_allclose(exit['target_world'][:,0],.2,atol=2e-7)
        np.testing.assert_allclose(exit['normals_world'],np.tile([1.,0.,0.],(3,1)),atol=2e-7)

    def test_minimax_feasibility_and_minimum_quadratic_step(self):
        A=np.array([[1.,0.],[0.,1.]]);b=np.array([.2,.1]);H=np.eye(2);g=np.zeros(2)
        x,r=linear_feasible_step(A,b,H,g,np.array([-1.,-1.]),np.array([1.,1.]))
        self.assertTrue(r['lp_success']);self.assertAlmostEqual(r['slack_mm'],0.)
        np.testing.assert_allclose(x,b,atol=3e-6)
        A=np.array([[1.],[-1.]]);b=np.array([1.,1.])
        x,r=linear_feasible_step(A,b,np.eye(1),np.zeros(1),np.array([-2.]),np.array([2.]))
        self.assertAlmostEqual(r['slack_mm'],1.,places=6);self.assertAlmostEqual(x[0],0.,places=6)


if __name__=='__main__':unittest.main()
