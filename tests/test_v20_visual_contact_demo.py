import unittest,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v20_demo_geometry import camera,world,project,resized_K,torch_rodrigues,make_scene,closest

class DemoGeometry(unittest.TestCase):
    def test_camera_roundtrip_and_halfpixel(self):
        T=np.eye(4);T[:3,:3]=torch_rodrigues(torch.tensor([.2,-.3,.1],dtype=torch.float64)).numpy();T[:3,3]=[.1,.2,-.1]
        p=np.array([[.1,.2,.5],[-.2,.1,1.]])
        np.testing.assert_allclose(camera(world(p,T),T),p,atol=1e-12)
        K=np.array([900.,910.,702.,705.]);k=resized_K(K)
        np.testing.assert_allclose(project(p,k),(project(p,K)+.5)*(960/1408)-.5)
    def test_camera_origin_demo_scale_preserves_pixels(self):
        p=np.array([[.1,.2,.5],[-.2,.1,1.]])
        K=np.array([900.,910.,702.,705.]);np.testing.assert_allclose(project(p*1.13,K),project(p,K))
        self.assertGreater(np.linalg.norm(1.13*p-p),.01)
    def test_zero_rotation_has_nonzero_gradient(self):
        r=torch.zeros(3,requires_grad=True);R=torch_rodrigues(r);loss=(R@torch.tensor([0.,0.,1.]))[0];loss.backward()
        self.assertGreater(abs(r.grad[1].item()),.9)
        np.testing.assert_allclose(R.detach().numpy(),np.eye(3))
    def test_nonzero_torch_numpy_export_transform_parity(self):
        from scipy.spatial.transform import Rotation
        rng=np.random.default_rng(2);v=rng.normal(size=(2,8,3))*.05+[.1,.15,.4];pivot=v[:,:1].copy()
        R=Rotation.from_rotvec([[.05,-.07,.02],[-.02,.03,.01]]).as_matrix();C=Rotation.from_rotvec([[.1,.2,.05],[-.1,.15,0]]).as_matrix();ct=np.array([[.01,.02,.03],[.02,-.02,.01]])
        scale=np.array([.96,1.03])[:,None,None];shift=np.array([[.001,-.002,.003],[-.002,.001,.002]])[:,None]
        rotated=(v-pivot)@R.transpose(0,2,1)+pivot;cam=(rotated-ct[:,None])@C;expected=(cam*scale+shift)@C.transpose(0,2,1)+ct[:,None]
        t=lambda a:torch.as_tensor(a,dtype=torch.float64)
        rotated_t=(t(v)-t(pivot))@t(R).transpose(1,2)+t(pivot)
        camera_t=(rotated_t-t(ct)[:,None])@t(C)
        actual=(camera_t*t(scale)+t(shift))@t(C).transpose(1,2)+t(ct)[:,None]
        np.testing.assert_allclose(actual.numpy(),expected,atol=1e-12)

    def test_triangle_contact_and_joint_geometry_occlusion(self):
        import open3d as o3d
        vertices=np.array([[-1,-1,1],[1,-1,1],[0,1,1]],float);faces=np.array([[0,1,2]])
        sc=make_scene(vertices,faces);q,n=closest(sc,np.array([[0,0,1.03]]));self.assertAlmostEqual(np.linalg.norm(q-[0,0,1.03]),.03,places=6)
        sc.add_triangles(o3d.core.Tensor((vertices*.5).astype('float32')),o3d.core.Tensor(faces.astype('uint32')))
        hit=sc.cast_rays(o3d.core.Tensor(np.array([[0,0,0,0,0,1]],np.float32)))
        self.assertEqual(int(hit['geometry_ids'].numpy()[0]),1)
        self.assertAlmostEqual(float(hit['t_hit'].numpy()[0]),.5)
if __name__=='__main__':unittest.main()
