import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate import pose,apply,gauge_errors,angle,sample,scene,distance

class GeometryTests(unittest.TestCase):
    def test_motion_gauges_do_not_create_errors(self):
        H=pose(Rotation.from_rotvec([.2,.3,.1]).as_matrix(),[.5,-.2,.7])
        B=pose(Rotation.from_rotvec([-.3,.1,.2]).as_matrix(),[-.1,.2,.05])
        G={i:pose(Rotation.from_rotvec([.1*i,.2*i,0]).as_matrix(),[.01*i,.2,.5]) for i in range(5)}
        D={i:pose(Rotation.from_rotvec([0,.04*i,.01*i]).as_matrix(),[.02*i,0,0]) for i in range(5)}
        C={i:np.linalg.inv(H)@D[i] for i in G};P={i:np.linalg.inv(H)@G[i]@np.linalg.inv(B) for i in G}
        for anchor in [0,2,4]:
            for row in gauge_errors(P,C,G,D,anchor).values():
                for v in row.values():self.assertLess(abs(v),1e-9)
    def test_static_source_misses_rotation(self):
        C={i:np.eye(4) for i in range(2)};P={i:pose(np.eye(3),[0,0,1]) for i in C}
        G={0:pose(np.eye(3),[0,0,1]),1:pose(Rotation.from_euler('y',45,degrees=True).as_matrix(),[0,0,1])}
        e=gauge_errors(P,C,G,C,0)
        self.assertAlmostEqual(e[1]['camera_rotation_deg'],45)
    def test_depth_scale_preserves_pinhole_projection(self):
        p=np.array([[.1,.2,1.],[-.1,.2,.8]])
        np.testing.assert_allclose(p[:,:2]/p[:,2,None],(p*1.4)[:,:2]/(p*1.4)[:,2,None])
    def test_triangle_distance_metric(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0]],float);f=np.array([[0,1,2]])
        np.testing.assert_allclose(distance(scene(v,f),[[.2,.2,.01]]),[.01],atol=1e-8)
    def test_similarity_oracle_recovers_known_transform(self):
        from shape_oracle import similarity
        rng=np.random.default_rng(4);x=rng.normal(size=(200,3))
        R=Rotation.from_rotvec([.5,-.2,.4]).as_matrix();s=.75;t=np.array([.1,.2,-.3])
        ss,rr,tt=similarity(x,s*(x@R.T)+t)
        self.assertAlmostEqual(ss,s,places=12)
        np.testing.assert_allclose(rr,R,atol=1e-12);np.testing.assert_allclose(tt,t,atol=1e-12)
    def test_native_anchor_transport_identity(self):
        C=pose(Rotation.from_rotvec([.1,.2,.3]).as_matrix(),[.1,.2,.3]);P=pose(Rotation.from_rotvec([-.1,.3,.1]).as_matrix(),[.2,0,.7])
        np.testing.assert_allclose(np.linalg.inv(C)@P@np.linalg.inv(P)@C,np.eye(4),atol=1e-12)

if __name__=='__main__':unittest.main()
