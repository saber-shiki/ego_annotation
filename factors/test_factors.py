

import unittest, sys
from pathlib import Path
import numpy as np
import torch

from utils.geometry import (
    camera,
    world,
    project,
    resized_K,
    torch_rodrigues,
)


from optimization.common.correspondences import make_scene, closest

class DemoGeometry(unittest.TestCase):
    def test_camera_roundtrip_and_halfpixel(self):
        T = np.eye(4)
        T[:3, :3] = torch_rodrigues(
            torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64)
        ).numpy()
        T[:3, 3] = [0.1, 0.2, -0.1]
        p = np.array([[0.1, 0.2, 0.5], [-0.2, 0.1, 1.0]])
        np.testing.assert_allclose(camera(world(p, T), T), p, atol=1e-12)
        K = np.array([900.0, 910.0, 702.0, 705.0])
        k = resized_K(K)
        np.testing.assert_allclose(
            project(p, k), (project(p, K) + 0.5) * (960 / 1408) - 0.5
        )

    def test_camera_origin_demo_scale_preserves_pixels(self):
        p = np.array([[0.1, 0.2, 0.5], [-0.2, 0.1, 1.0]])
        K = np.array([900.0, 910.0, 702.0, 705.0])
        np.testing.assert_allclose(project(p * 1.13, K), project(p, K))
        self.assertGreater(np.linalg.norm(1.13 * p - p), 0.01)

    def test_zero_rotation_has_nonzero_gradient(self):
        r = torch.zeros(3, requires_grad=True)
        R = torch_rodrigues(r)
        loss = (R @ torch.tensor([0.0, 0.0, 1.0]))[0]
        loss.backward()
        self.assertGreater(abs(r.grad[1].item()), 0.9)
        np.testing.assert_allclose(R.detach().numpy(), np.eye(3))

    def test_nonzero_torch_numpy_export_transform_parity(self):
        from scipy.spatial.transform import Rotation

        rng = np.random.default_rng(2)
        v = rng.normal(size=(2, 8, 3)) * 0.05 + [0.1, 0.15, 0.4]
        pivot = v[:, :1].copy()
        R = Rotation.from_rotvec([[0.05, -0.07, 0.02], [-0.02, 0.03, 0.01]]).as_matrix()
        C = Rotation.from_rotvec([[0.1, 0.2, 0.05], [-0.1, 0.15, 0]]).as_matrix()
        ct = np.array([[0.01, 0.02, 0.03], [0.02, -0.02, 0.01]])
        scale = np.array([0.96, 1.03])[:, None, None]
        shift = np.array([[0.001, -0.002, 0.003], [-0.002, 0.001, 0.002]])[:, None]
        rotated = (v - pivot) @ R.transpose(0, 2, 1) + pivot
        cam = (rotated - ct[:, None]) @ C
        expected = (cam * scale + shift) @ C.transpose(0, 2, 1) + ct[:, None]
        t = lambda a: torch.as_tensor(a, dtype=torch.float64)
        rotated_t = (t(v) - t(pivot)) @ t(R).transpose(1, 2) + t(pivot)
        camera_t = (rotated_t - t(ct)[:, None]) @ t(C)
        actual = (camera_t * t(scale) + t(shift)) @ t(C).transpose(1, 2) + t(ct)[
            :, None
        ]
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)

    def test_triangle_contact_and_joint_geometry_occlusion(self):
        import open3d as o3d

        vertices = np.array([[-1, -1, 1], [1, -1, 1], [0, 1, 1]], float)
        faces = np.array([[0, 1, 2]])
        sc = make_scene(vertices, faces)
        q, n = closest(sc, np.array([[0, 0, 1.03]]))
        self.assertAlmostEqual(np.linalg.norm(q - [0, 0, 1.03]), 0.03, places=6)
        sc.add_triangles(
            o3d.core.Tensor((vertices * 0.5).astype("float32")),
            o3d.core.Tensor(faces.astype("uint32")),
        )
        hit = sc.cast_rays(o3d.core.Tensor(np.array([[0, 0, 0, 0, 0, 1]], np.float32)))
        self.assertEqual(int(hit["geometry_ids"].numpy()[0]), 1)
        self.assertAlmostEqual(float(hit["t_hit"].numpy()[0]), 0.5)



"""Primary pad activation must couple to unsigned full-triangle fitting gradients."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from gates.interaction.contact import primary_contact_pads
from gates.interaction.contact import (visual_proximity_weights, primary_similarity_reachable,
    HAND_SCALE_BOUNDS, HAND_TRANSLATION_BOUND_M)
from factors.interaction.contact import unsigned_pad_loss
from optimization.common.correspondences import make_scene, closest
from utils.geometry import project


def payload(state='likely_contact', digits=None):
    row={'target_entity_id':'object:item','hand_side':'left','frame_start':10,'frame_end':80,
         'contact_state':state}
    if digits is not None:
        row['primary_contact_pads']={'digits':digits,'rationale':'Inspected frames show index pad support; occlusion uncertain.'}
    return {'interaction_judgments':[row]}


class PrimaryPads(unittest.TestCase):
    def test_interval_anatomy_and_invalid_declarations(self):
        ids=[10,37,80,123]
        selected=primary_contact_pads(payload(digits=['index','ring']),ids,'item')
        self.assertEqual(np.flatnonzero(selected[0,0]).tolist(),[1,3])
        self.assertFalse(selected[:,1].any());self.assertFalse(selected[-1].any())
        self.assertFalse(primary_contact_pads(None,ids,'item').any())
        for digits in (['index','index'],['wrist'],[1],'thumb'):
            with self.assertRaises(ValueError):primary_contact_pads(payload(digits=digits),ids,'item')
        bad=payload(digits=['index']);bad['interaction_judgments'][0]['primary_contact_pads']['rationale']=' '
        with self.assertRaises(ValueError):primary_contact_pads(bad,ids,'item')

    def test_noncontact_possible_unresolved_missing_visibility_do_not_expand(self):
        for state in ('no_contact','unresolved','possible_contact'):
            selected=primary_contact_pads(payload(state,['index']),[10],'item')
            self.assertFalse(selected.any())
            w=visual_proximity_weights([True]*5,[0.]*5,[.03]*5,state=state,primary=[True]*5,reachable=[True]*5)
            self.assertFalse(w.any())
        p=payload(digits=['index'])
        for state in ('out_of_frame','occluded','unresolved'):
            p['frames']=[{'frame_idx':10,'visibility':{'left':{'state':state},'object':{'state':'visible'}}}]
            self.assertFalse(primary_contact_pads(p,[10],'item').any())
        for designation in (None, [], ['index']):
            selected=primary_contact_pads(payload(digits=designation),[10],'item')[0,0]
            w=visual_proximity_weights([False]*5,[0.]*5,[.03]*5,primary=selected,reachable=[True]*5)
            self.assertFalse(w.any())
        self.assertFalse(visual_proximity_weights([True]*5,[0.]*5,[.03]*5,primary=[True]*5).any())
        self.assertFalse(visual_proximity_weights([True]*5,[0.]*5,[.03]*5,reachable=[True]*5).any())

    def test_primary_20_40mm_actual_triangle_gradient_and_finite_bound(self):
        scene=make_scene(np.array([[-1.,-1.,1.],[1.,-1.,1.],[0.,1.,1.]]),np.array([[0,1,2]]))
        for gap in (.020,.030,.040):
            points=np.tile([0.,0.,1.+gap],(5,1))
            target,_=closest(scene,points)
            reach=primary_similarity_reachable(points,target)
            self.assertTrue(reach.all())
            selected=primary_contact_pads(payload(digits=['index']),[10],'item')[0,0]
            weights=visual_proximity_weights([True]*5,[0.]*5,np.linalg.norm(points-target,axis=1),primary=selected,reachable=reach)
            self.assertEqual(np.flatnonzero(weights).tolist(),[1])
            # Same camera-origin scale, 12-sample pad and top-3 unsigned loss
            # as production, to a point inside a complete triangle (not vertices).
            logscale=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
            samples=torch.tensor(np.tile(points[1],(12,1)))
            targets=torch.tensor(np.tile(target[1],(12,1)))
            loss=unsigned_pad_loss(samples*torch.exp(logscale)-targets,float(weights[1]))
            loss.backward()
            self.assertGreater(logscale.grad.item(),0.)
            candidate_scale=1./(1.+gap)
            self.assertTrue(HAND_SCALE_BOUNDS[0] <= candidate_scale <= HAND_SCALE_BOUNDS[1])
            corrected=samples*candidate_scale
            self.assertLess(unsigned_pad_loss(corrected-targets,1.).item(),1e-10)
            np.testing.assert_allclose(project(points,np.array([600,600,300,300])),project(points*candidate_scale,np.array([600,600,300,300])))
        weights=visual_proximity_weights([True]*5,[0.]*5,[.05,np.nan,np.inf,-.1,.03],primary=[True]*5,reachable=[True]*5)
        self.assertEqual(np.flatnonzero(weights).tolist(),[4])

    def test_reachability_requires_one_shared_similarity_for_all_active_pads(self):
        points=np.array([[.5,.1,.5],[.5,-.1,.5],[.03,.04,.20],[.02,-.04,.20],[.01,.03,.20]])
        # Each point separately admits a scale/translation, but the two active
        # targets demand incompatible shared scales; no expanded gate is safe.
        target=points.copy(); target[0]*=1.18; target[1]*=.85
        independent=primary_similarity_reachable(points,target,active=[True,False,False,False,False])
        self.assertTrue(independent[0])
        independent=primary_similarity_reachable(points,target,active=[False,True,False,False,False])
        self.assertTrue(independent[1])
        self.assertFalse(primary_similarity_reachable(points,target,active=[True,True,False,False,False]).any())

    def test_shared_translation_rejects_individually_feasible_symmetric_targets(self):
        points=np.array([[-.03,0.,1.],[.03,0.,1.],[0.,.03,1.],[0.,-.03,1.],[0.,0.,1.]])
        target=np.array([[-.01,0.,1.],[.01,0.,1.],[0.,.03,1.],[0.,-.03,1.],[0.,0.,1.]])
        self.assertTrue(primary_similarity_reachable(points,target,active=[True,False,False,False,False])[0])
        self.assertTrue(primary_similarity_reachable(points,target,active=[False,True,False,False,False])[1])
        self.assertFalse(primary_similarity_reachable(points,target,active=[True,True,False,False,False]).any())


        points=np.tile([0.,0.,.1],(5,1))
        target=points+[.04,0.,0.]
        self.assertFalse(primary_similarity_reachable(points,target).any())
        target=points+[0.,0.,.04]
        self.assertTrue(primary_similarity_reachable(points,target).all())
        target=points+[0.,0.,.06]
        self.assertFalse(primary_similarity_reachable(points,target).any())
        target=points+[HAND_TRANSLATION_BOUND_M,0.,0.]
        self.assertTrue(primary_similarity_reachable(points,target).all())
        points[0]=np.nan;points[1,2]=-.1
        self.assertFalse(primary_similarity_reachable(points,target)[:2].any())


class FormalNormalizationTests(unittest.TestCase):
    def test_contact_probability_receives_geometry_gradient(self):
        from factors.interaction.contact import normal_deadband
        probability=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
        gap=torch.tensor([.03,.02],dtype=torch.float64,requires_grad=True)
        loss=normal_deadband(gap,.01,25000.,probability)
        loss.backward()
        self.assertAlmostEqual(probability.grad.item(),25000*(.02**2+.01**2)/2)
        np.testing.assert_allclose(gap.grad.numpy(),25000*.4*np.array([.02,.01]))

    def test_formal_root_uses_nine_visibility_denominator(self):
        from factors.hand.priors import formal_root_orientation
        delta=torch.ones((2,1,3),dtype=torch.float64)
        visibility=torch.tensor([1.,.5],dtype=torch.float64)
        self.assertAlmostEqual(formal_root_orientation(delta,visibility,9.).item(),3.)

if __name__ == '__main__':
    unittest.main()
