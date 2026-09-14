

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



class FormalKernelParityTests(unittest.TestCase):
    def assert_kernel(self, kernel, reference, value):
        a=torch.tensor(value,dtype=torch.float64,requires_grad=True)
        b=a.detach().clone().requires_grad_()
        actual=kernel(a);expected=reference(b)
        self.assertAlmostEqual(actual.item(),expected.item(),places=12)
        ga=torch.autograd.grad(actual,a)[0];gb=torch.autograd.grad(expected,b)[0]
        np.testing.assert_allclose(ga.detach(),gb.detach(),atol=1e-12,rtol=1e-12)

    def test_active_dense_and_depth_order_preserve_distinct_normalizations(self):
        from factors.interaction.physics import signed_active_penetration, signed_dense_penetration, visible_surface_depth_order
        normals=torch.tensor([[0.,0.,1.]]*3,dtype=torch.float64)
        depth=torch.tensor([.04,.01,-.02],dtype=torch.float64)
        def residual(moved):return torch.relu(depth-.005-(normals*moved).sum(1))
        self.assert_kernel(lambda v:signed_active_penetration(v,normals,depth,.005,7.),
            lambda v:7.*residual(v).square().mean(),np.zeros((3,3)))
        self.assert_kernel(lambda v:signed_dense_penetration(v,normals,depth,.005,7.),
            lambda v:7.*residual(v).square().sum()/(residual(v)>0).sum().clamp_min(1.),np.zeros((3,3)))
        target=torch.tensor([1.02,1.,.95],dtype=torch.float64)
        def order(v):return torch.relu(target-.01-v)
        self.assert_kernel(lambda v:visible_surface_depth_order(v,target,.01,9.),
            lambda v:9.*order(v).square().sum()/(order(v)>0).sum().clamp_min(1.),[.98,1.,1.])

    def test_observation_ray_object_and_magnitude_terms_match_reference_gradients(self):
        from factors.hand.observation import joint_pixel_deadband,joint_depth_deadband
        from factors.hand.priors import hand_ray_translation_prior
        from factors.object.pose import object_translation_prior
        from factors.common import magnitude_hinge
        weights=torch.tensor([1.,.2],dtype=torch.float64)
        base=torch.zeros((2,2),dtype=torch.float64)
        self.assert_kernel(lambda v:joint_pixel_deadband(v,base,weights,1.,3.),
            lambda v:3.*(weights*torch.relu(torch.linalg.norm(v,dim=1)-1.).square()).sum()/weights.sum().clamp_min(1.),[[2.,1.],[.1,.2]])
        target=torch.ones(2,dtype=torch.float64)
        self.assert_kernel(lambda v:joint_depth_deadband(v,target,weights,.02,3.),
            lambda v:3.*(weights*torch.relu(torch.abs(v-target)-.02).square()).sum()/weights.sum().clamp_min(1.),[1.05,.99])
        target=torch.zeros((2,3),dtype=torch.float64)
        self.assert_kernel(lambda v:hand_ray_translation_prior(v,target,weights),
            lambda v:(weights[:,None]*(v-target).square()).sum()/v.numel(),[[.01,0,0],[.02,0,0]])
        self.assert_kernel(lambda v:object_translation_prior(v,5.),lambda v:5.*v.square().mean(),[[.01,.02,0],[0,0,.03]])
        self.assert_kernel(lambda v:magnitude_hinge(v,.02,6.),
            lambda v:6.*torch.relu(torch.linalg.norm(v,dim=-1)-.02).square().mean(),[[.03,.04,0],[0,0,.01]])


"""Real RGB-D lifting/OpenCV PnP → pose-graph world-motion convention tests.

Only feature correspondence matching is a deterministic synthetic double. The
production source eligibility, lifting, PnP/refinement and edge residual run.
"""
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import observations.object.rgb_tracking as builder
from utils.data import RGBEdge
from factors.object.pose import rgb_edge_residual


def rgb_edge_fixture(moving=True):
    rng = np.random.default_rng(27)
    uv = np.array([(x, y) for y in range(170, 311, 28) for x in range(220, 421, 40)], float)
    wh = (640, 480)
    K4 = np.array([420., 410., 319.5, 239.5])
    depth = np.ones(wh[::-1], float)
    depth[uv[:, 1].astype(int), uv[:, 0].astype(int)] = rng.uniform(.8, 1.3, len(uv))
    Z = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    source_camera_points = np.c_[(uv[:, 0] - K4[2]) * Z / K4[0],
                                (uv[:, 1] - K4[3]) * Z / K4[1], Z]
    Cs, Ct = np.eye(4), np.eye(4)
    Cs[:3, :3] = Rotation.from_rotvec([.02, .04, -.03]).as_matrix()
    Cs[:3, 3] = [.1, -.1, .03]
    Ct[:3, :3] = Rotation.from_rotvec([-.03, .02, .04]).as_matrix()
    Ct[:3, 3] = [.05, -.04, .02]
    source_world = source_camera_points @ Cs[:3, :3].T + Cs[:3, 3]
    Rs = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
    ts = np.array([.1, .2, .8])
    Rt = Rotation.from_rotvec([.4, -.35, .2]).as_matrix() @ Rs if moving else Rs.copy()
    tt = ts + [.005, -.01, .03] if moving else ts.copy()
    # Independent material-point construction, not the builder's edge formula.
    canonical = (source_world - ts) @ Rs
    target_world = canonical @ Rt.T + tt
    target_camera_points = (target_world - Ct[:3, 3]) @ Ct[:3, :3]
    target_uv = target_camera_points[:, :2] / target_camera_points[:, 2:] * K4[:2] + K4[2:]
    assert (target_camera_points[:, 2] > 0).all()
    assert (target_uv > 10).all() and (target_uv < np.array(wh) - 10).all()
    geom = {
        "frame_idx": 7, "object_id": "tool", "rigid_pose_observation_eligible": True,
        "world_vertices_sample_m": source_world.tolist(),
        "camera_vertices_sample_m": source_camera_points.tolist(),
        "first_surface_depth_ownership": {
            "enabled": True, "fail_closed": False, "retained_fraction": 1., "failure_reasons": []},
    }
    obj = {"object_id": "tool", "rigid_pose_observation_eligible": True,
           "visible_geometry_candidate": geom}
    frames = {
        7: {"frame_idx": 7, "camera": {"T_world_camera_metric": Cs.tolist(),
            "intrinsics_fx_fy_cx_cy": K4.tolist()}, "objects": [obj]},
        31: {"frame_idx": 31, "camera": {"T_world_camera_metric": Ct.tolist(),
             "intrinsics_fx_fy_cx_cy": K4.tolist()}, "objects": [{"object_id": "tool"}]},
    }
    pairs = np.c_[np.arange(len(uv)), np.arange(len(uv))]
    def matches(_):
        return {"matches": pairs, "scores": np.ones(len(pairs))}
    args = SimpleNamespace(object_id="tool", min_confidence=.05, min_matches=20,
                           pnp_iterations=1000, reprojection_threshold_px=1.)
    kwargs = dict(source_idx=7, target_idx=31,
        features={7: {"keypoints": torch.tensor(uv[None])},
                  31: {"keypoints": torch.tensor(target_uv[None])}},
        masks={i: np.full(wh[::-1], 255, np.uint8) for i in frames}, frames=frames,
        depths=np.stack([depth, np.full_like(depth, np.nan)]),
        confidences=np.ones((2, *wh[::-1])), depth_pos={7: 0, 31: 1},
        depth_intrinsics=np.stack([K4, K4]), source_wh=wh, matcher=matches,
        device=torch.device("cpu"), args=args,
        initial_pose_by_frame={7: (Rs, ts), 31: (Rt, tt)},
        source_support_by_frame={7: np.ones(wh[::-1], bool)})
    return kwargs, source_world, target_world


def produce(kwargs):
    utils = ModuleType("lightglue.utils")
    utils.rbd = lambda value: value
    cv2.setRNGSeed(27)
    with patch.dict(sys.modules, {"lightglue.utils": utils}):
        return builder.feature_pair(**kwargs)


class ProductionRGBEdgeConvention(unittest.TestCase):
    def check_motion(self, moving):
        kwargs, source, target = rgb_edge_fixture(moving)
        poses = kwargs["initial_pose_by_frame"]
        before = copy.deepcopy(poses)
        row, motion, evidence = produce(kwargs)
        self.assertEqual(row["status"], "accepted")
        Rs, ts = poses[7]
        Rt, tt = poses[31]
        expected_R = Rt @ Rs.T
        expected_t = tt - expected_R @ ts
        np.testing.assert_allclose(motion[0], expected_R, atol=1e-7, rtol=0)
        np.testing.assert_allclose(motion[1], expected_t, atol=1e-7, rtol=0)
        np.testing.assert_allclose(source @ motion[0].T + motion[1], target, atol=1e-7, rtol=0)
        rr, tr = rgb_edge_residual(RGBEdge(7, 31, *motion, 1.), poses)
        np.testing.assert_allclose(rr, 0, atol=1e-7, rtol=0)
        np.testing.assert_allclose(tr, 0, atol=1e-7, rtol=0)
        np.testing.assert_allclose(evidence["canonical_points"] @ Rs.T + ts, source, atol=1e-10, rtol=0)
        for i in poses:
            for a, b in zip(poses[i], before[i]):
                np.testing.assert_array_equal(a, b)
        return kwargs, motion, evidence

    def test_moving_object_nonidentity_poses_and_cameras(self):
        kwargs, motion, _ = self.check_motion(True)
        Rs, _ = kwargs["initial_pose_by_frame"][7]
        Rt, _ = kwargs["initial_pose_by_frame"][31]
        self.assertGreater(Rotation.from_matrix(motion[0]).magnitude(), .5)
        # Converting this world motion into the common canonical gauge would
        # erase the measured action, not fix the pose graph's convention.
        double_converted = Rt.T @ motion[0] @ Rs
        np.testing.assert_allclose(double_converted, np.eye(3), atol=1e-7)
        self.assertGreater(np.linalg.norm(double_converted - motion[0]), .5)

    def test_static_object_nonidentity_pose_with_moving_camera(self):
        _, motion, _ = self.check_motion(False)
        np.testing.assert_allclose(motion[0], np.eye(3), atol=1e-7, rtol=0)
        np.testing.assert_allclose(motion[1], 0, atol=1e-7, rtol=0)

    def test_arbitrary_canonical_gauge_does_not_change_emitted_world_motion(self):
        kwargs, motion, _ = self.check_motion(True)
        Q = Rotation.from_rotvec([.8, .2, -.4]).as_matrix()
        kwargs["initial_pose_by_frame"] = {
            i: (R @ Q, t.copy()) for i, (R, t) in kwargs["initial_pose_by_frame"].items()}
        row, changed, _ = produce(kwargs)
        self.assertEqual(row["status"], "accepted")
        np.testing.assert_allclose(changed[0], motion[0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(changed[1], motion[1], atol=1e-7, rtol=0)
        rr, tr = rgb_edge_residual(RGBEdge(7, 31, *changed, 1.), kwargs["initial_pose_by_frame"])
        np.testing.assert_allclose(rr, 0, atol=1e-7, rtol=0)
        np.testing.assert_allclose(tr, 0, atol=1e-7, rtol=0)

if __name__ == '__main__':
    unittest.main()
