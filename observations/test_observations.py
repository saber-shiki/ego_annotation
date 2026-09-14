"""Synthetic CPU checks of local RGB seeding, loop filtering and surface authority."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from utils.data import PoseNode, RGBEdge, RGBReprojectionFactor, PointFactor
from observations.object.rgb_tracking import local_rotation_evidence, observed_rotation_authority
from optimization.object.pose import solve_scene_pose as solve, GraphConfig, residual_blocks


def fixture(count=9, ids=None):
    ids = ids or [17+31*i for i in range(count)]
    times = np.linspace(0, 2, count)
    truth = [Rot.from_rotvec(np.array([.4, -.2, 1.]) * t).as_matrix() for t in times]
    anchor = count//2
    nodes = [PoseNode(i, truth[anchor].copy() if k == anchor else np.eye(3),
                      np.array([0., 0., 1.]), None, 'synthetic', True, .1,
                      {'frame_idx': i, 'source': 'immutable'}) for k, i in enumerate(ids)]
    edges = []
    for k in range(count-1):
        # All local edges reversed, with world-relative translation (not tt-ts).
        a, b = k+1, k
        d = truth[b] @ truth[a].T
        edges.append(RGBEdge(ids[a], ids[b], d, nodes[b].initial_translation-d@nodes[a].initial_translation,
                             .8, {'source_3d_conditioning': .0001, 'inlier_fraction': .9, 'reprojection_median_px': .1}))
    for a, b in [(0, count-1), (1, count-2), (0, count-2), (1, count-1)]:
        edges.append(RGBEdge(ids[a], ids[b], Rot.from_euler('x', .05).as_matrix(), np.zeros(3),
                             2., {'source_3d_conditioning': .0001, 'inlier_fraction': 1., 'reprojection_median_px': .05}))
    return nodes, times, edges, truth, ids[anchor]


class RotationChainTests(unittest.TestCase):
    def test_large_rotation_multiple_planar_loops_and_full_solve(self):
        nodes, times, edges, truth, anchor = fixture()
        original = copy.deepcopy(nodes)
        cfg = GraphConfig(outer_iterations=3, inner_evaluations=35, observed_weight=0,
                          correction_velocity_rad_s=10, acceleration_rad_s2=100)
        result, report = solve(nodes, times, edges, anchor, cfg)
        for k, n in enumerate(nodes):
            self.assertLess(Rot.from_matrix(result[n.frame_idx][0] @ truth[k].T).magnitude(), .02)
            np.testing.assert_array_equal(n.initial_rotation, original[k].initial_rotation)
            self.assertEqual(n.source_row, original[k].source_row)
        self.assertGreater(np.degrees(Rot.from_matrix(result[nodes[-1].frame_idx][0] @ result[nodes[0].frame_idx][0].T).magnitude()), 120)
        self.assertTrue(all(e['weight_scale'] == 0 for e in report['local_RGB_rotation']['edges'][-4:]))
        self.assertEqual(set(report['stage_snapshots']), {'SO3', 'SE3'})
        np.testing.assert_array_equal(result[anchor][0], original[4].initial_rotation)

    def test_ids_edge_order_direction_and_rate_do_not_change_seed(self):
        results = []
        for count in (9, 17):
            ids = [10000 - k*117 for k in range(count)]
            nodes, times, edges, truth, anchor = fixture(count, ids)
            edges = [replace(e, source_frame_idx=e.target_frame_idx, target_frame_idx=e.source_frame_idx,
                             rotation_source_to_target_world=e.rotation_source_to_target_world.T,
                             translation_source_to_target_world_m=-e.rotation_source_to_target_world.T @ e.translation_source_to_target_world_m)
                     for e in reversed(edges)]
            seed, _, _, _ = local_rotation_evidence(nodes, times+37.5, edges, anchor)
            for n, r in zip(seed, truth):
                np.testing.assert_allclose(n.initial_rotation, r, atol=1e-12)
            results.append(seed[-1].initial_rotation)
        np.testing.assert_allclose(*results, atol=1e-12)

    def test_gap_bridge_disconnection_and_weak_support_do_not_trust_icp(self):
        nodes, times, edges, _, anchor = fixture()
        # Endpoint 3 loses depth and adjacent edges; a nearest valid-source edge bridges it.
        nodes[3] = replace(nodes[3], metric_observation=False, uncertainty=.95)
        kept = [e for e in edges[:8] if nodes[3].frame_idx not in (e.source_frame_idx,e.target_frame_idx)]
        kept.append(edges[3])  # reversed 4->3 is a valid metric-source gap bracket
        d = edges[3].rotation_source_to_target_world @ edges[2].rotation_source_to_target_world
        # Explicit reverse bridge 4->2 with correct composition.
        d = edges[2].rotation_source_to_target_world @ edges[3].rotation_source_to_target_world
        kept.append(RGBEdge(nodes[4].frame_idx,nodes[2].frame_idx,d,np.zeros(3),.8))
        seed, _, _, report = local_rotation_evidence(nodes,times,kept,anchor)
        self.assertFalse(report['local_disconnected_ids'])
        self.assertGreaterEqual(report['seed_rows'][3]['uncertainty'],.95)
        self.assertFalse(seed[3].metric_observation)
        # Only a long edge must remain uncertain, not bootstrap its own validator.
        seed, filtered, _, report = local_rotation_evidence(nodes,times,[edges[-1]],anchor)
        self.assertGreater(len(report['local_disconnected_ids']), 0)
        self.assertEqual(report['edges'][0]['reason'],'unverified_weak_or_disconnected_local_path')
        self.assertGreater(filtered[0].weight,0)
        np.testing.assert_array_equal(seed[0].initial_rotation,nodes[0].initial_rotation)
        weak = [replace(e,diagnostics={**e.diagnostics,'inlier_fraction':.01}) for e in edges[:8]] + [edges[-1]]
        _, _, _, report = local_rotation_evidence(nodes,times,weak,anchor)
        self.assertEqual(report['edges'][-1]['reason'],'unverified_weak_or_disconnected_local_path')

    def test_rejected_edges_and_pixels_share_decision_without_mutation(self):
        nodes,times,edges,_,anchor=fixture()
        bad=edges[-1];a,b=bad.source_frame_idx,bad.target_frame_idx
        ids=[n.frame_idx for n in nodes]
        f=RGBReprojectionFactor(a,b,ids.index(a),ids.index(b),np.array([[0.,0.,1.]]),
                                np.zeros((1,2)),np.array([100.,100.,0.,0.]),np.eye(3),np.zeros(3),np.ones(1),2.)
        seed, filtered, projections, report=local_rotation_evidence(nodes,times,edges,anchor,[f])
        self.assertEqual(projections[0].edge_weight,0)
        self.assertEqual(filtered[-1].weight,0)
        self.assertEqual(f.edge_weight,2.)
        self.assertEqual(edges[-1].weight,2.)
        self.assertNotIn(len(edges)-1,report['edges'][-1]['independent_path_edge_indices'])
        poses={n.frame_idx:(n.initial_rotation,n.initial_translation) for n in seed}
        def residual(fs):
            return np.concatenate([v for _,v in residual_blocks(poses,seed,times,filtered,[],fs,GraphConfig())])
        np.testing.assert_array_equal(residual(projections),residual([]))

    def test_planar_surface_retains_translation_but_not_rotation_authority(self):
        nodes,times,edges,_,anchor=fixture()
        xy=np.array([[x,y,0.] for x in (-.03,0,.03) for y in (-.02,0,.02)])
        nodes=[replace(n,observed_world=xy+n.initial_translation) for n in nodes]
        seed,_,_,chain=local_rotation_evidence(nodes,times,edges,anchor)
        scales=observed_rotation_authority(seed,chain)
        self.assertTrue(all(v==0 for v in scales.values()))
        poses={n.frame_idx:(n.initial_rotation,n.initial_translation) for n in seed}
        f=PointFactor(0,1,nodes[0].observed_world,nodes[1].observed_world,np.tile([0.,0.,1.],(9,1)),np.ones(9))
        def observed(p):
            blocks=residual_blocks(p,seed,times,[],[f],[],GraphConfig(),False,poses,scales)
            return np.concatenate([v for _,v in blocks[2*len(nodes):2*len(nodes)+2]])
        changed=dict(poses);r,t=changed[nodes[1].frame_idx]
        changed[nodes[1].frame_idx]=(Rot.from_euler('x',.2).as_matrix()@r,t)
        np.testing.assert_allclose(observed(changed),observed(poses),atol=1e-12)
        changed[nodes[1].frame_idx]=(r,t+[0,0,.01])
        self.assertGreater(np.linalg.norm(observed(changed)-observed(poses)),.1)


    unittest.main()

if __name__ == '__main__':
    unittest.main()
