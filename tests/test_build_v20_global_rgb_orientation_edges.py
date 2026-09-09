"""CPU-only mechanism tests; LightGlue matching is a labeled deterministic double."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_v20_global_rgb_orientation_edges as rgb
import fit_v20_late_window_prediction_only_se3 as se3
from build_v19_rgb_pose_initializer import initialize


class RGBMissingPose(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(52)
        uv = np.array([(x, y) for x in range(20, 61, 8) for y in range(20, 61, 8)], dtype=float)
        mask = np.ones((80, 80), np.uint8) * 255
        depth = np.ones((160, 160), np.float32)
        for x, y in uv:
            depth[round(2*y+.5), round(2*x+.5)] = rng.uniform(.7, 1.3)
        depth[40, 40] = 3.5  # known P09 quarantine inside an otherwise eligible source
        support = np.ones_like(depth, dtype=bool); support[40, 40] = False
        intr = np.array([130., 132., 79.5, 79.5])
        source_camera = np.eye(4)
        source_camera[:3, :3] = Rotation.from_rotvec([.02, .04, -.03]).as_matrix()
        source_camera[:3, 3] = [.1, -.1, .03]
        target_camera = np.eye(4)
        target_camera[:3, :3] = Rotation.from_rotvec([-.03, .02, .04]).as_matrix()
        target_camera[:3, 3] = [.05, -.04, .02]
        # Independent pinhole/rigid equations, not the producer's lifting helper.
        pixels = np.rint((uv + .5) * 2 - .5).astype(int)
        z = depth[pixels[:,1], pixels[:,0]].astype(float)
        camera_points = np.column_stack(((pixels[:,0]-intr[2])*z/intr[0],
                                         (pixels[:,1]-intr[3])*z/intr[1], z))
        world = camera_points @ source_camera[:3,:3].T + source_camera[:3,3]
        R = Rotation.from_rotvec([.04, -.06, .05]).as_matrix(); t = np.array([.03, .01, .02])
        Rc = target_camera[:3, :3].T @ R
        tc = target_camera[:3, :3].T @ (t-target_camera[:3, 3])
        K = np.array([[65., 0., 39.5], [0., 66., 39.5], [0., 0., 1.]])
        target_uv, _ = cv2.projectPoints(world.astype(np.float64).reshape(-1, 1, 3), cv2.Rodrigues(Rc)[0], tc.reshape(3,1), K, np.zeros(5))
        obj = {'object_id': 'thing', 'rigid_pose_observation_eligible': True,
               'visible_geometry_candidate': {'rigid_pose_observation_eligible': True,
                                              'world_vertices_sample_m': world[1:].tolist()}}
        frames = {2: {'frame_idx': 2, 'camera': {'T_world_camera_metric': source_camera.tolist()}, 'objects': [obj]},
                  0: {'frame_idx': 0, 'camera': {'T_world_camera_metric': target_camera.tolist()}, 'objects': [{'object_id':'thing'}]}}
        features = {2: {'keypoints': torch.tensor(uv[None])}, 0: {'keypoints': torch.tensor(target_uv.reshape(1,-1,2))}}
        args = SimpleNamespace(object_id='thing', min_confidence=.05, min_matches=30, min_inliers=30,
                               min_inlier_fraction=.2, pnp_iterations=1000, reprojection_threshold_px=2.5,
                               max_reprojection_median_px=1.5, max_edge_rotation_deg=25.)
        R0 = Rotation.from_rotvec([.2, -.1, .3]).as_matrix(); t0 = np.array([.1, .2, .8])
        poses = {2: (R0, t0)}
        pairs = np.column_stack((np.arange(len(uv)), np.arange(len(uv))))
        def matcher_double(_):
            return {'matches': pairs, 'scores': np.ones(len(pairs))}
        return dict(source_idx=2, target_idx=0, features=features, masks={2:mask,0:mask}, frames=frames,
                    depths=np.stack([depth, np.full_like(depth,np.nan)]), confidences=np.ones((2,160,160)),
                    depth_pos={2:0,0:1}, depth_intrinsics=np.stack([intr,intr]), source_wh=(160,160),
                    matcher=matcher_double, device=torch.device('cpu'), args=args, initial_pose_by_frame=poses,
                    source_support_by_frame={2:support}), world, R, t

    def pair(self, kwargs):
        utils = ModuleType('lightglue.utils'); utils.rbd = lambda value: value
        with patch.dict(sys.modules, {'lightglue.utils': utils}):
            return rgb.feature_pair(**kwargs)

    def test_opencv_pnp_missing_target_fixed_canonical_and_initializer(self):
        kwargs, world, R, t = self.fixture()
        original = copy.deepcopy(kwargs['initial_pose_by_frame'])
        with patch.object(rgb, 'lift_source_world', wraps=rgb.lift_source_world) as lift:
            row, motion, evidence = self.pair(kwargs)
        self.assertEqual(row['status'], 'accepted')
        self.assertEqual(row['source_samples_quarantined'], 1)
        self.assertEqual(len(lift.call_args.args[1]), len(world)-1)
        np.testing.assert_allclose(motion[0], R, atol=1e-7)
        np.testing.assert_allclose(motion[1], t, atol=1e-7)
        self.assertFalse(row['target_initial_pose_present'])
        self.assertEqual(set(kwargs['initial_pose_by_frame']), {2})
        R0,t0 = original[2]
        np.testing.assert_allclose(evidence['canonical_points'] @ R0.T + t0, world[1:], atol=1e-10)
        # Exercise the real hash-bound downstream source-fixed canonical decoder.
        base = {'object_id':'thing','pose_rows':[
            {'frame_idx':0,'status':'missing_accepted_metric_surface'},
            {'frame_idx':2,'status':'fit_to_visible_depth_samples','rigid_pose_observation_eligible':True,
             'rotation_world_from_completed_canonical_matrix':R0.tolist(),'translation_world_m':t0.tolist()}]}
        arrays = {'source_frame_idx':np.array([2]),'target_frame_idx':np.array([0]),
                  'rotation_source_to_target_world':np.array([motion[0]]),
                  'translation_source_to_target_world_m':np.array([motion[1]]),'accepted':np.array([True])}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'p14.json';path.write_text(json.dumps(base))
            data={**arrays,'metadata':np.array([json.dumps({'initial_pose_report':str(path),'object_id':'thing',
                    'input_sha256':{'initial_pose_report':rgb.sha256_file(path)}})]),
                  'reprojection_canonical_points':evidence['canonical_points'],
                  'reprojection_evidence_offsets':np.array([0,len(evidence['weights'])])}
            fixed,_=se3.fixed_rgb_source_world(data)
            np.testing.assert_allclose(fixed,world[1:],atol=1e-10)
            path.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):se3.fixed_rgb_source_world(data)
        out=initialize(base,{'frames':list(kwargs['frames'].values())},'thing',arrays,[row],2)
        target=out['pose_rows'][0]
        self.assertEqual(target['status'],'missing_accepted_metric_surface')
        self.assertTrue(target['rgb_initializer']['uncertain'])
        self.assertFalse(target['rgb_initializer']['direct_metric_observation_added'])
        from filter_v19_rigid_completion_multiview_support import trusted_pose_rows
        self.assertEqual([r['frame_idx'] for r in trusted_pose_rows(out)],[2])
        self.assertNotIn('translation_world_m',base['pose_rows'][0])

    def test_manifest_plane_and_P09_support_replay_feed_real_PnP(self):
        kwargs, _, R, t = self.fixture()
        yy, xx = np.mgrid[:160,:160]
        depth = (1 + .009*np.sin(xx/17) + .008*np.cos(yy/19)).astype(np.float32)
        depth[40,40] = 3.5
        intr = kwargs['depth_intrinsics'][0]
        uv = kwargs['features'][2]['keypoints'].numpy()[0]
        pixels = np.rint((uv+.5)*2-.5).astype(int)
        z = depth[pixels[:,1],pixels[:,0]].astype(float)
        camera_points=np.column_stack(((pixels[:,0]-79.5)*z/130,(pixels[:,1]-79.5)*z/132,z))
        source_T=np.asarray(kwargs['frames'][2]['camera']['T_world_camera_metric'])
        world=camera_points@source_T[:3,:3].T+source_T[:3,3]
        target_T=np.asarray(kwargs['frames'][0]['camera']['T_world_camera_metric'])
        predicted=(world@R.T+t-target_T[:3,3])@target_T[:3,:3]
        target_uv=predicted[:,:2]/predicted[:,2,None]*[65.,66.]+[39.5,39.5]
        kwargs['features'][0]['keypoints']=torch.tensor(target_uv[None])
        kwargs['depths']=np.stack([depth,np.full_like(depth,np.nan)])
        obj=kwargs['frames'][2]['objects'][0]
        geometry=obj['visible_geometry_candidate']
        geometry.update(source_width=160,source_height=160,frame_idx=2,
            world_vertices_sample_m=world[1:].tolist(),
            mask_depth_transform_contract={'mask_size_wh':[40,40],'depth_size_wh':[160,160],
                'A_depth_from_mask_coordinate_model':[[4,0,1.5],[0,4,1.5],[0,0,1]]},
            first_surface_depth_ownership={'input_valid_depth_pixels':25600,'retained_depth_pixels':25599,
                'removed_depth_pixels':1,'state':'validated_sparse_interior_and_boundary_depth_quarantined'})
        parameters={'min_depth_m':.1,'max_depth_m':5.,'robust_first_surface_depth_ownership':True}
        settings={'mad_sigma':2.5,'min_half_width_m':.03,'min_retained_fraction':.9,
                  'fail_raw_to_robust_extent_ratio':2.,'confidence_seed_percentile':95.,
                  'local_depth_step_max_m':.005,'max_removed_distance_inside_mask_px':10.,
                  'max_confidence_flagged_interior_fraction':.015,'min_interior_confidence_flagged_fraction':.95,
                  'max_unexplained_interior_pixels':5,'min_component_pixels':20,'max_small_component_fraction':.01}
        parameters.update({'first_surface_'+key:value for key,value in settings.items()})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest_path=root/'manifest.json';rows=[]
            for index in [0,2]:
                image=root/f'{index}.jpg';cv2.imwrite(str(image),np.full((80,80,3),125,np.uint8))
                mask=root/f'mask_{index}.png';cv2.imwrite(str(mask),np.full((40,40),255,np.uint8))
                kwargs['frames'][index]['time_s']=index/30
                kwargs['frames'][index]['objects'][0]['mask_path']=str(mask)
                rows.append({'frame_idx':index,'time_s':index/30,'source_width':160,'source_height':160,
                             'manifest_width':80,'manifest_height':80,'rgb':str(image)})
            manifest_path.write_text(json.dumps({'frames':rows}))
            images,native,masks,planes=rgb.preflight_manifest_images(manifest_path,kwargs['frames'],[0,2],'thing',(160,160))
            self.assertEqual(images[0].shape,(80,80,3))
            self.assertEqual(native[2].shape,(40,40))
            self.assertEqual(masks[2].shape,(80,80))
            self.assertEqual(planes[0]['source_size_wh'],[160,160])
            support=rgb.reproduce_source_depth_support(obj,native[2],depth,np.ones_like(depth),intr,parameters)
            expected=np.ones((160,160),bool);expected[40,40]=False
            np.testing.assert_array_equal(support,expected)
            kwargs['masks']=masks;kwargs['source_support_by_frame']={2:support}
            kwargs['target_evidence_size_wh']=(40,40)
            row,motion,evidence=self.pair(kwargs)
            self.assertEqual(row['status'],'accepted')
            self.assertEqual(row['source_samples_quarantined'],1)
            np.testing.assert_allclose(motion[0],R,atol=1e-6)
            np.testing.assert_allclose(motion[1],t,atol=1e-6)
            # Serialized coordinates match original mask raster, as expected by
            # material consumers, not the feature or source-video dimensions.
            np.testing.assert_allclose(evidence['target_uv'],(target_uv[1:]+.5)*.5-.5,atol=1e-9)
            np.testing.assert_allclose(evidence['target_intrinsics'],[32.5,33.,19.5,19.5],atol=1e-12)
            self.assertNotIn(0,kwargs['initial_pose_by_frame'])
            wrong=json.loads(manifest_path.read_text());wrong['frames'][0]['manifest_width']=160
            manifest_path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(RuntimeError,'RGB/manifest'):
                rgb.preflight_manifest_images(manifest_path,kwargs['frames'],[0,2],'thing',(160,160))

    def test_actual_1408_source_and_960_manifest_preflight_needs_no_target_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);rows=[];frames={}
            for idx in [0,30]:
                image=root/f'{idx}.jpg';mask=root/f'{idx}.png'
                cv2.imwrite(str(image),np.zeros((960,960,3),np.uint8))
                cv2.imwrite(str(mask),np.full((960,960),255,np.uint8))
                rows.append({'frame_idx':idx,'time_s':idx/30,'source_width':1408,'source_height':1408,
                    'manifest_width':960,'manifest_height':960,'rgb':str(image)})
                obj={'object_id':'thing','mask_path':str(mask)}
                if idx==0:obj['visible_geometry_candidate']={'source_width':1408,'source_height':1408,
                    'mask_depth_transform_contract':{'mask_size_wh':[960,960]}}
                frames[idx]={'frame_idx':idx,'time_s':idx/30,'objects':[obj]}
            path=root/'manifest.json';path.write_text(json.dumps({'frames':rows}))
            images,native,masks,_=rgb.preflight_manifest_images(path,frames,[0,30],'thing',(1408,1408))
            self.assertEqual(set(images),{0,30})
            np.testing.assert_array_equal(masks[30],native[30])
            # This is the exact old assertion's failure, independent of models.
            self.assertNotEqual(images[0].shape[1::-1],(1408,1408))
            import inspect
            main=inspect.getsource(rgb.main)
            self.assertLess(main.index('preflight_manifest_images('),main.index('SuperPoint('))

    def test_rejected_source_guard_precedes_matching_or_depth_access(self):
        for kind in ('missing_p09','explicitly_rejected','missing_pose','invalid_surface'):
            with self.subTest(kind=kind):
                kwargs,_,_,_=self.fixture()
                obj=kwargs['frames'][2]['objects'][0]
                if kind=='missing_p09':obj.pop('visible_geometry_candidate')
                if kind=='explicitly_rejected':obj['rigid_pose_observation_eligible']=False
                if kind=='missing_pose':kwargs['initial_pose_by_frame']={}
                if kind=='invalid_surface':obj['visible_geometry_candidate']['world_vertices_sample_m']=[]
                kwargs['depths']=None
                kwargs['matcher']=lambda _: self.fail('matcher must not be invoked')
                with patch.object(rgb,'lift_source_world',side_effect=AssertionError('depth lifted')):
                    row,motion,evidence=rgb.feature_pair(**kwargs)
                self.assertEqual(row['status'],'rejected');self.assertIsNone(motion);self.assertIsNone(evidence)

    def test_leading_interior_trailing_and_out_of_bound_targets(self):
        ids=list(range(12)); sources={2,3,8}
        pairs,rejections=rgb.plan_source_edge_pairs(ids,[0,3,6,9,11],sources,1,3)
        self.assertTrue(all(a in sources for a,b in pairs))
        for pair in [(2,0),(2,1),(3,4),(3,5),(8,5),(3,6),(8,6),(8,7),(8,9),(8,10),(8,11)]:
            self.assertIn(pair,pairs)
        self.assertIn((2,3),pairs)
        self.assertTrue(rejections)
        sparse,_=rgb.plan_source_edge_pairs(list(range(20)),[],{2},1,3)
        self.assertFalse(any(b==19 for _,b in sparse))
        full,_=rgb.plan_source_edge_pairs(ids,[0,3,6,9,11],set(ids),2,6)
        self.assertEqual(full,rgb.build_edge_pairs(ids,[0,3,6,9,11],2))
        poses={2:(np.eye(3),np.ones(3))}
        chain=rgb.diagnostic_motion_chain(2,poses,{(2,0):(np.eye(3),np.array([.1,.2,.3]))})
        self.assertEqual(set(chain),{0,2});self.assertEqual(set(poses),{2})

    def test_valid_initial_poses_does_not_fill_or_accept_invalid_se3(self):
        report={'pose_rows':[{'frame_idx':0,'status':'missing_accepted_metric_surface'},
            {'frame_idx':1,'rotation_world_from_completed_canonical_matrix':np.zeros((3,3)).tolist(),'translation_world_m':[0,0,0]},
            {'frame_idx':2,'rotation_world_from_completed_canonical_matrix':np.eye(3).tolist(),'translation_world_m':[0,0,1]}]}
        before=copy.deepcopy(report)
        self.assertEqual(set(rgb.valid_initial_poses(report)),{2});self.assertEqual(report,before)


if __name__=='__main__':unittest.main()
