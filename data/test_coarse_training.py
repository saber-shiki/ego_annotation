"""Synthetic mechanisms for the explicit coarse training seam (not accuracy tests)."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest
import numpy as np

from data.bundle import EpisodeBundle, file_sha256
from data.coarse_training import (ADAPTER, CoarseTrainingBundle, validate_archive, validate_coarse,
                                  source_rgb_rows, EXTRA_FEATURES)
from data.test_rl_data import make_synthetic_bundle
from data.immutable import freeze
from data.interaction import load_interaction_evidence
from data.interaction_features import compile_interaction_timeline
from envs.coarse_annotation_env import CoarseAnnotationEnv


def fixture(root):
    path, models = make_synthetic_bundle(root)
    source_bundle = EpisodeBundle.open(path)
    source = dict(source_bundle.load_observations())
    obj = source['objects']['object:synthetic']
    n = len(source['frame_idx'])
    ids = source['frame_idx']
    world = np.asarray(obj['observed_points_world'][0])
    target_world = world + obj['translation'][1] - obj['translation'][0]
    uv = target_world[:, :2] / target_world[:, 2, None] * 60 + 32
    edge = SimpleNamespace(source_frame_idx=0, target_frame_idx=1, source_world=world, target_uv=uv,
        weights=np.ones(len(world)), rotation_source_to_target_world=np.eye(3),
        translation_source_to_target_world_m=np.array([.01,0,0]), target_camera=np.eye(4), target_intrinsics=np.array([60,60,32,32]),
        diagnostics={'reprojection_median_px':.5})
    source.update(visual_geometry=SimpleNamespace(vertices_metric_camera=obj['vertices_canonical'], faces=obj['faces']),
        rgb_evidence=SimpleNamespace(edges=(edge,)),
        motion_preprocessing=SimpleNamespace(qualification={'relative_edge_weights':(1.,),'reprojection_edge_weights':(1.,)}),
        interaction_timelines={'object:synthetic':compile_interaction_timeline(load_interaction_evidence(None,ids,['object:synthetic']),ids,'object:synthetic')})
    source['image_size_wh'] = np.array([64,64])
    from scipy.spatial.transform import Rotation
    archive = {'frame_idx':ids,'time_s':source['time_s'],'T_world_camera':source['T_world_camera'],
        'intrinsics_source':source['intrinsics'],'image_size_wh':np.tile([64,64],(n,1)),
        'object_rotation':np.repeat(Rotation.from_euler('z',12,degrees=True).as_matrix()[None],n,axis=0),
        'object_translation_world':obj['translation']+np.array([0.,.025,0.]),
        'object_valid':obj['valid'],'source_pose_available':obj['pose_valid'],
        'source_metric_eligible':obj['metric_observation_eligible'],
        'translation_depth_inferred':~obj['metric_observation_eligible'],
        'vertices_canonical':obj['vertices_canonical'].astype(np.float64)*1.2,'faces':obj['faces']}
    for side in ('left','right'):
        for field in ('vertices','joints','faces','valid'):
            archive[f'{side}_{field}']=source['hands'][side][field]
    array_path = root/'coarse.npz'; np.savez(array_path,**archive)
    payload = {'schema':'ego.annotation.monocular_coarse_scene.v1', 'prediction_side_only':True,
        'contains_optimized_state':True,'contains_policy_results':False,'GT_or_CAD_inputs':False,
        'additional_camera_inputs':False,'source_MANO_unchanged':True,'signed_geometry_ready':False,
        'canonical_frame':'centered_native_appearance_camera_axes','canonical_native_pivot_camera_m':[0.,0.,0.],
        'residual_uniform_object_scale':1.2,'object_id':'object:synthetic',
        'frame_ids':ids.tolist(),'time_s':source['time_s'].tolist(), 'source_observations':str(path),
        'source_bundle_sha256':file_sha256(path),'arrays':{'path':str(array_path),'sha256':file_sha256(array_path)}}
    coarse = root/'coarse.json';coarse.write_text(json.dumps(payload))
    manifest = dict(source_bundle.manifest);manifest.update(schema='ego.annotation.observation_bundle.v2',preprocessing_only=True)
    fake = SimpleNamespace(manifest=freeze(manifest),manifest_path=path,source_video_id=source_bundle.source_video_id,
                           load_observations=lambda:source)
    with patch('data.coarse_training.EpisodeBundle.open',return_value=fake):
        bundle = CoarseTrainingBundle(coarse,file_sha256(coarse))
    return bundle,source,archive,models


class CoarseAdapterTests(unittest.TestCase):
    def test_authority_hash_timeline_and_validity_rejections(self):
        with tempfile.TemporaryDirectory() as td:
            b,s,a,_=fixture(Path(td)); validate_archive(a,s,b.coarse)
            for key in ('contains_policy_results','GT_or_CAD_inputs','signed_geometry_ready'):
                with self.assertRaises(ValueError):validate_coarse({**b.coarse,key:True},b.manifest_path)
            bad=dict(a);bad['frame_idx']=a['frame_idx'][::-1]
            with self.assertRaisesRegex(ValueError,'mismatch'):validate_archive(bad,s,b.coarse)
            bad=dict(a);bad['left_vertices']=a['left_vertices']+.01
            with self.assertRaisesRegex(ValueError,'MANO'):validate_archive(bad,s,b.coarse)
            bad=dict(a);bad['source_metric_eligible']=np.zeros(3,bool)
            with self.assertRaisesRegex(ValueError,'validity'):validate_archive(bad,s,b.coarse)
            b.array_path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):b.load()

    def test_zero_controlled_reward_full_export_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);b,s,a,models=fixture(root)
            cfg={'max_objects':1,'contact_variables_per_hand':5,'mano_models':models,
                 'preprocessed_state_adapter':ADAPTER,'reward_profile':'formal_annotation','reward_scale':1e-5}
            env=CoarseAnnotationEnv([b],cfg); obs,_=env.reset(seed=0)
            self.assertEqual(obs.shape,(480+EXTRA_FEATURES,))
            self.assertFalse(env.state.episode.objects['object:0']['signed_geometry_ready'])
            self.assertFalse(np.array_equal(env.state.episode.objects['object:0']['translation'],s['objects']['object:synthetic']['translation']))
            np.testing.assert_allclose(env.state.episode.objects['object:0']['vertices_canonical'],a['vertices_canonical'])
            np.testing.assert_allclose(env.state.object_rotation['object:0'],a['object_rotation'])
            src=env.state.episode.objects['object:0']['translation'].copy()
            expected=np.array(a['object_translation_world'])
            action=np.zeros(118,np.float32); rewards=[]
            for i in range(3):
                _,r,done,truncated,info=env.step(action);rewards.append(r)
            self.assertTrue(done);self.assertFalse(truncated)
            np.testing.assert_allclose(env.state.object_translation['object:0'],expected,atol=1e-10)
            self.assertAlmostEqual(sum(rewards),-1e-5*info['objective'],places=10)
            zero=info['objective'];export=env.export_annotation(root/'zero.json')
            self.assertEqual(export['document']['preprocessed_state_adapter']['name'],ADAPTER)
            with np.load(export['npz']) as z:np.testing.assert_array_equal(z['object_faces'],a['faces'])
            env.reset(seed=0);layout=env.state.episode.layout
            action[layout.block('object.object:0.translation').start]=.5
            action[layout.block('hand.left.articulation').start+3]=.1
            _,_,_,_,info=env.step(action)
            for i in range(2):_,_,_,_,info=env.step(np.zeros(118,np.float32))
            self.assertNotEqual(zero,info['objective'])
            self.assertTrue(any('rgb_reprojection' in k for k in info['factor_ledger']['terms']))
            np.testing.assert_array_equal(env.state.episode.objects['object:0']['translation'],src)
            # Object source priors are absent: a one-frame objective has no
            # relative-motion likelihood. Two endpoints must couple the action.
            from envs.dynamics import decode_candidate
            env.reset(seed=0);original=env.state.copy();changed=original.copy()
            physical=np.zeros(118,np.float32);physical[layout.block('object.object:0.translation').start]=.002
            decode_candidate(changed,physical,cfg)
            self.assertEqual(env.evaluator.objective(original,1).objective,env.evaluator.objective(changed,1).objective)
            self.assertNotEqual(env.evaluator.objective(original,3).objective,env.evaluator.objective(changed,3).objective)
            with self.assertRaises(ValueError):env.state.episode.objects['object:0']['translation'].setflags(write=True)

    def test_rgb_observation_summary_has_no_future_endpoint(self):
        with tempfile.TemporaryDirectory() as td:
            _,s,_,_=fixture(Path(td));rows,summary=source_rgb_rows(s,'object:synthetic')
            np.testing.assert_array_equal(summary[0],np.zeros(9))
            self.assertGreater(summary[1,6],0)
            self.assertEqual(len(rows),2)
            s['motion_preprocessing']=SimpleNamespace(qualification={'relative_edge_weights':(0.,),'reprojection_edge_weights':(0.,)})
            rows,summary=source_rgb_rows(s,'object:synthetic');self.assertFalse(rows)
            np.testing.assert_array_equal(summary,np.zeros((3,9)))

    def test_raw_environment_selection_requires_explicit_adapter(self):
        with self.assertRaisesRegex(ValueError,'explicit'):
            CoarseAnnotationEnv([],{'max_objects':1})


if __name__=='__main__':unittest.main()
