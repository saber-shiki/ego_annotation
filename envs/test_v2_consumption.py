"""Milestone2 v2 source → candidate → measurements → export, without PPO."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import sys
import numpy as np
from observations.inspection_fixture import build_fixture
from observations.test_inspection import rebind_manifest
from observations.interaction.adapter import assemble_interaction_evidence
from envs.v2_consumption import V2ConsumptionSession,CONTRACT,SCHEMA
from utils.state import sha


class V2ConsumptionTests(unittest.TestCase):
    def test_zero_control_full_native_source_baseline_and_export_not_old_policy_schema(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=build_fixture(root/'source');before={str(p):sha(p) for p in source.parent.rglob('*') if p.is_file()}
            with patch.dict(sys.modules,{'train':None,'rollout':None,'models':None,'skrl':None,'rewards.evaluator':None}):
                session=V2ConsumptionSession(source)
                self.assertEqual(session.initialization,'source_full_native')
                self.assertTrue(session.seed_available[0]);self.assertFalse(session.seed_available[1])
                self.assertFalse(session.source['objects'][session.entity]['pose_valid'][1])
                self.assertEqual(len(session.full_faces),1152)
                original=session.object_world(3).copy();hands=session.hand_vertices['left'].copy()
                for fid in session.ids:session.apply_control(int(fid))
                np.testing.assert_array_equal(session.object_world(3),original)
                self.assertIsNone(session.object_world(11))
                np.testing.assert_array_equal(session.hand_vertices['left'],hands)
                export=session.export(root/'zero')
                with np.load(export['npz'],allow_pickle=False) as z:
                    self.assertEqual(z['left_vertices_world'].shape,(2,778,3))
                    self.assertEqual(len(z['object_faces']),1152)
                    self.assertTrue(z['object_pose_valid'][0]);self.assertFalse(z['object_pose_valid'][1])
                    self.assertFalse(z['object_valid'][1])
                doc=json.loads(Path(export['json']).read_text());self.assertEqual(doc['schema'],SCHEMA)
                self.assertEqual(doc['candidate_initialization'],'source_full_native')
                self.assertFalse(doc['policy_executed']);self.assertFalse(doc['contract']['trainable'])
                self.assertTrue(all(sha(p)==h for p,h in before.items()))
                session.reset();self.assertEqual(session.history,[])
                np.testing.assert_array_equal(session.object_world(3),original)

    def test_rgb_seed_is_retained_as_explicit_comparison_only(self):
        with tempfile.TemporaryDirectory() as d:
            session=V2ConsumptionSession(build_fixture(Path(d)/'source'),initialization='rgb_seed_comparison')
            self.assertEqual(session.initialization,'rgb_seed_comparison')
            self.assertTrue(session.seed_available.all())
            self.assertEqual(session.initialization_role,'R4_derived_rgb_measurement_seed_comparison_only')

    def test_control_changes_real_geometry_rgb_residual_and_refreshed_pad_query(self):
        with tempfile.TemporaryDirectory() as d:
            source=build_fixture(Path(d)/'source');session=V2ConsumptionSession(source,initialization='rgb_seed_comparison')
            before=session.measure(11);geometry=session.object_world(11).copy()
            session.apply_control(11,object_rotation_rad=[0,.02,0],object_translation_m=[.005,0,0],left_translation_m=[0,.003,0])
            after=session.measure(11)
            self.assertGreater(np.max(np.abs(session.object_world(11)-geometry)),1e-3)
            self.assertGreater(after['hand_measurements']['left']['source_projection_delta_median_px'],0)
            self.assertNotEqual(before['RGB_edges'][0]['mean_squared_reprojection_px2'],after['RGB_edges'][0]['mean_squared_reprojection_px2'])
            self.assertNotEqual(before['hand_measurements']['left']['closest_targets_canonical'],after['hand_measurements']['left']['closest_targets_canonical'])
            self.assertEqual(before['hand_measurements']['left']['fixed_pad_vertex_ids'],after['hand_measurements']['left']['fixed_pad_vertex_ids'])
            self.assertFalse(after['metric_observation_eligible'])
            self.assertFalse(session.source['hands']['left']['vertices'].flags.writeable)

    def test_invalid_control_and_fixed_gauge_are_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            source=build_fixture(Path(d)/'source');session=V2ConsumptionSession(source,initialization='rgb_seed_comparison')
            before=session.object_rotation.copy()

            for fid,kwargs in [(3,{'object_translation_m':[.01,0,0]}),(11,{'right_translation_m':[.01,0,0]}),
                               (11,{'object_rotation_rad':[float('nan'),0,0]}),(4,{}),(11,{'object_translation_m':[1,0,0]})]:
                with self.assertRaises(ValueError):session.apply_control(fid,**kwargs)
                np.testing.assert_array_equal(session.object_rotation,before);self.assertEqual(session.revision,0)

    def test_source_visibility_controls_candidate_display_without_source_change(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=build_fixture(root/'source');m=json.loads(source.read_text())
            hand_path=source.parent/m['sources']['mano_archive']['path'];before=sha(hand_path)
            evidence_path=source.parent/'interaction.json';payload=json.loads(evidence_path.read_text())
            results={};receipts={}
            for entry in payload['source_judgments']:
                p=Path(entry['receipt']['path']);value=json.loads(p.read_text())
                if entry['judgment']=='visibility':
                    value['frames'][0]['visibility']['left']['state']='out_of_frame';p.write_text(json.dumps(value))
                results[entry['judgment']]=value;receipts[entry['judgment']]={'path':str(p),'sha256':sha(p)}
            evidence=assemble_interaction_evidence(object_id='object:synthetic',frame_ids=[3,11],
                batches=[{'frame_ids':[3,11],'judgments':results,'sources':receipts}])
            evidence_path.write_text(json.dumps(evidence));rebind_manifest(source,['interaction_judgments'])
            session=V2ConsumptionSession(source);output=session.export(root/'display')
            with np.load(output['npz'],allow_pickle=False) as z:
                self.assertTrue(z['left_source_valid'][0]);self.assertFalse(z['left_display_valid'][0])
                self.assertTrue(np.isfinite(z['left_vertices_world'][0]).all())
            self.assertEqual(sha(hand_path),before)

    def test_explicit_policy_compatibility_refuses_old_checkpoint_without_training(self):
        from observations.check_policy_compatibility import check_compatibility
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=build_fixture(root/'source');provenance=root/'agent.provenance.json'
            provenance.write_text(json.dumps({'schema':'ego.annotation.skrl_checkpoint/2.0.0','compatibility':{
                'dataset':{'bundle_schemas':['ego.annotation.observation_bundle.v1']},
                'observation_space':'Box(-inf, inf, (480,), float32)',
                'action_space':'Box(-inf, inf, (118,), float32)','environment_class':'envs.annotation_env:AnnotationEnv'}}))
            with patch.dict(sys.modules,{'train':None,'rollout':None,'models':None,'skrl':None}):
                result=check_compatibility(source,provenance)
            self.assertEqual(result['status'],'blocked_not_trainable')
            self.assertFalse(result['training_executed']);self.assertFalse(result['checkpoint_loaded'])
            self.assertEqual(result['unchanged_default_spaces'],{'observation':480,'actions':118})
            self.assertEqual([r['status'] for r in result['checks']],['refused','refused'])
            self.assertTrue(result['source_unchanged'])

    def test_training_contract_is_deliberately_incompatible(self):
        self.assertFalse(CONTRACT['trainable'])
        for key in ('policy_observation_layout','policy_action_layout','reward_profile'):
            self.assertIsNone(CONTRACT[key])


if __name__=='__main__':unittest.main()
