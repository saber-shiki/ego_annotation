"""Acceptance tests: real v2 reader/preprocessing/CLI, synthetic inputs, no PPO."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from observations.inspection_fixture import build_fixture
from observations.inspect import inspect_bundle
from observations.contracts import classification
from observations.interaction.adapter import assemble_interaction_evidence
from data.bundle import EpisodeBundle
from utils.state import sha


def rebind_manifest(path, names):
    payload=json.loads(path.read_text())
    old={name:payload['artifacts'][name]['sha256'] for name in names}
    def replace(value):
        if isinstance(value,dict):
            for name,digest in old.items():
                if value.get('sha256')==digest and value.get('path')==payload['artifacts'][name]['path']:
                    value['sha256']=sha(path.parent/value['path'])
            for child in value.values():replace(child)
        elif isinstance(value,list):
            for child in value:replace(child)
    replace(payload)
    payload['observation_roles']=classification(payload['artifacts'])
    path.write_text(json.dumps(payload,indent=2)+'\n')


class ObservationInspectionTests(unittest.TestCase):
    def test_full_readback_consumes_values_and_outputs_geometry_without_ppo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=build_fixture(root/'source');out=root/'review'
            before={str(p):sha(p) for p in source.parent.rglob('*') if p.is_file()}
            forbidden={'train':None,'rollout':None,'models':None,'skrl':None,'gymnasium':None,
                'envs.annotation_env':None,'legacy_optimize':None,'render':None,'utils.candidate_geometry':None}
            with patch.dict(sys.modules,forbidden),patch('utils.runtime.run_model',side_effect=AssertionError('no providers')):
                report=inspect_bundle(source,out)
            self.assertTrue(report['source_unchanged']);self.assertTrue(report['synthetic_fixture'])
            self.assertEqual(report['status'],'complete_with_measurement_gaps')
            self.assertEqual(len(report['groups']),11)
            self.assertEqual(report['groups']['rgb_frames']['summary']['decoded_frames'],2)
            raster=report['groups']['depth_masks_surface']['summary']
            self.assertEqual(raster['depth_frames_decoded'],1);self.assertEqual(raster['depth_missing_ids'],[11])
            self.assertEqual(raster['P09_retained_frames_replayed'],1)
            self.assertFalse(report['groups']['formal_factors']['summary']['legacy_contact_patch_prepared'])
            self.assertGreater(report['groups']['formal_factors']['summary']['source_first_surface_query_samples'],0)
            self.assertEqual(report['groups']['formal_factors']['summary']['factor_families']['visible_ownership'],2)
            self.assertEqual(report['groups']['rgb_motion']['summary']['edge_count'],1)
            self.assertTrue(report['groups']['visual_geometry']['summary']['seed_binding_performed'])
            self.assertLess(report['groups']['visual_geometry']['summary']['canonical_world_roundtrip_max_m'],1e-8)
            counts=json.loads((out/'review_geometry_counts.json').read_text())
            self.assertEqual(counts['3']['complete_native_at_anchor']['faces_total'],1152)
            self.assertEqual(counts['3']['complete_native_at_anchor']['faces_projectable'],1152)
            with np.load(out/'arrays/interaction_timeline.npz',allow_pickle=False) as z:
                self.assertEqual(z['primary_declared'].shape,(2,2,5))
                self.assertEqual(z['visibility'].shape,(2,3))
            with np.load(out/'arrays/rgb_edge_000000.npz',allow_pickle=False) as z:
                self.assertEqual(z['source_world'].shape,(30,3));self.assertLess(np.max(z['residual_matcher_norm_px']),1e-3)
            with (out/'consumer_matrix.csv').open(encoding='utf-8-sig') as handle:
                matrix=list(csv.DictReader(handle))
            self.assertTrue(all(row['new_v2_policy_connected']=='False' for row in matrix))
            self.assertTrue(all(row['not_connected_downstream'] for row in matrix))
            self.assertTrue(all(sha(p)==digest for p,digest in before.items()))
            self.assertTrue(all((out/p).is_file() for p in report['review_images']))
            with self.assertRaises(FileExistsError):inspect_bundle(source,out)
            with self.assertRaisesRegex(ValueError,'outside'):inspect_bundle(source,source.parent/'forbidden_output')

    def test_observation_loader_does_not_prepare_legacy_contact_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=build_fixture(Path(tmp)/'source');bundle=EpisodeBundle.open(source)
            with patch.dict(sys.modules,{'utils.candidate_geometry':None}):
                data=bundle.load_observations()
                self.assertIs(data,bundle.load_observations())
                self.assertFalse(data['legacy_contact_patch_prepared'])
                self.assertIsNotNone(data['appearance_evidence'])

    def test_missing_capability_is_incomplete_not_silent_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=build_fixture(root/'source');payload=json.loads(source.read_text())
            payload['capabilities'].remove('rgb_motion');payload['extensions'].pop('rgb_source_measurements')
            for key in ('rgb_source_npz','rgb_source_report'):
                payload['sources'].pop(key);payload['artifacts'].pop(key)
            payload['observation_roles']=classification(payload['artifacts']);source.write_text(json.dumps(payload))
            result=inspect_bundle(source,root/'review')
            self.assertEqual(result['status'],'incomplete_missing_capabilities');self.assertIn('rgb_motion',result['missing'])
            self.assertEqual(result['groups'],{})
            run=subprocess.run([sys.executable,'-B','-m','observations.inspect','--source-observations',str(source),
                '--output-dir',str(root/'cli')],capture_output=True,text=True)
            self.assertEqual(run.returncode,2,run.stderr)

    def test_empty_measurement_is_retained_without_identity_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=build_fixture(root/'source',empty_rgb=True)
            result=inspect_bundle(source,root/'review')
            self.assertFalse(result['missing']);self.assertEqual(result['groups']['rgb_motion']['summary']['edge_count'],0)
            self.assertTrue(any(w['kind']=='measured_empty_rgb_edges' for w in result['warnings']))
            seed=json.loads((root/'review/measurement_seed.json').read_text())
            self.assertFalse(seed['frames'][1]['available'])
            self.assertIsNone(seed['frames'][1]['rotation_world_from_canonical'])
            self.assertFalse(list((root/'review/arrays').glob('rgb_edge_*.npz')))

    def test_corrupt_lazy_nonreview_mask_and_bound_npz_fail_with_evidence(self):
        for defect in ('appearance','depth'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=build_fixture(root/'source');manifest=json.loads(source.read_text())
                if defect=='appearance':
                    path=source.parent/'appearance_11.png';path.write_bytes(b'not an image')
                else:
                    path=source.parent/manifest['artifacts']['conditioned_depth']['path'];path.write_bytes(b'not an npz')
                with self.assertRaises((ValueError,FileNotFoundError)):
                    inspect_bundle(source,root/'review',review_frames=[3])
                failure=json.loads((root/'review/failure.json').read_text())
                self.assertTrue(failure['partial_outputs_are_not_acceptance'])
                self.assertFalse((root/'review/inspection.json').exists())
                if defect=='appearance':self.assertEqual(failure['phase'],'appearance')

    def test_visibility_source_change_reaches_preprocessing_not_source_mano(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=build_fixture(root/'source');m=json.loads(source.read_text())
            mano=source.parent/m['sources']['mano_archive']['path'];before=sha(mano)
            first=inspect_bundle(source,root/'first')
            payload_path=source.parent/'interaction.json';payload=json.loads(payload_path.read_text())
            visibility_path=source.parent/'judgments/visibility.json';visibility=json.loads(visibility_path.read_text())
            visibility['frames'][0]['visibility']['left']['state']='out_of_frame';visibility_path.write_text(json.dumps(visibility))
            results={};receipts={}
            for entry in payload['source_judgments']:
                p=Path(entry['receipt']['path']);results[entry['judgment']]=json.loads(p.read_text())
                receipts[entry['judgment']]={'path':str(p),'sha256':sha(p)}
            value=assemble_interaction_evidence(object_id='object:synthetic',frame_ids=[3,11],
                batches=[{'frame_ids':[3,11],'judgments':results,'sources':receipts}])
            payload_path.write_text(json.dumps(value));rebind_manifest(source,['interaction_judgments'])
            second=inspect_bundle(source,root/'second')
            with np.load(root/'first/arrays/interaction_timeline.npz',allow_pickle=False) as a,np.load(root/'second/arrays/interaction_timeline.npz',allow_pickle=False) as b:
                self.assertTrue(a['display_observation_eligible'][0,1]);self.assertFalse(b['display_observation_eligible'][0,1])
                self.assertTrue(a['primary_visual_eligible'][0,0].any());self.assertFalse(b['primary_visual_eligible'][0,0].any())
            self.assertEqual(sha(mano),before)
            self.assertFalse(first['training_or_policy_executed']);self.assertFalse(second['training_or_policy_executed'])

    def test_depth_stream_normalizes_float16_like_the_P09_consumer(self):
        from observations.inspection_io import stream_depth
        from observations.object.surface import array_sha256
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'depth.npz';depth=np.arange(24,dtype=np.float16).reshape(2,3,4)/10+1
            np.savez(path,frame_idx=[3,11],depth=depth,confidence=np.ones_like(depth),
                source_size=[4,3],intrinsics_fx_fy_cx_cy=[[4.,4.,1.,1.],[4.,4.,1.,1.]])
            rows=list(stream_depth(path));self.assertEqual(len(rows),2)
            for i,(fid,d,c,K) in enumerate(rows):
                self.assertEqual(d.dtype,np.float32);self.assertEqual(c.dtype,np.float32)
                self.assertEqual(array_sha256(d),array_sha256(depth[i].astype(np.float32)))
            np.savez(path,frame_idx=[3,11],depth=depth[:1],confidence=np.ones_like(depth),
                source_size=[4,3],intrinsics_fx_fy_cx_cy=[[4.,4.,1.,1.],[4.,4.,1.,1.]])
            with self.assertRaisesRegex(ValueError,'frame count'):list(stream_depth(path))

    def test_foreign_review_frame_fails_before_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=build_fixture(root/'source')
            with self.assertRaisesRegex(ValueError,'review IDs'):inspect_bundle(source,root/'review',[4])
            failure=json.loads((root/'review/failure.json').read_text())
            self.assertEqual(failure['status'],'inspection_failed')

    def test_cli_complete_and_v1_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=build_fixture(root/'source')
            result=subprocess.run([sys.executable,'-B','-m','observations.inspect','--source-observations',str(source),
                '--output-dir',str(root/'review')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('complete_with_measurement_gaps',result.stdout)
            old=source.parent/'synthetic_hand_fixture/synthetic-a.manifest.json'
            result=subprocess.run([sys.executable,'-B','-m','observations.inspect','--source-observations',str(old),
                '--output-dir',str(root/'old')],capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            failure=json.loads((root/'old/failure.json').read_text())
            self.assertIn('no silent promotion',failure['message'])


if __name__=='__main__':unittest.main()
