"""Bounded source-data regressions; no providers, real MANO or training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from data.bundle import EpisodeBundle, file_sha256, validate_manifest
from data.immutable import freeze, frozen_array
from data.timeline import join_annotation_rows, timeline_ids
from data.test_rl_data import make_synthetic_bundle
import observations.test_manifest as manifest_fixtures
from observations.manifest import write_source_observation_manifest
from observations.contracts import SCHEMA_V2, validate_v2


class ObservationDataTests(unittest.TestCase):
    def test_frozen_data_stays_json_compatible(self):
        value = freeze({'identity': {'path': 'x'}, 'rows': [{'a': 1}]})
        self.assertEqual(json.loads(json.dumps(value)), {'identity': {'path': 'x'}, 'rows': [{'a': 1}]})
        with self.assertRaises(TypeError): value['identity']['path'] = 'other'
        with self.assertRaises(TypeError): value.update({'other': 1})
        self.assertIs(copy.deepcopy(value), value)
        a = frozen_array([1., 2.])
        with self.assertRaises(ValueError): a.setflags(write=True)
        with self.assertRaises(ValueError): frozen_array([object()], object)

    def test_timeline_join_does_not_invent_frames(self):
        manifest = {'frame_range': [3, 19], 'frame_ids': [3, 8, 19], 'time_s': [0., .1, .4]}
        rows = [{'frame_idx': i, 'time_s': t} for i, t in [(19,.4),(3,0.),(8,.1)]]
        ids, times, selected = join_annotation_rows(manifest, rows)
        self.assertEqual(ids.tolist(), [3, 8, 19])
        self.assertEqual([r['frame_idx'] for r in selected], ids.tolist())
        with self.assertRaisesRegex(ValueError, 'unique'): join_annotation_rows(manifest, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, 'timestamps'): join_annotation_rows({**manifest,'time_s':[0.,.2,.4]}, rows)
        with self.assertRaises(ValueError): timeline_ids([True, 2])
        with self.assertRaises(ValueError): timeline_ids([3, 2])

    def test_sparse_bundle_load_and_index(self):
        from data.build_index import build_index
        with tempfile.TemporaryDirectory() as temp:
            path, _ = make_synthetic_bundle(Path(temp))
            m = json.loads(path.read_text()); ids = [3,8,19]
            ann = Path(m['sources']['annotations']['path']); p = json.loads(ann.read_text())
            for i,row in enumerate(p['frames']): row['frame_idx'] = ids[i]
            p['frames'].reverse(); ann.write_text(json.dumps(p))
            m['sources']['annotations']['sha256'] = file_sha256(ann)
            mano = Path(m['sources']['mano_archive']['path'])
            with np.load(mano, allow_pickle=False) as z: arrays = dict(z)
            arrays['frame_idx'] = np.asarray(ids); np.savez(mano, **arrays)
            m['sources']['mano_archive']['sha256'] = file_sha256(mano)
            m.update(frame_range=[3,19], frame_ids=ids, time_s=[0.,1/30,2/30])
            path.write_text(json.dumps(m))
            b = EpisodeBundle.open(path); data = b.load()
            self.assertEqual(data['frame_idx'].tolist(), ids)
            self.assertEqual(build_index([path])['episodes'][0]['frame_count'], 3)
            with self.assertRaises(ValueError): data['hands']['left']['vertices'].setflags(write=True)
            # Existing env export serializes identity using dict() + json.dumps.
            json.dumps(dict(data['identity']))

    def test_v2_roles_and_capabilities_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'source.json'
            m = write_source_observation_manifest(output, run_root=root, case='c', object_id='o', track_id='t',
                frame_ids=[3,9],time_s=[0.,.2],source_timing={},
                artifacts=manifest_fixtures.SourceObservationManifestTests()._artifacts(root), bundle_version=2)
            self.assertEqual(m['schema'], SCHEMA_V2)
            self.assertEqual(m['observation_roles']['visible_annotations']['field_roles']['reconstructed_geometry_pose'],'R4')
            self.assertFalse(m['observation_roles']['conditioned_depth']['accuracy_certified'])
            validate_manifest(EpisodeBundle.open(output).manifest)
            bad = copy.deepcopy(m);bad['capabilities'].append('rgb_motion')
            with self.assertRaisesRegex(ValueError, 'missing'): validate_v2(bad)
            bad = copy.deepcopy(m);bad['observation_roles']['conditioned_depth']['accuracy_certified'] = True
            with self.assertRaisesRegex(ValueError, 'roles'): validate_v2(bad)
            from data.build_index import build_index
            with self.assertRaisesRegex(ValueError,'training adapter'): build_index([output])
            from data.dataset import EpisodeDataset
            from data.build_index import INDEX_SCHEMA
            dataset = EpisodeDataset({'schema':INDEX_SCHEMA,'episodes':[{'source_video_id':'fixture',
                'manifest':str(output),'manifest_sha256':file_sha256(output)}]})
            with self.assertRaisesRegex(ValueError,'training adapter'): dataset[0]

    def test_camera_time_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = make_synthetic_bundle(Path(temp))
            m = json.loads(path.read_text());m['frame_ids']=[0,1,2];m['time_s']=[0.,.1,.2];path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'timestamps'): EpisodeBundle.open(path).load()


if __name__ == '__main__': unittest.main()
