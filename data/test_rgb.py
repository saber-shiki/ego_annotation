"""RGB extension consumer contracts with sparse IDs and synthetic MANO arrays."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from data.bundle import EpisodeBundle, validate_manifest, write_manifest
from data.build_index import build_index
from data.dataset import EpisodeDataset
from data.test_rl_data import make_synthetic_bundle
from observations.object.rgb_evidence import EXTENSION, SCHEMA
from observations.test_rgb_source import source_fixture, run_fixture
from utils.state import sha


def rgb_bundle(root, *, empty=False):
    fixture = source_fixture(root/'source')
    run_fixture(fixture, matches=not empty)
    path, _ = make_synthetic_bundle(root/'prepared', frame_count=2)
    manifest = json.loads(path.read_text())
    mano = Path(manifest['sources']['mano_archive']['path'])
    with np.load(mano, allow_pickle=False) as z: arrays = dict(z)
    arrays['frame_idx'] = np.array(fixture['ids'])
    arrays['left_valid'][:] = 0; arrays['right_valid'][:] = 0
    np.savez(mano, **arrays)
    manifest['sources']['mano_archive']['sha256'] = sha(mano)
    manifest['sources'].pop('contact_factors')
    files = {'annotations': fixture['inputs']['annotations'],
        'conditioned_depth': fixture['inputs']['depth_npz'], 'raw_frames': fixture['inputs']['raw_frame_manifest'],
        'surface_report': fixture['inputs']['p09_adapter_report'],
        'rgb_source_npz': fixture['root']/'rgb.npz', 'rgb_source_report': fixture['root']/'rgb.json'}
    for name, file in files.items():
        manifest['sources'][name] = {'path': str(file), 'sha256': sha(file)}
    manifest['sources']['rgb_source_npz']['role'] = 'rgb_source_arrays'
    manifest['sources']['rgb_source_report']['role'] = 'rgb_source_report'
    manifest['frame_ids'] = fixture['ids']; manifest['frame_range'] = fixture['ids']
    manifest['time_s'] = fixture['times']; manifest['extensions'] = {EXTENSION: SCHEMA}
    write_manifest(path, manifest)
    return path, manifest


class RGBBundleTests(unittest.TestCase):
    def test_sparse_dataset_loads_real_typed_measurements_and_empty_results(self):
        for empty in (True, False):
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as directory:
                path, _ = rgb_bundle(Path(directory), empty=empty)
                index = build_index([path]); self.assertEqual(index['episodes'][0]['frame_count'], 2)
                loaded = EpisodeDataset(index)[0].load()
                np.testing.assert_array_equal(loaded['frame_idx'], [3, 11])
                self.assertEqual(len(loaded['rgb_evidence'].edges), 0 if empty else 1)
                self.assertFalse(loaded['objects']['object:synthetic']['pose_valid'].any())
                if not empty:
                    edge = loaded['rgb_evidence'].edges[0]
                    with self.assertRaises(ValueError): edge.target_camera.setflags(write=True)
                    with self.assertRaises(TypeError): loaded['rgb_evidence'].report['rows'][0]['status'] = 'changed'

    def test_unextended_v1_has_explicit_missing_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = make_synthetic_bundle(Path(directory), frame_count=2)
            self.assertIsNone(EpisodeBundle.open(path).load()['rgb_evidence'])

    def test_extension_pair_version_and_roles_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = rgb_bundle(Path(directory))
            for mutation in ('version', 'role', 'pair', 'input'):
                with self.subTest(mutation=mutation):
                    value = json.loads(json.dumps(manifest))
                    if mutation == 'version': value.pop('extensions')
                    if mutation == 'role': value['sources']['rgb_source_npz']['role'] = 'rgb_edges'
                    if mutation == 'pair': value['sources'].pop('rgb_source_report')
                    if mutation == 'input': value['sources'].pop('surface_report')
                    with self.assertRaises(ValueError): validate_manifest(value)
            bundle = EpisodeBundle.open(path)
            # Tampering after open must still fail at the actual consumer boundary.
            report = Path(manifest['sources']['rgb_source_report']['path'])
            report.write_text(report.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'RGB source artifact changed'): bundle.load()


if __name__ == '__main__':
    unittest.main()
