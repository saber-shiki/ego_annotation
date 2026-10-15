import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from observations.artifact_cache import digest, inventory, retain, verify, verify_tree


class CacheTests(unittest.TestCase):
    def test_retention_and_tamper_detection(self):
        with TemporaryDirectory() as d:
            root = Path(d); run = root/'run'; run.mkdir()
            bundle = run/'source_observations.json'; bundle.write_text('{}')
            (run/'model.npy').write_bytes(b'measurement bytes')
            prep = root/'prep'; prep.mkdir()
            (prep/'arrays.npz').write_bytes(b'preprocessing bytes')
            (prep/'inspection.json').write_text(json.dumps({
                'source_manifest_sha256':digest(bundle), 'source_unchanged':True,
                'status':'complete_preprocessing_readback','missing':[]}))
            with patch('data.bundle.EpisodeBundle.open', return_value=SimpleNamespace(
                    manifest={'schema':'synthetic_test','identity':{}})):
                receipt = retain(bundle, root/'cache', prep)
                self.assertEqual(len(receipt['run_files']),2)
                self.assertTrue(verify(root/'cache')['verified'])
                with self.assertRaises(FileExistsError): retain(bundle, root/'cache')
                (root/'cache'/'preprocessing'/'arrays.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'inventory changed'): verify(root/'cache')
                (root/'cache'/'run.tar').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'archive hash'): verify(root/'cache')

    def test_changed_source_and_symlink_rejected(self):
        with TemporaryDirectory() as d:
            root=Path(d); (root/'a').write_bytes(b'one'); rows=inventory(root)
            (root/'a').write_bytes(b'two')
            with self.assertRaisesRegex(ValueError,'inventory changed'): verify_tree(root,rows)
            (root/'link').symlink_to(root/'a')
            with self.assertRaisesRegex(ValueError,'symlink'): inventory(root)

    def test_bad_preprocessing_and_inside_output_rejected(self):
        with TemporaryDirectory() as d:
            root=Path(d); run=root/'run'; run.mkdir(); bundle=run/'source_observations.json';bundle.write_text('{}')
            with self.assertRaisesRegex(ValueError,'outside source'): retain(bundle,run/'cache')
            prep=root/'prep';prep.mkdir()
            (prep/'inspection.json').write_text(json.dumps({'source_manifest_sha256':'wrong'}))
            with patch('data.bundle.EpisodeBundle.open', return_value=SimpleNamespace(manifest={})):
                with self.assertRaisesRegex(ValueError,'exact bundle'): retain(bundle,root/'cache',prep)
            self.assertFalse((root/'cache').exists())


if __name__=='__main__': unittest.main()
