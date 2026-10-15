"""Evidence barrier tests; no GT assets or evaluator imports used."""
import json
from pathlib import Path
import tempfile
import unittest
from tools.surface_experiments.evaluate_frozen import verify_freeze, sha


class FreezeTests(unittest.TestCase):
    def test_evaluation_refuses_running_or_changed_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'case'; path.mkdir(); (path/'renders').mkdir()
            (root/'batch.exit').write_text('1\n')
            with self.assertRaises(ValueError):
                verify_freeze(root)
            arms = {}
            for arm in ('A', 'B', 'C'):
                d = path/arm; d.mkdir(); (d/'coarse_scene.npz').write_bytes(b'immutable')
                (d/'coarse_scene.json').write_text(json.dumps({'prediction_side_only': True, 'GT_or_CAD_inputs': False}))
                arms[arm] = {'npz_sha256': sha(d/'coarse_scene.npz'), 'manifest_sha256': sha(d/'coarse_scene.json')}
            (path/'completed.json').write_text('{}')
            (root/'inputs.json').write_text(json.dumps([{'case': 'case', 'output': str(path)}]))
            (root/'prediction_freeze.json').write_text(json.dumps({'cases': [{'case': 'case', 'completed_sha256': sha(path/'completed.json'), 'arms': arms, 'render_sha256': {}}]}))
            (root/'batch.exit').write_text('0\n')
            verify_freeze(root)
            (path/'B'/'coarse_scene.npz').write_bytes(b'mutated')
            with self.assertRaises(ValueError):
                verify_freeze(root)


if __name__ == '__main__':
    unittest.main()
