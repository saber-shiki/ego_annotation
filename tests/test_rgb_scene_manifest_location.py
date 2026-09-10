"""Source P01 manifest routing for nested candidate RGB optimization."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_rgb_scene_edges import resolve_raw_frame_manifest


class RGBManifestLocationTests(unittest.TestCase):
    def test_candidate_annotations_use_explicit_source_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            annotations = root / 'candidates/a/measurements/object_geometry/visible_geometry/tool/annotations.json'
            source = root / 'input/raw_frame_manifest/manifest.json'
            self.assertEqual(resolve_raw_frame_manifest(annotations, source), source.resolve())
            self.assertNotEqual(resolve_raw_frame_manifest(annotations, source),
                                root / 'candidates/a/input/raw_frame_manifest/manifest.json')

    def test_legacy_default_location_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            annotations = root / 'measurements/object_geometry/visible_geometry/tool/annotations.json'
            self.assertEqual(resolve_raw_frame_manifest(annotations),
                             root / 'input/raw_frame_manifest/manifest.json')


if __name__ == '__main__':
    unittest.main()
