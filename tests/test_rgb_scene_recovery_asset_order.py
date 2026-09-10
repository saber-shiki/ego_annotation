import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rgb_scene_recovery import validate_transition


class RerunInventoryOrderTests(unittest.TestCase):
    def manifests(self):
        assets = {'files': {'/rr/a.py': {'sha256': 'a'}, '/rr/b.so': {'sha256': 'b'}},
                  'trees': {}, 'scene_rrd_files': ['/rr/a.py', '/rr/b.so']}
        old = {'files': {'scripts/shared.py': 'unchanged'}, 'external_assets': assets}
        new = copy.deepcopy(old)
        new['files'].update({'scripts/build_v19_visible_geometry_from_sam2_depth.py': 'new-p09',
                             'scripts/rgb_scene_depth_order.py': 'new-depth',
                             'scripts/rgb_scene_metric_eligibility.py': 'new-eligibility',
                             'scripts/rgb_scene_recovery.py': 'new-recovery'})
        new['external_assets']['scene_rrd_files'].reverse()
        return old, new

    def check(self, old, new):
        return validate_transition(old, new, {}, {}, Path('/old'), Path('/new'))

    def test_inventory_order_only_is_equivalent_without_mutating_inputs(self):
        old, new = self.manifests()
        before = copy.deepcopy((old, new))
        self.assertEqual(self.check(old, new),
                         ['scripts/build_v19_visible_geometry_from_sam2_depth.py',
                          'scripts/rgb_scene_depth_order.py',
                          'scripts/rgb_scene_metric_eligibility.py',
                          'scripts/rgb_scene_recovery.py'])
        self.assertEqual((old, new), before)

    def test_changed_hash_membership_and_malformed_inventory_still_fail(self):
        for mutation in ('hash', 'missing', 'duplicate', 'dict'):
            with self.subTest(mutation=mutation):
                old, new = self.manifests()
                assets = new['external_assets']
                if mutation == 'hash':
                    assets['files']['/rr/a.py']['sha256'] = 'different'
                elif mutation == 'missing':
                    assets['scene_rrd_files'].pop()
                elif mutation == 'duplicate':
                    assets['scene_rrd_files'].append(assets['scene_rrd_files'][0])
                else:
                    assets['scene_rrd_files'] = dict.fromkeys(assets['scene_rrd_files'])
                with self.assertRaises(ValueError):
                    self.check(old, new)


if __name__ == '__main__':
    unittest.main()
