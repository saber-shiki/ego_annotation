"""Cold-run contract regressions using small, provenance-labelled actual fixtures."""
import copy
import json
from pathlib import Path
import unittest

FIXTURES = Path(__file__).with_name('fixtures')

class RetainedAppearanceTests(unittest.TestCase):
    def test_can_105_rejected_106_retained_by_producer_and_options(self):
        from gates.object.observation import retained_metric_alignment
        from observations.object.shape import metric_alignment
        from main import appearance_candidate_ids
        data = json.loads((FIXTURES / 'can_retained_metric.json').read_text())
        bad, good = data['frames']
        self.assertIsNone(retained_metric_alignment(bad['objects'][0], bad)[0])
        self.assertIsNone(metric_alignment(bad['objects'][0], bad))
        self.assertIsNotNone(retained_metric_alignment(good['objects'][0], good)[0])
        self.assertIsNotNone(metric_alignment(good['objects'][0], good))
        rows = {i: {'frame_idx': i, 'status': 'available', 'score': 1., 'anchor_eligible': True} for i in (105, 106)}
        self.assertEqual(appearance_candidate_ids(rows, data, 'object_1', 106), [106])
        with self.assertRaises(ValueError):
            appearance_candidate_ids(rows, data, 'object_1', 105)
        for change in ('missing_flag', 'failed_ownership', 'bad_camera'):
            frame = copy.deepcopy(good); obj = frame['objects'][0]
            if change == 'missing_flag': obj.pop('rigid_pose_observation_eligible')
            if change == 'failed_ownership': obj['visible_geometry_candidate']['object_surface_ownership_filter'] = {'fail_closed': True}
            if change == 'bad_camera': obj['visible_geometry_candidate']['world_vertices_sample_m'][0][0] += 1.
            self.assertIsNone(metric_alignment(obj, frame), change)

    def test_judge_does_not_send_ineligible_choices(self):
        from observations.judgments.judge_appearance_anchor import judge
        from unittest.mock import patch
        with patch('observations.judgments.judge_appearance_anchor.astra_request') as request:
            with self.assertRaises(ValueError):
                judge({'frame_ids': [105], 'appearance': {'frames': [{'frame_idx': 105, 'anchor_eligible': False}]}}, [])
            request.assert_not_called()
