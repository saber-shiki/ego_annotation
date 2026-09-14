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

class CandidateIdentityTests(unittest.TestCase):
    def test_actual_mug_149_preserves_148_instance_over_higher_category_score(self):
        from observations.object.owlv2 import CandidateIdentity, box_iou, check_candidate_mask_identity
        data = json.loads((FIXTURES / 'mug_identity_149.json').read_text())
        identity = CandidateIdentity()
        identity.select(148, [{'bbox_xyxy': data['frame148_box'], 'score': 1.}])
        selected, diag = identity.select(149, data['frame149_detections'])
        self.assertEqual(diag['selected_candidate_index'], 1)
        self.assertAlmostEqual(selected['score'], .33478835225105286)
        self.assertGreater(data['frame149_detections'][0]['score'], selected['score'])
        self.assertGreater(box_iou(data['frame148_box'], selected['bbox_xyxy']), .4)
        report = {'case_id': 'fixture', 'object_id': 'object_1', 'track_id': 'track_1', 'detections': [
            {'frame_idx': 149, 'image_width': 960, 'image_height': 960, 'selected_detection': selected, 'association': diag}]}
        wrong = {'149': {'visible': True, 'bbox_xyxy': data['frame149_wrong_mask_box']}}
        self.assertEqual(check_candidate_mask_identity(report, wrong, [960, 960])['status'], 'failed')
        correct = {'149': {'visible': True, 'bbox_xyxy': selected['bbox_xyxy']}}
        self.assertEqual(check_candidate_mask_identity(report, correct, [960, 960])['status'], 'ok')

    def test_occlusion_cannot_reseed_other_instance_and_ambiguity_fails(self):
        from observations.object.owlv2 import CandidateIdentity
        target = {'bbox_xyxy': [10, 10, 110, 110], 'score': .8}
        other = {'bbox_xyxy': [300, 300, 350, 350], 'score': .99}
        identity = CandidateIdentity(); identity.select(0, [target])
        self.assertIsNone(identity.select(1, [other])[0])
        self.assertEqual(identity.select(2, [target, other])[0], target)
        for frame in range(3, 7): identity.select(frame, [other])
        self.assertTrue(identity.failed)
        self.assertIsNone(identity.select(7, [target])[0])
        ambiguous = CandidateIdentity()
        self.assertIsNone(ambiguous.select(0, [target, {**other, 'score': .8}])[0])
        self.assertTrue(ambiguous.failed)
