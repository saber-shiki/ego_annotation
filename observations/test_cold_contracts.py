"""Cold-run contract regressions using small, provenance-labelled actual fixtures."""
import copy
import json
from pathlib import Path
import unittest

FIXTURES = Path(__file__).with_name('fixtures')

class RetainedAppearanceTests(unittest.TestCase):
    def test_can_105_rejected_106_retained_by_producer_and_options(self):
        from envs.qualification.object_observation import retained_metric_alignment
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
        import tempfile
        import numpy as np
        import cv2
        with tempfile.TemporaryDirectory() as folder:
            for box, expected in [(data['frame149_wrong_mask_box'], 'failed'), (selected['bbox_xyxy'], 'ok')]:
                mask = np.zeros((960,960), np.uint8)
                x0,y0,x1,y1=map(round,box);mask[y0:y1,x0:x1]=255
                path=Path(folder)/'mask.png';cv2.imwrite(str(path),mask)
                track={'149':{'visible':True,'mask_path':str(path),'bbox_xyxy':selected['bbox_xyxy']}}
                # A lying cached bbox must not hide a switch in the actual mask.
                self.assertEqual(check_candidate_mask_identity(report,track,[960,960])['status'],expected)

    def test_actual_mug_137_to_149_nested_extent_continues_without_candidate_rewrite(self):
        from observations.object.owlv2 import CandidateIdentity
        data = json.loads((FIXTURES / 'mug_occluded_extent.json').read_text())
        original = copy.deepcopy(data)
        identity = CandidateIdentity()
        states = {}
        for row in data['frames']:
            selected, diag = identity.select(row['frame_idx'], row['detections'])
            self.assertIsNotNone(selected, row['frame_idx'])
            self.assertFalse(identity.failed)
            self.assertEqual(len(diag['candidates']), len(row['detections']))
            self.assertIs(selected, row['detections'][diag['selected_candidate_index']])
            states[row['frame_idx']] = diag
        self.assertEqual(data, original)
        self.assertEqual(states[138]['selected_candidate_index'], 1)
        self.assertEqual(states[138]['state'], 'occluded_extent_ambiguity')
        self.assertEqual(states[138]['association_uncertainty'], 'occluded_extent_ambiguity')
        self.assertLess(states[138]['raw_distinct_cost_margin'], .12)
        self.assertEqual(states[138]['extent_alternatives'][0]['candidate_index'], 0)
        self.assertEqual(states[149]['selected_candidate_index'], 1)

    def test_nested_extent_needs_track_and_three_stable_boundaries(self):
        from observations.object.owlv2 import CandidateIdentity
        prior = {'bbox_xyxy': [100, 100, 200, 200], 'score': .8}
        narrow = {'bbox_xyxy': [125, 100, 200, 200], 'score': .7}
        wide = {'bbox_xyxy': [70, 100, 200, 200], 'score': .9}
        identity = CandidateIdentity(); identity.select(0, [prior])
        selected, diag = identity.select(1, [wide, narrow])
        self.assertIsNotNone(selected)
        self.assertEqual(diag['association_uncertainty'], 'occluded_extent_ambiguity')
        # Nesting without established history cannot resolve equal class scores.
        fresh = CandidateIdentity()
        self.assertIsNone(fresh.select(0, [{**wide, 'score': .8}, {**narrow, 'score': .8}])[0])
        self.assertTrue(fresh.failed)
        # Similar-area translated objects on opposite sides are competing tracks,
        # even though both individually pass the old continuity gate.
        identity = CandidateIdentity(); identity.select(0, [prior])
        left = {'bbox_xyxy': [80, 100, 180, 200], 'score': .8}
        right = {'bbox_xyxy': [120, 100, 220, 200], 'score': .8}
        selected, diag = identity.select(1, [left, right])
        self.assertIsNone(selected)
        self.assertEqual(diag['state'], 'identity_ambiguous')
        self.assertTrue(all(c['allowed'] for c in diag['candidates']))
        self.assertEqual(diag['extent_alternatives'], [])
        # Nested shapes after a gap or displaced from the prior's shared edges
        # must not acquire the occlusion exception.
        from observations.object.owlv2 import occluded_extent_alternative
        metrics = [{'allowed': True, 'iou': .6, 'center_distance_box_diagonals': .1}]*2
        self.assertFalse(occluded_extent_alternative(narrow['bbox_xyxy'], wide['bbox_xyxy'], prior['bbox_xyxy'], 2, metrics)['supported'])
        shifted = [[v+50 for v in d['bbox_xyxy']] for d in (narrow, wide)]
        self.assertFalse(occluded_extent_alternative(*shifted, prior['bbox_xyxy'], 1, metrics)['supported'])

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

class CompactTrajectoryTests(unittest.TestCase):
    def test_full_timeline_unknown_tail_and_request_byte_audit(self):
        import tempfile, hashlib, os
        import numpy as np
        from PIL import Image
        from unittest.mock import patch
        from utils.data import RKEY, TKEY
        from utils.runtime import encode_astra_request, astra_request
        from observations.judgments.judge_pose_trajectory import compact_trajectory_evidence, validate_compact
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); posterior = root/'posterior.json'
            rows = [{'frame_idx': i, 'time_s': i/30, RKEY: np.eye(3).tolist(), TKEY: [0, 0, 1+i*.01],
                     'metric_observation': i < 8, 'uncertainty': .2 if i < 8 else 1.,
                     'relative_evidence_connected': i < 8, 'depth_feasibility_uncertain': i >= 8} for i in range(10)]
            sources = [{**r, 'status': 'direct', 'rotation_observability': {'observable': True, 'score': .2}} for r in rows[:8]]
            sources += [{'frame_idx': i, 'status': 'missing_accepted_metric_surface'} for i in (8, 9)]
            posterior.write_text(json.dumps({'object_id': 'o', 'pose_rows': rows, 'source_pose_rows': sources, 'solver': {'trace': 'unused'*10000}}))
            evidence = compact_trajectory_evidence(posterior)
            self.assertEqual(evidence['frame_ids'], list(range(10)))
            self.assertIn(9, evidence['review_frame_ids'])
            self.assertIsNone(evidence['timeline'][-1][11]); self.assertIsNone(evidence['timeline'][-1][-1])
            self.assertEqual(evidence['source_posterior']['sha256'], hashlib.sha256(posterior.read_bytes()).hexdigest())
            broken = copy.deepcopy(evidence); broken['timeline'].pop()
            with self.assertRaises(ValueError): validate_compact(broken)
            image = root/'raw.png'; Image.new('RGB', (32,32)).save(image)
            payload, receipt = encode_astra_request('prompt', {'type':'object'}, evidence, [image])
            self.assertEqual(receipt['effective_payload_sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(receipt['ordered_images'][0]['sha256'], hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertTrue(json.loads(payload)['disable_fallbacks'])
            credential=root/'credential'; credential.write_text('test-only'); credential.chmod(0o600)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), patch('urllib.request.urlopen', side_effect=TimeoutError('test')) as call:
                audit=root/'request'
                with self.assertRaises(TimeoutError): astra_request('prompt', {'type':'object'}, evidence, [image], audit_dir=audit)
                call.assert_called_once()
                self.assertEqual(call.call_args.args[0].data, (audit/'effective_payload.json').read_bytes())
                self.assertEqual(call.call_args.args[0].data, payload)
                self.assertFalse(json.loads((audit/'transport_failure.json').read_text())['retry'])
