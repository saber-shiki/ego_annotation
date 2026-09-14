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
