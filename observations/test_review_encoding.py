"""CPU review encoding: no model calls, real failures are read-only diagnostics."""
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from utils.runtime import encode_astra_request, astra_stage_inputs
from observations.judgments.judge_pose_trajectory import (
    PROMPT, SCHEMA, MAX_REQUEST_BYTES, compact_trajectory_evidence, judge,
)


def fixture(root, count=9, high_entropy=False):
    from utils.data import RKEY, TKEY
    rows = [{'frame_idx': i, 'time_s': i/30, RKEY: np.eye(3).tolist(), TKEY: [0, 0, 1.],
             'metric_observation': i < 140, 'uncertainty': .2 if i < 140 else 1.,
             'relative_evidence_connected': i < 140, 'depth_feasibility_uncertain': i >= 140}
            for i in range(150)]
    sources = [{**r, 'status': 'direct'} for r in rows]
    path = root/'posterior.json'
    path.write_text(json.dumps({'object_id': 'fixture', 'pose_rows': rows, 'source_pose_rows': sources}))
    evidence = compact_trajectory_evidence(path)
    ids = list(range(count*3-3)) + [149] if count < 50 else list(range(150))
    evidence['review_frame_ids'] = ids
    evidence['review_images'] = []
    rng = np.random.default_rng(20260915)
    images = []
    for index in range(count):
        group = ids[index*3:index*3+3]
        size = (1242, 768) if high_entropy else (128*len(group), 384)
        data = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
        path = root/f'review_{index:03d}.jpg'
        Image.fromarray(data).save(path, quality=95)
        images.append(path)
        evidence['review_images'].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'frame_ids': group})
    return evidence, images


class AdaptiveReviewTests(unittest.TestCase):
    def assert_wire_binding(self, payload, audit, evidence, images):
        content = json.loads(payload)['input'][0]['content']
        recovered = json.loads(content[0]['text'].split('\n', 1)[1])
        self.assertEqual(recovered, evidence)
        self.assertEqual(recovered['frame_ids'], list(range(150)))
        self.assertEqual(recovered['timeline'][-1][0], 149)
        self.assertEqual(audit['effective_payload_sha256'], hashlib.sha256(payload).hexdigest())
        encoded = [c for c in content if c['type'] == 'input_image']
        self.assertEqual(len(encoded), len(images))
        self.assertEqual([idx for r in audit['ordered_images'] for idx in r['frame_ids']], evidence['review_frame_ids'])
        for record, part, image in zip(audit['ordered_images'], encoded, images):
            raw = base64.b64decode(part['image_url'].split(',', 1)[1])
            self.assertEqual(record['effective_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(record['original_sha256'], hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertEqual(record['effective_bytes'], len(raw))
            self.assertEqual(record['effective_dimensions_wh'], list(Image.open(io.BytesIO(raw)).size))

    def test_exact_2037316_boundary_next_tier_and_actual_judge_wire(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence, images = fixture(root)
            evidence['fixture_padding'] = ''
            initial, _ = encode_astra_request(PROMPT, SCHEMA, evidence, images)
            self.assertLess(len(initial), 2037316)
            evidence['fixture_padding'] = 'x' * (2037316-len(initial))
            initial, _ = encode_astra_request(PROMPT, SCHEMA, evidence, images)
            self.assertEqual(len(initial), 2037316)
            payload, audit = encode_astra_request(PROMPT, SCHEMA, evidence, images,
                max_request_bytes=MAX_REQUEST_BYTES, adaptive_review_images=True)
            self.assertEqual([a['encoding_level'] for a in audit['encoding_attempts']], ['original', 'jpeg_q80'])
            self.assertLessEqual(len(payload), MAX_REQUEST_BYTES)
            self.assert_wire_binding(payload, audit, evidence, images)
            plan = astra_stage_inputs(PROMPT, SCHEMA, evidence, images,
                max_request_bytes=MAX_REQUEST_BYTES, adaptive_review_images=True)
            self.assertEqual(plan['review_encoding'], audit['review_encoding_policy'])
            credential = root/'credential'; credential.write_text('fixture-only'); credential.chmod(0o600)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), patch('urllib.request.urlopen', side_effect=TimeoutError('CPU fixture')) as send:
                with self.assertRaises(TimeoutError): judge(evidence, images, audit_dir=root/'audit')
                self.assertEqual(send.call_count, 1)
                self.assertEqual(send.call_args.kwargs['timeout'], 120)
                self.assertEqual(send.call_args.args[0].data, payload)
                self.assertEqual((root/'audit/effective_payload.json').read_bytes(), payload)
                self.assertEqual(json.loads((root/'audit/request_manifest.json').read_text()), audit)
            for change in ('hash', 'order', 'coverage'):
                altered = copy.deepcopy(evidence)
                if change == 'hash': altered['review_images'][0]['sha256'] = '0'*64
                if change == 'order': altered['review_images'].reverse()
                if change == 'coverage': altered['review_images'][-1]['frame_ids'] = []
                with self.assertRaisesRegex(ValueError, 'review sheet'):
                    encode_astra_request(PROMPT, SCHEMA, altered, images,
                        max_request_bytes=MAX_REQUEST_BYTES, adaptive_review_images=True)

    def test_root_main_ask_executes_budgeted_judge_and_binds_policy(self):
        import ast
        import importlib
        from unittest.mock import Mock
        import main
        from utils.state import planned_stage_contract
        # Execute the actual nested main.ask, isolating only its stage/ledger
        # surroundings. No copied request logic or disconnected helper path.
        tree = ast.parse(Path(main.__file__).read_text())
        annotate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'annotate')
        ask = next(n for n in ast.walk(annotate) if isinstance(n, ast.FunctionDef) and n.name == 'ask')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence, images = fixture(root)
            evidence['fixture_padding'] = ''
            initial, _ = encode_astra_request(PROMPT, SCHEMA, evidence, images)
            evidence['fixture_padding'] = 'x' * (2037316-len(initial))
            recovery = Mock(); recovery.run.side_effect = lambda name, execute, **kw: execute()
            namespace = {'verify_code': lambda: None, 'importlib': importlib, 'ids': evidence['frame_ids'],
                'object_id': evidence['object_id'], 'planned_stage_contract': planned_stage_contract,
                'astra_stage_inputs': astra_stage_inputs, 'process_execution': {}, 'recovery': recovery,
                'ledger': Mock(), 'root': root, 'json': json}
            exec(compile(ast.Module(body=[ask], type_ignores=[]), main.__file__, 'exec'), namespace)
            result = {'object_id': 'fixture', 'trajectory_consistent': False, 'flagged_intervals': [], 'evidence_frames': [149]}
            raw = {'model': 'gpt-6-astra', 'status': 'completed', 'id': 'cpu-only', 'output': [
                {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': json.dumps(result)}]}]}
            class Response(io.BytesIO):
                headers = {'x-litellm-model-group': 'gpt-6-astra'}
            credential = root/'credential'; credential.write_text('fixture-only'); credential.chmod(0o600)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), patch('urllib.request.urlopen', return_value=Response(json.dumps(raw).encode())) as send:
                actual = namespace['ask']('pose_trajectory', evidence, images, root/'state/judgment.json')
            self.assertFalse(actual['trajectory_consistent'])
            self.assertEqual(actual['judgment_provenance']['encoding_attempts'][-1]['encoding_level'], 'jpeg_q80')
            self.assertLessEqual(len(send.call_args.args[0].data), MAX_REQUEST_BYTES)
            self.assertEqual(send.call_args.args[0].data, (root/'state/judgment.request/effective_payload.json').read_bytes())
            planned = recovery.run.call_args.kwargs['contract']
            self.assertIn('full_sheet_budget_v1', json.dumps(planned))
            self.assertIn('2000000', json.dumps(planned))

    def test_high_entropy_all_150_frames_exhausts_ladder_and_never_sends(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence, images = fixture(root, count=50, high_entropy=True)
            payload, audit = encode_astra_request(PROMPT, SCHEMA, evidence, images,
                max_request_bytes=MAX_REQUEST_BYTES, adaptive_review_images=True)
            self.assertFalse(audit['budget_satisfied'])
            self.assertEqual(len(audit['encoding_attempts']), 5)
            self.assertGreater(len(payload), MAX_REQUEST_BYTES)
            self.assert_wire_binding(payload, audit, evidence, images)
            self.assertEqual(audit['ordered_images'][0]['effective_dimensions_wh'], [576, 932])
            credential = root/'credential'; credential.write_text('fixture-only'); credential.chmod(0o600)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), patch('urllib.request.urlopen') as send:
                with self.assertRaisesRegex(ValueError, 'exceeds 2000000'):
                    judge(evidence, images, audit_dir=root/'audit')
                send.assert_not_called()
            self.assertEqual((root/'audit/effective_payload.json').read_bytes(), payload)
            self.assertEqual(json.loads((root/'audit/request_manifest.json').read_text()), audit)

    def test_small_payload_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence, images = fixture(Path(folder), count=1)
            expected, _ = encode_astra_request(PROMPT, SCHEMA, evidence, images)
            actual, audit = encode_astra_request(PROMPT, SCHEMA, evidence, images,
                max_request_bytes=MAX_REQUEST_BYTES, adaptive_review_images=True)
            self.assertEqual(actual, expected)
            self.assertEqual(len(audit['encoding_attempts']), 1)
            self.assertEqual(audit['ordered_images'][0]['original_sha256'], audit['ordered_images'][0]['effective_sha256'])

    @unittest.skipUnless(os.environ.get('EGO_CAN93_DIAGNOSTIC_ROOT'), 'optional read-only fixed93 can diagnostic')
    def test_real_can93_exact_failed_request_and_preserved_evidence(self):
        root = Path(os.environ['EGO_CAN93_DIAGNOSTIC_ROOT'])
        evidence = json.loads((root/'state/trajectory_review_evidence.json').read_text())
        # main.ask prefixes these keys before merging the compact evidence.
        evidence = {'frame_ids': evidence['frame_ids'], 'object_id': evidence['object_id'], **evidence}
        images = [Path(r['path']) for r in evidence['review_images']]
        original, _ = encode_astra_request(PROMPT, SCHEMA, evidence, images)
        self.assertEqual(len(original), 2037316)
        self.assertEqual(original, (root/'state/pose_trajectory_judgment.request/effective_payload.json').read_bytes())
        payload, audit = encode_astra_request(PROMPT, SCHEMA, evidence, images,
            max_request_bytes=MAX_REQUEST_BYTES, adaptive_review_images=True)
        self.assertEqual(len(images), 9)
        self.assertEqual(len(evidence['review_frame_ids']), 25)
        self.assertEqual(audit['encoding_attempts'][-1]['encoding_level'], 'jpeg_q80')
        self.assertLessEqual(len(payload), MAX_REQUEST_BYTES)
        self.assert_wire_binding(payload, audit, evidence, images)
        print(json.dumps({'diagnostic_only': True, 'original_bytes': len(original), 'effective_bytes': len(payload),
                          'effective_sha256': audit['effective_payload_sha256'], 'level': 'jpeg_q80',
                          'timeline_frames': len(evidence['timeline']), 'review_frames': len(evidence['review_frame_ids'])}))
