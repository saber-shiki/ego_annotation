"""CPU producer -> actual main gate/render/ask -> mock HTTP; no model verdict claims."""
import ast
import base64
import copy
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import main
from observations.object.owlv2 import CandidateIdentity
from observations.judgments import judge_mask_track as mask_judge
from utils.runtime import astra_stage_inputs, encode_astra_request
from utils.state import sha, load, planned_stage_contract


def main_mask_body():
    tree = ast.parse(Path(main.__file__).read_text())
    annotate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'annotate')
    block = next(n for n in ast.walk(annotate) if hasattr(n, 'body') and isinstance(n.body, list)
                 and any(isinstance(c, ast.FunctionDef) and c.name == '_stage_candidate_mask_identity' for c in n.body))
    start = next(i for i, n in enumerate(block.body) if isinstance(n, ast.FunctionDef) and n.name == '_stage_candidate_mask_identity')
    stop = next(i for i, n in enumerate(block.body[start:], start) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'base' for t in n.targets))
    ask = next(n for n in ast.walk(annotate) if isinstance(n, ast.FunctionDef) and n.name == 'ask')
    return compile(ast.Module(body=[ask, *block.body[start:stop]], type_ignores=[]), main.__file__, 'exec')


def fixture(root, uncertain=(138,)):
    for directory in ('state', 'raw', 'masks', 'detector/back-object'):
        (root/directory).mkdir(parents=True)
    obj = {'object_id': 'back-object', 'track_id': 'back-track',
           'description': 'Blue rear container; orange foreground container is a different instance'}
    def detection(box, score=.8):
        return {'bbox_xyxy': box, 'score': score, 'text_label': 'container'}
    prior = detection([100, 100, 200, 200])
    wide = detection([70, 100, 200, 200], .7)
    narrow = detection([125, 100, 200, 200], .9)
    identity = CandidateIdentity(); detections = []; frames = []; track = {}
    for frame in range(150):
        candidates = [wide, narrow] if frame in uncertain else [prior]
        chosen, association = identity.select(frame, candidates)
        if chosen is None:
            raise AssertionError(association)
        rgb = np.full((240, 260, 3), 255, np.uint8)
        rgb[100:200, 70 if frame in uncertain else 100:200] = [189, 121, 39]
        if frame in uncertain:
            rgb[100:200, 125:200] = [41, 130, 229]
        image = root/'raw'/f'{frame:06d}.png'; cv2.imwrite(str(image), rgb)
        mask = np.zeros((240, 260), np.uint8)
        x0, y0, x1, y1 = chosen['bbox_xyxy']; mask[y0:y1, x0:x1] = 255
        mp = root/'masks'/f'{frame:06d}.png'; cv2.imwrite(str(mp), mask)
        frames.append({'frame_idx': frame, 'raw_frame_path': str(image), 'source_width': 260, 'source_height': 240})
        detections.append({'frame_idx': frame, 'image_path': str(image), 'image_sha256': sha(image),
            'image_width': 260, 'image_height': 240, 'detections': copy.deepcopy(candidates),
            'selected_detection': chosen, 'association': association})
        track[str(frame)] = {'visible': True, 'mask_path': str(mp)}
    manifest = root/'manifest.json'; manifest.write_text(json.dumps({'frames': frames}))
    report = {'case_id': 'cpu-two-objects', 'object_id': obj['object_id'], 'track_id': obj['track_id'],
        'raw_frame_manifest_sha256': sha(manifest), 'detections': detections}
    (root/'detector/back-object/candidate_identity.json').write_text(json.dumps(report))
    track_path = root/'track.json'; track_path.write_text(json.dumps(track))
    sampled = np.linspace(0, 149, 12, dtype=int).tolist()
    recovery = Mock(); recovery.run.side_effect = lambda name, execute, **kw: execute()
    return {'verify_code': lambda: None, 'importlib': importlib, 'planned_stage_contract': planned_stage_contract,
        'astra_stage_inputs': astra_stage_inputs, 'process_execution': {}, 'recovery': recovery,
        'ledger': Mock(), 'root': root, 'ids': list(range(150)), 'object_id': obj['object_id'],
        'obj': obj, 'track': obj['track_id'], 'args': SimpleNamespace(case=report['case_id']),
        'detector': root/'detector', 'manifest': manifest, 'raw_by_id': {r['frame_idx']: r for r in frames},
        'frames': frames, 'sampled': sampled, 'track_path': track_path,
        'stage': lambda *a, **kw: lambda fn: fn(), 'load': load, 'record': Mock(),
        'json': json, 'Path': Path, 'np': np, 'cv2': cv2}


def wire_evidence(request):
    return json.loads(json.loads(request.data)['input'][0]['content'][0]['text'].split('\n', 1)[1])


def response(request, **kwargs):
    evidence = wire_evidence(request)
    # Deliberate CPU false only in the ambiguous batch, never a fabricated live verdict.
    result = {'object_id': evidence['object_id'], 'identity_consistent': 138 not in evidence['inspected_frame_ids'],
        'intervals': [{'frame_start': evidence['inspected_frame_ids'][0],
                       'frame_end': evidence['inspected_frame_ids'][-1], 'state': 'uncertain',
                       'rationale': 'CPU mock response only'}], 'uncertainties': ['CPU mock']}
    raw = {'model': 'gpt-6-astra', 'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant',
        'content': [{'type': 'output_text', 'text': json.dumps(result)}]}]}
    class Response(io.BytesIO):
        headers = {'x-litellm-model-group': 'gpt-6-astra'}
    return Response(json.dumps(raw).encode())


class MaskIdentityReviewTests(unittest.TestCase):
    def test_actual_main_uncertain_pixels_diagnostics_audit_and_false_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ns = fixture(root)
            credential = root/'credential'; credential.write_text('CPU fixture only'); credential.chmod(0o600)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), patch('urllib.request.urlopen', side_effect=response) as send:
                with self.assertRaisesRegex(ValueError, 'mask identity review rejected'):
                    exec(main_mask_body(), ns)
            batches = load(root/'state/mask_review_evidence.json')['batches']
            self.assertEqual(send.call_count, len(batches))
            first = wire_evidence(send.call_args_list[0].args[0])
            self.assertEqual(first['review_frame_ids'], [137, 138, 139])
            row = next(row for row in first['candidate_identity_checks']['frames'] if row['frame_idx'] == 138)
            self.assertEqual(row['mask_candidate_iou'], 1.)
            self.assertEqual(row['association_uncertainty'], 'occluded_extent_ambiguity')
            self.assertEqual(row['selected_detection']['bbox_xyxy'], [125, 100, 200, 200])
            self.assertEqual(row['competing_detections'][0]['detection']['bbox_xyxy'], [70, 100, 200, 200])
            self.assertAlmostEqual(row['raw_distinct_cost_margin'], .01080676184426832)
            for index, call in enumerate(send.call_args_list):
                audit = root/'state/mask_track_batches'/f'{index:04d}.request'
                self.assertEqual(call.args[0].data, (audit/'effective_payload.json').read_bytes())
                manifest = load(audit/'request_manifest.json')
                evidence = wire_evidence(call.args[0])
                self.assertEqual([f for image in manifest['ordered_images'] for f in image['frame_ids']], evidence['review_frame_ids'])
                self.assertEqual(manifest['request_model'], 'gpt-6-astra')
                self.assertTrue(manifest['disable_fallbacks'])
                self.assertLessEqual(len(call.args[0].data), 2_000_000)
            content = json.loads(send.call_args_list[0].args[0].data)['input'][0]['content']
            image = [part for part in content if part['type'] == 'input_image'][1]
            pixels = cv2.imdecode(np.frombuffer(base64.b64decode(image['image_url'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
            # Actual f138 raw front-object orange at left; green-tinted selected mask at right.
            np.testing.assert_allclose(pixels[186, 150], [41, 130, 229], atol=5)
            np.testing.assert_allclose(pixels[186, 410], [45, 165, 139], atol=5)
            self.assertTrue(np.any(pixels[:36] < 100))  # rendered frame/panel labels
            merged = load(root/'state/mask_track_judgment.json')
            self.assertFalse(merged['identity_consistent'])
            self.assertEqual(len(merged['intervals']), len(batches))
            self.assertEqual(len(merged['judgment_provenance']['batches']), len(batches))
            self.assertTrue(set(ns['sampled'] + [137, 138, 139]) <= set(merged['judgment_provenance']['required_frame_ids']))
            self.assertFalse((root/'state/base_annotations').exists())
            self.assertIn('full_sheet_budget_v1', json.dumps(ns['recovery'].run.call_args_list[0].kwargs['contract']))

    def test_no_uncertainty_preserves_single_12_image_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ns = fixture(root, uncertain=())
            credential = root/'credential'; credential.write_text('CPU'); credential.chmod(0o600)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), patch('urllib.request.urlopen', side_effect=response) as send:
                exec(main_mask_body(), ns)
            self.assertEqual(send.call_count, 1)
            evidence = wire_evidence(send.call_args.args[0])
            self.assertEqual(evidence['inspected_frame_ids'], ns['sampled'])
            self.assertNotIn('mask_review_batch', evidence)
            self.assertFalse((root/'state/mask_review_evidence.json').exists())
            images = [root/'renders/mask_review'/f'{frame:06d}.jpg' for frame in ns['sampled']]
            expected, _ = encode_astra_request(mask_judge.PROMPT, mask_judge.SCHEMA, evidence, images)
            self.assertEqual(send.call_args.args[0].data, expected)
            self.assertNotIn('review_encoding', json.dumps(ns['recovery'].run.call_args.kwargs['contract']))

    def test_actual_mask_ask_resume_binds_policy_mapping_diagnostics_and_run(self):
        from utils.state import RunState, StageRecovery
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); source = base/'source'; source.mkdir(); ns = fixture(source)
            credential = base/'credential'; credential.write_text('CPU'); credential.chmod(0o600)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), patch('urllib.request.urlopen', side_effect=response):
                with self.assertRaisesRegex(ValueError, 'mask identity review rejected'):
                    exec(main_mask_body(), ns)
            batch = load(source/'state/mask_review_evidence.json')['batches'][0]
            images = [Path(image['path']) for image in batch['review_images']]
            root = base/'run'; (root/'input').mkdir(parents=True)
            raw = root/'input/raw.mp4'; raw.write_bytes(b'CPU raw identity')
            contract = root/'input/runtime_input_contract.json'
            contract.write_text(json.dumps({'run_id': 'cpu-mask-run', 'run_root': str(root),
                'input_video': str(raw), 'input_video_sha256': sha(raw)}))
            ledger = RunState(root)
            identity = {'run_id': 'cpu-mask-run', 'agent_model': 'gpt-6-astra', 'code': 'fixed-cpu-source',
                'config': 'fixed-cpu-config', 'contract': sha(contract)}
            ns.update(root=root, ledger=ledger)
            path = root/'state/mask.json'
            ns['recovery'] = StageRecovery(ledger, identity)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), patch('urllib.request.urlopen', side_effect=response) as send:
                first = ns['ask']('mask_track', batch, images, path)
                self.assertEqual(send.call_count, 1)
            ns['recovery'].close()
            receipts = ledger.registry.read_bytes()
            ns['recovery'] = StageRecovery(ledger, identity, resume=True)
            with patch('urllib.request.urlopen') as send:
                self.assertEqual(first, ns['ask']('mask_track', batch, images, path))
                send.assert_not_called()
            ns['recovery'].close()
            for change in ('diagnostic', 'mapping', 'order', 'bytes', 'policy'):
                evidence = copy.deepcopy(batch); ordered = list(images)
                original = images[0].read_bytes()
                if change == 'diagnostic':
                    evidence['candidate_identity_checks']['frames'][1]['raw_distinct_cost_margin'] += .001
                elif change == 'mapping':
                    evidence['review_images'][0]['frame_ids'] = [138]
                elif change == 'order':
                    ordered.reverse()
                elif change == 'bytes':
                    images[0].write_bytes(original + b'changed')
                ns['recovery'] = StageRecovery(ledger, identity, resume=True)
                with patch.object(mask_judge, 'MAX_REQUEST_BYTES', 1_999_999 if change == 'policy' else 2_000_000), patch('urllib.request.urlopen') as send:
                    with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'planned stage contract changed'):
                        ns['ask']('mask_track', evidence, ordered, path)
                    send.assert_not_called()
                ns['recovery'].close(); images[0].write_bytes(original)
                self.assertEqual(ledger.registry.read_bytes(), receipts)
            with self.assertRaises(ValueError):
                StageRecovery(ledger, {**identity, 'run_id': 'another-run'}, resume=True)

    def test_many_uncertain_frames_all_reach_actual_http_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); uncertain = list(range(8, 149, 10)); ns = fixture(root, uncertain)
            credential = root/'credential'; credential.write_text('CPU'); credential.chmod(0o600)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), patch('urllib.request.urlopen', side_effect=response) as send:
                with self.assertRaisesRegex(ValueError, 'mask identity review rejected'):
                    exec(main_mask_body(), ns)
            evidences = [wire_evidence(call.args[0]) for call in send.call_args_list]
            covered = {frame for evidence in evidences for frame in evidence['review_frame_ids']}
            required = set(ns['sampled']) | {f for frame in uncertain for f in (frame-1, frame, frame+1)}
            self.assertEqual(covered, required)
            for frame in uncertain:
                self.assertTrue(any({frame-1, frame, frame+1} <= set(evidence['review_frame_ids']) for evidence in evidences))
            self.assertTrue(all(len(call.args[0].data) <= 2_000_000 for call in send.call_args_list))
            merged = load(root/'state/mask_track_judgment.json')
            self.assertEqual(len(merged['judgment_provenance']['batches']), len(evidences))

    def test_many_uncertain_frames_complete_context_and_coverage(self):
        ids = list(range(150)); uncertain = list(range(0, 150, 2))
        checks = {'frames': [{'frame_idx': frame, 'association_uncertainty': 'occluded_extent_ambiguity'} for frame in uncertain]}
        groups = mask_judge.uncertain_review_groups(checks, ids, [0, 149])
        self.assertEqual(set(range(150)), {frame for group in groups for frame in group})
        self.assertTrue(all(len(group) <= 6 for group in groups))
        for frame in uncertain:
            self.assertTrue(any(set(ids[max(0, frame-1):frame+2]) <= set(group) for group in groups))


if __name__ == '__main__':
    unittest.main()
