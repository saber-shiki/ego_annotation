"""CPU backend compaction/full-final-review transport tests; no live judgments."""
import ast
import copy
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import main
from observations.judgments import judge_backend_geometry as backend
from observations.judgments import judge_final_inspection as final
from utils.data import RKEY, TKEY
from utils.runtime import astra_stage_inputs, encode_astra_request
from utils.state import planned_stage_contract, sha


SELECTED = [0, 13, 27, 40, 54, 67, 81, 94, 108, 121, 135, 149]


def fixture(root, candidate=True):
    records = []
    for i in range(150):
        path = root / f'{i:06d}.png'
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
        records.append({'frame_idx': i, 'path': str(path), 'sha256': sha(path)})
    mesh = root / 'mesh.ply'; mesh.write_text('synthetic byte identity only')
    prior = root / 'prior.json'; prior.write_text('{}')
    ref = {'path': str(prior), 'sha256': sha(prior), 'bytes': 2, 'irrelevant_nested': {'trace': 'DO_NOT_SEND'}}
    rows = [{'frame_idx': i, 'time_s': i/30, 'object_id': 'item', RKEY: np.eye(3).tolist(),
             TKEY: [i/1000, 0, 1], 'metric_observation': True,
             'rotation_observability': {'observable': False, 'score': .1, 'surfels': ['DO_NOT_SEND']*100},
             'uncertainty': {'heavy_diagnostics': ['DO_NOT_SEND']*100},
             'surface_candidate': {'direct_surface_observation': True, 'trace': 'DO_NOT_SEND'},
             'solver_trace': 'DO_NOT_SEND'*1000} for i in range(150)]
    receipt = {'output_mesh_path': str(mesh), 'output_mesh_sha256': sha(mesh),
               'topology': {'vertex_count': 100, 'face_count': 200, 'face_index_sha256': 'f'*64,
                            'faces_exactly_preserved': True, 'faces': ['DO_NOT_SEND']*100},
               'native_qc': ref, 'static_image_scale': {'uniform_scale': 1.1, 'pivot_canonical_m': [0, 0, 0]},
               'trace': 'DO_NOT_SEND'}
    binding = {'schema': 'object_surface_candidate_binding_v1' if candidate else 'rgb_scene_visual_refined_binding_v1',
               'object_id': 'item', 'pose_rows': copy.deepcopy(rows), 'source_pose_rows': copy.deepcopy(rows),
               'source_prior': ref, 'refined_posterior': ref, 'appearance_mask_manifest': ref,
               'visual_mesh_receipt': receipt,
               'render_mesh_contract': {'required_mesh_path': str(mesh), 'required_mesh_sha256': sha(mesh)},
               'physical_authority': False}
    if candidate:
        binding.update(source_refined_pose_rows=copy.deepcopy(rows), source_visual_mesh_receipt=receipt,
                       source_binding=ref, shared_uniform_scale=1.1, scale_pivot_canonical_m=[0, 0, 0],
                       generated_mesh_used_as_residual_model=True, profile='surface_no_box')
    path = root / 'binding.json'; path.write_text(json.dumps(binding))
    selected = [records[i] for i in SELECTED]
    evidence = backend.compact_backend_evidence(path, object_id='item', backends=['sam3d'],
        frame_ids=list(range(150)), review_frame_ids=SELECTED, rendered_frames=selected)
    credential = root / 'credential'; credential.write_text('synthetic-only'); credential.chmod(0o600)
    return path, records, evidence, credential


def response_value(name):
    if name == 'backend_geometry':
        return {'object_id': 'item', 'backends': [{'backend': 'sam3d', 'visual_quality': 'poor',
                'rationale': 'synthetic defect', 'uncertainty': 'synthetic occlusion'}]}
    return {'object_id': 'item', 'visual_result': 'rejected', 'evidence_frames': [0],
            'defects': [{'frame_start': 0, 'frame_end': 0, 'rationale': 'synthetic defect'}],
            'physical_authority_granted': False, 'uncertainties': ['synthetic']}


def response(value):
    def send(request, **kwargs):
        raw = {'model': 'gpt-6-astra', 'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant',
            'content': [{'type': 'output_text', 'text': json.dumps(value)}]}]}
        class Response(io.BytesIO):
            headers = {'x-litellm-model-group': 'gpt-6-astra'}
        return Response(json.dumps(raw).encode())
    return send


def nested_ask(root, ids):
    tree = ast.parse(Path(main.__file__).read_text())
    annotate = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'annotate')
    ask = next(x for x in ast.walk(annotate) if isinstance(x, ast.FunctionDef) and x.name == 'ask')
    recovery = Mock(); recovery.run.side_effect = lambda name, execute, **kwargs: execute()
    ns = {'verify_code': lambda: None, 'importlib': importlib, 'ids': ids, 'object_id': 'item',
          'planned_stage_contract': planned_stage_contract, 'astra_stage_inputs': astra_stage_inputs,
          'process_execution': {}, 'recovery': recovery, 'ledger': Mock(), 'root': root, 'json': json}
    exec(compile(ast.Module(body=[ask], type_ignores=[]), main.__file__, 'exec'), ns)
    return ns['ask'], recovery


class BackendReviewBudgetTests(unittest.TestCase):
    def test_allowlist_large_binding_source_roles_selected_geometry_and_no_source_writes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path, records, e, _ = fixture(root)
            before = {p: sha(p) for p in root.iterdir()}
            text = json.dumps(e)
            self.assertGreater(path.stat().st_size, 3_000_000)
            self.assertLess(len(text.encode()), 65536)
            self.assertNotIn('DO_NOT_SEND', text)
            self.assertNotIn('solver_trace', text)
            self.assertNotIn('binding', e)
            self.assertEqual(e['binding_file'], {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size})
            self.assertEqual(e['frame_ids'], list(range(150)))
            self.assertEqual(e['review_frame_ids'], SELECTED)
            self.assertEqual(e['shared_uniform_scale'], 1.1)
            self.assertEqual(e['topology']['face_count'], 200)
            source = json.loads(path.read_text())
            for key in ('pose_rows', 'source_refined_pose_rows', 'source_pose_rows'):
                self.assertEqual([r[TKEY] for r in e['selected_binding_poses'][key]['selected_rows']],
                                 [source[key][i][TKEY] for i in SELECTED])
            self.assertIn('candidate', e['selected_binding_poses']['pose_rows']['role'])
            self.assertIn('posterior', e['selected_binding_poses']['source_refined_pose_rows']['role'])
            images = [Path(records[i]['path']) for i in SELECTED]
            self.assertEqual(backend.review_evidence(e, images), e)
            self.assertEqual(backend.review_evidence(backend.review_evidence(e, images), images), e)
            self.assertEqual(before, {p: sha(p) for p in root.iterdir()})
            print('synthetic backend: binding_bytes=%d evidence_bytes=%d images=12 source_frames=150' % (path.stat().st_size, len(text.encode())))

    def test_legacy_binding_supported_without_candidate_relabeling(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, e, _ = fixture(Path(td), candidate=False)
            self.assertEqual(e['binding_schema'], 'rgb_scene_visual_refined_binding_v1')
            self.assertNotIn('source_refined_pose_rows', e['selected_binding_poses'])
            self.assertEqual(e['selected_binding_poses']['pose_rows']['role'], 'original_observed_RGB_refined_posterior')
            self.assertEqual(e['scale_pivot_canonical_m'], [0, 0, 0])

    def test_actual_nested_ask_exact_planned_wire_original_frames_and_negative_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, records, e, credential = fixture(root)
            ask, recovery = nested_ask(root, list(range(150)))
            for name, evidence, chosen in (
                ('backend_geometry', e, [records[i] for i in SELECTED]),
                ('final_inspection', {'object_id': 'item', 'frame_ids': list(range(12)), 'rendered_frames': records[:12]}, records[:12])):
                with self.subTest(name=name), patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), \
                        patch('utils.runtime._astra_response', side_effect=response(response_value(name))) as send:
                    result = ask(name, evidence, [Path(r['path']) for r in chosen], root / (name+'.json'))
                self.assertEqual(send.call_count, 1)
                self.assertEqual(send.call_args.kwargs['timeout'], 120)
                wire = send.call_args.args[0].data
                audit_dir = root / (name+'.request')
                self.assertEqual(wire, (audit_dir/'effective_payload.json').read_bytes())
                planned = recovery.run.call_args.kwargs['contract']['inputs']
                actual = json.loads(json.loads(wire)['input'][0]['content'][0]['text'].split('\n', 1)[1])
                self.assertEqual(planned['evidence'], actual)
                self.assertEqual(planned['review_encoding']['max_request_bytes'], 3_000_000)
                self.assertLessEqual(len(wire), 3_000_000)
                audit = json.loads((audit_dir/'request_manifest.json').read_text())
                self.assertEqual([r['frame_ids'][0] for r in audit['ordered_images']], [r['frame_idx'] for r in chosen])
                for original, sent in zip(chosen, audit['ordered_images']):
                    self.assertEqual(sent['original_sha256'], sent['effective_sha256'])
                    self.assertEqual(sent['original_sha256'], original['sha256'])
                if name == 'backend_geometry': self.assertEqual(result['backends'][0]['visual_quality'], 'poor')
                else:
                    self.assertEqual(result['visual_result'], 'rejected')
                    self.assertFalse(result['physical_authority_granted'])

    def test_actual_main_tail_preserves_twelve_and_all_150_final_frames(self):
        tree = ast.parse(Path(main.__file__).read_text())
        annotate = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'annotate')
        body = next(x for x in annotate.body if isinstance(x, ast.Try)).body
        start = next(i for i, x in enumerate(body) if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'review_images' for t in x.targets))
        end = next(i for i, x in enumerate(body) if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'output' for t in x.targets))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path, records, _, credential = fixture(root)
            ask, recovery = nested_ask(root, list(range(150)))
            calls = []
            def send(request, **kwargs):
                evidence = json.loads(json.loads(request.data)['input'][0]['content'][0]['text'].split('\n', 1)[1])
                calls.append(evidence)
                name = 'backend_geometry' if 'backends' in evidence else 'final_inspection'
                value = response_value(name)
                if name == 'final_inspection':
                    value['evidence_frames'] = evidence['frame_ids']
                    value['defects'] = []
                return response(value)(request, **kwargs)
            ns = dict(root=root, binding=path, reviews=records, ids=list(range(150)), object_id='item',
                      np=np, Path=Path, ask=ask)
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), \
                    patch('utils.runtime._astra_response', side_effect=send):
                exec(compile(ast.Module(body=body[start-1:end], type_ignores=[]), main.__file__, 'exec'), ns)
            self.assertEqual(len(calls), 14)
            self.assertEqual(calls[0]['review_frame_ids'], SELECTED)
            self.assertEqual([i for e in calls[1:] for i in e['review_frame_ids']], list(range(150)))
            self.assertEqual([r['path'] for e in calls[1:] for r in e['review_images']], [r['path'] for r in records])
            self.assertEqual(len(ns['inspections']), 13)
            self.assertTrue(all(not r['physical_authority_granted'] for r in ns['inspections']))

    def test_both_cli_entry_seams_use_audit_and_real_encoding(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, records, e, credential = fixture(root)
            for name, evidence, chosen in (
                ('backend_geometry', e, [records[i] for i in SELECTED]),
                ('final_inspection', {'object_id': 'item', 'frame_ids': list(range(12)), 'rendered_frames': records[:12]}, records[:12])):
                ev = root / (name+'_evidence.json'); ev.write_text(json.dumps(evidence))
                for seam in ('main', 'module'):
                    output = root / (name+'_'+seam+'.json')
                    args = ['--evidence', str(ev), '--output', str(output)]
                    for row in chosen: args.extend(['--image', row['path']])
                    with self.subTest(name=name, seam=seam), patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), \
                            patch('utils.runtime._astra_response', side_effect=response(response_value(name))) as send:
                        if seam == 'main': main.main(['judge', name, *args])
                        else:
                            with patch('sys.argv', ['judge_'+name, *args]):
                                runpy.run_path(str(Path(main.__file__).parent/'observations/judgments'/('judge_'+name+'.py')), run_name='__main__')
                    self.assertTrue(output.exists())
                    self.assertEqual(send.call_count, 1)
                    self.assertTrue((output.with_suffix('.request')/'effective_payload.json').exists())

    def test_request_ladder_keeps_all_images_when_original_exceeds_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, records, e, credential = fixture(root)
            for module, evidence, chosen in (
                (backend, e, [records[i] for i in SELECTED]),
                (final, {'object_id': 'item', 'frame_ids': list(range(12)), 'rendered_frames': records[:12]}, records[:12])):
                images = [Path(r['path']) for r in chosen]
                bound = module.review_evidence(evidence, images)
                original, _ = encode_astra_request(module.PROMPT, module.SCHEMA, bound, images)
                audit_dir = root / module.__name__
                name = 'backend_geometry' if module is backend else 'final_inspection'
                with patch.object(module, 'MAX_REQUEST_BYTES', len(original)-1), \
                        patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), \
                        patch('utils.runtime._astra_response', side_effect=response(response_value(name))) as send:
                    module.judge(evidence, images, audit_dir=audit_dir)
                audit = json.loads((audit_dir/'request_manifest.json').read_text())
                self.assertGreater(len(audit['encoding_attempts']), 1)
                self.assertLess(len(send.call_args.args[0].data), len(original))
                self.assertEqual([r['frame_ids'][0] for r in audit['ordered_images']], [r['frame_idx'] for r in chosen])
                self.assertEqual(len(audit['ordered_images']), 12)

    def test_budget_exhaustion_missing_audit_and_timeout_no_retry_exact_wire(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, records, e, credential = fixture(root)
            for module, evidence, chosen in (
                (backend, e, [records[i] for i in SELECTED]),
                (final, {'object_id': 'item', 'frame_ids': list(range(12)), 'rendered_frames': records[:12]}, records[:12])):
                images = [Path(r['path']) for r in chosen]
                with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), \
                        patch('utils.runtime._astra_response') as send:
                    with self.assertRaisesRegex(ValueError, 'audit directory'): module.judge(evidence, images)
                    exhausted = root / (module.__name__+'_exhausted')
                    with patch.object(module, 'MAX_REQUEST_BYTES', 1):
                        with self.assertRaisesRegex(ValueError, 'exceeds 1'): module.judge(evidence, images, audit_dir=exhausted)
                    audit = json.loads((exhausted/'request_manifest.json').read_text())
                    self.assertFalse(audit['budget_satisfied']); self.assertEqual(len(audit['encoding_attempts']), 5)
                    send.assert_not_called()
                    send.side_effect = TimeoutError('synthetic read timeout')
                    failed = root / (module.__name__+'_timeout')
                    with self.assertRaises(TimeoutError): module.judge(evidence, images, audit_dir=failed)
                    self.assertEqual(send.call_count, 1)
                    self.assertEqual(send.call_args.kwargs['timeout'], 120)
                    self.assertEqual(send.call_args.args[0].data, (failed/'effective_payload.json').read_bytes())
                    failure = json.loads((failed/'transport_failure.json').read_text())
                    self.assertEqual(failure['error_type'], 'TimeoutError'); self.assertFalse(failure['retry'])
                    self.assertEqual(failure['request_sha256'], sha(failed/'effective_payload.json'))
                    self.assertTrue((failed/'connection_attempts.json').exists())

    def test_stale_binding_and_selected_geometry_tampering_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path, records, e, _ = fixture(root)
            images = [Path(records[i]['path']) for i in SELECTED]
            mutations = [('object', lambda b: b.update(object_id='other')),
                ('geometry', lambda b: b['selected_binding_poses']['pose_rows']['selected_rows'][0][TKEY].__setitem__(0, 99)),
                ('frame', lambda b: b['review_images'][0].update(frame_ids=[1])),
                ('hash', lambda b: b['binding_file'].update(sha256='0'*64)),
                ('coverage', lambda b: b['frame_ids'].pop())]
            with patch('utils.runtime._astra_response') as send:
                for label, mutate in mutations:
                    bad = copy.deepcopy(e); mutate(bad)
                    with self.subTest(label=label), self.assertRaises(ValueError): backend.judge(bad, images, audit_dir=root/label)
                with self.assertRaises(ValueError): backend.review_evidence(e, images[::-1])
                path.write_text(path.read_text()+' ')
                with self.assertRaisesRegex(ValueError, 'hash changed'): backend.review_evidence(e, images)
                send.assert_not_called()

    def test_builder_rejects_wrong_source_mapping_missing_unknown_and_duplicate_backends(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path, records, _, _ = fixture(root)
            args = dict(object_id='item', backends=['sam3d'], frame_ids=list(range(150)),
                        review_frame_ids=SELECTED, rendered_frames=[records[i] for i in SELECTED])
            for change in ({'backends': []}, {'backends': ['unknown']}, {'backends': ['sam3d', 'sam3d']},
                           {'object_id': 'other'}, {'frame_ids': list(range(149))}, {'review_frame_ids': [151]},
                           {'rendered_frames': [records[i] for i in SELECTED][::-1]}):
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    backend.compact_backend_evidence(path, **{**args, **change})
            bad = json.loads(path.read_text()); bad['pose_rows'][0]['object_id'] = 'other'; path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'pose object'): backend.compact_backend_evidence(path, **args)

    def test_oversized_allowlisted_text_and_stale_provenance_mesh_or_image_refuse(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path, records, e, _ = fixture(root)
            images = [Path(records[i]['path']) for i in SELECTED]
            args = dict(object_id='item', backends=['sam3d'], frame_ids=list(range(150)),
                        review_frame_ids=SELECTED, rendered_frames=[records[i] for i in SELECTED])
            original = path.read_bytes()
            bad = json.loads(original); bad['pose_rows'][0]['status'] = 'x'*65536
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, '64KiB'): backend.compact_backend_evidence(path, **args)
            path.write_bytes(original)
            with patch('utils.runtime._astra_response') as send:
                for file in (root/'mesh.ply', root/'prior.json', images[0]):
                    before = file.read_bytes(); file.write_bytes(before+b' ')
                    with self.subTest(file=file.name), self.assertRaisesRegex(ValueError, 'hash changed'):
                        backend.judge(e, images, audit_dir=root/'bad')
                    file.write_bytes(before)
                bad = copy.deepcopy(e); bad.pop('backends')
                with self.assertRaises(KeyError): backend.judge(bad, images, audit_dir=root/'bad')
                with self.assertRaisesRegex(ValueError, 'compact_backend_evidence'):
                    backend.judge({'binding': json.loads(original)}, images, audit_dir=root/'bad')
                path.unlink()
                with self.assertRaises(FileNotFoundError): backend.judge(e, images, audit_dir=root/'bad')
                send.assert_not_called()

    def test_backend_response_exactly_once_requested_and_same_object(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, records, e, credential = fixture(root)
            images = [Path(records[i]['path']) for i in SELECTED]
            good = response_value('backend_geometry')
            cases = [dict(good, backends=[]), dict(good, backends=good['backends']*2),
                     dict(good, backends=[dict(good['backends'][0], backend='trellis')]), dict(good, object_id='other')]
            for i, value in enumerate(cases):
                with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), \
                        patch('utils.runtime._astra_response', side_effect=response(value)), self.assertRaises(ValueError):
                    backend.judge(e, images, audit_dir=root/str(i))

            e = backend.compact_backend_evidence(root/'binding.json', object_id='item',
                backends=['sam3d', 'trellis'], frame_ids=list(range(150)), review_frame_ids=SELECTED,
                rendered_frames=[records[i] for i in SELECTED])
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), \
                    patch('utils.runtime._astra_response', side_effect=response(good)), self.assertRaisesRegex(ValueError, 'exactly once'):
                backend.judge(e, images, audit_dir=root/'missing_second')
            both = dict(good, backends=[*good['backends'], dict(good['backends'][0], backend='trellis')])
            with patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}), \
                    patch('utils.runtime._astra_response', side_effect=response(both)):
                result = backend.judge(e, images, audit_dir=root/'both')
            self.assertEqual([r['backend'] for r in result['backends']], ['sam3d', 'trellis'])

    def test_final_mapping_path_hash_frame_order_and_coverage_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, records, _, _ = fixture(root)
            e = {'object_id': 'item', 'frame_ids': list(range(12)), 'rendered_frames': records[:12]}
            images = [Path(r['path']) for r in records[:12]]
            bound = final.review_evidence(e, images)
            self.assertEqual(bound, final.review_evidence(bound, images))
            self.assertNotIn('review_images', e)
            bads = [dict(e, frame_ids=list(range(11))), dict(e, frame_ids=list(reversed(range(12)))),
                    dict(bound, review_frame_ids=list(range(11)))]
            bad = copy.deepcopy(e); bad['rendered_frames'][0]['sha256'] = '0'*64; bads.append(bad)
            bad = copy.deepcopy(e); bad['rendered_frames'][0]['frame_idx'] = 1; bads.append(bad)
            with patch('utils.runtime._astra_response') as send:
                for bad in bads:
                    with self.assertRaises(ValueError): final.judge(bad, images, audit_dir=root/'bad')
                with self.assertRaises(ValueError): final.judge(e, images[::-1], audit_dir=root/'bad')
                send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
