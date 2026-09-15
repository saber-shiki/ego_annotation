"""CPU rigidity full-sheet transport contracts; never a live model judgment."""
import ast
import copy
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import main
from observations.judgments import judge_rigidity as rigidity
from utils.runtime import astra_stage_inputs, encode_astra_request
from utils.state import planned_stage_contract, sha


def fixture(root):
    images=[];sequences=[]
    for index in range(2):
        path=root/f'sheet_{index}.png'
        Image.fromarray(np.random.default_rng(index).integers(0,256,(64,64,3),dtype=np.uint8)).save(path)
        frames=list(range(index*10,(index+1)*10))
        images.append(path);sequences.append({'sheet':str(path),'sha256':sha(path),'frame_ids':frames})
    evidence={'schema':'tracked_whole_rigidity_motion_evidence_v1','object_id':'item',
        'selected_object':{'object_id':'item','description':'synthetic object'},
        'frame_ids':list(range(20)),'motion_sequences':sequences}
    credential=root/'credential';credential.write_text('synthetic-fixture-only');credential.chmod(0o600)
    return evidence,images,credential


def response(request, **kwargs):
    value={'object_id':'item','physical_branch':'articulated','evidence_frames':[0,19],
        'rationale':'synthetic observed relative motion','uncertainties':['fixture'],
        'relative_component_motion':'observed','attached_components':[],
        'motion_evidence':[{'frame_start':0,'frame_end':19,'component':'lid','reference_body':'body',
            'relative_motion_observed':True,'observation':'synthetic lid motion'}]}
    raw={'model':'gpt-6-astra','status':'completed','output':[{'type':'message','role':'assistant',
        'content':[{'type':'output_text','text':json.dumps(value)}]}]}
    class Response(io.BytesIO):
        headers={'x-litellm-model-group':'gpt-6-astra'}
    return Response(json.dumps(raw).encode())


class RigidityBudgetTests(unittest.TestCase):
    def test_mapping_is_complete_idempotent_and_does_not_mutate_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            e,images,_=fixture(Path(td));before=copy.deepcopy(e)
            bound=rigidity.review_evidence(e,images)
            self.assertEqual(e,before)
            self.assertEqual(bound,rigidity.review_evidence(bound,images))
            self.assertEqual(bound['review_frame_ids'],list(range(20)))
            self.assertEqual([r['path'] for r in bound['review_images']],list(map(str,images)))
            self.assertEqual([i for r in bound['review_images'] for i in r['frame_ids']],e['frame_ids'])

    def test_encoder_wire_audit_keep_all_sheets_and_nonrigid_answer(self):
        from gates.object.geometry import require_rigid_tracked_whole
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);e,images,credential=fixture(root)
            bound=rigidity.review_evidence(e,images)
            original,_=encode_astra_request(rigidity.PROMPT,rigidity.SCHEMA,bound,images)
            with patch.object(rigidity,'MAX_REQUEST_BYTES',len(original)-1), \
                 patch.dict(os.environ,{'EGO_ASTRA_CREDENTIAL_FILE':str(credential)}), \
                 patch('utils.runtime._astra_response',side_effect=response) as send:
                actual=rigidity.judge(e,images,audit_dir=root/'audit')
            self.assertEqual(send.call_count,1)
            wire=send.call_args.args[0].data
            self.assertEqual(wire,(root/'audit/effective_payload.json').read_bytes())
            self.assertLess(len(wire),len(original))
            audit=json.loads((root/'audit/request_manifest.json').read_text())
            self.assertGreater(len(audit['encoding_attempts']),1)
            self.assertEqual(len(audit['ordered_images']),2)
            self.assertEqual([f for r in audit['ordered_images'] for f in r['frame_ids']],list(range(20)))
            self.assertEqual(actual['physical_branch'],'articulated')
            with self.assertRaisesRegex(ValueError,'articulated'): require_rigid_tracked_whole(actual)

    def test_bad_order_hash_and_coverage_refuse_before_http(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);e,images,credential=fixture(root)
            with patch('utils.runtime._astra_response') as send, \
                 patch.dict(os.environ,{'EGO_ASTRA_CREDENTIAL_FILE':str(credential)}):
                with self.assertRaisesRegex(ValueError,'path/order'): rigidity.judge(e,images[::-1],audit_dir=root/'badorder')
                bad=copy.deepcopy(e);bad['frame_ids']=bad['frame_ids'][:-1]
                with self.assertRaisesRegex(ValueError,'coverage'): rigidity.judge(bad,images,audit_dir=root/'badcoverage')
                images[0].write_bytes(images[1].read_bytes())
                with self.assertRaisesRegex(ValueError,'order/hash'): rigidity.judge(e,images,audit_dir=root/'badhash')
                send.assert_not_called()

    def test_exhaustion_and_missing_audit_refuse_before_http(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);e,images,credential=fixture(root)
            with patch('utils.runtime._astra_response') as send, \
                 patch.dict(os.environ,{'EGO_ASTRA_CREDENTIAL_FILE':str(credential)}):
                with self.assertRaisesRegex(ValueError,'audit directory'): rigidity.judge(e,images)
                with patch.object(rigidity,'MAX_REQUEST_BYTES',1):
                    with self.assertRaisesRegex(ValueError,'exceeds 1'): rigidity.judge(e,images,audit_dir=root/'exhausted')
                audit=json.loads((root/'exhausted/request_manifest.json').read_text())
                self.assertFalse(audit['budget_satisfied']);self.assertEqual(len(audit['encoding_attempts']),5)
                send.assert_not_called()

    def test_main_judge_cli_provides_required_audit_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);e,images,credential=fixture(root);ev=root/'evidence.json';ev.write_text(json.dumps(e))
            output=root/'result.json'
            argv=['judge','rigidity','--evidence',str(ev),'--output',str(output)]
            for image in images: argv.extend(['--image',str(image)])
            with patch.dict(os.environ,{'EGO_ASTRA_CREDENTIAL_FILE':str(credential)}), \
                 patch('utils.runtime._astra_response',side_effect=response):
                main.main(argv)
            self.assertTrue(output.exists());self.assertTrue((root/'result.request/effective_payload.json').exists())

    def test_actual_cold_ask_binds_effective_evidence_and_budget(self):
        tree=ast.parse(Path(main.__file__).read_text())
        annotate=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='annotate')
        ask=next(x for x in ast.walk(annotate) if isinstance(x,ast.FunctionDef) and x.name=='ask')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);e,images,credential=fixture(root)
            recovery=Mock();recovery.run.side_effect=lambda name,execute,**kwargs: execute()
            ns={'verify_code':lambda: None,'importlib':importlib,'ids':e['frame_ids'],'object_id':'item',
                'planned_stage_contract':planned_stage_contract,'astra_stage_inputs':astra_stage_inputs,
                'process_execution':{},'recovery':recovery,'ledger':Mock(),'root':root,'json':json}
            exec(compile(ast.Module(body=[ask],type_ignores=[]),main.__file__,'exec'),ns)
            with patch.dict(os.environ,{'EGO_ASTRA_CREDENTIAL_FILE':str(credential)}), \
                 patch('utils.runtime._astra_response',side_effect=response) as send:
                ns['ask']('rigidity',e,images,root/'state/rigidity.json')
            planned=recovery.run.call_args.kwargs['contract']['inputs']
            self.assertEqual(planned['review_encoding']['max_request_bytes'],3_000_000)
            wire=json.loads(send.call_args.args[0].data)
            actual=json.loads(wire['input'][0]['content'][0]['text'].split('\n',1)[1])
            self.assertEqual(planned['evidence'],actual)
            self.assertEqual(actual['review_frame_ids'],list(range(20)))
            self.assertEqual(len(actual['review_images']),2)


if __name__ == '__main__':
    unittest.main()
