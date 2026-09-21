"""Bounded strict-schema -> observation assembly -> real P17 loader regressions."""
from __future__ import annotations

import ast
import copy
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from jsonschema import validate

from actions.filters.contact import load_agent_interaction_judgments, contact_row_state_from_judgment
from observations.interaction.assembly import assemble_interaction_batch, interaction_adapter_contract, JUDGMENTS
from utils.state import RunState, StageRecovery, planned_stage_contract, sha


def fixture():
    ids = [17, 23, 41]
    base = {'frame_start': 17, 'frame_end': 41, 'rationale': 'synthetic image prior', 'uncertainty': 'ambiguous boundary'}
    payloads = {
        'contact': {'object_id': 'carton', 'status': 'ok', 'interaction_judgments': [
            {**base, 'target_entity_id': 'object:carton', 'hand_side': 'left',
             'contact_state': 'likely_contact', 'contact_prior_probability': .7}]},
        'pads': {'object_id': 'carton', 'intervals': [{**base, 'hand_side': 'left', 'digits': ['thumb']}]},
        'occlusion': {'object_id': 'carton', 'intervals': [{**base, 'hand_side': 'left', 'relation': 'hand_in_front'}]},
        'depth_reliability': {'object_id': 'carton', 'intervals': [
            {**base, 'entity': 'object', 'reliability': 'supported'},
            {**base, 'entity': 'left', 'reliability': 'mixed_depth'}]},
        'visibility': {'object_id': 'carton', 'frames': [{'frame_idx': i, 'visibility': {
            entity: {'state': 'partially_visible', 'occluder': None, 'uncertainty': 'synthetic uncertain visibility'}
            for entity in ('object', 'left', 'right')}} for i in ids]},
    }
    for name, payload in payloads.items():
        module = importlib.import_module('observations.judgments.judge_' + ('contact_pads' if name == 'pads' else name))
        validate(payload, module.SCHEMA)
        payload['judgment_provenance'] = {'kind': 'synthetic_test', 'judgment': name}
    return ids, payloads


class InteractionAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.ids, self.payloads = fixture()
        self.contract = interaction_adapter_contract({})

    def assemble(self, **kwargs):
        return assemble_interaction_batch(frame_ids=self.ids, object_id='carton', judgments=self.payloads,
            source_hashes={name: 'a' * 64 for name in JUDGMENTS}, contract=self.contract, **kwargs)

    def load(self, path):
        return load_agent_interaction_judgments(path, case='synthetic', target_entity_id='object:carton',
            frame_span=(self.ids[0], self.ids[-1]), sides={'left', 'right'})[0]

    def test_strict_schema_through_real_loader_sparse_ids_and_provenance(self):
        before = copy.deepcopy(self.payloads)
        assembled = self.assemble()
        self.assertEqual(self.payloads, before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'judgments.json'
            path.write_text(json.dumps(assembled))
            index = self.load(path)
        self.assertEqual(set(index), {(i, s) for i in self.ids for s in ('left', 'right')})
        row = index[23, 'left']
        self.assertEqual(row['contact_state'], 'likely_contact')
        self.assertEqual(row['occlusion_relation'], 'hand_in_front_of_object')
        self.assertEqual(row['depth_reliability'], 'hand_depth_unreliable')
        self.assertEqual(row['contact_support_uncertainty_m'], .06)
        self.assertEqual(row['uncertainty'], before['contact']['interaction_judgments'][0]['uncertainty'])
        self.assertEqual(row['evidence']['pads'], before['pads']['intervals'][0])
        self.assertEqual(row['evidence']['visibility'], before['visibility']['frames'][1])
        self.assertEqual(assembled['sources']['contact']['provenance'], before['contact']['judgment_provenance'])
        self.assertFalse(row['evidence']['adapter_contract']['physical_authority_granted'])
        self.assertIn('not_a_measurement', row['evidence']['adapter_contract']['support_uncertainty_kind'])

    def test_missing_evidence_is_unresolved_without_changing_p17_image_fallback(self):
        row = self.assemble()['interaction_judgments'][1]
        self.assertEqual(row['contact_state'], 'unresolved')
        self.assertEqual(row['occlusion_relation'], 'unresolved')
        self.assertEqual(row['depth_reliability'], 'mixed_or_unresolved')
        self.assertFalse(row['evidence']['contact_evidence_present'])
        self.assertIsNone(row['evidence']['contact'])
        self.assertEqual(row['contact_prior_probability'], .5)
        self.assertNotIn('primary_contact_pads', row)
        self.assertEqual(contact_row_state_from_judgment(row, image_candidate=True, allow_agent_create=False),
            (True, 'active_contact_patch'))
        self.assertEqual(contact_row_state_from_judgment(row, image_candidate=False, allow_agent_create=False),
            (True, 'inactive_unresolved_contact_by_agent_judgment'))
        self.payloads['contact']['interaction_judgments'][0]['contact_state'] = 'unresolved'
        row = self.assemble()['interaction_judgments'][0]
        self.assertTrue(row['evidence']['contact_evidence_present'])
        self.assertEqual(row['contact_prior_probability'], .7)

    def test_every_occlusion_vocabulary_and_depth_pair(self):
        for original, mapped in [('hand_in_front', 'hand_in_front_of_object'),
                ('object_in_front', 'object_in_front_of_hand'), ('separate', 'no_visible_occlusion'),
                ('unresolved', 'unresolved')]:
            self.payloads['occlusion']['intervals'][0]['relation'] = original
            self.assertEqual(self.assemble()['interaction_judgments'][0]['occlusion_relation'], mapped)
        for obj in ('supported', 'mixed_depth', 'unsupported', 'unresolved'):
            for hand in ('supported', 'mixed_depth', 'unsupported', 'unresolved'):
                self.payloads['depth_reliability']['intervals'][0]['reliability'] = obj
                self.payloads['depth_reliability']['intervals'][1]['reliability'] = hand
                expected = 'mixed_or_unresolved'
                if obj == 'supported' and hand == 'supported': expected = 'object_depth_reliable'
                if obj == 'supported' and hand in {'mixed_depth', 'unsupported'}: expected = 'hand_depth_unreliable'
                self.assertEqual(self.assemble()['interaction_judgments'][0]['depth_reliability'], expected)

    def test_absent_intervals_and_visibility_do_not_invent_evidence(self):
        self.payloads['contact']['interaction_judgments'] = []
        for name in ('pads', 'occlusion', 'depth_reliability'): self.payloads[name]['intervals'] = []
        self.payloads['visibility']['frames'] = self.payloads['visibility']['frames'][:1]
        rows = self.assemble()['interaction_judgments']
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r['contact_state'] == r['occlusion_relation'] == 'unresolved' for r in rows))
        self.assertIsNone(rows[-1]['evidence']['visibility'])

    def test_overlap_foreign_entity_and_time_mismatch_rejected(self):
        original = copy.deepcopy(self.payloads)
        for name in JUDGMENTS:
            with self.subTest(name=name):
                self.payloads = copy.deepcopy(original)
                key = 'interaction_judgments' if name == 'contact' else 'frames' if name == 'visibility' else 'intervals'
                self.payloads[name][key].append(copy.deepcopy(self.payloads[name][key][0]))
                with self.assertRaisesRegex(ValueError, 'overlapping|duplicate'): self.assemble()
                self.payloads = copy.deepcopy(original)
                self.payloads[name]['object_id'] = 'foreign'
                with self.assertRaisesRegex(ValueError, 'object mismatch'): self.assemble()
                self.payloads = copy.deepcopy(original)
                row = self.payloads[name][key][0]
                row['frame_idx' if name == 'visibility' else 'frame_end'] = 40
                with self.assertRaisesRegex(ValueError, 'timeline|foreign'): self.assemble()
        self.payloads = copy.deepcopy(original)
        self.payloads['contact']['interaction_judgments'][0]['target_entity_id'] = 'hand:carton'
        with self.assertRaisesRegex(ValueError, 'foreign target'): self.assemble()
        self.payloads = copy.deepcopy(original)
        self.payloads['depth_reliability']['intervals'][0]['entity'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'invalid strict judgment'): self.assemble()
        for start, end in ((41, 17), (17.0, 41), (True, 41)):
            self.payloads = copy.deepcopy(original)
            self.payloads['pads']['intervals'][0].update(frame_start=start, frame_end=end)
            with self.subTest(bounds=(start, end)), self.assertRaises(ValueError): self.assemble()

    def test_invalid_priors_and_frame_ids_rejected(self):
        for value in (float('nan'), float('inf'), -1, 0, True, '0.06', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                interaction_adapter_contract({'contact_support_uncertainty_m': value})
        self.contract = interaction_adapter_contract({'contact_support_uncertainty_m': .08})
        self.assertEqual(self.assemble()['interaction_judgments'][0]['contact_support_uncertainty_m'], .08)
        self.contract['adapter'] = 'unapproved_adapter'
        with self.assertRaisesRegex(ValueError, 'unsupported interaction adapter contract'): self.assemble()
        self.contract = interaction_adapter_contract({})
        for ids in ([17, 17], [23, 17], [True], [17.0], []):
            self.ids = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError): self.assemble()

    @unittest.skipUnless(os.environ.get('EGO_INTERACTION_JUDGMENT_BATCH'), 'optional read-only saved judgment batch')
    def test_saved_strict_judgment_batch_through_p17_loader(self):
        directory = Path(os.environ['EGO_INTERACTION_JUDGMENT_BATCH'])
        paths = {name: directory / (name + '.json') for name in JUDGMENTS}
        hashes = {name: sha(path) for name, path in paths.items()}
        payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
        ids = sorted(row['frame_idx'] for row in payloads['visibility']['frames'])
        object_id = payloads['contact']['object_id']
        kwargs = dict(case='saved_read_only', target_entity_id='object:' + object_id.removeprefix('object:'),
            frame_span=(min(ids), max(ids)), sides={'left', 'right'})
        with self.assertRaisesRegex(ValueError, 'occlusion_relation'):
            load_agent_interaction_judgments(paths['contact'], **kwargs)
        assembled = assemble_interaction_batch(frame_ids=ids, object_id=object_id,
            judgments=payloads, source_hashes=hashes, contract=interaction_adapter_contract({}))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'synthetic_compatibility_check.json'
            output.write_text(json.dumps(assembled, allow_nan=False))
            index, summary = load_agent_interaction_judgments(output, **kwargs)
        self.assertEqual(set(index), {(i, side) for i in ids for side in ('left', 'right')})
        self.assertEqual(hashes, {name: sha(path) for name, path in paths.items()})
        print('Read-only saved batch:', json.dumps(summary, sort_keys=True))

    def test_main_assembly_wiring_and_stage_recovery_binding(self):
        # Execute the actual main.py assembly seam, not a test reimplementation;
        # all five ask calls are synthetic files, never API/model invocations.
        source = Path(__file__).resolve().parents[1] / 'main.py'
        tree = ast.parse(source.read_text())
        annotate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'annotate')
        body = next(n for n in annotate.body if isinstance(n, ast.Try)).body
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'interaction')
        seam = compile(ast.fix_missing_locations(ast.Module(body=body[start:start + 4], type_ignores=[])), str(source), 'exec')
        text = ast.unparse(annotate)
        self.assertIn("contract.get('interaction_adapter') != interaction_contract", text)
        self.assertIn("'interaction_adapter': interaction_contract", text)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input').mkdir()
            (root / 'input' / 'runtime_input_contract.json').write_text(json.dumps({
                'run_id': 'synthetic', 'input_video_sha256': 'a' * 64}))
            ledger = RunState(root)
            identity = {'run_id': 'synthetic', 'agent_model': 'gpt-6-astra'}
            recovery = StageRecovery(ledger, identity)
            calls = []
            def ask(name, evidence, images, destination):
                key = 'pads' if name == 'contact_pads' else name
                calls.append(key)
                def produce():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(self.payloads[key]))
                    record('synthetic_' + key, [(destination, 'synthetic_judgment')])
                    return copy.deepcopy(self.payloads[key])
                return stage('synthetic_' + key, inputs={'judgment': self.payloads[key]},
                    parameters={'output': destination})(produce)
            def record(name, outputs):
                ledger.record(name, outputs, object_id='carton', frame_ids=self.ids)
            def stage(name, *, inputs, parameters):
                planned = planned_stage_contract('main.annotate.' + name, 'inprocess',
                    inputs=inputs, parameters=parameters, execution={'mode': 'synthetic_cpu'})
                return lambda function: recovery.run(name, function, contract=planned)
            namespace = dict(root=root, object_id='carton', ids=self.ids,
                raw_by_id={i: {'raw_frame_path': 'unused.jpg'} for i in self.ids},
                depth_images={i: 'unused_depth.jpg' for i in self.ids},
                interaction_contract=self.contract, ask=ask, sha=sha, Path=Path, json=json,
                assemble_interaction_batch=assemble_interaction_batch, stage=stage, record=record)
            exec(seam, namespace)
            path = namespace['judgments']
            original_sha = sha(path)
            payload = json.loads(path.read_text())
            self.assertEqual(set(calls), set(JUDGMENTS))
            self.assertEqual(len(self.load(path)), 6)
            self.assertEqual(payload['frames'], self.payloads['visibility']['frames'])
            self.assertEqual(payload['diagnostics']['occlusion'][0]['intervals'], self.payloads['occlusion']['intervals'])
            self.assertEqual(payload['timeline_evidence']['contact'][0]['intervals'], self.payloads['contact']['interaction_judgments'])
            recovery.close()
            recovery = StageRecovery(ledger, identity, resume=True)
            exec(seam, namespace)
            self.assertEqual(sha(path), original_sha)
            recovery.close()
            recovery = StageRecovery(ledger, identity, resume=True)
            namespace['interaction_contract'] = interaction_adapter_contract({'contact_support_uncertainty_m': .08})
            try:
                with self.assertRaisesRegex(ValueError, 'contract'):
                    exec(seam, namespace)
            finally:
                recovery.close()


if __name__ == '__main__':
    unittest.main()
