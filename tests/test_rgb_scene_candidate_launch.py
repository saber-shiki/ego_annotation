"""Producer-free launch preparation: bindings, quoting, and evidence recipes."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_rgb_scene_candidate_launch as launch


class CandidateLaunchPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='launch fixture ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / 'current bundle'
        (self.bundle / 'runtime').mkdir(parents=True)
        (self.bundle / 'scripts').mkdir()
        (self.bundle / 'scripts/run_rgb_scene_stage.py').write_text('raise RuntimeError("do not execute producers")\n')
        self.python = self.root / 'pinned python'
        self.python.symlink_to(sys.executable)
        self.cfg = self.bundle / 'runtime/runtime_config.json'
        self.cfg.write_text(json.dumps({'gpu': 2, 'main_python': str(self.python)}))
        for name in ('rgb_scene_system_prompt.md', 'rgb_scene_runtime_spec.md'):
            (self.bundle / 'runtime' / name).write_text('fixture instruction')
        (self.bundle / 'runtime/bundle_manifest.json').write_text('{"execution":"distinct from birth"}')
        self.run = self.root / 'source run'
        self.candidate = self.run / 'candidates/candidate-A'
        (self.candidate / 'state').mkdir(parents=True)
        (self.run / 'input').mkdir()
        self.source = self.run / 'input/runtime_input_contract.json'
        self.identity = {'run_id': str(uuid.uuid4()), 'run_root': str(self.run),
                         'input_video_sha256': 'b' * 64, 'bundle_manifest_sha256': 'c' * 64}
        self.source.write_text(json.dumps({**self.identity, 'case': 'case-A'}))
        # Deliberately not the hard-coded live event location. Context is authoritative.
        self.events = self.root / 'verified snapshot.jsonl'
        self.events.write_text(json.dumps({'returncode': 0, 'run_identity': self.identity}) + '\n')
        self.context = self.candidate / 'state/verified_source_context.json'
        self.context_value = {'schema': 'rgb_scene_verified_source_context_v1',
                              'source_contract': str(self.source), 'source_contract_sha256': launch.sha256(self.source),
                              'source_event_streams': [{'path': str(self.events), 'sha256': launch.sha256(self.events)}]}
        self.contract = {'schema': 'rgb_scene_candidate_v1', 'candidate_id': self.candidate.name,
                         'object_id': 'object-A', 'run_root': str(self.run), 'run_id': self.identity['run_id'],
                         'selected_frame_idx': 17, 'native_mode': 'new', 'source_identity': self.identity,
                         'source_contract_sha256': launch.sha256(self.source), 'source_context_path': str(self.context)}
        self.cp = self.candidate / 'candidate_contract.json'
        self.save_context()
        self.proposal = self.candidate / 'measurements/object_geometry/anchor_candidates/object-A/anchor_candidate_proposals.json'
        self.proposal.parent.mkdir(parents=True)
        self.proposal.write_text(json.dumps({'candidate_count': 4, 'ranked_candidates': [
            {'frame_idx': 17, 'proposal_score': .9, 'large_unused_field': 'x' * 20000}]}))
        self.p09 = self.candidate / 'measurements/object_geometry/visible_geometry/object-A/v19_visible_geometry_adapter_report.json'
        self.p09.parent.mkdir(parents=True)
        self.p09.write_text('{"status":"ok","output_frame_count":23}')
        self.annotations = self.p09.parent / 'annotations_v19_visible_geometry.json'
        self.annotations.write_text('{"large_annotation_placeholder_for_streaming_hash_test":true}')
        self.out = self.root / 'plan with spaces'

    def save_context(self):
        self.context.write_text(json.dumps(self.context_value))
        self.contract['source_context_sha256'] = launch.sha256(self.context)
        self.cp.write_text(json.dumps(self.contract))

    def prepare(self, **kwargs):
        return launch.materialize(self.bundle, self.cp, self.out, **{'gpu': 2, 'python': str(self.python), **kwargs})

    def rejected(self, **kwargs):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.prepare(**kwargs)
        self.assertFalse(self.out.exists())

    def inspect(self, result, name):
        command = shlex.split(result['inspection_recipes'][name])
        self.assertEqual(command, ['/bin/bash', str(self.out / f'inspect-{name}.sh')])
        run = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        self.assertLessEqual(len(run.stdout.encode()), 12001)
        return json.loads(run.stdout)

    def test_current_argv_symlink_space_paths_and_create_only(self):
        before = {str(p): launch.sha256(p) for p in self.run.rglob('*') if p.is_file()}
        result = self.prepare(image_width=641, image_height=479)
        self.assertEqual(len(result['stages']), 12)
        self.assertEqual(len(result['script_sha256']), 15)
        for name, stage, phase in launch.STAGES:
            body = Path(result['stages'][name]).read_text().splitlines()
            self.assertEqual(shlex.split(body[2]), ['cd', str(self.bundle)])
            argv = shlex.split(body[3])[1:]
            expected = [str(self.python), str(self.bundle / 'scripts/run_rgb_scene_stage.py'), '--stage', stage,
                        '--run-root', str(self.run), '--runtime-config', str(self.cfg), '--gpu', '2',
                        '--candidate-context', str(self.cp)]
            if phase:
                expected += ['--phase', phase]
            self.assertEqual(argv, expected)
            self.assertEqual(shlex.split(result['commands'][name]), ['/bin/bash', result['stages'][name]])
        for path in self.out.glob('*.sh'):
            subprocess.run(['/bin/bash', '-n', str(path)], check=True)
        recipes = json.loads((self.out / 'inspection_recipes.json').read_text())['commands']
        self.assertEqual(recipes, result['inspection_recipes'])
        for name in recipes:
            self.inspect(result, name)
        self.assertNotEqual(result['manifest_sha256'], self.identity['bundle_manifest_sha256'])
        self.assertEqual(before, {str(p): launch.sha256(p) for p in self.run.rglob('*') if p.is_file()})
        old = {p: p.read_bytes() for p in self.out.iterdir()}
        with self.assertRaisesRegex(ValueError, 'create-only'):
            self.prepare()
        self.assertEqual(old, {p: p.read_bytes() for p in self.out.iterdir()})

    def test_inspection_reports_and_annotation_hash_respond_to_evidence(self):
        result = self.prepare()
        initial = self.inspect(result, 'proposal')
        self.assertEqual(initial['candidate_count'], 4)
        self.assertNotIn('large_unused_field', initial['top_ranked'][0])
        self.proposal.write_text('{"candidate_count":6,"ranked_candidates":[]}')
        changed = self.inspect(result, 'proposal')
        self.assertEqual(changed['candidate_count'], 6)
        self.assertNotEqual(initial['evidence_sha256'], changed['evidence_sha256'])
        initial = self.inspect(result, 'P09')
        self.p09.write_text('{"status":"uncertain","output_frame_count":29}')
        self.annotations.write_text('changed geometry evidence')
        changed = self.inspect(result, 'P09')
        self.assertEqual(changed['output_frame_count'], 29)
        for path in (self.p09, self.annotations):
            self.assertNotEqual(initial['evidence_sha256'][str(path)], changed['evidence_sha256'][str(path)])

    def test_missing_bound_event_snapshot_leaves_no_plan(self):
        self.events.unlink()
        self.rejected()

    def test_tampered_bound_event_snapshot_leaves_no_plan(self):
        self.events.write_text('tampered')
        self.rejected()

    def test_legacy_event_binding_accepted_and_hash_checked(self):
        self.context_value.pop('source_event_streams')
        self.context_value.update(source_events=str(self.events), source_events_sha256=launch.sha256(self.events))
        self.save_context()
        self.prepare()
        self.out = self.root / 'second plan'
        self.events.write_text('tampered')
        self.rejected()

    def test_missing_event_binding_rejected(self):
        self.context_value.pop('source_event_streams')
        self.save_context()
        self.rejected()

    def test_missing_or_tampered_context_rejected(self):
        self.context.write_text('{}')
        self.rejected()
        self.context.unlink()
        self.rejected()

    def test_source_identity_and_birth_bundle_mismatch_rejected(self):
        for key in ('run_id', 'run_root', 'input_video_sha256', 'bundle_manifest_sha256'):
            with self.subTest(key=key):
                value = self.contract['source_identity'][key]
                self.contract['source_identity'][key] = 'wrong'
                self.cp.write_text(json.dumps(self.contract))
                self.rejected()
                self.contract['source_identity'][key] = value

    def test_source_contract_tamper_rejected(self):
        self.source.write_text('{}')
        self.rejected()

    def test_gpu_device_and_dimensions_mismatch_rejected(self):
        for kwargs in ({'gpu': 3}, {'gpu': True}, {'device': 'cpu'}, {'device': 'cuda:1'}, {'image_width': -1}):
            with self.subTest(kwargs=kwargs):
                self.rejected(**kwargs)

    def test_invalid_ids_frame_and_path_rejected(self):
        for key, value in (('object_id', '../escape'), ('candidate_id', 'wrong'), ('selected_frame_idx', True), ('schema', 'wrong')):
            old = self.contract[key]
            self.contract[key] = value
            self.cp.write_text(json.dumps(self.contract))
            self.rejected()
            self.contract[key] = old
        self.cp.write_text(json.dumps(self.contract))
        self.cp.rename(self.candidate / 'unrelated.json')
        self.cp = self.candidate / 'unrelated.json'
        self.rejected()

    def test_missing_stage_entry_rejected_before_create(self):
        (self.bundle / 'scripts/run_rgb_scene_stage.py').unlink()
        self.rejected()


if __name__ == '__main__':
    unittest.main()
