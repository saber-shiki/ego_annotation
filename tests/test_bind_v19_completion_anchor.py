"""P13 compatibility repair tests; no model inference or pose fitting is run."""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bind_v19_completion_anchor as binder
import fit_v18_compact_rigid_object_pose as p14
import solve_v19_rigid_object_pose_graph as p15


def anchor_fixture(root):
    points = [[0.1, 0, 1], [-0.1, 0, 1], [0, 0.1, 1],
              [0, -0.1, 1], [0, 0, 1.1], [0, 0, 0.9]]
    center = np.asarray(points).mean(axis=0).tolist()
    geometry = {'frame_idx': 7, 'object_id': 'thing', 'world_vertices_sample_m': points,
                'centroid_world_m': center, 'rigid_pose_observation_eligible': True}
    binding = {'selected_frame_idx': 7, 'visible_geometry_frame_idx': 7,
               'canonical_surface_frame_idx': 7, 'selected_vs_canonical_centroid_error_m': 0.0,
               'required_same_frame': True, 'required_centroid_tolerance_m': 1e-6, 'validated': True}
    evidence = {'case': 'clip', 'object_id': 'thing', 'selected_frame_idx': 7,
                'selected_anchor_atomic_binding': binding,
                'selected': {'frame_idx': 7, 'visible_geometry_candidate': copy.deepcopy(geometry)},
                'depth_fused_object_row': {'mesh_reconstruction': {
                    'anchor_frame_idx': 7, 'anchor_centroid_world_m': center, 'atomic_anchor_binding': True}}}
    evidence_path = root / 'evidence.json'
    evidence_path.write_text(json.dumps(evidence))
    completion = {'case': 'clip', 'object_id': 'thing', 'gt_consumed': False,
                  'inputs': {'evidence_report': str(evidence_path)},
                  'input_hashes': {'evidence_report': binder.sha256_bytes(evidence_path.read_bytes())},
                  'outputs': {'pose_hypothesis_mesh_labeled': str(root / 'pose.ply'),
                              'collision_eligible_mesh_labeled': str(root / 'physical.ply')},
                  'render_mesh_contract': {'required_mesh_path': str(root / 'intact.ply'),
                                           'required_mesh_sha256': 'original-mesh-hash', 'renderer_must_match': True},
                  'geometry_readiness': {'signed_geometry_ready': False},
                  'annotation_ready': False, 'uncertainty': 'hidden faces quarantined'}
    annotations = {'v19_visible_geometry_adapter': {'case': 'clip', 'object_id': 'thing', 'anchor_frame_idx': 7},
                   'frames': [{'frame_idx': 7, 'objects': [{'object_id': 'thing', 'visible_geometry_candidate': geometry}]}]}
    return completion, evidence, annotations


class BeforeFitting(Exception):
    pass


def check_actual_p14(annotations, completion, object_id, output_dir):
    """Run unchanged main through its actual hash/frame/centroid checks only.

    Diagnostic mesh loading/sampling is stubbed because it precedes the contract
    but contributes no anchor evidence. JSON reads, eligibility, SHA checking,
    and surfel/centroid checks are real. Trace stops before any pose fitting.
    """
    lines, start = inspect.getsourcelines(p14.main)
    stop_line = start + next(i for i, line in enumerate(lines) if line.strip() == 'depth_indices = sorted(observations)')
    result = {}
    def trace(frame, event, arg):
        if frame.f_code is p14.main.__code__ and event == 'line' and frame.f_lineno == stop_line:
            result.update(anchor_frame_idx=frame.f_locals['anchor_frame_idx'],
                          anchor_binding_centroid_error_m=frame.f_locals['anchor_binding_centroid_error_m'],
                          evidence_sha256=frame.f_locals['actual_evidence_sha256'],
                          binding=frame.f_locals['completion_binding'])
            raise BeforeFitting()
        return trace
    argv = ['p14-contract-check', '--annotations', str(annotations), '--completion-report', str(completion),
            '--object-id', object_id, '--output-dir', str(output_dir)]
    previous_trace = sys.gettrace()
    try:
        with patch.object(sys, 'argv', argv), patch.object(p14, 'load_mesh', return_value=None), \
             patch.object(p14, 'deterministic_sample_mesh', return_value=np.empty((0, 3))):
            sys.settrace(trace)
            try:
                p14.main()
            except BeforeFitting:
                pass
    finally:
        sys.settrace(previous_trace)
    if not result:
        raise AssertionError('actual P14 consumer checks were not reached')
    return result


def check_actual_p15(completion_path, annotations_path, pose_path, binding, evidence_sha):
    pose_path.write_text(json.dumps({'pose_rows': [], 'selected_anchor_atomic_binding': binding,
                                     'selected_anchor_evidence_report_sha256': evidence_sha}))
    args = types.SimpleNamespace(annotations=annotations_path, pose_report=pose_path,
                                 completion_report=completion_path, completed_mesh=None,
                                 frame_start=None, frame_end=None)
    # This is the first operation after the real P15 binding gate; no optimization.
    with patch.object(p15, 'load_mesh', side_effect=BeforeFitting):
        try:
            p15.run(args)
        except BeforeFitting:
            return
    raise AssertionError('actual P15 binding gate not reached')


class CompletionAnchorBinding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.completion, self.evidence, self.annotations = anchor_fixture(self.root)
        self.source = self.root / 'completion.json'
        self.source.write_text(json.dumps(self.completion))
        self.ann = self.root / 'annotations.json'
        self.ann.write_text(json.dumps(self.annotations))
        self.output = self.root / 'bound.json'

    def normalize(self):
        return binder.normalize_completion(self.completion, self.annotations, 'thing')

    def test_copy_preserves_all_source_fields_and_real_consumers_accept(self):
        before = {p: p.read_bytes() for p in [self.source, self.ann, self.root / 'evidence.json']}
        result = binder.bind_report(self.source, self.ann, 'thing', self.output)
        duplicate = self.root / 'bound-second-destination.json'
        binder.bind_report(self.source, self.ann, 'thing', duplicate)
        self.assertEqual(self.output.read_bytes(), duplicate.read_bytes())
        restored = copy.deepcopy(result)
        del restored['anchor_binding_normalization']
        for key in ['candidate_evidence_report', 'candidate_evidence_report_sha256', 'selected_anchor_atomic_binding']:
            del restored['inputs'][key]
        self.assertEqual(restored, self.completion)
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)
        checked = check_actual_p14(self.ann, self.output, 'thing', self.root / 'p14-check')
        self.assertEqual(checked['anchor_frame_idx'], 7)
        self.assertEqual(checked['anchor_binding_centroid_error_m'], 0.0)
        check_actual_p15(self.output, self.ann, self.root / 'pose.json', checked['binding'], checked['evidence_sha256'])

    def test_original_reproduces_actual_p14_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'lacks a byte-bound selected-anchor evidence report'):
            check_actual_p14(self.ann, self.source, 'thing', self.root / 'p14-check')

    def test_changed_evidence_rejected_not_rehashed_into_new_lineage(self):
        with (self.root / 'evidence.json').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'producer-declared P11 SHA256 mismatch'):
            binder.bind_report(self.source, self.ann, 'thing', self.output)
        self.assertFalse(self.output.exists())

    def test_missing_original_digest_rejected_even_with_new_alias(self):
        self.completion['inputs']['candidate_evidence_report_sha256'] = self.completion['input_hashes'].pop('evidence_report')
        with self.assertRaisesRegex(ValueError, 'original producer-declared'):
            self.normalize()

    def test_wrong_digest_rejected(self):
        self.completion['input_hashes']['evidence_report'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.normalize()

    def test_wrong_anchor_rejected(self):
        self.annotations['v19_visible_geometry_adapter']['anchor_frame_idx'] = 8
        with self.assertRaisesRegex(ValueError, 'selected_frame_idx mismatch'):
            self.normalize()

    def test_p11_frame_object_and_canonical_binding_rejected(self):
        for target, key, value in [
            (self.evidence, 'selected_frame_idx', 8),
            (self.evidence['selected_anchor_atomic_binding'], 'validated', False),
            (self.evidence['selected_anchor_atomic_binding'], 'canonical_surface_frame_idx', 8),
            (self.evidence['selected']['visible_geometry_candidate'], 'object_id', 'other'),
            (self.evidence['depth_fused_object_row']['mesh_reconstruction'], 'anchor_centroid_world_m', [9, 9, 9]),
        ]:
            with self.subTest(key=key):
                old = target[key];target[key] = value
                path = self.root / 'evidence.json'
                path.write_text(json.dumps(self.evidence))
                self.completion['input_hashes']['evidence_report'] = binder.sha256_bytes(path.read_bytes())
                with self.assertRaises(ValueError):
                    self.normalize()
                target[key] = old

    def test_wrong_object_rejected(self):
        for target in [self.completion, self.annotations['v19_visible_geometry_adapter'],
                       self.annotations['frames'][0]['objects'][0]['visible_geometry_candidate']]:
            with self.subTest(target=target):
                old = target['object_id'];target['object_id'] = 'other'
                with self.assertRaisesRegex(ValueError, 'object_id mismatch'):
                    self.normalize()
                target['object_id'] = old

    def test_centroid_and_same_centroid_changed_surface_rejected(self):
        geom = self.annotations['frames'][0]['objects'][0]['visible_geometry_candidate']
        geom['centroid_world_m'][0] += 1
        with self.assertRaisesRegex(ValueError, 'centroid mismatch'):
            self.normalize()
        geom['centroid_world_m'][0] -= 1
        geom['world_vertices_sample_m'][0][0] += 0.01
        geom['world_vertices_sample_m'][1][0] -= 0.01
        with self.assertRaisesRegex(ValueError, 'surfels changed'):
            self.normalize()

    def test_conflicting_existing_aliases_rejected(self):
        for key, value in [('candidate_evidence_report', '/different/evidence.json'),
                           ('candidate_evidence_report_sha256', '0' * 64),
                           ('selected_anchor_atomic_binding', {'validated': False})]:
            with self.subTest(key=key):
                self.completion['inputs'][key] = value
                with self.assertRaisesRegex(ValueError, 'existing inputs'):
                    self.normalize()
                del self.completion['inputs'][key]
        self.completion['selected_anchor_atomic_binding'] = {'validated': False}
        with self.assertRaisesRegex(ValueError, 'existing selected_anchor_atomic_binding'):
            self.normalize()

    def test_matching_aliases_are_idempotent(self):
        once = self.normalize()
        self.assertEqual(binder.normalize_completion(once, self.annotations, 'thing'), once)

    def test_positive_gt_rejected_in_each_source(self):
        for target in [self.completion, self.annotations, self.evidence]:
            with self.subTest(target=target):
                target['nested'] = {'gt_consumed': True}
                path = self.root / 'evidence.json'
                path.write_text(json.dumps(self.evidence))
                self.completion['input_hashes']['evidence_report'] = binder.sha256_bytes(path.read_bytes())
                with self.assertRaisesRegex(ValueError, 'GT provenance'):
                    self.normalize()
                del target['nested']

    def test_source_or_existing_output_never_overwritten(self):
        for output in [self.source, self.ann]:
            with self.assertRaisesRegex(ValueError, 'immutable'):
                binder.bind_report(self.source, self.ann, 'thing', output)
        self.output.write_text('untouched')
        with self.assertRaisesRegex(ValueError, 'immutable'):
            binder.bind_report(self.source, self.ann, 'thing', self.output)
        self.assertEqual(self.output.read_text(), 'untouched')


@unittest.skipUnless(os.environ.get('P14_BINDING_REAL_RUN'), 'set P14_BINDING_REAL_RUN for read-only actual evidence check')
class ActualFailedRun(unittest.TestCase):
    def test_actual_report_subprocess_and_unchanged_consumer(self):
        root = Path(os.environ['P14_BINDING_REAL_RUN']).resolve()
        output_root = Path(os.environ['P14_BINDING_CHECK_OUTPUT']).resolve()
        self.assertFalse(output_root.is_relative_to(root))
        source = root / 'measurements/geometry_completion/sam3d_milk_carton/completion_report.json'
        annotations = root / 'measurements/object_geometry/visible_geometry/milk_carton/annotations_v19_visible_geometry.json'
        original, original_sha = binder.read_report(source)
        evidence = Path(original['inputs']['evidence_report'])
        source_hashes = {p: binder.sha256_bytes(p.read_bytes()) for p in [source, evidence, annotations]}
        output = output_root / 'anchor_bound_completion_report.json'
        subprocess.run([sys.executable, str(Path(binder.__file__)), '--completion-report', str(source),
                        '--annotations', str(annotations), '--object-id', 'milk_carton', '--output', str(output)], check=True)
        with self.assertRaisesRegex(RuntimeError, 'lacks a byte-bound selected-anchor evidence report'):
            check_actual_p14(annotations, source, 'milk_carton', output_root / 'original-consumer-check')
        checked = check_actual_p14(annotations, output, 'milk_carton', output_root / 'bound-consumer-check')
        self.assertEqual(checked['anchor_frame_idx'], 92)
        self.assertEqual(checked['anchor_binding_centroid_error_m'], 0.0)
        check_actual_p15(output, annotations, output_root / 'p15-binding-check-pose.json', checked['binding'], checked['evidence_sha256'])
        normalized, _ = binder.read_report(output)
        for key in original:
            if key != 'inputs':
                self.assertEqual(normalized[key], original[key])
        for key in original['inputs']:
            self.assertEqual(normalized['inputs'][key], original['inputs'][key])
        for path, before in source_hashes.items():
            self.assertEqual(binder.sha256_bytes(path.read_bytes()), before)
        proof = {'source_completion_sha256': original_sha, 'source_hashes_unchanged': {str(p): h for p, h in source_hashes.items()},
                 'p14_actual_consumer': checked, 'p15_actual_binding_gate': 'passed_before_mesh_load',
                 'prediction_executed': False, 'diagnostic_mesh_loading_stubbed': True}
        (output_root / 'consumer_check.json').write_text(json.dumps(proof, indent=2) + '\n')
        print(json.dumps(proof))


if __name__ == '__main__':
    unittest.main()
