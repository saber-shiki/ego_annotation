"""Small adapter tests. Neural reconstruction/bridge/labeling are explicit stubs.

Only native P11 image materialization and the runner's stage/failure/path binding
are exercised here; these tests make no reconstruction-quality claim.
"""
import contextlib
import io
import json
import os
import socket
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_v19_sam3d_reconstruction as runner


class Sam3dOnlyRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'run'
        self.root.mkdir()
        self.out = self.root / 'sam3d'
        self.assets = Path(self.tmp.name) / 'assets'
        self.assets.mkdir()
        self.config = {'main_python': sys.executable, 'sam3d_python': sys.executable,
                       'compute_host': socket.gethostname(), 'gpu': 5}
        for key in ['sam3d_repo', 'sam3d_config', 'sam3d_moge_checkpoint', 'sam3d_dinov2_repo', 'sam3d_dinov2_checkpoint']:
            path = self.assets / key
            path.write_text('test asset, not model weights')
            self.config[key] = str(path)
        self.config_path = self.assets / 'config.json'
        self.config_path.write_text(json.dumps(self.config))
        for name in ['rgb.png', 'mask.png', 'crop.png']:
            Image.fromarray(np.ones((12, 20), np.uint8) * 255).save(self.root / name)
        self.evidence = self.root / 'evidence.json'
        self.payload = {'case': 'clip', 'object_id': 'thing', 'selected_frame_idx': 7, 'selected': {'frame_idx': 7, 'raw_frame_path': str(self.root / 'rgb.png'), 'mask_path': str(self.root / 'mask.png'), 'visible_geometry_candidate': {'mask_path': str(self.root / 'mask.png')}, 'trellis_conditioning_crop': {'crop_rgba': str(self.root / 'crop.png')}}}
        self.evidence.write_text(json.dumps(self.payload))
        self.before = self.evidence.read_bytes()
        self.calls = []
        self.fail_model = False
        self.native = runner.module(Path(runner.__file__).resolve().parents[1] / 'experiments/sam3d_p11_p12_branch/build_p11_dual_geometry_inputs.py', 'sam3d_test_native')

    def invoke(self):
        args = ['test', '--run-root', str(self.root), '--evidence-report', str(self.evidence), '--output-dir', str(self.out), '--runtime-config', str(self.config_path), '--gpu', '5']
        original_module = runner.module
        def modules(path, name):
            result = original_module(path, name)
            if name == 'integration_sam3d_contract':
                result.gpu_preflight = lambda gpu, required: {'physical_gpu_id': gpu, 'passed': True, 'test_stub': True}
            return result
        with patch.object(sys, 'argv', args), patch.object(runner, 'module', side_effect=modules), patch.object(runner.subprocess, 'run', side_effect=self.command), contextlib.redirect_stdout(io.StringIO()):
            runner.main()

    def command(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '5')
        def arg(flag):return Path(argv[argv.index(flag) + 1])
        output = arg('--output-dir')
        name = Path(argv[1]).name
        if name == 'build_p11_dual_geometry_inputs.py':
            self.native.run(types.SimpleNamespace(evidence_report=arg('--evidence-report'), output_dir=output, comparison_mask=None, allow_unverified_mask_provenance=False))
        elif name == 'remote_run_sam3d_objects_mesh_v7.py':
            if self.fail_model:return types.SimpleNamespace(returncode=7)
            tag, image, mask, seed = argv[argv.index('--case') + 1].split('|')
            case = output / tag
            case.mkdir(parents=True)
            for file in ['raw.ply', 'native.ply', 'native_cv.ply', 'raw.glb']:(case / file).write_text('test geometry stub')
            moge = {'checkpoint_sha256': 'da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f', 'network_resolution_allowed': False}
            dino = {'checkpoint_sha256': '36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51', 'hubconf_sha256': 'c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64', 'network_resolution_allowed': False}
            report = {'status': 'ok', 'name': tag, 'image': image, 'mask': mask, 'mesh': str(case / 'raw.ply'), 'glb': str(case / 'raw.glb'), 'native_pose_mesh_pytorch3d_camera': str(case / 'native.ply'), 'native_pose_mesh_opencv_camera': str(case / 'native_cv.ply'), 'native_pose_contract': {'quaternion_order': 'wxyz_scalar_first_pytorch3d', 'metric_status': 'native_monocular_scene_units_not_sensor_meters'}, 'moge_offline': moge, 'dinov2_offline': dino, 'huggingface_offline': {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'network_resolution_allowed': False}}
            (case / 'qc_sam3d_objects_mesh_v7.json').write_text(json.dumps(report))
        elif name == 'build_p13_sam3d_native_metric_bridge.py':
            output.mkdir()
            mesh = output / 'canonical.ply';mesh.write_text('test canonical geometry stub')
            (output / 'sam3d_native_sensor_metric_canonical_bridge_report.json').write_text(json.dumps({'outputs': {'metric_canonical_render_prior': str(mesh)}}))
            (output / 'p13_metric_canonical_input.json').write_text('{}')
        elif name == 'build_v18_compact_rigid_trellis_completion.py':
            self.assertIn('--input-mesh-already-metric-canonical', argv)
            output.mkdir()
            body, physical = output / 'body.ply', output / 'physical.ply'
            body.write_text('test pose mesh');physical.write_text('test observed surface')
            (output / 'v18_compact_rigid_trellis_completion_report.json').write_text(json.dumps({'status': 'ok', 'inputs': {}, 'outputs': {'pose_hypothesis_mesh_labeled': str(body), 'collision_eligible_mesh_labeled': str(physical)}, 'geometry_readiness': {'signed_volume_validated': False}}))
        else:self.fail('unexpected stage: ' + name)
        return types.SimpleNamespace(returncode=0)

    def test_native_input_and_sam3d_only_stage_bindings(self):
        self.invoke()
        report = json.loads((self.out / 'completion_report.json').read_text())
        self.assertEqual(self.before, self.evidence.read_bytes())
        self.assertEqual([Path(c[0][1]).name for c in self.calls], ['build_p11_dual_geometry_inputs.py', 'remote_run_sam3d_objects_mesh_v7.py', 'build_p13_sam3d_native_metric_bridge.py', 'build_v18_compact_rigid_trellis_completion.py'])
        self.assertFalse(report['phase_retention']['trellis_inference_executed'])
        self.assertFalse(report['geometry_readiness']['signed_volume_validated'])
        self.assertEqual(Path(report['render_mesh_contract']['required_mesh_path']).read_text(), 'test canonical geometry stub')
        self.assertNotEqual(report['render_mesh_contract']['required_mesh_path'], report['outputs']['pose_hypothesis_mesh_labeled'])

    def test_model_failure_stops_before_bridge_with_log(self):
        self.fail_model = True
        with self.assertRaisesRegex(RuntimeError, 'P12_SAM3D failed'):self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.out / 'completion_report.json').exists())
        self.assertEqual(json.loads((self.out / 'execution.json').read_text())['steps'][-1]['returncode'], 7)

    def test_external_prediction_reference_rejected_before_copy(self):
        external = self.assets / 'old_prediction.png';external.write_bytes((self.root / 'rgb.png').read_bytes())
        self.payload['selected']['raw_frame_path'] = str(external)
        self.evidence.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError, 'escapes run root'):self.invoke()
        self.assertFalse(self.calls)
        self.assertFalse(self.out.exists())

    def test_symlink_prediction_escape_rejected(self):
        external = self.assets / 'old_prediction.png';external.write_bytes((self.root / 'rgb.png').read_bytes())
        (self.root / 'rgb.png').unlink();(self.root / 'rgb.png').symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'escapes run root'):self.invoke()
        self.assertFalse(self.calls)

    def test_existing_output_rejected(self):
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, 'new output'):self.invoke()
        self.assertFalse(self.calls)


if __name__ == '__main__':unittest.main()
