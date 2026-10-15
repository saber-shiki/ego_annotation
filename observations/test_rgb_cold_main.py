"""Actual cold/resume main orchestration with synthetic upstream stage callbacks.

Only the RGB producer and manifest/dataset consumers run their real mechanisms.
All other perception/geometry/judgment stages are explicit synthetic fixtures.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from observations.test_rgb_source import source_fixture
from data.test_rl_data import make_synthetic_bundle
from utils.state import StageRecovery, sha


class ColdRGBMainTests(unittest.TestCase):
    def test_cold_main_worker_manifest_dataset_and_same_run_recovery(self):
        import main
        from data.build_index import build_index
        from data.dataset import EpisodeDataset
        from observations.object import rgb_tracking as rgb
        from observations.manifest import load_source_observation_manifest
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); root = base/'run'
            fixture = source_fixture(base/'fixture')
            prepared, _ = make_synthetic_bundle(base/'hands', frame_count=2)
            prepared = json.loads(prepared.read_text())
            models = base/'assets/mano'; models.mkdir(parents=True)
            (models/'MANO_LEFT.pkl').write_bytes(b'explicit synthetic hash fixture, never loaded')
            (models/'MANO_RIGHT.pkl').write_bytes(b'explicit synthetic hash fixture, never loaded')
            wrapper = models.parent/'wilor/models/mano_wrapper.py'; wrapper.parent.mkdir(parents=True)
            wrapper.write_text('# synthetic identity only; no model is loaded')
            asset_keys = ('sam2_source', 'unidepth_repo', 'unidepth_model', 'hawor_root', 'owlv2_model',
                'sam2_checkpoint', 'lightglue_root', 'sam3d_repo', 'sam3d_config', 'sam3d_moge_checkpoint',
                'sam3d_dinov2_repo', 'sam3d_dinov2_checkpoint', 'agent_credential_file')
            assets = {key: str(base/'not-provisioned') for key in asset_keys}
            for key in ('rgb_python', 'unidepth_python', 'hawor_python', 'owlv2_python', 'sam3d_python'):
                assets[key] = sys.executable
            assets.update(agent_model='gpt-6-astra', agent_provider='dexgem-responses',
                agent_disable_fallbacks=True, mano_models=str(models))
            assets_path = base/'assets.json'; assets_path.write_text(json.dumps(assets))
            video = base/'synthetic.mp4'; video.write_bytes(b'not a video: timing is a synthetic fixture')
            sensor = base/'sensor.json'; sensor.write_text('{}')
            ids = [0, 1]; times = [0., .4]
            timing = SimpleNamespace(presentation_ticks=ids, presentation_times_s=times,
                source_size_wh=(80, 60), report=lambda: {'frame_count': 2, 'source': 'synthetic test'})
            ann = root/'measurements/object_geometry/visible_geometry/synthetic/annotations_v19_visible_geometry.json'
            adapter = ann.parent/'v19_visible_geometry_adapter_report.json'
            raw_manifest = root/'input/raw_frame_manifest/manifest.json'
            bound = root/'state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz'
            hawor = root/'measurements/hand_candidates/hawor_world/hawor_world_hands.npz'
            bridge = root/'state/base_annotations/v19_mano_bridge_from_hawor_world.npz'
            geometry = root/'measurements/geometry_completion/sam3d_synthetic'
            completion = geometry/'completion_report.json'
            owned = root/'measurements/object_tracks/depth_order_owned/track'
            track_path = root/'measurements/object_tracks/sam2_owlv2_box_points/track/sam2/sam2_track.json'
            appearance = root/'measurements/demo_visual/synthetic/appearance_evidence/appearance_mask_manifest.json'
            interaction = root/'state/agent_interaction_judgments/synthetic.json'
            factors = root/'measurements/contact_visibility_factors/case/v19_visible_contact_ownership_factor_report.json'
            mesh = root/'mesh.ply'
            images = [raw_manifest.parent/f'{idx}.png' for idx in ids]
            rows = fixture['frames']
            hand_rows = json.loads(Path(prepared['sources']['annotations']['path']).read_text())['frames']
            for i, row in enumerate(rows):
                row['frame_idx'] = i; row['time_s'] = times[i]
                row['hands'] = hand_rows[i]['hands']
                for hand in row['hands']:
                    hand['metric_mano_state']['current_v18_camera_intrinsics_fx_fy_cx_cy'] = [90., 90., 39.5, 29.5]
                obj = row['objects'][0]; obj['object_id'] = 'synthetic'; obj['mask_path'] = str(root/f'mask{i}.png')
                if i == 0:
                    g = obj['visible_geometry_candidate']; g['frame_idx'] = i
                    g['centroid_world_m'] = np.mean(g['world_vertices_sample_m'], axis=0).tolist()
            def put(path, payload):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload))
            def setup_sources():
                raw_manifest.parent.mkdir(parents=True)
                for i, old in enumerate((3, 11)):
                    shutil.copyfile(base/f'fixture/input/{old}.png', images[i])
                    shutil.copyfile(base/f'fixture/mask{old}.png', root/f'mask{i}.png')
                put(raw_manifest, {'frames': [{'frame_idx': i, 'time_s': times[i], 'raw_frame_path': str(images[i]),
                    'source_width': 80, 'source_height': 60, 'manifest_width': 160, 'manifest_height': 120} for i in ids]})
                put(ann, {'frames': rows}); put(adapter, json.loads((base/'fixture/adapter.json').read_text()))
                bound.parent.mkdir(parents=True)
                with np.load(base/'fixture/depth.npz', allow_pickle=False) as z: arrays = dict(z)
                arrays['frame_idx'] = np.array([0]); np.savez(bound, **arrays)
                hawor.parent.mkdir(parents=True); shutil.copyfile(prepared['sources']['mano_archive']['path'], hawor)
                bridge.parent.mkdir(parents=True); np.savez(bridge, synthetic_only=[1])
                shutil.copyfile(prepared['sources']['object_mesh']['path'], mesh)
                put(completion, {'outputs': {'pose_hypothesis_mesh_labeled': str(mesh), 'collision_eligible_mesh_labeled': str(mesh)}})
                for path in (root/'state/candidate_mask_identity.json', root/'state/calibration/v19_camera_calibration_contract.json', track_path,
                    owned/'sam2_track.json', owned/'native_depth_metric_eligibility.json',
                    root/'measurements/object_candidates/object_box_prompts_owlv2/synthetic/candidate_identity.json'):
                    put(path, {})
                put(root/'measurements/object_geometry/anchor_candidates/synthetic/anchor_candidate_proposals.json',
                    {'review_candidates': [{'frame_idx': 0}]})
                shutil.copyfile(images[0], root/'measurements/object_geometry/anchor_candidates/synthetic/anchor_candidate_review.jpg')
                put(appearance, {'frames': [{'frame_idx': 0, 'status': 'available', 'anchor_eligible': True,
                    'score': 1., 'raw_rgb_path': str(images[0]), 'appearance_mask_path': str(root/'mask0.png')}]})
                put(appearance.with_name('anchor_decision.json'), {'selected_frame_idx': 0})
                put(interaction, {'object_id': 'synthetic'})
                put(factors, {'factor_rows': [], 'ownership_rows': []})
            real_run = StageRecovery.run
            def stage_run(recovery, name, function, *, contract):
                if name in ('rgb_source_measurements', 'source_observations'):
                    return real_run(recovery, name, function, contract=contract)
                def synthetic_stage():
                    if name == 'raw_frames': setup_sources(); return None
                    if name.startswith('judge_'):
                        kind = name.split(':')[0][6:]; path = Path(contract['parameters']['output'])
                        value = {'object_id': 'synthetic'}
                        if kind == 'object': value = {'objects': [{'object_id': 'synthetic', 'track_id': 'track',
                            'description': 'synthetic', 'detector_text_prompts': ['synthetic'],
                            'detector_prompt_frames': ids, 'detector_active_interval': ids}]}
                        if kind == 'mask_track': value = {'identity_consistent': True}
                        if kind == 'pose_anchor': value = {'selected_anchor_frame_idx': 0, 'visual_rationale': 'synthetic'}
                        if kind == 'appearance_anchor': value = {'selected_frame_idx': 0}
                        put(path, value); return value
                    if name == 'detector_text': return {'effective_prompts': ['synthetic']}
                    if name == 'mask_review': return {}, images, []
                    if name == 'base': return bridge
                    if name == 'surface': return ann
                    if name == 'rigidity_evidence': return {}, images
                    if name == 'metric_inputs': return root/'evidence.json', geometry, root/'native.json', {'image': images[0], 'mask': root/'mask0.png'}
                    if name == 'metric_geometry': return completion
                    if name == 'depth_review': return {i: images[i] for i in ids}
                    if name == 'interaction_assembly': return interaction
                    if name == 'interaction_factors': return factors
                    if contract['producer_type'] == 'subprocess': return {'uuid': 'synthetic-cpu-test'}
                    return None
                return real_run(recovery, name, synthetic_stage, contract=contract)
            launches = []
            def model(command, **kwargs):
                self.assertEqual(command[2], 'observations.object.rgb_tracking')
                self.assertIn('--observation-mode', command)
                self.assertNotIn('--initial-pose-report', command)
                launches.append(command)
                argv = command[3:].copy(); argv[argv.index('--device')+1] = 'cpu'
                rgb.build_rgb_edges(argv)
                return {'uuid': 'synthetic-cpu-test'}
            module = ModuleType('lightglue.utils'); module.rbd = lambda value: value
            count = len(fixture['world'])
            matched = {'matches': torch.tensor(np.column_stack([np.arange(count)]*2)), 'scores': torch.full((count,), .8)}
            features = {i: fixture['features'][old] for i, old in enumerate((3, 11))}
            with ExitStack() as stack:
                for target, value in (
                    ('utils.config.preflight_providers', lambda _: None),
                    ('utils.config.recovery_asset_manifest', lambda _: {'synthetic': 'no models'}),
                    ('utils.data.read_cfr_video_timing', lambda _: timing),
                    ('envs.qualification.camera.load_contract', lambda *a, **kw: ({}, {})),
                    ('envs.qualification.camera.plane_intrinsics', lambda *a, **kw: (np.array([90., 90., 39.5, 29.5]), None)),
                    ('envs.qualification.object_geometry.require_rigid_tracked_whole', lambda _: None),
                    ('utils.runtime.run_model', model),
                    ('observations.object.rgb_tracking.extract_rgb_features', lambda *a: features),
                    ('observations.object.rgb_tracking.build_rgb_matcher', lambda *a: lambda _: matched)):
                    stack.enter_context(patch(target, value))
                stack.enter_context(patch.object(StageRecovery, 'run', stage_run))
                stack.enter_context(patch.dict(sys.modules, {'lightglue.utils': module, 'legacy_optimize': None, 'optimize': None, 'render': None}))
                cold = ['observe', '--video', str(video), '--run-root', str(root), '--case', 'case',
                    '--assets-manifest', str(assets_path), '--sensor-calibration', str(sensor), '--gpu', '0']
                result = main.main(cold)
                manifest_path = Path(result['source_observations'])
                manifest = load_source_observation_manifest(manifest_path)
                self.assertEqual(manifest['extensions']['rgb_source_measurements'], 'ego.annotation.rgb_source_measurements.v1')
                dataset = EpisodeDataset(build_index([root]))
                self.assertEqual(len(dataset), 1)
                loaded = dataset[0].load()
                edge = loaded['rgb_evidence'].edges[0]
                np.testing.assert_allclose(edge.rotation_source_to_target_world, fixture['D'], atol=1e-5)
                self.assertFalse(loaded['objects']['synthetic']['pose_valid'][1])
                before = sha(manifest_path)
                journal_before = json.loads((root/'state/recovery/stages.json').read_text())
                resume = ['resume', '--run-root', str(root), '--assets-manifest', str(assets_path), '--gpu', '0']
                main.main(resume)
                self.assertEqual(len(launches), 1)
                self.assertEqual(sha(manifest_path), before)
                source_consumer = Path(main.__file__).resolve().parent/'data/rgb.py'
                original_consumer = source_consumer.read_bytes()
                try:
                    source_consumer.write_bytes(original_consumer + b'\n# synthetic source-change recovery probe\n')
                    with self.assertRaisesRegex(ValueError, 'source code changed'):
                        main.main(resume)
                finally:
                    source_consumer.write_bytes(original_consumer)
                journal = json.loads((root/'state/recovery/stages.json').read_text())
                self.assertEqual(journal['stages'], journal_before['stages'])
                rgb_stage = next(s for s in journal['stages'] if s['name'] == 'rgb_source_measurements')
                self.assertIn('source-world', json.dumps(rgb_stage['planned_contract']))
                report_path = root/manifest['sources']['rgb_source_report']['path']
                report_path.write_text(report_path.read_text()+' ')
                tampered_hash = sha(report_path)
                with self.assertRaises(ValueError): EpisodeDataset(build_index([manifest_path]))[0].load()
                # Recovery quarantines the changed suffix and reruns the real producer once.
                main.main(resume)
                self.assertEqual(len(launches), 2)
                # Index the delivered manifest, not historical copies in quarantine.
                self.assertEqual(len(EpisodeDataset(build_index([manifest_path]))[0].load()['rgb_evidence'].edges), 1)
                recovered = json.loads((root/'state/recovery/stages.json').read_text())
                cutoff = next(i for i, s in enumerate(journal['stages']) if s['name'] == 'rgb_source_measurements')
                self.assertEqual(recovered['stages'][:cutoff], journal['stages'][:cutoff])
                self.assertNotEqual(recovered['stages'][cutoff]['receipt_sha256'], rgb_stage['receipt_sha256'])
                archive = Path(recovered['continuation']['archive'])
                recovery_report = json.loads((archive/'recovery.json').read_text())
                self.assertEqual(recovery_report['reason'], 'missing_or_changed_stage_artifact_or_receipt')
                self.assertEqual(recovery_report['invalidated_stages'][0], 'rgb_source_measurements')
                self.assertIn('source_observations', recovery_report['invalidated_stages'])
                self.assertEqual(sha(archive/'artifacts'/report_path.relative_to(root)), tampered_hash)
                self.assertEqual(sha(archive/'artifacts/source_observations.json'), before)


if __name__ == '__main__':
    unittest.main()
