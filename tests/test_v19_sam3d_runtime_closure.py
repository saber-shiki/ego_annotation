"""CPU-only runtime closure/CLI tests. No agent, model inference or installation."""
import ast
from fractions import Fraction
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_v19_sam3d_integration_stage as stage
import prepare_v19_sam3d_runtime as prep
import v19_runtime_contracts as contracts
from v20_prediction_contracts import assert_prediction_only


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def producer_options(name):
    # Read the actual parser declarations without importing/initializing models.
    tree = ast.parse((ROOT / 'scripts' / name).read_text())
    flags = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith('--'):
                    flags.add(arg.value)
                    if any(k.arg == 'action' and isinstance(k.value, ast.Attribute) and k.value.attr == 'BooleanOptionalAction' for k in node.keywords):
                        flags.add('--no-' + arg.value[2:])
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name):
            continue
        try:
            values = ast.literal_eval(loop.iter)
        except (ValueError, TypeError):
            continue
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            for node in ast.walk(loop):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add) and isinstance(arg.left, ast.Constant) and arg.left.value == '--' and isinstance(arg.right, ast.Name) and arg.right.id == loop.target.id:
                            flags.add('--' + value)
    return flags


class StageContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'run'
        self.cfg = {'compute_host': socket.gethostname(), 'gpu': 5,
                    **{k: sys.executable for k in ['main_python','demo_python','rgb_python','unidepth_python']},
                    **{k: str(Path(self.temp.name) / 'models') for k in ['lightglue_root','mano_models','wilor_root','hawor_root','unidepth_repo','unidepth_model']}}
        self.config = write(Path(self.temp.name) / 'config.json', self.cfg)
        self.video = self.root / 'input/source.mp4'
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b'raw source bytes; decoder not invoked by dispatcher test')
        self.frames = [{'frame_idx': i, 'time_s': i / 10, 'source_width':80, 'source_height':48} for i in range(3)]
        self.manifest = write(self.root / 'input/raw_frame_manifest/manifest.json', {
            'input_video': str(self.video), 'full_source_timeline': True, 'frames': self.frames})
        self.input = write(self.root / 'input/runtime_input_contract.json', {
            'case':'clip','input_video':str(self.video),'input_video_sha256':contracts.sha(self.video)})
        write(self.root / 'state/object_selection.json', {'case':'clip','object_id':'thing','anchor_frame':0})
        write(self.root / 'measurements/object_candidates/object_plan_agent.json', {'objects':[{'object_id':'thing'}]})
        write(self.root / 'state/anchor_decisions/thing.json', {'object_id':'thing','selected_anchor_frame_idx':0})
        self.ann = write(self.root / 'measurements/object_geometry/visible_geometry/thing/annotations_v19_visible_geometry.json', {'frames':self.frames})
        self.geom = self.root / 'measurements/geometry_completion/sam3d_thing'
        self.pose = self.root / 'measurements/pose_fits/thing'
        self.mesh = self.root / 'measurements/source.ply'
        self.mesh.write_text('fixture triangle asset; no geometric claim')
        self.completion = {'outputs': {'pose_hypothesis_mesh_labeled':str(self.mesh),'completed_mesh_labeled':str(self.mesh)},
                           'render_mesh_contract':{'required_mesh_path':str(self.mesh),'required_mesh_sha256':contracts.sha(self.mesh)}}
        self.calls = []

    def invoke(self, which, fake=None):
        args = ['stage', '--stage', which, '--run-root',str(self.root),'--runtime-config',str(self.config),'--gpu','5']
        with patch.object(sys,'argv',args), patch.object(stage,'run_command',side_effect=fake or self.command), patch.object(stage,'RunLineage',return_value=None), patch.object(stage,'read_cfr_video_timing',return_value=SimpleNamespace(frame_rate=Fraction(10),source_size_wh=(80,48))), contextlib.redirect_stdout(io.StringIO()):
            stage.main()

    def command(self, key, argv, outputs, **kwargs):
        argv = list(map(str,argv)); self.calls.append((key,argv,outputs,kwargs))
        script = Path(argv[1]).name
        if script.endswith('.py'):
            flags = producer_options(script)
            if script == 'run_v19_offline_rgb_edges.py':
                flags |= producer_options('build_v20_global_rgb_orientation_edges.py')
            self.assertLessEqual({a for a in argv[2:] if a.startswith('--')},flags)
        for path in outputs:
            value = {'fixture':key}
            if key == 'P03b_camera':value={'source_plane_intrinsics_fx_fy_cx_cy':[900,900,35.5,21.5]}
            elif key == 'P14b_multiview':value=self.completion
            elif key == 'D00_demo_binding':
                source=load_source=contracts.load(Path(argv[argv.index('--pose-report')+1]))
                value={**source,'render_mesh_contract':self.completion['render_mesh_contract'],
                    'core_source_identity':{'sha256':contracts.sha(Path(argv[argv.index('--pose-report')+1]))}}
            elif key == 'P12_P13_SAM3D':value=self.completion
            elif key == 'P15_temporal_pose':value={'annotation_ready':True,'diagnostic_only':False,'pose_rows':self.frames}
            write(Path(path), value)

    def seed_pose(self):
        write(self.geom / 'completion_report.json',self.completion)
        write(self.root / 'state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz',{})

    def seed_interaction(self):
        self.seed_pose()
        write(self.geom / 'multiview/v19_multiview_supported_completion_report.json',self.completion)
        write(self.pose / 'temporal/v19_rigid_object_pose_graph_report.json',{'annotation_ready':True,'diagnostic_only':False,'pose_rows':self.frames})
        write(self.root / 'measurements/hand_candidates/hawor_world/hawor_world_hands.npz',{})
        write(self.root / 'state/agent_interaction_judgments/thing.json',{'interaction_judgments':[]})

    def test_geometry_requires_agent_selected_P11_identity(self):
        write(self.root/'state/physical_branch_decisions/thing.json',{'physical_branch':'rigid'})
        evidence = self.root / 'measurements/geometry_completion/rigid_evidence/clip/thing/evidence_bundle/evidence_bundle_report.json'
        write(evidence, {'case':'clip','object_id':'thing','selected_frame_idx':2})
        with self.assertRaisesRegex(ValueError,'P11 evidence'):
            self.invoke('geometry')
        self.assertFalse(self.calls)
        write(evidence, {'case':'clip','object_id':'thing','selected_frame_idx':0})
        write(self.root/'state/physical_branch_decisions/thing.json',{'physical_branch':'rigid'})
        self.invoke('geometry')
        self.assertEqual(self.calls[0][0],'P12_P13_SAM3D')

    def test_fixed_frontend_commands_match_producer_parsers(self):
        import run_v19_frontend_stage as frontend
        write(self.root/'measurements/object_candidates/object_plan_agent.json',{'objects':[{
            'object_id':'thing','track_id':'track','description':'selected object','detector_text_prompts':['object.'],
            'detector_prompt_frames':[0,2],'detector_active_interval':[0,2]}]})
        write(self.root/'state/physical_branch_decisions/thing.json',{'physical_branch':'rigid'})
        fake=SimpleNamespace(contract=contracts.load(self.input),require=lambda p:Path(p))
        for phase in frontend.PHASES:
            with self.subTest(phase=phase):
                write(self.manifest,{'frames':self.frames,'input_video':str(self.video)})
                argv=['frontend','--phase',phase,'--run-root',str(self.root),'--runtime-config',str(self.config),'--gpu','5']
                self.cfg.update(owlv2_python=sys.executable,owlv2_model='/models/owl',sam2_checkpoint='/models/sam2.pt')
                write(self.config,self.cfg)
                with patch.object(sys,'argv',argv),patch.object(frontend,'RunLineage',return_value=fake),patch.object(frontend,'run_command',side_effect=self.command):
                    frontend.main()
        self.assertEqual([c[0] for c in self.calls],list(frontend.PHASES))
        self.assertIn('--require-anchor-frame',self.calls[-2][1])
        self.assertIn('--selected-frame-idx',self.calls[-1][1])

    def test_camera_probe_conditioning_and_full_K_hawor(self):
        self.invoke('camera')
        self.assertEqual([c[0] for c in self.calls],['P03a_intrinsics_probe','P03b_camera','P03_conditioned_depth','P03c_depth_binding','P04_HaWoR'])
        self.assertNotIn('--camera-contract',self.calls[0][1])
        self.assertIn('unidepth_intrinsics_probe', ' '.join(self.calls[1][1]))
        self.assertIn('--camera-contract',self.calls[2][1])
        self.assertIn('source_rgb',self.calls[3][1])
        env=self.calls[-1][3]['env']
        self.assertEqual(env['EGO_HAWOR_CAMERA_INTRINSICS_FX_FY_CX_CY'],'900.0 900.0 35.5 21.5')
        self.assertEqual(env['EGO_HAWOR_IMG_FOCAL'],'900.0')
        self.assertEqual(Path(env['EGO_HAWOR_CLIP']).read_bytes(),self.video.read_bytes())
        self.assertEqual(self.calls[-1][3]['gpu'],5)

    def test_camera_rejects_manifest_CFR_mismatch_before_measurements(self):
        payload=contracts.load(self.manifest)
        payload['frames'][1]['time_s'] += .01
        write(self.manifest,payload)
        with self.assertRaisesRegex(ValueError,'constant-rate'):
            self.invoke('camera')
        self.assertFalse(self.calls)

    def test_supplied_sensor_does_not_probe_or_fallback(self):
        sensor=write(self.root/'input/sensor.json',{'source_plane_intrinsics_fx_fy_cx_cy':[900,900,35.5,21.5]})
        data=contracts.load(self.input);data.update(sensor_calibration_metadata=str(sensor),sensor_calibration_authority='prediction_side_sensor_metadata');write(self.input,data)
        self.invoke('camera')
        self.assertEqual(self.calls[0][0],'P03b_camera')
        self.assertIn('--sensor-calibration-contract',self.calls[0][1])
        self.assertNotIn('--unidepth-npz',self.calls[0][1])

    def test_pose_and_interaction_actual_CLI_bindings(self):
        self.seed_pose();self.invoke('pose')
        self.assertEqual([c[0] for c in self.calls],['P14_visible_pose','P14_RGB_observations','P14_RGB_initializer','P14b_multiview','P15_temporal_pose'])
        self.assertIn('--run-root',self.calls[2][1])
        self.assertIn('--complete-full-timeline-rigid-pose',self.calls[-1][1])
        self.seed_interaction();self.calls.clear();self.invoke('interaction')
        self.assertEqual([c[0] for c in self.calls],['P16_constraints','P17_contact_ownership','P18_interval','P18b_metric_preservation'])
        self.assertIn('--require-active-full-K-mano-contract',self.calls[2][1])
        self.assertIn('--wilor-mano-right',self.calls[2][1])
        self.assertIn(str(self.calls[2][2][0]),self.calls[3][1])

    def test_ready_demo_input_is_not_relabeled_or_mutated(self):
        self.seed_interaction()
        write(self.root / 'state/base_annotations/v19_mano_bridge_from_hawor_world.npz',{})
        physical=write(self.root / 'state/render_state/thing_rigid_render_state.json',{'sentinel':'P19'})
        source=self.pose / 'temporal/v19_rigid_object_pose_graph_report.json'
        before=source.read_bytes();core_before=physical.read_bytes()
        self.invoke('demo')
        bound=contracts.load(self.root / 'state/demo_inputs/thing_pose_mesh.json')
        self.assertIs(bound['annotation_ready'],True)
        self.assertIs(bound['diagnostic_only'],False)
        self.assertEqual(bound['core_source_identity']['sha256'],contracts.sha(source))
        self.assertEqual(source.read_bytes(),before);self.assertEqual(physical.read_bytes(),core_before)
        self.assertEqual([c[0] for c in self.calls],['D00_demo_binding','D01_demo_fit','D02_demo_render'])

    def test_P19_P18b_and_export_paths_remain_separate(self):
        self.seed_interaction()
        write(self.root / 'measurements/contact_nonpenetration/thing/v18_mano_object_constraint_state.json',{'sentinel':'P16'})
        p18b=write(self.root / 'measurements/mano_interval_correction/thing_metric_surface/clip/v18_joint_mano_interval_trajectory_state.json',{'sentinel':'P18b','rows':[1,2,3]})
        before=p18b.read_bytes();self.invoke('physical-render')
        self.assertIn(str(p18b),self.calls[0][1]);self.assertEqual(p18b.read_bytes(),before)
        write(self.root / 'measurements/demo_visual/thing/demo_state.npz',{})
        write(self.root / 'measurements/demo_visual/thing/fit_report.json',{})
        self.calls.clear();self.invoke('export')
        self.assertEqual([c[0] for c in self.calls],['P20_layered_annotations','P20_world_rrd','P20_package_video'])
        self.assertIn('--render-state',self.calls[0][1]);self.assertIn('--demo-state',self.calls[0][1])

    def test_outside_history_and_timeline_mismatch_stop_before_launch(self):
        value=contracts.load(self.ann);value['old_pose']=str(Path(self.temp.name)/'old/pose.json');write(self.ann,value)
        with self.assertRaisesRegex(ValueError,'escapes'):self.invoke('pose')
        self.assertFalse(self.calls)
        write(self.ann,{'frames':self.frames[:-1]})
        with self.assertRaisesRegex(ValueError,'frame-time'):self.invoke('pose')
        with self.assertRaisesRegex(ValueError,'absolute'):contracts.validate_references({'pose':'../history/pose.npz'},self.root)
        outside=Path(self.temp.name)/'outside.npz';outside.touch()
        link=self.root/'linked.npz';link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'escapes'):contracts.in_run(link,self.root)

    def test_failure_stops_subsequent_stage_and_records_log(self):
        self.seed_pose()
        with patch.object(contracts.subprocess,'run',return_value=SimpleNamespace(returncode=9)) as launch:
            with self.assertRaisesRegex(RuntimeError,'P14_visible_pose failed'):
                self.invoke('pose',contracts.run_command)
        self.assertEqual(launch.call_count,1)
        self.assertTrue((self.root/'logs/integration/P14_visible_pose.log').exists())
        event=json.loads((self.root/'logs/integration/events.jsonl').read_text().splitlines()[0])
        self.assertEqual(event['returncode'],9)
        self.assertFalse((self.pose/'rgb_initialized_pose.json').exists())

    def test_existing_partial_output_directory_refused(self):
        self.seed_pose();(self.pose/'visible').mkdir(parents=True)
        with patch.object(contracts.subprocess,'run') as launch:
            with self.assertRaisesRegex(ValueError,'directory already exists'):
                self.invoke('pose',contracts.run_command)
        launch.assert_not_called()


class BundleContracts(unittest.TestCase):
    def test_closure_includes_dynamic_experiment_and_offline_dependencies(self):
        selected=set(map(str,prep.closure(ROOT)))
        self.assertLessEqual(prep.EXPERIMENT_HELPERS,selected)
        for name in ['run_v19_offline_rgb_edges','v19_runtime_contracts','v20_rgbd_material_factors','build_v20_global_rgb_orientation_edges']:
            self.assertIn(f'scripts/{name}.py',selected)
        self.assertNotIn('scripts/v20_generated_first_hit_factors.py',selected)
        self.assertFalse(any('remote_run_trellis' in s for s in selected))
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)
            for rel in [*selected,prep.SPEC]:
                path=target/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,path)
            self.assertEqual(set(map(str,prep.closure(target))),selected)
            self.assertFalse((target/'AGENTS.md').exists())
            self.assertFalse((target/'tests').exists())
            # A dynamically referenced missing producer is a closure failure.
            (target/'scripts/build_v19_rgb_pose_initializer.py').unlink()
            with self.assertRaises(ValueError):prep.closure(target)

    def test_source_helper_imports_in_clean_bundle_without_history(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle=Path(temp)
            for rel in prep.closure(ROOT):
                dest=bundle/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
            code="import sys; sys.path.insert(0,'scripts'); import build_v19_rgb_pose_initializer,run_v19_sam3d_reconstruction,run_v19_sam3d_integration_stage; print('ok')"
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''};env.pop('PYTHONPATH',None)
            result=subprocess.run([sys.executable,'-c',code],cwd=bundle,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_raw_input_prepare_refuses_reuse_and_changed_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=root/'bundle';bundle.mkdir()
            source=bundle/'runtime/spec.md';source.parent.mkdir();source.write_text('fixture spec')
            write(bundle/'runtime/bundle_manifest.json',{'external_assets':{'files':{},'trees':{}},'preflight_status':'imports_checked','files':{'runtime/spec.md':contracts.sha(source)}})
            video=root/'raw.mp4';video.write_bytes(b'raw fixture')
            out=root/'run'
            with patch.object(prep,'read_cfr_video_timing',return_value=SimpleNamespace(report=lambda:{'test_only':True})):
                result=prep.prepare_run(bundle,out,video,'clip')
            self.assertEqual((out/'input/source.mp4').read_bytes(),video.read_bytes())
            self.assertEqual(result['input_video_sha256'],contracts.sha(video))
            self.assertFalse((out/'state/object_selection.json').exists())
            with self.assertRaisesRegex(ValueError,'fresh'):prep.prepare_run(bundle,out,video,'clip')
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):prep.prepare_run(bundle,root/'run2',video,'clip')

    def test_asset_config_rejects_history_key(self):
        with self.assertRaisesRegex(ValueError,'unexpected'):
            prep.validate_config({'old_prediction_npz':'/old/pose.npz'})

    def test_offline_RGB_weights_never_fall_back_to_download(self):
        from run_v19_offline_rgb_edges import bind_offline_weights
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'superpoint_v1.pth';path.write_bytes(b'fixture weight bytes')
            calls=[]
            torch=SimpleNamespace(hub=SimpleNamespace(),load=lambda p,**kw:calls.append(p) or {'loaded':True})
            bind_offline_weights({'superpoint_v1.pth':{'path':str(path),'sha256':contracts.sha(path)}},torch)
            result=torch.hub.load_state_dict_from_url('https://example/superpoint_v1.pth')
            self.assertTrue(result['loaded']);self.assertEqual(calls,[path])
            with self.assertRaisesRegex(ValueError,'undeclared'):
                torch.hub.load_state_dict_from_url('https://example/unknown.pth')
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'missing/changed'):
                torch.hub.load_state_dict_from_url('https://example/superpoint_v1.pth')
            self.assertEqual(len(calls),1)

    def test_P17_documented_judgment_fields_match_actual_consumer(self):
        from build_v19_visible_contact_ownership_factor import load_agent_interaction_judgments
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'judgments.json'
            row={'hand_side':'left','start_frame':0,'end_frame':2,'target_entity_id':'object:thing',
                 'contact_state':'possible_contact','occlusion_relation':'unresolved',
                 'contact_prior_probability':.4,'contact_support_uncertainty_m':.03,
                 'depth_reliability':'mixed_or_unresolved'}
            write(path,{'case':'clip','status':'ok','interaction_judgments':[row]})
            index,summary=load_agent_interaction_judgments(path,case='clip',target_entity_id='object:thing',frame_span=(0,2),sides={'left','right'})
            self.assertEqual(len(index),3)
            row.pop('contact_support_uncertainty_m')
            write(path,{'interaction_judgments':[row]})
            with self.assertRaises(ValueError):
                load_agent_interaction_judgments(path,case='clip',target_entity_id='object:thing',frame_span=(0,2),sides={'left'})

    def test_ready_allowance_is_opt_in_and_GT_checks_stay_strict(self):
        for ready in [False,True]:
            value={'object_id':'thing','annotation_ready':ready,'diagnostic_only':False,'nested':{'gt_consumed':False}}
            before=json.dumps(value,sort_keys=True)
            assert_prediction_only(value,label='demo',object_id='thing',allow_annotation_ready=True,allow_nondiagnostic_prediction=True)
            self.assertEqual(before,json.dumps(value,sort_keys=True))
            with self.assertRaises(RuntimeError):assert_prediction_only(value,label='default',object_id='thing')
            value['nested']['gt_consumed']=True
            with self.assertRaisesRegex(RuntimeError,'positive GT'):
                assert_prediction_only(value,label='demo',object_id='thing',allow_annotation_ready=True,allow_nondiagnostic_prediction=True)
        with self.assertRaisesRegex(RuntimeError,'object_id mismatch'):
            assert_prediction_only({'object_id':'other'},label='demo',object_id='thing',allow_annotation_ready=True,allow_nondiagnostic_prediction=True)


if __name__ == '__main__':unittest.main()
