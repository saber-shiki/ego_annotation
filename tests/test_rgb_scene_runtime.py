import copy,json,os,socket,sys,tempfile,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import rgb_scene_runtime as runtime
from v19_runtime_contracts import sha,live_gpu_probe
from prepare_rgb_scene_runtime import closure,resume_contract
from bind_rgb_scene_shape import validate_posterior
import run_rgb_scene_stage as entry

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class RuntimeTests(unittest.TestCase):
    def test_curated_dependency_closure_and_no_old_dispatch(self):
        selected=closure(ROOT);names={p.name for p in selected}
        self.assertIn('export_hawor_world.py',names)
        self.assertIn('rgb_scene_pose_graph.py',names);self.assertIn('fit_v20_visual_contact_demo.py',names)
        self.assertFalse(names & {'run_v19_sam3d_integration_stage.py','run_v19_additive_demo_branch.py','render_v20_visual_contact_demo.py','render_v19_annotation_layers_video.py','assemble_v19_annotation_layers.py'})
        with tempfile.TemporaryDirectory() as td:
            import shutil
            root=Path(td)
            for p in [*selected,Path('runtime/rgb_scene_runtime_spec.md')]:
                dest=root/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dest)
            self.assertEqual(closure(root),selected)
            for p in selected:
                if p.suffix=='.py':compile((root/p).read_text(),str(p),'exec')

    def test_gpu_authorization_does_not_weaken_core(self):
        cfg={'compute_host':socket.gethostname(),'gpu':2,'sam3d_min_free_mib':12000,
             'scene_gpu_policy':{'mode':'shared_idle','authorized':True,'uuid':'GPU-abc','minimum_free_mib':12000}}
        process='GPU-abc, 123, existing-job, 1000'
        def responses(free=12000,util=5,uuid='GPU-abc'):
            return [SimpleNamespace(stdout=f'2, {uuid}, A800, {free}, {util}'),SimpleNamespace(stdout=process)]
        with patch.object(runtime.subprocess,'run',side_effect=responses()):
            self.assertEqual(runtime.probe(cfg,2)['compute_processes'],[process])
        for args in ({'free':11999},{'util':6},{'uuid':'GPU-other'}):
            with patch.object(runtime.subprocess,'run',side_effect=responses(**args)):
                with self.assertRaises((ValueError,RuntimeError)):runtime.probe(cfg,2)
        with patch('v19_runtime_contracts.subprocess.run',side_effect=responses(util=0)):
            with self.assertRaises(RuntimeError):live_gpu_probe(2)

    def test_candidate_root_control_argument_is_not_recursive_input(self):
        candidate_root = Path('/tmp/run/candidates/example')
        lineage = object.__new__(runtime.SceneLineage)
        lineage.candidate = SimpleNamespace(root=candidate_root)
        with patch.object(runtime.RunLineage, 'inputs', return_value={'ok': 'ok'}) as parent:
            result = lineage.inputs(
                ['--candidate-root', str(candidate_root), '--output-dir',
                 str(candidate_root / 'measurements')], [], [candidate_root])
        self.assertEqual(result, {'ok': 'ok'})
        argv, output_roots, extra = parent.call_args.args
        self.assertIn('--candidate-root', argv)
        self.assertNotIn(str(candidate_root), argv)
        self.assertEqual(extra, [])

    def test_byte_bound_successful_resume_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';bundle=Path(td)/'bundle';root.mkdir();bundle.mkdir()
            cfg={'main_python':PYTHON};write(bundle/'runtime/runtime_config.json',cfg)
            write(bundle/'runtime/bundle_manifest.json',{'files':{'runtime/runtime_config.json':sha(bundle/'runtime/runtime_config.json')},'external_assets':{'files':{},'trees':{},'interpreters':{}}})
            raw=root/'input/source.mp4';raw.parent.mkdir();raw.write_bytes(b'synthetic raw identity only')
            contract={'run_id':str(uuid.uuid4()),'run_root':str(root),'case':'case','target_description':'target','input_video':str(raw),'input_video_sha256':sha(raw),'bundle_manifest_sha256':sha(bundle/'runtime/bundle_manifest.json')}
            write(root/'input/runtime_input_contract.json',contract)
            lineage=runtime.SceneLineage(root,bundle)
            output=root/'state/result.json'
            code='import pathlib,sys; p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True);p.write_text("{}")'
            argv=[PYTHON,'-c',code,str(output)]
            runtime.run_command('synthetic',argv,[output],root=root,cwd=bundle,env=dict(os.environ),lineage=lineage)
            rc=Path(td)/'resume.json';resume_contract(bundle,root,rc)
            original=(root/'logs/integration/events.jsonl').read_bytes()
            runtime.verify_resume(rc,root,bundle)
            with patch.dict(os.environ,{'RGB_SCENE_VERIFIED_RESUME':'1'}),patch.object(runtime.subprocess,'run',side_effect=AssertionError('completed producer relaunched')):
                runtime.run_command('synthetic',argv,[output],root=root,cwd=bundle,env=dict(os.environ),lineage=runtime.SceneLineage(root,bundle))
            self.assertEqual((root/'logs/integration/events.jsonl').read_bytes(),original)
            for target in (raw,output,bundle/'runtime/runtime_config.json',root/'input/runtime_input_contract.json'):
                before=target.read_bytes();target.write_bytes(before+b' ')
                with self.assertRaises(ValueError):runtime.verify_resume(rc,root,bundle)
                target.write_bytes(before)
            output.unlink()
            with self.assertRaises(ValueError):runtime.verify_resume(rc,root,bundle)

    def test_dispatch_reaches_refinement_fit_export_without_physical_state(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);obj='tool';cfg=root/'config.json'
            write(cfg,{'compute_host':socket.gethostname(),'gpu':0,'main_python':PYTHON,'rgb_python':PYTHON,'demo_python':PYTHON,'sam3d_python':PYTHON,'mano_models':'/assets/mano','lightglue_root':'/assets/lightglue','sam3d_repo':'/assets/sam3d','sam3d_config':'/assets/config','sam3d_moge_checkpoint':'/assets/moge','sam3d_dinov2_repo':'/assets/dino','sam3d_dinov2_checkpoint':'/assets/dino.pth','sam3d_environment':{}})
            write(root/'state/object_selection.json',{'case':'case','object_id':obj,'track_id':'track','anchor_frame':7})
            write(root/'measurements/object_candidates/object_plan_agent.json',{'objects':[{'object_id':obj,'track_id':'track'}]})
            write(root/f'state/anchor_decisions/{obj}.json',{'selected_anchor_frame_idx':7})
            frames=[{'frame_idx':i,'time_s':t} for i,t in [(7,0),(19,.3),(103,.8)]]
            write(root/'input/raw_frame_manifest/manifest.json',{'frames':frames})
            write(root/f'measurements/object_geometry/visible_geometry/{obj}/annotations_v19_visible_geometry.json',{'frames':frames})
            write(root/f'state/appearance_decisions/{obj}.json',{'object_id':obj,'selected_frame_idx':19})
            appearance=root/f'measurements/demo_visual/{obj}/appearance_evidence/appearance_mask_manifest.json'
            write(appearance,{'frames':[{'frame_idx':19,'status':'available','raw_rgb_path':str(root/'input/rgb.jpg'),'appearance_mask_path':str(root/'input/mask.png')}]})
            calls=[]
            class FakeLineage:
                def __init__(self,*a,**k):self.contract={'case':'case'}
                def require(self,p):return p
            def fake(key,argv,outputs,**kwargs):calls.append((key,list(map(str,argv)),kwargs))
            for stage in ('geometry','initial-pose','refined-pose','appearance','native-shape','shape','visual-fit','export'):
                with patch.object(entry,'SceneLineage',FakeLineage),patch.object(entry,'run_command',fake),patch.object(sys,'argv',['stage','--stage',stage,'--run-root',str(root),'--runtime-config',str(cfg),'--gpu','0']):entry.main()
            names=[v[0] for v in calls]
            self.assertIn('scene_refined_pose',names);self.assertIn('scene_refined_shape',names);self.assertIn('scene_visual_fit',names);self.assertEqual(names[-1],'scene_export')
            for key,argv,kw in calls:
                self.assertFalse(any('/physical' in v or '/temporal/' in v or 'render_v20' in v or 'P19' in v for v in argv))
            fit=next(argv for key,argv,_ in calls if key=='scene_visual_fit')
            self.assertIn(str(root/f'measurements/scene_shape/{obj}/pose_mesh.json'),fit)
            self.assertFalse(any(root.glob('measurements/contact*')))

    def test_source_prior_hash_mutation_and_explicit_binding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source.json';rows=[{'frame_idx':7,'rotation_world_from_completed_canonical_matrix':np.eye(3).tolist(),'translation_world_m':[0,0,1]}];write(source,{'pose_rows':rows})
            rep={'schema':'rgb_scene_refined_pose_v1','object_id':'a','source_prior':{'path':str(source),'sha256':sha(source)},'source_pose_rows':copy.deepcopy(rows),'pose_rows':copy.deepcopy(rows)}
            validate_posterior(rep,'a',root)
            source.write_text(source.read_text()+' ')
            with self.assertRaises(ValueError):validate_posterior(rep,'a',root)
            with self.assertRaises(ValueError):validate_posterior({**rep,'schema':'old_unready'},'a',root)

if __name__=='__main__':unittest.main()
