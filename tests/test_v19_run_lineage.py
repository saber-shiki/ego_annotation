"""Thin lineage tests with actual tiny subprocesses, never model inference."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v19_runtime_contracts import load,sha,run_command,validate_references
from v19_run_lineage import RunLineage
from v19_runtime_assets import verify_external_assets, verify_bundle
from prepare_v19_sam3d_runtime import prepare_run,import_preflight


def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class CurrentRunLineage(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.home=Path(self.tmp.name);self.bundle=self.home/'bundle';self.root=self.home/'run'
        script=self.bundle/'scripts/producer.py';script.parent.mkdir(parents=True)
        script.write_text("import argparse,json,pathlib\np=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--output-dir');a=p.parse_args();o=pathlib.Path(a.output_dir);o.mkdir(parents=True);(o/'manifest.json').write_text(json.dumps({'source':a.source}));(o/'mesh.ply').write_bytes(b'generated fixture triangles');(o/'frame.jpg').write_bytes(b'generated fixture image')\n")
        self.script=script
        self.asset=self.home/'assets/model.pt';self.asset.parent.mkdir();self.asset.write_bytes(b'weights')
        self.identity={'files':{str(self.asset):{'sha256':sha(self.asset),'resolved_path':str(self.asset.resolve())}},'trees':{}}
        write(self.bundle/'runtime/runtime_config.json',{'fixture_only':True})
        self.manifest=write(self.bundle/'runtime/bundle_manifest.json',{'preflight_status':'imports_checked',
            'external_assets':self.identity,'files':{'scripts/producer.py':sha(script),
                'runtime/runtime_config.json':sha(self.bundle/'runtime/runtime_config.json')}})
        self.video=self.home/'raw.mp4';self.video.write_bytes(b'raw fixture bytes')
        with patch('prepare_v19_sam3d_runtime.read_cfr_video_timing',return_value=SimpleNamespace(report=lambda:{})):
            prepare_run(self.bundle,self.root,self.video,'fixture')
        self.raw=self.root/'input/source.mp4'

    def produce(self,lineage,key='P01',source=None,out=None,script=None):
        out=out or self.root/'input/raw_frame_manifest'
        run_command(key,[sys.executable,script or self.script,'--source',source or self.raw,'--output-dir',out],
            [out/'manifest.json'],root=self.root,cwd=self.bundle,env={},lineage=lineage)
        return out/'manifest.json'

    def install_fixture_producer(self, code):
        # Pin the synthetic producer before any execution. This is test setup,
        # not a product API to register existing prediction outputs.
        self.script.write_text(code)
        manifest=load(self.manifest);manifest['files']['scripts/producer.py']=sha(self.script)
        write(self.manifest,manifest)
        contract_path=self.root/'input/runtime_input_contract.json'
        contract=load(contract_path);contract['bundle_manifest_sha256']=sha(self.manifest)
        write(contract_path,contract)

    def test_many_masks_reuse_checked_input_bytes_and_reject_changed_input(self):
        self.install_fixture_producer("""import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--output-dir');a=p.parse_args()
o=pathlib.Path(a.output_dir);o.mkdir(parents=True)
(o/'large_depth.npz').write_bytes(b'D' * (2 * 1024 * 1024))
masks=[]
for i in range(24):
 f=o/f'mask_{i}.png';f.write_bytes(b'mask fixture');masks.append(str(f))
(o/'manifest.json').write_text(json.dumps({'source':a.source,'masks':masks}))
""")
        lineage=RunLineage(self.root,self.bundle)
        depth_report=self.produce(lineage,'P03_fixture',out=self.root/'measurements/depth')
        depth=depth_report.parent/'large_depth.npz'
        annotations=self.produce(lineage,'P09',source=depth,out=self.root/'measurements/visible')
        before=depth.read_bytes();fresh=RunLineage(self.root,self.bundle)
        calls={};bytes_read={}
        def counted(path):
            import hashlib
            name=str(path);calls[name]=calls.get(name,0)+1
            digest=hashlib.sha256()
            with Path(path).open('rb') as handle:
                for block in iter(lambda:handle.read(1 << 20),b''):
                    bytes_read[name]=bytes_read.get(name,0)+len(block)
                    digest.update(block)
            return digest.hexdigest()
        with patch('v19_run_lineage.sha',side_effect=counted):
            # Exactly the P11 reference-walking route from the review.
            snapshots=fresh.inputs([str(annotations)],[])
            for mask in annotations.parent.glob('mask_*.png'):
                fresh.require(mask)
            self.assertEqual(calls[str(depth)],1)
            self.assertEqual(bytes_read[str(depth)],len(before))
            # Explicit post-command checks MUST re-read even cached inputs.
            snapshots[str(depth)]=fresh.checked_digest(depth)
            fresh.success({},snapshots,[],[])
            self.assertEqual(calls[str(depth)],2)
            depth.write_bytes(b'E' * len(before))
            with self.assertRaisesRegex(ValueError,'output changed'):
                fresh.require(annotations.parent/'mask_0.png')
            self.assertEqual(calls[str(depth)],3)
            with self.assertRaisesRegex(ValueError,'mutated source input'):
                fresh.success({}, {str(depth):snapshots[str(depth)]}, [], [])
            self.assertEqual(calls[str(depth)],4)

    def test_cached_digest_does_not_hide_producer_cycles(self):
        lineage=RunLineage(self.root,self.bundle)
        report=self.produce(lineage);lineage.require(report)
        event=lineage.records[str(report)]
        event['input_sha256'][str(report)]=sha(report)
        with self.assertRaisesRegex(ValueError,'cyclic producer lineage'):
            lineage.require(report)

    def test_P17_requires_selected_P15_and_known_P01_frame(self):
        self.install_fixture_producer("""import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--output-dir');p.add_argument('--report-name',default='manifest.json');a=p.parse_args()
o=pathlib.Path(a.output_dir);o.mkdir(parents=True)
f=o/'frame.jpg';f.write_bytes(b'raw frame fixture')
(o/a.report_name).write_text(json.dumps({'source':a.source,'frames':[{'frame_idx':0,'raw_frame_path':str(f)}]}))
""")
        lineage=RunLineage(self.root,self.bundle)
        p01=self.produce(lineage)
        frame=p01.parent/'frame.jpg'
        write(self.root/'state/object_selection.json',{'case':'fixture','object_id':'thing','anchor_frame':0})
        def pose(obj,phase,folder,name):
            output=self.root/f'measurements/pose_fits/{obj}/{folder}'/name
            run_command(phase,[sys.executable,self.script,'--source',p01,'--output-dir',output.parent,'--report-name',name],
                        [output],root=self.root,cwd=self.bundle,env={},lineage=lineage,
                        logdir=self.root/'logs/other' if obj=='other' else None)
            return output
        p14=pose('thing','P14_visible_pose','visible','v18_compact_rigid_object_pose_fit_report.json')
        wrong=pose('other','P15_temporal_pose','temporal','v19_rigid_object_pose_graph_report.json')
        p15=pose('thing','P15_temporal_pose','temporal','v19_rigid_object_pose_graph_report.json')
        judgment=self.root/'state/agent_interaction_judgments/thing.json'
        def declare(evidence):
            write(judgment,{'case':'fixture','target_entity_id':'object:thing','interaction_judgments':[],
                'visual_judgment_provenance':{'kind':'current_run_visual_judgment','phase':'P17_judgment',
                    'run_identity':lineage.identity,'evidence_sha256':{str(p):sha(p) for p in evidence}}})
        for evidence in ([p14],[p14,frame],[wrong,frame]):
            with self.subTest(evidence=evidence):
                declare(evidence)
                with self.assertRaisesRegex(ValueError,'selected-object current P15'):
                    lineage.require(judgment)
        for evidence in ([p15],[p15,p01],[p15,p14.parent/'frame.jpg']):
            with self.subTest(evidence=evidence):
                declare(evidence)
                with self.assertRaisesRegex(ValueError,'current P01 raw frame'):
                    lineage.require(judgment)
        declare([p15,frame]);before=judgment.read_bytes()
        lineage.require(judgment)
        self.assertEqual(judgment.read_bytes(),before)
        self.assertNotIn(str(judgment),lineage.records)
        # Correct filenames alone cannot substitute an earlier producer phase.
        event=lineage.records[str(p15)];event['phase']='P14_visible_pose'
        with self.assertRaisesRegex(ValueError,'selected-object current P15'):
            lineage.require(judgment)

    def test_successful_invocation_hashes_all_outputs_and_recursive_inputs(self):
        original=self.raw.read_bytes();lineage=RunLineage(self.root,self.bundle)
        first=self.produce(lineage)
        second=self.produce(lineage,'P09',first,self.root/'measurements/visible')
        fresh=RunLineage(self.root,self.bundle)
        fresh.require(second);fresh.require(first.parent/'mesh.ply')
        events=[json.loads(line) for line in fresh.events.read_text().splitlines()]
        self.assertEqual(len(events),2)
        self.assertEqual(events[0]['run_identity'],fresh.identity)
        self.assertEqual(events[0]['input_sha256'][str(self.raw)],sha(self.raw))
        self.assertIn(str(first.parent/'frame.jpg'),events[0]['output_sha256'])
        self.assertIn('scripts/producer.py',events[0]['code_sha256'])
        self.assertEqual(self.raw.read_bytes(),original)
        (first.parent/'mesh.ply').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'output changed'):
            RunLineage(self.root,self.bundle).require(first.parent/'mesh.ply')

    def test_copy_of_old_reports_and_events_into_new_root_is_not_fresh(self):
        lineage=RunLineage(self.root,self.bundle);report=self.produce(lineage)
        other=self.home/'other_run'
        with patch('prepare_v19_sam3d_runtime.read_cfr_video_timing',return_value=SimpleNamespace(report=lambda:{})):
            prepare_run(self.bundle,other,self.video,'fixture')
        copied=other/'input/raw_frame_manifest'
        shutil.copytree(report.parent,copied)
        shutil.copytree(self.root/'logs/integration',other/'logs/integration')
        with self.assertRaisesRegex(ValueError,'no successful current-run producer'):
            RunLineage(other,self.bundle).require(copied/'manifest.json')
        # Copied bytes at an unrecorded path in the SAME run are also denied.
        path=self.root/'state/copied.json';shutil.copyfile(report,path)
        with self.assertRaisesRegex(ValueError,'no successful current-run producer'):
            lineage.require(path)

    def test_agent_judgment_distinct_from_execution_and_bound_to_evidence(self):
        lineage=RunLineage(self.root,self.bundle);evidence=self.produce(lineage)
        path=self.root/'measurements/object_candidates/object_plan_agent.json'
        write(path,{'objects':[{'object_id':'thing'}]})
        with self.assertRaisesRegex(ValueError,'visual judgment'):
            lineage.require(path)
        payload=load(path)
        payload['visual_judgment_provenance']={'kind':'current_run_visual_judgment','phase':'P05',
            'run_identity':lineage.identity,'evidence_sha256':{str(evidence):sha(evidence)}}
        write(path,payload);lineage.require(path)
        self.assertNotIn(str(path),lineage.records)
        self.assertEqual(len(lineage.events.read_text().splitlines()),1)
        payload['visual_judgment_provenance']['run_identity']={**lineage.identity,'run_id':str(uuid.uuid4())}
        write(path,payload)
        with self.assertRaisesRegex(ValueError,'visual judgment'):
            RunLineage(self.root,self.bundle).require(path)

    def test_failed_command_cannot_issue_success_record_or_continue(self):
        failing=self.bundle/'scripts/fail.py';failing.write_text('raise SystemExit(7)')
        manifest=load(self.manifest);manifest['files']['scripts/fail.py']=sha(failing);write(self.manifest,manifest)
        contract=load(self.root/'input/runtime_input_contract.json');contract['bundle_manifest_sha256']=sha(self.manifest)
        write(self.root/'input/runtime_input_contract.json',contract)
        lineage=RunLineage(self.root,self.bundle)
        with self.assertRaisesRegex(RuntimeError,'failed'):
            self.produce(lineage,script=failing)
        event=json.loads(lineage.events.read_text())
        self.assertEqual(event['returncode'],7);self.assertNotIn('output_sha256',event)
        self.assertFalse(lineage.records)

    def test_external_assets_revalidated_before_prepare_and_use(self):
        self.asset.write_bytes(b'changed at same path')
        with self.assertRaisesRegex(ValueError,'external asset changed'):verify_bundle(self.bundle)
        with self.assertRaisesRegex(ValueError,'external asset changed'):RunLineage(self.root,self.bundle)
        with self.assertRaisesRegex(ValueError,'external asset changed'):
            prepare_run(self.bundle,self.home/'newrun',self.video,'fixture')
        self.assertFalse((self.home/'newrun').exists())

    def test_external_source_membership_and_interpreter_symlink_change(self):
        tree=self.home/'src';tree.mkdir();(tree/'model.py').write_text('original')
        identity={'files':{},'trees':{str(tree):{'kind':'source','members':['model.py']}}}
        verify_external_assets(identity)
        (tree/'new_import.py').write_text('added')
        with self.assertRaisesRegex(ValueError,'membership changed'):verify_external_assets(identity)
        first=self.home/'python-a';first.write_bytes(b'interpreter')
        second=self.home/'python-b';second.write_bytes(b'interpreter')
        link=self.home/'python';link.symlink_to(first)
        identity={'trees':{},'files':{str(link):{'sha256':sha(link),'resolved_path':str(first)}}}
        link.unlink();link.symlink_to(second)
        with self.assertRaisesRegex(ValueError,'external asset changed'):verify_external_assets(identity)

    def test_bare_nested_clip_references_rejected_but_prose_and_model_names_retained(self):
        for value in ('candidate.json','mesh.ply','clip.mp4','../old/pose.npz','old mesh.ply'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'absolute'):
                validate_references({'outer':[{'source':value}]},self.root)
        validate_references({'note':'the model may explain candidate.json','weight':'MANO_LEFT.pkl',
                             'config':'pipeline.yaml'},self.root)

    def test_runtime_entrypoints_never_request_nested_full_inventory(self):
        import ast
        root=Path(__file__).resolve().parents[1]/'scripts'
        expected={'run_v19_sam3d_reconstruction.py':'geometry','run_v19_offline_rgb_edges.py':'rgb_loader',
                  'build_v19_rgb_pose_initializer.py':'pose_initializer'}
        for name in [*expected,'run_v19_sam3d_integration_stage.py','run_v19_frontend_stage.py']:
            tree=ast.parse((root/name).read_text())
            calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='RunLineage']
            self.assertEqual(len(calls),1,name)
            roles=[kw.value for kw in calls[0].keywords if kw.arg=='asset_role']
            self.assertEqual(len(roles),1,name)
            if name in expected:self.assertEqual(ast.literal_eval(roles[0]),expected[name])
        source=(root/'run_v19_sam3d_integration_stage.py').read_text()
        self.assertIn("asset_role='geometry_dispatch' if a.stage=='geometry' else a.stage",source)

    def test_runtime_asset_roles_hash_only_consumed_subsets(self):
        import v19_runtime_assets as assets
        exe=self.home/'env/bin/python';exe.parent.mkdir(parents=True);exe.write_bytes(b'interpreter');exe.chmod(0o755)
        cfg={'main_python':str(exe),'rgb_python':str(exe),'demo_python':str(exe),
             'sam3d_python':str(exe),'sam3d_repo':str(self.home/'sam'),
             'sam3d_dinov2_repo':str(self.home/'dino'),'sam3d_config':str(self.home/'sam_weights/pipeline.yaml')}
        identity={'files':{},'trees':{},'interpreters':{str(exe):{'resolved_executable':str(exe),'venv_config':None}}}
        def pin(path,content):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            identity['files'][str(path)]={'sha256':sha(path),'resolved_path':str(path)}
            return path
        pin(exe,b'interpreter')
        pipeline=pin(Path(cfg['sam3d_config']),b'slat_generator_ckpt_path: generator.ckpt\n')
        weight=pin(pipeline.parent/'generator.ckpt',b'SAM weight')
        unused=pin(self.home/'hawor/checkpoint.pt',b'HaWoR weights')
        moge=pin(self.home/'sam_weights/moge.pth',b'loader-checked MoGe')
        dino=pin(self.home/'sam_weights/dino.pth',b'loader-checked DINO')
        lg=pin(self.home/'lightglue.pth',b'loader-checked LightGlue')
        cfg.update(sam3d_moge_checkpoint=str(moge),sam3d_dinov2_checkpoint=str(dino),
                   lightglue_weights={'fixture':{'path':str(lg),'sha256':sha(lg)}})
        for root in [self.home/'sam/sam3d_objects',self.home/'sam/notebook',self.home/'dino/dinov2']:
            pin(root/'model.py',b'# source');identity['trees'][str(root)]={'kind':'source','members':['model.py']}
        pin(self.home/'dino/hubconf.py',b'# hub source')
        dispatch=assets.runtime_asset_subset(identity,cfg,'geometry_dispatch')
        geometry=assets.runtime_asset_subset(identity,cfg,'geometry')
        self.assertNotIn(str(weight),dispatch['files']);self.assertIn(str(weight),geometry['files'])
        self.assertNotIn(str(moge),geometry['files'])  # actual SAM loader owns this fixed-hash check
        self.assertEqual(set(geometry['resolved_targets']),{str(moge),str(dino)})
        self.assertEqual(set(dispatch['files']) & set(geometry['files']),{str(exe)})
        for role in ('pose','pose_initializer','rgb_loader','export'):
            subset=assets.runtime_asset_subset(identity,cfg,role)
            self.assertEqual(set(subset['files']),{str(exe)})
            with patch.object(assets,'sha',wraps=sha) as hashes:
                assets.verify_external_assets(subset)
            self.assertEqual([str(c.args[0]) for c in hashes.call_args_list],[str(exe)])
        # Full preparation succeeds; retarget between preparation and scoped
        # consumption to an identical-byte file. No checkpoint byte reread is
        # needed to reject this identity change at the actual role boundary.
        verify_external_assets(identity)
        for checkpoint,role in ((moge,'geometry'),(dino,'geometry'),(lg,'rgb_loader')):
            with self.subTest(checkpoint=checkpoint):
                original=checkpoint.read_bytes()
                other=self.home/(checkpoint.name+'.identical')
                other.write_bytes(original)
                checkpoint.unlink();checkpoint.symlink_to(other)
                self.assertEqual(sha(checkpoint),identity['files'][str(checkpoint)]['sha256'])
                subset=assets.runtime_asset_subset(identity,cfg,role)
                self.assertIn(str(checkpoint),subset['resolved_targets'])
                with patch.object(assets,'sha',wraps=sha) as hashes:
                    with self.assertRaisesRegex(ValueError,'resolved target'):
                        assets.verify_external_assets(subset)
                hashes.assert_not_called()
                checkpoint.unlink();checkpoint.write_bytes(original)
        unused.write_bytes(b'changed unused model')
        assets.verify_external_assets(assets.runtime_asset_subset(identity,cfg,'pose'))
        with self.assertRaisesRegex(ValueError,'external asset changed'):
            assets.verify_external_assets(identity)
        weight.write_bytes(b'changed consumed SAM model')
        with self.assertRaisesRegex(ValueError,'external asset changed'):
            assets.verify_external_assets(assets.runtime_asset_subset(identity,cfg,'geometry'))
        with self.assertRaisesRegex(ValueError,'unknown runtime asset role'):
            assets.runtime_asset_subset(identity,cfg,'skip-everything')

    def test_native_profile_context_and_provider_contract(self):
        from prepare_v19_sam3d_runtime import validate_native_profile,NATIVE_PROFILE
        root=Path(__file__).resolve().parents[1]
        assets=validate_native_profile(root/NATIVE_PROFILE)
        self.assertEqual([p.name for p in assets],['shellgate.ts','shellgate-broker.py','package.json','package-lock.json'])
        path=self.home/'profile.md'
        path.write_text((root/NATIVE_PROFILE).read_text().replace('inheritGlobalContext: false','inheritGlobalContext: true'))
        with self.assertRaisesRegex(ValueError,'context/provider mismatch'):validate_native_profile(path)
        path.write_text((root/NATIVE_PROFILE).read_text().replace('tools: read, write, bash,','tools: arbitrary, read, write, bash,'))
        with self.assertRaisesRegex(ValueError,'tools mismatch'):validate_native_profile(path)

    def test_preflight_overlay_cannot_enable_GPU_network_or_PYTHONPATH(self):
        config={key:sys.executable for key in ('main_python','rgb_python','unidepth_python','owlv2_python','sam3d_python')}
        config.update({key:str(self.home) for key in ('lightglue_root','unidepth_repo','sam2_source','hawor_root','sam3d_repo','lightglue_torch_home')})
        config['sam3d_environment']={'CUDA_VISIBLE_DEVICES':'5','HF_HUB_OFFLINE':'0',
            'TRANSFORMERS_OFFLINE':'0','PYTHONPATH':'/old/source','LIDRA_SKIP_INIT':'true'}
        with patch('prepare_v19_sam3d_runtime.interpreter',return_value=Path(sys.executable)),patch('prepare_v19_sam3d_runtime.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')) as run:
            import_preflight(self.bundle,config)
        for call in run.call_args_list:
            env=call.kwargs['env'];self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(env['HF_HUB_OFFLINE'],'1');self.assertEqual(env['TRANSFORMERS_OFFLINE'],'1')
            self.assertNotIn('PYTHONPATH',env)


if __name__=='__main__':unittest.main()
