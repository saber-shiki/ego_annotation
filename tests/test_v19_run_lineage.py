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
