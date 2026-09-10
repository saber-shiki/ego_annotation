"""Actual temporary producer lineage for required decision image identities."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rgb_scene_runtime import SceneLineage, run_command, verify_resume
from prepare_rgb_scene_runtime import resume_contract
from v19_runtime_contracts import sha


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        base=Path(self.tmp.name);self.root=base/'run';self.root.mkdir();self.bundle=base/'bundle'
        write(self.bundle/'runtime/runtime_config.json',{})
        write(self.bundle/'runtime/bundle_manifest.json',{'files':{'runtime/runtime_config.json':sha(self.bundle/'runtime/runtime_config.json')},'external_assets':{'files':{},'trees':{},'interpreters':{}}})
        raw=self.root/'raw.mp4';raw.write_bytes(b'cpu fixture')
        write(self.root/'input/runtime_input_contract.json',{'run_id':str(uuid.uuid4()),'run_root':str(self.root),'case':'case',
            'input_video':str(raw),'input_video_sha256':sha(raw),'bundle_manifest_sha256':sha(self.bundle/'runtime/bundle_manifest.json')})
        self.lineage=SceneLineage(self.root,self.bundle)
        self.manifest=self.root/'input/raw_frame_manifest/manifest.json'
        self.rgb=[self.manifest.parent/f'{i}.jpg' for i in (7,19)]
        self.produce('P01',{self.manifest:{'frames':[{'frame_idx':i,'raw_frame_path':str(p)} for i,p in zip((7,19),self.rgb)]},**{p:'image fixture bytes' for p in self.rgb}})
        self.plan=self.root/'measurements/object_candidates/object_plan_agent.json'
        self.decision(self.plan,'P05',{'objects':[{'object_id':'cup'}]},[self.manifest,self.rgb[0]])
        self.anchor=self.root/'state/anchor_decisions/cup.json'
        directory=self.root/'measurements/object_geometry/anchor_candidates/cup'
        self.proposal=directory/'anchor_candidate_proposals.json';self.review=directory/'anchor_candidate_review.jpg'
        self.produce('P09-propose',{self.proposal:{'object_id':'cup','review_candidates':[{'frame_idx':7}],
            'outputs':{'anchor_candidate_review':str(self.review)}},self.review:'review fixture'})
        self.decision(self.anchor,'P09_anchor',{'object_id':'cup','selected_anchor_frame_idx':7},[self.proposal,self.review])
        self.appearance=self.root/'measurements/demo_visual/cup/appearance_evidence/appearance_mask_manifest.json'
        self.mask=self.appearance.parent/'19.png'
        self.produce('scene_appearance',{self.appearance:{'object_id':'cup','frames':[{'frame_idx':19,'status':'available',
            'raw_rgb_path':str(self.rgb[1]),'appearance_mask_path':str(self.mask)}]},self.mask:'mask fixture'})
        self.choice=self.root/'state/appearance_decisions/cup.json'
        self.decision(self.choice,'appearance_anchor',{'object_id':'cup','selected_frame_idx':19},[self.appearance,self.rgb[1],self.mask])

    def produce(self,phase,values):
        payload={str(p):json.dumps(value) for p,value in values.items()}
        code='import json,pathlib,sys; values=json.loads(sys.argv[1]); [(pathlib.Path(p).parent.mkdir(parents=True,exist_ok=True),pathlib.Path(p).write_text(v)) for p,v in values.items()]'
        run_command(phase,[sys.executable,'-c',code,json.dumps(payload)],list(values),root=self.root,cwd=self.bundle,env=dict(os.environ),lineage=self.lineage)

    def decision(self,path,phase,value,evidence):
        write(path,{**value,'visual_judgment_provenance':{'kind':'current_run_visual_judgment','phase':phase,
            'run_identity':self.lineage.identity,'evidence_sha256':{str(p):sha(p) for p in evidence}}})

    def test_complete_image_evidence_and_verified_resume(self):
        for p in (self.plan,self.anchor,self.choice):self.lineage.require(p)
        receipt=Path(self.tmp.name)/'resume.json';resume_contract(self.bundle,self.root,receipt)
        verify_resume(receipt,self.root,self.bundle)

    def test_required_omissions_wrong_frame_object_and_other_run(self):
        for path in (self.plan,self.anchor,self.choice):
            original=json.loads(path.read_text())
            for name in list(original['visual_judgment_provenance']['evidence_sha256']):
                value=copy.deepcopy(original);value['visual_judgment_provenance']['evidence_sha256'].pop(name);write(path,value)
                with self.subTest(path=path.name,omitted=name),self.assertRaises(ValueError):self.lineage.require(path)
                write(path,original)
            value=copy.deepcopy(original);value['visual_judgment_provenance']['run_identity']['run_id']=str(uuid.uuid4());write(path,value)
            with self.assertRaises(ValueError):self.lineage.require(path)
            write(path,original)
        for path,key in ((self.anchor,'selected_anchor_frame_idx'),(self.choice,'selected_frame_idx')):
            original=json.loads(path.read_text())
            for mutation in ({key:99},{'object_id':'other'}):
                write(path,{**original,**mutation})
                with self.assertRaises(ValueError):self.lineage.require(path)
            write(path,original)
        original=json.loads(self.choice.read_text());value=copy.deepcopy(original)
        evidence=value['visual_judgment_provenance']['evidence_sha256'];evidence.pop(str(self.rgb[1]));evidence[str(self.rgb[0])]=sha(self.rgb[0]);write(self.choice,value)
        with self.assertRaises(ValueError):self.lineage.require(self.choice)
        write(self.choice,original)
        # A byte-identical image from a different producer is not this manifest's image.
        event=self.lineage.records[str(self.rgb[1])];self.lineage.records[str(self.rgb[1])]={**event,'phase':'other'}
        with self.assertRaises(ValueError):self.lineage.require(self.choice)
        self.lineage.records[str(self.rgb[1])]=event

if __name__=='__main__':unittest.main()
