import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rgb_scene_candidates import CandidatePaths, prepare_candidate, import_native_receipt, load_candidate
from v19_runtime_contracts import sha


class CandidateWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name) / 'run'
        (self.root / 'input').mkdir(parents=True)
        self.video = self.root / 'input/source.mp4'; self.video.write_bytes(b'video')
        contract = {'schema':'rgb_scene_input_v1','run_id':'aa5f4afc-2dc4-4cfe-aae6-25cd98aafc11',
                    'run_root':str(self.root),'case':'case_dynamic','input_video':str(self.video),
                    'input_video_sha256':sha(self.video),'bundle_manifest_sha256':'b'*64,
                    'source_timing':{'frame_count':3,'fps_ratio':'30','time_base_s':'1/30',
                    'first_pts_s':0.,'duration_s':.1,'max_cfr_grid_error_s':0.,
                    'fps_source_field':'fixture','validation':'fixture'}}
        (self.root/'input/runtime_input_contract.json').write_text(json.dumps(contract))
        proposal=self.root/'measurements/object_geometry/anchor_candidates/tool'
        proposal.mkdir(parents=True)
        self.proposal=proposal/'anchor_candidate_proposals.json'
        self.proposal.write_text(json.dumps({'object_id':'tool','ranked_candidates':[
            {'frame_idx':7,'score':.9},{'frame_idx':42,'score':.8}]}))
        self.rgb=self.root/'frame-7.jpg'; self.mask=self.root/'mask-7.png'
        self.rgb.write_bytes(b'rgb-7'); self.mask.write_bytes(b'mask-7')
        self.native=Path(self.tmp.name)/'native'; self.native.mkdir()
        self.qcdir=self.native/'case_tool_frame_7'; self.qcdir.mkdir()
        paths={}
        for name in ('mesh.ply','native_pose_mesh_opencv_camera.ply','native_pose_mesh_pytorch3d_camera.ply','mesh.glb'):
            p=self.qcdir/name; p.write_bytes((b'ply triangle '+name.encode())); paths[name]=str(p)
        self.qc=self.qcdir/'qc_sam3d_objects_mesh_v7.json'
        self.qc.write_text(json.dumps({'status':'ok','seed':42,'image':str(self.rgb),'mask':str(self.mask),
            'mesh':paths['mesh.ply'],'native_pose_mesh_opencv_camera':paths['native_pose_mesh_opencv_camera.ply'],
            'native_pose_mesh_pytorch3d_camera':paths['native_pose_mesh_pytorch3d_camera.ply'],
            'glb':paths['mesh.glb']}))
        self.execution=self.native/'execution_receipt.json'
        self.execution.write_text(json.dumps({'returncode':0,'source_run_id':'aa5f4afc-2dc4-4cfe-aae6-25cd98aafc11','frame_idx':7,'seed':42,'native_qc':str(self.qc),
            'outputs':{str(p):{'path':str(p),'sha256':sha(p)} for p in [self.qc,*map(Path,paths.values())]}}))
        self.control=self.native/'control_contract.json'
        self.control.write_text(json.dumps({'frame_idx':7,'seed':42,'input_records':{'rgb':{'sha256':sha(self.rgb)},'owned_mask':{'sha256':sha(self.mask)}}}))


    def tearDown(self): self.tmp.cleanup()

    def test_ranking_first_and_explicit_dynamic_frame_are_isolated(self):
        first=prepare_candidate(self.root,'candidate-a')
        self.assertEqual(first['selected_frame_idx'],7)
        second=prepare_candidate(self.root,'candidate-b',frame=42)
        self.assertEqual(second['selected_frame_idx'],42)
        self.assertNotEqual(CandidatePaths(self.root,'candidate-a').root,
                            CandidatePaths(self.root,'candidate-b').root)
        with self.assertRaisesRegex(ValueError,'already exists'):
            prepare_candidate(self.root,'candidate-a')

    def test_import_requires_receipt_and_preserves_derived_mapping(self):
        value=prepare_candidate(self.root,'reuse-a',frame=7,native_mode='reuse',source_native=self.native)
        paths,_=load_candidate(paths:=self.root/'candidates/reuse-a/candidate_contract.json')
        result=import_native_receipt(paths,self.qc,self.execution,self.control,self.rgb,self.mask,model_identity={'seed':42})
        mapped=Path(result['qc_path'])
        self.assertEqual(json.loads(mapped.read_text())['derived_qc_mapping']['input_rgb_sha256'],sha(self.rgb))
        self.assertTrue(result['execution_receipt_path'])
        self.assertTrue(result['native_outputs']['mesh'].endswith('mesh.ply'))
        self.assertNotEqual(sha(mapped),sha(self.qc))
        fresh=prepare_candidate(self.root,'reuse-mutated',frame=7,native_mode='reuse',source_native=self.native)
        self.qc.write_text(json.dumps({'status':'ok','seed':42,'image':str(self.rgb),'mask':str(self.mask),
            'mesh':str(self.qcdir/'mesh.ply'),'native_pose_mesh_opencv_camera':str(self.qcdir/'native_pose_mesh_opencv_camera.ply'),
            'native_pose_mesh_pytorch3d_camera':str(self.qcdir/'native_pose_mesh_pytorch3d_camera.ply'),'glb':str(self.qcdir/'mesh.glb')}))
        (self.qcdir/'mesh.ply').write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError,'output differs'):
            import_native_receipt(fresh[0] if isinstance(fresh,tuple) else CandidatePaths(self.root,'reuse-mutated'), self.qc, self.execution, self.control, self.rgb, self.mask)

    def test_reuse_requires_explicit_source_and_invalid_frame(self):
        with self.assertRaisesRegex(ValueError,'reuse requires'):
            prepare_candidate(self.root,'bad-reuse',native_mode='reuse')
        with self.assertRaisesRegex(ValueError,'not in current'):
            prepare_candidate(self.root,'bad-frame',frame=145)


if __name__ == '__main__': unittest.main()
