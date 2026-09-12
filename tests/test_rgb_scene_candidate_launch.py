import json, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_rgb_scene_candidate_launch as launch

class CandidateLaunchPreparationTests(unittest.TestCase):
    def test_materializes_current_contract_without_producers(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); bundle=root/'bundle'; (bundle/'runtime').mkdir(parents=True)
            (bundle/'runtime/runtime_config.json').write_text('{"gpu":3}')
            (bundle/'scripts').mkdir(); (bundle/'scripts/run_rgb_scene_stage.py').write_text('')
            run=root/'run'; (run/'candidates'/'case_with_spaces').mkdir(parents=True); c=run/'candidates'/'case_with_spaces'
            contract={'schema':'rgb_scene_candidate_v1','candidate_id':'case_with_spaces','object_id':'obj','selected_frame_idx':17,'run_root':str(run),'run_id':'ded84007-0cd4-4717-b16a-2865e62eaea1','native_mode':'reuse','source_contract_sha256':'a'*64,'source_identity':{}}
            cp=c/'candidate_contract.json'; cp.write_text(json.dumps(contract))
            with self.assertRaises(ValueError): launch.materialize(bundle,cp,root/'bad-device',gpu=3,device='cpu',python=sys.executable)
            out=root/'plan with spaces'
            result=launch.materialize(bundle,cp,out,gpu=3,device='cuda:0',image_width=640,image_height=480,python=sys.executable)
            self.assertEqual(result['stage_count'],12)
            body=(out/'geometry.sh').read_text()
            self.assertIn('--gpu 3',body); self.assertIn('--candidate-context',body)
            self.assertFalse((run/'producer_output').exists())
            with self.assertRaises(ValueError): launch.materialize(bundle,cp,out,gpu=3,python=sys.executable)
            self.assertEqual(len(list(out.glob('*.sh'))),15)

if __name__ == '__main__': unittest.main()
