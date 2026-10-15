"""Small render/aggregation checks for the bounded training harness."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
import cv2
import numpy as np

from data.coarse_training import ADAPTER
from data.test_coarse_training import fixture
from envs.coarse_annotation_env import CoarseAnnotationEnv
from tools.rl_experiments.preprocessed_learning import family, render_comparison


class HarnessTests(unittest.TestCase):
    def test_named_family_aggregation(self):
        result=family({'object/object:0/rgb_edge/factor=1/rotation':2.,
                       'object/object:0/rgb_reprojection/factor=4':3.,
                       'hand/left/joint_pixel_deadband/frame=0':4.})
        self.assertEqual(result,{'object/rgb_rotation':2.,'object/rgb_reprojection':3.,'hand/joint_pixel_deadband':4.})

    def test_complete_actual_triangle_render_and_decode(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);b,_,_,models=fixture(root)
            cfg={'max_objects':1,'contact_variables_per_hand':5,'mano_models':models,
                 'preprocessed_state_adapter':ADAPTER,'reward_profile':'formal_annotation','reward_scale':1e-5}
            env=CoarseAnnotationEnv([b],cfg);env.reset(seed=0)
            for i in range(3):env.step(np.zeros(118,np.float32))
            zero=env.export_annotation(root/'zero.json')
            env.reset(seed=0)
            action=np.zeros(118,np.float32);action[105]=.1
            env.step(action)
            for i in range(2):env.step(np.zeros(118,np.float32))
            controlled=env.export_annotation(root/'controlled.json')
            manifest=root/'rollout.json';manifest.write_text(json.dumps({'episodes':[{'artifacts':[{'format':'npz','path':controlled['npz']}]}]}))
            evaluation=root/'evaluation.json';evaluation.write_text(json.dumps({'manifest':str(manifest)}))
            app=root/'appearance.json';rows=[]
            for i in range(3):
                image=root/f'rgb_{i}.png';mask=root/f'mask_{i}.png'
                cv2.imwrite(str(image),np.full((64,64,3),100,np.uint8));cv2.imwrite(str(mask),np.zeros((64,64),np.uint8))
                rows.append({'frame_idx':i,'raw_rgb_path':str(image),'appearance_mask_path':str(mask)})
            app.write_text(json.dumps({'frames':rows}))
            annotations=Path(b.source.manifest['sources']['annotations']['path'])
            b.source=SimpleNamespace(source_path=lambda key:annotations if key=='annotations' else app)
            candidate={'steps':256,'evaluation':str(evaluation)}
            render_comparison(root,{'npz':zero['npz']},candidate,candidate,b)
            result=json.loads((root/'renders/render_measurements.json').read_text())
            self.assertEqual(result['decoded_frame_counts'],[3,3])
            self.assertEqual(result['full_native_triangle_count'],4)
            self.assertTrue((root/'renders/review_sheet.jpg').is_file())


if __name__=='__main__':unittest.main()
