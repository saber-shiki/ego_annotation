"""CPU-only source visual processing; not candidate fitting or model inference."""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from scipy.spatial import cKDTree
from data.visual_preprocessing import source_hand_views, load_appearance_observations
from data.test_rl_data import make_synthetic_bundle
from data.bundle import EpisodeBundle
from utils.state import sha
from utils.mano import TIP_VERTICES


class StaticVisualTests(unittest.TestCase):
    def test_source_projection_and_pads_keep_complete_mano_and_invalid_mask(self):
        with tempfile.TemporaryDirectory() as d:
            path,_=make_synthetic_bundle(Path(d));data=EpisodeBundle.open(path).load()
            views=source_hand_views(data['hands'],data['frame_idx'],data['T_world_camera'],data['intrinsics'],data['image_size_wh'])
            np.testing.assert_array_equal(views['fit_size_wh'],[960,960])
            # Source K uses integer centers; half-pixel resize is not simply cx*15.
            np.testing.assert_allclose(views['intrinsics_fit'][0],[900.,900.,487.,487.])
            hand=views['hands']['left'];source=data['hands']['left']['vertices']
            expected=cKDTree(source[0]).query(source[0,np.asarray(TIP_VERTICES)],k=12)[1]
            np.testing.assert_array_equal(hand['pad_vertex_ids'][0],expected)
            self.assertEqual(hand['vertices_uv_fit'].shape,(3,778,2))
            self.assertFalse(views['hands']['right']['pad_membership_valid'][1].any())
            self.assertTrue((views['hands']['right']['pad_vertex_ids'][1]==-1).all())
            self.assertFalse(views['mano_lbs_replay_performed'])
            with self.assertRaises(ValueError):hand['vertices_uv_fit'].setflags(write=True)
            # Changing source camera translates projected geometry; no state update is invoked.
            T=np.array(data['T_world_camera']);T[:,0,3]+=.01
            shifted=source_hand_views(data['hands'],data['frame_idx'],T,data['intrinsics'],data['image_size_wh'])
            self.assertGreater(np.max(np.abs(shifted['hands']['left']['vertices_uv_fit']-hand['vertices_uv_fit'])),0)

    def test_appearance_is_lazy_bound_raw_mask_on_fixed_fit_plane(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ann=root/'ann.json';frame={'frame_idx':4,'time_s':.1,'source_width':64,'source_height':48,
                'camera':{'intrinsics_fx_fy_cx_cy':[50,50,31.5,23.5]}}
            ann.write_text(json.dumps({'frames':[frame]}))
            mask=root/'mask.png';a=np.zeros((48,64),np.uint8);a[10:30,20:40]=255;cv2.imwrite(str(mask),a)
            p=root/'appearance.json';payload={'object_id':'obj','source_annotations':str(ann),
                'source_annotations_sha256':sha(ann),'metric_owned_masks_consumed':False,
                'frames':[{'frame_idx':4,'time_s':.1,'status':'available','appearance_mask_path':str(mask),
                    'source_mask_path':str(mask),'appearance_mask_sha256':sha(mask),'source_mask_sha256':sha(mask),
                    'source_size_wh':[64,48]}]};p.write_text(json.dumps(payload))
            obs=load_appearance_observations(p,annotations_path=ann,frames=[frame],object_id='obj')
            prepared=obs.prepare_frame(4)
            self.assertEqual(prepared['mask'].shape,(720,960));self.assertEqual(prepared['contour_uv'].shape,(160,2))
            self.assertEqual(prepared['distance_transform_px'][200,400],0.)
            self.assertGreater(prepared['distance_transform_px'][0,0],0.)
            with self.assertRaises(ValueError):prepared['mask'].setflags(write=True)
            a[0,0]=255;cv2.imwrite(str(mask),a)
            with self.assertRaisesRegex(ValueError,'bytes'):obs.prepare_frame(4)
            payload['metric_owned_masks_consumed']=True;p.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'role'):load_appearance_observations(p,annotations_path=ann,frames=[frame],object_id='obj')


if __name__=='__main__':unittest.main()
