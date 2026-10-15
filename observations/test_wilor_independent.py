"""Closed-form independent backend tests; no model inference or case predictions."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from observations.camera.droid_independent import resize_frame,robust_depth_scale
from observations.hand.wilor_world import canonical_rotations
from observations.hand.wilor_calibrated import calibrated_translation
from observations.hand.wilor_perspective import fit_translation
from utils.gpu_sharing import select
from utils.hand_contract import independent_alignment_valid,MODE
from utils.state import sha


class IndependentBackendTests(unittest.TestCase):
    def test_authorized_micro_replay_tolerance_does_not_accept_geometry_errors(self):
        from utils.hand_contract import validate_replay_errors,FK_REPLAY_TOLERANCE_M
        self.assertEqual(FK_REPLAY_TOLERANCE_M,5e-6)
        validate_replay_errors(2.195593062480758e-6,2.100436376723683e-6)
        validate_replay_errors(5e-6,0.)
        for value in [5.001e-6,1e-3,float('nan'),float('inf'),-1e-9]:
            with self.assertRaises(ValueError):validate_replay_errors(value,0.)

    def test_droid_resize_K_before_crop(self):
        image=np.zeros((501,703,3),np.uint8);K=np.array([700,700,350,250])
        im,k=resize_frame(image,K,160000)
        factor=np.sqrt(160000/(501*703));wh=np.array([int(703*factor),int(501*factor)])
        scale=wh/[703,501]
        np.testing.assert_allclose(k,np.r_[K[:2]*scale,(K[2:]+.5)*scale-.5])
        self.assertEqual(im.shape[0]%8,0);self.assertEqual(im.shape[1]%8,0)

    def test_fp16_depth_archive_is_resampled_as_fp32(self):
        from observations.camera.droid_independent import resize_metric_depth
        depth=np.full((64,64),.75,np.float16)
        actual=resize_metric_depth(depth,(32,32))
        self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_array_equal(actual,np.full((32,32),.75,np.float32))

    def test_scale_excludes_current_dynamic_region(self):
        s=np.full((32,32),2.);m=s*1.7;m[:10]=200;exclude=np.zeros(s.shape,bool);exclude[:10]=True
        scale,meta=robust_depth_scale(s,m,exclude)
        self.assertAlmostEqual(scale,1.7);self.assertEqual(meta['pixels'],22*32)

    def test_scale_never_falls_back_to_old_camera(self):
        with self.assertRaises(ValueError):robust_depth_scale(np.ones((4,4)),np.ones((4,4)),np.ones((4,4),bool))

    def test_left_reflection_is_proper_rotation(self):
        R=Rotation.from_rotvec(np.array([[.2,-.1,.3]])).as_matrix();P=np.tile(R[:,None],(1,15,1,1));C=Rotation.from_rotvec([.1,.1,-.1]).as_matrix()[None]
        root,pose=canonical_rotations(R,P,'left',C);M=np.diag([-1.,1.,1.])
        np.testing.assert_allclose(root,C@M@R@M)
        np.testing.assert_allclose(np.linalg.det(root),1)
        np.testing.assert_allclose(np.linalg.det(pose),1)

    def test_shared_chooses_by_resources_not_occupant_count(self):
        rows=[dict(index=0,uuid='a',free_mib=9000,utilization=1),dict(index=4,uuid='b',free_mib=22000,utilization=80),dict(index=6,uuid='c',free_mib=11000,utilization=0)]
        self.assertEqual(select(8000,25,rows)['index'],6)
        with self.assertRaises(RuntimeError):select(18000,25,rows)

    def test_source_K_conversion_no_anatomical_rescaling(self):
        t=calibrated_translation([.1,.2,5],10000,[1408,1408],[975,975,702,706])
        self.assertAlmostEqual(t[2],.4875)
        np.testing.assert_allclose(t[:2],[.101,.199])

    def test_perspective_translation_recovers_without_GT(self):
        local=np.random.default_rng(1).normal(size=(21,3))*.02;K=np.array([975,975,702,706]);t=np.array([.01,-.02,.4])
        uv=(local+t)[:,:2]/(local+t)[:,2:]*K[:2]+K[2:]
        fit,report=fit_translation(local,uv,K,[0,0,.6]);np.testing.assert_allclose(fit,t,atol=1e-7)
        self.assertTrue(report['success'])

    def test_contract_not_accepted_from_flags_alone(self):
        with self.assertRaises(ValueError):independent_alignment_valid(dict(backend='wilor'),[1,1,0,0])

    def test_changed_producer_contract_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic_contract.json';path.write_text('{}')
            a=dict(backend='wilor',mode=MODE,HaWoR_inputs=False,active_contract_reinference_required=False,
                source_K_bound_to_projection=True,canonical_MANO_FK_replayed=True,independent_camera_backend='independent_droid_unidepth',
                producer_contract_path=str(path),producer_contract_sha256=sha(path),source_camera_intrinsics_fx_fy_cx_cy=[975,975,702,706])
            self.assertTrue(independent_alignment_valid(a,[975,975,702,706]))
            path.write_text('{"changed":true}')
            with self.assertRaises(ValueError):independent_alignment_valid(a,[975,975,702,706])


if __name__=='__main__':unittest.main()
