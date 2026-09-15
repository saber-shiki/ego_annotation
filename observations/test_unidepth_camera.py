"""CPU-only regressions for the deployed UniDepth infer camera adapter."""
import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from observations.camera.unidepth import (
    camera_ray_angular_error_deg, infer_unidepth, metric_radius_to_camera_z,
    resize_depth, run,
)


class DeployedInferFake:
    """Audited infer geometry, with a ray-dependent synthetic depth head.

    No UniDepth import/model. Keep this independent of adapter shape helpers so
    a wrong transform in production cannot also change the test oracle.
    """
    shape_constraints = {'ratio_bounds': [0.5, 2.5], 'pixels_min': 200000,
                         'pixels_max': 600000, 'shape_mult': 14}
    interpolation_mode = 'bilinear'

    @staticmethod
    def rays(K, height, width):
        y, x = torch.meshgrid(torch.arange(height, dtype=K.dtype) + .5,
                              torch.arange(width, dtype=K.dtype) + .5, indexing='ij')
        xyz = torch.stack(((x-K[0, 2])/K[0, 0], (y-K[1, 2])/K[1, 1], torch.ones_like(x)))
        return F.normalize(xyz, dim=0).unsqueeze(0)

    def encode_decode(self, inputs, image_metas):
        self.network_image = inputs['image'].clone()
        h, w = inputs['image'].shape[-2:]
        self.network_K = None if inputs['camera'] is None else inputs['camera'].clone()
        # Camera-head predictions deliberately differ from the supplied camera.
        head_K = torch.tensor([[321., 0., w*.43], [0., 456., h*.62], [0., 0., 1.]])
        self.head_K = head_K.clone()
        rays = self.rays(head_K if self.network_K is None else self.network_K, h, w)
        self.depth_head_rays = rays.clone()
        radius = 2. + .2 * rays[:, :1] + .3 * rays[:, 1:2]
        return inputs, {'points': rays * radius, 'rays': rays,
                        'intrinsics': head_K.unsqueeze(0),
                        'confidence': 1.5 + .1 * rays[:, 2:3]}

    def infer(self, rgb, camera=None):
        self.called_camera = None if camera is None else camera.clone()
        h, w = rgb.shape[-2:]
        low, high = self.shape_constraints['ratio_bounds']
        target = min(high, max(low, w/h))
        if w/h > target:
            hp, wp = int(w/target), w
        else:
            hp, wp = h, int(h*target)
        l, t = (wp-w)//2, (hp-h)//2
        r, b = wp-w-l, hp-h-t
        lo, hi = self.shape_constraints['pixels_min'], self.shape_constraints['pixels_max']
        if hasattr(self, 'resolution_level'):
            interval = (hi-lo)/10
            lo, hi = lo+self.resolution_level*interval, lo+(self.resolution_level+1)*interval
        s = math.sqrt(min(hi, max(lo, hp*wp))/(hp*wp))
        nh, nw = (math.ceil(int(n*s)/14)*14 for n in (hp, wp))
        self.geometry = (s, hp, wp, nh, nw, l, r, t, b)
        rgb = rgb.unsqueeze(0).float()/255
        rgb = (rgb-torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))/torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
        rgb = F.interpolate(F.pad(rgb, (l, r, t, b)), size=(nh, nw), mode='bilinear', align_corners=False)
        if camera is not None:
            # Deployed Camera crop/resize mutate storage.
            camera[0, 2] += l
            camera[1, 2] += t
            camera[:2] *= s
        _, out = self.encode_decode(inputs={'image': rgb, 'camera': camera}, image_metas=[])
        def post(value):
            return F.interpolate(value, size=(hp, wp), mode='bilinear', align_corners=False)[..., t:hp-b, l:wp-r]
        points = post(out['points'])
        K = out['intrinsics'].clone()
        K[:, :2] /= s
        K[:, 0, 2] -= l
        K[:, 1, 2] -= t
        self.returned = {'depth': points[:, -1:], 'radius': points.norm(dim=1, keepdim=True),
                         'rays': F.normalize(post(out['rays']), dim=1), 'intrinsics': K,
                         'confidence': post(out['confidence'])}
        return self.returned


class UniDepthCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def milk_K():
        return np.array([[260.24263088580244, 0., 488.98901089754963],
                         [0., 260.24263088580244, 487.56567105379963], [0., 0., 1.]])

    def test_old_960_failure_and_actual_corrected_adapter(self):
        K = self.milk_K()
        image = Image.new('RGB', (960, 960), (50, 100, 150))
        old = DeployedInferFake()
        old_prediction = old.infer(torch.zeros(3, 960, 960, dtype=torch.uint8), torch.tensor(K))
        old_error = camera_ray_angular_error_deg(old_prediction['rays'][0].numpy(), K)
        self.assertAlmostEqual(old_error['max'], 2.04912504765, places=7)
        new = DeployedInferFake()
        depth, _, rays, radius, confidence = infer_unidepth(new, image, torch.device('cpu'), K)
        self.assertEqual(new.network_image.shape[-2:], (784, 784))
        self.assertLess(camera_ray_angular_error_deg(rays, K)['max'], .03)
        self.assertGreater(float(np.max(np.abs(radius-old_prediction['radius'][0, 0].numpy()))), .005)
        self.assertFalse(torch.allclose(old.depth_head_rays.float(), new.depth_head_rays))
        np.testing.assert_array_equal(depth, new.returned['depth'][0, 0].numpy())
        np.testing.assert_array_equal(radius, new.returned['radius'][0, 0].numpy())
        np.testing.assert_array_equal(confidence, new.returned['confidence'][0, 0].numpy())
        np.testing.assert_allclose(new.network_image[0, :, 0, 0],
                                   (np.array([50, 100, 150])/255-[.485, .456, .406])/[.229, .224, .225], atol=2e-7)

    def check_geometry(self, height, width, conditioned=True, level=None):
        provider = DeployedInferFake()
        if level is not None:
            provider.resolution_level = level
        K = np.array([[width*.39, 0., width*.43], [0., height*.71, height*.62], [0., 0., 1.]])
        before = K.copy()
        # Coordinate ramp also checks that RGB resize samples the same plane.
        y, x = np.mgrid[:height, :width]
        pixels = np.stack((x % 251, y % 251, (x+y) % 251), axis=-1).astype(np.uint8)
        result = infer_unidepth(provider, Image.fromarray(pixels), torch.device('cpu'), K if conditioned else None)
        s, hp, wp, nh, nw, l, r, t, b = provider.geometry
        ax, ay = nw/wp, nh/hp
        network = K.copy()
        network[0, 0] *= ax
        network[1, 1] *= ay
        network[0, 2] = ax*(K[0, 2]+l+.5)
        network[1, 2] = ay*(K[1, 2]+t+.5)
        if conditioned:
            np.testing.assert_allclose(provider.network_K, network, rtol=2e-7, atol=5e-5)
            # Independently map network half centers to original integer pixels.
            yy, xx = np.mgrid[:nh, :nw]
            u, v = (xx+.5)/ax-l-.5, (yy+.5)/ay-t-.5
            expected = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(u)))
            expected /= np.linalg.norm(expected, axis=0)
            np.testing.assert_allclose(provider.depth_head_rays[0], expected, atol=3e-7)
            self.assertLess(camera_ray_angular_error_deg(result[2], K)['max'], .1)
        else:
            self.assertIsNone(provider.called_camera)
            self.assertIsNone(provider.network_K)
        # Inverse of actual scale/half centers, NOT input K or nominal scale.
        expected_head = provider.head_K.numpy().astype(np.float64)
        expected_head[0] /= ax
        expected_head[1] /= ay
        expected_head[0, 2] -= l+.5
        expected_head[1, 2] -= t+.5
        np.testing.assert_allclose(result[1], expected_head, rtol=2e-7, atol=5e-5)
        np.testing.assert_array_equal(K, before)
        if not conditioned:
            self.assertLess(camera_ray_angular_error_deg(result[2], result[1])['max'], .1)
        again = infer_unidepth(provider, Image.fromarray(pixels), torch.device('cpu'), K if conditioned else None)
        for first, second in zip(result, again):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(K, before)
        # Check actual normalized image at the depth-head seam independently.
        rgb = torch.tensor(pixels).permute(2, 0, 1).float().unsqueeze(0)/255
        rgb = (rgb-torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))/torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
        expected_image = F.interpolate(F.pad(rgb, (l, r, t, b)), (nh, nw), mode='bilinear', align_corners=False)
        torch.testing.assert_close(provider.network_image, expected_image, rtol=0, atol=0)

    def test_non_square_padding_unequal_focal_offcenter_and_repeat_calls(self):
        for h, w in [(641, 959), (121, 803), (803, 121), (960, 960)]:
            with self.subTest(height=h, width=w):
                self.check_geometry(h, w)

    def test_unconditioned_head_K_is_corrected_not_replaced(self):
        for h, w in [(641, 959), (121, 803), (803, 121)]:
            with self.subTest(height=h, width=w):
                self.check_geometry(h, w, conditioned=False)

    def test_float32_readonly_input_K_is_not_shared_with_mutating_provider(self):
        K = self.milk_K().astype(np.float32)
        K.setflags(write=False)
        before = K.tobytes()
        provider = DeployedInferFake()
        image = Image.new('RGB', (960, 960))
        first = infer_unidepth(provider, image, torch.device('cpu'), K)
        second = infer_unidepth(provider, image, torch.device('cpu'), K)
        self.assertEqual(K.tobytes(), before)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_resolution_level(self):
        self.check_geometry(641, 959, level=2)

    def test_invalid_K_does_not_call_provider(self):
        bad = [np.eye(4), np.zeros((3, 3)), np.eye(3)*np.nan]
        for i, j, value in [(0, 0, -1), (1, 1, 0), (0, 1, .1), (2, 0, .1), (2, 2, 2), (0, 0, 1e300)]:
            K = np.eye(3)
            K[i, j] = value
            bad.append(K)
        for K in bad:
            provider = DeployedInferFake()
            with self.subTest(K=K), self.assertRaises(ValueError):
                infer_unidepth(provider, Image.new('RGB', (20, 20)), torch.device('cpu'), K)
            self.assertFalse(hasattr(provider, 'called_camera'))

    def test_unsupported_provider_and_config_fail_clearly(self):
        for changes in [{'infer': None}, {'encode_decode': None}, {'interpolation_mode': 'bicubic'},
                        {'shape_constraints': {}}, {'resolution_level': 10},
                        {'shape_constraints': dict(DeployedInferFake.shape_constraints, shape_mult=16)},
                        {'shape_constraints': dict(DeployedInferFake.shape_constraints, ratio_bounds=[float('nan'), 2.5])}]:
            provider = DeployedInferFake()
            for key, value in changes.items():
                setattr(provider, key, value)
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, 'unsupported UniDepthV2'):
                infer_unidepth(provider, Image.new('RGB', (20, 20)), torch.device('cpu'))
        with self.assertRaises(ValueError):
            infer_unidepth(DeployedInferFake(), Image.new('RGB', (1, 20)), torch.device('cpu'))

    def test_missing_head_K_does_not_fall_back_to_supplied_K(self):
        provider = DeployedInferFake()
        original = provider.infer
        def missing(rgb, camera=None):
            out = original(rgb, camera)
            del out['intrinsics']
            return out
        provider.infer = missing
        with self.assertRaisesRegex(RuntimeError, 'missing camera-head intrinsics'):
            infer_unidepth(provider, Image.new('RGB', (20, 20)), torch.device('cpu'), np.eye(3))

    def test_unconditioned_export_resizes_radius_and_integer_center_K(self):
        provider = DeployedInferFake()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = Image.new('RGB', (32, 24), (50, 100, 150))
            image.save(root/'rgb.png')
            (root/'manifest.json').write_text(json.dumps({'frames': [{'frame_idx': 0, 'rgb': str(root/'rgb.png')}]}))
            args = argparse.Namespace(cpu=True, unidepth_repo=None, model_id='fake',
                                      manifest=root/'manifest.json', frame_start=0, frame_end=0,
                                      camera_contract=None, output_dir=root/'output',
                                      remote_root=None, local_root=None, source_width=47,
                                      source_height=39, min_valid_pixels=1)
            with patch('observations.camera.unidepth.load_model', return_value=provider), patch('builtins.print'):
                report = run(args)
            _, K, _, radius, confidence = infer_unidepth(provider, image, torch.device('cpu'))
            expected_K = np.array([K[0, 0]*47/32, K[1, 1]*39/24,
                                   (K[0, 2]+.5)*47/32-.5, (K[1, 2]+.5)*39/24-.5])
            with np.load(report['depth_archive']) as archive:
                np.testing.assert_allclose(archive['intrinsics_fx_fy_cx_cy'][0], expected_K)
                expected_depth = metric_radius_to_camera_z(resize_depth(radius, (39, 47)), expected_K)
                np.testing.assert_array_equal(archive['depth'][0], expected_depth.astype(np.float16))
                np.testing.assert_array_equal(archive['confidence'][0], resize_depth(confidence, (39, 47)).astype(np.float16))


if __name__ == '__main__':
    unittest.main()
