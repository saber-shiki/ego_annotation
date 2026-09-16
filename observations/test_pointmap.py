"""Source-named CPU tests for the shared-metric pointmap producer and official consumer."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from gates.camera.calibration import resize_affine
from observations.object.pointmap import build_pointmap, dense_pointmap, load_pointmap, file_record
from observations.object.sam3d import infer_shared_pointmap, run_one_case, pointmap_case_bindings
from observations.object.shape import validate_pointmap_conditioning, sam3d_candidate_record
from utils.state import sha


class FakePipeline:
    """CPU double of the observed official run->compute->preprocess->sample->pose route."""
    clip_pointmap_beyond_scale = None

    def __init__(self, failure=None):
        self.failure = failure
        self.ss_preprocessor = object()
        self.seen = None

    def depth_model(self, image):
        raise AssertionError('internal depth must never run')

    def compute_pointmap(self, image, pointmap=None):
        self.seen = pointmap.numpy().copy()
        points = pointmap.permute(2, 0, 1)
        if self.failure == 'compute_replace':
            points = points + 1
        return {'pointmap': points, 'pts_color': torch.tensor(image[..., :3].copy()),
                'intrinsics': torch.diag(torch.tensor([-0.3, -0.4, 1.0]))}

    def preprocess_image(self, image, preprocessor, pointmap=None):
        shift = pointmap.mean(dim=(1, 2))[None]
        scale = torch.ones((1, 1), dtype=torch.float32)
        result = {'pointmap': pointmap[None], 'rgb_pointmap': pointmap[None],
            'pointmap_scale': scale, 'pointmap_shift': shift,
            'rgb_pointmap_scale': scale, 'rgb_pointmap_shift': shift}
        if self.failure == 'no_normalization':
            result.pop('pointmap_scale')
        return result

    def sample_sparse_structure(self, input_dict):
        return {}

    def pose_decoder(self, result, scene_scale=None, scene_shift=None):
        return {}

    def run(self, image, mask=None, seed=None, with_layout_postprocess=False, pointmap=None, estimate_plane=False):
        if self.failure == 'internal_depth':
            self.depth_model(image)
        data = self.compute_pointmap(image, None if self.failure == 'omit_compute' else pointmap)
        if self.failure == 'active_K':
            data['intrinsics']
        p = data['pointmap']
        inputs = self.preprocess_image(image, self.ss_preprocessor, pointmap=p)
        out = self.sample_sparse_structure(inputs)
        if self.failure != 'omit_pose':
            shift = inputs['pointmap_shift'] + (1 if self.failure == 'pose_replace' else 0)
            self.pose_decoder(out, scene_scale=inputs['pointmap_scale'], scene_shift=shift)
        mesh = SimpleNamespace(vertices=np.array([[0., 0., 0.], [0.1, 0., 0.], [0., 0.1, 0.]]),
                               faces=np.array([[0, 1, 2]]))
        return {'mesh': [mesh], 'rotation': [1, 0, 0, 0], 'translation': [0, 0, 1], 'scale': [1],
                'glb': SimpleNamespace(export=lambda path: Path(path).write_bytes(b'fake glb'))}


class FakeInference:
    def __init__(self, failure=None):
        self._pipeline = FakePipeline(failure)
        self.failure = failure

    def __call__(self, image, mask, seed=None, pointmap=None):
        rgba = np.concatenate((image, (mask.astype(np.uint8)*255)[..., None]), axis=-1)
        if self.failure == 'rgb_replace':
            rgba = rgba.copy()
            rgba[0, 0, 0] += 1
        if self.failure == 'omit_public':
            pointmap = None
        if self.failure == 'replace_public':
            pointmap = torch.zeros_like(pointmap)
        return self._pipeline.run(rgba, seed=seed, pointmap=pointmap,
                                  with_layout_postprocess=self.failure == 'layout')


class PointmapTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.camera = self.root/'camera.json'
        self.depth = self.root/'depth.npz'
        self.manifest = self.root/'manifest.json'
        self.mask_source = self.root/'annotations.json'
        self.K = np.array([4., 6., 2.3, 2.8])
        ids = [7, 103]
        contract = {'schema': 'v19_camera_image_transform_contract_v2',
            'calibration_authority': 'estimated_from_rgb_depth_model',
            'intrinsics_model': 'constant_pinhole_fx_fy_cx_cy', 'intrinsics_coordinate_plane': 'calibration',
            'intrinsics_fx_fy_cx_cy': self.K.tolist(), 'frame_ids': ids,
            'calibration_plane': {'width': 8, 'height': 6},
            'image_planes': {'source_rgb': {'width': 8, 'height': 6,
                'A_plane_from_calibration': np.eye(3).tolist(), 'pixel_center_convention': 'integer_pixel_centers_opencv'}}}
        self.camera.write_text(json.dumps(contract))
        self.images = {}
        self.masks = {}
        rows = []
        for i, frame in enumerate(ids):
            image = self.root/f'rgb_{frame}.png'
            mask = self.root/f'mask_{frame}.png'
            Image.fromarray(np.full((3, 4, 3), 20+i, np.uint8)).save(image)
            mask_pixels = np.full((3, 4), 255, np.uint8)
            mask_pixels[0, 0] = 0 if i else 255
            Image.fromarray(mask_pixels).save(mask)
            self.images[frame], self.masks[frame] = image, mask
            rows.append({'frame_idx': frame, 'raw_frame_path': str(image), 'raw_frame_sha256': sha(image),
                         'source_width': 8, 'source_height': 6, 'manifest_width': 4, 'manifest_height': 3})
        self.manifest.write_text(json.dumps({'frames': rows}))
        self.mask_source.write_text(json.dumps({'frames': [{'frame_idx': frame,
            'objects': [{'object_id': 'object-A', 'mask_path': str(self.masks[frame])}]} for frame in ids]}))
        self.depth_values = np.stack([np.ones((6, 8), np.float32), np.full((6, 8), 2, np.float32)])
        self.depth_payload = dict(frame_idx=ids, depth=self.depth_values, source_size=[8, 6],
            intrinsics_fx_fy_cx_cy=np.tile(self.K, (2, 1)),
            source_depth_active_intrinsics_fx_fy_cx_cy=np.tile(self.K, (2, 1)),
            camera_contract_path=str(self.camera), camera_contract_sha256=sha(self.camera),
            camera_contract_plane='source_rgb', A_depth_from_calibration=np.eye(3),
            metadata_only_ray_relabel_override=False, depth_output_quantity='camera_z_m_from_model_inferred_rays')
        np.savez(self.depth, **self.depth_payload)

    def build(self, frame=7, **kwargs):
        args = dict(run_root=self.root, frame_idx=frame, image_path=self.images[frame], mask_path=self.masks[frame],
            depth_npz=self.depth, camera_contract=self.camera, raw_frame_manifest=self.manifest,
            output_dir=self.root/f'pointmap_{frame}', object_id='object-A', mask_source=self.mask_source)
        args.update(kwargs)
        return build_pointmap(**args)

    def consume(self, receipt, frame=7, inference=None):
        image, mask = self.images[frame], self.masks[frame]
        return infer_shared_pointmap(inference or FakeInference(), np.asarray(Image.open(image)),
            np.asarray(Image.open(mask)) > 0, 42, receipt_path=receipt, frame_idx=frame, image_path=image, mask_path=mask)

    def test_dense_half_pixel_mapping_and_single_axis_conversion(self):
        depth = np.arange(48, dtype=np.float32).reshape(6, 8)*0.01+1
        points, K, affine = dense_pointmap(depth, self.K, (4, 3))
        np.testing.assert_array_equal(affine, resize_affine((8, 6), (4, 3)))
        np.testing.assert_allclose(K, [2, 3, 0.9, 1.15])
        self.assertAlmostEqual(points[0, 0, 2], float(depth[:2, :2].mean()), places=6)
        self.assertGreater(points[0, 0, 0], 0)  # left-of-center -> positive P3D x
        y, x = np.indices((3, 4))
        np.testing.assert_allclose(-points[..., 0]/points[..., 2]*K[0]+K[2], x, atol=1e-6)
        np.testing.assert_allclose(-points[..., 1]/points[..., 2]*K[1]+K[3], y, atol=1e-6)
        for value in (np.nan, np.inf, 0., -1.):
            bad = depth.copy(); bad[0, 0] = value
            with self.assertRaises(ValueError):
                dense_pointmap(bad, self.K, (4, 3))
        with self.assertRaises(RuntimeError):
            dense_pointmap(depth, [-4, 6, 2.3, 2.8], (4, 3))

    def test_two_anchors_official_consumer_exact_points_and_diagnostic_negative_K(self):
        for frame in (7, 103):
            if frame == 103:
                appearance = self.root/'appearance.json'
                appearance.write_text(json.dumps({'frames': [{'frame_idx': frame, 'object_id': 'object-A',
                    'appearance_mask_path': str(self.masks[frame])}]}))
                receipt = self.build(frame, mask_source=appearance)
            else:
                receipt = self.build(frame)
            inference = FakeInference()
            original = inference._pipeline.compute_pointmap
            _, consumption = self.consume(receipt, frame, inference)
            expected = np.load(receipt.with_name('pointmap.npy'))
            np.testing.assert_array_equal(inference._pipeline.seen, expected)
            self.assertEqual(inference._pipeline.compute_pointmap, original)
            self.assertFalse(consumption['inferred_K_positive_focal'])
            self.assertFalse(consumption['inferred_K_read_by_generation'])
            self.assertEqual(consumption['depth_model_calls'], 0)
            self.assertEqual(consumption['frame_idx'], frame)
            self.assertEqual(float(expected[0, 0, 2]), 1 if frame == 7 else 2)
            case = {'geometry_profile': 'shared_metric_pointmap', 'external_pointmap': consumption}
            self.assertEqual(validate_pointmap_conditioning(case, self.images[frame], self.masks[frame],
                expected_frame_idx=frame, expected_pointmap_receipt=receipt,
                expected_geometry_profile='shared_metric_pointmap'), consumption)

    def test_consumer_rejects_omitted_replaced_or_inactive_points_and_restores_hooks(self):
        receipt = self.build()
        for failure in ('omit_public', 'replace_public', 'omit_compute', 'compute_replace', 'no_normalization',
                        'omit_pose', 'pose_replace', 'rgb_replace', 'active_K', 'layout', 'internal_depth'):
            with self.subTest(failure=failure):
                inference = FakeInference(failure)
                original = inference._pipeline.run
                with self.assertRaises(ValueError):
                    self.consume(receipt, inference=inference)
                self.assertEqual(inference._pipeline.run, original)

    def test_source_frame_shape_path_hash_failures(self):
        receipt = self.build()
        for frame, image, mask in ((103, self.images[7], self.masks[7]),
                                   (7, self.images[103], self.masks[7]), (7, self.images[7], self.masks[103])):
            with self.assertRaises(ValueError):
                load_pointmap(receipt, frame_idx=frame, image_path=image, mask_path=mask)
        for target in (self.depth, self.camera, self.manifest, self.mask_source, self.images[7], self.masks[7]):
            with self.subTest(target=target):
                data = target.read_bytes()
                target.write_bytes(data+b' ')
                with self.assertRaises(ValueError):
                    self.consume(receipt)
                target.write_bytes(data)
        payload = receipt.with_name('pointmap.npy')
        data = payload.read_bytes()
        np.save(payload, np.zeros((2, 2, 3), np.float32))
        value = json.loads(receipt.read_text())
        value['pointmap'] = file_record(payload)  # Even restamping cannot pass source replay.
        receipt.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.consume(receipt)
        payload.write_bytes(data)

    def test_builder_rejects_wrong_frame_rgb_camera_rays_and_shape(self):
        with self.assertRaises(ValueError):
            self.build(image_path=self.images[103])
        with self.assertRaises(ValueError):
            self.build(mask_path=self.masks[103])
        with self.assertRaises(ValueError):
            self.build(object_id='wrong-object')
        Image.fromarray(np.ones((4, 4), np.uint8)).save(self.masks[7])
        with self.assertRaises(ValueError):
            self.build()
        Image.fromarray(np.full((3, 4), 255, np.uint8)).save(self.masks[7])
        for key, value in (('frame_idx', [103, 7]), ('camera_contract_sha256', '0'*64),
                           ('metadata_only_ray_relabel_override', True), ('depth_output_quantity', 'radius_m'),
                           ('intrinsics_fx_fy_cx_cy', np.tile(self.K*2, (2, 1)))):
            with self.subTest(key=key):
                np.savez(self.depth, **{**self.depth_payload, key: value})
                with self.assertRaises((ValueError, RuntimeError)):
                    self.build()
        np.savez(self.depth, **self.depth_payload)
        receipt = self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertTrue(receipt.is_file())

    def test_candidate_receipt_rejects_omission_tampering_and_expected_profile_mismatch(self):
        receipt = self.build()
        _, consumption = self.consume(receipt)
        case = {'geometry_profile': 'shared_metric_pointmap', 'external_pointmap': consumption}
        for key, value in (('pointmap_preserved_exactly', False), ('frame_idx', 103),
                           ('with_layout_postprocess', True), ('pointmap_array_sha256', '0'*64),
                           ('inferred_K_read_by_generation', True), ('normalized_conditioning', {})):
            with self.subTest(key=key):
                changed = {**case, 'external_pointmap': {**consumption, key: value}}
                with self.assertRaises(ValueError):
                    validate_pointmap_conditioning(changed, self.images[7], self.masks[7], expected_frame_idx=7)
        for changed in ({'geometry_profile': 'shared_metric_pointmap'}, {}):
            with self.assertRaises(ValueError):
                validate_pointmap_conditioning(changed, self.images[7], self.masks[7],
                                              expected_geometry_profile='shared_metric_pointmap')
        self.assertIsNone(validate_pointmap_conditioning({}, self.images[7], self.masks[7]))

    def test_run_one_case_and_candidate_record_include_real_consumption(self):
        receipt = self.build()
        offline = {'moge_offline': {'checkpoint_sha256': 'da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f', 'network_resolution_allowed': False},
            'dinov2_offline': {'checkpoint_sha256': '36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51',
                'hubconf_sha256': 'c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64', 'network_resolution_allowed': False},
            'huggingface_offline': {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'network_resolution_allowed': False}}
        args = argparse.Namespace(geometry_profile='shared_metric_pointmap', pointmap_case=[f'case|7|{receipt}'],
            min_mask_pixels=1, mesh_name='mesh.ply', native_p3d_mesh_name='p3d.ply', native_opencv_mesh_name='cv.ply',
            min_native_positive_depth_fraction=0.99, glb_name='mesh.glb', gaussian_name='gs.ply', offline_asset_binding=offline)
        report = run_one_case(FakeInference(), self.root/'out', 'case', self.images[7], self.masks[7], 42, args)
        self.assertTrue(report['external_pointmap']['pointmap_preserved_exactly'])
        candidate = sam3d_candidate_record(self.root/'out/case/qc_sam3d_objects_mesh_v7.json', 'case', self.images[7],
            self.masks[7], {}, expected_frame_idx=7, expected_pointmap_receipt=receipt, expected_geometry_profile='shared_metric_pointmap')
        self.assertEqual(candidate['conditioning']['external_pointmap'], report['external_pointmap'])
        self.assertFalse(candidate['semantics']['collision_surface_ready'])

    def test_legacy_cli_default_unchanged_and_no_pointmap_fallback(self):
        self.assertEqual(pointmap_case_bindings(argparse.Namespace()), {})
        with self.assertRaises(ValueError):
            pointmap_case_bindings(argparse.Namespace(geometry_profile='shared_metric_pointmap'))
        with self.assertRaises(ValueError):
            pointmap_case_bindings(argparse.Namespace(pointmap_case=['case|7|foo']))


if __name__ == '__main__':
    unittest.main()
