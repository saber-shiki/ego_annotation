"""CPU contracts for real forward/reverse SAM2 traversal wiring, no weights."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from observations.object import sam2


class FakePredictor:
    def __init__(self, first=2, count=5):
        self.first, self.count = first, count
        self.calls = []
        self.seed_frames = []
        self.omit = set()
        self.bad_index = None

    def logits(self, positive=True):
        value = torch.full((1, 1, 64, 64), -1.)
        if positive:
            value[:, :, 10:30, 10:30] = 1.
        return value

    def init_state(self, **kwargs):
        self.count = len(list(Path(kwargs['video_path']).glob('*.jpg')))
        return {'fixture': True}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **kwargs):
        self.seed_frames.append(frame_idx)
        self.first = min(self.seed_frames)
        return frame_idx, [obj_id], self.logits()

    def propagate_in_video(self, state, **kwargs):
        self.calls.append(kwargs)
        reverse = kwargs.get('reverse', False)
        start = kwargs.get('start_frame_idx', self.first)
        indices = range(start, -1, -1) if reverse else range(start, self.count)
        if self.bad_index is not None:
            yield self.bad_index, [1], self.logits()
            return
        for index in indices:
            if index not in self.omit:
                # Deliberately different reverse anchor proves forward preservation.
                yield index, [1], self.logits(not (reverse and index == self.first))


class Sam2PrefixTests(unittest.TestCase):
    def infer(self, predictor, start=0):
        with contextlib.redirect_stdout(io.StringIO()):
            return sam2.propagate_full_timeline(predictor, {},
                sam2.selected_frames(start, start + predictor.count - 1), [start + predictor.first])

    def test_late_first_prompt_infers_prefix_without_reseeding(self):
        p = FakePredictor(first=27, count=150)
        outputs, report = self.infer(p)
        self.assertEqual(set(outputs), set(range(150)))
        self.assertEqual(p.calls, [{}, {'start_frame_idx': 27,
            'max_frame_num_to_track': 27, 'reverse': True}])
        self.assertTrue(all(outputs[i][1].sum() == 400 for i in range(150)))
        self.assertEqual(report['frame_counts'], {'forward': 123, 'reverse_prefix': 27})
        self.assertEqual(p.seed_frames, [])

    def test_zero_first_prompt_preserves_forward_only_path(self):
        p = FakePredictor(first=0)
        outputs, report = self.infer(p)
        self.assertEqual(p.calls, [{}])
        self.assertEqual(report['frame_counts'], {'forward': 5})
        self.assertEqual(len(outputs), 5)

    def test_nonzero_source_indices_use_local_reverse_index(self):
        outputs, report = self.infer(FakePredictor(), start=100)
        self.assertEqual(set(outputs), {100, 101, 102, 103, 104})
        self.assertEqual(report['first_prompt_local_frame'], 2)
        self.assertEqual(report['first_prompt_source_frame'], 102)
        self.assertEqual(int(outputs[102][1].sum()), 400)

    def test_uncomputed_frame_fails_instead_of_becoming_invisible(self):
        for omitted in [0, 3]:
            p = FakePredictor(); p.omit.add(omitted)
            with self.subTest(omitted=omitted), self.assertRaisesRegex(RuntimeError, 'did not infer'):
                self.infer(p)

    def test_empty_model_mask_remains_a_valid_nonvisibility_measurement(self):
        p = FakePredictor(first=0)
        p.logits = lambda positive=True: torch.full((1, 1, 64, 64), -1.)
        outputs, report = self.infer(p)
        self.assertTrue(report['full_timeline_inferred'])
        self.assertTrue(all(not row[1].any() for row in outputs.values()))

    def test_invalid_provider_index_or_timeline_fails(self):
        p = FakePredictor(); p.bad_index = 5
        with self.assertRaisesRegex(RuntimeError, 'invalid or duplicate'):
            self.infer(p)
        for frames, seeds in [([], [0]), ([{'frame_idx': 1}, {'frame_idx': 1}], [1]),
                               ([{'frame_idx': 1}], [0])]:
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                sam2.propagate_full_timeline(p, {}, frames, seeds)

    def test_duplicate_and_out_of_prefix_provider_results_fail(self):
        class InvalidPredictor(FakePredictor):
            def propagate_in_video(self, state, **kwargs):
                yield from super().propagate_in_video(state, **kwargs)
                if kwargs.get('reverse'):
                    yield 4, [1], self.logits()
        with self.assertRaisesRegex(RuntimeError, 'escaped'):
            self.infer(InvalidPredictor())

        class DuplicatePredictor(FakePredictor):
            def propagate_in_video(self, state, **kwargs):
                yield from super().propagate_in_video(state, **kwargs)
                yield self.first, [1], self.logits()
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            self.infer(DuplicatePredictor())

    def test_actual_worker_writes_prefix_masks_reports_and_full_video(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);video=root/'input.mp4';out=root/'output';prompts=root/'prompts/object'
            prompts.mkdir(parents=True)
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30.,(64,64))
            self.assertTrue(writer.isOpened())
            for i in range(5): writer.write(np.full((64,64,3),40+i*20,dtype=np.uint8))
            writer.release()
            payload={'track_id':'track','description':'synthetic object',
                'point_coordinate_frame':'source_video_pixels',
                'object_plan_payload':{'active_intervals':[{'start_frame':0,'end_frame':4}]},
                'point_prompts':[{'frame_idx':2,'target_visible':True,
                    'box_xyxy':[10,10,30,30],'positive_points':[],'negative_points':[]}]}
            (prompts/'object_point_prompts_vlm.json').write_text(json.dumps(payload))
            args=SimpleNamespace(clip=video,point_root=prompts.parent,output_root=out,
                frame_start=0,frame_end=4,sam2_image_width=64,render_width=64,
                model_cfg='synthetic_only',checkpoint=root/'not-loaded.pt',use_positive_prompt_box=False,
                prompt_box_pad_ratio=.18,prompt_box_min_pad_px=24.)
            predictor=FakePredictor();build=types.ModuleType('sam2.build_sam')
            build.build_sam2_video_predictor=lambda *a,**kw: predictor
            with contextlib.ExitStack() as cleanup, \
                 patch.object(sys, 'path', sys.path[:]), \
                 patch.dict(sys.modules, {'sam2':types.ModuleType('sam2'),'sam2.build_sam':build}), \
                 patch.dict('os.environ',{'EGO_SAM2_SOURCE':str(root)}), \
                 patch('torch.cuda.is_available',return_value=True), \
                 patch('torch.autocast',side_effect=lambda *a,**kw: contextlib.nullcontext()), \
                 contextlib.redirect_stdout(io.StringIO()):
                for backend in (torch.backends.cuda.matmul, torch.backends.cudnn):
                    cleanup.callback(setattr, backend, 'allow_tf32', backend.allow_tf32)
                summary=sam2.run(args)
            self.assertEqual(predictor.seed_frames,[2])
            self.assertEqual(summary['propagation']['frame_counts'],{'forward':3,'reverse_prefix':2})
            track=json.loads((out/'track/sam2/sam2_track.json').read_text())
            self.assertEqual(list(map(int,track)),list(range(5)))
            self.assertTrue(all(row['visible'] and Path(row['mask_path']).is_file() for row in track.values()))
            cap=cv2.VideoCapture(str(out/'sam2_multiobject_overlay.mp4'))
            self.assertTrue(cap.isOpened());self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),5)
            cap.release()


if __name__ == '__main__':
    unittest.main()
