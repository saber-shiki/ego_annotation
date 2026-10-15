"""Read-only, prediction-side mask/motion audit and full-video comparisons.

Bootstrap is evaluated with the SOURCE mesh, never the scaled final mesh.
All output is written into a fresh review directory, not prediction/source roots.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def stats(values):
    x = np.asarray([v for v in values if v is not None], float)
    if not len(x):
        return {'count': 0}
    return {'count': len(x), 'median': float(np.median(x)),
            'min': float(x.min()), 'max': float(x.max()),
            'p05': float(np.percentile(x, 5)), 'p95': float(np.percentile(x, 95))}


def hashes(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def summarize(scores):
    result = {domain + '_' + key: stats([x[domain][key] for x in scores])
              for domain in ('raw', 'visible') for key in ('iou', 'recall', 'precision')}
    result['raw_below_0_5'] = sum(x['raw']['iou'] < .5 for x in scores)
    if scores and 'boundary' in scores[0]:
        for key in ('precision', 'recall'):
            result['boundary_' + key] = stats([x['boundary'][key] for x in scores])
        result['visible_spill_pixels'] = stats([x['visible']['spill_pixels'] for x in scores])
    return result


class VideoWriter:
    """Browser-playable H.264 comparisons, using the project's existing ffmpeg."""
    def __init__(self, path, fps, size):
        self.pipe = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]}',
            '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '1',
            '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
            stdin=subprocess.PIPE)

    def isOpened(self):
        return self.pipe.poll() is None

    def write(self, image):
        self.pipe.stdin.write(np.ascontiguousarray(image).tobytes())

    def release(self):
        self.pipe.stdin.close()
        if self.pipe.wait():
            raise RuntimeError('Comparison video encoding failed')


def compare_videos(old, new, out, count, critical, source_masks):
    """Decode every frame of both actual triangle/MANO renders, without rerendering."""
    result = []
    for name in ('before_after_overlay', 'coarse_world'):
        caps = [cv2.VideoCapture(str(p / 'renders' / (name + '.mp4'))) for p in (old, new)]
        if not all(c.isOpened() for c in caps):
            raise RuntimeError('Cannot open full render: ' + name)
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        if abs(fps - caps[1].get(cv2.CAP_PROP_FPS)) > 1e-5:
            raise ValueError('Different video timestamps')
        writer = None
        mask_writer = None
        tiles = []
        sheets = []
        try:
            for i in range(count):
                frames = []
                for cap in caps:
                    ok, im = cap.read()
                    if not ok:
                        raise ValueError(f'Short render {name} at {i}')
                    frames.append(im[:, im.shape[1] // 2:] if name == 'before_after_overlay' else im)
                if frames[0].shape != frames[1].shape:
                    raise ValueError('Render dimensions differ')
                image = np.hstack(frames)
                # Keep full mesh/MANO panels, add labels in their existing header only.
                cv2.rectangle(image, (0, 0), (image.shape[1], 20), (0, 0, 0), -1)
                for j, label in enumerate(('jump_r2', 'mask experiment')):
                    cv2.putText(image, f'f{i} {label}', (5 + j * frames[0].shape[1], 14),
                                cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                if writer is None:
                    writer = VideoWriter(out / (name + '_comparison.mp4'), fps,
                                         (image.shape[1], image.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create comparison video')
                writer.write(image)
                if i in critical:
                    cv2.imwrite(str(out / f'{name}_{i:06d}.jpg'), image)
                if name == 'before_after_overlay':
                    # Visible raw SAM2 outlines are diagnostic, not amodal containment.
                    outlined = image.copy()
                    panel_width, panel_height = frames[0].shape[1], frames[0].shape[0] - 40
                    target = cv2.resize(source_masks[i].astype('uint8'), (panel_width, panel_height),
                                        interpolation=cv2.INTER_NEAREST_EXACT)
                    contours, _ = cv2.findContours(target, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    for j in (0, 1):
                        shifted = [c + np.array([[[j * panel_width, 40]]], dtype=c.dtype) for c in contours]
                        cv2.drawContours(outlined, shifted, -1, (40, 255, 60), 1)
                    if mask_writer is None:
                        mask_writer = VideoWriter(out / 'sam2_mesh_comparison.mp4', fps,
                                                  (image.shape[1], image.shape[0]))
                        if not mask_writer.isOpened():
                            raise RuntimeError('Cannot create mask comparison video')
                    mask_writer.write(outlined)
                    if i in critical:
                        cv2.imwrite(str(out / f'sam2_mesh_{i:06d}.jpg'), outlined)
                    tiles.append(cv2.resize(outlined, (320, round(image.shape[0] * 320 / image.shape[1]))))
                    if len(tiles) == 50 or i == count - 1:
                        while len(tiles) % 5:
                            tiles.append(np.zeros_like(tiles[0]))
                        sheet = np.vstack([np.hstack(tiles[k:k+5]) for k in range(0, len(tiles), 5)])
                        path = out / f'timeline_{len(sheets):02d}.jpg'
                        cv2.imwrite(str(path), sheet)
                        sheets.append(str(path))
                        tiles = []
            if any(cap.read()[0] for cap in caps):
                raise ValueError('Unexpected extra video frames')
        finally:
            for cap in caps:
                cap.release()
            if writer is not None:
                writer.release()
            if mask_writer is not None:
                mask_writer.release()
        result.append({'name': name, 'decoded_old_frames': count, 'decoded_new_frames': count,
                       'fps': fps, 'duration_s': count / fps, 'timeline_sheets': sheets})
    return result


def run(args):
    sys.path.insert(0, str(args.snapshot))
    from preprocessing.inputs import load_inputs
    from preprocessing.factors import hand_depths, mask_alignment
    from preprocessing.geometry import TriangleMesh
    cv2.setNumThreads(1)
    manifest = json.loads((args.snapshot / 'snapshot.json').read_text())
    expected = {str(args.snapshot / p): h for p, h in manifest['files'].items()}
    if hashes([Path(p) for p in expected]) != expected:
        raise RuntimeError('Snapshot hash mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    items = json.loads((args.run / 'inputs.json').read_text())
    if args.cases:
        selected = set(args.cases)
        if selected - {item['case'] for item in items}:
            raise ValueError('Unknown requested case')
        items = [item for item in items if item['case'] in selected]
    for item in items:
        case = item['case']
        new, old = args.run / case, args.oldrun / case
        if not all((p / 'completed.json').exists() for p in (old, new)):
            raise RuntimeError('Cannot evaluate unfinished predictions')
        out = args.output / case
        out.mkdir()
        files = [p for root in (old, new) for p in root.rglob('*') if p.is_file()]
        original_hashes = hashes(files)
        inp = load_inputs(Path(item['source']), 480, read_owned_masks=True)
        hd = hand_depths(inp, args.mask_width)
        zold = dict(np.load(old / 'coarse_scene.npz', allow_pickle=False))
        znew = dict(np.load(new / 'coarse_scene.npz', allow_pickle=False))
        seed_verified = False
        if (new / 'seed' / 'coarse_scene.npz').exists():
            seed = dict(np.load(new / 'seed' / 'coarse_scene.npz', allow_pickle=False))
            for key in ('object_rotation', 'object_translation_world', 'vertices_canonical', 'faces', 'uniform_object_scale'):
                np.testing.assert_array_equal(seed[key], zold[key], err_msg='Reproduced seed differs from frozen baseline: ' + key)
            seed_verified = True
        boots = [dict(np.load(root / 'bootstrap.npz', allow_pickle=False)) for root in (old, new)]
        for z in (zold, znew):
            np.testing.assert_array_equal(z['frame_idx'], inp.ids)
            np.testing.assert_array_equal(z['faces'], inp.faces)
            np.testing.assert_allclose(z['vertices_canonical'], inp.vertices * float(z['uniform_object_scale']), atol=1e-12)
            for side, hand in inp.hands.items():
                for key in ('vertices', 'joints', 'faces', 'valid'):
                    np.testing.assert_array_equal(z[side + '_' + key], hand[key])
        stages = [('old_bootstrap', boots[0], inp.vertices), ('new_bootstrap', boots[1], inp.vertices),
                  ('old_final', zold, zold['vertices_canonical']), ('new_final', znew, znew['vertices_canonical'])]
        scores = {}
        for label, poses, vertices in stages:
            mesh = TriangleMesh(vertices, inp.faces)
            scores[label] = [mask_alignment(mesh, poses['object_rotation'][i], poses['object_translation_world'][i], inp, hd, i)
                             for i in range(len(inp.ids))]
        per_frame = []
        for i, fid in enumerate(inp.ids):
            per_frame.append({'frame_idx': int(fid), 'position': i,
                              **{name: values[i] for name, values in scores.items()}})
        dump(out / 'per_frame.json', per_frame)
        delta = np.array([b['visible']['iou'] - a['visible']['iou'] for a, b in zip(scores['old_final'], scores['new_final'])])
        registration_delta = np.array([b['visible']['iou'] - a['visible']['iou'] for a, b in zip(scores['new_bootstrap'], scores['new_final'])])
        motion = {}
        for name, z in [('old', zold), ('new', znew)]:
            t = z['object_translation_world']; R = z['object_rotation']
            step = np.linalg.norm(np.diff(t, axis=0), axis=1) * 1000
            angle = np.degrees(Rotation.from_matrix(R[1:] @ R[:-1].transpose(0, 2, 1)).magnitude())
            motion[name] = {'translation_step_mm': stats(step), 'rotation_step_deg': stats(angle),
                            'translation_steps_mm': [0.] + step.tolist(), 'rotation_steps_deg': [0.] + angle.tolist()}
        worst = np.argsort(delta)[:5].tolist()
        worst += np.argsort([s['raw']['iou'] for s in scores['new_final']])[:5].tolist()
        critical = sorted(set([0, len(inp.ids)-1, 143, 148, 149] + worst))
        row = {'case': case, 'mask_width': args.mask_width, 'seed_bitwise_matches_baseline': seed_verified,
               **{name: summarize(value) for name, value in scores.items()},
               'old_scale': float(zold['uniform_object_scale']), 'new_scale': float(znew['uniform_object_scale']),
               'final_vs_old_visible_delta': stats(delta), 'regressed_vs_old_gt_015': int(np.sum(delta < -.015)),
               'registration_visible_delta': stats(registration_delta),
               'registration_regressed_gt_015': int(np.sum(registration_delta < -.015)),
               'motion': motion, 'critical_positions': critical}
        if (new / 'mask_repair.json').exists():
            repair = json.loads((new / 'mask_repair.json').read_text())
            from preprocessing.mask_contract import nonregression
            if len(repair['frames']) != len(inp.ids):
                raise ValueError('Incomplete mask repair report')
            changed = 0
            replay_differences = []
            for i, evidence in enumerate(repair['frames']):
                if evidence['frame_idx'] != int(inp.ids[i]):
                    raise ValueError('Mask repair frame mismatch')
                before, after = evidence['before'], evidence['after']
                if evidence.get('accepted_updates', 0):
                    changed += 1
                    ok, reason = nonregression(before['mask'], after['mask'])
                    if not ok:
                        raise ValueError('Recorded mask contract failure: ' + reason)
                if args.mask_width == repair['configuration']['registration_width']:
                    actual_before, actual_after = scores['old_final'][i], scores['new_final'][i]
                    # Exported poses are authoritative, not a camera-local solver report.
                    ok, why = nonregression(actual_before, actual_after)
                    if actual_before['visible']['available'] and not ok:
                        raise ValueError(f'Exported frame {evidence["frame_idx"]} violates mask contract: {why}')
                    if actual_after['visible']['iou'] < actual_before['visible']['iou'] - 1e-12:
                        raise ValueError('Exported pose lost seed visible IoU')
                    for domain in ('raw', 'visible', 'owned'):
                        for key in ('iou', 'recall', 'precision'):
                            recorded, replayed = after['mask'][domain][key], actual_after[domain][key]
                            if abs(recorded - replayed) > 1e-12:
                                replay_differences.append({'frame_idx': evidence['frame_idx'], 'domain': domain,
                                    'metric': key, 'reported': recorded, 'exported_pose_replay': replayed})
                if after['mask']['visible']['iou'] < before['mask']['visible']['iou'] - 1e-12:
                    raise ValueError('Final frame lost seed visible IoU')
            row['accepted_frame_refinements'] = changed
            row['reported_vs_exported_mask_differences'] = replay_differences
        row['videos'] = [] if args.no_videos else compare_videos(old, new, out, len(inp.ids), critical, inp.masks)
        row['surface'] = {name: json.loads((root / 'visible_surface_metrics.json').read_text()) for name, root in [('old', old), ('new', new)]}
        dump(out / 'summary.json', row)
        inp.receipt.check()
        if hashes(files) != original_hashes:
            raise RuntimeError('Prediction output changed during evaluation')
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('case', 'old_final', 'new_final', 'regressed_vs_old_gt_015', 'registration_regressed_gt_015')}), flush=True)
    if hashes([Path(p) for p in expected]) != expected:
        raise RuntimeError('Snapshot changed during evaluation')
    dump(args.output / 'summary.json', rows)
    if not args.no_videos:
        cards = []
        for row in rows:
            case = html.escape(row['case'], quote=True)
            old_iou, new_iou = (row[name]['raw_iou']['median'] for name in ('old_final', 'new_final'))
            cards.append(f'<section><h2>{case}</h2><p>Raw IoU median {old_iou:.3f} → {new_iou:.3f}; '
                f'visible IoU regressions &gt; .015: {row["regressed_vs_old_gt_015"]}. '
                'Left: jump_r2. Right: mask experiment. Green: visible SAM2, not an amodal mask.</p>'
                f'<video controls preload="metadata" src="{case}/sam2_mesh_comparison.mp4"></video>'
                f'<video controls preload="metadata" src="{case}/coarse_world_comparison.mp4"></video>'
                f'<p><a href="{case}/summary.json">Metrics</a> | <a href="{case}/per_frame.json">All frame evidence</a></p>'
                + ''.join(f'<a href="{case}/timeline_{i:02d}.jpg">Frames {i*50}–{min((i+1)*50-1, row["new_final"]["raw_iou"]["count"]-1)}</a> '
                          for i in range((row['new_final']['raw_iou']['count']+49)//50)) + '</section>')
        (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mask repair review</title>'
            '<style>body{font:16px sans-serif;max-width:1400px;margin:2em auto;background:#eee}'
            'section{background:white;padding:1em;margin:1em 0}video{width:49%;vertical-align:top}h2{font-size:16px}</style>'
            '<h1>Full triangle mesh / metric MANO comparisons</h1>'
            '<p>Prediction-side diagnostics only. No GT/CAD. World clips have independently framed fixed views; '
            'use the shared RGB view for pixel-alignment comparisons.</p>' + ''.join(cards))
    dump(args.output / 'completed.json', {'cases': len(rows), 'GT_read': False,
         'snapshot_verified': True, 'source_receipts_checked': True, 'prediction_hashes_unchanged': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'oldrun', 'snapshot', 'output'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--cases', nargs='+', help='Read only these completed case outputs')
    p.add_argument('--mask-width', type=int, default=192)
    p.add_argument('--no-videos', action='store_true', help='Additional-resolution audit only; primary review must decode videos')
    run(p.parse_args())
