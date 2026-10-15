"""Full-video visual review and fixed-population A/B/C geometry diagnostics (no GT)."""
import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(Path(path).read_text())


def review(root, output, cases=None):
    root = Path(root); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    results = []; page = ['<html><meta charset="utf-8"><body><h1>Frozen R6 / observed motion / +envelope</h1>']
    for item in read(root/'inputs.json'):
        if cases is not None and item['case'] not in cases:
            continue
        path = Path(item['output'])
        if not (path/'completed.json').is_file():
            raise ValueError('Case prediction is incomplete: '+item['case'])
        caseout = output/item['case']; caseout.mkdir()
        metrics = {arm: read(path/arm/'prediction_metrics.json') for arm in ('A', 'B', 'C')}
        arrays = {}
        for arm in metrics:
            with np.load(path/arm/'coarse_scene.npz', allow_pickle=False) as z:
                arrays[arm] = {k: np.array(z[k]) for k in ('object_rotation', 'object_translation_world', 'frame_idx')}
        comparisons = {}
        for left, right in [('A', 'B'), ('B', 'C')]:
            a, b = arrays[left], arrays[right]
            angle = np.degrees(Rotation.from_matrix(b['object_rotation']@a['object_rotation'].transpose(0, 2, 1)).magnitude())
            translation = np.linalg.norm(b['object_translation_world']-a['object_translation_world'], axis=1)*1000
            iou = np.array([r['mask']['visible']['iou'] for r in metrics[right]['frames']])-np.array([r['mask']['visible']['iou'] for r in metrics[left]['frames']])
            comparisons[left+'_to_'+right] = {'rotation_correction_median_deg': float(np.median(angle)),
                'rotation_correction_max_deg': float(angle.max()), 'translation_correction_median_mm': float(np.median(translation)),
                'translation_correction_max_mm': float(translation.max()), 'changed_frames': int(np.sum((angle>1e-5)|(translation>1e-5))),
                'visible_iou_gain_median': float(np.median(iou)), 'visible_iou_regressions_over_015': int(np.sum(iou<-.015)),
                'best_visible_frame': int(a['frame_idx'][np.argmax(iou)]), 'worst_visible_frame': int(a['frame_idx'][np.argmin(iou)]),
                'largest_rotation_frame': int(a['frame_idx'][np.argmax(angle)]), 'largest_translation_frame': int(a['frame_idx'][np.argmax(translation)])}
        ids = list(map(int, arrays['A']['frame_idx'])); cells = []
        for fid in ids:
            im = cv2.imread(str(path/'renders'/f'overlay_{fid:06d}.jpg'))
            if im is None:
                raise ValueError('Missing full-timeline image')
            # Five columns of three-arm thumbnails, three pages per complete case.
            cells.append(cv2.resize(im, (480, 150), interpolation=cv2.INTER_AREA))
        for offset in range(0, len(cells), 50):
            group = cells[offset:offset+50]
            while len(group)%5:
                group.append(np.full_like(cells[0], 255))
            sheet = np.vstack([np.hstack(group[j:j+5]) for j in range(0, len(group), 5)])
            cv2.imwrite(str(caseout/f'timeline_{offset:03d}.jpg'), sheet)
        selected = set([ids[0], ids[-1], ids[len(ids)//3], ids[2*len(ids)//3], *[i for i in (30, 60, 90, 120, 143, 148) if i in ids]])
        for c in comparisons.values():
            selected.update(c[k] for k in ('best_visible_frame', 'worst_visible_frame', 'largest_rotation_frame', 'largest_translation_frame'))
        for fid in sorted(selected):
            im = cv2.imread(str(path/'renders'/f'overlay_{fid:06d}.jpg')); cv2.imwrite(str(caseout/f'detail_{fid:06d}.jpg'), im)
        decoded = {}
        for video in sorted((path/'renders').glob('*.mp4')):
            cap = cv2.VideoCapture(str(video)); frames = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                frames += 1
            cap.release()
            if frames != len(ids):
                raise ValueError('Incomplete comparison video')
            decoded[video.name] = frames
        summary = {'case': item['case'], 'comparisons': comparisons, 'decoded_frames': decoded,
                   'arms': {arm: {'total_score': metrics[arm]['total'],
                       'raw_iou_median': float(np.median([r['mask']['raw']['iou'] for r in metrics[arm]['frames']])),
                       'visible_iou_median': float(np.median([r['mask']['visible']['iou'] for r in metrics[arm]['frames']])),
                       'first_hit_coverage_median': float(np.median([r['first_hit_coverage'] for r in metrics[arm]['frames'] if r['first_hit_coverage'] is not None])),
                       'envelope_inside_median': float(np.median([r['envelope_inside_fraction'] for r in metrics[arm]['frames'] if r['envelope_inside_fraction'] is not None]))} for arm in metrics}}
        results.append(summary); (caseout/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        page.append('<h2>'+html.escape(item['case'])+'</h2>')
        for video in sorted((path/'renders').glob('*.mp4')):
            page.append('<p>'+html.escape(video.name)+'</p><video controls width="960" src="'+html.escape(str(video))+'"></video>')
        for sheet in sorted(caseout.glob('timeline_*.jpg')):
            page.append('<img width="1200" src="'+html.escape(str(sheet.relative_to(output)))+'">')
        print(json.dumps(summary), flush=True)
    (output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    (output/'index.html').write_text('\n'.join(page+['</body></html>']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-root', type=Path, required=True); p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--case', action='append', help='Explicit completed-case QC subset; never a whole-batch claim')
    a = p.parse_args(); review(a.prediction_root, a.output_dir, a.case)
