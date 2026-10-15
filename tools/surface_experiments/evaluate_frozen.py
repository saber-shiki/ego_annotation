"""Post-freeze only: QA-qualified GT motion and unaligned full-surface evaluation.

This file is NEVER included in prediction snapshots. One A-derived object/world
gauge is shared across A/B/C. No arm-specific or per-frame GT alignment.
"""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def verify_freeze(root):
    root = Path(root)
    if (root/'batch.exit').read_text().strip() != '0':
        raise ValueError('Predictions have not all completed successfully')
    freeze = read(root/'prediction_freeze.json')
    bycase = {r['case']: r for r in freeze['cases']}
    for item in read(root/'inputs.json'):
        path = Path(item['output']); row = bycase[item['case']]
        if sha(path/'completed.json') != row['completed_sha256']:
            raise ValueError('Prediction completion changed after freeze')
        for arm in ('A', 'B', 'C'):
            manifest = read(path/arm/'coarse_scene.json')
            if not manifest['prediction_side_only'] or manifest['GT_or_CAD_inputs']:
                raise ValueError('GT-contaminated prediction protocol')
            for name, expected in [('coarse_scene.npz', row['arms'][arm]['npz_sha256']),
                                   ('coarse_scene.json', row['arms'][arm]['manifest_sha256'])]:
                if sha(path/arm/name) != expected:
                    raise ValueError('Frozen prediction changed: '+str(path/arm/name))
        for name, expected in row['render_sha256'].items():
            if sha(path/'renders'/name) != expected:
                raise ValueError('Frozen render changed')
    return freeze


def evaluate(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    freeze = verify_freeze(root)  # Barrier BEFORE importing/reading any GT.
    output.mkdir(parents=True, exist_ok=False)
    audit = Path('/mnt/user-home/kupingxin/ego_annotation_experiments/hot3d_s3_pose_audit_20260927')
    sys.path.insert(0, str(audit))
    from evaluate import load_case, pose, apply, sample, scene, distance, summ
    started = time.monotonic(); summary = []
    for item in read(root/'inputs.json'):
        case = item['case']; path = Path(item['output']); c = load_case(case)
        state = {}; prediction = {}
        for arm in ('A', 'B', 'C'):
            with np.load(path/arm/'coarse_scene.npz', allow_pickle=False) as z:
                state[arm] = {k: np.array(z[k]) for k in z.files}
            s = state[arm]; ids = list(map(int, s['frame_idx']))
            if ids != c['ids']:
                raise ValueError('GT/source timeline mismatch')
            np.testing.assert_allclose(s['T_world_camera'], np.array([c['C'][i] for i in ids]), atol=1e-10)
            np.testing.assert_array_equal(s['intrinsics_source'], np.array([c['Kgt'][i] for i in ids]))
            prediction[arm] = {i: pose(r, t) for i, r, t in zip(ids, s['object_rotation'], s['object_translation_world'])}
            if read(path/arm/'coarse_scene.json')['source_bundle_sha256'] != read(audit/case/'evaluation.json')['input_bundle_sha256']:
                raise ValueError('GT association to a different observation bundle')
        A = state['A']; vertices, faces = A['vertices_canonical'], A['faces']
        samples = sample(vertices, faces); gt_samples = sample(c['gv'], c['gf'])
        model_scene = scene(vertices, faces); gt_scene = scene(c['gv'], c['gf'])
        anchor = c['a']; world_gauge = c['D'][anchor] @ np.linalg.inv(c['C'][anchor])
        object_gauge = np.linalg.inv(world_gauge @ prediction['A'][anchor]) @ c['G'][anchor]
        metrics = {arm: read(path/arm/'prediction_metrics.json') for arm in ('A', 'B', 'C')}
        rows = []
        for ix, frame in enumerate(ids):
            row = {'frame_idx': frame, 'GT_quality_eligible': c['eligible'][frame]}
            G = np.linalg.inv(c['D'][frame]) @ c['G'][frame]
            for arm in ('A', 'B', 'C'):
                H = np.linalg.inv(c['C'][frame]) @ prediction[arm][frame]
                E = H @ object_gauge
                row[arm+'_camera_rotation_deg'] = float(np.degrees(Rotation.from_matrix(E[:3, :3]@G[:3, :3].T).magnitude()))
                row[arm+'_camera_translation_mm'] = float(np.linalg.norm(E[:3, 3]-G[:3, 3])*1000)
                # These distances do not use world_gauge/object_gauge at all.
                d1 = distance(gt_scene, apply(np.linalg.inv(G)@H, samples))*1000
                d2 = distance(model_scene, apply(np.linalg.inv(H)@G, gt_samples))*1000
                d = np.r_[d1, d2]
                row[arm+'_raw_surface_mean_mm'] = float(d.mean())
                row[arm+'_raw_surface_p95_mm'] = float(np.percentile(d, 95))
                row[arm+'_surface_within_5mm'] = float(np.mean(d <= 5))
                measured = metrics[arm]['frames'][ix]
                for domain in ('raw', 'visible'):
                    row[arm+'_'+domain+'_iou'] = measured['mask'][domain]['iou']
                row[arm+'_boundary_recall'] = measured['mask']['boundary']['recall']
                row[arm+'_first_hit_coverage'] = measured['first_hit_coverage']
                row[arm+'_first_hit_depth_mean_m'] = measured['first_hit_depth_mean_m']
                row[arm+'_envelope_inside_fraction'] = measured['envelope_inside_fraction']
            rows.append(row)
        qualified = [r for r in rows if r['GT_quality_eligible']]
        fields = ['camera_rotation_deg', 'camera_translation_mm', 'raw_surface_mean_mm', 'raw_surface_p95_mm',
                  'surface_within_5mm', 'raw_iou', 'visible_iou', 'boundary_recall', 'first_hit_coverage',
                  'first_hit_depth_mean_m', 'envelope_inside_fraction']
        arms = {arm: {field: summ([r[arm+'_'+field] for r in qualified if r[arm+'_'+field] is not None]) for field in fields} for arm in ('A', 'B', 'C')}
        paired = {}
        for left, right in [('A', 'B'), ('B', 'C')]:
            paired[left+'_to_'+right] = {field: summ([r[right+'_'+field]-r[left+'_'+field] for r in qualified if r[right+'_'+field] is not None and r[left+'_'+field] is not None]) for field in fields}
        corrections = {}
        for arm, base in [('B', 'A'), ('C', 'B')]:
            angle = np.degrees(Rotation.from_matrix(state[arm]['object_rotation'] @ state[base]['object_rotation'].transpose(0, 2, 1)).magnitude())
            delta = np.linalg.norm(state[arm]['object_translation_world']-state[base]['object_translation_world'], axis=1)*1000
            corrections[base+'_to_'+arm] = {'rotation_deg': summ(angle), 'translation_mm': summ(delta),
                'changed_frames': int(np.sum((angle>1e-5)|(delta>1e-5)))}
        result = {'case': case, 'all_frames': len(rows), 'GT_quality_frames': len(qualified), 'arms': arms,
                  'paired_changes': paired, 'actual_pose_changes': corrections,
                  'prediction_scores': {arm: {k: v for k, v in metrics[arm].items() if k!='frames'} for arm in metrics},
                  'observation_motion': read(path/'motion_observations.json'),
                  'GT_sidecar_sha256': sha(c['sidepath']), 'CAD_sha256': sha(c['cadpath']),
                  'common_motion_anchor': anchor, 'common_anchor_role': 'fixed_appearance_anchor_used_by_prediction', 'common_object_gauge': object_gauge.tolist(),
                  'metric_scope': 'A-derived single shared gauge for trajectory; no absolute-initial or symmetry-aware pose claim. Raw camera full-mesh/CAD surface has NO GT alignment; shape+scale+pose.',
                  'rows': rows}
        (output/(case+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        with (output/(case+'.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
        summary.append({k: v for k, v in result.items() if k not in ('rows', 'observation_motion', 'common_object_gauge')})
        print(json.dumps({'case': case, 'arms': {arm: {k: value['median'] for k, value in data.items() if k in fields[:3]} for arm, data in arms.items()}, 'pose_changes': corrections}), flush=True)
        del c, state, model_scene, gt_scene; gc.collect()
    verify_freeze(root)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output/'completed.json').write_text(json.dumps({'seconds': time.monotonic()-started, 'prediction_unchanged': True,
        'prediction_freeze_sha256': sha(root/'prediction_freeze.json'), 'cases': len(summary),
        'evaluator_sha256': sha(__file__)}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-root', type=Path, required=True); p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); evaluate(a.prediction_root, a.output_dir)
