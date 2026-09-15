"""Markdown report with per-sequence metrics, coverage and auditable protocol inputs."""
import json
from collections import defaultdict

import numpy as np

PROTOCOL_VERSION = 'hot3d-eval-v1-first-camera-se3'
HAND_KEYS = ['MPJPE_mm', 'PCK_AUC_pct', 'FINGERTIP_PCK_AUC_pct', 'MPVPE_mm',
             'PA_MPJPE_mm', 'W_MPJPE_mm', 'WA_MPJPE_mm', 'RTE_pct', 'Accel_m_s2', 'Jerk_3rd_diff_mm']
CAMERA_KEYS = ['ATE_first_mm', 'ATE_SE3_mm', 'ATE_Sim3_mm', 'RPE_T_mm', 'RPE_R_deg']


def number(x):
    return 'N/A' if x is None else f'{x:.6f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in row) + ' |' for row in rows])


def summary(results, group, key, reduction='mean'):
    # Clips within the same source sequence are combined before sequence weighting.
    values = defaultdict(list)
    for r in results:
        groups = r['hands'].values() if group == 'hands' else [r['camera']]
        for metrics in groups:
            v = metrics[key]['value']
            if v is not None:
                values[r['sequence']].append(v)
    means = [np.mean(v) for v in values.values()]
    return float(getattr(np, reduction)(means)) if means else None


def render_report(results):
    lines = ['# HOT3D evaluation', '', f'Protocol: `{PROTOCOL_VERSION}`', '',
             'World registration: `A = T_gt_wc(frames[0]) @ inv(T_pred_wc(frames[0]))`, once, without scale. '
             'A transforms every predicted camera, hand point and available visual object pose. '
             'No per-frame camera alignment. All unaligned scores use this GT-world gauge.', '',
             'Units: hand/camera distances mm except Accel (m/s²), AUC and RTE (%), RPE-R (degrees). '
             'Jerk (3rd-diff, mm) is an unnormalized third position difference. '
             'PCK uses ≤ thresholds 0:0.5:50 mm, trapezoidal integration / 50 × 100. '
             'MPVPE compares zero-articulation rest meshes (778 vertices, shape only).', '',
             'PA fits per-hand per-frame Sim3. W/WA use nonoverlapping 100-frame chunks of consecutive valid frames: '
             'W fits the first 2 × 21 points, WA all points in the chunk. Tails with ≥2 frames are retained; singletons are discarded. '
             'RTE fits the entire valid wrist trajectory with SE3, no scale; path length sums only consecutive valid edges. '
             'Accel uses prediction−GT second differences / (1/30)²; Jerk uses prediction third differences in mm, without dividing by dt³. '
             '**Jerk uses an explicit raw-third-difference convention, not Dyn-HaMR paper scaling: do not compare the numbers across papers.** '
             'Multiply the reported Jerk by 30³/1000 to obtain m/s³ at 30 Hz.', '',
             'Each table cell is a sequence/hand mean (camera ATE/RPE are sequence RMSE). '
             'Summary rows first average available hand/clip results within each source sequence, then weight source sequences equally. '
             'N/A is excluded from summaries, with coverage and reasons below. No visual-acceptance filtering.', '']
    for index, r in enumerate(results, 1):
        lines += [f'## Input {index}: {r["sequence"]}', '',
                  f'- Annotation: `{r["annotation"]}`',
                  f'- JSON SHA256: `{r["annotation_sha256"]}`',
                  f'- joints source: **{r["joints_source"]["description"]}**; archive SHA256: `{r["joints_source"].get("sha256", "not used")}`',
                  f'- Source frames: {r["source_indices"][0]}–{r["source_indices"][-1]} ({len(r["source_indices"])} evaluated frames).',
                  f'- Timestamp domain: TIME_CODE; mapping: {r["source"]["timestamp_route"]}; source route: {r["source"]["route"]}.',
                  f'- Provenance SHA256: `{r["source"]["source_sha256"]}`; sensor SHA256: `{r["source"]["sensor_sha256"]}`.',
                  f'- Manifest SHA256: `{r["source"]["manifest_sha256"]}`.',
                  f'- Projection gate: **PASS**, {len(r["gate"]["samples"])} samples; {r["gate"]["projection"]}.',
                  f'- Registered object poses: {r["registered_object_count"]}/{len(r["source_indices"])} (no object score specified).',
                  f'- Left shapedirs repair applied: {r["left_shapedirs_fixed"]}.']
        for side, provenance in r['mano'].items():
            lines.append(f'- MANO {side}: `{provenance["filename"]}`, SHA256 `{provenance["sha256"]}`.')
        for side, coverage in r['coverage'].items():
            lines.append(f'- {side} coverage: {coverage["valid"]}/{coverage["total"]}; GT available {coverage["gt_available"]}; '
                         f'预测缺 joints_world: {coverage["prediction_missing"]}.')
        lines += ['', '### Projection gate samples', '', table(
            ['Hand', 'Frame / source', 'TIME_CODE ns', 'Inside expanded box', 'Offset from box center (px)'],
            [[s['side'], f'{s["frame_idx"]} / {s["source_frame_idx"]}', s['timestamp_ns'], f'{s["inside"]}/{s["total"]}',
              str(s['mean_offset_from_box_center_px'])] for s in r['gate']['samples']]), '',
            '<details><summary>Projection points, boxes, frame mapping, registration and camera timestamp sampling</summary>', '',
            '```json', json.dumps({'gate': r['gate'], 'A': r['A'], 'source_frame_indices': r['source_indices'],
                                   'timestamps_ns': r['timestamps'], 'camera_sampling': r['camera_sampling']}, indent=2), '```', '', '</details>', '']
    lines += ['## Hand metrics', '']
    for keys, title in [(HAND_KEYS[:4], 'Official 19-landmark / shape group'), (HAND_KEYS[4:], '21-landmark temporal / alignment group')]:
        rows = [[r['sequence'], str(i), side] + [number(m[k]['value']) for k in keys]
                for i, r in enumerate(results, 1) for side, m in r['hands'].items()]
        rows.append(['**Sequence-equal mean**', '—', 'both'] + [number(summary(results, 'hands', k)) for k in keys])
        labels = {'MPVPE_mm': 'MPVPE (official, constant-β, mm)', 'Jerk_3rd_diff_mm': 'Jerk (3rd-diff, mm)'}
        lines += [f'### {title}', '', table(['Sequence', 'Input', 'Hand'] + [labels.get(k, k) for k in keys], rows), '']
        for i, r in enumerate(results, 1):
            lines.append(f'Input {i} joints source: {r["joints_source"]["description"]}.')
        lines.append('')
    diagnostic_rows = []
    for i, r in enumerate(results, 1):
        for side, metrics in r['hands'].items():
            d = metrics.get('MPVPE_mean_beta_diagnostic_mm')
            if d is not None:
                diagnostic_rows.append([r['sequence'], i, side, number(d['value']), f'{d["valid"]}/{d["total"]}', d['reason']])
    if diagnostic_rows:
        lines += ['### MPVPE(mean-β, diagnostic) — non-protocol', '',
                  'These values do not enter the official MPVPE summary. Each hand has its own vertex-comparison denominator.', '',
                  table(['Sequence', 'Input', 'Hand', 'MPVPE mm', 'Numerator / denominator', 'Protocol deviation'], diagnostic_rows), '',
                  '### Findings: prediction shape varies with time', '',
                  'The annotation carries frame-varying source_betas rather than a sequence-constant hand shape. '
                  + ('The recorded hand-model provenance names HaWoR; its per-frame β values were retained without an exported temporal canonical β. '
                     if any(r['hawor_provenance'] for r in results) else '') +
                  'Official-comparable MPVPE requires the prediction pipeline to select and export a sequence-constant canonical β. '
                  'This evaluator leaves the pipeline and frozen annotations unchanged.', '']
    rows = [[r['sequence'], str(i)] + [number(r['camera'][k]['value']) for k in CAMERA_KEYS] for i, r in enumerate(results, 1)]
    for reduction in ['mean', 'median']:
        rows.append([f'**Sequence {reduction}**', '—'] + [number(summary(results, 'camera', k, reduction)) for k in CAMERA_KEYS])
    lines += ['## Camera metrics', '', table(['Sequence', 'Input'] + CAMERA_KEYS, rows), '', '## Per-metric coverage', '',
              'Numerator = scored point samples (or trajectory poses/pairs); denominator = all annotation point/pose/window slots. '
              'MPVPE counts 778 vertex comparisons per valid shape sample. Temporal derivative slots require consecutive valid frame IDs.', '']
    rows = []
    for i, r in enumerate(results, 1):
        for side, metrics in list(r['hands'].items()) + [('camera', r['camera'])]:
            for key, m in metrics.items():
                rows.append([r['sequence'], i, side, key, f'{m["valid"]}/{m["total"]}', m['reason'] or '—'])
    lines += [table(['Sequence', 'Input', 'Entity', 'Metric', 'Numerator / denominator', 'Exclusion reason'], rows), '']
    return '\n'.join(lines)
