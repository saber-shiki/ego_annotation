"""Independent pose trajectory image judgment, fixed to gpt-6-astra."""
from utils.runtime import astra_request, judgment_cli

PROMPT = 'Inspect refined object trajectory against raw video, especially motion and occlusion transitions. Identify rotations or jumps inconsistent with visible evidence. Compact timeline values are posterior estimates, not ground truth. Review raw/candidate image pairs, including every listed unreliable interval and uncertain tail. Missing rotation observability is unknown, not observable. Do not clip poses or promote source/visual estimates into physical acceptance.'
SCHEMA = {'type': 'object',
 'properties': {'object_id': {'type': 'string', 'minLength': 1},
                'trajectory_consistent': {'type': 'boolean'},
                'flagged_intervals': {'type': 'array',
                                      'items': {'type': 'object',
                                                'properties': {'frame_start': {'type': 'integer',
                                                                               'minimum': 0},
                                                               'frame_end': {'type': 'integer',
                                                                             'minimum': 0},
                                                               'rationale': {'type': 'string',
                                                                             'minLength': 1},
                                                               'uncertainty': {'type': 'string',
                                                                               'minLength': 1}},
                                                'required': ['frame_start',
                                                             'frame_end',
                                                             'rationale',
                                                             'uncertainty'],
                                                'additionalProperties': False}},
                'evidence_frames': {'type': 'array',
                                    'items': {'type': 'integer', 'minimum': 0},
                                    'minItems': 1}},
 'required': ['object_id', 'trajectory_consistent', 'flagged_intervals', 'evidence_frames'],
 'additionalProperties': False}


COMPACT_SCHEMA = {
    'type': 'object', 'required': ['schema', 'source_posterior', 'frame_ids', 'timeline_columns', 'timeline', 'review_frame_ids', 'summary'],
    'properties': {'schema': {'const': 'trajectory_review_compact_v1'},
        'source_posterior': {'type': 'object', 'required': ['sha256', 'bytes']},
        'frame_ids': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 1},
        'timeline_columns': {'type': 'array', 'items': {'type': 'string'}},
        'timeline': {'type': 'array', 'items': {'type': 'array'}, 'minItems': 1},
        'review_frame_ids': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 1},
        'summary': {'type': 'object'}}}


def compact_trajectory_evidence(posterior_path):
    """All-frame facts, no solver traces/surfel arrays/repeated source rows.

    Values remain estimates. Missing observability remains null (never true).
    The uncertainty is the producer's heuristic support score, not covariance.
    """
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    from utils.data import RKEY, TKEY
    raw = Path(posterior_path).read_bytes()
    posterior = json.loads(raw)
    rows = posterior['pose_rows']
    ids = [int(row['frame_idx']) for row in rows]
    times = np.asarray([row['time_s'] for row in rows], float)
    if not ids or ids != sorted(set(ids)) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('compact trajectory requires unique chronological full timeline')
    sources = {r['frame_idx']: r for r in posterior['source_pose_rows']}
    if set(sources) != set(ids):
        raise ValueError('posterior/source timeline mismatch')
    def angle(a, b):
        return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.) / 2., -1., 1.))))
    columns = ['frame_idx', 'time_s', 'translation_world_m', 'rotation_world_from_canonical_matrix',
               'source_status', 'direct_pose_observation_source', 'metric_observation', 'rigid_pose_observation_eligible',
               'depth_feasibility_uncertain', 'support_uncertainty', 'relative_evidence_connected',
               'rotation_observable', 'rotation_observability_score', 'translation_step_m', 'rotation_step_deg',
               'source_translation_delta_m', 'source_rotation_delta_deg']
    timeline = []; previous = None
    for row in rows:
        source = sources[row['frame_idx']]
        rotation = np.asarray(row[RKEY], float); translation = np.asarray(row[TKEY], float)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError('invalid compact posterior pose')
        observable = source.get('rotation_observability') or {}
        step = 0. if previous is None else float(np.linalg.norm(translation - previous[1]))
        rotation_step = 0. if previous is None else angle(previous[0], rotation)
        source_has_pose = source.get(TKEY) is not None and source.get(RKEY) is not None
        timeline.append([row['frame_idx'], row['time_s'], translation.tolist(), rotation.tolist(), source.get('status'),
            source.get('direct_pose_observation_source'), row.get('metric_observation'), source.get('rigid_pose_observation_eligible'),
            row.get('depth_feasibility_uncertain'), row.get('uncertainty'), row.get('relative_evidence_connected'),
            observable.get('observable'), observable.get('score'), step, rotation_step,
            float(np.linalg.norm(translation - np.asarray(source[TKEY], float))) if source_has_pose else None, angle(np.asarray(source[RKEY], float), rotation) if source_has_pose else None])
        previous = rotation, translation
    # Every unreliable interval contributes its boundaries and midpoint; tails
    # cannot disappear behind an image quota. Largest jumps include both sides.
    review = set(ids[i] for i in np.unique(np.linspace(0, len(ids)-1, min(8, len(ids)), dtype=int)))
    intervals = []
    flags = [r[6] is not True or r[8] is True or r[10] is not True or r[11] is not True or r[9] is None or (r[9] is not None and r[9] >= .7) for r in timeline]
    start = None
    for i in range(len(ids) + 1):
        if i < len(ids) and flags[i] and start is None: start = i
        if start is not None and (i == len(ids) or not flags[i]):
            end = i - 1; intervals.append({'frame_start': ids[start], 'frame_end': ids[end], 'frame_count': i-start})
            review.update([ids[start], ids[(start+end)//2], ids[end]])
            start = None
    for column in (13, 14):
        for i in sorted(range(1, len(ids)), key=lambda i: (-(timeline[i][column] if timeline[i][column] is not None else -1.), ids[i]))[:3]:
            review.update([ids[i-1], ids[i]])
    metric_gaps = []
    for row in timeline:
        if row[6] is not True:
            if metric_gaps and metric_gaps[-1]['frame_end'] == row[0]-1:
                metric_gaps[-1]['frame_end'] = row[0]; metric_gaps[-1]['frame_count'] += 1
            else: metric_gaps.append({'frame_start':row[0],'frame_end':row[0],'frame_count':1})
    summary = {'claim_scope': 'posterior-derived estimates for visual review, not ground truth or physical acceptance',
        'uncertainty_kind': 'heuristic_support_uncertainty_not_marginal_covariance',
        'frame_count': len(ids), 'metric_frame_count': sum(r[6] is True for r in timeline),
        'direct_source_counts': {str(k): sum(r[5] == k for r in timeline) for k in sorted(set(r[5] for r in timeline), key=str)},
        'unreliable_intervals': intervals, 'missing_metric_support_intervals': metric_gaps, 'max_translation_step_m': max(r[13] for r in timeline),
        'max_rotation_step_deg': max(r[14] for r in timeline),
        'missing_timeline_frame_ids': sorted(set(range(ids[0], ids[-1]+1)) - set(ids)),
        'rotation_unobservable_frame_ids': [r[0] for r in timeline if r[11] is not True],
        'depth_uncertain_frame_ids': [r[0] for r in timeline if r[8] is True]}
    result = {'schema': 'trajectory_review_compact_v1', 'object_id': posterior['object_id'],
        'source_posterior': {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)},
        'frame_ids': ids, 'timeline_columns': columns, 'timeline': timeline,
        'review_frame_ids': sorted(review), 'summary': summary}
    validate_compact(result)
    return result


def validate_compact(evidence):
    import json
    import jsonschema
    jsonschema.validate(evidence, COMPACT_SCHEMA)
    if [row[0] for row in evidence['timeline']] != evidence['frame_ids'] or any(len(row) != len(evidence['timeline_columns']) for row in evidence['timeline']):
        raise ValueError('compact timeline coverage/column mismatch')
    if not set(evidence['review_frame_ids']) <= set(evidence['frame_ids']):
        raise ValueError('compact review outside timeline')
    if len(json.dumps(evidence, allow_nan=False).encode()) > 256_000:
        raise ValueError('compact trajectory facts exceed 256KB; retain evidence and split review explicitly')


def trajectory_review_sheets(evidence, raw_by_id, overlay_paths, output):
    """Readable raw/candidate pairs, with full-resolution sources hash-bound."""
    import hashlib
    from pathlib import Path
    from PIL import Image, ImageDraw
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    ids = evidence['review_frame_ids']
    if len(ids) != len(overlay_paths): raise ValueError('review overlay coverage mismatch')
    sheets = []; source_images = []
    for offset in range(0, len(ids), 3):
        group = ids[offset:offset+3]
        canvas = Image.new('RGB', (768, len(group)*414), 'black')
        draw = ImageDraw.Draw(canvas)
        for n, idx in enumerate(group):
            for col, path in enumerate((Path(raw_by_id[idx]['raw_frame_path']), Path(overlay_paths[offset+n]))):
                raw = path.read_bytes()
                source_images.append({'frame_idx': idx, 'role': 'raw' if col == 0 else 'posterior_mesh_overlay', 'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
                image = Image.open(path).convert('RGB'); image.thumbnail((384, 384))
                canvas.paste(image, (col*384, n*414+30))
                draw.text((col*384+8, n*414+8), f'frame {idx} | '+('RAW' if col == 0 else 'CANDIDATE OVERLAY'), fill='white')
        path = output/f'review_{offset//3:03d}.jpg'
        canvas.save(path, quality=85)
        sheets.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'frame_ids': group})
    return sheets, source_images


MAX_REQUEST_BYTES = 2_000_000


def judge(evidence, images, *, audit_dir=None):
    """Consume current-run image evidence and return a validated pose_trajectory judgment."""
    validate_compact(evidence)
    result, provenance = astra_request(PROMPT, SCHEMA, evidence, images, audit_dir=audit_dir,
        max_request_bytes=MAX_REQUEST_BYTES, adaptive_review_images=True)
    if 'object_id' in result and result['object_id'] != evidence['object_id']:
        raise ValueError('judgment changed selected object identity')
    allowed = set(evidence['frame_ids'])
    def check_frames(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {'frame_idx','frame_start','frame_end','selected_frame_idx','selected_anchor_frame_idx'} and child not in allowed:
                    raise ValueError('judgment frame outside input timeline')
                if key in {'evidence_frames','detector_prompt_frames','detector_active_interval'} and not set(child) <= allowed:
                    raise ValueError('judgment evidence frame outside input timeline')
                if 'frame_start' in value and value['frame_start'] > value['frame_end']:
                    raise ValueError('reversed judgment interval')
                check_frames(child)
        elif isinstance(value, list):
            for child in value:check_frames(child)
    check_frames(result)
    result['judgment_provenance'] = {'kind':'current_run_visual_judgment','judgment':'pose_trajectory',**provenance}
    return result


if __name__ == '__main__':
    judgment_cli(judge, audit_requests=True)
