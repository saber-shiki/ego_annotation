"""Independent mask track image judgment, fixed to gpt-6-astra."""
from utils.runtime import astra_request, judgment_cli

PROMPT = 'Inspect raw RGB and rendered mask overlays for selected-object identity, hand/table contamination, drift and gaps. Report failed intervals with image-grounded explanations; do not infer metric geometry from a good mask.'
SCHEMA = {'type': 'object',
 'properties': {'object_id': {'type': 'string', 'minLength': 1},
                'identity_consistent': {'type': 'boolean'},
                'intervals': {'type': 'array',
                              'items': {'type': 'object',
                                        'properties': {'frame_start': {'type': 'integer',
                                                                       'minimum': 0},
                                                       'frame_end': {'type': 'integer',
                                                                     'minimum': 0},
                                                       'state': {'type': 'string',
                                                                 'enum': ['usable',
                                                                          'contaminated',
                                                                          'missing',
                                                                          'uncertain']},
                                                       'rationale': {'type': 'string',
                                                                     'minLength': 1}},
                                        'required': ['frame_start',
                                                     'frame_end',
                                                     'state',
                                                     'rationale'],
                                        'additionalProperties': False}},
                'uncertainties': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}}},
 'required': ['object_id', 'identity_consistent', 'intervals', 'uncertainties'],
 'additionalProperties': False}


MAX_REQUEST_BYTES = 2_000_000
MAX_BATCH_FRAMES = 6


def judge(evidence, images, *, audit_dir=None):
    """Consume current-run image evidence and return a validated mask_track judgment."""
    result, provenance = astra_request(PROMPT, SCHEMA, evidence, images,
        **({'audit_dir': audit_dir, 'max_request_bytes': MAX_REQUEST_BYTES,
            'adaptive_review_images': True} if 'mask_review_batch' in evidence else {}))
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
    result['judgment_provenance'] = {'kind':'current_run_visual_judgment','judgment':'mask_track',**provenance}
    return result


def uncertain_review_groups(checks, frame_ids, sampled):
    """Each ambiguous frame is reviewed with both available temporal neighbors.

    Uncertain contexts are kept intact and packed into groups of <=6 frames;
    remaining general samples follow. No required frame is truncated.
    """
    uncertain = [row['frame_idx'] for row in checks['frames']
                 if row.get('association_uncertainty') or row.get('extent_alternatives')]
    if not uncertain:
        return []
    positions = {frame: index for index, frame in enumerate(frame_ids)}
    if len(positions) != len(frame_ids) or not set(uncertain + list(sampled)) <= set(positions):
        raise ValueError('mask review frame identity outside current timeline')
    groups = []
    for frame in uncertain:
        context = list(frame_ids[max(0, positions[frame]-1):positions[frame]+2])
        union = sorted(set(groups[-1] + context), key=positions.get) if groups else context
        if groups and len(union) <= MAX_BATCH_FRAMES:
            groups[-1] = union
        else:
            groups.append(context)
    covered = {frame for group in groups for frame in group}
    remaining = [frame for frame in sampled if frame not in covered]
    groups.extend(remaining[i:i+MAX_BATCH_FRAMES] for i in range(0, len(remaining), MAX_BATCH_FRAMES))
    return groups


def render_uncertain_reviews(checks, track, raw_by_id, frame_ids, sampled, output_dir):
    """Typed batch evidence: actual raw/mask pixels, source hashes and full mapping."""
    from pathlib import Path
    import cv2
    import numpy as np
    from utils.state import sha
    groups = uncertain_review_groups(checks, frame_ids, sampled)
    if not groups:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {row['frame_idx']: row for row in checks['frames']}
    mappings = {}
    for frame in sorted({frame for group in groups for frame in group}):
        raw_path = Path(raw_by_id[frame]['raw_frame_path'])
        raw = cv2.imread(str(raw_path))
        if raw is None:
            raise ValueError('mask review raw image missing')
        overlay = raw.copy()
        mask_path = track.get(str(frame), {}).get('mask_path')
        mask_hash = None
        if mask_path:
            mask_hash = sha(mask_path)
            if frame in rows and rows[frame]['mask_sha256'] is not None and mask_hash != rows[frame]['mask_sha256']:
                raise ValueError('mask review raster changed since identity check')
            mask = cv2.imread(str(mask_path), 0)
            if mask is None:
                raise ValueError('SAM2 declared mask image is missing')
            selected = cv2.resize(mask, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            overlay[selected] = (overlay[selected]*.5 + np.array([50, 200, 50])*.5).astype('uint8')
        # Full field of view in both panels, never crop to the favored box.
        width = min(480, raw.shape[1])
        height = max(1, round(raw.shape[0]*width/raw.shape[1]))
        sheet = np.full((height+36, width*2, 3), 255, np.uint8)
        for column, (image, label) in enumerate(((raw, 'RAW'), (overlay, 'MASK OVERLAY'))):
            sheet[36:, column*width:(column+1)*width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            cv2.putText(sheet, f'FRAME {frame} {label}', (column*width+8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
        path = output_dir/f'{frame:06d}.jpg'
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError('mask evidence render failed')
        mappings[frame] = {'path': str(path), 'sha256': sha(path), 'frame_ids': [frame],
            'panels': ['raw', 'actual_sam2_mask_overlay'],
            'raw_path': str(raw_path), 'raw_sha256': sha(raw_path),
            'mask_path': mask_path, 'mask_sha256': mask_hash,
            'source_dimensions_wh': [raw.shape[1], raw.shape[0]],
            'sheet_dimensions_wh': [width*2, height+36]}
    required = sorted(mappings)
    batches = []
    for index, group in enumerate(groups):
        batches.append({'schema': 'mask_identity_visual_batch_v1',
            'mask_review_batch': {'index': index, 'count': len(groups),
                'required_frame_ids': required, 'context': 'each uncertain frame with immediate timeline neighbors'},
            'review_instructions': 'Inspect every displayed frame, especially extent ambiguity: selected and competing boxes can describe separate overlapping instances. Compare RAW with the actual mask overlay and neighboring frames; bbox IoU alone does not establish identity. Report unresolved identity explicitly.',
            'track': {str(frame): track.get(str(frame), {}) for frame in group},
            'candidate_identity_checks': {**checks, 'frames': [rows[frame] for frame in group if frame in rows]},
            'inspected_frame_ids': group, 'review_frame_ids': group,
            'review_images': [mappings[frame] for frame in group]})
    return batches


def merge_batch_judgments(results, batches):
    """Preserve every verdict/interval/provenance; any identity rejection wins."""
    if not results or len(results) != len(batches):
        raise ValueError('incomplete mask review results')
    required = batches[0]['mask_review_batch']['required_frame_ids']
    if set(required) != {frame for batch in batches for frame in batch['review_frame_ids']}:
        raise ValueError('incomplete mask review frame coverage')
    if len({result['object_id'] for result in results}) != 1:
        raise ValueError('mask review batches changed object identity')
    return {'object_id': results[0]['object_id'],
        'identity_consistent': all(result['identity_consistent'] for result in results),
        'intervals': [interval for result in results for interval in result['intervals']],
        'uncertainties': [value for result in results for value in result['uncertainties']],
        'judgment_provenance': {'kind': 'current_run_visual_judgment', 'judgment': 'mask_track',
            'required_frame_ids': required, 'batches': [
                {'evidence': batch, 'result': result} for batch, result in zip(batches, results)]}}


if __name__ == '__main__':
    judgment_cli(judge, audit_requests=True)
