"""Independent final inspection image judgment, fixed to gpt-6-astra."""
from pathlib import Path
from utils.state import sha
from utils.runtime import astra_request, judgment_cli

PROMPT = 'Inspect the full-video final annotation at beginning, middle, end, motion/occlusion transitions and low-confidence intervals. Check object identity, hands, alignment, timing, explicit missing-state clearing and source/candidate/display authority. State concrete defects. Visual inspection cannot grant physical-ready or measured contact.'
SCHEMA = {'type': 'object',
 'properties': {'object_id': {'type': 'string', 'minLength': 1},
                'visual_result': {'type': 'string',
                                  'enum': ['reviewed_usable', 'reviewed_with_defects', 'rejected']},
                'evidence_frames': {'type': 'array',
                                    'items': {'type': 'integer', 'minimum': 0},
                                    'minItems': 1},
                'defects': {'type': 'array',
                            'items': {'type': 'object',
                                      'properties': {'frame_start': {'type': 'integer',
                                                                     'minimum': 0},
                                                     'frame_end': {'type': 'integer', 'minimum': 0},
                                                     'rationale': {'type': 'string',
                                                                   'minLength': 1}},
                                      'required': ['frame_start', 'frame_end', 'rationale'],
                                      'additionalProperties': False}},
                'physical_authority_granted': {'const': False},
                'uncertainties': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}}},
 'required': ['object_id',
              'visual_result',
              'evidence_frames',
              'defects',
              'physical_authority_granted',
              'uncertainties'],
 'additionalProperties': False}


MAX_REQUEST_BYTES = 3_000_000


def bind_review_images(frame_ids, rendered_frames, images):
    """Bind one full rendered overlay per frame, without resampling or rehashing trust."""
    if (not frame_ids or any(type(i) is not int or i < 0 for i in frame_ids)
            or len(set(frame_ids)) != len(frame_ids)
            or len(images) != len(frame_ids) or len(rendered_frames) != len(frame_ids)):
        raise ValueError('rendered review frame coverage mismatch')
    result = []
    for idx, row, image in zip(frame_ids, rendered_frames, images):
        path = Path(image).resolve()
        if row.get('frame_idx') != idx or Path(row['path']).resolve() != path:
            raise ValueError('rendered review frame/path/order mismatch')
        if row.get('sha256') != sha(path):
            raise ValueError('rendered review image hash changed')
        result.append({'path': str(path), 'sha256': row['sha256'], 'frame_ids': [idx]})
    if len({r['path'] for r in result}) != len(result):
        raise ValueError('rendered review requires one original image per frame')
    return result


def review_evidence(evidence, images):
    """Idempotent complete-batch mapping; renderer records are the original authority."""
    mapped = bind_review_images(evidence['frame_ids'], evidence['rendered_frames'], images)
    for key, expected in (('review_images', mapped), ('review_frame_ids', evidence['frame_ids'])):
        if key in evidence and evidence[key] != expected:
            raise ValueError('final inspection review mapping/coverage mismatch')
    return {**evidence, 'review_images': mapped, 'review_frame_ids': list(evidence['frame_ids'])}


def judge(evidence, images, *, audit_dir=None):
    """Consume every full-batch overlay with bounded, audited Astra transport."""
    evidence = review_evidence(evidence, images)
    result, provenance = astra_request(PROMPT, SCHEMA, evidence, images,
        audit_dir=audit_dir, max_request_bytes=MAX_REQUEST_BYTES, adaptive_review_images=True)
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
    result['judgment_provenance'] = {'kind':'current_run_visual_judgment','judgment':'final_inspection',**provenance}
    return result


if __name__ == '__main__':
    judgment_cli(judge, audit_requests=True)
