"""Independent final inspection image judgment, fixed to gpt-6-astra."""
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


def judge(evidence, images):
    """Consume current-run image evidence and return a validated final_inspection judgment."""
    result, provenance = astra_request(PROMPT, SCHEMA, evidence, images)
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
    judgment_cli(judge)
