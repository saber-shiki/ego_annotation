"""Independent occlusion image judgment, fixed to gpt-6-astra."""
from utils.runtime import astra_request, judgment_cli

PROMPT = 'Judge object/hand depth ordering and occluder ownership from image evidence. Keep ambiguous ownership unresolved. A missing mask or plausible completion does not prove an object is behind a hand.'
SCHEMA = {'type': 'object',
 'properties': {'object_id': {'type': 'string', 'minLength': 1},
                'intervals': {'type': 'array',
                              'items': {'type': 'object',
                                        'properties': {'hand_side': {'type': 'string',
                                                                     'enum': ['left', 'right']},
                                                       'frame_start': {'type': 'integer',
                                                                       'minimum': 0},
                                                       'frame_end': {'type': 'integer',
                                                                     'minimum': 0},
                                                       'relation': {'type': 'string',
                                                                    'enum': ['hand_in_front',
                                                                             'object_in_front',
                                                                             'separate',
                                                                             'unresolved']},
                                                       'rationale': {'type': 'string',
                                                                     'minLength': 1},
                                                       'uncertainty': {'type': 'string',
                                                                       'minLength': 1}},
                                        'required': ['hand_side',
                                                     'frame_start',
                                                     'frame_end',
                                                     'relation',
                                                     'rationale',
                                                     'uncertainty'],
                                        'additionalProperties': False}}},
 'required': ['object_id', 'intervals'],
 'additionalProperties': False}


def judge(evidence, images):
    """Consume current-run image evidence and return a validated occlusion judgment."""
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
    result['judgment_provenance'] = {'kind':'current_run_visual_judgment','judgment':'occlusion',**provenance}
    return result


if __name__ == '__main__':
    judgment_cli(judge)
