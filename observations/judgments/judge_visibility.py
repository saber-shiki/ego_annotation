"""Independent visibility image judgment, fixed to gpt-6-astra."""
from utils.runtime import astra_request, judgment_cli

PROMPT = 'Judge visibility for every supplied frame ID exactly once and each of object, left and right. Use actual images and temporal evidence. Out-of-frame differs from occluded; uncertain absence remains unresolved. Preserve uncertainty for inferred states and name the occluder when supported.'
SCHEMA = {'type': 'object',
 'properties': {'object_id': {'type': 'string', 'minLength': 1},
                'frames': {'type': 'array',
                           'items': {'type': 'object',
                                     'properties': {'frame_idx': {'type': 'integer', 'minimum': 0},
                                                    'visibility': {'type': 'object',
                                                                   'properties': {'object': {'type': 'object',
                                                                                             'properties': {'state': {'type': 'string',
                                                                                                                      'enum': ['visible',
                                                                                                                               'partially_visible',
                                                                                                                               'occluded',
                                                                                                                               'out_of_frame',
                                                                                                                               'unresolved']},
                                                                                                            'occluder': {'anyOf': [{'type': 'string',
                                                                                                                                    'minLength': 1},
                                                                                                                                   {'type': 'null'}]},
                                                                                                            'uncertainty': {'type': 'string',
                                                                                                                            'minLength': 1}},
                                                                                             'required': ['state',
                                                                                                          'occluder',
                                                                                                          'uncertainty'],
                                                                                             'additionalProperties': False},
                                                                                  'left': {'type': 'object',
                                                                                           'properties': {'state': {'type': 'string',
                                                                                                                    'enum': ['visible',
                                                                                                                             'partially_visible',
                                                                                                                             'occluded',
                                                                                                                             'out_of_frame',
                                                                                                                             'unresolved']},
                                                                                                          'occluder': {'anyOf': [{'type': 'string',
                                                                                                                                  'minLength': 1},
                                                                                                                                 {'type': 'null'}]},
                                                                                                          'uncertainty': {'type': 'string',
                                                                                                                          'minLength': 1}},
                                                                                           'required': ['state',
                                                                                                        'occluder',
                                                                                                        'uncertainty'],
                                                                                           'additionalProperties': False},
                                                                                  'right': {'type': 'object',
                                                                                            'properties': {'state': {'type': 'string',
                                                                                                                     'enum': ['visible',
                                                                                                                              'partially_visible',
                                                                                                                              'occluded',
                                                                                                                              'out_of_frame',
                                                                                                                              'unresolved']},
                                                                                                           'occluder': {'anyOf': [{'type': 'string',
                                                                                                                                   'minLength': 1},
                                                                                                                                  {'type': 'null'}]},
                                                                                                           'uncertainty': {'type': 'string',
                                                                                                                           'minLength': 1}},
                                                                                            'required': ['state',
                                                                                                         'occluder',
                                                                                                         'uncertainty'],
                                                                                            'additionalProperties': False}},
                                                                   'required': ['object',
                                                                                'left',
                                                                                'right'],
                                                                   'additionalProperties': False}},
                                     'required': ['frame_idx', 'visibility'],
                                     'additionalProperties': False},
                           'minItems': 1}},
 'required': ['object_id', 'frames'],
 'additionalProperties': False}


def judge(evidence, images):
    """Consume current-run image evidence and return a validated visibility judgment."""
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
    from actions.filters.visibility import validate_visibility
    validate_visibility(result, evidence['frame_ids'], evidence['object_id'])
    result['judgment_provenance'] = {'kind':'current_run_visual_judgment','judgment':'visibility',**provenance}
    return result


if __name__ == '__main__':
    judgment_cli(judge)
