"""Independent object image judgment, fixed to gpt-6-astra."""
from utils.runtime import astra_request, judgment_cli

PROMPT = 'Select the single manipulated object from actual raw video frames. Use open-vocabulary detector descriptions, no pixel clicks. Identify active intervals and representative frames, and distinguish evidence from uncertainty. The supplied frame_ids are the only legal frame identities.'
SCHEMA = {'type': 'object',
 'properties': {'objects': {'type': 'array',
                            'items': {'type': 'object',
                                      'properties': {'object_id': {'type': 'string',
                                                                   'minLength': 1},
                                                     'track_id': {'type': 'string', 'minLength': 1},
                                                     'description': {'type': 'string',
                                                                     'minLength': 1},
                                                     'physical_branch_hypotheses': {'type': 'array',
                                                                                    'items': {'type': 'string',
                                                                                              'minLength': 1},
                                                                                    'minItems': 1},
                                                     'evidence_frames': {'type': 'array',
                                                                         'items': {'type': 'integer',
                                                                                   'minimum': 0},
                                                                         'minItems': 1},
                                                     'expected_visible_intervals': {'type': 'array',
                                                                                    'items': {'type': 'object',
                                                                                              'properties': {'frame_start': {'type': 'integer',
                                                                                                                             'minimum': 0},
                                                                                                             'frame_end': {'type': 'integer',
                                                                                                                           'minimum': 0}},
                                                                                              'required': ['frame_start',
                                                                                                           'frame_end'],
                                                                                              'additionalProperties': False}},
                                                     'uncertainty_notes': {'type': 'array',
                                                                           'items': {'type': 'string',
                                                                                     'minLength': 1}},
                                                     'detector_text_prompts': {'type': 'array',
                                                                               'items': {'type': 'string',
                                                                                         'minLength': 1},
                                                                               'minItems': 1},
                                                     'detector_prompt_frames': {'type': 'array',
                                                                                'items': {'type': 'integer',
                                                                                          'minimum': 0},
                                                                                'minItems': 1},
                                                     'detector_active_interval': {'type': 'array',
                                                                                  'items': {'type': 'integer',
                                                                                            'minimum': 0},
                                                                                  'minItems': 2,
                                                                                  'maxItems': 2}},
                                      'required': ['object_id',
                                                   'track_id',
                                                   'description',
                                                   'physical_branch_hypotheses',
                                                   'evidence_frames',
                                                   'expected_visible_intervals',
                                                   'uncertainty_notes',
                                                   'detector_text_prompts',
                                                   'detector_prompt_frames',
                                                   'detector_active_interval'],
                                      'additionalProperties': False},
                            'minItems': 1,
                            'maxItems': 1}},
 'required': ['objects'],
 'additionalProperties': False}


def judge(evidence, images):
    """Consume current-run image evidence and return a validated object judgment."""
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
    result['judgment_provenance'] = {'kind':'current_run_visual_judgment','judgment':'object',**provenance}
    return result


if __name__ == '__main__':
    judgment_cli(judge)
