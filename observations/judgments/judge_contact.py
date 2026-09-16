"""Independent contact image judgment, fixed to gpt-6-astra."""
from utils.runtime import astra_request, judgment_cli

PROMPT = 'Judge visual contact independently per hand over nonoverlapping intervals. Use likely_contact, possible_contact, no_contact or unresolved. Provide prior probability and uncertainty grounded in images. Do not declare signed collision/nonpenetration from generated geometry or call an image prior a calibrated measurement.'
SCHEMA = {'type': 'object',
 'properties': {'object_id': {'type': 'string', 'minLength': 1},
                'status': {'type': 'string', 'enum': ['ok']},
                'interaction_judgments': {'type': 'array',
                                          'items': {'type': 'object',
                                                    'properties': {'target_entity_id': {'type': 'string',
                                                                                        'minLength': 1},
                                                                   'hand_side': {'type': 'string',
                                                                                 'enum': ['left',
                                                                                          'right']},
                                                                   'frame_start': {'type': 'integer',
                                                                                   'minimum': 0},
                                                                   'frame_end': {'type': 'integer',
                                                                                 'minimum': 0},
                                                                   'contact_state': {'type': 'string',
                                                                                     'enum': ['likely_contact',
                                                                                              'possible_contact',
                                                                                              'no_contact',
                                                                                              'unresolved']},
                                                                   'contact_prior_probability': {'type': 'number',
                                                                                                 'minimum': 0,
                                                                                                 'maximum': 1},
                                                                   'rationale': {'type': 'string',
                                                                                 'minLength': 1},
                                                                   'uncertainty': {'type': 'string',
                                                                                   'minLength': 1}},
                                                    'required': ['target_entity_id',
                                                                 'hand_side',
                                                                 'frame_start',
                                                                 'frame_end',
                                                                 'contact_state',
                                                                 'contact_prior_probability',
                                                                 'rationale',
                                                                 'uncertainty'],
                                                    'additionalProperties': False}}},
 'required': ['object_id', 'status', 'interaction_judgments'],
 'additionalProperties': False}


def judge(evidence, images):
    """Consume current-run image evidence and return a validated contact judgment."""
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
    from actions.filters.contact import interaction_states
    interaction_states(result, evidence['frame_ids'], evidence['object_id'])
    result['judgment_provenance'] = {'kind':'current_run_visual_judgment','judgment':'contact',**provenance}
    return result


if __name__ == '__main__':
    judgment_cli(judge)
