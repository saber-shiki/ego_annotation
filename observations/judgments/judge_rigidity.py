"""Independent rigidity image judgment, fixed to gpt-6-astra."""
from utils.runtime import astra_request, judgment_cli

PROMPT = 'Judge whether the manipulated entity is a rigid whole, rigid part, articulated assembly, deformable, or unresolved from visible motion. Explain evidence and uncertainty. Generated mesh topology does not establish rigidity or signed readiness.'
SCHEMA = {'type': 'object',
 'properties': {'object_id': {'type': 'string', 'minLength': 1},
                'physical_branch': {'type': 'string',
                                    'enum': ['rigid',
                                             'rigid_part',
                                             'articulated',
                                             'deformable',
                                             'unresolved']},
                'evidence_frames': {'type': 'array',
                                    'items': {'type': 'integer', 'minimum': 0},
                                    'minItems': 1},
                'rationale': {'type': 'string', 'minLength': 1},
                'uncertainties': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}}},
 'required': ['object_id', 'physical_branch', 'evidence_frames', 'rationale', 'uncertainties'],
 'additionalProperties': False}


def judge(evidence, images):
    """Consume current-run image evidence and return a validated rigidity judgment."""
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
    result['judgment_provenance'] = {'kind':'current_run_visual_judgment','judgment':'rigidity',**provenance}
    return result


if __name__ == '__main__':
    judgment_cli(judge)
