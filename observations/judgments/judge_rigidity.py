"""Independent rigidity image judgment, fixed to gpt-6-astra."""
from utils.runtime import astra_request, judgment_cli

PROMPT = """Judge rigidity of the SELECTED TRACKED OBJECT AS A WHOLE over the supplied interval.
Use selected_object to identify its boundary (including attached components); do not classify a different hand, tool, or background object.
Read each contiguous motion sequence in temporal order. Raw RGB is paired with selected-track and owned-surface outlines; masks establish membership/visibility, not rigidity.
Distinguish a whole object's rotation, translation, perspective change and hand occlusion from motion of a component RELATIVE TO ITS BODY.
Classify articulated only when the RGB sequence visibly demonstrates relative component motion: identify the component and reference body, cite a before/after frame pair, and explain the visible change in their relationship that whole-object motion cannot explain.
A seam, cap, hinge-like appearance, hand touching a component, or different image orientation alone does not demonstrate articulation. A possibly detachable cap may stay attached and co-move throughout this interval; record attachment/mechanism uncertainty in attached_components without inventing opening or separation.
Use rigid when the observed tracked whole moves consistently as a rigid unit and no relative component motion is observed; this describes this interval, not every possible behavior of the object. Use unresolved if visibility genuinely cannot distinguish the alternatives. Preserve uncertainty, conflicting observations and reasons; do not choose rigid merely to satisfy a downstream workflow.
Generated mesh topology does not establish rigidity or signed readiness."""
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
 'required': ['object_id', 'physical_branch', 'evidence_frames', 'rationale', 'uncertainties', 'relative_component_motion', 'attached_components', 'motion_evidence'],
 'additionalProperties': False}

SCHEMA['properties'].update({
    'relative_component_motion': {'type':'string','enum':['observed','not_observed','uncertain']},
    'attached_components': {'type':'array','items':{'type':'object','properties':{
        'component':{'type':'string','minLength':1},
        'attachment_state':{'type':'string','enum':['attached','separated','uncertain']},
        'relative_motion':{'type':'string','enum':['observed','not_observed','uncertain']},
        'evidence_frames':{'type':'array','items':{'type':'integer','minimum':0},'minItems':1},
        'rationale':{'type':'string','minLength':1}},
        'required':['component','attachment_state','relative_motion','evidence_frames','rationale'],'additionalProperties':False}},
    'motion_evidence': {'type':'array','minItems':1,'items':{'type':'object','properties':{
        'frame_start':{'type':'integer','minimum':0},'frame_end':{'type':'integer','minimum':0},
        'component':{'type':'string','minLength':1},'reference_body':{'type':'string','minLength':1},
        'relative_motion_observed':{'type':'boolean'},'observation':{'type':'string','minLength':1}},
        'required':['frame_start','frame_end','component','reference_body','relative_motion_observed','observation'],
        'additionalProperties':False}}})


def judge(evidence, images):
    """Consume current-run image evidence and return a validated rigidity judgment."""
    if evidence.get('schema') != 'tracked_whole_rigidity_motion_evidence_v1' or evidence.get('selected_object',{}).get('object_id') != evidence['object_id']:
        raise ValueError('rigidity requires selected-object plan and contiguous current-track motion evidence')
    covered=[frame for sequence in evidence['motion_sequences'] for frame in sequence['frame_ids']]
    if covered != evidence['frame_ids'] or len(images)!=len(evidence['motion_sequences']):
        raise ValueError('rigidity image/contiguous timeline coverage mismatch')
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
