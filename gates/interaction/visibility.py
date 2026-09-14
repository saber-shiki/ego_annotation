"""Full-timeline visibility and display qualification preserve source geometry."""
from __future__ import annotations
import copy
import numpy as np
from observations.common import pose_from_values
STATES = {'visible', 'partially_visible', 'occluded', 'out_of_frame', 'unresolved'}
POLICY = 'finite fitted pose/surface AND not out_of_frame; inferred visibility retains explicit judgment uncertainty'

def validate_visibility(judgment, frame_ids, object_id):
    if judgment.get('object_id') != object_id:
        raise ValueError('visibility selected object mismatch')
    rows = judgment.get('frames')
    if not isinstance(rows, list):
        raise ValueError('full visibility timeline required')
    result = {}
    for row in rows:
        idx = row.get('frame_idx')
        if type(idx) is not int or idx in result:
            raise ValueError('visibility requires unique integer frame IDs')
        visibility = row.get('visibility')
        if not isinstance(visibility, dict) or set(visibility) != {'object', 'left', 'right'}:
            raise ValueError('visibility requires object/left/right entries')
        for value in visibility.values():
            if not isinstance(value, dict) or value.get('state') not in STATES or 'occluder' not in value or (not (value['occluder'] is None or (isinstance(value['occluder'], str) and value['occluder'].strip()))) or (not isinstance(value.get('uncertainty'), str)) or (not value['uncertainty'].strip()):
                raise ValueError('visibility needs permitted state, occluder and explicit uncertainty')
        result[idx] = copy.deepcopy(visibility)
    if set(result) != set(map(int, frame_ids)):
        raise ValueError('visibility must cover exact full timeline')
    return result

def apply_display_visibility(state, judgment, object_id):
    """Called on the generic fitter result only by the standalone wrapper."""
    ids = state['frame_idx']
    visibility = validate_visibility(judgment, ids, object_id)
    result = dict(state)
    source_valid = []
    for (r, t) in zip(state['object_rotation'], state['object_translation_world']):
        try:
            pose_from_values(r, t)
            source_valid.append(True)
        except ValueError:
            source_valid.append(False)
    result['object_pose_valid'] = np.asarray(source_valid, dtype=bool)
    result['object_valid'] = result['object_pose_valid'] & np.array([visibility[int(i)]['object']['state'] != 'out_of_frame' for i in ids])
    for side in ('left', 'right'):
        result[side + '_display_valid'] = np.asarray(state[side + '_valid']) & np.array([visibility[int(i)][side]['state'] != 'out_of_frame' for i in ids])
    result['display_visibility_policy'] = np.asarray(POLICY)
    return result

def ownership_quarantine_mode(judgment: dict[str, Any] | None) -> str:
    from observations.interaction.evidence import HAND_FRONT_OCCLUSIONS
    from typing import Any
    from observations.interaction.evidence import NO_QUARANTINE_OCCLUSIONS
    if not isinstance(judgment, dict):
        return 'measurement_default'
    explicit = judgment.get('ownership_quarantine') or judgment.get('object_surface_policy')
    if isinstance(explicit, str):
        norm = explicit.strip()
        if norm in {'hand_projected', 'measurement_default', 'none', 'unresolved'}:
            return norm
        if norm in {'quarantine_projected_hand', 'quarantine_under_hand'}:
            return 'hand_projected'
        if norm in {'do_not_quarantine', 'object_visible'}:
            return 'none'
    occlusion = str(judgment.get('occlusion_relation') or '')
    if occlusion in NO_QUARANTINE_OCCLUSIONS:
        return 'none'
    if occlusion in HAND_FRONT_OCCLUSIONS:
        return 'hand_projected'
    return 'unresolved'
