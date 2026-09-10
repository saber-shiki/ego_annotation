"""Explicit standalone display visibility; never modifies source poses or MANO."""
import copy
import numpy as np
from rgb_scene_pose_io import pose_from_values

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
            if (not isinstance(value, dict) or value.get('state') not in STATES
                    or 'occluder' not in value or not (value['occluder'] is None or isinstance(value['occluder'], str) and value['occluder'].strip())
                    or not isinstance(value.get('uncertainty'), str) or not value['uncertainty'].strip()):
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
    for r, t in zip(state['object_rotation'], state['object_translation_world']):
        try:
            pose_from_values(r, t)
            source_valid.append(True)
        except ValueError:
            source_valid.append(False)
    result['object_pose_valid'] = np.asarray(source_valid, dtype=bool)
    result['object_valid'] = result['object_pose_valid'] & np.array([visibility[int(i)]['object']['state'] != 'out_of_frame' for i in ids])
    for side in ('left', 'right'):
        # Generic MANO validity is preserved verbatim, display eligibility separate.
        result[side + '_display_valid'] = np.asarray(state[side + '_valid']) & np.array([visibility[int(i)][side]['state'] != 'out_of_frame' for i in ids])
    result['display_visibility_policy'] = np.asarray(POLICY)
    return result
