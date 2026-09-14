from __future__ import annotations

def validate(nodes, times, edges, anchor):
    from utils.geometry import valid_rotation
    import numpy as np
    ids = [n.frame_idx for n in nodes]
    if len(set(ids)) != len(ids) or not ids or anchor not in ids:
        raise ValueError('unique full timeline and one gauge anchor required')
    times = np.asarray(times, float)
    if times.shape != (len(nodes),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('strictly increasing finite timestamps required')
    for n in nodes:
        if not valid_rotation(n.initial_rotation) or not np.isfinite(n.initial_translation).all() or np.shape(n.initial_translation) != (3,):
            raise ValueError('malformed source/initial pose')
        if not 0 <= n.uncertainty <= 1:
            raise ValueError('invalid uncertainty')
    for e in edges:
        if e.source_frame_idx not in ids or e.target_frame_idx not in ids or e.source_frame_idx == e.target_frame_idx:
            raise ValueError('edge identity outside timeline')
        if not valid_rotation(e.rotation_source_to_target_world) or np.shape(e.translation_source_to_target_world_m) != (3,) or (not np.isfinite(e.translation_source_to_target_world_m).all()) or (not np.isfinite(e.weight)) or (e.weight < 0):
            raise ValueError('malformed relative pose/precision')
    return times
