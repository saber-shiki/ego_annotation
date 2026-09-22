"""Explicit source IDs and timestamps; positions are never manufactured measurements."""
import numpy as np
from numbers import Integral


def timeline_ids(values):
    values = list(values)
    if not values or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0 for v in values):
        raise ValueError('frame IDs require nonempty nonnegative integers')
    if values != sorted(set(values)):
        raise ValueError('frame IDs must be sorted and unique')
    return [int(v) for v in values]


def timeline_times(values, count):
    times = np.asarray(values, dtype=np.float64)
    if times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('timestamps must cover exact IDs and increase strictly')
    return times


def declared_ids(manifest):
    start, end = manifest['frame_range']
    ids = timeline_ids(manifest['frame_ids']) if 'frame_ids' in manifest else list(range(start, end + 1))
    if [ids[0], ids[-1]] != [start, end]:
        raise ValueError('frame_ids endpoints must match frame_range')
    return ids


def join_annotation_rows(manifest, rows):
    lookup = {}
    for position, row in enumerate(rows):
        frame = row.get('frame_idx', position)
        if type(frame) is not int or frame < 0 or frame in lookup:
            raise ValueError('annotation frame IDs must be unique nonnegative integers')
        lookup[frame] = row
    ids = declared_ids(manifest)
    if not set(ids).issubset(lookup):
        raise ValueError('annotations do not cover declared frame_range/IDs')
    selected = [lookup[i] for i in ids]
    times = timeline_times([row['time_s'] for row in selected], len(ids))
    if 'time_s' in manifest:
        expected = timeline_times(manifest['time_s'], len(ids))
        if not np.array_equal(times, expected):
            raise ValueError('manifest timestamps differ from annotation timeline')
    return np.asarray(ids, np.int64), times, selected
