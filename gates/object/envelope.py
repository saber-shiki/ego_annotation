"""Frozen surface-frame validation; no physical acceptance authority."""
from pathlib import Path
import numpy as np
from utils.data import indexed
from utils.geometry import validate_transforms
from utils.state import sha, in_run


def check_references(inputs, root):
    for name, ref in inputs.items():
        path = in_run(ref['path'], root)
        if sha(path) != ref.get('sha256'):
            raise ValueError(f'immutable surface input changed: {name}')


def validate_envelopes(document, source):
    if (document.get('schema') != 'observed_surface_envelope_v1' or
            document.get('object_id') != source['object_id'] or
            document.get('run_root') != str(source['root']) or
            document.get('inputs') != source['inputs'] or
            document.get('frozen_before_optimization') is not True):
        raise ValueError('envelope/source binding mismatch')
    check_references(document['inputs'], source['root'])
    if document.get('collision_authority') is not False or document.get('calibrated_confidence_region') is not False:
        raise ValueError('envelope has no physical authority')
    sizes = np.asarray(document['fixed_shape_pca_long_mid_short_m'], float)
    q = source['vertices'] - np.median(source['vertices'], axis=0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    expected_sizes = np.sort(np.ptp(q @ vt.T, axis=0))[::-1]
    if sizes.shape != (3,) or not np.allclose(sizes, expected_sizes, rtol=0, atol=1e-12):
        raise ValueError('fixed original PCA size prior changed')
    rows = indexed(document['frames'])
    if set(rows) != set(source['frame_idx']):
        raise ValueError('envelope timeline mismatch')
    for i, idx in enumerate(source['frame_idx']):
        row = rows[idx]
        direct = bool(len(source['points_camera'][i]))
        if (row.get('object_id') != source['object_id'] or row.get('time_s') != source['times'][i] or
                row.get('per_frame_candidate_pose_used_for_construction') is not False or
                row.get('collision_authority') is not False or
                row.get('optimization_weight') != (1. if direct else 0.)):
            raise ValueError('envelope identity/weight/authority mismatch')
        kind = row.get('source_kind')
        if direct != (kind == 'direct_observation'):
            raise ValueError('missing frame cannot acquire direct envelope weight')
        if kind == 'unresolved_no_observation':
            if any(row.get(k) is not None for k in ('T_world_surface', 'lower_surface_m', 'upper_surface_m')):
                raise ValueError('unresolved surface cannot fabricate bounds')
            continue
        if kind not in ('direct_observation', 'transported_temporal_prior'):
            raise ValueError('unknown envelope source kind')
        S = validate_transforms([row['T_world_surface']])[0]
        lo, hi = np.asarray(row['lower_surface_m']), np.asarray(row['upper_surface_m'])
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
            raise ValueError('invalid surface envelope bounds')
        if direct:
            from observations.object.envelope import direct_bounds
            T = source['T'][i]
            p = source['points_camera'][i] @ T[:3, :3].T + T[:3, 3]
            expected_lo, expected_hi, _ = direct_bounds(p, S, sizes)
            if not np.allclose([lo, hi], [expected_lo, expected_hi], rtol=0, atol=1e-12):
                raise ValueError('direct observation bounds changed')
        elif row.get('donor_frame_idx') not in rows or rows[row['donor_frame_idx']]['source_kind'] != 'direct_observation':
            raise ValueError('transported envelope requires a direct donor')
    return document
