"""Anatomical contact declarations and shared visual reachability qualification."""
from __future__ import annotations
import numpy as np
from utils.data import frame_ids
SIDES = ('left', 'right')
PRIMARY_DIGITS = ('thumb', 'index', 'middle', 'ring', 'little')

def primary_contact_pads(payload, ids, object_id):
    """Optional P17 interval declarations; missing designation grants no reach.

    Names select existing MANO fingertip pads, never caller-supplied vertices.
    Expanded reach requires likely_contact and visible image support at fit time.
    """
    ids = frame_ids(ids)
    states = interaction_states(payload, ids, object_id)
    selected = np.zeros((len(ids), 2, 5), bool)
    if payload is None:
        return selected
    rows = payload.get('interaction_judgments', payload.get('judgments', payload.get('segments', [])))
    for row in rows:
        declaration = row.get('primary_contact_pads')
        if declaration is None:
            continue
        if not isinstance(declaration, dict) or set(declaration) != {'digits', 'rationale'} or (not isinstance(declaration['digits'], list)) or any((not isinstance(d, str) or d not in PRIMARY_DIGITS for d in declaration['digits'])) or (len(set(declaration['digits'])) != len(declaration['digits'])) or (not isinstance(declaration['rationale'], str)) or (not declaration['rationale'].strip()):
            raise ValueError('primary_contact_pads requires unique anatomical digits and nonempty rationale')
        target = row.get('target_entity_id', row.get('object_id', row.get('track_id')))
        if target and target.split(':', 1)[-1] != object_id.split(':', 1)[-1]:
            continue
        side = row.get('hand_side', row.get('side'))
        si = SIDES.index(side)
        start = int(row.get('frame_start', row.get('start_frame', int(min(ids)))))
        end = int(row.get('frame_end', row.get('end_frame', int(max(ids)))))
        mask = (ids >= start) & (ids <= end) & (states[:, si] == 'likely_contact')
        for digit in declaration['digits']:
            selected[mask, si, PRIMARY_DIGITS.index(digit)] = True
    for row in payload.get('frames', []):
        at = np.flatnonzero(ids == row['frame_idx'])
        visibility = row.get('visibility', {})
        for (si, side) in enumerate(SIDES):
            if any((visibility.get(entity, {}).get('state') not in {'visible', 'partially_visible'} for entity in (side, 'object'))):
                selected[at, si] = False
    return selected

def interaction_states(payload, ids, object_id):
    states = np.full((len(ids), 2), 'legacy_visual_hypothesis', dtype='U32')
    if payload is None:
        return states
    if not isinstance(payload, dict):
        raise ValueError('interaction judgments payload is not an object')
    ids = frame_ids(ids)
    if payload.get('schema') == 'v19_visible_contact_ownership_factor_v2':
        target = payload.get('target_entity_id')
        if not isinstance(target, str) or target.split(':', 1)[-1] != object_id.split(':', 1)[-1]:
            raise ValueError('P17 interaction report target mismatch')
        rows = payload.get('factor_rows')
        if not isinstance(rows, list):
            raise ValueError('P17 interaction report lacks factor_rows')
        states[:] = 'unresolved'
        lookup = {int(value): position for (position, value) in enumerate(ids)}
        occupied = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('invalid P17 factor row')
            idx = int(row['frame_idx'])
            side = row.get('hand_side', row.get('side'))
            state = row.get('agent_contact_state')
            if idx not in lookup or side not in SIDES or state not in {'likely_contact', 'possible_contact', 'no_contact', 'unresolved'}:
                raise ValueError('invalid P17 frame/side/contact state')
            key = (idx, side)
            if key in occupied:
                raise ValueError('duplicate P17 frame/side state')
            occupied.add(key)
            states[lookup[idx], SIDES.index(side)] = state
        return states
    if payload.get('status') not in (None, 'ok'):
        raise ValueError('interaction judgments status is not ok')
    rows = payload.get('interaction_judgments', payload.get('judgments', payload.get('segments')))
    if not isinstance(rows, list) or not rows:
        raise ValueError('missing interaction_judgments')
    states[:] = 'unresolved'
    occupied = np.zeros(states.shape, bool)
    matched_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('interaction judgment segment is not an object')
        target = row.get('target_entity_id', row.get('object_id', row.get('track_id')))
        if target is not None and (not isinstance(target, str)):
            raise ValueError('interaction judgment target must be a string')
        if target and target.split(':', 1)[-1] != object_id.split(':', 1)[-1]:
            continue
        side = row.get('hand_side', row.get('side'))
        if side not in SIDES:
            raise ValueError('interaction judgment side must be left or right')
        state = row.get('contact_state')
        if not isinstance(state, str) or state not in {'likely_contact', 'possible_contact', 'no_contact', 'unresolved'}:
            raise ValueError('invalid interaction contact_state')
        bounds = [row.get('frame_start', row.get('start_frame', int(min(ids)))), row.get('frame_end', row.get('end_frame', int(max(ids))))]
        try:
            (start, end) = (int(value) for value in bounds)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('interaction interval bounds must be integer frame IDs') from exc
        if any((isinstance(value, bool) or (isinstance(value, float) and value != int(value)) for value in bounds)):
            raise ValueError('interaction interval bounds must be integer frame IDs')
        if end < start:
            raise ValueError('inverted interaction interval')
        mask = (ids >= start) & (ids <= end)
        si = SIDES.index(side)
        if occupied[mask, si].any():
            raise ValueError('overlapping interaction judgments')
        occupied[mask, si] = True
        states[mask, si] = state
        matched_rows += int(mask.any())
    if not matched_rows:
        raise ValueError(f'interaction judgments contained no segments for target {object_id!r} on this timeline')
    return states
VISUAL_PROXIMITY_MAX_GAP_M = 0.012
PRIMARY_PROXIMITY_MAX_GAP_M = 0.045
HAND_TRANSLATION_BOUND_M = 0.025
HAND_SCALE_BOUNDS = (0.85, 1.18)
SHARED_TRANSLATION_TOLERANCE_M = 0.001

def primary_similarity_reachable(points_camera, targets_camera, active=None):
    """Test one shared camera-origin scale/translation for active primary pads.

    The same hand-level similarity correction must explain every declared
    primary pad.  A one-millimeter tolerance covers the fact that each closest
    triangle target is a sampled visual hypothesis; it is not a second free
    translation per pad.  Independent per-point feasibility is insufficient.
    """
    points = np.asarray(points_camera, float)
    targets = np.asarray(targets_camera, float)
    if points.shape != (5, 3) or targets.shape != (5, 3):
        raise ValueError('five camera-space pad/triangle points required')
    active = np.ones(5, bool) if active is None else np.asarray(active, bool)
    if active.shape != (5,):
        raise ValueError('five active primary-pad values required')
    reachable = np.zeros(5, bool)
    active_ids = np.flatnonzero(active)
    if not len(active_ids):
        return reachable
    (low, high) = HAND_SCALE_BOUNDS
    tolerance = SHARED_TRANSLATION_TOLERANCE_M
    for axis in range(3):
        for i in active_ids:
            (p, q) = (points[i, axis], targets[i, axis])
            if not np.isfinite([p, q]).all():
                return reachable
            bound = HAND_TRANSLATION_BOUND_M + tolerance
            if abs(p) < 1e-12:
                if abs(q) > bound:
                    return reachable
                continue
            (a, b) = sorted(((q - bound) / p, (q + bound) / p))
            (low, high) = (max(low, a), min(high, b))
            if low > high:
                return reachable
        for i in active_ids:
            for j in active_ids:
                if i == j:
                    continue
                coefficient = points[i, axis] - points[j, axis]
                lower = targets[i, axis] - targets[j, axis] - 2.0 * tolerance
                if abs(coefficient) < 1e-12:
                    if lower > 0.0:
                        return reachable
                elif coefficient > 0.0:
                    low = max(low, lower / coefficient)
                else:
                    high = min(high, lower / coefficient)
                if low > high:
                    return reachable
    reachable[active_ids] = True
    return reachable

def visual_proximity_weights(inside, image_distances, triangle_gaps, *, state='likely_contact', primary=None, reachable=None):
    """Demo-only image/triangle proximity weights; never a contact measurement."""
    inside = np.asarray(inside, dtype=bool)
    gaps = np.asarray(triangle_gaps, dtype=float)
    if inside.shape != (5,) or gaps.shape != (5,) or len(image_distances) != 5:
        raise ValueError('five fingertip proximity values required')
    result = np.zeros(5, np.float32)
    if state not in {'likely_contact', 'possible_contact', 'legacy_visual_hypothesis'}:
        return result
    primary = np.zeros(5, bool) if primary is None else np.asarray(primary, bool)
    reachable = np.zeros(5, bool) if reachable is None else np.asarray(reachable, bool)
    if primary.shape != (5,) or reachable.shape != (5,):
        raise ValueError('five primary/reachability values required')
    image_bounds = [16.0, 8.0, 8.0, 8.0, 8.0]
    strengths = [1.0, 0.12, 0.12, 0.12, 0.12]
    for finger in range(5):
        expanded = state == 'likely_contact' and primary[finger] and reachable[finger]
        bound = PRIMARY_PROXIMITY_MAX_GAP_M if expanded else VISUAL_PROXIMITY_MAX_GAP_M
        distance = image_distances[finger]
        if inside[finger] and distance is not None and np.isfinite(distance) and (float(distance) < (16.0 if expanded else image_bounds[finger])) and np.isfinite(gaps[finger]) and (0 <= gaps[finger] < bound):
            result[finger] = 1.0 if expanded else strengths[finger]
    return result

def validate_visual_proximity_contract(rep, appearance_manifest_path):
    embedded = rep.get('appearance_mask_manifest')
    receipt = rep.get('visual_mesh_receipt')
    if appearance_manifest_path is not None and (not isinstance(embedded, dict) or Path(embedded.get('path', '')).resolve() != appearance_manifest_path.resolve() or embedded.get('sha256') != sha(appearance_manifest_path)):
        raise ValueError('appearance manifest does not match pose binding identity')
    if not isinstance(receipt, dict) or receipt.get('demo_visual_proximity_fit_eligible') is not True or receipt.get('physical_contact_authority_eligible') is not False or any((receipt.get(key) is not False for key in ('generated_faces_contact_eligible', 'generated_faces_collision_eligible', 'generated_faces_sign_eligible', 'generated_faces_sdf_eligible', 'generated_faces_nonpenetration_eligible'))):
        raise ValueError('visual mesh lacks demo visual-proximity-only role contract')
    mesh_contract = rep.get('render_mesh_contract') or {}
    if not receipt.get('output_mesh_path') or not mesh_contract.get('required_mesh_path') or Path(receipt['output_mesh_path']).resolve() != Path(mesh_contract['required_mesh_path']).resolve() or (receipt.get('output_mesh_sha256') != mesh_contract.get('required_mesh_sha256')):
        raise ValueError('visual proximity role does not bind the rendered mesh identity')
from pathlib import Path
from utils.state import sha

def contact_patch_anchor_coherence(rows: list[FrameHandRow]) -> dict[str, Any]:
    from typing import Any
    import numpy as np
    from collections import Counter
    from utils.geometry import inverse_object
    from utils.data import numeric_summary
    from utils.data import FrameHandRow
    centroids_object: list[np.ndarray] = []
    row_spreads: list[float] = []
    support_uncertainties: list[float] = []
    anchor_states: Counter[str] = Counter()
    allowed_count = 0
    active_count = 0
    for row in rows:
        if row.contact_anchor_state:
            anchor_states[str(row.contact_anchor_state)] += 1
        if row.contact_anchor_residual_allowed:
            allowed_count += 1
        if row.contact_patch_factor_state != 'active_contact_patch' or len(row.contact_patch_target_world_m) == 0:
            continue
        active_count += 1
        targets_object = inverse_object(row.contact_patch_target_world_m, row.object_rotation_world_from_object, row.object_translation_world_m)
        centroid = np.median(targets_object, axis=0)
        centroids_object.append(centroid.astype(float))
        spread = np.linalg.norm(targets_object - centroid[None, :], axis=1)
        row_spreads.extend(spread.astype(float).tolist())
        support_uncertainties.append(float(row.contact_patch_support_uncertainty_m))
    if centroids_object:
        centroid_arr = np.asarray(centroids_object, dtype=float)
        median_centroid = np.median(centroid_arr, axis=0)
        centroid_dispersion = np.linalg.norm(centroid_arr - median_centroid[None, :], axis=1)
    else:
        median_centroid = np.zeros((3,), dtype=float)
        centroid_dispersion = np.asarray([], dtype=float)
    centroid_summary = numeric_summary(np.asarray(centroid_dispersion, dtype=float))
    support_summary = numeric_summary(np.asarray(support_uncertainties, dtype=float))
    dispersion_p95 = centroid_summary.get('p95') if isinstance(centroid_summary, dict) else None
    support_p95 = support_summary.get('p95') if isinstance(support_summary, dict) else None
    return {'active_contact_patch_rows_with_targets': int(active_count), 'contact_anchor_state_counts': dict(sorted(anchor_states.items())), 'contact_anchor_residual_allowed_count': int(allowed_count), 'object_frame_contact_centroid_median': median_centroid.astype(float).tolist(), 'object_frame_centroid_dispersion_m': centroid_summary, 'row_patch_spread_m': numeric_summary(np.asarray(row_spreads, dtype=float)), 'support_uncertainty_m': support_summary, 'centroid_dispersion_p95_exceeds_support_p95': bool(dispersion_p95 is not None and support_p95 is not None and (float(dispersion_p95) > float(support_p95))), 'claim_scope': 'Diagnostic support for a persistent object-frame A_t contact anchor. If stable anchor rows are absent or dispersion exceeds support, local contact must remain a sliding/bounded visible-surface hypothesis rather than a point-anchor force on H_t.'}

def load_visible_ownership_rows(report_path: Path | None, *, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    from typing import Any
    from pathlib import Path
    return load_factor_rows(report_path, 'ownership_rows', expected_family='visible_ownership', target_entity_id=target_entity_id)

def load_factor_rows(report_path: Path | None, row_key: str, *, expected_family: str, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    from typing import Any
    from pathlib import Path
    from utils.data import as_list
    from utils.data import load_json
    if report_path is None:
        return {}
    if not report_path.exists():
        raise FileNotFoundError(f'missing {expected_family} factor report: {report_path}')
    payload = load_json(report_path)
    if not isinstance(payload, dict):
        raise ValueError(f'{expected_family} factor report is not a JSON object: {report_path}')
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in as_list(payload.get(row_key)):
        if not isinstance(row, dict):
            continue
        key = validate_factor_row_contract(row, expected_family=expected_family, target_entity_id=target_entity_id, report_path=report_path)
        if key in out:
            raise ValueError(f'duplicate {expected_family} factor row for {key} while reading {report_path}')
        out[key] = row
    return out

def validate_factor_row_contract(row: dict[str, Any], *, expected_family: str, target_entity_id: str, report_path: Path) -> tuple[int, str]:
    from typing import Any
    from pathlib import Path
    from observations.interaction.evidence import FACTOR_REQUIRED_FIELDS
    missing = [field for field in FACTOR_REQUIRED_FIELDS if row.get(field) in (None, '')]
    if missing:
        raise ValueError(f'{expected_family} factor row lacks required fields {missing} in {report_path}: {row}')
    family = str(row.get('factor_family') or '')
    if family != expected_family:
        raise ValueError(f'factor row family {family!r} does not match expected {expected_family!r} in {report_path}: {row}')
    if str(row.get('target_entity_id')) != str(target_entity_id):
        raise ValueError(f"{expected_family} factor target {row.get('target_entity_id')} does not match solver target {target_entity_id} in {report_path}")
    provenance = row.get('provenance')
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError(f'{expected_family} factor row has empty/non-dict provenance in {report_path}: {row}')
    return (int(row['frame_idx']), str(row['hand_side']))

def merge_factor_row_maps(family: str, specific: dict[tuple[int, str], dict[str, Any]], generic: dict[tuple[int, str], dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    from typing import Any
    merged = dict(generic)
    overlap = sorted(set(merged) & set(specific))
    if overlap:
        raise ValueError(f'factor family {family} supplied by both generic and family-specific reports for keys: {overlap[:5]}')
    merged.update(specific)
    return merged

def load_generic_factor_reports(report_paths: list[Path] | None, *, target_entity_id: str) -> dict[str, dict[tuple[int, str], dict[str, Any]]]:
    from typing import Any
    from pathlib import Path
    from utils.data import as_list
    from utils.data import load_json
    out: dict[str, dict[tuple[int, str], dict[str, Any]]] = {'visible_ownership': {}, 'surface_eligibility': {}, 'visible_surface_track': {}, 'hand_observation_visibility': {}, 'hand_depth_shift_prior': {}, 'contact_patch': {}}
    for report_path in list(report_paths or []):
        if not report_path.exists():
            raise FileNotFoundError(f'missing generic factor report: {report_path}')
        payload = load_json(report_path)
        rows: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            rows.extend([r for r in as_list(payload.get('ownership_rows')) if isinstance(r, dict)])
            rows.extend([r for r in as_list(payload.get('factor_rows')) if isinstance(r, dict)])
        for row in rows:
            family = str(row.get('factor_family') or '')
            if family not in out:
                continue
            key = validate_factor_row_contract(row, expected_family=family, target_entity_id=target_entity_id, report_path=report_path)
            if key in out[family]:
                raise ValueError(f'duplicate generic {family} factor row for {key} while reading {report_path}')
            out[family][key] = row
    return out

def contact_patch_for_row(row: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    from typing import Any
    import numpy as np
    import argparse
    if not isinstance(row, dict):
        return {'state': 'missing_contact_patch_row', 'weight': 0.0, 'band_m': float(args.contact_patch_band_m), 'target_margin_m': float(args.contact_patch_target_margin_m), 'support_uncertainty_m': float(args.contact_patch_support_uncertainty_m), 'support_uncertainty_source': 'default_arg_missing_contact_patch_row', 'local_patch_support_state': None, 'local_patch_support_consumed': False, 'local_patch_support_uncertainty_m': None, 'global_object_support_uncertainty_m': None, 'local_patch_sample_count': 0, 'local_patch_temporal_sample_count': 0, 'max_vertices': int(args.max_contact_patch_vertices), 'prior_probability': 0.0, 'contact_anchor_state': None, 'contact_anchor_residual_allowed': False, 'contact_anchor_blockers': [], 'contact_pose_anchor_key': None}
    state = str(row.get('state') or 'active_contact_patch')
    try:
        raw_weight = float(row.get('weight', args.contact_patch_weight) or 0.0)
    except Exception:
        raw_weight = float(args.contact_patch_weight)
    try:
        base_weight = float(row.get('contact_patch_base_weight', row.get('base_weight', raw_weight)) or raw_weight)
    except Exception:
        base_weight = raw_weight
    weight = base_weight if bool(args.optimize_contact_state) else raw_weight
    try:
        band_m = float(row.get('contact_patch_band_m', row.get('band_m', args.contact_patch_band_m)) or args.contact_patch_band_m)
    except Exception:
        band_m = float(args.contact_patch_band_m)
    try:
        target_margin_m = float(row.get('contact_patch_target_margin_m', row.get('target_margin_m', args.contact_patch_target_margin_m)) or args.contact_patch_target_margin_m)
    except Exception:
        target_margin_m = float(args.contact_patch_target_margin_m)
    try:
        max_vertices = int(row.get('max_vertices', args.max_contact_patch_vertices) or args.max_contact_patch_vertices)
    except Exception:
        max_vertices = int(args.max_contact_patch_vertices)
    try:
        support_uncertainty_m = float(row.get('object_support_uncertainty_m', row.get('contact_patch_support_uncertainty_m', row.get('support_uncertainty_m', args.contact_patch_support_uncertainty_m))) or args.contact_patch_support_uncertainty_m)
    except Exception:
        support_uncertainty_m = float(args.contact_patch_support_uncertainty_m)
    raw_prior = row.get('contact_state_prior_probability', row.get('contact_patch_prior_probability', row.get('latent_contact_confidence', None)))
    try:
        prior_probability = float(raw_prior) if raw_prior is not None else 1.0 if max(0.0, weight) > 0.0 and state == 'active_contact_patch' else 0.0
    except Exception:
        prior_probability = 1.0 if max(0.0, weight) > 0.0 and state == 'active_contact_patch' else 0.0

    def optional_float(raw: Any) -> float | None:
        try:
            val = float(raw)
        except Exception:
            return None
        return val if np.isfinite(val) else None
    blockers = row.get('contact_anchor_blockers')
    local_unc = optional_float(row.get('local_patch_support_uncertainty_m'))
    global_unc = optional_float(row.get('global_object_support_uncertainty_m'))
    try:
        local_count = int(row.get('local_patch_sample_count', 0) or 0)
    except Exception:
        local_count = 0
    try:
        temporal_count = int(row.get('local_patch_temporal_sample_count', 0) or 0)
    except Exception:
        temporal_count = 0
    return {'state': state, 'weight': max(0.0, weight), 'prior_probability': float(np.clip(prior_probability, 0.0, 1.0)), 'band_m': max(0.0, band_m), 'target_margin_m': max(0.0, target_margin_m), 'support_uncertainty_m': max(0.0, support_uncertainty_m), 'support_uncertainty_source': row.get('contact_patch_support_uncertainty_source') if isinstance(row.get('contact_patch_support_uncertainty_source'), str) else None, 'local_patch_support_state': row.get('local_patch_support_state') if isinstance(row.get('local_patch_support_state'), str) else None, 'local_patch_support_consumed': bool(row.get('local_patch_support_consumed')), 'local_patch_support_uncertainty_m': local_unc, 'global_object_support_uncertainty_m': global_unc, 'local_patch_sample_count': max(0, local_count), 'local_patch_temporal_sample_count': max(0, temporal_count), 'max_vertices': max(0, max_vertices), 'contact_anchor_state': row.get('contact_anchor_state'), 'contact_anchor_residual_allowed': bool(row.get('contact_anchor_residual_allowed')), 'contact_anchor_blockers': blockers if isinstance(blockers, list) else [], 'contact_pose_anchor_key': row.get('contact_pose_anchor_key') if isinstance(row.get('contact_pose_anchor_key'), str) else None}

def load_visible_surface_track_rows(report_path: Path | None, *, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    from typing import Any
    from pathlib import Path
    return load_factor_rows(report_path, 'factor_rows', expected_family='visible_surface_track', target_entity_id=target_entity_id)

def load_surface_eligibility_rows(report_path: Path | None, *, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    from typing import Any
    from pathlib import Path
    return load_factor_rows(report_path, 'factor_rows', expected_family='surface_eligibility', target_entity_id=target_entity_id)

def load_agent_interaction_judgments(path: Path | None, *, case: str, target_entity_id: str, frame_span: tuple[int, int], sides: set[str]) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    from collections import Counter
    from observations.interaction.evidence import finite_float
    from observations.interaction.evidence import CONTACT_STATES
    from observations.interaction.evidence import load_json
    from observations.interaction.evidence import DEPTH_RELIABILITY_STATES
    from pathlib import Path
    from typing import Any
    from observations.interaction.evidence import target_matches
    from observations.interaction.evidence import OCCLUSION_RELATIONS
    if path is None:
        return ({}, {'enabled': False, 'path': None, 'segments': 0, 'frame_side_count': 0})
    if not path.exists():
        raise FileNotFoundError(f'missing agent interaction judgment JSON: {path}')
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f'agent interaction judgment must be a JSON object: {path}')
    if payload.get('status') not in (None, 'ok'):
        raise ValueError(f"agent interaction judgment status is not ok in {path}: {payload.get('status')!r}")
    if payload.get('case') not in (None, case):
        raise ValueError(f"agent interaction judgment case {payload.get('case')!r} does not match {case!r} in {path}")
    rows = payload.get('interaction_judgments', payload.get('judgments', payload.get('segments')))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'agent interaction judgment has no non-empty interaction_judgments/judgments/segments list: {path}')
    (start_default, end_default) = frame_span
    index: dict[tuple[int, str], dict[str, Any]] = {}
    kept_segments = 0
    state_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError(f'agent interaction judgment segment is not an object in {path}: {raw!r}')
        if not target_matches(raw.get('target_entity_id', raw.get('object_id', raw.get('track_id'))), target_entity_id):
            continue
        hand_side = str(raw.get('hand_side', raw.get('side', '')))
        if hand_side not in sides:
            continue
        s = int(raw.get('frame_start', raw.get('start_frame', start_default)))
        e = int(raw.get('frame_end', raw.get('end_frame', end_default)))
        if e < s:
            raise ValueError(f'agent interaction judgment has inverted interval {s}>{e} in {path}: {raw}')
        s = max(s, start_default)
        e = min(e, end_default)
        if e < s:
            continue
        contact_state = str(raw.get('contact_state', '')).strip()
        if contact_state not in CONTACT_STATES:
            raise ValueError(f'agent interaction judgment contact_state must be one of {sorted(CONTACT_STATES)} in {path}: {raw}')
        occlusion_relation = str(raw.get('occlusion_relation', raw.get('visibility_relation', ''))).strip()
        if occlusion_relation not in OCCLUSION_RELATIONS:
            raise ValueError(f'agent interaction judgment occlusion_relation must be one of {sorted(OCCLUSION_RELATIONS)} in {path}: {raw}')
        depth_reliability = str(raw.get('depth_reliability', 'mixed_or_unresolved')).strip()
        if depth_reliability not in DEPTH_RELIABILITY_STATES:
            raise ValueError(f'agent interaction judgment depth_reliability must be one of {sorted(DEPTH_RELIABILITY_STATES)} in {path}: {raw}')
        prior = finite_float(raw.get('contact_prior_probability'), field='contact_prior_probability', path=path)
        if not 0.0 <= prior <= 1.0:
            raise ValueError(f'agent interaction judgment contact_prior_probability must be in [0,1] in {path}: {raw}')
        support_unc = finite_float(raw.get('contact_support_uncertainty_m'), field='contact_support_uncertainty_m', path=path)
        if support_unc < 0.0:
            raise ValueError(f'agent interaction judgment contact_support_uncertainty_m must be nonnegative in {path}: {raw}')
        weight_mult = finite_float(raw.get('contact_weight_multiplier', 1.0), field='contact_weight_multiplier', path=path)
        if weight_mult < 0.0:
            raise ValueError(f'agent interaction judgment contact_weight_multiplier must be nonnegative in {path}: {raw}')
        normalized = {**raw, 'judgment_id': str(raw.get('judgment_id', f'{hand_side}_{s}_{e}')), 'frame_start': int(s), 'frame_end': int(e), 'hand_side': hand_side, 'target_entity_id': str(target_entity_id), 'contact_state': contact_state, 'occlusion_relation': occlusion_relation, 'depth_reliability': depth_reliability, 'contact_prior_probability': float(prior), 'contact_support_uncertainty_m': float(support_unc), 'contact_weight_multiplier': float(weight_mult), 'source_path': str(path)}
        kept_segments += 1
        state_counts[contact_state] += 1
        occlusion_counts[occlusion_relation] += 1
        for frame_idx in range(s, e + 1):
            key = (frame_idx, hand_side)
            if key in index:
                raise ValueError(f'overlapping agent interaction judgments for frame/side {key} in {path}: {index[key]} and {raw}')
            index[key] = normalized
    if kept_segments == 0:
        raise ValueError(f'agent interaction judgment contained no segments for case={case} target={target_entity_id} sides={sorted(sides)} span={frame_span}: {path}')
    return (index, {'enabled': True, 'path': str(path), 'segments': int(kept_segments), 'frame_side_count': int(len(index)), 'contact_state_counts': dict(state_counts), 'occlusion_relation_counts': dict(occlusion_counts)})

def contact_row_state_from_judgment(judgment: dict[str, Any] | None, *, image_candidate: bool, allow_agent_create: bool) -> tuple[bool, str]:
    from typing import Any
    if not isinstance(judgment, dict):
        return (bool(image_candidate), 'active_contact_patch' if image_candidate else 'missing_contact_patch_row')
    state = str(judgment.get('contact_state') or 'unresolved')
    if state == 'no_contact':
        return (True, 'inactive_no_contact_by_agent_judgment')
    if state in {'likely_contact', 'possible_contact'}:
        if image_candidate or allow_agent_create:
            return (True, 'active_contact_patch')
        return (True, 'inactive_agent_contact_without_image_support')
    if image_candidate:
        return (True, 'active_contact_patch')
    return (True, 'inactive_unresolved_contact_by_agent_judgment')
