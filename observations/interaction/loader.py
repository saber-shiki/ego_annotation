"""P17 observation consumer with explicit IDs; not a policy/action filter."""
from __future__ import annotations
from observations.interaction.adapter import SIDES

def load_agent_interaction_judgments(path: Path | None, *, case: str, target_entity_id: str, frame_span: tuple[int, int], sides: set[str], timeline: list[int] | None = None) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
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
    from observations.interaction.adapter import SCHEMA as ADAPTER_SCHEMA, timeline_ids
    declared = payload.get('frame_ids')
    if declared is not None:
        declared = timeline_ids(declared)
    if timeline is not None:
        timeline = timeline_ids(timeline)
        if declared is not None and declared != timeline:
            raise ValueError('interaction judgment timeline differs from annotation timeline')
    else:
        timeline = declared
    strict_adapter = payload.get('schema') == ADAPTER_SCHEMA
    if strict_adapter and timeline is None:
        raise ValueError('adapted interaction evidence requires an explicit timeline')
    index: dict[tuple[int, str], dict[str, Any]] = {}
    kept_segments = 0
    state_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError(f'agent interaction judgment segment is not an object in {path}: {raw!r}')
        if strict_adapter:
            from data.interaction import validate_adapted_interaction_row
            validate_adapted_interaction_row(raw)
            from observations.interaction.adapter import matches_object_identity
            if not matches_object_identity(raw.get('target_entity_id'), target_entity_id):
                raise ValueError('adapted interaction judgment foreign target identity')
            if raw.get('hand_side') not in SIDES:
                raise ValueError('adapted interaction judgment invalid hand identity')
            bounds = (raw.get('frame_start'), raw.get('frame_end'))
            if any(type(value) is not int or value not in timeline for value in bounds):
                raise ValueError('adapted interaction judgment bounds outside timeline')
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
        for frame_idx in (range(s, e + 1) if timeline is None else (f for f in timeline if s <= f <= e)):
            key = (frame_idx, hand_side)
            if key in index:
                raise ValueError(f'overlapping agent interaction judgments for frame/side {key} in {path}: {index[key]} and {raw}')
            index[key] = normalized
    if kept_segments == 0:
        raise ValueError(f'agent interaction judgment contained no segments for case={case} target={target_entity_id} sides={sorted(sides)} span={frame_span}: {path}')
    return (index, {'enabled': True, 'path': str(path), 'segments': int(kept_segments), 'frame_side_count': int(len(index)), 'contact_state_counts': dict(state_counts), 'occlusion_relation_counts': dict(occlusion_counts)})
