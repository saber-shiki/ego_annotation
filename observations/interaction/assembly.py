"""Join independent image priors for the unchanged P17 consumer; no metric inference."""
from __future__ import annotations

import copy
import importlib
import math

from jsonschema import ValidationError, validate

SIDES = ('left', 'right')
JUDGMENTS = ('contact', 'pads', 'occlusion', 'depth_reliability', 'visibility')
OCCLUSION_MAP = {
    'hand_in_front': 'hand_in_front_of_object',
    'object_in_front': 'object_in_front_of_hand',
    'separate': 'no_visible_occlusion',
    'unresolved': 'unresolved',
}


def interaction_adapter_contract(assets: dict) -> dict:
    """Explicit configuration prior, bound by main's asset/runtime/stage contracts."""
    key = 'contact_support_uncertainty_m'
    value = assets.get(key, .06)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('contact_support_uncertainty_m must be a finite positive configuration prior')
    return {
        'adapter': 'observations.interaction.assembly.v1',
        'contact_support_uncertainty_m': float(value),
        'support_uncertainty_source': 'assets_manifest.' + key if key in assets else 'adapter_default_0.06m',
        'support_uncertainty_kind': 'conservative_configuration_prior_not_a_measurement',
        'missing_contact_prior_probability': .5,
        'missing_contact_prior_kind': 'neutral_configuration_prior_not_a_judgment',
        'physical_authority_granted': False,
    }


def _interval_index(payload, name, ids, object_id):
    rows = payload['interaction_judgments' if name == 'contact' else 'intervals']
    entity_key = 'entity' if name == 'depth_reliability' else 'hand_side'
    index = {}
    for row in rows:
        start, end = row['frame_start'], row['frame_end']
        if type(start) is not int or type(end) is not int or start not in ids or end not in ids or end < start:
            raise ValueError(f'{name}: interval endpoints must belong to the declared timeline')
        if name == 'contact':
            if row['target_entity_id'] not in {object_id, 'object:' + object_id.removeprefix('object:')}:
                raise ValueError('contact: foreign target entity')
            if not math.isfinite(row['contact_prior_probability']):
                raise ValueError('contact: nonfinite probability')
        for frame in ids:
            if start <= frame <= end:
                key = (frame, row[entity_key])
                if key in index:
                    raise ValueError(f'{name}: overlapping evidence for {key}')
                index[key] = row
    return index


def assemble_interaction_batch(*, frame_ids, object_id, judgments, source_hashes, contract):
    """Expand only declared IDs; retain source rows/text and receipts without mutation.

    Visibility is joined as evidence, never used to invent depth order or contact.
    Missing contact is explicit unresolved, but P17's image-candidate behavior is
    intentionally unchanged. No pads are inferred when pad evidence is absent.
    """
    ids = list(frame_ids)
    if not ids or any(type(i) is not int or i < 0 for i in ids) or ids != sorted(set(ids)):
        raise ValueError('assembly requires sorted unique nonnegative integer frame IDs')
    if not isinstance(object_id, str) or not object_id:
        raise ValueError('assembly requires a selected object')
    if set(judgments) != set(JUDGMENTS) or set(source_hashes) != set(JUDGMENTS):
        raise ValueError('assembly requires all five independent judgment sources')
    expected = interaction_adapter_contract({'contact_support_uncertainty_m': contract.get('contact_support_uncertainty_m')})
    expected['support_uncertainty_source'] = contract.get('support_uncertainty_source')
    if contract != expected or expected['support_uncertainty_source'] not in {
        'assets_manifest.contact_support_uncertainty_m', 'adapter_default_0.06m'
    }:
        raise ValueError('unsupported interaction adapter contract')
    sources, indices = {}, {}
    for name in JUDGMENTS:
        payload = judgments[name]
        module = importlib.import_module('observations.judgments.judge_' + ('contact_pads' if name == 'pads' else name))
        try:
            validate({k: v for k, v in payload.items() if k != 'judgment_provenance'}, module.SCHEMA)
        except ValidationError as exc:
            raise ValueError(f'{name}: invalid strict judgment: {exc.message}') from exc
        if payload['object_id'] != object_id:
            raise ValueError(f'{name}: selected object mismatch')
        digest = source_hashes[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'{name}: source SHA256 required')
        sources[name] = {'judgment_sha256': digest, 'provenance': payload.get('judgment_provenance')}
        if name == 'visibility':
            indices[name] = {}
            for row in payload['frames']:
                frame = row['frame_idx']
                if type(frame) is not int or frame not in ids or frame in indices[name]:
                    raise ValueError('visibility: foreign or duplicate frame ID')
                indices[name][frame] = row
        else:
            indices[name] = _interval_index(payload, name, ids, object_id)
    rows = []
    for frame in ids:
        for side in SIDES:
            contact = indices['contact'].get((frame, side))
            pads = indices['pads'].get((frame, side))
            occlusion = indices['occlusion'].get((frame, side))
            hand_depth = indices['depth_reliability'].get((frame, side))
            object_depth = indices['depth_reliability'].get((frame, 'object'))
            depth_state = 'mixed_or_unresolved'
            if object_depth and object_depth['reliability'] == 'supported' and hand_depth:
                if hand_depth['reliability'] == 'supported':
                    depth_state = 'object_depth_reliable'
                elif hand_depth['reliability'] in {'mixed_depth', 'unsupported'}:
                    depth_state = 'hand_depth_unreliable'
            entry = dict(contact) if contact else {
                'contact_state': 'unresolved',
                'contact_prior_probability': contract['missing_contact_prior_probability'],
                'rationale': 'No contact judgment covers this declared frame and hand.',
                'uncertainty': 'Missing contact evidence; neutral configuration prior, not a model judgment.',
            }
            entry.update(target_entity_id='object:' + object_id.removeprefix('object:'),
                hand_side=side, frame_start=frame, frame_end=frame,
                occlusion_relation=OCCLUSION_MAP[occlusion['relation']] if occlusion else 'unresolved',
                depth_reliability=depth_state,
                contact_support_uncertainty_m=contract['contact_support_uncertainty_m'],
                evidence={'adapter_contract': contract, 'contact_evidence_present': contact is not None,
                    'source_judgment_sha256': source_hashes,
                    'contact': contact, 'pads': pads, 'occlusion': occlusion,
                    'hand_depth': hand_depth, 'object_depth': object_depth,
                    'visibility': indices['visibility'].get(frame)})
            if pads is not None:
                entry['primary_contact_pads'] = {'digits': pads['digits'], 'rationale': pads['rationale']}
            rows.append(entry)
    return copy.deepcopy({'interaction_judgments': rows, 'sources': sources})
