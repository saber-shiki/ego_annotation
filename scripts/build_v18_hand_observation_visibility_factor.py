#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--ownership-factor-report', type=Path, required=True)
    p.add_argument('--target-entity-id', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--joint-observation-weight-multiplier', type=float, default=0.12)
    return p.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    ownership = load_json(args.ownership_factor_report)
    factor_rows: list[dict[str, Any]] = []
    skipped = Counter()
    for row in as_list(ownership.get('ownership_rows')) if isinstance(ownership, dict) else []:
        if not isinstance(row, dict):
            continue
        if str(row.get('target_entity_id')) != str(args.target_entity_id):
            skipped['wrong_target'] += 1
            continue
        counts = row.get('counts') if isinstance(row.get('counts'), dict) else {}
        candidate_px = int(counts.get('candidate_non_object_owned_px') or counts.get('aligned_hand_entity_overlap_px') or 0)
        hard_px = int(counts.get('non_object_owned_px') or 0)
        hard_state = str(row.get('hard_ownership_state') or '')
        prompt_independent = bool(row.get('hard_ownership_prompt_independent'))
        if candidate_px <= 0:
            skipped['no_candidate'] += 1
            continue
        if hard_px > 0 or prompt_independent:
            skipped['independent_hard_ownership_not_occlusion_conflict'] += 1
            continue
        state = 'active_hand_observation_visibility'
        factor_rows.append({
            'factor_family': 'hand_observation_visibility',
            'target_entity_id': str(args.target_entity_id),
            'frame_idx': int(row['frame_idx']),
            'hand_side': str(row['hand_side']),
            'variable_affected': 'H_t',
            'observation_type': 'visible_object_surface_conflicts_with_mano_prompted_hand_candidate_without_independent_visible_hand',
            'residual_or_quarantine_rule': 'multiply MANO joint/pose visible-observation weights by joint_observation_weight_multiplier for this frame/side; keep visible object first-surface constraints active',
            'rendered_uncertainty_channel': 'ownership review frame renders MANO-prompt self-confirmed hand candidate as orange occluded_or_unresolved, not hard hand-owned',
            'state': state,
            'candidate_px': int(candidate_px),
            'hard_non_object_owned_px': int(hard_px),
            'joint_observation_weight_multiplier': float(args.joint_observation_weight_multiplier),
            'source_hard_ownership_state': hard_state,
            'provenance': {
                'ownership_factor_report': str(args.ownership_factor_report),
                'ownership_review_frame_path': row.get('review_frame_path'),
                'ownership_hand_prompt_source': ((row.get('provenance') or {}).get('hand_prompt_source') if isinstance(row.get('provenance'), dict) else None),
                'reason': 'candidate visible-hand/entity overlap came from MANO-seeded prompt without independent visible-hand confirmation',
            },
        })
    return {
        'method': 'v18_hand_observation_visibility_factor',
        'case': args.case,
        'target_entity_id': args.target_entity_id,
        'claim_scope': 'This factor downweights MANO observation anchoring for frame/side hand hypotheses that conflict with visible object first surfaces without independent visible-hand ownership evidence. It does not claim contact, object pose, or hand-owned object pixels.',
        'inputs': {'ownership_factor_report': str(args.ownership_factor_report)},
        'parameters': {'joint_observation_weight_multiplier': float(args.joint_observation_weight_multiplier)},
        'summary': {
            'factor_row_count': int(len(factor_rows)),
            'candidate_px_sum': int(sum(int(r['candidate_px']) for r in factor_rows)),
            'state_counts': dict(Counter(str(r.get('state')) for r in factor_rows)),
            'skipped_counts': dict(skipped),
        },
        'factor_rows': factor_rows,
    }


def main() -> None:
    args = parse_args()
    payload = build(args)
    out = args.output_root / args.case / 'v18_hand_observation_visibility_factor_report.json'
    write_json(out, payload)
    print(json.dumps({'status': 'ok', 'report': str(out), 'factor_rows': len(payload['factor_rows'])}, indent=2))


if __name__ == '__main__':
    main()
