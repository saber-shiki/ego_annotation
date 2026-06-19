#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def index_hands(ann: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for frame in ann.get('frames', []) if isinstance(ann.get('frames'), list) else []:
        try:
            frame_idx = int(frame.get('frame_idx'))
        except Exception:
            continue
        for hand in frame.get('hands', []) if isinstance(frame.get('hands'), list) else []:
            if isinstance(hand, dict):
                out[(frame_idx, str(hand.get('hand_side')))] = hand
    return out


def index_rows(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get('constraint_rows', []) if isinstance(report.get('constraint_rows'), list) else []:
        try:
            out[(int(row.get('frame_idx')), str(row.get('hand_side')))] = row
        except Exception:
            continue
    return out


def corrected_flag(hand: dict[str, Any]) -> bool:
    update = hand.get('compact_rigid_object_mano_constraint_update') if isinstance(hand.get('compact_rigid_object_mano_constraint_update'), dict) else {}
    metric = hand.get('metric_mano_state') if isinstance(hand.get('metric_mano_state'), dict) else {}
    return bool(update.get('coordinate_update_applied') is True or metric.get('compact_rigid_object_corrected_h_prime') is True)


def object_constraint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'object_id': row.get('object_id'),
        'surface_mesh_path': row.get('surface_mesh_path') or row.get('mesh_path'),
        'sign_mesh_path': row.get('sign_mesh_path'),
        'sign_mesh_source_report': row.get('sign_mesh_source_report'),
        'completed_surface_mesh_watertight': row.get('completed_surface_mesh_watertight', row.get('completed_mesh_watertight')),
        'sign_mesh_watertight': row.get('sign_mesh_watertight'),
        'observed_band_m': row.get('observed_band_m'),
        'near_surface_vertex_count': row.get('near_surface_vertex_count'),
        'near_surface_vertex_fraction': row.get('near_surface_vertex_fraction'),
        'surface_aabb_candidate_vertex_count': row.get('surface_aabb_candidate_vertex_count', row.get('aabb_candidate_vertex_count')),
        'surface_aabb_candidate_vertex_fraction': row.get('surface_aabb_candidate_vertex_fraction', row.get('aabb_candidate_vertex_fraction')),
        'sign_aabb_candidate_vertex_count': row.get('sign_aabb_candidate_vertex_count'),
        'sign_aabb_candidate_vertex_fraction': row.get('sign_aabb_candidate_vertex_fraction'),
        'penetrating_vertex_count': row.get('penetrating_vertex_count'),
        'nearest_surface_unsigned_m': row.get('nearest_surface_unsigned_m'),
        'signed_distance_m': row.get('signed_distance_m'),
        'application_state': row.get('candidate_application_state'),
        'visible_2d_consistency': row.get('candidate_visible_2d_consistency'),
        'translation_solver': row.get('candidate_translation_solver'),
        'reason': row.get('reason'),
    }


def uncertainty_update(row: dict[str, Any]) -> dict[str, Any]:
    state = str(row.get('candidate_application_state') or '')
    if state == 'no_penetration_no_coordinate_change_needed':
        h_state = 'validated_no_compact_rigid_object_coordinate_change'
        uncertainty: list[str] = []
    else:
        h_state = 'unchanged_with_compact_rigid_object_overlap_uncertainty'
        uncertainty = [
            'compact-rigid lid evidence was measured but no coordinate-changing H-prime is accepted for this row',
            'coordinate update is withheld unless a bounded correction is post-verified to eliminate local signed penetration without degrading visible 2D evidence',
        ]
    return {
        'method': 'build_v18_verified_hprime_annotation',
        'h_prime_state': h_state,
        'h_prime_equals_input_h': True,
        'coordinate_update_applied': False,
        'candidate_translation_world_m': row.get('candidate_translation_world_m'),
        'candidate_translation_norm_m': row.get('candidate_translation_norm_m'),
        'candidate_joint_reprojection_shift_px': row.get('candidate_joint_reprojection_shift_px'),
        'object_constraint': object_constraint_from_row(row),
        'uncertainty_added': uncertainty,
        'scope': 'Verified H-prime selector: unverified or nonconverged compact-rigid corrections remain uncertainty on the original MANO state.',
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--original-annotations', type=Path, required=True)
    ap.add_argument('--candidate-annotations', type=Path, required=True)
    ap.add_argument('--post-verify-report', type=Path, required=True)
    ap.add_argument('--output-annotations', type=Path, required=True)
    ap.add_argument('--summary', type=Path, required=True)
    args = ap.parse_args()

    original = load_json(args.original_annotations)
    candidate = load_json(args.candidate_annotations)
    report = load_json(args.post_verify_report)
    candidate_hands = index_hands(candidate)
    verify_rows = index_rows(report)

    accepted_keys: set[tuple[int, str]] = set()
    corrected_keys = {key for key, hand in candidate_hands.items() if corrected_flag(hand)}
    for key in corrected_keys:
        row = verify_rows.get(key)
        if row and row.get('candidate_application_state') == 'no_penetration_no_coordinate_change_needed':
            accepted_keys.add(key)

    output = copy.deepcopy(original)
    accepted = 0
    uncertainty = 0
    validated_no_change = 0
    corrected_rejected = 0
    for frame in output.get('frames', []) if isinstance(output.get('frames'), list) else []:
        frame_idx = int(frame.get('frame_idx'))
        hands = frame.get('hands', []) if isinstance(frame.get('hands'), list) else []
        for i, hand in enumerate(hands):
            if not isinstance(hand, dict):
                continue
            key = (frame_idx, str(hand.get('hand_side')))
            row = verify_rows.get(key)
            if key in accepted_keys:
                new_hand = copy.deepcopy(candidate_hands[key])
                update = new_hand.get('compact_rigid_object_mano_constraint_update') if isinstance(new_hand.get('compact_rigid_object_mano_constraint_update'), dict) else {}
                update['post_hprime_verification'] = {
                    'application_state': row.get('candidate_application_state') if row else None,
                    'nearest_surface_unsigned_m': row.get('nearest_surface_unsigned_m') if row else None,
                    'signed_distance_m': row.get('signed_distance_m') if row else None,
                    'penetrating_vertex_count': row.get('penetrating_vertex_count') if row else None,
                    'verified_no_additional_coordinate_change': True,
                }
                new_hand['compact_rigid_object_mano_constraint_update'] = update
                if isinstance(new_hand.get('metric_mano_state'), dict):
                    new_hand['metric_mano_state']['compact_rigid_object_constraint_update'] = update
                    new_hand['metric_mano_state']['compact_rigid_object_corrected_h_prime'] = True
                hands[i] = new_hand
                accepted += 1
            elif row is not None:
                upd = uncertainty_update(row)
                if key in corrected_keys:
                    corrected_rejected += 1
                    upd['rejected_candidate_hprime_reason'] = 'candidate correction existed in iterative annotation but failed post-H-prime verification and was reverted to original MANO coordinates'
                hand['compact_rigid_object_mano_constraint_update'] = upd
                if isinstance(hand.get('metric_mano_state'), dict):
                    hand['metric_mano_state']['compact_rigid_object_constraint_update'] = upd
                if upd['h_prime_state'] == 'validated_no_compact_rigid_object_coordinate_change':
                    validated_no_change += 1
                else:
                    uncertainty += 1

    summary = {
        'method': 'build_v18_verified_hprime_annotation',
        'status': 'ok',
        'original_annotations': str(args.original_annotations),
        'candidate_annotations': str(args.candidate_annotations),
        'post_verify_report': str(args.post_verify_report),
        'output_annotations': str(args.output_annotations),
        'corrected_candidate_rows_seen': len(corrected_keys),
        'accepted_verified_hprime_rows': accepted,
        'corrected_candidate_rows_reverted_to_uncertainty': corrected_rejected,
        'uncertainty_rows': uncertainty,
        'validated_no_change_rows': validated_no_change,
        'claim_scope': 'Only candidate H-prime rows that remain no-penetration under the post-correction signed test are kept as coordinate updates; all other measured rows keep original MANO coordinates with uncertainty.',
    }
    write_json(args.output_annotations, output)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
