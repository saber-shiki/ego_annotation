"""Trajectory evidence qualification without solver ownership."""
from __future__ import annotations
from typing import Any
import numpy as np

CORRECTED_POSE_STATUS = 'corrected_temporal_rigid_pose_graph'
COMPLETED_POSE_STATUS = 'completed_temporal_rigid_pose_uncertain'
STRICT_ROTATION_STEP_DEG = 15.0
CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS = {'max_rotation_step_deg': 18.0, 'max_count': 2, 'max_fraction': 0.015, 'max_translation_step_m': 0.02, 'max_endpoint_observability_score': 0.03}

def formal_object_numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'count': 0, 'median': None, 'p90': None, 'p95': None, 'max': None, 'mean': None}
    return {'count': int(arr.size), 'median': float(np.median(arr)), 'p90': float(np.percentile(arr, 90.0)), 'p95': float(np.percentile(arr, 95.0)), 'max': float(np.max(arr)), 'mean': float(np.mean(arr))}

def formal_object_rotation_observability_summary(observations: list[PoseObservation], minimum_score: float) -> dict[str, Any]:
    from utils.data import PoseObservation
    rows = []
    for observation in observations:
        centered = observation.observed_points_world - observation.observed_points_world.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        largest = max(float(eigenvalues[0]), 1e-12)
        normalized = eigenvalues / largest
        score = float(min(normalized[0] - normalized[1], normalized[1] - normalized[2]))
        rows.append({'frame_idx': int(observation.frame_idx), 'normalized_covariance_eigenvalues': normalized.astype(float).tolist(), 'rotation_observability_score': score, 'observable': bool(score >= float(minimum_score))})
    observable_count = int(sum((row['observable'] for row in rows)))
    return {'minimum_score': float(minimum_score), 'observable_frame_count': observable_count, 'observation_frame_count': int(len(rows)), 'observable_fraction': float(observable_count / max(1, len(rows))), 'score_summary': formal_object_numeric_summary([row['rotation_observability_score'] for row in rows]), 'rows': rows, 'interpretation': 'Conservative covariance-eigenvalue proxy; symmetric, line-like, or near-planar-isotropic visible support cannot establish all rotation axes.'}

def pose_graph_readiness(report: dict[str, Any]) -> dict[str, Any]:
    from typing import Any
    graph = report.get('graph_support') if isinstance(report.get('graph_support'), dict) else {}
    annotation_ready = report.get('annotation_ready')
    support_sufficient = graph.get('sufficient')
    explicitly_unready = annotation_ready is False or support_sufficient is False
    explicitly_ready = annotation_ready is True and support_sufficient is not False
    return {'pose_report_status': report.get('status'), 'annotation_ready': annotation_ready if isinstance(annotation_ready, bool) else None, 'graph_support_sufficient': support_sufficient if isinstance(support_sufficient, bool) else None, 'explicitly_unready': explicitly_unready, 'explicitly_ready': explicitly_ready, 'legacy_readiness_fields_missing': annotation_ready is None and support_sufficient is None}

def sparse_conditional_rotation_tail_decision(*, step_rows: list[dict[str, Any]], accepted_rows_by_idx: dict[int, dict[str, Any]], observability: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Bound a tiny underobservable 15--18 degree tail without changing poses.

    The strict 15 degree tier remains the default. The conditional tier can
    only admit adjacent direct metric-surface rows, never interpolated/held/RGB
    or generated-geometry rows, and every excess transition must be sparse,
    translation-continuous, and weakly or marginally rotation-observable.
    """
    import math
    import numpy as np
    import argparse
    from typing import Any
    strict_limit = float(args.max_rotation_step_deg)
    enabled = bool(getattr(args, 'allow_sparse_conditional_rotation_tail', False))
    conditional_limit = float(getattr(args, 'conditional_max_rotation_step_deg', 18.0))
    maximum_count = int(getattr(args, 'conditional_max_rotation_step_count', 2))
    maximum_fraction = float(getattr(args, 'conditional_max_rotation_step_fraction', 0.015))
    maximum_translation_m = float(getattr(args, 'conditional_max_rotation_step_translation_m', 0.02))
    maximum_endpoint_observability = float(getattr(args, 'conditional_max_endpoint_rotation_observability_score', 0.03))
    numeric_policy = {'strict_max_rotation_step_deg': strict_limit, 'conditional_max_rotation_step_deg': conditional_limit, 'conditional_max_rotation_step_fraction': maximum_fraction, 'conditional_max_translation_step_m': maximum_translation_m, 'conditional_max_endpoint_observability_score': maximum_endpoint_observability}
    if not all((math.isfinite(value) for value in numeric_policy.values())):
        raise RuntimeError(f'non-finite conditional rotation-tail policy: {numeric_policy}')
    if enabled and (not math.isclose(strict_limit, STRICT_ROTATION_STEP_DEG, abs_tol=1e-12)):
        raise RuntimeError(f'conditional rotation tail requires the fixed {STRICT_ROTATION_STEP_DEG:g}-degree strict tier')
    if conditional_limit < strict_limit:
        raise RuntimeError('conditional rotation-step limit must not be below the strict limit')
    if maximum_count < 0 or not 0.0 <= maximum_fraction <= 1.0 or maximum_translation_m < 0.0 or (maximum_endpoint_observability < 0.0):
        raise RuntimeError('invalid conditional rotation-tail bounds')
    configured = {'max_rotation_step_deg': conditional_limit, 'max_count': maximum_count, 'max_fraction': maximum_fraction, 'max_translation_step_m': maximum_translation_m, 'max_endpoint_observability_score': maximum_endpoint_observability}
    broadened = {key: value for (key, value) in configured.items() if value > CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS[key] + 1e-12}
    if broadened:
        raise RuntimeError(f'conditional rotation-tail CLI exceeds immutable policy ceilings: configured={broadened}, ceilings={CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS}')
    observability_by_idx = {int(row['frame_idx']): row for row in observability.get('rows', []) if isinstance(row, dict) and row.get('frame_idx') is not None}
    excess = [row for row in step_rows if float(row.get('rotation_step_deg') or 0.0) > strict_limit]
    evaluated = []
    for step in excess:
        from_idx = int(step['from_frame_idx'])
        to_idx = int(step['to_frame_idx'])
        previous = accepted_rows_by_idx.get(from_idx, {})
        current = accepted_rows_by_idx.get(to_idx, {})
        previous_score = observability_by_idx.get(from_idx, {}).get('rotation_observability_score')
        current_score = observability_by_idx.get(to_idx, {}).get('rotation_observability_score')
        endpoint_scores_available = bool(isinstance(previous_score, (int, float)) and isinstance(current_score, (int, float)) and np.isfinite([previous_score, current_score]).all())
        direct_metric_rows = bool(previous.get('status') == CORRECTED_POSE_STATUS and current.get('status') == CORRECTED_POSE_STATUS and (previous.get('direct_pose_observation_source') == 'adjacent_observed_metric_surfel_registration') and (current.get('direct_pose_observation_source') == 'adjacent_observed_metric_surfel_registration') and (previous.get('rigid_pose_observation_eligible') is True) and (current.get('rigid_pose_observation_eligible') is True))
        generated_geometry_excluded = bool(previous.get('generated_geometry_pose_evidence_consumed') is False and current.get('generated_geometry_pose_evidence_consumed') is False)
        criteria = {'adjacent_frame_step': int(step.get('frame_gap', -1)) == 1, 'within_conditional_rotation_limit': float(step['rotation_step_deg']) <= conditional_limit, 'within_conditional_translation_limit': float(step['translation_step_m']) <= maximum_translation_m, 'direct_adjacent_metric_surface_rows': direct_metric_rows, 'generated_geometry_pose_evidence_excluded': generated_geometry_excluded, 'endpoint_observability_available': endpoint_scores_available, 'endpoints_weak_or_marginally_observable': bool(endpoint_scores_available and float(previous_score) <= maximum_endpoint_observability and (float(current_score) <= maximum_endpoint_observability))}
        evaluated.append({**step, 'from_rotation_observability_score': previous_score, 'to_rotation_observability_score': current_score, 'criteria': criteria, 'transition_eligible': bool(all(criteria.values()))})
    excess_fraction = float(len(excess) / max(1, len(step_rows)))
    population_criteria = {'conditional_tier_enabled': enabled, 'has_strict_exceedance': bool(excess), 'exceedance_count_within_limit': len(excess) <= maximum_count, 'exceedance_fraction_within_limit': excess_fraction <= maximum_fraction, 'all_exceedances_individually_eligible': bool(excess and all((row['transition_eligible'] for row in evaluated)))}
    accepted = bool(all(population_criteria.values()))
    return {'strict_max_rotation_step_deg': strict_limit, 'strict_gate_passed': not excess, 'conditional_tier_enabled': enabled, 'conditional_max_rotation_step_deg': conditional_limit, 'conditional_max_rotation_step_count': maximum_count, 'conditional_max_rotation_step_fraction': maximum_fraction, 'conditional_max_translation_step_m': maximum_translation_m, 'conditional_max_endpoint_rotation_observability_score': maximum_endpoint_observability, 'immutable_policy_ceilings': dict(CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS), 'strict_exceedance_count': int(len(excess)), 'strict_exceedance_fraction': excess_fraction, 'conditional_tier_applied': accepted, 'gate_passed': bool(not excess or accepted), 'acceptance_mode': 'strict_max_rotation_step' if not excess else 'conditional_sparse_underobservable_rotation_tail' if accepted else 'failed_rotation_step_gate', 'population_criteria': population_criteria, 'conditional_transitions': evaluated, 'trajectory_values_modified_or_clipped': False, 'generated_geometry_pose_evidence_consumed': False, 'uncertainty': 'A sparse direct observed-metric rotation tail exceeds the strict 15-degree tier but remains at or below the bounded 18-degree conditional tier. The original trajectory is preserved without clipping; these transitions are explicitly low confidence.' if accepted else None}

def temporal_readiness_diagnostics(*, pose_rows: list[dict[str, Any]], annotations: dict[str, Any], observations: list[PoseObservation], full_timeline_completion: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from utils.data import PoseObservation
    from scipy.spatial.transform import Rotation
    import numpy as np
    import argparse
    from typing import Any
    timeline = sorted((int(frame['frame_idx']) for frame in annotations.get('frames', []) if isinstance(frame, dict) and frame.get('frame_idx') is not None and (args.frame_start is None or int(frame['frame_idx']) >= int(args.frame_start)) and (args.frame_end is None or int(frame['frame_idx']) <= int(args.frame_end))))
    timeline_set = set(timeline)
    accepted = []
    for row in pose_rows:
        idx = int(row.get('frame_idx', -1))
        if idx not in timeline_set or row.get('status') not in {CORRECTED_POSE_STATUS, COMPLETED_POSE_STATUS}:
            continue
        rotation = np.asarray(row.get('rotation_world_from_completed_canonical_matrix'), dtype=np.float64)
        translation = np.asarray(row.get('translation_world_m'), dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or (not np.isfinite(rotation).all()) or (not np.isfinite(translation).all()):
            continue
        accepted.append((idx, rotation, translation, row))
    accepted.sort(key=lambda value: value[0])
    direct_frames = sorted((int(observation.frame_idx) for observation in observations if observation.frame_idx in timeline_set))
    direct_fraction = float(len(direct_frames) / max(1, len(timeline)))
    internal_gaps = [direct_frames[i] - direct_frames[i - 1] for i in range(1, len(direct_frames))]
    max_direct_gap = max(internal_gaps) if internal_gaps else 0 if direct_frames else None
    leading_gap = direct_frames[0] - timeline[0] if direct_frames and timeline else None
    trailing_gap = timeline[-1] - direct_frames[-1] if direct_frames and timeline else None
    rotation_steps_deg = []
    translation_steps_m = []
    step_rows = []
    for (previous, current) in zip(accepted[:-1], accepted[1:]):
        (prev_idx, prev_rotation, prev_translation, _) = previous
        (idx, rotation, translation, _) = current
        relative = rotation @ prev_rotation.T
        rotation_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))
        translation_m = float(np.linalg.norm(translation - prev_translation))
        rotation_steps_deg.append(rotation_deg)
        translation_steps_m.append(translation_m)
        step_rows.append({'from_frame_idx': int(prev_idx), 'to_frame_idx': int(idx), 'frame_gap': int(idx - prev_idx), 'rotation_step_deg': rotation_deg, 'translation_step_m': translation_m})
    mode_counts = full_timeline_completion.get('mode_counts') if isinstance(full_timeline_completion.get('mode_counts'), dict) else {}
    completed_count = int(full_timeline_completion.get('completed_row_count') or 0)
    hold_count = int(mode_counts.get('nearest_visible_pose_hold') or 0)
    completed_fraction = float(completed_count / max(1, len(timeline)))
    hold_fraction = float(hold_count / max(1, len(timeline)))
    observability = formal_object_rotation_observability_summary(observations, float(args.min_rotation_observability_score))
    accepted_rows_by_idx = {idx: row for (idx, _rotation, _translation, row) in accepted}
    rotation_step_gate = sparse_conditional_rotation_tail_decision(step_rows=step_rows, accepted_rows_by_idx=accepted_rows_by_idx, observability=observability, args=args)
    max_rotation_step = max(rotation_steps_deg) if rotation_steps_deg else 0.0
    max_translation_step = max(translation_steps_m) if translation_steps_m else 0.0
    failure_reasons = []
    if len(accepted) != len(timeline):
        failure_reasons.append('incomplete_full_timeline_pose_rows')
    if direct_fraction < float(args.min_direct_pose_fraction):
        failure_reasons.append('insufficient_direct_pose_fraction')
    if max_direct_gap is None or max_direct_gap > int(args.max_direct_pose_gap_frames):
        failure_reasons.append('direct_pose_gap_too_large')
    if leading_gap is None or leading_gap > int(args.max_rigid_pose_extrapolation_gap_frames):
        failure_reasons.append('leading_pose_extrapolation_too_large')
    if trailing_gap is None or trailing_gap > int(args.max_rigid_pose_extrapolation_gap_frames):
        failure_reasons.append('trailing_pose_extrapolation_too_large')
    if completed_fraction > float(args.max_completed_pose_fraction):
        failure_reasons.append('completed_pose_fraction_too_large')
    if hold_fraction > float(args.max_nearest_hold_fraction):
        failure_reasons.append('nearest_hold_fraction_too_large')
    if not rotation_step_gate['gate_passed']:
        failure_reasons.append('rotation_step_jump')
    if max_translation_step > float(args.max_translation_step_m):
        failure_reasons.append('translation_step_jump')
    if observability['observable_fraction'] < float(args.min_rotation_observable_fraction):
        failure_reasons.append('insufficient_rotation_observability')
    return {'ready': not failure_reasons, 'failure_reasons': failure_reasons, 'timeline_frame_count': int(len(timeline)), 'accepted_full_timeline_pose_count': int(len(accepted)), 'direct_pose_count': int(len(direct_frames)), 'direct_pose_fraction': direct_fraction, 'direct_frame_span': [direct_frames[0], direct_frames[-1]] if direct_frames else None, 'max_direct_frame_gap': max_direct_gap, 'leading_direct_frame_gap': leading_gap, 'trailing_direct_frame_gap': trailing_gap, 'completed_pose_count': completed_count, 'completed_pose_fraction': completed_fraction, 'nearest_hold_count': hold_count, 'nearest_hold_fraction': hold_fraction, 'completion_mode_counts': mode_counts, 'rotation_step_deg': formal_object_numeric_summary(rotation_steps_deg), 'translation_step_m': formal_object_numeric_summary(translation_steps_m), 'max_rotation_step_deg': max_rotation_step, 'max_translation_step_m': max_translation_step, 'rotation_observability': observability, 'rotation_step_gate': rotation_step_gate, 'conditional_temporal_uncertainty': rotation_step_gate.get('uncertainty'), 'thresholds': {'min_direct_pose_fraction': float(args.min_direct_pose_fraction), 'max_direct_pose_gap_frames': int(args.max_direct_pose_gap_frames), 'max_completed_pose_fraction': float(args.max_completed_pose_fraction), 'max_nearest_hold_fraction': float(args.max_nearest_hold_fraction), 'max_rotation_step_deg': float(args.max_rotation_step_deg), 'allow_sparse_conditional_rotation_tail': bool(getattr(args, 'allow_sparse_conditional_rotation_tail', False)), 'conditional_max_rotation_step_deg': float(getattr(args, 'conditional_max_rotation_step_deg', 18.0)), 'conditional_max_rotation_step_count': int(getattr(args, 'conditional_max_rotation_step_count', 2)), 'conditional_max_rotation_step_fraction': float(getattr(args, 'conditional_max_rotation_step_fraction', 0.015)), 'conditional_max_rotation_step_translation_m': float(getattr(args, 'conditional_max_rotation_step_translation_m', 0.02)), 'conditional_max_endpoint_rotation_observability_score': float(getattr(args, 'conditional_max_endpoint_rotation_observability_score', 0.03)), 'max_translation_step_m': float(args.max_translation_step_m), 'min_rotation_observable_fraction': float(args.min_rotation_observable_fraction), 'min_rotation_observability_score': float(args.min_rotation_observability_score)}, 'step_rows': step_rows}
