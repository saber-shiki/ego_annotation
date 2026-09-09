#!/usr/bin/env python3
"""Copy a producer-byte-bound P13 completion into the strict P14 anchor contract.

No geometry is reconstructed or relabeled. The legacy evidence digest must
already be declared by the producer; a digest observed here cannot create it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from v20_prediction_contracts import find_positive_gt_flags


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_report(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    report = json.loads(data)
    if not isinstance(report, dict):
        raise ValueError(f'expected JSON object: {path}')
    return report, sha256_bytes(data)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f'{label} mismatch: {actual!r} != {expected!r}')


def centroid(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{label} must be a finite metric centroid')
    return result


def require_centroid(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if float(np.linalg.norm(actual - expected)) > 1.0e-6:
        raise ValueError(f'{label} centroid mismatch (tolerance 1e-6 m)')


def surfels(geom: dict[str, Any], label: str) -> np.ndarray:
    points = np.asarray(geom.get('world_vertices_sample_m'), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6 or not np.isfinite(points).all():
        raise ValueError(f'{label} lacks finite metric surfels')
    require_centroid(points.mean(axis=0), centroid(geom.get('centroid_world_m'), label), label)
    return points


def normalize_completion(completion: dict[str, Any], annotations: dict[str, Any],
                         object_id: str) -> dict[str, Any]:
    """Add only compatible aliases after checking the original P11/P09 binding."""
    inputs = completion.get('inputs') or {}
    evidence_value = inputs.get('evidence_report')
    expected_sha = (completion.get('input_hashes') or {}).get('evidence_report')
    if not evidence_value or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError('completion lacks original producer-declared P11 evidence path/SHA256')
    evidence, actual_sha = read_report(Path(evidence_value))
    require_equal(actual_sha, expected_sha, 'producer-declared P11 SHA256')
    for label, report in [('completion', completion), ('P11 evidence', evidence), ('P09 annotations', annotations)]:
        if find_positive_gt_flags(report):
            raise ValueError(f'{label} has positive GT provenance flags')
    adapter = annotations.get('v19_visible_geometry_adapter') or {}
    for label, report in [('completion', completion), ('P11 evidence', evidence), ('P09 adapter', adapter)]:
        require_equal(report.get('object_id'), object_id, f'{label} object_id')
        require_equal(report.get('case'), completion.get('case'), f'{label} case')
    anchor = adapter.get('anchor_frame_idx')
    if type(anchor) is not int or anchor < 0:
        raise ValueError('P09 adapter lacks integer anchor_frame_idx')
    binding = evidence.get('selected_anchor_atomic_binding')
    if not isinstance(binding, dict) or binding.get('validated') is not True:
        raise ValueError('P11 lacks validated selected_anchor_atomic_binding')
    for key in ['selected_frame_idx', 'visible_geometry_frame_idx', 'canonical_surface_frame_idx']:
        require_equal(binding.get(key), anchor, f'P11 binding {key}')
    require_equal(evidence.get('selected_frame_idx'), anchor, 'P11 selected frame')
    selected = evidence.get('selected') or {}
    require_equal(selected.get('frame_idx'), anchor, 'P11 selected row frame')
    evidence_geom = selected.get('visible_geometry_candidate') or {}
    frames = [f for f in annotations.get('frames', []) if f.get('frame_idx') == anchor]
    if len(frames) != 1:
        raise ValueError('P09 must contain exactly one selected anchor frame')
    objects = [o for o in frames[0].get('objects', []) if o.get('object_id') == object_id]
    if len(objects) != 1:
        raise ValueError('P09 anchor must contain exactly one selected object')
    geom = objects[0].get('visible_geometry_candidate') or {}
    for label, geometry in [('P11 geometry', evidence_geom), ('P09 geometry', geom)]:
        require_equal(geometry.get('frame_idx'), anchor, f'{label} frame')
        require_equal(geometry.get('object_id'), object_id, f'{label} object_id')
    evidence_points = surfels(evidence_geom, 'P11')
    current_points = surfels(geom, 'P09')
    require_centroid(current_points.mean(axis=0), evidence_points.mean(axis=0), 'P09/P11')
    # A different surface with the same centroid must not inherit the old lineage.
    if not np.array_equal(current_points, evidence_points):
        raise ValueError('P09/P11 selected anchor surfels changed')
    reconstruction = (evidence.get('depth_fused_object_row') or {}).get('mesh_reconstruction') or {}
    require_equal(reconstruction.get('anchor_frame_idx'), anchor, 'P11 canonical surface frame')
    require_equal(reconstruction.get('atomic_anchor_binding'), True, 'P11 canonical atomic binding')
    canonical_centroid = centroid(reconstruction.get('anchor_centroid_world_m'), 'P11 canonical')
    evidence_centroid = centroid(evidence_geom.get('centroid_world_m'), 'P11')
    require_centroid(canonical_centroid, evidence_centroid, 'P11 canonical/selected')
    error = float(np.linalg.norm(canonical_centroid - evidence_centroid))
    require_equal(binding.get('selected_vs_canonical_centroid_error_m'), error, 'P11 binding centroid error')
    aliases = {'candidate_evidence_report': evidence_value,
               'candidate_evidence_report_sha256': expected_sha,
               'selected_anchor_atomic_binding': binding}
    for key, value in aliases.items():
        if key in inputs:
            require_equal(inputs[key], value, f'existing inputs.{key}')
        if key in completion:
            require_equal(completion[key], value, f'existing {key}')
    result = copy.deepcopy(completion)
    result['inputs'].update(copy.deepcopy(aliases))
    return result


def bind_report(completion_path: Path, annotations_path: Path, object_id: str,
                output: Path) -> dict[str, Any]:
    if output.resolve() in {completion_path.resolve(), annotations_path.resolve()} or output.exists():
        raise ValueError('new output report required; source reports are immutable')
    completion, completion_sha = read_report(completion_path)
    annotations, annotations_sha = read_report(annotations_path)
    result = normalize_completion(completion, annotations, object_id)
    if 'anchor_binding_normalization' in result:
        raise ValueError('completion already has normalization provenance')
    result['anchor_binding_normalization'] = {
        'method': 'bind_v19_completion_anchor',
        'source_completion_report': str(completion_path.resolve()),
        'source_completion_report_sha256': completion_sha,
        'annotations': str(annotations_path.resolve()),
        'annotations_sha256': annotations_sha,
        'geometry_modified': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-report', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bind_report(args.completion_report, args.annotations, args.object_id, args.output)
    print(json.dumps({'output': str(args.output), 'geometry_modified': False}))


if __name__ == '__main__':
    main()
