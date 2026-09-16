"""Immutable shape/pose bindings; generated surfaces do not gain physical roles."""
from __future__ import annotations
from utils.data import load, RKEY, TKEY
from utils.state import sha, in_run
from observations.common import pose_from_values

def validate_posterior(report, object_id, root):
    if report.get('schema') != 'rgb_scene_refined_pose_v1' or report.get('object_id') != object_id:
        raise ValueError('explicit selected-object visual-refined posterior required')
    source = in_run(report['source_prior']['path'], root)
    if sha(source) != report['source_prior']['sha256'] or load(source)['pose_rows'] != report['source_pose_rows']:
        raise ValueError('immutable source prior changed')
    for row in report['pose_rows']:
        if pose_from_values(row.get(RKEY), row.get(TKEY)) is None:
            raise ValueError('missing posterior hypothesis')
    return report

def load_signed_face_authority(source_report: dict[str, Any], *, source_report_path: Path | None, mesh_path: Path, mesh_face_count: int) -> dict[str, Any]:
    """Return a validated authority mask and an audit dictionary.

    New reports opt in with either ``signed_face_authority_required=true`` or
    ``signed_geometry_consumer_policy=local_observation_authority_faces_only``.
    If such a report is malformed, ``usable`` is false; callers must deactivate
    signed physical factors rather than falling back to all faces.
    """
    from typing import Any
    import numpy as np
    from pathlib import Path
    from utils.state import sha256_file
    readiness = source_report.get('geometry_readiness') if isinstance(source_report.get('geometry_readiness'), dict) else {}
    outputs = source_report.get('outputs') if isinstance(source_report.get('outputs'), dict) else {}
    policy = str(readiness.get('signed_geometry_consumer_policy') or '')
    shared_proxy_source = str(readiness.get('signed_geometry_source') or '')
    is_shared_proxy_source = shared_proxy_source == 'shared_prediction_mask_depth_direct_pose_voxel_reconstruction'
    unsafe_shared_all_faces_policy = bool(policy == 'all_faces_of_validated_shared_proxy_signed_eligible' or (is_shared_proxy_source and policy != LOCAL_AUTHORITY_POLICY))
    required = bool(readiness.get('signed_face_authority_required') is True or policy == LOCAL_AUTHORITY_POLICY or unsafe_shared_all_faces_policy)
    declared_ready = readiness.get('signed_geometry_ready') is True
    mesh_path = mesh_path.expanduser().resolve()
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
        raise RuntimeError(f'signed topology mesh is missing or empty: {mesh_path}')
    mesh_sha256 = sha256_file(mesh_path)
    authority_value = readiness.get('signed_face_authority_npz') or outputs.get('signed_face_authority_npz')
    expected_authority_sha256 = str(readiness.get('signed_face_authority_npz_sha256') or (source_report.get('output_sha256') or {}).get('signed_face_authority_npz') or '')
    expected_mesh_sha256 = str(readiness.get('signed_face_authority_mesh_sha256') or '')
    if unsafe_shared_all_faces_policy:
        return {'usable': False, 'required': True, 'state': 'inactive_unsafe_shared_proxy_all_faces_policy_requires_rebuild', 'consumer_policy': policy, 'source_report': None if source_report_path is None else str(source_report_path), 'source_mesh': str(mesh_path), 'source_mesh_sha256': mesh_sha256, 'face_count': int(mesh_face_count), 'eligible_face_count': 0, 'eligible_face_fraction': 0.0, 'mask': np.zeros((int(mesh_face_count),), dtype=bool)}
    if not required:
        return {'usable': True, 'required': False, 'state': 'legacy_all_faces_without_local_authority_contract', 'consumer_policy': policy or 'legacy_unspecified_all_faces', 'source_report': None if source_report_path is None else str(source_report_path), 'source_mesh': str(mesh_path), 'source_mesh_sha256': mesh_sha256, 'face_count': int(mesh_face_count), 'eligible_face_count': int(mesh_face_count), 'eligible_face_fraction': 1.0 if int(mesh_face_count) else 0.0, 'mask': np.ones((int(mesh_face_count),), dtype=bool)}
    if not declared_ready:
        return {'usable': False, 'required': True, 'state': 'inactive_signed_geometry_not_declared_ready', 'consumer_policy': policy, 'source_report': None if source_report_path is None else str(source_report_path), 'source_mesh': str(mesh_path), 'source_mesh_sha256': mesh_sha256, 'face_count': int(mesh_face_count), 'eligible_face_count': 0, 'eligible_face_fraction': 0.0, 'mask': np.zeros((int(mesh_face_count),), dtype=bool)}
    if not authority_value:
        return {'usable': False, 'required': True, 'state': 'missing_required_signed_face_authority_npz', 'consumer_policy': policy, 'source_report': None if source_report_path is None else str(source_report_path), 'source_mesh': str(mesh_path), 'source_mesh_sha256': mesh_sha256, 'face_count': int(mesh_face_count), 'eligible_face_count': 0, 'eligible_face_fraction': 0.0, 'mask': np.zeros((int(mesh_face_count),), dtype=bool)}
    authority_path = Path(str(authority_value)).expanduser().resolve()
    if not authority_path.is_file() or authority_path.stat().st_size <= 0:
        raise RuntimeError(f'required signed face authority NPZ is missing or empty: {authority_path}')
    authority_sha256 = sha256_file(authority_path)
    if not expected_authority_sha256:
        raise RuntimeError('signed face authority report does not bind the NPZ SHA256')
    if authority_sha256 != expected_authority_sha256:
        raise RuntimeError(f'signed face authority NPZ hash mismatch: expected={expected_authority_sha256} actual={authority_sha256}')
    if not expected_mesh_sha256:
        raise RuntimeError('signed face authority report does not bind the topology mesh SHA256')
    if expected_mesh_sha256 != mesh_sha256:
        raise RuntimeError(f'signed face authority report is bound to different topology mesh bytes: expected={expected_mesh_sha256} actual={mesh_sha256}')
    with np.load(authority_path, allow_pickle=False) as archive:
        required_arrays = {'face_id', 'signed_distance_eligible', 'provenance_code', 'source_mesh_sha256', 'source_mesh_face_count', 'authority_schema', 'authority_role', 'readiness_passed'}
        missing = sorted(required_arrays - set(archive.files))
        if missing:
            raise RuntimeError(f'signed face authority NPZ lacks arrays {missing}: {authority_path}')
        face_id = np.asarray(archive['face_id'], dtype=np.int64)
        eligible = np.asarray(archive['signed_distance_eligible'], dtype=bool)
        archive_mesh_sha256 = _scalar_text(archive['source_mesh_sha256'], 'source_mesh_sha256')
        archive_face_count = int(np.asarray(archive['source_mesh_face_count']).reshape(-1)[0])
        provenance_code = np.asarray(archive['provenance_code'], dtype=np.uint8)
        authority_schema = _scalar_text(archive['authority_schema'], 'authority_schema')
        authority_role = _scalar_text(archive['authority_role'], 'authority_role')
        readiness_passed = bool(np.asarray(archive['readiness_passed']).reshape(-1)[0])
    expected_ids = np.arange(int(mesh_face_count), dtype=np.int64)
    if face_id.shape != expected_ids.shape or not np.array_equal(face_id, expected_ids):
        raise RuntimeError(f'signed face authority IDs do not exactly cover mesh faces: {face_id.shape} vs {expected_ids.shape}')
    if eligible.shape != (int(mesh_face_count),):
        raise RuntimeError(f'signed face authority mask shape {eligible.shape} != mesh face count {mesh_face_count}')
    if authority_schema != 'v19_mesh_bound_signed_face_authority_v1':
        raise RuntimeError(f'unsupported signed face authority schema {authority_schema!r}')
    if authority_role != 'selected_collision_surface':
        raise RuntimeError(f'signed face authority has non-consumable role {authority_role!r}')
    if not readiness_passed:
        raise RuntimeError('signed face authority sidecar did not pass geometry readiness')
    if provenance_code.shape != eligible.shape:
        raise RuntimeError('signed face authority provenance_code shape mismatch')
    if not np.array_equal(eligible, provenance_code == 3):
        raise RuntimeError('signed face authority mask disagrees with local-authority provenance code')
    if archive_face_count != int(mesh_face_count):
        raise RuntimeError(f'signed face authority declared face count {archive_face_count} != mesh face count {mesh_face_count}')
    if archive_mesh_sha256 != mesh_sha256:
        raise RuntimeError(f'signed face authority NPZ is bound to different topology mesh bytes: archive={archive_mesh_sha256} actual={mesh_sha256}')
    eligible_count = int(np.count_nonzero(eligible))
    return {'usable': bool(eligible_count > 0), 'required': True, 'state': 'active_mesh_bound_local_signed_face_authority' if eligible_count else 'inactive_zero_eligible_signed_faces', 'consumer_policy': policy, 'source_report': None if source_report_path is None else str(source_report_path), 'source_mesh': str(mesh_path), 'source_mesh_sha256': mesh_sha256, 'authority_npz': str(authority_path), 'authority_npz_sha256': authority_sha256, 'face_count': int(mesh_face_count), 'eligible_face_count': eligible_count, 'eligible_face_fraction': float(eligible_count / max(int(mesh_face_count), 1)), 'mask': eligible, 'provenance_code': provenance_code}
LOCAL_AUTHORITY_POLICY = 'local_observation_authority_faces_only'

def _scalar_text(value: np.ndarray, field: str) -> str:
    import numpy as np
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f'signed face authority field {field} must be scalar, got {array.shape}')
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode('utf-8')
    return str(item)

def validate_completed_mesh_contract(completed_mesh: Path, completion_report: Path | None) -> Path | None:
    from pathlib import Path
    if completed_mesh.name == 'trellis_mesh.ply' or any((part.startswith('trellis_') for part in completed_mesh.parts)):
        raise RuntimeError(f'completed mesh frame mismatch: this solver consumes a P13 completed-canonical mesh, not raw TRELLIS model output ({completed_mesh})')
    if completion_report is None:
        return None
    expected = completion_report_completed_mesh(completion_report)
    if not same_mesh_path(completed_mesh, expected) and completed_mesh.resolve(strict=False) != expected.resolve(strict=False):
        raise RuntimeError(f'completed mesh frame mismatch: pose rows and object-contact constraints are in the P13 completed-canonical frame, but --completed-mesh={completed_mesh} differs from {completion_report} outputs.completed_mesh_labeled={expected}')
    if not completed_mesh.exists() or completed_mesh.stat().st_size <= 0:
        raise RuntimeError(f'completed mesh {completed_mesh} is missing or empty')
    return expected

def same_mesh_path(a: Path, b: Path) -> bool:
    from pathlib import Path
    try:
        return a.exists() and b.exists() and a.samefile(b)
    except OSError:
        return False

def completion_report_completed_mesh(path: Path) -> Path:
    from pathlib import Path
    return completion_report_mesh_contract(path)['pose_hypothesis_mesh']

def completion_report_mesh_contract(path: Path) -> dict[str, Any]:
    from typing import Any
    from pathlib import Path
    from utils.data import load_json
    data = load_json(path)
    outputs = data.get('outputs') if isinstance(data, dict) else None
    if not isinstance(outputs, dict):
        raise RuntimeError(f'completion report {path} has no outputs object')
    pose_value = outputs.get('pose_hypothesis_mesh_labeled') or outputs.get('completed_mesh_labeled') or outputs.get('completed_mesh')
    if not pose_value:
        raise RuntimeError(f'completion report {path} has no pose hypothesis/completed mesh output')
    collision_value = outputs.get('collision_eligible_mesh_labeled')
    readiness = data.get('geometry_readiness') if isinstance(data.get('geometry_readiness'), dict) else {}
    return {'pose_hypothesis_mesh': Path(str(pose_value)), 'physical_surface_mesh': Path(str(collision_value or pose_value)), 'physical_surface_semantics': 'collision_eligible_mesh_labeled' if collision_value else 'legacy_completed_mesh_labeled_unknown_collision_readiness', 'geometry_readiness': readiness, 'signed_geometry_ready': readiness.get('signed_geometry_ready') if isinstance(readiness.get('signed_geometry_ready'), bool) else None, 'legacy_geometry_readiness_fields_missing': readiness.get('signed_geometry_ready') is None, 'completion_report_payload': data, 'completion_report_path': path}

def resolve_physical_surface_contract(completed_mesh: Path, physical_surface_mesh: Path | None, completion_report: Path | None) -> dict[str, Any]:
    from typing import Any
    from pathlib import Path
    validate_completed_mesh_contract(completed_mesh, completion_report)
    if completion_report is None:
        selected = physical_surface_mesh or completed_mesh
        contract = {'pose_hypothesis_mesh': completed_mesh, 'physical_surface_mesh': selected, 'physical_surface_semantics': 'explicit_physical_surface_without_readiness_report' if physical_surface_mesh is not None else 'legacy_completed_mesh_labeled_unknown_collision_readiness', 'geometry_readiness': {}, 'signed_geometry_ready': None, 'legacy_geometry_readiness_fields_missing': True, 'completion_report_payload': {}, 'completion_report_path': None}
    else:
        contract = completion_report_mesh_contract(completion_report)
        expected = contract['physical_surface_mesh']
        selected = physical_surface_mesh or expected
        if not same_mesh_path(selected, expected) and selected.resolve(strict=False) != expected.resolve(strict=False):
            raise RuntimeError(f'physical surface mismatch: object-relative factors must consume the completion report collision-eligible surface; --physical-surface-mesh={selected} differs from {completion_report} physical surface={expected}')
        contract['physical_surface_mesh'] = selected
    selected_path = Path(contract['physical_surface_mesh'])
    if not selected_path.exists() or selected_path.stat().st_size <= 0:
        raise RuntimeError(f'physical surface mesh {selected_path} is missing or empty')
    return contract
import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from utils.state import find_positive_gt_flags

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read_report(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    report = json.loads(data)
    if not isinstance(report, dict):
        raise ValueError(f'expected JSON object: {path}')
    return (report, sha256_bytes(data))

def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f'{label} mismatch: {actual!r} != {expected!r}')

def centroid(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{label} must be a finite metric centroid')
    return result

def require_centroid(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if float(np.linalg.norm(actual - expected)) > 1e-06:
        raise ValueError(f'{label} centroid mismatch (tolerance 1e-6 m)')

def surfels(geom: dict[str, Any], label: str) -> np.ndarray:
    points = np.asarray(geom.get('world_vertices_sample_m'), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6 or (not np.isfinite(points).all()):
        raise ValueError(f'{label} lacks finite metric surfels')
    require_centroid(points.mean(axis=0), centroid(geom.get('centroid_world_m'), label), label)
    return points

def normalize_completion(completion: dict[str, Any], annotations: dict[str, Any], object_id: str) -> dict[str, Any]:
    """Add only compatible aliases after checking the original P11/P09 binding."""
    inputs = completion.get('inputs') or {}
    evidence_value = inputs.get('evidence_report')
    expected_sha = (completion.get('input_hashes') or {}).get('evidence_report')
    if not evidence_value or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError('completion lacks original producer-declared P11 evidence path/SHA256')
    (evidence, actual_sha) = read_report(Path(evidence_value))
    require_equal(actual_sha, expected_sha, 'producer-declared P11 SHA256')
    for (label, report) in [('completion', completion), ('P11 evidence', evidence), ('P09 annotations', annotations)]:
        if find_positive_gt_flags(report):
            raise ValueError(f'{label} has positive GT provenance flags')
    adapter = annotations.get('v19_visible_geometry_adapter') or {}
    for (label, report) in [('completion', completion), ('P11 evidence', evidence), ('P09 adapter', adapter)]:
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
    for (label, geometry) in [('P11 geometry', evidence_geom), ('P09 geometry', geom)]:
        require_equal(geometry.get('frame_idx'), anchor, f'{label} frame')
        require_equal(geometry.get('object_id'), object_id, f'{label} object_id')
    evidence_points = surfels(evidence_geom, 'P11')
    current_points = surfels(geom, 'P09')
    require_centroid(current_points.mean(axis=0), evidence_points.mean(axis=0), 'P09/P11')
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
    aliases = {'candidate_evidence_report': evidence_value, 'candidate_evidence_report_sha256': expected_sha, 'selected_anchor_atomic_binding': binding}
    for (key, value) in aliases.items():
        if key in inputs:
            require_equal(inputs[key], value, f'existing inputs.{key}')
        if key in completion:
            require_equal(completion[key], value, f'existing {key}')
    result = copy.deepcopy(completion)
    result['inputs'].update(copy.deepcopy(aliases))
    return result

def bind_completion_anchor(completion_path: Path, annotations_path: Path, object_id: str, output: Path) -> dict[str, Any]:
    if output.resolve() in {completion_path.resolve(), annotations_path.resolve()} or output.exists():
        raise ValueError('new output report required; source reports are immutable')
    (completion, completion_sha) = read_report(completion_path)
    (annotations, annotations_sha) = read_report(annotations_path)
    result = normalize_completion(completion, annotations, object_id)
    if 'anchor_binding_normalization' in result:
        raise ValueError('completion already has normalization provenance')
    result['anchor_binding_normalization'] = {'method': 'bind_v19_completion_anchor', 'source_completion_report': str(completion_path.resolve()), 'source_completion_report_sha256': completion_sha, 'annotations': str(annotations_path.resolve()), 'annotations_sha256': annotations_sha, 'geometry_modified': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    return result

def observed_front_quality_decision(*, median_fraction: float, p95_fraction: float, native_projection_iou: float, maximum_median_fraction: float, strict_maximum_p95_fraction: float, conditional_maximum_p95_fraction: float, conditional_minimum_projection_iou: float) -> dict[str, Any]:
    """Apply a strict tier plus a bounded, explicitly uncertain P95-tail tier.

    The conditional tier is intentionally narrow: the robust median must still
    satisfy the unchanged strict limit, only the P95 tail may exceed the strict
    limit, and the native image-plane pose must have substantial mask overlap.
    This keeps a small partial-surface tail from suppressing a usable render
    prior without admitting a gross native-pose failure.
    """
    from typing import Any
    import numpy as np
    values = np.asarray([median_fraction, p95_fraction, native_projection_iou, maximum_median_fraction, strict_maximum_p95_fraction, conditional_maximum_p95_fraction, conditional_minimum_projection_iou], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f'non-finite observed-front quality inputs: {values}')
    if conditional_maximum_p95_fraction < strict_maximum_p95_fraction:
        raise RuntimeError('conditional P95 limit must not be below the strict P95 limit')
    median_passed = bool(median_fraction <= maximum_median_fraction)
    strict_p95_passed = bool(p95_fraction <= strict_maximum_p95_fraction)
    conditional_p95_passed = bool(p95_fraction <= conditional_maximum_p95_fraction)
    conditional_projection_passed = bool(native_projection_iou >= conditional_minimum_projection_iou)
    strict_passed = bool(median_passed and strict_p95_passed)
    conditional_tail_passed = bool(not strict_passed and median_passed and (p95_fraction > strict_maximum_p95_fraction) and conditional_p95_passed and conditional_projection_passed)
    if strict_passed:
        mode = 'strict_observed_front_quality'
    elif conditional_tail_passed:
        mode = 'conditional_p95_tail_uncertain_native_projection_supported'
    else:
        mode = 'failed_observed_front_quality'
    return {'quality_passed': bool(strict_passed or conditional_tail_passed), 'acceptance_mode': mode, 'strict_quality_passed': strict_passed, 'conditional_tail_quality_passed': conditional_tail_passed, 'conditional_tail_uncertainty': conditional_tail_passed, 'median_passed': median_passed, 'strict_p95_passed': strict_p95_passed, 'conditional_p95_passed': conditional_p95_passed, 'conditional_projection_passed': conditional_projection_passed, 'strict_maximum_p95_fraction_of_extent_diag': float(strict_maximum_p95_fraction), 'conditional_maximum_p95_fraction_of_extent_diag': float(conditional_maximum_p95_fraction), 'conditional_minimum_native_projection_iou': float(conditional_minimum_projection_iou)}


def require_rigid_tracked_whole(judgment):
    """Retain the selected rigid-workflow gate; do not relabel agent decisions."""
    if judgment['physical_branch'] != 'rigid':
        raise ValueError('selected standalone scene requires rigid tracked-whole evidence; saved judgment branch: '+judgment['physical_branch'])
    if (judgment['relative_component_motion'] != 'not_observed'
            or any(row['relative_motion_observed'] for row in judgment['motion_evidence'])
            or any(row['relative_motion'] == 'observed' or row['attachment_state'] == 'separated'
                   for row in judgment['attached_components'])):
        raise ValueError('rigid label conflicts with saved component-motion evidence; no annotation published')
