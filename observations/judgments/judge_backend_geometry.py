"""Independent backend geometry image judgment, fixed to gpt-6-astra."""
import hashlib
import json
import math
from pathlib import Path
from utils.runtime import astra_request, judgment_cli

PROMPT = 'Inspect the requested SAM3D/TRELLIS geometry in the supplied sampled full-frame final_visual overlays on raw backgrounds. Report missing structure, shape hallucination, orientation and scale mismatch independently once per requested backend. Conditioning RGB/mask file references are provenance only, not separately supplied images. Binding poses are not final_visual fitted state; surface candidates are not source posteriors. Do not claim full-timeline validation from samples. Visual plausibility grants no contact/collision/sign/SDF authority.'
SCHEMA = {'type': 'object',
 'properties': {'object_id': {'type': 'string', 'minLength': 1},
                'backends': {'type': 'array',
                             'items': {'type': 'object',
                                       'properties': {'backend': {'type': 'string',
                                                                  'enum': ['sam3d', 'trellis']},
                                                      'visual_quality': {'type': 'string',
                                                                         'enum': ['usable',
                                                                                  'poor',
                                                                                  'unresolved']},
                                                      'rationale': {'type': 'string',
                                                                    'minLength': 1},
                                                      'uncertainty': {'type': 'string',
                                                                      'minLength': 1}},
                                       'required': ['backend',
                                                    'visual_quality',
                                                    'rationale',
                                                    'uncertainty'],
                                       'additionalProperties': False}}},
 'required': ['object_id', 'backends'],
 'additionalProperties': False}


MAX_REQUEST_BYTES = 3_000_000
MAX_EVIDENCE_BYTES = 64 * 1024
COMPACT_SCHEMA = 'backend_geometry_review_compact_v1'


def _file_reference(path, expected_hash):
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_hash:
        raise ValueError('backend source file hash changed: ' + str(path))
    return {'path': str(path), 'sha256': digest, 'bytes': len(raw)}


def _scalar_fields(source, keys):
    # Never copy nested receipts, uncertainty dicts, solver traces or surfels.
    return {key: source[key] for key in keys if key in source
            and (source[key] is None or type(source[key]) in (str, bool, int, float))}


def compact_backend_evidence(binding_path, *, object_id, backends, frame_ids,
                             review_frame_ids, rendered_frames):
    """Read-only allowlist from a legacy or surface-candidate binding FILE.

    frame_ids identifies the entire source timeline; review_frame_ids names only
    the supplied overlays. Binding poses are NOT final_visual fitted state.
    """
    import numpy as np
    from observations.judgments.judge_final_inspection import bind_review_images
    from utils.data import RKEY, TKEY
    binding_path = Path(binding_path).resolve()
    raw = binding_path.read_bytes()
    binding = json.loads(raw)
    if (binding.get('schema') not in {'rgb_scene_visual_refined_binding_v1', 'object_surface_candidate_binding_v1'}
            or binding.get('object_id') != object_id):
        raise ValueError('backend binding schema/object identity mismatch')
    if (not backends or len(set(backends)) != len(backends)
            or not set(backends) <= {'sam3d', 'trellis'}):
        raise ValueError('backend review requires unique known backends')
    ids = [row['frame_idx'] for row in binding['pose_rows']]
    if (not ids or ids != list(frame_ids) or len(set(ids)) != len(ids)
            or any(type(i) is not int or i < 0 for i in ids)):
        raise ValueError('backend full source timeline mismatch')
    if not set(review_frame_ids) <= set(ids):
        raise ValueError('backend reviewed frame outside source timeline')
    mapped = bind_review_images(review_frame_ids, rendered_frames, [r['path'] for r in rendered_frames])
    candidate = binding['schema'] == 'object_surface_candidate_binding_v1'
    roles = {'source_pose_rows': 'original_source_prior',
             'pose_rows': 'generated_mesh_conditioned_surface_candidate' if candidate else 'original_observed_RGB_refined_posterior'}
    if candidate:
        roles['source_refined_pose_rows'] = 'original_observed_RGB_refined_posterior'
    selected = {}
    for key, role in roles.items():
        rows = binding[key]
        if [r['frame_idx'] for r in rows] != ids:
            raise ValueError('backend source/refined/candidate timeline mismatch')
        by_id = {r['frame_idx']: r for r in rows}
        compact = []
        for idx in review_frame_ids:
            row = by_id[idx]
            if 'object_id' in row and row['object_id'] != object_id:
                raise ValueError('backend selected pose object mismatch')
            item = _scalar_fields(row, ('frame_idx', 'time_s', 'status', 'metric_observation',
                'rigid_pose_observation_eligible', 'depth_feasibility_uncertain',
                'relative_evidence_connected', 'uncertainty', 'direct_pose_observation_source'))
            # Fixed-size numeric pose only; missing source poses stay missing.
            for field, shape in ((RKEY, (3, 3)), (TKEY, (3,))):
                value = row.get(field)
                if value is None:
                    item[field] = None
                else:
                    array = np.asarray(value, dtype=float)
                    if array.shape != shape or not np.isfinite(array).all():
                        raise ValueError('invalid selected backend pose')
                    item[field] = array.tolist()
            item['rotation_observability'] = _scalar_fields(row.get('rotation_observability') or {}, ('observable', 'score'))
            item['surface_candidate_support'] = _scalar_fields(row.get('surface_candidate') or {}, ('direct_surface_observation',))
            compact.append(item)
        selected[key] = {'role': role, 'selected_rows': compact}
    receipt = binding['visual_mesh_receipt']
    contract = binding['render_mesh_contract']
    mesh = _file_reference(contract['required_mesh_path'], contract['required_mesh_sha256'])
    if (Path(receipt['output_mesh_path']).resolve() != Path(mesh['path'])
            or receipt['output_mesh_sha256'] != mesh['sha256']):
        raise ValueError('backend mesh receipt/path/hash mismatch')
    refs = {}
    for key in ('source_binding', 'source_prior', 'refined_posterior', 'appearance_mask_manifest',
                'appearance_decision', 'envelope_report', 'surface_fit_report'):
        if key in binding:
            ref = binding[key]
            refs[key] = _file_reference(ref['path'], ref['sha256'])
    source_receipt = binding.get('source_visual_mesh_receipt', receipt)
    if 'native_qc' in source_receipt:
        ref = source_receipt['native_qc']
        refs['native_qc'] = _file_reference(ref['path'], ref['sha256'])
    scale = binding.get('shared_uniform_scale') if candidate else receipt.get('static_image_scale', {}).get('uniform_scale')
    pivot = binding.get('scale_pivot_canonical_m') if candidate else receipt.get('static_image_scale', {}).get('pivot_canonical_m')
    if pivot is not None:
        pivot_array = np.asarray(pivot, dtype=float)
        if pivot_array.shape != (3,) or not np.isfinite(pivot_array).all():
            raise ValueError('invalid backend scale pivot')
        pivot = pivot_array.tolist()
    if scale is not None and (type(scale) not in (int, float) or not math.isfinite(scale)):
        raise ValueError('invalid backend shared scale')
    evidence = {'schema': COMPACT_SCHEMA, 'object_id': object_id, 'backends': list(backends),
        'binding_file': {'path': str(binding_path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)},
        'binding_schema': binding['schema'], 'frame_ids': list(ids), 'source_frame_count': len(ids),
        'review_frame_ids': list(review_frame_ids), 'review_images': mapped,
        'rendered_frames': [{'frame_idx': idx, 'path': row['path'], 'sha256': row['sha256']}
                            for idx, row in zip(review_frame_ids, mapped)],
        'selected_binding_poses': selected, 'source_references': refs,
        'mesh': mesh, 'topology': _scalar_fields(receipt.get('topology', {}),
            ('vertex_count', 'face_count', 'face_index_sha256', 'vertices_float64_sha256', 'faces_exactly_preserved')),
        'shared_uniform_scale': scale, 'scale_pivot_canonical_m': pivot,
        'binding_identity': _scalar_fields(binding, ('case', 'track_id', 'profile')),
        'physical_authority_granted': False,
        'scope': 'Sampled full-frame final_visual overlays on raw backgrounds only; not full-timeline validation. '
                 'No separate conditioning RGB/mask images supplied. File references are provenance, not inspected images. '
                 'Numeric poses are binding estimates, not final_visual fitted state; candidate is not source posterior. '
                 'Generated geometry and visual judgment grant no contact/collision/sign/SDF authority.'}
    if len(json.dumps(evidence, allow_nan=False).encode()) >= MAX_EVIDENCE_BYTES:
        raise ValueError('backend compact evidence exceeds 64KiB; no frames may be dropped')
    return evidence


def review_evidence(evidence, images):
    """Rebuild from immutable source each call, rejecting stale or altered mappings."""
    if evidence.get('schema') != COMPACT_SCHEMA:
        raise ValueError('backend requires compact_backend_evidence from binding file')
    ref = evidence['binding_file']
    if _file_reference(ref['path'], ref['sha256']) != ref:
        raise ValueError('backend binding file identity changed')
    rebuilt = compact_backend_evidence(ref['path'], object_id=evidence['object_id'],
        backends=evidence['backends'], frame_ids=evidence['frame_ids'],
        review_frame_ids=evidence['review_frame_ids'], rendered_frames=evidence['rendered_frames'])
    if rebuilt != evidence:
        raise ValueError('backend compact evidence/frame/hash mapping changed')
    if [str(Path(p).resolve()) for p in images] != [r['path'] for r in rebuilt['review_images']]:
        raise ValueError('backend review image path/order mismatch')
    return rebuilt


def judge(evidence, images, *, audit_dir=None):
    """Consume current-run image evidence and return a validated backend_geometry judgment."""
    evidence = review_evidence(evidence, images)
    result, provenance = astra_request(PROMPT, SCHEMA, evidence, images,
        audit_dir=audit_dir, max_request_bytes=MAX_REQUEST_BYTES, adaptive_review_images=True)
    if 'object_id' in result and result['object_id'] != evidence['object_id']:
        raise ValueError('judgment changed selected object identity')
    allowed = set(evidence['frame_ids'])
    def check_frames(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {'frame_idx','frame_start','frame_end','selected_frame_idx','selected_anchor_frame_idx'} and child not in allowed:
                    raise ValueError('judgment frame outside input timeline')
                if key in {'evidence_frames','detector_prompt_frames','detector_active_interval'} and not set(child) <= allowed:
                    raise ValueError('judgment evidence frame outside input timeline')
                if 'frame_start' in value and value['frame_start'] > value['frame_end']:
                    raise ValueError('reversed judgment interval')
                check_frames(child)
        elif isinstance(value, list):
            for child in value:check_frames(child)
    check_frames(result)
    returned = [row['backend'] for row in result['backends']]
    if len(returned) != len(evidence['backends']) or set(returned) != set(evidence['backends']):
        raise ValueError('judgment must return exactly once per requested backend')
    result['judgment_provenance'] = {'kind':'current_run_visual_judgment','judgment':'backend_geometry',**provenance}
    return result


if __name__ == '__main__':
    judgment_cli(judge, audit_requests=True)
