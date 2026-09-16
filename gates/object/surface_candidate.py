"""Candidate-aware validation without weakening original refined-pose assertions."""
from pathlib import Path
import numpy as np
from observations.common import pose_from_values
from observations.object.shape import mesh_arrays, record
from gates.object.envelope import check_references
from gates.object.geometry import validate_posterior
from utils.data import load, indexed, RKEY, TKEY
from utils.state import in_run, sha


def validate_surface_candidate(binding, *, object_id, run_root):
    root = Path(run_root).resolve()
    if binding.get('schema') != 'object_surface_candidate_binding_v1' or binding.get('object_id') != object_id:
        raise ValueError('explicit selected-object surface candidate required')
    if binding.get('generated_mesh_used_as_residual_model') is not True:
        raise ValueError('candidate must disclose generated residual geometry')
    for name in ('annotation_ready', 'physical_authority', 'physical_contact_authority', 'collision_authority', 'signed_distance_authority'):
        if binding.get(name) is not False:
            raise ValueError('surface candidate has no physical authority')
    refs = {name: binding[name] for name in ('source_binding', 'refined_posterior',
        'appearance_mask_manifest', 'source_prior', 'envelope_report', 'surface_fit_report')}
    check_references(refs, root)
    source = load(refs['source_binding']['path'])
    for key in ('case', 'track_id', 'canonical_gauge'):
        if (key in source) != (key in binding) or binding.get(key) != source.get(key):
            raise ValueError('candidate source identity/gauge changed')
    posterior = validate_posterior(load(refs['refined_posterior']['path']), object_id, root)
    if (source.get('schema') != 'rgb_scene_visual_refined_binding_v1' or source.get('object_id') != object_id or
            source.get('pose_rows') != posterior['pose_rows'] or
            source.get('refined_posterior') != refs['refined_posterior'] or
            source.get('appearance_mask_manifest') != refs['appearance_mask_manifest'] or
            binding.get('source_prior') != posterior['source_prior'] or
            binding.get('source_pose_rows') != posterior['source_pose_rows'] or
            binding.get('source_refined_pose_rows') != posterior['pose_rows'] or
            binding.get('source_visual_mesh_receipt') != source['visual_mesh_receipt']):
        raise ValueError('candidate immutable source provenance changed')
    rows, original = indexed(binding['pose_rows']), indexed(posterior['pose_rows'])
    if set(rows) != set(original):
        raise ValueError('candidate must preserve entire source timeline')
    for idx, row in rows.items():
        if pose_from_values(row.get(RKEY), row.get(TKEY)) is None:
            raise ValueError('invalid candidate pose')
        if row.get('time_s') != original[idx].get('time_s'):
            raise ValueError('candidate/source timestamps changed')
    report = load(refs['surface_fit_report']['path'])
    check_references(report['inputs'], root)
    if report.get('object_id') != object_id or report.get('pose_rows') != binding['pose_rows']:
        raise ValueError('candidate pose/report mismatch')
    for report_name, binding_name in (('mesh_binding', 'source_binding'),
            ('refined_posterior', 'refined_posterior'), ('appearance', 'appearance_mask_manifest'),
            ('envelopes', 'envelope_report')):
        if report['inputs'].get(report_name) != refs[binding_name]:
            raise ValueError('candidate report/source mismatch')
    expected_weight = {'surface_no_box': 0., 'surface_weak_box': .15}.get(binding.get('profile'))
    if expected_weight is None or report.get('profile') != binding['profile'] or report.get('box_weight') != expected_weight:
        raise ValueError('named profile composition changed')
    mesh = in_run(binding['render_mesh_contract']['required_mesh_path'], root)
    receipt = binding['visual_mesh_receipt']
    if binding['render_mesh_contract'].get('renderer_must_match') is not True:
        raise ValueError('candidate renderer must match full scaled mesh')
    if receipt.get('generated_mesh_used_as_residual_model') is not True or receipt.get('physical_authority') is not False:
        raise ValueError('candidate receipt geometry authority mismatch')
    for role in ('contact', 'collision', 'sign', 'sdf', 'nonpenetration'):
        if receipt.get(f'generated_faces_{role}_eligible') is not False:
            raise ValueError('candidate generated faces cannot gain physical authority')
    if (sha(mesh) != binding['render_mesh_contract']['required_mesh_sha256'] or
            receipt.get('output_mesh_sha256') != sha(mesh) or Path(receipt['output_mesh_path']).resolve() != mesh):
        raise ValueError('candidate render mesh changed')
    source_mesh = in_run(source['render_mesh_contract']['required_mesh_path'], root)
    if sha(source_mesh) != source['render_mesh_contract']['required_mesh_sha256'] or receipt['source_mesh'] != record(source_mesh):
        raise ValueError('source complete mesh changed')
    v, f = mesh_arrays(source_mesh)
    scaled, sf = mesh_arrays(mesh)
    scale, pivot = binding['shared_uniform_scale'], np.asarray(binding['scale_pivot_canonical_m'])
    if (not np.isfinite(scale) or not .8 <= scale <= 1.35 or pivot.shape != (3,) or
            not np.allclose(pivot, v.mean(axis=0), rtol=0, atol=1e-12) or
            not np.array_equal(f, sf) or scaled.shape != v.shape or
            not np.allclose(scaled, pivot + scale*(v-pivot), rtol=0, atol=1e-6)):
        raise ValueError('candidate full topology/pivot scale replay mismatch')
    if report.get('scale') != scale or report.get('pivot_canonical_m') != pivot.tolist():
        raise ValueError('candidate scale/report mismatch')
    return binding


def validate_surface_state(state, binding, *, run_root):
    """The frozen visual tail must replay the selected mesh/poses, scale baked once."""
    validate_surface_candidate(binding, object_id=binding['object_id'], run_root=run_root)
    ids = np.asarray(state['frame_idx'])
    poses = indexed(binding['pose_rows'])
    if len(ids) != len(poses) or len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != set(poses):
        raise ValueError('candidate state timeline mismatch')
    rotations = np.asarray([poses[int(idx)][RKEY] for idx in ids])
    translations = np.asarray([poses[int(idx)][TKEY] for idx in ids])
    if not np.array_equal(state['time_s'], [poses[int(idx)]['time_s'] for idx in ids]):
        raise ValueError('candidate state timestamps changed')
    for key, expected in (('object_rotation', rotations), ('base_object_rotation', rotations),
                          ('object_translation_world', translations), ('base_object_translation_world', translations)):
        if not np.array_equal(state[key], expected):
            raise ValueError('frozen candidate pose changed: ' + key)
    delta = np.asarray(state.get('object_delta'))
    if delta.shape != (len(ids), 6) or not np.array_equal(delta, np.zeros_like(delta)):
        raise ValueError('frozen candidate object delta must be zero')
    vertices, faces = mesh_arrays(binding['render_mesh_contract']['required_mesh_path'])
    if (not np.array_equal(state['object_faces'], faces) or
            not np.array_equal(state['object_vertices_canonical'], vertices)):
        raise ValueError('candidate state full mesh changed or scale baked twice')
    return state
