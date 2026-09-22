"""Observation v2 role/capability contract; classification is not accuracy certification."""
from collections.abc import Mapping
from data.timeline import declared_ids, timeline_times

SCHEMA_V1 = 'ego.annotation.observation_bundle.v1'
SCHEMA_V2 = 'ego.annotation.observation_bundle.v2'
SUPPORTED_SCHEMAS = (SCHEMA_V1, SCHEMA_V2)

# Roles intentionally describe the producer's claim, never a boolean 'trusted'.
CLASSES = {
    'R0': 'source_input_or_metadata', 'R1': 'uncertain_model_prediction',
    'R2': 'derived_observation_evidence', 'R3': 'generated_prior',
    'R4': 'initializer_or_candidate_not_independent_measurement',
    'R5': 'diagnostic_or_encoded_representation',
}
ARTIFACT_CLASSES = {
    'raw_video': ['R0'], 'raw_frames': ['R2'], 'runtime_input_contract': ['R0'],
    'code_manifest': ['R0'], 'model_asset_manifest': ['R0'],
    'camera_calibration': ['R0', 'R1'], 'conditioned_depth': ['R1'],
    'hawor_world': ['R1'], 'mano_bridge': ['R2'], 'object_plan': ['R1'],
    'candidate_identity': ['R1', 'R2'], 'raw_object_masks': ['R1'],
    'owned_object_masks': ['R2'], 'metric_eligibility': ['R2'],
    'visible_annotations': ['R2', 'R4'], 'surface_report': ['R2'],
    'completion_report': ['R2', 'R3'], 'reconstructed_mesh': ['R2', 'R3'],
    'collision_mesh': ['R2'], 'appearance_masks': ['R2'],
    'appearance_anchor_decision': ['R1', 'R2'],
    'interaction_judgments': ['R1', 'R2'], 'interaction_factors': ['R2'],
    'rgb_source_npz': ['R2'], 'rgb_source_report': ['R2'],
    'visual_geometry_report': ['R2', 'R3'], 'visual_mesh_camera': ['R2', 'R3'],
    'visual_native_mesh': ['R3'], 'visual_native_qc': ['R3'],
}
CAPABILITY_SOURCES = {
    'explicit_timeline': (),
    'typed_interaction': ('interaction_judgments',),
    'rgb_motion': ('rgb_source_npz', 'rgb_source_report'),
    'full_visual_geometry': ('visual_geometry_report', 'visual_mesh_camera', 'visual_native_mesh', 'visual_native_qc', 'appearance_masks'),
}


def classification(records):
    unknown = set(records) - ARTIFACT_CLASSES.keys()
    if unknown:
        raise ValueError(f'unclassified observation artifacts: {sorted(unknown)}')
    return {name: {'classes': ARTIFACT_CLASSES[name], 'role': record['role'],
                   'artifact_sha256': record['sha256'], 'accuracy_certified': False,
                   'candidate_dependency': 'none',
                   **({'field_roles': {'visible_geometry_candidate': 'R2',
                                      'reconstructed_geometry_pose': 'R4'}} if name == 'visible_annotations' else {}),
                   **({'authority_rule': 'read calibration_authority; supplied metadata and estimated K are not equivalent'} if name == 'camera_calibration' else {})}
            for name, record in records.items()}


def validate_v2(manifest):
    if manifest.get('schema') != SCHEMA_V2:
        return
    if manifest.get('bundle_version') != 2:
        raise ValueError('v2 requires bundle_version=2')
    if 'frame_ids' not in manifest or 'time_s' not in manifest:
        raise ValueError('v2 requires explicit frame_ids and time_s')
    timeline_times(manifest['time_s'], len(declared_ids(manifest)))
    caps = manifest.get('capabilities')
    if not isinstance(caps, (list, tuple)) or len(set(caps)) != len(caps) or 'explicit_timeline' not in caps:
        raise ValueError('v2 requires unique capabilities including explicit_timeline')
    if set(caps) - CAPABILITY_SOURCES.keys():
        raise ValueError('unsupported observation capability')
    sources = manifest['sources']
    for cap, required in CAPABILITY_SOURCES.items():
        present = set(required).intersection(sources)
        if len(required) > 1 and present:
            if set(required) - sources.keys() or cap not in caps:
                raise ValueError(f'incomplete/undeclared capability: {cap}')
        if cap in caps and set(required) - sources.keys():
            raise ValueError(f'capability {cap} missing bound inputs')
    records = manifest.get('artifacts', {})
    if manifest.get('observation_roles') != classification(records):
        raise ValueError('v2 observation roles do not match bound artifacts')
    roles = {'rgb_source_npz':'rgb_source_arrays','rgb_source_report':'rgb_source_report',
             'visual_geometry_report':'visual_geometry_report','visual_mesh_camera':'visual_full_metric_camera_mesh',
             'visual_native_mesh':'visual_full_native_mesh','visual_native_qc':'visual_native_model_prior',
             'appearance_masks':'appearance_mask_observations'}
    for name, role in roles.items():
        if name in sources and sources[name].get('role') != role:
            raise ValueError('v2 source role mismatch: '+name)
    aliases = {'annotations':'visible_annotations','mano_archive':'hawor_world',
               'object_mesh':'reconstructed_mesh','contact_factors':'interaction_factors'}
    for name, artifact_name in aliases.items():
        if artifact_name in records and sources.get(name) != records[artifact_name]:
            raise ValueError('v2 source alias identity mismatch: '+name)
    for name, record in sources.items():
        if name in records and isinstance(record, Mapping) and record != records[name]:
            raise ValueError(f'v2 source/artifact identity disagreement: {name}')
    # Downstream training keeps its own version contract; v2 does not pretend to be v1.
    if manifest.get('preprocessing_only') is not True:
        raise ValueError('v2 requires explicit preprocessing_only boundary')
