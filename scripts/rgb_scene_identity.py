"""Portable immutable input identity, validated before publishing a final pair."""
from dataclasses import dataclass, asdict
from fractions import Fraction
import json
import math
import re
import uuid
from v19_runtime_contracts import load, sha, in_run


def digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('identity requires SHA256 digest')
    return value


@dataclass(frozen=True)
class InputIdentity:
    run_id: str
    case: str
    input_video_sha256: str
    bundle_manifest_sha256: str
    sensor_calibration_sha256: str | None
    input_contract_sha256: str
    source_timing_json: str
    target_description: str | None
    recovery_execution_json: str | None = None
    candidate_id: str | None = None
    candidate_contract_sha256: str | None = None
    native_reuse_provenance_json: str | None = None

    def __post_init__(self):
        try:
            if str(uuid.UUID(self.run_id)) != self.run_id:
                raise ValueError('noncanonical run UUID')
            if not isinstance(self.case, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', self.case):
                raise ValueError('invalid case')
            for value in (self.input_video_sha256, self.bundle_manifest_sha256, self.input_contract_sha256):
                digest(value)
            if self.sensor_calibration_sha256 is not None:
                digest(self.sensor_calibration_sha256)
            if self.target_description is not None and (not isinstance(self.target_description, str) or not self.target_description.strip()):
                raise ValueError('invalid target description')
            if self.recovery_execution_json is not None:
                recovery = json.loads(self.recovery_execution_json)
                if recovery.get('schema') != 'rgb_scene_recovery_execution_identity_v1':
                    raise ValueError('invalid recovery execution identity')
                for key in ('birth_bundle_manifest_sha256','execution_bundle_manifest_sha256','recovery_contract_sha256','source_events_sha256','retained_events_sha256'):
                    digest(recovery[key])
                if recovery['birth_bundle_manifest_sha256'] != self.bundle_manifest_sha256:
                    raise ValueError('recovery birth bundle identity mismatch')
            if self.candidate_id is not None and (not isinstance(self.candidate_id,str) or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',self.candidate_id) is None):
                raise ValueError('invalid candidate identity')
            if self.candidate_contract_sha256 is not None: digest(self.candidate_contract_sha256)
            timing = json.loads(self.source_timing_json)
            if (type(timing['frame_count']) is not int or timing['frame_count'] <= 0
                    or Fraction(timing['fps_ratio']) <= 0 or Fraction(timing['time_base_s']) <= 0
                    or any(type(timing[k]) not in (int,float) for k in ('first_pts_s','duration_s','max_cfr_grid_error_s'))
                    or not math.isfinite(timing['first_pts_s'])
                    or not math.isfinite(timing['duration_s']) or timing['duration_s'] <= 0
                    or not math.isfinite(timing['max_cfr_grid_error_s']) or timing['max_cfr_grid_error_s'] < 0
                    or not timing['fps_source_field'] or not timing['validation']):
                raise ValueError('invalid source timing')
        except (KeyError, TypeError, AttributeError, ZeroDivisionError) as exc:
            raise ValueError('malformed immutable input identity') from exc

    def document(self):
        result = asdict(self)
        result['source_timing'] = json.loads(result.pop('source_timing_json'))
        recovery = result.pop('recovery_execution_json')
        if recovery is not None:result['recovery_execution_identity'] = json.loads(recovery)
        result['schema'] = 'rgb_scene_input_identity_v1'
        return result


def input_identity(contract_path, run_root, bundle_manifest, recovery=None):
    contract = load(contract_path)
    if contract.get('schema') != 'rgb_scene_input_v1' or contract.get('run_root') != str(run_root.resolve()):
        raise ValueError('input contract run/schema mismatch')
    actual_bundle_sha = sha(bundle_manifest)
    if sha(in_run(contract['input_video'],run_root)) != contract['input_video_sha256']:
        raise ValueError('input raw/bundle identity changed')
    recovery_json = None
    if actual_bundle_sha != contract['bundle_manifest_sha256']:
        from rgb_scene_recovery import execution_identity
        value = execution_identity(recovery)
        if value is None or value['birth_bundle_manifest_sha256'] != contract['bundle_manifest_sha256'] or value['execution_bundle_manifest_sha256'] != actual_bundle_sha:
            raise ValueError('input raw/bundle identity changed')
        recovery_json = json.dumps(value,sort_keys=True,allow_nan=False)
    sensor = contract.get('sensor_calibration_metadata')
    if 'sensor_calibration_sha256' not in contract or (sha(in_run(sensor,run_root)) if sensor else None) != contract['sensor_calibration_sha256']:
        raise ValueError('sensor calibration identity changed')
    return InputIdentity(contract['run_id'], contract['case'], contract['input_video_sha256'],
                         contract['bundle_manifest_sha256'], contract['sensor_calibration_sha256'],
                         sha(contract_path), json.dumps(contract['source_timing'], sort_keys=True, allow_nan=False),
                         contract.get('target_description'), recovery_json)
