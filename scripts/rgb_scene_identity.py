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
        result['schema'] = 'rgb_scene_input_identity_v1'
        return result


def input_identity(contract_path, run_root, bundle_manifest):
    contract = load(contract_path)
    if contract.get('schema') != 'rgb_scene_input_v1' or contract.get('run_root') != str(run_root.resolve()):
        raise ValueError('input contract run/schema mismatch')
    if sha(in_run(contract['input_video'],run_root)) != contract['input_video_sha256'] or sha(bundle_manifest) != contract['bundle_manifest_sha256']:
        raise ValueError('input raw/bundle identity changed')
    sensor = contract.get('sensor_calibration_metadata')
    if 'sensor_calibration_sha256' not in contract or (sha(in_run(sensor,run_root)) if sensor else None) != contract['sensor_calibration_sha256']:
        raise ValueError('sensor calibration identity changed')
    return InputIdentity(contract['run_id'], contract['case'], contract['input_video_sha256'],
                         contract['bundle_manifest_sha256'], contract['sensor_calibration_sha256'],
                         sha(contract_path), json.dumps(contract['source_timing'], sort_keys=True, allow_nan=False),
                         contract.get('target_description'))
