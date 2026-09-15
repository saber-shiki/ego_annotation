"""Immutable prediction provenance and run containment."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any
import hashlib

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda : handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def in_run(value, root, *, must_exist=True):
    root = Path(root).resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f'clip-specific path must be absolute: {value}')
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f'clip-specific input escapes run root: {value}')
    if must_exist and (not resolved.is_file()):
        raise ValueError(f'input file missing: {value}')
    return resolved
_GT_KEYS = {'gt_consumed', 'gt_used_as_solver_input', 'gt_used_for_initialization', 'gt_used_for_pose_initialization', 'gt_used_for_candidate_generation', 'gt_used_for_solver_input', 'hot3d_gt_consumed', 'uses_gt', 'used_gt', 'used_ground_truth', 'ground_truth_consumed'}

def _key(value: Any) -> str:
    return str(value).strip().lower().replace('-', '_')

def find_positive_gt_flags(value: Any, path: str='$') -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for (key, child) in value.items():
            child_path = f'{path}.{key}'
            if _key(key) in _GT_KEYS and child is True:
                found.append(child_path)
            found.extend(find_positive_gt_flags(child, child_path))
    elif isinstance(value, Sequence) and (not isinstance(value, (str, bytes, bytearray))):
        for (index, child) in enumerate(value):
            found.extend(find_positive_gt_flags(child, f'{path}[{index}]'))
    return found

def assert_prediction_only(report: Mapping[str, Any], *, label: str, object_id: str | None=None, allow_annotation_ready: bool=False, allow_nondiagnostic_prediction: bool=False, allowed_schema_prefixes: tuple[str, ...]=(), allowed_statuses: tuple[str, ...]=()) -> None:
    flags = find_positive_gt_flags(report)
    if flags:
        raise RuntimeError(f'{label} has positive GT provenance flags: {flags[:20]}')
    if not allow_annotation_ready and report.get('annotation_ready') is True:
        raise RuntimeError(f'{label} cannot be annotation-ready input')
    if not allow_nondiagnostic_prediction and report.get('diagnostic_only') is False:
        raise RuntimeError(f'{label} is not diagnostic_only')
    if object_id is not None:
        actual = report.get('object_id')
        if actual is not None and str(actual) != str(object_id):
            raise RuntimeError(f'{label} object_id mismatch: {actual!r} != {object_id!r}')
    if allowed_schema_prefixes:
        schema = str(report.get('schema') or '')
        if not any((schema.startswith(prefix) for prefix in allowed_schema_prefixes)):
            raise RuntimeError(f'{label} schema is not allowlisted: {schema!r}')
    if allowed_statuses:
        status = str(report.get('status') or '')
        if status not in allowed_statuses:
            raise RuntimeError(f'{label} status is not allowlisted: {status!r}')

def require_equal_object_ids(*named_reports: tuple[str, Mapping[str, Any]]) -> str:
    values: dict[str, str] = {}
    for (label, report) in named_reports:
        value = report.get('object_id')
        if value is None or not str(value).strip():
            raise RuntimeError(f'{label} has no nonempty object_id')
        values[label] = str(value)
    if len(set(values.values())) != 1:
        raise RuntimeError(f'object_id mismatch: {values}')
    return next(iter(values.values()))

def append_prediction_stage(report: dict[str, Any], *, stage: str, input_hashes: Mapping[str, str | None], notes: Mapping[str, Any] | None=None) -> None:
    chain = report.get('prediction_provenance_chain')
    chain = list(chain) if isinstance(chain, list) else []
    chain.append({'stage': stage, 'gt_consumed': False, 'input_sha256': dict(input_hashes), 'notes': dict(notes or {})})
    report['prediction_provenance_chain'] = chain
    report['gt_consumed'] = False
    report['annotation_ready'] = False
    report['diagnostic_only'] = True
from dataclasses import dataclass, asdict
from fractions import Fraction
import json, math, re, uuid

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
    source_video_path: str | None = None
    source_sensor_calibration_path: str | None = None

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
                for key in ('birth_bundle_manifest_sha256', 'execution_bundle_manifest_sha256', 'recovery_contract_sha256', 'source_events_sha256', 'retained_events_sha256'):
                    digest(recovery[key])
                if recovery['birth_bundle_manifest_sha256'] != self.bundle_manifest_sha256:
                    raise ValueError('recovery birth bundle identity mismatch')
            if self.candidate_id is not None and (not isinstance(self.candidate_id, str) or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', self.candidate_id) is None):
                raise ValueError('invalid candidate identity')
            if self.candidate_contract_sha256 is not None:
                digest(self.candidate_contract_sha256)
            for original in (self.source_video_path, self.source_sensor_calibration_path):
                if original is not None and (not isinstance(original, str) or not original.strip()):
                    raise ValueError('invalid source input path')
            timing = json.loads(self.source_timing_json)
            if type(timing['frame_count']) is not int or timing['frame_count'] <= 0 or Fraction(timing['fps_ratio']) <= 0 or (Fraction(timing['time_base_s']) <= 0) or any((type(timing[k]) not in (int, float) for k in ('first_pts_s', 'duration_s', 'max_cfr_grid_error_s'))) or (not math.isfinite(timing['first_pts_s'])) or (not math.isfinite(timing['duration_s'])) or (timing['duration_s'] <= 0) or (not math.isfinite(timing['max_cfr_grid_error_s'])) or (timing['max_cfr_grid_error_s'] < 0) or (not timing['fps_source_field']) or (not timing['validation']):
                raise ValueError('invalid source timing')
        except (KeyError, TypeError, AttributeError, ZeroDivisionError) as exc:
            raise ValueError('malformed immutable input identity') from exc

    def document(self):
        result = asdict(self)
        result['source_timing'] = json.loads(result.pop('source_timing_json'))
        recovery = result.pop('recovery_execution_json')
        if recovery is not None:
            result['recovery_execution_identity'] = json.loads(recovery)
        result['schema'] = 'rgb_scene_input_identity_v1'
        return result
from utils.data import load
TRANSITION = 'P09_DEPTH_ORDER'

def execution_identity(context):
    if context is None:
        return None
    c = context['contract']
    value = {'schema': 'rgb_scene_recovery_execution_identity_v1', 'transition': TRANSITION, 'birth_bundle_manifest_sha256': c['run_identity']['bundle_manifest_sha256'], 'execution_bundle_manifest_sha256': context['execution_bundle_sha256'], 'recovery_contract_sha256': context['contract_sha256'], 'source_events_sha256': c['original_events_sha256'], 'retained_events_sha256': c['retained_events_sha256'], 'stop_identity': c['stop_identity']}
    if context.get('validator_revision_sha256') is not None:
        revision = context['validator_revision']
        value.update(validator_revision_sha256=context['validator_revision_sha256'], prior_execution_bundle_manifest_sha256=revision['prior_execution_bundle_manifest_sha256'], accepted_prior_receipts=revision['accepted_prior_receipts'], validator_revision_terminal_actor_identity=revision['terminal_actor_identity'])
    return value

def input_identity(contract_path, run_root, bundle_manifest, recovery=None):
    contract = load(contract_path)
    if contract.get('schema') != 'rgb_scene_input_v1' or contract.get('run_root') != str(run_root.resolve()):
        raise ValueError('input contract run/schema mismatch')
    actual_bundle_sha = sha(bundle_manifest)
    if sha(in_run(contract['input_video'], run_root)) != contract['input_video_sha256']:
        raise ValueError('input raw/bundle identity changed')
    recovery_json = None
    if actual_bundle_sha != contract['bundle_manifest_sha256']:
        value = execution_identity(recovery)
        if value is None or value['birth_bundle_manifest_sha256'] != contract['bundle_manifest_sha256'] or value['execution_bundle_manifest_sha256'] != actual_bundle_sha:
            raise ValueError('input raw/bundle identity changed')
        recovery_json = json.dumps(value, sort_keys=True, allow_nan=False)
    sensor = contract.get('sensor_calibration_metadata')
    if 'sensor_calibration_sha256' not in contract or (sha(in_run(sensor, run_root)) if sensor else None) != contract['sensor_calibration_sha256']:
        raise ValueError('sensor calibration identity changed')
    identity = InputIdentity(contract['run_id'], contract['case'], contract['input_video_sha256'], contract['bundle_manifest_sha256'], contract['sensor_calibration_sha256'], sha(contract_path), json.dumps(contract['source_timing'], sort_keys=True, allow_nan=False), contract.get('target_description'), recovery_json, source_video_path=contract.get('source_video_path'), source_sensor_calibration_path=contract.get('source_sensor_calibration_path'))
    stages_path = Path(run_root) / 'state/recovery/stages.json'
    if stages_path.exists():
        journal = load(stages_path)
        if journal['identity']['contract_sha256'] != sha(contract_path):
            raise ValueError('candidate journal input identity mismatch')
        candidate = journal.get('candidate')
        if candidate:
            from dataclasses import replace
            identity = replace(identity, candidate_id=candidate['id'], candidate_contract_sha256=document_sha(candidate))
    return identity

def require_run_artifact(path, root, role, *, object_id=None, frame_ids=None):
    """Bind a prediction artifact to this raw input and its registered producer.

    Asset references are not recursively classified as predictions: this check
    applies to explicit prediction roles, leaving licensed/model assets external.
    """
    root = Path(root).resolve()
    path = in_run(path, root)
    contract = load(root / 'input/runtime_input_contract.json')
    if contract.get('run_root') != str(root):
        raise ValueError('input contract belongs to another run')
    raw = in_run(contract['input_video'], root)
    if sha(raw) != contract['input_video_sha256']:
        raise ValueError('raw video identity changed')
    registry = load(root / 'state/producer_receipts.json')
    if registry.get('run_id') != contract['run_id'] or registry.get('input_video_sha256') != contract['input_video_sha256']:
        raise ValueError('producer registry raw/run identity mismatch')
    matches = [r for r in registry['records'] if r['path'] == str(path)]
    if len(matches) != 1:
        raise ValueError('prediction requires exactly one same-run producer receipt')
    receipt = matches[0]
    if receipt.get('role') != role or receipt.get('sha256') != sha(path):
        raise ValueError('prediction role/content differs from producer receipt')
    if object_id is not None and receipt.get('object_id') != object_id:
        raise ValueError('prediction selected object differs from producer receipt')
    if frame_ids is not None and receipt.get('frame_ids') != list(map(int, frame_ids)):
        raise ValueError('prediction full timeline differs from producer receipt')
    if not receipt.get('producer_id'):
        raise ValueError('missing producer identity')
    return receipt

def sha256_file(path: Path) -> str:
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

class RunState:
    """One raw-input identity and append-only producer records for a fresh run."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.contract = load(self.root / 'input/runtime_input_contract.json')
        self.registry = self.root / 'state/producer_receipts.json'
        if self.registry.exists():
            value = load(self.registry)
            if value['run_id'] != self.contract['run_id'] or value.get('input_video_sha256') != self.contract['input_video_sha256']:
                raise ValueError('producer ledger belongs to another run')
        else:
            self.registry.parent.mkdir(parents=True, exist_ok=True)
            self.registry.write_text(json.dumps({'run_id': self.contract['run_id'], 'input_video_sha256': self.contract['input_video_sha256'], 'records': []}, indent=2))

    def record(self, stage, outputs, *, object_id=None, frame_ids=(), command=None):
        import uuid
        value = load(self.registry)
        producer = str(uuid.uuid4())
        existing = {r['path'] for r in value['records']}
        records = []
        for (path, role) in outputs:
            path = in_run(path, self.root)
            if str(path) in existing:
                raise ValueError('producer attempted to overwrite a registered artifact')
            records.append({'path': str(path), 'sha256': sha(path), 'role': role, 'object_id': object_id, 'frame_ids': list(map(int, frame_ids)), 'producer_id': producer, 'stage': stage, 'command': command})
        value['records'].extend(records)
        temporary = self.registry.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
        temporary.replace(self.registry)
        return producer

def validated_raw_frames(source, identity):
    """Verify decoded RGB bytes against the current raw-video producer manifest."""
    from utils.data import RawFrameSource, indexed
    if not isinstance(source, RawFrameSource):
        raise ValueError('complete export requires a typed current-run raw-frame source')
    root = source.run_root.resolve()
    manifest = in_run(source.manifest, root)
    payload = load(manifest)
    frames = indexed(payload['frames'])
    ids = list(frames)
    if ids != list(range(len(ids))) or not payload.get('full_source_timeline'):
        raise ValueError('decoded raw source must cover its complete indexed timeline')
    receipt = require_run_artifact(manifest, root, 'raw_manifest', frame_ids=ids)
    contract = load(root / 'input/runtime_input_contract.json')
    if contract['run_id'] != identity.run_id or contract['input_video_sha256'] != identity.input_video_sha256 or payload.get('input_video_sha256') != identity.input_video_sha256 or (Path(payload['input_video']).resolve() != Path(contract['input_video']).resolve()):
        raise ValueError('raw manifest belongs to another video/run identity')
    for frame in frames.values():
        path = in_run(frame['raw_frame_path'], root)
        if sha(path) != frame.get('raw_frame_sha256'):
            raise ValueError('decoded RGB bytes differ from current producer manifest')
        if Path(frame['source_video']).resolve() != Path(contract['input_video']).resolve():
            raise ValueError('RGB frame declares another source video')
    return (frames, {'manifest_sha256': sha(manifest), 'producer_id': receipt['producer_id'], 'input_video_sha256': identity.input_video_sha256, 'frame_count': len(frames)})

def content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _stage_json(value):
    """Lossless JSON result types used by the main-owned stage functions."""
    if isinstance(value, Path):
        return {'__stage_path__': str(value)}
    if isinstance(value, tuple):
        return {'__stage_tuple__': [_stage_json(item) for item in value]}
    if isinstance(value, list):
        return [_stage_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stage_json(item) for key, item in value.items()}
    return value


def _stage_value(value):
    if isinstance(value, list):
        return [_stage_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {'__stage_path__'}:
            return Path(value['__stage_path__'])
        if set(value) == {'__stage_tuple__'}:
            return tuple(_stage_value(item) for item in value['__stage_tuple__'])
        return {key: _stage_value(item) for key, item in value.items()}
    return value


def document_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def planned_stage_contract(producer_id, producer_type, *, parameters, inputs, execution):
    """Declarative identity, supplied before execution; never inspect a callback.

    Input paths/values are explicit; StageRecovery additionally binds the full
    immutable dependency prefix (including baseline raw/config/model identity).
    Callers must hash any external stage inputs not covered by that prefix.
    """
    def normalize(value):
        if isinstance(value, Path):
            return str(value.expanduser().resolve())
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError('planned contract keys must be strings')
            return {key: normalize(item) for key, item in value.items()}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f'non-declarative planned contract value: {type(value).__name__}')
    if not producer_id or producer_type not in {'inprocess', 'subprocess', 'astra'}:
        raise ValueError('planned contract requires explicit producer identity/type')
    value = normalize({'schema': 'planned_stage_contract_v1', 'producer_id': producer_id,
        'producer_type': producer_type, 'parameters': parameters, 'inputs': inputs,
        'execution': execution})
    # Reject NaN/Infinity and detach all mutable caller-owned containers.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


class StageRecovery:
    """Same-run linear prefix recovery; main owns stage ordering and computation.

    Files, including auxiliary native outputs, are hashed before reuse. Public
    producer rows retain their original identities. Recovery never synthesizes
    successful model or judgment receipts from loose output files.
    """
    def __init__(self, ledger, identity, *, resume=False, rewind=None, reason=None, candidate=None):
        import fcntl
        if identity.get('agent_model') != 'gpt-6-astra':
            raise ValueError('stage recovery requires Astra model identity')
        self.ledger = ledger
        self.root = ledger.root
        self.control = self.root / 'state/recovery'
        self.control.mkdir(parents=True, exist_ok=True)
        self.lock = (self.control / 'lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError('another process owns this run recovery lock')
        self.path = self.control / 'stages.json'
        self.cache = {}
        self.position = 0
        self.failed = False
        self.identity = identity
        try:
            if resume:
                if not self.path.is_file():
                    raise ValueError('run has no recoverable stage receipts; legacy/frozen imports are refused')
                self.value = load(self.path)
                if self.value.get('schema') != 'same_run_stage_receipts_v2' or any(
                        'planned_contract' not in stage for stage in self.value.get('stages', [])):
                    raise ValueError('journal lacks planned stage contracts; legacy recovery refused (no restamping)')
                if self.value['identity'] != identity:
                    raise ValueError('run/code/model configuration/input identity changed; cross-version recovery refused')
                self._recover(rewind, reason, candidate)
            else:
                if self.path.exists():
                    raise ValueError('cold stage journal already exists')
                self.value = {'schema': 'same_run_stage_receipts_v2', 'identity': identity,
                              'baseline': self.snapshot(), 'stages': []}
                self.save()
        except BaseException:
            self.lock.close()
            raise

    def close(self):
        self.lock.close()

    def save(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.value, indent=2, sort_keys=True, allow_nan=False))
        temporary.replace(self.path)

    def snapshot(self):
        result = {}
        for path in sorted(self.root.rglob('*')):
            relative = str(path.relative_to(self.root))
            if relative.startswith(('quarantine/', 'state/recovery/')) or path == self.ledger.registry:
                continue
            if path.is_symlink():
                # Prediction outputs must not redirect writes or reads outside the run.
                raise ValueError(f'symlink in run artifacts is not recoverable: {relative}')
            if path.is_file():
                info = path.stat()
                stamp = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
                previous = self.cache.get(relative)
                digest = previous[1] if previous and previous[0] == stamp else sha(path)
                self.cache[relative] = (stamp, digest)
                result[relative] = digest
        return result

    def _recover(self, rewind, reason, candidate):
        import shutil
        from datetime import datetime, timezone
        current = self.snapshot()
        for name, digest in self.value['baseline'].items():
            if current.get(name) != digest:
                raise ValueError(f'immutable run input changed: {name}')
        stages = self.value['stages']
        cutoff = len(stages)
        cause = 'continue_missing_stage'
        previous = document_sha(self.identity)
        actual_records = load(self.ledger.registry)['records']
        for index, stage in enumerate(stages):
            valid = (stage['dependency_receipt_sha256'] == previous
                     and stage['receipt_sha256'] == document_sha({k:v for k,v in stage.items() if k != 'receipt_sha256'})
                     and all(current.get(name) == digest for name,digest in stage['outputs'].items())
                     and all(row in actual_records for row in stage['producer_records']))
            if not valid:
                cutoff = index
                cause = 'missing_or_changed_stage_artifact_or_receipt'
                break
            previous = stage['receipt_sha256']
        if rewind is not None:
            if not reason or not reason.strip():
                raise ValueError('explicit rewind requires a nonempty reason')
            matches = [i for i,s in enumerate(stages) if s['name'] == rewind]
            if len(matches) != 1:
                raise ValueError('rewind must name an existing stage exactly (see state/recovery/stages.json)')
            cutoff = min(cutoff, matches[0])
            cause = 'explicit_rewind: ' + reason
        if candidate is not None and (rewind is None or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', candidate)):
            raise ValueError('candidate label requires explicit rewind and a safe name')
        retained = dict(self.value['baseline'])
        retained_records = []
        for stage in stages[:cutoff]:
            retained.update(stage['outputs'])
            retained_records.extend(stage['producer_records'])
        unexpected = set(current) - set(retained)
        # Even a receipt-less worker result is evidence, never proof of success.
        identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + str(uuid.uuid4())[:8]
        archive = self.root / 'quarantine' / identifier
        archive.mkdir(parents=True)
        shutil.copyfile(self.path, archive / 'stages.before.json')
        shutil.copyfile(self.ledger.registry, archive / 'producers.before.json')
        for name in sorted(unexpected):
            source = self.root / name
            destination = archive / 'artifacts' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        # Remove empty generated directories so create-only producers can restart.
        for directory in sorted(self.root.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if directory.is_dir() and not directory.is_relative_to(self.root/'quarantine') and not directory.is_relative_to(self.control):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        self.value['stages'] = stages[:cutoff]
        self.value['continuation'] = {'archive': str(archive), 'reason': cause,
            'retained_receipt_sha256': [stage['receipt_sha256'] for stage in stages[:cutoff]]}
        if rewind is not None:
            self.value['candidate'] = {'id': candidate or ('attempt-' + identifier), 'reason': reason,
                'run_id': self.identity['run_id'], 'parent_candidate': self.value.get('candidate'),
                'branch_stage': rewind, 'retained_receipt_sha256': self.value['continuation']['retained_receipt_sha256']}
        registry = load(self.ledger.registry)
        if registry['records'] != retained_records:
            registry['records'] = retained_records
            self.ledger.registry.write_text(json.dumps(registry, indent=2, allow_nan=False))
        self.save()
        shutil.copyfile(self.path, archive / 'stages.after.json')
        shutil.copyfile(self.ledger.registry, archive / 'producers.after.json')
        (archive / 'recovery.json').write_text(json.dumps({'reason': cause, 'candidate': candidate,
            'identity': self.identity, 'retained_stages': [s['name'] for s in stages[:cutoff]],
            'invalidated_stages': [s['name'] for s in stages[cutoff:]],
            'quarantined_artifact_sha256': {name: current[name] for name in sorted(unexpected)},
            'before_sha256': sha(archive/'stages.before.json'), 'after_sha256': sha(self.path)}, indent=2))
        self.event('recovery', archive=str(archive), reason=cause)

    def event(self, action, **details):
        from datetime import datetime, timezone
        with (self.control/'events.jsonl').open('a') as handle:
            handle.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'action': action, **details})+'\n')

    def run(self, name, producer, *, contract):
        if self.failed:
            raise ValueError('failed stage requires explicit resume before further execution')
        stages = self.value['stages']
        before = self.snapshot()
        expected = dict(self.value['baseline'])
        for prior in stages:
            expected.update(prior['outputs'])
        if before != expected:
            raise ValueError('unreceipted or changed artifacts require explicit resume')
        if contract.get('schema') != 'planned_stage_contract_v1':
            raise ValueError('explicit planned stage contract required')
        planned = planned_stage_contract(contract['producer_id'], contract['producer_type'],
            parameters=contract['parameters'], inputs=contract['inputs'], execution=contract['execution'])
        planned['input_prefix_receipt_sha256'] = (stages[self.position-1]['receipt_sha256']
            if self.position else document_sha({'identity': self.identity, 'baseline': self.value['baseline']}))
        if self.position < len(stages):
            stage = stages[self.position]
            if stage['name'] != name:
                raise ValueError('main stage order differs from committed same-code receipt')
            if stage['planned_contract'] != planned:
                raise ValueError(f'planned stage contract changed for {name}; explicit rewind at this stage or earlier required')
            self.position += 1
            self.event('reuse', stage=name, receipt_sha256=stage['receipt_sha256'])
            return _stage_value(stage['result'])
        if any(stage['name'] == name for stage in stages):
            raise ValueError('stage names must be unique within one run')
        old_records = load(self.ledger.registry)['records']
        self.event('start', stage=name)
        try:
            result = producer()
            after = self.snapshot()
            changed = [key for key,digest in before.items() if after.get(key) != digest]
            if changed:
                raise ValueError('stage mutated earlier immutable artifacts: ' + ', '.join(changed[:10]))
            records = load(self.ledger.registry)['records']
            if records[:len(old_records)] != old_records:
                raise ValueError('stage rewrote prior producer records')
            stage = {'name': name, 'planned_contract': planned,
                'outputs': {key:value for key,value in after.items() if key not in before},
                'producer_records': records[len(old_records):], 'result': _stage_json(result),
                'dependency_receipt_sha256': stages[-1]['receipt_sha256'] if stages else document_sha(self.identity)}
            stage['receipt_sha256'] = document_sha(stage)
            stages.append(stage)
            self.save()
        except BaseException as error:
            self.failed = True
            self.event('failed', stage=name, error_type=type(error).__name__, message=str(error))
            raise
        self.position += 1
        self.event('success', stage=name, receipt_sha256=stage['receipt_sha256'])
        return result


def recovery_provenance(root):
    path = Path(root) / 'state/recovery/stages.json'
    if not path.exists():
        return None
    value = load(path)
    retained = set(value.get('continuation', {}).get('retained_receipt_sha256', []))
    return {'policy': 'same_run_same_code_and_assets_verified_prefix',
        'candidate': value.get('candidate'), 'continuation': value.get('continuation'),
        'stages': [{'name': stage['name'], 'receipt_sha256': stage['receipt_sha256'],
                    'execution': 'retained_original_producer' if stage['receipt_sha256'] in retained else 'executed_in_current_attempt',
                    'producer_ids': list(dict.fromkeys(row['producer_id'] for row in stage['producer_records']))}
                   for stage in value['stages']]}
