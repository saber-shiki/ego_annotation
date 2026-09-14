"""Immutable prediction provenance and run containment."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any
import hashlib

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
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
    if must_exist and not resolved.is_file():
        raise ValueError(f'input file missing: {value}')
    return resolved


_GT_KEYS = {
    "gt_consumed",
    "gt_used_as_solver_input",
    "gt_used_for_initialization",
    "gt_used_for_pose_initialization",
    "gt_used_for_candidate_generation",
    "gt_used_for_solver_input",
    "hot3d_gt_consumed",
    "uses_gt",
    "used_gt",
    "used_ground_truth",
    "ground_truth_consumed",
}


def _key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def find_positive_gt_flags(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _key(key) in _GT_KEYS and child is True:
                found.append(child_path)
            found.extend(find_positive_gt_flags(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(find_positive_gt_flags(child, f"{path}[{index}]"))
    return found


def assert_prediction_only(
    report: Mapping[str, Any],
    *,
    label: str,
    object_id: str | None = None,
    allow_annotation_ready: bool = False,
    allow_nondiagnostic_prediction: bool = False,
    allowed_schema_prefixes: tuple[str, ...] = (),
    allowed_statuses: tuple[str, ...] = (),
) -> None:
    flags = find_positive_gt_flags(report)
    if flags:
        raise RuntimeError(f"{label} has positive GT provenance flags: {flags[:20]}")
    if not allow_annotation_ready and report.get("annotation_ready") is True:
        raise RuntimeError(f"{label} cannot be annotation-ready input")
    if not allow_nondiagnostic_prediction and report.get("diagnostic_only") is False:
        raise RuntimeError(f"{label} is not diagnostic_only")
    if object_id is not None:
        actual = report.get("object_id")
        if actual is not None and str(actual) != str(object_id):
            raise RuntimeError(f"{label} object_id mismatch: {actual!r} != {object_id!r}")
    if allowed_schema_prefixes:
        schema = str(report.get("schema") or "")
        if not any(schema.startswith(prefix) for prefix in allowed_schema_prefixes):
            raise RuntimeError(f"{label} schema is not allowlisted: {schema!r}")
    if allowed_statuses:
        status = str(report.get("status") or "")
        if status not in allowed_statuses:
            raise RuntimeError(f"{label} status is not allowlisted: {status!r}")


def require_equal_object_ids(*named_reports: tuple[str, Mapping[str, Any]]) -> str:
    values: dict[str, str] = {}
    for label, report in named_reports:
        value = report.get("object_id")
        if value is None or not str(value).strip():
            raise RuntimeError(f"{label} has no nonempty object_id")
        values[label] = str(value)
    if len(set(values.values())) != 1:
        raise RuntimeError(f"object_id mismatch: {values}")
    return next(iter(values.values()))


def append_prediction_stage(
    report: dict[str, Any],
    *,
    stage: str,
    input_hashes: Mapping[str, str | None],
    notes: Mapping[str, Any] | None = None,
) -> None:
    chain = report.get("prediction_provenance_chain")
    chain = list(chain) if isinstance(chain, list) else []
    chain.append(
        {
            "stage": stage,
            "gt_consumed": False,
            "input_sha256": dict(input_hashes),
            "notes": dict(notes or {}),
        }
    )
    report["prediction_provenance_chain"] = chain
    report["gt_consumed"] = False
    report["annotation_ready"] = False
    report["diagnostic_only"] = True

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

from utils.data import load
TRANSITION = "P09_DEPTH_ORDER"


def execution_identity(context):
    if context is None: return None
    c = context["contract"]
    value = {"schema": "rgb_scene_recovery_execution_identity_v1", "transition": TRANSITION,
             "birth_bundle_manifest_sha256": c["run_identity"]["bundle_manifest_sha256"],
             "execution_bundle_manifest_sha256": context["execution_bundle_sha256"],
             "recovery_contract_sha256": context["contract_sha256"],
             "source_events_sha256": c["original_events_sha256"],
             "retained_events_sha256": c["retained_events_sha256"],
             "stop_identity": c["stop_identity"]}
    if context.get("validator_revision_sha256") is not None:
        revision = context["validator_revision"]
        value.update(validator_revision_sha256=context["validator_revision_sha256"],
                     prior_execution_bundle_manifest_sha256=revision["prior_execution_bundle_manifest_sha256"],
                     accepted_prior_receipts=revision["accepted_prior_receipts"],
                     validator_revision_terminal_actor_identity=revision["terminal_actor_identity"])
    return value


def input_identity(contract_path, run_root, bundle_manifest, recovery=None):
    contract = load(contract_path)
    if contract.get('schema') != 'rgb_scene_input_v1' or contract.get('run_root') != str(run_root.resolve()):
        raise ValueError('input contract run/schema mismatch')
    actual_bundle_sha = sha(bundle_manifest)
    if sha(in_run(contract['input_video'],run_root)) != contract['input_video_sha256']:
        raise ValueError('input raw/bundle identity changed')
    recovery_json = None
    if actual_bundle_sha != contract['bundle_manifest_sha256']:
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


def require_run_artifact(path, root, role, *, object_id=None, frame_ids=None):
    """Bind a prediction artifact to this raw input and its registered producer.

    Asset references are not recursively classified as predictions: this check
    applies to explicit prediction roles, leaving licensed/model assets external.
    """
    root=Path(root).resolve();path=in_run(path,root)
    contract=load(root/'input/runtime_input_contract.json')
    if contract.get('run_root')!=str(root):
        raise ValueError('input contract belongs to another run')
    raw=in_run(contract['input_video'],root)
    if sha(raw)!=contract['input_video_sha256']:
        raise ValueError('raw video identity changed')
    registry=load(root/'state/producer_receipts.json')
    if (registry.get('run_id')!=contract['run_id'] or
            registry.get('input_video_sha256')!=contract['input_video_sha256']):
        raise ValueError('producer registry raw/run identity mismatch')
    matches=[r for r in registry['records'] if r['path']==str(path)]
    if len(matches)!=1:
        raise ValueError('prediction requires exactly one same-run producer receipt')
    receipt=matches[0]
    if receipt.get('role')!=role or receipt.get('sha256')!=sha(path):
        raise ValueError('prediction role/content differs from producer receipt')
    if object_id is not None and receipt.get('object_id')!=object_id:
        raise ValueError('prediction selected object differs from producer receipt')
    if frame_ids is not None and receipt.get('frame_ids')!=list(map(int,frame_ids)):
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
        self.root=Path(root).resolve()
        self.contract=load(self.root/'input/runtime_input_contract.json')
        self.registry=self.root/'state/producer_receipts.json'
        if self.registry.exists():
            value=load(self.registry)
            if value['run_id']!=self.contract['run_id']:
                raise ValueError('producer ledger belongs to another run')
        else:
            self.registry.parent.mkdir(parents=True,exist_ok=True)
            self.registry.write_text(json.dumps({'run_id':self.contract['run_id'],
                'input_video_sha256':self.contract['input_video_sha256'],'records':[]},indent=2))

    def record(self, stage, outputs, *, object_id=None, frame_ids=(), command=None):
        import uuid
        value=load(self.registry);producer=str(uuid.uuid4())
        existing={r['path'] for r in value['records']}
        records=[]
        for path,role in outputs:
            path=in_run(path,self.root)
            if str(path) in existing:
                raise ValueError('producer attempted to overwrite a registered artifact')
            records.append({'path':str(path),'sha256':sha(path),'role':role,
                'object_id':object_id,'frame_ids':list(map(int,frame_ids)),
                'producer_id':producer,'stage':stage,'command':command})
        value['records'].extend(records)
        temporary=self.registry.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
        temporary.replace(self.registry)
        return producer


def validated_raw_frames(source, identity):
    """Verify decoded RGB bytes against the current raw-video producer manifest."""
    from utils.data import RawFrameSource, indexed
    if not isinstance(source,RawFrameSource):
        raise ValueError('complete export requires a typed current-run raw-frame source')
    root=source.run_root.resolve();manifest=in_run(source.manifest,root)
    payload=load(manifest);frames=indexed(payload['frames']);ids=list(frames)
    if ids!=list(range(len(ids))) or not payload.get('full_source_timeline'):
        raise ValueError('decoded raw source must cover its complete indexed timeline')
    receipt=require_run_artifact(manifest,root,'raw_manifest',frame_ids=ids)
    contract=load(root/'input/runtime_input_contract.json')
    if (contract['run_id']!=identity.run_id or contract['input_video_sha256']!=identity.input_video_sha256
            or payload.get('input_video_sha256')!=identity.input_video_sha256
            or Path(payload['input_video']).resolve()!=Path(contract['input_video']).resolve()):
        raise ValueError('raw manifest belongs to another video/run identity')
    for frame in frames.values():
        path=in_run(frame['raw_frame_path'],root)
        if sha(path)!=frame.get('raw_frame_sha256'):
            raise ValueError('decoded RGB bytes differ from current producer manifest')
        if Path(frame['source_video']).resolve()!=Path(contract['input_video']).resolve():
            raise ValueError('RGB frame declares another source video')
    return frames,{'manifest_sha256':sha(manifest),'producer_id':receipt['producer_id'],
                   'input_video_sha256':identity.input_video_sha256,'frame_count':len(frames)}
