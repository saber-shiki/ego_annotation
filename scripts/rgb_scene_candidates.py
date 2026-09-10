"""Generic create-only candidate namespaces for standalone RGB continuation.

A candidate is an isolated suffix of an already prepared run.  Source P01--P08
and the original selection/anchor are read-only; candidate receipts, decisions,
producer events, and meshes live below ``candidates/<candidate_id>``.
"""
from __future__ import annotations
import copy, json, re, shutil
from pathlib import Path
from v19_runtime_contracts import load, sha, in_run

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")

class CandidatePaths:
    """Stable mapping for one create-only candidate suffix."""
    def __init__(self, run_root: Path, candidate_id: str):
        self.run_root = Path(run_root).resolve()
        if not _ID.fullmatch(candidate_id) or candidate_id in {'.', '..'}:
            raise ValueError('unsafe candidate_id')
        self.candidate_id = candidate_id
        self.root = self.run_root / 'candidates' / candidate_id
        self.events = self.root / 'logs' / 'events.jsonl'
        self.contract_path = self.root / 'candidate_contract.json'

    def path(self, relative: str | Path) -> Path:
        value = Path(relative)
        if value.is_absolute() or '..' in value.parts:
            raise ValueError('candidate relative path required')
        return self.root / value

    def output(self, relative: str | Path) -> Path:
        return self.path(relative)

    def source(self, relative: str | Path) -> Path:
        value = Path(relative)
        if value.is_absolute() or '..' in value.parts:
            raise ValueError('source relative path required')
        return self.run_root / value

    @classmethod
    def from_context(cls, context: Path) -> 'CandidatePaths':
        context = Path(context).resolve()
        if context.name == 'candidate_contract.json':
            value = load(context)
            run_root = Path(value['run_root']).resolve()
            candidate = cls(run_root, value['candidate_id'])
            if candidate.contract_path != context:
                raise ValueError('candidate contract path identity mismatch')
            return candidate
        raise ValueError('candidate context must be candidate_contract.json')


def _identity(contract):
    return {k: contract[k] for k in ('run_id', 'run_root', 'input_video_sha256', 'bundle_manifest_sha256')}


def prepare_candidate(run_root: Path, candidate_id: str, *, frame: int | None = None,
                      native_mode: str = 'new', source_native: Path | None = None):
    """Create a candidate contract without mutating the source run."""
    run_root = Path(run_root).resolve(); paths = CandidatePaths(run_root, candidate_id)
    if paths.root.exists(): raise ValueError('candidate_id is create-only and already exists')
    contract_path = run_root / 'input/runtime_input_contract.json'
    source_contract = load(contract_path)
    if source_contract.get('run_root') != str(run_root): raise ValueError('source run identity mismatch')
    proposal = run_root / 'measurements/object_geometry/anchor_candidates'
    proposals = sorted(proposal.glob('*/anchor_candidate_proposals.json'))
    if len(proposals) != 1: raise ValueError('one current selected-object proposal required')
    proposal_value = load(proposals[0])
    ranked = proposal_value.get('ranked_candidates') or proposal_value.get('candidates')
    if not isinstance(ranked, list) or not ranked: raise ValueError('current ranked_candidates required')
    first = ranked[0]
    default_frame = first.get('frame_idx', first.get('frame_index'))
    selected = default_frame if frame is None else frame
    if type(selected) is not int: raise ValueError('candidate frame must be an integer')
    available = {r.get('frame_idx', r.get('frame_index')) for r in ranked}
    if selected not in available: raise ValueError('candidate frame is not in current ranked candidates')
    if native_mode not in {'new', 'reuse'}: raise ValueError('native_mode must be new or reuse')
    if native_mode == 'reuse' and source_native is None: raise ValueError('reuse requires explicit native receipt')
    paths.root.mkdir(parents=True)
    (paths.root/'logs').mkdir()
    object_id = proposal_value.get('object_id') or proposals[0].parent.name
    source_events = run_root/'logs/integration/events.jsonl'
    source_bundle_manifest = run_root/'runtime/bundle_manifest.json'
    source_context = paths.path('state/verified_source_context.json')
    source_context.parent.mkdir(parents=True, exist_ok=True)
    source_context.write_text(json.dumps({
        'schema':'rgb_scene_verified_source_context_v1',
        'source_contract':str(contract_path), 'source_contract_sha256':sha(contract_path),
        'source_proposal':str(proposals[0]), 'source_proposal_sha256':sha(proposals[0]),
        'source_events':str(source_events),
        'source_events_sha256':sha(source_events) if source_events.is_file() else None,
        'source_bundle_manifest':str(source_bundle_manifest),
        'source_bundle_manifest_sha256':sha(source_bundle_manifest) if source_bundle_manifest.is_file() else source_contract.get('bundle_manifest_sha256'),
    }, indent=2)+'\n')
    value = {'schema':'rgb_scene_candidate_v1','candidate_id':candidate_id,
             'run_id':source_contract['run_id'],'run_root':str(run_root),
             'source_contract_sha256':sha(contract_path),'source_proposal_path':str(proposals[0]),
             'source_proposal_sha256':sha(proposals[0]),'object_id':object_id,
             'selected_frame_idx':selected,'selection_mode':'explicit' if frame is not None else 'ranking_first',
             'default_ranked_frame_idx':default_frame,'native_mode':native_mode,
             'source_native':str(Path(source_native).resolve()) if source_native else None,
             'source_identity':_identity(source_contract),
             'source_context_path':str(source_context),'source_context_sha256':sha(source_context),
             'candidate_execution_identity':{'candidate_id':candidate_id,'candidate_contract_sha256':'pending',
                                             'source_bundle_manifest_sha256':source_contract['bundle_manifest_sha256']},
             'selection_requires_visual_confirmation':True}
    paths.contract_path.write_text(json.dumps(value, indent=2)+'\n')
    value['candidate_execution_identity']['candidate_contract_sha256']=sha(paths.contract_path)
    paths.contract_path.write_text(json.dumps(value, indent=2)+'\n')
    paths.events.write_text('')
    return value


def load_candidate(context: Path | CandidatePaths, *, require_exists=True):
    paths = context if isinstance(context, CandidatePaths) else CandidatePaths.from_context(Path(context))
    if require_exists and not paths.contract_path.is_file(): raise ValueError('candidate contract missing')
    value = load(paths.contract_path)
    if value.get('schema') != 'rgb_scene_candidate_v1' or value.get('candidate_id') != paths.candidate_id:
        raise ValueError('candidate contract schema/ID mismatch')
    if value.get('run_root') != str(paths.run_root): raise ValueError('candidate run identity mismatch')
    if type(value.get('selected_frame_idx')) is not int: raise ValueError('candidate frame identity missing')
    return paths, value


def append_event(paths: CandidatePaths, event):
    """Append-only candidate execution stream; failed events are retained too."""
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    with paths.events.open('a') as handle: handle.write(json.dumps(event, sort_keys=True)+'\n')


def _copy_receipt_asset(source: Path, dest: Path):
    source = source.resolve()
    if not source.is_file(): raise ValueError(f'native receipt asset missing: {source}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    if sha(dest) != sha(source): raise ValueError('native receipt asset copy changed bytes')


def import_native_receipt(paths: CandidatePaths, source_qc: Path, execution_receipt: Path,
                          control_contract: Path, image: Path, mask: Path, *, model_identity=None):
    """Materialize a verified complete native result with an explicit QC mapping.

    This is cache reuse, not inference: the source QC/receipt and any retained
    wrapper failure are copied as evidence and never restamped or overwritten.
    """
    _, candidate = load_candidate(paths)
    if candidate['native_mode'] != 'reuse': raise ValueError('candidate is not a reuse candidate')
    source_qc = Path(source_qc).resolve(); execution = Path(execution_receipt).resolve(); control_path=Path(control_contract).resolve()
    image = Path(image).resolve(); mask = Path(mask).resolve()
    source_native = Path(candidate['source_native']).resolve()
    if not source_native.is_dir(): raise ValueError('declared source native directory missing')
    for supplied in (source_qc, execution, control_path):
        if not supplied.is_relative_to(source_native):
            raise ValueError('native receipt/control is outside declared source_native')
    if not source_qc.is_file() or not execution.is_file() or not control_path.is_file() or not image.is_file() or not mask.is_file(): raise ValueError('native import input missing')
    qc = load(source_qc)
    if qc.get('status') != 'ok': raise ValueError('native model receipt is not successful')
    if Path(qc.get('image','')).resolve() != image or Path(qc.get('mask','')).resolve() != mask:
        raise ValueError('native receipt RGB/mask conditioning mismatch')
    execution_value = load(execution)
    control=load(control_path)
    expected_contract=execution_value.get('contract',{})
    if expected_contract.get('path') and Path(expected_contract['path']).resolve()!=control_path:
        raise ValueError('native execution/control contract mismatch')
    if expected_contract.get('sha256') and sha(control_path)!=expected_contract['sha256']:
        raise ValueError('native control contract changed')
    model_execution = execution_value.get('execution') if isinstance(execution_value.get('execution'),dict) else execution_value
    model_returncode = model_execution.get('returncode')
    model_cleanup = model_execution.get('cleanup', execution_value.get('cleanup',{}))
    if model_returncode != 0 or (model_cleanup and model_cleanup.get('status') not in {None,'stopped'}):
        raise ValueError('native model execution did not return zero with stopped cleanup')
    wrapper_disposition = {'returncode':execution_value.get('returncode'),
                           'status':execution_value.get('status'),
                           'failure_path':str(execution.parent/'failure.json') if (execution.parent/'failure.json').is_file() else None}
    if execution_value.get('source_run_id') not in {None,candidate['run_id']}:
        raise ValueError('native receipt source run mismatch')
    if execution_value.get('frame_idx') not in {None,candidate['selected_frame_idx']}:
        raise ValueError('native receipt frame mismatch')
    if execution_value.get('seed') not in {None,qc.get('seed')}:
        raise ValueError('native receipt seed mismatch')
    record_key={'rgb':'rgb','owned_mask':'owned_mask'}
    input_records=control.get('input_records',control.get('inputs',{}))
    for key,path in ((k,p) for k,p in record_key.items() for p in [image if k=='rgb' else mask]):
        record=input_records.get(key,{})
        if record.get('sha256')!=sha(path): raise ValueError(f'native receipt {key} bytes mismatch')
    if control.get('frame_idx')!=candidate['selected_frame_idx'] or control.get('seed')!=qc.get('seed'):
        raise ValueError('native control frame/seed mismatch')
    outputs=execution_value.get('outputs',{})
    native_qc=execution_value.get('native_qc')
    if native_qc and Path(native_qc).resolve()!=source_qc: raise ValueError('native QC path differs from execution receipt')
    if outputs:
        qc_records=[r for r in outputs.values() if isinstance(r,dict) and Path(r.get('path','')).resolve()==source_qc]
        if len(qc_records)!=1 or qc_records[0].get('sha256')!=sha(source_qc):
            raise ValueError('native output differs from execution receipt')
    actual_identity={'seed':qc.get('seed')}
    for source in (qc, execution_value, model_execution):
        if isinstance(source.get('model_identity'),dict): actual_identity.update(source['model_identity'])
    argv=execution_value.get('execution',{}).get('argv',execution_value.get('argv',[]))
    if isinstance(argv,list):
        for flag,key in (('--config','config'),('--moge-checkpoint','moge_checkpoint'),('--dinov2-checkpoint','dinov2_checkpoint')):
            if flag in argv:
                value=Path(argv[argv.index(flag)+1]).resolve();actual_identity[key]=str(value)
                if value.is_file(): actual_identity[key+'_sha256']=sha(value)
    for source,key in (('moge_offline','moge_checkpoint_sha256'),('dinov2_offline','dinov2_checkpoint_sha256')):
        if isinstance(qc.get(source),dict) and qc[source].get('checkpoint_sha256'):
            actual_identity[key]=qc[source]['checkpoint_sha256']
    requested_identity = model_identity
    if not isinstance(requested_identity,dict) or not requested_identity:
        raise ValueError('explicit receipt-bound native model identity required')

    if any(k not in actual_identity for k in requested_identity) or any(requested_identity[k] != actual_identity[k] for k in requested_identity):
        raise ValueError('native model identity is not bound by receipt')
    if 'seed' not in requested_identity or not any(k in requested_identity for k in ('config','config_sha256','moge_checkpoint','moge_checkpoint_sha256','dinov2_checkpoint','dinov2_checkpoint_sha256')):
        raise ValueError('native model identity must include seed and model/config binding')
    names = ('mesh','native_pose_mesh_opencv_camera','native_pose_mesh_pytorch3d_camera','glb')
    if any(not isinstance(qc.get(k), str) for k in names): raise ValueError('complete native mesh receipt required')
    out = paths.path(f'measurements/scene_native/{candidate["object_id"]}/imported')
    if out.exists(): raise ValueError('native import is create-only')
    out.mkdir(parents=True)
    input_out=paths.path('state/native_reuse_inputs'); input_out.mkdir(parents=True,exist_ok=True)
    mapped_image=input_out/('rgb'+image.suffix.lower()); mapped_mask=input_out/('mask'+mask.suffix.lower())
    _copy_receipt_asset(image,mapped_image); _copy_receipt_asset(mask,mapped_mask)
    copied = {}
    for key in names:
        src = Path(qc[key]).resolve()
        matching=[r for r in outputs.values() if isinstance(r,dict) and Path(r.get('path','')).resolve()==src]
        if outputs and (len(matching)!=1 or matching[0].get('sha256')!=sha(src)):
            raise ValueError('native output differs from execution receipt')
        dst = out / src.name
        _copy_receipt_asset(src, dst); copied[key] = str(dst)
    mapped = copy.deepcopy(qc)
    mapped.update(image=str(mapped_image), mask=str(mapped_mask), **copied)
    failure_source=execution.parent/'failure.json'
    failure_copy=None
    if failure_source.is_file():
        failure_copy=out/'source_wrapper_failure.json'; _copy_receipt_asset(failure_source,failure_copy)
    mapped['source_native_receipt'] = {'path':str(source_qc),'sha256':sha(source_qc),
                                      'execution_receipt':str(execution),'execution_receipt_sha256':sha(execution),
                                      'wrapper_failure_preserved':str(failure_copy) if failure_copy else None,
                                      'wrapper_failure_sha256':sha(failure_copy) if failure_copy else None}
    mapped['derived_qc_mapping'] = {'source_qc_sha256':sha(source_qc),'copied_assets':copied,
                                    'input_rgb_sha256':sha(image),'input_mask_sha256':sha(mask),
                                    'model_identity':actual_identity,
                                    'requested_model_identity':model_identity}
    destination = out/'qc_sam3d_objects_mesh_v7.json'; destination.write_text(json.dumps(mapped,indent=2)+'\n')
    result = {'schema':'rgb_scene_native_reuse_v1','status':'ok','candidate_id':paths.candidate_id,
              'qc_path':str(destination),'qc_sha256':sha(destination),
              'source_qc_path':str(source_qc),'source_qc_sha256':sha(source_qc),
              'execution_receipt_path':str(execution),'execution_receipt_sha256':sha(execution),
              'input_rgb_sha256':sha(image),'input_mask_sha256':sha(mask),
              'native_outputs':copied,'failure_preserved':mapped['source_native_receipt']['wrapper_failure_preserved']}
    receipt = paths.path('state/native_reuse_receipt.json'); receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    imported_files=[receipt,destination,mapped_image,mapped_mask,*map(Path,copied.values())]
    if failure_copy: imported_files.append(failure_copy)
    append_event(paths, {'phase':'native_import','candidate_id':paths.candidate_id,
                         'candidate_contract_sha256':candidate.get('candidate_execution_identity',{}).get('candidate_contract_sha256'),
                         'input_contract_sha256':sha(paths.contract_path),
                         'input_sha256':{str(p):sha(p) for p in (source_qc,execution,control_path,image,mask)},
                         'returncode':0,
                         'run_identity':candidate['source_identity'],'outputs':[str(p) for p in imported_files],
                         'output_sha256':{str(p):sha(p) for p in imported_files},
                         'kind':'verified_external_native_import','source_execution_receipt':str(execution),
                         'source_control_contract':str(control_path),
                         'source_control_contract_sha256':sha(control_path),
                         'source_model_identity':actual_identity,
                         'code_sha256':{'scripts/rgb_scene_candidates.py':sha(Path(__file__))}})
    return result
