"""Standalone guarded producer boundary and same-code verified resume.

The core live_gpu_probe and its callers are unchanged. Every heavy standalone
subprocess crosses this boundary (nested native reconstruction also rechecks).
"""
from pathlib import Path
import json, subprocess, time, os, hashlib
from v19_runtime_contracts import in_run, load, sha
from rgb_scene_lineage_base import RunLineage
from rgb_scene_candidates import CandidatePaths, load_candidate, append_event
from rgb_scene_recovery import event_code_is_valid
from rgb_scene_process import run_owned, OwnedProcessFailure, OUTER_TIMEOUT_S


def probe(config, gpu, minimum_free_mib=0):
    from v19_runtime_contracts import execution_target, live_gpu_probe
    execution_target(config, gpu)
    policy=config.get('scene_gpu_policy', {'mode':'exclusive'})
    floor=max(minimum_free_mib, int(config.get('sam3d_min_free_mib',12000)), int(policy.get('minimum_free_mib',0)))
    if policy['mode']=='exclusive': return live_gpu_probe(gpu,floor)
    if policy['mode']!='shared_idle' or policy.get('authorized') is not True:
        raise ValueError('standalone shared-idle authorization required')
    done=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=30)
    fields=[v.strip() for v in done.stdout.strip().split(',')]
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=30)
    if len(fields)!=5 or int(fields[0])!=gpu or fields[1]!=policy.get('uuid') or not fields[1].startswith('GPU-'):
        raise ValueError('malformed/mismatched GPU identity')
    record={'gpu':fields,'compute_processes':processes.stdout.splitlines(),'policy':policy,'minimum_free_mib':floor}
    if not 0<=int(fields[4])<=5 or int(fields[3])<floor:
        raise RuntimeError('standalone GPU unavailable: '+json.dumps(record))
    return record


class SceneLineage(RunLineage):
    def __init__(self, root, bundle, runtime_config=None, *, asset_role=None, candidate=None):
        self.candidate = candidate
        self.candidate_value = load_candidate(candidate)[1] if candidate is not None else None
        self.candidate_contract_sha256 = sha(candidate.contract_path) if candidate is not None else None
        super().__init__(root, bundle, runtime_config, asset_role=asset_role,
                         candidate_context=candidate is not None)
        if candidate is not None:
            context_path=Path(self.candidate_value.get('source_context_path',''))
            if not context_path.is_file() or sha(context_path)!=self.candidate_value.get('source_context_sha256'):
                raise ValueError('verified source context changed')
            context=load(context_path)
            streams=context.get('source_event_streams') or [
                {'path':context.get('source_events',''),'sha256':context.get('source_events_sha256')}]
            for stream in streams:
                source_events=Path(stream['path'])
                if not source_events.is_file() or stream['sha256']!=sha(source_events):
                    raise ValueError('verified source event snapshot changed')
                for line in source_events.read_text().splitlines():
                    event=json.loads(line)
                    if (event.get('returncode')==0 and 'error' not in event
                            and event.get('run_identity')==self.identity):
                        for path in event.get('output_sha256',{}): self.records[path]=event
            if candidate.events.exists():
                for line in candidate.events.read_text().splitlines():
                    event=json.loads(line)
                    accepted_import = (event.get('kind')=='verified_external_native_import'
                                       and event.get('candidate_id')==candidate.candidate_id
                                       and event.get('candidate_contract_sha256')==self.candidate_contract_sha256)
                    if (event.get('returncode')==0 and 'error' not in event
                            and event.get('run_identity')==self.identity
                            and (accepted_import or event_code_is_valid(event,self.code,self.recovery))):
                        for path in event.get('output_sha256',{}): self.records[path]=event
        if asset_role in {'demo','export'}:
            from v19_runtime_assets import verify_external_assets
            identity=self.manifest['external_assets']
            names=identity.get('scene_rrd_files')
            if not names:
                raise ValueError('standalone visual/export dependency inventory missing')
            verify_external_assets({'files':{name:identity['files'][name] for name in names},'trees':{}})

    def agent_phase(self,path):
        if self.candidate is not None and Path(path).is_relative_to(self.candidate.root):
            name=Path(path).name
            if 'anchor_decisions' in Path(path).parts:return 'P09_anchor'
            if 'physical_branch_decisions' in Path(path).parts or name=='object_selection.json':return 'P10'
            if 'appearance_decisions' in Path(path).parts:return 'appearance_anchor'
        if Path(path).name == 'object_selection.json': return 'P10'
        if 'appearance_decisions' in Path(path).parts: return 'appearance_anchor'
        return super().agent_phase(path)

    def producer_evidence(self,path,evidence,phase):
        name=str(path)
        if name not in evidence:
            raise ValueError('visual decision missing required evidence: '+name)
        checked=self.require(path)
        event=self.records.get(name,{})
        if event.get('phase')!=phase or name not in event.get('output_sha256',{}):
            raise ValueError('visual evidence has wrong current producer: '+name)
        if self.checked_digest(checked)!=evidence[name]:
            raise ValueError('visual evidence hash changed')
        return event

    def raw_image_evidence(self, evidence, selected=None, require_manifest=False):
        path=self.root/'input/raw_frame_manifest/manifest.json'
        self.require(path)
        producer=self.records.get(str(path),{})
        if producer.get('phase')!='P01' or str(path) not in producer.get('outputs',[]):
            raise ValueError('current P01 manifest producer required')
        if require_manifest:self.producer_evidence(path,evidence,'P01')
        for row in load(path)['frames']:
            name=row.get('raw_frame_path') or row.get('rgb')
            if (selected is None or (row['frame_idx'],name)==selected) and name in evidence:
                if self.producer_evidence(name,evidence,'P01') is producer:
                    return
            if selected is not None and row['frame_idx']==selected[0] and name not in evidence:
                derived=[candidate for candidate in evidence if self.candidate is not None
                         and Path(candidate).is_relative_to(self.candidate.root)
                         and Path(candidate).is_file() and Path(candidate).suffix.lower() in {'.jpg','.jpeg','.png'}]
                if any(self.checked_digest(candidate)==self.checked_digest(name) for candidate in derived):
                    return
        raise ValueError('required manifest-owned current P01 image missing')

    def require_judgment(self,path):
        base = self.candidate.root if self.candidate is not None and Path(path).resolve().is_relative_to(self.candidate.root) else self.root
        phase=self.agent_phase(path)
        if phase not in {'P05','P09_anchor','P10','appearance_anchor'}:
            return super().require_judgment(path)
        from v19_runtime_contracts import validate_references
        payload=load(path);validate_references(payload,self.root)
        source=payload.get('visual_judgment_provenance',{})
        if (source.get('kind')!='current_run_visual_judgment' or source.get('phase')!=phase
                or source.get('run_identity')!=self.identity or not source.get('evidence_sha256')):
            raise ValueError('decision needs current-run visual evidence identity')
        evidence=source['evidence_sha256']
        for name,digest in evidence.items():
            if self.checked_digest(self.require(name))!=digest:raise ValueError('visual evidence changed')
        if phase=='P05':
            self.raw_image_evidence(evidence,require_manifest=True)
            return
        plan=load(self.require(self.root/'measurements/object_candidates/object_plan_agent.json'))
        if len(plan['objects'])!=1:raise ValueError('one selected object required')
        obj=plan['objects'][0]['object_id']
        if phase in {'P09_anchor','appearance_anchor'}:
            directory='anchor_decisions' if phase=='P09_anchor' else 'appearance_decisions'
            if path!=base/f'state/{directory}/{obj}.json' or payload.get('object_id')!=obj:
                raise ValueError('visual decision selected object/path mismatch')
        if phase=='P09_anchor':
            directory=base/f'measurements/object_geometry/anchor_candidates/{obj}'
            proposal=directory/'anchor_candidate_proposals.json'
            producer=self.producer_evidence(proposal,evidence,'P09-propose')
            value=load(proposal);review=directory/'anchor_candidate_review.jpg'
            idx=payload.get('selected_anchor_frame_idx')
            if (value.get('object_id')!=obj or type(idx) is not int
                    or idx not in [r['frame_idx'] for r in value.get('review_candidates',[])]
                    or value.get('outputs',{}).get('anchor_candidate_review')!=str(review)):
                raise ValueError('anchor proposal object/frame/review mismatch')
            if self.producer_evidence(review,evidence,'P09-propose') is not producer:
                raise ValueError('anchor review producer mismatch')
            return
        if phase=='P10':
            annotations=base/f'measurements/object_geometry/visible_geometry/{obj}/annotations_v19_visible_geometry.json'
            producer=self.producer_evidence(annotations,evidence,'P09')
            images=[Path(name) for name in evidence if Path(name).suffix.lower() in {'.jpg','.jpeg','.png'}
                    and self.records.get(name,{}).get('phase')=='P09']
            if not images or not any(self.producer_evidence(image,evidence,'P09') is producer for image in images):
                raise ValueError('P10 decision needs a current P09-owned image')
            if path == base/f'state/physical_branch_decisions/{obj}.json':
                if payload.get('object_id')!=obj or payload.get('physical_branch')!='rigid':
                    raise ValueError('P10 rigid branch identity mismatch')
            elif path == base/'state/object_selection.json':
                anchor=load(self.require(base/f'state/anchor_decisions/{obj}.json'))
                track=plan['objects'][0].get('track_id',obj)
                if (payload.get('object_id')!=obj or payload.get('case')!=self.contract['case']
                        or payload.get('track_id')!=track
                        or payload.get('anchor_frame')!=anchor.get('selected_anchor_frame_idx')):
                    raise ValueError('P10 object selection differs from plan/anchor')
                self.require(base/f'state/physical_branch_decisions/{obj}.json')
            else: raise ValueError('P10 judgment path mismatch')
            return
        manifest=base/f'measurements/demo_visual/{obj}/appearance_evidence/appearance_mask_manifest.json'
        producer=self.producer_evidence(manifest,evidence,'scene_appearance')
        value=load(manifest);idx=payload.get('selected_frame_idx')
        rows=[r for r in value.get('frames',[]) if r.get('frame_idx')==idx]
        if value.get('object_id')!=obj or type(idx) is not int or len(rows)!=1 or rows[0].get('status')!='available':
            raise ValueError('appearance manifest selected object/frame mismatch')
        row=rows[0]
        if self.producer_evidence(row['appearance_mask_path'],evidence,'scene_appearance') is not producer:
            raise ValueError('appearance mask producer mismatch')
        self.raw_image_evidence(evidence,selected=(idx,row['raw_rgb_path']))

    def require_interaction_evidence(self, judgment, evidence):
        base = self.candidate.root if self.candidate is not None and Path(judgment).resolve().is_relative_to(self.candidate.root) else self.root
        selection=load(base/'state/object_selection.json');obj=selection['object_id']
        if judgment!=base/f'state/agent_interaction_judgments/{obj}.json':raise ValueError('interaction judgment selected object mismatch')
        posterior=str(base/f'state/scene/{obj}/refined_pose.json')
        if posterior not in evidence: raise ValueError('interaction judgment needs refined trajectory')
        self.require(posterior)
        self.raw_image_evidence(evidence)

    def _require(self,value,epoch):
        event=self.records.get(str(Path(value).resolve()))
        is_candidate_path=self.candidate is not None and Path(value).resolve().is_relative_to(self.candidate.root)
        expected_contract=sha(self.candidate.contract_path if is_candidate_path else self.contract_path)
        if event is not None and event.get('input_contract_sha256') != expected_contract:
            raise ValueError('prepared scene contract changed since producer')
        if event is not None and event.get('kind')=='verified_external_native_import':
            if self.candidate is None or event.get('candidate_id') != self.candidate.candidate_id or event.get('candidate_contract_sha256') != self.candidate_contract_sha256:
                raise ValueError('imported native event belongs to another candidate')
            for name,expected in event.get('input_sha256',{}).items():
                source_path=Path(name)
                if source_path.resolve().is_relative_to(self.root):
                    self.require(source_path)
                checked=self.checked_digest(source_path)
                if checked != expected:
                    raise ValueError('imported native source changed: '+name)
            if self.checked_digest(Path(value)) != event['output_sha256'][str(Path(value).resolve())]:
                raise ValueError('imported native output changed')
            epoch['paths'].add(str(Path(value).resolve()))
            return Path(value).resolve()
        if event is not None and not is_candidate_path:
            if self.checked_digest(Path(value)) != event['output_sha256'][str(Path(value).resolve())]:
                raise ValueError('source producer output changed')
            producer=id(event)
            if producer not in epoch['producers']:
                for name,expected in event.get('input_sha256',{}).items():
                    if self.checked_digest(self.require(name)) != expected:
                        raise ValueError('source producer input changed: '+name)
                epoch['producers'].add(producer)
            epoch['paths'].add(str(Path(value).resolve()))
            return Path(value).resolve()
        return super()._require(value,epoch)


    def success(self,event,inputs,outputs,output_roots):
        event['input_contract_sha256']=sha(self.candidate.contract_path if self.candidate is not None else self.contract_path)
        super().success(event,inputs,outputs,output_roots)

def run_command(key, argv, outputs, *, root, cwd, env, logdir=None, gpu=None, lineage=None, extra_inputs=(), capture_roots=(), timeout=OUTER_TIMEOUT_S):
    """One fresh invocation; preserve failures and never continue after failure."""
    root = Path(root).resolve()
    argv = list(map(str, argv))
    outputs = [in_run(path, root, must_exist=False) for path in outputs]
    if os.environ.get('RGB_SCENE_VERIFIED_RESUME') == '1' and any(path.exists() for path in outputs):
        events=[lineage.records.get(str(path)) for path in outputs] if lineage else []
        if not events or any(e is None or e.get('phase')!=key or e.get('argv')!=argv for e in events):
            raise ValueError('resume requires identical successful producer command')
        for event in events:
            for path in event['output_sha256']: lineage.require(path)
        return
    if lineage is not None and getattr(lineage,'candidate',None) is not None:
        candidate_events=lineage.candidate.events
        if candidate_events.exists():
            for line in candidate_events.read_text().splitlines():
                prior=json.loads(line)
                if prior.get('returncode') != 0 or 'error' in prior:
                    raise ValueError('failed/partial candidate is create-only; use a new candidate_id')
    for path in outputs:
        if path.exists():
            raise ValueError(f'{key} output already exists; not a fresh stage: {path}')

    output_roots=[]
    for flag in ('--output-dir', '--output-root', '--review-dir'):
        if flag in argv:
            directory = in_run(argv[argv.index(flag) + 1], root, must_exist=False)
            output_roots.append(directory)
            if directory.exists():
                raise ValueError(f'{key} output directory already exists: {directory}')
    for value in capture_roots:
        directory=in_run(value,root,must_exist=False)
        if directory.exists():raise ValueError(f'{key} output directory already exists: {directory}')
        output_roots.append(directory)
    logdir = Path(logdir) if logdir else (lineage.recovery['log_dir'] if lineage and lineage.recovery else root / 'logs/integration')
    if lineage is not None and getattr(lineage,'candidate',None) is not None:
        logdir = lineage.candidate.events.parent
        logdir.mkdir(parents=True, exist_ok=True)

    in_run(logdir, root, must_exist=False)
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f'{key}.log'
    if gpu is not None and env.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('heavy command environment changes authorized physical GPU')
    if env.get('HF_HUB_OFFLINE') != '1' or env.get('TRANSFORMERS_OFFLINE') != '1':
        # Lightweight CPU fixture commands need not import a model; prediction
        # environments always declare offline operation explicitly.
        if gpu is not None:
            raise ValueError('heavy prediction must use offline model assets')
    start = time.monotonic()
    event = {'phase': key, 'argv': argv, 'log': str(log), 'outputs': list(map(str, outputs)),
             'execution_environment':{k:v for k,v in env.items() if k.startswith('EGO_HAWOR_') or k in {'CUDA_VISIBLE_DEVICES','TORCH_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'}}}
    try:
        inputs=lineage.inputs(argv,output_roots,extra_inputs) if lineage else {}
        if gpu is not None:
            event['gpu_probe'] = probe(load(Path(cwd)/'runtime/runtime_config.json'), gpu)
        event.update(run_owned(key,argv,cwd=cwd,env=env,log=log,timeout=timeout))
        for path in outputs:
            in_run(path, root)
        if lineage:
            lineage.success(event,inputs,outputs,output_roots)
    except Exception as exc:
        if isinstance(exc,OwnedProcessFailure):event.update(exc.record)
        for directory in output_roots:
            nested=directory/'execution.json'
            if nested.is_file():
                event['nested_execution']=load(nested)
                failed=[s for s in event['nested_execution'].get('steps',[]) if 'error' in s]
                if failed:event['failed_inner_phase']=failed[-1]['phase']
        event['elapsed_s']=time.monotonic()-start
        event['error'] = str(exc)
        recovery = getattr(lineage, 'recovery', None)
        if recovery:
            blockers = root / 'state/runtime_recovery/P09_DEPTH_ORDER'
            failure_path = blockers / 'failure.json'
        else:
            blockers = root / 'state/runtime_blockers'
            failure_path = blockers / f'{key}.json'
        in_run(blockers, root, must_exist=False)
        blockers.mkdir(parents=True, exist_ok=True)
        # Recovery failure is create-only and cannot replace the migration journal,
        # retained old prefix, stop evidence, or original integration blocker/logs.
        with failure_path.open('x' if recovery else 'w') as handle:
            handle.write(json.dumps({
                **event, 'blocked_state_variable': key, 'next_required_repair': 'repair failed command/input before a new run',
            }, indent=2) + '\n')
        raise
    finally:
        event['elapsed_s'] = time.monotonic() - start
        with (logdir / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(event) + '\n')


def verify_resume(contract_path,root,bundle):
    value=load(contract_path)
    if value.get('schema')!='rgb_scene_verified_resume_v1' or value.get('run_root')!=str(root):
        raise ValueError('explicit scene resume contract required')
    if value['input_contract_sha256']!=sha(root/'input/runtime_input_contract.json') or value['bundle_sha256']!=sha(bundle/'runtime/bundle_manifest.json'):
        raise ValueError('resume raw/calibration/target/code/config identity changed')
    events=(root/'logs/integration/events.jsonl').read_bytes()
    if hashlib.sha256(events[:value['event_prefix_bytes']]).hexdigest()!=value['event_prefix_sha256']:
        raise ValueError('resume producer history changed')
    for line in events.splitlines():
        event=json.loads(line)
        if event.get('returncode')!=0 or 'error' in event:raise ValueError('failed producers require a fresh run, not resume')
    lineage=SceneLineage(root,bundle)
    for path,digest in value['evidence_sha256'].items():
        if sha(lineage.require(path))!=digest:raise ValueError('resume evidence changed')
    return value
