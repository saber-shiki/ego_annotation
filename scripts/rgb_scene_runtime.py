"""Standalone guarded producer boundary and same-code verified resume.

The core live_gpu_probe and its callers are unchanged. Every heavy standalone
subprocess crosses this boundary (nested native reconstruction also rechecks).
"""
from pathlib import Path
import json, subprocess, time, os, hashlib
from v19_runtime_contracts import in_run, load, sha
from rgb_scene_lineage_base import RunLineage


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
    def __init__(self, root, bundle, runtime_config=None, *, asset_role=None):
        super().__init__(root, bundle, runtime_config, asset_role=asset_role)
        if asset_role in {'demo','export'}:
            from v19_runtime_assets import verify_external_assets
            identity=self.manifest['external_assets']
            names=identity.get('scene_rrd_files')
            if not names:
                raise ValueError('standalone visual/export dependency inventory missing')
            verify_external_assets({'files':{name:identity['files'][name] for name in names},'trees':{}})

    def agent_phase(self,path):
        if 'appearance_decisions' in path.parts: return 'appearance_anchor'
        return super().agent_phase(path)

    def require_judgment(self,path):
        if self.agent_phase(path)!='appearance_anchor': return super().require_judgment(path)
        payload=load(path);source=payload.get('visual_judgment_provenance',{})
        if source.get('kind')!='current_run_visual_judgment' or source.get('phase')!='appearance_anchor' or source.get('run_identity')!=self.identity or not source.get('evidence_sha256'):
            raise ValueError('appearance anchor needs current model-assisted visual judgment')
        for name,digest in source['evidence_sha256'].items():
            if sha(self.require(name))!=digest: raise ValueError('appearance evidence changed')
        if not any(name.endswith('appearance_mask_manifest.json') for name in source['evidence_sha256']):
            raise ValueError('appearance decision lacks current appearance manifest')

    def require_interaction_evidence(self, judgment, evidence):
        selection=load(self.root/'state/object_selection.json');obj=selection['object_id']
        posterior=str(self.root/f'state/scene/{obj}/refined_pose.json')
        if posterior not in evidence: raise ValueError('interaction judgment needs refined trajectory')
        self.require(posterior)
        manifest=self.require(self.root/'input/raw_frame_manifest/manifest.json')
        if not any((f.get('raw_frame_path') or f.get('rgb')) in evidence for f in load(manifest)['frames']):
            raise ValueError('interaction judgment needs raw image evidence')

    def _require(self,value,epoch):
        event=self.records.get(str(Path(value).resolve()))
        if event is not None and event.get('input_contract_sha256')!=sha(self.contract_path):
            raise ValueError('prepared scene contract changed since producer')
        return super()._require(value,epoch)

    def success(self,event,inputs,outputs,output_roots):
        event['input_contract_sha256']=sha(self.contract_path)
        super().success(event,inputs,outputs,output_roots)

def run_command(key, argv, outputs, *, root, cwd, env, logdir=None, gpu=None, lineage=None, extra_inputs=(), capture_roots=()):
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
    logdir = Path(logdir) if logdir else root / 'logs/integration'
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
        with log.open('x') as handle:
            if gpu is not None:
                event['gpu_probe'] = probe(load(Path(cwd)/'runtime/runtime_config.json'), gpu)
                handle.write(json.dumps(event['gpu_probe']) + '\n')
                handle.flush()
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=1200)
        event['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'{key} failed: {log}')
        for path in outputs:
            in_run(path, root)
        if lineage:
            lineage.success(event,inputs,outputs,output_roots)
    except Exception as exc:
        event['error'] = str(exc)
        recovery = getattr(lineage, 'recovery', None)
        resume_phase = recovery['contract']['resume_phase'] if recovery else None
        recovery_failure = resume_phase in {'P14','P14_RGB'} or (key == 'P09' and resume_phase == 'P09')
        blockers = root / (f'state/runtime_recovery/{resume_phase}' if recovery_failure else 'state/runtime_blockers')
        in_run(blockers, root, must_exist=False)
        blockers.mkdir(parents=True, exist_ok=True)
        failure_path = blockers / ('failure.json' if recovery_failure else f'{key}.json')
        # A stopped run's original P09 blocker is a protected recovery input.
        # Retain any recovery-attempt failure separately and never replace it.
        with failure_path.open('x') as handle:
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
