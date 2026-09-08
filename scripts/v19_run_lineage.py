"""Current-run command lineage. Anti-misuse checks, not filesystem attestation.

Only run_command writes successful execution records. Agent visual judgments
are checked separately and never receive a fabricated model execution record.
"""
from pathlib import Path
import json
import uuid
from v19_runtime_contracts import in_run, load, sha, validate_references, CLIP_SUFFIXES
from v19_runtime_assets import verify_bundle

AGENT_PATHS = {
    'object_plan_agent.json':'P05', 'anchor_decisions':'P09_anchor',
    'physical_branch_decisions':'P10', 'agent_interaction_judgments':'P17_judgment',
}


class RunLineage:
    def __init__(self, root, bundle, runtime_config=None, *, asset_role=None):
        self.root=Path(root).resolve();self.bundle=Path(bundle).resolve()
        self.contract_path=in_run(self.root/'input/runtime_input_contract.json',self.root)
        self.contract=load(self.contract_path)
        uuid.UUID(self.contract['run_id'])
        if self.contract['run_root'] != str(self.root):raise ValueError('prepared run identity/root mismatch')
        self.manifest=verify_bundle(self.bundle,asset_role=asset_role)
        if runtime_config is not None and sha(runtime_config)!=sha(self.bundle/'runtime/runtime_config.json'):
            raise ValueError('runtime config differs from the prepared bundle')
        if sha(self.bundle/'runtime/bundle_manifest.json') != self.contract['bundle_manifest_sha256']:
            raise ValueError('run uses a different bundle')
        self.identity={k:self.contract[k] for k in ('run_id','run_root','input_video_sha256','bundle_manifest_sha256')}
        self.raw={str(self.contract_path):sha(self.contract_path),self.contract['input_video']:self.contract['input_video_sha256']}
        if self.contract.get('sensor_calibration_metadata'):
            self.raw[self.contract['sensor_calibration_metadata']]=self.contract['sensor_calibration_sha256']
        self.code={name:digest for name,digest in self.manifest['files'].items() if name.endswith(('.py','.sh'))}
        self.verified={};self.visiting=set();self.records={}
        self.events=self.root/'logs/integration/events.jsonl'
        if self.events.exists():
            for line in self.events.read_text().splitlines():
                event=json.loads(line)
                if event.get('returncode')==0 and 'error' not in event and event.get('run_identity')==self.identity:
                    for path in event.get('output_sha256',{}):self.records[path]=event
        for path in self.raw:self.require(path)

    def require(self, value):
        path=in_run(value,self.root);name=str(path)
        stat=path.stat();fingerprint=(stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if self.verified.get(name)==fingerprint:return path
        if name in self.visiting:raise ValueError('cyclic producer lineage')
        self.visiting.add(name)
        try:
            if name in self.raw:
                if sha(path)!=self.raw[name]:raise ValueError(f'prepared input changed: {name}')
            elif path.parent == self.root/'input/hawor_sequence' and path.suffix=='.mp4':
                if sha(path)!=self.identity['input_video_sha256']:raise ValueError('HaWoR raw copy changed')
            elif self.agent_phase(path):
                self.require_judgment(path)
            else:
                event=self.records.get(name)
                if event is None:raise ValueError(f'no successful current-run producer for {name}')
                if event['code_sha256'] != self.code:raise ValueError('producer code identity changed')
                if sha(path)!=event['output_sha256'][name]:raise ValueError(f'producer output changed: {name}')
                for source,digest in event['input_sha256'].items():
                    self.require(source)
                    if sha(source)!=digest:raise ValueError(f'producer input changed: {source}')
            self.verified[name]=fingerprint
            return path
        finally:self.visiting.remove(name)

    def agent_phase(self,path):
        if path.name=='object_plan_agent.json':return 'P05'
        for part,phase in AGENT_PATHS.items():
            if part in path.parts:return phase
        return None

    def require_judgment(self,path):
        payload=load(path);validate_references(payload,self.root)
        source=payload.get('visual_judgment_provenance',{})
        if (source.get('kind')!='current_run_visual_judgment' or source.get('phase')!=self.agent_phase(path)
                or source.get('run_identity')!=self.identity or not source.get('evidence_sha256')):
            raise ValueError(f'missing current-run visual judgment provenance: {path}')
        for name,digest in source['evidence_sha256'].items():
            self.require(name)
            if sha(name)!=digest:raise ValueError(f'visual judgment evidence changed: {name}')
        required = {'P05':'/input/raw_frame_manifest/', 'P09_anchor':'/object_geometry/anchor_candidates/',
                    'P10':'/object_geometry/visible_geometry/', 'P17_judgment':'/pose_fits/'}[source['phase']]
        if not any(required in name for name in source['evidence_sha256']):
            raise ValueError(f'visual judgment lacks phase evidence: {path}')

    def inputs(self, argv, output_roots, extra=()):
        selected=set()
        def include(path):
            path=Path(path)
            if any(path.resolve().is_relative_to(p) for p in output_roots):return
            if path.is_dir():
                for child in path.rglob('*'):
                    if child.is_file() and child.suffix not in {'.log','.jsonl'}:include(child)
            elif path.is_file():
                selected.add(str(self.require(path)))
        for value in [*argv,*map(str,extra)]:
            if isinstance(value,str) and Path(value).is_absolute():
                path=Path(value)
                if path.resolve().is_relative_to(self.root) and path != self.root:include(path)
        # Input reports refer to masks/meshes not always repeated on argv.
        # Producer-directory manifests protect all such files; avoid parsing
        # final package relative assets as source paths.
        for name in list(selected):
            if Path(name).suffix=='.json':
                def walk(value):
                    if isinstance(value,dict):
                        for child in value.values():walk(child)
                    elif isinstance(value,list):
                        for child in value:walk(child)
                    elif isinstance(value,str) and value.startswith('/') and '\n' not in value:
                        p=Path(value)
                        if p.suffix in CLIP_SUFFIXES and p.is_file() and p.resolve().is_relative_to(self.root) and str(p) not in selected:include(p)
                walk(load(name))
        return {name:sha(name) for name in sorted(selected)}

    def success(self,event,inputs,outputs,output_roots):
        for name,digest in inputs.items():
            if sha(name)!=digest:raise ValueError(f'producer mutated source input: {name}')
        files=set(outputs)
        for root in output_roots:
            files.update(p for p in root.rglob('*') if p.is_file() and p.suffix not in {'.log','.jsonl'})
        event.update(run_identity=self.identity,input_sha256=inputs,code_sha256=self.code,
                     output_sha256={str(in_run(p,self.root)):sha(p) for p in sorted(files)})
        for path in event['output_sha256']:self.records[path]=event
