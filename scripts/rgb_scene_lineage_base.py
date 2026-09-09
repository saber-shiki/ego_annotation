"""Current-run command lineage. Anti-misuse checks, not filesystem attestation.

Only run_command writes successful execution records. Agent visual judgments
are checked separately and never receive a fabricated model execution record.
"""
from pathlib import Path
from contextlib import contextmanager
import json
import uuid
from v19_runtime_contracts import in_run, load, sha, validate_references, CLIP_SUFFIXES
from v19_runtime_assets import verify_bundle
def event_code_is_valid(event,code,recovery):
    return event.get("code_sha256")==code


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
        self.recovery=None
        if self.contract["bundle_manifest_sha256"] != sha(self.bundle/"runtime/bundle_manifest.json"):
            raise ValueError("scene bundle changed")
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

    @contextmanager
    def verification_batch(self):
        """One synchronous verification epoch; nested requires share its DAG.

        Completed paths/producers and digest lookups expire on exit (also on
        failure). Separate public calls always walk ancestors again. Only the
        fingerprint-checked byte cache survives within this verifier instance.
        No producer command or concurrent writer belongs inside this scope.
        """
        owner=getattr(self,'_verification',None) is None
        if owner:
            self._verification={'paths':set(), 'producers':set(),
                                'active_producers':set(), 'digests':{}}
        try:
            yield self._verification
        finally:
            if owner:self._verification=None

    @staticmethod
    def fingerprint(path):
        stat=path.stat()
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

    def checked_digest(self, path):
        """Byte identity, with one bounded reconciliation of read-time metadata.

        A changed fingerprint permits one immediate full reread, never a retry
        loop. Both hashes must agree, the reread must be stable, and device/inode
        must not change. Callers still enforce the original expected hash.
        """
        path=Path(path);name=str(path)
        epoch=getattr(self,'_verification',None)
        if epoch is not None and name in epoch['digests']:
            return epoch['digests'][name]
        before=self.fingerprint(path)
        cached=self.verified.get(name)
        if cached is not None and cached[0]==before:
            digest=cached[1]
        else:
            digest=sha(path)
            after=self.fingerprint(path)
            if after!=before:
                second_before=self.fingerprint(path)
                second_digest=sha(path)
                second_after=self.fingerprint(path)
                if (digest!=second_digest or second_before!=second_after
                        or before[:2]!=after[:2] or before[:2]!=second_before[:2]):
                    raise ValueError(f'input changed during byte verification: {name}; '
                                     f'before={before}, after={after}, '
                                     f'reread_before={second_before}, reread_after={second_after}, '
                                     f'sha256={digest}, reread_sha256={second_digest}')
                after=second_after
            self.verified[name]=(after,digest)
        if epoch is not None:epoch['digests'][name]=digest
        return digest

    def require(self, value):
        with self.verification_batch() as epoch:
            return self._require(value,epoch)

    def _require(self, value, epoch):
        path=in_run(value,self.root);name=str(path)
        if name in self.visiting:raise ValueError('cyclic producer lineage')
        if name in epoch['paths']:return path
        self.visiting.add(name)
        try:
            digest=self.checked_digest(path)
            if name in self.raw:
                if digest!=self.raw[name]:raise ValueError(f'prepared input changed: {name}')
            elif path.parent == self.root/'input/hawor_sequence' and path.suffix=='.mp4':
                if digest!=self.identity['input_video_sha256']:raise ValueError('HaWoR raw copy changed')
            elif self.agent_phase(path):
                self.require_judgment(path)
            else:
                event=self.records.get(name)
                if event is None:raise ValueError(f'no successful current-run producer for {name}')
                if digest!=event['output_sha256'][name]:raise ValueError(f'producer output changed: {name}')
                producer=id(event)  # All outputs of one loaded event share this object.
                if producer in epoch['active_producers']:raise ValueError('cyclic producer lineage')
                if producer not in epoch['producers']:
                    if not event_code_is_valid(event,self.code,getattr(self,'recovery',None)):
                        raise ValueError('producer code identity changed')
                    epoch['active_producers'].add(producer)
                    try:
                        for source,expected in event['input_sha256'].items():
                            checked=self.require(source)
                            if self.checked_digest(checked)!=expected:raise ValueError(f'producer input changed: {source}')
                        epoch['producers'].add(producer)
                    finally:epoch['active_producers'].remove(producer)
            epoch['paths'].add(name)
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
            checked=self.require(name)
            if self.checked_digest(checked)!=digest:raise ValueError(f'visual judgment evidence changed: {name}')
        if source['phase']=='P17_judgment':
            self.require_interaction_evidence(path,source['evidence_sha256'])
            return
        required = {'P05':'/input/raw_frame_manifest/', 'P09_anchor':'/object_geometry/anchor_candidates/',
                    'P10':'/object_geometry/visible_geometry/'}[source['phase']]
        if not any(required in name for name in source['evidence_sha256']):
            raise ValueError(f'visual judgment lacks phase evidence: {path}')

    def require_interaction_evidence(self, judgment, evidence):
        """Validate declared P15/P01 evidence, not whether pixels were inspected."""
        selection=load(in_run(self.root/'state/object_selection.json',self.root))
        obj=selection['object_id']
        if judgment!=self.root/'state/agent_interaction_judgments'/f'{obj}.json':
            raise ValueError('P17 judgment does not match selected object')
        p15=str(self.root/f'measurements/pose_fits/{obj}/temporal/v19_rigid_object_pose_graph_report.json')
        event=self.records.get(p15,{})
        if p15 not in evidence or event.get('phase')!='P15_temporal_pose' or p15 not in event.get('outputs',[]):
            raise ValueError('P17 judgment requires selected-object current P15 report')
        manifest=self.require(self.root/'input/raw_frame_manifest/manifest.json')
        p01=self.records.get(str(manifest),{})
        if p01.get('phase')!='P01' or str(manifest) not in p01.get('outputs',[]):
            raise ValueError('P17 judgment requires current P01 raw-frame producer')
        for frame in load(manifest).get('frames',[]):
            name=frame.get('raw_frame_path') or frame.get('rgb')
            if name in evidence and self.records.get(name) is p01:
                return
        raise ValueError('P17 judgment requires at least one current P01 raw frame')

    def inputs(self, argv, output_roots, extra=()):
        with self.verification_batch():
            return self._inputs(argv,output_roots,extra)

    def _inputs(self, argv, output_roots, extra):
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
        return {name:self.checked_digest(name) for name in sorted(selected)}

    def success(self,event,inputs,outputs,output_roots):
        # Force fresh byte reads after the command, irrespective of the cache.
        for name,digest in inputs.items():
            if sha(name)!=digest:raise ValueError(f'producer mutated source input: {name}')
        files=set(outputs)
        for root in output_roots:
            files.update(p for p in root.rglob('*') if p.is_file() and p.suffix not in {'.log','.jsonl'})
        event.update(run_identity=self.identity,input_sha256=inputs,code_sha256=self.code,
                     output_sha256={str(in_run(p,self.root)):sha(p) for p in sorted(files)})
        if getattr(self,'recovery',None) is not None:
            event.update(execution_bundle_manifest_sha256=self.recovery['execution_bundle_sha256'],
                         recovery_contract_sha256=self.recovery['contract_sha256'])
        for path in event['output_sha256']:self.records[path]=event
