from __future__ import annotations
"""Astra-only structured image transport and isolated model process execution.

No model selector or fallback exists. The gateway response and routing group must
agree with the request before any judgment can be consumed.
"""
from pathlib import Path
import argparse
import base64
import json
import mimetypes
import os
import stat
import subprocess
import urllib.request
from utils.state import sha

ASTRA_MODEL = 'gpt-6-astra'
ASTRA_ENDPOINT = 'https://litellm.dex-gem.ai/v1/responses'


def astra_request(prompt, schema, evidence, images):
    credential_path=Path(os.environ['EGO_ASTRA_CREDENTIAL_FILE'])
    info=credential_path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError('Astra credential must be owner-only and owned by the runtime user')
    key=credential_path.read_text().strip()
    if not key:
        raise ValueError('empty Astra credential')
    images=[Path(p).resolve() for p in images]
    if not images:
        raise ValueError('visual judgment requires actual image evidence')
    content=[{'type':'input_text','text':'Return JSON. Current evidence JSON:\n'+json.dumps(evidence,allow_nan=False)}]
    image_hashes={}
    for image in images:
        mime=mimetypes.guess_type(image.name)[0]
        if mime not in {'image/png','image/jpeg','image/webp'}:
            raise ValueError(f'unsupported image evidence type: {image.suffix}')
        raw=image.read_bytes()
        import hashlib
        image_hashes[str(image)]=hashlib.sha256(raw).hexdigest()
        content.append({'type':'input_text','text':f'Image evidence: {image.name}'})
        content.append({'type':'input_image','image_url':f'data:{mime};base64,'+base64.b64encode(raw).decode(),'detail':'high'})
    payload={'model':ASTRA_MODEL,'disable_fallbacks':True,'store':False,'stream':False,
             'instructions':prompt+'\nReturn only a JSON object satisfying this schema:\n'+json.dumps(schema),
             'input':[{'role':'user','content':content}], 'text':{'format':{'type':'json_object'}}}
    request=urllib.request.Request(ASTRA_ENDPOINT,data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json','User-Agent':'ego-annotation/1.0','Authorization':'Bearer '+key},method='POST')
    with urllib.request.urlopen(request,timeout=120) as response:
        raw=json.load(response)
        group=response.headers.get('x-litellm-model-group')
        request_id=response.headers.get('x-request-id')
    if raw.get('model')!=ASTRA_MODEL or (group is not None and group!=ASTRA_MODEL):
        raise RuntimeError('Astra response model/routing group mismatch')
    if raw.get('status')!='completed' or raw.get('error'):
        raise RuntimeError('Astra judgment did not complete')
    text=''.join(part['text'] for message in raw.get('output',[])
                 if message.get('type')=='message' and message.get('role')=='assistant'
                 for part in message.get('content',[]) if part.get('type')=='output_text')
    result=json.loads(text)
    import jsonschema
    jsonschema.validate(result,schema)
    return result, {'request_model':ASTRA_MODEL,'response_model':raw['model'],
        'gateway_model_group':group,'disable_fallbacks':True,'response_id':raw.get('id'),
        'request_id':request_id,'image_sha256':image_hashes}


def judgment_cli(judge):
    parser=argparse.ArgumentParser(description=judge.__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--image',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('judgment output must be create-only')
    evidence=json.loads(args.evidence.read_text())
    value=judge(evidence,args.image)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
    return value


def inspect_gpu(index, *, expected_uuid=None, min_free_mib=12000):
    """Inspect exactly the requested physical GPU; never choose a replacement."""
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    records=[row.strip().split(',') for row in devices.splitlines() if row.strip()]
    selected=[row for row in records if int(row[0])==int(index)]
    if len(selected)!=1:raise RuntimeError('requested GPU index is unavailable')
    _,uuid,free=map(str.strip,selected[0]);free=int(free)
    if expected_uuid is not None and uuid!=expected_uuid:
        raise RuntimeError('requested GPU UUID changed; refusing another device')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True)
    occupants=[row for row in processes.splitlines() if row.split(',')[0].strip()==uuid]
    if occupants:raise RuntimeError('requested GPU has an active compute process')
    if free<int(min_free_mib):raise RuntimeError('requested GPU has insufficient free memory')
    return {'index':int(index),'uuid':uuid,'free_mib':free,'exclusive_compute_check':True}


def run_model(command, *, cwd, environment, gpu=None, expected_uuid=None,
              min_free_mib=12000, stdout=None):
    """Run in a provisioned environment after checking the pinned GPU at launch."""
    import contextlib, fcntl, tempfile
    with contextlib.ExitStack() as stack:
        resources=None
        if gpu is not None:
            resources=inspect_gpu(gpu,expected_uuid=expected_uuid,min_free_mib=min_free_mib)
            lock=stack.enter_context(open(Path(tempfile.gettempdir())/f"ego-annotation-{resources['uuid']}.lock",'a'))
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError('requested GPU is leased by another annotation') from exc
            environment={**environment,'CUDA_VISIBLE_DEVICES':resources['uuid']}
        subprocess.run([str(x) for x in command],cwd=cwd,env=environment,check=True,
                       stdout=stdout,stderr=subprocess.STDOUT if stdout is not None else None)
        return resources

def install_offline_dinov2_hub(local_repo: Path, local_checkpoint: Path) -> dict:
    """Route only the frozen DINOv2 hub request to byte-bound local assets."""
    from urllib.parse import urlparse
    import ast
    from utils.state import sha as sha256_file
    import torch
    local_repo = local_repo.expanduser().resolve()
    local_checkpoint = local_checkpoint.expanduser().resolve()
    hubconf = local_repo / 'hubconf.py'
    if not hubconf.is_file():
        raise RuntimeError(f'missing offline DINOv2 hubconf: {hubconf}')
    if not local_checkpoint.is_file():
        raise RuntimeError(f'missing offline DINOv2 checkpoint: {local_checkpoint}')
    actual_hubconf_sha256 = sha256_file(hubconf)
    expected_hubconf_sha256 = 'c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64'
    if actual_hubconf_sha256 != expected_hubconf_sha256:
        raise RuntimeError(f'offline DINOv2 hubconf hash mismatch: {hubconf}')
    imported_entries = {alias.asname or alias.name for node in ast.walk(ast.parse(hubconf.read_text(encoding='utf-8'), filename=str(hubconf))) if isinstance(node, ast.ImportFrom) for alias in node.names}
    if 'dinov2_vitl14_reg' not in imported_entries:
        raise RuntimeError(f'offline DINOv2 source lacks dinov2_vitl14_reg: {local_repo}')
    original_load = torch.hub.load
    expected_checkpoint_sha256 = '36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51'
    actual_checkpoint_sha256 = sha256_file(local_checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError(f'offline DINOv2 checkpoint hash mismatch: {local_checkpoint}')
    state = {'source': 'local', 'repo': str(local_repo), 'hubconf_sha256': actual_hubconf_sha256, 'expected_hubconf_sha256': expected_hubconf_sha256, 'required_entry': 'dinov2_vitl14_reg', 'available_entry_count': int(len(imported_entries)), 'intercepted_load_calls': 0, 'intercepted_checkpoint_load_calls': 0, 'checkpoint_path': str(local_checkpoint), 'checkpoint_sha256': actual_checkpoint_sha256, 'expected_checkpoint_sha256': expected_checkpoint_sha256, 'network_resolution_allowed': False}

    def offline_load(repo_or_dir, model, *load_args, **load_kwargs):
        if str(repo_or_dir) == 'facebookresearch/dinov2':
            state['intercepted_load_calls'] += 1
            load_kwargs.pop('source', None)
            load_kwargs.pop('skip_validation', None)
            return original_load(str(local_repo), model, *load_args, source='local', **load_kwargs)
        raise RuntimeError(f'undeclared torch.hub resolution is forbidden in model worker: {repo_or_dir}/{model}')

    def offline_state_dict_from_url(url, *load_args, **load_kwargs):
        filename = str(load_kwargs.get('file_name') or Path(urlparse(str(url)).path).name)
        if filename != 'dinov2_vitl14_reg4_pretrain.pth':
            raise RuntimeError(f'undeclared torch checkpoint URL is forbidden in model worker: {url}')
        checkpoint = local_checkpoint
        if not checkpoint.is_file():
            raise RuntimeError(f'offline DINOv2 checkpoint is missing; network download is forbidden: {checkpoint}')
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected_checkpoint_sha256:
            raise RuntimeError(f'offline DINOv2 checkpoint hash mismatch: {checkpoint}')
        state['intercepted_checkpoint_load_calls'] += 1
        state['checkpoint_path'] = str(checkpoint)
        state['checkpoint_sha256'] = actual_hash
        map_location = load_kwargs.get('map_location', None)
        return torch.load(checkpoint, map_location=map_location, weights_only=False)
    torch.hub.load_state_dict_from_url = offline_state_dict_from_url
    torch.hub.load = offline_load
    return state

from pathlib import Path
from urllib.parse import urlparse
from utils.state import sha as offline_sha

def bind_offline_weights(records, torch):
    """Replace URL resolution, not model code; only declared local bytes load."""

    def offline_load(url, *unused, **kwargs):
        name = kwargs.get('file_name') or Path(urlparse(url).path).name
        if name not in records:
            raise ValueError(f'undeclared Torch checkpoint request: {name}')
        record = records[name]
        path = Path(record['path'])
        if not path.is_file() or offline_sha(path) != record['sha256']:
            raise ValueError(f'offline LightGlue checkpoint missing/changed: {path}')
        return torch.load(path, map_location=kwargs.get('map_location'), weights_only=False)
    torch.hub.load_state_dict_from_url = offline_load
