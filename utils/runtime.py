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
from utils.astra_transport import astra_urlopen, POLICY as ASTRA_TRANSPORT_POLICY

ASTRA_MODEL = 'gpt-6-astra'
ASTRA_ENDPOINT = 'https://litellm.dex-gem.ai/v1/responses'


def stage_environment(environment, extra_keys=()):
    """Fingerprint execution-affecting settings without persisting credentials.

    Arbitrary configured worker-overlay keys are included as well. Volatile shell
    bookkeeping (PWD, SHLVL, test control files) is intentionally not an input.
    """
    import hashlib
    keys = {'PATH', 'HOME', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'PYTHONHASHSEED', 'LD_LIBRARY_PATH',
        'CUDA_HOME', 'CUDA_PATH', 'LIDRA_SKIP_INIT', 'ATTN_BACKEND', 'SPCONV_ALGO',
        'PYOPENGL_PLATFORM', 'TORCH_EXTENSIONS_DIR',
        'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUBLAS_WORKSPACE_CONFIG',
        'HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE',
        'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TORCH_HOME', 'XDG_CACHE_HOME',
        'EGO_SAM2_SOURCE', 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD',
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'CONDA_PREFIX', 'VIRTUAL_ENV',
        'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy',
        'SSL_CERT_FILE','SSL_CERT_DIR', *extra_keys}
    return {key: hashlib.sha256(str(environment[key]).encode()).hexdigest()
        if key in environment else None for key in sorted(keys)}


# Full-field review encoding only; no crops, image quotas, or unbounded retries.
# Stop at 75% dimensions / quality65 to preserve labels and per-frame detail.
REVIEW_IMAGE_LEVELS = (
    ('original', 1., None),
    ('jpeg_q80', 1., 80),
    ('jpeg_q70', 1., 70),
    ('jpeg_q70_scale875', .875, 70),
    ('jpeg_q65_scale750', .75, 65),
)


def review_encoding_policy(max_request_bytes):
    if not isinstance(max_request_bytes, int) or max_request_bytes <= 0:
        raise ValueError('adaptive review encoding requires a positive request budget')
    return {'name': 'full_sheet_budget_v1', 'max_request_bytes': max_request_bytes,
            'levels': [{'name': name, 'scale': scale, 'jpeg_quality': quality}
                       for name, scale, quality in REVIEW_IMAGE_LEVELS],
            'resize': 'lanczos', 'jpeg_subsampling': 2, 'crop': False}


def astra_stage_inputs(prompt, schema, evidence, images, *, max_request_bytes=None, adaptive_review_images=False):
    """Plan exactly the ordered visual request inputs, without calling Astra."""
    inputs = {'model': ASTRA_MODEL, 'endpoint': ASTRA_ENDPOINT, 'disable_fallbacks': True,
        'prompt': prompt, 'schema': schema, 'evidence': evidence, 'transport_policy': dict(ASTRA_TRANSPORT_POLICY),
        'images': [{'path': str(Path(image).resolve()), 'sha256': sha(image),
                    'mime': mimetypes.guess_type(str(image))[0]} for image in images]}
    if adaptive_review_images:
        inputs['review_encoding'] = review_encoding_policy(max_request_bytes)
    return inputs


def encode_astra_request(prompt, schema, evidence, images, *, max_request_bytes=None, adaptive_review_images=False):
    """Select the first fitting full-sheet tier; return exact wire bytes and audit.

    Read originals once; every tier derives from those bytes, not the previous
    lossy tier. Facts, image order and labels remain identical at every tier.
    An exhausted budget returns the final oversized bytes for pre-send auditing;
    astra_request refuses to send them. Non-review judgments keep original bytes.
    """
    import hashlib
    import io
    policy = review_encoding_policy(max_request_bytes) if adaptive_review_images else None
    images = [Path(p).resolve() for p in images]
    if not images:
        raise ValueError('visual judgment requires actual image evidence')
    sources = []
    for image in images:
        mime = mimetypes.guess_type(image.name)[0]
        if mime not in {'image/png', 'image/jpeg', 'image/webp'}:
            raise ValueError(f'unsupported image evidence type: {image.suffix}')
        raw = image.read_bytes()
        dimensions = None
        if adaptive_review_images:
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.load()
                if getattr(decoded, 'n_frames', 1) != 1:
                    raise ValueError('review sheet must be a single still image')
                dimensions = list(decoded.size)
        sources.append((image, mime, raw, dimensions))
    image_hashes = {str(p): hashlib.sha256(raw).hexdigest() for p, _, raw, _ in sources}
    review_images = evidence.get('review_images', []) if adaptive_review_images else []
    if adaptive_review_images:
        if len(review_images) != len(images) or any(
            Path(row['path']).resolve() != image or row['sha256'] != image_hashes[str(image)]
            for row, image in zip(review_images, images)):
            raise ValueError('review sheet order/hash does not match evidence mapping')
        if [idx for row in review_images for idx in row['frame_ids']] != evidence.get('review_frame_ids'):
            raise ValueError('review sheet frame coverage/order mismatch')
    evidence_text = 'Return JSON. Current evidence JSON:\n' + json.dumps(evidence, allow_nan=False)
    attempts = []
    levels = REVIEW_IMAGE_LEVELS if adaptive_review_images else REVIEW_IMAGE_LEVELS[:1]
    for level, scale, quality in levels:
        content = [{'type': 'input_text', 'text': evidence_text}]
        ordered_images = []
        for index, (image, original_mime, original, dimensions) in enumerate(sources):
            raw, mime, effective_dimensions = original, original_mime, dimensions
            if quality is not None:
                with Image.open(io.BytesIO(original)) as decoded:
                    sheet = decoded.convert('RGB')
                effective_dimensions = [max(1, round(side * scale)) for side in dimensions]
                if effective_dimensions != dimensions:
                    sheet = sheet.resize(tuple(effective_dimensions), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                sheet.save(buffer, format='JPEG', quality=quality, subsampling=2, optimize=False)
                raw, mime = buffer.getvalue(), 'image/jpeg'
            ordered_images.append({'index': index, 'path': str(image), 'sha256': image_hashes[str(image)],
                'original_sha256': image_hashes[str(image)], 'original_bytes': len(original),
                'original_dimensions_wh': dimensions, 'effective_sha256': hashlib.sha256(raw).hexdigest(),
                'effective_bytes': len(raw), 'effective_dimensions_wh': effective_dimensions,
                'effective_mime': mime, 'jpeg_quality': quality, 'scale': scale, 'encoding_level': level,
                'frame_ids': review_images[index]['frame_ids'] if adaptive_review_images else None})
            content.append({'type': 'input_text', 'text': f'Image evidence: {image.name}'})
            content.append({'type': 'input_image', 'image_url': f'data:{mime};base64,' + base64.b64encode(raw).decode(), 'detail': 'high'})
        payload = {'model': ASTRA_MODEL, 'disable_fallbacks': True, 'store': False, 'stream': False,
            'instructions': prompt + '\nReturn only a JSON object satisfying this schema:\n' + json.dumps(schema),
            'input': [{'role': 'user', 'content': content}], 'text': {'format': {'type': 'json_object'}}}
        request_bytes = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
        attempts.append({'encoding_level': level, 'payload_bytes': len(request_bytes),
                         'payload_sha256': hashlib.sha256(request_bytes).hexdigest()})
        if max_request_bytes is None or len(request_bytes) <= max_request_bytes:
            break
    input_bytes = json.dumps({'prompt': prompt, 'schema': schema, 'evidence': evidence}, sort_keys=True, allow_nan=False).encode()
    return request_bytes, {'request_model': ASTRA_MODEL, 'disable_fallbacks': True, 'endpoint': ASTRA_ENDPOINT,
        'input_sha256': hashlib.sha256(input_bytes).hexdigest(), 'effective_payload_sha256': hashlib.sha256(request_bytes).hexdigest(),
        'effective_payload_bytes': len(request_bytes), 'image_sha256': image_hashes,
        'ordered_images': ordered_images, 'review_encoding_policy': policy, 'encoding_attempts': attempts,
        'budget_satisfied': max_request_bytes is None or len(request_bytes) <= max_request_bytes,
        'prompt': prompt, 'schema': schema, 'gateway_model_group': None, 'transport_policy': dict(ASTRA_TRANSPORT_POLICY)}


def astra_request(prompt, schema, evidence, images, *, audit_dir=None, max_request_bytes=None, adaptive_review_images=False):
    if adaptive_review_images and audit_dir is None:
        raise ValueError('adaptive review transport requires an exact payload audit directory')
    credential_path=Path(os.environ['EGO_ASTRA_CREDENTIAL_FILE'])
    info=credential_path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError('Astra credential must be owner-only and owned by the runtime user')
    key=credential_path.read_text().strip()
    if not key:
        raise ValueError('empty Astra credential')
    request_bytes, request_provenance = encode_astra_request(prompt, schema, evidence, images,
        max_request_bytes=max_request_bytes, adaptive_review_images=adaptive_review_images)
    if audit_dir is not None:
        audit_dir=Path(audit_dir);audit_dir.mkdir(parents=True,exist_ok=False)
        (audit_dir/'effective_payload.json').write_bytes(request_bytes)
        (audit_dir/'request_manifest.json').write_text(json.dumps(request_provenance,indent=2))
    if max_request_bytes is not None and len(request_bytes)>max_request_bytes:
        raise ValueError(f'Astra request exceeds {max_request_bytes} bytes; exact payload retained for explicit review')
    request=urllib.request.Request(ASTRA_ENDPOINT,data=request_bytes,
        headers={'Content-Type':'application/json','User-Agent':'ego-annotation/1.0','Authorization':'Bearer '+key},method='POST')
    try:
        with astra_urlopen(request,timeout=120) as response:
            raw=json.load(response)
            group=response.headers.get('x-litellm-model-group')
            request_id=response.headers.get('x-request-id')
            connection_attempts=getattr(response,'astra_connection_attempts',[])
    except Exception as exc:
        if audit_dir is not None:
            (audit_dir/'transport_failure.json').write_text(json.dumps({'error_type':type(exc).__name__, 'request_sha256':request_provenance['effective_payload_sha256'], 'retry':False, 'gateway_model_group':None}))
        raise
    if audit_dir is not None:
        (audit_dir/'response_identity.json').write_text(json.dumps({'response_model':raw.get('model'),'gateway_model_group':group,'request_id':request_id,'response_id':raw.get('id')}))
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
        'request_id':request_id,'connection_attempts':connection_attempts,**{k:v for k,v in request_provenance.items() if k!='gateway_model_group'}}


def judgment_cli(judge, *, audit_requests=False):
    parser=argparse.ArgumentParser(description=judge.__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--image',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('judgment output must be create-only')
    evidence=json.loads(args.evidence.read_text())
    value=judge(evidence,args.image,audit_dir=args.output.with_suffix('.request')) if audit_requests else judge(evidence,args.image)
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
