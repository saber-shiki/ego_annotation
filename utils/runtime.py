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
import jsonschema
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


def run_model(command, *, cwd, environment):
    """Run an explicit model command in its provisioned environment; never retry elsewhere."""
    return subprocess.run([str(x) for x in command],cwd=cwd,env=environment,check=True)
