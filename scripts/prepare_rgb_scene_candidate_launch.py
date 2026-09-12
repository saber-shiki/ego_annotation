#!/usr/bin/env python3
"""Create-only, producer-free launch plan from current RGB contracts."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shlex, subprocess, uuid
from pathlib import Path

STAGES=(('frontend-P09_DEPTH_ORDER','frontend','P09_DEPTH_ORDER'),('frontend-P09-propose','frontend','P09-propose'),('frontend-P09','frontend','P09'),('frontend-P11','frontend','P11'),('geometry','geometry',None),('initial-pose','initial-pose',None),('refined-pose','refined-pose',None),('appearance','appearance',None),('native-shape','native-shape',None),('shape','shape',None),('visual-fit','visual-fit',None),('export','export',None))
_ID=re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')

def _path(value, *, directory=False):
    p=Path(value).expanduser().resolve(strict=True)
    if p.is_symlink() or (directory and not p.is_dir()): raise ValueError(f'invalid path: {p}')
    return p

def _json(path):
    try: value=json.loads(path.read_text())
    except (OSError,ValueError) as exc: raise ValueError(f'invalid JSON: {path}') from exc
    if not isinstance(value,dict): raise ValueError(f'JSON object required: {path}')
    return value

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspection_code(report, keys, *, ranked=False, evidence=()):
    """Standalone read-only recipe; no dependency on this developer helper."""
    return f'''import hashlib, json
from pathlib import Path
report = Path({str(report)!r})
d = json.loads(report.read_text())
def scalar(value):
    return value[:500] if isinstance(value, str) else value if value is None or isinstance(value, (bool, int, float)) else '<structured; inspect source>'
summary = {{k: scalar(d.get(k)) for k in {tuple(keys)!r}}}
if {ranked!r}:
    summary['top_ranked'] = [{{k: scalar(row.get(k)) for k in ('frame_idx', 'proposal_score', 'mask_area_px', 'component_count')}} for row in d.get('ranked_candidates', [])[:3]]
summary['evidence_sha256'] = {{}}
for name in {tuple(map(str, (report, *evidence)))!r}:
    digest = hashlib.sha256()
    with Path(name).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    summary['evidence_sha256'][name] = digest.hexdigest()
text = json.dumps(summary, indent=2)
if len(text.encode()) > 12000:
    raise ValueError('inspection summary exceeds 12000 bytes')
print(text)
'''


def materialize(bundle,candidate,output,*,gpu,device='cuda:0',image_width=None,image_height=None,python=None):
    bundle=_path(bundle,directory=True); candidate=_path(candidate); output=Path(output).expanduser().resolve()
    if output.exists(): raise ValueError(f'create-only output already exists: {output}')
    if output==bundle or output.is_relative_to(bundle): raise ValueError('output must be outside bundle')
    contract=_json(candidate)
    required=('candidate_id','object_id','selected_frame_idx','run_root')
    if any(k not in contract for k in required): raise ValueError('candidate contract missing required key')
    if not all(isinstance(contract[k],str) and _ID.fullmatch(contract[k]) for k in ('candidate_id','object_id')): raise ValueError('unsafe candidate/object ID')
    if type(contract['selected_frame_idx']) is not int or contract['selected_frame_idx']<0: raise ValueError('selected_frame_idx must be non-negative integer')
    run_root=_path(contract['run_root'],directory=True); runtime=bundle/'runtime/runtime_config.json'; _path(runtime); cfg=_json(runtime)
    if candidate != run_root/'candidates'/contract['candidate_id']/'candidate_contract.json': raise ValueError('candidate contract path/identity mismatch')
    if contract.get('schema') != 'rgb_scene_candidate_v1' or not isinstance(contract.get('run_id'),str): raise ValueError('candidate schema/identity missing')
    try: uuid.UUID(contract['run_id'])
    except ValueError as exc: raise ValueError('invalid candidate run_id') from exc
    if contract.get('native_mode') not in {'new','reuse'}: raise ValueError('invalid native_mode')
    if not isinstance(contract.get('source_contract_sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',contract['source_contract_sha256']): raise ValueError('source contract identity missing')
    if not isinstance(contract.get('source_identity'),dict): raise ValueError('source identity missing')
    identity=contract['source_identity']
    if identity.get('run_id') != contract['run_id'] or identity.get('run_root') != str(run_root): raise ValueError('source identity run mismatch')
    source_contract=run_root/'input/runtime_input_contract.json'
    context=Path(contract.get('source_context_path','')).expanduser()
    if not source_contract.is_file() or sha256(source_contract)!=contract['source_contract_sha256']: raise ValueError('source input contract hash mismatch')
    if not context.is_file() or sha256(context)!=contract.get('source_context_sha256'): raise ValueError('source context hash mismatch')
    context_value=_json(context)
    if context_value.get('source_contract_sha256') != contract['source_contract_sha256'] or context_value.get('source_contract') != str(source_contract): raise ValueError('source context binding mismatch')
    event_bindings=context_value.get('source_event_streams') or [{'path':context_value.get('source_events'), 'sha256':context_value.get('source_events_sha256')}]
    if not isinstance(event_bindings,list) or not event_bindings: raise ValueError('source event bindings missing')
    for binding in event_bindings:
        if not isinstance(binding,dict) or not isinstance(binding.get('path'),str) or not re.fullmatch(r'[0-9a-f]{64}',str(binding.get('sha256',''))): raise ValueError('invalid source event binding')
        event_path=Path(binding['path']).expanduser()
        if not event_path.is_absolute() or not event_path.is_file() or sha256(event_path)!=binding['sha256']: raise ValueError('source event snapshot hash mismatch')
    source_value=_json(source_contract)
    for key in ('run_id', 'run_root', 'input_video_sha256', 'bundle_manifest_sha256'):
        if not isinstance(identity.get(key),str) or identity[key] != source_value.get(key): raise ValueError(f'source identity mismatch: {key}')
    if type(gpu) is not int or gpu < 0 or type(cfg.get('gpu')) is not int or cfg['gpu']!=gpu: raise ValueError('gpu differs from current runtime config')
    if not isinstance(device,str) or device != 'cuda:0': raise ValueError('device must be cuda:0; current CLI does not support device overrides')
    if image_width is not None and (type(image_width) is not int or image_width<=0): raise ValueError('image_width must be positive integer')
    if image_height is not None and (type(image_height) is not int or image_height<=0): raise ValueError('image_height must be positive integer')
    if python is None: raise ValueError('pinned Python is required')
    lexical=Path(python).expanduser()
    if not lexical.is_absolute() or not os.access(lexical,os.X_OK): raise ValueError('pinned Python must be an executable absolute path')
    entry=_path(bundle/'scripts/run_rgb_scene_stage.py')
    bodies={}; scripts={}; hashes={}
    for name,stage,phase in STAGES:
        args=[str(lexical),str(entry),'--stage',stage,'--run-root',str(run_root),'--runtime-config',str(runtime),'--gpu',str(gpu),'--candidate-context',str(candidate)]
        if phase: args += ['--phase',phase]
        body='#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(bundle))+'\nexec '+' '.join(shlex.quote(x) for x in args)+'\n'
        bodies[name]=body
    proposal= candidate.parent/'measurements/object_geometry/anchor_candidates'/contract['object_id']/'anchor_candidate_proposals.json'
    p09= candidate.parent/'measurements/object_geometry/visible_geometry'/contract['object_id']/'v19_visible_geometry_adapter_report.json'
    recipes={
      'startup': inspection_code(candidate, ('candidate_id', 'object_id', 'selected_frame_idx', 'native_mode', 'selection_mode')),
      'proposal': inspection_code(proposal, ('status', 'object_id', 'candidate_count', 'review_candidate_count'), ranked=True),
      'P09': inspection_code(p09, ('status', 'object_id', 'output_frame_count', 'visible_metric_frame_count', 'anchor_frame_idx'), evidence=(p09.parent/'annotations_v19_visible_geometry.json',))
    }
    for name,code in recipes.items():
        compile(code, f'inspect-{name}', 'exec')
        bodies[f'inspect-{name}']='#!/bin/bash\nset -euo pipefail\nexec '+shlex.join([str(lexical), '-c', code])+'\n'
    for name,body in bodies.items():
        check=subprocess.run(['/bin/bash','-n'],input=body,text=True,capture_output=True,timeout=30)
        if check.returncode: raise ValueError(f'bash syntax failed: {name}')
        path=output/f'{name}.sh'
        if not name.startswith('inspect-'): scripts[name]=str(path)
        hashes[str(path)]=hashlib.sha256(body.encode()).hexdigest()
    metadata={'kind':'rgb_scene_candidate_launch_plan','bundle':str(bundle),'candidate_context':str(candidate),'run_root':str(run_root),'candidate_id':contract['candidate_id'],'object_id':contract['object_id'],'selected_frame_idx':contract['selected_frame_idx'],'python':str(lexical),'gpu':gpu,'device':device,'image_width':image_width,'image_height':image_height,'dimensions_metadata_only':True,'stages':scripts,'stage_count':len(STAGES),'inspection_recipes':{k:shlex.join(['/bin/bash',str(output/f'inspect-{k}.sh')]) for k in recipes},'judgment_paths':{k:str(candidate.parent/p) for k,p in {'P09_anchor':'state/anchor_decisions/'+contract['object_id']+'.json','P10_branch':'state/physical_branch_decisions/'+contract['object_id']+'.json','P10_selection':'state/object_selection.json','appearance_anchor':'state/appearance_decisions/'+contract['object_id']+'.json','P17':'state/agent_interaction_judgments/'+contract['object_id']+'.json'}.items()},'producer_execution':'not performed'}
    metadata['script_sha256']=hashes
    metadata['commands']={k:shlex.join(['/bin/bash',v]) for k,v in scripts.items()}
    metadata['startup_read_paths']=[str(bundle/'runtime'/p) for p in ('rgb_scene_system_prompt.md','rgb_scene_runtime_spec.md','runtime_config.json')]+[str(source_contract),str(candidate)]
    metadata['final_outputs']=[str(candidate.parent/'final'/p) for p in ('annotation.json','world.rrd')]
    metadata['manifest_sha256']=sha256(bundle/'runtime/bundle_manifest.json')
    # Serialize all contents before creating anything, including the command maps.
    contents={f'{name}.sh':body for name,body in bodies.items()}
    contents['inspection_recipes.json']=json.dumps({'commands':metadata['inspection_recipes'],'maximum_output_bytes':12001},indent=2)+'\n'
    contents['launch_manifest.json']=json.dumps(metadata,indent=2)+'\n'
    output.mkdir(parents=True)
    for name,body in contents.items():
        with (output/name).open('x') as handle: handle.write(body)

    return metadata

def main():
    p=argparse.ArgumentParser();
    p.add_argument('--bundle',required=True,type=Path); p.add_argument('--candidate-context','--candidate_context',dest='candidate_context',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gpu',required=True,type=int); p.add_argument('--device',default='cuda:0'); p.add_argument('--image-width',type=int); p.add_argument('--image-height',type=int); p.add_argument('--python',required=True)
    a=p.parse_args(); print(json.dumps(materialize(a.bundle,a.candidate_context,a.output,gpu=a.gpu,device=a.device,image_width=a.image_width,image_height=a.image_height,python=a.python),indent=2))
if __name__=='__main__': main()
