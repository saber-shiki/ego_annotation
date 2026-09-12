#!/usr/bin/env python3
"""Create-only, producer-free launch plan from current RGB contracts."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shlex
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

def materialize(bundle,candidate,output,*,gpu,device='cuda:0',image_width=None,image_height=None,python=None):
    bundle=_path(bundle,directory=True); candidate=_path(candidate); output=Path(output).expanduser().resolve()
    if output.exists(): raise ValueError(f'create-only output already exists: {output}')
    if output==bundle or output.is_relative_to(bundle): raise ValueError('output must be outside bundle')
    contract=_json(candidate)
    required=('candidate_id','object_id','selected_frame_idx','run_root')
    if any(k not in contract for k in required): raise ValueError('candidate contract missing required key')
    if not all(isinstance(contract[k],str) and _ID.fullmatch(contract[k]) for k in ('candidate_id','object_id')): raise ValueError('unsafe candidate/object ID')
    if type(contract['selected_frame_idx']) is not int or contract['selected_frame_idx']<0: raise ValueError('selected_frame_idx must be non-negative integer')
    run_root=_path(contract['run_root'],directory=True); runtime=bundle/'runtime/runtime_config.json'; _path(runtime)
    cfg=_json(runtime)
    if type(gpu) is not int or gpu<0: raise ValueError('gpu must be non-negative integer')
    if type(cfg.get('gpu')) is int and cfg['gpu']!=gpu: raise ValueError('gpu differs from current runtime config')
    if not isinstance(device,str) or not device: raise ValueError('device must be non-empty string')
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
    recipes={
      'startup': shlex.join([str(lexical),'-c',f'import json; print(json.load(open({str(candidate)!r}))["candidate_id"])']),
      'proposal': shlex.join([str(lexical),'-c',f'import hashlib,pathlib; print(hashlib.sha256(pathlib.Path({str(candidate)!r}).read_bytes()).hexdigest())']),
      'P09': shlex.join([str(lexical),'-c',f'import json; d=json.load(open({str(candidate)!r})); print(json.dumps({{"object_id":d["object_id"],"selected_frame_idx":d["selected_frame_idx"],"native_mode":d.get("native_mode")}}))'])
    }
    for name,body in bodies.items():
        check=__import__('subprocess').run(['/bin/bash','-n'],input=body,text=True,capture_output=True)
        if check.returncode: raise ValueError(f'bash syntax failed: {name}')
        path=output/f'{name}.sh'; scripts[name]=str(path); hashes[str(path)]=hashlib.sha256(body.encode()).hexdigest()
    metadata={'kind':'rgb_scene_candidate_launch_plan','bundle':str(bundle),'candidate_context':str(candidate),'run_root':str(run_root),'candidate_id':contract['candidate_id'],'object_id':contract['object_id'],'selected_frame_idx':contract['selected_frame_idx'],'python':str(lexical),'gpu':gpu,'device':device,'image_width':image_width,'image_height':image_height,'stages':scripts,'stage_count':len(STAGES),'inspection_recipes':{k:f'/bin/bash {output}/inspect-{k}.sh' for k in recipes},'judgment_paths':{k:str(candidate.parent/p) for k,p in {'P09_anchor':'state/anchor_decisions/'+contract['object_id']+'.json','P10_branch':'state/physical_branch_decisions/'+contract['object_id']+'.json','P10_selection':'state/object_selection.json','appearance_anchor':'state/appearance_decisions/'+contract['object_id']+'.json','P17':'state/agent_interaction_judgments/'+contract['object_id']+'.json'}.items()},'producer_execution':'not performed'}
    metadata['script_sha256']=hashes
    # All validation above and all bodies are complete before any output exists.
    output.mkdir(parents=True)
    for name,body in bodies.items(): (output/f'{name}.sh').write_text(body)
    for name,command in recipes.items(): (output/f'inspect-{name}.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec '+command+'\n')
    (output/'inspection_recipes.json').write_text(json.dumps({'commands':{k:f'/bin/bash {output}/inspect-{k}.sh' for k in recipes},'bounded':True,'device':device,'image_width':image_width,'image_height':image_height},indent=2)+'\n')
    (output/'launch_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata

def main():
    p=argparse.ArgumentParser();
    for n in ('bundle','candidate_context','output'): p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--gpu',required=True,type=int); p.add_argument('--device',default='cuda:0'); p.add_argument('--image-width',type=int); p.add_argument('--image-height',type=int); p.add_argument('--python',required=True)
    a=p.parse_args(); print(json.dumps(materialize(a.bundle,a.candidate_context,a.output,gpu=a.gpu,device=a.device,image_width=a.image_width,image_height=a.image_height,python=a.python),indent=2))
if __name__=='__main__': main()
