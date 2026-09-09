#!/usr/bin/env python3
"""Prepare an isolated, pinned raw-RGB scene bundle/run; never execute inference."""
from __future__ import annotations
import argparse,ast,json,os,re,shutil,socket,subprocess,sys,uuid
from pathlib import Path
from prepare_v19_sam3d_runtime import (validate_config,copy_package,NATIVE_PROVIDER,
    EXPERIMENT_HELPERS,SCRIPT_RE,ASSET_KEYS,FORBIDDEN_INSTRUCTIONS)
from v19_runtime_contracts import load,sha
from v19_runtime_assets import external_asset_identity,verify_bundle
from v19_video_timing import read_cfr_video_timing
from v20_prediction_contracts import find_positive_gt_flags

SPEC='runtime/rgb_scene_runtime_spec.md'
PROMPT='runtime/rgb_scene_system_prompt.md'
PROFILE='.pi/agents/ego-annotation.rgb-scene-runtime.md'
UNUSED_DYNAMIC={}

def closure(source):
    source = Path(source).resolve()
    available = {p.name: p.relative_to(source) for p in (source / 'scripts').glob('*') if p.suffix in {'.py', '.sh'}}
    available.update({Path(p).name: Path(p) for p in EXPERIMENT_HELPERS})
    modules = {Path(k).stem: v for k, v in available.items()}
    queue = [Path(p) for p in SCRIPT_RE.findall((source / SPEC).read_text())]
    selected = set()
    while queue:
        relative = queue.pop()
        if relative in selected:
            continue
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'missing/nonlocal runtime script: {relative}')
        selected.add(relative)
        text = path.read_text()
        queue.extend(Path(p) for p in SCRIPT_RE.findall(text))
        if path.suffix != '.py':
            queue.extend(relative for name, relative in available.items() if name in text)
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue.extend(modules[a.name.split('.')[0]] for a in node.names if a.name.split('.')[0] in modules)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split('.')[0] in modules:
                    queue.append(modules[node.module.split('.')[0]])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Covers script('name.py'), Path/.../'name.py', with_name and
                # literal filenames used by importlib loaders (including P13).
                if node.value in available and node.value not in UNUSED_DYNAMIC.get(str(relative), set()):
                    queue.append(available[node.value])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'script':
                if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value not in available:
                    raise ValueError(f'unresolved dynamic script dependency in {relative}')
    return sorted(selected)


def build(source,bundle,assets,report_dir):
    source,bundle,report_dir=map(lambda p:Path(p).resolve(),(source,bundle,report_dir))
    if bundle.exists() or bundle.is_relative_to(source):raise ValueError('new isolated bundle required')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('reviewed clean source revision required for bundling')
    cfg=load(assets);policy=cfg.pop('scene_gpu_policy',{'mode':'exclusive'})
    weights=validate_config(cfg)
    if policy.get('mode') not in {'exclusive','shared_idle'}:raise ValueError('unknown scene GPU policy')
    if policy['mode']=='shared_idle' and (policy.get('authorized') is not True or not str(policy.get('uuid','')).startswith('GPU-') or type(policy.get('minimum_free_mib')) is not int or policy['minimum_free_mib']<cfg['sam3d_min_free_mib']):raise ValueError('explicit shared idle UUID/free-memory authorization required')
    cfg['scene_gpu_policy']=policy
    for instruction in (SPEC,PROMPT,PROFILE):
        text=(source/instruction).read_text()
        for token in FORBIDDEN_INSTRUCTIONS:
            if token.lower() in text.lower():raise ValueError('forbidden runtime instruction: '+token)
    profile=(source/PROFILE).read_text()
    for required in ('name: rgb-scene-runtime','systemPromptMode: replace','inheritProjectContext: false','inheritGlobalContext: false','inheritSkills: false','defaultContext: fresh',f'subagentOnlyExtensions: {NATIVE_PROVIDER}'):
        if required not in profile:raise ValueError('native profile isolation mismatch')
    selected=closure(source)
    forbidden={'run_v19_sam3d_integration_stage.py','run_v19_additive_demo_branch.py','render_v20_visual_contact_demo.py','render_v19_annotation_layers_video.py','assemble_v19_annotation_layers.py'}
    if forbidden & {p.name for p in selected}:raise ValueError('scene closure includes unrelated pipeline dispatchers')
    native=Path(NATIVE_PROVIDER)
    native_assets=[native,native.with_name('shellgate-broker.py'),native.parent.parent/'package.json',native.parent.parent/'package-lock.json']
    if any(not p.is_file() for p in native_assets):raise ValueError('native provider dependency missing')
    bundle.mkdir(parents=True)
    for relative in [*selected,Path(SPEC),Path(PROMPT),Path(PROFILE)]:
        destination=bundle/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,destination)
    for key,package,destination in [('sam2_source','sam2','third_party/sam2'),('wilor_root','wilor','third_party/WiLoR'),('lightglue_root','lightglue','third_party/LightGlue')]:
        original=Path(cfg[key]);copy_package(original/package,bundle/destination/package)
        for name in ('LICENSE','license.txt','LICENSE_cctorch'):
            if (original/name).is_file():shutil.copyfile(original/name,bundle/destination/name)
        if key=='wilor_root':
            mano=bundle/destination/'mano_data';mano.mkdir(parents=True)
            shutil.copyfile(original/'mano_data/mano_mean_params.npz',mano/'mano_mean_params.npz')
            for side in ('LEFT','RIGHT'):shutil.copyfile(Path(cfg['mano_models'])/f'MANO_{side}.pkl',mano/f'MANO_{side}.pkl')
            cfg['mano_models']=str(mano)
        cfg[key]=str(bundle/destination)
    cfg['lightglue_weights']=weights
    # Inspect installed package identities through the pinned scene interpreter.
    result=subprocess.run([cfg['demo_python'],'-c',"import rerun,rerun_bindings,json; print(json.dumps({'version':rerun.__version__,'source':rerun.__file__,'bindings':rerun_bindings.__file__}))"],capture_output=True,text=True,check=True,timeout=1200,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    rr_identity=json.loads(result.stdout);cfg['scene_rerun_version']=rr_identity['version']
    (bundle/'runtime/runtime_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for p in bundle.rglob('*.py'):compile(p.read_text(),str(p),'exec')
    if closure(bundle)!=selected:raise ValueError('copied closure differs')
    identity=external_asset_identity(cfg)
    from v19_runtime_assets import tree_files
    identity['scene_rrd_files']=[]
    for directory in {str(Path(rr_identity['source']).parent),str(Path(rr_identity['bindings']).parent)}:
        members=tree_files(directory,'source')
        identity['scene_rrd_files'].extend(str(p) for p in members)
        identity['trees'][directory]={'kind':'source','members':[str(p.relative_to(directory)) for p in members]}
        for path in members:identity['files'][str(path)]={'sha256':sha(path),'resolved_path':str(path.resolve())}
    for path in native_assets:identity['files'][str(path)]={'sha256':sha(path),'resolved_path':str(path.resolve())}
    identity['native_tool_files']=list(map(str,native_assets))
    checks=[('main',cfg['main_python'],['run_rgb_scene_stage','rgb_scene_pose_graph','bind_rgb_scene_shape','numpy','scipy','cv2','open3d','trimesh','smplx'],[]),
            ('visual',cfg['demo_python'],['fit_rgb_scene_visual','fit_v20_visual_contact_demo','export_rgb_scene','rerun','rerun.experimental','torch','smplx','open3d'],[]),
            ('rgb',cfg['rgb_python'],['torch','cv2','lightglue'],[cfg['lightglue_root']]),
            ('depth',cfg['unidepth_python'],['torch','unidepth'],[cfg['unidepth_repo']]),
            ('detector',cfg['owlv2_python'],['torch','transformers'],[]),
            ('sam2',cfg['main_python'],['torch','sam2.build_sam'],[cfg['sam2_source']]),
            ('hawor',str(Path(cfg['hawor_root'])/'.venv_hawor/bin/python'),['torch','smplx','cv2'],[]),
            ('sam3d',cfg['sam3d_python'],['torch','hydra','sam3d_objects','inference'],[cfg['sam3d_repo'],str(Path(cfg['sam3d_repo'])/'notebook')])]
    results=[]
    for name,python,modules,paths in checks:
        code='import sys,importlib;sys.path[:0]='+repr([str(bundle/'scripts'),*paths])+';'+ ';'.join('importlib.import_module('+repr(m)+')' for m in modules)
        env={**os.environ,**(cfg['sam3d_environment'] if name=='sam3d' else {}),'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        done=subprocess.run([python,'-c',code],cwd=bundle,env=env,capture_output=True,text=True,timeout=1200)
        results.append({'name':name,'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr})
    manifest={'schema':'rgb_scene_bundle_v1','scripts':list(map(str,selected)),
              'files':{str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()},
              'external_assets':identity,'preflight_status':'imports_checked' if all(r['returncode']==0 for r in results) else 'failed',
              'source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()}
    (bundle/'runtime/bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report_dir.mkdir(parents=True,exist_ok=True)
    with (report_dir/'preflight.json').open('x') as f:json.dump({'bundle':str(bundle),'imports':results,'prediction_executed':False,'rerun':rr_identity},f,indent=2)
    if manifest['preflight_status']!='imports_checked':raise RuntimeError('bundle imports failed; inspect preflight report')
    return manifest


def prepare_run(bundle,root,video,case,target=None,sensor=None,sensor_source=None):
    bundle,root,video=map(lambda p:Path(p).resolve(),(bundle,root,video))
    if root.exists() or root.is_relative_to(bundle):raise ValueError('fresh run root outside bundle required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',case):raise ValueError('unsafe case')
    manifest=verify_bundle(bundle)
    if manifest['preflight_status']!='imports_checked':raise ValueError('import preflight required')
    timing=read_cfr_video_timing(video)
    if sensor:
        if find_positive_gt_flags(load(sensor)) or sensor_source is None or sha(sensor_source)!=sha(video):raise ValueError('prediction-side calibration/source byte mismatch')
    (root/'input').mkdir(parents=True)
    copied=root/'input/source.mp4';shutil.copyfile(video,copied)
    if sha(copied)!=sha(video):raise ValueError('raw copy mismatch')
    if sensor:shutil.copyfile(sensor,root/'input/sensor_calibration.json')
    contract={'schema':'rgb_scene_input_v1','run_id':str(uuid.uuid4()),'run_root':str(root),'case':case,'target_description':target,
        'input_video':str(copied),'input_video_sha256':sha(copied),'source_timing':timing.report(),
        'sensor_calibration_metadata':str(root/'input/sensor_calibration.json') if sensor else None,
        'sensor_calibration_sha256':sha(sensor) if sensor else None,'sensor_calibration_authority':'explicit_user_calibration' if sensor else None,
        'sensor_frame_intrinsics_key':None,'bundle_manifest_sha256':sha(bundle/'runtime/bundle_manifest.json')}
    (root/'input/runtime_input_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    for name in ('state','measurements','logs','renders'):(root/name).mkdir()
    return contract


def resume_contract(bundle,root,output):
    from rgb_scene_runtime import SceneLineage
    import hashlib
    bundle,root=Path(bundle).resolve(),Path(root).resolve()
    lineage=SceneLineage(root,bundle)
    events=(root/'logs/integration/events.jsonl').read_bytes()
    for line in events.splitlines():
        e=json.loads(line)
        if e.get('returncode')!=0 or 'error' in e:raise ValueError('failed/partial run cannot resume')
    evidence={}
    for name in lineage.records:evidence[name]=sha(lineage.require(name))
    for part in ('object_candidates','anchor_decisions','physical_branch_decisions','appearance_decisions','agent_interaction_judgments'):
        for p in root.rglob('*.json'):
            if part in p.parts and lineage.agent_phase(p):evidence[str(p)]=sha(lineage.require(p))
    value={'schema':'rgb_scene_verified_resume_v1','run_root':str(root),'input_contract_sha256':sha(lineage.contract_path),
           'bundle_sha256':sha(bundle/'runtime/bundle_manifest.json'),'event_prefix_bytes':len(events),
           'event_prefix_sha256':hashlib.sha256(events).hexdigest(),'evidence_sha256':evidence}
    with Path(output).open('x') as f:json.dump(value,f,indent=2)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('bundle')
    for n in ('source-root','bundle-root','asset-paths','report-dir'):b.add_argument('--'+n,type=Path,required=True)
    r=sub.add_parser('prepare-run')
    for n in ('bundle-root','run-root','video'):r.add_argument('--'+n,type=Path,required=True)
    r.add_argument('--case',required=True);r.add_argument('--target-description')
    r.add_argument('--sensor-metadata',type=Path);r.add_argument('--sensor-source-video',type=Path)
    s=sub.add_parser('resume-contract')
    for n in ('bundle-root','run-root','output'):s.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.command=='bundle':value=build(a.source_root,a.bundle_root,a.asset_paths,a.report_dir)
    elif a.command=='prepare-run':value=prepare_run(a.bundle_root,a.run_root,a.video,a.case,a.target_description,a.sensor_metadata,a.sensor_source_video)
    else:value=resume_contract(a.bundle_root,a.run_root,a.output)
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()
