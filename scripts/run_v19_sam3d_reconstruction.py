#!/usr/bin/env python3
"""SAM3D-only P11-native/P12/P13 production, preserving physical face labels.

No TRELLIS inference, A/B comparison or historical candidate is required.
The existing native bridge and shared labeling mechanism are reused unchanged.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from v19_run_lineage import RunLineage
from v20_prediction_contracts import assert_prediction_only
from v19_runtime_contracts import validate_references, runtime_environment, execution_target


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def inside(path, root):
    resolved=Path(path).expanduser().resolve()
    if not resolved.is_relative_to(root):raise ValueError(f'clip-specific input escapes run root: {path}')
    if not resolved.is_file():raise ValueError(f'input file missing: {path}')
    return resolved


def validate_clip_references(value, root):
    """Check prediction assets before legacy producers can copy an outside file."""
    validate_references(value, root, must_exist=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['run-root','evidence-report','output-dir','runtime-config']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    root=a.run_root.resolve();evidence=inside(a.evidence_report,root);out=a.output_dir.resolve()
    if not out.is_relative_to(root) or out.exists():raise ValueError('new output directory within run root required')
    config=json.loads(a.runtime_config.read_text());execution_target(config,a.gpu)
    ev=json.loads(evidence.read_text());assert_prediction_only(ev,label='SAM3D evidence')
    RunLineage(root,Path(__file__).resolve().parents[1],a.runtime_config,asset_role='geometry').require(evidence)
    validate_clip_references(ev,root)
    repo=Path(__file__).resolve().parents[1];exp=repo/'experiments/sam3d_p11_p12_branch'
    p12=module(exp/'run_p12_parallel_geometry_priors.py','integration_sam3d_contract')
    python=Path(os.path.abspath(Path(config['main_python']).expanduser()))
    sam_python=Path(os.path.abspath(Path(config['sam3d_python']).expanduser()))
    for exe in (python,sam_python):
        if not exe.is_file() or not os.access(exe,os.X_OK):raise ValueError(f'interpreter missing: {exe}')
    required=['sam3d_repo','sam3d_config','sam3d_moge_checkpoint','sam3d_dinov2_repo','sam3d_dinov2_checkpoint']
    for k in required:
        if not Path(config[k]).exists():raise ValueError(f'model asset missing: {k}')
    out.mkdir(parents=True);steps=[]
    def run(key,argv):
        started=time.monotonic();log=out/f'{key}.log'
        overlay=config.get('sam3d_environment',{}) if key=='P12_SAM3D' else {}
        if not isinstance(overlay,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in overlay.items()):
            raise ValueError('sam3d_environment must contain string environment values')
        environment={**runtime_environment(config,a.gpu),**overlay,'CUDA_VISIBLE_DEVICES':str(a.gpu)}
        try:
            with log.open('x') as f:done=subprocess.run(list(map(str,argv)),cwd=repo,stdout=f,stderr=subprocess.STDOUT,env=environment)
        except OSError as exc:
            steps.append({'phase':key,'argv':list(map(str,argv)),'error':str(exc),'log':str(log)})
            (out/'execution.json').write_text(json.dumps({'steps':steps},indent=2)+'\n')
            raise
        steps.append({'phase':key,'argv':list(map(str,argv)),'returncode':done.returncode,'elapsed_s':time.monotonic()-started,'log':str(log)})
        (out/'execution.json').write_text(json.dumps({'steps':steps,'source_artifacts_mutated':False},indent=2)+'\n')
        if done.returncode:raise RuntimeError(f'{key} failed: {log}')
    # Preflight validates model placement; this also checks live availability.
    probe=p12.gpu_preflight(a.gpu,int(config.get('sam3d_min_free_mib',12000)))
    run('P11_native_inputs',[python,exp/'build_p11_dual_geometry_inputs.py','--evidence-report',evidence,'--output-dir',out/'native_inputs'])
    native_path=out/'native_inputs/p11_dual_geometry_inputs_report.json';native=json.loads(native_path.read_text());contract=native['conditioning_contracts']['sam3d_objects_native']
    validate_clip_references(native,root)
    image=inside(contract['image'],root);mask=inside(contract['mask'],root)
    if contract.get('external_pointmap') is not None:raise ValueError('SAM3D external pointmap not authorized')
    name=p12.safe_id(f"{ev.get('case')}_{ev.get('object_id')}_frame_{ev['selected_frame_idx']}")
    command=[sam_python,repo/'scripts/remote_run_sam3d_objects_mesh_v7.py','--repo',config['sam3d_repo'],'--config',config['sam3d_config'],
             '--moge-checkpoint',config['sam3d_moge_checkpoint'],'--dinov2-repo',config['sam3d_dinov2_repo'],'--dinov2-checkpoint',config['sam3d_dinov2_checkpoint'],
             '--case',f'{name}|{image}|{mask}|{a.seed}','--output-dir',out/'native_reconstruction']
    run('P12_SAM3D',command)
    raw_report=inside(out/f'native_reconstruction/{name}/qc_sam3d_objects_mesh_v7.json',root)
    candidate=p12.sam3d_candidate_record(raw_report,name,image,mask,{'mode':'fresh_native_SAM3D_only','command':list(map(str,command)),'gpu_preflight':probe})
    # Native geometry records are clip outputs; weights/configs in the runner's
    # separate offline-asset binding intentionally remain reusable assets.
    validate_clip_references(candidate['native_outputs'],root)
    p12report=out/'sam3d_prior_report.json';p12report.write_text(json.dumps({'schema':'v19_sam3d_only_prior_v1','status':'ok','candidates':{'sam3d_objects':candidate},'gt_consumed':False,'native_input_report':str(native_path)},indent=2)+'\n')
    run('P13_native_bridge',[python,exp/'build_p13_sam3d_native_metric_bridge.py','--evidence-report',evidence,'--p12-report',p12report,'--output-dir',out/'metric_bridge'])
    bridge_path=out/'metric_bridge/sam3d_native_sensor_metric_canonical_bridge_report.json';bridge=json.loads(bridge_path.read_text())
    canonical=inside(bridge['outputs']['metric_canonical_render_prior'],root)
    # trellis-report is a historical argument NAME of the backend-neutral
    # labeling builder. The payload explicitly selects verified SAM3D identity.
    run('P13_face_support',[python,repo/'scripts/build_v18_compact_rigid_trellis_completion.py','--evidence-report',evidence,'--trellis-report',out/'metric_bridge/p13_metric_canonical_input.json',
             '--input-mesh-already-metric-canonical','--output-dir',out/'face_support','--observed-band-scale',math.sqrt(3.),'--silhouette-free-space-filter','--planar-slab-support-filter'])
    labeled_path=out/'face_support/v18_compact_rigid_trellis_completion_report.json';labeled=json.loads(labeled_path.read_text())
    report=dict(labeled)
    report.update(schema='v19_sam3d_only_completion_v1',source_model='sam3d_objects',gt_consumed=False)
    report['native_metric_bridge']={'path':str(bridge_path),'sha256':p12.sha256_file(bridge_path)}
    report['render_mesh_contract']={'required_mesh_path':str(canonical),'required_mesh_sha256':p12.sha256_file(canonical),'renderer_must_match':True}
    report['outputs']['intact_generated_render_mesh']=str(canonical)
    report['inputs']={**report['inputs'],'sam3d_prior_report':str(p12report),'sam3d_runner_report':str(raw_report),'source_run_root':str(root)}
    report['phase_retention']={'physical_face_support':True,'generated_single_view_sign_promoted':False,'trellis_inference_executed':False,'comparison_required':False}
    report['input_hashes']={'evidence_report':p12.sha256_file(evidence),'sam3d_prior_report':p12.sha256_file(p12report),'native_canonical_mesh':p12.sha256_file(canonical)}
    destination=out/'completion_report.json';destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'completion_report':str(destination),'intact_demo_mesh':str(canonical),'physical_surface':report['outputs']['collision_eligible_mesh_labeled']}))


if __name__=='__main__':main()
