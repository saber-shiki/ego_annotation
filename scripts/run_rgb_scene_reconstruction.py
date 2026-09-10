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
import sys
import time
from pathlib import Path
from bind_v19_completion_anchor import normalize_completion, read_report
from rgb_scene_runtime import SceneLineage as RunLineage, probe
from rgb_scene_process import run_owned, OwnedProcessFailure, RECONSTRUCTION_BUDGET_S
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


def recorded_step(key, argv, *, out, steps, repo, environment, timeout):
    """Persist the precise inner outcome, including cancellation and cleanup."""
    try:
        record=run_owned(key,argv,cwd=repo,env=environment,log=out/f'{key}.log',timeout=timeout)
    except OwnedProcessFailure as exc:
        steps.append(exc.record)
        (out/'execution.json').write_text(json.dumps({'steps':steps,'source_artifacts_mutated':False},indent=2)+'\n')
        raise
    steps.append(record)
    (out/'execution.json').write_text(json.dumps({'steps':steps,'source_artifacts_mutated':False},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['run-root','evidence-report','annotations','output-dir','runtime-config']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--candidate-context',type=Path)
    p.add_argument('--reuse-native-qc',type=Path)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    root=a.run_root.resolve();evidence=inside(a.evidence_report,root);out=a.output_dir.resolve()
    if not out.is_relative_to(root) or out.exists():raise ValueError('new output directory within run root required')
    config=json.loads(a.runtime_config.read_text());execution_target(config,a.gpu)
    ev,evidence_sha256=read_report(evidence);assert_prediction_only(ev,label='SAM3D evidence')
    annotations_path=inside(a.annotations,root)
    from rgb_scene_candidates import load_candidate
    candidate=None
    if a.candidate_context:
        candidate,_=load_candidate(a.candidate_context)
        if candidate.run_root!=root: raise ValueError('candidate context belongs to another run')
    lineage=RunLineage(root,Path(__file__).resolve().parents[1],a.runtime_config,asset_role='geometry',candidate=candidate)
    if a.reuse_native_qc is not None and candidate is None:
        raise ValueError('native reuse requires candidate context')
    lineage.require(evidence);lineage.require(annotations_path)
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
    deadline=time.monotonic()+RECONSTRUCTION_BUDGET_S
    def run(key,argv):
        overlay=config.get('sam3d_environment',{}) if key=='P12_SAM3D' else {}
        if not isinstance(overlay,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in overlay.items()):
            raise ValueError('sam3d_environment must contain string environment values')
        environment={**runtime_environment(config,a.gpu),**overlay,'CUDA_VISIBLE_DEVICES':str(a.gpu)}
        if key == 'P12_SAM3D': probe(config,a.gpu,int(config.get('sam3d_min_free_mib',12000)))
        recorded_step(key,argv,out=out,steps=steps,repo=repo,environment=environment,
                      timeout=max(0,deadline-time.monotonic()))
    # Preflight validates model placement; this also checks live availability.
    gpu_record=probe(config,a.gpu,int(config.get('sam3d_min_free_mib',12000)))
    run('P11_native_inputs',[python,exp/'build_p11_dual_geometry_inputs.py','--evidence-report',evidence,'--output-dir',out/'native_inputs'])
    native_path=out/'native_inputs/p11_dual_geometry_inputs_report.json';native=json.loads(native_path.read_text());contract=native['conditioning_contracts']['sam3d_objects_native']
    validate_clip_references(native,root)
    image=inside(contract['image'],root);mask=inside(contract['mask'],root)
    if contract.get('external_pointmap') is not None:raise ValueError('SAM3D external pointmap not authorized')
    name=p12.safe_id(f"{ev.get('case')}_{ev.get('object_id')}_frame_{ev['selected_frame_idx']}")
    if a.reuse_native_qc is not None:
        imported_qc=inside(a.reuse_native_qc,root); imported=json.loads(imported_qc.read_text())
        # Rebind only the conditioning paths: bytes and selected frame must be
        # identical to current P11 inputs. The imported QC and source receipt
        # remain immutable evidence; this is a derived path mapping.
        for key,current in (('image',image),('mask',mask)):
            old=inside(imported.get(key,''),root)
            if p12.sha256_file(old) != p12.sha256_file(current):
                raise ValueError(f'reused native {key} differs from current P11 conditioning')
        mapped=dict(imported); mapped['image']=str(image); mapped['mask']=str(mask)
        mapped['reuse_source_qc']={'path':str(imported_qc),'sha256':p12.sha256_file(imported_qc)}
        mapped['derived_qc_mapping']={**(mapped.get('derived_qc_mapping') or {}),
                                      'current_p11_image':str(image),'current_p11_mask':str(mask),
                                      'current_p11_image_sha256':p12.sha256_file(image),
                                      'current_p11_mask_sha256':p12.sha256_file(mask)}
        reuse_dir=out/'native_reuse_mapping';reuse_dir.mkdir(parents=True)
        raw_report=reuse_dir/'qc_sam3d_objects_mesh_v7.json';raw_report.write_text(json.dumps(mapped,indent=2)+'\n')
        candidate=p12.sam3d_candidate_record(raw_report,name,image,mask,{'mode':'verified_native_reuse','source_qc':str(imported_qc)})
    else:
        # Native geometry records are clip outputs; weights/configs in the runner's
        # separate offline-asset binding intentionally remain reusable assets.
        command=[sam_python,repo/'scripts/remote_run_sam3d_objects_mesh_v7.py','--repo',config['sam3d_repo'],'--config',config['sam3d_config'],
                 '--moge-checkpoint',config['sam3d_moge_checkpoint'],'--dinov2-repo',config['sam3d_dinov2_repo'],'--dinov2-checkpoint',config['sam3d_dinov2_checkpoint'],
                 '--case',f'{name}|{image}|{mask}|{a.seed}','--output-dir',out/'native_reconstruction']
        run('P12_SAM3D',command)
        raw_report=inside(out/f'native_reconstruction/{name}/qc_sam3d_objects_mesh_v7.json',root)
        candidate=p12.sam3d_candidate_record(raw_report,name,image,mask,{'mode':'fresh_native_SAM3D_only','command':list(map(str,command)),'gpu_preflight':gpu_record})
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
    report['input_hashes']={'evidence_report':evidence_sha256,'sam3d_prior_report':p12.sha256_file(p12report),'native_canonical_mesh':p12.sha256_file(canonical)}
    annotations,_=read_report(annotations_path)
    report=normalize_completion(report,annotations,str(ev['object_id']))
    destination=out/'completion_report.json';destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'completion_report':str(destination),'intact_demo_mesh':str(canonical),'physical_surface':report['outputs']['collision_eligible_mesh_labeled']}))


if __name__=='__main__':main()
