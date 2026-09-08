#!/usr/bin/env python3
"""Execute named same-run V19 SAM3D stages while retaining the physical branch.

Upstream P00-P11 and P17 visual judgments follow the runtime spec. This runner
never invents those observations and never substitutes demo state for P19.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
from v20_prediction_contracts import find_positive_gt_flags

STAGES=('geometry','pose','interaction','physical-render','demo','export')
ROOT=Path(__file__).resolve().parents[1]


def load(path):
    x=json.loads(Path(path).read_text())
    if not isinstance(x,dict):raise ValueError(f'expected JSON object: {path}')
    return x


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def file_in_run(value,root):
    p=Path(value).expanduser().resolve()
    if not p.is_relative_to(root) or not p.is_file():raise ValueError(f'missing or out-of-run input: {value}')
    return p


def interpreter(value):
    # Do not resolve a virtualenv symlink and accidentally select system Python.
    p=Path(os.path.abspath(Path(value).expanduser()))
    if not p.is_file() or not os.access(p,os.X_OK):raise ValueError(f'interpreter unavailable: {p}')
    return p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--runtime-config',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);a=p.parse_args();root=a.run_root.resolve();cfg=load(a.runtime_config)
    selection=load(file_in_run(root/'state/object_selection.json',root));case=str(selection['case']);obj=str(selection['object_id']);anchor=int(selection['anchor_frame'])
    for value in [case,obj]:
        if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in value):raise ValueError('unsafe case/object id')
    ann=file_in_run(root/f'measurements/object_geometry/visible_geometry/{obj}/annotations_v19_visible_geometry.json',root)
    annotations=load(ann);ids=[int(f['frame_idx']) for f in annotations['frames']]
    if not ids or len(set(ids))!=len(ids):raise ValueError('invalid annotation timeline')
    lo,hi=min(ids),max(ids)
    if anchor not in ids:raise ValueError('selected anchor absent')
    main_python=interpreter(cfg['main_python']);demo_python=interpreter(cfg.get('demo_python',cfg['main_python']));rgb_python=interpreter(cfg.get('rgb_python',cfg['main_python']))
    def script(name):return ROOT/'scripts'/name
    def run(key,argv,outputs):
        for dest in outputs:
            if Path(dest).exists():raise ValueError(f'{key} output already exists; not a fresh stage: {dest}')
        logdir=root/'logs/integration';logdir.mkdir(parents=True,exist_ok=True);log=logdir/f'{key}.log';started=time.monotonic()
        with log.open('w') as f:result=subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu)})
        event={'phase':key,'argv':list(map(str,argv)),'returncode':result.returncode,'elapsed_s':time.monotonic()-started,'log':str(log),'outputs':list(map(str,outputs))}
        with (logdir/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        if result.returncode:raise RuntimeError(f'{key} failed: {log}')
        for dest in outputs:file_in_run(dest,root)
    depth=root/'state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz'
    hawor=root/'measurements/hand_candidates/hawor_world/hawor_world_hands.npz';bridge=root/'state/base_annotations/v19_mano_bridge_from_hawor_world.npz'
    evidence=root/f'measurements/geometry_completion/rigid_evidence/{case}/{obj}/evidence_bundle/evidence_bundle_report.json'
    geom=root/f'measurements/geometry_completion/sam3d_{obj}';completion=geom/'completion_report.json'
    pose_root=root/f'measurements/pose_fits/{obj}';p14=pose_root/'visible/v18_compact_rigid_object_pose_fit_report.json'
    rgb=pose_root/'rgb/rgb_edges';initialized=pose_root/'rgb_initialized_pose.json'
    support=geom/'multiview/v19_multiview_supported_completion_report.json'
    p15=pose_root/'temporal/v19_rigid_object_pose_graph_report.json'
    p16=root/f'measurements/contact_nonpenetration/{obj}/v18_mano_object_constraint_state.json'
    judgments=root/f'state/agent_interaction_judgments/{obj}.json';p17root=root/f'measurements/contact_visibility_factors/{obj}'
    p17=p17root/case/'v19_visible_contact_ownership_factor_report.json'
    p18dir=root/f'measurements/mano_interval_correction/{obj}';p18=p18dir/case/'v18_joint_mano_interval_trajectory_state.json'
    p18b=root/f'measurements/mano_interval_correction/{obj}_metric_surface/{case}/v18_joint_mano_interval_trajectory_state.json'
    physical=root/f'state/render_state/{obj}_rigid_render_state.json'
    demo_pose=root/f'state/demo_inputs/{obj}_pose_mesh.json';demodir=root/f'measurements/demo_visual/{obj}';demo_state=demodir/'demo_state.npz'
    paths={'annotations':ann,'depth':depth,'hawor':hawor,'mano_bridge':bridge,'evidence':evidence,'completion':completion,'p14':p14,'rgb_initialized_pose':initialized,'multiview_support':support,'p15':p15,'p16':p16,'p17':p17,'p18':p18,'p18b':p18b,'physical_render_state':physical,'demo_state':demo_state}
    if a.stage=='geometry':
        file_in_run(evidence,root)
        run('P12_P13_SAM3D',[main_python,script('run_v19_sam3d_reconstruction.py'),'--run-root',root,'--evidence-report',evidence,'--output-dir',geom,'--runtime-config',a.runtime_config,'--gpu',a.gpu], [completion])
    elif a.stage=='pose':
        file_in_run(completion,root);file_in_run(depth,root)
        run('P14_visible_pose',[main_python,script('fit_v18_compact_rigid_object_pose.py'),'--annotations',ann,'--completion-report',completion,'--object-id',obj,'--output-dir',p14.parent],[p14])
        run('P14_RGB_observations',[rgb_python,script('build_v20_global_rgb_orientation_edges.py'),'--annotations',ann,'--depth-npz',depth,'--initial-pose-report',p14,'--object-id',obj,'--frame-start',lo,'--frame-end',hi,'--keyframe-interval',5,'--keyframe-neighbor-span',4,'--lightglue-root',cfg['lightglue_root'],'--device','cuda:0','--output-npz',str(rgb)+'.npz','--output-json',str(rgb)+'.json'],[Path(str(rgb)+'.npz'),Path(str(rgb)+'.json')])
        run('P14_RGB_initializer',[main_python,script('build_v19_rgb_pose_initializer.py'),'--annotations',ann,'--pose-report',p14,'--rgb-edge-npz',str(rgb)+'.npz','--rgb-edge-json',str(rgb)+'.json','--object-id',obj,'--anchor-frame',anchor,'--output',initialized],[initialized])
        run('P14b_multiview',[main_python,script('filter_v19_rigid_completion_multiview_support.py'),'--annotations',ann,'--completion-report',completion,'--pose-report',initialized,'--depth-npz',depth,'--object-id',obj,'--output-dir',support.parent],[support])
        run('P15_temporal_pose',[main_python,script('solve_v19_rigid_object_pose_graph.py'),'--annotations',ann,'--pose-report',initialized,'--completion-report',support,'--object-id',obj,'--complete-full-timeline-rigid-pose','--output-dir',p15.parent],[p15])
    elif a.stage=='interaction':
        for path in [hawor,depth,judgments,p15,support]:file_in_run(path,root)
        run('P16_constraints',[main_python,script('build_v18_mano_object_constraint_state.py'),'--annotations',ann,'--hawor-npz',hawor,'--pose-report',p15,'--completion-report',support,'--output-dir',p16.parent,'--object-id',obj],[p16])
        run('P17_contact_ownership',[main_python,script('build_v19_visible_contact_ownership_factor.py'),'--annotations',ann,'--case',case,'--target-entity-id','object:'+obj,'--frame-span',lo,hi,'--output-root',p17root,'--agent-interaction-judgment',judgments],[p17])
        body=file_in_run(load(completion)['outputs']['pose_hypothesis_mesh_labeled'],root)
        run('P18_interval',[main_python,script('solve_v18_joint_mano_interval_trajectory.py'),'--case',case,'--object-id','object:'+obj,'--annotations',ann,'--pose-report',p15,'--completed-mesh',body,'--completion-report',support,'--depth-npz',depth,'--wilor-root',cfg['wilor_root'],'--wilor-mano-left',Path(cfg['mano_models'])/'MANO_LEFT.pkl','--output-dir',p18dir,'--start-frame',lo,'--end-frame',hi,'--sides','left','right','--factor-report',p17,'--optimize-contact-state','--visible-surface-depth-order-term','--freeze-translation-without-visible-surface-support','--gate-translation-with-visible-surface-support','--translation-gate-min-visible-surface-depth-vertices',0],[p18])
        run('P18b_metric_preservation',[main_python,script('build_v19_mano_surface_hypothesis_state.py'),'--contact-state',p18,'--joint-source','hawor_npz','--hawor-npz',hawor,'--case',case,'--object-id',obj,'--output',p18b],[p18b])
    elif a.stage=='physical-render':
        for path in [p15,p16,p18b,support]:file_in_run(path,root)
        body=file_in_run(load(completion)['outputs']['pose_hypothesis_mesh_labeled'],root)
        run('P19a_physical_state',[main_python,script('build_v19_rigid_render_state.py'),'--case',case,'--object-id',obj,'--annotations',ann,'--pose-report',p15,'--completed-mesh',body,'--completion-report',support,'--constraint-report',p16,'--temporal-mano-state',p18b,'--output',physical],[physical])
        renderroot=root/f'renders/{obj}_physical';manifest=renderroot/case/'v19_rigid_state_render_manifest.json'
        run('P19b_physical_render',[main_python,script('render_v19_rigid_state_artifact.py'),'--render-state',physical,'--output-root',renderroot,'--world-view','local'],[manifest])
    elif a.stage=='demo':
        for path in [p15,hawor,bridge,judgments,physical,completion]:file_in_run(path,root)
        source=load(p15);geomreport=load(completion);contract=geomreport['render_mesh_contract'];file_in_run(contract['required_mesh_path'],root)
        # Copy only for visual fitting. Never mutate the source P15 readiness,
        # core physical state, metric MANO or P18b contact surface state.
        if demo_pose.exists():raise ValueError('demo input binding already exists')
        if find_positive_gt_flags(source):raise ValueError('non-prediction demo source')
        visual=dict(source);visual.update(diagnostic_only=True,annotation_ready=False,gt_consumed=False,render_mesh_contract=contract)
        visual['core_source_identity']={'pose_report':str(p15),'sha256':sha(p15),'source_readiness':source.get('annotation_ready'),'source_graph_support':source.get('graph_support')}
        demo_pose.parent.mkdir(parents=True,exist_ok=True);demo_pose.write_text(json.dumps(visual,indent=2)+'\n')
        run('D01_demo_fit',[demo_python,script('fit_v20_visual_contact_demo.py'),'--annotations',ann,'--pose-report',demo_pose,'--mano-bridge',bridge,'--hawor-params',hawor,'--mano-models',cfg['mano_models'],'--object-id',obj,'--device','cuda:0','--interaction-judgments',judgments,'--render-width',cfg.get('render_width',960),'--output-dir',demodir],[demo_state,demodir/'fit_report.json'])
        input_manifest=load(file_in_run(root/'input/raw_frame_manifest/manifest.json',root));video=input_manifest.get('input_video') or input_manifest.get('video')
        if not video:raise ValueError('raw manifest has no original video path')
        video=file_in_run(video,root)
        run('D02_demo_render',[demo_python,script('render_v20_visual_contact_demo.py'),'--state',demo_state,'--source-video',video,'--output-dir',root/f'renders/{obj}_demo'],[root/f'renders/{obj}_demo/render_report.json'])
    elif a.stage=='export':
        for path in [physical,judgments,hawor,demo_state,demodir/'fit_report.json']:file_in_run(path,root)
        final=root/'final'
        run('P20_layered_annotations',[main_python,script('assemble_v19_annotation_layers.py'),'--render-state',physical,'--annotations',ann,'--hawor-npz',hawor,'--interaction-judgments',judgments,'--demo-state',demo_state,'--demo-report',demodir/'fit_report.json','--run-root',root,'--output-dir',final],[final/'annotation_layers.json'])
        run('P20_world_rrd',[main_python,script('render_v19_annotation_layers_rrd.py'),'--package',final/'annotation_layers.json','--output',final/'world.rrd','--layer','both'],[final/'world.rrd'])
        manifest=load(file_in_run(root/'input/raw_frame_manifest/manifest.json',root))
        video=file_in_run(manifest['input_video'],root)
        source_contract=load(file_in_run(root/'input/runtime_input_contract.json',root))
        video_hash=source_contract['input_video_sha256']
        if sha(video)!=video_hash:raise ValueError('original input video changed')
        run('P20_package_video',[demo_python,script('render_v19_annotation_layers_video.py'),'--package',final/'annotation_layers.json','--source-video',video,'--source-video-sha256',video_hash,'--output-dir',final/'video'],[final/'video/annotation_layers.mp4',final/'video/render_report.json'])
    index=root/'state/integration_artifacts.json';existing=load(index) if index.exists() else {'schema':'v19_sam3d_integration_artifacts_v1','stages_executed':[]}
    existing['stages_executed'].append(a.stage);existing['paths']={k:str(v) for k,v in paths.items() if v.is_file()};existing['demo_is_additive']=True
    index.write_text(json.dumps(existing,indent=2)+'\n');print(json.dumps({'stage':a.stage,'status':'executed','artifact_index':str(index)}))


if __name__=='__main__':main()
