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
from v19_run_lineage import RunLineage
from v19_video_timing import read_cfr_video_timing, validate_package_cfr_times
from v19_camera_contract import load_contract as load_camera_contract, plane_intrinsics
from v19_runtime_contracts import (in_run, validate_references, validate_timeline,
    run_command, runtime_environment, execution_target)

STAGES=('camera','geometry','pose','pose-rgb','interaction','physical-render','demo','export')
ROOT=Path(__file__).resolve().parents[1]
_LINEAGE=None


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
    p=in_run(value,root)
    if _LINEAGE is not None and p != root/'state/object_selection.json':
        _LINEAGE.require(p)
    if p.suffix == '.json':
        payload=load(p)
        if find_positive_gt_flags(payload):raise ValueError(f'non-prediction source: {p}')
        validate_references(payload,root)
    return p


def interpreter(value):
    # Do not resolve a virtualenv symlink and accidentally select system Python.
    p=Path(os.path.abspath(Path(value).expanduser()))
    if not p.is_file() or not os.access(p,os.X_OK):raise ValueError(f'interpreter unavailable: {p}')
    return p


def hawor_source_intrinsics(camera, frame_ids, source_size_wh):
    """Read the authoritative source plane in both sensor and RGB-fallback V2 contracts."""
    payload, normalized = load_camera_contract(camera, expected_frame_ids=frame_ids)
    K, _ = plane_intrinsics(payload, normalized, plane_name='source_rgb',
                            actual_size_wh=source_size_wh, allow_implicit_resize=False)
    if K.shape != (4,) or not np.isfinite(K).all() or np.any(K[:2] <= 0):
        raise ValueError('invalid source K')
    if not np.isclose(K[0], K[1], rtol=1e-6, atol=1e-6):
        raise ValueError('HaWoR requires square focal; cannot average supplied fx/fy')
    return K


def copy_verified_raw_video(source, destination, expected_sha256):
    """Verify the new raw copy itself before it enters the lineage cache.

    The first read can refresh write-time metadata on CIFS. The destination
    byte check is required copy verification, not a metadata-only acceptance.
    Later lineage checks and post-command byte checks remain unchanged.
    """
    import shutil
    destination=Path(destination)
    if destination.exists():raise ValueError('HaWoR sequence already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
    copied_sha256=sha(destination)
    if copied_sha256 != expected_sha256:
        raise ValueError('HaWoR raw copy byte identity mismatch')
    return copied_sha256


def camera_stage(args, cfg, root):
    main_python=interpreter(cfg['main_python'])
    depth_python=interpreter(cfg['unidepth_python'])
    manifest_path=file_in_run(root/'input/raw_frame_manifest/manifest.json',root)
    manifest=load(manifest_path)
    contract=load(file_in_run(root/'input/runtime_input_contract.json',root))
    video=file_in_run(manifest['input_video'],root)
    if sha(video) != contract['input_video_sha256']:raise ValueError('input video changed')
    if not manifest.get('full_source_timeline'):raise ValueError('full raw-video timeline required')
    frames=manifest['frames'];width=frames[0]['source_width'];height=frames[0]['source_height']
    ids=validate_timeline(manifest,manifest)
    if ids != list(range(len(ids))):raise ValueError('raw manifest must index the full decoded video')
    timing=read_cfr_video_timing(video,expected_frame_count=len(ids))
    validate_package_cfr_times([f['time_s'] for f in sorted(frames,key=lambda f:f['frame_idx'])],timing.frame_rate)
    if timing.source_size_wh != (width,height):raise ValueError('source/manifest dimensions mismatch')
    env=runtime_environment(cfg,args.gpu)
    def run(key,argv,outputs,environment=None):
        run_command(key,argv,outputs,root=root,cwd=ROOT,env=environment or env,
                    lineage=_LINEAGE,extra_inputs=[camera,clip] if key=='P04_HaWoR' else [],
                    capture_roots=[handout,clip.with_suffix('')] if key=='P04_HaWoR' else [],
                    gpu=args.gpu if key in {'P03a_intrinsics_probe','P03_conditioned_depth','P04_HaWoR'} else None)
    probe=root/'measurements/depth_slam/unidepth_intrinsics_probe'
    final=root/'measurements/depth_slam/unidepth_full_frame'
    calibration=root/'state/calibration'
    camera=calibration/'v19_camera_calibration_contract.json'
    depth_name='unidepth_full_frame_depth_v3.npz'
    def depth_command(output):
        return [depth_python,ROOT/'scripts/run_unidepth_full_frame_v3.py','--manifest',manifest_path,
            '--output-dir',output,'--frame-start',0,'--frame-end',ids[-1],
            '--unidepth-repo',cfg['unidepth_repo'],'--model-id',cfg['unidepth_model'],
            '--remote-root',manifest_path.parent,'--local-root',manifest_path.parent,
            '--source-width',width,'--source-height',height]
    sensor=contract.get('sensor_calibration_metadata')
    if sensor:
        extra=['--sensor-calibration-contract',file_in_run(sensor,root),
            '--sensor-calibration-authority',contract['sensor_calibration_authority'],
            '--prediction-source-video',video,'--sensor-source-video',video]
        if contract.get('sensor_frame_intrinsics_key'):
            extra += ['--sensor-frame-intrinsics-key',contract['sensor_frame_intrinsics_key']]
    else:
        run('P03a_intrinsics_probe',depth_command(probe),[probe/depth_name])
        extra=['--unidepth-npz',probe/depth_name,'--fallback-reason','no supplied prediction-side calibration','--aggregation','median','--square-focal']
    run('P03b_camera',[main_python,ROOT/'scripts/resolve_v19_camera_contract.py','--case',contract['case'],
        '--raw-frame-manifest',manifest_path,'--output-dir',calibration,*extra],[camera])
    K=hawor_source_intrinsics(camera, ids, (width,height))
    run('P03_conditioned_depth',depth_command(final)+['--camera-contract',camera,
        '--camera-input-plane','manifest_rgb','--camera-output-plane','source_rgb'],[final/depth_name])
    bound=calibration/'depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz'
    run('P03c_depth_binding',[main_python,ROOT/'scripts/adapt_v19_depth_to_camera_contract.py',
        '--source-depth-npz',final/depth_name,'--camera-contract',camera,'--depth-plane','source_rgb',
        '--output-dir',bound.parent],[bound])
    handout=root/'measurements/hand_candidates/hawor_world'
    if handout.exists():raise ValueError('HaWoR output already exists')
    clip=root/'input/hawor_sequence'/f"{contract['case']}.mp4"
    copied_sha256=copy_verified_raw_video(video,clip,contract['input_video_sha256'])
    hawor_env={**env,'EGO_HAWOR_ROOT':cfg['hawor_root'],'EGO_HAWOR_CASE':contract['case'],
        'EGO_HAWOR_CLIP':str(clip),'EGO_HAWOR_CLIP_SHA256':copied_sha256,'EGO_HAWOR_OUTPUT_DIR':str(handout),
        'EGO_HAWOR_IMG_FOCAL':str(K[0]),'EGO_HAWOR_CAMERA_INTRINSICS_FX_FY_CX_CY':' '.join(map(str,K)),
        'EGO_HAWOR_FORCE_FOCAL_CACHE_REFRESH':'1'}
    run('P04_HaWoR',['bash',ROOT/'scripts/remote_run_hawor_export.sh'],
        [handout/'hawor_world_hands.npz',handout/'qc_hawor_world_hands.json'],hawor_env)


def main():
    global _LINEAGE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--runtime-config',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);a=p.parse_args();root=a.run_root.resolve();cfg=load(a.runtime_config)
    execution_target(cfg,a.gpu)
    asset_role = {'geometry': 'geometry_dispatch', 'pose-rgb': 'pose'}.get(a.stage, a.stage)
    _LINEAGE=RunLineage(root,ROOT,a.runtime_config,asset_role=asset_role)
    if a.stage == 'camera':
        try:
            camera_stage(a, cfg, root)
        except Exception as exc:
            # Preparation errors between subcommands must survive a shell exit
            # just like subprocess failures; do not rely on agent file tools.
            import traceback
            directory=in_run(root/'state/runtime_blockers',root,must_exist=False)
            directory.mkdir(parents=True,exist_ok=True)
            blocker={'phase':'camera','status':'failed','error_type':type(exc).__name__,
                     'error':str(exc),'argv':sys.argv,'traceback':traceback.format_exc(),
                     'next_required_repair':'repair the failed producer/consumer interface; preserve this run'}
            (directory/'camera_stage.json').write_text(json.dumps(blocker,indent=2)+'\n')
            raise
        return
    selection=load(file_in_run(root/'state/object_selection.json',root));case=str(selection['case']);obj=str(selection['object_id']);anchor=int(selection['anchor_frame'])
    for value in [case,obj]:
        if not value or value in {'.','..'} or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in value):raise ValueError('unsafe case/object id')
    ann=file_in_run(root/f'measurements/object_geometry/visible_geometry/{obj}/annotations_v19_visible_geometry.json',root)
    annotations=load(ann)
    raw_manifest=load(file_in_run(root/'input/raw_frame_manifest/manifest.json',root))
    ids=validate_timeline(annotations,raw_manifest)
    if not ids or len(set(ids))!=len(ids):raise ValueError('invalid annotation timeline')
    lo,hi=min(ids),max(ids)
    if anchor not in ids:raise ValueError('selected anchor absent')
    plan=load(file_in_run(root/'measurements/object_candidates/object_plan_agent.json',root))
    decision=load(file_in_run(root/f'state/anchor_decisions/{obj}.json',root))
    input_contract=load(file_in_run(root/'input/runtime_input_contract.json',root))
    if input_contract['case'] != case or not any(row.get('object_id') == obj for row in plan.get('objects',[])):
        raise ValueError('selection does not match the same-run input/P05 object plan')
    if decision.get('object_id') != obj or int(decision['selected_anchor_frame_idx']) != anchor:
        raise ValueError('selection does not match the P09 visual anchor decision')
    main_python=interpreter(cfg['main_python']);demo_python=interpreter(cfg.get('demo_python',cfg['main_python']));rgb_python=interpreter(cfg.get('rgb_python',cfg['main_python']))
    def script(name):return ROOT/'scripts'/name
    def run(key,argv,outputs):
        run_command(key,argv,outputs,root=root,cwd=ROOT,env=runtime_environment(cfg,a.gpu),
                    lineage=_LINEAGE,
                    extra_inputs=[ann.parent/'v19_visible_geometry_adapter_report.json',root/'input/raw_frame_manifest/manifest.json'] if key=='P14_RGB_full_timeline_observations' else [],
                    gpu=a.gpu if key in {'P12_P13_SAM3D','P14_RGB_full_timeline_observations','P18_interval','D01_demo_fit'} else None)
    depth=root/'state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz'
    hawor=root/'measurements/hand_candidates/hawor_world/hawor_world_hands.npz';bridge=root/'state/base_annotations/v19_mano_bridge_from_hawor_world.npz'
    evidence=root/f'measurements/geometry_completion/rigid_evidence/{case}/{obj}/evidence_bundle/evidence_bundle_report.json'
    geom=root/f'measurements/geometry_completion/sam3d_{obj}';original_completion=geom/'completion_report.json'
    completion=geom/'anchor_bound_completion_report.json'
    pose_root=root/f'measurements/pose_fits/{obj}';p14=pose_root/'visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json'
    rgb=pose_root/'rgb_full_timeline/rgb_edges';initialized=pose_root/'rgb_initialized_pose.json'
    support=geom/'multiview/v19_multiview_supported_completion_report.json'
    p15=pose_root/'temporal/v19_rigid_object_pose_graph_report.json'
    p16=root/f'measurements/contact_nonpenetration/{obj}/v18_mano_object_constraint_state.json'
    judgments=root/f'state/agent_interaction_judgments/{obj}.json';p17root=root/f'measurements/contact_visibility_factors/{obj}'
    p17=p17root/case/'v19_visible_contact_ownership_factor_report.json'
    p18dir=root/f'measurements/mano_interval_correction/{obj}';p18=p18dir/case/'v18_joint_mano_interval_trajectory_state.json'
    p18b=root/f'measurements/mano_interval_correction/{obj}_metric_surface/{case}/v18_joint_mano_interval_trajectory_state.json'
    physical=root/f'state/render_state/{obj}_rigid_render_state.json'
    demo_pose=root/f'state/demo_inputs/{obj}_pose_mesh.json';demodir=root/f'measurements/demo_visual/{obj}';demo_state=demodir/'demo_state.npz'
    paths={'annotations':ann,'depth':depth,'hawor':hawor,'mano_bridge':bridge,'evidence':evidence,'original_completion':original_completion,'completion':completion,'p14':p14,'rgb_initialized_pose':initialized,'multiview_support':support,'p15':p15,'p16':p16,'p17':p17,'p18':p18,'p18b':p18b,'physical_render_state':physical,'demo_state':demo_state}
    if a.stage=='geometry':
        file_in_run(root/f'state/physical_branch_decisions/{obj}.json',root)
        file_in_run(evidence,root)
        evidence_document=load(evidence)
        if evidence_document.get('case') != case or evidence_document.get('object_id') != obj or int(evidence_document['selected_frame_idx']) != anchor:
            raise ValueError('P11 evidence does not match the selected case/object/anchor')
        run('P12_P13_SAM3D',[main_python,script('run_v19_sam3d_reconstruction.py'),'--run-root',root,'--evidence-report',evidence,'--annotations',ann,'--output-dir',geom,'--runtime-config',a.runtime_config,'--gpu',a.gpu], [original_completion])
    elif a.stage in {'pose','pose-rgb'}:
        file_in_run(original_completion,root);file_in_run(depth,root)
        if a.stage=='pose':
            run('P13_anchor_binding',[main_python,script('bind_v19_completion_anchor.py'),'--completion-report',original_completion,'--annotations',ann,'--object-id',obj,'--output',completion],[completion])
            run('P14_anchor_bound_visible_pose',[main_python,script('fit_v18_compact_rigid_object_pose.py'),'--annotations',ann,'--completion-report',completion,'--object-id',obj,'--output-dir',p14.parent],[p14])
        else:
            # require() checks the successful same-run producer and byte lineage;
            # do not overwrite the completed binding/fit or the failed RGB log.
            file_in_run(completion,root);file_in_run(p14,root)
        if a.stage=='pose-rgb':
            file_in_run(ann.parent/'v19_visible_geometry_adapter_report.json',root)
        run('P14_RGB_full_timeline_observations',[rgb_python,script('run_v19_offline_rgb_edges.py'),'--run-root',root,'--runtime-config',a.runtime_config,'--annotations',ann,'--depth-npz',depth,'--initial-pose-report',p14,'--object-id',obj,'--frame-start',lo,'--frame-end',hi,'--keyframe-interval',5,'--keyframe-neighbor-span',4,'--lightglue-root',cfg['lightglue_root'],'--device','cuda:0','--output-npz',str(rgb)+'.npz','--output-json',str(rgb)+'.json'],[Path(str(rgb)+'.npz'),Path(str(rgb)+'.json')])
        run('P14_RGB_initializer',[main_python,script('build_v19_rgb_pose_initializer.py'),'--run-root',root,'--annotations',ann,'--pose-report',p14,'--rgb-edge-npz',str(rgb)+'.npz','--rgb-edge-json',str(rgb)+'.json','--object-id',obj,'--anchor-frame',anchor,'--output',initialized],[initialized])
        run('P14b_multiview',[main_python,script('filter_v19_rigid_completion_multiview_support.py'),'--annotations',ann,'--completion-report',completion,'--pose-report',initialized,'--depth-npz',depth,'--object-id',obj,'--output-dir',support.parent],[support])
        run('P15_temporal_pose',[main_python,script('solve_v19_rigid_object_pose_graph.py'),'--annotations',ann,'--pose-report',initialized,'--completion-report',support,'--object-id',obj,'--complete-full-timeline-rigid-pose','--output-dir',p15.parent],[p15])
    elif a.stage=='interaction':
        for path in [hawor,depth,judgments,p15,support]:file_in_run(path,root)
        run('P16_constraints',[main_python,script('build_v18_mano_object_constraint_state.py'),'--annotations',ann,'--hawor-npz',hawor,'--pose-report',p15,'--completion-report',support,'--output-dir',p16.parent,'--object-id',obj],[p16])
        run('P17_contact_ownership',[main_python,script('build_v19_visible_contact_ownership_factor.py'),'--annotations',ann,'--case',case,'--target-entity-id','object:'+obj,'--frame-span',lo,hi,'--output-root',p17root,'--agent-interaction-judgment',judgments],[p17])
        body=file_in_run(load(support)['outputs']['pose_hypothesis_mesh_labeled'],root)
        run('P18_interval',[main_python,script('solve_v18_joint_mano_interval_trajectory.py'),'--case',case,'--object-id','object:'+obj,'--annotations',ann,'--pose-report',p15,'--completed-mesh',body,'--completion-report',support,'--depth-npz',depth,'--wilor-root',cfg['wilor_root'],'--wilor-mano-left',Path(cfg['mano_models'])/'MANO_LEFT.pkl','--wilor-mano-right',Path(cfg['mano_models'])/'MANO_RIGHT.pkl','--device','cuda:0','--output-dir',p18dir,'--start-frame',lo,'--end-frame',hi,'--sides','left','right','--factor-report',p17,'--require-active-full-K-mano-contract','--optimize-contact-state','--visible-surface-depth-order-term','--freeze-translation-without-visible-surface-support','--gate-translation-with-visible-surface-support','--translation-gate-min-visible-surface-depth-vertices',0],[p18])
        run('P18b_metric_preservation',[main_python,script('build_v19_mano_surface_hypothesis_state.py'),'--contact-state',p18,'--joint-source','hawor_npz','--hawor-npz',hawor,'--case',case,'--object-id',obj,'--output',p18b],[p18b])
    elif a.stage=='physical-render':
        for path in [p15,p16,p18b,support]:file_in_run(path,root)
        body=file_in_run(load(support)['outputs']['pose_hypothesis_mesh_labeled'],root)
        run('P19a_physical_state',[main_python,script('build_v19_rigid_render_state.py'),'--case',case,'--object-id',obj,'--annotations',ann,'--pose-report',p15,'--completed-mesh',body,'--completion-report',support,'--constraint-report',p16,'--temporal-mano-state',p18b,'--output',physical],[physical])
        renderroot=root/f'renders/{obj}_physical';manifest=renderroot/case/'v19_rigid_state_render_manifest.json'
        run('P19b_physical_render',[main_python,script('render_v19_rigid_state_artifact.py'),'--render-state',physical,'--output-root',renderroot,'--world-view','local'],[manifest])
    elif a.stage=='demo':
        for path in [p15,hawor,bridge,judgments,physical,completion]:file_in_run(path,root)
        source=load(p15);geomreport=load(completion);contract=geomreport['render_mesh_contract'];file_in_run(contract['required_mesh_path'],root)
        run('D00_demo_binding',[main_python,script('build_v19_demo_pose_binding.py'),'--pose-report',p15,'--completion-report',completion,'--legacy-ready-p15-binding','--output',demo_pose],[demo_pose])
        run('D01_demo_fit',[demo_python,script('fit_v20_visual_contact_demo.py'),'--annotations',ann,'--pose-report',demo_pose,'--mano-bridge',bridge,'--hawor-params',hawor,'--mano-models',cfg['mano_models'],'--object-id',obj,'--device','cuda:0','--interaction-judgments',judgments,'--legacy-owned-mask','--render-width',cfg.get('render_width',960),'--output-dir',demodir],[demo_state,demodir/'fit_report.json'])
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
