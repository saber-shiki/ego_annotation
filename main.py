"""Observation-only producer for versioned ego-video source bundles."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from utils.data import indexed, load
from utils.state import sha









def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command', required=True)
    for command, help_text in (('observe', 'produce source observations from a raw video'),
                               ('annotate', 'deprecated name for observation-only production')):
        video=commands.add_parser(command, help=help_text)
        video.add_argument('--video',type=Path,required=True)
        video.add_argument('--run-root',type=Path,required=True)
        video.add_argument('--case',required=True)
        video.add_argument('--assets-manifest',type=Path,required=True)
        video.add_argument('--gpu',type=int,required=True)
        video.add_argument('--gpu-uuid')
        video.add_argument('--sensor-calibration',type=Path)
        video.add_argument('--target-description')
        video.add_argument('--observation-bundle-version', type=int, choices=(1,2), default=2,
            help='v2 adds complete preprocessing observations; v1 is legacy training compatibility')
    resume=commands.add_parser('resume',help='continue verified source-observation production')
    resume.add_argument('--run-root',type=Path,required=True)
    resume.add_argument('--assets-manifest',type=Path,required=True)
    resume.add_argument('--gpu',type=int,required=True)
    resume.add_argument('--gpu-uuid')
    resume.add_argument('--rewind',help='invalidate this named observation stage and its dependent suffix')
    resume.add_argument('--reason',help='required explanation for explicit rewind')
    resume.add_argument('--candidate',help='label the new same-run observation suffix')
    judgment=commands.add_parser('judge',help='run one independent Astra observation judgment')
    judgment.add_argument('name',choices=('object','mask_track','pose_anchor','rigidity','appearance_anchor',
        'pose_trajectory','contact','contact_pads','occlusion','depth_reliability','visibility',
        'backend_geometry','final_inspection'))
    judgment.add_argument('--evidence',type=Path,required=True)
    judgment.add_argument('--image',type=Path,action='append',required=True)
    judgment.add_argument('--output',type=Path,required=True)
    preflight=commands.add_parser('preflight',help='validate isolated observation-provider imports and assets')
    preflight.add_argument('--assets-manifest',type=Path,required=True)
    prepare=commands.add_parser('prepare',help='build a retained source observation or geometry input')
    prepare.add_argument('kind',choices=('image-targets','visible-surface-track','surface-eligibility','hand-visibility','signed-proxy','multiview-support'))
    prepare.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        from observations.object import surface, signed_geometry
        from observations.interaction import evidence
        builders={'visible-surface-track':(surface.build_visible_surface_track,surface.visible_track_parse_args),
            'surface-eligibility':(surface.build_face_eligibility,surface.face_eligibility_parse_args),
            'hand-visibility':(evidence.build_hand_visibility,evidence.hand_visibility_parse_args),
            'signed-proxy':(signed_geometry.signed_proxy_build,signed_geometry.signed_proxy_parse_args)}
        if args.kind=='image-targets':return surface.build_pose_image_targets(args.arguments)
        if args.kind=='multiview-support':return signed_geometry.filter_multiview_cli(args.arguments)
        build,parse=builders[args.kind];return build(parse(args.arguments))
    if args.command=='preflight':
        from utils.config import preflight_providers
        result=preflight_providers(load(args.assets_manifest));print(json.dumps(result,indent=2));return result
    if args.command in {'observe','annotate','resume'}:
        result=annotate(args);print(json.dumps(result,indent=2));return result
    import importlib
    if args.output.exists():raise ValueError('judgment output must be create-only')
    module=importlib.import_module('observations.judgments.judge_'+args.name)
    value=module.judge(load(args.evidence),args.image)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False)
    return value




def scene_review_summary(mask, trajectory, inspections):
    rejected=(not mask['identity_consistent'] or not trajectory['trajectory_consistent']
              or any(row['visual_result']=='rejected' for row in inspections))
    defects=(any(row['state']!='usable' for row in mask['intervals'])
             or bool(trajectory['flagged_intervals'])
             or any(row['visual_result']!='reviewed_usable' or bool(row.get('defects')) for row in inspections))
    status='rejected' if rejected else 'pending_visual_review' if not inspections else 'reviewed_with_defects' if defects else 'reviewed_usable'
    return {'status':status,'accepted':status=='reviewed_usable',
        'physical_authority_granted':False,
        'mask_track':{k:v for k,v in mask.items() if k!='judgment_provenance'},
        'pose_trajectory':{k:v for k,v in trajectory.items() if k!='judgment_provenance'},
        'final_inspections':[{k:v for k,v in row.items() if k!='judgment_provenance'} for row in inspections]}


def appearance_candidate_ids(rows, annotations, object_id, preferred):
    from observations.object.shape import metric_alignment, metric_object
    frames=indexed(annotations['frames'])
    def aligned(frame):
        return metric_alignment(metric_object(frames[frame],object_id),frames[frame]) is not None
    if rows[preferred].get('anchor_eligible') is not True or not aligned(preferred):raise ValueError('preferred appearance anchor lost retained metric alignment')
    ranked=sorted((row for row in rows.values() if row['status']=='available' and row.get('anchor_eligible') is True and aligned(row['frame_idx'])),
                  key=lambda row:(-float(row.get('score',0.)),row['frame_idx']))
    return list(dict.fromkeys([preferred,*[row['frame_idx'] for row in ranked[:12]]]))


def detector_prompt_contract(obj,images,assets,directory):
    """Use actual model tokenization, asking the same Astra judge to shorten if needed."""
    import os,subprocess
    from observations.judgments.judge_object import shorten_detector_prompts
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    prompts=list(obj['detector_text_prompts']);history=[]
    for attempt in range(3):
        prompt_path=directory/f'prompts_{attempt}.json';report_path=directory/f'tokens_{attempt}.json'
        prompt_path.write_text(json.dumps({'prompts':prompts},indent=2))
        subprocess.run([assets['owlv2_python'],'-m','observations.object.owlv2','text-contract',
            '--model',assets['owlv2_model'],'--prompts',str(prompt_path),'--output',str(report_path)],
            cwd=Path(__file__).resolve().parent,env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parent)+os.pathsep+os.environ.get('PYTHONPATH',''),
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},check=True)
        report=load(report_path);history.append({'token_report_sha256':sha(report_path),'report':report})
        if report['valid']:
            result={'object_id':obj['object_id'],'original_prompts':obj['detector_text_prompts'],
                'effective_prompts':[row['effective_prompt'] for row in report['prompts']],'truncated':False,'shortening_history':history}
            (directory/'contract.json').write_text(json.dumps(result,indent=2,allow_nan=False))
            return result
        if attempt==2:raise ValueError('Astra detector phrases still exceed real OWLv2 tokenizer budget after two shortening judgments')
        judgment=shorten_detector_prompts({'object_id':obj['object_id'],'description':obj['description'],
            'original_prompts':obj['detector_text_prompts'],'current_prompts':prompts,'token_budget':report},images)
        path=directory/f'shortening_{attempt}.json';path.write_text(json.dumps(judgment,indent=2,allow_nan=False))
        history[-1]['shortening_judgment']={'sha256':sha(path),'value':judgment}
        prompts=judgment['prompts']


def owned_surface_arguments(root, case, object_id, track_id, output, *, anchor=None):
    """Depth-order masks already excluded front hands; never subtract twice."""
    root=Path(root)
    owned=root/'measurements/object_tracks/depth_order_owned'/track_id
    argv=['--case',case,'--track-id',track_id,'--object-id',object_id,
        '--raw-frame-manifest',root/'input/raw_frame_manifest/manifest.json',
        '--sam2-track-json',owned/'sam2_track.json',
        '--metric-eligibility-manifest',owned/'native_depth_metric_eligibility.json',
        '--depth-npz',root/'state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz',
        '--output-dir',output,'--base-annotations',root/'state/base_annotations/annotations_v19_base.json',
        '--calibration-contract',root/'state/calibration/v19_camera_calibration_contract.json',
        '--object-plan',root/'measurements/object_candidates/object_plan_agent.json',
        '--preserve-source-index','--no-exclude-hand-bboxes']
    argv+=['--propose-anchor-candidates-only'] if anchor is None else ['--anchor-frame',anchor,'--require-anchor-frame']
    return list(map(str,argv))


def rgb_source_worker_arguments(*, annotations, manifest, depth, object_id, frame_ids,
                                adapter, assets_manifest, lightglue_root, output_npz, output_json):
    """The cold worker consumes observations only; no source/refined pose input."""
    return list(map(str, ['--observation-mode', 'source-world', '--annotations', annotations,
        '--raw-frame-manifest', manifest, '--depth-npz', depth, '--p09-adapter-report', adapter,
        '--object-id', object_id, '--frame-start', min(frame_ids), '--frame-end', max(frame_ids),
        '--keyframe-seconds', .5, '--source-gap-seconds', 2., '--keyframe-neighbor-span', 4,
        '--lightglue-root', lightglue_root, '--assets-manifest', assets_manifest,
        '--device', 'cuda:0', '--output-npz', output_npz, '--output-json', output_json]))


def annotate(args):
    """Cold single-object video annotation; every observation is produced in this run."""
    import importlib, os, shutil, uuid, re
    from types import SimpleNamespace
    import cv2
    from utils.data import read_cfr_video_timing
    from utils.state import RunState
    from utils.runtime import run_model
    from observations.camera import calibration
    from observations.common import build_base_annotations, base_parse_args
    from observations.object import surface, shape
    from envs.qualification.camera import load_contract, plane_intrinsics
    from envs.qualification.object_geometry import normalize_completion
    from observations.manifest import write_source_observation_manifest
    from utils.config import preflight_providers
    from utils.state import StageRecovery, planned_stage_contract
    from utils.runtime import stage_environment, astra_stage_inputs
    from utils.config import recovery_asset_manifest
    assets=load(args.assets_manifest)
    hand_backend=assets.get('hand_backend','hawor')
    if hand_backend not in {'hawor','wilor_only'}:
        raise ValueError('unsupported hand backend')
    if hand_backend=='wilor_only' and any(key in assets for key in ('hawor_root','hawor_python')):
        raise ValueError('WiLoR-only asset contract must exclude HaWoR providers')
    shared_gpu=assets.get('scene_gpu_policy',{}).get('mode')=='resource_aware_shared'
    if assets.get('agent_model')!='gpt-6-astra' or assets.get('agent_provider')!='dexgem-responses' or assets.get('agent_disable_fallbacks') is not True:
        raise ValueError('asset manifest must pin Astra without fallbacks')
    os.environ['EGO_ASTRA_CREDENTIAL_FILE']=assets['agent_credential_file']
    code_root=Path(__file__).resolve().parent
    code_files=[code_root/'main.py']
    for directory in ('utils','observations','actions','envs','data'):
        code_files.extend((code_root/directory).rglob('*.py'))
    code_hashes={str(p.relative_to(code_root)):sha(p) for p in code_files}
    root=args.run_root.resolve()
    recovering=getattr(args,'command',None)=='resume'
    code_manifest=root/'input/code_manifest.json'
    raw=root/'input/raw.mp4'
    if recovering:
        contract=load(root/'input/runtime_input_contract.json')
        if contract['run_root']!=str(root) or sha(raw)!=contract['input_video_sha256']:
            raise ValueError('run/raw identity changed')
        if load(code_manifest)!={'files':code_hashes} or sha(code_manifest)!=contract['bundle_manifest_sha256']:
            raise ValueError('source code changed; cross-version recovery refused')
        if sha(args.assets_manifest)!=contract.get('assets_manifest_sha256'):
            raise ValueError('model configuration changed; cross-config recovery refused')
        sensor=Path(contract['sensor_calibration_metadata']) if contract['sensor_calibration_metadata'] else None
        if (sha(sensor) if sensor else None)!=contract['sensor_calibration_sha256']:
            raise ValueError('sensor identity changed')
        if sha(root/'input/model_asset_manifest.json')!=contract.get('model_asset_manifest_sha256') or recovery_asset_manifest(assets)!=load(root/'input/model_asset_manifest.json'):
            raise ValueError('declared model asset bytes changed; asset reuse refused')
        args.case=contract['case'];args.target_description=contract['target_description']
        bundle_version=int(contract.get('observation_bundle_version',1))
        timing=read_cfr_video_timing(raw)
        if timing.report()!=contract['source_timing']:
            raise ValueError('raw timeline changed')
    else:
        preflight_providers(assets)
        asset_identity=recovery_asset_manifest(assets)
        root.mkdir(parents=True,exist_ok=False)
        (root/'input').mkdir();(root/'logs').mkdir()
        with args.video.open('rb') as source,raw.open('xb') as destination:shutil.copyfileobj(source,destination)
        if sha(raw)!=sha(args.video):raise ValueError('raw copy identity mismatch')
        timing=read_cfr_video_timing(raw)
        code_manifest.write_text(json.dumps({'files':code_hashes},sort_keys=True))
        (root/'input/model_asset_manifest.json').write_text(json.dumps(asset_identity,sort_keys=True))
        sensor=None
        if args.sensor_calibration is not None:
            sensor=root/'input/sensor_calibration.json';shutil.copyfile(args.sensor_calibration,sensor)
        bundle_version=int(getattr(args,'observation_bundle_version',2))
        contract={'observation_bundle_version':bundle_version,'schema':'rgb_scene_input_v1','run_id':str(uuid.uuid4()),'case':args.case,
            'run_root':str(root),'input_video':str(raw),'input_video_sha256':sha(raw),
            'source_video_path':str(Path(args.video).resolve()),
            'source_sensor_calibration_path':str(Path(args.sensor_calibration).resolve()) if args.sensor_calibration is not None else None,
            'bundle_manifest_sha256':sha(code_manifest),'sensor_calibration_metadata':str(sensor) if sensor else None,
            'sensor_calibration_sha256':sha(sensor) if sensor else None,'source_timing':timing.report(),
            'target_description':args.target_description,'assets_manifest_sha256':sha(args.assets_manifest),
            'model_asset_manifest_sha256':sha(root/'input/model_asset_manifest.json')}
        (root/'input/runtime_input_contract.json').write_text(json.dumps(contract,indent=2))
    ledger=RunState(root);ids=list(range(len(timing.presentation_ticks)));object_id=None
    recovery=StageRecovery(ledger,{'contract_sha256':sha(root/'input/runtime_input_contract.json'),
        'code_sha256':sha(code_manifest),'assets_manifest_sha256':sha(args.assets_manifest),
        'agent_model':'gpt-6-astra','run_id':contract['run_id']},resume=recovering,
        rewind=getattr(args,'rewind',None),reason=getattr(args,'reason',None),candidate=getattr(args,'candidate',None))
    try:
        import sys
        process_execution={'executable':str(Path(sys.executable).absolute()),
            'resolved_executable':str(Path(sys.executable).resolve()), 'executable_sha256':sha(sys.executable),
            'cwd':Path.cwd(), 'environment':stage_environment(os.environ)}
        def stage(name, *, inputs, parameters):
            def execute(function):
                verify_code()
                planned=planned_stage_contract('main.annotate.'+name,'inprocess',
                    inputs={'object_id':object_id,'frame_ids':ids,'values':inputs},
                    parameters=parameters,execution=process_execution)
                return recovery.run(name,function,contract=planned)
            return execute
        environment={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu),'HF_HUB_OFFLINE':'1',
            'TRANSFORMERS_OFFLINE':'1','EGO_SAM2_SOURCE':assets['sam2_source'],
            'PYTHONPATH':str(code_root)+os.pathsep+os.environ.get('PYTHONPATH','')}
        def verify_code():
            if sha(args.assets_manifest)!=contract['assets_manifest_sha256']:
                raise RuntimeError('model configuration changed during the run')
            if any(sha(code_root/name)!=digest for name,digest in code_hashes.items()):
                raise RuntimeError('source code changed during the run; deploy an immutable snapshot')
        def record(stage,outputs,command=None):
            verify_code()
            return ledger.record(stage,outputs,object_id=object_id,frame_ids=ids,command=command)
        gpu_uuid=args.gpu_uuid
        def worker(stage_name,python,module,argv,outputs,overlay=None):
            nonlocal gpu_uuid
            verify_code()
            command=[str(python),'-m',module,*map(str,argv)]
            worker_environment={**environment,**(overlay or {})}
            executable=Path(shutil.which(str(python),path=worker_environment.get('PATH')) or python).absolute()
            min_free=int(assets.get('sam3d_min_free_mib',12000))
            selected=None
            if shared_gpu:
                from utils.gpu_sharing import select
                min_free=int(assets['scene_gpu_policy']['stage_min_free_mib'].get(stage_name,8000))
                resource_path=root/'logs'/f'{stage_name}.resources.json'
                if recovering and resource_path.is_file():
                    selected=load(resource_path)
                else:
                    selected=select(min_free,assets['scene_gpu_policy'].get('max_utilization',25))
                worker_environment['CUDA_VISIBLE_DEVICES']=selected['uuid']
            planned=planned_stage_contract(module,'subprocess',
                parameters={'command':command,'outputs':outputs,'object_id':object_id,'frame_ids':ids},
                inputs={'argv':command[3:]},execution={'executable':executable,
                    'resolved_executable':executable.resolve(),'executable_sha256':sha(executable),
                    'argv':command[1:],'cwd':code_root,
                    'environment':stage_environment(worker_environment,(overlay or {}).keys()),
                    'gpu':selected['index'] if shared_gpu else args.gpu,'expected_uuid':selected['uuid'] if shared_gpu else gpu_uuid,
                    'gpu_policy':'resource_aware_shared' if shared_gpu else 'exclusive',
                    'min_free_mib':min_free})
            def execute():
                verify_code()
                (root/'logs').mkdir(parents=True,exist_ok=True)
                with (root/'logs'/f'{stage_name}.log').open('x') as log:
                    if shared_gpu:
                        from utils.gpu_sharing import run_shared
                        resources=run_shared(command,cwd=code_root,environment=worker_environment,selected=selected,
                            min_free_mib=min_free,max_utilization=assets['scene_gpu_policy'].get('max_utilization',25),stdout=log)
                    else:
                        resources=run_model(command,cwd=code_root,environment=worker_environment,
                            gpu=args.gpu,expected_uuid=gpu_uuid,min_free_mib=min_free,stdout=log)
                (root/'logs'/f'{stage_name}.resources.json').write_text(json.dumps(resources,indent=2))
                record(stage_name,outputs,command)
                return resources
            resources=recovery.run(stage_name,execute,contract=planned)
            gpu_uuid=resources['uuid']
        def ask(name,evidence,images,path):
            verify_code()
            module=importlib.import_module('observations.judgments.judge_'+name)
            request_evidence={'frame_ids':ids,**({'object_id':object_id} if object_id else {}),**evidence}
            planned=planned_stage_contract('observations.judgments.judge_'+name,'astra',
                parameters={'output':path,'role':'judgment'},
                inputs=astra_stage_inputs(module.PROMPT,module.SCHEMA,request_evidence,images,
                    **({'max_request_bytes':module.MAX_REQUEST_BYTES,'adaptive_review_images':True}
                       if name=='pose_trajectory' or (name=='mask_track' and 'mask_review_batch' in evidence) else {})),
                execution=process_execution)
            def execute():
                verify_code()
                if path.exists():raise ValueError('judgment output already exists')
                value=module.judge(request_evidence,images, audit_dir=path.with_suffix('.request')) if name=='pose_trajectory' or (name=='mask_track' and 'mask_review_batch' in evidence) else module.judge(request_evidence,images)
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False))
                ledger.record('judge_'+name,[(path,'judgment')],object_id=object_id,frame_ids=evidence.get('frame_ids',ids))
                return value
            return recovery.run('judge_'+name+':'+str(path.relative_to(root)),execute,contract=planned)
        manifest=root/'input/raw_frame_manifest/manifest.json'
        @stage('raw_frames', inputs={'video':raw},
            parameters={'output_dir':manifest.parent,'render_width':960,'frame_start':None,'frame_end':None,'jpeg_quality':95})
        def _stage_raw_frames():
            calibration.build(SimpleNamespace(video=raw,output_dir=manifest.parent,render_width=960,
                frame_start=None,frame_end=None,jpeg_quality=95))
            record('raw_frames',[(manifest,'raw_manifest')])
        frames=load(manifest)['frames'];raw_by_id=indexed(frames)
        sampled=[ids[i] for i in np.unique(np.linspace(0,len(ids)-1,min(12,len(ids)),dtype=int))]
        images=[Path(raw_by_id[i]['raw_frame_path']) for i in sampled]
        plan_path=root/'measurements/object_candidates/object_plan_agent.json'
        plan=ask('object',{'case':args.case,'target_description':args.target_description},images,plan_path)
        obj=plan['objects'][0];object_id=obj['object_id'];track=obj['track_id']
        if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',v) for v in (args.case,object_id,track)):
            raise ValueError('unsafe case/object/track identity')
        from observations.judgments import judge_object as shorten
        @stage('detector_text', inputs={'object':obj,'images':astra_stage_inputs(shorten.PROMPT,shorten.SCHEMA,obj,images),'shorten_prompt':shorten.SHORTEN_PROMPT,'shorten_schema':shorten.SHORTEN_SCHEMA,'assets':assets['owlv2_model']},
            parameters={'directory':root/'state/detector_text','python':assets['owlv2_python'],'module':'observations.object.owlv2','mode':'text-contract','max_attempts':3,'cwd':code_root,'offline':True})
        def _stage_detector_text():
            detector_contract=detector_prompt_contract(obj,images,assets,root/'state/detector_text')
            record('detector_text',[(root/'state/detector_text/contract.json','detector_text_contract')])
            return detector_contract
        detector_contract = _stage_detector_text
        width,height=timing.source_size_wh
        camera_dir=root/'state/calibration';camera=camera_dir/'v19_camera_calibration_contract.json'
        def depth_job(stage,output,conditioned=False):
            argv=['--manifest',manifest,'--output-dir',output,'--frame-start',0,'--frame-end',ids[-1],
                '--unidepth-repo',assets['unidepth_repo'],'--model-id',assets['unidepth_model'],
                '--source-width',width,'--source-height',height]
            if conditioned:argv+=['--camera-contract',camera,'--camera-input-plane','manifest_rgb','--camera-output-plane','source_rgb']
            path=output/'unidepth_full_frame_depth_v3.npz'
            worker(stage,assets['unidepth_python'],'observations.camera.unidepth',argv,[(path,'depth')])
            return path
        if sensor:
            extra=['--sensor-calibration-contract',sensor,'--sensor-calibration-authority','prediction_side_sensor_metadata',
                '--prediction-source-video',raw,'--sensor-source-video',raw]
        else:
            probe=depth_job('intrinsics_probe',root/'measurements/depth_slam/unidepth_intrinsics_probe')
            extra=['--unidepth-npz',probe,'--aggregation','median','--square-focal']
        @stage('camera', inputs={'manifest':manifest,'sensor_or_probe':extra},
            parameters={'case':args.case,'output_dir':camera_dir})
        def _stage_camera():
            calibration.resolve_camera(calibration.camera_parse_args(list(map(str,['--case',args.case,
                '--raw-frame-manifest',manifest,'--output-dir',camera_dir,*extra]))))
            record('camera',[(camera,'calibration')])
        camera_payload,camera_normal=load_contract(camera,expected_frame_ids=ids)
        K,_=plane_intrinsics(camera_payload,camera_normal,plane_name='source_rgb',actual_size_wh=(width,height))
        if not np.isclose(K[0],K[1],atol=1e-6,rtol=1e-6):raise ValueError('HaWoR requires square focal; sensor K cannot be averaged')
        depth=depth_job('conditioned_depth',root/'measurements/depth_slam/unidepth_full_frame',True)
        bound=camera_dir/'depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz'
        @stage('depth_binding', inputs={'depth':depth,'camera':camera},
            parameters={'depth_plane':'source_rgb','output_dir':bound.parent})
        def _stage_depth_binding():
            calibration.bind_depth_contract(calibration.depth_parse_args(list(map(str,['--source-depth-npz',depth,
                '--camera-contract',camera,'--depth-plane','source_rgb','--output-dir',bound.parent]))))
            record('depth_binding',[(bound,'depth')])
        independent_camera=None
        if hand_backend=='hawor':
            hawor=root/'measurements/hand_candidates/hawor_world/hawor_world_hands.npz'
            worker('hawor',assets['hawor_python'],'observations.hand.hawor',
                ['--hawor-root',assets['hawor_root'],'--run-root',root,'--video_path',raw,
                 '--camera-intrinsics',*K,'--output-dir',hawor.parent],[(hawor,'hands')])
        else:
            local_dir=root/'measurements/hand_candidates/wilor_camera'
            local_wilor=local_dir/'wilor_hands.npz'
            worker('wilor_camera',assets['wilor_prediction_python'],'observations.hand.wilor_calibrated',
                ['--clip',raw,'--camera-contract',camera,'--wilor-root',assets['wilor_prediction_root'],
                 '--output',local_dir,'--compute-host',assets['compute_host'],'--batch-size',2],
                [(local_wilor,'camera_local_wilor'),(local_wilor.with_suffix('.completed.json'),'wilor_producer')])
            perspective_dir=root/'measurements/hand_candidates/wilor_perspective'
            local_perspective=perspective_dir/'wilor_hands.npz'
            worker('wilor_perspective',assets['main_python'],'observations.hand.wilor_perspective',
                ['--source',local_wilor,'--output',perspective_dir],
                [(local_perspective,'camera_local_wilor'),(local_perspective.with_suffix('.completed.json'),'wilor_producer')])
            hawor=root/'measurements/hand_candidates/wilor_world/wilor_world_hands.npz'
        detector=root/'measurements/object_candidates/object_box_prompts_owlv2'
        owlargs=['--raw-frame-manifest',manifest,'--output-root',detector,'--review-dir',root/'renders/owlv2',
            '--case-id',args.case,'--object-id',object_id,'--track-id',track,'--description',obj['description'],
            '--prompt-frames',','.join(map(str,obj['detector_prompt_frames'])),
            '--active-start',obj['detector_active_interval'][0],'--active-end',obj['detector_active_interval'][1],
            '--owlv2-model',assets['owlv2_model'],'--box-threshold',.03,'--device','cuda','--box-only']
        for prompt in detector_contract['effective_prompts']:owlargs+=['--text-prompt',prompt]
        worker('owlv2',assets['owlv2_python'],'observations.object.owlv2',owlargs,
            [(detector/object_id/'object_point_prompts_vlm.json','object_boxes'),
             (detector/object_id/'candidate_identity.json','candidate_identity')])
        sam2=root/'measurements/object_tracks/sam2_owlv2_box_points';track_path=sam2/track/'sam2/sam2_track.json'
        worker('sam2',assets['unidepth_python'],'observations.object.sam2',
            ['--clip',raw,'--point-root',detector,'--output-root',sam2,'--checkpoint',assets['sam2_checkpoint'],
             '--frame-start',0,'--frame-end',ids[-1]],[(track_path,'object_masks')])
        @stage('candidate_mask_identity', inputs={'candidates':detector/object_id/'candidate_identity.json','track':track_path,'manifest':manifest},
            parameters={'output':root/'state/candidate_mask_identity.json'})
        def _stage_candidate_mask_identity():
            from observations.object.owlv2 import check_candidate_mask_identity
            from utils.state import sha
            candidate_report=load(detector/object_id/'candidate_identity.json')
            if (candidate_report['case_id'],candidate_report['object_id'],candidate_report['track_id'])!=(args.case,object_id,track):
                raise ValueError('candidate identity differs from Astra selected object plan')
            if candidate_report['raw_frame_manifest_sha256']!=sha(manifest):raise ValueError('candidate raw manifest changed')
            for row in candidate_report['detections']:
                raw_row=raw_by_id[row['frame_idx']]
                if Path(row['image_path']).resolve()!=Path(raw_row['raw_frame_path']).resolve() or row['image_sha256']!=sha(raw_row['raw_frame_path']):
                    raise ValueError('candidate frame not bound to current run raw bytes')
            value=check_candidate_mask_identity(candidate_report,load(track_path),
                [frames[0]['source_width'],frames[0]['source_height']])
            value['inputs']={'candidates_sha256':sha(detector/object_id/'candidate_identity.json'),'sam2_track_sha256':sha(track_path),'manifest_sha256':sha(manifest)}
            output=root/'state/candidate_mask_identity.json';output.write_text(json.dumps(value,indent=2))
            if value['status']!='ok':raise ValueError('mask-track candidate identity check failed; see candidate_mask_identity.json')
            record('candidate_mask_identity',[(output,'candidate_mask_identity')])
        @stage('mask_review', inputs={'track':track_path,'raw_manifest':manifest,'sampled':sampled,
            'candidate_checks':root/'state/candidate_mask_identity.json'},
            parameters={'output_dir':root/'renders/mask_review','uncertain_context_neighbors':1,'batch_frames':6,'panel_max_width':480})
        def _stage_mask_review():
            from observations.judgments.judge_mask_track import render_uncertain_reviews
            track_payload=load(track_path);mask_reviews=[]
            batches=render_uncertain_reviews(load(root/'state/candidate_mask_identity.json'),
                track_payload,raw_by_id,ids,sampled,root/'renders/mask_review')
            if batches:
                output=root/'state/mask_review_evidence.json'
                output.write_text(json.dumps({'schema':'mask_identity_visual_review_v1','batches':batches},indent=2,allow_nan=False))
                record('mask_review',[(output,'mask_identity_visual_evidence')])
                return track_payload,mask_reviews,batches
            for frame in sampled:
                raw_image=cv2.imread(raw_by_id[frame]['raw_frame_path'])
                row=track_payload.get(str(frame),{})
                mask_path=row.get('mask_path')
                if mask_path:
                    mask=cv2.imread(str(mask_path),0)
                    if mask is None:raise ValueError('SAM2 declared mask image is missing')
                    selected_mask=cv2.resize(mask,(raw_image.shape[1],raw_image.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                    raw_image[selected_mask]=(raw_image[selected_mask]*.5+np.array([50,200,50])*.5).astype('uint8')
                path=root/'renders/mask_review'/f'{frame:06d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(path),raw_image):raise RuntimeError('mask evidence render failed')
                mask_reviews.append(path)
            return track_payload, mask_reviews, []
        track_payload, mask_reviews, mask_batches = _stage_mask_review
        if mask_batches:
            batch_paths=[root/'state/mask_track_batches'/f'{index:04d}.json' for index in range(len(mask_batches))]
            results=[ask('mask_track',{'selected_object_plan':obj,**batch},
                [Path(image['path']) for image in batch['review_images']],path)
                for batch,path in zip(mask_batches,batch_paths)]
            @stage('mask_review_merge', inputs={'results':batch_paths,'evidence':root/'state/mask_review_evidence.json'},
                parameters={'output':root/'state/mask_track_judgment.json','identity_rule':'all_batches_true'})
            def _stage_mask_review_merge():
                from observations.judgments.judge_mask_track import merge_batch_judgments
                value=merge_batch_judgments(results,mask_batches)
                output=root/'state/mask_track_judgment.json'
                output.write_text(json.dumps(value,indent=2,allow_nan=False))
                record('judge_mask_track',[(output,'judgment')])
                return value
            mask_judgment=_stage_mask_review_merge
        else:
            mask_judgment=ask('mask_track',{'track':track_payload,'selected_object_plan':obj,'candidate_identity_checks':load(root/'state/candidate_mask_identity.json'),'inspected_frame_ids':sampled},mask_reviews,root/'state/mask_track_judgment.json')
        if mask_judgment['identity_consistent'] is False:
            raise ValueError('mask identity review rejected the selected-object track; no scene is published')
        if hand_backend=='wilor_only':
            independent_camera=root/'measurements/camera/droid_independent/droid_camera.npz'
            worker('independent_droid',assets['droid_python'],'observations.camera.droid_independent',
                ['--manifest',manifest,'--depth',bound,'--wilor',local_perspective,'--object-track',track_path,
                 '--droid-root',assets['droid_root'],'--weights',assets['droid_weights'],'--output',independent_camera.parent],
                [(independent_camera,'independent_camera'),(independent_camera.with_suffix('.json'),'independent_camera_provenance')])
            worker('wilor_world',assets['main_python'],'observations.hand.wilor_world',
                ['--wilor',local_perspective,'--camera',independent_camera,'--video',raw,
                 '--mano-models',assets['mano_models'],'--output',hawor.parent],
                [(hawor,'hands'),(hawor.parent/'mano_contract.json','independent_mano_contract')])
        base=root/'state/base_annotations/annotations_v19_base.json'
        @stage('base', inputs={'manifest':manifest,'depth':bound,'camera':camera,'hawor':hawor,'object_plan':plan_path,'sam2':sam2},
            parameters={'case':args.case,'output_dir':base.parent})
        def _stage_base():
            build_base_annotations(base_parse_args(list(map(str,['--case',args.case,'--raw-frame-manifest',manifest,
                '--output-dir',base.parent,'--depth-npz',bound,'--calibration-contract',camera,
                '--hawor-npz',hawor,'--object-plan',plan_path,'--sam2-output-root',sam2]))))
            bridge=base.parent/('mano_bridge_wilor_world.npz' if hand_backend=='wilor_only' else 'v19_mano_bridge_from_hawor_world.npz')
            record('base',[(base,'base_observations'),(bridge,'mano_bridge')])
            return bridge
        bridge = _stage_base
        owned=root/'measurements/object_tracks/depth_order_owned'/track
        @stage('ownership', inputs={'base':base,'track':track_path,'depth':bound,'camera':camera,'bridge':bridge,'hawor':hawor},
            parameters={'case':args.case,'object_id':object_id,'track_id':track,'output_dir':owned})
        def _stage_ownership():
            surface.build_depth_order(surface.depth_order_parse_args(list(map(str,['--base-annotations',base,
                '--raw-sam2-track',track_path,'--depth-npz',bound,'--calibration-contract',camera,
                '--mano-bridge',bridge,'--hawor-npz',hawor,'--output-dir',owned,'--case',args.case,
                '--object-id',object_id,'--track-id',track]))))
            record('ownership',[(owned/'sam2_track.json','owned_masks'),(owned/'native_depth_metric_eligibility.json','metric_eligibility')])
        proposals=root/'measurements/object_geometry/anchor_candidates'/object_id
        @stage('anchor_candidates', inputs={'argv':owned_surface_arguments(root,args.case,object_id,track,proposals)},
            parameters={'propose_anchor_candidates_only':True})
        def _stage_anchor_candidates():
            surface.build_visible_geometry(surface.surface_parse_args(owned_surface_arguments(root,args.case,object_id,track,proposals)))
        proposal_path=proposals/'anchor_candidate_proposals.json'
        decision=ask('pose_anchor',{'frame_ids':[r['frame_idx'] for r in load(proposal_path)['review_candidates']],'candidates':load(proposal_path),'candidate_report_path':str(proposal_path)},
            [proposals/'anchor_candidate_review.jpg'],root/'state/anchor_decisions'/f'{object_id}.json')
        anchor=decision['selected_anchor_frame_idx']
        visible=root/'measurements/object_geometry/visible_geometry'/object_id
        @stage('surface', inputs={'argv':owned_surface_arguments(root,args.case,object_id,track,visible,anchor=anchor)},
            parameters={'anchor':anchor,'require_anchor_frame':True})
        def _stage_surface():
            surface.build_visible_geometry(surface.surface_parse_args(owned_surface_arguments(root,args.case,object_id,track,visible,anchor=anchor)))
            ann=visible/'annotations_v19_visible_geometry.json'
            record('surface',[(ann,'observations'),(visible/'v19_visible_geometry_adapter_report.json','surface_report')])
            return ann
        ann = _stage_surface
        rgb_npz = root/'measurements/object_motion'/object_id/'rgb_source.npz'
        rgb_report = rgb_npz.with_suffix('.json')
        worker('rgb_source_measurements', assets['rgb_python'], 'observations.object.rgb_tracking',
            rgb_source_worker_arguments(annotations=ann, manifest=manifest, depth=bound,
                object_id=object_id, frame_ids=ids,
                adapter=visible/'v19_visible_geometry_adapter_report.json',
                assets_manifest=args.assets_manifest, lightglue_root=assets['lightglue_root'],
                output_npz=rgb_npz, output_json=rgb_report),
            [(rgb_npz, 'rgb_source_arrays'), (rgb_report, 'rgb_source_report')])
        @stage('rigidity_evidence', inputs={'root':root,'object_id':object_id,'plan':plan_path,'manifest':manifest,'track':track_path,'owned':owned/'sam2_track.json'},
            parameters={'output_dir':root/'state/rigidity_evidence'})
        def _stage_rigidity_evidence():
            return surface.build_rigidity_motion_evidence(root,object_id,plan_path,manifest,track_path,
                owned/'sam2_track.json',root/'state/rigidity_evidence')
        rigidity_evidence,rigidity_images=_stage_rigidity_evidence
        rigidity=ask('rigidity',rigidity_evidence,rigidity_images,root/'state/rigidity.json')
        from envs.qualification.object_geometry import require_rigid_tracked_whole
        require_rigid_tracked_whole(rigidity)
        @stage('metric_inputs', inputs={'annotations':ann,'visible':visible,'decision':decision},
            parameters={'anchor':anchor,'case':args.case,'object_id':object_id})
        def _stage_metric_inputs():
            evidence_root=root/'measurements/geometry_completion/rigid_evidence'
            surface.build_anchor_evidence(list(map(str,['--case',args.case,'--object-id',object_id,'--annotations',ann,
                '--depth-fused-report',visible/'v19_visible_geometry_depth_fused_report.json',
                '--output-root',evidence_root,'--selected-frame-idx',anchor,'--selection-note',decision['visual_rationale']])))
            evidence=evidence_root/args.case/object_id/'evidence_bundle/evidence_bundle_report.json'
            geometry=root/'measurements/geometry_completion'/('sam3d_'+object_id)
            shape.build_completion_inputs(shape.conditioning_parse_args(list(map(str,['--evidence-report',evidence,'--output-dir',geometry/'native_inputs']))))
            native_input=geometry/'native_inputs/p11_dual_geometry_inputs_report.json'
            conditioning=load(native_input)['conditioning_contracts']['sam3d_objects_native']
            return evidence, geometry, native_input, conditioning
        evidence, geometry, native_input, conditioning = _stage_metric_inputs
        def native_job(stage,image,mask,out,frame):
            name=f'{args.case}_{object_id}_frame_{frame}'
            argv=['--repo',assets['sam3d_repo'],'--config',assets['sam3d_config'],
                '--moge-checkpoint',assets['sam3d_moge_checkpoint'],'--dinov2-repo',assets['sam3d_dinov2_repo'],
                '--dinov2-checkpoint',assets['sam3d_dinov2_checkpoint'],'--case',f'{name}|{image}|{mask}|42','--output-dir',out]
            qc=out/name/'qc_sam3d_objects_mesh_v7.json'
            worker(stage,assets['sam3d_python'],'observations.object.sam3d',argv,[(qc,'native_geometry')],assets.get('sam3d_environment',{}))
            return name,qc,argv
        name,qc,command=native_job('metric_native',conditioning['image'],conditioning['mask'],geometry/'native_reconstruction',anchor)
        @stage('metric_geometry', inputs={'qc':qc,'name':name,'conditioning':conditioning,'command':command,'native_input':native_input,'evidence':evidence,'annotations':ann},
            parameters={'geometry':geometry,'observed_band_scale':float(np.sqrt(3.)),'silhouette_free_space_filter':True,'planar_slab_support_filter':True})
        def _stage_metric_geometry():
            candidate=shape.sam3d_candidate_record(qc,name,Path(conditioning['image']),Path(conditioning['mask']),{'mode':'fresh_native_SAM3D_only','command':list(map(str,command))})
            prior=geometry/'sam3d_prior_report.json';prior.write_text(json.dumps({'schema':'v19_sam3d_only_prior_v1','status':'ok',
                'candidates':{'sam3d_objects':candidate},'gt_consumed':False,'native_input_report':str(native_input)},indent=2))
            shape.bind_sam3d_metric(shape.native_bridge_parse_args(list(map(str,['--evidence-report',evidence,'--p12-report',prior,'--output-dir',geometry/'metric_bridge']))))
            metric_bridge=geometry/'metric_bridge/sam3d_native_sensor_metric_canonical_bridge_report.json'
            canonical=Path(load(metric_bridge)['outputs']['metric_canonical_render_prior'])
            shape.build_completion_face_support(list(map(str,['--evidence-report',evidence,'--trellis-report',geometry/'metric_bridge/p13_metric_canonical_input.json',
                '--input-mesh-already-metric-canonical','--output-dir',geometry/'face_support',
                '--observed-band-scale',np.sqrt(3.),'--silhouette-free-space-filter','--planar-slab-support-filter'])))
            completion=load(geometry/'face_support/v18_compact_rigid_trellis_completion_report.json')
            completion.update(schema='v19_sam3d_only_completion_v1',source_model='sam3d_objects',gt_consumed=False)
            completion['render_mesh_contract']={'required_mesh_path':str(canonical),'required_mesh_sha256':sha(canonical),'renderer_must_match':True}
            completion['outputs']['intact_generated_render_mesh']=str(canonical)
            completion=normalize_completion(completion,load(ann),object_id)
            completion_path=geometry/'completion_report.json';completion_path.write_text(json.dumps(completion,indent=2))
            return completion_path
        completion_path = _stage_metric_geometry
        appearance= root/'measurements/demo_visual'/object_id/'appearance_evidence/appearance_mask_manifest.json'
        @stage('appearance_inputs', inputs={'manifest':manifest,'annotations':ann,'track':track_path,'ownership_report':owned/'depth_order_ownership_report.json'},
            parameters={'root':root,'object_id':object_id,'case':args.case,'track_id':track,'output_dir':appearance.parent})
        def _stage_appearance_inputs():
            shape.build_appearance(shape.appearance_order_parse_args(list(map(str,['--run-root',root,'--object-id',object_id,
                '--case-id',args.case,'--track-id',track,'--raw-frame-manifest',manifest,'--annotations',ann,
                '--raw-sam2-track',track_path,'--depth-order-report',owned/'depth_order_ownership_report.json','--output-dir',appearance.parent]))))
        appearance_rows=indexed(load(appearance)['frames'])
        choice_path=root/'state/appearance_decisions'/f'{object_id}.json'
        preferred=load(appearance.with_name('anchor_decision.json'))['selected_frame_idx']
        appearance_candidates=appearance_candidate_ids(appearance_rows,load(ann),object_id,preferred)
        if not appearance_candidates:raise ValueError('no inspected available appearance anchor')
        appearance_images=[]
        for frame in appearance_candidates:
            appearance_images.extend([Path(appearance_rows[frame]['raw_rgb_path']),Path(appearance_rows[frame]['appearance_mask_path'])])
        choice=ask('appearance_anchor',{'frame_ids':appearance_candidates,'preferred_frame_idx':preferred,'appearance':load(appearance)},appearance_images,choice_path)
        selected=choice['selected_frame_idx'];selected_row=appearance_rows[selected]
        if selected not in appearance_candidate_ids(appearance_rows,load(ann),object_id,preferred):raise ValueError('selected appearance anchor lacks same-frame metric alignment')
        visual_report_path = None
        if bundle_version == 2:
            from observations.object.visual_geometry import can_reuse_native, build_visual_geometry
            @stage('appearance_native_choice', inputs={'metric_qc':qc,'decision':choice_path,
                'appearance':appearance,'metric_anchor':anchor},
                parameters={'seed':42,'same_frame_and_same_bytes_required':True})
            def _stage_appearance_native_choice():
                reuse=can_reuse_native(qc,selected_row['raw_rgb_path'],selected_row['appearance_mask_path'],
                    selected_frame=selected,native_frame=anchor,seed=42)
                receipt=root/'state/appearance_native_reuse.json'
                value={'reuse_same_run_native':reuse,'source_qc':str(qc),'source_qc_sha256':sha(qc),
                    'selected_frame_idx':selected,'metric_anchor_frame_idx':anchor,
                    'image_sha256':sha(selected_row['raw_rgb_path']),
                    'mask_sha256':sha(selected_row['appearance_mask_path']),
                    'scope':'same_verified_stage_prefix_not_foreign_cache_import',
                    'new_model_execution_claimed':False}
                receipt.write_text(json.dumps(value,indent=2))
                record('appearance_native_choice',[(receipt,'native_reuse_decision')])
                return value
            if _stage_appearance_native_choice['reuse_same_run_native']:
                appearance_qc=qc
            else:
                _,appearance_qc,_=native_job('appearance_native',selected_row['raw_rgb_path'],
                    selected_row['appearance_mask_path'],root/'measurements/appearance_native'/object_id,selected)
            visual_dir=root/'measurements/visual_geometry_observations'/object_id
            @stage('appearance_geometry', inputs={'annotations':ann,'appearance':appearance,
                'decision':choice_path,'native_qc':appearance_qc},
                parameters={'object_id':object_id,'output_dir':visual_dir,
                    'coordinate_frame':'opencv_camera_at_appearance_anchor','no_image_scale_search':True})
            def _stage_appearance_geometry():
                path=build_visual_geometry(annotations=ann,appearance=appearance,decision=choice_path,
                    native_qc=appearance_qc,output_dir=visual_dir,object_id=object_id,run_root=root)
                report_payload=load(path)
                record('appearance_geometry',[(path,'visual_geometry_report'),
                    (Path(report_payload['output_mesh']['path']),'visual_full_metric_camera_mesh')])
                return path
            visual_report_path=_stage_appearance_geometry
        @stage('depth_review', inputs={'bound':bound,'manifest':manifest},
            parameters={'output_dir':root/'renders/depth','percentiles':[2,98]})
        def _stage_depth_review():
            depth_images={}
            with np.load(bound,allow_pickle=False) as archive:
                depth_data=archive['depth'];depth_ids=archive['frame_idx']
                for position,frame in enumerate(depth_ids):
                    depth_frame=depth_data[position];finite=np.isfinite(depth_frame)&(depth_frame>0)
                    if not finite.any():raise ValueError('conditioned depth frame contains no positive measurements')
                    low,high=np.percentile(depth_frame[finite],[2,98])
                    gray=np.clip((np.nan_to_num(depth_frame,nan=low)-low)/max(high-low,1e-6)*255,0,255).astype('uint8')
                    color=cv2.applyColorMap(gray,cv2.COLORMAP_TURBO);color[~finite]=0
                    color=cv2.resize(color,(frames[0]['manifest_width'],frames[0]['manifest_height']))
                    cv2.putText(color,f'frame {int(frame)}: camera-z {low:.3f} .. {high:.3f} m',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                    path=root/'renders/depth'/f'{int(frame):06d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(path),color):raise RuntimeError('depth review raster write failed')
                    depth_images[int(frame)]=path
            return depth_images
        depth_images = {int(key):value for key,value in _stage_depth_review.items()}
        from observations.interaction.adapter import (assemble_interaction_evidence,
            SCHEMA as INTERACTION_SCHEMA, CONTACT_SUPPORT_UNCERTAINTY_PRIOR_M)
        interaction_batches=[]
        for offset in range(0,len(ids),12):
            batch=ids[offset:offset+12];batch_images=[Path(raw_by_id[i]['raw_frame_path']) for i in batch]
            evidence_batch={'frame_ids':batch}
            destination=root/'state/judgments'/f'batch_{offset:06d}'
            contact=ask('contact',evidence_batch,batch_images,destination/'contact.json')
            pads=ask('contact_pads',{**evidence_batch,'contact':contact},batch_images,destination/'pads.json')
            occlusion=ask('occlusion',evidence_batch,batch_images,destination/'occlusion.json')
            reliability=ask('depth_reliability',evidence_batch,batch_images+[depth_images[i] for i in batch],destination/'depth_reliability.json')
            visibility=ask('visibility',evidence_batch,batch_images,destination/'visibility.json')
            results={'contact':contact,'contact_pads':pads,'occlusion':occlusion,
                'depth_reliability':reliability,'visibility':visibility}
            filenames={'contact':'contact','contact_pads':'pads','occlusion':'occlusion',
                'depth_reliability':'depth_reliability','visibility':'visibility'}
            interaction_batches.append({'frame_ids':batch,'judgments':results,
                'sources':{name:{'path':str(destination/(filename+'.json')),
                    'sha256':sha(destination/(filename+'.json'))} for name,filename in filenames.items()}})
        @stage('interaction_assembly', inputs={'batches':interaction_batches},
            parameters={'output':root/'state/agent_interaction_judgments'/f'{object_id}.json',
                'schema':INTERACTION_SCHEMA,'contact_support_uncertainty_prior_m':CONTACT_SUPPORT_UNCERTAINTY_PRIOR_M})
        def _stage_interaction_assembly():
            interaction=assemble_interaction_evidence(object_id=object_id,frame_ids=ids,
                batches=interaction_batches,support_uncertainty_prior_m=CONTACT_SUPPORT_UNCERTAINTY_PRIOR_M)
            judgments=root/'state/agent_interaction_judgments'/f'{object_id}.json';judgments.parent.mkdir(parents=True,exist_ok=True)
            judgments.write_text(json.dumps(interaction,indent=2,allow_nan=False));record('interaction_assembly',[(judgments,'interaction_judgment')])
            return judgments
        judgments = _stage_interaction_assembly
        factor_root=root/'measurements/contact_visibility_factors'
        factor_path=factor_root/args.case/'v19_visible_contact_ownership_factor_report.json'
        @stage('interaction_factors',inputs={'annotations':ann,'judgments':judgments,'depth':bound},
            parameters={'case':args.case,'target_entity_id':'object:'+object_id.removeprefix('object:'),'output':factor_path})
        def _stage_interaction_factors():
            from observations.interaction.evidence import interaction_build, interaction_parse_args
            factor_args=interaction_parse_args(list(map(str,['--annotations',ann,'--case',args.case,
                '--target-entity-id','object:'+object_id.removeprefix('object:'),'--frame-span',min(ids),max(ids),
                '--output-root',factor_root,'--agent-interaction-judgment',judgments,
                '--require-camera-mano-contract-aligned'])))
            factor=interaction_build(factor_args)
            factor_path.parent.mkdir(parents=True,exist_ok=True)
            factor_path.write_text(json.dumps(factor,indent=2,allow_nan=False))
            record('interaction_factors',[(factor_path,'interaction_factors')])
            return factor_path
        factor_path=_stage_interaction_factors
        bundle_path=root/'source_observations.json'
        completion_payload=load(completion_path)
        completion_outputs=completion_payload.get('outputs') if isinstance(completion_payload.get('outputs'),dict) else {}
        mesh_path=Path(completion_outputs.get('pose_hypothesis_mesh_labeled') or completion_outputs.get('completed_mesh_labeled'))
        collision_mesh_path=Path(completion_outputs.get('collision_eligible_mesh_labeled') or mesh_path)
        if not collision_mesh_path.is_file():
            raise ValueError('completion report lacks a readable collision mesh for the source-observation contract')
        artifacts={
            'rgb_source_npz':(rgb_npz,'rgb_source_arrays'),
            'rgb_source_report':(rgb_report,'rgb_source_report'),
            'runtime_input_contract':(root/'input/runtime_input_contract.json','runtime_input_contract'),
            'code_manifest':(code_manifest,'code_manifest'),
            'model_asset_manifest':(root/'input/model_asset_manifest.json','model_asset_manifest'),
            'raw_video':(raw,'raw_video'),
            'raw_frames':(manifest,'raw_frame_manifest'),
            'camera_calibration':(camera,'camera_calibration'),
            'conditioned_depth':(bound,'conditioned_camera_z_depth'),
            'hawor_world':(hawor,'raw_hawor_world_mano' if hand_backend=='hawor' else 'raw_wilor_independent_world_mano'),
            'mano_bridge':(bridge,'raw_mano_bridge'),
            'object_plan':(plan_path,'object_plan'),
            'candidate_identity':(detector/object_id/'candidate_identity.json','object_candidate_identity'),
            'raw_object_masks':(track_path,'raw_object_masks'),
            'owned_object_masks':(owned/'sam2_track.json','depth_order_owned_object_masks'),
            'metric_eligibility':(owned/'native_depth_metric_eligibility.json','metric_surface_eligibility'),
            'visible_annotations':(ann,'visible_metric_annotations'),
            'surface_report':(visible/'v19_visible_geometry_adapter_report.json','visible_surface_report'),
            'completion_report':(completion_path,'reconstructed_shape_contract'),
            'reconstructed_mesh':(mesh_path,'reconstructed_object_mesh'),
            'collision_mesh':(collision_mesh_path,'collision_object_mesh'),
            'appearance_masks':(appearance,'appearance_mask_observations'),
            'appearance_anchor_decision':(choice_path,'appearance_anchor_decision'),
            'interaction_judgments':(judgments,'contact_visibility_occlusion_evidence'),
            'interaction_factors':(factor_path,'qualified_contact_visibility_factors'),
        }
        if independent_camera is not None:
            artifacts['independent_camera']=(independent_camera,'independent_droid_unidepth_camera')
            artifacts['independent_mano_contract']=(hawor.parent/'mano_contract.json','independent_wilor_mano_contract')
        if visual_report_path is not None:
            visual_report=load(visual_report_path)
            artifacts.update({
                'visual_geometry_report':(visual_report_path,'visual_geometry_report'),
                'visual_mesh_camera':(Path(visual_report['output_mesh']['path']),'visual_full_metric_camera_mesh'),
                'visual_native_mesh':(Path(visual_report['source_native_mesh']['path']),'visual_full_native_mesh'),
                'visual_native_qc':(appearance_qc,'visual_native_model_prior'),
            })
        @stage('source_observations',inputs={'artifacts':artifacts,'object_id':object_id,'track_id':track},
            parameters={'output':bundle_path,'schema':f'ego.annotation.observation_bundle.v{bundle_version}'})
        def _stage_source_observations():
            result=write_source_observation_manifest(output=bundle_path,run_root=root,case=args.case,
                object_id=object_id,track_id=track,frame_ids=ids,
                time_s=list(timing.presentation_times_s),source_timing=timing.report(),artifacts=artifacts,
                bundle_version=bundle_version,mano_assets=Path(assets['mano_models']),
                wilor_root=Path(assets['wilor_root']) if assets.get('wilor_root') else None)
            record('source_observations',[(bundle_path,'source_observations')])
            return result
        bundle=_stage_source_observations
        return {'run_root':str(root),'source_observations':str(bundle_path),'schema':bundle['schema'],
            'contains_optimized_state':False,'contains_policy_results':False}
    finally:
        recovery.close()





def cli():
    """Console script success is an exit status, not the returned artifact mapping."""
    main()


if __name__=='__main__':
    cli()
