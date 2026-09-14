"""Grouped prediction entry: prepared metric/RGB trajectory refinement."""
from __future__ import annotations
import argparse, copy, json, math
from pathlib import Path
import numpy as np
from utils.data import RGBEdge, RGBReprojectionFactor, indexed, RKEY, TKEY, load
from utils.state import sha, in_run, assert_prediction_only, require_run_artifact
from observations.common import build_nodes, pose_from_values
from optimization.problems import StandaloneScene
from optimization.solve import solve


def load_edges(npz_path, report_path, source_path, annotations_path, nodes, root):
    ids=[n.frame_idx for n in nodes]
    # Both serializations must originate in the same producer execution.
    source=load(source_path);object_id=source['object_id']
    npz_receipt=require_run_artifact(npz_path,root,'rgb_edges',object_id=object_id,frame_ids=ids)
    json_receipt=require_run_artifact(report_path,root,'rgb_edge_report',object_id=object_id,frame_ids=ids)
    if npz_receipt['producer_id']!=json_receipt['producer_id']:
        raise ValueError('NPZ/report producer pairing mismatch')
    report = load(report_path)
    with np.load(npz_path, allow_pickle=False) as z: data = dict(z)
    meta = json.loads(str(data['metadata'].reshape(-1)[0]))
    if meta.get('object_id')!=object_id:
        raise ValueError('RGB object identity mismatch')
    if any(report.get(key)!=value for key,value in meta.items()):
        raise ValueError('RGB NPZ/report metadata pairing mismatch')
    for key, path in [('initial_pose_report', source_path), ('annotations', annotations_path)]:
        if meta['input_sha256'][key] != sha(path) or Path(meta[key]).resolve() != Path(path).resolve():
            raise ValueError('current RGB source hash/identity mismatch')
    for key in ('depth_npz',):
        path = in_run(meta[key], root)
        require_run_artifact(path,root,'depth',frame_ids=ids)
        if sha(path) != meta['input_sha256'][key]: raise ValueError('RGB metric source mutated')
    positions = {n.frame_idx: i for i,n in enumerate(nodes)}
    source_rows = indexed(load(source_path)['pose_rows'])
    edges, projections = [], []
    offsets = data['reprojection_evidence_offsets']
    if len(offsets) != len(data['source_frame_idx'])+1 or offsets[0] != 0 or offsets[-1] != len(data['reprojection_canonical_points']) or np.any(np.diff(offsets)<0):
        raise ValueError('invalid RGB evidence offsets')
    rows = {(r['source_frame_idx'],r['target_frame_idx']):r for r in report['rows']}
    if len(rows)!=len(report['rows']):raise ValueError('duplicate RGB report edge')
    for a,b in zip(data['source_frame_idx'],data['target_frame_idx']):
        if int(a) not in positions or int(b) not in positions or (int(a),int(b)) not in rows:
            raise ValueError('RGB edge outside source timeline or report')
    for k, (a,b) in enumerate(zip(data['source_frame_idx'],data['target_frame_idx'])):
        a,b=int(a),int(b)
        if not data['accepted'][k]: continue
        row=rows[a,b]
        q=float(row['inlier_fraction'])*math.exp(-float(row['reprojection_median_px'])/2.)
        q *= max(.02,min(1.,float(row['source_3d_conditioning'])/.02))
        q *= .45 if row['topological_role']=='adjacent' else 1.
        e=RGBEdge(a,b,data['rotation_source_to_target_world'][k],data['translation_source_to_target_world_m'][k],max(.001,q),row)
        edges.append(e)
        lo,hi=map(int,offsets[k:k+2])
        if hi>lo:
            # Producer serialized canonical coordinates. Recover FIXED observed world
            # points once from its hash-bound source, never from the optimizer pose.
            r,t=pose_from_values(source_rows[a][RKEY],source_rows[a][TKEY])
            world=data['reprojection_canonical_points'][lo:hi]@r.T+t
            take=np.linspace(0,len(world)-1,min(96,len(world)),dtype=int)
            camera=data['reprojection_target_camera'][k]
            projections.append(RGBReprojectionFactor(a,b,positions[a],positions[b],world[take],
                data['reprojection_target_uv'][lo:hi][take],data['reprojection_target_intrinsics'][k],
                camera[:3,:3],camera[:3,3],data['reprojection_weights'][lo:hi][take],e.weight))
    return edges,projections


def refine(annotations_path, source_path, edge_path, edge_report, object_id, anchor, output, root):
    root=Path(root).resolve()
    for path in (annotations_path,source_path,edge_path,edge_report):in_run(path,root)
    in_run(output,root,must_exist=False)
    if output.exists(): raise ValueError('create-only posterior required')
    before=sha(source_path)
    source=load(source_path); annotations=load(annotations_path)
    assert_prediction_only(source, label='scene source prior', object_id=object_id)
    assert_prediction_only(annotations, label='scene observed geometry', object_id=object_id)
    nodes,times=build_nodes(annotations,source,object_id)
    ids=[n.frame_idx for n in nodes]
    require_run_artifact(annotations_path,root,'observations',object_id=object_id,frame_ids=ids)
    require_run_artifact(source_path,root,'source_pose',object_id=object_id,frame_ids=ids)
    if not any(n.frame_idx==anchor and n.metric_observation for n in nodes):
        raise ValueError('gauge anchor must have current retained metric support')
    edges,reprojections=load_edges(edge_path,edge_report,source_path,annotations_path,nodes,root)
    poses,diagnostics=solve(StandaloneScene(nodes,times,edges,anchor,reprojections=reprojections))
    seed_rows = {row['frame_idx']: row for row in diagnostics['local_RGB_rotation']['seed_rows']}
    rows=[]
    for n,stamp in zip(nodes,times):
        r,t=poses[n.frame_idx]
        rows.append({'frame_idx':n.frame_idx,'time_s':float(stamp),RKEY:r.tolist(),TKEY:t.tolist(),
                     'pose_source':'global_SO3_observed_RGB_SE3_visual_posterior',
                     'metric_observation':n.metric_observation,
                     'depth_feasibility_uncertain':n.frame_idx in diagnostics['depth_restoration']['affected_frame_ids'],
                     'uncertainty':max(seed_rows[n.frame_idx]['uncertainty'], .9 if n.frame_idx in diagnostics['depth_restoration']['affected_frame_ids'] else 0., n.uncertainty, .6 if n.frame_idx in diagnostics['relative_evidence_disconnected_ids'] else .2),
                     'uncertainty_kind':'heuristic_support_uncertainty_not_marginal_covariance',
                     'relative_evidence_connected':n.frame_idx not in diagnostics['relative_evidence_disconnected_ids'],
                     'local_RGB_seed':seed_rows[n.frame_idx],
                     'source_prior':n.source_row})
    report={'schema':'rgb_scene_refined_pose_v1','object_id':object_id,
            'prediction_side_only':True,'annotation_ready':False,'physical_contact_authority':False,
            'canonical_gauge':{'name':'initial_completed_canonical','anchor_frame_idx':anchor,'anchor_reason':'current_visual_selected_retained_metric_source'},
            'source_prior':{'path':str(source_path),'sha256':before},
            'evidence':{'annotations_sha256':sha(annotations_path),'rgb_npz_sha256':sha(edge_path),'rgb_report_sha256':sha(edge_report)},
            'source_pose_rows':copy.deepcopy(source['pose_rows']),'pose_rows':rows,'solver':diagnostics}
    if sha(source_path)!=before: raise ValueError('source prior mutated')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    return report



def fit_prepared(problem, run_root):
    from gates.object.geometry import validate_posterior
    from gates.interaction.visibility import validate_visibility, apply_display_visibility, POLICY
    binding=load(problem.pose_report)
    if binding.get('schema')!='rgb_scene_visual_refined_binding_v1':
        raise ValueError('typed visual-refined binding required')
    reference=binding['refined_posterior']
    if sha(reference['path'])!=reference['sha256']:
        raise ValueError('refined posterior changed')
    posterior=validate_posterior(load(reference['path']),problem.object_id,run_root)
    if binding['pose_rows']!=posterior['pose_rows']:
        raise ValueError('binding discarded refined trajectory')
    judgment=load(problem.interaction_judgments)
    validate_visibility(judgment,[r['frame_idx'] for r in binding['pose_rows']],problem.object_id)
    solve(problem)
    path=problem.output_dir/'demo_state.npz'
    with np.load(path,allow_pickle=False) as archive:
        state=apply_display_visibility(dict(archive),judgment,problem.object_id)
    np.savez_compressed(path,**state)
    path=problem.output_dir/'fit_report.json'
    report=load(path)
    report['display_visibility']={'policy':POLICY,
        'judgment_sha256':sha(problem.interaction_judgments),'source_geometry_mutated':False}
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command', required=True)
    pose=commands.add_parser('refine-pose', help='refine current-run metric and RGB evidence')
    for name in ('annotations','source-pose','rgb-edges','rgb-report','output','run-root'):
        pose.add_argument('--'+name,type=Path,required=True)
    pose.add_argument('--object-id',required=True)
    pose.add_argument('--anchor',type=int,required=True)
    visual=commands.add_parser('fit-visual',help='fit MANO/object from prepared current-run observations')
    for name in ('run-root','annotations','pose-report','mano-bridge','hawor-params','mano-models',
                 'output-dir','appearance-mask-manifest','interaction-judgments'):
        visual.add_argument('--'+name,type=Path,required=True)
    visual.add_argument('--object-id',required=True)
    visual.add_argument('--device',default='cpu')
    visual.add_argument('--iterations',type=int,default=240)
    visual.add_argument('--render-width',type=int)
    judgment=commands.add_parser('judge',help='run one independent Astra image judgment')
    judgment.add_argument('name',choices=('object','mask_track','pose_anchor','rigidity','appearance_anchor',
        'pose_trajectory','contact','contact_pads','occlusion','depth_reliability','visibility',
        'backend_geometry','final_inspection'))
    judgment.add_argument('--evidence',type=Path,required=True)
    judgment.add_argument('--image',type=Path,action='append',required=True)
    judgment.add_argument('--output',type=Path,required=True)
    formal=commands.add_parser('fit-formal',help='solve a named prepared formal MANO interval')
    formal.add_argument('--problem',type=Path,required=True)
    video=commands.add_parser('annotate',help='cold raw-video to fresh full scene annotation')
    video.add_argument('--video',type=Path,required=True)
    video.add_argument('--run-root',type=Path,required=True)
    video.add_argument('--case',required=True)
    video.add_argument('--assets-manifest',type=Path,required=True)
    video.add_argument('--gpu',type=int,required=True)
    video.add_argument('--gpu-uuid')
    video.add_argument('--sensor-calibration',type=Path)
    video.add_argument('--target-description')
    internal=commands.add_parser('_worker',help=argparse.SUPPRESS)
    internal.add_argument('kind',choices=['rgb'])
    internal.add_argument('arguments',nargs=argparse.REMAINDER)
    tail=commands.add_parser('formal-tail',help='prepared formal/HOT3D/RGB shared physical tail')
    tail.add_argument('--spec',type=Path,required=True)
    preflight=commands.add_parser('preflight',help='validate actual isolated provider imports and asset roots')
    preflight.add_argument('--assets-manifest',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='preflight':
        from utils.config import preflight_providers
        result=preflight_providers(load(args.assets_manifest));print(json.dumps(result,indent=2));return result
    if args.command=='formal-tail':return formal_tail(load(args.spec))
    if args.command=='annotate':
        result=annotate(args);print(json.dumps(result,indent=2));return result
    if args.command=='_worker':
        from observations.object.rgb_tracking import build_rgb_edges
        return build_rgb_edges(args.arguments)
    if args.command=='fit-formal':
        from optimization.problems import FormalManoInterval
        values=load(args.problem)
        for key in ('annotations','pose_report','completed_mesh','completion_report','output_dir','wilor_root','mano_left'):
            values[key]=Path(values[key])
        for key in ('depth_npz','factor_reports'):
            if key in values:values[key]=tuple(Path(p) for p in values[key])
        if values.get('hand_ray_prior') is not None:values['hand_ray_prior']=Path(values['hand_ray_prior'])
        return solve(FormalManoInterval(**values))
    if args.command=='judge':
        import importlib
        if args.output.exists():raise ValueError('judgment output must be create-only')
        module=importlib.import_module('observations.judgments.judge_'+args.name)
        value=module.judge(load(args.evidence),args.image)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False)
        return value
    if args.command=='refine-pose':
        return refine(args.annotations,args.source_pose,args.rgb_edges,args.rgb_report,
                      args.object_id,args.anchor,args.output,args.run_root)
    from optimization.problems import PreparedVisualFit
    values=vars(args).copy();values.pop('command');run_root=values.pop('run_root')
    return fit_prepared(PreparedVisualFit(**values),run_root)



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
    if not aligned(preferred):raise ValueError('preferred appearance anchor lost retained metric alignment')
    ranked=sorted((row for row in rows.values() if row['status']=='available' and aligned(row['frame_idx'])),
                  key=lambda row:(-float(row.get('score',0.)),row['frame_idx']))
    return list(dict.fromkeys([preferred,*[row['frame_idx'] for row in ranked[:12]]]))


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
    from gates.camera.calibration import load_contract, plane_intrinsics
    from gates.object.geometry import normalize_completion
    from optimization.problems import ObservedObjectPose, StandaloneShape
    from render import render_video, export_scene
    root=args.run_root.resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'input').mkdir();(root/'logs').mkdir()
    raw=root/'input/raw.mp4'
    with args.video.open('rb') as source,raw.open('xb') as destination:shutil.copyfileobj(source,destination)
    if sha(raw)!=sha(args.video):raise ValueError('raw copy identity mismatch')
    timing=read_cfr_video_timing(raw)
    assets=load(args.assets_manifest)
    if assets.get('agent_model')!='gpt-6-astra' or assets.get('agent_provider')!='dexgem-responses' or assets.get('agent_disable_fallbacks') is not True:
        raise ValueError('asset manifest must pin Astra without fallbacks')
    os.environ['EGO_ASTRA_CREDENTIAL_FILE']=assets['agent_credential_file']
    code_root=Path(__file__).resolve().parent
    code_manifest=root/'input/code_manifest.json'
    code_files=[code_root/'main.py',code_root/'render.py']
    for directory in ('utils','observations','gates','factors','optimization'):
        code_files.extend((code_root/directory).rglob('*.py'))
    code_hashes={str(p.relative_to(code_root)):sha(p) for p in code_files}
    code_manifest.write_text(json.dumps({'files':code_hashes},sort_keys=True))
    sensor=None
    if args.sensor_calibration is not None:
        sensor=root/'input/sensor_calibration.json';shutil.copyfile(args.sensor_calibration,sensor)
    contract={'schema':'rgb_scene_input_v1','run_id':str(uuid.uuid4()),'case':args.case,
        'run_root':str(root),'input_video':str(raw),'input_video_sha256':sha(raw),
        'bundle_manifest_sha256':sha(code_manifest),'sensor_calibration_metadata':str(sensor) if sensor else None,
        'sensor_calibration_sha256':sha(sensor) if sensor else None,'source_timing':timing.report(),
        'target_description':args.target_description}
    (root/'input/runtime_input_contract.json').write_text(json.dumps(contract,indent=2))
    ledger=RunState(root);ids=list(range(len(timing.presentation_ticks)));object_id=None
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu),'HF_HUB_OFFLINE':'1',
        'TRANSFORMERS_OFFLINE':'1','EGO_SAM2_SOURCE':assets['sam2_source'],
        'PYTHONPATH':str(code_root)+os.pathsep+os.environ.get('PYTHONPATH','')}
    def verify_code():
        if any(sha(code_root/name)!=digest for name,digest in code_hashes.items()):
            raise RuntimeError('source code changed during the run; deploy an immutable snapshot')
    def record(stage,outputs,command=None):
        verify_code()
        return ledger.record(stage,outputs,object_id=object_id,frame_ids=ids,command=command)
    gpu_uuid=args.gpu_uuid
    def worker(stage,python,module,argv,outputs,overlay=None):
        nonlocal gpu_uuid
        verify_code()
        command=[str(python),'-m',module,*map(str,argv)]
        with (root/'logs'/f'{stage}.log').open('x') as log:
            resources=run_model(command,cwd=code_root,environment={**environment,**(overlay or {})},
                gpu=args.gpu,expected_uuid=gpu_uuid,min_free_mib=int(assets.get('sam3d_min_free_mib',12000)),stdout=log)
        gpu_uuid=resources['uuid']
        (root/'logs'/f'{stage}.resources.json').write_text(json.dumps(resources,indent=2))
        record(stage,outputs,command)
    def ask(name,evidence,images,path):
        if path.exists():raise ValueError('judgment output already exists')
        module=importlib.import_module('observations.judgments.judge_'+name)
        value=module.judge({'frame_ids':ids,**({'object_id':object_id} if object_id else {}),**evidence},images)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False))
        ledger.record('judge_'+name,[(path,'judgment')],object_id=object_id,frame_ids=evidence.get('frame_ids',ids));return value
    manifest=root/'input/raw_frame_manifest/manifest.json'
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
    calibration.resolve_camera(calibration.camera_parse_args(list(map(str,['--case',args.case,
        '--raw-frame-manifest',manifest,'--output-dir',camera_dir,*extra]))))
    record('camera',[(camera,'calibration')])
    camera_payload,camera_normal=load_contract(camera,expected_frame_ids=ids)
    K,_=plane_intrinsics(camera_payload,camera_normal,plane_name='source_rgb',actual_size_wh=(width,height))
    if not np.isclose(K[0],K[1],atol=1e-6,rtol=1e-6):raise ValueError('HaWoR requires square focal; sensor K cannot be averaged')
    depth=depth_job('conditioned_depth',root/'measurements/depth_slam/unidepth_full_frame',True)
    bound=camera_dir/'depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz'
    calibration.bind_depth_contract(calibration.depth_parse_args(list(map(str,['--source-depth-npz',depth,
        '--camera-contract',camera,'--depth-plane','source_rgb','--output-dir',bound.parent]))))
    record('depth_binding',[(bound,'depth')])
    hawor=root/'measurements/hand_candidates/hawor_world/hawor_world_hands.npz'
    worker('hawor',assets['hawor_python'],'observations.hand.hawor',
        ['--hawor-root',assets['hawor_root'],'--run-root',root,'--video_path',raw,
         '--camera-intrinsics',*K,'--output-dir',hawor.parent],[(hawor,'hands')])
    detector=root/'measurements/object_candidates/object_box_prompts_owlv2'
    owlargs=['--raw-frame-manifest',manifest,'--output-root',detector,'--review-dir',root/'renders/owlv2',
        '--case-id',args.case,'--object-id',object_id,'--track-id',track,'--description',obj['description'],
        '--prompt-frames',','.join(map(str,obj['detector_prompt_frames'])),
        '--active-start',obj['detector_active_interval'][0],'--active-end',obj['detector_active_interval'][1],
        '--owlv2-model',assets['owlv2_model'],'--box-threshold',.03,'--device','cuda','--box-only']
    for prompt in obj['detector_text_prompts']:owlargs+=['--text-prompt',prompt]
    worker('owlv2',assets['owlv2_python'],'observations.object.owlv2',owlargs,
        [(detector/object_id/'object_point_prompts_vlm.json','object_boxes')])
    sam2=root/'measurements/object_tracks/sam2_owlv2_box_points';track_path=sam2/track/'sam2/sam2_track.json'
    worker('sam2',assets['unidepth_python'],'observations.object.sam2',
        ['--clip',raw,'--point-root',detector,'--output-root',sam2,'--checkpoint',assets['sam2_checkpoint'],
         '--frame-start',0,'--frame-end',ids[-1]],[(track_path,'object_masks')])
    track_payload=load(track_path);mask_reviews=[]
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
    mask_judgment=ask('mask_track',{'track':track_payload,'inspected_frame_ids':sampled},mask_reviews,root/'state/mask_track_judgment.json')
    if mask_judgment['identity_consistent'] is False:
        raise ValueError('mask identity review rejected the selected-object track; no scene is published')
    base=root/'state/base_annotations/annotations_v19_base.json'
    build_base_annotations(base_parse_args(list(map(str,['--case',args.case,'--raw-frame-manifest',manifest,
        '--output-dir',base.parent,'--depth-npz',bound,'--calibration-contract',camera,
        '--hawor-npz',hawor,'--object-plan',plan_path,'--sam2-output-root',sam2]))))
    bridge=base.parent/'v19_mano_bridge_from_hawor_world.npz'
    record('base',[(base,'base_observations'),(bridge,'mano_bridge')])
    owned=root/'measurements/object_tracks/depth_order_owned'/track
    surface.build_depth_order(surface.depth_order_parse_args(list(map(str,['--base-annotations',base,
        '--raw-sam2-track',track_path,'--depth-npz',bound,'--calibration-contract',camera,
        '--mano-bridge',bridge,'--hawor-npz',hawor,'--output-dir',owned,'--case',args.case,
        '--object-id',object_id,'--track-id',track]))))
    record('ownership',[(owned/'sam2_track.json','owned_masks'),(owned/'native_depth_metric_eligibility.json','metric_eligibility')])
    proposals=root/'measurements/object_geometry/anchor_candidates'/object_id
    surface.build_visible_geometry(surface.surface_parse_args(owned_surface_arguments(root,args.case,object_id,track,proposals)))
    proposal_path=proposals/'anchor_candidate_proposals.json'
    decision=ask('pose_anchor',{'frame_ids':[r['frame_idx'] for r in load(proposal_path)['review_candidates']],'candidates':load(proposal_path),'candidate_report_path':str(proposal_path)},
        [proposals/'anchor_candidate_review.jpg'],root/'state/anchor_decisions'/f'{object_id}.json')
    anchor=decision['selected_anchor_frame_idx']
    visible=root/'measurements/object_geometry/visible_geometry'/object_id
    surface.build_visible_geometry(surface.surface_parse_args(owned_surface_arguments(root,args.case,object_id,track,visible,anchor=anchor)))
    ann=visible/'annotations_v19_visible_geometry.json'
    record('surface',[(ann,'observations'),(visible/'v19_visible_geometry_adapter_report.json','surface_report')])
    rigidity=ask('rigidity',{},images,root/'state/rigidity.json')
    if rigidity['physical_branch']!='rigid':raise ValueError('selected standalone scene requires rigid object evidence')
    evidence_root=root/'measurements/geometry_completion/rigid_evidence'
    surface.build_anchor_evidence(list(map(str,['--case',args.case,'--object-id',object_id,'--annotations',ann,
        '--depth-fused-report',visible/'v19_visible_geometry_depth_fused_report.json',
        '--output-root',evidence_root,'--selected-frame-idx',anchor,'--selection-note',decision['visual_rationale']])))
    evidence=evidence_root/args.case/object_id/'evidence_bundle/evidence_bundle_report.json'
    geometry=root/'measurements/geometry_completion'/('sam3d_'+object_id)
    shape.build_completion_inputs(shape.conditioning_parse_args(list(map(str,['--evidence-report',evidence,'--output-dir',geometry/'native_inputs']))))
    native_input=geometry/'native_inputs/p11_dual_geometry_inputs_report.json'
    conditioning=load(native_input)['conditioning_contracts']['sam3d_objects_native']
    def native_job(stage,image,mask,out,frame):
        name=f'{args.case}_{object_id}_frame_{frame}'
        argv=['--repo',assets['sam3d_repo'],'--config',assets['sam3d_config'],
            '--moge-checkpoint',assets['sam3d_moge_checkpoint'],'--dinov2-repo',assets['sam3d_dinov2_repo'],
            '--dinov2-checkpoint',assets['sam3d_dinov2_checkpoint'],'--case',f'{name}|{image}|{mask}|42','--output-dir',out]
        qc=out/name/'qc_sam3d_objects_mesh_v7.json'
        worker(stage,assets['sam3d_python'],'observations.object.sam3d',argv,[(qc,'native_geometry')],assets.get('sam3d_environment',{}))
        return name,qc,argv
    name,qc,command=native_job('metric_native',conditioning['image'],conditioning['mask'],geometry/'native_reconstruction',anchor)
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
    source_pose=root/'measurements/pose_fits'/object_id/'visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json'
    solve(ObservedObjectPose(tuple(map(str,['--annotations',ann,'--completion-report',completion_path,'--object-id',object_id,'--output-dir',source_pose.parent]))))
    record('observed_pose',[(source_pose,'source_pose')])
    edges=root/'measurements/pose_fits'/object_id/'scene_rgb/edges.npz'
    worker('rgb_edges',assets['rgb_python'],'main',['_worker','rgb','--annotations',ann,'--raw-frame-manifest',manifest,
        '--depth-npz',bound,'--initial-pose-report',source_pose,'--object-id',object_id,'--frame-start',0,
        '--frame-end',ids[-1],'--keyframe-seconds',.5,'--source-gap-seconds',2.,'--keyframe-neighbor-span',4,
        '--lightglue-root',assets['lightglue_root'],'--assets-manifest',args.assets_manifest,'--device','cuda:0',
        '--output-npz',edges,'--output-json',edges.with_suffix('.json')],[(edges,'rgb_edges'),(edges.with_suffix('.json'),'rgb_edge_report')])
    posterior=root/'state/scene'/object_id/'refined_pose.json'
    refine(ann,source_pose,edges,edges.with_suffix('.json'),object_id,anchor,posterior,root)
    record('refined_pose',[(posterior,'candidate_pose')])
    appearance= root/'measurements/demo_visual'/object_id/'appearance_evidence/appearance_mask_manifest.json'
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
    _,native_qc,_=native_job('appearance_native',selected_row['raw_rgb_path'],selected_row['appearance_mask_path'],
        root/'measurements/scene_native'/object_id,selected)
    binding_dir=root/'measurements/scene_shape'/object_id
    solve(StandaloneShape(posterior,ann,appearance,choice_path,native_qc,binding_dir,object_id,root))
    binding=binding_dir/'pose_mesh.json'
    from render import pose_review_images
    pose_images=pose_review_images(load(ann),load(binding),hawor,sampled,root/'renders/refined_pose')
    trajectory_judgment=ask('pose_trajectory',{'posterior':load(posterior),'inspected_frame_ids':sampled},pose_images,
        root/'state/pose_trajectory_judgment.json')
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
    interaction={'object_id':object_id,'status':'ok','interaction_judgments':[],'frames':[],
        'diagnostics':{'occlusion':[],'depth_reliability':[]},'timeline_evidence':{'contact':[],'pads':[]}}
    for offset in range(0,len(ids),12):
        batch=ids[offset:offset+12];batch_images=[Path(raw_by_id[i]['raw_frame_path']) for i in batch]
        evidence_batch={'frame_ids':batch}
        destination=root/'state/judgments'/f'batch_{offset:06d}'
        contact=ask('contact',evidence_batch,batch_images,destination/'contact.json')
        pads=ask('contact_pads',{**evidence_batch,'contact':contact},batch_images,destination/'pads.json')
        occlusion=ask('occlusion',evidence_batch,batch_images,destination/'occlusion.json')
        reliability=ask('depth_reliability',evidence_batch,batch_images+[depth_images[i] for i in batch],destination/'depth_reliability.json')
        for name,result in [('occlusion',occlusion),('depth_reliability',reliability)]:
            interaction['diagnostics'][name].append({'role':'diagnostic_only_not_a_visual_fit_factor',
                'intervals':result['intervals'],'judgment_sha256':sha(destination/f'{name}.json')})
        visibility=ask('visibility',evidence_batch,batch_images,destination/'visibility.json')
        interaction['frames'].extend(visibility['frames'])
        interaction['timeline_evidence']['contact'].append({'judgment_sha256':sha(destination/'contact.json'),
            'intervals':contact['interaction_judgments'],'provenance':contact['judgment_provenance']})
        interaction['timeline_evidence']['pads'].append({'judgment_sha256':sha(destination/'pads.json'),
            'intervals':pads['intervals'],'provenance':pads['judgment_provenance']})
        for row in contact['interaction_judgments']:
            # Expand supplied intervals only to attach the independently judged anatomical pads.
            for frame in batch:
                if row['frame_start']<=frame<=row['frame_end']:
                    candidates=[r for r in pads['intervals'] if r['hand_side']==row['hand_side'] and r['frame_start']<=frame<=r['frame_end']]
                    if len(candidates)>1:raise ValueError('overlapping anatomical pad judgments')
                    entry={**row,'frame_start':frame,'frame_end':frame}
                    if candidates:entry['primary_contact_pads']={'digits':candidates[0]['digits'],'rationale':candidates[0]['rationale']}
                    interaction['interaction_judgments'].append(entry)
    judgments=root/'state/agent_interaction_judgments'/f'{object_id}.json';judgments.parent.mkdir(parents=True,exist_ok=True)
    judgments.write_text(json.dumps(interaction,indent=2));record('interaction_assembly',[(judgments,'interaction_judgment')])
    fit=root/'measurements/scene_fit'/object_id
    worker('visual_fit',assets['demo_python'],'main',['fit-visual','--run-root',root,'--annotations',ann,
        '--pose-report',binding,'--mano-bridge',bridge,'--hawor-params',hawor,'--mano-models',assets['mano_models'],
        '--object-id',object_id,'--appearance-mask-manifest',appearance,'--interaction-judgments',judgments,
        '--device','cuda:0','--output-dir',fit],[(fit/'demo_state.npz','visual_state'),(fit/'fit_report.json','visual_fit_report')])
    reviews=render_video(fit/'demo_state.npz',manifest,root/'renders/annotation.mp4')
    review_images=[Path(r['path']) for r in reviews]
    backend_judgment=ask('backend_geometry',{'backends':['sam3d'],'binding':load(binding)},
        [review_images[i] for i in np.unique(np.linspace(0,len(ids)-1,min(12,len(ids)),dtype=int))],root/'state/backend_judgment.json')
    inspections=[]
    for offset in range(0,len(ids),12):
        inspections.append(ask('final_inspection',{'frame_ids':ids[offset:offset+12]},review_images[offset:offset+12],
            root/'state/final_inspection'/f'{offset:06d}.json'))
    output=root/'final'
    review=scene_review_summary(mask_judgment,trajectory_judgment,inspections)
    review['backend_geometry']={'role':'diagnostic_only','judgment':{k:v for k,v in backend_judgment.items() if k!='judgment_provenance'},'judgment_sha256':sha(root/'state/backend_judgment.json')}
    result=export_scene(fit/'demo_state.npz',binding,fit/'fit_report.json',root,code_manifest,output,visual_review=review)
    return {'run_root':str(root),'annotation':str(output/'annotation.json'),'rrd':str(output/'world.rrd'),
        'video':str(root/'renders/annotation.mp4'),'annotation_ready':False,
        'visual_review':review}




def formal_tail(spec):
    """Prepared formal/shared tail: one physical solve precedes backend display split."""
    from types import SimpleNamespace
    from optimization.problems import FormalObjectTrajectory, FormalManoInterval
    from observations.interaction.evidence import interaction_build, interaction_parse_args, build_geometric_constraints
    from observations.object.signed_geometry import signed_proxy_build, signed_proxy_parse_args
    from gates.hand.acceptance import accept_mano_surface
    from render import build_formal_render_state, formal_state_parse_args, assemble_formal_layers, render_formal_rrd, render_formal_video
    root=Path(spec['run_root']).resolve();out=Path(spec['output']).resolve()
    in_run(out,root,must_exist=False);out.mkdir(parents=True,exist_ok=False)
    obj=spec['object_id'];case=spec['case'];ann=in_run(spec['annotations'],root)
    source=in_run(spec['source_pose'],root);completion=in_run(spec['completion_report'],root)
    hawor=in_run(spec['hawor_npz'],root);judgments=in_run(spec['interaction_judgments'],root)
    depth=[in_run(path,root) for path in spec['depth_npz']]
    images=load(ann)['frames'];frame_ids=[row['frame_idx'] for row in images]
    profile=spec.get('profile','formal')
    if profile not in {'formal','hot3d_shared','rgb_sam3d'}:raise ValueError('unknown retained formal path')
    if profile=='rgb_sam3d':
        from observations.object.rgb_tracking import initialize
        nodes,_=build_nodes(load(ann),load(source),obj)
        edge_path=in_run(spec['rgb_edges'],root);edge_report=in_run(spec['rgb_report'],root)
        load_edges(edge_path,edge_report,source,ann,nodes,root)
        with np.load(edge_path,allow_pickle=False) as archive:arrays=dict(archive)
        initialized=initialize(load(source),load(ann),obj,arrays,load(edge_report)['rows'],int(spec['anchor_frame']))
        source=out/'rgb_initialized_pose.json';source.write_text(json.dumps(initialized,indent=2,allow_nan=False))
    object_problem=FormalObjectTrajectory(ann,source,completion,out/'pose',obj)
    pose_report=solve(object_problem)
    pose_path=out/'pose/v19_rigid_object_pose_graph_report.json'
    physical_completion=completion
    if profile=='hot3d_shared':
        signed_proxy_build(signed_proxy_parse_args(list(map(str,['--case',case,'--object-id',obj,
            '--annotations',ann,'--direct-pose-report',source,'--pose-report',pose_path,
            '--observed-completion-report',completion,'--depth-npz',depth[0],'--output-dir',out/'signed_geometry']))))
        physical_completion=out/'signed_geometry/shared_signed_geometry_completion_report.json'
    factor_args=interaction_parse_args(list(map(str,['--annotations',ann,'--case',case,
        '--target-entity-id','object:'+obj.removeprefix('object:'),'--frame-span',min(frame_ids),max(frame_ids),
        '--output-root',out/'interaction','--agent-interaction-judgment',judgments,
        '--require-camera-mano-contract-aligned'])))
    factor=interaction_build(factor_args)
    factor_path=out/'interaction'/case/'v19_visible_contact_ownership_factor_report.json'
    factor_path.parent.mkdir(parents=True,exist_ok=True);factor_path.write_text(json.dumps(factor,indent=2))
    mesh=load(completion)['outputs'].get('pose_hypothesis_mesh_labeled') or load(completion)['outputs']['completed_mesh_labeled']
    mano=FormalManoInterval(case,obj,ann,pose_path,Path(mesh),physical_completion,depth,out/'mano',
        Path(spec['wilor_root']),Path(spec['mano_left']),min(frame_ids),max(frame_ids),
        profile='hot3d_shared' if profile=='hot3d_shared' else 'formal',device=spec.get('device','cpu'),factor_reports=[factor_path])
    solve(mano)
    contact_state=out/'mano'/case/'v18_joint_mano_interval_trajectory_state.json'
    accepted=out/'accepted_mano.json'
    accept_mano_surface(list(map(str,['--contact-state',contact_state,'--hawor-npz',hawor,
        '--annotations',ann,'--case',case,'--object-id',obj,'--accept-signed-full-mano','--output',accepted])))
    build_geometric_constraints(list(map(str,['--annotations',ann,'--hawor-npz',hawor,
        '--pose-report',pose_path,'--completion-report',physical_completion,
        '--output-dir',out/'constraints','--object-id',obj])))
    constraints=out/'constraints/v18_mano_object_constraint_state.json'
    state=out/'render_state.json'
    build_formal_render_state(formal_state_parse_args(list(map(str,['--case',case,'--object-id',obj,
        '--annotations',ann,'--pose-report',pose_path,'--completion-report',completion,
        '--constraint-report',constraints,'--temporal-mano-state',accepted,'--output',state]))))
    package=assemble_formal_layers(SimpleNamespace(render_state=state,annotations=ann,hawor_npz=hawor,
        interaction_judgments=judgments,output_dir=out/'package',run_root=root,demo_state=None,demo_report=None,allow_external_replay=False))
    package_path=out/'package/annotation.json'
    # The assembler returns the source-named package path; don't guess a stale file.
    if isinstance(package,(str,Path)):package_path=Path(package)
    render_formal_rrd(package_path,out/'world.rrd',layer='prediction')
    render_formal_video(package_path,in_run(spec['source_video'],root),out/'video')
    if spec.get('backend_geometry') is not None:
        from render import render_dual_backend
        render_dual_backend(package_path,in_run(spec['source_video'],root),spec['backend_geometry'],out/'dual_display')
    acceptance=load(accepted)['full_mano_acceptance']
    result={'profile':profile,'source_pose_sha256':sha(source),'pose_report':str(pose_path),
        'mano_candidate':str(contact_state),'mano_acceptance':acceptance,'package':str(package_path),
        'physical_pose_accepted':pose_report.get('annotation_ready') is True,
        'physical_mano_accepted':acceptance.get('accepted') is True}
    (out/'result.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    main()
