"""Bounded same-video PPO experiment for frozen coarse/v2 preprocessing.

No providers, GT, CAD or legacy fits. Uses official train/rollout builders.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from data.bundle import file_sha256
from data.coarse_training import CoarseTrainingDataset, build_index
from envs.coarse_annotation_env import CoarseAnnotationEnv
import train


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def seed():
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def source_hashes():
    return {str(p.relative_to(ROOT)):file_sha256(p) for p in ROOT.rglob('*') if p.is_file()
            and p.suffix in {'.py','.yaml'} and '__pycache__' not in p.parts and '.git' not in p.parts}


def family(terms):
    values=defaultdict(float)
    for key,value in terms.items():
        p=key.split('/')
        # Named object RGB factors include factor indices; group by physical kernel.
        if 'rgb_edge' in p:label='object/rgb_rotation'
        elif 'rgb_reprojection' in p:label='object/rgb_reprojection'
        else:label='/'.join((p[0],p[2])) if len(p)>2 else key
        values[label]+=float(value)
    return dict(values)


def evaluate(args):
    import rollout
    old=rollout.build_env;rows=[];final=[]
    def build(*a,**kw):
        env=old(*a,**kw);step=env.step
        def record(action):
            result=step(action);info=result[-1]
            rows.append({'frame_idx':int(info['frame_idx']),'objective':float(info['objective']),
                         'reward':float(result[1].item()),'raw_action':info['raw_action'].tolist(),
                         'executed_action':info['executed_action'].tolist()})
            if bool(result[2].any()):final.append(info['factor_ledger'])
            if bool(result[3].any()):raise RuntimeError('unexpected artificial truncation')
            return result
        env.step=record
        return env
    rollout.build_env=build
    out=args.output
    manifest,_=rollout.rollout(config_path=args.config,dataset_path=args.dataset,split='all',
        checkpoint_path=args.checkpoint,provenance_path=args.provenance,output_path=out/'rollout.json',device='cpu')
    if len(final)!=1:raise RuntimeError('expected one complete milk episode')
    ledger=final[0]
    if not np.isclose(sum(r['reward'] for r in rows),-ledger['reward_scale']*ledger['objective'],rtol=1e-5,atol=1e-6):
        raise RuntimeError('evaluation return does not telescope')
    write(out/'evaluation.json',{'objective':ledger['objective'],'named_families':family(ledger['terms']),
        'final_ledger':ledger,'steps':rows,'manifest':str(manifest)})
    print(json.dumps({'evaluation':str(out/'evaluation.json'),'objective':ledger['objective']}),flush=True)


def checkpoint_evaluate(args, agent, env, cfg, cfg_hash, boundary):
    import pickle
    folder=args.output/'checkpoints'/f'step_{boundary:04d}';folder.mkdir(parents=True)
    train.assert_agent_checkpoint_finite(agent)
    checkpoint=folder/'agent.pt';agent.save(str(checkpoint))
    provenance=folder/'agent.provenance.json'
    write(provenance,train.checkpoint_provenance(checkpoint=checkpoint,config_sha256=cfg_hash,cfg=cfg,env=env,split='all'))
    def rng_hash():
        return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state())) + torch.get_rng_state().numpy().tobytes()).hexdigest()
    rng_before=rng_hash()
    # A subprocess cannot update the training environment or running normalizers.
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--mode','evaluate','--config',str(args.config),
        '--dataset',str(args.dataset),'--checkpoint',str(checkpoint),'--provenance',str(provenance),'--output',str(folder/'evaluation')]
    p=subprocess.run(command,capture_output=True,text=True,timeout=900)
    (folder/'evaluation.stdout.log').write_text(p.stdout);(folder/'evaluation.stderr.log').write_text(p.stderr)
    if p.returncode:raise RuntimeError('checkpoint evaluation failed: '+p.stderr[-2500:])
    rng_after=rng_hash()
    if rng_before!=rng_after:raise RuntimeError('evaluation changed training RNG')
    value=json.loads((folder/'evaluation/evaluation.json').read_text())
    row={'steps':boundary,'objective':value['objective'],'named_families':value['named_families'],
         'checkpoint':str(checkpoint),'evaluation':str(folder/'evaluation/evaluation.json'),
         'training_rng_unchanged':True,'policy_std_mean':float(agent.policy.log_std_parameter.exp().mean().detach())}
    print(json.dumps(row),flush=True)
    return row


def experiment(args):
    from envs.dynamics import decode_candidate
    args.output.mkdir(parents=True,exist_ok=False)
    args.dataset=args.output/'index.json';write(args.dataset,build_index([args.coarse]))
    cfg,cfg_hash=train.load_runtime_config(args.config)
    identity=source_hashes();write(args.output/'source_identity.json',identity)
    write(args.output/'run_config.json',{'configuration':cfg,'config_sha256':cfg_hash,'source_root':str(ROOT),
        'coarse':str(args.coarse),'coarse_sha256':file_sha256(args.coarse),'seed':0,
        'device':'cpu','compute_host':os.uname().nodename,'scope':'same_video_not_generalization'})
    raw=CoarseAnnotationEnv(CoarseTrainingDataset(args.dataset,split='all'),cfg['environment'],'cpu')
    obs,_=raw.reset(seed=0);episode=raw.state.episode;layout=episode.layout
    original_object=raw.state.object_translation['object:0'].copy()
    original_hands={side:raw.state.hand_vertices[side].copy() for side in ('left','right')}
    total=0.;start=time.monotonic()
    for i in range(episode.frame_count):
        _,r,done,truncated,info=raw.step(np.zeros(layout.size,np.float32));total+=r
        if truncated:raise RuntimeError('zero baseline truncated')
    if not done or not np.isclose(total,-1e-5*info['objective'],rtol=1e-7):raise RuntimeError('zero return contract failed')
    np.testing.assert_allclose(raw.state.object_translation['object:0'],original_object,atol=1e-12)
    replay_errors={side:float(np.linalg.norm(raw.state.hand_vertices[side]-original_hands[side],axis=-1).max()) for side in original_hands}
    if max(replay_errors.values())>1e-4:raise RuntimeError('zero MANO replay changed source geometry')
    exported=raw.export_annotation(args.output/'zero/candidate.json')
    zero={'objective':info['objective'],'named_families':family(info['factor_ledger']['terms']),
          'final_ledger':info['factor_ledger'],'npz':exported['npz'],'frames':episode.frame_count,
          'observation_dim':len(obs),'action_dim':layout.size,'MANO_zero_replay_max_m':replay_errors,
          'seconds':time.monotonic()-start}
    write(args.output/'zero/evaluation.json',zero);print('ZERO '+json.dumps({k:v for k,v in zero.items() if k not in {'final_ledger','named_families'}}),flush=True)
    raw.reset(seed=0);initial=raw.state.copy();controls=[]
    # Object relative-motion/NN factors need at least two endpoint frames. A
    # first-frame-only objective has no object sigma prior and is intentionally
    # invariant to object translation; diagnose on a three-frame causal prefix.
    controlled_prefix=min(3,episode.frame_count)
    for name,coordinate,amount in [('hand.left.translation',2,.2),('hand.left.articulation',3,.2),
                                  ('object.object:0.translation',0,.2),('object.object:0.rotation',1,.2),('contact.left',0,.2)]:
        baseline=raw.evaluator.objective(initial.copy(),controlled_prefix)
        current=initial.copy();action=np.zeros(layout.size,np.float32);action[layout.block(name).start+coordinate]=amount
        decode_candidate(current,action*__import__('actions.scaling',fromlist=['ActionScales']).ActionScales(**cfg['environment']['action_scales']).vector(layout),cfg['environment'],'cpu')
        candidate=raw.evaluator.objective(current,controlled_prefix)
        controls.append({'block':name,'normalized_delta':amount,'prefix_frames':controlled_prefix,'objective_change':candidate.objective-baseline.objective,
                         'term_changes':{k:float(v-baseline.terms.get(k,0.)) for k,v in candidate.terms.items() if abs(v-baseline.terms.get(k,0.))>1e-14}})
        if candidate.objective==baseline.objective:raise RuntimeError('controlled action has no reward coupling: '+name)
    write(args.output/'controlled_actions.json',controls)
    # RGB needs two accepted endpoint states; test both source and target Jacobian coupling.
    full=raw.evaluator.objective(initial.copy(),episode.frame_count)
    for position in (0,1):
        current=initial.copy();current.position=position;action=np.zeros(layout.size,np.float32)
        action[layout.block('object.object:0.translation').start]=.002
        decode_candidate(current,action,cfg['environment'],'cpu');ledger=raw.evaluator.objective(current,episode.frame_count)
        before=family(full.terms);after=family(ledger.terms)
        if before.get('object/rgb_reprojection')==after.get('object/rgb_reprojection'):raise RuntimeError('RGB does not couple endpoint '+str(position))
    write(args.output/'mechanism_preflight.json',{'zero_reproduces_coarse_object':True,'full_native_faces':len(episode.objects['object:0']['faces']),
        'controlled_action_blocks_coupled':True,'RGB_both_endpoints_coupled':True,'source_signed_authority':False})
    del raw,initial
    env=train.build_env(cfg,dataset_path=args.dataset,split='all',device='cpu')
    # Old checkpoints must fail even though the residual action width is unchanged.
    import rollout
    old_path=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_learning_stage1_1024_v1/checkpoints/step_0256')
    if old_path.is_dir():
        try:rollout._verify_checkpoint_contract(provenance=json.loads((old_path/'agent.provenance.json').read_text()),checkpoint=old_path/'agent.pt',cfg=cfg,env=env)
        except ValueError as e:write(args.output/'old_checkpoint_rejection.json',{'rejected':True,'reason':str(e)})
        else:raise RuntimeError('old 480-feature checkpoint accepted coarse environment')
    agent=train.build_agent(env,cfg,training=True,experiment_dir=args.output/'runs',experiment_name='train')
    initial_policy={k:v.detach().clone() for k,v in agent.policy.state_dict().items()}
    rows=[];step=env.step
    def record(action):
        result=step(action);info=result[-1]
        rows.append({'frame_idx':int(info['frame_idx']),'objective':float(info['objective']),'reward':float(result[1].item()),
                     'terminated':bool(result[2].any()),'raw_action':info['raw_action'].tolist(),
                     'executed_action':info['executed_action'].tolist()})
        return result
    env.step=record
    checks=[];post=agent.post_interaction
    def callback(*a,**kw):
        post(*a,**kw)
        timestep=int(kw['timestep'] if 'timestep' in kw else a[0])+1
        if timestep in (256,512,768,1024):checks.append(checkpoint_evaluate(args,agent,env,cfg,cfg_hash,timestep))
    agent.post_interaction=callback
    start=time.monotonic();train.build_trainer(env,agent,cfg).train()
    if not train.finite_policy_parameter_changed(agent,initial_policy):raise RuntimeError('no finite policy update')
    if len(checks)!=4:raise RuntimeError('missing checkpoint boundaries')
    if source_hashes()!=identity:raise RuntimeError('code/config changed during training')
    ds=CoarseTrainingDataset(args.dataset,split='all');b=ds[0]
    _source=Path(b.coarse['source_observations'])
    if file_sha256(_source)!=b.coarse['source_bundle_sha256'] or file_sha256(b.array_path)!=b.coarse['arrays']['sha256']:raise RuntimeError('input changed during training')
    best=min(checks,key=lambda r:r['objective'])
    summary={'scope':'same_video_not_generalization','timesteps':1024,'zero_objective':zero['objective'],
             'checkpoints':checks,'best_trained_checkpoint':best,'improves_zero':best['objective']<zero['objective'],
             'learning_seconds':time.monotonic()-start,'annotation_quality_verified':False}
    write(args.output/'training_steps.json',rows);write(args.output/'summary.json',summary)
    render_comparison(args.output,zero,best,checks[-1],ds[0]);env.close()
    write(args.output/'completed.json',{'status':'completed_bounded_training_and_full_render','frames':episode.frame_count,
        'improves_zero_objective':summary['improves_zero'],'source_unchanged':True,'code_unchanged':source_hashes()==identity})
    print('COMPLETED '+json.dumps(summary),flush=True)


def export_npz(row):
    d=json.loads(Path(row['evaluation']).read_text());p=Path(d['manifest']);m=json.loads(p.read_text())
    for artifact in m['episodes'][0]['artifacts']:
        if artifact.get('format')=='npz':
            path=Path(artifact['path']);return path if path.is_absolute() else p.parent/path
    raise RuntimeError('missing complete candidate NPZ')


def render_comparison(output,zero,best,last,bundle,*,panel_labels=None):
    import cv2
    from scipy.spatial.transform import Rotation
    from preprocessing.geometry import TriangleMesh,camera_pose,resized_intrinsics
    from preprocessing.render import paint,look_at
    archives=[]
    paths = [Path(zero['npz']), export_npz(best)] + ([export_npz(last)] if last is not None else [])
    for path in paths:
        with np.load(path,allow_pickle=False) as z:archives.append(dict(z))
    source=bundle.source;annotations=json.loads(source.source_path('annotations').read_text())['frames']
    appearance=json.loads(source.source_path('appearance_masks').read_text());images={r['frame_idx']:r for r in appearance['frames']}
    mesh=TriangleMesh(archives[0]['object_vertices_canonical'],archives[0]['object_faces'])
    n=len(archives[0]['frame_idx']);width=256;size0=[annotations[0]['source_width'],annotations[0]['source_height']]
    wh=(width,2*round(width*size0[1]/size0[0]/2));fps=1/float(np.median(np.diff(archives[0]['time_s'])))
    center=np.median(archives[0]['object_translation_world'],axis=0);C0=archives[0]['T_world_camera'][0]
    radius=max(.2,max(float(np.max(np.linalg.norm(a['object_translation_world']-center,axis=1))) for a in archives)+.12)
    view=look_at(center+C0[:3,:3]@np.array([radius*.7,-radius*.25,-radius*2.5]),center,C0[:3,:3]@np.array([0.,-1.,0.]))
    kv=np.array([width*1.05,width*1.05,(width-1)/2,(width-1)/2]);out=output/'renders';out.mkdir()
    processes=[]
    columns = len(archives)
    for name,size in [('zero_best_last_overlay.mp4',(columns*width,wh[1]+40)),('zero_best_last_world.mp4',(columns*width,width+40))]:
        processes.append(subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(out/name)],stdin=subprocess.PIPE))
    scores=[[] for _ in archives];sheets=[]
    try:
        for i,fid in enumerate(archives[0]['frame_idx']):
            row=images[int(fid)];rgb=cv2.resize(cv2.imread(row['raw_rgb_path']),wh,interpolation=cv2.INTER_AREA)
            mask=cv2.resize(cv2.imread(row['appearance_mask_path'],0),wh,interpolation=cv2.INTER_NEAREST)>0
            panels=[];world_panels=[]
            for j,a in enumerate(archives):
                C=a['T_world_camera'][i];k=resized_intrinsics(a['intrinsics_render'][i],size0,wh)
                R,t=a['object_rotation'][i],a['object_translation_world'][i]
                rc,tc=camera_pose(R,t,C);_,z,sil=mesh.raster(k,rc,tc,wh)
                scores[j].append(float(np.count_nonzero(sil&mask)/max(1,np.count_nonzero(sil|mask))))
                im=rgb.copy();depth=np.full(wh[::-1],np.inf);paint(im,depth,z,(240,160,40),.55)
                world=np.full((width,width,3),242,np.uint8);wd=np.full((width,width),np.inf)
                rw,tw=camera_pose(R,t,view);_,wz,_=mesh.raster(kv,rw,tw,(width,width));paint(world,wd,wz,(90,180,245),.95)
                for side,col in [('left',(225,150,75)),('right',(95,185,225))]:
                    if not a[side+'_valid'][i]:continue
                    faces=a[side+'_faces'];faces=faces[i] if faces.ndim==3 else faces
                    hs=TriangleMesh(a[side+'_vertices_world'][i],faces)
                    _,hz,_=hs.raster(k,C[:3,:3].T,-C[:3,:3].T@C[:3,3],wh);paint(im,depth,hz,col,.45)
                    _,hz,_=hs.raster(kv,view[:3,:3].T,-view[:3,:3].T@view[:3,3],(width,width));paint(world,wd,hz,col,.95)
                cv2.drawContours(im,cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,220,0),1)
                labels = panel_labels or (('zero R6', 'direct candidate') if last is None else ('zero R6',f'best trained {best["steps"]}',f'last {last["steps"]}'))
                header=np.zeros((40,width,3),np.uint8);label=labels[j]
                cv2.putText(header,f'f{int(fid)} {label}',(5,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                cv2.putText(header,'full mesh/MANO; no signed claim',(5,33),cv2.FONT_HERSHEY_SIMPLEX,.32,(220,220,220),1)
                panels.append(np.vstack((header,im)));world_panels.append(np.vstack((header,world)))
            overlay=np.concatenate(panels,axis=1);processes[0].stdin.write(overlay.tobytes());processes[1].stdin.write(np.concatenate(world_panels,axis=1).tobytes())
            if i in (0,30,60,90,120,143,148,149):cv2.imwrite(str(out/f'review_{int(fid):06d}.jpg'),overlay);sheets.append(overlay)
    finally:
        for p in processes:p.stdin.close()
        codes=[p.wait() for p in processes]
    if any(codes):raise RuntimeError('render encoder failed')
    cv2.imwrite(str(out/'review_sheet.jpg'),np.vstack(sheets))
    stats=[]
    for j,a in enumerate(archives):
        rot=Rotation.from_matrix(a['object_rotation']@archives[0]['object_rotation'].transpose(0,2,1)).magnitude()
        dt=np.linalg.norm(a['object_translation_world']-archives[0]['object_translation_world'],axis=1)
        stats.append({'candidate':j,'raw_SAM2_silhouette_IoU_median':float(np.median(scores[j])),
            'object_rotation_correction_max_deg':float(np.degrees(rot.max())),
            'object_translation_correction_max_mm':float(dt.max()*1000)})
    counts=[]
    for name in ('zero_best_last_overlay.mp4','zero_best_last_world.mp4'):
        cap=cv2.VideoCapture(str(out/name));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame is None:raise RuntimeError('empty decoded frame')
            count+=1
        cap.release();counts.append(count)
    if counts!=[n,n]:raise RuntimeError('full-video output frame mismatch')
    write(out/'render_measurements.json',{'frames':n,'decoded_frame_counts':counts,'fps':fps,
        'full_native_triangle_count':len(archives[0]['object_faces']),'raw_IoU_is_not_amodal_or_GT':True,
        'candidate_measurements':stats,'per_frame_raw_IoU':scores})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['experiment','evaluate'],default='experiment')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/ppo_coarse_annotation.yaml')
    parser.add_argument('--coarse',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--provenance',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();seed()
    if args.mode=='experiment':experiment(args)
    else:evaluate(args)
