"""Paired reward-family ablation screen; unchanged full-action PPO and full evaluation."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from data.coarse_training import CoarseTrainingDataset,build_index
from envs.reward_ablation_coarse_env import RewardAblationCoarseEnv
from rewards.ablation import GROUPS,CONTRACT,grouped
from tools.rl_experiments.preprocessed_learning import write,source_hashes,render_comparison
from tools.rl_experiments.generated_collision_learning import audit_and_csv
from tools.rl_experiments.object_only_learning import optimizer_steps
import train
import rollout

VARIANTS=('full','generated_collision','hand_pixel','hand_surface_depth','hand_priors','hand_temporal',
          'object_geometry','object_rgb_rotation','object_rgb_reprojection','object_temporal','contact','hand_joint_depth','hand_bounds')
SEEDS=(0,1)


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def parameter_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().numpy();h.update(k.encode());h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    return h.hexdigest()


def metrics(a,base,rows,zero_prob):
    result={};n=len(a['frame_idx']);dt=np.diff(a['time_s'])
    C=a['T_world_camera'];K=a['intrinsics_render']
    def project(j):
        camera=np.einsum('tni,tij->tnj',j-C[:,None,:3,3],C[:,:3,:3])
        uv=camera[:,:,:2]/np.maximum(camera[:,:,2:3],1e-6)*K[:,None,:2]+K[:,None,2:]
        return uv,camera[:,:,2]
    for side in ['left','right']:
        valid=a[side+'_valid'].astype(bool);uv,z=project(a[side+'_joints_world']);buv,bz=project(base[side+'_joints_world'])
        shift=np.linalg.norm(uv-buv,axis=-1);depth=np.abs(z-bz)
        delta=a[side+'_vertices_world']-base[side+'_vertices_world'];length=np.linalg.norm(delta,axis=-1)
        velocity=np.diff(delta,axis=0)/dt[:,None,None]
        acceleration=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None,None]
        result[side]={'vertex_correction_mean_mm':float(length[valid].mean()*1000),
                      'vertex_correction_max_mm':float(length[valid].max()*1000),
                      'joint_pixel_median':float(np.median(shift[valid])),'joint_pixel_p95':float(np.percentile(shift[valid],95)),
                      'joint_pixel_max':float(shift[valid].max()),'joint_pixel_over_12px':int((shift[valid]>12).sum()),
                      'joint_depth_shift_p95_mm':float(np.percentile(depth[valid],95)*1000),
                      'nonpositive_joint_depth_count':int((z[valid]<=0).sum()),
                      'correction_velocity_p95_mm_s':float(np.percentile(np.linalg.norm(velocity,axis=-1),95)*1000),
                      'correction_acceleration_p95_mm_s2':float(np.percentile(np.linalg.norm(acceleration,axis=-1),95)*1000)}
    trans=a['object_translation_world']-base['object_translation_world']
    rot=Rotation.from_matrix(a['object_rotation']@base['object_rotation'].transpose(0,2,1)).as_rotvec()
    v=np.diff(trans,axis=0)/dt[:,None];acc=np.diff(v,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]
    result['object']={'translation_max_mm':float(np.linalg.norm(trans,axis=-1).max()*1000),
                      'rotation_max_deg':float(np.degrees(np.linalg.norm(rot,axis=-1).max())),
                      'correction_velocity_p95_mm_s':float(np.percentile(np.linalg.norm(v,axis=-1),95)*1000),
                      'correction_acceleration_p95_mm_s2':float(np.percentile(np.linalg.norm(acc,axis=-1),95)*1000),
                      'correction_rotvec_rate_p95_deg_s':float(np.degrees(np.percentile(np.linalg.norm(np.diff(rot,axis=0)/dt[:,None],axis=-1),95)))}
    probabilities=np.asarray([r['contact_mean_probability'] for r in rows])
    result['contact_prior_deviation_mean_abs']=np.abs(probabilities-np.asarray(zero_prob)).mean(axis=0).tolist()
    executed=np.asarray([r['executed_delta'] for r in rows])
    from actions.space import ActionLayout
    layout=ActionLayout(object_slots=('object:0',),contact_variables_per_hand=5)
    result['action_blocks']={block.name:{'coordinate_rms':float(np.sqrt(np.mean(executed[:,block.slice]**2))),
                                       'norm_p95':float(np.percentile(np.linalg.norm(executed[:,block.slice],axis=-1),95))} for block in layout.blocks}
    result['all_frames']=n
    return result


def evaluate(args):
    old=rollout.build_env;rows=[];final=[]
    def build(*a,**kw):
        env=old(*a,**kw);step=env.step
        def record(action):
            result=step(action);info=result[-1];i=env._env.state.position-1
            probability=1/(1+np.exp(-env._env.state.contact_logits[i]))
            rows.append({'frame_idx':int(info['frame_idx']),'training_objective':info['training_objective'],
                         'full_evaluation_objective':info['full_evaluation_objective'],'reward':float(result[1].item()),
                         'executed_delta':info['executed_delta'].tolist(),'raw_action':info['raw_action'].tolist(),
                         'contact_mean_probability':probability.mean(axis=-1).tolist()})
            if bool(result[2].any()):final.append(info)
            if bool(result[3].any()):raise RuntimeError('truncated ablation evaluation')
            return result
        env.step=record
        return env
    rollout.build_env=build
    manifest,_=rollout.rollout(config_path=args.config,dataset_path=args.dataset,split='all',
        checkpoint_path=args.checkpoint,provenance_path=args.provenance,output_path=args.output/'rollout.json',device='cpu')
    if len(final)!=1 or len(rows)!=150:raise RuntimeError('evaluation must cover complete milk timeline')
    info=final[0]
    if not np.isclose(sum(r['reward'] for r in rows),-1e-5*info['training_objective'],rtol=1e-5,atol=1e-6):raise RuntimeError('selected training reward return failed')
    if not np.isclose(info['training_objective']+info['removed_objective'],info['full_evaluation_objective'],rtol=1e-9):raise RuntimeError('full/ablated objective accounting failed')
    m=json.loads(Path(manifest).read_text());artifact=next(x for x in m['episodes'][0]['artifacts'] if x['format']=='npz')
    path=Path(artifact['path']);path=path if path.is_absolute() else Path(manifest).parent/path
    zero=json.loads((args.zero/'evaluation.json').read_text())
    with np.load(zero['npz'],allow_pickle=False) as z:base=dict(z)
    with np.load(path,allow_pickle=False) as z:a=dict(z)
    for key in ['frame_idx','time_s','T_world_camera','object_vertices_canonical','object_faces']:
        np.testing.assert_array_equal(a[key],base[key])
    geometric=metrics(a,base,rows,zero['contact_prior_probabilities'])
    collision=audit_and_csv(path,args.output)
    result={'steps':256,'seed':args.seed,'drop_group':args.group,'objective':info['full_evaluation_objective'],
            'training_objective':info['training_objective'],'removed_objective':info['removed_objective'],
            'original_formal_objective':info['original_formal_objective'],'generated_collision_objective':info['generated_collision_objective'],
            'full_group_costs':info['full_group_costs'],'full_ledger':info['full_factor_ledger'],'training_ledger':info['factor_ledger'],
            'manifest':str(manifest),'npz':str(path),'geometry':geometric,'collision':collision,'steps_trace':rows}
    write(args.output/'evaluation.json',result)
    print(json.dumps({k:result[k] for k in ['drop_group','seed','objective','training_objective','removed_objective']}),flush=True)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,cfg_hash=train.load_runtime_config(args.config)
    identity=source_hashes()
    ds=CoarseTrainingDataset(args.dataset,split='all')
    env=train.build_env(cfg,dataset_path=args.dataset,split='all',device='cpu',dataset=ds)
    agent=train.build_agent(env,cfg,training=True,experiment_dir=args.output/'runs',experiment_name='paired_ablation')
    initial_policy={k:v.detach().clone() for k,v in agent.policy.state_dict().items()}
    hashes={'actor':parameter_hash(agent.policy),'critic':parameter_hash(agent.value)}
    write(args.output/'initial_parameters.json',hashes)
    original_update=agent.update;updates=[]
    def update(*,timestep,timesteps):
        before=optimizer_steps(agent);original_update(timestep=timestep,timesteps=timesteps)
        updates.append({'step':int(timestep)+1,'optimizer_steps':optimizer_steps(agent)-before})
    agent.update=update
    trace=[];original_step=env.step
    def record(action):
        result=original_step(action);info=result[-1]
        trace.append({'frame_idx':int(info['frame_idx']),'training_objective':info['training_objective'],
                      'full_objective':info['full_evaluation_objective'],'removed_objective':info['removed_objective'],
                      'reward':float(result[1].item()),'executed_delta':info['executed_delta'].tolist(),
                      'terminated':bool(result[2].any())})
        return result
    env.step=record
    start=time.monotonic();train.build_trainer(env,agent,cfg).train()
    train.assert_agent_checkpoint_finite(agent)
    if not train.finite_policy_parameter_changed(agent,initial_policy) or len(trace)!=256 or len(updates)!=2:raise RuntimeError('incomplete paired PPO run')
    cp=args.output/'agent.pt';agent.save(str(cp))
    prov=args.output/'agent.provenance.json';write(prov,train.checkpoint_provenance(checkpoint=cp,config_sha256=cfg_hash,cfg=cfg,env=env,split='all'))
    write(args.output/'training_trace.json',trace);write(args.output/'updates.json',updates)
    env.close()
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--mode','evaluate','--config',str(args.config),
             '--dataset',str(args.dataset),'--zero',str(args.zero),'--seed',str(args.seed),'--group',args.group,
             '--checkpoint',str(cp),'--provenance',str(prov),'--output',str(args.output/'evaluation')]
    p=subprocess.run(command,capture_output=True,text=True,timeout=420)
    (args.output/'evaluation.stdout.log').write_text(p.stdout);(args.output/'evaluation.stderr.log').write_text(p.stderr)
    if p.returncode:raise RuntimeError('full independent evaluation failed: '+p.stderr[-2500:])
    if identity!=source_hashes():raise RuntimeError('frozen ablation runtime changed')
    value=json.loads((args.output/'evaluation/evaluation.json').read_text())
    row={'steps':256,'seed':args.seed,'drop_group':args.group,'objective':value['objective'],
         'training_objective':value['training_objective'],'removed_objective':value['removed_objective'],
         'original_formal_objective':value['original_formal_objective'],'generated_collision_objective':value['generated_collision_objective'],
         'full_group_costs':value['full_group_costs'],'geometry':value['geometry'],'collision':value['collision'],
         'checkpoint':str(cp),'evaluation':str(args.output/'evaluation/evaluation.json'),'npz':value['npz'],
         'initial_parameter_hashes':hashes,'final_actor_sha256':parameter_hash(agent.policy),'seconds':time.monotonic()-start}
    write(args.output/'assessment.json',row);write(args.output/'completed.json',{'status':'complete_paired_256step_full_150frame_eval','source_unchanged':True})
    print('COMPLETE '+json.dumps({k:row[k] for k in ['seed','drop_group','objective','training_objective','seconds']}),flush=True)


def make_zero(args,cfg):
    ds=CoarseTrainingDataset(args.dataset,split='all');env=RewardAblationCoarseEnv(ds,cfg['environment'],'cpu')
    obs,_=env.reset(seed=0);initial=env.state.copy();zero_prob=(1/(1+np.exp(-initial.contact_logits))).mean(axis=-1)
    rewards=[]
    for _ in range(150):_,r,done,truncated,info=env.step(np.zeros(118,np.float32));rewards.append(r)
    if not done or truncated or not np.isclose(sum(rewards),-1e-5*info['full_evaluation_objective'],rtol=1e-7):raise RuntimeError('common zero baseline failed')
    export=env.export_annotation(args.output/'zero/candidate.json')
    result={'objective':info['full_evaluation_objective'],'training_objective':info['training_objective'],
            'original_formal_objective':info['original_formal_objective'],'generated_collision_objective':info['generated_collision_objective'],
            'full_group_costs':info['full_group_costs'],'full_ledger':info['full_factor_ledger'],'npz':export['npz'],
            'contact_prior_probabilities':zero_prob.tolist(),'frames':150,'actions':118,'observations':len(obs)}
    result['collision']=audit_and_csv(export['npz'],args.output/'zero')
    write(args.output/'zero/evaluation.json',result)
    env.close();return result,ds


def run_job(job):
    log=Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as f:
        try:
            p=subprocess.run(job['command'],stdout=f,stderr=subprocess.STDOUT,timeout=720)
        except subprocess.TimeoutExpired:
            log.with_suffix('.exit_code').write_text('124\n')
            raise RuntimeError(f"ablation job timeout {job['group']} seed{job['seed']}; partial output={job['output']}; log={log}")
    exit_path=log.with_suffix('.exit_code');exit_path.write_text(str(p.returncode)+'\n')
    if p.returncode:raise RuntimeError(f"ablation job failed {job['group']} seed{job['seed']}: exit{p.returncode}; log={log}")
    return json.loads((Path(job['output'])/'assessment.json').read_text())


def matrix(args):
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,cfg_hash=train.load_runtime_config(args.config)
    if cfg['trainer']['timesteps']!=256 or cfg['ppo']['rollouts']!=128:raise ValueError('ablation screen must be256/128')
    identity=source_hashes();write(args.output/'source_identity.json',identity)
    args.dataset=args.output/'index.json';write(args.dataset,build_index([args.coarse]))
    write(args.output/'matrix_contract.json',{'variants':list(VARIANTS),'seeds':list(SEEDS),'jobs':26,'workers':4,
        'timesteps_per_job':256,'full_evaluation_frames':150,'job_timeout_s':720,'config_sha256':cfg_hash,
        'host':os.uname().nodename,'device':'cpu','base_config':cfg,'scope':'single_video_paired_screen_not_convergence_or_generalization'})
    start=time.monotonic();zero,ds=make_zero(args,cfg)
    print('ZERO '+json.dumps({k:zero[k] for k in ['objective','original_formal_objective','generated_collision_objective','full_group_costs']}),flush=True)
    jobs=[]
    for group in VARIANTS:
        for s in SEEDS:
            variant=json.loads(json.dumps(cfg));variant['environment']['reward_ablation']['drop_group']=group
            config_path=args.output/'configs'/f'{group}_seed{s}.yaml';config_path.parent.mkdir(exist_ok=True)
            config_path.write_text(yaml.safe_dump(variant,sort_keys=False))
            out=args.output/'runs'/group/f'seed_{s}'
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--mode','run','--config',str(config_path),
                     '--dataset',str(args.dataset),'--zero',str(args.output/'zero'),'--seed',str(s),'--group',group,'--output',str(out)]
            jobs.append({'group':group,'seed':s,'output':str(out),'log':str(args.output/'logs'/f'{group}_seed{s}.log'),'command':command})
    write(args.output/'job_manifest.json',jobs)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        iterator=iter(jobs);pending={}
        for _ in range(4):
            job=next(iterator);pending[pool.submit(run_job,job)]=job
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future)
                try:row=future.result()
                except Exception as e:
                    write(args.output/'blocked.json',{'failed_job':job,'error':str(e),'completed':len(results),'scope':'stop_new_dispatch_keep_partial_outputs'})
                    for f in pending:f.cancel()
                    raise
                results.append(row)
                print('RESULT '+json.dumps({'group':row['drop_group'],'seed':row['seed'],'J_full':row['objective'],'J_train':row['training_objective'],
                                          'collision_sum':row['collision']['inside_squared_hinge_sum_m2'],'pixel_p95_LR':[row['geometry'][s]['joint_pixel_p95'] for s in ['left','right']]}),flush=True)
                job=next(iterator,None)
                if job is not None:pending[pool.submit(run_job,job)]=job
    # Matching initial actor AND critic within each paired seed is mandatory.
    for s in SEEDS:
        entries=[r for r in results if r['seed']==s]
        if len({r['initial_parameter_hashes']['actor'] for r in entries})!=1 or len({r['initial_parameter_hashes']['critic'] for r in entries})!=1:raise RuntimeError('unmatched initial model parameters across reward variants')
    ordered=sorted(results,key=lambda r:(VARIANTS.index(r['drop_group']),r['seed']))
    write(args.output/'matrix_results.json',{'zero':zero,'runs':ordered,'paired_initial_parameters_match':True,
                                          'seconds_before_render':time.monotonic()-start})
    for group in VARIANTS:
        pair=[r for r in ordered if r['drop_group']==group]
        folder=args.output/'comparisons'/group;folder.mkdir(parents=True)
        render_comparison(folder,zero,pair[0],pair[1],ds[0],panel_labels=('Zero R6',f'{group} s0@256',f'{group} s1@256'))
        print('RENDER '+group,flush=True)
    if source_hashes()!=identity:raise RuntimeError('runtime changed during matrix')
    b=ds[0]
    for path,h in [(b.manifest_path,json.loads(args.dataset.read_text())['episodes'][0]['manifest_sha256']),
                   (b.array_path,b.coarse['arrays']['sha256']),(b.source.manifest_path,b.coarse['source_bundle_sha256'])]:
        if train._sha256(path)!=h:raise RuntimeError('matrix input changed')
    write(args.output/'completed.json',{'status':'complete_26_paired_full_action_reward_ablations_and_native_full_video_comparisons',
                                      'jobs':26,'variants':13,'seeds':list(SEEDS),'frames_each':150,'code_and_inputs_unchanged':True,
                                      'seconds':time.monotonic()-start})
    print('COMPLETE MATRIX seconds='+str(time.monotonic()-start),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['matrix','run','evaluate'],default='matrix')
    p.add_argument('--config',type=Path,default=ROOT/'configs/ppo_reward_ablation_coarse.yaml')
    p.add_argument('--coarse',type=Path);p.add_argument('--dataset',type=Path);p.add_argument('--zero',type=Path)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--group',default='full')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--provenance',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();seed(args.seed)
    if args.mode=='matrix':matrix(args)
    elif args.mode=='run':run(args)
    else:evaluate(args)
