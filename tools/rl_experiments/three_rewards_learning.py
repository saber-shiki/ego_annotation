"""Bounded Milk R6 debug: three geometric rewards, two seeds, no budget extension."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from data.coarse_training import build_index, CoarseTrainingDataset
from data.bundle import file_sha256
from envs.three_rewards_coarse_env import ThreeRewardsCoarseEnv
from rewards.evaluator import RewardEvaluator
from rewards.three_rewards import GROUPS
from tools.rl_experiments.preprocessed_learning import write, source_hashes, export_npz, render_comparison
from tools.rl_experiments.generated_collision_learning import audit_and_csv
from tools.rl_experiments.reward_ablation_learning import seed, parameter_hash, metrics
from tools.rl_experiments.object_only_learning import optimizer_steps, rng_hash, environment_hash
import train
import rollout

BOUNDARIES = (150, 300, 450, 600)


def actual_motion(archive):
    result = {}
    dt = np.diff(archive['time_s'])
    for side in ('left', 'right'):
        v = np.diff(archive[side+'_vertices_world'].astype(float), axis=0) / dt[:, None, None]
        a = np.diff(v, axis=0) / ((dt[1:] + dt[:-1]) * .5)[:, None, None]
        result[side] = {'absolute_velocity_p95_mm_s': float(np.percentile(np.linalg.norm(v,axis=-1),95)*1000),
                        'absolute_acceleration_p95_mm_s2': float(np.percentile(np.linalg.norm(a,axis=-1),95)*1000)}
    v = np.diff(archive['object_translation_world'],axis=0)/dt[:,None]
    a = np.diff(v,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]
    result['object'] = {'absolute_translation_velocity_p95_mm_s':float(np.percentile(np.linalg.norm(v,axis=-1),95)*1000),
                        'absolute_translation_acceleration_p95_mm_s2':float(np.percentile(np.linalg.norm(a,axis=-1),95)*1000)}
    return result


def evaluate(args):
    rows=[];final=[];raws=[];build=rollout.build_env
    def instrument(*a,**kw):
        env=build(*a,**kw);raws.append(env._unwrapped);step=env.step
        def record(action):
            result=step(action);info=result[-1];raw=env._unwrapped;i=raw.state.position-1
            rows.append({'frame_idx':int(info['frame_idx']),'reward':float(result[1].item()),
                         'objective':info['objective'],'zero_relative_objective':info['zero_relative_objective'],
                         'group_costs':info['factor_ledger']['terms'],
                         'raw_action':info['raw_action'].tolist(),'executed_delta':info['executed_delta'].tolist(),
                         'contact_mean_probability':(1/(1+np.exp(-raw.state.contact_logits[i]))).mean(-1).tolist()})
            if result[2].any():final.append(info)
            if result[3].any():raise RuntimeError('unexpected truncated evaluation')
            return result
        env.step=record;return env
    rollout.build_env=instrument
    manifest,_=rollout.rollout(config_path=args.config,dataset_path=args.dataset,split='all',checkpoint_path=args.checkpoint,
        provenance_path=args.provenance,output_path=args.output/'rollout.json',device='cpu')
    if len(final)!=1 or len(rows)!=150:raise RuntimeError('full150 required')
    info=final[0]
    if not np.isclose(sum(x['reward'] for x in rows),-info['zero_relative_objective'],rtol=2e-5,atol=1e-6):raise RuntimeError('centered return telescope')
    m=json.loads(Path(manifest).read_text());art=next(x for x in m['episodes'][0]['artifacts'] if x['format']=='npz')
    path=Path(art['path']);path=path if path.is_absolute() else Path(manifest).parent/path
    with np.load(path,allow_pickle=False) as z:archive=dict(z)
    zero=json.loads((args.run/'zero/evaluation.json').read_text())
    with np.load(zero['npz'],allow_pickle=False) as z:base=dict(z)
    for k in ('frame_idx','time_s','T_world_camera','object_vertices_canonical','object_faces'):
        np.testing.assert_array_equal(archive[k],base[k])
    cfg,_=train.load_runtime_config(args.config)
    original=RewardEvaluator(cfg['environment']).objective(raws[0].state,150).document()
    collision=audit_and_csv(path,args.output)
    result={'seed':args.seed,'steps':args.steps,'objective':info['objective'],'zero_relative_objective':info['zero_relative_objective'],
            'group_costs':info['factor_ledger']['terms'],'final_ledger':info['factor_ledger'],'manifest':str(manifest),'npz':str(path),
            'original_formal_objective':original['objective'],'original_formal_ledger':original,
            'geometry':metrics(archive,base,rows,zero['contact_prior_probabilities']),
            'absolute_motion':actual_motion(archive),'collision':collision,'steps_trace':rows}
    write(args.output/'evaluation.json',result)
    print(json.dumps({k:result[k] for k in ('seed','steps','objective','zero_relative_objective','group_costs','original_formal_objective')}),flush=True)


def preflight(args):
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'index.json',build_index([args.coarse]));cfg,h=train.load_runtime_config(args.config)
    write(args.output/'run_config.json',{'config':cfg,'config_sha256':h,'boundaries':BOUNDARIES,'seeds':[0,1],
            'timesteps_per_seed':600,'host':os.uname().nodename,'device':'single_thread_CPU','runtime':str(ROOT),
            'source':str(args.coarse),'source_sha256':file_sha256(args.coarse),'scope':'new_minimal_system_debug_not_single_factor_ablation_or_GT'})
    write(args.output/'source_identity.json',source_hashes())
    ds=CoarseTrainingDataset(args.output/'index.json',split='all');raw=ThreeRewardsCoarseEnv(ds,cfg['environment']);raw.reset(seed=0)
    rewards=[]
    for _ in range(150):_,r,done,trunc,info=raw.step(np.zeros(118,np.float32));rewards.append(r)
    if not done or trunc or max(abs(x) for x in rewards)>1e-8:raise RuntimeError('zero reference replay reward mismatch')
    exported=raw.export_annotation(args.output/'zero/candidate.json');state=raw.state.copy();state.position=0
    zcost=info['objective'];zero_prob=(1/(1+np.exp(-state.contact_logits))).mean(-1)
    with np.load(exported['npz'],allow_pickle=False) as z:archive=dict(z)
    original=RewardEvaluator(cfg['environment']).objective(state,150).document()
    zero={'objective':zcost,'zero_relative_objective':0.,'group_costs':info['factor_ledger']['terms'],'final_ledger':info['factor_ledger'],
          'npz':exported['npz'],'original_formal_objective':original['objective'],'original_formal_ledger':original,
          'contact_prior_probabilities':zero_prob.tolist(),'absolute_motion':actual_motion(archive),'collision':audit_and_csv(exported['npz'],args.output/'zero')}
    write(args.output/'zero/evaluation.json',zero)
    from envs.dynamics import decode_candidate,action_gate_mask
    mask,_=action_gate_mask(state,raw.cfg)
    if not mask[:102].all():raise RuntimeError('frame0 hand actions gated')
    rows=[];reference=raw.evaluator.objective(state,150)
    for side in ['left','right']:
        for block,amount,coords in [('translation',.002,range(3)),('root',.01,range(3)),('articulation',.01,[3])]:
            for coordinate in coords:
                for sign in [-1,1]:
                    c=state.copy();d=np.zeros(118);d[c.episode.layout.block(f'hand.{side}.{block}').start+coordinate]=sign*amount
                    decode_candidate(c,d,raw.cfg);ledger=raw.evaluator.objective(c,150)
                    rows.append({'hand':side,'block':block,'coordinate':coordinate,'physical_delta':sign*amount,
                                 'changes':{g:ledger.terms[g]-reference.terms[g] for g in GROUPS}})
        if not any(r['changes']['nonpenetration_contact']<0 for r in rows if r['hand']==side and r['block']=='translation'):
            raise RuntimeError('no collision improving direction for '+side)
        if not all(r['changes']['reference_alignment']>0 for r in rows if r['hand']==side):raise RuntimeError('alignment not coupled')
    for block,amount in [('translation',.002),('rotation',.01)]:
        c=state.copy();d=np.zeros(118);d[c.episode.layout.block('object.object:0.'+block).start]=amount
        decode_candidate(c,d,raw.cfg);ledger=raw.evaluator.objective(c,150)
        if ledger.terms['reference_alignment']<=reference.terms['reference_alignment']:raise RuntimeError('object alignment uncoupled')
        rows.append({'object_block':block,'changes':{g:ledger.terms[g]-reference.terms[g] for g in GROUPS}})
    c=state.copy();c.contact_logits[0]-=20;c.term_cache.pop(0,None);cost=raw.evaluator.objective(c,150)
    if not np.isclose(cost.terms['nonpenetration_contact'],reference.terms['nonpenetration_contact'],atol=1e-12):raise RuntimeError('probability bypass')
    write(args.output/'controlled_actions.json',{'both_hands_unfrozen':True,'reference_not_GT':True,'source_authority_unpromoted':True,
         'contact_probability_cannot_bypass_collision':True,'directions':rows})
    raw.close();write(args.output/'preflight_completed.json',{'status':'geometric_mechanisms_coupled','zero_reward_max_abs':max(abs(x) for x in rewards),'zero_groups':zero['group_costs']})
    print('PREFLIGHT '+json.dumps({'zero_J':zcost,'zero_groups':zero['group_costs'],'controls':len(rows)}),flush=True)


def train_seed(args):
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,h=train.load_runtime_config(args.config);identity=source_hashes()
    env=train.build_env(cfg,dataset_path=args.run/'index.json',split='all',device='cpu')
    agent=train.build_agent(env,cfg,training=True,experiment_dir=args.output/'unused',experiment_name='three_rewards')
    write(args.output/'initial_parameters.json',{'actor':parameter_hash(agent.policy),'critic':parameter_hash(agent.value)})
    trace=[];step=env.step
    def record(action):
        result=step(action);info=result[-1]
        trace.append({'frame_idx':int(info['frame_idx']),'reward':float(result[1].item()),'objective':info['objective'],
                      'zero_relative_objective':info['zero_relative_objective'],'group_costs':info['factor_ledger']['terms'],
                      'raw_action':info['raw_action'].tolist(),'executed_delta':info['executed_delta'].tolist(),'terminated':bool(result[2].any())})
        return result
    env.step=record;updates=[];old_update=agent.update;checks=[]
    def distributions(o):
        mean,extra=agent.policy.compute({'observations':agent._observation_preprocessor(o)},role='policy')
        std=extra['log_std'].clamp(-5,-2).exp().expand_as(mean)
        return mean,std
    def update(*,timestep,timesteps):
        mem=agent.memory;observations=mem.get_tensor_by_name('observations').reshape(150,-1).clone();actions=mem.get_tensor_by_name('actions').reshape(150,-1).clone()
        recorded=mem.get_tensor_by_name('log_prob').reshape(150,1).clone();values=mem.get_tensor_by_name('values').reshape(150,1).clone()
        with torch.no_grad():
            mean0,std0=distributions(observations)
            lp=torch.distributions.Normal(mean0,std0).log_prob(actions).sum(-1,keepdim=True)
            predicted,_=agent.value.compute({'observations':agent._observation_preprocessor(observations)},role='value')
            prob_error=float((lp-recorded).abs().max());value_error=float((predicted-values).abs().max())
            clip=float(((predicted-values).abs()>.2).float().mean())
        if prob_error>1e-4 or value_error>1e-4 or clip!=0.:raise RuntimeError('before-update PPO coordinates inconsistent')
        preprocessors=rollout._snapshot_preprocessors(agent);before=optimizer_steps(agent)
        old_update(timestep=timestep,timesteps=timesteps)
        rollout._assert_preprocessors_unchanged(agent,preprocessors)
        with torch.no_grad():
            mean1,std1=distributions(observations)
            kl=(torch.log(std1/std0)+(std0.square()+(mean0-mean1).square())/(2*std1.square())-.5).sum(-1)
            log=torch.distributions.Normal(mean1,std1).log_prob(actions).sum(-1,keepdim=True)-recorded
            row={'step':timestep+1,'optimizer_steps':optimizer_steps(agent)-before,'before_update_raw_logprob_max_error':prob_error,
                 'before_update_value_unit_max_error':value_error,'before_update_value_clip_fraction':clip,
                 'exact_conditional_Gaussian_KL_mean':float(kl.mean()),'exact_conditional_Gaussian_KL_max':float(kl.max()),
                 'post_update_clip_fraction':float(((log<np.log(.9))|(log>np.log(1.1))).float().mean()),
                 'policy_std_mean':float(agent.policy.log_std_parameter.exp().mean()),'preprocessor_unchanged':True}
        updates.append(row);print('UPDATE '+json.dumps(row),flush=True)
    agent.update=update;post=agent.post_interaction
    def callback(*a,**kw):
        post(*a,**kw);boundary=int(kw['timestep'])+1
        if boundary not in BOUNDARIES:return
        folder=args.output/'checkpoints'/f'step_{boundary:04d}';folder.mkdir(parents=True)
        train.assert_agent_checkpoint_finite(agent);cp=folder/'agent.pt';agent.save(str(cp));prov=folder/'agent.provenance.json'
        write(prov,train.checkpoint_provenance(checkpoint=cp,config_sha256=h,cfg=cfg,env=env,split='all'))
        before=(rng_hash(),environment_hash(env))
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--mode','evaluate','--config',str(args.config),'--run',str(args.run),
                 '--dataset',str(args.run/'index.json'),'--checkpoint',str(cp),'--provenance',str(prov),'--seed',str(args.seed),
                 '--steps',str(boundary),'--output',str(folder/'evaluation')]
        p=subprocess.run(command,capture_output=True,text=True,timeout=480)
        (folder/'evaluation.stdout.log').write_text(p.stdout);(folder/'evaluation.stderr.log').write_text(p.stderr)
        if p.returncode:raise RuntimeError('independent eval failed: '+p.stderr[-2000:])
        if before!=(rng_hash(),environment_hash(env)):raise RuntimeError('eval mutated training RNG or state')
        d=json.loads((folder/'evaluation/evaluation.json').read_text());row={k:d[k] for k in ('seed','steps','objective','zero_relative_objective','group_costs','original_formal_objective','geometry','absolute_motion','collision')}
        row.update(checkpoint=str(cp),evaluation=str(folder/'evaluation/evaluation.json'),training_RNG_state_unchanged=True)
        checks.append(row);write(folder/'assessment.json',row)
        print('CHECKPOINT '+json.dumps({k:row[k] for k in ('seed','steps','objective','zero_relative_objective','group_costs')}),flush=True)
    agent.post_interaction=callback
    start=time.monotonic();train.build_trainer(env,agent,cfg).train()
    if len(checks)!=4 or len(trace)!=600 or len(updates)!=4:raise RuntimeError('budget incomplete')
    if source_hashes()!=identity:raise RuntimeError('snapshot changed')
    write(args.output/'training_trace.json',trace);write(args.output/'updates.json',updates)
    write(args.output/'summary.json',{'seed':args.seed,'seconds':time.monotonic()-start,'checkpoints':checks,'best_trained':min(checks,key=lambda x:x['objective']),
                                    'scope':'short same_video_debug','timesteps':600})
    env.close();write(args.output/'completed.json',{'status':'600steps_4_full150_evals_complete','source_unchanged':True})
    print('COMPLETE seed'+str(args.seed),flush=True)


def render(args):
    summaries=[json.loads((args.run/f'seed_{s}/summary.json').read_text()) for s in (0,1)]
    ds=CoarseTrainingDataset(args.run/'index.json',split='all');zero=json.loads((args.run/'zero/evaluation.json').read_text())
    for label,rows in [('last',[s['checkpoints'][-1] for s in summaries]),('best',[s['best_trained'] for s in summaries])]:
        out=args.run/'comparisons'/label;out.mkdir(parents=True,exist_ok=False)
        render_comparison(out,zero,rows[0],rows[1],ds[0],panel_labels=('zero R6',f'seed0@{rows[0]["steps"]}',f'seed1@{rows[1]["steps"]}'))
    if source_hashes()!=json.loads((args.run/'source_identity.json').read_text()):raise RuntimeError('snapshot changed')
    write(args.run/'completed.json',{'status':'bounded_600x2_native150_render_complete','timesteps':1200,'source_unchanged':True,'annotation_improvement_not_yet_judged':True})
    print('RENDER COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['preflight','train','evaluate','render'],required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/ppo_three_rewards_coarse.yaml');p.add_argument('--coarse',type=Path)
    p.add_argument('--run',type=Path);p.add_argument('--dataset',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--provenance',type=Path);p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=600)
    args=p.parse_args();seed(args.seed)
    {'preflight':preflight,'train':train_seed,'evaluate':evaluate,'render':render}[args.mode](args)
