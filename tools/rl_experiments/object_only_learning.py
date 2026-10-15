"""600-step object-first PPO experiment. Raw complete J is always the verdict."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from data.bundle import file_sha256
from data.coarse_training import build_index, CoarseTrainingDataset
from envs.object_only_coarse_env import ObjectOnlyCoarseEnv, CONTRACT
from tools.rl_experiments.preprocessed_learning import write, seed, source_hashes, family, render_comparison
import train
import rollout

BOUNDARIES = (150, 300, 450, 600)


def rng_hash():
    import random
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state())) + torch.get_rng_state().numpy().tobytes()).hexdigest()


def environment_hash(env):
    state = env._env.state
    arrays = [state.contact_logits, state.applied]
    for name in ('hand_root','hand_articulation','hand_translation','hand_vertices','hand_joints','object_rotation','object_translation'):
        arrays.extend(getattr(state,name).values())
    h=hashlib.sha256()
    for a in arrays:h.update(np.asarray(a).tobytes())
    h.update(str((state.position,state.previous_objective)).encode())
    return h.hexdigest()


def geometry_check(path, zero_path, cfg):
    with np.load(path,allow_pickle=False) as z:current=dict(z)
    with np.load(zero_path,allow_pickle=False) as z:zero=dict(z)
    n=len(zero['frame_idx'])
    for key in ('frame_idx','time_s','T_world_camera','intrinsics_render','object_vertices_canonical','object_faces'):
        np.testing.assert_array_equal(current[key],zero[key])
    for side in ('left','right'):
        for suffix in ('vertices_world','joints_world','faces','valid','display_valid'):
            np.testing.assert_array_equal(current[side+'_'+suffix],zero[side+'_'+suffix])
    angles=Rotation.from_matrix(current['object_rotation']@zero['object_rotation'].transpose(0,2,1)).magnitude()
    trans=np.linalg.norm(current['object_translation_world']-zero['object_translation_world'],axis=1)
    bounds=cfg['environment']['object_only']
    if angles.max()>np.deg2rad(bounds['max_rotation_deg'])+1e-6 or trans.max()>bounds['max_translation_m']+1e-8:
        raise RuntimeError('exported candidate exceeded frame-relative trust ball')
    return {'frames':n,'full_native_faces':len(current['object_faces']),'hands_bitwise_equal_to_zero':True,
            'object_rotation_correction_max_deg':float(np.degrees(angles.max())),
            'object_translation_correction_max_mm':float(trans.max()*1000)}


def evaluate(args):
    records=[];final=[];old=rollout.build_env
    def build(*a,**kw):
        env=old(*a,**kw);step=env.step
        def capture(action):
            result=step(action);info=result[-1]
            records.append({'frame_idx':int(info['frame_idx']),'objective':float(info['objective']),
                'relative_objective':float(info['objective_relative_to_zero']),'relative_reward':float(result[1].item()),
                'raw_latent_action':info['raw_action'].tolist(),'executed_delta':info['executed_delta'].tolist()})
            if bool(result[2].any()):final.append(info)
            if bool(result[3].any()):raise RuntimeError('evaluation truncated')
            return result
        env.step=capture
        return env
    rollout.build_env=build
    manifest,_=rollout.rollout(config_path=args.config,dataset_path=args.dataset,split='all',
        checkpoint_path=args.checkpoint,provenance_path=args.provenance,output_path=args.output/'rollout.json',device='cpu')
    if len(final)!=1:raise RuntimeError('expected one complete natural episode')
    info=final[0];ledger=info['factor_ledger'];scale=ledger['reward_scale']
    reward=sum(r['relative_reward'] for r in records)
    if not np.isclose(reward,-scale*(ledger['objective']-info['zero_final_objective']),atol=1e-7,rtol=1e-5):raise RuntimeError('centered evaluation return failed')
    write(args.output/'evaluation.json',{'objective':ledger['objective'],'zero_objective':info['zero_final_objective'],
        'relative_return':reward,'named_families':family(ledger['terms']),'final_ledger':ledger,'steps':records,'manifest':str(manifest)})
    print(json.dumps({'objective':ledger['objective'],'relative_return':reward}),flush=True)


def evaluate_checkpoint(args,agent,env,cfg,cfg_hash,step):
    folder=args.output/'checkpoints'/f'step_{step:04d}';folder.mkdir(parents=True)
    train.assert_agent_checkpoint_finite(agent)
    checkpoint=folder/'agent.pt';agent.save(str(checkpoint))
    provenance=folder/'agent.provenance.json'
    write(provenance,train.checkpoint_provenance(checkpoint=checkpoint,config_sha256=cfg_hash,cfg=cfg,env=env,split='all'))
    before=(rng_hash(),environment_hash(env))
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--mode','evaluate','--config',str(args.config),
        '--dataset',str(args.dataset),'--checkpoint',str(checkpoint),'--provenance',str(provenance),'--output',str(folder/'evaluation')]
    completed=subprocess.run(command,capture_output=True,text=True,timeout=700)
    (folder/'evaluation.stdout.log').write_text(completed.stdout);(folder/'evaluation.stderr.log').write_text(completed.stderr)
    if completed.returncode:raise RuntimeError('checkpoint evaluation failed: '+completed.stderr[-3000:])
    if before!=(rng_hash(),environment_hash(env)):raise RuntimeError('subprocess evaluation changed training RNG/environment')
    value=json.loads((folder/'evaluation/evaluation.json').read_text())
    from tools.rl_experiments.preprocessed_learning import export_npz
    row={'steps':step,'objective':value['objective'],'named_families':value['named_families'],
         'relative_return':value['relative_return'],'checkpoint':str(checkpoint),
         'evaluation':str(folder/'evaluation/evaluation.json'),'training_rng_and_state_unchanged':True,
         'policy_latent_std':agent.policy.log_std_parameter.exp().detach().cpu().tolist()}
    row['geometry']=geometry_check(export_npz(row),args.output/'zero/candidate.npz',cfg)
    zero=json.loads((args.output/'zero/evaluation.json').read_text())
    for key,val in zero['named_families'].items():
        if key.startswith('hand/') and not np.isclose(value['named_families'][key],val,atol=1e-9,rtol=1e-9):raise RuntimeError('object policy changed hand objective')
    write(folder/'assessment.json',row);print('CHECKPOINT '+json.dumps(row),flush=True)
    return row


def optimizer_steps(agent):
    return max([int(v.get('step',0)) for v in agent.optimizer.state.values()] or [0])


def experiment(args):
    args.output.mkdir(parents=True,exist_ok=False)
    args.dataset=args.output/'index.json';write(args.dataset,build_index([args.coarse]))
    cfg,cfg_hash=train.load_runtime_config(args.config)
    if cfg['trainer']['timesteps']!=600 or cfg['ppo']['rollouts']!=150:raise ValueError('bounded experiment is exactly 600/150')
    identity=source_hashes();write(args.output/'source_identity.json',identity)
    write(args.output/'run_config.json',{'config':cfg,'config_sha256':cfg_hash,'scope':'same_video_not_generalization',
        'host':os.uname().nodename,'device':'cpu','source_root':str(ROOT),'coarse':str(args.coarse),
        'coarse_sha256':file_sha256(args.coarse),'seed':0,'joint_intervention_not_single_factor_ablation':True})
    dataset=CoarseTrainingDataset(args.dataset,split='all')
    raw=ObjectOnlyCoarseEnv(dataset,cfg['environment'],'cpu');obs,_=raw.reset(seed=0)
    count=raw.state.episode.frame_count
    if count!=150:raise RuntimeError('rollout must match exact 150-frame milk timeline')
    initial=raw.state.copy();rewards=[]
    for i in range(count):_,r,done,truncated,info=raw.step(np.zeros(6,np.float32));rewards.append(r)
    if not done or truncated or not np.allclose(rewards,0,atol=1e-10):raise RuntimeError('zero-centered baseline failed')
    export=raw.export_annotation(args.output/'zero/candidate.json')
    zero={'objective':info['objective'],'named_families':family(info['factor_ledger']['terms']),
          'final_ledger':info['factor_ledger'],'npz':export['npz'],'frames':count,'action_dim':6,'observation_dim':len(obs),
          'zero_centered_return':sum(rewards)}
    write(args.output/'zero/evaluation.json',zero);print('ZERO '+json.dumps({k:v for k,v in zero.items() if k not in ('final_ledger','named_families')}),flush=True)
    # Physical preflight: perturb each latent coordinate with both signs over a
    # three-frame prefix. No full/old hand coordinates exist in this policy.
    with np.load(export['npz'],allow_pickle=False) as z:
        zero_hands={side:np.array(z[side+'_vertices_world']) for side in ('left','right')}
    tests=[]
    for coordinate in range(6):
        for sign in (-1.,1.):
            raw.reset(seed=0);latents=np.zeros(6,np.float32);latents[coordinate]=sign*.1
            vals=[]
            for i in range(3):_,r,_,_,item=raw.step(latents if i==0 else np.zeros(6,np.float32));vals.append(r)
            if not np.isclose(sum(vals),-1e-5*item['objective_relative_to_zero'],atol=1e-10):raise RuntimeError('controlled prefix return failed')
            for side in ('left','right'):
                np.testing.assert_array_equal(raw.state.hand_vertices[side][:3],zero_hands[side][:3])
            tests.append({'coordinate':coordinate,'latent':float(latents[coordinate]),'prefix_relative_objective':item['objective_relative_to_zero'],
                          'relative_return':sum(vals),'raw_latent_shape':list(item['raw_action'].shape)})
    if not any(abs(t['prefix_relative_objective'])>1e-8 for t in tests):raise RuntimeError('object controls did not couple physical reward')
    write(args.output/'controlled_actions.json',tests)
    del raw,initial
    env=train.build_env(cfg,dataset_path=args.dataset,split='all',device='cpu',dataset=dataset)
    old=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_coarse_r6_learning1024_20261001_r2/checkpoints/step_0256')
    try:rollout._verify_checkpoint_contract(provenance=json.loads((old/'agent.provenance.json').read_text()),checkpoint=old/'agent.pt',cfg=cfg,env=env)
    except ValueError as e:write(args.output/'old_checkpoint_rejection.json',{'rejected':True,'reason':str(e)})
    else:raise RuntimeError('118D policy accepted for six-dimensional experiment')
    agent=train.build_agent(env,cfg,training=True,experiment_dir=args.output/'runs',experiment_name='object_only')
    initial_policy={k:v.detach().clone() for k,v in agent.policy.state_dict().items()}
    rows=[];original_step=env.step
    def record(action):
        result=original_step(action);info=result[-1]
        rows.append({'frame_idx':int(info['frame_idx']),'objective':float(info['objective']),
            'relative_objective':float(info['objective_relative_to_zero']),'reward':float(result[1].item()),
            'terminated':bool(result[2].any()),'raw_latent':info['raw_action'].tolist(),'executed_delta':info['executed_delta'].tolist()})
        return result
    env.step=record
    update_rows=[];tracked=[];old_track=agent.track_data
    def track(tag,value):
        if tag.startswith(('Loss /','Policy /')):tracked.append({'tag':tag,'value':float(value)})
        return old_track(tag,value)
    agent.track_data=track
    old_update=agent.update
    def update(*,timestep,timesteps):
        memory=agent.memory
        rew=memory.get_tensor_by_name('rewards').detach().clone()
        val=memory.get_tensor_by_name('values').detach().clone()
        old_log=memory.get_tensor_by_name('log_prob').detach().clone()
        sampled_actions=memory.get_tensor_by_name('actions').detach().clone()
        observations=memory.get_tensor_by_name('observations').detach().clone()
        before_steps=optimizer_steps(agent)
        old_update(timestep=timestep,timesteps=timesteps)
        # GaussianMixin samples even when taken_actions is supplied. Preserve
        # training RNG around this read-only post-update likelihood diagnostic.
        diagnostic_rng=torch.get_rng_state().clone()
        with torch.no_grad():
            x=observations.reshape(-1,observations.shape[-1]);a=sampled_actions.reshape(-1,6)
            _,outputs=agent.policy.act({'observations':agent._observation_preprocessor(x),'taken_actions':a},role='policy')
            difference=outputs['log_prob']-old_log.reshape(-1,1)
            ratio=torch.exp(difference)
            adv=memory.get_tensor_by_name('advantages')
        torch.set_rng_state(diagnostic_rng)
        update_rows.append({'timestep':int(timestep)+1,'optimizer_steps':optimizer_steps(agent)-before_steps,
            'approx_KL_post_update':float(((ratio-1)-difference).mean()),
            'clip_fraction_post_update':float((torch.abs(ratio-1)>.1).float().mean()),
            'rewards_mean':float(rew.mean()),'rewards_std':float(rew.std()),'rewards_sum':float(rew.sum()),
            'old_values_mean':float(val.mean()),'old_values_std':float(val.std()),
            'normalized_advantage_mean':float(adv.mean()),'normalized_advantage_std':float(adv.std()),
            'KL_is_post_update_diagnostic_not_internal_early_stop_measurement':True})
        write(args.output/'updates'/f'update_{int(timestep)+1:04d}.json',update_rows[-1])
    agent.update=update
    checks=[];old_post=agent.post_interaction
    def post(*a,**kw):
        old_post(*a,**kw)
        t=int(kw['timestep'] if 'timestep' in kw else a[0])+1
        if t in BOUNDARIES:checks.append(evaluate_checkpoint(args,agent,env,cfg,cfg_hash,t))
    agent.post_interaction=post
    start=time.monotonic();train.build_trainer(env,agent,cfg).train()
    train.assert_agent_checkpoint_finite(agent)
    if not train.finite_policy_parameter_changed(agent,initial_policy):raise RuntimeError('no finite policy updates')
    if len(checks)!=4:raise RuntimeError('incomplete checkpoint boundaries')
    if source_hashes()!=identity:raise RuntimeError('source changed during frozen experiment')
    bundle=dataset[0]
    for path,expected in [(bundle.array_path,bundle.coarse['arrays']['sha256']),
                          (bundle.source.manifest_path,bundle.coarse['source_bundle_sha256']),
                          (bundle.manifest_path,json.loads((args.output/'index.json').read_text())['episodes'][0]['manifest_sha256'])]:
        if file_sha256(path)!=expected:raise RuntimeError('frozen input changed during experiment')
    best=min(checks,key=lambda r:r['objective'])
    summary={'scope':'same_video_not_generalization','action_contract':CONTRACT,'timesteps':600,
        'zero_objective':zero['objective'],'checkpoints':checks,'best_trained_checkpoint':best,
        'improves_zero':best['objective']<zero['objective'],'learning_seconds':time.monotonic()-start,
        'all_hands_unchanged':True,'joint_intervention_not_single_factor_ablation':True,'annotation_quality_verified':False}
    write(args.output/'training_steps.json',rows);write(args.output/'ppo_update_diagnostics.json',{'updates':update_rows,'tracked_metrics':tracked})
    write(args.output/'summary.json',summary)
    render_comparison(args.output,zero,best,checks[-1],bundle)
    env.close()
    write(args.output/'completed.json',{'status':'complete_600step_experiment_full_render','frames':count,
        'improves_raw_zero_objective':summary['improves_zero'],'source_and_code_unchanged':True})
    print('COMPLETE '+json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['experiment','evaluate'],default='experiment')
    p.add_argument('--config',type=Path,default=ROOT/'configs/ppo_object_only_coarse.yaml')
    p.add_argument('--coarse',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--provenance',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();seed()
    if a.mode=='experiment':experiment(a)
    else:evaluate(a)
