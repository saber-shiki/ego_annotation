"""Replay-only PPO invariant diagnostic. Optimizer steps are forbidden.

Reconstruct the first paired rollout exactly, stop BEFORE the first PPO update,
and inspect the normalizer/GAE chain. Learned fixed actors are used only for
sensitivity tests, not as reconstructions of an unsaved historical update.
"""
from pathlib import Path
import argparse,copy,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from skrl.agents.torch.ppo.ppo import compute_gae
from tools.rl_experiments.reward_ablation_learning import seed,parameter_hash
import train


class CapturedBeforeUpdate(Exception):pass


def probability(policy,preprocessor,observations,actions):
    with torch.no_grad():
        mean,extra=policy.compute({'observations':preprocessor(observations)},role='policy')
        log_std=extra['log_std']
        if policy._g_clip_log_std:
            log_std=log_std.clamp(policy._g_min_log_std,policy._g_max_log_std)
        std=log_std.exp().expand_as(mean)
        log=torch.distributions.Normal(mean,std).log_prob(actions).sum(-1,keepdim=True)
    return mean,std,log


def compare(policy,old_scaler,new_scaler,x,a):
    m0,s0,l0=probability(policy,old_scaler,x,a);m1,s1,l1=probability(policy,new_scaler,x,a)
    delta=l1-l0
    analytical=(torch.log(s1/s0)+(s0.square()+(m0-m1).square())/(2*s1.square())-.5).sum(-1)
    return {'actor_parameters_frozen':True,'mean_action_difference_rms':float((m1-m0).square().mean().sqrt()),
            'analytic_conditional_KL_mean':float(analytical.mean()),'analytic_conditional_KL_max':float(analytical.max()),
            'logprob_change_abs_mean':float(delta.abs().mean()),'logprob_change_abs_max':float(delta.abs().max()),
            'ratio_outside_0_8_1_2_fraction':float(((delta<np.log(.8))|(delta>np.log(1.2))).float().mean()),
            'feature_normalized_difference_rms':float((new_scaler(x)-old_scaler(x)).square().mean().sqrt())}


def diagnostic(run,output,s):
    output.mkdir(parents=True,exist_ok=False)
    cfg,cfg_hash=train.load_runtime_config(run/'configs'/f'full_seed{s}.yaml')
    folder=run/'runs/full'/f'seed_{s}'
    old=json.loads((folder/'training_trace.json').read_text())[:128]
    initial_record=json.loads((folder/'initial_parameters.json').read_text())
    checkpoint_hash=train._sha256(folder/'agent.pt')
    env=train.build_env(cfg,dataset_path=run/'index.json',split='all',device='cpu')
    agent=train.build_agent(env,cfg,training=True,experiment_dir=output/'unused',experiment_name='no_optimizer_diagnostic')
    if parameter_hash(agent.policy)!=initial_record['actor'] or parameter_hash(agent.value)!=initial_record['critic']:
        raise RuntimeError('initial random model reconstruction differs')
    actor_before=parameter_hash(agent.policy);critic_before=parameter_hash(agent.value)
    captures={};traces=[];envstep=env.step
    def capture_step(action):
        result=envstep(action);info=result[-1]
        traces.append({'frame_idx':info['frame_idx'],'executed_delta':info['executed_delta'],'reward':float(result[1].item())})
        return result
    env.step=capture_step
    def forbidden_step(*args,**kwargs):raise RuntimeError('NO optimizer update allowed in this diagnostic')
    agent.optimizer.step=forbidden_step
    def stop_before_update(*,timestep,timesteps):
        for name in ['observations','actions','log_prob','values','rewards','terminated','truncated']:
            captures[name]=agent.memory.get_tensor_by_name(name).detach().clone()
        captures['last_observations']=agent._current_next_observations.detach().clone()
        raise CapturedBeforeUpdate()
    agent.update=stop_before_update
    try:train.build_trainer(env,agent,cfg,timesteps=128).train()
    except CapturedBeforeUpdate:pass
    else:raise RuntimeError('did not intercept before update')
    if len(traces)!=128:raise RuntimeError('incomplete diagnostic replay')
    for actual,expected in zip(traces,old):
        np.testing.assert_array_equal(actual['executed_delta'],np.asarray(expected['executed_delta'],np.float32))
        if actual['frame_idx']!=expected['frame_idx'] or actual['reward']!=expected['reward']:
            raise RuntimeError('first128 original replay mismatch')
    if actor_before!=parameter_hash(agent.policy) or critic_before!=parameter_hash(agent.value):raise RuntimeError('diagnostic changed trainable parameters')
    x=captures['observations'].reshape(128,-1);a=captures['actions'].reshape(128,-1)
    old_scaler=copy.deepcopy(agent._observation_preprocessor)
    initial_scaler=copy.deepcopy(old_scaler)
    updated_scaler=copy.deepcopy(old_scaler)
    for chunk in x.split(16):updated_scaler(chunk,train=True)
    _,_,lp=probability(agent.policy,old_scaler,x,a)
    initial_probability_error=float((lp-captures['log_prob'].reshape(-1,1)).abs().max())
    initial_control=compare(agent.policy,old_scaler,updated_scaler,x,a)
    saved=torch.load(folder/'agent.pt',map_location='cpu',weights_only=True)
    saved_policy=copy.deepcopy(agent.policy);saved_policy.load_state_dict(saved['policy'])
    saved_scaler=copy.deepcopy(old_scaler);saved_scaler.load_state_dict(saved['observation_preprocessor'])
    # Endpoint sensitivity only: this actor was never sampled using initial
    # scaler on these first128 actions. Analytic conditional KL avoids claiming
    # it is a historical PPO KL or evaluating sampled rewards as policy quality.
    learned_sensitivity=compare(saved_policy,initial_scaler,saved_scaler,x,a)
    unchanged_scaler_control=compare(saved_policy,saved_scaler,copy.deepcopy(saved_scaler),x,a)
    next_mean,_=agent.value.act({'observations':old_scaler(captures['last_observations']),'states':None},role='value')
    last_value=agent._value_preprocessor(next_mean,inverse=True).detach()
    actual_values=captures['values'];zero_values=torch.zeros_like(actual_values)
    endings=captures['terminated'];trunc=captures['truncated']
    def gae(reward,values,last):
        return compute_gae(rewards=reward,terminated=endings,truncated=trunc,values=values,last_values=last,
                           discount_factor=1.,lambda_coefficient=.95,time_limit_bootstrap=True)
    returns,advantage=gae(captures['rewards'],actual_values,last_value)
    critic_only_returns,critic_only_adv=gae(torch.zeros_like(captures['rewards']),actual_values,last_value)
    reward_only_returns,reward_only_adv=gae(captures['rewards'],zero_values,torch.zeros_like(last_value))
    def cosine(u,v):
        u=u.flatten().double();v=v.flatten().double();return float(torch.nn.functional.cosine_similarity(u,v,dim=0))
    group_sensitivity=[]
    for p in sorted((run/'runs').glob(f'*/seed_{s}/training_trace.json')):
        group=p.parents[1].name;t=json.loads(p.read_text())[:128]
        r=torch.tensor([q['reward'] for q in t],dtype=torch.float32).reshape(128,1,1)
        ret,adv=gae(r,actual_values,last_value)
        group_sensitivity.append({'group':group,'normalized_advantage_cosine_to_full':cosine(adv,advantage),
                                  'normalized_advantage_difference_rms':float((adv-advantage).square().mean().sqrt()),
                                  'first128_reward_sum':float(r.sum()),'first128_reward_std':float(r.std()),
                                  'return_rms':float(ret.square().mean().sqrt()),
                                  'original_initial_critic_unchanged':True})
    # Exact installed value-scaler ordering: values train=True THEN returns train=True.
    vscaler=copy.deepcopy(agent._value_preprocessor)
    value_encoded=vscaler(actual_values,train=True)
    after_values=copy.deepcopy(vscaler)
    return_encoded=vscaler(returns,train=True)
    values_in_return_scale=vscaler(actual_values)
    value_scale_mismatch=float((value_encoded-values_in_return_scale).abs().max())
    output_report={'scope':'no_optimizer_step_replay_diagnostic_not_new_training_or_annotation',
        'seed':s,'original_first128_actions_rewards_exact':True,'actor_and_critic_initial_identity_match':True,
        'actor_and_critic_parameters_unchanged':True,'optimizer_steps':0,
        'initial_raw_gaussian_stored_logprob_max_error':initial_probability_error,
        'initial_zero_mean_actor_normalizer_control':initial_control,
        'fixed_final_actor_initial_vs_final_normalizer_endpoint_sensitivity':learned_sensitivity,
        'fixed_final_actor_unchanged_normalizer_negative_control':unchanged_scaler_control,
        'normalizer_sensitivity_not_historical_firstupdate_attribution':True,
        'collected_observation_scaler_count':float(old_scaler.current_count),
        'simulated_first_epoch_sequential_minibatches_scaler_count':float(updated_scaler.current_count),
        'checkpoint_observation_scaler_count':float(saved_scaler.current_count),
        'advantage_full_vs_initial_critic_only_cosine':cosine(advantage,critic_only_adv),
        'advantage_full_vs_reward_only_cosine':cosine(advantage,reward_only_adv),
        'initial_critic_value_std':float(actual_values.std()),'initial_critic_value_mean':float(actual_values.mean()),
        'full_reward_std':float(captures['rewards'].std()),'group_firstrollout_advantage_sensitivity':group_sensitivity,
        'value_scaler_order_mismatch':{'encoded_old_values_vs_final_return_scale_max_difference':value_scale_mismatch,
            'values_encoded_at_count':float(after_values.current_count),'returns_encoded_at_count':float(vscaler.current_count),
            'description':'old values used for value clipping are encoded with different mean/variance from current predicted values and targets'},
        'source_checkpoint_sha256_unchanged':checkpoint_hash==train._sha256(folder/'agent.pt')}
    np.savez_compressed(output/'first128_replay_tensors.npz',**{k:v.cpu().numpy() for k,v in captures.items()})
    (output/'diagnostic.json').write_text(json.dumps(output_report,indent=2,allow_nan=False)+'\n')
    env.close();print(json.dumps(output_report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();seed(args.seed);diagnostic(args.run,args.output,args.seed)
