"""Bounded original full-action PPO with native generated-mesh collision reward."""
from __future__ import annotations
import argparse
import csv
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
from data.bundle import file_sha256
from data.coarse_training import build_index, CoarseTrainingDataset
from envs.generated_collision_coarse_env import GeneratedCollisionCoarseEnv
from tools.rl_experiments.preprocessed_learning import write, seed, source_hashes, family, export_npz, render_comparison
from tools.rl_experiments.object_only_learning import rng_hash, environment_hash, optimizer_steps
from utils.generated_collision import CONTRACT, audit_archive
import train
import rollout

BOUNDARIES = (256, 512, 768, 1024)


def objective_parts(ledger):
    terms=ledger['terms']
    original=sum(v for k,v in terms.items() if not k.startswith('hypothesis/'))
    collision=sum(v for k,v in terms.items() if k.startswith('hypothesis/'))
    return {'original_formal_objective':original,'generated_collision_objective':collision,'objective':ledger['objective']}


def audit_and_csv(path, folder):
    report=audit_archive(path)
    write(folder/'collision_audit.json',report)
    with (folder/'collision_frames.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['frame','time_s','hand','detected_crossing_or_inside','inside_vertices_0.1mm','inside_face_centers_0.1mm','inside_samples_1mm','max_inside_mm','crossing_edges'])
        for row in report['frames']:
            for side,stats in row['hands'].items():
                w.writerow([row['frame_idx'],row['time_s'],side,stats['state'],stats.get('inside_vertex_count_over_0_1mm',0),stats.get('inside_face_center_count_over_0_1mm',0),stats.get('inside_sample_count_over_1mm',0),stats.get('max_inside_depth_mm',0),stats.get('edge_crossings',0)])
    return {k:v for k,v in report.items() if k!='frames'}


def evaluate(args):
    rows=[];final=[];old=rollout.build_env
    def build(*a,**kw):
        env=old(*a,**kw);step=env.step
        def record(action):
            result=step(action);info=result[-1]
            rows.append({'frame_idx':int(info['frame_idx']),**objective_parts(info['factor_ledger']),
                         'reward':float(result[1].item()),'raw_action':info['raw_action'].tolist(),
                         'executed_delta':info['executed_delta'].tolist()})
            if bool(result[2].any()):final.append(info['factor_ledger'])
            if bool(result[3].any()):raise RuntimeError('unexpected truncated evaluation')
            return result
        env.step=record
        return env
    rollout.build_env=build
    manifest,_=rollout.rollout(config_path=args.config,dataset_path=args.dataset,split='all',
        checkpoint_path=args.checkpoint,provenance_path=args.provenance,output_path=args.output/'rollout.json',device='cpu')
    if len(final)!=1:raise RuntimeError('expected exactly one full episode')
    ledger=final[0]
    if not np.isclose(sum(r['reward'] for r in rows),-ledger['reward_scale']*ledger['objective'],rtol=1e-5,atol=1e-6):raise RuntimeError('return telescope failed')
    result={**objective_parts(ledger),'named_families':family(ledger['terms']),'final_ledger':ledger,'steps':rows,'manifest':str(manifest)}
    write(args.output/'evaluation.json',result)
    collision=audit_and_csv(export_npz({'evaluation':str(args.output/'evaluation.json')}),args.output)
    write(args.output/'collision_summary.json',collision)
    print(json.dumps({**objective_parts(ledger),'collision':collision}),flush=True)


def checkpoint(args,agent,env,cfg,cfg_hash,step):
    folder=args.output/'checkpoints'/f'step_{step:04d}';folder.mkdir(parents=True)
    train.assert_agent_checkpoint_finite(agent)
    checkpoint_path=folder/'agent.pt';agent.save(str(checkpoint_path))
    provenance=folder/'agent.provenance.json'
    write(provenance,train.checkpoint_provenance(checkpoint=checkpoint_path,config_sha256=cfg_hash,cfg=cfg,env=env,split='all'))
    before=(rng_hash(),environment_hash(env))
    p=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--mode','evaluate','--config',str(args.config),
        '--dataset',str(args.dataset),'--checkpoint',str(checkpoint_path),'--provenance',str(provenance),'--output',str(folder/'evaluation')],capture_output=True,text=True,timeout=800)
    (folder/'evaluation.stdout.log').write_text(p.stdout);(folder/'evaluation.stderr.log').write_text(p.stderr)
    if p.returncode:raise RuntimeError('independent checkpoint evaluation failed: '+p.stderr[-3000:])
    if before!=(rng_hash(),environment_hash(env)):raise RuntimeError('evaluation changed training RNG/state')
    value=json.loads((folder/'evaluation/evaluation.json').read_text())
    row={'steps':step,**objective_parts(value['final_ledger']),'named_families':value['named_families'],
         'checkpoint':str(checkpoint_path),'evaluation':str(folder/'evaluation/evaluation.json'),
         'collision':json.loads((folder/'evaluation/collision_summary.json').read_text()),'training_rng_and_state_unchanged':True,
         'policy_std_mean':float(agent.policy.log_std_parameter.exp().mean().detach())}
    write(folder/'assessment.json',row)
    print('CHECKPOINT '+json.dumps({k:v for k,v in row.items() if k not in ('named_families','collision')}),flush=True)
    print('COLLISION '+json.dumps({k:row['collision'][k] for k in ['affected_frame_count','max_inside_depth_mm','inside_squared_hinge_sum_m2','total_crossing_edges_over_all_hand_frames']}),flush=True)
    return row


def controlled_preflight(raw):
    raw.reset(seed=0);initial=raw.state.copy();layout=initial.episode.layout
    from envs.dynamics import decode_candidate, action_gate_mask
    mask,_=action_gate_mask(initial,raw.cfg)
    if not mask[:102].all():raise RuntimeError('present frame0 hands are action-frozen')
    base=objective_parts(raw.evaluator.objective(initial,1).document());rows=[]
    for side in ('left','right'):
        for block,coordinates in [('translation',range(3)),('root',range(3)),('articulation',[3])]:
            for coordinate in coordinates:
                for sign in (-1,1):
                    candidate=initial.copy();delta=np.zeros(118)
                    amount=.002 if block=='translation' else .01
                    delta[layout.block(f'hand.{side}.{block}').start+coordinate]=sign*amount
                    decode_candidate(candidate,delta,raw.cfg,'cpu')
                    values=objective_parts(raw.evaluator.objective(candidate,1).document())
                    rows.append({'hand':side,'block':block,'coordinate':coordinate,'physical_delta':sign*amount,
                                 'collision_change':values['generated_collision_objective']-base['generated_collision_objective'],
                                 'original_objective_change':values['original_formal_objective']-base['original_formal_objective'],
                                 'hand_geometry_max_change_m':float(np.linalg.norm(candidate.hand_vertices[side][0]-initial.hand_vertices[side][0],axis=1).max())})
        chosen=[r for r in rows if r['hand']==side and r['block']=='translation']
        if not any(r['collision_change']<0 for r in chosen):raise RuntimeError('no local collision improvement direction for '+side)
    # Full sampler contact block is real; its probability cannot disable body barrier.
    for side in ('left','right'):
        candidate=initial.copy();delta=np.zeros(118);delta[layout.block(f'contact.{side}').slice]=-20
        decode_candidate(candidate,delta,raw.cfg,'cpu')
        value=objective_parts(raw.evaluator.objective(candidate,1).document())
        if not np.isclose(value['generated_collision_objective'],base['generated_collision_objective'],atol=1e-7):raise RuntimeError('contact logit can bypass collision')
    return {'base':base,'signed_source_authority_unchanged':False if initial.episode.objects['object:0']['signed_geometry_ready'] else True,
            'both_hands_unfrozen':True,'contact_probability_cannot_bypass_body_barrier':True,'directions':rows}


def experiment(args):
    args.output.mkdir(parents=True,exist_ok=False)
    args.dataset=args.output/'index.json';write(args.dataset,build_index([args.coarse]))
    cfg,cfg_hash=train.load_runtime_config(args.config)
    identity=source_hashes();write(args.output/'source_identity.json',identity)
    write(args.output/'run_config.json',{'config':cfg,'config_sha256':cfg_hash,'source_root':str(ROOT),
        'coarse':str(args.coarse),'coarse_sha256':file_sha256(args.coarse),'seed':0,'device':'cpu',
        'host':os.uname().nodename,'scope':'single_video_generated_geometry_trust_experiment_not_GT_or_generalization'})
    ds=CoarseTrainingDataset(args.dataset,split='all')
    raw=GeneratedCollisionCoarseEnv(ds,cfg['environment'],'cpu');obs,_=raw.reset(seed=0)
    start=time.monotonic();count=raw.state.episode.frame_count
    initial=raw.state.copy();rewards=[]
    for i in range(count):_,r,done,truncated,info=raw.step(np.zeros(118,np.float32));rewards.append(r)
    if not done or truncated or not np.isclose(sum(rewards),-1e-5*info['objective'],rtol=1e-7):raise RuntimeError('zero replay telescope failed')
    for side in ('left','right'):
        if np.linalg.norm(raw.state.hand_vertices[side]-initial.hand_vertices[side],axis=-1).max()>1e-4:raise RuntimeError('zero MANO replay differs')
    np.testing.assert_array_equal(raw.state.object_rotation['object:0'],initial.object_rotation['object:0'])
    exported=raw.export_annotation(args.output/'zero/candidate.json')
    zero={**objective_parts(info['factor_ledger']),'named_families':family(info['factor_ledger']['terms']),
          'final_ledger':info['factor_ledger'],'npz':exported['npz'],'frames':count,'action_dim':118,'observation_dim':len(obs)}
    zero['collision']=audit_and_csv(exported['npz'],args.output/'zero')
    write(args.output/'zero/evaluation.json',zero)
    print('ZERO '+json.dumps({**objective_parts(info['factor_ledger']),'collision_frames':zero['collision']['affected_frame_count'],'seconds':time.monotonic()-start}),flush=True)
    write(args.output/'controlled_actions.json',controlled_preflight(raw))
    print('PREFLIGHT both hand action blocks coupled and collision has improving translation directions',flush=True)
    del raw,initial
    env=train.build_env(cfg,dataset_path=args.dataset,split='all',device='cpu',dataset=ds)
    old=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_coarse_r6_learning1024_20261001_r2/checkpoints/step_0256')
    try:rollout._verify_checkpoint_contract(provenance=json.loads((old/'agent.provenance.json').read_text()),checkpoint=old/'agent.pt',cfg=cfg,env=env)
    except ValueError as e:write(args.output/'old_checkpoint_rejection.json',{'rejected':True,'reason':str(e)})
    else:raise RuntimeError('old no-collision checkpoint accepted for new hypothesis')
    agent=train.build_agent(env,cfg,training=True,experiment_dir=args.output/'runs',experiment_name='generated_collision')
    initial_policy={k:v.detach().clone() for k,v in agent.policy.state_dict().items()}
    trace=[];step=env.step
    def record(action):
        result=step(action);info=result[-1]
        trace.append({'frame_idx':int(info['frame_idx']),**objective_parts(info['factor_ledger']),
                      'reward':float(result[1].item()),'terminated':bool(result[2].any()),
                      'raw_action':info['raw_action'].tolist(),'executed_delta':info['executed_delta'].tolist()})
        return result
    env.step=record
    diagnostics=[];old_update=agent.update
    def update(*,timestep,timesteps):
        memory=agent.memory
        old_log=memory.get_tensor_by_name('log_prob').detach().clone();a=memory.get_tensor_by_name('actions').detach().clone();o=memory.get_tensor_by_name('observations').detach().clone()
        before=optimizer_steps(agent);old_update(timestep=timestep,timesteps=timesteps)
        rng=torch.get_rng_state().clone()
        with torch.no_grad():
            _,out=agent.policy.act({'observations':agent._observation_preprocessor(o.reshape(-1,o.shape[-1])),'taken_actions':a.reshape(-1,118)},role='policy')
            log=out['log_prob']-old_log.reshape(-1,1);ratio=log.exp()
            diagnostics.append({'step':int(timestep)+1,'optimizer_steps':optimizer_steps(agent)-before,
                                'post_update_approx_KL':float(((ratio-1)-log).mean()),
                                'post_update_clip_fraction':float((torch.abs(ratio-1)>.2).float().mean()),
                                'policy_std_mean':float(agent.policy.log_std_parameter.exp().mean()),
                                'note':'post-update diagnostic, not internal KL early-stop estimate'})
        torch.set_rng_state(rng)
        write(args.output/'updates'/f'update_{int(timestep)+1:04d}.json',diagnostics[-1])
    agent.update=update
    checks=[];post=agent.post_interaction
    def callback(*a,**kw):
        post(*a,**kw);t=int(kw['timestep'] if 'timestep' in kw else a[0])+1
        if t in BOUNDARIES:checks.append(checkpoint(args,agent,env,cfg,cfg_hash,t))
    agent.post_interaction=callback
    learning=time.monotonic();train.build_trainer(env,agent,cfg).train()
    if len(checks)!=4 or not train.finite_policy_parameter_changed(agent,initial_policy):raise RuntimeError('incomplete training/finite update')
    train.assert_agent_checkpoint_finite(agent)
    if source_hashes()!=identity:raise RuntimeError('frozen runtime changed')
    b=ds[0]
    for path,expected in [(b.manifest_path,json.loads((args.output/'index.json').read_text())['episodes'][0]['manifest_sha256']),
                          (b.array_path,b.coarse['arrays']['sha256']),(b.source.manifest_path,b.coarse['source_bundle_sha256'])]:
        if file_sha256(path)!=expected:raise RuntimeError('input changed')
    best=min(checks,key=lambda r:r['objective'])
    summary={'scope':'same_video_generated_mesh_hypothesis_only','action_dim':118,'hands_frozen':False,'timesteps':1024,
             'zero_objective':zero['objective'],'zero_original_formal_objective':zero['original_formal_objective'],
             'zero_generated_collision_objective':zero['generated_collision_objective'],'zero_collision':zero['collision'],
             'checkpoints':checks,'best_trained_checkpoint':best,'improves_zero':best['objective']<zero['objective'],
             'learning_seconds':time.monotonic()-learning,'annotation_quality_verified':False,'source_authority_promoted':False}
    write(args.output/'training_steps.json',trace);write(args.output/'ppo_update_diagnostics.json',diagnostics);write(args.output/'summary.json',summary)
    render_comparison(args.output,zero,best,checks[-1],ds[0])
    env.close()
    write(args.output/'completed.json',{'status':'complete_full_action_1024_with_native_collision_audit_and_full_render',
                                     'frames':count,'code_input_unchanged':True,'improves_full_zero_objective':summary['improves_zero']})
    print('COMPLETE '+json.dumps({k:summary[k] for k in ['timesteps','improves_zero','learning_seconds','source_authority_promoted']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['experiment','evaluate'],default='experiment')
    p.add_argument('--config',type=Path,default=ROOT/'configs/ppo_generated_collision_coarse.yaml')
    p.add_argument('--coarse',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--provenance',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();seed()
    if args.mode=='experiment':experiment(args)
    else:evaluate(args)
