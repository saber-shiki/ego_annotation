"""Frozen reference frame120 perturbations, posthoc no-optimizer mechanism checks.

Training is already complete: these probes are NOT claimed as preflight evidence
or fed back into training, checkpoints, config, or exported prediction artifacts.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from data.coarse_training import CoarseTrainingDataset
from envs.three_rewards_coarse_env import ThreeRewardsCoarseEnv
from envs.dynamics import decode_candidate
from rewards.three_rewards import GROUPS
import train


def probe(run,output):
    cfg,_=train.load_runtime_config(ROOT/'configs/ppo_three_rewards_coarse.yaml')
    env=ThreeRewardsCoarseEnv(CoarseTrainingDataset(run/'index.json',split='all'),cfg['environment']);env.reset(seed=0)
    for _ in range(150):env.step(np.zeros(118,np.float32))
    state=env.state.copy();state.position=120
    original=env.evaluator.objective(state,150);rows=[]
    def record(c,name):
        ledger=env.evaluator.objective(c,150)
        before,bdetails=env.evaluator.frame_values(state,120);after,details=env.evaluator.frame_values(c,120)
        rows.append({'name':name,'frame_idx':120,'whole_timeline_delta':{g:ledger.terms[g]-original.terms[g] for g in GROUPS},
                     'frame_subcosts':details,'reference_frame_subcosts':bdetails})
    for side in ('left','right'):
        for axis in range(3):
            for sign in (-1,1):
                c=state.copy();d=np.zeros(118);d[c.episode.layout.block(f'hand.{side}.translation').start+axis]=sign*.002
                decode_candidate(c,d,env.cfg);record(c,f'{side}_translation_axis{axis}_{sign*2:+}mm')
    for block,amount in [('translation',.002),('rotation',.01)]:
        c=state.copy();d=np.zeros(118);d[c.episode.layout.block('object.object:0.'+block).start]=amount
        decode_candidate(c,d,env.cfg);record(c,'object_'+block)
    c=state.copy();d=np.zeros(118)
    for side in ('left','right'):d[c.episode.layout.block(f'hand.{side}.translation').start]=.002
    d[c.episode.layout.block('object.object:0.translation').start]=.002
    decode_candidate(c,d,env.cfg);record(c,'common_hand_object_world_X_plus2mm')
    common=rows[-1];np.testing.assert_allclose(common['frame_subcosts']['relative_interaction'],common['reference_frame_subcosts']['relative_interaction'],atol=1e-8)
    c=state.copy();c.contact_logits[120]-=20;c.term_cache.pop(120,None);record(c,'contact_logits_minus20_fixed_geometry')
    np.testing.assert_allclose(rows[-1]['whole_timeline_delta']['nonpenetration_contact'],0.,atol=1e-12)
    for side in ('left','right'):
        subset=[r for r in rows if r['name'].startswith(side+'_')]
        if not all(r['whole_timeline_delta']['reference_alignment']>0 for r in subset):raise RuntimeError('alignment disconnected')
    env.close()
    output.write_text(json.dumps({'scope':'posthoc_frame120_fixed_reference_probes_not_training_or_preflight','optimizer_steps':0,
        'frame_idx':120,'source_signed_authority_promoted':False,'rows':rows},indent=2)+'\n')
    print(json.dumps({'rows':[{k:r[k] for k in ('name','whole_timeline_delta')} for r in rows]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();probe(a.run,a.output)
