"""Read-only, paired reward-ablation analysis and full-timeline visual review assets."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
import yaml
from tools.rl_experiments.export_world_comparison_rrd import sha
from rewards.ablation import GROUPS,grouped,select_ledger
from rewards.evaluator import FactorLedger


def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().numpy();h.update(k.encode());h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    return h.hexdigest()


def scalar_metrics(row,render):
    g=row['geometry'];c=row['collision']
    values={'J_full':row['objective'],'J_train':row['training_objective'],'J_removed':row['removed_objective'],
            'J_original':row['original_formal_objective'],'J_collision':row['generated_collision_objective'],
            'collision_squared_sum_m2':c['inside_squared_hinge_sum_m2'],'max_inside_mm':c['max_inside_depth_mm'],
            'crossing_edges':c['total_crossing_edges_over_all_hand_frames'],'affected_frames':c['affected_frame_count'],
            'IoU_median':render['candidate_measurements'][row['seed']+1]['raw_SAM2_silhouette_IoU_median'],
            'IoU_mean':float(np.mean(render['per_frame_raw_IoU'][row['seed']+1])),
            'object_translation_max_mm':g['object']['translation_max_mm'],'object_rotation_max_deg':g['object']['rotation_max_deg'],
            'object_accel_p95_mm_s2':g['object']['correction_acceleration_p95_mm_s2'],
            'object_rot_rate_p95_deg_s':g['object']['correction_rotvec_rate_p95_deg_s']}
    for side in ['left','right']:
        for name in ['joint_pixel_p95','joint_pixel_max','joint_depth_shift_p95_mm','vertex_correction_mean_mm',
                     'correction_velocity_p95_mm_s','correction_acceleration_p95_mm_s2','nonpositive_joint_depth_count']:
            values[side+'_'+name]=g[side][name]
        values[side+'_contact_prob_abs_shift']=g['contact_prior_deviation_mean_abs'][['left','right'].index(side)]
    values.update({'cost_'+k:v for k,v in row['full_group_costs'].items()})
    for block,val in g['action_blocks'].items():
        values['action_norm95_'+block]=val['norm_p95']
    return values


def decoded(path):
    cap=cv2.VideoCapture(str(path));frames=[]
    fps=cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok,im=cap.read()
        if not ok:break
        if im is None:raise ValueError('empty decoded frame')
        frames.append(im)
    cap.release()
    if len(frames)!=150 or abs(fps-30)>1e-5:raise ValueError('full-video decode/fps mismatch '+str(path))
    return frames,fps


def labelled(im,text,width):
    im=cv2.resize(im,(width,round(im.shape[0]*width/im.shape[1])),interpolation=cv2.INTER_AREA)
    h=np.zeros((22,width,3),np.uint8);cv2.putText(h,text,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    return np.vstack([h,im])


def make_visual(run,out,variants):
    visual=out/'visual';visual.mkdir()
    receipts=[];representatives={}
    # The existing MP4 panels are Zero / ablation seed0 / ablation seed1,
    # not zero/best/last despite the historic generic filename.
    for group in variants:
        folder=visual/group;folder.mkdir();pair={}
        for kind in ['overlay','world']:
            source=run/'comparisons'/group/'renders'/f'zero_best_last_{kind}.mp4'
            frames,fps=decoded(source)
            # Two all-frame sheets (0-74 and75-149). They expose continuity,
            # dropout and gross misalignment; sub-mm collision is audited in 3D.
            for start in [0,75]:
                tiles=[cv2.resize(f,(384,round(f.shape[0]*.5)),interpolation=cv2.INTER_AREA) for f in frames[start:start+75]]
                page=np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,75,5)])
                cv2.imwrite(str(folder/f'{kind}_{start:03d}_{start+74:03d}.jpg'),page)
            representatives[group,kind]={i:frames[i] for i in [0,30,60,86,120,149]}
            for i in [0,30,60,86,120,149]:cv2.imwrite(str(folder/f'{kind}_{i:03d}.jpg'),frames[i])
            receipts.append({'group':group,'kind':kind,'source':str(source),'sha256':sha(source),'decoded_frames':len(frames),'fps':fps,
                             'panel_identity':['zero R6',group+' seed0@256',group+' seed1@256']})
            pair[kind]=frames
        # Full-timeline combined coarse overview, for bookkeeping/navigation.
        tiles=[]
        for i in range(150):
            im=np.vstack([pair['overlay'][i],pair['world'][i]])
            tiles.append(cv2.resize(im,(192,round(im.shape[0]*.25)),interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(folder/'all150_overlay_world_overview.jpg'),np.vstack([np.hstack(tiles[i:i+10]) for i in range(0,150,10)]))
    # Side by side SAME SEED comparisons: full vs drop, reusing only the already
    # rendered native/MANO frames. This avoids confusing seed differences with
    # ablation effects and does not rerasterize or alter source annotations.
    for group in variants:
        if group=='full':continue
        folder=visual/group
        for seed in [0,1]:
            tiles=[]
            for fid in [0,30,60,86,120,149]:
                parts=[]
                for kind in ['overlay','world']:
                    full=representatives['full',kind][fid][:,(seed+1)*256:(seed+2)*256]
                    drop=representatives[group,kind][fid][:,(seed+1)*256:(seed+2)*256]
                    parts.append(np.hstack([labelled(full,f'full s{seed} f{fid}',320),labelled(drop,f'drop {group} s{seed} f{fid}',320)]))
                tiles.append(np.vstack(parts))
            cv2.imwrite(str(folder/f'paired_seed{seed}_six_frames.jpg'),np.hstack([np.vstack(tiles[:3]),np.vstack(tiles[3:])]))
    (out/'video_verification.json').write_text(json.dumps({'decoded_comparisons':receipts,'videos':len(receipts),
        'total_decoded_frames':sum(r['decoded_frames'] for r in receipts),
        'visual_scope':'continuity, full-mesh/MANO presence and qualitative alignment; does not certify tiny geometry or physical contact'},indent=2)+'\n')
    return receipts


def analyze(run,out,snapshot):
    out.mkdir(parents=True,exist_ok=False)
    matrix=json.loads((run/'matrix_results.json').read_text());contract=json.loads((run/'matrix_contract.json').read_text())
    variants=contract['variants'];runs=matrix['runs'];by={(r['drop_group'],r['seed']):r for r in runs}
    if len(runs)!=26 or set(variants)!={'full',*GROUPS}:raise ValueError('incomplete matrix coverage')
    source=json.loads((run/'source_identity.json').read_text())
    for rel,h in source.items():
        if sha(snapshot/rel)!=h:raise ValueError('frozen source identity mismatch '+rel)
    identity_files={str(snapshot/rel):h for rel,h in source.items()}
    index=json.loads((run/'index.json').read_text());coarse=Path(index['episodes'][0]['manifest']);expected=index['episodes'][0]['manifest_sha256']
    if sha(coarse)!=expected:raise ValueError('coarse initializer manifest changed')
    d=json.loads(coarse.read_text())
    for raw,h in [(d['arrays']['path'],d['arrays']['sha256']),(d['source_observations'],d['source_bundle_sha256'])]:
        p=Path(raw);p=p if p.is_absolute() else coarse.parent/p
        if sha(p)!=h:raise ValueError('coarse/source input changed')
        identity_files[str(p)]=h
    zero=matrix['zero'];zero_path=Path(zero['npz'])
    zdoc=json.loads(zero_path.with_suffix('.json').read_text());zbound=zdoc['policy_rollout_array_layer']
    if sha(zero_path)!=zbound['sha256']:raise ValueError('zero archive hash mismatch')
    with np.load(zero_path,allow_pickle=False) as z:base=dict(z)
    zero_eval=json.loads((run/'zero/evaluation.json').read_text())
    full_training={s:json.loads((Path(by['full',s]['checkpoint']).parent/'training_trace.json').read_text()) for s in [0,1]}
    baseline_states={s:torch.load(by['full',s]['checkpoint'],map_location='cpu',weights_only=True) for s in [0,1]}
    full_archives={}
    for s in [0,1]:
        with np.load(by['full',s]['npz'],allow_pickle=False) as z:full_archives[s]=dict(z)
    reviewed=[];files={str(zero_path):sha(zero_path)};accounting=[];negative_controls=[]
    for row in runs:
        group,s=row['drop_group'],row['seed'];folder=Path(row['checkpoint']).parent
        cfg=json.loads(json.dumps(yaml.safe_load((run/'configs'/f'{group}_seed{s}.yaml').read_text())))
        expected_cfg=json.loads(json.dumps(contract['base_config']));expected_cfg['environment']['reward_ablation']['drop_group']=group
        if cfg!=expected_cfg:raise ValueError('extra config intervention '+group)
        cp=Path(row['checkpoint']);provenance=json.loads(cp.with_name('agent.provenance.json').read_text())
        if sha(cp)!=provenance['checkpoint']['sha256']:raise ValueError('checkpoint hash mismatch')
        if sha(run/'configs'/f'{group}_seed{s}.yaml')!=provenance['training_config_sha256']:raise ValueError('config checkpoint hash mismatch')
        ev=json.loads(Path(row['evaluation']).read_text());manifest_path=Path(ev['manifest']);manifest=json.loads(manifest_path.read_text())
        if manifest['executed_frame_count']!=150 or not manifest['preprocessors_frozen_during_evaluation'] or manifest['stochastic_evaluation']:raise ValueError('invalid complete deterministic eval')
        for a in manifest['episodes'][0]['artifacts']:
            p=Path(a['path']);p=p if p.is_absolute() else manifest_path.parent/p
            if sha(p)!=a['sha256']:raise ValueError('rollout artifact hash changed')
            files[str(p)]=a['sha256']
        files[str(cp)]=sha(cp)
        full=ev['full_ledger'];selected=ev['training_ledger']
        recomputed=grouped(full['terms'])
        np.testing.assert_allclose(list(recomputed.values()),list(row['full_group_costs'].values()),atol=1e-8)
        fl=FactorLedger(end_position=150,terms=full['terms'],statistics=full['statistics'],activation=full['activation'],reward_scale=full['reward_scale'])
        chosen,removed=select_ledger(fl,group)
        assert chosen.terms==selected['terms']
        np.testing.assert_allclose([fl.objective,chosen.objective,sum(removed.values())],
                                  [row['objective'],row['training_objective'],row['removed_objective']],atol=1e-8)
        trace=ev['steps_trace'];np.testing.assert_array_equal([x['frame_idx'] for x in trace],base['frame_idx'])
        np.testing.assert_allclose(sum(x['reward'] for x in trace),-full['reward_scale']*chosen.objective,rtol=1e-5,atol=1e-6)
        updates=json.loads((folder/'updates.json').read_text());assert updates==[{'step':128,'optimizer_steps':64},{'step':256,'optimizer_steps':64}]
        training=json.loads((folder/'training_trace.json').read_text());reference=full_training[s]
        if len(training)!=256:raise ValueError('incomplete training')
        actions=np.asarray([x['executed_delta'] for x in training]);baseline_actions=np.asarray([x['executed_delta'] for x in reference])
        if not np.array_equal(actions[:128],baseline_actions[:128]):raise ValueError('first rollout not paired')
        reward=np.array([x['reward'] for x in training]);reference_reward=np.array([x['reward'] for x in reference]);dropped=np.array([x['removed_objective'] for x in training])
        cpstate=torch.load(cp,map_location='cpu',weights_only=True)
        same={key:state_hash(cpstate[key])==state_hash(baseline_states[s][key]) for key in ['policy','value','observation_preprocessor','value_preprocessor']}
        with np.load(row['npz'],allow_pickle=False) as z:a=dict(z)
        for k in ['frame_idx','time_s','T_world_camera','intrinsics_render','object_vertices_canonical','object_faces','left_faces','right_faces','left_valid','right_valid']:
            np.testing.assert_array_equal(a[k],base[k])
        diff={}
        for side in ['left','right']:
            change=np.linalg.norm(a[side+'_vertices_world']-full_archives[s][side+'_vertices_world'],axis=-1)
            diff[side]={'max_mm':float(change.max()*1000),'mean_mm':float(change.mean()*1000),
                        'vertices_exact':bool(np.array_equal(a[side+'_vertices_world'],full_archives[s][side+'_vertices_world']))}
        diff['object_translation_max_mm']=float(np.linalg.norm(a['object_translation_world']-full_archives[s]['object_translation_world'],axis=-1).max()*1000)
        diff['object_rotation_exact']=bool(np.array_equal(a['object_rotation'],full_archives[s]['object_rotation']))
        render=json.loads((run/'comparisons'/group/'renders/render_measurements.json').read_text())
        vals=scalar_metrics(row,render)
        early={'first128_actions_identical':True,'first128_reward_different_count':int((reward[:128]!=reference_reward[:128]).sum()),
               'first128_max_reward_abs_change':float(np.max(np.abs(reward[:128]-reference_reward[:128]))),
               'first128_removed_prefix_max':float(dropped[:128].max()),
               'first128_full_prefix_max':float(max(x['full_objective'] for x in reference[:128])),
               'second128_actions_identical':bool(np.array_equal(actions[128:],baseline_actions[128:])),
               'second128_action_difference_rms':float(np.sqrt(np.mean((actions[128:]-baseline_actions[128:])**2))),
               'final_parameter_state_equal_to_full':same,'geometry_difference_to_paired_full':diff}
        reviewed.append({'group':group,'seed':s,'metrics':vals,'full_group_costs':row['full_group_costs'],'paired_influence':early,
                         'npz':row['npz'],'evaluation':row['evaluation']})
        if group in ['contact','hand_bounds','hand_joint_depth']:negative_controls.append({'group':group,'seed':s,**early})
        accounting.append({'group':group,'seed':s,'config_single_group_only':True,'artifact_hashes_verified':True,
                           'full_selected_ledger_verified':True,'return_telescope_verified':True,'initial_parameters_match':row['initial_parameter_hashes']==by['full',s]['initial_parameter_hashes']})
    by_review={(r['group'],r['seed']):r for r in reviewed}
    paired=[]
    for group in variants:
        if group=='full':continue
        delta={}
        for metric in reviewed[0]['metrics']:
            if metric in ['J_train','J_removed']:continue  # not comparable full objectives
            values=[by_review[group,s]['metrics'][metric]-by_review['full',s]['metrics'][metric] for s in [0,1]]
            delta[metric]={'seed0':values[0],'seed1':values[1],'mean':float(np.mean(values)),
                           'both_negative':bool(all(x<0 for x in values)), 'both_positive':bool(all(x>0 for x in values)),
                           'both_exact_zero':bool(all(x==0 for x in values))}
        paired.append({'drop_group':group,'delta_vs_same_seed_full':delta})
    result={'scope':'single_video256step_two_seed_screen_no_significance_or_generalization',
            'zero_objective':zero['objective'],'zero_original_J':zero['original_formal_objective'],
            'zero_collision_summary':{k:zero['collision'][k] for k in ['affected_frame_count','max_inside_depth_mm','inside_squared_hinge_sum_m2','total_crossing_edges_over_all_hand_frames']},
            'full_seed_gap_J':abs(by['full',0]['objective']-by['full',1]['objective']),
            'runs':reviewed,'paired_deltas':paired,'negative_controls':negative_controls,
            'below_zero_candidates':[{'group':r['group'],'seed':r['seed'],'J_full':r['metrics']['J_full'],
                                      'delta_zero':r['metrics']['J_full']-zero['objective']} for r in reviewed if r['metrics']['J_full']<zero['objective']]}
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['group','seed',*reviewed[0]['metrics']]
    with (out/'all_runs.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in reviewed:w.writerow({'group':r['group'],'seed':r['seed'],**r['metrics']})
    with (out/'paired_deltas.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['drop_group','metric','seed0_delta','seed1_delta','mean_delta','both_negative','both_positive','exact_zero'])
        for p in paired:
            for key,val in p['delta_vs_same_seed_full'].items():w.writerow([p['drop_group'],key,val['seed0'],val['seed1'],val['mean'],val['both_negative'],val['both_positive'],val['both_exact_zero']])
    evidence={'runs_verified':26,'source_code_hashes_verified':len(source),'source_and_zero_hashes_verified':True,
              'fixed_camera_native_faces_mano_topology_all_verified':True,'native_vertices':len(base['object_vertices_canonical']),
              'native_faces':len(base['object_faces']),'full_and_selected_accounting':accounting,'artifacts':files,
              'source_files':identity_files,'no_training_or_prediction_modifications':True}
    (out/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    videos=make_visual(run,out,variants)
    print(json.dumps({'output':str(out),'verified_runs':26,'decoded_videos':len(videos),'below_zero_candidates':result['below_zero_candidates'],'full_seed_gap_J':result['full_seed_gap_J']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.output,a.snapshot)
