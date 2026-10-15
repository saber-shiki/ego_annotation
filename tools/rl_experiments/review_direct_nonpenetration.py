"""Read-only direct nonpenetration review of full timeline and real geometry."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from data.coarse_training import CoarseTrainingDataset
from data.bundle import file_sha256
from tools.rl_experiments.review_comparison_videos import inspect as videos
from tools.rl_experiments.export_world_comparison_rrd import export_rrd,verify_roundtrip
from tools.rl_experiments.reward_ablation_learning import metrics
from tools.rl_experiments.three_rewards_learning import actual_motion
from tools.rl_experiments.audit_three_rewards_motion_contact import stats,full_native_motion


def inspect(run,out):
    out.mkdir(parents=True,exist_ok=False)
    completed=json.loads((run/'solve_completed.json').read_text());trace=json.loads((run/'solve_trace.json').read_text())
    ds=CoarseTrainingDataset(run/'index.json',split='all');source=ds[0].load()
    paths=[run/label/'candidate.npz' for label in ('zero','solved')]
    archives=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:archives.append(dict(z))
    base,a=archives
    for key in ['frame_idx','time_s','T_world_camera','intrinsics_render','object_vertices_canonical','object_faces','contact_logits','left_betas','right_betas']:
        np.testing.assert_array_equal(a[key],base[key])
    probability=(1/(1+np.exp(-a['contact_logits']))).mean(-1)
    rows=[{'executed_delta':r['physical_delta_118'],'contact_mean_probability':probability[i]} for i,r in enumerate(trace)]
    movement=metrics(a,base,rows,probability);motion={label:actual_motion(x) for label,x in zip(('zero','solved'),archives)}
    object_motion={label:full_native_motion(x,base) for label,x in zip(('zero','solved'),archives)}
    gaps={}
    for label,x in zip(('zero','solved'),archives):
        allgap={s:[] for s in ('left','right')};change={s:[] for s in allgap}
        for i,fid in enumerate(x['frame_idx']):
            for (frame,side,entity),row in source['contact'].items():
                if frame!=int(fid):continue
                ids=np.asarray(row['vertex_ids'],int);anchor=np.asarray(row['targets_object'],float)
                if not len(ids):continue
                now=np.linalg.norm(x[side+'_vertices_world'][i,ids]-(anchor@x['object_rotation'][i].T+x['object_translation_world'][i]),axis=-1)
                zero=np.linalg.norm(base[side+'_vertices_world'][i,ids]-(anchor@base['object_rotation'][i].T+base['object_translation_world'][i]),axis=-1)
                allgap[side].extend(now*1000);change[side].extend((now-zero)*1000)
        gaps[label]={'qualified_patch_distance_mm':{s:stats(v) for s,v in allgap.items()},'qualified_patch_increase_mm':{s:stats(v) for s,v in change.items()}}
    reports=[json.loads((run/label/'complete_triangle_audit.json').read_text()) for label in ('zero','solved')]
    frame_rows=[]
    for i,entry in enumerate(trace):
        r={'frame_idx':entry['frame_idx'],'iterations':entry['iterations'],'solver_feasible':entry['feasible']}
        for label,report in zip(('zero','solved'),reports):
            row=report['frames'][i];r[label+'_feasible']=row['feasible']
            for side,h in row['hands'].items():
                for key in ['max_inside_depth_mm','triangle_intersection_pair_count','inside_squared_sum_m2','nearest_surface_gap_mm']:
                    r[f'{label}_{side}_{key}']=h[key]
        frame_rows.append(r)
    with (out/'all_frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(frame_rows[0]));w.writeheader();w.writerows(frame_rows)
    per_hand={}
    for label,report in zip(('zero','solved'),reports):
        per_hand[label]={s:{'collision_free_frames':sum(f['hands'][s]['triangle_intersection_pair_count']==0 and f['hands'][s]['max_inside_depth_mm']<=.01 for f in report['frames']),
            'max_depth_mm':max(f['hands'][s]['max_inside_depth_mm'] for f in report['frames']),
            'triangle_pairs_total':sum(f['hands'][s]['triangle_intersection_pair_count'] for f in report['frames']),
            'inside_squared_total_m2':sum(f['hands'][s]['inside_squared_sum_m2'] for f in report['frames'])} for s in ('left','right')}
    videos(run/'comparison',out/'visual')
    items=[{'key':label,'name':'Zero R6' if label=='zero' else 'Direct solve (not certified)','steps':0,'objective':None,'path':str(path),'sha256':file_sha256(path)} for label,path in zip(('zero','solved'),paths)]
    rrd=out/'zero_direct_complete_collision.rrd';receipt=export_rrd(items,archives,rrd,collision_reports=reports)
    verification=verify_roundtrip(rrd,items,archives)
    (out/'rrd_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    histories=[h for r in trace for h in r['history']]
    result={'scope':'single_video_bounded_direct_geometry_attempt_no_GT_or_PPO','solve':completed,'movement':movement,
            'actual_motion':motion,'object_full_native_motion':object_motion,'qualified_contact_gaps':gaps,'per_hand_physics':per_hand,
            'audits':{label:{k:v for k,v in report.items() if k!='frames'} for label,report in zip(('zero','solved'),reports)},
            'linear_minimax_slack_mm':stats([h['slack_mm'] for h in histories if h['slack_mm'] is not None]),
            'QP_success_count':sum(h['qp_success'] for h in histories),'QP_attempts':len(histories),
            'nonlinear_rejected_step_count':sum(not h['accepted'] for h in histories),
            'rrd_path':str(rrd),'rrd_sha256':receipt['rrd_sha256']}
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['audits','solve','movement','object_full_native_motion']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inspect(a.run,a.output)
