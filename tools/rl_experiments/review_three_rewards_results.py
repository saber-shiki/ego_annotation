"""Read-only interpretation for complete three-reward Milk debugging outputs."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.rl_experiments.export_world_comparison_rrd import export_rrd,verify_roundtrip,sha
from tools.rl_experiments.review_comparison_videos import inspect as review_videos


def inspect(run,output):
    output.mkdir(parents=True,exist_ok=False)
    zero=json.loads((run/'zero/evaluation.json').read_text());source=json.loads((run/'source_identity.json').read_text())
    runtime=Path(json.loads((run/'run_config.json').read_text())['runtime'])
    for path,h in source.items():
        if sha(runtime/path)!=h:raise RuntimeError('frozen runtime mutated: '+path)
    rows=[];summaries=[]
    for seed in (0,1):
        summary=json.loads((run/f'seed_{seed}/summary.json').read_text());summaries.append(summary)
        updates=json.loads((run/f'seed_{seed}/updates.json').read_text())
        if len(updates)!=4 or not all(x['before_update_value_clip_fraction']==0 and x['preprocessor_unchanged'] for x in updates):
            raise RuntimeError('PPO invariant missing or failed')
        for row in summary['checkpoints']:
            cp=Path(row['checkpoint']);prov=json.loads(cp.with_name('agent.provenance.json').read_text())
            if sha(cp)!=prov['checkpoint']['sha256']:raise RuntimeError('checkpoint changed')
            path=Path(row['evaluation']);d=json.loads(path.read_text());m=json.loads(Path(d['manifest']).read_text())
            if m['executed_frame_count']!=150 or not m['preprocessors_frozen_during_evaluation']:raise RuntimeError('evaluation incomplete')
            for a in m['episodes'][0]['artifacts']:
                p=Path(a['path']);p=p if p.is_absolute() else Path(d['manifest']).parent/p
                if sha(p)!=a['sha256']:raise RuntimeError('export changed')
            r={'seed':seed,'step':row['steps'],'J_three':row['objective'],'delta_J_three':row['objective']-zero['objective'],
               'J_reference':row['group_costs']['reference_alignment'],'J_physics':row['group_costs']['nonpenetration_contact'],
               'J_temporal':row['group_costs']['temporal_continuity'],'J_original':row['original_formal_objective'],
               'delta_J_original':row['original_formal_objective']-zero['original_formal_objective'],
               'max_penetration_mm':row['collision']['max_inside_depth_mm'],
               'inside_squared_sum_m2':row['collision']['inside_squared_hinge_sum_m2'],
               'inside_squared_percent_change':100*(row['collision']['inside_squared_hinge_sum_m2']/zero['collision']['inside_squared_hinge_sum_m2']-1),
               'crossing_edges':row['collision']['total_crossing_edges_over_all_hand_frames'],
               'crossing_edges_percent_change':100*(row['collision']['total_crossing_edges_over_all_hand_frames']/zero['collision']['total_crossing_edges_over_all_hand_frames']-1),
               'penetration_affected_frames':row['collision']['affected_frame_count'],
               'object_shift_max_mm':row['geometry']['object']['translation_max_mm'],
               'object_rotation_max_deg':row['geometry']['object']['rotation_max_deg']}
            for side in ('left','right'):
                h=row['geometry'][side]
                for key in ('vertex_correction_mean_mm','vertex_correction_max_mm','joint_pixel_p95','joint_pixel_max','correction_velocity_p95_mm_s','correction_acceleration_p95_mm_s2'):
                    r[side+'_'+key]=h[key]
                for key,value in row['absolute_motion'][side].items():r[side+'_'+key]=value
            rows.append(r)
    with (output/'all_checkpoints.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Retain all-frame sheets from BOTH fixed endpoints and best-trained views.
    for label in ('last','best'):review_videos(run/'comparisons'/label,output/'visual'/label)
    items=[{'key':'zero','name':'Zero R6','steps':0,'objective':zero['objective'],'path':zero['npz'],'sha256':sha(Path(zero['npz']))}]
    reports=[json.loads((run/'zero/collision_audit.json').read_text())]
    for s in summaries:
        row=s['checkpoints'][-1];p=Path(row['evaluation']);d=json.loads(p.read_text())
        items.append({'key':f'seed{s["seed"]}','name':f'seed{s["seed"]}@600','steps':600,'objective':row['objective'],
                      'path':d['npz'],'sha256':sha(Path(d['npz']))})
        reports.append(json.loads((p.parent/'collision_audit.json').read_text()))
    archives=[]
    for item in items:
        with np.load(item['path'],allow_pickle=False) as z:archives.append(dict(z))
    rrd=output/'zero_seed0_seed1_last_collision.rrd'
    receipt=export_rrd(items,archives,rrd,collision_reports=reports)
    verification=verify_roundtrip(rrd,items,archives)
    (output/'rrd_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    report={'scope':'new_minimal_same_video_600x2_debug_not_GT_generalization_or_single_reward_causality',
            'zero':zero,'rows':rows,'updates':[json.loads((run/f'seed_{s}/updates.json').read_text()) for s in (0,1)],
            'rrd':str(rrd),'rrd_bytes':receipt['bytes'],'frozen_source_verified':len(source),
            'below_zero_rows':[r for r in rows if r['delta_J_three']<0]}
    (output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'rows':rows,'rrd':str(rrd)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inspect(a.run,a.output)
